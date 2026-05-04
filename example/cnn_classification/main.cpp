// MNIST CNN. Conv2D + Dense layers own their own parameters; the training
// loop is a single cg::ComputeGraph executed on Metal.

#include "cg.h"
#include "nn.h"
#include "autograd.h"
#include "metal_executor.h"
#include "printer.h"
#include "mnist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace cg;

// ---- Model ---------------------------------------------------------------
struct CNN {
    nn::Conv2D conv1;   // 1 -> 8, 3x3, stride=2, pad=1   ->  [N, 8, 14, 14]
    nn::Dense  fc;      // 1568 -> 10

    CNN(std::mt19937& rng)
        : conv1(/*in*/1, /*out*/8, /*kH*/3, /*kW*/3, /*stride*/2, /*pad*/1, rng)
        , fc(1568, 10, rng) {}

    // Returns logits (pre-softmax)
    Node* forward(ComputeGraph& g, Node* x_flat, int batch) {
        auto* x4d  = g.emplace<ReshapeNode>(x_flat,
                       std::vector<int>{batch, 784},
                       std::vector<int>{batch, 1, 28, 28});
        auto* c    = conv1(g, x4d, std::vector<int>{batch, 1, 28, 28});
        auto* a    = nn::relu(g, c);
        auto* flat = g.emplace<ReshapeNode>(a,
                       std::vector<int>{batch, 8, 14, 14},
                       std::vector<int>{batch, 1568});
        return       fc(g, flat, batch);
    }

    std::vector<Node*> params() const {
        auto p = conv1.params();
        for (auto* n : fc.params()) p.push_back(n);
        return p;
    }

    template <typename BB>
    void apply_sgd(ComputeGraph& g, BB& bb, float lr) {
        conv1.apply_sgd(g, bb, lr);
        fc.apply_sgd(g, bb, lr);
    }

    template <typename Exec>
    void refresh(Exec& exec) { conv1.refresh(exec); fc.refresh(exec); }
};

// ---- Training step ------------------------------------------------------
static std::pair<float, int>
train_step(CNN& m, const Tensor& X, const std::vector<int>& Y, float lr,
           metal::Executor& exec, bool print_graph)
{
    int batch = X.shape[0];
    int n_cls = 10;

    ComputeGraph g;
    auto* x      = g.emplace<InputNode>("X",  X);
    auto* oh     = g.emplace<InputNode>("OH", mnist::one_hot(Y, n_cls));

    auto* logits = m.forward(g, x, batch);
    auto* y      = nn::softmax(g, logits);

    auto* dz = nn::softmax_ce_backward(g, y, oh, batch);
    autograd::BackwardBuilder bb(g);
    bb.seed(logits, dz);
    bb.build(m.params());

    m.apply_sgd(g, bb, lr);

    if (print_graph) {
        std::cout << "--- TRAIN GRAPH ---\n";
        PrintVisitor p; g.accept(p); std::cout << "\n";
    }

    exec.clear(); g.accept(exec);

    const Tensor& y_v = exec.result(y);
    float loss = 0.0f; int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = y_v.data.data() + i * n_cls;
        loss -= std::log(std::max(row[Y[i]], 1e-9f));
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    loss /= batch;

    m.refresh(exec);
    return {loss, correct};
}

static float evaluate(CNN& m, const Tensor& X, const std::vector<int>& Y, metal::Executor& exec) {
    int batch = X.shape[0];
    int n_cls = 10;

    ComputeGraph g;
    auto* x      = g.emplace<InputNode>("X", X);
    auto* logits = m.forward(g, x, batch);
    auto* y      = nn::softmax(g, logits);

    exec.clear(); g.accept(exec);
    const Tensor& yv = exec.result(y);

    int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = yv.data.data() + i * n_cls;
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    return (float)correct / batch;
}

// ---- Main ---------------------------------------------------------------
int main() {
    std::cout << "Loading MNIST...\n";
    auto tr = mnist::load_train();
    auto te = mnist::load_test();
    std::cout << "  train: " << tr.labels.size() << "  test: " << te.labels.size() << "\n";

    constexpr int   batch_size = 64;
    constexpr int   n_epochs   = 3;
    constexpr float lr         = 0.1f;

    std::mt19937 rng(42);
    CNN model(rng);
    metal::Executor exec;
    std::cout << "[Metal] device: " << exec.device_name() << "\n\n";

    int n = (int)tr.labels.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    bool printed = false;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng);
        float loss = 0.0f; int correct = 0, batches = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int b = 0; b + batch_size <= n; b += batch_size) {
            Tensor X({batch_size, 784});
            std::vector<int> Y(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                int s = idx[b + i];
                std::memcpy(X.data.data() + i * 784,
                            tr.images.data.data() + s * 784, 784 * sizeof(float));
                Y[i] = tr.labels[s];
            }
            auto [l, c] = train_step(model, X, Y, lr, exec, !printed);
            printed = true;
            loss += l; correct += c; ++batches;
        }
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        float test_acc = evaluate(model, te.images, te.labels, exec);

        std::cout << "epoch " << (epoch + 1) << "/" << n_epochs
                  << "  loss "      << (loss / batches)
                  << "  train_acc " << (100.0f * correct / (batches * batch_size)) << "%"
                  << "  test_acc "  << (100.0f * test_acc) << "%"
                  << "  ("          << secs << "s)\n";
    }
    return 0;
}
