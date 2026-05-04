// MNIST MLP. Forward + backward + SGD update — all in one cg::ComputeGraph.
// Layers (nn::Dense) own their parameters; no manual W/b declarations.

#include "cg.h"
#include "nn.h"
#include "autograd.h"
#include "cpu.h"
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
struct MLP {
    nn::Dense fc1, fc2;
    MLP(std::mt19937& rng) : fc1(784, 128, rng), fc2(128, 10, rng) {}

    // Returns logits (pre-softmax)
    Node* forward(ComputeGraph& g, Node* x, int batch) {
        auto* z1 = fc1(g, x, batch);
        auto* a1 = nn::relu(g, z1);
        return     fc2(g, a1, batch);
    }

    std::vector<Node*> params() const {
        auto p = fc1.params();
        for (auto* n : fc2.params()) p.push_back(n);
        return p;
    }

    template <typename BB>
    void apply_sgd(ComputeGraph& g, BB& bb, float lr) {
        fc1.apply_sgd(g, bb, lr);
        fc2.apply_sgd(g, bb, lr);
    }

    template <typename Exec>
    void refresh(Exec& exec) { fc1.refresh(exec); fc2.refresh(exec); }
};

// ---- Train one batch ----------------------------------------------------
template <typename Executor>
static std::pair<float, int>
train_step(MLP& mlp, const Tensor& X, const std::vector<int>& Y, float lr,
           Executor& exec, bool print_graph)
{
    int batch = X.shape[0];
    int n_cls = 10;

    ComputeGraph g;
    auto* x  = g.emplace<InputNode>("X",  X);
    auto* oh = g.emplace<InputNode>("OH", mnist::one_hot(Y, n_cls));

    auto* logits = mlp.forward(g, x, batch);
    auto* y      = nn::softmax(g, logits);

    auto* dz = nn::softmax_ce_backward(g, y, oh, batch);
    autograd::BackwardBuilder bb(g);
    bb.seed(logits, dz);
    bb.build(mlp.params());

    mlp.apply_sgd(g, bb, lr);

    if (print_graph) {
        std::cout << "--- TRAIN GRAPH ---\n";
        PrintVisitor p; g.accept(p); std::cout << "\n";
    }

    exec.clear();
    g.accept(exec);

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

    mlp.refresh(exec);
    return {loss, correct};
}

template <typename Executor>
static float evaluate(MLP& mlp, const Tensor& X, const std::vector<int>& Y, Executor& exec) {
    int batch = X.shape[0];
    int n_cls = 10;

    ComputeGraph g;
    auto* x      = g.emplace<InputNode>("X", X);
    auto* logits = mlp.forward(g, x, batch);
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

template <typename Executor>
static void train(const char* label, const mnist::Dataset& tr, const mnist::Dataset& te,
                  Executor& exec)
{
    constexpr int   batch_size = 64, n_epochs = 5;
    constexpr float lr         = 0.1f;
    std::cout << "\n===== Training on " << label << " =====\n";

    std::mt19937 rng(42);
    MLP mlp(rng);

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
            auto [l, c] = train_step(mlp, X, Y, lr, exec, !printed);
            printed = true;
            loss += l; correct += c; ++batches;
        }
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        float test_acc = evaluate(mlp, te.images, te.labels, exec);

        std::cout << "[" << label << "] ep " << (epoch + 1) << "/" << n_epochs
                  << "  loss "      << (loss / batches)
                  << "  train_acc " << (100.0f * correct / (batches * batch_size)) << "%"
                  << "  test_acc "  << (100.0f * test_acc) << "%"
                  << "  ("          << secs << "s)\n";
    }
}

int main() {
    std::cout << "Loading MNIST...\n";
    auto tr = mnist::load_train();
    auto te = mnist::load_test();
    std::cout << "  train: " << tr.labels.size() << "  test: " << te.labels.size() << "\n";

    { cg::cpu::Executor   e; train("CPU",   tr, te, e); }
    { cg::metal::Executor e; train("Metal", tr, te, e); }
    return 0;
}
