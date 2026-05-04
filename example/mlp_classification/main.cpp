// MNIST MLP. Forward + backward + SGD update — all in ONE cg::ComputeGraph.
// Run on CPU and Metal back-to-back.

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

// MNIST loader and one_hot live in common/mnist.h

// ---- Model parameters ----------------------------------------------------
struct MLP { cg::Tensor W1, b1, W2, b2; };

static MLP init_mlp(int in_dim, int hidden, int out_dim, std::mt19937& rng) {
    auto he = [&](std::vector<int> shape, int fan_in) {
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / fan_in));
        cg::Tensor t(shape);
        for (auto& v : t.data) v = d(rng);
        return t;
    };
    return { he({in_dim, hidden},  in_dim),  cg::Tensor({1, hidden}),
             he({hidden, out_dim}, hidden),  cg::Tensor({1, out_dim}) };
}

// ---- One training step: forward + backward + SGD, single graph ----------
template <typename Executor>
static std::pair<float, int>
train_step(MLP& mlp, const cg::Tensor& X, const std::vector<int>& Y, float lr,
           Executor& exec, bool print_graph)
{
    using namespace cg;
    int batch = X.shape[0];
    int n_cls = mlp.W2.shape[1];

    ComputeGraph g;
    auto* x  = g.emplace<InputNode>("X",  X);
    auto* W1 = g.emplace<InputNode>("W1", mlp.W1);
    auto* b1 = g.emplace<InputNode>("b1", mlp.b1);
    auto* W2 = g.emplace<InputNode>("W2", mlp.W2);
    auto* b2 = g.emplace<InputNode>("b2", mlp.b2);
    auto* oh = g.emplace<InputNode>("OH", mnist::one_hot(Y, n_cls));

    // --- forward ---
    auto* z1 = nn::linear(g, x, W1, b1, batch);
    auto* a1 = nn::relu(g, z1);
    auto* z2 = nn::linear(g, a1, W2, b2, batch);
    auto* y  = nn::softmax(g, z2);

    // --- autograd: seed the gradient at z2 (logits) with the softmax+CE
    //     shortcut, then let BackwardBuilder emit the rest.
    auto* dz2 = nn::softmax_ce_backward(g, y, oh, batch);
    autograd::BackwardBuilder bb(g);
    bb.seed(z2, dz2);
    bb.build({W1, b1, W2, b2});  // only emit grads on paths leading to params

    // --- SGD update — gradients pulled from autograd ---
    auto* W1_new = nn::sgd_step(g, W1, bb.grad(W1), lr);
    auto* W2_new = nn::sgd_step(g, W2, bb.grad(W2), lr);
    auto* b1_new = nn::sgd_step(g, b1, bb.grad(b1), lr);
    auto* b2_new = nn::sgd_step(g, b2, bb.grad(b2), lr);

    if (print_graph) {
        std::cout << "--- TRAIN GRAPH ---\n";
        PrintVisitor p; g.accept(p); std::cout << "\n";
    }

    exec.clear();
    g.accept(exec);

    // Loss + accuracy from softmax output
    const cg::Tensor& y_v = exec.result(y);
    float loss = 0.0f; int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = y_v.data.data() + i * n_cls;
        loss -= std::log(std::max(row[Y[i]], 1e-9f));
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    loss /= batch;

    mlp.W1 = exec.result(W1_new);
    mlp.W2 = exec.result(W2_new);
    mlp.b1 = exec.result(b1_new);
    mlp.b2 = exec.result(b2_new);
    return {loss, correct};
}

// ---- Eval (forward only) -------------------------------------------------
template <typename Executor>
static float evaluate(const MLP& mlp, const cg::Tensor& X,
                      const std::vector<int>& Y, Executor& exec)
{
    using namespace cg;
    int batch = X.shape[0];
    int n_cls = mlp.W2.shape[1];

    ComputeGraph g;
    auto* x  = g.emplace<InputNode>("X",  X);
    auto* W1 = g.emplace<InputNode>("W1", mlp.W1);
    auto* b1 = g.emplace<InputNode>("b1", mlp.b1);
    auto* W2 = g.emplace<InputNode>("W2", mlp.W2);
    auto* b2 = g.emplace<InputNode>("b2", mlp.b2);

    auto* y = nn::softmax(g,
                nn::linear(g,
                  nn::relu(g, nn::linear(g, x, W1, b1, batch)),
                  W2, b2, batch));

    exec.clear(); g.accept(exec);
    const auto& yv = exec.result(y);

    int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = yv.data.data() + i * n_cls;
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    return (float)correct / batch;
}

// ---- Train loop ----------------------------------------------------------
template <typename Executor>
static void train(const char* label, const mnist::Dataset& tr, const mnist::Dataset& te,
                  Executor& exec)
{
    constexpr int batch_size = 64, n_epochs = 5;
    constexpr float lr = 0.1f;
    std::cout << "\n===== Training on " << label << " =====\n";

    std::mt19937 rng(42);
    MLP mlp = init_mlp(784, 128, 10, rng);

    int n = (int)tr.labels.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    bool printed = false;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng);
        float loss = 0.0f; int correct = 0, batches = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int b = 0; b + batch_size <= n; b += batch_size) {
            cg::Tensor X({batch_size, 784});
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
