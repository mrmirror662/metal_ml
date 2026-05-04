// CNN classifier on MNIST. Metal-only. Forward + autograd backward + SGD
// update built as a single cg::ComputeGraph.
//
// Architecture:
//   X [N, 1, 28, 28]
//     -> Conv2D(1->8, 3x3, stride=2, pad=1)   -> [N, 8, 14, 14]
//     -> ReLU
//     -> Reshape flatten                      -> [N, 1568]
//     -> Linear(1568 -> 10)
//     -> Softmax

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

// --- Model parameters ----------------------------------------------------
struct CNN {
    Tensor K1;   // [8, 1, 3, 3]   conv kernel
    Tensor W;    // [1568, 10]     linear weight
    Tensor b;    // [1, 10]        bias
};

static CNN init_cnn(std::mt19937& rng) {
    auto he = [&](std::vector<int> shape, int fan_in) {
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / fan_in));
        Tensor t(shape);
        for (auto& v : t.data) v = d(rng);
        return t;
    };
    return {
        he({8, 1, 3, 3}, 1 * 3 * 3),    // fan_in for conv = C_in * kH * kW
        he({1568, 10},   1568),
        Tensor({1, 10}),
    };
}

// --- Train one batch ------------------------------------------------------
static std::pair<float, int>
train_step(CNN& m, const Tensor& X, const std::vector<int>& Y, float lr,
           metal::Executor& exec, bool print_graph)
{
    int batch  = X.shape[0];
    int n_cls  = m.W.shape[1];

    ComputeGraph g;
    auto* x_flat = g.emplace<InputNode>("X",  X);                        // [N, 784]
    auto* K1     = g.emplace<InputNode>("K1", m.K1);
    auto* W      = g.emplace<InputNode>("W",  m.W);
    auto* b      = g.emplace<InputNode>("b",  m.b);
    auto* oh     = g.emplace<InputNode>("OH", mnist::one_hot(Y, n_cls));

    // --- Forward ---
    auto* x      = g.emplace<ReshapeNode>(x_flat,
                       std::vector<int>{batch, 784},
                       std::vector<int>{batch, 1, 28, 28});
    auto* c1     = nn::conv2d(g, x, K1,
                       std::vector<int>{batch, 1, 28, 28},
                       std::vector<int>{8, 1, 3, 3},
                       /*stride=*/2, /*pad=*/1);                          // [N, 8, 14, 14]
    auto* a1     = nn::relu(g, c1);
    auto* flat   = g.emplace<ReshapeNode>(a1,
                       std::vector<int>{batch, 8, 14, 14},
                       std::vector<int>{batch, 1568});
    auto* z      = g.emplace<MatMulNode>(flat, W);                        // [N, 10]
    auto* b_brd  = g.emplace<BroadcastNode>(b, 0, batch);
    auto* z2     = g.emplace<MatAddNode>(z, b_brd);
    auto* y      = nn::softmax(g, z2);

    // --- Autograd backward (seeded with softmax+CE shortcut) ---
    auto* dz2 = nn::softmax_ce_backward(g, y, oh, batch);
    autograd::BackwardBuilder bb(g);
    bb.seed(z2, dz2);
    bb.build({K1, W, b});

    // --- SGD update ---
    auto* K1_new = nn::sgd_step(g, K1, bb.grad(K1), lr);
    auto* W_new  = nn::sgd_step(g, W,  bb.grad(W),  lr);
    auto* b_new  = nn::sgd_step(g, b,  bb.grad(b),  lr);

    if (print_graph) {
        std::cout << "--- TRAIN GRAPH ---\n";
        PrintVisitor p; g.accept(p); std::cout << "\n";
    }

    exec.clear();
    g.accept(exec);

    // Loss + accuracy
    Tensor y_v = exec.result(y);
    float loss = 0.0f; int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = y_v.data.data() + i * n_cls;
        loss -= std::log(std::max(row[Y[i]], 1e-9f));
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    loss /= batch;

    m.K1 = exec.result(K1_new);
    m.W  = exec.result(W_new);
    m.b  = exec.result(b_new);
    return {loss, correct};
}

// --- Evaluate (forward only) ---------------------------------------------
static float evaluate(const CNN& m, const Tensor& X, const std::vector<int>& Y,
                      metal::Executor& exec)
{
    int batch = X.shape[0];
    int n_cls = m.W.shape[1];

    ComputeGraph g;
    auto* x_flat = g.emplace<InputNode>("X",  X);
    auto* K1     = g.emplace<InputNode>("K1", m.K1);
    auto* W      = g.emplace<InputNode>("W",  m.W);
    auto* b      = g.emplace<InputNode>("b",  m.b);

    auto* x   = g.emplace<ReshapeNode>(x_flat,
                   std::vector<int>{batch, 784},
                   std::vector<int>{batch, 1, 28, 28});
    auto* c1  = nn::conv2d(g, x, K1,
                   std::vector<int>{batch, 1, 28, 28},
                   std::vector<int>{8, 1, 3, 3}, 2, 1);
    auto* a1  = nn::relu(g, c1);
    auto* fl  = g.emplace<ReshapeNode>(a1,
                   std::vector<int>{batch, 8, 14, 14},
                   std::vector<int>{batch, 1568});
    auto* z   = g.emplace<MatMulNode>(fl, W);
    auto* zb  = g.emplace<MatAddNode>(z, g.emplace<BroadcastNode>(b, 0, batch));
    auto* y   = nn::softmax(g, zb);

    exec.clear(); g.accept(exec);
    Tensor y_v = exec.result(y);

    int correct = 0;
    for (int i = 0; i < batch; ++i) {
        const float* row = y_v.data.data() + i * n_cls;
        int pred = 0;
        for (int j = 1; j < n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    return (float)correct / batch;
}

// --- Train loop ----------------------------------------------------------
int main() {
    std::cout << "Loading MNIST...\n";
    auto tr = mnist::load_train();
    auto te = mnist::load_test();
    std::cout << "  train: " << tr.labels.size() << "  test: " << te.labels.size() << "\n";

    constexpr int   batch_size = 64;
    constexpr int   n_epochs   = 3;
    constexpr float lr         = 0.1f;

    std::mt19937 rng(42);
    CNN model = init_cnn(rng);
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
