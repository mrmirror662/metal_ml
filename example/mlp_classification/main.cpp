// MNIST MLP. ONE compute graph holds forward + backward + SGD update.
// - Training step: full accept (runs every node)
// - Eval step:     subgraph accept stopping at the softmax output, so the
//                  backward and SGD nodes are skipped — parameters aren't
//                  modified during evaluation.
//
// Graph structure is built once. Per-batch we mutate InputNode tensors in
// place; the executor detects existing buffers and just memcpy's new data.

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

    Node* forward(ComputeGraph& g, Node* x, int batch) {
        return fc2(g, nn::relu(g, fc1(g, x, batch)), batch);
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

// ---- Trainer: ONE graph for everything ----------------------------------
struct Trainer {
    static constexpr int batch = 64;
    static constexpr int n_cls = 10;

    MLP          mlp;
    ComputeGraph g;
    InputNode   *X_in  = nullptr;
    InputNode   *OH_in = nullptr;
    Node        *y     = nullptr;     // softmax output — used by both train & eval

    Trainer(std::mt19937& rng, float lr) : mlp(rng) {
        X_in  = g.emplace<InputNode>("X",  Tensor({batch, 784}));
        OH_in = g.emplace<InputNode>("OH", Tensor({batch, n_cls}));

        auto* logits = mlp.forward(g, X_in, batch);
        y = nn::softmax(g, logits);

        auto* dz = nn::softmax_ce_backward(g, y, OH_in, batch);
        autograd::BackwardBuilder bb(g);
        bb.seed(logits, dz);
        bb.build(mlp.params());
        mlp.apply_sgd(g, bb, lr);
    }
};

// ---- Helpers -------------------------------------------------------------
static void fill_train_batch(Trainer& t, const mnist::Dataset& tr,
                             const std::vector<int>& idx, int b_off)
{
    std::fill(t.OH_in->tensor.data.begin(), t.OH_in->tensor.data.end(), 0.0f);
    for (int i = 0; i < Trainer::batch; ++i) {
        int s = idx[b_off + i];
        std::memcpy(t.X_in->tensor.data.data() + i * 784,
                    tr.images.data.data()      + s * 784,
                    784 * sizeof(float));
        t.OH_in->tensor.data[i * Trainer::n_cls + tr.labels[s]] = 1.0f;
    }
}

static std::pair<float, int>
loss_and_acc(const Tensor& y, const std::vector<int>& Y) {
    float loss = 0.0f; int correct = 0;
    for (int i = 0; i < Trainer::batch; ++i) {
        const float* row = y.data.data() + i * Trainer::n_cls;
        loss -= std::log(std::max(row[Y[i]], 1e-9f));
        int pred = 0;
        for (int j = 1; j < Trainer::n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    return {loss / Trainer::batch, correct};
}

// ---- Eval: run forward subgraph (everything that y depends on) ----------
template <typename Executor>
static float evaluate(Trainer& t, const mnist::Dataset& te, Executor& exec) {
    int n = (int)te.labels.size();
    int correct = 0, total = 0;
    for (int b = 0; b + Trainer::batch <= n; b += Trainer::batch) {
        std::memcpy(t.X_in->tensor.data.data(),
                    te.images.data.data() + b * 784,
                    Trainer::batch * 784 * sizeof(float));
        exec.reset();
        t.g.accept(exec, t.y);              // <-- subgraph: forward only
        const Tensor& yv = exec.result(t.y);
        for (int i = 0; i < Trainer::batch; ++i) {
            const float* row = yv.data.data() + i * Trainer::n_cls;
            int pred = 0;
            for (int j = 1; j < Trainer::n_cls; ++j) if (row[j] > row[pred]) pred = j;
            if (pred == te.labels[b + i]) ++correct;
        }
        total += Trainer::batch;
    }
    return (float)correct / total;
}

// ---- Train loop ----------------------------------------------------------
template <typename Executor>
static void train(const char* label, const mnist::Dataset& tr, const mnist::Dataset& te,
                  Executor& exec)
{
    constexpr int   n_epochs = 5;
    constexpr float lr       = 0.1f;
    std::cout << "\n===== Training on " << label << " =====\n";

    std::mt19937 rng(42);
    Trainer trainer(rng, lr);

    int n = (int)tr.labels.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng);
        float loss = 0.0f; int correct = 0, batches = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int b = 0; b + Trainer::batch <= n; b += Trainer::batch) {
            fill_train_batch(trainer, tr, idx, b);

            exec.reset();
            trainer.g.accept(exec);          // full graph: fwd + bwd + SGD

            std::vector<int> Y(Trainer::batch);
            for (int i = 0; i < Trainer::batch; ++i) Y[i] = tr.labels[idx[b + i]];
            auto [l, c] = loss_and_acc(exec.result(trainer.y), Y);
            loss += l; correct += c; ++batches;

            trainer.mlp.refresh(exec);
        }
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        float test_acc = evaluate(trainer, te, exec);

        std::cout << "[" << label << "] ep " << (epoch + 1) << "/" << n_epochs
                  << "  loss "      << (loss / batches)
                  << "  train_acc " << (100.0f * correct / (batches * Trainer::batch)) << "%"
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
