// MNIST CNN. ONE compute graph holds forward + backward + SGD update.
// Eval runs a forward subgraph (g.accept(visitor, y)) — backward and SGD
// nodes are skipped, parameters aren't modified.

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
    nn::Conv2D conv1;
    nn::Dense  fc;

    CNN(ComputeGraph& g, std::mt19937& rng)
        : conv1(g, /*in*/1, /*out*/8, /*kH*/3, /*kW*/3, /*stride*/2, /*pad*/1, rng)
        , fc(g, 1568, 10, rng) {}

    Node* forward(ComputeGraph& g, Node* x_flat, int batch) {
        auto* x4d  = g.emplace<ReshapeNode>(x_flat,
                       std::vector<int>{batch, 784},
                       std::vector<int>{batch, 1, 28, 28});
        auto* c    = conv1(x4d, std::vector<int>{batch, 1, 28, 28});
        auto* a    = nn::relu(g, c);
        auto* flat = g.emplace<ReshapeNode>(a,
                       std::vector<int>{batch, 8, 14, 14},
                       std::vector<int>{batch, 1568});
        return       fc(flat, batch);
    }

    std::vector<Node*> params() const {
        auto p = conv1.params();
        for (auto* n : fc.params()) p.push_back(n);
        return p;
    }

    template <typename BB>
    void apply_sgd(BB& bb, float lr) {
        conv1.apply_sgd(bb, lr);
        fc.apply_sgd(bb, lr);
    }

    template <typename Exec>
    void refresh(Exec& exec) { conv1.refresh(exec); fc.refresh(exec); }
};

// ---- Trainer: one persistent graph --------------------------------------
struct Trainer {
    static constexpr int batch = 64;
    static constexpr int n_cls = 10;

    ComputeGraph g;                   // declared before model so its ref is valid
    CNN          model;
    InputNode   *X_in  = nullptr;
    InputNode   *OH_in = nullptr;
    Node        *y     = nullptr;

    Trainer(std::mt19937& rng, float lr) : model(g, rng) {
        X_in  = g.emplace<InputNode>("X",  Tensor({batch, 784}));
        OH_in = g.emplace<InputNode>("OH", Tensor({batch, n_cls}));

        auto* logits = model.forward(g, X_in, batch);
        y = nn::softmax(g, logits);

        auto* dz = nn::softmax_ce_backward(g, y, OH_in, batch);
        autograd::BackwardBuilder bb(g);
        bb.seed(logits, dz);
        bb.build(model.params());
        model.apply_sgd(bb, lr);
    }
};

static void fill_train_batch(Trainer& t, const mnist::Dataset& tr,
                             const std::vector<int>& idx, int b_off)
{
    std::fill(t.OH_in->tensor.begin(), t.OH_in->tensor.end(), 0.0f);
    for (int i = 0; i < Trainer::batch; ++i) {
        int s = idx[b_off + i];
        std::memcpy(t.X_in->tensor.data() + i * 784,
                    tr.images.data()      + s * 784,
                    784 * sizeof(float));
        t.OH_in->tensor[i * Trainer::n_cls + tr.labels[s]] = 1.0f;
    }
}

static std::pair<float, int>
loss_and_acc(const Tensor& y, const std::vector<int>& Y) {
    float loss = 0.0f; int correct = 0;
    for (int i = 0; i < Trainer::batch; ++i) {
        const float* row = y.data() + i * Trainer::n_cls;
        loss -= std::log(std::max(row[Y[i]], 1e-9f));
        int pred = 0;
        for (int j = 1; j < Trainer::n_cls; ++j) if (row[j] > row[pred]) pred = j;
        if (pred == Y[i]) ++correct;
    }
    return {loss / Trainer::batch, correct};
}

static float evaluate(Trainer& t, const mnist::Dataset& te, metal::Executor& exec) {
    int n = (int)te.labels.size();
    int correct = 0, total = 0;
    for (int b = 0; b + Trainer::batch <= n; b += Trainer::batch) {
        std::memcpy(t.X_in->tensor.data(),
                    te.images.data() + b * 784,
                    Trainer::batch * 784 * sizeof(float));
        exec.reset();
        t.g.accept(exec, t.y);              // forward subgraph only
        const Tensor& yv = exec.result(t.y);
        for (int i = 0; i < Trainer::batch; ++i) {
            const float* row = yv.data() + i * Trainer::n_cls;
            int pred = 0;
            for (int j = 1; j < Trainer::n_cls; ++j) if (row[j] > row[pred]) pred = j;
            if (pred == te.labels[b + i]) ++correct;
        }
        total += Trainer::batch;
    }
    return (float)correct / total;
}

int main() {
    std::cout << "Loading MNIST...\n";
    auto tr = mnist::load_train();
    auto te = mnist::load_test();
    std::cout << "  train: " << tr.labels.size() << "  test: " << te.labels.size() << "\n";

    constexpr int   n_epochs = 3;
    constexpr float lr       = 0.1f;

    std::mt19937 rng(42);
    Trainer trainer(rng, lr);
    metal::Executor exec;
    std::cout << "[Metal] device: " << exec.device_name() << "\n\n";

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
            trainer.g.accept(exec);

            std::vector<int> Y(Trainer::batch);
            for (int i = 0; i < Trainer::batch; ++i) Y[i] = tr.labels[idx[b + i]];
            auto [l, c] = loss_and_acc(exec.result(trainer.y), Y);
            loss += l; correct += c; ++batches;

            trainer.model.refresh(exec);
        }
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        float test_acc = evaluate(trainer, te, exec);

        std::cout << "epoch " << (epoch + 1) << "/" << n_epochs
                  << "  loss "      << (loss / batches)
                  << "  train_acc " << (100.0f * correct / (batches * Trainer::batch)) << "%"
                  << "  test_acc "  << (100.0f * test_acc) << "%"
                  << "  ("          << secs << "s)\n";
    }
    return 0;
}
