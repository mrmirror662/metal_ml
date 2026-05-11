#pragma once
// Composable NN building blocks.
//
// Two layers of API:
//
//   1) High-level "PyTorch-like" — preferred for new code:
//        - `nn::Tensor`            : (graph*, node*, shape) wrapper
//        - `nn::Module`            : base class with forward() + operator()
//        - `nn::Linear`            : fully-connected (was `nn::Dense`)
//        - `nn::Conv2D`            : 2D convolution with optional bias
//        - `nn::ConvTranspose2D`   : transposed 2D conv
//        - `nn::BatchNorm2D`       : inference-mode batch norm
//        - `nn::MaxPool2D`         : 2D max pool
//        - `nn::UpsampleNearest`   : nearest-neighbor upsample
//        - `nn::ReLU`/`Sigmoid`    : stateless modules (composable in Sequential)
//        - `nn::Sequential`        : chain modules
//      Stateless free functions for ad-hoc use:
//        - `nn::relu(t)`, `nn::sigmoid(t)`, `nn::softmax(t)`
//        - `nn::concat(a, b)`, `nn::maxpool2d(t, k, s, p)`, `nn::upsample_nearest(t, scale)`
//      Factory + run:
//        - `nn::input(g, tensor, is_constant)`  -> Tensor
//        - `t.eval(executor)`                   -> cg::Tensor result
//
//   2) Low-level subgraph emitters — for backends/autograd/custom code:
//        - `linear(g, x, W, b, batch)`, `conv2d(g, x, k, b, …)`
//        - `transposed_conv2d(g, x, k, b, …)`
//        - `relu_backward(g, dy, z)`, `sigmoid_backward(g, dy, y)`
//        - `softmax_ce_backward(g, y, one_hot, batch)`
//        - `matmul_grad_lhs/rhs(g, …)`, `bias_grad(g, dy)`
//        - `sgd_step(g, p, grad, lr)`
//      These remain because autograd.h composes them into backward subgraphs.

#include "cg.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cg::nn {

// ============================================================================
// LOW-LEVEL: subgraph emitters and gradient helpers (unchanged from before)
// ============================================================================

inline Node* linear(ComputeGraph& g, Node* x, Node* W, Node* b, int batch) {
    auto* z   = g.emplace<MatMulNode>(x, W);
    auto* brd = g.emplace<BroadcastNode>(b, 0, batch);
    return     g.emplace<MatAddNode>(z, brd);
}

inline Node* relu_node   (ComputeGraph& g, Node* x) { return g.emplace<MapNode>(x, MapOp::ReLU);    }
inline Node* softmax_node(ComputeGraph& g, Node* x) { return g.emplace<MapNode>(x, MapOp::Softmax); }
inline Node* sigmoid_node(ComputeGraph& g, Node* x) { return g.emplace<MapNode>(x, MapOp::Sigmoid); }

// dx = dy ⊙ (z > 0)
inline Node* relu_backward(ComputeGraph& g, Node* dy, Node* z) {
    auto* mask = g.emplace<MapNode>(z, MapOp::Step);
    return     g.emplace<HadamardNode>(dy, mask);
}

// dx = dy ⊙ y ⊙ (1 - y).  Composed without a "1 - x" primitive as dy * (y - y²).
inline Node* sigmoid_backward(ComputeGraph& g, Node* dy, Node* y) {
    auto* y2          = g.emplace<HadamardNode>(y, y);
    auto* neg_y2      = g.emplace<ScaleNode>(y2, -1.0f);
    auto* y_minus_y2  = g.emplace<MatAddNode>(y, neg_y2);
    return              g.emplace<HadamardNode>(dy, y_minus_y2);
}

// (y - target) / batch  — fused softmax + cross-entropy gradient
inline Node* softmax_ce_backward(ComputeGraph& g, Node* y, Node* one_hot, int batch) {
    auto* neg = g.emplace<ScaleNode>(one_hot, -1.0f);
    auto* sub = g.emplace<MatAddNode>(y, neg);
    return     g.emplace<ScaleNode>(sub, 1.0f / batch);
}

// For y = lhs @ rhs:   d_lhs = dy @ rhs.T   ;   d_rhs = lhs.T @ dy
inline Node* matmul_grad_lhs(ComputeGraph& g, Node* dy, Node* rhs) {
    auto* T = g.emplace<TransposeNode>(rhs);
    return    g.emplace<MatMulNode>(dy, T);
}
inline Node* matmul_grad_rhs(ComputeGraph& g, Node* lhs, Node* dy) {
    auto* T = g.emplace<TransposeNode>(lhs);
    return    g.emplace<MatMulNode>(T, dy);
}

// db = sum(dy, axis=0) — gradient w.r.t. a [1, K] bias broadcast along batch
inline Node* bias_grad(ComputeGraph& g, Node* dy) {
    return g.emplace<ReduceNode>(dy, ReduceOp::Sum, 0);
}

// Conv2D subgraph. Same as before; called by the high-level Conv2D module
// and by users who want explicit control.
inline Node* conv2d(ComputeGraph& g, Node* x, Node* k, Node* b,
                    const std::vector<int>& x_shape,    // [N, C, H, W]
                    const std::vector<int>& k_shape,    // [Cout, C, kH, kW]
                    int stride = 1, int pad = 0)
{
    int N = x_shape[0], H = x_shape[2], W = x_shape[3];
    int Cout = k_shape[0], C = k_shape[1], kH = k_shape[2], kW = k_shape[3];
    int Hout = (H + 2 * pad - kH) / stride + 1;
    int Wout = (W + 2 * pad - kW) / stride + 1;

    auto* patches = g.emplace<Im2ColNode>(x, x_shape, kH, kW, stride, pad);
    auto* k_mat   = g.emplace<ReshapeNode>(k, k_shape, std::vector<int>{Cout, C * kH * kW});
    auto* k_T     = g.emplace<TransposeNode>(k_mat);
    Node* y_flat  = g.emplace<MatMulNode>(patches, k_T);

    if (b) {
        auto* b_brd = g.emplace<BroadcastNode>(b, 0, N * Hout * Wout);
        y_flat      = g.emplace<MatAddNode>(y_flat, b_brd);
    }
    auto* y_nhwc = g.emplace<ReshapeNode>(y_flat,
        std::vector<int>{N * Hout * Wout, Cout},
        std::vector<int>{N, Hout, Wout, Cout});
    return g.emplace<TransposeNode>(y_nhwc, std::vector<int>{0, 3, 1, 2});
}

// Transposed Conv2D subgraph (PyTorch nn.ConvTranspose2d). No new primitive —
// matmul + col2im. Weight shape is [Cin, Cout, kH, kW]. Output dims:
//   Hout = (Hin - 1) * stride - 2*pad + kH.
inline Node* transposed_conv2d(ComputeGraph& g, Node* x, Node* k, Node* b,
                               const std::vector<int>& x_shape,
                               const std::vector<int>& k_shape,
                               int stride = 1, int pad = 0)
{
    int N = x_shape[0], Cin = x_shape[1], H = x_shape[2], W = x_shape[3];
    int Cin_k = k_shape[0], Cout = k_shape[1], kH = k_shape[2], kW = k_shape[3];
    if (Cin != Cin_k) throw std::runtime_error("transposed_conv2d: Cin mismatch");
    int Hout = (H - 1) * stride - 2 * pad + kH;
    int Wout = (W - 1) * stride - 2 * pad + kW;

    auto* x_nhwc = g.emplace<TransposeNode>(x, std::vector<int>{0, 2, 3, 1});
    auto* x_flat = g.emplace<ReshapeNode>(x_nhwc,
        std::vector<int>{N, H, W, Cin}, std::vector<int>{N * H * W, Cin});
    auto* k_mat  = g.emplace<ReshapeNode>(k, k_shape,
        std::vector<int>{Cin, Cout * kH * kW});
    auto* prod   = g.emplace<MatMulNode>(x_flat, k_mat);
    Node* y = g.emplace<Col2ImNode>(prod,
        std::vector<int>{N, Cout, Hout, Wout}, kH, kW, stride, pad);
    if (b) {
        auto* b_4d  = g.emplace<ReshapeNode>(b,
            std::vector<int>{1, Cout}, std::vector<int>{1, Cout, 1, 1});
        auto* b_h   = g.emplace<BroadcastNode>(b_4d, 2, Hout);
        auto* b_hw  = g.emplace<BroadcastNode>(b_h,  3, Wout);
        auto* b_all = g.emplace<BroadcastNode>(b_hw, 0, N);
        y = g.emplace<MatAddNode>(y, b_all);
    }
    return y;
}

// p_new = p + (-lr) * grad
inline Node* sgd_step(ComputeGraph& g, Node* p, Node* grad, float lr) {
    auto* s = g.emplace<ScaleNode>(grad, -lr);
    return    g.emplace<MatAddNode>(p, s);
}

// ============================================================================
// HIGH-LEVEL API — PyTorch-style modules
// ============================================================================

// `nn::Tensor` is a lightweight handle wrapping (graph, node, shape). Modules
// produce + consume Tensors, so shape threading is hidden from the user.
class Tensor {
public:
    Tensor() = default;
    Tensor(ComputeGraph* g, Node* n, std::vector<int> shape)
        : g_(g), node_(n), shape_(std::move(shape)) {}

    ComputeGraph*           graph() const { return g_; }
    Node*                   node()  const { return node_; }
    const std::vector<int>& shape() const { return shape_; }
    int                     ndim()  const { return (int)shape_.size(); }
    int                     size(int i) const {
        int idx = i < 0 ? (int)shape_.size() + i : i;
        return shape_.at(idx);
    }

    // Materialize the result of evaluating up to this node with the given
    // executor. Runs only the subgraph this Tensor depends on, then returns
    // the executor's result tensor. `Executor` must expose `result(Node*)`
    // (both cg::cpu::Executor and cg::metal::Executor do).
    template <typename Executor>
    const cg::Tensor& eval(Executor& exec) const {
        if (!g_ || !node_) throw std::runtime_error("Tensor::eval: empty handle");
        g_->accept(exec, node_);
        return exec.result(node_);
    }

private:
    ComputeGraph*    g_     = nullptr;
    Node*            node_  = nullptr;
    std::vector<int> shape_;
};

// Convenience: create an input Tensor. Returns a Tensor wrapping a new
// InputNode in the given graph.
inline Tensor input(ComputeGraph& g, std::string name, cg::Tensor t,
                    bool is_constant = false)
{
    auto shape = t.shape();
    auto* n = g.emplace<InputNode>(std::move(name), std::move(t), is_constant);
    return Tensor(&g, n, std::move(shape));
}

// Base class. PyTorch's nn.Module analog. Modules are value-typed; treat them
// as data + a forward() method. Construct with PyTorch-style config args;
// parameters are created at construction and registered into the graph lazily
// on first forward().
class Module {
public:
    virtual      ~Module() = default;
    virtual Tensor forward(Tensor x) = 0;
    Tensor       operator()(Tensor x) { return forward(x); }
};

// ----- Linear (was Dense) ---------------------------------------------------
class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true)
        : in_(in_features), out_(out_features), use_bias_(bias),
          W_({in_features, out_features}),
          b_(bias ? cg::Tensor({1, out_features}) : cg::Tensor()) {}

    // He init for ReLU networks. Optional — leave weights at zero for explicit
    // weight loading (e.g. safetensors).
    void reset_parameters(std::mt19937& rng) {
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / in_));
        for (auto& v : W_) v = d(rng);
        if (use_bias_) for (auto& v : b_) v = 0.0f;
    }

    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        if (!W_n_) {
            W_n_ = g->emplace<InputNode>("W", W_);
            if (use_bias_) b_n_ = g->emplace<InputNode>("b", b_);
            graph_ = g;
        }
        int batch = x.size(0);
        Node* y;
        if (use_bias_) y = linear(*g, x.node(), W_n_, b_n_, batch);
        else           y = g->emplace<MatMulNode>(x.node(), W_n_);
        return Tensor(g, y, {batch, out_});
    }

    // Parameter / tensor access (for safetensors loading, etc.).
    cg::Tensor&  weight()    { return W_; }
    cg::Tensor&  bias()      { return b_; }
    InputNode*   weight_node() const { return W_n_; }
    InputNode*   bias_node()   const { return b_n_; }

    std::vector<Node*> params() const {
        if (use_bias_) return {W_n_, b_n_};
        return {W_n_};
    }

    template <typename BB>
    void apply_sgd(BB& bb, float lr) {
        if (!W_n_ || !graph_)
            throw std::runtime_error("Linear::apply_sgd: not bound (call forward first)");
        Node* gW = bb.grad(W_n_);
        if (!gW) throw std::runtime_error("Linear::apply_sgd: missing grad for W");
        graph_->emplace<AssignNode>(W_n_, sgd_step(*graph_, W_n_, gW, lr));
        if (use_bias_) {
            Node* gb = bb.grad(b_n_);
            if (!gb) throw std::runtime_error("Linear::apply_sgd: missing grad for b");
            graph_->emplace<AssignNode>(b_n_, sgd_step(*graph_, b_n_, gb, lr));
        }
    }

private:
    int           in_, out_;
    bool          use_bias_;
    cg::Tensor    W_, b_;
    InputNode    *W_n_ = nullptr, *b_n_ = nullptr;
    ComputeGraph *graph_ = nullptr;   // captured at first forward
};

// ----- Conv2D ---------------------------------------------------------------
class Conv2D : public Module {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool bias = true)
        : Conv2D(in_channels, out_channels, kernel_size, kernel_size,
                 stride, padding, bias) {}

    Conv2D(int in_channels, int out_channels, int kH, int kW,
           int stride, int padding, bool bias)
        : in_ch_(in_channels), out_ch_(out_channels), kH_(kH), kW_(kW),
          stride_(stride), pad_(padding), use_bias_(bias),
          K_({out_channels, in_channels, kH, kW}),
          b_(bias ? cg::Tensor({1, out_channels}) : cg::Tensor()) {}

    void reset_parameters(std::mt19937& rng) {
        int fan_in = in_ch_ * kH_ * kW_;
        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / fan_in));
        for (auto& v : K_) v = d(rng);
        if (use_bias_) for (auto& v : b_) v = 0.0f;
    }

    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        if (!K_n_) {
            K_n_ = g->emplace<InputNode>("K", K_);
            if (use_bias_) b_n_ = g->emplace<InputNode>("b", b_);
            graph_ = g;
        }
        auto& xs = x.shape();
        Node* y = conv2d(*g, x.node(), K_n_, b_n_, xs, K_.shape(), stride_, pad_);
        int Ho = (xs[2] + 2*pad_ - kH_) / stride_ + 1;
        int Wo = (xs[3] + 2*pad_ - kW_) / stride_ + 1;
        return Tensor(g, y, {xs[0], out_ch_, Ho, Wo});
    }

    cg::Tensor&  weight()      { return K_; }
    cg::Tensor&  bias()        { return b_; }
    InputNode*   weight_node() const { return K_n_; }
    InputNode*   bias_node()   const { return b_n_; }

    std::vector<Node*> params() const {
        if (use_bias_) return {K_n_, b_n_};
        return {K_n_};
    }

    template <typename BB>
    void apply_sgd(BB& bb, float lr) {
        if (!K_n_) throw std::runtime_error("Conv2D::apply_sgd: not bound");
        Node* gK = bb.grad(K_n_); if (!gK) throw std::runtime_error("Conv2D::apply_sgd: no grad for K");
        graph_->emplace<AssignNode>(K_n_, sgd_step(*graph_, K_n_, gK, lr));
        if (use_bias_) {
            Node* gb = bb.grad(b_n_); if (!gb) throw std::runtime_error("Conv2D::apply_sgd: no grad for b");
            graph_->emplace<AssignNode>(b_n_, sgd_step(*graph_, b_n_, gb, lr));
        }
    }

private:
    int          in_ch_, out_ch_, kH_, kW_, stride_, pad_;
    bool         use_bias_;
    cg::Tensor   K_, b_;
    InputNode   *K_n_ = nullptr, *b_n_ = nullptr;
    ComputeGraph* graph_ = nullptr;
};

// ----- ConvTranspose2D ------------------------------------------------------
class ConvTranspose2D : public Module {
public:
    ConvTranspose2D(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 0, bool bias = true)
        : out_ch_(out_channels), kH_(kernel_size), kW_(kernel_size),
          stride_(stride), pad_(padding), use_bias_(bias),
          // PyTorch convention: weight shape [Cin, Cout, kH, kW]
          K_({in_channels, out_channels, kernel_size, kernel_size}),
          b_(bias ? cg::Tensor({1, out_channels}) : cg::Tensor()) {}

    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        if (!K_n_) {
            K_n_ = g->emplace<InputNode>("Kt", K_);
            if (use_bias_) b_n_ = g->emplace<InputNode>("bt", b_);
        }
        auto& xs = x.shape();
        Node* y = transposed_conv2d(*g, x.node(), K_n_, b_n_, xs, K_.shape(), stride_, pad_);
        int Ho = (xs[2] - 1) * stride_ - 2*pad_ + kH_;
        int Wo = (xs[3] - 1) * stride_ - 2*pad_ + kW_;
        return Tensor(g, y, {xs[0], out_ch_, Ho, Wo});
    }

    cg::Tensor& weight()      { return K_; }
    cg::Tensor& bias()        { return b_; }

private:
    int        out_ch_, kH_, kW_, stride_, pad_;
    bool       use_bias_;
    cg::Tensor K_, b_;
    InputNode *K_n_ = nullptr, *b_n_ = nullptr;
};

// ----- BatchNorm2D ----------------------------------------------------------
class BatchNorm2D : public Module {
public:
    explicit BatchNorm2D(int num_features, float eps = 1e-5f)
        : eps_(eps),
          gamma_({num_features}), beta_({num_features}),
          mean_({num_features}), var_({num_features})
    {
        for (int i = 0; i < num_features; ++i) { gamma_[i] = 1.0f; var_[i] = 1.0f; }
    }

    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        if (!gamma_n_) {
            gamma_n_ = g->emplace<InputNode>("bn.g", gamma_);
            beta_n_  = g->emplace<InputNode>("bn.b", beta_);
            mean_n_  = g->emplace<InputNode>("bn.m", mean_);
            var_n_   = g->emplace<InputNode>("bn.v", var_);
        }
        auto* out = g->emplace<BatchNorm2DNode>(x.node(), gamma_n_, beta_n_, mean_n_, var_n_, eps_);
        return Tensor(g, out, x.shape());
    }

    cg::Tensor& weight()       { return gamma_; }   // PyTorch naming
    cg::Tensor& bias()         { return beta_;  }
    cg::Tensor& running_mean() { return mean_;  }
    cg::Tensor& running_var()  { return var_;   }

private:
    float      eps_;
    cg::Tensor gamma_, beta_, mean_, var_;
    InputNode *gamma_n_ = nullptr, *beta_n_ = nullptr, *mean_n_ = nullptr, *var_n_ = nullptr;
};

// ----- MaxPool2D / UpsampleNearest (stateless modules) ---------------------
class MaxPool2D : public Module {
public:
    MaxPool2D(int kernel_size, int stride = 0, int padding = 0)
        : k_(kernel_size), stride_(stride > 0 ? stride : kernel_size), pad_(padding) {}

    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        auto& xs = x.shape();
        auto* out = g->emplace<MaxPool2DNode>(x.node(), xs, k_, stride_, pad_);
        int Ho = (xs[2] + 2*pad_ - k_) / stride_ + 1;
        int Wo = (xs[3] + 2*pad_ - k_) / stride_ + 1;
        return Tensor(g, out, {xs[0], xs[1], Ho, Wo});
    }
private:
    int k_, stride_, pad_;
};

class UpsampleNearest : public Module {
public:
    explicit UpsampleNearest(int scale) : scale_(scale) {}
    Tensor forward(Tensor x) override {
        auto* g = x.graph();
        auto& xs = x.shape();
        auto* out = g->emplace<UpsampleNearestNode>(x.node(), xs, scale_);
        return Tensor(g, out, {xs[0], xs[1], xs[2]*scale_, xs[3]*scale_});
    }
private:
    int scale_;
};

class ReLU : public Module {
public:
    Tensor forward(Tensor x) override {
        return Tensor(x.graph(), relu_node(*x.graph(), x.node()), x.shape());
    }
};

class Sigmoid : public Module {
public:
    Tensor forward(Tensor x) override {
        return Tensor(x.graph(), sigmoid_node(*x.graph(), x.node()), x.shape());
    }
};

class Softmax : public Module {
public:
    Tensor forward(Tensor x) override {
        return Tensor(x.graph(), softmax_node(*x.graph(), x.node()), x.shape());
    }
};

// ----- Stateless free functions --------------------------------------------
inline Tensor relu   (Tensor x) { return Tensor(x.graph(), relu_node   (*x.graph(), x.node()), x.shape()); }
inline Tensor sigmoid(Tensor x) { return Tensor(x.graph(), sigmoid_node(*x.graph(), x.node()), x.shape()); }
inline Tensor softmax(Tensor x) { return Tensor(x.graph(), softmax_node(*x.graph(), x.node()), x.shape()); }

// Channel-axis concat ([N, Ca, H, W] + [N, Cb, H, W] -> [N, Ca+Cb, H, W])
inline Tensor concat(Tensor a, Tensor b) {
    auto* g = a.graph();
    auto* n = g->emplace<ConcatNode>(a.node(), b.node(), a.shape(), b.shape());
    auto shape = a.shape();
    shape[1] += b.size(1);
    return Tensor(g, n, std::move(shape));
}

// One-shot max-pool / upsample without instantiating a Module.
inline Tensor maxpool2d(Tensor x, int k, int stride = 0, int pad = 0) {
    if (stride <= 0) stride = k;
    auto& xs = x.shape();
    auto* out = x.graph()->emplace<MaxPool2DNode>(x.node(), xs, k, stride, pad);
    int Ho = (xs[2] + 2*pad - k) / stride + 1;
    int Wo = (xs[3] + 2*pad - k) / stride + 1;
    return Tensor(x.graph(), out, {xs[0], xs[1], Ho, Wo});
}
inline Tensor upsample_nearest(Tensor x, int scale) {
    auto& xs = x.shape();
    auto* out = x.graph()->emplace<UpsampleNearestNode>(x.node(), xs, scale);
    return Tensor(x.graph(), out, {xs[0], xs[1], xs[2]*scale, xs[3]*scale});
}

// ----- Sequential ----------------------------------------------------------
//
// Holds child modules by std::unique_ptr<Module>. Forward chains them.
//
//   nn::Sequential model;
//   model.add<nn::Conv2D>(3, 16, 3, 1, 1);
//   model.add<nn::BatchNorm2D>(16);
//   model.add<nn::ReLU>();
//
class Sequential : public Module {
public:
    Sequential() = default;

    // Variadic-template add: `model.add<Conv2D>(3, 16, 3)` constructs in place.
    template <typename M, typename... Args>
    Sequential& add(Args&&... args) {
        layers_.push_back(std::make_unique<M>(std::forward<Args>(args)...));
        return *this;
    }
    // Take ownership of an externally constructed module.
    Sequential& add(std::unique_ptr<Module> m) {
        layers_.push_back(std::move(m));
        return *this;
    }

    Tensor forward(Tensor x) override {
        for (auto& l : layers_) x = (*l)(x);
        return x;
    }

    Module& at(size_t i) { return *layers_.at(i); }
    size_t  size() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Module>> layers_;
};

// ===== Aliases (don't break existing call sites) ==========================
//
// `nn::relu`/`nn::sigmoid`/`nn::softmax` already overload free-fn (above) and
// the in-place graph emitters (below) — name resolution picks the right one
// based on the first argument type (Tensor vs ComputeGraph&).

inline Node* relu   (ComputeGraph& g, Node* x) { return relu_node   (g, x); }
inline Node* softmax(ComputeGraph& g, Node* x) { return softmax_node(g, x); }
inline Node* sigmoid(ComputeGraph& g, Node* x) { return sigmoid_node(g, x); }

// Compatibility alias for the old class name; new code should prefer Linear.
using Dense = Linear;

} // namespace cg::nn
