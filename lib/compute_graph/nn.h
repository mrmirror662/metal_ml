#pragma once
// Composable NN building blocks. Each function emits a subgraph of existing
// primitives — no new node types or kernels. Adding new architectures means
// composing these, not extending the lib.

#include "cg.h"

namespace cg::nn {

// ===== Forward layers =====================================================

// y = x @ W + broadcast(b, axis=0, count=batch)
inline Node* linear(ComputeGraph& g, Node* x, Node* W, Node* b, int batch) {
    auto* z   = g.emplace<MatMulNode>(x, W);
    auto* brd = g.emplace<BroadcastNode>(b, 0, batch);
    return     g.emplace<MatAddNode>(z, brd);
}

inline Node* relu   (ComputeGraph& g, Node* x) { return g.emplace<MapNode>(x, MapOp::ReLU);    }
inline Node* softmax(ComputeGraph& g, Node* x) { return g.emplace<MapNode>(x, MapOp::Softmax); }

// ===== Activation derivatives =============================================

// dx = dy ⊙ (z > 0)
inline Node* relu_backward(ComputeGraph& g, Node* dy, Node* z) {
    auto* mask = g.emplace<MapNode>(z, MapOp::Step);
    return     g.emplace<HadamardNode>(dy, mask);
}

// (y - target) / batch  — fused softmax + cross-entropy gradient
inline Node* softmax_ce_backward(ComputeGraph& g, Node* y, Node* one_hot, int batch) {
    auto* neg = g.emplace<ScaleNode>(one_hot, -1.0f);
    auto* sub = g.emplace<MatAddNode>(y, neg);
    return     g.emplace<ScaleNode>(sub, 1.0f / batch);
}

// ===== MatMul derivatives =================================================
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

// ===== Conv2D (im2col + GEMM) =============================================
//
// Forward: y = conv2d(x, k, stride, pad) where
//   x: [N, C_in, H, W]   k: [C_out, C_in, kH, kW]   y: [N, C_out, Hout, Wout]
//
// Memory cost: im2col matrix is C_in*kH*kW× the input (standard cuDNN cost).
// Compute cost: dominated by the matmul, which uses the simd_tiled kernel.
inline Node* conv2d(ComputeGraph& g, Node* x, Node* k,
                    const std::vector<int>& x_shape,    // [N, C, H, W]
                    const std::vector<int>& k_shape,    // [Cout, C, kH, kW]
                    int stride = 1, int pad = 0)
{
    int N = x_shape[0], H = x_shape[2], W = x_shape[3];
    int Cout = k_shape[0], C = k_shape[1], kH = k_shape[2], kW = k_shape[3];
    int Hout = (H + 2 * pad - kH) / stride + 1;
    int Wout = (W + 2 * pad - kW) / stride + 1;

    // Patches:  [N*Hout*Wout, C*kH*kW]
    auto* patches = g.emplace<Im2ColNode>(x, x_shape, kH, kW, stride, pad);

    // Kernel as matrix: [Cout, C*kH*kW] → transpose → [C*kH*kW, Cout]
    auto* k_mat = g.emplace<ReshapeNode>(k, k_shape, std::vector<int>{Cout, C * kH * kW});
    auto* k_T   = g.emplace<TransposeNode>(k_mat);

    // [N*Hout*Wout, C*kH*kW] @ [C*kH*kW, Cout] = [N*Hout*Wout, Cout]
    auto* y_flat = g.emplace<MatMulNode>(patches, k_T);

    // Reshape to NCHW: [N*Hout*Wout, Cout] -> [N, Hout, Wout, Cout] -> [N, Cout, Hout, Wout]
    auto* y_nhwc = g.emplace<ReshapeNode>(y_flat,
        std::vector<int>{N * Hout * Wout, Cout},
        std::vector<int>{N, Hout, Wout, Cout});
    return g.emplace<TransposeNode>(y_nhwc, std::vector<int>{0, 3, 1, 2});
}

// ===== Optimizer steps ====================================================

// p_new = p + (-lr) * grad
inline Node* sgd_step(ComputeGraph& g, Node* p, Node* grad, float lr) {
    auto* s = g.emplace<ScaleNode>(grad, -lr);
    return    g.emplace<MatAddNode>(p, s);
}

} // namespace cg::nn
