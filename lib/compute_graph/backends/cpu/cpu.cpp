#include "cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cg::cpu {

// Iterate over every multi-index in `shape`, calling fn(flat_index, idx).
static void foreach_index(const std::vector<int>& shape,
                          const std::function<void(int, const std::vector<int>&)>& fn) {
    int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
    std::vector<int> idx(shape.size(), 0);
    for (int flat = 0; flat < total; ++flat) {
        fn(flat, idx);
        // advance idx (row-major)
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

void Executor::visit(cg::InputNode& node) {
    // For constant inputs, skip the tensor copy on every pass after the
    // first — the value can't have changed. Matches the Metal executor's
    // pin-weights behaviour so both backends benefit identically.
    if (node.is_constant) {
        auto it = results_.find(&node);
        if (it != results_.end() && it->second.shape() == node.tensor.shape()) return;
    }
    results_[&node] = node.tensor;
}

void Executor::visit(cg::MatAddNode& node) {
    const Tensor& A = results_.at(node.lhs);
    const Tensor& B = results_.at(node.rhs);
    if (A.shape() != B.shape())
        throw std::runtime_error("MatAdd: shape mismatch");

    Tensor C(A.shape());
    for (int i = 0; i < A.numel(); ++i)
        C[i] = A[i] + B[i];
    results_[&node] = std::move(C);
}

// Tiled matmul: C[MxN] += A[MxK] @ B[KxN]
// Tile size chosen to keep the A-tile and B-tile in L1 cache (~32KB).
static constexpr int TILE = 64;

static void tiled_matmul(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         int M, int K, int N) {
    for (int i = 0; i < M; i += TILE) {
        int i_end = std::min(i + TILE, M);
        for (int k = 0; k < K; k += TILE) {
            int k_end = std::min(k + TILE, K);
            for (int j = 0; j < N; j += TILE) {
                int j_end = std::min(j + TILE, N);
                // micro-kernel: accumulate the (i,k) x (k,j) tile into C(i,j)
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        float a_val = A[ii * K + kk]; // hoisted: reused across j
                        for (int jj = j; jj < j_end; ++jj)
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                    }
                }
            }
        }
    }
}

void Executor::visit(cg::HadamardNode& node) {
    const Tensor& A = results_.at(node.lhs);
    const Tensor& B = results_.at(node.rhs);
    if (A.shape() != B.shape())
        throw std::runtime_error("Hadamard: shape mismatch");

    Tensor C(A.shape());
    for (int i = 0; i < A.numel(); ++i)
        C[i] = A[i] * B[i];
    results_[&node] = std::move(C);
}

void Executor::visit(cg::MatMulNode& node) {
    const Tensor& A = results_.at(node.lhs);
    const Tensor& B = results_.at(node.rhs);

    int nd = A.ndim();
    if (nd < 2 || B.ndim() < 2)
        throw std::runtime_error("MatMul: inputs must be at least 2D");
    if (A.shape()[nd - 1] != B.shape()[B.ndim() - 2])
        throw std::runtime_error("MatMul: inner dimensions mismatch");

    int M = A.shape()[nd - 2];
    int K = A.shape()[nd - 1];
    int N = B.shape()[B.ndim() - 1];

    std::vector<int> batch_shape(A.shape().begin(), A.shape().begin() + nd - 2);
    int batch = 1;
    for (int d : batch_shape) batch *= d;

    std::vector<int> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor C(out_shape);  // zero-initialized

    for (int b = 0; b < batch; ++b) {
        tiled_matmul(A.data() + b * M * K,
                     B.data() + b * K * N,
                     C.data() + b * M * N,
                     M, K, N);
    }
    results_[&node] = std::move(C);
}

void Executor::visit(cg::ScaleNode& node) {
    const Tensor& in = results_.at(node.input);
    Tensor out(in.shape());
    std::memcpy(out.data(), in.data(), in.numel() * sizeof(float));
    for (auto& v : out) v *= node.scalar;
    results_[&node] = std::move(out);
}

void Executor::visit(cg::MapNode& node) {
    const Tensor& in = results_.at(node.input);
    Tensor out(in.shape());
    std::memcpy(out.data(), in.data(), in.numel() * sizeof(float));

    switch (node.op) {
        case cg::MapOp::ReLU:
            for (auto& v : out) v = std::max(0.0f, v);
            break;

        case cg::MapOp::Step:
            for (auto& v : out) v = v > 0.0f ? 1.0f : 0.0f;
            break;

        case cg::MapOp::Sigmoid:
            for (auto& v : out) v = 1.0f / (1.0f + std::exp(-v));
            break;

        case cg::MapOp::Softmax: {
            // row-wise softmax over the last dimension
            int rows  = out.numel() / out.shape().back();
            int cols  = out.shape().back();
            for (int i = 0; i < rows; ++i) {
                float* row = out.data() + i * cols;
                float  mx  = *std::max_element(row, row + cols);
                float  sum = 0.0f;
                for (int j = 0; j < cols; ++j) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
                for (int j = 0; j < cols; ++j)   row[j] /= sum;
            }
            break;
        }
    }
    results_[&node] = std::move(out);
}

void Executor::visit(cg::ReduceNode& node) {
    const Tensor& in = results_.at(node.input);
    int nd   = in.ndim();
    int axis = node.axis < 0 ? nd + node.axis : node.axis;
    if (axis < 0 || axis >= nd)
        throw std::runtime_error("Reduce: axis out of range");

    std::vector<int> out_shape = in.shape();
    out_shape[axis] = 1;
    Tensor out(out_shape);

    foreach_index(in.shape(), [&](int /*flat*/, const std::vector<int>& idx) {
        std::vector<int> out_idx = idx;
        out_idx[axis] = 0;
        int out_off = 0;
        for (int d = 0; d < nd; ++d) out_off += out_idx[d] * out.strides()[d];

        float val = in.at(idx);
        switch (node.op) {
            case cg::ReduceOp::Sum:  out[out_off] += val; break;
            case cg::ReduceOp::Max:
                if (idx[axis] == 0 || val > out[out_off]) out[out_off] = val; break;
            case cg::ReduceOp::Min:
                if (idx[axis] == 0 || val < out[out_off]) out[out_off] = val; break;
            case cg::ReduceOp::Mean: out[out_off] += val; break;
        }
    });

    if (node.op == cg::ReduceOp::Mean) {
        float n = (float)in.shape()[axis];
        for (auto& v : out) v /= n;
    }

    results_[&node] = std::move(out);
}

void Executor::visit(cg::TransposeNode& node) {
    const Tensor& in = results_.at(node.input);
    int nd = in.ndim();

    // build permutation: default = swap last two dims
    std::vector<int> perm = node.perm;
    if (perm.empty()) {
        perm.resize(nd);
        std::iota(perm.begin(), perm.end(), 0);
        if (nd >= 2) std::swap(perm[nd - 2], perm[nd - 1]);
    }

    std::vector<int> out_shape(nd);
    for (int i = 0; i < nd; ++i) out_shape[i] = in.shape()[perm[i]];
    Tensor out(out_shape);

    foreach_index(in.shape(), [&](int /*flat*/, const std::vector<int>& idx) {
        std::vector<int> out_idx(nd);
        for (int i = 0; i < nd; ++i) out_idx[i] = idx[perm[i]];
        out.at(out_idx) = in.at(idx);
    });

    results_[&node] = std::move(out);
}

void Executor::visit(cg::BroadcastNode& node) {
    const Tensor& in = results_.at(node.input);
    int nd = in.ndim();
    int axis = node.axis < 0 ? nd + node.axis : node.axis;
    if (axis < 0 || axis >= nd)
        throw std::runtime_error("Broadcast: axis out of range");
    if (in.shape()[axis] != 1)
        throw std::runtime_error("Broadcast: source axis must have size 1");

    std::vector<int> out_shape = in.shape();
    out_shape[axis] = node.count;
    Tensor out(out_shape);

    // outer = product of dims before axis, inner = product after
    int outer = 1, inner = 1;
    for (int d = 0;        d < axis; ++d) outer *= in.shape()[d];
    for (int d = axis + 1; d < nd;   ++d) inner *= in.shape()[d];

    for (int o = 0; o < outer; ++o)
        for (int c = 0; c < node.count; ++c)
            std::memcpy(out.data() + (o * node.count + c) * inner,
                        in.data()  + o * inner,
                        inner * sizeof(float));

    results_[&node] = std::move(out);
}

void Executor::visit(cg::ReshapeNode& node) {
    const Tensor& in = results_.at(node.input);
    int new_n = 1;
    for (int d : node.new_shape) new_n *= d;
    if (new_n != in.numel())
        throw std::runtime_error("Reshape: numel mismatch");

    Tensor out(node.new_shape);
    std::memcpy(out.data(), in.data(), in.numel() * sizeof(float));   // same data, new shape
    results_[&node] = std::move(out);
}

// im2col: [N, C, H, W] -> [N*Hout*Wout, C*kH*kW]
void Executor::visit(cg::Im2ColNode& node) {
    const Tensor& X = results_.at(node.input);
    if (X.ndim() != 4) throw std::runtime_error("im2col: expected 4D input");
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int kH = node.kH, kW = node.kW, s = node.stride, p = node.pad;
    int Hout = (H + 2 * p - kH) / s + 1;
    int Wout = (W + 2 * p - kW) / s + 1;
    int patch_size = C * kH * kW;
    int n_patches  = N * Hout * Wout;

    Tensor out({n_patches, patch_size});
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                int row = ((n * Hout) + oh) * Wout + ow;
                float* dst = out.data() + row * patch_size;
                for (int c = 0; c < C; ++c) {
                    for (int ky = 0; ky < kH; ++ky) {
                        int ih = oh * s + ky - p;
                        for (int kx = 0; kx < kW; ++kx) {
                            int iw = ow * s + kx - p;
                            int col = (c * kH + ky) * kW + kx;
                            dst[col] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                ? X[((n * C + c) * H + ih) * W + iw]
                                : 0.0f;
                        }
                    }
                }
            }
        }
    }
    results_[&node] = std::move(out);
}

// col2im: [N*Hout*Wout, C*kH*kW] -> [N, C, H, W] with overlap-add
void Executor::visit(cg::Col2ImNode& node) {
    const Tensor& In = results_.at(node.input);
    auto& s_out = node.output_shape;
    if (s_out.size() != 4) throw std::runtime_error("col2im: expected 4D output_shape");
    int N = s_out[0], C = s_out[1], H = s_out[2], W = s_out[3];
    int kH = node.kH, kW = node.kW, s = node.stride, p = node.pad;
    int Hout = (H + 2 * p - kH) / s + 1;
    int Wout = (W + 2 * p - kW) / s + 1;
    int patch_size = C * kH * kW;

    Tensor out(s_out);   // zero-initialized
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                int row = ((n * Hout) + oh) * Wout + ow;
                const float* src = In.data() + row * patch_size;
                for (int c = 0; c < C; ++c) {
                    for (int ky = 0; ky < kH; ++ky) {
                        int ih = oh * s + ky - p;
                        for (int kx = 0; kx < kW; ++kx) {
                            int iw = ow * s + kx - p;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            int col = (c * kH + ky) * kW + kx;
                            out[((n * C + c) * H + ih) * W + iw] += src[col];
                        }
                    }
                }
            }
        }
    }
    results_[&node] = std::move(out);
}

// MaxPool2D: [N,C,H,W] -> [N,C,Hout,Wout]
void Executor::visit(cg::MaxPool2DNode& node) {
    const Tensor& X = results_.at(node.input);
    if (X.ndim() != 4) throw std::runtime_error("MaxPool: expected 4D input");
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int k = node.k, s = node.stride, p = node.pad;
    int Hout = (H + 2 * p - k) / s + 1;
    int Wout = (W + 2 * p - k) / s + 1;

    Tensor out({N, C, Hout, Wout});
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < Hout; ++oh)
                for (int ow = 0; ow < Wout; ++ow) {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int ky = 0; ky < k; ++ky)
                        for (int kx = 0; kx < k; ++kx) {
                            int ih = oh * s + ky - p;
                            int iw = ow * s + kx - p;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            float v = X[((n * C + c) * H + ih) * W + iw];
                            if (v > mx) mx = v;
                        }
                    out[((n * C + c) * Hout + oh) * Wout + ow] = mx;
                }
    results_[&node] = std::move(out);
}

// UpsampleNearest: [N,C,H,W] -> [N,C,H*scale,W*scale]
void Executor::visit(cg::UpsampleNearestNode& node) {
    const Tensor& X = results_.at(node.input);
    if (X.ndim() != 4) throw std::runtime_error("Upsample: expected 4D input");
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    int s = node.scale;
    int Ho = H * s, Wo = W * s;

    Tensor out({N, C, Ho, Wo});
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < Ho; ++oh)
                for (int ow = 0; ow < Wo; ++ow)
                    out[((n * C + c) * Ho + oh) * Wo + ow] =
                        X[((n * C + c) * H + oh / s) * W + ow / s];
    results_[&node] = std::move(out);
}

// Concat along channel axis (axis=1): [N,Ca,H,W] + [N,Cb,H,W] -> [N,Ca+Cb,H,W]
void Executor::visit(cg::ConcatNode& node) {
    const Tensor& A = results_.at(node.a);
    const Tensor& B = results_.at(node.b);
    if (A.ndim() != 4 || B.ndim() != 4)
        throw std::runtime_error("Concat: expected 4D inputs");
    int N = A.shape()[0], Ca = A.shape()[1], H = A.shape()[2], W = A.shape()[3];
    int Cb = B.shape()[1];
    if (B.shape()[0] != N || B.shape()[2] != H || B.shape()[3] != W)
        throw std::runtime_error("Concat: N/H/W mismatch");

    Tensor out({N, Ca + Cb, H, W});
    int plane = H * W;
    for (int n = 0; n < N; ++n) {
        std::memcpy(out.data() + (n * (Ca + Cb))      * plane,
                    A.data()   + (n * Ca)             * plane,
                    Ca * plane * sizeof(float));
        std::memcpy(out.data() + (n * (Ca + Cb) + Ca) * plane,
                    B.data()   + (n * Cb)             * plane,
                    Cb * plane * sizeof(float));
    }
    results_[&node] = std::move(out);
}

// BatchNorm2D inference: y = gamma * (x - mean) / sqrt(var + eps) + beta
void Executor::visit(cg::BatchNorm2DNode& node) {
    const Tensor& X = results_.at(node.input);
    const Tensor& g = results_.at(node.gamma);
    const Tensor& bt = results_.at(node.beta);
    const Tensor& mu = results_.at(node.running_mean);
    const Tensor& va = results_.at(node.running_var);
    if (X.ndim() != 4) throw std::runtime_error("BatchNorm2D: expected 4D input");
    int N = X.shape()[0], C = X.shape()[1], H = X.shape()[2], W = X.shape()[3];
    if (g.numel() != C || bt.numel() != C || mu.numel() != C || va.numel() != C)
        throw std::runtime_error("BatchNorm2D: per-channel param size mismatch");

    Tensor out({N, C, H, W});
    // Precompute per-channel scale/bias so the inner loop is a single mul+add.
    std::vector<float> scale(C), shift(C);
    for (int c = 0; c < C; ++c) {
        float inv = 1.0f / std::sqrt(va[c] + node.eps);
        scale[c] = g[c] * inv;
        shift[c] = bt[c] - mu[c] * scale[c];
    }
    int plane = H * W;
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            const float* xp = X.data() + (n * C + c) * plane;
            float*       op = out.data() + (n * C + c) * plane;
            float s = scale[c], b = shift[c];
            for (int i = 0; i < plane; ++i) op[i] = xp[i] * s + b;
        }
    results_[&node] = std::move(out);
}

// Copy `value`'s tensor into `target` InputNode's host tensor. The next
// accept() pass will re-upload the updated value to the executor's cache,
// so SGD updates persist without a manual refresh().
void Executor::visit(cg::AssignNode& node) {
    const Tensor& v = results_.at(node.value);
    if (v.shape() != node.target->tensor.shape())
        throw std::runtime_error("Assign: shape mismatch with target '" + node.target->name + "'");
    node.target->tensor   = v;     // host-side write-back (next pass uploads this)
    results_[node.target] = v;     // exec.result(target) sees the new value
    results_[&node]       = v;     // assign node's "result" is the assigned value
}

const cg::Tensor& Executor::result(cg::Node* node) const {
    return results_.at(node);
}

} // namespace cg::cpu
