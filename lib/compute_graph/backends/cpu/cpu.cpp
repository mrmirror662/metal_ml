#include "cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
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

const cg::Tensor& Executor::result(cg::Node* node) const {
    return results_.at(node);
}

} // namespace cg::cpu
