#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ============================================================================
// Element-wise kernels
// ============================================================================

kernel void hadamard(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device       float* C     [[buffer(2)]],
    constant     uint&  numel [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) C[index] = A[index] * B[index];
}

kernel void mat_add(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device       float* C     [[buffer(2)]],
    constant     uint&  numel [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) C[index] = A[index] + B[index];
}

kernel void relu(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = max(0.0f, in[index]);
}

kernel void sigmoid_op(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = 1.0f / (1.0f + exp(-in[index]));
}

// Indicator: 1 if x > 0 else 0
kernel void step_op(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = in[index] > 0.0f ? 1.0f : 0.0f;
}

kernel void scale(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    constant     float& s     [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = in[index] * s;
}

// ============================================================================
// N-D transpose: out[idx] = in[unpermute(idx)]
// One thread per output element.
// `perm`/`shapes` are stored as flat constant buffers because Metal kernel
// constants can't be variable-length arrays. Caller packs them into 8 slots.
// ============================================================================
constant constexpr uint MAX_DIMS = 8;

kernel void transpose_nd(
    device const float* in        [[buffer(0)]],
    device       float* out       [[buffer(1)]],
    constant     uint*  in_shape  [[buffer(2)]],   // length = ndim
    constant     uint*  out_shape [[buffer(3)]],   // length = ndim
    constant     uint*  perm      [[buffer(4)]],   // length = ndim
    constant     uint&  ndim      [[buffer(5)]],
    constant     uint&  numel     [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;

    // Decompose flat output index -> multi-index (out_idx[0..ndim))
    uint out_idx[MAX_DIMS];
    uint rem = gid;
    for (uint d = ndim; d-- > 0;) {
        out_idx[d] = rem % out_shape[d];
        rem /= out_shape[d];
    }

    // Apply inverse permutation: in_idx[perm[i]] = out_idx[i]
    uint in_idx[MAX_DIMS];
    for (uint i = 0; i < ndim; ++i) in_idx[perm[i]] = out_idx[i];

    // Re-flatten using in_shape strides (row-major)
    uint in_flat = 0;
    uint stride  = 1;
    for (uint d = ndim; d-- > 0;) {
        in_flat += in_idx[d] * stride;
        stride  *= in_shape[d];
    }

    out[gid] = in[in_flat];
}

// ============================================================================
// Broadcast: replicate input along an existing size-1 axis.
// One thread per output element; computes the input index by clamping the
// broadcast axis coordinate to 0.
// ============================================================================
kernel void broadcast_axis(
    device const float* in        [[buffer(0)]],
    device       float* out       [[buffer(1)]],
    constant     uint*  out_shape [[buffer(2)]],
    constant     uint&  ndim      [[buffer(3)]],
    constant     uint&  axis      [[buffer(4)]],
    constant     uint&  numel     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;

    // Decompose flat output index into per-dim coords
    uint idx[8];
    uint rem = gid;
    for (uint d = ndim; d-- > 0;) { idx[d] = rem % out_shape[d]; rem /= out_shape[d]; }

    // Input has size 1 on `axis` — set that coord to 0, recompute flat index
    idx[axis] = 0;
    uint in_flat = 0, stride = 1;
    for (uint d = ndim; d-- > 0;) {
        in_flat += idx[d] * stride;
        // input shape is the same as output shape but with axis=1
        stride *= (d == axis) ? 1 : out_shape[d];
    }

    out[gid] = in[in_flat];
}

// ============================================================================
// im2col: input [N, C, H, W] -> output [N*Hout*Wout, C*kH*kW]
// One thread per output element (one cell within one patch).
// ============================================================================
kernel void im2col(
    device const float* X     [[buffer(0)]],
    device       float* Y     [[buffer(1)]],
    constant     uint&  N     [[buffer(2)]],
    constant     uint&  C     [[buffer(3)]],
    constant     uint&  H     [[buffer(4)]],
    constant     uint&  W     [[buffer(5)]],
    constant     uint&  kH    [[buffer(6)]],
    constant     uint&  kW    [[buffer(7)]],
    constant     uint&  stride[[buffer(8)]],
    constant     int&   pad   [[buffer(9)]],
    constant     uint&  Hout  [[buffer(10)]],
    constant     uint&  Wout  [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint patch_size = C * kH * kW;
    uint n_patches  = N * Hout * Wout;
    uint total      = n_patches * patch_size;
    if (gid >= total) return;

    uint row = gid / patch_size;
    uint col = gid % patch_size;

    uint n  = row / (Hout * Wout);
    uint r  = row % (Hout * Wout);
    uint oh = r   / Wout;
    uint ow = r   % Wout;

    uint c  = col / (kH * kW);
    uint k  = col % (kH * kW);
    uint ky = k   / kW;
    uint kx = k   % kW;

    int ih = (int)(oh * stride + ky) - pad;
    int iw = (int)(ow * stride + kx) - pad;

    if (ih >= 0 && (uint)ih < H && iw >= 0 && (uint)iw < W)
        Y[gid] = X[((n * C + c) * H + (uint)ih) * W + (uint)iw];
    else
        Y[gid] = 0.0f;
}

// ============================================================================
// col2im: inverse of im2col with overlap-add. One thread per output pixel.
// (output element gets contributions from every patch that covers it.)
// ============================================================================
kernel void col2im(
    device const float* In    [[buffer(0)]],
    device       float* Out   [[buffer(1)]],
    constant     uint&  N     [[buffer(2)]],
    constant     uint&  C     [[buffer(3)]],
    constant     uint&  H     [[buffer(4)]],
    constant     uint&  W     [[buffer(5)]],
    constant     uint&  kH    [[buffer(6)]],
    constant     uint&  kW    [[buffer(7)]],
    constant     uint&  stride[[buffer(8)]],
    constant     int&   pad   [[buffer(9)]],
    constant     uint&  Hout  [[buffer(10)]],
    constant     uint&  Wout  [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * H * W;
    if (gid >= total) return;

    uint w = gid % W;
    uint h = (gid / W) % H;
    uint c = (gid / (W * H)) % C;
    uint n = gid / (W * H * C);

    uint patch_size = C * kH * kW;
    float sum = 0.0f;

    // For each patch (oh, ow) that covers this pixel, find its contribution
    for (uint ky = 0; ky < kH; ++ky) {
        int oh_num = (int)h + pad - (int)ky;
        if (oh_num < 0 || oh_num % (int)stride != 0) continue;
        uint oh = (uint)(oh_num / (int)stride);
        if (oh >= Hout) continue;

        for (uint kx = 0; kx < kW; ++kx) {
            int ow_num = (int)w + pad - (int)kx;
            if (ow_num < 0 || ow_num % (int)stride != 0) continue;
            uint ow = (uint)(ow_num / (int)stride);
            if (ow >= Wout) continue;

            uint row = ((n * Hout) + oh) * Wout + ow;
            uint col = (c * kH + ky) * kW + kx;
            sum += In[row * patch_size + col];
        }
    }

    Out[gid] = sum;
}

// ============================================================================
// Numerically-stable row-wise softmax over the last dim.
// Each thread handles one row: find max, exp+sum, normalize.
// Naive but fine for small last-dim sizes.
// ============================================================================
kernel void softmax_lastdim(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  rows  [[buffer(2)]],
    constant     uint&  cols  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    uint base = gid * cols;

    float mx = in[base];
    for (uint i = 1; i < cols; ++i) mx = max(mx, in[base + i]);

    float sum = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        float v = exp(in[base + i] - mx);
        out[base + i] = v;
        sum += v;
    }
    for (uint i = 0; i < cols; ++i) out[base + i] /= sum;
}

// ============================================================================
// Reduction along one axis. Naive: one thread per output element.
// Op encoding:  0=Sum  1=Mean  2=Max  3=Min   (matches cg::ReduceOp ordering)
// ============================================================================
kernel void reduce_axis(
    device const float* in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  outer [[buffer(2)]],   // product of dims before axis
    constant     uint&  axis  [[buffer(3)]],   // size of reduced dim
    constant     uint&  inner [[buffer(4)]],   // product of dims after axis
    constant     uint&  op    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = outer * inner;
    if (gid >= total) return;

    uint o = gid / inner;
    uint i = gid % inner;
    uint base = o * axis * inner + i;

    float val;
    if      (op == 0u || op == 1u) val =  0.0f;
    else if (op == 2u)             val = -INFINITY;
    else                           val =  INFINITY;

    for (uint a = 0u; a < axis; ++a) {
        float v = in[base + a * inner];
        if      (op == 0u || op == 1u) val += v;
        else if (op == 2u)             val  = max(val, v);
        else                           val  = min(val, v);
    }
    if (op == 1u) val /= float(axis);
    out[gid] = val;
}

// ============================================================================
// MaxPool2D: [N,C,H,W] -> [N,C,Hout,Wout]. One thread per output element.
// ============================================================================
kernel void maxpool2d(
    device const float* X    [[buffer(0)]],
    device       float* Y    [[buffer(1)]],
    constant     uint&  N    [[buffer(2)]],
    constant     uint&  C    [[buffer(3)]],
    constant     uint&  H    [[buffer(4)]],
    constant     uint&  W    [[buffer(5)]],
    constant     uint&  k    [[buffer(6)]],
    constant     uint&  stride[[buffer(7)]],
    constant     int&   pad  [[buffer(8)]],
    constant     uint&  Hout [[buffer(9)]],
    constant     uint&  Wout [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * Hout * Wout;
    if (gid >= total) return;

    uint ow = gid % Wout;
    uint oh = (gid / Wout) % Hout;
    uint c  = (gid / (Wout * Hout)) % C;
    uint n  = gid / (Wout * Hout * C);

    float mx = -INFINITY;
    for (uint ky = 0; ky < k; ++ky)
        for (uint kx = 0; kx < k; ++kx) {
            int ih = (int)(oh * stride + ky) - pad;
            int iw = (int)(ow * stride + kx) - pad;
            if (ih < 0 || ih >= (int)H || iw < 0 || iw >= (int)W) continue;
            float v = X[((n * C + c) * H + (uint)ih) * W + (uint)iw];
            if (v > mx) mx = v;
        }
    Y[gid] = mx;
}

// ============================================================================
// UpsampleNearest: [N,C,H,W] -> [N,C,H*s,W*s]. One thread per output element.
// ============================================================================
kernel void upsample_nearest(
    device const float* X     [[buffer(0)]],
    device       float* Y     [[buffer(1)]],
    constant     uint&  N     [[buffer(2)]],
    constant     uint&  C     [[buffer(3)]],
    constant     uint&  H     [[buffer(4)]],
    constant     uint&  W     [[buffer(5)]],
    constant     uint&  scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint Ho = H * scale;
    uint Wo = W * scale;
    uint total = N * C * Ho * Wo;
    if (gid >= total) return;

    uint ow = gid % Wo;
    uint oh = (gid / Wo) % Ho;
    uint c  = (gid / (Wo * Ho)) % C;
    uint n  = gid / (Wo * Ho * C);

    uint ih = oh / scale;
    uint iw = ow / scale;
    Y[gid] = X[((n * C + c) * H + ih) * W + iw];
}

// ============================================================================
// Concat along channel axis: [N,Ca,H,W] + [N,Cb,H,W] -> [N,Ca+Cb,H,W].
// One thread per output element; pick source by channel index.
// ============================================================================
kernel void concat_channel(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device       float* Y     [[buffer(2)]],
    constant     uint&  N     [[buffer(3)]],
    constant     uint&  Ca    [[buffer(4)]],
    constant     uint&  Cb    [[buffer(5)]],
    constant     uint&  H     [[buffer(6)]],
    constant     uint&  W     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint Cout = Ca + Cb;
    uint total = N * Cout * H * W;
    if (gid >= total) return;

    uint w = gid % W;
    uint h = (gid / W) % H;
    uint c = (gid / (W * H)) % Cout;
    uint n = gid / (W * H * Cout);

    if (c < Ca) {
        Y[gid] = A[((n * Ca + c) * H + h) * W + w];
    } else {
        uint cb = c - Ca;
        Y[gid] = B[((n * Cb + cb) * H + h) * W + w];
    }
}

// ============================================================================
// BatchNorm2D inference: y[n,c,h,w] = gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]
// One thread per output element.
// ============================================================================
kernel void batchnorm2d_infer(
    device const float* X     [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device const float* mean  [[buffer(3)]],
    device const float* var   [[buffer(4)]],
    device       float* Y     [[buffer(5)]],
    constant     uint&  N     [[buffer(6)]],
    constant     uint&  C     [[buffer(7)]],
    constant     uint&  H     [[buffer(8)]],
    constant     uint&  W     [[buffer(9)]],
    constant     float& eps   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * H * W;
    if (gid >= total) return;

    uint c = (gid / (W * H)) % C;
    float inv = 1.0f / sqrt(var[c] + eps);
    Y[gid] = gamma[c] * (X[gid] - mean[c]) * inv + beta[c];
}

// ============================================================================
// MatMul — four implementations to compare
// All shapes: A[..., M, K] @ B[..., K, N] = C[..., M, N], batched over leading dims
// ============================================================================

// (1) Naive: one thread per output element. Each thread reads K elements from
//     each of A and B from device memory, no reuse.
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant     uint&  M [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.x, col = gid.y, batch = gid.z;
    if (row >= M || col >= N) return;

    uint a_off = batch * M * K + row * K;
    uint b_off = batch * K * N + col;

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k)
        sum += A[a_off + k] * B[b_off + k * N];

    C[batch * M * N + row * N + col] = sum;
}

// (2) Tiled: each threadgroup computes a 32x32 output block. Cooperatively
//     loads tiles of A and B into threadgroup (shared) memory so each value
//     is reused 32x by sibling threads in the group instead of being reloaded
//     from device memory.
constant constexpr uint TM = 32;
constant constexpr uint TN = 32;
constant constexpr uint TK = 32;

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant     uint&  M [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 gid  [[thread_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]])
{
    threadgroup float Atile[TM][TK];
    threadgroup float Btile[TK][TN];

    uint row   = gid.x;
    uint col   = gid.y;
    uint batch = gid.z;
    uint lr    = tid.x;
    uint lc    = tid.y;

    uint a_batch = batch * M * K;
    uint b_batch = batch * K * N;

    float sum = 0.0f;
    for (uint t = 0; t < K; t += TK) {
        uint ak = t + lc;
        uint bk = t + lr;

        Atile[lr][lc] = (row < M && ak < K) ? A[a_batch + row * K + ak] : 0.0f;
        Btile[lr][lc] = (bk < K && col < N) ? B[b_batch + bk * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TK; ++k)
            sum += Atile[lr][k] * Btile[k][lc];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[batch * M * N + row * N + col] = sum;
}

// (4) SIMD + threadgroup memory tiled: each threadgroup computes a 32x32
//     output, decomposed into a 4x4 grid of 8x8 SIMD-group sub-tiles. Per
//     K-iteration the threadgroup cooperatively loads a 32x16 slab of A
//     and 16x32 slab of B into threadgroup memory; each of the 16 SIMD
//     groups then runs `simdgroup_multiply_accumulate` against those
//     shared tiles 2x (TILE_K / 8 = 2). Each shared element is reused 4x.
constant constexpr uint OM = 32;
constant constexpr uint ON = 32;
constant constexpr uint OK = 16;

kernel void matmul_simd_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant     uint&  M [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  tidx [[thread_index_in_threadgroup]])
{
    threadgroup float A_tile[OM][OK];   // 32 x 16
    threadgroup float B_tile[OK][ON];   // 16 x 32

    uint tg_row = tgid.x * OM;
    uint tg_col = tgid.y * ON;
    uint batch  = tgid.z;

    // 4x4 grid of SIMD groups inside the threadgroup
    uint sg_row = sgid / 4;
    uint sg_col = sgid % 4;

    // Cooperative load layout: 512 threads load 32*16 (A) + 16*32 (B) = 1024 floats,
    // 2 per thread. Use one thread per element by splitting the linear thread index.
    uint a_row = tidx / OK;     // 0..31
    uint a_col = tidx % OK;     // 0..15
    uint b_row = tidx / ON;     // 0..15
    uint b_col = tidx % ON;     // 0..31

    simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);

    uint a_batch = batch * M * K;
    uint b_batch = batch * K * N;

    for (uint kt = 0; kt < K; kt += OK) {
        // Load A_tile (32 x 16) cooperatively
        uint ar = tg_row + a_row;
        uint ac = kt     + a_col;
        A_tile[a_row][a_col] = (ar < M && ac < K) ? A[a_batch + ar * K + ac] : 0.0f;

        // Load B_tile (16 x 32) cooperatively
        uint br = kt     + b_row;
        uint bc = tg_col + b_col;
        B_tile[b_row][b_col] = (br < K && bc < N) ? B[b_batch + br * N + bc] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SIMD group does TILE_K/8 = 2 mma's against shared tiles
        simdgroup_matrix<float, 8, 8> a, b;
        for (uint kk = 0; kk < OK; kk += 8) {
            simdgroup_load(a, &A_tile[sg_row * 8][kk], OK);
            simdgroup_load(b, &B_tile[kk][sg_col * 8], ON);
            simdgroup_multiply_accumulate(c, a, b, c);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint c_row = tg_row + sg_row * 8;
    uint c_col = tg_col + sg_col * 8;
    if (c_row < M && c_col < N)
        simdgroup_store(c, C + batch * M * N + c_row * N + c_col, N);
}

// ============================================================================
// f16 variants of the elementwise / structural kernels. When ComputeGraph is
// declared with Precision::F16, all device buffers are half-sized and the
// executor dispatches the *_f16 names below. Kernel bodies are mechanically
// identical to the float versions; arithmetic is done in `half` (Metal's
// math overloads handle the dispatch). BatchNorm computes in float for
// numerical stability (sqrt of small running_var).
// ============================================================================

kernel void mat_add_f16(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  C     [[buffer(2)]],
    constant     uint&  numel [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) C[index] = A[index] + B[index];
}

kernel void hadamard_f16(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  C     [[buffer(2)]],
    constant     uint&  numel [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) C[index] = A[index] * B[index];
}

kernel void relu_f16(
    device const half*  in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = max((half)0.0, in[index]);
}

kernel void step_op_f16(
    device const half*  in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = in[index] > (half)0.0 ? (half)1.0 : (half)0.0;
}

kernel void sigmoid_op_f16(
    device const half*  in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) {
        // Compute in float for stability; sigmoid(-large) underflows in half.
        float x = (float)in[index];
        out[index] = (half)(1.0f / (1.0f + exp(-x)));
    }
}

kernel void scale_f16(
    device const half*  in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    constant     float& s     [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = (half)((float)in[index] * s);
}

kernel void transpose_nd_f16(
    device const half*  in        [[buffer(0)]],
    device       half*  out       [[buffer(1)]],
    constant     uint*  in_shape  [[buffer(2)]],
    constant     uint*  out_shape [[buffer(3)]],
    constant     uint*  perm      [[buffer(4)]],
    constant     uint&  ndim      [[buffer(5)]],
    constant     uint&  numel     [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    uint out_idx[MAX_DIMS];
    uint rem = gid;
    for (uint d = ndim; d-- > 0;) { out_idx[d] = rem % out_shape[d]; rem /= out_shape[d]; }
    uint in_idx[MAX_DIMS];
    for (uint i = 0; i < ndim; ++i) in_idx[perm[i]] = out_idx[i];
    uint in_flat = 0, stride = 1;
    for (uint d = ndim; d-- > 0;) { in_flat += in_idx[d] * stride; stride *= in_shape[d]; }
    out[gid] = in[in_flat];
}

kernel void broadcast_axis_f16(
    device const half*  in        [[buffer(0)]],
    device       half*  out       [[buffer(1)]],
    constant     uint*  out_shape [[buffer(2)]],
    constant     uint&  ndim      [[buffer(3)]],
    constant     uint&  axis      [[buffer(4)]],
    constant     uint&  numel     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    uint idx[8];
    uint rem = gid;
    for (uint d = ndim; d-- > 0;) { idx[d] = rem % out_shape[d]; rem /= out_shape[d]; }
    idx[axis] = 0;
    uint in_flat = 0, stride = 1;
    for (uint d = ndim; d-- > 0;) {
        in_flat += idx[d] * stride;
        stride *= (d == axis) ? 1 : out_shape[d];
    }
    out[gid] = in[in_flat];
}

kernel void im2col_f16(
    device const half*  X     [[buffer(0)]],
    device       half*  Y     [[buffer(1)]],
    constant     uint&  N     [[buffer(2)]],
    constant     uint&  C     [[buffer(3)]],
    constant     uint&  H     [[buffer(4)]],
    constant     uint&  W     [[buffer(5)]],
    constant     uint&  kH    [[buffer(6)]],
    constant     uint&  kW    [[buffer(7)]],
    constant     uint&  stride[[buffer(8)]],
    constant     int&   pad   [[buffer(9)]],
    constant     uint&  Hout  [[buffer(10)]],
    constant     uint&  Wout  [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    uint patch_size = C * kH * kW;
    uint n_patches  = N * Hout * Wout;
    uint total      = n_patches * patch_size;
    if (gid >= total) return;
    uint row = gid / patch_size, col = gid % patch_size;
    uint n  = row / (Hout * Wout);
    uint r  = row % (Hout * Wout);
    uint oh = r   / Wout, ow = r % Wout;
    uint c  = col / (kH * kW);
    uint k  = col % (kH * kW);
    uint ky = k   / kW, kx = k % kW;
    int ih = (int)(oh * stride + ky) - pad;
    int iw = (int)(ow * stride + kx) - pad;
    Y[gid] = (ih >= 0 && (uint)ih < H && iw >= 0 && (uint)iw < W)
        ? X[((n * C + c) * H + (uint)ih) * W + (uint)iw]
        : (half)0.0;
}

kernel void maxpool2d_f16(
    device const half*  X    [[buffer(0)]],
    device       half*  Y    [[buffer(1)]],
    constant     uint&  N    [[buffer(2)]],
    constant     uint&  C    [[buffer(3)]],
    constant     uint&  H    [[buffer(4)]],
    constant     uint&  W    [[buffer(5)]],
    constant     uint&  k    [[buffer(6)]],
    constant     uint&  stride[[buffer(7)]],
    constant     int&   pad  [[buffer(8)]],
    constant     uint&  Hout [[buffer(9)]],
    constant     uint&  Wout [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * Hout * Wout;
    if (gid >= total) return;
    uint ow = gid % Wout;
    uint oh = (gid / Wout) % Hout;
    uint c  = (gid / (Wout * Hout)) % C;
    uint n  = gid / (Wout * Hout * C);
    half mx = -HALF_MAX;
    for (uint ky = 0; ky < k; ++ky)
        for (uint kx = 0; kx < k; ++kx) {
            int ih = (int)(oh * stride + ky) - pad;
            int iw = (int)(ow * stride + kx) - pad;
            if (ih < 0 || ih >= (int)H || iw < 0 || iw >= (int)W) continue;
            half v = X[((n * C + c) * H + (uint)ih) * W + (uint)iw];
            if (v > mx) mx = v;
        }
    Y[gid] = mx;
}

kernel void upsample_nearest_f16(
    device const half*  X     [[buffer(0)]],
    device       half*  Y     [[buffer(1)]],
    constant     uint&  N     [[buffer(2)]],
    constant     uint&  C     [[buffer(3)]],
    constant     uint&  H     [[buffer(4)]],
    constant     uint&  W     [[buffer(5)]],
    constant     uint&  scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint Ho = H * scale, Wo = W * scale;
    uint total = N * C * Ho * Wo;
    if (gid >= total) return;
    uint ow = gid % Wo;
    uint oh = (gid / Wo) % Ho;
    uint c  = (gid / (Wo * Ho)) % C;
    uint n  = gid / (Wo * Ho * C);
    Y[gid] = X[((n * C + c) * H + oh / scale) * W + ow / scale];
}

kernel void concat_channel_f16(
    device const half*  A     [[buffer(0)]],
    device const half*  B     [[buffer(1)]],
    device       half*  Y     [[buffer(2)]],
    constant     uint&  N     [[buffer(3)]],
    constant     uint&  Ca    [[buffer(4)]],
    constant     uint&  Cb    [[buffer(5)]],
    constant     uint&  H     [[buffer(6)]],
    constant     uint&  W     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint Cout = Ca + Cb;
    uint total = N * Cout * H * W;
    if (gid >= total) return;
    uint w = gid % W;
    uint h = (gid / W) % H;
    uint c = (gid / (W * H)) % Cout;
    uint n = gid / (W * H * Cout);
    if (c < Ca)  Y[gid] = A[((n * Ca + c)      * H + h) * W + w];
    else         Y[gid] = B[((n * Cb + c - Ca) * H + h) * W + w];
}

kernel void batchnorm2d_infer_f16(
    device const half*  X     [[buffer(0)]],
    device const half*  gamma [[buffer(1)]],
    device const half*  beta  [[buffer(2)]],
    device const half*  mean  [[buffer(3)]],
    device const half*  var   [[buffer(4)]],
    device       half*  Y     [[buffer(5)]],
    constant     uint&  N     [[buffer(6)]],
    constant     uint&  C     [[buffer(7)]],
    constant     uint&  H     [[buffer(8)]],
    constant     uint&  W     [[buffer(9)]],
    constant     float& eps   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = N * C * H * W;
    if (gid >= total) return;
    uint c = (gid / (W * H)) % C;
    // Compute in float — sqrt(small + eps) is precision-sensitive.
    float inv = 1.0f / sqrt((float)var[c] + eps);
    Y[gid] = (half)((float)gamma[c] * ((float)X[gid] - (float)mean[c]) * inv + (float)beta[c]);
}

// ============================================================================
// f32 <-> f16 elementwise conversion. Used to feed MPP matmul inputs (which
// operate on `half`) while keeping the rest of the lib's storage at float.
// ============================================================================
kernel void f32_to_f16(
    device const float* in    [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = (half)in[index];
}

kernel void f16_to_f32(
    device const half*  in    [[buffer(0)]],
    device       float* out   [[buffer(1)]],
    constant     uint&  numel [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < numel) out[index] = (float)in[index];
}

// ============================================================================
// MetalPerformancePrimitives matmul2d (Metal 4). Inputs are half tensors,
// output is float — mixed precision with f32 accumulation. Tile is 64 (M) x
// 32 (N) per threadgroup, 4 SIMD groups cooperate. Edge tiles handled by MPP.
//
// Dispatch:
//   grid        = MTL::Size( (N + 31) / 32, (M + 63) / 64, 1 )
//   threadgroup = MTL::Size( 128, 1, 1 )                    // 4 SIMD-groups × 32 lanes
// ============================================================================
kernel void matmul_mpp(
    device       half*  A_buf [[buffer(0)]],   // [M, K] row-major
    device       half*  B_buf [[buffer(1)]],   // [K, N] row-major
    device       half*  C_buf [[buffer(2)]],   // [M, N] row-major
    constant     uint&  M     [[buffer(3)]],
    constant     uint&  N     [[buffer(4)]],
    constant     uint&  K     [[buffer(5)]],
    uint2        tgid  [[threadgroup_position_in_grid]])
{
    // MPP dextents convention is (inner_dim, outer_dim) = (cols, rows) for
    // row-major storage. Inline tensors wrap the raw device pointer directly.
    // Output is half (the half×half→half combo is broadly supported). Caller
    // can convert back to float on the host side if needed.
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        A_buf, dextents<int32_t, 2>((int32_t)K, (int32_t)M));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        B_buf, dextents<int32_t, 2>((int32_t)N, (int32_t)K));
    auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        C_buf, dextents<int32_t, 2>((int32_t)N, (int32_t)M));

    constexpr auto desc = matmul2d_descriptor(
        /*tile_M=*/64, /*tile_N=*/32,
        /*K=*/static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/false,
        /*relaxed_precision=*/false);
    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0,             tgid.y * 64);
    auto mB = B.slice(tgid.x * 32,   0);
    auto mC = C.slice(tgid.x * 32,   tgid.y * 64);
    op.run(mA, mB, mC);
}

// (3) SIMD-group matrix: one SIMD group (32 threads) collaboratively computes
//     an 8x8 output tile using Apple's dedicated matrix hardware
//     (`simdgroup_matrix`). Each `simdgroup_multiply_accumulate` does a full
//     8x8x8 mma in a single hardware instruction.
kernel void matmul_simd(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    constant     uint&  M [[buffer(3)]],
    constant     uint&  N [[buffer(4)]],
    constant     uint&  K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    uint row_base = tgid.x * 8;
    uint col_base = tgid.y * 8;
    uint batch    = tgid.z;
    if (row_base >= M || col_base >= N) return;

    simdgroup_matrix<float, 8, 8> a, b, c;
    c = simdgroup_matrix<float, 8, 8>(0.0f);

    uint a_off = batch * M * K + row_base * K;
    uint b_off = batch * K * N + col_base;

    for (uint k = 0; k < K; k += 8) {
        simdgroup_load(a, A + a_off + k,     K);
        simdgroup_load(b, B + b_off + k * N, N);
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    simdgroup_store(c, C + batch * M * N + row_base * N + col_base, N);
}
