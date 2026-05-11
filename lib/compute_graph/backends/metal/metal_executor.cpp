// Metal-CPP implementation macros — must appear in exactly one translation unit
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"

#include "metal_executor.h"
#include "shader_path.h"

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cg::metal {

// RAII handle for an MTL::Buffer. Custom-deleter shared_ptr means:
//   * release() runs exactly once when the last owner drops
//   * copy = retain (extra owner), no manual retain/release pairs to balance
//   * reshape's "alias same buffer under two nodes" becomes shared ownership,
//     so dangling-after-realloc is structurally impossible
using MtlBufferPtr = std::shared_ptr<MTL::Buffer>;

static MtlBufferPtr wrap_buffer(MTL::Buffer* raw) {
    return MtlBufferPtr(raw, [](MTL::Buffer* b) { if (b) b->release(); });
}

// --- Device-side tensor: a buffer on the GPU plus its logical shape ---
struct MetalBuffer {
    MtlBufferPtr     buf;
    std::vector<int> shape;

    int numel() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
    }
};

struct MetalContext {
    MTL::Device*       device    = nullptr;
    MTL::CommandQueue* queue     = nullptr;
    MTL::Library*      library   = nullptr;
    cg::Precision      precision = cg::Precision::F32;

    std::unordered_map<std::string, MTL::ComputePipelineState*> psos;

    // Per-element size in bytes for device buffers under the current precision.
    size_t dtype_bytes() const { return precision == cg::Precision::F16 ? 2 : 4; }

    // Persists across runs of the same graph so subsequent runs reuse buffers.
    std::unordered_map<cg::Node*, MetalBuffer>        device_results;
    mutable std::unordered_map<cg::Node*, cg::Tensor> host_cache;
    mutable MTL::CommandBuffer*           pending = nullptr;
    mutable MTL::ComputeCommandEncoder*   encoder = nullptr;   // kept open across dispatches

    // Scratch f16 buffers used by the MPP matmul path. Three slots per
    // MatMulNode: 0=A_half, 1=B_half, 2=C_half. Persists across runs to
    // avoid realloc churn. Slot is encoded into the key as ptr-bits | slot
    // (slot < 4 always; node addresses are >=8-aligned so the low bits are free).
    mutable std::unordered_map<uintptr_t, MtlBufferPtr> f16_scratch;

    ~MetalContext() {
        if (encoder) encoder->endEncoding();
        if (pending) { pending->commit(); pending->waitUntilCompleted(); }
        // device_results' MtlBufferPtrs release themselves as the map clears.
        for (auto& [_, pso] : psos) pso->release();
        if (library) library->release();
        if (queue)   queue->release();
        if (device)  device->release();
    }

    // Ensure an open compute encoder on the current command buffer.
    // Reusing one encoder across many dispatches avoids per-op encoding +
    // teardown cost; Metal sequences dispatches in submission order and
    // tracks the implicit RAW dependencies between them automatically.
    //
    // computeCommandEncoder() returns an autoreleased object; we retain so
    // the pointer stays valid across visit() boundaries (autoreleasepool
    // would otherwise reclaim it between calls).
    MTL::ComputeCommandEncoder* get_encoder() const {
        if (!pending) pending = queue->commandBuffer();
        if (!encoder) {
            encoder = pending->computeCommandEncoder();
            encoder->retain();
        }
        return encoder;
    }

    void end_encoder() const {
        if (encoder) {
            encoder->endEncoding();
            encoder->release();
            encoder = nullptr;
        }
    }

    void flush() const {
        end_encoder();
        if (!pending) return;
        pending->commit();
        pending->waitUntilCompleted();
        pending = nullptr;
    }
};

// ---------- Helpers (free functions, only visible in this TU) ----------

struct ConstBytes { const void* data; size_t size; };

// Get an existing output buffer for `node`, or allocate a fresh one with
// the requested shape. Returns a reference into device_results so the
// caller can read .buf and dispatch writes into it.
static MetalBuffer& ensure_output(MetalContext& ctx, cg::Node* node, std::vector<int> shape) {
    int numel = 1;
    for (int d : shape) {
        if (d < 0) throw std::runtime_error("Metal: negative dim in output shape");
        numel *= d;
    }
    if (numel == 0) numel = 1;

    size_t bytes = (size_t)numel * ctx.dtype_bytes();

    auto it = ctx.device_results.find(node);
    if (it != ctx.device_results.end()) {
        MetalBuffer& mb = it->second;
        if (mb.shape == shape) return mb;
        // Shape changed (e.g. variable batch size) — drop the old buffer
        // (any aliased reshape views drop with it via shared_ptr) and realloc.
        mb.shape = std::move(shape);
        mb.buf = wrap_buffer(ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared));
        ctx.host_cache.erase(node);
        return mb;
    }
    MetalBuffer mb;
    mb.shape = std::move(shape);
    mb.buf = wrap_buffer(ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    auto [iter, _] = ctx.device_results.emplace(node, std::move(mb));
    return iter->second;
}

static cg::Tensor from_device(const MetalBuffer& mb, cg::Precision precision) {
    cg::Tensor t(mb.shape);
    if (precision == cg::Precision::F16) {
        const uint16_t* in = reinterpret_cast<const uint16_t*>(mb.buf->contents());
        float* out = t.data();
        for (int i = 0; i < mb.numel(); ++i) {
            _Float16 h;
            std::memcpy(&h, &in[i], sizeof(uint16_t));
            out[i] = (float)h;
        }
    } else {
        std::memcpy(t.data(), mb.buf->contents(), (size_t)mb.numel() * sizeof(float));
    }
    return t;
}

// Get-or-allocate a half-precision scratch buffer of `numel` halves for
// (node, slot). Returns the raw MTL::Buffer pointer for binding.
static MTL::Buffer* ensure_f16_scratch(MetalContext& ctx, cg::Node* node, int slot, int numel) {
    uintptr_t key = reinterpret_cast<uintptr_t>(node) | (uintptr_t)slot;
    auto it = ctx.f16_scratch.find(key);
    size_t bytes = (size_t)numel * sizeof(uint16_t);   // half = 16 bits
    if (it != ctx.f16_scratch.end()) {
        if (it->second->length() >= bytes) return it->second.get();
        // grew — realloc
        it->second = wrap_buffer(ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared));
        return it->second.get();
    }
    auto buf = wrap_buffer(ctx.device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    auto* raw = buf.get();
    ctx.f16_scratch.emplace(key, std::move(buf));
    return raw;
}

static MTL::ComputePipelineState* make_pso(MTL::Device* device, MTL::Library* library, const char* name) {
    auto* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
    auto* fn      = library->newFunction(fn_name);
    if (!fn) throw std::runtime_error(std::string("Metal: kernel not found: ") + name);

    NS::Error* err = nullptr;
    auto* pso = device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso)
        throw std::runtime_error(std::string("Metal: PSO failed: ")
                                 + err->localizedDescription()->utf8String());
    return pso;
}

static MTL::ComputePipelineState* lookup_pso(MetalContext& ctx, const std::string& kernel) {
    auto it = ctx.psos.find(kernel);
    if (it == ctx.psos.end())
        throw std::runtime_error("Metal: kernel not registered: " + kernel);
    return it->second;
}

// Encode one kernel into the persistent compute encoder. No commit/wait.
// All consecutive dispatches in a graph traversal share a single encoder,
// which Metal auto-serializes and dependency-tracks between threadgroups.
static void dispatch(
    MetalContext&                                              ctx,
    MTL::ComputePipelineState*                                 pso,
    const std::function<void(MTL::ComputeCommandEncoder*)>&    bind,
    MTL::Size                                                  grid,
    MTL::Size                                                  threadgroup)
{
    auto* enc = ctx.get_encoder();
    enc->setComputePipelineState(pso);
    bind(enc);
    enc->dispatchThreadgroups(grid, threadgroup);
}

// Common case: kernel with N inputs, 1 output, numel constant, optional extras.
// Slot order in the kernel: inputs, output, numel, extras...
static void dispatch_elementwise(
    MetalContext&                       ctx,
    const std::string&                  kernel,
    MetalBuffer&                        out,
    const std::vector<MTL::Buffer*>&    inputs,
    const std::vector<ConstBytes>&      extras = {})
{
    uint32_t numel = static_cast<uint32_t>(out.numel());

    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((numel + 255) / 256, 1, 1);

    dispatch(ctx, lookup_pso(ctx, kernel), [&](MTL::ComputeCommandEncoder* enc) {
        int slot = 0;
        for (auto* in : inputs)  enc->setBuffer(in, 0, slot++);
        enc->setBuffer(out.buf.get(), 0, slot++);
        enc->setBytes(&numel, sizeof(uint32_t), slot++);
        for (auto& c : extras)   enc->setBytes(c.data, c.size, slot++);
    }, grid, threadgroup);
}

// ---------- Executor ----------

Executor::Executor(MatMul matmul) : ctx_(std::make_unique<MetalContext>()), matmul_impl_(matmul) {
    ctx_->device = MTL::CreateSystemDefaultDevice();
    if (!ctx_->device) throw std::runtime_error("Metal: no GPU device found");
    ctx_->queue = ctx_->device->newCommandQueue();

    auto* path = NS::String::string(CG_METAL_SHADER_LIB, NS::UTF8StringEncoding);
    auto* url  = NS::URL::fileURLWithPath(path);
    NS::Error* err = nullptr;
    ctx_->library = ctx_->device->newLibrary(url, &err);
    if (!ctx_->library)
        throw std::runtime_error(std::string("Metal: failed to load shader lib: ")
                                 + err->localizedDescription()->utf8String());

    auto reg = [&](const char* name) {
        ctx_->psos[name] = make_pso(ctx_->device, ctx_->library, name);
    };
    reg("mat_add");
    reg("hadamard");
    reg("relu");
    reg("step_op");
    reg("sigmoid_op");
    reg("softmax_lastdim");
    reg("broadcast_axis");
    reg("scale");
    reg("reduce_axis");
    reg("transpose_nd");
    reg("im2col");
    reg("col2im");
    reg("maxpool2d");
    reg("upsample_nearest");
    reg("concat_channel");
    reg("batchnorm2d_infer");
    reg("matmul_naive");
    reg("matmul_tiled");
    reg("matmul_simd");
    reg("matmul_simd_tiled");
    reg("matmul_mpp");
    reg("f32_to_f16");
    reg("f16_to_f32");
    // f16 variants for the subset of ops that participate in seg inference.
    // Other ops (reduce/softmax/scale/hadamard/col2im/matmuls) only have
    // f32 variants today — graphs that use them at Precision::F16 will fail
    // at kernel lookup time, surfaced as a clear runtime error.
    reg("mat_add_f16");
    reg("hadamard_f16");
    reg("relu_f16");
    reg("step_op_f16");
    reg("sigmoid_op_f16");
    reg("scale_f16");
    reg("transpose_nd_f16");
    reg("broadcast_axis_f16");
    reg("im2col_f16");
    reg("maxpool2d_f16");
    reg("upsample_nearest_f16");
    reg("concat_channel_f16");
    reg("batchnorm2d_infer_f16");
}

Executor::~Executor() = default;

void Executor::on_precision(cg::Precision p) {
    // F16 requires MPP matmul: only matmul_mpp reads half buffers; the other
    // matmul kernels (Tiled/SIMD/SimdTiled/Naive) operate on float storage.
    // Silently upgrade so consumers can declare precision on the graph and
    // not have to remember to also pick a compatible matmul variant.
    ctx_->precision = p;
    if (p == cg::Precision::F16 && matmul_impl_ != MatMul::MPP) {
        matmul_impl_ = MatMul::MPP;
    }
}

const std::string Executor::device_name() const {
    return ctx_->device->name()->utf8String();
}

// Helper: append _f16 suffix to a kernel name when the current precision is F16.
static inline std::string K(MetalContext& ctx, const char* base) {
    return ctx.precision == cg::Precision::F16
        ? std::string(base) + "_f16"
        : std::string(base);
}

// ---------- Visit methods ----------

// Cast a host f32 tensor into the executor's device dtype and copy into `dst`.
// dst must be sized = numel * dtype_bytes(precision).
static void upload_tensor(const cg::Tensor& src, void* dst, cg::Precision p) {
    int numel = src.numel();
    if (p == cg::Precision::F16) {
        // CPU-side f32 -> f16 cast. Done once per constant (or per pass for
        // non-constant inputs like the image); avoids needing an extra GPU
        // dispatch + sync to do the conversion.
        uint16_t* out = reinterpret_cast<uint16_t*>(dst);
        const float* in = src.data();
        for (int i = 0; i < numel; ++i) {
            // _Float16 is the standard arm64 half type; reinterpret to bits.
            _Float16 h = (_Float16)in[i];
            std::memcpy(&out[i], &h, sizeof(uint16_t));
        }
    } else {
        std::memcpy(dst, src.data(), (size_t)numel * sizeof(float));
    }
}

void Executor::visit(cg::InputNode& node) {
    size_t bytes = (size_t)node.tensor.numel() * ctx_->dtype_bytes();
    auto it = ctx_->device_results.find(&node);
    if (it != ctx_->device_results.end()) {
        MetalBuffer& mb = it->second;
        // Reupload only valid if shape (and therefore size) is unchanged.
        if (mb.shape != node.tensor.shape()) {
            mb.shape = node.tensor.shape();
            mb.buf   = wrap_buffer(ctx_->device->newBuffer(bytes, MTL::ResourceStorageModeShared));
            upload_tensor(node.tensor, mb.buf->contents(), ctx_->precision);
            ctx_->host_cache.erase(&node);
            return;
        }
        // Skip the host->device copy when the node is marked constant.
        // The first visit (handled by the "not found in map" branch below)
        // populated the device buffer at construction, and the value won't
        // change for the rest of the session — saves a full weights copy
        // per pass, which for a ResNet U-Net is ~98 MB of memcpy / inference.
        if (node.is_constant) return;
        upload_tensor(node.tensor, mb.buf->contents(), ctx_->precision);
        return;
    }
    MetalBuffer mb;
    mb.shape = node.tensor.shape();
    mb.buf = wrap_buffer(ctx_->device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    upload_tensor(node.tensor, mb.buf->contents(), ctx_->precision);
    ctx_->device_results.emplace(&node, std::move(mb));
}

void Executor::visit(cg::MatAddNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.lhs);
    const MetalBuffer& B = ctx_->device_results.at(node.rhs);
    if (A.shape != B.shape)
        throw std::runtime_error("Metal MatAdd: shape mismatch");

    MetalBuffer& C = ensure_output(*ctx_, &node, A.shape);
    dispatch_elementwise(*ctx_, K(*ctx_, "mat_add"), C, {A.buf.get(), B.buf.get()});
}

void Executor::visit(cg::HadamardNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.lhs);
    const MetalBuffer& B = ctx_->device_results.at(node.rhs);
    if (A.shape != B.shape)
        throw std::runtime_error("Metal Hadamard: shape mismatch");

    MetalBuffer& C = ensure_output(*ctx_, &node, A.shape);
    dispatch_elementwise(*ctx_, K(*ctx_, "hadamard"), C, {A.buf.get(), B.buf.get()});
}

void Executor::visit(cg::MapNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    MetalBuffer& out = ensure_output(*ctx_, &node, in.shape);

    switch (node.op) {
        case cg::MapOp::ReLU:
            dispatch_elementwise(*ctx_, K(*ctx_, "relu"), out, {in.buf.get()});
            break;
        case cg::MapOp::Step:
            dispatch_elementwise(*ctx_, K(*ctx_, "step_op"), out, {in.buf.get()});
            break;
        case cg::MapOp::Sigmoid:
            dispatch_elementwise(*ctx_, K(*ctx_, "sigmoid_op"), out, {in.buf.get()});
            break;
        case cg::MapOp::Softmax: {
            int last = in.shape.back();
            uint32_t rows = static_cast<uint32_t>(in.numel() / last);
            uint32_t cols = static_cast<uint32_t>(last);

            MTL::Size threadgroup = MTL::Size(256, 1, 1);
            MTL::Size grid        = MTL::Size((rows + 255) / 256, 1, 1);

            dispatch(*ctx_, lookup_pso(*ctx_, "softmax_lastdim"), [&](MTL::ComputeCommandEncoder* enc) {
                enc->setBuffer(in.buf.get(),  0, 0);
                enc->setBuffer(out.buf.get(), 0, 1);
                enc->setBytes(&rows, sizeof(uint32_t), 2);
                enc->setBytes(&cols, sizeof(uint32_t), 3);
            }, grid, threadgroup);
            break;
        }
    }
}

void Executor::visit(cg::ScaleNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    MetalBuffer& out = ensure_output(*ctx_, &node, in.shape);
    dispatch_elementwise(*ctx_, K(*ctx_, "scale"), out, {in.buf.get()}, {{&node.scalar, sizeof(float)}});
}

void Executor::visit(cg::MatMulNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.lhs);
    const MetalBuffer& B = ctx_->device_results.at(node.rhs);

    int nd = (int)A.shape.size();
    if (nd < 2 || (int)B.shape.size() < 2)
        throw std::runtime_error("Metal MatMul: inputs must be at least 2D");
    if (A.shape[nd - 1] != B.shape[B.shape.size() - 2])
        throw std::runtime_error("Metal MatMul: inner dimensions mismatch");

    uint32_t M = A.shape[nd - 2];
    uint32_t K = A.shape[nd - 1];
    uint32_t N = B.shape[B.shape.size() - 1];

    std::vector<int> batch_shape(A.shape.begin(), A.shape.begin() + nd - 2);
    uint32_t batch = 1;
    for (int d : batch_shape) batch *= d;

    std::vector<int> out_shape = batch_shape;
    out_shape.push_back((int)M);
    out_shape.push_back((int)N);

    MetalBuffer& C = ensure_output(*ctx_, &node, std::move(out_shape));

    // ---- MPP path: Apple's matmul2d primitive (f16 inputs + f16 output).
    //
    // Two storage modes:
    //   F32 graphs: A, B, C live as float on the device. We allocate per-node
    //               half-precision scratch buffers, cast f32->f16 into them,
    //               dispatch matmul_mpp, then cast f16->f32 back into C.
    //   F16 graphs: A, B, C are *already* half on the device — skip both
    //               conversions and dispatch matmul_mpp directly against the
    //               existing buffers. Saves ~3 dispatches per matmul plus the
    //               scratch allocations.
    if (matmul_impl_ == MatMul::MPP) {
        MTL::Buffer *Ah = nullptr, *Bh = nullptr, *Ch = nullptr;
        bool half_storage = (ctx_->precision == cg::Precision::F16);

        if (half_storage) {
            // Buffers already hold halves.
            Ah = A.buf.get(); Bh = B.buf.get(); Ch = C.buf.get();
        } else {
            Ah = ensure_f16_scratch(*ctx_, &node, 0, (int)(M * K * batch));
            Bh = ensure_f16_scratch(*ctx_, &node, 1, (int)(K * N * batch));
            Ch = ensure_f16_scratch(*ctx_, &node, 2, (int)(M * N * batch));

            // f32 -> f16 for A and B (full batched buffer)
            auto cast_in = [&](MTL::Buffer* src, MTL::Buffer* dst, uint32_t numel) {
                MTL::Size tg = MTL::Size(256, 1, 1);
                MTL::Size g  = MTL::Size((numel + 255) / 256, 1, 1);
                dispatch(*ctx_, lookup_pso(*ctx_, "f32_to_f16"), [&](MTL::ComputeCommandEncoder* enc) {
                    enc->setBuffer(src, 0, 0);
                    enc->setBuffer(dst, 0, 1);
                    enc->setBytes(&numel, sizeof(uint32_t), 2);
                }, g, tg);
            };
            cast_in(A.buf.get(), Ah, M * K * batch);
            cast_in(B.buf.get(), Bh, K * N * batch);
        }

        // Per-batch MPP matmul. M, N, K are the same for each batch element;
        // we offset into the buffers per iteration via setBufferOffset.
        MTL::Size tg_mm = MTL::Size(128, 1, 1);                          // 4 simdgroups × 32
        MTL::Size g_mm  = MTL::Size((N + 31) / 32, (M + 63) / 64, 1);    // (N tiles, M tiles)

        for (uint32_t b = 0; b < batch; ++b) {
            size_t off_A = (size_t)b * M * K * sizeof(uint16_t);
            size_t off_B = (size_t)b * K * N * sizeof(uint16_t);
            size_t off_C = (size_t)b * M * N * sizeof(uint16_t);
            dispatch(*ctx_, lookup_pso(*ctx_, "matmul_mpp"), [&](MTL::ComputeCommandEncoder* enc) {
                enc->setBuffer(Ah, off_A, 0);
                enc->setBuffer(Bh, off_B, 1);
                enc->setBuffer(Ch, off_C, 2);
                enc->setBytes(&M, sizeof(uint32_t), 3);
                enc->setBytes(&N, sizeof(uint32_t), 4);
                enc->setBytes(&K, sizeof(uint32_t), 5);
            }, g_mm, tg_mm);
        }

        if (!half_storage) {
            // f16 scratch -> f32 output buffer.
            uint32_t numel = M * N * batch;
            MTL::Size tg = MTL::Size(256, 1, 1);
            MTL::Size g  = MTL::Size((numel + 255) / 256, 1, 1);
            dispatch(*ctx_, lookup_pso(*ctx_, "f16_to_f32"), [&](MTL::ComputeCommandEncoder* enc) {
                enc->setBuffer(Ch,           0, 0);
                enc->setBuffer(C.buf.get(),  0, 1);
                enc->setBytes(&numel, sizeof(uint32_t), 2);
            }, g, tg);
        }
        return;
    }

    const char* kernel = nullptr;
    MTL::Size   threadgroup, grid;
    switch (matmul_impl_) {
        case MatMul::Naive:
            kernel = "matmul_naive";
            threadgroup = MTL::Size(16, 16, 1);
            grid        = MTL::Size((M + 15) / 16, (N + 15) / 16, batch);
            break;
        case MatMul::Tiled:
            kernel = "matmul_tiled";
            threadgroup = MTL::Size(32, 32, 1);
            grid        = MTL::Size((M + 31) / 32, (N + 31) / 32, batch);
            break;
        case MatMul::SIMD:
            kernel = "matmul_simd";
            threadgroup = MTL::Size(32, 1, 1);
            grid        = MTL::Size((M + 7) / 8, (N + 7) / 8, batch);
            break;
        case MatMul::SimdTiled:
            kernel = "matmul_simd_tiled";
            threadgroup = MTL::Size(512, 1, 1);
            grid        = MTL::Size((M + 31) / 32, (N + 31) / 32, batch);
            break;
        case MatMul::MPP:
            __builtin_unreachable();   // handled above
    }

    dispatch(*ctx_, lookup_pso(*ctx_, kernel), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(A.buf.get(), 0, 0);
        enc->setBuffer(B.buf.get(), 0, 1);
        enc->setBuffer(C.buf.get(), 0, 2);
        enc->setBytes(&M, sizeof(uint32_t), 3);
        enc->setBytes(&N, sizeof(uint32_t), 4);
        enc->setBytes(&K, sizeof(uint32_t), 5);
    }, grid, threadgroup);
}

void Executor::visit(cg::ReduceNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    int nd   = (int)in.shape.size();
    int axis = node.axis < 0 ? nd + node.axis : node.axis;
    if (axis < 0 || axis >= nd)
        throw std::runtime_error("Metal Reduce: axis out of range");

    uint32_t outer = 1, inner = 1;
    for (int d = 0;        d < axis; ++d) outer *= in.shape[d];
    for (int d = axis + 1; d < nd;   ++d) inner *= in.shape[d];
    uint32_t axis_size = in.shape[axis];
    uint32_t op_code   = static_cast<uint32_t>(node.op);

    std::vector<int> out_shape = in.shape;
    out_shape[axis] = 1;
    MetalBuffer& out = ensure_output(*ctx_, &node, std::move(out_shape));

    uint32_t total_out = outer * inner;
    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total_out + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, "reduce_axis"), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(in.buf.get(),  0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        enc->setBytes(&outer,     sizeof(uint32_t), 2);
        enc->setBytes(&axis_size, sizeof(uint32_t), 3);
        enc->setBytes(&inner,     sizeof(uint32_t), 4);
        enc->setBytes(&op_code,   sizeof(uint32_t), 5);
    }, grid, threadgroup);
}

void Executor::visit(cg::TransposeNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    int nd = (int)in.shape.size();

    std::vector<int> perm = node.perm;
    if (perm.empty()) {
        perm.resize(nd);
        std::iota(perm.begin(), perm.end(), 0);
        if (nd >= 2) std::swap(perm[nd - 2], perm[nd - 1]);
    }

    std::vector<int> out_shape(nd);
    for (int i = 0; i < nd; ++i) out_shape[i] = in.shape[perm[i]];
    MetalBuffer& out = ensure_output(*ctx_, &node, std::move(out_shape));

    std::vector<uint32_t> in_shape_u(nd), out_shape_u(nd), perm_u(nd);
    for (int i = 0; i < nd; ++i) {
        in_shape_u[i]  = (uint32_t)in.shape[i];
        out_shape_u[i] = (uint32_t)out.shape[i];
        perm_u[i]      = (uint32_t)perm[i];
    }
    uint32_t ndim_u  = (uint32_t)nd;
    uint32_t numel_u = (uint32_t)out.numel();

    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((numel_u + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "transpose_nd")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(in.buf.get(),  0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        enc->setBytes(in_shape_u.data(),  in_shape_u.size()  * sizeof(uint32_t), 2);
        enc->setBytes(out_shape_u.data(), out_shape_u.size() * sizeof(uint32_t), 3);
        enc->setBytes(perm_u.data(),      perm_u.size()      * sizeof(uint32_t), 4);
        enc->setBytes(&ndim_u,  sizeof(uint32_t), 5);
        enc->setBytes(&numel_u, sizeof(uint32_t), 6);
    }, grid, threadgroup);
}

void Executor::visit(cg::BroadcastNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    int nd   = (int)in.shape.size();
    int axis = node.axis < 0 ? nd + node.axis : node.axis;
    if (axis < 0 || axis >= nd)
        throw std::runtime_error("Metal Broadcast: axis out of range");
    if (in.shape[axis] != 1)
        throw std::runtime_error("Metal Broadcast: source axis must have size 1");

    std::vector<int> out_shape = in.shape;
    out_shape[axis] = node.count;
    MetalBuffer& out = ensure_output(*ctx_, &node, std::move(out_shape));

    uint32_t numel = (uint32_t)out.numel();
    std::vector<uint32_t> shape_u(nd);
    for (int i = 0; i < nd; ++i) shape_u[i] = (uint32_t)out.shape[i];
    uint32_t ndim_u = (uint32_t)nd;
    uint32_t axis_u = (uint32_t)axis;

    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((numel + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "broadcast_axis")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(in.buf.get(),  0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        enc->setBytes(shape_u.data(), shape_u.size() * sizeof(uint32_t), 2);
        enc->setBytes(&ndim_u, sizeof(uint32_t), 3);
        enc->setBytes(&axis_u, sizeof(uint32_t), 4);
        enc->setBytes(&numel,  sizeof(uint32_t), 5);
    }, grid, threadgroup);
}

void Executor::visit(cg::ReshapeNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    int new_n = 1;
    for (int d : node.new_shape) new_n *= d;
    if (new_n != in.numel())
        throw std::runtime_error("Metal Reshape: numel mismatch");

    auto it = ctx_->device_results.find(&node);
    if (it != ctx_->device_results.end()) {
        // If the input's buffer was reallocated (e.g. shape change), refresh
        // the alias. Shared ownership: the old buffer drops automatically
        // when no other view holds it.
        if (it->second.buf.get() != in.buf.get()) it->second.buf = in.buf;
        it->second.shape = node.new_shape;
        return;
    }
    // First time: zero-copy view — both nodes share ownership of the buffer.
    MetalBuffer out;
    out.shape = node.new_shape;
    out.buf   = in.buf;
    ctx_->device_results.emplace(&node, std::move(out));
}

// im2col + col2im share an arg-binding pattern
static void bind_im2col_args(MTL::ComputeCommandEncoder* enc,
                             uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                             uint32_t kH, uint32_t kW, uint32_t stride, int32_t pad,
                             uint32_t Hout, uint32_t Wout)
{
    enc->setBytes(&N,      sizeof(uint32_t), 2);
    enc->setBytes(&C,      sizeof(uint32_t), 3);
    enc->setBytes(&H,      sizeof(uint32_t), 4);
    enc->setBytes(&W,      sizeof(uint32_t), 5);
    enc->setBytes(&kH,     sizeof(uint32_t), 6);
    enc->setBytes(&kW,     sizeof(uint32_t), 7);
    enc->setBytes(&stride, sizeof(uint32_t), 8);
    enc->setBytes(&pad,    sizeof(int32_t),  9);
    enc->setBytes(&Hout,   sizeof(uint32_t), 10);
    enc->setBytes(&Wout,   sizeof(uint32_t), 11);
}

void Executor::visit(cg::Im2ColNode& node) {
    const MetalBuffer& X = ctx_->device_results.at(node.input);
    if (X.shape.size() != 4) throw std::runtime_error("Metal im2col: expected 4D");
    uint32_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    uint32_t kH = node.kH, kW = node.kW, s = node.stride;
    int32_t  p  = node.pad;
    uint32_t Hout = (H + 2 * p - kH) / s + 1;
    uint32_t Wout = (W + 2 * p - kW) / s + 1;
    uint32_t patch_size = C * kH * kW;
    uint32_t n_patches  = N * Hout * Wout;
    uint32_t total      = n_patches * patch_size;

    MetalBuffer& out = ensure_output(*ctx_, &node, std::vector<int>{(int)n_patches, (int)patch_size});

    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "im2col")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(X.buf.get(),   0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        bind_im2col_args(enc, N, C, H, W, kH, kW, s, p, Hout, Wout);
    }, grid, threadgroup);
}

void Executor::visit(cg::Col2ImNode& node) {
    const MetalBuffer& In = ctx_->device_results.at(node.input);
    if (node.output_shape.size() != 4) throw std::runtime_error("Metal col2im: 4D output");
    uint32_t N = node.output_shape[0], C = node.output_shape[1];
    uint32_t H = node.output_shape[2], W = node.output_shape[3];
    uint32_t kH = node.kH, kW = node.kW, s = node.stride;
    int32_t  p  = node.pad;
    uint32_t Hout = (H + 2 * p - kH) / s + 1;
    uint32_t Wout = (W + 2 * p - kW) / s + 1;

    MetalBuffer& out = ensure_output(*ctx_, &node, node.output_shape);

    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((out.numel() + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, "col2im"), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(In.buf.get(),  0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        bind_im2col_args(enc, N, C, H, W, kH, kW, s, p, Hout, Wout);
    }, grid, threadgroup);
}

void Executor::visit(cg::MaxPool2DNode& node) {
    const MetalBuffer& X = ctx_->device_results.at(node.input);
    if (X.shape.size() != 4) throw std::runtime_error("Metal MaxPool: expected 4D");
    uint32_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    uint32_t k = node.k, s = node.stride;
    int32_t  p = node.pad;
    uint32_t Hout = (H + 2 * p - k) / s + 1;
    uint32_t Wout = (W + 2 * p - k) / s + 1;

    MetalBuffer& out = ensure_output(*ctx_, &node,
        std::vector<int>{(int)N, (int)C, (int)Hout, (int)Wout});

    uint32_t total = N * C * Hout * Wout;
    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "maxpool2d")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(X.buf.get(),   0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        enc->setBytes(&N,    sizeof(uint32_t), 2);
        enc->setBytes(&C,    sizeof(uint32_t), 3);
        enc->setBytes(&H,    sizeof(uint32_t), 4);
        enc->setBytes(&W,    sizeof(uint32_t), 5);
        enc->setBytes(&k,    sizeof(uint32_t), 6);
        enc->setBytes(&s,    sizeof(uint32_t), 7);
        enc->setBytes(&p,    sizeof(int32_t),  8);
        enc->setBytes(&Hout, sizeof(uint32_t), 9);
        enc->setBytes(&Wout, sizeof(uint32_t), 10);
    }, grid, threadgroup);
}

void Executor::visit(cg::UpsampleNearestNode& node) {
    const MetalBuffer& X = ctx_->device_results.at(node.input);
    if (X.shape.size() != 4) throw std::runtime_error("Metal Upsample: expected 4D");
    uint32_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    uint32_t scale = node.scale;
    uint32_t Ho = H * scale, Wo = W * scale;

    MetalBuffer& out = ensure_output(*ctx_, &node,
        std::vector<int>{(int)N, (int)C, (int)Ho, (int)Wo});

    uint32_t total = N * C * Ho * Wo;
    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "upsample_nearest")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(X.buf.get(),   0, 0);
        enc->setBuffer(out.buf.get(), 0, 1);
        enc->setBytes(&N,     sizeof(uint32_t), 2);
        enc->setBytes(&C,     sizeof(uint32_t), 3);
        enc->setBytes(&H,     sizeof(uint32_t), 4);
        enc->setBytes(&W,     sizeof(uint32_t), 5);
        enc->setBytes(&scale, sizeof(uint32_t), 6);
    }, grid, threadgroup);
}

void Executor::visit(cg::ConcatNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.a);
    const MetalBuffer& B = ctx_->device_results.at(node.b);
    if (A.shape.size() != 4 || B.shape.size() != 4)
        throw std::runtime_error("Metal Concat: expected 4D inputs");
    uint32_t N = A.shape[0], Ca = A.shape[1], H = A.shape[2], W = A.shape[3];
    uint32_t Cb = B.shape[1];
    if (B.shape[0] != (int)N || B.shape[2] != (int)H || B.shape[3] != (int)W)
        throw std::runtime_error("Metal Concat: N/H/W mismatch");

    MetalBuffer& out = ensure_output(*ctx_, &node,
        std::vector<int>{(int)N, (int)(Ca + Cb), (int)H, (int)W});

    uint32_t total = N * (Ca + Cb) * H * W;
    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "concat_channel")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(A.buf.get(),   0, 0);
        enc->setBuffer(B.buf.get(),   0, 1);
        enc->setBuffer(out.buf.get(), 0, 2);
        enc->setBytes(&N,  sizeof(uint32_t), 3);
        enc->setBytes(&Ca, sizeof(uint32_t), 4);
        enc->setBytes(&Cb, sizeof(uint32_t), 5);
        enc->setBytes(&H,  sizeof(uint32_t), 6);
        enc->setBytes(&W,  sizeof(uint32_t), 7);
    }, grid, threadgroup);
}

void Executor::visit(cg::BatchNorm2DNode& node) {
    const MetalBuffer& X  = ctx_->device_results.at(node.input);
    const MetalBuffer& g  = ctx_->device_results.at(node.gamma);
    const MetalBuffer& bt = ctx_->device_results.at(node.beta);
    const MetalBuffer& mu = ctx_->device_results.at(node.running_mean);
    const MetalBuffer& va = ctx_->device_results.at(node.running_var);
    if (X.shape.size() != 4) throw std::runtime_error("Metal BN2D: expected 4D input");
    uint32_t N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    if (g.numel() != (int)C || bt.numel() != (int)C ||
        mu.numel() != (int)C || va.numel() != (int)C)
        throw std::runtime_error("Metal BN2D: per-channel param size mismatch");

    MetalBuffer& out = ensure_output(*ctx_, &node,
        std::vector<int>{(int)N, (int)C, (int)H, (int)W});

    uint32_t total = N * C * H * W;
    float eps = node.eps;
    MTL::Size threadgroup = MTL::Size(256, 1, 1);
    MTL::Size grid        = MTL::Size((total + 255) / 256, 1, 1);

    dispatch(*ctx_, lookup_pso(*ctx_, K(*ctx_, "batchnorm2d_infer")), [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(X.buf.get(),   0, 0);
        enc->setBuffer(g.buf.get(),   0, 1);
        enc->setBuffer(bt.buf.get(),  0, 2);
        enc->setBuffer(mu.buf.get(),  0, 3);
        enc->setBuffer(va.buf.get(),  0, 4);
        enc->setBuffer(out.buf.get(), 0, 5);
        enc->setBytes(&N,   sizeof(uint32_t), 6);
        enc->setBytes(&C,   sizeof(uint32_t), 7);
        enc->setBytes(&H,   sizeof(uint32_t), 8);
        enc->setBytes(&W,   sizeof(uint32_t), 9);
        enc->setBytes(&eps, sizeof(float),    10);
    }, grid, threadgroup);
}

// Copy `value`'s device buffer into `target` InputNode's host tensor (and
// into target's device buffer, so any later read in the same pass sees the
// new value). Runs after the GPU has finished producing the value buffer.
//
// Note: this still pays one device→host round trip per parameter per step.
// A future optimization would alias the buffer (shared_ptr swap) and
// suppress the host→device upload on the next visit(InputNode), avoiding
// the round trip entirely.
void Executor::visit(cg::AssignNode& node) {
    const MetalBuffer& src = ctx_->device_results.at(node.value);
    MetalBuffer&       dst = ctx_->device_results.at(node.target);
    if (dst.shape != src.shape)
        throw std::runtime_error("Metal Assign: shape mismatch with target '" + node.target->name + "'");

    ctx_->flush();   // value buffer must be done writing before we read

    size_t bytes = src.numel() * sizeof(float);
    if ((int)node.target->tensor.numel() != src.numel())
        throw std::runtime_error("Metal Assign: target tensor size mismatch");
    std::memcpy(node.target->tensor.data(), src.buf->contents(), bytes);  // host sync
    std::memcpy(dst.buf->contents(),        src.buf->contents(), bytes);  // device coherence

    // Assign's result is the new value — alias the source buffer.
    auto& mb = ctx_->device_results[&node];
    mb.shape = src.shape;
    mb.buf   = src.buf;
    ctx_->host_cache.erase(&node);
}

const cg::Tensor& Executor::result(cg::Node* node) const {
    auto it = ctx_->host_cache.find(node);
    if (it != ctx_->host_cache.end()) return it->second;

    ctx_->flush();   // make sure the GPU is done before we read

    const MetalBuffer& mb = ctx_->device_results.at(node);
    auto [iter, _] = ctx_->host_cache.emplace(node, from_device(mb, ctx_->precision));
    return iter->second;
}

void Executor::reset() {
    // Between runs of the same graph: drop the previous run's host-cached
    // tensors so result() re-fetches, and discard any pending command buffer.
    ctx_->end_encoder();
    if (ctx_->pending) {
        ctx_->pending->commit();
        ctx_->pending->waitUntilCompleted();
        ctx_->pending = nullptr;
    }
    ctx_->host_cache.clear();
}

void Executor::clear() {
    ctx_->flush();
    ctx_->device_results.clear();   // shared_ptr deleters release MTL buffers
    ctx_->host_cache.clear();
}

} // namespace cg::metal
