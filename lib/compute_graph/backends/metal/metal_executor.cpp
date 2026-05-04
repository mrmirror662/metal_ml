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
    MTL::Device*       device  = nullptr;
    MTL::CommandQueue* queue   = nullptr;
    MTL::Library*      library = nullptr;

    std::unordered_map<std::string, MTL::ComputePipelineState*> psos;

    // Persists across runs of the same graph so subsequent runs reuse buffers.
    std::unordered_map<cg::Node*, MetalBuffer>        device_results;
    mutable std::unordered_map<cg::Node*, cg::Tensor> host_cache;
    mutable MTL::CommandBuffer*                       pending = nullptr;

    ~MetalContext() {
        if (pending) { pending->commit(); pending->waitUntilCompleted(); }
        // device_results' MtlBufferPtrs release themselves as the map clears.
        for (auto& [_, pso] : psos) pso->release();
        if (library) library->release();
        if (queue)   queue->release();
        if (device)  device->release();
    }

    void flush() const {
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

    auto it = ctx.device_results.find(node);
    if (it != ctx.device_results.end()) {
        MetalBuffer& mb = it->second;
        if (mb.shape == shape) return mb;
        // Shape changed (e.g. variable batch size) — drop the old buffer
        // (any aliased reshape views drop with it via shared_ptr) and realloc.
        mb.shape = std::move(shape);
        mb.buf = wrap_buffer(ctx.device->newBuffer(numel * sizeof(float), MTL::ResourceStorageModeShared));
        ctx.host_cache.erase(node);
        return mb;
    }
    MetalBuffer mb;
    mb.shape = std::move(shape);
    mb.buf = wrap_buffer(ctx.device->newBuffer(numel * sizeof(float), MTL::ResourceStorageModeShared));
    auto [iter, _] = ctx.device_results.emplace(node, std::move(mb));
    return iter->second;
}

static cg::Tensor from_device(const MetalBuffer& mb) {
    cg::Tensor t(mb.shape);
    std::memcpy(t.data(), mb.buf->contents(), mb.numel() * sizeof(float));
    return t;
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

// Encode one kernel into the graph-level command buffer. No commit/wait.
static void dispatch(
    MetalContext&                                              ctx,
    MTL::ComputePipelineState*                                 pso,
    const std::function<void(MTL::ComputeCommandEncoder*)>&    bind,
    MTL::Size                                                  grid,
    MTL::Size                                                  threadgroup)
{
    if (!ctx.pending)
        ctx.pending = ctx.queue->commandBuffer();

    auto* enc = ctx.pending->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    bind(enc);
    enc->dispatchThreadgroups(grid, threadgroup);
    enc->endEncoding();
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
    reg("softmax_lastdim");
    reg("broadcast_axis");
    reg("scale");
    reg("reduce_axis");
    reg("transpose_nd");
    reg("im2col");
    reg("col2im");
    reg("matmul_naive");
    reg("matmul_tiled");
    reg("matmul_simd");
    reg("matmul_simd_tiled");
}

Executor::~Executor() = default;

const std::string Executor::device_name() const {
    return ctx_->device->name()->utf8String();
}

// ---------- Visit methods ----------

void Executor::visit(cg::InputNode& node) {
    size_t bytes = node.tensor.numel() * sizeof(float);
    auto it = ctx_->device_results.find(&node);
    if (it != ctx_->device_results.end()) {
        MetalBuffer& mb = it->second;
        // Reupload only valid if shape (and therefore size) is unchanged.
        if (mb.shape != node.tensor.shape()) {
            mb.shape = node.tensor.shape();
            mb.buf   = wrap_buffer(ctx_->device->newBuffer(
                node.tensor.data(), bytes, MTL::ResourceStorageModeShared));
            ctx_->host_cache.erase(&node);
            return;
        }
        std::memcpy(mb.buf->contents(), node.tensor.data(), bytes);
        return;
    }
    MetalBuffer mb;
    mb.shape = node.tensor.shape();
    mb.buf = wrap_buffer(ctx_->device->newBuffer(
        node.tensor.data(), bytes, MTL::ResourceStorageModeShared));
    ctx_->device_results.emplace(&node, std::move(mb));
}

void Executor::visit(cg::MatAddNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.lhs);
    const MetalBuffer& B = ctx_->device_results.at(node.rhs);
    if (A.shape != B.shape)
        throw std::runtime_error("Metal MatAdd: shape mismatch");

    MetalBuffer& C = ensure_output(*ctx_, &node, A.shape);
    dispatch_elementwise(*ctx_, "mat_add", C, {A.buf.get(), B.buf.get()});
}

void Executor::visit(cg::HadamardNode& node) {
    const MetalBuffer& A = ctx_->device_results.at(node.lhs);
    const MetalBuffer& B = ctx_->device_results.at(node.rhs);
    if (A.shape != B.shape)
        throw std::runtime_error("Metal Hadamard: shape mismatch");

    MetalBuffer& C = ensure_output(*ctx_, &node, A.shape);
    dispatch_elementwise(*ctx_, "hadamard", C, {A.buf.get(), B.buf.get()});
}

void Executor::visit(cg::MapNode& node) {
    const MetalBuffer& in = ctx_->device_results.at(node.input);
    MetalBuffer& out = ensure_output(*ctx_, &node, in.shape);

    switch (node.op) {
        case cg::MapOp::ReLU:
            dispatch_elementwise(*ctx_, "relu", out, {in.buf.get()});
            break;
        case cg::MapOp::Step:
            dispatch_elementwise(*ctx_, "step_op", out, {in.buf.get()});
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
    dispatch_elementwise(*ctx_, "scale", out, {in.buf.get()}, {{&node.scalar, sizeof(float)}});
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

    dispatch(*ctx_, lookup_pso(*ctx_, "transpose_nd"), [&](MTL::ComputeCommandEncoder* enc) {
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

    dispatch(*ctx_, lookup_pso(*ctx_, "broadcast_axis"), [&](MTL::ComputeCommandEncoder* enc) {
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

    dispatch(*ctx_, lookup_pso(*ctx_, "im2col"), [&](MTL::ComputeCommandEncoder* enc) {
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

const cg::Tensor& Executor::result(cg::Node* node) const {
    auto it = ctx_->host_cache.find(node);
    if (it != ctx_->host_cache.end()) return it->second;

    ctx_->flush();   // make sure the GPU is done before we read

    const MetalBuffer& mb = ctx_->device_results.at(node);
    auto [iter, _] = ctx_->host_cache.emplace(node, from_device(mb));
    return iter->second;
}

void Executor::reset() {
    // Between runs of the same graph: drop the previous run's host-cached
    // tensors so result() re-fetches, and discard any pending command buffer.
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
