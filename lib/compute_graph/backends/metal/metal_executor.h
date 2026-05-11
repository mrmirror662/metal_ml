#pragma once
#include "cg.h"
#include <memory>

namespace cg::metal {

struct MetalContext;  // opaque — defined in .cpp, keeps Metal-CPP out of this header

class Executor : public cg::Visitor {
public:
    enum class MatMul { Naive, Tiled, SIMD, SimdTiled, MPP };

    // Tiled is the default — handles arbitrary shapes correctly.
    // Simd / SimdTiled require M, N, K divisible by 8.
    // MPP uses Apple's Metal Performance Primitives matmul2d (Metal 4 /
    // macOS 26+), mixed precision (f16 in, f16 out) with f32 accumulation.
    // Inputs are converted f32→f16 on the fly; output is converted f16→f32
    // back into the executor's f32 result buffer. MPP handles edge tiles, so
    // any M, N, K is valid; the kernel uses a 64×32 output tile and 4
    // SIMD-groups per threadgroup.
    explicit Executor(MatMul matmul = MatMul::Tiled);
    ~Executor();

    // Called by ComputeGraph::accept() before any visit(); records the
    // graph's declared precision so subsequent visits dispatch the right
    // kernel variant and allocate device buffers at the right element size.
    void on_precision(cg::Precision p) override;

    void visit(cg::InputNode&     node) override;
    void visit(cg::MatMulNode&    node) override;
    void visit(cg::MatAddNode&    node) override;
    void visit(cg::HadamardNode&  node) override;
    void visit(cg::MapNode&       node) override;
    void visit(cg::ScaleNode&     node) override;
    void visit(cg::ReduceNode&    node) override;
    void visit(cg::TransposeNode& node) override;
    void visit(cg::BroadcastNode& node) override;
    void visit(cg::ReshapeNode&   node) override;
    void visit(cg::Im2ColNode&    node) override;
    void visit(cg::Col2ImNode&    node) override;
    void visit(cg::MaxPool2DNode& node) override;
    void visit(cg::UpsampleNearestNode& node) override;
    void visit(cg::ConcatNode&    node) override;
    void visit(cg::BatchNorm2DNode& node) override;
    void visit(cg::AssignNode&    node) override;

    const std::string  device_name() const;
    const cg::Tensor&  result(cg::Node* node) const;  // implicitly flushes
    void               clear();                       // drop all results & buffers
    // Reset between runs of the same graph — clears host result cache, but
    // keeps device buffers alive so subsequent runs can reuse them.
    void               reset();

private:
    std::unique_ptr<MetalContext> ctx_;
    MatMul                        matmul_impl_;
};

} // namespace cg::metal
