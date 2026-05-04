#pragma once
#include "cg.h"
#include <memory>

namespace cg::metal {

struct MetalContext;  // opaque — defined in .cpp, keeps Metal-CPP out of this header

class Executor : public cg::Visitor {
public:
    enum class MatMul { Naive, Tiled, SIMD, SimdTiled };

    // Tiled is the default — handles arbitrary shapes correctly.
    // Simd / SimdTiled require M, N, K divisible by 8.
    explicit Executor(MatMul matmul = MatMul::Tiled);
    ~Executor();

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
