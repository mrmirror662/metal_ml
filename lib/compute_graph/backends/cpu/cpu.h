#pragma once
#include "cg.h"
#include <unordered_map>

namespace cg::cpu {

class Executor : public cg::Visitor {
public:
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

    // Retrieve the output tensor computed for a node
    const cg::Tensor& result(cg::Node* node) const;
    void              clear() { results_.clear(); }
    // For API parity with metal::Executor — CPU has no host cache to invalidate
    // (each visit overwrites its results_ entry, so a fresh accept() produces
    // fresh values).
    void              reset() {}

private:
    std::unordered_map<cg::Node*, cg::Tensor> results_;
};

} // namespace cg::cpu
