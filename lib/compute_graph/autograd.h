#pragma once
// Reverse-mode autograd for cg::ComputeGraph.
//
// Each visit() method below is the backward rule for one primitive — it reads
// the gradient at this node's output (already populated by consumers) and
// emits gradient subgraphs for each input, accumulating into the grads_ map.
//
// Walks the graph in reverse topological order from seeded nodes. Backward
// subgraphs are appended to the SAME ComputeGraph as the forward; everything
// runs in one Executor pass.

#include "cg.h"
#include "nn.h"

#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg::autograd {

class BackwardBuilder : public Visitor {
public:
    explicit BackwardBuilder(ComputeGraph& g) : g_(g) {}

    // Manually set the gradient at a forward node (entry point for autograd).
    void seed(Node* fwd, Node* grad) { grads_[fwd] = grad; }

    // Walk backward from every seeded node, emitting backward subgraphs.
    // If `wrt` is non-empty, only compute gradients along forward paths that
    // eventually reach those nodes — skips wasted work like grad of input data.
    void build(const std::vector<Node*>& wrt = {}) {
        // Collect all forward nodes reachable from seeds via the inputs() chain.
        std::vector<Node*> order;
        std::unordered_set<Node*> visited;
        std::function<void(Node*)> dfs = [&](Node* u) {
            if (!visited.insert(u).second) return;
            for (Node* dep : u->inputs()) dfs(dep);
            order.push_back(u);
        };
        for (auto& [n, _] : grads_) dfs(n);

        // Build the "useful" set: forward descendants of wrt within `order`.
        if (!wrt.empty()) {
            std::unordered_map<Node*, std::vector<Node*>> consumers;
            for (Node* n : order)
                for (Node* in : n->inputs())
                    consumers[in].push_back(n);

            std::vector<Node*> stack(wrt.begin(), wrt.end());
            for (Node* w : wrt) needed_.insert(w);
            while (!stack.empty()) {
                Node* n = stack.back(); stack.pop_back();
                auto it = consumers.find(n);
                if (it == consumers.end()) continue;
                for (Node* c : it->second)
                    if (needed_.insert(c).second) stack.push_back(c);
            }
            pruning_ = true;
        }

        // Reverse topo: visit each node after its consumers
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if (grads_.count(*it)) (*it)->accept(*this);
        }
    }

    // Get the gradient subgraph result node for a forward node
    Node* grad(Node* fwd) const {
        auto it = grads_.find(fwd);
        return it == grads_.end() ? nullptr : it->second;
    }

    // ===== Backward rules ================================================

    void visit(InputNode&) override { /* leaf — gradient stays */ }

    void visit(MatMulNode& n) override {
        Node* dy = grads_.at(&n);
        if (need(n.lhs)) accumulate(n.lhs, nn::matmul_grad_lhs(g_, dy, n.rhs));
        if (need(n.rhs)) accumulate(n.rhs, nn::matmul_grad_rhs(g_, n.lhs, dy));
    }

    void visit(MatAddNode& n) override {
        Node* dy = grads_.at(&n);
        if (need(n.lhs)) accumulate(n.lhs, dy);
        if (need(n.rhs)) accumulate(n.rhs, dy);
    }

    void visit(HadamardNode& n) override {
        Node* dy = grads_.at(&n);
        if (need(n.lhs)) accumulate(n.lhs, g_.emplace<HadamardNode>(dy, n.rhs));
        if (need(n.rhs)) accumulate(n.rhs, g_.emplace<HadamardNode>(dy, n.lhs));
    }

    void visit(ScaleNode& n) override {
        Node* dy = grads_.at(&n);
        if (need(n.input)) accumulate(n.input, g_.emplace<ScaleNode>(dy, n.scalar));
    }

    void visit(MapNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        switch (n.op) {
            case MapOp::ReLU:
                accumulate(n.input, nn::relu_backward(g_, dy, n.input));
                break;
            case MapOp::Step:
                /* zero gradient — Step is non-differentiable */
                break;
            case MapOp::Softmax:
                throw std::runtime_error(
                    "Autograd: Softmax is not differentiable here. "
                    "Seed at logits using nn::softmax_ce_backward.");
        }
    }

    void visit(BroadcastNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        accumulate(n.input, g_.emplace<ReduceNode>(dy, ReduceOp::Sum, n.axis));
    }

    void visit(TransposeNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        if (n.perm.empty()) {
            accumulate(n.input, g_.emplace<TransposeNode>(dy));
        } else {
            std::vector<int> inv(n.perm.size());
            for (int i = 0; i < (int)n.perm.size(); ++i) inv[n.perm[i]] = i;
            accumulate(n.input, g_.emplace<TransposeNode>(dy, inv));
        }
    }

    void visit(ReshapeNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        // Reverse reshape: gradient of input has the original input shape.
        accumulate(n.input, g_.emplace<ReshapeNode>(dy, n.new_shape, n.input_shape));
    }

    void visit(Im2ColNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        // im2col is differentiable; backward is col2im over the same params.
        // We need the input's original 4D shape to feed col2im — but we don't
        // statically know it. Caller must record it on the node.
        // (Im2ColNode doesn't store input_shape currently; see below.)
        accumulate(n.input,
            g_.emplace<Col2ImNode>(dy, n.input_shape, n.kH, n.kW, n.stride, n.pad));
    }

    void visit(Col2ImNode& n) override {
        Node* dy = grads_.at(&n);
        if (!need(n.input)) return;
        // dy has the [N,C,H,W] shape (same as col2im's output); im2col on it
        // produces the patches matrix that matches col2im's input shape.
        accumulate(n.input,
            g_.emplace<Im2ColNode>(dy, n.output_shape, n.kH, n.kW, n.stride, n.pad));
    }

    void visit(AssignNode&) override { /* not differentiable; SGD sinks */ }

    void visit(ReduceNode&) override {
        // Reduce backward needs the original input axis size; not yet supported.
        // (Reduce typically appears only in backward subgraphs, which we don't
        // differentiate further.)
        throw std::runtime_error("Autograd: Reduce backward not implemented");
    }

private:
    ComputeGraph& g_;
    std::unordered_map<Node*, Node*> grads_;
    std::unordered_set<Node*>        needed_;
    bool                             pruning_ = false;

    bool need(Node* fwd) const {
        return !pruning_ || needed_.count(fwd);
    }

    void accumulate(Node* fwd, Node* grad) {
        auto it = grads_.find(fwd);
        if (it == grads_.end()) {
            grads_[fwd] = grad;
        } else {
            // Multiple consumers: sum gradients
            grads_[fwd] = g_.emplace<MatAddNode>(it->second, grad);
        }
    }
};

} // namespace cg::autograd
