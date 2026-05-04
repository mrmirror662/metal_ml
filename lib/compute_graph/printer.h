#pragma once
#include "cg.h"
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace cg {

class PrintVisitor : public Visitor {
public:
    explicit PrintVisitor(std::ostream& out = std::cout) : out_(out) {}

    void visit(InputNode& node) override {
        std::ostringstream s;
        s << "Input \"" << node.name << "\" " << shape(node.tensor);
        print(s.str(), node);
    }

    void visit(MatMulNode& node) override {
        print("MatMul  (" + ref(node.lhs) + ", " + ref(node.rhs) + ")", node);
    }

    void visit(MatAddNode& node) override {
        print("MatAdd  (" + ref(node.lhs) + ", " + ref(node.rhs) + ")", node);
    }

    void visit(HadamardNode& node) override {
        print("Hadamard(" + ref(node.lhs) + ", " + ref(node.rhs) + ")", node);
    }

    void visit(MapNode& node) override {
        print("Map     " + op_str(node.op) + " (" + ref(node.input) + ")", node);
    }

    void visit(ScaleNode& node) override {
        print("Scale   x" + std::to_string(node.scalar) + " (" + ref(node.input) + ")", node);
    }

    void visit(ReduceNode& node) override {
        print("Reduce  " + op_str(node.op) + " axis=" + std::to_string(node.axis)
              + " (" + ref(node.input) + ")", node);
    }

    void visit(TransposeNode& node) override {
        std::string perm_s = node.perm.empty() ? "default" : vec_str(node.perm);
        print("Transpose perm=" + perm_s + " (" + ref(node.input) + ")", node);
    }

    void visit(BroadcastNode& node) override {
        print("Broadcast axis=" + std::to_string(node.axis)
              + " count=" + std::to_string(node.count)
              + " (" + ref(node.input) + ")", node);
    }

    void visit(ReshapeNode& node) override {
        print("Reshape " + vec_str(node.new_shape) + " (" + ref(node.input) + ")", node);
    }

    void visit(Im2ColNode& node) override {
        print("Im2Col k=" + std::to_string(node.kH) + "x" + std::to_string(node.kW)
              + " s=" + std::to_string(node.stride) + " p=" + std::to_string(node.pad)
              + " (" + ref(node.input) + ")", node);
    }

    void visit(Col2ImNode& node) override {
        print("Col2Im out=" + vec_str(node.output_shape)
              + " k=" + std::to_string(node.kH) + "x" + std::to_string(node.kW)
              + " s=" + std::to_string(node.stride) + " p=" + std::to_string(node.pad)
              + " (" + ref(node.input) + ")", node);
    }

private:
    std::ostream& out_;
    std::unordered_map<Node*, int> idx_;
    int counter_ = 0;

    void print(const std::string& desc, Node& node) {
        int i = counter_++;
        idx_[&node] = i;
        out_ << "[" << i << "] " << desc << "\n";
    }

    std::string ref(Node* n) const {
        auto it = idx_.find(n);
        return it != idx_.end() ? std::to_string(it->second) : "?";
    }

    static std::string shape(const Tensor& t) {
        std::string s = "[";
        for (int i = 0; i < (int)t.shape().size(); ++i) {
            s += std::to_string(t.shape()[i]);
            if (i + 1 < (int)t.shape().size()) s += "x";
        }
        return s + "]";
    }

    static std::string op_str(MapOp op) {
        switch (op) {
            case MapOp::ReLU:    return "ReLU";
            case MapOp::Softmax: return "Softmax";
            case MapOp::Step:    return "Step";
        }
        return "?";
    }

    static std::string op_str(ReduceOp op) {
        switch (op) {
            case ReduceOp::Sum:  return "Sum";
            case ReduceOp::Mean: return "Mean";
            case ReduceOp::Max:  return "Max";
            case ReduceOp::Min:  return "Min";
        }
        return "?";
    }

    static std::string vec_str(const std::vector<int>& v) {
        std::string s = "{";
        for (int i = 0; i < (int)v.size(); ++i) {
            s += std::to_string(v[i]);
            if (i + 1 < (int)v.size()) s += ",";
        }
        return s + "}";
    }
};

} // namespace cg
