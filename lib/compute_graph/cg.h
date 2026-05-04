#pragma once
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#define VERSION "0.0.1"

namespace cg {

std::string version();

// --- Tensor ---
struct Tensor {
    std::vector<int>   shape;
    std::vector<int>   strides;  // element strides, row-major
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape);                        // zero-initialized
    Tensor(std::vector<int> shape, std::vector<float> data);

    int ndim()  const { return (int)shape.size(); }
    int numel() const;

    // multi-index element access
    float&       at(const std::vector<int>& idx);
    float        at(const std::vector<int>& idx) const;

    static std::vector<int> make_strides(const std::vector<int>& shape);
};

// --- Enums ---
enum class MapOp    { ReLU, Softmax, Step };  // Step: x > 0 ? 1 : 0
enum class ReduceOp { Sum, Mean, Max, Min };

// --- Forward declarations ---
class InputNode;
class MatMulNode;
class MatAddNode;
class HadamardNode;
class MapNode;
class ScaleNode;
class ReduceNode;
class TransposeNode;
class BroadcastNode;
class ReshapeNode;
class Im2ColNode;
class Col2ImNode;

// --- Visitor interface ---
class Visitor {
public:
    virtual ~Visitor() = default;
    virtual void visit(InputNode&     node) = 0;
    virtual void visit(MatMulNode&    node) = 0;
    virtual void visit(MatAddNode&    node) = 0;
    virtual void visit(HadamardNode&  node) = 0;
    virtual void visit(MapNode&       node) = 0;
    virtual void visit(ScaleNode&     node) = 0;
    virtual void visit(ReduceNode&    node) = 0;
    virtual void visit(TransposeNode& node) = 0;
    virtual void visit(BroadcastNode& node) = 0;
    virtual void visit(ReshapeNode&   node) = 0;
    virtual void visit(Im2ColNode&    node) = 0;
    virtual void visit(Col2ImNode&    node) = 0;
};

// --- Abstract Node ---
class Node {
public:
    virtual ~Node() = default;
    virtual void accept(Visitor& v) = 0;
    virtual std::vector<Node*> inputs() const { return {}; }
};

// --- Concrete Nodes ---
class InputNode : public Node {
public:
    InputNode(std::string name, Tensor tensor)
        : name(std::move(name)), tensor(std::move(tensor)) {}
    void accept(Visitor& v) override { v.visit(*this); }

    std::string name;
    Tensor      tensor;
};

class MatMulNode : public Node {
public:
    MatMulNode(Node* lhs, Node* rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class MatAddNode : public Node {
public:
    MatAddNode(Node* lhs, Node* rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class HadamardNode : public Node {  // element-wise multiply, same shape
public:
    HadamardNode(Node* lhs, Node* rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class MapNode : public Node {
public:
    MapNode(Node* input, MapOp op) : input(input), op(op) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*  input;
    MapOp  op;
};

class ScaleNode : public Node {
public:
    ScaleNode(Node* input, float scalar) : input(input), scalar(scalar) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*  input;
    float  scalar;
};

class ReduceNode : public Node {
public:
    ReduceNode(Node* input, ReduceOp op, int axis)
        : input(input), op(op), axis(axis) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*     input;
    ReduceOp  op;
    int       axis;   // dimension to reduce; negative indexing supported
};

class TransposeNode : public Node {
public:
    // perm: full permutation of dims, e.g. {0,2,1,3} for a 4D tensor.
    // empty perm = swap last two dims.
    explicit TransposeNode(Node* input, std::vector<int> perm = {})
        : input(input), perm(std::move(perm)) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*             input;
    std::vector<int>  perm;
};

// Replicates `input` along an existing size-1 axis to size `count`.
// e.g. [1, hidden] with axis=0, count=batch -> [batch, hidden].
class BroadcastNode : public Node {
public:
    BroadcastNode(Node* input, int axis, int count)
        : input(input), axis(axis), count(count) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node* input;
    int   axis;
    int   count;
};

// View-only shape change — same data, new shape (must have matching numel).
// `input_shape` is recorded at construction so autograd can emit a reverse
// reshape without needing static shape inference.
class ReshapeNode : public Node {
public:
    ReshapeNode(Node* input, std::vector<int> input_shape, std::vector<int> new_shape)
        : input(input), input_shape(std::move(input_shape)), new_shape(std::move(new_shape)) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> input_shape;
    std::vector<int> new_shape;
};

// im2col: input [N, C, H, W] -> output [N*Hout*Wout, C*kH*kW]
// Each row of the output is one flattened input patch in scan order.
// Hout = (H + 2*pad - kH) / stride + 1, similarly Wout.
class Im2ColNode : public Node {
public:
    Im2ColNode(Node* input, std::vector<int> input_shape,
               int kH, int kW, int stride, int pad)
        : input(input), input_shape(std::move(input_shape)),
          kH(kH), kW(kW), stride(stride), pad(pad) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> input_shape;   // [N, C, H, W] — recorded for autograd
    int              kH, kW, stride, pad;
};

// col2im: input [N*Hout*Wout, C*kH*kW] -> output [N, C, H, W]
// Inverse of im2col with overlap-add (used in conv backward dX).
// Output shape must be specified explicitly because im2col's reverse mapping
// depends on the original spatial dims.
class Col2ImNode : public Node {
public:
    Col2ImNode(Node* input, std::vector<int> output_shape, int kH, int kW, int stride, int pad)
        : input(input), output_shape(std::move(output_shape)),
          kH(kH), kW(kW), stride(stride), pad(pad) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> output_shape;   // [N, C, H, W]
    int              kH, kW, stride, pad;
};

// --- ComputeGraph ---
class ComputeGraph {
public:
    template<typename T, typename... Args>
    T* emplace(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = node.get();
        nodes_.push_back(std::move(node));
        return ptr;
    }

    bool is_dag() const;
    void accept(Visitor& v);

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

} // namespace cg
