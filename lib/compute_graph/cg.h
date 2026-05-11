#pragma once
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define VERSION "0.0.1"

namespace cg {

std::string version();

// --- Tensor ---
//
// Owns a contiguous float buffer with a logical shape. Once constructed the
// shape is fixed: there is no resize() or reshape() that mutates in place,
// because those would let the shape/data invariant drift silently. To change
// the shape you build a new Tensor.
//
// Element data is freely mutable through data(), operator[], or range-for.
// Field access (`t.shape`, `t.data[i]`) is intentionally *not* offered —
// those forms made it possible to reassign t.shape without resizing data,
// which is exactly the class of bug this encapsulation removes.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int> shape);                        // zero-initialized
    Tensor(std::vector<int> shape, std::vector<float> data);

    const std::vector<int>& shape()   const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int  ndim()  const { return (int)shape_.size(); }
    int  numel() const { return (int)data_.size(); }

    // Bulk data access. The buffer size is fixed; do not resize through these.
    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }
    float*       begin()       { return data_.data(); }
    float*       end()         { return data_.data() + data_.size(); }
    const float* begin() const { return data_.data(); }
    const float* end()   const { return data_.data() + data_.size(); }

    float&       operator[](int i)       { return data_[i]; }
    float        operator[](int i) const { return data_[i]; }

    // Multi-index element access — bounds checked.
    float&       at(const std::vector<int>& idx);
    float        at(const std::vector<int>& idx) const;

    static std::vector<int> make_strides(const std::vector<int>& shape);

private:
    std::vector<int>   shape_;
    std::vector<int>   strides_;  // element strides, row-major
    std::vector<float> data_;
};

// --- Enums ---
enum class MapOp    { ReLU, Softmax, Step, Sigmoid };  // Step: x > 0 ? 1 : 0
enum class ReduceOp { Sum, Mean, Max, Min };

// Storage precision for device-side buffers. Host-side Tensors are always
// f32. Backends that support f16 may downconvert at the host->device boundary
// (and upconvert on read) when a graph is declared F16. F32 is the default
// and the only mode CPU backends support.
enum class Precision { F32, F16 };

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
class MaxPool2DNode;
class UpsampleNearestNode;
class ConcatNode;
class BatchNorm2DNode;
class AssignNode;

// --- Visitor interface ---
class Visitor {
public:
    virtual ~Visitor() = default;
    // Called by ComputeGraph::accept() once before any visit(), so backends
    // can adjust kernel/buffer choices based on the graph's declared
    // precision. Default no-op for visitors that don't care (e.g. CPU
    // executor, printer, autograd builder).
    virtual void on_precision(Precision /*p*/) {}
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
    virtual void visit(MaxPool2DNode& node) = 0;
    virtual void visit(UpsampleNearestNode& node) = 0;
    virtual void visit(ConcatNode&    node) = 0;
    virtual void visit(BatchNorm2DNode& node) = 0;
    virtual void visit(AssignNode&    node) = 0;
};

// --- Abstract Node ---
class Node {
public:
    virtual ~Node() = default;
    virtual void accept(Visitor& v) = 0;
    virtual std::vector<Node*> inputs() const { return {}; }
};

// Non-null Node*. Validates at construction so a null edge can never enter
// the graph. Implicit conversions in/out keep call-site code unchanged
// (g.emplace<MatMulNode>(x, W) continues to compile).
class NodeRef {
public:
    NodeRef(Node* p) : ptr_(p) {
        if (!p) throw std::runtime_error("NodeRef: input edge is null");
    }
    Node* get()        const { return ptr_; }
    operator Node*()   const { return ptr_; }
    Node* operator->() const { return ptr_; }
private:
    Node* ptr_;
};

// --- Concrete Nodes ---
// `is_constant` marks the tensor as immutable for the lifetime of a graph
// execution session: backends may upload the host data to the GPU once and
// then skip subsequent re-uploads. Use this for trained model weights /
// running statistics — anything that doesn't change between passes. For
// training-update targets (AssignNode destinations) and per-iteration
// inputs (the image batch), leave is_constant=false so the host value
// re-uploads each pass.
class InputNode : public Node {
public:
    InputNode(std::string name, Tensor tensor, bool is_constant = false)
        : name(std::move(name)), tensor(std::move(tensor)), is_constant(is_constant) {}
    void accept(Visitor& v) override { v.visit(*this); }

    std::string name;
    Tensor      tensor;
    bool        is_constant;
};

class MatMulNode : public Node {
public:
    MatMulNode(NodeRef lhs, NodeRef rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class MatAddNode : public Node {
public:
    MatAddNode(NodeRef lhs, NodeRef rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class HadamardNode : public Node {  // element-wise multiply, same shape
public:
    HadamardNode(NodeRef lhs, NodeRef rhs) : lhs(lhs), rhs(rhs) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {lhs, rhs}; }

    Node* lhs;
    Node* rhs;
};

class MapNode : public Node {
public:
    MapNode(NodeRef input, MapOp op) : input(input), op(op) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*  input;
    MapOp  op;
};

class ScaleNode : public Node {
public:
    ScaleNode(NodeRef input, float scalar) : input(input), scalar(scalar) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*  input;
    float  scalar;
};

class ReduceNode : public Node {
public:
    ReduceNode(NodeRef input, ReduceOp op, int axis)
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
    explicit TransposeNode(NodeRef input, std::vector<int> perm = {})
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
    BroadcastNode(NodeRef input, int axis, int count)
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
    ReshapeNode(NodeRef input, std::vector<int> input_shape, std::vector<int> new_shape)
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
    Im2ColNode(NodeRef input, std::vector<int> input_shape,
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
    Col2ImNode(NodeRef input, std::vector<int> output_shape, int kH, int kW, int stride, int pad)
        : input(input), output_shape(std::move(output_shape)),
          kH(kH), kW(kW), stride(stride), pad(pad) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> output_shape;   // [N, C, H, W]
    int              kH, kW, stride, pad;
};

// 2D max pool over [N, C, H, W] with square kernel `k`, stride `s`, padding `pad`.
// Output: [N, C, (H + 2*pad - k)/s + 1, (W + 2*pad - k)/s + 1].
// Padded positions are treated as -infinity (so they never win the max).
// Inference-only — backward would need to remember argmax indices.
class MaxPool2DNode : public Node {
public:
    MaxPool2DNode(NodeRef input, std::vector<int> input_shape, int k, int stride, int pad = 0)
        : input(input), input_shape(std::move(input_shape)), k(k), stride(stride), pad(pad) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> input_shape;   // [N, C, H, W]
    int              k, stride, pad;
};

// Nearest-neighbor 2D upsample over [N, C, H, W] by integer factor.
// Output is [N, C, H*scale, W*scale]. No learnable params.
class UpsampleNearestNode : public Node {
public:
    UpsampleNearestNode(NodeRef input, std::vector<int> input_shape, int scale)
        : input(input), input_shape(std::move(input_shape)), scale(scale) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {input}; }

    Node*            input;
    std::vector<int> input_shape;   // [N, C, H, W]
    int              scale;
};

// Concat two 4D tensors along the channel axis (axis=1).
// Both inputs must share [N, *, H, W]; output is [N, Ca+Cb, H, W].
// Restricted to channel-axis + two inputs because that's all U-Net needs;
// keeping the surface small avoids carrying a per-call axis arg into kernels.
class ConcatNode : public Node {
public:
    ConcatNode(NodeRef a, NodeRef b, std::vector<int> a_shape, std::vector<int> b_shape)
        : a(a), b(b), a_shape(std::move(a_shape)), b_shape(std::move(b_shape)) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {a, b}; }

    Node*            a;
    Node*            b;
    std::vector<int> a_shape;  // [N, Ca, H, W]
    std::vector<int> b_shape;  // [N, Cb, H, W]
};

// 2D BatchNorm in inference mode. Per channel of a [N, C, H, W] input:
//   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
// gamma, beta, running_mean, running_var are each [C]. eps is a fixed scalar.
// Training-mode backward is not implemented — this primitive exists to load
// already-trained weights (the four per-channel tensors).
class BatchNorm2DNode : public Node {
public:
    BatchNorm2DNode(NodeRef input, NodeRef gamma, NodeRef beta,
                    NodeRef running_mean, NodeRef running_var, float eps = 1e-5f)
        : input(input), gamma(gamma), beta(beta),
          running_mean(running_mean), running_var(running_var), eps(eps) {}
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override {
        return {input, gamma, beta, running_mean, running_var};
    }

    Node* input;
    Node* gamma;
    Node* beta;
    Node* running_mean;
    Node* running_var;
    float eps;
};

// In-place assignment: copies `value` into `target`'s tensor (host) and into
// the executor's device buffer for `target`. Used by SGD to write updated
// parameters back into the layer's InputNode in one accept() pass — no
// post-run refresh() needed. The next run's visit(InputNode) re-uploads the
// updated value from the host tensor.
//
// Both `target` and `value` are listed as inputs so topo order visits them
// before this node (target's buffer must exist; value must be computed).
class AssignNode : public Node {
public:
    AssignNode(InputNode* target, NodeRef value) : target(target), value(value) {
        if (!target) throw std::runtime_error("AssignNode: target is null");
    }
    void accept(Visitor& v) override { v.visit(*this); }
    std::vector<Node*> inputs() const override { return {target, value}; }

    InputNode* target;
    Node*      value;
};

// --- ComputeGraph ---
class ComputeGraph {
public:
    // Precision is declared at graph construction. Backends that support it
    // (currently: metal::Executor) will lay out device buffers and dispatch
    // kernels in the requested dtype. F32 is the default for backward compat.
    explicit ComputeGraph(Precision precision = Precision::F32) : precision_(precision) {}

    Precision precision() const { return precision_; }

    template<typename T, typename... Args>
    T* emplace(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = node.get();
        nodes_.push_back(std::move(node));
        return ptr;
    }

    bool is_dag() const;
    void accept(Visitor& v);              // visits every node
    void accept(Visitor& v, Node* root);  // only nodes that `root` depends on

private:
    Precision                          precision_;
    std::vector<std::unique_ptr<Node>> nodes_;
};

} // namespace cg
