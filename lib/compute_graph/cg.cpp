#include "cg.h"
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cg {

std::string version() { return VERSION; }

// --- Tensor ---

std::vector<int> Tensor::make_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    int s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
    return strides;
}

int Tensor::numel() const {
    if (shape.empty()) return 0;
    for (int d : shape)
        if (d < 0) throw std::runtime_error("Tensor: negative dimension in shape");
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
}

Tensor::Tensor(std::vector<int> shape)
    : shape(shape)
    , strides(make_strides(shape))
    , data(numel(), 0.0f)
{}

Tensor::Tensor(std::vector<int> shape_, std::vector<float> data_)
    : shape(std::move(shape_))
    , strides(make_strides(shape))
    , data(std::move(data_))
{
    int n = numel();
    if ((int)data.size() != n)
        throw std::runtime_error(
            "Tensor: data size " + std::to_string(data.size()) +
            " does not match shape numel " + std::to_string(n));
}

static int compute_offset(const std::vector<int>& shape,
                          const std::vector<int>& strides,
                          const std::vector<int>& idx) {
    if (idx.size() != shape.size())
        throw std::runtime_error(
            "Tensor::at: index rank " + std::to_string(idx.size()) +
            " does not match tensor rank " + std::to_string(shape.size()));
    int offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape[i])
            throw std::runtime_error(
                "Tensor::at: index " + std::to_string(idx[i]) +
                " out of bounds for axis " + std::to_string(i) +
                " of size " + std::to_string(shape[i]));
        offset += idx[i] * strides[i];
    }
    return offset;
}

float& Tensor::at(const std::vector<int>& idx) {
    return data[compute_offset(shape, strides, idx)];
}

float Tensor::at(const std::vector<int>& idx) const {
    return data[compute_offset(shape, strides, idx)];
}

// --- ComputeGraph ---

// Iterative post-order DFS. Stack frames are (node, child_iter_index); we
// push children one at a time so we don't blow the C stack on deep graphs
// (a 50-layer net with autograd + SGD comfortably exceeds a recursive limit).
static void topo_post_order(Node* root,
                            std::unordered_set<Node*>& visited,
                            std::vector<Node*>& order) {
    if (!root) throw std::runtime_error("ComputeGraph: null root in traversal");
    if (!visited.insert(root).second) return;

    struct Frame { Node* node; std::vector<Node*> ins; size_t i; };
    std::vector<Frame> stack;
    stack.push_back({root, root->inputs(), 0});

    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.i < f.ins.size()) {
            Node* dep = f.ins[f.i++];
            if (!dep) throw std::runtime_error("ComputeGraph: null input edge");
            if (visited.insert(dep).second)
                stack.push_back({dep, dep->inputs(), 0});
        } else {
            order.push_back(f.node);
            stack.pop_back();
        }
    }
}

bool ComputeGraph::is_dag() const {
    enum Color { White, Gray, Black };
    std::unordered_map<Node*, Color> color;
    for (auto& n : nodes_) color[n.get()] = White;

    struct Frame { Node* node; std::vector<Node*> ins; size_t i; };
    for (auto& root : nodes_) {
        if (color[root.get()] != White) continue;
        std::vector<Frame> stack;
        stack.push_back({root.get(), root->inputs(), 0});
        color[root.get()] = Gray;
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.i < f.ins.size()) {
                Node* dep = f.ins[f.i++];
                if (!dep) return false;
                Color c = color[dep];
                if (c == Gray) return false;
                if (c == White) {
                    color[dep] = Gray;
                    stack.push_back({dep, dep->inputs(), 0});
                }
            } else {
                color[f.node] = Black;
                stack.pop_back();
            }
        }
    }
    return true;
}

void ComputeGraph::accept(Visitor& v) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> order;
    for (auto& n : nodes_) topo_post_order(n.get(), visited, order);
    for (Node* n : order)  n->accept(v);
}

void ComputeGraph::accept(Visitor& v, Node* root) {
    if (!root) throw std::runtime_error("ComputeGraph::accept: root is null");
    std::unordered_set<Node*> visited;
    std::vector<Node*> order;
    topo_post_order(root, visited, order);
    for (Node* n : order) n->accept(v);
}

} // namespace cg
