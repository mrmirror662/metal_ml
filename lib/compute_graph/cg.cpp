#include "cg.h"
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
}

Tensor::Tensor(std::vector<int> shape)
    : shape(shape)
    , strides(make_strides(shape))
    , data(numel(), 0.0f)
{}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data)
    : shape(shape)
    , strides(make_strides(shape))
    , data(std::move(data))
{}

float& Tensor::at(const std::vector<int>& idx) {
    int offset = 0;
    for (int i = 0; i < (int)idx.size(); ++i)
        offset += idx[i] * strides[i];
    return data[offset];
}

float Tensor::at(const std::vector<int>& idx) const {
    int offset = 0;
    for (int i = 0; i < (int)idx.size(); ++i)
        offset += idx[i] * strides[i];
    return data[offset];
}

// --- ComputeGraph ---

bool ComputeGraph::is_dag() const {
    enum Color { White, Gray, Black };
    std::unordered_map<Node*, Color> color;
    for (auto& n : nodes_)
        color[n.get()] = White;

    std::function<bool(Node*)> dfs = [&](Node* u) -> bool {
        color[u] = Gray;
        for (Node* dep : u->inputs()) {
            if (color[dep] == Gray)                   return false;
            if (color[dep] == White && !dfs(dep))     return false;
        }
        color[u] = Black;
        return true;
    };

    for (auto& n : nodes_)
        if (color[n.get()] == White && !dfs(n.get()))
            return false;
    return true;
}

void ComputeGraph::accept(Visitor& v) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> order;

    std::function<void(Node*)> dfs = [&](Node* u) {
        if (!visited.insert(u).second) return;
        for (Node* dep : u->inputs()) dfs(dep);
        order.push_back(u);
    };

    for (auto& n : nodes_) dfs(n.get());
    for (Node* n : order)  n->accept(v);
}

} // namespace cg
