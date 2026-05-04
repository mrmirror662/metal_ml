#include "cg.h"
#include "cpu.h"
#include "metal_executor.h"
#include "printer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

static cg::Tensor random_tensor(std::vector<int> shape, unsigned seed = 42,
                                float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    cg::Tensor t(shape);
    for (auto& v : t) v = dist(rng);
    return t;
}

static std::string shape_str(const std::vector<int>& shape) {
    std::string s = "[";
    for (int i = 0; i < (int)shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < (int)shape.size()) s += " x ";
    }
    return s + "]";
}

template <typename Exec>
static double time_run(cg::ComputeGraph& g, Exec& exec) {
    auto t0 = std::chrono::high_resolution_clock::now();
    g.accept(exec);
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

static double max_abs_diff(const cg::Tensor& a, const cg::Tensor& b) {
    double m = 0.0;
    for (int i = 0; i < a.numel(); ++i)
        m = std::max(m, (double)std::abs(a[i] - b[i]));
    return m;
}

static void print_softmax_preview(const cg::Tensor& t, int n_rows = 5) {
    int cols = t.shape().back();
    for (int r = 0; r < n_rows && r * cols < t.numel(); ++r) {
        std::cout << "    row " << r << ": ";
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            std::cout << t[r * cols + c] << " ";
            sum += t[r * cols + c];
        }
        std::cout << "(sum=" << sum << ")\n";
    }
}

struct GraphHandles {
    cg::Node* matmul_out;   // [B, M, 512] — heavy compute output
    cg::Node* probs;        // [B, M, 4]   — projection + softmax
};

static GraphHandles build_graph(cg::ComputeGraph& graph, int B, int M, int K, int N) {
    // Xavier-ish init: stddev ~ 1/sqrt(fan_in) keeps activations bounded
    // through the chain instead of compounding to ±millions.
    constexpr float W_RANGE = 0.05f;
    constexpr float B_RANGE = 0.01f;

    auto* X      = graph.emplace<cg::InputNode>("X",      random_tensor({B, M, K},   1));
    auto* W1     = graph.emplace<cg::InputNode>("W1",     random_tensor({B, K, N},   2, -W_RANGE, W_RANGE));
    auto* b1     = graph.emplace<cg::InputNode>("b1",     random_tensor({B, M, N},   3, -B_RANGE, B_RANGE));
    auto* W2     = graph.emplace<cg::InputNode>("W2",     random_tensor({B, N, N},   4, -W_RANGE, W_RANGE));
    auto* b2     = graph.emplace<cg::InputNode>("b2",     random_tensor({B, M, N},   5, -B_RANGE, B_RANGE));
    auto* W3     = graph.emplace<cg::InputNode>("W3",     random_tensor({B, N, 512}, 6, -W_RANGE, W_RANGE));
    auto* W_proj = graph.emplace<cg::InputNode>("W_proj", random_tensor({B, 512, 4}, 7, -W_RANGE, W_RANGE));

    auto* h1     = graph.emplace<cg::MatMulNode>(X, W1);
    auto* h1_add = graph.emplace<cg::MatAddNode>(h1, b1);
    auto* h1_act = graph.emplace<cg::MapNode>(h1_add, cg::MapOp::ReLU);
    auto* h1_scl = graph.emplace<cg::ScaleNode>(h1_act, 0.5f);
    auto* h2     = graph.emplace<cg::MatMulNode>(h1_scl, W2);
    auto* h2_add = graph.emplace<cg::MatAddNode>(h2, b2);
    auto* h2_act = graph.emplace<cg::MapNode>(h2_add, cg::MapOp::ReLU);
    auto* mm_out = graph.emplace<cg::MatMulNode>(h2_act, W3);              // [B, M, 512]

    // Projection head: [B, M, 512] @ [B, 512, 4] -> [B, M, 4], then row-wise softmax.
    auto* logits = graph.emplace<cg::MatMulNode>(mm_out, W_proj);          // [B, M, 4]
    auto* probs  = graph.emplace<cg::MapNode>(logits, cg::MapOp::Softmax); // [B, M, 4]

    return {mm_out, probs};
}

int main() {
    constexpr int B = 8, M = 1024, K = 1024, N = 1024;

    cg::ComputeGraph graph;
    auto handles = build_graph(graph, B, M, K, N);

    cg::PrintVisitor printer;
    graph.accept(printer);
    std::cout << "\n";

    // --- CPU reference ---
    cg::cpu::Executor cpu_exec;
    double cpu_ms = time_run(graph, cpu_exec);
    cg::Tensor cpu_out   = cpu_exec.result(handles.matmul_out);  // copy for diff
    cg::Tensor cpu_probs = cpu_exec.result(handles.probs);

    // --- Metal: four matmul implementations ---
    auto bench_metal = [&](const char* label, cg::metal::Executor::MatMul impl) {
        cg::metal::Executor exec(impl);
        double ms = time_run(graph, exec);
        const auto& gpu_out   = exec.result(handles.matmul_out);
        const auto& gpu_probs = exec.result(handles.probs);
        double diff = max_abs_diff(cpu_out, gpu_out);
        std::cout << label << ": " << ms << " ms"
                  << "  (speedup " << (cpu_ms / ms) << "x"
                  << ", max_abs_diff " << diff << ")\n";
        std::cout << "    softmax " << shape_str(gpu_probs.shape()) << " (first 5 rows):\n";
        print_softmax_preview(gpu_probs);
        return ms;
    };

    cg::metal::Executor probe(cg::metal::Executor::MatMul::Naive);
    std::cout << "[Metal] device: " << probe.device_name() << "\n";
    std::cout << "matmul output shape: " << shape_str(cpu_out.shape()) << "\n\n";

    std::cout << "CPU              : " << cpu_ms << " ms\n";
    std::cout << "    softmax " << shape_str(cpu_probs.shape()) << " (first 5 rows):\n";
    print_softmax_preview(cpu_probs);
    std::cout << "\n";

    bench_metal("Metal naive      ", cg::metal::Executor::MatMul::Naive);
    bench_metal("Metal tiled      ", cg::metal::Executor::MatMul::Tiled);
    bench_metal("Metal simd       ", cg::metal::Executor::MatMul::SIMD);
    bench_metal("Metal simd+tiled ", cg::metal::Executor::MatMul::SimdTiled);

    return 0;
}
