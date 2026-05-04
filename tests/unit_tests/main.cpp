// Unit tests for cg primitives, CPU/Metal parity, and autograd correctness.
//
// Each TEST() block builds a small graph, runs it on CPU (and Metal where
// applicable), and verifies output against either a hand-computed result or
// a numerical reference. Autograd tests use finite differences as ground truth.

#include "test_framework.h"

#include "cg.h"
#include "nn.h"
#include "autograd.h"
#include "cpu.h"
#include "metal_executor.h"

#include <cmath>
#include <random>
#include <vector>

using namespace cg;

// =============================================================================
// Helpers
// =============================================================================

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    EXPECT_TRUE(a.shape() == b.shape());
    double m = 0.0;
    for (int i = 0; i < a.numel(); ++i)
        m = std::max(m, (double)std::abs(a[i] - b[i]));
    return m;
}

static Tensor mk(std::vector<int> shape, std::vector<float> data) {
    return Tensor(std::move(shape), std::move(data));
}

static Tensor random_tensor(std::vector<int> shape, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    Tensor t(shape);
    for (auto& v : t) v = dist(rng);
    return t;
}

template <typename Build>
static Tensor run_cpu(Build build) {
    ComputeGraph g;
    Node* out = build(g);
    cpu::Executor e;
    g.accept(e);
    return e.result(out);
}

template <typename Build>
static Tensor run_metal(Build build) {
    ComputeGraph g;
    Node* out = build(g);
    metal::Executor e;
    g.accept(e);
    return e.result(out);
}

// Verify CPU result matches expected, AND Metal matches CPU
template <typename Build>
static void check_cpu_and_metal(Build build, const Tensor& expected, double tol = 1e-5) {
    Tensor cpu = run_cpu(build);
    EXPECT_TRUE(cpu.shape() == expected.shape());
    EXPECT_NEAR(max_abs_diff(cpu, expected), 0.0, tol);

    Tensor met = run_metal(build);
    EXPECT_TRUE(met.shape() == cpu.shape());
    EXPECT_NEAR(max_abs_diff(met, cpu), 0.0, tol);
}

// =============================================================================
// Tensor basics
// =============================================================================

TEST(tensor_numel_and_strides) {
    Tensor t({2, 3, 4});
    EXPECT_TRUE(t.numel() == 24);
    EXPECT_TRUE(t.strides() == std::vector<int>({12, 4, 1}));
}

// =============================================================================
// Forward primitive correctness + CPU/Metal parity
// =============================================================================

TEST(matadd_correctness) {
    Tensor A = mk({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor B = mk({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor expected = mk({2, 3}, {11, 22, 33, 44, 55, 66});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<MatAddNode>(
            g.emplace<InputNode>("a", A),
            g.emplace<InputNode>("b", B));
    }, expected);
}

TEST(matmul_correctness_2d) {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    Tensor A = mk({2, 2}, {1, 2, 3, 4});
    Tensor B = mk({2, 2}, {5, 6, 7, 8});
    Tensor expected = mk({2, 2}, {19, 22, 43, 50});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<MatMulNode>(
            g.emplace<InputNode>("a", A),
            g.emplace<InputNode>("b", B));
    }, expected);
}

TEST(matmul_batched_3d) {
    // 2 batches of 2x2 matmul
    Tensor A = mk({2, 2, 2}, {1, 2, 3, 4,    5, 6, 7, 8});
    Tensor B = mk({2, 2, 2}, {1, 0, 0, 1,    2, 0, 0, 2});  // identity, then 2I
    Tensor expected = mk({2, 2, 2}, {1, 2, 3, 4,   10, 12, 14, 16});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<MatMulNode>(
            g.emplace<InputNode>("a", A),
            g.emplace<InputNode>("b", B));
    }, expected);
}

TEST(hadamard_correctness) {
    Tensor A = mk({2, 2}, {1, 2, 3, 4});
    Tensor B = mk({2, 2}, {5, 6, 7, 8});
    Tensor expected = mk({2, 2}, {5, 12, 21, 32});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<HadamardNode>(
            g.emplace<InputNode>("a", A),
            g.emplace<InputNode>("b", B));
    }, expected);
}

TEST(scale_correctness) {
    Tensor X = mk({3}, {1, -2, 3});
    Tensor expected = mk({3}, {2.5f, -5.0f, 7.5f});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ScaleNode>(g.emplace<InputNode>("x", X), 2.5f);
    }, expected);
}

TEST(map_relu) {
    Tensor X = mk({4}, {-2, -0.5f, 0, 1.5f});
    Tensor expected = mk({4}, {0, 0, 0, 1.5f});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<MapNode>(g.emplace<InputNode>("x", X), MapOp::ReLU);
    }, expected);
}

TEST(map_step) {
    Tensor X = mk({4}, {-1, 0, 1e-6f, 5});
    Tensor expected = mk({4}, {0, 0, 1, 1});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<MapNode>(g.emplace<InputNode>("x", X), MapOp::Step);
    }, expected);
}

TEST(map_softmax_rows_sum_to_one) {
    Tensor X = random_tensor({4, 6}, 1);
    Tensor cpu = run_cpu([&](ComputeGraph& g) {
        return g.emplace<MapNode>(g.emplace<InputNode>("x", X), MapOp::Softmax);
    });
    for (int r = 0; r < 4; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 6; ++c) sum += cpu[r * 6 + c];
        EXPECT_NEAR(sum, 1.0, 1e-5);
    }
    Tensor met = run_metal([&](ComputeGraph& g) {
        return g.emplace<MapNode>(g.emplace<InputNode>("x", X), MapOp::Softmax);
    });
    EXPECT_NEAR(max_abs_diff(cpu, met), 0.0, 1e-5);
}

TEST(reduce_sum_axis0) {
    Tensor X = mk({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor expected = mk({1, 3}, {5, 7, 9});  // sum down rows
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ReduceNode>(g.emplace<InputNode>("x", X), ReduceOp::Sum, 0);
    }, expected);
}

TEST(reduce_mean_axis1) {
    Tensor X = mk({2, 4}, {1, 2, 3, 4,   5, 6, 7, 8});
    Tensor expected = mk({2, 1}, {2.5f, 6.5f});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ReduceNode>(g.emplace<InputNode>("x", X), ReduceOp::Mean, 1);
    }, expected);
}

TEST(reduce_max_min) {
    Tensor X = mk({2, 3}, {1, 5, 2,    8, 3, 6});
    Tensor max_expected = mk({2, 1}, {5, 8});
    Tensor min_expected = mk({2, 1}, {1, 3});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ReduceNode>(g.emplace<InputNode>("x", X), ReduceOp::Max, 1);
    }, max_expected);
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ReduceNode>(g.emplace<InputNode>("x", X), ReduceOp::Min, 1);
    }, min_expected);
}

TEST(transpose_2d) {
    Tensor X = mk({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor expected = mk({3, 2}, {1, 4, 2, 5, 3, 6});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<TransposeNode>(g.emplace<InputNode>("x", X));
    }, expected);
}

TEST(transpose_3d_perm) {
    // Swap dims 1 and 2: [2, 3, 4] -> [2, 4, 3]
    Tensor X = random_tensor({2, 3, 4}, 7);
    Tensor cpu = run_cpu([&](ComputeGraph& g) {
        return g.emplace<TransposeNode>(g.emplace<InputNode>("x", X), std::vector<int>{0, 2, 1});
    });
    EXPECT_TRUE(cpu.shape() == std::vector<int>({2, 4, 3}));
    // Spot-check: cpu[a, b, c] should equal X[a, c, b]
    EXPECT_NEAR(cpu[(0 * 4 + 2) * 3 + 1], X[(0 * 3 + 1) * 4 + 2], 1e-6);

    Tensor met = run_metal([&](ComputeGraph& g) {
        return g.emplace<TransposeNode>(g.emplace<InputNode>("x", X), std::vector<int>{0, 2, 1});
    });
    EXPECT_NEAR(max_abs_diff(cpu, met), 0.0, 1e-6);
}

TEST(broadcast) {
    // [1, 3] broadcast axis=0, count=4 -> [4, 3] with each row identical
    Tensor X = mk({1, 3}, {7, 8, 9});
    Tensor expected = mk({4, 3}, {7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9});
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<BroadcastNode>(g.emplace<InputNode>("x", X), 0, 4);
    }, expected);
}

TEST(reshape) {
    Tensor X = mk({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor expected = mk({3, 2}, {1, 2, 3, 4, 5, 6});  // same data, new shape
    check_cpu_and_metal([&](ComputeGraph& g) {
        return g.emplace<ReshapeNode>(g.emplace<InputNode>("x", X),
            std::vector<int>{2, 3}, std::vector<int>{3, 2});
    }, expected);
}

TEST(im2col_basic) {
    // 1x1 image, 1 channel, 1x1 kernel — output is just the input value
    Tensor X = mk({1, 1, 2, 2}, {1, 2, 3, 4});
    // im2col with 2x2 kernel, stride=1, pad=0: 1 patch of size 4
    Tensor cpu = run_cpu([&](ComputeGraph& g) {
        return g.emplace<Im2ColNode>(g.emplace<InputNode>("x", X),
            std::vector<int>{1, 1, 2, 2}, 2, 2, 1, 0);
    });
    EXPECT_TRUE(cpu.shape() == std::vector<int>({1, 4}));
    // The single patch is the whole image
    EXPECT_NEAR(cpu[0], 1, 1e-6);
    EXPECT_NEAR(cpu[3], 4, 1e-6);

    Tensor met = run_metal([&](ComputeGraph& g) {
        return g.emplace<Im2ColNode>(g.emplace<InputNode>("x", X),
            std::vector<int>{1, 1, 2, 2}, 2, 2, 1, 0);
    });
    EXPECT_NEAR(max_abs_diff(cpu, met), 0.0, 1e-6);
}

TEST(im2col_col2im_parity) {
    // im2col then col2im with the same params should accumulate values.
    Tensor X = random_tensor({2, 3, 8, 8}, 11);
    auto build = [&](ComputeGraph& g) {
        auto* x  = g.emplace<InputNode>("x", X);
        auto* p  = g.emplace<Im2ColNode>(x, std::vector<int>{2, 3, 8, 8}, 3, 3, 1, 1);
        return    g.emplace<Col2ImNode>(p, std::vector<int>{2, 3, 8, 8}, 3, 3, 1, 1);
    };
    Tensor cpu = run_cpu(build);
    Tensor met = run_metal(build);
    EXPECT_NEAR(max_abs_diff(cpu, met), 0.0, 1e-5);
}

TEST(conv2d_forward_parity) {
    Tensor X = random_tensor({2, 3, 8, 8}, 1);
    Tensor K = random_tensor({4, 3, 3, 3}, 2);
    auto build = [&](ComputeGraph& g) {
        auto* x = g.emplace<InputNode>("x", X);
        auto* k = g.emplace<InputNode>("k", K);
        return nn::conv2d(g, x, k,
            std::vector<int>{2, 3, 8, 8},
            std::vector<int>{4, 3, 3, 3}, 1, 1);
    };
    Tensor cpu = run_cpu(build);
    EXPECT_TRUE(cpu.shape() == std::vector<int>({2, 4, 8, 8}));
    Tensor met = run_metal(build);
    EXPECT_NEAR(max_abs_diff(cpu, met), 0.0, 1e-5);
}

// =============================================================================
// Autograd correctness via numerical gradient
// =============================================================================

// Compute analytical gradient: dL/dX where L = sum(build(X)).
template <typename Build>
static Tensor analytical_grad(Build build, const Tensor& X) {
    // First, run forward to get output shape
    ComputeGraph g0;
    auto* x0 = g0.emplace<InputNode>("x", X);
    Node* out0 = build(g0, x0);
    cpu::Executor e0;
    g0.accept(e0);
    auto out_shape = e0.result(out0).shape();

    // Build with autograd, seed with ones (gradient of sum())
    ComputeGraph g;
    auto* x = g.emplace<InputNode>("x", X);
    Node* out = build(g, x);
    Tensor seed(out_shape);
    for (auto& v : seed) v = 1.0f;
    auto* s = g.emplace<InputNode>("s", seed);

    autograd::BackwardBuilder bb(g);
    bb.seed(out, s);
    bb.build({x});

    cpu::Executor e;
    g.accept(e);
    return e.result(bb.grad(x));
}

// Compute numerical gradient via finite differences
template <typename Build>
static Tensor numerical_grad(Build build, const Tensor& X, float eps = 1e-3f) {
    auto eval_loss = [&](const Tensor& Xv) {
        ComputeGraph g;
        auto* x = g.emplace<InputNode>("x", Xv);
        Node* out = build(g, x);
        cpu::Executor e;
        g.accept(e);
        const auto& y = e.result(out);
        double s = 0;
        for (int i = 0; i < y.numel(); ++i) s += y[i];
        return s;
    };

    Tensor grad(X.shape());
    for (int i = 0; i < X.numel(); ++i) {
        Tensor Xp = X; Xp[i] += eps;
        Tensor Xm = X; Xm[i] -= eps;
        grad[i] = (eval_loss(Xp) - eval_loss(Xm)) / (2.0 * eps);
    }
    return grad;
}

TEST(autograd_matmul_grad) {
    Tensor X = random_tensor({3, 4}, 5);
    Tensor W = random_tensor({4, 2}, 6);
    auto build = [&](ComputeGraph& g, Node* x) {
        auto* w = g.emplace<InputNode>("w", W);
        return g.emplace<MatMulNode>(x, w);
    };
    Tensor a = analytical_grad(build, X);
    Tensor n = numerical_grad(build, X);
    EXPECT_NEAR(max_abs_diff(a, n), 0.0, 1e-3);
}

TEST(autograd_relu_grad) {
    // Avoid the kink at 0 — push values away so finite differences stay valid
    Tensor X = random_tensor({2, 3}, 9);
    for (auto& v : X) v = (v >= 0 ? v + 0.2f : v - 0.2f);
    auto build = [&](ComputeGraph& g, Node* x) {
        return g.emplace<MapNode>(x, MapOp::ReLU);
    };
    Tensor a = analytical_grad(build, X);
    Tensor n = numerical_grad(build, X);
    EXPECT_NEAR(max_abs_diff(a, n), 0.0, 1e-3);
}

TEST(autograd_chain_matmul_relu) {
    Tensor X = random_tensor({3, 4}, 1);
    Tensor W = random_tensor({4, 2}, 2);
    auto build = [&](ComputeGraph& g, Node* x) {
        auto* w  = g.emplace<InputNode>("w", W);
        auto* z  = g.emplace<MatMulNode>(x, w);
        return     g.emplace<MapNode>(z, MapOp::ReLU);
    };
    Tensor a = analytical_grad(build, X);
    Tensor n = numerical_grad(build, X);
    EXPECT_NEAR(max_abs_diff(a, n), 0.0, 1e-3);
}

TEST(autograd_conv2d_grad) {
    Tensor X = random_tensor({1, 2, 4, 4}, 3);
    Tensor K = random_tensor({3, 2, 3, 3}, 4);
    auto build = [&](ComputeGraph& g, Node* x) {
        auto* k = g.emplace<InputNode>("k", K);
        return nn::conv2d(g, x, k,
            std::vector<int>{1, 2, 4, 4},
            std::vector<int>{3, 2, 3, 3}, 1, 1);
    };
    Tensor a = analytical_grad(build, X);
    Tensor n = numerical_grad(build, X);
    EXPECT_NEAR(max_abs_diff(a, n), 0.0, 1e-2);
}

// =============================================================================

int main() {
    return test::run_all();
}
