# metal_ml

A small tensor compute graph library with reverse-mode autograd, two backends
(CPU, Metal), and end-to-end MNIST classifiers built on top. Written to be
readable from front to back rather than to compete with production frameworks.

## Layout

```
lib/compute_graph/
  cg.h, cg.cpp          ComputeGraph, Tensor, primitive node types
  nn.h                  Composable graph helpers (linear, conv2d, ...) and
                        layer classes (Dense, Conv2D)
  autograd.h            Reverse-mode autograd via BackwardBuilder
  printer.h             Visitor that dumps the graph for debugging
  backends/cpu/         Single-thread tiled CPU executor
  backends/metal/       Metal-CPP executor + .metal compute kernels

example/
  common/               Shared MNIST loader (dataset auto-downloaded)
  mlp_classification/   784-128-10 MLP, runs on CPU and Metal
  cnn_classification/   Conv2D + Dense, Metal-only

tests/unit_tests/       Primitive correctness, CPU/Metal parity,
                        autograd vs finite-difference gradient checks

app/                    Matmul kernel benchmark
```

## Build

Requirements: macOS, Xcode (for the Metal toolchain), CMake 3.25 or newer.

```
cmake -B build -G Ninja
cmake --build build
```

At configure time, CMake fetches:

- Metal-CPP headers into `third_party/metal-cpp/`
- The MNIST dataset into `build/example/common/data/`

## Concepts

The library is built around a small set of primitive node types
(`MatMul`, `MatAdd`, `Hadamard`, `Map`, `Scale`, `Reduce`, `Transpose`,
`Broadcast`, `Reshape`, `Im2Col`, `Col2Im`). Higher-level operations are
*compositions of primitives* that emit subgraphs, not new node types.

A `Visitor` dispatches per node type. There are three visitors in the
codebase: `cpu::Executor`, `metal::Executor`, and `PrintVisitor`. Adding
a new backend means writing one new visitor.

Reverse-mode autograd is itself a `Visitor`: `autograd::BackwardBuilder`
walks the graph in reverse topological order from a seeded gradient and
emits gradient subgraphs into the same graph. Forward, backward, and
parameter updates all execute in a single pass.

## Usage

A complete training step. Layers (`nn::Dense`) own their parameter
tensors so weights and biases are not declared by hand:

```cpp
#include "cg.h"
#include "nn.h"
#include "autograd.h"
#include "metal_executor.h"

using namespace cg;

std::mt19937 rng(42);
nn::Dense fc1(784, 128, rng);
nn::Dense fc2(128,  10, rng);

cg::metal::Executor exec;

// One training step
ComputeGraph g;
auto* x  = g.emplace<InputNode>("X",  X);   // [batch, 784]
auto* oh = g.emplace<InputNode>("OH", one_hot);

// Forward
auto* z1     = fc1(g, x, batch);
auto* a1     = nn::relu(g, z1);
auto* logits = fc2(g, a1, batch);
auto* y      = nn::softmax(g, logits);

// Reverse-mode autograd
auto* dz = nn::softmax_ce_backward(g, y, oh, batch);
autograd::BackwardBuilder bb(g);
bb.seed(logits, dz);

std::vector<Node*> params = fc1.params();
for (auto* p : fc2.params()) params.push_back(p);
bb.build(params);

// SGD update graph emitted alongside
fc1.apply_sgd(g, bb, lr);
fc2.apply_sgd(g, bb, lr);

// Single dispatch — runs forward, backward, and parameter updates
exec.clear();
g.accept(exec);

// Pull updated weights back into the layer state
fc1.refresh(exec);
fc2.refresh(exec);
```

A graph at this level prints as:

```
[0] Input "X" [64x784]
[1] Input "W" [784x128]
[2] Input "b" [1x128]
[3] MatMul (0, 1)
[4] Broadcast axis=0 count=64 (2)
[5] MatAdd (3, 4)
[6] Map ReLU (5)
...
```

## Examples

- `./build/example/mlp_classification/mlp_classification` — trains a
  784-128-10 MLP on MNIST on CPU and then on Metal. Reaches ~97% test
  accuracy in 5 epochs.
- `./build/example/cnn_classification/cnn_classification` — trains a
  small CNN (one Conv2D + one Dense) on MNIST on Metal.
- `./build/app/app` — benchmarks four matmul implementations
  (naive, tiled, SIMD-group, SIMD+tiled) against the tiled CPU
  reference.

## Tests

```
./build/tests/unit_tests/unit_tests
```

Covers:

- Primitive correctness against hand-computed expected values
- CPU/Metal numerical parity for every primitive
- Autograd correctness via finite-difference comparisons

## Notes

Conv2D is implemented via `Im2Col` + matmul, which lets the library reuse
its single optimized matmul kernel rather than maintaining a separate
direct-convolution path. This is the same approach as cuDNN's implicit
GEMM. The Metal backend prefers the tiled matmul by default; the
SIMD-group matrix variant is fastest but requires shapes divisible by 8.

The library has no external runtime dependencies aside from Metal itself
and a C++20 compiler.
