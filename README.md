# metal_ml

A small tensor compute graph library with reverse-mode autograd, two backends
(CPU, Metal), and end-to-end examples (MNIST classifiers, pretrained image
segmentation). Written to be readable front-to-back rather than to compete
with production frameworks.

## Layout

```
lib/compute_graph/
  cg.h, cg.cpp          ComputeGraph, Tensor, primitive node types,
                        cg::Precision { F32, F16 }
  nn.h                  Low-level subgraph emitters AND PyTorch-style
                        Module API (Tensor, Linear, Conv2D, BatchNorm2D,
                        MaxPool2D, UpsampleNearest, Sequential, ...)
  autograd.h            Reverse-mode autograd via BackwardBuilder
  printer.h             Visitor that dumps the graph for debugging
  backends/cpu/         Single-thread tiled CPU executor
  backends/metal/       Metal-CPP executor + .metal kernels, including
                        Metal 4 `mpp::tensor_ops::matmul2d` (Apple's
                        matrix-hardware GEMM primitive)

example/
  common/               Shared MNIST loader (auto-downloaded)
  mlp_classification/   784→128→10 MLP, trains on CPU and Metal
  cnn_classification/   Conv2D + Linear, trains on Metal
  segmentation/         Pretrained U-Net (ResNet34 encoder, smp decoder)
                        for human segmentation; safetensors loader; runs
                        in F16 with MPP matmul

compare/                PyTorch reference scripts (uv inline-deps),
                        IoU/accuracy parity check vs our masks

tests/unit_tests/       Primitive correctness, CPU/Metal parity,
                        autograd vs finite-difference, MPP and F16
                        end-to-end chain parity

app/                    Matmul kernel benchmark (naive vs tiled vs
                        SIMD vs SIMD+tiled vs MPP)
```

## Build

macOS 26+ for the Metal 4 toolchain (MPP / tensor APIs). Xcode 17+, CMake 3.25+,
[uv](https://docs.astral.sh/uv/) for the segmentation example's one-shot
`.pth → .safetensors` conversion.

```
cmake -B build -G Ninja
cmake --build build
```

At configure time, CMake fetches:

- Metal-CPP headers into `third_party/metal-cpp/`
- The MNIST dataset (mlp/cnn examples)
- The pretrained U-Net weights + sample images (segmentation example)

## Concepts

The library is built around primitive node types — `MatMul`, `MatAdd`,
`Hadamard`, `Map`, `Scale`, `Reduce`, `Transpose`, `Broadcast`, `Reshape`,
`Im2Col`, `Col2Im`, `MaxPool2D`, `UpsampleNearest`, `Concat`, `BatchNorm2D`,
`Assign`. Higher-level operations like `Conv2D` are *compositions of
primitives* that emit subgraphs, not new node types.

A `Visitor` dispatches per node type. There are three: `cpu::Executor`,
`metal::Executor`, and `PrintVisitor`. Adding a backend means writing one
new visitor.

Reverse-mode autograd is itself a Visitor: `autograd::BackwardBuilder`
walks the graph in reverse topological order from a seeded gradient and
emits gradient + parameter-update subgraphs into the **same** graph.
Forward, backward, and SGD updates execute in a single `accept()` pass.

## Precision (Metal-only)

`ComputeGraph` takes a `cg::Precision` at construction. The Metal executor
reads it via `Visitor::on_precision()` and allocates device buffers + picks
kernel variants accordingly. Host-side `cg::Tensor`s are always f32;
conversion happens at upload / download boundaries.

```cpp
cg::ComputeGraph g(cg::Precision::F16);    // half device storage throughout
```

At F16 the Metal executor auto-selects `MatMul::MPP`, which uses Metal 4's
`mpp::tensor_ops::matmul2d` directly against the half buffers (no
conversion kernels needed).

## PyTorch-style API

Layer modules construct without a graph; parameters register lazily on the
first `forward()`. `cg::nn::Tensor` wraps `(graph*, node*, shape)` so users
don't thread shapes by hand.

```cpp
#include "cg.h"
#include "nn.h"
#include "metal_executor.h"

using namespace cg;

struct CNN {
    nn::Conv2D conv1{1, 8, /*kernel=*/3, /*stride=*/2, /*padding=*/1};
    nn::Linear fc{1568, 10};

    explicit CNN(std::mt19937& rng) {
        conv1.reset_parameters(rng);
        fc.reset_parameters(rng);
    }

    nn::Tensor forward(nn::Tensor x) {
        return fc(nn::relu(conv1(x)));
    }

    std::vector<Node*> params() const {
        auto p = conv1.params();
        for (auto* n : fc.params()) p.push_back(n);
        return p;
    }

    template <typename BB>
    void apply_sgd(BB& bb, float lr) {
        conv1.apply_sgd(bb, lr);
        fc.apply_sgd(bb, lr);
    }
};
```

Or with `nn::Sequential`:

```cpp
nn::Sequential model;
model.add<nn::Conv2D>(3, 16, 3, 1, 1)
     .add<nn::BatchNorm2D>(16)
     .add<nn::ReLU>()
     .add<nn::MaxPool2D>(2);
auto y = model(x);                          // chains modules
```

Training step (one persistent graph):

```cpp
ComputeGraph g;
auto X_in  = nn::input(g, "X",  X_tensor);
auto OH_in = g.emplace<InputNode>("OH", one_hot);

auto logits = model.forward(X_in);
auto y      = nn::softmax(logits);

auto* dz = nn::softmax_ce_backward(g, y.node(), OH_in, batch);
autograd::BackwardBuilder bb(g);
bb.seed(logits.node(), dz);
bb.build(model.params());
model.apply_sgd(bb, lr);

cg::metal::Executor exec;
g.accept(exec);                             // forward + backward + SGD in one pass
```

For inference where weights don't change, mark `InputNode`s `is_constant=true`
so the backend skips host→device upload after the first pass:

```cpp
auto* w = g.emplace<InputNode>("W", weight_tensor, /*is_constant=*/true);
```

## Examples

```
./build/example/mlp_classification/mlp_classification
./build/example/cnn_classification/cnn_classification
./build/example/segmentation/segmentation example/segmentation/samples/person_yoga.jpg
./build/example/segmentation/segmentation --bench 20 example/segmentation/samples/person_yoga.jpg
```

The segmentation example loads a pretrained SMP-Unet
(`FiniUdesa/unet-human-segmentation`, ResNet34 encoder + smp decoder, 1
class) and runs in F16 with MPP matmul. Output PNGs go to
`example/segmentation/outputs/<run_name>/{input,mask,overlay}.png`.

To compare against PyTorch (CPU + MPS), run the reference scripts in
`compare/`:

```
./compare/unet_pytorch.py --bench 20 --device mps
./compare/parity_check.py                              # IoU vs our masks
```

## Tests

```
./build/tests/unit_tests/unit_tests
```

Covers:

- Primitive correctness against hand-computed expected values
- CPU vs Metal numerical parity for every primitive
- Autograd correctness via finite-difference comparisons (including the
  fused-bias Conv2D path)
- MPP matmul parity vs the CPU reference
- F16 end-to-end chain parity vs F32

## Notes

Conv2D is implemented via `Im2Col` + matmul, which lets the library reuse
its matmul kernels rather than maintain a separate direct-conv path —
same approach as cuDNN's implicit GEMM. The Metal backend picks
`MatMul::Tiled` by default and `MatMul::MPP` when the graph is F16; MPP
uses Apple's matrix hardware via `mpp::tensor_ops::matmul2d` and handles
edge tiles automatically.

The library has no external runtime dependencies aside from Metal itself,
metal-cpp (auto-fetched), and a C++20 compiler.

## License

MIT. See [LICENSE](LICENSE).
