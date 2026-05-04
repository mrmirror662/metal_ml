# compare/

PyTorch reference implementations matching the examples in `example/`,
used to sanity-check that our results (loss curve, accuracy) are close
to a well-known framework.

The Python scripts use [uv](https://github.com/astral-sh/uv) with PEP 723
inline metadata — no virtualenv setup required.

## Run

First build and run our CNN to download MNIST and produce baseline numbers:

```
cmake --build ../build
../build/example/cnn_classification/cnn_classification
```

Then run the PyTorch mirror against the same dataset:

```
uv run cnn_pytorch.py
```

`uv` reads the dependency block at the top of the script and fetches
torch/numpy on first run; subsequent runs are instant.

## Architecture parity

Both implementations use the same model, hyperparameters, and dataset:

| | cg | PyTorch |
|---|---|---|
| Conv1   | `nn::Conv2D(1, 8, 3, 3, stride=2, pad=1)` | `nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False)` |
| ReLU    | `nn::relu`                                | `F.relu`                                 |
| Flatten | `ReshapeNode -> [N, 1568]`                | `x.flatten(1)`                           |
| Linear  | `nn::Dense(1568, 10)`                     | `nn.Linear(1568, 10)`                    |
| Loss    | softmax + cross-entropy                   | `F.cross_entropy`                        |
| Init    | He (fan_in)                               | He (fan_in), `nn.init.normal_`           |
| Optim   | plain SGD, lr=0.1                         | `torch.optim.SGD(lr=0.1)`                |

Different RNGs are used for init and batch shuffling
(`std::mt19937` vs `numpy.random.default_rng`), so the exact training
trajectories differ but the final accuracy and curves match within ~0.2%.
