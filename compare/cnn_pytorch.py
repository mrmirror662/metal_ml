#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
#
# PyTorch reference matching example/cnn_classification.
# Same architecture, optimizer, hyperparameters, and data path so the numbers
# are directly comparable to our cg implementation.
#
# Architecture:
#   X [N, 1, 28, 28]
#     -> Conv2D(1->8, 3x3, stride=2, pad=1, no bias)   -> [N, 8, 14, 14]
#     -> ReLU
#     -> Flatten                                        -> [N, 1568]
#     -> Linear(1568 -> 10)
#     -> Softmax
#
# Hyperparameters: batch=64, lr=0.1, plain SGD, 3 epochs, He init.

import gzip
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- MNIST loader (reuse the dataset already downloaded by CMake) --------
MNIST_DIR = Path(__file__).resolve().parent.parent / "build" / "example" / "common" / "data"


def load_idx(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:2] != b"\x00\x00":
        raise RuntimeError(f"bad IDX magic in {path}")
    typ = raw[2]
    nd  = raw[3]
    dims = struct.unpack(">" + "I" * nd, raw[4:4 + 4 * nd])
    if typ != 0x08:
        raise RuntimeError("only uint8 IDX supported")
    arr = np.frombuffer(raw[4 + 4 * nd:], dtype=np.uint8)
    return arr.reshape(dims)


def load_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = load_idx(MNIST_DIR / "train-images-idx3-ubyte").astype(np.float32) / 255.0
    train_y = load_idx(MNIST_DIR / "train-labels-idx1-ubyte").astype(np.int64)
    test_x  = load_idx(MNIST_DIR / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
    test_y  = load_idx(MNIST_DIR / "t10k-labels-idx1-ubyte").astype(np.int64)
    # [N, 28, 28] -> [N, 1, 28, 28]
    train_x = torch.from_numpy(train_x).unsqueeze(1)
    test_x  = torch.from_numpy(test_x).unsqueeze(1)
    return train_x, torch.from_numpy(train_y), test_x, torch.from_numpy(test_y)


# ---- Model ---------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # bias=False to match our cg model (which has no bias on the conv)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc    = nn.Linear(1568, 10)

        # He init to match our nn::Dense / nn::Conv2D
        torch.manual_seed(42)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=(2.0 / (1 * 3 * 3)) ** 0.5)
        nn.init.normal_(self.fc.weight,    mean=0.0, std=(2.0 / 1568) ** 0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = x.flatten(1)
        return self.fc(x)


# ---- Training loop -------------------------------------------------------
def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    train_x, train_y, test_x, test_y = load_dataset()
    print(f"  train: {len(train_y)}  test: {len(test_y)}")

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x  = test_x.to(device)
    test_y  = test_y.to(device)

    BATCH = 64
    EPOCHS = 3
    LR = 0.1

    model = CNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    n = len(train_y)
    rng = np.random.default_rng(42)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = rng.permutation(n)
        loss_sum = 0.0
        correct  = 0
        batches  = 0
        t0 = time.time()

        for b in range(0, n - BATCH + 1, BATCH):
            i = idx[b:b + BATCH]
            xb = train_x[i]
            yb = train_y[i]

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            correct  += (logits.argmax(1) == yb).sum().item()
            batches  += 1

        secs = time.time() - t0

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(test_x)
            test_acc = (logits.argmax(1) == test_y).float().mean().item()

        print(
            f"epoch {epoch}/{EPOCHS}"
            f"  loss {loss_sum / batches:.4f}"
            f"  train_acc {100 * correct / (batches * BATCH):.4f}%"
            f"  test_acc {100 * test_acc:.4f}%"
            f"  ({secs:.4f}s)"
        )


if __name__ == "__main__":
    main()
