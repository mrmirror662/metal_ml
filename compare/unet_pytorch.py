#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "segmentation_models_pytorch>=0.4",
#   "pillow",
#   "numpy",
# ]
# ///
#
# PyTorch reference for example/segmentation. Same model
# (FiniUdesa/unet-human-segmentation = smp.Unet, encoder=resnet34, classes=1),
# same preprocessing (ImageNet normalize, center-crop+resize to 256), so timing
# and mask outputs are directly comparable to the C++ Metal implementation.
#
# Usage:
#   compare/unet_pytorch.py [--bench N] [--device cpu|mps] [--out DIR] IMAGE [IMAGE ...]
#
#   --bench N   : run 1 warmup + N timed iterations of forward(), print stats
#   --device    : "mps" (default if available) or "cpu"
#   --out       : write input.png, mask.png, overlay.png next to each input here
#                 (default: compare/outputs_pytorch/<image_stem>/)
#
# If no images given, runs against example/segmentation/samples/*.jpg.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image


HERE       = Path(__file__).resolve().parent
REPO       = HERE.parent
WEIGHTS    = REPO / "build" / "example" / "segmentation" / "data" / "human_unet.pth"
SAMPLES    = REPO / "example" / "segmentation" / "samples"
IMG_DIM    = 256
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(requested: str | None) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS not available")
        return torch.device("mps")
    # auto: prefer mps on Mac
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device: torch.device) -> torch.nn.Module:
    if not WEIGHTS.exists():
        raise SystemExit(
            f"weights not found at {WEIGHTS}\n"
            "Run `cmake --build build` first — that downloads the .pth."
        )
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,           # we're loading trained weights below
        in_channels=3,
        classes=1,
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    elif isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


def center_crop_resize(img: Image.Image, dim: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((dim, dim), Image.BILINEAR)


def preprocess(img: Image.Image, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    img = img.convert("RGB")
    sized = center_crop_resize(img, IMG_DIM)
    arr = np.asarray(sized, dtype=np.float32) / 255.0    # HWC, [0, 1]
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]              # 1xCxHxW
    t = torch.from_numpy(arr).to(device)
    return t, sized


def save_outputs(sized: Image.Image, mask: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sized.save(out_dir / "input.png")
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_dir / "mask.png")

    rgb = np.asarray(sized).copy()
    fg  = mask > 0
    rgb[..., 0] = np.where(fg, (rgb[..., 0].astype(np.int32) + 255) // 2, rgb[..., 0])
    rgb[..., 1] = np.where(fg, rgb[..., 1] // 2, rgb[..., 1])
    rgb[..., 2] = np.where(fg, rgb[..., 2] // 2, rgb[..., 2])
    Image.fromarray(rgb, mode="RGB").save(out_dir / "overlay.png")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def bench(model: torch.nn.Module, x: torch.Tensor, device: torch.device, n: int) -> dict:
    # Warmup
    _ = run(model, x); sync(device)
    times_ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = run(model, x); sync(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    a = np.array(times_ms)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()), "std": float(a.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0, help="run N iters and print stats")
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--out", type=Path, default=None, help="output dir root")
    ap.add_argument("images", nargs="*")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        paths = sorted(SAMPLES.glob("*.jpg"))
        if not paths:
            print(f"no samples in {SAMPLES}", file=sys.stderr)
            return 1

    t0 = time.perf_counter()
    model = build_model(device)
    print(f"loaded model in {time.perf_counter() - t0:.2f}s")

    out_root = args.out or (HERE / "outputs_pytorch")

    all_results = []
    for p in paths:
        img = Image.open(p)
        x, sized = preprocess(img, device)

        if args.bench > 0:
            stats = bench(model, x, device, args.bench)
            print(f"  {p.name:30s}  mean {stats['mean']:6.2f} ms  "
                  f"min {stats['min']:6.2f}  max {stats['max']:6.2f}  std {stats['std']:5.2f}")
            all_results.append((p.name, stats))
        else:
            # Single inference + save
            t_start = time.perf_counter()
            y = run(model, x); sync(device)
            t_end = time.perf_counter()
            sig = torch.sigmoid(y).cpu().numpy()[0, 0]
            mask = (sig > 0.5).astype(np.uint8)
            fg_pct = 100.0 * mask.sum() / mask.size
            print(f"  {p.name:30s}  {(t_end - t_start) * 1000:6.1f} ms  fg {fg_pct:5.1f}%")
            save_outputs(sized, mask, out_root / p.stem)

    if args.bench > 0 and len(all_results) > 1:
        means = [r[1]["mean"] for r in all_results]
        print(f"\noverall mean across {len(all_results)} samples: "
              f"{sum(means) / len(means):.2f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
