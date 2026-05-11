#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
#
# Compare masks produced by our C++ Metal lib vs the PyTorch reference.
# Reports per-sample IoU and pixel accuracy.

from pathlib import Path
import numpy as np
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
CPP_DIR = ROOT / "example" / "segmentation" / "outputs"
PT_DIR  = ROOT / "compare" / "outputs_pytorch"

print(f"{'sample':<22}{'IoU':>8}{'px acc':>10}{'cg fg%':>9}{'pt fg%':>9}")
ious, accs = [], []
for cpp_dir in sorted(CPP_DIR.iterdir()):
    if not cpp_dir.is_dir():
        continue
    pt_dir = PT_DIR / cpp_dir.name
    if not pt_dir.exists():
        continue
    a = np.asarray(Image.open(cpp_dir / "mask.png").convert("L")) > 127
    b = np.asarray(Image.open(pt_dir  / "mask.png").convert("L")) > 127
    if a.shape != b.shape:
        a_img = Image.open(cpp_dir / "mask.png").convert("L").resize(b.shape[::-1], Image.NEAREST)
        a = np.asarray(a_img) > 127
    inter = (a & b).sum()
    union = (a | b).sum()
    iou   = inter / max(union, 1)
    acc   = (a == b).sum() / a.size
    ious.append(iou)
    accs.append(acc)
    print(f"{cpp_dir.name:<22}{iou:>8.4f}{acc:>10.4f}"
          f"{a.mean() * 100:>8.2f}%{b.mean() * 100:>8.2f}%")

if ious:
    print(f"\nmean IoU: {np.mean(ious):.4f}   mean pixel acc: {np.mean(accs):.4f}")
