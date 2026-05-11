#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
# Diff the actual preprocessed inputs the two implementations fed to the model.
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
for name in ["person_candid", "person_full", "person_group",
             "person_portrait", "person_walking", "person_yoga"]:
    a = np.asarray(Image.open(ROOT / "example/segmentation/outputs" / name / "input.png")).astype(np.int32)
    b = np.asarray(Image.open(ROOT / "compare/outputs_pytorch" / name / "input.png")).astype(np.int32)
    d = np.abs(a - b)
    print(f"{name:20s}  shape={a.shape}  max|Δ|={d.max():3d}  mean|Δ|={d.mean():.3f}")
