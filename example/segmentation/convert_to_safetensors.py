#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["torch", "safetensors", "packaging", "numpy"]
# ///
#
# One-shot conversion: .pth state_dict  ->  .safetensors
#
# Invoked at CMake configure time when the .safetensors output is missing.
# The conversion is deterministic and idempotent, so the cached result is
# reused on subsequent configures.

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a .pth state_dict to .safetensors")
    ap.add_argument("--input",  required=True, help="path to .pth file")
    ap.add_argument("--output", required=True, help="path to .safetensors output")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1

    print(f"loading {in_path}", flush=True)
    obj = torch.load(in_path, map_location="cpu", weights_only=False)

    # Some checkpoints wrap the state_dict; unwrap defensively.
    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        sd = obj
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    else:
        print(f"error: cannot extract state_dict from {type(obj)}", file=sys.stderr)
        return 1

    # Strip DataParallel "module." prefix; make tensors contiguous + float32 on CPU.
    cleaned = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            print(f"  skipping non-tensor: {k}", flush=True)
            continue
        key = k[len("module."):] if k.startswith("module.") else k
        cleaned[key] = v.contiguous().to(torch.float32).cpu()

    print(f"writing {out_path}  ({len(cleaned)} tensors)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cleaned, str(out_path))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
