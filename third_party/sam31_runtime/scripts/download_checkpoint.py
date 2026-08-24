#!/usr/bin/env python3
"""
Download the official SAM 3.1 checkpoint.

HF:
  - repo: facebook/sam3.1
  - file: sam3.1_multiplex.pt
  - gated: you must accept access terms and run `hf auth login`

ModelScope:
  - attempts the facebook/sam3.1 mirror
  - verifies that sam3.1_multiplex.pt actually exists after download
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "facebook/sam3.1"
CKPT_NAME = "sam3.1_multiplex.pt"

def download_hf(out_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(
        repo_id=REPO_ID,
        filename=CKPT_NAME,
        local_dir=str(out_dir),
    )
    return Path(p)

def download_modelscope(out_dir: Path) -> Path:
    from modelscope import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(
        snapshot_download(
            REPO_ID,
            local_dir=str(out_dir),
        )
    )
    candidates = list(model_dir.rglob(CKPT_NAME))
    if not candidates:
        candidates = list(out_dir.rglob(CKPT_NAME))
    if not candidates:
        raise FileNotFoundError(
            f"ModelScope download completed but {CKPT_NAME!r} was not found. "
            "Use --source hf after accepting Meta's gated-model terms."
        )
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints" / "sam3.1"),
    )
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    print(f"source     : {args.source}")
    print(f"output dir : {out}")

    if args.source == "hf":
        path = download_hf(out)
    else:
        path = download_modelscope(out)

    size_gib = path.stat().st_size / 2**30
    print(f"[OK] checkpoint: {path}")
    print(f"     size: {size_gib:.3f} GiB")

if __name__ == "__main__":
    main()
