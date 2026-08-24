#!/usr/bin/env python3
"""Copy the RealSense frontend files into an existing HumanEgo checkout."""
from pathlib import Path
import argparse
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--humanego", required=True)
    args = p.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    dst_root = Path(args.humanego).expanduser().resolve()

    if not (dst_root / "preprocess").is_dir():
        raise SystemExit(f"Not a HumanEgo checkout: {dst_root}")

    for src in (src_root / "preprocess").glob("RealSense*.py"):
        dst = dst_root / "preprocess" / src.name
        shutil.copy2(src, dst)
        print("copied", dst)

    cfg_src = src_root / "cfg" / "preprocess" / "base" / "RealSense.yaml"
    cfg_dst = dst_root / "cfg" / "preprocess" / "base" / "RealSense.yaml"
    cfg_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_src, cfg_dst)
    print("copied", cfg_dst)

    print("\nDone. No upstream HumanEgo source file was overwritten.")


if __name__ == "__main__":
    main()
