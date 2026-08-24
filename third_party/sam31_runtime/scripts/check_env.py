#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

def main():
    print("=" * 72)
    print("SAM 3.1 smoke-test environment")
    print("=" * 72)
    print("python:", sys.version.replace("\n", " "))
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda runtime:", torch.version.cuda)
            print("gpu:", torch.cuda.get_device_name(0))
            print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    except Exception as e:
        print("torch ERROR:", repr(e))

    try:
        import sam3
        print("sam3:", Path(sam3.__file__).resolve())
        from sam3.model_builder import build_sam3_predictor
        print("build_sam3_predictor: OK")
    except Exception as e:
        print("sam3 ERROR:", repr(e))

    try:
        import cv2
        print("opencv:", cv2.__version__)
    except Exception as e:
        print("opencv ERROR:", repr(e))

    try:
        import huggingface_hub
        print("huggingface_hub:", huggingface_hub.__version__)
    except Exception as e:
        print("huggingface_hub ERROR:", repr(e))
    print("=" * 72)

if __name__ == "__main__":
    main()
