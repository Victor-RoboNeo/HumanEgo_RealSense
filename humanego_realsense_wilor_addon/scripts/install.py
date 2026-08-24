#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the RealSense+WiLoR hand add-on into an existing HumanEgo checkout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once_or_all(text: str, old: str, new: str):
    if old in text:
        return text.replace(old, new), True
    return text, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanego", required=True, help="HumanEgo repo root")
    ap.add_argument(
        "--no-patch",
        action="store_true",
        help="Only copy the module; do not edit RealSensePreprocess.py",
    )
    args = ap.parse_args()

    repo = Path(args.humanego).expanduser().resolve()
    if not (repo / "preprocess").is_dir():
        raise SystemExit(f"Not a HumanEgo checkout: {repo}")

    required = [
        repo / "preprocess" / "WiLoRHands.py",
        repo / "preprocess" / "RealSenseSession.py",
        repo / "preprocess" / "RealSensePreprocess.py",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(
                f"Missing required file: {p}\n"
                "This add-on expects the current HumanEgo repo plus the "
                "RealSense frontend installed earlier."
            )

    src = (
        Path(__file__).resolve().parent.parent
        / "preprocess"
        / "RealSenseWiLoRHands.py"
    )
    dst = repo / "preprocess" / "RealSenseWiLoRHands.py"
    shutil.copy2(src, dst)
    print("[OK] installed", dst)

    if args.no_patch:
        return

    pp = repo / "preprocess" / "RealSensePreprocess.py"
    original = pp.read_text(encoding="utf-8")
    backup = pp.with_suffix(".py.bak_mediapipe_hands")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print("[OK] backup", backup)

    s = original
    changed = False

    replacements = [
        (
            "from preprocess.RealSenseHands import RealSenseHandsGenerator",
            "from preprocess.RealSenseWiLoRHands import RealSenseWiLoRHandsGenerator",
        ),
        (
            "from preprocess.RealSenseHands import RealSenseHands",
            "from preprocess.RealSenseWiLoRHands import RealSenseWiLoRHandsGenerator",
        ),
        (
            "RealSenseHandsGenerator(",
            "RealSenseWiLoRHandsGenerator(",
        ),
        (
            "RealSenseHands(",
            "RealSenseWiLoRHandsGenerator(",
        ),
    ]

    for old, new in replacements:
        s2, did = replace_once_or_all(s, old, new)
        if did:
            changed = True
            s = s2

    if not changed:
        print(
            "[WARN] Could not automatically find the old RealSenseHands import/call.\n"
            "       The new module was installed, but RealSensePreprocess.py was not changed.\n"
            "       Manually replace your MediaPipe RealSenseHandsGenerator with:\n"
            "       RealSenseWiLoRHandsGenerator"
        )
        return

    pp.write_text(s, encoding="utf-8")
    print("[OK] patched", pp)
    print()
    print("Check with:")
    print(
        "  grep -n 'RealSenseWiLoRHands\\|RealSenseWiLoRHandsGenerator' "
        "preprocess/RealSensePreprocess.py"
    )


if __name__ == "__main__":
    main()
