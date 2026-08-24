#!/usr/bin/env python3
"""
Compare an existing HumanEgo mask_obj1.png against SAM3.1 smoke-test masks.

No ground truth is assumed. The video is for visual diagnosis:
  RGB | current HumanEgo mask | SAM3.1 mask
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def overlay(img, mask, title):
    out = img.copy()
    sel = mask > 0
    if np.any(sel):
        layer = np.zeros_like(out)
        layer[:, :, 1] = 255
        out[sel] = (
            0.6 * out[sel].astype(np.float32)
            + 0.4 * layer[sel].astype(np.float32)
        ).astype(np.uint8)
        cnts, _ = cv2.findContours(
            (sel.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, cnts, -1, (0,255,0), 2)
    cv2.rectangle(out, (0,0), (out.shape[1],40), (0,0,0), -1)
    cv2.putText(
        out, title, (10,27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255,255,255), 2, cv2.LINE_AA
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--sam31-masks", required=True)
    ap.add_argument("--obj", default="obj1")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    session = Path(args.session).expanduser().resolve()
    sam_masks = Path(args.sam31_masks).expanduser().resolve()
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else sam_masks.parent / "compare_current_vs_sam31.mp4"
    )

    frames_root = session / "preprocess" / "all_data"
    entries = []
    for m in sorted(sam_masks.glob("*.png")):
        try:
            idx = int(m.stem)
        except Exception:
            continue
        d = frames_root / f"{idx:05d}"
        if not d.is_dir():
            d = frames_root / str(idx)
        rgb = d / "rgb.png"
        if not rgb.is_file():
            continue
        entries.append((idx, d, rgb, m))

    if not entries:
        raise RuntimeError("No comparable frames.")

    first = cv2.imread(str(entries[0][2]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w * 3, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    try:
        for idx, d, rgbp, sam31p in entries:
            img = cv2.imread(str(rgbp))
            cur = cv2.imread(str(d / f"mask_{args.obj}.png"), cv2.IMREAD_GRAYSCALE)
            new = cv2.imread(str(sam31p), cv2.IMREAD_GRAYSCALE)
            if cur is None:
                cur = np.zeros((h,w), np.uint8)
            if new is None:
                new = np.zeros((h,w), np.uint8)
            if cur.shape != (h,w):
                cur = cv2.resize(cur, (w,h), interpolation=cv2.INTER_NEAREST)
            if new.shape != (h,w):
                new = cv2.resize(new, (w,h), interpolation=cv2.INTER_NEAREST)

            raw = img.copy()
            cv2.putText(
                raw, f"RGB frame {idx}", (10,27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255,255,255), 2, cv2.LINE_AA
            )
            writer.write(cv2.hconcat([
                raw,
                overlay(img, cur, "Current DINO+SAM2"),
                overlay(img, new, "SAM3.1"),
            ]))
    finally:
        writer.release()

    print("[OK] saved:", out)


if __name__ == "__main__":
    main()
