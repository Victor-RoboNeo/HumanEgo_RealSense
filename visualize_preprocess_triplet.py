#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize HumanEgo preprocess as a 3-column video:

  left   original RGB
  middle RGB + all SAM3.1 masks (arm / obj1 / obj2 / ...)
  right  arm inpainted + virtual gripper + object masks (no arm)

This script only reads existing files; it does not rerun preprocess.

Example:
  python visualize_preprocess_triplet.py \
    --session realsense_linux_rgbd_imu_v2/captures/session_20260813_200913
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SESSION = (
    Path(__file__).resolve().parent
    / "realsense_linux_rgbd_imu_v2"
    / "captures"
    / "session_20260813_200913"
)

# BGR
MASK_COLORS = {
    "arm": (0, 180, 255),
    "obj1": (60, 220, 60),
    "obj2": (255, 80, 200),
    "obj3": (255, 160, 0),
    "obj4": (80, 80, 255),
}
DEFAULT_COLOR = (200, 200, 200)
HEADER_H = 42
LEGEND_H = 28


def frame_dirs(all_data: Path, start: int | None, end: int | None) -> list[Path]:
    dirs = sorted(
        (d for d in all_data.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if start is not None:
        dirs = [d for d in dirs if int(d.name) >= start]
    if end is not None:
        dirs = [d for d in dirs if int(d.name) <= end]
    return dirs


def load_bgr(path: Path, hw: tuple[int, int] | None = None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if hw is not None and img.shape[:2] != hw:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


def load_mask(path: Path, hw: tuple[int, int]):
    if not path.is_file():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def discover_mask_keys(frame_dir: Path) -> list[str]:
    keys = []
    for p in sorted(frame_dir.glob("mask_*.png")):
        key = p.stem[len("mask_") :]
        if key in {"arm_and_obj", "obj_and_arm"}:
            continue
        keys.append(key)
    # Stable order: objects first, arm last.
    objs = [k for k in keys if k.startswith("obj")]
    rest = [k for k in keys if k not in objs and k != "arm"]
    arm = ["arm"] if "arm" in keys else []
    return objs + rest + arm


def overlay_masks(rgb, masks: dict[str, np.ndarray], keys: list[str], alpha: float = 0.45):
    out = rgb.copy()
    for key in keys:
        m = masks.get(key)
        if m is None:
            continue
        sel = m > 0
        if not np.any(sel):
            continue
        color = MASK_COLORS.get(key, DEFAULT_COLOR)
        tint = np.zeros_like(out)
        tint[:, :] = color
        out[sel] = (
            (1.0 - alpha) * out[sel] + alpha * tint[sel]
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            (sel.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def put_header(img, title: str):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, HEADER_H), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.82, img, 0.18, 0, img)
    cv2.putText(
        img,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def legend_bar(width: int, keys: list[str]) -> np.ndarray:
    bar = np.full((LEGEND_H, width, 3), 18, dtype=np.uint8)
    x = 12
    for key in keys:
        color = MASK_COLORS.get(key, DEFAULT_COLOR)
        cv2.rectangle(bar, (x, 7), (x + 16, 21), color, -1)
        cv2.rectangle(bar, (x, 7), (x + 16, 21), (255, 255, 255), 1)
        label = key
        cv2.putText(
            bar,
            label,
            (x + 22, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        x += 22 + 8 * len(label) + 18
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    session = args.session.expanduser().resolve()
    all_data = session / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise SystemExit(f"missing all_data: {all_data}")

    dirs = frame_dirs(all_data, args.start, args.end)
    if not dirs:
        raise SystemExit("no numeric frame dirs")

    first_rgb = load_bgr(dirs[0] / "rgb.png")
    if first_rgb is None:
        raise SystemExit(f"cannot read {dirs[0] / 'rgb.png'}")
    hw = first_rgb.shape[:2]
    h, w = hw

    mask_keys = discover_mask_keys(dirs[0])
    obj_keys = [k for k in mask_keys if k != "arm"]

    out_path = (
        args.output.expanduser().resolve()
        if args.output
        else session / "preprocess" / "vis" / "triplet_rgb_masks_gripper.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_h = LEGEND_H + h
    canvas_w = w * 3
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise SystemExit(f"cannot open VideoWriter: {out_path}")

    n = len(dirs)
    wrote = 0
    try:
        for i, d in enumerate(dirs):
            rgb = load_bgr(d / "rgb.png", hw)
            if rgb is None:
                continue

            masks = {}
            for key in mask_keys:
                m = load_mask(d / f"mask_{key}.png", hw)
                if m is not None:
                    masks[key] = m

            mid = overlay_masks(rgb, masks, mask_keys)

            right_src = load_bgr(d / "rgb_WoArm_WArmObjKpts.png", hw)
            if right_src is None:
                right_src = load_bgr(d / "rgb_WoArm.png", hw)
            if right_src is None:
                right_src = rgb.copy()
            right = overlay_masks(right_src, masks, obj_keys, alpha=0.35)

            left = put_header(rgb.copy(), f"RGB  {d.name}")
            mid = put_header(mid, "masks  arm+obj")
            right = put_header(right, "WoArm + gripper + obj masks")

            row = cv2.hconcat([left, mid, right])
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:LEGEND_H] = legend_bar(canvas_w, mask_keys)
            canvas[LEGEND_H:] = row
            cv2.putText(
                canvas,
                f"{i + 1}/{n}",
                (canvas_w - 110, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            wrote += 1
            if wrote % 50 == 0 or wrote == n:
                print(f"wrote {wrote}/{n}")
    finally:
        writer.release()

    print(f"saved {wrote} frames -> {out_path}")


if __name__ == "__main__":
    main()
