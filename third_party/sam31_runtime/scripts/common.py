#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def numeric_frame_dirs(session: Path):
    root = session / "preprocess" / "all_data"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing HumanEgo frame directory: {root}")
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(dirs, key=lambda d: int(d.name))


def find_rgb(frame_dir: Path) -> Path | None:
    for name in ("rgb.png", "rgb.jpg", "aria_cam_rgb.jpg", "aria_cam_rgb.png"):
        p = frame_dir / name
        if p.is_file():
            return p
    return None


def build_jpeg_cache_from_session(
    session: Path,
    cache_dir: Path,
    start: int | None = None,
    end: int | None = None,
    stride: int = 1,
    max_frames: int = 0,
    jpeg_quality: int = 95,
):
    dirs = numeric_frame_dirs(session)
    picked = []
    for d in dirs:
        idx = int(d.name)
        if start is not None and idx < start:
            continue
        if end is not None and idx > end:
            continue
        if ((idx - (start or 0)) % max(1, stride)) != 0:
            continue
        p = find_rgb(d)
        if p is not None:
            picked.append((idx, p))
    if max_frames > 0:
        picked = picked[:max_frames]
    if not picked:
        raise RuntimeError("No RGB frames selected.")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"frames": []}
    for local_i, (source_idx, src) in enumerate(picked):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read {src}")
        dst = cache_dir / f"{local_i:05d}.jpg"
        ok = cv2.imwrite(
            str(dst), img,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )
        if not ok:
            raise RuntimeError(f"Failed to write {dst}")
        manifest["frames"].append({
            "local_idx": local_i,
            "source_idx": source_idx,
            "source_rgb": str(src.resolve()),
            "jpeg": str(dst.resolve()),
        })

    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def source_idx_for_local(manifest, local_idx: int) -> int:
    return int(manifest["frames"][local_idx]["source_idx"])


def load_local_frame(cache_dir: Path, local_idx: int):
    p = cache_dir / f"{local_idx:05d}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {p}")
    return img


def render_outputs_bgr(img_bgr: np.ndarray, outputs: dict, selected_obj_id=None):
    """
    Expected SAM3.1 output keys:
      out_probs, out_obj_ids, out_binary_masks, out_boxes_xywh
    """
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    probs = outputs.get("out_probs", [])
    ids = outputs.get("out_obj_ids", [])
    masks = outputs.get("out_binary_masks", [])
    boxes = outputs.get("out_boxes_xywh", [])

    ids_np = np.asarray(ids)
    probs_np = np.asarray(probs)

    for i in range(len(ids_np)):
        obj_id = int(ids_np[i])
        if selected_obj_id is not None and obj_id != int(selected_obj_id):
            continue

        mask = np.asarray(masks[i])
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        sel = mask > 0.5

        # Deterministic BGR colors from obj_id; no dependency on matplotlib.
        color = (
            int((37 * obj_id + 80) % 255),
            int((97 * obj_id + 130) % 255),
            int((173 * obj_id + 50) % 255),
        )
        if np.any(sel):
            layer = np.zeros_like(overlay)
            layer[:] = color
            overlay[sel] = (
                0.55 * overlay[sel].astype(np.float32)
                + 0.45 * layer[sel].astype(np.float32)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                (sel.astype(np.uint8) * 255),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        if i < len(boxes):
            x, y, bw, bh = np.asarray(boxes[i], dtype=float).reshape(4)
            # Official outputs are relative XYWH.
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            prob = float(probs_np[i]) if i < len(probs_np) else float("nan")
            cv2.putText(
                overlay,
                f"id={obj_id} p={prob:.2f}",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return overlay


def choose_obj_id(outputs: dict, mode: str = "highest_score", explicit=None) -> int:
    ids = np.asarray(outputs.get("out_obj_ids", []))
    probs = np.asarray(outputs.get("out_probs", []), dtype=float)
    masks = outputs.get("out_binary_masks", [])

    if explicit is not None:
        if int(explicit) not in [int(x) for x in ids.tolist()]:
            raise ValueError(
                f"Requested obj-id {explicit}, but prompt frame has IDs {ids.tolist()}"
            )
        return int(explicit)

    if len(ids) == 0:
        raise RuntimeError("SAM3.1 returned no objects on the prompt frame.")

    if mode == "highest_score":
        return int(ids[int(np.nanargmax(probs))])

    if mode == "largest":
        areas = []
        for m in masks:
            a = np.asarray(m)
            areas.append(float((a > 0.5).sum()))
        return int(ids[int(np.argmax(areas))])

    raise ValueError(mode)


def extract_selected_mask(outputs: dict, obj_id: int, target_hw=None):
    ids = np.asarray(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks", [])
    for i, oid in enumerate(ids.tolist()):
        if int(oid) != int(obj_id):
            continue
        m = np.asarray(masks[i])
        if m.ndim > 2:
            m = np.squeeze(m)
        m = (m > 0.5).astype(np.uint8) * 255
        if target_hw is not None and m.shape != tuple(target_hw):
            m = cv2.resize(
                m,
                (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return m
    if target_hw is None:
        return None
    return np.zeros(target_hw, dtype=np.uint8)


def parse_xy_points(specs: list[str] | None):
    pts = []
    if not specs:
        return pts
    for spec in specs:
        for item in spec.split(";"):
            item = item.strip()
            if not item:
                continue
            x, y = item.split(",")
            pts.append((float(x), float(y)))
    return pts


def abs_points_to_rel(points, w: int, h: int):
    return [[x / float(w), y / float(h)] for x, y in points]
