# -*- coding: utf-8 -*-
"""Re-estimate the frozen object origin from mask_obj1 + metric depth.

CoTracker triangulation can drift; the SAM mask centroid lifted through D455
depth is a better visual origin for the strawberry. Rotation is kept from the
preprocess VLM/triangulator pose so only the translation is corrected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from retarget.geometry import as_T


def _load_depth_m(depth_path: str | None, scale: float) -> np.ndarray | None:
    if not depth_path or not Path(depth_path).is_file():
        return None
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float32) * float(scale)


def _load_mask(mask_path: str | None) -> np.ndarray | None:
    if not mask_path or not Path(mask_path).is_file():
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return m


def lift_mask_centroid_cam(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    min_z: float = 0.08,
    max_z: float = 2.5,
    erode: int = 2,
) -> dict[str, Any] | None:
    """Robust 3D centroid of a binary mask in the camera optical frame."""
    m = (mask > 127).astype(np.uint8)
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        m = cv2.erode(m, k)
    if depth_m.shape[:2] != m.shape[:2]:
        depth_m = cv2.resize(depth_m, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(m > 0)
    if xs.size < 20:
        return None
    z = depth_m[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z >= min_z) & (z <= max_z)
    if valid.sum() < 20:
        return None
    xs, ys, z = xs[valid].astype(np.float64), ys[valid].astype(np.float64), z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])
    p_cam = np.median(pts, axis=0)
    uv = np.array([float(np.median(xs)), float(np.median(ys))], dtype=np.float64)
    return {
        "p_cam": p_cam,
        "uv": uv,
        "n": int(len(pts)),
        "z_med": float(np.median(z)),
    }


def estimate_mask_anchor(
    session: dict[str, Any],
    n_init: int = 10,
    min_z: float = 0.08,
    max_z: float = 2.5,
) -> dict[str, Any]:
    """Freeze a mask+depth world origin from the first `n_init` valid frames.

    Translation = median of per-frame mask centroids in world.
    Rotation    = preprocess `T_anchor_w` rotation (VLM), not recomputed.
    """
    T_src = as_T(session["T_anchor_w"])
    samples_w = []
    per_frame = []
    used = 0
    for fr in session["frames"]:
        mask = _load_mask(fr.get("mask_obj1_path"))
        depth = _load_depth_m(fr.get("depth_path"), fr.get("depth_scale", 0.001))
        rec = {"idx": fr["idx"], "uv": None, "p_cam": None, "p_w": None, "n": 0}
        if mask is not None and depth is not None:
            hit = lift_mask_centroid_cam(mask, depth, fr["K"], min_z=min_z, max_z=max_z)
            if hit is not None:
                T_cam_w = as_T(fr["T_cam_w"])
                p_w = T_cam_w[:3, :3] @ hit["p_cam"] + T_cam_w[:3, 3]
                rec.update({"uv": hit["uv"], "p_cam": hit["p_cam"], "p_w": p_w, "n": hit["n"]})
                if used < n_init:
                    samples_w.append(p_w)
                    used += 1
        per_frame.append(rec)

    if not samples_w:
        raise RuntimeError("no valid mask_obj1 + depth centroid; cannot re-estimate origin")

    t_w = np.median(np.stack(samples_w, axis=0), axis=0)
    T_mask_w = T_src.copy()
    T_mask_w[:3, 3] = t_w
    delta = t_w - T_src[:3, 3]
    return {
        "T_mask_w": T_mask_w,
        "T_src_w": T_src,
        "t_w": t_w,
        "delta_w": delta,
        "n_init_used": used,
        "n_init_requested": n_init,
        "per_frame": per_frame,
    }
