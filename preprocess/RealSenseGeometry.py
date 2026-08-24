# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def backproject_pixels(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.column_stack([x, y, z])


def project_points(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    z = np.maximum(xyz[:, 2], 1e-9)
    u = xyz[:, 0] / z * K[0, 0] + K[0, 2]
    v = xyz[:, 1] / z * K[1, 1] + K[1, 2]
    return np.column_stack([u, v])


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return (T[:3, :3] @ xyz.T).T + T[:3, 3]


def robust_local_depth(
    depth_m: np.ndarray,
    uv: np.ndarray,
    radius: int = 4,
    min_depth: float = 0.08,
    max_depth: float = 4.0,
) -> np.ndarray:
    """Median depth in a local square around each color-space landmark."""
    h, w = depth_m.shape
    out = np.full((len(uv),), np.nan, dtype=np.float64)

    for i, (u, v) in enumerate(np.asarray(uv)):
        x = int(round(float(u)))
        y = int(round(float(v)))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[
            np.isfinite(patch)
            & (patch >= min_depth)
            & (patch <= max_depth)
        ]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def align_depth_to_color_numpy(
    depth_raw: np.ndarray,
    depth_scale: float,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    T_depth_to_color: np.ndarray,
    color_hw: tuple[int, int],
    min_depth: float = 0.08,
    max_depth: float = 6.0,
    stride: int = 1,
) -> np.ndarray:
    """
    Offline depth->color z-buffer projection.

    This is a pinhole fallback for sessions that were NOT captured using
    RealSense rs.align().  For best hand geometry, record with
    --spatial-align-depth and this function is bypassed.
    """
    h_d, w_d = depth_raw.shape
    h_c, w_c = color_hw

    ys, xs = np.mgrid[0:h_d:stride, 0:w_d:stride]
    z = depth_raw[::stride, ::stride].astype(np.float64) * depth_scale

    valid = (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        return np.zeros((h_c, w_c), dtype=np.float32)

    u = xs[valid].astype(np.float64)
    v = ys[valid].astype(np.float64)
    zz = z[valid]

    xyz_d = backproject_pixels(np.column_stack([u, v]), zz, K_depth)
    xyz_c = transform_points(T_depth_to_color, xyz_d)

    valid_c = xyz_c[:, 2] > 1e-6
    xyz_c = xyz_c[valid_c]
    uv_c = project_points(xyz_c, K_color)

    px = np.rint(uv_c[:, 0]).astype(np.int32)
    py = np.rint(uv_c[:, 1]).astype(np.int32)
    zz_c = xyz_c[:, 2].astype(np.float32)

    inside = (px >= 0) & (px < w_c) & (py >= 0) & (py < h_c)
    px, py, zz_c = px[inside], py[inside], zz_c[inside]

    flat_idx = py * w_c + px
    zbuf = np.full(h_c * w_c, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, flat_idx, zz_c)
    zbuf[~np.isfinite(zbuf)] = 0.0
    return zbuf.reshape(h_c, w_c)


def fill_missing_landmark_depths(
    z: np.ndarray,
    world_lms_mp: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill occasional stereo holes while keeping observed metric depths intact.
    Preference:
      1) median of valid palm/MCP depth
      2) median of all valid landmarks
      3) optionally add MediaPipe relative z offsets around wrist
    """
    z = np.asarray(z, dtype=np.float64).copy()
    valid = np.isfinite(z) & (z > 0.05)

    if np.sum(valid) < 3:
        return z

    # MP palm anchors: wrist, index/middle/ring/pinky MCP.
    palm_ids = np.array([0, 5, 9, 13, 17], dtype=int)
    palm_valid = palm_ids[valid[palm_ids]]
    base = (
        float(np.median(z[palm_valid]))
        if len(palm_valid)
        else float(np.median(z[valid]))
    )

    missing = ~valid
    if world_lms_mp is not None and np.isfinite(z[0]):
        rel_z = np.asarray(world_lms_mp)[:, 2] - float(world_lms_mp[0, 2])
        predicted = z[0] + rel_z
        z[missing] = predicted[missing]
    else:
        z[missing] = base

    return z
