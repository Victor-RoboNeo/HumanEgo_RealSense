# -*- coding: utf-8 -*-
"""SE(3) helpers and the HumanEgo virtual-gripper local frame."""
from __future__ import annotations

import numpy as np

# T_hand = T_ee @ T_align  =>  T_ee = T_hand @ inv(T_align)
# Example from cfg/inference/example_dualarm.yaml (Trossen vs Aria-MPS hand).
TROSSEN_T_ALIGN = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# VisualKpts.yaml: two-finger wireframe in the midpoint / pinch frame.
# X = finger spread, Y = approach (tips at Y=0, root at -Y), Z = palm normal.
GRIPPER_WIDTH_OPEN = 0.18
GRIPPER_WIDTH_CLOSED = 0.05
GRIPPER_FINGER_LEN = 0.08
GRIPPER_ROOT_DEPTH = 0.08


def as_T(x) -> np.ndarray:
    T = np.asarray(x, dtype=np.float64).reshape(4, 4)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    T = as_T(T)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    R, t = T[:3, :3], T[:3, 3]
    return (R @ pts.T).T + t


def project_cam(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns (uv Nx2, valid mask)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pts_cam = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((len(pts_cam), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv[valid, 0] = K[0, 0] * x + K[0, 2]
        uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def gripper_wireframe(grasp: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Local-frame gripper polyline. Returns (Nx3 points, edge index pairs)."""
    width = GRIPPER_WIDTH_CLOSED if grasp > 0.5 else GRIPPER_WIDTH_OPEN
    half = width / 2.0
    fl = GRIPPER_FINGER_LEN
    rd = GRIPPER_ROOT_DEPTH
    pts = np.array(
        [
            [0.0, -(fl + rd), 0.0],  # 0 root
            [-half, -fl, 0.0],       # 1 left base
            [half, -fl, 0.0],        # 2 right base
            [-half, 0.0, 0.0],       # 3 left tip
            [half, 0.0, 0.0],        # 4 right tip
        ],
        dtype=np.float64,
    )
    edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
    return pts, edges


def camera_frustum_local(size: float = 0.04) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Simple camera pyramid in the optical frame (+Z forward, +X right, +Y down)."""
    d = size
    w, h = size * 0.6, size * 0.45
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [-w, -h, d],
            [w, -h, d],
            [w, h, d],
            [-w, h, d],
        ],
        dtype=np.float64,
    )
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return pts, edges


def yaw_of_xy(R: np.ndarray) -> float:
    """Yaw of a rotation about Z, using the X-axis projected to XY."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation, right-handed, angle in radians."""
    k = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(angles, dtype=np.float64))


def yaw_about_up(
    R_cam_in_world: np.ndarray,
    up: np.ndarray,
    axis0: np.ndarray,
    axis_t: np.ndarray | None = None,
) -> float:
    """Signed yaw about `up` of a camera axis, relative to `axis0`.

    Prefer optical +Z. If that axis is nearly vertical (down-looking), the
    caller should pass camera +X instead so in-place heading stays observable.
    """
    R = np.asarray(R_cam_in_world, dtype=np.float64).reshape(3, 3)
    up = normalize(up)
    v0 = np.asarray(axis0, dtype=np.float64).reshape(3)
    vt = R[:, 2] if axis_t is None else np.asarray(axis_t, dtype=np.float64).reshape(3)

    def horiz(v):
        h = v - float(np.dot(v, up)) * up
        n = float(np.linalg.norm(h))
        return None if n < 1e-6 else h / n

    a, b = horiz(v0), horiz(vt)
    if a is None or b is None:
        return 0.0
    s = float(np.dot(up, np.cross(a, b)))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arctan2(s, c))


def heading_axis(R: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Most-horizontal camera axis: +Z unless it is too aligned with gravity."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    up = normalize(up)
    z = R[:, 2]
    if abs(float(np.dot(z, up))) < 0.92:
        return z
    return R[:, 0]
