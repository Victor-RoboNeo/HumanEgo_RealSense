# -*- coding: utf-8 -*-
"""Camera-frame replay GT (y_action + y_base) for HumanEgo_mobile.

Paper App. E.3 and FlowMatchingDataloader:

    T_ref = metadata.world_transforms["cam0"]   # static OpenCV camera
    T_*_in_ref = inv(T_ref) @ T_*_w

OpenCV cam0 is X-right, Y-down, Z-forward. The handheld lever-arm is still
wiped so a mobile chassis (`T_cam_in_base = I`) can replay the same GT:

    T_cam_virt(t) = { R_yaw(t) @ R_cam0,  t_cam0 }
    T_hand_w(t)   = T_cam_virt(t) @ T_hand_in_cam(t)
    y_action      = inv(T_cam0) @ T_hand_w          # paper camera_frame
    y_base        = (0, 0, gravity_yaw) after wipe

Replay:

    T_ee_in_base = T_hand_in_cam @ inv(T_align)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from retarget.geometry import (
    as_T,
    axis_angle_R,
    invert,
    normalize,
    unwrap_angles,
    yaw_about_up,
)


def gravity_up_in_cam0(mode: str = "camera-neg-y") -> np.ndarray:
    if mode == "camera-neg-y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if mode == "world-z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"unknown up mode {mode}")


def T_floor_from_cam0(
    up_w: np.ndarray,
    origin_w: np.ndarray,
    right_w: np.ndarray,
) -> np.ndarray:
    """Z-up floor frame: X=image right, Y=forward, Z=up (matches RGB)."""
    z = normalize(up_w)
    x = np.asarray(right_w, dtype=np.float64).reshape(3)
    x = x - float(np.dot(x, z)) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = x - float(np.dot(x, z)) * z
    x = normalize(x)
    y = normalize(np.cross(z, x))
    T = np.eye(4, dtype=np.float64)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = np.asarray(origin_w, dtype=np.float64).reshape(3)
    return T


def camera_ref_T(T_cam0: np.ndarray, up_mode: str = "camera-neg-y") -> np.ndarray:
    """Floor frame for diagnostics only: origin at cam0, X=right, Y=forward, Z=up.

    Not used as T_ref. Paper / dataloader camera_frame is OpenCV cam0.
    """
    T_cam0 = as_T(T_cam0)
    up = gravity_up_in_cam0(up_mode)
    up_w = T_cam0[:3, :3] @ up
    return T_floor_from_cam0(up_w, T_cam0[:3, 3], right_w=T_cam0[:3, 0])


def estimate_yaw_pivot(
    t_cam: np.ndarray,
    yaws: np.ndarray,
    up_w: np.ndarray,
    t_cam0: np.ndarray,
    min_yaw_deg: float = 8.0,
) -> np.ndarray:
    """Vertical axis the camera *would* arc around (used only to measure radius)."""
    up = normalize(up_w)
    t0 = np.asarray(t_cam0, dtype=np.float64).reshape(3)
    A_rows, b_rows = [], []
    for t, yaw in zip(t_cam, yaws):
        if abs(float(np.rad2deg(yaw))) < min_yaw_deg:
            continue
        R = axis_angle_R(up, float(yaw))
        A = np.eye(3) - R
        P = np.eye(3) - np.outer(up, up)
        A_rows.append(P @ A)
        b_rows.append(P @ (np.asarray(t, dtype=np.float64).reshape(3) - R @ t0))
    if len(A_rows) < 3:
        return t0.copy()
    A_all = np.concatenate(A_rows, axis=0)
    b_all = np.concatenate(b_rows, axis=0)
    p, *_ = np.linalg.lstsq(A_all, b_all, rcond=None)
    p = p - float(np.dot(p - t0, up)) * up
    return p


def se2_xyyaw(T_in_ref: np.ndarray, yaw: float | None = None) -> np.ndarray:
    """(x, y, yaw) in OpenCV cam0. Optional gravity-yaw override for y_base."""
    T = as_T(T_in_ref)
    if yaw is None:
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return np.array([T[0, 3], T[1, 3], float(yaw)], dtype=np.float64)


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def retarget_yaw_world(
    session: dict[str, Any],
    T_obj0_w: np.ndarray,
    T_align: np.ndarray | None = None,
    up_mode: str = "camera-neg-y",
    wipe_radius: bool = True,
) -> dict[str, Any]:
    T_align = np.eye(4, dtype=np.float64) if T_align is None else as_T(T_align)
    T_align_inv = invert(T_align)
    T_obj0 = as_T(T_obj0_w)
    T_cam0 = as_T(session.get("T_cam0_w", session["frames"][0]["T_cam_w"]))
    up = gravity_up_in_cam0(up_mode)
    up_w = T_cam0[:3, :3] @ up
    axis0 = T_cam0[:3, 0]
    T_ref = T_cam0
    T_w2ref = invert(T_ref)
    T_obj_in_ref = T_w2ref @ T_obj0
    T_floor = camera_ref_T(T_cam0, up_mode)

    yaws = []
    t_icp = []
    for fr in session["frames"]:
        T_cam_icp = as_T(fr["T_cam_w"])
        yaws.append(yaw_about_up(T_cam_icp[:3, :3], up_w, axis0, T_cam_icp[:3, 0]))
        t_icp.append(T_cam_icp[:3, 3])
    yaws = unwrap_angles(yaws)
    t_icp = np.stack(t_icp)
    pivot = estimate_yaw_pivot(t_icp, yaws, up_w, T_cam0[:3, 3])
    radius = float(np.linalg.norm((T_cam0[:3, 3] - pivot) - np.dot(T_cam0[:3, 3] - pivot, up_w) * up_w))

    frames_out = []
    y_base_list = []
    for fr, yaw in zip(session["frames"], yaws):
        T_cam_icp = as_T(fr["T_cam_w"])
        R_yaw = axis_angle_R(up_w, float(yaw))

        # Ghost: the lever-arm arc we are about to delete.
        T_cam_arc = np.eye(4, dtype=np.float64)
        T_cam_arc[:3, :3] = R_yaw @ T_cam0[:3, :3]
        t_arc = pivot + R_yaw @ (T_cam0[:3, 3] - pivot)
        t_arc = t_arc - float(np.dot(t_arc, up_w)) * up_w + float(np.dot(T_cam0[:3, 3], up_w)) * up_w
        T_cam_arc[:3, 3] = t_arc

        # Chassis / camera for the robot: yaw about cam0, radius = 0.
        T_cam_virt = np.eye(4, dtype=np.float64)
        T_cam_virt[:3, :3] = R_yaw @ T_cam0[:3, :3]
        T_cam_virt[:3, 3] = T_cam0[:3, 3] if wipe_radius else t_arc
        T_cam_in_ref = T_w2ref @ T_cam_virt
        T_base_in_ref = T_cam_in_ref  # T_cam_in_base = I
        y_base = se2_xyyaw(T_base_in_ref, yaw=float(yaw) if wipe_radius else None)
        y_base_list.append(y_base)

        rec = {
            "idx": fr["idx"],
            "ts": fr["ts"],
            "K": fr["K"],
            "w": fr["w"],
            "h": fr["h"],
            "rgb_path": fr["rgb_path"],
            "is_finished": fr["is_finished"],
            "yaw": float(yaw),
            "y_base": y_base,
            "T_cam_w": T_cam_icp,
            "T_cam_arc": T_cam_arc,
            "T_cam_virt": T_cam_virt,
            "T_cam_in_ref": T_cam_in_ref,
            "T_cam_in_star": T_cam_in_ref,
            "T_base_in_ref": T_base_in_ref,
            "T_hand_in_ref": None,
            "T_hand_in_star": None,
            "T_hand_in_obj": None,
            "T_hand_in_cam": None,
            "T_ee_in_base": None,
            "grasp": fr["grasp"],
        }
        if fr["T_hand_w"] is not None:
            T_hand_in_cam = invert(T_cam_icp) @ as_T(fr["T_hand_w"])
            T_hand_w_virt = T_cam_virt @ T_hand_in_cam
            rec["T_hand_in_cam"] = T_hand_in_cam
            rec["T_hand_in_star"] = T_w2ref @ T_hand_w_virt
            rec["T_hand_in_obj"] = invert(T_obj0) @ T_hand_w_virt
            rec["T_hand_in_ref"] = rec["T_hand_in_star"]
            rec["T_ee_in_base"] = T_hand_in_cam @ T_align_inv
        frames_out.append(rec)

    y_base = np.stack(y_base_list)
    y_base_delta = np.zeros_like(y_base)
    y_base_delta[:, 2] = np.unwrap(y_base[:, 2]) - y_base[0, 2]
    if not wipe_radius:
        y_base_delta[:, :2] = y_base[:, :2] - y_base[0, :2]

    return {
        "session_path": session["session_path"],
        "hand_side": session["hand_side"],
        "anchor_key": session["anchor_key"],
        "frame_mode": "camera_frame",
        "T_anchor_w": T_ref,
        "T_ref_w": T_ref,
        "T_obj0_w": T_obj0,
        "T_cam0_w": T_cam0,
        "T_floor_w": T_floor,
        "obj_in_ref": {session["anchor_key"]: T_obj_in_ref},
        "T_pivot_w": np.array(
            [[1, 0, 0, pivot[0]], [0, 1, 0, pivot[1]], [0, 0, 1, pivot[2]], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        "T_align": T_align,
        "up_w": up_w,
        "up_mode": up_mode,
        "wipe_radius": wipe_radius,
        "radius_m": radius,
        "fps": session["fps"],
        "yaw": yaws,
        "y_base": y_base,
        "y_base_delta": y_base_delta,
        "frames": frames_out,
    }


def build_session_yaw_gt(
    session_path: str,
    hand_side: str = "right",
    up_mode: str = "camera-neg-y",
    wipe_radius: bool = True,
    mask_init_frames: int = 10,
) -> dict[str, Any]:
    """Per-session cache: OpenCV cam0 ref + de-arced hand/camera (JSON cam0 untouched)."""
    from retarget.mask_anchor import estimate_mask_anchor
    from retarget.session import load_session

    session = load_session(session_path, hand_side=hand_side)
    try:
        mask_info = estimate_mask_anchor(session, n_init=mask_init_frames)
        T_obj0 = mask_info["T_mask_w"]
    except Exception:
        T_obj0 = session["T_anchor_w"]
    yaw_rt = retarget_yaw_world(
        session, T_obj0_w=T_obj0, up_mode=up_mode, wipe_radius=wipe_radius
    )
    by_idx: dict[int, dict[str, Any]] = {}
    for fr in yaw_rt["frames"]:
        T_hand_w = None
        if fr.get("T_hand_in_cam") is not None:
            T_hand_w = fr["T_cam_virt"] @ fr["T_hand_in_cam"]
        by_idx[int(fr["idx"])] = {
            "c2w": fr["T_cam_virt"],
            "T_hand_w": T_hand_w,
            "yaw": float(fr["yaw"]),
        }
    return {
        "T_ref": yaw_rt["T_ref_w"],
        "T_obj0": yaw_rt["T_obj0_w"],
        "T_cam0": yaw_rt["T_cam0_w"],
        "anchor_key": yaw_rt["anchor_key"],
        "by_idx": by_idx,
        "radius_m": yaw_rt["radius_m"],
        "wipe_radius": wipe_radius,
        "frame_mode": "camera_frame",
    }
