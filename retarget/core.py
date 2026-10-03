# -*- coding: utf-8 -*-
"""Retarget HumanEgo preprocess trajectories into a shared reference frame.

Paper App. E.3 / FlowMatchingDataloader camera_frame:

    T_ref          = metadata.world_transforms["cam0"]   # static OpenCV cam0
    T_hand_in_ref  = inv(T_ref) @ T_hand_w
    T_cam_in_ref   = inv(T_ref) @ T_cam_w

cam0 is X-right, Y-down, Z-forward. With T_cam_in_base = I:

    T_ee_in_base   = inv(T_cam_in_ref) @ T_hand_in_ref @ inv(T_align)
                   = T_hand_in_cam @ inv(T_align)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from retarget.geometry import as_T, invert


def retarget_frames(
    session: dict[str, Any],
    T_align: np.ndarray | None = None,
    T_anchor_w: np.ndarray | None = None,
    frame_mode: str = "camera_frame",
    up_mode: str = "camera-neg-y",
) -> dict[str, Any]:
    del up_mode  # kept so run.py can pass it; unused for OpenCV cam0
    T_align = np.eye(4, dtype=np.float64) if T_align is None else as_T(T_align)
    T_align_inv = invert(T_align)
    T_cam0 = as_T(session.get("T_cam0_w", session["frames"][0]["T_cam_w"]))

    if frame_mode == "camera_frame":
        T_ref = T_cam0
    elif frame_mode == "anchor_frame":
        T_ref = as_T(session["T_anchor_w"] if T_anchor_w is None else T_anchor_w)
    else:
        raise ValueError(f"Unknown frame_mode: {frame_mode}")
    T_w2ref = invert(T_ref)

    out_frames: list[dict[str, Any]] = []
    for fr in session["frames"]:
        T_cam_w = as_T(fr["T_cam_w"])
        T_cam_in_ref = T_w2ref @ T_cam_w
        rec: dict[str, Any] = {
            "idx": fr["idx"],
            "ts": fr["ts"],
            "K": fr["K"],
            "w": fr["w"],
            "h": fr["h"],
            "rgb_path": fr["rgb_path"],
            "mask_obj1_path": fr.get("mask_obj1_path"),
            "is_finished": fr["is_finished"],
            "T_cam_w": T_cam_w,
            "T_cam_in_ref": T_cam_in_ref,
            "T_hand_in_ref": None,
            "T_hand_in_cam": None,
            "T_ee_in_base": None,
            "grasp": fr["grasp"],
        }
        if fr["T_hand_w"] is not None:
            T_hand_in_ref = T_w2ref @ as_T(fr["T_hand_w"])
            T_hand_in_cam = invert(T_cam_in_ref) @ T_hand_in_ref
            rec["T_hand_in_ref"] = T_hand_in_ref
            rec["T_hand_in_cam"] = T_hand_in_cam
            rec["T_ee_in_base"] = T_hand_in_cam @ T_align_inv
        out_frames.append(rec)

    obj_in_ref = {}
    for k, T_ow in session["frames"][0]["obj_poses_w"].items():
        obj_in_ref[k] = T_w2ref @ as_T(T_ow)

    return {
        "session_path": session["session_path"],
        "hand_side": session["hand_side"],
        "anchor_key": session["anchor_key"],
        "frame_mode": frame_mode,
        "T_anchor_w": T_ref,
        "T_ref_w": T_ref,
        "T_w2ref": T_w2ref,
        "T_align": T_align,
        "fps": session["fps"],
        "obj_in_ref": obj_in_ref,
        "frames": out_frames,
    }


def stack_traj(frames: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,4,4) poses and (N,) frame indices for frames where `key` is set."""
    Ts, idxs = [], []
    for fr in frames:
        T = fr.get(key)
        if T is None:
            continue
        Ts.append(T)
        idxs.append(fr["idx"])
    if not Ts:
        return np.zeros((0, 4, 4), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return np.stack(Ts, axis=0), np.asarray(idxs, dtype=np.int32)
