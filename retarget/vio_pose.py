# -*- coding: utf-8 -*-
"""Replace the preprocess ICP c2w with vio_c2w.csv for replay.

Hand and object poses are rebased with the same per-frame camera change, so
a point in the camera (WiLoR + depth) stays put. Only the camera trajectory
that supplies yaw changes.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.geometry import as_T, invert


def load_vio_c2w(session_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(session_path) / "vio_c2w.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise RuntimeError(f"empty {path}")
    ts = np.array([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    trans = np.array([[float(row[k]) for k in ("tx", "ty", "tz")] for row in rows])
    quat = np.array([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in rows])
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    poses[:, :3, 3] = trans
    return ts, poses


def apply_vio_c2w(session: dict[str, Any]) -> dict[str, Any]:
    """Write VIO color-camera c2w into the loaded session, in place."""
    vio_ts, vio_T = load_vio_c2w(session["session_path"])
    frames = session["frames"]
    if len(vio_T) != len(frames):
        raise RuntimeError(f"vio frames {len(vio_T)} != session frames {len(frames)}")
    frame_ts = np.array([int(fr["ts"]) for fr in frames], dtype=np.int64)
    dt_ms = np.abs(vio_ts.astype(np.float64) - frame_ts.astype(np.float64)) / 1e6
    if float(np.max(dt_ms)) > 1.0:
        raise RuntimeError(f"vio timestamps do not match the session (max {dt_ms.max():.3f} ms)")

    T_icp0 = as_T(frames[0]["T_cam_w"])
    for fr, T_vio in zip(frames, vio_T):
        T_icp = as_T(fr["T_cam_w"])
        rebase = T_vio @ invert(T_icp)
        if fr["T_hand_w"] is not None:
            fr["T_hand_w"] = rebase @ as_T(fr["T_hand_w"])
        fr["obj_poses_w"] = {key: rebase @ as_T(pose) for key, pose in fr["obj_poses_w"].items()}
        fr["T_cam_w"] = np.asarray(T_vio, dtype=np.float64)

    world0 = vio_T[0] @ invert(T_icp0)
    session["T_cam0_w"] = world0 @ as_T(session["T_cam0_w"])
    session["T_anchor_w"] = world0 @ as_T(session["T_anchor_w"])
    for fr in frames:
        fr["T_cam0_w"] = session["T_cam0_w"]
        fr["T_anchor_w"] = session["T_anchor_w"]
    session["pose_source"] = "vio"
    return session
