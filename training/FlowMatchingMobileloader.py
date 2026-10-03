# -*- coding: utf-8 -*-
"""Mobile-manipulation dataloader: camera-frame y_action + SE(2) y_base.

ICT / y_action use JSON `world_transforms.cam0` (paper / FlowMatchingDataloader).
`wipe_cam_radius` only de-arcs the hand/camera in world; cam0 is not rewritten.

    T_hand_to_world ← T_cam_virt @ T_hand_in_cam
    y_base          ← (xy of T_cam_virt in cam0, gravity yaw)

Replay:

    T_ee_in_base = inv(T_base_world) @ T_ee_world
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from training.FlowMatchingDataloader import FlowMatchingDataloader
from utils.utils_math import normalize_pos


def _wrap_pi(yaw: np.ndarray) -> np.ndarray:
    return (yaw + np.pi) % (2.0 * np.pi) - np.pi


def _se2_from_T_world_zup(T: np.ndarray) -> np.ndarray:
    """Project a 4x4 world pose onto SE(2) assuming world Z is up (Aria MPS)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    t = T[:3, 3]
    R = T[:3, :3]
    fwd = R[:, 2].copy()
    fwd[2] = 0.0
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        fwd = R[:, 0].copy()
        fwd[2] = 0.0
        n = float(np.linalg.norm(fwd)) + 1e-8
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    out = np.eye(4, dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    out[0, 0], out[0, 1] = c, -s
    out[1, 0], out[1, 1] = s, c
    out[0, 3], out[1, 3], out[2, 3] = t[0], t[1], t[2]
    return out


def _se2_label_in_ref(T_base_ref: np.ndarray) -> np.ndarray:
    """(x, y, yaw) of a 4x4 expressed in the reference frame's XY."""
    R = T_base_ref[:3, :3]
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([T_base_ref[0, 3], T_base_ref[1, 3], yaw], dtype=np.float32)


def _session_path_from_json(json_path: str) -> str:
    # .../<session>/preprocess/all_data/<frame>/training_data.json
    frame_dir = os.path.dirname(json_path)
    all_data = os.path.dirname(frame_dir)
    preprocess = os.path.dirname(all_data)
    return os.path.dirname(preprocess)


class FlowMatchingMobileloader(FlowMatchingDataloader):
    """ICT / y_action in JSON OpenCV cam0; y_base is in-place gravity yaw after wiping the arc."""

    def __init__(
        self,
        *args,
        T_cam_in_base=None,
        wipe_cam_radius: bool = True,
        yaw_up: str = "camera-neg-y",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if T_cam_in_base is None:
            self.T_cam_in_base = np.eye(4, dtype=np.float32)
        else:
            self.T_cam_in_base = np.array(T_cam_in_base, dtype=np.float32).reshape(4, 4)
        self.wipe_cam_radius = bool(wipe_cam_radius)
        self.yaw_up = yaw_up
        self._yaw_gt_cache: dict[str, dict[str, Any]] = {}
        print(
            f"  - Mobile base  : T_cam_in_base={'I' if np.allclose(self.T_cam_in_base, np.eye(4)) else 'custom'}"
            f"  wipe_cam_radius={self.wipe_cam_radius}  up={self.yaw_up}"
            f"  frame_mode={self.frame_mode}"
        )

    def _get_yaw_gt(self, json_path: str) -> Optional[dict[str, Any]]:
        session_path = _session_path_from_json(json_path)
        cached = self._yaw_gt_cache.get(session_path)
        if cached is not None:
            return cached if cached else None
        try:
            from retarget.yaw_world import build_session_yaw_gt

            gt = build_session_yaw_gt(
                session_path,
                hand_side=self.single_hand_side,
                up_mode=self.yaw_up,
                wipe_radius=True,
            )
            self._yaw_gt_cache[session_path] = gt
            print(
                f"[FlowMatchingMobileloader] camera_frame GT  "
                f"session={os.path.basename(session_path)}  "
                f"r={gt['radius_m']*100:.1f} cm  frames={len(gt['by_idx'])}"
            )
            return gt
        except Exception as exc:
            print(f"[FlowMatchingMobileloader] camera_frame GT failed for {session_path}: {exc}")
            self._yaw_gt_cache[session_path] = {}
            return None

    def _read_json(self, path: str):
        d = self._read_json_raw(path)
        if not d or not self.wipe_cam_radius or not path:
            return d
        gt = self._get_yaw_gt(path)
        if not gt:
            return d
        try:
            idx = int(d.get("metadata", {}).get("idx", os.path.basename(os.path.dirname(path))))
        except Exception:
            return d
        rec = gt["by_idx"].get(idx)
        if rec is None:
            return d
        out = copy.deepcopy(d)
        key = self.hand_entity_key
        side = self.single_hand_side
        if rec["T_hand_w"] is not None:
            hands = out.setdefault("entities", {}).setdefault(key, {}) or {}
            if side in hands:
                hands[side]["T_hand_to_world"] = rec["T_hand_w"].tolist()
            out["entities"][key] = hands
        return out

    def _T_base_world_from_frame(self, d: dict, json_path: str | None = None) -> Optional[np.ndarray]:
        if self.wipe_cam_radius and json_path:
            gt = self._get_yaw_gt(json_path)
            if gt:
                try:
                    idx = int(d.get("metadata", {}).get("idx", -1))
                except Exception:
                    idx = -1
                rec = gt["by_idx"].get(idx)
                if rec is not None:
                    T_cam_w = np.asarray(rec["c2w"], dtype=np.float64).reshape(4, 4)
                    T_base = T_cam_w @ np.linalg.inv(self.T_cam_in_base.astype(np.float64))
                    return T_base.astype(np.float32)
        c2w = d.get("metadata", {}).get("c2w")
        if c2w is None:
            return None
        try:
            T_cam_w = np.array(c2w, dtype=np.float64).reshape(4, 4)
        except Exception:
            return None
        T_base_full = T_cam_w @ np.linalg.inv(self.T_cam_in_base.astype(np.float64))
        return _se2_from_T_world_zup(T_base_full)

    def _encode_base_label(self, se2_ref: np.ndarray) -> np.ndarray:
        xy = normalize_pos(se2_ref[:2], self.pos_mean[:2], self.pos_std[:2])
        yaw = float(_wrap_pi(np.array(se2_ref[2], dtype=np.float32))) / np.pi
        return np.array([xy[0], xy[1], yaw], dtype=np.float32)

    def _build_targets(
        self,
        json_path0: str,
        T_w2ref_base: np.ndarray,
        idx: int,
        stride: int,
        d0: dict,
    ) -> Dict[str, torch.Tensor]:
        y_dict = super()._build_targets(json_path0, T_w2ref_base, idx, stride, d0)

        K = self.pred_horizon
        y_base = np.zeros((K, 3), dtype=np.float32)
        se2_ref0 = None

        for k in range(1, K + 1):
            fut_path = self._get_future_json_path(json_path0, k * stride)
            dk = self._read_json(fut_path)
            if not dk:
                if k > 1:
                    y_base[k - 1] = y_base[k - 2]
                continue

            T_base_w = self._T_base_world_from_frame(dk, fut_path)
            if T_base_w is None:
                if k > 1:
                    y_base[k - 1] = y_base[k - 2]
                continue

            T_base_ref = T_w2ref_base @ T_base_w
            se2_ref = _se2_label_in_ref(T_base_ref)
            if self.wipe_cam_radius:
                gt = self._get_yaw_gt(fut_path)
                try:
                    fut_idx = int(dk.get("metadata", {}).get("idx", -1))
                except Exception:
                    fut_idx = -1
                rec = (gt or {}).get("by_idx", {}).get(fut_idx)
                if rec is not None and rec.get("yaw") is not None:
                    se2_ref = np.array(
                        [se2_ref[0], se2_ref[1], float(rec["yaw"])], dtype=np.float32
                    )
            if se2_ref0 is None:
                se2_ref0 = se2_ref.copy()

            if self.action_mode == "delta":
                dxy = se2_ref[:2] - se2_ref0[:2]
                dyaw = float(_wrap_pi(np.array(se2_ref[2] - se2_ref0[2], dtype=np.float32)))
                y_base[k - 1] = self._encode_base_label(
                    np.array([dxy[0], dxy[1], dyaw], dtype=np.float32)
                )
            else:
                y_base[k - 1] = self._encode_base_label(se2_ref)

        y_dict["y_base"] = torch.from_numpy(y_base).float()
        return y_dict
