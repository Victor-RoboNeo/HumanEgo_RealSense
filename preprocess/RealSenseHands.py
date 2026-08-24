# -*- coding: utf-8 -*-
"""
RealSense RGB-D hand frontend producing HumanEgo's aria_hands.json schema.

The implementation deliberately reuses:
    preprocess.AriaHandsTypes
    preprocess.AriaHandsOptimizer

It follows HumanEgo's own MediaPipe->Aria landmark ordering, but replaces
MediaPipe's monocular depth heuristic with metric D455 depth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils.utils_io import load_cfg
from preprocess.AriaHandsTypes import (
    MidpointFrameBuilder,
    AriaHandsJointAngles,
    AriaHandData,
    AriaHandsData,
    AriaHands,
)
from preprocess.AriaHandsOptimizer import AriaHandsOptimizer
from preprocess.RealSenseGeometry import (
    align_depth_to_color_numpy,
    backproject_pixels,
    fill_missing_landmark_depths,
    robust_local_depth,
)


# HumanEgo MediaPipeHands.py mapping:
# MP_TO_ARIA[aria_idx] = mediapipe_idx
MP_TO_ARIA = [
    4, 8, 12, 16, 20,
    0, 2, 3,
    5, 6, 7,
    9, 10, 11,
    13, 14, 15,
    17, 18, 19,
    -1,
]


def remap_mp_to_aria(kpts_mp_21: np.ndarray) -> np.ndarray:
    out = np.zeros((21, kpts_mp_21.shape[1]), dtype=kpts_mp_21.dtype)
    for aria_idx in range(20):
        out[aria_idx] = kpts_mp_21[MP_TO_ARIA[aria_idx]]
    # Palm center = mean(Wrist, IndexMCP, MiddleMCP), matching upstream.
    out[20] = (
        kpts_mp_21[0] + kpts_mp_21[5] + kpts_mp_21[9]
    ) / 3.0
    return out


class RealSenseHandsGenerator:
    def __init__(
        self,
        mps_path: str,
        cfg_path: str,
        aria_cam,
        session,
        model_path: Optional[str] = None,
        local_depth_radius: int = 5,
        min_valid_depth_landmarks: int = 5,
    ):
        self.mps_path = mps_path
        self.cfg = load_cfg(cfg_path)
        self.aria_cam = aria_cam
        self.session = session
        self.local_depth_radius = int(local_depth_radius)
        self.min_valid_depth_landmarks = int(min_valid_depth_landmarks)

        import mediapipe as mp
        self._mp = mp

        if model_path is None:
            candidate = (
                Path(__file__).resolve().parents[1]
                / "weights" / "mediapipe" / "hand_landmarker.task"
            )
            if candidate.is_file():
                model_path = str(candidate)
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    model_path = hf_hub_download(
                        repo_id="Leo-TX/mediapipe-hand",
                        filename="hand_landmarker.task",
                    )
                except Exception as exc:
                    raise FileNotFoundError(
                        "hand_landmarker.task not found. Put it at "
                        "weights/mediapipe/hand_landmarker.task"
                    ) from exc

        BaseOptions = mp.tasks.BaseOptions
        Options = mp.tasks.vision.HandLandmarkerOptions

        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            Options(
                base_options=BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.mid_builder = MidpointFrameBuilder()

    def _depth_on_color_grid(self, idx: int) -> np.ndarray:
        if self.session.spatial_aligned:
            return self.session.depth_m(idx)

        return align_depth_to_color_numpy(
            depth_raw=self.session.depth_raw(idx, copy=False),
            depth_scale=self.session.depth_scale,
            K_depth=self.session.K_depth,
            K_color=self.session.K_color,
            T_depth_to_color=self.session.T_depth_to_color,
            color_hw=(self.session.height, self.session.width),
            stride=1,
        )

    def _metric_hand_3d(
        self,
        uv_mp: np.ndarray,
        world_lms_mp: np.ndarray,
        depth_color_m: np.ndarray,
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        z = robust_local_depth(
            depth_color_m,
            uv_mp,
            radius=self.local_depth_radius,
            min_depth=0.08,
            max_depth=4.0,
        )

        valid_before = np.isfinite(z) & (z > 0.05)
        if int(np.sum(valid_before)) < self.min_valid_depth_landmarks:
            return None

        z = fill_missing_landmark_depths(z, world_lms_mp)

        if np.sum(np.isfinite(z)) < 18:
            return None

        # Any residual hole uses robust palm depth.
        valid = np.isfinite(z) & (z > 0.05)
        z[~valid] = np.nanmedian(z[valid])

        xyz = backproject_pixels(uv_mp, z, K).astype(np.float32)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    @staticmethod
    def _wrist_frame(kpts_aria: np.ndarray) -> Optional[np.ndarray]:
        wrist = kpts_aria[5]
        palm = kpts_aria[20]
        index_mcp = kpts_aria[8]
        middle_mcp = kpts_aria[11]

        y = palm - wrist
        ny = np.linalg.norm(y)
        if ny < 1e-6:
            return None
        y /= ny

        lateral = index_mcp - middle_mcp
        x = np.cross(y, lateral)
        nx = np.linalg.norm(x)
        if nx < 1e-6:
            return None
        x /= nx

        z = np.cross(x, y)
        z /= np.linalg.norm(z) + 1e-9
        y = np.cross(z, x)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.column_stack([x, y, z])
        T[:3, 3] = wrist
        return T

    def _build_hand(
        self,
        xyz_aria: np.ndarray,
        uv_aria: np.ndarray,
        confidence: float,
        c2w: np.ndarray,
        is_right: bool,
    ) -> Optional[AriaHandData]:
        wrist_pose = self._wrist_frame(xyz_aria)
        if wrist_pose is None:
            return None

        thumb = xyz_aria[0]
        index = xyz_aria[1]
        wrist = xyz_aria[5]
        middle_mcp = xyz_aria[11]

        pinch = float(np.linalg.norm(thumb - index))
        palm_size = float(np.linalg.norm(middle_mcp - wrist))

        # Same scale-normalized choice used by HumanEgo MediaPipe path.
        if palm_size > 0.01:
            grasp_state = int((pinch / palm_size) < 1.0)
        else:
            threshold = float(getattr(self.cfg, "grasp_threshold", 0.105))
            grasp_state = int(pinch < threshold)

        return AriaHandData(
            d2c=np.eye(4, dtype=np.float64),
            c2w=c2w,
            is_right=is_right,
            confidence=float(confidence),
            wrist_pose=wrist_pose,
            palm_pose=wrist_pose.copy(),
            hand_keypoints_3d=xyz_aria,
            hand_keypoints_2d=uv_aria,
            grasp_state=grasp_state,
            joint_angles=AriaHandsJointAngles.from_keypoints_3d(xyz_aria),
        )

    def _filter_by_confidence(self, obj: AriaHands, threshold=0.3):
        for f in obj.hands:
            for attr in ("hand_r", "hand_l"):
                h = getattr(f, attr)
                if h is not None and float(h.confidence) < threshold:
                    setattr(f, attr, None)

    def _suppress_short(self, obj: AriaHands, min_frames: int):
        for attr in ("hand_r", "hand_l"):
            present = [getattr(f, attr) is not None for f in obj.hands]
            i = 0
            while i < len(present):
                if not present[i]:
                    i += 1
                    continue
                j = i
                while j < len(present) and present[j]:
                    j += 1
                if (j - i) < min_frames:
                    for k in range(i, j):
                        setattr(obj.hands[k], attr, None)
                i = j

    def _smooth_grasp(self, obj: AriaHands, size: int):
        size = max(1, int(size))
        for attr in ("hand_r", "hand_l"):
            states = np.array([
                getattr(f, attr).grasp_state
                if getattr(f, attr) is not None else 0
                for f in obj.hands
            ], dtype=np.float32)
            if len(states) == 0:
                continue
            g = (uniform_filter1d(states, size=size) > 0.5).astype(int)
            for i, f in enumerate(obj.hands):
                h = getattr(f, attr)
                if h is not None:
                    h.grasp_state = int(g[i])

    @staticmethod
    def _robust_rot(matrix):
        try:
            return R.from_matrix(matrix)
        except ValueError:
            U, _, Vt = np.linalg.svd(matrix)
            if np.linalg.det(U @ Vt) < 0:
                U[:, -1] *= -1
            return R.from_matrix(U @ Vt)

    def _assign_world_kinematics(self, obj: AriaHands):
        """
        Recompute raw world-space wrist/midpoint states after temporal cleaning.
        This mirrors the data fields consumed by AriaHandsOptimizer/DatasetGen.
        """
        dt = 1.0 / max(float(self.aria_cam.fps), 1e-6)

        caches = {
            "hand_r": {"wrist": None, "mid": None, "Rmid": None},
            "hand_l": {"wrist": None, "mid": None, "Rmid": None},
        }

        for f in obj.hands:
            c2w = self.aria_cam.cam[f.idx].c2w

            for attr in ("hand_r", "hand_l"):
                h = getattr(f, attr)
                if h is None or h.wrist_pose is None:
                    continue

                p_cam = h.wrist_pose[:3, 3]
                R_cam = h.wrist_pose[:3, :3]
                p_w = c2w[:3, :3] @ p_cam + c2w[:3, 3]
                R_w = c2w[:3, :3] @ R_cam

                T_wrist = np.eye(4)
                T_wrist[:3, :3] = R_w
                T_wrist[:3, 3] = p_w
                h.wrist_pose_raw_world = T_wrist

                prev = caches[attr]["wrist"]
                if prev is not None:
                    h.wrist_lin_vel_raw_world = (p_w - prev["p"]) / dt
                    rel = prev["R"].T @ R_w
                    h.wrist_ang_vel_raw_world = (
                        self._robust_rot(rel).as_rotvec() / dt
                    )
                caches[attr]["wrist"] = {"p": p_w, "R": R_w}

                kp = h.hand_keypoints_3d
                if kp is None or len(kp) < 21:
                    continue

                def to_world(p):
                    return c2w[:3, :3] @ p + c2w[:3, 3]

                thumb = to_world(kp[0])
                index = to_world(kp[1])
                thumb_base = to_world(kp[6])
                index_base = to_world(kp[8])

                h.thumb_translation_raw_world = thumb
                h.index_translation_raw_world = index
                h.thumb_base_raw_world = thumb_base
                h.index_base_raw_world = index_base

                midpoint = 0.5 * (thumb + index)
                h.midpoint_translation_raw_world = midpoint

                Rmid = self.mid_builder.build(
                    thumb_w=thumb,
                    index_w=index,
                    thumb_base_w=thumb_base,
                    index_base_w=index_base,
                    wrist_w=p_w,
                    midpoint_w=midpoint,
                    prev_R=caches[attr]["Rmid"],
                )
                if Rmid is None:
                    Rmid = (
                        caches[attr]["Rmid"]
                        if caches[attr]["Rmid"] is not None
                        else R_w.copy()
                    )

                Tmid = np.eye(4)
                Tmid[:3, :3] = Rmid
                Tmid[:3, 3] = midpoint
                h.midpoint_pose_raw_world = Tmid
                h.midpoint_orientation_raw_world = Rmid.flatten()

                prev_mid = caches[attr]["mid"]
                if prev_mid is not None:
                    h.midpoint_lin_vel_raw_world = (
                        midpoint - prev_mid["p"]
                    ) / dt
                    rel = prev_mid["R"].T @ Rmid
                    h.midpoint_ang_vel_raw_world = (
                        self._robust_rot(rel).as_rotvec() / dt
                    )

                caches[attr]["mid"] = {"p": midpoint, "R": Rmid}
                caches[attr]["Rmid"] = Rmid

    def get_aria_hands(self) -> AriaHands:
        obj = AriaHands(mps_path=self.mps_path)

        for i, cam in enumerate(
            tqdm(self.aria_cam.cam, desc="RealSense RGB-D Hands")
        ):
            img_rgb = cv2.cvtColor(cam.img, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=img_rgb,
            )
            result = self.landmarker.detect(mp_img)

            depth_color = self._depth_on_color_grid(i)

            hand_r = None
            hand_l = None

            if result.hand_landmarks and result.handedness:
                for lms, world_lms, handedness in zip(
                    result.hand_landmarks,
                    result.hand_world_landmarks,
                    result.handedness,
                ):
                    label = handedness[0].category_name
                    confidence = float(handedness[0].score)

                    uv_mp = np.array(
                        [[lm.x * cam.w, lm.y * cam.h] for lm in lms],
                        dtype=np.float32,
                    )
                    world_mp = np.array(
                        [[lm.x, lm.y, lm.z] for lm in world_lms],
                        dtype=np.float32,
                    )

                    xyz_mp = self._metric_hand_3d(
                        uv_mp,
                        world_mp,
                        depth_color,
                        cam.k,
                    )
                    if xyz_mp is None:
                        continue

                    xyz_aria = remap_mp_to_aria(xyz_mp)
                    uv_aria = remap_mp_to_aria(
                        np.column_stack([uv_mp, np.zeros(21, dtype=np.float32)])
                    )[:, :2]

                    h = self._build_hand(
                        xyz_aria,
                        uv_aria,
                        confidence,
                        cam.c2w,
                        is_right=(label == "Right"),
                    )
                    if h is None:
                        continue

                    if label == "Right":
                        if hand_r is None or h.confidence > hand_r.confidence:
                            hand_r = h
                    else:
                        if hand_l is None or h.confidence > hand_l.confidence:
                            hand_l = h

            obj.hands.append(
                AriaHandsData(
                    idx=cam.idx,
                    ts=cam.ts,
                    hand_r=hand_r,
                    hand_l=hand_l,
                )
            )
            obj.tss.append(cam.ts)

        self._filter_by_confidence(obj, threshold=0.3)
        self._suppress_short(
            obj,
            min_frames=int(getattr(self.cfg, "hand_min_frames", 5)),
        )

        # Populate the raw world-space fields before HumanEgo's own optimizer.
        self._assign_world_kinematics(obj)

        self._smooth_grasp(
            obj,
            size=int(getattr(self.cfg, "grasp_smooth_win", 5)),
        )

        optimizer = AriaHandsOptimizer(
            self.cfg,
            1.0 / max(float(self.aria_cam.fps), 1e-6),
        )
        optimizer.run(obj)

        self._smooth_grasp(
            obj,
            size=int(getattr(self.cfg, "grasp_smooth_win", 5)),
        )

        try:
            self.landmarker.close()
        except Exception:
            pass

        return obj
