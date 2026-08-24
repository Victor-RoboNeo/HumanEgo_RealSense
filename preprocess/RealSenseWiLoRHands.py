# -*- coding: utf-8 -*-
"""
RealSense + WiLoR hand frontend for HumanEgo.

Purpose
-------
Replace the MediaPipe stage in the RealSense HumanEgo frontend with WiLoR,
while preserving HumanEgo's Aria-compatible hand schema and downstream code.

Design
------
RGB -> WiLoR hand detection + 21-joint reconstruction
         |
         +-> D455 aligned depth, when available
              -> robust metric scale/translation fitting
         |
         v
Aria 21-joint ordering
-> AriaHandData
-> HumanEgo temporal cleaning
-> AriaHandsOptimizer
-> aria_hands.json

This class deliberately subclasses HumanEgo's own WiLoRHandsGenerator so that
wrist frame construction, grasp estimation, midpoint/gripper construction,
temporal interpolation, optimization, and JSON schema stay consistent with
upstream HumanEgo.

Expected HumanEgo modules:
    preprocess/WiLoRHands.py
    preprocess/RealSenseSession.py

The current HumanEgo WiLoR adapter uses wilor-mini:
    wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from preprocess.WiLoRHands import (
    WiLoRHandsGenerator,
    remap_wilor_to_aria,
)
from preprocess.AriaHandsTypes import AriaHands, AriaHandsData
from preprocess.AriaHandsOptimizer import AriaHandsOptimizer
from preprocess.AriaHandsOps import AriaHandsOps

try:
    from preprocess.RealSenseSession import RealSenseSession
except Exception:
    RealSenseSession = None

try:
    from preprocess.RealSenseGeometry import align_depth_to_color_numpy
except Exception:
    align_depth_to_color_numpy = None


def _local_median_depth(
    depth_m: np.ndarray,
    uv: np.ndarray,
    radius: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Robust local-median depth lookup for N pixel coordinates."""
    h, w = depth_m.shape[:2]
    out = np.full((len(uv),), np.nan, dtype=np.float32)

    for i, (u, v) in enumerate(np.asarray(uv, dtype=np.float32)):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)

        vals = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float32).reshape(-1)
        vals = vals[
            np.isfinite(vals)
            & (vals >= float(min_depth))
            & (vals <= float(max_depth))
        ]
        if vals.size:
            out[i] = float(np.median(vals))

    return out


def _backproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel + metric Z -> XYZ in the RGB camera frame."""
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    xyz = np.empty((len(uv), 3), dtype=np.float64)
    xyz[:, 2] = z
    xyz[:, 0] = (uv[:, 0] - cx) * z / fx
    xyz[:, 1] = (uv[:, 1] - cy) * z / fy
    return xyz


def _robust_scalar_similarity(
    q: np.ndarray,
    p: np.ndarray,
    *,
    fit_scale: bool = True,
    scale_min: float = 0.55,
    scale_max: float = 1.80,
    huber_delta_m: float = 0.035,
    iterations: int = 5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Fit p ~= s*q + t with IRLS Huber weights.

    Rotation is intentionally NOT fitted. WiLoR already predicts articulation
    and orientation in camera coordinates; D455 depth is used mainly to pin
    metric scale/translation and reject impossible depth.

    Returns:
        scale, translation(3,), residuals(N,)
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q and p must both have shape (N,3)")
    if len(q) == 0:
        raise ValueError("no points")

    w = np.ones((len(q),), dtype=np.float64)
    s = 1.0
    t = np.median(p - q, axis=0)

    for _ in range(max(1, int(iterations))):
        ws = float(np.sum(w)) + 1e-12
        q_bar = np.sum(q * w[:, None], axis=0) / ws
        p_bar = np.sum(p * w[:, None], axis=0) / ws

        if fit_scale and len(q) >= 4:
            dq = q - q_bar
            dp = p - p_bar
            den = float(np.sum(w[:, None] * dq * dq))
            if den > 1e-10:
                s = float(np.sum(w[:, None] * dq * dp) / den)
                s = float(np.clip(s, scale_min, scale_max))
            else:
                s = 1.0
        else:
            s = 1.0

        t = p_bar - s * q_bar
        r = np.linalg.norm(s * q + t[None, :] - p, axis=1)

        delta = max(float(huber_delta_m), 1e-4)
        w = np.ones_like(r)
        bad = r > delta
        w[bad] = delta / np.maximum(r[bad], 1e-9)

    residuals = np.linalg.norm(s * q + t[None, :] - p, axis=1)
    return s, t, residuals


class RealSenseWiLoRHandsGenerator(WiLoRHandsGenerator):
    """
    WiLoR-based HumanEgo hand generator with optional D455 metric-depth fusion.

    Constructor is intentionally compatible with typical RealSenseHandsGenerator
    call sites. Extra positional/keyword arguments are tolerated so the add-on
    can be installed without rewriting unrelated preprocessing code.
    """

    def __init__(
        self,
        mps_path: str,
        cfg_path: str,
        aria_cam,
        session=None,
        *args,
        **kwargs,
    ):
        super().__init__(mps_path, cfg_path, aria_cam)

        if session is not None:
            self.rs_session = session
        elif RealSenseSession is not None:
            try:
                self.rs_session = RealSenseSession(mps_path)
            except Exception as exc:
                print(f"[WiLoR-RGBD] RealSenseSession unavailable: {exc}")
                self.rs_session = None
        else:
            self.rs_session = None

        # RealSense-specific defaults. These do not have to exist in AriaHands.yaml.
        self.depth_radius = int(
            getattr(self.cfg, "realsense_wilor_depth_radius", 5)
        )
        self.min_depth_m = float(
            getattr(self.cfg, "realsense_wilor_min_depth_m", 0.08)
        )
        self.max_depth_m = float(
            getattr(self.cfg, "realsense_wilor_max_depth_m", 2.5)
        )
        self.min_depth_joints = int(
            getattr(self.cfg, "realsense_wilor_min_depth_joints", 4)
        )
        self.fit_scale = bool(
            getattr(self.cfg, "realsense_wilor_fit_scale", True)
        )
        self.scale_min = float(
            getattr(self.cfg, "realsense_wilor_scale_min", 0.55)
        )
        self.scale_max = float(
            getattr(self.cfg, "realsense_wilor_scale_max", 1.80)
        )
        self.huber_delta_m = float(
            getattr(self.cfg, "realsense_wilor_huber_delta_m", 0.035)
        )
        self.depth_residual_gate_m = float(
            getattr(self.cfg, "realsense_wilor_depth_residual_gate_m", 0.08)
        )

        # IMPORTANT: upstream Aria hand_min_frames is designed for MPS tracks and
        # can be too aggressive for frame-wise vision estimators.
        self.conf_threshold = float(
            getattr(self.cfg, "realsense_wilor_conf_threshold", 0.20)
        )
        self.min_frames = int(
            getattr(self.cfg, "realsense_wilor_min_frames", 3)
        )
        print("self.min_frames: ", self.min_frames)
        self.interp_gap = int(
            getattr(self.cfg, "realsense_wilor_interp_gap", 5)
        )

        self.stats = {
            "frames": 0,
            "wilor_detections": 0,
            "right_detections": 0,
            "left_detections": 0,
            "rgbd_fused": 0,
            "mono_fallback": 0,
            "depth_failed": 0,
            "pre_filter_right": 0,
            "pre_filter_left": 0,
            "post_filter_right": 0,
            "post_filter_left": 0,
        }

    # ------------------------------------------------------------------
    # RGB-D access / fusion
    # ------------------------------------------------------------------

    def _depth_color_m(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.rs_session is None:
            return None

        try:
            if bool(getattr(self.rs_session, "spatial_aligned", False)):
                d = self.rs_session.depth_m(frame_idx)
                return np.asarray(d, dtype=np.float32)

            # If depth is not already color-aligned, use the geometry helper
            # from the RealSense frontend when available.
            if align_depth_to_color_numpy is not None:
                raw = self.rs_session.depth_raw(frame_idx, copy=False)
                return align_depth_to_color_numpy(
                    depth_raw=raw,
                    depth_scale=self.rs_session.depth_scale,
                    K_depth=self.rs_session.K_depth,
                    K_color=self.rs_session.K_color,
                    T_depth_to_color=self.rs_session.T_depth_to_color,
                    color_hw=(self.rs_session.height, self.rs_session.width),
                    stride=1,
                ).astype(np.float32)

            # Last fallback: session may already expose depth_m in color pixels.
            return np.asarray(self.rs_session.depth_m(frame_idx), dtype=np.float32)

        except Exception:
            return None

    def _fuse_wilor_with_depth(
        self,
        kpts_2d_wilor: np.ndarray,
        kpts_3d_rel: np.ndarray,
        K: np.ndarray,
        depth_m: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Align WiLoR's relative 3D hand to D455 metric 3D samples.

        Returns camera-frame WiLoR-order 21x3 points and diagnostics.
        """
        diag = {
            "used_depth": False,
            "num_depth_joints": 0,
            "scale": 1.0,
            "median_residual_m": None,
        }

        if depth_m is None:
            return None, diag

        z = _local_median_depth(
            depth_m,
            kpts_2d_wilor,
            radius=self.depth_radius,
            min_depth=self.min_depth_m,
            max_depth=self.max_depth_m,
        )
        valid = np.isfinite(z)

        if int(valid.sum()) < self.min_depth_joints:
            return None, diag

        p_depth = _backproject(
            np.asarray(kpts_2d_wilor)[valid],
            z[valid],
            K,
        )
        q = np.asarray(kpts_3d_rel, dtype=np.float64)[valid]

        try:
            s, t, residual = _robust_scalar_similarity(
                q,
                p_depth,
                fit_scale=self.fit_scale,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                huber_delta_m=self.huber_delta_m,
                iterations=5,
            )

            # One robust re-fit after removing obvious depth/background outliers.
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med))) + 1e-6
            gate = min(
                self.depth_residual_gate_m,
                med + 3.5 * 1.4826 * mad,
            )
            inlier = residual <= max(gate, 0.02)

            if int(inlier.sum()) >= self.min_depth_joints:
                s, t, residual = _robust_scalar_similarity(
                    q[inlier],
                    p_depth[inlier],
                    fit_scale=self.fit_scale,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    huber_delta_m=self.huber_delta_m,
                    iterations=5,
                )

            out = (
                s * np.asarray(kpts_3d_rel, dtype=np.float64)
                + t[None, :]
            ).astype(np.float32)

            if (
                not np.all(np.isfinite(out))
                or np.any(out[:, 2] <= 0.02)
                or np.median(out[:, 2]) > self.max_depth_m + 0.5
            ):
                return None, diag

            diag.update({
                "used_depth": True,
                "num_depth_joints": int(valid.sum()),
                "scale": float(s),
                "median_residual_m": float(np.median(residual)),
            })
            return out, diag

        except Exception:
            return None, diag

    def _monocular_fallback(
        self,
        wilor_preds: dict,
        kpts_2d_wilor: np.ndarray,
        kpts_3d_wilor: np.ndarray,
        K: np.ndarray,
        h_img: int,
        w_img: int,
    ) -> Optional[np.ndarray]:
        # Same primary path as upstream HumanEgo WiLoRHands.py.
        if "pred_cam_t_full" in wilor_preds:
            pred_cam_t = np.asarray(
                wilor_preds["pred_cam_t_full"][0],
                dtype=np.float32,
            )
            joints_cam = kpts_3d_wilor + pred_cam_t[None, :]
            wrist_z = float(joints_cam[0, 2])
            if 0.05 < wrist_z < 3.0:
                return joints_cam.astype(np.float32)

        return self._recover_absolute_3d(
            kpts_2d_wilor,
            kpts_3d_wilor,
            K,
            h_img,
            w_img,
        )

    # ------------------------------------------------------------------
    # Main HumanEgo-compatible pipeline
    # ------------------------------------------------------------------

    def get_aria_hands(self) -> AriaHands:
        aria_hands = AriaHands(mps_path=self.mps_path)
        dt = 1.0 / float(self.aria_cam.fps)

        # Reset caches when reusing an object.
        self.prev_r_cache = None
        self.prev_l_cache = None
        self.prev_r_mid_cache = None
        self.prev_l_mid_cache = None
        self.prev_r_mid_R = None
        self.prev_l_mid_R = None

        for cam_data in tqdm(
            self.aria_cam.cam,
            total=len(self.aria_cam),
            desc="RealSense WiLoR Hands",
        ):
            self.stats["frames"] += 1

            img_bgr = getattr(cam_data, "img", None)
            if img_bgr is None:
                frame_dir = os.path.join(
                    self.mps_path,
                    "preprocess",
                    "all_data",
                    f"{cam_data.idx:05d}",
                )
                for name in ("rgb.png", "aria_cam_rgb.jpg", "aria_cam_rgb.png"):
                    p = os.path.join(frame_dir, name)
                    if os.path.isfile(p):
                        img_bgr = cv2.imread(p)
                        if img_bgr is not None:
                            break

            if img_bgr is None:
                aria_hands.hands.append(
                    AriaHandsData(cam_data.idx, cam_data.ts)
                )
                aria_hands.tss.append(cam_data.ts)
                continue

            h_img, w_img = img_bgr.shape[:2]
            K = np.asarray(cam_data.k, dtype=np.float64)
            c2w = np.asarray(cam_data.c2w, dtype=np.float64)
            depth_m = self._depth_color_m(int(cam_data.idx))

            # Capture detector confidence per handedness, as upstream HumanEgo does.
            cls_best_conf = {}
            try:
                yolo_res = self.wilor_pipe.hand_detector(
                    img_bgr,
                    conf=self.conf_threshold,
                    verbose=False,
                )[0]
                for yd in yolo_res:
                    bd = yd.boxes.data.cpu().numpy().squeeze()
                    if bd.ndim >= 1 and bd.shape[-1] >= 5:
                        cls_id = int(yd.boxes.cls.cpu().item())
                        score = float(bd[4])
                        cls_best_conf[cls_id] = max(
                            score,
                            cls_best_conf.get(cls_id, 0.0),
                        )
            except Exception:
                pass

            try:
                detections = self.wilor_pipe.predict(img_bgr)
            except Exception as exc:
                print(f"[WiLoR-RGBD] frame {cam_data.idx}: predict failed: {exc}")
                detections = []

            hand_r = None
            hand_l = None

            for det in detections:
                wilor_preds = det.get("wilor_preds")
                if wilor_preds is None:
                    continue

                self.stats["wilor_detections"] += 1

                is_right = bool(det.get("is_right", 1.0) >= 0.5)
                if is_right:
                    self.stats["right_detections"] += 1
                else:
                    self.stats["left_detections"] += 1

                kpts_2d_wilor = np.asarray(
                    wilor_preds["pred_keypoints_2d"][0],
                    dtype=np.float32,
                )
                kpts_3d_wilor = np.asarray(
                    wilor_preds["pred_keypoints_3d"][0],
                    dtype=np.float32,
                )

                kpts_cam, depth_diag = self._fuse_wilor_with_depth(
                    kpts_2d_wilor,
                    kpts_3d_wilor,
                    K,
                    depth_m,
                )

                if kpts_cam is not None:
                    self.stats["rgbd_fused"] += 1
                else:
                    self.stats["depth_failed"] += 1
                    kpts_cam = self._monocular_fallback(
                        wilor_preds,
                        kpts_2d_wilor,
                        kpts_3d_wilor,
                        K,
                        h_img,
                        w_img,
                    )
                    if kpts_cam is not None:
                        self.stats["mono_fallback"] += 1

                if kpts_cam is None:
                    continue

                confidence = cls_best_conf.get(
                    1 if is_right else 0,
                    None,
                )
                if confidence is None:
                    confidence = self._compute_reproj_confidence(
                        kpts_cam,
                        kpts_2d_wilor,
                        K,
                        h_img,
                        w_img,
                    )

                kpts_cam_aria = remap_wilor_to_aria(kpts_cam)
                kpts_2d_aria = remap_wilor_to_aria(
                    np.column_stack([
                        kpts_2d_wilor,
                        np.zeros((21,), dtype=np.float32),
                    ])
                )[:, :2]

                h_data = self._build_hand_data(
                    kpts_cam_aria,
                    kpts_2d_aria,
                    float(confidence),
                    c2w,
                    K,
                    h_img,
                    w_img,
                    is_right=is_right,
                )

                # Useful for debugging; serializer ignores unknown attrs in most
                # HumanEgo versions, so keep diagnostics private when unsupported.
                try:
                    h_data.realsense_wilor_depth_diag = depth_diag
                except Exception:
                    pass

                if is_right:
                    if hand_r is None or confidence > hand_r.confidence:
                        hand_r = h_data
                else:
                    if hand_l is None or confidence > hand_l.confidence:
                        hand_l = h_data

            frame_data = AriaHandsData(
                cam_data.idx,
                cam_data.ts,
                hand_r,
                hand_l,
            )
            self._compute_and_assign_vel(frame_data, c2w, dt)

            aria_hands.hands.append(frame_data)
            aria_hands.tss.append(cam_data.ts)

        self.stats["pre_filter_right"] = sum(
            x.hand_r is not None for x in aria_hands.hands
        )
        self.stats["pre_filter_left"] = sum(
            x.hand_l is not None for x in aria_hands.hands
        )

        # RealSense/WiLoR-specific temporal cleaning.
        self._filter_by_confidence(
            aria_hands,
            conf_th=self.conf_threshold,
        )
        self._suppress_short_hands(
            aria_hands,
            min_frames=self.min_frames,
        )
        self._interpolate_hand_trajectories(
            aria_hands,
            max_gap=self.interp_gap,
        )
        self._smooth_grasp_detection(
            aria_hands,
            size=self.cfg.grasp_smooth_win,
        )

        self.stats["post_filter_right"] = sum(
            x.hand_r is not None for x in aria_hands.hands
        )
        self.stats["post_filter_left"] = sum(
            x.hand_l is not None for x in aria_hands.hands
        )

        optimizer = AriaHandsOptimizer(self.cfg, dt)
        optimizer.run(aria_hands)
        self._smooth_grasp_detection(
            aria_hands,
            size=self.cfg.grasp_smooth_win,
        )

        os.makedirs(
            os.path.join(self.mps_path, "preprocess"),
            exist_ok=True,
        )

        try:
            AriaHandsOps.save_hands_analysis_plots_two(
                aria_hands,
                os.path.join(self.mps_path, "preprocess"),
                dt,
                self.cfg,
            )
        except Exception as exc:
            print(f"[WiLoR-RGBD] analysis plot warning: {exc}")

        self._print_stats()
        return aria_hands

    def _print_stats(self) -> None:
        s = self.stats
        print()
        print("=" * 58)
        print(" RealSense + WiLoR hand frontend")
        print("=" * 58)
        for key in (
            "frames",
            "wilor_detections",
            "right_detections",
            "left_detections",
            "rgbd_fused",
            "mono_fallback",
            "depth_failed",
            "pre_filter_right",
            "pre_filter_left",
            "post_filter_right",
            "post_filter_left",
        ):
            print(f"{key:24s}: {s[key]}")
        print("=" * 58)
        print()
