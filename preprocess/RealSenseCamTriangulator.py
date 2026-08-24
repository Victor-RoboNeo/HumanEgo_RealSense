# -*- coding: utf-8 -*-
"""
RealSense RGB-D replacement for HumanEgo CamTriangulator.

Why:
    Original HumanEgo CamTriangulator performs multi-view DLT, which requires
    sufficient camera parallax. With D455 RGB-D, every tracked pixel already has
    metric depth, so we directly lift CoTracker points to 3D and robustly fuse
    them across the object-centric frames.

Output schema is intentionally compatible with HumanEgo DatasetGen:
    preprocess/camtriangulator_results.json
        cam0_c2w
        objects/<obj>/
            points_3d_world
            points_3d_cam0
            object_to_cam0_matrix
            info
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

from utils.utils_io import load_cfg
from preprocess.OrientAnything import (
    estimate_frame_pca1,
    estimate_frame_pca2,
    estimate_frame_vlm,
)
from preprocess.RealSenseSession import RealSenseSession
from preprocess.RealSenseGeometry import (
    align_depth_to_color_numpy,
    backproject_pixels,
    robust_local_depth,
)


class RealSenseCamTriangulatorManager:
    def __init__(self):
        self.initialized = False
        self.objects_3d = {}
        self.T_limit = 0
        self.T_c02w = None
        self.T_w2c0 = None
        self.session = None
        self.cfg = None

    @staticmethod
    def _frame_index_from_path(path: str) -> int:
        return int(Path(path).parent.name)

    @staticmethod
    def _robust_fuse(points: np.ndarray, outlier_m: float) -> np.ndarray:
        """Median + Euclidean outlier rejection, then mean of inliers."""
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 1:
            return points[0]
        med = np.median(points, axis=0)
        dist = np.linalg.norm(points - med[None, :], axis=1)
        keep = dist <= float(outlier_m)
        if np.any(keep):
            return np.mean(points[keep], axis=0)
        return med

    def _depth_color_m(self, session_idx: int):
        if self.session.spatial_aligned:
            return self.session.depth_m(session_idx)

        return align_depth_to_color_numpy(
            depth_raw=self.session.depth_raw(session_idx, copy=False),
            depth_scale=self.session.depth_scale,
            K_depth=self.session.K_depth,
            K_color=self.session.K_color,
            T_depth_to_color=self.session.T_depth_to_color,
            color_hw=(self.session.height, self.session.width),
            stride=1,
        )

    def _lift_object(
        self,
        obj_key: str,
        tracks_all: np.ndarray,
        vis_all: np.ndarray,
        relevant_image_paths,
        step_size: int,
    ):
        T, N, _ = tracks_all.shape

        # Same smoothing idea as HumanEgo, but guard short sequences.
        smooth_window = int(getattr(self.cfg, "smooth_window", 7))
        smooth_poly = int(getattr(self.cfg, "smooth_polyorder", 2))
        if smooth_window % 2 == 0:
            smooth_window += 1
        if T >= smooth_window and smooth_window > smooth_poly:
            for n in range(N):
                for d in range(2):
                    tracks_all[:, n, d] = savgol_filter(
                        tracks_all[:, n, d],
                        smooth_window,
                        smooth_poly,
                    )

        indices = list(range(0, T, max(1, int(step_size))))
        if (T - 1) not in indices:
            indices.append(T - 1)

        local_radius = int(getattr(self.cfg, "realsense_depth_radius", 5))
        min_depth = float(getattr(self.cfg, "realsense_min_depth_m", 0.08))
        max_depth = float(getattr(self.cfg, "realsense_max_depth_m", 4.0))
        outlier_m = float(getattr(self.cfg, "realsense_fusion_outlier_m", 0.08))
        min_obs = int(getattr(self.cfg, "realsense_min_observations", 2))
        vis_thresh = float(getattr(self.cfg, "realsense_visibility_thresh", 0.5))

        per_kpt_world = [[] for _ in range(N)]
        diagnostics = {
            "num_keypoints": int(N),
            "sampled_frames": len(indices),
            "valid_depth_observations_per_keypoint": [0 for _ in range(N)],
        }

        depth_cache = {}

        for t in indices:
            frame_dir = Path(relevant_image_paths[t]).parent
            session_idx = self._frame_index_from_path(relevant_image_paths[t])

            cam_json = frame_dir / "aria_cam_rgb.json"
            with cam_json.open() as f:
                cam_d = json.load(f)

            c2w = np.asarray(cam_d["c2w"], dtype=np.float64)
            K = np.asarray(cam_d["k"], dtype=np.float64)

            if session_idx not in depth_cache:
                depth_cache[session_idx] = self._depth_color_m(session_idx)
            depth_m = depth_cache[session_idx]

            for n in range(N):
                if float(vis_all[t, n]) <= vis_thresh:
                    continue

                uv = np.asarray(tracks_all[t, n], dtype=np.float64).reshape(1, 2)
                z = robust_local_depth(
                    depth_m,
                    uv,
                    radius=local_radius,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )[0]

                if not np.isfinite(z) or z <= 0:
                    continue

                p_c = backproject_pixels(
                    uv,
                    np.asarray([z], dtype=np.float64),
                    K,
                )[0]

                p_w = c2w[:3, :3] @ p_c + c2w[:3, 3]
                if np.all(np.isfinite(p_w)):
                    per_kpt_world[n].append(p_w)
                    diagnostics[
                        "valid_depth_observations_per_keypoint"
                    ][n] += 1

        pts_w = []
        kept_kpt_ids = []

        for n, obs in enumerate(per_kpt_world):
            if len(obs) < min_obs:
                continue
            fused = self._robust_fuse(np.asarray(obs), outlier_m)
            if np.all(np.isfinite(fused)):
                pts_w.append(fused)
                kept_kpt_ids.append(n)

        diagnostics["kept_keypoint_ids"] = kept_kpt_ids
        diagnostics["num_fused_3d_points"] = len(pts_w)

        if len(pts_w) < 3:
            raise ValueError(
                f"║[RealSense RGB-D Error] {obj_key}: only {len(pts_w)} "
                f"valid fused 3D keypoints. Depth observations per keypoint="
                f"{diagnostics['valid_depth_observations_per_keypoint']}. "
                "Check depth/RGB spatial alignment, object mask/keypoints, "
                "or lower realsense_min_observations."
            )

        pts_w = np.asarray(pts_w, dtype=np.float64)
        pts_c0 = (
            self.T_w2c0[:3, :3] @ pts_w.T
            + self.T_w2c0[:3, 3][:, None]
        ).T

        return pts_w, pts_c0, diagnostics

    def _initialize(
        self,
        image_path,
        cfg_path,
        all_image_paths,
        mps_path,
    ):
        self.cfg = load_cfg(cfg_path)
        self.session = RealSenseSession(mps_path)

        tracks_path = os.path.join(
            mps_path, "preprocess", "cotracker_results.json"
        )
        with open(tracks_path) as f:
            cotracker_data = json.load(f)

        relevant_image_paths = list(all_image_paths)
        T_provided = len(relevant_image_paths)
        step_size = int(getattr(self.cfg, "step", 1))

        pose_method_cfg = getattr(self.cfg, "pose_method", "vlm")
        if isinstance(pose_method_cfg, dict) or hasattr(
            pose_method_cfg, "keys"
        ):
            pm_dict = dict(pose_method_cfg)
            print(
                f"║ [RGB-D Triangulator] Pose Estimation Methods: {pm_dict}"
            )
        else:
            pm_dict = None
            print(
                "║ [RGB-D Triangulator] Pose Estimation Method:",
                str(pose_method_cfg).upper(),
            )

        print(
            f"║ [RGB-D Triangulator] Direct depth lifting, step={step_size}"
        )

        obj_keys = sorted(
            k for k in cotracker_data.keys() if k.startswith("obj")
        )
        if not obj_keys:
            raise ValueError("No obj* keys in cotracker_results.json")

        # camera0 exactly matches original CamTriangulator contract.
        cam0_p = relevant_image_paths[0].replace(
            "rgb.png", "aria_cam_rgb.json"
        )
        with open(cam0_p) as f:
            cam0_d = json.load(f)

        self.T_c02w = np.asarray(cam0_d["c2w"], dtype=np.float64)
        self.T_w2c0 = np.linalg.inv(self.T_c02w)

        # Needed only for VLM orientation.
        cam0_raw = cv2.imread(relevant_image_paths[0])
        cam0_rgb = (
            cv2.cvtColor(cam0_raw, cv2.COLOR_BGR2RGB)
            if cam0_raw is not None
            else None
        )

        anchor_key = obj_keys[0]
        anchor_center_cam = None
        global_T_limit = 0

        for obj_key in obj_keys:
            tracks_all = np.asarray(
                cotracker_data[obj_key]["tracks"], dtype=np.float64
            )[:T_provided]
            vis_all = np.asarray(
                cotracker_data[obj_key]["visibility"], dtype=np.float64
            )[:T_provided]

            global_T_limit = max(global_T_limit, tracks_all.shape[0])

            print(
                f"║ [RGB-D Triangulator] Lifting {obj_key}: "
                f"{tracks_all.shape[1]} keypoints"
            )

            pts_w, pts_c0, diagnostics = self._lift_object(
                obj_key,
                tracks_all.copy(),
                vis_all,
                relevant_image_paths,
                step_size,
            )

            t_c0 = pts_c0.mean(axis=0)
            is_anchor = obj_key == anchor_key

            if pm_dict is not None:
                curr_method = str(
                    pm_dict.get(
                        obj_key, pm_dict.get("default", "vlm")
                    )
                ).lower()
            else:
                curr_method = str(pose_method_cfg).lower()

            if curr_method == "vlm":
                tracks0 = tracks_all[0]
                vis0 = vis_all[0] > 0.5
                valid = tracks0[vis0]

                if cam0_rgb is None or len(valid) == 0:
                    crop = (
                        Image.fromarray(cam0_rgb)
                        if cam0_rgb is not None
                        else None
                    )
                else:
                    pad = 40
                    x1, y1 = np.min(valid, axis=0) - pad
                    x2, y2 = np.max(valid, axis=0) + pad
                    h, w = cam0_rgb.shape[:2]
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    crop = Image.fromarray(cam0_rgb[y1:y2, x1:x2])

                T_o2c0, info = estimate_frame_vlm(
                    image=crop,
                    t_cam=t_c0,
                    is_anchor=is_anchor,
                    anchor_center_cam=anchor_center_cam,
                    do_rm_bkg=True,
                )

            elif curr_method == "pca1":
                T_o2c0, info = estimate_frame_pca1(
                    pts_cam=pts_c0,
                    is_anchor=is_anchor,
                    anchor_center_cam=anchor_center_cam,
                )

            elif curr_method == "pca2":
                T_o2c0, info = estimate_frame_pca2(
                    pts_cam=pts_c0,
                    is_anchor=is_anchor,
                    anchor_center_cam=anchor_center_cam,
                )

            else:
                raise ValueError(
                    f"Unknown pose method {curr_method} for {obj_key}"
                )

            if is_anchor:
                anchor_center_cam = T_o2c0[:3, 3].copy()

            # Preserve upstream info but add depth diagnostics.
            if not isinstance(info, dict):
                info = {"upstream_info": info}
            info["realsense_rgbd"] = diagnostics
            info["reconstruction_method"] = (
                "multi_frame_metric_depth_fusion"
            )

            self.objects_3d[obj_key] = {
                "points_3d_world": pts_w,
                "points_3d_cam0": pts_c0,
                "T_o2c0": np.asarray(T_o2c0, dtype=np.float64),
                "T_o2w_vis": self.T_c02w
                @ np.asarray(T_o2c0, dtype=np.float64),
                "frame_info": info,
            }

        self.T_limit = global_T_limit

        result = {
            "cam0_c2w": self.T_c02w.tolist(),
            "reconstruction_method": "realsense_rgbd_depth_fusion",
            "objects": {},
        }

        for k, v in self.objects_3d.items():
            result["objects"][k] = {
                "points_3d_world": v["points_3d_world"].tolist(),
                "points_3d_cam0": v["points_3d_cam0"].tolist(),
                "object_to_cam0_matrix": v["T_o2c0"].tolist(),
                "info": v["frame_info"],
            }

        out_path = os.path.join(
            mps_path, "preprocess", "camtriangulator_results.json"
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

        print(
            "║ [RGB-D Triangulator] Saved compatible results:",
            out_path,
        )

        self.initialized = True

    def _draw(self, image_path, frame_idx):
        img = cv2.imread(image_path)
        if img is None:
            return img

        cam_p = image_path.replace(
            "rgb.png", "aria_cam_rgb.json"
        )
        with open(cam_p) as f:
            cam_d = json.load(f)

        w2c = np.linalg.inv(
            np.asarray(cam_d["c2w"], dtype=np.float64)
        )
        K = np.asarray(cam_d["k"], dtype=np.float64)

        def project(p_w):
            p_c = w2c[:3, :3] @ p_w + w2c[:3, 3]
            if p_c[2] <= 1e-4:
                return None
            uvh = K @ p_c
            u = int(uvh[0] / uvh[2])
            v = int(uvh[1] / uvh[2])
            return (u, v)

        for obj_key, d in self.objects_3d.items():
            for p in d["points_3d_world"]:
                uv = project(p)
                if uv is not None:
                    cv2.circle(
                        img, uv, 4, (0, 255, 0), -1, cv2.LINE_AA
                    )

            T_o2w = d["T_o2w_vis"]
            o = T_o2w[:3, 3]
            uv0 = project(o)
            if uv0 is not None:
                cv2.circle(img, uv0, 5, (0, 215, 255), -1)
                cv2.putText(
                    img,
                    obj_key,
                    (uv0[0] + 8, uv0[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                colors = [
                    (0, 0, 255),
                    (0, 255, 0),
                    (255, 0, 0),
                ]
                for axis in range(3):
                    end = o + T_o2w[:3, axis] * 0.12
                    uv1 = project(end)
                    if uv1 is not None:
                        cv2.arrowedLine(
                            img,
                            uv0,
                            uv1,
                            colors[axis],
                            2,
                            tipLength=0.2,
                        )

        return img

    def process(
        self,
        image_path,
        cfg_path,
        frame_idx,
        all_image_paths,
        mps_path,
    ):
        if not self.initialized:
            self._initialize(
                image_path,
                cfg_path,
                all_image_paths,
                mps_path,
            )

        img = self._draw(image_path, frame_idx)

        if frame_idx == min(
            self.T_limit - 1, len(all_image_paths) - 1
        ):
            cv2.imwrite(
                os.path.join(
                    mps_path,
                    "preprocess",
                    "camtriangulator_vis.png",
                ),
                img,
            )

        return img


_MANAGER = RealSenseCamTriangulatorManager()


def reset_realsense_camtriangulator():
    global _MANAGER
    _MANAGER = RealSenseCamTriangulatorManager()


def run_realsense_camtriangulator(
    image_path,
    cfg_path,
    frame_idx,
    all_image_paths=None,
    mps_path=None,
):
    return _MANAGER.process(
        image_path,
        cfg_path,
        frame_idx,
        all_image_paths,
        mps_path,
    )
