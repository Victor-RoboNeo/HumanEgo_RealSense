# -*- coding: utf-8 -*-
"""
HumanEgo RealSense entry point.

Frontend:
    RealSense raw session
      -> RealSenseCam (aria_cam_rgb.json)
      -> RealSenseHands (aria_hands.json)
      -> RealSenseSlam (aria_slam.json)
      -> RealSensePhases (aria_phases.json + aria_phases_results.json)

Downstream is kept source-compatible:
    DINOSAM -> KptsSelector -> CoTracker -> CamTriangulator
    -> LaMa -> VisualKpts -> DatasetGen
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from utils.utils_io import load_cfg_dynamic_task
from preprocess.RealSenseSession import RealSenseSession
from preprocess.RealSensePose import (
    DepthICPPoseProvider,
    ExternalTrajectoryPoseProvider,
)
from preprocess.RealSenseCam import RealSenseCamGenerator
from preprocess.RealSenseWiLoRHands import RealSenseWiLoRHandsGenerator
from preprocess.RealSenseSlam import RealSenseSlamGenerator
from preprocess.RealSensePhases import RealSensePhasesGenerator


class RealSensePreprocess:
    def __init__(
        self,
        session_path,
        cfg_path,
        task,
        pose_backend="depth_icp",
        trajectory=None,
        frontend_cfg=None,
        whole_clip=False,
        frontend_only=False,
        export_video=False,
        export_gif=False,
    ):
        self.mps_path = str(Path(session_path).resolve())
        self.cfg_path = cfg_path
        self.task = task
        self.pose_backend = pose_backend
        self.trajectory = trajectory
        self.whole_clip = whole_clip
        self.frontend_only = frontend_only
        self.export_video = export_video
        self.export_gif = export_gif

        self.cfg = load_cfg_dynamic_task(
            self.cfg_path, self.mps_path, self.task
        )
        self.session = RealSenseSession(self.mps_path)

        self.num_total_frames = len(self.session)
        self.start_idx = 0
        self.end_idx = len(self.session) - 1

        self.frontend_cfg = self._load_frontend_cfg(frontend_cfg)

    @staticmethod
    def _load_frontend_cfg(path):
        import yaml
        default = {
            "icp": {
                "voxel_size": 0.025,
                "max_correspondence": 0.08,
                "stride": 3,
                "min_depth": 0.20,
                "max_depth": 4.0,
                "max_iteration": 25,
                "min_fitness": 0.12,
            },
            "hands": {
                "local_depth_radius": 5,
                "min_valid_depth_landmarks": 5,
                "model_path": None,
            },
        }
        if path is None:
            return default
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for key, val in user.items():
            if isinstance(val, dict) and key in default:
                default[key].update(val)
            else:
                default[key] = val
        return default

    def _pose_provider(self):
        if self.pose_backend == "external":
            if not self.trajectory:
                raise ValueError("--trajectory is required for external pose")
            return ExternalTrajectoryPoseProvider(self.trajectory)

        if self.pose_backend == "depth_icp":
            return DepthICPPoseProvider(**self.frontend_cfg["icp"])

        raise ValueError(f"Unknown pose backend: {self.pose_backend}")

    def preprocess_realsense_frontend(self):
        print("║ [RealSense] Computing camera trajectory...")
        c2w = self._pose_provider().compute(self.session)

        print("║ [RealSense] Exporting HumanEgo camera schema...")
        cam_gen = RealSenseCamGenerator(
            self.session,
            c2w,
            save_depth_png=True,
        )
        self.aria_cam_rgb = cam_gen.get_aria_cam()
        cam_gen.save(self.aria_cam_rgb)

        print("║ [RealSense] RGB-D hand reconstruction...")
        hand_cfg = self.frontend_cfg["hands"]
        hands_gen = RealSenseWiLoRHandsGenerator(
            mps_path=self.mps_path,
            cfg_path=self.cfg.AriaHands_path,
            aria_cam=self.aria_cam_rgb,
            session=self.session,
            model_path=hand_cfg.get("model_path"),
            local_depth_radius=hand_cfg.get("local_depth_radius", 5),
            min_valid_depth_landmarks=hand_cfg.get(
                "min_valid_depth_landmarks", 5
            ),
        )
        self.aria_hands = hands_gen.get_aria_hands()
        self.aria_hands.save_hands_json(
            filename="aria_hands.json"
        )

        print("║ [RealSense] Deriving HumanEgo SLAM kinematics...")
        slam_gen = RealSenseSlamGenerator(
            self.mps_path, self.aria_cam_rgb
        )
        self.aria_slam = slam_gen.get_aria_slam()
        self.aria_slam.save_aria_slam_json()

        print("║ [RealSense] HumanEgo phase segmentation...")
        phase_gen = RealSensePhasesGenerator(
            self.mps_path,
            self.cfg.AriaPhases_path,
            self.aria_cam_rgb,
            self.aria_slam,
            self.aria_hands,
            whole_clip=self.whole_clip,
        )
        self.aria_phases = phase_gen.get_aria_phases()
        self.aria_phases.save_aria_phases_json()

        marker = Path(self.mps_path) / "preprocess" / "realsense_frontend.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("w") as f:
            json.dump(
                {
                    "backend": "realsense",
                    "pose_backend": self.pose_backend,
                    "trajectory": self.trajectory,
                    "spatial_aligned_depth": self.session.spatial_aligned,
                    "num_frames": len(self.session),
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # The following methods intentionally mirror Preprocess.py downstream.
    # ------------------------------------------------------------------

    def preprocess_indices(self):
        phases_json_path = os.path.join(
            self.mps_path, "preprocess", "aria_phases_results.json"
        )
        with open(phases_json_path) as f:
            phases_data = json.load(f)

        windows_dict = (
            phases_data.get("stage_window_check", {}).get("windows", {})
        )

        def get_image_list_from_keys(keys):
            allowed_idx = []
            for key in keys:
                for win in windows_dict.get(key, []):
                    if not isinstance(win, (list, tuple)) or len(win) != 2:
                        continue
                    s, e = int(win[0]), int(win[1])
                    a, b = (s, e) if s <= e else (e, s)
                    allowed_idx.extend(range(a, b + 1))
            res = []
            for i in sorted(set(allowed_idx)):
                p = os.path.join(
                    self.mps_path,
                    "preprocess",
                    "all_data",
                    f"{i:05d}",
                    "rgb.png",
                )
                if os.path.exists(p):
                    res.append(p)
            return res

        self.manip_image_list = get_image_list_from_keys(["0", "4"])
        self.nav_image_list = get_image_list_from_keys(["1", "2"])
        self.transition_image_list = get_image_list_from_keys(["3"])
        self.raw_manip_image_list = get_image_list_from_keys(["0", "3", "4"])

        self.all_image_list = []
        for i in range(self.num_total_frames):
            p = os.path.join(
                self.mps_path,
                "preprocess",
                "all_data",
                f"{i:05d}",
                "rgb.png",
            )
            if os.path.exists(p):
                self.all_image_list.append(p)

        if not self.raw_manip_image_list:
            # A conservative fallback for a failed phase heuristic.
            print(
                "║ [RealSense] No manipulation window found; "
                "falling back to the whole clip."
            )
            self.manip_image_list = list(self.all_image_list)
            self.raw_manip_image_list = list(self.all_image_list)

        first_manip = self.raw_manip_image_list[0]
        try:
            split_idx = self.all_image_list.index(first_manip)
        except ValueError:
            split_idx = 0

        pre_frames_count = min(
            split_idx,
            self.cfg.CoTracker.object_centric_max_frames,
        )
        self.object_centric_image_list = self.all_image_list[
            split_idx - pre_frames_count : split_idx
        ]

        if (
            len(self.object_centric_image_list)
            < self.cfg.CoTracker.object_centric_min_frames
        ):
            needed = (
                self.cfg.CoTracker.object_centric_min_frames
                - len(self.object_centric_image_list)
            )
            self.object_centric_image_list += self.raw_manip_image_list[:needed]

        ref_idx = (
            self.cfg.CoTracker.ref_idx
            if abs(self.cfg.CoTracker.ref_idx)
            < len(self.object_centric_image_list)
            else -1
        )
        self.object_ref_img_path = self.object_centric_image_list[ref_idx]

        print(
            f"║ [RealSense] Frames: manip={len(self.manip_image_list)}, "
            f"nav={len(self.nav_image_list)}, "
            f"transition={len(self.transition_image_list)}, "
            f"object-centric={len(self.object_centric_image_list)}"
        )

    def preprocess_downstream(self):
        # Lazy imports keep --frontend-only light.
        from preprocess.DINOSAMOps import run_dinosam, print_dinosam_stats
        from preprocess.SAM2VideoObjectPropagator import run_sam2_video_object_propagation
        from preprocess.sam31.SAM31ObjectMask import run_sam31_object_masks
        from preprocess.KptsSelector import run_kptsselector
        from preprocess.CoTrackerOffline import (
            run_cotracker_offline,
            print_cotracker_offline_stats,
        )
        from preprocess.RealSenseCamTriangulator import run_realsense_camtriangulator
        from preprocess.Lama import run_lama, print_lama_stats
        from preprocess.VisualKpts import run_visualkpts
        from preprocess.DatasetGen import run_datasetgen

        dinosam_list = self.object_centric_image_list + self.raw_manip_image_list
        # print(f"║ [DINOSAM] {len(dinosam_list)} frames")
        # for img_path in tqdm(dinosam_list, desc="DINO+SAM2"):
        #     if os.path.exists(img_path):
        #         run_dinosam(
        #             cfg_path=self.cfg.DINOSAM_path,
        #             image_path=img_path,
        #         )
        # print_dinosam_stats()

        # # Repair occlusion-sensitive object masks with SAM2 video memory
        # # BEFORE KptsSelector / CoTracker.
        # run_sam2_video_object_propagation(
        #     image_paths=dinosam_list,
        #     cfg_path=self.cfg.DINOSAM_path,
        #     mps_path=self.mps_path,
        #     min_area_ratio=0.001,
        #     max_area_ratio=0.60,
        #     backup_dino_masks=True,
        #     offload_video_to_cpu=True,
        #     offload_state_to_cpu=False,
        #     vos_optimized=False,
        # )

        # All masks (obj* and arm) come from SAM3.1; each prompt is tracked independently.
        run_sam31_object_masks(
            image_paths=dinosam_list,
            cfg_path=self.cfg.DINOSAM_path,
            mps_path=self.mps_path,
            reference_image_path=self.object_ref_img_path,
        )

        ref_frame_dir = os.path.dirname(self.object_ref_img_path)
        obj_keys = [
            k for k in self.cfg.DINOSAM.dinosam_prompt.keys()
            if k.startswith("obj")
        ]
        all_kpts_data = {}
        for obj_key in obj_keys:
            src = os.path.join(ref_frame_dir, f"mask_{obj_key}.png")
            dst = os.path.join(
                self.mps_path,
                "preprocess",
                f"dinosam_mask_{obj_key}.png",
            )
            if os.path.exists(src):
                shutil.copy2(src, dst)
                kpts = run_kptsselector(
                    cfg_path=self.cfg.KptsSelector_path,
                    mask_path=dst,
                    save_img_path=os.path.join(
                        self.mps_path,
                        "preprocess",
                        f"kptsselector_vis_{obj_key}.png",
                    ),
                    rgb_path=self.object_ref_img_path,
                )
                if kpts:
                    all_kpts_data[obj_key] = kpts

        with open(
            os.path.join(
                self.mps_path,
                "preprocess",
                "kptsselector_results.json",
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "method": "AUTO_MASK_PCA_EDGE",
                    "objects": all_kpts_data,
                },
                f,
                indent=4,
            )

        cotracker_list = (
            self.object_centric_image_list + self.raw_manip_image_list
        )
        print(f"║ [CoTracker] {len(cotracker_list)} frames")
        for i, img_path in enumerate(
            tqdm(cotracker_list, desc="CoTracker")
        ):
            if os.path.exists(img_path):
                run_cotracker_offline(
                    image_path=img_path,
                    cfg_path=self.cfg.CoTracker_path,
                    frame_idx=i,
                    all_image_paths=cotracker_list,
                    mps_path=self.mps_path,
                )
        print_cotracker_offline_stats()

        tri_list = self.object_centric_image_list
        print(f"║ [CamTriangulator] {len(tri_list)} frames")
        for i, img_path in enumerate(
            tqdm(tri_list, desc="CamTriangulator")
        ):
            if os.path.exists(img_path):
                run_realsense_camtriangulator(
                    image_path=img_path,
                    cfg_path=self.cfg.CamTriangulator_path,
                    frame_idx=i,
                    all_image_paths=tri_list,
                    mps_path=self.mps_path,
                )

        lama_list = self.object_centric_image_list + self.raw_manip_image_list
        for img_path in tqdm(lama_list, desc="LaMa"):
            if os.path.exists(img_path):
                run_lama(
                    image_path=img_path,
                    cfg_path=self.cfg.Lama_path,
                )
        print_lama_stats()

        visual_list = (
            self.object_centric_image_list + self.raw_manip_image_list
        )
        for i, img_path in enumerate(
            tqdm(visual_list, desc="VisualKpts")
        ):
            if os.path.exists(img_path):
                run_visualkpts(
                    image_path=img_path,
                    cfg_path=self.cfg.VisualKpts_path,
                    frame_idx=i,
                    all_image_paths=visual_list,
                    mps_path=self.mps_path,
                )

        if self.manip_image_list:
            run_datasetgen(
                image_list=self.manip_image_list,
                mps_path=self.mps_path,
                cfg_path=self.cfg.DatasetGen_path,
            )

    def run(self):
        self.preprocess_realsense_frontend()

        if self.frontend_only:
            print("║ [RealSense] frontend-only complete.")
            return

        self.preprocess_indices()
        self.preprocess_downstream()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--cfg_path",
        default="./cfg/preprocess/base/Preprocess.yaml",
    )
    parser.add_argument("--task", default=None)

    parser.add_argument(
        "--pose-backend",
        choices=["depth_icp", "external"],
        default="depth_icp",
    )
    parser.add_argument("--trajectory", default=None)
    parser.add_argument(
        "--frontend-cfg",
        default="./cfg/preprocess/base/RealSense.yaml",
    )
    parser.add_argument(
        "--whole-clip",
        action="store_true",
        help="Treat the whole recording as manipulation (debug/baseline).",
    )
    parser.add_argument("--frontend-only", action="store_true")

    args = parser.parse_args()

    engine = RealSensePreprocess(
        session_path=args.session,
        cfg_path=args.cfg_path,
        task=args.task,
        pose_backend=args.pose_backend,
        trajectory=args.trajectory,
        frontend_cfg=args.frontend_cfg,
        whole_clip=args.whole_clip,
        frontend_only=args.frontend_only,
    )
    engine.run()


if __name__ == "__main__":
    main()
