# -*- coding: utf-8 -*-
"""
Source-compatible phase segmentation without importing Aria/MPS.
"""
from __future__ import annotations

import numpy as np

from utils.utils_io import load_cfg
from preprocess.AriaPhasesTypes import AriaPhases, AriaPhasesFrame
from preprocess.AriaPhasesOps import AriaPhasesOps


class RealSensePhasesGenerator:
    def __init__(
        self,
        mps_path: str,
        cfg_path: str,
        aria_cam,
        aria_slam,
        aria_hands=None,
        whole_clip: bool = False,
    ):
        self.mps_path = mps_path
        self.cfg = load_cfg(cfg_path)
        self.aria_cam = aria_cam
        self.aria_slam = aria_slam
        self.aria_hands = aria_hands
        self.whole_clip = bool(whole_clip)

    def get_aria_phases(self):
        n = len(self.aria_slam.frames)
        tss = [f.ts for f in self.aria_slam.frames]
        v = np.array([f.v for f in self.aria_slam.frames], dtype=float)
        w = np.array([f.w for f in self.aria_slam.frames], dtype=float)
        yaw = np.array(
            [f.yaw_unwrapped_deg for f in self.aria_slam.frames],
            dtype=float,
        )

        if self.whole_clip:
            stop = np.ones(n, dtype=np.uint8)
            mode = np.zeros(n, dtype=np.int32)
            n_finished = min(
                n,
                int(getattr(self.cfg, "finished_frames", 60)),
            )
            if n_finished > 0:
                mode[-n_finished:] = 4
            mode_stats = {
                "stop_frames": int(np.sum(mode == 0)),
                "forward_frames": 0,
                "rotate_frames": 0,
                "transition_frames": 0,
                "finished_frames": int(np.sum(mode == 4)),
            }
        else:
            stop = AriaPhasesOps.compute_stop_mask_from_vw(
                v,
                w,
                self.cfg.v_stop_thresh,
                self.cfg.w_stop_thresh,
                self.cfg.stop_hold_frames,
                self.cfg.stop_debounce_frames,
            )
            stop = AriaPhasesOps.close_binary_mask(
                stop,
                kernel_size=self.cfg.stop_debounce_frames
            )

            # OpenCV morphology may return (N, 1); HumanEgo phase ops expect (N,)
            stop = np.asarray(stop).reshape(-1)

            stop = AriaPhasesOps.postprocess_stop_mask(
                stop,
                yaw,
                self.cfg.stop_min_on_frames,
                self.cfg.stop_yaw_veto_deg,
            )
            stop = AriaPhasesOps.apply_stop_offset(
                stop, self.cfg.stop_offset_frames
            )

            mode, _, mode_stats = AriaPhasesOps.compute_mode_from_stop_vw(
                stop,
                v,
                w,
                yaw,
                self.cfg.w_rot_thresh,
                self.cfg.v_rot_max,
                self.cfg,
            )

            if self.aria_hands is not None:
                mode, mode_stats = (
                    AriaPhasesOps.refine_manip_phases_with_hand_vel(
                        mode,
                        self.aria_hands,
                        getattr(self.cfg, "manip_clean_vel_thresh", 0.15),
                        getattr(self.cfg, "manip_clean_wait_frames", 5),
                        getattr(self.cfg, "manip_clean_manual_offset", 15),
                    )
                )

            mode = AriaPhasesOps.inject_finished_phase(
                mode,
                getattr(self.cfg, "finished_frames", 60),
            )

        out = AriaPhases(mps_path=self.mps_path)
        names = {
            0: "STOP",
            1: "FORWARD",
            2: "ROTATE",
            3: "TRANSITION",
            4: "FINISHED",
        }

        for i in range(n):
            out.frames.append(
                AriaPhasesFrame(
                    idx=self.aria_slam.frames[i].idx,
                    ts=tss[i],
                    mode=int(mode[i]),
                    stop=int(np.ravel(stop)[i]),
                    mode_str=names.get(int(mode[i]), "UNKNOWN"),
                    v=float(v[i]),
                    w=float(w[i]),
                    yaw_u_deg=float(yaw[i]),
                )
            )

        duration_s = (tss[-1] - tss[0]) * 1e-9 if n > 1 else 0.0

        # HumanEgo Preprocess.preprocess_indices() consumes stage_window_check.
        out.summary = {
            "total_frames": n,
            "fps_median": self.aria_cam.fps,
            "duration_s": duration_s,
            "segment_summary_opt": (
                AriaPhasesOps.segment_summary_from_masks(
                    stop, v, w, self.cfg.v_walk_thresh
                )
                if not self.whole_clip else {}
            ),
            "stage_window_check": AriaPhasesOps.stage_window_check(
                mode, [4, 3, 2, 1, 0], True, False
            ),
            "mode_stats_opt": mode_stats,
            "hyperparams": {
                "realsense_whole_clip": self.whole_clip,
            },
        }
        return out
