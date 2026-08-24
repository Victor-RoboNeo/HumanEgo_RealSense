# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from preprocess.AriaCamTypes import AriaCam, AriaCamData


class RealSenseCamGenerator:
    """
    Emits the SAME AriaCam / aria_cam_rgb.json schema consumed by HumanEgo.
    """

    def __init__(self, session, c2w_list, save_depth_png=True):
        if len(session) != len(c2w_list):
            raise ValueError("session and pose trajectory length mismatch")
        self.session = session
        self.c2w_list = c2w_list
        self.save_depth_png = bool(save_depth_png)

    def get_aria_cam(self) -> AriaCam:
        s = self.session
        cam = AriaCam()
        cam.mps_path = str(s.root)
        cam.fps = int(round(s.fps))
        cam.first_ts = s.timestamp_ns(0)
        cam.fov = s.fov_y_deg
        cam.h = s.height
        cam.w = s.width
        cam.k = s.K_color.copy()
        cam.d = s.color_dist.copy()
        cam.c2d = s.T_color_to_depth.copy()

        for i in range(len(s)):
            c2w = np.asarray(self.c2w_list[i], dtype=np.float64)
            # Keep HumanEgo relation c2w = d2w @ c2d.
            d2w = c2w @ s.T_depth_to_color

            frame = AriaCamData(
                idx=i,
                ts=s.timestamp_ns(i),
                img=s.rgb(i),
                fov=s.fov_y_deg,
                h=s.height,
                w=s.width,
                k=s.K_color.copy(),
                d=s.color_dist.copy(),
                c2w=c2w,
                c2d=s.T_color_to_depth.copy(),
                d2w=d2w,
            )
            cam.tss.append(frame.ts)
            cam.cam.append(frame)

        return cam

    def save(self, aria_cam: AriaCam):
        """
        Use upstream serializer for exact compatibility, then append RealSense-only
        depth metadata without changing any required HumanEgo fields.
        """
        aria_cam.save_aria_cam_json(label="rgb")

        if not self.save_depth_png:
            return

        all_data = Path(self.session.root) / "preprocess" / "all_data"

        for i in range(len(self.session)):
            frame_dir = all_data / f"{i:05d}"
            depth_path = frame_dir / "depth.png"
            cv2.imwrite(
                str(depth_path),
                self.session.depth_raw(i, copy=False),
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )

            p = frame_dir / "aria_cam_rgb.json"
            with p.open() as f:
                data = json.load(f)

            data["sensor_backend"] = "realsense"
            data["depth_path"] = str(depth_path)
            data["depth_scale_m_per_unit"] = self.session.depth_scale
            data["depth_spatially_aligned_to_rgb"] = self.session.spatial_aligned
            data["source_capture_index"] = int(
                self.session.records[i].capture_index
            )
            data["source_host_monotonic_ns"] = int(
                self.session.records[i].host_ns
            )

            with p.open("w") as f:
                json.dump(data, f, indent=2)
