# -*- coding: utf-8 -*-
"""
AriaSlam-compatible kinematic serialization from RealSense c2w.
No projectaria_tools import.
"""
from __future__ import annotations

import math
import os
import numpy as np

from preprocess.AriaSlamTypes import AriaSlam, AriaSlamFrame


class RealSenseSlamGenerator:
    def __init__(self, mps_path: str, aria_cam):
        self.mps_path = mps_path
        self.aria_cam = aria_cam

    @staticmethod
    def rotmat_to_rpy_zyx(Rm):
        sy = math.sqrt(Rm[0,0]**2 + Rm[1,0]**2)
        if sy > 1e-6:
            roll = math.atan2(Rm[2,1], Rm[2,2])
            pitch = math.atan2(-Rm[2,0], sy)
            yaw = math.atan2(Rm[1,0], Rm[0,0])
        else:
            roll = math.atan2(-Rm[1,2], Rm[1,1])
            pitch = math.atan2(-Rm[2,0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    def get_aria_slam(self):
        out = AriaSlam(mps_path=self.mps_path)
        yaws = []

        for cam in self.aria_cam.cam:
            r, p, y = self.rotmat_to_rpy_zyx(cam.c2w[:3, :3])
            f = AriaSlamFrame(
                idx=cam.idx,
                ts=cam.ts,
                c2w=cam.c2w,
                k=cam.k,
            )
            f.t_world = cam.c2w[:3, 3].astype(np.float64)
            f.rpy_deg = np.degrees([r, p, y])
            yaws.append(y)
            out.frames.append(f)

        if not out.frames:
            return out

        yaws_u = np.unwrap(np.asarray(yaws, dtype=np.float64))
        t0 = out.frames[0].t_world.copy()
        rpy0 = out.frames[0].rpy_deg.copy()

        for i, f in enumerate(out.frames):
            f.yaw_unwrapped_deg = math.degrees(yaws_u[i])
            f.delta_t = f.t_world - t0

            d = f.rpy_deg - rpy0
            d[2] = math.degrees(
                ((math.radians(d[2]) + math.pi) % (2 * math.pi)) - math.pi
            )
            f.delta_rpy_deg = d

            if i > 0:
                prev = out.frames[i - 1]
                dt = max((f.ts - prev.ts) * 1e-9, 1e-6)
                f.v = float(np.linalg.norm(f.t_world - prev.t_world) / dt)
                f.w = float(abs(yaws_u[i] - yaws_u[i-1]) / dt)

        return out
