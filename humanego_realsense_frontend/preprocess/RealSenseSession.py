# -*- coding: utf-8 -*-
"""
Offline reader for the high-throughput Linux D455 recorder.

Expected session files:
    metadata.json
    calibration.json
    synchronized.csv
    rgb.bgr8
    depth.z16

This module intentionally has no Project Aria dependency.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class RealSenseFrameRecord:
    idx: int
    saved_index: int
    capture_index: int
    host_ns: int
    color_ms: float
    depth_ms: float
    reference_ms: float
    color_domain: str
    depth_domain: str


def _T_from_extrinsics(ex: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(ex["rotation"], dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(ex["translation"], dtype=np.float64)
    return T


def _K(intr: dict) -> np.ndarray:
    return np.array([
        [float(intr["fx"]), 0.0, float(intr["ppx"])],
        [0.0, float(intr["fy"]), float(intr["ppy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


class RealSenseSession:
    """
    Random-access interface over the recorder output.

    HumanEgo timestamps are nanoseconds.  We map the RGB-D device reference
    timestamp (ms) to integer ns.  If the session was recorded with
    --spatial-align-depth, the depth raster already uses the color image grid.
    """

    def __init__(self, session_dir: str):
        self.root = Path(session_dir).expanduser().resolve()

        with (self.root / "metadata.json").open() as f:
            self.metadata = json.load(f)
        with (self.root / "calibration.json").open() as f:
            self.calibration = json.load(f)

        self.width = int(self.metadata["width"])
        self.height = int(self.metadata["height"])
        self.fps = float(self.metadata.get("video_fps", 30))
        self.depth_scale = float(
            self.metadata.get(
                "depth_scale_m_per_unit",
                self.calibration["depth_scale_m_per_unit"],
            )
        )

        self.spatial_aligned = bool(
            self.metadata.get("spatial_align_depth", False)
        )
        self.storage = self.metadata.get("storage", "raw")
        if self.storage != "raw":
            raise ValueError(
                "RealSenseSession currently expects --storage raw. "
                "Re-record with the high-FPS recorder or convert image mode."
            )

        self.K_color = _K(self.calibration["color_intrinsics"])
        self.K_depth = _K(self.calibration["depth_intrinsics"])
        self.color_dist = np.asarray(
            self.calibration["color_intrinsics"].get("coeffs", [0,0,0,0,0]),
            dtype=np.float64,
        )
        self.depth_dist = np.asarray(
            self.calibration["depth_intrinsics"].get("coeffs", [0,0,0,0,0]),
            dtype=np.float64,
        )

        self.T_depth_to_color = _T_from_extrinsics(
            self.calibration["depth_to_color_extrinsics"]
        )
        self.T_color_to_depth = np.linalg.inv(self.T_depth_to_color)

        self.rgb_frame_bytes = self.width * self.height * 3
        self.depth_frame_bytes = self.width * self.height * 2

        self.rgb_path = self.root / "rgb.bgr8"
        self.depth_path = self.root / "depth.z16"

        if not self.rgb_path.is_file() or not self.depth_path.is_file():
            raise FileNotFoundError(
                f"Missing rgb.bgr8/depth.z16 under {self.root}"
            )

        self.records: List[RealSenseFrameRecord] = []
        with (self.root / "synchronized.csv").open(newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                self.records.append(
                    RealSenseFrameRecord(
                        idx=idx,
                        saved_index=int(row["saved_index"]),
                        capture_index=int(row.get("capture_index", idx + 1)),
                        host_ns=int(row["host_monotonic_ns"]),
                        color_ms=float(row["color_timestamp_ms"]),
                        depth_ms=float(row["depth_timestamp_ms"]),
                        reference_ms=float(row["rgbd_reference_timestamp_ms"]),
                        color_domain=row.get("color_timestamp_domain", "unknown"),
                        depth_domain=row.get("depth_timestamp_domain", "unknown"),
                    )
                )

        if not self.records:
            raise RuntimeError("synchronized.csv contains no RGB-D frames")

        self.timestamps_ns = [
            int(round(r.reference_ms * 1e6)) for r in self.records
        ]

        # Memmaps avoid per-frame open/seek and keep preprocessing fast.
        n_rgb = self.rgb_path.stat().st_size // self.rgb_frame_bytes
        n_depth = self.depth_path.stat().st_size // self.depth_frame_bytes
        n = min(len(self.records), n_rgb, n_depth)
        if n < len(self.records):
            self.records = self.records[:n]
            self.timestamps_ns = self.timestamps_ns[:n]

        self._rgb = np.memmap(
            self.rgb_path,
            mode="r",
            dtype=np.uint8,
            shape=(n, self.height, self.width, 3),
        )
        self._depth = np.memmap(
            self.depth_path,
            mode="r",
            dtype=np.uint16,
            shape=(n, self.height, self.width),
        )

    def __len__(self):
        return len(self.records)

    def rgb(self, idx: int, copy: bool = True) -> np.ndarray:
        x = self._rgb[idx]
        return np.array(x, copy=True) if copy else x

    def depth_raw(self, idx: int, copy: bool = True) -> np.ndarray:
        x = self._depth[idx]
        return np.array(x, copy=True) if copy else x

    def depth_m(self, idx: int) -> np.ndarray:
        return self.depth_raw(idx, copy=False).astype(np.float32) * self.depth_scale

    def timestamp_ns(self, idx: int) -> int:
        return self.timestamps_ns[idx]

    @property
    def fov_y_deg(self) -> float:
        fy = self.K_color[1, 1]
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * fy))))

    def write_depth_png(self, idx: int, path: str, compression: int = 1):
        cv2.imwrite(
            str(path),
            self.depth_raw(idx, copy=False),
            [cv2.IMWRITE_PNG_COMPRESSION, compression],
        )
