# -*- coding: utf-8 -*-
"""
Pose backends replacing MPS closed-loop trajectory.

- external: preferred for serious experiments; accepts TUM-like trajectory CSV.
- depth_icp: fully local baseline using Open3D point-to-plane ICP.

The output is CAMERA-to-WORLD c2w for every RGB frame.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from preprocess.RealSenseGeometry import backproject_pixels


class PoseProvider:
    def compute(self, session) -> List[np.ndarray]:
        raise NotImplementedError


class ExternalTrajectoryPoseProvider(PoseProvider):
    """
    CSV columns:
        timestamp_ns, tx,ty,tz, qx,qy,qz,qw
    or:
        timestamp_ms, tx,ty,tz, qx,qy,qz,qw

    Transform convention: each row is COLOR-CAMERA -> WORLD.
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

    def compute(self, session):
        ts = []
        trans = []
        quats = []

        with self.csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if "timestamp_ns" in row:
                    t_ns = int(float(row["timestamp_ns"]))
                elif "timestamp_ms" in row:
                    t_ns = int(round(float(row["timestamp_ms"]) * 1e6))
                else:
                    raise KeyError("trajectory needs timestamp_ns or timestamp_ms")

                ts.append(t_ns)
                trans.append([float(row[k]) for k in ("tx", "ty", "tz")])
                quats.append([float(row[k]) for k in ("qx", "qy", "qz", "qw")])

        ts = np.asarray(ts, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        rotations = Rotation.from_quat(np.asarray(quats, dtype=np.float64))

        order = np.argsort(ts)
        ts = ts[order]
        trans = trans[order]
        rotations = rotations[order]

        qts = np.asarray(session.timestamps_ns, dtype=np.float64)
        qts_clip = np.clip(qts, ts[0], ts[-1])

        xyz = np.column_stack([
            np.interp(qts_clip, ts, trans[:, d]) for d in range(3)
        ])
        slerp = Slerp(ts, rotations)
        Rq = slerp(qts_clip).as_matrix()

        out = []
        for Rm, t in zip(Rq, xyz):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rm
            T[:3, 3] = t
            out.append(T)
        return out


class DepthICPPoseProvider(PoseProvider):
    """
    RGB-D odometry baseline.

    If depth was captured with --spatial-align-depth, ICP works in the color
    camera frame directly. Otherwise ICP works in the native depth-camera frame
    and is converted to c2w using the recorded depth<->color extrinsics.
    """

    def __init__(
        self,
        voxel_size=0.025,
        max_correspondence=0.08,
        stride=3,
        min_depth=0.20,
        max_depth=4.0,
        max_iteration=25,
        min_fitness=0.12,
    ):
        self.voxel_size = float(voxel_size)
        self.max_correspondence = float(max_correspondence)
        self.stride = int(stride)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_iteration = int(max_iteration)
        self.min_fitness = float(min_fitness)

    def _pcd(self, depth_m, K):
        import open3d as o3d

        h, w = depth_m.shape
        ys, xs = np.mgrid[0:h:self.stride, 0:w:self.stride]
        z = depth_m[::self.stride, ::self.stride]

        valid = (
            np.isfinite(z)
            & (z >= self.min_depth)
            & (z <= self.max_depth)
        )
        uv = np.column_stack([
            xs[valid].astype(np.float64),
            ys[valid].astype(np.float64),
        ])
        pts = backproject_pixels(uv, z[valid].astype(np.float64), K)

        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pts)
        )
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(0.06, self.voxel_size * 3.0),
                max_nn=30,
            )
        )
        return pcd

    def compute(self, session):
        import open3d as o3d

        K = session.K_color if session.spatial_aligned else session.K_depth

        # Pose of the geometry frame used by ICP.
        if session.spatial_aligned:
            geom_to_world = np.eye(4, dtype=np.float64)  # color0 = world
        else:
            # Make color0 = world while ICP geometry is depth0:
            # depth->world = color->world @ depth->color
            geom_to_world = session.T_depth_to_color.copy()

        c2w_out = []

        prev_pcd = None
        last_rel = np.eye(4, dtype=np.float64)

        for i in range(len(session)):
            depth = session.depth_m(i)
            curr_pcd = self._pcd(depth, K)

            if prev_pcd is not None:
                # source=current, target=previous:
                # result maps current geometry frame -> previous geometry frame.
                result = o3d.pipelines.registration.registration_icp(
                    curr_pcd,
                    prev_pcd,
                    self.max_correspondence,
                    last_rel,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.max_iteration
                    ),
                )

                if (
                    np.isfinite(result.fitness)
                    and result.fitness >= self.min_fitness
                    and np.all(np.isfinite(result.transformation))
                ):
                    T_cur_to_prev = np.asarray(
                        result.transformation, dtype=np.float64
                    )
                    geom_to_world = geom_to_world @ T_cur_to_prev
                    last_rel = T_cur_to_prev
                else:
                    last_rel = np.eye(4, dtype=np.float64)

            if session.spatial_aligned:
                c2w = geom_to_world.copy()
            else:
                # HumanEgo relation:
                # c2w = d2w @ c2d
                c2w = geom_to_world @ session.T_color_to_depth

            c2w_out.append(c2w)
            prev_pcd = curr_pcd

        return c2w_out
