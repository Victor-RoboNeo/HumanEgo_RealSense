# -*- coding: utf-8 -*-
"""ARX LIFT kinematics for camera-frame replay.

The URDF has no camera link. The replay camera is fixed on ``h_link2`` at
zero head joints: OpenCV X right, Y down, Z forward, and forward is the
base +X the arm reaches toward. Both the head and the right arm sit on
``lift_1``, so lift height does not change the hand target in the camera.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 7)]
HEAD_JOINTS = ["joint_h_1", "joint_h_2"]
LIFT_JOINT = "joint_lift"


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    v = 1.0 - c
    rot = np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot
    return out


def _origin(xyz, rpy) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rpy_matrix(*rpy)
    out[:3, 3] = xyz
    return out


class LiftModel:
    def __init__(self, urdf_path: str | Path | None = None):
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parents[1] / "hardware" / "ARX_Model" / "LIFT" / "lift.urdf"
        root = ET.parse(urdf_path).getroot()
        self.joints: dict[str, dict] = {}
        for node in root.findall("joint"):
            origin = node.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
            axis_node = node.find("axis")
            axis = np.array([float(v) for v in (axis_node.get("xyz", "0 0 1") if axis_node is not None else "0 0 1").split()])
            limit = node.find("limit")
            self.joints[node.get("name")] = {
                "type": node.get("type"),
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "origin": _origin(xyz, rpy),
                "axis": axis,
                "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else -np.pi,
                "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else np.pi,
            }
        self.by_child = {joint["child"]: (name, joint) for name, joint in self.joints.items()}
        # URDF marks the arm continuous. Keep a soft range so IK cannot wind.
        for name in ARM_JOINTS:
            self.joints[name]["lower"] = -np.pi
            self.joints[name]["upper"] = np.pi
        self.T_cam_in_head = np.eye(4)
        # Columns are camera X, Y, Z written in h_link2. At the zero head pose
        # the grippers point along base -X, so that axis is forward.
        # Camera: X = base +Y (right), Y = base -Z (down), Z = base -X (forward).
        self.T_cam_in_head[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        spread = np.array([0.087156, 0.996195, 0.0])
        spread = spread / np.linalg.norm(spread)
        approach = np.array([1.0, 0.0, 0.0])
        approach = approach - np.dot(approach, spread) * spread
        approach = approach / np.linalg.norm(approach)
        palm = np.cross(spread, approach)
        self.T_tool_in_link6 = np.eye(4)
        self.T_tool_in_link6[:3, 0] = spread
        self.T_tool_in_link6[:3, 1] = approach
        self.T_tool_in_link6[:3, 2] = palm
        self.T_tool_in_link6[:3, 3] = np.array([0.074942, -0.006555, -0.005])

    def fk(self, link: str, q: dict[str, float] | None = None) -> np.ndarray:
        q = q or {}
        if link == "base_link":
            return np.eye(4)
        name, joint = self.by_child[link]
        parent = self.fk(joint["parent"], q)
        angle = float(q.get(name, 0.0))
        if joint["type"] in ("revolute", "continuous"):
            motion = _axis_rotation(joint["axis"], angle)
        elif joint["type"] == "prismatic":
            motion = np.eye(4)
            motion[:3, 3] = joint["axis"] * angle
        else:
            motion = np.eye(4)
        return parent @ joint["origin"] @ motion

    def camera_in_base(self, lift: float = 0.0) -> np.ndarray:
        q = {LIFT_JOINT: float(lift)}
        return self.fk("h_link2_1", q) @ self.T_cam_in_head

    def tool_in_base(self, arm_q: np.ndarray, lift: float = 0.0) -> np.ndarray:
        q = {LIFT_JOINT: float(lift)}
        for name, value in zip(ARM_JOINTS, arm_q):
            q[name] = float(value)
        return self.fk("right_arm_link6", q) @ self.T_tool_in_link6

    def link_points(self, arm_q: np.ndarray, lift: float, grasp: float) -> dict[str, np.ndarray]:
        q = {LIFT_JOINT: float(lift)}
        for name, value in zip(ARM_JOINTS, arm_q):
            q[name] = float(value)
        closed = float(grasp) > 0.5
        q["right_catch_joint1"] = 0.0 if closed else -0.0445
        q["right_catch_joint2"] = 0.0 if closed else 0.0445
        names = [
            "base_link",
            "wheel_1",
            "wheel_2",
            "wheel_3",
            "lift_1",
            "h_base_link",
            "h_link1_1",
            "h_link2_1",
            "right_arm_base_link",
            "right_arm_link1",
            "right_arm_link2",
            "right_arm_link3",
            "right_arm_link4",
            "right_arm_link5",
            "right_arm_link6",
            "right_arm_left_catch",
            "right_arm_right_catch",
        ]
        return {name: self.fk(name, q)[:3, 3] for name in names}


def _rotvec(rot: np.ndarray) -> np.ndarray:
    cos_ang = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.arccos(cos_ang))
    if ang < 1e-8:
        return np.zeros(3)
    return (ang / (2.0 * np.sin(ang))) * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )


def pose_error(current: np.ndarray, target: np.ndarray, rot_weight: float = 0.15) -> np.ndarray:
    rot = target[:3, :3] @ current[:3, :3].T
    return np.concatenate([target[:3, 3] - current[:3, 3], rot_weight * _rotvec(rot)])


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    total = np.sum(np.stack(rotations), axis=0)
    left, _, right = np.linalg.svd(total)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        left = left.copy()
        left[:, -1] *= -1.0
    return left @ right


def solve_arm(
    model: LiftModel,
    target_in_base: np.ndarray,
    q0: np.ndarray | None = None,
    lift: float = 0.0,
    iters: int = 40,
    rot_weight: float = 0.15,
) -> np.ndarray:
    """Damped least squares. Warm-starts from ``q0`` and a few elbow seeds."""
    seeds = []
    if q0 is not None:
        seeds.append(np.asarray(q0, dtype=np.float64))
    seeds.extend(
        [
            np.zeros(6),
            np.array([0.8, -1.0, 1.2, -0.8, 0.6, 0.0]),
            np.array([-0.6, -0.8, 1.4, -1.0, -0.4, 0.4]),
            np.array([1.2, -1.2, 0.6, -0.4, 1.0, -0.5]),
        ]
    )
    best_q, best_err = seeds[0], np.inf
    # Warm start gets the full budget. Extra seeds only run if it is still short.
    budgets = [iters] + [16] * (len(seeds) - 1)
    for seed, budget in zip(seeds, budgets):
        if best_err < 0.015:
            break
        q = _ik_once(model, target_in_base, seed, lift, budget, rot_weight)
        err = float(np.linalg.norm(pose_error(model.tool_in_base(q, lift), target_in_base, rot_weight)[:3]))
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def _ik_once(
    model: LiftModel,
    target: np.ndarray,
    q0: np.ndarray,
    lift: float,
    iters: int,
    rot_weight: float,
) -> np.ndarray:
    q = np.asarray(q0, dtype=np.float64).copy()
    lower = np.array([model.joints[name]["lower"] for name in ARM_JOINTS])
    upper = np.array([model.joints[name]["upper"] for name in ARM_JOINTS])
    eps = 1e-4
    for _ in range(iters):
        current = model.tool_in_base(q, lift)
        err = pose_error(current, target, rot_weight)
        if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 0.02:
            break
        jac = np.zeros((6, 6))
        for i in range(6):
            step = np.zeros(6)
            step[i] = eps
            jac[:, i] = (pose_error(model.tool_in_base(q + step, lift), target, rot_weight) - err) / eps
        damp = 1e-2 * np.eye(6)
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damp, err)
        delta = np.clip(delta, -0.35, 0.35)
        q = np.clip(q - delta, lower, upper)
    return q
