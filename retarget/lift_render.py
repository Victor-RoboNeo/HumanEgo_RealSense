# -*- coding: utf-8 -*-
"""Blank-background mesh render of the LIFT replay. No simulator.

    python -m retarget.lift_render --session HumanEgo_data/strawberry/session_20260920_163411
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numba
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retarget.geometry import invert
from retarget.lift_model import ARM_JOINTS, LiftModel, _rpy_matrix, solve_arm
from retarget.mask_anchor import estimate_mask_anchor
from retarget.session import load_session
from retarget.vio_pose import apply_vio_c2w
from retarget.yaw_world import retarget_yaw_world

MESH_DIR = _ROOT / "hardware" / "ARX_Model" / "LIFT" / "meshes" / "lift"
WIDTH, HEIGHT = 960, 720
FX = FY = 820.0


def _load_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if 84 + count * 50 != len(raw):
        raise ValueError(f"not a binary STL: {path}")
    dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tris = np.frombuffer(memoryview(raw)[84:84 + count * 50], dtype=dtype)["v"].astype(np.float64)
    return np.ascontiguousarray(tris)


def _voxel_decimate(tris: np.ndarray, pitch: float) -> np.ndarray:
    """Keep one triangle per occupied voxel so the surface stays closed."""
    quant = np.round(tris.reshape(-1, 3) / pitch).astype(np.int32)
    keys = np.ascontiguousarray(quant).view(np.dtype((np.void, 12))).ravel()
    unique, inverse = np.unique(keys, return_inverse=True)
    corners = inverse.reshape(-1, 3)
    keep = (corners[:, 0] != corners[:, 1]) & (corners[:, 1] != corners[:, 2]) & (corners[:, 0] != corners[:, 2])
    corners = corners[keep]
    ordered = np.sort(corners, axis=1).astype(np.int64)
    face_keys = np.ascontiguousarray(ordered).view(np.dtype((np.void, 24))).ravel()
    _, index = np.unique(face_keys, return_index=True)
    corners = corners[index]
    coords = unique.view(np.int32).reshape(-1, 3).astype(np.float64) * pitch
    return np.ascontiguousarray(coords[corners])


def _link_meshes(pitch: float = 0.016) -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]:
    root = ET.parse(_ROOT / "hardware" / "ARX_Model" / "LIFT" / "lift.urdf").getroot()
    colors = {
        "base_link": (168, 172, 176),
        "wheel_1": (70, 70, 74),
        "wheel_2": (70, 70, 74),
        "wheel_3": (70, 70, 74),
        "lift_1": (150, 154, 160),
        "h_base_link": (186, 186, 190),
        "h_link1_1": (186, 186, 190),
        "h_link2_1": (196, 196, 200),
    }
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        if visual is None or visual.find("geometry") is None:
            continue
        mesh = visual.find("geometry").find("mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        scale = float(mesh.get("scale", "1 1 1").split()[0])
        path = MESH_DIR / Path(mesh.get("filename")).name
        tris = _voxel_decimate(_load_stl(path) * scale, pitch)
        rot = _rpy_matrix(*rpy)
        pts = tris.reshape(-1, 3)
        pts = (rot @ pts.T).T + xyz
        color = colors.get(name, (176, 178, 184))
        if name.startswith("right_arm"):
            color = (168, 176, 188)
        if "catch" in name:
            color = (120, 124, 130)
        out[name] = (pts.reshape(-1, 3, 3), color)
    return out


def _qmap(arm_q: np.ndarray, grasp: float, lift: float = 0.0, head: float = 0.0) -> dict[str, float]:
    q = {"joint_lift": float(lift), "joint_h_1": 0.0, "joint_h_2": float(head)}
    for name, value in zip(ARM_JOINTS, arm_q):
        q[name] = float(value)
    for i in range(1, 7):
        q[f"left_arm_joint{i}"] = 0.0
    closed = float(grasp) > 0.5
    q["right_catch_joint1"] = 0.0 if closed else -0.0445
    q["right_catch_joint2"] = 0.0 if closed else 0.0445
    q["left_catch_joint1"] = -0.0445
    q["left_catch_joint2"] = 0.0445
    return q


def _world_from_ref(points: np.ndarray, T_cam0: np.ndarray) -> np.ndarray:
    """Reference camera coordinates to the gravity-aligned VIO world, Z up."""
    return points @ T_cam0[:3, :3].T + T_cam0[:3, 3]


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.column_stack([right, down, forward])
    return rot


def _shade_colors(normals: np.ndarray, base: tuple[int, int, int]) -> np.ndarray:
    light = np.array([0.35, -0.25, 0.90])
    light = light / np.linalg.norm(light)
    nd = np.abs(normals @ light)
    intensity = 0.36 + 0.64 * nd
    rgb = np.clip(np.array(base) * intensity[:, None], 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


@numba.njit
def _zbuffer(image, zbuf, uv, depth, colors):
    height, width = zbuf.shape
    for face in range(uv.shape[0]):
        x0, y0 = uv[face, 0, 0], uv[face, 0, 1]
        x1, y1 = uv[face, 1, 0], uv[face, 1, 1]
        x2, y2 = uv[face, 2, 0], uv[face, 2, 1]
        z0, z1, z2 = depth[face, 0], depth[face, 1], depth[face, 2]
        if z0 < 0.05 or z1 < 0.05 or z2 < 0.05:
            continue
        minx = int(np.floor(min(x0, x1, x2)))
        maxx = int(np.ceil(max(x0, x1, x2)))
        miny = int(np.floor(min(y0, y1, y2)))
        maxy = int(np.ceil(max(y0, y1, y2)))
        if maxx < 0 or maxy < 0 or minx >= width or miny >= height:
            continue
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx >= width:
            maxx = width - 1
        if maxy >= height:
            maxy = height - 1
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1.0:
            continue
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                w0 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)
                w1 = (x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)
                w2 = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y)
                if area > 0.0:
                    inside = w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0
                else:
                    inside = w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0
                if not inside:
                    continue
                z = (w0 * z0 + w1 * z1 + w2 * z2) / area
                if z < zbuf[y, x]:
                    zbuf[y, x] = z
                    image[y, x, 0] = colors[face, 0]
                    image[y, x, 1] = colors[face, 1]
                    image[y, x, 2] = colors[face, 2]


def render_frame(
    model: LiftModel,
    meshes: dict,
    T_base: np.ndarray,
    arm_q: np.ndarray,
    grasp: float,
    lift: float,
    head: float,
    obj_cam0: np.ndarray,
    T_cam0: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    floor_z: float,
    table_h: float,
    table_xy: np.ndarray,
) -> np.ndarray:
    rot_wc = _look_at(eye, target)
    img = np.full((HEIGHT, WIDTH, 3), 244, dtype=np.uint8)
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    def project(world: np.ndarray):
        cam = (world - eye) @ rot_wc
        z = cam[..., 2]
        uv = np.empty(world.shape[:-1] + (2,))
        uv[..., 0] = FX * cam[..., 0] / z + cx
        uv[..., 1] = FY * cam[..., 1] / z + cy
        return uv, z

    batches_uv = []
    batches_z = []
    batches_c = []

    def push(world_tris: np.ndarray, color: tuple[int, int, int]) -> None:
        uv, depth = project(world_tris)
        normals = np.cross(world_tris[:, 1] - world_tris[:, 0], world_tris[:, 2] - world_tris[:, 0])
        length = np.linalg.norm(normals, axis=1)
        ok = length > 1e-10
        normals = normals[ok] / length[ok, None]
        batches_uv.append(np.ascontiguousarray(uv[ok]))
        batches_z.append(np.ascontiguousarray(depth[ok]))
        batches_c.append(_shade_colors(normals, color))

    # Floor grid in the Z-up plane, fixed under the yawing base.
    grid = []
    for value in np.linspace(-0.8, 1.6, 13):
        grid.append([[value, -0.6, floor_z], [value, 1.8, floor_z]])
        grid.append([[-0.8, value, floor_z], [1.6, value, floor_z]])
    for a, b in grid:
        uv, z = project(np.asarray([a, b], dtype=np.float64))
        if np.any(z < 0.08):
            continue
        cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), (214, 214, 214), 1, cv2.LINE_AA)

    x0, y0 = table_xy[0]
    x1, y1 = table_xy[1]
    push(_box(x0, x1, y0, y1, table_h - 0.025, table_h), (46, 46, 50))
    for cx_leg, cy_leg in ((x0 + 0.08, y0 + 0.08), (x1 - 0.08, y0 + 0.08), (x0 + 0.08, y1 - 0.08), (x1 - 0.08, y1 - 0.08)):
        push(_box(cx_leg - 0.02, cx_leg + 0.02, cy_leg - 0.02, cy_leg + 0.02, floor_z, table_h - 0.025), (88, 88, 92))

    q = _qmap(arm_q, grasp, lift, head)
    link_T = {name: T_base @ model.fk(name, q) for name in meshes}
    for name, (tris, color) in meshes.items():
        posed = link_T[name]
        rot, trans = posed[:3, :3], posed[:3, 3]
        cam0 = tris @ rot.T + trans
        push(_world_from_ref(cam0, T_cam0), color)

    # Object anchor, a small marker so the reach target stays visible.
    center = _world_from_ref(obj_cam0.reshape(1, 3), T_cam0)[0]
    phi = np.linspace(0, np.pi, 6)
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    sphere = []
    radius = 0.018
    for i in range(len(phi) - 1):
        for j in range(len(theta)):
            def sph(ii, jj):
                return center + radius * np.array(
                    [
                        np.sin(phi[ii]) * np.cos(theta[jj % len(theta)]),
                        np.sin(phi[ii]) * np.sin(theta[jj % len(theta)]),
                        np.cos(phi[ii]),
                    ]
                )
            sphere.append([sph(i, j), sph(i + 1, j), sph(i + 1, j + 1)])
            sphere.append([sph(i, j), sph(i + 1, j + 1), sph(i, j + 1)])
    push(np.asarray(sphere, dtype=np.float64), (80, 140, 230))
    uv = np.concatenate(batches_uv, axis=0)
    depth = np.concatenate(batches_z, axis=0)
    colors = np.concatenate(batches_c, axis=0)
    zbuf = np.full((HEIGHT, WIDTH), np.inf)
    _zbuffer(img, zbuf, uv, depth, colors)
    return img


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = (
        (4, 5, 6), (4, 6, 7),
        (0, 2, 1), (0, 3, 2),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    )
    return np.asarray([[corners[a], corners[b], corners[c]] for a, b, c in faces], dtype=np.float64)


def _scene_planes(session: dict) -> tuple[float, float, np.ndarray]:
    """Floor height, table height, and the table XY extent, in the VIO frame.

    Depth shows two horizontal surfaces. The lower one is the floor the robot
    stands on. The upper one is the table, and it is not the ground.
    """
    chunks = []
    for idx in (0, 8, 16):
        fr = session["frames"][idx]
        depth = cv2.imread(str(fr["depth_path"]), cv2.IMREAD_UNCHANGED)
        z = depth.astype(np.float64) * float(fr.get("depth_scale", 0.001))
        K = np.asarray(fr["K"], dtype=np.float64)
        T = np.asarray(fr["T_cam_w"], dtype=np.float64)
        height, width = z.shape
        ys, xs = np.mgrid[0:height:3, 0:width:3]
        zz = z[::3, ::3]
        valid = (zz > 0.25) & (zz < 2.2)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = np.column_stack([(xs[valid] - cx) * zz[valid] / fx, (ys[valid] - cy) * zz[valid] / fy, zz[valid]])
        chunks.append((T[:3, :3] @ pts.T).T + T[:3, 3])
    cloud = np.concatenate(chunks, axis=0)
    hist, edges = np.histogram(cloud[:, 2], bins=120, range=(-1.55, -0.40))
    peaks: list[float] = []
    for i in np.argsort(hist)[::-1]:
        if hist[i] < 200:
            break
        center = float(0.5 * (edges[i] + edges[i + 1]))
        if any(abs(center - peak) < 0.08 for peak in peaks):
            continue
        peaks.append(center)
        if len(peaks) == 2:
            break
    low, high = sorted(peaks)
    floor_z = float(np.median(cloud[np.abs(cloud[:, 2] - low) < 0.03, 2]))
    table_z = float(np.median(cloud[np.abs(cloud[:, 2] - high) < 0.02, 2]))
    table = cloud[np.abs(cloud[:, 2] - table_z) < 0.02]
    return floor_z, table_z, np.percentile(table[:, :2], [5, 95], axis=0)


def _wheel_contacts(model: LiftModel, meshes: dict) -> np.ndarray:
    """Lowest wheel vertices in the base frame at zero lift. These ride with the chassis."""
    pts = []
    for name in ("wheel_1", "wheel_2", "wheel_3"):
        T = model.fk(name, {"joint_lift": 0.0})
        verts = meshes[name][0].reshape(-1, 3)
        base = (T[:3, :3] @ verts.T).T + T[:3, 3]
        pts.append(base[np.argsort(base[:, 2])[:40]])
    return np.concatenate(pts, axis=0)


def _camera_mount(model: LiftModel, lift: float, head: float) -> np.ndarray:
    q = {"joint_lift": float(lift), "joint_h_2": float(head)}
    return model.fk("h_link2_1", q) @ model.T_cam_in_head


def _yaw_about_z(forward: np.ndarray) -> float:
    """Base yaw that aims the camera's forward axis along `forward` in the horizontal plane.

    At this mount the camera looks along base -X, including while the head pitches.
    """
    return float(np.arctan2(forward[1], forward[0]) - np.pi)


def render_session(session_path: Path, out_dir: Path, fps: float = 15.0, pose: str = "vio", write_video: bool = True) -> Path:
    session = load_session(session_path, hand_side="right")
    if pose == "vio":
        apply_vio_c2w(session)
    elif pose != "icp":
        raise ValueError(f"unknown pose source {pose}")
    mask = estimate_mask_anchor(session, n_init=10)
    yaw_rt = retarget_yaw_world(session, T_obj0_w=mask["T_mask_w"])
    npz_path = out_dir / "lift_replay.npz"
    saved = np.load(npz_path)
    arm_q = saved["arm_q"]
    model = LiftModel()
    meshes = _link_meshes()
    floor_vio, table_vio, table_xy = _scene_planes(session)
    # Ground is Z = 0. The recorded camera origin sits above that floor.
    T_cam0 = np.asarray(yaw_rt["T_cam0_w"], dtype=np.float64).copy()
    T_cam0[2, 3] -= floor_vio
    table_h = float(table_vio - floor_vio)
    obj = next(iter(yaw_rt["obj_in_ref"].values()))[:3, 3]
    obj_w = _world_from_ref(obj.reshape(1, 3), T_cam0)[0]
    obj_h = float(obj_w[2])
    contacts = _wheel_contacts(model, meshes)
    z_contact = float(contacts[:, 2].min())
    forward0 = T_cam0[:3, :3] @ np.array([0.0, 0.0, 1.0])
    head = float(np.arctan2(-forward0[2], np.linalg.norm(forward0[:2])))
    base_z = -z_contact
    # The lift only raises the arm platform to the table. The arm tracks the retarget.
    platform_z0 = float(model.fk("right_arm_base_link", {"joint_lift": 0.0})[2, 3])
    lift = float(np.clip(table_h - base_z - platform_z0, 0.0, 0.86))
    lifts = np.full(len(yaw_rt["frames"]), lift)
    platform_h = base_z + platform_z0 + lift
    print(
        f"[lift-render] ground 0  table {table_h*100:.1f} cm  "
        f"platform {platform_h*100:.1f} cm  object {obj_h*100:.1f} cm  "
        f"lift {lift*100:.1f} cm"
    )
    # Stand beside the table and solve the retargeted hand pose from that shoulder.
    shoulder0 = model.fk("right_arm_base_link", {"joint_lift": float(lifts[0])})[:3, 3]
    reach = 0.58
    foot = []
    for name in ("base_link", "wheel_1", "wheel_2", "wheel_3"):
        posed = model.fk(name, {"joint_lift": 0.0})
        verts = meshes[name][0].reshape(-1, 3)
        local = (posed[:3, :3] @ verts.T).T + posed[:3, 3]
        foot.append(local[:, :2])
    foot = np.concatenate(foot, axis=0)
    foot = foot[:: max(len(foot) // 80, 1)]
    table_min, table_max = table_xy[0], table_xy[1]
    bases = []
    targets = []
    hands = []
    for i, fr in enumerate(yaw_rt["frames"]):
        lift = float(lifts[i])
        T_cam_w = T_cam0 @ np.asarray(fr["T_cam_in_ref"], dtype=np.float64)
        yaw = _yaw_about_z(T_cam_w[:3, 2])
        c, s = np.cos(yaw), np.sin(yaw)
        R_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if fr["T_hand_in_ref"] is not None:
            hand_w = _world_from_ref(np.asarray(fr["T_hand_in_ref"])[:3, 3].reshape(1, 3), T_cam0)[0]
            hands.append(hand_w.copy())
        else:
            hands.append(None)
            hand_w = _world_from_ref(obj.reshape(1, 3), T_cam0)[0]
        look = T_cam_w[:3, 2].copy()
        look[2] = 0.0
        look = look / max(np.linalg.norm(look), 1e-6)
        shoulder_xy = hand_w[:2] - 0.34 * look[:2]
        base_xy = shoulder_xy - R_base[:2, :2] @ shoulder0[:2]
        # Keep the chassis on the floor beside the table, not underneath it.
        for _ in range(48):
            world_xy = (R_base[:2, :2] @ foot.T).T + base_xy
            inside = (
                (world_xy[:, 0] > table_min[0] - 0.03)
                & (world_xy[:, 0] < table_max[0] + 0.03)
                & (world_xy[:, 1] > table_min[1] - 0.03)
                & (world_xy[:, 1] < table_max[1] + 0.03)
            )
            if not np.any(inside):
                break
            base_xy = base_xy - 0.04 * look[:2]
        T_world = np.eye(4)
        T_world[:3, :3] = R_base
        T_world[:3, 3] = np.array([base_xy[0], base_xy[1], base_z])
        bases.append(invert(T_cam0) @ T_world)
        if fr["T_hand_in_ref"] is None:
            targets.append(None)
            continue
        hand_base = invert(T_world) @ (T_cam0 @ np.asarray(fr["T_hand_in_ref"], dtype=np.float64))
        point = hand_base[:3, 3].copy()
        offset = point - shoulder0
        dist = float(np.linalg.norm(offset))
        if dist > reach:
            point = shoulder0 + offset * (reach / dist)
        # Retarget tracks the hand position. The gripper approaches the table from above.
        approach = np.array([0.0, 0.0, -1.0])
        spread = np.array([0.0, 1.0, 0.0])
        spread = spread - np.dot(spread, approach) * approach
        palm = np.cross(spread, approach)
        target = np.eye(4)
        target[:3, 0], target[:3, 1], target[:3, 2] = spread, approach, palm
        target[:3, 3] = point
        targets.append(target)
    # No demonstrated hand: keep the chassis where the hand track left it.
    # Snapping over to the object is what made the base teleport.
    tracked = [i for i, hand in enumerate(hands) if hand is not None]
    if tracked:
        for i in range(tracked[0]):
            bases[i] = bases[tracked[0]].copy()
        for i in range(tracked[-1] + 1, len(bases)):
            bases[i] = bases[tracked[-1]].copy()
    arm_solved = np.zeros((len(yaw_rt["frames"]), 6))
    q_arm = np.array([0.6, 1.2, -1.4, -0.9, 0.4, 0.0])
    pos_err = []
    for i, target in enumerate(targets):
        if target is None:
            arm_solved[i] = q_arm
            pos_err.append(np.nan)
            continue
        q_arm = solve_arm(model, target, q_arm, lift=float(lifts[i]), iters=30, rot_weight=0.1)
        arm_solved[i] = q_arm
        got = model.tool_in_base(q_arm, float(lifts[i]))
        pos_err.append(float(np.linalg.norm(got[:3, 3] - target[:3, 3])))
    present = [i for i, err in enumerate(pos_err) if not np.isnan(err)]
    if present:
        arm_solved[: present[0]] = arm_solved[present[0]]
    arm_q = arm_solved
    pos_err = np.asarray(pos_err)
    tool_gap = []
    for i, q in enumerate(arm_solved):
        tool = (T_cam0 @ bases[i]) @ model.tool_in_base(q, float(lifts[i]))
        tool_gap.append(float(np.linalg.norm(tool[:3, 3] - obj_w)))
    tool_gap = np.asarray(tool_gap)
    print(
        f"[lift-render] IK pos median {np.nanmedian(pos_err)*100:.2f} cm  "
        f"max {np.nanmax(pos_err)*100:.2f} cm  "
        f"tool-object median {np.median(tool_gap)*100:.1f} cm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base_T = np.stack([T_cam0 @ np.asarray(b) for b in bases])
    hand_pos = np.full((len(hands), 3), np.nan)
    for i, hand in enumerate(hands):
        if hand is not None:
            hand_pos[i] = hand
    grasp = np.array([0.0 if fr["grasp"] is None else float(fr["grasp"]) for fr in yaw_rt["frames"]])
    traj = out_dir / "lift_mujoco_traj.npz"
    np.savez(
        traj,
        base_T=base_T,
        hand_pos=hand_pos,
        arm_q=arm_q,
        lift=lifts,
        head=np.array([head]),
        grasp=grasp,
        table_h=np.array([table_h]),
        table_xy=np.asarray(table_xy),
        obj_w=obj_w,
        fps=np.array([fps]),
    )
    print(f"[lift-render] wrote {traj}")
    if not write_video:
        return traj
    q0 = arm_q[0]
    T_base0 = bases[0]
    centers = []
    pose0 = _qmap(q0, 0.0, float(lifts[0]), head)
    for name, (tris, _) in meshes.items():
        posed = T_base0 @ model.fk(name, pose0)
        mid = tris.mean(axis=1)
        centers.append(_world_from_ref(mid @ posed[:3, :3].T + posed[:3, 3], T_cam0))
    cloud = np.concatenate(centers, axis=0)
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    look = 0.5 * (lo + hi)
    direction = np.array([0.75, 1.05, 0.42])
    direction = direction / np.linalg.norm(direction)
    eye = look + direction * (float(np.linalg.norm(hi - lo)) * 1.25)
    video = out_dir / "lift_replay_blank.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-crf", "18",
            str(video),
        ],
        stdin=subprocess.PIPE,
    )
    valid = np.flatnonzero(~np.isnan(arm_q[:, 0]))
    last = arm_q[valid[0]].copy() if len(valid) else np.zeros(6)
    stills = {
        int(yaw_rt["frames"][0]["idx"]),
        int(yaw_rt["frames"][len(yaw_rt["frames"]) // 2]["idx"]),
        int(yaw_rt["frames"][int(np.argmax(np.abs(yaw_rt["yaw"])))]["idx"]),
    }
    for i, fr in enumerate(yaw_rt["frames"]):
        q = arm_q[i]
        lift = float(lifts[i])
        T_base = bases[i]
        yaw = _yaw_about_z((T_cam0 @ np.asarray(fr["T_cam_in_ref"]))[:3, 2])
        img = render_frame(
            model, meshes, T_base, q, fr["grasp"] or 0.0, lift, head, obj, T_cam0, eye, look, 0.0, table_h, table_xy,
        )
        cv2.putText(
            img,
            f"ground 0  table {table_h*100:.1f} cm  platform {platform_h*100:.1f} cm  obj {obj_h*100:.1f} cm",
            (24, HEIGHT - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        encoder.stdin.write(np.ascontiguousarray(img).tobytes())
        if int(fr["idx"]) in stills:
            cv2.imwrite(str(out_dir / f"lift_blank_{int(fr['idx']):04d}.png"), img)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode lift_replay_blank.mp4")
    print(f"[lift-render] wrote {video}")
    return video


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the LIFT replay on a blank background.")
    parser.add_argument("--session", default="HumanEgo_data/strawberry/session_20260920_163411")
    parser.add_argument("--out", default="")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pose", choices=["vio", "icp"], default="vio")
    args = parser.parse_args()
    session = Path(args.session)
    if not session.is_absolute():
        session = (_ROOT / session).resolve()
    out = Path(args.out) if args.out else _ROOT / "retarget" / "outputs" / session.name
    if not out.is_absolute():
        out = (_ROOT / out).resolve()
    render_session(session, out, fps=args.fps, pose=args.pose)


if __name__ == "__main__":
    main()
