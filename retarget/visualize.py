# -*- coding: utf-8 -*-
"""Visualize retargeted camera + gripper in the object frame and on RGB."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from retarget.core import stack_traj
from retarget.geometry import (
    camera_frustum_local,
    gripper_wireframe,
    invert,
    project_cam,
    transform_points,
)


def _set_equal_3d(ax, pts: np.ndarray, pad: float = 0.08) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    r = float(np.max(np.abs(pts - c))) + pad
    r = max(r, 0.15)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_axes(ax, T: np.ndarray, length: float, lw: float = 1.4) -> None:
    p = T[:3, 3]
    R = T[:3, :3] * length
    ax.quiver(p[0], p[1], p[2], R[0, 0], R[1, 0], R[2, 0], color="#c0392b", linewidth=lw)
    ax.quiver(p[0], p[1], p[2], R[0, 1], R[1, 1], R[2, 1], color="#27ae60", linewidth=lw)
    ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2], color="#2980b9", linewidth=lw)


def _polyline_collection(pts: np.ndarray, edges, color, lw=1.2) -> Line3DCollection:
    segs = [(pts[i], pts[j]) for i, j in edges]
    return Line3DCollection(segs, colors=color, linewidths=lw)


def save_object_frame_figure(retargeted: dict[str, Any], out_path: Path) -> None:
    frames = retargeted["frames"]
    T_cam, cam_idx = stack_traj(frames, "T_cam_in_ref")
    T_hand, hand_idx = stack_traj(frames, "T_hand_in_ref")

    fig = plt.figure(figsize=(11.5, 5.4), dpi=160)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    is_cam = retargeted.get("frame_mode", "camera_frame") == "camera_frame"
    origin_name = "cam0" if is_cam else "obj1"
    all_pts = [np.zeros((1, 3))]
    ax3.scatter([0], [0], [0], c="#f1c40f", s=60, zorder=5, label=f"{origin_name} origin")
    _draw_axes(ax3, np.eye(4), 0.08, lw=2.0)
    obj_in_ref = retargeted.get("obj_in_ref") or {}
    if is_cam and obj_in_ref:
        T_obj = next(iter(obj_in_ref.values()))
        ax3.scatter([T_obj[0, 3]], [T_obj[1, 3]], [T_obj[2, 3]], c="#27ae60", s=50, marker="X", label="obj1")
        all_pts.append(T_obj[:3, 3].reshape(1, 3))

    if len(T_cam):
        pc = T_cam[:, :3, 3]
        ax3.plot(pc[:, 0], pc[:, 1], pc[:, 2], color="#8e44ad", lw=1.6, label="camera")
        all_pts.append(pc)
        stride = max(1, len(T_cam) // 12)
        fr_pts, fr_edges = camera_frustum_local(0.035)
        for T in T_cam[::stride]:
            wp = transform_points(T, fr_pts)
            ax3.add_collection3d(_polyline_collection(wp, fr_edges, "#8e44ad", 0.8))
        _draw_axes(ax3, T_cam[0], 0.05)
        _draw_axes(ax3, T_cam[-1], 0.05)

    if len(T_hand):
        ph = T_hand[:, :3, 3]
        ax3.plot(ph[:, 0], ph[:, 1], ph[:, 2], color="#e67e22", lw=1.8, label="gripper")
        all_pts.append(ph)
        g_pts, g_edges = gripper_wireframe(0.0)
        stride = max(1, len(T_hand) // 10)
        for T in T_hand[::stride]:
            wp = transform_points(T, g_pts)
            ax3.add_collection3d(_polyline_collection(wp, g_edges, "#e67e22", 1.0))
        _draw_axes(ax3, T_hand[0], 0.04)
        _draw_axes(ax3, T_hand[-1], 0.04)

    _set_equal_3d(ax3, np.concatenate(all_pts, axis=0))
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title("OpenCV cam0 (X right, Y down, Z forward)" if is_cam else "Object frame (strawberry origin)")
    ax3.view_init(elev=22, azim=-55)
    ax3.legend(loc="upper left", fontsize=8)

    if len(T_cam):
        ax2.plot(T_cam[:, 0, 3], T_cam[:, 1, 3], color="#8e44ad", lw=1.6, label="camera XY")
        ax2.scatter(T_cam[0, 0, 3], T_cam[0, 1, 3], c="#8e44ad", s=28, marker="o")
        ax2.scatter(T_cam[-1, 0, 3], T_cam[-1, 1, 3], c="#8e44ad", s=36, marker="*")
    if len(T_hand):
        ax2.plot(T_hand[:, 0, 3], T_hand[:, 1, 3], color="#e67e22", lw=1.8, label="gripper XY")
        ax2.scatter(T_hand[0, 0, 3], T_hand[0, 1, 3], c="#e67e22", s=28, marker="o")
        ax2.scatter(T_hand[-1, 0, 3], T_hand[-1, 1, 3], c="#e67e22", s=36, marker="*")
    ax2.scatter([0], [0], c="#f1c40f", s=70, marker="X", label=origin_name, zorder=5)
    if is_cam and obj_in_ref:
        ax2.scatter([T_obj[0, 3]], [T_obj[1, 3]], c="#27ae60", s=50, marker="X", label="obj1", zorder=5)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("X_ref (m)")
    ax2.set_ylabel("Y_ref (m)")
    ax2.set_title("Top-down in cam0 XY (image plane)" if is_cam else "Top-down in object XY")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    if retargeted.get("T_floor_w") is not None:
        title = "camera_frame GT  ·  JSON cam0  ·  y_base = in-place yaw"
        ax3.set_xlabel("X (m)  image right")
        ax3.set_ylabel("Y (m)  image down")
        ax3.set_zlabel("Z (m)  forward")
        ax3.set_title("OpenCV cam0 after wiping camera radius")
        ax3.view_init(elev=22, azim=-55)
        ax2.set_xlabel("X_cam0 (m)  right")
        ax2.set_ylabel("Y_cam0 (m)  down")
        ax2.set_title("y_action in paper camera_frame")
    else:
        title = (
            "Retargeted  ·  T_x_in_ref = inv(cam0) @ T_x_world  (paper camera_frame)"
            if is_cam
            else "Retargeted  ·  T_x_in_ref = inv(T_anchor) @ T_x_world"
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_compensation_figure(retargeted: dict[str, Any], out_path: Path) -> None:
    """Camera travel vs gripper-in-object vs gripper-in-camera (arm residual)."""
    frames = retargeted["frames"]
    idxs, d_cam, d_hand_ref, d_hand_cam, grasp = [], [], [], [], []
    for fr in frames:
        idxs.append(fr["idx"])
        d_cam.append(float(np.linalg.norm(fr["T_cam_in_ref"][:3, 3])))
        if fr["T_hand_in_ref"] is None:
            d_hand_ref.append(np.nan)
            d_hand_cam.append(np.nan)
            grasp.append(np.nan)
        else:
            d_hand_ref.append(float(np.linalg.norm(fr["T_hand_in_ref"][:3, 3])))
            d_hand_cam.append(float(np.linalg.norm(fr["T_hand_in_cam"][:3, 3])))
            grasp.append(fr["grasp"])

    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
    ax.plot(idxs, d_cam, color="#8e44ad", lw=1.8, label="|T_cam_in_ref|  camera vs cam0")
    ax.plot(idxs, d_hand_ref, color="#e67e22", lw=1.8, label="|T_hand_in_ref|  gripper vs cam0")
    ax.plot(idxs, d_hand_cam, color="#16a085", lw=1.8, label="|T_ee_in_base|  gripper vs camera")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance (m)")
    ax.set_title("Base compensation: camera yaws in place; gripper-in-cam is the arm residual")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_projected_gripper(img: np.ndarray, T_hand_cam: np.ndarray, K: np.ndarray, grasp: float) -> None:
    pts, edges = gripper_wireframe(grasp)
    pts_c = transform_points(T_hand_cam, pts)
    uv, valid = project_cam(K, pts_c)
    h, w = img.shape[:2]
    color = (255, 255, 255) if grasp > 0.5 else (0, 200, 255)
    for i, j in edges:
        if not (valid[i] and valid[j]):
            continue
        p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            pass
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(img, p1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(img, p2, 3, color, -1, cv2.LINE_AA)

    origin = transform_points(T_hand_cam, np.zeros((1, 3)))[0]
    axes = transform_points(
        T_hand_cam,
        np.array([[0.04, 0, 0], [0, 0.04, 0], [0, 0, 0.04]], dtype=np.float64),
    )
    uv_o, v_o = project_cam(K, origin.reshape(1, 3))
    uv_a, v_a = project_cam(K, axes)
    if v_o[0]:
        o = (int(round(uv_o[0, 0])), int(round(uv_o[0, 1])))
        cols = [(0, 0, 255), (0, 200, 0), (255, 80, 80)]
        for k in range(3):
            if not v_a[k]:
                continue
            a = (int(round(uv_a[k, 0])), int(round(uv_a[k, 1])))
            cv2.line(img, o, a, cols[k], 2, cv2.LINE_AA)


def _project_world_to_uv(T_cam_w: np.ndarray, K: np.ndarray, p_w: np.ndarray) -> tuple[int, int] | None:
    R, t = T_cam_w[:3, :3], T_cam_w[:3, 3]
    p_c = R.T @ (np.asarray(p_w, dtype=np.float64) - t)
    uv, valid = project_cam(K, p_c.reshape(1, 3))
    if not valid[0]:
        return None
    return int(round(uv[0, 0])), int(round(uv[0, 1]))


def _draw_uv_marker(img: np.ndarray, uv, color, marker, label: str, origin_xy=(8, -8)) -> None:
    if uv is None:
        return
    h, w = img.shape[:2]
    if not (0 <= uv[0] < w and 0 <= uv[1] < h):
        return
    cv2.drawMarker(img, uv, color, marker, 18, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (uv[0] + origin_xy[0], uv[1] + origin_xy[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def write_overlay_video(
    retargeted: dict[str, Any],
    out_path: Path,
    fps: float | None = None,
    T_mask_w: np.ndarray | None = None,
    mask_uv_by_idx: dict[int, np.ndarray] | None = None,
) -> int:
    fps = float(fps or retargeted.get("fps") or 15.0)
    T_src_w = np.asarray(retargeted["T_anchor_w"], dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    for fr in retargeted["frames"]:
        rgb_path = fr.get("rgb_path")
        if not rgb_path or not Path(rgb_path).is_file():
            continue
        img = cv2.imread(rgb_path)
        if img is None:
            continue
        T_cam_w = fr.get("T_cam_w")
        if T_cam_w is None:
            T_cam_w = np.linalg.inv(fr["T_cam_in_ref"]) @ T_src_w

        uv_src = _project_world_to_uv(T_cam_w, fr["K"], T_src_w[:3, 3])
        _draw_uv_marker(img, uv_src, (0, 220, 255), cv2.MARKER_TILTED_CROSS, "cotracker")
        if T_mask_w is not None:
            uv_mask = _project_world_to_uv(T_cam_w, fr["K"], T_mask_w[:3, 3])
            _draw_uv_marker(img, uv_mask, (80, 220, 80), cv2.MARKER_CROSS, "mask+depth", (8, 14))
        live = None if mask_uv_by_idx is None else mask_uv_by_idx.get(fr["idx"])
        if live is not None:
            p = (int(round(live[0])), int(round(live[1])))
            _draw_uv_marker(img, p, (255, 180, 80), cv2.MARKER_DIAMOND, "live mask", (8, 28))

        if fr["T_hand_in_cam"] is not None:
            _draw_projected_gripper(img, fr["T_hand_in_cam"], fr["K"], float(fr["grasp"] or 0.0))
            tag = "CLOSED" if fr["grasp"] > 0.5 else "OPEN"
        else:
            tag = "no hand"
        cv2.putText(
            img,
            f"idx {fr['idx']:04d}  gripper {tag}  yellow=cotracker  green=mask+depth",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        writer.write(img)
        n += 1
    if writer is not None:
        writer.release()
    return n


def save_anchor_compare_figure(
    session: dict[str, Any],
    mask_info: dict[str, Any],
    out_path: Path,
    frame_idx: int = 0,
) -> None:
    """Frame-0 RGB with both origins, plus XY of the two frozen world origins."""
    frames = {fr["idx"]: fr for fr in session["frames"]}
    if frame_idx not in frames:
        frame_idx = session["frames"][0]["idx"]
    fr = frames[frame_idx]
    img = cv2.imread(fr["rgb_path"])
    if img is None:
        raise FileNotFoundError(fr["rgb_path"])
    img = img.copy()
    T_src = mask_info["T_src_w"]
    T_mask = mask_info["T_mask_w"]
    uv_src = _project_world_to_uv(fr["T_cam_w"], fr["K"], T_src[:3, 3])
    uv_mask = _project_world_to_uv(fr["T_cam_w"], fr["K"], T_mask[:3, 3])
    live = None
    for rec in mask_info["per_frame"]:
        if rec["idx"] == frame_idx:
            live = rec.get("uv")
            break
    mask = None
    if fr.get("mask_obj1_path") and Path(fr["mask_obj1_path"]).is_file():
        mask = cv2.imread(fr["mask_obj1_path"], cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (80, 220, 80), 1, cv2.LINE_AA)
    _draw_uv_marker(img, uv_src, (0, 220, 255), cv2.MARKER_TILTED_CROSS, "cotracker origin")
    _draw_uv_marker(img, uv_mask, (80, 220, 80), cv2.MARKER_CROSS, "mask+depth origin", (8, 14))
    if live is not None:
        p = (int(round(live[0])), int(round(live[1])))
        _draw_uv_marker(img, p, (255, 180, 80), cv2.MARKER_DIAMOND, "this-frame mask", (8, 28))

    delta = mask_info["delta_w"]
    fig = plt.figure(figsize=(11.2, 4.8), dpi=150)
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"frame {frame_idx:04d}  |dt|={np.linalg.norm(delta)*100:.1f} cm")
    ax0.axis("off")

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.scatter([T_src[0, 3]], [T_src[1, 3]], c="#f1c40f", s=90, marker="X", label="cotracker T_anchor", zorder=5)
    ax1.scatter([T_mask[0, 3]], [T_mask[1, 3]], c="#27ae60", s=90, marker="+", linewidths=2, label="mask+depth origin", zorder=5)
    ax1.annotate(
        "",
        xy=(T_mask[0, 3], T_mask[1, 3]),
        xytext=(T_src[0, 3], T_src[1, 3]),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
    )
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("world X (m)  color0")
    ax1.set_ylabel("world Y (m)")
    ax1.set_title("Frozen origins in collection world")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _map_pt(
    p_xy: np.ndarray,
    origin: np.ndarray,
    scale: float,
    canvas: int,
    flip_y: bool = False,
) -> tuple[int, int]:
    u = int(round((p_xy[0] - origin[0]) * scale + canvas / 2.0))
    y = p_xy[1] - origin[1]
    v = int(round(((-y) if flip_y else y) * scale + canvas / 2.0))
    return u, v


def write_topdown_video(retargeted: dict[str, Any], out_path: Path, fps: float | None = None) -> int:
    """Animate camera + gripper in object XY (same plane as vis_object_frame top-down)."""
    frames = retargeted["frames"]
    T_cam, _ = stack_traj(frames, "T_cam_in_ref")
    T_hand, _ = stack_traj(frames, "T_hand_in_ref")
    pts = [np.zeros((1, 2))]
    if len(T_cam):
        pts.append(T_cam[:, :2, 3])
    if len(T_hand):
        pts.append(T_hand[:, :2, 3])
    all_xy = np.concatenate(pts, axis=0)
    span = float(np.max(np.linalg.norm(all_xy - all_xy.mean(axis=0), axis=1)) + 0.12)
    canvas = 720
    scale = (canvas * 0.42) / max(span, 1e-3)
    origin = all_xy.mean(axis=0)
    fps = float(fps or retargeted.get("fps") or 15.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas, canvas)
    )

    flip_y = False
    is_cam = retargeted.get("frame_mode") == "camera_frame"
    origin_label = "cam0" if is_cam else "obj1"
    cam_hist: list[tuple[int, int]] = []
    hand_hist: list[tuple[int, int]] = []
    last_hand = None
    n = 0
    obj_xy = None
    obj_in_ref = retargeted.get("obj_in_ref") or {}
    if is_cam and obj_in_ref:
        obj_xy = next(iter(obj_in_ref.values()))[:2, 3]
    for fr in frames:
        img = np.full((canvas, canvas, 3), 24, dtype=np.uint8)
        o = _map_pt(np.zeros(2), origin, scale, canvas, flip_y)
        cv2.drawMarker(img, o, (0, 220, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(img, origin_label, (o[0] + 10, o[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        if obj_xy is not None:
            op = _map_pt(obj_xy, origin, scale, canvas, flip_y)
            cv2.drawMarker(img, op, (80, 200, 80), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(img, "obj1", (op[0] + 10, op[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)

        c = _map_pt(fr["T_cam_in_ref"][:2, 3], origin, scale, canvas, flip_y)
        cam_hist.append(c)
        if len(cam_hist) > 1:
            cv2.polylines(img, [np.array(cam_hist, np.int32)], False, (180, 80, 180), 2, cv2.LINE_AA)
        # Floor heading: W* Y* is forward; otherwise use optical +Z XY.
        if flip_y:
            fwd = fr["T_cam_in_ref"][:2, 1]
        else:
            fwd = fr["T_cam_in_ref"][:2, 2]
        tip = _map_pt(
            fr["T_cam_in_ref"][:2, 3] + 0.06 * fwd / (np.linalg.norm(fwd) + 1e-8),
            origin,
            scale,
            canvas,
            flip_y,
        )
        cv2.arrowedLine(img, c, tip, (200, 120, 220), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, c, 5, (200, 120, 220), -1)

        if fr["T_hand_in_ref"] is not None:
            last_hand = fr
        if last_hand is not None:
            h = _map_pt(last_hand["T_hand_in_ref"][:2, 3], origin, scale, canvas, flip_y)
            hand_hist.append(h)
            if len(hand_hist) > 1:
                cv2.polylines(img, [np.array(hand_hist, np.int32)], False, (40, 140, 255), 2, cv2.LINE_AA)
            g_pts, g_edges = gripper_wireframe(float(last_hand["grasp"] or 0.0))
            g_w = transform_points(last_hand["T_hand_in_ref"], g_pts)
            col = (255, 255, 255) if last_hand["grasp"] > 0.5 else (40, 180, 255)
            for i, j in g_edges:
                p1 = _map_pt(g_w[i, :2], origin, scale, canvas, flip_y)
                p2 = _map_pt(g_w[j, :2], origin, scale, canvas, flip_y)
                cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
            cv2.circle(img, h, 4, col, -1)

        tag = "OpenCV cam0  X=right Y=down" if is_cam else "object XY"
        cv2.putText(
            img,
            f"{tag}  idx {fr['idx']:04d}  purple=camera  orange=gripper",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
        n += 1
    writer.release()
    return n


def save_poses_npz(retargeted: dict[str, Any], out_path: Path) -> None:
    frames = retargeted["frames"]
    n = len(frames)
    T_cam = np.stack([fr["T_cam_in_ref"] for fr in frames])
    T_hand = np.full((n, 4, 4), np.nan)
    T_ee = np.full((n, 4, 4), np.nan)
    grasp = np.full((n,), np.nan)
    has = np.zeros((n,), dtype=np.uint8)
    idxs = np.array([fr["idx"] for fr in frames], dtype=np.int32)
    for i, fr in enumerate(frames):
        if fr["T_hand_in_ref"] is None:
            continue
        has[i] = 1
        T_hand[i] = fr["T_hand_in_ref"]
        T_ee[i] = fr["T_ee_in_base"]
        grasp[i] = fr["grasp"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        idx=idxs,
        T_cam_in_ref=T_cam,
        T_hand_in_ref=T_hand,
        T_ee_in_base=T_ee,
        grasp=grasp,
        has_hand=has,
        T_anchor_w=retargeted["T_anchor_w"],
        T_align=retargeted["T_align"],
        anchor_key=np.array(retargeted["anchor_key"]),
        **(
            {
                "yaw": np.asarray(retargeted["yaw"], dtype=np.float64),
                "y_base": np.asarray(retargeted["y_base"], dtype=np.float64),
                "y_base_delta": np.asarray(retargeted["y_base_delta"], dtype=np.float64),
                "T_cam0_w": retargeted["T_cam0_w"],
                "T_floor_w": retargeted["T_floor_w"],
                "T_pivot_w": retargeted["T_pivot_w"],
                "T_base_in_ref": np.stack([fr["T_base_in_ref"] for fr in frames]),
                "T_cam_virt": np.stack([fr["T_cam_virt"] for fr in frames]),
                "radius_m": np.array(retargeted.get("radius_m", np.nan)),
                "wipe_radius": np.array(bool(retargeted.get("wipe_radius", True))),
            }
            if retargeted.get("y_base") is not None
            else {}
        ),
    )


def save_yaw_world_figure(
    yaw_retargeted: dict[str, Any],
    icp_retargeted: dict[str, Any] | None,
    out_path: Path,
) -> None:
    """Wipe-radius chassis vs ICP arc; y_action gripper in paper OpenCV cam0."""
    frames = yaw_retargeted["frames"]
    idxs = np.array([fr["idx"] for fr in frames])
    yaw = np.asarray(yaw_retargeted["yaw"], dtype=np.float64)
    y_base = np.asarray(yaw_retargeted["y_base"], dtype=np.float64)
    y_base_delta = np.asarray(yaw_retargeted["y_base_delta"], dtype=np.float64)
    T_floor = np.asarray(
        yaw_retargeted.get("T_floor_w", yaw_retargeted.get("T_Wstar")),
        dtype=np.float64,
    )
    T_w2floor = np.linalg.inv(T_floor)
    t_icp_star = np.stack([(T_w2floor @ fr["T_cam_w"])[:3, 3] for fr in frames])
    t_arc_star = np.stack([(T_w2floor @ fr["T_cam_arc"])[:3, 3] for fr in frames])
    t_virt_star = np.stack([(T_w2floor @ fr["T_cam_virt"])[:3, 3] for fr in frames])
    T_hand_y, _ = stack_traj(frames, "T_hand_in_ref")
    T_hand_i_list = []
    T_w2ref = np.linalg.inv(np.asarray(yaw_retargeted["T_ref_w"], dtype=np.float64))
    for fr in frames:
        if fr.get("T_hand_in_cam") is None:
            continue
        T_hand_i_list.append(T_w2ref @ fr["T_cam_w"] @ fr["T_hand_in_cam"])
    T_hand_i = np.stack(T_hand_i_list) if T_hand_i_list else None
    _ = icp_retargeted
    pivot_star = (T_w2floor @ yaw_retargeted["T_pivot_w"])[:3, 3]
    r_m = float(yaw_retargeted.get("radius_m") or 0.0)

    fig = plt.figure(figsize=(12.2, 8.2), dpi=150)
    ax_floor = fig.add_subplot(2, 2, 1)
    ax_yaw = fig.add_subplot(2, 2, 2)
    ax_grip = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)

    ax_floor.plot(t_icp_star[:, 0], t_icp_star[:, 1], color="#bdc3c7", lw=1.2, label="ICP camera (arc)")
    ax_floor.plot(t_arc_star[:, 0], t_arc_star[:, 1], color="#95a5a6", lw=1.2, ls="--", label="lever-arm ghost")
    ax_floor.scatter(t_virt_star[:, 0], t_virt_star[:, 1], c="#8e44ad", s=18, zorder=4, label="y_base chassis (r=0)")
    ax_floor.scatter([0], [0], c="#f1c40f", s=80, marker="X", zorder=5, label="cam0")
    p_obj_floor = (T_w2floor @ np.asarray(yaw_retargeted["T_obj0_w"], dtype=np.float64))[:3, 3]
    ax_floor.scatter([p_obj_floor[0]], [p_obj_floor[1]], c="#27ae60", s=70, marker="X", zorder=5, label="obj1")
    ax_floor.scatter([pivot_star[0]], [pivot_star[1]], c="#7f8c8d", s=28, marker="o", zorder=5, label="wiped pivot")
    stride = max(1, len(frames) // 16)
    for fr in frames[::stride]:
        T_cf = T_w2floor @ fr["T_cam_virt"]
        p = T_cf[:3, 3]
        f = 0.12 * T_cf[:3, 1]
        ax_floor.annotate(
            "",
            xy=(p[0] + f[0], p[1] + f[1]),
            xytext=(p[0], p[1]),
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.0),
        )
    ax_floor.set_aspect("equal", adjustable="datalim")
    ax_floor.set_xlabel("floor X (m)  image right")
    ax_floor.set_ylabel("floor Y (m)  forward")
    ax_floor.set_title(f"Wipe radius r={r_m*100:.0f} cm  ·  chassis spins in place")
    ax_floor.grid(True, alpha=0.3)
    ax_floor.legend(fontsize=7)

    ax_yaw.plot(idxs, np.rad2deg(yaw), color="#e67e22", lw=1.6, label="body yaw")
    ax_yaw.plot(idxs, np.rad2deg(y_base_delta[:, 2]), color="#8e44ad", lw=1.2, ls=":", label="y_base Δyaw")
    ax_yaw.set_xlabel("frame")
    ax_yaw.set_ylabel("deg")
    ax_yaw.set_title("y_base = (0, 0, yaw) after wiping xy")
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.legend(fontsize=7)

    if T_hand_i is not None and len(T_hand_i):
        ax_grip.plot(
            T_hand_i[:, 0, 3], T_hand_i[:, 1, 3],
            color="#bdc3c7", lw=1.3, label="gripper if ICP cam kept",
        )
    if len(T_hand_y):
        ax_grip.plot(
            T_hand_y[:, 0, 3], T_hand_y[:, 1, 3],
            color="#e67e22", lw=1.8, label="y_action  T_hand_in_ref",
        )
        ax_grip.scatter(T_hand_y[0, 0, 3], T_hand_y[0, 1, 3], c="#e67e22", s=28)
        ax_grip.scatter(T_hand_y[-1, 0, 3], T_hand_y[-1, 1, 3], c="#e67e22", s=40, marker="*")
    ax_grip.scatter([0], [0], c="#f1c40f", s=70, marker="X", zorder=5, label="cam0")
    t_obj = yaw_retargeted.get("obj_in_ref") or {}
    if t_obj:
        p_obj = next(iter(t_obj.values()))[:3, 3]
        ax_grip.scatter([p_obj[0]], [p_obj[1]], c="#27ae60", s=50, marker="X", zorder=5, label="obj1")
    ax_grip.set_aspect("equal", adjustable="datalim")
    ax_grip.set_xlabel("X_cam0 (m)  right")
    ax_grip.set_ylabel("Y_cam0 (m)  down")
    ax_grip.set_title("y_action in paper camera_frame (OpenCV cam0)")
    ax_grip.grid(True, alpha=0.3)
    ax_grip.legend(fontsize=7)

    ax_d.plot(idxs, np.linalg.norm(t_icp_star - t_icp_star[0], axis=1), color="#bdc3c7", lw=1.4, label="|Δt| ICP")
    ax_d.plot(idxs, np.linalg.norm(y_base[:, :2] - y_base[0, :2], axis=1), color="#8e44ad", lw=1.8, label="|Δxy| y_base")
    ax_d.set_xlabel("frame")
    ax_d.set_ylabel("m")
    ax_d.set_title("Mobile GT: xy frozen, only yaw is sent to the chassis")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=7)

    fig.suptitle(
        "Replay GT  ·  T_ee_in_base = inv(T_base) @ T_ee_world  ·  T_cam_in_base = I",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _horizon_uv_line(K: np.ndarray, up_cam: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Vanishing line of the gravity-horizontal plane, in pixels."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ux, uy, uz = np.asarray(up_cam, dtype=np.float64).reshape(3)
    pts = []
    if abs(uy) > 1e-6:
        for u in (0.0, float(width - 1)):
            v = cy - (fy / uy) * (ux * (u - cx) / fx + uz)
            pts.append((u, v))
    elif abs(ux) > 1e-6:
        for v in (0.0, float(height - 1)):
            u = cx - (fx / ux) * (uy * (v - cy) / fy + uz)
            pts.append((u, v))
    else:
        return None
    return np.array(pts, dtype=np.float64)


def _clip_line_to_image(p0, p1, w: int, h: int):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0, w - 1 - x0, y0, h - 1 - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            u0 = max(u0, t)
        else:
            u1 = min(u1, t)
        if u0 > u1:
            return None
    a = (int(round(x0 + u0 * dx)), int(round(y0 + u0 * dy)))
    b = (int(round(x0 + u1 * dx)), int(round(y0 + u1 * dy)))
    return a, b


def _floor_grid_in_cam(T_floor_in_cam: np.ndarray, z_floor: float, half: float = 0.55, step: float = 0.10):
    xs = np.arange(-half, half + 1e-9, step)
    ys = np.arange(-half, half + 1e-9, step)
    segs = []
    for x in xs:
        p0 = (T_floor_in_cam @ np.array([x, -half, z_floor, 1.0]))[:3]
        p1 = (T_floor_in_cam @ np.array([x, half, z_floor, 1.0]))[:3]
        segs.append((p0, p1))
    for y in ys:
        p0 = (T_floor_in_cam @ np.array([-half, y, z_floor, 1.0]))[:3]
        p1 = (T_floor_in_cam @ np.array([half, y, z_floor, 1.0]))[:3]
        segs.append((p0, p1))
    return segs


def _draw_horizon_on_rgb(img, K, T_cam_w, T_floor_w, up_w, z_floor) -> None:
    h, w = img.shape[:2]
    T_floor_in_cam = invert(T_cam_w) @ T_floor_w
    up_cam = T_cam_w[:3, :3].T @ np.asarray(up_w, dtype=np.float64).reshape(3)

    hz = _horizon_uv_line(K, up_cam, w, h)
    if hz is not None:
        clipped = _clip_line_to_image(hz[0], hz[1], w, h)
        if clipped is not None:
            cv2.line(img, clipped[0], clipped[1], (40, 220, 255), 2, cv2.LINE_AA)
            mid = ((clipped[0][0] + clipped[1][0]) // 2, (clipped[0][1] + clipped[1][1]) // 2)
            cv2.putText(
                img, "horizon", (mid[0] + 8, max(18, mid[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 255), 1, cv2.LINE_AA,
            )

    for a, b in _floor_grid_in_cam(T_floor_in_cam, z_floor):
        uv, valid = project_cam(K, np.stack([a, b]))
        if not (valid[0] and valid[1]):
            continue
        p0 = (int(round(uv[0, 0])), int(round(uv[0, 1])))
        p1 = (int(round(uv[1, 0])), int(round(uv[1, 1])))
        cv2.line(img, p0, p1, (255, 180, 40), 1, cv2.LINE_AA)

    origin = (T_floor_in_cam @ np.array([0.0, 0.0, z_floor, 1.0]))[:3]
    axes = np.array(
        [[0.18, 0.0, z_floor, 1.0], [0.0, 0.18, z_floor, 1.0], [0.0, 0.0, z_floor + 0.18, 1.0]]
    )
    a_c = (T_floor_in_cam @ axes.T).T[:, :3]
    uv_o, v_o = project_cam(K, origin.reshape(1, 3))
    uv_a, v_a = project_cam(K, a_c)
    if v_o[0]:
        o = (int(round(uv_o[0, 0])), int(round(uv_o[0, 1])))
        labels = [("Xh", (0, 0, 255)), ("Yh", (0, 200, 0)), ("Zh up", (255, 80, 80))]
        for k, (lab, col) in enumerate(labels):
            if not v_a[k]:
                continue
            a = (int(round(uv_a[k, 0])), int(round(uv_a[k, 1])))
            cv2.line(img, o, a, col, 2, cv2.LINE_AA)
            cv2.putText(img, lab, (a[0] + 4, a[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)


def save_cam_horizon_figure(yaw_retargeted: dict[str, Any], out_path: Path, frame_idx: int = 0) -> None:
    """New gravity-horizontal expressed in OpenCV camera coordinates."""
    frames = yaw_retargeted["frames"]
    fr = next((f for f in frames if int(f["idx"]) == int(frame_idx)), frames[0])
    T_cam_w = np.asarray(fr.get("T_cam_virt", fr["T_cam_w"]), dtype=np.float64)
    T_floor_w = np.asarray(yaw_retargeted["T_floor_w"], dtype=np.float64)
    up_w = np.asarray(yaw_retargeted["up_w"], dtype=np.float64).reshape(3)
    T_obj0 = np.asarray(yaw_retargeted["T_obj0_w"], dtype=np.float64)
    z_floor = float((invert(T_floor_w) @ T_obj0)[2, 3])
    K = np.asarray(fr["K"], dtype=np.float64)
    T_floor_in_cam = invert(T_cam_w) @ T_floor_w
    up_cam = T_cam_w[:3, :3].T @ up_w

    img = cv2.imread(str(fr["rgb_path"]))
    if img is None:
        img = np.zeros((int(fr["h"]), int(fr["w"]), 3), dtype=np.uint8)
    vis = img.copy()
    _draw_horizon_on_rgb(vis, K, T_cam_w, T_floor_w, up_w, z_floor)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12.4, 8.6), dpi=150)
    ax_img = fig.add_subplot(2, 2, 1)
    ax3 = fig.add_subplot(2, 2, 2, projection="3d")
    ax_side = fig.add_subplot(2, 2, 3, projection="3d")
    ax_leg = fig.add_subplot(2, 2, 4)
    ax_leg.axis("off")

    ax_img.imshow(vis_rgb)
    ax_img.set_title(f"RGB  idx {fr['idx']:04d}  ·  cyan grid = new horizontal  ·  yellow = horizon")
    ax_img.axis("off")

    grid = _floor_grid_in_cam(T_floor_in_cam, z_floor, half=0.5, step=0.1)
    all_pts = [np.zeros((1, 3))]
    for a, b in grid:
        ax3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#27ae60", lw=0.8, alpha=0.85)
        ax_side.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#27ae60", lw=0.8, alpha=0.85)
        all_pts.append(a.reshape(1, 3))
        all_pts.append(b.reshape(1, 3))

    ax3.quiver(0, 0, 0, 0.12, 0, 0, color="#c0392b", lw=2.0, label="cam X right")
    ax3.quiver(0, 0, 0, 0, 0.12, 0, color="#1abc9c", lw=2.0, label="cam Y down")
    ax3.quiver(0, 0, 0, 0, 0, 0.12, color="#2980b9", lw=2.0, label="cam Z forward")
    ax3.quiver(0, 0, 0, *(0.18 * up_cam), color="#f1c40f", lw=2.4, label="gravity up")

    R_f = T_floor_in_cam[:3, :3]
    t_f = (T_floor_in_cam @ np.array([0.0, 0.0, z_floor, 1.0]))[:3]
    ax3.quiver(t_f[0], t_f[1], t_f[2], *(0.14 * R_f[:, 0]), color="#e74c3c", lw=1.8)
    ax3.quiver(t_f[0], t_f[1], t_f[2], *(0.14 * R_f[:, 1]), color="#2ecc71", lw=1.8)
    ax3.quiver(t_f[0], t_f[1], t_f[2], *(0.14 * R_f[:, 2]), color="#3498db", lw=1.8)
    all_pts.append(t_f.reshape(1, 3))

    fr_pts, fr_edges = camera_frustum_local(0.04)
    ax3.add_collection3d(_polyline_collection(fr_pts, fr_edges, "#8e44ad", 1.0))
    ax_side.add_collection3d(_polyline_collection(fr_pts, fr_edges, "#8e44ad", 1.0))
    ax_side.quiver(0, 0, 0, *(0.18 * up_cam), color="#f1c40f", lw=2.4)
    ax_side.quiver(0, 0, 0, 0, 0.12, 0, color="#1abc9c", lw=2.0, label="cam Y down")
    ax_side.quiver(0, 0, 0, 0, 0, 0.12, color="#2980b9", lw=2.0, label="cam Z fwd")

    pts = np.concatenate(all_pts, axis=0)
    _set_equal_3d(ax3, pts, pad=0.12)
    _set_equal_3d(ax_side, pts, pad=0.12)
    ax3.set_xlabel("X_cam right (m)")
    ax3.set_ylabel("Y_cam down (m)")
    ax3.set_zlabel("Z_cam forward (m)")
    ax3.set_title("New horizontal in camera coords")
    ax3.view_init(elev=-18, azim=-70)
    ax3.legend(loc="upper left", fontsize=7)
    ax_side.set_xlabel("X_cam")
    ax_side.set_ylabel("Y_cam down")
    ax_side.set_zlabel("Z_cam fwd")
    ax_side.set_title("Side: OpenCV Y-down vs gravity-up")
    ax_side.view_init(elev=0, azim=0)

    ax_leg.text(
        0.02, 0.95,
        "New horizontal in camera frame\n"
        "  OpenCV cam0:  +X right,  +Y down,  +Z forward\n"
        "  New level:    Xh = image right on the floor\n"
        "                Yh = forward on the floor\n"
        "                Zh = gravity up\n"
        f"  Table at Zh = {z_floor:+.3f} m  (object height vs cam0)\n"
        "  Yellow line = horizon (vanishing line of Zh=const)\n"
        "  Cyan grid   = that new horizontal, at table height",
        va="top", ha="left", fontsize=10, family="monospace",
        transform=ax_leg.transAxes,
    )
    fig.suptitle("Camera-frame gravity horizontal", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_horizon_overlay_video(yaw_retargeted: dict[str, Any], out_path: Path, fps: float | None = None) -> int:
    """Project the new horizontal onto every RGB frame."""
    frames = yaw_retargeted["frames"]
    T_floor_w = np.asarray(yaw_retargeted["T_floor_w"], dtype=np.float64)
    up_w = np.asarray(yaw_retargeted["up_w"], dtype=np.float64).reshape(3)
    T_obj0 = np.asarray(yaw_retargeted["T_obj0_w"], dtype=np.float64)
    z_floor = float((invert(T_floor_w) @ T_obj0)[2, 3])
    fps = float(fps or yaw_retargeted.get("fps") or 15.0)
    sample = cv2.imread(str(frames[0]["rgb_path"]))
    h, w = sample.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for fr in frames:
        img = cv2.imread(str(fr["rgb_path"]))
        if img is None:
            continue
        T_cam_w = np.asarray(fr.get("T_cam_virt", fr["T_cam_w"]), dtype=np.float64)
        _draw_horizon_on_rgb(img, fr["K"], T_cam_w, T_floor_w, up_w, z_floor)
        cv2.putText(
            img,
            f"new horizontal  idx {fr['idx']:04d}  yaw {np.rad2deg(fr['yaw']):+.1f} deg",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
        n += 1
    writer.release()
    return n


def _shadow_text(img, text, org, scale, color, thick=1) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _xz_mapper(x0: int, y0: int, size: int, cx: float, cz: float, half: float):
    scale = (size * 0.44) / max(half, 1e-3)
    px_cx = x0 + size * 0.5
    px_cy = y0 + size * 0.5

    def px(x: float, z: float) -> tuple[int, int]:
        return int(round(px_cx + (float(x) - cx) * scale)), int(round(px_cy - (float(z) - cz) * scale))

    return px, scale


def _draw_xz_grid(img, px, cx: float, cz: float, half: float, step: float) -> None:
    x0, x1 = cx - half, cx + half
    z0, z1 = cz - half, cz + half
    i0, i1 = int(np.floor(x0 / step)), int(np.ceil(x1 / step))
    j0, j1 = int(np.floor(z0 / step)), int(np.ceil(z1 / step))
    for i in range(i0, i1 + 1):
        v = i * step
        col = (110, 110, 120) if abs(v) < 1e-9 else (70, 70, 78)
        cv2.line(img, px(v, z0), px(v, z1), col, 1, cv2.LINE_AA)
    for j in range(j0, j1 + 1):
        v = j * step
        col = (110, 110, 120) if abs(v) < 1e-9 else (70, 70, 78)
        cv2.line(img, px(x0, v), px(x1, v), col, 1, cv2.LINE_AA)


def _draw_trail(img, pts, px, color, thickness=2) -> None:
    run: list[tuple[int, int]] = []
    for p in pts:
        if p is None:
            if len(run) >= 2:
                cv2.polylines(img, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)
            run = []
            continue
        run.append(px(p[0], p[1]))
    if len(run) >= 2:
        cv2.polylines(img, [np.array(run, np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_gripper_xz(img, T: np.ndarray, grasp: float, px, color) -> None:
    pts, edges = gripper_wireframe(float(grasp or 0.0))
    pw = transform_points(T, pts)
    uv = [px(p[0], p[2]) for p in pw]
    for i, j in edges:
        cv2.line(img, uv[i], uv[j], color, 2, cv2.LINE_AA)
    cv2.circle(img, px(T[0, 3], T[2, 3]), 4, color, -1, cv2.LINE_AA)


def _draw_chassis_xz(img, R: np.ndarray, px, color, length=0.16) -> None:
    fwd = R[:3, 2]
    right = R[:3, 0]
    nose = px(length * fwd[0], length * fwd[2])
    left = px(-0.05 * right[0], -0.05 * right[2])
    right_pt = px(0.05 * right[0], 0.05 * right[2])
    origin = px(0.0, 0.0)
    cv2.line(img, left, right_pt, color, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, origin, nose, color, 2, cv2.LINE_AA, tipLength=0.28)


def write_mobile_map_video(
    yaw_retargeted: dict[str, Any],
    out_path: Path,
    fps: float | None = None,
) -> int:
    """Camera-frame mobile-manipulation map used as the retarget QA video.

    Left: measured RGB, hand in the real camera.
    Middle: OpenCV cam0 after the mapping. Chassis stays at the origin and
    only yaws; orange is y_action (wiped), gray is the hand before the wipe.
    Right: arm command in the chassis, T_ee_in_base. The object swings in
    this frame as the base yaws.
    Horizontal plane of the maps is camera X (right) and Z (forward).
    Gravity is camera -Y, so that plane is the floor.
    """
    frames = yaw_retargeted["frames"]
    if not frames:
        return 0
    T_obj = np.asarray(next(iter(yaw_retargeted["obj_in_ref"].values())), dtype=np.float64)
    T_obj_w = np.asarray(yaw_retargeted["T_obj0_w"], dtype=np.float64)
    T_w2ref = invert(np.asarray(yaw_retargeted["T_ref_w"], dtype=np.float64))
    y_base = np.asarray(yaw_retargeted["y_base"], dtype=np.float64)
    yaw_deg = np.rad2deg(y_base[:, 2])
    radius_m = float(yaw_retargeted.get("radius_m") or 0.0)

    raw_xz: list[tuple[float, float] | None] = []
    replay_xz: list[tuple[float, float] | None] = []
    ee_xz: list[tuple[float, float] | None] = []
    obj_b_xz: list[tuple[float, float] | None] = []
    bound = [(0.0, 0.0), (float(T_obj[0, 3]), float(T_obj[2, 3]))]
    for fr in frames:
        if fr["T_hand_in_ref"] is None or fr["T_hand_in_cam"] is None:
            raw_xz.append(None)
            replay_xz.append(None)
            ee_xz.append(None)
            obj_b_xz.append(None)
            continue
        raw = T_w2ref @ fr["T_cam_w"] @ fr["T_hand_in_cam"]
        ee = fr["T_ee_in_base"]
        obj_b = invert(fr["T_cam_in_ref"]) @ T_obj
        raw_xz.append((float(raw[0, 3]), float(raw[2, 3])))
        replay_xz.append((float(fr["T_hand_in_ref"][0, 3]), float(fr["T_hand_in_ref"][2, 3])))
        ee_xz.append((float(ee[0, 3]), float(ee[2, 3])))
        obj_b_xz.append((float(obj_b[0, 3]), float(obj_b[2, 3])))
        bound.extend([raw_xz[-1], replay_xz[-1], ee_xz[-1], obj_b_xz[-1]])
    arr = np.asarray(bound, dtype=np.float64)
    pad = 0.08
    x_min = min(float(arr[:, 0].min()) - pad, -0.2)
    x_max = max(float(arr[:, 0].max()) + pad, 0.2)
    z_min = min(float(arr[:, 1].min()) - pad, -0.12)
    z_max = max(float(arr[:, 1].max()) + pad, 0.2)
    cx = 0.5 * (x_min + x_max)
    cz = 0.5 * (z_min + z_max)
    half = 0.5 * max(x_max - x_min, z_max - z_min)
    step = 0.2 if half > 0.55 else 0.1

    sample = cv2.imread(str(frames[0]["rgb_path"]))
    if sample is None:
        body_h, rgb_w = int(frames[0]["h"]), int(frames[0]["w"])
    else:
        body_h, rgb_w = sample.shape[:2]
    map_w = body_h
    header_h, footer_h = 36, 112
    canvas_h = header_h + body_h + footer_h
    canvas_w = rgb_w + 2 * map_w
    if canvas_w % 2:
        canvas_w += 1
    if canvas_h % 2:
        footer_h += 1
        canvas_h += 1
    fps = float(fps or yaw_retargeted.get("fps") or 15.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))

    y0 = header_h
    x_rgb, x_cam, x_arm = 0, rgb_w, rgb_w + map_w
    px_cam, _ = _xz_mapper(x_cam, y0, map_w, cx, cz, half)
    px_arm, _ = _xz_mapper(x_arm, y0, map_w, cx, cz, half)
    still_ids = {
        int(frames[0]["idx"]),
        int(frames[len(frames) // 2]["idx"]),
        int(frames[-1]["idx"]),
    }
    n = 0
    for i, fr in enumerate(frames):
        canvas = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (16, 16, 16), -1)
        _shadow_text(
            canvas,
            "camera-frame mobile map    T_hand_in_cam0 = yaw(base) @ T_ee_in_base    OpenCV  X right  Y down  Z forward",
            (12, 24),
            0.48,
            (235, 235, 235),
        )

        img = cv2.imread(str(fr["rgb_path"])) if fr.get("rgb_path") else None
        if img is None:
            img = np.zeros((body_h, rgb_w, 3), dtype=np.uint8)
        elif img.shape[0] != body_h or img.shape[1] != rgb_w:
            img = cv2.resize(img, (rgb_w, body_h), interpolation=cv2.INTER_AREA)
        if fr["T_hand_in_cam"] is not None:
            _draw_projected_gripper(img, fr["T_hand_in_cam"], fr["K"], float(fr["grasp"] or 0.0))
        uv_obj = _project_world_to_uv(fr["T_cam_w"], fr["K"], T_obj_w[:3, 3])
        _draw_uv_marker(img, uv_obj, (80, 200, 80), cv2.MARKER_CROSS, "obj")
        tag = "no hand" if fr["T_hand_in_cam"] is None else ("CLOSED" if (fr["grasp"] or 0) > 0.5 else "OPEN")
        _shadow_text(img, f"measured camera   idx {fr['idx']:04d}   {tag}", (10, 22), 0.5, (240, 240, 240))
        canvas[y0:y0 + body_h, x_rgb:x_rgb + rgb_w] = img

        for px, x_panel in ((px_cam, x_cam), (px_arm, x_arm)):
            _draw_xz_grid(canvas, px, cx, cz, half, step)
            cv2.arrowedLine(canvas, px(0, 0), px(0.16, 0), (40, 40, 220), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(canvas, px(0, 0), px(0, 0.16), (220, 140, 40), 2, cv2.LINE_AA, tipLength=0.25)
            _shadow_text(canvas, "X right    Z forward", (x_panel + 10, y0 + body_h - 36), 0.4, (180, 180, 180))

        _draw_trail(canvas, raw_xz[: i + 1], px_cam, (130, 130, 130), 1)
        _draw_trail(canvas, replay_xz[: i + 1], px_cam, (40, 150, 255), 2)
        obj_px = px_cam(T_obj[0, 3], T_obj[2, 3])
        cv2.drawMarker(canvas, obj_px, (80, 210, 80), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        _shadow_text(canvas, "obj", (obj_px[0] + 8, obj_px[1] - 8), 0.4, (80, 210, 80))
        for j in range(0, i + 1, 8):
            Rj = frames[j]["T_cam_in_ref"][:3, :3]
            _draw_chassis_xz(canvas, Rj, px_cam, (120, 70, 120), length=0.28)
        _draw_chassis_xz(canvas, fr["T_cam_in_ref"][:3, :3], px_cam, (230, 140, 230), length=0.38)
        if fr["T_hand_in_ref"] is not None:
            _draw_gripper_xz(canvas, fr["T_hand_in_ref"], float(fr["grasp"] or 0.0), px_cam, (40, 150, 255))
        _shadow_text(canvas, "cam0 replay    orange = y_action after wipe", (x_cam + 10, y0 + 22), 0.45, (240, 240, 240))
        _shadow_text(canvas, "gray = hand before wipe    green = obj", (x_cam + 10, y0 + 42), 0.42, (210, 210, 210))
        _shadow_text(
            canvas,
            f"chassis yaw {yaw_deg[i]:+.1f} deg   xy fixed   r={radius_m*100:.0f} cm wiped",
            (x_cam + 10, y0 + body_h - 16),
            0.42,
            (220, 180, 220),
        )

        _draw_chassis_xz(canvas, np.eye(3), px_arm, (220, 120, 220), length=0.18)
        _draw_trail(canvas, obj_b_xz[: i + 1], px_arm, (80, 180, 80), 2)
        _draw_trail(canvas, ee_xz[: i + 1], px_arm, (40, 150, 255), 2)
        if obj_b_xz[i] is not None:
            ob_px = px_arm(*obj_b_xz[i])
            cv2.drawMarker(canvas, ob_px, (80, 210, 80), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
            _shadow_text(canvas, "obj", (ob_px[0] + 8, ob_px[1] - 8), 0.4, (80, 210, 80))
        if fr["T_ee_in_base"] is not None:
            _draw_gripper_xz(canvas, fr["T_ee_in_base"], float(fr["grasp"] or 0.0), px_arm, (40, 150, 255))
        _shadow_text(canvas, "chassis frame    arm = T_ee_in_base", (x_arm + 10, y0 + 22), 0.45, (240, 240, 240))
        _shadow_text(canvas, "base fixed    obj swings in with yaw", (x_arm + 10, y0 + 42), 0.42, (210, 210, 210))
        if fr["T_ee_in_base"] is not None:
            ee = fr["T_ee_in_base"][:3, 3]
            _shadow_text(
                canvas,
                f"ee  X {ee[0]:+.2f}  Ydn {ee[1]:+.2f}  Z {ee[2]:+.2f} m",
                (x_arm + 10, y0 + body_h - 16),
                0.42,
                (180, 210, 255),
            )

        cv2.line(canvas, (x_cam, y0), (x_cam, y0 + body_h), (60, 60, 60), 1)
        cv2.line(canvas, (x_arm, y0), (x_arm, y0 + body_h), (60, 60, 60), 1)

        fy = y0 + body_h
        cv2.rectangle(canvas, (0, fy), (canvas_w, canvas_h), (16, 16, 16), -1)
        plot_x0, plot_x1 = 70, canvas_w - 16
        plot_y0, plot_y1 = fy + 18, fy + 78
        cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (80, 80, 80), 1)
        y_lo = float(min(0.0, yaw_deg.min()) - 2.0)
        y_hi = float(max(0.0, yaw_deg.max()) + 2.0)

        def _yaw_pt(k: int) -> tuple[int, int]:
            u = plot_x0 + (plot_x1 - plot_x0) * (k / max(len(frames) - 1, 1))
            v = plot_y1 - (plot_y1 - plot_y0) * ((yaw_deg[k] - y_lo) / max(y_hi - y_lo, 1e-3))
            return int(round(u)), int(round(v))

        cv2.polylines(canvas, [np.array([_yaw_pt(k) for k in range(len(frames))], np.int32)], False, (200, 120, 220), 2, cv2.LINE_AA)
        cv2.circle(canvas, _yaw_pt(i), 4, (40, 180, 255), -1, cv2.LINE_AA)
        _shadow_text(canvas, "yaw", (12, plot_y0 + 28), 0.45, (200, 160, 210))
        yb = y_base[i]
        shift = ""
        if raw_xz[i] is not None and replay_xz[i] is not None:
            d = float(np.hypot(raw_xz[i][0] - replay_xz[i][0], raw_xz[i][1] - replay_xz[i][1]))
            shift = f"   wipe shift xz {d*100:.0f} cm"
        _shadow_text(
            canvas,
            f"y_base  x {yb[0]:+.3f}  y {yb[1]:+.3f} m   yaw {yaw_deg[i]:+.1f} deg{shift}",
            (12, canvas_h - 16),
            0.48,
            (230, 230, 230),
        )
        writer.write(canvas)
        if int(fr["idx"]) in still_ids:
            cv2.imwrite(str(out_path.with_name(f"mobile_map_camera_{int(fr['idx']):04d}.png")), canvas)
        n += 1
    writer.release()
    return n

