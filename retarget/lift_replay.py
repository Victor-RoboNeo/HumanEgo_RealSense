# -*- coding: utf-8 -*-
"""Replay a camera-frame trajectory on the ARX LIFT model.

Base yaw follows ``y_base``. The camera stays fixed on the head, so the
right-arm target is ``T_cam_in_base @ T_hand_in_cam``. Lift and head joints
stay at zero: lift does not change the target relative to the camera.

    python -m retarget.lift_replay --session HumanEgo_data/strawberry/session_20260920_163411
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retarget.geometry import invert, project_cam
from retarget.lift_model import LiftModel, mean_rotation, pose_error, solve_arm
from retarget.mask_anchor import estimate_mask_anchor
from retarget.session import load_session
from retarget.visualize import _draw_projected_gripper, _project_world_to_uv, _shadow_text
from retarget.vio_pose import apply_vio_c2w
from retarget.yaw_world import retarget_yaw_world

_CHAIN = [
    ("wheel_1", "wheel_2"),
    ("wheel_2", "wheel_3"),
    ("wheel_3", "wheel_1"),
    ("lift_1", "h_base_link"),
    ("h_base_link", "h_link1_1"),
    ("h_link1_1", "h_link2_1"),
    ("lift_1", "right_arm_base_link"),
    ("right_arm_base_link", "right_arm_link1"),
    ("right_arm_link1", "right_arm_link2"),
    ("right_arm_link2", "right_arm_link3"),
    ("right_arm_link3", "right_arm_link4"),
    ("right_arm_link4", "right_arm_link5"),
    ("right_arm_link5", "right_arm_link6"),
    ("right_arm_link6", "right_arm_left_catch"),
    ("right_arm_link6", "right_arm_right_catch"),
]


def _solve_trajectory(model: LiftModel, yaw_rt: dict, lift: float) -> dict:
    """Position IK, then one fixed pinch-to-gripper rotation, then a light orientation pass."""
    T_cam_base = model.camera_in_base(lift)
    q = np.zeros(6)
    posed = []
    for fr in yaw_rt["frames"]:
        if fr["T_hand_in_cam"] is None:
            posed.append(None)
            continue
        target = T_cam_base @ fr["T_hand_in_cam"]
        q = solve_arm(model, target, q, lift=lift, rot_weight=0.0, iters=30)
        posed.append((q.copy(), target, model.tool_in_base(q, lift)))
    rel = [achieved[:3, :3].T @ target[:3, :3] for _, target, achieved in (item for item in posed if item is not None)]
    align = mean_rotation(rel)
    model.T_tool_in_link6[:3, :3] = model.T_tool_in_link6[:3, :3] @ align
    q = None
    rows = []
    for item in posed:
        if item is None:
            rows.append(None)
            continue
        q_seed, target, _ = item
        q = solve_arm(model, target, q if q is not None else q_seed, lift=lift, rot_weight=0.04, iters=25)
        achieved = model.tool_in_base(q, lift)
        err = pose_error(achieved, target, rot_weight=1.0)
        rows.append(
            {
                "q": q.copy(),
                "target": target,
                "achieved": achieved,
                "pos_m": float(np.linalg.norm(err[:3])),
                "rot_deg": float(np.rad2deg(np.linalg.norm(err[3:]))),
            }
        )
    return {"T_cam_in_base": T_cam_base, "rows": rows, "lift": lift, "R_tool_align": align}


def _mapper(x0, y0, size, cx, cz, half):
    scale = (size * 0.46) / max(half, 1e-3)
    px = x0 + size * 0.5
    py = y0 + size * 0.5

    def map_pt(a, b):
        return int(round(px + (a - cx) * scale)), int(round(py - (b - cz) * scale))

    return map_pt


def _draw_view(img, pts_ab, segments, mapper, color, thickness=2):
    for i, j in segments:
        cv2.line(img, mapper(*pts_ab[i]), mapper(*pts_ab[j]), color, thickness, cv2.LINE_AA)


def write_lift_replay_video(yaw_rt: dict, solved: dict, out_path: Path, fps: float = 15.0) -> int:
    model_pts_cache = None
    model = LiftModel()
    frames = yaw_rt["frames"]
    T_cam_base = solved["T_cam_in_base"]
    T_obj = next(iter(yaw_rt["obj_in_ref"].values()))
    T_obj_w = np.asarray(yaw_rt["T_obj0_w"])
    T_cam0 = np.asarray(yaw_rt["T_cam0_w"])
    sample = cv2.imread(str(frames[0]["rgb_path"]))
    body_h, rgb_w = sample.shape[:2]
    map_w = body_h
    header, footer = 36, 64
    canvas_w = rgb_w + 2 * map_w
    canvas_h = header + body_h + footer
    if canvas_w % 2:
        canvas_w += 1
    if canvas_h % 2:
        canvas_h += 1

    # Bounds in camera XZ and in (forward Z, up).
    bound_xz, bound_zu = [], []

    def accumulate(p):
        bound_xz.append((p[0], p[2]))
        bound_zu.append((p[2], -p[1]))

    for fr, row in zip(frames, solved["rows"]):
        accumulate(T_obj[:3, 3])
        if fr["T_hand_in_ref"] is not None:
            accumulate(fr["T_hand_in_ref"][:3, 3])
        if row is None:
            continue
        T_base = fr["T_cam_in_ref"] @ invert(T_cam_base)
        pts = model.link_points(row["q"], solved["lift"], fr["grasp"] or 0.0)
        for p in pts.values():
            accumulate((T_base @ np.append(p, 1.0))[:3])
        accumulate((T_base @ np.append(row["achieved"][:3, 3], 1.0))[:3])
    arr_xz = np.asarray(bound_xz)
    arr_zu = np.asarray(bound_zu)

    def center_half(arr):
        pad = 0.08
        lo = arr.min(axis=0) - pad
        hi = arr.max(axis=0) + pad
        c = 0.5 * (lo + hi)
        half = 0.5 * float(np.max(hi - lo))
        return c, max(half, 0.3)

    c_xz, h_xz = center_half(arr_xz)
    c_zu, h_zu = center_half(arr_zu)
    y0 = header
    map_xz = _mapper(rgb_w, y0, map_w, c_xz[0], c_xz[1], h_xz)
    map_zu = _mapper(rgb_w + map_w, y0, map_w, c_zu[0], c_zu[1], h_zu)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = str(yaw_rt.get("pose_source", "icp"))
    writer = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{canvas_w}x{canvas_h}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-crf", "18",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    still = {int(frames[0]["idx"]), int(frames[len(frames) // 2]["idx"]), int(frames[int(np.argmax(np.abs(yaw_rt["yaw"])))]["idx"])}
    n = 0
    chain_idx = {name: i for i, name in enumerate(
        ["wheel_1", "wheel_2", "wheel_3", "lift_1", "h_base_link", "h_link1_1", "h_link2_1",
         "right_arm_base_link", "right_arm_link1", "right_arm_link2", "right_arm_link3",
         "right_arm_link4", "right_arm_link5", "right_arm_link6", "right_arm_left_catch", "right_arm_right_catch"]
    )}
    edges = [(chain_idx[a], chain_idx[b]) for a, b in _CHAIN]
    for fr, row in zip(frames, solved["rows"]):
        canvas = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (canvas_w, header), (16, 16, 16), -1)
        _shadow_text(canvas, f"LIFT replay {source}    base yaw    camera fixed on head    right arm -> hand in camera", (12, 24), 0.48, (235, 235, 235))
        img = cv2.imread(str(fr["rgb_path"]))
        if img is None:
            img = np.zeros((body_h, rgb_w, 3), dtype=np.uint8)
        elif img.shape[:2] != (body_h, rgb_w):
            img = cv2.resize(img, (rgb_w, body_h))
        if fr["T_hand_in_cam"] is not None:
            _draw_projected_gripper(img, fr["T_hand_in_cam"], fr["K"], float(fr["grasp"] or 0.0))
        uv_obj = _project_world_to_uv(fr["T_cam_w"], fr["K"], T_obj_w[:3, 3])
        if uv_obj is not None:
            cv2.drawMarker(img, uv_obj, (80, 200, 80), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        if row is not None:
            T_base = fr["T_cam_in_ref"] @ invert(T_cam_base)
            T_ach_cam0 = T_base @ row["achieved"]
            T_ach_w = T_cam0 @ T_ach_cam0
            p_real = invert(fr["T_cam_w"]) @ T_ach_w
            uv, valid = project_cam(fr["K"], p_real[:3, 3].reshape(1, 3))
            if valid[0]:
                cv2.circle(img, (int(round(uv[0, 0])), int(round(uv[0, 1]))), 6, (255, 180, 40), 2, cv2.LINE_AA)
        tag = "no hand" if row is None else f"arm {row['pos_m']*100:.1f} cm"
        _shadow_text(img, f"measured   idx {fr['idx']:04d}   {tag}", (10, 22), 0.5, (240, 240, 240))
        canvas[y0:y0 + body_h, :rgb_w] = img

        for mapper, kind in ((map_xz, "xz"), (map_zu, "zu")):
            if row is None:
                continue
            T_base = fr["T_cam_in_ref"] @ invert(T_cam_base)
            pts = model.link_points(row["q"], solved["lift"], fr["grasp"] or 0.0)
            order = list(chain_idx.keys())
            xyz = []
            for name in order:
                p = (T_base @ np.append(pts[name], 1.0))[:3]
                xyz.append((p[0], p[2]) if kind == "xz" else (p[2], -p[1]))
            color = (180, 160, 210) if kind == "xz" else (180, 180, 160)
            _draw_view(canvas, xyz, edges, mapper, color)
            hand = fr["T_hand_in_ref"]
            if hand is not None:
                hp = (hand[0, 3], hand[2, 3]) if kind == "xz" else (hand[2, 3], -hand[1, 3])
                cv2.drawMarker(canvas, mapper(*hp), (40, 150, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            ap = T_ach_cam0[:3, 3]
            app = (ap[0], ap[2]) if kind == "xz" else (ap[2], -ap[1])
            cv2.circle(canvas, mapper(*app), 5, (255, 200, 40), -1, cv2.LINE_AA)
            op = T_obj[:3, 3]
            opp = (op[0], op[2]) if kind == "xz" else (op[2], -op[1])
            cv2.drawMarker(canvas, mapper(*opp), (80, 210, 80), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        _shadow_text(canvas, "cam0   X right  Z forward    orange target  cyan tool", (rgb_w + 8, y0 + 20), 0.4, (220, 220, 220))
        _shadow_text(canvas, "side   Z forward  up    cyan = LIFT tool", (rgb_w + map_w + 8, y0 + 20), 0.4, (220, 220, 220))
        yaw = float(np.rad2deg(fr["yaw"]))
        extra = ""
        if row is not None:
            extra = f"   tool err {row['pos_m']*100:.1f} cm  {row['rot_deg']:.0f} deg"
        cv2.rectangle(canvas, (0, y0 + body_h), (canvas_w, canvas_h), (16, 16, 16), -1)
        _shadow_text(
            canvas,
            f"y_base yaw {yaw:+.1f} deg   xy 0   lift {solved['lift']:.2f} m   head joints 0{extra}",
            (12, canvas_h - 24),
            0.5,
            (230, 230, 230),
        )
        writer.stdin.write(np.ascontiguousarray(canvas).tobytes())
        if int(fr["idx"]) in still:
            cv2.imwrite(str(out_path.with_name(f"lift_replay_{int(fr['idx']):04d}.png")), canvas)
        n += 1
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode {out_path.name}")
    return n


def replay_session(session_path: Path, out_dir: Path, fps: float = 15.0, pose: str = "vio") -> dict:
    session = load_session(session_path, hand_side="right")
    if pose == "vio":
        apply_vio_c2w(session)
    elif pose != "icp":
        raise ValueError(f"unknown pose source {pose}")
    mask = estimate_mask_anchor(session, n_init=10)
    yaw_rt = retarget_yaw_world(session, T_obj0_w=mask["T_mask_w"])
    yaw_rt["pose_source"] = pose
    model = LiftModel()
    solved = _solve_trajectory(model, yaw_rt, lift=0.0)
    # Camera on the robot must land on the trajectory camera.
    T_cam_base = solved["T_cam_in_base"]
    cam_err = []
    for fr in yaw_rt["frames"]:
        T_base = fr["T_cam_in_ref"] @ invert(T_cam_base)
        cam_err.append(np.linalg.norm((T_base @ T_cam_base - fr["T_cam_in_ref"])))
    pos = [row["pos_m"] for row in solved["rows"] if row is not None]
    rot = [row["rot_deg"] for row in solved["rows"] if row is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "lift_replay.mp4"
    n = write_lift_replay_video(yaw_rt, solved, video, fps=fps)
    qs = np.full((len(solved["rows"]), 6), np.nan)
    pos_a = np.full((len(solved["rows"]),), np.nan)
    for i, row in enumerate(solved["rows"]):
        if row is None:
            continue
        qs[i] = row["q"]
        pos_a[i] = row["pos_m"]
    np.savez_compressed(
        out_dir / "lift_replay.npz",
        arm_q=qs,
        pos_err_m=pos_a,
        yaw=np.asarray(yaw_rt["yaw"]),
        lift=np.array([solved["lift"]]),
        T_cam_in_base=T_cam_base,
        R_tool_align=solved["R_tool_align"],
        pose_source=np.array(pose),
    )
    stats = {
        "frames": n,
        "with_hand": len(pos),
        "pos_med_cm": float(np.median(pos) * 100),
        "pos_max_cm": float(np.max(pos) * 100),
        "rot_med_deg": float(np.median(rot)),
        "rot_max_deg": float(np.max(rot)),
        "cam_close": float(np.max(cam_err)),
        "video": str(video),
    }
    print(
        f"[lift] pose {pose}  yaw [{np.rad2deg(yaw_rt['yaw']).min():+.1f}, {np.rad2deg(yaw_rt['yaw']).max():+.1f}] deg  "
        f"hand frames {stats['with_hand']}/{stats['frames']}  "
        f"tool pos median {stats['pos_med_cm']:.1f} cm  max {stats['pos_max_cm']:.1f} cm  "
        f"rot median {stats['rot_med_deg']:.0f} deg  max {stats['rot_max_deg']:.0f} deg"
    )
    print(f"[lift] camera-on-base matches trajectory camera, max |T| {stats['cam_close']:.2e}")
    print(f"[lift] wrote {video}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a camera-frame trajectory on ARX LIFT.")
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
    replay_session(session, out, fps=args.fps, pose=args.pose)


if __name__ == "__main__":
    main()
