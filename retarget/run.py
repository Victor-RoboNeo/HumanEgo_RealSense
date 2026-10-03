# -*- coding: utf-8 -*-
"""Run object-centric retarget + visualization on a preprocess session.

Example:
    python -m retarget.run \\
        --session HumanEgo_data/strawberry/session_20260920_163411
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python retarget/run.py` from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retarget.core import retarget_frames
from retarget.geometry import TROSSEN_T_ALIGN
from retarget.mask_anchor import estimate_mask_anchor
from retarget.session import load_session
from retarget.visualize import (
    save_anchor_compare_figure,
    save_cam_horizon_figure,
    save_compensation_figure,
    save_object_frame_figure,
    save_poses_npz,
    save_yaw_world_figure,
    write_horizon_overlay_video,
    write_mobile_map_video,
    write_overlay_video,
    write_topdown_video,
)
from retarget.yaw_world import retarget_yaw_world

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retarget HumanEgo gripper/camera into the camera frame.")
    p.add_argument(
        "--session",
        type=str,
        default="HumanEgo_data/strawberry/session_20260920_163411",
        help="Session directory that contains preprocess/all_data",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: retarget/outputs/<session_name>)",
    )
    p.add_argument("--hand-side", type=str, default="right", choices=["right", "left"])
    p.add_argument(
        "--frame-mode",
        type=str,
        default="camera_frame",
        choices=["camera_frame", "anchor_frame"],
        help="Shared ref for y_action. Paper default: camera_frame.",
    )
    p.add_argument(
        "--t-align",
        type=str,
        default="identity",
        choices=["identity", "trossen"],
        help="identity keeps the preprocess pinch frame; trossen maps to the example EE.",
    )
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument(
        "--mask-init-frames",
        type=int,
        default=10,
        help="How many early mask+depth centroids to median for the corrected origin.",
    )
    p.add_argument(
        "--up",
        type=str,
        default="camera-neg-y",
        choices=["camera-neg-y", "world-z"],
        help="Gravity in cam0. Down-looking RealSense: camera-neg-y.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    session_path = Path(args.session)
    if not session_path.is_absolute():
        session_path = (_ROOT / session_path).resolve()

    out_dir = Path(args.out) if args.out else _ROOT / "retarget" / "outputs" / session_path.name
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    T_align = TROSSEN_T_ALIGN if args.t_align == "trossen" else None
    session = load_session(session_path, hand_side=args.hand_side)
    n_hand = sum(1 for fr in session["frames"] if fr["T_hand_w"] is not None)
    print(f"[retarget] session={session_path}")
    print(f"[retarget] frames={len(session['frames'])}  with_hand={n_hand}  side={args.hand_side}")
    print(f"[retarget] frame_mode={args.frame_mode}  T_align={args.t_align}")

    retargeted = retarget_frames(
        session, T_align=T_align, frame_mode=args.frame_mode, up_mode=args.up
    )
    mask_info = estimate_mask_anchor(session, n_init=args.mask_init_frames)
    delta = mask_info["delta_w"]
    print(
        f"[retarget] mask+depth vs cotracker: "
        f"dt=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) m  "
        f"|dt|={np.linalg.norm(delta)*100:.1f} cm  "
        f"n_init={mask_info['n_init_used']}"
    )
    retargeted_mask = retarget_frames(
        session,
        T_align=T_align,
        T_anchor_w=mask_info["T_mask_w"],
        frame_mode="anchor_frame",
        up_mode=args.up,
    )
    yaw_rt = retarget_yaw_world(
        session,
        T_obj0_w=mask_info["T_mask_w"],
        T_align=T_align,
        up_mode=args.up,
    )
    pvt = yaw_rt["T_pivot_w"][:3, 3]
    xy_span = np.linalg.norm(yaw_rt["y_base"][:, :2] - yaw_rt["y_base"][0, :2], axis=1).max()
    print(
        f"[retarget] yaw-world wipe_radius={yaw_rt['wipe_radius']}  "
        f"r={yaw_rt['radius_m']*100:.1f} cm  "
        f"yaw=[{np.rad2deg(yaw_rt['yaw'].min()):+.1f}, {np.rad2deg(yaw_rt['yaw'].max()):+.1f}] deg  "
        f"y_base |Δxy|_max={xy_span*100:.2f} cm  "
        f"pivot=({pvt[0]:+.2f},{pvt[1]:+.2f},{pvt[2]:+.2f})"
    )
    mask_uv = {
        rec["idx"]: rec["uv"]
        for rec in mask_info["per_frame"]
        if rec.get("uv") is not None
    }

    fig3d = out_dir / "vis_object_frame.png"
    fig3d_mask = out_dir / "vis_object_frame_mask.png"
    fig_comp = out_dir / "vis_compensation.png"
    fig_anchor = out_dir / "vis_anchor_compare.png"
    npz_path = out_dir / "retargeted_poses.npz"
    overlay = out_dir / "overlay_gripper.mp4"
    topdown = out_dir / "topdown_object_xy.mp4"
    topdown_mask = out_dir / "topdown_object_xy_mask.mp4"
    topdown_yaw = out_dir / "topdown_object_xy_yaw.mp4"
    fig_yaw = out_dir / "vis_yaw_world.png"
    fig3d_yaw = out_dir / "vis_object_frame_yaw.png"
    fig_horizon = out_dir / "vis_cam_horizon.png"
    overlay_horizon = out_dir / "overlay_horizon.mp4"
    mobile_map = out_dir / "mobile_map_camera.mp4"
    npz_yaw = out_dir / "retargeted_poses_yaw.npz"

    save_object_frame_figure(retargeted, fig3d)
    print(f"[retarget] wrote {fig3d}")
    save_object_frame_figure(retargeted_mask, fig3d_mask)
    print(f"[retarget] wrote {fig3d_mask}  (origin = mask+depth)")
    save_compensation_figure(retargeted_mask, fig_comp)
    print(f"[retarget] wrote {fig_comp}")
    save_anchor_compare_figure(session, mask_info, fig_anchor, frame_idx=0)
    print(f"[retarget] wrote {fig_anchor}")
    save_poses_npz(retargeted, npz_path)
    print(f"[retarget] wrote {npz_path}  (paper camera_frame / JSON cam0)")

    n_ov = write_overlay_video(
        retargeted,
        overlay,
        fps=args.fps,
        T_mask_w=mask_info["T_mask_w"],
        mask_uv_by_idx=mask_uv,
    )
    print(f"[retarget] overlay frames={n_ov} -> {overlay}")
    n_td = write_topdown_video(retargeted, topdown, fps=args.fps)
    print(f"[retarget] topdown frames={n_td} -> {topdown}")
    n_tdm = write_topdown_video(retargeted_mask, topdown_mask, fps=args.fps)
    print(f"[retarget] topdown(mask) frames={n_tdm} -> {topdown_mask}")
    save_yaw_world_figure(yaw_rt, retargeted_mask, fig_yaw)
    print(f"[retarget] wrote {fig_yaw}")
    save_object_frame_figure(yaw_rt, fig3d_yaw)
    print(f"[retarget] wrote {fig3d_yaw}  (wipe radius, y_action in OpenCV cam0)")
    n_tdy = write_topdown_video(yaw_rt, topdown_yaw, fps=args.fps)
    print(f"[retarget] topdown(yaw) frames={n_tdy} -> {topdown_yaw}")
    save_poses_npz(yaw_rt, npz_yaw)
    print(f"[retarget] wrote {npz_yaw}")
    n_map = write_mobile_map_video(yaw_rt, mobile_map, fps=args.fps)
    print(f"[retarget] mobile map frames={n_map} -> {mobile_map}")
    save_cam_horizon_figure(yaw_rt, fig_horizon, frame_idx=0)
    print(f"[retarget] wrote {fig_horizon}  (new horizontal in camera coords)")
    n_hz = write_horizon_overlay_video(yaw_rt, overlay_horizon, fps=args.fps)
    print(f"[retarget] horizon overlay frames={n_hz} -> {overlay_horizon}")
    print(f"[retarget] done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
