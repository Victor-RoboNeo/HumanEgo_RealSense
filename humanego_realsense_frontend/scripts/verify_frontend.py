#!/usr/bin/env python3
"""Verify that a RealSense frontend session satisfies HumanEgo's file contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAM_REQUIRED = {
    "idx", "ts", "fov", "h", "w", "k", "d", "c2w", "c2d", "d2w", "rgb_path", "fps"
}
SLAM_REQUIRED = {
    "idx", "ts", "t_world", "rpy_deg", "delta_t_world",
    "delta_rpy_deg", "linear_speed_mps", "angular_speed_rps",
    "yaw_unwrapped_deg",
}
PHASE_REQUIRED = {
    "idx", "ts", "stop", "mode", "mode_str",
    "linear_speed_mps", "angular_speed_rps", "yaw_unwrapped_deg",
}


def check_keys(data, required, path):
    missing = required - set(data.keys())
    if missing:
        raise AssertionError(f"{path}: missing fields {sorted(missing)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    session = Path(args.session).resolve()
    all_data = session / "preprocess" / "all_data"

    if not all_data.is_dir():
        raise SystemExit(f"missing {all_data}")

    dirs = sorted(x for x in all_data.iterdir() if x.is_dir())
    if args.max_frames > 0:
        dirs = dirs[: args.max_frames]

    if not dirs:
        raise SystemExit("no per-frame directories")

    bad = 0
    hands_with_pose = 0

    for d in dirs:
        try:
            cam_p = d / "aria_cam_rgb.json"
            slam_p = d / "aria_slam.json"
            phase_p = d / "aria_phases.json"
            hands_p = d / "aria_hands.json"
            rgb_p = d / "rgb.png"

            for pth in (cam_p, slam_p, phase_p, hands_p, rgb_p):
                if not pth.exists():
                    raise AssertionError(f"missing {pth}")

            cam = json.loads(cam_p.read_text())
            slam = json.loads(slam_p.read_text())
            phase = json.loads(phase_p.read_text())
            hands = json.loads(hands_p.read_text())

            check_keys(cam, CAM_REQUIRED, cam_p)
            check_keys(slam, SLAM_REQUIRED, slam_p)
            check_keys(phase, PHASE_REQUIRED, phase_p)

            c2w = np.asarray(cam["c2w"], dtype=float)
            c2d = np.asarray(cam["c2d"], dtype=float)
            d2w = np.asarray(cam["d2w"], dtype=float)
            if not np.allclose(c2w, d2w @ c2d, atol=1e-5):
                raise AssertionError(f"{cam_p}: c2w != d2w @ c2d")

            K = np.asarray(cam["k"], dtype=float)
            if K.shape != (3, 3) or c2w.shape != (4, 4):
                raise AssertionError(f"{cam_p}: bad matrix shape")

            # HumanEgo AriaHands serializer uses right/left hand dictionaries.
            for key in ("hand_r", "hand_l", "right", "left"):
                h = hands.get(key)
                if isinstance(h, dict):
                    if (
                        h.get("midpoint_translation_opt_world") is not None
                        and h.get("midpoint_orientation_opt_world") is not None
                    ):
                        hands_with_pose += 1

        except Exception as exc:
            bad += 1
            print("[FAIL]", d.name, exc)

    phases_summary = session / "preprocess" / "aria_phases_results.json"
    if not phases_summary.is_file():
        bad += 1
        print("[FAIL] missing", phases_summary)
    else:
        summary = json.loads(phases_summary.read_text())
        if "stage_window_check" not in summary:
            bad += 1
            print("[FAIL] aria_phases_results.json missing stage_window_check")

    print()
    print("frames checked       :", len(dirs))
    print("frames failed        :", bad)
    print("hand optimized poses :", hands_with_pose)
    print("RESULT               :", "PASS" if bad == 0 else "FAIL")

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
