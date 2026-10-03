# -*- coding: utf-8 -*-
"""Load per-frame training_data.json from a HumanEgo preprocess session."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from retarget.geometry import as_T


def _remap_obs_path(raw: str | None, frame_dir: Path) -> str | None:
    if not raw:
        return None
    name = Path(raw).name
    local = frame_dir / name
    if local.is_file():
        return str(local)
    p = Path(raw)
    if p.is_file():
        return str(p)
    return str(local)


def load_session(session_path: str | Path, hand_side: str = "right") -> dict[str, Any]:
    session_path = Path(session_path).resolve()
    all_data = session_path / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"missing {all_data}")

    frames: list[dict[str, Any]] = []
    for frame_dir in sorted(p for p in all_data.iterdir() if p.is_dir()):
        json_path = frame_dir / "training_data.json"
        if not json_path.is_file():
            continue
        with open(json_path, "r") as f:
            d = json.load(f)

        meta = d["metadata"]
        ents = d.get("entities") or {}
        hands = ents.get("hands") or {}
        objs = ents.get("objects") or {}
        obs = d.get("obs") or {}

        T_anchor_w = as_T(meta["world_transforms"]["virtual_static_anchor"])
        wt = meta.get("world_transforms") or {}
        T_cam0_meta = wt.get("cam0")
        T_cam_w = as_T(meta["c2w"])
        K = np.asarray(meta["k"], dtype=np.float64).reshape(3, 3)

        hand = hands.get(hand_side)
        T_hand_w = as_T(hand["T_hand_to_world"]) if hand else None
        grasp = float(hand["grasp"]) if hand else None

        obj_poses = {k: as_T(v["T_obj_to_world"]) for k, v in objs.items()}

        cam_json = frame_dir / "aria_cam_rgb.json"
        depth_scale = 0.001
        if cam_json.is_file():
            with open(cam_json, "r") as f:
                cam_meta = json.load(f)
            depth_scale = float(cam_meta.get("depth_scale_m_per_unit", 0.001))

        frames.append(
            {
                "idx": int(meta.get("idx", int(frame_dir.name))),
                "ts": meta.get("ts"),
                "fps": float(meta.get("fps", 30.0)),
                "w": int(meta.get("w", 640)),
                "h": int(meta.get("h", 480)),
                "K": K,
                "anchor_key": meta.get("anchor_key", "obj1"),
                "is_finished": float(meta.get("is_finished", 0.0)),
                "T_anchor_w": T_anchor_w,
                "T_cam0_w": as_T(T_cam0_meta) if T_cam0_meta is not None else T_cam_w,
                "T_cam_w": T_cam_w,
                "T_hand_w": T_hand_w,
                "grasp": grasp,
                "obj_poses_w": obj_poses,
                "rgb_path": _remap_obs_path(obs.get("rgb_path"), frame_dir),
                "mask_obj1_path": _remap_obs_path(
                    obs.get("mask_obj1_path") or str(frame_dir / "mask_obj1.png"),
                    frame_dir,
                ),
                "depth_path": _remap_obs_path(str(frame_dir / "depth.png"), frame_dir),
                "depth_scale": depth_scale,
                "frame_dir": str(frame_dir),
            }
        )

    if not frames:
        raise RuntimeError(f"no training_data.json under {all_data}")

    T_anchor_w = frames[0]["T_anchor_w"]
    T_cam0_w = frames[0].get("T_cam0_w", frames[0]["T_cam_w"])
    return {
        "session_path": str(session_path),
        "hand_side": hand_side,
        "anchor_key": frames[0]["anchor_key"],
        "T_anchor_w": T_anchor_w,
        "T_cam0_w": T_cam0_w,
        "fps": frames[0]["fps"],
        "frames": frames,
    }
