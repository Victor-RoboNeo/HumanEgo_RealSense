# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from utils.utils_io import load_cfg


def _cfg_get(cfg, name, default=None):
    try:
        value = getattr(cfg, name)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        return cfg.get(name, default)
    except Exception:
        return default


def _chronological_unique(image_paths):
    by_idx = {}
    for path in image_paths:
        p = Path(path).expanduser().absolute()
        try:
            idx = int(p.parent.name)
        except ValueError as exc:
            raise ValueError(f"SAM31 expects numeric HumanEgo frame dirs: {p}") from exc
        if idx in by_idx and by_idx[idx] != str(p):
            raise RuntimeError(f"Duplicate HumanEgo frame id {idx}")
        by_idx[idx] = str(p)
    return [by_idx[i] for i in sorted(by_idx)]


def _mask_prompts(cfg):
    """All DINOSAM text prompts: obj*, arm, and any other non-empty keys."""
    prompts = dict(cfg.dinosam_prompt)
    out = {}
    for key, value in prompts.items():
        key = str(key)
        value = str(value).strip()
        if not value or value.lower() in {"none", "null"}:
            continue
        out[key] = value
    return out


def _rebuild_combined_masks(image_paths, mask_keys):
    for img_path in image_paths:
        frame_dir = Path(img_path).parent
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        for key in mask_keys:
            p = frame_dir / f"mask_{key}.png"
            if not p.is_file():
                continue
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = cv2.bitwise_or(
                combined, (m > 0).astype(np.uint8) * 255
            )
        cv2.imwrite(str(frame_dir / "mask_arm_and_obj.png"), combined)


def run_sam31_object_masks(*, image_paths, cfg_path, mps_path, reference_image_path):
    """All HumanEgo masks (obj* and arm) from SAM3.1 text + two-pass video tracking."""
    image_paths = _chronological_unique(image_paths)
    if not image_paths:
        raise ValueError("[SAM31] empty image path list")

    reference = str(Path(reference_image_path).expanduser().absolute())
    if reference not in image_paths:
        raise RuntimeError(
            "[SAM31] object_ref_img_path is not in the object-processing window: "
            f"{reference}"
        )

    cfg = load_cfg(cfg_path)
    objects = _mask_prompts(cfg)
    if not objects:
        raise RuntimeError("[SAM31] no non-empty prompts in DINOSAM config")

    threshold = float(_cfg_get(cfg, "sam31_threshold", 0.50))
    save_debug_video = bool(_cfg_get(cfg, "sam31_save_debug_video", True))
    fps = float(_cfg_get(cfg, "fps", 15.0))

    session_root = Path(mps_path).expanduser().absolute()
    preprocess_dir = session_root / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    request_path = preprocess_dir / ".sam31_v2_request.json"
    request = {
        "image_paths": image_paths,
        "reference_image_path": reference,
        "objects": objects,
        "threshold": threshold,
        "fps": fps,
        "save_debug_video": save_debug_video,
        "output_root": str(preprocess_dir),
    }
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    humanego_root = Path(__file__).absolute().parents[2]
    launcher = humanego_root / "preprocess" / "sam31" / "run_sam31.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"[SAM31] launcher missing: {launcher}")

    print(
        f"║ [SAM31-v2] frames={len(image_paths)}, "
        f"ref={Path(reference).parent.name}, objects={list(objects)}"
    )

    completed = subprocess.run(
        [str(launcher), "--request", str(request_path)],
        cwd=str(humanego_root),
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"[SAM31-v2] uv runtime failed with exit code {completed.returncode}"
        )

    report_path = preprocess_dir / ".sam31_v2" / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"[SAM31-v2] missing runtime report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for obj_key, info in report.get("objects", {}).items():
        print(
            f"║ [SAM31-v2] {obj_key}: "
            f"nonempty={info.get('nonempty_frames', 0)}/{len(image_paths)}"
        )
    _rebuild_combined_masks(image_paths, objects.keys())
    return report
