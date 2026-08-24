# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from utils.utils_io import load_cfg


def _largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m * 255
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def _quality(mask: np.ndarray, min_area_ratio=0.001, max_area_ratio=0.60) -> float:
    if mask is None:
        return -1.0
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return -1.0
    ratio = area / float(mask.size)
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return -1.0
    largest = _largest_component(mask)
    coherence = np.count_nonzero(largest) / max(area, 1)
    return float(coherence * np.sqrt(ratio))


def _prepare_jpeg_video(image_paths: List[str], cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    complete = all((cache_dir / f"{i:05d}.jpg").is_file() for i in range(len(image_paths)))
    if complete:
        return
    for p in cache_dir.glob("*.jpg"):
        p.unlink()
    for i, src in enumerate(tqdm(image_paths, desc="SAM2 JPEG cache")):
        img = cv2.imread(src)
        if img is None:
            raise RuntimeError(f"Cannot read RGB: {src}")
        if not cv2.imwrite(str(cache_dir / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed writing SAM2 frame {i}")


def _choose_reference(image_paths: List[str], obj_key: str, min_area_ratio: float, max_area_ratio: float) -> Tuple[int, np.ndarray, dict]:
    best_idx, best_mask, best_score = -1, None, -1.0
    candidates = []
    for i, img_path in enumerate(image_paths):
        mp = Path(img_path).parent / f"mask_{obj_key}.png"
        if not mp.is_file():
            candidates.append({"frame": i, "score": -1.0, "area_ratio": 0.0})
            continue
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        score = _quality(mask, min_area_ratio, max_area_ratio)
        ratio = float(np.count_nonzero(mask) / mask.size) if mask is not None else 0.0
        candidates.append({"frame": i, "score": score, "area_ratio": ratio})
        if score > best_score:
            best_idx = i
            best_mask = _largest_component(mask)
            best_score = score
    if best_idx < 0 or best_mask is None:
        raise RuntimeError(f"[SAM2Video] No valid reference mask for {obj_key}")
    return best_idx, best_mask, {"best_frame": best_idx, "best_score": best_score, "candidates": candidates}


def _backup(image_paths: List[str], obj_key: str):
    for p in image_paths:
        d = Path(p).parent
        src = d / f"mask_{obj_key}.png"
        dst = d / f"mask_{obj_key}_dino.png"
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def _propagate_one(predictor, video_dir: Path, image_paths: List[str], ref_idx: int, ref_mask: np.ndarray,
                   offload_video_to_cpu=True, offload_state_to_cpu=False) -> Dict[int, np.ndarray]:
    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )
    predictor.add_new_mask(state, frame_idx=ref_idx, obj_id=1, mask=(ref_mask > 0))
    out = {ref_idx: (ref_mask > 0).astype(np.uint8) * 255}

    for frame_idx, _, logits in predictor.propagate_in_video(
        state, start_frame_idx=ref_idx, max_frame_num_to_track=len(image_paths), reverse=False
    ):
        m = (logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
        out[int(frame_idx)] = _largest_component(m)

    for frame_idx, _, logits in predictor.propagate_in_video(
        state, start_frame_idx=ref_idx, max_frame_num_to_track=len(image_paths), reverse=True
    ):
        m = (logits[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
        out[int(frame_idx)] = _largest_component(m)

    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _rebuild_combined(image_paths: List[str], obj_keys: List[str]):
    for img_path in image_paths:
        d = Path(img_path).parent
        img = cv2.imread(img_path)
        if img is None:
            continue
        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        arm_path = d / "mask_arm.png"
        if arm_path.is_file():
            arm = cv2.imread(str(arm_path), cv2.IMREAD_GRAYSCALE)
            if arm is not None:
                combined = cv2.bitwise_or(combined, arm)
        for k in obj_keys:
            mp = d / f"mask_{k}.png"
            if mp.is_file():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    combined = cv2.bitwise_or(combined, m)
        cv2.imwrite(str(d / "mask_arm_and_obj.png"), combined)


def run_sam2_video_object_propagation(
    image_paths: List[str],
    cfg_path: str,
    mps_path: str,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.60,
    backup_dino_masks: bool = True,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
    vos_optimized: bool = False,
):
    cfg = load_cfg(cfg_path)
    prompts = dict(cfg.dinosam_prompt)
    obj_keys = [k for k, v in prompts.items() if k.startswith("obj") and str(v).strip()]
    if not obj_keys:
        raise RuntimeError("[SAM2Video] No obj* prompts in DINOSAM config")

    pre = Path(mps_path) / "preprocess"
    video_dir = pre / "_sam2_video_jpegs"
    _prepare_jpeg_video(image_paths, video_dir)

    ckpt = hf_hub_download(repo_id=cfg.sam2_repo_id, filename=cfg.sam2_checkpoint_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"║ [SAM2Video] Building predictor on {device}")
    predictor = build_sam2_video_predictor(
        cfg.sam2_config, ckpt, device=device, vos_optimized=vos_optimized
    )

    diag = {"num_frames": len(image_paths), "objects": {}}
    try:
        for obj_key in obj_keys:
            ref_idx, ref_mask, info = _choose_reference(
                image_paths, obj_key, min_area_ratio, max_area_ratio
            )
            print(f"║ [SAM2Video] {obj_key}: reference={ref_idx}, score={info['best_score']:.5f}")
            if backup_dino_masks:
                _backup(image_paths, obj_key)
            masks = _propagate_one(
                predictor, video_dir, image_paths, ref_idx, ref_mask,
                offload_video_to_cpu, offload_state_to_cpu,
            )
            for i, m in masks.items():
                cv2.imwrite(str(Path(image_paths[i]).parent / f"mask_{obj_key}.png"), m)
            info["propagated_frames"] = len(masks)
            info["area_ratio_after"] = [
                float(np.count_nonzero(masks[i]) / masks[i].size) if i in masks else 0.0
                for i in range(len(image_paths))
            ]
            diag["objects"][obj_key] = info

        _rebuild_combined(image_paths, obj_keys)
        p = pre / "sam2_video_object_masks.json"
        p.write_text(json.dumps(diag, indent=2))
        print(f"║ [SAM2Video] done -> {p}")
    finally:
        try:
            predictor.to("cpu")
        except Exception:
            pass
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return diag
