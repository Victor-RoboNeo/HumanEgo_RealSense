# -*- coding: utf-8 -*-
"""
Fixed SAM2 video object-mask propagation for RealSense-HumanEgo.

Key fixes:
1. JPEG cache is valid ONLY when its manifest exactly matches current image_paths.
2. Any stale/extra JPEG frames force a full cache rebuild.
3. Predictor frame count is checked before propagation.
4. SAM2 frame_idx is bounds-checked before mapping back to image_paths.
5. Original DINO masks can still be backed up as mask_obj*_dino.png.

Drop-in API:
    run_sam2_video_object_propagation(
        image_paths=...,
        cfg_path=...,
        mps_path=...,
        ...
    )
"""

from __future__ import annotations

import hashlib
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
    mask01 = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, 8)
    if n <= 1:
        return mask01 * 255
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def _mask_quality(
    mask: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
) -> float:
    if mask is None:
        return -1.0

    fg = mask > 0
    area = int(np.count_nonzero(fg))
    if area <= 0:
        return -1.0

    h, w = mask.shape[:2]
    ratio = area / float(h * w)

    if ratio < min_area_ratio or ratio > max_area_ratio:
        return -1.0

    largest = _largest_component(mask)
    largest_area = int(np.count_nonzero(largest))
    coherence = largest_area / max(area, 1)

    return float(coherence * np.sqrt(ratio))


def _frame_manifest(image_paths: List[str]) -> dict:
    """
    Manifest identifies the exact ordered source-frame sequence.
    SAM2 local frame i must correspond exactly to image_paths[i].
    """
    resolved = [str(Path(p).resolve()) for p in image_paths]
    payload = "\n".join(resolved).encode("utf-8")
    return {
        "num_frames": len(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_paths": resolved,
    }


def _cache_is_exact(cache_dir: Path, expected: dict) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not cache_dir.is_dir() or not manifest_path.is_file():
        return False

    try:
        actual = json.loads(manifest_path.read_text())
    except Exception:
        return False

    if actual.get("num_frames") != expected["num_frames"]:
        return False

    if actual.get("sha256") != expected["sha256"]:
        return False

    jpgs = sorted(cache_dir.glob("*.jpg"))

    # Exact count: extra stale JPEGs are NOT allowed.
    if len(jpgs) != expected["num_frames"]:
        return False

    expected_names = [
        f"{i:05d}.jpg"
        for i in range(expected["num_frames"])
    ]
    if [p.name for p in jpgs] != expected_names:
        return False

    return True


def _prepare_jpeg_video(
    image_paths: List[str],
    cache_dir: Path,
    jpeg_quality: int = 95,
) -> dict:
    """
    Build an EXACT contiguous JPEG sequence:
        00000.jpg ... N-1.jpg

    Old behavior only checked that the first N cached files existed, so an old
    cache with 381 JPEGs could be incorrectly reused for a new 79-frame list.
    """
    if not image_paths:
        raise ValueError("[SAM2Video] image_paths is empty")

    expected = _frame_manifest(image_paths)

    if _cache_is_exact(cache_dir, expected):
        print(
            f"║ [SAM2Video] Reusing exact JPEG cache: "
            f"{expected['num_frames']} frames"
        )
        return expected

    if cache_dir.exists():
        print(
            "║ [SAM2Video] Stale/mismatched JPEG cache detected; rebuilding..."
        )
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(
        tqdm(image_paths, desc="SAM2 JPEG cache")
    ):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                f"[SAM2Video] Cannot read RGB frame {i}: {src}"
            )

        dst = cache_dir / f"{i:05d}.jpg"
        ok = cv2.imwrite(
            str(dst),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(
                f"[SAM2Video] Failed writing cache frame: {dst}"
            )

    (cache_dir / "manifest.json").write_text(
        json.dumps(expected, indent=2),
        encoding="utf-8",
    )

    # Final hard validation.
    if not _cache_is_exact(cache_dir, expected):
        raise RuntimeError(
            "[SAM2Video] JPEG cache validation failed after rebuild."
        )

    print(
        f"║ [SAM2Video] Built exact JPEG cache: "
        f"{expected['num_frames']} frames"
    )
    return expected


def _choose_reference(
    image_paths: List[str],
    obj_key: str,
    min_area_ratio: float,
    max_area_ratio: float,
) -> Tuple[int, np.ndarray, dict]:
    best_idx = -1
    best_mask = None
    best_score = -1.0
    candidates = []

    for i, img_path in enumerate(image_paths):
        frame_dir = Path(img_path).parent
        mask_path = frame_dir / f"mask_{obj_key}.png"

        if not mask_path.is_file():
            candidates.append({
                "frame": i,
                "score": -1.0,
                "area_ratio": 0.0,
                "exists": False,
            })
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            continue

        score = _mask_quality(
            mask,
            min_area_ratio,
            max_area_ratio,
        )
        area_ratio = np.count_nonzero(mask) / float(mask.size)

        candidates.append({
            "frame": i,
            "score": float(score),
            "area_ratio": float(area_ratio),
            "exists": True,
        })

        if score > best_score:
            best_score = score
            best_idx = i
            best_mask = _largest_component(mask)

    if best_idx < 0 or best_mask is None:
        raise RuntimeError(
            f"[SAM2Video] No valid reference mask for {obj_key}."
        )

    return best_idx, best_mask, {
        "best_frame": int(best_idx),
        "best_score": float(best_score),
        "candidates": candidates,
    }


def _backup_object_masks(
    image_paths: List[str],
    obj_key: str,
) -> None:
    for p in image_paths:
        frame_dir = Path(p).parent
        src = frame_dir / f"mask_{obj_key}.png"
        dst = frame_dir / f"mask_{obj_key}_dino.png"
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def _state_num_frames(state) -> int | None:
    """
    SAM2 versions expose num_frames in inference_state.
    Return None when unavailable rather than assuming.
    """
    if isinstance(state, dict):
        n = state.get("num_frames")
        if n is not None:
            try:
                return int(n)
            except Exception:
                pass

        images = state.get("images")
        if images is not None:
            try:
                return int(len(images))
            except Exception:
                pass

    return None


def _propagate_one_object(
    predictor,
    video_dir: Path,
    image_paths: List[str],
    ref_idx: int,
    ref_mask: np.ndarray,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> Dict[int, np.ndarray]:

    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )

    state_n = _state_num_frames(state)
    expected_n = len(image_paths)

    if state_n is not None and state_n != expected_n:
        raise RuntimeError(
            "[SAM2Video] Predictor/cache frame mismatch: "
            f"predictor={state_n}, image_paths={expected_n}. "
            "Delete preprocess/_sam2_video_jpegs and retry."
        )

    predictor.add_new_mask(
        inference_state=state,
        frame_idx=int(ref_idx),
        obj_id=1,
        mask=(ref_mask > 0),
    )

    outputs: Dict[int, np.ndarray] = {
        int(ref_idx): (ref_mask > 0).astype(np.uint8) * 255
    }

    def consume(iterator, direction: str):
        for frame_idx, obj_ids, mask_logits in iterator:
            i = int(frame_idx)

            # This should never trigger after exact cache validation.
            # Raise instead of silently corrupting frame mapping.
            if not (0 <= i < expected_n):
                raise RuntimeError(
                    f"[SAM2Video] {direction} returned frame_idx={i}, "
                    f"but image_paths has {expected_n} frames "
                    f"(valid 0..{expected_n - 1}). "
                    "This indicates a stale/mismatched SAM2 video cache."
                )

            mask = (
                mask_logits[0, 0] > 0.0
            ).detach().cpu().numpy().astype(np.uint8) * 255

            outputs[i] = _largest_component(mask)

    consume(
        predictor.propagate_in_video(
            state,
            start_frame_idx=int(ref_idx),
            max_frame_num_to_track=expected_n,
            reverse=False,
        ),
        "forward",
    )

    consume(
        predictor.propagate_in_video(
            state,
            start_frame_idx=int(ref_idx),
            max_frame_num_to_track=expected_n,
            reverse=True,
        ),
        "reverse",
    )

    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def _save_outputs(
    image_paths: List[str],
    obj_key: str,
    masks_by_frame: Dict[int, np.ndarray],
) -> None:
    n = len(image_paths)

    for i, mask in masks_by_frame.items():
        if not (0 <= int(i) < n):
            raise RuntimeError(
                f"[SAM2Video] Refusing to save invalid frame_idx={i}; "
                f"image_paths length={n}"
            )

        out = (
            Path(image_paths[int(i)]).parent
            / f"mask_{obj_key}.png"
        )
        cv2.imwrite(str(out), mask)


def _rebuild_combined_masks(
    image_paths: List[str],
    obj_keys: List[str],
) -> None:
    for img_path in image_paths:
        frame_dir = Path(img_path).parent
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        combined = np.zeros(
            img.shape[:2],
            dtype=np.uint8,
        )

        arm_path = frame_dir / "mask_arm.png"
        if arm_path.is_file():
            arm = cv2.imread(
                str(arm_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if arm is not None:
                combined = cv2.bitwise_or(combined, arm)

        for obj_key in obj_keys:
            p = frame_dir / f"mask_{obj_key}.png"
            if p.is_file():
                m = cv2.imread(
                    str(p),
                    cv2.IMREAD_GRAYSCALE,
                )
                if m is not None:
                    combined = cv2.bitwise_or(
                        combined,
                        m,
                    )

        cv2.imwrite(
            str(frame_dir / "mask_arm_and_obj.png"),
            combined,
        )


def run_sam2_video_object_propagation(
    image_paths: List[str],
    cfg_path: str,
    mps_path: str,
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.60,
    backup_dino_masks: bool = True,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
    vos_optimized: bool = False,
):
    """
    Drop-in replacement for the previous temporal patch.
    """

    if not image_paths:
        raise ValueError("[SAM2Video] image_paths is empty")

    cfg = load_cfg(cfg_path)
    prompts = dict(cfg.dinosam_prompt)

    obj_keys = [
        k for k, prompt in prompts.items()
        if str(k).startswith("obj") and str(prompt).strip()
    ]

    if not obj_keys:
        raise RuntimeError(
            "[SAM2Video] No obj* prompts found in DINOSAM config."
        )

    session_pre = Path(mps_path) / "preprocess"
    video_dir = session_pre / "_sam2_video_jpegs"

    manifest = _prepare_jpeg_video(
        image_paths,
        video_dir,
    )

    ckpt_path = hf_hub_download(
        repo_id=cfg.sam2_repo_id,
        filename=cfg.sam2_checkpoint_name,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"║ [SAM2Video] Building predictor on {device}; "
        f"frames={len(image_paths)}"
    )

    predictor = build_sam2_video_predictor(
        cfg.sam2_config,
        ckpt_path,
        device=device,
        vos_optimized=bool(vos_optimized),
    )

    diagnostics = {
        "num_frames": len(image_paths),
        "cache_manifest": manifest,
        "objects": {},
    }

    try:
        for obj_key in obj_keys:
            ref_idx, ref_mask, ref_info = _choose_reference(
                image_paths,
                obj_key,
                min_area_ratio,
                max_area_ratio,
            )

            print(
                f"║ [SAM2Video] {obj_key}: "
                f"reference local_idx={ref_idx}, "
                f"source_frame={Path(image_paths[ref_idx]).parent.name}, "
                f"score={ref_info['best_score']:.5f}"
            )

            if backup_dino_masks:
                _backup_object_masks(
                    image_paths,
                    obj_key,
                )

            masks = _propagate_one_object(
                predictor=predictor,
                video_dir=video_dir,
                image_paths=image_paths,
                ref_idx=ref_idx,
                ref_mask=ref_mask,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )

            _save_outputs(
                image_paths,
                obj_key,
                masks,
            )

            diagnostics["objects"][obj_key] = {
                **ref_info,
                "propagated_frames": len(masks),
                "source_reference_frame": (
                    Path(image_paths[ref_idx]).parent.name
                ),
            }

        _rebuild_combined_masks(
            image_paths,
            obj_keys,
        )

        diag_path = (
            session_pre
            / "sam2_video_object_masks.json"
        )
        diag_path.write_text(
            json.dumps(diagnostics, indent=2),
            encoding="utf-8",
        )

        print(
            f"║ [SAM2Video] complete: "
            f"{len(image_paths)} frames; "
            f"diagnostics={diag_path}"
        )

    finally:
        try:
            predictor.to("cpu")
        except Exception:
            pass

        del predictor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return diagnostics
