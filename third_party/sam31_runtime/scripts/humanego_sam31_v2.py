#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import time
import types
import uuid
from pathlib import Path

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).absolute()
RUNTIME_ROOT = SCRIPT_PATH.parent.parent
SAM3_ROOT = RUNTIME_ROOT / "third_party" / "sam3"
CHECKPOINT = RUNTIME_ROOT / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"
EXPECTED_VENV = RUNTIME_ROOT / ".venv"


def _normalize_prompt(text: str) -> str:
    text = " ".join(str(text).strip().split())
    while text.endswith("."):
        text = text[:-1].rstrip()
    return text


def _frame_index_from_path(path: str) -> int:
    parent = Path(path).parent.name
    try:
        return int(parent)
    except ValueError as exc:
        raise ValueError(
            f"Expected numeric HumanEgo frame directory, got {parent!r}: {path}"
        ) from exc


def _sort_unique_image_paths(image_paths: list[str]) -> list[str]:
    by_idx = {}
    for p in image_paths:
        p_abs = str(Path(p).expanduser().absolute())
        idx = _frame_index_from_path(p_abs)
        if idx in by_idx and by_idx[idx] != p_abs:
            raise RuntimeError(
                f"Two different RGB paths claim HumanEgo frame {idx}: "
                f"{by_idx[idx]} vs {p_abs}"
            )
        by_idx[idx] = p_abs
    return [by_idx[i] for i in sorted(by_idx)]


def _patch_start_session_compat(predictor):
    def _patched_start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        requested = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            requested["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            requested["video_loader_type"] = self.video_loader_type

        sig = inspect.signature(self.model.init_state)
        accepted_names = set(sig.parameters)
        kwargs = {k: v for k, v in requested.items() if k in accepted_names}
        removed = [k for k in requested if k not in accepted_names]
        if removed:
            print(
                "[SAM31 compat] ignored unsupported init_state kwargs:",
                ", ".join(removed),
                flush=True,
            )

        inference_state = self.model.init_state(**kwargs)
        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return {"session_id": session_id}

    predictor.start_session = types.MethodType(_patched_start_session, predictor)
    return predictor


def _assert_runtime_layout() -> None:
    errors = []
    if not EXPECTED_VENV.is_dir():
        errors.append(f"missing runtime venv: {EXPECTED_VENV}")
    if not SAM3_ROOT.is_dir():
        errors.append(f"missing SAM3 source tree: {SAM3_ROOT}")
    if not CHECKPOINT.is_file():
        errors.append(f"missing SAM3.1 checkpoint: {CHECKPOINT}")
    if errors:
        raise RuntimeError("\n".join(errors))


def _self_check() -> int:
    _assert_runtime_layout()
    sys.path.insert(0, str(SAM3_ROOT))
    import torch
    import sam3

    print("=" * 78)
    print("HumanEgo SAM3.1 UV runtime self-check")
    print("=" * 78)
    print("runtime root  :", RUNTIME_ROOT)
    print("sys.executable:", sys.executable)
    print("sys.prefix    :", sys.prefix)
    print("expected venv :", EXPECTED_VENV)
    print("cv2           :", cv2.__version__)
    print("torch         :", torch.__version__)
    print("CUDA          :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(
            "GPU          :",
            torch.cuda.get_device_name(0),
        )

        print(
            "cuDNN version:",
            torch.backends.cudnn.version(),
        )

        # --------------------------------------------------
        # Actually execute one cuDNN-backed convolution.
        #
        # torch.cuda.is_available() alone is not enough to
        # detect mixed cuDNN shared-library environments.
        # --------------------------------------------------

        print(
            "[CHECK] running CUDA/cuDNN convolution..."
        )

        import torch.nn.functional as F

        x = torch.randn(
            1,
            3,
            64,
            64,
            device="cuda",
        )

        weight = torch.randn(
            16,
            3,
            3,
            3,
            device="cuda",
        )

        y = F.conv2d(
            x,
            weight,
            padding=1,
        )

        torch.cuda.synchronize()

        print(
            "[OK] CUDA/cuDNN convolution:",
            tuple(y.shape),
        )
    print("sam3 source   :", Path(sam3.__file__).absolute())
    print("checkpoint    :", CHECKPOINT)
    print("=" * 78)

    if not Path(sys.prefix).samefile(EXPECTED_VENV):
        raise RuntimeError(
            "uv did not select sam31_humanego_smoketest/.venv. "
            f"sys.prefix={sys.prefix}"
        )
    print("[OK] uv is using the intended SAM3.1 .venv")
    return 0


def _build_predictor():
    _assert_runtime_layout()
    sys.path.insert(0, str(SAM3_ROOT))
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=str(CHECKPOINT),
        version="sam3.1",
        use_fa3=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    return _patch_start_session_compat(predictor)


def _prepare_cache(image_paths: list[str], cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, src in enumerate(image_paths):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read RGB: {src}")
        dst = cache_dir / f"{local_idx:05d}.jpg"
        if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 97]):
            raise RuntimeError(f"Cannot write SAM3.1 JPEG cache: {dst}")


def _to_numpy(value):
    if value is None:
        return None
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _try_choose_instance(outputs: dict) -> tuple[int, float] | None:
    ids = _to_numpy(outputs.get("out_obj_ids"))
    probs = _to_numpy(outputs.get("out_probs"))
    if ids is None:
        return None
    ids = np.asarray(ids).reshape(-1)
    if len(ids) == 0:
        return None
    if probs is None:
        probs = np.ones(len(ids), dtype=np.float32)
    else:
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    chosen = int(np.nanargmax(probs))
    return int(ids[chosen]), float(probs[chosen])


def _choose_reference_instance(outputs: dict) -> tuple[int, float]:
    hit = _try_choose_instance(outputs)
    if hit is None:
        raise RuntimeError("SAM3.1 found no matching object on reference frame")
    return hit


def _extract_mask(outputs: dict, obj_id: int, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    ids = _to_numpy(outputs.get("out_obj_ids"))
    masks = outputs.get("out_binary_masks")
    if ids is None or masks is None:
        return np.zeros((h, w), dtype=np.uint8)
    ids = np.asarray(ids).reshape(-1)
    for i, oid in enumerate(ids.tolist()):
        if int(oid) != int(obj_id):
            continue
        m = _to_numpy(masks[i])
        if m is None:
            break
        m = np.squeeze(m)
        if m.shape != (h, w):
            m = cv2.resize(
                m.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        return (m > 0.5).astype(np.uint8) * 255
    return np.zeros((h, w), dtype=np.uint8)


def _run_forward_sequence(
    predictor,
    cache_dir: Path,
    prompt: str,
    threshold: float,
    sequence_length: int,
    hw: tuple[int, int],
) -> tuple[list[np.ndarray], dict]:
    started = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(cache_dir),
    })
    session_id = started["session_id"]
    try:
        prompt_response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": float(threshold),
        })
        prompt_outputs = prompt_response["outputs"]
        selected_id, reference_score = _choose_reference_instance(prompt_outputs)
        outputs = {0: prompt_outputs}

        # SAFE path: local frame 0 + upstream default full forward propagation.
        # No direction, no start index and no max-frame bound are supplied.
        for response in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
        }):
            idx = int(response["frame_index"])
            if not 0 <= idx < sequence_length:
                raise RuntimeError(
                    f"SAM3.1 emitted invalid local frame {idx}; "
                    f"valid range is [0,{sequence_length - 1}]"
                )
            outputs[idx] = response["outputs"]

        masks = [
            _extract_mask(outputs.get(i, {}), selected_id, hw)
            for i in range(sequence_length)
        ]
        return masks, {
            "selected_obj_id": selected_id,
            "reference_score": reference_score,
            "received_frames": sorted(outputs),
            "nonempty_frames": int(sum(np.count_nonzero(m) > 0 for m in masks)),
        }
    finally:
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception as exc:
            print("[SAM31] close_session warning:", repr(exc), flush=True)


def _probe_text_on_frame(
    predictor,
    image_path: str,
    prompt: str,
    threshold: float,
    probe_dir: Path,
) -> bool:
    """True if SAM3.1 text prompt hits on this single RGB frame."""
    _prepare_cache([image_path], probe_dir)
    started = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(probe_dir),
    })
    session_id = started["session_id"]
    try:
        prompt_response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": float(threshold),
        })
        return _try_choose_instance(prompt_response["outputs"]) is not None
    finally:
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception as exc:
            print("[SAM31] close_session warning:", repr(exc), flush=True)


def _find_first_appearance(
    predictor,
    ordered_paths: list[str],
    prompt: str,
    threshold: float,
    probe_dir: Path,
    stride: int,
) -> tuple[int | None, int]:
    """
    Return (index into ordered_paths, probe_count).
    Stride forward until the first hit, then walk back inside the last miss window.
    """
    n = len(ordered_paths)
    stride = max(1, int(stride))
    probed = 0
    last_miss = -1
    probe_idx = list(range(0, n, stride))
    if probe_idx[-1] != n - 1:
        probe_idx.append(n - 1)

    def _hit(i: int) -> bool:
        nonlocal probed
        probed += 1
        return _probe_text_on_frame(
            predictor, ordered_paths[i], prompt, threshold, probe_dir
        )

    for i in probe_idx:
        if _hit(i):
            first = i
            for j in range(last_miss + 1, i):
                if _hit(j):
                    first = j
                    break
            return first, probed
        last_miss = i
    return None, probed


def _empty_masks(hw: tuple[int, int], n: int) -> list[np.ndarray]:
    h, w = hw
    return [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]


def _write_object_masks(
    ordered_paths: list[str],
    masks: list[np.ndarray],
    obj_key: str,
    prompt: str,
    anchor_idx: int,
    fps: float,
    save_debug_video: bool,
    run_dir: Path,
    extra_report: dict,
) -> dict:
    frame_info = []
    for src, mask in zip(ordered_paths, masks):
        dst = Path(src).parent / f"mask_{obj_key}.png"
        if not cv2.imwrite(str(dst), mask):
            raise RuntimeError(f"Could not write {dst}")
        nz = int(np.count_nonzero(mask))
        frame_info.append({
            "source_frame": _frame_index_from_path(src),
            "mask_nonzero": nz,
            "mask_area_ratio": float(nz / max(mask.size, 1)),
        })
    if save_debug_video:
        _save_debug_video(
            ordered_paths, masks, run_dir / f"{obj_key}_track.mp4",
            obj_key, prompt, anchor_idx, fps
        )
    info = {
        "prompt": prompt,
        "anchor_source_frame": _frame_index_from_path(ordered_paths[anchor_idx])
        if 0 <= anchor_idx < len(ordered_paths) else None,
        "anchor_ordered_idx": anchor_idx,
        "nonempty_frames": int(sum(x["mask_nonzero"] > 0 for x in frame_info)),
        "frames": frame_info,
    }
    info.update(extra_report)
    print(
        f"[SAM31] {obj_key}: nonempty="
        f"{info['nonempty_frames']}/{len(ordered_paths)}",
        flush=True,
    )
    return info


def _save_debug_video(
    ordered_paths: list[str],
    masks: list[np.ndarray],
    out_path: Path,
    obj_key: str,
    prompt: str,
    reference_idx: int,
    fps: float,
) -> None:
    if not ordered_paths:
        return
    first = cv2.imread(ordered_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w * 2, h),
    )
    if not writer.isOpened():
        print(f"[SAM31] warning: cannot write debug video {out_path}")
        return
    try:
        for i, (src, mask) in enumerate(zip(ordered_paths, masks)):
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                continue
            overlay = img.copy()
            sel = mask > 0
            if np.any(sel):
                tint = np.zeros_like(img)
                tint[:, :, 1] = 255
                overlay[sel] = (
                    0.60 * overlay[sel].astype(np.float32)
                    + 0.40 * tint[sel].astype(np.float32)
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            source_idx = _frame_index_from_path(src)
            label = f"{obj_key}={prompt} | source={source_idx:05d}"
            if i == reference_idx:
                label += " | REF"
            cv2.putText(
                overlay, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA
            )
            writer.write(cv2.hconcat([img, overlay]))
    finally:
        writer.release()


def _run_request(request_path: Path) -> int:
    req = json.loads(request_path.read_text(encoding="utf-8"))
    ordered_paths = _sort_unique_image_paths(req["image_paths"])
    if not ordered_paths:
        raise RuntimeError("Request contains no RGB frames")

    reference_path = str(Path(req["reference_image_path"]).expanduser().absolute())
    try:
        ref_idx = ordered_paths.index(reference_path)
    except ValueError as exc:
        raise RuntimeError(
            "Reference image must be inside the chronological SAM3.1 frame list. "
            f"reference={reference_path}"
        ) from exc

    objects = {}
    for key, value in req["objects"].items():
        prompt = _normalize_prompt(value)
        if prompt:
            objects[str(key)] = prompt
    if not objects:
        raise RuntimeError("No non-empty SAM3.1 prompts")

    threshold = float(req.get("threshold", 0.50))
    fps = float(req.get("fps", 15.0))
    save_debug_video = bool(req.get("save_debug_video", True))
    anchor_mode = str(req.get("anchor", "reference")).strip().lower()
    if anchor_mode not in {"reference", "first_appearance"}:
        raise ValueError(
            f"Unknown SAM3.1 anchor mode {anchor_mode!r}; "
            "use 'reference' or 'first_appearance'"
        )
    search_stride = max(1, int(req.get("search_stride", 8)))
    output_root = Path(req["output_root"]).expanduser().absolute()
    run_dir = output_root / ".sam31_v2"
    run_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(ordered_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first RGB: {ordered_paths[0]}")
    hw = first.shape[:2]

    predictor = _build_predictor()
    report = {
        "backend": "sam31_uv_v2",
        "runtime_root": str(RUNTIME_ROOT),
        "checkpoint": str(CHECKPOINT),
        "threshold": threshold,
        "anchor": anchor_mode,
        "search_stride": search_stride,
        "ordered_source_frames": [_frame_index_from_path(x) for x in ordered_paths],
        "shared_reference_source_frame": _frame_index_from_path(reference_path),
        "shared_reference_ordered_idx": ref_idx,
        "objects": {},
    }

    print(
        f"[SAM31] frames={len(ordered_paths)} "
        f"shared_ref={_frame_index_from_path(reference_path):05d} "
        f"anchor={anchor_mode} objects={list(objects)}",
        flush=True,
    )

    forward_cache = run_dir / "forward_frames"
    reverse_cache = run_dir / "reverse_frames"
    probe_dir = run_dir / "probe_frame"

    for obj_key, prompt in objects.items():
        print(f"[SAM31] {obj_key}: prompt={prompt!r}", flush=True)

        if anchor_mode == "first_appearance":
            appear_idx, nprobe = _find_first_appearance(
                predictor, ordered_paths, prompt, threshold, probe_dir, search_stride
            )
            if appear_idx is None:
                print(
                    f"[SAM31] {obj_key}: never detected "
                    f"(probed {nprobe} frames); writing empty masks",
                    flush=True,
                )
                report["objects"][obj_key] = _write_object_masks(
                    ordered_paths, _empty_masks(hw, len(ordered_paths)),
                    obj_key, prompt, -1, fps, save_debug_video, run_dir,
                    {"forward": None, "reverse": None, "probes": nprobe},
                )
                continue

            print(
                f"[SAM31] {obj_key}: first appearance "
                f"source={_frame_index_from_path(ordered_paths[appear_idx]):05d} "
                f"(probed {nprobe} frames)",
                flush=True,
            )
            forward_paths = ordered_paths[appear_idx:]
            _prepare_cache(forward_paths, forward_cache)
            forward_masks, fdiag = _run_forward_sequence(
                predictor, forward_cache, prompt, threshold, len(forward_paths), hw
            )
            merged_masks = _empty_masks(hw, appear_idx) + forward_masks
            report["objects"][obj_key] = _write_object_masks(
                ordered_paths, merged_masks,
                obj_key, prompt, appear_idx, fps, save_debug_video, run_dir,
                {"forward": fdiag, "reverse": None, "probes": nprobe},
            )
            continue

        forward_paths = ordered_paths[ref_idx:]
        reverse_paths = list(reversed(ordered_paths[: ref_idx + 1]))
        print(
            f"[SAM31] {obj_key}: shared-ref split "
            f"forward={len(forward_paths)}, reverse={len(reverse_paths)}",
            flush=True,
        )
        _prepare_cache(forward_paths, forward_cache)
        _prepare_cache(reverse_paths, reverse_cache)
        forward_masks, fdiag = _run_forward_sequence(
            predictor, forward_cache, prompt, threshold, len(forward_paths), hw
        )
        reverse_masks, rdiag = _run_forward_sequence(
            predictor, reverse_cache, prompt, threshold, len(reverse_paths), hw
        )
        merged_masks = list(reversed(reverse_masks[1:])) + forward_masks
        if len(merged_masks) != len(ordered_paths):
            raise RuntimeError(
                f"Internal merge error: {len(merged_masks)} masks for "
                f"{len(ordered_paths)} frames"
            )
        report["objects"][obj_key] = _write_object_masks(
            ordered_paths, merged_masks,
            obj_key, prompt, ref_idx, fps, save_debug_video, run_dir,
            {"forward": fdiag, "reverse": rdiag},
        )

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[SAM31] report:", report_path, flush=True)
    return 0

def _run_session_direct(
    session,
    start,
    end,
    reference,
    object_specs,
    threshold,
    fps,
    save_debug_video,
):
    """
    Standalone HumanEgo session test.

    This runs entirely inside the SAM3.1 uv runtime.
    No jq, no HumanEgo Python environment, no external request builder.
    """

    session = Path(session).expanduser().absolute()

    all_data = (
        session
        / "preprocess"
        / "all_data"
    )

    if not all_data.is_dir():
        raise FileNotFoundError(
            f"HumanEgo all_data not found: {all_data}"
        )

    image_paths = []

    for frame_dir in all_data.iterdir():

        if not frame_dir.is_dir():
            continue

        try:
            frame_idx = int(frame_dir.name)
        except ValueError:
            continue

        if start is not None and frame_idx < start:
            continue

        if end is not None and frame_idx > end:
            continue

        rgb = frame_dir / "rgb.png"

        if rgb.is_file():
            image_paths.append(
                str(rgb.absolute())
            )

    image_paths = _sort_unique_image_paths(
        image_paths
    )

    if not image_paths:
        raise RuntimeError(
            "No RGB frames selected"
        )

    # ------------------------------------------------------
    # Reference frame
    # ------------------------------------------------------

    reference_path = (
        all_data
        / f"{int(reference):05d}"
        / "rgb.png"
    )

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference RGB does not exist: "
            f"{reference_path}"
        )

    reference_path = str(
        reference_path.absolute()
    )

    if reference_path not in image_paths:
        raise RuntimeError(
            f"Reference frame {reference} "
            f"is outside selected range "
            f"{start}..{end}"
        )

    # ------------------------------------------------------
    # Objects
    #
    # --object obj1=strawberry
    # --object obj2=bowl
    # ------------------------------------------------------

    objects = {}

    for spec in object_specs:

        if "=" not in spec:
            raise ValueError(
                f"Invalid --object {spec!r}. "
                f"Expected e.g. obj1=strawberry"
            )

        key, prompt = spec.split(
            "=",
            1,
        )

        key = key.strip()
        prompt = _normalize_prompt(
            prompt
        )

        if not key:
            raise ValueError("Empty mask key in --object")

        if not prompt:
            raise ValueError(
                f"Empty prompt for {key}"
            )

        objects[key] = prompt

    if not objects:
        raise RuntimeError(
            "No object prompts supplied"
        )

    output_root = (
        session
        / "preprocess"
    )

    request = {
        "image_paths":
            image_paths,

        "reference_image_path":
            reference_path,

        "objects":
            objects,

        "threshold":
            float(threshold),

        "fps":
            float(fps),

        "save_debug_video":
            bool(save_debug_video),

        "output_root":
            str(output_root),
    }

    request_path = (
        output_root
        / ".sam31_v2_direct_request.json"
    )

    request_path.write_text(
        json.dumps(
            request,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("SAM3.1 direct HumanEgo test")
    print("=" * 78)
    print("session   :", session)
    print(
        "frames    :",
        len(image_paths),
    )
    print(
        "range     :",
        _frame_index_from_path(
            image_paths[0]
        ),
        "->",
        _frame_index_from_path(
            image_paths[-1]
        ),
    )
    print(
        "reference :",
        reference,
    )
    print(
        "objects   :",
        objects,
    )
    print(
        "threshold :",
        threshold,
    )
    print("=" * 78)

    return _run_request(
        request_path
    )

def main() -> int:

    ap = argparse.ArgumentParser()

    # ------------------------------------------------------
    # Runtime sanity check
    # ------------------------------------------------------

    ap.add_argument(
        "--self-check",
        action="store_true",
    )

    # ------------------------------------------------------
    # Existing request JSON mode
    # ------------------------------------------------------

    ap.add_argument(
        "--request",
        default=None,
    )

    # ------------------------------------------------------
    # Direct HumanEgo session mode
    # ------------------------------------------------------

    ap.add_argument(
        "--session",
        default=None,
    )

    ap.add_argument(
        "--start",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--end",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--reference",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--object",
        action="append",
        default=[],
        help=(
            "Object prompt, e.g. "
            "--object obj1=strawberry. "
            "Can be repeated."
        ),
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=15.0,
    )

    ap.add_argument(
        "--no-debug-video",
        action="store_true",
    )

    args = ap.parse_args()

    # ------------------------------------------------------
    # 1. Environment self check
    # ------------------------------------------------------

    if args.self_check:
        return _self_check()

    # ------------------------------------------------------
    # 2. Existing request mode
    # ------------------------------------------------------

    if args.request:

        return _run_request(
            Path(
                args.request
            )
            .expanduser()
            .absolute()
        )

    # ------------------------------------------------------
    # 3. Direct session mode
    # ------------------------------------------------------

    if args.session:

        if args.reference is None:
            ap.error(
                "--reference is required "
                "with --session"
            )

        if not args.object:
            ap.error(
                "At least one --object is "
                "required with --session"
            )

        return _run_session_direct(
            session=args.session,
            start=args.start,
            end=args.end,
            reference=args.reference,
            object_specs=args.object,
            threshold=args.threshold,
            fps=args.fps,
            save_debug_video=(
                not args.no_debug_video
            ),
        )

    ap.error(
        "Use --request, --session, "
        "or --self-check"
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
if __name__ == "__main__":
    raise SystemExit(main())
