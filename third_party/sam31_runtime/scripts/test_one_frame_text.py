#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM 3.1 one-frame text-prompt smoke test.

Fixes current upstream SAM3.1 error:

    TypeError:
    Sam3MultiplexTrackingWithInteractivity.init_state()
    got an unexpected keyword argument 'offload_state_to_cpu'

The fix is applied at runtime, without modifying third_party/sam3.

Usage:

CKPT="$(pwd)/checkpoints/sam3.1/sam3.1_multiplex.pt"

python scripts/test_one_frame_text.py \
  --image /data/home/chenxiangyu/victor/humanego/HumanEgo/realsense_linux_rgbd_imu_v2/captures/session_20260813_200913/preprocess/all_data/00288/rgb.png \
  --prompt "strawberry" \
  --checkpoint "$CKPT" \
  --output ./sam31_strawberry_00288.jpg
"""

from __future__ import annotations

import argparse
import inspect
import tempfile
import time
import types
import uuid
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Runtime fix for current SAM3.1 upstream start_session bug
# ============================================================

def patched_start_session(
    self,
    resource_path,
    session_id=None,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    """
    Replacement for Sam3BasePredictor.start_session().

    The important difference is that init_state kwargs are filtered
    using inspect.signature() before they are passed into the
    SAM3/SAM3.1 model.
    """

    init_kwargs = {
        "resource_path": resource_path,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }

    # Preserve predictor-specific options if present.
    if hasattr(self, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = self.async_loading_frames

    if hasattr(self, "video_loader_type"):
        init_kwargs["video_loader_type"] = self.video_loader_type

    # --------------------------------------------------------
    # Critical SAM3.1 compatibility fix
    # --------------------------------------------------------

    sig = inspect.signature(self.model.init_state)
    valid_params = set(sig.parameters.keys())

    filtered_kwargs = {
        k: v
        for k, v in init_kwargs.items()
        if k in valid_params
    }

    print()
    print("=" * 72)
    print("[SAM3.1] init_state compatibility patch")
    print("=" * 72)
    print("requested kwargs:")
    for k in init_kwargs:
        print(f"  {k}")

    print("accepted kwargs:")
    for k in filtered_kwargs:
        print(f"  {k}")

    removed = set(init_kwargs) - set(filtered_kwargs)

    if removed:
        print("removed unsupported kwargs:")
        for k in removed:
            print(f"  {k}")

    print("=" * 72)
    print()

    inference_state = self.model.init_state(
        **filtered_kwargs
    )

    if session_id is None:
        session_id = str(uuid.uuid4())

    self._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }

    return {
        "session_id": session_id,
    }


# ============================================================
# Visualization
# ============================================================

def _to_numpy(x):
    if x is None:
        return None

    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    return np.asarray(x)


def render_outputs(
    image_bgr: np.ndarray,
    outputs: dict,
) -> np.ndarray:

    vis = image_bgr.copy()

    h, w = vis.shape[:2]

    obj_ids = _to_numpy(
        outputs.get("out_obj_ids")
    )

    probs = _to_numpy(
        outputs.get("out_probs")
    )

    masks = _to_numpy(
        outputs.get("out_binary_masks")
    )

    boxes = _to_numpy(
        outputs.get("out_boxes_xywh")
    )

    if obj_ids is None:
        print("[WARNING] no out_obj_ids")
        return vis

    obj_ids = np.asarray(obj_ids).reshape(-1)

    if probs is not None:
        probs = np.asarray(probs).reshape(-1)

    print()
    print("=" * 72)
    print("SAM3.1 outputs")
    print("=" * 72)

    for i, obj_id in enumerate(obj_ids):

        obj_id = int(obj_id)

        score = (
            float(probs[i])
            if probs is not None and i < len(probs)
            else float("nan")
        )

        print(
            f"object {i}: "
            f"id={obj_id}, "
            f"score={score:.4f}"
        )

        # Stable visualization color.
        color = (
            int((83 * obj_id + 60) % 255),
            int((149 * obj_id + 100) % 255),
            int((211 * obj_id + 160) % 255),
        )

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        if masks is not None and i < len(masks):

            mask = np.asarray(masks[i])

            mask = np.squeeze(mask)

            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_bool = mask > 0.5

            area = int(mask_bool.sum())

            print(
                f"    mask area = "
                f"{area} px "
                f"({area / (h*w):.4%})"
            )

            if area > 0:

                overlay = np.zeros_like(vis)
                overlay[:] = color

                vis[mask_bool] = (
                    0.55
                    * vis[mask_bool].astype(np.float32)
                    +
                    0.45
                    * overlay[mask_bool].astype(np.float32)
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    (
                        mask_bool.astype(np.uint8)
                        * 255
                    ),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                cv2.drawContours(
                    vis,
                    contours,
                    -1,
                    color,
                    2,
                )

        # ----------------------------------------------------
        # Bounding box
        # SAM3 outputs relative XYWH
        # ----------------------------------------------------

        if boxes is not None and i < len(boxes):

            box = np.asarray(
                boxes[i],
                dtype=np.float32,
            ).reshape(-1)

            if len(box) >= 4:

                x, y, bw, bh = box[:4]

                # Relative coordinate output.
                if (
                    max(
                        abs(x),
                        abs(y),
                        abs(bw),
                        abs(bh),
                    )
                    <= 2.0
                ):
                    x *= w
                    bw *= w
                    y *= h
                    bh *= h

                x1 = int(round(x))
                y1 = int(round(y))
                x2 = int(round(x + bw))
                y2 = int(round(y + bh))

                cv2.rectangle(
                    vis,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                label = (
                    f"id={obj_id} "
                    f"p={score:.2f}"
                )

                cv2.putText(
                    vis,
                    label,
                    (
                        x1,
                        max(22, y1 - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    print("=" * 72)
    print()

    return vis


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="./sam31_output.jpg",
    )

    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help=(
            "Enable Flash Attention 3. "
            "Leave disabled for first smoke test."
        ),
    )

    args = parser.parse_args()

    image_path = (
        Path(args.image)
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        Path(args.checkpoint)
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"image not found: {image_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: "
            f"{checkpoint_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV cannot read: "
            f"{image_path}"
        )

    print()
    print("=" * 72)
    print("SAM 3.1 single-frame smoke test")
    print("=" * 72)
    print("image      :", image_path)
    print("shape      :", image.shape)
    print("prompt     :", args.prompt)
    print("checkpoint :", checkpoint_path)
    print("output     :", output_path)
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Import SAM3 only after argument checks.
    # --------------------------------------------------------

    from sam3.model_builder import (
        build_sam3_predictor,
    )

    print("[1/4] Building SAM3.1 predictor...")

    predictor = build_sam3_predictor(
        checkpoint_path=str(
            checkpoint_path
        ),
        version="sam3.1",
        use_fa3=bool(args.use_fa3),
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )

    # --------------------------------------------------------
    # Runtime monkey patch.
    # No third_party/sam3 source modification needed.
    # --------------------------------------------------------

    predictor.start_session = types.MethodType(
        patched_start_session,
        predictor,
    )

    print("[2/4] Predictor ready.")

    # --------------------------------------------------------
    # SAM3.1 video predictor expects a video resource.
    # Make a temporary one-frame video directory.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="sam31_oneframe_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        frame_path = (
            temp_dir
            / "00000.jpg"
        )

        ok = cv2.imwrite(
            str(frame_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98,
            ],
        )

        if not ok:
            raise RuntimeError(
                "Failed to create temporary "
                "SAM3.1 frame."
            )

        print(
            "[3/4] Starting one-frame "
            "SAM3.1 session..."
        )

        start_response = (
            predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(
                        temp_dir
                    ),
                }
            )
        )

        session_id = (
            start_response[
                "session_id"
            ]
        )

        print(
            "session id:",
            session_id,
        )

        try:

            print(
                "[4/4] Running text prompt:",
                repr(args.prompt),
            )

            response = (
                predictor.handle_request(
                    {
                        "type":
                            "add_prompt",

                        "session_id":
                            session_id,

                        "frame_index":
                            0,

                        "text":
                            args.prompt,
                    }
                )
            )

            outputs = response[
                "outputs"
            ]

            vis = render_outputs(
                image,
                outputs,
            )

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            header_h = 48

            canvas = np.zeros(
                (
                    vis.shape[0]
                    + header_h,
                    vis.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

            canvas[
                header_h:
            ] = vis

            cv2.putText(
                canvas,
                (
                    "SAM3.1 | prompt: "
                    f"{args.prompt}"
                ),
                (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            ok = cv2.imwrite(
                str(output_path),
                canvas,
            )

            if not ok:
                raise RuntimeError(
                    f"Failed writing "
                    f"{output_path}"
                )

            print()
            print(
                "[OK] output saved:"
            )
            print(
                output_path
            )

        finally:

            try:
                predictor.handle_request(
                    {
                        "type":
                            "close_session",

                        "session_id":
                            session_id,
                    }
                )

            except Exception as exc:

                print(
                    "[WARNING] "
                    "close_session failed:",
                    repr(exc),
                )


if __name__ == "__main__":
    main()