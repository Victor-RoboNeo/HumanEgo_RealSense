#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 3.1 text + point refinement + video tracking for HumanEgo.

Includes a runtime workaround for the current SAM3.1 upstream
`offload_state_to_cpu` init_state incompatibility.

Example:
    --prompt "strawberry"
    --pos "398,238"
    --neg "470,285;500,320"

Positive/negative coordinates are absolute pixels on the prompt frame.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import types
import uuid
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (
    abs_points_to_rel,
    build_jpeg_cache_from_session,
    choose_obj_id,
    extract_selected_mask,
    load_local_frame,
    parse_xy_points,
    render_outputs_bgr,
    source_idx_for_local,
)


def _patch_start_session_for_sam31(predictor):
    """Filter unsupported init_state kwargs at runtime."""

    def _patched_start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        init_kwargs = {
            "resource_path":
                resource_path,

            "offload_video_to_cpu":
                offload_video_to_cpu,

            "offload_state_to_cpu":
                offload_state_to_cpu,
        }

        if hasattr(
            self,
            "async_loading_frames",
        ):
            init_kwargs[
                "async_loading_frames"
            ] = (
                self.async_loading_frames
            )

        if hasattr(
            self,
            "video_loader_type",
        ):
            init_kwargs[
                "video_loader_type"
            ] = (
                self.video_loader_type
            )

        sig = inspect.signature(
            self.model.init_state
        )

        valid = set(
            sig.parameters.keys()
        )

        filtered = {
            k: v
            for k, v
            in init_kwargs.items()
            if k in valid
        }

        removed = [
            k
            for k
            in init_kwargs
            if k not in valid
        ]

        if removed:
            print(
                "[SAM3.1 compat] "
                "filtered unsupported "
                "init_state kwargs:",
                ", ".join(
                    removed
                ),
            )

        inference_state = (
            self.model.init_state(
                **filtered
            )
        )

        if not session_id:
            session_id = (
                str(
                    uuid.uuid4()
                )
            )

        self._all_inference_states[
            session_id
        ] = {
            "state":
                inference_state,

            "session_id":
                session_id,

            "start_time":
                time.time(),

            "last_use_time":
                time.time(),
        }

        return {
            "session_id":
                session_id
        }

    predictor.start_session = (
        types.MethodType(
            _patched_start_session,
            predictor,
        )
    )

    return predictor


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--session",
        required=True,
    )

    ap.add_argument(
        "--prompt",
        required=True,
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
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
        "--stride",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--max-frames",
        type=int,
        default=90,
    )

    ap.add_argument(
        "--prompt-local-index",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--select",
        choices=[
            "highest_score",
            "largest",
        ],
        default="highest_score",
    )

    ap.add_argument(
        "--obj-id",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--pos",
        action="append",
        default=[],
        help=(
            'Positive point(s), e.g. '
            '--pos "398,238;405,242"'
        ),
    )

    ap.add_argument(
        "--neg",
        action="append",
        default=[],
        help=(
            'Negative point(s), e.g. '
            '--neg "470,285;500,320"'
        ),
    )

    ap.add_argument(
        "--fps",
        type=float,
        default=15.0,
    )

    ap.add_argument(
        "--use-fa3",
        action="store_true",
    )

    ap.add_argument(
        "--output-dir",
        default=(
            "./sam31_video_refine_test"
        ),
    )

    args = ap.parse_args()

    from sam3.model_builder import (
        build_sam3_predictor,
    )

    session = (
        Path(
            args.session
        )
        .expanduser()
        .resolve()
    )

    checkpoint = (
        Path(
            args.checkpoint
        )
        .expanduser()
        .resolve()
    )

    out_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = (
        out_dir
        / "jpeg_cache"
    )

    manifest = (
        build_jpeg_cache_from_session(
            session,
            cache,
            start=args.start,
            end=args.end,
            stride=args.stride,
            max_frames=args.max_frames,
        )
    )

    n = len(
        manifest[
            "frames"
        ]
    )

    if not (
        0
        <= args.prompt_local_index
        < n
    ):
        raise ValueError(
            f"--prompt-local-index "
            f"must be in "
            f"[0, {n - 1}]"
        )

    predictor = (
        build_sam3_predictor(
            checkpoint_path=str(
                checkpoint
            ),
            version="sam3.1",
            use_fa3=bool(
                args.use_fa3
            ),
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    )

    predictor = (
        _patch_start_session_for_sam31(
            predictor
        )
    )

    start_resp = (
        predictor.handle_request(
            {
                "type":
                    "start_session",

                "resource_path":
                    str(cache),
            }
        )
    )

    sid = start_resp[
        "session_id"
    ]

    try:

        # --------------------------------------------------
        # 1. Text concept prompt
        # --------------------------------------------------

        text_resp = (
            predictor.handle_request(
                {
                    "type":
                        "add_prompt",

                    "session_id":
                        sid,

                    "frame_index":
                        int(
                            args.prompt_local_index
                        ),

                    "text":
                        args.prompt,

                    "output_prob_thresh":
                        float(
                            args.threshold
                        ),
                }
            )
        )

        text_out = (
            text_resp[
                "outputs"
            ]
        )

        ids = np.asarray(
            text_out.get(
                "out_obj_ids",
                [],
            )
        ).reshape(-1)

        probs = np.asarray(
            text_out.get(
                "out_probs",
                [],
            )
        ).reshape(-1)

        print(
            "text IDs    :",
            ids.tolist(),
        )

        print(
            "text scores :",
            probs.tolist(),
        )

        selected_id = (
            choose_obj_id(
                text_out,
                mode=args.select,
                explicit=args.obj_id,
            )
        )

        print(
            "selected ID :",
            selected_id,
        )

        prompt_img = (
            load_local_frame(
                cache,
                args.prompt_local_index,
            )
        )

        h, w = (
            prompt_img.shape[:2]
        )

        # --------------------------------------------------
        # 2. Positive / negative refinement
        # --------------------------------------------------

        pos = (
            parse_xy_points(
                args.pos
            )
        )

        neg = (
            parse_xy_points(
                args.neg
            )
        )

        all_points = (
            pos + neg
        )

        labels = (
            [1] * len(pos)
            +
            [0] * len(neg)
        )

        if all_points:

            rel_points = (
                abs_points_to_rel(
                    all_points,
                    w,
                    h,
                )
            )

            refine_resp = (
                predictor.handle_request(
                    {
                        "type":
                            "add_prompt",

                        "session_id":
                            sid,

                        "frame_index":
                            int(
                                args.prompt_local_index
                            ),

                        "points":
                            rel_points,

                        "point_labels":
                            labels,

                        "obj_id":
                            int(
                                selected_id
                            ),

                        "rel_coordinates":
                            True,

                        "output_prob_thresh":
                            float(
                                args.threshold
                            ),

                        "clear_old_points":
                            True,
                    }
                )
            )

            prompt_out = (
                refine_resp[
                    "outputs"
                ]
            )

            print(
                "visual refinement applied: "
                f"+{len(pos)} / "
                f"-{len(neg)}"
            )

        else:

            prompt_out = (
                text_out
            )

            print(
                "No --pos/--neg supplied: "
                "text-only propagation."
            )

        before = (
            render_outputs_bgr(
                prompt_img,
                text_out,
                selected_obj_id=(
                    selected_id
                ),
            )
        )

        after = (
            render_outputs_bgr(
                prompt_img,
                prompt_out,
                selected_obj_id=(
                    selected_id
                ),
            )
        )

        for x, y in pos:

            cv2.circle(
                after,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                6,
                (
                    0,
                    255,
                    0,
                ),
                -1,
            )

        for x, y in neg:

            cv2.circle(
                after,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                6,
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )

        cv2.imwrite(
            str(
                out_dir
                / "prompt_before_after.jpg"
            ),
            cv2.hconcat(
                [
                    before,
                    after,
                ]
            ),
        )

        # --------------------------------------------------
        # 3. Video propagation
        # --------------------------------------------------

        outputs_by_local = {}

        for response in tqdm(
            predictor.handle_stream_request(
                {
                    "type":
                        "propagate_in_video",

                    "session_id":
                        sid,

                    "propagation_direction":
                        "both",

                    "start_frame_index":
                        int(
                            args.prompt_local_index
                        ),

                    "max_frame_num_to_track":
                        n,

                    "output_prob_thresh":
                        float(
                            args.threshold
                        ),
                }
            ),
            total=n,
            desc=(
                "SAM3.1 refined propagate"
            ),
        ):

            outputs_by_local[
                int(
                    response[
                        "frame_index"
                    ]
                )
            ] = (
                response[
                    "outputs"
                ]
            )

        outputs_by_local.setdefault(
            int(
                args.prompt_local_index
            ),
            prompt_out,
        )

        mask_root = (
            out_dir
            / "selected_masks"
        )

        mask_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = (
            cv2.VideoWriter(
                str(
                    out_dir
                    / (
                        "selected_track_"
                        "refined.mp4"
                    )
                ),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                float(
                    args.fps
                ),
                (
                    w * 2,
                    h,
                ),
            )
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create "
                "selected_track_refined.mp4"
            )

        report = {
            "session":
                str(session),

            "prompt":
                args.prompt,

            "threshold":
                float(
                    args.threshold
                ),

            "selected_obj_id":
                int(
                    selected_id
                ),

            "prompt_local_index":
                int(
                    args.prompt_local_index
                ),

            "prompt_source_idx":
                int(
                    source_idx_for_local(
                        manifest,
                        args.prompt_local_index,
                    )
                ),

            "positive_points":
                pos,

            "negative_points":
                neg,

            "frames":
                [],
        }

        try:

            for local_i in range(n):

                img = (
                    load_local_frame(
                        cache,
                        local_i,
                    )
                )

                current_out = (
                    outputs_by_local.get(
                        local_i,
                        {},
                    )
                )

                mask = (
                    extract_selected_mask(
                        current_out,
                        selected_id,
                        target_hw=(
                            img.shape[:2]
                        ),
                    )
                )

                source_idx = (
                    source_idx_for_local(
                        manifest,
                        local_i,
                    )
                )

                cv2.imwrite(
                    str(
                        mask_root
                        / (
                            f"{source_idx:05d}.png"
                        )
                    ),
                    mask,
                )

                vis = (
                    render_outputs_bgr(
                        img,
                        current_out,
                        selected_obj_id=(
                            selected_id
                        ),
                    )
                )

                cv2.putText(
                    vis,
                    (
                        f"local={local_i} "
                        f"source={source_idx} "
                        f"obj={selected_id}"
                    ),
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(
                    cv2.hconcat(
                        [
                            img,
                            vis,
                        ]
                    )
                )

                nz = int(
                    np.count_nonzero(
                        mask
                    )
                )

                report[
                    "frames"
                ].append(
                    {
                        "local_idx":
                            int(
                                local_i
                            ),

                        "source_idx":
                            int(
                                source_idx
                            ),

                        "mask_nonzero":
                            nz,

                        "mask_area_ratio":
                            float(
                                nz
                                / mask.size
                            ),
                    }
                )

        finally:
            writer.release()

        (
            out_dir
            / "report.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        nonempty = sum(
            int(
                x[
                    "mask_nonzero"
                ]
                > 0
            )
            for x
            in report[
                "frames"
            ]
        )

        print("=" * 72)

        print(
            f"selected mask non-empty: "
            f"{nonempty}/{n}"
        )

        print(
            "before/after:",
            out_dir
            / "prompt_before_after.jpg",
        )

        print(
            "video       :",
            out_dir
            / (
                "selected_track_"
                "refined.mp4"
            ),
        )

        print(
            "masks       :",
            mask_root,
        )

        print(
            "report      :",
            out_dir
            / "report.json",
        )

        print("=" * 72)

    finally:

        try:

            predictor.handle_request(
                {
                    "type":
                        "close_session",

                    "session_id":
                        sid,
                }
            )

        except Exception as exc:

            print(
                "[WARN] close_session failed:",
                repr(exc),
            )


if __name__ == "__main__":
    main()