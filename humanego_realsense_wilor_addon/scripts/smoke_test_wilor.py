#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick WiLoR detection smoke test on one RGB image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="/tmp/wilor_smoke.jpg")
    args = ap.parse_args()

    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except Exception as exc:
        raise SystemExit(
            "Cannot import wilor-mini.\n"
            "Install with the commands in README.md.\n"
            f"Original error: {exc}"
        )

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = WiLorHandPose3dEstimationPipeline(
        device=device,
        dtype=dtype,
        verbose=False,
    )
    detections = pipe.predict(img)

    print("detections:", len(detections))
    vis = img.copy()

    for i, det in enumerate(detections):
        pred = det.get("wilor_preds")
        if pred is None:
            continue
        right = bool(det.get("is_right", 1.0) >= 0.5)
        bbox = det.get("hand_bbox")
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255,255,255), 2)

        pts = np.asarray(pred["pred_keypoints_2d"][0], dtype=np.float32)
        for j, (u, v) in enumerate(pts):
            cv2.circle(vis, (int(round(u)), int(round(v))), 3,
                       (0, 255, 255) if right else (255, 180, 0), -1)
        print(
            f"hand {i}:",
            "right" if right else "left",
            "bbox=", bbox,
            "joints2d=", pts.shape,
            "joints3d=", np.asarray(pred["pred_keypoints_3d"]).shape,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
