#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize CURRENT HumanEgo object masks as MP4.

Default:
  Use the exact ordered frame sequence that the current SAM2 temporal
  propagator actually received (from its manifest).

Optional:
  --all-frames
  Visualize every numeric all_data frame. Frames outside the current temporal
  input sequence are explicitly labelled OUTSIDE TEMPORAL INPUT.

Default 3 columns:
  RGB | DINO backup | FINAL temporal mask

Use --final-only for:
  RGB | FINAL temporal mask

This script never reruns or modifies DINO/SAM2.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def temporal_source_paths(session: Path):
    pre = session / "preprocess"
    diag_path = pre / "sam2_video_object_masks.json"
    diag = read_json(diag_path)
    if isinstance(diag, dict):
        manifest = diag.get("cache_manifest")
        if isinstance(manifest, dict):
            xs = manifest.get("source_paths")
            if isinstance(xs, list) and xs:
                return [Path(x) for x in xs], str(diag_path)

    manifest_path = pre / "_sam2_video_jpegs" / "manifest.json"
    m = read_json(manifest_path)
    if isinstance(m, dict):
        xs = m.get("source_paths")
        if isinstance(xs, list) and xs:
            return [Path(x) for x in xs], str(manifest_path)

    return [], "no temporal manifest"


def all_rgb_paths(session: Path):
    root = session / "preprocess" / "all_data"
    if not root.is_dir():
        return []
    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    return [d / "rgb.png" for d in dirs if (d / "rgb.png").is_file()]


def load_mask(path: Path, hw):
    if not path.is_file():
        return np.zeros(hw, np.uint8), False
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(hw, np.uint8), False
    if m.shape != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m, True


def lc_ratio(mask):
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area <= 0:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / area


def panel(rgb, mask, title, exists, outside=False):
    out = rgb.copy()
    sel = mask > 0
    if np.any(sel):
        green = np.zeros_like(out)
        green[:, :, 1] = 255
        out[sel] = (0.55*out[sel] + 0.45*green[sel]).astype(np.uint8)
        contours, _ = cv2.findContours(
            (sel.astype(np.uint8)*255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, (0,255,0), 2)

    ar = np.count_nonzero(mask) / float(mask.size)
    status = "MISSING" if not exists else ("EMPTY" if ar == 0 else "MASK")
    if outside:
        status = "OUTSIDE TEMPORAL INPUT | " + status

    cv2.rectangle(out, (0,0), (out.shape[1],76), (0,0,0), -1)
    cv2.putText(out, title, (12,26), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{status}  area={ar:.4f}  LC={lc_ratio(mask):.3f}",
                (12,57), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (255,255,255), 1, cv2.LINE_AA)
    return out, ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--obj", default="obj1")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="Show every all_data frame instead of only the exact SAM2 temporal input.",
    )
    args = ap.parse_args()

    session = Path(args.session).expanduser().resolve()
    tpaths, manifest_source = temporal_source_paths(session)
    temporal_set = {str(p.resolve()) for p in tpaths if p.is_file()}

    if args.all_frames:
        paths = all_rgb_paths(session)
        seq_name = "ALL all_data frames"
    else:
        paths = [p for p in tpaths if p.is_file()]
        seq_name = "EXACT current temporal input"
        if not paths:
            raise RuntimeError(
                "No temporal manifest found. Use --all-frames to visualize all existing masks."
            )

    if not paths:
        raise RuntimeError("No RGB frames found")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {paths[0]}")
    h, w = first.shape[:2]

    cols = 2 if args.final_only else 3
    out = Path(args.output).expanduser().resolve() if args.output else (
        session / "preprocess" / "vis" /
        (f"{args.obj}_mask_all_frames.mp4" if args.all_frames
         else f"{args.obj}_mask_current_temporal.mp4")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps, (w*cols, h+36)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create MP4. Check OpenCV FFmpeg support.")

    n = len(paths)
    stats = {
        "final_nonempty": 0,
        "final_empty": 0,
        "final_missing": 0,
        "dino_nonempty": 0,
        "inside_temporal": 0,
        "outside_temporal": 0,
    }

    try:
        for i, rp in enumerate(paths):
            rgb = cv2.imread(str(rp))
            if rgb is None:
                continue
            if rgb.shape[:2] != (h,w):
                rgb = cv2.resize(rgb, (w,h))

            d = rp.parent
            frame_id = d.name
            in_temporal = str(rp.resolve()) in temporal_set
            stats["inside_temporal" if in_temporal else "outside_temporal"] += 1

            fm, fex = load_mask(d/f"mask_{args.obj}.png", (h,w))
            dm, dex = load_mask(d/f"mask_{args.obj}_dino.png", (h,w))

            rgbp = rgb.copy()
            cv2.rectangle(rgbp, (0,0), (w,76), (0,0,0), -1)
            cv2.putText(rgbp, "RGB", (12,26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(rgbp, f"source frame {frame_id}", (12,57),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (255,255,255), 1, cv2.LINE_AA)

            fp, far = panel(
                rgb, fm, "FINAL mask_obj", fex,
                outside=(args.all_frames and not in_temporal)
            )
            if not fex:
                stats["final_missing"] += 1
            elif far == 0:
                stats["final_empty"] += 1
            else:
                stats["final_nonempty"] += 1

            if args.final_only:
                row = cv2.hconcat([rgbp, fp])
            else:
                dp, dar = panel(
                    rgb, dm, "DINO backup",
                    dex, outside=(args.all_frames and not in_temporal)
                )
                if dex and dar > 0:
                    stats["dino_nonempty"] += 1
                row = cv2.hconcat([rgbp, dp, fp])

            canvas = np.zeros((h+36, row.shape[1], 3), np.uint8)
            canvas[36:] = row
            cv2.putText(
                canvas,
                f"{seq_name} | {i+1}/{n} | frame={frame_id} | "
                f"in_temporal={'YES' if in_temporal else 'NO'}",
                (12,25), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (255,255,255), 1, cv2.LINE_AA
            )
            writer.write(canvas)

            if (i+1) % 25 == 0 or i+1 == n:
                print(f"processed {i+1}/{n}")
    finally:
        writer.release()

    print()
    print("="*72)
    print("OBJ MASK VIDEO SUMMARY")
    print("="*72)
    print("Session          :", session)
    print("Object           :", args.obj)
    print("Sequence mode    :", seq_name)
    print("Temporal manifest:", manifest_source)
    print("Frames in video  :", n)
    print("Inside temporal  :", stats["inside_temporal"])
    print("Outside temporal :", stats["outside_temporal"])
    print("Final non-empty  :", stats["final_nonempty"])
    print("Final empty      :", stats["final_empty"])
    print("Final missing    :", stats["final_missing"])
    if not args.final_only:
        print("DINO non-empty   :", stats["dino_nonempty"])
        print("NOTE: DINO backup may be stale if an old mask_obj*_dino.png was kept.")
    print("Saved            :", out)
    print("="*72)


if __name__ == "__main__":
    main()
