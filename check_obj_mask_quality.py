#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def largest_component_ratio(mask01):
    area = int(mask01.sum())
    if area <= 0:
        return 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / float(area)


def classify(mask, min_area_ratio, max_area_ratio, min_lc_ratio):
    m = mask > 0
    nz = int(m.sum())
    if nz == 0:
        return "EMPTY", 0.0, 0.0
    ar = nz / float(m.size)
    lc = largest_component_ratio(m)
    if ar < min_area_ratio:
        return "TOO_SMALL", ar, lc
    if ar > max_area_ratio:
        return "TOO_LARGE", ar, lc
    if lc < min_lc_ratio:
        return "FRAGMENTED", ar, lc
    return "PLAUSIBLE", ar, lc


def overlay(rgb, mask, status, frame_name, ar, lc):
    out = rgb.copy()
    sel = mask > 0
    if np.any(sel):
        green = np.zeros_like(out)
        green[:, :, 1] = 255
        out[sel] = (0.55*out[sel] + 0.45*green[sel]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0,255,0), 2)
    cv2.rectangle(out, (0,0), (out.shape[1],74), (0,0,0), -1)
    cv2.putText(out, f"{frame_name} | {status}", (12,27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"area={ar:.4f}  largest_component={lc:.3f}", (12,56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--obj", default="obj1")
    ap.add_argument("--min-area-ratio", type=float, default=0.002)
    ap.add_argument("--max-area-ratio", type=float, default=0.35)
    ap.add_argument("--min-largest-component-ratio", type=float, default=0.70)
    ap.add_argument("--make-samples", action="store_true")
    ap.add_argument("--samples-per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    session = Path(args.session).expanduser().resolve()
    all_data = session / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(all_data)

    out_dir = session / "preprocess" / f"mask_quality_{args.obj}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, by_class = [], {}
    frame_dirs = sorted(d for d in all_data.iterdir() if d.is_dir())

    for d in frame_dirs:
        mp = d / f"mask_{args.obj}.png"
        if not mp.exists():
            status, nz, ar, lc, exists = "MISSING", 0, 0.0, 0.0, 0
        else:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                status, nz, ar, lc, exists = "MISSING", 0, 0.0, 0.0, 0
            else:
                status, ar, lc = classify(mask, args.min_area_ratio,
                                          args.max_area_ratio,
                                          args.min_largest_component_ratio)
                nz, exists = int(np.count_nonzero(mask)), 1

        row = dict(frame=d.name, status=status, mask_exists=exists,
                   nonzero=nz, area_ratio=ar, largest_component_ratio=lc)
        rows.append(row)
        by_class.setdefault(status, []).append(d)

    csv_path = out_dir / "mask_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["frame","status","mask_exists","nonzero","area_ratio","largest_component_ratio"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["status"] for r in rows)
    total = len(rows)
    has_content = total - counts["MISSING"] - counts["EMPTY"]

    print("="*68)
    print(f"Object Mask Quality Report: {args.obj}")
    print("="*68)
    print("Session :", session)
    print("Frames  :", total)
    print()
    for k in ["PLAUSIBLE","EMPTY","MISSING","TOO_SMALL","FRAGMENTED","TOO_LARGE"]:
        n = counts[k]
        print(f"{k:12s}: {n:4d} / {total:4d} ({n/max(total,1):6.2%})")
    print("-"*68)
    print(f"HAS CONTENT : {has_content:4d} / {total:4d} ({has_content/max(total,1):6.2%})")
    print(f"PLAUSIBLE   : {counts['PLAUSIBLE']:4d} / {total:4d} ({counts['PLAUSIBLE']/max(total,1):6.2%})")
    print("CSV         :", csv_path)
    print()
    print("注：PLAUSIBLE 只表示面积/连通性合理，不保证语义上一定是正确物体。")
    print("="*68)

    if args.make_samples:
        rng = random.Random(args.seed)
        sample_root = out_dir / "samples"
        sample_root.mkdir(exist_ok=True)

        for status, dirs in by_class.items():
            target = sample_root / status
            target.mkdir(exist_ok=True)
            chosen = dirs[:]
            rng.shuffle(chosen)

            for d in chosen[:args.samples_per_class]:
                rgb = cv2.imread(str(d/"rgb.png"))
                if rgb is None:
                    continue
                mp = d / f"mask_{args.obj}.png"
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
                if mask is None:
                    mask = np.zeros(rgb.shape[:2], np.uint8)
                s2, ar, lc = classify(mask, args.min_area_ratio,
                                      args.max_area_ratio,
                                      args.min_largest_component_ratio)
                if not mp.exists():
                    s2 = "MISSING"
                vis = overlay(rgb, mask, s2, d.name, ar, lc)
                cv2.imwrite(str(target/f"{d.name}.jpg"), vis)

        print("Sample overlays:", sample_root)


if __name__ == "__main__":
    main()
