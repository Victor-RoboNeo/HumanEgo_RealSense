#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect resulting aria_hands.json coverage after the WiLoR frontend."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()

    root = Path(args.session) / "preprocess" / "all_data"
    total = r = l = both_null = r_raw = l_raw = r_opt = l_opt = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        p = d / "aria_hands.json"
        if not p.exists():
            continue
        total += 1
        x = json.loads(p.read_text())
        hr, hl = x.get("hand_r"), x.get("hand_l")

        if hr is None and hl is None:
            both_null += 1
        if hr is not None:
            r += 1
            r_raw += int(hr.get("midpoint_pose_raw_world") is not None)
            r_opt += int(hr.get("midpoint_pose_opt_world") is not None)
        if hl is not None:
            l += 1
            l_raw += int(hl.get("midpoint_pose_raw_world") is not None)
            l_opt += int(hl.get("midpoint_pose_opt_world") is not None)

    print("frames              :", total)
    print("both hands NULL     :", both_null)
    print("right exists        :", r)
    print("left exists         :", l)
    print("right raw midpoint  :", r_raw)
    print("left raw midpoint   :", l_raw)
    print("right opt midpoint  :", r_opt)
    print("left opt midpoint   :", l_opt)
    if total:
        print("both-null ratio     :", f"{both_null/total:.2%}")
        print("right coverage      :", f"{r/total:.2%}")
        print("left coverage       :", f"{l/total:.2%}")


if __name__ == "__main__":
    main()
