#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 HumanEgo / RealSense frontend 生成的手部识别结果，并在 RGB 上叠加关键点后导出视频。

默认输入目录结构：
<session>/preprocess/all_data/00000/
    rgb.png
    aria_hands.json
<session>/preprocess/all_data/00001/
    ...

用法：
python visualize_realsense_hands_video.py \
    --session /path/to/session \
    --output /path/to/output.mp4

可选：
  --start / --end      只渲染部分帧
  --fps                手动指定输出 fps，默认自动从 metadata / aria_cam_rgb_config 推断
  --thickness          连线粗细
  --radius             点半径
  --show-index         显示关键点编号
  --max-frames         最多渲染多少帧（调试）

注意：
1. 这个脚本不重新跑 MediaPipe，只读取已经生成好的 aria_hands.json。
2. 如果某一帧没有 hand_r / hand_l，则该帧只显示 RGB。
3. 为了兼容不同 serializer，这里对多个可能字段名做了兼容处理。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Aria 21 点顺序（与前面 RealSense frontend 保持一致）
# 0  ThumbTip
# 1  IndexTip
# 2  MiddleTip
# 3  RingTip
# 4  PinkyTip
# 5  Wrist
# 6  ThumbMCP
# 7  ThumbIP
# 8  IndexMCP
# 9  IndexPIP
# 10 IndexDIP
# 11 MiddleMCP
# 12 MiddlePIP
# 13 MiddleDIP
# 14 RingMCP
# 15 RingPIP
# 16 RingDIP
# 17 PinkyMCP
# 18 PinkyPIP
# 19 PinkyDIP
# 20 PalmCenter
ARIA_HAND_EDGES = [
    # thumb
    (5, 6), (6, 7), (7, 0),
    # index
    (5, 8), (8, 9), (9, 10), (10, 1),
    # middle
    (5, 11), (11, 12), (12, 13), (13, 2),
    # ring
    (5, 14), (14, 15), (15, 16), (16, 3),
    # pinky
    (5, 17), (17, 18), (18, 19), (19, 4),
    # palm support
    (20, 8), (20, 11), (20, 14), (20, 17), (20, 5),
]

RIGHT_COLOR = (60, 200, 255)   # BGR
LEFT_COLOR = (255, 120, 80)    # BGR
TEXT_BG = (20, 20, 20)
TEXT_FG = (240, 240, 240)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_fps(session_dir: Path) -> float:
    candidates = [
        session_dir / "metadata.json",
        session_dir / "preprocess" / "aria_cam_rgb_config.json",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data = load_json(p)
            for key in ("video_fps", "fps"):
                if key in data and data[key] is not None:
                    return float(data[key])
        except Exception:
            pass
    return 30.0


def get_frame_dirs(session_dir: Path) -> List[Path]:
    all_data = session_dir / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"未找到目录: {all_data}")
    frame_dirs = sorted([p for p in all_data.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise RuntimeError(f"{all_data} 下没有任何帧目录")
    return frame_dirs


def _to_xy_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        return None
    if arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        return None
    return arr


def extract_hand_keypoints_2d(hand_data: Dict[str, Any]) -> Optional[np.ndarray]:
    candidate_keys = [
        "hand_keypoints_2d",
        "keypoints_2d",
        "landmarks_2d",
        "kpts_2d",
    ]
    for key in candidate_keys:
        if key in hand_data and hand_data[key] is not None:
            arr = _to_xy_array(hand_data[key])
            if arr is not None and len(arr) >= 21:
                return arr[:21]
    return None


def extract_confidence(hand_data: Dict[str, Any]) -> Optional[float]:
    for key in ("confidence", "score", "conf"):
        if key in hand_data and hand_data[key] is not None:
            try:
                return float(hand_data[key])
            except Exception:
                pass
    return None


def extract_grasp_state(hand_data: Dict[str, Any]) -> Optional[int]:
    for key in ("grasp_state", "grasp", "is_grasp"):
        if key in hand_data and hand_data[key] is not None:
            try:
                return int(hand_data[key])
            except Exception:
                pass
    return None


def draw_label(img: np.ndarray, text: str, x: int, y: int):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x = max(0, x)
    y = max(th + baseline + 2, y)
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 8, y + 4), TEXT_BG, -1)
    cv2.putText(img, text, (x + 4, y - 2), font, scale, TEXT_FG, thick, cv2.LINE_AA)


def draw_hand(
    img: np.ndarray,
    kpts: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
    confidence: Optional[float],
    grasp_state: Optional[int],
    thickness: int = 2,
    radius: int = 3,
    show_index: bool = False,
):
    h, w = img.shape[:2]
    valid = np.isfinite(kpts).all(axis=1)

    # edges
    for i, j in ARIA_HAND_EDGES:
        if i >= len(kpts) or j >= len(kpts):
            continue
        if not (valid[i] and valid[j]):
            continue
        x1, y1 = map(int, np.round(kpts[i]))
        x2, y2 = map(int, np.round(kpts[j]))
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # points
    for idx, (x, y) in enumerate(kpts):
        if not valid[idx]:
            continue
        x_i, y_i = int(round(float(x))), int(round(float(y)))
        if 0 <= x_i < w and 0 <= y_i < h:
            cv2.circle(img, (x_i, y_i), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x_i, y_i), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
            if show_index:
                cv2.putText(
                    img,
                    str(idx),
                    (x_i + 4, y_i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    # label near wrist / palm center
    anchor_candidates = [5, 20, 11, 8]
    anchor = None
    for a in anchor_candidates:
        if a < len(kpts) and valid[a]:
            anchor = kpts[a]
            break
    if anchor is not None:
        parts = [label]
        if confidence is not None:
            parts.append(f"{confidence:.2f}")
        if grasp_state is not None:
            parts.append("grasp" if grasp_state > 0 else "open")
        draw_label(img, " | ".join(parts), int(anchor[0]) + 8, int(anchor[1]) - 8)


def draw_header(img: np.ndarray, text: str):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_video(
    session_dir: Path,
    output_path: Path,
    fps: float,
    start: int,
    end: Optional[int],
    thickness: int,
    radius: int,
    show_index: bool,
    max_frames: int,
):
    frame_dirs = get_frame_dirs(session_dir)
    frame_dirs = frame_dirs[start : (end + 1) if end is not None else None]
    if max_frames > 0:
        frame_dirs = frame_dirs[:max_frames]
    if not frame_dirs:
        raise RuntimeError("筛选后没有可渲染的帧")

    first_rgb = cv2.imread(str(frame_dirs[0] / "rgb.png"), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"无法读取 {frame_dirs[0] / 'rgb.png'}")

    h, w = first_rgb.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    total = len(frame_dirs)
    print(f"[INFO] 输出视频: {output_path}")
    print(f"[INFO] 帧数: {total}, 分辨率: {w}x{h}, fps: {fps:.2f}")

    for idx, frame_dir in enumerate(frame_dirs):
        rgb_path = frame_dir / "rgb.png"
        hands_path = frame_dir / "aria_hands.json"

        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 跳过，无法读取: {rgb_path}")
            continue

        if hands_path.is_file():
            try:
                hands = load_json(hands_path)
            except Exception as e:
                print(f"[WARN] 跳过手部 JSON 读取失败: {hands_path}: {e}")
                hands = {}
        else:
            hands = {}

        # 兼容两种可能命名
        hand_r = hands.get("hand_r") or hands.get("right")
        hand_l = hands.get("hand_l") or hands.get("left")

        if isinstance(hand_r, dict):
            kpts = extract_hand_keypoints_2d(hand_r)
            if kpts is not None:
                draw_hand(
                    img,
                    kpts,
                    RIGHT_COLOR,
                    label="Right",
                    confidence=extract_confidence(hand_r),
                    grasp_state=extract_grasp_state(hand_r),
                    thickness=thickness,
                    radius=radius,
                    show_index=show_index,
                )

        if isinstance(hand_l, dict):
            kpts = extract_hand_keypoints_2d(hand_l)
            if kpts is not None:
                draw_hand(
                    img,
                    kpts,
                    LEFT_COLOR,
                    label="Left",
                    confidence=extract_confidence(hand_l),
                    grasp_state=extract_grasp_state(hand_l),
                    thickness=thickness,
                    radius=radius,
                    show_index=show_index,
                )

        draw_header(img, f"frame {start + idx:05d} / {start + total - 1:05d}")
        writer.write(img)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"[INFO] 已渲染 {idx + 1}/{total}")

    writer.release()
    print("[DONE] 视频导出完成")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True, help="session 根目录")
    parser.add_argument("--output", type=str, required=True, help="输出 mp4 路径")
    parser.add_argument("--fps", type=float, default=None, help="输出 fps；默认自动推断")
    parser.add_argument("--start", type=int, default=0, help="起始帧 index")
    parser.add_argument("--end", type=int, default=None, help="结束帧 index（包含）")
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--show-index", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="最多渲染多少帧，仅调试用")
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = Path(args.session).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    fps = float(args.fps) if args.fps is not None else infer_fps(session_dir)

    render_video(
        session_dir=session_dir,
        output_path=output_path,
        fps=fps,
        start=args.start,
        end=args.end,
        thickness=args.thickness,
        radius=args.radius,
        show_index=args.show_index,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
