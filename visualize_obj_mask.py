from pathlib import Path
import cv2
import numpy as np

root = Path(
    "captures/session_20260812_143405/preprocess/all_data"
)

frames = sorted(root.glob("*/mask_obj1.png"))

if not frames:
    raise RuntimeError("No mask_obj1.png found")

writer = None

for mask_path in frames:

    frame_dir = mask_path.parent
    rgb_path = frame_dir / "rgb.png"

    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if rgb is None or mask is None:
        continue

    mask_bool = mask > 0

    overlay = rgb.copy()

    # mask 区域用绿色高亮
    green = np.zeros_like(rgb)
    green[:, :, 1] = 255

    overlay[mask_bool] = (
        0.55 * rgb[mask_bool]
        + 0.45 * green[mask_bool]
    ).astype(np.uint8)

    # contour
    contours, _ = cv2.findContours(
        (mask_bool.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2,
    )

    ratio = np.count_nonzero(mask) / mask.size

    cv2.putText(
        overlay,
        f"frame {frame_dir.name} | mask={ratio:.4f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if ratio == 0:
        cv2.putText(
            overlay,
            "MASK LOST",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if writer is None:
        h, w = overlay.shape[:2]
        writer = cv2.VideoWriter(
            "obj1_mask_vis.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h),
        )

    writer.write(overlay)

if writer:
    writer.release()

print("Saved: obj1_mask_vis.mp4")
