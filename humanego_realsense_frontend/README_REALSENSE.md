# HumanEgo RealSense Frontend — drop-in MPS replacement

This add-on is written against the current HumanEgo `main` preprocessing
interfaces.  It does **not** fake an Aria VRS/MPS folder.  Instead it replaces
the *frontend producers* while keeping the JSON/file contract used by the
released downstream pipeline.

## What is replaced

Original:

```text
sample.vrs
  ├─ Aria MPS closed-loop SLAM
  └─ Aria MPS 3D hand tracking
        ↓
AriaCam / AriaHands / AriaSlam / AriaPhases
        ↓
DINOSAM → CoTracker → CamTriangulator → DatasetGen
```

RealSense:

```text
D455 recorder session
  ├─ RGB + Depth + calibration + timestamps
  └─ IMU raw/synchronized
        ↓
PoseProvider
  ├─ depth_icp baseline
  └─ external trajectory (recommended)
        ↓
RealSenseCam
        +
MediaPipe 2D hands + D455 metric depth
        ↓
RealSenseHands
        ↓
RealSenseSlam / RealSensePhases
        ↓
SAME downstream HumanEgo stages
```

## Exact HumanEgo-compatible outputs

The frontend creates:

```text
<session>/
└── preprocess/
    ├── aria_cam_rgb_config.json
    ├── aria_phases_results.json
    ├── all_data/
    │   ├── 00000/
    │   │   ├── rgb.png
    │   │   ├── depth.png
    │   │   ├── aria_cam_rgb.json
    │   │   ├── aria_hands.json
    │   │   ├── aria_slam.json
    │   │   └── aria_phases.json
    │   └── ...
    └── realsense_frontend.json
```

Those are the filenames consumed by the released HumanEgo downstream
preprocessing and `DatasetGen`.

## 1. Record data

Use the high-FPS Linux recorder from the previous package.

For HumanEgo hand reconstruction, the preferred command is:

```bash
python realsense_rgbd_imu_highfps.py \
  --record \
  --storage raw \
  --spatial-align-depth \
  --output ./captures
```

`--spatial-align-depth` is strongly recommended for the first implementation:
it lets MediaPipe RGB landmarks directly sample metric depth in the same pixel
grid.  The frontend also has an offline pinhole depth->color projection fallback
for unaligned recordings.

## 2. Install into HumanEgo

From this add-on directory:

```bash
python scripts/install_into_humanego.py \
  --humanego /path/to/HumanEgo
```

It only adds `RealSense*.py` and `RealSense.yaml`; it does not overwrite the
released HumanEgo files.

Install additional dependencies in the HumanEgo environment:

```bash
uv pip install open3d mediapipe huggingface_hub pyyaml
```

## 3. Smoke test geometry

From the HumanEgo repo root:

```bash
PYTHONPATH=. python /path/to/addon/tests/test_geometry.py
```

Expected:

```text
RealSense frontend geometry smoke test: PASS
```

## 4. Run the frontend only

### Baseline pose: Depth ICP

```bash
python -m preprocess.RealSensePreprocess \
  --session /path/to/captures/session_YYYYMMDD_HHMMSS \
  --task serve_bread \
  --pose-backend depth_icp \
  --frontend-only
```

This is a **baseline**, not a claim that frame-to-frame ICP matches Aria MPS
closed-loop SLAM.  Inspect the resulting c2w trajectory before training.

### Recommended pose: external RGB-D/VIO/SLAM trajectory

CSV schema:

```text
timestamp_ns,tx,ty,tz,qx,qy,qz,qw
```

Each pose must be:

```text
T_color_camera_to_world
```

Run:

```bash
python -m preprocess.RealSensePreprocess \
  --session /path/to/session \
  --task serve_bread \
  --pose-backend external \
  --trajectory /path/to/trajectory.csv \
  --frontend-only
```

The provider interpolates translation and rotation onto every recorded RGB-D
timestamp.

## 5. Run full HumanEgo preprocessing

After validating the frontend:

```bash
python -m preprocess.RealSensePreprocess \
  --session /path/to/session \
  --task serve_bread \
  --pose-backend external \
  --trajectory /path/to/trajectory.csv
```

The downstream path is:

```text
RealSense frontend
→ indices
→ DINOSAM
→ KptsSelector
→ CoTracker
→ CamTriangulator
→ LaMa
→ VisualKpts
→ DatasetGen
→ training_data.json
```

Your existing `cfg/preprocess/tasks/<task>.yaml` is still required because
HumanEgo's Grounding-DINO object prompts and task-specific parameters are still
used.

## 6. Debug a short clip as all-manipulation

If SLAM phase segmentation is not tuned yet:

```bash
python -m preprocess.RealSensePreprocess \
  --session /path/to/session \
  --task serve_bread \
  --pose-backend depth_icp \
  --whole-clip \
  --frontend-only
```

This marks the recording as manipulation plus the trailing FINISHED frames.
It is useful for validating hand/camera schemas before tuning navigation phase
thresholds.

## Coordinate conventions

The add-on preserves HumanEgo's required relationship:

```text
T_c2w = T_d2w @ T_c2d
```

For native depth ICP:
- geometry pose is native depth-camera -> world
- recorded RealSense depth/color extrinsics convert it to color c2w

For aligned depth:
- ICP geometry is already on the color camera pixel grid
- first color-camera frame is world identity

Hand 3D landmarks are always represented in the color camera frame before
being transformed by c2w to world.

## Hand reconstruction

The landmark ordering is deliberately identical to HumanEgo's
`MediaPipeHands.py`:

```text
Aria 0  ThumbTip
Aria 1  IndexTip
...
Aria 5  Wrist
Aria 6  ThumbMCP
Aria 8  IndexMCP
Aria 11 MiddleMCP
...
Aria 20 PalmCenter
```

Difference from upstream MediaPipe ablation:

```text
upstream MediaPipe:
    monocular 2D + assumed physical hand scale → estimated wrist depth

this RealSense frontend:
    MediaPipe 2D landmarks + D455 depth → metric XYZ per landmark
```

The output then reuses HumanEgo's:
- `AriaHandsJointAngles`
- `MidpointFrameBuilder`
- `AriaHandsOptimizer`
- `AriaHands.save_aria_hands_json()`

so `DatasetGen` sees the fields it already expects, including optimized
midpoint hand pose and grasp state.

## Important limitation of the baseline

Aria MPS provides closed-loop globally optimized localization.  Local
frame-to-frame depth ICP does not.  For long mobile sequences, use the
`external` backend with a proper RGB-D/VIO/SLAM trajectory (e.g. your chosen
odometry stack) and keep the RealSense frontend responsible only for schema
conversion and RGB-D hand lifting.


## 7. Verify the HumanEgo contract

After `--frontend-only`, run:

```bash
python /path/to/addon/scripts/verify_frontend.py \
  --session /path/to/session
```

It checks:

```text
rgb.png
aria_cam_rgb.json
aria_hands.json
aria_slam.json
aria_phases.json
aria_phases_results.json
```

and verifies the camera transform identity used by HumanEgo:

```text
c2w == d2w @ c2d
```

Only proceed to DINOSAM/CoTracker/DatasetGen after this passes.
