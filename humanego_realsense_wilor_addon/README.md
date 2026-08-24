# HumanEgo RealSense → WiLoR hand add-on

This package replaces the **MediaPipe hand frontend** in the RealSense version
of HumanEgo with **WiLoR**, while keeping the downstream HumanEgo interface:

```text
aria_hands.json
    ↓
AriaHandsOptimizer
    ↓
VisualKpts / virtual gripper
    ↓
DatasetGen
    ↓
HumanEgo training
```

The add-on uses HumanEgo's own `preprocess/WiLoRHands.py` as the compatibility
layer and adds **D455 metric-depth fusion**.

## Why this implementation

Current upstream HumanEgo already ships a `WiLoRHands.py` ablation that:

- calls `WiLorHandPose3dEstimationPipeline`,
- remaps WiLoR/MANO 21 joints to the Aria 21-joint ordering,
- builds the same `AriaHandData`,
- computes the midpoint/gripper frame,
- runs `AriaHandsOptimizer`.

This add-on subclasses that implementation rather than reimplementing the
HumanEgo hand schema.

The important extra change for your RealSense setup is:

```text
WiLoR relative 3D hand
        +
D455 aligned depth samples
        ↓
robust metric scale + translation fit
        ↓
metric camera-frame 21 joints
```

If D455 depth is temporarily invalid, it falls back to WiLoR's monocular
camera estimate instead of deleting the whole hand.

---

## 1. Install WiLoR support in the CURRENT HumanEgo environment

HumanEgo's current official WiLoR adapter uses `wilor-mini`.

In your activated `humanego` environment:

```bash
cd ~/victor/HumanEgo

python -m pip install --no-deps \
  "wilor-mini @ git+https://github.com/warmshao/WiLoR-mini.git"

python -m pip install \
  yacs pytorch-lightning gdown webdataset filterpy ffmpeg-python
```

If your HumanEgo installation has never installed the optional hand stack,
the repository's supported full route is:

```bash
SKIP_HAND=0 bash setup.sh
```

That also installs MediaPipe/HaMeR, so the two commands above are the smaller
WiLoR-only path.

Verify:

```bash
python - <<'PY'
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import \
    WiLorHandPose3dEstimationPipeline
print("WiLoR import: OK")
PY
```

---

## 2. Smoke-test WiLoR on ONE frame first

Do this before running 381 frames:

```bash
python scripts/smoke_test_wilor.py \
  --image captures/session_20260812_143405/preprocess/all_data/00140/rgb.png \
  --output /tmp/wilor_smoke.jpg
```

Expected:

```text
detections: 1
# or detections: 2
...
saved: /tmp/wilor_smoke.jpg
```

Open `/tmp/wilor_smoke.jpg` and make sure the 21 points are actually on the
hands.

---

## 3. Install the RealSense + WiLoR frontend

From this add-on folder:

```bash
python scripts/install.py \
  --humanego ~/victor/HumanEgo
```

This:

1. copies `preprocess/RealSenseWiLoRHands.py`;
2. backs up `RealSensePreprocess.py`;
3. replaces `RealSenseHandsGenerator` with `RealSenseWiLoRHandsGenerator`.

Check:

```bash
cd ~/victor/HumanEgo

grep -n "RealSenseWiLoRHands\|RealSenseWiLoRHandsGenerator" \
  preprocess/RealSensePreprocess.py

python -m py_compile \
  preprocess/RealSenseWiLoRHands.py \
  preprocess/RealSensePreprocess.py
```

If the patcher cannot recognize your locally modified
`RealSensePreprocess.py`, manually change:

```python
from preprocess.RealSenseHands import RealSenseHandsGenerator
```

to:

```python
from preprocess.RealSenseWiLoRHands import RealSenseWiLoRHandsGenerator
```

and:

```python
RealSenseHandsGenerator(...)
```

to:

```python
RealSenseWiLoRHandsGenerator(...)
```

The new constructor tolerates the existing RealSense `session` argument.

---

## 4. Optional config

Append the contents of:

```text
AriaHands_WiLoR_RGBD_config_snippet.yaml
```

to your existing:

```text
cfg/preprocess/base/AriaHands.yaml
```

The two most important settings are:

```yaml
realsense_wilor_conf_threshold: 0.20
realsense_wilor_min_frames: 3
```

Do **not** reuse Aria MPS's `hand_min_frames: 30` for this frontend.
Frame-wise vision hand detection can contain short gaps, and a 30-frame
minimum can erase nearly all detections.

---

## 5. Run only the frontend first

```bash
cd ~/victor/HumanEgo

python -m preprocess.RealSensePreprocess \
  --session captures/session_20260812_143405 \
  --task test \
  --pose-backend depth_icp \
  --whole-clip \
  --frontend-only
```

You should now see:

```text
RealSense WiLoR Hands: 100%|...
==========================================================
 RealSense + WiLoR hand frontend
==========================================================
frames                  : ...
wilor_detections        : ...
rgbd_fused              : ...
mono_fallback           : ...
pre_filter_right        : ...
pre_filter_left         : ...
post_filter_right       : ...
post_filter_left        : ...
==========================================================
```

Then check the actual JSON coverage:

```bash
python /path/to/addon/scripts/check_aria_hands.py \
  --session captures/session_20260812_143405
```

The previous MediaPipe result was 100% both-hands-null. The first milestone is
simply that `right exists` / `left exists` are now non-zero and that
`midpoint_pose_opt_world` starts appearing.

---

## 6. Then run full HumanEgo

Once hand JSON coverage is sane:

```bash
python -m preprocess.RealSensePreprocess \
  --session captures/session_20260812_143405 \
  --task test \
  --pose-backend depth_icp
```

`VisualKpts` should then receive non-null:

```text
midpoint_pose_opt_world
```

and the virtual gripper markers can be rendered into:

```text
rgb_WArmObjKpts.png
rgb_WoArm_WArmObjKpts.png
```

---

## RGB-D fusion details

For valid D455 joint-depth samples, the add-on computes metric points:

```text
(u_i, v_i, z_i) -> p_i^D455 in RGB camera coordinates
```

and fits:

```text
p_i^D455 ≈ s * q_i^WiLoR + t
```

with robust Huber IRLS.

It deliberately does **not** fit an extra 3D rotation because WiLoR's predicted
hand orientation/articulation should remain tied to the image evidence. Depth
mainly corrects global metric scale and translation.

Depth outliers from fingers, object occlusion, or background leakage are
downweighted and rejected. If there are too few valid depth joints, the
frontend uses WiLoR's monocular camera prediction instead of writing `null`.

---

## Roll back

The installer creates:

```text
preprocess/RealSensePreprocess.py.bak_mediapipe_hands
```

Restore with:

```bash
cp \
  preprocess/RealSensePreprocess.py.bak_mediapipe_hands \
  preprocess/RealSensePreprocess.py
```
