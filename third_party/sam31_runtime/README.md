# SAM 3.1 × HumanEgo smoke-test package

This package is deliberately **not** a HumanEgo backend replacement yet.

Its purpose is to answer, on your actual recordings:

1. Does SAM 3.1 text-conditioned segmentation separate the manipulated object
   from the hand better than the current Grounding-DINO + SAM2 pipeline?
2. Does SAM 3.1 maintain the same object identity through hand-object occlusion?
3. Does adding a small visual refinement (positive point on object, negative
   point on hand) eliminate the "object + hand merged into one mask" failure?

## Important API detail

The current official SAM 3.1 video example demonstrates:

- text-prompt concept detection/tracking
- point-prompt creation/refinement of an object
- stateful video propagation

The official SAM3 image API also supports positive/negative **box exemplars**,
but the official SAM 3.1 multiplex video notebook currently demonstrates text
+ point refinement rather than an external exemplar-image API.

Therefore this smoke-test uses the currently documented SAM 3.1 video path:

```text
text concept
    +
positive/negative visual points (optional)
    +
video tracking
```

For your strawberry/hand case this is useful: place a positive point inside the
strawberry and one or more negative points on the hand area that gets merged.

## 0. Environment

SAM3 upstream currently requires Python >= 3.12 and PyTorch >= 2.7. Keep it in
a separate environment from the current HumanEgo Python 3.11 env.

Using uv:

```bash
bash scripts/setup_uv.sh
source .venv/bin/activate
```

The setup script follows the current official README's CUDA/PyTorch install
direction and pulls the latest facebookresearch/sam3 code.

## 1. Checkpoint

The official Hugging Face model is gated. First request/accept access to:

```text
facebook/sam3.1
```

Then:

```bash
hf auth login

python scripts/download_checkpoint.py --source hf
```

Expected file:

```text
checkpoints/sam3.1/sam3.1_multiplex.pt
```

A ModelScope mirror test is also included:

```bash
python scripts/download_checkpoint.py --source modelscope
```

The script verifies that `sam3.1_multiplex.pt` actually exists; if not, use the
official gated Hugging Face route.

Set:

```bash
CKPT="$(pwd)/checkpoints/sam3.1/sam3.1_multiplex.pt"
```

## 2. One-frame text smoke test

Use the exact problematic strawberry frame:

```bash
python scripts/test_one_frame_text.py \
  --image /path/to/problem_rgb.png \
  --prompt "strawberry" \
  --checkpoint "$CKPT" \
  --output /tmp/sam31_strawberry.jpg
```

This uses the official SAM 3.1 **video predictor on a one-frame JPEG sequence**.
That avoids mixing the SAM3.1 multiplex checkpoint with an older image-only
builder path.

Look for:

- Does the mask contain only the strawberry?
- Does it exclude the hand?
- Does it return multiple strawberry-like instances?
- Is the target highest-score or largest instance?

## 3. Short HumanEgo video: text only

Start small on the RTX 4090:

```bash
python scripts/test_video_text.py \
  --session ~/victor/HumanEgo/captures/<SESSION> \
  --prompt "strawberry" \
  --checkpoint "$CKPT" \
  --max-frames 60
```

Outputs:

```text
<session>/preprocess/sam31_smoketest/
├── jpeg_cache/
├── prompt_all_instances.jpg
├── selected_track.mp4
├── selected_masks/
└── report.json
```

By default, the selected instance is the highest-scoring object on the prompt
frame. You can instead use:

```bash
--select largest
```

or explicitly:

```bash
--obj-id 3
```

### Process a specific source-frame interval

```bash
python scripts/test_video_text.py \
  --session ~/victor/HumanEgo/captures/<SESSION> \
  --prompt "strawberry" \
  --checkpoint "$CKPT" \
  --start 120 \
  --end 220 \
  --max-frames 0
```

`--prompt-local-index` refers to the local index inside the selected cache,
not the original source frame number.

## 4. Text + visual refinement

If text-only still merges the hand:

```bash
python scripts/test_video_text_refine.py \
  --session ~/victor/HumanEgo/captures/<SESSION> \
  --prompt "strawberry" \
  --checkpoint "$CKPT" \
  --max-frames 60 \
  --pos "398,238" \
  --neg "470,285;500,320"
```

Meaning:

```text
positive point(s) -> target strawberry
negative point(s) -> hand / arm pixels that must be excluded
```

You can pass multiple `--pos` / `--neg` arguments or semicolon-separated points.

Outputs:

```text
<session>/preprocess/sam31_refine_smoketest/
├── prompt_before_after.jpg
├── selected_track_refined.mp4
├── selected_masks/
└── report.json
```

## 5. Compare directly with current HumanEgo masks

After SAM3.1 tracking:

```bash
python scripts/compare_with_current_mask.py \
  --session ~/victor/HumanEgo/captures/<SESSION> \
  --sam31-masks ~/victor/HumanEgo/captures/<SESSION>/preprocess/sam31_refine_smoketest/selected_masks \
  --obj obj1
```

This produces:

```text
RGB | current DINO+SAM2 | SAM3.1
```

in one video.

## 6. What counts as a pass?

For the first smoke test, do not optimize thresholds yet. A useful pass is:

- hand and arm are not merged into the object mask
- same object ID survives hand occlusion
- no obvious identity switch to bowl/table/hand
- mask is non-empty across most manipulation frames
- visual-refinement points improve the failure frame and remain stable after
  propagation

If these pass, then it is worth writing the real HumanEgo backend:

```text
object_mask_backend: dinosam2 | sam31
```

with SAM3.1 masks exported as the same `mask_obj1.png` contract used by
KptsSelector/CoTracker.

## 7. 4090 memory note

SAM3.1's video session loads video-frame state and is substantially heavier than
your current SAM2 path. Start with 60-90 frames. If that is stable, increase the
window. The scripts intentionally default to 90 frames and support start/end
windows to make debugging safe.
