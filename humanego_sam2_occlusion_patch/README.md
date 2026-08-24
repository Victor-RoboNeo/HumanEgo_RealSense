# HumanEgo SAM2 Occlusion Patch

Install from this directory:

```bash
python patch_humanego.py --humanego ~/victor/HumanEgo
```

Then:

```bash
cd ~/victor/HumanEgo
python -m py_compile preprocess/SAM2VideoObjectPropagator.py preprocess/RealSensePreprocess.py
```

Run the same preprocessing command as before.

New pipeline:

```text
per-frame DINO+SAM2
  -> choose best obj reference mask
  -> SAM2 video forward/backward propagation
  -> repaired mask_obj*.png
  -> KptsSelector
  -> CoTracker
```

Original object masks are backed up as `mask_obj1_dino.png` etc.
Diagnostics are written to `preprocess/sam2_video_object_masks.json`.
