#!/usr/bin/env python3
from pathlib import Path
import argparse, shutil

ap = argparse.ArgumentParser()
ap.add_argument('--humanego', default='.')
args = ap.parse_args()
root = Path(args.humanego).resolve()
pre = root / 'preprocess'
src = pre / 'RealSensePreprocess.py'
mod_src = Path(__file__).resolve().parent / 'SAM2VideoObjectPropagator.py'
mod_dst = pre / 'SAM2VideoObjectPropagator.py'

if not src.is_file():
    raise SystemExit(f'Missing {src}')
shutil.copy2(mod_src, mod_dst)
print('[OK] installed', mod_dst)

s = src.read_text()
old_import = 'from preprocess.DINOSAMOps import run_dinosam, print_dinosam_stats'
new_import = old_import + '\n        from preprocess.SAM2VideoObjectPropagator import run_sam2_video_object_propagation'
if 'from preprocess.SAM2VideoObjectPropagator import' not in s:
    if old_import not in s:
        raise SystemExit('Could not find DINOSAMOps import')
    s = s.replace(old_import, new_import, 1)

marker = '        print_dinosam_stats()\n'
insert = '''        print_dinosam_stats()\n\n        # Repair occlusion-sensitive object masks with SAM2 video memory\n        # BEFORE KptsSelector / CoTracker.\n        run_sam2_video_object_propagation(\n            image_paths=dinosam_list,\n            cfg_path=self.cfg.DINOSAM_path,\n            mps_path=self.mps_path,\n            min_area_ratio=0.001,\n            max_area_ratio=0.60,\n            backup_dino_masks=True,\n            offload_video_to_cpu=True,\n            offload_state_to_cpu=False,\n            vos_optimized=False,\n        )\n'''
if '# Repair occlusion-sensitive object masks with SAM2 video memory' not in s:
    if marker not in s:
        raise SystemExit('Could not find print_dinosam_stats()')
    s = s.replace(marker, insert, 1)

src.write_text(s)
print('[OK] patched', src)
