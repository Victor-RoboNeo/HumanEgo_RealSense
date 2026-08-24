#!/usr/bin/env python3
from pathlib import Path

p = Path("preprocess/RealSensePreprocess.py")
s = p.read_text()

old_import = "from preprocess.CamTriangulator import run_camtriagulator"
new_import = (
    "from preprocess.RealSenseCamTriangulator "
    "import run_realsense_camtriangulator"
)

old_call = "run_camtriagulator("
new_call = "run_realsense_camtriangulator("

if new_import not in s:
    if old_import not in s:
        raise SystemExit(
            "Could not find CamTriangulator import in RealSensePreprocess.py"
        )
    s = s.replace(old_import, new_import, 1)

if new_call not in s:
    if old_call not in s:
        raise SystemExit(
            "Could not find run_camtriagulator call"
        )
    s = s.replace(old_call, new_call)

p.write_text(s)
print("[OK] RealSensePreprocess now uses RGB-D triangulator")
