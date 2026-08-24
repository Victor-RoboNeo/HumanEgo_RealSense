# Third-party runtimes

This folder vendors the extra runtimes HumanEgo needs so the GitHub clone is
self-contained. **Nothing here is a symlink.**

## `sam31_runtime/`

Copied from the local SAM 3.1 smoke-test package (facebookresearch/sam3 at
commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`).

| Path | What it is |
|------|------------|
| `scripts/` | HumanEgo SAM 3.1 worker (`humanego_sam31_v2.py`) + setup / checkpoint download |
| `third_party/sam3/` | Official SAM 3 source, copied as real files (no nested `.git`) |
| `checkpoints/sam3.1/` | Tokenizer / config. The gated weight `sam3.1_multiplex.pt` is **not** in git |

SAM 3.1 needs its own Python 3.12 + PyTorch 2.10 environment (separate from the
HumanEgo conda env). See [USAGE.md](../USAGE.md#31-sam-31-runtime).

```bash
cd third_party/sam31_runtime
bash scripts/setup_uv.sh
python scripts/download_checkpoint.py --source hf
```
