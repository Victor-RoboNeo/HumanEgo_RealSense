# HumanEgo RealSense 复现

在 [HumanEgo](https://github.com/TX-Leo/HumanEgo)（Wang et al., 2026）上的本地复现：用 **Intel RealSense D455 RGB-D** 采集第一人称操作视频，不再依赖 Project Aria + MPS。

下游训练接口保持不变（`training_data.json` + flow-matching 策略）。改动集中在前端：相机位姿、手、物体 mask。

```
D455 RGB + Depth + IMU
  -> 相机运动     depth ICP（或外部轨迹）
  -> 手 21 关节   WiLoR + 对齐深度尺度/平移
  -> 物体 / 手臂  SAM 3.1 文本 + 视频跟踪
  -> 与原版相同的 CoTracker / 三角化 / LaMa / DatasetGen
  -> HumanEgo 策略训练
```

相对原版的三处替换：

| 原版 | 本仓库 |
|------|--------|
| Aria VRS + MPS SLAM | RealSense session + `depth_icp` |
| Aria MPS 手跟踪 | WiLoR 相对 3D 手 + D455 度量深度融合 |
| Grounding DINO + SAM 2 | SAM 3.1（`third_party/sam31_runtime`，真实文件，不是软链接） |

当前任务配置是 **`pick_strawberry`**（`obj1`: strawberry，`obj2`: pink plate）。

完整参数表见 [USAGE.md](USAGE.md)。

## 管线

1. **Frontend**（`preprocess.RealSensePreprocess`）  
   从 session 读 RGB / 对齐深度 / 标定。`PoseProvider` 用相邻帧深度 ICP 估相机运动（`--pose-backend depth_icp`）。WiLoR 出相对 3D 手，再用 D455 深度做鲁棒尺度+平移；深度失效时回退到 WiLoR 单目估计，不整手清空。输出仍是 HumanEgo 合同文件：`aria_cam_rgb.json`、`aria_hands.json`、`aria_slam.json`、`aria_phases.json`。

2. **SAM 3.1 masks**  
   文本提示分割物体和手臂，视频跟踪保持 id。权重在独立 Python 3.12 / uv 环境里跑，由 `preprocess/sam31/run_sam31.sh` 隔离调用，避免和 HumanEgo 的 conda CUDA 混装。写出 `mask_obj*.png` / `mask_arm.png`。

3. **Downstream**  
   KptsSelector → CoTracker → 深度抬升 / 三角化 → LaMa 抹手臂 → VisualKpts 画虚拟夹爪 → DatasetGen。抓取后把物体 6DoF 锁到手并向前传播（kinematic latch）。

4. **Train**  
   `training.FlowMatchingTrainer`，ICT + flow matching，配置 `cfg/training/pick_strawberry/HumanEgo.yaml`。

## 安装

需要 NVIDIA GPU（驱动 ≥ 525 / CUDA 12.x）。**两套环境不能混用。**

```bash
# 1) HumanEgo：Python 3.11 + WiLoR + RealSense
conda create -n humanego python=3.11 -y
conda activate humanego
SKIP_HAND=0 SKIP_HARDWARE=0 bash setup.sh

# 2) SAM 3.1：Python 3.12 + uv（仓库内已 vendoring sam3 源码）
# 先申请 https://huggingface.co/facebook/sam3.1 权限
cd third_party/sam31_runtime
bash scripts/setup_uv.sh
hf auth login
python scripts/download_checkpoint.py --source hf
# 期望：checkpoints/sam3.1/sam3.1_multiplex.pt
```

`setup.sh` 默认跳过手部和相机驱动。本复现必须 `SKIP_HAND=0 SKIP_HARDWARE=0`。

## 跑通 pick_strawberry

session 目录来自 D455 采集（见 [`humanego_realsense_frontend/README_REALSENSE.md`](humanego_realsense_frontend/README_REALSENSE.md)）。

```bash
conda activate humanego

# 先只跑手部前端，确认 aria_hands.json 不是双手全 null
python -m preprocess.RealSensePreprocess \
  --session /path/to/captures/session_YYYYMMDD_HHMMSS \
  --task pick_strawberry \
  --pose-backend depth_icp \
  --whole-clip \
  --frontend-only

python scripts/check_aria_hands.py \
  --session /path/to/captures/session_YYYYMMDD_HHMMSS

# 全流程：SAM 3.1 → CoTracker → DatasetGen
python -m preprocess.RealSensePreprocess \
  --session /path/to/captures/session_YYYYMMDD_HHMMSS \
  --task pick_strawberry \
  --pose-backend depth_icp \
  --whole-clip

# 训练（至少 2 条 session；编号 000 默认留作 eval）
python -m training.FlowMatchingTrainer \
  --task pick_strawberry --use_cfg --job HumanEgo
```

`--whole-clip` 把整段当作 manipulation，适合短 clip，不做 Aria 式 nav / transition 切分。

任务提示在 `cfg/preprocess/tasks/pick_strawberry.yaml`。WiLoR 段最短帧数不要沿用 Aria 的 `hand_min_frames: 30`，见 `cfg/preprocess/base/AriaHands.yaml` 里的 `realsense_wilor_*`。

## 仓库

| 路径 | 作用 |
|------|------|
| `preprocess/RealSensePreprocess.py` | RealSense 入口：前端 + SAM 3.1 + 下游 |
| `preprocess/RealSenseWiLoRHands.py` | WiLoR + D455 深度融合 |
| `preprocess/sam31/` | 调用隔离的 SAM 3.1 worker |
| `third_party/sam31_runtime/` | SAM 3 源码与脚本（实文件，无软链接） |
| `cfg/preprocess/tasks/pick_strawberry.yaml` | 草莓任务检测提示 |
| `cfg/training/pick_strawberry/HumanEgo.yaml` | 训练配方 |
| `humanego_realsense_frontend/` | 采集与前端说明 |
| `USAGE.md` | 每个 CLI / YAML 参数 |

`data/`、`runs/`、`*.pt` 不进 git。SAM 3.1 权重约 3.3 GB，克隆后按上面步骤下载。

## 许可

上游 HumanEgo 为 [PolyForm Noncommercial 1.0.0](LICENSE)。SAM 3 源码见 `third_party/sam31_runtime/third_party/sam3/LICENSE`。方法请同时引用原论文：

<a id="cite"></a>

```bibtex
@misc{humanego2026,
  title         = {HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos},
  author        = {Wang, Zhi and He, Botao and Yu, Kelin and Lee, Seungjae and Gao, Ruohan and Huang, Furong and Aloimonos, Yiannis},
  year          = {2026},
  eprint        = {2605.24934},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
