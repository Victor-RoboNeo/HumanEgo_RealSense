# HumanEgo 使用文档

本文是本仓库的参数手册，配合根目录 [README.md](README.md)（RealSense + WiLoR + SAM 3.1 复现说明）使用。方法基于 [TX-Leo/HumanEgo](https://github.com/TX-Leo/HumanEgo)；SAM 3.1 运行时在 `third_party/sam31_runtime`（真实文件，不是软链接）。

更细的阶段说明还在：

- 预处理产物：[`preprocess/README.md`](preprocess/README.md)
- 训练：[`training/README.md`](training/README.md)
- 真机推理模板：[`inference/README.md`](inference/README.md)
- Aria 采集：[`datacollection/README.md`](datacollection/README.md)

---

## 1. 仓库结构

```
HumanEgo/
├── setup.sh / requirements.txt     # conda 环境一键安装
├── cfg/                            # 所有 YAML：预处理 / 训练 / 推理 / 采集
├── preprocess/                     # Aria + RealSense 预处理
├── training/                       # flow-matching 策略训练
├── inference/                      # 真机部署模板（需相机+机械臂）
├── datacollection/                 # Aria MPS 提交
├── scripts/                        # 下数据、WiLoR 冒烟测试
├── third_party/sam31_runtime/      # 已 vendoring 的 SAM 3.1（真实目录）
├── data/                           # 数据集（git 忽略，需自行下载）
└── runs/                           # 训练输出（git 忽略）
```

**不会进 GitHub 的内容：** `data/`、`runs/`、RealSense 采集、SAM 3.1 权重
`sam3.1_multiplex.pt`（约 3.3 GB，需自行申请后下载）。

---

## 2. 环境安装

需要 NVIDIA GPU、驱动 ≥ 525（CUDA 12.x）。HumanEgo 主环境是 **Python 3.11**；
SAM 3.1 必须另开 **Python 3.12** 的 uv 环境，两者不能混用。

```bash
conda create -n humanego python=3.11 -y
conda activate humanego
bash setup.sh
```

### 2.1 `setup.sh` 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `SKIP_HARDWARE` | `1` | `1`：不装 `pyrealsense2`、`trossen-arm`。真机采集/部署设 `0` |
| `SKIP_HAND` | `1` | `1`：不装 MediaPipe / WiLoR / HaMeR。RealSense+WiLoR 或手部消融设 `0` |
| `PREDOWNLOAD` | `0` | `1`：立刻从 HuggingFace 拉手部/朝向等权重；否则第一次跑再下 |
| `PIP_INDEX_URL` | 清华镜像 | pip 源；可覆盖 |

```bash
SKIP_HAND=0 bash setup.sh                         # + 手部追踪
SKIP_HARDWARE=0 SKIP_HAND=0 bash setup.sh         # 完整（相机+机械臂+手）
PREDOWNLOAD=1 bash setup.sh                       # 预下载权重
```

`setup.sh` 还会：按特殊顺序装 numpy/chumpy、用 `--no-deps` 钉死
`projectaria-tools==1.7.1`、从 GitHub 装 CoTracker 与 Orient-Anything V2，
并给 chumpy / HaMeR 打 Python 3.11 兼容补丁。

---

## 3. SAM 3.1 runtime（本仓库 vendoring）

RealSense 预处理里的物体/手臂 mask 走 SAM 3.1。代码已在
`third_party/sam31_runtime/`，**不是软链接**。权重仍需单独下载（gated）。

### 3.1 安装独立环境

需要先装 [uv](https://docs.astral.sh/uv/)。

```bash
cd third_party/sam31_runtime
bash scripts/setup_uv.sh          # Python 3.12 venv + torch 2.10 + 可编辑安装 sam3
```

`setup_uv.sh` 变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `VENV` | `<runtime>/.venv` | uv 虚拟环境路径 |
| `SAM3_ROOT` | `<runtime>/third_party/sam3` | 已拷贝的 sam3 源码 |
| `PYTHON_VERSION` | `3.12` | SAM3 要求 ≥ 3.12 |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu128` | PyTorch 2.10 cu128 索引 |

### 3.2 下载权重

先在 HuggingFace 申请 [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) 权限，再：

```bash
hf auth login
python third_party/sam31_runtime/scripts/download_checkpoint.py --source hf
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--source` | `hf` | `hf`：官方 HuggingFace；`modelscope`：国内镜像（可能没有 multiplex 权重） |

期望文件：`third_party/sam31_runtime/checkpoints/sam3.1/sam3.1_multiplex.pt`。

HumanEgo 通过 `preprocess/sam31/run_sam31.sh` 在隔离环境里调用 worker，会清掉父进程的
`LD_LIBRARY_PATH` / `PYTHONPATH`，避免 conda CUDA 与 uv CUDA 混装。

---

## 4. 数据

### 4.1 下载官方数据集

```bash
python scripts/download_data.py --task serve_bread --num 2 --input-only
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `--task` | `serve_bread` | `serve_bread` / `water_flowers` / `all` |
| `--num` | `1` | 每个任务取前 N 条；`all` 表示全部 |
| `--input-only` | 关 | 只要原始 VRS+MPS，自己跑预处理 |
| `--out` | `./data` | 下载目录 |
| `--keep-tar` | 关 | 解压后保留 `all_data.tar` |

数据布局：`data/<task>/aria/mps_<task>_<id>_vrs/`。

### 4.2 自己采 Aria

见 [`datacollection/README.md`](datacollection/README.md)。`python -m datacollection.AriaMPS --vrs_file ./data/NAME.vrs` 会读 `cfg/datacollection/AriaMPS.yaml`。

### 4.3 RealSense 采集

见 [`humanego_realsense_frontend/README_REALSENSE.md`](humanego_realsense_frontend/README_REALSENSE.md)。session 目录直接作为 `--session` 传给 `preprocess.RealSensePreprocess`。

---

## 5. 预处理 — Aria（官方路径）

```bash
python -m preprocess.Preprocess \
  --mps_path ./data/serve_bread/aria/mps_serve_bread_000_vrs \
  --task serve_bread
```

阶段顺序：`aria` → `indices` → `dinosam`（Grounding DINO + SAM2）→ `kptsselector` → `cotracker` → `camtriangulator` → `lama` → `visualkpts` → `datasetgen`。

### 5.1 CLI

| 参数 | 默认 | 含义 |
|------|------|------|
| `--mps_path` | 必填 | 单条 `mps_*_vrs` 目录，或含多条 recording 的父目录（批量） |
| `--cfg_path` | `./cfg/preprocess/base/Preprocess.yaml` | 主配置，指向各子模块 YAML |
| `--task` | `None` | 读取 `cfg/preprocess/tasks/<task>.yaml` 并深合并到 base。缺少该文件时仍能跑，但 `dinosam_prompt` 是占位符 `"a pretty cat ."`，检测不到你的物体 |
| `--range START END` | 无 | 批量模式下只处理编号在 `[START, END)` 的 recording |
| `--no-video` | 关 | 不导出 mp4 |
| `--no-gif` | 关 | 不导出 gif |

### 5.2 `cfg/preprocess/base/Preprocess.yaml`

| 字段 | 含义 |
|------|------|
| `AriaCam_path` 等 `*_path` | 各阶段 YAML 的相对路径 |
| `hand_tracking_methods` | 要跑的手部方法。发布默认只有 `["aria_mps"]`。`mediapipe` / `wilor` / `hamer` 是消融：需 `SKIP_HAND=0 bash setup.sh`，并在 `Preprocess.py` 里取消注释 `self.preprocess_hands()` |

任务覆盖写在 `cfg/preprocess/tasks/<task>.yaml`，只写与 base 不同的字段。

---

## 6. 预处理 — RealSense

```bash
python -m preprocess.RealSensePreprocess \
  --session /path/to/captures/session_YYYYMMDD_HHMMSS \
  --task pick_strawberry \
  --pose-backend depth_icp \
  --whole-clip
```

物体/手臂 mask 走 SAM 3.1（不是 DINO+SAM2）。手部默认 WiLoR + D455 深度融合。

### 6.1 CLI

| 参数 | 默认 | 含义 |
|------|------|------|
| `--session` | 必填 | RealSense 采集 session 根目录 |
| `--cfg_path` | `./cfg/preprocess/base/Preprocess.yaml` | 下游阶段（kpts / cotracker / lama / datasetgen）配置 |
| `--task` | `None` | 同 Aria：`cfg/preprocess/tasks/<task>.yaml` |
| `--pose-backend` | `depth_icp` | `depth_icp`：用相邻帧深度 ICP 估相机运动；`external`：读外部轨迹文件 |
| `--trajectory` | `None` | `--pose-backend external` 时的轨迹路径 |
| `--frontend-cfg` | `./cfg/preprocess/base/RealSense.yaml` | ICP / 手部前端专用配置 |
| `--whole-clip` | 关 | 把整段当作 manipulation（不做 Aria 式 nav/transition 切分），适合短 clip 调试 |
| `--frontend-only` | 关 | 只跑相机/手/SLAM/phase 前端，停在 `aria_hands.json`，不跑 SAM / CoTracker |

只跑手部前端：

```bash
python -m preprocess.RealSensePreprocess --session <SESSION> --task test \
  --pose-backend depth_icp --whole-clip --frontend-only
```

### 6.2 `cfg/preprocess/base/RealSense.yaml`

**ICP（`icp:`）**

| 字段 | 默认 | 含义 |
|------|------|------|
| `voxel_size` | `0.025` | 点云下采样体素边长（米） |
| `max_correspondence` | `0.08` | ICP 最大对应点距离（米） |
| `stride` | `3` | 每隔多少帧做一次 ICP |
| `min_depth` / `max_depth` | `0.20` / `4.0` | 参与 ICP 的深度范围（米） |
| `max_iteration` | `25` | ICP 迭代次数 |
| `min_fitness` | `0.12` | 低于此 fitness 视为失败 |

**手（`hands:`）**

| 字段 | 默认 | 含义 |
|------|------|------|
| `model_path` | `null` | MediaPipe 手模型路径；`null` 则用仓库权重或自动下载 |
| `local_depth_radius` | `5` | 每个 2D 关键点取深度时的邻域半径（像素） |
| `min_valid_depth_landmarks` | `5` | 有效深度关键点少于此数则该手丢弃 |

WiLoR 阈值在 `AriaHands.yaml`（见 §7.2）。

---

## 7. 预处理 YAML — 每个字段

路径：`cfg/preprocess/base/<Module>.yaml`。任务文件只覆盖需要改的键。

### 7.1 `AriaCam.yaml` — 去畸变虚拟针孔

| 字段 | 含义 |
|------|------|
| `fps` | 输出元数据/视频帧率 |
| `cam_size.rgb/slam_l/slam_r.{w,h,fov}` | 去畸变图分辨率与垂直视场（度）。焦距 `f = W / (2 tan(fov/2))` |
| `cam_labels.*` | VRS 里的流名称，必须与 `.vrs` 内部 label 一致 |

### 7.2 `AriaHands.yaml` — 手轨迹滤波 / 抓取

| 字段 | 默认 | 含义 |
|------|------|------|
| `grasp_threshold` | `0.105` | 指尖距离（米）低于此视为 grasp。按物体尺寸调 |
| `grasp_smooth_win` | `5` | 指尖距离滑动平均窗口 |
| `grasp_flicker_max_len` | `5` | 抑制 0-1-0 闪烁的最大长度（帧） |
| `hand_conf_threshold` | `0.8` | MPS 置信度下限 |
| `hand_min_frames` | `30` | 短于此的检测段当 ghost 丢掉。**RealSense+WiLoR 不要用 30**，见下面两项 |
| `hand_interp_max_gap` | `10` | 线性/SLERP 插值填洞的最大间隔（帧） |
| `hand_selection` | `"none"` | `auto`（最高置信）/ `left` / `right` / `none`（双手都留） |
| `smooth_sg_window` | `21` | Savitzky–Golay 窗口，必须奇数 |
| `smooth_sg_polyorder` | `2` | SG 多项式阶 |
| `smooth_ema_alpha_x` / `_y` | `0.15` | EMA：越接近 0 越平滑 |
| `smooth_min_valid_frames` | `6` | 少于此点数不平滑 |
| `smooth_fill_max_gap` | `10` | 平滑前预插值的最大空洞 |
| `opt_v_limit` | `0.6` | 优化器线速度上限（m/s） |
| `opt_w_limit` | `0.8` | 优化器角速度上限（rad/s） |
| `realsense_wilor_min_frames` | `30` | WiLoR 段最短帧数。视觉检测有空洞时建议改小（如 `3`） |
| `realsense_wilor_interp_gap` | `0` | WiLoR 插值最大空洞 |
| `realsense_wilor_conf_threshold` | （snippet） | WiLoR 检测置信阈值，见 `AriaHands_WiLoR_RGBD_config_snippet.yaml`，建议 `0.20` |

### 7.3 `AriaPhases.yaml` — Navigation / Transition / Manipulation

| 字段 | 含义 |
|------|------|
| `v_stop_thresh` | 线速度低于此（m/s）可能是停止 |
| `w_stop_thresh` | 角速度低于此（rad/s）认为头几乎没转 |
| `stop_hold_frames` | 连续低速多少帧才开始标 stop |
| `stop_debounce_frames` | stop 内允许的速度毛刺帧数 |
| `stop_min_on_frames` | stop 段最短长度，更短的删掉 |
| `stop_yaw_veto_deg` | stop 期间 yaw 变化超过此（度）则取消 stop（排除扫视） |
| `stop_offset_frames` | stop 起点再延后若干帧，等减速结束 |
| `transition_offset_frames` | 模式切换时插入的 Transition 缓冲 |
| `mode_min_run_frames` | 任一模式最短持续，防闪烁 |
| `w_rot_thresh` / `v_rot_max` | 原地旋转：角速度下限、线速度上限 |
| `v_walk_thresh` | 走路统计用的速度阈值 |
| `manip_clean_vel_thresh` | 手速过快则把 MANIP 改成 Transition |
| `manip_clean_wait_frames` | 检查手速的稳定窗口 |
| `manip_clean_manual_offset` | 从 MANIP 边界再裁掉的帧数 |
| `finished_frames` | 演示末尾标 `is_finished=1.0` 的帧数 |

### 7.4 `AriaSlam.yaml`

| 字段 | 含义 |
|------|------|
| `traj_future_len` | 向前投影的未来帧数 |
| `traj_step` | 轨迹点采样步长 |
| `ground_offset` | 把轨迹放到地面的垂直偏移（米） |
| `sensitivity_range` | mini-radar UI 范围（米） |

### 7.5 `DINOSAM.yaml` — 检测分割 / SAM 3.1

| 字段 | 含义 |
|------|------|
| `dino_model_id` | Grounding DINO HuggingFace id |
| `box_threshold` | DINO 框置信阈值。漏检就降低（如 `0.15`） |
| `sam2_repo_id` / `sam2_checkpoint_name` / `sam2_config` | SAM2 模型与配置 |
| `dinosam_prompt.obj1, obj2, …` | **必填。** 每个被操作物体的开放词表提示，短语末尾加空格+句点，如 `"strawberry ."`。键名成为全程 object id |
| `dinosam_prompt.arm` | 要分割并 inpaint 掉的手/臂，如 `"human arms . human hands ."` |
| `sam31_threshold` | SAM 3.1 实例分数阈值（仅 RealSense 路径） |
| `sam31_save_debug_video` | 是否写 `preprocess/.sam31_v2/` 调试视频 |

### 7.6 `KptsSelector.yaml` — 物体轮廓关键点

| 字段 | 含义 |
|------|------|
| `kpts_patch_close` / `kpts_close_kernel` | 形态学闭运算，填 mask 内孔 |
| `kpts_patch_erode` / `kpts_erode_kernel` / `kpts_erode_iters` | 腐蚀去毛刺 |
| `kpts_use_inner_edge` / `kpts_edge_erode_kernel` | 把跟踪点往物体内部收一圈，CoTracker 更稳 |
| `kpts_min_edge_pixels` | 内收后像素太少则回退到未腐蚀 mask |
| `kpts_n_bands` | 关键点数量 = `kpts_n_bands * 2` |

### 7.7 `CoTracker.yaml`

| 字段 | 含义 |
|------|------|
| `cotracker_res` | 模型内部方形分辨率。更大更准、更吃显存 |
| `cotracker_chunk_size` | 一次送入的连续帧数。OOM 就降到 50/30 |
| `object_centric_max_frames` | 操作开始前最多保留的接近（Nav）帧 |
| `object_centric_min_frames` | 物体中心窗口最短长度；Nav 太短会从 MANIP 借帧 |
| `ref_idx` | 初始化 query 的帧。`0` 第一帧；`-1` 为 Nav 最后一帧（手触物前、最近且未遮挡） |
| `cotracker_viz_trail_len` | 可视化拖尾长度（帧） |

### 7.8 `CamTriangulator.yaml` — 2D 轨迹抬到 3D

| 字段 | 含义 |
|------|------|
| `step` | 多视图抽样步长。越大越快、3D 越稀 |
| `smooth_window` / `smooth_polyorder` | 2D 轨迹 SG 滤波 |
| `ba_f_scale` | 点 BA 的 Huber 尺度，控制外点抑制 |
| `pose_method.{obj1,…,default}` | 朝向：`vlm`（Orient-Anything，语义）/ `pca1`（星形几何）/ `pca2`（物体中心，可处理立着/倒下） |
| `pca_aniso_thresh` | 仅 `pca1`：各向异性阈值，区分细长 vs 对称物体 |
| `axes_len_m` / `point_radius_m` / `line_radius_m` | PLY 可视化尺寸 |

### 7.9 `DepthLifter.yaml` — RealSense 单目深度抬升

| 字段 | 含义 |
|------|------|
| `pose_method` | 同 CamTriangulator：`vlm` / `pca1` / `pca2` |
| `depth_median_frames` | 对可见帧深度做中值；`1` 关闭 |
| `min_valid_depth_m` / `max_valid_depth_m` | 有效深度范围（米） |
| `pca_anisotropy_threshold` | 细长 vs 对称 |
| `point_radius_m` / `axes_len_m` | 可视化 |

### 7.10 `Lama.yaml` — 抹掉手臂

| 字段 | 含义 |
|------|------|
| `lama_repo_id` / `lama_filename` | HuggingFace 上的 ONNX 权重 |
| `lama_mask_dilation` | mask 膨胀核。残留影子/轮廓就加大；旁边有不能擦的结构就减小 |
| `lama_input_size` | 网络内部方形分辨率，默认 512 |

### 7.11 `VisualKpts.yaml` — 虚拟夹爪叠加

| 字段 | 含义 |
|------|------|
| `visualkpts_gripper_width_max` / `_min` | 张开/闭合夹爪宽度（米） |
| `visualkpts_gripper_finger_len` / `_root_depth` | 手指长度、腕到指根深度 |
| `visualkpts_pts_per_line` | 每根“骨头”上的点密度 |
| `visualkpts_trail_*` | 历史拖尾长度、步长、透明度、线宽 |
| `visualkpts_obj_motion_th` | 物体拖尾的像素位移阈值，避免静物抖尾 |
| `visualkpts_radius_current` / `visualkpts_white_core_radius` | 关键点半径与白芯 |
| `color_hand_*` / `color_hand_closed_*` | 张开/闭合手的 BGR 颜色 |
| `obj_colors` / `color_object_trail` | 多物体与拖尾颜色 |

### 7.12 `DatasetGen.yaml` — 写出 `training_data.json`

| 字段 | 含义 |
|------|------|
| `min_hand_confidence` | 低于此不写入该帧手数据 |
| `save_debug_raw` | JSON 里是否再存调试矩阵 |
| `disable_kinematic_latching_left` / `_right` | `false`：抓住后把物体 6DoF 锁到手并向前传播（遮挡下轨迹仍连续）。`true`：物体永远停在初始三角化位姿 |
| `export_obj_and_traj_ply` / `ply_show_object_kpts` | 是否写 `object_centric.ply` |
| `obj_kpt_radius_m` / `obj_line_radius_m` / `obj_axes_len_m` | 物体点/轴可视化 |
| `traj_use_hand` | `both` / `right_first` / `left_first` / `right_only` / `left_only` |
| `traj_stride` / `traj_max_points` | 轨迹上 mini-axis 间隔、点数上限 |
| `traj_point_radius_m` / `traj_line_radius_m` / `traj_axes_len_m` | 轨迹可视化 |
| `export_object_centric_png` | 是否写诊断 PNG |
| `png_view_elev` / `png_view_azim` | 3D 图相机仰角/方位角 |
| `png_fixed_range_m` | 坐标轴固定范围（米），保证跨序列可比 |
| `png_dpi` / `png_fill_ratio` | PNG 分辨率与轨迹占窗比例 |

### 7.13 `RobotPreprocess.yaml` / `RobotDatasetGen.yaml`

机器人域预处理主配置，只声明子模块路径。`RobotDatasetGen`：

| 字段 | 含义 |
|------|------|
| `gripper_grasp_threshold` | `grasp > 阈值` 视为抓住（`grasp = 1 - gripper_q`） |
| `disable_kinematic_latching_*` | 同 DatasetGen |
| `finished_tail_frames` | 若采集时没标 finished，自动把最后 N 帧标 1.0；`0` 关闭 |
| 其余 `ply_*` / `traj_*` / `png_*` | 同 DatasetGen |

### 7.14 任务覆盖示例

已提供：`serve_bread`、`water_flowers`、`pick_strawberry`、`test`。新建任务：

1. 写 `cfg/preprocess/tasks/<your_task>.yaml`（至少 `DINOSAM.dinosam_prompt`）
2. 写 `cfg/training/<your_task>/HumanEgo.yaml`（可从 `serve_bread` 拷）
3. 数据放到 `data/<your_task>/aria/mps_<your_task>_<id>_vrs/`（至少 2 条，`000` 默认留作 eval）

---

## 8. 训练

```bash
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo
```

解析顺序：**TrainConfig 默认值 → YAML（`--use_cfg`）→ CLI**。输出在
`runs/<task>/<job>/`（有 `--exp` 时为 `runs/<task>/<exp>/<job>/`）。存在 `latest.pt` 会自动 resume。

`data_sources: {aria: N}` 时自动发现 `data/<task>/aria/mps_*`，**把编号 `000` 留作 eval**，后面 N 条训练。

### 8.1 CLI（覆盖 YAML）

未列出的字段只能写在 YAML 里（如 `data_sources`、`data_root`、`eval_source`）。

**优化**

| 参数 | 含义 |
|------|------|
| `--epochs` / `--batch_size` / `--lr` / `--weight_decay` / `--grad_clip` | 轮数、batch、AdamW lr、权重衰减、梯度裁剪 |
| `--use_lr_schedule` | 余弦 lr + warmup |
| `--warmup_steps` / `--min_lr_ratio` | warmup 步数；lr 下限 = `lr * min_lr_ratio` |
| `--use_amp` | 混合精度 |
| `--use_ema` / `--ema_decay` | 权重指数滑动平均 |

**数据 / 范式**

| 参数 | 含义 |
|------|------|
| `--image_size H W` | 输入图像高宽 |
| `--pred_horizon` | 预测未来步数（action chunk） |
| `--single_hand` / `--single_hand_side` | 单手任务及哪只手 |
| `--img_name` | 使用的预处理图像文件名；`None` 为无视觉、仅状态 |
| `--centric_mode` | `object_centric` / `ego_centric` |
| `--frame_mode` | `anchor_frame`（相对锚物体）/ `camera_frame` |
| `--action_mode` | `absolute` / `delta` |
| `--max_ict` | Interaction-Centric Token 上限（手+物体） |
| `--hand_tracking_method` | `aria_mps` / `mediapipe` / `wilor` / `hamer` |
| `--use_pcd_features` | 注入显式点云特征 |
| `--use_region_attn` | 可学习区域注意力 |
| `--use_ot_cfm` | minibatch OT 条件流匹配（更直的流） |
| `--use_aux_obj_dynamics` / `--use_aux_visual_foresight` / `--use_aux_temporal_contrastive` | 辅助头：物体动力学 / 未来 2D heatmap / 未来 ICT 对比 |

**增强**

| 参数 | 含义 |
|------|------|
| `--enable_augmentation` | 总开关 |
| `--enable_aug_img` | 颜色/图像增强 |
| `--enable_aug_rrc` | random-resized-crop |
| `--enable_aug_target_jittering` | 目标位姿抖动 |
| `--enable_aug_cutout` | cutout |
| `--enable_aug_temporal_stride` | 时间步长增强 |
| `--enable_aug_interpolation` | 子步插值增强 |

**结构 / 损失 / 流**

| 参数 | 含义 |
|------|------|
| `--patch_size` / `--vision_embed_dim` / `--num_decoder_layers` / `--num_heads` / `--mlp_ratio` / `--dropout` | ViT/decoder 结构 |
| `--w_flow` / `--w_pos` / `--w_rot` / `--w_g` / `--w_done` | 流匹配、手位置、旋转、抓取、done 损失权重 |
| `--w_foresight` / `--w_contrastive` | 两个辅助头权重 |
| `--use_state_noise` / `--state_pos_noise_std` / `--state_rot_noise_deg` / `--state_grasp_noise_std` | 状态噪声 |
| `--num_inference_steps` | 采样时 ODE 步数 |
| `--model_h_weighting` | 时间步权重：`uniform` / `linear` / `exp` |
| `--model_h_beta` | `exp` 加权的 β |

**路径 / 运行**

| 参数 | 含义 |
|------|------|
| `--task` | 数据与配置目录名 |
| `--job` | **必填。** YAML 文件名（无后缀），如 `HumanEgo` |
| `--exp` | 实验组：配置变成 `cfg/training/<task>/<exp>/<job>.yaml` |
| `--use_cfg` | 从上述路径加载 YAML |
| `--train_data` / `--eval_data` | 显式 session 路径列表，覆盖自动发现 |
| `--data_num` | 训练 session 数量上限 |
| `--num_workers` / `--device` / `--seed` | DataLoader workers、设备、随机种子 |
| `--eval_every` / `--vis_eval_every` | 每隔多少 epoch 做 eval / 渲视频 |

**遗留开关**（与发布配方对齐，不要随便改）：`--use_pre_norm`、`--use_ctx_norm`、`--use_done_in_flow`、`--use_legacy_image_loading`、`--use_legacy_rng`。

### 8.2 YAML 里仅有的字段

| 字段 | 含义 |
|------|------|
| `data_root` | 数据集根，默认 `./data` |
| `data_sources` | 如 `{aria: 60}`：每种来源取前 N 条（eval 占用 `000` 之后） |
| `eval_source` | 哪类来源提供 hold-out eval |
| `make_video` | eval 是否渲视频 |

发布配方：`cfg/training/serve_bread/HumanEgo.yaml`（以及 `water_flowers`、`pick_strawberry`）。改任务时优先拷这份，而不是从 TrainConfig 默认值起步。

冒烟：`--epochs 5 --data_num 1`。

---

## 9. 推理（真机模板，不能开箱即用）

```bash
SKIP_HARDWARE=0 bash setup.sh
python inference/run_inference.py cfg/inference/example_dualarm.yaml
```

需要：checkpoint 旁的 `config.json` + `dataset_stats.json`、手眼标定 `T_base_in_cam`、相机与机械臂驱动。

### 9.1 `cfg/inference/example_dualarm.yaml`

**`policy`**

| 字段 | 含义 |
|------|------|
| `ckpt` | `latest.pt` 路径 |
| `num_inference_steps` | 流 ODE 步数，常见 10–20。更大更平滑更慢 |

**`control`**

| 字段 | 含义 |
|------|------|
| `control_hz` | 控制频率，`dt = 1/hz` |
| `exec_horizon` | 每次规划执行的预测步数。更小更闭环、算力更高 |
| `done_threshold` | done 概率超过此则停 |
| `alpha_pos` / `alpha_rot` | 位置 EMA / 旋转 Slerp。更大更稳更滞后 |
| `max_pos_step` | 每步末端最大位移（米），安全笼 |
| `grasp_threshold` | 预测 grasp 超过此则闭合夹爪 |
| `safe_z_min` | 基座系 Z 下限。模板尚未全部接到驱动，仍需驱动层限位 + 急停 |

**`perception`**

| 字段 | 含义 |
|------|------|
| `anchor_key` | 定义 object-centric 坐标系的物体，通常 `obj1` |
| `n_init_frames` | 初始位姿用多少帧 RGB-D 平均 |
| `object_prompts` | 各物体开放词表提示 |
| `erase_prompt` | 要从图像里抹掉的内容（机械臂/夹爪） |
| `dinosam_cfg_path` / `lama_cfg_path` / `visualkpts_cfg_path` | 参考感知管线配置 |

**`camera` / `robot`**

| 字段 | 含义 |
|------|------|
| `camera.cfg_path` | 如 `CamRS.yaml` |
| `robot.sides` | `["left","right"]` 或 `["right"]` |
| `robot.cfg_paths` | 每臂驱动 YAML，必须含 `T_base_in_cam` |
| `robot.T_align` | 你的末端系 → 训练时的“手”系。Aria-MPS checkpoint 用文件里那组旋转；同 EE 约定的遥操作数据用单位阵 |

### 9.2 `CamRS.yaml`

| 字段 | 含义 |
|------|------|
| `rs_device_serial` | RealSense 序列号；空字符串用第一台 |
| `fps` / `w_raw` / `h_raw` | 传感器采集 |
| `w` / `h` | crop/resize 后分辨率，应对齐训练图像 |
| `warmup_frames_number` | 丢弃的启动帧 |
| `lift_roi_radius_px` | 深度抬升时关键点邻域 |
| `lift_min_depth_m` / `lift_max_depth_m` | 深度范围 |
| `lift_mad_k` / `lift_abs_gate_m` / `lift_min_valid` | 鲁棒深度滤波 |
| `lift_edge_dist_min_px` | 离深度边缘太近则丢弃 |
| `object_centric_n_frames` | 初始物体位姿平均帧数 |
| `cluster_eps_m` / `cluster_min_points` | 3D 聚类 |
| `point_robust_mad_k` | 点云离群 MAD 阈值 |

---

## 10. SAM 3.1 独立脚本（不经过 HumanEgo 主环境）

在 `third_party/sam31_runtime` 的 uv 环境中运行。

**`humanego_sam31_v2.py`**（也被 `run_sam31.sh` 调用）

| 参数 | 含义 |
|------|------|
| `--self-check` | 只检查环境/权重 |
| `--request PATH` | 读 HumanEgo 写出的 JSON 请求 |
| `--session` | RealSense/Aria session 根目录（直接模式） |
| `--start` / `--end` | 源帧区间 |
| `--reference` | 提示帧的源 index |
| `--object obj1=strawberry.` | 可重复。物体 id=文本提示 |
| `--threshold` | 分数阈值，默认 `0.50` |
| `--fps` | 调试视频帧率 |
| `--no-debug-video` | 不写调试 mp4 |

**冒烟**

```bash
python scripts/test_one_frame_text.py --image <rgb.png> --prompt strawberry --checkpoint <ckpt> --output /tmp/out.jpg
python scripts/test_video_text.py --session <SESSION> --prompt strawberry --checkpoint <ckpt> --max-frames 60
python scripts/test_video_text_refine.py --session <SESSION> --prompt strawberry --pos "398,238" --neg "470,285"
python scripts/compare_with_current_mask.py --session <SESSION> --sam31-masks <masks_dir> --obj obj1
```

`test_video_text.py` / `test_video_text_refine.py` 常用：`--start`/`--end` 源区间，`--max-frames` 上限（`0` 表示不截断），`--select largest` 或 `--obj-id` 选实例，`--pos`/`--neg` 为 `x,y` 点（分号或多个 flag）。

---

## 11. 可视化与检查脚本

| 脚本 | 主要参数 | 作用 |
|------|----------|------|
| `visualize_preprocess_triplet.py` | `--session` `--output` `--fps` `--start` `--end` | RGB / mask / overlay 三联视频 |
| `visualize_current_obj_masks.py` | `--session` `--obj` `--fps` `--output` `--final-only` | 当前物体 mask 视频 |
| `visualize_realsense_hands_video.py` | `--session` `--output` `--fps` `--start` `--end` `--thickness` `--radius` `--show-index` `--max-frames` | 把手骨架画到 RGB |
| `check_obj_mask_quality.py` | `--session` `--obj` `--min-area-ratio` `--max-area-ratio` `--min-largest-component-ratio` `--make-samples` | mask 面积/连通域质检 |
| `scripts/smoke_test_wilor.py` | `--image` `--output` | 单帧 WiLoR |
| `scripts/check_aria_hands.py` | `--session` | 统计 `aria_hands.json` 覆盖率 |

---

## 12. `datacollection/AriaMPS.yaml`

| 字段 | 含义 |
|------|------|
| `username` / `password` | Project Aria 账号。不要把真实密码提交到 git |
| `features` | `SLAM` / `HAND_TRACKING` / `EYE_GAZE` |
| `no_ui` | 无界面 |
| `force` | 覆盖已有 MPS 结果 |
| `workers` | 并行 worker 数 |

也可用环境变量 `ARIA_MPS_UNAME` / `ARIA_MPS_PASSW`。

---

## 13. 推荐工作流（RealSense + 草莓）

```bash
# 1) HumanEgo conda
conda activate humanego
SKIP_HAND=0 SKIP_HARDWARE=0 bash setup.sh

# 2) SAM 3.1（独立）
cd third_party/sam31_runtime && bash scripts/setup_uv.sh
python scripts/download_checkpoint.py --source hf
cd ../..

# 3) 只跑手部前端，确认 aria_hands.json 不是全 null
python -m preprocess.RealSensePreprocess \
  --session /path/to/session --task pick_strawberry \
  --pose-backend depth_icp --whole-clip --frontend-only

# 4) 全流程（SAM3.1 mask → CoTracker → DatasetGen）
python -m preprocess.RealSensePreprocess \
  --session /path/to/session --task pick_strawberry \
  --pose-backend depth_icp --whole-clip

# 5) 训练（至少 2 条 session；000 为 eval）
python -m training.FlowMatchingTrainer \
  --task pick_strawberry --use_cfg --job HumanEgo
```

任务提示在 `cfg/preprocess/tasks/pick_strawberry.yaml`：`obj1=strawberry .`，`obj2=a pink plate .`。

---

## 14. 发布到 GitHub

本目录已经 `git init` 并完成初始提交。本机没有 GitHub 登录凭据，需要你在自己的账号下建仓库再推：

```bash
cd /data/home/chenxiangyu/victor/HumanEgo
export PATH="/data/home/chenxiangyu/miniconda3/envs/humanego/bin:$PATH"

# 若已安装 GitHub CLI 并完成 gh auth login：
gh repo create HumanEgo --private --source . --remote origin --push

# 或在网页新建空仓库后：
git remote add origin git@github.com:<YOUR_USER>/HumanEgo.git
git push -u origin main
```

不要把 `data/`、`runs/`、`*.pt` 权重推上去（已在 `.gitignore`）。SAM 3.1 权重请克隆后按 §3.2 自行下载。

## 15. 许可证与第三方

HumanEgo 使用 [PolyForm Noncommercial License 1.0.0](LICENSE)（非商业）。SAM 3 源码在
`third_party/sam31_runtime/third_party/sam3/LICENSE`，权重使用须遵守 Meta / HuggingFace 条款。

请同时引用上游 HumanEgo 论文（见 [README.md](README.md) 文末 BibTeX）。
