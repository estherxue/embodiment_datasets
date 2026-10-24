# LIBERO（libero_spatial）评测笔记

- 仓库：`embodiment_datasets/LIBERO`，commit `8f1084e`（2025-03-15，master），上游 https://github.com/Lifelong-Robot-Learning/LIBERO
- 执行环境：本地 Intel Mac 只做了安装与 pilot（见文末），正式评测在 **Kaggle GPU script kernel** 上跑：`cindyxue1122/libero-spatial-eval`（https://www.kaggle.com/code/cindyxue1122/libero-spatial-eval）
- kernel 源码：`embodiment_datasets/kaggle/libero-eval/`（`driver_template.py` + `build_kernel.py` → 单文件 `libero_spatial_eval.py`，本地 `local_eval/*.py` 以字符串内嵌）
- Kaggle 拉回的结果：`embodiment_datasets/kaggle/outputs/libero-spatial-eval/run{1..5}/`，每个目录对应 kernel 的一个版本（`summary.md`、`summary.json`、`replay/*.json`、`logs/*.log`、`libero-spatial-eval.log` 是 Kaggle 的 kernel 日志）；**run5 是最终完整结果**，run2/run4 是软件渲染下的回放，run1/run3 是失败的版本（留作排错记录）。`embodiment_datasets/kaggle/libero-eval/merge_notes.py` 从这些目录重新生成本文第 5、6 节。

## 0. 结论摘要

<!-- KAGGLE_RESULTS_SUMMARY -->
- **预训练权重**：LIBERO 官方没有发布任何可被 `libero/lifelong/evaluate.py` 加载的 checkpoint，Hub 上带 "libero" 的 200+ 模型全是 VLA/扩散策略（OpenVLA、π0/π0.5、SmolVLA、GR00T…），需要各自的推理栈且在 T4/CPU 上不可行 → 按要求走 demo rollout + 协议走通。
- **demo 回放（libero_spatial，10 任务 × 50 条 demo，开环动作回放，成功判定与评测协议相同）**：合计 **451/500 = 90.2%**（95% Wilson CI [87.3%, 92.5%]），单任务 82%~96%，跨任务标准差 5.7 个百分点。在 Mesa 软件渲染（v2/v4）和 NVIDIA T4 渲染（v5）下跑了两遍：**500 条记录的成功标志、成功步数、最大状态漂移完全一致（差 0.00e+00）**，即回放是确定性的，跨 run 方差为 0；渲染后端只影响速度（55 s/demo → 9 s/demo）。
- **evaluate.py 协议走通**：用随机初始化的 `Multitask/BCTransformerPolicy`（5.38 M 参数）按原协议（固定 `pruned_init` 初始状态、20 个 `SubprocVectorEnv` 并行 episode、5 步零动作预热、`max_steps=600`、`success = done`）在 10 个任务上各跑了 20 个 episode，流程全部跑通、每任务 600 步跑满、成功 0/20（随机策略，数字无意义），单任务约 11~12 分钟（T4 EGL 渲染，4 核 CPU）。
- 为此在 Kaggle 上做的三处修复：`PYTHONPATH=<repo>`（可编辑安装不生效）、`setuptools<81`（`pkg_resources`）、把 `SubprocVectorEnv` 的 worker 改成 `spawn`（NVIDIA EGL 不能跨 fork）并注册 NVIDIA 的 EGL ICD（否则是 llvmpipe 软件渲染）。
- 输出：`embodiment_datasets/kaggle/outputs/libero-spatial-eval/run5/`（最终结果：`summary.md`、`summary.json`、`replay/*.json`、`logs/*.log`）；`run2/`、`run4/` 是软件渲染下的回放；kernel 源码 `embodiment_datasets/kaggle/libero-eval/`。
<!-- /KAGGLE_RESULTS_SUMMARY -->

## 1. 任务套件结构（按源码 `libero/libero/benchmark/__init__.py`、`libero_suite_task_map.py` 核对）

| 套件 | 注册名（`get_benchmark_dict()` key） | 任务数 | 说明 | init_states 维度（每任务） |
|---|---|---|---|---|
| LIBERO-Spatial | `libero_spatial` | 10 | 同一场景（桌面 + 黑碗/盘子/ramekin/饼干盒/柜子/炉子），只改空间关系；语言全是 "pick up the black bowl *<位置>* and place it on the plate" | (50, 92) |
| LIBERO-Object | `libero_object` | 10 | 同一布局换 10 种物体；"pick up the *<object>* and place it in the basket" | (50, 110) |
| LIBERO-Goal | `libero_goal` | 10 | 同一场景同一批物体，换 10 种目标（开抽屉、放碗、开炉子…） | (50, 79) |
| LIBERO-10（LIBERO-Long） | `libero_10` | 10 | 长程、两阶段任务，来自 LIBERO-100 | (50, 45~123)，随场景不同 |
| LIBERO-90 | `libero_90` | 90 | LIBERO-100 其余 90 个短任务，用作预训练；只支持 task_order 0 | (50, 45~123) |

- **LIBERO-100 = LIBERO-10 + LIBERO-90**（`bddl_files/` 与 `init_files/` 下只有这 5 个目录，没有单独的 libero_100；`LIBERO_100` 这个 Benchmark 类在源码里有注册但 `task_maps` 里没有同名 key，直接实例化会报错；数据集下载脚本的 `libero_100` 选项会下载 libero_10 + libero_90 两个目录）。
- 每个任务 = 一个 `Task(name, language, problem="Libero", problem_folder=<suite>, bddl_file=<name>.bddl, init_states_file=<name>.pruned_init)`；语言指令直接由文件名还原（`grab_language_from_filename`）。
- 文件位置（由 `~/.libero/config.yaml` 指定，首次 `import libero` 时若没有该文件会 `input()` 交互式询问）：
  - `benchmark_root = libero/libero`，`bddl_files = libero/libero/bddl_files/<suite>/<task>.bddl`
  - `init_states = libero/libero/init_files/<suite>/<task>.pruned_init`（50 个 MuJoCo 完整状态向量；旁边的 `.init` 有 100 个，评测用的是 pruned 版）
  - `datasets = <repo>/datasets/<suite>/<task>_demo.hdf5`
  - `assets = libero/libero/assets`
- 10 任务套件有 21 种固定任务顺序 `task_orders[0..20]`（`task_order_index`，默认 0 = 0..9 原序），用于 lifelong 实验的顺序随机化。
- 演示数据（`datasets/libero_spatial/*_demo.hdf5`，HF `yifengzhu-hf/LIBERO-datasets`，libero_spatial 共 5.8 GB）：每任务 50 条 demo；每条含 `actions (T,7)`、`states (T,92)`（MuJoCo 状态，回放用）、`robot_states (T,9)`、`dones`、`rewards`、`obs/{agentview_rgb, eye_in_hand_rgb (T,128,128,3) uint8, ee_pos, ee_ori, ee_states, gripper_states (T,2), joint_states (T,7)}`；libero_spatial 的 demo 长度均值 101 步（75~168）。
- 环境：robosuite 1.4.0 + mujoco，Panda 机械臂，`OSC_POSE` 控制器（delta 位姿 + 夹爪，动作 7 维，范围 [-1,1]），`control_freq=20`，`horizon=1000`，相机 `agentview` + `robot0_eye_in_hand`，默认 128×128。
- 策略输入（`configs/data/default.yaml`）：`rgb = [agentview_rgb, eye_in_hand_rgb]`，`low_dim = [gripper_states(2), joint_states(7)]`，`seq_len=10`；语言用 `bert-base-cased` 的 pooler 输出（768 维，`task_embedding_format=bert`）。

### libero_spatial 的 10 个任务（task_order 0）

| id | 语言指令 |
|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | pick up the black bowl from table center and place it on the plate |
| 3 | pick up the black bowl on the cookie box and place it on the plate |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 5 | pick up the black bowl on the ramekin and place it on the plate |
| 6 | pick up the black bowl next to the cookie box and place it on the plate |
| 7 | pick up the black bowl on the stove and place it on the plate |
| 8 | pick up the black bowl next to the plate and place it on the plate |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate |

（其余四个套件的完整任务名/语言在 `local_eval/suite_structure.json`。）

## 2. 评测协议（按 `libero/lifelong/evaluate.py`、`libero/lifelong/metric.py::evaluate_one_task_success`、`configs/eval/default.yaml` 核对）

1. **环境**：`OffScreenRenderEnv(bddl_file_name, camera_heights=cfg.data.img_h, camera_widths=cfg.data.img_w)`，`SubprocVectorEnv` 开 `env_num` 个子进程（`evaluate.py` 硬编码 `env_num = 20`；训练中评测用 `min(num_procs=20, n_eval=20)`）。
2. **固定初始状态**：`init_states = torch.load(<task>.pruned_init)`（50 个），第 i 轮用 `indices = arange(i*env_num, (i+1)*env_num) % 50`，`env.set_init_state(init_states[indices])` 直接把 MuJoCo 状态写进 sim（`set_state_from_flattened` + `forward`），所以每个 episode 的初始布局是确定的、跨方法可比。
3. **预热**：先 `env.step(zeros(env_num, 7))` 5 步让物理稳定。
4. **rollout**：`while steps < cfg.eval.max_steps(=600)`：`raw_obs_to_tensor_obs` 把 robosuite 观测按 `obs_key_mapping` 映射成策略输入（图像经 `robomimic ObsUtils.process_obs`），`actions = algo.policy.get_action(data)`，`env.step(actions)`；`dones[k] = dones[k] or done[k]`，全部 done 则提前结束。
5. **成功判定**：`BDDLBaseDomain.step` 把 robosuite 的 `done` 覆盖成 `self._check_success()`（即 bddl 里 goal 谓词是否满足，与 horizon 无关），**一旦某步成功即记成功**（sticky）。
6. **指标**：`success_rate = num_success / n_eval`（`evaluate.py` 里是 `/ env_num`，n_eval=20 → 20 个 episode）；另外 `evaluate.py` 还会算该任务数据集上的 BC loss（当前版本 `test_loss` 固定 0.0 占位）。
7. **输出**：`torch.save({"loss", "success_rate"}, <experiments>_saved/<benchmark>_<algo>_<policy>_<seed>_{ep|load}<k>_on<task>.stats`，可选 `--save-videos` 保存 agentview 视频。
8. **checkpoint 布局**：`experiments/<BENCHMARK>/<Algo>/<Policy>_seed<seed>/run_NNN/`，multitask 为 `multitask_model_ep{0,5,…,50}.pth`，lifelong 为 `task{k}_model.pth`；文件是 `{"state_dict", "cfg", "previous_masks"}` 的 torch pickle（`torch_load_model`），因此 **torch ≥ 2.6 需要 `weights_only=False`**（本次用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` + torch 2.5.1 双保险）。
9. 算法/策略枚举：`--algo {base(Sequential), er, ewc, packnet, multitask}`，`--policy {bc_rnn_policy, bc_transformer_policy, bc_vilt_policy}`；lifelong 评测要指定 `--load_task k`（用学完第 k 个任务后的模型评 `--task_id`）。
10. 备注：LIBERO 官方配置对所有套件都用 `max_steps=600`；后来 OpenVLA 等代码库的 LIBERO 评测脚本改用按套件的步数上限（spatial 220 / object 280 / goal 300 / 10 520）并用 `libero_*_no_noops` 数据集，和这里不是同一协议，比较数字时要注意。

`local_eval/evaluate_cpu.py` 是上游 `evaluate.py` 的最小改动版本（标 `# [local_eval patch]`）：`--device_id` 可省略（CPU）、`--env-num` 替代硬编码 20、`--max-steps` 可覆盖、结尾多打一行 `EVAL_SUMMARY {...}`；其余逻辑不变。

## 3. 预训练权重：结论 = 没有可直接评的 LIBERO 原生权重

- LIBERO 仓库和 README 只给了数据集和训练命令（`lifelong.main`），**没有发布任何 lifelong/multitask 算法的 checkpoint**。
- Hub 搜索（`search=libero`，`bc_transformer`、`BCTransformerPolicy`、`libero lifelong`、`bc_rnn` 均无命中）：带 "libero" 的 200+ 模型全是 VLA/扩散策略（OpenVLA-7B 及 OFT 变体、`lerobot/pi05_libero*`、`lerobot/smolvla_libero`、`HuggingFaceVLA/smolvla_libero`、GR00T、MolmoAct、X-VLA、MiniVLA、NORA、FLOWER…），全部需要各自的推理栈（LeRobot 的 `libero` env、OpenVLA 的评测脚本等），不是 `libero/lifelong/evaluate.py` 能加载的 `.pth`；在无 GPU 的本机或 Kaggle 单卡上跑 3B/7B 模型也不现实。
- 因此按任务要求走 **demo rollout**：把 50 条 demo 的动作在 `OffScreenRenderEnv` 里开环回放，用与评测相同的 `success = done` 判定；并用一个**随机初始化**的 `Multitask/BCTransformerPolicy` checkpoint 把 `evaluate.py` 的协议代码路径完整走一遍（数字无意义，只验流程）。
- 如果以后要评真实权重，最轻量的路线是 LeRobot 侧的 `lerobot/smolvla_libero`（~450M 参数）或 `lerobot/pi05_libero*`，用 LeRobot 自带的 libero 环境与 eval 脚本，而不是本仓库的 `evaluate.py`。

## 4. Kaggle 运行记录

<!-- KAGGLE_RUN -->
- Kernel：`cindyxue1122/libero-spatial-eval`（script kernel，GPU + internet，私有）。每个版本都是从零装环境：`pip install uv` → `uv venv --python 3.10 /kaggle/tmp/venv` → `git clone LIBERO && git checkout 8f1084e` → `uv pip install torch==2.5.1 torchvision==0.20.1`（CPU 构建，`--index-url https://download.pytorch.org/whl/cpu`）→ `uv pip install -r requirements.txt huggingface_hub h5py mujoco==2.3.2 setuptools` → `uv pip install -e .` → 写 `~/.libero/config.yaml` → `MUJOCO_GL=egl` 冒烟 → `benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface` → `fetch_bert.py` → 评测。整套安装 + 数据集下载不到 3 分钟（Kaggle 到 HF 的带宽约 200 MB/s，5.8 GB 用 25~36 s）。
- 机器：Kaggle 分配的是 2× Tesla T4（15 GB）、31 GB 内存、Ubuntu 24.04、基础镜像 Python 3.13；CPU 亲和掩码是 4 个核（`os.sched_getaffinity` = {0,1,2,3}；日志里 `nproc` 显示 1~2 只是因为 GNU nproc 会读 `OMP_NUM_THREADS`）。v2 回放慢的主因是 EGL 落到了 Mesa/llvmpipe 软件渲染（见 v3 诊断），不是核数。
- 版本记录：

| 版本 | 时长 | 结果 |
|---|---|---|
| v1 | 2 min | 安装、数据集、bert 全部成功；渲染冒烟失败：`ModuleNotFoundError: No module named 'libero'`——LIBERO 顶层 `libero/` 没有 `__init__.py`，`uv pip install -e .` 生成的可编辑安装在仓库目录之外找不到包（本地 Mac 之所以能跑是因为 `.pth` 里直接写了仓库路径）。修复：所有 venv 子进程加 `PYTHONPATH=/kaggle/tmp/LIBERO`（上游自己的脚本也是靠 `init_path.py` 往 `sys.path` 插仓库根目录）。 |
| v2 | 243 min | EGL 渲染冒烟通过（建环境 + reset 26 s）；demo 回放 2 个进程并行，**每条 demo 38~71 s**（本地 Mac 12 s），4 小时预算内只跑完 task 0~8（450 条），task 9 跑到第 45 条被超时杀掉；协议走通全部失败：`import wandb` → `from pkg_resources import parse_version` → `ModuleNotFoundError: No module named 'pkg_resources'`（uv 建的 venv 默认不带 setuptools）。 |
| v3 | 2 min | 确认了两件事：`/usr/share/glvnd/egl_vendor.d/` 里只有 `50_mesa.json`，而 `/usr/local/nvidia/lib64` 下有 `libEGL_nvidia.so.0`、`libnvidia-eglcore`——即 v2 的 EGL 实际是 Mesa 软件渲染，脚本已自动补写 `10_nvidia.json`；`uv pip install setuptools` 装的是 84.2.1，这个版本已经删掉了 `pkg_resources`，所以 `import wandb` 仍然失败，设置阶段中止。 |
| v4 | 16 min | `GL_RENDERER = Tesla T4/PCIe/SSE2`（NVIDIA EGL 生效）；task 9 回放 50 条：47/50，**10.3 s/demo**（v2 软件渲染下 55 s）；协议走通仍失败：`SubprocVectorEnv` 的 20 个 fork 子进程在 `eglCreateContext` 报 `EGL_BAD_ALLOC`——父进程 import 时已初始化 NVIDIA EGL，fork 出来的子进程不能再用。 |
| v5 | 158 min | **全部跑通**：协议评测 10 任务 × 20 episode × 600 步（egl + spawn，每任务 650~730 s，共 115 min）；随后 GPU 渲染下重跑 10 × 50 条回放（2 进程，36~41 min，9.2 s/demo），结果与 v2/v4 逐条一致。 |
<!-- /KAGGLE_RUN -->

## 5. demo 回放结果（libero_spatial，每任务 50 条 demo，开环动作回放）

<!-- KAGGLE_REPLAY -->
| task | 来源 run | demo 数 | 成功 | 成功率 | Bernoulli std | SE | 95% CI (Wilson) | 结束时仍成功 | 平均成功步数 / demo 长度 | 状态漂移 mean / max | 秒/demo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | run5 | 50 | 48 | 96% | 0.196 | 0.028 | [87%, 99%] | 96% | 81.6 / 101.4 | 4.86 / 11.54 | 7 |
| 1 | run5 | 50 | 45 | 90% | 0.300 | 0.042 | [79%, 96%] | 90% | 110.4 / 134.1 | 5.49 / 12.85 | 9 |
| 2 | run5 | 50 | 48 | 96% | 0.196 | 0.028 | [87%, 99%] | 96% | 98.8 / 117.6 | 3.46 / 8.51 | 8 |
| 3 | run5 | 50 | 45 | 90% | 0.300 | 0.042 | [79%, 96%] | 90% | 84.3 / 101.0 | 4.54 / 10.02 | 8 |
| 4 | run5 | 50 | 41 | 82% | 0.384 | 0.054 | [69%, 90%] | 82% | 130.1 / 149.6 | 4.35 / 8.83 | 11 |
| 5 | run5 | 50 | 41 | 82% | 0.384 | 0.054 | [69%, 90%] | 82% | 101.1 / 115.9 | 6.35 / 27.02 | 9 |
| 6 | run5 | 50 | 48 | 96% | 0.196 | 0.028 | [87%, 99%] | 96% | 109.1 / 126.2 | 5.79 / 16.43 | 9 |
| 7 | run5 | 50 | 42 | 84% | 0.367 | 0.052 | [71%, 92%] | 84% | 122.6 / 142.2 | 5.58 / 13.17 | 11 |
| 8 | run5 | 50 | 46 | 92% | 0.271 | 0.038 | [81%, 97%] | 92% | 100.9 / 119.3 | 4.77 / 13.56 | 9 |
| 9 | run5 | 50 | 47 | 94% | 0.237 | 0.034 | [84%, 98%] | 94% | 123.6 / 137.6 | 5.49 / 12.48 | 11 |

- 合计 451/500 = **90.2%**（95% Wilson CI [87.3%, 92.5%]）；任务间成功率均值 90.2%，跨任务标准差 5.7 个百分点，任务级样本方差 0.0032。
- 成功判定与评测协议相同：`env.step` 返回的 `done`（= `_check_success()`），某一步成功即记成功；“结束时仍成功”是回放完全部动作后再查一次 `check_success()`。
- 状态漂移 = 回放过程中 `env.get_sim_state()` 与数据集里记录的 `states` 的最大 L2 距离，用来说明开环回放的物理可复现程度（MuJoCo 非确定性 + 版本差异）。
<!-- /KAGGLE_REPLAY -->

可重复性：同一套回放在 v2（task 0~8）/v4（task 9，Mesa llvmpipe 软件渲染）和 v5（全部 10 任务，NVIDIA T4 EGL 渲染）各跑了一遍，500 条记录逐条比较：成功标志 500/500 相同、成功步数 500/500 相同、最大状态漂移 500/500 相同（最大差值 0.00e+00）。回放是完全确定性的（固定 `env.seed(10000)` + `set_init_state(states[0])` + 记录的动作序列），跨 run 方差为 0；上表的 Bernoulli 方差/CI 描述的是 50 条 demo 之间的差异，不是重复实验的波动。失败的 49 条 demo 是开环回放的物理漂移导致（如 task 4/5 漂移最大，成功率也最低），与渲染无关。

## 6. evaluate.py 协议走通（随机初始化策略，数字无意义）

<!-- KAGGLE_PROTOCOL -->
- 来源 run：run5；checkpoint：UNTRAINED (randomly initialised) Multitask/BCTransformerPolicy; success numbers are NOT policy performance, only a protocol check
- 最终配置：{'backend': 'egl', 'env_num': 20}；跳过的任务：[]

| task | 渲染后端 | episodes (env_num) | 实际步数 | max_steps | 成功数 | success_rate | rollout 秒 | 整段 wall 秒 | rc |
|---|---|---|---|---|---|---|---|---|---|
| 0 | egl | 20 | 600 | 600 | 0 | 0.0 | 721.1 | 729.0 | 0 |
| 1 | egl | 20 | 600 | 600 | 0 | 0.0 | 699.8 | 709.0 | 0 |
| 2 | egl | 20 | 600 | 600 | 0 | 0.0 | 642.2 | 650.5 | 0 |
| 3 | egl | 20 | 600 | 600 | 0 | 0.0 | 663.4 | 671.5 | 0 |
| 4 | egl | 20 | 600 | 600 | 0 | 0.0 | 675.6 | 683.7 | 0 |
| 5 | egl | 20 | 600 | 600 | 0 | 0.0 | 650.0 | 657.9 | 0 |
| 6 | egl | 20 | 600 | 600 | 0 | 0.0 | 671.3 | 679.5 | 0 |
| 7 | egl | 20 | 600 | 600 | 0 | 0.0 | 705.8 | 713.5 | 0 |
| 8 | egl | 20 | 600 | 600 | 0 | 0.0 | 676.6 | 684.9 | 0 |
| 9 | egl | 20 | 600 | 600 | 0 | 0.0 | 692.3 | 700.3 | 0 |
<!-- /KAGGLE_PROTOCOL -->

解读：10 个任务的流程全部按原协议走完（20 个 episode 对应 `pruned_init` 的前 20 个初始状态，`steps_run = max_steps = 600` 说明没有任何 episode 提前成功），成功率 0/20 是随机初始化策略的预期结果，只证明 checkpoint 布局 → `torch_load_model` → BERT 任务嵌入 → `SubprocVectorEnv` → `raw_obs_to_tensor_obs` → `policy.get_action` → `.stats` 落盘这条链路可用。真正的策略评测只需把 `experiments/.../multitask_model_ep{N}.pth` 换成训练得到的 checkpoint（`lifelong.main benchmark_name=LIBERO_SPATIAL lifelong=multitask policy=bc_transformer_policy`）。耗时：每任务 650~730 s，其中约 60 s 是 20 个 spawn worker 的启动与建环境，其余是 600 步 × 20 env 的物理 + 渲染 + 策略前向（T4 渲染，4 核 CPU）。

## 7. 本地 Intel Mac 上的尝试（已中止，仅供参考）

- 环境：Python 3.10.20 venv（uv），torch 2.2.2 / torchvision 0.17.2（macOS x86_64 官方 wheel 的上限），numpy 1.22.4，robosuite 1.4.0，mujoco 2.3.2，transformers 4.21.1，`requirements.txt` 原样可装；渲染走 macOS 默认 CGL 离屏后端，无需设置 `MUJOCO_GL`。
- 数据集：`benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface` 成功（5.8 GB，10 个 hdf5，`check_libero_dataset` 校验通过）。
- pilot：task 0 两条 demo 回放均 success（demo_0 第 82/98 步成功，demo_1 第 73/84 步成功，最大状态漂移 2.7/3.6），建环境 4~36 s，单条 demo 回放约 12 s（4 核 i7，4 进程并行时 45 s）。随机 checkpoint 生成成功（5.38 M 参数，task_embs (10, 768)）。
- 4 个 worker 的全量回放（每任务 20 条）跑了约 1.5 分钟后按用户要求停止并改用 Kaggle，本地未产出完整数字。

## 8. 踩坑与解法

- `import libero` 时若 `~/.libero/config.yaml` 不存在会 `input()` 询问路径，非交互环境直接 EOFError：先手动写好 config.yaml（Kaggle 脚本在安装后、首次 import 前写入，并把 `datasets` 指到 `/kaggle/tmp/datasets`）。
- `requirements.txt` 钉死 `numpy==1.22.4`、`matplotlib==3.5.3`、`opencv-python==4.6.0.66`，只有 Python ≤ 3.10 有 wheel：Kaggle 基础镜像 Python 版本更高，所以用 `uv venv --python 3.10` 单独建环境，不碰基础镜像。
- `transformers==4.21.1` 把 `huggingface-hub` 限制在 <1.0，数据集下载脚本里的 `local_dir_use_symlinks` 参数因此仍可用。
- torch ≥ 2.6 `torch.load` 默认 `weights_only=True`，LIBERO 的 checkpoint 与 init 文件是 pickle：用 torch 2.5.1 并设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
- Kaggle Linux 上离屏渲染：`MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl`（GPU kernel），失败则 `apt-get install libosmesa6` 后切 `osmesa`；脚本先跑一个"建环境 + 1 步 + 图像非常数"的冒烟测试再决定后端。
- `evaluate.py` 硬编码 `cuda:N` 与 20 个并行 env：用 `local_eval/evaluate_cpu.py` 的 `--device_id`/`--env-num` 补丁；内存不够时自动退到 10 个 env。
- 磁盘：数据集、venv、仓库、HF 缓存都放 `/kaggle/tmp`，`/kaggle/working`（20 GB 上限、会被打包成 kernel 输出）只放 JSON/日志。

<!-- KAGGLE_PITFALLS -->
- `uv pip install -e .` 对没有顶层 `__init__.py` 的 LIBERO 不生效（见 v1）：统一用 `PYTHONPATH=<repo>`。
- uv venv 没有 `setuptools`，`wandb==0.13.1` 在 import 时就要 `pkg_resources`：显式 `uv pip install setuptools`。
- `evaluate.py` 找不到 checkpoint / 数据集时是 `sys.exit(0)`，返回码看不出失败：驱动脚本只认日志末尾的 `EVAL_SUMMARY {...}` 行，没有就视为失败并重试。
- `SubprocVectorEnv` 用 fork 启动 worker；父进程 `import mujoco`/robosuite 时已经初始化了 NVIDIA EGL display，fork 出的子进程调用 `eglCreateContext` 直接 `EGL_BAD_ALLOC`。LIBERO 的 env 工厂已经是 `CloudpickleWrapper`，所以把 `venv.py` 里的 `Process` 换成 `multiprocessing.get_context("spawn").Process` 即可；软件渲染（osmesa）没有这个问题但慢 5 倍。
- Kaggle 容器里 NVIDIA 驱动库被挂载了，但 GLVND 的 EGL vendor 目录只有 Mesa 的 ICD：`MUJOCO_GL=egl` 能跑但走 llvmpipe 软件渲染（v2：每步约 0.4 s）。补一个 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`（`library_path: libEGL_nvidia.so.0`）即可切到 T4。
- `uv pip install setuptools` 会装最新版（84.x），其中 `pkg_resources` 已被移除，`wandb==0.13.1` import 即炸：钉 `setuptools<81`。
- GNU `nproc` 读 `OMP_NUM_THREADS`，设了线程数之后它显示的不是真实核数；看 `os.sched_getaffinity(0)` 或 `/sys/fs/cgroup/cpu.max`。
- 旧版 `kaggle` CLI（1.5.12）从 `kaggleusercontent.com` 拉大文件时偶发 `SSLEOFError`，重试即可；kernel 输出文件以 `/kaggle/working` 为根打包，日志单独是 `<slug>.log`。
<!-- /KAGGLE_PITFALLS -->
