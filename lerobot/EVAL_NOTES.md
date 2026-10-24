# LeRobot：预训练 ACT 在 PushT / ALOHA 仿真插接上的评测记录

> 仓库：`embodiment_datasets/lerobot`，commit `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`（`chore: Bump to 0.3.3`，2025-08-06）。
> 主结果在 Kaggle GPU kernel 上跑（Linux + CUDA），本地 Intel Mac（CPU）的结果作为附表。
> 每项评测固定 `n_episodes=50, batch_size=50`，并记录 `--seed`；episode i 的环境 seed = seed + i。

## 1. 评测了哪些权重（以及为什么）

| 代号 | Hub 权重 | 任务 / 观测 | 来源说明 | 作者自报成绩 |
|---|---|---|---|---|
| `pusht_act_keypoints` | `alexandersoare/act_pusht_keypoints` | PushT，`obs_type=environment_state_agent_pos`（16 维 T 块关键点 + 2 维 agent 位置，**不是像素**） | LeRobot 核心维护者 2024-07 发布的 ACT（chunk 16 / n_action_steps 16，VAE，min-max 归一化），数据集 `lerobot/pusht_keypoints`。Hydra 旧格式，需转换 config.json（见 §4） | 64.0% / 500 episodes（仓库自带 eval_info.json） |
| `pusht_act_pixels` | `arclabmit/pusht_act_model` | PushT，`obs_type=pixels_agent_pos`（96×96 图像 + 2 维 agent 位置） | 社区 2025-08 用 lerobot 新格式训练的 ACT（数据 `lerobot/pusht`，200k steps，batch 64），但沿用了 ALOHA 的默认 chunk_size=100 / n_action_steps=100：PushT 是 10 fps，相当于每次开环执行 10 秒，基本不工作 | 无 |
| `aloha_insertion_act` | `lerobot/act_aloha_sim_insertion_human` | gym-aloha `AlohaInsertion-v0`，顶视相机 480×640 + 14 维关节状态 | LeRobot 官方 ACT（数据 `lerobot/aloha_sim_insertion_human`，100k steps，chunk 100 / n_action_steps 100，VAE，resnet18） | 83.0% / 500 episodes（avg_max_reward 3.546） |
| `pusht_diffusion`（参照，非 ACT） | `lerobot/diffusion_pusht` | PushT，像素 | LeRobot 官方 Diffusion Policy，用来对照“同一评测流程下官方像素策略能到多少” | 65.4% / 500 episodes |
| `pusht_random`（下限） | 均匀随机动作 | PushT | 自写脚本，同 seed、同 300 步上限 | — |
| `pusht_act_pepijn223`（可选） | `pepijn223/act-pusht` | PushT，像素 | 社区 2025-01 的旧格式 ACT，转换后可加载，本地结果 0% | 无 |

**官方 `lerobot/act_pusht` 不存在**（Hub API 404），LeRobot 官方只发布了 `diffusion_pusht`（PushT）和 `act_aloha_sim_*`（ALOHA）。
Hub 上以 `act_pusht` 搜索到的二十来个仓库全是社区训练结果；其中真正能用的只有 keypoints 版，所以 PushT 的 ACT 主结果用的是 **关键点观测** 的 ACT，而不是像素 ACT，这一点在解读数字时要注意。

## 2. 结果

<!-- KAGGLE_RESULTS_START -->
### 2.a Kaggle（主结果）：Tesla T4，torch 2.7.1+cu126，MUJOCO_GL=egl，kernel `cindyxue1122/lerobot-act-eval-pusht-aloha`，总耗时 6.1 h

| run | 权重 | seed | 成功 | pc_success | 样本方差 | Bernoulli 方差 | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| pusht_act_keypoints_seed1000 | alexandersoare/act_pusht_keypoints（关键点 ACT） | 1000 | 32/50 | 64.0% | 0.235 | 0.230 | [50.1, 75.9] | 95.5 | 0.949 | 32 s |
| pusht_act_keypoints_seed2000 | alexandersoare/act_pusht_keypoints（关键点 ACT） | 2000 | 30/50 | 60.0% | 0.245 | 0.240 | [46.2, 72.4] | 102.5 | 0.932 | 29 s |
| pusht_act_keypoints_seed3000 | alexandersoare/act_pusht_keypoints（关键点 ACT） | 3000 | 32/50 | 64.0% | 0.235 | 0.230 | [50.1, 75.9] | 98.3 | 0.918 | 30 s |
| pusht_act_pixels_seed1000 | arclabmit/pusht_act_model（像素 ACT） | 1000 | 0/50 | 0.0% | 0.000 | 0.000 | [0.0, 7.1] | 45.3 | 0.424 | 67 s |
| pusht_act_pixels_seed2000 | arclabmit/pusht_act_model（像素 ACT） | 2000 | 0/50 | 0.0% | 0.000 | 0.000 | [0.0, 7.1] | 44.4 | 0.371 | 67 s |
| pusht_act_pixels_seed3000 | arclabmit/pusht_act_model（像素 ACT） | 3000 | 3/50 | 6.0% | 0.058 | 0.056 | [2.1, 16.2] | 42.2 | 0.438 | 68 s |
| aloha_insertion_act_seed1000 | lerobot/act_aloha_sim_insertion_human | 1000 | 6/50 | 12.0% | 0.108 | 0.106 | [5.6, 23.8] | 180.7 | 1.940 | 5184 s |
| aloha_insertion_act_seed2000 | lerobot/act_aloha_sim_insertion_human | 2000 | 4/50 | 8.0% | 0.075 | 0.074 | [3.2, 18.8] | 213.0 | 2.120 | 5287 s |
| aloha_insertion_act_seed3000 | lerobot/act_aloha_sim_insertion_human | 3000 | 6/50 | 12.0% | 0.108 | 0.106 | [5.6, 23.8] | 208.0 | 2.080 | 5242 s |
| pusht_diffusion_seed1000 | lerobot/diffusion_pusht（非 ACT 参照） | 1000 | 33/50 | 66.0% | 0.229 | 0.224 | [52.2, 77.6] | 114.7 | 0.972 | 273 s |
| pusht_random_seed1000 | 随机动作（下限） | 1000 | 0/50 | 0.0% | 0.000 | 0.000 | [0.0, 7.1] | 1.2 | 0.086 | 47 s |
| pusht_act_pepijn223_seed1000 | pepijn223/act-pusht（像素 ACT，可选） | 1000 | 0/50 | 0.0% | 0.000 | 0.000 | [0.0, 7.1] | 37.6 | 0.403 | 67 s |

- **alexandersoare/act_pusht_keypoints（关键点 ACT）** 跨 3 个 seed：均值 62.7%，seed 间标准差 2.3 个百分点（64%, 60%, 64%）。
- **arclabmit/pusht_act_model（像素 ACT）** 跨 3 个 seed：均值 2.0%，seed 间标准差 3.5 个百分点（0%, 0%, 6%）。
- **lerobot/act_aloha_sim_insertion_human** 跨 3 个 seed：均值 10.7%，seed 间标准差 2.3 个百分点（12%, 8%, 12%）。

解读：
- PushT 上 **关键点 ACT 62.7% ± 2.3**，与作者自报的 64%/500 episodes 一致；官方 Diffusion Policy 在同一流程下 66.0%（作者自报 65.4%），说明评测流程本身是对的。三个 seed 的结果与本地 macOS CPU 跑出的 66/58/66 略有出入（GPU 与 CPU 的浮点差异导致个别 episode 翻转），差异在置信区间内。
- 像素 ACT（arclabmit）2.0% ± 3.5，与本地完全一致（PushT 仿真是确定性的）：chunk/n_action_steps=100 的 ALOHA 默认超参直接套到 10 fps 的 PushT 上，每次开环执行 10 秒，无法完成精细对齐（avg_max_reward≈0.4 说明能推到一半重叠就停了）。
- **ALOHA 插接 ACT 只有 10.7% ± 2.3，远低于作者自报的 83%。** 原因是仿真器版本：该权重（及其 83% 的 eval_info.json）对应 gym-aloha 0.1.1 + MuJoCo 2.3.7（lerobot 0.3.3 发布时 gym-aloha 只有 0.1.1，且钉死 `mujoco<3`），而 2025-10 之后发布的 gym-aloha 0.1.2+ 改用 MuJoCo ≥3，Kaggle 上解析到了 gym-aloha 0.1.5 + mujoco 3.8.1 + dm-control 1.0.41。MuJoCo 3 的接触求解和渲染都与 2.3.7 不同，像素策略对渲染差异也敏感。max_reward 分布（seed 1000）：0 分 13 个、1 分 9 个、2 分 2 个、3 分（插销触到插座）20 个、4 分（插入成功）6 个，即策略流程大体对、最后插入阶段失败，符合物理差异导致的表现。
- 每个 ALOHA seed 在 T4 上跑了约 87 分钟（50 个同步并行 env × 400 步，每步 480×640 EGL 渲染），这也是整个 kernel 跑了 6 小时的原因。


### 2.a' Kaggle 补跑：ALOHA 插接，仿真器钉到权重发布时的版本（gym-aloha 0.1.1 / mujoco 2.3.7 / dm-control 1.0.14 / numpy 1.26.4）

kernel `cindyxue1122/lerobot-act-eval-aloha-mj237`（脚本 `embodiment_datasets/kaggle/lerobot-eval-aloha-mj237/`），Tesla T4，torch 2.7.1+cu126，MUJOCO_GL=egl，总耗时 3.9 h。

| run | seed | 成功 | pc_success | 样本方差 | Bernoulli 方差 | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| aloha_insertion_act_seed1000 | 1000 | 8/50 | 16.0% | 0.137 | 0.134 | [8.3, 28.5] | 243.2 | 2.360 | 4585 s |
| aloha_insertion_act_seed2000 | 2000 | 10/50 | 20.0% | 0.163 | 0.160 | [11.2, 33.0] | 309.1 | 2.780 | 4621 s |
| aloha_insertion_act_seed3000 | 3000 | 7/50 | 14.0% | 0.123 | 0.120 | [7.0, 26.2] | 267.6 | 2.520 | 4587 s |

- 跨 3 个 seed：**均值 16.7%，seed 间标准差 3.1 个百分点**（16%, 20%, 14%）。
- 每个 seed 的 max_reward 分布（0/1/2/3/4 分的 episode 数；3 = 插销触到插座，4 = 插入成功）：seed 1000 [8, 6, 4, 24, 8]，seed 2000 [1, 7, 4, 28, 10]，seed 3000 [5, 6, 4, 28, 7]。
- 与 MuJoCo 3.8.1 下的结果相比：成功率 10.7% → 16.7%，avg_max_reward 1.9–2.1 → 2.4–2.8，avg_sum_reward 181–213 → 243–309。仿真器版本确实有影响，但**仍远低于作者自报的 83%（avg_max_reward 3.55）**，剩下的差距没有找到确定原因。可能的因素：作者的 eval_info.json 是 2025 年初用当时的 lerobot 代码跑的，权重后来被转成了新格式（2025-01-29 仓库重建、2025-03-06 配置重构），lerobot 0.3.3 的 ACT/eval 实现与当时可能有细节差异；像素策略对渲染器（驱动、EGL 实现）差异也敏感。如果要继续排查，最便宜的办法是用同一流程评 `lerobot/act_aloha_sim_transfer_cube_human`（更简单的任务）：如果它也大幅低于自报成绩，问题在代码/权重格式而不是插接任务本身。

**最终给 ALOHA 插接的数字：ACT 成功率 16.7%（50 episodes × 3 seeds，seed 间标准差 3.1 点，单 seed 内 Bernoulli 方差 0.12–0.16），评测环境 = 权重发布时的仿真器版本；MuJoCo 3.8.1 下为 10.7%（标准差 2.3 点）。**

<!-- KAGGLE_RESULTS_END -->

### 2.b 附表：本地 Intel Mac（CPU）结果，lerobot 同一 commit，torch 2.2.2

| run | 权重 | seed | 成功 | pc_success | 样本方差 | Bernoulli 方差 | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| pusht_act_keypoints_seed1000 | alexandersoare/act_pusht_keypoints | 1000 | 33/50 | 66.0% | 0.229 | 0.224 | [52.2, 77.6] | 96.3 | 0.949 | 102 s |
| pusht_act_keypoints_seed2000 | 同上 | 2000 | 29/50 | 58.0% | 0.249 | 0.244 | [44.2, 70.6] | 102.4 | 0.932 | 83 s |
| pusht_act_keypoints_seed3000 | 同上 | 3000 | 33/50 | 66.0% | 0.229 | 0.224 | [52.2, 77.6] | 99.7 | 0.919 | 185 s |
| pusht_act_pixels_seed1000 | arclabmit/pusht_act_model | 1000 | 0/50 | 0.0% | 0 | 0 | [0.0, 7.1] | 45.3 | 0.424 | 139 s |
| pusht_act_pixels_seed2000 | 同上 | 2000 | 0/50 | 0.0% | 0 | 0 | [0.0, 7.1] | 44.4 | 0.371 | 161 s |
| pusht_act_pixels_seed3000 | 同上 | 3000 | 3/50 | 6.0% | 0.058 | 0.056 | [2.1, 16.2] | 41.8 | 0.432 | 174 s |
| pusht_act_pepijn223_seed1000 | pepijn223/act-pusht | 1000 | 0/50 | 0.0% | 0 | 0 | [0.0, 7.1] | 37.6 | 0.403 | 176 s |
| pusht_random_seed1000 | 随机动作 | 1000 | 0/50 | 0.0% | 0 | 0 | [0.0, 7.1] | 1.2 | 0.086 | 115 s |

- keypoints ACT 跨 3 个 seed：均值 63.3%，seed 间标准差 4.6 个百分点（与作者自报的 64%/500 episodes 一致）。
- 像素 ACT（arclabmit）跨 3 个 seed：均值 2.0%，标准差 3.5 个百分点；avg_max_reward≈0.4 说明它会把 T 块推到一半覆盖，但 100 步开环动作块让它完成不了精细对齐。
- ALOHA 插接和 diffusion_pusht 在本地跑到一半被终止（CPU 太慢，改去 Kaggle），没有本地数字。

## 3. 评测流程与命令

Kaggle kernel：`embodiment_datasets/kaggle/lerobot-eval/lerobot_eval_kernel.py`（元数据 `kernel-metadata.json`，slug `cindyxue1122/lerobot-act-eval-pusht-aloha`，GPU + internet，script 类型）。
ALOHA 复现 kernel：`embodiment_datasets/kaggle/lerobot-eval-aloha-mj237/lerobot_eval_aloha_mj237.py`（同一脚本的变体：只跑 ALOHA 3 个 seed，constraints 里追加 `gym-aloha==0.1.1 mujoco==2.3.7 dm-control==1.0.14 numpy<2`）。
kernel 做的事情：

```bash
pip install uv && uv venv --python 3.10 /kaggle/tmp/venv
git clone https://github.com/huggingface/lerobot /kaggle/tmp/lerobot && git -C /kaggle/tmp/lerobot checkout b883328e
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126   # Kaggle P100 是 sm_60，cu128 轮子已不含 Pascal
uv pip install --no-deps torchcodec==0.4.0
uv pip install -e "/kaggle/tmp/lerobot[pusht,aloha]" -c constraints.txt                            # constraints 把 torch 三件套钉死
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy                                   # EGL 不可用时自动退到 osmesa
python convert_old_act_ckpt.py alexandersoare/act_pusht_keypoints /kaggle/tmp/ckpts/act_pusht_keypoints

lerobot-eval --policy.path=/kaggle/tmp/ckpts/act_pusht_keypoints --policy.device=cuda \
  --env.type=pusht --env.obs_type=environment_state_agent_pos \
  --eval.n_episodes=50 --eval.batch_size=50 --seed=1000 --output_dir=/kaggle/working/eval/pusht_act_keypoints_seed1000
lerobot-eval --policy.path=arclabmit/pusht_act_model --policy.device=cuda --env.type=pusht \
  --eval.n_episodes=50 --eval.batch_size=50 --seed=1000 --output_dir=/kaggle/working/eval/pusht_act_pixels_seed1000
lerobot-eval --policy.path=lerobot/act_aloha_sim_insertion_human --policy.device=cuda \
  --env.type=aloha --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=50 --eval.batch_size=50 --seed=1000 --output_dir=/kaggle/working/eval/aloha_insertion_act_seed1000
```

统计口径（`outputs/eval_stats.py`，kernel 里内嵌了同一份逻辑）：
- `pc_success` = 成功 episode 数 / 50；成功判定来自环境 `info["is_success"]`（PushT：T 块与目标重叠 ≥95%；ALOHA 插接：reward 达到 4，即插销插入插座）。
- 方差：每 episode 成功 0/1 的样本方差（n-1）和 Bernoulli 方差 p(1-p)；均值标准误 = sqrt(p(1-p)/50)；95% CI 用 Wilson 区间。
- 跨 seed：同一权重 3 个 seed × 50 episodes 的 pc_success 均值与标准差。
- 同时记录 `avg_sum_reward`、`avg_max_reward`（eval.py 原生输出）与耗时。

## 4. 踩坑与解法

1. **官方没有 ACT-PushT 权重。** 用 Hub API 逐个检查社区权重，最后选 keypoints 版（作者是 LeRobot 维护者，自带 500 episodes 的 eval_info.json 可对照）。像素版社区权重质量都很差，照实记录而不是挑数。
2. **旧格式权重加载。** 2024 年的权重带 `config.yaml` + 旧 `config.json`（`input_shapes` / `input_normalization_modes` 字段），lerobot 0.3.x 的 draccus 配置读不了。`outputs/convert_old_act_ckpt.py` 把它改写成 `type/input_features/output_features/normalization_mapping` 的新格式，`model.safetensors` 的 key 没变，`strict=True` 加载验证通过。
3. **Intel Mac 上的 torch 上限。** 官方 PyTorch macOS x86_64 轮子止于 2.2.2，而 lerobot 0.3.3 要求 torchvision>=0.21（对应 torch 2.6）。本地用 `uv pip install -e ".[all]" --override overrides-macos-x86.txt`（torch==2.2.2、torchvision==0.17.2、numpy<2）强行降级，ACT 只用 resnet18/FrozenBatchNorm 等老 API，能跑；torchcodec 在 darwin x86 被 pyproject 的 marker 排除。ALOHA 在 4 核 CPU 上 50 episodes 跑了 23 分钟还没完，这是转去 Kaggle 的直接原因。
4. **Kaggle 的 GPU 可能是 P100（sm_60）。** Kaggle 自带 torch 是 cu128 构建，已不含 Pascal 的 kernel image，会报 `no kernel image is available`。kernel 里在独立 venv 装 torch 2.7.1+cu126，先做 matmul/conv 自检，失败再退到 2.6.0+cu124，再失败退到 CPU。
5. **渲染后端。** gym-aloha（dm_control/mujoco）需要 OpenGL 上下文：Kaggle 上先试 `MUJOCO_GL=egl`，不行就 `apt-get install libosmesa6` 换 `osmesa`（纯 CPU 渲染，慢但稳）。PushT 用 pygame，设 `SDL_VIDEODRIVER=dummy` 即可。
6. **gym-aloha 的 MuJoCo 大版本变化。** lerobot 0.3.3 的 extra 只写了 `gym-aloha>=0.1.1`，2025-10 之后 pip 会解析到 gym-aloha 0.1.2+（要求 mujoco>=3），而 ALOHA 权重是在 gym-aloha 0.1.1 + mujoco 2.3.7 下训练/评测的。评旧权重时要显式钉 `gym-aloha==0.1.1 mujoco==2.3.7 dm-control==1.0.14 numpy<2`（mujoco 2.3.7 的轮子是 numpy 1.x ABI）。
7. **Kaggle 输出目录。** `/kaggle/working` 下的所有东西都会被当作 kernel 输出保存，venv / 源码 / HF 缓存都放到 `/kaggle/tmp`，只把 `eval/*/eval_info.json`、日志、summary 留在 working。

## 5. 输出文件

- Kaggle 拉回的结果：`embodiment_datasets/kaggle/outputs/lerobot-act-eval-pusht-aloha/`（第一轮，全部任务）和 `embodiment_datasets/kaggle/outputs/lerobot-act-eval-aloha-mj237/`（ALOHA 复现，旧版仿真器）（`summary.md`、`summary.json`、`eval/<run>/eval_info.json`、`logs/*.log`、`env/requirements-freeze.txt`、kernel 日志）。
- 本地 macOS 结果：`embodiment_datasets/lerobot/outputs/eval/<run>/eval_info.json`，日志在 `embodiment_datasets/lerobot/outputs/logs/`，转换后的权重在 `embodiment_datasets/lerobot/outputs/converted_ckpts/`。
- 工具脚本：`embodiment_datasets/lerobot/outputs/convert_old_act_ckpt.py`、`eval_stats.py`、`pusht_random_baseline.py`。
