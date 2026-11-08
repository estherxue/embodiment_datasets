# embodied-eval-notes.md — 具身评测：跑了什么、N 多少、成功率与置信区间、方差从哪来

更新：2026-10-28。所有数字都能在本仓库里找到对应的 `eval_info.json`（每个 episode 的成功标志、seed、回报）；细节与踩坑见 `lerobot/EVAL_NOTES.md` 和 `LIBERO/EVAL_NOTES.md`。
评测全部在 Kaggle GPU kernel（Tesla T4）上跑，kernel 源码在 `kaggle/<kernel>/`，拉回的原始输出在 `kaggle/outputs/<kernel>/`。

## 0. 结论先行

| 跑了什么 | N | 成功率 | 95% CI（Wilson） | 参照 |
|---|---|---|---|---|
| PushT，预训练 ACT（关键点观测，`alexandersoare/act_pusht_keypoints`） | 3 seeds × 50 = 150 | **62.7%** | [54.7, 70.0] | 作者 64.0% / 500 eps |
| PushT，预训练 ACT（像素，社区 `arclabmit/pusht_act_model`） | 150 | 2.0% | [0.7, 5.7] | —（chunk 100 套在 10 fps 任务上，基本不工作） |
| PushT，预训练 Diffusion Policy（`lerobot/diffusion_pusht`，非 ACT 参照） | 50 | 66.0% | [52.2, 77.6] | 官方 65.4% / 500 eps |
| ALOHA 插接，预训练 ACT（`lerobot/act_aloha_sim_insertion_human`），MuJoCo 2.3.7 | 150 | **16.7%** | [11.6, 23.4] | 模型卡 20.6% / 500 eps（p=0.29，一致） |
| 同上，MuJoCo 3.8.1（pip 默认解析到的版本） | 150 | 10.7% | [6.7, 16.6] | 比模型卡显著低（p=0.006） |
| ALOHA transfer cube，预训练 ACT（流程正向对照） | 2 × 50 = 100 | 76.0% | [66.8, 83.3] | 模型卡 83.0% / 500 eps（p=0.10） |
| ALOHA 插接，同一权重，时限改成 demo 长度 500 步 | 2 × 50 = 100 | 21.0% | [14.2, 30.0] | 模型卡 20.6%（p=0.93）；vs 400 步 16.7%（p=0.39） |
| **PushT，自己短训的 ACT（10k 步，3 个训练 seed × 3 个评测 seed）** | 9 × 50 = 450 / 变体 | 关键点 **22.7%**；像素 8.0% | [19.0, 26.8]；[5.8, 10.9] | 100k 步的同配方权重 64%；**训练 seed 间 10% / 25% / 33%** |
| **PushT，ACT 训满 100k 步（关键点 3 个训练 seed；像素 1 个）** | 关键点 3 × 500 = 1500；像素 500 | 关键点 **47.6%**（49.4 / 49.6 / 43.8）；像素 35.0% | [45.1, 50.1]；[30.9, 39.3] | 作者同配方 100k 步 64.0%：差 16 pts，p=2e-10，三个 seed 都显著低，不是 seed 或 checkpoint 的运气；配置 / 模型代码 / 数据逐项一致，作者代码 + 配置原样复跑 45.4%，开 AMP 46.8%，五次运行 47.0% ± 2.5：**公开配方复现不出公开权重**（§3.8） |
| **PushT，同协议短训其他策略（10k 步，训练 seed 1，3 评测 seed × 50）** | 各 150 | Diffusion 关键点 **38.7%**；VQ-BeT 像素 10.0%；Diffusion 像素 7.3%；TD-MPC 关键点 0.0% | [31.2, 46.7]；[6.2, 15.8]；[4.1, 12.7]；[0.0, 2.5] | 同模态 ACT 10k：关键点 22.7%（Diffusion 显著高，p=1e-4）、像素 8.0%（三个像素方法分不开）；各自官方参照 73.3% / 56.0% / 66.0%（§3.7） |
| PushT，官方参照 VQ-BeT（`lerobot/vqbet_pusht`，像素）与 Diffusion Policy 关键点（`lerobot/diffusion_pusht_keypoints`，Hydra 旧格式转换后） | 各 150 | 56.0%；73.3% | [48.0, 63.7]；[65.7, 79.8] | 模型卡 57.0%（p=0.83）/ 71.0%（p=0.58），两份都复现 |
| LIBERO libero_spatial，demo 开环回放（无可加载的官方权重） | 10 任务 × 50 = 500 | 90.2% | [87.3, 92.5] | 回放上限，不是策略成绩 |

方差从哪来（展开见 §4）：n=50 时 **Bernoulli 采样误差 5–7 个百分点**是主项；换评测 seed（= 换一批初始状态）带来的 seed 间标准差只有 2–3 个百分点，**小于**采样误差本身，说明初始状态没有额外的方差分量；仿真器版本能把 ALOHA 插接移动 6 个百分点，是比 seed 大得多的系统性因素；**训练 seed 是最大的一项**：同配方 10k 步的三个 checkpoint 成功率 10% / 25% / 33%，训练 seed 间标准差 11.6 个百分点，是评测 seed 分量（4.6）和采样误差（5.9）的两倍多（§3、§4c）。训满 100k 后训练 seed 分量缩到 3.3 pts（三个 seed 49.4 / 49.6 / 43.8%），但相邻 checkpoint 之间仍在 ±4 到 9 pts 震荡，loss 全程单调降，**选哪个 checkpoint 仍是比评测噪声大的方差来源**（§3.5）。

## 1. 统计口径（所有表共用）

- **N**：每次评测 50 个 episode（`--eval.n_episodes=50 --eval.batch_size=50`），episode i 的环境 seed = `--seed` + i（lerobot-eval 的约定），三次评测用 `--seed` 1000 / 2000 / 3000，即 150 个互不重叠的初始状态。
- **成功**：环境自己的 `info["is_success"]`。PushT：T 块与目标区域重叠 ≥ 95%；ALOHA 插接：reward 达到 4（插销插入插座）；transfer cube：reward 4（左臂接住方块）。
- **成功率** p̂ = k / n；**置信区间**用 Wilson 95% 区间（n=50、p̂≈0.6 时宽约 ±13 个百分点，n=150 时约 ±8，n=500 时约 ±4）。
- **方差**：每个 episode 成功与否是 0/1 变量，单次评测内的样本方差 = Σ(xᵢ − p̂)² / (n − 1) ≈ p̂(1 − p̂)；均值的标准误 SE = sqrt(p̂(1 − p̂)/n)。跨 seed 的标准差是 3 个 p̂ 的样本标准差，用来和 SE 比：如果不比 SE 大，就没有证据说明 seed 之间有"额外"方差。
- **确定性**：ACT 推理不采样（VAE 隐变量取 0），PushT（pymunk）和 MuJoCo 给定初始状态后是确定性的，所以同一权重 + 同一 seed 重跑应逐 episode 完全一致（§3 有实测）。Diffusion Policy 推理有噪声采样，由 torch seed 固定。

## 2. 预训练权重评测（详表）

数据来源：`kaggle/outputs/lerobot-act-eval-pusht-aloha/`（PushT 全部 + ALOHA MuJoCo 3.8.1）、`kaggle/outputs/lerobot-act-eval-aloha-mj237/`（ALOHA MuJoCo 2.3.7）、`kaggle/outputs/lerobot-act-eval-aloha-transfer-mj237/`（transfer cube 对照）、`kaggle/outputs/lerobot-act-eval-aloha-ep500-mj237/`（插接 500 步口径）。lerobot 0.3.3（b883328e），torch 2.7.1+cu126。

| 组 | run | 成功 k/n | 成功率 | Wilson 95% CI | 样本方差 | episode seed 范围 | avg_max_reward |
|---|---|---|---|---|---|---|---|
| PushT · ACT keypoints（alexandersoare） | pusht_act_keypoints_seed1000 | 32/50 | 64.0% | [50.1, 75.9] | 0.235 | 1000–1049 | 0.949 |
| PushT · ACT keypoints（alexandersoare） | pusht_act_keypoints_seed2000 | 30/50 | 60.0% | [46.2, 72.4] | 0.245 | 2000–2049 | 0.932 |
| PushT · ACT keypoints（alexandersoare） | pusht_act_keypoints_seed3000 | 32/50 | 64.0% | [50.1, 75.9] | 0.235 | 3000–3049 | 0.918 |
| PushT · ACT pixels（arclabmit） | pusht_act_pixels_seed1000 | 0/50 | 0.0% | [0.0, 7.1] | 0.000 | 1000–1049 | 0.424 |
| PushT · ACT pixels（arclabmit） | pusht_act_pixels_seed2000 | 0/50 | 0.0% | [0.0, 7.1] | 0.000 | 2000–2049 | 0.371 |
| PushT · ACT pixels（arclabmit） | pusht_act_pixels_seed3000 | 3/50 | 6.0% | [2.1, 16.2] | 0.058 | 3000–3049 | 0.438 |
| PushT · Diffusion（lerobot 官方，参照） | pusht_diffusion_seed1000 | 33/50 | 66.0% | [52.2, 77.6] | 0.229 | 1000–1049 | 0.972 |
| PushT · 随机动作 | pusht_random_seed1000 | 0/50 | 0.0% | [0.0, 7.1] | 0.000 | 1000–1049 | 0.086 |
| PushT · ACT pixels（pepijn223） | pusht_act_pepijn223_seed1000 | 0/50 | 0.0% | [0.0, 7.1] | 0.000 | 1000–1049 | 0.403 |
| ALOHA 插接 · ACT，MuJoCo 3.8.1 | aloha_insertion_act_seed1000 | 6/50 | 12.0% | [5.6, 23.8] | 0.108 | 1000–1049 | 1.940 |
| ALOHA 插接 · ACT，MuJoCo 3.8.1 | aloha_insertion_act_seed2000 | 4/50 | 8.0% | [3.2, 18.8] | 0.075 | 2000–2049 | 2.120 |
| ALOHA 插接 · ACT，MuJoCo 3.8.1 | aloha_insertion_act_seed3000 | 6/50 | 12.0% | [5.6, 23.8] | 0.108 | 3000–3049 | 2.080 |
| ALOHA 插接 · ACT，MuJoCo 2.3.7 | aloha_insertion_act_seed1000 | 8/50 | 16.0% | [8.3, 28.5] | 0.137 | 1000–1049 | 2.360 |
| ALOHA 插接 · ACT，MuJoCo 2.3.7 | aloha_insertion_act_seed2000 | 10/50 | 20.0% | [11.2, 33.0] | 0.163 | 2000–2049 | 2.780 |
| ALOHA 插接 · ACT，MuJoCo 2.3.7 | aloha_insertion_act_seed3000 | 7/50 | 14.0% | [7.0, 26.2] | 0.123 | 3000–3049 | 2.520 |
| ALOHA transfer cube · ACT，MuJoCo 2.3.7（对照） | aloha_transfer_act_seed1000 | 37/50 | 74.0% | [60.4, 84.1] | 0.196 | 1000–1049 | 3.300 |
| ALOHA transfer cube · ACT，MuJoCo 2.3.7（对照） | aloha_transfer_act_seed2000 | 39/50 | 78.0% | [64.8, 87.2] | 0.175 | 2000–2049 | 3.480 |
| ALOHA 插接 · ACT，MuJoCo 2.3.7，episode_length=500 | aloha_insertion_act_ep500_seed1000 | 10/50 | 20.0% | [11.2, 33.0] | 0.163 | 1000–1049 | 2.420 |
| ALOHA 插接 · ACT，MuJoCo 2.3.7，episode_length=500 | aloha_insertion_act_ep500_seed2000 | 11/50 | 22.0% | [12.8, 35.2] | 0.175 | 2000–2049 | 2.820 |

按组汇总（跨 seed 标准差 vs. 单次 n=50 本身的 Bernoulli SE）：

| 组 | seeds | 跨 seed 均值 | 跨 seed 标准差 (pts) | 单次 n=50 的 Bernoulli SE (pts) | 合并 k/N | 合并成功率 | 合并 Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| PushT · ACT keypoints（alexandersoare） | 3 | 62.7% | 2.3 | 6.8 | 94/150 | 62.7% | [54.7, 70.0] |
| PushT · ACT pixels（arclabmit） | 3 | 2.0% | 3.5 | 2.0 | 3/150 | 2.0% | [0.7, 5.7] |
| ALOHA 插接 · ACT，MuJoCo 3.8.1 | 3 | 10.7% | 2.3 | 4.4 | 16/150 | 10.7% | [6.7, 16.6] |
| ALOHA 插接 · ACT，MuJoCo 2.3.7 | 3 | 16.7% | 3.1 | 5.3 | 25/150 | 16.7% | [11.6, 23.4] |
| ALOHA transfer cube · ACT，MuJoCo 2.3.7（对照） | 2 | 76.0% | 2.8 | 6.0 | 76/100 | 76.0% | [66.8, 83.3] |
| ALOHA 插接 · ACT，MuJoCo 2.3.7，episode_length=500 | 2 | 21.0% | 1.4 | 5.8 | 21/100 | 21.0% | [14.2, 30.0] |

说明：
- 官方没有 `lerobot/act_pusht`，PushT 的 ACT 主结果是关键点观测版本（LeRobot 维护者发布，自带 500 episodes 的 eval_info.json 可对照）。像素版社区权重质量差，照实记录。
- ALOHA 插接模型卡自报 20.6%；该仓库 `eval_info.json` 里的 83.0% 是 transfer cube 的文件被错传（sha256 相同），不是插接成绩——这是之前"ALOHA 远低于作者"的全部来源，详见 `lerobot/EVAL_NOTES.md` §6。
- LIBERO（`LIBERO/EVAL_NOTES.md`）：Hub 上没有能被 `libero/lifelong/evaluate.py` 加载的权重，所以做了 demo 开环回放（libero_spatial 10 任务 × 50 条，451/500 = 90.2%，单任务 41–48/50，跨任务标准差 5.7 pts）和随机初始化策略的协议走通（10 任务 × 20 episodes，0/20，预期）。回放在 Mesa 软件渲染和 NVIDIA EGL 下 500 条记录逐条一致，跨 run 方差为 0。

## 3. 短训：ACT 在 PushT 上训 10k 步（训练回路）

kernel `cindyxue1122/lerobot-train-act-pusht`（源码 `kaggle/lerobot-train-act-pusht/`，输出 `kaggle/outputs/lerobot-train-act-pusht/`）。lerobot 0.3.3 的 `lerobot-train`，数据集取 Hub 的 v2.1 分支（main 已是 v3.0，0.3.3 读不了）。

| | kp（关键点） | px（像素） |
|---|---|---|
| 数据集 | `lerobot/pusht_keypoints`（206 条 demo，25,650 帧，10 fps） | `lerobot/pusht`（同一批 demo，96×96 图像） |
| 观测 | 16 维 T 块关键点 + 2 维 agent 位置 | 96×96 RGB + 2 维 agent 位置，resnet18（ImageNet 初始化） |
| ACT 超参 | chunk 16 / n_action_steps 16，4 层 encoder / 4 层 decoder，VAE，kl 10，dropout 0.1 | 同左 |
| 归一化 | ENV/STATE/ACTION min-max | VISUAL mean-std，STATE/ACTION min-max |
| 优化 | AdamW lr 3e-5（backbone 1e-5），wd 1e-4，grad clip 10，batch 64，10,000 步（≈ 25 个 epoch） | 同左 |
| 训练 seed | 1 / 2 / 3 | 1 / 2 / 3 |
| 评测 | 每个 checkpoint × 评测 seed 1000 / 2000 / 3000 × 50 episodes | 同左 |

配方与 `alexandersoare/act_pusht_keypoints` 相同（它训了 100k 步，得 64%）；10k 步只是看训练回路通不通、loss 是否在降、评测流程能否接上，不是为了冲成绩。

### 3.1 训练回路

6 次训练全部正常结束（rc=0，checkpoint 完整），Tesla T4 上关键点变体 30 min（5.5 it/s）、像素变体 37 min（4.5 it/s），整个 kernel 含 19 次评测共 221 min。loss 是 ACT 的 L1 + 10·KL（lerobot 日志里的 `loss`），每 100 步记一次：

| run | step 100 | 1k | 2k | 5k | 10k | 最后 1k 步均值 | 4–5k 步均值 | grad norm 100 → 10k |
|---|---|---|---|---|---|---|---|---|
| kp_train1 | 4.41 | 0.290 | 0.145 | 0.083 | 0.068 | 0.068 | 0.089 | 89 → 3.6 |
| kp_train2 | 5.38 | 0.295 | 0.152 | 0.086 | 0.066 | 0.067 | 0.090 | 108 → 3.4 |
| kp_train3 | 4.43 | 0.279 | 0.144 | 0.084 | 0.065 | 0.067 | 0.089 | 94 → 3.4 |
| px_train1 | 4.45 | 0.275 | 0.121 | 0.054 | 0.036 | 0.037 | 0.060 | 90 → 3.7 |
| px_train2 | 5.42 | 0.280 | 0.128 | 0.054 | 0.036 | 0.036 | 0.061 | 109 → 4.0 |
| px_train3 | 4.39 | 0.266 | 0.123 | 0.053 | 0.035 | 0.036 | 0.060 | 91 → 3.6 |

- loss 单调下降，10k 步时还没平：最后 1k 步的均值比 4–5k 步低 25–40%，没有过拟合迹象（这是训练 loss，lerobot 没有验证集；ACT 不做验证）。
- 三个训练 seed 的 loss 曲线几乎重合（10k 步时 0.065–0.068），**但下面的成功率差 3 倍**：这个阶段的训练 loss 不能用来预测成功率，也不能用来挑 seed。
- 像素变体的 loss 更低（0.036 vs 0.067），成功率却更低，因为两者的 L1 在不同归一化尺度上，不能横比。
- kernel 自带的 `train/<run>/train_log.json` 是 v1 版解析的，1k 步以后的 `step` 字段被 lerobot 的 `1K` 缩写截成 1、2…（仓库里的 kernel 源码已改）；上表由 `kaggle/lerobot-train-act-pusht/train_analysis.py` 直接从 `logs/<run>.log` 重新解析，loss 值本身没错。

### 3.2 成功率：3 个训练 seed × 3 个评测 seed，每格 50 episodes

关键点（kp）：

| 训练 seed \ 评测 seed | 1000 | 2000 | 3000 | 行均值 | 行合并 k/150 |
|---|---|---|---|---|---|
| 1 | 10% (5) | 6% (3) | 14% (7) | 10.0% | 15/150 = 10.0% [6.2, 15.8] |
| 2 | 30% (15) | 26% (13) | 20% (10) | 25.3% | 38/150 = 25.3% [19.0, 32.8] |
| 3 | 30% (15) | 30% (15) | 38% (19) | 32.7% | 49/150 = 32.7% [25.7, 40.5] |
| 列均值 | 23.3% | 20.7% | 24.0% | **22.7%** | 102/450 = 22.7% [19.0, 26.8] |

像素（px）：

| 训练 seed \ 评测 seed | 1000 | 2000 | 3000 | 行均值 | 行合并 k/150 |
|---|---|---|---|---|---|
| 1 | 6% (3) | 4% (2) | 4% (2) | 4.7% | 7/150 = 4.7% [2.3, 9.3] |
| 2 | 6% (3) | 12% (6) | 14% (7) | 10.7% | 16/150 = 10.7% [6.7, 16.6] |
| 3 | 6% (3) | 10% (5) | 10% (5) | 8.7% | 13/150 = 8.7% [5.1, 14.3] |
| 列均值 | 6.0% | 8.7% | 9.3% | **8.0%** | 36/450 = 8.0% [5.8, 10.9] |

- 10k 步 vs 100k 步（同配方 `alexandersoare/act_pusht_keypoints`，§2 测得 62.7%）：22.7% 差得很远（z=−9，p≈1e-19），符合预期——10k 步只是 1/10 的训练预算，目的是看回路。
- 像素 10k 步的 8.0% 已经高于社区 `arclabmit/pusht_act_model` 的 2.0%（p=0.01），后者的问题（chunk 100）不是训练不足。
- avg_max_reward（T 块与目标区域的最大重叠率）：kp 0.69–0.88，px 0.47–0.69；成功要求 ≥ 0.95，所以大多数 episode 是"推到一大半但没对齐"。

### 3.3 确定性检查

`kp_train1` 用评测 seed 1000 跑两次（`kp_train1_eval1000` 与 `_repeat`）：5/50 vs 5/50，50 个 episode 的 success / max_reward / sum_reward 逐条相同。同一权重 + 同一 seed 的评测是确定的，和 §1 的预期一致；评测 seed 之间的差异因此全部来自初始状态。

### 3.4 方差分解（单因素方差分析，训练 seed 为因子，评测 seed 为重复）

| 变体 | 训练 seed 间标准差（3 个行均值） | 评测 seed 间标准差（checkpoint 内合并） | Bernoulli SE（n=50） | F(2,6) | p | 训练 seed 方差分量（标准差） |
|---|---|---|---|---|---|---|
| kp | 11.6 pts | 4.6 pts | 5.9 pts | 19.2 | 0.0025 | 11.3 pts |
| px | 3.1 pts | 2.8 pts | 3.8 pts | 3.5 | 0.10 | 2.6 pts |

- **关键点变体：训练 seed 是最大的方差来源。** 三个 checkpoint 的行均值 10 / 25 / 33% 之间的差异远超评测噪声（F=19，p=0.003），训练 seed 1 与 3 的合并成功率 15/150 vs 49/150 的双比例检验 p=0.013。训练 seed 分量的标准差约 11 pts，是评测 seed 分量（4.6）和 n=50 采样误差（5.9）的两倍多；而评测 seed 分量本身和 §4(b) 一样，没有超出采样误差。
- **像素变体：成功率太低（8%），分不开。** 训练 seed 间 3.1 pts、评测 seed 间 2.8 pts，都在 Bernoulli SE 3.8 pts 以内，F=3.5 不显著。
- 含义：**从头训的 checkpoint 单报一个数字是没有锚的。** 在 10k 步这个阶段，换一个训练 seed 可以把 PushT 成功率从 10% 变成 33%，比把 n 从 50 加到 450 能缩掉的误差还大。比较两个训练配方时，必须报多个训练 seed 的均值和 seed 间标准差，评测 seed 和 n 反而是次要的。100k 步时这个分量会不会收窄，这里没有数据（§2 的 100k 步权重只有一个）。

### 3.5 训满 100k 步：关键点 3 个训练 seed，像素 1 个（kernel `cindyxue1122/lerobot-train-act-pusht-100k-{kp,px,kp-seed2,kp-seed3}`，源码 `kaggle/lerobot-train-act-pusht-100k/`，汇总脚本 `analyze_100k.py`）

做法：seed 1 的两个变体从 10k checkpoint 热启动再训 90k（`lerobot-train --policy.path=…`：权重、归一化统计量、模型配置从 checkpoint 读，AdamW 动量重新开始，见 §3.6）；关键点 seed 2、3 从零训满 100k（上一轮只保留了 seed 1 的 checkpoint）。每 10k 步存一个 checkpoint，训练进行中就在同一块 GPU 上用 3 个评测 seed × 50 episodes 评它；100k checkpoint 用 10 个 seed × 50 = 500 episodes，与作者 eval_info.json 的 N 相同。T4 上 5.2–6.2 it/s，每条轨迹 4.5–5 h（seed 1 的 summary.md 里 `killed_by_budget=True` 是 v1 脚本在最终评测后例行 kill 已退出进程留下的标签，`steps_done=90000, rc=0` 才是实际状态，模板已改）。顺带一个复现检查：seed 2、3 从零训的 10k checkpoint 评出来与 §3.2 里 10k kernel 的 kp_train2 / kp_train3 逐 seed 相同（25.3% / 32.7%），同一 seed 在不同 kernel 里训出的权重一致。

**成功率随步数（每格 3 评测 seed × 50 = 150 episodes，%）**

| 训练 seed | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k | 60k–100k 均值 | 100k，500 episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1（热启动） | 10.0 | 10.0 | 24.0 | 18.0 | 52.0 | 44.7 | 48.7 | 41.3 | 30.0 | 48.7 | 42.7 | 247/500 = 49.4% [45.0, 53.8] |
| 2（从零） | 25.3 | 24.7 | 24.7 | 24.0 | 26.7 | 36.0 | 39.3 | 42.0 | 52.7 | 45.3 | 43.1 | 248/500 = 49.6% [45.2, 54.0] |
| 3（从零） | 32.7 | 10.7 | 26.0 | 30.7 | 34.0 | 38.0 | 34.0 | 30.0 | 38.0 | 42.0 | 36.4 | 219/500 = 43.8% [39.5, 48.2] |
| 像素，seed 1（热启动） | 4.7 | 16.7 | 16.7 | 24.0 | 28.0 | 28.0 | 30.7 | 29.3 | 34.7 | 32.7 | 31.1 | 175/500 = 35.0% [30.9, 39.3] |

loss（L1 + 10·KL，最后 1k 步均值）三条关键点轨迹几乎重合：10k 0.067 → 50k 0.038 → 100k 0.027–0.029，100k 时还在降；像素 0.036 → 0.012。

**三条轨迹合起来看到的东西**

- **与作者的差距是系统性的，不是 seed 或 checkpoint 的运气。** 100k 处三个 seed 合并 714/1500 = 47.6% [45.1, 50.1]，作者 320/500 = 64.0%，z=−6.4，p=2e-10；三个 seed 各自都显著低（p ≤ 4e-6）；三条轨迹 30 个 checkpoint 里最好的一个是 52.7%，没有一个碰到作者的区间 [59.7, 68.1]。作者权重在本流程重测是 62.7%（§2），评测侧没问题，差异在训练侧。候选原因：作者用的是 2024-07 的 lerobot（Hydra 配置）而这里是 0.3.3，两者之间的 ACT 训练代码、数据集格式（v1 → v2.1 转换）或默认超参可能有差异，这里没有逐项对比；要确认需要用作者的 config.yaml 在当时的 lerobot 版本上重训一次（§3.8 做了：45.4%，这些候选全部排除）。
- **训练 seed 分量随训练变小。** 10k 时三个 seed 是 10 / 25 / 33%（标准差 11.6 pts，§3.4）；100k 时是 49.4 / 49.6 / 43.8%（标准差 3.3 pts，方差分量 2.5 pts，F(2,27)=2.4，不显著）。训练越充分，seed 之间越像。
- **checkpoint 之间的震荡不随训练消失。** 50k 到 100k 六个 checkpoint 的标准差，三个 seed 分别 7.9 / 8.8 / 4.2 pts，都不低于 n=150 的采样误差 4 pts；seed 1 的 90k 与 100k 差 19 pts（p=0.0009），seed 2 的 50k 与 90k 差 26 pts。loss 全程单调下降，成功率却从 50k 起就在平台上来回跳。像素版没有这一层（2.7 pts，低于采样误差 3.8）。
- 对评测方法的含义：**报从头训的结果时，至少报多个训练 seed，并且用最后 k 个 checkpoint 的均值或事先定好的选择规则，不要报单个 checkpoint**。这里 60k–100k 五个 checkpoint 的均值是 42.7 / 43.1 / 36.4%，平均 40.7%，比任何一条轨迹上"最好的那个 checkpoint"保守 10 个点。
- 热启动的影响看不出来：seed 1（热启动）与 seed 2（从零）的 100k 结果 49.4% 与 49.6%，差异远小于 seed 间噪声。

### 3.6 热启动时 AdamW 动量重置是什么

AdamW 给每个参数维护一阶矩 m（梯度的滑动平均，β1=0.9，记忆约 10 步）和二阶矩 v（梯度平方的滑动平均，β2=0.999，记忆约 1000 步），步长由 m̂/√v̂ 决定；这两个量存在 checkpoint 的 `training_state/optimizer_state.safetensors` 里，体积是权重的两倍。`--policy.path` 只加载权重和归一化统计量，m、v 从零开始，前几步退化成"每个参数不论梯度大小都走约 lr 的一步"。日志里的瞬态：关键点 loss 0.067 → 0.080、grad norm 3.5 → 6.2（重启后第一个 100 步），第二个 100 步起 0.066 / 3.2；像素 0.036 → 0.050、3.8 → 6.4，同样 200 步内恢复。90k 步里占 0.1%，权重衰减是解耦无状态的，ACT 没有 lr 调度器，所以这是与严格 resume 的唯一区别，终点上看不出影响。

### 3.7 其他策略在 PushT 上，同一协议（kernel `cindyxue1122/lerobot-train-others-pusht-{cpu,gpu}`，源码 `kaggle/lerobot-train-others-pusht/`，汇总脚本 `analyze_others.py`）

做法：ACT 之外再训四个 lerobot 自带策略，协议与 §3 完全相同（10k 步、batch 64、训练 seed 1、评测 seed 1000/2000/3000 × 50），外加两份官方权重用同一评测流程作参照。周 GPU 配额在 10-30 用完，便宜的部分先在 CPU kernel 上跑（10-30，4 核无 GPU），其余在 10-31 配额刷新后由 GPU kernel（T4，4.2 h）跑。

| 策略 | 参数量 | 观测 | 评测 seed 1000 / 2000 / 3000 | 合并 | Wilson 95% CI | 同模态 ACT 10k（3 训练 seed 合并） | 官方参照（同流程重测 / 模型卡） |
|---|---|---|---|---|---|---|---|
| TD-MPC（离线，评测时 CEM 规划） | 2.7M | 关键点 | 0 / 0 / 0% | 0/150 = 0.0% | [0.0, 2.5] | 22.7%（p=2e-10） | 无 |
| Diffusion Policy，DDIM-10 | 249M | 关键点 | 46 / 28 / 42% | 58/150 = **38.7%** | [31.2, 46.7] | 22.7%（p=1e-4，高）；ACT 训练 seed 1 是 10.0% | `lerobot/diffusion_pusht_keypoints` 73.3% / 71.0% |
| VQ-BeT（2k VQ-VAE + 8k 策略步） | 37.5M | 像素 | 6 / 8 / 16% | 15/150 = 10.0% | [6.2, 15.8] | 8.0%（p=0.45） | `lerobot/vqbet_pusht`（250k 步）56.0% / 57.0% |
| Diffusion Policy，DDPM-100 | 263M | 像素 | 10 / 6 / 6% | 11/150 = 7.3% | [4.1, 12.7] | 8.0%（p=0.79） | `lerobot/diffusion_pusht` 66.0%（50 eps）/ 65.4% |

训练侧：三个 GPU 训练都正常收敛。loss（每 100 步均值）Diffusion 关键点 0.659（100 步）→ 0.052（1k）→ 0.035（5k）→ 0.023（10k），像素 0.633 → 0.054 → 0.040 → 0.027，10k 时都还在降；VQ-BeT 前 2k 步只训 VQ-VAE（loss 0.28 → 0.063），切到策略阶段后 loss 是 3332 → 504（`offset_loss_weight=10000` 把偏移项放大了，量级正常）。T4 上 2.2–2.5 it/s，每个 10k 步 65–75 min。评测速度：Diffusion 关键点 DDIM-10 6.3 环境步/s，VQ-BeT 3.9，Diffusion 像素 DDPM-100 1.2，TD-MPC 带 CEM 规划 1.1（CPU 上是 40 s 一步，GPU 快 40 多倍）。TD-MPC 的训练在 CPU kernel 上做（0.2 s/步，138 min，loss 1.13 → 0.089（1k）→ 0.048（10k）），checkpoint 经 Kaggle 数据集 `cindyxue1122/tdmpc-pusht-kp-10k` 交给 GPU kernel 评；VQ-BeT 像素在 CPU 上 9 s/步、5.4 h 只到 4k 步就撞上 kernel 的 12 h 上限，重新在 GPU 上训。

看到的东西：

- **同样 10k 步、同样数据，关键点 Diffusion Policy 38.7% 显著高于 ACT 的 22.7%（p=1e-4），是 ACT 训练 seed 1（10.0%）的近 4 倍。** 但这是单个训练 seed 的数字：§3.4 测到 ACT 在 10k 时训练 seed 间标准差 11.6 pts，Diffusion 这边没有测这一项；三个评测 seed 46 / 28 / 42% 的离散（标准差 9.5 pts，略大于 n=50 的采样误差 6.9，卡方同质性 p=0.15）也提示这个 checkpoint 本身不稳。与它自己的官方参照 73.3% 比还差 35 pts（p=2e-9），10k 步远没训完。
- **像素输入下三个方法在 10k 步都没起来**：ACT 8.0%、VQ-BeT 10.0%、Diffusion 7.3%，两两都分不开（p ≥ 0.4），而它们的官方参照是 57% / 65%。10k 步对 ResNet18 从头学视觉特征不够，与 §3.5 里像素 ACT 要到 50k 步才过 28% 一致。VQ-BeT 官方配方是 20k 步 VQ-VAE + 230k 步策略，这里是 2k + 8k。
- **TD-MPC 0/150，avg_max_reward 只有 0.07–0.11，T 块几乎没被推动。** TD-MPC 本来是在线 RL 方法（学动力学、奖励和 Q 模型，评测时用 CEM 在模型里规划）；0.3.3 的 `lerobot-train` 只有离线阶段，这里是 10k 步离线拟合 206 条 demo。本地用同一 checkpoint 关掉规划（`--policy.use_mpc=false`，只用策略先验 π，CPU 上 10 环境步/s）再评 3 × 50：仍是 0/150，但 avg_max_reward 升到 0.21 / 0.26 / 0.22（带规划是 0.07 / 0.11 / 0.10）。所以两头都有问题：10k 步离线训出的策略本身推不成功，CEM 在学得不准的动力学 / 奖励 / Q 模型上规划还把它弄得更差（规划专门挑模型估高的动作，这是离线 RL 里模型误差被利用的典型现象）。要让 TD-MPC 在 PushT 上工作需要在线交互阶段，这超出本协议；结果文件在 `kaggle/outputs/lerobot-train-others-pusht-gpu/local_tdmpc_no_mpc/`。
- 评测流程对三种新策略都成立：两份官方参照 56.0% / 73.3% 与模型卡 57.0% / 71.0% 一致（p=0.83 / 0.58），Hydra 旧格式的 Diffusion 权重转换（156 个张量逐一对上）没有丢东西；参照的跨评测 seed 标准差 4.0 / 3.1 pts，在采样误差内，与 §4(b) 一致。
- 对评测方法的含义：跨方法比较时，每个方法只有一个训练 seed 的话，方法间差异和训练 seed 噪声（10k 时 ACT 测到 11.6 pts）混在一起；这里 Diffusion 比 ACT 高 16 pts，大于这个噪声，像素三方的 2–3 pts 差异则完全在噪声里。

### 3.8 47.6% 对 64.0%：排查训练侧差异（2026-11-08；kernel `cindyxue1122/lerobot-train-act-pusht-100k-kp-seed2-amp`、`cindyxue1122/lerobot-train-act-pusht-oldcode`，源码 `kaggle/lerobot-train-act-pusht-100k/`、`kaggle/lerobot-train-act-pusht-oldcode/`）

§3.5 的结论是差距在训练侧。先把作者 Hub 上的配置、作者上传当天的 lerobot 代码和数据集的两个修订版逐项对比，再用两个 kernel 做确认实验。

**逐项对比**

| 项 | 作者（2024-07） | 这里（lerobot 0.3.3） | 一致？ |
|---|---|---|---|
| 配置：模型结构（dim 512 / 8 头 / FF 3200 / 编码 4 层 / 解码 4 层 / VAE 编码 4 层 / latent 32 / dropout 0.1 / kl 10）、chunk 16 / n_action_steps 16、归一化（ENV / STATE / ACTION 全部 min-max）、lr 3e-5 / backbone 1e-5 / weight decay 1e-4 / grad clip 10、batch 64、100k 步、无 lr 调度 | `config.yaml`（Hub 上保存的 Hydra 完整配置） | 各 kernel 存下的 `train_config.json` | 全部一致 |
| 配置：`use_amp`（混合精度） | true | false | **不同** |
| 配置：训练 seed / num_workers | 100000 / 8 | 1、2、3 / 4 | 不同，但 §3.5 三个 seed 都低 14–20 pts |
| ACT 模型代码 | commit 471eab3d（PR #314，2024-07-11，作者 07-10 上传权重的次日合入，作者用的就是这个分支） | b883328e | 关键点路径上**没有数学改动**：此后的改动是 `n_vae_encoder_layers` 修复（PR #400，两边都是 4，无影响）、推理时关 VAE（PR #1588，只影响评测期）、图像多相机路径重构、temporal ensembling（未用） |
| 优化器 | AdamW，backbone 单独一个参数组，无调度器 | 同 | 一致 |
| 数据 | `lerobot/pusht_keypoints` revision v1.5（老代码的 `CODEBASE_VERSION`） | revision v2.1 | 25,650 帧的 environment_state / state / action / timestamp / reward **逐元素相同**（max \|diff\| = 0），min / max 统计量精确相同，mean / std 差 1e-4（min-max 归一化用不到）；比对脚本 `kaggle/lerobot-train-act-pusht-oldcode/data_check/` |

配置、模型、数据三项都排除，只剩 (a) AMP，(b) 两版代码在训练循环 / 数据加载上某个上表没覆盖的差异，(c) 作者那一次运行本身。

**确认实验**（同日并行，T4，各约 4.5 h；每 10k 步评 3 评测 seed × 50，100k 处 10 × 50 = 500；汇总脚本 `analyze_100k.py` 已并入这两行）

| 运行 | 代码 / 数据 / 配置 | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k | 60k–100k 均值 | 100k，500 eps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 2 + AMP | 0.3.3 / v2.1 / §3.5 的 seed 2 只加 `--policy.use_amp=true`（只动一个变量） | 8.0 | 32.7 | 32.0 | 22.7 | 38.0 | 43.3 | 33.3 | 39.3 | 44.0 | 47.3 | 41.5 | 234/500 = 46.8% [42.5, 51.2] |
| 作者代码 + 作者配置原样 | lerobot 471eab3d、torch 2.3.1 / v1.5 / `config.yaml` 原样（seed 100000、AMP、Hydra `train.py`），checkpoint 转成 0.3.3 格式后评 | 6.0 | 10.7 | 26.0 | 18.7 | 18.0 | 27.3 | 37.3 | 41.3 | 41.3 | 42.0 | 37.9 | 227/500 = 45.4% [41.1, 49.8] |

- **AMP 不是原因。** seed 2 开 AMP 46.8% 对不开 49.6%，差异在噪声内（p=0.38）；与三个 seed 合并的 47.6% 分不开（p=0.76）。训练速度 6.2 对 6.0 it/s，这个模型小，AMP 基本不提速。
- **代码版本、数据修订、随机种子、torch 版本也不是原因。** 作者的代码、配置、数据、seed 原样复跑，100k 处 45.4%，与我们 0.3.3 的三个 seed 分不开（p=0.39），离作者自报的 64.0% 差 5.9 个标准误（p=3.5e-9）。loss 曲线（10k 0.066 → 100k 0.029）和成功率随步数的形状与 §3.5 一样，一路都在 §3.5 三条轨迹的范围内。
- **五次独立的 100k 训练**（0.3.3 三个 seed、0.3.3 + AMP、作者代码）全在 43.8%–49.6%，合并 1175/2500 = 47.0% [45.0, 49.0]，运行间标准差 2.5 pts；64.0% 在这组数里是 6.8 个标准差外的点，"那次运行手气好"解释不通。
- 所以：**公开的配方（`config.yaml` + 当时的代码 + 数据集）在这里稳定给出 47% 左右，复现不出公开权重的 64%。** 权重本身是真的好（本流程重测 62.7%，§2），差异只能来自公开配方没记录的东西：作者训练时用的是尚未合入的 PR 分支，分支状态可能与合入的 commit 不同；或训练过程（实际步数、是否多次 resume、从多个运行里挑选）与 `config.yaml` 所述不同。这一步无法再从外部推进，要问作者。
- 对评测方法的含义：**模型卡上的数字是"这份权重"的成绩，不是"这个配方"的成绩。** 要引用一个配方的期望成功率，得自己跑多个 seed：这里 ACT 关键点配方 100k 步的期望值是 47% ± 2.5（5 次运行），不是 64%；用模型卡数字当配方基线会高估 17 个点。

## 4. 方差从哪来

**(a) Bernoulli 采样（episode 数 n）。** 单次评测的成功率是 n 个 0/1 的均值，SE = sqrt(p(1−p)/n)。这是 n=50 时最大的不确定性来源：

| p | n=50 | n=150 | n=500 |
|---|---|---|---|
| 0.63（PushT 关键点 ACT） | SE 6.8 pts，Wilson 宽 ±13 | 3.9 | 2.2 |
| 0.17（ALOHA 插接） | SE 5.3 pts，Wilson 宽 ±11 | 3.1 | 1.7 |
| 0.76（transfer cube） | SE 6.0 pts | 3.5 | 1.9 |

所以"64% vs 60%"这种差异在 n=50 下没有信息量；要分辨 5 个百分点的差异，n 需要几百。

**(b) 评测 seed = 初始状态。** PushT 的 `reset(seed)` 用 seed 采样 T 块位姿和 agent 位置（本地验证：同 seed 两次 reset 初始状态逐元素相同，seed 1000 与 1001 的状态最大差约 290 像素；50 个连续 seed 的 agent 初始位置标准差约 100–120 像素，T 块关键点约 100 像素）。ALOHA 的 `reset(seed)` 同样采样插销/插座（或方块）的位姿。策略和仿真器本身确定（§1），所以**换评测 seed 的全部效果就是换一批初始状态**。实测跨 seed 标准差 2.3–3.1 pts，都小于对应的 SE 4.4–6.8 pts（§2 汇总表），也就是三批初始状态之间的差异完全能用采样噪声解释，没有"某批初始状态特别难"的额外分量。

**(c) 训练 seed。** 同一配方换训练 seed（初始化、数据顺序、dropout）得到不同权重，这是预训练权重评测看不到的分量。§3.4 用 3 个训练 seed × 3 个评测 seed 的 3×3 设计来拆：关键点变体训练 seed 间标准差 11.6 pts（方差分量 11.3 pts），评测 seed 间 4.6 pts，Bernoulli SE 5.9 pts，F(2,6)=19.2，p=0.003——**训练 seed 是从头训时最大的方差来源**，是另外两项的两倍多；像素变体成功率只有 8%，三项都在 3 pts 左右，分不开（p=0.10）。预训练权重只有一个训练 seed，所以 §2 的所有 CI 都不包含这一项：它们是"这份权重"的成功率区间，不是"这个配方"的。训满 100k 的三条关键点轨迹（§3.5）显示训练 seed 分量随训练变小（10k 时 11.6 pts → 100k 时 3.3 pts，F(2,27)=2.4 不显著），但第三层仍在：同一轨迹相邻 checkpoint 之间的差异，50k 到 100k 三个 seed 的标准差 7.9 / 8.8 / 4.2 pts，都不低于 n=150 的采样误差 4 pts；像素版没有这一层（2.7 pts）。

**(d) 仿真器与硬件。** 同一权重、同一 seed：MuJoCo 3.8.1 → 2.3.7 把 ALOHA 插接从 10.7% 提到 16.7%（6 pts，比任何 seed 效应都大）；PushT 关键点 ACT 在本地 CPU（torch 2.2.2）与 Kaggle GPU 上同 seed 的结果 66/58/66 vs 64/60/64——个别 episode 因浮点差异翻转，总体在 CI 内。这类差异是系统性的，不会随 n 变小，只能靠钉版本来消除（三个 kernel 和本地 venv 现在都钉在 gym-aloha 0.1.1 / mujoco 2.3.7）。

**(e) 评测协议。** 时限：ALOHA 插接 demo 每条 500 帧，而模型卡评测用 400 步（Hub discussion #1 报告 500 步时 23.2% vs 20.6%）；我们的插接数字都是 400 步口径，500 步补测（kernel `cindyxue1122/lerobot-act-eval-aloha-ep500-mj237`，同权重、同 seed 1000/2000、MuJoCo 2.3.7）得到 10/50 和 11/50，合并 21.0% [14.2, 30.0]，与模型卡 20.6% 重合（p=0.93），比 400 步口径高约 4 个点但不显著（p=0.39）。同 seed 逐一看，seed 1000 从 8/50 变 10/50、seed 2000 从 10/50 变 11/50：初始状态没变，多出来的成功是原本到 400 步还卡在 3 分（触到插座未插入）的 episode。所以时限是一个会系统性移动几个点的协议参数，报数字时要带上 episode_length。LIBERO 则相反：每个任务固定 50 个初始状态（`.pruned_init`），不随 seed 采样，所以 LIBERO 的跨 run 方差只来自策略本身（回放时为 0）。

## 5. 产出物索引

- `kaggle/outputs/lerobot-act-eval-pusht-aloha/eval/<run>/eval_info.json`：PushT 全部 + ALOHA MuJoCo 3.8.1，12 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-mj237/`：ALOHA 插接 MuJoCo 2.3.7，3 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-transfer-mj237/`：transfer cube 对照，2 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-ep500-mj237/`：插接 500 步口径，2 次评测。
- `kaggle/outputs/lerobot-train-act-pusht/`：短训 6 个 checkpoint 的训练日志（`train/<run>/train_log.json`，每 100 步的 loss / grad norm）、18+1 次评测、seed-1 的两个 checkpoint（`ckpts/`，`model.safetensors` 不入 git）。
- `kaggle/outputs/lerobot-train-act-pusht-100k-kp/`、`.../-100k-px/`、`.../-100k-kp-seed2/`、`.../-100k-kp-seed3/`：训满 100k 的训练日志（`train/<variant>_cont/train_log.json`，步数已按行号修正）、每 10k 步 3 次评测 + 100k 步 10 次评测的 `eval_info.json`、100k 步 checkpoint（`ckpts/`，`model.safetensors` 不入 git）。
- `kaggle/outputs/lerobot-train-others-pusht-cpu/`：两份官方参照（VQ-BeT、Diffusion 关键点）各 3 次评测、TD-MPC 关键点 10k 步训练日志与 checkpoint（`model.safetensors` 不入 git；Kaggle 数据集 `cindyxue1122/tdmpc-pusht-kp-10k`）、VQ-BeT 像素版训到 4k 步的日志。
- `kaggle/outputs/lerobot-train-others-pusht-gpu/`：VQ-BeT 像素、Diffusion 关键点、Diffusion 像素各 10k 步的训练日志（`train/<run>/train_log.json`）、12 次评测（含 TD-MPC checkpoint 的 3 次）、VQ-BeT checkpoint（`ckpts/`，`model.safetensors` 不入 git；两个 Diffusion 的 1 GB 权重没留）。对比表和 p 值由 `kaggle/lerobot-train-others-pusht/analyze_others.py kaggle/outputs` 重算。
- `kaggle/outputs/lerobot-train-act-pusht-100k-kp-seed2-amp/`：§3.8 的 AMP 对照（seed 2 + `use_amp=true`），格式同上。
- `kaggle/outputs/lerobot-train-act-pusht-oldcode/`：§3.8 作者代码 + 配置原样复跑：训练日志（`train/old_act_kp/train_log.json`）、每 10k 步 3 次 + 100k 步 10 次评测、100k checkpoint 的原格式与转换后两份（`model.safetensors` 不入 git）、`env/author_config.yaml`、两套 venv 的 `requirements-freeze-*.txt`。
- `kaggle/outputs/libero-spatial-eval/run5/`：LIBERO 回放与协议走通的逐条记录。
- 统计脚本：`lerobot/outputs/eval_stats.py`（Wilson CI、方差）；各 kernel 内嵌同一份逻辑生成 `summary.md`。
