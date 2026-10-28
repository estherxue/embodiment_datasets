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
| LIBERO libero_spatial，demo 开环回放（无可加载的官方权重） | 10 任务 × 50 = 500 | 90.2% | [87.3, 92.5] | 回放上限，不是策略成绩 |

方差从哪来（展开见 §4）：n=50 时 **Bernoulli 采样误差 5–7 个百分点**是主项；换评测 seed（= 换一批初始状态）带来的 seed 间标准差只有 2–3 个百分点，**小于**采样误差本身，说明初始状态没有额外的方差分量；仿真器版本能把 ALOHA 插接移动 6 个百分点，是比 seed 大得多的系统性因素；**训练 seed 是最大的一项**：同配方 10k 步的三个 checkpoint 成功率 10% / 25% / 33%，训练 seed 间标准差 11.6 个百分点，是评测 seed 分量（4.6）和采样误差（5.9）的两倍多（§3、§4c）。

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

## 4. 方差从哪来

**(a) Bernoulli 采样（episode 数 n）。** 单次评测的成功率是 n 个 0/1 的均值，SE = sqrt(p(1−p)/n)。这是 n=50 时最大的不确定性来源：

| p | n=50 | n=150 | n=500 |
|---|---|---|---|
| 0.63（PushT 关键点 ACT） | SE 6.8 pts，Wilson 宽 ±13 | 3.9 | 2.2 |
| 0.17（ALOHA 插接） | SE 5.3 pts，Wilson 宽 ±11 | 3.1 | 1.7 |
| 0.76（transfer cube） | SE 6.0 pts | 3.5 | 1.9 |

所以"64% vs 60%"这种差异在 n=50 下没有信息量；要分辨 5 个百分点的差异，n 需要几百。

**(b) 评测 seed = 初始状态。** PushT 的 `reset(seed)` 用 seed 采样 T 块位姿和 agent 位置（本地验证：同 seed 两次 reset 初始状态逐元素相同，seed 1000 与 1001 的状态最大差约 290 像素；50 个连续 seed 的 agent 初始位置标准差约 100–120 像素，T 块关键点约 100 像素）。ALOHA 的 `reset(seed)` 同样采样插销/插座（或方块）的位姿。策略和仿真器本身确定（§1），所以**换评测 seed 的全部效果就是换一批初始状态**。实测跨 seed 标准差 2.3–3.1 pts，都小于对应的 SE 4.4–6.8 pts（§2 汇总表），也就是三批初始状态之间的差异完全能用采样噪声解释，没有"某批初始状态特别难"的额外分量。

**(c) 训练 seed。** 同一配方换训练 seed（初始化、数据顺序、dropout）得到不同权重，这是预训练权重评测看不到的分量。§3.4 用 3 个训练 seed × 3 个评测 seed 的 3×3 设计来拆：关键点变体训练 seed 间标准差 11.6 pts（方差分量 11.3 pts），评测 seed 间 4.6 pts，Bernoulli SE 5.9 pts，F(2,6)=19.2，p=0.003——**训练 seed 是从头训时最大的方差来源**，是另外两项的两倍多；像素变体成功率只有 8%，三项都在 3 pts 左右，分不开（p=0.10）。预训练权重只有一个训练 seed，所以 §2 的所有 CI 都不包含这一项：它们是"这份权重"的成功率区间，不是"这个配方"的。

**(d) 仿真器与硬件。** 同一权重、同一 seed：MuJoCo 3.8.1 → 2.3.7 把 ALOHA 插接从 10.7% 提到 16.7%（6 pts，比任何 seed 效应都大）；PushT 关键点 ACT 在本地 CPU（torch 2.2.2）与 Kaggle GPU 上同 seed 的结果 66/58/66 vs 64/60/64——个别 episode 因浮点差异翻转，总体在 CI 内。这类差异是系统性的，不会随 n 变小，只能靠钉版本来消除（三个 kernel 和本地 venv 现在都钉在 gym-aloha 0.1.1 / mujoco 2.3.7）。

**(e) 评测协议。** 时限：ALOHA 插接 demo 每条 500 帧，而模型卡评测用 400 步（Hub discussion #1 报告 500 步时 23.2% vs 20.6%）；我们的插接数字都是 400 步口径，500 步补测（kernel `cindyxue1122/lerobot-act-eval-aloha-ep500-mj237`，同权重、同 seed 1000/2000、MuJoCo 2.3.7）得到 10/50 和 11/50，合并 21.0% [14.2, 30.0]，与模型卡 20.6% 重合（p=0.93），比 400 步口径高约 4 个点但不显著（p=0.39）。同 seed 逐一看，seed 1000 从 8/50 变 10/50、seed 2000 从 10/50 变 11/50：初始状态没变，多出来的成功是原本到 400 步还卡在 3 分（触到插座未插入）的 episode。所以时限是一个会系统性移动几个点的协议参数，报数字时要带上 episode_length。LIBERO 则相反：每个任务固定 50 个初始状态（`.pruned_init`），不随 seed 采样，所以 LIBERO 的跨 run 方差只来自策略本身（回放时为 0）。

## 5. 产出物索引

- `kaggle/outputs/lerobot-act-eval-pusht-aloha/eval/<run>/eval_info.json`：PushT 全部 + ALOHA MuJoCo 3.8.1，12 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-mj237/`：ALOHA 插接 MuJoCo 2.3.7，3 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-transfer-mj237/`：transfer cube 对照，2 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-ep500-mj237/`：插接 500 步口径，2 次评测。
- `kaggle/outputs/lerobot-train-act-pusht/`：短训 6 个 checkpoint 的训练日志（`train/<run>/train_log.json`，每 100 步的 loss / grad norm）、18+1 次评测、seed-1 的两个 checkpoint（`ckpts/`，`model.safetensors` 不入 git）。
- `kaggle/outputs/libero-spatial-eval/run5/`：LIBERO 回放与协议走通的逐条记录。
- 统计脚本：`lerobot/outputs/eval_stats.py`（Wilson CI、方差）；各 kernel 内嵌同一份逻辑生成 `summary.md`。
