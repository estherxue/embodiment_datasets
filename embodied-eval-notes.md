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
| **PushT，自己短训的 ACT（10k 步，3 个训练 seed × 3 个评测 seed）** | 见 §3 | <!-- TRAIN_HEADLINE --> | | 100k 步的同配方权重 64% |
| LIBERO libero_spatial，demo 开环回放（无可加载的官方权重） | 10 任务 × 50 = 500 | 90.2% | [87.3, 92.5] | 回放上限，不是策略成绩 |

方差从哪来（展开见 §4）：n=50 时 **Bernoulli 采样误差 5–7 个百分点**是主项；换评测 seed（= 换一批初始状态）带来的 seed 间标准差只有 2–3 个百分点，**小于**采样误差本身，说明初始状态没有额外的方差分量；仿真器版本能把 ALOHA 插接移动 6 个百分点，是比 seed 大得多的系统性因素；训练 seed 的影响见 §3。

## 1. 统计口径（所有表共用）

- **N**：每次评测 50 个 episode（`--eval.n_episodes=50 --eval.batch_size=50`），episode i 的环境 seed = `--seed` + i（lerobot-eval 的约定），三次评测用 `--seed` 1000 / 2000 / 3000，即 150 个互不重叠的初始状态。
- **成功**：环境自己的 `info["is_success"]`。PushT：T 块与目标区域重叠 ≥ 95%；ALOHA 插接：reward 达到 4（插销插入插座）；transfer cube：reward 4（左臂接住方块）。
- **成功率** p̂ = k / n；**置信区间**用 Wilson 95% 区间（n=50、p̂≈0.6 时宽约 ±13 个百分点，n=150 时约 ±8，n=500 时约 ±4）。
- **方差**：每个 episode 成功与否是 0/1 变量，单次评测内的样本方差 = Σ(xᵢ − p̂)² / (n − 1) ≈ p̂(1 − p̂)；均值的标准误 SE = sqrt(p̂(1 − p̂)/n)。跨 seed 的标准差是 3 个 p̂ 的样本标准差，用来和 SE 比：如果不比 SE 大，就没有证据说明 seed 之间有"额外"方差。
- **确定性**：ACT 推理不采样（VAE 隐变量取 0），PushT（pymunk）和 MuJoCo 给定初始状态后是确定性的，所以同一权重 + 同一 seed 重跑应逐 episode 完全一致（§3 有实测）。Diffusion Policy 推理有噪声采样，由 torch seed 固定。

## 2. 预训练权重评测（详表）

数据来源：`kaggle/outputs/lerobot-act-eval-pusht-aloha/`（PushT 全部 + ALOHA MuJoCo 3.8.1）、`kaggle/outputs/lerobot-act-eval-aloha-mj237/`（ALOHA MuJoCo 2.3.7）、`kaggle/outputs/lerobot-act-eval-aloha-transfer-mj237/`（transfer cube 对照）。lerobot 0.3.3（b883328e），torch 2.7.1+cu126。

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

按组汇总（跨 seed 标准差 vs. 单次 n=50 本身的 Bernoulli SE）：

| 组 | seeds | 跨 seed 均值 | 跨 seed 标准差 (pts) | 单次 n=50 的 Bernoulli SE (pts) | 合并 k/N | 合并成功率 | 合并 Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| PushT · ACT keypoints（alexandersoare） | 3 | 62.7% | 2.3 | 6.8 | 94/150 | 62.7% | [54.7, 70.0] |
| PushT · ACT pixels（arclabmit） | 3 | 2.0% | 3.5 | 2.0 | 3/150 | 2.0% | [0.7, 5.7] |
| ALOHA 插接 · ACT，MuJoCo 3.8.1 | 3 | 10.7% | 2.3 | 4.4 | 16/150 | 10.7% | [6.7, 16.6] |
| ALOHA 插接 · ACT，MuJoCo 2.3.7 | 3 | 16.7% | 3.1 | 5.3 | 25/150 | 16.7% | [11.6, 23.4] |
| ALOHA transfer cube · ACT，MuJoCo 2.3.7（对照） | 2 | 76.0% | 2.8 | 6.0 | 76/100 | 76.0% | [66.8, 83.3] |

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

<!-- TRAIN_RESULT -->
（kernel 2026-10-28 启动，跑完后填入：loss 曲线摘要、每个 checkpoint × 评测 seed 的成功率、方差分解、同 seed 重跑的确定性检查。）

## 4. 方差从哪来

**(a) Bernoulli 采样（episode 数 n）。** 单次评测的成功率是 n 个 0/1 的均值，SE = sqrt(p(1−p)/n)。这是 n=50 时最大的不确定性来源：

| p | n=50 | n=150 | n=500 |
|---|---|---|---|
| 0.63（PushT 关键点 ACT） | SE 6.8 pts，Wilson 宽 ±13 | 3.9 | 2.2 |
| 0.17（ALOHA 插接） | SE 5.3 pts，Wilson 宽 ±11 | 3.1 | 1.7 |
| 0.76（transfer cube） | SE 6.0 pts | 3.5 | 1.9 |

所以"64% vs 60%"这种差异在 n=50 下没有信息量；要分辨 5 个百分点的差异，n 需要几百。

**(b) 评测 seed = 初始状态。** PushT 的 `reset(seed)` 用 seed 采样 T 块位姿和 agent 位置（本地验证：同 seed 两次 reset 初始状态逐元素相同，seed 1000 与 1001 的状态最大差约 290 像素；50 个连续 seed 的 agent 初始位置标准差约 100–120 像素，T 块关键点约 100 像素）。ALOHA 的 `reset(seed)` 同样采样插销/插座（或方块）的位姿。策略和仿真器本身确定（§1），所以**换评测 seed 的全部效果就是换一批初始状态**。实测跨 seed 标准差 2.3–3.1 pts，都小于对应的 SE 4.4–6.8 pts（§2 汇总表），也就是三批初始状态之间的差异完全能用采样噪声解释，没有"某批初始状态特别难"的额外分量。

**(c) 训练 seed。** 同一配方换训练 seed（初始化、数据顺序、dropout）得到不同权重，这是预训练权重评测看不到的分量。§3 用 3 个训练 seed × 3 个评测 seed 的 3×3 设计来拆：<!-- TRAIN_VARIANCE -->（待填）

**(d) 仿真器与硬件。** 同一权重、同一 seed：MuJoCo 3.8.1 → 2.3.7 把 ALOHA 插接从 10.7% 提到 16.7%（6 pts，比任何 seed 效应都大）；PushT 关键点 ACT 在本地 CPU（torch 2.2.2）与 Kaggle GPU 上同 seed 的结果 66/58/66 vs 64/60/64——个别 episode 因浮点差异翻转，总体在 CI 内。这类差异是系统性的，不会随 n 变小，只能靠钉版本来消除（三个 kernel 和本地 venv 现在都钉在 gym-aloha 0.1.1 / mujoco 2.3.7）。

**(e) 评测协议。** 时限：ALOHA 插接 demo 每条 500 帧，而模型卡评测用 400 步（Hub discussion #1 报告 500 步时 23.2% vs 20.6%）；我们的插接数字都是 400 步口径，500 步补测 kernel `cindyxue1122/lerobot-act-eval-aloha-ep500-mj237` 的结果：<!-- EP500_RESULT -->（待填）。LIBERO 则相反：每个任务固定 50 个初始状态（`.pruned_init`），不随 seed 采样，所以 LIBERO 的跨 run 方差只来自策略本身（回放时为 0）。

## 5. 产出物索引

- `kaggle/outputs/lerobot-act-eval-pusht-aloha/eval/<run>/eval_info.json`：PushT 全部 + ALOHA MuJoCo 3.8.1，12 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-mj237/`：ALOHA 插接 MuJoCo 2.3.7，3 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-transfer-mj237/`：transfer cube 对照，2 次评测。
- `kaggle/outputs/lerobot-act-eval-aloha-ep500-mj237/`：插接 500 步口径，2 次评测。
- `kaggle/outputs/lerobot-train-act-pusht/`：短训 6 个 checkpoint 的训练日志（`train/<run>/train_log.json`，每 100 步的 loss / grad norm）、18+1 次评测、seed-1 的两个 checkpoint（`ckpts/`，`model.safetensors` 不入 git）。
- `kaggle/outputs/libero-spatial-eval/run5/`：LIBERO 回放与协议走通的逐条记录。
- 统计脚本：`lerobot/outputs/eval_stats.py`（Wilson CI、方差）；各 kernel 内嵌同一份逻辑生成 `summary.md`。
