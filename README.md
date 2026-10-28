# embodiment_datasets/ — LeRobot 与 LIBERO 评测（2026-10-23 ~ 10-24）

本地是 Intel Mac（无 GPU，官方 PyTorch 在 macOS x86 最高 2.2.2），两套环境都能装、能跑但慢，最终评测改在 Kaggle GPU kernel（2× Tesla T4）上完成。本地 venv 和部分结果保留作对照。

| 目录 | 内容 |
|---|---|
| `lerobot/` | LeRobot clone（commit b883328e，v0.3.3）+ 本地 `.venv` + **`EVAL_NOTES.md`**（PushT ACT、ALOHA 插接 ACT 的全部结果、命令、踩坑） |
| `LIBERO/` | LIBERO clone（commit 8f1084e）+ 本地 `.venv` + `datasets/libero_spatial`（5.8 GB）+ `local_eval/`（回放/协议脚本）+ **`EVAL_NOTES.md`**（套件结构、评测协议、demo 回放、协议走通） |
| `kaggle/lerobot-eval/`、`kaggle/lerobot-eval-aloha-mj237/`、`kaggle/lerobot-eval-aloha-transfer-mj237/`、`kaggle/lerobot-eval-aloha-ep500-mj237/`、`kaggle/lerobot-train-act-pusht/`、`kaggle/libero-eval/` | 六个 Kaggle script kernel 的源码与 `kernel-metadata.json`（训练 kernel 另带 `train_analysis.py`） |
| **`embodied-eval-notes.md`** | 一页汇总：跑了什么、N、成功率与 Wilson CI、方差来源（Bernoulli / 评测 seed / 训练 seed / 仿真器 / 协议） |
| `kaggle/outputs/<kernel>/` | 从 Kaggle 拉回的 `summary.md` / `summary.json` / `eval_info.json` / 日志 / 视频 |

## 结果一眼看

**LeRobot（lerobot-eval，n=50 episodes/seed，seed 1000/2000/3000）**

| 权重 | 任务 | 成功率（3 seeds） | 跨 seed 均值 ± 标准差 |
|---|---|---|---|
| alexandersoare/act_pusht_keypoints（ACT，关键点观测；官方没有 act_pusht） | PushT | 64 / 60 / 64% | 62.7% ± 2.3 |
| arclabmit/pusht_act_model（ACT，像素；社区权重，基本不工作） | PushT | 0 / 0 / 6% | 2.0% ± 3.5 |
| lerobot/act_aloha_sim_insertion_human，MuJoCo 3.8.1 | AlohaInsertion-v0 | 12 / 8 / 12% | 10.7% ± 2.3 |
| 同上，MuJoCo 2.3.7 + gym-aloha 0.1.1（权重发布时版本） | AlohaInsertion-v0 | 16 / 20 / 14% | 16.7% ± 3.1（模型卡 20.6%） |
| lerobot/act_aloha_sim_transfer_cube_human（正向对照），MuJoCo 2.3.7 | AlohaTransferCube-v0 | 74 / 78%（2 seeds） | 76.0% ± 2.8（模型卡 83.0%） |
| lerobot/act_aloha_sim_insertion_human，MuJoCo 2.3.7，episode_length=500 | AlohaInsertion-v0 | 20 / 22%（2 seeds） | 21.0% ± 1.4（模型卡 20.6%，400 步） |
| lerobot/diffusion_pusht（非 ACT 参照） | PushT | 66% | 与官方 65.4% 一致 |

**自己短训（lerobot-train，ACT on PushT，10k 步，配方同 alexandersoare 的 100k 步权重）**

| 变体 | 训练 seed 1 / 2 / 3（各 3 个评测 seed × 50 的均值） | 合并 9 × 50 | 训练 seed 间 / 评测 seed 间标准差 |
|---|---|---|---|
| 关键点 | 10.0 / 25.3 / 32.7% | 102/450 = 22.7%（CI [19.0, 26.8]） | 11.6 / 4.6 pts（F(2,6)=19.2，p=0.003） |
| 像素 | 4.7 / 10.7 / 8.7% | 36/450 = 8.0%（CI [5.8, 10.9]） | 3.1 / 2.8 pts（不显著） |

训练回路通：loss 单调降到 0.067 / 0.036 且 10k 步未平；同权重同 seed 重跑逐 episode 一致。关键点变体三个训练 seed 的 loss 几乎一样、成功率却是 10% vs 33%——从头训的 checkpoint 单报一个数没有锚，训练 seed 是比评测 seed 和 n 更大的方差来源。详见 `embodied-eval-notes.md` §3。

ALOHA 插接模型卡自报 20.6%（500 episodes；原论文 20%），MuJoCo 2.3.7 下的 16.7% 与之统计上一致（p=0.29）。之前笔记里写的“自报 83%”来自该仓库的 eval_info.json，排查发现那是 transfer cube 的文件被错传进了插接仓库，用同一流程评 transfer cube 得 76%（模型卡 83%，p=0.10），说明流程能复现作者数字。排查过程见 `lerobot/EVAL_NOTES.md` §6；插接按 500 步（demo 长度）口径补测得 21.0%（21/100），与模型卡重合，比 400 步口径高约 4 个点但不显著。

**LIBERO（libero_spatial）**

- 没有可被 `libero/lifelong/evaluate.py` 加载的官方/社区 checkpoint → 做 demo 回放 + 协议走通。
- demo 回放：10 任务 × 50 条 demo，451/500 = 90.2%（Wilson 95% CI [87.3, 92.5]），单任务 82%~96%，两次渲染后端下 500 条记录完全一致。
- 协议走通：随机初始化 BCTransformerPolicy 按原协议（固定 init states、20 并行 env、max_steps 600）10 任务各 20 episodes 全部跑通，0 成功（预期）。
- 套件结构：spatial/object/goal/10 各 10 个任务，libero_90 有 90 个，LIBERO-100 = 10 + 90；每任务 50 个固定初始状态、50 条 demo。
