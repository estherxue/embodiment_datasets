# embodiment_datasets/ — LeRobot 与 LIBERO 评测（2026-10-23 ~ 10-24）

本地是 Intel Mac（无 GPU，官方 PyTorch 在 macOS x86 最高 2.2.2），两套环境都能装、能跑但慢，最终评测改在 Kaggle GPU kernel（2× Tesla T4）上完成。本地 venv 和部分结果保留作对照。

| 目录 | 内容 |
|---|---|
| `lerobot/` | LeRobot clone（commit b883328e，v0.3.3）+ 本地 `.venv` + **`EVAL_NOTES.md`**（PushT ACT、ALOHA 插接 ACT 的全部结果、命令、踩坑） |
| `LIBERO/` | LIBERO clone（commit 8f1084e）+ 本地 `.venv` + `datasets/libero_spatial`（5.8 GB）+ `local_eval/`（回放/协议脚本）+ **`EVAL_NOTES.md`**（套件结构、评测协议、demo 回放、协议走通） |
| `kaggle/lerobot-eval/`、`kaggle/lerobot-eval-aloha-mj237/`、`kaggle/libero-eval/` | 三个 Kaggle script kernel 的源码与 `kernel-metadata.json` |
| `kaggle/outputs/<kernel>/` | 从 Kaggle 拉回的 `summary.md` / `summary.json` / `eval_info.json` / 日志 / 视频 |

## 结果一眼看

**LeRobot（lerobot-eval，n=50 episodes/seed，seed 1000/2000/3000）**

| 权重 | 任务 | 成功率（3 seeds） | 跨 seed 均值 ± 标准差 |
|---|---|---|---|
| alexandersoare/act_pusht_keypoints（ACT，关键点观测；官方没有 act_pusht） | PushT | 64 / 60 / 64% | 62.7% ± 2.3 |
| arclabmit/pusht_act_model（ACT，像素；社区权重，基本不工作） | PushT | 0 / 0 / 6% | 2.0% ± 3.5 |
| lerobot/act_aloha_sim_insertion_human，MuJoCo 3.8.1 | AlohaInsertion-v0 | 12 / 8 / 12% | 10.7% ± 2.3 |
| 同上，MuJoCo 2.3.7 + gym-aloha 0.1.1（权重发布时版本） | AlohaInsertion-v0 | 16 / 20 / 14% | 16.7% ± 3.1 |
| lerobot/diffusion_pusht（非 ACT 参照） | PushT | 66% | 与官方 65.4% 一致 |

ALOHA 远低于作者自报的 83%，原因未定位，详见 `lerobot/EVAL_NOTES.md`。

**LIBERO（libero_spatial）**

- 没有可被 `libero/lifelong/evaluate.py` 加载的官方/社区 checkpoint → 做 demo 回放 + 协议走通。
- demo 回放：10 任务 × 50 条 demo，451/500 = 90.2%（Wilson 95% CI [87.3, 92.5]），单任务 82%~96%，两次渲染后端下 500 条记录完全一致。
- 协议走通：随机初始化 BCTransformerPolicy 按原协议（固定 init states、20 并行 env、max_steps 600）10 任务各 20 episodes 全部跑通，0 成功（预期）。
- 套件结构：spatial/object/goal/10 各 10 个任务，libero_90 有 90 个，LIBERO-100 = 10 + 90；每任务 50 个固定初始状态、50 条 demo。
