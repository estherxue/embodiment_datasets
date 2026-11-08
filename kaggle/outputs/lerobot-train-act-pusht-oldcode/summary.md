# ACT on PushT keypoints, retrained with the author's 2024-07 lerobot code and config.yaml, Kaggle

- training code: lerobot `471eab3d7e04c8dd298f917ab8cea5f28aaac8e2` (2024-07-11), torch 2.3.1+cu121, dataset lerobot/pusht_keypoints revision v1.5; config: alexandersoare/act_pusht_keypoints config.yaml verbatim + `wandb.enable=false training.eval_freq=-1 training.save_freq=10000 training.log_freq=100 training.num_workers=4 training.online_steps_between_rollouts=1 training.online_env_seed=1`
- evaluation code: lerobot `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3), checkpoints converted to the 0.3.3 config format
- device: GPU Tesla T4  |  eval torch 2.7.1+cu126
- training: 100000 steps, batch 64, save every 10000; rc=0, wall 256.4 min, 6.5 it/s, killed_by_budget=False
- evaluation: intermediate checkpoints seeds [1000, 2000, 3000] x 50 episodes; last checkpoint seeds [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] x 50
- started 2026-11-08 04:25:23 UTC, elapsed 257.6 min

## Training loss (L1 + 10*KL), mean over the 10 log points before each checkpoint

| step | loss (mean of last 1k steps) | grad norm | lr |
|---|---|---|---|
| 10000 | 0.0660 | 3.026 | 3.0e-05 |
| 20000 | 0.0566 | 2.566 | 3.0e-05 |
| 30000 | 0.0511 | 2.459 | 3.0e-05 |
| 40000 | 0.0455 | 1.790 | 3.0e-05 |
| 50000 | 0.0414 | 2.064 | 3.0e-05 |
| 60000 | 0.0379 | 1.636 | 3.0e-05 |
| 70000 | 0.0359 | 1.973 | 3.0e-05 |
| 80000 | 0.0320 | 2.029 | 3.0e-05 |
| 90000 | 0.0317 | 1.584 | 3.0e-05 |
| 100000 | 0.0294 | 1.616 | 3.0e-05 |

## Success rate per checkpoint (lerobot 0.3.3 eval, 50 episodes per seed)

| step | seeds | per-seed success | pooled k/N | pooled % | Wilson 95% CI | std across seeds (pts) |
|---|---|---|---|---|---|---|
| 10000 | 3 | 8 / 2 / 8 | 9/150 | 6.0% | [3.2, 11.0] | 3.5 |
| 20000 | 3 | 14 / 10 / 8 | 16/150 | 10.7% | [6.7, 16.6] | 3.1 |
| 30000 | 3 | 14 / 26 / 38 | 39/150 | 26.0% | [19.6, 33.6] | 12.0 |
| 40000 | 3 | 24 / 22 / 10 | 28/150 | 18.7% | [13.2, 25.7] | 7.6 |
| 50000 | 3 | 20 / 20 / 14 | 27/150 | 18.0% | [12.7, 24.9] | 3.5 |
| 60000 | 3 | 36 / 24 / 22 | 41/150 | 27.3% | [20.8, 35.0] | 7.6 |
| 70000 | 3 | 50 / 30 / 32 | 56/150 | 37.3% | [30.0, 45.3] | 11.0 |
| 80000 | 3 | 44 / 36 / 44 | 62/150 | 41.3% | [33.8, 49.3] | 4.6 |
| 90000 | 3 | 36 / 40 / 48 | 62/150 | 41.3% | [33.8, 49.3] | 6.1 |
| 100000 | 10 | 42 / 40 / 44 / 46 / 42 / 50 / 58 / 36 / 42 / 54 | 227/500 | 45.4% | [41.1, 49.8] | 6.7 |

## Last checkpoint (100000 steps, 227/500 = 45.4%) vs references

- vs alexandersoare/act_pusht_keypoints eval_info.json (100k steps, 64.0%): 64.0%, z=-5.91, p=3.5e-09
- vs our lerobot 0.3.3 runs, 100k steps, train seeds 1-3 pooled (47.6%): 47.6%, z=-0.85, p=0.39
