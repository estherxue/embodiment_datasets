# Other LeRobot policies on PushT: 10k-step training + official references, Kaggle (cpu)

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3), dataset revision v2.1
- device: cpu  |  GPU: None  |  torch 2.7.1+cpu
- training: 10000 steps, batch 64, train seed 1
- evaluation: seeds [1000, 2000, 3000], n_episodes=50, batch_size=50
- started 2026-10-30 01:05:13 UTC, elapsed 386.5 min

## Training runs

| run | policy | params | rc | loss @1k | loss @5k | final loss (mean last 1k steps) | train wall (min) | it/s |
|---|---|---|---|---|---|---|---|---|
| tdmpc_kp | TD-MPC, keypoints + agent pos, offline | 2.7M | 0 | 0.3271 | 0.0946 | 0.0726 | 137.6 | 1.2 |

## Evaluations (50 episodes each)

| run | policy | eval seed | ok | success | pc_success | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|
| ref_vqbet_pusht_eval1000 | official lerobot/vqbet_pusht (250k steps; card 57.0% / 500 eps) | 1000 | rc=0 | 30/50 | 60.0% | [46.2, 72.4] | 87.39 | 0.831 | 223 |
| ref_vqbet_pusht_eval2000 | official lerobot/vqbet_pusht (250k steps; card 57.0% / 500 eps) | 2000 | rc=0 | 28/50 | 56.0% | [42.3, 68.8] | 95.74 | 0.853 | 216 |
| ref_vqbet_pusht_eval3000 | official lerobot/vqbet_pusht (250k steps; card 57.0% / 500 eps) | 3000 | rc=0 | 26/50 | 52.0% | [38.5, 65.2] | 89.63 | 0.795 | 213 |
| ref_diffusion_pusht_keypoints_eval1000 | official lerobot/diffusion_pusht_keypoints (Hydra ckpt converted; card 71.0% / 500 eps) | 1000 | rc=0 | 38/50 | 76.0% | [62.6, 85.7] | 97.94 | 0.976 | 1111 |
| ref_diffusion_pusht_keypoints_eval2000 | official lerobot/diffusion_pusht_keypoints (Hydra ckpt converted; card 71.0% / 500 eps) | 2000 | rc=0 | 35/50 | 70.0% | [56.2, 80.9] | 109.37 | 0.968 | 1109 |
| ref_diffusion_pusht_keypoints_eval3000 | official lerobot/diffusion_pusht_keypoints (Hydra ckpt converted; card 71.0% / 500 eps) | 3000 | rc=0 | 37/50 | 74.0% | [60.4, 84.1] | 95.80 | 0.974 | 1106 |
| tdmpc_kp_eval1000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 | 1000 | rc=-9 FAILED | - | - | - | - | - | 3600 |
| tdmpc_kp_eval2000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 | 2000 | rc=-9 FAILED | - | - | - | - | - | 3600 |
| tdmpc_kp_eval3000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 | 3000 | rc=-9 FAILED | - | - | - | - | - | 3600 |

## Across eval seeds

| group | policy | seeds | mean pc_success | std across seeds (pts) | pooled k/N | pooled pc | Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| ref_vqbet_pusht | official lerobot/vqbet_pusht (250k steps; card 57.0% / 500 eps) | [1000, 2000, 3000] | 56.0% | 4.0 | 84/150 | 56.0% | [48.0, 63.7] |
| ref_diffusion_pusht_keypoints | official lerobot/diffusion_pusht_keypoints (Hydra ckpt converted; card 71.0% / 500 eps) | [1000, 2000, 3000] | 73.3% | 3.1 | 110/150 | 73.3% | [65.7, 79.8] |
