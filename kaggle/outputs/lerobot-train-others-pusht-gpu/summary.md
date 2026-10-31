# Other LeRobot policies on PushT: 10k-step training + official references, Kaggle (cuda)

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3), dataset revision v2.1
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- training: 10000 steps, batch 64, train seed 1
- evaluation: seeds [1000, 2000, 3000], n_episodes=50, batch_size=50
- started 2026-10-31 07:33:54 UTC, elapsed 252.4 min

## Training runs

| run | policy | params | rc | loss @1k | loss @5k | final loss (mean last 1k steps) | train wall (min) | it/s |
|---|---|---|---|---|---|---|---|---|
| vqbet_px | VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps) | 37.5M | 0 | 0.0887 | 752.0886 | 518.0303 | 75.1 | 2.2 |
| diffusion_kp | Diffusion Policy, keypoints + agent pos, DDIM-10 | 248.8M | 0 | 0.1293 | 0.0344 | 0.0232 | 65.5 | 2.5 |
| diffusion_px | Diffusion Policy, pixels + agent pos, DDPM-100 | 262.7M | 0 | 0.1281 | 0.0394 | 0.0268 | 72.1 | 2.3 |

## Evaluations (50 episodes each)

| run | policy | eval seed | ok | success | pc_success | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|
| tdmpc_kp_cpu10k_eval1000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 (trained in the cpu kernel) | 1000 | rc=0 | 0/50 | 0.0% | [0.0, 7.1] | 7.64 | 0.073 | 312 |
| tdmpc_kp_cpu10k_eval2000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 (trained in the cpu kernel) | 2000 | rc=0 | 0/50 | 0.0% | [0.0, 7.1] | 12.27 | 0.107 | 338 |
| tdmpc_kp_cpu10k_eval3000 | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 (trained in the cpu kernel) | 3000 | rc=0 | 0/50 | 0.0% | [0.0, 7.1] | 6.63 | 0.099 | 338 |
| vqbet_px_eval1000 | VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps), 10000 steps, train seed 1 | 1000 | rc=0 | 3/50 | 6.0% | [2.1, 16.2] | 105.97 | 0.628 | 75 |
| vqbet_px_eval2000 | VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps), 10000 steps, train seed 1 | 2000 | rc=0 | 4/50 | 8.0% | [3.2, 18.8] | 118.63 | 0.691 | 75 |
| vqbet_px_eval3000 | VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps), 10000 steps, train seed 1 | 3000 | rc=0 | 8/50 | 16.0% | [8.3, 28.5] | 95.88 | 0.629 | 74 |
| diffusion_kp_eval1000 | Diffusion Policy, keypoints + agent pos, DDIM-10, 10000 steps, train seed 1 | 1000 | rc=0 | 23/50 | 46.0% | [33.0, 59.6] | 140.88 | 0.983 | 51 |
| diffusion_kp_eval2000 | Diffusion Policy, keypoints + agent pos, DDIM-10, 10000 steps, train seed 1 | 2000 | rc=0 | 14/50 | 28.0% | [17.5, 41.7] | 135.44 | 0.901 | 51 |
| diffusion_kp_eval3000 | Diffusion Policy, keypoints + agent pos, DDIM-10, 10000 steps, train seed 1 | 3000 | rc=0 | 21/50 | 42.0% | [29.4, 55.8] | 122.69 | 0.951 | 51 |
| diffusion_px_eval1000 | Diffusion Policy, pixels + agent pos, DDPM-100, 10000 steps, train seed 1 | 1000 | rc=0 | 5/50 | 10.0% | [4.3, 21.4] | 104.13 | 0.718 | 271 |
| diffusion_px_eval2000 | Diffusion Policy, pixels + agent pos, DDPM-100, 10000 steps, train seed 1 | 2000 | rc=0 | 3/50 | 6.0% | [2.1, 16.2] | 107.44 | 0.661 | 271 |
| diffusion_px_eval3000 | Diffusion Policy, pixels + agent pos, DDPM-100, 10000 steps, train seed 1 | 3000 | rc=0 | 3/50 | 6.0% | [2.1, 16.2] | 106.18 | 0.707 | 269 |

## Across eval seeds

| group | policy | seeds | mean pc_success | std across seeds (pts) | pooled k/N | pooled pc | Wilson 95% CI |
|---|---|---|---|---|---|---|---|
| tdmpc_kp_cpu10k | TD-MPC, keypoints + agent pos, offline, 10000 steps, train seed 1 (trained in the cpu kernel) | [1000, 2000, 3000] | 0.0% | 0.0 | 0/150 | 0.0% | [0.0, 2.5] |
| vqbet_px | VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps), 10000 steps, train seed 1 | [1000, 2000, 3000] | 10.0% | 5.3 | 15/150 | 10.0% | [6.2, 15.8] |
| diffusion_kp | Diffusion Policy, keypoints + agent pos, DDIM-10, 10000 steps, train seed 1 | [1000, 2000, 3000] | 38.7% | 9.5 | 58/150 | 38.7% | [31.2, 46.7] |
| diffusion_px | Diffusion Policy, pixels + agent pos, DDPM-100, 10000 steps, train seed 1 | [1000, 2000, 3000] | 7.3% | 2.3 | 11/150 | 7.3% | [4.1, 12.7] |
