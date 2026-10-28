# Short ACT training on PushT (LeRobot) + multi-seed evaluation, Kaggle

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3), dataset revision v2.1
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- training: 10000 steps, batch 64, seeds [1, 2, 3]  |  policy args: `--policy.type=act --policy.push_to_hub=false --policy.chunk_size=16 --policy.n_action_steps=16 --policy.n_decoder_layers=4 --policy.optimizer_lr=3e-5`
- evaluation: seeds [1000, 2000, 3000], n_episodes=50, batch_size=50
- started 2026-10-28 01:12:50 UTC, elapsed 220.8 min

## Training runs

| run | variant | train seed | rc | final loss | min loss | train wall (min) | it/s |
|---|---|---|---|---|---|---|---|
| kp_train1 | kp | 1 | 0 | 0.0680 | 0.0660 | 30.1 | 5.5 |
| kp_train2 | kp | 2 | 0 | 0.0660 | 0.0660 | 29.7 | 5.6 |
| kp_train3 | kp | 3 | 0 | 0.0650 | 0.0650 | 29.8 | 5.6 |
| px_train1 | px | 1 | 0 | 0.0360 | 0.0360 | 37.6 | 4.4 |
| px_train2 | px | 2 | 0 | 0.0360 | 0.0350 | 37.0 | 4.5 |
| px_train3 | px | 3 | 0 | 0.0350 | 0.0350 | 37.2 | 4.5 |

## Evaluations (50 episodes each)

| run | policy | eval seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kp_train1_eval1000 | ACT keypoints (16-D keypoints + agent pos), train seed 1, 10000 steps | 1000 | rc=0 | 5/50 | 10.0% | 0.0918 | 0.0900 | [4.3, 21.4] | 108.14 | 0.687 | 31 |
| kp_train1_eval2000 | ACT keypoints (16-D keypoints + agent pos), train seed 1, 10000 steps | 2000 | rc=0 | 3/50 | 6.0% | 0.0576 | 0.0564 | [2.1, 16.2] | 116.29 | 0.696 | 30 |
| kp_train1_eval3000 | ACT keypoints (16-D keypoints + agent pos), train seed 1, 10000 steps | 3000 | rc=0 | 7/50 | 14.0% | 0.1229 | 0.1204 | [7.0, 26.2] | 108.83 | 0.717 | 30 |
| kp_train1_eval1000_repeat | ACT keypoints (16-D keypoints + agent pos), train seed 1, 10000 steps | 1000 | rc=0 | 5/50 | 10.0% | 0.0918 | 0.0900 | [4.3, 21.4] | 108.14 | 0.687 | 30 |
| kp_train2_eval1000 | ACT keypoints (16-D keypoints + agent pos), train seed 2, 10000 steps | 1000 | rc=0 | 15/50 | 30.0% | 0.2143 | 0.2100 | [19.1, 43.8] | 112.29 | 0.821 | 31 |
| kp_train2_eval2000 | ACT keypoints (16-D keypoints + agent pos), train seed 2, 10000 steps | 2000 | rc=0 | 13/50 | 26.0% | 0.1963 | 0.1924 | [15.9, 39.6] | 115.11 | 0.768 | 30 |
| kp_train2_eval3000 | ACT keypoints (16-D keypoints + agent pos), train seed 2, 10000 steps | 3000 | rc=0 | 10/50 | 20.0% | 0.1633 | 0.1600 | [11.2, 33.0] | 116.61 | 0.751 | 29 |
| kp_train3_eval1000 | ACT keypoints (16-D keypoints + agent pos), train seed 3, 10000 steps | 1000 | rc=0 | 15/50 | 30.0% | 0.2143 | 0.2100 | [19.1, 43.8] | 132.20 | 0.830 | 30 |
| kp_train3_eval2000 | ACT keypoints (16-D keypoints + agent pos), train seed 3, 10000 steps | 2000 | rc=0 | 15/50 | 30.0% | 0.2143 | 0.2100 | [19.1, 43.8] | 109.59 | 0.755 | 30 |
| kp_train3_eval3000 | ACT keypoints (16-D keypoints + agent pos), train seed 3, 10000 steps | 3000 | rc=0 | 19/50 | 38.0% | 0.2404 | 0.2356 | [25.9, 51.8] | 120.25 | 0.880 | 30 |
| px_train1_eval1000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 1, 10000 steps | 1000 | rc=0 | 3/50 | 6.0% | 0.0576 | 0.0564 | [2.1, 16.2] | 57.32 | 0.466 | 67 |
| px_train1_eval2000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 1, 10000 steps | 2000 | rc=0 | 2/50 | 4.0% | 0.0392 | 0.0384 | [1.1, 13.5] | 75.39 | 0.556 | 67 |
| px_train1_eval3000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 1, 10000 steps | 3000 | rc=0 | 2/50 | 4.0% | 0.0392 | 0.0384 | [1.1, 13.5] | 88.70 | 0.686 | 67 |
| px_train2_eval1000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 2, 10000 steps | 1000 | rc=0 | 3/50 | 6.0% | 0.0576 | 0.0564 | [2.1, 16.2] | 80.19 | 0.604 | 67 |
| px_train2_eval2000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 2, 10000 steps | 2000 | rc=0 | 6/50 | 12.0% | 0.1078 | 0.1056 | [5.6, 23.8] | 78.95 | 0.602 | 68 |
| px_train2_eval3000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 2, 10000 steps | 3000 | rc=0 | 7/50 | 14.0% | 0.1229 | 0.1204 | [7.0, 26.2] | 80.50 | 0.655 | 68 |
| px_train3_eval1000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 3, 10000 steps | 1000 | rc=0 | 3/50 | 6.0% | 0.0576 | 0.0564 | [2.1, 16.2] | 71.04 | 0.574 | 69 |
| px_train3_eval2000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 3, 10000 steps | 2000 | rc=0 | 5/50 | 10.0% | 0.0918 | 0.0900 | [4.3, 21.4] | 81.41 | 0.684 | 67 |
| px_train3_eval3000 | ACT pixels (96x96 image + agent pos, resnet18), train seed 3, 10000 steps | 3000 | rc=0 | 5/50 | 10.0% | 0.0918 | 0.0900 | [4.3, 21.4] | 83.05 | 0.585 | 68 |

## Across eval seeds (same checkpoint)

| checkpoint | eval seeds | mean pc_success | std across eval seeds (pts) | values |
|---|---|---|---|---|
| kp_train1 | [1000, 2000, 3000] | 10.0% | 4.0 | [10.0, 6.0, 14.0] |
| kp_train2 | [1000, 2000, 3000] | 25.3% | 5.0 | [30.0, 26.0, 20.0] |
| kp_train3 | [1000, 2000, 3000] | 32.7% | 4.6 | [30.0, 30.0, 38.0] |
| px_train1 | [1000, 2000, 3000] | 4.7% | 1.2 | [6.0, 4.0, 4.0] |
| px_train2 | [1000, 2000, 3000] | 10.7% | 4.2 | [6.0, 12.0, 14.0] |
| px_train3 | [1000, 2000, 3000] | 8.7% | 2.3 | [6.0, 10.0, 10.0] |

## Across training seeds (mean over eval seeds per checkpoint)

| variant | checkpoints | mean | std across training seeds (pts) | per-checkpoint means |
|---|---|---|---|---|
| kp | 3 | 22.7% | 11.6 | [10.0, 25.3, 32.7] |
| px | 3 | 8.0% | 3.1 | [4.7, 10.7, 8.7] |

## Determinism check (same checkpoint, same eval seed, run twice)

- kp_train1_eval1000_repeat vs kp_train1_eval1000: 5/50 vs 5/50, per-episode outcomes identical: True
