# LeRobot ACT evaluation on Kaggle

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3)
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- MUJOCO_GL: egl  |  n_episodes=50 batch_size=50
- started 2026-10-23 13:58:50 UTC, elapsed 367.2 min

| run | policy | env / obs | seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pusht_act_keypoints_seed1000 | alexandersoare/act_pusht_keypoints (converted) | --env.type=pusht --env.obs_type=environment_state_agent_pos | 1000 | rc=0 | 32/50 | 64.0% | 0.2351 | 0.2304 | [50.1, 75.9] | 95.50 | 0.949 | 32 |
| aloha_insertion_act_seed1000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 1000 | rc=0 | 6/50 | 12.0% | 0.1078 | 0.1056 | [5.6, 23.8] | 180.74 | 1.940 | 5184 |
| pusht_act_keypoints_seed2000 | alexandersoare/act_pusht_keypoints (converted) | --env.type=pusht --env.obs_type=environment_state_agent_pos | 2000 | rc=0 | 30/50 | 60.0% | 0.2449 | 0.2400 | [46.2, 72.4] | 102.51 | 0.932 | 29 |
| pusht_act_keypoints_seed3000 | alexandersoare/act_pusht_keypoints (converted) | --env.type=pusht --env.obs_type=environment_state_agent_pos | 3000 | rc=0 | 32/50 | 64.0% | 0.2351 | 0.2304 | [50.1, 75.9] | 98.31 | 0.918 | 30 |
| pusht_act_pixels_seed1000 | arclabmit/pusht_act_model | --env.type=pusht | 1000 | rc=0 | 0/50 | 0.0% | 0.0000 | 0.0000 | [0.0, 7.1] | 45.25 | 0.424 | 67 |
| aloha_insertion_act_seed2000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 2000 | rc=0 | 4/50 | 8.0% | 0.0751 | 0.0736 | [3.2, 18.8] | 213.04 | 2.120 | 5287 |
| aloha_insertion_act_seed3000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 3000 | rc=0 | 6/50 | 12.0% | 0.1078 | 0.1056 | [5.6, 23.8] | 208.04 | 2.080 | 5242 |
| pusht_act_pixels_seed2000 | arclabmit/pusht_act_model | --env.type=pusht | 2000 | rc=0 | 0/50 | 0.0% | 0.0000 | 0.0000 | [0.0, 7.1] | 44.44 | 0.371 | 67 |
| pusht_act_pixels_seed3000 | arclabmit/pusht_act_model | --env.type=pusht | 3000 | rc=0 | 3/50 | 6.0% | 0.0576 | 0.0564 | [2.1, 16.2] | 42.24 | 0.438 | 68 |
| pusht_diffusion_seed1000 | lerobot/diffusion_pusht (non-ACT reference) | --env.type=pusht | 1000 | rc=0 | 33/50 | 66.0% | 0.2290 | 0.2244 | [52.2, 77.6] | 114.68 | 0.972 | 273 |
| pusht_random_seed1000 | uniform random actions | --env.type=pusht | 1000 | rc=0 | 0/50 | 0.0% | 0.0000 | 0.0000 | [0.0, 7.1] | 1.22 | 0.086 | 47 |
| pusht_act_pepijn223_seed1000 | pepijn223/act-pusht (converted) | --env.type=pusht | 1000 | rc=0 | 0/50 | 0.0% | 0.0000 | 0.0000 | [0.0, 7.1] | 37.62 | 0.403 | 67 |

## Across seeds (same policy, 50 episodes per seed)

| group | seeds | mean pc_success | std across seeds (pts) | values |
|---|---|---|---|---|
| pusht_act_keypoints | [1000, 2000, 3000] | 62.7% | 2.3 | [64.0, 60.0, 64.0] |
| aloha_insertion_act | [1000, 2000, 3000] | 10.7% | 2.3 | [12.0, 8.0, 12.0] |
| pusht_act_pixels | [1000, 2000, 3000] | 2.0% | 3.5 | [0.0, 0.0, 6.0] |
| pusht_diffusion | [1000] | 66.0% | nan | [66.0] |
| pusht_random | [1000] | 0.0% | nan | [0.0] |
| pusht_act_pepijn223 | [1000] | 0.0% | nan | [0.0] |
