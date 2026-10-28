# LeRobot ACT evaluation on Kaggle (ALOHA transfer cube control run, checkpoint-era simulator: gym-aloha 0.1.1 / mujoco 2.3.7)

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3)
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- MUJOCO_GL: egl  |  n_episodes=50 batch_size=50
- started 2026-10-24 13:49:48 UTC, elapsed 141.1 min

| run | policy | env / obs | seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aloha_transfer_act_seed1000 | lerobot/act_aloha_sim_transfer_cube_human | --env.type=aloha --env.task=AlohaTransferCube-v0 | 1000 | rc=0 | 37/50 | 74.0% | 0.1963 | 0.1924 | [60.4, 84.1] | 209.56 | 3.300 | 4276 |
| aloha_transfer_act_seed2000 | lerobot/act_aloha_sim_transfer_cube_human | --env.type=aloha --env.task=AlohaTransferCube-v0 | 2000 | rc=0 | 39/50 | 78.0% | 0.1751 | 0.1716 | [64.8, 87.2] | 224.34 | 3.480 | 4054 |

## Across seeds (same policy, 50 episodes per seed)

| group | seeds | mean pc_success | std across seeds (pts) | values |
|---|---|---|---|---|
| aloha_transfer_act | [1000, 2000] | 76.0% | 2.8 | [74.0, 78.0] |
