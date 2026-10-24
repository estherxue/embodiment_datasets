# LeRobot ACT evaluation on Kaggle (ALOHA, checkpoint-era simulator: gym-aloha 0.1.1 / mujoco 2.3.7)

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3)
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- MUJOCO_GL: egl  |  n_episodes=50 batch_size=50
- started 2026-10-23 20:12:52 UTC, elapsed 232.5 min

| run | policy | env / obs | seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aloha_insertion_act_seed1000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 1000 | rc=0 | 8/50 | 16.0% | 0.1371 | 0.1344 | [8.3, 28.5] | 243.22 | 2.360 | 4585 |
| aloha_insertion_act_seed2000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 2000 | rc=0 | 10/50 | 20.0% | 0.1633 | 0.1600 | [11.2, 33.0] | 309.10 | 2.780 | 4621 |
| aloha_insertion_act_seed3000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 | 3000 | rc=0 | 7/50 | 14.0% | 0.1229 | 0.1204 | [7.0, 26.2] | 267.58 | 2.520 | 4587 |

## Across seeds (same policy, 50 episodes per seed)

| group | seeds | mean pc_success | std across seeds (pts) | values |
|---|---|---|---|---|
| aloha_insertion_act | [1000, 2000, 3000] | 16.7% | 3.1 | [16.0, 20.0, 14.0] |
