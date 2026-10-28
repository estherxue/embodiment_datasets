# LeRobot ACT evaluation on Kaggle (ALOHA insertion, episode_length=500, checkpoint-era simulator: gym-aloha 0.1.1 / mujoco 2.3.7)

- lerobot commit: `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3)
- device: cuda  |  GPU: Tesla T4  |  torch 2.7.1+cu126
- MUJOCO_GL: egl  |  n_episodes=50 batch_size=50
- started 2026-10-28 00:54:13 UTC, elapsed 216.1 min

| run | policy | env / obs | seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aloha_insertion_act_ep500_seed1000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 --env.episode_length=500 | 1000 | rc=0 | 10/50 | 20.0% | 0.1633 | 0.1600 | [11.2, 33.0] | 315.22 | 2.420 | 6190 |
| aloha_insertion_act_ep500_seed2000 | lerobot/act_aloha_sim_insertion_human | --env.type=aloha --env.task=AlohaInsertion-v0 --env.episode_length=500 | 2000 | rc=0 | 11/50 | 22.0% | 0.1751 | 0.1716 | [12.8, 35.2] | 433.12 | 2.820 | 6604 |

## Across seeds (same policy, 50 episodes per seed)

| group | seeds | mean pc_success | std across seeds (pts) | values |
|---|---|---|---|---|
| aloha_insertion_act_ep500 | [1000, 2000] | 21.0% | 1.4 | [20.0, 22.0] |
