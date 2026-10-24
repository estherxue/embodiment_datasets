# LIBERO libero_spatial on Kaggle - summary

- started (UTC): 2026-10-23T14:20:58.042121  elapsed: 242.9 min
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB  nproc: 4  render backend: egl
- LIBERO commit: 8f1084e; dataset: 5.81 GiB, 10 hdf5 files

## Demo replay (open-loop action replay of the recorded demos, success = env.step done)

| task | n | succ | rate | SE | mean steps→succ | mean len | mean max drift | s/demo | language |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 48 | 96% | 0.0277 | 81.58333333333333 | 101.36 | 4.864 | 38.8 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | 50 | 45 | 90% | 0.0424 | 110.42222222222222 | 134.14 | 5.486 | 56.4 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | 50 | 48 | 96% | 0.0277 | 98.77083333333333 | 117.64 | 3.463 | 54.3 | pick up the black bowl from table center and place it on the plate |
| 3 | 50 | 45 | 90% | 0.0424 | 84.28888888888889 | 101.04 | 4.544 | 47.4 | pick up the black bowl on the cookie box and place it on the plate |
| 4 | 50 | 41 | 82% | 0.0543 | 130.09756097560975 | 149.58 | 4.345 | 71.1 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 5 | 50 | 41 | 82% | 0.0543 | 101.1219512195122 | 115.92 | 6.353 | 48.3 | pick up the black bowl on the ramekin and place it on the plate |
| 6 | 50 | 48 | 96% | 0.0277 | 109.14583333333333 | 126.24 | 5.793 | 54.9 | pick up the black bowl next to the cookie box and place it on the plate |
| 7 | 50 | 42 | 84% | 0.0518 | 122.57142857142857 | 142.22 | 5.584 | 70.7 | pick up the black bowl on the stove and place it on the plate |
| 8 | 50 | 46 | 92% | 0.0384 | 100.91304347826087 | 119.26 | 4.774 | 55.5 | pick up the black bowl next to the plate and place it on the plate |

overall: 404/450 = 89.8%  (mean of per-task rates 89.8%)

## evaluate.py protocol check with an UNTRAINED checkpoint (numbers are meaningless by construction)

| task | env_num (episodes) | steps run | max_steps | successes | success_rate | seconds | rc |
|---|---|---|---|---|---|---|---|
| 0 | 10 | None | None | None | None | None | 0 |
| 1 | 10 | None | None | None | None | None | 0 |
| 2 | 10 | None | None | None | None | None | 0 |
| 3 | 10 | None | None | None | None | None | 0 |
| 4 | 10 | None | None | None | None | None | 0 |
| 5 | 10 | None | None | None | None | None | 0 |
| 6 | 10 | None | None | None | None | None | 0 |
| 7 | 10 | None | None | None | None | None | 0 |
| 8 | 10 | None | None | None | None | None | 0 |
| 9 | 10 | None | None | None | None | None | 0 |

## Stages

| stage | rc | seconds |
|---|---|---|
| 00_sysinfo | 0 | 0.6 |
| 01_apt | 0 | 17.3 |
| 02_uv | 0 | 4.3 |
| 02_venv | 0 | 1.8 |
| 03_clone | 0 | 13.2 |
| 04_torch | 0 | 4.2 |
| 04_reqs | 0 | 21.9 |
| 04_libero | 0 | 1.0 |
| 05_import | 0 | 5.7 |
| 06_smoke_egl | 0 | 29.2 |
| 07_dataset | 0 | 24.6 |
| 08_bert | 0 | 7.7 |
| 09_replay_A | 0 | 13430.6 |
| 09_replay_B | -9 | 14400.6 |
| 10_random_ckpt | 0 | 12.8 |
| 10_eval_task0_env20 | 0 | 3.0 |
| 10_eval_task0_env10 | 0 | 2.7 |
| 10_eval_task1_env10 | 0 | 2.6 |
| 10_eval_task2_env10 | 0 | 2.6 |
| 10_eval_task3_env10 | 0 | 2.7 |
| 10_eval_task4_env10 | 0 | 2.7 |
| 10_eval_task5_env10 | 0 | 2.7 |
| 10_eval_task6_env10 | 0 | 2.6 |
| 10_eval_task7_env10 | 0 | 2.7 |
| 10_eval_task8_env10 | 0 | 2.8 |
| 10_eval_task9_env10 | 0 | 2.7 |
| 11_disk | 0 | 0.5 |
