# LIBERO libero_spatial on Kaggle - summary

- started (UTC): 2026-10-23T20:15:37.292093  elapsed: 15.7 min
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB  nproc: 4  render backend: egl
- LIBERO commit: 8f1084e; dataset: 5.81 GiB, 10 hdf5 files
- renderer: GL_RENDERER: b'Tesla T4/PCIe/SSE2' | GL_VENDOR: b'NVIDIA Corporation' | GL_VERSION: b'4.6.0 NVIDIA 580.178.04'  step cost: STEP_MS 196.2 ms/step (50 zero-action steps incl. 2x128x128 rendering)

## Demo replay (open-loop action replay of the recorded demos, success = env.step done)

| task | n | succ | rate | SE | mean steps→succ | mean len | mean max drift | s/demo | language |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 50 | 47 | 94% | 0.0336 | 123.6 | 137.6 | 5.491 | 10.3 | pick up the black bowl on the wooden cabinet and place it on the plate |

overall: 47/50 = 94.0%  (mean of per-task rates 94.0%)

## evaluate.py protocol check with an UNTRAINED checkpoint (numbers are meaningless by construction)

| task | env_num (episodes) | steps run | max_steps | successes | success_rate | rollout s | wall s | rc |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | None | None | None | None | None | 53.7 | 0 |
| 1 | 10 | None | None | None | None | None | 20.1 | 0 |
| 2 | 10 | None | None | None | None | None | 20.7 | 0 |
| 3 | 10 | None | None | None | None | None | 20.8 | 0 |
| 4 | 10 | None | None | None | None | None | 20.8 | 0 |
| 5 | 10 | None | None | None | None | None | 20.4 | 0 |
| 6 | 10 | None | None | None | None | None | 20.5 | 0 |
| 7 | 10 | None | None | None | None | None | 20.4 | 0 |
| 8 | 10 | None | None | None | None | None | 20.3 | 0 |
| 9 | 10 | None | None | None | None | None | 20.3 | 0 |

## Stages

| stage | rc | seconds |
|---|---|---|
| 00_sysinfo | 0 | 0.6 |
| 01_apt | 0 | 20.7 |
| 01_egl_vendor | 0 | 0.1 |
| 02_uv | 0 | 4.0 |
| 02_venv | 0 | 1.6 |
| 03_clone | 0 | 23.4 |
| 04_torch | 0 | 4.2 |
| 04_reqs | 0 | 14.9 |
| 04_libero | 0 | 1.0 |
| 05_import | 0 | 10.3 |
| 05_import_eval_deps | 0 | 9.6 |
| 06_smoke_egl | 0 | 27.6 |
| 07_dataset | 0 | 28.8 |
| 08_bert | 0 | 6.5 |
| 10_random_ckpt | 0 | 7.7 |
| 10_eval_task0_env20 | 0 | 33.0 |
| 10_eval_task0_env10 | 0 | 20.6 |
| 09_replay_A | 0 | 530.0 |
| 10_eval_task1_env10 | 0 | 20.1 |
| 10_eval_task2_env10 | 0 | 20.7 |
| 10_eval_task3_env10 | 0 | 20.8 |
| 10_eval_task4_env10 | 0 | 20.8 |
| 10_eval_task5_env10 | 0 | 20.4 |
| 10_eval_task6_env10 | 0 | 20.5 |
| 10_eval_task7_env10 | 0 | 20.4 |
| 10_eval_task8_env10 | 0 | 20.3 |
| 10_eval_task9_env10 | 0 | 20.3 |
| 11_disk | 0 | 0.4 |
