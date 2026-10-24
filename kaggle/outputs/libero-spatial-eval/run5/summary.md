# LIBERO libero_spatial on Kaggle - summary

- started (UTC): 2026-10-23T20:37:51.197400  elapsed: 158.1 min
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB  nproc: 4  render backend: egl
- LIBERO commit: 8f1084e; dataset: 5.81 GiB, 10 hdf5 files
- renderer: GL_RENDERER: b'Tesla T4/PCIe/SSE2' | GL_VENDOR: b'NVIDIA Corporation' | GL_VERSION: b'4.6.0 NVIDIA 580.178.04'  step cost: STEP_MS 192.3 ms/step (50 zero-action steps incl. 2x128x128 rendering)

## Demo replay (open-loop action replay of the recorded demos, success = env.step done)

| task | n | succ | rate | SE | mean steps→succ | mean len | mean max drift | s/demo | language |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 48 | 96% | 0.0277 | 81.6 | 101.4 | 4.864 | 7.5 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | 50 | 45 | 90% | 0.0424 | 110.4 | 134.1 | 5.486 | 8.9 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | 50 | 48 | 96% | 0.0277 | 98.8 | 117.6 | 3.463 | 8.2 | pick up the black bowl from table center and place it on the plate |
| 3 | 50 | 45 | 90% | 0.0424 | 84.3 | 101.0 | 4.544 | 7.9 | pick up the black bowl on the cookie box and place it on the plate |
| 4 | 50 | 41 | 82% | 0.0543 | 130.1 | 149.6 | 4.345 | 10.7 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 5 | 50 | 41 | 82% | 0.0543 | 101.1 | 115.9 | 6.353 | 8.7 | pick up the black bowl on the ramekin and place it on the plate |
| 6 | 50 | 48 | 96% | 0.0277 | 109.1 | 126.2 | 5.793 | 9.5 | pick up the black bowl next to the cookie box and place it on the plate |
| 7 | 50 | 42 | 84% | 0.0518 | 122.6 | 142.2 | 5.584 | 10.9 | pick up the black bowl on the stove and place it on the plate |
| 8 | 50 | 46 | 92% | 0.0384 | 100.9 | 119.3 | 4.774 | 8.7 | pick up the black bowl next to the plate and place it on the plate |
| 9 | 50 | 47 | 94% | 0.0336 | 123.6 | 137.6 | 5.491 | 10.7 | pick up the black bowl on the wooden cabinet and place it on the plate |

overall: 451/500 = 90.2%  (mean of per-task rates 90.2%)

## evaluate.py protocol check with an UNTRAINED checkpoint (numbers are meaningless by construction)

| task | backend | env_num (episodes) | steps run | max_steps | successes | success_rate | rollout s | wall s | rc |
|---|---|---|---|---|---|---|---|---|---|
| 0 | egl | 20 | 600 | 600 | 0 | 0.0 | 721.1 | 729.0 | 0 |
| 1 | egl | 20 | 600 | 600 | 0 | 0.0 | 699.8 | 709.0 | 0 |
| 2 | egl | 20 | 600 | 600 | 0 | 0.0 | 642.2 | 650.5 | 0 |
| 3 | egl | 20 | 600 | 600 | 0 | 0.0 | 663.4 | 671.5 | 0 |
| 4 | egl | 20 | 600 | 600 | 0 | 0.0 | 675.6 | 683.7 | 0 |
| 5 | egl | 20 | 600 | 600 | 0 | 0.0 | 650.0 | 657.9 | 0 |
| 6 | egl | 20 | 600 | 600 | 0 | 0.0 | 671.3 | 679.5 | 0 |
| 7 | egl | 20 | 600 | 600 | 0 | 0.0 | 705.8 | 713.5 | 0 |
| 8 | egl | 20 | 600 | 600 | 0 | 0.0 | 676.6 | 684.9 | 0 |
| 9 | egl | 20 | 600 | 600 | 0 | 0.0 | 692.3 | 700.3 | 0 |

## Stages

| stage | rc | seconds |
|---|---|---|
| 00_sysinfo | 0 | 0.7 |
| 01_apt | 0 | 19.0 |
| 01_egl_vendor | 0 | 0.1 |
| 02_uv | 0 | 4.3 |
| 02_venv | 0 | 1.8 |
| 03_clone | 0 | 17.4 |
| 03_patch_venv_spawn | 0 | 0.0 |
| 04_torch | 0 | 4.0 |
| 04_reqs | 0 | 14.2 |
| 04_libero | 0 | 4.1 |
| 05_import | 0 | 6.3 |
| 05_import_eval_deps | 0 | 8.8 |
| 06_smoke_egl | 0 | 27.4 |
| 07_dataset | 0 | 22.6 |
| 08_bert | 0 | 10.0 |
| 10_random_ckpt | 0 | 7.6 |
| 10_eval_task0_egl_env20 | 0 | 729.0 |
| 10_eval_task1_egl_env20 | 0 | 709.0 |
| 10_eval_task2_egl_env20 | 0 | 650.5 |
| 10_eval_task3_egl_env20 | 0 | 671.5 |
| 10_eval_task4_egl_env20 | 0 | 683.7 |
| 10_eval_task5_egl_env20 | 0 | 657.9 |
| 10_eval_task6_egl_env20 | 0 | 679.5 |
| 10_eval_task7_egl_env20 | 0 | 713.5 |
| 10_eval_task8_egl_env20 | 0 | 684.9 |
| 10_eval_task9_egl_env20 | 0 | 700.3 |
| 09_replay_A | 0 | 2190.0 |
| 09_replay_B | 0 | 2450.0 |
| 11_disk | 0 | 0.4 |
