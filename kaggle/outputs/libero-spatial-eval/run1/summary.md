# LIBERO libero_spatial on Kaggle - summary

- started (UTC): 2026-10-23T13:58:40.911485  elapsed: 1.6 min
- GPU: Tesla T4, 15360 MiB
Tesla T4, 15360 MiB  nproc: 4  render backend: NONE (egl and osmesa both failed)
- LIBERO commit: 8f1084e; dataset: 5.81 GiB, 10 hdf5 files


## Stages

| stage | rc | seconds |
|---|---|---|
| 00_sysinfo | 0 | 0.6 |
| 01_apt | 0 | 15.0 |
| 02_uv | 0 | 3.7 |
| 02_venv | 0 | 1.6 |
| 03_clone | 0 | 11.8 |
| 04_torch | 0 | 6.0 |
| 04_reqs | 0 | 11.8 |
| 04_libero | 0 | 1.0 |
| 05_import | 0 | 6.2 |
| 06_smoke_egl | 1 | 0.2 |
| 06_smoke_osmesa | 1 | 0.2 |
| 07_dataset | 0 | 36.3 |
| 08_bert | 0 | 2.3 |
| 11_disk | 0 | 0.5 |
