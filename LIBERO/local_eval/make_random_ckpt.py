"""Build an UNTRAINED (randomly initialised) Multitask/BCTransformerPolicy checkpoint in the exact
layout that libero/lifelong/evaluate.py expects, so the evaluation code path can be exercised on a
CPU-only machine. The resulting success rate is NOT a meaningful number.

Run from the LIBERO repo root:
    python local_eval/make_random_ckpt.py            # -> experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep0.pth
"""
import os
import yaml
import torch
from easydict import EasyDict
from hydra import compose, initialize
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import (control_seed, create_experiment_dir, get_task_embs,
                                   safe_device, torch_save_model)

# same config composition as libero/lifelong/main.py (hydra), with the overrides used by the README
with initialize(config_path="../libero/configs", version_base=None):
    hydra_cfg = compose(config_name="config", overrides=[
        "benchmark_name=LIBERO_SPATIAL", "lifelong=multitask", "policy=bc_transformer_policy",
        "seed=10000", "device=cpu",
    ])
cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
control_seed(cfg.seed)
cfg.folder = get_libero_path("datasets")
cfg.bddl_folder = get_libero_path("bddl_files")
cfg.init_states_folder = get_libero_path("init_states")

benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
_, shape_meta = get_dataset(
    dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(0)),
    obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len)
print("shape_meta:", shape_meta)

descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
task_embs = get_task_embs(cfg, descriptions)  # bert-base-cased pooler output, cached under ./bert
print("task_embs:", tuple(task_embs.shape))

create_experiment_dir(cfg)  # ./experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_XXX
cfg.shape_meta = shape_meta
algo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device)
n_params = sum(p.numel() for p in algo.policy.parameters())
path = os.path.join(cfg.experiment_dir, "multitask_model_ep0.pth")
torch_save_model(algo.policy, path, cfg=cfg, previous_masks=None)
print(f"saved untrained policy ({n_params/1e6:.2f} M params) to {path}")
