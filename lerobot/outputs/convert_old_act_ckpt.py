"""Convert a 2024/Jan-2025 (Hydra-era) lerobot ACT checkpoint to the draccus config.json format used by lerobot v0.3.3.

The model.safetensors key names did not change, so only config.json needs rewriting.
Usage: python outputs/convert_old_act_ckpt.py <hub_repo_id> <out_dir>
"""
import json, shutil, sys
from huggingface_hub import hf_hub_download
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

repo, out = sys.argv[1], sys.argv[2]
old = json.load(open(hf_hub_download(repo, "config.json")))
TYPE = {"observation.image": "VISUAL", "observation.state": "STATE", "observation.environment_state": "ENV", "action": "ACTION"}
MODE = {"mean_std": "MEAN_STD", "min_max": "MIN_MAX", "identity": "IDENTITY"}
new = {"type": "act", "n_obs_steps": old["n_obs_steps"], "normalization_mapping": {}, "input_features": {}, "output_features": {},
       "device": "cpu", "use_amp": False}
for k, shape in old["input_shapes"].items():
    new["input_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["input_normalization_modes"][k]]
for k, shape in old["output_shapes"].items():
    new["output_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["output_normalization_modes"][k]]
for k in ["chunk_size", "n_action_steps", "vision_backbone", "pretrained_backbone_weights", "replace_final_stride_with_dilation",
          "pre_norm", "dim_model", "n_heads", "dim_feedforward", "feedforward_activation", "n_encoder_layers", "n_decoder_layers",
          "use_vae", "latent_dim", "n_vae_encoder_layers", "dropout", "kl_weight"]:
    new[k] = old[k]
new["temporal_ensemble_coeff"] = old.get("temporal_ensemble_coeff", old.get("temporal_ensemble_momentum"))
import os; os.makedirs(out, exist_ok=True)
json.dump(new, open(f"{out}/config.json", "w"), indent=2)
shutil.copy(hf_hub_download(repo, "model.safetensors"), f"{out}/model.safetensors")
for extra in ("config.yaml", "README.md", "eval_info.json"):
    try: shutil.copy(hf_hub_download(repo, extra), f"{out}/{extra}.orig")
    except Exception: pass
# verify: strict load (raises on any missing/unexpected key)
cfg = PreTrainedConfig.from_pretrained(out)
pol = ACTPolicy.from_pretrained(out, config=cfg, strict=True)
print(f"{repo} -> {out}: strict load OK, {sum(p.numel() for p in pol.parameters())/1e6:.1f}M params; "
      f"inputs={list(cfg.input_features)} norm={cfg.normalization_mapping} chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps}")
print("normalize_inputs buffers finite:", all(torch.isfinite(b).all().item() for n, b in pol.named_buffers() if 'normalize' in n))
