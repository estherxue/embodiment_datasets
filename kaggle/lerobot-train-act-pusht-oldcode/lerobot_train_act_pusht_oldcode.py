#!/usr/bin/env python3
"""Kaggle script kernel: re-train alexandersoare/act_pusht_keypoints with the AUTHOR's code and config, then evaluate the
checkpoints with our lerobot 0.3.3 protocol.

Why: our lerobot 0.3.3 runs of the same recipe reach 47.6% at 100k steps (3 training seeds, 1500 episodes) while the
author's checkpoint scores 64.0% (62.7% when we re-evaluate it). The author's config.yaml, the ACT model code on the
keypoint path and the dataset arrays (revision v1.5 vs v2.1) are identical between the two code bases, see
embodied-eval-notes.md §3.8. This kernel removes the remaining differences at once by running the 2024-07 lerobot code
(commit 471eab3d, "Make ACT compatible with observation.environment_state", merged the day after the author's upload)
with the author's config.yaml verbatim: Hydra train.py, seed 100000, use_amp=true, dataset revision v1.5, torch 2.3.
A checkpoint is written every 10k steps; each one is converted to the 0.3.3 config format (model.safetensors keys are
unchanged) and evaluated with lerobot 0.3.3 lerobot-eval exactly like our other PushT runs: 3 eval seeds x 50 episodes
per checkpoint, 10 seeds x 50 = 500 episodes for the last one. Evaluation runs on the same GPU while training continues.

Outputs: /kaggle/working/{train/old_act_kp/train_log.json, eval/<run>/eval_info.json, ckpts/old_act_kp_100000/
          {pretrained_model (old format), converted}, env/*, logs/*.log, summary.json, summary.md}
"""
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
LOGS, EVAL_DIR, TRAIN_DIR, ENV_DIR, CKPT_OUT = WORK / "logs", WORK / "eval", WORK / "train", WORK / "env", WORK / "ckpts"

LEROBOT_REPO = "https://github.com/huggingface/lerobot"
OLD_COMMIT = "471eab3d"    # 2024-07-11 Make ACT compatible with "observation.environment_state" (#314); CODEBASE_VERSION v1.5
NEW_COMMIT = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"  # chore: Bump to 0.3.3 (#1690), 2025-08-06
OLD_SRC, OLD_VENV = TMP / "lerobot_old", TMP / "venv_old"
NEW_SRC, NEW_VENV = TMP / "lerobot", TMP / "venv"
OLD_PY = str(OLD_VENV / "bin" / "python")
NEW_PY = str(NEW_VENV / "bin" / "python")
LEROBOT_EVAL = str(NEW_VENV / "bin" / "lerobot-eval")
AUTHOR_REPO = "alexandersoare/act_pusht_keypoints"
AUTHOR_CFG_URL = f"https://huggingface.co/{AUTHOR_REPO}/resolve/main/config.yaml"

RUN = "old_act_kp"
STEPS = 100_000
SAVE_FREQ = 10_000
LOG_FREQ = 100           # lerobot prints "step:1K" (lossy), so the step is recovered from the line index
BATCH_TRAIN = 64
# everything else comes from the author's config.yaml verbatim (seed 100000, use_amp true, lr 3e-5, chunk 16, ...)
HYDRA_OVERRIDES = (f"wandb.enable=false training.eval_freq=-1 training.save_freq={SAVE_FREQ} training.log_freq={LOG_FREQ} "
                   f"training.num_workers=4 training.online_steps_between_rollouts=1 training.online_env_seed=1")
EVAL_ENV = "--env.type=pusht --env.obs_type=environment_state_agent_pos"
EVAL_SEEDS = [1000, 2000, 3000]
FINAL_EVAL_SEEDS = [1000 * i for i in range(1, 11)]
N_EPISODES, BATCH_EVAL = 50, 50
TRAIN_BUDGET_S = 10.0 * 3600   # kill training after this (Kaggle hard limit 12 h); the final 500-episode eval still fits
POLL_S = 30
REF_AUTHOR = (320, 500, "alexandersoare/act_pusht_keypoints eval_info.json (100k steps, 64.0%)")
REF_OURS = (714, 1500, "our lerobot 0.3.3 runs, 100k steps, train seeds 1-3 pooled (47.6%)")

SUMMARY = {
    "meta": {"run": RUN, "old_commit": OLD_COMMIT, "new_commit": NEW_COMMIT, "author_repo": AUTHOR_REPO, "steps": STEPS,
             "save_freq": SAVE_FREQ, "batch_train": BATCH_TRAIN, "hydra_overrides": HYDRA_OVERRIDES, "eval_seeds": EVAL_SEEDS,
             "final_eval_seeds": FINAL_EVAL_SEEDS, "n_episodes": N_EPISODES, "batch_eval": BATCH_EVAL,
             "ref_author": REF_AUTHOR, "ref_ours": REF_OURS, "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
    "train": {},
    "runs": {},
    "errors": [],
}
RUN_ENV = {
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": str(TMP / "hf"),
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
}

GPU_CHECK = r'''
import json, torch
info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
if info["cuda_available"]:
    info["device_name"] = torch.cuda.get_device_name(0)
    try:
        x = torch.randn(512, 512, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize()
        info["gpu_ok"] = bool(y == y)
    except Exception as e:
        info["gpu_ok"] = False; info["error"] = repr(e)[:300]
else:
    info["gpu_ok"] = False
print("GPU_CHECK " + json.dumps(info), flush=True)
'''

# Hydra-era ACT checkpoint (pretrained_model dir with the dataclass config.json) -> lerobot 0.3.3 config.json format.
# Same mapping as the one used to evaluate the author's checkpoint (kaggle/lerobot-eval); runs inside the 0.3.3 venv.
CONVERT = r'''
import json, os, shutil, sys
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

src, out = sys.argv[1], sys.argv[2]
old = json.load(open(f"{src}/config.json"))
TYPE = {"observation.image": "VISUAL", "observation.state": "STATE", "observation.environment_state": "ENV", "action": "ACTION"}
MODE = {"mean_std": "MEAN_STD", "min_max": "MIN_MAX", "identity": "IDENTITY"}
new = {"type": "act", "n_obs_steps": old["n_obs_steps"], "normalization_mapping": {}, "input_features": {},
       "output_features": {}, "device": "cpu", "use_amp": False}
for k, shape in old["input_shapes"].items():
    new["input_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["input_normalization_modes"][k]]
for k, shape in old["output_shapes"].items():
    new["output_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["output_normalization_modes"][k]]
for k in ["chunk_size", "n_action_steps", "vision_backbone", "pretrained_backbone_weights",
          "replace_final_stride_with_dilation", "pre_norm", "dim_model", "n_heads", "dim_feedforward",
          "feedforward_activation", "n_encoder_layers", "n_decoder_layers", "use_vae", "latent_dim",
          "n_vae_encoder_layers", "dropout", "kl_weight"]:
    new[k] = old[k]
new["temporal_ensemble_coeff"] = old.get("temporal_ensemble_coeff", old.get("temporal_ensemble_momentum"))
os.makedirs(out, exist_ok=True)
json.dump(new, open(f"{out}/config.json", "w"), indent=2)
shutil.copy(f"{src}/model.safetensors", f"{out}/model.safetensors")
cfg = PreTrainedConfig.from_pretrained(out)
pol = ACTPolicy.from_pretrained(out, config=cfg, strict=True)  # raises on any missing / unexpected key
print(f"CONVERT_OK {src} -> {out}: {sum(p.numel() for p in pol.parameters()) / 1e6:.1f}M params; "
      f"inputs={list(cfg.input_features)} norm={cfg.normalization_mapping} chunk={cfg.chunk_size} "
      f"n_action_steps={cfg.n_action_steps}", flush=True)
print("normalize buffers finite:", all(torch.isfinite(b).all().item() for n, b in pol.named_buffers() if "normalize" in n))
'''


# ----------------------------------------------------------------------------- utilities
def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def save_summary():
    SUMMARY["meta"]["elapsed_s"] = round(time.time() - T0, 1)
    with open(WORK / "summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=2)
    try:
        write_summary_md()
    except Exception:
        SUMMARY["errors"].append("write_summary_md: " + traceback.format_exc()[-800:])


def log_tail(name, n=40, width=220):
    path = LOGS / f"{name}.log"
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()[-n:]
    return [line.split("\r")[-1][:width] for line in lines]


def start_bg(cmd, log_name, env=None):
    """bash -c cmd in the background; stdout+stderr -> logs/<log_name>.log; returns the Popen."""
    log(f"$ {cmd}")
    e = dict(os.environ)
    e.update(RUN_ENV)
    e.update(env or {})
    f = open(LOGS / f"{log_name}.log", "a")
    f.write(f"\n===== [{time.strftime('%H:%M:%S')}] $ {cmd}\n")
    f.flush()
    return subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT, env=e,
                            start_new_session=True)


def sh(cmd, log_name, env=None, timeout=None, tail=20):
    p = start_bg(cmd, log_name, env)
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        rc = -9
        with open(LOGS / f"{log_name}.log", "a") as f:
            f.write(f"\n===== TIMEOUT after {timeout}s, killed\n")
    if tail:
        for line in log_tail(log_name, tail):
            print("    | " + line, flush=True)
    log(f"-> rc={rc}  ({log_name})")
    return rc


def grep_log(log_name, prefix):
    for line in reversed(log_tail(log_name, 400, width=4000)):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def write_script(name, body):
    path = TMP / name
    path.write_text(body.lstrip())
    return path


# ----------------------------------------------------------------------------- statistics
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se else float("nan")
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if z == z else float("nan")
    return z, pval


def stats_from_eval_info(path):
    with open(path) as f:
        info = json.load(f)
    eps = info["per_episode"]
    n = len(eps)
    k = int(sum(1 for e in eps if e["success"]))
    lo_w, hi_w = wilson_ci(k, n)
    agg = info["aggregated"]
    return dict(n_episodes=n, n_success=k, pc_success=100 * k / n if n else float("nan"), ci95_wilson=[100 * lo_w, 100 * hi_w],
                avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"], eval_s=agg["eval_s"],
                successes=[bool(e["success"]) for e in eps])


LOG_RE = re.compile(r"step:(\S+).*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+|nan|inf).*?lr:([0-9.eE+-]+)")


def parse_train_log(log_name):
    curve = []
    for line in (LOGS / f"{log_name}.log").read_text(errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            step = LOG_FREQ * (len(curve) + 1)
            curve.append(dict(step=step, total_step=step, step_str=m.group(1), loss=float(m.group(2)),
                              grad_norm=float(m.group(3)) if m.group(3) not in ("nan", "inf") else None, lr=float(m.group(4))))
    return curve


def write_summary_md():
    meta, t = SUMMARY["meta"], SUMMARY["train"]
    lines = [f"# ACT on PushT keypoints, retrained with the author's 2024-07 lerobot code and config.yaml, Kaggle", "",
             f"- training code: lerobot `{meta.get('old_commit_full', OLD_COMMIT)}` (2024-07-11), torch {meta.get('old_torch')}, "
             f"dataset lerobot/pusht_keypoints revision v1.5; config: {AUTHOR_REPO} config.yaml verbatim + `{HYDRA_OVERRIDES}`",
             f"- evaluation code: lerobot `{NEW_COMMIT}` (v0.3.3), checkpoints converted to the 0.3.3 config format",
             f"- device: GPU {meta.get('gpu', {}).get('device_name')}  |  eval torch {meta.get('gpu', {}).get('torch')}",
             f"- training: {STEPS} steps, batch {BATCH_TRAIN}, save every {SAVE_FREQ}; rc={t.get('rc')}, wall {t.get('wall_s', 0) / 60:.1f} min, "
             f"{t.get('it_s', '-')} it/s, killed_by_budget={t.get('killed')}",
             f"- evaluation: intermediate checkpoints seeds {EVAL_SEEDS} x {N_EPISODES} episodes; last checkpoint seeds {FINAL_EVAL_SEEDS} x {N_EPISODES}",
             f"- started {meta['started_utc']} UTC, elapsed {meta.get('elapsed_s', 0) / 60:.1f} min", "",
             "## Training loss (L1 + 10*KL), mean over the 10 log points before each checkpoint", "",
             "| step | loss (mean of last 1k steps) | grad norm | lr |", "|---|---|---|---|"]
    curve = t.get("curve") or []
    for s in range(SAVE_FREQ, STEPS + 1, SAVE_FREQ):
        pts = [c for c in curve if s - 1000 < c["total_step"] <= s]
        if pts:
            gn = [c["grad_norm"] for c in pts if c["grad_norm"] is not None]
            lines.append(f"| {s} | {sum(c['loss'] for c in pts) / len(pts):.4f} | {sum(gn) / len(gn) if gn else float('nan'):.3f} | {pts[-1]['lr']:.1e} |")
    lines += ["", "## Success rate per checkpoint (lerobot 0.3.3 eval, 50 episodes per seed)", "",
              "| step | seeds | per-seed success | pooled k/N | pooled % | Wilson 95% CI | std across seeds (pts) |", "|---|---|---|---|---|---|---|"]
    by_step = {}
    for r in SUMMARY["runs"].values():
        if r.get("stats"):
            by_step.setdefault(r["total_step"], []).append(r)
    for s in sorted(by_step):
        rs = sorted(by_step[s], key=lambda r: r["seed"])
        pcs = [r["stats"]["pc_success"] for r in rs]
        k = sum(r["stats"]["n_success"] for r in rs)
        n = sum(r["stats"]["n_episodes"] for r in rs)
        mean = sum(pcs) / len(pcs)
        std = math.sqrt(sum((p - mean) ** 2 for p in pcs) / (len(pcs) - 1)) if len(pcs) > 1 else float("nan")
        lo, hi = wilson_ci(k, n)
        lines.append(f"| {s} | {len(rs)} | {' / '.join(f'{p:.0f}' for p in pcs)} | {k}/{n} | {100 * k / n:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] | {std:.1f} |")
    last = max(by_step) if by_step else None
    if last is not None:
        rs = by_step[last]
        k = sum(r["stats"]["n_success"] for r in rs)
        n = sum(r["stats"]["n_episodes"] for r in rs)
        lines += ["", f"## Last checkpoint ({last} steps, {k}/{n} = {100 * k / n:.1f}%) vs references", ""]
        for rk, rn, rl in (REF_AUTHOR, REF_OURS):
            z, p = two_prop_z(k, n, rk, rn)
            lines.append(f"- vs {rl}: {100 * rk / rn:.1f}%, z={z:+.2f}, p={p:.2g}")
    if SUMMARY["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e[:500]}" for e in SUMMARY["errors"]]
    (WORK / "summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- steps
def find_uv():
    uv = shutil.which("uv")
    if uv:
        return uv
    sh(f"{sys.executable} -m pip install -q uv", "01_install")
    return shutil.which("uv") or f"{sys.executable} -m uv"


def gpu_check(py, log_name):
    script = write_script("gpu_check.py", GPU_CHECK)
    sh(f"{py} {script}", log_name)
    raw = grep_log(log_name, "GPU_CHECK ")
    try:
        return json.loads(raw) if raw else {"gpu_ok": False, "error": "no GPU_CHECK line"}
    except Exception:
        return {"gpu_ok": False, "error": raw}


def step_install_old(uv):
    """2024-07 lerobot in its own venv. Its pyproject (poetry) is installed with --no-deps and the dependencies are pinned
    to period versions by hand: the loose ranges (gymnasium>=0.29.1, moviepy>=1.0.3, ...) would resolve to 2025 majors."""
    L = "01_install_old"
    assert sh(f"{uv} venv --python 3.10 {OLD_VENV}", L) == 0, "uv venv (old) failed"
    assert sh(f"git clone -q {LEROBOT_REPO} {OLD_SRC} && cd {OLD_SRC} && git checkout -q {OLD_COMMIT} && "
              f"git log -1 --format='OLD_COMMIT_FULL %H %cd %s' --date=short", L) == 0, "git clone/checkout (old) failed"
    SUMMARY["meta"]["old_commit_full"] = (grep_log(L, "OLD_COMMIT_FULL ") or "").split(" ")[0]
    assert sh(f"{uv} pip install --python {OLD_PY} torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
              L) == 0, "torch (old) install failed"
    assert sh(f"{uv} pip install --python {OLD_PY} --no-deps -e {OLD_SRC}", L) == 0, "lerobot (old) install failed"
    deps = ('"numpy<2" termcolor "omegaconf>=2.3" "wandb>=0.16.3,<0.19" "imageio[ffmpeg]>=2.34" "hydra-core>=1.3.2" "einops>=0.8" '
            '"pymunk>=6.6" "diffusers>=0.27.2,<0.28" "huggingface-hub[hf-transfer]>=0.23,<0.24" "gymnasium==0.29.1" '
            '"datasets>=2.19,<3" "av>=12" "moviepy>=1.0.3,<2" "deepdiff>=7"')
    assert sh(f"{uv} pip install --python {OLD_PY} {deps}", L) == 0, "old dependencies install failed"
    assert sh(f"{OLD_PY} -c \"import lerobot, torch, datasets, huggingface_hub, gymnasium, hydra, av; "
              f"print('OLD_TORCH', torch.__version__, 'datasets', datasets.__version__, 'hub', huggingface_hub.__version__)\"", L) == 0, \
        "old lerobot import failed"
    SUMMARY["meta"]["old_torch"] = (grep_log(L, "OLD_TORCH ") or "").split(" ")[0]
    gpu = gpu_check(OLD_PY, "02_gpu_check_old")
    SUMMARY["meta"]["gpu_old"] = gpu
    assert gpu.get("gpu_ok"), f"GPU not usable from the old venv: {gpu}"
    sh(f"{uv} pip freeze --python {OLD_PY} > {ENV_DIR}/requirements-freeze-old.txt; wc -l {ENV_DIR}/requirements-freeze-old.txt", L, tail=2)
    # the author's resolved Hydra config, used verbatim as the primary config
    cfg_dir = TMP / "author_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    assert sh(f"curl -sSL -o {cfg_dir}/config.yaml {AUTHOR_CFG_URL} && head -c 400 {cfg_dir}/config.yaml", L, tail=12) == 0, "config.yaml download failed"
    shutil.copy(cfg_dir / "config.yaml", ENV_DIR / "author_config.yaml")
    save_summary()
    return cfg_dir


def step_install_new(uv):
    """lerobot 0.3.3 for evaluation, same recipe as the other kernels."""
    L = "01_install_new"
    assert sh(f"{uv} venv --python 3.10 {NEW_VENV}", L) == 0, "uv venv (new) failed"
    assert sh(f"git clone -q {LEROBOT_REPO} {NEW_SRC} && cd {NEW_SRC} && git checkout -q {NEW_COMMIT} && "
              f"git log -1 --format='%H %cd %s' --date=short", L) == 0, "git clone/checkout (new) failed"
    assert sh(f"{uv} pip install --python {NEW_PY} torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126",
              L) == 0, "torch (new) install failed"
    assert sh(f"{uv} pip install --python {NEW_PY} --no-deps torchcodec==0.4.0", L) == 0, "torchcodec install failed"
    (TMP / "constraints.txt").write_text("torch==2.7.1\ntorchvision==0.22.1\ntorchcodec==0.4.0\n")
    assert sh(f'{uv} pip install --python {NEW_PY} -e "{NEW_SRC}[pusht]" -c {TMP}/constraints.txt', L) == 0, "lerobot (new) install failed"
    gpu = gpu_check(NEW_PY, "02_gpu_check_new")
    SUMMARY["meta"]["gpu"] = gpu
    assert gpu.get("gpu_ok"), f"GPU not usable from the new venv: {gpu}"
    sh(f"{uv} pip freeze --python {NEW_PY} > {ENV_DIR}/requirements-freeze-new.txt; wc -l {ENV_DIR}/requirements-freeze-new.txt", L, tail=2)
    save_summary()


def convert_checkpoint(pm_dir, step):
    out = TMP / "converted" / f"{step:06d}"
    script = write_script("convert_old_act_ckpt.py", CONVERT)
    rc = sh(f"{NEW_PY} {script} {pm_dir} {out}", f"convert_{step:06d}", timeout=900, tail=3)
    msg = grep_log(f"convert_{step:06d}", "CONVERT_OK ")
    SUMMARY["train"].setdefault("converted", {})[str(step)] = msg or f"FAILED rc={rc}"
    assert msg, f"conversion of {pm_dir} failed (rc={rc})"
    return out


def run_eval(ckpt_dir, total_step, eval_seed):
    name = f"{RUN}_s{total_step:06d}_eval{eval_seed}"
    out = EVAL_DIR / name
    if out.exists() and not (out / "eval_info.json").exists():
        shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_EVAL} --policy.path={ckpt_dir} --policy.device=cuda {EVAL_ENV} "
           f"--eval.n_episodes={N_EPISODES} --eval.batch_size={BATCH_EVAL} --seed={eval_seed} --output_dir={out}")
    t = time.time()
    rc = sh(cmd, name, timeout=2400, tail=4)
    rec = {"total_step": total_step, "seed": eval_seed, "policy_path": str(ckpt_dir), "rc": rc, "wall_s": round(time.time() - t, 1), "cmd": cmd}
    info = out / "eval_info.json"
    if info.exists():
        try:
            rec["stats"] = stats_from_eval_info(info)
            s = rec["stats"]
            log(f"RESULT {name}: {s['n_success']}/{s['n_episodes']} = {s['pc_success']:.1f}%  "
                f"CI95 [{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}]  avg_max_reward={s['avg_max_reward']:.3f}")
        except Exception:
            rec["error"] = "stats: " + traceback.format_exc()[-600:]
    else:
        rec["error"] = " / ".join(log_tail(name, 12))
        log(f"FAILED {name}: no eval_info.json (rc={rc})")
    vids = out / "videos"
    if vids.exists():
        for p in vids.iterdir():
            if p.name != "eval_episode_0.mp4":
                p.unlink()
    SUMMARY["runs"][name] = rec
    save_summary()
    return rec


def evaluate_checkpoint(step_dir, final):
    step = int(step_dir.name)
    pm = step_dir / "pretrained_model"
    for f in ("config.json", "config.yaml"):
        if (pm / f).exists():
            shutil.copy(pm / f, ENV_DIR / f"{RUN}_{step:06d}.{f}")
    conv = convert_checkpoint(pm, step)
    for es in (FINAL_EVAL_SEEDS if final else EVAL_SEEDS):
        run_eval(conv, step, es)
    # drop optimizer/rng state of older checkpoints to save disk; keep pretrained_model of every checkpoint until the end
    for other in step_dir.parent.iterdir():
        if other.is_dir() and not other.is_symlink() and other != step_dir and (other / "training_state.pth").exists():
            (other / "training_state.pth").unlink()


def current_last(out):
    last = out / "checkpoints" / "last"
    if last.is_symlink():
        try:
            return last.resolve()
        except OSError:
            return None
    return None


def main():
    for d in (TMP, LOGS, EVAL_DIR, TRAIN_DIR, ENV_DIR, CKPT_OUT):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    sh("nvidia-smi || true; echo; python -V; nproc; free -g; df -h /kaggle/working /kaggle/tmp /tmp | cat", "00_env_info", tail=30)
    shutil.copy(LOGS / "00_env_info.log", ENV_DIR / "env_info.txt")
    uv = find_uv()
    try:
        assert sh(f"{uv} --version", "01_install_old") == 0, "uv not available"
        cfg_dir = step_install_old(uv)
    except Exception:
        SUMMARY["errors"].append("install old: " + traceback.format_exc()[-1500:])
        log("INSTALL (old) FAILED:\n" + traceback.format_exc())
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    out = TMP / "train" / RUN
    shutil.rmtree(out, ignore_errors=True)
    cmd = (f"cd {OLD_SRC} && {OLD_PY} lerobot/scripts/train.py --config-path={cfg_dir} --config-name=config "
           f"hydra.run.dir={out} hydra.job.name={RUN} {HYDRA_OVERRIDES}")
    t_train = time.time()
    p = start_bg(cmd, RUN)
    SUMMARY["train"] = dict(cmd=cmd, rc=None, killed=False)
    save_summary()

    try:  # the 0.3.3 evaluation venv is installed while the old code trains (CPU/network only)
        step_install_new(uv)
    except Exception:
        SUMMARY["errors"].append("install new: " + traceback.format_exc()[-1500:])
        log("INSTALL (new) FAILED, killing training:\n" + traceback.format_exc())
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    done = set()
    killed = False
    while True:
        rc = p.poll()
        last = current_last(out)
        if last is not None and last.name not in done and (last / "pretrained_model" / "model.safetensors").exists():
            final = rc is not None or int(last.name) >= STEPS
            if not final and time.time() - T0 > TRAIN_BUDGET_S:
                final = True
            log(f"new checkpoint {last.name}, final={final}; training rc={rc}")
            try:
                evaluate_checkpoint(last, final)
            except Exception:
                SUMMARY["errors"].append(f"eval {last.name}: " + traceback.format_exc()[-800:])
            done.add(last.name)
            SUMMARY["train"]["curve"] = parse_train_log(RUN)
            save_summary()
            if final:
                rc = p.poll()
                if rc is None:  # still running: last checkpoint reached under the time budget, or the trainer still exiting
                    log("final checkpoint evaluated while training still runs: killing training")
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    killed = int(last.name) < STEPS
                    rc = p.returncode
                break
        if rc is not None:
            last = current_last(out)
            if last is None or last.name in done:
                break
            continue
        if time.time() - T0 > TRAIN_BUDGET_S and (last is None or last.name in done):
            log("time budget reached with no new checkpoint: killing training")
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            killed = True
            rc = p.returncode
            break
        time.sleep(POLL_S)

    last = current_last(out)
    if last is not None and (last / "pretrained_model" / "model.safetensors").exists():
        step = int(last.name)
        have = {r["seed"] for r in SUMMARY["runs"].values() if r.get("total_step") == step and r.get("stats")}
        missing = [es for es in FINAL_EVAL_SEEDS if es not in have]
        if missing:
            try:
                conv = TMP / "converted" / f"{step:06d}"
                if not (conv / "model.safetensors").exists():
                    conv = convert_checkpoint(last / "pretrained_model", step)
                for es in missing:
                    run_eval(conv, step, es)
            except Exception:
                SUMMARY["errors"].append(f"final eval {step}: " + traceback.format_exc()[-800:])
        keep = CKPT_OUT / f"{RUN}_{step:06d}"
        shutil.copytree(last / "pretrained_model", keep / "pretrained_model", dirs_exist_ok=True)
        conv = TMP / "converted" / f"{step:06d}"
        if conv.exists():
            shutil.copytree(conv, keep / "converted", dirs_exist_ok=True)
    curve = parse_train_log(RUN)
    wall = time.time() - t_train
    SUMMARY["train"].update(rc=rc, killed=killed, wall_s=round(wall, 1), curve=curve,
                            steps_done=curve[-1]["total_step"] if curve else 0,
                            it_s=round(curve[-1]["total_step"] / wall, 2) if curve else None)
    TRAIN_DIR.joinpath(RUN).mkdir(parents=True, exist_ok=True)
    json.dump(dict(name=RUN, rc=rc, killed=killed, wall_s=round(wall, 1), steps=STEPS, batch=BATCH_TRAIN, cmd=cmd, curve=curve),
              open(TRAIN_DIR / RUN / "train_log.json", "w"), indent=1)
    if curve:
        log(f"TRAIN {RUN}: rc={rc}, {curve[-1]['total_step']} steps, {wall / 60:.1f} min ({curve[-1]['total_step'] / wall:.1f} it/s), "
            f"final loss {curve[-1]['loss']:.4f}")
    save_summary()
    print("\n" + (WORK / "summary.md").read_text(), flush=True)
    log("KERNEL_DONE")


if __name__ == "__main__":
    main()
