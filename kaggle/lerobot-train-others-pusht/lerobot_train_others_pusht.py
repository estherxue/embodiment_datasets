#!/usr/bin/env python3
"""Kaggle script kernel: other LeRobot policies on PushT, same protocol as the ACT short-training kernel
(lerobot-train-act-pusht): 10k steps, batch 64, train seed 1, then lerobot-eval with 3 eval seeds x 50 episodes.

Trained here (lerobot 0.3.3 defaults unless noted):
  tdmpc_kp      TD-MPC, keypoints + agent pos, offline only (no online RL), ENV/STATE/ACTION min-max
  vqbet_px      VQ-BeT, 96x96 pixels + agent pos; 2k VQ-VAE steps then 8k policy steps (official: 20k of 250k)
  diffusion_kp  Diffusion Policy, keypoints + agent pos, DDIM 10 inference steps (= lerobot/diffusion_pusht_keypoints)
  diffusion_px  Diffusion Policy, pixels + agent pos, DDPM 100 (= lerobot/diffusion_pusht recipe)
Official references evaluated with the same eval protocol:
  lerobot/vqbet_pusht               250k steps, model card 57.0% / 500 episodes
  lerobot/diffusion_pusht_keypoints 2024-07 Hydra-format checkpoint (converted in-kernel), card 71.0% / 500 episodes
  (lerobot/diffusion_pusht was evaluated earlier: 66% / 50 episodes, card 65.4%)

Outputs: /kaggle/working/{train/<run>/train_log.json, eval/<run>/eval_info.json, ckpts/<run>/pretrained_model (only if
          model.safetensors < 200 MB), logs/*.log, env/*, summary.json, summary.md}
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
LEROBOT_COMMIT = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"  # chore: Bump to 0.3.3 (#1690), 2025-08-06
SRC = TMP / "lerobot"
VENV = TMP / "venv"
PY = str(VENV / "bin" / "python")
LEROBOT_TRAIN = str(VENV / "bin" / "lerobot-train")
LEROBOT_EVAL = str(VENV / "bin" / "lerobot-eval")

STEPS = 10_000
LOG_FREQ = 100  # lerobot prints "step:1K" (lossy), so the step is recovered from the line index
BATCH_TRAIN = 64
TRAIN_SEED = 1
EVAL_SEEDS = [1000, 2000, 3000]
N_EPISODES, BATCH_EVAL = 50, 50
BUDGET_S = 10 * 3600  # do not start new training runs after this (Kaggle hard limit: 12 h)
KEEP_CKPT_MAX_BYTES = 200_000_000
KP_ENV = "--env.type=pusht --env.obs_type=environment_state_agent_pos"
PX_ENV = "--env.type=pusht"
KP_NORM = '{"ENV":"MIN_MAX","STATE":"MIN_MAX","ACTION":"MIN_MAX"}'
MODELS = {  # run order = dict order (cheap first)
    "tdmpc_kp": dict(dataset="lerobot/pusht_keypoints", env=KP_ENV, norm=KP_NORM,
                     policy="--policy.type=tdmpc", label="TD-MPC, keypoints + agent pos, offline"),
    "vqbet_px": dict(dataset="lerobot/pusht", env=PX_ENV, norm=None,
                     policy="--policy.type=vqbet --policy.n_vqvae_training_steps=2000",
                     label="VQ-BeT, pixels + agent pos (2k VQ-VAE + 8k policy steps)"),
    "diffusion_kp": dict(dataset="lerobot/pusht_keypoints", env=KP_ENV, norm=KP_NORM,
                         policy="--policy.type=diffusion --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10",
                         label="Diffusion Policy, keypoints + agent pos, DDIM-10"),
    "diffusion_px": dict(dataset="lerobot/pusht", env=PX_ENV, norm=None,
                         policy="--policy.type=diffusion", label="Diffusion Policy, pixels + agent pos, DDPM-100"),
}
REFS = {  # official checkpoints, evaluated with the same protocol
    "ref_vqbet_pusht": dict(repo="lerobot/vqbet_pusht", env=PX_ENV, convert=None,
                            label="official lerobot/vqbet_pusht (250k steps; card 57.0% / 500 eps)"),
    "ref_diffusion_pusht_keypoints": dict(repo="lerobot/diffusion_pusht_keypoints", env=KP_ENV, convert="diffusion",
                                          label="official lerobot/diffusion_pusht_keypoints (Hydra ckpt converted; card 71.0% / 500 eps)"),
}

SUMMARY = {
    "meta": {
        "lerobot_commit": LEROBOT_COMMIT, "steps": STEPS, "batch_train": BATCH_TRAIN, "train_seed": TRAIN_SEED,
        "eval_seeds": EVAL_SEEDS, "n_episodes": N_EPISODES, "batch_eval": BATCH_EVAL, "models": MODELS, "refs": REFS,
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    },
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

# Hydra-era (2024) lerobot checkpoints carry config.yaml + an old config.json (input_shapes / input_normalization_modes);
# lerobot 0.3.x reads the draccus format (type / input_features / output_features / normalization_mapping). The
# model.safetensors keys did not change (verified against the file header: 156 tensors, 248.8M params, no mismatch).
CONVERT_OLD_DIFFUSION = r'''
import json, os, shutil, sys
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
repo, out = sys.argv[1], sys.argv[2]
old = json.load(open(hf_hub_download(repo, "config.json")))
TYPE = {"observation.image": "VISUAL", "observation.state": "STATE", "observation.environment_state": "ENV", "action": "ACTION"}
MODE = {"mean_std": "MEAN_STD", "min_max": "MIN_MAX", "identity": "IDENTITY"}
new = {"type": "diffusion", "n_obs_steps": old["n_obs_steps"], "normalization_mapping": {}, "input_features": {},
       "output_features": {}, "device": "cuda", "use_amp": False}
for k, shape in old["input_shapes"].items():
    new["input_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["input_normalization_modes"][k]]
for k, shape in old["output_shapes"].items():
    new["output_features"][k] = {"type": TYPE[k], "shape": shape}
    new["normalization_mapping"][TYPE[k]] = MODE[old["output_normalization_modes"][k]]
for k in ["horizon", "n_action_steps", "vision_backbone", "crop_shape", "crop_is_random", "pretrained_backbone_weights",
          "use_group_norm", "spatial_softmax_num_keypoints", "down_dims", "kernel_size", "n_groups", "diffusion_step_embed_dim",
          "use_film_scale_modulation", "noise_scheduler_type", "num_train_timesteps", "beta_schedule", "beta_start", "beta_end",
          "prediction_type", "clip_sample", "clip_sample_range", "num_inference_steps", "do_mask_loss_for_padding"]:
    if k in old:
        new[k] = old[k]
os.makedirs(out, exist_ok=True)
json.dump(new, open(f"{out}/config.json", "w"), indent=2)
shutil.copy(hf_hub_download(repo, "model.safetensors"), f"{out}/model.safetensors")
for extra in ("config.yaml", "README.md", "eval_info.json"):
    try:
        shutil.copy(hf_hub_download(repo, extra), f"{out}/{extra}.orig")
    except Exception:
        pass
cfg = PreTrainedConfig.from_pretrained(out)
pol = DiffusionPolicy.from_pretrained(out, config=cfg, strict=True)
print(f"CONVERT_OK {repo} -> {out}: {sum(p.numel() for p in pol.parameters()) / 1e6:.1f}M params, "
      f"inputs={list(cfg.input_features)} norm={cfg.normalization_mapping} scheduler={cfg.noise_scheduler_type}/{cfg.num_inference_steps}",
      flush=True)
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


def sh(cmd, log_name, env=None, timeout=None, tail=20):
    """bash -c cmd; stdout+stderr -> logs/<log_name>.log; prints the tail; returns returncode (-9 on timeout)."""
    log(f"$ {cmd}")
    e = dict(os.environ)
    e.update(RUN_ENV)
    e.update(env or {})
    path = LOGS / f"{log_name}.log"
    with open(path, "a") as f:
        f.write(f"\n===== [{time.strftime('%H:%M:%S')}] $ {cmd}\n")
        f.flush()
        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT, env=e,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = -9
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
    return str(path)


# ----------------------------------------------------------------------------- statistics
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def stats_from_eval_info(path):
    with open(path) as f:
        info = json.load(f)
    eps = info["per_episode"]
    n = len(eps)
    succ = [1.0 if e["success"] else 0.0 for e in eps]
    k = int(sum(succ))
    p = k / n if n else float("nan")
    lo_w, hi_w = wilson_ci(k, n)
    agg = info["aggregated"]
    return dict(n_episodes=n, n_success=k, pc_success=100 * p, ci95_wilson=[100 * lo_w, 100 * hi_w],
                avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"], eval_s=agg["eval_s"],
                successes=[bool(e["success"]) for e in eps])


def write_summary_md():
    meta = SUMMARY["meta"]
    lines = ["# Other LeRobot policies on PushT: 10k-step training + official references, Kaggle", "",
             f"- lerobot commit: `{meta['lerobot_commit']}` (v0.3.3), dataset revision v2.1",
             f"- device: {meta.get('device')}  |  GPU: {meta.get('gpu', {}).get('device_name')}  |  torch {meta.get('gpu', {}).get('torch')}",
             f"- training: {meta['steps']} steps, batch {meta['batch_train']}, train seed {meta['train_seed']}",
             f"- evaluation: seeds {meta['eval_seeds']}, n_episodes={meta['n_episodes']}, batch_size={meta['batch_eval']}",
             f"- started {meta['started_utc']} UTC, elapsed {meta.get('elapsed_s', 0) / 60:.1f} min", "",
             "## Training runs", "",
             "| run | policy | params | rc | loss @1k | loss @5k | final loss (mean last 1k steps) | train wall (min) | it/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, t in SUMMARY["train"].items():
        curve = t.get("curve") or []
        def at(step):
            pts = [c["loss"] for c in curve if step - 1000 < c["step"] <= step]
            return f"{sum(pts) / len(pts):.4f}" if pts else "-"
        its = f"{meta['steps'] / t['wall_s']:.1f}" if t.get("wall_s") and t.get("ok") else "-"
        lines.append(f"| {name} | {MODELS[name]['label']} | {t.get('params', '-')} | {t.get('rc')} | {at(1000)} | {at(5000)} | "
                     f"{at(STEPS)} | {t.get('wall_s', 0) / 60:.1f} | {its} |")
    lines += ["", "## Evaluations (50 episodes each)", "",
              "| run | policy | eval seed | ok | success | pc_success | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in SUMMARY["runs"].items():
        s = r.get("stats")
        if s:
            lines.append(f"| {name} | {r['policy']} | {r['seed']} | rc={r['rc']} | {s['n_success']}/{s['n_episodes']} | "
                         f"{s['pc_success']:.1f}% | [{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}] | "
                         f"{s['avg_sum_reward']:.2f} | {s['avg_max_reward']:.3f} | {s['eval_s']:.0f} |")
        else:
            lines.append(f"| {name} | {r['policy']} | {r['seed']} | rc={r.get('rc')} FAILED | - | - | - | - | - | {r.get('wall_s', 0):.0f} |")
    lines += ["", "## Across eval seeds", "",
              "| group | policy | seeds | mean pc_success | std across seeds (pts) | pooled k/N | pooled pc | Wilson 95% CI |",
              "|---|---|---|---|---|---|---|---|"]
    groups = {}
    for name, r in SUMMARY["runs"].items():
        if r.get("stats"):
            groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        ps = [x["stats"]["pc_success"] for x in rs]
        m = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - m) ** 2 for x in ps) / (len(ps) - 1)) if len(ps) > 1 else float("nan")
        k = sum(x["stats"]["n_success"] for x in rs)
        n = sum(x["stats"]["n_episodes"] for x in rs)
        lo, hi = wilson_ci(k, n)
        lines.append(f"| {g} | {rs[0]['policy']} | {[x['seed'] for x in rs]} | {m:.1f}% | {sd:.1f} | {k}/{n} | "
                     f"{100 * k / n:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] |")
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


def gpu_check():
    script = write_script("gpu_check.py", GPU_CHECK)
    sh(f"{PY} {script}", "02_gpu_check")
    raw = grep_log("02_gpu_check", "GPU_CHECK ")
    try:
        return json.loads(raw) if raw else {"gpu_ok": False, "error": "no GPU_CHECK line"}
    except Exception:
        return {"gpu_ok": False, "error": raw}


def step_install():
    sh("nvidia-smi || true; echo; python -V; nproc; free -g; df -h /kaggle/working /kaggle/tmp /tmp | cat", "00_env_info", tail=30)
    shutil.copy(LOGS / "00_env_info.log", ENV_DIR / "env_info.txt")
    uv = find_uv()
    assert sh(f"{uv} --version", "01_install") == 0, "uv not available"
    assert sh(f"{uv} venv --python 3.10 {VENV}", "01_install") == 0, "uv venv failed"
    assert sh(f"git clone -q {LEROBOT_REPO} {SRC} && cd {SRC} && git checkout -q {LEROBOT_COMMIT} && "
              f"git log -1 --format='%H %cd %s' --date=short", "01_install") == 0, "git clone/checkout failed"
    assert sh(f"{uv} pip install --python {PY} torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126",
              "01_install") == 0, "torch install failed"
    assert sh(f"{uv} pip install --python {PY} --no-deps torchcodec==0.4.0", "01_install") == 0, "torchcodec install failed"
    (TMP / "constraints.txt").write_text("torch==2.7.1\ntorchvision==0.22.1\ntorchcodec==0.4.0\n")
    assert sh(f'{uv} pip install --python {PY} -e "{SRC}[pusht]" -c {TMP}/constraints.txt', "01_install") == 0, \
        "lerobot install failed"
    gpu = gpu_check()
    SUMMARY["meta"]["gpu"] = gpu
    assert gpu.get("gpu_ok"), f"GPU not usable: {gpu}"
    SUMMARY["meta"]["device"] = "cuda"
    sh(f"{uv} pip freeze --python {PY} > {ENV_DIR}/requirements-freeze.txt; wc -l {ENV_DIR}/requirements-freeze.txt",
       "01_install", tail=3)
    save_summary()


LOG_RE = re.compile(r"step:(\S+).*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+|nan|inf).*?lr:([0-9.eE+-]+)")
PARAMS_RE = re.compile(r"num_learnable_params=(\d+)")


def parse_train_log(log_name):
    curve, params = [], None
    for line in (LOGS / f"{log_name}.log").read_text(errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            curve.append(dict(step=LOG_FREQ * (len(curve) + 1), step_str=m.group(1), loss=float(m.group(2)),
                              grad_norm=float(m.group(3)) if m.group(3) not in ("nan", "inf") else None, lr=float(m.group(4))))
        pm = PARAMS_RE.search(line)
        if pm:
            params = int(pm.group(1))
    return curve, params


def run_train(name):
    m = MODELS[name]
    out = TMP / "train" / name
    shutil.rmtree(out, ignore_errors=True)  # lerobot refuses to start if output_dir exists
    norm = f" '--policy.normalization_mapping={m['norm']}'" if m["norm"] else ""
    cmd = (f"{LEROBOT_TRAIN} --dataset.repo_id={m['dataset']} {m['env']} {m['policy']} --policy.push_to_hub=false "
           f"--policy.device=cuda{norm} --output_dir={out} --job_name={name} --steps={STEPS} --save_freq={STEPS} "
           f"--eval_freq=0 --log_freq={LOG_FREQ} --batch_size={BATCH_TRAIN} --num_workers=4 --seed={TRAIN_SEED} --wandb.enable=false")
    t = time.time()
    rc = sh(cmd, name, timeout=3 * 3600, tail=6)
    wall = round(time.time() - t, 1)
    ckpt = out / "checkpoints" / f"{STEPS:06d}" / "pretrained_model"
    rec = dict(rc=rc, wall_s=wall, cmd=cmd, dataset=m["dataset"], ckpt=str(ckpt))
    try:
        rec["curve"], params = parse_train_log(name)
        rec["params"] = f"{params / 1e6:.1f}M" if params else "-"
        (TRAIN_DIR / name).mkdir(parents=True, exist_ok=True)
        json.dump(dict(name=name, rc=rc, wall_s=wall, steps=STEPS, batch=BATCH_TRAIN, params=params, cmd=cmd, curve=rec["curve"]),
                  open(TRAIN_DIR / name / "train_log.json", "w"), indent=1)
    except Exception:
        rec["error"] = "parse: " + traceback.format_exc()[-400:]
    ok = (ckpt / "model.safetensors").exists()  # pretrained_model is written before the training-state save
    rec["ok"] = ok
    if ok and rc != 0:
        log(f"WARNING {name}: checkpoint exists but lerobot-train exited rc={rc}; continuing with the checkpoint")
    if ok:
        for f in ("config.json", "train_config.json"):
            if (ckpt / f).exists():
                shutil.copy(ckpt / f, ENV_DIR / f"{name}.{f}")
        size = (ckpt / "model.safetensors").stat().st_size
        if size < KEEP_CKPT_MAX_BYTES:
            shutil.copytree(ckpt, CKPT_OUT / name / "pretrained_model", dirs_exist_ok=True)
        else:
            log(f"{name}: model.safetensors is {size / 1e6:.0f} MB, not kept as a kernel output")
        if rec.get("curve"):
            log(f"TRAIN_OK {name}: {len(rec['curve'])} log points, final loss {rec['curve'][-1]['loss']:.4f}, "
                f"{wall / 60:.1f} min ({STEPS / wall:.1f} it/s)")
    else:
        rec["error"] = (rec.get("error", "") + " / ".join(log_tail(name, 12)))[:1500]
        log(f"TRAIN_FAILED {name}: rc={rc}")
    SUMMARY["train"][name] = rec
    save_summary()
    return str(ckpt) if ok else None


def run_eval(group, label, policy_path, env, eval_seed):
    name = f"{group}_eval{eval_seed}"
    out = EVAL_DIR / name
    if out.exists() and not (out / "eval_info.json").exists():
        shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_EVAL} --policy.path={policy_path} --policy.device=cuda {env} "
           f"--eval.n_episodes={N_EPISODES} --eval.batch_size={BATCH_EVAL} --seed={eval_seed} --output_dir={out}")
    t = time.time()
    rc = sh(cmd, name, timeout=3600, tail=6)
    rec = {"group": group, "policy": label, "policy_path": policy_path, "env": env, "seed": eval_seed, "rc": rc,
           "wall_s": round(time.time() - t, 1), "cmd": cmd}
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


def prepare_ref(name):
    r = REFS[name]
    if r["convert"] == "diffusion":
        out = TMP / "ckpts" / name
        script = write_script("convert_old_diffusion.py", CONVERT_OLD_DIFFUSION)
        rc = sh(f"{PY} {script} {r['repo']} {out}", f"convert_{name}", timeout=1800, tail=4)
        msg = grep_log(f"convert_{name}", "CONVERT_OK ")
        SUMMARY["meta"][f"{name}_convert"] = msg or f"FAILED rc={rc}"
        if not msg:
            return None
        shutil.copy(out / "config.json", ENV_DIR / f"{name}.config.json")
        return str(out)
    return r["repo"]


def over_budget():
    return time.time() - T0 > BUDGET_S


def main():
    for d in (TMP, LOGS, EVAL_DIR, TRAIN_DIR, ENV_DIR, CKPT_OUT):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    try:
        step_install()
    except Exception:
        SUMMARY["errors"].append("install: " + traceback.format_exc()[-1500:])
        log("INSTALL FAILED:\n" + traceback.format_exc())
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    for name, r in REFS.items():  # references first: cheap, and useful even if a training run fails
        try:
            path = prepare_ref(name)
            if path:
                for es in EVAL_SEEDS:
                    run_eval(name, r["label"], path, r["env"], es)
            else:
                SUMMARY["errors"].append(f"{name}: checkpoint not available")
        except Exception:
            SUMMARY["errors"].append(f"{name}: " + traceback.format_exc()[-800:])
            save_summary()

    for name, m in MODELS.items():
        if over_budget():
            SUMMARY["errors"].append(f"time budget exhausted before training {name}")
            break
        try:
            ckpt = run_train(name)
            if ckpt:
                for es in EVAL_SEEDS:
                    run_eval(name, f"{m['label']}, {STEPS} steps, train seed {TRAIN_SEED}", ckpt, m["env"], es)
        except Exception:
            SUMMARY["errors"].append(f"{name}: " + traceback.format_exc()[-800:])
            save_summary()

    save_summary()
    print("\n" + (WORK / "summary.md").read_text(), flush=True)
    log("KERNEL_DONE")


if __name__ == "__main__":
    main()
