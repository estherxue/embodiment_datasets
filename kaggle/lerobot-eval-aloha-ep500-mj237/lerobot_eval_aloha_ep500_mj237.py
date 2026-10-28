#!/usr/bin/env python3
"""Kaggle script kernel (variant): ALOHA sim insertion with --env.episode_length=500 instead of the lerobot default 400,
same simulator pins as the checkpoint era (gym-aloha==0.1.1, mujoco==2.3.7, dm-control==1.0.14, numpy<2).

Why: Hub discussion #1 on lerobot/act_aloha_sim_insertion_human points out that the insertion demos are 500 frames
(10 s) while the model card eval used episode_length=400 (the transfer-cube default); the reporter measured 23.2%
with 500 instead of 20.6%. This kernel measures the same thing with our pipeline (2 seeds x 50 episodes).

Original full pipeline docstring follows.

Everything runs inside a uv-managed Python 3.10 venv under /kaggle/tmp (NOT /kaggle/working, which is the
kernel output). Every step is wrapped so that partial results are always saved to /kaggle/working.

Pipeline
  1. uv venv 3.10 + lerobot @ b883328e (v0.3.3, 2025-08-06) with torch 2.7.1+cu126 (Kaggle's P100 is sm_60:
     cu128 wheels dropped Pascal, cu126 still ships it) and the [pusht,aloha] extras.
  2. GPU self-check -> fallback torch 2.6.0+cu124 -> fallback CPU.
  3. Render self-check for gym-aloha: MUJOCO_GL=egl -> osmesa (apt libosmesa6).
  4. Convert the Hydra-era checkpoints alexandersoare/act_pusht_keypoints and pepijn223/act-pusht to the
     draccus config.json format (model.safetensors keys are unchanged; config.json is rewritten).
  5. lerobot-eval, n_episodes=50, batch_size=50, seeds recorded:
       pusht_act_keypoints   alexandersoare/act_pusht_keypoints     seeds 1000/2000/3000  <- main ACT-on-PushT result
       aloha_insertion_act_ep500   lerobot/act_aloha_sim_insertion_human  seeds 1000/2000, episode_length=500
       pusht_act_pixels      arclabmit/pusht_act_model              seeds 1000/2000/3000  (community pixel ACT, weak)
       pusht_diffusion       lerobot/diffusion_pusht                seed 1000             (non-ACT reference)
       pusht_random          uniform random actions                 seed 1000             (floor)
       pusht_act_pepijn223   pepijn223/act-pusht (converted)        seed 1000             (optional)
  6. summary.json / summary.md: success rate, sample + Bernoulli variance, 95% CI, across-seed mean/std.

Outputs: /kaggle/working/{eval/<run>/eval_info.json, logs/*.log, env/*, summary.json, summary.md}
"""
import json
import math
import os
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
LOGS, EVAL_DIR, ENV_DIR, CKPTS = WORK / "logs", WORK / "eval", WORK / "env", TMP / "ckpts"

LEROBOT_REPO = "https://github.com/huggingface/lerobot"
LEROBOT_COMMIT = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"  # chore: Bump to 0.3.3 (#1690), 2025-08-06
SRC = TMP / "lerobot"
VENV = TMP / "venv"
PY = str(VENV / "bin" / "python")
LEROBOT_EVAL = str(VENV / "bin" / "lerobot-eval")
N_EPISODES, BATCH = 50, 50
SIM_PINS = "gym-aloha==0.1.1\nmujoco==2.3.7\ndm-control==1.0.14\nnumpy<2\n"  # checkpoint-era simulator
BUDGET_S = 10 * 3600  # do not start new runs after this (Kaggle hard limit: 12 h)

SUMMARY = {
    "meta": {
        "lerobot_commit": LEROBOT_COMMIT,
        "n_episodes": N_EPISODES,
        "batch_size": BATCH,
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    },
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
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
}

# ----------------------------------------------------------------------------- helper scripts (written to disk)
GPU_CHECK = r'''
import json, torch
info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
if info["cuda_available"]:
    info["device_name"] = torch.cuda.get_device_name(0)
    info["capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import torch.nn.functional as F
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        c = F.conv2d(torch.randn(2, 3, 64, 64, device="cuda"), torch.randn(8, 3, 3, 3, device="cuda")).sum().item()
        torch.cuda.synchronize()
        info["gpu_ok"] = bool(y == y and c == c)
    except Exception as e:  # e.g. "no kernel image is available for execution on the device"
        info["gpu_ok"] = False
        info["error"] = repr(e)[:300]
else:
    info["gpu_ok"] = False
print("GPU_CHECK " + json.dumps(info), flush=True)
'''

RENDER_CHECK = r'''
import os, sys, time
import gymnasium as gym
import gym_aloha  # noqa: F401
t = time.time()
env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, info = env.reset(seed=0)
img = obs["pixels"]["top"]
for _ in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
frame = env.render()
print("RENDER_OK", os.environ.get("MUJOCO_GL"), tuple(img.shape), str(img.dtype), round(float(img.mean()), 2),
      "render()", tuple(frame.shape), f"{time.time() - t:.1f}s", flush=True)
'''

PUSHT_CHECK = r'''
import gymnasium as gym
import gym_pusht  # noqa: F401
for obs_type in ("pixels_agent_pos", "environment_state_agent_pos"):
    env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode="rgb_array",
                   visualization_width=384, visualization_height=384, max_episode_steps=300)
    obs, _ = env.reset(seed=0)
    obs, r, te, tr, info = env.step(env.action_space.sample())
    print("PUSHT_OK", obs_type, {k: getattr(v, "shape", v) for k, v in obs.items()}, "is_success" in info, flush=True)
'''

CONVERT = r'''
"""Convert a Hydra-era (2024 / Jan-2025) lerobot ACT checkpoint to the draccus config.json format of lerobot v0.3.x.
model.safetensors key names did not change, so only config.json is rewritten.   usage: convert.py <repo_id> <out_dir>
"""
import json, os, shutil, sys
from huggingface_hub import hf_hub_download
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

repo, out = sys.argv[1], sys.argv[2]
old = json.load(open(hf_hub_download(repo, "config.json")))
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
shutil.copy(hf_hub_download(repo, "model.safetensors"), f"{out}/model.safetensors")
for extra in ("config.yaml", "README.md", "eval_info.json"):
    try:
        shutil.copy(hf_hub_download(repo, extra), f"{out}/{extra}.orig")
    except Exception:
        pass
cfg = PreTrainedConfig.from_pretrained(out)
pol = ACTPolicy.from_pretrained(out, config=cfg, strict=True)  # raises on any missing / unexpected key
print(f"CONVERT_OK {repo} -> {out}: {sum(p.numel() for p in pol.parameters()) / 1e6:.1f}M params; "
      f"inputs={list(cfg.input_features)} norm={cfg.normalization_mapping} chunk={cfg.chunk_size} "
      f"n_action_steps={cfg.n_action_steps}", flush=True)
print("normalize buffers finite:", all(torch.isfinite(b).all().item() for n, b in pol.named_buffers() if "normalize" in n))
'''

RANDOM_BASELINE = r'''
"""Uniform-random-action baseline on gym_pusht/PushT-v0 with the same seeds / episode length as lerobot-eval."""
import json, sys, time
import numpy as np, gymnasium as gym, gym_pusht  # noqa: F401
start_seed, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
               visualization_width=384, visualization_height=384, max_episode_steps=300, disable_env_checker=True)
rows, t = [], time.time()
for ep in range(n):
    seed = start_seed + ep
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    sum_r, max_r, succ, done = 0.0, 0.0, False, False
    while not done:
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        sum_r += r; max_r = max(max_r, r); succ = succ or bool(info.get("is_success", False)); done = term or trunc
    rows.append(dict(episode_ix=ep, sum_reward=sum_r, max_reward=max_r, success=succ, seed=seed))
agg = dict(avg_sum_reward=float(np.mean([x["sum_reward"] for x in rows])),
           avg_max_reward=float(np.mean([x["max_reward"] for x in rows])),
           pc_success=100 * float(np.mean([x["success"] for x in rows])),
           eval_s=time.time() - t, eval_ep_s=(time.time() - t) / n)
json.dump(dict(per_episode=rows, aggregated=agg), open(out, "w"), indent=2)
print("RANDOM_OK", agg, flush=True)
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
    return [line.split("\r")[-1][:width] for line in lines]  # drop tqdm carriage-return noise


def sh(cmd, log_name, env=None, timeout=None, cwd=None, tail=20):
    """bash -c cmd; stdout+stderr -> logs/<log_name>.log; prints the tail; returns returncode (-9 on timeout)."""
    log(f"$ {cmd}")
    e = dict(os.environ)
    e.update(RUN_ENV)
    e.update(env or {})
    path = LOGS / f"{log_name}.log"
    with open(path, "a") as f:
        f.write(f"\n===== [{time.strftime('%H:%M:%S')}] $ {cmd}\n")
        f.flush()
        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT,
                             env=e, cwd=cwd, start_new_session=True)
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
    """Return the JSON / text after the last line starting with `prefix` in a log."""
    for line in reversed(log_tail(log_name, 400, width=4000)):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def write_script(name, body):
    path = TMP / name
    path.write_text(body.lstrip())
    return str(path)


# ----------------------------------------------------------------------------- statistics (same as outputs/eval_stats.py)
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
    var_sample = sum((s - p) ** 2 for s in succ) / (n - 1) if n > 1 else float("nan")
    var_pop = p * (1 - p)
    se = math.sqrt(var_pop / n) if n else float("nan")
    lo_w, hi_w = wilson_ci(k, n)
    agg = info["aggregated"]
    seeds = [e.get("seed") for e in eps]
    return dict(
        n_episodes=n, n_success=k, pc_success=100 * p,
        success_var_sample=var_sample, success_std_sample=math.sqrt(var_sample) if var_sample == var_sample else float("nan"),
        success_var_bernoulli=var_pop, success_std_bernoulli=math.sqrt(var_pop), se_mean=se,
        ci95_normal=[100 * (p - 1.96 * se), 100 * (p + 1.96 * se)], ci95_wilson=[100 * lo_w, 100 * hi_w],
        avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"],
        eval_s=agg["eval_s"], eval_ep_s=agg.get("eval_ep_s"),
        episode_seeds=[min(s for s in seeds if s is not None), max(s for s in seeds if s is not None)] if any(s is not None for s in seeds) else None,
        reported_pc_success=agg["pc_success"],
    )


def write_summary_md():
    runs = SUMMARY["runs"]
    meta = SUMMARY["meta"]
    lines = ["# LeRobot ACT evaluation on Kaggle (ALOHA insertion, episode_length=500, checkpoint-era simulator: gym-aloha 0.1.1 / mujoco 2.3.7)", "",
             f"- lerobot commit: `{meta['lerobot_commit']}` (v0.3.3)",
             f"- device: {meta.get('device')}  |  GPU: {meta.get('gpu', {}).get('device_name')}  |  torch {meta.get('gpu', {}).get('torch')}",
             f"- MUJOCO_GL: {meta.get('mujoco_gl')}  |  n_episodes={meta['n_episodes']} batch_size={meta['batch_size']}",
             f"- started {meta['started_utc']} UTC, elapsed {meta.get('elapsed_s', 0) / 60:.1f} min", "",
             "| run | policy | env / obs | seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        s = r.get("stats")
        if s:
            lines.append(f"| {name} | {r['policy']} | {r['env']} | {r['seed']} | rc={r['rc']} | {s['n_success']}/{s['n_episodes']} | "
                         f"{s['pc_success']:.1f}% | {s['success_var_sample']:.4f} | {s['success_var_bernoulli']:.4f} | "
                         f"[{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}] | {s['avg_sum_reward']:.2f} | {s['avg_max_reward']:.3f} | {s['eval_s']:.0f} |")
        else:
            lines.append(f"| {name} | {r['policy']} | {r['env']} | {r['seed']} | rc={r.get('rc')} FAILED | - | - | - | - | - | - | - | {r.get('wall_s', 0):.0f} |")
    lines += ["", "## Across seeds (same policy, 50 episodes per seed)", "",
              "| group | seeds | mean pc_success | std across seeds (pts) | values |", "|---|---|---|---|---|"]
    groups = {}
    for name, r in runs.items():
        if r.get("stats"):
            groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        ps = [x["stats"]["pc_success"] for x in rs]
        m = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - m) ** 2 for x in ps) / (len(ps) - 1)) if len(ps) > 1 else float("nan")
        lines.append(f"| {g} | {[x['seed'] for x in rs]} | {m:.1f}% | {sd:.1f} | {[round(x, 1) for x in ps]} |")
    if SUMMARY["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e[:500]}" for e in SUMMARY["errors"]]
    (WORK / "summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- steps
def step_env_info():
    sh("nvidia-smi || true; echo; python -V; echo; head -3 /etc/os-release; nproc; free -g; "
       "df -h /kaggle/working /tmp | cat; echo; env | grep -E '^(CUDA|NVIDIA|KAGGLE)' | sort",
       "00_env_info", tail=40)
    shutil.copy(LOGS / "00_env_info.log", ENV_DIR / "env_info.txt")


def find_uv():
    uv = shutil.which("uv")
    if uv:
        return uv
    sh(f"{sys.executable} -m pip install -q uv", "01_install")
    uv = shutil.which("uv")
    return uv or f"{sys.executable} -m uv"


def install_torch(uv, torch_v, tv_v, tc_v, cu):
    rc = sh(f"{uv} pip install --python {PY} --reinstall-package torch --reinstall-package torchvision "
            f"torch=={torch_v} torchvision=={tv_v} --index-url https://download.pytorch.org/whl/{cu}", "01_install")
    rc2 = sh(f"{uv} pip install --python {PY} --no-deps torchcodec=={tc_v}", "01_install")
    (TMP / "constraints.txt").write_text(f"torch=={torch_v}\ntorchvision=={tv_v}\ntorchcodec=={tc_v}\n" + SIM_PINS)
    return rc == 0 and rc2 == 0


def gpu_check():
    script = write_script("gpu_check.py", GPU_CHECK)
    sh(f"{PY} {script}", "02_gpu_check")
    raw = grep_log("02_gpu_check", "GPU_CHECK ")
    try:
        return json.loads(raw) if raw else {"gpu_ok": False, "error": "no GPU_CHECK line"}
    except Exception:
        return {"gpu_ok": False, "error": raw}


def step_install():
    uv = find_uv()
    assert sh(f"{uv} --version", "01_install") == 0, "uv not available"
    assert sh(f"{uv} venv --python 3.10 {VENV}", "01_install") == 0, "uv venv failed"
    assert sh(f"git clone -q {LEROBOT_REPO} {SRC} && cd {SRC} && git checkout -q {LEROBOT_COMMIT} && "
              f"git log -1 --format='%H %cd %s' --date=short", "01_install") == 0, "git clone/checkout failed"
    assert install_torch(uv, "2.7.1", "0.22.1", "0.4.0", "cu126"), "torch install failed"
    assert sh(f'{uv} pip install --python {PY} -e "{SRC}[pusht,aloha]" -c {TMP}/constraints.txt', "01_install") == 0, \
        "lerobot install failed"
    assert sh(f'{uv} pip install --python {PY} "gym-aloha==0.1.1" "mujoco==2.3.7" "dm-control==1.0.14" "numpy<2" -c {TMP}/constraints.txt',
              "01_install") == 0, "sim pin install failed"
    sh(f"{PY} -c 'import gym_aloha, mujoco, dm_control, numpy; print(\"SIM_VERSIONS\", mujoco.__version__, dm_control.__version__, numpy.__version__)'", "01_install")
    sh(f"{PY} -c 'import lerobot, torch, torchvision, numpy; print(\"VERSIONS\", lerobot.__version__, torch.__version__, "
       f"torchvision.__version__, numpy.__version__)'", "01_install")

    gpu = gpu_check()
    SUMMARY["meta"]["gpu"] = gpu
    if not gpu.get("gpu_ok"):
        log(f"GPU check failed with torch 2.7.1+cu126: {gpu}. Trying torch 2.6.0+cu124 ...")
        if install_torch(uv, "2.6.0", "0.21.0", "0.2.1", "cu124"):
            gpu = gpu_check()
            SUMMARY["meta"]["gpu_fallback_2_6"] = gpu
    device = "cuda" if gpu.get("gpu_ok") else "cpu"
    SUMMARY["meta"]["device"] = device
    log(f"DEVICE = {device}  gpu={gpu}")

    sh(f"{uv} pip freeze --python {PY} > {ENV_DIR}/requirements-freeze.txt; wc -l {ENV_DIR}/requirements-freeze.txt",
       "01_install", tail=3)
    save_summary()
    return device


def step_render_check():
    """gym-aloha needs an OpenGL context. Try EGL (GPU), then OSMesa (CPU) after apt-installing the libs."""
    sh("apt-get update -qq > /dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "libegl1 libgl1 libglvnd0 libosmesa6 libglfw3 > /dev/null 2>&1; ls /usr/lib/x86_64-linux-gnu/ | grep -i -E 'EGL|OSMesa' | head",
       "03_render_check", tail=8)
    script = write_script("render_check.py", RENDER_CHECK)
    for gl in ("egl", "osmesa"):
        rc = sh(f"{PY} {script}", "03_render_check", env={"MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl}, timeout=600)
        ok = rc == 0 and grep_log("03_render_check", "RENDER_OK") is not None
        if ok:
            RUN_ENV["MUJOCO_GL"] = gl
            RUN_ENV["PYOPENGL_PLATFORM"] = gl
            SUMMARY["meta"]["mujoco_gl"] = gl
            SUMMARY["meta"]["render_check"] = grep_log("03_render_check", "RENDER_OK")
            log(f"ALOHA rendering OK with MUJOCO_GL={gl}")
            return True
    SUMMARY["meta"]["mujoco_gl"] = None
    SUMMARY["errors"].append("ALOHA render check failed for egl and osmesa: " + " / ".join(log_tail("03_render_check", 6)))
    return False


def step_pusht_check():
    script = write_script("pusht_check.py", PUSHT_CHECK)
    rc = sh(f"{PY} {script}", "03_pusht_check", timeout=600)
    SUMMARY["meta"]["pusht_check_rc"] = rc
    return rc == 0


def step_convert(repo, out_name):
    script = write_script("convert_old_act_ckpt.py", CONVERT)
    out = CKPTS / out_name
    rc = sh(f"{PY} {script} {repo} {out}", f"04_convert_{out_name}", timeout=1200)
    ok = rc == 0 and (out / "model.safetensors").exists()
    if ok:
        shutil.copy(out / "config.json", ENV_DIR / f"{out_name}.converted_config.json")
    else:
        SUMMARY["errors"].append(f"convert {repo} failed: " + " / ".join(log_tail(f"04_convert_{out_name}", 8)))
    return str(out) if ok else None


def run_eval(group, name, policy_label, policy_path, env_args, seed, device, timeout=3600, batch=BATCH):
    out = EVAL_DIR / name
    if out.exists() and not (out / "eval_info.json").exists():  # stale partial output from a failed attempt
        shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_EVAL} --policy.path={policy_path} --policy.device={device} {env_args} "
           f"--eval.n_episodes={N_EPISODES} --eval.batch_size={batch} --seed={seed} --output_dir={out}")
    t = time.time()
    rc = sh(cmd, name, timeout=timeout, tail=8)
    rec = {"group": group, "policy": policy_label, "policy_path": str(policy_path), "env": env_args, "seed": seed,
           "device": device, "batch_size": batch, "mujoco_gl": RUN_ENV.get("MUJOCO_GL"), "rc": rc,
           "wall_s": round(time.time() - t, 1), "cmd": cmd}
    info = out / "eval_info.json"
    if info.exists():
        try:
            rec["stats"] = stats_from_eval_info(info)
            s = rec["stats"]
            log(f"RESULT {name}: {s['n_success']}/{s['n_episodes']} = {s['pc_success']:.1f}%  "
                f"CI95 [{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}]  avg_max_reward={s['avg_max_reward']:.3f}  "
                f"eval_s={s['eval_s']:.0f}")
        except Exception:
            rec["error"] = "stats: " + traceback.format_exc()[-600:]
    else:
        rec["error"] = " / ".join(log_tail(name, 12))
        log(f"FAILED {name}: no eval_info.json (rc={rc})")
    SUMMARY["runs"][name] = rec
    save_summary()
    return rec


def over_budget():
    return time.time() - T0 > BUDGET_S


def main():
    for d in (TMP, LOGS, EVAL_DIR, ENV_DIR, CKPTS):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    device = "cpu"
    try:
        step_env_info()
        device = step_install()
    except Exception:
        SUMMARY["errors"].append("install: " + traceback.format_exc()[-1500:])
        log("INSTALL FAILED:\n" + traceback.format_exc())
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    aloha_ok = False
    try:
        aloha_ok = step_render_check()
    except Exception:
        SUMMARY["errors"].append("checks: " + traceback.format_exc()[-1000:])
    save_summary()

    kp_dir = pep_dir = None  # (PushT conversions skipped in this ALOHA-only variant)

    PUSHT_KP = "--env.type=pusht --env.obs_type=environment_state_agent_pos"
    PUSHT_PX = "--env.type=pusht"
    ALOHA = "--env.type=aloha --env.task=AlohaInsertion-v0 --env.episode_length=500"

    def aloha_run(seed):
        if not aloha_ok:
            SUMMARY["runs"][f"aloha_insertion_act_ep500_seed{seed}"] = {
                "group": "aloha_insertion_act_ep500", "policy": "lerobot/act_aloha_sim_insertion_human", "env": ALOHA,
                "seed": seed, "rc": None, "error": "skipped: render check failed"}
            return
        rec = run_eval("aloha_insertion_act_ep500", f"aloha_insertion_act_ep500_seed{seed}", "lerobot/act_aloha_sim_insertion_human",
                       "lerobot/act_aloha_sim_insertion_human", ALOHA, seed, device, timeout=9000)
        if "stats" not in rec and not over_budget():  # retry once with fewer parallel envs (GL contexts / memory)
            log("ALOHA run failed, retrying once with batch_size=10")
            run_eval("aloha_insertion_act_ep500", f"aloha_insertion_act_ep500_seed{seed}", "lerobot/act_aloha_sim_insertion_human",
                     "lerobot/act_aloha_sim_insertion_human", ALOHA, seed, device, timeout=9000, batch=10)

    def kp_run(seed):
        if kp_dir:
            run_eval("pusht_act_keypoints", f"pusht_act_keypoints_seed{seed}", "alexandersoare/act_pusht_keypoints (converted)",
                     kp_dir, PUSHT_KP, seed, device, timeout=2400)

    def px_run(seed):
        run_eval("pusht_act_pixels", f"pusht_act_pixels_seed{seed}", "arclabmit/pusht_act_model",
                 "arclabmit/pusht_act_model", PUSHT_PX, seed, device, timeout=2400)

    plan = [
        lambda: aloha_run(1000),
        lambda: aloha_run(2000),
    ]
    for i, step in enumerate(plan):
        if over_budget():
            SUMMARY["errors"].append(f"time budget exhausted before plan step {i}")
            break
        try:
            step()
        except Exception:
            SUMMARY["errors"].append(f"plan step {i}: " + traceback.format_exc()[-800:])
            save_summary()

    save_summary()
    print("\n" + (WORK / "summary.md").read_text(), flush=True)
    log("KERNEL_DONE")


if __name__ == "__main__":
    main()
