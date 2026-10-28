#!/usr/bin/env python3
"""Kaggle script kernel: short ACT training runs on PushT with LeRobot (10k steps each) to exercise the training
loop, then evaluate every checkpoint with 3 eval seeds x 50 episodes so the variance can be decomposed.

lerobot @ b883328e (v0.3.3), datasets at Hub revision v2.1 (the lerobot 0.3.3 default; main is v3.0 now).

Variants
  kp : dataset lerobot/pusht_keypoints, obs = 16-D T-block keypoints + 2-D agent pos. Same recipe as
       alexandersoare/act_pusht_keypoints (chunk 16 / n_action_steps 16 / 4 decoder layers / min-max norm /
       batch 64 / lr 3e-5), which that author trained for 100k steps -> 64% success.
  px : dataset lerobot/pusht, obs = 96x96 image + 2-D agent pos, same ACT hyper-parameters, resnet18 backbone
       (ImageNet init), image mean/std norm, state/action min-max.
Training seeds 1/2/3 per variant, 10k steps, batch 64, checkpoint at step 10k, no in-training eval.
Evaluation: lerobot-eval, seeds 1000/2000/3000, 50 episodes each (env seed = seed + episode index), batch 50,
plus one repeated eval with the same seed (kp, train seed 1) to show policy + simulator are deterministic.

Outputs: /kaggle/working/{train/<run>/train_log.json, eval/<run>/eval_info.json,
          ckpts/<variant>_train1/pretrained_model (seed-1 checkpoints only), logs/*.log, env/*, summary.json, summary.md}
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
LOG_FREQ = 100  # lerobot prints "step:1K" (format_big_number, lossy), so the step is recovered from the line index
BATCH_TRAIN = 64
TRAIN_SEEDS = [1, 2, 3]
EVAL_SEEDS = [1000, 2000, 3000]
N_EPISODES, BATCH_EVAL = 50, 50
BUDGET_S = 9 * 3600  # do not start new training runs after this (Kaggle hard limit: 12 h)
POLICY_ARGS = ("--policy.type=act --policy.push_to_hub=false --policy.chunk_size=16 --policy.n_action_steps=16 "
               "--policy.n_decoder_layers=4 --policy.optimizer_lr=3e-5")
VARIANTS = {
    "kp": dict(dataset="lerobot/pusht_keypoints", env="--env.type=pusht --env.obs_type=environment_state_agent_pos",
               norm='{"ENV":"MIN_MAX","STATE":"MIN_MAX","ACTION":"MIN_MAX"}',
               label="ACT keypoints (16-D keypoints + agent pos)"),
    "px": dict(dataset="lerobot/pusht", env="--env.type=pusht",
               norm='{"VISUAL":"MEAN_STD","STATE":"MIN_MAX","ACTION":"MIN_MAX"}',
               label="ACT pixels (96x96 image + agent pos, resnet18)"),
}

SUMMARY = {
    "meta": {
        "lerobot_commit": LEROBOT_COMMIT,
        "steps": STEPS, "batch_train": BATCH_TRAIN, "train_seeds": TRAIN_SEEDS, "eval_seeds": EVAL_SEEDS,
        "n_episodes": N_EPISODES, "batch_eval": BATCH_EVAL, "policy_args": POLICY_ARGS,
        "variants": VARIANTS,
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
    info["capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import torch.nn.functional as F
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        c = F.conv2d(torch.randn(2, 3, 64, 64, device="cuda"), torch.randn(8, 3, 3, 3, device="cuda")).sum().item()
        torch.cuda.synchronize()
        info["gpu_ok"] = bool(y == y and c == c)
    except Exception as e:
        info["gpu_ok"] = False
        info["error"] = repr(e)[:300]
else:
    info["gpu_ok"] = False
print("GPU_CHECK " + json.dumps(info), flush=True)
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
        success_var_sample=var_sample, success_var_bernoulli=var_pop, se_mean=se,
        ci95_wilson=[100 * lo_w, 100 * hi_w],
        avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"],
        eval_s=agg["eval_s"], eval_ep_s=agg.get("eval_ep_s"),
        episode_seeds=[min(seeds), max(seeds)] if all(s is not None for s in seeds) else None,
        successes=[bool(e["success"]) for e in eps],
    )


def write_summary_md():
    meta = SUMMARY["meta"]
    lines = ["# Short ACT training on PushT (LeRobot) + multi-seed evaluation, Kaggle", "",
             f"- lerobot commit: `{meta['lerobot_commit']}` (v0.3.3), dataset revision v2.1",
             f"- device: {meta.get('device')}  |  GPU: {meta.get('gpu', {}).get('device_name')}  |  torch {meta.get('gpu', {}).get('torch')}",
             f"- training: {meta['steps']} steps, batch {meta['batch_train']}, seeds {meta['train_seeds']}  |  policy args: `{meta['policy_args']}`",
             f"- evaluation: seeds {meta['eval_seeds']}, n_episodes={meta['n_episodes']}, batch_size={meta['batch_eval']}",
             f"- started {meta['started_utc']} UTC, elapsed {meta.get('elapsed_s', 0) / 60:.1f} min", "",
             "## Training runs", "",
             "| run | variant | train seed | rc | final loss | min loss | train wall (min) | it/s |",
             "|---|---|---|---|---|---|---|---|"]
    for name, t in SUMMARY["train"].items():
        curve = t.get("curve") or []
        losses = [c["loss"] for c in curve if c.get("loss") is not None]
        final = f"{losses[-1]:.4f}" if losses else "-"
        mn = f"{min(losses):.4f}" if losses else "-"
        its = f"{meta['steps'] / t['wall_s']:.1f}" if t.get("wall_s") and t.get("rc") == 0 else "-"
        lines.append(f"| {name} | {t['variant']} | {t['seed']} | {t.get('rc')} | {final} | {mn} | {t.get('wall_s', 0) / 60:.1f} | {its} |")
    lines += ["", "## Evaluations (50 episodes each)", "",
              "| run | policy | eval seed | ok | success | pc_success | var (sample) | var (Bernoulli) | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in SUMMARY["runs"].items():
        s = r.get("stats")
        if s:
            lines.append(f"| {name} | {r['policy']} | {r['seed']} | rc={r['rc']} | {s['n_success']}/{s['n_episodes']} | "
                         f"{s['pc_success']:.1f}% | {s['success_var_sample']:.4f} | {s['success_var_bernoulli']:.4f} | "
                         f"[{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}] | {s['avg_sum_reward']:.2f} | {s['avg_max_reward']:.3f} | {s['eval_s']:.0f} |")
        else:
            lines.append(f"| {name} | {r['policy']} | {r['seed']} | rc={r.get('rc')} FAILED | - | - | - | - | - | - | - | {r.get('wall_s', 0):.0f} |")
    lines += ["", "## Across eval seeds (same checkpoint)", "",
              "| checkpoint | eval seeds | mean pc_success | std across eval seeds (pts) | values |", "|---|---|---|---|---|"]
    groups = {}
    for name, r in SUMMARY["runs"].items():
        if r.get("stats") and not r.get("repeat"):
            groups.setdefault(r["group"], []).append(r)
    model_means = {}
    for g, rs in groups.items():
        ps = [x["stats"]["pc_success"] for x in rs]
        m = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - m) ** 2 for x in ps) / (len(ps) - 1)) if len(ps) > 1 else float("nan")
        model_means.setdefault(rs[0]["variant"], []).append(m)
        lines.append(f"| {g} | {[x['seed'] for x in rs]} | {m:.1f}% | {sd:.1f} | {[round(x, 1) for x in ps]} |")
    lines += ["", "## Across training seeds (mean over eval seeds per checkpoint)", "",
              "| variant | checkpoints | mean | std across training seeds (pts) | per-checkpoint means |", "|---|---|---|---|---|"]
    for v, ms in model_means.items():
        m = sum(ms) / len(ms)
        sd = math.sqrt(sum((x - m) ** 2 for x in ms) / (len(ms) - 1)) if len(ms) > 1 else float("nan")
        lines.append(f"| {v} | {len(ms)} | {m:.1f}% | {sd:.1f} | {[round(x, 1) for x in ms]} |")
    reps = [(n, r) for n, r in SUMMARY["runs"].items() if r.get("repeat") and r.get("stats")]
    if reps:
        lines += ["", "## Determinism check (same checkpoint, same eval seed, run twice)", ""]
        for n, r in reps:
            orig = SUMMARY["runs"].get(r["repeat"], {}).get("stats")
            same = orig and orig["successes"] == r["stats"]["successes"]
            lines.append(f"- {n} vs {r['repeat']}: {r['stats']['n_success']}/{r['stats']['n_episodes']} vs "
                         f"{orig['n_success'] if orig else '?'}/{orig['n_episodes'] if orig else '?'}, "
                         f"per-episode outcomes identical: {same}")
    if SUMMARY["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e[:500]}" for e in SUMMARY["errors"]]
    (WORK / "summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- steps
def step_env_info():
    sh("nvidia-smi || true; echo; python -V; echo; head -3 /etc/os-release; nproc; free -g; "
       "df -h /kaggle/working /tmp | cat", "00_env_info", tail=30)
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
    (TMP / "constraints.txt").write_text(f"torch=={torch_v}\ntorchvision=={tv_v}\ntorchcodec=={tc_v}\n")
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
    assert sh(f'{uv} pip install --python {PY} -e "{SRC}[pusht]" -c {TMP}/constraints.txt', "01_install") == 0, \
        "lerobot install failed"
    sh(f"{PY} -c 'import lerobot, torch, torchvision, numpy, torchcodec; print(\"VERSIONS\", lerobot.__version__, torch.__version__, "
       f"torchvision.__version__, numpy.__version__, torchcodec.__version__)'", "01_install")
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
    script = write_script("pusht_check.py", PUSHT_CHECK)
    SUMMARY["meta"]["pusht_check_rc"] = sh(f"{PY} {script}", "03_pusht_check", timeout=600)
    save_summary()
    return device


LOG_RE = re.compile(r"step:(\S+).*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+|nan|inf).*?lr:([0-9.eE+-]+)")
EPOCH_RE = re.compile(r"epch:([0-9.]+)")


def parse_train_log(log_name):
    curve = []
    for line in (LOGS / f"{log_name}.log").read_text(errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            ep = EPOCH_RE.search(line)
            curve.append(dict(step=LOG_FREQ * (len(curve) + 1), step_str=m.group(1), loss=float(m.group(2)),
                              grad_norm=float(m.group(3)) if m.group(3) not in ("nan", "inf") else None,
                              lr=float(m.group(4)), epoch=float(ep.group(1)) if ep else None))
    return curve


def run_train(variant, seed, device):
    v = VARIANTS[variant]
    name = f"{variant}_train{seed}"
    out = TMP / "train" / name
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)  # lerobot refuses to start if output_dir exists
    cmd = (f"{LEROBOT_TRAIN} --dataset.repo_id={v['dataset']} {v['env']} {POLICY_ARGS} --policy.device={device} "
           f"'--policy.normalization_mapping={v['norm']}' --output_dir={out} --job_name={name} --steps={STEPS} "
           f"--save_freq={STEPS} --eval_freq=0 --log_freq={LOG_FREQ} --batch_size={BATCH_TRAIN} --num_workers=4 "
           f"--seed={seed} --wandb.enable=false")
    t = time.time()
    rc = sh(cmd, name, timeout=4 * 3600, tail=6)
    wall = round(time.time() - t, 1)
    ckpt = out / "checkpoints" / f"{STEPS:06d}" / "pretrained_model"
    rec = dict(variant=variant, seed=seed, rc=rc, wall_s=wall, cmd=cmd, dataset=v["dataset"], ckpt=str(ckpt))
    try:
        rec["curve"] = parse_train_log(name)
        (TRAIN_DIR / name).mkdir(parents=True, exist_ok=True)
        json.dump(dict(variant=variant, seed=seed, rc=rc, wall_s=wall, steps=STEPS, batch=BATCH_TRAIN, cmd=cmd,
                       curve=rec["curve"]), open(TRAIN_DIR / name / "train_log.json", "w"), indent=1)
    except Exception:
        rec["error"] = "parse: " + traceback.format_exc()[-400:]
    ok = (ckpt / "model.safetensors").exists()  # pretrained_model is written before the training-state/rng save
    if ok and rc != 0:
        log(f"WARNING {name}: checkpoint exists but lerobot-train exited rc={rc}; continuing with the checkpoint")
    if ok:
        for f in ("config.json", "train_config.json"):
            if (ckpt / f).exists():
                shutil.copy(ckpt / f, ENV_DIR / f"{name}.{f}")
        if seed == TRAIN_SEEDS[0]:  # keep one full checkpoint per variant as a kernel output
            dst = CKPT_OUT / name / "pretrained_model"
            shutil.copytree(ckpt, dst, dirs_exist_ok=True)
        if rec.get("curve"):
            log(f"TRAIN_OK {name}: {len(rec['curve'])} log points, final loss {rec['curve'][-1]['loss']:.4f}, "
                f"{wall / 60:.1f} min ({STEPS / wall:.1f} it/s)")
    else:
        rec["error"] = (rec.get("error", "") + " / ".join(log_tail(name, 12)))[:1500]
        log(f"TRAIN_FAILED {name}: rc={rc}")
    SUMMARY["train"][name] = rec
    save_summary()
    return str(ckpt) if ok else None


def run_eval(variant, train_seed, ckpt, eval_seed, device, repeat_of=None):
    v = VARIANTS[variant]
    group = f"{variant}_train{train_seed}"
    name = f"{group}_eval{eval_seed}" + ("_repeat" if repeat_of else "")
    out = EVAL_DIR / name
    if out.exists() and not (out / "eval_info.json").exists():
        shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_EVAL} --policy.path={ckpt} --policy.device={device} {v['env']} "
           f"--eval.n_episodes={N_EPISODES} --eval.batch_size={BATCH_EVAL} --seed={eval_seed} --output_dir={out}")
    t = time.time()
    rc = sh(cmd, name, timeout=2400, tail=6)
    rec = {"group": group, "variant": variant, "train_seed": train_seed, "policy": f"{v['label']}, train seed {train_seed}, {STEPS} steps",
           "policy_path": ckpt, "env": v["env"], "seed": eval_seed, "device": device, "rc": rc,
           "wall_s": round(time.time() - t, 1), "cmd": cmd}
    if repeat_of:
        rec["repeat"] = repeat_of
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
    # videos of the first episodes are useful but large: keep only episode 0 per run
    vids = out / "videos"
    if vids.exists():
        for p in vids.iterdir():
            if p.name != "eval_episode_0.mp4":
                p.unlink()
    SUMMARY["runs"][name] = rec
    save_summary()
    return rec


def over_budget():
    return time.time() - T0 > BUDGET_S


def main():
    for d in (TMP, LOGS, EVAL_DIR, TRAIN_DIR, ENV_DIR, CKPT_OUT):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    try:
        step_env_info()
        device = step_install()
    except Exception:
        SUMMARY["errors"].append("install: " + traceback.format_exc()[-1500:])
        log("INSTALL FAILED:\n" + traceback.format_exc())
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    for variant in ("kp", "px"):
        for seed in TRAIN_SEEDS:
            if over_budget():
                SUMMARY["errors"].append(f"time budget exhausted before training {variant} seed {seed}")
                break
            try:
                ckpt = run_train(variant, seed, device)
                if ckpt:
                    for es in EVAL_SEEDS:
                        run_eval(variant, seed, ckpt, es, device)
                    if variant == "kp" and seed == TRAIN_SEEDS[0]:
                        run_eval(variant, seed, ckpt, EVAL_SEEDS[0], device, repeat_of=f"kp_train{seed}_eval{EVAL_SEEDS[0]}")
            except Exception:
                SUMMARY["errors"].append(f"{variant} seed {seed}: " + traceback.format_exc()[-800:])
                save_summary()

    save_summary()
    print("\n" + (WORK / "summary.md").read_text(), flush=True)
    log("KERNEL_DONE")


if __name__ == "__main__":
    main()
