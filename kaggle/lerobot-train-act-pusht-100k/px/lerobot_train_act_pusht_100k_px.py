#!/usr/bin/env python3
"""Kaggle script kernel: continue the 10k-step ACT PushT checkpoint (train seed 1, from kernel
cindyxue1122/lerobot-train-act-pusht, mounted under /kaggle/input) for 90k more steps = 100k total, one variant per
kernel (VARIANT below; build.py writes the kp and px copies).

Warm start = `lerobot-train --policy.path=<10k pretrained_model>`: weights, normalization stats and the policy config
come from the checkpoint; AdamW moments start fresh (the 10k kernel kept pretrained_model only, not training_state).
ACT has no lr scheduler, so that reset is the only difference from a true resume. Dataset order restarts with seed 1.

Every 10k steps lerobot-train writes a checkpoint; this script evaluates each new one while training continues on the
same GPU (3 eval seeds x 50 episodes). The last checkpoint (100k total, or whatever exists at the time budget) gets
10 seeds x 50 = 500 episodes, the N of alexandersoare/act_pusht_keypoints' eval_info.json (64.0% = 320/500).

Outputs: /kaggle/working/{train/<variant>_cont/train_log.json, eval/<run>/eval_info.json,
          ckpts/<variant>_100k/pretrained_model, logs/*.log, env/*, summary.json, summary.md}
"""
VARIANT = "px"  # written by build.py

import glob
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

OFFSET = 10_000          # steps already done by the input checkpoint
STEPS = 90_000           # steps in this run; total = OFFSET + step
SAVE_FREQ = 10_000
LOG_FREQ = 100           # lerobot prints "step:1K" (lossy), so the step is recovered from the line index
BATCH_TRAIN = 64
TRAIN_SEED = 1
EVAL_SEEDS = [1000, 2000, 3000]
FINAL_EVAL_SEEDS = [1000 * i for i in range(1, 11)]
N_EPISODES, BATCH_EVAL = 50, 50
TRAIN_BUDGET_S = 10.0 * 3600   # kill training after this (Kaggle hard limit 12 h); final 500-episode eval still fits
POLL_S = 30
REF = {"kp": (320, 500, "alexandersoare/act_pusht_keypoints eval_info.json, 100k steps, 64.0%"), "px": None}
VARIANTS = {
    "kp": dict(dataset="lerobot/pusht_keypoints", env="--env.type=pusht --env.obs_type=environment_state_agent_pos",
               label="ACT keypoints (16-D keypoints + agent pos)"),
    "px": dict(dataset="lerobot/pusht", env="--env.type=pusht",
               label="ACT pixels (96x96 image + agent pos, resnet18)"),
}
V = VARIANTS[VARIANT]
RUN = f"{VARIANT}_cont"

SUMMARY = {
    "meta": {
        "variant": VARIANT, "lerobot_commit": LEROBOT_COMMIT, "offset": OFFSET, "steps": STEPS, "save_freq": SAVE_FREQ,
        "batch_train": BATCH_TRAIN, "train_seed": TRAIN_SEED, "eval_seeds": EVAL_SEEDS, "final_eval_seeds": FINAL_EVAL_SEEDS,
        "n_episodes": N_EPISODES, "batch_eval": BATCH_EVAL, "reference": REF[VARIANT],
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
    succ = [1.0 if e["success"] else 0.0 for e in eps]
    k = int(sum(succ))
    p = k / n if n else float("nan")
    lo_w, hi_w = wilson_ci(k, n)
    agg = info["aggregated"]
    return dict(n_episodes=n, n_success=k, pc_success=100 * p, ci95_wilson=[100 * lo_w, 100 * hi_w],
                avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"], eval_s=agg["eval_s"],
                successes=[bool(e["success"]) for e in eps])


LOG_RE = re.compile(r"step:(\S+).*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+|nan|inf).*?lr:([0-9.eE+-]+)")


def parse_train_log(log_name):
    curve = []
    for line in (LOGS / f"{log_name}.log").read_text(errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m:
            step = LOG_FREQ * (len(curve) + 1)
            curve.append(dict(step=step, total_step=OFFSET + step, step_str=m.group(1), loss=float(m.group(2)),
                              grad_norm=float(m.group(3)) if m.group(3) not in ("nan", "inf") else None,
                              lr=float(m.group(4))))
    return curve


def write_summary_md():
    meta = SUMMARY["meta"]
    t = SUMMARY["train"]
    lines = [f"# ACT on PushT ({VARIANT}): continue the 10k-step checkpoint to {OFFSET + STEPS} steps, Kaggle", "",
             f"- lerobot commit: `{meta['lerobot_commit']}` (v0.3.3), dataset revision v2.1, {V['dataset']}",
             f"- device: {meta.get('device')}  |  GPU: {meta.get('gpu', {}).get('device_name')}  |  torch {meta.get('gpu', {}).get('torch')}",
             f"- input checkpoint: `{meta.get('input_ckpt')}` ({OFFSET} steps, train seed {TRAIN_SEED})",
             f"- training: {STEPS} more steps (total {OFFSET + STEPS}), batch {BATCH_TRAIN}, seed {TRAIN_SEED}, save every {SAVE_FREQ}; "
             f"rc={t.get('rc')}, wall {t.get('wall_s', 0) / 60:.1f} min, {t.get('it_s', '-')} it/s, killed_by_budget={t.get('killed')}",
             f"- evaluation: intermediate checkpoints seeds {EVAL_SEEDS} x {N_EPISODES} episodes; last checkpoint seeds {FINAL_EVAL_SEEDS} x {N_EPISODES}",
             f"- started {meta['started_utc']} UTC, elapsed {meta.get('elapsed_s', 0) / 60:.1f} min", "",
             "## Training loss (L1 + 10*KL), mean over the 10 log points before each checkpoint", "",
             "| total step | loss (mean of last 1k steps) | grad norm | lr |", "|---|---|---|---|"]
    curve = t.get("curve") or []
    for s in range(OFFSET + SAVE_FREQ, OFFSET + STEPS + 1, SAVE_FREQ):
        pts = [c for c in curve if s - 1000 < c["total_step"] <= s]
        if pts:
            lines.append(f"| {s} | {sum(c['loss'] for c in pts) / len(pts):.4f} | "
                         f"{sum(c['grad_norm'] or 0 for c in pts) / len(pts):.2f} | {pts[-1]['lr']:.1e} |")
    lines += ["", "## Success rate per checkpoint (k/50 per eval seed; Wilson 95% CI on the pooled row)", ""]
    groups = {}
    for name, r in SUMMARY["runs"].items():
        if r.get("stats"):
            groups.setdefault(r["total_step"], []).append(r)
    seeds_all = sorted({r["seed"] for rs in groups.values() for r in rs})
    lines += ["| total step | " + " | ".join(str(s) for s in seeds_all) + " | pooled k/N | pc | Wilson 95% CI |",
              "|---|" + "---|" * (len(seeds_all) + 3)]
    for ts in sorted(groups):
        by_seed = {r["seed"]: r["stats"] for r in groups[ts]}
        cells = [f"{by_seed[s]['n_success']}" if s in by_seed else "" for s in seeds_all]
        k = sum(x["n_success"] for x in by_seed.values())
        n = sum(x["n_episodes"] for x in by_seed.values())
        lo, hi = wilson_ci(k, n)
        lines.append(f"| {ts} | " + " | ".join(cells) + f" | {k}/{n} | {100 * k / n:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] |")
    if REF[VARIANT] and groups:
        ts = max(groups)
        k = sum(r["stats"]["n_success"] for r in groups[ts])
        n = sum(r["stats"]["n_episodes"] for r in groups[ts])
        rk, rn, rl = REF[VARIANT]
        z, p = two_prop_z(k, n, rk, rn)
        lines += ["", f"- last checkpoint ({ts} steps) {k}/{n} = {100 * k / n:.1f}% vs reference {rk}/{rn} = {100 * rk / rn:.1f}% "
                      f"({rl}): two-proportion z={z:.2f}, p={p:.3f}"]
    lines += ["", "## Evaluations", "",
              "| run | total step | eval seed | success | pc_success | 95% CI (Wilson) | avg_sum_reward | avg_max_reward | eval_s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, r in SUMMARY["runs"].items():
        s = r.get("stats")
        if s:
            lines.append(f"| {name} | {r['total_step']} | {r['seed']} | {s['n_success']}/{s['n_episodes']} | {s['pc_success']:.1f}% | "
                         f"[{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}] | {s['avg_sum_reward']:.2f} | {s['avg_max_reward']:.3f} | {s['eval_s']:.0f} |")
        else:
            lines.append(f"| {name} | {r['total_step']} | {r['seed']} | FAILED rc={r.get('rc')} | - | - | - | - | - |")
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
    script = TMP / "gpu_check.py"
    script.write_text(GPU_CHECK.lstrip())
    sh(f"{PY} {script}", "02_gpu_check")
    raw = grep_log("02_gpu_check", "GPU_CHECK ")
    try:
        return json.loads(raw) if raw else {"gpu_ok": False, "error": "no GPU_CHECK line"}
    except Exception:
        return {"gpu_ok": False, "error": raw}


def step_install():
    sh("nvidia-smi || true; echo; python -V; nproc; free -g; df -h /kaggle/working /kaggle/tmp /tmp | cat; "
       "echo; find /kaggle/input -maxdepth 4 | head -40", "00_env_info", tail=40)
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


def find_input_ckpt():
    hits = sorted(glob.glob(f"/kaggle/input/**/ckpts/{VARIANT}_train1/pretrained_model/model.safetensors", recursive=True))
    assert hits, f"no input checkpoint for {VARIANT}_train1 under /kaggle/input (kernel_sources must list lerobot-train-act-pusht)"
    ckpt = Path(hits[0]).parent
    # copy out of the read-only input mount so lerobot can read it like a normal directory
    local = TMP / "input_ckpt" / VARIANT / "pretrained_model"
    shutil.copytree(ckpt, local, dirs_exist_ok=True)
    shutil.copy(local / "config.json", ENV_DIR / "input_ckpt.config.json")
    shutil.copy(local / "train_config.json", ENV_DIR / "input_ckpt.train_config.json")
    SUMMARY["meta"]["input_ckpt"] = str(ckpt)
    log(f"input checkpoint: {ckpt}")
    return local


def run_eval(ckpt_dir, total_step, eval_seed):
    name = f"{VARIANT}_s{total_step:06d}_eval{eval_seed}"
    out = EVAL_DIR / name
    if out.exists() and not (out / "eval_info.json").exists():
        shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_EVAL} --policy.path={ckpt_dir} --policy.device=cuda {V['env']} "
           f"--eval.n_episodes={N_EPISODES} --eval.batch_size={BATCH_EVAL} --seed={eval_seed} --output_dir={out}")
    t = time.time()
    rc = sh(cmd, name, timeout=2400, tail=4)
    rec = {"total_step": total_step, "seed": eval_seed, "policy_path": str(ckpt_dir), "rc": rc,
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


def evaluate_checkpoint(step_dir, final):
    step = int(step_dir.name)
    total = OFFSET + step
    pm = step_dir / "pretrained_model"
    for f in ("config.json", "train_config.json"):
        if (pm / f).exists():
            shutil.copy(pm / f, ENV_DIR / f"{RUN}_{total:06d}.{f}")
    for es in (FINAL_EVAL_SEEDS if final else EVAL_SEEDS):
        run_eval(pm, total, es)
    # drop optimizer/rng state of older checkpoints to save disk; keep pretrained_model of every checkpoint until the end
    for other in step_dir.parent.iterdir():
        if other.is_dir() and not other.is_symlink() and other != step_dir and (other / "training_state").exists():
            shutil.rmtree(other / "training_state", ignore_errors=True)


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
    try:
        step_install()
        in_ckpt = find_input_ckpt()
    except Exception:
        SUMMARY["errors"].append("install: " + traceback.format_exc()[-1500:])
        log("INSTALL FAILED:\n" + traceback.format_exc())
        save_summary()
        print("KERNEL_DONE (install failed)", flush=True)
        return

    out = TMP / "train" / RUN
    shutil.rmtree(out, ignore_errors=True)
    cmd = (f"{LEROBOT_TRAIN} --policy.path={in_ckpt} --policy.device=cuda --policy.push_to_hub=false "
           f"--dataset.repo_id={V['dataset']} {V['env']} --output_dir={out} --job_name={RUN} --steps={STEPS} --save_freq={SAVE_FREQ} --eval_freq=0 "
           f"--log_freq={LOG_FREQ} --batch_size={BATCH_TRAIN} --num_workers=4 --seed={TRAIN_SEED} --wandb.enable=false")
    t_train = time.time()
    p = start_bg(cmd, RUN)
    SUMMARY["train"] = dict(cmd=cmd, rc=None, killed=False)
    done = set()
    killed = False
    while True:
        rc = p.poll()
        last = current_last(out)
        if last is not None and last.name not in done:
            final = rc is not None or int(last.name) >= STEPS
            if not final and time.time() - T0 > TRAIN_BUDGET_S:
                final = True
            log(f"new checkpoint {last.name} (total {OFFSET + int(last.name)}), final={final}; training rc={rc}")
            try:
                evaluate_checkpoint(last, final)
            except Exception:
                SUMMARY["errors"].append(f"eval {last.name}: " + traceback.format_exc()[-800:])
            done.add(last.name)
            SUMMARY["train"]["curve"] = parse_train_log(RUN)
            save_summary()
            if final and rc is None:
                log("time budget reached: killing training after the final evaluation")
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                killed = True
                rc = p.returncode
                break
        if rc is not None:
            # training ended: make sure its last checkpoint (if written after the previous poll) is evaluated
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
        total = OFFSET + int(last.name)
        have = {r["seed"] for r in SUMMARY["runs"].values() if r.get("total_step") == total and r.get("stats")}
        for es in FINAL_EVAL_SEEDS:
            if es not in have:
                try:
                    run_eval(last / "pretrained_model", total, es)
                except Exception:
                    SUMMARY["errors"].append(f"final eval {total} seed {es}: " + traceback.format_exc()[-600:])

    wall = round(time.time() - t_train, 1)
    curve = parse_train_log(RUN)
    n_steps_done = curve[-1]["step"] if curve else 0
    SUMMARY["train"].update(rc=rc, killed=killed, wall_s=wall, steps_done=n_steps_done,
                            it_s=round(n_steps_done / wall, 2) if wall else None, curve=curve)
    for line in log_tail(RUN, 6):
        print("    | " + line, flush=True)
    log(f"training rc={rc} killed={killed} steps_done={n_steps_done} wall={wall / 60:.1f} min")
    (TRAIN_DIR / RUN).mkdir(parents=True, exist_ok=True)
    json.dump(dict(variant=VARIANT, offset=OFFSET, steps=STEPS, rc=rc, killed=killed, wall_s=wall, batch=BATCH_TRAIN,
                   cmd=cmd, curve=curve), open(TRAIN_DIR / RUN / "train_log.json", "w"), indent=1)
    last = current_last(out)
    if last is not None and (last / "pretrained_model" / "model.safetensors").exists():
        dst = CKPT_OUT / f"{VARIANT}_{OFFSET + int(last.name)}" / "pretrained_model"
        shutil.copytree(last / "pretrained_model", dst, dirs_exist_ok=True)
        log(f"kept checkpoint {last.name} -> {dst}")
    save_summary()
    print("\n" + (WORK / "summary.md").read_text(), flush=True)
    log("KERNEL_DONE")


if __name__ == "__main__":
    main()
