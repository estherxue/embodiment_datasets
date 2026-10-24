#!/usr/bin/env python
"""Kaggle script kernel: LIBERO (commit 8f1084e) libero_spatial evaluation flow.

Stages (each wrapped; failures are recorded in /kaggle/working/summary.json and the script always exits 0):
  0 sysinfo  1 apt (EGL/OSMesa libs)  2 uv + Python 3.10 venv  3 clone LIBERO  4 install deps  5 ~/.libero/config.yaml
  6 render smoke test (egl -> osmesa fallback)  7 download libero_spatial (HF)  8 bert-base-cased
  9 demo replay: 10 tasks x 50 demos (2 parallel procs)  10 evaluate.py protocol with an UNTRAINED checkpoint
  11 summary.json / summary.md
Everything heavy lives in /kaggle/tmp; /kaggle/working only gets results + logs.
"""
import datetime
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

T0 = time.time()
WORK = "/kaggle/working"
TMP = "/kaggle/tmp"
try:
    os.makedirs(TMP, exist_ok=True)
except Exception:
    TMP = "/tmp/libero_tmp"
    os.makedirs(TMP, exist_ok=True)
LOGS = f"{WORK}/logs"
os.makedirs(LOGS, exist_ok=True)
REPO = f"{TMP}/LIBERO"
VENV = f"{TMP}/venv"
VPY = f"{VENV}/bin/python"
DATASETS = f"{TMP}/datasets"
COMMIT = "8f1084e"
DEADLINE = T0 + 9.0 * 3600  # do not START new long work after this (Kaggle hard-kills at 12h)
N_DEMOS = 50
REPLAY_TASKS = list(range(10))  # v2 did tasks 0-8 and v4 task 9 (software vs GPU renderer); v5 repeats all 10 on the GPU renderer
REPLAY_BUDGET_S = 2 * 3600
PROTOCOL_ENV_NUM = 20
PROTOCOL_BUDGET_S = 3 * 3600  # wall-clock budget for protocol tasks 1..9 (task 0 always runs)

summary = {"started_utc": datetime.datetime.utcnow().isoformat(), "stages": {}, "env": {}, "results": {}}

BASE_ENV = dict(os.environ)
BASE_ENV["PYTHONPATH"] = REPO  # the editable install does not expose `libero` outside the repo root; LIBERO itself relies on init_path.py/cwd
BASE_ENV.update({
    "PYTHONUNBUFFERED": "1",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",  # torch>=2.6 defaults torch.load(weights_only=True); LIBERO pickles cfg objects
    "TOKENIZERS_PARALLELISM": "false",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "UV_CACHE_DIR": f"{TMP}/uv-cache",
    "HF_HOME": f"{TMP}/hf",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})

# ----------------------------------------------------------------------------------------------- helpers

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    with open(f"{WORK}/driver.log", "a") as f:
        f.write(line + "\n")


def save_summary():
    summary["elapsed_min"] = round((time.time() - T0) / 60, 1)
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)


def tail(path, n=12, width=300):
    try:
        with open(path, errors="replace") as f:
            return [l.rstrip()[:width] for l in f.read().splitlines()[-n:]]
    except Exception:
        return []


def start(name, cmd, cwd=None, env=None):
    """Start a bash command in its own session; stdout+stderr -> logs/<name>.log."""
    logf = f"{LOGS}/{name}.log"
    log(f"[{name}] $ {cmd}")
    f = open(logf, "ab")
    f.write(f"\n$ {cmd}\n".encode())
    f.flush()
    p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd,
                         env=env or BASE_ENV, start_new_session=True)
    p._logf, p._fh, p._name, p._t0 = logf, f, name, time.time()
    return p


def finish(p, rc):
    p._fh.close()
    dt = time.time() - p._t0
    for l in tail(p._logf):
        print("    | " + l, flush=True)
    log(f"[{p._name}] rc={rc} ({dt:.0f}s)")
    summary["stages"].setdefault(p._name, {}).update({"rc": rc, "seconds": round(dt, 1)})
    save_summary()
    return rc


def run(name, cmd, timeout=3600, cwd=None, env=None):
    p = start(name, cmd, cwd=cwd, env=env)
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"[{name}] TIMEOUT after {timeout}s -> killing process group")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        rc = -9
    return finish(p, rc)


def wait_all(procs, timeout):
    """Wait for several started processes; kill the stragglers at the deadline."""
    end = time.time() + timeout
    rcs = {}
    pending = list(procs)
    while pending and time.time() < end:
        for p in list(pending):
            rc = p.poll()
            if rc is not None:
                rcs[p._name] = finish(p, rc)
                pending.remove(p)
        time.sleep(10)
    for p in pending:
        log(f"[{p._name}] TIMEOUT -> killing")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        rcs[p._name] = finish(p, -9)
    return rcs


def write(path, content, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, mode)


def time_left():
    return DEADLINE - time.time()


# ----------------------------------------------------------------------------------------------- embedded files
EMBED = __EMBED__

SMOKE_PY = r'''
import os, sys, time
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
try:
    import mujoco
    from OpenGL import GL
    _ctx = mujoco.GLContext(64, 64)
    _ctx.make_current()
    print("GL_RENDERER:", GL.glGetString(GL.GL_RENDERER), "| GL_VENDOR:", GL.glGetString(GL.GL_VENDOR), "| GL_VERSION:", GL.glGetString(GL.GL_VERSION), flush=True)
    _ctx.free()
except Exception as e:
    print("GL query failed:", repr(e), flush=True)
t = time.time()
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
obs = env.reset()
print("env built+reset in %.1fs" % (time.time() - t), flush=True)
t1 = time.time()
for _ in range(50):
    obs, r, done, info = env.step(np.zeros(7))
print("STEP_MS %.1f ms/step (50 zero-action steps incl. 2x128x128 rendering)" % ((time.time() - t1) / 50 * 1000), flush=True)
img = obs["agentview_image"]
print("obs keys:", sorted(obs.keys()))
print("agentview_image", img.shape, img.dtype, "std=%.2f" % float(img.std()), "| eye_in_hand", obs["robot0_eye_in_hand_image"].shape)
print("robot0 state: joint_pos", np.asarray(obs["robot0_joint_pos"]).shape, "gripper_qpos", np.asarray(obs["robot0_gripper_qpos"]).shape)
print("action dim:", env.env.action_dim if hasattr(env.env, "action_dim") else "?")
assert float(img.std()) > 1.0, "rendered image is (almost) constant -> renderer broken"
env.close()
print("SMOKE_OK backend=%s %.1fs" % (os.environ.get("MUJOCO_GL"), time.time() - t), flush=True)
'''

HF_FALLBACK_PY = r'''
import os, sys
from huggingface_hub import snapshot_download
dst = sys.argv[1]
p = snapshot_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset", local_dir=dst,
                      allow_patterns="libero_spatial/*", max_workers=8)
print("snapshot at", p)
'''

AGG_PY = r'''
"""Aggregate replay JSONs -> one table (stdout JSON)."""
import glob, json, os, sys
import numpy as np
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "libero_spatial_task*_replay.json")), key=lambda s: int(s.split("task")[-1].split("_")[0])):
    d = json.load(open(f))
    recs = d["records"]
    succ = [r["success"] for r in recs]
    n, k = len(succ), int(sum(succ))
    p = k / max(1, n)
    rows.append(dict(task_id=d["task_id"], task=d["task"], language=d["language"], n_demos=n, n_success=k,
                     success_rate=round(p, 4), bernoulli_std=round(float(np.sqrt(p * (1 - p))), 4),
                     se=round(float(np.sqrt(p * (1 - p) / max(1, n))), 4),
                     mean_steps_to_success=None if d.get("mean_steps_to_success") is None else round(d["mean_steps_to_success"], 1),
                     mean_demo_len=round(d["mean_demo_len"], 1),
                     mean_max_state_drift=round(float(np.mean([r["max_state_drift"] for r in recs])), 3),
                     max_state_drift=round(float(np.max([r["max_state_drift"] for r in recs])), 3),
                     final_success_rate=round(float(np.mean([r["final_success"] for r in recs])), 4),
                     mean_seconds_per_demo=round(float(np.mean([r["seconds"] for r in recs])), 1)))
print(json.dumps(rows))
'''

# ----------------------------------------------------------------------------------------------- stages

def stage_sysinfo():
    run("00_sysinfo", "nvidia-smi || true; python -V; nproc; free -g; df -h / /kaggle/working /tmp 2>/dev/null; "
                      "head -2 /etc/os-release; echo HOME=$HOME; echo PYTHONPATH=$PYTHONPATH; echo cgroup cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null); "
                      "python3 -c 'import os; print(\"affinity\", sorted(os.sched_getaffinity(0)))'; ldconfig -p | grep -i -E 'libEGL|OSMesa' || true",
        timeout=300)
    summary["env"]["hostname"] = os.uname().nodename
    try:
        summary["env"]["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        summary["env"]["gpu"] = f"n/a ({e})"
    summary["env"]["nproc"] = os.cpu_count()
    summary["env"]["kaggle_python"] = sys.version.split()[0]
    save_summary()


def stage_apt():
    run("01_apt", "export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq || true) && "
                  "apt-get install -y -qq --no-install-recommends libegl1 libgl1 libglvnd0 libosmesa6 libosmesa6-dev > /dev/null; "
                  "ldconfig -p | grep -i -E 'libEGL|OSMesa' || true", timeout=900)
    # GLVND picks the EGL vendor from /usr/share/glvnd/egl_vendor.d/*.json. In the Kaggle container the NVIDIA driver
    # libraries are mounted but the ICD json may be missing -> EGL silently falls back to Mesa/llvmpipe (software).
    run("01_egl_vendor", "echo '--- vendor dir:'; ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null; cat /usr/share/glvnd/egl_vendor.d/*.json 2>/dev/null; "
                         "echo '--- nvidia libs:'; ls /usr/local/nvidia/lib64 2>/dev/null | grep -i -E 'egl|glsi|eglcore|nvidia-gl' | head; "
                         "echo '--- cpu affinity:'; nproc; python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))'", timeout=300)
    try:
        have_nv = "libEGL_nvidia.so.0" in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=60).stdout
        icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if have_nv and not os.path.exists(icd):
            os.makedirs(os.path.dirname(icd), exist_ok=True)
            with open(icd, "w") as f:
                json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f, indent=1)
            log(f"wrote {icd} (NVIDIA EGL ICD was not registered for GLVND)")
            summary["env"]["nvidia_icd"] = "written"
        else:
            summary["env"]["nvidia_icd"] = f"have_nvidia_lib={have_nv} icd_exists={os.path.exists(icd)}"
            log(f"nvidia icd: {summary['env']['nvidia_icd']}")
    except Exception as e:
        log(f"icd check failed: {e!r}")
    save_summary()


def stage_uv_venv():
    rc = run("02_uv", f"{sys.executable} -m pip install -q uv && uv --version", timeout=600)
    if rc != 0:
        run("02_uv_curl", f"curl -LsSf https://astral.sh/uv/install.sh | sh && cp $HOME/.local/bin/uv /usr/local/bin/uv; uv --version", timeout=600)
    rc = run("02_venv", f"uv venv --python 3.10 {VENV} && {VPY} -V", timeout=900)
    return rc == 0


def stage_clone():
    rc = run("03_clone", f"rm -rf {REPO}; git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO {REPO} && "
                         f"cd {REPO} && git checkout -q {COMMIT} && git log -1 --format='%H %cd %s' --date=short", timeout=900)
    # SubprocVectorEnv forks its workers; a forked child of a process that already initialised NVIDIA EGL fails with
    # EGL_BAD_ALLOC at eglCreateContext. env_fns are already cloudpickle-wrapped, so switching the workers to "spawn" is enough.
    rc2 = run("03_patch_venv_spawn", f"cd {REPO} && sed -i 's/^from multiprocessing.context import Process$/"
                                     f"from multiprocessing import get_context as _get_context; Process = _get_context(\"spawn\").Process  # [kaggle patch]/' "
                                     f"libero/libero/envs/venv.py && grep -n 'kaggle patch' libero/libero/envs/venv.py && git diff --stat", timeout=120)
    summary["env"]["venv_spawn_patch_rc"] = rc2
    return rc == 0 and rc2 == 0


def stage_install():
    uvpip = f"uv pip install --python {VPY} -q"
    rc = run("04_torch", f"{uvpip} torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu", timeout=1800)
    rc2 = run("04_reqs", f"cd {REPO} && {uvpip} -r requirements.txt huggingface_hub h5py mujoco==2.3.2 'setuptools<81'", timeout=2400)
    if rc2 != 0:
        # robomimic 0.2.0 pulls egl_probe (compiles with cmake); fall back to --no-deps for robomimic
        log("requirements install failed -> retry without robomimic deps")
        run("04_reqs_fallback", f"cd {REPO} && grep -v -i robomimic requirements.txt > /tmp/req_norm.txt && "
                                f"{uvpip} -r /tmp/req_norm.txt huggingface_hub h5py mujoco==2.3.2 && "
                                f"{uvpip} --no-deps robomimic==0.2.0 && {uvpip} tensorboard tensorboardX psutil termcolor imageio imageio-ffmpeg",
            timeout=2400)
    rc3 = run("04_libero", f"cd {REPO} && {uvpip} -e . && uv pip freeze --python {VPY} > {WORK}/requirements-kaggle-freeze.txt && "
                           f"grep -i -E '^(torch|torchvision|numpy|robosuite|mujoco|bddl|robomimic|transformers|hydra-core|gym|h5py|huggingface-hub|tokenizers)=' {WORK}/requirements-kaggle-freeze.txt",
              timeout=900)
    return rc3 == 0


def stage_libero_config():
    cfg_dir = os.path.expanduser("~/.libero")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(DATASETS, exist_ok=True)
    root = f"{REPO}/libero/libero"
    cfg = {"benchmark_root": root, "bddl_files": f"{root}/bddl_files", "init_states": f"{root}/init_files",
           "assets": f"{root}/assets", "datasets": DATASETS}
    with open(f"{cfg_dir}/config.yaml", "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
    shutil.copy(f"{cfg_dir}/config.yaml", f"{WORK}/libero_config.yaml")
    summary["env"]["libero_config"] = cfg
    log(f"wrote {cfg_dir}/config.yaml: {cfg}")
    # importing libero must now NOT prompt
    rc = run("05_import", f"cd {REPO} && {VPY} -c \"import torch, numpy, robosuite, mujoco, h5py, transformers; "
                          f"from libero.libero import get_libero_path, benchmark; print('torch', torch.__version__, 'numpy', numpy.__version__, "
                          f"'robosuite', robosuite.__version__, 'mujoco', mujoco.__version__); print('datasets path', get_libero_path('datasets')); "
                          f"print('benchmarks', list(benchmark.get_benchmark_dict().keys()))\"", timeout=600)
    rc2 = run("05_import_eval_deps", f"cd {REPO} && {VPY} -c \"import wandb, hydra, easydict, omegaconf, yaml, pkg_resources; "
                                     f"import robomimic.utils.obs_utils, robomimic.utils.tensor_utils; from libero.lifelong.algos import *; "
                                     f"from libero.lifelong.metric import evaluate_success; from libero.lifelong.main import get_task_embs; "
                                     f"from libero.lifelong.utils import torch_load_model; print('eval deps OK')\"", timeout=600)
    return rc == 0 and rc2 == 0


def stage_smoke_render():
    write(f"{REPO}/local_eval/smoke_render.py", SMOKE_PY)
    for backend in ("egl", "osmesa"):
        env = dict(BASE_ENV, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        rc = run(f"06_smoke_{backend}", f"cd {REPO} && {VPY} local_eval/smoke_render.py", timeout=900, env=env)
        if rc == 0:
            for l in tail(f"{LOGS}/06_smoke_{backend}.log", n=60, width=400):
                if l.startswith("GL_RENDERER:"):
                    summary["env"]["gl_renderer"] = l
                if l.startswith("STEP_MS"):
                    summary["env"]["step_ms"] = l
            BASE_ENV["MUJOCO_GL"] = backend
            BASE_ENV["PYOPENGL_PLATFORM"] = backend
            summary["env"]["render_backend"] = backend
            save_summary()
            log(f"render backend = {backend}")
            return True
    summary["env"]["render_backend"] = "NONE (egl and osmesa both failed)"
    save_summary()
    return False


def stage_dataset():
    rc = run("07_dataset", f"cd {REPO} && {VPY} benchmark_scripts/download_libero_datasets.py --datasets libero_spatial "
                           f"--use-huggingface --download-dir {DATASETS} < /dev/null", timeout=3600)
    files = sorted(glob.glob(f"{DATASETS}/libero_spatial/*.hdf5"))
    if len(files) < 10:
        log(f"official download script left {len(files)} files -> HF snapshot_download fallback")
        write(f"{REPO}/local_eval/hf_fallback.py", HF_FALLBACK_PY)
        run("07_dataset_fallback", f"cd {REPO} && {VPY} local_eval/hf_fallback.py {DATASETS}", timeout=3600)
        files = sorted(glob.glob(f"{DATASETS}/libero_spatial/*.hdf5"))
    sizes = {os.path.basename(f): round(os.path.getsize(f) / 2**20, 1) for f in files}
    summary["env"]["dataset_files_MiB"] = sizes
    summary["env"]["dataset_total_GiB"] = round(sum(os.path.getsize(f) for f in files) / 2**30, 2)
    save_summary()
    log(f"dataset: {len(files)} hdf5 files, {summary['env']['dataset_total_GiB']} GiB")
    return len(files) == 10


def stage_bert():
    write(f"{REPO}/local_eval/fetch_bert.py", EMBED["fetch_bert.py"])
    run("08_bert", f"cd {REPO} && {VPY} local_eval/fetch_bert.py", timeout=1200)


def stage_replay():
    write(f"{REPO}/local_eval/replay_demos.py", EMBED["replay_demos.py"])
    out = f"{WORK}/replay"
    os.makedirs(out, exist_ok=True)
    half = (len(REPLAY_TASKS) + 1) // 2
    groups = [REPLAY_TASKS[:half], REPLAY_TASKS[half:]] if len(REPLAY_TASKS) > 1 else [REPLAY_TASKS]
    procs = [start(f"09_replay_{'AB'[i]}", f"cd {REPO} && {VPY} local_eval/replay_demos.py --suite libero_spatial --tasks {' '.join(map(str, g))} "
                                           f"--n-demos {N_DEMOS} --out {out}") for i, g in enumerate(groups) if g]
    budget = max(600, min(REPLAY_BUDGET_S, time_left() - 3600))
    rcs = wait_all(procs, timeout=budget)
    write(f"{REPO}/local_eval/agg_replay.py", AGG_PY)
    try:
        res = subprocess.run([VPY, f"{REPO}/local_eval/agg_replay.py", out], capture_output=True, text=True, timeout=300, env=BASE_ENV)
        rows = json.loads(res.stdout.strip().splitlines()[-1]) if res.stdout.strip() else []
        if res.returncode != 0:
            log("agg stderr: " + res.stderr[-500:])
    except Exception as e:
        log(f"aggregation failed: {e}")
        rows = []
    summary["results"]["replay"] = {"rcs": rcs, "n_demos_per_task": N_DEMOS, "per_task": rows}
    if rows:
        tot_n = sum(r["n_demos"] for r in rows)
        tot_k = sum(r["n_success"] for r in rows)
        summary["results"]["replay"]["overall"] = {"n": tot_n, "k": tot_k, "success_rate": round(tot_k / max(1, tot_n), 4),
                                                   "mean_task_success_rate": round(sum(r["success_rate"] for r in rows) / len(rows), 4)}
        for r in rows:
            log(f"replay task {r['task_id']}: {r['n_success']}/{r['n_demos']} = {100*r['success_rate']:.0f}%  drift mean {r['mean_max_state_drift']}  {r['mean_seconds_per_demo']}s/demo")
        log(f"replay overall: {tot_k}/{tot_n} = {100*tot_k/max(1,tot_n):.1f}%")
    save_summary()
    return bool(rows)


def parse_eval_summary(logf):
    for l in reversed(tail(logf, n=400, width=10000)):
        if l.startswith("EVAL_SUMMARY "):
            try:
                return json.loads(l[len("EVAL_SUMMARY "):])
            except Exception:
                return None
    return None


PROTO = {"attempts": [("egl", PROTOCOL_ENV_NUM), ("osmesa", PROTOCOL_ENV_NUM), ("osmesa", 10)], "idx": 0, "t_start": None, "last_task_s": 0.0}


def protocol_prepare():
    write(f"{REPO}/local_eval/make_random_ckpt.py", EMBED["make_random_ckpt.py"])
    write(f"{REPO}/local_eval/evaluate_cpu.py", EMBED["evaluate_cpu.py"])
    rc = run("10_random_ckpt", f"cd {REPO} && {VPY} local_eval/make_random_ckpt.py && find experiments -name '*.pth' | xargs ls -la", timeout=1800)
    summary["results"]["protocol"] = {"random_ckpt_rc": rc, "per_task": [],
                                      "note": "UNTRAINED (randomly initialised) Multitask/BCTransformerPolicy; success numbers are NOT policy performance, only a protocol check"}
    save_summary()
    PROTO["t_start"] = time.time()
    return rc == 0


def protocol_task(task_id):
    res = summary["results"]["protocol"]
    t = time.time()
    rc, ev, backend, env_num = -1, None, None, None
    while PROTO["idx"] < len(PROTO["attempts"]):
        backend, env_num = PROTO["attempts"][PROTO["idx"]]
        name = f"10_eval_task{task_id}_{backend}_env{env_num}"
        env = dict(BASE_ENV, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        rc = run(name, f"cd {REPO} && {VPY} local_eval/evaluate_cpu.py --benchmark libero_spatial --task_id {task_id} --algo multitask "
                       f"--policy bc_transformer_policy --seed 10000 --ep 0 --env-num {env_num}; free -g | head -2", timeout=5400, env=env)
        ev = parse_eval_summary(f"{LOGS}/{name}.log")
        if ev is not None:
            break
        log(f"task {task_id}: attempt (MUJOCO_GL={backend}, env_num={env_num}) failed rc={rc} -> next fallback")
        PROTO["idx"] += 1
    PROTO["last_task_s"] = time.time() - t
    res["per_task"].append({"task_id": task_id, "rc": rc, "backend": backend, "env_num": env_num, "eval_summary": ev, "wall_s": round(PROTO["last_task_s"], 1)})
    res["config_final"] = {"backend": backend, "env_num": env_num}
    log(f"protocol task {task_id}: rc={rc} backend={backend} env_num={env_num} wall={PROTO['last_task_s']:.0f}s summary={ev}")
    save_summary()
    return ev is not None


def protocol_remaining(task_ids):
    res = summary["results"]["protocol"]
    for task_id in task_ids:
        used = time.time() - PROTO["t_start"]
        if used + PROTO["last_task_s"] > PROTOCOL_BUDGET_S or time_left() < PROTO["last_task_s"] + 900:
            skipped = [t for t in task_ids if t >= task_id]
            log(f"protocol: skipping tasks {skipped} (budget: used {used/60:.0f} min, last task {PROTO['last_task_s']/60:.0f} min)")
            res["skipped_tasks"] = skipped
            save_summary()
            return
        protocol_task(task_id)


def stage_final():
    run("11_disk", f"du -sh {TMP}/* {WORK}/* 2>/dev/null | sort -h | tail -15; free -g | head -2", timeout=300)
    r = summary["results"]
    lines = ["# LIBERO libero_spatial on Kaggle - summary", "",
             f"- started (UTC): {summary['started_utc']}  elapsed: {round((time.time()-T0)/60,1)} min",
             f"- GPU: {summary['env'].get('gpu')}  nproc: {summary['env'].get('nproc')}  render backend: {summary['env'].get('render_backend')}",
             f"- LIBERO commit: {COMMIT}; dataset: {summary['env'].get('dataset_total_GiB')} GiB, {len(summary['env'].get('dataset_files_MiB', {}))} hdf5 files",
             f"- renderer: {summary['env'].get('gl_renderer')}  step cost: {summary['env'].get('step_ms')}", ""]
    rep = r.get("replay", {})
    if rep.get("per_task"):
        lines += ["## Demo replay (open-loop action replay of the recorded demos, success = env.step done)", "",
                  "| task | n | succ | rate | SE | mean steps→succ | mean len | mean max drift | s/demo | language |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in rep["per_task"]:
            lines.append(f"| {t['task_id']} | {t['n_demos']} | {t['n_success']} | {100*t['success_rate']:.0f}% | {t['se']} | {t['mean_steps_to_success']} | "
                         f"{t['mean_demo_len']} | {t['mean_max_state_drift']} | {t['mean_seconds_per_demo']} | {t['language']} |")
        o = rep.get("overall", {})
        lines += ["", f"overall: {o.get('k')}/{o.get('n')} = {100*o.get('success_rate',0):.1f}%  (mean of per-task rates {100*o.get('mean_task_success_rate',0):.1f}%)", ""]
    pro = r.get("protocol", {})
    if pro.get("per_task"):
        lines += ["## evaluate.py protocol check with an UNTRAINED checkpoint (numbers are meaningless by construction)", "",
                  "| task | backend | env_num (episodes) | steps run | max_steps | successes | success_rate | rollout s | wall s | rc |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in pro["per_task"]:
            e = t.get("eval_summary") or {}
            lines.append(f"| {t['task_id']} | {t.get('backend')} | {e.get('episodes', t['env_num'])} | {e.get('steps_run')} | {e.get('max_steps')} | {e.get('num_success')} | {e.get('success_rate')} | {e.get('seconds')} | {t.get('wall_s')} | {t['rc']} |")
        if pro.get("skipped_tasks"):
            lines.append(f"\nskipped (time budget): {pro['skipped_tasks']}")
    lines += ["", "## Stages", "", "| stage | rc | seconds |", "|---|---|---|"]
    for k, v in summary["stages"].items():
        lines.append(f"| {k} | {v.get('rc')} | {v.get('seconds')} |")
    with open(f"{WORK}/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    save_summary()
    print("\n".join(lines), flush=True)


def main():
    log(f"start; WORK={WORK} TMP={TMP}")
    stage_sysinfo()
    stage_apt()
    ok = stage_uv_venv()
    ok = ok and stage_clone()
    ok = ok and stage_install()
    ok = ok and stage_libero_config()
    if not ok:
        log("environment setup failed -> skipping evaluation stages")
        return
    render_ok = stage_smoke_render()
    data_ok = stage_dataset()
    stage_bert()
    if render_ok and data_ok:
        if protocol_prepare():
            if protocol_task(0):                  # the required protocol check, full 20-env protocol
                protocol_remaining(list(range(1, 10)))
        stage_replay()                            # demo replay of REPLAY_TASKS x N_DEMOS (repeatability check vs kernel v2/v4)
    else:
        log(f"skipping replay/protocol: render_ok={render_ok} data_ok={data_ok}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER EXCEPTION:\n" + traceback.format_exc())
        summary["driver_exception"] = traceback.format_exc()
    finally:
        try:
            stage_final()
        except Exception:
            log("final stage failed:\n" + traceback.format_exc())
        log("ALL_DONE")
        sys.exit(0)
