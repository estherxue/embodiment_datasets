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
EMBED = {'replay_demos.py': '"""Replay recorded LIBERO demonstrations (open-loop action replay) in OffScreenRenderEnv.\n\nFor every task of a suite: reset env -> set_init_state(states[0] of the demo) -> step the\nrecorded actions -> success := env.step(...) returned done (== env._check_success()).\nThis mirrors libero/lifelong/metric.py::evaluate_one_task_success except that the actions\ncome from the dataset instead of a policy.\n\nUsage (from the LIBERO repo root, venv activated):\n    python local_eval/replay_demos.py --suite libero_spatial --n-demos 10 --tasks 0 1 2\n"""\nimport argparse\nimport json\nimport os\nimport time\n\nimport h5py\nimport numpy as np\n\nfrom libero.libero import benchmark, get_libero_path\nfrom libero.libero.envs import OffScreenRenderEnv\n\n\ndef parse_args():\n    p = argparse.ArgumentParser()\n    p.add_argument("--suite", default="libero_spatial")\n    p.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids (default: all)")\n    p.add_argument("--n-demos", type=int, default=10, help="demos per task (dataset has 50)")\n    p.add_argument("--seed", type=int, default=10000)\n    p.add_argument("--camera-obs", type=int, default=1, help="1: render 128x128 cameras like eval")\n    p.add_argument("--out", default="local_eval/results")\n    return p.parse_args()\n\n\ndef main():\n    args = parse_args()\n    os.makedirs(args.out, exist_ok=True)\n    bench = benchmark.get_benchmark_dict()[args.suite]()\n    task_ids = args.tasks if args.tasks else list(range(bench.n_tasks))\n    datasets_dir = get_libero_path("datasets")\n    bddl_dir = get_libero_path("bddl_files")\n\n    for task_id in task_ids:\n        task = bench.get_task(task_id)\n        bddl = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)\n        demo_path = os.path.join(datasets_dir, bench.get_task_demonstration(task_id))\n        out_path = os.path.join(args.out, f"{args.suite}_task{task_id}_replay.json")\n        print(f"[task {task_id}] {task.name}\\n    lang: {task.language}", flush=True)\n\n        t_env = time.time()\n        env = OffScreenRenderEnv(\n            bddl_file_name=bddl, camera_heights=128, camera_widths=128,\n            use_camera_obs=bool(args.camera_obs),\n        )\n        env.seed(args.seed)\n        print(f"    env built in {time.time() - t_env:.1f}s", flush=True)\n\n        f = h5py.File(demo_path, "r")\n        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[: args.n_demos]\n        records = []\n        for d in demos:\n            states = f[f"data/{d}/states"][()]\n            actions = f[f"data/{d}/actions"][()]\n            t0 = time.time()\n            env.reset()                       # same as the eval loop: reset, then set the fixed state\n            env.set_init_state(states[0])\n            success, success_step = False, None\n            max_dev = 0.0\n            for t, a in enumerate(actions):\n                obs, r, done, info = env.step(a)\n                if t + 1 < len(states):       # drift vs. the recorded state (skip the time entry)\n                    dev = float(np.linalg.norm(env.get_sim_state()[1:] - states[t + 1][1:]))\n                    max_dev = max(max_dev, dev)\n                if done:\n                    success, success_step = True, t + 1\n                    break\n            final_success = bool(env.check_success())\n            rec = dict(task_id=task_id, demo=d, n_actions=int(len(actions)), success=bool(success),\n                       success_step=success_step, final_success=final_success,\n                       max_state_drift=round(max_dev, 4), seconds=round(time.time() - t0, 1))\n            records.append(rec)\n            print(f"    {d}: success={success} step={success_step}/{len(actions)} "\n                  f"drift={max_dev:.3f} ({rec[\'seconds\']}s)", flush=True)\n        f.close()\n        env.close()\n\n        n_succ = sum(r["success"] for r in records)\n        summary = dict(suite=args.suite, task_id=task_id, task=task.name, language=task.language,\n                       n_demos=len(records), n_success=n_succ,\n                       success_rate=n_succ / max(1, len(records)),\n                       mean_steps_to_success=float(np.mean([r["success_step"] for r in records if r["success"]]))\n                       if n_succ else None,\n                       mean_demo_len=float(np.mean([r["n_actions"] for r in records])),\n                       records=records)\n        with open(out_path, "w") as fo:\n            json.dump(summary, fo, indent=1)\n        print(f"[task {task_id}] success {n_succ}/{len(records)} = {summary[\'success_rate\']:.2f} -> {out_path}", flush=True)\n    print("REPLAY_DONE", flush=True)\n\n\nif __name__ == "__main__":\n    main()\n', 'evaluate_cpu.py': '"""CPU-runnable copy of libero/lifelong/evaluate.py.\n\nDifferences from the upstream script are marked with "# [local_eval patch]":\n  * --device_id is optional; when omitted the checkpoint/policy run on CPU (upstream hard-codes cuda:N)\n  * --env-num replaces the hard-coded 20 parallel SubprocVectorEnv workers (= number of evaluated episodes)\n  * --max-steps optionally overrides cfg.eval.max_steps (upstream protocol: 600 for libero_spatial)\nEverything else (checkpoint lookup, fixed init states, 5 warm-up zero-action steps, success == done from\nenv.step, success_rate = num_success / env_num, stats file naming) is unchanged.\n"""\nimport argparse\nimport sys\nimport os\n\n# TODO: find a better way for this?\nos.environ["TOKENIZERS_PARALLELISM"] = "false"\nimport hydra\nimport json\nimport numpy as np\nimport pprint\nimport time\nimport torch\nimport wandb\nimport yaml\nfrom easydict import EasyDict\nfrom hydra.utils import get_original_cwd, to_absolute_path\nfrom omegaconf import DictConfig, OmegaConf\nfrom torch.utils.data import DataLoader\nfrom transformers import AutoModel, pipeline, AutoTokenizer, logging\nfrom pathlib import Path\n\nfrom libero.libero import get_libero_path\nfrom libero.libero.benchmark import get_benchmark\nfrom libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv\nfrom libero.libero.utils.time_utils import Timer\nfrom libero.libero.utils.video_utils import VideoWriter\nfrom libero.lifelong.algos import *\nfrom libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset\nfrom libero.lifelong.metric import (\n    evaluate_loss,\n    evaluate_success,\n    raw_obs_to_tensor_obs,\n)\nfrom libero.lifelong.utils import (\n    control_seed,\n    safe_device,\n    torch_load_model,\n    NpEncoder,\n    compute_flops,\n)\n\nfrom libero.lifelong.main import get_task_embs\n\nimport robomimic.utils.obs_utils as ObsUtils\nimport robomimic.utils.tensor_utils as TensorUtils\n\nimport time\n\n\nbenchmark_map = {\n    "libero_10": "LIBERO_10",\n    "libero_spatial": "LIBERO_SPATIAL",\n    "libero_object": "LIBERO_OBJECT",\n    "libero_goal": "LIBERO_GOAL",\n}\n\nalgo_map = {\n    "base": "Sequential",\n    "er": "ER",\n    "ewc": "EWC",\n    "packnet": "PackNet",\n    "multitask": "Multitask",\n}\n\npolicy_map = {\n    "bc_rnn_policy": "BCRNNPolicy",\n    "bc_transformer_policy": "BCTransformerPolicy",\n    "bc_vilt_policy": "BCViLTPolicy",\n}\n\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description="Evaluation Script")\n    parser.add_argument("--experiment_dir", type=str, default="experiments")\n    # for which task suite\n    parser.add_argument(\n        "--benchmark",\n        type=str,\n        required=True,\n        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],\n    )\n    parser.add_argument("--task_id", type=int, required=True)\n    # method detail\n    parser.add_argument(\n        "--algo",\n        type=str,\n        required=True,\n        choices=["base", "er", "ewc", "packnet", "multitask"],\n    )\n    parser.add_argument(\n        "--policy",\n        type=str,\n        required=True,\n        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],\n    )\n    parser.add_argument("--seed", type=int, required=True)\n    parser.add_argument("--ep", type=int)\n    parser.add_argument("--load_task", type=int)\n    parser.add_argument("--device_id", type=int, default=None)  # [local_eval patch] optional\n    parser.add_argument("--save-videos", action="store_true")\n    parser.add_argument("--env-num", type=int, default=20)  # [local_eval patch]\n    parser.add_argument("--max-steps", type=int, default=None)  # [local_eval patch]\n    # parser.add_argument(\'--save_dir\',  type=str, required=True)\n    args = parser.parse_args()\n    # [local_eval patch] cpu when no device id is given\n    args.device_id = "cpu" if args.device_id is None else "cuda:" + str(args.device_id)\n    args.save_dir = f"{args.experiment_dir}_saved"\n\n    if args.algo == "multitask":\n        assert args.ep in list(\n            range(0, 50, 5)\n        ), "[error] ep should be in [0, 5, ..., 50]"\n    else:\n        assert args.load_task in list(\n            range(10)\n        ), "[error] load_task should be in [0, ..., 9]"\n    return args\n\n\ndef main():\n    args = parse_args()\n    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/\n\n    experiment_dir = os.path.join(\n        args.experiment_dir,\n        f"{benchmark_map[args.benchmark]}/"\n        + f"{algo_map[args.algo]}/"\n        + f"{policy_map[args.policy]}_seed{args.seed}",\n    )\n\n    # find the checkpoint\n    experiment_id = 0\n    for path in Path(experiment_dir).glob("run_*"):\n        if not path.is_dir():\n            continue\n        try:\n            folder_id = int(str(path).split("run_")[-1])\n            if folder_id > experiment_id:\n                experiment_id = folder_id\n        except BaseException:\n            pass\n    if experiment_id == 0:\n        print(f"[error] cannot find the checkpoint under {experiment_dir}")\n        sys.exit(0)\n\n    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")\n    try:\n        if args.algo == "multitask":\n            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")\n            sd, cfg, previous_mask = torch_load_model(\n                model_path, map_location=args.device_id\n            )\n        else:\n            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")\n            sd, cfg, previous_mask = torch_load_model(\n                model_path, map_location=args.device_id\n            )\n    except:\n        print(f"[error] cannot find the checkpoint at {str(model_path)}")\n        sys.exit(0)\n\n    cfg.folder = get_libero_path("datasets")\n    cfg.bddl_folder = get_libero_path("bddl_files")\n    cfg.init_states_folder = get_libero_path("init_states")\n\n    cfg.device = args.device_id\n    if args.max_steps is not None:  # [local_eval patch]\n        print(f"[local_eval patch] overriding cfg.eval.max_steps {cfg.eval.max_steps} -> {args.max_steps}")\n        cfg.eval.max_steps = args.max_steps\n    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)\n    algo.policy.previous_mask = previous_mask\n\n    if cfg.lifelong.algo == "PackNet":\n        algo.eval()\n        for module_idx, module in enumerate(algo.policy.modules()):\n            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):\n                weight = module.weight.data\n                mask = algo.previous_masks[module_idx].to(cfg.device)\n                weight[mask.eq(0)] = 0.0\n                weight[mask.gt(args.task_id + 1)] = 0.0\n                # we never train norm layers\n            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):\n                module.eval()\n\n    algo.policy.load_state_dict(sd)\n\n    if not hasattr(cfg.data, "task_order_index"):\n        cfg.data.task_order_index = 0\n\n    # get the benchmark the task belongs to\n    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)\n    descriptions = [benchmark.get_task(i).language for i in range(10)]\n    task_embs = get_task_embs(cfg, descriptions)\n    benchmark.set_task_embs(task_embs)\n\n    task = benchmark.get_task(args.task_id)\n\n    ### ======================= start evaluation ============================\n\n    # 1. evaluate dataset loss\n    try:\n        dataset, shape_meta = get_dataset(\n            dataset_path=os.path.join(\n                cfg.folder, benchmark.get_task_demonstration(args.task_id)\n            ),\n            obs_modality=cfg.data.obs.modality,\n            initialize_obs_utils=True,\n            seq_len=cfg.data.seq_len,\n        )\n        dataset = GroupedTaskDataset(\n            [dataset], task_embs[args.task_id : args.task_id + 1]\n        )\n    except:\n        print(\n            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"\n        )\n        sys.exit(0)\n\n    algo.eval()\n\n    test_loss = 0.0\n\n    # 2. evaluate success rate\n    if args.algo == "multitask":\n        save_folder = os.path.join(\n            args.save_dir,\n            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",\n        )\n    else:\n        save_folder = os.path.join(\n            args.save_dir,\n            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",\n        )\n\n    video_folder = os.path.join(\n        args.save_dir,\n        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",\n    )\n\n    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:\n        env_args = {\n            "bddl_file_name": os.path.join(\n                cfg.bddl_folder, task.problem_folder, task.bddl_file\n            ),\n            "camera_heights": cfg.data.img_h,\n            "camera_widths": cfg.data.img_w,\n        }\n\n        env_num = args.env_num  # [local_eval patch] upstream: env_num = 20\n        env = SubprocVectorEnv(\n            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]\n        )\n        env.reset()\n        env.seed(cfg.seed)\n        algo.reset()\n\n        init_states_path = os.path.join(\n            cfg.init_states_folder, task.problem_folder, task.init_states_file\n        )\n        init_states = torch.load(init_states_path)\n        indices = np.arange(env_num) % init_states.shape[0]\n        init_states_ = init_states[indices]\n\n        dones = [False] * env_num\n        steps = 0\n        obs = env.set_init_state(init_states_)\n        task_emb = benchmark.get_task_emb(args.task_id)\n\n        num_success = 0\n        for _ in range(5):  # simulate the physics without any actions\n            env.step(np.zeros((env_num, 7)))\n\n        with torch.no_grad():\n            while steps < cfg.eval.max_steps:\n                steps += 1\n\n                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)\n                actions = algo.policy.get_action(data)\n                obs, reward, done, info = env.step(actions)\n                video_writer.append_vector_obs(\n                    obs, dones, camera_name="agentview_image"\n                )\n\n                # check whether succeed\n                for k in range(env_num):\n                    dones[k] = dones[k] or done[k]\n                if all(dones):\n                    break\n\n            for k in range(env_num):\n                num_success += int(dones[k])\n\n        success_rate = num_success / env_num\n        env.close()\n\n        eval_stats = {\n            "loss": test_loss,\n            "success_rate": success_rate,\n        }\n\n        os.system(f"mkdir -p {args.save_dir}")\n        torch.save(eval_stats, save_folder)\n    print(\n        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"\n    )\n    print(f"Results are saved at {save_folder}")\n    print(test_loss, success_rate)\n    # [local_eval patch] machine-readable summary\n    print("EVAL_SUMMARY " + json.dumps(dict(\n        benchmark=args.benchmark, task_id=args.task_id, algo=args.algo, policy=args.policy,\n        episodes=env_num, steps_run=steps, max_steps=cfg.eval.max_steps,\n        num_success=num_success, success_rate=success_rate, seconds=round(t.get_elapsed_time(), 1))))\n\n\nif __name__ == "__main__":\n    main()\n', 'make_random_ckpt.py': '"""Build an UNTRAINED (randomly initialised) Multitask/BCTransformerPolicy checkpoint in the exact\nlayout that libero/lifelong/evaluate.py expects, so the evaluation code path can be exercised on a\nCPU-only machine. The resulting success rate is NOT a meaningful number.\n\nRun from the LIBERO repo root:\n    python local_eval/make_random_ckpt.py            # -> experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep0.pth\n"""\nimport os\nimport yaml\nimport torch\nfrom easydict import EasyDict\nfrom hydra import compose, initialize\nfrom omegaconf import OmegaConf\n\nfrom libero.libero import get_libero_path\nfrom libero.libero.benchmark import get_benchmark\nfrom libero.lifelong.algos import get_algo_class\nfrom libero.lifelong.datasets import get_dataset\nfrom libero.lifelong.utils import (control_seed, create_experiment_dir, get_task_embs,\n                                   safe_device, torch_save_model)\n\n# same config composition as libero/lifelong/main.py (hydra), with the overrides used by the README\nwith initialize(config_path="../libero/configs", version_base=None):\n    hydra_cfg = compose(config_name="config", overrides=[\n        "benchmark_name=LIBERO_SPATIAL", "lifelong=multitask", "policy=bc_transformer_policy",\n        "seed=10000", "device=cpu",\n    ])\ncfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))\ncontrol_seed(cfg.seed)\ncfg.folder = get_libero_path("datasets")\ncfg.bddl_folder = get_libero_path("bddl_files")\ncfg.init_states_folder = get_libero_path("init_states")\n\nbenchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)\n_, shape_meta = get_dataset(\n    dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(0)),\n    obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len)\nprint("shape_meta:", shape_meta)\n\ndescriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]\ntask_embs = get_task_embs(cfg, descriptions)  # bert-base-cased pooler output, cached under ./bert\nprint("task_embs:", tuple(task_embs.shape))\n\ncreate_experiment_dir(cfg)  # ./experiments/LIBERO_SPATIAL/Multitask/BCTransformerPolicy_seed10000/run_XXX\ncfg.shape_meta = shape_meta\nalgo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device)\nn_params = sum(p.numel() for p in algo.policy.parameters())\npath = os.path.join(cfg.experiment_dir, "multitask_model_ep0.pth")\ntorch_save_model(algo.policy, path, cfg=cfg, previous_masks=None)\nprint(f"saved untrained policy ({n_params/1e6:.2f} M params) to {path}")\n', 'fetch_bert.py': '"""Robust (resumable, retried) download of bert-base-cased into ./bert-base-cased so that\ntransformers 4.21\'s from_pretrained("bert-base-cased") resolves to the local directory when run\nfrom the repo root (its own downloader has no resume and kept failing on this network)."""\nimport time, os\nfrom huggingface_hub import hf_hub_download\ndst = "bert-base-cased"\nfor fname in ["config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json", "pytorch_model.bin"]:\n    for attempt in range(1, 30):\n        try:\n            p = hf_hub_download("bert-base-cased", fname, local_dir=dst)\n            print("ok", p, os.path.getsize(p), flush=True); break\n        except Exception as e:\n            print(f"[{fname} attempt {attempt}] {type(e).__name__}: {str(e)[:120]}", flush=True); time.sleep(5)\nprint("BERT_DONE", flush=True)\n'}

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
