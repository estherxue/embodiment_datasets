"""Replay recorded LIBERO demonstrations (open-loop action replay) in OffScreenRenderEnv.

For every task of a suite: reset env -> set_init_state(states[0] of the demo) -> step the
recorded actions -> success := env.step(...) returned done (== env._check_success()).
This mirrors libero/lifelong/metric.py::evaluate_one_task_success except that the actions
come from the dataset instead of a policy.

Usage (from the LIBERO repo root, venv activated):
    python local_eval/replay_demos.py --suite libero_spatial --n-demos 10 --tasks 0 1 2
"""
import argparse
import json
import os
import time

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="*", default=None, help="task ids (default: all)")
    p.add_argument("--n-demos", type=int, default=10, help="demos per task (dataset has 50)")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--camera-obs", type=int, default=1, help="1: render 128x128 cameras like eval")
    p.add_argument("--out", default="local_eval/results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    bench = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = args.tasks if args.tasks else list(range(bench.n_tasks))
    datasets_dir = get_libero_path("datasets")
    bddl_dir = get_libero_path("bddl_files")

    for task_id in task_ids:
        task = bench.get_task(task_id)
        bddl = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
        demo_path = os.path.join(datasets_dir, bench.get_task_demonstration(task_id))
        out_path = os.path.join(args.out, f"{args.suite}_task{task_id}_replay.json")
        print(f"[task {task_id}] {task.name}\n    lang: {task.language}", flush=True)

        t_env = time.time()
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=128, camera_widths=128,
            use_camera_obs=bool(args.camera_obs),
        )
        env.seed(args.seed)
        print(f"    env built in {time.time() - t_env:.1f}s", flush=True)

        f = h5py.File(demo_path, "r")
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[: args.n_demos]
        records = []
        for d in demos:
            states = f[f"data/{d}/states"][()]
            actions = f[f"data/{d}/actions"][()]
            t0 = time.time()
            env.reset()                       # same as the eval loop: reset, then set the fixed state
            env.set_init_state(states[0])
            success, success_step = False, None
            max_dev = 0.0
            for t, a in enumerate(actions):
                obs, r, done, info = env.step(a)
                if t + 1 < len(states):       # drift vs. the recorded state (skip the time entry)
                    dev = float(np.linalg.norm(env.get_sim_state()[1:] - states[t + 1][1:]))
                    max_dev = max(max_dev, dev)
                if done:
                    success, success_step = True, t + 1
                    break
            final_success = bool(env.check_success())
            rec = dict(task_id=task_id, demo=d, n_actions=int(len(actions)), success=bool(success),
                       success_step=success_step, final_success=final_success,
                       max_state_drift=round(max_dev, 4), seconds=round(time.time() - t0, 1))
            records.append(rec)
            print(f"    {d}: success={success} step={success_step}/{len(actions)} "
                  f"drift={max_dev:.3f} ({rec['seconds']}s)", flush=True)
        f.close()
        env.close()

        n_succ = sum(r["success"] for r in records)
        summary = dict(suite=args.suite, task_id=task_id, task=task.name, language=task.language,
                       n_demos=len(records), n_success=n_succ,
                       success_rate=n_succ / max(1, len(records)),
                       mean_steps_to_success=float(np.mean([r["success_step"] for r in records if r["success"]]))
                       if n_succ else None,
                       mean_demo_len=float(np.mean([r["n_actions"] for r in records])),
                       records=records)
        with open(out_path, "w") as fo:
            json.dump(summary, fo, indent=1)
        print(f"[task {task_id}] success {n_succ}/{len(records)} = {summary['success_rate']:.2f} -> {out_path}", flush=True)
    print("REPLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
