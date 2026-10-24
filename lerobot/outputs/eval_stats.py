"""Summarize one or more lerobot eval_info.json files: success rate, Bernoulli variance/std, 95% CI.

Usage: python outputs/eval_stats.py path/to/eval_info.json [more.json ...]
"""
import json
import math
import sys


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def summarize(path):
    with open(path) as f:
        info = json.load(f)
    eps = info["per_episode"]
    n = len(eps)
    succ = [1.0 if e["success"] else 0.0 for e in eps]
    k = int(sum(succ))
    p = k / n
    # sample (unbiased, n-1) and population (Bernoulli p(1-p)) variance of per-episode success
    var_sample = sum((s - p) ** 2 for s in succ) / (n - 1) if n > 1 else float("nan")
    var_pop = p * (1 - p)
    se = math.sqrt(var_pop / n)
    lo_n, hi_n = p - 1.96 * se, p + 1.96 * se
    lo_w, hi_w = wilson_ci(k, n)
    sum_r = [e["sum_reward"] for e in eps]
    max_r = [e["max_reward"] for e in eps]
    seeds = [e["seed"] for e in eps]
    agg = info["aggregated"]
    print(f"file: {path}")
    print(f"  n_episodes            : {n}   seeds {min(seeds)}..{max(seeds)}")
    print(f"  successes             : {k}/{n}  ->  pc_success = {100*p:.1f}%  (eval.py reports {agg['pc_success']:.1f}%)")
    print(f"  success sample var    : {var_sample:.4f}   (std {math.sqrt(var_sample):.4f})")
    print(f"  success Bernoulli var : {var_pop:.4f}   (std {math.sqrt(var_pop):.4f}),  SE of mean = {se:.4f}")
    print(f"  95% CI (normal)       : [{100*lo_n:.1f}%, {100*hi_n:.1f}%]")
    print(f"  95% CI (Wilson)       : [{100*lo_w:.1f}%, {100*hi_w:.1f}%]")
    print(f"  avg_sum_reward        : {agg['avg_sum_reward']:.3f}   (min {min(sum_r):.3f}, max {max(sum_r):.3f})")
    print(f"  avg_max_reward        : {agg['avg_max_reward']:.3f}   (min {min(max_r):.3f}, max {max(max_r):.3f})")
    print(f"  eval wall time        : {agg['eval_s']:.1f}s total, {agg['eval_ep_s']:.2f}s/episode")
    return dict(path=path, n=n, k=k, p=p, var_sample=var_sample, var_pop=var_pop, se=se,
                ci_normal=(lo_n, hi_n), ci_wilson=(lo_w, hi_w),
                avg_sum_reward=agg["avg_sum_reward"], avg_max_reward=agg["avg_max_reward"], eval_s=agg["eval_s"])


if __name__ == "__main__":
    rows = [summarize(p) for p in sys.argv[1:]]
    if len(rows) > 1:
        ps = [r["p"] for r in rows]
        m = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - m) ** 2 for x in ps) / (len(ps) - 1))
        print(f"\nacross {len(rows)} runs: mean pc_success = {100*m:.1f}%, std across runs = {100*sd:.1f} pts, "
              f"values = {[round(100*x,1) for x in ps]}")
