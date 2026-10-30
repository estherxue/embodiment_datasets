"""Combine the 100k-step ACT PushT keypoint runs (seed 1 = 10k kernel + continuation, seeds 2/3 from scratch):
success per checkpoint, final 500-episode results, checkpoint-to-checkpoint spread, train-seed decomposition at 100k.
Usage: python3 -I analyze_100k.py <kaggle/outputs root>"""
import json, math, os, sys, glob

root = sys.argv[1]
REF = (320, 500)  # alexandersoare/act_pusht_keypoints eval_info.json


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); z = (k1 / n1 - k2 / n2) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def sd(xs):
    m = sum(xs) / len(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")


# ---- load: {train_seed: {total_step: {eval_seed: (k, n)}}}
runs = {}
SRC = {1: "lerobot-train-act-pusht-100k-kp", 2: "lerobot-train-act-pusht-100k-kp-seed2", 3: "lerobot-train-act-pusht-100k-kp-seed3"}
for ts, d in SRC.items():
    p = os.path.join(root, d, "summary.json")
    if not os.path.exists(p):
        print(f"(missing {d})"); continue
    S = json.load(open(p))
    for name, r in S["runs"].items():
        if r.get("stats"):
            runs.setdefault(ts, {}).setdefault(r["total_step"], {})[r["seed"]] = (r["stats"]["n_success"], r["stats"]["n_episodes"])
    if ts == 1:  # the 10k checkpoint of seed 1 was evaluated in the 10k kernel
        S10 = json.load(open(os.path.join(root, "lerobot-train-act-pusht", "summary.json")))
        for name, r in S10["runs"].items():
            if name.startswith("kp_train1_eval") and not r.get("repeat") and r.get("stats"):
                runs[1].setdefault(10000, {})[r["seed"]] = (r["stats"]["n_success"], r["stats"]["n_episodes"])
seeds = sorted(runs)
steps = sorted({s for ts in runs for s in runs[ts]})

# ---- table: success per checkpoint, first 3 eval seeds pooled (150 eps) so all checkpoints are comparable
print("## success % per checkpoint (eval seeds 1000/2000/3000 pooled, n=150)")
print("| train seed | " + " | ".join(f"{s // 1000}k" for s in steps) + " | last-5 mean | 100k, 500 eps |")
print("|---|" + "---|" * (len(steps) + 2))
pc3, final, last5 = {}, {}, {}
for ts in seeds:
    row = []
    for s in steps:
        d = runs[ts].get(s, {})
        k = sum(d[e][0] for e in (1000, 2000, 3000) if e in d); n = sum(d[e][1] for e in (1000, 2000, 3000) if e in d)
        pc3[(ts, s)] = 100 * k / n if n else None
        row.append(f"{100 * k / n:.1f}" if n else "")
    d = runs[ts].get(100000, {})
    k = sum(v[0] for v in d.values()); n = sum(v[1] for v in d.values()); final[ts] = (k, n)
    l5 = [pc3[(ts, s)] for s in (60000, 70000, 80000, 90000, 100000) if pc3.get((ts, s)) is not None]
    last5[ts] = sum(l5) / len(l5) if l5 else float("nan")
    lo, hi = wilson(k, n) if n else (float("nan"),) * 2
    print(f"| {ts} | " + " | ".join(row) + f" | {last5[ts]:.1f} | {k}/{n} = {100 * k / n:.1f}% [{lo:.1f}, {hi:.1f}] |")

print("\n## checkpoint-to-checkpoint spread, 50k-100k (3-seed pooled values, n=150 each)")
for ts in seeds:
    vals = [pc3[(ts, s)] for s in (50000, 60000, 70000, 80000, 90000, 100000) if pc3.get((ts, s)) is not None]
    m = sum(vals) / len(vals)
    print(f"- seed {ts}: {[round(v, 1) for v in vals]} mean {m:.1f} std {sd(vals):.1f} (Bernoulli SE at n=150: {100 * math.sqrt(m / 100 * (1 - m / 100) / 150):.1f})")

print("\n## 100k checkpoint: train-seed decomposition (10 eval seeds x 50 per train seed)")
mat = {ts: [100 * runs[ts][100000][e][0] / runs[ts][100000][e][1] for e in sorted(runs[ts][100000])] for ts in seeds if 100000 in runs[ts]}
means = {ts: sum(v) / len(v) for ts, v in mat.items()}
E = min(len(v) for v in mat.values())
grand = sum(means.values()) / len(means)
ms_train = E * sum((m - grand) ** 2 for m in means.values()) / (len(means) - 1)
ms_eval = sum((x - means[ts]) ** 2 for ts, v in mat.items() for x in v) / (len(means) * (E - 1))
F = ms_train / ms_eval
comp = max(0.0, (ms_train - ms_eval) / E)
print(f"- per-train-seed means: {{{', '.join(f'{ts}: {m:.1f}%' for ts, m in means.items())}}}; std across train seeds {sd(list(means.values())):.1f} pts")
print(f"- within-checkpoint std across eval seeds (pooled) {math.sqrt(ms_eval):.1f} pts; Bernoulli SE n=50 at p={grand / 100:.2f}: {100 * math.sqrt(grand / 100 * (1 - grand / 100) / 50):.1f}")
print(f"- one-way ANOVA F({len(means) - 1},{len(means) * (E - 1)}) = {F:.2f}; train-seed variance component std = {math.sqrt(comp):.1f} pts")
K = sum(final[ts][0] for ts in final); N = sum(final[ts][1] for ts in final)
lo, hi = wilson(K, N)
print(f"- pooled over train seeds at 100k: {K}/{N} = {100 * K / N:.1f}% [{lo:.1f}, {hi:.1f}]; vs reference {REF[0]}/{REF[1]} = {100 * REF[0] / REF[1]:.1f}%: z={ztest(K, N, *REF)[0]:.2f}, p={ztest(K, N, *REF)[1]:.2g}")
for ts in seeds:
    k, n = final[ts]; z, p = ztest(k, n, *REF)
    print(f"  - seed {ts}: {k}/{n} = {100 * k / n:.1f}% vs reference: z={z:.2f}, p={p:.2g}")
print(f"- mean of last-5 checkpoints (60k-100k): {{{', '.join(f'{ts}: {v:.1f}%' for ts, v in last5.items())}}}, mean {sum(last5.values()) / len(last5):.1f}%, std {sd(list(last5.values())):.1f}")
