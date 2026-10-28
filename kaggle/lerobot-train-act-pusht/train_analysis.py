"""Analyse the lerobot-train-act-pusht kernel outputs: re-parse raw training logs (step from line index), build the
training table, the (training seed x eval seed) success matrix and a one-way variance decomposition per variant."""
import json, math, os, re, sys, glob
root = sys.argv[1]
S = json.load(open(os.path.join(root, "summary.json")))
LOG_RE = re.compile(r"step:(\S+).*?loss:([0-9.eE+-]+).*?grdn:([0-9.eE+-]+|nan|inf).*?lr:([0-9.eE+-]+)")
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h), 100*(c+h)
print("## training")
print("| run | rc | wall (min) | it/s | loss@100 | loss@1k | loss@5k | loss@10k | min loss (step) |")
print("|---|---|---|---|---|---|---|---|---|")
curves = {}
for name, t in S["train"].items():
    pts = []
    for line in open(os.path.join(root, "logs", f"{name}.log"), errors="replace"):
        m = LOG_RE.search(line)
        if m:
            pts.append((100*(len(pts)+1), float(m.group(2))))
    curves[name] = pts
    at = {st: l for st, l in pts}
    mn = min(pts, key=lambda x: x[1]) if pts else (None, float("nan"))
    its = f"{S['meta']['steps']/t['wall_s']:.1f}" if t.get("wall_s") else "-"
    print(f"| {name} | {t.get('rc')} | {t.get('wall_s',0)/60:.1f} | {its} | {at.get(100, float('nan')):.3f} | {at.get(1000, float('nan')):.3f} | {at.get(5000, float('nan')):.3f} | {at.get(10000, float('nan')):.3f} | {mn[1]:.3f} ({mn[0]}) |")
print()
print("## evals")
print("| run | k/n | pc | Wilson CI | avg_max_reward | eval_s |")
print("|---|---|---|---|---|---|")
mat = {}  # variant -> {train_seed: {eval_seed: p}}
for name, r in S["runs"].items():
    s = r.get("stats")
    if not s: print(f"| {name} | FAILED | | | | |"); continue
    print(f"| {name} | {s['n_success']}/{s['n_episodes']} | {s['pc_success']:.1f}% | [{s['ci95_wilson'][0]:.1f}, {s['ci95_wilson'][1]:.1f}] | {s['avg_max_reward']:.3f} | {s['eval_s']:.0f} |")
    if not r.get("repeat"):
        mat.setdefault(r["variant"], {}).setdefault(r["train_seed"], {})[r["seed"]] = s["pc_success"]
print()
print("## matrix + decomposition")
for v, by_ts in mat.items():
    tss = sorted(by_ts); ess = sorted({e for d in by_ts.values() for e in d})
    print(f"### {v}")
    print("| train seed \\ eval seed | " + " | ".join(str(e) for e in ess) + " | mean |")
    print("|---|" + "---|"*(len(ess)+1))
    means = []
    for ts in tss:
        row = [by_ts[ts].get(e, float('nan')) for e in ess]; m = sum(row)/len(row); means.append(m)
        print(f"| {ts} | " + " | ".join(f"{x:.0f}%" for x in row) + f" | {m:.1f}% |")
    allp = [by_ts[ts][e] for ts in tss for e in ess if e in by_ts[ts]]
    gm = sum(allp)/len(allp); T, E = len(tss), len(ess)
    ms_train = E*sum((m-gm)**2 for m in means)/(T-1) if T > 1 else float('nan')
    ms_eval = sum((by_ts[ts][e]-means[i])**2 for i, ts in enumerate(tss) for e in ess if e in by_ts[ts])/(T*(E-1))
    pbar = gm/100; bern = 100*100*pbar*(1-pbar)/50
    var_train = max(0.0, (ms_train-ms_eval)/E) if T > 1 else float('nan')
    print(f"- grand mean {gm:.1f}%; std across training-seed means {math.sqrt(sum((m-gm)**2 for m in means)/(T-1)) if T>1 else float('nan'):.1f} pts; "
          f"within-checkpoint std across eval seeds (pooled) {math.sqrt(ms_eval):.1f} pts; Bernoulli SE for n=50 at p={gm:.0f}%: {math.sqrt(bern):.1f} pts")
    print(f"- one-way ANOVA: MS_train={ms_train:.1f}, MS_eval={ms_eval:.1f} (pts^2); F={ms_train/ms_eval if ms_eval else float('nan'):.2f}; "
          f"estimated training-seed variance component = max(0,(MS_train-MS_eval)/E) = {var_train:.1f} pts^2 -> std {math.sqrt(var_train):.1f} pts")
    K = sum(S['runs'][n]['stats']['n_success'] for n, r in S['runs'].items() if r.get('variant')==v and not r.get('repeat') and r.get('stats'))
    N = sum(S['runs'][n]['stats']['n_episodes'] for n, r in S['runs'].items() if r.get('variant')==v and not r.get('repeat') and r.get('stats'))
    print(f"- pooled {K}/{N} = {100*K/N:.1f}% Wilson [{wilson(K,N)[0]:.1f}, {wilson(K,N)[1]:.1f}]")
print()
print("## determinism")
for n, r in S["runs"].items():
    if r.get("repeat") and r.get("stats"):
        o = S["runs"][r["repeat"]]["stats"]
        print(f"- {n} vs {r['repeat']}: {r['stats']['n_success']} vs {o['n_success']} successes; per-episode identical: {o['successes']==r['stats']['successes']}")
if S.get("errors"): print("\n## errors"); [print("-", e[:300]) for e in S["errors"]]
