#!/usr/bin/env python3
"""Compare the other-policy kernels (lerobot-train-others-pusht-{cpu,gpu}) with the ACT 10k kernel and the references.

usage: python analyze_others.py kaggle/outputs
"""
import json
import re
import sys
from math import erf, exp, sqrt
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "kaggle/outputs")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def chi2_3seeds(ks, n):
    """chi-square homogeneity of 3 binomial(n) counts (df=2, survival = exp(-x/2))."""
    p = sum(ks) / (3 * n)
    if p in (0, 1):
        return 0.0, 1.0
    stat = sum((k - n * p) ** 2 / (n * p * (1 - p)) for k in ks)
    return stat, exp(-stat / 2)


def load_runs(kernel):
    """{group: {eval_seed: (k, n)}} from a kernel's summary.json (runs without stats are skipped)."""
    s = json.load(open(root / kernel / "summary.json"))
    out = {}
    for name, r in s["runs"].items():
        m = re.fullmatch(r"(.*)_eval(\d+)", name)
        if m and "stats" in r:
            out.setdefault(m.group(1), {})[int(m.group(2))] = (r["stats"]["n_success"], r["stats"]["n_episodes"])
    return out


act = load_runs("lerobot-train-act-pusht")
cpu = load_runs("lerobot-train-others-pusht-cpu")
gpu = load_runs("lerobot-train-others-pusht-gpu")


def pooled(groups):
    k = sum(v[0] for g in groups for v in g.values())
    n = sum(v[1] for g in groups for v in g.values())
    return k, n


ROWS = [  # (label, {seed: (k, n)} or pooled (k, n))
    ("ACT kp 10k, train seed 1", act["kp_train1"]),
    ("ACT kp 10k, train seeds 1-3 pooled", pooled([act["kp_train1"], act["kp_train2"], act["kp_train3"]])),
    ("ACT px 10k, train seed 1", act["px_train1"]),
    ("ACT px 10k, train seeds 1-3 pooled", pooled([act["px_train1"], act["px_train2"], act["px_train3"]])),
    ("TD-MPC kp 10k (offline, CEM planning at eval)", gpu["tdmpc_kp_cpu10k"]),
    ("VQ-BeT px 10k (2k VQ-VAE + 8k policy)", gpu["vqbet_px"]),
    ("Diffusion kp 10k, DDIM-10", gpu["diffusion_kp"]),
    ("Diffusion px 10k, DDPM-100", gpu["diffusion_px"]),
    ("ref lerobot/vqbet_pusht (250k)", cpu["ref_vqbet_pusht"]),
    ("ref lerobot/diffusion_pusht_keypoints (Hydra, converted)", cpu["ref_diffusion_pusht_keypoints"]),
]
print("| policy | seeds 1000/2000/3000 (%) | pooled | Wilson 95% CI | std across seeds (pts) | chi2 homogeneity p |")
print("|---|---|---|---|---|---|")
R = {}
for label, v in ROWS:
    if isinstance(v, dict):
        ks = [v[s][0] for s in sorted(v)]
        n = v[sorted(v)[0]][1]
        k, N = sum(ks), n * len(ks)
        pcs = [100 * x / n for x in ks]
        mean = sum(pcs) / len(pcs)
        std = sqrt(sum((p - mean) ** 2 for p in pcs) / (len(pcs) - 1))
        _, ph = chi2_3seeds(ks, n) if len(ks) == 3 else (None, float("nan"))
        seeds = " / ".join(f"{p:.0f}" for p in pcs)
        extra = f"{std:.1f} | {ph:.2f}"
    else:
        k, N = v
        seeds, extra = "-", "- | -"
    lo, hi = wilson(k, N)
    R[label] = (k, N)
    print(f"| {label} | {seeds} | {k}/{N} = {100 * k / N:.1f}% | [{lo:.1f}, {hi:.1f}] | {extra} |")

print("\nTwo-proportion z-tests")
PAIRS = [
    ("Diffusion kp 10k, DDIM-10", "ACT kp 10k, train seeds 1-3 pooled"),
    ("Diffusion kp 10k, DDIM-10", "ACT kp 10k, train seed 1"),
    ("Diffusion kp 10k, DDIM-10", "ref lerobot/diffusion_pusht_keypoints (Hydra, converted)"),
    ("VQ-BeT px 10k (2k VQ-VAE + 8k policy)", "ACT px 10k, train seeds 1-3 pooled"),
    ("VQ-BeT px 10k (2k VQ-VAE + 8k policy)", "ref lerobot/vqbet_pusht (250k)"),
    ("Diffusion px 10k, DDPM-100", "ACT px 10k, train seeds 1-3 pooled"),
    ("Diffusion px 10k, DDPM-100", "VQ-BeT px 10k (2k VQ-VAE + 8k policy)"),
    ("TD-MPC kp 10k (offline, CEM planning at eval)", "ACT kp 10k, train seeds 1-3 pooled"),
]
for a, b in PAIRS:
    z, p = ztest(*R[a], *R[b])
    print(f"  {a}  vs  {b}: {100 * R[a][0] / R[a][1]:.1f}% vs {100 * R[b][0] / R[b][1]:.1f}%, z={z:+.2f}, p={p:.2g}")
for label, card in [("ref lerobot/vqbet_pusht (250k)", (285, 500)), ("ref lerobot/diffusion_pusht_keypoints (Hydra, converted)", (355, 500))]:
    z, p = ztest(*R[label], *card)
    print(f"  {label} vs model card {100 * card[0] / card[1]:.1f}%/500: z={z:+.2f}, p={p:.2f}")
dp = ztest(*R["Diffusion px 10k, DDPM-100"], 33, 50)
print(f"  Diffusion px 10k vs lerobot/diffusion_pusht 33/50 (earlier kernel): z={dp[0]:+.2f}, p={dp[1]:.2g}")
