"""Compare the v1.5 normalization stats (meta_data/stats.safetensors) with the stats lerobot 0.3.3 aggregates from the
v2.1 per-episode stats, and with the raw data.  usage: compare_stats.py <v1.5 snapshot dir> <v2.1 dataset dir>
"""
import glob, json, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa
from safetensors.numpy import load_file
v15, v21 = sys.argv[1], sys.argv[2]
st15 = load_file(v15 + "/meta_data/stats.safetensors")
eps = [json.loads(l) for l in open(v21 + "/meta/episodes_stats.jsonl")]
df = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(v21 + "/data/chunk-*/*.parquet"))]).to_pandas()
for key in ["observation.environment_state", "observation.state", "action"]:
    raw = np.stack(df[key].to_numpy()).astype(np.float64)
    agg_min = np.min([np.array(e["stats"][key]["min"]) for e in eps], axis=0)
    agg_max = np.max([np.array(e["stats"][key]["max"]) for e in eps], axis=0)
    counts = np.array([e["stats"][key]["count"][0] for e in eps], dtype=np.float64)
    means = np.stack([np.array(e["stats"][key]["mean"]) for e in eps]); stds = np.stack([np.array(e["stats"][key]["std"]) for e in eps])
    agg_mean = (means * counts[:, None]).sum(0) / counts.sum()
    agg_var = ((stds ** 2 + (means - agg_mean) ** 2) * counts[:, None]).sum(0) / counts.sum()
    for name, o, n, r in [("min", st15[key + "/min"].ravel(), agg_min, raw.min(0)), ("max", st15[key + "/max"].ravel(), agg_max, raw.max(0)),
                          ("mean", st15[key + "/mean"].ravel(), agg_mean, raw.mean(0)), ("std", st15[key + "/std"].ravel(), np.sqrt(agg_var), raw.std(0))]:
        print(f"{key}/{name}: v1.5 vs v2.1-agg max|diff| {np.abs(o - n).max():.4g} | v1.5 vs raw {np.abs(o - r).max():.4g} | v2.1-agg vs raw {np.abs(n - r).max():.4g} | v1.5[:4]={np.round(o[:4], 2)}")
print("v1.5 info:", json.load(open(v15 + "/meta_data/info.json")))
i21 = json.load(open(v21 + "/meta/info.json")); print("v2.1 info:", {k: i21[k] for k in ("codebase_version", "fps", "total_episodes", "total_frames")})
