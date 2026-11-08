"""Compare lerobot/pusht_keypoints revision v1.5 (what the author's code loaded) with v2.1 (what 0.3.3 loads), frame by frame.
usage: compare_v15_v21.py <dir to download v1.5 into> <local v2.1 dataset dir, e.g. ~/.cache/huggingface/lerobot/lerobot/pusht_keypoints>
Result 2026-11-08: all 25,650 frames identical (max |diff| = 0) for environment_state / state / action / timestamp / rewards."""
import glob, json, sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

v15 = snapshot_download("lerobot/pusht_keypoints", repo_type="dataset", revision="v1.5",
                        allow_patterns=["data/*", "meta_data/*"], local_dir=sys.argv[1])
v21_root = sys.argv[2]
print("v1.5 files:", sorted(glob.glob(v15 + "/**/*", recursive=True))[:12])
print("v2.1 root:", v21_root)

def load(files):
    tabs = [pq.read_table(f) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tabs).to_pandas()

a = load(sorted(glob.glob(v15 + "/data/*.parquet")))
b = load(sorted(glob.glob(v21_root + "/data/chunk-*/*.parquet")))
print("rows", len(a), len(b)); print("v1.5 cols", list(a.columns)); print("v2.1 cols", list(b.columns))
a = a.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
b = b.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
for col in ["observation.environment_state", "observation.state", "action", "timestamp", "episode_index", "frame_index", "next.reward", "next.done", "next.success"]:
    if col not in a or col not in b:
        print(col, "missing in", "v1.5" if col not in a else "v2.1"); continue
    x = np.stack(a[col].to_numpy()) if a[col].dtype == object else a[col].to_numpy()
    y = np.stack(b[col].to_numpy()) if b[col].dtype == object else b[col].to_numpy()
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    print(f"{col}: shapes {x.shape} {y.shape}, dtypes {a[col].dtype}/{b[col].dtype}, max|diff| {np.abs(x - y).max() if x.shape == y.shape else 'shape mismatch'}")
# normalization stats are compared by compare_stats.py (v2.1 stores per-episode stats, not a stats.json)
