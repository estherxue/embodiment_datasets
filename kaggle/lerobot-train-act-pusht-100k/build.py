"""Write the kernel copies of train_100k_template.py (Kaggle script kernels take no arguments)."""
import json
from pathlib import Path

here = Path(__file__).parent
tpl = (here / "train_100k_template.py").read_text()
MARK = {"VARIANT": 'VARIANT = "kp"  # replaced by build.py', "TRAIN_SEED": "TRAIN_SEED = 1  # replaced by build.py",
        "INIT": 'INIT = "ckpt"  # replaced by build.py'}
for m in MARK.values():
    assert tpl.count(m) == 1, m
KERNELS = [  # (variant, train seed, init, dir / id suffix)
    ("kp", 1, "ckpt", "kp"), ("px", 1, "ckpt", "px"),
    ("kp", 2, "scratch", "kp-seed2"), ("kp", 3, "scratch", "kp-seed3"),
]
for v, seed, init, suffix in KERNELS:
    d = here / suffix
    d.mkdir(exist_ok=True)
    name = f"lerobot_train_act_pusht_100k_{suffix.replace('-', '_')}.py"
    src = (tpl.replace(MARK["VARIANT"], f'VARIANT = "{v}"  # written by build.py')
              .replace(MARK["TRAIN_SEED"], f"TRAIN_SEED = {seed}  # written by build.py")
              .replace(MARK["INIT"], f'INIT = "{init}"  # written by build.py'))
    (d / name).write_text(src)
    meta = {
        "id": f"cindyxue1122/lerobot-train-act-pusht-100k-{suffix}",
        "title": f"lerobot-train-act-pusht-100k-{suffix}",
        "code_file": name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["cindyxue1122/lerobot-train-act-pusht"] if init == "ckpt" else [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", d / name)
