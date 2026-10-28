"""Write the kp and px kernel copies of train_100k_template.py (Kaggle script kernels take no arguments)."""
import json
from pathlib import Path

here = Path(__file__).parent
tpl = (here / "train_100k_template.py").read_text()
assert tpl.count('VARIANT = "kp"  # replaced by build.py') == 1
for v in ("kp", "px"):
    d = here / v
    d.mkdir(exist_ok=True)
    name = f"lerobot_train_act_pusht_100k_{v}.py"
    (d / name).write_text(tpl.replace('VARIANT = "kp"  # replaced by build.py', f'VARIANT = "{v}"  # written by build.py'))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"cindyxue1122/lerobot-train-act-pusht-100k-{v}",
        "title": f"lerobot-train-act-pusht-100k-{v}",
        "code_file": name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": ["cindyxue1122/lerobot-train-act-pusht"],
    }, indent=2) + "\n")
    print("wrote", d / name)
