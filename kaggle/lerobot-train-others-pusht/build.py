"""Write the cpu and gpu kernel copies of train_others_template.py (Kaggle script kernels take no arguments)."""
import json
from pathlib import Path

here = Path(__file__).parent
tpl = (here / "train_others_template.py").read_text()
MARK = {"DEVICE": 'DEVICE = "cuda"  # replaced by build.py', "ONLY_MODELS": "ONLY_MODELS = None  # replaced by build.py",
        "ONLY_REFS": "ONLY_REFS = None  # replaced by build.py"}
for m in MARK.values():
    assert tpl.count(m) == 1, m
KERNELS = {  # suffix: (device, models, refs, enable_gpu, dataset_sources)
    "cpu": ("cpu", ["tdmpc_kp", "vqbet_px"], ["ref_vqbet_pusht", "ref_diffusion_pusht_keypoints"], False, []),
    "gpu": ("cuda", ["vqbet_px", "diffusion_kp", "diffusion_px"], ["tdmpc_kp_cpu10k"], True, ["cindyxue1122/tdmpc-pusht-kp-10k"]),
}
for suffix, (device, models, refs, gpu, datasets) in KERNELS.items():
    d = here / suffix
    d.mkdir(exist_ok=True)
    name = f"lerobot_train_others_pusht_{suffix}.py"
    src = (tpl.replace(MARK["DEVICE"], f'DEVICE = "{device}"  # written by build.py')
              .replace(MARK["ONLY_MODELS"], f"ONLY_MODELS = {models!r}  # written by build.py")
              .replace(MARK["ONLY_REFS"], f"ONLY_REFS = {refs!r}  # written by build.py"))
    (d / name).write_text(src)
    meta = {
        "id": f"cindyxue1122/lerobot-train-others-pusht-{suffix}",
        "title": f"lerobot-train-others-pusht-{suffix}",
        "code_file": name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": datasets, "competition_sources": [], "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", d / name)
