"""Embed the proven local_eval scripts into the Kaggle driver (push uploads a single code_file)."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
libero = os.path.join(here, "..", "..", "LIBERO", "local_eval")
embed = {}
for name in ["replay_demos.py", "evaluate_cpu.py", "make_random_ckpt.py", "fetch_bert.py"]:
    with open(os.path.join(libero, name)) as f:
        embed[name] = f.read()
tmpl = open(os.path.join(here, "driver_template.py")).read()
assert "__EMBED__" in tmpl
out = tmpl.replace("__EMBED__", repr(embed))
with open(os.path.join(here, "libero_spatial_eval.py"), "w") as f:
    f.write(out)
print("wrote libero_spatial_eval.py", len(out), "bytes; embedded:", {k: len(v) for k, v in embed.items()})
