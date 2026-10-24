"""Robust (resumable, retried) download of bert-base-cased into ./bert-base-cased so that
transformers 4.21's from_pretrained("bert-base-cased") resolves to the local directory when run
from the repo root (its own downloader has no resume and kept failing on this network)."""
import time, os
from huggingface_hub import hf_hub_download
dst = "bert-base-cased"
for fname in ["config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json", "pytorch_model.bin"]:
    for attempt in range(1, 30):
        try:
            p = hf_hub_download("bert-base-cased", fname, local_dir=dst)
            print("ok", p, os.path.getsize(p), flush=True); break
        except Exception as e:
            print(f"[{fname} attempt {attempt}] {type(e).__name__}: {str(e)[:120]}", flush=True); time.sleep(5)
print("BERT_DONE", flush=True)
