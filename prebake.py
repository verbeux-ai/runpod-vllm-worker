#!/usr/bin/env python3
"""
Baixa modelo HF com retry + heartbeat.

hf_transfer (Rust) tem bug de stall em sockets — ignora timeouts do Python.
Visto duas vezes: trava num arquivo por 10+ min sem retornar erro nem deixar retry rodar.
Solução: DESABILITAR hf_transfer e usar download sequencial padrão.
Trade-off: ~15min vs ~5min, mas previsível.
"""
import os
import sys
import time
import fnmatch
import subprocess
import threading

# DESLIGA hf_transfer — bug de stall conhecido em arquivos grandes
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import huggingface_hub  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = sys.argv[1]
PATTERNS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["*"]

print(f"[prebake] repo={REPO}", flush=True)
print(f"[prebake] patterns={PATTERNS}", flush=True)
print(f"[prebake] huggingface_hub={huggingface_hub.__version__}", flush=True)
print(f"[prebake] hf_transfer DISABLED (avoiding stall bug)", flush=True)

api = HfApi()
info = api.model_info(REPO, files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
selected = [f for f in sizes.keys() if any(fnmatch.fnmatch(f, p) for p in PATTERNS)]
total_bytes = sum(sizes.get(f, 0) for f in selected)
print(f"[prebake] {len(selected)} arquivos / {total_bytes/1e9:.2f}GB total", flush=True)

done = threading.Event()
def heartbeat():
    started = time.time()
    while not done.wait(15):
        try:
            r = subprocess.run(["du", "-sh", "/models"], capture_output=True, text=True, timeout=5)
            print(f"[prebake/hb t={time.time()-started:.0f}s] {r.stdout.strip()}", flush=True)
        except Exception:
            pass

threading.Thread(target=heartbeat, daemon=True).start()

t0 = time.time()
for i, fname in enumerate(selected, 1):
    size_gb = sizes.get(fname, 0) / 1e9
    print(f"[prebake] [{i}/{len(selected)}] {fname} ({size_gb:.2f}GB)", flush=True)
    last_err = None
    for attempt in range(1, 6):
        try:
            f_t0 = time.time()
            hf_hub_download(repo_id=REPO, filename=fname)
            elapsed = time.time() - f_t0
            speed = (sizes.get(fname, 0) / 1e6) / elapsed if elapsed > 0 else 0
            print(f"[prebake]   ✓ {elapsed:.1f}s ({speed:.1f} MB/s)", flush=True)
            break
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[prebake]   ✗ attempt {attempt}/5: {type(e).__name__}: {e}", flush=True)
            time.sleep(wait)
    else:
        done.set()
        print(f"[prebake] FAILED em {fname}: {last_err}", flush=True)
        sys.exit(1)

done.set()
elapsed = time.time() - t0
print(f"[prebake] ✓ DONE em {elapsed:.1f}s (média {total_bytes/1e6/elapsed:.1f} MB/s)", flush=True)
