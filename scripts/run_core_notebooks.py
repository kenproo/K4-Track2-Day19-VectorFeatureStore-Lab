"""Run core notebooks 01 to 04 sequentially with nbconvert."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

notebooks = [
    "01_embeddings_index.ipynb",
    "02_hybrid_search_rrf.ipynb",
    "03_search_api_benchmark.ipynb",
    "04_feast_feature_store.ipynb",
]

jupyter = str(ROOT / ".venv" / "Scripts" / "jupyter.exe")
env = os.environ.copy()
env["PYTHONUTF8"] = "1"

print("=" * 60)
print("Executing 4 Core Notebooks (NB1 - NB4)")
print("=" * 60)

for nb in notebooks:
    nb_path = ROOT / "notebooks" / nb
    print(f"\n[RUNNING] {nb} ...", flush=True)
    t0 = time.perf_counter()
    cmd = [
        jupyter,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(nb_path),
        "--ExecutePreprocessor.timeout=600",
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if res.returncode == 0:
        print(f"[SUCCESS] {nb} finished in {elapsed:.1f}s")
    else:
        print(f"[FAILED]  {nb} failed in {elapsed:.1f}s")
        print("STDERR:")
        print(res.stderr[-1000:])
        sys.exit(1)

print("\n" + "=" * 60)
print("All 4 Core Notebooks executed successfully with outputs preserved!")
print("=" * 60)
