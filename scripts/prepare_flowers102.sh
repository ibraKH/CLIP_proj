#!/usr/bin/env bash
set -e
echo "[*] Preparing Flowers-102 via torchvision auto-download."
echo "This script simply runs a tiny Python snippet to trigger the download cache."
python - <<'PY'
from torchvision.datasets import Flowers102
from pathlib import Path
root = Path("data/flowers102")
root.mkdir(parents=True, exist_ok=True)
for split in ["train","val","test"]:
    Flowers102(root=str(root), split=split, download=True)
print("Downloaded to", root.resolve())
PY
echo "Done. If you hit a network/SSL issue, download manually and place files under data/flowers102."
