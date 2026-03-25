#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate midas

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== ZoeDepth ==="
cd "$ROOT_DIR/other_MODELS/ZoeDepth"
python save_preds_ZoeDepth.py

echo "=== MiDaS ==="
cd "$ROOT_DIR/other_MODELS/MiDaS"
python save_preds_MiDaS.py

echo "=== AdaBins ==="
cd "$ROOT_DIR/other_MODELS/AdaBins"
python save_preds_AdaBins.py

conda deactivate

conda activate test

echo "=== DA Relative ==="
cd "$ROOT_DIR"
python save_preds_DA.py

echo "=== DA Metric ==="
cd "$ROOT_DIR/other_MODELS/DA_metric"
python save_preds_DA_metric.py

echo "=== UniDepth ==="
cd "$ROOT_DIR/other_MODELS/UniDepth"
python save_preds_UniDepth.py

echo "=== UniK3D ==="
cd "$ROOT_DIR/other_MODELS/UniK3D"
python save_preds_UniK3D.py

echo "=== All done ==="

conda deactivate
