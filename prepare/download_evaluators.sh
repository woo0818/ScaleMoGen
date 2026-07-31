#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is required. Install it with: pip install gdown" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract evaluator archives." >&2
  exit 1
fi

mkdir -p checkpoint_dir/humanml3d checkpoint_dir/snapmogen

echo "[ScaleMoGen][Prepare] Downloading HumanML3D evaluator assets"
cd "${ROOT_DIR}/checkpoint_dir/humanml3d"
gdown --fuzzy "https://drive.google.com/file/d/1sr73tfFk2O3-IL5brnZnylWi8oIVw_Hw/view?usp=drive_link" -O humanml3d_evaluator.zip
unzip -q -o humanml3d_evaluator.zip
rm -f humanml3d_evaluator.zip

echo "[ScaleMoGen][Prepare] Downloading SnapMoGen evaluator assets"
cd "${ROOT_DIR}/checkpoint_dir/snapmogen"
gdown --fuzzy "https://drive.google.com/file/d/1PfK_X_LuWz5rEZ__SXdUrZr-gxUbgUqc/view?usp=drive_link" -O snapmogen_evaluator.zip
unzip -q -o snapmogen_evaluator.zip
rm -f snapmogen_evaluator.zip

echo "[ScaleMoGen][Prepare] Evaluator assets are ready."
