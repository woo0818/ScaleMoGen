#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SNAPMOGEN_DIR="${ROOT_DIR}/data/SnapMoGen"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required for SnapMoGen dataset download." >&2
  echo "Install it with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "${SNAPMOGEN_DIR}"

echo "[ScaleMoGen][Prepare] Downloading SnapMoGen dataset to data/SnapMoGen"
huggingface-cli download Ericguo5513/SnapMoGen --repo-type dataset --local-dir "${SNAPMOGEN_DIR}"

cat <<'EOF'

[ScaleMoGen][Prepare] SnapMoGen dataset is ready under data/SnapMoGen.

HumanML3D is not downloaded by this script. Follow the official HumanML3D
instructions, then copy or symlink the processed dataset to:

  data/HumanML3D

Reference: https://github.com/EricGuo5513/HumanML3D
EOF
