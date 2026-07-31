#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is required. Install it with: pip install gdown" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract GloVe assets." >&2
  exit 1
fi

echo "[ScaleMoGen][Prepare] Downloading GloVe assets used by HumanML3D evaluators"
gdown --fuzzy "https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing" -O glove.zip
rm -rf glove
unzip -q -o glove.zip
rm -f glove.zip

echo "[ScaleMoGen][Prepare] GloVe assets are ready under glove/."
