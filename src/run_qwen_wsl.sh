#!/usr/bin/env bash
# Run from WSL using the shared local model/data cache and the fixed ICU scorer.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
: "${KHMER_HF_HOME:?Set KHMER_HF_HOME to the WSL path of the existing Hugging Face cache}"
export HF_HOME="$KHMER_HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export KHMER_SCORE_PYTHON="$PWD/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1
exec "${KHMER_PYTHON:-$HOME/khmer-asr-runtime/bin/python}" "$@"
