#!/usr/bin/env bash
set -euo pipefail
windows_profile="${KHMER_WINDOWS_PROFILE:-${1:-}}"
: "${windows_profile:?Set KHMER_WINDOWS_PROFILE to the Windows user profile containing the Hugging Face cache}"
if [[ "$windows_profile" != /mnt/* ]]; then
  windows_profile="$(wslpath -u "$windows_profile")"
fi
export KHMER_HF_HOME="$windows_profile/.cache/huggingface"
exec bash "$(dirname -- "$0")/run_qwen_wsl.sh" app.py
