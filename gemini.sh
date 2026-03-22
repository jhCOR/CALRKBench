#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/elicer/CALRK_Bench/src"

# gemini
python "$SCRIPT_DIR/type1_eval.py"  --model "gemini-3-flash-preview" --reasoning-enabled --reasoning-effort "low" --key "gemini"
python "$SCRIPT_DIR/type2_eval.py"  --model "gemini-3-flash-preview" --reasoning-enabled --reasoning-effort "low" --key "gemini"
python "$SCRIPT_DIR/type3_eval.py"  --model "gemini-3-flash-preview" --reasoning-enabled --reasoning-effort "low" --key "gemini"

# In practice, we conducted our experiments using OpenRouter, but for convenience, we also provide support for the official Gemini API.
#  If you wish to use OpenRouter, please set `--service="openrouter"`.
