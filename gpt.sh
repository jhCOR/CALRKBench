#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/elicer/CALRK_Bench/src"

# gpt
python "$SCRIPT_DIR/type1_eval.py"  --model "gpt-5-2025-08-07" --reasoning-enabled --reasoning-effort "low" --key "openai_key"
python "$SCRIPT_DIR/type2_eval.py"  --model "gpt-5-2025-08-07" --reasoning-enabled --reasoning-effort "low" --key "openai_key"
python "$SCRIPT_DIR/type3_eval.py"  --model "gpt-5-2025-08-07" --reasoning-enabled --reasoning-effort "low" --key "openai_key"

# In practice, we conducted our experiments using OpenRouter, but for convenience, we also provide support for the official GPT API.
#  If you wish to use OpenRouter, please set `--service="openrouter"`.