#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/elicer/CALRK_Bench/src"

# Qwen 
python "$SCRIPT_DIR/type1_eval.py"  --model "qwen/qwen3-30b-a3b-thinking-2507" --reasoning-enabled --service "openrouter" --key "open_router"
python "$SCRIPT_DIR/type2_eval.py"  --model "qwen/qwen3-30b-a3b-thinking-2507" --reasoning-enabled --service "openrouter" --key "open_router"
python "$SCRIPT_DIR/type3_eval.py"  --model "qwen/qwen3-30b-a3b-thinking-2507" --reasoning-enabled --service "openrouter" --key "open_router"