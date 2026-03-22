#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/home/elicer/CALRK_Bench/src"

# solar
python "$SCRIPT_DIR/type1_eval.py"  --model "solar-pro3-260126"  --reasoning-effort "low" --key "upstage"
python "$SCRIPT_DIR/type2_eval.py"  --model "solar-pro3-260126"  --reasoning-effort "low" --key "upstage"
python "$SCRIPT_DIR/type3_eval.py"  --model "solar-pro3-260126"  --reasoning-effort "low" --key "upstage"
