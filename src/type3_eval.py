from __future__ import annotations

import argparse
import time
import re
import os
import random
from typing import Any, Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from models import call_llm
from eval_utils import append_jsonl, build_experiment_output_dir, load_json_any, save_json, parse_letter

DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_INPUT_PATH = "/home/elicer/CALRK_Bench/dataset/calrk_type3.jsonl"
DEFAULT_OUTPUT_ROOT = "/home/elicer/CALRK_Bench/experiment"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "--bench-path", dest="input_path", default=DEFAULT_INPUT_PATH)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    ap.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, help="minimal|low|medium|high|none")
    ap.add_argument("--reasoning-enabled", action="store_true", help="reasoning.enabled=true 전달")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--service", default="none")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--key", type=str, default=None)
    args = ap.parse_args()
    print(args.input_path, flush=True)

    output_dir = build_experiment_output_dir(
        output_root=args.output_root,
        type_name="Type3",
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        reasoning_enabled=args.reasoning_enabled,
    )
    out_jsonl = os.path.join(output_dir, "predictions.jsonl")
    out_summary = os.path.join(output_dir, "summary.json")

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    bench: List[Dict[str, Any]] = load_json_any(args.input_path)
    if args.shuffle:
        if args.seed is not None:
            random.Random(args.seed).shuffle(bench)
        else:
            random.shuffle(bench)
    if args.limit and args.limit > 0:
        bench = bench[:args.limit]

    ## Evaluation Start
    total, correct, invalid = 0, 0, 0
    n_errors = 0
    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)

    pbar = tqdm(bench, desc=f"Eval Type3 on {args.model}")
    for item in pbar:
        total += 1
        
        prompt = (item.get("prompt") or "").strip()
        raw, call_meta = call_llm(
            model=args.model,
            prompt=prompt,
            service=args.service,
            key=args.key,
            reasoning_effort=args.reasoning_effort,
            reasoning_enabled=args.reasoning_enabled,
        )
        pred = parse_letter(raw)
        gold = (item.get("answer") or "").strip().upper()
        ok = (pred == gold)
        
        if pred is None: invalid += 1
        if not call_meta["ok"]: n_errors += 1
        if ok: correct += 1

        category = gold
        cat_total[category] += 1
        if ok: cat_correct[category] += 1

        question_data = item.get("question", {})
        choices = question_data.get("choices", {})

        if total == 1:
            append_jsonl(out_jsonl, {
            "choices": choices,
        })

        append_jsonl(out_jsonl, {
            "id": item.get("id"),
            "case_name": item.get("case_name"),
            "gold": gold,
            "pred": pred,
            "is_correct": ok,
            "raw": raw,
            "status": "ok" if call_meta["ok"] else "error",
            "error": call_meta["error"],
            "attempts": call_meta["attempts"],
            "used_reasoning": call_meta.get("used_reasoning"),
            "reasoning_fallback": call_meta.get("fallback_without_reasoning"),
        })

        acc = correct / total if total else 0.0
        pbar.set_postfix(acc=f"{acc:.3f}", inv=f"{invalid/total:.2f}")

        if args.sleep:
            time.sleep(args.sleep)

    summary = {
        "model": args.model,
        "bench_path": args.input_path,
        "total_n": total,
        "total_accuracy": (correct / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
        "errors": {
            "n_errors": n_errors,
        },
        "reasoning": {
            "enabled": args.reasoning_enabled,
            "effort": args.reasoning_effort,
        },
        "by_type": {
            c: {"n": cat_total[c], "acc": cat_correct[c]/cat_total[c]} 
            for c in sorted(cat_total.keys())
        }
    }

    save_json(out_summary, summary)
    print("\n" + "="*30)
    print(f"Eval Done: Acc {summary['total_accuracy']:.2%}")
    print(f"Results: {out_jsonl}")

if __name__ == "__main__":
    main()
