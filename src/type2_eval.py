from __future__ import annotations

import argparse
import json
import os
import time
import re
import random
from typing import Any, Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from models import call_llm
from eval_utils import append_jsonl, build_experiment_output_dir, load_json_any, save_json, parse_letter

DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_INPUT_PATH = "/home/elicer/CALRK_Bench/dataset/calrk_type2.jsonl"
DEFAULT_OUTPUT_ROOT = "/home/elicer/CALRK_Bench/experiment"

ANSWER_MARKER_RE = re.compile(
    r"(?:최종\s*답(?:변)?|정답|답|FINAL\s*ANSWER|ANSWER)\s*[:：\-]?\s*\(?\s*([ABCD])\s*\)?",
    re.IGNORECASE,
)
SINGLE_LETTER_LINE_RE = re.compile(r"^\s*\(?\s*([ABCD])\s*\)?\s*[.)]?\s*$", re.IGNORECASE)
NO_NEED_PREFIX = "추가 법률 참조 없음"

def build_prompt(item: Dict[str, Any], prompt_style:str) -> str:

    q = (item.get("question") or "").strip()
    instr = (item.get("instruction") or "").strip()
    choices = item.get("choices") or {}

    opt_lines = []
    for k in ["A", "B", "C", "D"]:
        opt_lines.append(f"{k}) {choices.get(k, '').strip()}")

    post_fix_instruction = None
    if prompt_style == "CoT":
        post_fix_instruction = '''
            - 차근 차근 단계 별로 생각하고, 답은 마지막 줄에서 '답: (정답)'의 형태로 응답하라
        '''
    else:
        post_fix_instruction = '''
            - 정답은 반드시 A/B/C/D 중 하나의 글자만 출력하라.
            - 다른 설명, 공백, 문장부호를 출력하지 마라.
        '''
    prompt = f"""
        {instr}

        [질문]
        {q}

        [보기]
        {chr(10).join(opt_lines)}

        {post_fix_instruction}
        """.strip()
    return prompt

def parse_letter_cot(text: str) -> Optional[str]:

    if not text:
        return None

    t = text.strip()
    if not t:
        return None

    only = t.upper()
    if only in ("A", "B", "C", "D"):
        return only

    marker_matches = list(ANSWER_MARKER_RE.finditer(t))
    if marker_matches:
        return marker_matches[-1].group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines[-5:]):
        m = SINGLE_LETTER_LINE_RE.match(ln)
        if m:
            return m.group(1).upper()

    return None

def infer_no_need_pos(item: Dict[str, Any]) -> Optional[str]:
    meta = item.get("meta") or {}
    pos = str(meta.get("no_need_pos") or "").strip().upper()
    if pos in ("A", "B", "C", "D"):
        return pos

    choices = item.get("choices") or {}
    for k in ("A", "B", "C", "D"):
        v = choices.get(k, "")
        if isinstance(v, str) and v.startswith(NO_NEED_PREFIX):
            return k
    return None

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
    ap.add_argument("--cot", action="store_true", help="CoT 프롬프트와 CoT 전용 답안 파서를 사용")
    args = ap.parse_args()

    letter_parser = parse_letter_cot if args.cot else parse_letter

    output_dir = build_experiment_output_dir(
        output_root=args.output_root,
        type_name="Type2",
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
    total = 0
    correct = 0
    invalid = 0
    n_errors = 0

    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)
    cat_invalid = defaultdict(int)

    pred_no_need_all = 0
    need_case_total = 0
    need_case_pred_no_need = 0

    mode_label = "CoT" if args.cot else "plain"
    pbar = tqdm(bench, desc=f"Eval Type2 ({mode_label}) on {args.model}")
    for item in pbar:
        total += 1

        prompt = build_prompt(item, mode_label)
        raw, call_meta = call_llm(
            model=args.model,
            prompt=prompt,
            service=args.service,
            key=args.key,
            reasoning_effort=args.reasoning_effort,
            reasoning_enabled=args.reasoning_enabled,
        )
        pred = letter_parser(raw)
        gold = (item.get("answer") or "").strip().upper()
        ok = (pred == gold)

        if pred is None:
            invalid += 1

        if not call_meta["ok"]:
            n_errors += 1

        if ok:
            correct += 1

        meta = item.get("meta") or {}
        category = (meta.get("category") or item.get("category") or "UNKNOWN").strip() or "UNKNOWN"
        no_need_pos = infer_no_need_pos(item)
        pred_is_no_need = (
            pred is not None and
            no_need_pos is not None and
            pred == no_need_pos
        )
        if pred_is_no_need:
            pred_no_need_all += 1
        if category in ("need_law", "need_law_partial"):
            need_case_total += 1
            if pred_is_no_need:
                need_case_pred_no_need += 1

        cat_total[category] += 1
        if ok:
            cat_correct[category] += 1
        if pred is None:
            cat_invalid[category] += 1

        append_jsonl(out_jsonl, {
            "id": item.get("id"),
            "type": item.get("type"),
            "category": category,
            "gold": gold,
            "pred": pred,
            "is_correct": ok,
            "raw": raw,
            "status": "ok" if call_meta["ok"] else "error",
            "error": call_meta["error"],
            "attempts": call_meta["attempts"],
            "used_reasoning": call_meta.get("used_reasoning"),
            "reasoning_fallback": call_meta.get("fallback_without_reasoning"),
            "no_need_pos": no_need_pos,
            "pred_is_no_need": pred_is_no_need,
            "cot": args.cot,
        })

        acc = correct / total if total else 0.0
        inv_rate = invalid / total if total else 0.0
        pbar.set_postfix(acc=f"{acc:.3f}", invalid=f"{inv_rate:.3f}")

        if args.sleep:
            time.sleep(args.sleep)

    summary = {
        "model": args.model,
        "bench_path": args.input_path,
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
        "errors": {
            "n_errors": n_errors,
        },
        "reasoning": {
            "enabled": args.reasoning_enabled,
            "effort": args.reasoning_effort,
        },
        "cot": args.cot,
        "by_category": {},
        "additional": {
            "pred_no_need_rate_all": (pred_no_need_all / total) if total else 0.0,
            "pred_eq_no_need_pos_error_rate_when_need": (
                need_case_pred_no_need / need_case_total
            ) if need_case_total else 0.0,
            "need_case_n": need_case_total,
        },
    }

    for c in sorted(cat_total.keys()):
        n = cat_total[c]
        summary["by_category"][c] = {
            "n": n,
            "accuracy": (cat_correct[c] / n) if n else 0.0,
            "invalid_rate": (cat_invalid[c] / n) if n else 0.0,
        }

    save_json(out_summary, summary)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preds saved: {out_jsonl}")
    print(f"Summary saved: {out_summary}")


if __name__ == "__main__":
    main()
