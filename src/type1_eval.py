# -*- coding: utf-8 -*-
import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from models import call_llm
from eval_utils import (
    append_jsonl,
    build_experiment_output_dir,
    load_json_any,
    safe_write_json,
    parse_letter
)

DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_INPUT_PATH = "/home/elicer/CALRK_Bench/dataset/calrk_type1.jsonl"
DEFAULT_OUTPUT_ROOT = "/home/elicer/CALRK_Bench/experiment"

def build_mcq_prompt(sample: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any], str]:

    sid = sample["id"]
    gt = sample.get("answer")
    gt_norm = parse_letter(gt) if isinstance(gt, str) else None

    meta = dict(sample.get("meta") or {})
    for k in ("case_kind", "question_target", "pair_id"):
        if k not in meta and sample.get(k) is not None:
            meta[k] = sample.get(k)
    for k in ("doctrine", "scenario_tag", "anchor", "favor"):
        if k not in meta and sample.get(k) is not None:
            meta[k] = sample.get(k)

    prompt = (sample.get("prompt") or "").strip()
    choices = sample.get("choices") or {}

    if prompt:
        prompt = prompt.strip() + "\n\n정답은 반드시 A/B/C/D 중 하나의 글자만 출력하라. 다른 설명은 출력하지 마라."

    return prompt, gt_norm, meta, sid

@dataclass
class Counter:
    n: int = 0
    correct: int = 0

    def add(self, ok: bool):
        self.n += 1
        if ok:
            self.correct += 1

    def acc(self) -> float:
        return 0.0 if self.n == 0 else self.correct / self.n

def group_key(meta: Dict[str, Any], key: str) -> str:
    v = meta.get(key, "NA")
    return "NA" if v is None else str(v)

def build_report(pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = Counter()
    by_case_kind: Dict[str, Counter] = {}
    by_qtarget: Dict[str, Counter] = {}
    by_kind_qtarget: Dict[str, Counter] = {}
    by_scenario: Dict[str, Counter] = {}

    n_errors = 0
    n_invalid = 0

    for r in pred_rows:
        ok = bool(r.get("correct"))
        overall.add(ok)

        if r.get("status") != "ok":
            n_errors += 1
        if r.get("pred") is None:
            n_invalid += 1

        meta = r.get("meta", {}) or {}

        k_kind = group_key(meta, "case_kind")
        k_tgt = group_key(meta, "question_target")
        kt = f"{k_kind}::{k_tgt}"

        by_case_kind.setdefault(k_kind, Counter()).add(ok)
        by_qtarget.setdefault(k_tgt, Counter()).add(ok)
        by_kind_qtarget.setdefault(kt, Counter()).add(ok)

        k_sce = group_key(meta, "scenario_tag")
        by_scenario.setdefault(k_sce, Counter()).add(ok)

    def pack(d: Dict[str, Counter]) -> Dict[str, Any]:
        out = {}
        for k, c in sorted(d.items(), key=lambda x: (-x[1].n, x[0])):
            out[k] = {
                "n": c.n,
                "correct": c.correct,
                "acc": round(c.acc(), 4),
            }
        return out

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "n": overall.n,
            "correct": overall.correct,
            "acc": round(overall.acc(), 4),
        },
        "errors": {
            "n_errors": n_errors,
            "n_invalid_pred": n_invalid,
        },
        "by_case_kind": pack(by_case_kind),
        "by_question_target": pack(by_qtarget),
        "by_case_kind__question_target": pack(by_kind_qtarget),
        "by_scenario_tag": pack(by_scenario),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT_PATH, help="input json/jsonl (generator dataset jsonl 포함)")
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model name")
    ap.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, help="minimal|low|medium|high|none")
    ap.add_argument("--reasoning-enabled", action="store_true", help="reasoning.enabled=true 전달")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--service", default="none")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--key", type=str, default=None)
    args = ap.parse_args()
    print(args.input, flush=True)

    output_dir = build_experiment_output_dir(
        output_root=args.output_root,
        type_name="Type1",
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        reasoning_enabled=args.reasoning_enabled,
    )
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(pred_path):
        os.remove(pred_path)

    data = load_json_any(args.input)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    ## Evaluation Start
    pred_rows: List[Dict[str, Any]] = []

    total = len(data)
    processed = 0

    pbar = tqdm(data, total=total, desc=f"Evaluating ({args.model})", unit="item")
    for sample in pbar:
        prompt, gt_norm, meta, sid = build_mcq_prompt(sample)
        raw, call_meta = call_llm(
            model=args.model,
            prompt=prompt,
            service=args.service,
            key=args.key,
            reasoning_effort=args.reasoning_effort,
            reasoning_enabled=args.reasoning_enabled,
        )
        pred = parse_letter(raw)
        correct = (pred is not None) and (gt_norm is not None) and (pred == gt_norm)

        row = {
            "id": sid,
            "status": "ok" if call_meta["ok"] else "error",
            "error": call_meta.get("error"),
            "used_reasoning": call_meta.get("used_reasoning"),
            "pred_raw": raw,
            "pred": pred,
            "ground_truth": gt_norm,
            "correct": bool(correct),
            "meta": meta,
        }
        append_jsonl(pred_path, row)
        pred_rows.append(row)
        processed += 1

        if processed % 10 == 0:
            rep = build_report(pred_rows)
            safe_write_json(report_path, rep)

        rep = build_report(pred_rows)
        pbar.set_postfix({
            "acc": rep["overall"]["acc"],
            "processed": processed,
            "errors": rep["errors"]["n_errors"],
            "invalid": rep["errors"]["n_invalid_pred"],
        })

    rep = build_report(pred_rows)
    safe_write_json(report_path, rep)

    print("========================================")
    print(f"Done. outdir = {output_dir}")
    print(f"- model      : {args.model}")
    print(f"- predictions: {pred_path}")
    print(f"- report     : {report_path}")
    print(f"- total input: {total}")
    print(f"- processed  : {processed}")
    print(f"- acc        : {rep['overall']['acc']}")
    print("========================================")

if __name__ == "__main__":
    main()
