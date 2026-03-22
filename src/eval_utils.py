from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict

def load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    save_json(tmp, obj)
    os.replace(tmp, path)

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _slugify_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "na"

LETTER_RE = re.compile(r"\b([ABCD])\b")
def parse_letter(text: str) -> Optional[str]:
    if not text: return None
    t = text.strip().upper()
    if t in ("A", "B", "C", "D"): return t
    m = LETTER_RE.search(t)
    if m: return m.group(1)
    return None

def build_experiment_output_dir(
    output_root: str,
    type_name: str,
    model: str,
    reasoning_effort: str,
    reasoning_enabled: bool,
) -> str:
    model_token = _slugify_token(model)
    reasoning_mode = "on" if reasoning_enabled else "off"
    reasoning_token = _slugify_token(f"{reasoning_mode}-{reasoning_effort}")
    date_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{model_token}__reasoning-{reasoning_token}__{date_token}"
    run_dir = os.path.join(output_root, type_name, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
