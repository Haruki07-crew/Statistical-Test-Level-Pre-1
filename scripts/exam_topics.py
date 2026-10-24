"""全セットの論点（topic）を章ごとに並べて表示する。同じ型の問題がほかのセットにないかを確かめるため。

使い方: .venv/bin/python scripts/exam_topics.py [--chapter N]
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exam_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--chapter", type=int)
a = ap.parse_args()
rows = []
for p in L.set_files():
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        continue
    for pr in d.get("problems") or []:
        for it in pr.get("items") or []:
            rows.append((it.get("chapter") or 0, p.stem, it.get("id"), str(it.get("topic", "")).strip()))
for ch, s, iid, t in sorted(rows):
    if a.chapter and ch != a.chapter:
        continue
    print(f"第{ch:>2}章  {iid:<10} {t}")
