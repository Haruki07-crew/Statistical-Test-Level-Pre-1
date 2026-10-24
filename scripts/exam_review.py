"""見直し（段階3）の記録。math-checker の指摘をすべて直したあとで実行し、そのときの内容のハッシュを保存する。

使い方: .venv/bin/python scripts/exam_review.py --set N [--items id1,id2]   （省略時はセットの全小問）
見直しの指摘と直した内容は verify/exam_review/setNN.md に書く。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exam_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", type=int, required=True)
ap.add_argument("--items")
a = ap.parse_args()
data, missing = L.load_set(L.SETS / f"set{a.set:02d}.yaml")
if missing:
    sys.exit(f"generator にない «»: {missing}")
want = set(a.items.split(",")) if a.items else None
p = L.REVIEW / f"set{a.set:02d}.json"
rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
n = 0
for pr, it in L.items_of(data):
    if want is None or it["id"] in want:
        rec[it["id"]] = L.item_hash(pr, it)
        n += 1
L.REVIEW.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"set{a.set:02d}: 見直し済みとして {n} 問を記録")
