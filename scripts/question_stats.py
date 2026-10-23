"""章ごとの問題数を Markdown の表で表示する（REPORT_IMPROVE.md 用）。

使い方: .venv/bin/python scripts/question_stats.py
"""
import collections
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["choice", "numeric", "truefalse", "derivation"]
LEVELS = ["基本", "標準", "応用"]


def main():
    plan = yaml.safe_load((ROOT / "content" / "plan.yaml").read_text(encoding="utf-8"))
    titles = {c["no"]: c["title"] for c in plan["chapters"]}
    print("| 章 | 章名 | 問題数 | 4択 | 数値 | 正誤 | 導出 | 基本/標準/応用 | 公開 |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for p in sorted((ROOT / "content" / "questions").glob("ch*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        qs = d["questions"]
        res = json.loads((ROOT / "verify" / "questions_results" / f"{p.stem}.json").read_text(encoding="utf-8"))
        t = collections.Counter(q["type"] for q in qs)
        lv = collections.Counter(q["level"] for q in qs)
        pub = sum(1 for q in qs if res.get(q["id"], {}).get("ok"))
        print(f"| {d['chapter']} | {titles[d['chapter']]} | {len(qs)} | " + " | ".join(str(t[x]) for x in TYPES)
              + f" | {'/'.join(str(lv[x]) for x in LEVELS)} | {pub} |")
        tot["n"] += len(qs); tot["pub"] += pub
        for x in TYPES: tot[x] += t[x]
        for x in LEVELS: tot[x] += lv[x]
    print(f"| 計 | | **{tot['n']}** | " + " | ".join(str(tot[x]) for x in TYPES)
          + f" | {'/'.join(str(tot[x]) for x in LEVELS)} | {tot['pub']} |")


if __name__ == "__main__":
    main()
