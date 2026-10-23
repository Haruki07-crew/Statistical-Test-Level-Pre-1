"""問題の点検：正解の選択肢だけ目立って長い4択や、正誤問題の正解の偏りを一覧にする。

使い方: .venv/bin/python scripts/audit_questions.py
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def plain(s):
    return re.sub(r"\$[^$]*\$", "X", str(s)).strip()


def main():
    long_ok, tf, n_choice, n_tf = [], [0, 0], 0, 0
    for p in sorted((ROOT / "content" / "questions").glob("ch*.yaml")):
        for q in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("questions") or []:
            if q.get("type") == "choice":
                n_choice += 1
                lens = [len(plain(c)) for c in q["choices"]]
                a = q["answer"]
                others = [l for i, l in enumerate(lens) if i != a]
                if lens[a] >= 12 and lens[a] > 1.5 * max(others):
                    long_ok.append(f"{q['id']}（正解 {lens[a]}字、ほか最大 {max(others)}字）")
            elif q.get("type") == "truefalse":
                n_tf += 1
                tf[1 if q["answer"] else 0] += 1
    print(f"4択 {n_choice} 問のうち、正解だけ長いもの {len(long_ok)} 問")
    for x in long_ok:
        print("  " + x)
    print(f"正誤 {n_tf} 問：正しい {tf[1]} / 誤り {tf[0]}")


if __name__ == "__main__":
    main()
