"""問題の点検：正解の選択肢だけ目立って長い4択、解説での選択肢の位置への言及（選択肢は毎回シャッフルされる）、正誤問題の正解の偏りを一覧にする。

使い方: .venv/bin/python scripts/audit_questions.py
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def plain(s):
    return re.sub(r"\$[^$]*\$", "X", str(s)).strip()


def main():
    long_ok, tf, n_choice, n_tf, pos = [], [0, 0], 0, 0, []
    pos_re = re.compile(r"[1-4１-４一二三四]つ目|[1-4１-４]番目の選択肢|最初の選択肢|最後の選択肢|選択肢\s*[A-DＡ-Ｄ]")
    for p in sorted((ROOT / "content" / "questions").glob("ch*.yaml")):
        for q in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("questions") or []:
            if q.get("type") == "choice":
                n_choice += 1
                if any(pos_re.search(str(q.get(k) or "")) for k in ("explanation", "hint")):
                    pos.append(q["id"])
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
    print(f"4択の解説・ヒントで選択肢の位置に言及しているもの {len(pos)} 問" + (f"：{', '.join(pos)}" if pos else ""))
    print(f"正誤 {n_tf} 問：正しい {tf[1]} / 誤り {tf[0]}")


if __name__ == "__main__":
    main()
