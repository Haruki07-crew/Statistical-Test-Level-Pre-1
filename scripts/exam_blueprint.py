"""模擬試験の出題配分表 spec/research/exam-blueprint.md を作る。

- 計画：分野（章のまとまり）ごとの問題数は全セット共通（exam_lib.GROUPS）。分野の中の章は、セットごとにずらして順に回す。
- 実績：作ったセットの yaml から、章・難易度・形式の数を数える。
使い方: .venv/bin/python scripts/exam_blueprint.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exam_lib as L  # noqa: E402

N_SETS = 30


def planned(s):
    """セット s（1〜30）の章の割り当て（25 個）。"""
    out = []
    for name, rng, k in L.GROUPS:
        chs = list(rng)
        out += [chs[((s - 1) * k + j) % len(chs)] for j in range(k)]
    return out


def main():
    lines = ["# 模擬試験の出題配分表", "",
             "自動生成（scripts/exam_blueprint.py）。分野ごとの問題数と難易度の割合は全セット共通で、scripts/check_exam.py が検査する。", "",
             "## 分野ごとの問題数（1セット25問）", "", "| 分野 | 章 | 問題数 |", "|---|---|---|"]
    for name, rng, k in L.GROUPS:
        lines.append(f"| {name} | 第{rng[0]}〜{rng[-1]}章 | {k} |")
    lines += ["", f"難易度：基本 {L.LEVELS['基本']}・標準 {L.LEVELS['標準']}・応用 {L.LEVELS['応用']}（およそ 3:5:2）。"
              "形式：5肢選択 17〜19 問・数値入力 6〜8 問。定義を問うだけの問題は 2 問まで。", "",
              "配分の根拠：出題範囲表の大項目と、よく出る分野（spec/research/exam-style.md）。確率分布と回帰は毎回多めに、"
              "確率過程は 1 問とした。分野の中の章はセットごとに順に回し、30 セットで全章が何度も出るようにした。", "",
              "## 計画：セットごとの章の割り当て", "", "| セット | 章（25問） |", "|---|---|"]
    for s in range(1, N_SETS + 1):
        lines.append(f"| {s:02d} | {' '.join(str(c) for c in planned(s))} |")
    lines += ["", "## 実績：作ったセット", "", "| セット | 公開 | 章（数） | 基本/標準/応用 | 5肢/数値 |", "|---|---|---|---|---|"]
    for p in L.set_files():
        data, _ = L.load_set(p)
        its = [it for _, it in L.items_of(data)]
        chs = Counter(it.get("chapter") for it in its)
        lv = Counter(it.get("level") for it in its)
        ty = Counter(it.get("type") for it in its)
        st = L.status(data)
        pub = "○" if st and all(v and r for v, r in st.values()) and len(its) == L.N_ITEMS else "×"
        lines.append(f"| {data['set']:02d} | {pub} | {' '.join(f'{c}:{n}' for c, n in sorted(chs.items()))} | "
                     f"{lv['基本']}/{lv['標準']}/{lv['応用']} | {ty['choice']}/{ty['numeric']} |")
    (L.ROOT / "spec" / "research" / "exam-blueprint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("spec/research/exam-blueprint.md を書きました")


if __name__ == "__main__":
    main()
