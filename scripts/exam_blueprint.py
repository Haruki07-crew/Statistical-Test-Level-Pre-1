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

N_SETS = 30 if L.LEVEL == "pre1" else 10
OUT = L.ROOT / "spec" / "research" / ("exam-blueprint.md" if L.LEVEL == "pre1" else "g2/exam-blueprint.md")


def planned(s):
    """セット s の章の割り当て（N_ITEMS 個）。"""
    out = []
    for name, rng, k in L.GROUPS:
        chs = list(rng)
        out += [chs[((s - 1) * k + j) % len(chs)] for j in range(k)]
    return out


def main():
    lines = ["# 模擬試験の出題配分表", "",
             "自動生成（scripts/exam_blueprint.py）。分野ごとの問題数と難易度の割合は全セット共通で、scripts/check_exam.py が検査する。", "",
             f"## 分野ごとの問題数（1セット{L.N_ITEMS}問）", "", "| 分野 | 章 | 問題数 |", "|---|---|---|"]
    for name, rng, k in L.GROUPS:
        lines.append(f"| {name} | 第{rng[0]}〜{rng[-1]}章 | {k} |")
    if L.LEVEL == "pre1":
        lines += ["", f"難易度：基本 {L.LEVELS['基本']}・標準 {L.LEVELS['標準']}・応用 {L.LEVELS['応用']}（およそ 3:5:2）。"
                  "形式：5肢選択 17〜19 問・数値入力 6〜8 問。定義を問うだけの問題は 2 問まで。", "",
                  "配分の根拠：出題範囲表の大項目と、よく出る分野（spec/research/exam-style.md）。確率分布と回帰は毎回多めに、"
                  "確率過程は 1 問とした。分野の中の章はセットごとに順に回し、30 セットで全章が何度も出るようにした。", ""]
    else:
        lines += ["", f"難易度：基本 {L.LEVELS['基本']}・標準 {L.LEVELS['標準']}・応用 {L.LEVELS['応用']}。"
                  f"形式：すべて5肢選択（本番は4〜5肢）。定義を問うだけの問題は {L.MAX_DEF} 問まで。", "",
                  "配分の根拠：公式の出題範囲表（spec/research/g2/format.md）。記述統計とデータの集め方がおよそ3分の1、"
                  "確率と標本分布がおよそ4分の1、推定・検定がおよそ3割、回帰と分散分析が残り。", ""]
    lines += ["## 計画：セットごとの章の割り当て", "", f"| セット | 章（{L.N_ITEMS}問） |", "|---|---|"]
    for s in range(1, N_SETS + 1):
        lines.append(f"| {s:02d} | {' '.join(str(c) for c in planned(s))} |")
    lines += ["", "## 実績：作ったセット", "", "| セット | 公開 | 章（数） | 基本/標準/応用 | 5肢/数値 |", "|---|---|---|---|---|"]
    for p in L.set_files():
        try:
            data, _ = L.load_set(p)
        except Exception:  # 書きかけのセットは数えない
            lines.append(f"| {p.stem[3:]} | × | （作成中） | | |")
            continue
        its = [it for _, it in L.items_of(data)]
        chs = Counter(it.get("chapter") for it in its)
        lv = Counter(it.get("level") for it in its)
        ty = Counter(it.get("type") for it in its)
        st = L.status(data)
        pub = "○" if st and all(v and r for v, r in st.values()) and len(its) == L.N_ITEMS else "×"
        lines.append(f"| {data['set']:02d} | {pub} | {' '.join(f'{c}:{n}' for c, n in sorted(chs.items()))} | "
                     f"{lv['基本']}/{lv['標準']}/{lv['応用']} | {ty['choice']}/{ty['numeric']} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(L.ROOT)} を書きました")


if __name__ == "__main__":
    main()
