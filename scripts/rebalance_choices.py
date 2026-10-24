"""模擬試験の選択肢を並べ替えて、各セットの正解の位置（①〜⑤）をならす。
誤答の理由（distractors）も同じ順に並べ替え、answer を書き換える。正解以外の選択肢の相対的な順序は保つ。
使い方: .venv/bin/python scripts/rebalance_choices.py [--dry] content/g2/exam_sets/setNN.yaml ...
（並べ替えはセット番号で決まる乱数による。2 回実行すると、1 回目の結果をさらに並べ替えてしまうので、正解をいったん①に置いて書いたセットに 1 回だけ使う）"""
import random
import re
import sys
from pathlib import Path

dry = "--dry" in sys.argv
files = [a for a in sys.argv[1:] if not a.startswith("--")]


def block(lines, start):
    """start の行（'choices:' など）の次から、10 字下げの '- ' で始まる 1 行の項目を集める。"""
    out = []
    i = start + 1
    while i < len(lines) and lines[i].startswith("          - "):
        out.append(i)
        i += 1
    # 続きの行（さらに深い字下げ）があれば 1 行 1 項目ではない
    if i < len(lines) and lines[i].startswith("           ") and not lines[i].startswith("          - "):
        return None
    return out


for fn in files:
    p = Path(fn)
    lines = p.read_text().split("\n")
    setno = int(re.search(r"set(\d+)", p.name).group(1))
    items = []          # (choices_idx, distractors_idx, answer_line_idx, answer)
    i = 0
    while i < len(lines):
        if re.match(r"^      - id: s\d+p\d+[a-z]$", lines[i]):
            j = i + 1
            ch = ds = an = None
            while j < len(lines) and not re.match(r"^(      |  )- id: ", lines[j]):
                if lines[j] == "        choices:":
                    ch = block(lines, j)
                elif lines[j] == "        distractors:":
                    ds = block(lines, j)
                elif lines[j].startswith("        answer: "):
                    an = j
                j += 1
            if ch is not None or ds is not None:
                if not (ch and ds and len(ch) == 5 and len(ds) == 5 and an is not None):
                    print("SKIP", lines[i].strip(), ch and len(ch), ds and len(ds))
                else:
                    items.append((ch, ds, an, int(lines[an].split(":")[1])))
            i = j
        else:
            i += 1
    k = len(items)
    rng = random.Random(1000 + setno)
    while True:
        target = [t % 5 for t in range(k)]
        rng.shuffle(target)
        # 同じ位置が 3 問以上続かないようにする
        if all(not (target[x] == target[x + 1] == target[x + 2]) for x in range(k - 2)):
            break
    before = [a for *_, a in items]
    for (ch, ds, an, a), t in zip(items, target):
        for idx in (ch, ds):
            vals = [lines[x] for x in idx]
            cor = vals.pop(a)
            vals.insert(t, cor)
            for x, v in zip(idx, vals):
                lines[x] = v
        lines[an] = f"        answer: {t}"
    cnt = [target.count(v) for v in range(5)]
    print(p.name, "choice items", k, "before", [before.count(v) for v in range(5)], "after", cnt)
    if not dry:
        p.write_text("\n".join(lines))
