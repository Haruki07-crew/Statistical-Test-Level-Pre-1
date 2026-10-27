"""教科書（章ごとの本文）の検算の共通部品。verify/<級>/book/chNN.py から使う。

    from bookcheck import Book
    b = Book("g2", 1)                 # content/g2/book/ch01.md を読む
    b.num("標本平均", 52.375, "52.4")  # 計算した値が、本文に書いた「52.4」に丸めた値と合い、本文に「52.4」があるか
    b.check("中央値は平均より小さい", med < mean, med)
    b.done()

num は、書いた数字の桁から許容差を決める（「52.4」なら ±0.05。tol を渡せばそれを使う）。
本文に数字が出てくるかは、数式の中も含めて文字列で探す（"52.4" が "152.43" の一部として見つかる誤りを避けるため、前後が数字や小数点でないことも確かめる）。
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"pre1": ROOT / "content", "g2": ROOT / "content" / "g2", "tb": ROOT / "content" / "tb"}


class Book:
    def __init__(self, level, no):
        self.path = DIRS[level] / "book" / f"ch{no:02d}.md"
        text = self.path.read_text(encoding="utf-8")
        end = text.find("\n---\n", 4)
        self.fm = yaml.safe_load(text[4:end])
        self.body = text[end + 5:]
        self.name = f"{level} 第{no}章"
        self.results = []

    def check(self, what, ok, value=None):
        self.results.append(bool(ok))
        print(f"[{'OK' if ok else 'NG'}] {what} / 値: {value}")
        return ok

    def has(self, s):
        return re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", self.body) is not None

    def num(self, what, value, shown, tol=None):
        """value を計算した値、shown を本文に書いた数字（文字列）として照合する。"""
        s = shown.replace(",", "").replace("−", "-").rstrip("%")
        d = len(s.split(".")[1]) if "." in s else 0
        t = tol if tol is not None else 0.5 * 10 ** -d + 1e-9
        scale = 100 if shown.endswith("%") else 1
        ok = abs(float(value) * scale - float(s)) <= t
        found = self.has(shown)
        return self.check(f"{what}（本文「{shown}」）" + ("" if found else " ※本文に見つからない"), ok and found, value)

    def chart(self, i=1):
        return self.fm["charts"][i - 1]["params"]

    def done(self):
        ng = self.results.count(False)
        print(f"{self.name}の検算: OK {len(self.results) - ng} / NG {ng}")
        sys.exit(1 if ng or not self.results else 0)
