"""教科書（修士の章）の検算の共通部品。

check(term, what, ok, value) で結果を記録し、done(name) で OK/NG を数えて終了コードを返す。
chart(term, i) は用語ページの charts の i 番目の params を返す（図の値を検算した値と照合するため）。
body(term) は用語ページの本文を返す（本文に書いた数値がそのまま入っているかを確かめるため）。
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "content" / "tb"
results = []


def _page(term):
    p = next(CONTENT.glob(f"ch*/{term}.md"))
    text = p.read_text(encoding="utf-8")
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end]), text[end + 5:]


def chart(term, i=0):
    return _page(term)[0]["charts"][i]["params"]


def body(term):
    return _page(term)[1]


def check(term, what, ok, value=None):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-4):
    return abs(float(x) - float(claim)) <= tol


def allclose(xs, ys, tol=5e-4):
    return len(xs) == len(ys) and all(close(a, b, tol) for a, b in zip(xs, ys))


def done(name):
    ng = results.count(False)
    print(f"{name}の検算: OK {len(results) - ng} / NG {ng}")
    sys.exit(1 if ng else 0)
