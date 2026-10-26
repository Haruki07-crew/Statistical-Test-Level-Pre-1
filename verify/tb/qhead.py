"""教科書の問題の検算スクリプトで共通に使う準備（JD_LEVEL=tb にしてから qcheck を読み込む）。"""
import os
import pathlib
import sys

os.environ.setdefault("JD_LEVEL", "tb")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qcheck import Q  # noqa: E402,F401


def chosen(q, qid):
    """正解の選択肢の文字列。"""
    it = q.qs[qid]
    return str(it["choices"][it["answer"]])
