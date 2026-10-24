"""2級の問題の検算スクリプトで共通に使う準備（JD_LEVEL=g2 にしてから qcheck を読み込む）。"""
import os
import sys
import pathlib
os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qcheck import Q  # noqa: E402,F401


def chosen(q, qid):
    """正解の選択肢の文字列。"""
    it = q.qs[qid]
    return str(it["choices"][it["answer"]])
