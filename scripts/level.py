"""級（準1級・2級）ごとの置き場所。環境変数 JD_LEVEL で切り替える（pre1：準1級＝既定、g2：2級）。

準1級は今までどおり content/・verify/・generator/・docs/index.html、2級は content/g2/・verify/g2/・generator/g2/・docs/g2/index.html。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVEL = os.environ.get("JD_LEVEL", "pre1")
if LEVEL not in ("pre1", "g2"):
    raise SystemExit(f"JD_LEVEL は pre1 か g2: {LEVEL}")


def paths(level):
    if level == "g2":
        return {"content": ROOT / "content" / "g2", "verify": ROOT / "verify" / "g2", "gen": ROOT / "generator" / "g2",
                "out": ROOT / "docs" / "g2" / "index.html"}
    return {"content": ROOT / "content", "verify": ROOT / "verify", "gen": ROOT / "generator", "out": ROOT / "docs" / "index.html"}


P = paths(LEVEL)
CONTENT, VERIFY, GEN, OUT = P["content"], P["verify"], P["gen"], P["out"]
PRE1_CONTENT = ROOT / "content"
NAME = {"pre1": "準1級", "g2": "2級"}[LEVEL]
