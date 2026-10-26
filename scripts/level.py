"""級（準1級・2級・教科書）ごとの置き場所。環境変数 JD_LEVEL で切り替える（pre1：準1級＝既定、g2：2級、tb：統計学の教科書）。

準1級は今までどおり content/・verify/・generator/・docs/index.html、2級は content/g2/・verify/g2/・generator/g2/・docs/g2/index.html、
教科書（学部1年〜修士2年）は content/tb/・verify/tb/・docs/tb/index.html。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVEL = os.environ.get("JD_LEVEL", "pre1")
LEVELS = ("pre1", "g2", "tb")
if LEVEL not in LEVELS:
    raise SystemExit(f"JD_LEVEL は pre1・g2・tb のどれか: {LEVEL}")


def paths(level):
    if level in ("g2", "tb"):
        return {"content": ROOT / "content" / level, "verify": ROOT / "verify" / level, "gen": ROOT / "generator" / level,
                "out": ROOT / "docs" / level / "index.html"}
    return {"content": ROOT / "content", "verify": ROOT / "verify", "gen": ROOT / "generator", "out": ROOT / "docs" / "index.html"}


P = paths(LEVEL)
CONTENT, VERIFY, GEN, OUT = P["content"], P["verify"], P["gen"], P["out"]
PRE1_CONTENT = ROOT / "content"
NAME = {"pre1": "準1級", "g2": "2級", "tb": "教科書"}[LEVEL]
