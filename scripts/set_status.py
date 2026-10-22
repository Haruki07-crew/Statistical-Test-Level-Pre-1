"""plan.yaml の status を更新する。
使い方: .venv/bin/python scripts/set_status.py <status> --chapter N   （その章でファイルがある用語すべて）
        .venv/bin/python scripts/set_status.py <status> id1 id2 ...
"""
import argparse
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("status", choices=["planned", "written", "verified", "on_hold"])
ap.add_argument("ids", nargs="*")
ap.add_argument("--chapter", type=int)
a = ap.parse_args()
plan = yaml.safe_load((ROOT / "content/plan.yaml").read_text(encoding="utf-8"))
n = 0
for ch in plan["chapters"]:
    for t in ch["terms"]:
        exists = (ROOT / f"content/ch{ch['no']:02d}/{t['id']}.md").exists()
        if (a.chapter == ch["no"] and exists) or t["id"] in a.ids:
            t["status"] = a.status
            n += 1
yaml.safe_dump(plan, open(ROOT / "content/plan.yaml", "w"), allow_unicode=True, sort_keys=False, width=200)
print(f"{n} 件を {a.status} にしました")
