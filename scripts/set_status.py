"""plan.yaml の status を更新する。
使い方: .venv/bin/python scripts/set_status.py <status> --chapter N   （その章でファイルがある用語すべて）
        .venv/bin/python scripts/set_status.py <status> id1 id2 ...
"""
import argparse
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import CONTENT  # noqa: E402  級ごとの置き場所（JD_LEVEL）
ap = argparse.ArgumentParser()
ap.add_argument("status", choices=["planned", "written", "verified", "on_hold"])
ap.add_argument("ids", nargs="*")
ap.add_argument("--chapter", type=int)
a = ap.parse_args()
plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
n = 0
for ch in plan["chapters"]:
    for t in ch["terms"]:
        exists = (CONTENT / f"ch{ch['no']:02d}/{t['id']}.md").exists()
        if t.get("from"):  # 準1級と共通の用語はそのまま
            continue
        if (a.chapter == ch["no"] and exists) or t["id"] in a.ids:
            t["status"] = a.status
            n += 1
head = "".join(ln for ln in (CONTENT / "plan.yaml").read_text(encoding="utf-8").splitlines(True) if ln.startswith("#"))
(CONTENT / "plan.yaml").write_text(head + yaml.safe_dump(plan, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
print(f"{n} 件を {a.status} にしました")
