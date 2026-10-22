"""PROGRESS.md の「最終更新」と「章ごとの状況」表を plan.yaml から作り直す。"""
import datetime
import re
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent.parent
plan = yaml.safe_load((ROOT / "content/plan.yaml").read_text(encoding="utf-8"))
rows = ["| 章 | 予定 | 作成 | 検算済 | 保留 | 備考 |", "|---|---|---|---|---|---|"]
tot = [0, 0, 0, 0]
for ch in plan["chapters"]:
    ts = ch["terms"]
    w = sum((ROOT / f"content/ch{ch['no']:02d}/{t['id']}.md").exists() for t in ts)
    v = sum(t["status"] == "verified" for t in ts)
    h = sum(t["status"] == "on_hold" for t in ts)
    abc = "/".join(str(sum(t["importance"] == k for t in ts)) for k in "ABC")
    rows.append(f"| {ch['no']} | {len(ts)} | {w} | {v} | {h} | {ch['title']}（A/B/C={abc}） |")
    tot = [tot[0] + len(ts), tot[1] + w, tot[2] + v, tot[3] + h]
rows.append(f"| 計 | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | |")
p = ROOT / "PROGRESS.md"
s = p.read_text(encoding="utf-8")
s = re.sub(r"最終更新: .*", "最終更新: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), s)
s = re.sub(r"(## 章ごとの状況\n).*?(\n## 保留)", lambda m: m.group(1) + "\n".join(rows) + "\n" + m.group(2), s, flags=re.S)
p.write_text(s, encoding="utf-8")
print(f"PROGRESS.md 更新: 予定 {tot[0]} / 作成 {tot[1]} / 検算済 {tot[2]} / 保留 {tot[3]}")
