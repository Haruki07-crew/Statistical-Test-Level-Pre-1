"""spec/research/chNN.md の用語表を plan.yaml に取り込む（既存の status は保つ）。"""
import re, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
plan = yaml.safe_load((ROOT/"content/plan.yaml").read_text(encoding="utf-8"))
existing = {t["id"]: t for ch in plan["chapters"] for t in ch["terms"]}
for ch in plan["chapters"]:
    p = ROOT/f"spec/research/ch{ch['no']:02d}.md"
    if ch["no"] < 7 or not p.exists():
        continue
    terms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", cells[0]) and cells[3] in ("A","B","C"):
            old = existing.get(cells[0], {})
            terms.append({"id": cells[0], "title": cells[1], "importance": cells[3], "status": old.get("status","planned")})
    ch["terms"] = terms
yaml.safe_dump(plan, open(ROOT/"content/plan.yaml","w"), allow_unicode=True, sort_keys=False, width=200)
