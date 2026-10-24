"""content/roadmap.yaml の検査。

使い方:
    .venv/bin/python scripts/check_roadmap.py

検査すること:
- stages が 1〜4 の4つで、title と desc がある
- terms の id がすべて plan.yaml にあり、plan.yaml の全用語をちょうど1回ずつ含む
- stage は 1〜4、score は 0〜100 の整数、minutes は正の整数
- prereq は plan.yaml の id で、リスト中で自分より前にあり、段階が自分以下。3個以内。自分自身を含まない
- 前提の関係に循環がない
- 段階の番号がリストの中で減らない（段階1→2→3→4 の順に並ぶ）
- reason が空でなく、60字以内で、$ とバックスラッシュを含まない
- see（統合先）がある用語は、統合先を prereq に含む
エラーがあれば終了コード1。
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"


def main():
    errors = []
    plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    plan_terms = {t["id"]: t for ch in plan["chapters"] for t in ch["terms"]}
    rm = yaml.safe_load((CONTENT / "roadmap.yaml").read_text(encoding="utf-8"))

    stages = rm.get("stages") or []
    if [s.get("no") for s in stages] != [1, 2, 3, 4]:
        errors.append("stages の no が 1,2,3,4 になっていない")
    for s in stages:
        if not s.get("title") or not s.get("desc"):
            errors.append(f"stage {s.get('no')} に title か desc がない")

    terms = rm.get("terms") or []
    ids = [t.get("id") for t in terms]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        errors.append(f"重複した id: {dup}")
    unknown = [i for i in ids if i not in plan_terms]
    if unknown:
        errors.append(f"plan.yaml にない id: {unknown}")
    missing = [i for i in plan_terms if i not in set(ids)]
    if missing:
        errors.append(f"roadmap にない id（{len(missing)}語）: {missing[:20]}")

    pos = {}
    stage_of = {}
    last_stage = 0
    for i, t in enumerate(terms):
        tid = t.get("id")
        st = t.get("stage")
        if st not in (1, 2, 3, 4):
            errors.append(f"{tid}: stage が不正 {st}")
            continue
        if st < last_stage:
            errors.append(f"{tid}: 段階 {st} が段階 {last_stage} の後にある")
        last_stage = max(last_stage, st)
        sc = t.get("score")
        if not isinstance(sc, int) or not 0 <= sc <= 100:
            errors.append(f"{tid}: score が 0〜100 の整数でない ({sc})")
        mn = t.get("minutes")
        if not isinstance(mn, int) or mn <= 0:
            errors.append(f"{tid}: minutes が正の整数でない ({mn})")
        pre = t.get("prereq")
        if not isinstance(pre, list):
            errors.append(f"{tid}: prereq がリストでない")
            pre = []
        if len(pre) > 3:
            errors.append(f"{tid}: prereq が {len(pre)} 個（3個まで）")
        for p in pre:
            if p == tid:
                errors.append(f"{tid}: 自分自身を前提にしている")
            elif p not in plan_terms:
                errors.append(f"{tid}: 前提 {p} が plan.yaml にない")
            elif p not in pos:
                errors.append(f"{tid}: 前提 {p} が自分より前にない")
            elif stage_of[p] > st:
                errors.append(f"{tid}: 前提 {p} の段階 {stage_of[p]} が自分の段階 {st} より大きい")
        see = plan_terms.get(tid, {}).get("see")
        if see and see not in pre:
            errors.append(f"{tid}: 統合先 {see} が prereq にない")
        reason = t.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{tid}: reason が空")
        else:
            if "$" in reason or "\\" in reason:
                errors.append(f"{tid}: reason に $ かバックスラッシュがある")
            if len(reason) > 60:
                errors.append(f"{tid}: reason が {len(reason)} 字（60字まで）")
        pos[tid] = i
        stage_of[tid] = st

    # 循環の検査（前提が必ず前にあれば循環はありえないが、念のため独立に確かめる）
    graph = {t.get("id"): list(t.get("prereq") or []) for t in terms}
    state = {}

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            errors.append("循環: " + " -> ".join(path + [n]))
            return
        state[n] = 1
        for p in graph.get(n, []):
            visit(p, path + [n])
        state[n] = 2

    sys.setrecursionlimit(10000)
    for n in graph:
        visit(n, [])

    cnt = Counter(t.get("stage") for t in terms)
    print(f"roadmap.yaml: {len(terms)} 語 / plan.yaml: {len(plan_terms)} 語 / 段階別 "
          + ", ".join(f"{s}:{cnt[s]}" for s in sorted(cnt)))
    if errors:
        print(f"エラー {len(errors)} 件")
        for e in errors:
            print("  -", e)
        return 1
    print("OK: エラーなし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
