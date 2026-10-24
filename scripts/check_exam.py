"""模擬試験の自動検査（段階1）と、検算・見直しの状況の一覧。validate.py から呼ばれる。

使い方: .venv/bin/python scripts/check_exam.py [--set N] [--status]
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exam_lib as L  # noqa: E402


def plan_ids():
    plan = yaml.safe_load((L.CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    return {t["id"] for ch in plan["chapters"] for t in ch.get("terms") or []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    ids = plan_ids()
    n_err = 0
    topics = {}
    for path in L.set_files():
        try:
            data, missing = L.load_set(path)
        except Exception as e:  # generator や YAML の誤り
            print(f"[ERROR] {path.name}: 読み込めません: {e}")
            n_err += 1
            continue
        for _, it in L.items_of(data):
            topics.setdefault(str(it.get("topic", "")).strip(), []).append(it.get("id"))
        if a.set and data["set"] != a.set:
            continue
        err, warn = L.check_set(data, missing, ids)
        for e in err:
            print(f"[ERROR] {e}")
        for w in warn:
            print(f"[WARN]  {w}")
        n_err += len(err)
        if a.status:
            st = L.status(data)
            v = sum(1 for x in st.values() if x[0])
            r = sum(1 for x in st.values() if x[1])
            print(f"set{data['set']:02d}: 小問 {len(st)} / 検算OK {v} / 見直しOK {r} / 自動検査エラー {len(err)}")
    for t, its in topics.items():
        if t and len(its) > 1:
            print(f"[ERROR] 同じ論点（topic）が複数の問題にあります: {t}: {its}")
            n_err += 1
    print(f"模擬試験: セット {len(L.set_files())} 件 / 自動検査エラー {n_err} 件")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
