"""問題の検算用の共通部品。

使い方は spec/QUESTIONS.md を参照。done() で verify/questions_results/chNN.json に
{問題id: {"ok": bool, "hash": 内容のハッシュ}} を保存し、build.py はハッシュが一致して ok の問題だけを公開する。
"""
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from level import CONTENT, VERIFY  # noqa: E402  級ごとの置き場所（JD_LEVEL）
RESULTS = VERIFY / "questions_results"


def qhash(q):
    """問題の内容のハッシュ（検算後に書き換えたら公開されないようにするため）。"""
    body = {k: v for k, v in q.items() if k != "status"}
    s = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class Q:
    def __init__(self, chapter):
        self.chapter = chapter
        path = CONTENT / "questions" / f"ch{chapter:02d}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.qs = {q["id"]: q for q in data.get("questions") or []}
        self.res = {}  # id → [ok, ...]

    def _rec(self, qid, ok, what, value=""):
        if qid not in self.qs:
            print(f"[NG] {qid} / 問題ファイルにない id / {what}")
            self.res.setdefault("__missing__", []).append(False)
            return
        self.res.setdefault(qid, []).append(bool(ok))
        print(f"[{'OK' if ok else 'NG'}] {qid} / {what}" + (f" / 値: {value}" if value != "" else ""))

    def num(self, qid, value, what="数値の正解"):
        q = self.qs.get(qid)
        if q is None or q.get("type") != "numeric":
            self._rec(qid, False, f"{what}（numeric の問題ではない）")
            return
        ans, tol = float(q["answer"]), float(q.get("tolerance", 0))
        v = float(value)
        self._rec(qid, abs(v - ans) <= tol, f"{what} answer={ans} tol={tol}", f"{v:.6g}")

    def check(self, qid, ok, what):
        self._rec(qid, ok, what)

    def concept(self, qid, why="数値を含まない概念問題"):
        self._rec(qid, True, f"概念問題：{why}")

    def done(self):
        ng = sum(1 for v in self.res.values() for ok in v if not ok)
        okc = sum(1 for v in self.res.values() for ok in v if ok)
        unchecked = [i for i in self.qs if i not in self.res]
        out = {}
        for qid, q in self.qs.items():
            r = self.res.get(qid)
            out[qid] = {"ok": bool(r) and all(r), "hash": qhash(q)}
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"ch{self.chapter:02d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        passed = sum(1 for v in out.values() if v["ok"])
        for i in unchecked:
            print(f"[未検算] {i}")
        print(f"第{self.chapter}章 問題の検算: OK {okc} / NG {ng} / 問題 {len(self.qs)} 問中 公開可 {passed} 問 / 未検算 {len(unchecked)} 問")
        sys.exit(1 if ng else 0)
