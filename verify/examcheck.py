"""模擬試験の検算（段階2）の共通部品。verify/exam_setNN.py から使う。

E = Exam(1) で set01 を読み込み（generator の値を差し込んだ状態）、
  E.num(小問id, 別の方法で求めた値)        … 数値入力の正解と一致するか（許容範囲の 1/10 以内）
  E.choice(小問id, 別の方法で求めた値の文字列) … 正解の選択肢にその文字列があり、他の選択肢にはないか
  E.check(小問id, 条件, 説明)              … 文章の選択肢など、条件が成り立つか
  E.done()                                … verify/exam_results/setNN.json に {小問id: {ok, hash}} を保存
すべての小問を少なくとも1回検査しないと ok にならない。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import exam_lib as L  # noqa: E402


class Exam:
    def __init__(self, no):
        self.no = no
        self.data, missing = L.load_set(L.SETS / f"set{no:02d}.yaml")
        if missing:
            print(f"[NG] generator にない «»: {sorted(missing)}")
        self.items = {it["id"]: (p, it) for p, it in L.items_of(self.data)}
        self.res = {}

    def _rec(self, iid, ok, what, value=""):
        if iid not in self.items:
            print(f"[NG] {iid} / 問題ファイルにない id / {what}")
            self.res.setdefault("__missing__", []).append(False)
            return
        self.res.setdefault(iid, []).append(bool(ok))
        print(f"[{'OK' if ok else 'NG'}] {iid} / {what}" + (f" / 値: {value}" if value != "" else ""))

    def item(self, iid):
        return self.items[iid][1]

    def num(self, iid, value, what="数値の正解（別の方法）"):
        it = self.item(iid)
        tol = it.get("tolerance") or 0
        ok = it.get("type") == "numeric" and abs(float(it["answer"]) - float(value)) <= tol / 10 + 1e-12
        self._rec(iid, ok, f"{what}: 正解 {it.get('answer')}", round(float(value), 10))

    def choice(self, iid, text, what="正解の選択肢（別の方法）"):
        it = self.item(iid)
        norm = lambda s: re.sub(r"\s+", "", str(s))
        hits = [i for i, c in enumerate(it.get("choices") or []) if norm(text) in norm(c)]
        self._rec(iid, hits == [it.get("answer")], f"{what}: 「{text}」を含む選択肢 {hits}／正解 {it.get('answer')}")

    def check(self, iid, cond, what):
        self._rec(iid, cond, what)

    def done(self):
        out, bad = {}, 0
        for iid, (p, it) in self.items.items():
            r = self.res.get(iid, [])
            ok = bool(r) and all(r)
            if not ok:
                bad += 1
                print(f"[NG] {iid} / " + ("検算していない" if not r else "NG がある"))
            out[iid] = {"ok": ok, "hash": L.item_hash(p, it)}
        L.RESULTS.mkdir(parents=True, exist_ok=True)
        (L.RESULTS / f"set{self.no:02d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"set{self.no:02d}: 検算 OK {len(out) - bad} / NG {bad}")
        if "__missing__" in self.res:
            bad += 1
        sys.exit(1 if bad else 0)
