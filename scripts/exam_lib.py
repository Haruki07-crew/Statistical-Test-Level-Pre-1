"""模擬試験（content/exam_sets/setNN.yaml）の読み込み・値の差し込み・自動検査。形式は spec/EXAM.md。

- 問題文などの «名前» は generator/setNN.py の values() が返す値で置き換える（数値を手で書かないため）。
- 自動検査（段階1）：必須項目、5肢選択の選択肢が5つで重複なし、正解が1つ、数値入力の丸め方と許容範囲、
  «» の未定義、用語 id の存在、セットの章配分・難易度の割合。
- 検算（段階2）は verify/exam_setNN.py → verify/exam_results/setNN.json、
  見直し（段階3）は scripts/exam_review.py → verify/exam_review/setNN.json。どちらも問題の内容のハッシュで記録する。
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import CONTENT, GEN, LEVEL, ROOT, VERIFY  # noqa: E402  級ごとの置き場所（JD_LEVEL）

SETS = CONTENT / "exam_sets"
RESULTS = VERIFY / "exam_results"
REVIEW = VERIFY / "exam_review"

MINUTES = 90
if LEVEL == "g2":
    # 2級：35問・90分・選択式だけ（spec/research/g2/format.md、配分は spec/research/g2/exam-blueprint.md）
    N_ITEMS = 35
    LEVELS = {"基本": 14, "標準": 16, "応用": 5}
    TYPES = ("choice",)
    GROUPS = [
        ("データの記述と時系列", range(1, 6), 9), ("データの集め方", range(6, 7), 2), ("確率と確率分布", range(7, 10), 6),
        ("標本分布", range(10, 11), 3), ("推定", range(11, 12), 4), ("検定", range(12, 14), 6),
        ("回帰分析", range(14, 15), 3), ("実験計画と分散分析", range(15, 16), 2),
    ]
    MAX_DEF = 3
else:
    N_ITEMS = 25
    LEVELS = {"基本": 7, "標準": 13, "応用": 5}
    TYPES = ("choice", "numeric")
    # 章のまとまりと、1セットあたりの問題数（spec/research/exam-blueprint.md）
    GROUPS = [
        ("確率と分布", range(1, 7), 5), ("漸近理論と推定", range(7, 10), 3), ("検定", range(10, 14), 3),
        ("確率過程", range(14, 16), 1), ("回帰分析", range(16, 20), 4), ("実験計画と標本調査", range(20, 22), 2),
        ("多変量解析", range(22, 27), 3), ("時系列・分割表・欠測", range(27, 30), 2), ("モデル選択・ベイズ・シミュレーション・因果", range(30, 34), 2),
    ]
    MAX_DEF = 2
PH = re.compile(r"«([A-Za-z_][A-Za-z0-9_]*)»")
ITEM_TEXT = ("question", "explanation", "answer_format")
EXP_HEADS = ("**方針**", "**途中式**", "**答え**", "**間違えやすい点**")


def group_of(ch):
    for i, (_, r, _) in enumerate(GROUPS):
        if ch in r:
            return i
    return -1


def set_files():
    return sorted(SETS.glob("set*.yaml"))


def load_values(no):
    p = GEN / f"set{no:02d}.py"
    if not p.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"gen_set{no:02d}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.values()


def fill(v, env, missing):
    """«名前» を差し込む。値全体が «名前» 1つなら、その値の型（数値など）のまま返す。"""
    if isinstance(v, str):
        m = PH.fullmatch(v.strip())
        if m:
            if m.group(1) not in env:
                missing.add(m.group(1))
                return v
            return env[m.group(1)]

        def rep(mm):
            if mm.group(1) not in env:
                missing.add(mm.group(1))
                return mm.group(0)
            return str(env[mm.group(1)])
        return PH.sub(rep, v)
    if isinstance(v, list):
        return [fill(x, env, missing) for x in v]
    if isinstance(v, dict):
        return {k: fill(x, env, missing) for k, x in v.items()}
    return v


def load_set(path):
    """セットを読み、値を差し込んだものを返す。戻り値: (セット, 未定義の «» の集合)"""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    no = int(data.get("set") or re.search(r"\d+", path.stem).group())
    vals = load_values(no)
    missing = set()
    probs = []
    for p in data.get("problems") or []:
        env = dict(vals.get("_common", {}))
        env.update(vals.get(p.get("id"), {}))
        q = fill(p, env, missing)
        probs.append(q)
    data["problems"] = probs
    data["set"] = no
    return data, missing


def items_of(data):
    for p in data.get("problems") or []:
        for it in p.get("items") or []:
            yield p, it


def item_hash(p, it):
    """小問の内容（状況設定を含む）のハッシュ。検算・見直しのあとに書き換えたら公開しない。"""
    body = {"context": p.get("context"), "title": p.get("title"), "item": it}
    s = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def num_like(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_set(data, missing, plan_ids):
    """自動検査。戻り値: (エラー, 警告) の文字列リスト"""
    err, warn = [], []
    no = data["set"]
    tag = f"set{no:02d}"
    if missing:
        err.append(f"{tag}: generator にない «»: {sorted(missing)}")
    if data.get("minutes") != MINUTES:
        err.append(f"{tag}: minutes は {MINUTES}")
    items = list(items_of(data))
    if len(items) != N_ITEMS:
        err.append(f"{tag}: 小問は {N_ITEMS} 問（いま {len(items)} 問）")
    ids = set()
    lv = {k: 0 for k in LEVELS}
    grp = [0] * len(GROUPS)
    n_def = 0
    for p, it in items:
        iid = it.get("id", "?")
        w = f"{tag} {iid}"
        if iid in ids:
            err.append(f"{w}: id が重複")
        ids.add(iid)
        if not str(iid).startswith(str(p.get("id"))):
            err.append(f"{w}: 小問の id は大問の id（{p.get('id')}）で始める")
        for k in ("chapter", "level", "type", "question", "answer", "explanation", "terms", "topic"):
            if it.get(k) in (None, "", []):
                err.append(f"{w}: {k} がない")
        ch = it.get("chapter")
        if isinstance(ch, int) and group_of(ch) >= 0:
            grp[group_of(ch)] += 1
        else:
            err.append(f"{w}: chapter が不正: {ch}")
        if it.get("level") in lv:
            lv[it["level"]] += 1
        else:
            err.append(f"{w}: level は 基本/標準/応用")
        if it.get("definition"):
            n_def += 1
        for t in it.get("terms") or []:
            if t not in plan_ids:
                err.append(f"{w}: terms の {t} が plan.yaml にない")
        exp = str(it.get("explanation") or "")
        for h in EXP_HEADS:
            if h not in exp:
                err.append(f"{w}: 解説に {h} がない（方針→途中式→答え→間違えやすい点 の順）")
        pos = [exp.find(h) for h in EXP_HEADS]
        if all(x >= 0 for x in pos) and pos != sorted(pos):
            err.append(f"{w}: 解説の見出しの順番が違う")
        typ = it.get("type")
        if typ == "choice":
            c = it.get("choices") or []
            if len(c) != 5:
                err.append(f"{w}: 5肢選択は選択肢が5つ")
            norm = [re.sub(r"\s+", "", str(x)) for x in c]
            if len(set(norm)) != len(norm):
                err.append(f"{w}: 選択肢が重複: {c}")
            a = it.get("answer")
            if not (isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(c)):
                err.append(f"{w}: answer は正解の選択肢の番号 0〜4（ただ1つ）")
            d = it.get("distractors") or []
            if len(d) != len(c) or (isinstance(a, int) and 0 <= a < len(d) and d[a] != "正解"):
                err.append(f"{w}: distractors は選択肢と同じ数で、正解の位置に「正解」、誤答の位置にその誤りの理由を書く")
            elif any(not str(x).strip() for x in d):
                err.append(f"{w}: distractors に空がある")
        elif typ == "numeric" and "numeric" in TYPES:
            if not num_like(it.get("answer")):
                err.append(f"{w}: 数値入力の answer は数値（generator の値）: {it.get('answer')!r}")
            tol = it.get("tolerance")
            if not num_like(tol) or tol <= 0:
                err.append(f"{w}: tolerance（許容範囲）がない")
            fmt = str(it.get("answer_format") or "")
            if not fmt:
                err.append(f"{w}: answer_format（丸め方）がない")
            if "±" not in str(it.get("question", "")) + fmt and "許容" not in fmt:
                err.append(f"{w}: 許容範囲を問題文か answer_format に書く")
        else:
            err.append(f"{w}: type は {' / '.join(TYPES)}")
    if lv != LEVELS and len(items) == N_ITEMS:
        err.append(f"{tag}: 難易度の割合が {LEVELS} と違う: {lv}")
    want = [g[2] for g in GROUPS]
    if grp != want and len(items) == N_ITEMS:
        err.append(f"{tag}: 章の配分が出題配分表と違う: " + ", ".join(f"{GROUPS[i][0]} {grp[i]}/{want[i]}" for i in range(len(GROUPS)) if grp[i] != want[i]))
    if n_def > MAX_DEF:
        err.append(f"{tag}: 定義を問うだけの問題は{MAX_DEF}問まで（いま {n_def} 問）")
    return err, warn


def results(no, kind):
    p = (RESULTS if kind == "verify" else REVIEW) / f"set{no:02d}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def status(data):
    """各小問の公開可否。戻り値: {小問id: (検算OK, 見直しOK)}"""
    no = data["set"]
    ver, rev = results(no, "verify"), results(no, "review")
    out = {}
    for p, it in items_of(data):
        h = item_hash(p, it)
        v = ver.get(it["id"]) or {}
        out[it["id"]] = (bool(v.get("ok")) and v.get("hash") == h, rev.get(it["id"]) == h)
    return out
