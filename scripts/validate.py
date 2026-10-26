"""用語ファイル（content/chNN/*.md）の検査。

使い方:
    .venv/bin/python scripts/validate.py              # 全部
    .venv/bin/python scripts/validate.py --only ID    # 1用語だけ
    .venv/bin/python scripts/validate.py --chapter N  # 1章だけ
    .venv/bin/python scripts/validate.py --questions  # 問題ファイル（content/questions/）だけ

エラーがあれば終了コード1。警告は表示のみ。
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import CONTENT, LEVEL, PRE1_CONTENT, paths  # noqa: E402  級ごとの置き場所（JD_LEVEL）

HEADINGS = ["一言でいうと", "定義・数式", "数式の読み解き", "具体例", "現場での使われ方",
            "グラフ・イメージ", "試験のポイント", "よくある間違い"]
if LEVEL == "tb":  # 教科書は試験向けではないので「学習のポイント」にする（DECISIONS.md 2026-10-26）
    HEADINGS = [("学習のポイント" if h == "試験のポイント" else h) for h in HEADINGS]
MAX_CHAPTER = 33 if LEVEL != "tb" else 80
REQUIRED = ["id", "title", "reading", "en", "chapter", "importance", "related"]
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CHART_TYPES = {"pmf", "pdf", "cdf", "pdf-cdf", "survival-hazard", "scatter", "custom"}
DISTS = {
    "bernoulli": ["p"], "binom": ["n", "p"], "hypergeom": ["N", "K", "n"], "poisson": ["lambda"],
    "geom": ["p", "start"], "nbinom": ["r", "p"], "unif": ["a", "b"], "norm": ["mu", "sigma"],
    "lognorm": ["mu", "sigma"], "expon": ["rate"], "gamma": ["shape", "scale"], "beta": ["a", "b"],
    "cauchy": ["loc", "scale"], "weibull": ["shape", "scale"], "logistic": ["loc", "scale"],
    "chi2": ["df"], "t": ["df"], "f": ["df1", "df2"], "ncx2": ["df", "nc"], "nct": ["df", "nc"],
    "ncf": ["df1", "df2", "nc"], "normmix": ["w", "mu1", "sigma1", "mu2", "sigma2"],
}
DISCRETE = {"bernoulli", "binom", "hypergeom", "poisson", "geom", "nbinom"}
SCATTER_MODELS = {"bvnorm", "spurious", "regression"}
SH_DISTS = {"expon", "weibull", "gamma", "lognorm"}


def load_plan():
    plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    terms = {}
    for ch in plan["chapters"]:
        for t in ch.get("terms") or []:
            terms.setdefault(t["id"], []).append((ch["no"], t))
    return plan, terms


def custom_chart_ids():
    ids = set()
    for p in (ROOT / "src").glob("charts*.js"):
        ids |= set(re.findall(r"(?:registerCustom|\breg)\(\s*['\"]([\w-]+)['\"]", p.read_text(encoding="utf-8")))
    return ids


def split_front(text):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    return text[4:end], text[end + 5:]


def strip_math(body):
    """$$…$$ と $…$ を取り除いた本文を返す（$ の対応検査用）。"""
    b = body.replace("\\$", "")
    b = re.sub(r"\$\$.*?\$\$", "", b, flags=re.S)
    return b


def check_dollars(body, err):
    b = strip_math(body)
    if b.count("$$"):
        err("$$ の対応がとれていません")
    for ln, line in enumerate(b.split("\n"), 1):
        if line.count("$") % 2:
            err(f"本文 {ln} 行付近で $ の数が奇数です: {line.strip()[:50]}")


def check_charts(charts, imp, has_chart_heading, err, warn, customs):
    if charts is None:
        charts = []
    if not isinstance(charts, list):
        err("charts はリストにしてください")
        return
    if not charts:
        if imp == "A":
            err("重要度Aには charts が必要です")
        elif imp == "B":
            warn("重要度Bにはできるだけ charts を付けてください")
        if has_chart_heading:
            warn("charts がないのに「グラフ・イメージ」節があります")
        return
    if not has_chart_heading:
        err("charts があるのに「グラフ・イメージ」節がありません")
    if len(charts) > 3:
        warn("charts は3個までを目安にしてください")
    for i, c in enumerate(charts, 1):
        tag = f"charts[{i}]"
        if not isinstance(c, dict):
            err(f"{tag} が辞書ではありません")
            continue
        typ = c.get("type")
        if typ not in CHART_TYPES:
            err(f"{tag} type が不正: {typ}")
            continue
        if not c.get("caption"):
            err(f"{tag} caption がありません")
        params = c.get("params") or {}
        if not isinstance(params, dict):
            err(f"{tag} params は辞書にしてください")
            params = {}
        if typ in ("pmf", "pdf", "cdf", "pdf-cdf", "survival-hazard"):
            d = c.get("dist")
            if d not in DISTS:
                err(f"{tag} dist が不正: {d}")
            else:
                for k in params:
                    if k not in DISTS[d]:
                        err(f"{tag} dist={d} に無いパラメータ: {k}")
                for k in DISTS[d]:
                    if k not in params and not (d == "geom" and k == "start"):
                        err(f"{tag} dist={d} のパラメータ {k} がありません")
                if typ == "pmf" and d not in DISCRETE:
                    err(f"{tag} pmf には離散分布を指定してください（{d}）")
                if typ in ("pdf", "pdf-cdf") and d in DISCRETE and typ == "pdf":
                    err(f"{tag} pdf には連続分布を指定してください（{d}）")
                if typ == "survival-hazard" and d not in SH_DISTS:
                    err(f"{tag} survival-hazard の dist は {sorted(SH_DISTS)} のどれか")
                for cmp in c.get("compare") or []:
                    for k in (cmp.get("params") or {}):
                        if k not in DISTS.get(cmp.get("dist", d), []):
                            err(f"{tag} compare のパラメータが不正: {k}")
        elif typ == "scatter":
            if c.get("model") not in SCATTER_MODELS:
                err(f"{tag} scatter の model が不正: {c.get('model')}")
        elif typ == "custom":
            cid = c.get("id")
            if not cid:
                err(f"{tag} custom の id がありません")
            elif cid not in customs:
                warn(f"{tag} カスタム図 '{cid}' は charts.js に未実装です（準備中と表示されます）")
        for s in c.get("sliders") or []:
            if not isinstance(s, dict) or "param" not in s:
                err(f"{tag} sliders の各要素に param が必要です")
                continue
            if typ != "custom" and typ != "scatter" and s["param"] not in params:
                err(f"{tag} スライダーの param '{s['param']}' が params にありません")
            for k in ("min", "max", "step"):
                if k not in s:
                    err(f"{tag} スライダー {s['param']} に {k} がありません")


def check_file(path, plan_terms, customs, all_aliases):
    errors, warnings = [], []
    err = lambda m: errors.append(m)
    warn = lambda m: warnings.append(m)
    text = path.read_text(encoding="utf-8")
    fm_text, body = split_front(text)
    if fm_text is None:
        return ["フロントマター（--- で囲む）がありません"], [], None
    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        return [f"フロントマターのYAMLが読めません: {e}"], [], None
    if not isinstance(fm, dict):
        return ["フロントマターが辞書ではありません"], [], None
    for k in REQUIRED:
        if k not in fm or fm[k] is None:
            err(f"必須項目 {k} がありません")
    tid = fm.get("id")
    if tid is not None:
        if not ID_RE.match(str(tid)):
            err(f"id の形式が不正: {tid}")
        if path.stem != tid:
            err(f"ファイル名と id が違います: {path.name} / {tid}")
        if tid not in plan_terms:
            err(f"id '{tid}' が plan.yaml にありません")
        else:
            no, pt = plan_terms[tid][0]
            if fm.get("chapter") != no:
                err(f"chapter {fm.get('chapter')} が plan.yaml の章 {no} と違います")
            if path.parent.name != f"ch{no:02d}":
                err(f"置き場所が違います（ch{no:02d}/ に置く）")
            if fm.get("importance") != pt.get("importance"):
                warn(f"importance {fm.get('importance')} が plan.yaml（{pt.get('importance')}）と違います")
            if fm.get("title") != pt.get("title"):
                warn(f"title が plan.yaml（{pt.get('title')}）と違います")
    if fm.get("importance") not in ("A", "B", "C"):
        err(f"importance は A/B/C: {fm.get('importance')}")
    if not isinstance(fm.get("chapter"), int) or not 1 <= fm.get("chapter", 0) <= MAX_CHAPTER:
        err(f"chapter は1〜{MAX_CHAPTER}の整数")
    rd = fm.get("reading") or ""
    if rd and not re.fullmatch(r"[ぁ-ゟー・a-zA-Z0-9０-９ 　\-]+", str(rd)):
        warn(f"reading はひらがなで書いてください: {rd}")
    for k in ("aliases", "tags", "related"):
        if k in fm and fm[k] is not None and not isinstance(fm[k], list):
            err(f"{k} はリストにしてください")
    rel = fm.get("related") or []
    if isinstance(rel, list):
        if not 2 <= len(rel) <= 6:
            warn(f"related は2〜6個（今 {len(rel)} 個）")
        for r in rel:
            if r not in plan_terms:
                err(f"related の '{r}' が plan.yaml にありません")
            if r == tid:
                err("related に自分自身が入っています")
    for a in fm.get("aliases") or []:
        key = str(a)
        if key in all_aliases and all_aliases[key] != tid:
            err(f"別名 '{a}' が他の用語（{all_aliases[key]}）と重複しています")
    # 見出し
    heads = re.findall(r"^## (.+?)\s*$", body, flags=re.M)
    has_chart = "グラフ・イメージ" in heads
    expected = [h for h in HEADINGS if h != "グラフ・イメージ" or has_chart]
    if heads != expected:
        missing = [h for h in HEADINGS if h not in heads and h != "グラフ・イメージ"]
        extra = [h for h in heads if h not in HEADINGS]
        msg = "見出しの順番・表記が違います"
        if missing:
            msg += f"／不足: {missing}"
        if extra:
            msg += f"／余分: {extra}"
        err(msg)
    if re.search(r"^# ", body, flags=re.M):
        err("本文に # 見出し（h1）は使わないでください")
    if re.search(r"^## 関連用語", body, flags=re.M):
        err("「関連用語」は本文に書かず related に入れてください")
    check_dollars(body, err)
    if "\\bm{" in body:
        err("\\bm は使えません。\\boldsymbol を使ってください")
    check_charts(fm.get("charts"), fm.get("importance"), has_chart, err, warn, customs)
    # 分量の目安（数式を除いた字数）
    plain = re.sub(r"\$\$.*?\$\$", "", body, flags=re.S)
    plain = re.sub(r"\$[^$\n]*\$", "", plain)
    n = len(re.sub(r"\s", "", plain))
    # 2級は1語あたりの内容が準1級より軽いので、目安を2割下げる（DECISIONS.md 2026-10-24）
    # 教科書（大学院の内容）は準1級と同じ目安
    lo = ({"A": 1200, "B": 800, "C": 400} if LEVEL == "g2" else {"A": 1500, "B": 1000, "C": 500}).get(fm.get("importance"), 0)
    if n < lo:
        warn(f"本文が短めです（{n}字）")
    return errors, warnings, fm


Q_ID_RE = re.compile(r"^q\d{2}-[a-z0-9]+(-[a-z0-9]+)*$")
Q_TYPES = {"choice", "numeric", "truefalse", "derivation"}
Q_LEVELS = {"基本", "標準", "応用"}
Q_TEXT = ["question", "hint", "explanation"]


def check_questions(plan_terms, chapter=None):
    """content/questions/chNN.yaml の検査。戻り値：(エラー数, 警告数, ファイル数, 問題数)"""
    n_err = n_warn = n_files = n_q = 0
    seen = {}
    files = sorted((CONTENT / "questions").glob("ch*.yaml"))
    if chapter:
        files = [p for p in files if p.stem == f"ch{chapter:02d}"]
    for p in files:
        n_files += 1
        rel = p.relative_to(ROOT)
        errs, warns = [], []
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            print(f"[ERROR] {rel}: YAML として読めません: {str(e).splitlines()[0]}")
            n_err += 1
            continue
        m = re.match(r"ch(\d{2})$", p.stem)
        chno = int(m.group(1)) if m else None
        if data.get("chapter") != chno:
            errs.append(f"chapter が {data.get('chapter')}（ファイル名は第{chno}章）")
        qs = data.get("questions")
        if not isinstance(qs, list):
            errs.append("questions がリストではありません")
            qs = []
        for q in qs:
            n_q += 1
            if not isinstance(q, dict):
                errs.append("問題が辞書ではありません")
                continue
            qid = q.get("id", "?")
            E = lambda msg: errs.append(f"{qid}: {msg}")
            if not isinstance(qid, str) or not Q_ID_RE.match(qid):
                E("id の形式が不正（q + 章2桁 + 用語id + 連番）")
            elif not qid.startswith(f"q{chno:02d}-"):
                E(f"id が q{chno:02d}- で始まっていません")
            if qid in seen:
                E(f"id が {seen[qid]} と重複しています")
            seen[qid] = rel
            term = q.get("term")
            if term not in plan_terms:
                E(f"term '{term}' が plan.yaml にありません")
            elif plan_terms[term][0][0] != chno:
                E(f"term '{term}' は第{plan_terms[term][0][0]}章の用語です")
            if q.get("level") not in Q_LEVELS:
                E(f"level が不正: {q.get('level')}（基本／標準／応用）")
            typ = q.get("type")
            if typ not in Q_TYPES:
                E(f"type が不正: {typ}")
            for k in Q_TEXT:
                v = q.get(k)
                if not isinstance(v, str) or not v.strip():
                    E(f"{k} がありません")
            texts = [q.get(k) for k in Q_TEXT] + list(q.get("choices") or []) + list(q.get("steps") or [])
            if typ == "derivation" or typ == "choice":
                texts.append(q.get("answer") if typ == "derivation" else "")
            for t in texts:
                if isinstance(t, str):
                    body = t.replace("\\$", "")
                    body = re.sub(r"\$\$.+?\$\$", "", body, flags=re.S)
                    if body.count("$") % 2:
                        E("$ の数が奇数です（数式の閉じ忘れ）")
                        break
            ans = q.get("answer")
            if typ == "choice":
                ch = q.get("choices")
                if not isinstance(ch, list) or len(ch) != 4 or not all(isinstance(c, (str, int, float)) for c in ch):
                    E("choices は4つの文字列にしてください")
                elif len({str(c).strip() for c in ch}) != 4:
                    E("choices に同じ選択肢があります")
                if not isinstance(ans, int) or isinstance(ans, bool) or not 0 <= ans <= 3:
                    E("choice の answer は 0〜3 の添字です")
            elif typ == "numeric":
                if not isinstance(ans, (int, float)) or isinstance(ans, bool):
                    E("numeric の answer は数値です")
                tol = q.get("tolerance")
                if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
                    E("numeric には正の tolerance が必要です")
            elif typ == "truefalse":
                if not isinstance(ans, bool):
                    E("truefalse の answer は true / false です")
            elif typ == "derivation":
                st = q.get("steps")
                if not isinstance(st, list) or len(st) < 2 or not all(isinstance(x, str) and x.strip() for x in st):
                    E("derivation には2つ以上の steps が必要です")
                if not isinstance(ans, str) or not ans.strip():
                    E("derivation の answer は最終結果の文字列です")
        for e in errs:
            print(f"[ERROR] {rel}: {e}")
        for w in warns:
            print(f"[WARN]  {rel}: {w}")
        n_err += len(errs)
        n_warn += len(warns)
    return n_err, n_warn, n_files, n_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--chapter", type=int)
    ap.add_argument("--questions", action="store_true")
    args = ap.parse_args()

    plan, plan_terms = load_plan()
    if args.questions:
        e, w, nf, nq = check_questions(plan_terms, args.chapter)
        print(f"問題ファイル {nf} 件 / 問題 {nq} 問 / エラー {e} 件 / 警告 {w} 件")
        sys.exit(1 if e else 0)
    total_err = total_warn = 0
    for tid, occ in plan_terms.items():
        if len(occ) > 1:
            print(f"[ERROR] plan.yaml: id '{tid}' が複数の章にあります: {[o[0] for o in occ]}")
            total_err += 1
        if not ID_RE.match(tid):
            print(f"[ERROR] plan.yaml: id の形式が不正: {tid}")
            total_err += 1
        see = occ[0][1].get("see")
        if see is not None and (see not in plan_terms or see == tid):
            print(f"[ERROR] plan.yaml: {tid} の see（統合先）'{see}' が plan.yaml にありません")
            total_err += 1
        st = occ[0][1].get("status")
        if st not in ("planned", "written", "verified", "on_hold"):
            print(f"[ERROR] plan.yaml: {tid} の status が不正: {st}")
            total_err += 1
    customs = custom_chart_ids()
    files = sorted(CONTENT.glob("ch*/*.md"))

    # 別名の表（全ファイルから。title も別名との衝突対象に入れる）。ほかの級の解説を共通で使う用語（from）も含める
    all_aliases = {}
    shared_files = [f for tid, occ in plan_terms.items() if occ[0][1].get("from") in ("pre1", "g2")
                    for f in sorted(paths(occ[0][1]["from"])["content"].glob(f"ch*/{tid}.md"))[:1]]
    for p in shared_files + files:
        fm_text, _ = split_front(p.read_text(encoding="utf-8"))
        try:
            fm = yaml.safe_load(fm_text) if fm_text else None
        except yaml.YAMLError:
            fm = None
        if isinstance(fm, dict):
            for a in [fm.get("title")] + list(fm.get("aliases") or []):
                if a is not None:
                    all_aliases.setdefault(str(a), fm.get("id"))

    targets = files
    if args.only:
        targets = [p for p in files if p.stem == args.only]
        if not targets:
            print(f"[ERROR] {args.only}.md が見つかりません")
            sys.exit(1)
    elif args.chapter:
        targets = [p for p in files if p.parent.name == f"ch{args.chapter:02d}"]

    seen = {}
    for p in targets:
        errors, warnings, fm = check_file(p, plan_terms, customs, all_aliases)
        rel = p.relative_to(ROOT)
        if fm and fm.get("id") in seen:
            errors.append(f"id が {seen[fm['id']]} と重複しています")
        if fm and fm.get("id"):
            seen[fm["id"]] = rel
        for e in errors:
            print(f"[ERROR] {rel}: {e}")
        for w in warnings:
            print(f"[WARN]  {rel}: {w}")
        total_err += len(errors)
        total_warn += len(warnings)

    # 状態の整合（全体検査のときだけ）
    if not args.only and not args.chapter:
        existing = {p.stem for p in files}
        for tid, occ in plan_terms.items():
            st = occ[0][1].get("status")
            src = occ[0][1].get("from")
            if src is not None:  # ほかの級の解説を共通で使う用語（2級・教科書）。ファイルは元の級の側で検査する
                if src not in ("pre1", "g2") or src == LEVEL:
                    print(f"[ERROR] plan.yaml: {tid} の from は pre1 か g2（この級以外）: {src}")
                    total_err += 1
                    continue
                if not list(paths(src)["content"].glob(f"ch*/{tid}.md")):
                    print(f"[ERROR] plan.yaml: {tid} は from: {src} ですが元の級に解説がありません")
                    total_err += 1
                if tid in existing:
                    print(f"[ERROR] plan.yaml: {tid} は from: {src} ですが、この級にもファイルがあります（どちらかにする）")
                    total_err += 1
                continue
            if st in ("written", "verified") and tid not in existing:
                print(f"[ERROR] plan.yaml: {tid} は {st} ですがファイルがありません")
                total_err += 1
            if st == "planned" and tid in existing:
                print(f"[WARN]  plan.yaml: {tid} はファイルがありますが status が planned です")
                total_warn += 1

    # 2級：共通の用語の注記（content/g2/notes.yaml）は、from: pre1 の用語だけに付ける
    if not args.only and not args.chapter and (CONTENT / "notes.yaml").exists():
        notes = yaml.safe_load((CONTENT / "notes.yaml").read_text(encoding="utf-8")) or {}
        shared_ids = {tid for tid, occ in plan_terms.items() if occ[0][1].get("from") == "pre1"}
        for tid in sorted(set(notes) - shared_ids):
            print(f"[ERROR] notes.yaml: {tid} は準1級と共通の用語ではありません")
            total_err += 1
        for tid in sorted(shared_ids - set(notes)):
            print(f"[WARN]  notes.yaml: 共通の用語 {tid} に注記がありません")
            total_warn += 1
        for tid, v in notes.items():
            if "$" in str(v) or len(str(v)) > 120:
                print(f"[ERROR] notes.yaml: {tid} の注記は $ を使わず 120 字以内にする")
                total_err += 1

    if not args.only:
        e, w, nf, nq = check_questions(plan_terms, args.chapter)
        print(f"問題ファイル {nf} 件 / 問題 {nq} 問 / エラー {e} 件 / 警告 {w} 件")
        total_err += e
        total_warn += w
    if not args.only and not args.chapter:
        # 数式の全数チェック（scripts/check_math.py）。MJ_DIR があれば MathJax での変換テストも行う
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_math.py")], capture_output=True, text=True)
        out = r.stdout.strip().splitlines()
        for ln in out:
            if ln.startswith(("[MATH]", "[TEX]")) or r.returncode:
                print(ln)
            elif ln.startswith(("数式チェック", "MathJax")):
                print(ln)
        if r.returncode:
            total_err += 1
            print("[ERROR] check_math.py で数式の問題が見つかりました")
        # ロードマップ（content/roadmap.yaml）の検査：全用語を1回ずつ、前提が先、循環なし
        if (CONTENT / "roadmap.yaml").exists():
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_roadmap.py")], capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-3000:])
                print("[ERROR] check_roadmap.py でロードマップの問題が見つかりました")
                total_err += 1
            else:
                print("ロードマップ: " + ((r.stdout.strip().splitlines() or ["OK"])[-1]))
        # 模擬試験の自動検査。書きかけのセットでビルドが止まらないよう警告として数える（公開は build.py が3段階の確認で決める）
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_exam.py")], capture_output=True, text=True)
        for ln in r.stdout.strip().splitlines():
            if ln.startswith("[ERROR]"):
                print("[WARN]  模擬試験: " + ln[8:])
                total_warn += 1
            elif ln.startswith("模擬試験"):
                print(ln)
    print(f"検査ファイル {len(targets)} 件 / エラー {total_err} 件 / 警告 {total_warn} 件")
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
