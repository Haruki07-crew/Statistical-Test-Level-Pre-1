"""用語ファイル（content/chNN/*.md）の検査。

使い方:
    .venv/bin/python scripts/validate.py              # 全部
    .venv/bin/python scripts/validate.py --only ID    # 1用語だけ
    .venv/bin/python scripts/validate.py --chapter N  # 1章だけ

エラーがあれば終了コード1。警告は表示のみ。
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"

HEADINGS = ["一言でいうと", "定義・数式", "数式の読み解き", "具体例", "現場での使われ方",
            "グラフ・イメージ", "試験のポイント", "よくある間違い"]
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
    p = ROOT / "src" / "charts.js"
    if not p.exists():
        return set()
    return set(re.findall(r"registerCustom\(\s*['\"]([\w-]+)['\"]", p.read_text(encoding="utf-8")))


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
    if not isinstance(fm.get("chapter"), int) or not 1 <= fm.get("chapter", 0) <= 32:
        err("chapter は1〜32の整数")
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
    lo = {"A": 1500, "B": 1000, "C": 500}.get(fm.get("importance"), 0)
    if n < lo:
        warn(f"本文が短めです（{n}字）")
    return errors, warnings, fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--chapter", type=int)
    args = ap.parse_args()

    plan, plan_terms = load_plan()
    total_err = total_warn = 0
    for tid, occ in plan_terms.items():
        if len(occ) > 1:
            print(f"[ERROR] plan.yaml: id '{tid}' が複数の章にあります: {[o[0] for o in occ]}")
            total_err += 1
        if not ID_RE.match(tid):
            print(f"[ERROR] plan.yaml: id の形式が不正: {tid}")
            total_err += 1
        st = occ[0][1].get("status")
        if st not in ("planned", "written", "verified", "on_hold"):
            print(f"[ERROR] plan.yaml: {tid} の status が不正: {st}")
            total_err += 1
    customs = custom_chart_ids()
    files = sorted(CONTENT.glob("ch*/*.md"))

    # 別名の表（全ファイルから。title も別名との衝突対象に入れる）
    all_aliases = {}
    for p in files:
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
            if st in ("written", "verified") and tid not in existing:
                print(f"[ERROR] plan.yaml: {tid} は {st} ですがファイルがありません")
                total_err += 1
            if st == "planned" and tid in existing:
                print(f"[WARN]  plan.yaml: {tid} はファイルがありますが status が planned です")
                total_warn += 1

    print(f"検査ファイル {len(targets)} 件 / エラー {total_err} 件 / 警告 {total_warn} 件")
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
