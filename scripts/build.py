"""content/ から docs/index.html（準1級）と docs/g2/index.html（2級）を生成する。

使い方: .venv/bin/python scripts/build.py            # 両方の級
        JD_LEVEL=g2 .venv/bin/python scripts/build.py # 2級だけ（service worker は更新しない）
級ごとに先に validate.py を実行し、エラーがあれば止める。
"""
import os
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import CONTENT, LEVEL, OUT, PRE1_CONTENT, ROOT, VERIFY, paths  # noqa: E402  級ごとの置き場所（JD_LEVEL）

SRC = ROOT / "src"
DOCS = ROOT / "docs"

SECTION_KEYS = {
    "一言でいうと": "summary", "定義・数式": "definition", "数式の読み解き": "reading",
    "具体例": "example", "現場での使われ方": "field", "グラフ・イメージ": "chart",
    "試験のポイント": "exam", "よくある間違い": "mistake",
}


def split_front(text):
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end]), text[end + 5:]


def protect_math(body):
    """数式を記号に置き換える。戻り値：置換後の本文、復元用の辞書。"""
    store = {}

    def keep(kind, tex):
        key = f"MATHX{len(store):05d}{kind}X"
        store[key] = (kind, tex)
        return key

    body = body.replace("\\$", keep("D", "\\$"))
    body = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n\n" + keep("B", m.group(1)) + "\n\n", body, flags=re.S)
    body = re.sub(r"\$([^$\n]+?)\$", lambda m: keep("I", m.group(1)), body)
    return body, store


def restore_math(h, store):
    def rep_block(m):
        _, tex = store[m.group(1)]
        return f'<div class="math-display">$${html.escape(tex.strip(), quote=False)}$$</div>'

    h = re.sub(r"<p>\s*(MATHX\d{5}BX)\s*</p>", rep_block, h)
    h = re.sub(r"(MATHX\d{5}BX)", rep_block, h)

    def rep(m):
        kind, tex = store[m.group(0)]
        if kind == "D":
            return "\\$"
        return f'<m>${html.escape(tex, quote=False)}$</m>'  # 短いタグでサイズを節約（CSS では m 要素として扱う）

    return re.sub(r"MATHX\d{5}[ID]X", rep, h)


MD = markdown.Markdown(extensions=["tables", "sane_lists"])


def render_body(body):
    prot, store = protect_math(body)
    MD.reset()
    h = MD.convert(prot)
    h = restore_math(h, store)
    h = h.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # 節ごとに <section> で包む
    parts = re.split(r"<h2>(.*?)</h2>", h)
    out = [parts[0]] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        key = SECTION_KEYS.get(title, "other")
        out.append(f'<section class="sec" data-sec="{key}" id="sec-{key}"><h2>{title}</h2>{parts[i + 1]}</section>')
    return "".join(out)


def render_inline(text):
    """問題の選択肢など短い文を HTML にする（段落1つなら <p> を外す）。"""
    h = render_body(str(text))
    m = re.fullmatch(r"\s*<p>(.*?)</p>\s*", h, flags=re.S)
    return m.group(1) if m and "<p>" not in m.group(1) else h


def load_questions(plan_ids, shared=()):
    """検算に通り、内容が検算時から変わっていない問題だけを返す。戻り値：(問題の一覧, 章ごとの {全問, 公開})
    2級では、準1級と共通の用語（shared）について、準1級の問題のうち level が「基本」のものも使う（キーは "準1級"）。"""
    sys.path.insert(0, str(ROOT / "verify"))
    from qcheck import qhash
    out, stat = [], {}
    sources = [(p, VERIFY, None) for p in sorted((CONTENT / "questions").glob("ch*.yaml"))]
    if shared:
        sources += [(p, paths("pre1")["verify"], "準1級") for p in sorted((PRE1_CONTENT / "questions").glob("ch*.yaml"))]
    for p, vdir, src in sources:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        res_path = vdir / "questions_results" / f"{p.stem}.json"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
        qs = data.get("questions") or []
        if src:
            qs = [q for q in qs if q["term"] in shared and q["level"] == "基本"]
            if not qs:
                continue
        pub = 0
        for q in qs:
            r = res.get(q["id"])
            if not r or not r.get("ok") or r.get("hash") != qhash(q) or q["term"] not in plan_ids:
                continue
            pub += 1
            item = {"id": q["id"], "term": q["term"], "lv": q["level"], "type": q["type"],
                    "q": render_body(q["question"]), "h": render_body(q["hint"]), "e": render_body(q["explanation"])}
            if q["type"] == "choice":
                item["c"] = [render_inline(c) for c in q["choices"]]
                item["a"] = q["answer"]
            elif q["type"] == "numeric":
                item["a"] = q["answer"]
                item["tol"] = q["tolerance"]
            elif q["type"] == "truefalse":
                item["a"] = bool(q["answer"])
            else:
                item["a"] = render_inline(q["answer"])
                item["steps"] = [render_inline(x) for x in q["steps"]]
            out.append(item)
        key = src or data.get("chapter")
        a0, b0 = stat.get(key, (0, 0))
        stat[key] = (a0 + len(qs), b0 + pub)
    return out, stat


def load_exams(plan_ids):
    """模擬試験：自動検査・検算・見直しの3段階すべてに通った（ハッシュが一致する）小問だけでできたセットを返す。
    戻り値: (公開するセットの一覧, {セット番号: (小問数, 検算OK, 見直しOK, 自動検査エラー数)})"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import exam_lib as L
    out, stat = [], {}
    for path in L.set_files():
        try:
            data, missing = L.load_set(path)
        except Exception as e:  # 書きかけのセットなどは公開しない
            print(f"模擬試験 {path.name}: 読み込めないため公開しません（{e}）")
            continue
        err, _ = L.check_set(data, missing, plan_ids)
        st = L.status(data)
        stat[data["set"]] = (len(st), sum(v for v, _ in st.values()), sum(r for _, r in st.values()), len(err))
        if err or len(st) != L.N_ITEMS or not all(v and r for v, r in st.values()):
            continue
        probs, items = [], []
        for pi, p in enumerate(data["problems"]):
            probs.append({"id": p["id"], "t": p.get("title", ""), "ctx": render_body(p.get("context") or "")})
            for it in p["items"]:
                x = {"id": it["id"], "p": pi, "ch": it["chapter"], "lv": it["level"], "type": it["type"],
                     "q": render_body(it["question"]), "e": render_body(it["explanation"]),
                     "terms": [t for t in it.get("terms") or [] if t in plan_ids]}
                if it["type"] == "choice":
                    x["c"] = [render_inline(c) for c in it["choices"]]
                    x["a"] = it["answer"]
                else:
                    x["a"] = float(it["answer"])
                    x["tol"] = float(it["tolerance"])
                    x["fmt"] = render_inline(it["answer_format"])
                items.append(x)
        # 版：選択肢の並びと正解が変わると、保存した途中経過の「選んだ番号」がずれるので、画面側で見分けられるようにする
        import hashlib
        sig = json.dumps([(x["id"], x.get("c"), x["a"]) for x in items], ensure_ascii=False)
        out.append({"no": data["set"], "title": data.get("title") or f"模擬試験 第{data['set']}回", "min": data["minutes"],
                    "pts": 100 / L.N_ITEMS, "probs": probs, "items": items, "v": hashlib.sha1(sig.encode()).hexdigest()[:10]})
    return out, stat


def load_roadmap(plan_ids):
    p = CONTENT / "roadmap.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    terms = [{"id": t["id"], "st": t["stage"], "sc": t["score"], "pre": [x for x in t.get("prereq") or [] if x in plan_ids],
              "min": t.get("minutes", 0), "why": t.get("reason", "")} for t in d.get("terms") or [] if t.get("id") in plan_ids]
    return {"stages": d.get("stages") or [], "terms": terms}


def render_formulas(titles=None):
    """content/formulas.md（公式まとめ）を節ごとの HTML にする。「→ 用語名」は用語ページへのボタンにする。"""
    p = CONTENT / "formulas.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        text = split_front(text)[1]
    prot, store = protect_math(text)
    MD.reset()
    h = restore_math(MD.convert(prot), store)
    h = h.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    if titles:
        names = sorted(titles, key=len, reverse=True)
        pat = re.compile("→ ?(" + "|".join(re.escape(n) for n in names) + ")")
        h = pat.sub(lambda m: f'→ <button type="button" class="f-link" data-go="{titles[m.group(1)]}">{m.group(1)}</button>', h)
    parts = re.split(r"<h2>(.*?)</h2>", h)
    out = [parts[0]] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        k = (i + 1) // 2
        out.append(f'<section class="sec" data-sec="f{k}" id="sec-f{k}"><h2>{parts[i].strip()}</h2>{parts[i + 1]}</section>')
    return "".join(out)


def write_pwa(index_bytes):
    """ホーム画面に追加するための manifest・アイコン・service worker を docs/ に置く。
    service worker は docs/ に1つで、両方の級のページを保存する（版は両方のページのハッシュ）。"""
    import hashlib
    pwa = SRC / "pwa"
    for f in pwa.iterdir():
        if f.name == "sw.js":
            ver = hashlib.sha256(index_bytes).hexdigest()[:12]
            (DOCS / f.name).write_text(f.read_text(encoding="utf-8").replace("{{VERSION}}", ver), encoding="utf-8")
        elif f.is_file():
            (DOCS / f.name).write_bytes(f.read_bytes())


def write_manifest(site):
    """2級のページの manifest（アイコンは docs/ のものを使う）。"""
    m = json.loads((SRC / "pwa" / "manifest.json").read_text(encoding="utf-8"))
    m.update({"name": site["full"], "short_name": site["short"], "description": site["desc"], "theme_color": site["theme"]})
    for ic in m["icons"]:
        ic["src"] = "../" + ic["src"]
    (OUT.parent / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plain_text(body):
    t = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    t = re.sub(r"\$[^$\n]*\$", " ", t)
    t = re.sub(r"^#+ ", "", t, flags=re.M)
    t = re.sub(r"[*_`|>#]", "", t)
    t = re.sub(r"-{3,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


SITE_PRE1 = {
    "level": "pre1", "name": "準1級 用語辞典", "full": "統計検定準1級 用語辞典", "short": "準1級辞典",
    "desc": "統計検定準1級の用語を『統計学実践ワークブック』の章立てに沿って解説する辞典", "theme": "#1d6b66",
    "asset": "", "kp": "jd-", "other": {"label": "2級", "href": "g2/"},
}


def load_site():
    """級ごとの表示の設定。準1級は既定値（画面の文言は app.js の既定のまま）、2級は content/g2/site.yaml。"""
    if LEVEL == "pre1":
        return dict(SITE_PRE1)
    return yaml.safe_load((CONTENT / "site.yaml").read_text(encoding="utf-8"))


def main():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate.py")], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0:
        print(r.stdout)
        print("validate.py でエラーがあるためビルドを中止します")
        sys.exit(1)
    print(f"validate: {last}")

    plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    site = load_site()
    files = sorted(CONTENT.glob("ch*/*.md"))
    # 2級：準1級の解説を共通で使う用語（plan の from: pre1）。章は2級の章に置き換え、関連用語は2級にあるものだけ残す
    shared = {}
    for ch in plan["chapters"]:
        for t in ch.get("terms") or []:
            if t.get("from") == "pre1":
                shared[t["id"]] = (ch["no"], t["importance"])
                files += sorted(PRE1_CONTENT.glob(f"ch*/{t['id']}.md"))[:1]
    plan_all = {t["id"] for ch in plan["chapters"] for t in ch.get("terms") or []}
    terms = {}
    for p in files:
        fm, body = split_front(p.read_text(encoding="utf-8"))
        if fm["id"] in shared:
            fm["chapter"], fm["importance"] = shared[fm["id"]]
            fm["related"] = [r for r in fm.get("related") or [] if r in plan_all]
        terms[fm["id"]] = {
            "id": fm["id"], "title": fm["title"], "reading": fm.get("reading", ""), "en": fm.get("en", "") or "",
            "aliases": [str(a) for a in fm.get("aliases") or []], "chapter": fm["chapter"],
            "importance": fm["importance"], "tags": [str(t) for t in fm.get("tags") or []],
            "related": fm.get("related") or [], "charts": fm.get("charts") or [],
            "html": render_body(body),  # 検索用の本文テキストはファイルサイズ節約のため app.js で html から作る
        }
        if fm["id"] in shared:
            terms[fm["id"]]["shared"] = 1
    chapters = []
    for ch in plan["chapters"]:
        chapters.append({
            "no": ch["no"], "title": ch["title"],
            "terms": [dict({"id": t["id"], "title": t["title"], "importance": t["importance"], "status": t["status"]},
                           **({"see": t["see"]} if t.get("see") else {}))
                      for t in ch.get("terms") or []],
        })
    plan_ids = {t["id"] for ch in chapters for t in ch["terms"]}
    questions, qstat = load_questions(plan_ids, set(shared))
    exams, estat = load_exams(plan_ids)
    data = {"chapters": chapters, "terms": terms, "questions": questions, "exams": exams, "roadmap": load_roadmap(plan_ids),
            "formulas": render_formulas({t["title"]: t["id"] for ch in chapters for t in ch["terms"] if t["id"] in terms}),
            "site": {k: site[k] for k in ("level", "name", "kicker", "lead", "kp", "other", "parts", "steps", "exam") if site.get(k) is not None}}
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    tpl = (SRC / "template.html").read_text(encoding="utf-8")
    css = (SRC / "app.css").read_text(encoding="utf-8")
    js = "\n".join((SRC / f).read_text(encoding="utf-8") for f in ("dist.js", "charts.js", "charts_custom.js", "app.js"))
    for k in ("name", "full", "short", "desc", "theme", "asset", "kp", "level"):
        tpl = tpl.replace("{{" + k.upper() + "}}", html.escape(str(site[k])))
    out = tpl.replace("{{CSS}}", css).replace("{{DATA}}", data_json).replace("{{JS}}", js.replace("</script", "<\\/script"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    if LEVEL == "pre1":
        (OUT.parent / ".nojekyll").touch()
    else:
        write_manifest(site)

    print(f"【{site['name']}】{OUT.relative_to(ROOT)}")
    print("章 | 予定 | 作成 | 検算済 | 章名")
    tot = [0, 0, 0]
    for ch in chapters:
        n = len(ch["terms"])
        w = sum(1 for t in ch["terms"] if t["id"] in terms)
        v = sum(1 for t in ch["terms"] if t["status"] == "verified")
        tot = [tot[0] + n, tot[1] + w, tot[2] + v]
        print(f"{ch['no']:>2} | {n:>4} | {w:>4} | {v:>6} | {ch['title']}")
    print(f"合計: 予定 {tot[0]} / 作成 {tot[1]} / 検算済 {tot[2]} / 未作成 {tot[0] - tot[1]}")
    qa = sum(v[0] for v in qstat.values())
    print("問題（公開／全問）: " + "  ".join((f"第{k}章" if isinstance(k, int) else f"{k}の基本問題（共通の用語）") + f" {v[1]}/{v[0]}" for k, v in sorted(qstat.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))))
    print(f"問題 合計: 公開 {len(questions)} / 全問 {qa}（検算に通っていない問題は公開しない）")
    print("模擬試験（小問 / 検算OK / 見直しOK / 自動検査エラー）: " + ("  ".join(f"set{k:02d} {v[0]}/{v[1]}/{v[2]}/{v[3]}" for k, v in sorted(estat.items())) or "なし"))
    print(f"模擬試験 公開: {len(exams)} セット（3段階の確認に通ったセットだけ）")
    size = OUT.stat().st_size
    print(f"出力: {OUT.relative_to(ROOT)}  {size / 1024 / 1024:.2f} MB" + ("  （5MB超）" if size > 5 * 1024 * 1024 else ""))


def build_all():
    """準1級を作り、2級は別のプロセスで作ってから、両方のページを保存する service worker を書く。"""
    main()
    if not paths("g2")["content"].joinpath("plan.yaml").exists():
        write_pwa(OUT.read_bytes())
        return
    r = subprocess.run([sys.executable, __file__], env=dict(os.environ, JD_LEVEL="g2"), text=True)
    if r.returncode:
        sys.exit(r.returncode)
    write_pwa(OUT.read_bytes() + paths("g2")["out"].read_bytes())


if __name__ == "__main__":
    if os.environ.get("JD_LEVEL"):
        main()
    else:
        build_all()
