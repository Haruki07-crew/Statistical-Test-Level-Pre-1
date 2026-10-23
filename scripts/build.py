"""content/ から docs/index.html を生成する。

使い方: .venv/bin/python scripts/build.py
先に validate.py を実行し、エラーがあれば止める。
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
SRC = ROOT / "src"
OUT = ROOT / "docs" / "index.html"

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


def load_questions(plan_ids):
    """検算に通り、内容が検算時から変わっていない問題だけを返す。戻り値：(問題の一覧, 章ごとの {全問, 公開})"""
    sys.path.insert(0, str(ROOT / "verify"))
    from qcheck import qhash
    out, stat = [], {}
    for p in sorted((CONTENT / "questions").glob("ch*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        res_path = ROOT / "verify" / "questions_results" / f"{p.stem}.json"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
        qs = data.get("questions") or []
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
        stat[data.get("chapter")] = (len(qs), pub)
    return out, stat


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
    """ホーム画面に追加するための manifest・アイコン・service worker を docs/ に置く。"""
    import hashlib
    docs = OUT.parent
    pwa = SRC / "pwa"
    for f in pwa.iterdir():
        if f.name == "sw.js":
            ver = hashlib.sha256(index_bytes).hexdigest()[:12]
            (docs / f.name).write_text(f.read_text(encoding="utf-8").replace("{{VERSION}}", ver), encoding="utf-8")
        elif f.is_file():
            (docs / f.name).write_bytes(f.read_bytes())


def plain_text(body):
    t = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.S)
    t = re.sub(r"\$[^$\n]*\$", " ", t)
    t = re.sub(r"^#+ ", "", t, flags=re.M)
    t = re.sub(r"[*_`|>#]", "", t)
    t = re.sub(r"-{3,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate.py")], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode != 0:
        print(r.stdout)
        print("validate.py でエラーがあるためビルドを中止します")
        sys.exit(1)
    print(f"validate: {last}")

    plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    terms = {}
    for p in sorted(CONTENT.glob("ch*/*.md")):
        fm, body = split_front(p.read_text(encoding="utf-8"))
        terms[fm["id"]] = {
            "id": fm["id"], "title": fm["title"], "reading": fm.get("reading", ""), "en": fm.get("en", "") or "",
            "aliases": [str(a) for a in fm.get("aliases") or []], "chapter": fm["chapter"],
            "importance": fm["importance"], "tags": [str(t) for t in fm.get("tags") or []],
            "related": fm.get("related") or [], "charts": fm.get("charts") or [],
            "html": render_body(body),  # 検索用の本文テキストはファイルサイズ節約のため app.js で html から作る
        }
    chapters = []
    for ch in plan["chapters"]:
        chapters.append({
            "no": ch["no"], "title": ch["title"],
            "terms": [dict({"id": t["id"], "title": t["title"], "importance": t["importance"], "status": t["status"]},
                           **({"see": t["see"]} if t.get("see") else {}))
                      for t in ch.get("terms") or []],
        })
    plan_ids = {t["id"] for ch in chapters for t in ch["terms"]}
    questions, qstat = load_questions(plan_ids)
    data = {"chapters": chapters, "terms": terms, "questions": questions, "formulas": render_formulas({t["title"]: t["id"] for ch in chapters for t in ch["terms"] if t["id"] in terms})}
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    tpl = (SRC / "template.html").read_text(encoding="utf-8")
    css = (SRC / "app.css").read_text(encoding="utf-8")
    js = "\n".join((SRC / f).read_text(encoding="utf-8") for f in ("dist.js", "charts.js", "charts_custom.js", "app.js"))
    out = tpl.replace("{{CSS}}", css).replace("{{DATA}}", data_json).replace("{{JS}}", js.replace("</script", "<\\/script"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    (OUT.parent / ".nojekyll").touch()
    write_pwa(out.encode("utf-8"))

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
    print("問題（公開／全問）: " + "  ".join(f"第{k}章 {v[1]}/{v[0]}" for k, v in sorted(qstat.items())))
    print(f"問題 合計: 公開 {len(questions)} / 全問 {qa}（検算に通っていない問題は公開しない）")
    size = OUT.stat().st_size
    print(f"出力: {OUT.relative_to(ROOT)}  {size / 1024 / 1024:.2f} MB" + ("  （5MB超）" if size > 5 * 1024 * 1024 else ""))


if __name__ == "__main__":
    main()
