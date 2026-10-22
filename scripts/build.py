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
        return f'<span class="math-inline">${html.escape(tex, quote=False)}$</span>'

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
            "html": render_body(body), "text": plain_text(body),
        }
    chapters = []
    for ch in plan["chapters"]:
        chapters.append({
            "no": ch["no"], "title": ch["title"],
            "terms": [{"id": t["id"], "title": t["title"], "importance": t["importance"], "status": t["status"]}
                      for t in ch.get("terms") or []],
        })
    data = {"chapters": chapters, "terms": terms}
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    tpl = (SRC / "template.html").read_text(encoding="utf-8")
    css = (SRC / "app.css").read_text(encoding="utf-8")
    js = "\n".join((SRC / f).read_text(encoding="utf-8") for f in ("dist.js", "charts.js", "app.js"))
    out = tpl.replace("{{CSS}}", css).replace("{{DATA}}", data_json).replace("{{JS}}", js.replace("</script", "<\\/script"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    (OUT.parent / ".nojekyll").touch()

    print("章 | 予定 | 作成 | 検算済 | 章名")
    tot = [0, 0, 0]
    for ch in chapters:
        n = len(ch["terms"])
        w = sum(1 for t in ch["terms"] if t["id"] in terms)
        v = sum(1 for t in ch["terms"] if t["status"] == "verified")
        tot = [tot[0] + n, tot[1] + w, tot[2] + v]
        print(f"{ch['no']:>2} | {n:>4} | {w:>4} | {v:>6} | {ch['title']}")
    print(f"合計: 予定 {tot[0]} / 作成 {tot[1]} / 検算済 {tot[2]} / 未作成 {tot[0] - tot[1]}")
    size = OUT.stat().st_size
    print(f"出力: {OUT.relative_to(ROOT)}  {size / 1024 / 1024:.2f} MB" + ("  （5MB超）" if size > 5 * 1024 * 1024 else ""))


if __name__ == "__main__":
    main()
