"""教科書（章ごとに頭から読み通す本文）の読み込み・HTML への変換・検査。build.py・validate.py・check_book.py から使う。

置き場所: <級の content>/book/chNN.md、検算は <級の verify>/book/chNN.py、検算の記録は <級の verify>/book/results.json。
統計学の教科書（tb）の学部の章（第1〜44章）は、2級・準1級の教科書の章を同じ順に並べて共通で使い、修士の章（第45章〜）だけ content/tb/book/ に書く。

本文の書き方（くわしくは spec/BOOK_GUIDE.md）:
  ## 見出し              … 節（ページ上部の「節へ移動」に出る）
  :::定義 題名 〜 :::    … 囲み（定義・定理・命題・例題・問題 は章ごとに番号が付く。例・ポイント・注意・発展・コラム は番号なし）
  :::解答 〜 :::         … 直前の例題・問題の解答（たたんで表示）
  [[用語id]]・[[用語id|表示]] … 辞典の用語ページへのリンク
  {{chart 1}}            … 前付け charts の 1 番目の図。{{chart 用語id 1}} は用語ページの図を使う
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import ROOT, paths  # noqa: E402

KINDS = {  # 囲みの種類 → (CSS の名前, 番号を付けるか)
    "定義": ("def", True), "定理": ("thm", True), "命題": ("thm", True), "例題": ("ex", True), "問題": ("prob", True),
    "例": ("eg", False), "ポイント": ("point", False), "注意": ("warn", False), "発展": ("adv", False), "コラム": ("col", False),
    "解答": ("ans", False),
}
BLOCK_RE = re.compile(r"^:::(\S+)[ \t]*(.*)$")
LINK_RE = re.compile(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]")
CHART_RE = re.compile(r"^\{\{chart(?: ([a-z0-9-]+))? (\d+)\}\}$")
LEVEL_LABEL = {"pre1": "準1級", "g2": "2級", "tb": "統計学"}


def book_dir(level):
    return paths(level)["content"] / "book"


def results_path(level):
    return paths(level)["verify"] / "book" / "results.json"


def file_hash(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16]


def load_results(level):
    p = results_path(level)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def split_front(text):
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


def parse_blocks(body):
    """本文を [("text", 文), ("block", 種類, 題名, 文)] に分ける。対応しない ::: は ValueError。"""
    out, buf, cur = [], [], None
    for ln in body.split("\n"):
        m = BLOCK_RE.match(ln.strip())
        if ln.strip() == ":::":
            if cur is None:
                raise ValueError("対応する ::: の開始がない閉じ ::: があります")
            out.append(("block", cur[0], cur[1], "\n".join(buf)))
            buf, cur = [], None
        elif m:
            if cur is not None:
                raise ValueError(f"囲み「{cur[0]}」が閉じる前に「{m.group(1)}」が始まっています（入れ子はできません）")
            if m.group(1) not in KINDS:
                raise ValueError(f"囲みの種類「{m.group(1)}」は使えません（{'・'.join(KINDS)}）")
            out.append(("text", "\n".join(buf)))
            buf, cur = [], (m.group(1), m.group(2).strip())
        else:
            buf.append(ln)
    if cur is not None:
        raise ValueError(f"囲み「{cur[0]}」が閉じていません")
    out.append(("text", "\n".join(buf)))
    return out


def all_titles():
    """すべての級の用語の id → 題名（リンク先がない級でも、表示する文字に使う）。"""
    titles = {}
    for lv in ("pre1", "g2", "tb"):
        p = paths(lv)["content"] / "plan.yaml"
        if p.exists():
            for ch in yaml.safe_load(p.read_text(encoding="utf-8"))["chapters"]:
                for t in ch.get("terms") or []:
                    titles.setdefault(t["id"], t["title"])
    return titles


def render_chapter(p, no, terms, titles, md_html):
    """1章を HTML にする。terms: この級で開ける用語 id の集合。md_html: Markdown（数式つき）→ HTML の関数。
    戻り値: (html, 本文に出てきた用語 id の一覧, 文字数)"""
    fm, body = split_front(p.read_text(encoding="utf-8"))
    links, store = [], {}

    def keep(html):
        key = f"BOOKX{len(store):04d}X"
        store[key] = html
        return key

    def link(m):
        tid, label = m.group(1), m.group(2)
        text = label or titles.get(tid, tid)
        if tid in terms:
            if tid not in links:
                links.append(tid)
            return keep(f'<button type="button" class="f-link" data-go="{tid}">{text}</button>')
        return text

    def chart(m):
        src, i = m.group(1), int(m.group(2)) - 1
        return "\n\n" + keep(f'<div class="chart-lazy" data-chart="{src or ""}:{i}"></div>') + "\n\n"

    def prep(text):
        text = LINK_RE.sub(link, text)
        return "\n".join(chart(CHART_RE.match(ln.strip())) if CHART_RE.match(ln.strip()) else ln for ln in text.split("\n"))

    count = {}
    parts = []
    for blk in parse_blocks(body):
        if blk[0] == "text":
            parts.append(md_html(prep(blk[1])))
            continue
        _, kind, title, text = blk
        css, numbered = KINDS[kind]
        inner = md_html(prep(text))
        if kind == "解答":
            parts.append(f'<details class="bx bx-ans"><summary>{title or "解答"}</summary><div class="bx-b">{inner}</div></details>')
            continue
        label = kind
        if numbered:
            count[kind] = count.get(kind, 0) + 1
            label += f" {no}.{count[kind]}"
        head = f'<p class="bx-h"><b>{label}</b>{"　" + LINK_RE.sub(lambda m: m.group(2) or titles.get(m.group(1), m.group(1)), title) if title else ""}</p>'
        parts.append(f'<div class="bx bx-{css}">{head}<div class="bx-b">{inner}</div></div>')
    h = "".join(parts)
    for k in reversed(list(store)):   # chart の置き場所は <p> で包まれるので外す
        h = h.replace(f"<p>{k}</p>", store[k]).replace(k, store[k])
    secs = re.split(r"<h2>(.*?)</h2>", h)
    out = [secs[0]] if secs[0].strip() else []
    for i in range(1, len(secs), 2):
        k = (i + 1) // 2
        out.append(f'<section class="sec" data-sec="b{k}" id="sec-b{k}"><h2>{secs[i].strip()}</h2>{secs[i + 1]}</section>')
    plain = re.sub(r"<[^>]+>|\$\$.*?\$\$|\$[^$]*\$", "", h, flags=re.S)
    return "".join(out), links, len(plain), fm


def entries(level):
    """この級の教科書の章の並び。[(キー, 元の級, 元の章番号, この級での章番号)]"""
    if level != "tb":
        plan = yaml.safe_load((paths(level)["content"] / "plan.yaml").read_text(encoding="utf-8"))
        return [(str(c["no"]), level, c["no"], c["no"]) for c in plan["chapters"]]
    import make_plan_tb as M
    out, seen = [], set()
    for i, (_, src, nos, _) in enumerate(M.UNDERGRAD):
        for no in nos:
            if (src, no) not in seen:
                seen.add((src, no))
                out.append((f"{src}-{no}", src, no, i + 1))
    plan = yaml.safe_load((paths("tb")["content"] / "plan.yaml").read_text(encoding="utf-8"))
    out += [(str(c["no"]), "tb", c["no"], c["no"]) for c in plan["chapters"] if c["no"] > len(M.UNDERGRAD)]
    return out


def src_titles():
    out = {}
    for lv in ("pre1", "g2", "tb"):
        p = paths(lv)["content"] / "plan.yaml"
        if p.exists():
            out[lv] = {c["no"]: c["title"] for c in yaml.safe_load(p.read_text(encoding="utf-8"))["chapters"]}
    return out


CH_REF = re.compile(r"(2級の|準1級の)?第(\s*)(\d+)((?:\s*[・〜]\s*\d+)*)(\s*)章")


def renumber(text, src, nums):
    """共通で使う章の本文の「第 N 章」を、元の級の番号からこの級の通し番号に直す（統計学の教科書用）。
    nums: {元の級: {元の章番号: この級での章番号}}。「2級の第 N 章」のように級を書いた参照も、その級の対応で直す。"""
    def sub(m):
        lv = {"2級の": "g2", "準1級の": "pre1"}.get(m.group(1) or "", src)
        table = nums.get(lv, {})
        parts = re.split(r"(\s*[・〜]\s*)", m.group(3) + m.group(4))
        out = []
        for i, s in enumerate(parts):
            if i % 2 == 0:
                n = table.get(int(s))
                if n is None:
                    return m.group(0)
                if len(out) >= 2 and out[-2] == str(n):  # 2 つの元の章が同じ章になったときは 1 つにまとめる
                    out.pop()
                    continue
                out.append(str(n))
            else:
                out.append(s.strip())
        return "第" + m.group(2) + "".join(out) + m.group(5) + "章"
    return CH_REF.sub(sub, text)


def load_book(level, terms, md_html):
    """build.py 用。検算に通り、内容が検算時から変わっていない章だけ本文を入れる（ほかは「準備中」として目次にだけ出す）。"""
    titles, stitles = all_titles(), src_titles()
    res = {lv: load_results(lv) for lv in ("pre1", "g2", "tb")}
    chs, stat = [], {"all": 0, "pub": 0}
    nums = {}
    for _, src, no, here in entries(level):
        nums.setdefault(src, {})[no] = here
    for key, src, no, here in entries(level):
        p = book_dir(src) / f"ch{no:02d}.md"
        stat["all"] += 1
        item = {"k": key, "no": here, "t": stitles[src][no]}
        if src != level:
            item["src"] = f"{LEVEL_LABEL[src]}の教科書 第{no}章"
        r = res[src].get(f"ch{no:02d}")
        if p.exists() and r and r.get("ok") and r.get("hash") == file_hash(p):
            html, links, n, fm = render_chapter(p, no, terms, titles, md_html)
            own = [t for t in fm.get("terms") or [] if t in terms]
            goals = [md_html(g).removeprefix("<p>").removesuffix("</p>") for g in fm.get("goals") or []]
            pre = fm.get("prereq", "")
            if src != level:
                html, pre = renumber(html, src, nums), renumber(pre, src, nums)
                goals = [renumber(g, src, nums) for g in goals]
            item.update({"html": html, "goals": goals,
                         "pre": pre, "terms": own + [t for t in links if t not in own],
                         "charts": fm.get("charts") or [], "min": max(5, round(n / 450))})
            if fm.get("title"):
                item["t"] = fm["title"]
            stat["pub"] += 1
        chs.append(item)
    return chs, stat


def check_chapter(p, level, plan_terms, customs):
    """validate.py 用。戻り値: (エラー, 警告)"""
    err, warn = [], []
    text = p.read_text(encoding="utf-8")
    try:
        fm, body = split_front(text)
    except yaml.YAMLError as e:
        return [f"前付けの YAML が読めません: {e}"], warn
    m = re.fullmatch(r"ch(\d{2})", p.stem)
    if not m:
        return [f"ファイル名は chNN.md にしてください"], warn
    if fm.get("chapter") != int(m.group(1)):
        err.append(f"chapter が {fm.get('chapter')} でファイル名と合いません")
    for k in ("title", "goals", "prereq"):
        if not fm.get(k):
            err.append(f"前付けに {k} がありません")
    for t in fm.get("terms") or []:
        if t not in plan_terms:
            err.append(f"terms の用語 {t} がこの級の plan.yaml にありません")
    try:
        blocks = parse_blocks(body)
    except ValueError as e:
        return err + [str(e)], warn
    last = None
    for b in blocks:
        if b[0] == "block":
            if b[1] == "解答" and last not in ("例題", "問題"):
                err.append("「解答」の囲みは、例題か問題の直後に置いてください")
            last = b[1]
        elif b[1].strip():
            last = None
    for tid, _ in LINK_RE.findall(body):
        if tid not in plan_terms:
            err.append(f"リンク先の用語 [[{tid}]] がこの級の plan.yaml にありません")
    charts = fm.get("charts") or []
    for ln in body.split("\n"):
        cm = CHART_RE.match(ln.strip())
        if ln.strip().startswith("{{") and not cm:
            err.append(f"図の指定の書き方が違います: {ln.strip()}")
        if cm and not cm.group(1) and not 1 <= int(cm.group(2)) <= len(charts):
            err.append(f"{ln.strip()}: 前付けの charts は {len(charts)} 個です")
        if cm and cm.group(1) and cm.group(1) not in plan_terms:
            err.append(f"{ln.strip()}: 用語 {cm.group(1)} がこの級にありません")
    for c in charts:
        if c.get("type") == "custom" and c.get("id") not in customs:
            err.append(f"charts のカスタム図 {c.get('id')} は src/charts_custom.js にありません")
    heads = re.findall(r"^## (.+)$", body, flags=re.M)
    if not heads or heads[0] != "はじめに":
        err.append("最初の節は「## はじめに」にしてください")
    if "まとめ" not in heads:
        err.append("「## まとめ」の節がありません")
    if "章末問題" not in heads:
        warn.append("「## 章末問題」の節がありません")
    if body.count("$$") % 2:
        err.append("$$ の数が奇数です")
    if not (paths(level)["verify"] / "book" / p.name.replace(".md", ".py")).exists():
        warn.append("検算のスクリプト verify/…/book/" + p.name.replace(".md", ".py") + " がありません（検算に通るまで公開されません）")
    # スマホ幅（375px）で 1 行に収まらない長さのインライン数式（おおよその目安：記号を除いた長さ 26 字超）
    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", re.sub(r"\$\$.*?\$\$", "", body, flags=re.S)):
        tex = re.sub(r"\\(?:frac|dfrac|tfrac)", "", m.group(1))
        tex = re.sub(r"\\(mathrm|text|operatorname|left|right|,|;|quad|qquad|displaystyle)", "", tex)
        tex = re.sub(r"\\[A-Za-z]+", "x", tex)
        if len(re.sub(r"[{}\s^_]", "", tex)) > 22:
            warn.append(f"長いインライン数式（スマホではみ出すおそれ）: ${m.group(1)[:60]}$")
    n = len(re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$|\s", "", body, flags=re.S))
    if n < 3000:
        warn.append(f"本文が短めです（{n} 字）")
    return err, warn
