"""数式の全数チェック。用語の本文・問題・公式まとめ・模擬試験・ロードマップのすべての文から、
(1) $ の対応崩れ、(2) 数式の外に残った LaTeX コマンドや $、(3) 数式の中の { } ・\\left \\right・\\begin \\end の対応崩れ
を探す。build.py と同じ変換（render_body）を通した結果で調べるので、Markdown 変換で壊れた場合も見つかる。

node と mathjax-full が使えるときは（環境変数 MJ_DIR に mathjax-full を入れたフォルダを指定）、
全数式を MathJax で実際に組み、エラーになる数式を一覧にする（verify/tex_node.mjs）。

使い方: .venv/bin/python scripts/check_math.py [--tex]   （validate.py からも呼ばれる）
終了コード: 問題があれば 1
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import CONTENT  # noqa: E402  級ごとの置き場所（JD_LEVEL）
sys.path.insert(0, str(ROOT / "scripts"))
from build import render_body, split_front  # noqa: E402

LATEX_OUT = re.compile(r"\\(?:frac|dfrac|tfrac|sum|prod|int|mathrm|mathbf|boldsymbol|sqrt|bar|hat|tilde|alpha|beta|gamma|delta|"
                       r"epsilon|varepsilon|theta|lambda|mu|sigma|Sigma|rho|chi|phi|Phi|pi|omega|left|right|begin|end|le|ge|leq|geq|"
                       r"neq|ne|cdot|times|infty|partial|log|exp|text|operatorname|lim|binom|overline|underline|mid|in|sim|approx)(?![A-Za-z])")
Q_TEXT_KEYS = ("question", "hint", "explanation", "answer", "choices", "steps")


def texts():
    """(場所, 文) をすべて返す。"""
    for p in sorted(CONTENT.glob("ch*/*.md")):
        _, body = split_front(p.read_text(encoding="utf-8"))
        yield f"{p.relative_to(ROOT)}", body
    for p in sorted((CONTENT / "questions").glob("ch*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for q in data.get("questions") or []:
            for k in Q_TEXT_KEYS:
                v = q.get(k)
                if q.get("type") in ("choice", "numeric", "truefalse") and k == "answer":
                    continue
                for i, s in enumerate(v if isinstance(v, list) else [v]):
                    if isinstance(s, str):
                        yield f"{p.relative_to(ROOT)} {q.get('id')} {k}{'[%d]' % i if isinstance(v, list) else ''}", s
    f = CONTENT / "formulas.md"
    if f.exists():
        t = f.read_text(encoding="utf-8")
        if t.startswith("---\n"):
            t = split_front(t)[1]
        yield "content/formulas.md", t
    import exam_lib
    for p in exam_lib.set_files():
        try:
            data, _ = exam_lib.load_set(p)   # generator の値を差し込んだあとの文で調べる
        except Exception as e:  # 書きかけのセットは check_exam.py がエラーとして報告するので、ここでは飛ばす
            print(f"[SKIP] {p.relative_to(ROOT)}: 読み込めないため数式チェックを省略（{str(e)[:80]}）")
            continue
        for blk in data.get("problems") or []:
            for k in ("context", "title"):
                if isinstance(blk.get(k), str):
                    yield f"{p.relative_to(ROOT)} {blk.get('id')} {k}", blk[k]
            for q in blk.get("items") or []:
                for k in ("question", "explanation", "choices", "unit", "answer_format", "related_note"):
                    v = q.get(k)
                    for i, s in enumerate(v if isinstance(v, list) else [v]):
                        if isinstance(s, str):
                            yield f"{p.relative_to(ROOT)} {q.get('id')} {k}{'[%d]' % i if isinstance(v, list) else ''}", s
    for p in sorted((CONTENT / "book").glob("ch*.md")):   # 教科書：囲みの行・リンク・図の指定を外した本文で調べる
        _, body = split_front(p.read_text(encoding="utf-8"))
        body = re.sub(r"^:::.*$|^\{\{chart.*\}\}$", "", body, flags=re.M)
        body = re.sub(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), body)
        yield f"{p.relative_to(ROOT)}", body
    r = CONTENT / "roadmap.yaml"
    if r.exists():
        data = yaml.safe_load(r.read_text(encoding="utf-8")) or {}
        for t in data.get("terms") or []:
            if isinstance(t.get("reason"), str):
                yield f"content/roadmap.yaml {t.get('id')} reason", t["reason"]


def math_parts(body):
    """本文から (表示数式か, TeX) を取り出す（build.py の protect_math と同じ規則）。"""
    b = body.replace("\\$", "")
    out = [(True, m.group(1)) for m in re.finditer(r"\$\$(.+?)\$\$", b, flags=re.S)]
    b = re.sub(r"\$\$(.+?)\$\$", " ", b, flags=re.S)
    out += [(False, m.group(1)) for m in re.finditer(r"\$([^$\n]+?)\$", b)]
    rest = re.sub(r"\$([^$\n]+?)\$", " ", b)
    return out, rest


def brace_ok(tex):
    t = re.sub(r"\\[{}]", "", tex.replace("\\\\", ""))
    d = 0
    for c in t:
        d += c == "{"
        d -= c == "}"
        if d < 0:
            return False
    return d == 0


def check_text(where, body, problems):
    parts, rest = math_parts(body)
    # (1) $ の対応
    for ln, line in enumerate(rest.split("\n"), 1):
        if "$" in line:
            problems.append((where, f"{ln} 行目: $ の対応が崩れています: {line.strip()[:60]}"))
    # (3) 数式の中
    for disp, tex in parts:
        if not brace_ok(tex):
            problems.append((where, f"{{ }} の対応が崩れています: {tex[:60]}"))
        nl, nr = len(re.findall(r"\\left(?![a-z])", tex)), len(re.findall(r"\\right(?![a-z])", tex))
        if nl != nr:
            problems.append((where, f"\\left と \\right の数が合いません: {tex[:60]}"))
        if sorted(re.findall(r"\\begin\{(\w+\*?)\}", tex)) != sorted(re.findall(r"\\end\{(\w+\*?)\}", tex)):
            problems.append((where, f"\\begin と \\end が合いません: {tex[:60]}"))
    # (2) 変換後の HTML で、数式の外に残った LaTeX
    h = render_body(body)
    h = re.sub(r"<m>.*?</m>", " ", h, flags=re.S)
    h = re.sub(r'<div class="math-display">.*?</div>', " ", h, flags=re.S)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", h)).replace("\\$", "")
    for m in LATEX_OUT.finditer(txt):
        problems.append((where, f"数式の外に LaTeX が残っています: …{txt[max(0, m.start() - 20):m.end() + 20].strip()}…"))
    if "$" in txt:
        i = txt.index("$")
        problems.append((where, f"数式の外に $ が残っています: …{txt[max(0, i - 20):i + 20].strip()}…"))
    return parts


def tex_test(all_math):
    """mathjax-full で全数式を組む。戻り値: (調べた数, エラーの一覧) / 使えなければ None"""
    mj = os.environ.get("MJ_DIR")
    if not mj or not (Path(mj) / "node_modules" / "mathjax-full").exists():
        return None
    uniq = sorted({(d, t) for _, d, t in all_math})
    tmp = ROOT / "verify" / "ui_out"
    tmp.mkdir(parents=True, exist_ok=True)
    src = tmp / "tex_all.json"
    src.write_text(json.dumps([{"d": d, "t": t} for d, t in uniq], ensure_ascii=False), encoding="utf-8")
    r = subprocess.run(["node", str(ROOT / "verify" / "tex_node.mjs"), str(src)], capture_output=True, text=True,
                       env=dict(os.environ, MJ_DIR=mj))
    if r.returncode not in (0, 1):
        print(r.stderr[-2000:])
        return None
    bad = json.loads(r.stdout)
    where = {}
    for w, d, t in all_math:
        where.setdefault((d, t), w)
    return len(uniq), [(where.get((b["d"], b["t"]), "?"), b["t"], b["err"]) for b in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true", help="mathjax-full で全数式を組む（MJ_DIR が必要）")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    problems, all_math, n = [], [], 0
    for where, body in texts():
        n += 1
        for d, t in check_text(where, body, problems):
            all_math.append((where, d, t))
    for w, msg in problems:
        print(f"[MATH] {w}: {msg}")
    print(f"数式チェック: 文 {n} 件 / 数式 {len(all_math)} 個 / 問題 {len(problems)} 件")
    bad_tex = 0
    if args.tex or os.environ.get("MJ_DIR"):
        res = tex_test(all_math)
        if res is None:
            print("MathJax 変換テスト: 省略（MJ_DIR に mathjax-full がありません）")
        else:
            cnt, bad = res
            for w, t, e in bad:
                print(f"[TEX]  {w}: {e}: {t[:80]}")
            print(f"MathJax 変換テスト: 異なる数式 {cnt} 個 / エラー {len(bad)} 個")
            bad_tex = len(bad)
    sys.exit(1 if problems or bad_tex else 0)


if __name__ == "__main__":
    main()
