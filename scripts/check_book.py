"""教科書の章の検算を実行し、結果（ファイルのハッシュつき）を <級の verify>/book/results.json に記録する。
build.py は、検算に通っていて、そのあと本文が変わっていない章だけを公開する。

使い方: JD_LEVEL=g2 .venv/bin/python scripts/check_book.py [--ch 3] [--status]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level import LEVEL, ROOT, paths  # noqa: E402
from book import book_dir, file_hash, load_results, results_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    files = sorted(book_dir(LEVEL).glob("ch*.md"))
    res = load_results(LEVEL)
    if a.status:
        for p in files:
            r = res.get(p.stem)
            st = "未検算" if not r else ("NG" if not r["ok"] else ("OK" if r["hash"] == file_hash(p) else "検算後に変更あり"))
            print(f"{p.stem}: {st}")
        return
    if a.ch:
        files = [p for p in files if p.stem == f"ch{a.ch:02d}"]
    bad = 0
    for p in files:
        script = paths(LEVEL)["verify"] / "book" / f"{p.stem}.py"
        if not script.exists():
            print(f"{p.stem}: 検算のスクリプト {script.relative_to(ROOT)} がありません")
            res[p.stem] = {"ok": False, "hash": file_hash(p), "t": time.strftime("%Y-%m-%d %H:%M")}
            bad += 1
            continue
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT,
                           env={"PYTHONPATH": str(ROOT / "verify"), "PATH": "/usr/bin:/bin"})
        out = (r.stdout + r.stderr).strip().splitlines()
        for ln in out:
            if ln.startswith("[NG]") or "Error" in ln or "Traceback" in ln:
                print("  " + ln)
        print(f"{p.stem}: {out[-1] if out else '(出力なし)'}")
        res[p.stem] = {"ok": r.returncode == 0, "hash": file_hash(p), "t": time.strftime("%Y-%m-%d %H:%M")}
        bad += r.returncode != 0
    rp = results_path(LEVEL)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
