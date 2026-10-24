"""verify/questions_chNN.py（準1級）と verify/g2/questions_ch*.py（2級）をすべて実行し、章ごとの公開可能な問題数を表示する。

使い方: .venv/bin/python scripts/verify_questions.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    bad = 0
    for p in sorted((ROOT / "verify").glob("questions_ch*.py")) + sorted((ROOT / "verify" / "g2").glob("questions_ch*.py")):
        r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, cwd=p.parent)
        lines = r.stdout.strip().splitlines()
        if p.parent.name == "g2":
            lines = [("2級 " + ln if "問題の検算" in ln else ln) for ln in lines]
            for ln in lines[:-1]:
                if "問題の検算" in ln:
                    print(ln)
        for ln in lines:
            if ln.startswith("[NG]") or ln.startswith("[未検算]"):
                print("  " + ln)
        print((lines[-1] if lines else "") + ("" if r.returncode == 0 else f"  （終了コード {r.returncode}）"))
        if r.returncode != 0:
            bad += 1
            if r.stderr:
                print(r.stderr[-800:])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
