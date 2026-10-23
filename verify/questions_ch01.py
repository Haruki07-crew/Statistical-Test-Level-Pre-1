"""第1章 事象と確率：問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

from fractions import Fraction as Fr
from itertools import product

q = Q(1)

# --- 条件付き確率 ---
# 1: P(A|B) != P(B|A) の反例（6の目 / 偶数の目）
die = range(1, 7)
A = {6}; B = {2, 4, 6}
pAB = Fr(len(A & B), 6) / Fr(len(B), 6)
pBA = Fr(len(A & B), 6) / Fr(len(A), 6)
q.check("q01-conditional-probability-1", pAB == Fr(1, 3) and pBA == 1 and pAB != pBA,
        "反例で P(A|B)=1/3, P(B|A)=1 → 一般には等しくない（answer=false）")

# 2: 非復元抽出、全事象の列挙で確認
balls = ["R"] * 5 + ["W"] * 3
cnt = tot = 0
for i in range(8):
    for j in range(8):
        if i == j:
            continue
        tot += 1
        cnt += balls[i] == "R" and balls[j] == "R"
q.num("q01-conditional-probability-2", cnt / tot)

# 3: 和が8の条件で6を含む確率
outs = [(a, b) for a in die for b in die]
cond = [o for o in outs if sum(o) == 8]
p = Fr(sum(6 in o for o in cond), len(cond))
choices = [Fr(2, 5), Fr(1, 3), Fr(11, 36), Fr(5, 36)]
q.check("q01-conditional-probability-3", p == choices[0] and all(c != p for c in choices[1:]),
        f"正解 {p}、誤答はいずれも一致しない")
q.check("q01-conditional-probability-3", Fr(sum(6 in o for o in outs), 36) == Fr(11, 36) and Fr(len(cond), 36) == Fr(5, 36),
        "誤答 11/36（条件なし）・5/36（和が8の確率）の由来")

# --- 全確率の公式 ---
q.num("q01-law-of-total-probability-1", 0.5 * 0.02 + 0.3 * 0.03 + 0.2 * 0.05)
# 2: 反例 A1=A2=Ω で右辺 = 2P(B) != P(B)
pB = Fr(1, 2)
q.check("q01-law-of-total-probability-2", 1 * pB + 1 * pB != pB, "重なりがあると右辺が 2P(B) になる反例（answer=false）")

# --- 事象の独立 ---
q.check("q01-independence-of-events-1", 0 != Fr(1, 6) * Fr(1, 2), "排反 P(A∩B)=0 と独立 P(A)P(B)>0 は両立しない（answer=true）")
q.num("q01-independence-of-events-2", 0.4 + 0.5 - 0.4 * 0.5)
coins = list(product("HT", repeat=2))
P = lambda ev: Fr(sum(1 for w in coins if ev(w)), 4)
EA = lambda w: w[0] == "H"; EB = lambda w: w[1] == "H"; EC = lambda w: w[0] == w[1]
pair = all(P(lambda w, x=x, y=y: x(w) and y(w)) == P(x) * P(y) for x, y in [(EA, EB), (EA, EC), (EB, EC)])
mutual = P(lambda w: EA(w) and EB(w) and EC(w)) == P(EA) * P(EB) * P(EC)
q.check("q01-independence-of-events-3", pair and not mutual, "どの2つも独立・相互独立ではない → 選択肢0のみ正しい")

# --- ベイズの定理 ---
post = Fr(4, 10) * Fr(3, 100) / (Fr(6, 10) * Fr(1, 100) + Fr(4, 10) * Fr(3, 100))
ch = [Fr(2, 3), Fr(4, 10), Fr(3, 100), Fr(1, 3)]
q.check("q01-bayes-theorem-1", post == ch[0] and all(c != post for c in ch[1:]), f"事後確率 {post}、誤答は一致しない")
q.num("q01-bayes-theorem-2", 0.02 * 0.9 / (0.02 * 0.9 + 0.98 * 0.05))
p1 = 0.02 * 0.9 / (0.02 * 0.9 + 0.98 * 0.05)
two = 0.02 * 0.81 / (0.02 * 0.81 + 0.98 * 0.0025)
seq = p1 * 0.9 / (p1 * 0.9 + (1 - p1) * 0.05)
q.num("q01-bayes-theorem-3", two)
q.check("q01-bayes-theorem-3", abs(two - seq) < 1e-12, "逐次更新と一括計算が一致")

# --- 条件付き独立 ---
pH1 = 0.5 * 0.5 + 0.5 * 0.8
pHH = 0.5 * 0.5**2 + 0.5 * 0.8**2
q.check("q01-conditional-independence-1", abs(pHH - pH1 * pH1) > 1e-6, f"反例：P(表表)={pHH} ≠ P(表)^2={pH1**2}（answer=false）")
q.num("q01-conditional-independence-2", pHH / pH1)

# 3: 導出。ランダムな同時分布で、C の下で条件付き独立なら P(A|B∩C)=P(A|C) を数値確認
import numpy as np
rng = np.random.default_rng(1)
ok = True
for _ in range(100):
    pc = rng.uniform(0.1, 0.9); a = rng.uniform(0.05, 0.95); b = rng.uniform(0.05, 0.95)
    pABC = pc * a * b           # C の下で独立に作る
    pBC = pc * b
    ok &= abs(pABC / pBC - a) < 1e-12
q.check("q01-conditional-independence-3", ok, "条件付き独立な分布で P(A|B∩C)=P(A|C) を数値確認")

# --- 包除原理（用語の見直しで追加） ---
import numpy as _np
_n = sum(1 for i in range(1, 211) if i % 2 == 0 or i % 3 == 0 or i % 7 == 0)
q.num("q01-inclusion-exclusion-principle-1", _n)
q.check("q01-inclusion-exclusion-principle-1", 105 + 70 + 30 - 35 - 15 - 10 + 5 == _n, "包除原理の式と数え上げが一致")
_ok = 1 - 3 * (5 / 6) ** 5 + 3 * (4 / 6) ** 5 - (1 / 2) ** 5
_rng = _np.random.default_rng(1)
_sim = _rng.integers(1, 7, size=(400000, 5))
_mc = _np.mean((_sim == 1).any(axis=1) & (_sim == 2).any(axis=1) & (_sim == 3).any(axis=1))
from itertools import product as _prod
_exact = sum(1 for w in _prod(range(1, 7), repeat=5) if {1, 2, 3} <= set(w)) / 6 ** 5
_wrong = [1 - 3 * (5 / 6) ** 5 + 3 * (4 / 6) ** 5, (1 - (5 / 6) ** 5) ** 3, 1 - (1 / 2) ** 5]
q.check("q01-inclusion-exclusion-principle-2", abs(_exact - _ok) < 1e-12 and round(_ok, 3) == 0.158 and abs(_mc - _ok) < 0.005, f"正解 0.158（全列挙 {_exact:.5f}）")
q.check("q01-inclusion-exclusion-principle-2", [round(w, 3) for w in _wrong] == [0.189, 0.214, 0.969] and all(abs(w - _ok) > 0.02 for w in _wrong), "誤答 0.189/0.214/0.969 は正解と異なる")

# --- 導出問題（追加） ---
def _deriv01():
    import sympy as _S
    pi_, s_, f_ = _S.symbols("pi s f", positive=True)
    Pp = s_ * pi_ + f_ * (1 - pi_)                      # 全確率の公式
    post = s_ * pi_ / Pp
    odds = post / (f_ * (1 - pi_) / Pp)
    ok1 = _S.simplify(odds - (s_ / f_) * pi_ / (1 - pi_)) == 0
    O = s_ * pi_ / (f_ * (1 - pi_))
    ok2 = _S.simplify(O / (1 + O) - post) == 0
    v = post.subs({pi_: _S.Rational(2, 100), s_: _S.Rational(9, 10), f_: _S.Rational(5, 100)})
    q.check("q01-bayes-theorem-4", ok1 and ok2 and abs(float(v) - 0.2687) < 1e-3,
            "sympy: 事後オッズ=尤度比×事前オッズ、O/(1+O)=sπ/(sπ+f(1-π))（π=0.02,s=0.9,f=0.05 で 0.269）")
_deriv01()

q.done()
