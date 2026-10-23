"""第5章（離散型分布）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb
from scipy import stats

q = Q(5)

# --- ベルヌーイ ---
P = sp.Symbol("p")
var = P * (1 - P)
pstar = sp.solve(sp.diff(var, P), P)[0]
q.check("q05-bernoulli-distribution-1", pstar == sp.Rational(1, 2) and var.subs(P, pstar) == sp.Rational(1, 4)
        and var.subs(P, pstar) != sp.Rational(1, 2), "最大は p=1/2 で値は 1/4（1/2 ではない＝誤り）")
p = 0.2
m3 = p * (1 - p) ** 3 + (1 - p) * (-p) ** 3
q.num("q05-bernoulli-distribution-2", m3)
q.check("q05-bernoulli-distribution-2", abs(stats.bernoulli(p).stats("s") * (p * (1 - p)) ** 1.5 - m3) < 1e-12,
        "scipy の歪度から求めた3次中心モーメントと一致")

# --- 二項 ---
q.num("q05-binomial-distribution-1", stats.binom.pmf(2, 6, 0.4))
cands = [(40, 0.3), (40, 0.7), (60, 0.2), (30, 0.4)]
ok = [abs(n * pp - 12) < 1e-9 and abs(n * pp * (1 - pp) - 8.4) < 1e-9 for n, pp in cands]
q.check("q05-binomial-distribution-2", ok == [True, False, False, False], "正解は (40,0.3) だけが期待値12・分散8.4")
approx = stats.norm.sf((59.5 - 50) / 5)
q.num("q05-binomial-distribution-3", approx)
q.check("q05-binomial-distribution-3", abs(stats.binom.sf(59, 100, 0.5) - 0.0284) < 5e-5
        and abs(stats.norm.sf(2) - 0.0228) < 5e-5, "解説の正確値0.0284・修正なし0.0228")

# --- 超幾何 ---
q.num("q05-hypergeometric-distribution-1", stats.hypergeom.pmf(2, 12, 5, 4))
q.num("q05-hypergeometric-distribution-2", stats.hypergeom.var(20, 8, 5))
N, K, n = 20, 8, 5
q.check("q05-hypergeometric-distribution-3", abs(stats.hypergeom.mean(N, K, n) - n * K / N) < 1e-12,
        "非復元でも期待値は nK/N（小さくならない＝誤り）")

# --- 有限母集団修正 ---
q.num("q05-finite-population-correction-1", np.sqrt((500 - 100) / 499 * 400 / 100))
# 全数調査：母集団を実際に作って n=N の標本平均の分散が0
pop = np.arange(1, 8, dtype=float)
Np = len(pop)
q.check("q05-finite-population-correction-2", (Np - Np) / (Np - 1) == 0
        and abs(Np * (3 / Np) * (1 - 3 / Np) * (Np - Np) / (Np - 1)) < 1e-12, "n=N で FPC=0、超幾何の分散も0")

# --- フィッシャーの正確検定 ---
res = stats.fisher_exact([[4, 1], [1, 4]], alternative="greater")
q.num("q05-fisher-exact-test-1", res.pvalue)
q.check("q05-fisher-exact-test-1", abs(res.pvalue - (comb(5, 4) * comb(5, 1) + 1) / comb(10, 5)) < 1e-12,
        "超幾何で手計算した 26/252 と一致")
q.concept("q05-fisher-exact-test-2", "周辺固定で超幾何分布になることの記述（数値なし）")

# --- ポアソン ---
q.num("q05-poisson-distribution-1", stats.poisson.pmf(2, 3))
conv = sum(stats.poisson.pmf(k, 1.5) * stats.poisson.pmf(2 - k, 2.5) for k in range(3))
q.num("q05-poisson-distribution-2", conv)
q.check("q05-poisson-distribution-2", abs(conv - stats.poisson.pmf(2, 4)) < 1e-12, "畳み込み=Po(4)")
cond = stats.poisson.pmf(1, 1.5) * stats.poisson.pmf(3, 2.5) / stats.poisson.pmf(4, 4)
q.num("q05-poisson-distribution-3", cond)
q.check("q05-poisson-distribution-3", abs(cond - stats.binom.pmf(1, 4, 0.375)) < 1e-12, "Bin(4,0.375) と一致")

# --- 幾何（失敗回数。scipy の geom は試行回数なので loc=-1 でずらす） ---
g = lambda pp: stats.geom(pp, loc=-1)
q.num("q05-geometric-distribution-1", g(0.25).mean())
q.check("q05-geometric-distribution-1", g(0.25).pmf(0) == 0.25, "台が0から（失敗回数の定義）")
q.num("q05-geometric-distribution-2", g(0.2).sf(3))
cp = g(0.3).sf(6) / g(0.3).sf(2)
q.check("q05-geometric-distribution-3", abs(cp - 0.7 ** 4) < 1e-12 and abs(round(0.7 ** 4, 3) - 0.240) < 1e-9,
        "正解 0.7^4=0.240")
wrong = [0.7 ** 7, 0.7 ** 3, 0.3 * 0.7 ** 4]
q.check("q05-geometric-distribution-3", all(abs(w - cp) > 0.05 for w in wrong)
        and abs(g(0.3).pmf(4) - 0.3 * 0.7 ** 4) < 1e-12, "誤答は正解と異なる（0.3×0.7^4 は P(X=4)）")

# --- 負の二項（scipy の nbinom は失敗回数） ---
q.num("q05-negative-binomial-distribution-1", stats.nbinom.pmf(2, 3, 0.5))
okd = True
for r, pp in [(3, 0.4), (5, 0.7), (1, 0.25)]:
    m, v = stats.nbinom.stats(r, pp, moments="mv")
    okd &= abs(m - r * (1 - pp) / pp) < 1e-9 and abs(v - r * (1 - pp) / pp ** 2) < 1e-9
    # 幾何分布の和として（モンテカルロではなく畳み込みで）
    pm = np.array([stats.geom.pmf(k + 1, pp) for k in range(400)])
    conv_ = pm.copy()
    for _ in range(r - 1):
        conv_ = np.convolve(conv_, pm)[:400]
    okd &= np.allclose(conv_[:30], stats.nbinom.pmf(np.arange(30), r, pp))
q.check("q05-negative-binomial-distribution-2", bool(okd), "E=rq/p, V=rq/p^2、幾何分布の和が NB と一致")

# --- 多項 ---
q.num("q05-multinomial-distribution-1", stats.multinomial.pmf([2, 2, 1], 5, [0.5, 0.3, 0.2]))
cov = stats.multinomial.cov(10, [0.5, 0.3, 0.2])[0, 1]
q.check("q05-multinomial-distribution-2", abs(cov - (-1.5)) < 1e-12, "正解 -1.5")
q.check("q05-multinomial-distribution-2", all(abs(w - cov) > 1e-6 for w in [0, 1.5, -0.15]), "誤答は正解と異なる")

# --- 導出問題（追加） ---
def _deriv05():
    import sympy as _S
    from scipy import stats as _st
    N_, K_, n_ = _S.symbols("N K n", positive=True)
    p = K_ / N_
    cov = K_ * (K_ - 1) / (N_ * (N_ - 1)) - K_ ** 2 / N_ ** 2
    ok = _S.simplify(cov + K_ * (N_ - K_) / (N_ ** 2 * (N_ - 1))) == 0
    V = n_ * p * (1 - p) + n_ * (n_ - 1) * cov
    ok &= _S.simplify(V - n_ * p * (1 - p) * (N_ - n_) / (N_ - 1)) == 0
    for (NN, KK, nn) in [(20, 8, 5), (12, 5, 4), (50, 7, 20)]:
        m, v = _st.hypergeom.stats(NN, KK, nn, moments="mv")
        ok &= abs(m - nn * KK / NN) < 1e-9 and abs(v - float(V.subs({N_: NN, K_: KK, n_: nn}))) < 1e-9
    q.check("q05-hypergeometric-distribution-4", bool(ok), "sympy: Cov=-K(N-K)/(N²(N-1))、V=np(1-p)(N-n)/(N-1)、scipy と一致")
    lam = _S.symbols("lambda", positive=True); k = _S.symbols("k", integer=True, nonnegative=True)
    pmf = _S.exp(-lam) * lam ** k / _S.factorial(k)
    E1 = _S.simplify(_S.summation(k * pmf, (k, 0, _S.oo)))
    E2 = _S.simplify(_S.summation(k * (k - 1) * pmf, (k, 0, _S.oo)))
    ok2 = _S.simplify(E1 - lam) == 0 and _S.simplify(E2 - lam ** 2) == 0 and _S.simplify(E2 + E1 - E1 ** 2 - lam) == 0
    q.check("q05-poisson-distribution-4", bool(ok2), "sympy: Σk p=λ、Σk(k-1)p=λ²、V=λ")
_deriv05()

q.done()
