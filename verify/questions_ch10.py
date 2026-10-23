"""第10章（検定の基礎と検定法の導出）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb
from scipy import stats

q = Q(10)
N = stats.norm

# --- 仮説 ---
q.concept("q10-null-and-alternative-hypothesis-1", "仮説の立て方（数値なし）")
q.concept("q10-null-and-alternative-hypothesis-2", "棄却されないことの解釈（数値なし）")

# --- 過誤 ---
q.num("q10-type-i-and-ii-errors-1", N.sf(0.6 / (1 / 3)))
q.num("q10-type-i-and-ii-errors-2", N.cdf((0.6 - 1) / (1 / 3)))
a2, b2 = N.sf(0.8 * 3), N.cdf((0.8 - 1) * 3)
q.check("q10-type-i-and-ii-errors-3", a2 < N.sf(1.8) and b2 > N.cdf(-1.2) and abs(a2 - 0.008) < 5e-4 and abs(b2 - 0.274) < 5e-4,
        "c を上げると α は減り β は増える（誤り）")

# --- 有意水準 ---
q.num("q10-significance-level-1", stats.binom.sf(8, 10, 0.5))
q.check("q10-significance-level-1", abs(stats.binom.sf(7, 10, 0.5) - 0.0547) < 5e-5, "X>=8 は 0.0547")
mus = np.linspace(-3, 0, 301)
K = N.sf(1.645 - np.sqrt(16) * mus)
q.check("q10-significance-level-2", abs(K.max() - N.sf(1.645)) < 1e-12 and abs(N.sf(1.645) - 0.05) < 1e-4, "上限は μ=0 で 0.05（正しい）")

# --- p 値 ---
q.num("q10-p-value-1", 2 * N.sf(2.3))
q.num("q10-p-value-2", stats.binom.sf(9, 12, 0.5))
q.check("q10-p-value-2", abs(stats.binomtest(10, 12, 0.5, alternative="greater").pvalue - 79 / 4096) < 1e-12, "binomtest と一致")
rng = np.random.default_rng(10)
z = rng.standard_normal(200000)
pv = N.sf(z)
q.check("q10-p-value-3", stats.kstest(pv, "uniform").pvalue > 0.01 and abs(np.mean(pv <= 0.05) - 0.05) < 0.003, "p 値は一様（正しい）")

# --- 棄却域 ---
q.check("q10-rejection-region-1", abs(N.isf(0.005) - 2.576) < 5e-4 and abs(2 * N.sf(2.326) - 0.02) < 1e-3
        and abs(2 * N.sf(1.96) - 0.05) < 1e-3, "正解 2.576")
q.num("q10-rejection-region-2", 50 + 1.645 * 6 / np.sqrt(36))

# --- 検出力 ---
d = np.sqrt(25) * 5 / 10
q.num("q10-power-function-1", N.cdf(d - 1.645))
q.num("q10-power-function-2", N.cdf(d - 1.96) + N.cdf(-d - 1.96))
q.check("q10-one-sided-two-sided-test-1", 1.8 > N.isf(0.05) and 1.8 < N.isf(0.025)
        and abs(N.sf(1.8) - 0.036) < 5e-4 and abs(2 * N.sf(1.8) - 0.072) < 5e-4, "片側で棄却・両側で非棄却（正しい）")

# --- ネイマン・ピアソン ---
xs = sp.symbols("x1:5", real=True)
n_ = 4
lr = sp.exp(-sum((x - 1) ** 2 for x in xs) / 2 + sum(x ** 2 for x in xs) / 2)
q.check("q10-neyman-pearson-lemma-1", sp.expand(-sum((x - 1) ** 2 for x in xs) / 2 + sum(x ** 2 for x in xs) / 2 - (sum(xs) - sp.Rational(n_, 2))) == 0, "尤度比 = exp(Σx - n/2)")
sim = rng.standard_normal((200000, 9)).mean(1)
q.check("q10-neyman-pearson-lemma-1", abs(np.mean(sim > N.isf(0.05) / 3) - 0.05) < 0.003, "c=z_α/√n でサイズ α")
c = -np.log(0.95)
q.check("q10-neyman-pearson-lemma-2", abs(stats.expon.cdf(c) - 0.05) < 1e-12, "サイズ 0.05")
q.num("q10-neyman-pearson-lemma-2", stats.expon(scale=1 / 2).cdf(c))

# --- UMP ---
q.concept("q10-uniformly-most-powerful-test-1", "両側対立に UMP がないこと（数値なし）")
cmin = next(k for k in range(21) if stats.binom.sf(k - 1, 20, 0.3) <= 0.05)
q.num("q10-uniformly-most-powerful-test-2", cmin)
q.check("q10-uniformly-most-powerful-test-2", abs(stats.binom.sf(8, 20, 0.3) - 0.113) < 5e-4 and abs(stats.binom.sf(9, 20, 0.3) - 0.048) < 5e-4,
        "P(X>=9)=0.113, P(X>=10)=0.048")

# --- 単調尤度比 ---
T, n, l0, l1 = sp.symbols("T n lambda0 lambda1", positive=True)
ratio = (l1 / l0) ** T * sp.exp(-n * (l1 - l0))
dT = sp.diff(ratio, T)
q.check("q10-monotone-likelihood-ratio-1", all(float(dT.subs({T: t, n: 5, l0: 1, l1: 2})) > 0 for t in range(0, 30)), "T について増加（正しい）")

# --- 尤度比検定 ---
l = lambda lam: 30 * np.log(lam) - 10 * lam
q.num("q10-likelihood-ratio-test-1", 2 * (l(3) - l(2)))
q.check("q10-likelihood-ratio-test-1", 2 * (l(3) - l(2)) > stats.chi2.isf(0.05, 1), "3.841 を超える")
ys = sp.symbols("y1:6")
mu0, s2 = sp.symbols("mu0 sigma2", positive=True)
yb = sum(ys) / 5
ell = lambda m: -sum((y - m) ** 2 for y in ys) / (2 * s2)
q.check("q10-likelihood-ratio-test-2", sp.simplify(2 * (ell(yb) - ell(mu0)) - 5 * (yb - mu0) ** 2 / s2) == 0, "-2logΛ = n(x̄-μ0)²/σ²")

# --- ウィルクス ---
q.check("q10-wilks-theorem-1", (4 - 2) == 2 and len({2, 3, 4, 1}) == 4, "自由度 4-2=2")
q.num("q10-wilks-theorem-2", stats.chi2.sf(6.2, 2))

# --- ワルド・スコア ---
q.num("q10-wald-test-1", 0.1 / np.sqrt(0.6 * 0.4 / 100))
q.concept("q10-wald-test-2", "ワルドは I(θ̂) を使う（数値なし）")
p = sp.Symbol("p", positive=True)
U = (60 - 100 * p) / (p * (1 - p))
nI = 100 / (p * (1 - p))
q.num("q10-score-test-1", float((U / sp.sqrt(nI)).subs(p, sp.Rational(1, 2))))
lam = sp.Symbol("lambda", positive=True)
Ul = 30 / lam - 10
S = (Ul ** 2 / (10 / lam)).subs(lam, 2)
q.num("q10-score-test-2", float(S))
q.check("q10-score-test-2", abs(10 / 3 * (3 - 2) ** 2 - 3.333) < 1e-3 and abs(2 * (l(3) - l(2)) - 4.328) < 1e-3, "ワルド 3.333、LR 4.328")

# --- 多重検定 ---
q.num("q10-multiple-testing-1", 1 - 0.95 ** 10)
q.num("q10-multiple-testing-2", 1 - 0.95 ** (1 / 5))
q.num("q10-bonferroni-correction-1", 0.05 / 8)
pv = np.array([0.004, 0.011, 0.020, 0.30])
nb = int((pv <= 0.05 / 4).sum())
q.check("q10-bonferroni-correction-2", nb == 2 and int((pv <= 0.05).sum()) == 3, "ボンフェローニで2個")


def holm(p, a=0.05):
    ps = np.sort(p)
    m = len(p)
    k = 0
    for i, v in enumerate(ps):
        if v <= a / (m - i):
            k += 1
        else:
            break
    return k


q.num("q10-holm-method-1", holm(pv))
ok = True
for _ in range(2000):
    pp = rng.uniform(0, 0.1, 6)
    ok &= holm(pp) >= int((pp <= 0.05 / 6).sum())
q.check("q10-holm-method-2", bool(ok), "ランダムな p 値でホルム ≥ ボンフェローニ（正しい）")


def bh(p, qq=0.05):
    ps = np.sort(p)
    m = len(p)
    ks = [k for k in range(1, m + 1) if ps[k - 1] <= k / m * qq]
    return max(ks) if ks else 0


pf = np.array([0.010, 0.028, 0.035, 0.039, 0.60])
q.num("q10-false-discovery-rate-1", bh(pf))
try:
    from scipy.stats import false_discovery_control
    q.check("q10-false-discovery-rate-1", int((false_discovery_control(pf) <= 0.05).sum()) == 4 and holm(pf) == 1,
            "scipy の BH 調整と一致、ホルムは1個")
except ImportError:
    q.check("q10-false-discovery-rate-1", holm(pf) == 1, "ホルムは1個")

# --- 標本サイズ ---
q.num("q10-sample-size-determination-test-1", int(np.ceil(((1.645 + 0.842) * 10 / 4) ** 2)))
q.check("q10-sample-size-determination-test-1", int(np.ceil(((1.96 + 0.842) * 2.5) ** 2)) == 50
        and N.cdf(np.sqrt(39) * 0.4 - 1.645) >= 0.8 > N.cdf(np.sqrt(38) * 0.4 - 1.645), "両側50、n=39で検出力80%以上")

# --- 無作為化検定 ---
g = (0.05 - stats.binom.sf(8, 10, 0.5)) / stats.binom.pmf(8, 10, 0.5)
q.num("q10-randomized-test-1", g)
q.check("q10-randomized-test-1", abs(stats.binom.sf(8, 10, 0.5) + g * stats.binom.pmf(8, 10, 0.5) - 0.05) < 1e-12, "サイズちょうど0.05")

q.done()
