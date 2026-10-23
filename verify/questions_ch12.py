"""第12章（一般の分布に関する検定法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb, ceil
from scipy import stats

q = Q(12)

# --- 母比率の検定 ---
z = (228 - 400 * 0.5) / np.sqrt(400 * 0.25)
q.num("q12-proportion-test-1", z)
crit = lambda za: ceil(20 + za * np.sqrt(100 * 0.2 * 0.8))
c = crit(1.645)
q.check("q12-proportion-test-2", c == 27 and (27 - 20) / 4 >= 1.645 > (26 - 20) / 4, "正解 27（26 では Z=1.5 で未達）")
q.check("q12-proportion-test-2", crit(1.96) == 28 and crit(2.326) == 30, "誤答 28（両側1.96）、30（2.326）は正解と異なる")
q.concept("q12-proportion-test-3", "検定の分母は p0 を使う（数値なし）")

# --- 二項検定 ---
q.num("q12-binomial-test-1", stats.binom.sf(8, 10, 0.5))
q.check("q12-binomial-test-1", abs(stats.binom.sf(8, 10, 0.5) - 11 / 1024) < 1e-12, "11/1024 と一致")
sf = lambda cc: stats.binom.sf(cc - 1, 15, 0.2)
cmin = min(cc for cc in range(16) if sf(cc) <= 0.05)
q.check("q12-binomial-test-2", cmin == 7, "最小の c は 7")
q.check("q12-binomial-test-2", abs(sf(5) - 0.164) < 5e-4 and abs(sf(6) - 0.061) < 5e-4 and abs(sf(7) - 0.018) < 5e-4,
        "解説の 0.164, 0.061, 0.018")
p_right = stats.binom.sf(6, 12, 0.3)
q.num("q12-binomial-test-3", 2 * p_right)
q.check("q12-binomial-test-3", p_right < stats.binom.cdf(7, 12, 0.3) and abs(p_right - 0.0386) < 5e-5,
        "右側が小さいほうの片側 p 値で 0.0386")

# --- ポアソン母数の検定 ---
q.num("q12-poisson-parameter-test-1", stats.poisson.sf(16, 10))
q.num("q12-poisson-parameter-test-2", (4.5 - 4) / np.sqrt(4 / 50))

# --- 2標本比率 ---
p1, p2, pp = 60 / 200, 66 / 300, 126 / 500
z2 = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / 200 + 1 / 300))
q.num("q12-two-sample-proportion-test-1", z2)
chi = stats.chi2_contingency([[60, 140], [66, 234]], correction=False)[0]
q.check("q12-two-sample-proportion-test-2", abs(chi - z2 ** 2) < 1e-9 and abs(chi - 4.07) < 0.01, "χ²=Z²=4.074（正しい）")

# --- 適合度 ---
q.num("q12-goodness-of-fit-test-1", stats.chisquare([25, 17, 15, 23, 24, 16]).statistic)
O = np.array([99, 27, 24, 10]); E = 160 * np.array([9, 3, 3, 1]) / 16
x2 = stats.chisquare(O, E).statistic
q.check("q12-goodness-of-fit-test-2", abs(x2 - 2.40) < 1e-9 and x2 < stats.chi2.ppf(0.95, 3)
        and abs(stats.chi2.ppf(0.95, 3) - 7.815) < 1e-3, "正解 2.40・棄却されない（df=3）")
q.check("q12-goodness-of-fit-test-2", abs(((O - E) ** 2 / O).sum() - 2.65) < 0.005 and ((O - E) ** 2).sum() == 126,
        "誤答 2.65（観測度数で割る）・126（割り忘れ）は正解と異なる")

# --- ピアソン統計量 ---
Os = np.array([12, 30, 8, 50]); Es = 100 * np.array([0.1, 0.3, 0.2, 0.4])
q.check("q12-pearson-chi-square-statistic-1", abs(((Os - Es) ** 2 / Es).sum() - ((Os ** 2 / Es).sum() - 100)) < 1e-9,
        "簡便式が一致（正しい）")
X, n, p0 = sp.symbols("X n p0", positive=True)
chi_sym = (X - n * p0) ** 2 / (n * p0) + ((n - X) - n * (1 - p0)) ** 2 / (n * (1 - p0))
q.check("q12-pearson-chi-square-statistic-2", sp.simplify(chi_sym - (X - n * p0) ** 2 / (n * p0 * (1 - p0))) == 0,
        "k=2 の χ² が Z² に一致（sympy）")

# --- 独立性 ---
q.num("q12-independence-test-1", stats.chi2_contingency([[30, 20], [20, 30]], correction=False)[0])
q.check("q12-independence-test-2", (3 - 1) * (4 - 1) == 6 and stats.chi2_contingency(np.arange(1, 13).reshape(3, 4) + 5)[2] == 6
        and len({6, 11, 12, 5}) == 4, "自由度 6（scipy でも 6）、誤答は異なる")
r = stats.chi2_contingency([[20, 15, 15], [10, 15, 25]], correction=False)
q.num("q12-independence-test-3", r[0])
q.check("q12-independence-test-3", r[2] == 2 and r[0] < stats.chi2.ppf(0.95, 2), "df=2 で 5.991 未満")

# --- 同質性：行和を固定した期待度数と独立性の期待度数が一致 ---
T = np.array([[18, 22, 10], [30, 45, 25]])
Eind = stats.contingency.expected_freq(T)
Ehom = np.outer(T.sum(1), T.sum(0) / T.sum())
q.check("q12-homogeneity-test-1", np.allclose(Eind, Ehom), "期待度数・統計量は同じ（異なる＝誤り）")

# --- G 統計量 ---
Og = np.array([30, 50, 20]); Eg = np.array([25, 50, 25])
G = 2 * (Og * np.log(Og / Eg)).sum()
q.num("q12-g-statistic-1", G)
q.check("q12-g-statistic-1", abs(stats.power_divergence(Og, Eg, lambda_="log-likelihood").statistic - G) < 1e-9
        and abs(stats.chisquare(Og, Eg).statistic - 2.0) < 1e-9, "scipy の G と一致・ピアソンは 2.00")
Tg = np.array([[10, 20, 15], [25, 12, 18], [8, 9, 30]])
rg = stats.chi2_contingency(Tg, lambda_="log-likelihood")
q.check("q12-g-statistic-2", rg[2] == 4 and rg[2] != 3 * 3 - 1, "G の自由度も (r-1)(c-1)=4（rc-1 ではない＝誤り）")

# --- 期待度数 ---
q.num("q12-expected-frequency-1", 60 * 45 / 150)

# --- 自由度 ---
q.check("q12-degrees-of-freedom-gof-1", 6 - 1 - 1 == 4 and len({4, 5, 3, 6}) == 4, "6-1-1=4、誤答は異なる")
q.check("q12-degrees-of-freedom-gof-2", 8 - 1 - 2 == 5, "8-1-2=5（正しい）")

# --- コクランの規則 ---
def cochran(e):
    e = np.array(e)
    return (e >= 1).all() and (e < 5).mean() <= 0.2
cands = [[12, 9, 8, 8, 7, 6, 6, 5, 4, 3], [12, 9, 8, 8, 7, 6, 5, 4, 3, 2],
         [15, 12, 10, 9, 8, 7, 6, 6, 5, 0.8], [10, 8, 6, 4.5, 4.5, 4, 3, 3, 2, 2]]
q.check("q12-chi-square-approximation-condition-1", [cochran(c_) for c_ in cands] == [True, False, False, False],
        "1つ目だけが規則を満たす")

# --- フィッシャーの z 変換 ---
zr = np.arctanh(0.6); se = 1 / np.sqrt(25)
lo, hi = np.tanh(zr - 1.96 * se), np.tanh(zr + 1.96 * se)
q.num("q12-variance-stabilizing-transform-1", lo)
q.check("q12-variance-stabilizing-transform-1", abs(hi - 0.795) < 5e-4, "上限 0.795")

# --- 2標本ポアソン ---
pv = stats.binom.sf(29, 35, 2 / 3)
q.num("q12-poisson-two-sample-test-1", pv)
q.check("q12-poisson-two-sample-test-1", abs(stats.binomtest(30, 35, 2 / 3, alternative="greater").pvalue - pv) < 1e-12,
        "binomtest と一致")

# --- KS ---
q.num("q12-kolmogorov-smirnov-test-1", stats.kstest([0.1, 0.3, 0.35, 0.9], "uniform").statistic)

# --- 分散指数 ---
x = np.array([2, 5, 0, 7, 1, 3])
D = ((x - x.mean()) ** 2).sum() / x.mean()
q.num("q12-dispersion-test-1", D)
q.check("q12-dispersion-test-1", D > stats.chi2.ppf(0.95, 5) and abs(stats.chi2.ppf(0.95, 5) - 11.07) < 5e-3,
        "χ²(5) の上側5%点 11.07 を超える")

# --- 導出問題（追加） ---
def _deriv12():
    import numpy as _N
    import sympy as _S
    E, d = _S.symbols("E d", positive=True)
    term = (E + d) * _S.log(1 + d / E)
    ser = _S.series(term, d, 0, 3).removeO()
    ok = _S.simplify(ser - (d + d ** 2 / (2 * E))) == 0
    # 数値：期待度数に近い観測で G と X² がほぼ一致、差は 3 次の大きさ
    Ev = _N.array([25.0, 50.0, 25.0])
    for Ov in (_N.array([27.0, 49.0, 24.0]), _N.array([30.0, 50.0, 20.0])):
        G = 2 * (Ov * _N.log(Ov / Ev)).sum(); X2 = ((Ov - Ev) ** 2 / Ev).sum()
        ok &= abs(G - X2) < 0.05 * X2 + 0.01
    q.check("q12-g-statistic-3", bool(ok), "sympy: (E+d)log(1+d/E)=d+d²/(2E)+O(d³)、数値で G≈X²")
_deriv12()

q.done()
