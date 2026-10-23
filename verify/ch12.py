"""第12章 検算スクリプト（適合度検定・分割表・比率とポアソンの検定）。
数値は scipy.stats（chisquare, chi2_contingency, power_divergence, binomtest, fisher_exact, kstest など）、
記号計算は sympy（k=2 での χ²=Z²、G の2次近似、条件付き二項分布の導出）。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5 など）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の数倍）。
"""
from math import sqrt, comb, exp, factorial, log
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


rng = np.random.default_rng(20261023)
norm, chi2, binom, poisson = stats.norm, stats.chi2, stats.binom, stats.poisson


def pearson(O, E):
    O, E = np.asarray(O, float), np.asarray(E, float)
    return float(((O - E) ** 2 / E).sum())


def gstat(O, E):
    O, E = np.asarray(O, float), np.asarray(E, float)
    m = O > 0
    return float(2 * (O[m] * np.log(O[m] / E[m])).sum())


# ================= binomial-test =================
T = "binomial-test"
cs = [comb(15, k) for k in (12, 13, 14, 15)]
check(T, "例1 二項係数 455,105,15,1", cs == [455, 105, 15, 1], cs)
check(T, "例1 和=576", sum(cs) == 576 and 2 ** 15 == 32768, sum(cs))
p1 = binom.sf(11, 15, 0.5)
check(T, "例1 P(X≥12)=576/32768=0.0176", close(p1, 0.0176) and close(576 / 32768, p1, 1e-15), p1)
r = stats.binomtest(12, 15, 0.5, alternative="greater")
check(T, "例1 binomtest 右片側 p=0.0176", close(r.pvalue, 0.0176), r.pvalue)
check(T, "例1 P(X≥11)=0.0592", close(binom.sf(10, 15, 0.5), 0.0592), binom.sf(10, 15, 0.5))
check(T, "例1 棄却域 {X≥12}（最小の c）", binom.sf(10, 15, .5) > .05 and binom.sf(11, 15, .5) <= .05, "c=12")
r2 = stats.binomtest(12, 15, 0.5)
check(T, "例1 両側 p=0.0352", close(r2.pvalue, 0.0352), r2.pvalue)
check(T, "図1 F(11)=0.982", close(binom.cdf(11, 15, .5), 0.982, 5e-4), binom.cdf(11, 15, .5))
check(T, "例2 P(X=0)=0.8^20=0.0115", close(binom.pmf(0, 20, .2), 0.0115), binom.pmf(0, 20, .2))
check(T, "例2 P(X=1)=0.0576", close(binom.pmf(1, 20, .2), 0.0576) and close(20 * .2 * .8 ** 19, binom.pmf(1, 20, .2), 1e-12), binom.pmf(1, 20, .2))
check(T, "例2 左片側 P(X≤1)=0.0692", close(binom.cdf(1, 20, .2), 0.0692), binom.cdf(1, 20, .2))
check(T, "例2 2倍法 0.1384", close(2 * binom.cdf(1, 20, .2), 0.1384), 2 * binom.cdf(1, 20, .2))
pm = binom.pmf(np.arange(21), 20, .2)
ks_set = [k for k in range(21) if pm[k] <= pm[1] * (1 + 1e-7)]
check(T, "例2 P(X=k)≤P(X=1) は k=0,1,7..20", ks_set == [0, 1] + list(range(7, 21)), ks_set)
pb = pm[ks_set].sum()
check(T, "例2 (b) 合計 0.1559", close(pb, 0.1559), pb)
check(T, "例2 binomtest 両側（(b)法）と一致", close(stats.binomtest(1, 20, .2).pvalue, pb, 1e-12), stats.binomtest(1, 20, .2).pvalue)
check(T, "例2 どちらも5%で棄却されない", pb > .05 and 2 * binom.cdf(1, 20, .2) > .05, "")
# p0=0.5 で (a)=(b)（n=5..30, 全 x）
ok = True
for n in range(5, 31):
    for x in range(n + 1):
        a = min(1, 2 * min(binom.cdf(x, n, .5), binom.sf(x - 1, n, .5)))
        ok &= abs(a - stats.binomtest(x, n, .5).pvalue) < 1e-12
check(T, "p0=0.5 では (a)2倍法=(b)（n=5..30 全x）", ok, ok)
# 実サイズ ≤ α（n=5..60, p0 の格子）
ok = True
for n in range(5, 61):
    for p0 in (0.1, 0.2, 0.3, 0.5, 0.7):
        c = next(c for c in range(n + 2) if binom.sf(c - 1, n, p0) <= 0.05)
        ok &= binom.sf(c - 1, n, p0) <= 0.05
check(T, "右片側の実サイズ ≤ 0.05（n=5..60）", ok, ok)

# ================= chi-square-approximation-condition =================
T = "chi-square-approximation-condition"
tab = np.array([[3, 7], [8, 2]])
c2, pv, dof, E = stats.chi2_contingency(tab, correction=False)
check(T, "期待度数 5.5, 4.5", np.allclose(E, [[5.5, 4.5], [5.5, 4.5]]), E.tolist())
check(T, "5未満は 2/4 セル=50%", (E < 5).sum() == 2, (E < 5).mean())
check(T, "公式 20·(3·2−7·8)²/(10·10·11·9)=50000/9900", 20 * (3 * 2 - 7 * 8) ** 2 == 50000 and 10 * 10 * 11 * 9 == 9900, 50000 / 9900)
check(T, "χ²=5.051", close(c2, 5.051, 5e-4), c2)
check(T, "p=0.0246", close(pv, 0.0246), pv)
cy, pvy, _, _ = stats.chi2_contingency(tab, correction=True)
check(T, "イェーツ χ²_Y=3.232", close(cy, 3.232, 5e-4), cy)
check(T, "イェーツ p=0.0722", close(pvy, 0.0722), pvy)
pf = stats.fisher_exact(tab).pvalue
check(T, "フィッシャー両側 p=0.0698", close(pf, 0.0698), pf)
check(T, "修正なしは有意・正確検定は有意でない", pv < .05 < pf, (pv, pf))
check(T, "図 χ²(1) 右すそ≈0.025（5.05）", close(chi2.sf(5.05, 1), 0.025, 5e-4), chi2.sf(5.05, 1))
check(T, "図 正確 p≈0.070", close(pf, 0.070, 5e-4), pf)

# ================= degrees-of-freedom-gof =================
T = "degrees-of-freedom-gof"
days = np.array([22, 25, 20, 15, 10, 8])
lam = (np.arange(6) * days).sum() / 100
check(T, "λ̂=190/100=1.9", (np.arange(6) * days).sum() == 190 and lam == 1.9, lam)
Ep = 100 * poisson.pmf(np.arange(4), 1.9)
Etail = 100 * poisson.sf(3, 1.9)
Eall = np.r_[Ep, Etail]
claimE = [14.957, 28.418, 26.997, 17.098, 12.530]
check(T, "期待度数 14.957,28.418,26.997,17.098,12.530", np.allclose(Eall, claimE, atol=5e-4), Eall.round(4).tolist())
check(T, "E(4)=8.12, E(5)=3.09（5件単独は5未満）", 100 * poisson.pmf(5, 1.9) < 5, (100 * poisson.pmf(4, 1.9), 100 * poisson.pmf(5, 1.9)))
O5 = np.array([22, 25, 20, 15, 18])
contrib = (O5 - Eall) ** 2 / Eall
check(T, "寄与 3.317,0.411,1.814,0.257,2.388", np.allclose(contrib, [3.317, .411, 1.814, .257, 2.388], atol=5e-4), contrib.round(4).tolist())
xs = contrib.sum()
check(T, "χ²=8.187", close(xs, 8.187, 5e-4), xs)
r = stats.chisquare(O5, Eall, ddof=1)
check(T, "chisquare(ddof=1) 統計量一致", close(r.statistic, xs, 1e-10), r.statistic)
check(T, "χ²_0.05(3)=7.815", close(chi2.isf(.05, 3), 7.815, 5e-4), chi2.isf(.05, 3))
check(T, "p=0.0423（ν=3）", close(r.pvalue, 0.0423), r.pvalue)
check(T, "χ²_0.05(4)=9.488", close(chi2.isf(.05, 4), 9.488, 5e-4), chi2.isf(.05, 4))
check(T, "p=0.0850（ν=4）", close(chi2.sf(xs, 4), 0.0850), chi2.sf(xs, 4))
check(T, "図 5%点 7.81 と 9.49、8.19 はその間", close(chi2.isf(.05, 3), 7.81, 5e-3) and close(chi2.isf(.05, 4), 9.49, 5e-3) and 7.81 < 8.19 < 9.49, "")
check(T, "図 params の observed/expected が本文と一致", list(O5) == [22, 25, 20, 15, 18], claimE)
check(T, "0件と4件以上が期待より多い", O5[0] > Eall[0] and O5[4] > Eall[4], "")
# 自由度の表：分割表 (rc-1)-(r-1)-(c-1)=(r-1)(c-1)
r_, c_ = sp.symbols("r c")
check(T, "(rc−1)−(r−1)−(c−1)=(r−1)(c−1)", sp.expand(r_ * c_ - 1 - (r_ - 1) - (c_ - 1) - (r_ - 1) * (c_ - 1)) == 0, "sympy")
# シミュレーション：Po(1.9), n=100, 5カテゴリ。グループ化 MLE → χ²(3)、生データ平均 → χ²(3)〜χ²(4) の間
Nsim = 20000
X = rng.poisson(1.9, (Nsim, 100))
cnt = np.stack([(X == k).sum(1) for k in range(4)] + [(X >= 4).sum(1)], 1)
from scipy.optimize import minimize_scalar


def grouped_mle(o):
    def nll(l):
        pr = np.r_[poisson.pmf(np.arange(4), l), poisson.sf(3, l)]
        return -(o * np.log(pr)).sum()
    return minimize_scalar(nll, bounds=(0.5, 5), method="bounded").x


def stat_for(l, o):
    e = 100 * np.r_[poisson.pmf(np.arange(4), l), poisson.sf(3, l)]
    return ((o - e) ** 2 / e).sum()


S_raw = np.array([stat_for(X[i].mean(), cnt[i]) for i in range(Nsim)])
S_grp = np.array([stat_for(grouped_mle(cnt[i]), cnt[i]) for i in range(5000)])
rej_grp = (S_grp >= chi2.isf(.05, 3)).mean()
rej_raw3 = (S_raw >= chi2.isf(.05, 3)).mean()
rej_raw4 = (S_raw >= chi2.isf(.05, 4)).mean()
check(T, "sim グループ化MLE: χ²(3) の5%点での棄却率≈0.05（5000回, 許容±0.012）", abs(rej_grp - .05) <= .012, rej_grp)
check(T, "sim 生データ平均: 平均が3と4の間（2万回）", 3 - 0.05 < S_raw.mean() < 4 + 0.05, S_raw.mean())
check(T, "sim 生データ平均: ν=3 なら棄却率≥0.05−0.006、ν=4 なら≤0.05+0.006", rej_raw3 >= .044 and rej_raw4 <= .056, (rej_raw3, rej_raw4))

# ================= expected-frequency =================
T = "expected-frequency"
tab = np.array([[45, 15], [50, 30], [25, 35]])
c2, pv, dof, E = stats.chi2_contingency(tab, correction=False)
check(T, "例1 期待度数 36,24/48,32/36,24", np.allclose(E, [[36, 24], [48, 32], [36, 24]]), E.tolist())
check(T, "例1 行和・列和が観測と一致", np.allclose(E.sum(1), tab.sum(1)) and np.allclose(E.sum(0), tab.sum(0)), (E.sum(1), E.sum(0)))
contrib = ((tab - E) ** 2 / E).ravel()
check(T, "例1 寄与 2.250,3.375,0.083,0.125,3.361,5.042", np.allclose(contrib, [2.25, 3.375, .083, .125, 3.361, 5.042], atol=5e-4), contrib.round(4).tolist())
check(T, "例1 χ²=14.236", close(c2, 14.236, 5e-4), c2)
check(T, "例1 自由度2", dof == 2, dof)
check(T, "例1 p=0.00081", close(pv, 0.00081, 5e-6), pv)
check(T, "例1 全体の「利用する」60%", tab[:, 0].sum() / 200 == 0.6, 0.6)
check(T, "例2 100×55/300=18.333", close(100 * 55 / 300, 18.333, 5e-4), 100 * 55 / 300)
check(T, "読み解き 0.3×0.4=12%", close(0.3 * 0.4, 0.12, 1e-12), 0.12)
# 2×2 で |O−E| = |ad−bc|/n（喫煙例）
t22 = np.array([[30, 70], [25, 175]])
E22 = stats.contingency.expected_freq(t22)
dev = np.abs(t22 - E22)
check(T, "2×2 の |O−E| は全セル |ad−bc|/n", np.allclose(dev, abs(30 * 175 - 70 * 25) / 300), dev.tolist())
# ランダムな表で (O−E) の行和・列和が0
ok = True
for _ in range(200):
    tt = rng.integers(1, 50, (rng.integers(2, 6), rng.integers(2, 6)))
    D = tt - stats.contingency.expected_freq(tt)
    ok &= np.allclose(D.sum(0), 0) and np.allclose(D.sum(1), 0)
check(T, "(O−E) の行和・列和=0（ランダム200表）", ok, ok)
check(T, "E[O_i]=nπ_i（多項分布, n=50, π=(.2,.3,.5), 10万回, 許容±0.05）",
      np.allclose(rng.multinomial(50, [.2, .3, .5], 100000).mean(0), [10, 15, 25], atol=.05), "")

# ================= g-statistic =================
T = "g-statistic"
O1 = [14, 22, 18, 25, 16, 25]
G1 = gstat(O1, [20] * 6)
check(T, "例1 G=5.588", close(G1, 5.588, 5e-4), G1)
check(T, "例1 power_divergence(lambda_=0) と一致", close(stats.power_divergence(O1, lambda_="log-likelihood").statistic, G1, 1e-10), "")
check(T, "例1 χ²=5.500", close(pearson(O1, [20] * 6), 5.5, 1e-12), pearson(O1, [20] * 6))
check(T, "例1 p(G)=0.348", close(chi2.sf(G1, 5), 0.348, 5e-4), chi2.sf(G1, 5))
check(T, "例1 p(χ²)=0.358", close(chi2.sf(5.5, 5), 0.358, 5e-4), chi2.sf(5.5, 5))
O2 = [12, 17, 16, 19, 14, 42]
cb = (np.array(O2) - 20) ** 2 / 20
check(T, "例2 寄与 3.20,0.45,0.80,0.05,1.80,24.20", np.allclose(cb, [3.2, .45, .8, .05, 1.8, 24.2]), cb.tolist())
check(T, "例2 χ²=30.50", close(cb.sum(), 30.5, 1e-12), cb.sum())
check(T, "例2 p(χ²)=1.2e-5", close(chi2.sf(30.5, 5), 1.2e-5, 5e-7), chi2.sf(30.5, 5))
G2 = gstat(O2, [20] * 6)
check(T, "例2 G=25.46", close(G2, 25.46, 5e-3), G2)
check(T, "例2 p(G)=1.1e-4", close(chi2.sf(G2, 5), 1.1e-4, 5e-6), chi2.sf(G2, 5))
check(T, "図1 G=25.46 の右すそ≈0.0001", close(chi2.sf(25.46, 5), 1e-4, 5e-5), chi2.sf(25.46, 5))
check(T, "図1 X²=30.5 の右すそ≈0.00001", close(chi2.sf(30.5, 5), 1e-5, 5e-6), chi2.sf(30.5, 5))
check(T, "図2 params observed が例2と一致", O2 == [12, 17, 16, 19, 14, 42], O2)
tab3 = np.array([[50, 30, 20], [30, 40, 30]])
E3 = stats.contingency.expected_freq(tab3)
check(T, "例3 χ²=8.429", close(pearson(tab3, E3), 8.429, 5e-4), pearson(tab3, E3))
G3 = gstat(tab3, E3)
check(T, "例3 G=8.500", close(G3, 8.5, 5e-4), G3)
check(T, "例3 p(G)=0.0143（ν=2）", close(chi2.sf(G3, 2), 0.0143), chi2.sf(G3, 2))
check(T, "例3 chi2_contingency(lambda_=log-likelihood) と一致", close(stats.chi2_contingency(tab3, lambda_="log-likelihood")[0], G3, 1e-10), "")
# 2次近似：G の δ についての展開の2次項が Σδ²/E
d, e = sp.symbols("delta E", positive=True)
term = 2 * (e + d) * sp.log(1 + d / e)
ser = sp.series(term, d, 0, 3).removeO()
check(T, "2(E+δ)log(1+δ/E) = 2δ + δ²/E + O(δ³)", sp.simplify(ser - (2 * d + d ** 2 / e)) == 0, ser)
# Λ = Π(E/O)^O → −2logΛ = 2ΣO log(O/E)（数値）
O = np.array(O1, float)
check(T, "−2log Π(E/O)^O = G", close(-2 * np.log(np.prod((20 / O) ** O)), G1, 1e-9), "")
check(T, "対数の中を E/O にすると符号反転", close(gstat(O1, [20] * 6), -2 * (O * np.log(20 / O)).sum(), 1e-12), "")
# シミュレーション：公平なさいころ n=120 で G ~ χ²(5)
M = rng.multinomial(120, [1 / 6] * 6, 200000)
Gs = 2 * np.where(M > 0, M * np.log(np.where(M > 0, M, 1) / 20), 0).sum(1)
Xs = ((M - 20) ** 2 / 20).sum(1)
rg, rx = (Gs >= chi2.isf(.05, 5)).mean(), (Xs >= chi2.isf(.05, 5)).mean()
check(T, "sim G の5%棄却率≈0.05（20万回, 許容±0.004）", abs(rg - .05) <= .004, rg)
check(T, "sim X² の5%棄却率≈0.05（20万回, 許容±0.003）", abs(rx - .05) <= .003, rx)

# ================= goodness-of-fit-test =================
T = "goodness-of-fit-test"
cb = (np.array(O1) - 20) ** 2 / 20
check(T, "例1 寄与 1.80,0.20,0.20,1.25,0.80,1.25", np.allclose(cb, [1.8, .2, .2, 1.25, .8, 1.25]), cb.tolist())
r = stats.chisquare(O1)
check(T, "例1 χ²=5.50", close(r.statistic, 5.5, 1e-12), r.statistic)
check(T, "例1 χ²_0.05(5)=11.07", close(chi2.isf(.05, 5), 11.07, 5e-3), chi2.isf(.05, 5))
check(T, "例1 p=0.358", close(r.pvalue, 0.358, 5e-4), r.pvalue)
check(T, "図2 F(5.5)≈0.642", close(chi2.cdf(5.5, 5), 0.642, 5e-4), chi2.cdf(5.5, 5))
check(T, "図1 params observed が例1と一致", O1 == [14, 22, 18, 25, 16, 25], O1)
O = [125, 70, 35, 20]
E = 250 * np.array([.4, .3, .2, .1])
check(T, "例2 期待度数 100,75,50,25", np.allclose(E, [100, 75, 50, 25]), E.tolist())
cb = (np.array(O) - E) ** 2 / E
check(T, "例2 寄与 6.250,0.333,4.500,1.000", np.allclose(cb, [6.25, .333, 4.5, 1], atol=5e-4), cb.round(4).tolist())
r = stats.chisquare(O, E)
check(T, "例2 χ²=12.083", close(r.statistic, 12.083, 5e-4), r.statistic)
check(T, "例2 χ²_0.05(3)=7.815", close(chi2.isf(.05, 3), 7.815, 5e-4), "")
check(T, "例2 p=0.0071", close(r.pvalue, 0.0071), r.pvalue)
check(T, "簡便式 ΣO²/E−n（例1・例2）", close((np.array(O1) ** 2 / 20).sum() - 120, 5.5, 1e-10) and close((np.array(O) ** 2 / E).sum() - 250, r.statistic, 1e-10), "")
# 2カテゴリで χ²=Z²
x, n, p0 = 124, 200, .55
z = (x - n * p0) / sqrt(n * p0 * (1 - p0))
check(T, "2カテゴリ χ²=Z²（124/200, p0=.55）", close(stats.chisquare([x, n - x], [n * p0, n * (1 - p0)]).statistic, z ** 2, 1e-10), z ** 2)
M = rng.multinomial(120, [1 / 6] * 6, 200000)
Xs = ((M - 20) ** 2 / 20).sum(1)
check(T, "sim H0 下 X² の平均≈5=k−1（20万回, 許容±0.03）", abs(Xs.mean() - 5) <= .03, Xs.mean())

# ================= homogeneity-test =================
T = "homogeneity-test"
tab = np.array([[12, 188], [20, 180], [28, 172]])
c2, pv, dof, E = stats.chi2_contingency(tab, correction=False)
check(T, "共通不良率 60/600=0.10", tab[:, 0].sum() / 600 == 0.1, 0.1)
check(T, "期待度数 各工場 20,180", np.allclose(E, [[20, 180]] * 3), E.tolist())
cb = ((tab - E) ** 2 / E).ravel()
check(T, "寄与 3.200,0.356,0,0,3.200,0.356", np.allclose(cb, [3.2, .356, 0, 0, 3.2, .356], atol=5e-4), cb.round(4).tolist())
check(T, "χ²=7.111", close(c2, 7.111, 5e-4), c2)
check(T, "自由度 (3−1)(2−1)=2", dof == 2, dof)
check(T, "χ²_0.05(2)=5.991", close(chi2.isf(.05, 2), 5.991, 5e-4), chi2.isf(.05, 2))
check(T, "p=0.0286", close(pv, 0.0286), pv)
rs, csum, N = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True), tab.sum()
adj = (tab - E) / np.sqrt(E * (1 - rs / N) * (1 - csum / N))
check(T, "調整済み標準化残差 工場1不良 −2.31、工場3 +2.31", close(adj[0, 0], -2.31, 5e-3) and close(adj[2, 0], 2.31, 5e-3), adj[:, 0].round(4).tolist())
check(T, "図 統計量 7.11", close(c2, 7.11, 5e-3), c2)
check(T, "ν=r(c−1)−(c−1)=(r−1)(c−1)", sp.expand(r_ * (c_ - 1) - (c_ - 1) - (r_ - 1) * (c_ - 1)) == 0, "sympy")
# 行和固定（3つの独立な二項）での棄却率≈0.05
Xb = rng.binomial(200, .1, (100000, 3))
tb = np.stack([Xb, 200 - Xb], 2)
Eb = tb.sum(1, keepdims=True) * tb.sum(2, keepdims=True) / 600
Sb = ((tb - Eb) ** 2 / Eb).sum((1, 2))
rb = (Sb >= chi2.isf(.05, 2)).mean()
check(T, "sim 行和固定・H0 で χ²(2) の5%棄却率≈0.05（10万回, 許容±0.004）", abs(rb - .05) <= .004, rb)

# ================= independence-test =================
T = "independence-test"
tab = np.array([[50, 30, 20], [30, 40, 30]])
c2, pv, dof, E = stats.chi2_contingency(tab, correction=False)
check(T, "例1 期待度数 40,35,25", np.allclose(E, [[40, 35, 25]] * 2), E.tolist())
check(T, "例1 2×(2.500+0.714+1.000)", np.allclose([100 / 40, 25 / 35, 25 / 25], [2.5, .714, 1], atol=5e-4), "")
check(T, "例1 χ²=8.429", close(c2, 8.429, 5e-4), c2)
check(T, "例1 自由度2、χ²_0.05(2)=5.991", dof == 2 and close(chi2.isf(.05, 2), 5.991, 5e-4), dof)
check(T, "例1 p=0.0148", close(pv, 0.0148), pv)
check(T, "図2 F(8.43)≈0.985、p≈0.015", close(chi2.cdf(8.43, 2), 0.985, 5e-4) and close(chi2.sf(8.43, 2), .015, 5e-4), chi2.cdf(8.43, 2))
check(T, "図1 統計量 8.43", close(c2, 8.43, 5e-3), c2)
a_, b_, c__, d_ = 30, 70, 25, 175
n = 300
check(T, "例2 ad−bc=3500", a_ * d_ - b_ * c__ == 3500, a_ * d_ - b_ * c__)
check(T, "例2 分母 100×200×55×245=269,500,000", 100 * 200 * 55 * 245 == 269500000, 100 * 200 * 55 * 245)
f22 = n * (a_ * d_ - b_ * c__) ** 2 / ((a_ + b_) * (c__ + d_) * (a_ + c__) * (b_ + d_))
c2b, pvb, _, _ = stats.chi2_contingency([[a_, b_], [c__, d_]], correction=False)
check(T, "例2 χ²=13.636（公式=chi2_contingency）", close(f22, 13.636, 5e-4) and close(c2b, f22, 1e-9), f22)
check(T, "例2 p=0.00022", close(pvb, 0.00022, 5e-6), pvb)
cy, pvy, _, _ = stats.chi2_contingency([[a_, b_], [c__, d_]], correction=True)
check(T, "例2 イェーツ 12.493", close(cy, 12.493, 5e-4), cy)
check(T, "例2 イェーツ p=0.00041", close(pvy, 0.00041, 5e-6), pvy)
# 2×2 公式の一般的成立（ランダム）
ok = True
for _ in range(200):
    a, b, c, dd = rng.integers(1, 100, 4)
    nn = a + b + c + dd
    ok &= close(stats.chi2_contingency([[a, b], [c, dd]], correction=False)[0], nn * (a * dd - b * c) ** 2 / ((a + b) * (c + dd) * (a + c) * (b + dd)), 1e-8)
check(T, "2×2 公式 n(ad−bc)²/… が一般に成立（ランダム200表）", ok, ok)
# 総数のみ固定（多項）で H0 下の棄却率
P = np.outer([.5, .5], [.4, .35, .25]).ravel()
Mm = rng.multinomial(200, P, 100000).reshape(-1, 2, 3)
rsum, csum2 = Mm.sum(2, keepdims=True), Mm.sum(1, keepdims=True)
Em = rsum * csum2 / 200
Sm = ((Mm - Em) ** 2 / Em).sum((1, 2))
rm = (Sm >= chi2.isf(.05, 2)).mean()
check(T, "sim 多項（総数固定）・独立で χ²(2) の5%棄却率≈0.05（10万回, 許容±0.004）", abs(rm - .05) <= .004, rm)

# ================= kolmogorov-smirnov-test =================
T = "kolmogorov-smirnov-test"
x = np.array([0.05, 0.12, 0.18, 0.21, 0.33, 0.36, 0.41, 0.45, 0.52, 0.70])
i = np.arange(1, 11)
dplus = i / 10 - x
dminus = x - (i - 1) / 10
check(T, "表 i/10−X_(i)", np.allclose(dplus, [.05, .08, .12, .19, .17, .24, .29, .35, .38, .30]), dplus.round(4).tolist())
check(T, "もう一方の最大 0.05", close(dminus.max(), 0.05, 1e-12), dminus.max())
D = max(dplus.max(), dminus.max())
check(T, "D10=0.38（i=9, x=0.52 の直後）", close(D, .38, 1e-12) and dplus.argmax() == 8, D)
r = stats.kstest(x, "uniform", method="exact")
check(T, "kstest 統計量=0.38", close(r.statistic, .38, 1e-12), r.statistic)
check(T, "正確 p=0.083", close(r.pvalue, 0.083, 5e-4), r.pvalue)
check(T, "n=10 正確な5%棄却限界 0.409", close(stats.kstwo.isf(.05, 10), 0.409, 5e-4), stats.kstwo.isf(.05, 10))
check(T, "1.358/√10=0.429", close(1.358 / sqrt(10), 0.429, 5e-4), 1.358 / sqrt(10))
pa = stats.kstwobign.sf(sqrt(10) * .38)
check(T, "漸近 p=0.111", close(pa, 0.111, 5e-4), pa)
check(T, "漸近の5%点 1.358、1%点 1.628", close(stats.kstwobign.isf(.05), 1.358, 5e-4) and close(stats.kstwobign.isf(.01), 1.628, 5e-4),
      (stats.kstwobign.isf(.05), stats.kstwobign.isf(.01)))
ser = lambda t: 1 - 2 * sum((-1) ** (j - 1) * exp(-2 * j * j * t * t) for j in range(1, 101))
check(T, "コルモゴロフ分布の級数 = kstwobign.cdf（t=0.5..2）", all(close(ser(t), stats.kstwobign.cdf(t), 1e-10) for t in np.linspace(.5, 2, 16)), "")
check(T, "図 params data が本文と一致", x.tolist() == [0.05, 0.12, 0.18, 0.21, 0.33, 0.36, 0.41, 0.45, 0.52, 0.70], "")
# 分布に依存しない：指数分布データの D_n と一様の D_n が同分布
Ne = 50000
Ue = np.sort(rng.random((Ne, 10)), 1)
Ee = np.sort(rng.exponential(2.0, (Ne, 10)), 1)


def Dn(Fx):
    return np.maximum((i / 10 - Fx).max(1), (Fx - (i - 1) / 10).max(1))


D_u, D_e = Dn(Ue), Dn(1 - np.exp(-Ee / 2.0))
ksu = stats.kstest(D_u, stats.kstwo(10).cdf).statistic
kse = stats.kstest(D_e, stats.kstwo(10).cdf).statistic
check(T, "sim 一様・指数（Exp(平均2)）のどちらでも D10 ~ kstwo(10)（各5万回, KS<0.0073=1%点）", ksu < .0073 and kse < .0073, (ksu, kse))
check(T, "sim D10 の5%棄却率≈0.05（kstwo の 0.409, 許容±0.005）", abs((D_e >= stats.kstwo.isf(.05, 10)).mean() - .05) <= .005, (D_e >= stats.kstwo.isf(.05, 10)).mean())
# 母数を推定すると保守的（正規, n=30）
Z = rng.normal(5, 2, (20000, 30))
Zs = np.sort((Z - Z.mean(1, keepdims=True)) / Z.std(1, ddof=1, keepdims=True), 1)
Fz = norm.cdf(Zs)
ii = np.arange(1, 31)
Dz = np.maximum((ii / 30 - Fz).max(1), (Fz - (ii - 1) / 30).max(1))
rz = (Dz >= stats.kstwo.isf(.05, 30)).mean()
check(T, "sim 平均・分散を推定した正規では棄却率が 0.05 より大幅に小さい（<0.01）", rz < .01, rz)

# ================= pearson-chi-square-statistic =================
T = "pearson-chi-square-statistic"
O = np.array([62, 48, 45, 40, 55])
cb = (O - 50) ** 2 / 50
check(T, "例1 寄与 2.88,0.08,0.50,2.00,0.50", np.allclose(cb, [2.88, .08, .5, 2, .5]), cb.tolist())
check(T, "例1 χ²=5.96", close(cb.sum(), 5.96, 1e-12), cb.sum())
check(T, "例1 ΣO²=12798、255.96−250", (O ** 2).sum() == 12798 and close((O ** 2).sum() / 50 - 250, 5.96, 1e-12), (O ** 2).sum())
check(T, "例1 χ²_0.05(4)=9.488", close(chi2.isf(.05, 4), 9.488, 5e-4), "")
check(T, "例1 p=0.202", close(stats.chisquare(O).pvalue, 0.202, 5e-4), stats.chisquare(O).pvalue)
check(T, "図1 月・木で 4.88", close(cb[0] + cb[3], 4.88, 1e-12), cb[0] + cb[3])
check(T, "図1 params observed 合計250", O.sum() == 250, O.sum())
check(T, "例2 期待度数 110, 90", close(200 * .55, 110, 1e-9) and close(200 * .45, 90, 1e-9), "")
check(T, "例2 14²/110=1.7818、14²/90=2.1778", close(196 / 110, 1.7818) and close(196 / 90, 2.1778), (196 / 110, 196 / 90))
c2 = 196 / 110 + 196 / 90
check(T, "例2 χ²=3.9596", close(c2, 3.9596), c2)
z = (124 - 110) / sqrt(200 * .55 * .45)
check(T, "例2 Z=1.990、Z²=χ²", close(z, 1.990, 5e-4) and close(z * z, c2, 1e-10), z)
check(T, "例2 p=0.0466（χ²(1)=両側 z）", close(chi2.sf(c2, 1), 0.0466) and close(2 * norm.sf(z), chi2.sf(c2, 1), 1e-12), chi2.sf(c2, 1))
Xs_, n_, pi_ = sp.symbols("X n pi", positive=True)
lhs = (Xs_ - n_ * pi_) ** 2 / (n_ * pi_) + (Xs_ - n_ * pi_) ** 2 / (n_ * (1 - pi_))
check(T, "k=2 の恒等式（sympy）", sp.simplify(lhs - (Xs_ - n_ * pi_) ** 2 / (n_ * pi_ * (1 - pi_))) == 0, "")
check(T, "χ²(ν) 平均=ν、分散=2ν（ν=1..30）", all(close(chi2.mean(v), v, 1e-9) and close(chi2.var(v), 2 * v, 1e-9) for v in range(1, 31)), "")
check(T, "(O−E) の和は0", (O - 50).sum() == 0, 0)
# 割合でなく度数：n を10倍で統計量10倍
check(T, "同じ割合で n を10倍 → 統計量10倍", close(stats.chisquare(O * 10).statistic, 59.6, 1e-9), stats.chisquare(O * 10).statistic)

# ================= poisson-parameter-test =================
T = "poisson-parameter-test"
pe = poisson.sf(40, 30)
check(T, "例1 正確 p=P(T≥41)=0.0323", close(pe, 0.0323), pe)
z = (41 - 30) / sqrt(30)
check(T, "例1 √30=5.477、Z=2.008", close(sqrt(30), 5.477, 5e-4) and close(z, 2.008, 5e-4), z)
check(T, "例1 正規近似 p=0.0223", close(norm.sf(z), 0.0223), norm.sf(z))
zc = (40.5 - 30) / sqrt(30)
check(T, "例1 連続修正 Z=1.917", close(zc, 1.917, 5e-4), zc)
check(T, "例1 連続修正 p=0.0276", close(norm.sf(zc), 0.0276), norm.sf(zc))
check(T, "例1 P(T≥40)=0.0463", close(poisson.sf(39, 30), 0.0463), poisson.sf(39, 30))
check(T, "例1 P(T≥39)=0.0648", close(poisson.sf(38, 30), 0.0648), poisson.sf(38, 30))
check(T, "例1 棄却域 {T≥40}", poisson.sf(39, 30) <= .05 < poisson.sf(38, 30), "")
check(T, "図1 F(40)≈0.968", close(poisson.cdf(40, 30), 0.968, 5e-4), poisson.cdf(40, 30))
s5 = sum(exp(-2) * 2 ** k / factorial(k) for k in range(6))
check(T, "例2 Σ_{k≤5}=0.9834", close(s5, 0.9834), s5)
check(T, "例2 P(X≥6)=0.0166", close(1 - s5, 0.0166) and close(poisson.sf(5, 2), 1 - s5, 1e-12), 1 - s5)
z2 = 4 / sqrt(2)
check(T, "例2 Z=2.828、p=0.0023", close(z2, 2.828, 5e-4) and close(norm.sf(z2), 0.0023), norm.sf(z2))
check(T, "例2 約7分の1", 6.5 <= (1 - s5) / norm.sf(z2) <= 7.5, (1 - s5) / norm.sf(z2))
check(T, "例2 P(X=0)=0.135、P(X=6)=0.012", close(poisson.pmf(0, 2), .135, 5e-4) and close(poisson.pmf(6, 2), .012, 5e-4), (poisson.pmf(0, 2), poisson.pmf(6, 2)))
pmf = poisson.pmf(np.arange(200), 2)
pb = pmf[pmf <= poisson.pmf(6, 2) * (1 + 1e-7)].sum()
check(T, "例2 小さい順に足す両側 p=0.0166", close(pb, 0.0166), pb)
check(T, "例2 2倍法 0.0331", close(2 * (1 - s5), 0.0331), 2 * (1 - s5))
ok = all(close(poisson.sf(t - 1, mu), chi2.cdf(2 * mu, 2 * t), 1e-12) for t in range(1, 60) for mu in (0.5, 2, 7.3, 30))
check(T, "P(T≥t)=P(χ²_{2t}≤2nλ0)", ok, ok)
Xp = rng.poisson(3, (200000, 10)).sum(1)
ksp = np.abs(np.bincount(Xp, minlength=80)[:80] / 200000 - poisson.pmf(np.arange(80), 30)).max()
check(T, "sim 再生性 ΣPo(3)×10 ~ Po(30)（20万回, 確率の最大差<0.002）", ksp < .002, ksp)

# ================= poisson-two-sample-test =================
T = "poisson-two-sample-test"
check(T, "例1 率 1.5, 0.7、率比 2.14", 18 / 12 == 1.5 and close(7 / 10, .7, 1e-12) and close(1.5 / .7, 2.14, 5e-3), 1.5 / .7)
p0 = 12 / 22
check(T, "例1 p0=12/22=0.5455、期待値 13.64", close(p0, .5455) and close(25 * p0, 13.64, 5e-3), (p0, 25 * p0))
pe = stats.binomtest(18, 25, p0, alternative="greater").pvalue
check(T, "例1 正確 p=P(X_A≥18)=0.0584", close(pe, 0.0584), pe)
z = (18 - 25 * p0) / sqrt(25 * p0 * (1 - p0))
check(T, "例1 Z=1.753", close(z, 1.753, 5e-4), z)
check(T, "例1 近似 p=0.0398", close(norm.sf(z), 0.0398), norm.sf(z))
check(T, "例1 正確は非有意・近似は有意", pe > .05 > norm.sf(z), "")
check(T, "図 F(17)≈0.942（params p=0.5455）", close(binom.cdf(17, 25, .5455), .942, 5e-4) and close(binom.cdf(17, 25, p0), .942, 5e-4), binom.cdf(17, 25, .5455))
check(T, "図 P(X≥18)≈0.058（params p=0.5455）", close(binom.sf(17, 25, .5455), .058, 5e-4), binom.sf(17, 25, .5455))
check(T, "例2 P(X≥15)=0.0207（Bin(20,.5)）", close(binom.sf(14, 20, .5), 0.0207), binom.sf(14, 20, .5))
check(T, "例2 両側 0.0414", close(2 * binom.sf(14, 20, .5), 0.0414) and close(stats.binomtest(15, 20, .5).pvalue, 2 * binom.sf(14, 20, .5), 1e-12), 2 * binom.sf(14, 20, .5))
k, m, mu1, mu2 = sp.symbols("k m mu1 mu2", positive=True)
lhs = (sp.exp(-mu1) * mu1 ** k / sp.factorial(k) * sp.exp(-mu2) * mu2 ** (m - k) / sp.factorial(m - k)) / (sp.exp(-(mu1 + mu2)) * (mu1 + mu2) ** m / sp.factorial(m))
rhs = sp.binomial(m, k) * (mu1 / (mu1 + mu2)) ** k * (mu2 / (mu1 + mu2)) ** (m - k)
ok = all(abs(float((lhs - rhs).subs({k: kk, m: mm, mu1: a, mu2: b}))) < 1e-12 for kk, mm in ((0, 5), (3, 7), (10, 25)) for a, b in ((1.5, 0.7), (18, 7)))
check(T, "条件付き分布の導出式（sympy 数値代入）", ok, ok)
lam, t1, t2 = sp.symbols("lambda t1 t2", positive=True)
check(T, "H0 で p0=λt1/(λt1+λt2)=t1/(t1+t2)", sp.simplify(lam * t1 / (lam * t1 + lam * t2) - t1 / (t1 + t2)) == 0, "")
# シミュレーション：λ=0.9, t=(12,10)、M=25 条件付き分布
A = rng.poisson(0.9 * 12, 400000)
B = rng.poisson(0.9 * 10, 400000)
sel = A[(A + B) == 25]
emp = np.bincount(sel, minlength=26)[:26] / len(sel)
err = np.abs(emp - binom.pmf(np.arange(26), 25, p0)).max()
check(T, f"sim M=25 で条件付けた X_A ~ Bin(25,12/22)（{len(sel)}件, 確率の最大差<0.01）", err < .01, err)

# ================= proportion-test =================
T = "proportion-test"
se = sqrt(.55 * .45 / 200)
check(T, "例1 SE=0.035178", close(se, 0.035178, 5e-7), se)
z = (.62 - .55) / se
check(T, "例1 Z=1.990", close(z, 1.990, 5e-4), z)
check(T, "例1 z_0.05=1.645", close(norm.isf(.05), 1.645, 5e-4), norm.isf(.05))
check(T, "例1 p=0.0233", close(norm.sf(z), 0.0233), norm.sf(z))
zc = (124 - .5 - 110) / sqrt(49.5)
check(T, "例1 np0(1−p0)=49.5、Zc=1.919", close(200 * .55 * .45, 49.5, 1e-9) and close(zc, 1.919, 5e-4), zc)
check(T, "例1 連続修正 p=0.0275", close(norm.sf(zc), 0.0275), norm.sf(zc))
pe = binom.sf(123, 200, .55)
check(T, "例1 正確 P(X≥124)=0.0270", close(pe, 0.0270), pe)
check(T, "図2 F(1.99)≈0.977", close(norm.cdf(1.99), .977, 5e-4), norm.cdf(1.99))
se2 = sqrt(.05 * .95 / 400)
z2 = -.02 / se2
check(T, "例2 SE=0.010897", close(se2, 0.010897, 5e-7), se2)
check(T, "例2 Z=−1.835", close(z2, -1.835, 5e-4), z2)
check(T, "例2 p=0.0332", close(norm.cdf(z2), 0.0332), norm.cdf(z2))
check(T, "例2 正確 P(X≤12)=0.0355", close(binom.cdf(12, 400, .05), 0.0355), binom.cdf(12, 400, .05))
check(T, "n=200, p0=0.01 → np0=2", 200 * .01 == 2, 2)
X_, n_, p_ = sp.symbols("X n p", positive=True)
lhs = (X_ - n_ * p_) ** 2 / (n_ * p_) + ((n_ - X_) - n_ * (1 - p_)) ** 2 / (n_ * (1 - p_))
check(T, "Z² = 2カテゴリのカイ二乗（sympy）", sp.simplify(lhs - (X_ - n_ * p_) ** 2 / (n_ * p_ * (1 - p_))) == 0, "")
Xb = rng.binomial(200, .55, 200000)
Zs = (Xb - 110) / sqrt(49.5)
rr = (Zs >= norm.isf(.05)).mean()
check(T, "sim n=200,p0=.55 の右片側5% z 検定の棄却率≈0.05（正確値 " + f"{binom.sf(np.ceil(110 + 1.6449 * sqrt(49.5)) - 1, 200, .55):.4f}" + ", 許容±0.01）", abs(rr - .05) <= .01, rr)

# ================= two-sample-proportion-test =================
T = "two-sample-proportion-test"
se = sqrt(.27 * .73 * (2 / 500))
z = .06 / se
check(T, "例1 pooled=0.27、SE=0.028078", close(se, 0.028078, 5e-7), se)
check(T, "例1 Z=2.137", close(z, 2.137, 5e-4), z)
check(T, "例1 両側 p=0.0326", close(2 * norm.sf(z), 0.0326), 2 * norm.sf(z))
tab = np.array([[150, 350], [120, 380]])
c2, pv, _, _ = stats.chi2_contingency(tab, correction=False)
check(T, "例1 χ²=4.566=Z²", close(c2, 4.566, 5e-4) and close(c2, z * z, 1e-9), c2)
check(T, "例1 χ² の p=0.0326", close(pv, 0.0326), pv)
seu = sqrt(.3 * .7 / 500 + .24 * .76 / 500)
check(T, "例1 プールしない SE=0.028014", close(seu, 0.028014, 5e-7), seu)
lo, hi = .06 - norm.isf(.025) * seu, .06 + norm.isf(.025) * seu
check(T, "例1 95%CI [0.0051, 0.1149]", close(lo, .0051) and close(hi, .1149), (lo, hi))
check(T, "図1 F(2.14)≈0.984、右片側≈0.016、両側≈0.032", close(norm.cdf(2.14), .984, 5e-4) and close(norm.sf(2.14), .016, 5e-4) and close(2 * norm.sf(2.14), .032, 5e-4), norm.cdf(2.14))
check(T, "図2 モザイクの統計量 4.566", close(c2, 4.566, 5e-4), c2)
se2 = sqrt(.33 * .67 * (1 / 80 + 1 / 120))
z2 = .2 / se2
check(T, "例2 36/80=.45, 30/120=.25, 66/200=.33", 36 / 80 == .45 and 30 / 120 == .25 and 66 / 200 == .33, "")
check(T, "例2 SE=0.067869", close(se2, 0.067869, 5e-7), se2)
check(T, "例2 Z=2.947", close(z2, 2.947, 5e-4), z2)
check(T, "例2 両側 p=0.0032", close(2 * norm.sf(z2), 0.0032), 2 * norm.sf(z2))
check(T, "例2 (0.45+0.25)/2=0.35≠0.33", close((.45 + .25) / 2, .35, 1e-12), "")
# H0 下の棄却率
X1 = rng.binomial(500, .27, 200000)
X2 = rng.binomial(500, .27, 200000)
ph = (X1 + X2) / 1000
Zs = (X1 / 500 - X2 / 500) / np.sqrt(ph * (1 - ph) * (2 / 500))
rr = (np.abs(Zs) >= 1.959964).mean()
check(T, "sim H0 下の両側5%棄却率≈0.05（20万回, 許容±0.003）", abs(rr - .05) <= .003, rr)



# ######################################################################
# ================= 重要度Cの用語 =================
# 主張値は本文の表示桁で丸めて一致するか（cc：表示桁の 0.5 単位 + extra）。
# 乱数は専用の Generator（seed=20261023+章番号）を使い、許容誤差を各チェックに明記。
def _dec(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s else 0


def cc(term, what, x, s, extra=0.0):
    tol = 0.5 * 10 ** (-_dec(s)) * (1 + 1e-9) + extra
    check(term, f"{what} = {s}", abs(float(x) - float(s)) <= tol, round(float(x), _dec(s) + 3))


def mcC(term, what, est, target, tol):
    check(term, f"{what}（シミュ、許容±{tol}）", abs(float(est) - target) <= tol, round(float(est), 5))

rngC = np.random.default_rng(20261023 + 12)

# ================= variance-stabilizing-transform =================
T = "variance-stabilizing-transform"
pp, lam, rr = sp.symbols("p lambda r", positive=True)
g = sp.asin(sp.sqrt(pp))
check(T, "d/dp arcsin√p = 1/(2√(p(1-p)))", sp.simplify(sp.diff(g, pp) - 1 / (2 * sp.sqrt(pp * (1 - pp)))) == 0, "")
check(T, "{g'(p)}²·p(1-p)/n = 1/(4n)", sp.simplify(sp.diff(g, pp)**2 * pp * (1 - pp) - sp.Rational(1, 4)) == 0, "")
check(T, "ポアソン {(√λ)'}²·λ = 1/4", sp.simplify(sp.diff(sp.sqrt(lam), lam)**2 * lam - sp.Rational(1, 4)) == 0, "")
check(T, "相関 {(atanh r)'}²·(1-r²)² = 1", sp.simplify(sp.diff(sp.atanh(rr), rr)**2 * (1 - rr**2)**2 - 1) == 0, "")
check(T, "2arcsin√p なら分散 1/n", sp.simplify(sp.diff(2 * g, pp)**2 * pp * (1 - pp) - 1) == 0, "")
for p, sv in [(0.1, "0.0018"), (0.3, "0.0042"), (0.5, "0.0050")]:
    cc(T, f"p(1-p)/50（p={p}）", p * (1 - p) / 50, sv)
cc(T, "比 0.0050/0.0018", 0.25 / 0.09, "2.8", 0.0)
k = np.arange(51)
for p, sv in [(0.1, 0.0057), (0.3, 0.0051), (0.5, 0.0051)]:
    w = stats.binom.pmf(k, 50, p); a = np.arcsin(np.sqrt(k / 50))
    ex = (w * a**2).sum() - (w * a).sum()**2
    cc(T, f"V[arcsin√p̂] 正確計算（p={p}）", ex, f"{sv:.4f}")
    Bs = rngC.binomial(50, p, 200000)
    # 推定分散の標準誤差は約 2e-5 → 許容 8e-5（表示桁の丸め 5e-5 + 1.5SE）
    mcC(T, f"V[arcsin√p̂] 20万回（p={p}）≈{sv}", np.arcsin(np.sqrt(Bs / 50)).var(), sv, 8e-5)
cc(T, "1/(4·50)", 1 / 200, "0.005")
a1, a2 = np.arcsin(sqrt(0.30)), np.arcsin(sqrt(0.24))
cc(T, "差 arcsin√0.30-arcsin√0.24", a1 - a2, "0.06767")
cc(T, "分母 √(2/(4·500))", sqrt(2 / 2000), "0.03162")
Zv = (a1 - a2) / sqrt(2 / 2000)
cc(T, "Z", Zv, "2.140")
cc(T, "両側 p値", 2 * stats.norm.sf(Zv), "0.032")
ph = 270 / 1000
cc(T, "通常の2標本比率 z", (0.30 - 0.24) / sqrt(ph * (1 - ph) * (2 / 500)), "2.137")

# ================= dispersion-test =================
T = "dispersion-test"
d = np.array([1, 7, 2, 10, 3, 0, 8, 12, 2, 4, 9, 2])
cc(T, "x̄", d.mean(), "5.0")
cc(T, "偏差平方和", ((d - d.mean())**2).sum(), "176")
cc(T, "s²", d.var(ddof=1), "16.0")
cc(T, "分散指数", d.var(ddof=1) / d.mean(), "3.2")
Dv = ((d - d.mean())**2).sum() / d.mean()
cc(T, "D", Dv, "35.2")
check(T, "D = (n-1)s²/x̄", abs(Dv - 11 * d.var(ddof=1) / d.mean()) < 1e-12, "")
check(T, "D = ピアソンの χ²（期待度数 x̄）", abs(Dv - (((d - d.mean())**2) / d.mean()).sum()) < 1e-12, "")
cc(T, "χ²_0.05(11)", stats.chi2.isf(0.05, 11), "19.675")
cc(T, "p値", stats.chi2.sf(Dv, 11), "0.0002")
P = rngC.poisson(5.0, (200000, 12))
Ds = ((P - P.mean(1, keepdims=True))**2).sum(1) / P.mean(1)
mcC(T, "H0 下 P(D>19.675)≈0.05（λ=5, n=12）", (Ds > 19.675).mean(), 0.05, 0.004)
mcC(T, "H0 下 E[D]≈n-1=11", Ds.mean(), 11, 0.05)

# --- 統合候補からの論点の追記: kolmogorov-smirnov-test ---
check("kolmogorov-smirnov-test", "2標本の5%棄却限界 1.358√((m+n)/(mn))：コルモゴロフ分布の95%点 1.358、m=n=8 で 0.679",
      abs(stats.kstwobign.ppf(0.95) - 1.358) < 5e-4 and abs(1.358 * sqrt(16 / 64) - 0.679) < 1e-12, stats.kstwobign.ppf(0.95))

# ================= 集計 =================
nok, nng = sum(results), len(results) - sum(results)
print(f"\n合計: OK {nok} / NG {nng}")
