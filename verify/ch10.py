"""第10章 検算スクリプト（仮説検定の基礎・検定法の導出・多重検定）。
記号計算は sympy（NP 補題の対数尤度比、尤度比・ワルド・スコア統計量の導出、標本サイズ公式）、
数値は scipy.stats、主張値との比較は表示桁の丸め幅（4桁なら 5e-5、3桁なら 5e-4）に 1e-12 を加えたもの。
z 値は本文の丸め値（1.645 など）ではなく正確な分位点で計算し、丸め値で書かれた式は両方を確認する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の数倍）。
"""
from math import comb, exp, sqrt, log
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-12


def zero(expr):
    return sp.simplify(expr) == 0


def series_ok(xs, ys, f, tol):
    err = max(abs(f(x) - y) for x, y in zip(xs, ys))
    return err <= tol + 1e-12, err


rng = np.random.default_rng(20261023)
N = stats.norm
Phi = N.cdf
z = lambda a: N.isf(a)
binom, pois, chi2 = stats.binom, stats.poisson, stats.chi2

# 共通の記号
xb, mu0, mu1, mu, s, n = sp.symbols("xbar mu0 mu1 mu sigma n", positive=True)
xs_ = sp.symbols("x1:6", real=True)


def bh(p, q):
    ps = np.sort(p)
    m = len(ps)
    ok = np.nonzero(ps <= np.arange(1, m + 1) / m * q)[0]
    return 0 if len(ok) == 0 else ok.max() + 1


def holm(p, a):
    ps = np.sort(p)
    m = len(ps)
    k = 0
    while k < m and ps[k] <= a / (m - k):
        k += 1
    return k


p5 = np.array([0.004, 0.011, 0.019, 0.030, 0.250])

# ================= neyman-pearson-lemma =================
T = "neyman-pearson-lemma"
xi = sp.symbols("x1:4", real=True)
nn = 3
ll = lambda m_: -sum((x_ - m_)**2 for x_ in xi) / (2 * s**2)
xbar_ = sum(xi) / nn
lhs = ll(mu1) - ll(mu0)
rhs = nn * (mu1 - mu0) / s**2 * xbar_ - nn * (mu1**2 - mu0**2) / (2 * s**2)
check(T, "例1 正規の logΛ = n(μ1-μ0)x̄/σ² - n(μ1²-μ0²)/(2σ²)（n=3 で記号確認）", zero(lhs - rhs), "")
c = z(0.05) / 3
check(T, "例1 c=1.645/3≈0.5483", close(c, 0.5483) and close(1.645 / 3, 0.5483), c)
pw = 1 - Phi((c - 1) * 3)
check(T, "例1 検出力 Φ(1.355)≈0.9123", close(pw, 0.9123) and close(Phi(1.355), 0.9123, 1e-4), pw)
Tt, p0, p1, nb = sp.symbols("T p0 p1 n", positive=True)
lb = Tt * sp.log(p1 / p0) + (nb - Tt) * sp.log((1 - p1) / (1 - p0))
rb = Tt * sp.log(p1 * (1 - p0) / (p0 * (1 - p1))) + nb * sp.log((1 - p1) / (1 - p0))
check(T, "例2 ベルヌーイ logΛ の整理", zero(sp.expand_log(lb - rb, force=True)), "")
check(T, "例2 P0.5(T≥9)=0.0107, P0.5(T≥8)=0.0547", close(binom.sf(8, 10, .5), 0.0107) and close(binom.sf(7, 10, .5), 0.0547), (binom.sf(8, 10, .5), binom.sf(7, 10, .5)))
check(T, "例2 P0.8(T≥9)≈0.3758", close(binom.sf(8, 10, .8), 0.3758), binom.sf(8, 10, .8))
g = (0.05 - binom.sf(8, 10, .5)) / binom.pmf(8, 10, .5)
check(T, "例2 γ≈0.8933", close(g, 0.8933), g)
check(T, "例2 無作為化検定のサイズ 0.05", close(binom.sf(8, 10, .5) + g * binom.pmf(8, 10, .5), 0.05, 1e-12), "")
pwr = binom.sf(8, 10, .8) + g * binom.pmf(8, 10, .8)
check(T, "例2 無作為化検定の検出力≈0.6456", close(pwr, 0.6456), pwr)
check(T, "例3 logΛ=0.4055T-5（log1.5）", close(log(1.5), 0.4055), log(1.5))
check(T, "例3 Po(10): P(T≥15)=0.0835, P(T≥16)=0.0487", close(pois.sf(14, 10), 0.0835) and close(pois.sf(15, 10), 0.0487), (pois.sf(14, 10), pois.sf(15, 10)))
check(T, "例3 検出力 P(Po(15)≥16)≈0.4319", close(pois.sf(15, 15), 0.4319), pois.sf(15, 15))
l0, l1, S_ = sp.symbols("lambda0 lambda1 S", positive=True)
le = (10 * sp.log(l1) - l1 * S_) - (10 * sp.log(l0) - l0 * S_)
check(T, "例4 指数の logΛ = n log(λ1/λ0) - (λ1-λ0)Σx", zero(sp.expand_log(le - (10 * sp.log(l1 / l0) - (l1 - l0) * S_), force=True)), "")
c4 = chi2.isf(0.05, 20) / 2
check(T, "例4 χ²0.05(20)=31.410, c≈15.705", close(chi2.isf(0.05, 20), 31.410, 5e-4) and close(c4, 15.705, 5e-4), c4)
check(T, "例4 検出力 P(χ²(20)>15.705)≈0.7347", close(chi2.sf(c4, 20), 0.7347), chi2.sf(c4, 20))
Ssim = rng.exponential(2.0, (200000, 10)).sum(axis=1)  # λ=0.5
check(T, "例4 シミュ: λ=0.5 で P(ΣX>15.705)（許容0.004）", abs((Ssim > c4).mean() - 0.7347) < 0.004, (Ssim > c4).mean())

# ================= likelihood-ratio-test =================
T = "likelihood-ratio-test"
ll60 = lambda p: 60 * log(p) + 40 * log(1 - p)
xs = [0.4 + 0.02 * i for i in range(21)]
ys = [-8.109, -6.538, -5.15, -3.938, -2.894, -2.014, -1.293, -0.731, -0.327, -0.082, 0, -0.084, -0.342, -0.782, -1.416, -2.258, -3.328, -4.648, -6.25, -8.172, -10.465]
ok, err = series_ok(xs, ys, lambda p: ll60(p) - ll60(0.6), 5e-4)
check(T, "図 ℓ(p)-ℓ(0.6) の21点（許容5e-4）", ok, err)
check(T, "図 境目 -χ²0.05(1)/2=-1.921", close(-chi2.isf(.05, 1) / 2, -1.921, 5e-4), -chi2.isf(.05, 1) / 2)
check(T, "グラフ説明 p0=0.5 の落差2.014>1.921, p0=0.52 の落差1.293<1.921", close(ll60(.6) - ll60(.5), 2.014, 5e-4) and close(ll60(.6) - ll60(.52), 1.293, 5e-4), "")
x5 = sp.symbols("y1:6", real=True)
xb5 = sum(x5) / 5
dec = sum((x_ - mu0)**2 for x_ in x5) - (sum((x_ - xb5)**2 for x_ in x5) + 5 * (xb5 - mu0)**2)
check(T, "例1 Σ(x-μ0)²=Σ(x-x̄)²+n(x̄-μ0)²（n=5）", zero(sp.expand(dec)), "")
sh, d_, nn_ = sp.symbols("sigmahat d n", positive=True)
tt = sp.sqrt(nn_) * d_ / sp.sqrt(nn_ * sh**2 / (nn_ - 1))
check(T, "例2 σ̂0²/σ̂²=1+t²/(n-1)", zero((sh**2 + d_**2) / sh**2 - (1 + tt**2 / (nn_ - 1))), "")
lr2 = 20 * log(1 + 4.84 / 19)
check(T, "例2 -2logΛ=20log(1+4.84/19)≈4.539", close(lr2, 4.539, 5e-4), lr2)
check(T, "例2 χ²近似 p≈0.0331, t 検定 p≈0.0404", close(chi2.sf(lr2, 1), 0.0331) and close(2 * stats.t.sf(2.2, 19), 0.0404), (chi2.sf(lr2, 1), 2 * stats.t.sf(2.2, 19)))
check(T, "例3 60log1.2=10.939, 40log0.8=-8.926", close(60 * log(1.2), 10.939, 5e-4) and close(40 * log(0.8), -8.926, 5e-4), "")
lr3 = 2 * (60 * log(1.2) + 40 * log(0.8))
check(T, "例3 -2logΛ≈4.027, p≈0.0448", close(lr3, 4.027, 5e-4) and close(chi2.sf(lr3, 1), 0.0448), (lr3, chi2.sf(lr3, 1)))
lr4 = 2 * (30 * log(1.5) - 10)
check(T, "例4 ポアソン -2logΛ≈4.328, p≈0.0375", close(lr4, 4.328, 5e-4) and close(chi2.sf(lr4, 1), 0.0375), (lr4, chi2.sf(lr4, 1)))
lam_, xbar, nn2 = sp.symbols("lambda xbar n", positive=True)
lexp = lambda l: nn2 * sp.log(l) - l * nn2 * xbar
expr = 2 * (lexp(1 / xbar) - lexp(l0)) - 2 * nn2 * (l0 * xbar - 1 - sp.log(l0 * xbar))
check(T, "例4 指数 -2logΛ=2n{λ0x̄-1-log(λ0x̄)}（記号）", zero(sp.expand_log(expr, force=True)), "")
lr5 = 40 * (0.5 - log(1.5))
check(T, "例4 指数 40{0.5-0.4055}≈3.781, p≈0.0518", close(lr5, 3.781, 5e-4) and close(chi2.sf(lr5, 1), 0.0518), (lr5, chi2.sf(lr5, 1)))

# ================= wilks-theorem =================
T = "wilks-theorem"
check(T, "χ²(1) 5%=3.841, 1%=6.635, χ²(2) 5%=5.991, χ²(5) 5%=11.070",
      close(chi2.isf(.05, 1), 3.841, 5e-4) and close(chi2.isf(.01, 1), 6.635, 5e-4) and close(chi2.isf(.05, 2), 5.991, 5e-4) and close(chi2.isf(.05, 5), 11.070, 5e-4), "")
O = np.array([5, 8, 9, 8, 10, 20])
G = 2 * (O * np.log(O / 10)).sum()
check(T, "例1 G=11.757, p≈0.0383", close(G, 11.757, 5e-4) and close(chi2.sf(G, 5), 0.0383), (G, chi2.sf(G, 5)))
X2 = ((O - 10)**2 / 10).sum()
check(T, "例1 ピアソン 13.4, p≈0.0199", close(X2, 13.4, 1e-9) and close(chi2.sf(X2, 5), 0.0199), (X2, chi2.sf(X2, 5)))
G2 = 2 * (30 * log(30 / 24) + 18 * log(18 / 24))
check(T, "例2 -2logΛ≈3.032, p≈0.0816", close(G2, 3.032, 5e-4) and close(chi2.sf(G2, 1), 0.0816), (G2, chi2.sf(G2, 1)))
xx = np.arange(101)
with np.errstate(divide="ignore", invalid="ignore"):
    ph = xx / 100
    Gb = 2 * (np.where(xx > 0, xx * np.log(ph / .5), 0) + np.where(xx < 100, (100 - xx) * np.log((1 - ph) / .5), 0))
exact = binom.pmf(xx, 100, .5)[Gb > chi2.isf(.05, 1)].sum()
check(T, "例3 n=100,p=0.5 の棄却率の正確値≈0.057", close(exact, 0.057, 5e-4), exact)
sim = rng.binomial(100, .5, 200000)
check(T, "例3 シミュ20万回の棄却率≈0.057（許容0.0015）", abs(np.isin(sim, xx[Gb > chi2.isf(.05, 1)]).mean() - 0.057) < 0.0015, np.isin(sim, xx[Gb > chi2.isf(.05, 1)]).mean())
check(T, "境界の帰無仮説: 0.5χ²(0)+0.5χ²(1) で 3.841 超の確率 0.025<0.05（保守的）", close(0.5 * chi2.sf(chi2.isf(.05, 1), 1), 0.025, 1e-9), 0.5 * chi2.sf(chi2.isf(.05, 1), 1))
# 指数分布 n=50 で -2logΛ が χ²(1) に近いこと（傍証）
Xe = rng.exponential(1.0, (200000, 50)).mean(axis=1)
Le = 2 * 50 * (Xe - 1 - np.log(Xe))
check(T, "シミュ 指数 n=50 の -2logΛ>3.841 の割合≈0.05（許容0.004）", abs((Le > chi2.isf(.05, 1)).mean() - 0.05) < 0.004, (Le > chi2.isf(.05, 1)).mean())

# ================= wald-test =================
T = "wald-test"
Tg = np.arange(10, 36)
lh = Tg / 10
Wv = 10 * (lh - 2)**2 / lh
LRv = 2 * (Tg * np.log(Tg / 20) - (Tg - 20))
Sv = (Tg - 20)**2 / 20
yW = [10, 7.364, 5.333, 3.769, 2.571, 1.667, 1, 0.529, 0.222, 0.053, 0, 0.048, 0.182, 0.391, 0.667, 1, 1.385, 1.815, 2.286, 2.793, 3.333, 3.903, 4.5, 5.121, 5.765, 6.429]
yL = [6.137, 4.848, 3.74, 2.8, 2.013, 1.37, 0.859, 0.474, 0.207, 0.051, 0, 0.049, 0.194, 0.429, 0.751, 1.157, 1.643, 2.206, 2.842, 3.551, 4.328, 5.172, 6.08, 7.051, 8.083, 9.173]
yS = [5, 4.05, 3.2, 2.45, 1.8, 1.25, 0.8, 0.45, 0.2, 0.05, 0, 0.05, 0.2, 0.45, 0.8, 1.25, 1.8, 2.45, 3.2, 4.05, 5, 6.05, 7.2, 8.45, 9.8, 11.25]
for lab, v_, y_ in [("W=n(λ̂-2)²/λ̂", Wv, yW), ("-2logΛ", LRv, yL), ("S=(T-20)²/20", Sv, yS)]:
    e = np.abs(v_ - np.array(y_)).max()
    check(T, f"図 {lab} の26点（許容5e-4）", e <= 5e-4 + 1e-12, e)
hi, lo = Tg > 20, Tg < 20
check(T, "図の説明 T>20 で W<LR<S, T<20 で W>LR>S", np.all((Wv < LRv)[hi] & (LRv < Sv)[hi]) and np.all((Wv > LRv)[lo] & (LRv > Sv)[lo]), "")
# 表 1/{nI(θ̂)}
th = sp.Symbol("theta", positive=True)
xv = sp.Symbol("x", real=True)
tab = {"Bern": (xv * sp.log(th) + (1 - xv) * sp.log(1 - th), {1: th, 2: th}, th * (1 - th)),
       "Po": (xv * sp.log(th) - th, {1: th, 2: th + th**2}, th),
       "Ex": (sp.log(th) - th * xv, {1: 1 / th, 2: 2 / th**2}, th**2),
       "N(σ=1 換算 σ²)": (-(xv - th)**2 / (2 * s**2), {1: th, 2: th**2 + s**2}, s**2)}
for name, (lf, moms, inv) in tab.items():
    d2 = sp.expand(-sp.diff(lf, th, 2))
    I = sp.simplify(d2.subs(xv**2, moms[2]).subs(xv, moms[1]))
    check(T, f"表 {name}: 1/I(θ)={inv}", zero(1 / I - inv), sp.simplify(1 / I))
W1 = 0.01 / 0.0024
check(T, "例1 W≈4.167, Z_W≈2.041, p≈0.0412", close(W1, 4.167, 5e-4) and close(sqrt(W1), 2.041, 5e-4) and close(chi2.sf(W1, 1), 0.0412), (W1, sqrt(W1), chi2.sf(W1, 1)))
ci = (0.6 - 1.96 * sqrt(.0024), 0.6 + 1.96 * sqrt(.0024))
check(T, "例1 ワルド区間 (0.504, 0.696)", close(ci[0], 0.504, 5e-4) and close(ci[1], 0.696, 5e-4), ci)
check(T, "例2 W≈3.333, p≈0.0679", close(1 / 0.3, 3.333, 5e-4) and close(chi2.sf(1 / .3, 1), 0.0679), chi2.sf(1 / .3, 1))
check(T, "例2 尤度比4.328, スコア5.000（3.841 をはさむ）", close(lr4, 4.328, 5e-4) and (1 / .3 < 3.841 < lr4), "")
lt = sp.Symbol("lambda", positive=True)
dm = sp.simplify((lt / 10) * sp.diff(sp.log(lt), lt)**2)
check(T, "例3 デルタ法 V[logλ̂]≈(λ/n)/λ²=1/(nλ)", zero(dm - 1 / (10 * lt)), dm)
Wpsi = 30 * log(1.5)**2
check(T, "例3 W_ψ=30×0.4055²≈4.932, p≈0.0264", close(Wpsi, 4.932, 5e-4) and close(chi2.sf(Wpsi, 1), 0.0264), (Wpsi, chi2.sf(Wpsi, 1)))

# ================= score-test =================
T = "score-test"
xs = [1 + 0.25 * i for i in range(17)]
ys = [20, 14, 10, 7.143, 5, 3.333, 2, 0.909, 0, -0.769, -1.429, -2, -2.5, -2.941, -3.333, -3.684, -4]
ok, err = series_ok(xs, ys, lambda l: 30 / l - 10, 5e-4)
check(T, "図 U(λ)=30/λ-10 の17点（許容5e-4）", ok, err)
xk, nk, pk = sp.symbols("x n p0", positive=True)
U = sp.diff(xk * sp.log(pk) + (nk - xk) * sp.log(1 - pk), pk)
check(T, "ベルヌーイ U(p0)=(x-np0)/{p0(1-p0)}", zero(U - (xk - nk * pk) / (pk * (1 - pk))), "")
check(T, "ベルヌーイ S=U²/(nI)=(x-np0)²/{np0(1-p0)}", zero(U**2 / (nk / (pk * (1 - pk))) - (xk - nk * pk)**2 / (nk * pk * (1 - pk))), "")
check(T, "ベルヌーイ Z_S=(p̂-p0)/√{p0(1-p0)/n}（両辺の2乗と符号で確認）", zero(U**2 / (nk / (pk * (1 - pk))) - (xk / nk - pk)**2 / (pk * (1 - pk) / nk)) and zero(sp.factor(U * pk * (1 - pk)) - nk * (xk / nk - pk)), "")
Tk = sp.Symbol("T", positive=True)
Up = sp.diff(Tk * sp.log(l0) - nk * l0, l0)
check(T, "ポアソン U=T/λ0-n, S=(T-nλ0)²/(nλ0)", zero(Up - (Tk / l0 - nk)) and zero(Up**2 / (nk / l0) - (Tk - nk * l0)**2 / (nk * l0)), "")
Ue = sp.diff(nk * sp.log(l0) - l0 * nk * xbar, l0)
check(T, "指数 U=n/λ0-Σx, nI=n/λ0², S=n(λ0x̄-1)²", zero(Ue - (nk / l0 - nk * xbar)) and zero(Ue**2 / (nk / l0**2) - nk * (l0 * xbar - 1)**2), "")
# E[U]=0, V[U]=nI（ポアソン、1個あたり）
Ux = xv / th - 1
check(T, "ポアソン E[U]=0, V[U]=I=1/λ（1個）", zero(Ux.subs(xv, th)) and zero(sp.expand(Ux**2).subs(xv**2, th + th**2).subs(xv, th) - 1 / th), "")
# 変換不変性：ψ=logλ
psi = sp.Symbol("psi", real=True)
lpsi = Tk * psi - nk * sp.exp(psi)
Ups = sp.diff(lpsi, psi)
Ips = sp.simplify(-sp.diff(lpsi, psi, 2).subs(Tk, nk * sp.exp(psi)))  # E[T]=nλ
Sps = sp.simplify((Ups**2 / Ips).subs(psi, sp.log(l0)))
check(T, "スコア統計量は ψ=logλ に変換しても同じ（記号）", zero(Sps - (Tk - nk * l0)**2 / (nk * l0)), Sps)
check(T, "例1 S=4.000, Z=2, p≈0.0455", close(100 / 25, 4.0, 1e-12) and close(chi2.sf(4, 1), 0.0455), chi2.sf(4, 1))
check(T, "例2 U(2)=5, nI=5, S=5, p≈0.0253", (30 / 2 - 10 == 5) and (10 / 2 == 5) and close(chi2.sf(5, 1), 0.0253), chi2.sf(5, 1))
check(T, "例3 U(1)=-10, nI=20, S=5.000", (20 - 20 * 1.5 == -10) and close(100 / 20, 5, 1e-12), "")
check(T, "例4 S=4/1.8≈2.222, p≈0.1360", close(4 / 1.8, 2.222, 5e-4) and close(chi2.sf(4 / 1.8, 1), 0.1360), chi2.sf(4 / 1.8, 1))
# スコア統計量のサイズ（ポアソン n=10, λ=2）の傍証
Ts = rng.poisson(20, 400000)
check(T, "シミュ ポアソン U(λ0)/√(nI) の平均≈0, 分散≈1（許容0.01）", abs(((Ts / 2 - 10) / sqrt(5)).mean()) < 0.01 and abs(((Ts / 2 - 10) / sqrt(5)).var() - 1) < 0.01, ((Ts / 2 - 10) / sqrt(5)).var())

# ================= monotone-likelihood-ratio =================
T = "monotone-likelihood-ratio"
xs = [-4 + 0.5 * i for i in range(21)]
yN = [-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
yC = [-0.425, -0.472, -0.531, -0.603, -0.693, -0.802, -0.916, -0.956, -0.693, 0, 0.693, 0.956, 0.916, 0.802, 0.693, 0.603, 0.531, 0.472, 0.425, 0.386, 0.353]
ok, err = series_ok(xs, yN, lambda x: N.logpdf(x, 1) - N.logpdf(x, 0), 1e-9)
check(T, "図 正規 log比 = x-0.5 の21点", ok, err)
ok, err = series_ok(xs, yC, lambda x: stats.cauchy.logpdf(x, 1) - stats.cauchy.logpdf(x, 0), 5e-4)
check(T, "図 コーシー log比 の21点（許容5e-4）", ok, err)
Lt = [1.4**t * 0.6**(5 - t) for t in range(6)]
check(T, "例1 Λ(t) の表 6値", all(close(a, b) for a, b in zip(Lt, [0.0778, 0.1814, 0.4234, 0.9878, 2.3050, 5.3782])), [round(v, 5) for v in Lt])
check(T, "例1 Λ(t)=binom比, 隣どうしの比 7/3≈2.333", all(close(Lt[t], binom.pmf(t, 5, .7) / binom.pmf(t, 5, .5), 1e-12) for t in range(6)) and close(1.4 / .6, 2.333, 5e-4), "")
xc = sp.Symbol("x", real=True)
Lc = (1 + xc**2) / (1 + (xc - 1)**2)
crit = sp.solve(sp.diff(Lc, xc), xc)
xm = max(crit, key=lambda r: float(r))
check(T, "例2 コーシー Λ の最大点 (1+√5)/2≈1.618", zero(xm - (1 + sp.sqrt(5)) / 2) and close(float(xm), 1.618, 5e-4), crit)
check(T, "例2 最大値≈2.618, Λ(3)=2.000, Λ(10)≈1.232, 極限1", close(float(Lc.subs(xc, xm)), 2.618, 5e-4) and Lc.subs(xc, 3) == 2 and close(101 / 82, 1.232, 5e-4) and sp.limit(Lc, xc, sp.oo) == 1, float(Lc.subs(xc, xm)))
# 表 η(θ), T
pp = sp.Symbol("p", positive=True)
check(T, "表 ベルヌーイ η=log(p/(1-p))：log f = x·η + log(1-p)", zero(sp.expand_log(xv * sp.log(pp) + (1 - xv) * sp.log(1 - pp) - (xv * sp.log(pp / (1 - pp)) + sp.log(1 - pp)), force=True)), "")
check(T, "表 正規(σ既知) η=μ/σ² の係数", zero(sp.diff(sp.expand(-(xv - mu)**2 / (2 * s**2)), xv).subs(xv, 0) - mu / s**2), "")
check(T, "表 正規 N(0,σ²) η=-1/(2σ²) の係数（x²）", zero(sp.diff(-(xv**2) / (2 * s**2), xv, 2) / 2 + 1 / (2 * s**2)), "")
# 一様 U(0,θ)：θ1>θ0 で比が max の非減少関数（数値）
mx = np.linspace(0.01, 2, 400)
ratio = np.where(mx <= 1, (1 / 1.5**5) / 1, 1e300)  # θ0=1, θ1=1.5, n=5（θ0 の台の外は +∞ を大きな値で代用）
ratio = np.where(mx > 1.5, np.nan, ratio)
check(T, "一様 U(0,θ): f(x;1.5)/f(x;1) は max の非減少関数（定数→∞）", np.all(np.diff(ratio[~np.isnan(ratio)]) >= 0), "")

# ================= uniformly-most-powerful-test =================
T = "uniformly-most-powerful-test"
one = lambda d: 1 - Phi(z(.05) - d)
two = lambda d: 1 - Phi(z(.025) - d) + Phi(-z(.025) - d)
check(T, "例1 右片側 μ=104,106,96: 0.6388, 0.9123, 0.0001", close(one(2), .6388) and close(one(3), .9123) and close(one(-2), .0001), (one(2), one(3), one(-2)))
check(T, "例1 両側 μ=104,106,96: 0.5160, 0.8508, 0.5160", close(two(2), .5160) and close(two(3), .8508) and close(two(-2), .5160), (two(2), two(3), two(-2)))
dd = np.linspace(0.01, 5, 500)
check(T, "μ>100 で右片側が常に両側を上回る", np.all(one(dd) > two(dd)), "")
check(T, "例2 c=16: P(Po10≥16)=0.0487≤0.05<P(≥15)=0.0835", pois.sf(15, 10) <= 0.05 < pois.sf(14, 10), "")
K = [pois.sf(15, 5 * l) for l in [1.5, 2, 2.5, 3, 4]]
check(T, "例2 K(λ) 表 0.0046, 0.0487, 0.1940, 0.4319, 0.8435", all(close(a, b) for a, b in zip(K, [0.0046, 0.0487, 0.1940, 0.4319, 0.8435])), [round(k, 5) for k in K])
check(T, "例2 K(λ) は λ の増加関数", np.all(np.diff([pois.sf(15, 5 * l) for l in np.linspace(.5, 5, 100)]) > 0), "")
check(T, "MP 棄却限界 μ0+zασ/√n は μ1 によらない（σ=10,n=25 → 103.29）", close(100 + z(.05) * 2, 103.29, 5e-3), 100 + z(.05) * 2)

# ================= power-function =================
T = "power-function"
check(T, "例1 右片側表 0.0500, 0.2595, 0.6388, 0.9123, 0.9907", all(close(one(d), v) for d, v in zip(range(5), [.05, .2595, .6388, .9123, .9907])), [round(one(d), 5) for d in range(5)])
check(T, "例1 両側表 0.0500, 0.1701, 0.5160, 0.8508, 0.9793", all(close(two(d), v) for d, v in zip(range(5), [.05, .1701, .5160, .8508, .9793])), [round(two(d), 5) for d in range(5)])
check(T, "例1 Φ(0.355)≈0.6388", close(Phi(0.355), .6388, 1e-4), Phi(0.355))
dlt = sp.Symbol("delta", real=True)
za = sp.Symbol("z_a", positive=True)
Phis = lambda u: (1 + sp.erf(u / sp.sqrt(2))) / 2
check(T, "K(μ0)=1-Φ(zα)=α, 両側でも δ=0 で α（数値）", close(one(0), .05, 1e-12) and close(two(0), .05, 1e-12), "")
check(T, "右片側 K は δ の増加関数（導関数 φ(zα-δ)>0）", zero(sp.diff(1 - Phis(za - dlt), dlt) - sp.exp(-(za - dlt)**2 / 2) / sp.sqrt(2 * sp.pi)), "")
Kb = [binom.sf(7, 10, p) for p in [.5, .6, .7, .8, .9]]
check(T, "例2 K(p) 表 0.0547, 0.1673, 0.3828, 0.6778, 0.9298", all(close(a, b) for a, b in zip(Kb, [.0547, .1673, .3828, .6778, .9298])), [round(k, 5) for k in Kb])
tc = stats.t.isf(.025, 24)
pt = stats.nct.sf(tc, 24, 2) + stats.nct.cdf(-tc, 24, 2)
check(T, "例3 t(24) 両側5%点 2.064, 非心度2 の検出力≈0.4840", close(tc, 2.064, 5e-4) and close(pt, .4840), (tc, pt))

# ================= sample-size-determination-test =================
T = "sample-size-determination-test"
nn3, dl, zb = sp.symbols("n delta z_b", positive=True)
sol = sp.solve(sp.Eq(za - sp.sqrt(nn3) * dl / s, -zb), nn3)
check(T, "片側公式 n=((zα+zβ)σ/δ)² を解いて確認", len(sol) == 1 and zero(sol[0] - ((za + zb) * s / dl)**2), sol)
z05, z025, z2, z1 = z(.05), z(.025), z(.2), z(.1)
check(T, "z0.05=1.645, z0.025=1.960, z0.2=0.842, z0.1=1.282", close(z05, 1.645, 5e-4) and close(z025, 1.960, 5e-4) and close(z2, .842, 5e-4) and close(z1, 1.282, 5e-4), (z05, z025, z2, z1))
# 本文は表示した z（1.645, 1.960, 0.842, 1.282）で計算しているので、その値で照合し、正確な z でも結論の n が同じことを確認
Z05, Z025, Z2, Z1 = 1.645, 1.960, 0.842, 1.282
r1 = (Z05 + Z2) * 10 / 5
check(T, "例1 片側 (1.645+0.842)×10/5=4.974, 2乗≈24.74 → n=25（正確な z でも 24.73→25）", close(r1, 4.974, 5e-4) and close(r1**2, 24.74, 5e-3) and int(np.ceil(r1**2)) == 25 and int(np.ceil(((z05 + z2) * 2)**2)) == 25, (r1, r1**2))
r2 = ((Z025 + Z2) * 10 / 5)**2
check(T, "例1 両側 ≈31.40 → n=32", close(r2, 31.40, 5e-3) and int(np.ceil(r2)) == 32 and int(np.ceil(((z025 + z2) * 2)**2)) == 32, r2)
check(T, "例1 片側 n=25 の検出力 Φ(0.855)≈0.8038", close(1 - Phi(z05 - 2.5), .8038) and close(Phi(.855), .8038, 1e-4), 1 - Phi(z05 - 2.5))
r3 = ((Z025 + Z1) * 2)**2
check(T, "例1 検出力90% 両側 ≈42.04 → n=43", close(r3, 42.04, 5e-3) and int(np.ceil(r3)) == 43 and int(np.ceil(((z025 + z1) * 2)**2)) == 43, r3)
tp = lambda n_: stats.nct.sf(stats.t.isf(.025, n_ - 1), n_ - 1, sqrt(n_) * .5) + stats.nct.cdf(-stats.t.isf(.025, n_ - 1), n_ - 1, sqrt(n_) * .5)
check(T, "例1 t 検定の検出力 n=32,33,34: 0.782, 0.795, 0.808", close(tp(32), .782, 5e-4) and close(tp(33), .795, 5e-4) and close(tp(34), .808, 5e-4), (tp(32), tp(33), tp(34)))
check(T, "例1 t 検定で 0.8 を超える最小 n=34", tp(33) < .8 <= tp(34), "")
r4 = 2 * (Z025 + Z2)**2 * 100 / 25
check(T, "例2 2群 ≈62.81 → 各群63", close(r4, 62.81, 5e-3) and int(np.ceil(r4)) == 63 and int(np.ceil(2 * (z025 + z2)**2 * 4)) == 63, r4)
r5 = ((Z05 * .5 + Z2 * sqrt(.24)) / .1)**2
check(T, "例3 母比率 ≈152.52 → n=153", close(r5, 152.52, 5e-3) and int(np.ceil(r5)) == 153 and int(np.ceil(((z05 * .5 + z2 * sqrt(.24)) / .1)**2)) == 153, r5)
cc = .5 + z05 * sqrt(.25 / 153)
pw3 = 1 - Phi((cc - .6) / sqrt(.24 / 153))
check(T, "例3 n=153 の検出力≈0.801（正規近似）", close(pw3, .801, 5e-4), pw3)
r6 = ((z05 + z2) * 10 / 2.5)**2
check(T, "グラフ説明 μ1=102.5 で n≈99（4倍）", int(np.ceil(r6)) == 99, r6)
check(T, "図の説明 n=25, μ1=105 の検出力≈0.80", close(one(2.5), .80, 5e-3), one(2.5))
sim = rng.normal(105, 10, (200000, 25)).mean(axis=1)
check(T, "シミュ n=25, μ=105 の検出力≈0.8038（許容0.003）", abs((sim > 100 + z05 * 2).mean() - .8038) < .003, (sim > 100 + z05 * 2).mean())

# ================= null-and-alternative-hypothesis =================
T = "null-and-alternative-hypothesis"
check(T, "例1 z=(498.2-500)/1=-1.8", close((498.2 - 500) / (4 / 4), -1.8, 1e-9), "")
check(T, "例1 片側 p≈0.0359, 両側≈0.0719", close(Phi(-1.8), .0359) and close(2 * Phi(-1.8), .0719), (Phi(-1.8), 2 * Phi(-1.8)))
check(T, "例2 ΣC(20,k),k≥15 =21700, /2^20≈0.0207", sum(comb(20, k) for k in range(15, 21)) == 21700 and 2**20 == 1048576 and close(21700 / 2**20, .0207), 21700 / 2**20)
check(T, "複合帰無仮説: 第1種の過誤は μ=μ0 で最大（右片側）", np.all(np.diff(one(np.linspace(-3, 0, 100))) > 0), "")

# ================= one-sided-two-sided-test =================
T = "one-sided-two-sided-test"
check(T, "棄却限界 1.645, 1.960", close(z05, 1.645, 5e-4) and close(z025, 1.960, 5e-4), "")
check(T, "z0.025²=χ²0.05(1)=3.841", close(z025**2, chi2.isf(.05, 1), 1e-9) and close(chi2.isf(.05, 1), 3.841, 5e-4), z025**2)
check(T, "本文の「1.96²≈3.84」（1.96²=3.8416、修正後）", close(1.96**2, 3.84, 5e-3), 1.96**2)
check(T, "例1 片側 p≈0.0359, 両側≈0.0719", close(1 - Phi(1.8), .0359) and close(2 * (1 - Phi(1.8)), .0719), "")
check(T, "例2 表（104, 106, 96）", close(one(2), .6388) and close(one(3), .9123) and close(1 - Phi(3.645), .0001) and close(two(-2), .5160), 1 - Phi(3.645))
zz = rng.standard_normal(1000000)
check(T, "シミュ H0 で両側 |Z|>1.96 の割合≈0.05（許容0.001）", abs((np.abs(zz) > z025).mean() - .05) < .001, (np.abs(zz) > z025).mean())

# ================= p-value =================
T = "p-value"
check(T, "例1 z=2.1 片側≈0.0179, 両側≈0.0357", close(1 - Phi(2.1), .0179) and close(2 * (1 - Phi(2.1)), .0357), (1 - Phi(2.1), 2 * (1 - Phi(2.1))))
check(T, "例2 二項係数 15504,4845,1140,190,20,1", [comb(20, k) for k in range(15, 21)] == [15504, 4845, 1140, 190, 20, 1], "")
check(T, "例2 片側≈0.0207, 両側≈0.0414", close(binom.sf(14, 20, .5), .0207) and close(2 * binom.sf(14, 20, .5), .0414), 2 * binom.sf(14, 20, .5))
check(T, "例3 t(15)=2.3 両側 p≈0.0362, t0.025(15)≈2.131", close(2 * stats.t.sf(2.3, 15), .0362) and close(stats.t.isf(.025, 15), 2.131, 5e-4), 2 * stats.t.sf(2.3, 15))
check(T, "例3 Po(3) P(X≥7)≈0.0335", close(pois.sf(6, 3), .0335) and close(1 - sum(exp(-3) * 3**k / np.prod(range(1, k + 1)) for k in range(7)), .0335), pois.sf(6, 3))
pv = 1 - Phi(rng.standard_normal(1000000))
check(T, "シミュ H0 で p 値 ~ U(0,1)：P(p≤0.05)≈0.05（許容0.001）, KS 統計量<0.002", abs((pv <= .05).mean() - .05) < .001 and stats.kstest(pv, stats.uniform.cdf).statistic < .002, (pv <= .05).mean())

# ================= rejection-region =================
T = "rejection-region"
check(T, "例1 50±3.92 → 46.08, 53.92", close(50 - 1.96 * 2, 46.08, 1e-9) and close(50 + 1.96 * 2, 53.92, 1e-9), "")
check(T, "例1 54.5 棄却, 52.0 棄却しない", abs(54.5 - 50) / 2 > 1.96 and abs(52 - 50) / 2 <= 1.96, "")
q = chi2.ppf(.05, 10)
check(T, "例2 χ²0.95(10)=3.940, ΣX<197.0", close(q, 3.940, 5e-4) and close(q * 50, 197.0, .05), (q, q * 50))
check(T, "例2 検出力 P(χ²(10)<2×3.9403)≈0.3595（正確な分位点で。丸めた 7.880 では 0.35944）", close(chi2.cdf(2 * q, 10), .3595) and close(chi2.cdf(7.880, 10), .3595, 1e-4), (chi2.cdf(2 * q, 10), chi2.cdf(7.880, 10)))
sm = rng.exponential(50, (200000, 5)).sum(axis=1)
check(T, "例2 シミュ θ=50 で P(ΣX<197.0)（許容0.004）", abs((sm < q * 50).mean() - .3595) < .004, (sm < q * 50).mean())
check(T, "例3 χ²0.05(9)=16.919, S²>7.520", close(chi2.isf(.05, 9), 16.919, 5e-4) and close(chi2.isf(.05, 9) * 4 / 9, 7.520, 5e-4), chi2.isf(.05, 9) * 4 / 9)

# ================= significance-level =================
T = "significance-level"
check(T, "z 表 0.10:1.282/1.645, 0.05:1.645/1.960, 0.01:2.326/2.576",
      all(close(z(a), v1, 5e-4) and close(z(a / 2), v2, 5e-4) for a, v1, v2 in [(.1, 1.282, 1.645), (.05, 1.645, 1.960), (.01, 2.326, 2.576)]), "")
sz = lambda m_: 1 - Phi(z05 - sqrt(20) * m_)
check(T, "複合仮説 μ=-0.1:0.0182, μ=-0.2:0.0056, μ=0:0.05", close(sz(-.1), .0182) and close(sz(-.2), .0056) and close(sz(0), .05, 1e-12), (sz(-.1), sz(-.2)))
check(T, "例1 P(X≥13..16)=0.1316, 0.0577, 0.0207, 0.0059", all(close(binom.sf(c_ - 1, 20, .5), v) for c_, v in [(13, .1316), (14, .0577), (15, .0207), (16, .0059)]), [round(binom.sf(c_ - 1, 20, .5), 5) for c_ in (13, 14, 15, 16)])
check(T, "例2 両側 0.0414, P(X≤6)+P(X≥14)=0.1153", close(binom.cdf(5, 20, .5) + binom.sf(14, 20, .5), .0414) and close(binom.cdf(6, 20, .5) + binom.sf(13, 20, .5), .1153), binom.cdf(6, 20, .5) + binom.sf(13, 20, .5))
check(T, "±3σ の外に出る確率≈0.27%", close(2 * (1 - Phi(3)), .0027, 5e-5), 2 * (1 - Phi(3)))
check(T, "5σ 片側≈2.9e-7", close(1 - Phi(5), 2.9e-7, 5e-9), N.sf(5))

# ================= type-i-and-ii-errors =================
T = "type-i-and-ii-errors"
check(T, "例1 α=(45+10+1)/1024=56/1024≈0.0547", comb(10, 8) + comb(10, 9) + 1 == 56 and close(56 / 1024, .0547), "")
check(T, "例1 1-β≈0.6778, β≈0.3222", close(binom.sf(7, 10, .8), .6778) and close(binom.cdf(7, 10, .8), .3222), "")
check(T, "例1 表 X≥7: α=0.1719, β=0.1209; X≥9: α=0.0107, β=0.6242",
      close(binom.sf(6, 10, .5), .1719) and close(binom.cdf(6, 10, .8), .1209) and close(binom.sf(8, 10, .5), .0107) and close(binom.cdf(8, 10, .8), .6242), (binom.cdf(6, 10, .8), binom.cdf(8, 10, .8)))
b2 = Phi(z05 - .5 * sqrt(20))
check(T, "例2 √20/2=2.236, β≈0.2772, 検出力 0.7228", close(.5 * sqrt(20), 2.236, 5e-4) and close(b2, .2772) and close(1 - b2, .7228) and close(Phi(-.591), .2772, 1e-4), b2)
check(T, "例2 α=0.01: β≈0.5360", close(Phi(z(.01) - .5 * sqrt(20)), .5360), Phi(z(.01) - .5 * sqrt(20)))
check(T, "例2 n=40: β≈0.0646", close(Phi(z05 - .5 * sqrt(40)), .0646), Phi(z05 - .5 * sqrt(40)))
sim = rng.normal(.5, 1, (200000, 20)).mean(axis=1)
check(T, "シミュ μ=0.5, n=20 の β≈0.2772（許容0.003）", abs((sim <= z05 / sqrt(20)).mean() - .2772) < .003, (sim <= z05 / sqrt(20)).mean())

# ================= multiple-testing =================
T = "multiple-testing"
mm = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
y1 = [0.05, 0.0975, 0.1426, 0.1855, 0.2262, 0.2649, 0.3366, 0.4013, 0.4596, 0.5367, 0.6415, 0.7226, 0.7854, 0.8715, 0.9231]
y2 = [0.05, 0.0494, 0.0492, 0.0491, 0.049, 0.049, 0.0489, 0.0489, 0.0489, 0.0489, 0.0488, 0.0488, 0.0488, 0.0488, 0.0488]
ok, err = series_ok(mm, y1, lambda m_: 1 - .95**m_, 5e-5)
check(T, "図 1-0.95^m の15点（許容5e-5）", ok, err)
ok, err = series_ok(mm, y2, lambda m_: 1 - (1 - .05 / m_)**m_, 5e-5)
check(T, "図 1-(1-0.05/m)^m の15点（許容5e-5）", ok, err)
check(T, "例1 1-0.95^20≈0.6415, 期待個数1", close(1 - .95**20, .6415) and 20 * .05 == 1.0, 1 - .95**20)
check(T, "例2 log0.5/log0.95≈13.51 → 14回（13回 <0.5 ≤ 14回）", close(log(.5) / log(.95), 13.51, 5e-3) and 1 - .95**13 < .5 <= 1 - .95**14, log(.5) / log(.95))
check(T, "例3 C(5,2)=10, 1-0.95^10≈0.4013, 1-0.995^10≈0.0489", comb(5, 2) == 10 and close(1 - .95**10, .4013) and close(1 - .995**10, .0489), 1 - .995**10)
check(T, "読み解き m=10 で約0.40, m=20 で約0.64", close(1 - .95**10, .40, 5e-3) and close(1 - .95**20, .64, 5e-3), "")
check(T, "1-(1-α)^m ≤ mα（m=1..50）", all(1 - .95**m_ <= .05 * m_ + 1e-15 for m_ in range(1, 51)), "")
U20 = rng.uniform(size=(200000, 20))
check(T, "シミュ 独立20検定の FWER≈0.6415（許容0.003）", abs((U20 <= .05).any(axis=1).mean() - .6415) < .003, (U20 <= .05).any(axis=1).mean())

# ================= bonferroni-correction =================
T = "bonferroni-correction"
mb = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
yb = [0.8508, 0.776, 0.7277, 0.6923, 0.6643, 0.6412, 0.6047, 0.5765, 0.5536, 0.5258, 0.4907, 0.4641, 0.4428, 0.4101, 0.3857]
pb = lambda m_: 1 - Phi(z(.05 / m_ / 2) - 3) + Phi(-z(.05 / m_ / 2) - 3)
ok, err = series_ok(mb, yb, pb, 5e-5)
check(T, "図 δ=3 両側 α/m の検出力 15点（許容5e-5）", ok, err)
check(T, "シダック m=10: 1-0.95^0.1≈0.00512", close(1 - .95**.1, .00512, 5e-6), 1 - .95**.1)
adj = np.minimum(1, 5 * p5)
check(T, "例1 調整済み p 値 0.020, 0.055, 0.095, 0.150, 1.000", np.allclose(adj, [.02, .055, .095, .15, 1.0]), adj)
check(T, "例1 ボンフェローニ1個, ホルム2個, BH 4個, 補正なし4個", (p5 <= .01).sum() == 1 and holm(p5, .05) == 2 and bh(p5, .05) == 4 and (p5 <= .05).sum() == 4, "")
check(T, "例2 1-0.05/3≈0.9833, z0.05/6≈2.394, 比≈1.22", close(1 - .05 / 3, .9833) and close(z(.05 / 6), 2.394, 5e-4) and close(z(.05 / 6) / z025, 1.22, 5e-3), z(.05 / 6))
check(T, "例3 単独の検出力≈0.851, z0.0025≈2.807, 10項目で≈0.577", close(pb(1), .851, 5e-4) and close(z(.0025), 2.807, 5e-4) and close(pb(10), .577, 5e-4), (pb(1), pb(10)))
# 依存があっても FWER≤α（等相関 ρ=0.8 の正規、m=10、すべて H0 真）
rho, mT = 0.8, 10
Zc = sqrt(rho) * rng.standard_normal((200000, 1)) + sqrt(1 - rho) * rng.standard_normal((200000, mT))
pc = 2 * N.sf(np.abs(Zc))
fw_dep = (pc <= .05 / mT).any(axis=1).mean()
Zi = rng.standard_normal((200000, mT))
fw_ind = (2 * N.sf(np.abs(Zi)) <= .05 / mT).any(axis=1).mean()
check(T, "シミュ 相関0.8でも FWER≤0.05、独立より小さい（保守的）", fw_dep < .05 and fw_dep < fw_ind - .005, (fw_dep, fw_ind))
check(T, "シミュ 独立 m=10 の FWER≈1-(1-0.005)^10=0.0489（許容0.002）", abs(fw_ind - (1 - .995**10)) < .002, fw_ind)

# ================= holm-method =================
T = "holm-method"
check(T, "図 ホルム基準 0.01, 0.0125, 0.0167, 0.025, 0.05", np.allclose(np.round(.05 / (5 - np.arange(5)), 4), [.01, .0125, .0167, .025, .05]), .05 / (5 - np.arange(5)))
mp = (5 - np.arange(5)) * p5
check(T, "例1 (m-k+1)p 0.020, 0.044, 0.057, 0.060, 0.250", np.allclose(mp, [.02, .044, .057, .06, .25]), mp)
check(T, "例1 調整済み p（累積最大）", np.allclose(np.minimum(1, np.maximum.accumulate(mp)), [.02, .044, .057, .06, .25]), "")
check(T, "例1 0.004≤0.01, 0.011≤0.0125, 0.019>0.0167 → 2個棄却", holm(p5, .05) == 2 and .019 > .05 / 3, "")
check(T, "ボンフェローニの基準 α/m ≤ α/(m-k+1)（一様に強い）", all(.05 / 5 <= .05 / (5 - k) for k in range(5)), "")
# 依存下での FWER（m=10、m0=6 が真、残り4は効果 4、等相関 0.5）
rho = 0.5
Zh = sqrt(rho) * rng.standard_normal((100000, 1)) + sqrt(1 - rho) * rng.standard_normal((100000, 10))
Zh[:, 6:] += 4
ph_ = 2 * N.sf(np.abs(Zh))
order = np.argsort(ph_, axis=1)
srt = np.take_along_axis(ph_, order, axis=1)
okk = srt <= .05 / (10 - np.arange(10))
nrej = np.where(okk.all(axis=1), 10, np.argmin(okk, axis=1))
rejected_true = np.array([np.any(order[i, :nrej[i]] < 6) for i in range(len(nrej))])
check(T, "シミュ 相関0.5、m0=6 でホルム法の FWER≤0.05（許容: 0.05+0.002）", rejected_true.mean() <= .052, rejected_true.mean())

# ================= false-discovery-rate =================
T = "false-discovery-rate"
check(T, "図 BH 基準 kα/m = 0.01..0.05", np.allclose(np.arange(1, 6) * .05 / 5, [.01, .02, .03, .04, .05]), "")
r = 5 * p5 / np.arange(1, 6)
check(T, "例1 5p/k = 0.0200, 0.0275, 0.0317, 0.0375, 0.2500", all(close(a, b) for a, b in zip(r, [.02, .0275, .0317, .0375, .25])), r)
qv = np.minimum.accumulate(np.minimum(1, r)[::-1])[::-1]
check(T, "例1 調整済み p（右からの累積最小）同じ値", all(close(a, b) for a, b in zip(qv, [.02, .0275, .0317, .0375, .25])), qv)
check(T, "例1 k*=4（○○○○×）", bh(p5, .05) == 4 and list(p5 <= np.arange(1, 6) * .01) == [True] * 4 + [False], "")
check(T, "例2 E[V]=45, E[S]=80, 45/125=0.36", 900 * .05 == 45 and 100 * .8 == 80 and close(45 / 125, .36, 1e-12), "")
# BH の FDR = m0/m·q（独立）、FDR≤FWER
m_, m0, q_ = 20, 15, .05
reps = 100000
Zf = rng.standard_normal((reps, m_))
Zf[:, m0:] += 3
pf = N.sf(Zf)
ordf = np.argsort(pf, axis=1)
sf_ = np.take_along_axis(pf, ordf, axis=1)
below = sf_ <= np.arange(1, m_ + 1) / m_ * q_
kst = np.where(below.any(axis=1), m_ - np.argmax(below[:, ::-1], axis=1), 0)
rank_of = np.argsort(ordf, axis=1)
rej = rank_of < kst[:, None]
V = rej[:, :m0].sum(axis=1)
R = rej.sum(axis=1)
fdp = np.where(R > 0, V / np.maximum(R, 1), 0)
fdr = fdp.mean()
fwer = (V >= 1).mean()
check(T, "シミュ 独立 m=20, m0=15 で BH の FDR≈m0q/m=0.0375（許容0.002）", abs(fdr - .0375) < .002, fdr)
check(T, "シミュ FDR≤FWER", fdr <= fwer, (fdr, fwer))
Zn = rng.standard_normal((reps, m_))
pn = N.sf(Zn)
sn = np.sort(pn, axis=1)
anyrej = (sn <= np.arange(1, m_ + 1) / m_ * q_).any(axis=1)
check(T, "シミュ 全帰無仮説が真なら FDR=FWER≈q=0.05（許容0.002）", abs(anyrej.mean() - .05) < .002, anyrej.mean())


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

rngC = np.random.default_rng(20261023 + 10)

# ================= randomized-test =================
T = "randomized-test"
b = stats.binom(20, 0.5)
cc(T, "P(T≥15)", b.sf(14), "0.0207")
cc(T, "P(T≥14)", b.sf(13), "0.0577")
cc(T, "P(T=14)", b.pmf(14), "0.0370")
g = (0.05 - b.sf(14)) / b.pmf(14)
cc(T, "γ（正確）", g, "0.7928")
size = b.sf(14) + g * b.pmf(14)
check(T, "サイズ = 0.05", abs(size - 0.05) < 1e-12, size)
b7 = stats.binom(20, 0.7)
cc(T, "p=0.7 で P(T≥15)", b7.sf(14), "0.4164")
cc(T, "p=0.7 で P(T=14)", b7.pmf(14), "0.1916")
cc(T, "無作為化検定の検出力", b7.sf(14) + g * b7.pmf(14), "0.5683")
Tn = rngC.binomial(20, 0.5, 400000)
rej = (Tn >= 15) | ((Tn == 14) & (rngC.random(400000) < g))
mcC(T, "無作為化検定の棄却率 ≈0.05", rej.mean(), 0.05, 0.002)


print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
