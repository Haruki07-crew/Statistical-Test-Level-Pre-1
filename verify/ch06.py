"""第6章 検算スクリプト（連続型分布・標本分布・非心分布）。
記号計算は sympy、数値は scipy.stats、主張値との比較は小数第4位（許容 5e-5、表示桁が少ないものは丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from math import sqrt, log, exp, pi, atan, gamma as G
import numpy as np
import sympy as sp
from scipy import stats, integrate, optimize

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def zero(expr):
    return sp.simplify(expr) == 0


rng = np.random.default_rng(20261023)
N = 400000
t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
a, b, lam, beta, alpha, sig, k = sp.symbols("a b lambda beta alpha sigma k", positive=True)
mu = sp.Symbol("mu", real=True)
Phi = stats.norm.cdf


def ks(sample, dist, args=()):
    """KS 統計量。分布名を凍結分布の cdf に直して渡す（scipy の文字列指定の不具合回避）。"""
    return stats.kstest(sample, getattr(stats, dist)(*args).cdf)


def moments_from_mgf(M):
    m1 = sp.simplify(sp.limit(sp.diff(M, t), t, 0))
    m2 = sp.simplify(sp.limit(sp.diff(M, t, 2), t, 0))
    return m1, sp.simplify(m2 - m1**2)


# ================= uniform-distribution =================
T = "uniform-distribution"
f = 1 / (b - a)
EX = sp.integrate(x * f, (x, a, b))
EX2 = sp.integrate(x**2 * f, (x, a, b))
check(T, "E=(a+b)/2", zero(EX - (a + b) / 2), sp.factor(EX))
check(T, "V=(b-a)^2/12", zero(EX2 - EX**2 - (b - a) ** 2 / 12), sp.factor(EX2 - EX**2))
M = sp.integrate(sp.exp(t * x) * f, (x, a, b), conds="none")
check(T, "M(t)=(e^{tb}-e^{ta})/(t(b-a))", zero(M - (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))), "sympy")
Mu = (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))
m1, v = moments_from_mgf(Mu)
check(T, "M(0)=1, M'(0)=(a+b)/2, V=(b-a)^2/12（MGF の極限）", zero(sp.limit(Mu, t, 0) - 1) and zero(m1 - (a + b) / 2) and zero(v - (b - a) ** 2 / 12), (m1, sp.factor(v)))
phi_num = integrate.quad(lambda u: np.cos(1.3 * u), -1, 2)[0] / 3 + 1j * integrate.quad(lambda u: np.sin(1.3 * u), -1, 2)[0] / 3
phi_cl = (np.exp(1j * 1.3 * 2) - np.exp(1j * 1.3 * -1)) / (1j * 1.3 * 3)
check(T, "特性関数 (e^{itb}-e^{ita})/(it(b-a))（a=-1,b=2,t=1.3 数値積分）", abs(phi_num - phi_cl) < 1e-10, phi_cl)
u = sp.Symbol("u", positive=True)
check(T, "E[U]=1/2, E[U^2]=1/3, V[U]=1/12", sp.integrate(u, (u, 0, 1)) == sp.Rational(1, 2) and sp.integrate(u**2, (u, 0, 1)) == sp.Rational(1, 3), "sympy")
U = rng.uniform(size=N)
check(T, "-log U/λ ~ Exp(λ)（λ=2, KS 統計量<0.005）", ks(-np.log(U) / 2, "expon", args=(0, 0.5)).statistic < 0.005, ks(-np.log(U) / 2, "expon", args=(0, 0.5)).statistic)
Us = np.sort(rng.uniform(size=(100000, 5)), axis=1)
ksmax = max(ks(Us[:, j], "beta", args=(j + 1, 5 - j)).statistic for j in range(5))
check(T, "順序統計量 U_(k) ~ Beta(k,n-k+1)（n=5, 各KS<0.01）", ksmax < 0.01, ksmax)
Xn = rng.normal(size=N)
check(T, "確率積分変換 Φ(Z)~U(0,1)（KS<0.005）", ks(Phi(Xn), "uniform").statistic < 0.005, ks(Phi(Xn), "uniform").statistic)
S2 = rng.uniform(size=N) + rng.uniform(size=N)
check(T, "U1+U2 は三角形分布（KS<0.005）", ks(S2, "triang", args=(0.5, 0, 2)).statistic < 0.005, ks(S2, "triang", args=(0.5, 0, 2)).statistic)
check(T, "例1 E=5, V≈8.333, SD≈2.887, P(X≥7)=0.3", close(100 / 12, 8.333, 5e-4) and close(sqrt(100 / 12), 2.887, 5e-4) and close(stats.uniform(0, 10).sf(7), 0.3, 1e-12), sqrt(100 / 12))
check(T, "例2 0.9^5≈0.5905, E[M]=5/6≈0.8333=Beta(5,1) 平均", close(0.9**5, 0.5905) and close(sp.integrate(5 * u**5, (u, 0, 1)), 0.8333) and close(stats.beta(5, 1).mean(), 5 / 6, 1e-12), 0.9**5)
Mx = rng.uniform(size=(100000, 5)).max(axis=1)
check(T, "例2 最大値の P(M≤0.9)（シミュ、許容0.005）", abs(np.mean(Mx <= 0.9) - 0.59049) < 0.005, np.mean(Mx <= 0.9))
check(T, "丸め誤差 U(-h/2,h/2) の分散 h^2/12（h=0.1）", close(stats.uniform(-0.05, 0.1).var(), 0.01 / 12, 1e-12), stats.uniform(-0.05, 0.1).var())

# ================= normal-distribution =================
T = "normal-distribution"
z = sp.Symbol("z", real=True)
pdfN = sp.exp(-(x - mu) ** 2 / (2 * sig**2)) / (sp.sqrt(2 * sp.pi) * sig)
check(T, "密度の積分=1", zero(sp.integrate(pdfN, (x, -sp.oo, sp.oo)) - 1), "sympy")
MZ = sp.integrate(sp.exp(t * z) * sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi), (z, -sp.oo, sp.oo))
check(T, "M_Z(t)=e^{t^2/2}", zero(MZ - sp.exp(t**2 / 2)), MZ)
check(T, "平方完成 tz-z^2/2=-(z-t)^2/2+t^2/2", zero(t * z - z**2 / 2 - (-(z - t) ** 2 / 2 + t**2 / 2)), "sympy")
MN = sp.exp(mu * t + sig**2 * t**2 / 2)
check(T, "M_X(t)=e^{μt}M_Z(σt)=exp(μt+σ²t²/2)", zero(sp.exp(mu * t) * MZ.subs(t, sig * t) - MN), "sympy")
m1, v = moments_from_mgf(MN)
check(T, "E=μ, V=σ²（MGF 微分）", zero(m1 - mu) and zero(v - sig**2), (m1, v))
K = mu * t + sig**2 * t**2 / 2
check(T, "K'(0)=μ, K''(0)=σ², K'''=0", zero(sp.diff(K, t).subs(t, 0) - mu) and zero(sp.diff(K, t, 2) - sig**2) and sp.diff(K, t, 3) == 0, "sympy")
phiN = sp.integrate(sp.exp(sp.I * t * z) * sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi), (z, -sp.oo, sp.oo))
check(T, "特性関数 φ_Z(t)=e^{-t²/2}（→ exp(iμt-σ²t²/2)）", zero(phiN - sp.exp(-t**2 / 2)), phiN)
check(T, "歪度0・超過尖度0", stats.norm.stats(moments="sk") == (0.0, 0.0), stats.norm.stats(moments="sk"))
d2 = sp.diff(pdfN, x, 2)
check(T, "変曲点 μ±σ", zero(d2.subs(x, mu + sig)) and zero(d2.subs(x, mu - sig)), "sympy")
m1s, m2s = sp.symbols("m1 m2", real=True)
s1, s2 = sp.symbols("s1 s2", positive=True)
check(T, "再生性 MGF 積", zero(sp.exp(m1s * t + s1**2 * t**2 / 2) * sp.exp(m2s * t + s2**2 * t**2 / 2) - sp.exp((m1s + m2s) * t + (s1**2 + s2**2) * t**2 / 2)), "sympy")
X1 = rng.normal(1, 2, N); X2 = rng.normal(-3, 1.5, N)
check(T, "X1+X2~N(-2, 6.25)（シミュ KS<0.005）", ks(X1 + X2, "norm", args=(-2, 2.5)).statistic < 0.005, ks(X1 + X2, "norm", args=(-2, 2.5)).statistic)
check(T, "例1 Φ(-2)≈0.0228, Φ(1.25)-Φ(-1.25)≈0.7887", close(Phi(-2), 0.0228) and close(Phi(1.25) - Phi(-1.25), 0.7887), (Phi(-2), Phi(1.25) - Phi(-1.25)))
check(T, "例2 √325≈18.028, 20/18.028≈1.1094, P≈0.1336", close(sqrt(325), 18.028, 5e-4) and close(20 / sqrt(325), 1.1094) and close(stats.norm(110, sqrt(325)).sf(130), 0.1336), stats.norm(110, sqrt(325)).sf(130))
check(T, "例3 2Φ(-2)≈0.0455", close(2 * Phi(-2), 0.0455), 2 * Phi(-2))
Xs = rng.normal(5, 2, size=(100000, 6))
S2s = Xs.var(axis=1, ddof=1)
check(T, "正規標本で (n-1)S²/σ² ~ χ²(n-1)（n=6, KS<0.006）, corr(X̄,S²)≈0", ks(5 * S2s / 4, "chi2", args=(5,)).statistic < 0.006 and abs(np.corrcoef(Xs.mean(axis=1), S2s)[0, 1]) < 0.01, ks(5 * S2s / 4, "chi2", args=(5,)).statistic)

# ================= standard-normal-distribution =================
T = "standard-normal-distribution"
phi = sp.exp(-z**2 / 2) / sp.sqrt(2 * sp.pi)
EZ = [sp.integrate(z**n * phi, (z, -sp.oo, sp.oo)) for n in range(9)]
check(T, "E=0, V=1, 奇数次=0", EZ[1] == 0 and EZ[2] == 1 and EZ[3] == 0 and EZ[5] == 0 and EZ[7] == 0, EZ[:4])
check(T, "E[Z^4]=3, E[Z^6]=15, E[Z^8]=105=(2k)!/(2^k k!)", EZ[4] == 3 and EZ[6] == 15 and EZ[8] == 105 and all(EZ[2 * kk] == sp.factorial(2 * kk) / (2**kk * sp.factorial(kk)) for kk in range(5)), EZ[4::2])
check(T, "漸化式 E[Z^n]=(n-1)E[Z^{n-2}]", all(EZ[n] == (n - 1) * EZ[n - 2] for n in range(2, 9)), "n=2..8")
check(T, "φ'(z)=-zφ(z)", zero(sp.diff(phi, z) + z * phi), "sympy")
ser = sp.series(sp.exp(t**2 / 2), t, 0, 9).removeO()
check(T, "MGF 展開の係数 E[Z^{2k}]", all(sp.simplify(ser.coeff(t, 2 * kk) * sp.factorial(2 * kk) - EZ[2 * kk]) == 0 for kk in range(5)), "k=0..4")
check(T, "z_0.05=1.645, z_0.025=1.960, z_0.005=2.576", close(stats.norm.isf(0.05), 1.645, 5e-4) and close(stats.norm.isf(0.025), 1.960, 5e-4) and close(stats.norm.isf(0.005), 2.576, 5e-4), [stats.norm.isf(q) for q in (0.05, 0.025, 0.005)])
check(T, "φ(0)=1/√(2π)≈0.3989", close(1 / sqrt(2 * pi), 0.3989), 1 / sqrt(2 * pi))
yv = sp.Symbol("y", positive=True)
FY = 2 * sp.erf(sp.sqrt(yv) / sp.sqrt(2)) / 2  # 2Φ(√y)-1 = erf(√y/√2)
fY = sp.diff(FY, yv)
check(T, "Z² の密度 = y^{-1/2}e^{-y/2}/√(2π) = χ²(1)", zero(fY - yv ** sp.Rational(-1, 2) * sp.exp(-yv / 2) / sp.sqrt(2 * sp.pi)), sp.simplify(fY))
check(T, "例1 1-Φ(1.5)=0.0668, Φ(2)-Φ(-1)≈0.8186", close(stats.norm.sf(1.5), 0.0668) and close(Phi(2) - Phi(-1), 0.8186), (stats.norm.sf(1.5), Phi(2) - Phi(-1)))
check(T, "例2 0.6827, 0.9545, 0.9973", close(1 - 2 * Phi(-1), 0.6827) and close(1 - 2 * Phi(-2), 0.9545) and close(1 - 2 * Phi(-3), 0.9973), [1 - 2 * Phi(-q) for q in (1, 2, 3)])
EabsZ = sp.integrate(2 * z * phi, (z, 0, sp.oo))
check(T, "例3 E|Z|=√(2/π)≈0.7979", zero(EabsZ - sp.sqrt(2 / sp.pi)) and close(sqrt(2 / pi), 0.7979), float(EabsZ))
check(T, "例4 片側 p≈0.0107, 両側≈0.0214", close(stats.norm.sf(2.3), 0.0107) and close(2 * stats.norm.sf(2.3), 0.0214), stats.norm.sf(2.3))
check(T, "±3 を超える確率 約0.27%", close(2 * Phi(-3), 0.0027, 5e-5), 2 * Phi(-3))
check(T, "1.96²≈3.84（修正後）, χ²(1) 上側5%点 3.841=z_0.025², V[Z²]=2", close(1.96**2, 3.84, 5e-3) and close(stats.norm.isf(0.025) ** 2, 3.841, 5e-4) and close(stats.chi2.isf(0.05, 1), 3.841, 5e-4) and EZ[4] - 1 == 2, stats.chi2.isf(0.05, 1))

# ================= exponential-distribution =================
T = "exponential-distribution"
xp = sp.Symbol("x", positive=True)
fe = lam * sp.exp(-lam * xp)
check(T, "F(x)=1-e^{-λx}", zero(sp.integrate(fe.subs(xp, u), (u, 0, xp)) - (1 - sp.exp(-lam * xp))), "sympy")
Me = lam / (lam - t)
Me_int = (lam * sp.integrate(sp.exp(-k * xp), (xp, 0, sp.oo))).subs(k, lam - t)  # e^{tx}λe^{-λx}=λe^{-(λ-t)x}, k=λ-t>0
check(T, "M(t)=λ/(λ-t)（t<λ）", zero(Me_int - Me), sp.simplify(Me_int))
check(T, "M'=λ/(λ-t)², M''=2λ/(λ-t)³", zero(sp.diff(Me, t) - lam / (lam - t) ** 2) and zero(sp.diff(Me, t, 2) - 2 * lam / (lam - t) ** 3), "sympy")
m1, v = moments_from_mgf(Me)
check(T, "E=1/λ, V=1/λ², CV=1", zero(m1 - 1 / lam) and zero(v - 1 / lam**2), (m1, v))
check(T, "特性関数 λ/(λ-it)（λ=2,t=0.7 数値積分）", abs(complex(integrate.quad(lambda y: np.cos(0.7 * y) * 2 * np.exp(-2 * y), 0, np.inf)[0], integrate.quad(lambda y: np.sin(0.7 * y) * 2 * np.exp(-2 * y), 0, np.inf)[0]) - 2 / (2 - 0.7j)) < 1e-9, 2 / (2 - 0.7j))
check(T, "中央値 log2/λ, ハザード λ", zero(1 - sp.exp(-lam * sp.log(2) / lam) - sp.Rational(1, 2)) and zero(fe / sp.exp(-lam * xp) - lam), "sympy")
s_, t_ = sp.symbols("s t_", positive=True)
check(T, "無記憶性 S(s+t)/S(s)=S(t)", zero(sp.exp(-lam * (s_ + t_)) / sp.exp(-lam * s_) - sp.exp(-lam * t_)), "sympy")
Ex = rng.exponential(1 / 0.5, N)
cond = Ex[Ex > 2] - 2
check(T, "無記憶性（シミュ、λ=0.5, P(X>5|X>2) 許容0.005）", abs(np.mean(cond > 3) - exp(-1.5)) < 0.005, np.mean(cond > 3))
Sg = rng.exponential(1 / 1.5, size=(N, 4)).sum(axis=1)
check(T, "Exp(λ) 4個の和 ~ Ga(4, 1/λ)（KS<0.005）", ks(Sg, "gamma", args=(4, 0, 1 / 1.5)).statistic < 0.005, ks(Sg, "gamma", args=(4, 0, 1 / 1.5)).statistic)
Mn = np.minimum(np.minimum(rng.exponential(10, N), rng.exponential(5, N)), rng.exponential(1 / 0.3, N))
check(T, "最小値 ~ Exp(Σλ)（0.1,0.2,0.3 → 平均1.667, 許容0.01）", abs(Mn.mean() - 1 / 0.6) < 0.01 and ks(Mn, "expon", args=(0, 1 / 0.6)).statistic < 0.005, Mn.mean())
check(T, "Exp(1/2) = χ²(2)", np.allclose(stats.expon(scale=2).pdf(np.linspace(0.1, 10, 50)), stats.chi2(2).pdf(np.linspace(0.1, 10, 50))), "pdf 一致")
pp = 0.01; Dl = 0.01
check(T, "幾何→指数の極限（p=λΔ, Δ→0）", abs((1 - 0.5 * 1e-6) ** (3 / 1e-6) - exp(-1.5)) < 1e-6, (1 - 0.5e-6) ** 3e6)
check(T, "例1 中央値≈1.386, P(X>3)≈0.2231", close(log(2) / 0.5, 1.386, 5e-4) and close(exp(-1.5), 0.2231) and close(stats.expon(scale=2).sf(3), 0.2231), exp(-1.5))
check(T, "例1 P(X>5|X>2)≈0.2231", close(stats.expon(scale=2).sf(5) / stats.expon(scale=2).sf(2), 0.2231), stats.expon(scale=2).sf(5) / stats.expon(scale=2).sf(2))
X1 = rng.exponential(10, N); X2 = rng.exponential(5, N); X3 = rng.exponential(1 / 0.3, N)
first3 = np.mean((X3 < X1) & (X3 < X2))
check(T, "例2 E[T]=1/0.6≈1.667, 部品3が最初の確率 0.5（シミュ 許容0.005）", close(1 / 0.6, 1.667, 5e-4) and abs(first3 - 0.5) < 0.005, first3)

# ================= gamma-distribution =================
T = "gamma-distribution"
fg = xp ** (alpha - 1) * sp.exp(-xp / beta) / (sp.gamma(alpha) * beta**alpha)
Mg = (1 - beta * t) ** (-alpha)
Mg_num = all(abs(integrate.quad(lambda y: np.exp(tv * y + stats.gamma.logpdf(y, av, scale=bv)), 0, 400)[0] - (1 - bv * tv) ** (-av)) < 1e-7 for av, bv, tv in [(2.5, 1.5, 0.3), (0.7, 2, -0.4), (4, 0.5, 1.2)])
check(T, "M(t)=(1-βt)^{-α}（数値積分3組）", Mg_num, "quad")
m1, v = moments_from_mgf(Mg)
check(T, "E=αβ, V=αβ²", zero(m1 - alpha * beta) and zero(v - alpha * beta**2), (m1, v))
Kg = -alpha * sp.log(1 - beta * t)
check(T, "K'=αβ/(1-βt), K''=αβ²/(1-βt)²", zero(sp.diff(Kg, t) - alpha * beta / (1 - beta * t)) and zero(sp.diff(Kg, t, 2) - alpha * beta**2 / (1 - beta * t) ** 2), "sympy")
check(T, "レート表記 (λ/(λ-t))^α = (1-t/λ)^{-α}, 特性関数 (1-iβt)^{-α}（数値 α=2.5,β=1.5,t=0.6）", zero((lam / (lam - t)) ** 2 - (1 - t / lam) ** (-2)) and abs(complex(integrate.quad(lambda y: np.cos(0.6 * y) * stats.gamma.pdf(y, 2.5, scale=1.5), 0, np.inf)[0], integrate.quad(lambda y: np.sin(0.6 * y) * stats.gamma.pdf(y, 2.5, scale=1.5), 0, np.inf)[0]) - (1 - 1.5j * 0.6) ** (-2.5)) < 1e-7, "quad")
mode = sp.solve(sp.diff(sp.log(fg), xp), xp)
check(T, "最頻値 (α-1)β", len(mode) == 1 and zero(mode[0] - (alpha - 1) * beta), mode)
check(T, "Γ(1/2)=√π, Γ(n)=(n-1)!, Γ(α+1)=αΓ(α)", sp.gamma(sp.Rational(1, 2)) == sp.sqrt(sp.pi) and sp.gamma(6) == sp.factorial(5) and zero(sp.gamma(alpha + 1) - alpha * sp.gamma(alpha)), "sympy")
Xg = rng.gamma(2.5, 1, N)
check(T, "βX ~ Ga(α,β)（β=3, KS<0.005）", ks(3 * Xg, "gamma", args=(2.5, 0, 3)).statistic < 0.005, ks(3 * Xg, "gamma", args=(2.5, 0, 3)).statistic)
Y1 = rng.gamma(1.3, 2, N); Y2 = rng.gamma(2.2, 2, N)
check(T, "再生性 Ga(1.3,2)+Ga(2.2,2)=Ga(3.5,2)（KS<0.005）", ks(Y1 + Y2, "gamma", args=(3.5, 0, 2)).statistic < 0.005, ks(Y1 + Y2, "gamma", args=(3.5, 0, 2)).statistic)
B = Y1 / (Y1 + Y2)
check(T, "X/(X+Y)~Beta(a,b), X+Y と無相関（KS<0.005, |r|<0.01）", ks(B, "beta", args=(1.3, 2.2)).statistic < 0.005 and abs(np.corrcoef(B, Y1 + Y2)[0, 1]) < 0.01, ks(B, "beta", args=(1.3, 2.2)).statistic)
check(T, "Ga(k/2,2)=χ²(k)（k=5 pdf 一致）", np.allclose(stats.gamma(2.5, scale=2).pdf(np.linspace(0.1, 20, 50)), stats.chi2(5).pdf(np.linspace(0.1, 20, 50))), "pdf")
check(T, "P(T_n>t)=P(N(t)≤n-1)（n=4,λ=0.7,t=5）", close(stats.gamma(4, scale=1 / 0.7).sf(5), stats.poisson(3.5).cdf(3), 1e-12), stats.gamma(4, scale=1 / 0.7).sf(5))
check(T, "例1 E=6, V=12", stats.gamma(3, scale=2).mean() == 6 and close(stats.gamma(3, scale=2).var(), 12, 1e-9), stats.gamma(3, scale=2).var())
check(T, "例1 P(T>10)=18.5e^{-5}≈0.1247", close(18.5 * exp(-5), 0.1247) and close(stats.gamma(3, scale=2).sf(10), 18.5 * exp(-5), 1e-12), stats.gamma(3, scale=2).sf(10))
check(T, "例2 Ga(3.5,2)=χ²(7), 平均7, 分散14", close(stats.gamma(3.5, scale=2).mean(), 7, 1e-12) and close(stats.gamma(3.5, scale=2).var(), 14, 1e-9) and close(stats.chi2(4).var(), stats.gamma(2, scale=2).var(), 1e-9), "scipy")

# ================= beta-distribution =================
T = "beta-distribution"
xb = sp.Symbol("x", positive=True)
Bab = sp.gamma(a) * sp.gamma(b) / sp.gamma(a + b)
check(T, "B(a,b)=Γ(a)Γ(b)/Γ(a+b)（数値 a=2.3,b=3.7）", close(integrate.quad(lambda y: y**1.3 * (1 - y) ** 2.7, 0, 1)[0], G(2.3) * G(3.7) / G(6), 1e-10), G(2.3) * G(3.7) / G(6))
EB1 = sp.gammasimp(sp.gamma(a + 1) * sp.gamma(a + b) / (sp.gamma(a) * sp.gamma(a + b + 1)))
EB2 = sp.gammasimp(sp.gamma(a + 2) * sp.gamma(a + b) / (sp.gamma(a) * sp.gamma(a + b + 2)))
check(T, "E=a/(a+b)", zero(EB1 - a / (a + b)), EB1)
check(T, "E[X²]=a(a+1)/((a+b)(a+b+1))", zero(EB2 - a * (a + 1) / ((a + b) * (a + b + 1))), EB2)
check(T, "V=ab/((a+b)²(a+b+1))", zero(EB2 - EB1**2 - a * b / ((a + b) ** 2 * (a + b + 1))), "sympy")
check(T, "E[X^k]=Π(a+r)/(a+b+r)（a=2.3,b=3.7,k=1..5）", all(close(stats.beta(2.3, 3.7).moment(kk), np.prod([(2.3 + r) / (6 + r) for r in range(kk)]), 1e-10) for kk in range(1, 6)), "scipy")
check(T, "MGF=1+Σ E[X^k]t^k/k!（a=2,b=3,t=1.5 数値）", close(integrate.quad(lambda y: np.exp(1.5 * y) * stats.beta.pdf(y, 2, 3), 0, 1)[0], 1 + sum(np.prod([(2 + r) / (5 + r) for r in range(kk)]) * 1.5**kk / sp.factorial(kk) for kk in range(1, 40)), 1e-10), "級数")
modeb = sp.solve(sp.diff((a - 1) * sp.log(xb) + (b - 1) * sp.log(1 - xb), xb), xb)
check(T, "最頻値 (a-1)/(a+b-2)", len(modeb) == 1 and zero(modeb[0] - (a - 1) / (a + b - 2)), modeb)
check(T, "1-X ~ Beta(b,a)", np.allclose(stats.beta(2, 5).pdf(1 - np.linspace(0.01, 0.99, 50)), stats.beta(5, 2).pdf(np.linspace(0.01, 0.99, 50))), "pdf")
check(T, "Beta(1,1)=U(0,1)", np.allclose(stats.beta(1, 1).pdf(np.linspace(0.01, 0.99, 20)), 1), "pdf")
Bx = rng.beta(3, 4, N)
check(T, "Beta(m/2,n/2) → (X/m)/((1-X)/n) ~ F(m,n)（m=6,n=8, KS<0.005）", ks((Bx / 6) / ((1 - Bx) / 8), "f", args=(6, 8)).statistic < 0.005, ks((Bx / 6) / ((1 - Bx) / 8), "f", args=(6, 8)).statistic)
pr = np.linspace(0.01, 0.99, 99)
post = stats.binom.pmf(7, 20, pr) * stats.beta.pdf(pr, 2, 3)
check(T, "共役事前: 事後∝Beta(a+x,b+n-x)（x=7,n=20,a=2,b=3）", np.allclose(post / post.sum(), stats.beta.pdf(pr, 9, 16) / stats.beta.pdf(pr, 9, 16).sum()), "比例")
check(T, "例1 E=0.2, V≈0.01455, SD≈0.1206, 最頻値0.125", close(stats.beta(2, 8).mean(), 0.2, 1e-12) and close(stats.beta(2, 8).var(), 0.01455, 5e-6) and close(stats.beta(2, 8).std(), 0.1206) and close(1 / 8, 0.125, 1e-12), stats.beta(2, 8).var())
check(T, "例1 事後 Beta(5,25): E≈0.1667, SD≈0.0669", close(stats.beta(5, 25).mean(), 0.1667) and close(stats.beta(5, 25).std(), 0.0669), stats.beta(5, 25).std())
terms = [stats.binom.pmf(j, 5, 0.3) for j in (3, 4, 5)]
check(T, "例2 0.1323+0.0284+0.0024≈0.1631 = Beta(3,3).cdf(0.3)", close(terms[0], 0.1323) and close(terms[1], 0.0284, 5e-5 + 1e-12) and close(terms[2], 0.0024, 5e-5) and close(sum(terms), 0.1631) and close(stats.beta(3, 3).cdf(0.3), sum(terms), 1e-12), (terms, stats.beta(3, 3).cdf(0.3)))

# ================= lognormal-distribution =================
T = "lognormal-distribution"
Yl = sp.Symbol("Y", real=True)
ky = sp.Symbol("k", positive=True)
check(T, "E[X^k]=M_Y(k)=e^{kμ+k²σ²/2}", zero(MN.subs(t, ky) - sp.exp(ky * mu + ky**2 * sig**2 / 2)), "sympy")
check(T, "V=e^{2μ+σ²}(e^{σ²}-1)", zero(sp.exp(2 * mu + 2 * sig**2) - sp.exp(2 * mu + sig**2) - sp.exp(2 * mu + sig**2) * (sp.exp(sig**2) - 1)), "sympy")
ln = stats.lognorm(0.7, scale=exp(0.4))
check(T, "scipy で E, V, E[X^3]（μ=0.4,σ=0.7）", close(ln.mean(), exp(0.4 + 0.245), 1e-10) and close(ln.var(), exp(0.8 + 0.49) * (exp(0.49) - 1), 1e-9) and close(ln.moment(3), exp(1.2 + 9 * 0.49 / 2), 1e-7), ln.mean())
fl = sp.exp(-(sp.log(xp) - mu) ** 2 / (2 * sig**2)) / (sp.sqrt(2 * sp.pi) * sig * xp)
model = sp.solve(sp.diff(sp.log(fl), xp), xp)
check(T, "最頻値 e^{μ-σ²}, 中央値 e^μ", len(model) == 1 and zero(model[0] - sp.exp(mu - sig**2)) and close(ln.median(), exp(0.4), 1e-12), model)
check(T, "MGF は t>0 で発散（積分の被積分関数が x→∞ で増大）", float(sp.limit(sp.log(sp.exp(0.1 * xp) * fl.subs({mu: 0, sig: 1})), xp, sp.oo)) == float("inf"), "limit=∞")
L1 = rng.lognormal(0.3, 0.5, N); L2 = rng.lognormal(-1, 0.4, N)
check(T, "積の再生性 LN(-0.7, 0.41)（KS<0.005）", ks(L1 * L2, "lognorm", args=(sqrt(0.41), 0, exp(-0.7))).statistic < 0.005, ks(L1 * L2, "lognorm", args=(sqrt(0.41), 0, exp(-0.7))).statistic)
check(T, "X^a ~ LN(aμ, a²σ²)（a=2, KS<0.005）", ks(L1**2, "lognorm", args=(1.0, 0, exp(0.6))).statistic < 0.005, ks(L1**2, "lognorm", args=(1.0, 0, exp(0.6))).statistic)
check(T, "和は対数正規でない（log(L1+L1') の歪度が0から有意にずれる）", abs(stats.skew(np.log(L1 + rng.lognormal(0.3, 1.5, N)))) > 0.05, stats.skew(np.log(L1 + rng.lognormal(0.3, 1.5, N))))
le = stats.lognorm(0.5, scale=400)
check(T, "例 平均≈453.3, 最頻値≈311.5, SD≈241.6", close(le.mean(), 453.3, 0.05) and close(400 * exp(-0.25), 311.5, 0.05) and close(le.std(), 241.6, 0.05), (le.mean(), 400 * exp(-0.25), le.std()))
check(T, "例 z=1.3863, P(X>800)≈0.0828", close(log(2) / 0.5, 1.3863) and close(le.sf(800), 0.0828), le.sf(800))
check(T, "例 P(X>平均)=P(Z>0.25)≈0.401", close(le.sf(le.mean()), 0.401, 5e-4) and close(stats.norm.sf(0.25), 0.401, 5e-4), le.sf(le.mean()))

# ================= cauchy-distribution =================
T = "cauchy-distribution"
fc = 1 / (sp.pi * (1 + x**2))
check(T, "標準形の積分=1, F(x)=1/2+arctan(x)/π", zero(sp.integrate(fc, (x, -sp.oo, sp.oo)) - 1) and zero(sp.integrate(fc, (x, -sp.oo, xp)) - (sp.Rational(1, 2) + sp.atan(xp) / sp.pi)), "sympy")
check(T, "∫_0^∞ x/(π(1+x²))dx = ∞", sp.integrate(xp / (sp.pi * (1 + xp**2)), (xp, 0, sp.oo)) == sp.oo, "oo")
phc = complex(integrate.quad(lambda y: stats.cauchy.pdf(y, 1, 2), -np.inf, np.inf, weight="cos", wvar=0.8)[0] if False else 0)
# 特性関数: 余弦部・正弦部を QAWF で
cpart = integrate.quad(lambda y: stats.cauchy.pdf(y), 0, np.inf, weight="cos", wvar=0.8)[0] * 2
check(T, "特性関数 e^{-|t|}（標準, t=0.8, QAWF）", close(cpart, exp(-0.8), 1e-7), cpart)
re_ = integrate.quad(lambda y: stats.cauchy.pdf(y, 1, 2) * np.cos(0.5 * y), -2000, 2000, limit=2000)[0]
im_ = integrate.quad(lambda y: stats.cauchy.pdf(y, 1, 2) * np.sin(0.5 * y), -2000, 2000, limit=2000)[0]
target = np.exp(1j * 0.5 * 1 - 2 * 0.5)
check(T, "特性関数 exp(iμt-γ|t|)（μ=1,γ=2,t=0.5, 有限区間 許容1e-3）", abs(complex(re_, im_) - target) < 1e-3, complex(re_, im_))
check(T, "四分位点 μ±γ", close(stats.cauchy(1, 2).ppf(0.25), -1, 1e-12) and close(stats.cauchy(1, 2).ppf(0.75), 3, 1e-12), "scipy")
check(T, "φ(t/n)^n = e^{-|t|}", zero(sp.exp(-sp.Abs(t) / 7) ** 7 - sp.exp(-sp.Abs(t))), "sympy")
Cm = rng.standard_cauchy(size=(20000, 100)).mean(axis=1)
check(T, "100個の平均も標準コーシー（KS<0.015）, P(|X̄|>1)≈0.5（許容0.015）", ks(Cm, "cauchy").statistic < 0.015 and abs(np.mean(np.abs(Cm) > 1) - 0.5) < 0.015, np.mean(np.abs(Cm) > 1))
Z1 = rng.normal(size=N); Z2 = rng.normal(size=N)
check(T, "Z1/Z2 ~ 標準コーシー（KS<0.005）", ks(Z1 / Z2, "cauchy").statistic < 0.005, ks(Z1 / Z2, "cauchy").statistic)
check(T, "t(1) = 標準コーシー", np.allclose(stats.t(1).pdf(np.linspace(-5, 5, 41)), stats.cauchy.pdf(np.linspace(-5, 5, 41))), "pdf")
check(T, "tan U ~ 標準コーシー（KS<0.005）", ks(np.tan(rng.uniform(-pi / 2, pi / 2, N)), "cauchy").statistic < 0.005, "KS")
check(T, "例1 P(X>3)≈0.1024, P(Z>3)≈0.00135, 約76倍", close(0.5 - atan(3) / pi, 0.1024) and close(stats.norm.sf(3), 0.00135, 5e-6) and round((0.5 - atan(3) / pi) / stats.norm.sf(3)) == 76, (0.5 - atan(3) / pi) / stats.norm.sf(3))
check(T, "例1 P(|X|>1)=0.5", close(1 - 2 / pi * atan(1), 0.5, 1e-12), 1 - 2 / pi * atan(1))
check(T, "例2 P(|Z|>10)≈0", 2 * stats.norm.sf(10) < 1e-20, 2 * stats.norm.sf(10))
cross = optimize.brentq(lambda y: stats.cauchy.pdf(y) - stats.norm.pdf(y), 1, 3)
check(T, "正規との密度の交点 |x|≈1.85", close(cross, 1.85, 0.005) and stats.cauchy.pdf(0) < stats.norm.pdf(0), cross)

# ================= chi-squared-distribution =================
T = "chi-squared-distribution"
fch = xp ** (k / 2 - 1) * sp.exp(-xp / 2) / (2 ** (k / 2) * sp.gamma(k / 2))
check(T, "密度 = Ga(k/2,2)（sympy 一致）", zero(fch - fg.subs({alpha: k / 2, beta: 2})), "sympy")
E1 = sp.integrate(sp.exp(t * z**2) * phi, (z, -sp.oo, sp.oo), conds="none")
check(T, "E[e^{tZ²}]=(1-2t)^{-1/2}（t=0.2,-1 で数値）", all(close(integrate.quad(lambda y: np.exp(-(1 - 2 * tv) * y * y / 2) / sqrt(2 * pi), -np.inf, np.inf)[0], (1 - 2 * tv) ** -0.5, 1e-8) for tv in (0.2, -1, 0.45)), "quad")
Mc = (1 - 2 * t) ** (-k / 2)
m1, v = moments_from_mgf(Mc)
check(T, "E=k, V=2k（MGF）", zero(m1 - k) and zero(v - 2 * k), (m1, v))
Kc = -k / 2 * sp.log(1 - 2 * t)
check(T, "K'(0)=k, K''(0)=2k", zero(sp.diff(Kc, t).subs(t, 0) - k) and zero(sp.diff(Kc, t, 2).subs(t, 0) - 2 * k), "sympy")
check(T, "特性関数 (1-2it)^{-k/2}（k=3,t=0.4 数値）", abs(complex(integrate.quad(lambda y: np.cos(0.4 * y) * stats.chi2.pdf(y, 3), 0, np.inf, limit=200)[0], integrate.quad(lambda y: np.sin(0.4 * y) * stats.chi2.pdf(y, 3), 0, np.inf, limit=200)[0]) - (1 - 0.8j) ** -1.5) < 1e-6, (1 - 0.8j) ** -1.5)
check(T, "最頻値 max(k-2,0)（k=1,2,5,10）", all(close(optimize.minimize_scalar(lambda y: -stats.chi2.pdf(y, kk), bounds=(1e-9, 30), method="bounded").x, max(kk - 2, 0), 1e-3) for kk in (2, 5, 10)) and stats.chi2.pdf(0.001, 1) > stats.chi2.pdf(0.01, 1), "数値最大化")
Zk = rng.normal(size=(N, 3))
check(T, "Z1²+Z2²+Z3² ~ χ²(3)（KS<0.005）", ks((Zk**2).sum(axis=1), "chi2", args=(3,)).statistic < 0.005, ks((Zk**2).sum(axis=1), "chi2", args=(3,)).statistic)
W1 = rng.chisquare(2, N) + rng.chisquare(5, N)
check(T, "再生性 χ²(2)+χ²(5)=χ²(7)（KS<0.005）", ks(W1, "chi2", args=(7,)).statistic < 0.005, ks(W1, "chi2", args=(7,)).statistic)
Xs = rng.normal(3, 2, size=(100000, 10))
check(T, "μ既知 Σ(Xi-μ)²/σ² ~ χ²(n)（n=10, KS<0.006）", ks(((Xs - 3) ** 2).sum(axis=1) / 4, "chi2", args=(10,)).statistic < 0.006, "KS")
check(T, "正規近似 (W-k)/√(2k)（k=200, 上側5%点の差<0.1）", abs((stats.chi2.isf(0.05, 200) - 200) / 20 - 1.645) < 0.1, (stats.chi2.isf(0.05, 200) - 200) / 20)
check(T, "例1 E=3, V=6, 上側5%点7.815, √7.815≈2.80", close(stats.chi2.isf(0.05, 3), 7.815, 5e-4) and close(sqrt(7.815), 2.80, 5e-3), stats.chi2.isf(0.05, 3))
check(T, "例2 9×6/4=13.5, P(χ²(9)>13.5)≈0.1413", close(stats.chi2.sf(13.5, 9), 0.1413), stats.chi2.sf(13.5, 9))
lo, hi = stats.chi2.ppf(0.025, 9), stats.chi2.ppf(0.975, 9)
check(T, "例3 分位点 2.700, 19.023, 区間 [2.366, 16.664]", close(lo, 2.700, 5e-4) and close(hi, 19.023, 5e-4) and close(45 / hi, 2.366, 5e-4) and close(45 / lo, 16.664, 5e-4), (45 / hi, 45 / lo))
ratios = [(stats.chi2.isf(0.05, kk) - kk) / sqrt(2 * kk) for kk in range(3, 31)]
check(T, "上側5%点 ≈ k + (おおよそ1.8〜2.0)√(2k)（k=3..30, 丸めで1.75以上を許容）", min(ratios) >= 1.75 and max(ratios) <= 2.0, (min(ratios), max(ratios)))

# ================= t-distribution =================
T = "t-distribution"
nu = 7
fpdf = lambda y, nv: G((nv + 1) / 2) / (sqrt(nv * pi) * G(nv / 2)) * (1 + y * y / nv) ** (-(nv + 1) / 2)
check(T, "密度の式 = scipy t.pdf（ν=3,7）", all(np.allclose([fpdf(y, nv) for y in np.linspace(-4, 4, 17)], stats.t.pdf(np.linspace(-4, 4, 17), nv)) for nv in (3, 7)), "pdf")
check(T, "分散 ν/(ν-2)（ν=5,10）", close(stats.t(5).var(), 5 / 3, 1e-10) and close(stats.t(10).var(), 1.25, 1e-10), stats.t(5).var())
check(T, "分散 1<ν≤2 で無限大, ν=1 で平均なし（∫|t|f は発散）", np.isinf(stats.t(1.5).var()) and np.isinf(stats.t(2).var()) and sp.integrate(xp / (sp.pi * (1 + xp**2)), (xp, 0, sp.oo)) == sp.oo, (stats.t(1.5).var(), stats.t(2).var()))
check(T, "r 次モーメントは r<ν で存在（ν=4: E|T|^3 有限, E T^4 発散）", np.isfinite(stats.t(4).moment(3)) and np.isinf(stats.t(4).moment(4)), (stats.t(4).moment(3), stats.t(4).moment(4)))
cpt = integrate.quad(lambda y: stats.t.pdf(y, 1), 0, np.inf, weight="cos", wvar=1.2)[0] * 2
check(T, "ν=1 の特性関数 e^{-|t|}（QAWF）", close(cpt, exp(-1.2), 1e-7), cpt)
Zt = rng.normal(size=N); Wt = rng.chisquare(5, N)
check(T, "Z/√(W/ν) ~ t(ν)（ν=5, KS<0.005）", ks(Zt / np.sqrt(Wt / 5), "t", args=(5,)).statistic < 0.005, ks(Zt / np.sqrt(Wt / 5), "t", args=(5,)).statistic)
Xs = rng.normal(10, 3, size=(100000, 8))
Tst = (Xs.mean(axis=1) - 10) / (Xs.std(axis=1, ddof=1) / sqrt(8))
check(T, "正規標本の (X̄-μ)/(S/√n) ~ t(n-1)（n=8, KS<0.006）", ks(Tst, "t", args=(7,)).statistic < 0.006, ks(Tst, "t", args=(7,)).statistic)
check(T, "T² ~ F(1,ν)（ν=5, KS<0.005）", ks((Zt / np.sqrt(Wt / 5)) ** 2, "f", args=(1, 5)).statistic < 0.005, "KS")
check(T, "ν→∞ で N(0,1)（ν=10^6 の上側2.5%点）", close(stats.t.isf(0.025, 1e6), 1.96, 5e-3), stats.t.isf(0.025, 1e6))
se = 3.1 / sqrt(10); tq = stats.t.isf(0.025, 9)
check(T, "例1 SE≈0.9803, t_0.025(9)=2.262, 半幅≈2.218, [50.082, 54.518]", close(se, 0.9803) and close(tq, 2.262, 5e-4) and close(tq * se, 2.218, 5e-4) and close(52.3 - tq * se, 50.082, 5e-4) and close(52.3 + tq * se, 54.518, 5e-4), (tq * se, 52.3 - tq * se, 52.3 + tq * se))
check(T, "例1 1.96 を使うと ±1.921", close(1.96 * se, 1.921, 5e-4), 1.96 * se)
check(T, "例2 P(|T|>2)(ν=5)≈0.1019, P(|Z|>2)≈0.0455, 2倍以上", close(2 * stats.t.sf(2, 5), 0.1019) and close(2 * stats.norm.sf(2), 0.0455) and 2 * stats.t.sf(2, 5) > 2 * 2 * stats.norm.sf(2), 2 * stats.t.sf(2, 5))
tab = {1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 30: 2.042, 120: 1.980}
check(T, "例2 上側2.5%点の表", all(close(stats.t.isf(0.025, d), v_, 5e-4) for d, v_ in tab.items()) and close(stats.norm.isf(0.025), 1.960, 5e-4), {d: round(stats.t.isf(0.025, d), 3) for d in tab})

# ================= f-distribution =================
T = "f-distribution"
mF, nF = 6, 11
fF = lambda y: (mF / nF) ** (mF / 2) * y ** (mF / 2 - 1) * (1 + mF / nF * y) ** (-(mF + nF) / 2) / (G(mF / 2) * G(nF / 2) / G((mF + nF) / 2))
check(T, "密度の式 = scipy f.pdf（m=6,n=11）", np.allclose([fF(y) for y in np.linspace(0.1, 5, 20)], stats.f.pdf(np.linspace(0.1, 5, 20), mF, nF)), "pdf")
check(T, "E=n/(n-2), V=2n²(m+n-2)/(m(n-2)²(n-4))（m=6,n=11 / m=3,n=9）", all(close(stats.f(m_, n_).mean(), n_ / (n_ - 2), 1e-10) and close(stats.f(m_, n_).var(), 2 * n_**2 * (m_ + n_ - 2) / (m_ * (n_ - 2) ** 2 * (n_ - 4)), 1e-9) for m_, n_ in [(6, 11), (3, 9)]), stats.f(6, 11).var())
nn = sp.Symbol("n", positive=True)
EinvV = sp.integrate(fch.subs(k, nn) / xp, (xp, 0, sp.oo), conds="none")
check(T, "E[1/V]=1/(n-2)（V~χ²(n)）", zero(sp.gammasimp(EinvV) - 1 / (nn - 2)), sp.gammasimp(EinvV))
U6 = rng.chisquare(6, N); V11 = rng.chisquare(11, N)
Fs = (U6 / 6) / (V11 / 11)
check(T, "(U/m)/(V/n) ~ F(m,n)（KS<0.005）, 1/F ~ F(n,m)", ks(Fs, "f", args=(6, 11)).statistic < 0.005 and ks(1 / Fs, "f", args=(11, 6)).statistic < 0.005, ks(Fs, "f", args=(6, 11)).statistic)
check(T, "F_{1-α}(m,n)=1/F_α(n,m)", close(stats.f.ppf(0.05, 6, 11), 1 / stats.f.isf(0.05, 11, 6), 1e-10), stats.f.ppf(0.05, 6, 11))
A = rng.normal(0, 2, size=(100000, 10)); Bm = rng.normal(5, 3, size=(100000, 12))
piv = (A.var(axis=1, ddof=1) / 4) / (Bm.var(axis=1, ddof=1) / 9)
check(T, "分散比のピボット ~ F(n1-1,n2-1)（KS<0.006）", ks(piv, "f", args=(9, 11)).statistic < 0.006, ks(piv, "f", args=(9, 11)).statistic)
yF = 6 * Fs / 11
check(T, "mX/n/(1+mX/n) ~ Beta(m/2,n/2)（KS<0.005）", ks(yF / (1 + yF), "beta", args=(3, 5.5)).statistic < 0.005, "KS")
check(T, "n→∞ で mF→χ²(m)（n=10^6 の上側5%点）", close(6 * stats.f.isf(0.05, 6, 1e6), stats.chi2.isf(0.05, 6), 5e-3), 6 * stats.f.isf(0.05, 6, 1e6))
Fobs = 8.5 / 3.2
pF = 2 * min(stats.f.sf(Fobs, 9, 11), stats.f.cdf(Fobs, 9, 11))
check(T, "例1 F≈2.656, F_0.025(9,11)=3.588, 下側≈0.2556, 両側p≈0.130", close(Fobs, 2.656, 5e-4) and close(stats.f.isf(0.025, 9, 11), 3.588, 5e-4) and close(1 / stats.f.isf(0.025, 11, 9), 0.2556) and close(stats.f.ppf(0.025, 9, 11), 0.2556) and close(pF, 0.130, 5e-4), (stats.f.isf(0.025, 9, 11), stats.f.ppf(0.025, 9, 11), pF))
check(T, "例2 F_0.05(2,27)≈3.354, E=27/25=1.08", close(stats.f.isf(0.05, 2, 27), 3.354, 5e-4) and close(stats.f(2, 27).mean(), 1.08, 1e-12), stats.f.isf(0.05, 2, 27))
check(T, "F_0.025(11,9)≈3.912", close(stats.f.isf(0.025, 11, 9), 3.912, 5e-4), stats.f.isf(0.025, 11, 9))

# ================= t-statistic =================
T = "t-statistic"
t1 = 2.3 / (3.1 / sqrt(10))
check(T, "例1 t≈2.346, SE 0.9803, 両側p≈0.0436, t_0.025(9)=2.262<t", close(t1, 2.346, 5e-4) and close(2 * stats.t.sf(t1, 9), 0.0436) and t1 > stats.t.isf(0.025, 9), (t1, 2 * stats.t.sf(t1, 9)))
r_ = stats.ttest_1samp(np.array([-1.0, 1.0]) * 0 + 0, 0) if False else None
sp2 = (7 * 9 + 9 * 12.25) / 16
se2 = sqrt(sp2 * (1 / 8 + 1 / 10)); t2 = 3.3 / se2
res = stats.ttest_ind_from_stats(24.5, 3.0, 8, 21.2, 3.5, 10, equal_var=True)
check(T, "例2 s_p²≈10.828, SE≈1.5609, t≈2.114, p≈0.0505, t_0.025(16)=2.120", close(sp2, 10.828, 5e-4) and close(se2, 1.5609) and close(t2, 2.114, 5e-4) and close(res.pvalue, 0.0505) and close(res.statistic, t2, 1e-10) and close(stats.t.isf(0.025, 16), 2.120, 5e-4), (sp2, se2, t2, res.pvalue))
check(T, "グラフ 自由度9で cdf(2.346)≈0.978, 片側≈0.022", close(stats.t.cdf(t1, 9), 0.978, 5e-4) and close(stats.t.sf(t1, 9), 0.022, 5e-4), stats.t.cdf(t1, 9))
X_ = rng.normal(0, 1, size=(60000, 8)); Y_ = rng.normal(0, 1, size=(60000, 10))
Sp = np.sqrt((7 * X_.var(axis=1, ddof=1) + 9 * Y_.var(axis=1, ddof=1)) / 16)
T2 = (X_.mean(axis=1) - Y_.mean(axis=1)) / (Sp * sqrt(1 / 8 + 1 / 10))
check(T, "等分散2標本 t ~ t(n1+n2-2)（シミュ KS<0.008）", ks(T2, "t", args=(16,)).statistic < 0.008, ks(T2, "t", args=(16,)).statistic)
# 回帰係数の t 検定: 自由度 n-p-1、t² = 1自由度の F
nr, pr_ = 30, 2
Xr = np.column_stack([np.ones(nr), rng.normal(size=(nr, pr_))])
yr = Xr @ np.array([1.0, 0.5, 0.0]) + rng.normal(size=nr)
bh, rss = np.linalg.lstsq(Xr, yr, rcond=None)[:2]
s2r = rss[0] / (nr - pr_ - 1)
seb = np.sqrt(s2r * np.diag(np.linalg.inv(Xr.T @ Xr)))
tb = bh[2] / seb[2]
rss0 = np.sum((yr - Xr[:, :2] @ np.linalg.lstsq(Xr[:, :2], yr, rcond=None)[0]) ** 2)
Fpart = (rss0 - rss[0]) / s2r
check(T, "回帰係数: t² = 部分 F（1自由度、自由度 n-p-1）", close(tb**2, Fpart, 1e-8), (tb**2, Fpart))
check(T, "|t|>t_{α/2} ⇔ 区間が θ0 を含まない（例1）", (52.3 - stats.t.isf(0.025, 9) * se > 50) == (t1 > stats.t.isf(0.025, 9)), 52.3 - stats.t.isf(0.025, 9) * se)

# ================= normal-mixture-distribution =================
T = "normal-mixture-distribution"
w, m1_, m2_ = sp.symbols("w m1 m2", real=True)
M2 = w * sp.exp(m1_ * t + s1**2 * t**2 / 2) + (1 - w) * sp.exp(m2_ * t + s2**2 * t**2 / 2)
e1, vv = moments_from_mgf(M2)
check(T, "MGF から E=wμ1+(1-w)μ2", zero(e1 - (w * m1_ + (1 - w) * m2_)), e1)
check(T, "MGF から V=wσ1²+(1-w)σ2²+w(1-w)(μ1-μ2)²", zero(vv - (w * s1**2 + (1 - w) * s2**2 + w * (1 - w) * (m1_ - m2_) ** 2)), "sympy")
check(T, "全分散の公式 E[V|C]+V[E|C] の第2項", zero(w * m1_**2 + (1 - w) * m2_**2 - (w * m1_ + (1 - w) * m2_) ** 2 - w * (1 - w) * (m1_ - m2_) ** 2), "sympy")
mixpdf = lambda y, ww=0.5, a1=158, b1=5.5, a2=171, b2=6: ww * stats.norm.pdf(y, a1, b1) + (1 - ww) * stats.norm.pdf(y, a2, b2)
cf = complex(integrate.quad(lambda y: np.cos(0.05 * y) * mixpdf(y), 100, 240, limit=200)[0], integrate.quad(lambda y: np.sin(0.05 * y) * mixpdf(y), 100, 240, limit=200)[0])
cf_cl = 0.5 * np.exp(1j * 158 * 0.05 - 30.25 * 0.0025 / 2) + 0.5 * np.exp(1j * 171 * 0.05 - 36 * 0.0025 / 2)
check(T, "特性関数（例のパラメータ, t=0.05 数値積分）", abs(cf - cf_cl) < 1e-8, cf_cl)


def n_modes(dmu, sg=1.0, ww=0.5):
    ys = np.linspace(-6, dmu + 6, 20001)
    d = mixpdf(ys, ww, 0, sg, dmu, sg)
    return int(np.sum((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])))


check(T, "等重み等分散: 差>2σ で2峰, ≤2σ で1峰（差 1.9,2.0,2.1）", n_modes(1.9) == 1 and n_modes(2.0) == 1 and n_modes(2.1) == 2, (n_modes(1.9), n_modes(2.0), n_modes(2.1)))
check(T, "平均同じ・分散違いの混合は尖度>3（0.5N(0,1)+0.5N(0,9)）", 0.5 * 3 * 1 + 0.5 * 3 * 81 > 3 * (0.5 * 1 + 0.5 * 9) ** 2, (0.5 * 3 + 0.5 * 243) / 25)
EXm = 0.5 * 158 + 0.5 * 171
VXm = 0.5 * 30.25 + 0.5 * 36 + 0.25 * 169
check(T, "例 E=164.5, V=33.125+42.25=75.375, SD≈8.682", EXm == 164.5 and close(VXm, 75.375, 1e-12) and close(0.5 * 30.25 + 0.5 * 36, 33.125, 1e-12) and close(sqrt(VXm), 8.682, 5e-4), sqrt(VXm))
check(T, "例 V を数値積分で確認", close(integrate.quad(lambda y: (y - 164.5) ** 2 * mixpdf(y), 100, 240)[0], 75.375, 1e-6), "quad")
z1, z2 = 17 / 5.5, 4 / 6
pm = 0.5 * stats.norm.sf(z1) + 0.5 * stats.norm.sf(z2)
check(T, "例 z=3.0909, 0.6667, P(X>175)≈0.1267", close(z1, 3.0909) and close(z2, 0.6667) and close(pm, 0.1267), pm)
post = 0.5 * stats.norm.pdf(165, 158, 5.5) / mixpdf(165)
check(T, "例 P(女性|165)≈0.4445", close(post, 0.4445), post)
ys = np.linspace(140, 190, 500001)
dd = mixpdf(ys)
im = np.where((dd[1:-1] > dd[:-2]) & (dd[1:-1] > dd[2:]))[0] + 1
iv = np.where((dd[1:-1] < dd[:-2]) & (dd[1:-1] < dd[2:]))[0] + 1
check(T, "グラフ 山は159付近と169付近の2つ（各±1）, 谷は間に1つ", len(im) == 2 and abs(ys[im[0]] - 159) <= 1 and abs(ys[im[1]] - 169) <= 1 and len(iv) == 1 and ys[im[0]] < ys[iv[0]] < ys[im[1]], (ys[im].round(2).tolist(), ys[iv].round(2).tolist(), (dd[im] / dd[iv[0]]).round(4).tolist()))
Cl = rng.uniform(size=N) < 0.5
Xm = np.where(Cl, rng.normal(158, 5.5, N), rng.normal(171, 6, N))
check(T, "シミュ: 平均・分散・P(X>175)（許容 0.07=標準誤差0.0137の約5倍, 0.5, 0.003）", abs(Xm.mean() - 164.5) < 0.07 and abs(Xm.var() - 75.375) < 0.5 and abs(np.mean(Xm > 175) - pm) < 0.003, (Xm.mean(), Xm.var(), np.mean(Xm > 175)))
check(T, "wX1+(1-w)X2 の分散 w²σ1²+(1-w)²σ2²（正規）", close(np.var(0.5 * rng.normal(158, 5.5, N) + 0.5 * rng.normal(171, 6, N)), 0.25 * 30.25 + 0.25 * 36, 0.1), 0.25 * 30.25 + 0.25 * 36)

# ================= bivariate-normal-distribution =================
T = "bivariate-normal-distribution"
uu, vv_, rho = sp.symbols("u v rho", real=True)
check(T, "平方完成 (u²-2ρuv+v²)/(1-ρ²)=u²+(v-ρu)²/(1-ρ²)", zero((uu**2 - 2 * rho * uu * vv_ + vv_**2) / (1 - rho**2) - uu**2 - (vv_ - rho * uu) ** 2 / (1 - rho**2)), "sympy")
mx, my_, sx, sy, xx, yy = 1.0, -2.0, 2.0, 3.0, 0.7, 1.1
rv = 0.6
Sig = np.array([[sx**2, rv * sx * sy], [rv * sx * sy, sy**2]])
def fbiv(x_, y_):
    u_ = (x_ - mx) / sx; v_ = (y_ - my_) / sy
    return exp(-(u_**2 - 2 * rv * u_ * v_ + v_**2) / (2 * (1 - rv**2))) / (2 * pi * sx * sy * sqrt(1 - rv**2))
check(T, "密度の式 = scipy multivariate_normal", close(fbiv(xx, yy), stats.multivariate_normal([mx, my_], Sig).pdf([xx, yy]), 1e-12), fbiv(xx, yy))
check(T, "f = f_X × N(条件付き平均, σY²(1-ρ²))", close(fbiv(xx, yy), stats.norm.pdf(xx, mx, sx) * stats.norm.pdf(yy, my_ + rv * sy / sx * (xx - mx), sy * sqrt(1 - rv**2)), 1e-12), "数値")
check(T, "ρ=0 で f=f_X f_Y", close(stats.multivariate_normal([0, 0], [[4, 0], [0, 9]]).pdf([0.3, -1]), stats.norm.pdf(0.3, 0, 2) * stats.norm.pdf(-1, 0, 3), 1e-14), "数値")
t1_, t2_ = sp.symbols("t1 t2", real=True)
sxs, sys_ = sp.symbols("sx sy", positive=True)
mxs, mys = sp.symbols("mx my", real=True)
Mb2 = sp.exp(mxs * t1_ + mys * t2_ + (sxs**2 * t1_**2 + 2 * rho * sxs * sys_ * t1_ * t2_ + sys_**2 * t2_**2) / 2)
cov = sp.diff(Mb2, t1_, t2_).subs({t1_: 0, t2_: 0}) - mxs * mys
check(T, "MGF から Cov=ρσXσY, E=μ", zero(cov - rho * sxs * sys_) and zero(sp.diff(Mb2, t1_).subs({t1_: 0, t2_: 0}) - mxs), sp.simplify(cov))
aa, bb = sp.symbols("a b", real=True)
Mlin = Mb2.subs({t1_: aa * t, t2_: bb * t})
check(T, "線形結合 aX+bY の MGF = 正規 MGF（平均・分散）", zero(sp.log(Mlin) - ((aa * mxs + bb * mys) * t + (aa**2 * sxs**2 + 2 * aa * bb * rho * sxs * sys_ + bb**2 * sys_**2) * t**2 / 2)) or zero(sp.expand(sp.expand_log(sp.log(Mlin), force=True)) - sp.expand((aa * mxs + bb * mys) * t + (aa**2 * sxs**2 + 2 * aa * bb * rho * sxs * sys_ + bb**2 * sys_**2) * t**2 / 2)), "sympy")
XY = rng.multivariate_normal([170, 170], [[36, 18], [18, 36]], N)
sel = XY[np.abs(XY[:, 0] - 182) < 0.5, 1]
check(T, "条件付き分布（シミュ, X≈182: 平均176±0.3, SD5.196±0.25）", abs(sel.mean() - 176) < 0.3 and abs(sel.std() - 5.196) < 0.25, (sel.mean(), sel.std(), len(sel)))
check(T, "例1 E=176, SD≈5.196, z≈0.7698, P≈0.2207", close(170 + 0.5 * 12, 176, 1e-12) and close(6 * sqrt(0.75), 5.196, 5e-4) and close(4 / (6 * sqrt(0.75)), 0.7698) and close(stats.norm.sf(4 / (6 * sqrt(0.75))), 0.2207), stats.norm.sf(4 / (6 * sqrt(0.75))))
check(T, "例1 P(Y>180)=1-Φ(10/6)≈0.0478", close(stats.norm.sf(10 / 6), 0.0478), stats.norm.sf(10 / 6))
check(T, "例2 V[Y-X]=36, P(Y-X≥5)≈0.2023", close(36 + 36 - 36, 36, 0) and close(stats.norm.sf(5 / 6), 0.2023) and abs(np.mean(XY[:, 1] - XY[:, 0] >= 5) - 0.2023) < 0.003, (stats.norm.sf(5 / 6), np.mean(XY[:, 1] - XY[:, 0] >= 5)))

# ================= multivariate-normal-distribution =================
T = "multivariate-normal-distribution"
S3 = np.array([[400, 60, 150], [60, 100, 30], [150, 30, 225]], float)
mu3 = np.array([5, 3, 4.0]); w3 = np.array([0.4, 0.3, 0.3])
sd3 = np.sqrt(np.diag(S3)); cor = S3 / np.outer(sd3, sd3)
check(T, "例1 SD 20,10,15・相関 0.3,0.5,0.2", np.allclose(sd3, [20, 10, 15]) and np.allclose([cor[0, 1], cor[0, 2], cor[1, 2]], [0.3, 0.5, 0.2]) and np.all(np.linalg.eigvalsh(S3) > 0), cor.round(3).tolist())
VR = w3 @ S3 @ w3
check(T, "例1 E[R]=4.1, V=64+9+20.25+2(7.2+18+2.7)=149.05", close(w3 @ mu3, 4.1, 1e-12) and close(VR, 149.05, 1e-9) and close(64 + 9 + 20.25 + 2 * (7.2 + 18 + 2.7), 149.05, 1e-9) and close(0.4 * 0.3 * 60, 7.2, 1e-12) and close(0.4 * 0.3 * 150, 18, 1e-12) and close(0.3 * 0.3 * 30, 2.7, 1e-12), VR)
check(T, "例1 SD≈12.209, P(R<0)≈0.3685", close(sqrt(VR), 12.209, 5e-4) and close(Phi(-4.1 / sqrt(VR)), 0.3685), Phi(-4.1 / sqrt(VR)))
X3 = rng.multivariate_normal(mu3, S3, N)
R = X3 @ w3
check(T, "例1 シミュ P(R<0)（許容0.003）", abs(np.mean(R < 0) - 0.3685) < 0.003, np.mean(R < 0))
S2m = np.array([[1, 0.5], [0.5, 1]])
Si = np.linalg.inv(S2m)
check(T, "例2 Σ^{-1}=(1/0.75)[[1,-0.5],[-0.5,1]]", np.allclose(Si, np.array([[1, -0.5], [-0.5, 1]]) / 0.75), Si.round(4).tolist())
D1 = np.array([1.5, 1.5]) @ Si @ np.array([1.5, 1.5]); D2 = np.array([1.5, -1.5]) @ Si @ np.array([1.5, -1.5])
check(T, "例2 D²=3, 9, ユークリッド4.5", close(D1, 3, 1e-12) and close(D2, 9, 1e-12), (D1, D2))
check(T, "例2 P(D²>3)≈0.2231, P(D²>9)≈0.0111, 5.991", close(stats.chi2.sf(3, 2), 0.2231) and close(stats.chi2.sf(9, 2), 0.0111) and close(exp(-4.5), 0.0111) and close(stats.chi2.isf(0.05, 2), 5.991, 5e-4) and D1 < 5.991 < D2, (stats.chi2.sf(3, 2), stats.chi2.sf(9, 2)))
D2s = np.einsum("ij,jk,ik->i", X3 - mu3, np.linalg.inv(S3), X3 - mu3)
check(T, "2次形式 ~ χ²(p)（p=3, KS<0.005）", ks(D2s, "chi2", args=(3,)).statistic < 0.005, ks(D2s, "chi2", args=(3,)).statistic)
Am = np.array([[1, -1, 0], [0.5, 0.5, 1]]); bvec = np.array([1, 2.0])
Y3 = X3 @ Am.T + bvec
check(T, "線形変換 AX+b の平均・共分散（シミュ 相対誤差<1%）", np.allclose(Y3.mean(axis=0), Am @ mu3 + bvec, atol=0.2) and np.allclose(np.cov(Y3.T), Am @ S3 @ Am.T, rtol=0.01, atol=1), np.cov(Y3.T).round(1).tolist())
# 条件付き分布: X1 | (X2,X3)
S11 = S3[:1, :1]; S12 = S3[:1, 1:]; S22 = S3[1:, 1:]
x2v = np.array([5.0, 10.0])
cm = mu3[0] + (S12 @ np.linalg.solve(S22, x2v - mu3[1:]))[0]
cv = (S11 - S12 @ np.linalg.solve(S22, S12.T))[0, 0]
Pm = np.linalg.inv(S3)
check(T, "条件付き分布の公式 = 精度行列からの値", close(cv, 1 / Pm[0, 0], 1e-9) and close(cm, mu3[0] - (Pm[0, 1:] @ (x2v - mu3[1:])) / Pm[0, 0], 1e-9), (cm, cv))
check(T, "p=2 で2変量の公式に戻る", close(S3[0, 1] / S3[1, 1], 0.3 * 20 / 10, 1e-12) and close(S3[0, 0] - S3[0, 1] ** 2 / S3[1, 1], 400 * (1 - 0.09), 1e-9), "数値")
tv3 = np.array([0.01, -0.02, 0.015])
check(T, "MGF exp(t'μ+t'Σt/2)（シミュ 相対誤差<1%）", abs(np.mean(np.exp(X3 @ tv3)) / exp(tv3 @ mu3 + tv3 @ S3 @ tv3 / 2) - 1) < 0.01, np.mean(np.exp(X3 @ tv3)))
pdf3 = stats.multivariate_normal(mu3, S3).pdf([1, 2, 3])
d_ = np.array([1, 2, 3]) - mu3
check(T, "密度の式 = scipy", close(pdf3, exp(-0.5 * d_ @ np.linalg.inv(S3) @ d_) / ((2 * pi) ** 1.5 * sqrt(np.linalg.det(S3))), 1e-15), pdf3)

# ================= noncentral-chi-squared-distribution =================
T = "noncentral-chi-squared-distribution"
check(T, "E=k+λ, V=2(k+2λ)（scipy ncx2, k=3,λ=5）", close(stats.ncx2(3, 5).mean(), 8, 1e-10) and close(stats.ncx2(3, 5).var(), 26, 1e-9), stats.ncx2(3, 5).stats())
muv = sp.Symbol("m", real=True)
M1n = sp.integrate(sp.exp(t * (z + muv) ** 2) * phi, (z, -sp.oo, sp.oo), conds="none")
Mnc = lambda tv, kk, ll: (1 - 2 * tv) ** (-kk / 2) * np.exp(ll * tv / (1 - 2 * tv))
check(T, "MGF（数値積分 k=3,λ=5, t=0.2,-0.5）", all(close(integrate.quad(lambda y: np.exp(tv * y + stats.ncx2.logpdf(y, 3, 5)), 0, 600, limit=400)[0], Mnc(tv, 3, 5), 1e-6) for tv in (0.2, -0.5)), Mnc(0.2, 3, 5))
lmb = sp.Symbol("l", positive=True)
Mncs = (1 - 2 * t) ** (-k / 2) * sp.exp(lmb * t / (1 - 2 * t))
e_, v_ = moments_from_mgf(Mncs)
check(T, "MGF 微分で E=k+λ, V=2k+4λ", zero(e_ - (k + lmb)) and zero(v_ - (2 * k + 4 * lmb)), (e_, sp.expand(v_)))
check(T, "再生性（MGF の積）", zero(Mncs.subs({k: 2, lmb: 1}) * Mncs.subs({k: 3, lmb: 4}) - Mncs.subs({k: 5, lmb: 5})), "sympy")
cfn = complex(integrate.quad(lambda y: np.cos(0.3 * y) * stats.ncx2.pdf(y, 3, 5), 0, np.inf, limit=400)[0], integrate.quad(lambda y: np.sin(0.3 * y) * stats.ncx2.pdf(y, 3, 5), 0, np.inf, limit=400)[0])
check(T, "特性関数（数値積分 t=0.3）", abs(cfn - (1 - 0.6j) ** -1.5 * np.exp(1j * 5 * 0.3 / (1 - 0.6j))) < 1e-6, cfn)
Wn = ((rng.normal(size=(N, 3)) + np.array([2.0, -1.0, 0.0])) ** 2).sum(axis=1)
Wr = ((rng.normal(size=(N, 3)) + np.array([0.0, 0.0, sqrt(5)])) ** 2).sum(axis=1)
check(T, "Σ(Zi+μi)² ~ χ²(3,5)、μ の向きによらない（KS<0.005）", ks(Wn, "ncx2", args=(3, 5)).statistic < 0.005 and ks(Wr, "ncx2", args=(3, 5)).statistic < 0.005, ks(Wn, "ncx2", args=(3, 5)).statistic)
yy_ = np.linspace(0.5, 20, 30)
mixp = sum(stats.poisson.pmf(j, 2.5) * stats.chi2.pdf(yy_, 3 + 2 * j) for j in range(80))
check(T, "ポアソン混合 Po(λ/2) の χ²(k+2J) = ncx2", np.allclose(mixp, stats.ncx2.pdf(yy_, 3, 5), atol=1e-12), "pdf")
check(T, "V[X²]=2+4μ²（μ=1.5 sympy）", zero(sp.integrate((z + sp.Rational(3, 2)) ** 4 * phi, (z, -sp.oo, sp.oo)) - (1 + sp.Rational(9, 4)) ** 2 - (2 + 4 * sp.Rational(9, 4))), "sympy")
pw1 = stats.ncx2.sf(stats.chi2.isf(0.05, 1), 1, 4)
pz = Phi(2 - 1.959964) + Phi(-2 - 1.959964)
check(T, "例1 P(χ²(1,4)>3.841)≈0.5160 = Φ(0.04)+Φ(-3.96)", close(pw1, 0.5160) and close(pz, 0.5160) and close(pw1, pz, 1e-6) and close(Phi(0.04) + Phi(-3.96), 0.5160), (pw1, Phi(0.04) + Phi(-3.96)))
c3 = stats.chi2.isf(0.05, 3)
pows = {n_: stats.ncx2.sf(c3, 3, 0.25 * n_) for n_ in (20, 40, 43, 44)}
check(T, "例2 0.4²+0.3²=0.25, 棄却限界7.815", close(0.4**2 + 0.3**2, 0.25, 1e-12) and close(c3, 7.815, 5e-4), c3)
check(T, "例2 検出力 n=20: 0.4405, n=40: 0.7611, n=44: 0.8039, n=43<0.8", close(pows[20], 0.4405) and close(pows[40], 0.7611) and close(pows[44], 0.8039) and pows[43] < 0.8, pows)
check(T, "λ=0 で棄却域の確率=0.05", close(stats.chi2.sf(c3, 3), 0.05, 1e-12), stats.chi2.sf(c3, 3))

# ================= noncentral-t-distribution =================
T = "noncentral-t-distribution"
ETn = lambda nv, d: d * sqrt(nv / 2) * G((nv - 1) / 2) / G(nv / 2)
check(T, "期待値の式 = scipy nct（ν=9,δ=2.53 / ν=4,δ=-1）", all(close(stats.nct(nv, d).mean(), ETn(nv, d), 1e-8) for nv, d in [(9, 2.53), (4, -1)]), stats.nct(9, 2.53).mean())
check(T, "分散 ν(1+δ²)/(ν-2)-E²", all(close(stats.nct(nv, d).var(), nv * (1 + d * d) / (nv - 2) - ETn(nv, d) ** 2, 1e-7) for nv, d in [(9, 2.53), (5, 1.2)]), stats.nct(9, 2.53).var())
check(T, "δ>0 で右に歪む, E>δ", stats.nct(9, 2.53).stats(moments="s") > 0 and ETn(9, 2.53) > 2.53, (stats.nct(9, 2.53).stats(moments="s"), ETn(9, 2.53)))
Tn = (rng.normal(size=N) + 2.53) / np.sqrt(rng.chisquare(9, N) / 9)
check(T, "(Z+δ)/√(W/ν) ~ t(ν,δ)（KS<0.005）", ks(Tn, "nct", args=(9, 2.53)).statistic < 0.005, ks(Tn, "nct", args=(9, 2.53)).statistic)
Xs = rng.normal(0.8, 1, size=(100000, 10))
Tst = Xs.mean(axis=1) / (Xs.std(axis=1, ddof=1) / sqrt(10))
check(T, "正規標本の t 統計量 ~ t(n-1, √n d)（n=10,d=0.8, KS<0.006）", ks(Tst, "nct", args=(9, sqrt(10) * 0.8)).statistic < 0.006, ks(Tst, "nct", args=(9, sqrt(10) * 0.8)).statistic)


def tpow(n_, d=0.8):
    c = stats.t.isf(0.025, n_ - 1); dl = sqrt(n_) * d
    return dl, c, stats.nct.sf(c, n_ - 1, dl) + stats.nct.cdf(-c, n_ - 1, dl)


tabt = {10: (2.530, 2.262, 0.6162), 15: (3.098, 2.145, 0.8213), 20: (3.578, 2.093, 0.9239)}
okt = all(close(tpow(n_)[0], v_[0], 5e-4) and close(tpow(n_)[1], v_[1], 5e-4) and close(tpow(n_)[2], v_[2]) for n_, v_ in tabt.items())
check(T, "例 表（δ・棄却限界・検出力 0.6162/0.8213/0.9239）", okt, {n_: tuple(round(v, 4) for v in tpow(n_)) for n_ in tabt})
check(T, "例 n=14 では 80% 未満（15個程度が必要）", tpow(14)[2] < 0.8 <= tpow(15)[2], tpow(14)[2])
check(T, "例 シミュ検出力 n=10（許容0.005）", abs(np.mean(np.abs(Tst) > stats.t.isf(0.025, 9)) - 0.6162) < 0.005, np.mean(np.abs(Tst) > stats.t.isf(0.025, 9)))
zp = Phi(sqrt(10) * 0.8 - 1.959964) + Phi(-sqrt(10) * 0.8 - 1.959964)
check(T, "例 z 検定の公式で n=10: 約0.7156", close(zp, 0.7156), zp)

# ================= noncentral-f-distribution =================
T = "noncentral-f-distribution"
check(T, "E=n(m+λ)/(m(n-2))（scipy ncf, m=2,n=27,λ=7.917）", close(stats.ncf(2, 27, 7.917).mean(), 27 * (2 + 7.917) / (2 * 25), 1e-8), stats.ncf(2, 27, 7.917).mean())
Fn = (rng.noncentral_chisquare(2, 7.917, N) / 2) / (rng.chisquare(27, N) / 27)
check(T, "(U/m)/(V/n) ~ F(m,n,λ)（KS<0.005）", ks(Fn, "ncf", args=(2, 27, 7.917)).statistic < 0.005, ks(Fn, "ncf", args=(2, 27, 7.917)).statistic)
check(T, "T'² ~ F(1,n,δ²)（KS<0.005）", ks(Tn**2, "ncf", args=(1, 9, 2.53**2)).statistic < 0.005, ks(Tn**2, "ncf", args=(1, 9, 2.53**2)).statistic)
check(T, "両側 t の検出力 = F(1,n,δ²) の検出力", close(tpow(10)[2], stats.ncf.sf(stats.f.isf(0.05, 1, 9), 1, 9, 10 * 0.64), 1e-8), stats.ncf.sf(stats.f.isf(0.05, 1, 9), 1, 9, 6.4))
mus = np.array([50, 52, 55.0])
dev = (mus - mus.mean()) ** 2
check(T, "例 μ̄=52.333, 偏差² 5.444+0.111+7.111=12.667, λ≈7.917", close(mus.mean(), 52.333, 5e-4) and np.allclose(dev, [5.444, 0.111, 7.111], atol=5e-4) and close(dev.sum(), 12.667, 5e-4) and close(10 * dev.sum() / 16, 7.917, 5e-4), (dev.round(4).tolist(), 10 * dev.sum() / 16))


def fpow(r):
    lam_ = r * dev.sum() / 16; dfd = 3 * (r - 1)
    c = stats.f.isf(0.05, 2, dfd)
    return lam_, dfd, c, stats.ncf.sf(c, 2, dfd, lam_)


tabf = {10: (7.917, 27, 3.354, 0.6604), 15: (11.875, 42, 3.220, 0.8542), 20: (15.833, 57, 3.159, 0.9442)}
okf = all(close(fpow(r)[0], v_[0], 5e-4) and fpow(r)[1] == v_[1] and close(fpow(r)[2], v_[2], 5e-4) and close(fpow(r)[3], v_[3]) for r, v_ in tabf.items())
check(T, "例 表（λ・自由度・棄却限界・検出力 0.6604/0.8542/0.9442）", okf, {r: tuple(round(float(v), 4) for v in fpow(r)) for r in tabf})
check(T, "例 r=13 では 80% 未満, r=14 で 80% 以上（修正後: 14区画以上）", fpow(13)[3] < 0.8 <= fpow(14)[3], (fpow(13)[3], fpow(14)[3]))
check(T, "グラフ λ=7.92 で面積≈0.66", close(stats.ncf.sf(stats.f.isf(0.05, 2, 27), 2, 27, 7.92), 0.66, 0.005), stats.ncf.sf(stats.f.isf(0.05, 2, 27), 2, 27, 7.92))
# 一元配置のシミュレーション
reps = 60000
Yg = rng.normal(mus[None, :, None], 4, size=(reps, 3, 10))
gm = Yg.mean(axis=2); SA = 10 * ((gm - gm.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
SE = ((Yg - gm[:, :, None]) ** 2).sum(axis=(1, 2))
Fa = (SA / 2) / (SE / 27)
check(T, "一元配置の F ~ F(2,27,λ)（シミュ KS<0.008, 検出力±0.006）", ks(Fa, "ncf", args=(2, 27, 10 * dev.sum() / 16)).statistic < 0.008 and abs(np.mean(Fa > stats.f.isf(0.05, 2, 27)) - 0.6604) < 0.006, np.mean(Fa > stats.f.isf(0.05, 2, 27)))
check(T, "S_A/σ² ~ χ²(k-1,λ)（KS<0.008）", ks(SA / 16, "ncx2", args=(2, 10 * dev.sum() / 16)).statistic < 0.008, ks(SA / 16, "ncx2", args=(2, 10 * dev.sum() / 16)).statistic)

print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
