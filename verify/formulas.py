"""公式まとめ（content/formulas.md）の検算スクリプト。
分布表の期待値・分散・MGF・PGF は scipy.stats と数値積分・数値和で、主要な公式は数値例・シミュレーションで確かめる。
シミュレーションは seed=20261023。許容誤差は各チェックに明記。
"""
from itertools import combinations, permutations
from math import comb, exp, sqrt, log, gamma as G, pi, factorial
import numpy as np
from scipy import stats, integrate, special, optimize

results = []
rng = np.random.default_rng(20261023)


def check(sec, what, ok, value=""):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {sec} / {what} / 値: {value}")


def rel(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ================= 離散型分布 =================
S = "離散型分布"


def disc_check(name, pmf, support, mean, var, mgf, pgf, ts=(-0.3, 0.1, 0.2), ss=(0.3, 0.8, 1.1)):
    xs = np.array(support, dtype=float)
    p = np.array([pmf(int(x)) for x in support])
    m = float((xs * p).sum()); v = float((xs**2 * p).sum() - m**2)
    ok = rel(p.sum(), 1, 1e-9) and rel(m, mean, 1e-7) and rel(v, var, 1e-7)
    check(S, f"{name} 確率の和=1・期待値・分散", ok, (round(m, 6), round(v, 6)))
    if mgf is not None:
        check(S, f"{name} MGF（数値和と一致）", all(rel(float((np.exp(t * xs) * p).sum()), mgf(t), 1e-7) for t in ts), ts)
    if pgf is not None:
        check(S, f"{name} PGF（数値和と一致）", all(rel(float((s ** xs * p).sum()), pgf(s), 1e-7) for s in ss), ss)


pp, q = 0.3, 0.7
disc_check("ベルヌーイ", lambda x: stats.bernoulli.pmf(x, pp), [0, 1], pp, pp * q, lambda t: q + pp * exp(t), lambda s: q + pp * s)
n = 12
disc_check("二項", lambda x: comb(n, x) * pp**x * q ** (n - x), range(n + 1), n * pp, n * pp * q,
           lambda t: (q + pp * exp(t)) ** n, lambda s: (q + pp * s) ** n)
check(S, "二項 scipy の mean/var と一致", rel(stats.binom(n, pp).mean(), n * pp) and rel(stats.binom(n, pp).var(), n * pp * q))
lam = 2.5
disc_check("ポアソン", lambda x: stats.poisson.pmf(x, lam), range(200), lam, lam,
           lambda t: exp(lam * (exp(t) - 1)), lambda s: exp(lam * (s - 1)))
disc_check("幾何（失敗回数）", lambda x: pp * q**x, range(400), q / pp, q / pp**2,
           lambda t: pp / (1 - q * exp(t)), lambda s: pp / (1 - q * s))
check(S, "幾何 scipy（loc=-1 で台0〜）と一致", rel(stats.geom(pp, loc=-1).mean(), q / pp) and rel(stats.geom(pp, loc=-1).var(), q / pp**2))
r = 4
disc_check("負の二項（失敗回数）", lambda x: comb(r + x - 1, x) * pp**r * q**x, range(600), r * q / pp, r * q / pp**2,
           lambda t: (pp / (1 - q * exp(t))) ** r, lambda s: (pp / (1 - q * s)) ** r)
check(S, "負の二項 scipy.nbinom と一致", rel(stats.nbinom(r, pp).mean(), r * q / pp) and rel(stats.nbinom(r, pp).var(), r * q / pp**2))
N, K, nn = 20, 7, 6
disc_check("超幾何", lambda x: comb(K, x) * comb(N - K, nn - x) / comb(N, nn), range(0, min(nn, K) + 1),
           nn * K / N, nn * K / N * (1 - K / N) * (N - nn) / (N - 1), None, None)
check(S, "超幾何 scipy.hypergeom と一致", rel(stats.hypergeom(N, K, nn).var(), nn * K / N * (1 - K / N) * (N - nn) / (N - 1)))
mm = 6
disc_check("離散一様 1..m", lambda x: 1 / mm, range(1, mm + 1), (mm + 1) / 2, (mm**2 - 1) / 12,
           lambda t: exp(t) * (exp(mm * t) - 1) / (mm * (exp(t) - 1)), lambda s: s * (1 - s**mm) / (mm * (1 - s)))
# 母関数からのモーメント（数値微分）
h = 1e-4
M = lambda t: (q + pp * exp(t)) ** n
Gf = lambda s: (q + pp * s) ** n
m1 = (M(h) - M(-h)) / (2 * h); m2 = (M(h) - 2 * M(0) + M(-h)) / h**2
g1 = (Gf(1 + h) - Gf(1 - h)) / (2 * h); g2 = (Gf(1 + h) - 2 * Gf(1) + Gf(1 - h)) / h**2
check(S, "E[X]=M'(0)=G'(1)、E[X(X-1)]=G''(1)（二項）", rel(m1, n * pp, 1e-5) and rel(g1, n * pp, 1e-5) and rel(g2, n * (n - 1) * pp**2, 1e-4)
      and rel(m2, n * pp * q + (n * pp) ** 2, 1e-4), (m1, g2))
check(S, "P(X=k)=G^(k)(0)/k!（二項 k=3、多項式係数）", rel(np.polynomial.polynomial.polypow([q, pp], n)[3], stats.binom.pmf(3, n, pp), 1e-9), stats.binom.pmf(3, n, pp))
# 試行回数の流儀
check(S, "試行回数の幾何 Y=X+1：E=1/p、MGF=pe^t/(1-qe^t)", rel(sum((x + 1) * pp * q**x for x in range(400)), 1 / pp) and
      rel(sum(exp(0.2 * (x + 1)) * pp * q**x for x in range(400)), pp * exp(0.2) / (1 - q * exp(0.2))))

# ================= 連続型分布 =================
S = "連続型分布"


def cont_check(name, dist, pdf, mean, var, mgf=None, ts=(-0.4, 0.1, 0.3), lo=-np.inf, hi=np.inf):
    tot = integrate.quad(pdf, lo, hi, limit=200)[0]
    m = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=200)[0]
    v = integrate.quad(lambda x: (x - mean) ** 2 * pdf(x), lo, hi, limit=200)[0]
    pdf_ok = all(rel(pdf(x), dist.pdf(x), 1e-8) for x in dist.ppf([0.1, 0.4, 0.8]))
    ok = rel(tot, 1, 1e-6) and rel(m, mean, 1e-5) and rel(v, var, 1e-5) and rel(dist.mean(), mean, 1e-8) and rel(dist.var(), var, 1e-8) and pdf_ok
    check(S, f"{name} 密度（scipyと一致）・積分=1・期待値・分散", ok, (round(m, 6), round(v, 6)))
    if mgf is not None:
        check(S, f"{name} MGF（数値積分と一致）", all(rel(integrate.quad(lambda x: exp(t * x) * pdf(x), max(lo, -200), min(hi, 200), limit=400)[0], mgf(t), 1e-6) for t in ts), ts)


a, b = 1.0, 4.0
cont_check("一様", stats.uniform(a, b - a), lambda x: 1 / (b - a), (a + b) / 2, (b - a) ** 2 / 12,
           lambda t: (exp(t * b) - exp(t * a)) / (t * (b - a)), lo=a, hi=b)
mu, sg = 1.5, 2.0
cont_check("正規", stats.norm(mu, sg), lambda x: exp(-(x - mu) ** 2 / (2 * sg**2)) / (sqrt(2 * pi) * sg), mu, sg**2,
           lambda t: exp(mu * t + sg**2 * t**2 / 2))
lam = 2.0
cont_check("指数（レート）", stats.expon(scale=1 / lam), lambda x: lam * exp(-lam * x), 1 / lam, 1 / lam**2,
           lambda t: lam / (lam - t), lo=0)
al, be = 2.5, 1.5
cont_check("ガンマ（形状・尺度）", stats.gamma(al, scale=be), lambda x: x ** (al - 1) * exp(-x / be) / (G(al) * be**al), al * be, al * be**2,
           lambda t: (1 - be * t) ** (-al), ts=(-0.4, 0.1, 0.5), lo=0)
lr = 1 / be
check(S, "ガンマのレート表記 MGF=(1-t/λ)^(-α)", rel((1 - 0.3 / lr) ** (-al), (1 - be * 0.3) ** (-al), 1e-12))
check(S, "Exp(λ)=Ga(1,1/λ)、χ²(k)=Ga(k/2,2)", rel(stats.gamma(1, scale=1 / lam).pdf(0.7), stats.expon(scale=1 / lam).pdf(0.7)) and
      rel(stats.gamma(3.5, scale=2).pdf(4.2), stats.chi2(7).pdf(4.2)))
ba, bb = 2.0, 3.0
cont_check("ベータ", stats.beta(ba, bb), lambda x: x ** (ba - 1) * (1 - x) ** (bb - 1) / special.beta(ba, bb), ba / (ba + bb),
           ba * bb / ((ba + bb) ** 2 * (ba + bb + 1)), lo=0, hi=1)


def beta_mgf_series(t, a_, b_, K=60):
    s, c = 1.0, 1.0
    for k in range(1, K):
        c *= (a_ + k - 1) / (a_ + b_ + k - 1)
        s += c * t**k / factorial(k)
    return s


check(S, "ベータ MGF の級数（数値積分と一致、t=-2,1,3）",
      all(rel(integrate.quad(lambda x: exp(t * x) * stats.beta(ba, bb).pdf(x), 0, 1)[0], beta_mgf_series(t, ba, bb), 1e-8) for t in (-2, 1, 3)),
      beta_mgf_series(1, ba, bb))
check(S, "ベータ MGF = 合流型超幾何 1F1(a;a+b;t)", rel(special.hyp1f1(ba, ba + bb, 1.7), beta_mgf_series(1.7, ba, bb), 1e-9))
k = 5
cont_check("カイ二乗", stats.chi2(k), lambda x: x ** (k / 2 - 1) * exp(-x / 2) / (2 ** (k / 2) * G(k / 2)), k, 2 * k,
           lambda t: (1 - 2 * t) ** (-k / 2), lo=0)
nu = 7
cont_check("t", stats.t(nu), lambda x: G((nu + 1) / 2) / (sqrt(nu * pi) * G(nu / 2)) * (1 + x**2 / nu) ** (-(nu + 1) / 2), 0, nu / (nu - 2))
check(S, "t の MGF は存在しない（t=0.5 で ∫ が発散：上限を広げると増え続ける）",
      integrate.quad(lambda x: exp(0.5 * x) * stats.t(nu).pdf(x), 0, 200)[0] > 100 * integrate.quad(lambda x: exp(0.5 * x) * stats.t(nu).pdf(x), 0, 50)[0])
fm, fn = 5, 10
fpdf = lambda x: (fm / fn) ** (fm / 2) * x ** (fm / 2 - 1) * (1 + fm * x / fn) ** (-(fm + fn) / 2) / special.beta(fm / 2, fn / 2)
cont_check("F", stats.f(fm, fn), fpdf, fn / (fn - 2), 2 * fn**2 * (fm + fn - 2) / (fm * (fn - 2) ** 2 * (fn - 4)), lo=0)
lm, ls = 0.3, 0.6
cont_check("対数正規", stats.lognorm(ls, scale=exp(lm)), lambda x: exp(-(log(x) - lm) ** 2 / (2 * ls**2)) / (sqrt(2 * pi) * ls * x),
           exp(lm + ls**2 / 2), exp(2 * lm + ls**2) * (exp(ls**2) - 1), lo=0)
check(S, "対数正規の MGF は存在しない（t=0.1 で ∫_0^U e^{tx}f が U とともに発散）",
      integrate.quad(lambda x: exp(0.1 * x - (log(x) - lm) ** 2 / (2 * ls**2) - log(x)), 1, 3000, limit=500)[0] > 1e10)
cm, cg = 1.0, 2.0
check(S, "コーシー 密度が scipy と一致、C(0,1)=t(1)", all(rel(cg / (pi * (cg**2 + (x - cm) ** 2)), stats.cauchy(cm, cg).pdf(x), 1e-10) for x in (-3, 0.5, 4))
      and rel(stats.cauchy().pdf(1.3), stats.t(1).pdf(1.3), 1e-12))
check(S, "コーシー 期待値が存在しない（∫|x|f が発散：上限1e6で>8）", integrate.quad(lambda x: 2 * x * stats.cauchy().pdf(x), 0, 1e6, limit=500)[0] > 8)
wk, we = 1.7, 2.5
cont_check("ワイブル（形状・尺度）", stats.weibull_min(wk, scale=we), lambda x: wk / we * (x / we) ** (wk - 1) * exp(-((x / we) ** wk)),
           we * G(1 + 1 / wk), we**2 * (G(1 + 2 / wk) - G(1 + 1 / wk) ** 2), lo=0)

# ================= 期待値と分散 =================
S = "期待値と分散"
X = rng.normal(1, 2, 400000); Y = 0.5 * X + rng.normal(0, 1, 400000)
check(S, "V[X+Y]=V[X]+V[Y]+2Cov（標本で恒等的に成立）", rel(np.var(X + Y), np.var(X) + np.var(Y) + 2 * np.cov(X, Y, bias=True)[0, 1], 1e-9))
check(S, "Cov(aX+b,cY+d)=ac Cov", rel(np.cov(3 * X + 1, -2 * Y + 5, bias=True)[0, 1], -6 * np.cov(X, Y, bias=True)[0, 1], 1e-9))
Sig = np.array([[2, 0.6, 0.3], [0.6, 1, -0.2], [0.3, -0.2, 1.5]]); A = np.array([[1, 2, 0], [0, 1, -1]])
Z = rng.multivariate_normal([0, 0, 0], Sig, 400000)
check(S, "V[AX]=AΣA^T（シミュ、各成分の差<0.05）", np.max(np.abs(np.cov((Z @ A.T).T) - A @ Sig @ A.T)) < 0.05, np.max(np.abs(np.cov((Z @ A.T).T) - A @ Sig @ A.T)))
# 全分散の公式：λ~Ga(α,尺度β)、Y|λ~Po(λ)
al, be = 3.0, 2.0
Lm = rng.gamma(al, be, 400000); Yp = rng.poisson(Lm)
check(S, "全分散 V[Y]=E[V|λ]+V[E|λ]=αβ+αβ²（ガンマ・ポアソン、シミュ相対差<2%）", abs(Yp.var() / (al * be + al * be**2) - 1) < 0.02, (Yp.var(), al * be + al * be**2))
xs = rng.exponential(1.0, (100000, 5))
check(S, "E[s²]=σ²（Exp(1)、n=5、シミュ相対差<1%）", abs(xs.var(axis=1, ddof=1).mean() - 1) < 0.01, xs.var(axis=1, ddof=1).mean())
check(S, "正規の尖度 β2=3、歪度0", rel(stats.norm.stats(moments="k") + 3, 3) and rel(stats.norm.stats(moments="s") + 1, 1))
check(S, "キュムラント K'(0)=μ, K''(0)=σ²（ポアソンで K=λ(e^t-1)）", rel((2.5 * (exp(h) - 1) - 2.5 * (exp(-h) - 1)) / (2 * h), 2.5, 1e-6)
      and rel((2.5 * (exp(h) - 1) + 2.5 * (exp(-h) - 1)) / h**2, 2.5, 1e-4))
rho, sx, sy, mx, my = 0.6, 2.0, 3.0, 1.0, -1.0
B2 = rng.multivariate_normal([mx, my], [[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]], 600000)
sel = np.abs(B2[:, 0] - 2.0) < 0.05
check(S, "2変量正規の条件付き平均・分散（x=2付近、差<0.05/相対<3%）",
      abs(B2[sel, 1].mean() - (my + rho * sy / sx * (2 - mx))) < 0.05 and abs(B2[sel, 1].var() / (sy**2 * (1 - rho**2)) - 1) < 0.03,
      (B2[sel, 1].mean(), B2[sel, 1].var()))
Zt = rng.normal(size=200000); Xa = Zt + rng.normal(size=200000); Ya = 2 * Zt + rng.normal(size=200000)
R = np.corrcoef([Xa, Ya, Zt]); rxy, rxz, ryz = R[0, 1], R[0, 2], R[1, 2]
ex = Xa - np.polyval(np.polyfit(Zt, Xa, 1), Zt); ey = Ya - np.polyval(np.polyfit(Zt, Ya, 1), Zt)
check(S, "偏相関の式 = z で回帰した残差どうしの相関", rel((rxy - rxz * ryz) / sqrt((1 - rxz**2) * (1 - ryz**2)), np.corrcoef(ex, ey)[0, 1], 1e-8),
      np.corrcoef(ex, ey)[0, 1])

# ================= 変数変換 =================
S = "変数変換"
U = rng.uniform(size=400000)
E = -np.log(U) / 2.0
check(S, "逆関数法 -log U/λ ~ Exp(λ)（KS検定 p>0.01）", stats.kstest(E, stats.expon(scale=0.5).cdf).pvalue > 0.01, stats.kstest(E, stats.expon(scale=0.5).cdf).pvalue)
f_y = lambda y: stats.norm.pdf(np.log(y), 0.3, 0.6) / y
check(S, "1変数の変数変換で対数正規の密度が出る", all(rel(f_y(y), stats.lognorm(0.6, scale=exp(0.3)).pdf(y), 1e-10) for y in (0.5, 1, 2.5)))
conv = integrate.quad(lambda x: stats.expon.pdf(x) * stats.expon.pdf(2.3 - x), 0, 2.3)[0]
check(S, "畳み込み Exp(1)+Exp(1) の密度 = Ga(2,1)", rel(conv, stats.gamma(2).pdf(2.3), 1e-8), conv)
check(S, "再生性：二項・ポアソン・負の二項（PGF の積）",
      rel(((q + pp * 0.7) ** 3) * ((q + pp * 0.7) ** 5), (q + pp * 0.7) ** 8, 1e-12) and
      rel(sum(stats.poisson.pmf(j, 1.2) * stats.poisson.pmf(4 - j, 2.3) for j in range(5)), stats.poisson.pmf(4, 3.5), 1e-10) and
      rel(sum(stats.nbinom.pmf(j, 2, 0.4) * stats.nbinom.pmf(5 - j, 3, 0.4) for j in range(6)), stats.nbinom.pmf(5, 5, 0.4), 1e-10))
g2 = rng.gamma(1.5, 2.0, 300000) + rng.gamma(2.5, 2.0, 300000)
check(S, "再生性：Ga(1.5,2)+Ga(2.5,2)=Ga(4,2)（KS p>0.01）", stats.kstest(g2, stats.gamma(4, scale=2).cdf).pvalue > 0.01)
Zs = rng.normal(size=(200000, 6)) * 2 + 1
ss2 = Zs.var(axis=1, ddof=1)
check(S, "(n-1)s²/σ² ~ χ²(n-1)（n=6、KS p>0.01）", stats.kstest(5 * ss2 / 4, stats.chi2(5).cdf).pvalue > 0.01)
check(S, "X̄ と s² は無相関（正規）", abs(np.corrcoef(Zs.mean(axis=1), ss2)[0, 1]) < 0.01, np.corrcoef(Zs.mean(axis=1), ss2)[0, 1])
check(S, "T²~F(1,k)：t_{0.025}(8)² = F_{0.05}(1,8)", rel(stats.t.ppf(0.975, 8) ** 2, stats.f.ppf(0.95, 1, 8), 1e-9))
check(S, "1/F~F(n,m)：F_{0.05}(4,9)=1/F_{0.95}(9,4)", rel(stats.f.ppf(0.95, 4, 9), 1 / stats.f.ppf(0.05, 9, 4), 1e-9))
U1, U2 = rng.uniform(size=200000), rng.uniform(size=200000)
Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * pi * U2); Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * pi * U2)
check(S, "ボックス・ミュラー：N(0,1)（KS p>0.01）かつ無相関", stats.kstest(Z1, "norm").pvalue > 0.01 and stats.kstest(Z2, "norm").pvalue > 0.01 and abs(np.corrcoef(Z1, Z2)[0, 1]) < 0.01)
nk, kk, xv = 7, 3, 0.4
fk = factorial(nk) / (factorial(kk - 1) * factorial(nk - kk)) * xv ** (kk - 1) * (1 - xv) ** (nk - kk)
check(S, "順序統計量の密度：U(0,1) の X_(3) (n=7) = Be(3,5)", rel(fk, stats.beta(kk, nk - kk + 1).pdf(xv), 1e-10), fk)
check(S, "Box-Cox は λ→0 で log y", rel((2.7**1e-8 - 1) / 1e-8, log(2.7), 1e-6))

# ================= 極限定理 =================
S = "極限定理"
check(S, "マルコフ・チェビシェフの不等式（Exp(1)、a=3、k=2）", stats.expon.sf(3) <= 1 / 3 and (stats.expon.sf(3)) <= 1 / 4)
bn, bp = 50, 0.3
approx = stats.norm.cdf((12 + 0.5 - bn * bp) / sqrt(bn * bp * (1 - bp)))
check(S, "連続修正つき正規近似 P(X≤12), Bin(50,0.3)（真値との差<0.01）", abs(approx - stats.binom.cdf(12, bn, bp)) < 0.01, (approx, stats.binom.cdf(12, bn, bp)))
check(S, "少数の法則 Bin(1000,0.002)≈Po(2)（各確率の差<0.001）", max(abs(stats.binom.pmf(j, 1000, 0.002) - stats.poisson.pmf(j, 2)) for j in range(10)) < 0.001)
xb = rng.exponential(2.0, (100000, 200)).mean(axis=1)
gx = np.log(xb)
check(S, "デルタ法：log X̄ の分散 ≈ (1/μ)²σ²/n（Exp 平均2、n=200、相対差<3%）", abs(gx.var() / ((1 / 2) ** 2 * 4 / 200) - 1) < 0.03, gx.var())
ph = rng.binomial(100, 0.2, 200000) / 100
check(S, "分散安定化 arcsin√p̂ の分散 ≈ 1/(4n)（n=100、相対差<5%）", abs(np.arcsin(np.sqrt(ph)).var() / (1 / 400) - 1) < 0.05, np.arcsin(np.sqrt(ph)).var())
check(S, "分散安定化 √X の分散 ≈ 1/4（Po(30)、差<0.01）", abs(np.sqrt(rng.poisson(30, 200000)).var() - 0.25) < 0.01)
rr = []
C = [[1, 0.5], [0.5, 1]]
for _ in range(20000):
    d = rng.multivariate_normal([0, 0], C, 30)
    rr.append(np.corrcoef(d.T)[0, 1])
zr = np.arctanh(rr)
check(S, "フィッシャーの z：artanh r の分散 ≈ 1/(n-3)（n=30、相対差<5%）、平均≈artanh ρ", abs(zr.var() * 27 - 1) < 0.05 and abs(zr.mean() - np.arctanh(0.5)) < 0.02, (zr.var() * 27, zr.mean()))
cl = (rng.uniform(size=(100000, 30)).mean(axis=1) - 0.5) / sqrt(1 / 12 / 30)
check(S, "中心極限定理：一様分布の標準化平均（n=30）の分位点が N(0,1) に近い", abs(np.quantile(cl, 0.975) - 1.96) < 0.03)

# ================= 推定量の性質 =================
S = "推定量の性質"
th = rng.normal(0, 1, (100000, 10))
s2b = th.var(axis=1)
check(S, "MSE=V+bias²（σ̂² 1/n 版、恒等式）", rel(((s2b - 1) ** 2).mean(), s2b.var() + (s2b.mean() - 1) ** 2, 1e-9))


def fisher_num(logf, theta, xs_, ws):
    hh = 1e-4
    d2 = [(logf(x, theta + hh) - 2 * logf(x, theta) + logf(x, theta - hh)) / hh**2 for x in xs_]
    return -float(np.dot(ws, d2))


xs_b = [0, 1]; check(S, "I(p)=1/(p(1-p))（ベルヌーイ、数値微分）",
                     rel(fisher_num(lambda x, p_: x * log(p_) + (1 - x) * log(1 - p_), 0.3, xs_b, [0.7, 0.3]), 1 / 0.21, 1e-4))
xs_p = list(range(80)); check(S, "I(λ)=1/λ（ポアソン）", rel(fisher_num(lambda x, l: x * log(l) - l, 2.5, xs_p, stats.poisson.pmf(xs_p, 2.5)), 1 / 2.5, 1e-4))
check(S, "I(λ)=1/λ²（指数）", rel(integrate.quad(lambda x: (1 / 2 - x) ** 2 * 2 * exp(-2 * x), 0, np.inf)[0], 1 / 4, 1e-6))
check(S, "I(σ²)=1/(2σ⁴)（正規、μ既知、スコアの2乗の期待値）",
      rel(integrate.quad(lambda x: (-1 / (2 * 4) + x**2 / (2 * 16)) ** 2 * stats.norm(0, 2).pdf(x), -np.inf, np.inf)[0], 1 / (2 * 16), 1e-6))
score_mean = integrate.quad(lambda x: (2.0 / 1 - x) * 0 + (1 / 2 - x) * 2 * exp(-2 * x), 0, np.inf)[0]
check(S, "E[U]=0（指数、λ=2）", abs(score_mean) < 1e-9, score_mean)
lh = 1 / rng.exponential(0.5, (100000, 100)).mean(axis=1)
check(S, "MLE の漸近分散 1/(nI)：指数 λ̂=1/x̄ の分散 ≈ λ²/n（λ=2,n=100、相対差<5%）", abs(lh.var() / (4 / 100) - 1) < 0.05, lh.var())
check(S, "クラメール・ラオ：ポアソンの x̄ は下限 λ/n を達成", rel(rng.poisson(2.5, (100000, 20)).mean(axis=1).var(), 2.5 / 20, 0.03))
um = rng.uniform(0, 3, (200000, 5)).max(axis=1)
check(S, "U(0,θ) の max の期待値 nθ/(n+1)", abs(um.mean() - 5 * 3 / 6) < 0.01, um.mean())
gd = rng.gamma(2.5, 1.5, 2000000)
check(S, "モーメント法 α̂=x̄²/σ̂², β̂=σ̂²/x̄（大標本で真値に近い）", abs(gd.mean() ** 2 / gd.var() - 2.5) < 0.02 and abs(gd.var() / gd.mean() - 1.5) < 0.01)

# ================= 区間推定 =================
S = "区間推定"
reps = 40000


def coverage(fn):
    return np.mean([fn() for _ in range(reps)])


def cov_var():
    x = rng.normal(5, 2, 12); s2 = x.var(ddof=1)
    return 11 * s2 / stats.chi2.ppf(0.975, 11) <= 4 <= 11 * s2 / stats.chi2.ppf(0.025, 11)


c = coverage(cov_var); check(S, "母分散の区間の被覆率≈0.95（n=12、±0.006）", abs(c - 0.95) < 0.006, c)


def cov_ratio():
    x = rng.normal(0, 2, 10); y = rng.normal(0, 1, 15); rt = x.var(ddof=1) / y.var(ddof=1)
    return rt / stats.f.ppf(0.975, 9, 14) <= 4 <= rt * stats.f.ppf(0.975, 14, 9)


c = coverage(cov_ratio); check(S, "分散比の区間の被覆率≈0.95（m=10,n=15、±0.006）", abs(c - 0.95) < 0.006, c)


def cov_pool():
    x = rng.normal(1, 1.5, 8); y = rng.normal(0, 1.5, 11)
    sp2 = (7 * x.var(ddof=1) + 10 * y.var(ddof=1)) / 17
    hw = stats.t.ppf(0.975, 17) * sqrt(sp2 * (1 / 8 + 1 / 11))
    return abs(x.mean() - y.mean() - 1) <= hw


c = coverage(cov_pool); check(S, "平均の差（等分散）の区間の被覆率≈0.95（±0.006）", abs(c - 0.95) < 0.006, c)
z = stats.norm.ppf(0.975); nn, xx = 40, 6; ph = xx / nn
wl = (ph + z**2 / (2 * nn) - z * sqrt(ph * (1 - ph) / nn + z**2 / (4 * nn**2))) / (1 + z**2 / nn)
wu = (ph + z**2 / (2 * nn) + z * sqrt(ph * (1 - ph) / nn + z**2 / (4 * nn**2))) / (1 + z**2 / nn)
check(S, "ウィルソン区間の端点で |p̂-p|/√(p(1-p)/n)=z（スコア検定の反転）",
      all(rel(abs(ph - p_) / sqrt(p_ * (1 - p_) / nn), z, 1e-9) for p_ in (wl, wu)), (wl, wu))
check(S, "相関の区間 tanh(z ± z_{α/2}/√(n-3)) は z 変換の逆", rel(np.tanh(np.arctanh(0.42)), 0.42, 1e-12))
E_, sgm = 0.5, 3.0
nreq = (z * sgm / E_) ** 2
check(S, "必要な標本サイズ：n=(zσ/E)² で誤差幅がちょうど E", rel(z * sgm / sqrt(nreq), E_, 1e-12), nreq)
pg = np.linspace(0.01, 0.99, 99)
check(S, "比率の標本サイズは p=1/2 で最大（z²/(4E²)）", pg[np.argmax(pg * (1 - pg))] == 0.5)

# ================= 検定統計量 =================
S = "検定統計量"
x = rng.normal(10, 2, 9); y = rng.normal(11, 4, 14)
m_, n_ = len(x), len(y)
sp2 = ((m_ - 1) * x.var(ddof=1) + (n_ - 1) * y.var(ddof=1)) / (m_ + n_ - 2)
tp = (x.mean() - y.mean()) / sqrt(sp2 * (1 / m_ + 1 / n_))
check(S, "2標本 t（等分散）が scipy と一致", rel(tp, stats.ttest_ind(x, y).statistic, 1e-10), tp)
v1, v2 = x.var(ddof=1) / m_, y.var(ddof=1) / n_
tw = (x.mean() - y.mean()) / sqrt(v1 + v2); nuw = (v1 + v2) ** 2 / (v1**2 / (m_ - 1) + v2**2 / (n_ - 1))
check(S, "ウェルチの t と自由度（p 値が scipy と一致）", rel(2 * stats.t.sf(abs(tw), nuw), stats.ttest_ind(x, y, equal_var=False).pvalue, 1e-9), (tw, nuw))
t1 = (x.mean() - 9) / (x.std(ddof=1) / sqrt(m_))
check(S, "1標本 t が scipy と一致", rel(t1, stats.ttest_1samp(x, 9).statistic, 1e-10))
d = rng.normal(0.5, 1, 12); e = rng.normal(0, 1, 12)
check(S, "対応のある t = 差の1標本 t", rel(stats.ttest_rel(d, e).statistic, stats.ttest_1samp(d - e, 0).statistic, 1e-10))
xr = rng.normal(size=25); yr = 0.4 * xr + rng.normal(size=25); rv = np.corrcoef(xr, yr)[0, 1]
tr = rv * sqrt(23) / sqrt(1 - rv**2)
check(S, "無相関の t（p 値が pearsonr と一致）", rel(2 * stats.t.sf(abs(tr), 23), stats.pearsonr(xr, yr)[1], 1e-8), tr)
pT = 3; nT = 15; T2 = []
for _ in range(20000):
    Xs = rng.normal(size=(nT, pT)); xb_ = Xs.mean(axis=0); Sinv = np.linalg.inv(np.cov(Xs.T))
    T2.append(nT * xb_ @ Sinv @ xb_)
FT = (nT - pT) / (pT * (nT - 1)) * np.array(T2)
check(S, "ホテリング：(n-p)T²/(p(n-1)) ~ F(p,n-p)（KS p>0.01）", stats.kstest(FT, stats.f(pT, nT - pT).cdf).pvalue > 0.01)
mu0, mu1, sgp, nz, al_ = 0, 0.5, 1.0, 25, 0.05
pw = stats.norm.cdf(sqrt(nz) * (mu1 - mu0) / sgp - stats.norm.ppf(1 - al_))
simz = (rng.normal(mu1, sgp, (100000, nz)).mean(axis=1) - mu0) / (sgp / sqrt(nz)) > stats.norm.ppf(1 - al_)
check(S, "片側 z 検定の検出力の式（シミュ差<0.005）", abs(simz.mean() - pw) < 0.005, (pw, simz.mean()))
nreq = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) * sgp / (mu1 - mu0)) ** 2
check(S, "標本サイズ式で検出力がちょうど 0.8", rel(stats.norm.cdf(sqrt(nreq) * (mu1 - mu0) / sgp - stats.norm.ppf(0.95)), 0.8, 1e-10), nreq)
pv = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
holm = []
for i, pi_ in enumerate(np.sort(pv)):
    if pi_ <= 0.05 / (len(pv) - i):
        holm.append(pi_)
    else:
        break
bh = max([i + 1 for i, pi_ in enumerate(np.sort(pv)) if pi_ <= (i + 1) * 0.05 / len(pv)], default=0)
check(S, "ホルム法（手計算で1個）・BH 法（2個、scipy.false_discovery_control と一致）",
      len(holm) == 1 and bh == 2 and bh == int((stats.false_discovery_control(pv) <= 0.05).sum()), (len(holm), bh))

# ================= 尤度による検定 =================
S = "尤度による検定"
lr_ = []
for _ in range(30000):
    xx = rng.poisson(3.0, 20); lh_ = xx.mean()
    l1 = np.sum(xx * np.log(lh_) - lh_) if lh_ > 0 else -20 * lh_
    lr_.append(2 * (l1 - np.sum(xx * np.log(3.0) - 3.0)))
check(S, "ウィルクス：-2logΛ ~ χ²(1)（ポアソン、棄却率≈0.05、±0.006）", abs(np.mean(np.array(lr_) > stats.chi2.ppf(0.95, 1)) - 0.05) < 0.006, np.mean(np.array(lr_) > stats.chi2.ppf(0.95, 1)))
nb, xb_, p0 = 50, 18, 0.25; phat = xb_ / nb
score = (phat - p0) ** 2 / (p0 * (1 - p0) / nb)
U0 = xb_ / p0 - (nb - xb_) / (1 - p0)
check(S, "スコア統計量 U(θ0)²/(nI(θ0)) = 二項の z²", rel(U0**2 / (nb / (p0 * (1 - p0))), score, 1e-10), score)
check(S, "ワルド統計量 = (p̂-p0)²/(p̂(1-p̂)/n)", rel((phat - p0) ** 2 * nb / (phat * (1 - phat)), (phat - p0) ** 2 / (phat * (1 - phat) / nb), 1e-12))

# ================= カイ二乗検定 =================
S = "カイ二乗検定"
tab = np.array([[23, 17], [11, 29]]); a_, b_, c_, d_ = tab.ravel(); nt = tab.sum()
den = (a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_)
check(S, "2×2 の簡便式 = chi2_contingency（補正なし）", rel(nt * (a_ * d_ - b_ * c_) ** 2 / den, stats.chi2_contingency(tab, correction=False)[0], 1e-10))
check(S, "イェーツ補正の式 = chi2_contingency（補正あり）", rel(nt * (abs(a_ * d_ - b_ * c_) - nt / 2) ** 2 / den, stats.chi2_contingency(tab, correction=True)[0], 1e-10))
tab3 = np.array([[12, 18, 30], [25, 15, 20], [8, 22, 10]])
Eij = tab3.sum(1, keepdims=True) * tab3.sum(0, keepdims=True) / tab3.sum()
res = stats.chi2_contingency(tab3)
check(S, "独立性：E=n_i.n_.j/n、自由度 (r-1)(c-1)", np.allclose(Eij, res[3]) and res[2] == 4 and rel(((tab3 - Eij) ** 2 / Eij).sum(), res[0], 1e-10))
check(S, "G=2ΣO log(O/E) = power_divergence(log-likelihood)",
      rel(2 * (tab3 * np.log(tab3 / Eij)).sum(), stats.chi2_contingency(tab3, lambda_="log-likelihood")[0], 1e-10))
obs = np.array([18, 25, 31, 26]); ex = np.full(4, 25)
check(S, "適合度 Σ(O-E)²/E = scipy.chisquare", rel(((obs - ex) ** 2 / ex).sum(), stats.chisquare(obs, ex)[0], 1e-12))
# 推定母数で自由度が減る：ポアソンを λ̂ で当てはめた適合度のシミュ（4カテゴリ 0,1,2,3+ → 自由度 2）
chis = []
for _ in range(20000):
    xx = rng.poisson(1.5, 200); lh_ = xx.mean()
    o = np.array([(xx == 0).sum(), (xx == 1).sum(), (xx == 2).sum(), (xx >= 3).sum()])
    pr = stats.poisson.pmf([0, 1, 2], lh_); pr = np.append(pr, 1 - pr.sum())
    chis.append(((o - 200 * pr) ** 2 / (200 * pr)).sum())
check(S, "推定母数1個で自由度 k-1-1（平均≈2、±0.1）", abs(np.mean(chis) - 2) < 0.1, np.mean(chis))
mb = []
for _ in range(30000):
    cnt = rng.multinomial(60, [0.3, 0.15, 0.15, 0.4]); bq, cq = cnt[1], cnt[2]
    mb.append((bq - cq) ** 2 / (bq + cq))
check(S, "マクネマー (b-c)²/(b+c) ~ χ²(1)（棄却率≈0.05、±0.008）", abs(np.mean(np.array(mb) > 3.841) - 0.05) < 0.008, np.mean(np.array(mb) > 3.841))
dz = []
for _ in range(20000):
    xx = rng.poisson(4, 30); dz.append(((xx - xx.mean()) ** 2).sum() / xx.mean())
check(S, "過分散の検定統計量の平均≈n-1（ポアソン、n=30、±0.3）", abs(np.mean(dz) - 29) < 0.3, np.mean(dz))

# ================= ノンパラメトリック =================
S = "ノンパラメトリック"
m_, n_ = 4, 5; Nn = m_ + n_
W = np.array([sum(c) for c in combinations(range(1, Nn + 1), m_)])
check(S, "順位和 W の期待値 m(N+1)/2・分散 mn(N+1)/12（全列挙）", rel(W.mean(), m_ * (Nn + 1) / 2) and rel(W.var(), m_ * n_ * (Nn + 1) / 12), (W.mean(), W.var()))
Uv = W - m_ * (m_ + 1) / 2
check(S, "U=W-m(m+1)/2 の平均 mn/2、mannwhitneyu と一致", rel(Uv.mean(), m_ * n_ / 2) and
      stats.mannwhitneyu([1.2, 3.4, 0.5, 2.2], [2.0, 4.1, 3.0, 5.5, 0.9]).statistic == sum(sorted([1.2, 3.4, 0.5, 2.2, 2.0, 4.1, 3.0, 5.5, 0.9]).index(v) + 1 for v in [1.2, 3.4, 0.5, 2.2]) - 10)
ns = 7
Tp = np.array([sum(i + 1 for i in range(ns) if (mask >> i) & 1) for mask in range(2**ns)])
check(S, "符号付き順位 T+ の期待値 n(n+1)/4・分散 n(n+1)(2n+1)/24（全列挙）", rel(Tp.mean(), ns * (ns + 1) / 4) and rel(Tp.var(), ns * (ns + 1) * (2 * ns + 1) / 24))
nk_ = 6
taus = [stats.kendalltau(range(nk_), pmt)[0] for pmt in permutations(range(nk_))]
check(S, "ケンドールの τ の分散 2(2n+5)/(9n(n-1))（全列挙 n=6）", rel(np.var(taus), 2 * (2 * nk_ + 5) / (9 * nk_ * (nk_ - 1)), 1e-9), np.var(taus))
xs_ = [3.1, 1.2, 5.6, 4.4, 2.0, 6.3, 0.7]; ys_ = [2.2, 1.9, 6.1, 3.3, 3.0, 5.0, 1.1]
rx, ry = stats.rankdata(xs_), stats.rankdata(ys_)
check(S, "スピアマン 1-6Σd²/(n(n²-1)) = spearmanr", rel(1 - 6 * ((rx - ry) ** 2).sum() / (7 * 48), stats.spearmanr(xs_, ys_)[0], 1e-12))
grp = [[2.1, 3.5, 1.8, 4.0], [5.2, 6.1, 4.8], [3.3, 7.0, 6.6, 5.9, 8.1]]
allr = stats.rankdata(np.concatenate(grp)); Ng = len(allr); idx = np.cumsum([0] + [len(g) for g in grp])
Hk = 12 / (Ng * (Ng + 1)) * sum(allr[idx[i]:idx[i + 1]].sum() ** 2 / len(grp[i]) for i in range(3)) - 3 * (Ng + 1)
check(S, "クラスカル・ウォリス H = scipy.kruskal", rel(Hk, stats.kruskal(*grp).statistic, 1e-10), Hk)

# ================= 回帰分析・回帰診断 =================
S = "回帰分析"
nr, pr_ = 30, 3
Xd = np.column_stack([np.ones(nr), rng.normal(size=(nr, pr_))])
yv = Xd @ np.array([1, 0.5, -0.3, 0.0]) + rng.normal(0, 1, nr)
XtXi = np.linalg.inv(Xd.T @ Xd); bh_ = XtXi @ Xd.T @ yv; yh = Xd @ bh_; ev = yv - yh
Se = ev @ ev; ST = ((yv - yv.mean()) ** 2).sum(); SR = ((yh - yv.mean()) ** 2).sum()
check(S, "S_T=S_R+S_e、R²=1-S_e/S_T=corr(y,ŷ)²", rel(ST, SR + Se, 1e-10) and rel(1 - Se / ST, np.corrcoef(yv, yh)[0, 1] ** 2, 1e-10))
R2 = 1 - Se / ST; s2r = Se / (nr - pr_ - 1)
check(S, "自由度調整済み R² の2つの形が一致", rel(1 - (Se / (nr - pr_ - 1)) / (ST / (nr - 1)), 1 - (1 - R2) * (nr - 1) / (nr - pr_ - 1), 1e-12))
check(S, "F=(S_R/p)/(S_e/(n-p-1)) = (R²/p)/((1-R²)/(n-p-1))", rel((SR / pr_) / s2r, (R2 / pr_) / ((1 - R2) / (nr - pr_ - 1)), 1e-10))
tj = bh_ / np.sqrt(s2r * np.diag(XtXi))
Xr = Xd[:, [0, 1, 2]]; br = np.linalg.lstsq(Xr, yv, rcond=None)[0]; Se0 = ((yv - Xr @ br) ** 2).sum()
check(S, "部分 F（1係数）= t²", rel((Se0 - Se) / s2r, tj[3] ** 2, 1e-9), tj[3] ** 2)
xs1 = Xd[:, 1]; lr1 = stats.linregress(xs1, yv)
Sxx = ((xs1 - xs1.mean()) ** 2).sum(); Sxy = ((xs1 - xs1.mean()) * (yv - yv.mean())).sum()
check(S, "単回帰 β̂1=Sxy/Sxx、R²=r²、SE=σ̂/√Sxx（linregress と一致）",
      rel(Sxy / Sxx, lr1.slope, 1e-10) and rel(lr1.rvalue**2, (Sxy**2 / (Sxx * ((yv - yv.mean()) ** 2).sum())), 1e-10) and
      rel(sqrt(((yv - lr1.intercept - lr1.slope * xs1) ** 2).sum() / (nr - 2) / Sxx), lr1.stderr, 1e-10))
# 信頼区間・予測区間の被覆率
x0 = np.array([1, 0.5, -1, 0.3]); beta_true = np.array([1, 0.5, -0.3, 0.0]); cm_, cp_ = 0, 0
Xf = np.column_stack([np.ones(20), rng.normal(size=(20, 3))]); XtXf = np.linalg.inv(Xf.T @ Xf); lev0 = x0 @ XtXf @ x0
tq = stats.t.ppf(0.975, 16)
for _ in range(20000):
    yy = Xf @ beta_true + rng.normal(size=20); bb_ = XtXf @ Xf.T @ yy; s_ = sqrt(((yy - Xf @ bb_) ** 2).sum() / 16)
    y0 = x0 @ bb_; ynew = x0 @ beta_true + rng.normal()
    cm_ += abs(y0 - x0 @ beta_true) <= tq * s_ * sqrt(lev0); cp_ += abs(y0 - ynew) <= tq * s_ * sqrt(1 + lev0)
check(S, "平均の信頼区間・予測区間の被覆率≈0.95（±0.007）", abs(cm_ / 20000 - 0.95) < 0.007 and abs(cp_ / 20000 - 0.95) < 0.007, (cm_ / 20000, cp_ / 20000))
Rj2 = 1 - ((Xd[:, 1] - Xd[:, [0, 2, 3]] @ np.linalg.lstsq(Xd[:, [0, 2, 3]], Xd[:, 1], rcond=None)[0]) ** 2).sum() / ((Xd[:, 1] - Xd[:, 1].mean()) ** 2).sum()
Rc = np.linalg.inv(np.corrcoef(Xd[:, 1:].T))
check(S, "VIF=1/(1-R_j²) = 相関行列の逆行列の対角", rel(1 / (1 - Rj2), Rc[0, 0], 1e-10), 1 / (1 - Rj2))
Zs_ = (Xd[:, 1:] - Xd[:, 1:].mean(0)) / Xd[:, 1:].std(0, ddof=1); ysd = (yv - yv.mean()) / yv.std(ddof=1)
bstd = np.linalg.lstsq(Zs_, ysd, rcond=None)[0]
check(S, "標準化偏回帰係数 = β̂j s_xj/s_y", np.allclose(bstd, bh_[1:] * Xd[:, 1:].std(0, ddof=1) / yv.std(ddof=1)))
lam_r = 2.0; brg = np.linalg.solve(Xd.T @ Xd + lam_r * np.eye(4), Xd.T @ yv)
obj = lambda bb: ((yv - Xd @ bb) ** 2).sum() + lam_r * (bb**2).sum()
check(S, "リッジ解 (X'X+λI)^{-1}X'y が罰則付き二乗和の最小点", rel(optimize.minimize(obj, np.zeros(4)).fun, obj(brg), 1e-6))
Wm = np.diag(rng.uniform(0.5, 2, nr)); bw = np.linalg.solve(Xd.T @ Wm @ Xd, Xd.T @ Wm @ yv)
Wh = np.sqrt(Wm)
check(S, "加重最小二乗 = √W を掛けた OLS", np.allclose(bw, np.linalg.lstsq(Wh @ Xd, Wh @ yv, rcond=None)[0]))

S = "回帰診断"
Hm = Xd @ XtXi @ Xd.T; hii = np.diag(Hm)
check(S, "Σh_ii=p+1、1/n≤h_ii≤1", rel(hii.sum(), pr_ + 1, 1e-10) and np.all(hii >= 1 / nr - 1e-12) and np.all(hii <= 1))
ri = ev / np.sqrt(s2r * (1 - hii))
Dd, Ti, Loo = [], [], []
for i in range(nr):
    keep = np.arange(nr) != i
    bi = np.linalg.lstsq(Xd[keep], yv[keep], rcond=None)[0]
    Dd.append(((yh - Xd @ bi) ** 2).sum() / ((pr_ + 1) * s2r))
    si = sqrt(((yv[keep] - Xd[keep] @ bi) ** 2).sum() / (nr - pr_ - 2))
    Ti.append(ev[i] / (si * sqrt(1 - hii[i]))); Loo.append(yv[i] - Xd[i] @ bi)
Ti = np.array(Ti)
check(S, "クックの距離：定義（1点除外）= r²/(p+1)·h/(1-h)", np.allclose(Dd, ri**2 / (pr_ + 1) * hii / (1 - hii)))
check(S, "一個抜き残差 = e_i/(1-h_ii)", np.allclose(Loo, ev / (1 - hii)))
dff = [(yh[i] - Xd[i] @ np.linalg.lstsq(Xd[np.arange(nr) != i], yv[np.arange(nr) != i], rcond=None)[0]) /
       (sqrt(((yv[np.arange(nr) != i] - Xd[np.arange(nr) != i] @ np.linalg.lstsq(Xd[np.arange(nr) != i], yv[np.arange(nr) != i], rcond=None)[0]) ** 2).sum() / (nr - pr_ - 2)) * sqrt(hii[i]))
       for i in range(nr)]
check(S, "DFFITS = t_i √(h/(1-h))", np.allclose(dff, Ti * np.sqrt(hii / (1 - hii))))
tsim = []
for _ in range(5000):
    yy = Xd @ beta_true + rng.normal(size=nr); ee = yy - Hm @ yy
    keep = np.arange(nr) != 0; bi = np.linalg.lstsq(Xd[keep], yy[keep], rcond=None)[0]
    tsim.append(ee[0] / (sqrt(((yy[keep] - Xd[keep] @ bi) ** 2).sum() / (nr - pr_ - 2)) * sqrt(1 - hii[0])))
check(S, "スチューデント化残差 ~ t(n-p-2)（KS p>0.01）", stats.kstest(tsim, stats.t(nr - pr_ - 2).cdf).pvalue > 0.01)
check(S, "V[e_i]=σ²(1-h_ii)（シミュ、i=0）", abs(np.var([(yy := Xd @ beta_true + rng.normal(size=nr))[0] - (Hm @ yy)[0] for _ in range(20000)]) / (1 - hii[0]) - 1) < 0.03)
et = np.zeros(2000); eps = rng.normal(size=2000)
for t_ in range(1, 2000):
    et[t_] = 0.6 * et[t_ - 1] + eps[t_]
dw = (np.diff(et) ** 2).sum() / (et**2).sum(); r1 = (et[1:] * et[:-1]).sum() / (et**2).sum()
check(S, "DW ≈ 2(1-ρ̂1)（差<0.01）", abs(dw - 2 * (1 - r1)) < 0.01, (dw, 2 * (1 - r1)))

# ================= モデル選択 =================
S = "モデル選択"
sig2 = Se / nr
ll = np.sum(stats.norm.logpdf(yv, yh, sqrt(sig2)))
check(S, "正規回帰の -2ℓ = n log(2πS_e/n) + n", rel(-2 * ll, nr * log(2 * pi * Se / nr) + nr, 1e-10), -2 * ll)
check(S, "BIC の罰則が AIC より重いのは n≥8（log n>2）", log(7) < 2 < log(8))
kc = 5; nc = 40
check(S, "補正AIC = -2ℓ+2kn/(n-k-1) と同じ", rel(2 * kc + 2 * kc * (kc + 1) / (nc - kc - 1), 2 * kc * nc / (nc - kc - 1), 1e-12))
cpv = []
for _ in range(20000):
    yy = Xd @ beta_true + rng.normal(size=nr)
    sf = ((yy - Hm @ yy) ** 2).sum() / (nr - 4)
    Hs = Xd[:, :3] @ np.linalg.inv(Xd[:, :3].T @ Xd[:, :3]) @ Xd[:, :3].T
    cpv.append(((yy - Hs @ yy) ** 2).sum() / sf - nr + 2 * 3)
check(S, "Cp：正しいモデル（係数3個）で平均≈p=3（±0.2）", abs(np.mean(cpv) - 3) < 0.2, np.mean(cpv))
kl = integrate.quad(lambda x: stats.norm.pdf(x) * (stats.norm.logpdf(x) - stats.norm.logpdf(x, 1, 2)), -30, 30)[0]
check(S, "KL 情報量 ≥ 0（N(0,1) と N(1,4) で閉形式 log2+(1+1)/8-1/2 と一致）", kl > 0 and rel(kl, log(2) + 2 / 8 - 0.5, 1e-8), kl)

# ================= 分散分析 =================
S = "分散分析"
gs = [rng.normal(m, 1, k) for m, k in [(0, 5), (0.5, 7), (1.2, 6), (0.3, 8)]]
ya = np.concatenate(gs); Na = len(ya); aa = len(gs)
SA = sum(len(g) * (g.mean() - ya.mean()) ** 2 for g in gs); SE_ = sum(((g - g.mean()) ** 2).sum() for g in gs)
check(S, "一元配置 S_T=S_A+S_E、F が f_oneway と一致", rel(((ya - ya.mean()) ** 2).sum(), SA + SE_, 1e-10) and
      rel((SA / (aa - 1)) / (SE_ / (Na - aa)), stats.f_oneway(*gs).statistic, 1e-10))
A_, B_, rp = 3, 4, 2
Y3 = rng.normal(size=(A_, B_, rp)) + np.arange(A_)[:, None, None] * 0.5
gm = Y3.mean(); ma = Y3.mean((1, 2)); mb_ = Y3.mean((0, 2)); mab = Y3.mean(2)
SA2 = B_ * rp * ((ma - gm) ** 2).sum(); SB2 = A_ * rp * ((mb_ - gm) ** 2).sum()
SAB = rp * ((mab - ma[:, None] - mb_[None, :] + gm) ** 2).sum(); SE2 = ((Y3 - mab[:, :, None]) ** 2).sum()
check(S, "二元配置（繰り返しあり）の平方和の分解と自由度の和", rel(((Y3 - gm) ** 2).sum(), SA2 + SB2 + SAB + SE2, 1e-10) and
      (A_ - 1) + (B_ - 1) + (A_ - 1) * (B_ - 1) + A_ * B_ * (rp - 1) == A_ * B_ * rp - 1)
kL = 5
check(S, "ラテン方格の自由度 3(k-1)+(k-1)(k-2)=k²-1", 3 * (kL - 1) + (kL - 1) * (kL - 2) == kL**2 - 1)
msa_r, mse_r = [], []
for _ in range(20000):
    eff = rng.normal(0, 0.8, 4); yy = eff[:, None] + rng.normal(0, 1, (4, 6))
    msa_r.append(6 * ((yy.mean(1) - yy.mean()) ** 2).sum() / 3); mse_r.append(((yy - yy.mean(1, keepdims=True)) ** 2).sum() / 20)
check(S, "変量模型 E[MS_A]=σ²+nσ_A²、E[MS_E]=σ²（相対差<3%）", abs(np.mean(msa_r) / (1 + 6 * 0.64) - 1) < 0.03 and abs(np.mean(mse_r) - 1) < 0.02, (np.mean(msa_r), np.mean(mse_r)))
alpha_f = np.array([-0.5, 0, 0.2, 0.3]); msa_f = []
for _ in range(20000):
    yy = alpha_f[:, None] + rng.normal(0, 1, (4, 6)); msa_f.append(6 * ((yy.mean(1) - yy.mean()) ** 2).sum() / 3)
check(S, "母数模型 E[MS_A]=σ²+nΣα²/(a-1)（相対差<3%）", abs(np.mean(msa_f) / (1 + 6 * (alpha_f**2).sum() / 3) - 1) < 0.03)
check(S, "テューキーの q は studentized_range と比べる（q_{0.05}(4,20)≈3.958）", abs(stats.studentized_range.ppf(0.95, 4, 20) - 3.958) < 0.002)

# ================= 標本調査 =================
S = "標本調査"
pop = np.array([3, 5, 6, 8, 9, 11.0]); Np, nps = len(pop), 3
S2 = pop.var(ddof=1); means = np.array([pop[list(c)].mean() for c in combinations(range(Np), nps)])
check(S, "単純無作為抽出 V(ȳ)=(1-f)S²/n（全列挙）", rel(means.var(), (1 - nps / Np) * S2 / nps, 1e-12), means.var())
check(S, "σ² 版 (N-n)/(N-1)·σ²/n と同じ", rel((Np - nps) / (Np - 1) * pop.var() / nps, (1 - nps / Np) * S2 / nps, 1e-12))
s2s = np.array([pop[list(c)].var(ddof=1) for c in combinations(range(Np), nps)])
check(S, "E[s²]=S²（全列挙）", rel(s2s.mean(), S2, 1e-12))
popb = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0.]); Pb = popb.mean(); Nb, nb_ = 10, 4
pv_ = np.array([popb[list(c)].mean() for c in combinations(range(Nb), nb_)])
check(S, "母比率 V(p̂)=(N-n)/(N-1)·P(1-P)/n（全列挙）", rel(pv_.var(), (Nb - nb_) / (Nb - 1) * Pb * (1 - Pb) / nb_, 1e-12))
est = np.array([(1 - nb_ / Nb) * pp_ * (1 - pp_) / (nb_ - 1) for pp_ in pv_])
check(S, "V(p̂) の不偏推定 (1-f)p̂(1-p̂)/(n-1)（全列挙）", rel(est.mean(), pv_.var(), 1e-12))
n0 = (1.96 * 10 / 2) ** 2; Nf = 500; nfp = n0 / (1 + n0 / Nf)
check(S, "有限母集団の標本サイズ n0/(1+n0/N) で誤差幅がちょうど E", rel(1.96 * sqrt((1 - nfp / Nf) * 100 / nfp), 2, 1e-12), nfp)
st1 = np.array([2, 4, 5, 7.0]); st2 = np.array([10, 13, 15, 16, 20, 22.0]); n1s, n2s = 2, 3
Wv = np.array([4, 6]) / 10
ests = [Wv[0] * st1[list(c1)].mean() + Wv[1] * st2[list(c2)].mean() for c1 in combinations(range(4), n1s) for c2 in combinations(range(6), n2s)]
Vst = Wv[0] ** 2 * (1 - n1s / 4) * st1.var(ddof=1) / n1s + Wv[1] ** 2 * (1 - n2s / 6) * st2.var(ddof=1) / n2s
check(S, "層化抽出 V(ȳ_st) の式（全列挙）、不偏", rel(np.var(ests), Vst, 1e-12) and rel(np.mean(ests), np.concatenate([st1, st2]).mean(), 1e-12), Vst)
Nh = np.array([400, 300, 300]); Sh = np.array([5.0, 10.0, 20.0]); Wh_ = Nh / Nh.sum(); nT_ = 100; NT = Nh.sum()


def vst(nh):
    return (Wh_**2 * (1 - nh / Nh) * Sh**2 / nh).sum()


nprop = nT_ * Wh_; nney = nT_ * Nh * Sh / (Nh * Sh).sum()
check(S, "比例配分の分散 (1-f)/n ΣW_hS_h²", rel(vst(nprop), (1 - nT_ / NT) / nT_ * (Wh_ * Sh**2).sum(), 1e-12))
check(S, "ネイマン配分の分散 (ΣW_hS_h)²/n - ΣW_hS_h²/N", rel(vst(nney), (Wh_ * Sh).sum() ** 2 / nT_ - (Wh_ * Sh**2).sum() / NT, 1e-12))
resn = optimize.minimize(lambda z_: vst(np.append(z_, nT_ - z_.sum())), nprop[:2], bounds=[(1, 98), (1, 98)])
check(S, "ネイマン配分が総数固定で分散最小（数値最適化と一致）", rel(resn.fun, vst(nney), 1e-5), resn.fun)
yp = np.array([4.0, 7, 2, 9, 5]); pis = np.array([0.4, 0.6, 0.2, 0.6, 0.2])  # Σπ=2 → n=2 の非等確率抽出
# π_i を持つ設計：全ペアに確率を振り、包含確率が pis となる設計を作る
pairs = list(combinations(range(5), 2))
Aeq = np.array([[1.0 if i in pr else 0.0 for pr in pairs] for i in range(5)])
sol = optimize.linprog(np.zeros(len(pairs)), A_eq=np.vstack([Aeq, np.ones(len(pairs))]), b_eq=np.append(pis, 1), bounds=[(0, 1)] * len(pairs))
ht = sum(pr_p * sum(yp[i] / pis[i] for i in pr) for pr_p, pr in zip(sol.x, pairs))
check(S, "ホルビッツ・トンプソン推定量の不偏性（非等確率の設計を全列挙）", sol.success and rel(ht, yp.sum(), 1e-8), ht)
Npop = 2000
xpop = rng.gamma(4, 5, Npop); ypop = 2 * xpop + rng.normal(0, 4, Npop); Rr = ypop.mean() / xpop.mean(); nrs = 100
rat = []; reg = []
bpop = np.cov(xpop, ypop)[0, 1] / xpop.var(ddof=1)
for _ in range(20000):
    ii = rng.choice(Npop, nrs, replace=False)
    rat.append(ypop[ii].mean() / xpop[ii].mean() * xpop.mean())
    reg.append(ypop[ii].mean() + bpop * (xpop.mean() - xpop[ii].mean()))
Vr = (1 - nrs / Npop) / nrs * (ypop.var(ddof=1) - 2 * Rr * np.cov(xpop, ypop)[0, 1] + Rr**2 * xpop.var(ddof=1))
rho_p = np.corrcoef(xpop, ypop)[0, 1]; Vg = (1 - nrs / Npop) / nrs * ypop.var(ddof=1) * (1 - rho_p**2)
check(S, "比推定の近似分散（シミュ相対差<5%）", abs(np.var(rat) / Vr - 1) < 0.05, (np.var(rat), Vr))
check(S, "回帰推定の近似分散 (1-f)/n S_y²(1-ρ²)（係数既知、シミュ相対差<5%）", abs(np.var(reg) / Vg - 1) < 0.05, (np.var(reg), Vg))
Mc, rho_c, ncl = 10, 0.2, 2000
cl_ = rng.normal(0, sqrt(rho_c), (ncl, 1)) + rng.normal(0, sqrt(1 - rho_c), (ncl, Mc))
check(S, "集落抽出の計画効果 ≈ 1+(M-1)ρ（シミュ相対差<5%）", abs(cl_.mean(1).var() / (1 / Mc) / (1 + (Mc - 1) * rho_c) - 1) < 0.05, cl_.mean(1).var() * Mc)

# ================= 多変量解析 =================
S = "多変量解析"
Sg = np.array([[4, 1.2, 0.8], [1.2, 2, 0.3], [0.8, 0.3, 1]])
lamv, vec = np.linalg.eigh(Sg); lamv, vec = lamv[::-1], vec[:, ::-1]
check(S, "主成分 V[Z_j]=λ_j、固有値の和=trΣ", np.allclose([vec[:, j] @ Sg @ vec[:, j] for j in range(3)], lamv) and rel(lamv.sum(), np.trace(Sg), 1e-12))
Zx = rng.multivariate_normal([0, 0, 0], Sg, 500000); Zp = Zx @ vec
check(S, "主成分負荷量 Corr(Z_1,X_k)=√λ1 a_1k/σ_k（シミュ差<0.005）",
      np.max(np.abs([np.corrcoef(Zp[:, 0], Zx[:, kq])[0, 1] - sqrt(lamv[0]) * vec[kq, 0] / sqrt(Sg[kq, kq]) for kq in range(3)])) < 0.005)
Rm = Sg / np.sqrt(np.outer(np.diag(Sg), np.diag(Sg)))
check(S, "相関行列の固有値の和 = p", rel(np.linalg.eigvalsh(Rm).sum(), 3, 1e-12))
D2 = np.einsum("ij,jk,ik->i", Zx, np.linalg.inv(Sg), Zx)
check(S, "マハラノビス距離 D² ~ χ²(p)（平均≈p、95%点の超過率≈0.05）", abs(D2.mean() - 3) < 0.02 and abs(np.mean(D2 > stats.chi2.ppf(0.95, 3)) - 0.05) < 0.002)
m1v, m2v = np.array([1.0, 0.5]), np.array([-0.5, 0.0]); Sd = np.array([[1, 0.3], [0.3, 0.8]]); Si = np.linalg.inv(Sd)
Delta = sqrt((m1v - m2v) @ Si @ (m1v - m2v))
X1 = rng.multivariate_normal(m1v, Sd, 300000); X2 = rng.multivariate_normal(m2v, Sd, 300000)
hf = lambda Xs: (Xs - (m1v + m2v) / 2) @ Si @ (m1v - m2v)
err = 0.5 * np.mean(hf(X1) < 0) + 0.5 * np.mean(hf(X2) > 0)
check(S, "線形判別の誤判別率 Φ(-Δ/2)（シミュ差<0.003）", abs(err - stats.norm.cdf(-Delta / 2)) < 0.003, (err, stats.norm.cdf(-Delta / 2)))
x_t = np.array([0.4, 0.1]); p1_ = 0.3
post = p1_ * stats.multivariate_normal(m1v, Sd).pdf(x_t) / (p1_ * stats.multivariate_normal(m1v, Sd).pdf(x_t) + (1 - p1_) * stats.multivariate_normal(m2v, Sd).pdf(x_t))
check(S, "判別関数 h(x)+log(π1/π2) = 事後確率の対数オッズ", rel(hf(x_t) + log(p1_ / (1 - p1_)), log(post / (1 - post)), 1e-9))
xa_, xb2 = X1[:200], X2[:300]
Sw = np.cov(xa_.T) * 199 + np.cov(xb2.T) * 299; wv = np.linalg.solve(Sw, xa_.mean(0) - xb2.mean(0))
fisher_ratio = lambda w: (w @ (xa_.mean(0) - xb2.mean(0))) ** 2 / (w @ Sw @ w)
check(S, "フィッシャーの判別方向 S_W^{-1}(x̄1-x̄2) が分離度の最大点", all(fisher_ratio(wv) >= fisher_ratio(wv + rng.normal(0, 0.3, 2)) for _ in range(200)))
TP, FP, FN, TN = 40, 10, 20, 130
prc, rec = TP / (TP + FP), TP / (TP + FN)
check(S, "F値 = 2PR/(P+R) = 2TP/(2TP+FP+FN)", rel(2 * prc * rec / (prc + rec), 2 * TP / (2 * TP + FP + FN), 1e-12))
Lf = np.array([[0.8, 0.1], [0.7, 0.3], [0.2, 0.9]]); Psi = np.diag(1 - (Lf**2).sum(1))
Ff = rng.normal(size=(400000, 2)) @ Lf.T + rng.normal(size=(400000, 3)) * np.sqrt(np.diag(Psi))
check(S, "因子分析 Σ=ΛΛ'+Ψ（共通性+独自分散=1、シミュ差<0.01）", np.max(np.abs(np.cov(Ff.T) - (Lf @ Lf.T + Psi))) < 0.01)
Aw = rng.normal(size=(5, 2)); Bw = rng.normal(size=(7, 2)) + 2
wss = lambda M_: ((M_ - M_.mean(0)) ** 2).sum()
check(S, "ウォード法の増加量 n_An_B/(n_A+n_B)||差||²", rel(wss(np.vstack([Aw, Bw])) - wss(Aw) - wss(Bw), 5 * 7 / 12 * ((Aw.mean(0) - Bw.mean(0)) ** 2).sum(), 1e-10))

# ================= 時系列解析 =================
S = "時系列解析"


def acf(x_, h_):
    x_ = x_ - x_.mean()
    return (x_[h_:] * x_[:-h_]).sum() / (x_**2).sum()


Tn = 400000; eps = rng.normal(size=Tn)
phi, cc = 0.7, 1.5
ar = np.zeros(Tn); ar[0] = cc / (1 - phi)
for t_ in range(1, Tn):
    ar[t_] = cc + phi * ar[t_ - 1] + eps[t_]
check(S, "AR(1) 平均 c/(1-φ)・分散 σ²/(1-φ²)・ρ(h)=φ^h", abs(ar.mean() - 5) < 0.03 and abs(ar.var() / (1 / (1 - phi**2)) - 1) < 0.02
      and all(abs(acf(ar, hh_) - phi**hh_) < 0.01 for hh_ in (1, 2, 3)), (ar.mean(), ar.var(), acf(ar, 2)))
p1, p2 = 0.5, 0.3
a2 = np.zeros(Tn)
for t_ in range(2, Tn):
    a2[t_] = p1 * a2[t_ - 1] + p2 * a2[t_ - 2] + eps[t_]
r1t = p1 / (1 - p2); r2t = p1 * r1t + p2
check(S, "AR(2) ρ(1)=φ1/(1-φ2)、ρ(2)=φ1ρ(1)+φ2（シミュ差<0.01）", abs(acf(a2, 1) - r1t) < 0.01 and abs(acf(a2, 2) - r2t) < 0.01, (acf(a2, 1), r1t))


def ar2_stationary_roots(f1, f2):
    return np.all(np.abs(np.roots([-f2, -f1, 1])) > 1)


grid = [(f1, f2) for f1 in np.linspace(-2.4, 2.4, 49) for f2 in np.linspace(-1.4, 1.4, 29)]
check(S, "AR(2) の定常三角形 ⇔ 特性根が単位円の外（格子で一致）",
      all(ar2_stationary_roots(f1, f2) == ((f1 + f2 < 1) and (f2 - f1 < 1) and (abs(f2) < 1)) for f1, f2 in grid
          if min(abs(f1 + f2 - 1), abs(f2 - f1 - 1), abs(abs(f2) - 1)) > 1e-6))
th_ = 0.6
ma = eps[1:] + th_ * eps[:-1]
check(S, "MA(1) γ(0)=σ²(1+θ²)、ρ(1)=θ/(1+θ²)、ρ(2)=0", abs(ma.var() - (1 + th_**2)) < 0.01 and abs(acf(ma, 1) - th_ / (1 + th_**2)) < 0.005 and abs(acf(ma, 2)) < 0.005)
tg = np.linspace(-5, 5, 10001)
check(S, "MA(1) の |ρ(1)|≤1/2（θ=±1 で最大）", np.max(np.abs(tg / (1 + tg**2))) <= 0.5 + 1e-12)
pac2 = (acf(ar, 2) - acf(ar, 1) ** 2) / (1 - acf(ar, 1) ** 2)
check(S, "AR(1) の偏自己相関はラグ2で≈0", abs(pac2) < 0.01, pac2)
rw = np.cumsum(rng.normal(size=(20000, 50)), axis=1)
check(S, "ランダムウォーク V[X_t]=tσ²（t=50、相対差<3%）", abs(rw[:, -1].var() / 50 - 1) < 0.03)
hz = 3; Tsim = 200000
start = rng.normal(5, sqrt(1 / (1 - phi**2)), Tsim); xx = start.copy()
for _ in range(hz):
    xx = cc + phi * xx + rng.normal(size=Tsim)
pred = 5 + phi**hz * (start - 5)
check(S, "AR(1) h期先予測 μ+φ^h(X_T-μ)、誤差分散 σ²(1-φ^{2h})/(1-φ²)（h=3）",
      abs((xx - pred).mean()) < 0.01 and abs((xx - pred).var() / ((1 - phi ** (2 * hz)) / (1 - phi**2)) - 1) < 0.02)

# ================= 分割表 =================
S = "分割表"
pr4 = np.array([0.15, 0.25, 0.1, 0.5]); nmt = 800
cells = rng.multinomial(nmt, pr4, 100000)
cells = cells[(cells > 0).all(1)]
lor = np.log(cells[:, 0] * cells[:, 3] / (cells[:, 1] * cells[:, 2]))
se_th = sqrt((1 / (nmt * pr4)).sum())
check(S, "SE(log OR) = √(1/a+1/b+1/c+1/d)（シミュ相対差<3%）", abs(lor.std() / se_th - 1) < 0.03, (lor.std(), se_th))
r1n, r2n = 300, 500; p1r, p2r = 0.3, 0.15
A1 = rng.binomial(r1n, p1r, 100000); C1 = rng.binomial(r2n, p2r, 100000)
lrr = np.log((A1 / r1n) / (C1 / r2n))
se_rr = sqrt(1 / (r1n * p1r) - 1 / r1n + 1 / (r2n * p2r) - 1 / r2n)
check(S, "SE(log RR) = √(1/a-1/(a+b)+1/c-1/(c+d))（シミュ相対差<3%）", abs(lrr.std() / se_rr - 1) < 0.03, (lrr.std(), se_rr))
check(S, "リスク差の SE（シミュ相対差<3%）", abs((A1 / r1n - C1 / r2n).std() / sqrt(p1r * (1 - p1r) / r1n + p2r * (1 - p2r) / r2n) - 1) < 0.03)
check(S, "φ² = χ²/n（2×2）", rel(((a_ * d_ - b_ * c_) / sqrt(den)) ** 2, stats.chi2_contingency(tab, correction=False)[0] / nt, 1e-12))
chi3 = stats.chi2_contingency(tab3, correction=False)[0]
check(S, "クラメールの V = scipy.contingency.association", rel(sqrt(chi3 / (tab3.sum() * 2)), stats.contingency.association(tab3, method="cramer"), 1e-12))
strata = [np.array([[10, 20], [5, 30]]), np.array([[8, 12], [6, 25]]), np.array([[15, 10], [9, 14]])]
mh = sum(t[0, 0] * t[1, 1] / t.sum() for t in strata) / sum(t[0, 1] * t[1, 0] / t.sum() for t in strata)
check(S, "MH 共通オッズ比：全層で OR が等しい表なら その OR に一致", rel(sum(t[0, 0] * t[1, 1] / t.sum() for t in [np.array([[4, 2], [2, 4]]), np.array([[8, 1], [2, 1]])]) /
      sum(t[0, 1] * t[1, 0] / t.sum() for t in [np.array([[4, 2], [2, 4]]), np.array([[8, 1], [2, 1]])]), 4.0, 1e-12) and mh > 0, mh)
rs_ = tab3.sum(1, keepdims=True) / tab3.sum(); cs_ = tab3.sum(0, keepdims=True) / tab3.sum()
adj = (tab3 - Eij) / np.sqrt(Eij * (1 - rs_) * (1 - cs_))
pi_r = np.array([0.3, 0.3, 0.4]); pi_c = np.array([0.2, 0.5, 0.3]); adjs = []
for _ in range(20000):
    tt = rng.multinomial(300, np.outer(pi_r, pi_c).ravel()).reshape(3, 3)
    ee_ = tt.sum(1, keepdims=True) * tt.sum(0, keepdims=True) / 300
    adjs.append(((tt - ee_) / np.sqrt(ee_ * (1 - tt.sum(1, keepdims=True) / 300) * (1 - tt.sum(0, keepdims=True) / 300)))[0, 0])
check(S, "調整済み残差の分散≈1（独立のもと、相対差<3%）", abs(np.var(adjs) - 1) < 0.03, np.var(adjs))
mu_ind = Eij
check(S, "独立モデル log μ_ij = λ+λ_i+λ_j（期待度数の対数の交互作用が0）",
      np.allclose(np.log(mu_ind) - np.log(mu_ind).mean(1, keepdims=True) - np.log(mu_ind).mean(0, keepdims=True) + np.log(mu_ind).mean(), 0))

# ================= ベイズ法 =================
S = "ベイズ法"
grid = np.linspace(1e-6, 1 - 1e-6, 200001)


def grid_mean(post, g):
    w = post / post.sum()
    return (g * w).sum()


aB, bB, nB, xB = 2.0, 3.0, 20, 7
post = grid**(aB - 1) * (1 - grid)**(bB - 1) * grid**xB * (1 - grid)**(nB - xB)
check(S, "ベータ・二項：事後平均 (a+x)/(a+b+n)", rel(grid_mean(post, grid), (aB + xB) / (aB + bB + nB), 1e-6))
check(S, "ベータ・二項：MAP (a'-1)/(a'+b'-2)", abs(grid[np.argmax(post)] - (aB + xB - 1) / (aB + bB + nB - 2)) < 1e-5)
xsg = np.array([0, 2, 1, 4, 0, 3])
postg = grid**(aB - 1) * (1 - grid)**(bB - 1) * np.prod([grid * (1 - grid)**xv_ for xv_ in xsg], axis=0)
check(S, "幾何・ベータ：事後 Be(a+n, b+Σx) の平均", rel(grid_mean(postg, grid), (aB + 6) / (aB + bB + 6 + xsg.sum()), 1e-6))
lg = np.linspace(1e-6, 30, 300001); aG, bG = 3.0, 2.0  # レート表記
xp = np.array([4, 2, 5, 3, 6])
postp = lg**(aG - 1) * np.exp(-bG * lg) * lg**xp.sum() * np.exp(-len(xp) * lg)
check(S, "ガンマ・ポアソン（レート）：事後平均 (α+Σx)/(β+n)", rel(grid_mean(postp, lg), (aG + xp.sum()) / (bG + len(xp)), 1e-6))
xe = np.array([0.5, 1.2, 0.3, 2.0])
poste = lg**(aG - 1) * np.exp(-bG * lg) * lg**len(xe) * np.exp(-xe.sum() * lg)
check(S, "ガンマ・指数（レート）：事後平均 (α+n)/(β+Σx)", rel(grid_mean(poste, lg), (aG + len(xe)) / (bG + xe.sum()), 1e-6))
mu0_, t0, sgn, nn_, xbar = 1.0, 2.0, 3.0, 10, 2.5
mg = np.linspace(-20, 20, 400001)
postn = stats.norm.pdf(mg, mu0_, t0) * stats.norm.pdf(xbar, mg, sgn / sqrt(nn_))
tn2 = 1 / (1 / t0**2 + nn_ / sgn**2); mun = tn2 * (mu0_ / t0**2 + nn_ * xbar / sgn**2)
w_ = postn / postn.sum()
check(S, "正規・正規：事後平均と事後分散", rel((mg * w_).sum(), mun, 1e-6) and rel(((mg - mun) ** 2 * w_).sum(), tn2, 1e-5), (mun, tn2))
a1, b1 = aG + xp.sum(), bG + len(xp)
pred = [integrate.quad(lambda l: stats.poisson.pmf(k_, l) * stats.gamma(a1, scale=1 / b1).pdf(l), 0, 60)[0] for k_ in range(6)]
check(S, "ガンマ・ポアソンの予測分布 = NB(α', β'/(β'+1))（失敗回数）", np.allclose(pred, stats.nbinom.pmf(range(6), a1, b1 / (b1 + 1)), atol=1e-9))
check(S, "ジェフリーズ：ベルヌーイ √I ∝ p^{-1/2}(1-p)^{-1/2} = Be(1/2,1/2) の形",
      rel(sqrt(1 / (0.3 * 0.7)) / sqrt(1 / (0.6 * 0.4)), stats.beta(0.5, 0.5).pdf(0.3) / stats.beta(0.5, 0.5).pdf(0.6), 1e-12))
m1_ = integrate.quad(lambda p_: stats.binom.pmf(xB, nB, p_), 0, 1)[0]; m0_ = stats.binom.pmf(xB, nB, 0.5)
pM1 = m1_ * 0.5 / (m1_ * 0.5 + m0_ * 0.5)
check(S, "事後オッズ = ベイズファクター × 事前オッズ", rel(pM1 / (1 - pM1), (m1_ / m0_) * 1, 1e-10), m1_ / m0_)

# ================= マルコフ連鎖 =================
S = "マルコフ連鎖"
Pm = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
check(S, "チャップマン・コルモゴロフ P^(5)=P^(2)P^(3)", np.allclose(np.linalg.matrix_power(Pm, 5), np.linalg.matrix_power(Pm, 2) @ np.linalg.matrix_power(Pm, 3)))
w, vl = np.linalg.eig(Pm.T); piv = np.real(vl[:, np.argmin(np.abs(w - 1))]); piv /= piv.sum()
check(S, "定常分布 πP=π、P^n の各行が π に収束", np.allclose(piv @ Pm, piv) and np.allclose(np.linalg.matrix_power(Pm, 200), np.tile(piv, (3, 1))), piv)
aa_, bb2 = 0.3, 0.1; P2 = np.array([[1 - aa_, aa_], [bb2, 1 - bb2]])
check(S, "2状態の定常分布 (b/(a+b), a/(a+b))", np.allclose(np.array([bb2, aa_]) / (aa_ + bb2) @ P2, np.array([bb2, aa_]) / (aa_ + bb2)))
# 平均再帰時間：状態0へ戻るまでの期待時間を連立方程式で
Q = Pm[1:, 1:]; tt_ = np.linalg.solve(np.eye(2) - Q, np.ones(2)); m0 = 1 + Pm[0, 1:] @ tt_
check(S, "平均再帰時間 m_i = 1/π_i", rel(m0, 1 / piv[0], 1e-10), m0)
Pb_ = np.array([[0.6, 0.4, 0], [0.2, 0.5, 0.3], [0, 0.5, 0.5]])
wb, vb = np.linalg.eig(Pb_.T); pib = np.real(vb[:, np.argmin(np.abs(wb - 1))]); pib /= pib.sum()
check(S, "詳細つり合い（三重対角）π_i p_ij = π_j p_ji", np.allclose(pib[:, None] * Pb_, (pib[:, None] * Pb_).T))
Nr, pr_w = 6, 0.45; qw = 1 - pr_w
Ar = np.eye(Nr + 1); br_ = np.zeros(Nr + 1); br_[Nr] = 1
for i in range(1, Nr):
    Ar[i, i - 1] = -qw; Ar[i, i + 1] = -pr_w
hs = np.linalg.solve(Ar, br_)
check(S, "破産問題 P(N に到達)=(1-(q/p)^k)/(1-(q/p)^N)（連立方程式と一致）",
      np.allclose(hs, [(1 - (qw / pr_w) ** k_) / (1 - (qw / pr_w) ** Nr) for k_ in range(Nr + 1)]))
Ad = np.eye(Nr + 1); bd = np.ones(Nr + 1); bd[0] = bd[Nr] = 0
for i in range(1, Nr):
    Ad[i, i - 1] = -0.5; Ad[i, i + 1] = -0.5
check(S, "p=1/2 の平均時間 k(N-k)、到達確率 k/N", np.allclose(np.linalg.solve(Ad, bd), [k_ * (Nr - k_) for k_ in range(Nr + 1)])
      and np.allclose(np.linalg.solve(np.where(np.arange(Nr + 1)[:, None] == np.arange(Nr + 1), 1.0, 0) + (Ad - np.eye(Nr + 1)), br_), np.arange(Nr + 1) / Nr))

# ================= 確率過程 =================
S = "確率過程"
lmb, tT = 2.0, 3.0
arr = rng.exponential(1 / lmb, (100000, 40)).cumsum(axis=1)
Nt = (arr <= tT).sum(axis=1)
check(S, "到着間隔 Exp(λ) → N(t)~Po(λt)（平均・分散≈6）", abs(Nt.mean() - 6) < 0.03 and abs(Nt.var() - 6) < 0.1, (Nt.mean(), Nt.var()))
check(S, "n番目の到着 S_n ~ Ga(n,1/λ)（n=5、KS p>0.01）", stats.kstest(arr[:, 4], stats.gamma(5, scale=1 / lmb).cdf).pvalue > 0.01)
thin = rng.binomial(Nt, 0.3)
check(S, "間引き：強度 λp（平均・分散≈1.8）", abs(thin.mean() - 1.8) < 0.02 and abs(thin.var() - 1.8) < 0.05)
sup = rng.poisson(1.0 * tT, 100000) + rng.poisson(2.5 * tT, 100000)
check(S, "合成：強度 λ1+λ2（分散=平均≈10.5）", abs(sup.mean() - 10.5) < 0.05 and abs(sup.var() - 10.5) < 0.15)
sel = arr[Nt == 6][:, :6]
check(S, "N(t)=n のもとで到着時刻は一様分布の順序統計量（1番目の平均 t/(n+1)）", abs(sel[:, 0].mean() - tT / 7) < 0.01, sel[:, 0].mean())
Ncp = rng.poisson(lmb * tT, 200000); Ycp = np.array([rng.gamma(2, 1.5, k_).sum() for k_ in Ncp[:50000]])
EY, EY2 = 3.0, 2 * 1.5**2 + 9.0
check(S, "複合ポアソン E=λtE[Y]、V=λtE[Y²]（相対差<3%）", abs(Ycp.mean() / (6 * EY) - 1) < 0.03 and abs(Ycp.var() / (6 * EY2) - 1) < 0.03, (Ycp.mean(), Ycp.var()))
steps = 2000; Bm = np.cumsum(rng.normal(0, sqrt(1 / steps), (20000, steps)), axis=1)
check(S, "ブラウン運動 Cov(B(s),B(t))=min(s,t)（s=0.3,t=0.8）", abs(np.cov(Bm[:, 599], Bm[:, 1599])[0, 1] - 0.3) < 0.015)
pmax = np.mean(Bm.max(axis=1) >= 1.0)
check(S, "反射原理 P(max B≥a)=2P(B(1)≥a)（a=1、離散化の偏りを考えて差<0.015）", abs(pmax - 2 * stats.norm.sf(1.0)) < 0.015, (pmax, 2 * stats.norm.sf(1.0)))
la_, mu_q = 0.8, 1.0; rq = la_ / mu_q
ncust = 400000; ia = rng.exponential(1 / la_, ncust); sv = rng.exponential(1 / mu_q, ncust)
wq = np.zeros(ncust)
for i in range(1, ncust):
    wq[i] = max(0.0, wq[i - 1] + sv[i - 1] - ia[i])
check(S, "M/M/1 W_q=ρ/(μ-λ)、W=1/(μ-λ)（リンドレー漸化式、相対差<5%）",
      abs(wq[10000:].mean() / (rq / (mu_q - la_)) - 1) < 0.05 and abs((wq + sv)[10000:].mean() / (1 / (mu_q - la_)) - 1) < 0.05, wq[10000:].mean())
check(S, "M/M/1 L=ρ/(1-ρ)=Σn(1-ρ)ρ^n、リトル L=λW、L_q=λW_q", rel(sum(k_ * (1 - rq) * rq**k_ for k_ in range(2000)), rq / (1 - rq), 1e-10)
      and rel(la_ / (mu_q - la_), rq / (1 - rq), 1e-12) and rel(la_ * rq / (mu_q - la_), rq**2 / (1 - rq), 1e-12))
lamb_bd = [0.8] * 30; mu_bd = [1.0] * 30
pib_ = np.array([np.prod([lamb_bd[k_ - 1] / mu_bd[k_ - 1] for k_ in range(1, n_ + 1)]) for n_ in range(30)])
check(S, "出生死滅過程の定常分布の積の式（M/M/1 で (1-ρ)ρ^n）", np.allclose(pib_ / pib_.sum(), (1 - rq) * rq ** np.arange(30) / (1 - rq**30)))

# ================= 生存・欠測 =================
S = "生存・欠測"
lam_h = 0.5
check(S, "S(t)=exp(-∫h)（ワイブル k=1.7,η=2.5 の h=f/S）", rel(exp(-integrate.quad(lambda u: stats.weibull_min(1.7, scale=2.5).pdf(u) / stats.weibull_min(1.7, scale=2.5).sf(u), 0, 2.0)[0]),
                                                      stats.weibull_min(1.7, scale=2.5).sf(2.0), 1e-8))
tv = np.sort(rng.exponential(2, 30)); km = np.cumprod([1 - 1 / (30 - i) for i in range(30)])
check(S, "打ち切りなしなら KM = 経験生存関数", np.allclose(km, 1 - np.arange(1, 31) / 30))
kmv = []
for _ in range(20000):
    T_ = rng.exponential(2.0, 60); Cc = rng.exponential(4.0, 60); obs_ = np.minimum(T_, Cc); dlt = T_ <= Cc
    o = np.argsort(obs_); obs_, dlt = obs_[o], dlt[o]; S_ = 1.0; gw = 0.0
    for i in range(60):
        if obs_[i] > 1.5:
            break
        ni = 60 - i
        if dlt[i]:
            S_ *= 1 - 1 / ni; gw += 1 / (ni * (ni - 1)) if ni > 1 else 0
    kmv.append((S_, S_**2 * gw))
kmv = np.array(kmv)
check(S, "KM はほぼ不偏、グリーンウッドの分散 ≈ 実際の分散（相対差<6%）", abs(kmv[:, 0].mean() - exp(-1.5 / 2)) < 0.005 and abs(kmv[:, 1].mean() / kmv[:, 0].var() - 1) < 0.06,
      (kmv[:, 0].mean(), kmv[:, 1].mean(), kmv[:, 0].var()))
Ql = np.array([1.2, 1.5, 1.1, 1.4, 1.3]); Ul = np.array([0.04, 0.05, 0.045, 0.05, 0.04])
Bq = Ql.var(ddof=1)
check(S, "ルービンの規則 T=Ū+(1+1/m)B", rel(Ul.mean() + 1.2 * Bq, Ul.mean() + (1 + 1 / 5) * ((Ql - Ql.mean()) ** 2).sum() / 4, 1e-12), Ul.mean() + 1.2 * Bq)

# ================= シミュレーション =================
S = "シミュレーション"
hm = rng.uniform(size=(2000, 500)) ** 2
check(S, "モンテカルロ積分 ∫x²=1/3、標準誤差 σ_h/√n", abs(hm.mean() - 1 / 3) < 0.001 and abs(hm.mean(1).std() / (sqrt(4 / 45) / sqrt(500)) - 1) < 0.05)
gs_ = rng.normal(4, 1, 400000)
isv = np.mean((gs_ > 4) * stats.norm.pdf(gs_) / stats.norm.pdf(gs_, 4, 1))
check(S, "重点サンプリング P(Z>4)（相対差<2%）", abs(isv / stats.norm.sf(4) - 1) < 0.02, isv)
cR = 1.5 / 1  # Be(2,2) の最大 1.5、提案 U(0,1)
Xp = rng.uniform(size=400000); acc = rng.uniform(size=400000) < stats.beta(2, 2).pdf(Xp) / cR
check(S, "棄却法：採択率 1/c、採択標本は目標分布", abs(acc.mean() - 1 / cR) < 0.003 and stats.kstest(Xp[acc], stats.beta(2, 2).cdf).pvalue > 0.01)
xm, ch = 0.0, []
for _ in range(200000):
    yprop = xm + rng.normal(0, 1)
    if rng.uniform() < min(1, exp(-0.5 * yprop**2 + 0.5 * xm**2)):
        xm = yprop
    ch.append(xm)
ch = np.array(ch[5000:])
check(S, "MH（対称提案）で N(0,1) が目標：平均≈0、分散≈1（±0.05）", abs(ch.mean()) < 0.05 and abs(ch.var() - 1) < 0.05, (ch.mean(), ch.var()))
xl, seen = 1, set()
for _ in range(20):
    xl = (5 * xl + 3) % 16; seen.add(xl)
check(S, "線形合同法 (a=5,c=3,m=16) は周期 m", len(seen) == 16)
dat = rng.exponential(1, 25); thf = dat.var()
jk = np.array([np.delete(dat, i).var() for i in range(25)])
check(S, "ジャックナイフの偏り補正で σ̂²(1/n版) が不偏分散に一致", rel(thf - 24 * (jk.mean() - thf), dat.var(ddof=1), 1e-10))
check(S, "ジャックナイフ分散：平均の場合 s²/n に一致", rel(24 / 25 * ((np.array([np.delete(dat, i).mean() for i in range(25)]) - np.mean([np.delete(dat, i).mean() for i in range(25)])) ** 2).sum(), dat.var(ddof=1) / 25, 1e-10))

print()
print(f"合計: {len(results)} 件 / OK {sum(results)} / NG {len(results) - sum(results)}")
