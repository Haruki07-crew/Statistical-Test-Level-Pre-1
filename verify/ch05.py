"""第5章 検算スクリプト（離散型分布）。
記号計算は sympy、数値は scipy.stats、主張値との比較は小数第4位（許容 5e-5、表示桁が少ないものは丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb, exp, sqrt, log, factorial, floor
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def zero(expr):
    return sp.simplify(expr) == 0


rng = np.random.default_rng(20261023)
t, s = sp.symbols("t s", real=True)
x = sp.Symbol("x", integer=True, nonnegative=True)
n, r, m = sp.symbols("n r m", integer=True, positive=True)
p, lam, l1, l2, theta, mu = sp.symbols("p lambda lambda1 lambda2 theta mu", positive=True)
q = 1 - p

# ================= bernoulli-distribution =================
T = "bernoulli-distribution"
M = 1 - p + p * sp.exp(t)
check(T, "M(t)=E[e^{tX}]=1-p+pe^t", zero((1 - p) * sp.exp(0) + p * sp.exp(t) - M), M)
check(T, "M'(0)=p, M''(0)=p", zero(sp.diff(M, t).subs(t, 0) - p) and zero(sp.diff(M, t, 2).subs(t, 0) - p), "sympy")
check(T, "E[X^k]=p (k=1..6)", all(zero(sp.diff(M, t, kk).subs(t, 0) - p) for kk in range(1, 7)), "sympy")
check(T, "V=p(1-p)", zero(p - p**2 - p * (1 - p)), "sympy")
G = 1 - p + p * s
check(T, "G(s)=1-p+ps, G'(1)=p", zero(sp.diff(G, s).subs(s, 1) - p), "sympy")
mu3 = sp.expand(p * (1 - p) ** 3 + (1 - p) * (0 - p) ** 3)
skew = sp.simplify(mu3 / (p * (1 - p)) ** sp.Rational(3, 2))
check(T, "歪度=(1-2p)/sqrt(p(1-p))", all(close(skew.subs(p, pv), (1 - 2 * pv) / sqrt(pv * (1 - pv)), 1e-12) for pv in [0.1, 0.3, 0.7]), "p=0.1,0.3,0.7")
vmax = sp.solve(sp.diff(p * (1 - p), p), p)
check(T, "p(1-p) の最大は p=0.5 で 0.25", vmax == [sp.Rational(1, 2)] and (p * (1 - p)).subs(p, sp.Rational(1, 2)) == sp.Rational(1, 4), vmax)
check(T, "例1 V=0.0475, SD≈0.2179", close(0.05 * 0.95, 0.0475, 1e-12) and close(sqrt(0.0475), 0.2179), sqrt(0.0475))
check(T, "例2 V=0.21, sqrt(0.21)≈0.4583, 歪度≈0.8729", close(0.21, 0.3 * 0.7, 1e-12) and close(sqrt(0.21), 0.4583) and close(stats.bernoulli(0.3).stats(moments="s"), 0.8729), float(stats.bernoulli(0.3).stats(moments="s")))
check(T, "例3 V[p̂]=0.0021, SE≈0.0458", close(0.21 / 100, 0.0021, 1e-12) and close(sqrt(0.0021), 0.0458), sqrt(0.0021))
xs = rng.binomial(1, 0.3, size=(200000, 10)).sum(axis=1)
emp = np.bincount(xs, minlength=11) / len(xs)
check(T, "Bern(0.3)×10 の和 = Bin(10,0.3)（シミュ、各確率の差<0.005）", np.max(np.abs(emp - stats.binom.pmf(range(11), 10, 0.3))) < 0.005, np.max(np.abs(emp - stats.binom.pmf(range(11), 10, 0.3))))
check(T, "尤度 ∏p^x(1-p)^{1-x} の最大点 p̂=x̄", zero(sp.solve(sp.diff(sp.Symbol("S") * sp.log(p) + (sp.Symbol("N") - sp.Symbol("S")) * sp.log(1 - p), p), p)[0] - sp.Symbol("S") / sp.Symbol("N")), "S/N")

# ================= binomial-distribution =================
T = "binomial-distribution"
Mb = (1 - p + p * sp.exp(t)) ** n
Mb_sum = sum(sp.exp(t * kk) * comb(6, kk) * p**kk * q ** (6 - kk) for kk in range(7))
check(T, "M(t)=(1-p+pe^t)^n（n=6で和と一致）", zero(sp.expand(Mb_sum - Mb.subs(n, 6))), "sympy")
d1 = sp.diff(Mb, t); d2 = sp.diff(Mb, t, 2)
check(T, "M'(t)=n(1-p+pe^t)^{n-1}pe^t", zero(d1 - n * (1 - p + p * sp.exp(t)) ** (n - 1) * p * sp.exp(t)), "sympy")
d2c = n * (n - 1) * (1 - p + p * sp.exp(t)) ** (n - 2) * p**2 * sp.exp(2 * t) + n * (1 - p + p * sp.exp(t)) ** (n - 1) * p * sp.exp(t)
check(T, "M''(t) 式（n=2..8 で一致）", all(zero(sp.expand((d2 - d2c).subs(n, nv))) for nv in range(2, 9)), "sympy")
E1 = sp.simplify(d1.subs(t, 0)); E2 = sp.simplify(d2.subs(t, 0))
check(T, "E=np, E[X^2]=n(n-1)p^2+np, V=np(1-p)", zero(E1 - n * p) and zero(E2 - n * (n - 1) * p**2 - n * p) and zero(E2 - E1**2 - n * p * (1 - p)), (E1, sp.factor(E2 - E1**2)))
Gb = (1 - p + p * s) ** n
check(T, "G(s)=(1-p+ps)^n, G'(1)=np", zero(sp.diff(Gb, s).subs(s, 1) - n * p), "sympy")
check(T, "再生性 M^m M^n = M^{m+n}", zero((1 - p + p * sp.exp(t)) ** m * (1 - p + p * sp.exp(t)) ** n - (1 - p + p * sp.exp(t)) ** (m + n)), "sympy")
sim = rng.binomial(7, 0.35, 200000) + rng.binomial(5, 0.35, 200000)
diff = np.max(np.abs(np.bincount(sim, minlength=13) / 200000 - stats.binom.pmf(range(13), 12, 0.35)))
check(T, "Bin(7,.35)+Bin(5,.35)=Bin(12,.35)（シミュ、差<0.005）", diff < 0.005, diff)
sim = rng.binomial(7, 0.2, 200000) + rng.binomial(5, 0.6, 200000)
check(T, "p が異なる和は二項でない（分散が二項の式と不一致）", abs(sim.var() - 12 * (sim.mean() / 12) * (1 - sim.mean() / 12)) > 0.2, (sim.var(), 12 * (sim.mean() / 12) * (1 - sim.mean() / 12)))
check(T, "例1 P(X=3)≈0.2668, 0.7^7≈0.08235", close(stats.binom.pmf(3, 10, 0.3), 0.2668) and close(0.7**7, 0.08235, 5e-6) and comb(10, 3) == 120, stats.binom.pmf(3, 10, 0.3))
pp = stats.binom.pmf([0, 1, 2], 10, 0.3)
check(T, "例1 P(0),P(1),P(2)=0.0282,0.1211,0.2335, P(X≤2)≈0.3828", all(close(a, b) for a, b in zip(pp, [0.0282, 0.1211, 0.2335])) and close(stats.binom.cdf(2, 10, 0.3), 0.3828), (pp, stats.binom.cdf(2, 10, 0.3)))
check(T, "例1 E=3, V=2.1", close(10 * 0.3, 3, 1e-12) and close(10 * 0.3 * 0.7, 2.1, 1e-12), "")
z1 = (35.5 - 40) / sqrt(24); z2 = (35 - 40) / sqrt(24)
check(T, "例2 正確 P(X≤35)≈0.1795", close(stats.binom.cdf(35, 100, 0.4), 0.1795), stats.binom.cdf(35, 100, 0.4))
check(T, "例2 連続修正 z≈-0.9186, Φ≈0.1792", close(z1, -0.9186) and close(stats.norm.cdf(z1), 0.1792), (z1, stats.norm.cdf(z1)))
check(T, "例2 修正なし z≈-1.0206, Φ≈0.1537", close(z2, -1.0206) and close(stats.norm.cdf(z2), 0.1537), (z2, stats.norm.cdf(z2)))
check(T, "例3 0.998^1000≈0.1351, e^-2≈0.1353", close(0.998**1000, 0.1351) and close(exp(-2), 0.1353), (0.998**1000, exp(-2)))
check(T, "例3 P(X=3)≈0.1806, ポアソン≈0.1804", close(stats.binom.pmf(3, 1000, 0.002), 0.1806) and close(8 * exp(-2) / 6, 0.1804), (stats.binom.pmf(3, 1000, 0.002), 8 * exp(-2) / 6))
ok = True
for nv, pv in [(10, 0.3), (7, 0.45), (20, 0.13), (9, 0.6)]:
    mode = int(np.argmax(stats.binom.pmf(range(nv + 1), nv, pv)))
    ok &= mode == floor((nv + 1) * pv)
pm = stats.binom.pmf(range(10), 9, 0.5)  # (n+1)p=5 整数
check(T, "最頻値 floor((n+1)p)、整数なら2つ", ok and close(pm[5], pm[4], 1e-12), "")
check(T, "n=1000 の誤差幅約±3%（1.96*sqrt(.25/1000)）", close(1.96 * sqrt(0.25 / 1000), 0.031, 0.001), 1.96 * sqrt(0.25 / 1000))
lim = sp.limit(sp.binomial(n, 3) * (lam / n) ** 3 * (1 - lam / n) ** (n - 3), n, sp.oo)
check(T, "少数の法則 x=3 の極限 λ^3e^{-λ}/3!", zero(lim - lam**3 * sp.exp(-lam) / 6), lim)

# ================= poisson-distribution =================
T = "poisson-distribution"
Mp = sp.exp(lam * (sp.exp(t) - 1))
zz = sp.Symbol("z", positive=True)  # z = λe^t
Mp_sum = (sp.exp(-lam) * sp.summation(zz**x / sp.factorial(x), (x, 0, sp.oo))).subs(zz, lam * sp.exp(t))
check(T, "M(t)=exp{λ(e^t-1)}（級数和）", zero(Mp_sum - Mp), Mp_sum)
check(T, "M'=λe^tM, M''=λe^tM+(λe^t)^2M", zero(sp.diff(Mp, t) - lam * sp.exp(t) * Mp) and zero(sp.diff(Mp, t, 2) - (lam * sp.exp(t) + lam**2 * sp.exp(2 * t)) * Mp), "sympy")
e1 = sp.diff(Mp, t).subs(t, 0); e2 = sp.simplify(sp.diff(Mp, t, 2).subs(t, 0))
check(T, "E=λ, E[X^2]=λ+λ^2, V=λ", zero(e1 - lam) and zero(e2 - lam - lam**2) and zero(e2 - e1**2 - lam), (e1, e2))
K = lam * (sp.exp(t) - 1)
check(T, "キュムラント K^(j)(0)=λ（j=1..5）", all(zero(sp.diff(K, t, j).subs(t, 0) - lam) for j in range(1, 6)), "sympy")
Gp = sp.exp(lam * (s - 1))
check(T, "G'(1)=λ, G''(1)=λ^2, V=λ", zero(sp.diff(Gp, s).subs(s, 1) - lam) and zero(sp.diff(Gp, s, 2).subs(s, 1) - lam**2), "sympy")
check(T, "再生性 M1*M2 = exp{(λ1+λ2)(e^t-1)}", zero(sp.exp(l1 * (sp.exp(t) - 1)) * sp.exp(l2 * (sp.exp(t) - 1)) - sp.exp((l1 + l2) * (sp.exp(t) - 1))), "sympy")
check(T, "歪度=1/sqrt(λ)", all(close(stats.poisson(lv).stats(moments="s"), 1 / sqrt(lv), 1e-10) for lv in [0.5, 2, 9]), "")
ok = all(close(stats.poisson.cdf(kk - 1, lv), stats.chi2.sf(2 * lv, 2 * kk), 1e-10) for kk in [1, 3, 6] for lv in [0.7, 2.0, 5.5])
check(T, "P(X≤k-1)=P(χ²_{2k}>2λ)", ok, "k=1,3,6; λ=0.7,2,5.5")
xs = rng.poisson(2.0, 400000); ys = rng.poisson(3.0, 400000)
sel = xs[xs + ys == 5]
emp = np.bincount(sel, minlength=6)[:6] / len(sel)
dd = np.max(np.abs(emp - stats.binom.pmf(range(6), 5, 0.4)))
check(T, "X+Y=5 条件付きで X~Bin(5,0.4)（シミュ、差<0.01）", dd < 0.01, dd)
sv = np.var(np.sqrt(rng.poisson(30, 400000)))
check(T, "平方根変換 V[√X]≈1/4（λ=30 シミュ、許容0.01）", abs(sv - 0.25) < 0.01, sv)
check(T, "例1 P(0)≈0.1353, P(2)≈0.2707", close(stats.poisson.pmf(0, 2), 0.1353) and close(stats.poisson.pmf(2, 2), 0.2707), (stats.poisson.pmf(0, 2), stats.poisson.pmf(2, 2)))
check(T, "例1 P(X≥3)=1-5e^-2≈0.3233", close(stats.poisson.sf(2, 2), 0.3233) and close(1 - 5 * exp(-2), stats.poisson.sf(2, 2), 1e-12), stats.poisson.sf(2, 2))
check(T, "例2 P(X+Y=4)=256/24 e^-4≈0.1954", close(stats.poisson.pmf(4, 4), 0.1954) and close(256 / 24 * exp(-4), stats.poisson.pmf(4, 4), 1e-12), stats.poisson.pmf(4, 4))
check(T, "例3 0.998^1000≈0.1351, e^-2≈0.1353", close(stats.binom.pmf(0, 1000, 0.002), 0.1351) and close(exp(-2), 0.1353), "")

# ================= geometric-distribution =================
T = "geometric-distribution"
Gg = p / (1 - q * s)
qq = sp.Symbol("qq", positive=True)
w = sp.Symbol("w", positive=True)  # w = qs, 0<w<1
gsum = sp.summation(w**x, (x, 0, sp.oo))
gsum = gsum.args[0][0] if isinstance(gsum, sp.Piecewise) else gsum
check(T, "G(s)=Σ s^x p q^x = p/(1-qs)（|qs|<1）", zero(p * gsum.subs(w, q * s) - Gg), p * gsum)
check(T, "G'(s)=pq/(1-qs)^2, G''(s)=2pq^2/(1-qs)^3", zero(sp.diff(Gg, s) - p * q / (1 - q * s) ** 2) and zero(sp.diff(Gg, s, 2) - 2 * p * q**2 / (1 - q * s) ** 3), "sympy")
g1 = sp.simplify(sp.diff(Gg, s).subs(s, 1)); g2 = sp.simplify(sp.diff(Gg, s, 2).subs(s, 1))
check(T, "E=q/p, G''(1)=2q^2/p^2, V=q/p^2", zero(g1 - q / p) and zero(g2 - 2 * q**2 / p**2) and zero(g2 + g1 - g1**2 - q / p**2), (g1, sp.simplify(g2 + g1 - g1**2)))
Mg = Gg.subs(s, sp.exp(t))
check(T, "M(t)=p/(1-qe^t), M'(0)=q/p", zero(sp.diff(Mg, t).subs(t, 0) - q / p), "sympy")
MY = p * sp.exp(t) / (1 - q * sp.exp(t))
eY = sp.simplify(sp.diff(MY, t).subs(t, 0)); vY = sp.simplify(sp.diff(MY, t, 2).subs(t, 0) - eY**2)
check(T, "試行回数版 Y: M=pe^t/(1-qe^t), E=1/p, V=q/p^2", zero(eY - 1 / p) and zero(vY - q / p**2) and zero(MY - sp.exp(t) * Mg), (eY, vY))
check(T, "試行回数版 G_Y=ps/(1-qs)=s G_X", zero(p * s / (1 - q * s) - s * Gg), "sympy")
ok = all(close(stats.geom.sf(kk - 1, pv, loc=-1), (1 - pv) ** kk, 1e-12) for pv in [0.2, 0.5] for kk in [0, 1, 4, 9])
check(T, "上側確率 P(X≥k)=q^k（scipy geom loc=-1）", ok, "")
check(T, "無記憶性 P(X≥s+t|X≥s)=P(X≥t)", close(stats.geom.sf(12, 1 / 6, loc=-1) / stats.geom.sf(9, 1 / 6, loc=-1), stats.geom.sf(2, 1 / 6, loc=-1), 1e-12), "s=10,t=3")
check(T, "MGF 存在範囲 t<-log q（qe^t<1 ⇔ t<-log q）", all(abs(float((sp.solve(sp.Eq(q * sp.exp(t), 1), t)[0] + sp.log(q)).subs(p, pv))) < 1e-12 for pv in [0.1, 0.5, 0.9]), sp.solve(sp.Eq(q * sp.exp(t), 1), t))
X1 = stats.geom(1 / 6, loc=-1)
check(T, "例1 E=5, V=30, E[Y]=6", close(X1.mean(), 5, 1e-9) and close(X1.var(), 30, 1e-9), (X1.mean(), X1.var()))
check(T, "例1 P(X=2)=25/216≈0.1157", F(1, 6) * F(5, 6) ** 2 == F(25, 216) and close(X1.pmf(2), 0.1157), X1.pmf(2))
check(T, "例1 P(X≥6)=(5/6)^6≈0.3349", close((5 / 6) ** 6, 0.3349) and close(X1.sf(5), (5 / 6) ** 6, 1e-12), (5 / 6) ** 6)
check(T, "例2 P(X≥13|X≥10)=(5/6)^3≈0.5787", close(X1.sf(12) / X1.sf(9), 0.5787), X1.sf(12) / X1.sf(9))
xb, N_ = sp.symbols("xbar N", positive=True)
ll = N_ * sp.log(p) + N_ * xb * sp.log(1 - p)
check(T, "MLE p̂=1/(1+x̄)", zero(sp.solve(sp.diff(ll, p), p)[0] - 1 / (1 + xb)), sp.solve(sp.diff(ll, p), p))
sim = rng.geometric(0.3, size=(200000, 4)).sum(axis=1) - 4
dd = np.max(np.abs(np.bincount(sim, minlength=40)[:40] / 200000 - stats.nbinom.pmf(range(40), 4, 0.3)))
check(T, "Geo(0.3)×4 の和 = NB(4,0.3)（シミュ、差<0.005）", dd < 0.005, dd)

# ================= negative-binomial-distribution =================
T = "negative-binomial-distribution"
ok = all(close(comb(rv + xv - 1, xv) * pv**rv * (1 - pv) ** xv, stats.nbinom.pmf(xv, rv, pv), 1e-12) for rv in [1, 3, 5] for pv in [0.3, 0.7] for xv in range(8))
check(T, "pmf C(r+x-1,x)p^r q^x = scipy nbinom", ok, "")
ok = all(sp.binomial(rv + xv - 1, xv) == (-1) ** xv * sp.binomial(-rv, xv) for rv in range(1, 6) for xv in range(8))
check(T, "C(r+x-1,x)=(-1)^x C(-r,x)", ok, "")
check(T, "確率の和=1（r=3,p=0.4 で x≤400）", close(sum(stats.nbinom.pmf(range(400), 3, 0.4)), 1, 1e-12), "")
Gn = p**r * (1 - q * s) ** (-r)
check(T, "G'(s)=rqp^r(1-qs)^{-r-1}, G''(s)=r(r+1)q^2p^r(1-qs)^{-r-2}", zero(sp.diff(Gn, s) - r * q * p**r * (1 - q * s) ** (-r - 1)) and zero(sp.diff(Gn, s, 2) - r * (r + 1) * q**2 * p**r * (1 - q * s) ** (-r - 2)), "sympy")
n1 = sp.simplify(sp.diff(Gn, s).subs(s, 1)); n2 = sp.simplify(sp.diff(Gn, s, 2).subs(s, 1))
check(T, "E=rq/p, G''(1)=r(r+1)q^2/p^2, V=rq/p^2", zero(n1 - r * q / p) and zero(n2 - r * (r + 1) * q**2 / p**2) and zero(n2 + n1 - n1**2 - r * q / p**2), (n1, sp.simplify(n2 + n1 - n1**2)))
check(T, "G = (幾何の G)^r", zero(Gn - Gg**r), "sympy")
Mn = Gn.subs(s, sp.exp(t))
check(T, "M(t)=(p/(1-qe^t))^r, M'(0)=rq/p", zero(sp.diff(Mn, t).subs(t, 0) - r * q / p), "sympy")
check(T, "再生性 G_{r1}G_{r2}=G_{r1+r2}", zero(Gn.subs(r, 2) * Gn.subs(r, 3) - Gn.subs(r, 5)), "sympy")
mu_ = r * q / p
check(T, "V=μ/p=μ+μ^2/r", zero(r * q / p**2 - mu_ / p) and zero(mu_ / p - (mu_ + mu_**2 / r)), "sympy")
ok = all(close(comb(yv - 1, rv - 1) * pv**rv * (1 - pv) ** (yv - rv), stats.nbinom.pmf(yv - rv, rv, pv), 1e-12) for rv in [2, 4] for pv in [0.35] for yv in range(rv, rv + 8))
check(T, "試行回数版 P(Y=y)=C(y-1,r-1)p^r q^{y-r}", ok, "")
dd = max(abs(stats.nbinom.pmf(kk, 2000, 2000 / (2000 + 3)) - stats.poisson.pmf(kk, 3)) for kk in range(15))
check(T, "μ=3 固定で r→∞ (r=2000) で Po(3) に近い（差<0.001）", dd < 0.001, dd)
# ガンマ・ポアソン混合（形状 r、尺度 θ）
lv = sp.Symbol("l", positive=True)
mix = sp.integrate(lv**x * sp.exp(-lv) / sp.factorial(x) * lv ** (r - 1) * sp.exp(-lv / theta) / (sp.gamma(r) * theta**r), (lv, 0, sp.oo), conds="none")
ok = all(close(mix.subs({x: xv, r: rv, theta: thv}).evalf(), stats.nbinom.pmf(xv, rv, 1 / (1 + thv)), 1e-10) for xv in range(6) for rv in [1, 3] for thv in [0.5, 2.0])
check(T, "ガンマ(形状r,尺度θ)・ポアソン混合 = NB(r,1/(1+θ))（積分）", ok, "x=0..5, r=1,3, θ=0.5,2")
lam_s = rng.gamma(2.5, 1.6, 400000); sim = rng.poisson(lam_s)
dd = np.max(np.abs(np.bincount(sim, minlength=20)[:20] / 400000 - stats.nbinom.pmf(range(20), 2.5, 1 / 2.6)))
check(T, "混合（r=2.5 実数, θ=1.6）シミュ（差<0.005）", dd < 0.005, dd)
check(T, "例1 P(X=2)=6×0.064×0.36=0.1382", close(stats.nbinom.pmf(2, 3, 0.4), 0.1382) and comb(4, 2) == 6 and close(6 * 0.064 * 0.36, 0.13824, 1e-12), stats.nbinom.pmf(2, 3, 0.4))
check(T, "例1 E=4.5, V=11.25", close(stats.nbinom.mean(3, 0.4), 4.5, 1e-9) and close(stats.nbinom.var(3, 0.4), 11.25, 1e-9), "")
check(T, "例1 P(X≤2)≈0.3174", close(stats.nbinom.cdf(2, 3, 0.4), 0.3174), stats.nbinom.cdf(2, 3, 0.4))
check(T, "例2 r=16/8=2, p=2/6=1/3, 分散=12", F(16, 8) == 2 and F(2, 6) == F(1, 3) and close(stats.nbinom.var(2, 1 / 3), 12, 1e-9) and close(stats.nbinom.mean(2, 1 / 3), 4, 1e-9), "")
check(T, "例2 P(X=0)=(1/3)^2≈0.1111, e^-4≈0.0183", close(stats.nbinom.pmf(0, 2, 1 / 3), 0.1111) and close(exp(-4), 0.0183), (stats.nbinom.pmf(0, 2, 1 / 3), exp(-4)))
check(T, "例2 P(X≥10): NB≈0.0751, Po≈0.0081", close(stats.nbinom.sf(9, 2, 1 / 3), 0.0751) and close(stats.poisson.sf(9, 4), 0.0081), (stats.nbinom.sf(9, 2, 1 / 3), stats.poisson.sf(9, 4)))

# ================= hypergeometric-distribution =================
T = "hypergeometric-distribution"
Nn, Kk, ii = sp.symbols("N K n", positive=True)
cov = Kk * (Kk - 1) / (Nn * (Nn - 1)) - Kk**2 / Nn**2
pK = Kk / Nn
check(T, "Cov(I_i,I_j)=-K(N-K)/(N^2(N-1))=-p(1-p)/(N-1)", zero(cov + Kk * (Nn - Kk) / (Nn**2 * (Nn - 1))) and zero(cov + pK * (1 - pK) / (Nn - 1)), sp.factor(cov))
V = ii * pK * (1 - pK) - ii * (ii - 1) * pK * (1 - pK) / (Nn - 1)
check(T, "V=np(1-p)-n(n-1)p(1-p)/(N-1)=np(1-p)(N-n)/(N-1)", zero(V - ii * pK * (1 - pK) * (Nn - ii) / (Nn - 1)), "sympy")
ok = all(close(stats.hypergeom(Nv, Kv, nv).var(), nv * Kv / Nv * (1 - Kv / Nv) * (Nv - nv) / (Nv - 1), 1e-10) and close(stats.hypergeom(Nv, Kv, nv).mean(), nv * Kv / Nv, 1e-10) for Nv, Kv, nv in [(20, 5, 4), (50, 20, 10), (13, 9, 7)])
check(T, "scipy の E=nK/N, V=np(1-p)(N-n)/(N-1)", ok, "")
H = stats.hypergeom(10, 3, 6)  # (M,n,N)=(総数,当たり,抽出)
sup = [xv for xv in range(0, 7) if H.pmf(xv) > 0]
check(T, "台 max(0,n-(N-K))..min(n,K)（N=10,K=3,n=6 → 0..3）", sup == list(range(max(0, 6 - 7), min(6, 3) + 1)), sup)
H2 = stats.hypergeom(10, 7, 6)
sup2 = [xv for xv in range(0, 7) if H2.pmf(xv) > 0]
check(T, "台の下限（N=10,K=7,n=6 → 3..6）", sup2 == [3, 4, 5, 6], sup2)
H = stats.hypergeom(20, 5, 4)
check(T, "例1 C(20,4)=4845, 分子1365, 2275", comb(20, 4) == 4845 and comb(15, 4) == 1365 and 5 * comb(15, 3) == 2275, "")
check(T, "例1 P(0)≈0.2817, P(1)≈0.4696, 1-P(0)≈0.7183", close(H.pmf(0), 0.2817) and close(H.pmf(1), 0.4696) and close(1 - H.pmf(0), 0.7183), (H.pmf(0), H.pmf(1)))
check(T, "例1 E=1, V=0.75×16/19≈0.6316", close(H.mean(), 1, 1e-12) and close(H.var(), 0.6316), H.var())
check(T, "例1 二項 V=0.75, 0.75^4≈0.3164", close(stats.binom.var(4, 0.25), 0.75, 1e-12) and close(0.75**4, 0.3164), 0.75**4)
H = stats.hypergeom(500, 50, 10)
check(T, "例2 P(X=0)≈0.3452, 0.9^10≈0.3487", close(H.pmf(0), 0.3452) and close(comb(450, 10) / comb(500, 10), H.pmf(0), 1e-12) and close(0.9**10, 0.3487), (H.pmf(0), 0.9**10))
check(T, "例2 分散 0.8838 と 0.9", close(H.var(), 0.8838) and close(10 * 0.1 * 0.9, 0.9, 1e-12), H.var())
dd = max(abs(stats.hypergeom(100000, 30000, 8).pmf(kk) - stats.binom.pmf(kk, 8, 0.3)) for kk in range(9))
check(T, "N→∞ で Bin(n,p) に収束（N=1e5 で差<1e-3）", dd < 1e-3, dd)

# ================= finite-population-correction =================
T = "finite-population-correction"
pop = np.array([3.0, 7.0, 1.0, 9.0, 4.0, 12.0, 6.0, 2.0])
Np = len(pop); muP = pop.mean(); sig2 = pop.var(); S2 = pop.var(ddof=1)
for nv in [1, 3, 5, 8]:
    means = [np.mean(c) for c in combinations(pop, nv)]
    vtrue = np.var(means)
    ok = close(np.mean(means), muP, 1e-12) and close(vtrue, (Np - nv) / (Np - 1) * sig2 / nv, 1e-10) and close(vtrue, (1 - nv / Np) * S2 / nv, 1e-10)
    check(T, f"全列挙 N=8,n={nv}: E[ȳ]=μ, V=(N-n)/(N-1)σ²/n=(1-n/N)S²/n", ok, vtrue)
pairs = [(pop[i] - muP) * (pop[j] - muP) for i in range(Np) for j in range(Np) if i != j]
check(T, "Cov(Y_i,Y_j)=-σ²/(N-1)（全順序対の平均）", close(np.mean(pairs), -sig2 / (Np - 1), 1e-10), np.mean(pairs))
sg, NN, nn = sp.symbols("sigma2 N n", positive=True)
check(T, "nσ²-n(n-1)σ²/(N-1)=nσ²(N-n)/(N-1)", zero(nn * sg - nn * (nn - 1) * sg / (NN - 1) - nn * sg * (NN - nn) / (NN - 1)), "sympy")
S2s = sg * NN / (NN - 1)
check(T, "σ²=(N-1)/N S² を代入して2式一致", zero((NN - nn) / (NN - 1) * sg / nn - (1 - nn / NN) * S2s / nn), "sympy")
check(T, "例1 900/999≈0.9009, V≈3.6036, SE≈1.8983", close(900 / 999, 0.9009) and close(900 / 999 * 4, 3.6036) and close(sqrt(900 / 999 * 4), 1.8983), (900 / 999 * 4, sqrt(900 / 999 * 4)))
check(T, "例1 修正なし SE=2, 約5%小さい", close(20 / sqrt(100), 2, 1e-12) and abs((1 - sqrt(900 / 999 * 4) / 2) - 0.05) < 0.005, 1 - sqrt(900 / 999 * 4) / 2)
v2 = 450 / 499 * 0.21 / 50
check(T, "例2 V≈0.003788, SE≈0.0615, 修正なし≈0.0648", close(v2, 0.003788, 5e-7) and close(sqrt(v2), 0.0615) and close(sqrt(0.21 / 50), 0.0648), (v2, sqrt(v2), sqrt(0.21 / 50)))
check(T, "比率版 = 超幾何 V/n²（N=500,K=150,n=50）", close(stats.hypergeom(500, 150, 50).var() / 2500, v2, 1e-12), stats.hypergeom(500, 150, 50).var() / 2500)

# ================= fisher-exact-test =================
T = "fisher-exact-test"
a_, b_, c_, d_ = 7, 1, 2, 6
nT = a_ + b_ + c_ + d_
f1 = F(comb(a_ + b_, a_) * comb(c_ + d_, c_), comb(nT, a_ + c_))
f2 = F(factorial(a_ + b_) * factorial(c_ + d_) * factorial(a_ + c_) * factorial(b_ + d_), factorial(nT) * factorial(a_) * factorial(b_) * factorial(c_) * factorial(d_))
check(T, "P(a) の2つの表現が一致", f1 == f2, f1)
H = stats.hypergeom(16, 9, 8)
check(T, "C(16,8)=12870, 分子252, 9", comb(16, 8) == 12870 and comb(9, 7) * comb(7, 1) == 252 and comb(9, 8) == 9, "")
check(T, "P(a=7)≈0.01958, P(a=8)≈0.00070", close(H.pmf(7), 0.01958, 5e-6) and close(H.pmf(8), 0.00070, 5e-6), (H.pmf(7), H.pmf(8)))
_, p_g = stats.fisher_exact([[7, 1], [2, 6]], alternative="greater")
check(T, "片側 p≈0.0203（fisher_exact greater）", close(p_g, 0.0203) and close(H.sf(6), p_g, 1e-12), p_g)
check(T, "分布の対称性 P(k)=P(9-k)", all(close(H.pmf(kk), H.pmf(9 - kk), 1e-14) for kk in range(1, 9)), "")
or_, p_2 = stats.fisher_exact([[7, 1], [2, 6]], alternative="two-sided")
check(T, "両側 p≈0.0406 = P(1)+P(2)+P(7)+P(8)（fisher_exact）", close(p_2, 0.0406) and close(sum(H.pmf([1, 2, 7, 8])), p_2, 1e-12), p_2)
check(T, "標本オッズ比=21", close(or_, 21, 1e-12), or_)
ex = np.outer([8, 8], [9, 7]) / 16
check(T, "期待度数の最小=3.5", close(ex.min(), 3.5, 1e-12), ex)
check(T, "紅茶 1/C(8,4)=1/70≈0.0143", comb(8, 4) == 70 and close(stats.hypergeom(8, 4, 4).pmf(4), 0.0143), stats.hypergeom(8, 4, 4).pmf(4))
check(T, "紅茶 3杯以上 17/70≈0.2429", comb(4, 3) * comb(4, 1) == 16 and close(stats.hypergeom(8, 4, 4).sf(2), 0.2429), stats.hypergeom(8, 4, 4).sf(2))
_, pt = stats.fisher_exact([[4, 0], [0, 4]], alternative="greater")
check(T, "紅茶 全問正解 片側 p=1/70<0.05", close(pt, 1 / 70, 1e-12), pt)

# ================= multinomial-distribution =================
T = "multinomial-distribution"
pv = [0.5, 0.3, 0.2]
# 真値 0.08505 ちょうど（四捨五入で 0.0851）なので許容は丸め幅 5e-5 + 浮動小数誤差
check(T, "例1 10!/(5!3!2!)=2520, 確率≈0.0851", factorial(10) // (factorial(5) * factorial(3) * factorial(2)) == 2520 and close(stats.multinomial.pmf([5, 3, 2], 10, pv), 0.0851, 5e-5 + 1e-12) and close(2520 * 0.03125 * 0.027 * 0.04, stats.multinomial.pmf([5, 3, 2], 10, pv), 1e-12), stats.multinomial.pmf([5, 3, 2], 10, pv))
check(T, "例1 周辺 Bin(10,0.5) P(5)≈0.2461", close(stats.binom.pmf(5, 10, 0.5), 0.2461), stats.binom.pmf(5, 10, 0.5))
C = stats.multinomial.cov(10, pv)
Cc = np.array([[2.5, -1.5, -1.0], [-1.5, 2.1, -0.6], [-1.0, -0.6, 1.6]])
check(T, "例2 分散共分散行列・行和0", np.allclose(C, Cc) and np.allclose(C.sum(axis=1), 0), C)
check(T, "例2 Cov=-1.5, ρ≈-0.6547", close(C[0, 1], -1.5, 1e-12) and close(C[0, 1] / sqrt(C[0, 0] * C[1, 1]), -0.6547) and close(-sqrt(0.15 / 0.35), -0.6547), C[0, 1] / sqrt(C[0, 0] * C[1, 1]))
# 条件付き: X3=2 のとき X1 ~ Bin(8, 0.625)
cond = np.array([stats.multinomial.pmf([k1, 8 - k1, 2], 10, pv) for k1 in range(9)])
cond /= cond.sum()
check(T, "例2 X3=2 条件付きで X1~Bin(8,0.625), 期待値5", np.allclose(cond, stats.binom.pmf(range(9), 8, 0.625)) and close(8 * 0.625, 5, 1e-12), cond @ np.arange(9))
check(T, "例3 6!/6^6=720/46656≈0.0154", close(stats.multinomial.pmf([1] * 6, 6, [1 / 6] * 6), 0.0154) and close(720 / 46656, stats.multinomial.pmf([1] * 6, 6, [1 / 6] * 6), 1e-12), 720 / 46656)
p1, p2, p3 = sp.symbols("p1 p2 p3", positive=True)
t1, t2, t3 = sp.symbols("t1 t2 t3", real=True)
Mm = (p1 * sp.exp(t1) + p2 * sp.exp(t2) + p3 * sp.exp(t3)) ** n
z0 = {t1: 0, t2: 0, t3: 0}
EXY = sp.diff(Mm, t1, t2).subs(z0).subs(p3, 1 - p1 - p2)
EX1 = sp.diff(Mm, t1).subs(z0).subs(p3, 1 - p1 - p2)
EX1sq = sp.diff(Mm, t1, 2).subs(z0).subs(p3, 1 - p1 - p2)
check(T, "MGF: E[XiXj]=n(n-1)pipj, Cov=-npipj", zero(EXY - n * (n - 1) * p1 * p2) and zero(EXY - EX1 * n * p2 - (-n * p1 * p2)), sp.simplify(EXY))
check(T, "MGF: E=npi, V=npi(1-pi)", zero(EX1 - n * p1) and zero(EX1sq - EX1**2 - n * p1 * (1 - p1)), "sympy")
check(T, "まとめ法: n(pi+pj)(1-pi-pj)-npi(1-pi)-npj(1-pj)=-2npipj", zero(n * (p1 + p2) * (1 - p1 - p2) - n * p1 * (1 - p1) - n * p2 * (1 - p2) + 2 * n * p1 * p2), "sympy")
rho_e = -n * p1 * p2 / sp.sqrt(n * p1 * (1 - p1) * n * p2 * (1 - p2)) + sp.sqrt(p1 * p2 / ((1 - p1) * (1 - p2)))
check(T, "相関=-sqrt(pipj/((1-pi)(1-pj)))（数値代入）", all(abs(float(rho_e.subs({n: nv, p1: a, p2: b}))) < 1e-12 for nv in [3, 10] for a, b in [(0.5, 0.3), (0.1, 0.6), (0.25, 0.25)]), "sympy")
check(T, "比率差の分散 {pi(1-pi)+pj(1-pj)+2pipj}/n", zero((n * p1 * (1 - p1) + n * p2 * (1 - p2) - 2 * (-n * p1 * p2)) / n**2 - (p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2) / n), "sympy")
check(T, "PGF (Σpisi)^n の ∂/∂s1 at 1 = np1", zero(sp.diff((p1 * s + p2 + p3) ** n, s).subs(s, 1).subs(p3, 1 - p1 - p2) - n * p1), "sympy")
sim = rng.multinomial(4, pv, 200000) + rng.multinomial(6, pv, 200000)
ok = np.allclose(sim.mean(axis=0), 10 * np.array(pv), atol=0.02) and np.allclose(np.cov(sim.T), Cc, atol=0.05)
check(T, "再生性 Mult(4)+Mult(6)=Mult(10)（シミュ、平均±0.02・共分散±0.05）", ok, np.cov(sim.T).round(3))
Y = rng.poisson([1.0, 2.0, 3.0], size=(400000, 3))
sel = Y[Y.sum(axis=1) == 4]
emp = np.mean(np.all(sel == [1, 1, 2], axis=1))
check(T, "ポアソンを合計で条件付け → Mult(4;1/6,2/6,3/6)（P(1,1,2), 許容0.005）", abs(emp - stats.multinomial.pmf([1, 1, 2], 4, [1 / 6, 2 / 6, 3 / 6])) < 0.005, (emp, stats.multinomial.pmf([1, 1, 2], 4, [1 / 6, 2 / 6, 3 / 6])))
Xs = rng.multinomial(200, [0.4, 0.3, 0.2, 0.1], 100000)
chi = ((Xs - 200 * np.array([0.4, 0.3, 0.2, 0.1])) ** 2 / (200 * np.array([0.4, 0.3, 0.2, 0.1]))).sum(axis=1)
check(T, "ピアソン χ² ≈ 自由度 k-1=3（シミュ、上側5%点超過率 0.05±0.005）", abs(np.mean(chi > stats.chi2.ppf(0.95, 3)) - 0.05) < 0.005, np.mean(chi > stats.chi2.ppf(0.95, 3)))

print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
