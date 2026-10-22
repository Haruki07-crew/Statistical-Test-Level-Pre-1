"""第2章 検算スクリプト（確率分布と母関数）。
記号計算は sympy、数値は scipy.stats / scipy.integrate、主張値との比較は小数第4位（許容 5e-5、表示桁が少ないものは丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from fractions import Fraction as F
from math import comb, exp, gamma, sqrt, log, pi, atan
import numpy as np
import sympy as sp
from scipy import stats, integrate

results = []

def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")

def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol

def zero(expr):
    return sp.simplify(expr) == 0

rng = np.random.default_rng(20261023)
t, s, x, y, u = sp.symbols("t s x y u", real=True)
n, k = sp.symbols("n k", integer=True, positive=True)
p, lam, mu, a, b = sp.symbols("p lambda mu a b", positive=True)
sig, gam, alpha, beta = sp.symbols("sigma gamma alpha beta", positive=True)
q = 1 - p

# 共通の表（joint/marginal/conditional）
tab = {(0, 0): F(1, 10), (0, 1): F(2, 10), (1, 0): F(2, 10), (1, 1): F(3, 10), (2, 0): F(1, 10), (2, 1): F(1, 10)}
pX = {i: sum(v for (xx, yy), v in tab.items() if xx == i) for i in range(3)}
pY = {j: sum(v for (xx, yy), v in tab.items() if yy == j) for j in range(2)}

# ================= moment-generating-function =================
T = "moment-generating-function"
# 二項
Mb = (p * sp.exp(t) + q) ** n
Mb_sum = sum(sp.exp(t * kk) * comb(5, kk) * p**kk * q**(5 - kk) for kk in range(6))
check(T, "二項 M(t)=(pe^t+q)^n（n=5で和と一致）", zero(Mb_sum - Mb.subs(n, 5)), "sympy")
d1 = sp.diff(Mb, t); d2 = sp.diff(Mb, t, 2)
check(T, "二項 M'(t)=npe^t(pe^t+q)^{n-1}", zero(d1 - n * p * sp.exp(t) * (p * sp.exp(t) + q) ** (n - 1)), "sympy")
d2_claim = n * p * sp.exp(t) * (p * sp.exp(t) + q) ** (n - 1) + n * (n - 1) * p**2 * sp.exp(2 * t) * (p * sp.exp(t) + q) ** (n - 2)
check(T, "二項 M''(t) 式（n=3..8 で記号一致）", all(zero(sp.expand((d2 - d2_claim).subs(n, nv))) for nv in range(3, 9)), "sympy")
E1 = sp.simplify(d1.subs(t, 0)); E2 = sp.simplify(d2.subs(t, 0))
check(T, "二項 E[X]=np, E[X^2]=np+n(n-1)p^2, V=np(1-p)", zero(E1 - n * p) and zero(E2 - (n * p + n * (n - 1) * p**2)) and zero(E2 - E1**2 - n * p * (1 - p)), (E1, sp.factor(E2 - E1**2)))
# ポアソン
Mp = sp.exp(lam * (sp.exp(t) - 1))
zz = sp.Symbol("z", positive=True)  # z = λe^t
Mp_sum = (sp.exp(-lam) * sp.summation(zz**k / sp.factorial(k), (k, 0, sp.oo))).subs(zz, lam * sp.exp(t))
check(T, "ポアソン M(t)=exp{λ(e^t-1)}（級数和）", zero(Mp_sum - Mp), sp.simplify(Mp_sum))
check(T, "ポアソン M'=λe^tM, M''=(λe^t+λ^2e^{2t})M", zero(sp.diff(Mp, t) - lam * sp.exp(t) * Mp) and zero(sp.diff(Mp, t, 2) - (lam * sp.exp(t) + lam**2 * sp.exp(2 * t)) * Mp), "sympy")
e1 = sp.diff(Mp, t).subs(t, 0); e2 = sp.simplify(sp.diff(Mp, t, 2).subs(t, 0))
check(T, "ポアソン E=λ, E[X^2]=λ+λ^2, V=λ", zero(e1 - lam) and zero(e2 - lam - lam**2) and zero(e2 - e1**2 - lam), (e1, e2))
# 正規
xx = sp.Symbol("xx", real=True)
lhs = t * xx - (xx - mu) ** 2 / (2 * sig**2)
rhs = -(xx - (mu + sig**2 * t)) ** 2 / (2 * sig**2) + mu * t + sig**2 * t**2 / 2
check(T, "正規 平方完成の恒等式", zero(lhs - rhs), "sympy")
Mn_int = sp.integrate(sp.exp(t * xx) * sp.exp(-(xx - mu) ** 2 / (2 * sig**2)) / (sp.sqrt(2 * sp.pi) * sig), (xx, -sp.oo, sp.oo))
Mn = sp.exp(mu * t + sig**2 * t**2 / 2)
check(T, "正規 M(t)=exp(μt+σ²t²/2)（積分）", zero(sp.simplify(Mn_int) - Mn), sp.simplify(Mn_int))
check(T, "正規 M'=(μ+σ²t)M, M''=σ²M+(μ+σ²t)²M", zero(sp.diff(Mn, t) - (mu + sig**2 * t) * Mn) and zero(sp.diff(Mn, t, 2) - (sig**2 + (mu + sig**2 * t) ** 2) * Mn), "sympy")
check(T, "正規 E=μ, E[X^2]=σ²+μ², V=σ²", zero(sp.diff(Mn, t).subs(t, 0) - mu) and zero(sp.diff(Mn, t, 2).subs(t, 0) - sig**2 - mu**2), "sympy")
# キュムラント
for name, M, m_, v_ in [("ポアソン", Mp, lam, lam), ("正規", Mn, mu, sig**2), ("二項", Mb, n * p, n * p * q)]:
    K = sp.log(M)
    check(T, f"キュムラント {name} K'(0)=平均, K''(0)=分散", zero(sp.diff(K, t).subs(t, 0) - m_) and zero(sp.diff(K, t, 2).subs(t, 0) - v_), "sympy")
Kb = n * sp.log(p * sp.exp(t) + q)
check(T, "二項 K'(t)=npe^t/(pe^t+q), K''(t)=npqe^t/(pe^t+q)^2", zero(sp.diff(Kb, t) - n * p * sp.exp(t) / (p * sp.exp(t) + q)) and zero(sp.diff(Kb, t, 2) - n * p * q * sp.exp(t) / (p * sp.exp(t) + q) ** 2), "sympy")
Mf = sp.Function("M")(t)
Kg = sp.log(Mf)
k2 = sp.diff(Kg, t, 2)
check(T, "K''=(M''M-(M')^2)/M^2", zero(k2 - (sp.diff(Mf, t, 2) * Mf - sp.diff(Mf, t) ** 2) / Mf**2), "sympy")
# K'''(0)=3次中心モーメント（二項で確認）
K3 = sp.diff(Kb, t, 3).subs(t, 0)
mu3 = sum(comb(5, j) * F(3, 10)**j * F(7, 10)**(5 - j) * (j - F(3, 2))**3 for j in range(6))
check(T, "K'''(0)=3次中心モーメント（Bin(5,0.3)）", sp.Rational(mu3.numerator, mu3.denominator) == sp.nsimplify(K3.subs({n: 5, p: sp.Rational(3, 10)})), float(mu3))
# 表のMGF
Mgeo = sp.summation(sp.exp(t * k) * p * q**k, (k, 0, sp.oo))
pg = sp.Rational(1, 3); tg = sp.Rational(1, 5)  # t < -log(2/3)≈0.405
check(T, "表 幾何（失敗回数） p/(1-qe^t)", close(sum(float(sp.exp(tg * kk) * pg * (1 - pg) ** kk) for kk in range(400)), float(pg / (1 - (1 - pg) * sp.exp(tg)))), float(pg / (1 - (1 - pg) * sp.exp(tg))))
Mexp = sp.integrate(sp.exp(t * xx) * lam * sp.exp(-lam * xx), (xx, 0, sp.oo), conds="none")
check(T, "表 指数 λ/(λ-t)", zero(Mexp - lam / (lam - t)), sp.simplify(Mexp))
for al, la, tt in [(2.5, 1.5, 0.7), (0.7, 2.0, -0.3)]:
    num = integrate.quad(lambda z: np.exp(tt * z) * stats.gamma.pdf(z, al, scale=1 / la), 0, 400, limit=200)[0]
    check(T, f"表 ガンマ(α={al},λ={la}) (λ/(λ-t))^α, t={tt}", close(num, (la / (la - tt)) ** al, 1e-7), num)
for kk, tt in [(3, 0.2), (5, -0.4)]:
    num = integrate.quad(lambda z: np.exp(tt * z) * stats.chi2.pdf(z, kk), 0, 400, limit=200)[0]
    check(T, f"表 χ²({kk}) (1-2t)^(-k/2), t={tt}", close(num, (1 - 2 * tt) ** (-kk / 2), 1e-7), num)
Mu = sp.integrate(sp.exp(t * xx) / (b - a), (xx, a, b))
Mu = Mu.args[0][0] if isinstance(Mu, sp.Piecewise) else Mu
check(T, "表 一様 (e^{bt}-e^{at})/((b-a)t), t→0で1", zero(sp.simplify(Mu - (sp.exp(b * t) - sp.exp(a * t)) / ((b - a) * t))) and sp.limit((sp.exp(b * t) - sp.exp(a * t)) / ((b - a) * t), t, 0) == 1, "sympy")
check(T, "ガンマ 尺度β流儀 (1-βt)^{-α}（β=1/λ, 数値 3点）", all(close((lv / (lv - tv)) ** av, (1 - tv / lv) ** (-av), 1e-12) and close(integrate.quad(lambda z: np.exp(tv * z) * stats.gamma.pdf(z, av, scale=1 / lv), 0, 400, limit=200)[0], (1 - tv / lv) ** (-av), 1e-7) for av, lv, tv in [(2.5, 1.5, 0.7), (0.7, 2.0, -0.3), (4, 0.5, 0.2)]), "λ=1/β")
# 性質4 1次変換・標準化
X = stats.norm(1, 2); aa, bb, tt = 1.5, -0.7, 0.3
lhs_ = integrate.quad(lambda z: np.exp(tt * (aa * z + bb)) * X.pdf(z), -40, 40)[0]
rhs_ = np.exp(bb * tt) * np.exp(1 * aa * tt + 4 * (aa * tt) ** 2 / 2)
check(T, "M_{aX+b}(t)=e^{bt}M_X(at)（N(1,4)で数値）", close(lhs_, rhs_, 1e-8), lhs_)
check(T, "Z=(X-μ)/σ の MGF = e^{-μt/σ}M_X(t/σ) = e^{t²/2}", zero(sp.exp(-mu * t / sig) * Mn.subs(t, t / sig) - sp.exp(t**2 / 2)), "sympy")
# 例1
M10 = Mb.subs({n: 10, p: sp.Rational(3, 10)})
m1 = sp.diff(M10, t).subs(t, 0); m2 = sp.diff(M10, t, 2).subs(t, 0)
check(T, "例1 M'(0)=3, M''(0)=11.1, V=2.1", sp.nsimplify(m1) == 3 and sp.nsimplify(m2) == sp.Rational(111, 10) and sp.nsimplify(m2 - m1**2) == sp.Rational(21, 10), (float(m1), float(m2), float(m2 - m1**2)))
# 例2
check(T, "例2 (0.4+0.6e^t)^5 = Bin(5,0.6), E=3, V=1.2", zero(Mb.subs({n: 5, p: sp.Rational(3, 5)}) - (sp.Rational(2, 5) + sp.Rational(3, 5) * sp.exp(t)) ** 5) and stats.binom(5, .6).mean() == 3 and close(stats.binom(5, .6).var(), 1.2, 1e-12), (3, 1.2))
# 例3
check(T, "例3 e^{2(e^t-1)}e^{3(e^t-1)}=e^{5(e^t-1)}", zero(Mp.subs(lam, 2) * Mp.subs(lam, 3) - Mp.subs(lam, 5)), "sympy")
sim = rng.poisson(2, 10**6) + rng.poisson(3, 10**6)
check(T, "例3 シミュレーション X+Y の平均・分散≈5（N=1e6, 許容0.02）、P(0)≈e^{-5}（許容0.0005）", abs(sim.mean() - 5) < 0.02 and abs(sim.var() - 5) < 0.02 and abs((sim == 0).mean() - exp(-5)) < 5e-4, (sim.mean(), sim.var(), (sim == 0).mean()))
check(T, "例3 e^{-5}≈0.0067", close(exp(-5), 0.0067), exp(-5))
check(T, "M_X(t)=G_X(e^t)（ポアソン）", zero(sp.exp(lam * (s - 1)).subs(s, sp.exp(t)) - Mp), "sympy")

# ================= probability-generating-function =================
T = "probability-generating-function"
Gb = (q + p * s) ** n
check(T, "二項 G(s)=(q+ps)^n（n=4で和）", zero(sum(comb(4, kk) * (p * s) ** kk * q ** (4 - kk) for kk in range(5)) - Gb.subs(n, 4)), "sympy")
check(T, "二項 G'=np(q+ps)^{n-1}, G''=n(n-1)p^2(q+ps)^{n-2}", zero(sp.diff(Gb, s) - n * p * (q + p * s) ** (n - 1)) and zero(sp.diff(Gb, s, 2) - n * (n - 1) * p**2 * (q + p * s) ** (n - 2)), "sympy")
g1 = sp.simplify(sp.diff(Gb, s).subs(s, 1)); g2 = sp.simplify(sp.diff(Gb, s, 2).subs(s, 1))
check(T, "二項 G'(1)=np, G''(1)=n(n-1)p^2, V=np(1-p)", zero(g1 - n * p) and zero(g2 - n * (n - 1) * p**2) and zero(g2 + g1 - g1**2 - n * p * (1 - p)), "sympy")
Gp = sp.exp(lam * (s - 1))
check(T, "ポアソン G(s)=e^{λ(s-1)}（級数和）", zero(sp.summation(s**k * sp.exp(-lam) * lam**k / sp.factorial(k), (k, 0, sp.oo)) - Gp), "sympy")
check(T, "ポアソン G'=λe^{λ(s-1)}, G''=λ²e^{λ(s-1)}, V=λ", zero(sp.diff(Gp, s) - lam * Gp) and zero(sp.diff(Gp, s, 2) - lam**2 * Gp) and zero(lam**2 + lam - lam**2 - lam), "sympy")
Gg = p / (1 - q * s)
Gg_sum = sp.summation(s**k * p * q**k, (k, 0, sp.oo))
check(T, "幾何（失敗回数） G=p/(1-qs)（|qs|<1で級数和, 数値3点）", all(close(sum(pv * ((1 - pv) * sv) ** j for j in range(2000)), pv / (1 - (1 - pv) * sv), 1e-12) for pv, sv in [(0.3, 0.5), (0.3, 1.3), (0.6, -2.0)]), "OK")
check(T, "幾何 G'=pq/(1-qs)^2, G''=2pq^2/(1-qs)^3", zero(sp.diff(Gg, s) - p * q / (1 - q * s) ** 2) and zero(sp.diff(Gg, s, 2) - 2 * p * q**2 / (1 - q * s) ** 3), "sympy")
gg1 = sp.simplify(sp.diff(Gg, s).subs(s, 1)); gg2 = sp.simplify(sp.diff(Gg, s, 2).subs(s, 1))
check(T, "幾何 G'(1)=q/p, G''(1)=2q²/p², V=q/p²", zero(gg1 - q / p) and zero(gg2 - 2 * q**2 / p**2) and zero(gg2 + gg1 - gg1**2 - q / p**2), (gg1, gg2))
Gg2 = p * s / (1 - q * s)
h1 = sp.diff(Gg2, s).subs(s, 1); h2 = sp.diff(Gg2, s, 2).subs(s, 1)
check(T, "幾何（試行回数） G=ps/(1-qs), E=1/p, V=q/p²", zero(h1 - 1 / p) and zero(h2 + h1 - h1**2 - q / p**2), "sympy")
r = sp.Symbol("r", positive=True)
Gnb = (p / (1 - q * s)) ** r
n1 = sp.diff(Gnb, s).subs(s, 1); n2 = sp.diff(Gnb, s, 2).subs(s, 1)
check(T, "負の二項 G=(p/(1-qs))^r, E=rq/p, V=rq/p²", zero(n1 - r * q / p) and zero(n2 + n1 - n1**2 - r * q / p**2), "sympy")
nb = stats.nbinom(4, 0.3)
check(T, "負の二項 scipy(r=4,p=0.3) で E,V", close(nb.mean(), 4 * .7 / .3, 1e-9) and close(nb.var(), 4 * .7 / .09, 1e-9), (nb.mean(), nb.var()))
check(T, "幾何 scipy(p=0.3, 失敗回数) E=q/p, V=q/p²", close(stats.geom(0.3, loc=-1).mean(), .7 / .3, 1e-9) and close(stats.geom(0.3, loc=-1).var(), .7 / .09, 1e-9), "OK")
# 性質1 確率の取り出し（Po(2)）
G2 = Gp.subs(lam, 2)
check(T, "P(X=k)=G^{(k)}(0)/k!（Po(2), k=0..5）", all(zero(sp.diff(G2, s, kk).subs(s, 0) / sp.factorial(kk) - sp.exp(-2) * 2**kk / sp.factorial(kk)) for kk in range(6)), "sympy")
check(T, "G^{(3)}(1)=E[X(X-1)(X-2)]（Po(2)で8）", zero(sp.diff(G2, s, 3).subs(s, 1) - 8) and close(sum(stats.poisson.pmf(j, 2) * j * (j - 1) * (j - 2) for j in range(80)), 8, 1e-9), 8)
# 例1
G6 = sp.Rational(1, 6) / (1 - sp.Rational(5, 6) * s)
a1 = sp.diff(G6, s).subs(s, 1); a2 = sp.diff(G6, s, 2).subs(s, 1)
check(T, "例1 G'(1)=5, G''(1)=50, V=30, SD≈5.477", a1 == 5 and a2 == 50 and a2 + a1 - a1**2 == 30 and close(sqrt(30), 5.477, 5e-4), (a1, a2, sqrt(30)))
# 例2
G2d = sp.expand(sp.Rational(1, 36) * sum(s**j for j in range(1, 7)) ** 2)
check(T, "例2 s^7 の係数 6/36≈0.1667", G2d.coeff(s, 7) == sp.Rational(1, 6) and close(1 / 6, 0.1667), G2d.coeff(s, 7))
# 例3
GS = Gp.subs(lam, 3).subs(s, sp.Rational(3, 5) + sp.Rational(2, 5) * s)
check(T, "例3 G_N(G_X(s))=e^{1.2(s-1)}", zero(GS - sp.exp(sp.Rational(6, 5) * (s - 1))), "sympy")
check(T, "例3 e^{-1.2}≈0.3012", close(exp(-1.2), 0.3012), exp(-1.2))
Nn = rng.poisson(3, 10**6); Sx = rng.binomial(Nn, 0.4)
check(T, "例3 シミュレーション S の平均・分散≈1.2（N=1e6, 許容0.01）, P(S=0)≈0.3012（許容0.002）", abs(Sx.mean() - 1.2) < 0.01 and abs(Sx.var() - 1.2) < 0.01 and abs((Sx == 0).mean() - exp(-1.2)) < 0.002, (Sx.mean(), Sx.var(), (Sx == 0).mean()))
gy = sp.symbols("g", positive=True)
check(T, "E[S]=E[N]E[X]（G_N(G_X)を微分）", zero(sp.diff(GS, s).subs(s, 1) - 3 * sp.Rational(2, 5)), "1.2")
# 再生性（幾何の和→負の二項）
check(T, "幾何×r 個の和の G = 負の二項の G（r=3）", zero(Gg**3 - Gnb.subs(r, 3)), "sympy")

# ================= characteristic-function =================
T = "characteristic-function"
phin = sp.exp(sp.I * mu * t - sig**2 * t**2 / 2)
check(T, "正規 φ(t)=M(it)", zero(Mn.subs(t, sp.I * t) - phin), "sympy")
for mm, ss_, tt in [(1.0, 2.0, 0.7)]:
    re = integrate.quad(lambda z: np.cos(tt * z) * stats.norm.pdf(z, mm, ss_), -40, 40)[0]
    im = integrate.quad(lambda z: np.sin(tt * z) * stats.norm.pdf(z, mm, ss_), -40, 40)[0]
    val = np.exp(1j * mm * tt - ss_**2 * tt**2 / 2)
    check(T, "正規 φ 数値積分(μ=1,σ=2,t=0.7)", abs(re + 1j * im - val) < 1e-8, re + 1j * im)
check(T, "ポアソン φ=exp{λ(e^{it}-1)}=M(it)", zero(Mp.subs(t, sp.I * t) - sp.exp(lam * (sp.exp(sp.I * t) - 1))), "sympy")
la, tt = 2.0, 0.9
re = integrate.quad(lambda z: np.cos(tt * z) * la * np.exp(-la * z), 0, np.inf)[0]
im = integrate.quad(lambda z: np.sin(tt * z) * la * np.exp(-la * z), 0, np.inf)[0]
check(T, "指数 φ=λ/(λ-it)（λ=2,t=0.9 数値）", abs(re + 1j * im - la / (la - 1j * tt)) < 1e-8, re + 1j * im)
phiu = sp.integrate(sp.exp(sp.I * t * xx) / (2 * a), (xx, -a, a))
check(T, "一様U(-a,a) φ=sin(at)/(at)", zero(sp.simplify(sp.expand_complex(sp.simplify(phiu.rewrite(sp.sin)))) - sp.sin(a * t) / (a * t)) if not isinstance(phiu, sp.Piecewise) else zero(sp.simplify(phiu.args[0][0].rewrite(sp.sin)) - sp.sin(a * t) / (a * t)), "sympy")
check(T, "一様 t→0 で 1", sp.limit(sp.sin(a * t) / (a * t), t, 0) == 1, 1)
for m_, g_, tt in [(0.5, 1.5, 0.8), (0.0, 1.0, -2.0)]:
    # X=μ+γZ（Z 標準コーシー）。E[cos(sZ)]=2∫_0^∞cos(sz)f(z)dz をフーリエ積分（QAWF）で数値計算、E[sin(sZ)]=0
    s0 = g_ * abs(tt)
    phi0 = 2 * integrate.quad(lambda z: 1 / (pi * (1 + z * z)), 0, np.inf, weight="cos", wvar=s0)[0]
    num = np.exp(1j * m_ * tt) * phi0
    val = np.exp(1j * m_ * tt - g_ * abs(tt))
    check(T, f"コーシー φ=exp(iμt-γ|t|)（μ={m_},γ={g_},t={tt} 数値, 許容1e-8）", abs(num - val) < 1e-8, num)
# 1次変換
check(T, "φ_{aX+b}(t)=e^{ibt}φ_X(at)（正規で記号）", zero(sp.exp(sp.I * b * t) * phin.subs(t, a * t) - sp.exp(sp.I * (a * mu + b) * t - a**2 * sig**2 * t**2 / 2)), "sympy")
# 標本平均
nn = sp.Symbol("nn", positive=True)
check(T, "{φ(t/n)}^n=e^{-|t|}（コーシー）", zero((sp.exp(-sp.Abs(t) / nn)) ** nn - sp.exp(-sp.Abs(t))), "sympy")
# 例1
check(T, "例1 P(|X|>1)=1-(2/π)arctan1=0.5", close(1 - 2 / pi * atan(1), 0.5, 1e-12) and close(2 * stats.cauchy.sf(1), 0.5, 1e-12), 2 * stats.cauchy.sf(1))
xb = rng.standard_cauchy((200_000, 100)).mean(axis=1)
pr = (np.abs(xb) > 1).mean()
check(T, "例1 シミュレーション 100個平均で P(|Xbar|>1)≈0.5（2e5 反復, 許容0.005）", abs(pr - 0.5) < 0.005, pr)
check(T, "例1 正規 P(|Z|>10)≈0", 2 * stats.norm.sf(10) < 1e-20, 2 * stats.norm.sf(10))
# 例2
phie = 2 / (2 - sp.I * t)
dphi = sp.diff(phie, t)
check(T, "例2 φ'(t)=2i/(2-it)^2, φ'(0)=0.5i, E[X]=0.5", zero(dphi - 2 * sp.I / (2 - sp.I * t) ** 2) and zero(dphi.subs(t, 0) - sp.I / 2) and zero(dphi.subs(t, 0) / sp.I - sp.Rational(1, 2)), dphi.subs(t, 0))
check(T, "性質5 φ''(0)=i²E[X²]（指数λ=2: E[X²]=0.5）", zero(sp.diff(phie, t, 2).subs(t, 0) - sp.I**2 * sp.Rational(1, 2)), "sympy")
check(T, "|φ|≤1, φ(0)=1（指数λ=2, t∈[-50,50]）", all(abs(2 / (2 - 1j * tv)) <= 1 + 1e-15 for tv in np.linspace(-50, 50, 1001)) and phie.subs(t, 0) == 1, "OK")

# ================= cumulative-distribution-function =================
T = "cumulative-distribution-function"
die = [F(1, 6)] * 6
Fd = lambda v: sum(die[i] for i in range(6) if i + 1 <= v)
check(T, "例1 F(3)=0.5, F(3.7)=0.5, F(5)-F(2)=0.5", Fd(3) == F(1, 2) and Fd(3.7) == F(1, 2) and Fd(5) - Fd(2) == F(1, 2), (Fd(3), Fd(3.7), Fd(5) - Fd(2)))
Fe = sp.integrate(sp.Rational(1, 2) * sp.exp(-u / 2), (u, 0, x))
check(T, "例2 F(x)=1-e^{-0.5x}", zero(Fe - (1 - sp.exp(-x / 2))), Fe)
check(T, "例2 F(2)≈0.6321", close(stats.expon(scale=2).cdf(2), 0.6321), stats.expon(scale=2).cdf(2))
check(T, "例2 F(3)-F(1)≈0.3834 (0.6065-0.2231)", close(stats.expon(scale=2).cdf(3) - stats.expon(scale=2).cdf(1), 0.3834) and close(exp(-.5), .6065) and close(exp(-1.5), .2231), stats.expon(scale=2).cdf(3) - stats.expon(scale=2).cdf(1))
check(T, "例3 F_M(0.5)=0.125, f_M=3x^2", close(0.5**3, 0.125, 1e-12) and zero(sp.diff(x**3, x) - 3 * x**2), 0.125)
mx = rng.random((10**6, 3)).max(axis=1)
check(T, "例3 シミュレーション P(M≤0.5)≈0.125（N=1e6, 許容0.002）", abs((mx <= .5).mean() - .125) < 0.002, (mx <= .5).mean())
uu = stats.norm.cdf(rng.normal(size=10**5))
check(T, "確率積分変換 F(X)~U(0,1)（KS検定 p>0.01）", stats.kstest(uu, "uniform").pvalue > 0.01, stats.kstest(uu, "uniform").pvalue)

# ================= survival-function =================
T = "survival-function"
check(T, "例1 S(5)=e^{-0.5}≈0.6065, S(8)/S(3)=e^{-0.5}", close(exp(-.5), .6065) and close(exp(-.8) / exp(-.3), .6065), exp(-.5))
check(T, "例1 平均寿命 ∫e^{-0.1t}dt=10", sp.integrate(sp.exp(-t / 10), (t, 0, sp.oo)) == 10, 10)
check(T, "例2 S(5)≈0.7788, S(10)≈0.3679, S(10)/S(5)≈0.4724", close(exp(-.25), .7788) and close(exp(-1), .3679) and close(exp(-.75), .4724), (exp(-.25), exp(-1), exp(-.75)))
mw = integrate.quad(lambda z: np.exp(-(z / 10) ** 2), 0, np.inf)[0]
check(T, "例2 平均寿命 10Γ(1.5)≈8.862（∫S と一致）", close(10 * gamma(1.5), 8.862, 5e-4) and close(mw, 10 * gamma(1.5), 1e-8) and close(stats.weibull_min(2, scale=10).mean(), mw, 1e-8), mw)
pg_ = stats.geom(0.3, loc=-1)
check(T, "離散 E[T]=Σ_{k≥1}P(T≥k)=Σ_{k≥0}P(T>k)（幾何 p=0.3）", close(sum(pg_.sf(kk - 1) for kk in range(1, 400)), pg_.mean(), 1e-9) and close(sum(pg_.sf(kk) for kk in range(0, 400)), pg_.mean(), 1e-9), pg_.mean())
check(T, "連続 E[T]=∫S（ガンマ(2.5, 尺度1.3)）", close(integrate.quad(stats.gamma(2.5, scale=1.3).sf, 0, np.inf)[0], 2.5 * 1.3, 1e-7), 3.25)

# ================= hazard-function =================
T = "hazard-function"
kk_, eta = sp.symbols("k eta", positive=True)
Sw = sp.exp(-(t / eta) ** kk_)
fw = -sp.diff(Sw, t)
hw = sp.simplify(fw / Sw)
check(T, "ワイブル h=f/S=(k/η)(t/η)^{k-1}", zero(hw - kk_ / eta * (t / eta) ** (kk_ - 1)), hw)
check(T, "h=-d/dt log S（ワイブル）", zero(-sp.diff(sp.log(Sw), t) - hw), "sympy")
Hw = sp.integrate((kk_ / eta * (u / eta) ** (kk_ - 1)), (u, 0, t))
check(T, "H(t)=(t/η)^k, S=e^{-H}, f=h e^{-H}", zero(sp.simplify(Hw) - (t / eta) ** kk_) and zero(fw - hw * sp.exp(-(t / eta) ** kk_)), sp.simplify(Hw))
check(T, "指数 h=λ", zero(sp.simplify((lam * sp.exp(-lam * t)) / sp.exp(-lam * t)) - lam), lam)
check(T, "幾何（離散）h(k)=P(T=k|T≥k)=p", all(close(stats.geom(0.3).pmf(j) / stats.geom(0.3).sf(j - 1), 0.3, 1e-12) for j in range(1, 30)), 0.3)
h1w = (2 / sp.Integer(10)) * (t / 10)
check(T, "例1 h(t)=0.02t, h(5)=0.1, h(10)=0.2, S(10)=e^{-1}≈0.3679", zero(h1w - t / 50) and h1w.subs(t, 5) == sp.Rational(1, 10) and h1w.subs(t, 10) == sp.Rational(1, 5) and close(exp(-1), .3679), "OK")
hs = [F(200, 1000), F(120, 800), F(68, 680)]
S3 = (1 - hs[0]) * (1 - hs[1]) * (1 - hs[2])
check(T, "例2 h=0.20,0.15,0.10, S(3)=0.612=612/1000", hs == [F(1, 5), F(3, 20), F(1, 10)] and S3 == F(612, 1000) and 1000 - 200 - 120 - 68 == 612 and 800 - 120 == 680, float(S3))
check(T, "例3 0.7788^2≈0.6065", close(0.7788**2, 0.6065), 0.7788**2)
# 直列系
check(T, "直列系 ハザードの和（2つのワイブル）", zero(-sp.diff(sp.log(Sw.subs({kk_: 2, eta: 3}) * Sw.subs({kk_: 1.5, eta: 5})), t) - (hw.subs({kk_: 2, eta: 3}) + hw.subs({kk_: 1.5, eta: 5}))), "sympy")
# ガンマのハザード → 1/β
for al_, be_ in [(2, 2), (0.5, 2)]:
    hg = lambda z: stats.gamma(al_, scale=be_).pdf(z) / stats.gamma(al_, scale=be_).sf(z)
    v = [hg(z) for z in (1, 5, 20, 60)]
    mono = all(np.diff(v) > 0) if al_ > 1 else all(np.diff(v) < 0)
    check(T, f"図2 ガンマ(α={al_},β={be_}) ハザードが単調で 1/β に近づく", mono and abs(v[-1] - 1 / be_) < 0.03, v)

# ================= joint-probability-function =================
T = "joint-probability-function"
check(T, "例1 合計1・周辺 (0.3,0.5,0.2),(0.4,0.6)", sum(tab.values()) == 1 and [pX[i] for i in range(3)] == [F(3, 10), F(5, 10), F(2, 10)] and [pY[j] for j in range(2)] == [F(4, 10), F(6, 10)], "OK")
check(T, "例1 P(X+Y=2)=0.40", sum(v for (i, j), v in tab.items() if i + j == 2) == F(4, 10), 0.4)
check(T, "例1 p(0,0)=0.10 vs 0.12", pX[0] * pY[0] == F(12, 100), float(pX[0] * pY[0]))
EX = sum(i * v for (i, j), v in tab.items()); EY = sum(j * v for (i, j), v in tab.items())
EXY = sum(i * j * v for (i, j), v in tab.items()); EX2 = sum(i * i * v for (i, j), v in tab.items())
VX = EX2 - EX**2; VY = EY - EY**2; cov = EXY - EX * EY
check(T, "例1 E[X]=0.9, E[Y]=0.6, E[XY]=0.50, Cov=-0.04", (EX, EY, EXY, cov) == (F(9, 10), F(6, 10), F(1, 2), F(-4, 100)), (float(EX), float(EY), float(EXY), float(cov)))
corr = float(cov) / sqrt(float(VX * VY))
check(T, "例1 V[X]=0.49(E[X²]=1.3), V[Y]=0.24, ρ≈-0.1166", EX2 == F(13, 10) and VX == F(49, 100) and VY == F(24, 100) and close(corr, -0.1166), corr)
draw = {(1, 1): F(3, 5) * F(2, 4), (1, 0): F(3, 5) * F(2, 4), (0, 1): F(2, 5) * F(3, 4), (0, 0): F(2, 5) * F(1, 4)}
check(T, "例2 p=0.3,0.3,0.3,0.1 合計1", list(draw.values()) == [F(3, 10)] * 3 + [F(1, 10)] and sum(draw.values()) == 1, [float(v) for v in draw.values()])
from itertools import permutations
balls = ["R"] * 3 + ["W"] * 2
perms = list(permutations(range(5), 2))
pr11 = F(sum(balls[i] == "R" and balls[j] == "R" for i, j in perms), len(perms))
pY1 = F(sum(balls[j] == "R" for i, j in perms), len(perms))
check(T, "例2 全列挙 p(1,1)=0.3, P(Y=1)=0.6, 0.36≠0.3", pr11 == F(3, 10) and pY1 == F(3, 5) and pY1**2 == F(36, 100), (float(pr11), float(pY1)))

# ================= marginal-probability-function =================
T = "marginal-probability-function"
c = F(1, sum(i + j for i in (1, 2) for j in (1, 2)))
pm = {(i, j): c * (i + j) for i in (1, 2) for j in (1, 2)}
pX1 = pm[(1, 1)] + pm[(1, 2)]; pX2 = pm[(2, 1)] + pm[(2, 2)]
check(T, "例1 c=1/12, p_X=(5/12,7/12), E[X]=19/12≈1.5833", c == F(1, 12) and (pX1, pX2) == (F(5, 12), F(7, 12)) and close(pX1 + 2 * pX2, 1.5833), float(pX1 + 2 * pX2))
check(T, "例1 p(1,1)=2/12≈0.1667 vs 25/144≈0.1736", pm[(1, 1)] == F(2, 12) and pX1**2 == F(25, 144) and close(F(25, 144), 0.1736), (float(pm[(1, 1)]), float(pX1**2)))
check(T, "例2 p_Y(赤)=2/4·3/5+3/4·2/5=0.6", F(2, 4) * F(3, 5) + F(3, 4) * F(2, 5) == F(3, 5), 0.6)
check(T, "図 表[[0.1,0.3],[0.3,0.3]] 行列和 (0.4,0.6)", (F(1, 10) + F(3, 10), F(3, 10) + F(3, 10)) == (F(2, 5), F(3, 5)), "OK")

# ================= marginal-distribution =================
T = "marginal-distribution"
check(T, "例1 周辺 (0.3,0.5,0.2),(0.4,0.6)", [pX[i] for i in range(3)] == [F(3, 10), F(1, 2), F(1, 5)] and [pY[j] for j in range(2)] == [F(2, 5), F(3, 5)], "OK")
pb2 = comb(10, 2) * .2**2 * .8**8
mult = sum(stats.multinomial.pmf([2, nb_, 8 - nb_], 10, [.2, .3, .5]) for nb_ in range(9))
check(T, "例2 P(N_A=2)≈0.3020（二項式＝多項和）", close(pb2, .3020) and close(mult, pb2, 1e-12), (pb2, mult))
check(T, "例3 P(X>180)=P(Z>10/6)≈0.0478", close(stats.norm.sf(10 / 6), 0.0478), stats.norm.sf(10 / 6))
Z = rng.multivariate_normal([170, 60], [[36, 0.9 * 6 * 8], [0.9 * 6 * 8, 64]], size=10**6)
check(T, "例3 シミュレーション ρ=0.9 でも P(X>180)≈0.0478（N=1e6, 許容0.001）", abs((Z[:, 0] > 180).mean() - 0.0478) < 0.001, (Z[:, 0] > 180).mean())

# ================= joint-probability-density-function =================
T = "joint-probability-density-function"
fxy = x + y
check(T, "例1 ∫∫(x+y)=1", sp.integrate(fxy, (y, 0, 1), (x, 0, 1)) == 1, 1)
check(T, "例1 P(X≤1/2,Y≤1/2)=1/8", sp.integrate(fxy, (y, 0, sp.Rational(1, 2)), (x, 0, sp.Rational(1, 2))) == sp.Rational(1, 8), "1/8")
inner = sp.integrate(fxy, (y, 0, 1 - x))
check(T, "例1 内側=(1-x²)/2, P(X+Y≤1)=1/3", zero(inner - (1 - x**2) / 2) and sp.integrate(inner, (x, 0, 1)) == sp.Rational(1, 3), "1/3")
EXc = sp.integrate(x * (x + sp.Rational(1, 2)), (x, 0, 1)); EXYc = sp.integrate(x * y * fxy, (y, 0, 1), (x, 0, 1))
check(T, "例1 E[X]=7/12≈0.5833, E[XY]=1/3, Cov=-1/144≈-0.0069", EXc == sp.Rational(7, 12) and EXYc == sp.Rational(1, 3) and EXYc - EXc**2 == sp.Rational(-1, 144) and close(-1 / 144, -0.0069), float(EXYc - EXc**2))
pe = sp.integrate(sp.exp(-x - y), (y, 0, 1 - x), (x, 0, 1))
check(T, "例2 P(X+Y≤1)=1-2e^{-1}≈0.2642（=Gamma(2,1) cdf）", zero(pe - (1 - 2 * sp.exp(-1))) and close(float(pe), 0.2642) and close(stats.gamma(2).cdf(1), float(pe), 1e-12), float(pe))
check(T, "例2 内側 ∫e^{-x-y}dy=e^{-x}-e^{-1}", zero(sp.integrate(sp.exp(-x - y), (y, 0, 1 - x)) - (sp.exp(-x) - sp.exp(-1))), "sympy")
check(T, "例3 ∫∫2=1, P(Y>1/2)=0.25", sp.integrate(2, (y, 0, x), (x, 0, 1)) == 1 and sp.integrate(2, (x, y, 1), (y, sp.Rational(1, 2), 1)) == sp.Rational(1, 4), 0.25)
sx, sy, rho_ = sp.symbols("s_x s_y rho", positive=True)
mx_, my_ = sp.symbols("m_x m_y", real=True)
fbv = 1 / (2 * sp.pi * sx * sy * sp.sqrt(1 - rho_**2)) * sp.exp(-1 / (2 * (1 - rho_**2)) * ((x - mx_) ** 2 / sx**2 - 2 * rho_ * (x - mx_) * (y - my_) / (sx * sy) + (y - my_) ** 2 / sy**2))
fbv_num = sp.lambdify((x, y), fbv.subs({mx_: 1, my_: -2, sx: 1.5, sy: 0.8, rho_: 0.6}))
pts = [(0.3, -1.7), (2.0, -2.5), (-1, -3)]
mvn = stats.multivariate_normal([1, -2], [[2.25, 0.6 * 1.5 * 0.8], [0.6 * 1.5 * 0.8, 0.64]])
check(T, "2変量正規の密度式（scipy と一致）", all(close(fbv_num(*pt), mvn.pdf(pt), 1e-12) for pt in pts), [fbv_num(*pt) for pt in pts])
Fxy = sp.integrate(u + sp.Symbol("v"), (sp.Symbol("v"), 0, y), (u, 0, x))
check(T, "f=∂²F/∂x∂y（例1）", zero(sp.diff(Fxy, x, y) - (x + y)), "sympy")

# ================= marginal-probability-density-function =================
T = "marginal-probability-density-function"
fX = sp.integrate(2, (y, 0, x)); fY = sp.integrate(2, (x, y, 1))
check(T, "例1 f_X=2x, f_Y=2(1-y)", zero(fX - 2 * x) and zero(fY - 2 * (1 - y)), (fX, fY))
check(T, "例1 E[X]=2/3, E[Y]=1/3", sp.integrate(x * 2 * x, (x, 0, 1)) == sp.Rational(2, 3) and sp.integrate(y * 2 * (1 - y), (y, 0, 1)) == sp.Rational(1, 3), "2/3, 1/3")
check(T, "例2 f_X=x+1/2, P(X≤1/2)=0.375", zero(sp.integrate(x + y, (y, 0, 1)) - x - sp.Rational(1, 2)) and sp.integrate(x + sp.Rational(1, 2), (x, 0, sp.Rational(1, 2))) == sp.Rational(3, 8), 0.375)
fbv_l = sp.lambdify((x, y, rho_), fbv.subs({mx_: 1, my_: -2, sx: 1.5, sy: 0.8}))
check(T, "2変量正規の周辺 = N(μX,σX²)（ρ=-0.8,0,0.6,0.95 で数値積分）", all(close(integrate.quad(lambda yy: fbv_l(xv, yy, rv), -np.inf, np.inf)[0], stats.norm.pdf(xv, 1, 1.5), 1e-9) for rv in (-0.8, 0.0, 0.6, 0.95) for xv in (-1.0, 1.0, 3.2)), "OK")

# ================= conditional-probability-function =================
T = "conditional-probability-function"
c1 = [tab[(i, 1)] / pY[1] for i in range(3)]; c0 = [tab[(i, 0)] / pY[0] for i in range(3)]
check(T, "例1 p(x|1)=(1/3,1/2,1/6), p(x|0)=(0.25,0.5,0.25)", c1 == [F(1, 3), F(1, 2), F(1, 6)] and c0 == [F(1, 4), F(1, 2), F(1, 4)], "OK")
E1c = sum(i * c1[i] for i in range(3)); E0c = sum(i * c0[i] for i in range(3))
check(T, "例1 E[X|Y=1]=5/6≈0.8333, E[X|Y=0]=1, E[X]=0.9", E1c == F(5, 6) and E0c == 1 and E0c * pY[0] + E1c * pY[1] == F(9, 10), float(E1c))
V1 = sum(i * i * c1[i] for i in range(3)) - E1c**2; V0 = sum(i * i * c0[i] for i in range(3)) - E0c**2
EV = pY[0] * V0 + pY[1] * V1; VE = pY[0] * E0c**2 + pY[1] * E1c**2 - F(81, 100)
check(T, "例1 V[X|0]=0.5, V[X|1]=17/36≈0.4722", V0 == F(1, 2) and V1 == F(17, 36) and close(V1, .4722), float(V1))
check(T, "例1 E[V]≈0.4833, V[E]≈0.0067, 合計0.49", close(EV, .4833) and close(VE, .0067) and EV + VE == F(49, 100), (float(EV), float(VE)))
kk2 = sp.Symbol("kk", integer=True, nonnegative=True)
expr = (sp.exp(-2) * 2**kk2 / sp.factorial(kk2) * sp.exp(-3) * 3**(5 - kk2) / sp.factorial(5 - kk2)) / (sp.exp(-5) * 5**5 / sp.factorial(5))
check(T, "例2 条件付き分布=B(5,0.4)（k=0..5）", all(zero(expr.subs(kk2, j) - sp.binomial(5, j) * sp.Rational(2, 5)**j * sp.Rational(3, 5)**(5 - j)) for j in range(6)), "sympy")
check(T, "例2 P(X=2|X+Y=5)=0.3456", close(stats.binom.pmf(2, 5, .4), .3456, 1e-12), stats.binom.pmf(2, 5, .4))
xs = rng.poisson(2, 2 * 10**6); ys = rng.poisson(3, 2 * 10**6); m5 = (xs + ys) == 5
check(T, "例2 シミュレーション P(X=2|X+Y=5)≈0.3456（N=2e6, 許容0.005）", abs((xs[m5] == 2).mean() - .3456) < 0.005, (xs[m5] == 2).mean())
# 二項の和で条件付け→超幾何
hg = [comb(4, j) * comb(6, 3 - j) / comb(10, 3) for j in range(4)]
cond = [stats.binom.pmf(j, 4, .3) * stats.binom.pmf(3 - j, 6, .3) / stats.binom.pmf(3, 10, .3) for j in range(4)]
check(T, "独立な二項の和で条件付け→超幾何（n1=4,n2=6,和3）", all(close(aa_, bb_, 1e-12) for aa_, bb_ in zip(hg, cond)), hg)

# ================= conditional-probability-density-function =================
T = "conditional-probability-density-function"
cm = 60 + 0.6 * 8 / 6 * (176 - 170); cv = 64 * (1 - .36)
check(T, "例1 E=64.8, V=40.96=6.4²", close(cm, 64.8, 1e-12) and close(cv, 40.96, 1e-12) and close(sqrt(cv), 6.4, 1e-12), (cm, cv))
check(T, "例1 z=0.8125, P≈0.2083", close((70 - 64.8) / 6.4, 0.8125, 1e-12) and close(stats.norm.sf(0.8125), 0.2083), stats.norm.sf(0.8125))
check(T, "例1 P(Y>70)=P(Z>1.25)≈0.1056, 約2倍", close(stats.norm.sf(1.25), 0.1056) and 1.8 < stats.norm.sf(0.8125) / stats.norm.sf(1.25) < 2.2, stats.norm.sf(0.8125) / stats.norm.sf(1.25))
# 条件付き分布の公式（記号）
# f(x,y) = N(x;μX,σX²)·N(y; 条件付き平均, 条件付き分散) を対数で恒等式として確認（一般の記号のまま）
fXn = sp.exp(-(x - mx_) ** 2 / (2 * sx**2)) / (sp.sqrt(2 * sp.pi) * sx)
target = sp.exp(-(y - (my_ + rho_ * sy / sx * (x - mx_))) ** 2 / (2 * sy**2 * (1 - rho_**2))) / sp.sqrt(2 * sp.pi * sy**2 * (1 - rho_**2))
diff_log = sp.expand_log(sp.log(fbv), force=True) - sp.expand_log(sp.log(fXn * target), force=True)
check(T, "2変量正規 Y|X=x ~ N(μY+ρσY/σX(x-μX), σY²(1-ρ²))（f=f_X·f(y|x) の恒等式）", zero(sp.simplify(sp.expand(diff_log))), "sympy")
Zs = rng.multivariate_normal([170, 60], [[36, .6 * 48], [.6 * 48, 64]], size=4 * 10**6)
band = np.abs(Zs[:, 0] - 176) < 0.2
check(T, "例1 シミュレーション 176±0.2 の帯で平均≈64.8, SD≈6.4（許容0.15）, P(Y>70)≈0.2083（許容0.01）", abs(Zs[band, 1].mean() - 64.8) < .15 and abs(Zs[band, 1].std() - 6.4) < .15 and abs((Zs[band, 1] > 70).mean() - .2083) < .01, (Zs[band, 1].mean(), Zs[band, 1].std(), (Zs[band, 1] > 70).mean()))
fyx = (x + y) / (x + sp.Rational(1, 2))
check(T, "例2 f(y|1/2)=y+1/2, P(Y≤1/2|·)=0.375, E=7/12≈0.5833", zero(fyx.subs(x, sp.Rational(1, 2)) - y - sp.Rational(1, 2)) and sp.integrate(y + sp.Rational(1, 2), (y, 0, sp.Rational(1, 2))) == sp.Rational(3, 8) and sp.integrate(y * (y + sp.Rational(1, 2)), (y, 0, 1)) == sp.Rational(7, 12), "OK")
check(T, "例2 ∫f(y|x)dy=1", zero(sp.integrate(fyx, (y, 0, 1)) - 1), "sympy")
check(T, "例3 f(y|x)=1/x, E[Y|x]=x/2, E[Y]=1/3", zero(2 / (2 * x) - 1 / x) and zero(sp.integrate(y / x, (y, 0, x)) - x / 2) and sp.integrate(x / 2 * 2 * x, (x, 0, 1)) == sp.Rational(1, 3), "1/3")
fy_marg = sp.integrate(1 / x, (x, y, 1))
check(T, "試験 X~U(0,1),Y|X~U(0,x): f_Y=-log y, E[Y]=0.25", zero(fy_marg + sp.log(y)) and sp.integrate(-y * sp.log(y), (y, 0, 1)) == sp.Rational(1, 4), fy_marg)

# ---------------- 集計 ----------------
ok = sum(results); ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
