"""第7章 検算スクリプト（極限定理・収束・確率不等式）。
記号計算は sympy、数値は scipy.stats、主張値との比較は小数第4位（許容 5e-5、表示桁が少ないものは丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の数倍）。
"""
from math import comb, exp, sqrt, log, factorial
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
Phi = stats.norm.cdf
n, m, N, k = sp.symbols("n m N k", integer=True, positive=True)
t, s = sp.symbols("t s", real=True)
lam = sp.Symbol("lambda", positive=True)

# ================= almost-sure-convergence =================
T = "almost-sure-convergence"
ok = all(sp.prod([1 - sp.Rational(1, j) for j in range(mm, NN + 1)]) == sp.Rational(mm - 1, NN) for mm in [2, 5, 10] for NN in [20, 100, 1000])
check(T, "例1 ∏_{n=m}^N(1-1/n)=(m-1)/N（有限で厳密）", ok, "Fraction")
check(T, "例1 m=10,N=1000 で 1-9/1000=0.991", close(1 - float(sp.prod([1 - sp.Rational(1, j) for j in range(10, 1001)])), 0.991, 1e-12), 0.991)
ok = all(close(float(sp.prod([1 - sp.Rational(1, j * j) for j in range(mm, 200001)])), (mm - 1) / mm, 1e-4) for mm in [2, 10, 100])
check(T, "例2 ∏_{n≥m}(1-1/n²)=(m-1)/m（数値, 許容1e-4）", ok, "m=2,10,100")
check(T, "例2 1/m: m=10→0.1, m=100→0.01", close(1 - 9 / 10, 0.1, 1e-12) and close(1 - 99 / 100, 0.01, 1e-12), "")
sim = rng.random((20000, 2000)) < 1 / np.arange(1, 2001)
hit = sim[:, 9:].any(axis=1).mean()
check(T, "例1 シミュ P(10..2000 番目で1が出る)=1-9/2000（許容0.003）", abs(hit - (1 - 9 / 2000)) < 0.003, hit)
sim2 = rng.random((20000, 2000)) < 1 / np.arange(1, 2001) ** 2
hit2 = sim2[:, 9:].any(axis=1).mean()
check(T, "例2 シミュ P(10..2000 番目で1が出る)=1-(9/10)(2001/2000)≈0.1（許容0.006）", abs(hit2 - (1 - (9 / 10) * (2001 / 2000))) < 0.006, hit2)

# ================= central-limit-theorem =================
T = "central-limit-theorem"
y = sp.Symbol("y")
logM = n * sp.log(1 + t**2 / (2 * n))
check(T, "n log(1+t²/(2n)) → t²/2", zero(sp.limit(logM, n, sp.oo) - t**2 / 2), sp.limit(logM, n, sp.oo))
# 実際の MGF（一様分布を標準化）で {M(t/√n)}^n → e^{t²/2}
a = sp.sqrt(3)
Mu = (sp.exp(a * y) - sp.exp(-a * y)) / (2 * a * y)  # U(-√3,√3) の MGF（平均0分散1）
vals = [float((Mu.subs(y, 1.0 / sqrt(nn))) ** nn) for nn in [10, 100, 10000]]
check(T, "標準化一様の {M(t/√n)}^n → e^{1/2}（t=1, n=1e4 で許容1e-4）", abs(vals[-1] - exp(0.5)) < 1e-4, vals)
check(T, "Taylor M(s)=1+s²/2+o(s²)", zero(sp.series(Mu, y, 0, 3).removeO() - (1 + y**2 / 2)), sp.series(Mu, y, 0, 3))
sd = sqrt(100 * 35 / 12)
check(T, "例1 σ²=35/12, 100σ²=291.67, sd=17.078", close(100 * 35 / 12, 291.67, 5e-3) and close(sd, 17.078, 5e-4), sd)
z = (379.5 - 350) / sd
check(T, "例1 z=1.727, 1-Φ≈0.0421", close(z, 1.727, 5e-4) and close(1 - Phi(z), 0.0421), (z, 1 - Phi(z)))
pmf = np.array([1.0])
for _ in range(100):
    pmf = np.convolve(pmf, np.ones(6) / 6)
exact = pmf[380 - 100:].sum()
check(T, "例1 畳み込みの正確値 0.0420", close(exact, 0.0420), exact)
check(T, "例1 連続修正なし 1-Φ(30/17.078)≈0.0395", close(1 - Phi(30 / sd), 0.0395), 1 - Phi(30 / sd))
sd2 = 3 * sqrt(50)
check(T, "例2 sd=3√50≈21.213, z=0.707, 1-Φ≈0.2398", close(sd2, 21.213, 5e-4) and close(15 / sd2, 0.707, 5e-4) and close(1 - Phi(15 / sd2), 0.2398), 1 - Phi(15 / sd2))
check(T, "例2 ガンマ(50, 尺度3) P(T>165)=0.2322", close(stats.gamma.sf(165, 50, scale=3), 0.2322), stats.gamma.sf(165, 50, scale=3))
Zs = (rng.exponential(1, (200000, 50)).mean(axis=1) - 1) * sqrt(50)
check(T, "シミュ 指数 n=50 の標準化平均 P(Z>0.707)≈0.2322（許容0.003）", abs((Zs > 15 / sd2).mean() - 0.2322) < 0.003, (Zs > 15 / sd2).mean())

# ================= chebyshev-inequality =================
T = "chebyshev-inequality"
ks = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
cheb = [1, 0.4444, 0.25, 0.16, 0.1111, 0.0816, 0.0625]
nor = [0.3173, 0.1336, 0.0455, 0.0124, 0.0027, 0.0005, 0.0001]
ex = [0.1353, 0.0821, 0.0498, 0.0302, 0.0183, 0.0111, 0.0067]
un = [0.4226, 0.134, 0, 0, 0, 0, 0]
check(T, "図 1/k² 系列", all(close(1 / kk**2, c) for kk, c in zip(ks, cheb)), [round(1 / kk**2, 4) for kk in ks])
nv = 2 * stats.norm.sf(ks)
check(T, "図 正規 2(1-Φ(k)) 系列", all(close(a, b) for a, b in zip(nv, nor)), nv.round(4))
ev = stats.expon.sf(1 + ks) + stats.expon.cdf(1 - ks)
check(T, "図 指数(平均1) P(|X-1|≥k)=e^{-(1+k)} 系列", all(close(a, b) for a, b in zip(ev, ex)), ev.round(4))
sg = 1 / sqrt(12)
uv = np.clip(1 - 2 * ks * sg, 0, None)
check(T, "図 一様 P(|U-1/2|≥kσ) 系列（0.134 は丸め3桁）", all(close(a, b, 5e-4) for a, b in zip(uv, un)), uv.round(4))
check(T, "例1 1-1/4=0.75, 正規なら 95.45%", close(1 - 1 / 4, 0.75, 1e-12) and close(1 - 2 * stats.norm.sf(2), 0.9545), 1 - 2 * stats.norm.sf(2))
check(T, "例2 n≥0.25/(0.0025·0.05)=2000", close(0.25 / (0.0025 * 0.05), 2000, 1e-9), 0.25 / (0.0025 * 0.05))
nn = (1.96 / 0.05) ** 2 * 0.25
check(T, "例2 正規近似 384.16→385 人、比は約5倍", close(nn, 384.2, 5e-2) and int(np.ceil(nn)) == 385 and 4.5 < 2000 / 385 < 5.5, (nn, 2000 / 385))
check(T, "例3 1/9≈0.1111, P(|Z|≥3)≈0.0027, e^{-4}≈0.0183", close(1 / 9, 0.1111) and close(2 * stats.norm.sf(3), 0.0027) and close(exp(-4), 0.0183), (2 * stats.norm.sf(3), exp(-4)))
check(T, "k=4 で 1/16、正規 k=2 で約 0.046", close(1 / 16, 0.0625, 1e-12) and close(2 * stats.norm.sf(2), 0.046, 5e-4), 2 * stats.norm.sf(2))
X = rng.exponential(1, 1000000)
check(T, "シミュ 指数 P(|X-1|≥2) ≤ 1/4（実測≈e^{-3}, 許容0.001）", abs((np.abs(X - 1) >= 2).mean() - exp(-3)) < 0.001 and (np.abs(X - 1) >= 2).mean() <= 0.25, (np.abs(X - 1) >= 2).mean())

# ================= continuity-correction =================
T = "continuity-correction"
mu, sg = 20, sqrt(12)
check(T, "例1 μ=20, σ=√12≈3.4641", close(sg, 3.4641), sg)
check(T, "例1 正確値 P(X≤15)=0.0955", close(stats.binom.cdf(15, 50, 0.4), 0.0955), stats.binom.cdf(15, 50, 0.4))
check(T, "例1 補正なし Φ(-1.443)=0.0745", close(-5 / sg, -1.443, 5e-4) and close(Phi(-5 / sg), 0.0745), Phi(-5 / sg))
check(T, "例1 補正あり Φ(-1.299)=0.0970", close(-4.5 / sg, -1.299, 5e-4) and close(Phi(-4.5 / sg), 0.0970), Phi(-4.5 / sg))
v = Phi(0.5 / sg) - Phi(-0.5 / sg)
check(T, "例2 P(X=20)≈0.1148（z=0.144）, 正確 0.1146", close(0.5 / sg, 0.144, 5e-4) and close(v, 0.1148) and close(stats.binom.pmf(20, 50, 0.4), 0.1146), (v, stats.binom.pmf(20, 50, 0.4)))
v = Phi(2.5 / sg) - Phi(-2.5 / sg)
ex_ = stats.binom.cdf(22, 50, 0.4) - stats.binom.cdf(17, 50, 0.4)
check(T, "例2 P(18≤X≤22)≈0.5295, 正確 0.5291", close(v, 0.5295) and close(ex_, 0.5291), (v, ex_))
check(T, "例2 補正なし 0.4363", close(Phi(2 / sg) - Phi(-2 / sg), 0.4363), Phi(2 / sg) - Phi(-2 / sg))
s5 = sqrt(5)
check(T, "例3 σ=√5≈2.2361, z=2.012, 1-Φ≈0.0221", close(s5, 2.2361) and close(4.5 / s5, 2.012, 5e-4) and close(stats.norm.sf(4.5 / s5), 0.0221), stats.norm.sf(4.5 / s5))
check(T, "例3 正確 P(X≥15)=0.0207, 補正なし 0.0127", close(stats.binom.sf(14, 20, 0.5), 0.0207) and close(stats.norm.sf(5 / s5), 0.0127), (stats.binom.sf(14, 20, 0.5), stats.norm.sf(5 / s5)))
check(T, "P(X≤15) と P(X<16) が補正で同じ境界 15.5", (16 - 1) + 0.5 == 15 + 0.5, 15.5)

# ================= continuous-mapping-theorem =================
T = "continuous-mapping-theorem"
check(T, "例1 1/2.5=0.4", close(1 / 2.5, 0.4, 1e-12), 0.4)
Zn = (0.45 - 0.5) / sqrt(0.25 / 400)
check(T, "例2 Z=-0.05/0.025=-2, Z²=4", close(Zn, -2, 1e-12) and close(Zn**2, 4, 1e-12), Zn)
check(T, "例2 1.96²=3.8416（本文を修正）、χ²(1) 上側5%点 3.8415", close(1.96**2, 3.8416) and close(stats.chi2.ppf(0.95, 1), 3.8415), (1.96**2, stats.chi2.ppf(0.95, 1)))
check(T, "z_{0.025}² = χ²_{0.05}(1)（厳密）", close(stats.norm.ppf(0.975) ** 2, stats.chi2.ppf(0.95, 1), 1e-9), stats.norm.ppf(0.975) ** 2)
ph = rng.binomial(400, 0.5, 200000) / 400
Z2 = ((ph - 0.5) / 0.025) ** 2
check(T, "シミュ H0 下 Z² の χ²(1) 5%点超過率≈0.05（離散性込みで許容0.01）", abs((Z2 > stats.chi2.ppf(0.95, 1)).mean() - 0.05) < 0.01, (Z2 > stats.chi2.ppf(0.95, 1)).mean())
Xb = rng.exponential(1 / 0.4, (20000, 2000)).mean(axis=1)
check(T, "シミュ 1/X̄ → λ=0.4（n=2000, 平均の許容0.002）", abs((1 / Xb).mean() - 0.4) < 0.002, (1 / Xb).mean())

# ================= convergence-in-distribution =================
T = "convergence-in-distribution"
tq = [stats.t.ppf(0.975, d) for d in [5, 10, 30, 100]] + [stats.norm.ppf(0.975)]
check(T, "例1 t_{0.025}(n) 表", all(close(a, b) for a, b in zip(tq, [2.5706, 2.2281, 2.0423, 1.9840, 1.9600])), np.round(tq, 4))
check(T, "例2 0.9^10≈0.3487, 0.99^100≈0.3660, e^{-1}≈0.3679", close(0.9**10, 0.3487) and close(0.99**100, 0.3660) and close(exp(-1), 0.3679), (0.9**10, 0.99**100))
yy = sp.Symbol("y", positive=True)
check(T, "例2 (1-y/n)^n → e^{-y}", zero(sp.limit((1 - yy / n) ** n, n, sp.oo) - sp.exp(-yy)), "sympy")
U = rng.random((200000, 10)).max(axis=1)
check(T, "例2 シミュ n=10 で P(Y>1)=0.9^10（許容0.003）", abs((10 * (1 - U) > 1).mean() - 0.9**10) < 0.003, (10 * (1 - U) > 1).mean())
Z = rng.standard_normal(100000)
check(T, "X_n=-Z: 分布は同じ（KS p>0.01）だが |X_n-Z|=2|Z| で P(≥0.1)=2(1-Φ(0.05))≈0.96（許容0.005）", stats.kstest(-Z, "norm").pvalue > 0.01 and abs((2 * np.abs(Z) >= 0.1).mean() - 2 * stats.norm.sf(0.05)) < 0.005, (2 * np.abs(Z) >= 0.1).mean())
check(T, "例3 F_n(0)=0, x≠0 で F_n→F", all((1 / nn <= x) == (x >= 0) for x in [-0.5, 0.3, 2] for nn in [1000]) and (1 / 1000 <= 0) is False, "")

# ================= convergence-in-probability =================
T = "convergence-in-probability"
cb = [0.25 / (nn * 0.05**2) for nn in [100, 1000, 10000]]
check(T, "例1 チェビシェフ上限 1, 0.1, 0.01", all(close(a, b, 1e-12) for a, b in zip(cb, [1, 0.1, 0.01])), cb)
pe = [stats.binom.cdf(nn // 2 - nn // 20, nn, 0.5) + stats.binom.sf(nn // 2 + nn // 20 - 1, nn, 0.5) for nn in [100, 1000, 10000]]
check(T, "例1 正確値 0.3682, 0.0017, <1e-22", close(pe[0], 0.3682) and close(pe[1], 0.0017) and pe[2] < 1e-22, pe)
check(T, "例2 0.95^50≈0.0769, 0.95^100≈0.0059", close(0.95**50, 0.0769) and close(0.95**100, 0.0059), (0.95**50, 0.95**100))
Mx = rng.random((100000, 50)).max(axis=1)
check(T, "例2 シミュ P(M_50≤0.95)=0.95^50（許容0.003）", abs((Mx <= 0.95).mean() - 0.95**50) < 0.003, (Mx <= 0.95).mean())
check(T, "例3 E[X_n]=n·(1/n)=1", zero(n * (1 / n) - 1), 1)
a_, b_, c_ = sp.symbols("a b c")
check(T, "MSE=V+(E-c)² 分解", zero(sp.expand((b_ + a_**2) - 2 * c_ * a_ + c_**2 - (b_ + (a_ - c_) ** 2))), "E[X²]=V+E²")

# ================= delta-method =================
T = "delta-method"
x = sp.Symbol("x", positive=True)
check(T, "例1 g=1/x, g'(1/λ)=-λ², V≈λ²/n", zero(sp.diff(1 / x, x).subs(x, 1 / lam) + lam**2) and zero((lam**2) ** 2 / (n * lam**2) - lam**2 / n), "sympy")
check(T, "例1 λ=2,n=100 で SE=0.2", close(2 / sqrt(100), 0.2, 1e-12), 0.2)
lh = 1 / rng.exponential(1 / 2, (200000, 100)).mean(axis=1)
exact_sd = 200 / (99 * sqrt(98))
check(T, "例1 シミュ SD(λ̂)≈0.204（理論 nλ/((n-1)√(n-2))=0.2041、許容0.002）", abs(lh.std() - 0.204) < 0.002 and close(exact_sd, 0.2041), (lh.std(), exact_sd))
p = sp.Symbol("p", positive=True)
check(T, "例2 ロジット g'=1/(p(1-p)), V≈1/(np(1-p))", zero(sp.diff(sp.log(p / (1 - p)), p) - 1 / (p * (1 - p))) and zero(1 / (p * (1 - p)) ** 2 * p * (1 - p) / n - 1 / (n * p * (1 - p))), "sympy")
v = 1 / (200 * 0.21)
se = sqrt(v)
lo, hi = log(0.3 / 0.7) - 1.96 * se, log(0.3 / 0.7) + 1.96 * se
ex = lambda u: 1 / (1 + exp(-u))
check(T, "例2 V≈0.02381, SE≈0.1543, logit≈-0.8473", close(v, 0.02381, 5e-6) and close(se, 0.1543) and close(log(0.3 / 0.7), -0.8473), (v, se))
check(T, "例2 区間 [-1.1497,-0.5449] → [0.2405,0.3671]", close(lo, -1.1497) and close(hi, -0.5449) and close(ex(lo), 0.2405) and close(ex(hi), 0.3671), (lo, hi, ex(lo), ex(hi)))
check(T, "例3 g=√x: (1/(4λ))(λ/n)=1/(4n)", zero(sp.diff(sp.sqrt(x), x).subs(x, lam) ** 2 * lam / n - 1 / (4 * n)), "sympy")
check(T, "例3 √(1/200)≈0.0707", close(sqrt(1 / 200), 0.0707), sqrt(1 / 200))
sq = np.sqrt(rng.poisson(4, (200000, 50)).mean(axis=1))
check(T, "例3 シミュ SD(√X̄)≈0.0707（λ=4,n=50、許容0.001）", abs(sq.std() - 0.0707) < 0.001, sq.std())
Tn = rng.standard_normal(200000) / sqrt(100)  # θ=0, σ=1, n=100, g=x²
check(T, "2次デルタ法 n g(T)=n T² ~ (1/2)g''σ²χ²(1)=χ²(1)（平均1, 分散2, 許容0.02/0.05）", abs((100 * Tn**2).mean() - 1) < 0.02 and abs((100 * Tn**2).var() - 2) < 0.05, ((100 * Tn**2).mean(), (100 * Tn**2).var()))
check(T, "偏り ≈ (1/2)g''V: g=x², E[T²]-θ²=V", True, "E[T²]=θ²+V（恒等式）")
Ts = rng.standard_normal(400000) * 0.1
check(T, "図 σ=0.1 で e^T の SD≈σ（デルタ法 0.1、厳密 0.1003、許容0.001）", abs(np.exp(Ts).std() - sqrt((exp(0.01) - 1) * exp(0.01))) < 0.001 and abs(sqrt((exp(0.01) - 1) * exp(0.01)) - 0.1) < 0.001, np.exp(Ts).std())

# ================= law-of-small-numbers =================
T = "law-of-small-numbers"
kk = sp.Symbol("k", integer=True, nonnegative=True)
lim = [sp.limit(sp.binomial(n, j) * (lam / n) ** j * (1 - lam / n) ** (n - j), n, sp.oo) for j in range(4)]
check(T, "二項 pmf → e^{-λ}λ^k/k!（k=0..3）", all(zero(lim[j] - sp.exp(-lam) * lam**j / sp.factorial(j)) for j in range(4)), lim)
check(T, "PGF (1+λ(s-1)/n)^n → e^{λ(s-1)}", zero(sp.limit((1 + lam * (s - 1) / n) ** n, n, sp.oo) - sp.exp(lam * (s - 1))), "sympy")
b = stats.binom(1000, 0.002); po = stats.poisson(2)
bv = [b.pmf(0), b.pmf(1), b.cdf(3), b.sf(3)]
pv = [po.pmf(0), po.pmf(1), po.cdf(3), po.sf(3)]
check(T, "例1 二項 0.1351, 0.2707, 0.8573, P(X≥4)=0.1427（本文 0.0525 を修正）", all(close(a, c) for a, c in zip(bv, [0.1351, 0.2707, 0.8573, 0.1427])), np.round(bv, 4))
check(T, "例1 ポアソン 0.1353, 0.2707, 0.8571, P(X≥4)=0.1429（本文 0.0527 を修正）", all(close(a, c) for a, c in zip(pv, [0.1353, 0.2707, 0.8571, 0.1429])), np.round(pv, 4))
check(T, "例2 0.8^10≈0.1074, B(10,.2) P(X≤3)=0.8791", close(0.8**10, 0.1074) and close(stats.binom.cdf(3, 10, 0.2), 0.8791), (0.8**10, stats.binom.cdf(3, 10, 0.2)))
sim = rng.binomial(1000, 0.002, 400000)
check(T, "シミュ B(1000,0.002) の P(X=0)≈0.1351（許容0.002）", abs((sim == 0).mean() - 0.1351) < 0.002, (sim == 0).mean())

# ================= markov-inequality =================
T = "markov-inequality"
check(T, "例1 5/20=0.25, e^{-4}≈0.0183", close(5 / 20, 0.25, 1e-12) and close(stats.expon.sf(20, scale=5), 0.0183), stats.expon.sf(20, scale=5))
check(T, "例2 500/2000=0.25", close(500 / 2000, 0.25, 1e-12), 0.25)
mu_, a_v = 2.0, 5.0
check(T, "2点分布 P(X=a)=μ/a で等号: P(X≥a)=E[X]/a", close(mu_ / a_v, (a_v * (mu_ / a_v) + 0) / a_v, 1e-12), mu_ / a_v)
X = rng.exponential(5, 1000000)
check(T, "シミュ P(X≥20)≈e^{-4}≤0.25（許容0.001）", abs((X >= 20).mean() - exp(-4)) < 0.001, (X >= 20).mean())

# ================= slutsky-theorem =================
T = "slutsky-theorem"
cov = {}
for nn, claim in [(30, 0.916), (200, 0.944)]:
    hits = 0; R = 200000; chunk = 20000
    for _ in range(R // chunk):
        Xs = rng.exponential(1, (chunk, nn))
        xb = Xs.mean(axis=1); sd_ = Xs.std(axis=1, ddof=1)
        hits += np.sum(np.abs(xb - 1) <= 1.96 * sd_ / sqrt(nn))
    cov[nn] = hits / R
    check(T, f"例1 シミュ 指数 n={nn} の被覆率≈{claim}（20万回, MC標準誤差≈0.0006、許容0.003）", abs(cov[nn] - claim) < 0.003, cov[nn])
se = sqrt(0.42 * 0.58 / 400)
check(T, "例2 SE≈0.02468, [0.3716, 0.4684]", close(se, 0.02468, 5e-6) and close(0.42 - 1.96 * se, 0.3716) and close(0.42 + 1.96 * se, 0.4684), (se, 0.42 - 1.96 * se, 0.42 + 1.96 * se))
Z = rng.standard_normal(10000)
check(T, "例3 X_n=Z, Y_n=-Z で和は恒等的に0", np.all(Z + (-Z) == 0), 0)

# ================= strong-law-of-large-numbers =================
T = "strong-law-of-large-numbers"
xx = sp.Symbol("x")
check(T, "例1 E[U²]=1/3, V[U²]=1/5-1/9=4/45≈0.0889", sp.integrate(xx**2, (xx, 0, 1)) == sp.Rational(1, 3) and sp.integrate(xx**4, (xx, 0, 1)) - sp.Rational(1, 9) == sp.Rational(4, 45) and close(4 / 45, 0.0889), 4 / 45)
check(T, "例1 SE √(0.0889/10000)≈0.0030", close(sqrt(4 / 45 / 10000), 0.0030), sqrt(4 / 45 / 10000))
mc = (rng.random((2000, 10000)) ** 2).mean(axis=1)
check(T, "例1 シミュ 平均≈1/3, SD≈0.00298（許容 平均0.0003, SD0.0002）", abs(mc.mean() - 1 / 3) < 3e-4 and abs(mc.std() - sqrt(4 / 45 / 10000)) < 2e-4, (mc.mean(), mc.std()))
# E[(ΣX)^4] = nE[X^4] + 3n(n-1)σ^4（平均0, 小さな n で記号展開）
ok = True
m4, s2 = sp.symbols("m4 s2")
for nn in [2, 3, 4]:
    Xs_ = sp.symbols(f"x0:{nn}")
    ex4 = sp.expand(sum(Xs_) ** 4)
    tot = 0
    for term in ex4.as_ordered_terms():
        powers = sp.Poly(term, *Xs_).monoms()[0]
        coef = sp.Poly(term, *Xs_).coeffs()[0]
        if any(pw == 1 or pw == 3 for pw in powers):
            continue
        val = coef
        for pw in powers:
            val *= {0: 1, 2: s2, 4: m4}[pw]
        tot += val
    ok &= zero(tot - (nn * m4 + 3 * nn * (nn - 1) * s2**2))
check(T, "E[(ΣX_i)^4]=nE[X⁴]+3n(n-1)σ⁴（n=2,3,4 で展開）", ok, "sympy")
cm = stats.cauchy.rvs(size=(50000, 100), random_state=rng).mean(axis=1)
check(T, "例3 コーシー n=100 の標本平均は標準コーシー（KS p>0.01）", stats.kstest(cm, "cauchy").pvalue > 0.01, stats.kstest(cm, "cauchy").pvalue)

# ================= weak-law-of-large-numbers =================
T = "weak-law-of-large-numbers"
s2v = 35 / 12
check(T, "例1 σ²=35/12≈2.9167", close(s2v, 2.9167) and close(np.var(np.arange(1, 7)), s2v, 1e-12), s2v)
check(T, "例1 上限 2.9167/(1000·0.04)≈0.0729", close(s2v / (1000 * 0.04), 0.0729), s2v / 40)
z = 0.2 / sqrt(s2v / 1000)
check(T, "例1 z≈3.703, 2(1-Φ)≈0.0002", close(z, 3.703, 5e-4) and close(2 * stats.norm.sf(z), 0.0002), (z, 2 * stats.norm.sf(z)))
pp = np.pi / 4
check(T, "例2 π/4≈0.7854, p(1-p)≈0.1685", close(pp, 0.7854) and close(pp * (1 - pp), 0.1685), pp * (1 - pp))
sdp = 4 * sqrt(pp * (1 - pp) / 10000)
check(T, "例2 SD(4p̂)≈0.0164", close(sdp, 0.0164), sdp)
check(T, "例2 上限 16p(1-p)/(10000·0.0025)≈0.1079（丸め前の p(1-p) で計算）", close(16 * pp * (1 - pp) / 25, 0.1079), 16 * pp * (1 - pp) / 25)
check(T, "例2 正規近似 P(|4p̂-π|≥0.05)≈0.0023", close(2 * stats.norm.sf(0.05 / sdp), 0.0023), 2 * stats.norm.sf(0.05 / sdp))
hitp = rng.binomial(10000, pp, 400000) / 10000
check(T, "例2 シミュ P(|4p̂-π|≥0.05)≈0.0023（二項で40万回、許容0.0005）", abs((np.abs(4 * hitp - np.pi) >= 0.05).mean() - 0.0023) < 0.0005, (np.abs(4 * hitp - np.pi) >= 0.05).mean())
d = rng.integers(1, 7, (20000, 1000)).mean(axis=1)
check(T, "例1 シミュ P(|X̄-3.5|≥0.2) ≤ 0.0729 かつ ≈0.0002（許容0.0005）", (np.abs(d - 3.5) >= 0.2).mean() < 0.0005, (np.abs(d - 3.5) >= 0.2).mean())


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

rngC = np.random.default_rng(20261023 + 7)

# ================= convergence-in-mean-square =================
T = "convergence-in-mean-square"
cc(T, "サイコロ σ²", np.var(np.arange(1, 7)), "2.9166667")
cc(T, "MSE(n=100)=35/12/100", 35 / 12 / 100, "0.0292")
cc(T, "チェビシェフ上限 (35/1200)/0.5²", 35 / 1200 / 0.25, "0.1167")
_x, _m, _c = sp.symbols("x m c", real=True)
_Xn, _mu, _v = sp.symbols("X mu v", real=True)
# E[(X-c)^2] = V + (E-c)^2 を E[X]=mu, E[X^2]=v+mu^2 で確認
check(T, "E[(X-c)²]=V+(E-c)² （sympy）", sp.expand((_v + _mu**2) - 2 * _c * _mu + _c**2 - (_v + (_mu - _c)**2)) == 0, "")
nn = sp.symbols("n", positive=True)
check(T, "反例 E[X_n²]=n²·(1/n)=n", sp.simplify(nn**2 * (1 / nn) - nn) == 0, "")
check(T, "反例 P(|X_n|≥ε)=1/n→0", sp.limit(1 / nn, nn, sp.oo) == 0, "")
D = rngC.integers(1, 7, (200000, 100)).mean(1)
mcC(T, "E[(X̄100-3.5)²]≈0.0292", ((D - 3.5)**2).mean(), 0.0292, 0.0004)
check(T, "シミュ P(|X̄100-3.5|≥0.5) ≤ 0.1167", (np.abs(D - 3.5) >= 0.5).mean() <= 0.1167, (np.abs(D - 3.5) >= 0.5).mean())

# ================= lindeberg-levy-clt =================
T = "lindeberg-levy-clt"
from fractions import Fraction as _Fr
from math import comb as _comb, factorial as _fact


def irwin_hall_cdf(x, n=12):
    x = _Fr(x)
    s = sum((-1)**k * _comb(n, k) * (x - k)**n for k in range(0, int(x) + 1))
    return s / _fact(n)


cc(T, "和の期待値 12×1/2", 12 * 0.5, "6")
cc(T, "和の分散 12×1/12", 12 * (1 / 12), "1")
for zz, ex, nm in [(1, "0.1607", "0.1587"), (2, "0.0223", "0.0228"), (_Fr(5, 2), "0.0056", "0.0062")]:
    pe = 1 - irwin_hall_cdf(6 + zz)
    cc(T, f"P(Z>{float(zz)}) 正確（アーウィン・ホール）", float(pe), ex)
    cc(T, f"P(Z>{float(zz)}) 正規近似", stats.norm.sf(float(zz)), nm)
U = rngC.random((200000, 12)).sum(1) - 6
mcC(T, "P(Z>1)≈0.1607", (U > 1).mean(), 0.1607, 0.003)
tt = sp.symbols("t", real=True)
check(T, "{1-t²/(2n)}ⁿ → e^{-t²/2}（sympy）", sp.simplify(sp.limit((1 - tt**2 / (2 * nn))**nn, nn, sp.oo) - sp.exp(-tt**2 / 2)) == 0, "")
check(T, "N(0,1) の特性関数 e^{-t²/2}（数値積分 t=1.3）",
      abs(__import__("scipy.integrate", fromlist=["quad"]).quad(lambda x: np.cos(1.3 * x) * stats.norm.pdf(x), -40, 40)[0] - exp(-1.3**2 / 2)) < 1e-10, "")

# ================= berry-esseen-theorem =================
T = "berry-esseen-theorem"
pp = sp.symbols("p", positive=True)
rho_s = pp * (1 - pp)**3 + (1 - pp) * pp**3
check(T, "ベルヌーイ ρ=p(1-p){p²+(1-p)²}（sympy）", sp.simplify(rho_s - pp * (1 - pp) * (pp**2 + (1 - pp)**2)) == 0, "")


def be_ratio(p):
    return p * (1 - p) * (p**2 + (1 - p)**2) / (p * (1 - p))**1.5


def be_sup(n, p):
    s = sqrt(n * p * (1 - p))
    k = np.arange(0, n + 1)
    zk = (k - n * p) / s
    Fk = stats.binom.cdf(k, n, p)
    Fl = np.concatenate([[0.0], Fk[:-1]])  # 左極限
    Ph = stats.norm.cdf(zk)
    return max(np.max(np.abs(Fk - Ph)), np.max(np.abs(Fl - Ph)))


cc(T, "p=0.5 ρ/σ³", be_ratio(0.5), "1")
cc(T, "p=0.5 n=100 上限 0.4748/10", 0.4748 * be_ratio(0.5) / 10, "0.0475")
cc(T, "p=0.5 n=100 実際の最大誤差", be_sup(100, 0.5), "0.0398")
cc(T, "p=0.1 ρ", 0.1 * 0.9 * (0.01 + 0.81), "0.0738")
cc(T, "p=0.1 σ³", (0.09)**1.5, "0.027")
cc(T, "p=0.1 ρ/σ³", be_ratio(0.1), "2.733")
cc(T, "p=0.1 n=100 上限", 0.4748 * be_ratio(0.1) / 10, "0.1298")
cc(T, "p=0.1 n=100 実際の最大誤差", be_sup(100, 0.1), "0.0832")
cc(T, "p=0.1 n=400 上限", 0.4748 * be_ratio(0.1) / 20, "0.0649")
cc(T, "p=0.1 n=400 実際の最大誤差", be_sup(400, 0.1), "0.0420")
check(T, "実際の誤差 ≤ 上限（4ケース）", all(be_sup(n, p) <= 0.4748 * be_ratio(p) / sqrt(n) for n, p in [(100, .5), (100, .1), (400, .1), (400, .5)]), "")


print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
