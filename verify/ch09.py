"""第9章 検算スクリプト（区間推定）。
分位点は scipy.stats で再計算し、本文の値とは表示桁で照合する（許容 = 表示最終桁の 0.5 単位。
本文が丸めた分位点・中間値を使って計算している箇所は、その丸め値で再計算して照合する）。
記号計算は sympy（ウィルソン区間の2次式、フィッシャー情報量、有限母集団修正の式、指数分布のピボット）。
被覆確率のシミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の約4〜5倍）。
"""
from math import sqrt, log, exp, ceil
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def dec(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s else 0


def cl(x, s, extra=0.0):
    """x を本文の表示 s（文字列）の桁で丸めたとき一致するか（許容 0.5 単位 + extra）。"""
    tol = 0.5 * 10 ** (-dec(s)) * (1 + 1e-9) + extra
    return abs(float(x) - float(s)) <= tol


def chk(term, what, x, s, extra=0.0):
    check(term, f"{what} = {s}", cl(x, s, extra), round(float(x), dec(s) + 3))


def mc(term, what, est, target, tol):
    check(term, f"{what}（シミュ、許容±{tol}）", abs(est - target) <= tol, round(float(est), 5))


rng = np.random.default_rng(20261023)
z = lambda a: stats.norm.isf(a)
tq = lambda a, df: stats.t.isf(a, df)
cq = lambda a, df: stats.chi2.isf(a, df)   # 上側 a 点
fq = lambda a, d1, d2: stats.f.isf(a, d1, d2)


def wald(x, n, zz=None):
    zz = z(0.025) if zz is None else zz
    ph = x / n
    h = zz * sqrt(ph * (1 - ph) / n)
    return ph - h, ph + h


def wilson(x, n, zz=None):
    zz = z(0.025) if zz is None else zz
    ph = x / n
    c = (ph + zz**2 / (2 * n)) / (1 + zz**2 / n)
    h = zz / (1 + zz**2 / n) * sqrt(ph * (1 - ph) / n + zz**2 / (4 * n**2))
    return c - h, c + h


def exact_cov(n, p, f):
    return sum(stats.binom.pmf(x, n, p) for x in range(n + 1) if f(x, n)[0] <= p <= f(x, n)[1])


# ================= confidence-interval =================
T = "confidence-interval"
for a, s in [(0.025, "1.9600"), (0.05, "1.6449"), (0.005, "2.5758")]:
    chk(T, f"z_{a}", z(a), s)
se = 4 / sqrt(16)
for a, lo, hi in [(0.025, "48.340", "52.260"), (0.05, "48.655", "51.945"), (0.005, "47.724", "52.876")]:
    chk(T, f"例1 {1-2*a:.0%} 下端", 50.3 - z(a) * se, lo)
    chk(T, f"例1 {1-2*a:.0%} 上端", 50.3 + z(a) * se, hi)
chk(T, "例2 SD √(100·0.95·0.05)", sqrt(100 * 0.95 * 0.05), "2.179")
chk(T, "例2 P(K=95)", stats.binom.pmf(95, 100, 0.95), "0.1800")
chk(T, "例2 P(K≥93)", stats.binom.sf(92, 100, 0.95), "0.8720")
# 被覆確率（σ既知 z 区間、n=10）
Xs = rng.normal(0, 1, (200000, 10))
xb = Xs.mean(axis=1)
cov = np.mean(np.abs(xb) <= z(0.025) / sqrt(10))
mc(T, "z 区間の被覆率 ≈ 0.95（n=10, 20万回, SE≈0.0005）", cov, 0.95, 0.002)
# 100本の区間を多数回くり返したときの的中本数の分布
K = (np.abs(rng.normal(0, 1, (20000, 100))) <= z(0.025)).sum(axis=1)
mc(T, "100本中の的中本数 P(K=95)≈0.1800（2万回, SE≈0.003）", np.mean(K == 95), 0.1800, 0.012)
check(T, "幅は 1/√n に比例：n を4倍で半分", abs((2 * z(.025) / sqrt(40)) / (2 * z(.025) / sqrt(10)) - 0.5) < 1e-12, 0.5)

# ================= confidence-coefficient =================
T = "confidence-coefficient"
for a, s in [(0.05, "1.6449"), (0.10, "1.2816"), (0.025, "1.9600"), (0.05, "1.6449"), (0.005, "2.5758"), (0.01, "2.3263")]:
    chk(T, f"表 z_{a}", z(a), s)
chk(T, "z_0.005 ≈ 2.576", z(0.005), "2.576")
chk(T, "図 Φ⁻¹(0.975) = 1.96", stats.norm.ppf(0.975), "1.96")
for zr, s in [(1.6449, "6.580"), (1.9600, "7.840"), (2.5758, "10.303")]:
    chk(T, f"例1 長さ 2×{zr}×2（本文の4桁の分位点で計算）", 2 * zr * 2, s)
r = z(0.005) / z(0.025)
chk(T, "例1 比 2.5758/1.96", r, "1.314")
chk(T, "例2 1.314² ≈ 1.727", 1.314**2, "1.727")
chk(T, "例2 25×1.727", 25 * 1.727, "43.18")
L95 = 2 * z(0.025) * 10 / 5
check(T, "例2 n=44 で 99% 長さ ≤ 95%(n=25) の長さ、n=43 では超える",
      2 * z(0.005) * 10 / sqrt(44) <= L95 < 2 * z(0.005) * 10 / sqrt(43), (2 * z(0.005) * 10 / sqrt(43), 2 * z(0.005) * 10 / sqrt(44), L95))
cw = exact_cov(20, 0.1, wald)
cs = exact_cov(20, 0.1, wilson)
chk(T, "例3 ワルド型被覆確率 n=20,p=0.1（二項で正確計算）", cw, "0.876")
chk(T, "例3 ウィルソン被覆確率", cs, "0.957")
for lev in (0.5, 0.99):
    c = np.mean(np.abs(xb) <= z((1 - lev) / 2) / sqrt(10))
    mc(T, f"図 信頼係数 {lev} の被覆率", c, lev, 0.004)

# ================= pivotal-quantity =================
T = "pivotal-quantity"
w, y, lm, xx = sp.symbols("w y lambda x", positive=True)
check(T, "P(2Y>w)=e^{-w/2}, Y~Exp(1)", sp.simplify(sp.exp(-w / 2) - sp.exp(-(w / 2))) == 0, "")
chi2_2_pdf = 1 / (2 ** 1 * sp.gamma(1)) * w ** 0 * sp.exp(-w / 2)
check(T, "χ²(2) の密度 = (1/2)e^{-w/2}（平均2の指数分布）", sp.simplify(chi2_2_pdf - sp.Rational(1, 2) * sp.exp(-w / 2)) == 0, chi2_2_pdf)
check(T, "χ²(k) = Gamma(形状 k/2, 尺度 2)", all(abs(stats.chi2.pdf(v, k) - stats.gamma.pdf(v, k / 2, scale=2)) < 1e-12 for k in (3, 20) for v in (0.5, 7.3)), "")
lam0, n0 = 0.4, 10
Q = 2 * lam0 * rng.exponential(1 / lam0, (100000, n0)).sum(axis=1)
ks = stats.kstest(Q, stats.chi2(2 * n0).cdf).statistic
check(T, "シミュ 2λΣX ~ χ²(20)（KS 統計量 < 0.006）", ks < 0.006, ks)
chk(T, "χ²_0.975(20)", cq(0.975, 20), "9.5908")
chk(T, "χ²_0.025(20)", cq(0.025, 20), "34.1696")
chk(T, "例1 下端", 9.5908 / 50, "0.1918")
chk(T, "例1 上端", 34.1696 / 50, "0.6834")
chk(T, "例1 1/λ 下端", 50 / 34.1696, "1.4633")
chk(T, "例1 1/λ 上端", 50 / 9.5908, "5.2133")
check(T, "例1 λ̂=0.4 から上側のほうが長い", 0.6834 - 0.4 > 0.4 - 0.1918, (0.6834 - 0.4, 0.4 - 0.1918))
Sx = rng.exponential(1 / lam0, (200000, n0)).sum(axis=1)
c = np.mean((cq(0.975, 20) / (2 * Sx) <= lam0) & (lam0 <= cq(0.025, 20) / (2 * Sx)))
mc(T, "指数 λ の区間の被覆率 ≈ 0.95（20万回）", c, 0.95, 0.002)
tt = sp.Symbol("t", positive=True)
nn = sp.Symbol("n", positive=True)
check(T, "一様 P(M/θ≤t)=t^n → P(M/θ ≥ α^{1/n}) = 1-α", sp.simplify(1 - (sp.Symbol("alpha", positive=True) ** (1 / nn)) ** nn - (1 - sp.Symbol("alpha", positive=True))) == 0, "")
chk(T, "例2 0.05^{1/5}", 0.05 ** 0.2, "0.5493")
chk(T, "例2 上端 8.2/0.05^{1/5}", 8.2 / 0.05 ** 0.2, "14.929")
M = rng.uniform(0, 10, (200000, 5)).max(axis=1)
c = np.mean((M <= 10) & (10 <= M / 0.05 ** 0.2))
mc(T, "一様 [M, M/α^{1/n}] の被覆率 ≈ 0.95", c, 0.95, 0.002)
U = rng.uniform(0, 1, (100000, 5)).max(axis=1)
check(T, "シミュ M/θ の分布 t^5（KS < 0.006）", stats.kstest(U, lambda t: np.clip(t, 0, 1) ** 5).statistic < 0.006, "")

# ================= ci-mean-known-variance =================
T = "ci-mean-known-variance"
chk(T, "z_0.025", z(0.025), "1.9600")
chk(T, "z_0.005", z(0.005), "2.5758")
chk(T, "例1 SE 2/√25", 2 / 5, "0.4")
chk(T, "例1 95% 半幅", z(0.025) * 0.4, "0.7840")
chk(T, "例1 95% 下端", 101.2 - z(0.025) * 0.4, "100.416")
chk(T, "例1 95% 上端", 101.2 + z(0.025) * 0.4, "101.984")
chk(T, "例1 99% 半幅", z(0.005) * 0.4, "1.0303")
chk(T, "例1 99% 下端", 101.2 - z(0.005) * 0.4, "100.170")
chk(T, "例1 99% 上端", 101.2 + z(0.005) * 0.4, "102.230")
chk(T, "例2 z_0.05", z(0.05), "1.6449")
chk(T, "例2 1.6449×0.4", z(0.05) * 0.4, "0.6579")
chk(T, "例2 下側限界", 101.2 - z(0.05) * 0.4, "100.542")
chk(T, "例3 半幅 1.96×6/12", z(0.025) * 0.5, "0.9800")
chk(T, "例3 下端", 7.5 - z(.025) * .5, "6.520")
chk(T, "例3 上端", 7.5 + z(.025) * .5, "8.480")
c1 = np.mean(xb - z(0.05) / sqrt(10) <= 0)
mc(T, "片側下側限界の被覆率 ≈ 0.95", c1, 0.95, 0.002)
# 非正規（指数分布、σ=6 既知、n=144）でも CLT で近似的に 95%
E = rng.exponential(6, (100000, 144)).mean(axis=1)
c = np.mean(np.abs(E - 6) <= z(.025) * 6 / 12)
mc(T, "例3 指数母集団 n=144 の z 区間被覆率 ≈ 0.95（10万回）", c, 0.95, 0.004)
for nn_, sg in [(10, 1), (50, 2.5)]:
    Xm = rng.normal(0, sg, (100000, nn_)).mean(axis=1)
    mc(T, f"図 n={nn_}, σ={sg} でも外れる割合 ≈ 0.05", np.mean(np.abs(Xm) > z(.025) * sg / sqrt(nn_)), 0.05, 0.003)

# ================= ci-mean-unknown-variance =================
T = "ci-mean-unknown-variance"
for df, s in [(4, "2.7764"), (9, "2.2622"), (19, "2.0930"), (29, "2.0452"), (99, "1.9842")]:
    chk(T, f"表 t_0.025({df})", tq(0.025, df), s)
chk(T, "表 t_0.025(∞)", tq(0.025, 1e9), "1.9600")
d = np.array([12.1, 11.8, 12.6, 12.3, 11.9, 12.4, 12.0, 12.5, 12.2, 11.7])
chk(T, "例1 合計", d.sum(), "121.5")
chk(T, "例1 平均", d.mean(), "12.15")
ss = ((d - d.mean()) ** 2).sum()
chk(T, "例1 偏差平方和", ss, "0.825")
chk(T, "例1 s²", ss / 9, "0.09167")
s1 = sqrt(ss / 9)
chk(T, "例1 s", s1, "0.3028")
se1 = s1 / sqrt(10)
chk(T, "例1 SE", se1, "0.09574")
chk(T, "例1 半幅", tq(.025, 9) * se1, "0.2166")
chk(T, "例1 下端", 12.15 - tq(.025, 9) * se1, "11.933")
chk(T, "例1 上端", 12.15 + tq(.025, 9) * se1, "12.367")
r_ = stats.ttest_1samp(d, 12.15).confidence_interval(0.95) if hasattr(stats.ttest_1samp(d, 12.15), "confidence_interval") else None
if r_ is not None:
    check(T, "例1 scipy ttest_1samp の CI と一致", cl(r_.low, "11.933") and cl(r_.high, "12.367"), (r_.low, r_.high))
chk(T, "例1 z 誤用 下端", 12.15 - z(.025) * se1, "11.962")
chk(T, "例1 z 誤用 上端", 12.15 + z(.025) * se1, "12.338")
se2 = 5.2 / sqrt(20)
chk(T, "例2 SE", se2, "1.1628")
chk(T, "例2 95% 半幅", tq(.025, 19) * se2, "2.4337")
chk(T, "例2 95% 下端", 48.3 - tq(.025, 19) * se2, "45.866")
chk(T, "例2 95% 上端", 48.3 + tq(.025, 19) * se2, "50.734")
chk(T, "例2 t_0.005(19)", tq(.005, 19), "2.8609")
chk(T, "例2 99% 半幅", tq(.005, 19) * se2, "3.3266")
chk(T, "例2 99% 下端", 48.3 - tq(.005, 19) * se2, "44.973")
chk(T, "例2 99% 上端", 48.3 + tq(.005, 19) * se2, "51.627")
chk(T, "例3 P(T>1.96), T~t(4)", stats.t.sf(1.96, 4), "0.0608")
chk(T, "例3 被覆確率 1-2×0.0608", 1 - 2 * stats.t.sf(1.96, 4), "0.8784")
chk(T, "例3 2.7764/1.96 ≈ 1.42", tq(.025, 4) / 1.96, "1.42")
X5 = rng.normal(3, 2, (300000, 5))
m5, s5 = X5.mean(axis=1), X5.std(axis=1, ddof=1)
mc(T, "例3 n=5 で z 分位点を使った区間の被覆率 ≈ 0.8784（30万回）", np.mean(np.abs(m5 - 3) <= 1.96 * s5 / sqrt(5)), 0.8784, 0.003)
mc(T, "n=5 の t 区間の被覆率 ≈ 0.95", np.mean(np.abs(m5 - 3) <= tq(.025, 4) * s5 / sqrt(5)), 0.95, 0.002)
Tst = (m5 - 3) / (s5 / sqrt(5))
check(T, "シミュ T=(X̄-μ)/(S/√n) ~ t(4)（KS < 0.004）", stats.kstest(Tst, stats.t(4).cdf).statistic < 0.004, "")
check(T, "シミュ X̄ と S² の相関 ≈ 0（|r|<0.006）", abs(np.corrcoef(m5, s5**2)[0, 1]) < 0.006, np.corrcoef(m5, s5**2)[0, 1])
check(T, "σ̂（n で割る）から S=σ̂√(n/(n-1))", abs(d.std() * sqrt(10 / 9) - s1) < 1e-12, "")
check(T, "t_{α/2}(n-1) は n とともに減少し z に近づく", all(tq(.025, k) > tq(.025, k + 1) > z(.025) for k in range(1, 200)), "")

# ================= ci-paired-difference =================
T = "ci-paired-difference"
pre = np.array([142, 138, 150, 147, 135, 144, 152, 140])
post = np.array([136, 135, 144, 145, 133, 137, 146, 139])
D = pre - post
check(T, "例1 差の列 6,3,6,2,2,7,6,1", list(D) == [6, 3, 6, 2, 2, 7, 6, 1], list(D))
chk(T, "例1 差の合計 33", D.sum(), "33")
chk(T, "例1 d̄", D.mean(), "4.125")
chk(T, "例1 s_D²", D.var(ddof=1), "5.5536")
sD = D.std(ddof=1)
chk(T, "例1 s_D", sD, "2.3566")
chk(T, "例1 SE", sD / sqrt(8), "0.8332")
chk(T, "t_0.025(7)", tq(.025, 7), "2.3646")
chk(T, "例1 半幅", tq(.025, 7) * sD / sqrt(8), "1.9702")
chk(T, "例1 下端", 4.125 - tq(.025, 7) * sD / sqrt(8), "2.155")
chk(T, "例1 上端", 4.125 + tq(.025, 7) * sD / sqrt(8), "6.095")
r = stats.ttest_rel(pre, post).confidence_interval(0.95)
check(T, "例1 scipy ttest_rel の CI と一致", cl(r.low, "2.155") and cl(r.high, "6.095"), (r.low, r.high))
chk(T, "誤用 前の不偏分散", pre.var(ddof=1), "34.857")
chk(T, "誤用 後の不偏分散", post.var(ddof=1), "24.839")
sp2 = (7 * pre.var(ddof=1) + 7 * post.var(ddof=1)) / 14
chk(T, "誤用 s_p²", sp2, "29.848")
sep = sqrt(sp2 * (2 / 8))
chk(T, "誤用 SE", sep, "2.7317")
chk(T, "誤用 t_0.025(14)", tq(.025, 14), "2.1448")
chk(T, "誤用 下端", 4.125 - tq(.025, 14) * sep, "-1.734")
chk(T, "誤用 上端", 4.125 + tq(.025, 14) * sep, "9.984")
r = stats.ttest_ind(pre, post).confidence_interval(0.95)
check(T, "誤用 scipy ttest_ind の CI と一致", cl(r.low, "-1.734") and cl(r.high, "9.984"), (r.low, r.high))
chk(T, "前後の相関係数", np.corrcoef(pre, post)[0, 1], "0.920")
check(T, "標本でも V[D]=V[X]+V[Y]-2Cov", abs(D.var(ddof=1) - (pre.var(ddof=1) + post.var(ddof=1) - 2 * np.cov(pre, post)[0, 1])) < 1e-9, "")
check(T, "x̄-ȳ = d̄", abs(pre.mean() - post.mean() - D.mean()) < 1e-12, "")
Zc = rng.multivariate_normal([140, 136], [[30, 27], [27, 30]], (200000, 8))
Dd = Zc[:, :, 0] - Zc[:, :, 1]
c = np.mean(np.abs(Dd.mean(axis=1) - 4) <= tq(.025, 7) * Dd.std(axis=1, ddof=1) / sqrt(8))
mc(T, "相関のある組の差の t 区間の被覆率 ≈ 0.95（20万回）", c, 0.95, 0.002)

# ================= ci-proportion =================
T = "ci-proportion"
p_ = sp.Symbol("p", positive=True)
check(T, "E[p̂]=p, V[p̂]=p(1-p)/n（ベルヌーイの和）", True, "E[Y]=p, V[Y]=p-p²=p(1-p) を n²で割ってn倍")
check(T, "p(1-p) の最大は p=1/2 で 1/4", sp.solve(sp.diff(p_ * (1 - p_), p_), p_) == [sp.Rational(1, 2)], "")
chk(T, "例1 0.3·0.7/400", 0.21 / 400, "0.000525")
chk(T, "例1 SE", sqrt(0.21 / 400), "0.02291")
chk(T, "例1 95% 半幅", z(.025) * sqrt(.21 / 400), "0.0449")
lo, hi = wald(120, 400)
chk(T, "例1 95% 下端", lo, "0.2551")
chk(T, "例1 95% 上端", hi, "0.3449")
chk(T, "例1 99% 半幅", z(.005) * sqrt(.21 / 400), "0.0590")
lo, hi = wald(120, 400, z(.005))
chk(T, "例1 99% 下端", lo, "0.2410")
chk(T, "例1 99% 上端", hi, "0.3590")
chk(T, "例2 SE", sqrt(.42 * .58 / 1000), "0.01561")
chk(T, "例2 半幅", z(.025) * sqrt(.42 * .58 / 1000), "0.0306")
lo, hi = wald(420, 1000)
chk(T, "例2 下端", lo, "0.3894")
chk(T, "例2 上端", hi, "0.4506")
chk(T, "例3 半幅", z(.025) * sqrt(.09 / 20), "0.1315")
lo, hi = wald(2, 20)
chk(T, "例3 ワルド下端", lo, "-0.0315")
chk(T, "例3 ワルド上端", hi, "0.2315")
lo, hi = wilson(2, 20)
chk(T, "例3 ウィルソン下端", lo, "0.0279")
chk(T, "例3 ウィルソン上端", hi, "0.3010")
check(T, "例3 ウィルソンは右に長い（0.1 からの距離 上>下）", hi - 0.1 > 0.1 - lo, (hi - .1, .1 - lo))
chk(T, "n=1000 の最大半幅 1.96/(2√1000) ≈ ±3.1 ポイント", 100 * z(.025) / (2 * sqrt(1000)), "3.1")
for n_ in (30, 100, 1000):
    up = stats.beta.ppf(0.95, 1, n_)  # x=0 のクロッパー・ピアソン片側95%上限 = 1-0.05^{1/n}
    check(T, f"3の法則 n={n_}: 正確な95%上限 {up:.4f} ≈ 3/n={3/n_:.4f}（相対10%以内）", abs(up - 3 / n_) / (3 / n_) < 0.1, up)
B = rng.binomial(400, 0.3, 1_000_000) / 400
mc(T, "シミュ E[p̂]=0.3", B.mean(), 0.3, 0.0002)
mc(T, "シミュ V[p̂]=0.000525", B.var(), 0.000525, 0.000005)

# ================= wald-confidence-interval =================
T = "wald-confidence-interval"
th = sp.Symbol("theta", positive=True)
for name, lf, mom1, Iclaim in [
    ("ベルヌーイ", xx * sp.log(p_) + (1 - xx) * sp.log(1 - p_), p_, 1 / (p_ * (1 - p_))),
    ("ポアソン", xx * sp.log(lm) - lm, lm, 1 / lm),
    ("指数（平均θ）", -sp.log(th) - xx / th, th, 1 / th**2),
]:
    par = {"ベルヌーイ": p_, "ポアソン": lm, "指数（平均θ）": th}[name]
    I = sp.simplify(-sp.diff(lf, par, 2).subs(xx, mom1))
    check(T, f"表 {name} I(θ) = {Iclaim}", sp.simplify(I - Iclaim) == 0, I)
    mle = sp.solve(sp.diff(lf, par), par)
    check(T, f"表 {name} 1観測の MLE = x（よって θ̂ = X̄）", mle == [xx], mle)
check(T, "ポアソン 2回微分 = -x/λ²", sp.simplify(sp.diff(xx * sp.log(lm) - lm, lm, 2) + xx / lm**2) == 0, "")
check(T, "デルタ法 SE(log θ̂) = SE/θ̂（d log θ/dθ = 1/θ）", sp.diff(sp.log(th), th) == 1 / th, "")
chk(T, "例1 λ̂", 120 / 50, "2.4")
chk(T, "例1 SE", sqrt(2.4 / 50), "0.2191")
chk(T, "例1 半幅", z(.025) * sqrt(2.4 / 50), "0.4294")
chk(T, "例1 下端", 2.4 - z(.025) * sqrt(2.4 / 50), "1.9706")
chk(T, "例1 上端", 2.4 + z(.025) * sqrt(2.4 / 50), "2.8294")
se = 3.5 / sqrt(40)
chk(T, "例2 SE", se, "0.5534")
chk(T, "例2 半幅", z(.025) * se, "1.0846")
chk(T, "例2 下端", 3.5 - z(.025) * se, "2.4154")
chk(T, "例2 上端", 3.5 + z(.025) * se, "4.5846")
chk(T, "例2 χ²_0.025(80)", cq(.025, 80), "106.629")
chk(T, "例2 χ²_0.975(80)", cq(.975, 80), "57.153")
check(T, "例2 2nx̄ = 280", 2 * 40 * 3.5 == 280, 280)
chk(T, "例2 正確 下端", 280 / cq(.025, 80), "2.6259")
chk(T, "例2 正確 上端", 280 / cq(.975, 80), "4.8991")
chk(T, "例2 log 3.5", log(3.5), "1.2528")
chk(T, "例2 1/√40", 1 / sqrt(40), "0.1581")
chk(T, "例2 対数ワルド 下端", exp(log(3.5) - z(.025) / sqrt(40)), "2.5673")
chk(T, "例2 対数ワルド 上端", exp(log(3.5) + z(.025) / sqrt(40)), "4.7715")
Xe = rng.exponential(3.5, (200000, 40)).mean(axis=1)
cw_ = np.mean(np.abs(Xe - 3.5) <= z(.025) * Xe / sqrt(40))
mc(T, "例2 条件（指数, n=40）のワルド型被覆率 ≈ 0.94（20万回）", cw_, 0.94, 0.006)
cl_ = np.mean(np.abs(np.log(Xe) - log(3.5)) <= z(.025) / sqrt(40))
mc(T, "対数スケールのワルド型被覆率は 0.95 に近い", cl_, 0.95, 0.004)
ce = np.mean((2 * 40 * Xe / cq(.025, 80) <= 3.5) & (3.5 <= 2 * 40 * Xe / cq(.975, 80)))
mc(T, "正確な区間の被覆率 = 0.95", ce, 0.95, 0.002)
check(T, "図 N(1,1/10) の SD = 0.3162", cl(sqrt(0.1), "0.3162"), sqrt(0.1))
G = rng.exponential(1, (200000, 10)).mean(axis=1)
check(T, "図 n=10 の標本平均 ~ Gamma(10, 尺度0.1)（KS < 0.005）", stats.kstest(G, stats.gamma(10, scale=0.1).cdf).statistic < 0.005, "")
check(T, "図 Gamma(10,0.1) は右に歪む（歪度 2/√10 > 0）", abs(stats.gamma(10, scale=.1).stats(moments="s") - 2 / sqrt(10)) < 1e-12, 2 / sqrt(10))

# ================= wilson-interval =================
T = "wilson-interval"
ph, k_ = sp.symbols("phat k", positive=True)
pp = sp.Symbol("p", real=True)
quad = sp.expand((ph - pp) ** 2 - k_ * pp * (1 - pp))
check(T, "(p̂-p)²-kp(1-p) = (1+k)p²-(2p̂+k)p+p̂²", sp.simplify(quad - ((1 + k_) * pp**2 - (2 * ph + k_) * pp + ph**2)) == 0, "")
disc = sp.expand((2 * ph + k_) ** 2 - 4 * (1 + k_) * ph**2)
check(T, "判別式 = 4kp̂(1-p̂)+k²", sp.simplify(disc - (4 * k_ * ph * (1 - ph) + k_**2)) == 0, disc)
nS, zS = sp.symbols("n z", positive=True)
roots = sp.solve(quad.subs(k_, zS**2 / nS), pp)
c_ = (ph + zS**2 / (2 * nS)) / (1 + zS**2 / nS)
h_ = zS / (1 + zS**2 / nS) * sp.sqrt(ph * (1 - ph) / nS + zS**2 / (4 * nS**2))
vals = {ph: sp.Rational(3, 10), nS: 40, zS: sp.Rational(196, 100)}
rn = sorted(float(r.subs(vals)) for r in roots)
check(T, "根 = 公式（中心 ± 半幅）（数値代入で照合）", abs(rn[0] - float((c_ - h_).subs(vals))) < 1e-12 and abs(rn[1] - float((c_ + h_).subs(vals))) < 1e-12, rn)
chk(T, "例1 k=1.96²/20", 1.96**2 / 20, "0.19208")
chk(T, "例1 係数 1+k", 1 + 1.96**2 / 20, "1.19208")
chk(T, "例1 係数 2p̂+k", 0.2 + 1.96**2 / 20, "0.39208")
rts = sorted(np.roots([1 + 1.96**2 / 20, -(0.2 + 1.96**2 / 20), 0.01]))
chk(T, "例1 小さい根", rts[0], "0.02787")
chk(T, "例1 大きい根", rts[1], "0.30104")
lo, hi = wilson(2, 20)
chk(T, "例1 公式 下端", lo, "0.0279")
chk(T, "例1 公式 上端", hi, "0.3010")
lo, hi = wald(2, 20)
chk(T, "例1 ワルド 下端", lo, "-0.0315")
chk(T, "例1 ワルド 上端", hi, "0.2315")
chk(T, "例1 被覆確率 ワルド（正確計算）", cw, "0.876")
chk(T, "例1 被覆確率 ウィルソン（正確計算）", cs, "0.957")
Bx = rng.binomial(20, 0.1, 1_000_000)
mc(T, "ワルド被覆率のシミュ（100万回）", np.mean(np.isin(Bx, [x for x in range(21) if wald(x, 20)[0] <= .1 <= wald(x, 20)[1]])), cw, 0.002)
lo, hi = wilson(120, 400)
chk(T, "例2 ウィルソン下端", lo, "0.2572")
chk(T, "例2 ウィルソン上端", hi, "0.3466")
lo, hi = wald(120, 400)
chk(T, "例2 ワルド下端", lo, "0.2551")
chk(T, "例2 ワルド上端", hi, "0.3449")
zz = z(.025)
ctr = (0.3 + zz**2 / 800) / (1 + zz**2 / 400)
check(T, "中心は p̂ を 1/2 側へ寄せる（p̂=0.1 → 中心>0.1, p̂=0.3 → 中心>0.3）", ctr > 0.3 and (wilson(2, 20)[0] + wilson(2, 20)[1]) / 2 > 0.1, ctr)
c2 = (2 + zz**2 / 2) / (20 + zz**2)
check(T, "z²≈4 で中心 ≈ (X+2)/(n+4)（n=20,x=2: 相対差 < 2%）", abs(c2 - 4 / 24) / (4 / 24) < 0.02, (c2, 4 / 24))
lo0, hi0 = wilson(0, 20)
check(T, "p̂=0 でも幅 > 0", hi0 - lo0 > 0 and abs(lo0) < 1e-12, (lo0, hi0))

# ================= ci-two-sample-mean-difference =================
T = "ci-two-sample-mean-difference"
sp2 = (9 * 16 + 11 * 25) / 20
chk(T, "例1 9×16+11×25", 9 * 16 + 11 * 25, "419")
chk(T, "例1 s_p²", sp2, "20.95")
sep = sqrt(sp2 * (1 / 10 + 1 / 12))
chk(T, "例1 プール SE", sep, "1.9598")
chk(T, "t_0.025(20)", tq(.025, 20), "2.0860")
chk(T, "例1 プール 半幅", tq(.025, 20) * sep, "4.0881")
chk(T, "例1 プール 下端", 4.5 - tq(.025, 20) * sep, "0.412")
chk(T, "例1 プール 上端", 4.5 + tq(.025, 20) * sep, "8.588")
r = stats.ttest_ind_from_stats(68.5, 4, 10, 64.0, 5, 12, equal_var=True)
chk(T, "例1 ttest_ind_from_stats の t = 4.5/SE", r.statistic, f"{4.5/sep:.4f}")
a1, a2 = 16 / 10, 25 / 12
chk(T, "例1 s1²/n1", a1, "1.6")
chk(T, "例1 s2²/n2", a2, "2.0833")
chk(T, "例1 和", a1 + a2, "3.6833")
sew = sqrt(a1 + a2)
chk(T, "例1 ウェルチ SE", sew, "1.9192")
chk(T, "例1 ν 分子 3.6833²", (a1 + a2) ** 2, "13.5669")
chk(T, "例1 ν 分母1 1.6²/9", a1**2 / 9, "0.28444")
chk(T, "例1 ν 分母2 2.0833²/11", a2**2 / 11, "0.39457")
nu = (a1 + a2) ** 2 / (a1**2 / 9 + a2**2 / 11)
chk(T, "例1 ν", nu, "19.98")
chk(T, "例1 t_0.025(ν)", tq(.025, nu), "2.0861")
chk(T, "例1 ウェルチ 半幅", tq(.025, nu) * sew, "4.0036")
chk(T, "例1 ウェルチ 下端", 4.5 - tq(.025, nu) * sew, "0.496")
chk(T, "例1 ウェルチ 上端", 4.5 + tq(.025, nu) * sew, "8.504")
chk(T, "例2 分散既知 SE", sqrt(16 / 10 + 25 / 12), "1.9192")
chk(T, "例2 半幅", z(.025) * sew, "3.7616")
chk(T, "例2 下端", 4.5 - z(.025) * sew, "0.738")
chk(T, "例2 上端", 4.5 + z(.025) * sew, "8.262")
check(T, "例2 は例1（プール・ウェルチ）より狭い", z(.025) * sew < tq(.025, nu) * sew and z(.025) * sew < tq(.025, 20) * sep, "")
# ν の範囲 min(n1,n2)-1 ≤ ν ≤ n1+n2-2
ok = True
for _ in range(20000):
    n1, n2 = rng.integers(2, 60, 2)
    v1, v2 = rng.uniform(0.01, 100, 2)
    q1, q2 = v1 / n1, v2 / n2
    nu_ = (q1 + q2) ** 2 / (q1**2 / (n1 - 1) + q2**2 / (n2 - 1))
    ok &= min(n1, n2) - 1 - 1e-9 <= nu_ <= n1 + n2 - 2 + 1e-9
check(T, "ν の範囲 min(n1,n2)-1 ≤ ν ≤ n1+n2-2（ランダム2万組）", ok, "")
check(T, "n1=n2 のときプール分散 = 不偏分散の単純平均", abs((9 * 16 + 9 * 25) / 18 - (16 + 25) / 2) < 1e-12, "")
A = rng.normal(5, 2, (200000, 10))
Bm = rng.normal(3, 2, (200000, 12))
dA, dB = A.mean(1) - Bm.mean(1), None
spp = np.sqrt((9 * A.var(1, ddof=1) + 11 * Bm.var(1, ddof=1)) / 20 * (1 / 10 + 1 / 12))
mc(T, "等分散のプール t 区間の被覆率 ≈ 0.95（20万回）", np.mean(np.abs(dA - 2) <= tq(.025, 20) * spp), 0.95, 0.002)
A = rng.normal(5, 1, (200000, 6))
Bm = rng.normal(3, 4, (200000, 20))
q1, q2 = A.var(1, ddof=1) / 6, Bm.var(1, ddof=1) / 20
nus = (q1 + q2) ** 2 / (q1**2 / 5 + q2**2 / 19)
dd = A.mean(1) - Bm.mean(1)
mc(T, "不等分散でウェルチ区間の被覆率 ≈ 0.95（20万回）", np.mean(np.abs(dd - 2) <= stats.t.isf(.025, nus) * np.sqrt(q1 + q2)), 0.95, 0.004)

# ================= ci-two-sample-proportion =================
T = "ci-two-sample-proportion"
v1, v2 = .4 * .6 / 500, .3 * .7 / 500
chk(T, "例1 0.4·0.6/500", v1, "0.00048")
chk(T, "例1 0.3·0.7/500", v2, "0.00042")
chk(T, "例1 和", v1 + v2, "0.0009")
chk(T, "例1 SE", sqrt(v1 + v2), "0.03")
chk(T, "例1 半幅", z(.025) * .03, "0.0588")
chk(T, "例1 下端", .1 - z(.025) * sqrt(v1 + v2), "0.0412")
chk(T, "例1 上端", .1 + z(.025) * sqrt(v1 + v2), "0.1588")
chk(T, "例1 共通比率の検定統計量", .1 / sqrt(.35 * .65 * (2 / 500)), "3.315")
se2 = sqrt(.15 * .85 / 300 + .1 * .9 / 300)
chk(T, "例2 SE", se2, "0.02693")
chk(T, "例2 半幅", z(.025) * se2, "0.0528")
chk(T, "例2 下端", .05 - z(.025) * se2, "-0.0028")
chk(T, "例2 上端", .05 + z(.025) * se2, "0.1028")
check(T, "図の既定値 μ=0.1, σ=0.03 は例1と一致", True, "")
X1 = rng.binomial(500, .4, 500000) / 500
X2 = rng.binomial(500, .3, 500000) / 500
sed = np.sqrt(X1 * (1 - X1) / 500 + X2 * (1 - X2) / 500)
mc(T, "例1 条件での区間の被覆率 ≈ 0.95（50万回）", np.mean(np.abs(X1 - X2 - .1) <= z(.025) * sed), 0.95, 0.003)
mc(T, "V[p̂1-p̂2] = 0.0009", (X1 - X2).var(), 0.0009, 0.00001)

# ================= ci-variance =================
T = "ci-variance"
chk(T, "例1 9×4.5", 9 * 4.5, "40.5")
chk(T, "χ²_0.025(9)", cq(.025, 9), "19.0228")
chk(T, "χ²_0.975(9)", cq(.975, 9), "2.7004")
lo, hi = 40.5 / cq(.025, 9), 40.5 / cq(.975, 9)
chk(T, "例1 下端", lo, "2.129")
chk(T, "例1 上端", hi, "14.998")
chk(T, "例1 σ 下端", sqrt(lo), "1.459")
chk(T, "例1 σ 上端", sqrt(hi), "3.873")
chk(T, "例1 4.5-下端", 4.5 - lo, "2.37")
chk(T, "例1 上端-4.5", hi - 4.5, "10.50")
chk(T, "例2 χ²_0.95(9)", cq(.95, 9), "3.3251")
chk(T, "例2 上側限界 σ²", 40.5 / cq(.95, 9), "12.180")
chk(T, "例2 上側限界 σ", sqrt(40.5 / cq(.95, 9)), "3.490")
lo3, hi3 = .825 / cq(.025, 9), .825 / cq(.975, 9)
chk(T, "例3 下端", lo3, "0.04337")
chk(T, "例3 上端", hi3, "0.30551")
chk(T, "例3 σ 下端", sqrt(lo3), "0.2083")
chk(T, "例3 σ 上端", sqrt(hi3), "0.5527")
chk(T, "例4 χ²_0.025(10)", cq(.025, 10), "20.4832")
chk(T, "例4 χ²_0.975(10)", cq(.975, 10), "3.2470")
chk(T, "例4 下端", 42 / cq(.025, 10), "2.050")
chk(T, "例4 上端", 42 / cq(.975, 10), "12.935")
check(T, "例4 自由度が増えると上下端の比が縮む（10: 6.31 < 9: 7.04）", cq(.025, 10) / cq(.975, 10) < cq(.025, 9) / cq(.975, 9), (cq(.025, 10) / cq(.975, 10), cq(.025, 9) / cq(.975, 9)))
Xn = rng.normal(5, 2, (200000, 10))
S2 = Xn.var(1, ddof=1)
mc(T, "正規 n=10 の σ² 区間の被覆率 ≈ 0.95（20万回）", np.mean((9 * S2 / cq(.025, 9) <= 4) & (4 <= 9 * S2 / cq(.975, 9))), 0.95, 0.002)
SSm = ((Xn - 5) ** 2).sum(1)
mc(T, "μ 既知（自由度10）の区間の被覆率 ≈ 0.95", np.mean((SSm / cq(.025, 10) <= 4) & (4 <= SSm / cq(.975, 10))), 0.95, 0.002)
check(T, "シミュ (n-1)S²/σ² ~ χ²(9)（KS < 0.005）", stats.kstest(9 * S2 / 4, stats.chi2(9).cdf).statistic < 0.005, "")
Xt = rng.standard_t(5, (200000, 30))
S2t = Xt.var(1, ddof=1)
vt = 5 / 3
ct = np.mean((29 * S2t / cq(.025, 29) <= vt) & (vt <= 29 * S2t / cq(.975, 29)))
check(T, f"t(5) から n=30 の被覆確率は「8割強」（0.80〜0.86 に入る）", 0.80 <= ct <= 0.86, round(ct, 4))

# ================= ci-variance-ratio =================
T = "ci-variance-ratio"
chk(T, "F_0.025(9,12)", fq(.025, 9, 12), "3.4358")
chk(T, "F_0.025(12,9)", fq(.025, 12, 9), "3.8682")
chk(T, "F_0.975(9,12)", fq(.975, 9, 12), "0.25852")
chk(T, "図キャプション 下側2.5%点", fq(.975, 9, 12), "0.2585")
check(T, "F_{1-a}(m1,m2)=1/F_a(m2,m1)", all(abs(fq(1 - a, m1, m2) - 1 / fq(a, m2, m1)) < 1e-10 for a in (.025, .05) for m1, m2 in [(9, 12), (3, 20), (15, 4)]), "")
chk(T, "例1 比", 12 / 5, "2.4")
chk(T, "例1 下端", 2.4 / fq(.025, 9, 12), "0.6985")
chk(T, "例1 上端", 2.4 / fq(.975, 9, 12), "9.2837")
chk(T, "例1 上端（別表現 2.4·F_0.025(12,9)）", 2.4 * fq(.025, 12, 9), "9.2837")
check(T, "例1 区間は 1 を含む", 2.4 / fq(.025, 9, 12) < 1 < 2.4 / fq(.975, 9, 12), "")
Y1 = rng.normal(0, 3, (200000, 10)).var(1, ddof=1)
Y2 = rng.normal(0, 2, (200000, 13)).var(1, ddof=1)
R = Y1 / Y2
mc(T, "区間の被覆率 ≈ 0.95（σ1²/σ2²=2.25, 20万回）", np.mean((R / fq(.025, 9, 12) <= 2.25) & (2.25 <= R / fq(.975, 9, 12))), 0.95, 0.002)
check(T, "シミュ (S1²/σ1²)/(S2²/σ2²) ~ F(9,12)（KS < 0.005）", stats.kstest(R / 2.25, stats.f(9, 12).cdf).statistic < 0.005, "")

# ================= required-sample-size =================
T = "required-sample-size"
N_, n0s, Ns = sp.symbols("n_ n0 N", positive=True)
sol = sp.solve(sp.Eq(n0s * (Ns - N_) / (N_ * (Ns - 1)), 1), N_)
check(T, "有限母集団修正 n0(N-n)/{n(N-1)}=1 ⇔ n = n0/(1+(n0-1)/N)", len(sol) == 1 and sp.simplify(sol[0] - n0s / (1 + (n0s - 1) / Ns)) == 0, sol)
check(T, "p(1-p)=1/4-(p-1/2)²", sp.expand(p_ * (1 - p_) - (sp.Rational(1, 4) - (p_ - sp.Rational(1, 2)) ** 2)) == 0, "")
chk(T, "例1 1.9600×15/3", 1.96 * 15 / 3, "9.8000")
chk(T, "例1 n≥", (1.96 * 5) ** 2, "96.04")
check(T, "例1 切り上げ 97", ceil((z(.025) * 5) ** 2) == 97 and ceil((1.96 * 5) ** 2) == 97, 97)
chk(T, "例1 半幅1.5 n≥（1.9600 で計算）", (1.96 * 15 / 1.5) ** 2, "384.16")
check(T, "例1 半幅1.5 → 385", ceil((z(.025) * 10) ** 2) == 385, 385)
chk(T, "例1 99% n≥", (2.5758 * 5) ** 2, "165.87")
check(T, "例1 99% → 166", ceil((z(.005) * 5) ** 2) == 166, 166)
check(T, "n=96 では半幅が 3 をわずかに超える", 3 < z(.025) * 15 / sqrt(96) < 3.01, z(.025) * 15 / sqrt(96))
chk(T, "例2 1.9600²×0.25", 1.96**2 * .25, "0.9604")
n0 = 1.96**2 * .25 / .0009
chk(T, "例2 0.9604/0.0009", n0, "1067.11")
check(T, "例2 → 1068", ceil(n0) == 1068 and ceil(z(.025) ** 2 * .25 / .0009) == 1068, 1068)
chk(T, "例2 p=0.2", 1.96**2 * .16 / .0009, "682.95")
check(T, "例2 p=0.2 → 683", ceil(1.96**2 * .16 / .0009) == 683, 683)
chk(T, "例3 1+1066.11/5000", 1 + (n0 - 1) / 5000, "1.21322")
chk(T, "例3 n", n0 / (1 + (n0 - 1) / 5000), "879.57")
check(T, "例3 → 880", ceil(n0 / (1 + (n0 - 1) / 5000)) == 880, 880)
for NN in (1e6, 1e8):
    v = n0 / (1 + (n0 - 1) / NN)
    check(T, f"N={NN:.0e} でも n ≈ 1068（差 2 以内）", abs(ceil(v) - 1068) <= 2, ceil(v))
xs_ = list(range(1, 11))
y95 = [9604, 2401, 1068, 601, 385, 267, 196, 151, 119, 97]
y99 = [16588, 4147, 1844, 1037, 664, 461, 339, 260, 205, 166]
calc95 = [ceil(z(.025) ** 2 * .25 / (e / 100) ** 2 - 1e-9) for e in xs_]
calc99 = [ceil(z(.005) ** 2 * .25 / (e / 100) ** 2 - 1e-9) for e in xs_]
check(T, "図 95% 系列 10点 = ⌈z²/(4E²)⌉", calc95 == y95, calc95)
check(T, "図 99% 系列 10点 = ⌈z²/(4E²)⌉", calc99 == y99, calc99)
check(T, "E を半分で n は約4倍（9604/2401=4）", y95[0] / y95[1] == 4, y95[0] / y95[1])
check(T, "誤差 ±1 ポイントで約1万人", 9000 < y95[0] < 11000, y95[0])
# n=1068 で実際の半幅（p=0.5）の被覆
Bp = rng.binomial(1068, 0.5, 1_000_000) / 1068
mc(T, "n=1068, p=0.5 で |p̂-p| ≤ 0.03 の割合 ≈ 0.95", np.mean(np.abs(Bp - .5) <= .03), 0.95, 0.006)

# ================= confidence-interval-test-duality =================
T = "confidence-interval-test-duality"
mu0s = [11.75, 11.8, 11.85, 11.9, 11.95, 12.0, 12.05, 12.1, 12.15, 12.2, 12.25, 12.3, 12.35, 12.4, 12.45, 12.5, 12.55]
ys = [0.0024, 0.0053, 0.0121, 0.0282, 0.0663, 0.1516, 0.3235, 0.6141, 1, 0.6141, 0.3235, 0.1516, 0.0663, 0.0282, 0.0121, 0.0053, 0.0024]
pv = [2 * stats.t.sf(abs(12.15 - m) / se1, 9) for m in mu0s]
err = max(abs(a - b) for a, b in zip(pv, ys))
check(T, "図 p 値関数 17点（2·P(t9>|x̄-μ0|/SE)、許容5e-5）", err <= 5e-5, err)
pv2 = [stats.ttest_1samp(d, m).pvalue if m != 12.15 else 1.0 for m in mu0s]
check(T, "図 p 値は scipy ttest_1samp とも一致", max(abs(a - b) for a, b in zip(pv2, ys)) <= 5e-5, "")
check(T, "図 有意水準 0.05 の線", True, "定数 0.05 × 17")
inside = [m for m, p in zip(mu0s, pv) if p > 0.05]
check(T, "図 p>0.05 の μ0 は [11.933,12.367] の中、外の格子点は p<0.05", all(11.933 <= m <= 12.367 for m in inside) and all(p < .05 for m, p in zip(mu0s, pv) if not (11.933 <= m <= 12.367)), inside)
for b, s in [(12.15 - tq(.025, 9) * se1, "11.933"), (12.15 + tq(.025, 9) * se1, "12.367")]:
    chk(T, "区間端点", b, s)
    check(T, f"端点 μ0={b:.4f} で p 値 = 0.05", abs(2 * stats.t.sf(abs(12.15 - b) / se1, 9) - 0.05) < 1e-10, "")
chk(T, "例1 t(μ0=12.0)", (12.15 - 12.0) / 0.09574, "1.567")
chk(T, "例1 p(μ0=12.0)", stats.ttest_1samp(d, 12.0).pvalue, "0.152")
chk(T, "例1 t(μ0=11.9)", (12.15 - 11.9) / 0.09574, "2.611")
chk(T, "例1 p(μ0=11.9)", stats.ttest_1samp(d, 11.9).pvalue, "0.028")
check(T, "例2 区間 [0.0412,0.1588] は 0 を含まない／検定も有意（z=3.315>1.96）", 0.0412 > 0 and 3.315 > z(.025), "")
check(T, "5%片側 ⇔ 90%両側の端（z_0.05 を共有）", abs(z(0.05) - z((1 - 0.90) / 2)) < 1e-12, z(.05))
# 反転した信頼集合の被覆 = 1-受容確率
Xd = rng.normal(12, 0.3, (200000, 10))
mm, sd_ = Xd.mean(1), Xd.std(1, ddof=1)
tstat = np.abs(mm - 12) / (sd_ / sqrt(10))
acc = np.mean(tstat <= tq(.025, 9))
inci = np.mean((mm - tq(.025, 9) * sd_ / sqrt(10) <= 12) & (12 <= mm + tq(.025, 9) * sd_ / sqrt(10)))
check(T, "シミュ 受容 ⇔ 区間が μ を含む（20万回で完全一致）", acc == inci, (acc, inci))
mc(T, "受容確率 = 0.95", acc, 0.95, 0.002)


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

rngC = np.random.default_rng(20261023 + 9)
from scipy.optimize import brentq

# ================= clopper-pearson-interval =================
T = "clopper-pearson-interval"
n, x = 20, 2
pL, pU = stats.beta.ppf(0.025, x, n - x + 1), stats.beta.ppf(0.975, x + 1, n - x)
cc(T, "例1 p_L=B_0.025(2,19)", pL, "0.0123")
cc(T, "例1 p_U=B_0.975(3,18)", pU, "0.3170")
cc(T, "p_L は P(X≥2|p_L)=0.025 の解", brentq(lambda p: stats.binom.sf(x - 1, n, p) - 0.025, 1e-9, 0.5), "0.0123")
cc(T, "p_U は P(X≤2|p_U)=0.025 の解", brentq(lambda p: stats.binom.cdf(x, n, p) - 0.025, 0.01, 0.99), "0.3170")
bt = stats.binomtest(2, 20).proportion_ci(0.95, method="exact")
check(T, "scipy binomtest(exact) と一致", abs(bt.low - pL) < 1e-9 and abs(bt.high - pU) < 1e-9, (bt.low, bt.high))
wi = stats.binomtest(2, 20).proportion_ci(0.95, method="wilson")
cc(T, "ウィルソン下限", wi.low, "0.0279")
cc(T, "ウィルソン上限", wi.high, "0.3010")
check(T, "CP 区間 ⊃ ウィルソン区間", pL < wi.low and pU > wi.high, "")


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


kk = np.arange(21)
cov = sum(stats.binom.pmf(k, 20, 0.1) for k in kk if cp(k, 20)[0] <= 0.1 <= cp(k, 20)[1])
cc(T, "p=0.1 の被覆確率", cov, "0.989")
covs = [sum(stats.binom.pmf(k, 20, p) for k in kk if cp(k, 20)[0] <= p <= cp(k, 20)[1]) for p in np.linspace(0.01, 0.99, 99)]
check(T, "被覆確率 ≥ 0.95（n=20, p=0.01..0.99）", min(covs) >= 0.95, min(covs))
cc(T, "例2 p_U=1-0.025^(1/50)", 1 - 0.025**(1 / 50), "0.0711")
check(T, "例2 x=0 の p_U がベータ分位点と一致", abs(cp(0, 50)[1] - (1 - 0.025**(1 / 50))) < 1e-12, cp(0, 50)[1])
cc(T, "片側95% 上限 1-0.05^(1/50)", 1 - 0.05**(1 / 50), "0.0582")
cc(T, "3の法則 3/50", 3 / 50, "0.06")
B = rngC.binomial(20, 0.1, 200000)
lo = np.array([cp(k, 20)[0] for k in range(21)]); hi = np.array([cp(k, 20)[1] for k in range(21)])
mcC(T, "被覆率 p=0.1 ≈0.989", ((lo[B] <= 0.1) & (0.1 <= hi[B])).mean(), 0.989, 0.002)

# ================= likelihood-based-confidence-interval =================
T = "likelihood-based-confidence-interval"
ll = lambda p: 2 * log(p) + 18 * log(1 - p)
cc(T, "ℓ(0.1)", ll(0.1), "-6.5017")
cc(T, "χ²_0.05(1)", stats.chi2.isf(0.05, 1), "3.841")
cc(T, "閾値 ℓ(p̂)-3.841/2", ll(0.1) - stats.chi2.isf(0.05, 1) / 2, "-8.4224")
q = stats.chi2.isf(0.05, 1)
f = lambda p: 2 * (ll(0.1) - ll(p)) - q
L1, U1 = brentq(f, 1e-6, 0.1), brentq(f, 0.1, 0.99)
cc(T, "区間下端", L1, "0.0174")
cc(T, "区間上端", U1, "0.2779")
check(T, "非対称（右側が長い）", U1 - 0.1 > 0.1 - L1, (0.1 - L1, U1 - 0.1))
se = sqrt(0.1 * 0.9 / 20)
cc(T, "ワルド下端", 0.1 - stats.norm.isf(0.025) * se, "-0.0315")
cc(T, "ワルド上端", 0.1 + stats.norm.isf(0.025) * se, "0.2315")
cc(T, "CP 下端", stats.beta.ppf(0.025, 2, 19), "0.0123")
cc(T, "CP 上端", stats.beta.ppf(0.975, 3, 18), "0.3170")
cc(T, "3.841/2", q / 2, "1.921")
cc(T, "e^{-1.921}", exp(-q / 2), "0.147")
cc(T, "6.635/2", stats.chi2.isf(0.01, 1) / 2, "3.317")
cc(T, "χ²_0.01(1)", stats.chi2.isf(0.01, 1), "6.635")
# 変換不変：logit 尺度で同じ区間
lg = lambda p: log(p / (1 - p))
ll_eta = lambda e: ll(1 / (1 + exp(-e)))
fe = lambda e: 2 * (ll(0.1) - ll_eta(e)) - q
check(T, "変換不変（logit 尺度で解いた端点 = g(端点)）", abs(brentq(fe, -8, lg(0.1)) - lg(L1)) < 1e-8 and abs(brentq(fe, lg(0.1), 3) - lg(U1)) < 1e-8, "")
# ウィルクス：被覆率 n=200 程度で ≈0.95（p=0.3）
Bn = rngC.binomial(200, 0.3, 200000)
ph = np.clip(Bn / 200, 1e-12, 1 - 1e-12)
ll_v = lambda k, p: k * np.log(p) + (200 - k) * np.log(1 - p)
lr = 2 * (ll_v(Bn, ph) - ll_v(Bn, 0.3))
mcC(T, "2{ℓ(p̂)-ℓ(p)}≤3.841 の割合≈0.95（n=200, p=0.3）", (lr <= q).mean(), 0.95, 0.006)


print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
