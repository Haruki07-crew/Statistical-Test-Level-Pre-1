"""第4章 検算スクリプト（変数変換とデータ変換）。
厳密計算は sympy、数値は numpy / scipy。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
import math
import numpy as np
import sympy as sp
from scipy import stats, optimize, integrate

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


rng = np.random.default_rng(20261023)
N = 1_000_000
x, y, z, u, v = sp.symbols("x y z u v", real=True)
yp = sp.symbols("y", positive=True)

# ============ change-of-variables ============
T = "change-of-variables"
# 例1: Y=-2 log X, X~U(0,1)
xinv = sp.solve(sp.Eq(-2 * sp.log(x), yp), x)[0]
fY = sp.simplify(1 * sp.Abs(sp.diff(xinv, yp)))
check(T, "例1 x=e^{-y/2}, f_Y=e^{-y/2}/2（sympy）", sp.simplify(xinv - sp.exp(-yp / 2)) == 0 and sp.simplify(fY - sp.exp(-yp / 2) / 2) == 0, fY)
check(T, "例1 f_Y = χ²(2) 密度・積分1", np.allclose([float(fY.subs(yp, t)) for t in (0.3, 1, 4)], stats.chi2(2).pdf([0.3, 1, 4])) and sp.integrate(fY, (yp, 0, sp.oo)) == 1, True)
# CDF 経由の導出（減少関数）: F_Y(y)=1-F_X(e^{-y/2}) を微分
FY = 1 - sp.exp(-yp / 2)
check(T, "例1 分布関数を微分しても同じ密度", sp.simplify(sp.diff(FY, yp) - fY) == 0, sp.diff(FY, yp))
check(T, "例1 P(Y>3)=e^{-1.5}≈0.2231", close(math.exp(-1.5), 0.2231) and close(stats.chi2(2).sf(3), 0.2231), round(math.exp(-1.5), 5))
Ys = -2 * np.log(rng.uniform(size=N))
check(T, "例1 シミュ P(Y>3)（許容0.002）, 平均2（許容0.01）", abs((Ys > 3).mean() - 0.2231) < 0.002 and abs(Ys.mean() - 2) < 0.01, (round((Ys > 3).mean(), 4), round(Ys.mean(), 4)))
# 例2: Z^2
phi = lambda t: sp.exp(-t**2 / 2) / sp.sqrt(2 * sp.pi)
r1, r2 = sp.sqrt(yp), -sp.sqrt(yp)
jac = sp.Abs(sp.diff(r1, yp))
fZ2 = sp.simplify(phi(r1) * jac + phi(r2) * sp.Abs(sp.diff(r2, yp)))
check(T, "例2 |dz/dy|=1/(2√y), f_Y=e^{-y/2}/√(2πy)（sympy）", sp.simplify(jac - 1 / (2 * sp.sqrt(yp))) == 0 and sp.simplify(fZ2 - sp.exp(-yp / 2) / sp.sqrt(2 * sp.pi * yp)) == 0, fZ2)
check(T, "例2 χ²(1) 密度と一致, f(1)≈0.2420", np.allclose([float(fZ2.subs(yp, t)) for t in (0.2, 1, 3)], stats.chi2(1).pdf([0.2, 1, 3])) and close(float(fZ2.subs(yp, 1)), 0.2420), round(float(fZ2.subs(yp, 1)), 5))
half = sp.integrate(phi(r1) * jac, (yp, 0, sp.oo))
check(T, "片方の枝だけだと積分は1/2", sp.simplify(half - sp.Rational(1, 2)) == 0, half)
Z = rng.standard_normal(N)
check(T, "例2 シミュ P(Z²≤1)=0.6827（許容0.002）", abs((Z**2 <= 1).mean() - stats.chi2(1).cdf(1)) < 0.002, round((Z**2 <= 1).mean(), 4))
# 例3: 和と比
xs, ys = u * v, u * (1 - v)
Jm = sp.Matrix([[sp.diff(xs, u), sp.diff(xs, v)], [sp.diff(ys, u), sp.diff(ys, v)]])
J = sp.simplify(Jm.det())
check(T, "例3 ヤコビ行列 [[v,u],[1-v,-u]], J=-u", Jm == sp.Matrix([[v, u], [1 - v, -u]]) and sp.simplify(J + u) == 0, J)
fUV = sp.simplify(sp.exp(-(xs + ys)) * u)
check(T, "例3 f_UV=u e^{-u}（vに依存しない）, 積分1", sp.simplify(fUV - u * sp.exp(-u)) == 0 and sp.integrate(fUV.subs(u, sp.Symbol("w", positive=True)), (sp.Symbol("w", positive=True), 0, sp.oo)) == 1, fUV)
check(T, "例3 P(U≤2)=1-3e^{-2}≈0.5940 = Ga(2,1)", close(1 - 3 * math.exp(-2), 0.5940) and close(stats.gamma(2).cdf(2), 0.5940), round(1 - 3 * math.exp(-2), 5))
X1, X2 = rng.exponential(size=N), rng.exponential(size=N)
U_, V_ = X1 + X2, X1 / (X1 + X2)
check(T, "例3 シミュ P(U≤2)（許容0.002）, V~U(0,1)（KS p>0.01）, 相関≈0（許容0.005）",
      abs((U_ <= 2).mean() - 0.5940) < 0.002 and stats.kstest(V_[:100000], "uniform").pvalue > 0.01 and abs(np.corrcoef(U_, V_)[0, 1]) < 0.005,
      (round((U_ <= 2).mean(), 4), round(np.corrcoef(U_, V_)[0, 1], 4)))
# 図: e^Z は対数正規, |dz/dy|=1/y
fL = phi((sp.log(yp) - 0) / sp.Rational(1, 2)) / sp.Rational(1, 2) * sp.Abs(sp.diff(sp.log(yp), yp))
check(T, "図 Y=e^Z (σ=0.5) が lognorm 密度（scipy）と一致", np.allclose([float(fL.subs(yp, t)) for t in (0.5, 1, 2)], stats.lognorm(0.5).pdf([0.5, 1, 2])), True)
# 典型: -log U → Exp(1)
check(T, "-log U ~ Exp(1)（シミュ KS p>0.01）", stats.kstest(-np.log(rng.uniform(size=100000)), "expon").pvalue > 0.01, True)
# たたみ込み: V=X 補助変数で X+Y の密度
s = sp.Symbol("s", positive=True); w = sp.Symbol("w", positive=True)
conv = sp.integrate(sp.exp(-w) * sp.exp(-(s - w)), (w, 0, s))
check(T, "補助変数で周辺化（Exp+Exp の和の密度 s e^{-s}）", sp.simplify(conv - s * sp.exp(-s)) == 0, conv)

# ============ box-cox-transformation ============
T = "box-cox-transformation"
lam = sp.Symbol("lambda")


def bc(yv, l):
    return np.log(yv) if l == 0 else (yv**l - 1) / l


check(T, "λ→0 の極限が log y（sympy）", sp.limit((yp**lam - 1) / lam, lam, 0) == sp.log(yp), sp.limit((yp**lam - 1) / lam, lam, 0))
check(T, "d y^(λ)/dy = y^{λ-1}", sp.simplify(sp.diff((yp**lam - 1) / lam, yp) - yp**(lam - 1)) == 0, sp.diff((yp**lam - 1) / lam, yp))
check(T, "y=1 で値0・傾き1（λ=-2..2）", all(abs(bc(1.0, l)) < 1e-12 and abs((bc(1 + 1e-7, l) - bc(1 - 1e-7, l)) / 2e-7 - 1) < 1e-5 for l in np.arange(-2, 2.01, 0.1).round(1)), True)
check(T, "λ<0 でも増加関数", all(np.all(np.diff(bc(np.linspace(0.1, 5, 200), l)) > 0) for l in (-2, -1, -0.5)), True)
vals = [bc(4.0, 1), bc(4.0, 0.5), bc(4.0, 0), bc(4.0, -1)]
check(T, "例1 y=4: 3, 2, 1.3863, 0.75", close(vals[0], 3) and close(vals[1], 2) and close(vals[2], 1.3863) and close(vals[3], 0.75), [round(a, 5) for a in vals])
dat = np.array([1, 2, 3, 5, 8, 13, 21, 34], float)
n = len(dat); slog = np.log(dat).sum()
check(T, "例2 Σlog y≈14.6165", close(slog, 14.6165), round(slog, 5))


def s2(l):
    t = bc(dat, l)
    return ((t - t.mean())**2).mean()


def lp(l):
    return -n / 2 * np.log(s2(l)) + (l - 1) * slog


table = {-1: (0.0947, -19.804), -0.5: (0.2762, -16.778), 0: (1.2725, -15.580), 0.5: (9.7734, -16.427), 1: (115.3594, -18.992)}
for l, (sc, lc) in table.items():
    check(T, f"例2 表 λ={l}: σ̂²={sc}, ℓp={lc}", close(s2(l), sc) and close(lp(l), lc, 5e-4), (round(s2(l), 5), round(lp(l), 4)))
res = optimize.minimize_scalar(lambda l: -lp(l), bounds=(-2, 2), method="bounded", options={"xatol": 1e-10})
check(T, "例2 λ̂≈0.039（数値最大化）", close(res.x, 0.039, 5e-4), round(res.x, 5))
_, lam_sp = stats.boxcox(dat)
check(T, "例2 scipy.stats.boxcox の λ̂ と一致（1e-4）", abs(lam_sp - res.x) < 1e-4, round(lam_sp, 5))
# 完全対数尤度を μ,σ² で最大化した値と ℓp+定数 が一致することの確認
const = -n / 2 * (np.log(2 * np.pi) + 1)
full = lambda l: stats.norm(bc(dat, l).mean(), np.sqrt(s2(l))).logpdf(bc(dat, l)).sum() + (l - 1) * slog
check(T, "ℓ(λ,μ̂,σ̂²) = ℓp(λ) + 定数（-n/2(log2π+1)）", all(abs(full(l) - lp(l) - const) < 1e-9 for l in (-1, 0, 0.5, 1)), round(const, 4))
# ヤコビアン項込みの尤度は元データの密度として積分1（λ=0.5 で例示）
dens = lambda t: stats.norm(1, 0.7).pdf(bc(t, 0.5)) * t**(0.5 - 1)
tot = integrate.quad(dens, 0, np.inf)[0] + stats.norm(1, 0.7).cdf(-2)  # 変換の値域は (-2,∞) なので切断分を補正
check(T, "ヤコビアン y^{λ-1} を掛けると元の y の密度（値域内質量と一致）", abs(integrate.quad(dens, 0, np.inf)[0] - stats.norm(1, 0.7).sf(-2)) < 1e-7, round(tot, 6))
check(T, "χ²_1(0.05)=3.841", close(stats.chi2(1).ppf(0.95), 3.841, 5e-4), round(stats.chi2(1).ppf(0.95), 4))
check(T, "σ̂² 最小は λ=-1（表の範囲で）", min(table, key=lambda l: s2(l)) == -1, -1)

# ============ log-transformation ============
T = "log-transformation"
mu, sig = sp.Symbol("mu", real=True), sp.Symbol("sigma", positive=True)
fx = sp.exp(-(sp.log(yp) - mu)**2 / (2 * sig**2)) / (sp.sqrt(2 * sp.pi) * sig) * sp.diff(sp.log(yp), yp)
check(T, "対数正規密度＝変数変換の結果（scipy と一致）", np.allclose([float(fx.subs({mu: 1, sig: 0.5, yp: t})) for t in (0.5, 2, 5)], stats.lognorm(0.5, scale=math.e).pdf([0.5, 2, 5])), True)
m, s_ = sp.symbols("m s", positive=True)
check(T, "デルタ法 V[log X]≈(s/m)²", sp.simplify(sp.diff(sp.log(x), x).subs(x, m)**2 * s_**2 - (s_ / m)**2) == 0, (s_ / m)**2)
check(T, "log10 x = log x / log 10", close(math.log10(37) - math.log(37) / math.log(10), 0, 1e-12), True)
check(T, "例1 log2,8,32 ≈ 0.6931, 2.0794, 3.4657, 差 1.3863", close(math.log(2), 0.6931) and close(math.log(8), 2.0794) and close(math.log(32), 3.4657) and close(math.log(4), 1.3863), True)
check(T, "例1 幾何平均8, 算術平均14", close(stats.gmean([2, 8, 32]), 8, 1e-9) and close(np.mean([2, 8, 32]), 14), stats.gmean([2, 8, 32]))
LN = stats.lognorm(0.5, scale=math.e)
check(T, "例2 中央値 e≈2.7183, 平均 e^1.125≈3.0802", close(LN.median(), 2.7183) and close(LN.mean(), 3.0802), (round(LN.median(), 5), round(LN.mean(), 5)))
mu0, sg0 = 0.3, 0.8
L2 = stats.lognorm(sg0, scale=math.exp(mu0))
modeL = optimize.minimize_scalar(lambda t: -L2.pdf(t), bounds=(1e-3, 5), method="bounded", options={"xatol": 1e-10}).x
check(T, "対数正規 平均・分散・中央値・最頻値の公式（μ=0.3,σ=0.8）",
      close(L2.mean(), math.exp(mu0 + sg0**2 / 2), 1e-9) and close(L2.var(), math.exp(2 * mu0 + sg0**2) * (math.exp(sg0**2) - 1), 1e-9)
      and close(L2.median(), math.exp(mu0), 1e-9) and close(modeL, math.exp(mu0 - sg0**2), 1e-6), round(modeL, 5))
g1 = rng.normal(10, 2, N); g2 = rng.normal(100, 20, N)
sd1, sd2 = np.log(g1[g1 > 0]).std(), np.log(g2[g2 > 0]).std()
check(T, "例3 デルタ法 SD[log X]≈s/m: 2/10=20/100=0.2", close(2 / 10, 0.2) and close(20 / 100, 0.2), 0.2)
check(T, "例3 シミュ 2群の log後SD が一致（許容0.002）かつ≈0.2（1次近似誤差込み許容0.02）", abs(sd1 - sd2) < 0.002 and abs(sd1 - 0.2) < 0.02 and abs(sd2 - 0.2) < 0.02, (round(sd1, 4), round(sd2, 4)))
check(T, "b=0.05 → 約5.13%", close((math.exp(0.05) - 1) * 100, 5.13, 5e-3), round((math.exp(0.05) - 1) * 100, 4))
check(T, "e^{E[log X]} ≤ E[X]（イェンセン, 例2）", math.e < LN.mean(), True)
lsk = rng.beta(5, 1.5, 200000)
check(T, "左に歪んだデータは対数で歪みが強まる（Beta(5,1.5)）", stats.skew(np.log(lsk)) < stats.skew(lsk) < 0, (round(stats.skew(lsk), 3), round(stats.skew(np.log(lsk)), 3)))

# ============ logit-logistic-transformation ============
T = "logit-logistic-transformation"
p = sp.Symbol("p", positive=True)
logit = sp.log(p / (1 - p)); sigm = 1 / (1 + sp.exp(-x))
check(T, "σ(logit p)=p, logit(σ(x))=x", sp.simplify(sigm.subs(x, logit) - p) == 0 and sp.simplify(sp.expand_log(logit.subs(p, sigm), force=True) - x) == 0, True)
check(T, "σ=e^x/(1+e^x)", sp.simplify(sigm - sp.exp(x) / (1 + sp.exp(x))) == 0, True)
check(T, "σ'=σ(1-σ), logit'=1/(p(1-p))", sp.simplify(sp.diff(sigm, x) - sigm * (1 - sigm)) == 0 and sp.simplify(sp.diff(logit, p) - 1 / (p * (1 - p))) == 0, True)
check(T, "σ' の最大は x=0 で 1/4", sp.diff(sigm, x).subs(x, 0) == sp.Rational(1, 4) and sp.solve(sp.diff(sigm, x, 2), x) == [0], sp.Rational(1, 4))
check(T, "logit(1-p)=-logit(p)", sp.simplify(sp.expand_log(logit.subs(p, 1 - p) + logit, force=True)) == 0, True)
sg = lambda t: 1 / (1 + math.exp(-t))
lg = lambda q: math.log(q / (1 - q))
check(T, "例1 logit(0.8)≈1.3863, logit(0.2)≈-1.3863, odds(0.8)=4", close(lg(0.8), 1.3863) and close(lg(0.2), -1.3863) and close(0.8 / 0.2, 4), round(lg(0.8), 5))
check(T, "例1 σ(-1)≈0.2689, σ(2)≈0.8808", close(sg(-1), 0.2689) and close(sg(2), 0.8808), (round(sg(-1), 5), round(sg(2), 5)))
tab = {2: 0.1192, 4: 0.2689, 6: 0.5000, 8: 0.7311}
check(T, "例2 予測確率の表", all(close(sg(-3 + 0.5 * k), c) for k, c in tab.items()), [round(sg(-3 + 0.5 * k), 4) for k in tab])
check(T, "例2 x=-β0/β1=6, e^0.5≈1.6487", close(3 / 0.5, 6) and close(math.exp(0.5), 1.6487), round(math.exp(0.5), 5))
check(T, "例2 x=4 で傾き≈0.0983, x=6 で 0.125", close(0.5 * sg(-1) * (1 - sg(-1)), 0.0983) and close(0.5 * 0.25, 0.125), round(0.5 * sg(-1) * (1 - sg(-1)), 5))
check(T, "例2 オッズ比 = 隣接 x のオッズの比", close((sg(-0.5) / (1 - sg(-0.5))) / (sg(-1) / (1 - sg(-1))), math.exp(0.5), 1e-12), True)
nn, ph = 100, 0.2
pp, nn_s = sp.symbols("p n", positive=True)
dv = sp.simplify(sp.diff(logit, p).subs(p, pp)**2 * pp * (1 - pp) / nn_s)
check(T, "例3 デルタ法 V=1/(np(1-p))=0.0625, SE=0.25", sp.simplify(dv - 1 / (nn_s * pp * (1 - pp))) == 0 and close(1 / (nn * ph * (1 - ph)), 0.0625), dv)
lo, hi = lg(0.2) - 1.96 * 0.25, lg(0.2) + 1.96 * 0.25
check(T, "例3 ロジット区間 (-1.8763, -0.8963)", close(lo, -1.8763) and close(hi, -0.8963), (round(lo, 5), round(hi, 5)))
check(T, "例3 戻した区間 (0.1328, 0.2898)", close(sg(lo), 0.1328) and close(sg(hi), 0.2898), (round(sg(lo), 5), round(sg(hi), 5)))
# デルタ法の妥当性（シミュ）: logit(p̂) の分散
ys_ = rng.binomial(100, 0.2, 200000); ys_ = ys_[(ys_ > 0) & (ys_ < 100)]
vs = np.var(np.log(ys_ / (100 - ys_)))
check(T, "例3 シミュ V[logit p̂]≈0.0625（許容0.006）", abs(vs - 0.0625) < 0.006, round(vs, 5))
check(T, "ロジスティック分布 分散 π²s²/3, SD(s=1)≈1.8138", close(stats.logistic(scale=1.3).var(), math.pi**2 * 1.69 / 3, 1e-9) and close(stats.logistic.std(), 1.8138), round(stats.logistic.std(), 5))
check(T, "σ((x-μ)/s) = ロジスティック分布の CDF", np.allclose(stats.logistic(1, 2).cdf([-1, 0, 3]), [sg((t - 1) / 2) for t in (-1, 0, 3)]), True)
gx = np.linspace(-10, 10, 200001)
md = np.max(np.abs(1 / (1 + np.exp(-gx)) - stats.norm.cdf(gx / 1.7)))
check(T, "σ(x)≈Φ(x/1.7), 最大差≈0.0095（許容5e-4）", close(md, 0.0095, 5e-4), round(md, 5))
check(T, "p 数%でオッズ≈p（p=0.03, 相対誤差<4%）", abs(0.03 / 0.97 - 0.03) / 0.03 < 0.04, round(0.03 / 0.97, 5))
check(T, "経験ロジット y=0 でも有限", math.isfinite(math.log((0 + 0.5) / (20 - 0 + 0.5))), round(math.log(0.5 / 20.5), 4))

# ============ power-transformation ============
T = "power-transformation"
l_, mu_, beta, c = sp.symbols("lambda mu beta c", positive=True)
dvar = (sp.diff(x**l_, x).subs(x, mu_))**2 * (c * mu_**beta)**2
check(T, "デルタ法 V[X^λ]≈(λμ^{λ-1})²σ²", sp.simplify(sp.diff(x**l_, x).subs(x, mu_) - l_ * mu_**(l_ - 1)) == 0, sp.diff(x**l_, x).subs(x, mu_))
check(T, "λ=1-β で μ に依存しない", sp.simplify(sp.diff(dvar.subs(l_, 1 - beta), mu_)) == 0, sp.simplify(dvar.subs(l_, 1 - beta)))
check(T, "μ の指数 = 2(λ-1+β)", sp.simplify(sp.expand_power_base(sp.powsimp(dvar / (l_**2 * c**2), force=True), force=True) - mu_**(2 * (l_ - 1 + beta))) == 0, True)
check(T, "β=1/2→平方根, β=1→対数(λ=0), β=2→逆数(λ=-1)", [1 - b for b in (0.5, 1, 2)] == [0.5, 0, -1], [1 - b for b in (0.5, 1, 2)])
check(T, "ポアソン デルタ法 V[√X]≈1/4", sp.simplify((sp.Rational(1, 2) * mu_**sp.Rational(-1, 2))**2 * mu_) == sp.Rational(1, 4), sp.Rational(1, 4))


def var_sqrt_pois(m_):
    k = np.arange(0, int(m_ + 40 * math.sqrt(m_) + 50))
    pk = stats.poisson(m_).pmf(k)
    e1 = (np.sqrt(k) * pk).sum(); e2 = (k * pk).sum()
    return e2 - e1**2


for m_, cv in [(4, 0.3056), (16, 0.2565), (64, 0.2515)]:
    vv = var_sqrt_pois(m_)
    check(T, f"例 ポアソン μ={m_}: V[X]={m_}, V[√X]≈{cv}", close(stats.poisson(m_).var(), m_) and close(vv, cv), round(vv, 5))
xx = np.array([0.5, 1, 2, 5])
check(T, "λ<0 で大小逆転、-X^λ で順序保存", np.all(np.diff(xx**-1.0) < 0) and np.all(np.diff(-xx**-1.0) > 0), True)
check(T, "λ<1 で右歪みが弱まる（対数正規データ）", stats.skew(np.sqrt(L2.rvs(200000, random_state=1))) < stats.skew(L2.rvs(200000, random_state=1)), True)

# ============ probit-transformation ============
T = "probit-transformation"
check(T, "例1 probit(0.975)=1.9600, probit(0.8)≈0.8416, probit(0.5)=0", close(stats.norm.ppf(0.975), 1.9600) and close(stats.norm.ppf(0.8), 0.8416) and stats.norm.ppf(0.5) == 0, round(stats.norm.ppf(0.8), 5))
check(T, "probit(0.95)≈1.645", close(stats.norm.ppf(0.95), 1.645, 5e-4), round(stats.norm.ppf(0.95), 5))
check(T, "例1 比 1.3863/0.8416≈1.647", close(math.log(4) / stats.norm.ppf(0.8), 1.647, 5e-4), round(math.log(4) / stats.norm.ppf(0.8), 5))
check(T, "Φ^{-1}(1-p)=-Φ^{-1}(p)", np.allclose(stats.norm.ppf(0.9 - np.array([0.1, 0.3])) , -stats.norm.ppf(0.1 + np.array([0.1, 0.3]))), True)
pv = 0.3; h = 1e-6
num = (stats.norm.ppf(pv + h) - stats.norm.ppf(pv - h)) / (2 * h)
check(T, "dΦ^{-1}/dp = 1/φ(Φ^{-1}(p))（p=0.3 数値微分, 1e-5）", abs(num - 1 / stats.norm.pdf(stats.norm.ppf(pv))) < 1e-5, round(num, 5))
check(T, "例2 表 0.1587, 0.5000, 0.8413", all(close(stats.norm.cdf(-2 + 0.5 * k), c) for k, c in [(2, 0.1587), (4, 0.5), (6, 0.8413)]), [round(stats.norm.cdf(-2 + 0.5 * k), 4) for k in (2, 4, 6)])
check(T, "例2 ED50=4, 傾き φ(1)·0.5≈0.2420×0.5=0.1210", close(2 / 0.5, 4) and close(stats.norm.pdf(1), 0.2420) and close(stats.norm.pdf(1) * 0.5, 0.1210), round(stats.norm.pdf(1) * 0.5, 5))
xl = rng.uniform(0, 8, N); ystar = -2 + 0.5 * xl + rng.standard_normal(N)
sel = np.abs(xl - 6) < 0.05
check(T, "潜在変数モデル P(Y*>0|x=6)≈Φ(1)（シミュ, 許容0.01）", abs((ystar[sel] > 0).mean() - stats.norm.cdf(1)) < 0.01, round((ystar[sel] > 0).mean(), 4))
check(T, "分散1のロジスティック分布 scale=√3/π≈0.5513", close(math.sqrt(3) / math.pi, 0.5513) and close(stats.logistic(scale=0.5513).var(), 1, 5e-4), round(math.sqrt(3) / math.pi, 5))
check(T, "係数比 1.6〜1.8（π/√3≈1.814, 1.7, 例1の1.647）", 1.6 <= 1.647 <= 1.8 and 1.6 <= 1.7 <= 1.8 and abs(math.pi / math.sqrt(3) - 1.8138) < 5e-5, round(math.pi / math.sqrt(3), 4))

ok_n = sum(results); ng_n = len(results) - ok_n
print(f"\n合計: OK {ok_n} / NG {ng_n}")

# --- 追加用語（用語の見直し）: sum-of-random-variables ---
T = "sum-of-random-variables"
P1, P2 = stats.poisson(1.2), stats.poisson(1.8)
terms = [P1.pmf(j) * P2.pmf(2 - j) for j in range(3)]
check(T, "例1 畳み込み項 ≈0.0807, 0.1075, 0.0358, 和≈0.2240", close(terms[0], 0.0807) and close(terms[1], 0.1075) and close(terms[2], 0.0358) and close(sum(terms), 0.2240), [round(v, 5) for v in terms])
conv = np.convolve(P1.pmf(np.arange(60)), P2.pmf(np.arange(60)))[:40]
check(T, "例1 畳み込み全体 = Po(3)（0..39）, P(Z=2)=e^-3·9/2, P(Z≤1)=4e^-3≈0.1991", np.allclose(conv, stats.poisson(3).pmf(np.arange(40)), atol=1e-14) and close(math.exp(-3) * 4.5, 0.2240) and close(4 * math.exp(-3), 0.1991), round(4 * math.exp(-3), 5))
l1, l2 = sp.symbols("lambda_1 lambda_2", positive=True)
zi = 5
conv_s = sum(sp.exp(-l1) * l1**j / sp.factorial(j) * sp.exp(-l2) * l2 ** (zi - j) / sp.factorial(zi - j) for j in range(zi + 1))
check(T, "ポアソン畳み込み = e^{-(λ1+λ2)}(λ1+λ2)^z/z!（z=5, sympy）", sp.simplify(conv_s - sp.exp(-l1 - l2) * (l1 + l2) ** zi / sp.factorial(zi)) == 0, "")
pp, tt, mm, nn = sp.symbols("p t m n", positive=True)
check(T, "二項 MGF の積 (pe^t+q)^m(pe^t+q)^n=(pe^t+q)^{m+n}", sp.simplify((pp * sp.exp(tt) + 1 - pp) ** mm * (pp * sp.exp(tt) + 1 - pp) ** nn - (pp * sp.exp(tt) + 1 - pp) ** (mm + nn)) == 0, "")
zs = sp.symbols("z", positive=True)
f1 = sp.integrate(1, (x, 0, zs)); f2 = sp.integrate(1, (x, zs - 1, 1))
check(T, "例2 三角分布 f=z（0≤z≤1）, 2−z（1≤z≤2）, 積分1, P(Z>1.5)=0.125", f1 == zs and sp.simplify(f2 - (2 - zs)) == 0 and sp.integrate(zs, (zs, 0, 1)) + sp.integrate(2 - zs, (zs, 1, 2)) == 1
      and sp.integrate(2 - zs, (zs, sp.Rational(3, 2), 2)) == sp.Rational(1, 8), "")
Zu = rng.uniform(size=N) + rng.uniform(size=N)
check(T, "例2 シミュ P(Z>1.5)（許容0.002）", abs((Zu > 1.5).mean() - 0.125) < 0.002, round((Zu > 1.5).mean(), 4))
zD = (0 - 4) / math.sqrt(5)
check(T, "例3 D~N(4,5), z≈−1.789, P(D<0)≈0.0368", close(zD, -1.789, 5e-4) and close(stats.norm.cdf(zD), 0.0368), (round(zD, 4), round(stats.norm.cdf(zD), 5)))
Ds = rng.normal(10, 2, N) - rng.normal(6, 1, N)
check(T, "例3 シミュ 平均4・分散5（許容0.02）, P(D<0)（許容0.001）", abs(Ds.mean() - 4) < 0.02 and abs(Ds.var() - 5) < 0.02 and abs((Ds < 0).mean() - 0.0368) < 0.001, (round(Ds.mean(), 4), round(Ds.var(), 4)))
Es = rng.exponential(1, (200_000, 5)).sum(axis=1)
check(T, "図 指数(平均1)5個の和 = ガンマ(5,1)（KS p>0.001）", stats.kstest(Es, stats.gamma(5).cdf).pvalue > 0.001, round(stats.kstest(Es, stats.gamma(5).cdf).pvalue, 4))
Ga = rng.gamma(2, 3, N) + rng.gamma(3.5, 3, N)
check(T, "ガンマ再生性 Ga(2,3)+Ga(3.5,3)=Ga(5.5,3)（平均16.5・分散49.5, 許容0.05/0.3）", abs(Ga.mean() - 16.5) < 0.05 and abs(Ga.var() - 49.5) < 0.3, (round(Ga.mean(), 3), round(Ga.var(), 3)))
Cs_ = stats.cauchy(1, 2).rvs(N, random_state=1) + stats.cauchy(-3, 0.5).rvs(N, random_state=2)
check(T, "コーシー 位置・尺度が足し算 C(1,2)+C(−3,0.5)=C(−2,2.5)（中央値・四分位）", abs(np.median(Cs_) + 2) < 0.02 and abs((np.quantile(Cs_, 0.75) - np.quantile(Cs_, 0.25)) / 2 - 2.5) < 0.03, round(float(np.median(Cs_)), 4))
check(T, "V[2X]=4σ², V[X1+X2]=2σ²（σ=1, シミュ許容0.02）", abs(np.var(2 * rng.standard_normal(N)) - 4) < 0.03 and abs(np.var(rng.standard_normal(N) + rng.standard_normal(N)) - 2) < 0.02, "")

ok_n = sum(results); ng_n = len(results) - ok_n
print(f"\n合計（追加用語を含む）: OK {ok_n} / NG {ng_n}")
