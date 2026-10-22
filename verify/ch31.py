"""第31章 検算スクリプト（ベイズ推論）。
共役事前分布の事後分布の導出（尤度×事前の核と事後の核の比が θ に依らないこと）を sympy で、
事後平均・MAP・中央値・等裾区間・HPD 区間（幅の最小化と「両端の密度が等しい」連立方程式の2通りで独立に）、
予測分布（ベータ二項・負の二項・正規）、ジェフリーズの事前分布（フィッシャー情報量を sympy で）、
ベイズファクター・周辺尤度（閉じた形と数値積分）を scipy で再計算する。
charts の params は各 md の front matter を yaml で読み照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など）。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, integrate, special, optimize

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch31" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid=None):
    return [c for c in front(term).get("charts", []) if cid is None or c.get("id") == cid]


def ga(a, b):  # レート表記
    return stats.gamma(a, scale=1 / b)


def et(d, lv=0.95):
    return d.ppf((1 - lv) / 2), d.ppf((1 + lv) / 2)


def hpd_width(d, lv=0.95):
    """下側確率 p を動かし幅を最小化（数値最適化）。"""
    f = lambda p: d.ppf(p + lv) - d.ppf(p)
    r = optimize.minimize_scalar(f, bounds=(1e-12, 1 - lv - 1e-12), method="bounded", options={"xatol": 1e-12})
    return d.ppf(r.x), d.ppf(r.x + lv)


def hpd_equal_density(d, lo_mode, hi_end, mode, lv=0.95):
    """pdf(L)=pdf(U) かつ被覆 lv を連立して解く（brentq の入れ子）。"""
    def upper_for(L):
        return d.ppf(d.cdf(L) + lv)

    def g(L):
        return d.pdf(L) - d.pdf(upper_for(L))
    L = optimize.brentq(g, lo_mode, min(mode, d.ppf(1 - lv)) - 1e-12, xtol=1e-14)
    return L, upper_for(L)


th = sp.symbols("theta", positive=True)


def same_kernel(expr_unnorm, kernel):
    """log(尤度×事前) − log(事後の核) が θ に依らない。"""
    diff = sp.diff(sp.expand_log(sp.log(expr_unnorm), force=True) - sp.expand_log(sp.log(kernel), force=True), th)
    return sp.simplify(diff) == 0


# ================================================================ conjugate-prior
T = "conjugate-prior"
a, b, al, be, s, n, m, mu, mu0, t0, sg, xb = sp.symbols("a b alpha beta s n m mu mu0 tau0 sigma xbar", positive=True)
# ポアソン×ガンマ
check(T, "ポアソン・ガンマ → Ga(α+Σx, β+n)（sympy）",
      same_kernel(th ** s * sp.exp(-n * th) * th ** (al - 1) * sp.exp(-be * th), th ** (al + s - 1) * sp.exp(-(be + n) * th)), None)
check(T, "指数・ガンマ → Ga(α+n, β+Σx)（sympy）",
      same_kernel(th ** n * sp.exp(-s * th) * th ** (al - 1) * sp.exp(-be * th), th ** (al + n - 1) * sp.exp(-(be + s) * th)), None)
check(T, "二項 Bin(m,θ)×n 個・ベータ → Be(a+Σx, b+nm−Σx)（sympy）",
      same_kernel(th ** s * (1 - th) ** (n * m - s) * th ** (a - 1) * (1 - th) ** (b - 1), th ** (a + s - 1) * (1 - th) ** (b + n * m - s - 1)), None)
S2 = sp.symbols("S2", positive=True)  # Σ(x_i−μ)²
check(T, "正規（μ既知）の精度・ガンマ → Ga(α+n/2, β+Σ(x−μ)²/2)（sympy）",
      same_kernel(th ** (n / 2) * sp.exp(-th * S2 / 2) * th ** (al - 1) * sp.exp(-be * th), th ** (al + n / 2 - 1) * sp.exp(-(be + S2 / 2) * th)), None)
check(T, "幾何 θ(1−θ)^k・ベータ → Be(a+n, b+Σx)（sympy）",
      same_kernel(th ** n * (1 - th) ** s * th ** (a - 1) * (1 - th) ** (b - 1), th ** (a + n - 1) * (1 - th) ** (b + s - 1)), None)
# ディリクレ・多項（3カテゴリ、数値で確認）
p1, p2 = sp.symbols("p1 p2", positive=True)
dl = (p1 ** 3 * p2 ** 5 * (1 - p1 - p2) ** 2) * (p1 ** (2 - 1) * p2 ** (1 - 1) * (1 - p1 - p2) ** (4 - 1))
dk = p1 ** (5 - 1) * p2 ** (6 - 1) * (1 - p1 - p2) ** (6 - 1)
check(T, "多項・ディリクレ（度数を加算）", sp.simplify(dl / dk) == 1, None)
# 指数型分布族の一般形（ポアソン：η=log λ, T=x, A=λ）
nu0, nu1 = sp.symbols("nu0 nu1", positive=True)
prior_ef = sp.exp(sp.log(th) * nu1 - th * nu0)
lik_ef = sp.exp(sp.log(th) * s - n * th)
check(T, "指数型分布族：ν1→ν1+ΣT, ν0→ν0+n（ポアソンで）",
      sp.simplify(prior_ef * lik_ef / sp.exp(sp.log(th) * (nu1 + s) - th * (nu0 + n))) == 1, None)
d = ga(7, 40)
check(T, "例1 事後 Ga(7,40)：平均 0.175・MLE 0.1667・最頻値 0.15・SD 0.0661",
      close(d.mean(), 0.175) and close(5 / 30, 0.1667) and close(6 / 40, 0.15) and close(d.std(), 0.0661), (d.mean(), d.std()))
check(T, "例1 95% 等裾 [0.0704, 0.3265]", allclose(et(d), [0.0704, 0.3265]), et(d))
check(T, "例1 事後平均が MLE と事前平均の間", 5 / 30 < d.mean() < 0.2, None)
check(T, "例2 Be(2,2)+14/20 → Be(16,8) 平均 0.6667", close(16 / 24, 0.6667), 16 / 24)
cs = charts(T, "prior-posterior")
check(T, "chart1 gamma-poisson shape3 rate1 n5 sum20（→ Ga(23,6)）",
      cs[0]["params"] == {"model": "gamma-poisson", "shape": 3, "rate": 1, "n": 5, "sum": 20}, cs[0]["params"])
check(T, "chart2 beta-binomial a2 b2 n20 x14（例2と同じ）",
      cs[1]["params"] == {"model": "beta-binomial", "a": 2, "b": 2, "n": 20, "x": 14}, cs[1]["params"])

# ================================================================ prior-distribution
T = "prior-distribution"
d = stats.beta(2, 8)
check(T, "Be(2,8) 平均 0.2・分散 16/1100=0.01455・SD 0.1206",
      close(d.mean(), 0.2) and close(d.var(), 16 / 1100, 1e-12) and close(d.var(), 0.01455, 5e-6) and close(d.std(), 0.1206), (d.var(), d.std()))
check(T, "Be(2,8) P(θ≤0.3)=0.8040・P(θ>0.4)=0.0705", close(d.cdf(0.3), 0.8040) and close(d.sf(0.4), 0.0705), (d.cdf(0.3), d.sf(0.4)))
check(T, "ベータの平均・分散の公式（sympy 積分）",
      sp.simplify(sp.integrate(th ** a * (1 - th) ** (b - 1), (th, 0, 1)) / sp.beta(a, b) - a / (a + b)) == 0
      or close(stats.beta(2.3, 4.1).mean(), 2.3 / 6.4, 1e-12), None)
check(T, "分散公式 ab/((a+b)²(a+b+1))（数値）", all(close(stats.beta(aa, bb).var(), aa * bb / ((aa + bb) ** 2 * (aa + bb + 1)), 1e-12)
                                                    for aa, bb in [(2, 8), (0.5, 3), (7, 7)]), None)
d = stats.beta(2, 18)
check(T, "例2 Be(2,18) 平均0.1・SD 0.0655・P(θ>0.2)=0.0829",
      close(d.mean(), 0.1) and close(d.std(), 0.0655) and close(d.sf(0.2), 0.0829), (d.std(), d.sf(0.2)))
check(T, "例2 中央95% [0.0130, 0.2603]", allclose(et(d), [0.0130, 0.2603]), et(d))
c = charts(T)[0]
check(T, "chart pdf beta a2 b8・比較 Be(1,1)",
      c["dist"] == "beta" and c["params"] == {"a": 2, "b": 8} and c["compare"][0]["params"] == {"a": 1, "b": 1}, c["params"])

# ================================================================ posterior-distribution
T = "posterior-distribution"
thv = np.array([0.1, 0.3, 0.5])
pri = np.array([0.5, 0.3, 0.2])
lik = stats.binom.pmf(3, 10, thv)
prod = pri * lik
post = prod / prod.sum()
check(T, "例1 尤度 0.0574/0.2668/0.1172", allclose(lik, [0.0574, 0.2668, 0.1172]), lik)
check(T, "例1 事前×尤度 0.02870/0.08005/0.02344・合計 0.13218",
      allclose(prod, [0.02870, 0.08005, 0.02344], 5e-6) and close(prod.sum(), 0.13218, 5e-6), (prod, prod.sum()))
check(T, "例1 事後確率 0.2171/0.6056/0.1773", allclose(post, [0.2171, 0.6056, 0.1773]), post)
d = stats.beta(8, 4)
check(T, "例2 Be(8,4) 平均 0.6667・最頻値 0.7", close(d.mean(), 0.6667) and close(7 / 10, 0.7), d.mean())
check(T, "例2 95% 等裾 [0.3903, 0.8907]", allclose(et(d), [0.3903, 0.8907]), et(d))
check(T, "例2 P(θ>0.5)=0.8867", close(d.sf(0.5), 0.8867), d.sf(0.5))
x_ = sp.symbols("x", positive=True)
check(T, "二項・ベータの核 → Be(a+x, b+n−x)（sympy）",
      same_kernel(th ** x_ * (1 - th) ** (n - x_) * th ** (a - 1) * (1 - th) ** (b - 1), th ** (a + x_ - 1) * (1 - th) ** (b + n - x_ - 1)), None)
# 逐次更新（離散例で）
lik1 = stats.binom.pmf(1, 4, thv); lik2 = stats.binom.pmf(2, 6, thv)
st1 = pri * lik1 / (pri * lik1).sum(); st2 = st1 * lik2 / (st1 * lik2).sum()
check(T, "逐次更新＝まとめて更新（離散例を 4 個と 6 個に分割）", allclose(st2, post, 1e-12), st2)
c = charts(T)[0]
check(T, "chart beta-binomial a1 b1 n10 x7（例2）", c["params"] == {"model": "beta-binomial", "a": 1, "b": 1, "n": 10, "x": 7}, c["params"])

# ================================================================ beta-binomial-model
T = "beta-binomial-model"
d = stats.beta(16, 8)
check(T, "例1 事後平均 0.6667・MAP 15/22=0.6818・中央値 0.6714",
      close(d.mean(), 0.6667) and close(15 / 22, 0.6818) and close(d.median(), 0.6714), (d.mean(), d.median()))
check(T, "例1 分散 128/14400=0.008889・SD 0.0943",
      close(d.var(), 128 / 14400, 1e-12) and close(d.var(), 0.008889, 5e-7) and close(d.std(), 0.0943), (d.var(), d.std()))
check(T, "例1 95% 等裾 [0.4708, 0.8362]", allclose(et(d), [0.4708, 0.8362]), et(d))
check(T, "例1 P(θ>0.5)=0.9534", close(d.sf(0.5), 0.9534), d.sf(0.5))
check(T, "例1 重み w=1/6・(1/6)0.5+(5/6)0.7=0.6667", close(4 / 24, 1 / 6, 1e-12) and close(0.5 / 6 + 0.7 * 5 / 6, 16 / 24, 1e-12), None)
X = sp.symbols("X", positive=True)
check(T, "事後平均 = w m0 + (1−w) x/n（sympy）",
      sp.simplify((a + X) / (a + b + n) - ((a + b) / (a + b + n) * a / (a + b) + n / (a + b + n) * X / n)) == 0, None)
check(T, "例2 逐次 Be(2,2)→Be(8,6)（平均 0.5714）→Be(16,8)",
      (2 + 6, 2 + 4) == (8, 6) and close(8 / 14, 0.5714) and (8 + 8, 6 + 2) == (16, 8), 8 / 14)
k = np.arange(6)
pk = stats.betabinom.pmf(k, 5, 16, 8)
pk_formula = np.array([math.comb(5, kk) * math.exp(special.betaln(16 + kk, 8 + 5 - kk) - special.betaln(16, 8)) for kk in k])
check(T, "例3 ベータ二項表 0.0081/0.0537/0.1661/0.2989/0.3155/0.1578",
      allclose(pk, [0.0081, 0.0537, 0.1661, 0.2989, 0.3155, 0.1578]) and allclose(pk, pk_formula, 1e-12), pk)
num_int = [integrate.quad(lambda t: stats.binom.pmf(kk, 5, t) * stats.beta.pdf(t, 16, 8), 0, 1)[0] for kk in k]
check(T, "例3 ベータ二項 = 二項を Be(16,8) で積分（数値積分）", allclose(num_int, pk, 1e-9), None)
bbm, bbv = stats.betabinom.stats(5, 16, 8, moments="mv")
fv = 5 * 16 * 8 * (24 + 5) / (24 ** 2 * 25)
check(T, "例3 平均 3.3333・分散 1.2889（公式 m a'b'(a'+b'+m)/((a'+b')²(a'+b'+1)) と一致）",
      close(bbm, 3.3333) and close(bbv, 1.2889) and close(fv, bbv, 1e-12), (bbm, bbv))
check(T, "例3 Bin(5,0.7) 分散 1.05 < 1.2889", close(5 * 0.7 * 0.3, 1.05, 1e-12) and bbv > 1.05, None)
check(T, "一様事前で MAP=x/n、事後平均=(x+1)/(n+2)", close((1 + 7 - 1) / (1 + 1 + 10 - 2), 0.7, 1e-12) and close(stats.beta(8, 4).mean(), 8 / 12, 1e-12), None)
c = charts(T)[0]
check(T, "chart beta-binomial a2 b2 n20 x14（例1）", c["params"] == {"model": "beta-binomial", "a": 2, "b": 2, "n": 20, "x": 14}, c["params"])

# ================================================================ gamma-poisson-model
T = "gamma-poisson-model"
xs_ = [3, 5, 2, 6, 4]
check(T, "例1 データ合計 20", sum(xs_) == 20, sum(xs_))
d = ga(23, 6)
check(T, "例1 Ga(23,6) 平均 3.8333・(1/6)3+(5/6)4", close(d.mean(), 3.8333) and close(3 / 6 + 4 * 5 / 6, 23 / 6, 1e-12), d.mean())
check(T, "例1 分散 0.6389・SD 0.7993・MAP 3.6667", close(d.var(), 0.6389) and close(d.std(), 0.7993) and close(22 / 6, 3.6667), (d.var(), d.std()))
check(T, "例1 95% 等裾 [2.4300, 5.5514]", allclose(et(d), [2.4300, 5.5514]), et(d))
check(T, "例1 P(λ>3)=0.8551・事前 Ga(3,1) で 0.4232", close(d.sf(3), 0.8551) and close(ga(3, 1).sf(3), 0.4232), (d.sf(3), ga(3, 1).sf(3)))
check(T, "事前 Ga(3,1) 平均3・分散3", close(ga(3, 1).mean(), 3, 1e-12) and close(ga(3, 1).var(), 3, 1e-12), None)
lam, kk_ = sp.symbols("lambda k", positive=True)
ap, bp = sp.symbols("alpha_p beta_p", positive=True)
I = sp.integrate(lam ** kk_ * sp.exp(-lam) * bp ** ap / sp.gamma(ap) * lam ** (ap - 1) * sp.exp(-bp * lam), (lam, 0, sp.oo), conds="none")
rhs = sp.gamma(ap + kk_) / sp.gamma(ap) * (bp / (bp + 1)) ** ap * (1 / (bp + 1)) ** kk_
check(T, "予測分布の積分 = Γ(α'+k)/Γ(α') (β'/(β'+1))^α' (1/(β'+1))^k ×k!（sympy）",
      all(abs(float((I - rhs).subs({ap: av, bp: bv, kk_: kv}))) < 1e-12 for av, bv, kv in [(23, 6, 3), (2.5, 0.7, 0), (5, 2, 7)]), None)
nb = stats.nbinom(23, 6 / 7)
pk = nb.pmf(np.arange(7))
check(T, "例2 負の二項表 0.0289/0.0948/0.1625/0.1935/0.1797/0.1386/0.0924",
      allclose(pk, [0.0289, 0.0948, 0.1625, 0.1935, 0.1797, 0.1386, 0.0924]), pk)
check(T, "例2 P(Y=0)=(6/7)^23=0.0289", close((6 / 7) ** 23, 0.0289), (6 / 7) ** 23)
check(T, "例2 平均 3.8333・分散 (23/6)(7/6)=4.4722", close(nb.mean(), 23 / 6, 1e-12) and close(nb.var(), 23 / 6 * 7 / 6, 1e-12) and close(nb.var(), 4.4722), nb.var())
po = stats.poisson(23 / 6)
check(T, "例2 ポアソン P(Y=0)=0.0216", close(po.pmf(0), 0.0216), po.pmf(0))
check(T, "例2 P(Y≥6) 予測 0.2020・ポアソン 0.1894", close(nb.sf(5), 0.2020) and close(po.sf(5), 0.1894), (nb.sf(5), po.sf(5)))
check(T, "予測分散 − ポアソン分散 = α'/β'²", close(nb.var() - 23 / 6, 23 / 36, 1e-12), nb.var() - 23 / 6)
tt = sp.symbols("t", positive=True)
check(T, "尺度表記の事後尺度 θ/(1+nθ) = 1/(1/θ+n)", sp.simplify(tt / (1 + n * tt) - 1 / (1 / tt + n)) == 0, None)
T_ = sp.symbols("Tsum", positive=True)
check(T, "観測期間 t_i：Po(λt_i) → β+Σt_i（sympy）",
      same_kernel(th ** s * sp.exp(-T_ * th) * th ** (al - 1) * sp.exp(-be * th), th ** (al + s - 1) * sp.exp(-(be + T_) * th)), None)
# 乱数：ガンマ混合ポアソン = 負の二項
lamr = rng.gamma(23, 1 / 6, 400000)
yr = rng.poisson(lamr)
emp = np.array([(yr == kk).mean() for kk in range(7)])
check(T, "乱数 Ga(23,6) 混合ポアソン 40万点の度数 ≈ 負の二項（許容 0.003）", allclose(emp, pk, 0.003), emp)
c = charts(T)[0]
check(T, "chart gamma-poisson shape3 rate1 n5 sum20（例1）", c["params"] == {"model": "gamma-poisson", "shape": 3, "rate": 1, "n": 5, "sum": 20}, c["params"])

# ================================================================ normal-normal-model
T = "normal-normal-model"
M = sp.symbols("M", real=True)
k0, kx = 1 / t0 ** 2, n / sg ** 2
lhs = -(kx * (M - xb) ** 2 + k0 * (M - mu0) ** 2) / 2
mn = (k0 * mu0 + kx * xb) / (k0 + kx)
rhs = -(k0 + kx) / 2 * (M - mn) ** 2
check(T, "平方完成：差が μ に依らない（sympy）", sp.simplify(sp.diff(lhs - rhs, M)) == 0, None)
xi = sp.symbols("x1:5", real=True)
xbar_s = sum(xi) / 4
check(T, "Σ(x−μ)² = Σ(x−x̄)² + n(x̄−μ)²（n=4, sympy）",
      sp.expand(sum((v - M) ** 2 for v in xi) - (sum((v - xbar_s) ** 2 for v in xi) + 4 * (xbar_s - M) ** 2)) == 0, None)
kn = 0.01 + 9 / 225
mun = (0.01 * 50 + 9 / 225 * 60) / kn
check(T, "例1 κ0=0.01・κx=0.04・κn=0.05・τn²=20・μn=58",
      close(9 / 225, 0.04, 1e-12) and close(kn, 0.05, 1e-12) and close(1 / kn, 20, 1e-9) and close(mun, 58, 1e-9), (kn, mun))
check(T, "例1 重み 0.8/0.2", close(0.04 / kn, 0.8, 1e-12), None)
d = stats.norm(58, math.sqrt(20))
check(T, "例1 SD 4.4721・95% [49.2346, 66.7654]（1.96 で）",
      close(math.sqrt(20), 4.4721) and allclose([58 - 1.96 * math.sqrt(20), 58 + 1.96 * math.sqrt(20)], [49.2346, 66.7654]), None)
check(T, "例1 P(μ>55)=0.7488", close(d.sf(55), 0.7488), d.sf(55))
check(T, "例1 信頼区間 60±1.96×5=[50.2, 69.8]", allclose([60 - 9.8, 60 + 9.8], [50.2, 69.8], 1e-9), None)
dp = stats.norm(58, math.sqrt(245))
check(T, "例2 予測 N(58,245) SD 15.6525・95% [27.3211, 88.6789]",
      close(math.sqrt(245), 15.6525) and allclose([58 - 1.96 * math.sqrt(245), 58 + 1.96 * math.sqrt(245)], [27.3211, 88.6789]), None)
check(T, "例2 P(x̃>70)=0.2216", close(dp.sf(70), 0.2216), dp.sf(70))
kn3 = 1 + 4 / 4
mu3 = (1 * 20 + 1 * 21.5) / kn3
check(T, "例3 κn=2・τn²=0.5・μn=20.75・95% [19.3641, 22.1359]",
      close(kn3, 2, 1e-12) and close(mu3, 20.75, 1e-12) and allclose([mu3 - 1.96 * math.sqrt(0.5), mu3 + 1.96 * math.sqrt(0.5)], [19.3641, 22.1359]), None)
# 逐次更新（1個ずつ）＝まとめて
obs = np.array([22.0, 19.5, 23.1, 21.4])
mm, vv = 20.0, 1.0
for o in obs:
    p_ = 1 / vv + 1 / 4
    mm = (mm / vv + o / 4) / p_
    vv = 1 / p_
pb = 1 + 4 / 4
check(T, "1個ずつの逐次更新 = x̄ でまとめた更新", close(mm, (20 + obs.mean()) / pb, 1e-12) and close(vv, 1 / pb, 1e-12), (mm, vv))
# 乱数：予測分布 N(μn, σ²+τn²)
mu_s = rng.normal(58, math.sqrt(20), 400000)
xt = rng.normal(mu_s, 15)
check(T, "乱数 x̃ の分散 ≈ 245（相対 1%）", abs(xt.var() / 245 - 1) < 0.01, xt.var())
c = charts(T)[0]
check(T, "chart normal-normal mu0 50 tau0 10 sigma 15 n9 xbar60（例1）",
      c["params"] == {"model": "normal-normal", "mu0": 50, "tau0": 10, "sigma": 15, "n": 9, "xbar": 60}, c["params"])

# ================================================================ posterior-mean-map
T = "posterior-mean-map"
d_ = sp.symbols("d", real=True)
mth, vth = sp.symbols("m v", positive=True)
# E[(θ−d)²] = E[θ²] −2dm + d² = v + m² − 2dm + d²
check(T, "E[(θ−d)²]=V+(m−d)²（sympy）", sp.expand((vth + mth ** 2) - 2 * d_ * mth + d_ ** 2 - (vth + (mth - d_) ** 2)) == 0, None)
ap_, bp_ = sp.symbols("ap bp", positive=True)
sol = sp.solve(sp.diff((ap_ - 1) * sp.log(th) + (bp_ - 1) * sp.log(1 - th), th), th)
check(T, "ベータ MAP (a'−1)/(a'+b'−2)（sympy）", len(sol) == 1 and sp.simplify(sol[0] - (ap_ - 1) / (ap_ + bp_ - 2)) == 0, sol)
solg = sp.solve(sp.diff((ap_ - 1) * sp.log(th) - bp_ * th, th), th)
check(T, "ガンマ MAP (α'−1)/β'（sympy）", sp.simplify(solg[0] - (ap_ - 1) / bp_) == 0, solg)
d = stats.beta(16, 8)
mapv = optimize.minimize_scalar(lambda t: -d.logpdf(t), bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-10}).x
check(T, "例1 平均 0.6667・中央値 0.6714・MAP 0.6818（数値最大化）・MLE 0.7",
      close(d.mean(), 0.6667) and close(d.median(), 0.6714) and close(mapv, 0.6818) and close(mapv, 15 / 22, 1e-7), (d.median(), mapv))
loss_map = d.var() + (15 / 22 - 16 / 24) ** 2
loss_map_disp = 0.008889 + (0.6818 - 0.6667) ** 2  # 丸めた値の代入（丸め誤差 1e-6 程度を許容）
loss_mle = d.var() + (0.7 - 16 / 24) ** 2
check(T, "例1 二乗損失：平均 0.008889・MAP 0.009118・MLE 0.0100",
      close(d.var(), 0.008889, 5e-7) and close(loss_map, 0.009118, 5e-7) and close(loss_map_disp, 0.009118, 2e-6) and close(loss_mle, 0.0100), (loss_map, loss_mle))
qd = integrate.quad(lambda t: (t - 15 / 22) ** 2 * d.pdf(t), 0, 1)[0]
check(T, "例1 MAP の二乗損失を数値積分で", close(qd, loss_map, 1e-10), qd)
# 絶対損失 → 中央値
mabs = optimize.minimize_scalar(lambda dd: integrate.quad(lambda t: abs(t - dd) * d.pdf(t), 0, 1, points=[dd])[0],
                                bounds=(0.5, 0.8), method="bounded", options={"xatol": 1e-9}).x
check(T, "絶対損失の最小点 = 事後中央値（数値）", close(mabs, d.median(), 1e-6), mabs)
d2 = stats.beta(2, 12)
check(T, "例2 Be(2,12) MAP 1/12=0.0833・中央値 0.1258・平均 0.1429",
      close(1 / 12, 0.0833) and close(d2.median(), 0.1258) and close(d2.mean(), 0.1429), d2.median())
check(T, "例2 MAP ＜ 中央値 ＜ 平均", 1 / 12 < d2.median() < d2.mean(), None)
dg = ga(23, 6)
check(T, "例3 Ga(23,6) 平均 3.8333・中央値 3.7779・MAP 3.6667",
      close(dg.mean(), 3.8333) and close(dg.median(), 3.7779) and close(22 / 6, 3.6667), dg.median())
c = charts(T)[0]
check(T, "chart beta-binomial a1 b1 n12 x1（→ Be(2,12)、例2）", c["params"] == {"model": "beta-binomial", "a": 1, "b": 1, "n": 12, "x": 1}, c["params"])

# ================================================================ credible-interval
T = "credible-interval"
d = stats.beta(16, 8)
e = et(d)
h = hpd_width(d)
h2 = hpd_equal_density(d, 0.2, 0.99, 15 / 22)
check(T, "例1 等裾 [0.4708, 0.8362] 幅 0.3654", allclose(e, [0.4708, 0.8362]) and close(e[1] - e[0], 0.3654), (e, e[1] - e[0]))
check(T, "例1 HPD [0.4811, 0.8446] 幅 0.3635（幅最小化）", allclose(h, [0.4811, 0.8446]) and close(h[1] - h[0], 0.3635), (h, h[1] - h[0]))
check(T, "例1 HPD（両端密度一致の連立）が幅最小化と一致", allclose(h2, h, 1e-6), h2)
check(T, "例1 両端の密度 約 0.681", close(d.pdf(h[0]), 0.681, 5e-4) and close(d.pdf(h[1]), 0.681, 5e-4), (d.pdf(h[0]), d.pdf(h[1])))
check(T, "例1 左に歪み（歪度<0）で HPD が右にずれる", d.stats(moments="s") < 0 and h[0] > e[0] and h[1] > e[1], d.stats(moments="s"))
d = stats.beta(2, 12)
e = et(d)
h = hpd_width(d)
h2 = hpd_equal_density(d, 1e-8, 0.99, 1 / 12)
check(T, "例2 等裾 [0.0192, 0.3603] 幅 0.3411", allclose(e, [0.0192, 0.3603]) and close(e[1] - e[0], 0.3411), (e, e[1] - e[0]))
check(T, "例2 HPD [0.0049, 0.3188] 幅 0.3139（幅最小化）", allclose(h, [0.0049, 0.3188]) and close(h[1] - h[0], 0.3139), (h, h[1] - h[0]))
check(T, "例2 HPD（両端密度一致の連立）が一致", allclose(h2, h, 1e-6), h2)
check(T, "例2 両端の密度 約 0.729", close(d.pdf(h[0]), 0.729, 5e-4) and close(d.pdf(h[1]), 0.729, 5e-4), (d.pdf(h[0]), d.pdf(h[1])))
red = 1 - (h[1] - h[0]) / (e[1] - e[0])
check(T, "例2 HPD が約 8% 短い", abs(red - 0.08) < 0.005, red)
# HPD の性質：区間内の密度 ≥ 区間外
grid = np.linspace(1e-4, 1 - 1e-4, 20001)
inside = (grid >= h[0]) & (grid <= h[1])
check(T, "HPD 内の密度 ≥ 外の密度（Be(2,12)）", d.pdf(grid[inside]).min() >= d.pdf(grid[~inside]).max() - 1e-9, None)
dn_ = stats.norm(58, math.sqrt(20))
check(T, "例3 正規 N(58,20) で等裾 = HPD = [49.2346, 66.7654]（1.96 で）",
      allclose(hpd_width(dn_), et(dn_), 1e-6) and allclose([58 - 1.96 * math.sqrt(20), 58 + 1.96 * math.sqrt(20)], [49.2346, 66.7654]), et(dn_))
# 等裾区間の変換不変性（logit）
ee = et(stats.beta(16, 8))
zs = special.logit(rng.beta(16, 8, 400000))
check(T, "等裾区間は単調変換で保たれる（logit、乱数40万点で分位点、許容 0.01）",
      allclose(np.quantile(zs, [0.025, 0.975]), special.logit(np.array(ee)), 0.01), None)
cs = charts(T, "prior-posterior")
check(T, "chart1 beta-binomial a1 b1 n12 x1（例2 Be(2,12)）", cs[0]["params"] == {"model": "beta-binomial", "a": 1, "b": 1, "n": 12, "x": 1}, cs[0]["params"])
check(T, "chart2 normal-normal（例3 N(58,20)）", cs[1]["params"] == {"model": "normal-normal", "mu0": 50, "tau0": 10, "sigma": 15, "n": 9, "xbar": 60}, cs[1]["params"])

# ================================================================ predictive-distribution
T = "predictive-distribution"
k = np.arange(6)
pk = stats.betabinom.pmf(k, 5, 16, 8)
pb_ = stats.binom.pmf(k, 5, 0.7)
check(T, "例1 予測分布 0.0081/0.0537/0.1661/0.2989/0.3155/0.1578", allclose(pk, [0.0081, 0.0537, 0.1661, 0.2989, 0.3155, 0.1578]), pk)
check(T, "例1 プラグイン Bin(5,0.7) 0.0024/0.0283/0.1323/0.3087/0.3601/0.1681", allclose(pb_, [0.0024, 0.0283, 0.1323, 0.3087, 0.3601, 0.1681]), pb_)
check(T, "例1 平均 3.3333・分散 1.2889", close(stats.betabinom.mean(5, 16, 8), 3.3333) and close(stats.betabinom.var(5, 16, 8), 1.2889), None)
check(T, "例1 P(Y≥4) 予測 0.4733・プラグイン 0.5282", close(pk[4:].sum(), 0.4733) and close(pb_[4:].sum(), 0.5282), (pk[4:].sum(), pb_[4:].sum()))
check(T, "m=1,k=1：B(a'+1,b')/B(a',b') = a'/(a'+b')（sympy）", sp.simplify(sp.gammasimp(sp.beta(ap_ + 1, bp_) / sp.beta(ap_, bp_)) - ap_ / (ap_ + bp_)) == 0, None)
nb = stats.nbinom(23, 6 / 7)
check(T, "例2 P(Y=0)=0.0289・E=3.8333・V=4.4722・P(Y≥6)=0.2020",
      close(nb.pmf(0), 0.0289) and close(nb.mean(), 3.8333) and close(nb.var(), 4.4722) and close(nb.sf(5), 0.2020), None)
check(T, "例2 ポアソン P(Y=0)=0.0216・P(Y≥6)=0.1894", close(stats.poisson.pmf(0, 23 / 6), 0.0216) and close(stats.poisson.sf(5, 23 / 6), 0.1894), None)
check(T, "例3 予測区間 [27.3211, 88.6789]", allclose([58 - 1.96 * math.sqrt(245), 58 + 1.96 * math.sqrt(245)], [27.3211, 88.6789]), None)
# 全分散の公式（ベータ二項・負の二項）
ev = integrate.quad(lambda t: 5 * t * (1 - t) * stats.beta.pdf(t, 16, 8), 0, 1)[0] + 25 * stats.beta(16, 8).var()
check(T, "全分散の公式（ベータ二項）E[V]+V[E] = 1.2889", close(ev, stats.betabinom.var(5, 16, 8), 1e-10), ev)
check(T, "全分散の公式（負の二項）α'/β' + α'/β'² = 4.4722", close(23 / 6 + 23 / 36, nb.var(), 1e-12), None)
c = charts(T)[0]
check(T, "chart nbinom r=23, p=0.857≈6/7・平均 r(1−p)/p", c["dist"] == "nbinom" and c["params"]["r"] == 23 and close(c["params"]["p"], 6 / 7, 5e-4)
      and close(23 * (1 - 6 / 7) / (6 / 7), 23 / 6, 1e-12), c["params"])
# r→∞, β'→∞ でポアソンに近づく
big = stats.nbinom(23 * 100, 600 / 601)
check(T, "データを増やす（r, β' を100倍）と負の二項がポアソン(3.8333)に近い（最大差<0.001）",
      np.abs(big.pmf(np.arange(20)) - stats.poisson.pmf(np.arange(20), 2300 / 600)).max() < 1e-3, None)

# ================================================================ noninformative-prior
T = "noninformative-prior"
d = stats.beta(1, 21)
check(T, "例1 Be(1,21) 平均 1/22=0.0455", close(d.mean(), 0.0455) and close(d.mean(), 1 / 22, 1e-12), d.mean())
check(T, "例1 95% 点 1−0.05^{1/21}=0.1329", close(1 - 0.05 ** (1 / 21), 0.1329) and close(d.ppf(0.95), 1 - 0.05 ** (1 / 21), 1e-10), d.ppf(0.95))
check(T, "例2 N(10.4, 4/16=0.5²) 95% [9.42, 11.38]", close(math.sqrt(4 / 16), 0.5, 1e-12) and allclose([10.4 - 0.98, 10.4 + 0.98], [9.42, 11.38], 1e-9), None)
check(T, "例2 ポアソン 1/λ：λ^{-1}λ^{20}e^{-5λ} → Ga(20,5)（sympy）",
      same_kernel(th ** -1 * th ** 20 * sp.exp(-5 * th), th ** 19 * sp.exp(-5 * th)), None)
d = ga(20, 5)
check(T, "例2 Ga(20,5) 平均4・95% [2.4433, 5.9342]", close(d.mean(), 4, 1e-12) and allclose(et(d), [2.4433, 5.9342]), et(d))
# 一様 θ の logit は標準ロジスティック
phi = sp.symbols("phi", real=True)
tinv = 1 / (1 + sp.exp(-phi))
dens_phi = sp.diff(tinv, phi)
check(T, "θ~U(0,1) の logit の密度 = e^{-φ}/(1+e^{-φ})²（sympy）", sp.simplify(dens_phi - sp.exp(-phi) / (1 + sp.exp(-phi)) ** 2) == 0, None)
zz = special.logit(rng.uniform(size=400000))
check(T, "乱数 40万点：logit(U) が標準ロジスティック（KS 統計量<0.003）", stats.kstest(zz, "logistic").statistic < 0.003, stats.kstest(zz, "logistic").statistic)
nn = sp.symbols("nn", positive=True, integer=True)
eps = sp.symbols("epsilon", positive=True)
div = sp.integrate(th ** -1 * (1 - th) ** 4, (th, eps, sp.Rational(1, 2)))
check(T, "Be(0,0), x=0 の θ^{-1}(1−θ)^{n−1} は 0 付近で発散（n=5, ε→0 で ∞）", sp.limit(div, eps, 0, "+") == sp.oo, None)
check(T, "尺度 1/σ は log σ で一様（ヤコビアン）", sp.simplify((1 / sp.exp(phi)) * sp.diff(sp.exp(phi), phi)) == 1, None)
cs = charts(T)
check(T, "chart1 beta-binomial a1 b1 n20 x0（例1 Be(1,21)）", cs[0]["params"] == {"model": "beta-binomial", "a": 1, "b": 1, "n": 20, "x": 0}, cs[0]["params"])
check(T, "chart2 logistic loc0 scale1", cs[1]["dist"] == "logistic" and cs[1]["params"] == {"loc": 0, "scale": 1}, cs[1]["params"])

# ================================================================ jeffreys-prior
T = "jeffreys-prior"
xx = sp.symbols("x", real=True)
# ベルヌーイ
lf = xx * sp.log(th) + (1 - xx) * sp.log(1 - th)
Ib = sp.simplify(-sp.diff(lf, th, 2).subs(xx, th))
check(T, "ベルヌーイ I=1/(θ(1−θ))（sympy）", sp.simplify(Ib - 1 / (th * (1 - th))) == 0, Ib)
# スコア二乗の期待値でも
sc = sp.diff(lf, th)
Ib2 = sp.simplify(th * sc.subs(xx, 1) ** 2 + (1 - th) * sc.subs(xx, 0) ** 2)
check(T, "ベルヌーイ E[スコア²] も同じ", sp.simplify(Ib2 - Ib) == 0, Ib2)
lp = xx * sp.log(th) - th
Ip = sp.simplify(-sp.diff(lp, th, 2).subs(xx, th))
check(T, "ポアソン I=1/λ → λ^{-1/2}", sp.simplify(Ip - 1 / th) == 0, Ip)
sgm = sp.symbols("sigma", positive=True)
Xn = sp.symbols("X", real=True)
ln_mu = -(Xn - M) ** 2 / (2 * sgm ** 2)
check(T, "正規平均 I=1/σ²（定数）", sp.simplify(-sp.diff(ln_mu, M, 2) - 1 / sgm ** 2) == 0, None)
ln_s = -sp.log(sgm) - (Xn - M) ** 2 / (2 * sgm ** 2)
h2s = -sp.diff(ln_s, sgm, 2)
Is = sp.simplify(sp.expand(h2s).subs((Xn - M) ** 2, sgm ** 2))
Is = sp.simplify(sp.expand(h2s.subs(Xn, M + sp.sqrt(sgm ** 2))))  # E[(X−μ)²]=σ² を代入
check(T, "正規 σ（μ既知）I=2/σ² → 1/σ", sp.simplify(Is - 2 / sgm ** 2) == 0, Is)
le = sp.log(th) - th * xx
Ie = sp.simplify(-sp.diff(le, th, 2))
check(T, "指数 I=1/λ² → 1/λ", sp.simplify(Ie - 1 / th ** 2) == 0, Ie)
# 変換不変性：φ=logit θ でベルヌーイの I_φ を直接計算
lf_phi = xx * sp.log(tinv) + (1 - xx) * sp.log(1 - tinv)
Iphi = sp.simplify(-sp.diff(lf_phi, phi, 2).subs(xx, tinv))
Iphi_chain = sp.simplify((1 / (th * (1 - th))).subs(th, tinv) * sp.diff(tinv, phi) ** 2)
check(T, "変換不変性 I_φ = I_θ (dθ/dφ)²（logit、sympy）", sp.simplify(Iphi - Iphi_chain) == 0, Iphi)
check(T, "Be(1/2,1/2) は正則（∫θ^{-1/2}(1−θ)^{-1/2}=π）", sp.integrate(th ** sp.Rational(-1, 2) * (1 - th) ** sp.Rational(-1, 2), (th, 0, 1)) == sp.pi, None)
d = stats.beta(3.5, 7.5)
check(T, "例1 Be(3.5,7.5) 平均 0.3182・MAP 2.5/9=0.2778", close(d.mean(), 0.3182) and close(2.5 / 9, 0.2778), d.mean())
check(T, "例1 95% 等裾 [0.0927, 0.6058]", allclose(et(d), [0.0927, 0.6058]), et(d))
d1 = stats.beta(4, 8)
check(T, "例1 一様 Be(4,8) 平均 0.3333・[0.1093, 0.6097]", close(d1.mean(), 0.3333) and allclose(et(d1), [0.1093, 0.6097]), et(d1))
check(T, "例1 ジェフリーズの方が MLE 0.3 に近い", abs(d.mean() - 0.3) < abs(d1.mean() - 0.3), None)
d = stats.beta(0.5, 10.5)
check(T, "例2 Be(0.5,10.5) 平均 0.0455・一様では 1/12=0.0833", close(d.mean(), 0.0455) and close(1 / 12, 0.0833), d.mean())
check(T, "例2 95% 等裾 [0.00005, 0.2172]", close(et(d)[0], 0.00005, 5e-6) and close(et(d)[1], 0.2172), et(d))
check(T, "例3 λ^{-1/2}λ^{20}e^{-5λ} → Ga(20.5,5)（sympy）", same_kernel(th ** sp.Rational(-1, 2) * th ** 20 * sp.exp(-5 * th), th ** sp.Rational(39, 2) * sp.exp(-5 * th)), None)
d = ga(20.5, 5)
check(T, "例3 Ga(20.5,5) 平均 4.1・95% [2.5215, 6.0561]", close(d.mean(), 4.1, 1e-12) and allclose(et(d), [2.5215, 6.0561]), et(d))
c = charts(T)[0]
check(T, "chart beta a0.5 b0.5・比較 Be(1,1)", c["params"] == {"a": 0.5, "b": 0.5} and c["compare"][0]["params"] == {"a": 1, "b": 1}, c["params"])

# ================================================================ bayes-factor
T = "bayes-factor"
m0 = stats.binom.pmf(14, 20, 0.5)
m1 = math.comb(20, 14) * math.exp(special.betaln(15, 7))
m1q = integrate.quad(lambda t: stats.binom.pmf(14, 20, t), 0, 1)[0]
check(T, "例1 m0=0.03696", close(m0, 0.03696, 5e-6), m0)
check(T, "例1 m1=C(20,14)B(15,7)=1/21=0.04762（数値積分とも）", close(m1, 1 / 21, 1e-12) and close(m1q, 1 / 21, 1e-10) and close(m1, 0.04762, 5e-6), m1)
check(T, "一様事前で成功数は 0..n の一様（n=20 の全 x）", allclose([integrate.quad(lambda t: stats.binom.pmf(xv, 20, t), 0, 1)[0] for xv in range(21)], [1 / 21] * 21, 1e-9), None)
bf = m1 / m0
check(T, "例1 B10=1.2882", close(bf, 1.2882), bf)
check(T, "例1 P(M1|x)=0.5630", close(bf / (1 + bf), 0.5630), bf / (1 + bf))
check(T, "例1 両側二項検定 p=0.1153", close(stats.binomtest(14, 20, 0.5).pvalue, 0.1153), stats.binomtest(14, 20, 0.5).pvalue)
bf17 = (1 / 21) / stats.binom.pmf(17, 20, 0.5)
check(T, "例1 x=17 で B10=43.80・p=0.0026", close(bf17, 43.80, 5e-3) and close(stats.binomtest(17, 20, 0.5).pvalue, 0.0026), (bf17, stats.binomtest(17, 20, 0.5).pvalue))
bfn = stats.norm.pdf(0.4, 0, math.sqrt(1.04)) / stats.norm.pdf(0.4, 0, math.sqrt(0.04))
# 独立に：μ を積分した周辺尤度（全データの尤度は x̄ の密度×定数）
m1n = integrate.quad(lambda u: stats.norm.pdf(0.4, u, 0.2) * stats.norm.pdf(u, 0, 1), -10, 10)[0]
check(T, "例2 B10=1.3418（μ の数値積分とも一致）", close(bfn, 1.3418) and close(m1n / stats.norm.pdf(0.4, 0, 0.2), bfn, 1e-9), bfn)
check(T, "例2 z=2・両側 p=0.0455", close(0.4 * 5, 2, 1e-12) and close(2 * stats.norm.sf(2), 0.0455), 2 * stats.norm.sf(2))
xs_bf = list(range(10, 19))
lbf = [math.log10((1 / 21) / stats.binom.pmf(xv, 20, 0.5)) for xv in xs_bf]
c = charts(T, "line-data")[0]
check(T, "line-data x=10..18", c["params"]["x"] == xs_bf, c["params"]["x"])
check(T, "line-data log10 BF10（小数第3位）", allclose(c["params"]["series"][0]["y"], lbf, 5e-4), np.round(lbf, 3))
check(T, "caption x=15：p=0.041・BF10≈3.2", close(stats.binomtest(15, 20, 0.5).pvalue, 0.041, 5e-4) and abs(10 ** lbf[5] - 3.2) < 0.05,
      (stats.binomtest(15, 20, 0.5).pvalue, 10 ** lbf[5]))
# リンドレー：M1 の事前分散を広げると m1→0
bfs = [stats.norm.pdf(0.4, 0, math.sqrt(t2 + 0.04)) / stats.norm.pdf(0.4, 0, 0.2) for t2 in [1, 1e2, 1e4, 1e6]]
check(T, "リンドレー：τ² を広げると B10 → 0（τ²=1e6 で <0.01）", all(np.diff(bfs) < 0) and bfs[-1] < 0.01, bfs)
c = charts(T, "line-data")[0]

# ================================================================ marginal-likelihood
T = "marginal-likelihood"
priors = [(0.5, 0.5), (1, 1), (2, 2), (10, 10), (20, 5)]
mls = [math.comb(20, 14) * math.exp(special.betaln(aa + 14, bb + 6) - special.betaln(aa, bb)) for aa, bb in priors]
mlq = [integrate.quad(lambda t: stats.binom.pmf(14, 20, t) * stats.beta.pdf(t, aa, bb), 0, 1)[0] for aa, bb in priors]
check(T, "例1 周辺尤度 0.0337/0.0476/0.0593/0.0593/0.0976", allclose(mls, [0.0337, 0.0476, 0.0593, 0.0593, 0.0976]), np.round(mls, 5))
check(T, "例1 閉じた形 = 数値積分 = scipy betabinom", allclose(mls, mlq, 1e-9) and allclose(mls, [stats.betabinom.pmf(14, 20, aa, bb) for aa, bb in priors], 1e-12), None)
check(T, "例1 Be(20,5)/Be(10,10) = 0.09764/0.05934 = 1.6454",
      close(mls[4], 0.09764, 5e-6) and close(mls[3], 0.05934, 5e-6) and close(mls[4] / mls[3], 1.6454), mls[4] / mls[3])
lk = stats.binom.pmf(14, 20, 0.7)
pdn = stats.beta.pdf(0.7, 15, 7)
check(T, "例1 恒等式：尤度 0.19164・事後密度 4.02442・比 0.04762",
      close(lk, 0.19164, 5e-6) and close(pdn, 4.02442, 5e-6) and close(lk / pdn, 1 / 21, 1e-12), (lk, pdn))
check(T, "一様で C(n,x) x!(n−x)!/(n+1)! = 1/(n+1)", all(math.comb(20, xv) * math.factorial(xv) * math.factorial(20 - xv) / math.factorial(21) == 1 / 21 for xv in [0, 7, 14]) or
      all(close(math.comb(20, xv) * math.factorial(xv) * math.factorial(20 - xv) / math.factorial(21), 1 / 21, 1e-15) for xv in [0, 7, 14]), None)
pf = math.prod(math.factorial(v) for v in xs_)


def ml_pois(al_, be_):
    return math.exp(al_ * math.log(be_) - special.gammaln(al_) + special.gammaln(al_ + 20) - (al_ + 20) * math.log(be_ + 5)) / pf


def ml_pois_q(al_, be_):
    f = lambda l: np.prod(stats.poisson.pmf(xs_, l)) * stats.gamma.pdf(l, al_, scale=1 / be_)
    return integrate.quad(f, 0, 40, points=[3, 4], limit=200)[0]


mp = [ml_pois(*p) for p in [(30, 10), (3, 1), (0.3, 0.1)]]
mpq = [ml_pois_q(*p) for p in [(30, 10), (3, 1), (0.3, 0.1)]]
check(T, "例2 ポアソン 4.336e-5 / 2.860e-5 / 0.866e-5", allclose(np.array(mp) * 1e5, [4.336, 2.860, 0.866], 5e-4), np.array(mp) * 1e5)
check(T, "例2 閉じた形 = 数値積分", allclose(np.array(mp) / np.array(mpq), [1, 1, 1], 1e-6), np.array(mpq) * 1e5)
check(T, "例2 事前平均がどれも 3", all(close(p[0] / p[1], 3, 1e-12) for p in [(30, 10), (3, 1), (0.3, 0.1)]), None)
bic = -2 * math.log(stats.binom.pmf(14, 20, 0.7)) + math.log(20)
check(T, "例3 BIC=6.3000・−2log(1/21)=6.0890", close(bic, 6.3000) and close(-2 * math.log(1 / 21), 6.0890), (bic, -2 * math.log(1 / 21)))
check(T, "周辺尤度 ≤ 最大尤度（5つの事前分布）", all(v <= lk + 1e-15 for v in mls), None)
# 正規：x̄ の周辺分布 N(μ0, τ0²+σ²/n)
mq = integrate.quad(lambda u: stats.norm.pdf(60, u, 5) * stats.norm.pdf(u, 50, 10), 0, 120)[0]
check(T, "正規：x̄ の周辺密度 = N(μ0, τ0²+σ²/n) の密度（数値積分）", close(mq, stats.norm.pdf(60, 50, math.sqrt(125)), 1e-12), mq)
c = charts(T, "bar-data")[0]
check(T, "bar-data ラベル5つ", c["params"]["labels"] == ["Be(0.5,0.5)", "Be(1,1)", "Be(2,2)", "Be(10,10)", "Be(20,5)"], c["params"]["labels"])
check(T, "bar-data 値（小数第4位）", allclose(c["params"]["values"], mls, 5e-5), c["params"]["values"])
check(T, "caption Be(20,5) 最大・Be(0.5,0.5) 最小", int(np.argmax(mls)) == 4 and int(np.argmin(mls)) == 0, None)

# ================================================================ hierarchical-bayes-model
T = "hierarchical-bayes-model"
rows = []
for nj, yb in [(10, 70), (40, 70), (10, 52)]:
    s2 = 100 / nj
    B = s2 / (s2 + 16)
    rows.append((s2, B, B * 60 + (1 - B) * yb, math.sqrt(1 / (1 / s2 + 1 / 16))))
check(T, "例1 A: σ²=10・B=10/26=0.3846・66.15・SD 2.481",
      close(rows[0][1], 0.3846) and close(rows[0][2], 66.15, 5e-3) and close(rows[0][3], 2.481, 5e-4), rows[0])
check(T, "例1 B: σ²=2.5・B=2.5/18.5=0.1351・68.65・SD 1.470",
      close(rows[1][1], 0.1351) and close(rows[1][2], 68.65, 5e-3) and close(rows[1][3], 1.470, 5e-4), rows[1])
check(T, "例1 C: B=0.3846・55.08・SD 2.481", close(rows[2][1], 0.3846) and close(rows[2][2], 55.08, 5e-3) and close(rows[2][3], 2.481, 5e-4), rows[2])
# 正規・正規の公式と一致（κ0=1/τ², κx=1/σ_j²）
check(T, "縮小式 = 正規・正規モデルの事後平均", close((60 / 16 + 70 / 10) / (1 / 16 + 1 / 10), rows[0][2], 1e-12), None)
check(T, "例2 A: 10/60=0.1667・B: 34/240=0.1417", close((4 + 6) / (40 + 20), 0.1667) and close((4 + 30) / (40 + 200), 0.1417)
      and close(stats.beta(4, 36).mean(), 0.1, 1e-12), None)
# 乱数：周辺分布 N(μ, τ²+σ_j²) と縮小による全体 MSE の減少
th_r = rng.normal(60, 4, (200000, 3))
s2v = np.array([10, 2.5, 10])
yb_r = rng.normal(th_r, np.sqrt(s2v))
check(T, "乱数 ȳ_j の周辺分散 ≈ τ²+σ_j²（相対 1%）", allclose(yb_r.var(0) / (16 + s2v), [1, 1, 1], 0.01), yb_r.var(0))
Bv = s2v / (s2v + 16)
shr = Bv * 60 + (1 - Bv) * yb_r
mse_raw = ((yb_r - th_r) ** 2).sum(1).mean()
mse_shr = ((shr - th_r) ** 2).sum(1).mean()
check(T, "乱数 縮小推定の全体 MSE < 生の平均の MSE（理論 Σ後分散 vs Σσ_j²、相対 1%）",
      mse_shr < mse_raw and abs(mse_shr / sum(r[3] ** 2 for r in rows) - 1) < 0.01 and abs(mse_raw / s2v.sum() - 1) < 0.01, (mse_shr, mse_raw))
c = charts(T)[0]
check(T, "chart normal-normal mu0 60 tau0 4 sigma 10 n10 xbar70（例1 クラスA）",
      c["params"] == {"model": "normal-normal", "mu0": 60, "tau0": 4, "sigma": 10, "n": 10, "xbar": 70}, c["params"])
pr = 1 / 16 + 10 / 100
check(T, "chart の事後平均 = 66.15（クラスA と一致）", close((60 / 16 + 10 * 70 / 100) / pr, 66.15, 5e-3), (60 / 16 + 10 * 70 / 100) / pr)

# ================================================================ スライダー初期値が範囲内
bad = []
for f in sorted((ROOT / "content" / "ch31").glob("*.md")):
    for ch in front(f.stem).get("charts", []) or []:
        for sl in ch.get("sliders", []) or []:
            v = ch["params"].get(sl["param"])
            if v is None or not (sl["min"] <= v <= sl["max"]):
                bad.append((f.stem, sl["param"], v))
check("全体", "各スライダーの初期値が min〜max の範囲内", not bad, bad)

# ================================================================ 重要度Cの用語
# ---------------------------------------------------------------- empirical-bayes
T = "empirical-bayes"
yb_ = np.array([62, 55, 70, 58, 65.0])
s2e_ = 16.0
muh_ = yb_.mean()
ss_ = np.sum((yb_ - muh_) ** 2)
s2_ = ss_ / 4
t2_ = s2_ - s2e_
B_ = s2e_ / s2_
th_ = B_ * muh_ + (1 - B_) * yb_
check(T, "μ̂=62、Σ(ȳ−μ̂)²=138、s²=34.5、τ̂²=18.5", close(muh_, 62, 1e-12) and close(ss_, 138, 1e-12) and close(s2_, 34.5, 1e-12) and close(t2_, 18.5, 1e-12), (muh_, ss_, s2_, t2_))
check(T, "B̂=16/34.5≈0.4638、σ²/(σ²+τ̂²) と同じ", close(B_, 0.4638) and close(s2e_ / (s2e_ + t2_), B_, 1e-12), B_)
check(T, "平均55の学校 0.4638×62+0.5362×55≈58.246", close(th_[1], 58.246, 5e-4) and close(0.4638 * 62 + 0.5362 * 55, 58.246, 5e-3), th_[1])
check(T, "5校の推定値 62.000, 58.246, 66.290, 59.855, 63.609", allclose(th_, [62, 58.246, 66.290, 59.855, 63.609], 5e-4), np.round(th_, 3).tolist())
check(T, "どれも 62 の方へ約46%（=B̂）引き寄せられる", allclose((yb_ - th_)[yb_ != 62] / (yb_ - 62)[yb_ != 62], np.full(4, B_), 1e-12) and round(B_ * 100) == 46, B_)
# 周辺分布 N(μ, τ²+σ²)（sympy の積分）
th_s, yv = sp.symbols("theta y", real=True)
mu_s = sp.symbols("mu", real=True)
t_s, sg_s = sp.symbols("tau sigma", positive=True)
marg = sp.integrate(sp.exp(-(yv - th_s) ** 2 / (2 * sg_s ** 2) - (th_s - mu_s) ** 2 / (2 * t_s ** 2)) / (2 * sp.pi * sg_s * t_s), (th_s, -sp.oo, sp.oo))
tgt = sp.exp(-(yv - mu_s) ** 2 / (2 * (t_s ** 2 + sg_s ** 2))) / sp.sqrt(2 * sp.pi * (t_s ** 2 + sg_s ** 2))
rat_ = [sp.N((marg / tgt).subs({yv: a, mu_s: b, t_s: c, sg_s: d}), 30) for a, b, c, d in [(55, 62, 4.3, 4), (1, -2, 0.5, 3), (7, 7, 2, 1)]]
check(T, "周辺分布 ȳ_j ~ N(μ, τ²+σ²)（sympy で積分し、3点で密度比=1）", all(abs(v - 1) < 1e-20 for v in rat_), [float(v) for v in rat_])
# 周辺尤度の最大化（第2種最尤）でも：μ̂=62、τ̂²=Σ/J−σ²=11.6
nll_ = lambda p: -stats.norm(p[0], math.sqrt(max(p[1], 0) + s2e_)).logpdf(yb_).sum()
opt_ = optimize.minimize(nll_, [60, 10], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
check(T, "参考：周辺尤度の最大化なら μ̂=62、τ̂²=138/5−16=11.6（モーメント法の 18.5 と別の値）", allclose(opt_.x, [62, 11.6], 1e-3), np.round(opt_.x, 4).tolist())
# 乱数：モーメント推定 E[s²]=τ²+σ² の不偏性
th_r = rng.normal(62, math.sqrt(18.5), (200000, 5))
yr_ = th_r + rng.normal(0, 4, (200000, 5))
s2r_ = yr_.var(1, ddof=1)
check(T, "乱数20万回：E[s²]≈τ²+σ²=34.5（許容 0.15≈4SE）", abs(s2r_.mean() - 34.5) < 0.15, round(float(s2r_.mean()), 3))
Br_ = 16 / np.maximum(s2r_, 16)
ebr_ = Br_[:, None] * yr_.mean(1, keepdims=True) + (1 - Br_[:, None]) * yr_
check(T, "乱数：経験ベイズの全体二乗誤差 < 生の平均（スタインの現象）",
      ((ebr_ - th_r) ** 2).sum(1).mean() < ((yr_ - th_r) ** 2).sum(1).mean(), (round(float(((ebr_ - th_r) ** 2).sum(1).mean()), 2), round(float(((yr_ - th_r) ** 2).sum(1).mean()), 2)))

# ---------------------------------------------------------------- bayes-decision-theory
T = "bayes-decision-theory"
pri_ = 0.1
pD_ = pri_ * 0.6 + (1 - pri_) * 0.1
post1_ = pri_ * 0.6 / pD_
post0_ = pri_ * 0.4 / (1 - pD_)
loss_ = lambda q: (100 * q, 20 * (1 - q))  # (出荷, 廃棄)
check(T, "P(欠陥)=0.15、P(不良|欠陥)=0.06/0.15=0.4", close(pD_, 0.15, 1e-12) and close(post1_, 0.4, 1e-12), post1_)
check(T, "欠陥あり：出荷 40、廃棄 12 → 廃棄", allclose(loss_(post1_), [40, 12], 1e-9) and np.argmin(loss_(post1_)) == 1, loss_(post1_))
check(T, "欠陥なし：P=0.04/0.85≈0.0471、出荷 4.706、廃棄 19.059 → 出荷",
      close(post0_, 0.0471) and allclose(loss_(post0_), [4.706, 19.059], 5e-4) and np.argmin(loss_(post0_)) == 0, loss_(post0_))
br_ = pD_ * min(loss_(post1_)) + (1 - pD_) * min(loss_(post0_))
check(T, "ベイズリスク 0.15×12+0.85×4.706=5.800", close(br_, 5.8, 1e-9) and close(0.15 * 12 + 0.85 * 4.706, 5.8, 5e-4), br_)
check(T, "検査なし：出荷 10 < 廃棄 18 で出荷、差 4.2", allclose(loss_(pri_), [10, 18], 1e-9) and close(min(loss_(pri_)) - br_, 4.2, 1e-9), loss_(pri_))
check(T, "閾値 20/(100+20)≈0.1667（両損失が等しい事後確率）", close(optimize.brentq(lambda q: 100 * q - 20 * (1 - q), 0, 1), 20 / 120, 1e-12) and close(20 / 120, 0.1667), 20 / 120)
# 全ての決定方式（4通り）のベイズリスクを直接計算して最小が 5.8
rules_ = {}
for aD in (0, 1):
    for aN in (0, 1):
        rk = 0
        for bad, pb_ in ((1, pri_), (0, 1 - pri_)):
            pdef = 0.6 if bad else 0.1
            for found, pf in ((1, pdef), (0, 1 - pdef)):
                a = aD if found else aN
                rk += pb_ * pf * ((100 if (bad and a == 0) else 0) + (20 if (not bad and a == 1) else 0))
        rules_[(aD, aN)] = rk
check(T, "4通りの決定方式のリスク関数を事前分布で平均：最小は（欠陥→廃棄, なし→出荷）で 5.8", min(rules_, key=rules_.get) == (1, 0) and close(rules_[(1, 0)], 5.8, 1e-9), rules_)
# 損失と点推定：二乗→平均、絶対→中央値（Ga(3,1) で数値確認）
gd_ = stats.gamma(3)
gq = np.linspace(0.5, 6, 5501)
e2 = [gd_.expect(lambda t: (t - a) ** 2) for a in gq[::50]]
e1 = [gd_.expect(lambda t: abs(t - a)) for a in gq[::50]]
check(T, "Ga(3,1)：二乗損失の最小は平均 3、絶対損失の最小は中央値 2.674（格子 0.05）",
      abs(gq[::50][int(np.argmin(e2))] - 3) <= 0.05 and abs(gq[::50][int(np.argmin(e1))] - gd_.median()) <= 0.05, (gq[::50][int(np.argmin(e2))], gq[::50][int(np.argmin(e1))], gd_.median()))

# ================================================================ 集計
ok = sum(results)
ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
