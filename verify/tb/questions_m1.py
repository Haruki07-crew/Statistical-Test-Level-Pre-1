"""教科書 修士1年（第45〜51章）：問題の検算（章ごとに結果を保存する）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qhead import Q, chosen  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

ng = 0


def run(q):
    global ng
    try:
        q.done()
    except SystemExit as e:
        ng += int(bool(e.code))


rng = np.random.default_rng(4500)

# ---------- 第45章 ----------
q = Q(45)
# σ-加法族を総当たりで生成
Om = frozenset({1, 2, 3, 4})
F = {frozenset(), Om, frozenset({1}), frozenset({2, 3})}
while True:
    new = set(F) | {Om - a for a in F} | {a | b for a in F for b in F}
    if new == F:
        break
    F = new
q.check("q45-probability-space-1", len(F) == 8 and chosen(q, "q45-probability-space-1") == "8", f"生成された σ-加法族の大きさ {len(F)}")
q.num("q45-probability-space-2", sum(0.7 ** (2 * j - 1) * 0.3 for j in range(1, 400)))
g = rng.geometric(0.3, 400000)
q.check("q45-probability-space-2", abs((g % 2 == 0).mean() - 0.7 / 1.7) < 0.003, "シミュレーション")
u = rng.random(10 ** 6)
q.check("q45-lebesgue-integral-expectation-1", abs((50 ** 2 * (u <= 1 / 50)).mean() - 50) < 2 and chosen(q, "q45-lebesgue-integral-expectation-1").startswith("$X_n\\to0$（ほとんど確実に）だが"), "E[X₅₀]≈50")
q.num("q45-lebesgue-integral-expectation-2", integrate.quad(lambda t: (1 + t) ** -3, 0, np.inf)[0])
xs = (1 - rng.random(10 ** 6)) ** (-1 / 3) - 1  # P(X>t)=(1+t)^-3 の逆関数法
q.check("q45-lebesgue-integral-expectation-2", abs(xs.mean() - 0.5) < 0.01, f"シミュレーション {xs.mean():.4f}")
xx = np.linspace(0.01, 5, 50)
q.check("q45-radon-nikodym-theorem-1", np.allclose(stats.expon(scale=0.5).pdf(xx) / stats.expon.pdf(xx), 2 * np.exp(-xx)) and chosen(q, "q45-radon-nikodym-theorem-1") == "$2e^{-x}$"
        and abs(integrate.quad(lambda t: 2 * np.exp(-2 * t), 0, np.inf)[0] - 1) < 1e-9, "密度の比 2e^{−x}")
nn = np.arange(2, 10 ** 7, dtype=float)
hs = 1 / (nn * np.log(nn))
dd = hs.sum() - hs[: 10 ** 5 - 1].sum()
q.check("q45-borel-cantelli-lemma-1", abs(dd - np.log(np.log(1e7) / np.log(1e5))) < 0.01 and "n\\log n" in chosen(q, "q45-borel-cantelli-lemma-1"),
        f"Σ1/(n log n) の 10⁵〜10⁷ の部分 {dd:.4f} は log log の差 {np.log(np.log(1e7) / np.log(1e5)):.4f} と一致（log log N で発散）。他は和が有限か独立でない")
An = [sum(0.7 ** (k - 1) * 0.3 for k in range(1, m + 1)) for m in (1, 5, 20, 80)]
q.check("q45-probability-space-3", all(np.diff(An) > 0) and abs(An[-1] - 1) < 1e-10, "A_m = {m 回目までに表} の確率が増えて 1 に近づく（下からの連続性の例）")
q.check("q45-lebesgue-integral-expectation-3", abs(integrate.quad(lambda t: stats.gamma(2.5, scale=1.5).sf(t), 0, np.inf)[0] - 3.75) < 1e-6
        and abs(sum(stats.geom(0.25).sf(k) for k in range(0, 400)) - 4) < 1e-9, "ガンマ分布と幾何分布で裾の公式を確かめる")
q.num("q45-radon-nikodym-theorem-2", integrate.quad(lambda t: t * 2 * t, 0, 1)[0])
uq = rng.random(10 ** 6)
q.check("q45-radon-nikodym-theorem-2", abs((uq * 2 * uq).mean() - 2 / 3) < 0.003 and abs(np.sqrt(rng.random(10 ** 6)).mean() - 2 / 3) < 0.003, "P から引いて重み 2x・Q から直接（逆関数 √U）")
from scipy.special import zeta  # noqa: E402
q.num("q45-borel-cantelli-lemma-2", zeta(1.5))
run(q)

# ---------- 第46章 ----------
q = Q(46)
z = rng.normal(size=(20000, 400))
for n_ in (100, 400):
    pass
sq = [(400 * z[:, :n].mean(1) ** 2).mean() * (n / 400) for n in (400,)]
m100 = (z[:, :100].mean(1) ** 2).mean() * 100
m400 = (z[:, :400].mean(1) ** 2).mean() * 400
q.check("q46-stochastic-order-notation-1", abs(m100 - 1) < 0.05 and abs(m400 - 1) < 0.05 and chosen(q, "q46-stochastic-order-notation-1") == "$O_p(n^{-1})$", f"n·E[(X̄−μ)²]≈1（{m100:.3f}, {m400:.3f}）")
Z = rng.normal(size=100000)
ev, od = (1 * (Z + 3)), (-1 * (Z + 3))
q.check("q46-stochastic-order-notation-2", abs(ev.mean() - 3) < 0.02 and abs(od.mean() + 3) < 0.02 and (np.abs(ev) < 7).mean() > 0.99, "有界だが偶数・奇数で分布が違う（answer=false）")
p1, p2, n = 0.5, 0.3, 1600
q.num("q46-multivariate-clt-1", np.sqrt((p1 + p2 - (p1 - p2) ** 2) / n))
mm = rng.multinomial(n, [p1, p2, 0.2], 200000) / n
q.check("q46-multivariate-clt-1", abs((mm[:, 0] - mm[:, 1]).std() - 0.0218) < 3e-4 and abs(np.sqrt((0.25 + 0.21) / 1600) - 0.0170) < 5e-5, "シミュレーションと独立の場合 0.0170")
# 比の漸近分散：数値例でシミュレーションと比べる
muX, muY, sX, sY, sXY = 2.0, 3.0, 1.0, 1.5, 0.6
r0 = muY / muX
V = (sY ** 2 - 2 * r0 * sXY + r0 ** 2 * sX ** 2) / muX ** 2
xy = rng.multivariate_normal([muX, muY], [[sX ** 2, sXY], [sXY, sY ** 2]], (5000, 400))
R = xy[:, :, 1].mean(1) / xy[:, :, 0].mean(1)
q.check("q46-multivariate-clt-2", abs(400 * R.var() - V) < 0.04, f"n·V[R]={400 * R.var():.4f}・式 {V:.4f}")
ratio = lambda x: ((x - x.mean()) ** 2).max() / ((x - x.mean()) ** 2).sum()
q.check("q46-lindeberg-feller-clt-1", ratio(2.0 ** np.arange(1, 41)) > 0.7 and ratio(np.arange(1, 1001.0)) < 0.01 and ratio(np.arange(1000) % 2 * 1.0) < 0.01
        and ratio(np.sqrt(np.arange(1, 1001.0))) < 0.02 and chosen(q, "q46-lindeberg-feller-clt-1").startswith("$x_i=2^i$"), "2^i だけ比が 0 に向かわない")
q.num("q46-glivenko-cantelli-theorem-1", int(np.ceil(np.log(40) / (2 * 0.05 ** 2))))
q.check("q46-glivenko-cantelli-theorem-1", 2 * np.exp(-2 * 738 * 0.0025) <= 0.05 < 2 * np.exp(-2 * 737 * 0.0025), "737 では足りず 738 で満たす")
D1, D2 = [], []
for _ in range(3000):
    a = np.sort(rng.exponential(1, 50)); Fa = 1 - np.exp(-a); b = np.sort(rng.normal(size=50)); Fb = stats.norm.cdf(b); i = np.arange(1, 51)
    D1.append(max((i / 50 - Fa).max(), (Fa - (i - 1) / 50).max())); D2.append(max((i / 50 - Fb).max(), (Fb - (i - 1) / 50).max()))
q.check("q46-glivenko-cantelli-theorem-2", abs(np.quantile(D1, .95) - stats.kstwo(50).ppf(.95)) < 0.01 and abs(np.quantile(D2, .95) - stats.kstwo(50).ppf(.95)) < 0.01, "指数分布と正規分布で同じ分布（answer=true）")
q.num("q46-bootstrap-theory-1", 1 - 0.9 ** 10)
xb10 = np.arange(10.0)
q.check("q46-bootstrap-theory-1", abs((xb10[rng.integers(0, 10, (200000, 10))].max(1) == 9).mean() - (1 - 0.9 ** 10)) < 0.004, "再標本のシミュレーション")
zz46 = stats.norm.isf(0.025)
q.num("q46-local-alternatives-1", stats.norm.cdf(-zz46 + 3) + stats.norm.cdf(-zz46 - 3))
q.check("q46-local-alternatives-1", abs(stats.ncx2.sf(zz46 ** 2, 1, 9) - (stats.norm.cdf(-zz46 + 3) + stats.norm.cdf(-zz46 - 3))) < 1e-10, "非心カイ二乗（非心度 9）と一致")
q.num("q46-empirical-process-1", min(0.2, 0.7) - 0.2 * 0.7)
uu46 = rng.random((200000, 1))
q.check("q46-empirical-process-1", abs(np.cov((uu46[:, 0] <= 0.2), (uu46[:, 0] <= 0.7))[0, 1] - 0.06) < 0.002, "指示関数の共分散のシミュレーション")
Xo = rng.normal(size=(20000, 500)) * 3
Yo = rng.normal(size=(20000, 500)) / np.sqrt(np.arange(1, 501))
prod = np.abs(Xo * Yo)
q.check("q46-stochastic-order-notation-3", (prod[:, 499] > 0.1).mean() < (prod[:, 9] > 0.1).mean() and (prod[:, 499] > 0.1).mean() < 0.5, "O_p(1)×o_p(1) の確率が 0 へ")
mm3 = rng.multinomial(200, [0.2, 0.3, 0.5], 400000) / 200
q.check("q46-multivariate-clt-3", abs(np.cov(mm3[:, 0], mm3[:, 1])[0, 1] + 0.2 * 0.3 / 200) < 1e-5, "共分散 −p_jp_k/n")
xg = rng.normal(size=(200000, 40))
Fn = (xg <= 0.5).mean(1)
q.check("q46-glivenko-cantelli-theorem-3", abs(Fn.mean() - stats.norm.cdf(0.5)) < 0.001 and abs(Fn.var() - stats.norm.cdf(0.5) * stats.norm.sf(0.5) / 40) < 1e-4, "期待値 F(x)・分散 F(1−F)/n")
pr10 = 1 / np.arange(1, 10001)
q.num("q46-lindeberg-feller-clt-2", (pr10 * (1 - pr10)).sum())
q.num("q46-bootstrap-theory-2", 10 * (1 - 0.9 ** 10))
q.check("q46-bootstrap-theory-2", abs(np.mean([len(set(rng.integers(0, 10, 10))) for _ in range(40000)]) - 6.513) < 0.02, "再標本のシミュレーション")
zsum = stats.norm.isf(0.025) + stats.norm.isf(0.2)
q.num("q46-local-alternatives-2", int(np.ceil(zsum ** 2 * 0.25 / 0.05 ** 2)))
kk = np.arange(786); rej785 = np.abs(kk - 785 * 0.5) / np.sqrt(785 * 0.25) > 1.96
q.check("q46-local-alternatives-2", 0.8 <= stats.binom.pmf(kk, 785, 0.55)[rej785].sum() < 0.82, f"n=785 の二項分布での正確な検出力 {stats.binom.pmf(kk, 785, 0.55)[rej785].sum():.3f}")
ue = rng.random((100000, 50))
q.check("q46-empirical-process-2", ((ue <= 0).mean(1).var() == 0) and ((ue <= 1).mean(1).var() == 0) and abs(50 * (ue <= 0.5).mean(1).var() - 0.25) < 0.005, "両端で分散 0、t=0.5 で 0.25（answer=true）")
run(q)

# ---------- 第47章 ----------
q = Q(47)
n, lam = 5, 1.7
T = rng.poisson(n * lam, 10 ** 6)
q.check("q47-lehmann-scheffe-theorem-1", abs((T * (T - 1) / n ** 2).mean() - lam ** 2) < 0.01 and abs((T / n).astype(float).__pow__(2).mean() - (lam ** 2 + lam / n)) < 0.01
        and chosen(q, "q47-lehmann-scheffe-theorem-1") == "$\\dfrac{T(T-1)}{n^2}$", "E[T(T−1)]/n² = λ²、E[X̄²]=λ²+λ/n")
q.num("q47-lehmann-scheffe-theorem-2", 11 / 10 * 8.8)
e4 = rng.exponential(3, (10 ** 6, 4))
q.num("q47-basu-theorem-1", 2 / 20)
q.check("q47-basu-theorem-1", abs((e4[:, 0] ** 2 / e4.sum(1) ** 2).mean() - 0.1) < 0.001 and abs(stats.beta(1, 3).moment(2) - 0.1) < 1e-12, "シミュレーションと Beta(1,3) の 2 次モーメント")
q.num("q47-m-estimator-1", stats.laplace.var() / (1 / (4 * stats.laplace.pdf(0) ** 2)))
lp = stats.laplace.rvs(size=(10000, 2000), random_state=rng)
q.check("q47-m-estimator-1", abs(lp.mean(1).var() / np.median(lp, 1).var() - 2) < 0.15, "シミュレーション")
# モデルの誤り：ポアソン回帰（切片だけ）で過分散のデータ → I^{-1} は小さすぎる
y = rng.negative_binomial(2, 0.4, (20000, 200))  # 平均 3、分散 7.5
mu_hat = y.mean(1)
q.check("q47-m-estimator-2", abs(200 * np.log(mu_hat).var() - 7.5 / 9) < 0.05 and abs(1 / 3 - 7.5 / 9) > 0.4, "I^{-1}=1/μ=0.333 に対し実際は σ²/μ²=0.833（answer=false）")
x3 = np.array([1, 2, 3.]); y3 = np.array([1, 3, 2.]); b3 = (x3 * y3).sum() / (x3 * x3).sum(); e3 = y3 - b3 * x3
q.num("q47-sandwich-variance-estimator-1", (x3 ** 2 * e3 ** 2).sum() / 14 ** 2)
q.check("q47-sandwich-variance-estimator-1", abs((e3 ** 2).sum() / 2 / 14 - 0.0689) < 5e-5 and abs((x3 ** 2 * e3 ** 2).sum() - 10.7857) < 5e-5, "ふつうの分散 0.0689・Σx²e² 10.7857")
q.check("q47-sandwich-variance-estimator-2", chosen(q, "q47-sandwich-variance-estimator-2").startswith("回帰係数の推定値は変わらず"), "係数は同じ最小二乗推定値")
lam3, n3 = 1.4, 6
T3 = rng.poisson(n3 * lam3, 10 ** 6)
q.check("q47-lehmann-scheffe-theorem-3", abs(((1 - 1 / n3) ** T3).mean() - np.exp(-lam3)) < 0.002, "E[(1−1/n)^T] = e^{−λ}")
nm = rng.normal(0, 2, (20000, 1000))
q.check("q47-m-estimator-3", abs(1000 * np.median(nm, 1).var() - np.pi * 4 / 2) < 0.3, "N(0,4) の中央値の n·分散 ≈ πσ²/2 = 6.28")
xh = rng.uniform(1, 3, (20000, 200)); eh = rng.normal(size=(20000, 200)) * xh
bh = (xh * (2 * xh + eh)).sum(1) / (xh ** 2).sum(1)
res = (2 * xh + eh) - bh[:, None] * xh
vh = (xh ** 2 * res ** 2).sum(1) / ((xh ** 2).sum(1)) ** 2
q.check("q47-sandwich-variance-estimator-3", abs(vh.mean() / bh.var() - 1) < 0.05, f"サンドイッチ型の平均 / 実際の分散 = {vh.mean() / bh.var():.3f}")
xn8 = rng.normal(1, 2, (400000, 8))
q.num("q47-basu-theorem-2", (1 + 4 / 8) * 4)
q.check("q47-basu-theorem-2", abs((xn8.mean(1) ** 2 * xn8.var(1, ddof=1)).mean() - 6) < 0.05, "シミュレーション")
run(q)

# ---------- 第48章 ----------
q = Q(48)
Rk = lambda p, n, a, b: (n * p * (1 - p) + (a - b * p) ** 2) / (n + b) ** 2
q.num("q48-loss-function-and-risk-1", Rk(0.3, 20, 1, 2))
xb = rng.binomial(20, 0.3, 10 ** 6)
q.check("q48-loss-function-and-risk-1", abs((((xb + 1) / 22 - 0.3) ** 2).mean() - 0.00901) < 5e-5 and abs(0.3 * 0.7 / 20 - 0.0105) < 1e-12, "シミュレーションと X/20 の 0.0105")
post = stats.gamma(3, scale=0.5)
grid = np.linspace(0, 5, 2001)
best = grid[np.argmin([post.expect(lambda t: abs(t - a)) for a in grid[::10]]) * 10]
q.check("q48-loss-function-and-risk-2", abs(best - post.median()) < 0.03 and chosen(q, "q48-loss-function-and-risk-2") == "事後中央値", f"絶対損失の最小 {best:.3f}・中央値 {post.median():.3f}")
ds = np.linspace(2, 10, 8001)
q.num("q48-admissibility-1", ds[np.argmin((2 * 4 + (4 - ds) ** 2) / ds ** 2)])
q.check("q48-admissibility-1", abs(2 / 6 - (8 + 4) / 36) < 1e-12 and abs((8 + 0) / 16 - 0.5) < 1e-12, "d=6 で 1/3、d=4 で 0.5")
q.concept("q48-admissibility-2", "θ=θ₀ でリスク 0 なので優越されない（answer=true）")
q.num("q48-minimax-estimator-1", 1 / (4 * (np.sqrt(16) + 1) ** 2))
q.check("q48-minimax-estimator-1", np.ptp([Rk(p, 16, 2, 4) for p in np.linspace(0, 1, 101)]) < 1e-15 and abs(0.25 / 16 - 0.0156) < 5e-5, "リスク一定・X/16 の最大 0.0156")
q.check("q48-minimax-estimator-2", chosen(q, "q48-minimax-estimator-2").startswith("リスクが一定のベイズ規則"), "定理")
q.num("q48-james-stein-estimator-1", 1 - 4 / 10)
X12 = rng.normal(size=(400000, 12))
q.num("q48-james-stein-estimator-2", 12 - 10 ** 2 / 10)
q.check("q48-james-stein-estimator-2", abs(((((1 - 10 / (X12 ** 2).sum(1)))[:, None] * X12) ** 2).sum(1).mean() - 2) < 0.03, "シミュレーション")
post3 = stats.gamma(4, scale=0.5)
aa = np.linspace(0, 6, 6001)
q.check("q48-loss-function-and-risk-3", abs(aa[np.argmin([post3.var() + (post3.mean() - a) ** 2 for a in aa])] - post3.mean()) < 1e-3
        and abs(post3.expect(lambda t: (t - 1.7) ** 2) - (post3.var() + (post3.mean() - 1.7) ** 2)) < 1e-8, "E[(θ−a)²] = 分散 + (平均−a)²")
gd = lambda n, d: (2 * (n - 1) + (n - 1 - d) ** 2) / d ** 2
dd = np.linspace(2, 30, 28001)
q.check("q48-admissibility-3", all(abs(dd[np.argmin(gd(n, dd))] - (n + 1)) < 2e-3 and abs(gd(n, n + 1) - 2 / (n + 1)) < 1e-12 for n in (5, 10, 20)), "最小は d=n+1、MSE 2σ⁴/(n+1)")
pp = np.linspace(0, 1, 101)
q.check("q48-minimax-estimator-3", all(np.ptp([(n * x * (1 - x) + n / 4 * (1 - 2 * x) ** 2) / (n + np.sqrt(n)) ** 2 for x in pp]) < 1e-14 for n in (9, 25, 100)), "a=√n/2 でリスクが一定")
X7 = rng.normal(size=(400000, 7))
q.check("q48-james-stein-estimator-3", abs(((((1 - 5 / (X7 ** 2).sum(1)))[:, None] * X7) ** 2).sum(1).mean() - 2) < 0.03 and abs((1 / (X7 ** 2).sum(1)).mean() - 1 / 5) < 0.003, "p=7 でもリスク 2、E[1/χ²₇]=1/5")
run(q)

# ---------- 第49章 ----------
q = Q(49)
q.num("q49-kernel-density-estimation-1", 1.06 * 2 * 1000 ** -0.2)
q.check("q49-kernel-density-estimation-2", abs(32 ** -0.2 - 0.5) < 1e-12 and abs(32 ** -0.8 - 1 / 16) < 1e-12 and chosen(q, "q49-kernel-density-estimation-2") == "バンド幅 1/2 倍、MISE 1/16 倍", "32^{-1/5}=1/2・32^{-4/5}=1/16")
xk = np.array([0.5, 1.5, 2.2, 2.8, 3.6]); yk = np.array([1, 3, 4, 5, 9.0])
q.num("q49-nadaraya-watson-estimator-1", yk[np.abs(xk - 2) <= 1].mean())
from nadaraya_check import boundary_ratio  # noqa: E402
r1, r2 = boundary_ratio()
q.check("q49-nadaraya-watson-estimator-2", abs(r1 - 2) < 0.1 and r2 >= 3.8 and chosen(q, "q49-nadaraya-watson-estimator-2").startswith("境界では片側にしか"), f"h を半分にすると境界の偏りは NW で 1/{r1:.2f}、局所線形で 1/{r2:.2f}")
q.num("q49-influence-function-1", 3 ** 2 - 1)
eps = 1e-6
mix_var = (1 - eps) * 1 + eps * 9 - (eps * 3) ** 2
q.check("q49-influence-function-1", abs((mix_var - 1) / eps - 8) < 1e-3, "混合分布 (1−ε)N(0,1)+εδ₃ の分散の微分")
from scipy.stats import trim_mean  # noqa: E402
base = rng.normal(size=100)
bad1 = np.r_[base[:90], np.full(10, 1e6)]
bad2 = np.r_[base[:89], np.full(11, 1e6)]
q.check("q49-influence-function-2", abs(trim_mean(bad1, 0.1)) < 5 and trim_mean(bad2, 0.1) > 1e3 and chosen(q, "q49-influence-function-2") == "0.1", "10 個の外れ値は除かれ、11 個で壊れる")
q.num("q49-huber-estimator-1", min(1, 1.345 / 3))
hh = 0.05
f_ = stats.norm(0.3, 1)
Ef = integrate.quad(lambda u: stats.norm.pdf(u) * f_.pdf(0.8 - hh * u), -10, 10)[0]
f2 = f_.pdf(0.8) * (((0.8 - 0.3) ** 2) - 1)
q.check("q49-kernel-density-estimation-3", abs((Ef - f_.pdf(0.8)) / (hh ** 2 / 2 * f2) - 1) < 0.01, "偏り / (h²/2·f'') ≈ 1（h=0.05）")
xk3 = rng.random(30); yk3 = rng.normal(size=30); wk3 = stats.norm.pdf((0.5 - xk3) / 0.1)
grid = np.linspace(-3, 3, 60001)
q.check("q49-nadaraya-watson-estimator-3", abs(grid[np.argmin([(wk3 * (yk3 - a) ** 2).sum() for a in grid])] - (wk3 * yk3).sum() / wk3.sum()) < 2e-4, "重み付き 2 乗誤差の最小 = 重み付き平均")
q.check("q49-influence-function-3", abs(((1 - 1e-4) * 1.0 + 1e-4 * 7.0 - 1.0) / 1e-4 - 6.0) < 1e-6, "μ=1, x=7 で IF=6")
psi = lambda r: max(-1.345, min(1.345, r))
q.check("q49-huber-estimator-2", psi(0.8) == 0.8 and psi(-2.5) == -1.345 and chosen(q, "q49-huber-estimator-2").startswith("$\\psi(0.8)=0.8$、$\\psi(-2.5)=-1.345$"), "ψ(0.8)=0.8、ψ(−2.5)=−1.345")
run(q)

# ---------- 第50章 ----------
q = Q(50)
q.num("q50-linear-mixed-model-1", 6 * 4 / (6 * 4 + 12))
uu = rng.normal(0, 2, (200000, 1)); yy = uu + rng.normal(0, np.sqrt(12), (200000, 6))
bl = np.cov(uu[:, 0], yy.mean(1))[0, 1] / yy.mean(1).var()
q.check("q50-linear-mixed-model-1", abs(bl - 2 / 3) < 0.01, f"E[u|ȳ] の傾き（回帰で）{bl:.4f}")
q.num("q50-linear-mixed-model-2", 1 + 19 * 0.05)
u2 = rng.normal(0, np.sqrt(0.05), (100000, 1)); y2 = u2 + rng.normal(0, np.sqrt(0.95), (100000, 20))
q.check("q50-linear-mixed-model-2", abs(y2.mean(1).var() / (1 / 20) - 1.95) < 0.03, "シミュレーション")
q.num("q50-reml-1", (0.75 * 30 - 10) / 5)
q.check("q50-reml-1", (30 - 10) / 5 == 4, "REML は 4")
from gee_sim import marginal_truth  # noqa: E402
q.concept("q50-generalized-estimating-equations-1", "推定方程式の期待値が作業相関によらず 0（第50章の GEE のシミュレーションで独立の作業相関でも被覆率 0.95 を確認済み）")
mt = marginal_truth(1.0)
q.check("q50-generalized-estimating-equations-2", abs(mt - 0.662) < 0.002 and mt < 0.8 and chosen(q, "q50-generalized-estimating-equations-2") == "GEE の係数の絶対値のほうが小さい", f"母集団平均 {mt:.3f} < 個体ごと 0.8")
q.num("q50-glmm-1", 4 / (4 + np.pi ** 2 / 3))
lat = rng.normal(0, 2, (200000, 1)) + stats.logistic.rvs(size=(200000, 2), random_state=rng)
q.check("q50-glmm-1", abs(np.corrcoef(lat.T)[0, 1] - 0.549) < 0.005, "潜在変数（ランダム切片＋ロジスティック誤差）の相関のシミュレーション")
su2_, s2_, n_ = 4.0, 9.0, 5
uu3 = rng.normal(0, 2, 400000); yb3 = uu3 + rng.normal(0, 3, (400000, n_)).mean(1)
q.check("q50-linear-mixed-model-3", abs(np.cov(uu3, yb3)[0, 1] / yb3.var() - n_ * su2_ / (n_ * su2_ + s2_)) < 0.005, "E[u|ȳ] の傾き = nσu²/(nσu²+σ²)")
q.concept("q50-generalized-estimating-equations-3", "D_i, V_i が y によらないことから E[U]=0（第50章の GEE のシミュレーションで、独立の作業相関でも偏りが出ないことを確認済み）")
q.num("q50-reml-2", 45 / (12 - 3))
Xr = np.c_[np.ones(12), rng.normal(size=(12, 2))]
rss = [((yy_ - Xr @ np.linalg.lstsq(Xr, yy_, rcond=None)[0]) ** 2).sum() for yy_ in rng.normal(0, 2, (40000, 12))]
q.check("q50-reml-2", abs(np.mean(rss) / 9 - 4) < 0.05 and abs(np.mean(rss) / 12 - 3) < 0.05, "RSS/(n−p) は σ²=4 に不偏、RSS/n は 3 に偏る")
c_ = 16 * np.sqrt(3) / (15 * np.pi)
q.num("q50-glmm-2", 1.2 / np.sqrt(1 + c_ ** 2 * 1.5 ** 2))
run(q)

# ---------- 第51章 ----------
q = Q(51)
Pp = 0.5 + 0.5; K = Pp / (Pp + 1)
q.num("q51-kalman-filter-1", 10 + K * (12 - 10))
q.num("q51-kalman-filter-2", (-1 + np.sqrt(5)) / 2)
a_, P_ = 0.0, 1.0
for _ in range(200):
    Pp_ = P_ + 1; K_ = Pp_ / (Pp_ + 1); P_ = (1 - K_) * Pp_
q.check("q51-kalman-filter-2", abs(K_ - 0.618) < 5e-4, "ゲインの反復が収束する値")
q.num("q51-garch-model-1", np.log(0.5) / np.log(0.95))
w_, al, be = 0.05, 0.1, 0.85
N_ = 400000; zz = rng.normal(size=N_); rr = np.empty(N_); s2 = 1.0
for t_ in range(N_):
    rr[t_] = np.sqrt(s2) * zz[t_]; s2 = w_ + al * rr[t_] ** 2 + be * s2
rr = rr[1000:]
c1 = np.corrcoef(rr[:-1], rr[1:])[0, 1]; c2 = np.corrcoef(rr[:-1] ** 2, rr[1:] ** 2)[0, 1]
q.check("q51-garch-model-2", abs(c1) < 0.01 and c2 > 0.1, f"r の自己相関 {c1:.4f}・r² の自己相関 {c2:.3f}（answer=false）")
mats = [np.array([[0.9, 0.5], [0.1, 0.6]]), np.array([[1.0, 0.1], [0, 0.5]]), np.array([[0.5, 0.3], [0.2, 0.4]]), np.array([[0.6, 0.6], [0.6, 0.6]])]
stat = [np.abs(np.linalg.eigvals(M)).max() < 1 for M in mats]
q.check("q51-var-model-1", stat == [False, False, True, False] and q.qs["q51-var-model-1"]["answer"] == 2, f"固有値の最大 {[round(float(np.abs(np.linalg.eigvals(M)).max()), 3) for M in mats]}")
A2 = np.linalg.matrix_power(np.array([[0.5, 0.2], [0.1, 0.6]]), 2)
q.num("q51-var-model-2", A2[1, 0])
rej = 0
for _ in range(2000):
    xw = np.cumsum(rng.normal(size=200)); yw = np.cumsum(rng.normal(size=200)); Xw = np.c_[np.ones(200), xw]
    bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]; ew = yw - Xw @ bw
    rej += abs(bw[1] / np.sqrt((ew @ ew) / 198 / ((xw - xw.mean()) ** 2).sum())) > 1.96
q.check("q51-cointegration-1", rej / 2000 > 0.75 and chosen(q, "q51-cointegration-1").startswith("傾きの t 検定が、関係がないのに"), f"有意になる割合 {rej / 2000:.3f}")
ok_ = True
for qq in (0.25, 1.0, 3.0):
    P__ = (qq + np.sqrt(qq * qq + 4 * qq)) / 2
    ok_ &= abs(P__ ** 2 - qq * P__ - qq) < 1e-12 and abs(P__ / (P__ + 1) - (-qq + np.sqrt(qq * qq + 4 * qq)) / 2) < 1e-12
q.check("q51-kalman-filter-3", ok_, "P²−qP−q=0 と K の 2 つの表し方が一致（q=0.25, 1, 3）")
q.check("q51-garch-model-3", abs(rr.var() - 0.05 / (1 - 0.95)) < 0.08, f"シミュレーションの分散 {rr.var():.3f} ≈ ω/(1−α−β)=1")
A3 = np.array([[0.5, 0.2], [0.1, 0.6]]); c3 = np.array([1.0, 0.5]); y3 = np.zeros(2)
for _ in range(300):
    y3 = c3 + A3 @ y3
q.check("q51-var-model-3", np.allclose(y3, np.linalg.solve(np.eye(2) - A3, c3)) and np.allclose(np.linalg.solve(np.eye(2) - A3, c3), [2.778, 1.944], atol=5e-4), "反復の極限 = (I−A)⁻¹c")
q.num("q51-cointegration-2", -0.25 * (10 - 2 * 3))
run(q)
print(f"修士1年（第45〜51章）の問題の検算: NG のある章 {ng}")
sys.exit(1 if ng else 0)
