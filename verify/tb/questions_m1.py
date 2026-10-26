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
lp = stats.laplace.rvs(size=(20000, 400), random_state=rng)
q.check("q47-m-estimator-1", abs(lp.mean(1).var() / np.median(lp, 1).var() - 2) < 0.15, "シミュレーション")
# モデルの誤り：ポアソン回帰（切片だけ）で過分散のデータ → I^{-1} は小さすぎる
y = rng.negative_binomial(2, 0.4, (20000, 200))  # 平均 3、分散 7.5
mu_hat = y.mean(1)
q.check("q47-m-estimator-2", abs(200 * np.log(mu_hat).var() - 7.5 / 9) < 0.05 and abs(1 / 3 - 7.5 / 9) > 0.4, "I^{-1}=1/μ=0.333 に対し実際は σ²/μ²=0.833（answer=false）")
x3 = np.array([1, 2, 3.]); y3 = np.array([1, 3, 2.]); b3 = (x3 * y3).sum() / (x3 * x3).sum(); e3 = y3 - b3 * x3
q.num("q47-sandwich-variance-estimator-1", (x3 ** 2 * e3 ** 2).sum() / 14 ** 2)
q.check("q47-sandwich-variance-estimator-1", abs((e3 ** 2).sum() / 2 / 14 - 0.0689) < 5e-5 and abs((x3 ** 2 * e3 ** 2).sum() - 10.7857) < 5e-5, "ふつうの分散 0.0689・Σx²e² 10.7857")
q.check("q47-sandwich-variance-estimator-2", chosen(q, "q47-sandwich-variance-estimator-2").startswith("回帰係数の推定値は変わらず"), "係数は同じ最小二乗推定値")
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
run(q)
print(f"修士1年（第45〜48章）の問題の検算: NG のある章 {ng}")
sys.exit(1 if ng else 0)
