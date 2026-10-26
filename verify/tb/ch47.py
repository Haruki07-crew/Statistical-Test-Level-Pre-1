"""教科書 第47章（推定の理論の発展）の検算。"""
import numpy as np
from scipy import integrate, stats
from scipy.special import gammaln

from sandwich_sim import run
from tbcheck import allclose, body, chart, check, close, done

rng = np.random.default_rng(470)

T = "lehmann-scheffe-theorem"
p = chart(T)
ns = p["x"]
check(T, "図：4 つの推定量の MSE の式", allclose(p["series"][0]["y"], [1 / (3 * n) for n in ns], 6e-6)
      and allclose(p["series"][1]["y"], [1 / (n * (n + 2)) for n in ns], 6e-6) and allclose(p["series"][2]["y"], [2 / ((n + 1) * (n + 2)) for n in ns], 6e-6)
      and allclose(p["series"][3]["y"], [1 / (n + 1) ** 2 for n in ns], 6e-6))
x = rng.random((300000, 10))
mx = x.max(1)
est = {"2X̄": 2 * x.mean(1), "UMVUE": 11 / 10 * mx, "max": mx, "opt": 12 / 11 * mx}
mse = {k: ((v - 1) ** 2).mean() for k, v in est.items()}
check(T, "シミュレーション（n=10）：MSE 0.0333・0.00833・0.0152・0.00826、UMVUE は不偏",
      close(mse["2X̄"], 1 / 30, 4e-4) and close(mse["UMVUE"], 1 / 120, 1e-4) and close(mse["max"], 2 / 132, 2e-4) and close(mse["opt"], 1 / 121, 1e-4)
      and close(est["UMVUE"].mean(), 1, 1e-3), mse)
cs = np.linspace(0.9, 1.2, 3001)
check(T, "例2 MSE 最小の定数倍は (n+2)/(n+1)", close(cs[np.argmin([c * c * 10 / 12 - 2 * c * 10 / 11 + 1 for c in cs])], 12 / 11, 2e-4))
xp = np.arange(0, 200)
lam = 1.3
pmf = stats.poisson.pmf(xp, lam)
check(T, "例3 E[(−1)^X] = e^{−2λ}", close((pmf * (-1.0) ** xp).sum(), np.exp(-2 * lam), 1e-12))

T = "m-estimator"
p = chart(T)
k = 1.345


def hub(d):
    e2 = integrate.quad(lambda z: min(z * z, k * k) * d.pdf(z), -np.inf, np.inf, limit=200)[0]
    return e2 / (d.cdf(k) - d.cdf(-k)) ** 2


nus = p["x"]
check(T, "図：平均 ν/(ν−2)・中央値 1/(4f(0)²)・フーバー", allclose(p["series"][0]["y"], [v / (v - 2) for v in nus], 6e-5)
      and allclose(p["series"][1]["y"], [1 / (4 * stats.t(v).pdf(0) ** 2) for v in nus], 6e-5) and allclose(p["series"][2]["y"], [hub(stats.t(v)) for v in nus], 6e-5))
check(T, "例1・2 正規：π/2≈1.571・効率 0.637・フーバー 1.053（効率 95%）、t3：中央値 1.851・フーバー 1.556、ラプラス 2 と 1",
      close(1 / (4 * stats.norm.pdf(0) ** 2), 1.571) and close(2 / np.pi, 0.637) and close(hub(stats.norm), 1.053) and close(1 / hub(stats.norm), 0.95, 0.005)
      and close(1 / (4 * stats.t(3).pdf(0) ** 2), 1.851) and close(hub(stats.t(3)), 1.556) and close(stats.laplace.var(), 2, 1e-12)
      and close(1 / (4 * stats.laplace.pdf(0) ** 2), 1, 1e-12))
z = stats.t(3).rvs((20000, 400), random_state=rng)
med = np.median(z, 1)
check(T, "シミュレーション：t3 で n·V[中央値]≈1.85、n·V[平均]≈3", close(400 * med.var(), 1.851, 0.08) and close(400 * z.mean(1).var(), 3, 0.25), (400 * med.var(), 400 * z.mean(1).var()))


def huber_loc(s, it=30):
    th = np.median(s, 1)
    for _ in range(it):
        r = s - th[:, None]
        w = np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-12))
        th = (w * s).sum(1) / w.sum(1)
    return th


check(T, "シミュレーション：t3 で n·V[フーバー]≈1.56", close(400 * huber_loc(z).var(), 1.556, 0.08), 400 * huber_loc(z).var())

T = "sandwich-variance-estimator"
p = chart(T)
res = {g: run(g) for g in p["x"]}
check(T, "図：被覆率（ふつう・HC0・HC3）がシミュレーションの値と一致",
      allclose(p["series"][0]["y"], [res[g][4] for g in p["x"]], 6e-5) and allclose(p["series"][1]["y"], [res[g][5] for g in p["x"]], 6e-5)
      and allclose(p["series"][2]["y"], [res[g][6] for g in p["x"]], 6e-5))
r2 = res[2]
check(T, "例1 γ=2：SD 0.414・ふつう 0.346・HC0 0.398・HC3 0.410・被覆率 0.90 / 0.939 / 0.946",
      all(close(a, b, 6e-4) for a, b in zip(r2[:4], (0.414, 0.346, 0.398, 0.410))) and close(r2[4], 0.90, 0.005) and close(r2[5], 0.939, 6e-4) and close(r2[6], 0.946, 6e-4))
r0 = res[0]
check(T, "例2 γ=0：SD 0.355・ふつう 0.348・HC3 0.353・被覆率 約 0.95", all(close(a, b, 6e-4) for a, b in zip((r0[0], r0[1], r0[3]), (0.355, 0.348, 0.353)))
      and close(r0[4], 0.95, 0.005) and close(r0[6], 0.95, 0.005))
r2b = run(2, seed=4701)
check(T, "別の乱数でも γ=2 の被覆率の傾向は同じ（ふつう < 0.91、HC3 > 0.94）", r2b[4] < 0.91 and r2b[6] > 0.94, (r2b[4], r2b[6]))
check(T, "e² ≈ 7.4 倍・ふつうの区間は 0.87 まで下がる", close(np.exp(2), 7.39, 0.005) and close(res[3][4], 0.87, 0.005))

T = "basu-theorem"
n = 5
x = rng.random((400000, n)) * 3
mx, mn = x.max(1), x.min(1)
v = mn / mx
check(T, "例2 E[V]=1/n=0.2、相関ほぼ 0", close(v.mean(), 0.2, 0.002) and abs(np.corrcoef(mx, v)[0, 1]) < 0.01, (v.mean(), np.corrcoef(mx, v)[0, 1]))
check(T, "例2 E[X(1)]=θ/(n+1)・E[X(n)]=nθ/(n+1)", close(mn.mean(), 3 / 6, 0.003) and close(mx.mean(), 15 / 6, 0.003))
e = rng.exponential(2, (400000, n))
S = e.sum(1)
U = e[:, 0] / S
check(T, "例3 E[U]=1/n・Var[U]=2/30−0.04≈0.0267（Beta(1,4) の分散）", close(U.mean(), 0.2, 0.002) and close(U.var(), 0.0267, 4e-4)
      and close(2 / 30 - 0.04, stats.beta(1, 4).var(), 1e-12) and close(stats.beta(1, 4).var(), 0.0267, 5e-5))
xn = rng.normal(3, 2, (200000, 8))
check(T, "例1 正規標本の X̄ と S² の相関ほぼ 0", abs(np.corrcoef(xn.mean(1), xn.var(1, ddof=1))[0, 1]) < 0.01, np.corrcoef(xn.mean(1), xn.var(1, ddof=1))[0, 1])
check(T, "本文の値", all(s in body(T) for s in ("0.2", "0.0267", "Beta}(1,n-1)")))
done("第47章（推定の理論の発展）")
