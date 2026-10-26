"""教科書 第48章（統計的決定理論）の検算。"""
import numpy as np
from scipy import integrate, stats

from tbcheck import allclose, body, chart, check, close, done

rng = np.random.default_rng(480)


def R(p, n, a, b):  # δ=(X+a)/(n+b) の 2 乗損失のリスク
    return (n * p * (1 - p) + (a - b * p) ** 2) / (n + b) ** 2


def R_mc(p, n, a, b, reps=400000):
    x = rng.binomial(n, p, reps)
    return (((x + a) / (n + b) - p) ** 2).mean()


T = "loss-function-and-risk"
p = chart(T)
n = 10
check(T, "図：3 つのリスク関数（式）", allclose(p["series"][0]["y"], [R(q, n, 0, 0) for q in p["x"]], 6e-6)
      and allclose(p["series"][1]["y"], [R(q, n, 1, 2) for q in p["x"]], 6e-6)
      and allclose(p["series"][2]["y"], [R(q, n, np.sqrt(n) / 2, np.sqrt(n)) for q in p["x"]], 6e-6))
check(T, "式をシミュレーションで確かめる（p=0.3、3 つ）", all(close(R_mc(0.3, n, a, b), R(0.3, n, a, b), 2e-4) for a, b in ((0, 0), (1, 2), (np.sqrt(n) / 2, np.sqrt(n)))))
check(T, "例1 0.025・0.01736・0.00694・0.01443", close(R(.5, n, 0, 0), .025, 1e-9) and close(R(.5, n, 1, 2), .01736, 5e-6) and close(R(0, n, 1, 2), .00694, 5e-6)
      and close(1 / (4 * (np.sqrt(n) + 1) ** 2), .01443, 5e-6))
br = [integrate.quad(lambda q: R(q, n, a, b), 0, 1)[0] for a, b in ((0, 0), (1, 2), (np.sqrt(n) / 2, np.sqrt(n)))]
check(T, "例2 ベイズリスク 1/60・1/72・0.0144", close(br[0], 1 / 60, 1e-9) and close(br[1], 1 / 72, 1e-9) and close(br[2], 0.0144, 5e-5), br)
grid = np.linspace(0, 1, 2001)
# 一様事前分布のベイズ規則：x ごとに事後期待損失を最小にする値は事後平均 (x+1)/(n+2)
best = [grid[np.argmin([(stats.beta(x + 1, n - x + 1).var() + (stats.beta(x + 1, n - x + 1).mean() - a) ** 2) for a in grid])] for x in (0, 3, 10)]
check(T, "2 乗損失のベイズ規則は事後平均 (x+1)/(n+2)", allclose(best, [(x + 1) / (n + 2) for x in (0, 3, 10)], 6e-4), best)
post = stats.beta(4, 8)
ab = grid[np.argmin([post.expect(lambda t: abs(t - a)) for a in grid[::20]]) * 20]
check(T, "絶対損失のベイズ規則は事後中央値", close(ab, post.median(), 0.011), (ab, post.median()))

T = "admissibility"
p = chart(T)
ns = p["x"]
f = lambda n, d: (2 * (n - 1) + (n - 1 - d) ** 2) / d ** 2
check(T, "図：MSE/σ⁴ = 2/(n−1)・(2n−1)/n²・2/(n+1)", allclose(p["series"][0]["y"], [f(n, n - 1) for n in ns], 6e-6)
      and allclose(p["series"][1]["y"], [f(n, n) for n in ns], 6e-6) and allclose(p["series"][2]["y"], [f(n, n + 1) for n in ns], 6e-6))
ds = np.linspace(5, 15, 10001)
check(T, "MSE を最小にする d は n+1（n=10）", close(ds[np.argmin(f(10, ds))], 11, 2e-3))
for s2 in (0.5, 3.0):
    x = rng.normal(0, np.sqrt(s2), (300000, 10))
    ss = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    m = [((ss / d - s2) ** 2).mean() / s2 ** 2 for d in (9, 10, 11)]
    check(T, f"例1 シミュレーション σ²={s2}：0.222・0.19・0.182", allclose(m, [2 / 9, 0.19, 2 / 11], 0.006), m)

T = "minimax-estimator"
p = chart(T)
n = 100
check(T, "図：n=100 のリスク", allclose(p["series"][0]["y"], [R(q, n, 0, 0) for q in p["x"]], 6e-7)
      and allclose(p["series"][1]["y"], [R(q, n, 5, 10) for q in p["x"]], 6e-7) and allclose(p["series"][2]["y"], [R(q, n, 1, 2) for q in p["x"]], 6e-7))
c = 1 / (4 * (np.sqrt(n) + 1) ** 2)
check(T, "例1 リスク 1/484≈0.00207、境目 0.292・0.708、n=10 で 0.175・0.825",
      close(c, 1 / 484, 1e-12) and close(c, 0.00207, 5e-6) and close((1 - np.sqrt(1 - 4 * c * n)) / 2, 0.292) and close((1 + np.sqrt(1 - 4 * c * n)) / 2, 0.708)
      and close((1 - np.sqrt(1 - 40 / (4 * (np.sqrt(10) + 1) ** 2))) / 2, 0.175))
check(T, "例1 a=√n/2 でリスクが p によらない", np.ptp([R(q, n, 5, 10) for q in np.linspace(0, 1, 101)]) < 1e-15)
check(T, "証明：一定リスクのベイズ規則は Beta(√n/2,√n/2) の事後平均 (X+5)/110", close((0 + 5) / (100 + 10), stats.beta(5, 105).mean(), 1e-12))
tau = np.array([1, 10, 100, 1000.0])
check(T, "例2 ベイズリスク τ²(σ²/n)/(τ²+σ²/n) → σ²/n（σ²/n=0.04）", close((tau ** 2 * 0.04 / (tau ** 2 + 0.04))[-1], 0.04, 1e-8)
      and all(np.diff(tau ** 2 * 0.04 / (tau ** 2 + 0.04)) > 0))

T = "james-stein-estimator"
p = chart(T)
pd = 10


def js_risk(r):
    k = np.arange(0, 400)
    return pd - (pd - 2) ** 2 * (stats.poisson.pmf(k, r * r / 2) / (pd - 2 + 2 * k)).sum()


check(T, "図：リスク p − (p−2)² E[1/‖X‖²]（非心カイ二乗をポアソン混合で）", allclose(p["series"][1]["y"], [js_risk(r) for r in p["x"]], 6e-4))
pp = []
for r in p["x"]:
    th = np.zeros(pd)
    th[0] = r
    X = rng.normal(size=(200000, pd)) + th
    cc = np.maximum(0, 1 - (pd - 2) / (X ** 2).sum(1))
    pp.append(((cc[:, None] * X - th) ** 2).sum(1).mean())
check(T, "図：正部分のリスク（別の乱数で ±0.03）", allclose(p["series"][2]["y"], pp, 0.03), [round(v, 3) for v in pp])
X = rng.normal(size=(200000, pd))
check(T, "θ=0 で JS のリスク 2（シミュレーション）", close((((1 - (pd - 2) / (X ** 2).sum(1))[:, None] * X) ** 2).sum(1).mean(), 2, 0.03))
check(T, "例1 2・4.38・7.96・1.26", close(js_risk(0), 2, 1e-9) and close(js_risk(2), 4.376, 5e-4) and close(js_risk(5), 7.964, 5e-4) and close(pp[0], 1.26, 0.03))
x = np.array([2.1, -0.4, 1.3, 0.8, -1.5, 3.0, 0.2, -0.9, 1.7, 0.5])
m, S = x.mean(), ((x - x.mean()) ** 2).sum()
d = m + (1 - 7 / S) * (x - m)
check(T, "例2 平均 0.68・S 17.516・係数 0.600・3.0→2.07・−1.5→−0.63", close(m, 0.68, 1e-12) and close(S, 17.516, 1e-9) and close(1 - 7 / S, 0.600, 5e-4)
      and close(d[5], 2.07, 0.005) and close(d[4], -0.63, 0.005), np.round(d, 3))
th = np.full(pd, 0.7)
Xs = rng.normal(size=(200000, pd)) + th
mm = Xs.mean(1, keepdims=True)
SS = ((Xs - mm) ** 2).sum(1, keepdims=True)
check(T, "例2 真の値がすべて等しいとき、平均へ縮める形のリスクは 3", close((((mm + (1 - 7 / SS) * (Xs - mm)) - th) ** 2).sum(1).mean(), 3, 0.03))
check(T, "本文の値", all(s in body(T) for s in ("4.38", "7.96", "1.26", "17.516", "0.600")))
done("第48章（統計的決定理論）")
