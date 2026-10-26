"""教科書 第52章（高次元統計）の検算。"""
import numpy as np
from scipy import integrate, stats

from lasso_sim import errors, lasso
from tbcheck import allclose, body, chart, check, close, done

T = "concentration-inequality"
p = chart(T)
t = 0.05
exact = lambda n: stats.binom.cdf(np.floor(n * (0.5 - t) + 1e-9), n, 0.5) + stats.binom.sf(np.ceil(n * (0.5 + t) - 1e-9) - 1, n, 0.5)
check(T, "図：チェビシェフ・ヘフディング・正確な確率", allclose(p["series"][0]["y"], [min(1, 0.25 / (n * t * t)) for n in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], [min(1, 2 * np.exp(-2 * n * t * t)) for n in p["x"]], 6e-5) and allclose(p["series"][2]["y"], [exact(n) for n in p["x"]], 6e-5))
check(T, "例1 0.1・0.0135・0.0017・約 8 倍", close(0.25 / (1000 * 0.0025), 0.1, 1e-12) and close(2 * np.exp(-5), 0.0135, 5e-5) and close(exact(1000), 0.0017, 5e-5)
      and close(2 * np.exp(-5) / exact(1000), 8, 0.5))
rng = np.random.default_rng(52)
x = rng.random((200000, 2050))
dev = np.abs(x.mean(1) - 0.5) >= 0.03
check(T, "例2 n ≥ 2049.4（n=2050 の一様分布で外れる割合は 5% 以下）", close(np.log(40) / 0.0018, 2049.4, 0.05) and dev.mean() <= 0.05, dev.mean())
em = lambda q: integrate.quad(lambda v: v * q * stats.norm.pdf(v) * stats.norm.cdf(v) ** (q - 1), -10, 12)[0]
check(T, "例3 E max 3.24（上界 3.72）・3.85（4.29）", close(em(1000), 3.24, 0.005) and close(np.sqrt(2 * np.log(1000)), 3.72, 0.005)
      and close(em(10000), 3.85, 0.005) and close(np.sqrt(2 * np.log(10000)), 4.29, 0.005))
z = rng.normal(size=(20000, 1000))
check(T, "和集合上界：P(max|Z| > √(2 log(2p/δ))) ≤ δ（p=1000、δ=0.05）", (np.abs(z).max(1) > np.sqrt(2 * np.log(2000 / 0.05))).mean() <= 0.05)
lam = np.linspace(0.001, 5, 5000)
check(T, "ヘフディングの補題：Bernoulli(0.5) で E e^{λ(X−0.5)} ≤ e^{λ²/8}", np.all(np.cosh(lam / 2) <= np.exp(lam ** 2 / 8) + 1e-15))

T = "high-dimensional-regression"
p = chart(T)
E = {q: errors(q) for q in [10, 20, 50, 80, 90, 100, 120, 200, 400, 1000]}
check(T, "図：log₁₀ 誤差（同じ乱数で計算し直す）", all(allclose(p["series"][k]["y"], [np.log10(E[q][k]) for q in p["x"]], 6e-4) for k in range(3)))
tab = {10: (0.110, 0.348, 0.050), 50: (1.011, 0.558, 0.058), 90: (10.69, 0.561, 0.055), 200: (3.581, 0.760, 0.048), 1000: (4.609, 0.999, 0.053)}
check(T, "表の値", all(allclose(E[q], v, 6e-4 if q != 90 else 6e-3) for q, v in tab.items()), {q: np.round(E[q], 4) for q in tab})
check(T, "理論式 50/49≈1.02・5×0.9+100/899≈4.61、p=100 で 1 万超", close(50 / 49, 1.02, 0.005) and close(5 * 0.9 + 100 / 899, 4.61, 0.005) and E[100][0] > 1e4)
check(T, "p<n の最小二乗の誤差 ≈ σ²p/(n−p−1)（p=10・50・80）", all(close(E[q][0], q / (99 - q), 0.06 * q / (99 - q) + 0.01) for q in (10, 50, 80)))

T = "lasso-theory"
# ラッソの解が KKT 条件を満たすことを確かめる
rng = np.random.default_rng(5200)
X = rng.normal(size=(80, 120))
b = np.zeros(120)
b[:4] = 1.5
y = X @ b + rng.normal(size=80)
lm = 0.3
bl = lasso(X, y, lm, tol=1e-12, it=5000)
g = X.T @ (y - X @ bl) / 80
nz = np.abs(bl) > 1e-10
check(T, "座標降下法の解が KKT 条件を満たす", np.allclose(g[nz], lm * np.sign(bl[nz]), atol=1e-6) and np.all(np.abs(g[~nz]) <= lm + 1e-6))
zz = np.array([3.1, -0.4, 1.2, 0.05, -2.2])
check(T, "例1 ソフト閾値 2.6・0・0.7・0・−1.7", allclose(np.sign(zz) * np.maximum(np.abs(zz) - 0.5, 0), [2.6, 0, 0.7, 0, -1.7], 1e-12)
      and allclose(lasso(np.eye(5) * np.sqrt(5), zz * np.sqrt(5), 0.5), [2.6, 0, 0.7, 0, -1.7], 1e-9))
p = chart(T)
pp, s = 200, 5
rec = {1.0: [], 2.0: []}
for n in p["x"]:
    rng = np.random.default_rng(520 + n)
    ok = {1.0: 0, 2.0: 0}
    for _ in range(100):
        X = rng.normal(size=(n, pp))
        b = np.zeros(pp)
        b[:s] = 1
        e = rng.normal(size=n)
        y = X @ b + e
        for mult in (1.0, 2.0):
            bl = lasso(X, y, mult * np.sqrt(2 * np.log(pp) / n))
            ok[mult] += set(np.nonzero(np.abs(bl) > 1e-10)[0]) == set(range(s))
    for mult in (1.0, 2.0):
        rec[mult].append(ok[mult] / 100)
check(T, "図：正しく選ぶ確率（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], rec[1.0], 1e-9) and allclose(p["series"][1]["y"], rec[2.0], 1e-9), rec)
rng = np.random.default_rng(720)
c1, c2, e1, e2 = [], [], [], []
for _ in range(100):
    X = rng.normal(size=(200, pp))
    b = np.zeros(pp)
    b[:s] = 1
    y = X @ b + rng.normal(size=200)
    bl = lasso(X, y, 2 * np.sqrt(2 * np.log(pp) / 200))
    S = np.nonzero(np.abs(bl) > 1e-10)[0]
    bp = np.zeros(pp)
    bp[S] = np.linalg.lstsq(X[:, S], y, rcond=None)[0]
    c1.append(bl[:s].mean()); c2.append(bp[:s].mean()); e1.append(((bl - b) ** 2).sum()); e2.append(((bp - b) ** 2).sum())
check(T, "例2 2λ=0.46・係数の平均 0.52 → ポストラッソ 1.00・誤差 1.19 → 0.025",
      close(2 * np.sqrt(2 * np.log(200) / 200), 0.46, 0.005) and close(np.mean(c1), 0.52, 0.006) and close(np.mean(c2), 1.00, 0.005)
      and close(np.mean(e1), 1.19, 0.005) and close(np.mean(e2), 0.025, 5e-4), (np.mean(c1), np.mean(c2), np.mean(e1), np.mean(e2)))
check(T, "本文の値", all(v in body(T) for v in ("54%", "0.52", "1.00", "1.19", "0.025")))
done("第52章（高次元統計）")
