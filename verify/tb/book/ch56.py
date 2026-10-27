"""統計学の教科書 第56章「セミパラメトリック推論と予測の不確実性」の検算（効率限界・DML・コンフォーマル予測）。"""
import math

import numpy as np
from bookcheck import Book

b = Book("tb", 56)
rng = np.random.default_rng(56)
# 例題：効率限界
e = {1: 0.8, 0: 0.2}
tau = {1: 2.0, 0: 1.0}
part = {x: 1 / e[x] + 1 / (1 - e[x]) for x in (0, 1)}
b.num("X=1 の項", part[1], "6.25")
vt = 0.5 * (2 - 1.5) ** 2 + 0.5 * (1 - 1.5) ** 2
b.num("Var τ(X)", vt, "0.25")
bound = 0.5 * part[0] + 0.5 * part[1] + vt
b.num("効率限界", bound, "6.5")
b.num("無作為化", 4 + vt, "4.25")
b.num("比", bound / 4.25, "1.5", tol=0.05)
# 有効影響関数（オラクルの AIPW）の分散が限界に一致するかシミュレーション
N = 2_000_000
X = rng.integers(0, 2, N)
ex = np.where(X == 1, 0.8, 0.2)
T = (rng.uniform(size=N) < ex).astype(float)
m0 = 3.0 * X
m1 = m0 + np.where(X == 1, 2.0, 1.0)
Y = np.where(T == 1, m1, m0) + rng.standard_normal(N)
phi = m1 - m0 + T * (Y - m1) / ex - (1 - T) * (Y - m0) / (1 - ex)
b.check("AIPW の影響関数の分散 ≈ 6.5", abs(phi.var() - 6.5) < 0.03, phi.var())
# DML のシミュレーション
def nw(xtr, ytr, xte, h):
    W = np.exp(-((xte[:, None] - xtr[None, :]) / h) ** 2 / 2)
    return W @ ytr / W.sum(1)
R, n, th = 500, 500, 1.0
naive, dml, cov = [], [], []
for _ in range(R):
    x = rng.uniform(-2, 2, n)
    t = np.sin(2 * x) + x + rng.standard_normal(n)
    y = th * t + np.cos(2 * x) + 1.5 * x + rng.standard_normal(n)
    tc = t - t.mean()
    naive.append(tc @ (y - y.mean()) / (tc @ tc))
    idx = rng.permutation(n); folds = [idx[: n // 2], idx[n // 2:]]
    rt = np.empty(n); ry = np.empty(n)
    for k in range(2):
        te, tr = folds[k], folds[1 - k]
        rt[te] = t[te] - nw(x[tr], t[tr], x[te], 0.2)
        ry[te] = y[te] - nw(x[tr], y[tr], x[te], 0.2)
    bh = rt @ ry / (rt @ rt)
    res = ry - bh * rt
    se = math.sqrt(np.mean(rt ** 2 * res ** 2) / np.mean(rt ** 2) ** 2 / n)
    dml.append(bh); cov.append(abs(bh - th) < 1.96 * se)
b.num("単純な回帰の傾き", np.mean(naive), "1.728", tol=0.005)
b.num("偏り", np.mean(naive) - 1, "0.73", tol=0.005)
b.check("DML の偏り ≈ −0.002", abs(np.mean(dml) - 1) < 0.006, np.mean(dml) - 1)
b.check("DML の被覆率 ≈ 94.8%", abs(np.mean(cov) - 0.948) < 0.02 and b.has("94.8%"), np.mean(cov))
# 理論上の単純な回帰の傾き：1 + Cov(T, g(X))/Var(T)
xs = np.linspace(-2, 2, 400001)
mx = np.sin(2 * xs) + xs; gx = np.cos(2 * xs) + 1.5 * xs
slope = 1 + np.mean((mx - mx.mean()) * (gx - gx.mean())) / (mx.var() + 1)
b.check("単純な回帰の傾きの理論値と一致", abs(slope - np.mean(naive)) < 0.01, slope)
# コンフォーマル予測
r = sorted([0.2, 1.5, 0.7, 0.1, 2.3, 0.9, 0.4, 1.1, 3.0, 0.6, 0.3, 1.8, 0.8, 0.5, 1.3, 2.0, 0.05, 1.0, 0.95])
k = math.ceil(20 * 0.9)
b.num("順位", k, "18")
b.num("q", r[k - 1], "2.3")
b.num("下端", 10 - r[k - 1], "7.7")
b.num("上端", 10 + r[k - 1], "12.3")
# 被覆率の保証をシミュレーションで（裾の重い誤差、悪いモデルでも）
hits = []
for _ in range(20000):
    z = rng.standard_t(2, 20) * 3 + 5
    q = np.sort(abs(z[:19] - 4))[17]   # わざと外れた予測値 4
    hits.append(abs(z[19] - 4) <= q)
b.check("被覆率 = 18/20 = 0.9", abs(np.mean(hits) - 0.9) < 0.006, np.mean(hits))
# 章末問題
b.num("問1", 4 / 0.5 + 4 / 0.5, "16")
b.num("問3 順位", math.ceil(100 * 0.95), "95")
b.num("問3 上限", 0.95 + 1 / 100, "0.96")
b.done()
