"""統計学の教科書 第47章「推定の理論の発展」の検算（厳密計算・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("tb", 47)
rng = np.random.default_rng(47)
# 47.1 ポアソンの UMVUE
n, lam = 10, 1.0
b.num("e^{-1}", math.exp(-1), "0.368")
b.num("素朴な推定量の分散", math.exp(-1) * (1 - math.exp(-1)), "0.233")
t = np.arange(0, 80)
pt = st.poisson.pmf(t, n * lam)
mean_u = (pt * 0.9 ** t).sum()
var_u = (pt * 0.81 ** t).sum() - mean_u ** 2
b.check("UMVUE は不偏", abs(mean_u - math.exp(-1)) < 1e-12, mean_u)
b.num("UMVUE の分散", var_u, "0.0142")
b.check("= e^{-1.9} − e^{-2}", abs(var_u - (math.exp(-1.9) - math.exp(-2))) < 1e-12, "")
b.check("約 16 分の 1", abs(0.2325 / var_u - 16.3) < 0.3, 0.2325 / var_u)
b.num("CR 下限", math.exp(-2) * lam / n, "0.0135")
T = rng.poisson(n * lam, 200000)
X1 = rng.binomial(T, 1 / n)
b.check("条件付き分布 X1|T ~ B(T,1/n) のシミュレーション", abs(np.mean(X1 == 0) - math.exp(-1)) < 0.003, "")
# 47.2 バス
E = rng.exponential(1, (200000, 5))
r = E[:, 0] / E.sum(1)
b.num("E[X1/T] (n=5)", 1 / 5, "0.2")
b.check("シミュレーションで E[X1/T] ≈ 0.2", abs(r.mean() - 0.2) < 0.002, r.mean())
b.check("X1/T と T は無相関", abs(np.corrcoef(r, E.sum(1))[0, 1]) < 0.01, "")
N = rng.standard_normal((200000, 10))
b.check("正規分布の X̄ と S² は無相関（2乗も）", abs(np.corrcoef(N.mean(1) ** 2, N.var(1, ddof=1))[0, 1]) < 0.01, "")
# 47.4 誤ったモデル
b.num("本当の標準誤差", math.sqrt(2 / 100), "0.141")
b.num("モデルの標準誤差", 2 / math.sqrt(100), "0.200")
R = 20000
Xs = rng.gamma(2, 1, (R, 100))
m = Xs.mean(1)
mod = m / 10
sand = Xs.std(1) / 10
c1, c2 = np.mean(abs(m - 2) < 1.96 * mod), np.mean(abs(m - 2) < 1.96 * sand)
b.check("被覆率 約99% と 約94%", abs(c1 - 0.99) < 0.006 and abs(c2 - 0.94) < 0.006, (c1, c2))
b.check("約 1.4 倍", abs(0.2 / math.sqrt(0.02) - 1.41) < 0.01, "")
# 回帰の不均一分散
x = np.repeat(np.arange(1, 11), 10).astype(float)
s = x ** 2 / 20
X = np.c_[np.ones(100), x]
XtXi = np.linalg.inv(X.T @ X)
xc = x - x.mean()
b.num("傾きの本当の標準誤差", math.sqrt((xc ** 2 * s ** 2).sum()) / (xc ** 2).sum(), "0.108")
sc, sw, cc, cw = [], [], [], []
for _ in range(20000):
    y = 1 + 2 * x + rng.standard_normal(100) * s
    bb = XtXi @ X.T @ y
    e = y - X @ bb
    a = math.sqrt((e @ e / 98) * XtXi[1, 1])
    w = math.sqrt((XtXi @ (X.T * e ** 2) @ X @ XtXi)[1, 1])
    sc.append(a); sw.append(w)
    cc.append(abs(bb[1] - 2) < 1.96 * a); cw.append(abs(bb[1] - 2) < 1.96 * w)
b.num("通常の標準誤差の平均", np.mean(sc), "0.087", tol=0.002)
b.num("サンドイッチの平均", np.mean(sw), "0.104", tol=0.002)
b.check("被覆率 約88% と 約93%", abs(np.mean(cc) - 0.88) < 0.008 and abs(np.mean(cw) - 0.93) < 0.008, (np.mean(cc), np.mean(cw)))
# 章末問題
nn, p = 6, 0.3
Tb = np.arange(nn + 1)
pb = st.binom.pmf(Tb, nn, p)
b.check("問1 T(T−1)/(n(n−1)) は p² の不偏推定量", abs((pb * Tb * (Tb - 1) / (nn * (nn - 1))).sum() - p * p) < 1e-12, "")
U = rng.uniform(0, 3, (200000, 5))
ratio = U.min(1) / U.max(1)
b.check("問2 比と最大値は無相関", abs(np.corrcoef(ratio, U.max(1))[0, 1]) < 0.01, "")
b.done()
