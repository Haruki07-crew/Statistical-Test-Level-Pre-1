"""準1級 教科書 第18章「質的回帰」の検算（IRLS を自分で実装し、scipy の最適化と照合）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import optimize as op, stats as st

b = Book("pre1", 18)
x = np.array([1, 2, 3, 4, 5, 6.])
n = np.full(6, 20.)
y = np.array([2, 5, 7, 13, 15, 18.])
b.check("表の効果あり", b.has("| 効果あり（20 匹中） | 2 | 5 | 7 | 13 | 15 | 18 |"), "")
X = np.column_stack([np.ones(6), x])
bb = np.zeros(2)
for _ in range(50):
    eta = X @ bb
    p = 1 / (1 + np.exp(-eta))
    W = n * p * (1 - p)
    z = eta + (y - n * p) / W
    bb = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
p = 1 / (1 + np.exp(-X @ bb))
W = n * p * (1 - p)
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (W[:, None] * X))))
nll = lambda c: -(y * np.log(1 / (1 + np.exp(-(c[0] + c[1] * x)))) + (n - y) * np.log(1 - 1 / (1 + np.exp(-(c[0] + c[1] * x))))).sum()
opt = op.minimize(nll, [0, 0], method="BFGS")
b.check("IRLS = 直接の最尤推定", np.allclose(opt.x, bb, atol=1e-4), (opt.x, bb))
b.num("切片", bb[0], "-2.948")
b.num("傾き", bb[1], "0.842")
b.num("オッズ比", math.exp(bb[1]), "2.32")
b.num("OR 下限", math.exp(bb[1] - 1.96 * se[1]), "1.72")
b.num("OR 上限", math.exp(bb[1] + 1.96 * se[1]), "3.14")
b.num("ED50", -bb[0] / bb[1], "3.50")
b.num("SE", se[1], "0.154")
b.num("ワルド", bb[1] / se[1], "5.46")
ll = lambda q: (y * np.log(q) + (n - y) * np.log(1 - q)).sum()
sat = y / n
D = 2 * (ll(sat) - ll(p))
D0 = 2 * (ll(sat) - ll(np.full(6, y.sum() / n.sum())))
b.num("逸脱度", D, "0.59")
b.num("切片だけ", D0, "43.57")
b.num("差", D0 - D, "42.97")
b.check("スコア X'(y−μ) = 0", np.allclose(X.T @ (y - n * p), 0, atol=1e-8), "")
pr = op.minimize(lambda c: -(y * st.norm.logcdf(c[0] + c[1] * x) + (n - y) * st.norm.logsf(c[0] + c[1] * x)).sum(), [-2, 0.5])
b.num("プロビットの傾き", pr.x[1], "0.502")
b.num("比", bb[1] / pr.x[1], "1.68")
b.num("π/√3", math.pi / math.sqrt(3), "1.81")
# オッズ
for q, s in [(0.2, "0.25"), (0.5, "1"), (0.8, "4")]:
    b.num("オッズ", q / (1 - q), s)
# ポアソン回帰
cnt = np.array([12, 18, 9, 25, 30.])
t = np.array([10, 12, 8, 15, 16.])
Xp = np.column_stack([np.ones(5), [0, 0, 0, 1, 1.]])
bp = np.zeros(2)
for _ in range(100):
    mu = t * np.exp(Xp @ bp)
    bp = bp + np.linalg.solve(Xp.T @ (mu[:, None] * Xp), Xp.T @ (cnt - mu))
mu = t * np.exp(Xp @ bp)
sep = np.sqrt(np.diag(np.linalg.inv(Xp.T @ (mu[:, None] * Xp))))
b.num("一般道の率", math.exp(bp[0]), "1.30")
b.num("係数", bp[0], "0.262")
b.num("率の比の係数", bp[1], "0.311")
b.num("率の比", math.exp(bp[1]), "1.36")
b.num("SE", sep[1], "0.209")
b.num("率の比 下限", math.exp(bp[1] - 1.96 * sep[1]), "0.91")
b.num("率の比 上限", math.exp(bp[1] + 1.96 * sep[1]), "2.06")
b.check("率の比 = 単純な率の比（55/31）/（39/30）", abs(math.exp(bp[1]) - (55 / 31) / (39 / 30)) < 1e-9, "")
Dp = 2 * np.sum(cnt * np.log(cnt / mu) - (cnt - mu))
b.num("ポアソンの逸脱度", Dp, "0.82")
# 章末問題
b.num("問1 e^0.69", math.exp(0.69), "1.99")
o = 0.1 / 0.9 * math.exp(0.69)
b.num("問1 オッズ（本文の丸めた値 0.111×1.99）", 0.111 * 1.99, "0.221")
b.num("問1 確率", o / (1 + o), "0.181")
b.num("問2 ロジット", -2.948 + 0.842 * 4, "0.420")
b.num("問2 確率", 1 / (1 + math.exp(-0.420)), "0.603")
b.num("問3 差", 61.8 - 52.4, "9.4")
b.done()
