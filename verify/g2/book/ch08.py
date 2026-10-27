"""2級 教科書 第8章「確率変数と期待値」の数値の検算。"""
import itertools
from fractions import Fraction as F

import numpy as np
from bookcheck import Book

b = Book("g2", 8)
die = np.arange(1, 7)
b.num("一様分布の区間確率", (5 - 3) / 10, "0.2")
b.check("F(4)=4/6", F(4, 6) == F(sum(1 for v in die if v <= 4), 6), "")
b.check("さいころの目の和 21/6 = 3.5", F(21, 6) == F(7, 2), "")
b.num("ゲームの期待値（もらえる額）", 100 * 3.5, "350")
b.num("ゲームの損得の期待値", 100 * 3.5 - 300, "50")
b.check("E[X²] = 91/6", F(int((die ** 2).sum()), 6) == F(91, 6), "")
b.check("V = 91/6 − 49/4 = 35/12", F(91, 6) - F(49, 4) == F(35, 12), "")
b.num("分散", 35 / 12, "2.917")
b.num("標準偏差", np.sqrt(35 / 12), "1.708")
# 一様分布の分散（数値積分）
xs = np.linspace(0, 10, 200001)
f = np.full_like(xs, 0.1)
EX = np.trapezoid(xs * f, xs)
VX = np.trapezoid((xs - EX) ** 2 * f, xs)
b.num("一様分布の期待値", EX, "5", tol=1e-6)
b.num("一様分布の分散", VX, "8.33")
b.num("ゲームの分散", 100 ** 2 * 35 / 12, "29167")
b.num("ゲームの標準偏差", 100 * np.sqrt(35 / 12), "171")
# 同時分布
J = np.array([[.30, .10], [.20, .15], [.10, .15]])
X, Y = np.array([1, 2, 3]), np.array([0, 1])
b.check("図2の表が本文と一致", np.allclose(b.chart(2)["table"], J), "")
px, py = J.sum(1), J.sum(0)
b.check("周辺 X 0.40・0.35・0.25", np.allclose(px, [.4, .35, .25]), px)
b.check("周辺 Y 0.60・0.40", np.allclose(py, [.6, .4]), py)
b.check("合計 1", abs(J.sum() - 1) < 1e-12, J.sum())
cond = J[:, 1] / py[1]
b.check("条件付き 0.25・0.375・0.375", np.allclose(cond, [.25, .375, .375]), cond)
b.num("独立なら p(1,0)", px[0] * py[0], "0.24")
EX2, EY2 = (X * px).sum(), (Y * py).sum()
EXY = (np.outer(X, Y) * J).sum()
b.num("E[X]", EX2, "1.85")
b.num("E[Y]", EY2, "0.40")
b.num("E[XY]", EXY, "0.85")
b.num("共分散", EXY - EX2 * EY2, "0.11")
# さいころ 2 個の和
S = [a + c for a, c in itertools.product(die, die)]
pm = {s: F(S.count(s), 36) for s in range(2, 13)}
b.check("図1の確率が計算と一致", np.allclose(b.chart(1)["values"], [round(float(pm[s]), 4) for s in range(2, 13)], atol=1e-9), "")
ES = sum(s * p for s, p in pm.items())
VS = sum(s * s * p for s, p in pm.items()) - ES ** 2
b.check("和の期待値 7", ES == 7, ES)
b.check("和の分散 35/6", VS == F(35, 6), VS)
b.num("35/6", 35 / 6, "5.83")
# 分散投資
VR = 0.25 * 100 + 0.25 * 36 + 2 * 0.25 * (-0.5 * 10 * 6)
b.num("E[R]", 0.5 * 5 + 0.5 * 3, "4")
b.num("V[R]", VR, "19")
b.num("sd[R]", np.sqrt(VR), "4.36")
b.num("独立のときの sd", np.sqrt(25 + 9), "5.83")
rng = np.random.default_rng(8)
Z = rng.multivariate_normal([5, 3], [[100, -30], [-30, 36]], size=400000)
b.check("シミュレーションでも sd ≈ 4.36", abs((0.5 * Z[:, 0] + 0.5 * Z[:, 1]).std() - 4.36) < 0.03, (0.5 * Z[:, 0] + 0.5 * Z[:, 1]).std())
# 章末問題
v, p = np.array([0, 1, 2, 3]), np.array([.1, .3, .4, .2])
e1 = (v * p).sum()
e2 = (v * v * p).sum()
b.num("問1 E[X]", e1, "1.7")
b.num("問1 E[X²]", e2, "3.7")
b.num("問1 V[X]", e2 - e1 ** 2, "0.81")
b.num("問1 E[Y]", 2 * e1 - 1, "2.4")
b.num("問1 V[Y]", 4 * (e2 - e1 ** 2), "3.24")
b.num("問2 E[D]", 50.5 - 50.0, "0.5")
b.num("問2 V[D]", 0.3 ** 2 + 0.4 ** 2, "0.25")
b.num("問2 sd[D]", np.sqrt(0.25), "0.5")
J3 = np.array([[.12, .28], [.18, .42]])
b.check("問3 独立", np.allclose(J3, np.outer(J3.sum(1), J3.sum(0))), np.outer(J3.sum(1), J3.sum(0)))
b.done()
