"""準1級 教科書 第14章「マルコフ連鎖」の検算（有理数の厳密計算・行列計算・シミュレーション）。"""
import numpy as np
import sympy as sp
from bookcheck import Book

b = Book("pre1", 14)
R = sp.Rational
P = sp.Matrix([[R(6, 10), R(3, 10), R(1, 10)], [R(3, 10), R(4, 10), R(3, 10)], [R(2, 10), R(4, 10), R(4, 10)]])
b.check("図の推移確率行列が本文と一致", b.chart(1)["P"] == [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]] and b.chart(1)["start"] == 2, "")
P2 = P ** 2
b.check("P² の値", P2 == sp.Matrix([[R(47, 100), R(34, 100), R(19, 100)], [R(36, 100), R(37, 100), R(27, 100)], [R(32, 100), R(38, 100), R(30, 100)]]), P2)
Pn = np.array(P, dtype=float)
p0 = np.array([0, 0, 1.])
for n, shown in [(1, ["0.2", "0.4", "0.4"]), (2, ["0.32", "0.38", "0.30"]), (5, ["0.390", "0.362", "0.249"]), (10, ["0.393", "0.361", "0.246"])]:
    v = p0 @ np.linalg.matrix_power(Pn, n)
    for x, s in zip(v, shown):
        b.num(f"{n} 日後", x, s)
a, c, d = sp.symbols("a c d")
sol = sp.solve([0.6 * a + 0.3 * c + 0.2 * d - a, 0.3 * a + 0.4 * c + 0.4 * d - c, a + c + d - 1], [a, c, d])
pi = [R(24, 61), R(22, 61), R(15, 61)]
b.check("定常分布 24/61・22/61・15/61", sp.Matrix([pi]) * P == sp.Matrix([pi]), "")
for x, s in zip(pi, ["0.393", "0.361", "0.246"]):
    b.num("定常分布", float(x), s)
b.num("雨の平均再帰時間", 61 / 15, "4.07")
rng = np.random.default_rng(14)
st_, cnt, ret, last = 2, np.zeros(3), [], 0
for t in range(1, 400001):
    st_ = rng.choice(3, p=Pn[st_])
    cnt[st_] += 1
    if st_ == 2:
        ret.append(t - last)
        last = t
b.check("時間平均 ≈ 定常分布（エルゴード性）", np.allclose(cnt / cnt.sum(), [float(x) for x in pi], atol=0.005), cnt / cnt.sum())
b.check("平均再帰時間 ≈ 4.07（シミュレーション）", abs(np.mean(ret[1:]) - 61 / 15) < 0.05, np.mean(ret[1:]))
b.num("晴→曇の流れ", float(pi[0]) * 0.3, "0.118")
b.num("曇→晴の流れ", float(pi[1]) * 0.3, "0.108")
# 破産問題
Q = sp.Matrix([[0, R(2, 5), 0], [R(3, 5), 0, R(2, 5)], [0, R(3, 5), 0]])
Rm = sp.Matrix([[R(3, 5), 0], [0, 0], [0, R(2, 5)]])
N = (sp.eye(3) - Q).inv()
h = N * Rm
m = N * sp.Matrix([1, 1, 1])
b.check("破産の確率 9/13", h[1, 0] == R(9, 13), h[1, 0])
b.check("回数の期待値 50/13", m[1] == R(50, 13), m[1])
b.num("破産の確率", 9 / 13, "0.692")
b.num("回数の期待値", 50 / 13, "3.85")
# 連続時間
b.num("稼働率", 0.5 / 0.52, "0.962")
b.num("50/52", 50 / 52, "0.962")
# 章末問題
b.num("問1 π1", 0.3 / 0.5, "0.6")
b.num("問1 π2", 0.2 / 0.5, "0.4")
b.num("問2", 1 / 0.4, "2.5")
Q3 = sp.Matrix([[0, R(1, 2)], [R(1, 2), 0]])
N3 = (sp.eye(2) - Q3).inv()
b.check("問3 到達確率 1/3・回数 2", (N3 * sp.Matrix([[R(1, 2), 0], [0, R(1, 2)]]))[0, 1] == R(1, 3) and (N3 * sp.Matrix([1, 1]))[0] == 2, "")
b.done()
