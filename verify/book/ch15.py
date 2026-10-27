"""準1級 教科書 第15章「確率過程の基礎」の検算（正確な式とシミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 15)
rng = np.random.default_rng(15)
b.num("2 時間で 4 件", st.poisson(6).pmf(4), "0.134")
b.num("2 時間で 10 件以上", st.poisson(6).sf(9), "0.084")
b.num("30 分以上空く", math.exp(-1.5), "0.223")
# ポアソン過程のシミュレーション
gaps = rng.exponential(1 / 3, (100000, 40))
times = gaps.cumsum(1)
N2 = (times <= 2).sum(1)
b.check("間隔が指数なら 2 時間の件数は Po(6)", abs(np.mean(N2 == 4) - st.poisson(6).pmf(4)) < 0.004, np.mean(N2 == 4))
keep = rng.uniform(size=times.shape) < 1 / 3
Nb = ((times <= 2) & keep).sum(1)
b.check("間引き：Po(2)", abs(Nb.mean() - 2) < 0.02 and abs(Nb.var() - 2) < 0.04, (Nb.mean(), Nb.var()))
b.num("複合ポアソンの平均", 3 * 2, "6")
b.num("複合ポアソンの分散", 3 * (1 + 4), "15")
# ランダムウォーク
b.num("100 歩の |S| の平均", math.sqrt(200 / math.pi), "8.0")
S = rng.choice([-1, 1], (100000, 100)).sum(1)
b.check("シミュレーション", abs(np.abs(S).mean() - 7.96) < 0.1, np.abs(S).mean())
q, p = 0.55, 0.45
r = q / p
pr = (1 - r ** 10) / (1 - r ** 20)
b.num("p=0.45 の到達確率", pr, "0.119")
b.num("p=0.45 の平均回数", 10 / (q - p) - 20 / (q - p) * pr, "76", tol=0.5)
b.num("公平な賭けの回数", 10 * 10, "100")
# 反射原理
b.num("最大値が 1 以上", 2 * st.norm.sf(1), "0.317")
b.num("終点が 1 以上", st.norm.sf(1), "0.159")
n = 2000
W = rng.normal(0, math.sqrt(1 / n), (20000, n)).cumsum(1)
b.check("反射原理（シミュレーション、離散化で少し小さめ）", 0.29 < np.mean(W.max(1) >= 1) < 0.32, np.mean(W.max(1) >= 1))
# M/M/1
lam, mu = 4, 5
rho = lam / mu
L = rho / (1 - rho)
b.num("ρ", rho, "0.8")
b.num("L", L, "4")
b.num("W", L / lam, "1")
b.num("Wq", L / lam - 1 / mu, "0.8")
b.num("Wq 分", (L / lam - 1 / mu) * 60, "48")
# M/M/1 のシミュレーション（リンドレーの式で待ち時間）
A = rng.exponential(1 / lam, 400000)
Sv = rng.exponential(1 / mu, 400000)
wq = np.zeros(400000)
for i in range(1, 400000):
    wq[i] = max(0.0, wq[i - 1] + Sv[i - 1] - A[i])
b.check("待ち時間の平均 ≈ 0.8（シミュレーション）", abs(wq[20000:].mean() - 0.8) < 0.04, wq[20000:].mean())
# M/M/2
c, m2 = 2, 2.5
a = lam / m2
rho2 = a / c
p0 = 1 / (sum(a ** k / math.factorial(k) for k in range(c)) + a ** c / math.factorial(c) / (1 - rho2))
Lq = p0 * a ** c * rho2 / (math.factorial(c) * (1 - rho2) ** 2)
b.num("M/M/2 の Wq", Lq / lam, "0.71")
b.num("M/M/2 の W", Lq / lam + 1 / m2, "1.11")
# 章末問題
b.num("問1(1)", math.exp(-1), "0.368")
b.num("問1(2) 平均", 3 / 2, "1.5")
b.num("問1(2) 分散", 3 / 4, "0.75")
b.num("問2", (1 - 0.5 ** 2) / (1 - 0.5 ** 4), "0.8")
b.num("問3 L", 0.9 / 0.1, "9")
b.num("問3 L'", 0.95 / 0.05, "19")
b.num("問3 W'", 19 / 9.5, "2")
b.num("問3 増加率 %", (9.5 / 9 - 1) * 100, "5.6")
b.done()
