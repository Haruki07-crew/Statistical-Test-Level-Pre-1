"""準1級 教科書 第29章「不完全データの統計処理」の検算（直接計算・数値最適化・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import optimize as op, stats as st

b = Book("pre1", 29)
q = np.array([10.2, 10.8, 9.9, 10.5, 10.6])
u = np.array([0.40, 0.42, 0.38, 0.41, 0.39])
b.num("q̄", q.mean(), "10.4")
b.num("Ū", u.mean(), "0.40")
b.num("B", q.var(ddof=1), "0.125")
T = u.mean() + 1.2 * q.var(ddof=1)
b.num("T", T, "0.55")
b.num("SE", math.sqrt(T), "0.742")
b.num("1 組の SE", math.sqrt(0.40), "0.63")
# EM
y = np.array([125, 18, 20, 34])
th = 0.5
hist = []
for _ in range(50):
    x = y[0] * (th / 4) / (0.5 + th / 4)
    th = (x + y[3]) / (x + y[1] + y[2] + y[3])
    hist.append(th)
for v, s in zip(hist[:3], ["0.608", "0.624", "0.626"]):
    b.num("EM の反復", v, s)
b.num("収束値", hist[-1], "0.627")
ll = lambda t: -(y[0] * np.log(0.5 + t / 4) + (y[1] + y[2]) * np.log((1 - t) / 4) + y[3] * np.log(t / 4))
b.check("直接の最尤推定と一致", abs(op.minimize_scalar(ll, bounds=(0.01, 0.99), method="bounded").x - hist[-1]) < 1e-5, "")
b.check("尤度は減らない", all(ll(hist[i + 1]) <= ll(hist[i]) + 1e-12 for i in range(20)), "")
# KM
t = np.array([3, 5, 5, 8, 10, 12, 15, 18, 20, 24.])
ev = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
S, var, H = 1, 0, 0
rows = {}
for ti in np.unique(t):
    d = ((t == ti) & (ev == 1)).sum()
    r = (t >= ti).sum()
    if d > 0:
        S *= 1 - d / r
        var += d / (r * (r - d))
        H += d / r
        rows[ti] = (r, d, S, S * math.sqrt(var), H)
for ti, n_, s_ in [(3, 10, "0.900"), (5, 9, "0.800"), (8, 7, "0.686"), (12, 5, "0.549"), (15, 4, "0.411"), (20, 2, "0.206")]:
    b.check(f"リスク集合 t={ti}", rows[ti][0] == n_, rows[ti][0])
    b.num(f"KM t={ti}", rows[ti][2], s_)
b.num("グリーンウッド SE t=12", rows[12][3], "0.172")
b.num("NA H(20)", rows[20][4], "1.304")
b.num("e^{-H}", math.exp(-rows[20][4]), "0.271")
# ログランク
tA = np.array([4, 7, 9, 12, 15, 20.])
eA = np.array([1, 1, 0, 1, 1, 0])
tB = np.array([2, 3, 5, 6, 8, 10.])
eB = np.array([1, 1, 1, 0, 1, 1])
tt = np.concatenate([tA, tB])
ee = np.concatenate([eA, eB])
g = np.r_[np.zeros(6), np.ones(6)]
O = E = V = 0
for ti in np.unique(tt[ee == 1]):
    r = tt >= ti
    n_ = r.sum()
    n1 = (r & (g == 0)).sum()
    d = ((tt == ti) & (ee == 1)).sum()
    O += ((tt == ti) & (ee == 1) & (g == 0)).sum()
    E += d * n1 / n_
    V += d * (n1 / n_) * (1 - n1 / n_) * (n_ - d) / max(n_ - 1, 1)
b.num("O", O, "4")
b.num("E", E, "6.33")
b.num("V", V, "1.60")
b.num("χ²", (O - E) ** 2 / V, "3.40")
b.num("p", st.chi2.sf((O - E) ** 2 / V, 1), "0.065")
# 章末問題
b.num("問2 t=2", 0.8, "0.8")
b.num("問2 t=6", 0.8 * (1 - 2 / 3), "0.267")
b.num("問3 T", 1 + 4 / 3, "2.33")
b.done()
