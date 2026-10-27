"""統計学の教科書 第57章「生存時間解析の理論」の検算（NA・KM・ログランク・部分尤度・競合リスク）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("tb", 57)
rng = np.random.default_rng(57)
t = np.array([5, 8, 12, 15, 20, 2, 3, 4, 6, 9.])
d = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
x = np.array([0] * 5 + [1] * 5, float)


def na_km(t, d, upto):
    A = S = V = 0.0
    S = 1.0
    for tj in sorted(set(t[d == 1])):
        if tj > upto:
            break
        n = (t >= tj).sum(); dj = ((t == tj) & (d == 1)).sum()
        A += dj / n; V += dj / n ** 2; S *= 1 - dj / n
    return A, V, S


A, V, S = na_km(t, d, 9)
b.num("NA(9)", A, "0.771")
b.num("分散", V, "0.133")
b.num("標準誤差", math.sqrt(V), "0.365")
b.num("KM(9)", S, "0.429")
b.num("exp(−NA)", math.exp(-A), "0.463")
b.check("時刻 5 のリスク集合は 7 人", (t >= 5).sum() == 7, "")
# ログランク
U = Var = E = 0.0
for tj in sorted(set(t[d == 1])):
    R = t >= tj; n = R.sum(); n1 = (R & (x == 1)).sum()
    dj = ((t == tj) & (d == 1)).sum(); d1 = ((t == tj) & (d == 1) & (x == 1)).sum()
    U += d1 - n1 * dj / n; E += n1 * dj / n
    Var += n1 * (n - n1) * dj * (n - dj) / (n * n * (n - 1))
b.num("期待イベント数", E, "1.813")
b.num("U", U, "2.187")
b.num("分散", Var, "1.111")
b.num("χ²", U * U / Var, "4.30")
b.num("p", st.chi2.sf(U * U / Var, 1), "0.038")


# コックスの部分尤度
def score(beta):
    s = I = 0.0
    for i in np.where(d == 1)[0]:
        R = t >= t[i]
        w = np.exp(beta * x[R])
        xb = (w * x[R]).sum() / w.sum(); x2 = (w * x[R] ** 2).sum() / w.sum()
        s += x[i] - xb; I += x2 - xb ** 2
    return s, I


s0, I0 = score(0)
b.check("スコア U(0) = ログランクの U", abs(s0 - U) < 1e-12, s0)
b.check("情報量 = ログランクの分散（タイなし）", abs(I0 - Var) < 1e-12, I0)
beta = 0.0
for _ in range(50):
    s, I = score(beta); beta += s / I
b.num("β̂", beta, "2.02")
b.num("ハザード比", math.exp(beta), "7.6")
b.num("標準誤差", 1 / math.sqrt(score(beta)[1]), "1.13")
# マルチンゲール：N − 補償子 の平均が 0（指数分布＋独立な打ち切りのシミュレーション）
lam = 0.2
T = rng.exponential(1 / lam, 400000); C = rng.exponential(10, 400000)
obs = np.minimum(T, C); ev = T <= C
tt = 3.0
M = (ev & (obs <= tt)).astype(float) - lam * np.minimum(obs, tt)
b.check("M(t) の平均 ≈ 0、分散 ≈ 補償子の平均", abs(M.mean()) < 0.003 and abs(M.var() - (lam * np.minimum(obs, tt)).mean()) < 0.005, (M.mean(), M.var()))
# NA の分散公式の確認（シミュレーション）
est = []
for _ in range(3000):
    Ti = rng.exponential(5, 200); Ci = rng.uniform(0, 15, 200)
    o = np.minimum(Ti, Ci); e = (Ti <= Ci).astype(int)
    est.append(na_km(o, e, 3)[0])
b.check("NA(3) は Λ(3)=0.6 に近い", abs(np.mean(est) - 0.6) < 0.01, np.mean(est))
# 競合リスク
F1 = lambda s: 0.1 / 0.3 * (1 - math.exp(-0.3 * s))
b.num("F1(20)", F1(20), "0.333")
b.num("1−KM", 1 - math.exp(-2), "0.865")
T1 = rng.exponential(10, 500000); T2 = rng.exponential(5, 500000)
b.check("シミュレーション F1(20)", abs(np.mean((T1 < T2) & (T1 <= 20)) - F1(20)) < 0.003, "")
# 章末問題
t2 = np.array([1, 2, 3, 3, 5, 7.]); d2 = np.array([1, 0, 1, 1, 1, 0])
b.num("問1", na_km(t2, d2, 5)[0], "1.167")
b.num("問2 期待", 6 * 2 / 10, "1.2")
b.num("問2 分散", 6 * 4 * 2 * 8 / (100 * 9), "0.427")
b.num("問3", 0.05 / 0.2, "0.25")
b.done()
