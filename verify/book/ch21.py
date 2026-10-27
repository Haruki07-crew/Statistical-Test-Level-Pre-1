"""準1級 教科書 第21章「標本調査法」の検算（総当たり・式の直接計算・シミュレーション）。"""
import itertools

import numpy as np
from bookcheck import Book

b = Book("pre1", 21)
P = {(0, 1): 0.5, (0, 2): 0.3, (1, 2): 0.2}
y = np.array([8, 10, 6.])
pi = np.zeros(3)
for s, p in P.items():
    for i in s:
        pi[i] += p
b.check("包含確率 0.8・0.7・0.5", np.allclose(pi, [0.8, 0.7, 0.5]), pi)
ht = {s: sum(y[i] / pi[i] for i in s) for s in P}
for s, v in [((0, 1), "24.29"), ((0, 2), "22"), ((1, 2), "26.29")]:
    b.num(f"HT {s}", ht[s], v)
b.num("期待値", sum(P[s] * ht[s] for s in P), "24")
b.num("HT の分散", sum(P[s] * (ht[s] - 24) ** 2 for s in P), "2.29")
srs = [3 * np.mean(y[list(s)]) for s in itertools.combinations(range(3), 2)]
b.num("単純無作為の分散", np.var(srs), "6")
# 単純無作為抽出の分散（シミュレーション）
rng = np.random.default_rng(21)
pop = rng.normal(50, 10, 500)
S2 = pop.var(ddof=1)
means = np.array([rng.choice(pop, 100, replace=False).mean() for _ in range(40000)])
b.check("V[ȳ] = (1−n/N)S²/n（シミュレーション）", abs(means.var() - 0.8 * S2 / 100) / (0.8 * S2 / 100) < 0.03, (means.var(), 0.8 * S2 / 100))
# 層化
W = np.array([0.6, 0.3, 0.1])
S = np.array([10, 30, 60.])
Yb = np.array([50, 80, 120.])
n = 200
gm = (W * Yb).sum()
b.num("全体平均", gm, "66")
b.num("層内分散の平均", (W * S ** 2).sum(), "690")
b.num("層平均の分散", (W * (Yb - gm) ** 2).sum(), "504")
S2t = (W * S ** 2).sum() + (W * (Yb - gm) ** 2).sum()
b.num("全体の分散", S2t, "1194")
b.num("SRS", S2t / n, "5.97")
b.num("比例", (W * S ** 2).sum() / n, "3.45")
b.num("ネイマン", (W * S).sum() ** 2 / n, "2.21")
na = n * W * S / (W * S).sum()
for v, s in zip(na, ["57", "86", "57"]):
    b.num("ネイマン配分", v, s, tol=0.5)
# ネイマン配分が最小（数値最適化で確認）
best = min(((W ** 2 * S ** 2 / np.array([a, c, n - a - c])).sum(), a, c) for a in range(1, 199) for c in range(1, 200 - a))
b.check("整数配分の最適は 57・86・57 付近", abs(best[1] - 57) <= 1 and abs(best[2] - 86) <= 1, best)
# 計画効果
b.num("deff", 1 + 19 * 0.05, "1.95")
b.num("有効標本", 400 / 1.95, "205")
# 比推定
x = np.array([12, 8, 15, 10, 5.])
yy = np.array([30, 22, 41, 27, 13.])
b.num("x̄", x.mean(), "10")
b.num("ȳ", yy.mean(), "26.6")
b.num("比", yy.mean() / x.mean(), "2.66")
b.num("比推定", yy.mean() / x.mean() * 9.4, "25.0")
# 章末問題
b.num("問1 なし", np.sqrt(25 / 200), "0.354")
b.num("問1 つき", np.sqrt(0.8 * 25 / 200), "0.316")
b.num("問2 n1", 100 * 20000 / 60000, "33", tol=0.5)
b.num("問2 n2", 100 * 40000 / 60000, "67", tol=0.5)
b.num("問3 deff", 1 + 2 * 0.4, "1.8")
b.num("問3 有効", 900 / 1.8, "500")
b.done()
