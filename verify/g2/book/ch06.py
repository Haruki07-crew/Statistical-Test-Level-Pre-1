"""2級 教科書 第6章「データの集め方」の数値の検算（層化抽出のシミュレーションを含む）。"""
import numpy as np
from bookcheck import Book

b = Book("g2", 6)
b.num("抽出間隔 4000/200", 4000 / 200, "20")
sel = [13 + 20 * j for j in range(200)]
b.check("13 から始めると最後は 3993", sel[-1] == 3993 and len(sel) == 200, sel[-1])
W = np.array([2000, 1200, 800]) / 4000
b.check("層の割合 0.5・0.3・0.2", np.allclose(W, [0.5, 0.3, 0.2]), W)
for v, s in zip(200 * W, ["100", "60", "40"]):
    b.num("比例配分", v, s)
b.num("層化の推定値", (W * [4.0, 6.0, 3.0]).sum(), "4.4")

# シミュレーション：層ごとの平均・標準偏差が本文どおりの 4000 人の母集団を作り、200 人の標本を 2 万回
rng = np.random.default_rng(6)


def mk(n, m, s):
    v = rng.normal(size=n)
    v = (v - v.mean()) / v.std()
    return m + s * v


pop = [mk(2000, 4.0, 1.0), mk(1200, 6.0, 1.2), mk(800, 3.0, 0.8)]
allp = np.concatenate(pop)
b.num("母集団の平均", allp.mean(), "4.4")
R = 20000
srs, st = np.empty(R), np.empty(R)
for r in range(R):
    srs[r] = rng.choice(allp, 200, replace=False).mean()
    st[r] = (rng.choice(pop[0], 100, replace=False).sum() + rng.choice(pop[1], 60, replace=False).sum() + rng.choice(pop[2], 40, replace=False).sum()) / 200
b.num("単純無作為抽出のぶれ", srs.std(), "0.10", tol=0.006)
b.num("層化抽出のぶれ", st.std(), "0.07", tol=0.006)
b.num("0.2 以上外れる割合（単純）%", np.mean(np.abs(srs - 4.4) >= 0.2) * 100, "5.9", tol=0.4)
b.num("0.2 以上外れる割合（層化）%", np.mean(np.abs(st - 4.4) >= 0.2) * 100, "0.5", tol=0.2)
# 理論値でも確認（有限母集団修正つきの標準誤差）
f = 200 / 4000
S2 = allp.var(ddof=1)
Sw = sum(w * p.var(ddof=1) for w, p in zip(W, pop))
b.check("理論の標準誤差（単純）≈ 0.10", abs(np.sqrt((1 - f) * S2 / 200) - 0.10) < 0.006, np.sqrt((1 - f) * S2 / 200))
b.check("理論の標準誤差（層化）≈ 0.07", abs(np.sqrt((1 - f) * Sw / 200) - 0.07) < 0.006, np.sqrt((1 - f) * Sw / 200))

# 無回答
b.num("全体の平均", 4.0 * 0.6 + 6.5 * 0.4, "5.0")
b.num("ずれ", 5.0 - 4.0, "1.0")
# 章末問題
b.num("問2 間隔", 6000 / 300, "20")
b.num("問2 5 番目", 7 + 20 * 4, "87")
b.num("問2 最後", 7 + 20 * 299, "5987")
W3 = np.array([18000, 9000, 3000]) / 30000
for v, s in zip(600 * W3, ["360", "180", "60"]):
    b.num("問3 配分", v, s)
b.num("問3 推定値", (W3 * [25, 28, 22]).sum(), "25.6")
b.done()
