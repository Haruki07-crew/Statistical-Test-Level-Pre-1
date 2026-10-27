"""統計学の教科書 第53章「統計的学習理論」の検算（組合せ・粉砕の全探索・シミュレーション）。"""
import itertools
import math
from math import comb

import numpy as np
from bookcheck import Book

b = Book("tb", 53)
rng = np.random.default_rng(53)
b.num("有限個の上界", math.sqrt(math.log(2 * 1000 / 0.05) / 2000), "0.073")
# 直線による分類のラベル付けの数（一般の位置の 20 点、全探索の代わりに公式 2Σ_{i≤2}C(n−1,i) と小さい n の全探索）
def line_labelings(pts):
    """平面の点に直線（切片つき線形分類器）で実現できるラベルを、多数の方向で数える。"""
    labs = set()
    for th in np.linspace(0, 2 * np.pi, 20000, endpoint=False):
        w = np.array([math.cos(th), math.sin(th)])
        proj = pts @ w
        o = np.sort(proj)
        cuts = np.concatenate([[o[0] - 1], (o[:-1] + o[1:]) / 2, [o[-1] + 1]])
        for c in cuts:
            labs.add(tuple((proj > c).astype(int)))
    return len(labs)
pts = rng.uniform(size=(6, 2))
b.check("6 点の全探索 = 2(1+5+10)=32", line_labelings(pts) == 2 * (1 + 5 + 10), line_labelings(pts))
b.check("3 点は粉砕できる", line_labelings(rng.uniform(size=(3, 2))) == 8, "")
b.num("n=20 の付け方", 2 * (1 + 19 + comb(19, 2)), "382")
b.num("ザウアーの上界", sum(comb(20, i) for i in range(4)), "1351")
b.num("2^20", 2 ** 20 / 1e4, "105", tol=0.5)
# 例題
b.num("上界 n=100", math.sqrt(math.log(2 * 101 / 0.05) / 200), "0.204")
ex = {100: [], 1000: []}
dev = []
for n in (100, 1000):
    for _ in range(4000):
        x = rng.uniform(size=n)
        y = (x > 0.3).astype(int)
        y = np.where(rng.uniform(size=n) < 0.1, 1 - y, y)
        o = np.argsort(x); xs = x[o]; ys = y[o]
        c1 = np.concatenate([[0], np.cumsum(ys)])
        c0 = np.concatenate([[0], np.cumsum(1 - ys)])
        err = (c1 + (c0[-1] - c0)) / n
        ts = np.concatenate([[0], xs])
        k = np.argmin(err)
        ex[n].append(0.8 * abs(ts[k] - 0.3))
        if n == 100:
            dev.append(np.max(abs(err - (0.1 + 0.8 * abs(ts - 0.3)))))
b.check("一様なずれの平均 ≈ 0.086", abs(np.mean(dev) - 0.086) < 0.004, np.mean(dev))
b.check("95% 点 ≈ 0.134", abs(np.quantile(dev, 0.95) - 0.134) < 0.006, np.quantile(dev, 0.95))
b.check("超過リスク n=100 ≈ 0.011", abs(np.mean(ex[100]) - 0.011) < 0.0015, np.mean(ex[100]))
b.check("超過リスク n=1000 ≈ 0.0011", abs(np.mean(ex[1000]) - 0.0011) < 0.00015, np.mean(ex[1000]))
# R(t) の式を数値積分で
t = 0.55
xs = np.linspace(0, 1, 2000001)
R = np.mean(np.where((xs > t) == (xs > 0.3), 0.1, 0.9))
b.check("R(t)=0.1+0.8|t−0.3|", abs(R - (0.1 + 0.8 * 0.25)) < 1e-5, R)
# ラデマッハ複雑度（線形、‖w‖≤1、‖x‖≤1）: E‖Σσx‖/n ≤ 1/√n
n, k = 100, 50
X = rng.standard_normal((n, k)); X /= np.linalg.norm(X, axis=1, keepdims=True)
sig = rng.choice([-1, 1], (5000, n))
rad = np.mean(np.linalg.norm(sig @ X, axis=1)) / n
b.check("ラデマッハ複雑度 ≤ BC/√n", rad <= 1 / math.sqrt(n), rad)
# 章末問題
b.num("問1", math.log(4e7) / 0.005, "3500.9")
def shatter_interval(m):
    p = np.arange(m)
    ok = set()
    for a in range(-1, m + 1):
        for c in range(a, m + 1):
            ok.add(tuple(((p >= a) & (p <= c)).astype(int)))
        ok.add(tuple([0] * m))
    return len(ok) == 2 ** m
b.check("問2 区間は 2 点を粉砕し 3 点は粉砕しない", shatter_interval(2) and not shatter_interval(3), "")
b.num("問3", sum(comb(10, i) for i in range(4)), "176")
b.num("問3 比", 176 / 1024 * 100, "17", tol=0.5)
b.done()
