"""準1級 教科書 第33章「統計的因果推論」の検算（調整の公式とシミュレーション）。"""
import numpy as np
from bookcheck import Book

b = Book("pre1", 33)
pz = {"y": .6, "o": .4}
pt = {"y": .3, "o": .7}
py = {("y", 1): .9, ("y", 0): .8, ("o", 1): .6, ("o", 0): .5}
pT = sum(pz[z] * pt[z] for z in pz)
n1 = sum(pz[z] * pt[z] * py[(z, 1)] for z in pz) / pT
n0 = sum(pz[z] * (1 - pt[z]) * py[(z, 0)] for z in pz) / (1 - pT)
b.num("治療の割合", pT, "0.46")
b.num("治療ありの回復率", n1, "0.717")
b.num("治療なしの回復率", n0, "0.733")
b.num("単純な差", n1 - n0, "-0.016")
b.num("E[Y(1)]", sum(pz[z] * py[(z, 1)] for z in pz), "0.78")
b.num("E[Y(0)]", sum(pz[z] * py[(z, 0)] for z in pz), "0.68")
b.num("ATE", 0.78 - 0.68, "0.10")
# シミュレーション：潜在結果から ATE を直接計算
rng = np.random.default_rng(33)
N = 1000000
old = rng.uniform(size=N) < 0.4
u = rng.uniform(size=N)
Y1 = u < np.where(old, 0.6, 0.9)
Y0 = u < np.where(old, 0.5, 0.8)
T = rng.uniform(size=N) < np.where(old, 0.7, 0.3)
Y = np.where(T, Y1, Y0)
b.check("潜在結果の差の平均 ≈ 0.10", abs((Y1.astype(float) - Y0).mean() - 0.10) < 0.002, (Y1.astype(float) - Y0).mean())
b.check("観測の単純な差 ≈ −0.016", abs(Y[T].mean() - Y[~T].mean() + 0.016) < 0.003, Y[T].mean() - Y[~T].mean())
Tr = rng.uniform(size=N) < 0.5
Yr = np.where(Tr, Y1, Y0)
b.check("無作為化すれば単純な差 ≈ ATE", abs(Yr[Tr].mean() - Yr[~Tr].mean() - 0.10) < 0.003, "")
# 合流点の選択の偏り
ta, lu = rng.normal(size=N), rng.normal(size=N)
succ = ta + lu > 1.5
b.check("合流点で選ぶと負の相関", np.corrcoef(ta[succ], lu[succ])[0, 1] < -0.3 and abs(np.corrcoef(ta, lu)[0, 1]) < 0.01, np.corrcoef(ta[succ], lu[succ])[0, 1])
# フロントドア
px = {1: .5, 0: .5}
pm1 = {1: .8, 0: .2}
pyy = {(1, 1): .7, (1, 0): .5, (0, 1): .4, (0, 0): .2}


def do(x):
    return sum((pm1[x] if m else 1 - pm1[x]) * sum(pyy[(m, xp)] * px[xp] for xp in (0, 1)) for m in (0, 1))


b.num("do(1)", do(1), "0.54")
b.num("do(0)", do(0), "0.36")
b.num("効果", do(1) - do(0), "0.18")
naive = lambda x: sum((pm1[x] if m else 1 - pm1[x]) * pyy[(m, x)] for m in (0, 1))
b.num("喫煙者", naive(1), "0.64")
b.num("非喫煙者", naive(0), "0.26")
b.num("単純な差", naive(1) - naive(0), "0.38")
# 章末問題
b.num("問1 E[Y(1)]", 0.5 * 0.7 + 0.5 * 0.4, "0.55")
b.num("問1 E[Y(0)]", 0.5 * 0.5 + 0.5 * 0.2, "0.35")
b.num("問1 ATE", 0.20, "0.20")
b.done()
