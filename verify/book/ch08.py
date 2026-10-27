"""準1級 教科書 第8章「統計的推定の基礎」の検算（sympy・正確な分布・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 8)
rng = np.random.default_rng(8)
data = np.array([2.1, 0.4, 3.7, 1.2, 0.8, 5.3, 1.9, 0.6, 2.8, 1.2])
b.check("本文のデータ", b.has("2.1、0.4、3.7、1.2、0.8、5.3、1.9、0.6、2.8、1.2"), "")
b.num("合計", data.sum(), "20.0")
b.num("平均", data.mean(), "2.0")
b.num("λ̂", 1 / data.mean(), "0.5")
b.num("標準誤差", 0.5 / math.sqrt(10), "0.158")
# MSE（σ⁴ 単位、χ²(n−1) の平均 n−1・分散 2(n−1) から）
n = sp.symbols("n", positive=True)
for d, expr in [(n - 1, 2 / (n - 1)), (n, (2 * n - 1) / n ** 2), (n + 1, 2 / (n + 1))]:
    mse = 2 * (n - 1) / d ** 2 + ((n - 1) / d - 1) ** 2
    b.check(f"MSE（d={d}）", sp.simplify(mse - expr) == 0, sp.simplify(mse))
for v, s in [(2 / 9, "0.222"), (19 / 100, "0.190"), (2 / 11, "0.182")]:
    b.num("n=10 の MSE", v, s)
x = rng.normal(0, 1, (300000, 10))
S = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
b.check("シミュレーションでも d=n+1 が最小", ((S / 11 - 1) ** 2).mean() < ((S / 10 - 1) ** 2).mean() < ((S / 9 - 1) ** 2).mean(), "")
# 情報量
th, xx = sp.symbols("theta x", positive=True)
for name, logf, I in [("ベルヌーイ", xx * sp.log(th) + (1 - xx) * sp.log(1 - th), None), ("ポアソン", xx * sp.log(th) - th, 1 / th), ("指数", sp.log(th) - th * xx, 1 / th ** 2)]:
    d2 = sp.diff(logf, th, 2)
    if name == "ベルヌーイ":
        Ival = sp.simplify(-(d2.subs(xx, 1) * th + d2.subs(xx, 0) * (1 - th)))
        b.check("ベルヌーイ I = 1/(p(1−p))", sp.simplify(Ival - 1 / (th * (1 - th))) == 0, Ival)
    elif name == "ポアソン":
        b.check("ポアソン I = 1/λ（E[X]=λ を代入）", sp.simplify(-d2.subs(xx, th) - I) == 0, "")
    else:
        b.check("指数 I = 1/λ²", sp.simplify(-d2 - I) == 0, "")
# 一様分布
u = rng.uniform(0, 5, (300000, 10))
b.num("(n+1)/n 最大値の分散", 25 / (10 * 12), "0.208")
b.num("2x̄ の分散", 25 / 30, "0.833")
b.check("シミュレーション", abs((1.1 * u.max(1)).var() - 0.2083) < 0.003 and abs((2 * u.mean(1)).var() - 0.8333) < 0.006, "")
# ラオ・ブラックウェル
lam, nn = 1.0, 10
b.num("素朴な推定量の分散", math.exp(-1) * (1 - math.exp(-1)), "0.233")
vrb = math.exp(-2 * lam) * (math.exp(lam / nn) - 1)
b.num("改良した推定量の分散", vrb, "0.0142")
b.num("CR 下限", math.exp(-2 * lam) * lam / nn, "0.0135")
y = rng.poisson(lam, (400000, nn))
rb = ((nn - 1) / nn) ** y.sum(1)
b.check("改良した推定量は不偏・分散一致（シミュレーション）", abs(rb.mean() - math.exp(-1)) < 0.001 and abs(rb.var() - vrb) < 0.0005, (rb.mean(), rb.var()))
b.check("16 分の 1 以下", vrb < math.exp(-1) * (1 - math.exp(-1)) / 16, "")
pmf = [st.binom(7, 0.1).pmf(0)]
b.check("T=t のもとで X1 ～ Bin(t,1/n)（t=7 の P(X1=0)）", abs(pmf[0] - 0.9 ** 7) < 1e-12, "")
# 最尤推定量の漸近正規性（指数）
e = rng.exponential(2, (200000, 100))
lh = 1 / e.mean(1)
b.check("√n(λ̂−λ) の分散 ≈ λ²（λ=0.5、n=100）", abs(100 * lh.var() - 0.25) < 0.01, 100 * lh.var())
# 順序統計量
b.check("一様の k 番目は Beta(k, n−k+1)", abs(np.mean(np.sort(rng.uniform(size=(100000, 6)), 1)[:, 1] <= 0.3) - st.beta(2, 5).cdf(0.3)) < 0.004, "")
mn = rng.exponential(1, (200000, 5)).min(1)
b.check("指数の最小値は平均 1/n", abs(mn.mean() - 0.2) < 0.002, mn.mean())
# 章末問題
b.num("問1 SE", math.sqrt(0.21 / 100), "0.0458")
b.num("問4 MSE(B)", 2 + 1, "3")
b.done()
