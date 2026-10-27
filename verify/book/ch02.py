"""準1級 教科書 第2章「確率分布と母関数」の検算（sympy の記号計算と数値積分・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("pre1", 2)
x, y, t, s = sp.symbols("x y t s", real=True)
R = sp.Rational
f = 6 * x * (1 - x)
b.check("c = 6", sp.integrate(x * (1 - x), (x, 0, 1)) == R(1, 6), "")
b.check("F(x) = 3x² − 2x³", sp.expand(sp.integrate(f, (x, 0, x)) - (3 * x ** 2 - 2 * x ** 3)) == 0, "")
b.check("P(1/4≤X≤3/4) = 11/16", sp.integrate(f, (x, R(1, 4), R(3, 4))) == R(11, 16), "")
b.check("式の途中 (27/16−54/64)−(3/16−2/64) = 11/16", (R(27, 16) - R(54, 64)) - (R(3, 16) - R(2, 64)) == R(11, 16), "")
EX, EX2 = sp.integrate(x * f, (x, 0, 1)), sp.integrate(x ** 2 * f, (x, 0, 1))
b.check("E=1/2, E[X²]=3/10, V=1/20", (EX, EX2, EX2 - EX ** 2) == (R(1, 2), R(3, 10), R(1, 20)), (EX, EX2))
b.check("Beta(2,2) と一致", abs(st.beta(2, 2).var() - 0.05) < 1e-12, "")
# ワイブル
for tt, shown in [(5, "0.779"), (10, "0.368"), (20, "0.018")]:
    b.num(f"S({tt})", math.exp(-(tt / 10) ** 2), shown)
b.num("h(5)", 2 * 5 / 100, "0.1")
b.num("h(10)", 2 * 10 / 100, "0.2")
w = st.weibull_min(2, scale=10)
b.check("scipy のワイブルで S と h", abs(w.sf(5) - math.exp(-0.25)) < 1e-12 and abs(w.pdf(5) / w.sf(5) - 0.1) < 1e-12, "")
# 同時密度 x+y
g = x + y
fx = sp.integrate(g, (y, 0, 1))
b.check("f_X = x + 1/2", sp.simplify(fx - (x + R(1, 2))) == 0, fx)
ce = sp.simplify(sp.integrate(y * g, (y, 0, 1)) / fx)
b.check("E[Y|X=x] = (3x+2)/(3(2x+1))", sp.simplify(ce - (3 * x + 2) / (3 * (2 * x + 1))) == 0, ce)
b.check("x=0 で 2/3、x=1 で 5/9", ce.subs(x, 0) == R(2, 3) and ce.subs(x, 1) == R(5, 9), "")
b.check("P(X+Y≤1) = 1/3", sp.integrate(g, (y, 0, 1 - x), (x, 0, 1)) == R(1, 3), "")
exy = sp.integrate(x * y * g, (x, 0, 1), (y, 0, 1))
ex = sp.integrate(x * fx, (x, 0, 1))
b.check("E[XY]=1/3, E[X]=7/12, Cov=−1/144", exy == R(1, 3) and ex == R(7, 12) and exy - ex ** 2 == R(-1, 144), (exy, ex))
# MGF
p, n, lam, mu, sig = sp.symbols("p n lambda mu sigma", positive=True)
M = (1 - p + p * sp.exp(t)) ** n
m1 = sp.diff(M, t).subs(t, 0)
m2 = sp.diff(M, t, 2).subs(t, 0)
b.check("二項：np・np(1−p)", sp.simplify(m1 - n * p) == 0 and sp.simplify(m2 - m1 ** 2 - n * p * (1 - p)) == 0, "")
E_ = lam / (lam - t)
b.check("指数：1/λ, 2/λ², 6/λ³", [sp.simplify(sp.diff(E_, t, k).subs(t, 0)) for k in (1, 2, 3)] == [1 / lam, 2 / lam ** 2, 6 / lam ** 3], "")
b.check("ポアソンの MGF（数値）", abs(sum(math.exp(0.3 * k) * st.poisson(2).pmf(k) for k in range(80)) - math.exp(2 * (math.exp(0.3) - 1))) < 1e-10, "")
b.check("正規の MGF（数値積分）", abs(integrate.quad(lambda u: math.exp(0.5 * u) * st.norm(1, 2).pdf(u), -60, 60)[0] - math.exp(0.5 + 4 * 0.25 / 2)) < 1e-8, "")
# 確率母関数の合成
GY = sp.exp(3 * (s - 1)).subs(s, R(3, 5) + R(2, 5) * s)
b.check("合成で Po(1.2)", sp.simplify(GY - sp.exp(R(6, 5) * (s - 1))) == 0, GY)
rng = np.random.default_rng(2)
N = rng.poisson(3, 400000)
Y = rng.binomial(N, 0.4)
b.check("間引き（シミュレーション）：平均と分散 ≈ 1.2", abs(Y.mean() - 1.2) < 0.01 and abs(Y.var() - 1.2) < 0.02, (Y.mean(), Y.var()))
# コーシー
c = rng.standard_cauchy((100000, 50)).mean(1)
b.check("コーシーの平均もコーシー（四分位数 ±1）", abs(np.quantile(c, 0.75) - 1) < 0.03 and abs(np.quantile(c, 0.25) + 1) < 0.03, np.quantile(c, [0.25, 0.75]))
b.check("標準正規の特性関数 e^{-t²/2}（数値積分）", abs(integrate.quad(lambda u: math.cos(1.3 * u) * st.norm.pdf(u), -40, 40)[0] - math.exp(-1.3 ** 2 / 2)) < 1e-10, "")
# 章末問題
F1 = 1 - (1 + x) * sp.exp(-x)
b.check("問1 密度 x e^{-x}", sp.simplify(sp.diff(F1, x) - x * sp.exp(-x)) == 0, "")
b.num("問1 P(X>2)", 3 * math.exp(-2), "0.406")
b.check("問1 ハザード x/(1+x)", sp.simplify(x * sp.exp(-x) / (1 - F1) - x / (1 + x)) == 0, "")
M2 = (1 / (1 - 2 * t)) ** 3
b.check("問2 E=6, E[X²]=48, V=12", sp.diff(M2, t).subs(t, 0) == 6 and sp.diff(M2, t, 2).subs(t, 0) == 48, "")
G3 = (1 + s + s ** 2) / 3
b.check("問3 E=1, G''(1)=2/3, V=2/3", sp.diff(G3, s).subs(s, 1) == 1 and sp.diff(G3, s, 2).subs(s, 1) == R(2, 3), "")
b.check("問4 f_X = 2(1−x), f_Y = 2y", sp.integrate(2, (y, x, 1)) == 2 - 2 * x and sp.integrate(2, (x, 0, y)) == 2 * y, "")
b.done()
