"""準1級 教科書 第4章「変数変換」の検算（sympy・scipy・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("pre1", 4)
rng = np.random.default_rng(4)
u = rng.uniform(size=10 ** 6)
b.check("−log U は平均 1 の指数分布", abs((-np.log(u)).mean() - 1) < 0.005 and abs(np.mean(-np.log(u) > 2) - math.exp(-2)) < 0.002, "")
b.num("対数正規の平均（μ=0,σ=1）", math.exp(0.5), "1.65")
b.check("scipy の対数正規の平均", abs(st.lognorm(1).mean() - math.exp(0.5)) < 1e-12, "")
yy = np.linspace(0.05, 5, 50)
b.check("Z² の密度 = χ²(1)", np.allclose(np.exp(-yy / 2) / np.sqrt(2 * np.pi * yy), st.chi2(1).pdf(yy)), "")
b.num("P(Z²≤1)", 2 * st.norm.cdf(1) - 1, "0.683")
# 和と割合
x_, y_ = rng.exponential(size=10 ** 6), rng.exponential(size=10 ** 6)
U, V = x_ + y_, x_ / (x_ + y_)
b.check("U ～ Ga(2,1)", abs(np.mean(U <= 2) - st.gamma(2).cdf(2)) < 0.002, "")
b.check("V ～ U(0,1)", abs(np.mean(V <= 0.3) - 0.3) < 0.002, "")
b.check("U と V は独立（相関 ≈ 0、条件付きでも一様）", abs(np.corrcoef(U, V)[0, 1]) < 0.005 and abs(np.mean(V[U > 3] <= 0.3) - 0.3) < 0.005, "")
uu, vv = sp.symbols("u v", positive=True)
J = sp.Matrix([[vv, uu], [1 - vv, -uu]]).det()
b.check("ヤコビアン = −u", sp.simplify(J + uu) == 0, J)
# 三角分布
c = b.chart(1)
b.check("図の三角分布の値", np.allclose(c["series"][0]["y"], [min(z, 2 - z) for z in c["x"]]), "")
s = rng.uniform(size=10 ** 6) + rng.uniform(size=10 ** 6)
b.num("P(Z≤1.5)", 1 - 0.5 * 0.5 ** 2, "0.875")
b.check("シミュレーション", abs(np.mean(s <= 1.5) - 0.875) < 0.002, np.mean(s <= 1.5))
# 再生性
g = rng.gamma(2, 3, 10 ** 6) + rng.gamma(3, 3, 10 ** 6)
b.check("Ga(2,3)+Ga(3,3) ≈ Ga(5,3)", abs(np.mean(g <= 15) - st.gamma(5, scale=3).cdf(15)) < 0.002, "")
b.check("χ²(k) = Ga(k/2, 2)", np.allclose(st.chi2(7).pdf(yy), st.gamma(3.5, scale=2).pdf(yy)), "")
# 平方根変換
for lam, shown in [(4, "0.306"), (25, "0.253"), (100, "0.251")]:
    b.num(f"√X の分散（平均 {lam}）", np.sqrt(rng.poisson(lam, 2 * 10 ** 6)).var(), shown, tol=0.003)
yv = sp.symbols("y", positive=True)
lam_ = sp.symbols("lambda")
b.check("Box-Cox は λ→0 で log y", sp.limit((yv ** lam_ - 1) / lam_, lam_, 0) == sp.log(yv), "")
# ロジット・プロビット
b.num("logit(0.8)", math.log(4), "1.386")
b.num("probit(0.8)", st.norm.ppf(0.8), "0.842")
b.num("比（0.8）", math.log(4) / st.norm.ppf(0.8), "1.65")
b.num("比（0.9）", math.log(9) / st.norm.ppf(0.9), "1.71")
# 章末問題
yy2 = sp.symbols("y", positive=True)
b.check("問1 E[Y] = 1/3", sp.integrate(yy2 / (2 * sp.sqrt(yy2)), (yy2, 0, 1)) == sp.Rational(1, 3), "")
b.num("問2 期待値", 5 * 3, "15")
b.num("問2 分散", 5 * 9, "45")
X2, Y2 = rng.normal(size=10 ** 6), rng.normal(size=10 ** 6)
b.check("問3 U,V は N(0,2) で無相関", abs((X2 + Y2).var() - 2) < 0.01 and abs(np.corrcoef(X2 + Y2, X2 - Y2)[0, 1]) < 0.005, "")
b.num("問4 logit(0.2)", math.log(0.25), "-1.386")
b.num("問4 オッズ比", (0.5 / 0.5) / (0.2 / 0.8), "4")
b.done()
