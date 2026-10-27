"""準1級 教科書 第3章「分布の特性値」の検算（sympy・数値計算・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 3)
x, lam = sp.symbols("x lambda", positive=True)
R = sp.Rational
f = lam * sp.exp(-lam * x)
m = [sp.simplify(sp.integrate(x ** k * f, (x, 0, sp.oo))) for k in range(1, 5)]
b.check("E[X^k] = k!/λ^k", m == [1 / lam, 2 / lam ** 2, 6 / lam ** 3, 24 / lam ** 4], m)
mu = m[0]
c2 = sp.simplify(m[1] - mu ** 2)
c3 = sp.simplify(m[2] - 3 * mu * m[1] + 2 * mu ** 3)
c4 = sp.simplify(m[3] - 4 * mu * m[2] + 6 * mu ** 2 * m[1] - 3 * mu ** 4)
b.check("μ2=1/λ², μ3=2/λ³, μ4=9/λ⁴", (c2, c3, c4) == (lam ** -2, 2 / lam ** 3, 9 / lam ** 4), (c2, c3, c4))
b.check("歪度 2・尖度 9（scipy の超過尖度 6）", st.expon.stats(moments="s") == 2 and st.expon.stats(moments="k") == 6, "")
b.check("非負の E[X] = ∫S", sp.integrate(sp.exp(-lam * x), (x, 0, sp.oo)) == 1 / lam, "")
b.num("log 2", math.log(2), "0.693")
b.num("中央値 λ=0.5", math.log(2) / 0.5, "1.39")
b.num("90% 点 λ=0.5", -math.log(0.1) / 0.5, "4.61")
rng = np.random.default_rng(3)
u = rng.uniform(size=10 ** 6)
b.check("逆関数法（指数分布の平均 2）", abs((-np.log(1 - u) / 0.5).mean() - 2) < 0.01, "")
z = rng.normal(size=10 ** 6)
b.check("Cov[X, X²] ≈ 0", abs(np.cov(z, z ** 2)[0, 1]) < 0.01, np.cov(z, z ** 2)[0, 1])
# 偏相関 = 残差の相関
Z = rng.normal(size=10 ** 5)
X = 0.8 * Z + rng.normal(size=10 ** 5)
Y = 0.5 * X + 0.6 * Z + rng.normal(size=10 ** 5)
r = np.corrcoef([X, Y, Z])
pc = (r[0, 1] - r[0, 2] * r[1, 2]) / math.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))
ex = X - np.polyval(np.polyfit(Z, X, 1), Z)
ey = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
b.check("偏相関の式 = 残差の相関", abs(pc - np.corrcoef(ex, ey)[0, 1]) < 1e-10, (pc, np.corrcoef(ex, ey)[0, 1]))
# 階層二項
b.check("E[X(1−X)] = 1/6", sp.integrate(x * (1 - x), (x, 0, 1)) == R(1, 6), "")
b.check("V[Y] = 10/6 + 100/12 = 10", R(10, 6) + R(100, 12) == 10, "")
b.num("100/12", 100 / 12, "8.33")
b.num("Bin(10,0.5) の分散", 2.5, "2.5")
p = [float(sp.integrate(sp.binomial(10, k) * x ** k * (1 - x) ** (10 - k), (x, 0, 1))) for k in range(11)]
b.check("Y は 0〜10 の離散一様分布（各 1/11）", np.allclose(p, 1 / 11), p[:3])
b.check("離散一様の分散 (11²−1)/12 = 10", (11 ** 2 - 1) / 12 == 10, "")
# 分散共分散行列
S = np.array([[4, 1, -1], [1, 9, 2], [-1, 2, 1.]])
a = np.array([0.5, 0.3, 0.2])
b.num("a'Σa", a @ S @ a, "2.19")
A = np.array([[1, 1, 0], [1, -1, 0.]])
b.check("AΣA' = [[15,−5],[−5,11]]", np.allclose(A @ S @ A.T, [[15, -5], [-5, 11]]), A @ S @ A.T)
ev = np.linalg.eigvalsh(S)
for v, s in zip(ev, ["0.146", "4.28", "9.57"]):
    b.num("固有値", v, s)
Xs = rng.multivariate_normal([0, 0, 0], S, size=400000)
b.check("シミュレーションでも a'Σa ≈ 2.19", abs((Xs @ a).var() - 2.19) < 0.02, (Xs @ a).var())
# 章末問題
b.check("問1 μ4 = 1/80、尖度 1.8", sp.integrate(x ** 4, (x, -R(1, 2), R(1, 2))) == R(1, 80) and R(1, 80) / R(1, 12) ** 2 == R(9, 5), "")
b.check("問1 scipy の一様分布の超過尖度 −1.2", abs(st.uniform.stats(moments="k") + 1.2) < 1e-12, "")
VS = 250000 * 50 + 4000000 * 50
b.num("問2 E[S]", 2000 * 50, "100000")
b.num("問2 V[S]", VS, "212500000")
b.num("問2 sd", math.sqrt(VS), "14577")
N = rng.poisson(50, 100000)
Ssim = np.array([rng.normal(2000, 500, n).sum() for n in N[:30000]])
b.check("問2 シミュレーション", abs(Ssim.std() - 14577) < 250, Ssim.std())
b.num("問4 V[2X1−X2]", 4 * 4 + 9 - 2 * 2 * (-3), "37")
b.num("問4 共分散", 4 - 9, "-5")
b.done()
