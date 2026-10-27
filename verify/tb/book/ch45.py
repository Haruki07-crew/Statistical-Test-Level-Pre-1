"""統計学の教科書 第45章「測度論的確率論」の検算（数値積分・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("tb", 45)
rng = np.random.default_rng(45)
b.num("保険の期待値", 0.3 * 10, "3")
b.num("∫S", integrate.quad(lambda x: 0.3 * math.exp(-x / 10), 0, np.inf)[0], "3")
b.num("P(X>5)", 0.3 * math.exp(-0.5), "0.182")
N = 10 ** 6
X = np.where(rng.uniform(size=N) < 0.7, 0, rng.exponential(10, N))
b.check("シミュレーション", abs(X.mean() - 3) < 0.03 and abs(np.mean(X > 5) - 0.182) < 0.002, "")
U = rng.uniform(size=N)
for n in (10, 100, 1000):
    Xn = n * (U < 1 / n)
    b.check(f"E[X_n] ≈ 1（n={n}）", abs(Xn.mean() - 1) < 0.1, Xn.mean())
b.check("X_n → 0（n=1000 でほぼすべて 0）", np.mean(1000 * (U < 1 / 1000) == 0) > 0.998, "")
w = lambda x: math.exp(x - 0.5)
b.check("E_P[dQ/dP] = 1", abs(integrate.quad(lambda x: w(x) * st.norm.pdf(x), -30, 30)[0] - 1) < 1e-10, "")
q = integrate.quad(lambda x: w(x) * st.norm.pdf(x), 2, 40)[0]
b.num("Q(X>2)", q, "0.159")
b.check("= 1−Φ(1)", abs(q - st.norm.sf(1)) < 1e-10, "")
b.num("P(X>2)", st.norm.sf(2), "0.023")
b.num("π²/6", math.pi ** 2 / 6, "1.645")
# ボレル・カンテリ（シミュレーション）
n = np.arange(1, 200001)
cnt2 = [(rng.uniform(size=n.size) < 1 / n ** 2).sum() for _ in range(200)]
b.check("1/n² の事象は少数回（平均 ≈ π²/6）", abs(np.mean(cnt2) - math.pi ** 2 / 6) < 0.25, np.mean(cnt2))
cnt1 = [(rng.uniform(size=n.size) < 1 / n).sum() for _ in range(200)]
b.check("1/n の事象は log n 程度に増え続ける", abs(np.mean(cnt1) - (np.log(200000) + 0.5772)) < 0.6, np.mean(cnt1))
# 章末問題
b.num("問2 E[X]", 0.8 * 2, "1.6")
b.num("問2 E[X²]", 0.8 * 13 / 3, "3.47")
b.check("問2 U(1,3) の E[X²] = 13/3", abs(integrate.quad(lambda x: x * x / 2, 1, 3)[0] - 13 / 3) < 1e-12, "")
b.check("問4 E_P[dQ/dP] = 1", abs(integrate.quad(lambda x: 0.5 * math.exp(-x / 2), 0, np.inf)[0] - 1) < 1e-10, "")
b.done()
