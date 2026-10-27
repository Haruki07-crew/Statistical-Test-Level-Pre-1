"""準1級 教科書 第27章「時系列解析」の検算（理論式・シミュレーション・特性方程式の数値解）。"""
import numpy as np
from bookcheck import Book

b = Book("pre1", 27)
rng = np.random.default_rng(27)
e = rng.normal(size=300000)
x = np.zeros_like(e)
for t in range(1, len(e)):
    x[t] = 0.7 * x[t - 1] + e[t]
b.num("AR(1) 分散", 1 / (1 - 0.49), "1.96")
b.check("シミュレーションの分散・自己相関", abs(x.var() - 1.96) < 0.03 and abs(np.corrcoef(x[:-1], x[1:])[0, 1] - 0.7) < 0.005 and abs(np.corrcoef(x[:-2], x[2:])[0, 1] - 0.49) < 0.006, "")
z = np.roots([0.5, -1.2, 1])
b.num("解の実部", z[0].real, "1.2")
b.num("解の虚部", abs(z[0].imag), "0.748")
b.num("|z|", abs(z[0]), "1.41")
b.num("周期", 2 * np.pi / abs(np.angle(z[0])), "11", tol=0.5)
rho = [1, 1.2 / 1.5]
for h in range(2, 9):
    rho.append(1.2 * rho[-1] - 0.5 * rho[-2])
b.check("図の自己相関が理論値と一致", np.allclose(b.chart(1)["values"], np.round(rho, 3)), np.round(rho, 3))
r1, r2 = 0.6, 0.5
p1 = r1 * (1 - r2) / (1 - r1 ** 2)
p2 = (r2 - r1 ** 2) / (1 - r1 ** 2)
b.num("φ1", p1, "0.469")
b.num("φ2", p2, "0.219")
b.check("ユール・ウォーカーを満たす", abs(p1 + p2 * r1 - r1) < 1e-12 and abs(p1 * r1 + p2 - r2) < 1e-12, "")
# AR(2) の PACF はラグ 3 で 0（シミュレーション）
y = np.zeros_like(e)
for t in range(2, len(e)):
    y[t] = 1.2 * y[t - 1] - 0.5 * y[t - 2] + e[t]
X = np.column_stack([y[2:-1], y[1:-2], y[:-3]])
coef = np.linalg.lstsq(X, y[3:], rcond=None)[0]
b.check("AR(2) の PACF はラグ 3 でほぼ 0", abs(coef[2]) < 0.01, coef)
b.num("MA(1) θ=0.5", 0.5 / 1.25, "0.4")
b.num("MA(1) θ=2", 2 / 5, "0.4")
# 予測
b.num("1 期先", 0.7 * 2, "1.4")
b.num("2 期先", 0.49 * 2, "0.98")
b.num("2 期先の誤差分散", 1 + 0.49, "1.49")
b.num("1 期先 下限", 1.4 - 1.96, "-0.56")
b.num("1 期先 上限", 1.4 + 1.96, "3.36")
b.num("2 期先 下限", 0.98 - 1.96 * np.sqrt(1.49), "-1.41")
b.num("2 期先 上限", 0.98 + 1.96 * np.sqrt(1.49), "3.37")
# 章末問題
b.num("問1 分散", 3 / 0.75, "4")
b.num("問1 ρ(3)", 0.5 ** 3, "0.125")
b.check("問2 解は 1 と −2", np.allclose(sorted(np.roots([-0.5, -0.5, 1]).real), [-2, 1]), np.roots([-0.5, -0.5, 1]))
th = np.roots([0.3, -1, 0.3])
b.check("問3 θ = 1/3 と 3", np.allclose(sorted(th), [1 / 3, 3]), th)
b.done()
