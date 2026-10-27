"""2級 教科書 第10章「標本分布」の数値の検算（分布表の値は scipy、近似はシミュレーションと正確な値で確かめる）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("g2", 10)
rng = np.random.default_rng(10)
b.num("SE n=25", 10 / np.sqrt(25), "2")
b.num("SE n=100", 10 / np.sqrt(100), "1")
x = rng.exponential(3, (200000, 25))
b.check("V[X̄]=σ²/n（シミュレーション、指数分布 σ=3、n=25）", abs(x.mean(1).var() - 9 / 25) < 0.005, x.mean(1).var())
b.num("正規の ±2σ 外", 2 * st.norm.sf(2), "0.046")
b.num("正規の ±3σ 外", 2 * st.norm.sf(3), "0.003")
# CLT 例題
b.num("SE", 3 / np.sqrt(36), "0.5")
b.num("P(Z>2)", st.norm.sf(2), "0.0228")
a = (4 / 3) ** 2
g = rng.gamma(a, 4 / a, (200000, 36)).mean(1)
b.check("ガンマ母集団の平均 4・sd 3", abs(st.gamma(a, scale=4 / a).mean() - 4) < 1e-12 and abs(st.gamma(a, scale=4 / a).std() - 3) < 1e-12, "")
b.num("シミュレーション P(X̄>5) %", np.mean(g > 5) * 100, "2.8", tol=0.15)
b.check("正確な分布（ガンマの和）でも約 2.8%", abs(st.gamma(36 * a, scale=4 / a / 36).sf(5) - 0.028) < 0.0015, st.gamma(36 * a, scale=4 / a / 36).sf(5))
# 硬貨
b.num("sd", np.sqrt(100 * 0.25), "5")
b.num("z", (59.5 - 50) / 5, "1.9")
b.num("P(Z≥1.9)", st.norm.sf(1.9), "0.0287")
b.num("正確な値", st.binom(100, 0.5).sf(59), "0.0284")
b.num("連続修正なし", st.norm.sf(2), "0.0228")
b.num("約 0.029", st.norm.sf(1.9), "0.029")
se = np.sqrt(0.25 / 1000)
b.num("世論調査の SE", se, "0.0158")
b.num("下限 %", (0.5 - 1.96 * se) * 100, "46.9")
b.num("上限 %", (0.5 + 1.96 * se) * 100, "53.1")
p = rng.binomial(1000, 0.5, 200000) / 1000
b.check("シミュレーション：95% 程度が範囲内", abs(np.mean((p >= 0.469) & (p <= 0.531)) - 0.95) < 0.01, np.mean((p >= 0.469) & (p <= 0.531)))
# カイ二乗・t・F
w = (rng.normal(size=(200000, 5)) ** 2).sum(1)
b.check("χ²(5)：平均 5・分散 10（シミュレーション）", abs(w.mean() - 5) < 0.03 and abs(w.var() - 10) < 0.15, (w.mean(), w.var()))
s2 = rng.normal(10, 2, (200000, 10)).var(1, ddof=1)
b.check("(n−1)S²/σ² ～ χ²(9)（シミュレーション）", abs(np.mean(9 * s2 / 4 > st.chi2.ppf(0.95, 9)) - 0.05) < 0.003, np.mean(9 * s2 / 4 > st.chi2.ppf(0.95, 9)))
for dfv, s in [(5, "2.571"), (9, "2.262"), (15, "2.131"), (20, "2.086"), (30, "2.042")]:
    b.num(f"t_0.025({dfv})", st.t.ppf(0.975, dfv), s)
b.num("z_0.05", st.norm.ppf(0.95), "1.645")
b.num("z_0.025", st.norm.ppf(0.975), "1.960")
b.num("z_0.005", st.norm.ppf(0.995), "2.576")
b.num("χ²_0.975(9)", st.chi2.ppf(0.025, 9), "2.700")
b.num("χ²_0.05(9)", st.chi2.ppf(0.95, 9), "16.92")
b.num("χ²_0.025(9)", st.chi2.ppf(0.975, 9), "19.02")
b.num("F_0.05(3,10)", st.f.ppf(0.95, 3, 10), "3.708")
b.check("F の下側の関係", abs(st.f.ppf(0.05, 3, 10) - 1 / st.f.ppf(0.95, 10, 3)) < 1e-12, "")
# 章末問題
b.num("問1", 12 / 4, "3")
b.num("問1 n", 12 ** 2, "144")
b.num("問2 SE", 0.6, "0.6")
b.num("問2 z", 1 / 0.6, "1.67")
b.num("問2 表の値", st.norm.sf(1.67), "0.0475")
b.num("問3 sd", np.sqrt(400 * 0.1 * 0.9), "6")
b.num("問3 z", (30.5 - 40) / 6, "-1.58")
b.num("問3 表の値", st.norm.cdf(-1.58), "0.0571")
b.check("問3 正確な値も近い", abs(st.binom(400, 0.1).cdf(30) - 0.0571) < 0.01, st.binom(400, 0.1).cdf(30))
b.done()
