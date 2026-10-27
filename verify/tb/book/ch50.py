"""統計学の教科書 第50章「線形混合モデルと相関のあるデータ」の検算（厳密計算・数値積分・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("tb", 50)
rng = np.random.default_rng(50)
# 50.1 デザイン効果
de = 1 + 19 * 0.1
b.num("デザイン効果", de, "2.9")
b.num("実質的な標本の大きさ", 600 / de, "207")
b.num("標準誤差の比", math.sqrt(de), "1.7")
b.num("独立とみなした標準誤差", math.sqrt(1 / 600), "0.041")
se = math.sqrt((0.1 + 0.9 / 20) / 30)
b.num("本当の標準誤差", se, "0.070")
a, m = 30, 20
Y = rng.normal(0, math.sqrt(0.1), (40000, a, 1)) + rng.normal(0, math.sqrt(0.9), (40000, a, m))
b.check("シミュレーションの全体平均の標準偏差", abs(Y.mean((1, 2)).std() - se) < 0.002, Y.mean((1, 2)).std())
# 共分散行列で厳密に
V = 0.9 * np.eye(m) + 0.1 * np.ones((m, m))
b.check("1校の平均の分散 = (ρ + (1−ρ)/m)", abs(np.ones(m) @ V @ np.ones(m) / m ** 2 - (0.1 + 0.9 / 20)) < 1e-14, "")
# 50.2 BLUP
lam = lambda t2, s2, mi: t2 / (t2 + s2 / mi)
b.num("λ（9人）", lam(4, 36, 9), "0.5")
b.num("予測 +3", lam(4, 36, 9) * 6, "3")
b.num("λ（36人）", lam(4, 36, 36), "0.8")
# BLUP = 条件付き期待値 E[u | ȳ]（正規）を数値で
u = rng.normal(0, 2, 400000)
yb = u + rng.normal(0, 6 / 3, 400000)
b.check("E[u|ȳ] の傾き = λ（シミュレーション）", abs(np.cov(u, yb)[0, 1] / yb.var() - 0.5) < 0.01, "")
# 50.3 ML と REML
a, m, t2, s2 = 10, 10, 1.0, 4.0
b.num("E[MS_B]", s2 + m * t2, "14")
b.num("最尤の平均（近似）", ((a - 1) / a * 14 - 4) / m, "0.86")
ml, rm = [], []
for _ in range(20000):
    y = rng.normal(0, 1, (a, 1)) + rng.normal(0, 2, (a, m))
    g = y.mean(1)
    msb = m * g.var(ddof=1)
    msw = ((y - g[:, None]) ** 2).sum() / (a * (m - 1))
    ml.append(max(((a - 1) / a * msb - msw) / m, 0))
    rm.append(max((msb - msw) / m, 0))
b.num("最尤のシミュレーション平均", np.mean(ml), "0.857", tol=0.01)
b.check("REML はほぼ不偏", abs(np.mean(rm) - 1) < 0.02, np.mean(rm))
b.check("14% 小さい", abs(1 - 0.86 - 0.14) < 1e-12 and b.has("14%"), "")
# 最尤推定量の式を直接の尤度最大化で確認
y = rng.normal(0, 1, (a, 1)) + rng.normal(0, 2, (a, m))
g = y.mean(1); msb = m * g.var(ddof=1); msw = ((y - g[:, None]) ** 2).sum() / (a * (m - 1))
from scipy import optimize
def nll(p):
    t2, s2 = math.exp(p[0]), math.exp(p[1])
    Vm = s2 * np.eye(m) + t2 * np.ones((m, m))
    mu = y.mean()
    sign, ld = np.linalg.slogdet(Vm)
    Vi = np.linalg.inv(Vm)
    r = y - mu
    return a * ld + np.einsum("ij,jk,ik->", r, Vi, r)
res = optimize.minimize(nll, [0, 1], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
b.check("最尤推定量の式 ((a−1)/a·MSB − MSW)/m", abs(math.exp(res.x[0]) - ((a - 1) / a * msb - msw) / m) < 1e-4, (math.exp(res.x[0]), ((a - 1) / a * msb - msw) / m))
# 50.5 GLMM の周辺効果
tau2 = 2
marg = lambda x: integrate.quad(lambda v: st.norm.pdf(v, 0, math.sqrt(tau2)) / (1 + math.exp(-(x + v))), -30, 30)[0]
lg = lambda p: math.log(p / (1 - p))
b.num("近似", 1 / math.sqrt(1 + 0.346 * tau2), "0.77")
b.num("数値積分", lg(marg(1)) - lg(marg(0)), "0.73")
b.check("0.346 = (16√3/(15π))²", abs((16 * math.sqrt(3) / (15 * math.pi)) ** 2 - 0.346) < 0.0005, "")
# 章末問題
b.num("問1 デザイン効果", 1 + 3 * 0.3, "1.9")
b.num("問1 実質", 200 / 1.9, "105")
b.num("問2 3個", lam(2, 18, 3), "0.25")
b.num("問2 18個", lam(2, 18, 18), "0.667")
b.num("問3 REML", (30 - 6) / 4, "6")
b.num("問3 最尤", (0.8 * 30 - 6) / 4, "4.5")
b.done()
