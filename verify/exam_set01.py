"""模擬試験 第1回（set01）の検算。generator/set01.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値最適化）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(1)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1
# (a) 3 つの欠陥の有無の原子確率を同時確率から逆算し、「どれか1つ以上」を足し上げる
pa, pb, pc, pab, pac, pbc, pabc = 0.03, 0.04, 0.02, 0.012, 0.006, 0.008, 0.004
atoms = {(1, 1, 1): pabc, (1, 1, 0): pab - pabc, (1, 0, 1): pac - pabc, (0, 1, 1): pbc - pabc}
atoms[(1, 0, 0)] = pa - atoms[(1, 1, 0)] - atoms[(1, 0, 1)] - pabc
atoms[(0, 1, 0)] = pb - atoms[(1, 1, 0)] - atoms[(0, 1, 1)] - pabc
atoms[(0, 0, 1)] = pc - atoms[(1, 0, 1)] - atoms[(0, 1, 1)] - pabc
E.check("s01p01a", all(v >= 0 for v in atoms.values()), "与えた確率が矛盾しない（原子確率がすべて非負）")
E.num("s01p01a", rnd(sum(atoms.values()), 3), "原子確率の和")
# (b) 総当たりで超幾何の分散
cnt = {}
for c in combinations(range(40), 5):
    x = sum(1 for i in c if i < 4)
    cnt[x] = cnt.get(x, 0) + 1
tot = sum(cnt.values())
mean = sum(k * v for k, v in cnt.items()) / tot
var = sum(k * k * v for k, v in cnt.items()) / tot - mean ** 2
E.choice("s01p01b", f(var, 3), "40C5 通りの総当たりで V[X]")
E.check("s01p01b", abs(stats.hypergeom(40, 4, 5).var() - var) < 1e-12, "scipy の超幾何分布と一致")
# (c) scipy の正規分布で連続修正つき近似
pr = stats.norm.sf(11.5, 8, np.sqrt(400 * 0.02 * 0.98))
E.choice("s01p01c", f"${f(pr, 3)}$", "scipy norm.sf(11.5, 8, 2.8)")
nearest("s01p01c", stats.binom.sf(11, 400, 0.02))  # 正確な二項確率にも最も近いのは正解

# ---- 大問2
s = sp.symbols("s")
G = (sp.Rational(4, 5) / (1 - sp.Rational(1, 5) * s)) ** 2
ser = sp.series(G, s, 0, 80).removeO()
probs = [float(ser.coeff(s, k)) for k in range(80)]
m1 = sum(k * p for k, p in enumerate(probs))
m2 = sum(k * k * p for k, p in enumerate(probs))
E.choice("s01p02a", f(m2 - m1 ** 2, 3), "級数展開で求めた確率から分散")
E.check("s01p02a", abs(stats.nbinom(2, 0.8).var() - (m2 - m1 ** 2)) < 1e-9, "scipy nbinom(2,0.8) の分散と一致")
# (b) 請求額をガンマ(形状4, 尺度5; 平均20 sd10)と置き、複合分布の積率母関数 G(M_X(t)) を微分
t = sp.symbols("t")
MS = G.subs(s, (1 - 5 * t) ** -4)
ES = sp.diff(MS, t).subs(t, 0)
ES2 = sp.diff(MS, t, 2).subs(t, 0)
E.num("s01p02b", rnd(float(sp.sqrt(ES2 - ES ** 2)), 2), "sympy で G(M_X(t)) を2回微分")
Nsim = rng.negative_binomial(2, 0.8, size=400_000)
S = np.array([rng.gamma(4, 5, size=k).sum() if k else 0.0 for k in Nsim])
E.check("s01p02b", abs(S.std() - 17.32) < 0.15, f"シミュレーションの標準偏差 {S.std():.3f} ≈ 17.32")
# (c) 事後平均を数値積分
lik = lambda l: stats.poisson(5 * l).pmf(4) * stats.gamma(2, scale=0.25).pdf(l)
Z = integrate.quad(lik, 0, np.inf)[0]
pm = integrate.quad(lambda l: l * lik(l), 0, np.inf)[0] / Z
E.choice("s01p02c", f"${f(pm, 3)}$", "事前×尤度を数値積分して事後平均")
E.check("s01p02c", abs(stats.nbinom(2, 0.8).pmf(3) - integrate.quad(lambda l: stats.poisson(l).pmf(3) * stats.gamma(2, scale=0.25).pdf(l), 0, np.inf)[0]) < 1e-10,
        "ガンマ混合ポアソンが NB(2,0.8) になること（文脈の主張）")

# ---- 大問3
lam, y = sp.symbols("lambda y", positive=True)
FY = 1 - sp.exp(-lam * y ** 2)  # P(T <= y^2)
fY = sp.diff(FY, y)
E.check("s01p03a", sp.simplify(fY - 2 * lam * y * sp.exp(-lam * y ** 2)) == 0, "分布関数の微分で密度を導出")
E.check("s01p03a", sp.simplify(sp.integrate(fY, (y, 0, sp.oo)) - 1) == 0, "密度の積分が 1")
E.choice("s01p03a", "2\\lambdaye^{-\\lambday^2}", "正解の式を含む選択肢がただ1つ")
for k, wrong in [(0, lam * sp.exp(-lam * y ** 2)), (2, 2 * lam * y * sp.exp(-lam * y))]:
    E.check("s01p03a", sp.simplify(wrong - fY) != 0, f"選択肢 {k} は導出した密度と異なる")
# (b) E[c/ΣY²] = λ となる c を、Y を実際に発生させるモンテカルロと sympy の積分で求める
true_l = 0.4
Ys = np.sqrt(rng.exponential(1 / true_l, size=(400_000, 10)))
c_hat = true_l / np.mean(1 / (Ys ** 2).sum(1))
E.check("s01p03b", abs(c_hat - 9) < 0.06, f"シミュレーションで不偏にする定数 c ≈ {c_hat:.3f}")
w = sp.Symbol("w", positive=True)
Einv = sp.integrate(1 / w * lam ** 10 * w ** 9 * sp.exp(-lam * w) / sp.factorial(9), (w, 0, sp.oo))
c_exact = sp.simplify(lam / Einv)
E.num("s01p03b", rnd(float(c_exact) / 25.6, 3), f"sympy の積分で c = {c_exact}")

# ---- 大問4
r = stats.ttest_ind_from_stats(14.2, 6.0, 12, 8.5, 5.0, 10, equal_var=True)
E.choice("s01p04a", f(r.statistic, 2), "scipy ttest_ind_from_stats(equal_var=True)")
rw = stats.ttest_ind_from_stats(14.2, 6.0, 12, 8.5, 5.0, 10, equal_var=False)
E.check("s01p04a", f(rw.statistic, 2) != f(r.statistic, 2), "ウェルチの値は別の選択肢")
za, zb = 1.645, 1.282
nn = 1
while stats.norm.sf(za - 4 / (6 * np.sqrt(2 / nn))) < 1 - stats.norm.sf(zb) - 1e-12:
    nn += 1
E.num("s01p04b", nn, "検出力関数を n=1,2,… と総当たり（与えた点で）")
nn2 = 1
while stats.norm.sf(stats.norm.isf(0.05) - 4 / (6 * np.sqrt(2 / nn2))) < 0.90:
    nn2 += 1
E.check("s01p04b", nn2 == nn, f"正確なパーセント点でも n={nn2}")

# ---- 大問5
F = (48 / 2) / ((138 - 48) / 15)
E.choice("s01p05a", f"F={f(F, 2)}$であり、$3.68$を超えるので帰無仮説を棄却する", "F 値と判定")
E.check("s01p05a", abs(stats.f.ppf(0.95, 2, 15) - 3.68) < 0.005 and stats.f.sf(F, 2, 15) < 0.05, "scipy F(2,15) の 5% 点と p 値")
E.check("s01p05a", abs(stats.f.ppf(0.95, 15, 2) - 19.43) < 0.01 and abs(stats.f.ppf(0.95, 2, 17) - 3.59) < 0.01, "問題文の他の F 点")
lo = (0.5 / 0.2) * stats.f.ppf(0.025, 15, 10)  # 1/F_{0.025}(10,15) = F(15,10) の下側 2.5% 点
E.num("s01p05b", rnd(lo, 2), "逆向きの自由度の下側 2.5% 点を掛ける形で下限")
E.check("s01p05b", abs(stats.f.ppf(0.975, 10, 15) - 3.06) < 0.005 and abs(stats.f.ppf(0.975, 15, 10) - 3.52) < 0.005
        and abs(stats.f.ppf(0.95, 10, 15) - 2.54) < 0.005, "問題文の F の点")
A = rng.normal(0, 1.0, size=(100_000, 11)).var(axis=1, ddof=1)
B = rng.normal(0, 0.5, size=(100_000, 16)).var(axis=1, ddof=1)
cover = ((A / B) / 3.06 <= 4.0) & (4.0 <= (A / B) * 3.52)
E.check("s01p05b", abs(cover.mean() - 0.95) < 0.004, f"シミュレーションで被覆率 {cover.mean():.4f}")

# ---- 大問6
obs = np.array([38, 82, 54, 26])
pk = stats.poisson(1.35).pmf([0, 1, 2])
pk = np.append(pk, 1 - pk.sum())
ex = np.array([rnd(200 * q, 2) for q in pk])  # 問題文の指示：期待度数は小数第2位まで
chi = ((obs - ex) ** 2 / ex).sum()
E.choice("s01p06a", f"約${f(chi, 1)}$、自由度は2で、$5.991$を超えるので帰無仮説を棄却する", "scipy の poisson で期待度数")
E.check("s01p06a", stats.chi2.sf(chi, 2) < 0.05 < stats.chi2.sf(chi, 3), "df=2 なら棄却、df=3 なら棄却しない")
E.check("s01p06a", abs(np.exp(-1.35) - 0.2592) < 5e-5, "e^{-1.35} の値")
P = rng.normal(0, 1, size=(3, 2))
P = P - P.mean(0) + [2, 4]
Q = rng.normal(0, 1, size=(2, 2))
Q = Q - Q.mean(0) + [5, 8]
ess = lambda X: ((X - X.mean(0)) ** 2).sum()
E.choice("s01p06b", f(ess(np.vstack([P, Q])) - ess(P) - ess(Q), 1), "具体的な点を作って併合前後の群内平方和を直接計算")

# ---- 大問7：行列の累乗とシミュレーション
Pm = np.array([[0.90, 0.08, 0.02], [0, 0.75, 0.25], [1, 0, 0]])
cost = np.linalg.matrix_power(Pm, 500)[0] @ np.array([0, 2, 10])
E.num("s01p07a", rnd(cost, 2), "P^500 の行で長期平均")
cum = Pm.cumsum(1)
state = np.zeros(2000, dtype=int)
acc = 0.0
for _ in range(1000):
    state = (rng.random(2000)[:, None] > cum[state]).sum(1)
    acc += np.array([0, 2, 10])[state].mean()
E.check("s01p07a", abs(acc / 1000 - cost) < 0.03, f"2000 本の連鎖を 1000 日シミュレーション {acc / 1000:.4f}")

# ---- 大問8
n, p = 30, 3
X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
yv = X @ [1, 2, -1, 0.5] + rng.normal(size=n) * 2
bh = np.linalg.lstsq(X, yv, rcond=None)[0]
e = yv - X @ bh
sse, sst = e @ e, ((yv - yv.mean()) ** 2).sum()
F_def = ((sst - sse) / p) / (sse / (n - p - 1))
r2 = 1 - sse / sst
E.check("s01p08a", abs(F_def - (r2 / p) / ((1 - r2) / (n - p - 1))) < 1e-9, "平方和の定義と R² 形式が一致（乱数データ）")
E.choice("s01p08a", f((0.7 / 3) / (0.3 / 26), 2), "R² 形式で F")
E.check("s01p08a", abs(stats.f.ppf(0.95, 3, 26) - 2.98) < 0.005, "F(3,26) の 5% 点")
H = X @ np.linalg.inv(X.T @ X) @ X.T
s2 = sse / (n - p - 1)
i = 5
Xi, yi = np.delete(X, i, 0), np.delete(yv, i)
bi = np.linalg.lstsq(Xi, yi, rcond=None)[0]
cook_def = ((X @ bh - X @ bi) ** 2).sum() / ((p + 1) * s2)
ri = e[i] / np.sqrt(s2 * (1 - H[i, i]))
E.check("s01p08b", abs(cook_def - ri ** 2 / (p + 1) * H[i, i] / (1 - H[i, i])) < 1e-10, "1点除いて再推定した定義と公式が一致（乱数データ）")
E.choice("s01p08b", f"${f(2.0 ** 2 / 4 * 0.4 / 0.6, 3)}$", "公式に h=0.4, r=2 を代入")

# ---- 大問9
lo, hi = np.exp(0.85 - 1.96 * 0.3), np.exp(0.85 + 1.96 * 0.3)
E.choice("s01p09a", f"({f(lo, 2)},\\{f(hi, 2)})", "numpy の exp で区間")
bad = [x_ for x_, v_ in [(0.262, 1.300), (0.357, 1.429), (0.55, 1.733), (0.85, 2.34), (1.15, 3.158), (1.344, 3.834), (1.438, 4.212)]
       if abs(np.exp(x_) - v_) > 6e-4]
E.check("s01p09a", not bad, f"e^x の表の値（誤り {bad}）")
b_, c_ = 25, 10
z = (b_ - (b_ + c_) / 2) / np.sqrt((b_ + c_) / 4)  # 不一致ペアの二項分布の正規近似
mc = z ** 2
E.choice("s01p09b", f"{f(mc, 2)}$で、有意水準5%で有意だが1%", "二項分布の正規近似 z² としてマクネマー統計量")
E.check("s01p09b", stats.chi2.ppf(0.95, 1) < mc < stats.chi2.ppf(0.99, 1), "5% で有意・1% で有意でない")
E.check("s01p09b", stats.binomtest(25, 35, 0.5).pvalue < 0.05, "正確な二項検定でも 5% で有意")
pear = stats.chi2_contingency([[80, 25], [10, 35]], correction=False)[0]
E.check("s01p09b", f(pear, 2) in str(E.item("s01p09b")["choices"][1]), "独立性の検定の値（誤答）は scipy と一致")

# ---- 大問10：直交な X を作り、係数ごとに格子で目的関数を総当たり最小化
Qo, _ = np.linalg.qr(rng.normal(size=(20, 3)))
bols = np.array([2.4, -0.6, 1.1])
yy = Qo @ bols + (np.eye(20) - Qo @ Qo.T) @ rng.normal(size=20)
grid = np.round(np.arange(-3, 3.0001, 0.001), 3)
sol = []
for j in range(3):
    others = [k for k in range(3) if k != j]
    rest = yy - Qo[:, others] @ bols[others]
    obj = ((rest[:, None] - np.outer(Qo[:, j], grid)) ** 2).sum(0) + 1.6 * np.abs(grid)
    sol.append(grid[int(np.argmin(obj))])
E.check("s01p10a", np.allclose(np.linalg.lstsq(Qo, yy, rcond=None)[0], bols), "作った y の最小二乗推定値が問題文と一致")
txt = "(" + ",\\".join("0" if abs(v) < 1e-9 else f(v, 2) for v in sol) + ")"
E.choice("s01p10a", txt, f"格子探索の最小点 {sol}")

# ---- 大問11
R = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
ev = np.linalg.eigvalsh(R)
E.choice("s01p11a", f(100 * ev.max() / ev.sum(), 1) + "\\%", "numpy の固有値")
mu1, mu2 = np.array([10, 20.0]), np.array([14, 26.0])
Sig = np.array([[4.0, 3.0], [3.0, 9.0]])
wv = np.linalg.solve(Sig, mu2 - mu1)
c0 = wv @ (mu1 + mu2) / 2
Xs = rng.multivariate_normal(mu1, Sig, size=2_000_000)
err = ((Xs @ wv) > c0).mean()
nearest("s01p11b", err)
E.check("s01p11b", abs(err - stats.norm.cdf(-np.sqrt((mu2 - mu1) @ wv) / 2)) < 0.001, f"シミュレーションの誤判別率 {err:.4f} ≈ Φ(-Δ/2)")
bad = [z_ for z_, v_ in [(1.15, 0.8749), (1.33, 0.9082), (1.41, 0.9207), (2.31, 0.9896), (2.67, 0.9962)] if abs(stats.norm.cdf(z_) - v_) > 6e-5]
E.check("s01p11b", not bad, f"Φ の表の値（誤り {bad}）")

# ---- 大問12
xT, phi, c, s2 = 26, 0.6, 8, 16
m, v = xT, 0.0
for _ in range(2):
    m, v = c + phi * m, phi ** 2 * v + s2
E.num("s01p12a", rnd(m + 1.96 * np.sqrt(v), 2), "条件付き平均・分散の漸化式")
X1 = c + phi * xT + rng.normal(0, 4, size=1_000_000)
X2 = c + phi * X1 + rng.normal(0, 4, size=1_000_000)
E.check("s01p12a", abs(np.quantile(X2, 0.975) - 31.30) < 0.05, f"シミュレーションの 97.5% 点 {np.quantile(X2, 0.975):.3f}")

# ---- 大問13
Nh, Sh = np.array([6000, 3000, 1000]), np.array([2, 4, 10.0])
Wt = Nh / Nh.sum()
best = None
for n1 in range(1, 199):
    for n3 in range(1, 200 - n1):
        n2 = 200 - n1 - n3
        vv = (Wt ** 2 * Sh ** 2 / np.array([n1, n2, n3])).sum()
        if best is None or vv < best[0]:
            best = (vv, n3)
E.choice("s01p13a", f"{best[1]}人", "整数配分の総当たりで分散を最小化（FPC は無視）")
aic = {"A": -2 * -410.0 + 2 * 3, "B": -2 * -409.2 + 2 * 4, "C": -2 * -407.4 + 2 * 6}
bestm = min(aic, key=aic.get)
E.choice("s01p13b", f"{bestm}（AIC$={f(aic[bestm], 1)}$）", "AIC の比較")
E.done()
