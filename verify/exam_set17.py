"""模擬試験 第17回（set17）の検算。generator/set17.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・数値積分・数値最適化）で正解を求め直す。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(17)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    v = nums(iid)
    k = int(np.argmin([abs(x - value) for x in v]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']}")


def distinct_rounded(iid, d):
    v = nums(iid)
    E.check(iid, len({f(x, d) for x in v}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {v}")


def ans_text(iid):
    it = E.item(iid)
    return it["choices"][it["answer"]]


# ---- 大問1：4! 通りの返し方を総当たり
der = sum(1 for p in permutations(range(4)) if all(p[i] != i for i in range(4)))
E.choice("s17p01a", f"${f(der / 24, 3)}$", f"総当たりで完全順列 {der}/24")
distinct_rounded("s17p01a", 3)

# ---- 大問2
xs = np.array([1, 2, 3, 4, 5])
ps = np.array([0.10, 0.50, 0.15, 0.20, 0.05])
dist = stats.rv_discrete(values=(xs, ps))
E.num("s17p02a", rnd(dist.stats(moments="s"), 2), "scipy rv_discrete の歪度")
# (b) 60 組の合計人数の正確な分布（畳み込み）と、連続修正つき正規近似
pmf = np.zeros(6)
pmf[1:] = ps
tot = np.array([1.0])
for _ in range(60):
    tot = np.convolve(tot, pmf)
exact = tot[171:].sum()
mu, var = dist.mean() * 60, dist.var() * 60
approx = stats.norm.sf(170.5, mu, np.sqrt(var))
E.choice("s17p02b", f"${f(1 - stats.norm.cdf(rnd((170.5 - mu) / np.sqrt(var), 2)), 3)}$", "scipy の Φ で連続修正つき近似")
E.check("s17p02b", abs(approx - exact) < 0.01, f"正確な確率 {exact:.4f} と近似 {approx:.4f} が近い")
phis = {0.21: 0.5832, 1.63: 0.9484, 1.69: 0.9545, 1.75: 0.9599}
E.check("s17p02b", all(f(stats.norm.cdf(z), 4) == f(v, 4) for z, v in phis.items()), "Φ の表")
distinct_rounded("s17p02b", 3)
# (c) scipy.special.rel_entr で KL
q1 = np.full(5, 0.2)
q2 = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
k1, k2 = special.rel_entr(ps, q1).sum(), special.rel_entr(ps, q2).sum()
E.check("s17p02c", k2 < k1, f"D(p||q1)={k1:.4f} > D(p||q2)={k2:.4f}")
E.choice("s17p02c", f"$q_2$（$D={f(k2, 3)}$）", "rel_entr で KL")
E.check("s17p02c", f(special.rel_entr(q2, ps).sum(), 3) == "0.107", "逆向きの KL の誤答は 0.107")
lnv = {0.25: -1.3863, 0.5: -0.6931, 0.75: -0.2877, 1.25: 0.2231, 2: 0.6931, 2.5: 0.9163}
E.check("s17p02c", all(f(np.log(u), 4) == f(v, 4) for u, v in lnv.items()), "対数の表")

# ---- 大問3
over = stats.binom.cdf(2, 103, 0.05)
E.num("s17p03a", rnd(over, 3), "binom.cdf(2, 103, 0.05)")
E.check("s17p03a", f(0.95 ** 101, 6) == "0.005625", "0.95^101 の表示値")
sim = rng.binomial(103, 0.05, size=1_000_000)
E.check("s17p03a", abs((sim <= 2).mean() - over) < 0.002, f"シミュレーション {(sim <= 2).mean():.4f}")
# (b) 2×2 表のカイ二乗（連続修正なし）の平方根がプールした z
tab = np.array([[32, 368], [6, 194]])
chi2, pv, _, _ = stats.chi2_contingency(tab, correction=False)
E.choice("s17p03b", f"$z={f(np.sqrt(chi2), 2)}$で1.96を超えるので", "χ² 統計量の平方根")
E.check("s17p03b", pv < 0.05, f"p 値 {pv:.4f} < 0.05")
# (c) ロジスティック回帰を数値最適化で当てはめ、観測情報量から標準誤差
D = np.r_[np.ones(400), np.zeros(200)]
y = np.r_[np.ones(32), np.zeros(368), np.ones(6), np.zeros(194)]
X = np.column_stack([np.ones(600), D])
nll = lambda b: -(y * (X @ b) - np.log1p(np.exp(X @ b))).sum()
b = optimize.minimize(nll, np.zeros(2), method="BFGS", options={"gtol": 1e-10}).x
pr = special.expit(X @ b)
Info = X.T @ (X * (pr * (1 - pr))[:, None])
se1 = np.sqrt(np.linalg.inv(Info)[1, 1])
E.num("s17p03c", rnd(b[1] / se1, 2), "最尤推定と情報行列で z")
E.check("s17p03c", f(np.log(6208 / 2208), 4) == "1.0338", "log(6208/2208)")

# ---- 大問4
n = 10
lam = 1.0
T = rng.gamma(n, 1 / lam, size=2_000_000)
v_sim = ((n - 1) / T).var()
eff = (lam ** 2 / n) / v_sim
E.check("s17p04a", abs(eff - 0.8) < 0.01, f"シミュレーションの効率 {eff:.4f}")
t = sp.symbols("t", positive=True)
l = sp.symbols("lambda", positive=True)
dens = l ** n * t ** (n - 1) * sp.exp(-l * t) / sp.factorial(n - 1)
E2 = sp.simplify(sp.integrate(dens / t ** 2, (t, 0, sp.oo)))
E1 = sp.simplify(sp.integrate(dens / t, (t, 0, sp.oo)))
E.check("s17p04a", sp.simplify(E1 - l / 9) == 0 and sp.simplify(E2 - l ** 2 / 72) == 0, "sympy で E[1/T]・E[1/T²]（問題文の式）")
V = sp.simplify(81 * E2 - l ** 2)
E.choice("s17p04a", f"${f(float(sp.simplify((l ** 2 / n) / V)), 2)}$", "sympy で効率")
c = 5 * stats.chi2.isf(0.05, 20) / 2
E.num("s17p04b", rnd(5 * 31.41 / 2, 1), "表の χ² 点から")
E.check("s17p04b", abs(stats.gamma(10, scale=5).sf(78.5) - 0.05) < 0.001 and abs(c - 78.5) < 0.05, f"gamma(10, scale=5) の上側 5% 点 {c:.3f}")
E.check("s17p04b", [f(stats.chi2.isf(0.05, 10), 2), f(stats.chi2.isf(0.05, 20), 2), f(stats.chi2.isf(0.95, 20), 2)] == ["18.31", "31.41", "10.85"], "χ² の表")

# ---- 大問5
sig = 0.6
Y = np.exp(rng.normal(0, sig, size=2_000_000))
frac = (Y > np.exp(sig ** 2 / 2)).mean()
E.choice("s17p05a", f"${f(stats.norm.sf(sig / 2), 3)}$", "norm.sf(σ/2)")
E.check("s17p05a", abs(frac - stats.norm.sf(0.3)) < 0.002, f"シミュレーション {frac:.4f}")
E.check("s17p05a", all(f(stats.norm.cdf(z), 4) == v for z, v in [(0.18, "0.5714"), (0.30, "0.6179"), (0.60, "0.7257")]), "Φ の表")
F = 0.64 ** 2 / 0.40 ** 2
fa, fb, fc = stats.f.isf(0.025, 12, 15), stats.f.isf(0.05, 12, 15), stats.f.isf(0.025, 13, 16)
E.check("s17p05b", [f(fa, 2), f(fb, 2), f(fc, 2)] == ["2.96", "2.48", "2.85"], f"F の点 {fa:.4f} {fb:.4f} {fc:.4f}")
E.check("s17p05b", 2 * stats.f.sf(F, 12, 15) > 0.05 > stats.f.sf(F, 12, 15), "両側 p>0.05、片側 p<0.05")
E.choice("s17p05b", f"$F={f(F, 2)}$は$F_{{0.025}}(12,15)=2.96$を超えないので、帰無仮説を棄却しない", "scipy で判定")
# (c) 要約統計量から scipy の ttest_ind_from_stats で t を出し、区間を逆算
r_ = stats.ttest_ind_from_stats(8.20, 0.64, 13, 7.90, 0.40, 16, equal_var=True)
se = 0.30 / r_.statistic
E.check("s17p05c", f(stats.t.isf(0.025, 27), 3) == "2.052", "t_0.025(27)")
E.num("s17p05c", rnd(0.30 - 2.052 * se, 2), "ttest_ind_from_stats の t から標準誤差")

# ---- 大問6
th = sp.symbols("theta", real=True)
pth = sp.exp(th) / (1 + sp.exp(th))
Fp = 1 - (1 - pth) ** 2
dens6 = sp.simplify(sp.diff(Fp, th))
target = 2 * sp.exp(th) / (1 + sp.exp(th)) ** 3
E.check("s17p06a", sp.simplify(dens6 - target) == 0, "分布関数を θ で微分した密度と一致")
smp = special.logit(rng.beta(1, 2, size=400_000))
hist, edges = np.histogram(smp, bins=40, range=(-4, 2))
mid = (edges[:-1] + edges[1:]) / 2
emp = hist / (len(smp) * (edges[1] - edges[0]))
E.check("s17p06a", np.max(np.abs(emp - sp.lambdify(th, target)(mid))) < 0.01, "シミュレーションのヒストグラムと一致")
E.choice("s17p06a", r"$\dfrac{2e^{\theta}}{(1+e^{\theta})^{3}}$", "正しい式の選択肢")
# (b) θ の事後密度（事前密度 × 尤度）を数値最大化
logpost = lambda t_: -(np.log(2) + t_ - 3 * np.log1p(np.exp(t_)) + 7 * t_ - 20 * np.log1p(np.exp(t_)))
th_hat = optimize.minimize_scalar(logpost, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
E.num("s17p06b", rnd(special.expit(th_hat), 3), "θ の事後密度を数値最大化して p に戻す")
E.check("s17p06b", abs(stats.beta(8, 15).ppf(0.5) - special.expit(np.median(special.logit(rng.beta(8, 15, 200_000))))) < 0.003, "中央値は変換で対応（参考）")

# ---- 大問7
P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi /= pi.sum()
E.num("s17p07a", rnd(1 / pi[0], 2), "固有ベクトルの定常分布から 1/π_A")
# 初到達時間の連立方程式（B, C から A までの期待時間）で直接
Q = P[1:, 1:]
h = np.linalg.solve(np.eye(2) - Q, np.ones(2))
ret = 1 + P[0, 1:] @ h
E.check("s17p07a", abs(ret - 1 / pi[0]) < 1e-10, f"初到達時間から {ret:.4f}")

# ---- 大問8：自作の KM とグリーンウッドを、別の書き方（時点の総当たり）で
times = np.array([2, 3, 4, 4, 6, 7, 8, 9, 10, 12])
event = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
Sh, g = 1.0, 0.0
for tt in np.unique(times[(event == 1) & (times <= 6)]):
    nrisk = (times >= tt).sum()
    d = ((times == tt) & (event == 1)).sum()
    Sh *= 1 - d / nrisk
    g += d / (nrisk * (nrisk - d))
E.choice("s17p08a", f"${f(Sh * np.sqrt(g), 3)}$", f"KM {Sh:.4f} とグリーンウッド")
distinct_rounded("s17p08a", 3)

# ---- 大問9：人工データ（x̄=25, Sxx=250, n=20）で numpy の行列から平均の信頼区間
xg = np.linspace(-1, 1, 20)
xg = 25 + (xg - xg.mean()) * np.sqrt(250 / ((xg - xg.mean()) ** 2).sum())
Xd = np.column_stack([np.ones(20), xg])
x0 = np.array([1, 30.0])
se_mean = 8.0 * np.sqrt(x0 @ np.linalg.inv(Xd.T @ Xd) @ x0)
lo = 12 + 3 * 30 - stats.t.isf(0.025, 18) * se_mean
E.choice("s17p09a", f"${f(lo, 2)}$", "行列 x0ᵀ(XᵀX)⁻¹x0 と scipy の t")
E.check("s17p09a", f(stats.t.isf(0.025, 18), 3) == "2.101" and f(stats.t.isf(0.025, 19), 3) == "2.093", "t の点")
# (b) t 分布（裾が重い）の標本の QQ プロットで両端が外側に外れることをシミュレーションで確認
z = np.sort(rng.standard_t(3, size=2000))
z = (z - z.mean()) / z.std()
theo = stats.norm.ppf((np.arange(1, 2001) - 0.5) / 2000)
E.check("s17p09b", z[:10].mean() < theo[:10].mean() and z[-10:].mean() > theo[-10:].mean(), "裾の重い分布：左端は下、右端は上")
zr = np.sort(rng.exponential(size=2000))
zr = (zr - zr.mean()) / zr.std()
E.check("s17p09b", zr[:10].mean() > theo[:10].mean() and zr[-10:].mean() > theo[-10:].mean(), "右に歪んだ分布：両端とも上（選択肢1は別の形）")
E.check("s17p09b", "重い" in ans_text("s17p09b"), "正解は裾が重い")

# ---- 大問10：チェック関数を格子上で直接最小化
yy = np.array([22, 25, 18, 30, 27, 24, 35, 21, 26, 29, 41])
grid = np.arange(15, 45.001, 0.01)
loss = [((yy - q) * (0.8 - (yy - q < 0))).sum() for q in grid]
qmin = grid[int(np.argmin(loss))]
E.choice("s17p10a", f"${int(round(qmin))}$", f"格子で最小化 {qmin:.2f}")
E.check("s17p10a", np.quantile(yy, 0.8, method="inverted_cdf") == 30, "numpy の逆分布関数型の分位点と一致")

# ---- 大問11：データを作って一元配置分散分析から ω²
g4 = [rng.normal(size=8) for _ in range(4)]
# 平方和を 36（群間）・84（群内）に合わせる
gm = [x - x.mean() for x in g4]
sw = sum((x ** 2).sum() for x in gm)
gm = [x * np.sqrt(84 / sw) for x in gm]
eff_ = np.array([1.0, -1.0, 0.5, -0.5])
eff_ = eff_ * np.sqrt(36 / (8 * (eff_ ** 2).sum()))
data = [gm[i] + 10 + eff_[i] for i in range(4)]
allv = np.concatenate(data)
ssa = sum(8 * (d.mean() - allv.mean()) ** 2 for d in data)
sst = ((allv - allv.mean()) ** 2).sum()
mse = (sst - ssa) / 28
E.check("s17p11a", abs(ssa - 36) < 1e-9 and abs(sst - 120) < 1e-9, "作ったデータの平方和")
E.check("s17p11a", abs(stats.f_oneway(*data).statistic - (ssa / 3) / mse) < 1e-9, "scipy f_oneway と一致")
E.choice("s17p11a", f"${f((ssa - 3 * mse) / (sst + mse), 3)}$", "データから ω²")
distinct_rounded("s17p11a", 3)

# ---- 大問12：母集団シミュレーション（無回答者は満足度が低い）で推定量の不偏性を確認
est = (300 * 4.2 + 200 * 3.6) / 500
E.choice("s17p12a", f"${f(est, 2)}$", "人数で重み付け")
popN = 20000
resp = rng.random(popN) < 0.6
ys = np.where(resp, rng.normal(4.2, 0.8, popN), rng.normal(3.6, 0.8, popN))
ests = []
for _ in range(2000):
    idx = rng.choice(popN, 500, replace=False)
    r_i, m_i = idx[resp[idx]], idx[~resp[idx]]
    fu = rng.choice(m_i, 40, replace=False)
    ests.append(len(r_i) / 500 * ys[r_i].mean() + len(m_i) / 500 * ys[fu].mean())
E.check("s17p12a", abs(np.mean(ests) - ys.mean()) < 0.01, f"二相抽出の推定量は不偏（平均 {np.mean(ests):.4f}、母平均 {ys.mean():.4f}）")
distinct_rounded("s17p12a", 2)

# ---- 大問13
# (a) 適当な共分散行列で、a'a=1 のもとでの a'Σa の最大値が最大固有値になることを数値最適化で確認
A = rng.normal(size=(6, 6))
S6 = A @ A.T
best = max(-optimize.minimize(lambda a: -(a @ S6 @ a) / (a @ a), rng.normal(size=6)).fun for _ in range(5))
E.check("s17p13a", abs(best - np.linalg.eigvalsh(S6).max()) < 1e-6, "制約つき最大値＝最大固有値（数値最適化）")
E.check("s17p13a", "最大固有値 $\\lambda_1$" in ans_text("s17p13a") and "\\boldsymbol a^\\top\\boldsymbol a=1" in ans_text("s17p13a"), "正解の記述")
# (b)
Lam = np.array([[0.70, 0.30], [0.20, 0.60]])
Tm = np.array([[0.8, -0.6], [0.6, 0.8]])
Ls = Lam @ Tm
E.check("s17p13b", np.allclose(Tm @ Tm.T, np.eye(2)) and np.allclose(Lam @ Lam.T, Ls @ Ls.T), "直交回転で ΛΛᵀ 不変")
E.choice("s17p13b", f"負荷量$({f(Ls[0, 0], 2)},\\ {f(Ls[0, 1], 2)})$、共通性${f((Ls[0] ** 2).sum(), 2)}$", "行列積で回転")
# (c) 距離は 2 次元の点 A(0,0), B(1,0), C(0,1.2), D(2,1) から（配置として実現可能）
pts = {"A": (0, 0), "B": (1, 0), "C": (0, 1.2), "D": (2, 1)}
order = ["AB", "BD", "AC", "BC", "AD", "CD"]
dd = np.array([rnd(math.dist(pts[p[0]], pts[p[1]]), 2) for p in order])
E.check("s17p13c", list(dd) == [1.00, 1.41, 1.20, 1.56, 2.24, 2.01], f"表の距離は 2 次元配置から {dd}")
# 単調回帰を制約つき最小二乗（SLSQP）として解く
cons = [{"type": "ineq", "fun": (lambda h, i=i: h[i + 1] - h[i])} for i in range(5)]
dh = optimize.minimize(lambda h: ((dd - h) ** 2).sum(), dd.copy(), constraints=cons, method="SLSQP", options={"ftol": 1e-14}).x
E.num("s17p13c", rnd(np.sqrt(((dd - dh) ** 2).sum() / (dd ** 2).sum()), 3), "SLSQP の単調回帰でストレス")

# ---- 大問14：シミュレーションで Q の帰無分布の確認と、値の計算
rk = np.array([0.15, -0.10, 0.12])
Qv = 100 * 102 * (rk ** 2 / (100 - np.arange(1, 4))).sum()
E.choice("s17p14a", f"$Q={f(Qv, 2)}$で、自由度1の3.84を超えるので、残差に自己相関が残っている", "Q を直接")
E.check("s17p14a", [f(stats.chi2.isf(0.05, k), 2) for k in (1, 2, 3)] == ["3.84", "5.99", "7.81"], "χ² の表")
# ARMA(1,1) を当てはめた残差の Q の分布が自由度 1 に近いことを、簡易シミュレーションで確認（AR(1) で代用：df=K−1）
qs = []
for _ in range(1500):
    e = rng.normal(size=301)
    xsim = np.zeros(301)
    for i in range(1, 301):
        xsim[i] = 0.5 * xsim[i - 1] + e[i]
    xsim = xsim[1:]
    ph = (xsim[1:] @ xsim[:-1]) / (xsim[:-1] @ xsim[:-1])
    res = xsim[1:] - ph * xsim[:-1]
    res -= res.mean()
    nn = len(res)
    r = np.array([(res[k:] @ res[:-k]) / (res @ res) for k in (1, 2)])
    qs.append(nn * (nn + 2) * (r ** 2 / (nn - np.arange(1, 3))).sum())
E.check("s17p14a", abs(np.mean(qs) - 1) < 0.2, f"係数を 1 つ推定した残差の Q（K=2）の平均 {np.mean(qs):.3f} ≈ 自由度 1")

E.done()
