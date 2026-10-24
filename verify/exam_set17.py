"""模擬試験 第17回（set17）の検算。generator/set17.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・数値積分・数値最適化）で正解を求め直す。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, special, stats

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
W = stats.weibull_min(2, scale=90)
E.num("s17p02a", rnd(W.ppf(0.75) - W.ppf(0.25), 1), "scipy weibull_min の ppf")
E.check("s17p02a", f(np.log(0.75), 4) == "-0.2877" and f(np.log(0.25), 4) == "-1.3863", "対数の表")
xs = np.array([1, 2, 3, 4, 5])
ps = np.array([0.10, 0.50, 0.15, 0.20, 0.05])
q1 = np.full(5, 0.2)
q2 = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
k1, k2 = special.rel_entr(ps, q1).sum(), special.rel_entr(ps, q2).sum()
E.check("s17p02b", k2 < k1, f"D(p||q1)={k1:.4f} > D(p||q2)={k2:.4f}")
E.choice("s17p02b", f"$q_2$（$D={f(k2, 3)}$）", "rel_entr で KL")
E.check("s17p02b", f(special.rel_entr(q2, ps).sum(), 3) == "0.107", "逆向きの KL の誤答は 0.107")
lnv = {0.25: -1.3863, 0.5: -0.6931, 0.75: -0.2877, 1.25: 0.2231, 2: 0.6931, 2.5: 0.9163}
E.check("s17p02b", all(f(np.log(u), 4) == f(v, 4) for u, v in lnv.items()), "対数の表")
obs = np.array([4, 20, 6, 8, 2])
ex = 40 * q2
om, em = np.r_[obs[:3], obs[3:].sum()], np.r_[ex[:3], ex[3:].sum()]
res = stats.chisquare(om, em)
E.check("s17p02c", (ex < 5).sum() == 2 and abs(res.statistic - 4.0) < 1e-12 and res.pvalue > 0.05, f"併合後 χ²={res.statistic:.2f}、p={res.pvalue:.3f}（scipy chisquare）")
E.check("s17p02c", [f(stats.chi2.isf(0.05, k), 2) for k in (2, 3, 4)] == ["5.99", "7.81", "9.49"], "χ² の表")
E.choice("s17p02c", f"4人と5人を併合して$\\chi^2={f(res.statistic, 2)}$、自由度3の7.81を超えないので、モデルは棄却されない", "scipy で判定")

# ---- 大問3
over = stats.binom.cdf(2, 103, 0.05)
E.num("s17p03a", rnd(over, 3), "binom.cdf(2, 103, 0.05)")
E.check("s17p03a", f(0.95 ** 101, 6) == "0.005625", "0.95^101 の表示値")
# (b) ベルヌーイの 3 次絶対中心モーメントを定義から（scipy の分布で期待値）
B = stats.bernoulli(0.05)
rho = B.expect(lambda x: abs(x - 0.05) ** 3)
bound = 0.4748 * rho / (B.std() ** 3 * np.sqrt(103))
E.choice("s17p03b", f"${f(bound, 3)}$", "bernoulli.expect で ρ を求めて上界")
sup = max(abs(stats.binom.cdf(k, 103, 0.05) - stats.norm.cdf((k - 5.15) / np.sqrt(103 * 0.0475))) for k in range(0, 30))
E.check("s17p03b", sup <= bound, f"実際の最大誤差 {sup:.4f} は上界以下")
distinct_rounded("s17p03b", 3)
# (c) ロジスティック回帰を数値最適化で当てはめ、情報行列から標準誤差
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
sig = 0.6
Y = np.exp(rng.normal(0, sig, size=2_000_000))
E.choice("s17p04a", f"${f(stats.norm.sf(sig / 2), 3)}$", "norm.sf(σ/2)")
E.check("s17p04a", abs((Y > np.exp(sig ** 2 / 2)).mean() - stats.norm.sf(0.3)) < 0.002, "シミュレーション")
E.check("s17p04a", all(f(stats.norm.cdf(z), 4) == v for z, v in [(0.18, "0.5714"), (0.30, "0.6179"), (0.60, "0.7257")]), "Φ の表")
z = (8.21 - 8.00) / (0.6 / 5)
pz = 2 * stats.norm.sf(z)
E.choice("s17p04b", f"$z={f(z, 2)}$、両側$p$値は{f(pz, 3)}で0.05を超えるので、帰無仮説を棄却しない", "scipy の正規分布で p 値")
E.check("s17p04b", f(stats.norm.cdf(0.35), 4) == "0.6368" and f(stats.norm.cdf(1.75), 4) == "0.9599", "Φ の表")
xsamp = rng.normal(size=10)
xsamp = (xsamp - xsamp.mean()) / xsamp.std(ddof=1) * np.sqrt(0.30) + 7.95
ci = stats.t.interval(0.95, 9, loc=xsamp.mean(), scale=stats.sem(xsamp))
E.num("s17p04c", rnd(ci[0], 2), "平均 7.95・不偏分散 0.30 のデータで scipy t.interval")
E.check("s17p04c", f(stats.t.isf(0.025, 9), 3) == "2.262", "t_0.025(9)")

# ---- 大問5
th = sp.symbols("theta", real=True)
pth = sp.exp(th) / (1 + sp.exp(th))
dens = sp.simplify(sp.diff(1 - (1 - pth) ** 2, th))
target = 2 * sp.exp(th) / (1 + sp.exp(th)) ** 3
E.check("s17p05a", sp.simplify(dens - target) == 0, "分布関数を θ で微分した密度と一致")
smp = special.logit(rng.beta(1, 2, size=400_000))
hist, edges = np.histogram(smp, bins=40, range=(-4, 2))
mid = (edges[:-1] + edges[1:]) / 2
emp = hist / (len(smp) * (edges[1] - edges[0]))
E.check("s17p05a", np.max(np.abs(emp - sp.lambdify(th, target)(mid))) < 0.01, "シミュレーションのヒストグラムと一致")
E.choice("s17p05a", r"$\dfrac{2e^{\theta}}{(1+e^{\theta})^{3}}$", "正しい式の選択肢")
logpost = lambda t_: -(t_ - 3 * np.log1p(np.exp(t_)) + 7 * t_ - 20 * np.log1p(np.exp(t_)))
th_hat = optimize.minimize_scalar(logpost, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
E.num("s17p05b", rnd(special.expit(th_hat), 3), "θ の事後密度を数値最大化して p に戻す")
E.check("s17p05b", f((8 - 1) / (23 - 2), 3) != f(special.expit(th_hat), 3), "p の尺度の MAP とは異なる")

# ---- 大問6
P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi /= pi.sum()
E.num("s17p06a", rnd(1 / pi[0], 2), "固有ベクトルの定常分布から 1/π_A")
h = np.linalg.solve(np.eye(2) - P[1:, 1:], np.ones(2))
E.check("s17p06a", abs(1 + P[0, 1:] @ h - 1 / pi[0]) < 1e-10, "初到達時間の連立方程式と一致")

# ---- 大問7
obs_ = rng.normal(size=40)
obs_ = (obs_ - obs_.mean()) / obs_.std(ddof=1) * 2.0 + 6.0
filled = np.r_[obs_, np.full(10, obs_.mean())]
E.num("s17p07a", rnd(filled.var(ddof=1), 2), "不偏分散 4.0 のデータに平均値代入して numpy で")
E.choice("s17p07b", f"${f(2000 * np.sqrt((1 - 100 / 2000) * 144 / 100), 0)}$", "公式から直接")
# シミュレーション：有限母集団から 100 組を非復元抽出したときの N ȳ の標準偏差
pop = rng.normal(38, 12, size=2000)
pop = (pop - pop.mean()) / pop.std(ddof=1) * 12 + 38
tots = np.array([2000 * rng.choice(pop, 100, replace=False).mean() for _ in range(40000)])
E.check("s17p07b", abs(tots.std() - 2339) < 25, f"シミュレーションの標準偏差 {tots.std():.1f}")

# ---- 大問8
tt = (-1.35 + 1) / 0.14
E.choice("s17p08a", f"$t={f(tt, 2)}$で$|t|>2.010$なので、弾力性は$-1$と異なり", "t を直接")
E.check("s17p08a", f(stats.t.isf(0.025, 49), 3) == "2.010" and 2 * stats.t.sf(abs(tt), 49) < 0.05, "t 点と p 値")
# (b) 不均一分散のシミュレーション：OLS は不偏だが通常の SE は過小
xx = np.linspace(1, 10, 60)
bs, ses = [], []
for _ in range(3000):
    yy = 1 + 2 * xx + rng.normal(scale=0.3 * xx ** 1.5)
    Xm = np.column_stack([np.ones(60), xx])
    bb, rss, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
    bs.append(bb[1])
    ses.append(np.sqrt(rss[0] / 58 * np.linalg.inv(Xm.T @ Xm)[1, 1]))
E.check("s17p08b", abs(np.mean(bs) - 2) < 0.02 and np.mean(ses) < np.std(bs), f"係数は不偏（平均 {np.mean(bs):.3f}）、通常の SE {np.mean(ses):.3f} < 実際のばらつき {np.std(bs):.3f}")
E.check("s17p08b", "不均一分散" in ans_text("s17p08b"), "正解は不均一分散")

# ---- 大問9
# MA(1) の自己相関を ψ 重みから、偏自己相関をダービン・レビンソンで
r1, r2 = 0.5 / 1.25, 0.0
phi11 = r1
phi22 = (r2 - phi11 * r1) / (1 - phi11 * r1)
xs_ = rng.normal(size=400_001)
ma = xs_[1:] + 0.5 * xs_[:-1]
R = np.array([[1, np.corrcoef(ma[1:], ma[:-1])[0, 1]], [np.corrcoef(ma[1:], ma[:-1])[0, 1], 1]])
rr = np.array([np.corrcoef(ma[1:], ma[:-1])[0, 1], np.corrcoef(ma[2:], ma[:-2])[0, 1]])
pacf2 = np.linalg.solve(R, rr)[1]
E.choice("s17p09a", f"${f(phi22, 3)}$", "ダービン・レビンソンの漸化式")
E.check("s17p09a", abs(pacf2 - phi22) < 0.01, f"シミュレーションの偏自己相関 {pacf2:.4f}")
relu = lambda u: np.maximum(0, u)
x_in = np.array([1.0, 1.0])
W1 = np.array([[1.0, -2.0], [0.5, 3.0]])
b1 = np.array([0.5, -1.0])
yhat = 2.0 + np.array([1.5, 2.0]) @ relu(W1 @ x_in + b1)
E.choice("s17p09b", f"${f(yhat, 2)}$", "行列で順伝播")

# ---- 大問10：対数尤度を数値最大化
cnt = 50
ll = lambda lam: -(100 * np.log(lam) - cnt * lam - cnt * np.log1p(-np.exp(-lam)))
lam_hat = optimize.minimize_scalar(ll, bounds=(0.1, 5), method="bounded", options={"xatol": 1e-10}).x
v = nums("s17p10a")
E.check("s17p10a", int(np.argmin([abs(c - lam_hat) for c in v])) == E.item("s17p10a")["answer"], f"数値最大化 λ̂={lam_hat:.4f} に最も近い選択肢が正解")
E.check("s17p10a", all(f(np.exp(-l), 4) == e for l, e in [(1.0, "0.3679"), (1.2, "0.3012"), (1.59, "0.2039"), (1.8, "0.1653"), (2.0, "0.1353")]), "e^{-λ} の表")

# ---- 大問11
d = 0.9 / 1.5
pw = stats.norm.sf(1.96 - d) + stats.norm.cdf(-1.96 - d)
E.choice("s17p11a", f"${f(pw, 3)}$", "scipy で両側の棄却確率")
xb = rng.normal(30.9, 1.5, size=2_000_000)
E.check("s17p11a", abs((np.abs((xb - 30) / 1.5) > 1.96).mean() - pw) < 0.001, "シミュレーション")
E.check("s17p11a", all(f(stats.norm.cdf(zz), 4) == vv for zz, vv in [(0.60, "0.7257"), (1.36, "0.9131"), (2.56, "0.9948")]), "Φ の表")
distinct_rounded("s17p11a", 3)

# ---- 大問12：直交表から交互作用列を求め、F 点を確認
L8 = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 2, 2, 2], [1, 2, 2, 1, 1, 2, 2], [1, 2, 2, 2, 2, 1, 1],
               [2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 2, 1, 2, 1], [2, 2, 1, 1, 2, 2, 1], [2, 2, 1, 2, 1, 1, 2]])
inter = lambda a, b_: np.where(L8[:, a - 1] == L8[:, b_ - 1], 1, 2)
col = [c + 1 for c in range(7) if np.array_equal(L8[:, c], inter(1, 4))]
E.check("s17p12a", col == [5] and np.array_equal(L8[:, 2], inter(1, 2)) and np.array_equal(L8[:, 5], inter(2, 4)), f"A×C の列 {col}、A×B は列3、B×C は列6")
E.check("s17p12a", [f(stats.f.isf(0.05, 1, k), 2) for k in (2, 3, 6)] == ["18.51", "10.13", "5.99"], "F の表")
Fv = 20.0 / ((1.5 + 0.5) / 2)
E.choice("s17p12a", f"A×Cは列5に現れ、$F={f(Fv, 1)}$は$F_{{0.05}}(1,2)=18.51$を超えるので5%で有意である", "F を計算")
E.check("s17p12a", stats.f.sf(Fv, 1, 2) < 0.05, f"p 値 {stats.f.sf(Fv, 1, 2):.4f}")

# ---- 大問13
A6 = rng.normal(size=(6, 6))
S6 = A6 @ A6.T
best = max(-optimize.minimize(lambda a: -(a @ S6 @ a) / (a @ a), rng.normal(size=6)).fun for _ in range(5))
E.check("s17p13a", abs(best - np.linalg.eigvalsh(S6).max()) < 1e-6, "制約つき最大値＝最大固有値（数値最適化）")
E.check("s17p13a", "最大固有値 $\\lambda_1$" in ans_text("s17p13a"), "正解の記述")
LA = np.array([[0.58, 0.56], [0.56, 0.42], [-0.28, 0.54]])
LB = np.array([[0.80, 0.10], [0.70, 0.00], [0.10, 0.60]])
T = np.linalg.lstsq(LA, LB, rcond=None)[0]
E.check("s17p13b", np.allclose(T @ T.T, np.eye(2), atol=1e-9) and np.allclose(LA @ T, LB, atol=1e-9), "B は A の直交回転")
vm = lambda L: (L ** 2).var(axis=0).sum()
E.choice("s17p13b", f"B（$V={f(vm(LB), 3)}$）", "numpy var で V")
E.check("s17p13b", vm(LB) > vm(LA), "V_B > V_A")
# (c) 実際に指示行列を作って対応分析の総イナーシャと非零特異値の数を求める
resp = np.column_stack([rng.integers(0, 3, 500), rng.integers(0, 4, 500), rng.integers(0, 2, 500)])
Z = np.hstack([np.eye(k)[resp[:, j]] for j, k in enumerate((3, 4, 2))])
Pm = Z / Z.sum()
rm, cm = Pm.sum(1), Pm.sum(0)
Sres = (Pm - np.outer(rm, cm)) / np.sqrt(np.outer(rm, cm))
sv = np.linalg.svd(Sres, compute_uv=False)
E.check("s17p13c", abs((sv ** 2).sum() - 2.0) < 1e-9 and (sv > 1e-9).sum() == 6, f"総イナーシャ {(sv ** 2).sum():.4f}、非零の次元 {(sv > 1e-9).sum()}")
E.choice("s17p13c", "総イナーシャ$2.00$、次元数6", "指示行列の SVD")

E.done()
