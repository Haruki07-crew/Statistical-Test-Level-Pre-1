"""模擬試験 第22回（set22）の検算。generator/set22.py とは別の方法（scipy の分布関数・全列挙・
シミュレーション・sympy・数値積分）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import permutations, combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(22)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [[float(v) for v in re.findall(r"-?\d+\.?\d*", str(c).replace("−", "-"))] for c in E.item(iid)["choices"]]


def near(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1
# (a) 20 箱の全順列を数え上げる代わりに、破損箱の位置の組（C(20,3) 通り）を全列挙
cnt = sum(1 for c in combinations(range(20), 3) if min(c) == 2)
p = Fraction(cnt, math.comb(20, 3))
E.num("s22p01a", rnd(float(p), 3), "破損箱の位置の全列挙で最初の破損が3番目")
sim = rng.permuted(np.tile(np.r_[np.ones(3), np.zeros(17)], (200000, 1)), axis=1)
first = sim.argmax(axis=1)
E.check("s22p01a", abs((first == 2).mean() - float(p)) < 0.003, f"シミュレーション {(first == 2).mean():.4f}")
# (c) 半幅条件を満たす最小の n を総当たり
z = stats.norm.ppf(0.975)
n = next(k for k in range(1, 5000) if 1.96 * math.sqrt(0.15 * 0.85 / k) <= 0.03)
E.num("s22p01b", n, "半幅 ≤ 0.03 となる最小の n を総当たり")
E.check("s22p01b", abs(z - 1.96) < 0.001, f"z_0.025={z:.4f}")

# ---- 大問2
pa = stats.poisson.sf(5, 3.0)
E.num("s22p02a", rnd(pa, 3), "scipy poisson.sf(5, 3)")
E.check("s22p02a", abs(math.exp(-3) - 0.0498) < 5e-5, "e^-3 = 0.0498")
E.check("s22p02a", abs((rng.poisson(1.5, (300000, 2)).sum(1) > 5).mean() - pa) < 0.003, "2日分の需要のシミュレーション")
# (b) 期待損失を需要分布の仮定のもとで数値的に最小化して分位点が τ=0.8 になることを確かめる（正規分布で例示）
Y = rng.normal(150, 20, 400000)
qs = np.arange(100, 200, 0.5)
loss = [np.mean(400 * np.maximum(Y - q, 0) + 100 * np.maximum(q - Y, 0)) for q in qs]
qopt = qs[int(np.argmin(loss))]
E.check("s22p02b", abs(stats.norm.cdf(qopt, 150, 20) - 0.8) < 0.01, f"期待損失の最小点の分布関数値 {stats.norm.cdf(qopt, 150, 20):.3f} ≈ 0.8")
E.choice("s22p02b", f"${121 + 44 * 1}$", "τ=0.8 の推定式で x=1")

# ---- 大問3
# (a) 数値計算でハザード
S = lambda t: 0.6 * math.exp(-t) + 0.4 * math.exp(-t / 3)
dens = lambda t: -(S(t + 1e-6) - S(t - 1e-6)) / 2e-6
h = dens(3) / S(3)
E.choice("s22p03a", f"${f(h, 3)}$", "数値微分による f/S")
E.check("s22p03a", abs(math.exp(-1) - 0.3679) < 5e-5, "e^-1 = 0.3679")
# (b) scipy の順序統計量（ベータ分布）と数値積分
vmed = 120 ** 2 * stats.beta(3, 3).var()
vmean = 120 ** 2 / 12 / 5
E.choice("s22p03b", f"${f(vmed / vmean, 3)}$", "beta(3,3) の分散から比")
U = rng.uniform(0, 120, (200000, 5))
r_sim = np.median(U, axis=1).var() / U.mean(axis=1).var()
E.check("s22p03b", abs(r_sim - vmed / vmean) < 0.05, f"シミュレーションの比 {r_sim:.3f}")

# ---- 大問4
a6 = 6.0
ER = integrate.quad(lambda r: r * 2 * r / a6 ** 2, 0, a6)[0]
E.choice("s22p04a", f"${f(ER, 2)}$", "数値積分 E[R]")
th = rng.uniform(0, 2 * np.pi, 400000)
rad = a6 * np.sqrt(rng.uniform(0, 1, 400000))
pts = rng.uniform(-a6, a6, (800000, 2))
inside = pts[(pts ** 2).sum(1) <= a6 ** 2]
E.check("s22p04a", abs(np.sqrt((inside ** 2).sum(1)).mean() - 4.0) < 0.02, "円内一様点の棄却法シミュレーション")
rs = np.array([0.9, 2.4, 3.9, 5.4])
fpdf = lambda r: 2 * r / a6 ** 2
gpdf = lambda r: 1 / a6
E.choice("s22p04b", f"${f(np.mean(rs * fpdf(rs) / gpdf(rs)), 2)}$", "重要度サンプリング推定量を直接計算")
uu = rng.uniform(0, a6, 400000)
E.check("s22p04b", abs(np.mean(uu * fpdf(uu) / gpdf(uu)) - 4.0) < 0.02, "多数の提案値で不偏性を確認")

# ---- 大問5
# (b) 遷移率行列の定常分布を数値的に解く
lam, mu = 20.0, 10.0
Q = np.zeros((4, 4))
for i in range(4):
    if i < 3:
        Q[i, i + 1] = lam
    if i > 0:
        Q[i, i - 1] = i * mu
    Q[i, i] = -Q[i].sum()
A = np.vstack([Q.T, np.ones(4)])
pi = np.linalg.lstsq(A, np.r_[np.zeros(4), 1], rcond=None)[0]
E.num("s22p05a", rnd(lam * (1 - pi[3]), 2), "生成行列から定常分布を解いて λ(1−π3)")
# 離散事象シミュレーション
t, busy, acc, arr = 0.0, [], 0, 0
T_END = 20000.0
nxt = rng.exponential(1 / lam)
while nxt < T_END:
    busy = [e for e in busy if e > nxt]
    arr += 1
    if len(busy) < 3:
        busy.append(nxt + rng.exponential(1 / mu))
        acc += 1
    nxt += rng.exponential(1 / lam)
E.check("s22p05a", abs(acc / T_END - lam * (1 - pi[3])) < 0.15, f"シミュレーションの受付率 {acc / T_END:.3f}/時")
Nn = rng.poisson(20, 1000000)
E.choice("s22p02c", "$0.250$", "デルタ法の値")
E.check("s22p02c", abs(np.sqrt(Nn).var() - 0.25) < 0.02, f"シミュレーションの V[√N]={np.sqrt(Nn).var():.4f}")
# (b) フィッシャーの方法
pv = np.array([0.08, 0.12, 0.06])
X = -2 * np.log(pv).sum()
E.check("s22p05b", all(abs(math.log(p_) - q_) < 5e-5 for p_, q_ in zip(pv, [-2.5257, -2.1203, -2.8134])), "示した対数")
E.check("s22p05b", abs(stats.chi2.ppf(0.95, 6) - 12.59) < 0.005 and abs(stats.chi2.ppf(0.95, 3) - 7.81) < 0.005, "χ² の点")
E.check("s22p05b", stats.combine_pvalues(pv, method="fisher").pvalue < 0.05, f"scipy combine_pvalues p={stats.combine_pvalues(pv, method='fisher').pvalue:.4f}")
E.choice("s22p05b", f"$X={f(X, 2)}$が$\\chi^2_{{0.05}}(6)", "−2Σlog p")
U = rng.uniform(size=(200000, 3))
E.check("s22p05b", abs((-2 * np.log(U).sum(1) > 12.59).mean() - 0.05) < 0.003, "帰無仮説のもとのシミュレーションで棄却率 5%")

# ---- 大問6
TA = np.array([1, 2, 2, 3]); TB = np.array([2, 3, 4, 4, 5])
rk = stats.rankdata(np.r_[TA, TB])
Wt = rk[:4].sum()
_, cnts = np.unique(np.r_[TA, TB], return_counts=True)
Vt = 4 * 5 / 12 * (10 - np.sum(cnts ** 3 - cnts) / (9 * 8))
E.choice("s22p06a", f"${f((Wt - 20) / math.sqrt(Vt), 2)}$", "scipy rankdata の中間順位と補正分散")
# 全並べ替え（同順位あり）による W の分散と一致することを確かめる
allv = np.r_[TA, TB]
Ws = [rk[list(c)].sum() for c in combinations(range(9), 4)]
E.check("s22p06a", abs(np.var(Ws) - Vt) < 1e-9, f"全 126 通りの並べ替えでの W の分散 {np.var(Ws):.4f}")
z_mw = stats.mannwhitneyu(TA, TB, alternative="two-sided", method="asymptotic", use_continuity=False)
E.check("s22p06a", abs(stats.norm.ppf(z_mw.pvalue / 2) - (Wt - 20) / math.sqrt(Vt)) < 1e-6, "scipy mannwhitneyu（補正つき正規近似）の p 値と一致")
# (b) 非心 t 分布の正規近似。示した t・Φ の値を scipy で確認し、近似値を計算
E.check("s22p06b", abs(stats.t.ppf(0.95, 15) - 1.753) < 5e-4 and abs(stats.t.ppf(0.975, 15) - 2.131) < 5e-4, "t の点")
E.check("s22p06b", all(abs(stats.norm.cdf(z) - v) < 5e-5 for z, v in [(0.37, 0.6443), (0.75, 0.7734), (0.86, 0.8051), (1.13, 0.8708)]), "Φ の値")
delta = 5 * math.sqrt(16) / 8
E.choice("s22p06b", f"${f(stats.norm.cdf(round(delta - stats.t.ppf(0.95, 15), 2)), 3)}$", "scipy の t 点から近似検出力")
exact = stats.nct.sf(stats.t.ppf(0.95, 15), 15, delta)
E.check("s22p06b", abs(exact - 0.773) < 0.04, f"非心 t の正確な検出力 {exact:.3f} も近い（近似の妥当性）")

# ---- 大問7
xx = sp.symbols("x")
sol = sp.solve(sp.Eq(2.0 + 0.5 * xx, 2.0 + 0.5 * xx + 1.5 - 0.30 * xx), xx)
E.choice("s22p07a", f"${f(float(sol[0]), 1)}$", "sympy で 2 直線の交点")
x7 = np.array([2, 4, 5, 8.0])
y7 = np.array([1.0, 2.4, 2.5, 4.8])
W = np.diag(1 / x7 ** 2)
Xm = x7[:, None]
b_wls = np.linalg.solve(Xm.T @ W @ Xm, Xm.T @ W @ y7)[0]
E.choice("s22p07b", f"${f(b_wls, 3)}$", "行列で (X'WX)^{-1}X'Wy")
# 分散が x² に比例するときの効率：シミュレーションで WLS の分散が最小
sims = []
for _ in range(20000):
    yy = 0.55 * x7 + rng.normal(0, 0.1 * x7)
    sims.append(((yy / x7).mean(), yy.sum() / x7.sum(), (x7 * yy).sum() / (x7 ** 2).sum()))
sims = np.array(sims)
E.check("s22p07b", sims[:, 0].var() < sims[:, 1].var() < sims[:, 2].var(), "シミュレーションで重み 1/x² の分散が最小")

# ---- 大問8
y = np.array([12, 8, 20, 10])
xg = np.array([0, 0, 1, 1])
off = np.log(np.array([5000, 5000, 8000, 7000]))
# ニュートン法でポアソン回帰を当てはめる
beta = np.zeros(2)
X = np.c_[np.ones(4), xg]
for _ in range(50):
    mu8 = np.exp(X @ beta + off)
    beta = beta + np.linalg.solve(X.T @ (mu8[:, None] * X), X.T @ (y - mu8))
mu8 = np.exp(X @ beta + off)
E.check("s22p08a", np.allclose(mu8, [10, 10, 16, 14], atol=1e-6), f"当てはめ値 {np.round(mu8, 4)}")
Dv = 2 * np.sum(y * np.log(y / mu8) - (y - mu8))
E.choice("s22p08a", f"$D={f(Dv, 2)}$で、自由度2の5%点5.99を下回り", "当てはめ直した逸脱度")
T8 = np.array([[30, 5, 1], [20, 8, 3], [10, 6, 5]])
xs8, ys8 = [], []
for i in range(3):
    for j in range(3):
        xs8 += [i] * T8[i, j]; ys8 += [j] * T8[i, j]
xs8, ys8 = np.array(xs8), np.array(ys8)
dx = np.sign(xs8[:, None] - xs8[None, :]); dy = np.sign(ys8[:, None] - ys8[None, :])
Cc = np.sum((dx * dy) > 0) / 2; Dd = np.sum((dx * dy) < 0) / 2
E.num("s22p08b", rnd((Cc - Dd) / (Cc + Dd), 3), "全ての対を総当たりで比べた γ")
E.check("s22p08b", Cc == 960 and Dd == 322, f"C={Cc}, D={Dd}")

# ---- 大問9
cnt = {}
for w in (0, 1):
    for k in (0, 1):
        cnt[(w, k)] = (0.2 if w else 0.8) * (0.1 if k else 0.9)
pd = sum(v for (w, k), v in cnt.items() if w or k)
p1 = sum(v for (w, k), v in cnt.items() if w) / pd
p2 = cnt[(1, 1)] / sum(v for (w, k), v in cnt.items() if k)
E.choice("s22p09a", f"${f(p1, 3)}$と${f(p2, 3)}$", "同時確率の全列挙")

# ---- 大問10
M = np.array([20, 30, 25, 15, 10.0])
aa = np.array([2, 3, 1, 2, 2.0])
ph = aa.sum() / M.sum()
V10 = (1 - 5 / 200) / (5 * M.mean() ** 2) * np.var(aa - ph * M, ddof=1)
E.num("s22p10a", rnd(math.sqrt(V10), 4), "numpy の不偏分散で比推定の分散")
# 線形化（テイラー展開）による別表現：Var(ȳ_a/ȳ_M) ≈ (1-f)/n · S²_{a-pM}/M̄² と一致
sa2 = np.var(aa, ddof=1); sm2 = np.var(M, ddof=1); sam = np.cov(aa, M, ddof=1)[0, 1]
V10b = (1 - 5 / 200) / 5 * (sa2 - 2 * ph * sam + ph ** 2 * sm2) / M.mean() ** 2
E.check("s22p10a", abs(V10 - V10b) < 1e-12, "分散・共分散を使う線形化の式と一致")

# ---- 大問11
R = np.full((3, 3), 0.6) + 0.4 * np.eye(3)
w, V = np.linalg.eigh(R)
v1 = V[:, -1]
zv = np.array([1.2, 0.3, -0.6])
E.choice("s22p11a", f"${f(np.sum((zv - (v1 @ zv) * v1) ** 2), 2)}$", "numpy の固有ベクトルで射影")
E.check("s22p11a", abs(w[-1] - 2.2) < 1e-9, "最大固有値 2.2")
Lr = np.array([[0.8, 0.1], [0.7, 0.2], [0.1, 0.8], [0.2, 0.7]])
Vv = np.sum(np.var(Lr ** 2, axis=0))
E.choice("s22p11b", f"${f(Vv, 3)}$", "numpy で列ごとの 2 乗の分散の和")
th = np.pi / 4
L0 = Lr @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
E.check("s22p11b", np.allclose((L0 ** 2).sum(1), (Lr ** 2).sum(1)) and np.sum(np.var(L0 ** 2, axis=0)) < Vv, "回転で共通性不変・回転前の基準は小さい")
# 回転角を全探索するとバリマックス基準は回転後の解（角度 0）で最大
angs = np.linspace(-np.pi / 4, np.pi / 4, 721)
vals = [np.sum(np.var((Lr @ np.array([[np.cos(t_), -np.sin(t_)], [np.sin(t_), np.cos(t_)]])) ** 2, axis=0)) for t_ in angs]
E.check("s22p11b", abs(angs[int(np.argmax(vals))]) < 0.01, "角度の全探索で最大は与えた解")
R12 = np.array([[0.5, 0.2], [0.1, 0.4]])
Rfull = np.block([[np.eye(2), R12], [R12.T, np.eye(2)]])
E.check("s22p11c", np.all(np.linalg.eigvalsh(Rfull) > 0), "4 変数の相関行列が正定値")
cc = np.linalg.svd(R12, compute_uv=False)
E.num("s22p11c", rnd(cc[0], 2), "特異値分解で第1正準相関")
# 一般式 Rxx^-1 Rxy Ryy^-1 Ryx の固有値でも確認
ev = np.sort(np.linalg.eigvals(R12 @ R12.T).real)[::-1]
E.check("s22p11c", abs(math.sqrt(ev[0]) - cc[0]) < 1e-12, "固有値の平方根と一致")

# ---- 大問12：MA(1) をシミュレーションし、ε_3 を条件にした x_4 の条件付き平均を確認する代わりに、
# sympy で誤差を連立方程式から解き、予測式を求める
e1, e2, e3 = sp.symbols("e1 e2 e3")
sol = sp.solve([sp.Eq(53, 50 + e1), sp.Eq(47, 50 + e2 + sp.Rational(3, 5) * e1), sp.Eq(54, 50 + e3 + sp.Rational(3, 5) * e2)], [e1, e2, e3])
x4 = 50 + sp.Rational(3, 5) * sol[e3]
E.choice("s22p12a", f"$({f(float(x4), 2)},\\ 50.00)$", "sympy で誤差を解いて 1 期先・2 期先")
# 2 期先の予測が平均になること：シミュレーションで x_{t+2} と過去の値が無相関
eps = rng.normal(0, 1, 500001)
xx = 50 + eps[1:] + 0.6 * eps[:-1]
E.check("s22p12a", abs(np.corrcoef(xx[2:], xx[:-2])[0, 1]) < 0.01 and abs(np.corrcoef(xx[1:], xx[:-1])[0, 1] - 0.6 / 1.36) < 0.01,
        "シミュレーションで ρ1=θ/(1+θ²)・ρ2=0")

# ---- 大問13：平均と誤差分散が問題文どおりになるデータを作り、回帰（1次・2次）で平方和を分解する
m13 = np.array([20.0, 24.0, 30.0])
lev = np.repeat([20, 30, 40], 5)
ybar = np.repeat(m13, 5)
xlin = (lev - 30) / 10.0
Xl = np.c_[np.ones(15), xlin]
fit = Xl @ np.linalg.lstsq(Xl, ybar, rcond=None)[0]
ss_lin = np.sum((fit - ybar.mean()) ** 2)
E.choice("s22p13a", f"平方和${f(ss_lin, 1)}$、$F={f(ss_lin / 8.0, 2)}$", "1 次回帰の回帰平方和")
E.check("s22p13a", abs(stats.f.ppf(0.95, 1, 12) - 4.75) < 0.005, "F_0.05(1,12)=4.75")

# ---- 大問14
Sig = np.array([[100, 40, 30], [40, 64, 24], [30, 24, 36.0]])
E.check("s22p14a", np.all(np.linalg.eigvalsh(Sig) > 0), "Σ が正定値")
A = np.array([[1, 1, 0], [0, 0, 1.0]])
mS = A @ np.array([100, 80, 60.0]); SS = A @ Sig @ A.T
cm = mS[0] + SS[0, 1] / SS[1, 1] * (72 - mS[1]); cv = SS[0, 0] - SS[0, 1] ** 2 / SS[1, 1]
E.choice("s22p14a", f"${f(1 - stats.norm.cdf(round((210 - cm) / math.sqrt(cv), 2)), 3)}$", "行列で条件付き分布を求め scipy の Φ")
E.check("s22p14a", all(abs(stats.norm.cdf(z_) - v_) < 5e-5 for z_, v_ in [(0.77, 0.7794), (0.94, 0.8264), (1.32, 0.9066), (1.92, 0.9726)]), "Φ の値")
Z = rng.multivariate_normal([100, 80, 60], Sig, 2000000)
sel = np.abs(Z[:, 2] - 72) < 0.5
E.check("s22p14a", abs(((Z[sel, 0] + Z[sel, 1]) > 210).mean() - 0.174) < 0.01, f"シミュレーション {((Z[sel, 0] + Z[sel, 1]) > 210).mean():.3f}")

E.done()
