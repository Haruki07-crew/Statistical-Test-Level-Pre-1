"""模擬試験 第4回（set04）の検算。generator/set04.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・方程式の数値解）で正解を求め直す。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(4)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value, what=""):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']} {what}")


def table_ok(iid, pairs, tol):
    bad = [(a, b) for a, b in pairs if abs(a - b) > tol]
    E.check(iid, not bad, f"問題文の数表の値が正しい（許容 {tol}）: {bad}")


# ---- 大問1
# (a) scipy の二項分布で尤度を作り事後確率
LA = stats.binom.pmf(2, 20, 0.02)
LB = stats.binom.pmf(2, 20, 0.06)
post = 0.3 * LB / (0.3 * LB + 0.7 * LA)
E.num("s04p01a", rnd(post, 3), "scipy の二項確率でベイズの定理")
table_ok("s04p01a", [(0.98 ** 18, 0.6951), (0.94 ** 18, 0.3283)], 5e-5)
sim = rng.random(400_000) < 0.3
dfc = rng.binomial(20, np.where(sim, 0.06, 0.02))
E.check("s04p01a", abs(sim[dfc == 2].mean() - post) < 0.01, f"シミュレーション {sim[dfc == 2].mean():.4f}")
# (b) scipy の負の二項分布（r=3, p=5/6）で平均・分散が一致し、P(X=0)
nb = stats.nbinom(3, 5 / 6)
E.check("s04p01b", abs(nb.mean() - 0.6) < 1e-12 and abs(nb.var() - 0.72) < 1e-12, "nbinom(3,5/6) の平均0.6・分散0.72")
E.choice("s04p01b", f"${f(nb.pmf(0), 3)}$", "nbinom.pmf(0)")
table_ok("s04p01b", [(math.exp(-0.6), 0.5488)], 5e-5)
# (c) 期待度数を行列演算で求め、調整済み残差を計算
O = np.array([[18, 382], [22, 378], [40, 360]], float)
Nn = O.sum()
Ex = np.outer(O.sum(1), O.sum(0)) / Nn
adj = (O - Ex) / np.sqrt(Ex * np.outer(1 - O.sum(1) / Nn, 1 - O.sum(0) / Nn))
E.num("s04p01c", rnd(adj[2, 0], 2), "行列演算で調整済み残差（夜・不良）")
E.check("s04p01c", abs(adj[2, 0] + adj[2, 1]) < 1e-12, "2列の表では同じ行の調整済み残差は符号違い（整合性）")

# ---- 大問2
# (a) sympy で分布関数 P(aV^3<=w) を微分
w, a, v = sp.symbols("w a v", positive=True)
FW = 1 - sp.exp(-((w / a) ** sp.Rational(1, 3)) ** 2 / 64)
fW = sp.diff(FW, w)
target = w ** sp.Rational(-1, 3) * sp.exp(-(w / a) ** sp.Rational(2, 3) / 64) / (96 * a ** sp.Rational(2, 3))
E.check("s04p02a", sp.simplify(fW - target) == 0, "分布関数の微分が正解の式と一致")
E.check("s04p02a", abs(float(sp.integrate(target.subs(a, 2), (w, 0, sp.oo))) - 1) < 1e-9, "正解の密度の積分が 1")
E.check("s04p02a", E.item("s04p02a")["answer"] == 3 and "96" in E.item("s04p02a")["choices"][3], "正解の選択肢は係数 1/96")
# (b) 正確な q とモンテカルロ
q = stats.weibull_min(2, scale=8).sf(12)
Nreq = math.ceil(q * (1 - q) / 0.005 ** 2)
E.choice("s04p02b", f"${Nreq}$", "scipy weibull_min の生存関数から N")
table_ok("s04p02b", [(math.exp(-2.25), 0.1054)], 5e-5)
est = (stats.weibull_min(2, scale=8).rvs(size=(400, Nreq), random_state=rng) > 12).mean(1)
E.check("s04p02b", abs(est.std() - 0.005) < 0.0006, f"N={Nreq} での推定値の標準偏差 {est.std():.5f} ≈ 0.005")

# ---- 大問3
# (a) sympy で確率母関数の積を展開
s = sp.symbols("s")
G = (sp.Rational(9, 10) + sp.Rational(1, 10) * s) ** 2 * sp.exp(sp.Rational(2, 5) * (s - 1))
c2 = sp.series(G, s, 0, 3).removeO().coeff(s, 2)
E.num("s04p03a", rnd(float(c2), 4), "sympy で G(s) の s^2 の係数")
conv = sum(stats.binom.pmf(k, 2, 0.1) * stats.poisson.pmf(2 - k, 0.4) for k in range(3))
E.check("s04p03a", abs(conv - float(c2)) < 1e-12, "scipy でたたみ込み")
table_ok("s04p03a", [(math.exp(-0.4), 0.6703)], 5e-5)
# (b) |p̂−p| = z√(p(1−p)/n) を p について数値的に解く（ウィルソン区間の定義）
z = 1.96
g = lambda p: (0.04 - p) ** 2 - z ** 2 * p * (1 - p) / 50
up = optimize.brentq(g, 0.04, 0.999)
E.num("s04p03b", rnd(up, 3), "スコア検定の境界を brentq で解いた上限")
E.check("s04p03b", abs(stats.binomtest(2, 50).proportion_ci(0.95, method="wilson").high - up) < 1e-4, "scipy binomtest の Wilson 区間（z=1.95996）とほぼ一致")

# ---- 大問4
# (a) 三角分布の裾を sympy で積分し、シミュレーションでも確認
s_ = sp.symbols("s_")
ex_tail = 2 * sp.integrate((2 - s_) / 4, (s_, sp.Rational(3, 2), 2))
E.choice("s04p04a", f"${f(float(ex_tail), 4)}$", "sympy で三角分布の裾を積分")
Ssim = rng.uniform(-1, 1, size=(2, 1_000_000)).sum(0)
E.check("s04p04a", abs((np.abs(Ssim) > 1.5).mean() - float(ex_tail)) < 0.001, f"シミュレーション {(np.abs(Ssim) > 1.5).mean():.4f}")
table_ok("s04p04a", [(stats.norm.cdf(1.84), 0.9671)], 5e-5)
# (b) 指定の平均・標準偏差をもつデータを作り scipy の Welch 検定
def data(n, m, sd):
    x = rng.normal(size=n)
    return (x - x.mean()) / x.std(ddof=1) * sd + m
res = stats.ttest_ind(data(8, 101.2, 2.0), data(12, 98.9, 4.5), equal_var=False)
E.choice("s04p04b", f"t={f(res.statistic, 2)},\\ \\nu\\approx {f(res.df, 1)}", "scipy ttest_ind(equal_var=False) の t と df")

# ---- 大問5
# (a) 上限は達成される（0 と 3 の 2 点分布）ことを確認し、ポアソンの値は上限より小さい
two_pt = {0: 0.75, 3: 0.25}
E.check("s04p05a", abs(sum(k * p for k, p in two_pt.items()) - 0.75) < 1e-12 and two_pt[3] == 0.25,
        "平均0.75の2点分布で P(X≥3)=0.25 が達成される（上限が最良）")
E.choice("s04p05a", f"${f(0.75 / 3, 3)}$", "E[X]/a")
E.check("s04p05a", abs(stats.poisson(0.75).sf(2) - 0.0404) < 2e-4, "ポアソンの P(X≥3)（誤答の値）")
table_ok("s04p05a", [(math.exp(-0.75), 0.4724)], 5e-5)
# (b) 条件付き分布 X1|T=9 ~ B(9,1/12) と、シミュレーションでの条件付き頻度
rb = stats.binom.pmf(0, 9, 1 / 12)
E.choice("s04p05b", f"${f(rb, 3)}$", "binom.pmf(0, 9, 1/12)")
X = rng.poisson(0.75, size=(600_000, 12))
T = X.sum(1)
E.check("s04p05b", abs((X[T == 9, 0] == 0).mean() - rb) < 0.01, f"シミュレーションで P(X1=0|T=9)={(X[T == 9, 0] == 0).mean():.4f}")
lam = 0.5
EU = sum(stats.poisson(12 * lam).pmf(t) * (11 / 12) ** t for t in range(200))
E.check("s04p05b", abs(EU - math.exp(-lam)) < 1e-12, "(11/12)^T が e^{-λ} の不偏推定量（λ=0.5）")

# ---- 大問6
# (a) scipy の power_divergence（対数尤度比）
obs = np.array([18, 12, 20])
exp = 50 * np.array([1000, 1500, 2500]) / 5000
Gs, pv = stats.power_divergence(obs, exp, lambda_="log-likelihood")
E.choice("s04p06a", f"$G={f(Gs, 2)}$で、$5.991$を超えるので", "power_divergence の G と判定")
E.check("s04p06a", Gs > stats.chi2(2).ppf(0.95) and abs(stats.chi2(2).ppf(0.95) - 5.991) < 5e-4 and abs(stats.chi2(3).ppf(0.95) - 7.815) < 5e-4
        and abs(stats.chi2(2).ppf(0.99) - 9.210) < 5e-4, "上側点の値と棄却の判定")
table_ok("s04p06a", [(math.log(1.8), 0.5878), (math.log(0.8), -0.2231)], 5e-5)
# (b) プロビットの逆推定：scipy の norm.ppf と方程式の数値解
xq = optimize.brentq(lambda x: stats.norm.cdf(-6.0 + 0.025 * x) - 0.10, 0, 600)
E.choice("s04p06b", f"${f(xq, 1)}$", "brentq で Φ(β0+β1x)=0.1 を解く")
table_ok("s04p06b", [(stats.norm.ppf(0.9), 1.2816), (stats.norm.ppf(0.95), 1.645), (math.log(9), 2.1972)], 5e-4)

# ---- 大問7：生成行列の零空間から定常分布
lam7, mu7 = 0.05, 0.5
Q = np.array([[-2 * lam7, 2 * lam7, 0], [mu7, -(mu7 + lam7), lam7], [0, mu7, -mu7]])
A = np.vstack([Q.T, np.ones(3)])
pi = np.linalg.lstsq(A, np.array([0, 0, 0, 1.0]), rcond=None)[0]
E.num("s04p07a", rnd(pi[2], 4), "生成行列 Q の πQ=0 を解いた")
# 連続時間のシミュレーション
t, st, tin = 0.0, 0, np.zeros(3)
for _ in range(300_000):
    rate = [2 * lam7, lam7 + mu7, mu7][st]
    dt = rng.exponential(1 / rate)
    tin[st] += dt
    if st == 0:
        st = 1
    elif st == 2:
        st = 1
    else:
        st = 2 if rng.random() < lam7 / (lam7 + mu7) else 0
E.check("s04p07a", abs(tin[2] / tin.sum() - pi[2]) < 0.002, f"シミュレーション {tin[2] / tin.sum():.4f}")

# ---- 大問8：scipy の正確な順位和検定
A8, B8 = [412, 455, 498], [305, 360, 389, 421, 440]
mw = stats.mannwhitneyu(A8, B8, alternative="greater", method="exact")
E.choice("s04p08a", f"${f(mw.pvalue, 3)}$", "mannwhitneyu(method='exact')")
cnt = sum(1 for c in combinations(range(1, 9), 3) if sum(c) >= 19)
E.check("s04p08a", cnt == 4, "総当たりで順位和19以上は4通り")
table_ok("s04p08a", [(stats.norm.cdf(1.64), 0.9495)], 5e-5)

# ---- 大問9
# 平均30・Sxx=160 の x（38 を含む）と、残差標準偏差 12 になる y を作って行列計算
x9 = np.array([38, 22] + [30.0] * 18)
dev = x9 - x9.mean()
extra = 160 - (dev ** 2).sum()
x9[2], x9[3] = 30 + math.sqrt(extra / 2), 30 - math.sqrt(extra / 2)
assert abs(x9.mean() - 30) < 1e-12 and abs(((x9 - 30) ** 2).sum() - 160) < 1e-9
Xm = np.column_stack([np.ones(20), x9])
H = Xm @ np.linalg.inv(Xm.T @ Xm) @ Xm.T
e = rng.normal(size=20)
e = e - H @ e
e = e / np.sqrt((e ** 2).sum() / 18) * 12
y9 = -300 + 25 * x9 + e
beta = np.linalg.lstsq(Xm, y9, rcond=None)[0]
s2 = ((y9 - Xm @ beta) ** 2).sum() / 18
x0 = np.array([1, 36])
var_pred = s2 * (1 + x0 @ np.linalg.inv(Xm.T @ Xm) @ x0)
up = x0 @ beta + stats.t(18).ppf(0.975) * math.sqrt(var_pred)
E.num("s04p09a", rnd(up, 1), "行列計算と scipy の t 分布で予測区間の上限")
E.check("s04p09a", abs(stats.t(18).ppf(0.975) - 2.101) < 5e-4 and abs(stats.t(19).ppf(0.975) - 2.093) < 5e-4
        and abs(stats.t(18).ppf(0.95) - 1.734) < 5e-4, "t 表の値")
# (c) 正確な有限 n の分散からの有効標本サイズとシミュレーション
phi, n = 0.6, 120
k = np.arange(n)
R = phi ** np.abs(k[:, None] - k[None, :])
neff = n * n / R.sum()
nearest("s04p09b", neff, "（有限 n の正確な分散）")
nearest("s04p09b", n * (1 - phi) / (1 + phi), "（n→∞ の極限）")
u = rng.normal(size=(20_000, n + 200))
ee = np.zeros_like(u)
ee[:, 0] = u[:, 0] / math.sqrt(1 - phi ** 2)
for t_ in range(1, n + 200):
    ee[:, t_] = phi * ee[:, t_ - 1] + u[:, t_]
sim_ratio = ee[:, 200:].mean(1).var() / (1 / (1 - phi ** 2) / n)
E.check("s04p09b", abs(n / sim_ratio - neff) < 2.5, f"シミュレーションの有効標本サイズ {n / sim_ratio:.1f}")

# ---- 大問10
# (a) 平均の分散 (1−n/N)S²/n から半幅を直接計算し、条件を満たす最小の n を探す
half = lambda n: 1.96 * math.sqrt((1 - n / 50) * 40 ** 2 / n)
n_min = next(n for n in range(1, 51) if half(n) <= 15)
E.choice("s04p10a", f"${n_min}$", "半幅 1.96√((1−n/N)S²/n) ≤ 15 となる最小の n を探索")
E.check("s04p10a", half(n_min) <= 15 < half(n_min - 1), "1 つ少ないと半幅が 15 を超える")
# (b) 変換 y/√x = β√x + 誤差（等分散）に通常の最小二乗法
xw = np.array([10, 20, 40, 50, 80.0])
yw = np.array([32, 58, 130, 148, 252.0])
bw = np.linalg.lstsq((np.sqrt(xw))[:, None], yw / np.sqrt(xw), rcond=None)[0][0]
E.choice("s04p10b", f"${f(bw, 3)}$", "√x で割って変換した OLS")
# (c) 行列のサンドイッチ公式 (X'X)^{-1} X' diag(e²) X (X'X)^{-1}
xh = np.array([10, 20, 30, 40, 50.0])
eh = np.array([3, -6, -3, 12, -6.0])
Xh = np.column_stack([np.ones(5), xh])
E.check("s04p10c", np.allclose(Xh.T @ eh, 0), "与えた残差が最小二乗の正規方程式を満たす（X'e=0）")
XtXi = np.linalg.inv(Xh.T @ Xh)
V = XtXi @ Xh.T @ np.diag(eh ** 2) @ Xh @ XtXi
E.choice("s04p10c", f"${f(math.sqrt(V[1, 1]), 3)}$", "行列のサンドイッチ公式")
E.check("s04p10c", f(math.sqrt((eh ** 2).sum() / 3 * XtXi[1, 1]), 3) == "0.279", "通常の標準誤差（誤答）")

# ---- 大問11：4 回の試作の計画行列から効果と別名を確認
Am, Bm, Cm = np.array([1, -1, -1, 1]), np.array([-1, 1, -1, 1]), np.array([-1, -1, 1, 1])  # a, b, c, abc
yv = np.array([20, 26, 18, 30.0])
E.check("s04p11a", np.all(Am * Bm * Cm == 1), "4 回の試作が I=ABC を満たす")
E.check("s04p11a", np.array_equal(Am, Bm * Cm) and not np.array_equal(Am, Am * Bm), "A の列と BC の列が一致（別名）、AB とは一致しない")
effA = yv[Am == 1].mean() - yv[Am == -1].mean()
E.choice("s04p11a", f"推定値{f(effA, 1)}、別名は交互作用BC", "計画行列から A の効果")
coef = np.linalg.lstsq(np.column_stack([np.ones(4), Am, Bm, Cm]), yv, rcond=None)[0]
E.check("s04p11a", abs(2 * coef[1] - effA) < 1e-12, "回帰係数の 2 倍が効果と一致")

# ---- 大問12
# (a) 固有ベクトルの単位長と主成分得点
e1 = np.array([0.7, 0.5, 0.5, 0.1])
zz = (np.array([406, 2.08, 49.2, 11.0]) - np.array([400, 2.0, 50, 10])) / np.array([5, 0.1, 2, 0.5])
E.choice("s04p12a", f"${f(zz @ e1, 2)}$", "numpy で標準化と内積")
E.check("s04p12a", abs(np.linalg.norm(e1) - 1) < 1e-12 and abs(math.sqrt(2.6) - 1.6125) < 5e-5, "固有ベクトルの長さ1・√2.6 の値")
# (b) 対応分析：標準化残差行列の特異値分解から行の主座標を求め、その距離を計算
Nmat = np.array([[30, 10, 10], [10, 20, 20], [20, 30, 50.0]])
P = Nmat / Nmat.sum()
rmass, cmass = P.sum(1), P.sum(0)
Smat = (P - np.outer(rmass, cmass)) / np.sqrt(np.outer(rmass, cmass))
U, sv, Vt = np.linalg.svd(Smat)
Frow = (U * sv) / np.sqrt(rmass)[:, None]
dist = np.linalg.norm(Frow[0] - Frow[1])
E.choice("s04p12b", f"${f(dist, 3)}$", "SVD の行主座標間のユークリッド距離")
E.check("s04p12b", abs((rmass * (Frow ** 2).sum(1)).sum() - (sv ** 2).sum()) < 1e-12, "総イナーシャ = 特異値の2乗和（整合性）")
# (c) 期待損失が等しくなる点を数値的に解く
lossG = lambda c: 0.95 * 1 * stats.norm(50, 4).pdf(c)
lossD = lambda c: 0.05 * 10 * stats.norm(58, 4).pdf(c)
cth = optimize.brentq(lambda c: lossD(c) - lossG(c), 40, 70)
E.choice("s04p12c", f"${f(cth, 2)}$", "brentq で π_D c_m f_D = π_G c_f f_G を解く")
table_ok("s04p12c", [(math.log(1.9), 0.6419), (math.log(19), 2.9444), (math.log(10), 2.3026)], 5e-5)
# 境界の左右で期待損失最小の判定が切り替わる
E.check("s04p12c", lossD(cth + 0.5) > lossG(cth + 0.5) and lossD(cth - 0.5) < lossG(cth - 0.5), "境界の右で不良と判定")

# ---- 大問13：個票に展開して IPW を計算し、標準化とも比べる
rows = []
for zv, xv, nn, yv in [(1, 1, 50, 10), (1, 0, 150, 45), (0, 1, 200, 10), (0, 0, 100, 8)]:
    rows += [(zv, xv, 1)] * yv + [(zv, xv, 0)] * (nn - yv)
D = np.array(rows)
Z, Xt, Y = D[:, 0], D[:, 1], D[:, 2]
ps = np.array([Xt[Z == zv].mean() for zv in Z])
ipw = (Xt * Y / ps).mean() - ((1 - Xt) * Y / (1 - ps)).mean()
E.choice("s04p13a", f"${f(ipw, 3)}$", "個票の IPW")
std = sum((Z == zv).mean() * (Y[(Z == zv) & (Xt == 1)].mean() - Y[(Z == zv) & (Xt == 0)].mean()) for zv in (0, 1))
E.check("s04p13a", abs(std - ipw) < 1e-12, "バックドア調整（標準化）と一致")

E.done()
