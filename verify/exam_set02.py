"""模擬試験 第2回の検算（generator/set02.py とは別の方法で正解を求め直す）。

使い方: .venv/bin/python verify/exam_set02.py
"""
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(2)
rng = np.random.default_rng(20261024)


def rd(x, nd):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def choices(iid):
    return E.item(iid)["choices"]


def nearest(iid, value):
    """数値の選択肢のうち value に最も近いものの番号"""
    vals = []
    for c in choices(iid):
        try:
            vals.append(float(str(c).replace("%", "").replace(" 円", "").replace(",", "")))
        except ValueError:
            vals.append(np.inf)
    return int(np.argmin([abs(v - value) for v in vals]))


# ---- s02p01a: sympy で PPV(q)=0.5 を解き、シミュレーションで確認
q = sp.symbols("q", positive=True)
ppv = sp.Rational(2, 1000) * sp.Rational(95, 100) / (sp.Rational(2, 1000) * sp.Rational(95, 100) + sp.Rational(998, 1000) * (1 - q))
qmin = sp.solve(sp.Eq(ppv, sp.Rational(1, 2)), q)[0]
E.num("s02p01a", rd(100 * float(qmin), 2), "sympy で PPV=0.5 となる特異度を解く（%）")
nsim = 5_000_000
fraud = rng.random(nsim) < 0.002
alert = np.where(fraud, rng.random(nsim) < 0.95, rng.random(nsim) < 1 - float(qmin))
E.check("s02p01a", abs(fraud[alert].mean() - 0.5) < 0.03, f"その特異度でのシミュレーションの陽性的中率 {fraud[alert].mean():.3f}")
E.check("s02p01a", float(ppv.subs(q, sp.Rational(99, 100))) < 0.2, "現状（特異度 99%）の陽性的中率は 2 割未満")

# ---- s02p02a: 対数の平均から幾何平均
g = np.exp(np.mean(np.log([1.20, 0.90, 1.15, 0.95]))) - 1
E.choice("s02p02a", f"{100 * g:.2f}%", "対数の平均の指数で求めた幾何平均")
E.check("s02p02a", abs(100 * g - 4.22) < 0.005, "幾何平均 4.22%")

# ---- s02p02b: scipy の2変量正規とモンテカルロ
cov = np.array([[25, -0.3 * 15], [-0.3 * 15, 9]])
mean = np.array([1.2, 0.6])
wv = np.array([0.6, 0.4])
p_exact = stats.norm.cdf(-4, wv @ mean, np.sqrt(wv @ cov @ wv))
xs = rng.multivariate_normal(mean, cov, size=400000)
p_mc = np.mean(xs @ wv < -4)
E.check("s02p02b", nearest("s02p02b", p_exact) == E.item("s02p02b")["answer"], f"scipy の正確な値 {p_exact:.5f} に最も近い選択肢が正解")
E.check("s02p02b", abs(p_mc - p_exact) < 0.002, f"モンテカルロ {p_mc:.5f}")

# ---- s02p02c: カイ二乗分布の p 値と棄却限界
stat = 24 * 25.0 / 16
E.check("s02p02c", stats.chi2.sf(stat, 24) < 0.05, f"片側 p 値 {stats.chi2.sf(stat, 24):.4f} < 0.05 で棄却")
E.check("s02p02c", stat < stats.chi2.ppf(0.975, 24) and stat < stats.chi2.ppf(0.95, 25), "誤答 1・4 の棄却限界では棄却されない（結論が異なる）")
E.choice("s02p02c", f"検定統計量は ${stat:.2f}$ で、$\\chi^2_{{24}}(0.05)={stats.chi2.ppf(0.95, 24):.2f}$ を超えるので $H_0$ を棄却する")

# ---- s02p03a: scipy の対数正規分布
ln = stats.lognorm(s=0.2, scale=100 * np.exp(0.05))
E.choice("s02p03a", f"中央値 {ln.median():.2f} 円、平均 {ln.mean():.2f} 円", "scipy の lognorm の中央値と平均")
smp = 100 * np.exp(rng.normal(0.05, 0.2, 1_000_000))
E.check("s02p03a", abs(smp.mean() - ln.mean()) < 0.1 and abs(np.median(smp) - ln.median()) < 0.1, f"シミュレーションの平均 {smp.mean():.2f}、中央値 {np.median(smp):.2f}")

# ---- s02p03b: 混合分布として生成したシミュレーションと sympy の積分
xx = sp.symbols("x", positive=True)
EL = sp.integrate(xx * sp.diff(1 - sp.Rational(2, 5) * sp.exp(-xx / 5), xx), (xx, 0, sp.oo))  # 0 の点確率は E に寄与しないので、連続部分 F'(x) に x を掛けて積分
lsim = np.where(rng.random(1_000_000) < 0.6, 0.0, rng.exponential(5, 1_000_000))
E.check("s02p03b", abs(float(EL) - 2.0) < 1e-12 and abs(lsim.mean() - 2.0) < 0.02 and abs(np.mean(lsim == 0) - 0.6) < 0.002,
        f"sympy の E[L]={EL}、シミュレーション平均 {lsim.mean():.3f}、P(L=0)={np.mean(lsim == 0):.3f}")
E.choice("s02p03b", f"$P(L=0)={1 - 0.4:.1f}$、$E[L]={float(EL):.1f}$", "点確率と期待値")

# ---- s02p04a: scipy の正確な z で求め直す
pa, pb, n = 192 / 2400, 240 / 2400, 2400
se = np.sqrt(stats.binom.var(n, pa) / n**2 + stats.binom.var(n, pb) / n**2)
lo = (pb - pa) - stats.norm.ppf(0.975) * se
E.num("s02p04a", rd(100 * lo, 2), "二項分布の分散と正確な z_{0.025} で求めた下限（%pt）")

# ---- s02p04b: 2群の二項分布からのシミュレーションで検出力
reps = 400000
xa = rng.binomial(2400, 0.08, reps) / 2400
xb = rng.binomial(2400, 0.10, reps) / 2400
se_hat = np.sqrt(0.08 * 0.92 / 2400 + 0.10 * 0.90 / 2400)
power_sim = np.mean((xb - xa) / se_hat > 1.96)
power_exact = stats.norm.sf(1.96 - 0.02 / se_hat)
E.check("s02p04b", nearest("s02p04b", power_exact) == E.item("s02p04b")["answer"], f"正規近似の正確な値 {power_exact:.4f} に最も近い選択肢が正解")
E.check("s02p04b", abs(power_sim - power_exact) < 0.02, f"二項分布のシミュレーションでの検出力 {power_sim:.4f}")

# ---- s02p04c: scipy のケンドールの tau から
tau = stats.kendalltau(range(1, 7), [2, 1, 3, 6, 4, 5]).statistic
nn = 6
z = tau * (nn * (nn - 1) / 2) / np.sqrt(nn * (nn - 1) * (2 * nn + 5) / 18)
E.num("s02p04c", rd(z, 2), f"scipy の tau={tau:.3f} から Z")

# ---- s02p05a: ポアソン分布の確率
p0 = stats.poisson.pmf(0, 24 * 0.25 * 0.25)
E.choice("s02p05a", f"{p0:.4f}", "scipy のポアソン確率（率 6/時 × 0.25 時）")
# 間引きのシミュレーション
sims = rng.poisson(24 * 0.25, 200000)
coup = rng.binomial(sims, 0.25)
E.check("s02p05a", abs(np.mean(coup == 0) - p0) < 0.004, f"注文数を発生させて間引いたシミュレーション {np.mean(coup == 0):.4f}")

# ---- s02p05b: 二項分布の正確な値とポアソン分布
pois = stats.poisson.sf(2, 2)
binom = stats.binom.sf(2, 2000, 0.001)
E.check("s02p05b", nearest("s02p05b", pois) == E.item("s02p05b")["answer"] and abs(pois - binom) < 0.001, f"ポアソン {pois:.4f}、二項 {binom:.4f}")
E.choice("s02p05b", f"{1 - 0.1353 * 5:.4f}", "表の e^{-2} での値")

# ---- s02p06a: 級内相関をもつ集落データのシミュレーションで分散比を求める
k, m, icc = 40, 20, 0.05
reps = 40000
b = rng.normal(0, np.sqrt(icc), (reps, k, 1))
e_ = rng.normal(0, np.sqrt(1 - icc), (reps, k, m))
var_cl = (b + e_).mean(axis=(1, 2)).var()
deff_sim = var_cl / (1 / (k * m))
E.check("s02p06a", nearest("s02p06a", 800 / deff_sim) == E.item("s02p06a")["answer"], f"シミュレーションの設計効果 {deff_sim:.3f}、有効標本サイズ {800 / deff_sim:.1f}")
E.choice("s02p06a", f"{800 / (1 + (m - 1) * icc):.0f}", "公式 1+(m-1)ρ")

# ---- s02p06b: 個人レベルの母集団を作って欠測させ、完全ケース平均と重みづけ平均を比べる（分数で厳密にも計算）
Wf = [Fraction(6, 10), Fraction(3, 10), Fraction(1, 10)]
rf = [Fraction(5, 10), Fraction(7, 10), Fraction(9, 10)]
yf = [Fraction(68, 10), Fraction(74, 10), Fraction(81, 10)]
cc = sum(w_ * r_ * y_ for w_, r_, y_ in zip(Wf, rf, yf)) / sum(w_ * r_ for w_, r_ in zip(Wf, rf))
adj = sum(w_ * y_ for w_, y_ in zip(Wf, yf))
E.num("s02p06b", rd(float(cc - adj), 3), f"分数で厳密に {cc - adj}")
Np = 2_000_000
tier = rng.choice(3, Np, p=[0.6, 0.3, 0.1])
ysat = np.array([6.8, 7.4, 8.1])[tier] + rng.normal(0, 1.5, Np)
resp = rng.random(Np) < np.array([0.5, 0.7, 0.9])[tier]
cc_sim = ysat[resp].mean()
adj_sim = sum(w_ * ysat[resp & (tier == h)].mean() for h, w_ in enumerate([0.6, 0.3, 0.1]))
E.check("s02p06b", abs((cc_sim - adj_sim) - float(cc - adj)) < 0.01, f"シミュレーションの偏り {cc_sim - adj_sim:.4f}")

# ---- s02p07a: 打ち切り尤度の数値最大化と数値ヘッセ行列
t = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12], float)
dlt = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])


def nll(lam):
    return -np.sum(dlt * stats.expon.logpdf(t, scale=1 / lam) + (1 - dlt) * stats.expon.logsf(t, scale=1 / lam))


res = optimize.minimize_scalar(nll, bounds=(1e-4, 2), method="bounded", options={"xatol": 1e-10})
lh = res.x
hstep = 1e-5
hess = (nll(lh + hstep) - 2 * nll(lh) + nll(lh - hstep)) / hstep**2
se7 = 1 / np.sqrt(hess)
E.choice("s02p07a", f"$\\hat\\lambda={lh:.4f}$、標準誤差 ${se7:.4f}$", "数値最大化と数値ヘッセ行列")

# ---- s02p08a: 指定の標準偏差をもつデータを作り、標準化して回帰し直す
n8 = 40
X = rng.normal(size=(n8, 2))
X = (X - X.mean(0)) / X.std(0) * np.array([2.4, 6.5])
noise = rng.normal(size=n8)
noise -= np.column_stack([np.ones(n8), X]) @ np.linalg.lstsq(np.column_stack([np.ones(n8), X]), noise, rcond=None)[0]
base = 2.10 + X @ np.array([0.85, 0.12])
target_var = 3.0**2 - base.var()
yv = base + noise / noise.std() * np.sqrt(target_var)
Zs = (X - X.mean(0)) / X.std(0)
ys = (yv - yv.mean()) / yv.std()
bstd = np.linalg.lstsq(np.column_stack([np.ones(n8), Zs]), ys, rcond=None)[0][1]
E.check("s02p08a", abs(yv.std() - 3.0) < 1e-9, "作ったデータの y の標準偏差が 3.0")
E.num("s02p08a", rd(bstd, 2), f"標準化したデータで回帰し直した係数 {bstd:.4f}")

# ---- s02p08b: 自己相関をもつ残差列を作り d と 1-d/2 を確かめる
e = np.zeros(40)
e[0] = rng.normal()
for i in range(1, 40):
    e[i] = 0.58 * e[i - 1] + rng.normal()
d_sim = np.sum(np.diff(e) ** 2) / np.sum(e**2)
r1 = np.sum(e[1:] * e[:-1]) / np.sum(e**2)
E.check("s02p08b", abs(d_sim - 2 * (1 - r1)) < 0.15, f"d≈2(1-r1) の近似（シミュレーション d={d_sim:.2f}, r1={r1:.2f}）")
d = 58.8 / 70
E.choice("s02p08b", f"$d={d:.2f}<d_L$ なので正の系列相関があると判断し、1次の自己相関はおよそ ${1 - d / 2:.2f}$", "d と 1-d/2")
E.check("s02p08b", d < 1.39, "d < d_L")
# 問題文の d_L・d_U（n=40、切片を除く説明変数 2）を、ダービン・ワトソンの上下限分布の 5% 点のシミュレーションで求め直す
# DW 行列の固有値 ν_j=2(1-cos(πj/n))、切片を含むパラメータ数 k=3 で
# d_L=Σ_{i=1}^{n-k} ν_i ξ_i^2/Σξ_i^2、d_U=Σ_{i=1}^{n-k} ν_{i+k-1} ξ_i^2/Σξ_i^2（許容誤差 0.005）
ndw, kdw = 40, 3
nu = np.sort(2 * (1 - np.cos(np.pi * np.arange(ndw) / ndw)))
xi2 = rng.standard_normal((400000, ndw - kdw)) ** 2
dL_sim = np.quantile(xi2 @ nu[1:ndw - kdw + 1] / xi2.sum(1), 0.05)
dU_sim = np.quantile(xi2 @ nu[kdw:ndw] / xi2.sum(1), 0.05)
E.check("s02p08b", abs(dL_sim - 1.39) < 0.005 and abs(dU_sim - 1.60) < 0.005, f"上下限分布の 5% 点 d_L={dL_sim:.3f}、d_U={dU_sim:.3f}")

# ---- s02p08c: sympy で変換式を展開し、シミュレーションでも推定を確かめる
b0, b1, b2, rho, x1t, x1s, x2t, x2s = sp.symbols("b0 b1 b2 rho x1t x1s x2t x2s")
lhs = (b0 + b1 * x1t + b2 * x2t) - rho * (b0 + b1 * x1s + b2 * x2s)
const = sp.expand(lhs).subs({x1t: 0, x1s: 0, x2t: 0, x2s: 0})
E.check("s02p08c", sp.simplify(const - b0 * (1 - rho)) == 0 and sp.simplify(sp.diff(lhs, x1t) - b1) == 0, "変換後の切片は b0(1-ρ)、傾きは b1 のまま")
E.choice("s02p08c", f"$\\hat\\beta_0={0.84 / (1 - 0.58):.2f}$、$\\hat\\beta_1={0.80:.2f}$", "切片を (1-ρ) で割り戻す")
T8, rho8 = 20000, 0.58
x1 = rng.normal(5, 2, T8)
x2 = rng.normal(15, 6, T8)
u = np.zeros(T8)
for t_ in range(1, T8):
    u[t_] = rho8 * u[t_ - 1] + rng.normal()
yy8 = 2.0 + 0.8 * x1 + 0.1 * x2 + u
Xs = np.column_stack([np.ones(T8 - 1), x1[1:] - rho8 * x1[:-1], x2[1:] - rho8 * x2[:-1]])
co = np.linalg.lstsq(Xs, yy8[1:] - rho8 * yy8[:-1], rcond=None)[0]
E.check("s02p08c", abs(co[0] / (1 - rho8) - 2.0) < 0.15 and abs(co[1] - 0.8) < 0.02, f"シミュレーション：切片/(1-ρ)={co[0] / (1 - rho8):.3f}、傾き {co[1]:.3f}")

# ---- s02p09a: 正確な指数関数で確率・オッズ比・リスク比
ex = lambda eta: 1 / (1 + np.exp(-eta))
P = {(cc, ll): ex(-2.2 + 0.6 * cc + 0.8 * ll) for cc in (0, 1) for ll in (0, 1)}
odds = lambda p: p / (1 - p)
or1, or0 = odds(P[1, 1]) / odds(P[0, 1]), odds(P[1, 0]) / odds(P[0, 0])
d1, d0 = P[1, 1] - P[0, 1], P[1, 0] - P[0, 0]
E.check("s02p09a", abs(or1 - or0) < 1e-12 and abs(or1 - 1.822) < 0.001, f"オッズ比は層によらず {or1:.4f}（誤答 1 は誤り）")
E.check("s02p09a", d1 > d0 + 0.02, f"確率の差 L=1: {d1:.4f} > L=0: {d0:.4f}（誤答 0・3 は誤り、正解 4）")
E.check("s02p09a", abs(P[1, 1] / P[0, 1] - 1.82) > 0.1 and abs(P[1, 0] / P[0, 0] - 1.82) > 0.1, "リスク比は 1.82 ではない（誤答 2 は誤り）")
E.check("s02p09a", E.item("s02p09a")["answer"] == 4, "正解の番号")

# ---- s02p09b: 介入した母集団のシミュレーションと正確な値
ace = 0.3 * d1 + 0.7 * d0
E.num("s02p09b", rd(100 * ace, 1), f"正確な指数関数でのバックドア調整 {100 * ace:.3f}")
Nn = 2000000
L = rng.random(Nn) < 0.3
y1 = rng.random(Nn) < np.where(L, P[1, 1], P[1, 0])
y0 = rng.random(Nn) < np.where(L, P[0, 1], P[0, 0])
E.check("s02p09b", abs(100 * (y1.mean() - y0.mean()) - 100 * ace) < 0.15, f"do 介入のシミュレーション {100 * (y1.mean() - y0.mean()):.2f}")

# ---- s02p10a: 計画行列の階数から自由度を求めて F
A = np.repeat(np.arange(3), 8)
B = np.tile(np.repeat(np.arange(2), 4), 3)
dA = (A[:, None] == np.arange(3)).astype(float)
dB = (B[:, None] == np.arange(2)).astype(float)
cell = (A * 2 + B)[:, None] == np.arange(6)
rk_add = np.linalg.matrix_rank(np.column_stack([dA, dB]))
rk_full = np.linalg.matrix_rank(cell.astype(float))
df_ab, df_e = rk_full - rk_add, 24 - rk_full
F10 = (16 / df_ab) / ((166 - 48 - 30 - 16) / df_e)
E.choice("s02p10a", f"{F10:.2f}", f"計画行列の階数で自由度（A×B {df_ab}、残差 {df_e}）")

# ---- s02p11a: 乱数の相関行列で「負荷量 = √λ × 固有ベクトル」を確かめてから計算
Xr = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))
R = np.corrcoef(Xr.T)
lam, vec = np.linalg.eigh(R)
zsc = ((Xr - Xr.mean(0)) / Xr.std(0)) @ vec[:, -1]
emp = np.array([np.corrcoef(zsc, Xr[:, j])[0, 1] for j in range(5)])
E.check("s02p11a", np.allclose(emp, np.sqrt(lam[-1]) * vec[:, -1]), "標本の主成分得点と変数の相関が √λ a_j に一致")
E.choice("s02p11a", f"{np.sqrt(2.35) * 0.52:.3f}", "√2.35 × 0.52")

# ---- s02p11b: 精度行列を逆行列に戻し、共分散から偏相関を計算
Om = np.array([[1.6, -0.6, 0.4], [-0.6, 2.0, -0.9], [0.4, -0.9, 1.5]])
Sg = np.linalg.inv(Om)
cond_cov = Sg[np.ix_([0, 2], [0, 2])] - np.outer(Sg[[0, 2], 1], Sg[1, [0, 2]]) / Sg[1, 1]
pcor = cond_cov[0, 1] / np.sqrt(cond_cov[0, 0] * cond_cov[1, 1])
E.num("s02p11b", rd(pcor, 3), "条件付き共分散行列からの偏相関")
E.check("s02p11b", np.all(np.linalg.eigvalsh(Om) > 0), "精度行列が正定値（分散共分散行列の逆行列として成り立つ）")

# ---- s02p11c: 標準化残差行列の特異値分解
tab = np.array([[60, 30, 30], [20, 30, 30]], float)
chi2v = stats.chi2_contingency(tab, correction=False)[0]
Pm = tab / tab.sum()
rr, cc = Pm.sum(1), Pm.sum(0)
Sres = (Pm - np.outer(rr, cc)) / np.sqrt(np.outer(rr, cc))
sv = np.linalg.svd(Sres, compute_uv=False)
E.check("s02p11c", abs(np.sum(sv**2) - chi2v / tab.sum()) < 1e-12 and np.sum(sv > 1e-10) == 1, f"特異値の2乗和 {np.sum(sv ** 2):.4f}、非零の特異値 {np.sum(sv > 1e-10)} 個")
E.choice("s02p11c", f"{np.sum(sv ** 2):.4f}、{np.sum(sv > 1e-10)}", "SVD による総イナーシャと次元数")

# ---- s02p12a: sympy で解き、シミュレーションで自己相関を確認
th = sp.symbols("theta", real=True)
roots = sp.solve(sp.Eq(th / (1 + th**2), sp.Rational(2, 5)), th)
inv = [float(r_) for r_ in roots if abs(float(r_)) < 1][0]
s2 = 2.5 / (1 + inv**2)
E.choice("s02p12a", f"$\\theta={inv:.2f}$、$\\sigma^2={s2:.2f}$", "sympy の解のうち反転可能なもの")
eps = rng.normal(0, np.sqrt(s2), 400001)
xt = eps[1:] + inv * eps[:-1]
acf1 = np.corrcoef(xt[1:], xt[:-1])[0, 1]
E.check("s02p12a", abs(acf1 - 0.4) < 0.01 and abs(xt.var() - 2.5) < 0.03, f"シミュレーションの自己相関 {acf1:.3f}、分散 {xt.var():.3f}")

# ---- s02p13a: 数値最大化で M、受容棄却法のシミュレーションで提案回数
M = -optimize.minimize_scalar(lambda v: -stats.beta.pdf(v, 2, 3), bounds=(0, 1), method="bounded").fun
counts = []
for _ in range(200):
    acc, tries = 0, 0
    while acc < 1000:
        yv = rng.random(3000)
        uv = rng.random(3000)
        ok = uv <= stats.beta.pdf(yv, 2, 3) / M
        need = 1000 - acc
        idx = np.flatnonzero(ok)
        if len(idx) >= need:
            tries += idx[need - 1] + 1
            acc = 1000
        else:
            tries += 3000
            acc += len(idx)
    counts.append(tries)
E.choice("s02p13a", f"{1000 * M:.0f}", "数値最大化した M × 1000")
E.check("s02p13a", abs(np.mean(counts) - 1000 * M) < 10, f"シミュレーションの平均提案回数 {np.mean(counts):.1f}")

E.done()
