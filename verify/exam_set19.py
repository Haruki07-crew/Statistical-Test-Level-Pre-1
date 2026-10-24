"""模擬試験 第19回（set19）の検算。generator/set19.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(19)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


def table_ok(iid, given, true, tol, what):
    E.check(iid, abs(given - true) <= tol, f"{what}: 問題文 {given} / 正確な値 {true:.5f}")


# ---- 大問1
# (a) 正確な二項分布で、候補の n ごとに「有意水準 5% 以下の棄却域」と p=0.08 での検出力を計算し、正規近似の n が妥当か確かめる
table_ok("s19p01a", 1.645, stats.norm.ppf(0.95), 0.0006, "z_0.05")
table_ok("s19p01a", 0.842, stats.norm.ppf(0.80), 0.0006, "z_0.20")
from scipy.optimize import brentq  # noqa: E402
root = brentq(lambda m: stats.norm.sf((0.05 + stats.norm.ppf(0.95) * np.sqrt(0.0475 / m) - 0.08) / np.sqrt(0.0736 / m)) - 0.8, 50, 2000)
E.choice("s19p01a", f"${math.ceil(root)}$", f"正規近似の検出力の式を数値的に解く（{root:.2f}）")
n_ = 383
crit = 0.05 + stats.norm.ppf(0.95) * np.sqrt(0.05 * 0.95 / n_)
pw = stats.norm.sf((crit - 0.08) / np.sqrt(0.08 * 0.92 / n_))
E.check("s19p01a", 0.799 < pw < 0.81, f"n=383 の正規近似の検出力 {pw:.4f}")
c_ex = stats.binom.ppf(0.95, n_, 0.05)
E.check("s19p01a", 0.75 < stats.binom.sf(c_ex, n_, 0.08) < 0.86, f"正確な二項分布での検出力 {stats.binom.sf(c_ex, n_, 0.08):.3f}（近似が妥当）")
# (b) 母集団を作り、結果依存抽出の標本でロジスティック回帰を当てはめて切片のずれを確かめる
N = 400_000
xx = rng.uniform(20, 100, N)
b0_true = 2.0 - np.log(10)
yy = rng.random(N) < expit(b0_true - 0.06 * xx)
keep = yy | (rng.random(N) < 0.10)
X = np.column_stack([np.ones(keep.sum()), xx[keep]])
Y = yy[keep].astype(float)
beta = np.zeros(2)
for _ in range(30):  # ニュートン法
    pr = expit(X @ beta)
    W = pr * (1 - pr)
    beta += np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (Y - pr))
E.check("s19p01b", abs(beta[0] - 2.0) < 0.08 and abs(beta[1] + 0.06) < 0.002,
        f"抽出データの推定 β0={beta[0]:.3f}（母集団 {b0_true:.3f}+log10）、β1={beta[1]:.4f}")
E.choice("s19p01b", f"${f(expit(b0_true - 0.06 * 60), 4)}$", "母集団の真の切片で確率")
E.check("s19p01b", abs(math.exp(1.6) - 4.9530) < 5e-5 and abs(math.log(10) - 2.3026) < 5e-5, "e^1.6 と log10")

# ---- 大問2
data = np.array([1, 0, 2, 0, 1, 3, 0, 1, 0, 2, 1, 1])
D = ((data - data.mean()) ** 2).sum() / data.mean()
table_ok("s19p02a", 19.68, stats.chi2.ppf(0.95, 11), 0.006, "χ²_{0.05}(11)")
table_ok("s19p02a", 21.03, stats.chi2.ppf(0.95, 12), 0.006, "χ²_{0.05}(12)")
table_ok("s19p02a", 4.575, stats.chi2.ppf(0.05, 11), 0.0006, "χ²_{0.95}(11)")
E.choice("s19p02a", f"$D={f(D, 1)}$で19.68より小さいので、帰無仮説を棄却しない", "np で D を計算し上側点と比較")
E.check("s19p02a", stats.chi2.sf(D, 11) > 0.05, f"p 値 {stats.chi2.sf(D, 11):.3f} > 0.05")
# 帰無仮説のもとで D の上側 5% がほぼ 19.68 になることをシミュレーションで（λ=1、n=12）
sims = rng.poisson(1.0, size=(200_000, 12))
mm = sims.mean(1)
ok = mm > 0
Ds = ((sims - mm[:, None]) ** 2).sum(1)[ok] / mm[ok]
E.check("s19p02a", 0.03 < np.mean(Ds > 19.68) < 0.07, f"シミュレーションでの棄却率 {np.mean(Ds > 19.68):.3f}")
# (b) 数値最適化で対数尤度を最大化
from scipy.optimize import minimize_scalar  # noqa: E402
nll = lambda l: -(60 * np.log(l) - 40 * l - 40 * np.log(1 - np.exp(-l)))
lh = minimize_scalar(nll, bounds=(0.01, 5), method="bounded").x
it = E.item("s19p02b")
vals2 = [float(c.strip("$")) for c in it["choices"]]
E.check("s19p02b", int(np.argmin([abs(v - lh) for v in vals2])) == it["answer"], f"対数尤度の最大点 {lh:.4f} に最も近い選択肢が正解")
for lv, ev in ((0.5, 0.6065), (0.667, 0.5132), (0.874, 0.4173), (1.165, 0.3119), (1.5, 0.2231)):
    table_ok("s19p02b", ev, math.exp(-lv), 0.00006, f"e^-{lv}")
# (c) シミュレーション：不正の有無、ルール A,B の同時分布を表どおりに作る
nsim = 20_000_000
fraud = rng.random(nsim) < 0.002
u = rng.random(nsim)
# 不正：both 0.75, A のみ 0.15, B のみ 0.05, なし 0.05 ／ 正常：both 0.004, A のみ 0.016, B のみ 0.026, なし 0.954
bothA = np.where(fraud, u < 0.75, u < 0.004)
E.check("s19p02c", abs(fraud[bothA].mean() - 0.2731) < 0.01, f"シミュレーション {fraud[bothA].mean():.4f}")
post = 0.002 * 0.75 / (0.002 * 0.75 + 0.998 * 0.004)
E.choice("s19p02c", f"${f(post, 3)}$", "ベイズの定理（直接）")
E.check("s19p02c", 0.75 <= min(0.90, 0.80) and 0.004 <= min(0.02, 0.03) and 0.90 + 0.80 - 0.75 <= 1 and 0.02 + 0.03 - 0.004 <= 1, "表の同時陽性率が周辺の陽性率と矛盾しない")

# ---- 大問3
DL, L3 = sp.Rational(1, 20), sp.Rational(1, 5)
dd, t = sp.symbols("d t", positive=True)
FD = sp.integrate(L3 * sp.exp(-L3 * t), (t, -sp.log(dd) / DL, sp.oo))
E.check("s19p03a", sp.simplify(sp.diff(FD, dd) - 4 * dd ** 3) == 0, "sympy で分布関数を微分")
Ts = rng.exponential(5, 1_000_000)
Dsim = np.exp(-0.05 * Ts)
E.check("s19p03a", stats.kstest(Dsim[:20000], stats.beta(4, 1).cdf).pvalue > 0.01, "シミュレーションが Beta(4,1) と整合")
E.choice("s19p03a", "$4d^{3}$", "正解の式")
E.num("s19p03b", rnd(100 * stats.beta(4, 1).std(), 2), "Beta(4,1) の標準偏差")
E.check("s19p03b", abs(100 * Dsim.std() - 16.33) < 0.05, f"シミュレーション {100 * Dsim.std():.3f}")

# ---- 大問4
# (a) 母集団を作り、回帰推定量と標本平均の分散をシミュレーションで比べる（相関 0.8）
Npop = 200_000
inc = rng.normal(0, 1, Npop)
con = 0.8 * inc + 0.6 * rng.normal(0, 1, Npop)
Xbar_pop = inc.mean()
reg_est, mean_est = [], []
for _ in range(4000):
    idx = rng.choice(Npop, 144, replace=False)
    xs_, ys_ = inc[idx], con[idx]
    bb = np.cov(xs_, ys_)[0, 1] / xs_.var(ddof=1)
    reg_est.append(ys_.mean() + bb * (Xbar_pop - xs_.mean()))
    idx2 = rng.choice(Npop, 400, replace=False)
    mean_est.append(con[idx2].mean())
ratio = np.var(reg_est) / np.var(mean_est)
E.check("s19p04a", abs(ratio - 1) < 0.1, f"回帰推定量(n=144)と標本平均(n=400)の分散の比 {ratio:.3f}≈1")
E.num("s19p04a", round(400 * (1 - np.corrcoef(inc, con)[0, 1] ** 2)), f"母集団の相関 {np.corrcoef(inc, con)[0, 1]:.4f} から 400(1−ρ²)")
# (b) 実際に指示行列を作って対応分析し、総イナーシャと非零の特異値の数を数える
Qs = [3, 4, 2]
resp = np.column_stack([rng.integers(0, k, 500) for k in Qs])
Z = np.hstack([np.eye(k)[resp[:, j]] for j, k in enumerate(Qs)])
P = Z / Z.sum()
r_ = P.sum(1)
c_ = P.sum(0)
Sres = (P - np.outer(r_, c_)) / np.sqrt(np.outer(r_, c_))
sv = np.linalg.svd(Sres, compute_uv=False)
inertia = (sv ** 2).sum()
ndim = int((sv > 1e-9).sum())
E.choice("s19p04b", f"総イナーシャ {f(inertia, 1)}、次元数 {ndim}", "乱数の回答で指示行列を対応分析")

# ---- 大問5
# (a) 条件に合うデータを作って OLS の平均応答の標準誤差を行列で計算
table_ok("s19p05a", 2.101, stats.t.ppf(0.975, 18), 0.0006, "t_0.025(18)")
xd = rng.normal(size=20)
xd = (xd - xd.mean()) / np.sqrt(((xd - xd.mean()) ** 2).sum()) * np.sqrt(2000) + 60
Xd = np.column_stack([np.ones(20), xd])
x0v = np.array([1, 80])
se_m = 0.30 * np.sqrt(x0v @ np.linalg.inv(Xd.T @ Xd) @ x0v)
E.choice("s19p05a", f"${f(2.101 * se_m, 3)}$", "x0'(X'X)^{-1}x0 を行列で計算")
# (b) BIC から事後確率
b12 = math.exp((416.8 - 412.6) / 2)
E.choice("s19p05b", f"${f(b12 / (1 + b12), 3)}$", "math.exp で直接")
for xv, ev in ((2.1, 8.166), (-2.1, 0.1225), (4.2, 66.69)):
    table_ok("s19p05b", ev, math.exp(xv), 0.0006 if abs(ev) < 10 else 0.006, f"e^{xv}")

# ---- 大問6
e = np.array([0.8, 1.1, 0.6, -0.2, -0.9, -1.2, -0.4, 0.2])
try:
    from statsmodels.stats.stattools import durbin_watson
    dw, how = durbin_watson(e), "statsmodels durbin_watson"
except ImportError:
    dw, how = np.sum(np.diff(e) ** 2) / np.sum(e ** 2), "np.diff"
E.num("s19p06a", rnd(dw, 2), how)
th = [1, 0.5, -0.3]
g = [sum(th[i] * th[i + k] for i in range(3 - k)) for k in range(3)]
E.choice("s19p06b", f"\\rho(1)={f(g[1] / g[0], 3)},\\ \\rho(2)={f(g[2] / g[0], 3)},\\ \\rho(3)={f(0, 3)}", "係数の畳み込みで自己共分散")
eps = rng.normal(size=600_002)
u = eps[2:] + 0.5 * eps[1:-1] - 0.3 * eps[:-2]
ac = [np.corrcoef(u[:-k], u[k:])[0, 1] for k in (1, 2, 3)]
E.check("s19p06b", abs(ac[0] - 0.261) < 0.006 and abs(ac[1] + 0.224) < 0.006 and abs(ac[2]) < 0.006, f"シミュレーション {np.round(ac, 3)}")

# ---- 大問7
pts = np.array([[0, 0], [0.5, 1.0], [-0.5, 0.5], [1.0, 0], [0, 1.2], [0.6, 0.3]])
lab = np.array(["正常", "延滞", "延滞", "正常", "延滞", "正常"])
dist = np.linalg.norm(pts - np.array([0.2, 0.4]), axis=1)
order = np.argsort(dist)


def vote(k):
    v = lab[order[:k]]
    return "正常" if (v == "正常").sum() > k / 2 else "延滞"


nb3 = "・".join(f"P{i + 1}" for i in order[:3])
E.choice("s19p07a", f"$k=3$ の近傍は {nb3} で「{vote(3)}」、$k=5$ では「{vote(5)}」と判定される", "numpy の距離で最近傍")
# (b) 平均二乗誤差を λ の格子で数値的に最小化し、シミュレーションでも確かめる
lams = np.linspace(0, 5, 50001)
msev = (0.25 + lams ** 2) / (1 + lams) ** 2
k = int(np.argmin(msev))
E.choice("s19p07b", f"$\\lambda={f(lams[k], 2)}$、平均二乗誤差 {f(msev[k], 3)}", "格子で最小化")
bs = rng.normal(1.0, 0.5, 1_000_000)
E.check("s19p07b", abs(np.mean((bs / 1.25 - 1.0) ** 2) - 0.200) < 0.002, f"シミュレーションの平均二乗誤差 {np.mean((bs / 1.25 - 1.0) ** 2):.4f}")
post = stats.beta(6, 94)
prof = integrate.quad(lambda th_: (6 - 96 * th_) * post.pdf(th_), 0, 1)[0]
E.choice("s19p07c", f"期待利益 {f(prof, 2)} 万円で、貸出を{'続ける' if prof > 0 else 'やめる'}", "事後分布で期待利益を数値積分")

# ---- 大問8
E.choice("s19p08a", f"${f(stats.hypergeom(50, 3, 10).sf(0), 3)}$", "scipy hypergeom")
E.check("s19p08a", abs(1 - sp.binomial(47, 10) / sp.binomial(50, 10) - stats.hypergeom(50, 3, 10).sf(0)) < 1e-12, "二項係数の比で確認")
bf = np.array([12, 9, 15, 7, 10, 8, 14, 11, 6, 13, 9, 10])
af = np.array([9, 7, 11, 7, 8, 9, 10, 8, 5, 9, 10, 7])
dif = bf - af
nz = dif[dif != 0]
res = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5, alternative="greater")
E.num("s19p08b", rnd(res.pvalue, 3), "scipy binomtest（0 を除く）")

# ---- 大問9
# (a) √μ の区間を作って 2 乗する（scipy の正規分布の点を使う）と、被覆率のシミュレーション
zq = stats.norm.ppf(0.975)
lo9, hi9 = (4 - zq / 2) ** 2, (4 + zq / 2) ** 2
E.choice("s19p09a", f"({f(lo9, 2)},\\ {f(hi9, 2)})", "scipy の 1.96 で区間を作り 2 乗")
mus = rng.poisson(16, 400_000)
cov = np.mean(((np.sqrt(mus) - zq / 2) ** 2 < 16) & (16 < (np.sqrt(mus) + zq / 2) ** 2))
E.check("s19p09a", 0.93 < cov < 0.97, f"μ=16 での被覆率 {cov:.3f}")
Pm = np.array([[0.9, 0.06, 0.04], [0.5, 0.4, 0.1], [0.6, 0, 0.4]])
E.choice("s19p09b", f"${f(np.linalg.matrix_power(Pm, 2)[0, 2], 3)}$", "推移確率行列の 2 乗")
stt = np.zeros(300_000, dtype=int)
cum = Pm.cumsum(1)
for _ in range(2):
    uu = rng.random(stt.size)
    stt = (uu[:, None] > cum[stt]).sum(1)
E.check("s19p09b", abs((stt == 2).mean() - 0.058) < 0.002, f"シミュレーション {(stt == 2).mean():.4f}")

# ---- 大問10
tabs = np.array([[[80, 20], [60, 40]], [[20, 30], [15, 35]]])
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
    cmh = StratifiedTable([t_ for t_ in tabs]).test_null_odds(correction=False).statistic
    how = "statsmodels StratifiedTable.test_null_odds"
except ImportError:
    num, den = 0.0, 0.0
    for t_ in tabs:
        hg = stats.hypergeom(t_.sum(), t_[:, 0].sum(), t_[0].sum())
        num += t_[0, 0] - hg.mean()
        den += hg.var()
    cmh, how = num ** 2 / den, "scipy hypergeom の平均と分散"
E.choice("s19p10a", f"${f(cmh, 2)}$", how)
num, den = 0.0, 0.0
for t_ in tabs:
    hg = stats.hypergeom(t_.sum(), t_[:, 0].sum(), t_[0].sum())
    num += t_[0, 0] - hg.mean()
    den += hg.var()
E.check("s19p10a", abs(num ** 2 / den - cmh) < 1e-6, "超幾何分布の平均・分散からも同じ値")

# ---- 大問11
xm = np.where(rng.random(4_000_000) < 0.9, rng.normal(0, 1, 4_000_000), rng.normal(0, 3, 4_000_000))
kurt_exact = 3 * (0.9 * 1 + 0.1 * 81) / (0.9 + 0.9) ** 2
E.choice("s19p11a", f"${f(kurt_exact, 2)}$", "混合のモーメントを直接")
E.check("s19p11a", abs(stats.kurtosis(xm, fisher=False) - 8.33) < 0.15, f"シミュレーション {stats.kurtosis(xm, fisher=False):.3f}")
mu, sg, nn = 1.0, 4.0, 48
J = np.array([1 / sg, -mu / (2 * sg ** 3)])
V = np.diag([sg ** 2 / nn, 2 * sg ** 4 / nn])
E.num("s19p11b", rnd(np.sqrt(J @ V @ J), 3), "勾配と分散行列の 2 次形式")
xs_ = rng.normal(0.25, 1, size=(100_000, 48))
srs = xs_.mean(1) / xs_.std(1, ddof=1)
E.check("s19p11b", abs(srs.std() - 0.147) < 0.006, f"シミュレーションの標準偏差 {srs.std():.4f}")

# ---- 大問12
S12 = np.array([[5.1, 3.1, 1.6], [3.1, 5.1, 1.6], [1.6, 1.6, 2.6]])
wv, vv = np.linalg.eigh(S12)
E.check("s19p12a", np.allclose(np.abs(vv[:, -1]), [2 / 3, 2 / 3, 1 / 3]) and np.all(wv > 0), f"最大固有値の固有ベクトルが (2/3,2/3,1/3)、固有値 {np.round(wv, 4)}")
E.num("s19p12a", rnd(wv[-1] / wv.sum(), 3), "numpy の固有値分解")

# ---- 大問13
Yt = np.array([[9, 13, 11], [7, 10, 7], [12, 15, 15], [8, 10, 9]], float)
gm = Yt.mean()
ssa = 3 * ((Yt.mean(1) - gm) ** 2).sum()
res_ = Yt - Yt.mean(1, keepdims=True) - Yt.mean(0, keepdims=True) + gm
sse = (res_ ** 2).sum()
E.num("s19p13a", rnd((ssa / 3) / (sse / 6), 2), "交互作用の残差行列から誤差平方和")
table_ok("s19p13a", 4.757, stats.f.ppf(0.95, 3, 6), 0.0006, "F_0.05(3,6)")
E.check("s19p13a", abs(((Yt - gm) ** 2).sum() - 85.0) < 1e-9, "全体の平方和 85.0")

E.done()
