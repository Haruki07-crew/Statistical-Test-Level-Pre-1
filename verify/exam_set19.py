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
# (a) 2×2 表のカイ二乗（連続修正なし）の平方根
tab = np.array([[45, 455], [16, 384]])
chi2 = stats.chi2_contingency(tab, correction=False)[0]
E.choice("s19p01a", f"${f(np.sqrt(chi2), 2)}$", "2×2 表のカイ二乗統計量の平方根")
try:
    from statsmodels.stats.proportion import proportions_ztest
    zz = proportions_ztest([45, 16], [500, 400])[0]
    E.check("s19p01a", abs(zz - np.sqrt(chi2)) < 1e-9, f"statsmodels proportions_ztest {zz:.4f}")
except ImportError:
    pass
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
n = 10
lam = sp.symbols("lam", positive=True)
s_ = sp.symbols("s", positive=True)
dens = lam ** n * s_ ** (n - 1) * sp.exp(-lam * s_) / sp.factorial(n - 1)
m1 = sp.simplify(sp.integrate((n - 1) / s_ * dens, (s_, 0, sp.oo)))
m2 = sp.simplify(sp.integrate(((n - 1) / s_) ** 2 * dens, (s_, 0, sp.oo)))
var = sp.simplify(m2 - m1 ** 2)
x = sp.symbols("x", positive=True)
info = -sp.integrate(sp.diff(sp.log(lam * sp.exp(-lam * x)), lam, 2) * lam * sp.exp(-lam * x), (x, 0, sp.oo))
eff = sp.simplify((1 / (n * info)) / var)
E.check("s19p02a", sp.simplify(m1 - lam) == 0, "sympy：(n−1)/S は不偏")
E.choice("s19p02a", f"${f(float(eff), 3)}$", f"sympy でガンマ密度から分散と情報量を積分（効率 {eff}）")
# (b)
for xv, Fv, tol in ((5.43, 0.0005, 0.00006), (10.85, 0.050, 0.0006), (21.70, 0.643, 0.0006), (31.41, 0.950, 0.0006), (43.40, 0.998, 0.0006)):
    table_ok("s19p02b", Fv, stats.chi2.cdf(xv, 20), tol, f"F({xv})")
cut = stats.chi2.ppf(0.05, 20) / 0.2
pw = stats.gamma(10, scale=1 / 0.2).cdf(cut)
E.choice("s19p02b", f"${f(pw, 3)}$", "scipy ガンマ分布で P(S ≤ 棄却限界 | λ=0.2)")
Ssim = rng.gamma(10, 1 / 0.2, 400_000)
E.check("s19p02b", abs((Ssim <= cut).mean() - 0.643) < 0.003, f"シミュレーション {(Ssim <= cut).mean():.4f}")


# (c) 数値積分で KL
def kl(l0, l1):
    return integrate.quad(lambda t: l0 * np.exp(-l0 * t) * ((np.log(l0) - l0 * t) - (np.log(l1) - l1 * t)), 0, np.inf)[0]


kA, kB = kl(0.1, 0.2), kl(0.1, 0.05)
E.choice("s19p02c", f"A: {f(kA, 3)}、B: {f(kB, 3)}。{'A' if kA < kB else 'B'} のほうが近い", "数値積分で KL")
E.check("s19p02c", abs(math.log(2) - 0.6931) < 5e-5, "log2")

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
# (a) 件数で重み付け、二相推定量の不偏性をシミュレーション
E.num("s19p04a", rnd((240 * 9.0 + 160 * 6.0) / 400, 2), "回答層・非回答層の件数で重み付け")
pop_r = rng.normal(9, 3, 6000)
pop_n = rng.normal(6, 3, 4000)
ests = []
for _ in range(2000):
    smp_r = rng.choice(pop_r.size, 240, replace=False)
    sub = rng.choice(pop_n.size, 40, replace=False)
    ests.append(0.6 * pop_r[smp_r].mean() + 0.4 * pop_n[sub].mean())
E.check("s19p04a", abs(np.mean(ests) - (0.6 * pop_r.mean() + 0.4 * pop_n.mean())) < 0.03, "シミュレーションで二相推定量が不偏")
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
# (c) チェック損失
xs = np.array([20, 30, 40, 50, 60])
ys = np.array([9, 12, 16, 17, 24])


def ch(a, tau=0.9):
    u = ys - (a + 0.3 * xs)
    return float(np.sum(u * (tau - (u < 0))))


L = {k: ch(a) for k, a in (("A", 2), ("B", 4), ("C", 5))}
best = min(L, key=L.get)
E.choice("s19p04c", "、".join(f"{k}: {f(L[k], 1)}" for k in "ABC") + f"。{best} を選ぶ", "チェック関数 u(τ−I(u<0)) を numpy で計算")

# ---- 大問5
A = np.array([[40, 30], [30, 90]])
b = np.linalg.solve(A, [22, 27])
se = np.sqrt(0.25 * np.linalg.inv(A)[0, 0])
E.choice("s19p05a", f"${f(b[0] / se, 2)}$", "numpy で正規方程式と逆行列")
Lc = np.linalg.cholesky(A)
Zr = rng.normal(size=(30, 2))
Zr -= Zr.mean(0)
Zr = Zr @ np.linalg.inv(np.linalg.cholesky(Zr.T @ Zr)).T @ Lc.T
Xd = np.column_stack([np.ones(30), Zr])
E.check("s19p05a", np.allclose(Zr.T @ Zr, A) and np.isclose(np.linalg.inv(Xd.T @ Xd)[1, 1], np.linalg.inv(A)[0, 0]),
        "積和行列を再現したデータで、切片つき (X'X)^{-1} の成分が中心化した積和行列の逆行列と一致")

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
E.choice("s19p07b", f"${f(float(stats.binom(100, 0.02).stats(moments='s')), 3)}$", "scipy binom(100,0.02) の歪度")
E.check("s19p07b", abs(float(stats.bernoulli(0.02).stats(moments="s")) - 6.857) < 0.001, "1 件の歪度 6.857")
post = stats.beta(6, 94)
prof = integrate.quad(lambda th_: (6 - 96 * th_) * post.pdf(th_), 0, 1)[0]
E.choice("s19p07c", f"期待利益 {f(prof, 2)} 万円で、貸出を{'続ける' if prof > 0 else 'やめる'}", "事後分布で期待利益を数値積分")

# ---- 大問8
perm = list(permutations(range(4)))
der = sum(all(p[i] != i for i in range(4)) for p in perm) / len(perm)
E.choice("s19p08a", f"${f(der, 3)}$", "24 通りの総当たり")
E.choice("s19p08b", f"${f(stats.hypergeom(50, 3, 10).sf(0), 3)}$", "scipy hypergeom")
E.check("s19p08b", abs(1 - sp.binomial(47, 10) / sp.binomial(50, 10) - stats.hypergeom(50, 3, 10).sf(0)) < 1e-12, "二項係数の比で確認")

# ---- 大問9
for dfv, tv in ((9, 2.262), (13, 2.160), (23, 2.069)):
    table_ok("s19p09a", tv, stats.t.ppf(0.975, dfv), 0.0006, f"t_0.025({dfv})")
r1 = stats.ttest_ind_from_stats(5.2, np.sqrt(0.9), 10, 4.4, np.sqrt(0.3), 15, equal_var=False)
se9 = 0.8 / r1.statistic
nu = (0.09 + 0.02) ** 2 / (0.09 ** 2 / 9 + 0.02 ** 2 / 14)
tq = {9: 2.262, 13: 2.160, 23: 2.069}[math.floor(nu)]   # 問題文の数表の値（上で scipy と照合済み）
h = tq * se9
E.choice("s19p09a", f"({f(0.8 - h, 3)},\\ {f(0.8 + h, 3)})", f"scipy ttest_ind_from_stats(equal_var=False) の SE と自由度 {nu:.2f}・数表の t 点")
h_exact = stats.t.ppf(0.975, math.floor(nu)) * se9
E.check("s19p09a", abs(h_exact - h) < 0.001, f"正確な t 点でも半幅の差は {abs(h_exact - h):.5f}（他の選択肢と取り違えない）")
Pm = np.array([[0.9, 0.06, 0.04], [0.5, 0.4, 0.1], [0.6, 0, 0.4]])
w, v = np.linalg.eig(Pm.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi /= pi.sum()
E.check("s19p09b", np.allclose(pi, [60 / 71, 6 / 71, 5 / 71]), f"固有ベクトルで定常分布 {pi}")
st, cnt, rets = 2, 0, []
cum = Pm.cumsum(1)
for uu in rng.random(400_000):
    st = int((uu > cum[st]).sum())
    cnt += 1
    if st == 2:
        rets.append(cnt)
        cnt = 0
E.choice("s19p09b", f"${f(1 / pi[2], 1)}$", "1/π_S")
E.check("s19p09b", abs(np.mean(rets) - 14.2) < 0.3, f"シミュレーションの平均再帰時間 {np.mean(rets):.2f}")

# ---- 大問10
vals = [8, 11, 12, 5, 7, 9]
obs = np.mean(vals[:3]) - np.mean(vals[3:])
diffs = []
for c in combinations(range(6), 3):
    g1 = [vals[i] for i in c]
    g2 = [vals[i] for i in range(6) if i not in c]
    diffs.append(np.mean(g1) - np.mean(g2))
E.num("s19p10a", rnd(np.mean(np.array(diffs) >= obs - 1e-12), 2), "20 通りの平均の差を総当たり")
res = stats.permutation_test(([8, 11, 12], [5, 7, 9]), lambda a_, b_: np.mean(a_) - np.mean(b_),
                             permutation_type="independent", alternative="greater", n_resamples=np.inf)
E.num("s19p10a", rnd(res.pvalue, 2), "scipy permutation_test")

# ---- 大問11
mu, sg, nn = 1.0, 4.0, 48
J = np.array([1 / sg, -mu / (2 * sg ** 3)])
V = np.diag([sg ** 2 / nn, 2 * sg ** 4 / nn])
E.num("s19p11a", rnd(np.sqrt(J @ V @ J), 3), "勾配と分散行列の 2 次形式")
xs_ = rng.normal(0.25, 1, size=(100_000, 48))
srs = xs_.mean(1) / xs_.std(1, ddof=1)
E.check("s19p11a", abs(srs.std() - 0.147) < 0.006, f"シミュレーションの標準偏差 {srs.std():.4f}")

# ---- 大問12
R = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]])
wv, vv = np.linalg.eigh(R)
wv, vv = wv[::-1], vv[:, ::-1]
E.check("s19p12a", np.allclose(wv[:2], [2.004, 0.613], atol=6e-4) and np.allclose(np.abs(vv[:, 0]), [0.580, 0.613, 0.536], atol=6e-4)
        and np.allclose(np.abs(vv[:, 1]), [0.559, 0.179, 0.810], atol=6e-4), "表の固有値・固有ベクトルは相関行列 R のもの（3 桁）")
E.check("s19p12a", np.sign(vv[0, 1]) == np.sign(vv[1, 1]) != np.sign(vv[2, 1]) and len(set(np.sign(vv[:, 0]))) == 1, "固有ベクトルの成分の符号の関係")
E.num("s19p12a", rnd(1 - wv[2] * vv[0, 2] ** 2, 3), "1 − 第3主成分の負荷量の2乗（丸めない固有分解）")

# ---- 大問13
Yt = np.array([[9, 13, 11], [7, 10, 7], [12, 15, 15], [8, 10, 9]], float)
gm = Yt.mean()
ssa = 3 * ((Yt.mean(1) - gm) ** 2).sum()
res_ = Yt - Yt.mean(1, keepdims=True) - Yt.mean(0, keepdims=True) + gm
sse = (res_ ** 2).sum()
E.num("s19p13a", rnd((ssa / 3) / (sse / 6), 2), "交互作用の残差行列から誤差平方和")
table_ok("s19p13a", 4.757, stats.f.ppf(0.95, 3, 6), 0.0006, "F_0.05(3,6)")
E.check("s19p13a", abs(((Yt - gm) ** 2).sum() - 85.0) < 1e-9, "全体の平方和 85.0")

# ---- 大問14
lor = np.log(80 * 320 / (120 * 80))
s14 = np.sqrt(1 / 80 + 1 / 120 + 1 / 80 + 1 / 320)
lo, hi = np.exp(lor - stats.norm.ppf(0.975) * s14), np.exp(lor + stats.norm.ppf(0.975) * s14)
try:
    from statsmodels.stats.contingency_tables import Table2x2
    lo2, hi2 = Table2x2(np.array([[80, 120], [80, 320]])).oddsratio_confint()
    E.check("s19p14a", abs(lo2 - lo) < 1e-9 and abs(hi2 - hi) < 1e-9, "statsmodels Table2x2 と一致")
except ImportError:
    pass
E.choice("s19p14a", f"({f(lo, 3)},\\ {f(hi, 3)})", "丸めない対数オッズ比と scipy の 1.96")
for xv, ev in ((0.6066, 1.834), (1.3550, 3.877), (0.6667, 1.948), (1.2949, 3.651)):
    table_ok("s19p14a", ev, math.exp(xv), 0.0006, f"e^{xv}")
table_ok("s19p14a", 0.9808, math.log(8 / 3), 0.00006, "log(8/3)")

E.done()
