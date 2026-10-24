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
E.num("s19p04a", round(400 * (1 - 0.8 ** 2)), "400(1−ρ²)（上のシミュレーションで同じ分散になることを確認）")

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

# ---- 大問7
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

# ---- 大問9
# (a) √μ の区間を作って 2 乗する（scipy の正規分布の点を使う）と、被覆率のシミュレーション
zq = stats.norm.ppf(0.975)
lo9, hi9 = (4 - zq / 2) ** 2, (4 + zq / 2) ** 2
E.choice("s19p09a", f"({f(lo9, 2)},\\ {f(hi9, 2)})", "scipy の 1.96 で区間を作り 2 乗")
mus = rng.poisson(16, 400_000)
cov = np.mean(((np.sqrt(mus) - zq / 2) ** 2 < 16) & (16 < (np.sqrt(mus) + zq / 2) ** 2))
E.check("s19p09a", 0.93 < cov < 0.97, f"μ=16 での被覆率 {cov:.3f}")

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

# ---- 作り替えた小問（2026-10-24）
# 大問1 (a) 大標本の z 検定：scipy の正規分布で p 値、区間。シミュレーションで、差 0.2 ポイントでも n=90000 ならほぼ毎回有意
n1a = 90000
z1a = (0.052 - 0.05) / math.sqrt(0.05 * 0.95 / n1a)
pv1a = 2 * stats.norm.sf(z1a)
h1a = stats.norm.ppf(0.975) * math.sqrt(0.052 * 0.948 / n1a)
table_ok("s19p01a", 0.9970, stats.norm.cdf(2.75), 0.00005, "Φ(2.75)")
E.choice("s19p01a", f"$z={f(z1a, 2)}$で両側$p$値は約{f(pv1a, 3)}。5%水準で有意だが、95%信頼区間({f((0.052 - h1a) * 100, 2)}%,{f((0.052 + h1a) * 100, 2)}%)", "scipy norm で p 値と区間")
pw1a = np.mean(np.abs((rng.binomial(n1a, 0.052, 20000) / n1a - 0.05) / math.sqrt(0.05 * 0.95 / n1a)) > 1.96)
E.check("s19p01a", pw1a > 0.7 and (0.052 + h1a - 0.05) < 0.01, f"真の率 5.2% のとき有意になる割合 {pw1a:.3f}、区間の上端の差 {(0.052 + h1a - 0.05) * 100:.2f} ポイント < 1")

# 大問2 (a) 2 標本 t：scipy の t 分布で区間と p 値
se2a = 3.2 * math.sqrt(0.1)
for q_, tv_ in ((0.975, 2.024), (0.995, 2.712)):
    table_ok("s19p02a", tv_, stats.t.ppf(q_, 38), 0.0006, f"t({q_}, 38)")
tt2a = stats.ttest_ind_from_stats(12.4, 3.2, 20, 10.1, 3.2, 20, equal_var=True)
E.check("s19p02a", 0.01 < tt2a.pvalue < 0.05, f"scipy ttest_ind_from_stats: t={tt2a.statistic:.3f}、p={tt2a.pvalue:.4f}")
c95 = [2.3 + sgn * stats.t.ppf(0.975, 38) * se2a for sgn in (-1, 1)]
c99 = [2.3 + sgn * stats.t.ppf(0.995, 38) * se2a for sgn in (-1, 1)]
E.choice("s19p02a", f"95%信頼区間は$({f(c95[0], 2)},\\{f(c95[1], 2)})$で0を含まないので5%水準で有意。99%信頼区間$({f(c99[0], 2)},\\{f(c99[1], 2)})$は0を含む", "scipy t で区間")
# (b) ずれのある指数分布：2 次元の数値最大化（μ ≤ min の制約つき）と、尤度が μ について増加すること
xs2b = np.array([3.2, 4.1, 3.5, 6.8, 5.0, 9.4, 3.9, 4.5])
def ll2b(mu, th):
    return -np.inf if mu > xs2b.min() or th <= 0 else -len(xs2b) * np.log(th) - (xs2b - mu).sum() / th
grid_mu = np.linspace(2.0, 3.2, 1201)
grid_th = np.linspace(0.5, 5, 4501)
LL = np.array([[ll2b(m_, t_) for t_ in grid_th[::10]] for m_ in grid_mu[::10]])
im, it_ = np.unravel_index(np.argmax(LL), LL.shape)
th_best = grid_th[np.argmax([ll2b(3.2, t_) for t_ in grid_th])]
E.check("s19p02b", grid_mu[::10][im] == 3.2 and abs(th_best - 1.85) < 0.002, f"格子探索の最大点 μ={grid_mu[::10][im]:.2f}、θ={th_best:.3f}")
E.choice("s19p02b", f"(\\hat\\mu,\\hat\\theta)=({f(xs2b.min(), 1)},\\{f(xs2b.mean() - xs2b.min(), 2)})", "最小値と平均−最小値")
sims2b = 2.5 + rng.exponential(2.0, (200_000, 8))
E.check("s19p02b", abs(sims2b.min(1).mean() - (2.5 + 2.0 / 8)) < 0.01, f"E[x_(1)]=μ+θ/n のシミュレーション {sims2b.min(1).mean():.4f}（理論 2.75）")

# 大問4 (b) 数量化 III 類＝0・1 表の対応分析：自明な解（特異値 1）を除いた最大の特異値の 2 乗が、世帯スコアと選択肢スコアの相関の 2 乗
Z4 = (rng.random((60, 5)) < np.array([0.6, 0.3, 0.5, 0.2, 0.4])).astype(float)
Z4 = Z4[Z4.sum(1) > 0]
rs4, cs4 = Z4.sum(1), Z4.sum(0)
M4 = Z4 / np.sqrt(np.outer(rs4, cs4))
U4, sv4, Vt4 = np.linalg.svd(M4)
ysc, xsc = U4[:, 1] / np.sqrt(rs4 / Z4.sum()), Vt4[1] / np.sqrt(cs4 / Z4.sum())
ii, jj = np.nonzero(Z4)
corr4 = np.corrcoef(ysc[ii], xsc[jj])[0, 1]
E.check("s19p04b", abs(sv4[0] - 1) < 1e-10 and abs(abs(corr4) - sv4[1]) < 1e-8,
        f"自明な解の特異値 {sv4[0]:.6f}=1、第 1 の解の相関 {abs(corr4):.5f} = 特異値 {sv4[1]:.5f}（固有値はその 2 乗 {sv4[1] ** 2:.5f}）")
E.choice("s19p04b", "相関が最大になるように両方のスコアを決める方法で、0・1の表に対応分析を当てはめたものと同じ解", "定義の確認")

# 大問6 (a) 動的な回帰の反応をシミュレーション（誤差なしで x を 1 上げて走らせる）
y_base, y_up = [0.0], [0.0]
for m_ in range(3):
    y_up.append(0.30 * 1 + 0.60 * y_up[-1])
    y_base.append(0.60 * y_base[-1])
E.num("s19p06a", rnd(y_up[3] - y_base[3], 3), "漸化式を 3 か月走らせる")
# (b) 独立なランダムウォークどうしの回帰：t の分布と DW をシミュレーション
nrw, big, rr2, dws = 60, 0, [], []
for _ in range(3000):
    yy_ = np.cumsum(rng.normal(size=nrw))
    xx_ = np.cumsum(rng.normal(size=nrw))
    Xm = np.column_stack([np.ones(nrw), xx_])
    bh, res_, *_ = np.linalg.lstsq(Xm, yy_, rcond=None)
    e_ = yy_ - Xm @ bh
    s2_ = e_ @ e_ / (nrw - 2)
    tv = bh[1] / math.sqrt(s2_ * np.linalg.inv(Xm.T @ Xm)[1, 1])
    big += abs(tv) > stats.t.ppf(0.975, nrw - 2)
    dws.append(np.sum(np.diff(e_) ** 2) / (e_ @ e_))
E.check("s19p06b", big / 3000 > 0.5 and np.median(dws) < 0.5, f"独立なランダムウォークどうしで |t| が 5% 点を超える割合 {big / 3000:.2f}（本来 0.05）、DW の中央値 {np.median(dws):.2f}")
E.check("s19p06b", f(9.7 ** 2 / (9.7 ** 2 + 58), 2) == "0.62", "t=9.7、n=60 と R²=0.62 が整合")
E.choice("s19p06b", "見かけの回帰の疑いが強い", "解釈")

# 大問7 (a) 写像 φ(x)=(1,√2x,x²) の内積がカーネルの値に等しいこと、x² で完全に分かれること
phi7 = lambda x_: np.array([1, math.sqrt(2) * x_, x_ * x_])
E.check("s19p07a", abs(phi7(-2.0) @ phi7(1.8) - (1 + (-2.0) * 1.8) ** 2) < 1e-12, f"φ(x)'φ(z)={phi7(-2.0) @ phi7(1.8):.4f}=K")
bad7, good7 = np.array([-2.0, -1.6, 1.8, 2.2]), np.array([-0.6, 0.2, 0.9])
allx = np.sort(np.concatenate([bad7, good7]))
one_thr = any(np.all(bad7 > c_) and np.all(good7 < c_) or np.all(bad7 < c_) and np.all(good7 > c_) for c_ in (allx[:-1] + allx[1:]) / 2)
E.check("s19p07a", not one_thr and (bad7 ** 2).min() > (good7 ** 2).max(), "x の 1 本のしきい値では分けられず、x² では分けられる")
E.choice("s19p07a", f"K(-2.0,1.8)={f((1 + (-2.0) * 1.8) ** 2, 2)}$", "カーネルの値")

# 大問8 (a) scipy で二項とポアソン、シミュレーション
q8 = math.exp(-1.5)
table_ok("s19p08a", 0.2231, q8, 0.00005, "e^{-1.5}")
E.choice("s19p08a", f"${f(stats.binom.pmf(2, 10, 0.2231), 3)}$", "scipy binom（問題文の値で）")
cnt8 = (rng.poisson(1.5, (200_000, 10)) == 0).sum(1)
E.check("s19p08a", abs(np.mean(cnt8 == 2) - stats.binom.pmf(2, 10, q8)) < 0.004, f"シミュレーション {np.mean(cnt8 == 2):.4f}")
# (b) scipy の kruskal
kw = stats.kruskal([12, 15, 9, 14], [7, 10, 6, 11], [3, 8, 5, 4])
E.num("s19p08b", rnd(kw.statistic, 2), f"scipy kruskal（p 値 {kw.pvalue:.4f}）")

# 大問9 (b) ポアソン過程をシミュレーションし、N(10)=8 となった道の N(30) の平均と分散
arr = rng.poisson(0.5 * 10, 2_000_000)
sel9 = arr == 8
n30 = arr[sel9] + rng.poisson(0.5 * 20, sel9.sum())
E.check("s19p09b", abs(n30.mean() - 18) < 0.05 and abs(n30.var() - 10) < 0.15, f"N(10)=8 の条件で N(30) の平均 {n30.mean():.3f}、分散 {n30.var():.3f}")
E.choice("s19p09b", "期待値18、分散10", "独立増分")

# 大問10 numpy の反復比例当てはめで G²
n10 = np.array([[[80, 20], [60, 40]], [[20, 30], [15, 35]]], float)
def ipf10(margs):
    m_ = np.ones_like(n10)
    for _ in range(3000):
        for ax in margs:
            keep_ = tuple(i for i in range(3) if i not in ax)
            m_ = m_ * n10.sum(axis=keep_, keepdims=True) / m_.sum(axis=keep_, keepdims=True)
    return 2 * np.sum(n10 * np.log(n10 / m_))
g_ind, g_ci, g_h = ipf10([(0, 1), (2,)]), ipf10([(0, 1), (0, 2)]), ipf10([(0, 1), (0, 2), (1, 2)])
for q_, cv_ in ((1, 3.84), (2, 5.99), (3, 7.81)):
    table_ok("s19p10a", cv_, stats.chi2.isf(0.05, q_), 0.006, f"χ²({q_}) 5% 点")
E.choice("s19p10a", f"$G^2={f(g_h, 2)}$は3.84を下回り、督促と解消の関連（オッズ比）は2つの層で等しいとみてよい。$[LS][LR]$との差{f(g_ci - g_h, 2)}", "numpy の反復比例当てはめ")
E.check("s19p10a", f(g_ind, 2) == "44.45" and f(g_ci, 2) == "10.76" and g_ci > 5.99 and g_ind > 7.81, f"表の G²：{g_ind:.3f}、{g_ci:.3f}、{g_h:.4f}")
mh = sum((a_ - (a_ + b_) * (a_ + c_) / (a_ + b_ + c_ + d_)) for a_, b_, c_, d_ in ((80, 20, 60, 40), (20, 30, 15, 35)))
vmh = sum((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_) / ((a_ + b_ + c_ + d_) ** 2 * (a_ + b_ + c_ + d_ - 1)) for a_, b_, c_, d_ in ((80, 20, 60, 40), (20, 30, 15, 35)))
E.check("s19p10a", abs(mh ** 2 / vmh - (g_ci - g_h)) < 1.0, f"CMH 統計量 {mh ** 2 / vmh:.3f} と G² の差 {g_ci - g_h:.3f} がほぼ同じ（解説の主張）")

# 大問11 (a) numpy の二次形式と、シミュレーション
Sig11 = np.array([[400, 120, -60], [120, 225, 30], [-60, 30, 100]], float)
w11 = np.array([0.5, 0.3, 0.2])
E.choice("s19p11a", f"${f(math.sqrt(w11 @ Sig11 @ w11), 2)}$", "w'Σw を numpy で")
R11 = rng.multivariate_normal(np.zeros(3), Sig11, 400_000) @ w11
E.check("s19p11a", abs(R11.std() - math.sqrt(w11 @ Sig11 @ w11)) < 0.05 and np.all(np.linalg.eigvalsh(Sig11) > 0), f"シミュレーションの標準偏差 {R11.std():.3f}（Σ は正定値）")
# (b) 正確な分布（畳み込み）と比べ、問題の正規近似の値を独立に計算
pmf = np.array([1.0])
for nn_, pp_ in ((200, 0.01), (300, 0.04), (100, 0.08)):
    pmf = np.convolve(pmf, stats.binom.pmf(np.arange(nn_ + 1), nn_, pp_))
mu11 = 200 * 0.01 + 300 * 0.04 + 100 * 0.08
v11 = 200 * 0.01 * 0.99 + 300 * 0.04 * 0.96 + 100 * 0.08 * 0.92
z11 = round((31.5 - mu11) / math.sqrt(v11), 2)
for zz_, pv_ in ((2.06, 0.9803), (2.08, 0.9812), (2.19, 0.9857)):
    table_ok("s19p11b", pv_, stats.norm.cdf(zz_), 0.00005, f"Φ({zz_})")
E.num("s19p11b", rnd(stats.norm.sf(z11), 4), f"scipy norm.sf(z={z11})")
E.check("s19p11b", abs(np.dot(np.arange(pmf.size), pmf) - mu11) < 1e-9 and abs(np.dot(np.arange(pmf.size) ** 2, pmf) - mu11 ** 2 - v11) < 1e-8,
        f"畳み込みの平均・分散が {mu11}、{v11:.2f} に一致（正確な P(X≥32)={pmf[32:].sum():.4f}）")

E.done()
