"""模擬試験 第26回（set26）の検算。generator/set26.py とは別の方法（scipy の分布関数・シミュレーション・
総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(26)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


# ---- 大問1 3^6 通りの割り当てを総当たり
cnt = sum(1 for a in itertools.product(range(3), repeat=6) if len(set(a)) == 3)
E.num("s26p01a", rnd(cnt / 3 ** 6, 3), f"総当たりで {cnt}/729")
E.check("s26p01a", f(1 - 3 * (2 / 3) ** 6, 3) == "0.737", "包除原理を途中でやめた値（誤答の根拠）")

# ---- 大問2
# (a) 差の標準偏差と前後の標準偏差を満たすデータを作り、scipy の ttest_rel と ttest_ind で
before = rng.normal(200, 20, 8)
before = (before - before.mean()) / before.std(ddof=1) * 20 + 200
dd = rng.normal(size=8)
dd = dd - dd.mean() - np.polyfit(before, dd, 1)[0] * (before - before.mean())   # before と無相関に
dd = dd / dd.std(ddof=1) * 4.8 + 6.0
after = before - dd
tr = stats.ttest_rel(before, after)
E.check("s26p02a", abs(tr.statistic - 6.0 / (4.8 / math.sqrt(8))) < 1e-9 and tr.pvalue < 0.05, f"ttest_rel: t={tr.statistic:.3f}、p={tr.pvalue:.4f}")
ti = stats.ttest_ind_from_stats(200, 20, 8, 194, 19, 8, equal_var=False)
E.check("s26p02a", abs(ti.statistic - 6 / math.sqrt(400 / 8 + 361 / 8)) < 1e-9 and ti.pvalue > 0.05, f"対応を無視した t={ti.statistic:.3f}、p={ti.pvalue:.3f}")
E.check("s26p02a", abs(stats.t.ppf(0.975, 7) - 2.365) < 5e-4 and abs(stats.t.ppf(0.975, 14) - 2.145) < 5e-4, "t の点")
E.choice("s26p02a", f"対応のある$t$検定で$t={f(tr.statistic, 2)}$。2.365を上回る", "scipy ttest_rel")
# (b) シミュレーションの T の分布と scipy の nct を比べる
Xs = rng.normal(0.5, 1.0, (200_000, 16))
Ts = Xs.mean(1) / (Xs.std(1, ddof=1) / 4)
nct = stats.nct(15, 2.0)
E.check("s26p02b", abs(Ts.mean() - nct.mean()) < 0.01 and abs(np.quantile(Ts, 0.9) - nct.ppf(0.9)) < 0.03,
        f"シミュレーションの T の平均 {Ts.mean():.3f}（非心 t の平均 {nct.mean():.3f}）、90% 点 {np.quantile(Ts, 0.9):.3f}（{nct.ppf(0.9):.3f}）")
E.check("s26p02b", stats.skew(Ts) > 0.1, f"T の分布は右に歪む（歪度 {stats.skew(Ts):.3f}）、平行移動ではない")
E.choice("s26p02b", "自由度15、非心度$\\sqrt{16}\\times0.5=2.0$の非心$t$分布", "scipy nct")

# ---- 大問3
# (a) パレートの対数尤度を数値最大化
xs3 = np.array([1.5, 2.0, 1.2, 4.0, 3.0])
for x_, l_ in zip(xs3, [0.4055, 0.6931, 0.1823, 1.3863, 1.0986]):
    E.check("s26p03a", abs(math.log(x_) - l_) < 5e-5, f"log {x_} の値")
grid = np.linspace(0.2, 5, 480001)
ll3 = len(xs3) * np.log(grid) - (grid + 1) * np.log(xs3).sum()
a_hat = grid[np.argmax(ll3)]
E.choice("s26p03a", f"十分統計量$\\sum\\log x_i$、最尤推定値$\\hat\\alpha={f(a_hat, 2)}$", f"格子で対数尤度を最大化 {a_hat:.4f}")
fit = stats.pareto.fit(xs3, floc=0, fscale=1)
E.check("s26p03a", abs(fit[0] - a_hat) < 1e-3, f"scipy pareto.fit の形状 {fit[0]:.4f}")
# (b) 3 つの統計量と正確な p 値
x3, l0 = 12, 6
zw, zs = (x3 - l0) / math.sqrt(x3), (x3 - l0) / math.sqrt(l0)
G = 2 * (x3 * math.log(x3 / l0) - (x3 - l0))
E.check("s26p03b", zw < 1.96 < zs and G > stats.chi2.isf(0.05, 1), f"Z_W={zw:.3f}、Z_S={zs:.3f}、G={G:.3f}")
E.choice("s26p03b", f"スコア検定（$Z_S={f(zs, 2)}$）と尤度比検定（$G={f(G, 1)}$）は棄却し、ワルド検定（$Z_W={f(zw, 2)}$）は棄却しない", "math で直接（表の丸めた値と同じ桁）")
pex = stats.poisson.sf(11, 6)
E.check("s26p03b", abs(pex - 0.020) < 0.002, f"正確な片側 p 値 P(X≥12)={pex:.4f}（解説の 0.020）")

# ---- 大問4
# (a) シミュレーション：p 値の分布
Z0 = rng.normal(0, 1, 1_000_000)
P0 = stats.norm.sf(Z0)
Z1 = rng.normal(2, 1, 1_000_000)
P1 = stats.norm.sf(Z1)
E.check("s26p04a", stats.kstest(P0[:50000], "uniform").pvalue > 0.01 and abs(np.median(P1) - stats.norm.sf(2)) < 0.0005,
        f"H0 で P は一様、H1 で P の中央値 {np.median(P1):.4f}")
E.check("s26p04a", abs(np.mean(P1 <= 0.05) - stats.norm.cdf(2 - 1.645)) < 0.002 and f(stats.norm.cdf(0.355), 2) == "0.64", f"攻撃時に P≤0.05 の割合 {np.mean(P1 <= 0.05):.4f}")
E.choice("s26p04a", f"異常がないとき$P$は区間$(0,1)$の一様分布。攻撃を受けているときの$P$の中央値は${f(stats.norm.sf(2), 4)}$", "1−Φ(2)")
# (b) 数値積分とシミュレーション
fy = lambda y: integrate.quad(lambda l: math.exp(-l) * l * math.exp(-l * y), 0, np.inf)[0]
E.check("s26p04b", all(abs(fy(y) - 1 / (1 + y) ** 2) < 1e-8 for y in (0.1, 1.0, 3.7)), "数値積分の周辺密度 = 1/(1+y)²")
Lm = rng.exponential(1.0, 2_000_000)
Ym = rng.exponential(1 / Lm)
E.check("s26p04b", abs(np.mean(Ym > 1) - 0.5) < 0.002 and abs(np.median(Ym) - 1) < 0.01, f"シミュレーション P(Y>1)={np.mean(Ym > 1):.4f}、中央値 {np.median(Ym):.3f}")
part = [integrate.quad(lambda y: y / (1 + y) ** 2, 0, M)[0] for M in (1e2, 1e4, 1e6)]
E.check("s26p04b", part[2] - part[1] > 4 and part[1] - part[0] > 4, f"∫_0^M y f(y)dy が M とともに増え続ける（{[round(v, 2) for v in part]}）")
E.choice("s26p04b", "f_Y(y)=\\dfrac{1}{(1+y)^2}$で、$P(Y>1)=0.5$。$E[Y]$は存在しない", "周辺密度と上側確率")

# ---- 大問5
# (a) 定義どおり
E.choice("s26p05a", f"\\dfrac{{24+1}}{{999+1}}={f(Fraction(25, 1000), 3)}$", "(k+1)/(B+1)")
# 小さな例で、(k+1)/(B+1) が帰無仮説のもとで P(p≤α)≤α を満たすこと
pv = []
for _ in range(4000):
    a_ = rng.normal(size=10)
    tobs = a_[:5].mean() - a_[5:].mean()
    perms = np.array([(lambda z: z[:5].mean() - z[5:].mean())(rng.permutation(a_)) for _ in range(99)])
    pv.append((1 + (perms >= tobs).sum()) / 100)
E.check("s26p05a", np.mean(np.array(pv) <= 0.05) <= 0.06, f"帰無仮説のもとで p≤0.05 となる割合 {np.mean(np.array(pv) <= 0.05):.4f}")
# (b) M/M/2：生成行列の定常分布（状態を 200 で打ち切り）
K = 200
lam, mu = 1.6, 1.0
Q = np.zeros((K + 1, K + 1))
for n_ in range(K + 1):
    if n_ < K:
        Q[n_, n_ + 1] = lam
    if n_ > 0:
        Q[n_, n_ - 1] = mu * min(n_, 2)
    Q[n_, n_] = -Q[n_].sum()
A = np.vstack([Q.T, np.ones(K + 1)])
pi = np.linalg.lstsq(A, np.r_[np.zeros(K + 1), 1], rcond=None)[0]
Lq = sum(max(n_ - 2, 0) * pi[n_] for n_ in range(K + 1))
E.num("s26p05b", rnd(Lq / lam, 2), f"生成行列の定常分布から L_q={Lq:.4f}、リトルの公式")
E.check("s26p05b", abs(pi[0] - 1 / 9) < 1e-9 and abs(pi[2:].sum() - 6.4 / 9) < 1e-9, "p0=1/9、待つ確率 6.4/9")
# (c) 歪んだ分布（対数正規）のシミュレーションで被覆率
mu_ln, s_ln = math.log(180), 0.53
mtrue = math.exp(mu_ln + s_ln ** 2 / 2)
Xc = rng.lognormal(mu_ln, s_ln, (20000, 400))
lo_ = Xc.mean(1) - 1.96 * Xc.std(1, ddof=1) / 20
hi_ = Xc.mean(1) + 1.96 * Xc.std(1, ddof=1) / 20
E.check("s26p05c", abs(np.mean((lo_ < mtrue) & (mtrue < hi_)) - 0.95) < 0.01, f"n=400 の対数正規データで近似区間の被覆率 {np.mean((lo_ < mtrue) & (mtrue < hi_)):.4f}")
E.choice("s26p05c", f"$({f(210 - 1.96 * 6, 2)},\\{f(210 + 1.96 * 6, 2)})$。中心極限定理", "210±1.96×120/√400")

# ---- 大問6 共分散行列から線形変換で
S6 = np.array([[25, 6], [6, 16]], float)
Lt = np.array([[1, 1], [1, -1]], float)
C6 = Lt @ S6 @ Lt.T
E.num("s26p06a", rnd(C6[0, 1] / math.sqrt(C6[0, 0] * C6[1, 1]), 3), "変換した共分散行列 LΣL' から")
Z6 = rng.multivariate_normal([0, 0], S6, 1_000_000)
E.check("s26p06a", abs(np.corrcoef(Z6[:, 0] + Z6[:, 1], Z6[:, 0] - Z6[:, 1])[0, 1] - 0.2296) < 0.003, "シミュレーション")

# ---- 大問7
# (a) scipy の friedmanchisquare
A7 = [12.1, 30.5, 8.4, 20.2, 15.0]
B7 = [13.0, 31.2, 8.1, 22.5, 15.8]
C7 = [14.2, 33.0, 9.0, 21.4, 16.6]
fr = stats.friedmanchisquare(A7, B7, C7)
E.choice("s26p07a", f"$Q={f(fr.statistic, 1)}$で5.99を上回り", f"scipy friedmanchisquare（p={fr.pvalue:.4f}）")
kw = stats.kruskal(A7, B7, C7)
E.check("s26p07a", f(kw.statistic, 2) == "0.32" and abs(stats.chi2.isf(0.05, 2) - 5.99) < 0.005, f"まとめて順位をつけたクラスカル・ウォリス H={kw.statistic:.3f}（誤答の値）")
# (b) 二項分布
E.choice("s26p07b", f"0.95^3={f(0.95 ** 3, 3)}$で、0.95より小さい", "0.95^3")
E.check("s26p07b", f(stats.binom.pmf(1, 20, 0.05), 3) == "0.377" and abs(stats.binom.mean(20, 0.05) - 1) < 1e-12, "20 個中ちょうど 1 個外れる確率 0.377、期待値 1")
cover = rng.random((200_000, 3)) < 0.95
E.check("s26p07b", abs(cover.all(1).mean() - 0.857) < 0.003, f"シミュレーション {cover.all(1).mean():.4f}")

# ---- 大問8
# (a) ガウス・マルコフ：不均一分散のとき OLS より WLS の分散が小さいことをシミュレーションで（条件が必要なことの確認）
xg = np.linspace(1, 10, 30)
Xg = np.column_stack([np.ones(30), xg])
W = 1 / xg ** 2
b_ols, b_wls = [], []
for _ in range(20000):
    yg = 1 + 2 * xg + rng.normal(0, 1, 30) * xg
    b_ols.append(np.linalg.lstsq(Xg, yg, rcond=None)[0][1])
    b_wls.append(np.linalg.solve(Xg.T @ (Xg * W[:, None]), Xg.T @ (W * yg))[1])
E.check("s26p08a", abs(np.mean(b_ols) - 2) < 0.02 and np.var(b_wls) < np.var(b_ols),
        f"不均一分散では OLS は不偏（平均 {np.mean(b_ols):.3f}）だが WLS の分散 {np.var(b_wls):.4f} < OLS {np.var(b_ols):.4f}")
E.choice("s26p08a", "誤差の平均が0で、分散が一定で互いに無相関なら、最小二乗推定量は線形不偏推定量の中で分散が最小", "定理の内容")
# (b) 層ごとの重み行列の要素数を数える
layers = [3, 5, 4, 1]
E.num("s26p08b", sum((layers[i] + 1) * layers[i + 1] for i in range(3)), "重み行列とバイアスの要素数")
# (c) データを作ってハット行列の対角とマハラノビス距離の関係を確かめ、公式で値を出す
Xh = rng.multivariate_normal([60, 300], [[100, 150], [150, 900]], 41)
Hm = np.column_stack([np.ones(41), Xh])
hii = np.diag(Hm @ np.linalg.inv(Hm.T @ Hm) @ Hm.T)
dif = Xh - Xh.mean(0)
D2 = np.einsum("ij,jk,ik->i", dif, np.linalg.inv(np.cov(Xh.T)), dif)
E.check("s26p08c", np.allclose(hii, 1 / 41 + D2 / 40), "乱数データで h_ii = 1/n + D²/(n−1)")
E.num("s26p08c", rnd(1 / 41 + 8 / 40, 3), "公式に D²=8 を代入")

# ---- 大問9 ロジスティック分布の標準偏差と、プロビットとロジットの当てはめ
E.check("s26p09a", abs(stats.logistic.std() - math.pi / math.sqrt(3)) < 1e-12, f"標準ロジスティック分布の標準偏差 {stats.logistic.std():.4f}")
xt = rng.uniform(20, 40, 20000)
yt = (rng.random(20000) < stats.logistic.cdf(-27 + 0.9 * xt)).astype(float)
Xt = np.column_stack([np.ones(20000), xt - 30])


def fit_glm(cdf, pdf):
    b = np.zeros(2)
    for _ in range(50):
        eta = Xt @ b
        p_ = np.clip(cdf(eta), 1e-12, 1 - 1e-12)
        d_ = pdf(eta)
        w_ = d_ ** 2 / (p_ * (1 - p_))
        z_ = eta + (yt - p_) / d_
        b = np.linalg.solve(Xt.T @ (Xt * w_[:, None]), Xt.T @ (w_ * z_))
    return b


bl_ = fit_glm(stats.logistic.cdf, stats.logistic.pdf)
bp_ = fit_glm(stats.norm.cdf, stats.norm.pdf)
E.check("s26p09a", 1.5 < bl_[1] / bp_[1] < 1.9, f"同じデータでロジットの係数 {bl_[1]:.3f}、プロビットの係数 {bp_[1]:.3f}、比 {bl_[1] / bp_[1]:.3f}")
E.choice("s26p09a", f"約{f(0.90 / (math.pi / math.sqrt(3)), 1)}。誤差の標準偏差がロジットモデルの方が約1.81倍大きい", "0.90/1.81")

# ---- 大問10 EM を反復（1 回目と収束先）
fail10 = [8, 12, 15, 20, 25]
cens10 = [18, 30, 30]
th = 20.0
hist = []
for _ in range(200):
    th = (sum(fail10) + sum(c + th for c in cens10)) / 8
    hist.append(th)
E.num("s26p10a", rnd(hist[0], 2), "EM の 1 回目")
E.check("s26p10a", abs(hist[-1] - 158 / 5) < 1e-9, f"収束先 {hist[-1]:.4f} = 総観察時間/故障数 31.6（解説）")
ll10 = lambda t: -5 * math.log(t) - 158 / t
E.check("s26p10a", abs(max(np.linspace(20, 50, 30001), key=ll10) - 31.6) < 0.002, "打ち切りつき指数分布の対数尤度の最大点 31.6")

# ---- 大問11 乱塊法のデータを作り、式の分子/分母（完全無作為化法の誤差分散の推定量）が不偏であることを確かめる
sb2, se2 = 25 / 4, 5.0          # E[MS_ブロック]=σ²+tσ_b²=30、E[MS_誤差]=σ²=5 となる設定
ests, msbs, mses = [], [], []
for _ in range(20000):
    blk = rng.normal(0, math.sqrt(sb2), 6)
    y11 = np.array([0, 1, 2, 3.0])[None, :] + blk[:, None] + rng.normal(0, math.sqrt(se2), (6, 4))
    gm = y11.mean()
    msb = 4 * ((y11.mean(1) - gm) ** 2).sum() / 5
    mse = ((y11 - y11.mean(1, keepdims=True) - y11.mean(0, keepdims=True) + gm) ** 2).sum() / 15
    msbs.append(msb)
    mses.append(mse)
    ests.append((5 * msb + 18 * mse) / 23)
target = se2 + sb2 * 5 * 4 / 23
E.check("s26p11a", abs(np.mean(msbs) - 30) < 0.3 and abs(np.mean(mses) - 5) < 0.05 and abs(np.mean(ests) - target) < 0.1,
        f"E[MS_ブロック]≈{np.mean(msbs):.2f}、E[MS_誤差]≈{np.mean(mses):.3f}、式の分子/分母の平均 {np.mean(ests):.3f}（期待値 {target:.3f}）")
RE = (5 * 30 + 18 * 5) / (23 * 5)
E.choice("s26p11a", f"\\mathrm{{RE}}={f(RE, 2)}$。完全無作為化法で同じ精度を得るには、約{f(RE, 1)}倍", "公式")

# ---- 大問12 層別抽出のシミュレーション
W12 = np.array([0.5, 0.3, 0.2])
S12 = np.array([4, 10, 20.0])
n_prop = 100 * W12
n_ney = 100 * W12 * S12 / (W12 * S12).sum()
vprop = (W12 ** 2 * S12 ** 2 / n_prop).sum()
vney = (W12 ** 2 * S12 ** 2 / n_ney).sum()
E.num("s26p12a", rnd(vprop / vney, 2), f"層ごとの分散の和（n_h={np.round(n_ney, 1)}）")
est_p = sum(W12[h] * rng.normal(0, S12[h], (100000, int(n_prop[h]))).mean(1) for h in range(3))
E.check("s26p12a", abs(est_p.var() / vprop - 1) < 0.02, f"比例配分のシミュレーション分散 {est_p.var():.4f}（式 {vprop:.4f}）")

# ---- 大問13
# (a) 自己共分散を時点ごとに計算して定常性を判定（シミュレーション）
T13, R13 = 6, 400_000
Aa, Bb = rng.normal(size=R13), rng.normal(size=R13)
om = 0.7
Xi = np.array([Aa * np.cos(om * t) + Bb * np.sin(om * t) for t in range(T13)])
Xiv = np.array([Aa for _ in range(T13)])
Xii = np.cumsum(rng.normal(size=(T13, R13)), axis=0)
cov_i = [np.mean(Xi[t] * Xi[t + 1]) for t in range(T13 - 1)]
E.check("s26p13a", np.ptp(cov_i) < 0.02 and abs(np.mean(cov_i) - math.cos(om)) < 0.02 and np.ptp([Xiv[t].var() for t in range(T13)]) < 1e-12,
        f"(i) の Cov(X_t, X_t+1) は t によらず cos ω={math.cos(om):.3f}、(iv) は一定")
E.check("s26p13a", Xii[-1].var() / Xii[0].var() > 5, f"(ii) の分散は t とともに増える（{Xii[0].var():.2f}→{Xii[-1].var():.2f}）")
E.choice("s26p13a", "(i)と(iv)", "定常性の判定")
# (b) numpy の固有値分解
S13 = np.array([[100, 6], [6, 1.0]])
w13, v13 = np.linalg.eigh(S13)
R13m = S13 / np.sqrt(np.outer(np.diag(S13), np.diag(S13)))
wr, vr = np.linalg.eigh(R13m)
E.check("s26p13b", abs(abs(v13[0, -1]) - 0.998) < 0.001 and abs(wr[-1] / 2 - 0.8) < 1e-12 and np.allclose(np.abs(vr[:, -1]), [2 ** -0.5] * 2),
        f"共分散の第 1 固有ベクトル {np.round(v13[:, -1], 3)}、相関の固有値 {wr}")
E.choice("s26p13b", f"寄与率は約{f(w13[-1] / w13.sum(), 3)}。相関行列（相関係数0.6）：第1主成分は$(1,1)/\\sqrt2$の向きで寄与率0.8", "numpy eigh")
# (c) 各点の y f とヒンジ損失
P13 = np.array([[1, 0], [2, 1], [0, 0], [0.5, 0.5], [1, 2]])
y13 = np.array([1, 1, -1, -1, 1])
m13 = y13 * (P13 @ np.array([2, -1]) - 1)
E.check("s26p13c", list(np.where(m13 <= 1)[0] + 1) == [1, 3, 4, 5] and abs(np.maximum(0, 1 - m13).sum() - 2.5) < 1e-12, f"y f = {m13}")
E.choice("s26p13c", f"サポートベクターはP1・P3・P4・P5、ヒンジ損失の合計2.5、マージンの幅{f(2 / math.sqrt(5), 3)}", "numpy で y f")

# ---- 大問14
# (a) ガワーの類似度を変数ごとに
E.num("s26p14a", rnd((1 + (1 - abs(2.0 - 6.5) / 10) + (1 - abs(1 - 4) / 6)) / 3, 3), "変数ごとの類似度の平均")
# (b) 潜在的な結果の表から
Y1 = np.array([3, 3, 4, 2, 1, 1])
Y0 = np.array([5, 4, 6, 2, 1, 2])
T14 = np.array([1, 1, 1, 0, 0, 0], bool)
ate, att = (Y1 - Y0).mean(), (Y1 - Y0)[T14].mean()
naive = Y1[T14].mean() - Y0[~T14].mean()
E.check("s26p14b", abs(naive - (att + (Y0[T14].mean() - Y0[~T14].mean()))) < 1e-12, "単純な比較 = ATT + 選択の偏り")
E.choice("s26p14b", f"ATE$={f(ate, 2)}$、ATT$={f(att, 2)}$、単純な比較$={f(naive, 2)}$（パッチを当てたサーバの方が件数が多く見える）", "numpy で平均")

E.done()
