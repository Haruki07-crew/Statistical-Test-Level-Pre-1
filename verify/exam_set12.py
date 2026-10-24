"""模擬試験 第12回（set12）の検算。generator/set12.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・数値積分・総当たり）で正解を求め直し、問題文に示した数表・関数値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(12)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value, what=""):
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']} {what}")


def distinct(iid, d):
    xs = nums(iid)
    E.check(iid, len({round(x, d) for x in xs}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {xs}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1 正規の比とコーシー
# (a) 極座標変換のヤコビアンを sympy で、偏角が一様になることと P(|R|>3) をシミュレーションと scipy で確かめる
d_, th_ = sp.symbols("d theta", positive=True)
Jac = sp.Matrix([d_ * sp.cos(th_), d_ * sp.sin(th_)]).jacobian([d_, th_]).det()
E.check("s12p01a", sp.simplify(Jac - d_) == 0, f"極座標のヤコビアン {sp.simplify(Jac)}")
r_, w_, sg = sp.symbols("r w sigma", positive=True)
dens = 2 * sp.integrate(w_ / (2 * sp.pi * sg ** 2) * sp.exp(-(1 + r_ ** 2) * w_ ** 2 / (2 * sg ** 2)), (w_, 0, sp.oo))
E.check("s12p01a", sp.simplify(dens - 1 / (sp.pi * (1 + r_ ** 2))) == 0, f"別ルート（比の変数変換）で R は標準コーシー {sp.simplify(dens)}")
X = rng.normal(0, 3.0, 1_000_000)
Y = rng.normal(0, 3.0, 1_000_000)
R = X / Y
ang = np.mod(np.arctan2(Y, X), 2 * np.pi)
E.check("s12p01a", abs(np.histogram(ang, bins=8, range=(0, 2 * np.pi))[0] / 1e6 - 1 / 8).max() < 0.003, "σ=3 のシミュレーションで偏角が一様")
E.check("s12p01a", close((np.abs(R) > 3).mean(), 2 * stats.cauchy.sf(3), 0.002) and close((np.abs(R) <= 1).mean(), 0.5, 0.003),
        f"シミュレーション P(|R|>3)={(np.abs(R) > 3).mean():.4f}")
E.check("s12p01a", close(math.atan(1 / 3), 0.3218, 5e-5), "arctan(1/3)=0.3218")
E.choice("s12p01a", f"${f(2 * stats.cauchy.sf(3), 3)}$", "scipy cauchy.sf(3) の 2 倍")
distinct("s12p01a", 3)
for v, s_ in ((math.pi, 3.1416), (math.atan(math.sqrt(10)), 1.2645), (math.atan(10), 1.4711),
              (stats.norm.cdf(1), 0.8413), (stats.norm.cdf(3.16), 0.9992)):
    E.check("s12p01b", close(v, s_, 5e-5), f"関数値 {s_}")
Rm = stats.cauchy.rvs(size=(200_000, 10), random_state=rng).mean(axis=1)
E.check("s12p01b", close((np.abs(Rm) <= 1).mean(), 0.5, 0.004), f"10 個の平均のシミュレーション {(np.abs(Rm) <= 1).mean():.4f}")
E.choice("s12p01b", f"${f(stats.cauchy.cdf(1) - stats.cauchy.cdf(-1), 3)}$", "scipy cauchy")
distinct("s12p01b", 3)

# ---- 大問2 抜取検査とベリー・エッセン
E.check("s12p02a", f(0.95 ** 19, 4) == "0.3774" and f(0.95 ** 20, 4) == "0.3585", "0.95^19, 0.95^20")
E.choice("s12p02a", f"${f(stats.binom.sf(1, 20, 0.05), 3)}$", "scipy binom.sf(1,20,0.05)")
distinct("s12p02a", 3)
# (b) 加算推定量 (S+1)/(m+2) の平均二乗誤差を二項分布の全列挙で
sv = np.arange(0, 49)
pmf = stats.binom.pmf(sv, 48, 0.05)
mse = float((pmf * ((sv + 1) / 50 - 0.05) ** 2).sum())
E.choice("s12p02b", f"${f(mse / 0.05 ** 2, 3)}$", "二項分布を全列挙した平均二乗誤差 ÷ ε²")
ex = float((pmf * (np.abs((sv + 1) / 50 - 0.05) >= 0.05 - 1e-12)).sum())
E.check("s12p02b", ex <= mse / 0.05 ** 2, f"実際の確率 {ex:.4f} は上界以下")
mses = [float((stats.binom.pmf(np.arange(m_ + 1), m_, 0.05) * ((np.arange(m_ + 1) + 1) / (m_ + 2) - 0.05) ** 2).sum()) for m_ in (48, 480, 4800)]
E.check("s12p02b", mses[0] > mses[1] > mses[2] and mses[2] < 1e-5, f"m を増やすと MSE→0（平均二乗収束） {mses}")
distinct("s12p02b", 3)

# ---- 大問3
g = stats.gamma(2, scale=0.5)
E.check("s12p03a", close(g.moment(1), 1, 1e-12) and close(g.moment(2), 1.5, 1e-12) and close(g.moment(3), 3, 1e-12),
        "示したモーメントはガンマ(2, 0.5) と一致")
E.num("s12p03a", rnd(float(g.stats(moments="s")), 2), "scipy gamma の歪度")
lam = 1.0 ** 2 + 0.5 ** 2 + 0.5 ** 2
E.choice("s12p03b", f"${f(stats.ncx2(3, lam).var(), 0)}$", "scipy ncx2(3, 1.5).var()")
Zs = rng.normal([1.0, 0.5, -0.5], 1, size=(1_000_000, 3))
Wv = (Zs ** 2).sum(axis=1).var()
E.check("s12p03b", close(Wv, 12, 0.15), f"シミュレーションの分散 {Wv:.3f}")
distinct("s12p03b", 2)

# ---- 大問4 一様分布の最大値
for v, s_ in ((math.log(0.05), -2.9957), (math.log(0.9), -0.10536), (math.log(0.1), -2.3026), (math.log(0.025), -3.6889)):
    E.check("s12p04a", close(v, s_, 6e-5 if abs(s_) > 1 else 6e-6), f"ln の値 {s_}")
nmin = next(n for n in range(1, 500) if stats.beta(n, 1).sf(0.9) >= 0.95)  # 最大値/θ ~ Beta(n,1)
E.choice("s12p04a", f"${nmin}$", "最大値/θ がベータ(n,1) に従うことから総当たり")
Um = rng.uniform(0, 1, size=(200_000, nmin)).max(axis=1)
E.check("s12p04a", (Um >= 0.9).mean() >= 0.947, f"n={nmin} のシミュレーション {(Um >= 0.9).mean():.4f}")

# ---- 大問5 制御変量
x = sp.symbols("x", positive=True)


def ex(h):
    return sp.integrate(h * sp.exp(-x), (x, 0, sp.oo))


vy = ex(sp.exp(-2 * x)) - ex(sp.exp(-x)) ** 2
cv = ex(x * sp.exp(-x)) - ex(x) * ex(sp.exp(-x))
vx = ex(x ** 2) - ex(x) ** 2
ratio = sp.nsimplify(1 - cv ** 2 / (vx * vy))
E.choice("s12p05a", f"${f(float(ratio), 3)}$", f"sympy で 1-ρ² = {ratio}")
Xs = rng.exponential(1, 2_000_000)
Ys = np.exp(-Xs)
c = -np.cov(Xs, Ys)[0, 1] / Xs.var()
E.check("s12p05a", close((Ys + c * (Xs - 1)).var() / Ys.var(), 0.25, 0.005), f"シミュレーション（推定した c={c:.3f}）の分散比")
distinct("s12p05a", 3)

# ---- 大問6
for (q, dfv), s_ in (((0.975, 11), 21.92), ((0.025, 11), 3.816), ((0.975, 12), 23.34), ((0.025, 12), 4.404), ((0.95, 2), 5.991)):
    E.check("s12p06a", close(stats.chi2.ppf(q, dfv), s_, 0.006 if s_ > 10 else 6e-4), f"χ² の点 {s_}")
E.check("s12p06a", close(stats.t.ppf(0.975, 11), 2.201, 5e-4) and close(math.sqrt(13 / 12), 1.0408, 5e-5), "t_11(0.025), √(13/12)")
h = stats.t.ppf(0.975, 11) * 0.22 * math.sqrt(1 + 1 / 12)
E.choice("s12p06a", f"${f(h, 3)}$", "scipy の t 点で予測区間の半幅")
smp = rng.normal(0, 0.22, size=(200_000, 13))
xb = smp[:, :12].mean(axis=1)
sd = smp[:, :12].std(axis=1, ddof=1)
covp = (np.abs(smp[:, 12] - xb) <= stats.t.ppf(0.975, 11) * sd * math.sqrt(13 / 12)).mean()
E.check("s12p06a", close(covp, 0.95, 0.003), f"予測区間の被覆率 {covp:.4f}")
distinct("s12p06a", 3)
s2 = np.array([0.0484, 0.0225, 0.0729])
for v, s_ in ((math.log(s2.mean()), -3.0379), (math.log(0.0484), -3.0283), (math.log(0.0225), -3.7942), (math.log(0.0729), -2.6187)):
    E.check("s12p06b", close(v, s_, 6e-5), f"ln の値 {s_}")
# scipy.stats.bartlett は分散だけでは呼べないので、不偏分散がちょうど s2 になるデータを作って渡す
base = rng.normal(size=12)
base = (base - base.mean()) / base.std(ddof=1)
data = [base * math.sqrt(v) + 100 for v in s2]
E.num("s12p06b", rnd(stats.bartlett(*data).statistic, 2), "scipy.stats.bartlett")

# ---- 大問7
E.check("s12p07a", close(math.asin(math.sqrt(0.06)), 0.2475, 5e-5) and close(math.asin(math.sqrt(0.03)), 0.1741, 5e-5), "arcsin の値")
z = (math.asin(math.sqrt(0.06)) - math.asin(math.sqrt(0.03))) / math.sqrt(1 / 1200 + 1 / 2000)
E.num("s12p07a", rnd(z, 2), "丸めなしの逆正弦変換")
ps = rng.binomial(300, 0.04, 400_000) / 300
E.check("s12p07a", close(np.arcsin(np.sqrt(ps)).var() * 1200, 1, 0.05), "変換後の分散 ≈ 1/(4n) をシミュレーションで確認")
tab = np.array([[6, 2], [2, 6]])
E.choice("s12p07b", f"${f(stats.chi2_contingency(tab, correction=False)[0], 2)}$", "scipy chi2_contingency（修正なし）")
E.check("s12p07b", f(stats.chi2_contingency(tab, correction=True)[0], 2) in E.item("s12p07b")["choices"][1], "イェーツ修正の値（選択肢 1）")
distinct("s12p07b", 2)

# ---- 大問8 ブラウン運動をシミュレーション
for zz, s_ in ((1.25, 0.8944), (0.63, 0.7357), (2.50, 0.9938)):
    E.check("s12p08a", close(stats.norm.cdf(zz), s_, 5e-5), f"Φ({zz})")
nst, T = 2000, 4.0
dt = T / nst
hits = 0
for _ in range(10):
    inc = rng.normal(0, math.sqrt(0.04 * dt), size=(2_000, nst))
    hits += (np.cumsum(inc, axis=1).max(axis=1) >= 0.5).sum()
psim = hits / 20_000
E.check("s12p08a", close(psim, 0.211, 0.02), f"離散化シミュレーション {psim:.4f}（離散化でわずかに小さく出る）")
E.choice("s12p08a", f"${f(2 * stats.norm.sf(0.5, 0, math.sqrt(0.16)), 3)}$", "2P(B_T≥a)")

# ---- 大問9
basis = np.array([1, 16, max(16 - 10, 0)])
E.choice("s12p09a", f"${f(basis @ np.array([50.0, 1.2, -1.5]), 1)}$", "基底ベクトルとの内積")
distinct("s12p09a", 1)
tg = np.linspace(0, 20, 10)
Xd = np.column_stack([np.ones(10), tg, np.maximum(tg - 10, 0)])
Hm = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
E.check("s12p09b", close(np.trace(Hm), 3, 1e-10), "実際のスプライン計画でハット行列の対角和が 3")
# 残差の分散 σ²(1-h) を実際の計画（等間隔）でのシミュレーションで確かめ、答えは分数で
ysim = rng.normal(size=(200_000, 10)) * 1.0
Esim = ysim - ysim @ Hm.T
E.check("s12p09b", np.allclose(Esim.var(axis=0), 1 - np.diag(Hm), atol=0.01), "残差の分散が σ²(1-h_ii) になる（シミュレーション）")
h10 = Fr(3) - Fr("2.30")
s2_ = Fr("6.30") / (10 - 3)
E.num("s12p09b", rnd(math.sqrt(float(s2_ * (1 - h10))), 2), "分数で h=3-2.30、s²=6.30/7")
for q, s_ in ((10 / 11, 1.335), (0.909, 1.335), (0.95, 1.645)):
    E.check("s12p09c", close(stats.norm.ppf(q), s_, 6e-4), f"分位点 {s_}")
qmax = stats.norm.ppf((10 - 0.5) / 10)
qmin = stats.norm.ppf((1 - 0.5) / 10)
right = "右" if 2.95 > qmax else "左"
E.check("s12p09c", 2.95 > qmax and -1.20 > qmin, "最大の点は直線の外側（上）、最小の点は内側（上）→ 右の裾だけ重い")
E.choice("s12p09c", f"{f(qmax, 3)}、{right}の裾が正規分布より重い", "scipy norm.ppf(0.95) と両端の位置")

# ---- 大問10 ポアソン回帰を数値最適化で当てはめ直し、対数尤度から逸脱度を求める
from scipy.optimize import minimize  # noqa: E402
yv10 = np.array([3, 7, 5, 9])
xs10 = np.array([1.0, 2.0, 3.0, 4.0])
nll = lambda b: -(yv10 * (b[0] + b[1] * xs10) - np.exp(b[0] + b[1] * xs10)).sum()  # noqa: E731
fit = minimize(nll, [1.0, 0.1], method="BFGS", options={"gtol": 1e-12})
mu10 = np.exp(fit.x[0] + fit.x[1] * xs10)
E.check("s12p10a", all(f(m_, 2) == s_ for m_, s_ in zip(mu10, ["3.81", "5.00", "6.57", "8.62"])), f"当てはめ値 {np.round(mu10, 4)} が表と一致")
for v, s_ in ((math.log(3 / 3.81), -0.23902), (math.log(7 / 5.00), 0.33647), (math.log(5 / 6.57), -0.27308), (math.log(9 / 8.62), 0.04314)):
    E.check("s12p10a", close(v, s_, 6e-6), f"ln の値 {s_}")
dev = 2 * (stats.poisson.logpmf(yv10, yv10).sum() - stats.poisson.logpmf(yv10, mu10).sum())
nearest("s12p10a", dev, "（丸めない当てはめ値での逸脱度）")
E.choice("s12p10a", f"${f(dev, 2)}$", "scipy の対数尤度（飽和モデルとの差の2倍）")
distinct("s12p10a", 2)

# ---- 大問11 分割区法
E.check("s12p11a", close(stats.f.ppf(0.95, 2, 6), 5.14, 0.005), f"F(2,6) 5% 点 {stats.f.ppf(0.95, 2, 6):.3f}")
# A の効果がないときに MS_A / MS_主区誤差 が F(2,6) に従う（第1種の誤りが 5%）ことをシミュレーションで確認
nb, na, nbb, reps = 4, 3, 2, 20_000
yv = rng.normal(0, 1, (reps, nb, 1, 1)) + rng.normal(0, 1.5, (reps, nb, na, 1)) + rng.normal(0, 0.7, (reps, nb, na, nbb))
m_ba = yv.mean(axis=3)
gA = m_ba.mean(axis=1)
gB = m_ba.mean(axis=2)
gm = m_ba.mean(axis=(1, 2))
ssa = nb * nbb * ((gA - gm[:, None]) ** 2).sum(axis=1)
sse1 = nbb * ((m_ba - gA[:, None, :] - gB[:, :, None] + gm[:, None, None]) ** 2).sum(axis=(1, 2))
Fs = (ssa / 2) / (sse1 / 6)
rej = (Fs > stats.f.ppf(0.95, 2, 6)).mean()
E.check("s12p11a", close(rej, 0.05, 0.006), f"主区誤差で割った F の第1種の誤り {rej:.4f}")
E.choice("s12p11a", f"${f(12.0 / 2.0, 1)}$", "MS_A / MS_主区誤差")
distinct("s12p11a", 1)

# ---- 大問12
yv12 = np.array([3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10])
means = [yv12[i::3].mean() for i in range(3)]
vs = float(np.mean([(m - yv12.mean()) ** 2 for m in means]))
E.choice("s12p12a", f"${f(vs, 3)}$", "3 通りの系統標本を列挙")
allm = [np.mean(cc) for cc in combinations(yv12, 4)]
E.check("s12p12a", E.item("s12p12a")["choices"][3] == f"${f(np.var(allm), 3)}$", "単純無作為抽出の分散（選択肢 3）を全組合せで確認")
distinct("s12p12a", 3)
times = np.array([10, 12, 15, 15, 18, 20, 25, 30])
event = np.array([1, 0, 1, 1, 0, 1, 0, 1])
H = sum(event[times == t].sum() / (times >= t).sum() for t in np.unique(times[event == 1]) if t <= 20)
E.num("s12p12b", rnd(H, 3), "リスク集合を配列から数えて計算")

# ---- 大問13
evs = np.array([2.4, 1.3, 0.7, 0.4])
# 固有値 2.4,1.3,0.7,0.4,0.2 をもつ 5×5 の相関行列を実際に作って確認（固有値を保つ回転で対角を 1 にできる）
lam5 = np.append(evs, 5 - evs.sum())
M5 = np.diag(lam5)
for _ in range(200):  # ギブンス回転で対角を 1 にそろえる（Bendel–Mickey 法の簡易版）
    dg = np.diag(M5)
    i = int(np.argmax(dg))
    j = int(np.argmin(dg))
    if abs(dg[i] - 1) < 1e-12 and abs(dg[j] - 1) < 1e-12:
        break
    a_, b_, c_ = M5[i, i], M5[j, j], M5[i, j]
    # 回転角 θ で新しい (i,i) 成分を 1 にする
    th = sp.symbols("th")
    sol = [float(s_) for s_ in sp.solve(sp.Eq(a_ * sp.cos(th) ** 2 + b_ * sp.sin(th) ** 2 + 2 * c_ * sp.sin(th) * sp.cos(th), 1), th)
           if abs(float(sp.im(s_))) < 1e-12]
    tval = sol[0]
    G = np.eye(5)
    G[i, i] = G[j, j] = math.cos(tval)
    G[i, j] = math.sin(tval)
    G[j, i] = -math.sin(tval)
    M5 = G @ M5 @ G.T
E.check("s12p13a", np.allclose(np.diag(M5), 1, atol=1e-8), "対角が 1 の相関行列を構成できた")
ev = np.sort(np.linalg.eigvalsh(M5))[::-1]
k = int((ev > 1).sum())
E.choice("s12p13a", f"{k} 個、{f(100 * ev[:k].sum() / ev.sum(), 1)} %", "構成した相関行列の固有値の数値計算")
r12, r13, r23 = 0.56, 0.48, 0.42
lam_ = np.sqrt([r12 * r13 / r23, r12 * r23 / r13, r13 * r23 / r12])
Rimp = np.outer(lam_, lam_)
E.check("s12p13b", close(Rimp[0, 1], r12, 1e-12) and close(Rimp[0, 2], r13, 1e-12) and close(Rimp[1, 2], r23, 1e-12),
        "負荷量の外積が 3 つの相関を再現")
E.choice("s12p13b", f"${f(1 - lam_[0] ** 2, 2)}$", "独自分散")
distinct("s12p13b", 2)
# (c) ICA：順序と尺度の不定性を数値で確認（非正規な独立信号を混合し、置換・尺度を変えても観測が同じ）
S3 = rng.uniform(-1, 1, size=(3, 1000))
A3 = rng.normal(size=(3, 3))
P3 = np.eye(3)[[2, 0, 1]]
D3 = np.diag([2.0, -0.5, 3.0])
E.check("s12p13c", np.allclose(A3 @ S3, (A3 @ np.linalg.inv(D3) @ np.linalg.inv(P3)) @ (P3 @ D3 @ S3)), "置換と尺度を変えても観測は同じ（順序・尺度は決まらない）")
# 正規な独立信号は回転しても独立（共分散が単位行列のまま、高次の情報なし）
G2 = rng.normal(size=(2, 200_000))
th = 0.7
Rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
E.check("s12p13c", np.allclose(np.cov(Rot @ G2), np.eye(2), atol=0.01) and abs(stats.kurtosis((Rot @ G2)[0])) < 0.03,
        "正規信号は回転後も無相関・尖度0で区別できない（選択肢 2 は誤り）")
E.check("s12p13c", E.item("s12p13c")["answer"] == 1 and "順序と尺度" in E.item("s12p13c")["choices"][1], "正解は順序と尺度の不定性の記述")

# ---- 大問14
tab = np.array([[30, 12], [22, 63]])
E.num("s12p14a", rnd(stats.contingency.odds_ratio(tab, kind="sample").statistic, 2), "scipy odds_ratio(sample)")
# (b) 未観測の U（0/1）を含む構造モデルを具体的に作り、観測確率を再現することを確かめてから、
#     介入 do(X=x) のもとでの P(Y=1) を U について直接平均する（フロントドアの式は使わない）。
a = {1: 0.9, 0: 0.1}  # P(U=1 | X=x)
px = {1: 0.4, 0: 0.6}
q = sum(px[x_] * a[x_] for x_ in (0, 1))
pm = {1: 0.10, 0: 0.50}
dy = {1: 0.10 / (a[1] - a[0]), 0: 0.02 / (a[1] - a[0])}
yu = {}
for m_, base_ in ((1, 0.20), (0, 0.04)):
    yu[(m_, 0)] = base_ - a[0] * dy[m_]
    yu[(m_, 1)] = yu[(m_, 0)] + dy[m_]
obs = {(m_, x_): (1 - a[x_]) * yu[(m_, 0)] + a[x_] * yu[(m_, 1)] for m_ in (0, 1) for x_ in (0, 1)}
target = {(1, 1): 0.30, (1, 0): 0.20, (0, 1): 0.06, (0, 0): 0.04}
E.check("s12p14b", all(close(obs[k_], target[k_], 1e-12) for k_ in target) and all(0 < v < 1 for v in yu.values()),
        "U つき構造モデルが観測の P(Y|M,X) と P(X) を再現する")
Nn = 2_000_000
Usim = rng.random(Nn) < q
res_do = []
for x_ in (0, 1):
    Msim = rng.random(Nn) < pm[x_]
    py_ = np.where(Msim, np.where(Usim, yu[(1, 1)], yu[(1, 0)]), np.where(Usim, yu[(0, 1)], yu[(0, 0)]))
    res_do.append((rng.random(Nn) < py_).mean())
exact = [sum((pm[x_] if m_ else 1 - pm[x_]) * ((1 - q) * yu[(m_, 0)] + q * yu[(m_, 1)]) for m_ in (0, 1)) for x_ in (0, 1)]
cnt = {(1, 1): (200, 60), (0, 1): (1800, 108), (1, 0): (1500, 300), (0, 0): (1500, 60)}  # (M,X): (ロット数, 腐敗)
E.check("s12p14b", close(2000 / 5000, px[1], 1e-12) and close(200 / 2000, pm[1], 1e-12) and close(1500 / 3000, pm[0], 1e-12)
        and all(close(k_ / n_, target[mx], 1e-12) for mx, (n_, k_) in cnt.items()), "問題文の集計表が P(X)・P(M|X)・P(Y|M,X) を与える")
it14 = E.item("s12p14b")["question"]
E.check("s12p14b", all(str(v) in it14 for v in (5000, 200, 60, 1800, 108, 1500, 300)), "集計表の数値が問題文にある")
E.num("s12p14b", rnd(exact[1], 3), "構造モデルで U について平均した P(Y=1|do(X=1))")
E.check("s12p14b", close(res_do[1], exact[1], 0.001), f"介入のシミュレーション {res_do[1]:.4f}")

E.done()
