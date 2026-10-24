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
r_, w_, sg = sp.symbols("r w sigma", positive=True)
dens = 2 * sp.integrate(w_ / (2 * sp.pi * sg ** 2) * sp.exp(-(1 + r_ ** 2) * w_ ** 2 / (2 * sg ** 2)), (w_, 0, sp.oo))
E.check("s12p01a", sp.simplify(dens - 1 / (sp.pi * (1 + r_ ** 2))) == 0, f"sympy で周辺密度 {sp.simplify(dens)}")
X = rng.normal(0, 3.0, 1_000_000)
Y = rng.normal(0, 3.0, 1_000_000)
R = X / Y
E.check("s12p01a", close((np.abs(R) <= 1).mean(), 0.5, 0.003) and close((R <= 2).mean(), stats.cauchy.cdf(2), 0.003),
        "σ=3 のシミュレーションでも標準コーシーの確率と一致")
E.choice("s12p01a", r"\dfrac{1}{\pi(1+r^2)}", "正解の式")
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
p = 0.05
Bv = np.array([0, 1])
pr = np.array([1 - p, p])
rho3 = (np.abs(Bv - p) ** 3 * pr).sum()
sig = math.sqrt((pr * (Bv - p) ** 2).sum())
E.check("s12p02b", close(math.sqrt(0.0475), 0.2179, 5e-5), "√0.0475")
be = 0.4748 * rho3 / (sig ** 3 * math.sqrt(400))
nearest("s12p02b", be, "（丸めなしの上界）")
kk = np.arange(0, 60)
dist = np.max(np.abs(stats.binom.cdf(kk, 400, p) - stats.norm.cdf((kk - 20) / math.sqrt(400 * p * (1 - p)))))
E.check("s12p02b", dist <= be, f"実際の最大のずれ {dist:.4f} は上界以下")
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
ss = 11 * 0.0484
lo, hi = math.sqrt(ss / stats.chi2.ppf(0.975, 11)), math.sqrt(ss / stats.chi2.ppf(0.025, 11))
E.choice("s12p06a", f"({f(lo, 3)},\\ {f(hi, 3)})", "scipy のパーセント点で区間")
smp = rng.normal(0, 0.2, size=(200_000, 12))
s2s = smp.var(axis=1, ddof=1)
cov = ((11 * s2s / stats.chi2.ppf(0.975, 11) <= 0.04) & (0.04 <= 11 * s2s / stats.chi2.ppf(0.025, 11))).mean()
E.check("s12p06a", close(cov, 0.95, 0.003), f"被覆率 {cov:.4f}")
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
E.num("s12p09b", rnd(float(Fr(3) - Fr("2.30")), 2), "分数で 3-2.30")
for q, s_ in ((9 / 11, 0.908), (0.85, 1.036), (0.90, 1.282), (0.95, 1.645)):
    E.check("s12p09c", close(stats.norm.ppf(q), s_, 6e-4), f"分位点 {s_}")
E.choice("s12p09c", f"${f(stats.norm.ppf((9 - 0.5) / 10), 3)}$", "scipy norm.ppf(0.85)")

# ---- 大問10 トービット：数値積分
mu, sgm = 0.5, 1.0
ey = integrate.quad(lambda y: y * stats.norm.pdf(y, mu, sgm), 0, np.inf)[0]
E.check("s12p10a", close(stats.norm.cdf(0.5), 0.6915, 5e-5) and close(stats.norm.pdf(0.5), 0.3521, 5e-5), "Φ(0.5), φ(0.5)")
E.choice("s12p10a", f"${f(ey, 3)}$", "数値積分 ∫_0^∞ y φ")
E.check("s12p10a", close(np.maximum(rng.normal(mu, sgm, 2_000_000), 0).mean(), ey, 0.002), "シミュレーション")
distinct("s12p10a", 3)

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
d = np.array([1.0, 2.0, 2.5, 3.0, 1.5, 2.0])
dh = np.array([1.0, 1.8, 2.5, 3.0, 1.7, 2.0])
E.choice("s12p13c", f"${f(np.linalg.norm(d - dh) / np.linalg.norm(d), 3)}$", "ノルムの比")
distinct("s12p13c", 3)

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
E.num("s12p14b", rnd(exact[1] - exact[0], 3), "構造モデルで U について平均した介入効果")
E.check("s12p14b", close(res_do[1] - res_do[0], exact[1] - exact[0], 0.002), f"介入のシミュレーション {res_do[1] - res_do[0]:.4f}")

E.done()
