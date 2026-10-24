"""模擬試験 第8回（set08）の検算。generator/set08.py とは別の方法（状態の総当たり・scipy の分布関数・
sympy の積分・シミュレーション・数値最適化・別のアルゴリズム）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(8)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return str(abs(v) if v == 0 else v)


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def table_ok(iid, pairs, tol, what):
    bad = [(a, b) for a, b in pairs if abs(a - b) > tol]
    E.check(iid, not bad, f"{what}（ずれ {bad}）")


# ---- 大問1
# (a) 4 機器（L, A1, A2, D）の稼働・停止 16 通りを総当たり
pr = {"L": 0.99, "A1": 0.95, "A2": 0.95, "D": 0.98}
p_down = p_ldown = 0.0
for st in product([0, 1], repeat=4):
    s_ = dict(zip(pr, st))
    w = np.prod([pr[k] if v else 1 - pr[k] for k, v in s_.items()])
    up = s_["L"] and (s_["A1"] or s_["A2"]) and s_["D"]
    if not up:
        p_down += w
        if not s_["L"]:
            p_ldown += w
E.choice("s08p01a", f"${f(p_ldown / p_down, 3)}$", "16 状態の総当たりで P(L 停止 | サービス停止)")
# (b) h(t)=0.2t は形状 2・尺度 √10 のワイブル分布
W = stats.weibull_min(2, scale=np.sqrt(10))
E.check("s08p01b", abs(W.pdf(1.7) / W.sf(1.7) - 0.2 * 1.7) < 1e-12, "ワイブル(2, √10) のハザードが 0.2t")
E.num("s08p01b", rnd(W.sf(3) / W.sf(2), 3), "scipy のワイブル分布で S(3)/S(2)")
table_ok("s08p01b", [(np.exp(-0.1), 0.9048), (np.exp(-0.4), 0.6703), (np.exp(-0.5), 0.6065), (np.exp(-0.9), 0.4066)], 5e-5, "e^x の値")

# ---- 大問2
t, u = sp.symbols("t u", positive=True)
l1, l2 = sp.Rational(1, 2), sp.Rational(1, 4)
conv = sp.simplify(sp.integrate(l1 * sp.exp(-l1 * u) * l2 * sp.exp(-l2 * (t - u)), (u, 0, t)))
target = sp.Rational(1, 2) * (sp.exp(-t / 4) - sp.exp(-t / 2))
E.check("s08p02a", sp.simplify(conv - target) == 0, f"sympy の畳み込み {conv}")
E.choice("s08p02a", "$0.5\\left(e^{-0.25t}-e^{-0.5t}\\right)$", "正解の式を含む選択肢がただ1つ")
for k, wrong in [(0, -target), (1, sp.Rational(1, 8) * sp.exp(-sp.Rational(3, 4) * t)),
                 (2, sp.Rational(1, 2) * (sp.exp(-t / 2) / 2 + sp.exp(-t / 4) / 4)), (3, target / 4)]:
    E.check("s08p02a", sp.simplify(wrong - conv) != 0, f"選択肢 {k} は畳み込みの結果と異なる")
# (b) 分散を数値積分で求め、n を 1 から順に調べる
dens = lambda x: 0.5 * (np.exp(-x / 4) - np.exp(-x / 2))
m1 = integrate.quad(lambda x: x * dens(x), 0, np.inf)[0]
m2 = integrate.quad(lambda x: x * x * dens(x), 0, np.inf)[0]
vT = m2 - m1 ** 2
n_ch = next(n for n in range(1, 10 ** 5) if vT / (n * 0.25) <= 0.05 + 1e-12)
z = stats.norm.ppf(0.975)
n_cl = next(n for n in range(1, 10 ** 5) if 1.96 * np.sqrt(vT / n) <= 0.5)
E.choice("s08p02b", f"チェビシェフ：$n={n_ch}$、中心極限定理：$n={n_cl}$", f"数値積分の分散 {vT:.4f} と n の総当たり")
E.check("s08p02b", abs(z - 1.96) < 5e-4 and abs(stats.norm.ppf(0.95) - 1.645) < 5e-4, "正規分布の点")
Tbar = rng.exponential(2, size=(20000, n_cl)).mean(1) + rng.exponential(4, size=(20000, n_cl)).mean(1)
cov = (np.abs(Tbar - 6) < 0.5).mean()
E.check("s08p02b", abs(cov - 0.95) < 0.01, f"n={n_cl} のシミュレーションで P(|T̄−μ|<0.5) = {cov:.4f}")

# ---- 大問3
Sig = np.array([[16.0, 6.0], [6.0, 9.0]])
res = optimize.minimize_scalar(lambda w: np.array([w, 1 - w]) @ Sig @ np.array([w, 1 - w]), bounds=(-2, 3), method="bounded",
                               options={"xatol": 1e-10})
E.num("s08p03a", rnd(res.fun, 2), f"数値最適化（w={res.x:.5f}）")
X = rng.multivariate_normal([0, 0], Sig, size=1_000_000)
vsim = (X @ np.array([res.x, 1 - res.x])).var()
E.check("s08p03a", abs(vsim - res.fun) < 0.05, f"シミュレーションの分散 {vsim:.4f}")
# (b) 条件付き確率：同時密度の数値積分で P(X2>4 | X1=6) を出す
mvn = stats.multivariate_normal([0, 0], Sig)
num = integrate.quad(lambda x2: mvn.pdf([6, x2]), 4, np.inf)[0]
den = integrate.quad(lambda x2: mvn.pdf([6, x2]), -np.inf, np.inf)[0]
nearest("s08p03b", num / den)
E.choice("s08p03b", "$0.251$", "表の Φ(0.67) を使った値")
table_ok("s08p03b", [(stats.norm.cdf(z_), v_) for z_, v_ in [(0.26, 0.6026), (0.58, 0.7190), (0.67, 0.7486), (1.33, 0.9082)]], 6e-5, "Φ の表")

# ---- 大問4：リンドレーの漸化式で M/M/1 をシミュレーション
lam, mu = 12, 15
nq = 2_000_000
A = rng.exponential(1 / lam, nq)
S = rng.exponential(1 / mu, nq)
wq = np.empty(nq)
wq[0] = 0
for i in range(1, nq):
    wq[i] = max(0.0, wq[i - 1] + S[i - 1] - A[i])
sim = wq[100_000:].mean()
E.check("s08p04a", abs(sim - 4 / 15) < 0.01, f"シミュレーションの平均待ち時間 {sim:.4f} 秒")
nearest("s08p04a", 1000 * sim)

# ---- 大問5
n_cp = next(n for n in range(1, 10 ** 5) if stats.beta.ppf(0.95, 1, n) <= 0.002)
E.num("s08p05a", n_cp, "クロッパー・ピアソン上限 beta.ppf(0.95, 1, n) を総当たり")
table_ok("s08p05a", [(np.log(0.05), -2.9957), (np.log(0.025), -3.6889), (np.log(0.998), -0.0020020)], 6e-5, "対数の値")
E.check("s08p05a", 1 - 0.05 ** (1 / n_cp) <= 0.002 < 1 - 0.05 ** (1 / (n_cp - 1)), "n-1 では上限が 0.002 を超える")
b = stats.binom(36, 1 / 3)
zz = (6 - b.mean()) / b.std()
E.choice("s08p05b", f"z={f(zz, 2)}", "scipy の二項分布の平均と標準偏差で標準化")
E.check("s08p05b", stats.binomtest(6, 36, 1 / 3).pvalue < 0.05, "正確な二項検定でも 5% で有意（解説の結論）")

# ---- 大問6
old = np.array([128, 135, 117, 142, 150, 131, 119, 126])
new = np.array([116, 130, 120, 134, 140, 125, 121, 122])
tr = stats.ttest_rel(old, new)
E.choice("s08p06a", f"${f(tr.statistic, 2)}$", "scipy ttest_rel")
E.check("s08p06a", f(stats.ttest_ind(old, new).statistic, 2) in E.item("s08p06a")["choices"][0], "対応を無視した ttest_ind の値は選択肢 0")
E.check("s08p06a", abs(stats.t.ppf(0.975, 7) - 2.365) < 5e-4 and stats.ttest_rel(old, new).pvalue < 0.05, "t(7) の 2.5% 点と有意性")
bt = stats.binomtest(int((old - new > 0).sum()), 8, 0.5, alternative="greater")
E.choice("s08p06b", f"${f(bt.pvalue, 3)}$", "scipy binomtest(alternative='greater')")

# ---- 大問7
# (a) 乱数データで「係数の分散 = σ²/{Sxx(1−R²)}」を確かめ、R²=0.84 に当てはめる
n = 60
Z = rng.normal(size=(n, 3))
Z[:, 0] += 1.5 * Z[:, 1] + Z[:, 2]
Xd = np.column_stack([np.ones(n), Z])
XtXi = np.linalg.inv(Xd.T @ Xd)
aux = np.linalg.lstsq(Xd[:, [0, 2, 3]], Z[:, 0], rcond=None)[0]
ra = Z[:, 0] - Xd[:, [0, 2, 3]] @ aux
R2a = 1 - (ra @ ra) / ((Z[:, 0] - Z[:, 0].mean()) ** 2).sum()
Sxx = ((Z[:, 0] - Z[:, 0].mean()) ** 2).sum()
E.check("s08p07a", abs(XtXi[1, 1] - 1 / (Sxx * (1 - R2a))) < 1e-12, "(X'X)^{-1} の対角 = 1/{Sxx(1−R²)}（乱数データ）")
vif = 1 / (1 - 0.84)
E.choice("s08p07a", f"VIF $={f(vif, 2)}$、標準誤差は ${f(np.sqrt(vif), 1)}$ 倍", "VIF と √VIF")
# (b)
pval = stats.chi2.sf(60 * 0.15, 3)
E.check("s08p07b", 0.01 < pval < 0.05, f"χ²(3) の p 値 {pval:.4f} は 1% と 5% の間")
E.check("s08p07b", E.item("s08p07b")["answer"] == 0 and "自由度 3 のカイ二乗分布の 5% 点を超える" in E.item("s08p07b")["choices"][0],
        "正解の文は 5% で棄却")
E.check("s08p07b", stats.chi2.sf(9.0, 4) > 0.05 and stats.chi2.ppf(0.99, 3) > 9.0, "他の選択肢の結論は誤り（df4 なら棄却しない・1% では棄却しない）")
table_ok("s08p07b", [(stats.chi2.ppf(0.95, 2), 5.99), (stats.chi2.ppf(0.95, 3), 7.81), (stats.chi2.ppf(0.99, 3), 11.34),
                     (stats.chi2.ppf(0.95, 4), 9.49), (stats.chi2.ppf(0.99, 4), 13.28)], 6e-3, "χ² の点")
# (c) 共分散の連立方程式を sympy で解く
b1, c2 = sp.symbols("b1 c2")
S11, S22, S12 = 4, 25, sp.Rational(6)   # s1=2, s2=5, r=0.6
c1 = sp.Rational(62, 100) * S11            # 単回帰の傾き = c1/S11
sol = sp.solve([S11 * b1 + S12 * sp.Rational(12, 100) - c1, S12 * b1 + S22 * sp.Rational(12, 100) - c2], [b1, c2])
E.choice("s08p07c", f"${f(float(sol[b1]), 2)}$", "正規方程式を共分散で書いて sympy で解く")
xx = rng.normal(size=(50, 2))
xx[:, 1] += 0.8 * xx[:, 0]
yy = 1 + 0.5 * xx[:, 0] + 0.3 * xx[:, 1] + rng.normal(size=50)
bf = np.linalg.lstsq(np.column_stack([np.ones(50), xx]), yy, rcond=None)[0]
bs = np.polyfit(xx[:, 0], yy, 1)[0]
dl = np.polyfit(xx[:, 0], xx[:, 1], 1)[0]
E.check("s08p07c", abs(bs - (bf[1] + bf[2] * dl)) < 1e-10, "乱数データで単回帰の傾き = b1 + b2 δ（厳密）")

# ---- 大問8
# (a) 小さな母集団で πps 系統抽出を作り、HT 推定量の不偏性を総当たりで確かめてから値を計算
sizes = np.array([3, 1, 4, 2, 5, 1, 2, 2], float)
yv = 2 * sizes + rng.integers(0, 3, len(sizes))
k = 2
pi = k * sizes / sizes.sum()
cum = np.concatenate([[0], np.cumsum(pi)])
starts = np.linspace(0, 1, 2001)[:-1] + 0.00025
ests = []
for s0 in starts:
    pts = s0 + np.arange(k)
    sel = [int(np.searchsorted(cum, q, side="right") - 1) for q in pts]
    ests.append(sum(yv[j] / pi[j] for j in sel))
E.check("s08p08a", abs(np.mean(ests) - yv.sum()) < 0.05, f"πps 系統抽出で HT の期待値 {np.mean(ests):.3f} ≈ 総計 {yv.sum():.0f}")
E.num("s08p08a", float(np.dot([45, 18, 96], 1 / np.array([0.30, 0.15, 0.60]))), "重み 1/π の内積")
# (b) ポアソン GLM をシミュレーションデータで当てはめて、オフセットの扱いを確かめる
mreq = rng.uniform(10, 100, 4000)
vv = rng.integers(0, 2, 4000)
yc = rng.poisson(mreq * np.exp(-1.2 + 0.35 * vv))
Xg = np.column_stack([np.ones(4000), vv])
beta = np.zeros(2)
for _ in range(30):   # IRLS
    eta = np.log(mreq) + Xg @ beta
    muv = np.exp(eta)
    zw = eta - np.log(mreq) + (yc - muv) / muv
    beta = np.linalg.solve(Xg.T @ (muv[:, None] * Xg), Xg.T @ (muv * zw))
E.check("s08p08b", abs(beta[0] + 1.2) < 0.03 and abs(beta[1] - 0.35) < 0.03, f"IRLS で係数を復元 {beta}")
E.choice("s08p08b", f"${f(50 * np.exp(-1.2 + 0.35), 2)}$", "50·exp(β0+β1)")
table_ok("s08p08b", [(np.exp(-0.85), 0.4274), (np.exp(-1.2), 0.3012), (np.exp(0.35), 1.4191)], 5e-5, "e^x の値")

# ---- 大問9：±1 コードの回帰で交互作用係数を求め、効果 = 2×係数
runs = {"1": 120, "a": 96, "b": 112, "ab": 80, "c": 110, "ac": 90, "bc": 104, "abc": 70}
rows, yy9 = [], []
for key, val in runs.items():
    a_, b_, c_ = [1 if ch in key else -1 for ch in "abc"]
    rows.append([1, a_, b_, c_, a_ * b_, a_ * c_, b_ * c_, a_ * b_ * c_])
    yy9.append(val)
coef = np.linalg.solve(np.array(rows, float), np.array(yy9, float))
E.num("s08p09a", 2 * coef[4], "飽和モデルの回帰係数 ×2")

# ---- 大問10
P = np.array([[1, 2], [2, 1], [2, 3], [8, 8], [9, 6], [4, 4]], float)
best = None
for mask in range(1, 2 ** 6 - 1):
    g = np.array([(mask >> i) & 1 for i in range(6)])
    w_ = sum(((P[g == h] - P[g == h].mean(0)) ** 2).sum() for h in (0, 1))
    if best is None or w_ < best[0]:
        best = (w_, g)
E.choice("s08p10a", f"${f(best[0], 2)}$", "2 分割の総当たりで最小のクラスター内平方和（収束値と一致）")
E.check("s08p10a", sorted(np.where(best[1] == best[1][0])[0].tolist()) == [0, 1, 2, 5], "最良の分割は {P1,P2,P3,P6} と {P4,P5}")
# (b) 分数で厳密に
pa_, f1a, f1n, f2a, f2n = Fraction(2, 100), Fraction(6, 10), Fraction(1, 100), Fraction(7, 10), Fraction(2, 10)
joint = {"a": pa_ * f1a * (1 - f2a), "n": (1 - pa_) * f1n * (1 - f2n)}
post = joint["a"] / (joint["a"] + joint["n"])
E.choice("s08p10b", f"${f(float(post), 3)}$", f"分数で厳密に {post}")
Ncl = 2_000_000
att = rng.random(Ncl) < 0.02
F1 = np.where(att, rng.random(Ncl) < 0.6, rng.random(Ncl) < 0.01)
F2 = np.where(att, rng.random(Ncl) < 0.7, rng.random(Ncl) < 0.2)
sel = F1 & ~F2
E.check("s08p10b", abs(att[sel].mean() - float(post)) < 0.02, f"シミュレーション {att[sel].mean():.4f}")
# (c) 指定した辺だけをもつ精度行列を作り、条件付き共分散を計算
idx = {v: i for i, v in enumerate("RCMTE")}
K = np.eye(5) * 3.0
for a_, b_ in ["RC", "RM", "CT", "MT", "TE"]:
    K[idx[a_], idx[b_]] = K[idx[b_], idx[a_]] = 0.6
assert np.all(np.linalg.eigvalsh(K) > 0)
Sg = np.linalg.inv(K)


def pcov(a, b, given):
    A_, G = [idx[a], idx[b]], [idx[g] for g in given]
    if not G:
        return Sg[A_[0], A_[1]]
    Sc = Sg[np.ix_(A_, A_)] - Sg[np.ix_(A_, G)] @ np.linalg.inv(Sg[np.ix_(G, G)]) @ Sg[np.ix_(G, A_)]
    return Sc[0, 1]


truth = [abs(pcov("R", "E", "T")) < 1e-12, abs(pcov("R", "T", "C")) < 1e-12, abs(pcov("C", "M", "R")) < 1e-12,
         abs(pcov("R", "E", "")) < 1e-12, abs(Sg[idx["C"], idx["M"]]) < 1e-12]
E.check("s08p10c", truth == [True, False, False, False, False], f"条件付き共分散が 0 になるかの判定 {truth}")

# ---- 大問11：Yule–Walker を行列で解き、その係数の AR(2) の理論自己相関が r1, r2 に戻るか
R = np.array([[1, 0.8], [0.8, 1]])
ph = np.linalg.solve(R, [0.8, 0.5])
rho1 = ph[0] / (1 - ph[1])
rho2 = ph[0] * rho1 + ph[1]
E.check("s08p11a", abs(rho1 - 0.8) < 1e-12 and abs(rho2 - 0.5) < 1e-12, "推定した係数の理論自己相関が (0.8, 0.5)")
roots = np.roots([-ph[1], -ph[0], 1])
E.check("s08p11a", np.all(np.abs(roots) > 1), f"特性方程式の根 {np.abs(roots)} が単位円の外（定常）")
E.num("s08p11a", rnd(100 + ph @ [10, 6], 2), "numpy で YW を解いて予測（小数第2位に丸め）")
xs = np.zeros(400_000)
eps = rng.normal(size=400_000)
for i in range(2, len(xs)):
    xs[i] = ph[0] * xs[i - 1] + ph[1] * xs[i - 2] + eps[i]
xc = xs[1000:] - xs[1000:].mean()
r1s = (xc[1:] @ xc[:-1]) / (xc @ xc)
E.check("s08p11a", abs(r1s - 0.8) < 0.01, f"シミュレーションの標本自己相関 r1 = {r1s:.4f}")

# ---- 大問12：エフロンの「右への再配分」アルゴリズムで KM を計算
obs = [(1, 1), (2, 1), (2, 0), (3, 1), (4, 1), (4, 1), (5, 0), (6, 1), (8, 0), (9, 1)]
order = sorted(range(10), key=lambda i: (obs[i][0], -obs[i][1]))   # 同時刻は復旧を先に
mass = np.full(10, 0.1)
for pos, i in enumerate(order):
    if obs[i][1] == 0:
        rest = order[pos + 1:]
        if rest:
            mass[rest] += mass[i] / len(rest)
        mass[i] = 0
S6 = 1 - sum(mass[i] for i in range(10) if obs[i][1] == 1 and obs[i][0] <= 6)
E.choice("s08p12a", f"${f(S6, 3)}$", f"右への再配分で S(6) = {S6:.5f}")

# ---- 大問13
x = stats.pareto(2.5, scale=10).ppf(0.84)
nearest("s08p13a", x)
E.choice("s08p13a", f"${f(10 * 2.0814, 2)}$", "与えた 0.16^{-0.4} で計算した値")
table_ok("s08p13a", [(0.16 ** 0.4, 0.4805), (0.16 ** -0.4, 2.0814), (0.84 ** -0.4, 1.0722)], 6e-5, "べき乗の値")
table_ok("s08p13a", [(0.16 ** -2.5, 97.66)], 5e-3, "0.16^{-2.5}（小数第2位）")
th, nn = 1.0, 10
Us = rng.uniform(0, th, size=(400_000, nn))
T1 = 2 * Us.mean(1)
T2 = (nn + 1) / nn * Us.max(1)
ratio = T2.var() / T1.var()
E.check("s08p13b", abs(T1.mean() - 1) < 0.002 and abs(T2.mean() - 1) < 0.002, "両方とも不偏（シミュレーション）")
nearest("s08p13b", ratio)
yv_ = sp.symbols("y", positive=True)
nS = sp.Integer(nn)
EY = sp.integrate(yv_ * nS * yv_ ** (nS - 1), (yv_, 0, 1))
EY2 = sp.integrate(yv_ ** 2 * nS * yv_ ** (nS - 1), (yv_, 0, 1))
vT2 = ((nS + 1) / nS) ** 2 * (EY2 - EY ** 2)
E.choice("s08p13b", f"${f(float(vT2 / (sp.Rational(1, 3) / nS)), 2)}$", "sympy で順序統計量の積分")

# ---- 大問14：隠れた U をもつ SCM を作り、真の介入効果とフロントドア式が一致することを確かめてから値を計算
for trial in range(3):
    pu = rng.uniform(0.2, 0.8)
    pxu = rng.uniform(0.1, 0.9, 2)          # P(X=1|U)
    pmx = rng.uniform(0.1, 0.9, 2)          # P(M=1|X)
    pymu = rng.uniform(0.05, 0.95, (2, 2))  # P(Y=1|M,U)
    PU = np.array([1 - pu, pu])
    j = np.zeros((2, 2, 2, 2))              # U, X, M, Y
    for uu, xv, mv, yv2 in product([0, 1], repeat=4):
        j[uu, xv, mv, yv2] = (PU[uu] * (pxu[uu] if xv else 1 - pxu[uu]) * (pmx[xv] if mv else 1 - pmx[xv])
                              * (pymu[mv, uu] if yv2 else 1 - pymu[mv, uu]))
    obsd = j.sum(0)                          # X, M, Y
    Px = obsd.sum((1, 2))
    Pm_x = obsd.sum(2) / Px[:, None]
    Py_xm = obsd[:, :, 1] / obsd.sum(2)
    fd = [sum(Pm_x[xv, mv] * sum(Py_xm[x2, mv] * Px[x2] for x2 in (0, 1)) for mv in (0, 1)) for xv in (0, 1)]
    true = [sum(PU[uu] * sum((pmx[xv] if mv else 1 - pmx[xv]) * pymu[mv, uu] for mv in (0, 1)) for uu in (0, 1)) for xv in (0, 1)]
    E.check("s08p14a", np.allclose(fd, true), f"乱数 SCM {trial}: フロントドア式 = 真の介入効果")
Pm_x = np.array([[0.8, 0.2], [0.3, 0.7]])
Py_xm = np.array([[0.05, 0.20], [0.10, 0.30]])
Px = np.array([0.6, 0.4])
do = np.einsum("xm,m->x", Pm_x, np.einsum("km,k->m", Py_xm, Px))
E.num("s08p14a", do[1] - do[0], "einsum でフロントドア式")
E.done()
