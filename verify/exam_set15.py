"""模擬試験 第15回（set15）の検算。generator/set15.py とは別の方法（scipy の分布関数・数値積分・シミュレーション・
sympy・行列計算・定義からの直接計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(15)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.findall(r"-?\d+\.?\d*", str(c))[-1]) for c in E.item(iid)["choices"]]


def nearest(iid, value, what=""):
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']} {what}")


def distinct(iid, d):
    xs = nums(iid)
    E.check(iid, len({f(x, d) for x in xs}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {xs}")


# ---- 大問1
# (a) 数値積分で条件付き確率
fxy = lambda x, y: x + y
E.check("s15p01a", abs(integrate.dblquad(lambda y, x: fxy(x, y), 0, 1, 0, 1)[0] - 1) < 1e-10, "同時密度の積分が 1")
fx = integrate.quad(lambda y: fxy(0.25, y), 0, 1)[0]
pc = integrate.quad(lambda y: fxy(0.25, y), 0.5, 1)[0] / fx
E.choice("s15p01a", f"${f(pc, 3)}$", "数値積分で P(Y>1/2 | X=1/4)")
distinct("s15p01a", 3)
# 誤答「X<=1/4 で条件付け」の値も数値積分で確認
le = integrate.dblquad(lambda y, x: fxy(x, y), 0, 0.25, 0.5, 1)[0] / integrate.dblquad(lambda y, x: fxy(x, y), 0, 0.25, 0, 1)[0]
E.check("s15p01a", f(le, 3) in E.item("s15p01a")["choices"][4], f"誤答 X<=1/4 の値 {le:.4f}")
# (b) scipy の χ² 上側確率と表の値、シミュレーション
for c, v in [(2.25, "0.9869"), (13.5, "0.1413"), (20.25, "0.0164"), (22.5, "0.0074")]:
    E.check("s15p01b", f(stats.chi2.sf(c, 9), 4) == v, f"表 P(χ²_9>{c})={v}")
E.choice("s15p01b", f"${f(stats.chi2.sf(9 * 1.5 ** 2, 9), 4)}$", "scipy chi2.sf(9×1.5², 9)")
X = rng.normal(0, 1, size=(400_000, 10))
sim = np.mean(X.std(axis=1, ddof=1) > 1.5)
E.check("s15p01b", abs(sim - 0.0164) < 0.001, f"シミュレーションで P(S>1.5σ)={sim:.4f}")
# (c) scipy の χ² 分位点で区間
lo, hi = stats.chi2.ppf([0.025, 0.975], 9)
E.check("s15p01c", f(lo, 2) == "2.70" and f(hi, 2) == "19.02", f"χ²_9 の 2.5% 点 {lo:.4f}, {hi:.4f}")
ci = (math.sqrt(9 * 2.6 ** 2 / 19.02), math.sqrt(9 * 2.6 ** 2 / 2.70))
E.choice("s15p01c", f"({f(ci[0], 2)},\\{f(ci[1], 2)})", "表の点から σ の区間")
ci_exact = (math.sqrt(9 * 2.6 ** 2 / hi), math.sqrt(9 * 2.6 ** 2 / lo))
E.check("s15p01c", f(ci_exact[0], 2) == f(ci[0], 2) and f(ci_exact[1], 2) == f(ci[1], 2), f"正確な分位点でも同じ {ci_exact}")
# 被覆確率のシミュレーション
Xs = rng.normal(0, 2.0, size=(100_000, 10))
S2 = Xs.var(axis=1, ddof=1)
cov = np.mean((9 * S2 / 19.02 < 4.0) & (4.0 < 9 * S2 / 2.70))
E.check("s15p01c", abs(cov - 0.95) < 0.005, f"区間の被覆確率 {cov:.4f}")

# ---- 大問2
w = [Fraction(5, 10), Fraction(3, 10), Fraction(2, 10)]
fa = [Fraction(10, 100), Fraction(20, 100), Fraction(35, 100)]
post = w[2] * fa[2] / sum(a * b for a, b in zip(w, fa))
E.choice("s15p02a", f"${f(float(post), 3)}$", f"分数で厳密に {post}")
distinct("s15p02a", 3)
# (b) scipy の負の二項分布（失敗回数で数える）：3 回目の成功までの失敗が 7 回
v = stats.nbinom.pmf(7, 3, 0.2)
E.num("s15p02b", rnd(v, 4), "scipy nbinom.pmf(7, 3, 0.2)")
sim = rng.random((200_000, 10)) < 0.2
cs = sim.cumsum(axis=1)
hit = np.mean((cs[:, 9] == 3) & (cs[:, 8] == 2))
E.check("s15p02b", abs(hit - v) < 0.002, f"シミュレーション {hit:.4f}")

# ---- 大問3
x = np.array([2, 7, 1, 5, 8, 3, 0, 6])
D = ((x - x.mean()) ** 2).sum() / x.mean()
c7, c7h, c8 = stats.chi2.isf(0.05, 7), stats.chi2.isf(0.025, 7), stats.chi2.isf(0.05, 8)
E.check("s15p03a", (f(c7, 2), f(c7h, 2), f(c8, 2)) == ("14.07", "16.01", "15.51"), "χ² の点（表）")
E.check("s15p03a", abs(D - 15.0) < 1e-12 and stats.chi2.sf(D, 7) < 0.05 < stats.chi2.sf(D, 8), f"D={D}, p(df7)={stats.chi2.sf(D, 7):.4f}")
E.choice("s15p03a", "自由度7の上側5%点14.07を超えるので、帰無仮説を棄却し過分散と判断する", "D>14.07 で棄却・過分散")
E.check("s15p03a", f"D={f(D, 1)}" in E.item("s15p03a")["choices"][1].replace("$", ""), "正解の選択肢の D の値")
# (b) scipy のポアソン分布
g = (0.05 - stats.poisson.cdf(1, 6)) / stats.poisson.pmf(2, 6)
E.check("s15p03b", f(math.exp(-6), 6) == "0.002479", "e^{-6}=0.002479")
E.num("s15p03b", rnd(g, 2), f"scipy poisson で γ={g:.5f}")
size = stats.poisson.cdf(1, 6) + rnd(g, 2) * stats.poisson.pmf(2, 6)
E.check("s15p03b", abs(size - 0.05) < 0.0005, f"γ=0.73 での検定の大きさ {size:.5f}")

# ---- 大問4
# (a) マルコフの上界は 2 点分布で到達（平均 12 のまま P(X>=60)=0.2）
pts, pr = np.array([0, 60]), np.array([0.8, 0.2])
E.check("s15p04a", abs((pts * pr).sum() - 12) < 1e-12, "2 点分布 {0:0.8, 60:0.2} の平均が 12（上界が最良）")
E.choice("s15p04a", "$0.200$", "E[X]/a")
E.check("s15p04a", f(math.exp(-5), 4) == "0.0067", "誤答の e^{-5}")
# (b) sympy で連立方程式
k, th = sp.symbols("k theta", positive=True)
sol = sp.solve([k * th - sp.Rational(126, 10), k * th ** 2 - sp.Rational(504, 10)], [k, th], dict=True)[0]
E.num("s15p04b", rnd(float(sol[k]), 2), f"sympy で k={sol[k]}")
smp = rng.gamma(float(sol[k]), float(sol[th]), size=2_000_000)
E.check("s15p04b", abs(smp.mean() - 12.6) < 0.05 and abs(smp.var() - 50.4) < 0.5, "推定値のガンマ分布の平均・分散")

# ---- 大問5
R = np.array([[1, 0.50, -0.60], [0.50, 1, -0.55], [-0.60, -0.55, 1]])
P = np.linalg.inv(R)
pcor = -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
E.choice("s15p05a", f"${f(pcor, 3)}$", "相関行列の逆行列から偏相関")
distinct("s15p05a", 3)
# (b) 決定係数を経由した F（総平方和は任意の値で打ち消し合う）
TSS = 7.3
r2r, r2f = 1 - 4.20 / TSS, 1 - 3.40 / TSS
F = ((r2f - r2r) / 2) / ((1 - r2f) / 56)
E.choice("s15p05b", f"${f(F, 2)}$", "決定係数の差から F")
E.check("s15p05b", f(stats.f.isf(0.05, 2, 56), 2) == "3.16", "F_0.05(2,56)=3.16（解説）")
distinct("s15p05b", 2)
# (c) Cp を総当たりで
s2 = 3.40 / (60 - 4)
cps = {m: rss / s2 - 60 + 2 * kk for m, rss, kk in [("M1", 4.20, 2), ("M2", 3.46, 3), ("M3", 3.70, 3), ("M4", 3.40, 4)]}
best = min(cps, key=cps.get)
E.choice("s15p05c", f"{best}（$C_p={f(cps[best], 2)}$）", f"Cp {cps}")

# ---- 大問6
# 関係式 (n-2)s² = (n-3)s_(i)² + e_i²/(1-h_i) を、乱数データで当てはめ直して確認
xx = rng.uniform(0, 10, 12)
yy = 1 + 2 * xx + rng.normal(0, 1, 12)
Xd = np.column_stack([np.ones(12), xx])
H = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
res = yy - H @ yy
s2all = (res ** 2).sum() / 10
i = 3
Xm, ym = np.delete(Xd, i, 0), np.delete(yy, i)
bm = np.linalg.lstsq(Xm, ym, rcond=None)[0]
s2m = ((ym - Xm @ bm) ** 2).sum() / 9
E.check("s15p06a", abs(10 * s2all - (9 * s2m + res[i] ** 2 / (1 - H[i, i]))) < 1e-9, "関係式を再当てはめで確認")
# r_i から t_i を求める別の公式
ri = 1.60 / (0.8 * math.sqrt(1 - 0.36))
ti = ri * math.sqrt((12 - 3) / (12 - 2 - ri ** 2))
E.choice("s15p06a", f"${f(ti, 2)}$", "t_i = r_i √((n-3)/(n-2-r_i²))")
E.check("s15p06a", f(stats.t.isf(0.025, 9), 3) == "2.262", "t_0.025(9)=2.262")
distinct("s15p06a", 2)

# ---- 大問7
a = [1, 2, 3, 4, 5, 6, 7, 8]
b = [2, 1, 4, 3, 6, 8, 5, 7]
E.num("s15p07a", rnd(stats.spearmanr(a, b).statistic, 3), "scipy spearmanr")
eta1, eta2 = 1.2 - 1.5, 2.8 - 1.5
pv = expit(eta2) - expit(eta1)
nearest("s15p07b", pv, "（正確な exp で計算）")
E.choice("s15p07b", f"${f(pv, 3)}$", "scipy expit の差")
for z, v in [(0.3, "1.3499"), (-1.3, "0.2725"), (-1.2, "0.3012"), (-2.8, "0.0608")]:
    E.check("s15p07b", f(math.exp(z), 4) == v, f"e^{z}={v}")
distinct("s15p07b", 3)

# ---- 大問8：基本行列 N=(I-Q)^{-1} の行和
Qm = np.array([[0.85, 0.10], [0.20, 0.60]])
Nm = np.linalg.inv(np.eye(2) - Qm)
E.num("s15p08a", rnd(Nm[0].sum(), 1), "基本行列の行和")
# シミュレーション
st = np.zeros(200_000, dtype=int)
steps = np.zeros(200_000)
alive = np.ones(200_000, dtype=bool)
T = np.array([[0.85, 0.10, 0.05], [0.20, 0.60, 0.20]])
while alive.any():
    u = rng.random(alive.sum())
    cur = st[alive]
    cum = T[cur].cumsum(axis=1)
    nxt = (u[:, None] > cum).sum(axis=1)
    steps[alive] += 1
    st[alive] = nxt
    alive[alive] = nxt < 2
E.check("s15p08a", abs(steps.mean() - 12.5) < 0.1, f"シミュレーションの平均 {steps.mean():.3f}")

# ---- 大問9：個票を作って共分散の比（2 段階最小二乗法と同じ）
n1 = n0 = 1000
Z = np.r_[np.ones(n1), np.zeros(n0)]
Xp = np.r_[np.ones(700), np.zeros(300), np.ones(300), np.zeros(700)]
Yp = np.r_[np.ones(120), np.zeros(880), np.ones(180), np.zeros(820)]
iv = np.cov(Z, Yp)[0, 1] / np.cov(Z, Xp)[0, 1]
# 2SLS：第1段で X を Z に回帰、第2段で Y を当てはめ値に回帰
Zd = np.column_stack([np.ones(2000), Z])
xhat = Zd @ np.linalg.lstsq(Zd, Xp, rcond=None)[0]
b2 = np.linalg.lstsq(np.column_stack([np.ones(2000), xhat]), Yp, rcond=None)[0][1]
E.check("s15p09a", abs(iv - b2) < 1e-9, f"共分散の比と 2SLS が一致 {iv:.4f}")
E.choice("s15p09a", f"${f(b2, 3)}$", "2 段階最小二乗法")
distinct("s15p09a", 3)

# ---- 大問10
from scipy.stats import studentized_range  # noqa: E402
qv = studentized_range.isf(0.05, 3, 15)
E.check("s15p10a", f(qv, 2) == "3.67" and f(stats.t.isf(0.025, 15), 3) == "2.131", f"q={qv:.4f}, t=2.131")
hsd = 3.67 * math.sqrt(0.36 / 6)
means = {"A": 2.10, "B": 2.85, "C": 3.30}
sig = [(u, v) for u in means for v in means if u < v and abs(means[u] - means[v]) > hsd]
E.check("s15p10a", sig == [("A", "C")], f"HSD={hsd:.4f} で有意な組 {sig}")
E.choice("s15p10a", f"基準{f(hsd, 2)}、AとCの組だけが有意", "HSD とその結論")
# 同時有意水準のシミュレーション（帰無仮説のもと、3 群すべての対）
Ysim = rng.normal(0, 0.6, size=(100_000, 3, 6))
mm = Ysim.mean(axis=2)
mse = Ysim.var(axis=2, ddof=1).mean(axis=1)
rng_ = mm.max(axis=1) - mm.min(axis=1)
fw = np.mean(rng_ > 3.67 * np.sqrt(mse / 6))
E.check("s15p10a", abs(fw - 0.05) < 0.004, f"テューキー法の群全体の誤り率 {fw:.4f}")

# ---- 大問11
Wh = np.array([0.6, 0.3, 0.1])
Sh = np.array([8, 12, 20])
mh = np.array([70, 60, 50])
nh = np.array([60, 30, 10])
sims = (Wh * rng.normal(mh, Sh / np.sqrt(nh), size=(400_000, 3))).sum(axis=1)
se_exact = math.sqrt((Wh ** 2 * Sh ** 2 / nh).sum())
E.choice("s15p11a", f"${f(se_exact, 2)}$", "Σ W_h² S_h²/n_h の平方根")
E.check("s15p11a", abs(sims.std() - se_exact) < 0.01, f"シミュレーションの標準誤差 {sims.std():.4f}")
distinct("s15p11a", 2)
Qv = np.array([64.2, 65.0, 63.6, 64.8, 64.4])
Uv = np.array([1.30, 1.25, 1.35, 1.28, 1.32])
Tv = Uv.mean() + (1 + 1 / 5) * Qv.var(ddof=1)
E.num("s15p11b", rnd(math.sqrt(Tv), 2), f"numpy で T={Tv:.4f}")

# ---- 大問12
Lm = np.array([[0.80, 0.20], [0.75, 0.10], [0.70, 0.30], [0.20, 0.85], [0.10, 0.65]])
contrib = (Lm ** 2).sum(axis=0)
E.choice("s15p12a", f"${f(contrib[1] / 5, 3)}$", "負荷量行列の列の 2 乗和 / 5")
# 直交回転しても共通性と寄与の合計は変わらない（解説の主張）
ang = 0.4
Rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
L2 = Lm @ Rot
E.check("s15p12a", np.allclose((L2 ** 2).sum(axis=1), (Lm ** 2).sum(axis=1)) and abs((L2 ** 2).sum() - (Lm ** 2).sum()) < 1e-12,
        "直交回転で共通性と寄与の合計が不変")
distinct("s15p12a", 3)
# 確認的因子分析の自由度：15 個の分散共分散 − (負荷 5 + 独自分散 5 + 因子相関 1)
E.check("s15p12b", 5 * 6 // 2 - (5 + 5 + 1) == 4, "CFA の自由度 4")
E.choice("s15p12b", f"${f(math.sqrt((13.6 - 4) / (4 * 300)), 3)}$", "RMSEA")
distinct("s15p12b", 3)

# ---- 大問13
pts = np.array([[130, 6.0], [140, 6.2], [128, 7.6]])
Dm = squareform(pdist(pts / np.array([15, 1.0])))
np.fill_diagonal(Dm, np.inf)
i, j = np.unravel_index(np.argmin(Dm), Dm.shape)
pair = "PQR"[min(i, j)] + "と" + "PQR"[max(i, j)]
E.choice("s15p13a", f"{pair}（距離{f(Dm[i, j], 2)}）", "scipy pdist（標準化）")
Draw = squareform(pdist(pts))
np.fill_diagonal(Draw, np.inf)
E.check("s15p13a", np.unravel_index(np.argmin(Draw), Draw.shape) in [(0, 2), (2, 0)], "標準化しないと P と R が最も近い")

# ---- 大問14
tabs = [np.array([[10, 40], [20, 30]]), np.array([[6, 54], [9, 51]])]
num = sum(t[0, 0] * t[1, 1] / t.sum() for t in tabs)
den = sum(t[0, 1] * t[1, 0] / t.sum() for t in tabs)
E.num("s15p14a", rnd(num / den, 2), "MH 推定量（表から直接）")
# MH 推定量は各層のオッズ比の重み付き平均（重み b_k c_k / n_k）でもある
ors = [t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]) for t in tabs]
wts = [t[0, 1] * t[1, 0] / t.sum() for t in tabs]
E.check("s15p14a", abs(sum(o * w for o, w in zip(ors, wts)) / sum(wts) - num / den) < 1e-12, "重み付き平均の形と一致")
pb = stats.betabinom.pmf(2, 2, 25, 5)
E.num("s15p14b", rnd(pb, 3), "scipy betabinom.pmf(2; n=2, a=25, b=5)")
pq = integrate.quad(lambda p: p ** 2 * stats.beta(25, 5).pdf(p), 0, 1)[0]
E.check("s15p14b", abs(pq - pb) < 1e-10, "事後分布での E[p²] を数値積分")

E.done()
