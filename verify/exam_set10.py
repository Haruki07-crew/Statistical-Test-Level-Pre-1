"""模擬試験 第10回（set10）の検算。generator/set10.py とは別の方法（scipy の分布関数・シミュレーション・
総当たり・数値積分・実際に当てはめ直す回帰）で正解を求め直し、問題文に示した表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import product
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(10)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    out = []
    for c in E.item(iid)["choices"]:
        m = re.findall(r"-?\d+\.?\d*", str(c).replace("−", "-"))
        out.append([float(x) for x in m])
    return out


def nearest(iid, value, pos=0):
    vals = [v[pos] for v in nums(iid)]
    k = int(np.argmin([abs(x - value) for x in vals]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1
# (a) 4 通りの結果を総当たりして独立性を確かめる
outc = list(product([0, 1], repeat=2))
P = lambda ev: sum(Fr(1, 4) for o in outc if ev(o))
A = lambda o: o[0] == 1
B = lambda o: o[1] == 1
C = lambda o: o[0] + o[1] == 1
pair = all(P(lambda o, x=x, y=y: x(o) and y(o)) == P(x) * P(y) for x, y in [(A, B), (A, C), (B, C)])
triple = P(lambda o: A(o) and B(o) and C(o)) == P(A) * P(B) * P(C)
E.check("s10p01a", pair and not triple and P(C) == Fr(1, 2), "総当たり：対ごとに独立、3 つでは独立でない、P(C)=1/2")
E.check("s10p01a", P(lambda o: C(o) and A(o) and B(o)) != P(C) * P(lambda o: A(o) and B(o)), "C は A∩B と独立でない（選択肢 3 は誤り）")
E.choice("s10p01a", "それぞれ独立だが", "正解の文がただ1つ")
# (b) MH 推定量を「層別オッズ比の重み付き平均（重み b c / n）」として計算
tabs = [(12, 48, 15, 125), (20, 30, 30, 120)]
w = [b * c / sum(t) for (a, b, c, d), t in zip(tabs, tabs)]
ors = [a * d / (b * c) for a, b, c, d in tabs]
mh = sum(wi * oi for wi, oi in zip(w, ors)) / sum(w)
E.num("s10p01b", rnd(mh, 2), "層別オッズ比の重み付き平均")
# 条件付き最尤（共通オッズ比）の推定値も近いこと（MH は一致推定量）
def score(lp):
    psi = np.exp(lp)
    s = 0
    for a, b, c, d in tabs:
        m1, n1_, n2_ = a + c, a + b, c + d
        xs = np.arange(max(0, m1 - n2_), min(m1, n1_) + 1)
        lw = np.array([stats.hypergeom.logpmf(x, n1_ + n2_, n1_, m1) for x in xs]) + xs * lp
        pr = np.exp(lw - lw.max()); pr /= pr.sum()
        s += a - (xs * pr).sum()
    return s
cmle = np.exp(optimize.brentq(score, -3, 3))
E.check("s10p01b", abs(cmle - mh) < 0.05, f"条件付き最尤推定値 {cmle:.3f} と近い")

# ---- 大問2
mn = stats.multinomial(8, [0.6, 0.25, 0.15])
E.num("s10p02a", rnd(mn.pmf([5, 2, 1]), 3), "scipy multinomial.pmf")
joint = mn.pmf([5, 2, 1])
marg = sum(mn.pmf([5, r_, 3 - r_]) for r_ in range(4))
E.choice("s10p02b", f"${f(joint / marg, 3)}$", "同時確率 ÷ 周辺確率（総当たり）")
E.check("s10p02b", abs(marg - stats.binom.pmf(5, 8, 0.6)) < 1e-12, "S の周辺分布が B(8,0.6)")

# ---- 大問3：Σx と Σlog x が同じで標本分散の違う 2 組のデータを作り、尤度が一致することを確かめる
x1 = np.array([1.0, 4.0, 8.5])
def g(v):
    a, b = v
    c = x1.sum() - a - b
    return [np.log(a) + np.log(b) + np.log(c) - np.log(x1).sum(), a - 2.0]
sol = optimize.fsolve(g, [2.0, 3.0])
x2 = np.array([sol[0], sol[1], x1.sum() - sol.sum()])
ok = abs(x2.sum() - x1.sum()) < 1e-9 and abs(np.log(x2).sum() - np.log(x1).sum()) < 1e-9 and (x2 > 0).all()
diffs = [stats.gamma.logpdf(x1, a, scale=t).sum() - stats.gamma.logpdf(x2, a, scale=t).sum()
         for a, t in rng.uniform(0.5, 5, size=(20, 2))]
E.check("s10p03a", ok and max(abs(d) for d in diffs) < 1e-8, "Σx と Σlog x が等しい 2 組で尤度比が母数によらず 1（十分統計量）")
E.check("s10p03a", abs(x1.var() - x2.var()) > 0.1, f"2 組の標本分散 {x1.var():.3f}, {x2.var():.3f} は違う（モーメント推定量は T の関数でない）")
x3 = np.array([2.0, 4.0, 7.5])  # Σx は同じ、Σlog x が違う
E.check("s10p03a", np.ptp([stats.gamma.logpdf(x1, a, scale=t).sum() - stats.gamma.logpdf(x3, a, scale=t).sum()
                           for a, t in [(1, 1), (3, 2), (5, 0.5)]]) > 1e-3, "Σx だけ同じでは尤度比が母数に依存（x̄ だけでは十分でない）")
E.choice("s10p03a", "\\sumx_i,\\\\sum\\logx_i", "正解の選択肢")

# ---- 大問4
E.choice("s10p04a", f"${f(stats.binom.pmf(2, 6, 1 / 3), 3)}$", "binom.pmf(2, 6, 1/3)")
# シミュレーション：率 6/時の過程を発生させ、1 時間に 6 台だった回だけで最初の 20 分の台数を数える
lamh = 6.0
cnt = rng.poisson(lamh, size=600_000)
sel = np.where(cnt == 6)[0][:60_000]
first = np.array([(rng.uniform(0, 60, 6) < 20).sum() for _ in sel])
E.check("s10p04a", abs((first == 2).mean() - 0.3292) < 0.01, f"シミュレーション {(first == 2).mean():.4f}")
E.check("s10p04a", abs(np.exp(-2) - 0.1353) < 5e-5, "e^{-2}=0.1353")
ok37 = stats.norm.sf(60, 1.5 * 37, 0.4 * np.sqrt(37)) <= 0.05
ok38 = stats.norm.sf(60, 1.5 * 38, 0.4 * np.sqrt(38)) <= 0.05
E.check("s10p04b", ok37 and not ok38, "scipy：37 台は 5% 以下、38 台は超える")
E.choice("s10p04b", "37台", "正解の台数")
E.check("s10p04b", abs(stats.norm.isf(0.05) - 1.645) < 5e-4 and abs(stats.norm.isf(0.025) - 1.960) < 5e-4, "z 点")
W = np.minimum(rng.exponential(4, 400_000), rng.exponential(6, 400_000))
E.check("s10p04c", abs((W > 2).mean() - 0.4346) < 0.003, f"シミュレーション P(W>2)={(W > 2).mean():.4f}")
E.num("s10p04c", rnd(stats.expon.sf(2, scale=4) * stats.expon.sf(2, scale=6), 3), "scipy expon.sf の積")
E.check("s10p04c", abs(np.exp(-5 / 6) - 0.4346) < 5e-5 and abs(np.exp(-0.2) - 0.8187) < 5e-5, "e の値")

# ---- 大問5
ev = integrate.quad(lambda t: 600 / t / 10, 20, 30)[0]
nearest("s10p05a", ev)
E.check("s10p05a", abs(np.log(1.5) - 0.4055) < 5e-5, "log 1.5")
Ts = rng.uniform(20, 30, 1_000_000)
E.check("s10p05a", abs((600 / Ts).mean() - 24.33) < 0.02, f"シミュレーション E[V]={(600 / Ts).mean():.3f}")
Aa = np.array([38.5, 42.0, 35.2, 40.8, 45.3, 36.9, 39.4, 41.6])
Bb = np.array([35.3, 40.5, 36.0, 38.4, 41.2, 36.3, 37.5, 38.9])
dd = Aa - Bb
ci = stats.t.interval(0.95, 7, loc=dd.mean(), scale=stats.sem(dd))
E.num("s10p05b", rnd(ci[0], 2), f"scipy t.interval（正確な t 点）下限 {ci[0]:.4f}")
E.check("s10p05b", abs(stats.t.ppf(0.975, 7) - 2.365) < 5e-4 and abs(stats.t.ppf(0.975, 14) - 2.145) < 5e-4
        and abs(stats.t.ppf(0.95, 7) - 1.895) < 5e-4, "t の点")

# ---- 大問6
pv = {"A": 0.019, "B": 0.003, "C": 0.300, "D": 0.011, "E": 0.023}
# 閉検定手順（ボンフェローニで各交差仮説を検定）として求める：ホルム法と同値
names = list(pv)
rej = []
for h in names:
    subsets = [s_ for k in range(1, 6) for s_ in __import__("itertools").combinations(names, k) if h in s_]
    if all(min(pv[x] for x in s_) <= 0.05 / len(s_) for s_ in subsets):
        rej.append(h)
E.check("s10p06a", sorted(rej) == ["B", "D"], f"閉検定手順で棄却 {rej}")
E.check("s10p06a", E.item("s10p06a")["choices"].index("B と D") == E.item("s10p06a")["answer"], "正解の選択肢が「B と D」")
E.check("s10p06a", E.item("s10p06a")["choices"].count("B と D") == 1, "B と D の選択肢はただ1つ")
wv = [1.2, 3.5, 0.8, 2.1, 4.4, 1.7]
ov = [2.9, 5.1, 3.8, 6.2, 2.4, 4.9, 7.0, 3.1]
rs = stats.ranksums(wv, ov)
E.num("s10p06b", rnd(rs.statistic, 2), f"scipy ranksums の z={rs.statistic:.4f}")
E.check("s10p06b", abs(np.sqrt(60) - 7.746) < 5e-4, "√60")

# ---- 大問7
# (a) 実際に回帰を当てはめ直して t² = 部分 F の恒等式を確かめ、その式で値を出す
X = np.column_stack([np.ones(48), rng.normal(size=(48, 6))])
y = X @ rng.normal(size=7) + rng.normal(size=48)
b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
e = y - X @ b
s2 = e @ e / 41
cov = s2 * np.linalg.inv(X.T @ X)
tval = b[6] / np.sqrt(cov[6, 6])
sst = ((y - y.mean()) ** 2).sum()
R2f = 1 - e @ e / sst
br, *_ = np.linalg.lstsq(X[:, :6], y, rcond=None)
er = y - X[:, :6] @ br
R2r = 1 - er @ er / sst
E.check("s10p07a", abs((R2f - R2r) - tval ** 2 * (1 - R2f) / 41) < 1e-10, "乱数データで t² と決定係数の関係を確認")
Rr = optimize.brentq(lambda q: (0.82 - q) / ((1 - 0.82) / 41) - 2.5 ** 2, 0, 0.82)
E.choice("s10p07a", f"${f(Rr, 3)}$", "部分 F=t² の式を数値的に解く")
# (b) 削除残差の公式を乱数データで当てはめ直して確認し、値を計算
H = X @ np.linalg.inv(X.T @ X) @ X.T
i = 3
Xi, yi = np.delete(X, i, 0), np.delete(y, i)
bi, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
s2i = ((yi - Xi @ bi) ** 2).sum() / 40
E.check("s10p07b", abs(s2i - (e @ e - e[i] ** 2 / (1 - H[i, i])) / 40) < 1e-10, "当てはめ直しで削除後の誤差分散の公式を確認")
s2i_v = (900 - 11.2 ** 2 / 0.75) / 40
E.choice("s10p07b", f"${f(11.2 / np.sqrt(s2i_v * 0.75), 2)}$", "外的スチューデント化残差")
rint = 11.2 / np.sqrt(900 / 41 * 0.75)
E.check("s10p07b", abs(rint * np.sqrt(40 / (41 - rint ** 2)) - 11.2 / np.sqrt(s2i_v * 0.75)) < 1e-12, "内的から外的への変換式とも一致")

# ---- 大問8
# (a) r=-0.48 となるデータを作り、scipy の回帰の傾きの t 値と比べる
z1, z2 = rng.normal(size=(2, 22))
z1 = (z1 - z1.mean()) / z1.std()
z2 = z2 - z2.mean() - (z2 @ z1) / (z1 @ z1) * z1
z2 /= z2.std()
yy = -0.48 * z1 + np.sqrt(1 - 0.48 ** 2) * z2
lr = stats.linregress(z1, yy)
E.check("s10p08a", abs(lr.rvalue + 0.48) < 1e-12, "r=-0.48 のデータを作れた")
E.choice("s10p08a", f"${f(lr.slope / lr.stderr, 2)}$", "回帰の傾きの t 値（無相関検定と同値）")
E.check("s10p08a", abs(stats.t.ppf(0.975, 20) - 2.086) < 5e-4, "t_{0.025}(20)")
# (b) 構造モデルからデータを発生させて 2 段階最小二乗法
nsim = 2_000_000
Z = rng.normal(0, np.sqrt(0.5), nsim)
Vv = rng.normal(0, 1, nsim)
Vv = Vv * np.sqrt(0.32)
Ww = (3.6 / 0.32) * Vv + rng.normal(0, np.sqrt(50 - 3.6 ** 2 / 0.32), nsim)
Xs = 1.6 * Z + Vv
Ys = -3.0 * Xs + Ww
C = np.cov(np.vstack([Xs, Ys, Z]))
E.check("s10p08b", abs(C[0, 0] - 1.6) < 0.01 and abs(C[0, 1] + 1.2) < 0.05 and abs(C[2, 0] - 0.8) < 0.01, "発生させたデータの分散・共分散が表と一致")
Xhat = np.polyval(np.polyfit(Z, Xs, 1), Z)
b2sls = np.polyfit(Xhat, Ys, 1)[0]
E.check("s10p08b", abs(b2sls + 3.0) < 0.02, f"2 段階最小二乗法の推定値 {b2sls:.3f}")
E.choice("s10p08b", f"${f(-2.40 / 0.80, 2)}$", "Cov(Z,Y)/Cov(Z,X)")

# ---- 大問9
# (a) 復元 PPS を多数回くり返す人工母集団で不偏性を確かめ、与えたデータで値を計算
xs_pop = rng.integers(10, 60, 400)
ys_pop = xs_pop * rng.uniform(18, 27, 400)
pp = xs_pop / xs_pop.sum()
idx = rng.choice(400, size=(200_000, 4), p=pp)
est = (ys_pop[idx] / pp[idx]).mean(1)
E.check("s10p09a", abs(est.mean() / ys_pop.sum() - 1) < 0.002, "人工母集団でハンセン・ハーウィッツ推定量が不偏")
rooms = np.array([40, 25, 80, 16]); gs = np.array([900, 600, 1700, 400])
E.num("s10p09a", rnd((gs / (rooms / 12800)).mean(), 0), "y/p の平均")
# (b) 隠れた U を含む構造モデルを総当たりで作り、フロントドア式が真の介入効果に一致することを確かめてから値を計算
def frontdoor(pt, pm, py):
    inner = {m: sum(py[(m, t)] * (pt if t else 1 - pt) for t in (0, 1)) for m in (0, 1)}
    return {t: pm[t] * inner[1] + (1 - pm[t]) * inner[0] for t in (0, 1)}
pu = 0.3; pt_u = {0: 0.2, 1: 0.7}; pm_t = {0: 0.25, 1: 0.8}; py_mu = {(0, 0): 0.1, (0, 1): 0.4, (1, 0): 0.5, (1, 1): 0.9}
truth = {t: sum((pu if u else 1 - pu) * (pm_t[t] if m else 1 - pm_t[t]) * py_mu[(m, u)] for u in (0, 1) for m in (0, 1)) for t in (0, 1)}
jt = {}
for u, t, m in product((0, 1), repeat=3):
    jt[(u, t, m)] = (pu if u else 1 - pu) * (pt_u[u] if t else 1 - pt_u[u]) * (pm_t[t] if m else 1 - pm_t[t])
PT = sum(v for (u, t, m), v in jt.items() if t == 1)
PM = {t: sum(v for (u, t2, m), v in jt.items() if t2 == t and m == 1) / sum(v for (u, t2, m), v in jt.items() if t2 == t) for t in (0, 1)}
PY = {(m, t): sum(v * py_mu[(m, u)] for (u, t2, m2), v in jt.items() if t2 == t and m2 == m) / sum(v for (u, t2, m2), v in jt.items() if t2 == t and m2 == m) for m in (0, 1) for t in (0, 1)}
fd = frontdoor(PT, PM, PY)
E.check("s10p09b", abs((fd[1] - fd[0]) - (truth[1] - truth[0])) < 1e-12, "隠れた交絡のある構造モデルでフロントドア式が真の効果に一致")
fdv = frontdoor(Fr(2, 5), {1: Fr(7, 10), 0: Fr(1, 5)}, {(1, 1): Fr(7, 10), (1, 0): Fr(1, 2), (0, 1): Fr(3, 10), (0, 0): Fr(1, 5)})
E.choice("s10p09b", f"${f(float(fdv[1] - fdv[0]), 3)}$", f"分数で計算した効果 {fdv[1] - fdv[0]}")

# ---- 大問10
d = stats.norm.cdf(-0.3) - stats.norm.cdf(-0.6)
E.num("s10p10a", rnd(d, 3), "scipy norm.cdf の差")
E.check("s10p10a", all(abs(stats.norm.cdf(z) - v) < 5e-5 for z, v in [(0.3, 0.6179), (0.6, 0.7257), (0.9, 0.8159)]), "Φ の値")

# ---- 大問11：平均と誤差平方和が問題どおりになるデータを作り、平方和を直接計算
groups = []
for m in (52.0, 45.5, 43.0):
    z = rng.normal(size=8); z = z - z.mean(); z = z / np.sqrt((z ** 2).sum()) * np.sqrt(756 / 3)
    groups.append(m + z)
ally = np.concatenate(groups)
SST = ((ally - ally.mean()) ** 2).sum()
SSE = sum(((g_ - g_.mean()) ** 2).sum() for g_ in groups)
SSA = SST - SSE
E.check("s10p11a", abs(SSE - 756) < 1e-9, "誤差平方和 756")
om = (SSA - 2 * SSE / 21) / (SST + SSE / 21)
E.choice("s10p11a", f"${f(om, 3)}$", f"データから直接 ω²（η²={SSA / SST:.3f}）")
Fv = stats.f_oneway(*groups)
E.check("s10p11a", Fv.pvalue < 0.05 and abs(Fv.statistic - 4.80) < 0.01, f"scipy f_oneway F={Fv.statistic:.3f}")

# ---- 大問12
from scipy.cluster.hierarchy import linkage  # noqa: E402
pts = np.array([[100, 2.0], [130, 5.0], [190, 5.2], [60, 3.5]]) / np.array([50, 1.0])
Lk = linkage(pts, "single")
pair = sorted(int(v) for v in Lk[0, :2])
E.check("s10p12a", pair == [1, 2], f"scipy linkage の最初の併合 {pair}（Q と R）")
E.choice("s10p12a", f"QとR（距離{f(Lk[0, 2], 2)}）", "最初の併合距離")
raw = linkage(pts * np.array([50, 1.0]), "single")
E.check("s10p12a", sorted(int(v) for v in raw[0, :2]) == [0, 1], "尺度をそろえないと P と Q が最初")
L0 = np.array([[0.58, 0.56], [0.44, 0.58], [-0.37, 0.66]])
T = np.array([[0.8, -0.6], [0.6, 0.8]])
Lr = L0 @ T
E.check("s10p12b", np.allclose(T @ T.T, np.eye(2)) and np.allclose((Lr ** 2).sum(1), (L0 ** 2).sum(1)), "直交回転で共通性が不変")
E.choice("s10p12b", f"${f((Lr[:, 0] ** 2).sum(), 2)}$", "行列積で回転後の第1因子の寄与")
# (c) 割合どおりの 100 人の人工データにダミー変数回帰を当てはめる
rows = []
for st, ps in [(0, 60), (1, 40)]:
    for tr, pt in [(0, 50), (1, 30), (2, 20)]:
        rows += [(st, tr)] * (ps * pt // 100)
sc_s = {0: -1.4, 1: 2.1}; sc_t = {0: 0.6, 1: -0.2, 2: -1.2}
Xd = np.array([[1, s_, t_ == 1, t_ == 2] for s_, t_ in rows], dtype=float)
yd = np.array([6.0 + sc_s[s_] + sc_t[t_] for s_, t_ in rows])
bd, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
pred = bd @ np.array([1, 1, 1, 0])
E.check("s10p12c", abs(yd.mean() - 6.0) < 1e-12, "スコアの基準化で全体平均が定数項 6.0 になる")
E.choice("s10p12c", f"予測値{f(pred, 1)}千円、宿泊の有無", "ダミー変数回帰の予測とレンジ")

# ---- 大問13：長い系列を発生させ、1 期先予測 X_t − θ ε_t の誤差分散が最小であることと値を確かめる
th = 0.6
eps = rng.normal(size=200_000)
dx = eps[1:] - th * eps[:-1]
Xser = np.concatenate([[0.0], np.cumsum(dx)])
pred_ok = Xser[:-1] - th * eps[:-1]   # X_t − θ ε_t（ε_t は既知）
err_ok = ((Xser[1:] - pred_ok) ** 2).mean()
err_rw = ((Xser[1:] - Xser[:-1]) ** 2).mean()
E.check("s10p13a", err_ok < err_rw and abs(err_ok - 1) < 0.01, f"予測誤差分散 {err_ok:.3f}（ランダムウォーク予測 {err_rw:.3f}）")
fhat = 120
fhat = fhat + (1 - th) * (130 - fhat)  # 指数平滑の更新
E.choice("s10p13a", f"${f(fhat, 0)}$", "指数平滑法の更新式")

# ---- 大問14
ratio = (stats.gamma.pdf(0.6, 5, scale=1 / 5) / stats.gamma.pdf(1.0, 5, scale=1 / 5)) * (stats.expon.pdf(1.0) / stats.expon.pdf(0.6))
nearest("s10p14a", min(1, ratio))
E.check("s10p14a", abs(np.exp(1.6) - 4.953) < 5e-4 and abs(np.exp(2) - 7.389) < 5e-4, "e^{1.6}, e^2")
# 独立型 MH を実際に回して目標のガンマ分布の平均 1 に近づくこと
cur, acc, chain = 1.0, 0, []
for _ in range(200_000):
    prop = rng.exponential(1.0)
    a = (prop / cur) ** 4 * np.exp(-5 * (prop - cur)) * np.exp(prop - cur)
    if rng.uniform() < a:
        cur = prop
    chain.append(cur)
E.check("s10p14a", abs(np.mean(chain[5000:]) - 1.0) < 0.02, f"MH 連鎖の平均 {np.mean(chain[5000:]):.3f} ≈ 1")

E.done()
