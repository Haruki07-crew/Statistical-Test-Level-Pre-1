"""模擬試験 第21回（set21）の検算。generator/set21.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値最適化）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(21)
rng = np.random.default_rng(20261021)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def ans_idx(iid):
    return E.item(iid)["answer"]


# ---- 大問1：ポリアの壺を列挙（玉に番号を付けて全通りを数える）と sympy の分数で
from fractions import Fraction as Fr
num = den = Fr(0)
for c1, p1 in (("R", Fr(3, 5)), ("W", Fr(2, 5))):
    box = {"R": 3, "W": 2}
    box[c1] += 2
    p2 = Fr(box["R"], sum(box.values()))
    den += p1 * p2
    if c1 == "R":
        num += p1 * p2
E.num("s21p01a", f(float(num / den), 4), f"分数で {num / den}")
u = rng.random((2, 1_000_000))
first_r = u[0] < 0.6
second_r = np.where(first_r, u[1] < 5 / 7, u[1] < 3 / 7)
E.check("s21p01a", abs(first_r[second_r].mean() - 5 / 7) < 0.003, f"シミュレーション {first_r[second_r].mean():.4f}")

# ---- 大問2
n, T = 40, 28
for p in (0.2, 0.5, 0.7, 0.93):
    ev = sum(stats.binom.pmf(t, n, p) * t * (n - t) / (n * (n - 1)) for t in range(n + 1))
    E.check("s21p02a", abs(ev - p * (1 - p)) < 1e-12, f"p={p} で E[T(n−T)/(n(n−1))]=p(1−p)")
    evm = sum(stats.binom.pmf(t, n, p) * (t / n) * (1 - t / n) for t in range(n + 1))
    E.check("s21p02a", abs(evm - p * (1 - p)) > 1e-4, f"p={p} で最尤推定量 p̂(1−p̂) は偏る")
E.choice("s21p02a", f"${f(T * (n - T) / (n * (n - 1)), 4)}$", "推定値")
# (b) 一致性をシミュレーションで：n を大きくしたとき |推定量−p|>0.05 の確率
p = 0.7
res = {}
for nn in (100, 10_000):
    Tn = rng.binomial(nn, p, 20_000)
    X1 = rng.random(20_000) < p
    res[nn] = {"T/n": np.mean(abs(Tn / nn - p) > 0.05), "(T+1)/(n+2)": np.mean(abs((Tn + 1) / (nn + 2) - p) > 0.05),
               "X1": np.mean(abs(X1 - p) > 0.05), "(T+n)/2n": np.mean(abs((Tn + nn) / (2 * nn) - p) > 0.05)}
ok = (res[10_000]["T/n"] < 0.001 and res[10_000]["(T+1)/(n+2)"] < 0.001 and res[10_000]["X1"] > 0.9 and res[10_000]["(T+n)/2n"] > 0.99)
E.check("s21p02b", ok, f"n=10000 で外れる確率 {res[10_000]}")
E.check("s21p02b", "T_n/n" in E.item("s21p02b")["choices"][ans_idx("s21p02b")] and "チェビシェフ" in E.item("s21p02b")["choices"][ans_idx("s21p02b")],
        "正解の選択肢は T_n/n の一致性の記述（他の4つは上のシミュレーションと反例で誤り）")

# ---- 大問3
Mx = np.maximum.reduce([rng.exponential(20, 2_000_000), rng.exponential(30, 2_000_000), rng.exponential(60, 2_000_000)])
emax_int = integrate.quad(lambda t: 1 - stats.expon(scale=20).cdf(t) * stats.expon(scale=30).cdf(t) * stats.expon(scale=60).cdf(t), 0, np.inf)[0]
E.choice("s21p03a", f"${f(emax_int, 0)}$ 分", f"生存関数の数値積分 {emax_int:.4f}")
E.check("s21p03a", abs(Mx.mean() - emax_int) < 0.2, f"シミュレーションの平均 {Mx.mean():.3f}")
dens = lambda x: (4 - x)
Zc = integrate.quad(dens, 0, 4)[0]
med = optimize.brentq(lambda m: integrate.quad(dens, 0, m)[0] / Zc - 0.5, 0, 4)
E.num("s21p03b", f(med, 2), "数値積分した分布関数の 0.5 点")
U = rng.random(1_000_000)
xs = 4 - 4 * np.sqrt(1 - U)  # 逆関数法：F(x)=1-(1-x/4)^2
E.check("s21p03b", abs(np.median(xs) - med) < 0.01, f"逆関数法で発生させた標本の中央値 {np.median(xs):.4f}")

# ---- 大問4
from scipy.special import softmax
ll = np.array([-210.4, -206.9, -206.3]); kk = np.array([1, 2, 3])
aic = -2 * ll + 2 * kk
wts = softmax(-aic / 2)
E.num("s21p04a", f(wts.max(), 3), f"softmax(−AIC/2) の最大 {wts}")
E.check("s21p04a", int(np.argmin(aic)) == 1 and abs(np.exp(-2.5) - 0.0821) < 5e-5 and abs(np.exp(-0.4) - 0.6703) < 5e-5, "AIC 最小は M2、e の値")
zc = (35 - 0.5 - stats.binom.mean(250, 0.1)) / stats.binom.std(250, 0.1)
E.num("s21p04b", f(zc, 2), f"scipy の二項分布の平均・標準偏差で {zc:.4f}")
E.check("s21p04b", stats.binom.sf(34, 250, 0.1) < 0.05 and zc > 1.645, f"正確な片側 p 値 {stats.binom.sf(34, 250, 0.1):.4f} も 5% 未満で結論が一致")
# (c) 系統抽出：開始番号ごとの標本を列挙して包含確率と2次の包含確率
N, k = 12000, 40
samples = [set(range(r, N, k)) for r in range(k)]
incl = np.zeros(N)
for sm in samples:
    for i in sm:
        incl[i] += 1 / k
pair_same = sum(1 for sm in samples if 0 in sm and 40 in sm) / k
pair_diff = sum(1 for sm in samples if 0 in sm and 41 in sm) / k
E.check("s21p04c", np.allclose(incl, 1 / 40) and all(len(sm) == 300 for sm in samples) and len(samples) == 40, "包含確率はすべて 1/40、標本は 40 通り")
E.check("s21p04c", pair_same == 1 / 40 and pair_diff == 0, "2 冊が同時に選ばれる確率は 1/40 か 0")
E.check("s21p04c", "1/«k»" not in E.item("s21p04c")["choices"][ans_idx("s21p04c")] and "$1/40$" in E.item("s21p04c")["choices"][ans_idx("s21p04c")], "正解の選択肢の内容")

# ---- 大問5
r5 = stats.ttest_ind_from_stats(68.5, 8.0, 12, 62.0, 4.0, 10, equal_var=False)
se5 = 6.5 / r5.statistic
v1_, v2_ = 64 / 12, 16 / 10
nu = (v1_ + v2_) ** 2 / (v1_ ** 2 / 11 + v2_ ** 2 / 9)
tq = stats.t.ppf(0.975, int(nu))
E.choice("s21p05a", f"({f(6.5 - tq * se5, 2)},\\ {f(6.5 + tq * se5, 2)})", f"scipy のウェルチ検定の標準誤差と自由度 {nu:.3f}→{int(nu)}")
E.check("s21p05a", all(abs(stats.t.ppf(0.975, d) - v) < 0.0005 for d, v in [(9, 2.262), (16, 2.120), (20, 2.086)]), "t 点")
a = rng.normal(0, 8, (100_000, 12)); b = rng.normal(0, 4, (100_000, 10))
va, vb = a.var(1, ddof=1) / 12, b.var(1, ddof=1) / 10
nus = np.floor((va + vb) ** 2 / (va ** 2 / 11 + vb ** 2 / 9))
cov = np.abs(a.mean(1) - b.mean(1)) <= stats.t.ppf(0.975, nus) * np.sqrt(va + vb)
E.check("s21p05a", abs(cov.mean() - 0.95) < 0.005, f"ウェルチ区間のシミュレーション被覆率 {cov.mean():.4f}")
tab = np.array([[18, 12], [16, 24]])
rr = (tab[0, 0] / tab[0].sum()) / (tab[1, 0] / tab[1].sum())
from scipy.stats.contingency import odds_ratio, relative_risk
orr = odds_ratio(tab, kind="sample").statistic
rrs = relative_risk(18, 30, 16, 40).relative_risk
E.choice("s21p05b", f"リスク比{f(rrs, 2)}、オッズ比{f(orr, 2)}、NNT{f(1 / (18 / 30 - 16 / 40), 1)}", "scipy の relative_risk・odds_ratio")

# ---- 大問6：データを作って実際に交互作用モデルを当てはめる
xg = rng.uniform(40, 95, 400)
Dg = (rng.random(400) < 0.4).astype(float)
yg = 1.20 + 0.020 * xg + 0.90 * Dg - 0.010 * Dg * xg  # 誤差なしで係数が復元される
Xm = np.column_stack([np.ones(400), xg, Dg, Dg * xg])
bh = np.linalg.lstsq(Xm, yg, rcond=None)[0]
pred = lambda d, x: bh @ [1, x, d, d * x]
dif = pred(1, 70) - pred(0, 70)
cross = optimize.brentq(lambda x: pred(1, x) - pred(0, x), 0, 200)
E.choice("s21p06a", f"差{f(dif, 2)}、等しくなる得点{f(cross, 0)}点", "当てはめた式で 70 点の差と交点（数値解）")
# (b) 不均一分散のシミュレーション：OLS は不偏だが通常の SE は過小
xs6 = np.linspace(1, 10, 50)
Xs = np.column_stack([np.ones(50), xs6])
bs, ses = [], []
for _ in range(20_000):
    e = rng.normal(0, 0.3 * xs6)
    y = 1 + 0.5 * xs6 + e
    b, rs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    bs.append(b[1])
    s2 = rs[0] / 48
    ses.append(np.sqrt(s2 * np.linalg.inv(Xs.T @ Xs)[1, 1]))
bs = np.array(bs)
E.check("s21p06b", abs(bs.mean() - 0.5) < 0.003 and np.mean(ses) < bs.std() * 0.95,
        f"不均一分散でも傾きは不偏（平均 {bs.mean():.4f}）、通常の SE の平均 {np.mean(ses):.4f} < 実際の標準偏差 {bs.std():.4f}")
E.check("s21p06b", "頑健な標準誤差" in E.item("s21p06b")["choices"][ans_idx("s21p06b")] and "不偏のまま" in E.item("s21p06b")["choices"][ans_idx("s21p06b")], "正解の選択肢の内容")

# ---- 大問7
pz = stats.norm.sf(64.5, 60, 15 / 6)
E.choice("s21p07a", f"${f(pz, 3)}$", "scipy norm.sf(64.5, 60, 2.5)")
for x, v in [(0.3, 0.6179), (0.72, 0.7642), (1.8, 0.9641)]:
    E.check("s21p07a", abs(stats.norm.cdf(x) - v) < 5e-5, f"Φ({x})={v}")
for rr in (0.6, -0.6):
    w, V = np.linalg.eigh(np.array([[1, rr], [rr, 1]]))
    E.check("s21p07b", np.allclose(sorted(w), [0.4, 1.6]) and abs(max(w) / 2 - 0.8) < 1e-12, f"r={rr} の固有値 {w}、寄与率 80%")
w, V = np.linalg.eigh(np.array([[1, 0.6], [0.6, 1]]))
E.check("s21p07b", np.allclose(np.abs(V[:, 0]), [1 / np.sqrt(2)] * 2) and V[0, 0] * V[1, 0] < 0, "第2主成分の固有ベクトルは差の方向 (1,−1)/√2")
E.choice("s21p07b", "第1主成分の寄与率は$80\\%$である。第2主成分", "正解の文")

# ---- 大問8
pmf = {0: 0.4, 1: 0.3, 2: 0.2, 4: 0.1}
xs8 = np.repeat(list(pmf), [int(v * 10_000) for v in pmf.values()])
E.choice("s21p08a", f"${f(stats.kurtosis(xs8, fisher=True), 2)}$", "比率どおりの 10000 個のデータで scipy の尖度（Fisher）")
g = stats.geom(0.3)
EN = sum(min(k, 5) * g.pmf(k) for k in range(1, 400))
sim = np.minimum(rng.geometric(0.3, 2_000_000), 5)
E.choice("s21p08b", f"期待値{f(EN, 2)}回、確率{f(g.sf(5), 3)}", f"scipy の幾何分布で E[min(X,5)]、シミュレーション {sim.mean():.4f}")
E.check("s21p08b", abs(sim.mean() - EN) < 0.005, "シミュレーションと一致")

# ---- 大問9
P = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.4, 0.6]])
st = 0
paths = 0.0
for i1 in range(3):
    for i2 in range(3):
        paths += P[0, i1] * P[i1, i2] * P[i2, 2]
E.num("s21p09a", f(paths, 3), f"経路の総当たりで {paths:.5f}")
cur = np.zeros(1_000_000, dtype=int)
cum = P.cumsum(1)
for _ in range(3):
    u = rng.random(cur.size)
    cur = (u[:, None] > cum[cur]).sum(1)
E.check("s21p09a", abs((cur == 2).mean() - paths) < 0.002, f"シミュレーション {(cur == 2).mean():.4f}")

# ---- 大問10
nmin = next(n for n in range(1, 30) if stats.binom.sf(n - 1, n, 0.6) <= 0.05)
b_ = stats.binom.cdf(nmin - 1, nmin, 0.9)
E.choice("s21p10a", f"$n={nmin}$、第2種{f(b_, 3)}", "scipy の二項分布で最小の n と第 2 種の誤り")
E.check("s21p10a", all(abs(x ** n - v) < 5e-5 for x, n, v in [(0.6, 5, 0.0778), (0.6, 6, 0.0467), (0.6, 7, 0.0280), (0.6, 8, 0.0168), (0.9, 5, 0.5905), (0.9, 6, 0.5314), (0.9, 8, 0.4305)]), "示したべき乗の値")

# ---- 大問11：データを作って statsmodels を使わず、平方和を直接再現して F 値を確認
SA, SB, SAB, SE = 180, 420, 60, 540
FA = (SA / 2) / ((SE + SAB) / (24 + 6))
E.choice("s21p11a", f"${f(FA, 2)}$", "平方和と自由度から")
# 平方和がちょうどこの値になるデータを作り、交互作用なしのモデルの残差平方和から確認
rng2 = np.random.default_rng(1)
A_ = np.repeat(np.arange(3), 12); B_ = np.tile(np.repeat(np.arange(4), 3), 3)
X_full = np.column_stack([np.ones(36)] + [(A_ == i) * 1.0 for i in (1, 2)] + [(B_ == j) * 1.0 for j in (1, 2, 3)])
yv = rng2.normal(0, 1, 36)
rss_add = np.sum((yv - X_full @ np.linalg.lstsq(X_full, yv, rcond=None)[0]) ** 2)
X_noA = X_full[:, [0, 3, 4, 5]]
rss_noA = np.sum((yv - X_noA @ np.linalg.lstsq(X_noA, yv, rcond=None)[0]) ** 2)
F_pool = ((rss_noA - rss_add) / 2) / (rss_add / 30)
cells = yv.reshape(3, 4, 3)
SE_ = ((cells - cells.mean(2, keepdims=True)) ** 2).sum(); gm = yv.mean()
SA_ = 12 * ((cells.mean((1, 2)) - gm) ** 2).sum(); SB_ = 9 * ((cells.mean((0, 2)) - gm) ** 2).sum()
SAB_ = ((cells - gm) ** 2).sum() - SE_ - SA_ - SB_
E.check("s21p11a", abs(F_pool - (SA_ / 2) / ((SE_ + SAB_) / 30)) < 1e-9, "釣り合い型では交互作用なしモデルの F 検定がプールした F と一致（df 2, 30）")
lam = rng.gamma(15, 1 / 6, 2_000_000)
pred0 = np.exp(-lam).mean()
nb = stats.nbinom.pmf(0, 15, 6 / 7)
E.choice("s21p11b", f"${f(nb, 3)}$", f"scipy の負の二項分布 nbinom(15, 6/7) の P(0)、事後からのシミュレーション {pred0:.4f}")
E.check("s21p11b", abs(pred0 - nb) < 0.001 and abs((6 / 7) ** 15 - 0.0990) < 5e-5 and abs((6 / 7) ** 12 - 0.1573) < 5e-5 and abs(np.exp(-2.4) - 0.0907) < 5e-5 and abs(np.exp(-2.5) - 0.0821) < 5e-5, "シミュレーションと示した値")

# ---- 大問12
eta = 0.4
pl = stats.logistic.cdf(eta)
me = (stats.logistic.cdf(eta + 1e-6) - stats.logistic.cdf(eta - 1e-6)) / 2e-6 * 0.30
E.choice("s21p12a", f"$\\hat p={f(pl, 3)}$、限界効果{f(me, 3)}", "scipy のロジスティック分布の分布関数と数値微分")
E.check("s21p12a", abs(np.exp(0.4) - 1.4918) < 5e-5, "e^0.4")
obs = rng.normal(0, 1, 40)
obs = (obs - obs.mean()) / obs.std(ddof=1) * np.sqrt(1.2) + 2.0
full = np.r_[obs, np.full(10, obs.mean())]
E.num("s21p12b", f(full.var(ddof=1), 3), "実際に平均で埋めたデータの不偏分散")
lz = 0.5 + rng.normal(0, 0.6, 2_000_000)
mlog = stats.lognorm(s=0.6, scale=np.exp(0.5)).mean()
E.choice("s21p12c", f"${f(mlog, 2)}$", f"scipy の対数正規分布の平均 {mlog:.4f}、シミュレーション {np.exp(lz).mean():.4f}")
E.check("s21p12c", all(abs(np.exp(x) - v) < 5e-5 for x, v in [(0.32, 1.3771), (0.5, 1.6487), (0.68, 1.9739), (0.86, 2.3632)]), "示した指数の値")

# ---- 大問13
Dm = np.array([[0, 1.0, 3.5, 6.0, 5.5], [1.0, 0, 2.0, 5.0, 4.8], [3.5, 2.0, 0, 4.0, 4.2], [6.0, 5.0, 4.0, 0, 2.5], [5.5, 4.8, 4.2, 2.5, 0]])
def silhouette(D, lab, i):
    lab = np.asarray(lab)
    own = [j for j in range(len(lab)) if lab[j] == lab[i] and j != i]
    a_ = D[i, own].mean()
    b_ = min(D[i, lab == g].mean() for g in set(lab) if g != lab[i])
    return (b_ - a_) / max(a_, b_)
silC = silhouette(Dm, [0, 0, 0, 1, 1], 2)
E.choice("s21p13a", f"${f(silC, 2)}$", f"一般のシルエット関数で {silC:.4f}")
sil_all = [silhouette(Dm, [0, 0, 0, 1, 1], i) for i in range(5)]
E.check("s21p13a", all(-1 <= v <= 1 for v in sil_all), f"全員のシルエット {np.round(sil_all, 3)}")
def H(*cnt):
    pr = np.array(cnt) / sum(cnt)
    return stats.entropy(pr, base=2)
ig = H(8, 12) - (6 / 20 * H(5, 1) + 14 / 20 * H(3, 11))
E.num("s21p13b", f(ig, 2), f"scipy の entropy（底 2）で {ig:.4f}")
E.check("s21p13b", abs(H(4, 6) - 0.971) < 5e-4 and abs(H(1, 5) - 0.650) < 5e-4 and abs(H(3, 11) - 0.750) < 5e-4, "示した h の値")

E.done()
