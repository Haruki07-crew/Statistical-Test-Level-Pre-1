"""第28章 検算スクリプト（分割表）。
2×2表の指標（RR・RD・OR）と対数の標準誤差・信頼区間、条件付き最尤推定と正確な信頼区間
（scipy.stats.contingency.odds_ratio）、マクネマー検定、イェーツの補正、対数線形モデルと階層モデルの
G²・自由度（IPF を numpy で独立に実装）、マンテル・ヘンツェル推定量・検定、シンプソンのパラドックス、
標本抽出モデルの条件付けの連鎖を sympy/numpy/scipy で独立に再計算し、本文と照合する。
charts の params は各 md の front matter を yaml で読み、本文の表・式から再計算して照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4、第2位なら 5e-3）。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import itertools
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats
from scipy.stats.contingency import odds_ratio as sc_odds_ratio

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch28" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid):
    return [c for c in front(term).get("charts", []) if c.get("id") == cid]


def chart_type(term, ctype):
    return [c for c in front(term).get("charts", []) if c.get("type") == ctype]


def pearson(t):
    t = np.asarray(t, float)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return float(((t - e) ** 2 / e).sum()), e


def g2(obs, fit):
    obs = np.asarray(obs, float)
    fit = np.asarray(fit, float)
    m = obs > 0
    return float(2 * (obs[m] * np.log(obs[m] / fit[m])).sum())


z975 = stats.norm.ppf(0.975)

# ================================================================ two-by-two-table
T = "two-by-two-table"
a, b, c, d = 30, 70, 15, 85
n1, n2, m1, m2, n = a + b, c + d, a + c, b + d, a + b + c + d
p1, p2 = a / n1, c / n2
check(T, "例1 RR=2.000, RD=0.150", close(p1 / p2, 2.0, 5e-4) and close(p1 - p2, 0.15, 5e-4), (p1 / p2, p1 - p2))
check(T, "例1 OR=2550/1050=2.429", a * d == 2550 and b * c == 1050 and close(a * d / (b * c), 2.429, 5e-4), a * d / (b * c))
den = n1 * n2 * m1 * m2
chi = n * (a * d - b * c) ** 2 / den
check(T, "例1 分母 69,750,000・χ²=6.452", den == 69_750_000 and a * d - b * c == 1500 and close(chi, 6.452, 5e-4), chi)
pv = stats.chi2.sf(chi, 1)
check(T, "例1 p=0.0111", close(pv, 0.0111), pv)
chi_sc = stats.chi2_contingency([[a, b], [c, d]], correction=False)[0]
check(T, "簡便式=期待度数を使う χ²（scipy）", close(chi_sc, chi, 1e-10), chi_sc)
# 記号: a-E11=(ad-bc)/n, Σ1/E=n^3/(n1n2m1m2)
A, B, C, D = sp.symbols("a b c d", positive=True)
N = A + B + C + D
N1, N2, M1, M2 = A + B, C + D, A + C, B + D
E = [[N1 * M1 / N, N1 * M2 / N], [N2 * M1 / N, N2 * M2 / N]]
obs = [[A, B], [C, D]]
devs_ok = all(sp.simplify((obs[i][j] - E[i][j]) - (1 if i == j else -1) * (A * D - B * C) / N) == 0 for i in range(2) for j in range(2))
check(T, "sympy: 4セルのずれが ±(ad−bc)/n", devs_ok, devs_ok)
s_inv = sp.simplify(sum(1 / E[i][j] for i in range(2) for j in range(2)) - N ** 3 / (N1 * N2 * M1 * M2))
check(T, "sympy: Σ1/Ê=n³/(n1n2m1m2)", s_inv == 0, s_inv)
chi_sym = sum((obs[i][j] - E[i][j]) ** 2 / E[i][j] for i in range(2) for j in range(2))
d0 = sp.simplify(chi_sym - N * (A * D - B * C) ** 2 / (N1 * N2 * M1 * M2))
check(T, "sympy: ピアソンχ²=n(ad−bc)²/(n1n2m1m2)", d0 == 0, d0)
P1, P2 = sp.symbols("p1 p2", positive=True)
d1 = sp.simplify(P1 / (1 - P1) / (P2 / (1 - P2)) - P1 / P2 * (1 - P2) / (1 - P1))
check(T, "sympy: OR=RR×(1−p2)/(1−p1)", d1 == 0, d1)
check(T, "例2 2×0.85/0.70=2.429", close(2 * 0.85 / 0.70, 2.429, 5e-4), 2 * 0.85 / 0.70)
ph = (a + c) / n
z = (p1 - p2) / math.sqrt(ph * (1 - ph) * (1 / n1 + 1 / n2))
check(T, "例2 p̂=0.225, z=2.540, z²=6.452", close(ph, 0.225, 5e-4) and close(z, 2.540, 5e-4) and close(z * z, 6.452, 5e-4) and close(z * z, chi, 1e-10), (ph, z, z * z))
chiY = n * (abs(a * d - b * c) - n / 2) ** 2 / den
check(T, "例2 イェーツ χ²_Y=5.620, p=0.0178", close(chiY, 5.620, 5e-4) and close(stats.chi2.sf(chiY, 1), 0.0178), (chiY, stats.chi2.sf(chiY, 1)))
fp = stats.fisher_exact([[a, b], [c, d]])[1]
check(T, "例2 フィッシャー両側 p=0.0171", close(fp, 0.0171), fp)
check(T, "OR は転置で不変", (a * d) / (b * c) == (a * d) / (c * b), "ad/bc")
mo = charts(T, "mosaic")[0]["params"]
check(T, "chart mosaic の表=例1", mo["table"] == [[30, 70], [15, 85]], mo["table"])

# ================================================================ relative-risk
T = "relative-risk"
a, n1, c, n2 = 24, 200, 12, 200
rr = (a / n1) / (c / n2)
check(T, "例1 p̂=0.12/0.06, RR=2.000", close(a / n1, 0.12) and close(c / n2, 0.06) and close(rr, 2.0, 5e-4), rr)
t1, t2 = 1 / a - 1 / n1, 1 / c - 1 / n2
se = math.sqrt(t1 + t2)
check(T, "例1 0.03667+0.07833=0.11500, SE=0.3391", close(t1, 0.03667, 5e-6) and close(t2, 0.07833, 5e-6) and close(t1 + t2, 0.115, 5e-6) and close(se, 0.3391), (t1, t2, se))
lo, hi = math.log(rr) - z975 * se, math.log(rr) + z975 * se
check(T, "例1 log2=0.6931, 対数区間 (0.0285, 1.3578)", close(math.log(2), 0.6931) and close(lo, 0.0285) and close(hi, 1.3578), (lo, hi))
check(T, "例1 RR区間 (1.029, 3.888)", close(math.exp(lo), 1.029, 5e-4) and close(math.exp(hi), 3.888, 5e-4), (math.exp(lo), math.exp(hi)))
zz = math.log(rr) / se
check(T, "例1 z=2.044, p=0.041", close(zz, 2.044, 5e-4) and close(2 * stats.norm.sf(zz), 0.041, 5e-4), (zz, 2 * stats.norm.sf(zz)))
check(T, "区間は上側に長い", math.exp(hi) - 2 > 2 - math.exp(lo), (2 - math.exp(lo), math.exp(hi) - 2))
check(T, "log2=−log0.5=0.693", close(math.log(2), 0.693, 5e-4) and close(-math.log(0.5), 0.693, 5e-4), math.log(2))
# デルタ法: 1/a−1/n1 = b/(a n1)、V[log p̂]=(1-p)/(np)
An, Nn, Pp = sp.symbols("a n p", positive=True)
d2 = sp.simplify((1 - An / Nn) / An - (1 / An - 1 / Nn))
check(T, "sympy: (1−a/n1)/a=1/a−1/n1", d2 == 0, d2)
d3 = sp.simplify((1 / An - 1 / Nn) - (Nn - An) / (An * Nn))
check(T, "sympy: 1/a−1/n1=b/(a n1)", d3 == 0, d3)
d4 = sp.simplify((1 / Pp) ** 2 * Pp * (1 - Pp) / Nn - (1 - Pp) / (Nn * Pp))
check(T, "sympy: デルタ法 V[log p̂]=(1−p)/(np)", d4 == 0, d4)
# 乱数: 積二項で log RR の分散
sims = 200000
x1 = rng.binomial(10 * n1, 0.12, sims)
x2 = rng.binomial(10 * n2, 0.06, sims)
mask = (x1 > 0) & (x2 > 0)
v_emp = 10 * np.var(np.log(x1[mask] / n1) - np.log(x2[mask] / n2))
v_th = (1 - 0.12) / (n1 * 0.12) + (1 - 0.06) / (n2 * 0.06)
check(T, "乱数20万回（各群 n=2000、漸近近似の確認）: 10·V[log RR̂]≈0.1150（相対許容 3%）", abs(v_emp / v_th - 1) < 0.03, (v_emp, v_th))
# 例2
check(T, "例2 コホート RR=2.000", close(0.30 / 0.15, 2.0, 5e-4), 2.0)
f = (30 / 170) / (15 / 185)
check(T, "例2 症例対照の形式的 RR=2.176", close(f, 2.176, 5e-4), f)
check(T, "例2 非発症者2倍 70→140, 85→170", 70 * 2 == 140 and 85 * 2 == 170, (140, 170))
check(T, "例2 OR 30·170/(140·15)=2.429 = コホートの OR", close(30 * 170 / (140 * 15), 2.429, 5e-4) and abs(30 * 170 / (140 * 15) - 30 * 85 / (70 * 15)) < 1e-12, 30 * 170 / (140 * 15))
ld = charts(T, "line-data")[0]["params"]
xs = ld["x"]
rr_s = [s for s in ld["series"] if s["label"].startswith("相対")][0]["y"]
or_s = [s for s in ld["series"] if s["label"].startswith("オッズ")][0]["y"]
or_th = [(2 * p / (1 - 2 * p)) / (p / (1 - p)) for p in xs]
check(T, "chart line-data: RR=2 固定, OR=2(1−p2)/(1−2p2)", all(v == 2 for v in rr_s) and allclose(or_th, or_s), np.round(or_th, 4).tolist())
check(T, "試験のポイント 0.002/0.001 と 0.4/0.2 はどちらも RR=2", close(0.002 / 0.001, 2, 1e-9) and close(0.4 / 0.2, 2, 1e-9), "2")

# ================================================================ risk-difference
T = "risk-difference"
p1, p2 = 0.12, 0.06
v1, v2 = p1 * (1 - p1) / 200, p2 * (1 - p2) / 200
se = math.sqrt(v1 + v2)
check(T, "例1 RD=0.060", close(p1 - p2, 0.06, 5e-4), p1 - p2)
check(T, "例1 0.000528+0.000282=0.000810, SE=0.02846", close(v1, 0.000528, 5e-7) and close(v2, 0.000282, 5e-7) and close(v1 + v2, 0.00081, 5e-7) and close(se, 0.02846, 5e-6), (v1, v2, se))
lo, hi = 0.06 - z975 * se, 0.06 + z975 * se
check(T, "例1 95%区間 (0.0042, 0.1158)", close(lo, 0.0042) and close(hi, 0.1158), (lo, hi))
check(T, "例1 1/0.06=16.7", close(1 / 0.06, 16.7, 5e-2), 1 / 0.06)
check(T, "例2 まれ: RR=2, RD=0.001, NNT=1000", close(0.002 / 0.001, 2, 1e-9) and close(0.001, 0.002 - 0.001, 1e-12) and close(1 / (0.002 - 0.001), 1000, 1e-6), 1000)
check(T, "例2 よくある: RR=2, RD=0.2, NNT=5, 影響200倍", close(0.4 / 0.2, 2, 1e-9) and close(0.4 - 0.2, 0.2, 1e-12) and close(1 / 0.2, 5, 1e-9) and close(0.2 / 0.001, 200, 1e-6), 200)
d5 = sp.simplify(P1 - P2 - P2 * (P1 / P2 - 1))
check(T, "sympy: RD=p2(RR−1)", d5 == 0, d5)
x1 = rng.binomial(200, p1, sims) / 200
x2 = rng.binomial(200, p2, sims) / 200
check(T, "乱数20万回: V[p̂1−p̂2]≈0.000810（相対許容 2%）", abs(np.var(x1 - x2) / 0.00081 - 1) < 0.02, np.var(x1 - x2))
bd = charts(T, "bar-data")[0]["params"]
check(T, "chart bar-data=(0.12, 0.06)", bd["values"] == [0.12, 0.06], bd["values"])

# ================================================================ odds-ratio-interval
T = "odds-ratio-interval"
a, b, c, d = 40, 60, 20, 80
orh = a * d / (b * c)
check(T, "例1 OR=3200/1200=2.667, log=0.9808", a * d == 3200 and b * c == 1200 and close(orh, 2.667, 5e-4) and close(math.log(orh), 0.9808), (orh, math.log(orh)))
terms = [1 / a, 1 / b, 1 / c, 1 / d]
V = sum(terms)
check(T, "例1 0.02500+0.01667+0.05000+0.01250=0.10417", allclose(terms, [0.025, 0.01667, 0.05, 0.0125], 5e-6) and close(V, 0.10417, 5e-6), V)
se = math.sqrt(V)
check(T, "例1 SE=0.3227", close(se, 0.3227), se)
lo, hi = math.log(orh) - z975 * se, math.log(orh) + z975 * se
check(T, "例1 対数区間 (0.3483, 1.6134)", close(lo, 0.3483) and close(hi, 1.6134), (lo, hi))
check(T, "例1 OR区間 (1.417, 5.020)", close(math.exp(lo), 1.417, 5e-4) and close(math.exp(hi), 5.020, 5e-4), (math.exp(lo), math.exp(hi)))
zz = math.log(orh) / se
check(T, "例1 z=3.039, p=0.0024", close(zz, 3.039, 5e-4) and close(2 * stats.norm.sf(zz), 0.0024), (zz, 2 * stats.norm.sf(zz)))
check(T, "例1 1/20 が分散のほぼ半分", 0.45 < (1 / c) / V < 0.5, (1 / c) / V)
res = sc_odds_ratio([[a, b], [c, d]], kind="conditional")
ci = res.confidence_interval(0.95)
check(T, "例1 条件付き最尤 2.653（scipy）", close(res.statistic, 2.653, 5e-4), res.statistic)
check(T, "例1 正確な95%区間 (1.358, 5.313)（scipy）", close(ci.low, 1.358, 5e-4) and close(ci.high, 5.313, 5e-4), (ci.low, ci.high))
# 条件付き最尤を超幾何の非心分布で独立に確認
from scipy.optimize import brentq
m1_, n1_, n2_ = a + c, a + b, c + d
xs_ = np.arange(max(0, m1_ - n2_), min(n1_, m1_) + 1)
logc = np.array([math.lgamma(n1_ + 1) - math.lgamma(x + 1) - math.lgamma(n1_ - x + 1) + math.lgamma(n2_ + 1) - math.lgamma(m1_ - x + 1) - math.lgamma(n2_ - m1_ + x + 1) for x in xs_])


def ncmean(psi):
    w = logc + xs_ * math.log(psi)
    w = np.exp(w - w.max())
    return float((xs_ * w).sum() / w.sum())


cmle = brentq(lambda s: ncmean(s) - a, 0.5, 20)
check(T, "例1 条件付き最尤を非心超幾何の E[a]=a で独立に解く 2.653", close(cmle, 2.653, 5e-4), cmle)
# デルタ法 sympy
p = sp.symbols("p11 p12 p21 p22", positive=True)
nn = sp.symbols("n", positive=True)
g = sp.Matrix([1 / p[0], -1 / p[1], -1 / p[2], 1 / p[3]])
Sig = sp.Matrix(4, 4, lambda i, j: ((p[i] if i == j else 0) - p[i] * p[j]) / nn)
vv = sp.simplify((g.T * Sig * g)[0].subs(p[3], 1 - p[0] - p[1] - p[2]))
target = sum(1 / (nn * pi) for pi in p).subs(p[3], 1 - p[0] - p[1] - p[2])
check(T, "sympy: デルタ法 gᵀΣg=Σ1/(np_ij)（多項）", sp.simplify(vv - target) == 0, vv)
# 乱数: 多項で log OR の分散
pr = np.array([a, b, c, d]) / 200
M = rng.multinomial(2000, pr, sims)
mk = (M > 0).all(1)
lor = np.log(M[mk, 0] * M[mk, 3] / (M[mk, 1] * M[mk, 2]))
check(T, "乱数20万回（多項 n=2000、漸近近似の確認）: 10·V[log OR̂]≈0.10417（相対許容 3%）", abs(10 * np.var(lor) / V - 1) < 0.03, 10 * np.var(lor))
# 例2
a2, b2, c2, d2_ = 5.5, 15.5, 0.5, 20.5
or2 = a2 * d2_ / (b2 * c2)
se2 = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2_)
check(T, "例2 補正 OR=14.55, SE=1.515, log=2.6775", close(or2, 14.55, 5e-3) and close(se2, 1.515, 5e-4) and close(math.log(or2), 2.6775), (or2, se2, math.log(or2)))
lo2, hi2 = math.exp(math.log(or2) - z975 * se2), math.exp(math.log(or2) + z975 * se2)
check(T, "例2 95%区間 (0.747, 283.4)", close(lo2, 0.747, 5e-4) and close(hi2, 283.4, 5e-2), (lo2, hi2))
check(T, "例2 1/0.5=2 が分散のほとんど", 2 / se2 ** 2 > 0.85, 2 / se2 ** 2)
fp2 = stats.fisher_exact([[5, 15], [0, 20]])[1]
check(T, "例2 フィッシャー両側 p=0.047", close(fp2, 0.047, 5e-4), fp2)
pdfc = chart_type(T, "pdf")[0]["params"]
check(T, "chart pdf N(0.9808, 0.3227)", close(pdfc["mu"], math.log(orh)) and close(pdfc["sigma"], se), pdfc)

# ================================================================ contingency-table
T = "contingency-table"
tab = np.array([[10, 30, 40], [25, 35, 20], [40, 25, 15]])
check(T, "例1 行計80, 列計 75/90/75, 総数240", tab.sum(1).tolist() == [80, 80, 80] and tab.sum(0).tolist() == [75, 90, 75] and tab.sum() == 240, tab.sum(0))
check(T, "例1 現金の行割合 0.125/0.3125/0.5", allclose(tab[:, 0] / 80, [0.125, 0.3125, 0.5], 1e-12), tab[:, 0] / 80)
x2, Ee = pearson(tab)
check(T, "例1 期待度数 25,30,25", allclose(Ee, [[25, 30, 25]] * 3, 1e-12), Ee[0])
check(T, "例1 χ²=33.667", close(x2, 33.667, 5e-4), x2)
pv = stats.chi2.sf(x2, 4)
check(T, "例1 df=4, p≈8.7e-7", abs(pv / 8.7e-7 - 1) < 0.006, pv)
Vc = math.sqrt(x2 / (240 * 2))
check(T, "例2 クラメール V=0.265", close(Vc, 0.265, 5e-4), Vc)
pr_ = (tab - Ee) / np.sqrt(Ee)
check(T, "例2 ピアソン残差 20代現金 −3.0, 20代スマホ +3.0, 60代現金 +3.0", allclose([pr_[0, 0], pr_[0, 2], pr_[2, 0]], [-3, 3, 3], 5e-2), pr_.round(3).tolist())
check(T, "例2 この3セルが最大の |残差|", sorted(np.abs(pr_).ravel())[-3] >= 3 - 1e-9 and sorted(np.abs(pr_).ravel())[-4] < 3, np.abs(pr_).round(3).tolist())
Ii, Jj = sp.symbols("I J", positive=True, integer=True)
check(T, "sympy: (IJ−1)−{(I−1)+(J−1)}=(I−1)(J−1)", sp.expand((Ii * Jj - 1) - ((Ii - 1) + (Jj - 1)) - (Ii - 1) * (Jj - 1)) == 0, "0")
check(T, "n を2倍にすると χ² も2倍", close(pearson(2 * tab)[0], 2 * x2, 1e-9), pearson(2 * tab)[0])
mo = charts(T, "mosaic")[0]["params"]
check(T, "chart mosaic の表=例1", mo["table"] == tab.tolist(), mo["table"])

# ================================================================ mcnemar-test
T = "mcnemar-test"
a, b, c, d = 40, 8, 20, 32
check(T, "例1 前0.48, 後0.60, 不一致28", close((a + b) / 100, 0.48, 1e-12) and close((a + c) / 100, 0.60, 1e-12) and b + c == 28, (0.48, 0.6))
cm = (b - c) ** 2 / (b + c)
check(T, "例1 χ²_M=144/28=5.143, p=0.0233", close(cm, 5.143, 5e-4) and close(stats.chi2.sf(cm, 1), 0.0233), (cm, stats.chi2.sf(cm, 1)))
cmc = (abs(b - c) - 1) ** 2 / (b + c)
check(T, "例1 連続修正 4.321, p=0.0376", close(cmc, 4.321, 5e-4) and close(stats.chi2.sf(cmc, 1), 0.0376), (cmc, stats.chi2.sf(cmc, 1)))
ex = stats.binomtest(b, b + c, 0.5).pvalue
ex2 = stats.binom.cdf(8, 28, 0.5) + stats.binom.sf(19, 28, 0.5)
check(T, "例1 正確な二項検定 両側 p=0.0357（≤8 または ≥20）", close(ex, 0.0357) and close(ex2, ex, 1e-12), ex)
check(T, "chart caption 正確な p 値 0.036", close(ex, 0.036, 5e-4), ex)
se = math.sqrt((b + c) - (b - c) ** 2 / 100) / 100
check(T, "例1 √26.56/100=0.0515", close((b + c) - (b - c) ** 2 / 100, 26.56, 1e-9) and close(se, 0.0515), se)
lo, hi = 0.12 - z975 * se, 0.12 + z975 * se
check(T, "例1 95%区間 (0.019, 0.221)", close(lo, 0.019, 5e-4) and close(hi, 0.221, 5e-4), (lo, hi))
x2u = stats.chi2_contingency([[48, 52], [60, 40]], correction=False)
check(T, "例2 独立2群とみなした χ²=2.899, p=0.0887", close(x2u[0], 2.899, 5e-4) and close(x2u[1], 0.0887), (x2u[0], x2u[1]))
# sympy: z²=(b−c)²/(b+c)
Bs, Cs = sp.symbols("b c", positive=True)
zs = (Bs - (Bs + Cs) / 2) / sp.sqrt((Bs + Cs) / 4)
check(T, "sympy: z=(b−c)/√(b+c), z²=(b−c)²/(b+c)", sp.simplify(zs ** 2 - (Bs - Cs) ** 2 / (Bs + Cs)) == 0, sp.simplify(zs))
# sympy: V[(n21−n12)/n] の多項分散
P12, P21, Nm = sp.symbols("p12 p21 n", positive=True)
var = (Nm * P21 * (1 - P21) + Nm * P12 * (1 - P12) + 2 * Nm * P12 * P21) / Nm ** 2
check(T, "sympy: V[(c−b)/n]={(p12+p21)−(p12−p21)²}/n", sp.simplify(var - ((P12 + P21) - (P12 - P21) ** 2) / Nm) == 0, "0")
# 乱数: 対角は差に影響しない、分散式
pm = np.array([0.4, 0.08, 0.2, 0.32])
Mm = rng.multinomial(100, pm, sims)
dif = (Mm[:, 2] - Mm[:, 1]) / 100
vth = ((0.08 + 0.2) - (0.08 - 0.2) ** 2) / 100
check(T, "乱数20万回: V[(c−b)/n]≈0.002656（相対許容 2%）", abs(np.var(dif) / vth - 1) < 0.02, (np.var(dif), vth))
# H0 で b|(b+c)~Bin(·,1/2)
pm0 = np.array([0.4, 0.14, 0.14, 0.32])
M0 = rng.multinomial(100, pm0, sims)
sel = (M0[:, 1] + M0[:, 2]) == 28
emp = np.mean(M0[sel, 1] <= 8)
check(T, "乱数: H0 で b|(b+c=28) の P(b≤8)≈Bin(28,0.5) の 0.0178（許容 0.006）", abs(emp - stats.binom.cdf(8, 28, 0.5)) < 0.006, (emp, sel.sum()))
mo = charts(T, "mosaic")[0]["params"]
pm_c = chart_type(T, "pmf")[0]
check(T, "chart mosaic=例1・pmf Bin(28,0.5)", mo["table"] == [[40, 8], [20, 32]] and pm_c["params"] == {"n": 28, "p": 0.5}, (mo["table"], pm_c["params"]))

# ================================================================ yates-correction
T = "yates-correction"
a, b, c, d = 12, 8, 5, 15
n = 40
den = 20 * 20 * 17 * 23
check(T, "例1 ad−bc=140, 分母 156,400", a * d - b * c == 140 and den == 156400, den)
x0 = n * 140 ** 2 / den
xy = n * (140 - 20) ** 2 / den
check(T, "例1 補正なし 5.013, p=0.0252", close(x0, 5.013, 5e-4) and close(stats.chi2.sf(x0, 1), 0.0252), (x0, stats.chi2.sf(x0, 1)))
check(T, "例1 補正あり 3.683, p=0.0550", close(xy, 3.683, 5e-4) and close(stats.chi2.sf(xy, 1), 0.0550), (xy, stats.chi2.sf(xy, 1)))
xy_sc = stats.chi2_contingency([[a, b], [c, d]], correction=True)[0]
check(T, "例1 補正あり = Σ(|n−Ê|−0.5)²/Ê（scipy）", close(xy_sc, xy, 1e-10), xy_sc)
fe = stats.fisher_exact([[a, b], [c, d]])
check(T, "例1 フィッシャー両側 p=0.0536", close(fe[1], 0.0536), fe[1])
_, Ey = pearson([[a, b], [c, d]])
check(T, "例1 期待度数 8.5, 11.5, 8.5, 11.5", allclose(Ey, [[8.5, 11.5], [8.5, 11.5]], 1e-12), Ey.ravel())
f1 = stats.fisher_exact([[a, b], [c, d]], alternative="greater")[1]
n1s = stats.norm.sf(math.sqrt(x0))
n1y = stats.norm.sf(math.sqrt(xy))
check(T, "例1 片側 正確 0.0268, 補正なし 0.0126, 補正あり 0.0275", close(f1, 0.0268) and close(n1s, 0.0126) and close(n1y, 0.0275), (f1, n1s, n1y))
check(T, "|n_ij−Ê_ij|=|ad−bc|/n（=3.5）", allclose(np.abs(np.array([[a, b], [c, d]]) - Ey), [[3.5] * 2] * 2, 1e-12) and close(140 / 40, 3.5, 1e-12), 3.5)
mo = charts(T, "mosaic")[0]["params"]
check(T, "chart mosaic=例1", mo["table"] == [[12, 8], [5, 15]], mo["table"])

# ================================================================ loglinear-model
T = "loglinear-model"
t22 = np.array([[40, 60], [20, 80]], float)
L = np.log(t22)
check(T, "例1 log n=(3.6889,4.0943;2.9957,4.3820)", allclose(L, [[3.6889, 4.0943], [2.9957, 4.3820]]), L.round(4).tolist())
lam = L.mean()
rm, cmn = L.mean(1), L.mean(0)
check(T, "例1 λ̂=3.7902, 行平均 3.8916, 列平均 3.3423", close(lam, 3.7902) and close(rm[0], 3.8916) and close(cmn[0], 3.3423), (lam, rm[0], cmn[0]))
check(T, "例1 λA=±0.1014, λB=∓0.4479", allclose(rm - lam, [0.1014, -0.1014]) and allclose(cmn - lam, [-0.4479, 0.4479]), (rm - lam, cmn - lam))
lab = L - rm[:, None] - cmn[None, :] + lam
check(T, "例1 λAB11=0.2452, 4λAB11=0.9808=log OR", close(lab[0, 0], 0.2452) and close(4 * lab[0, 0], 0.9808) and close(4 * lab[0, 0], math.log(40 * 80 / (60 * 20)), 1e-12), lab[0, 0])
check(T, "和0制約で λ11=λ22=−λ12=−λ21", allclose([lab[1, 1], -lab[0, 1], -lab[1, 0]], [lab[0, 0]] * 3, 1e-12), lab.round(4).tolist())
base22 = L[1, 1] - L[1, 0] - L[0, 1] + L[0, 0]
check(T, "例1 基準カテゴリ λAB22=0.9808", close(base22, 0.9808), base22)
check(T, "例1 本文の4項の和 4.3820−2.9957−4.0943+3.6889=0.9808", close(4.3820 - 2.9957 - 4.0943 + 3.6889, 0.9808, 5e-4), 4.3820 - 2.9957 - 4.0943 + 3.6889)
# 例2
mu_ind = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
G = g2(tab, mu_ind)
check(T, "例2 G²=34.299", close(G, 34.299, 5e-4), G)
G_sc = stats.chi2_contingency(tab, lambda_="log-likelihood")[0]
check(T, "例2 G²（scipy log-likelihood と一致）", close(G_sc, G, 1e-9), G_sc)
check(T, "例2 df=9−5=4, p≈6.5e-7", abs(stats.chi2.sf(G, 4) / 6.5e-7 - 1) < 0.008, stats.chi2.sf(G, 4))
Lm = np.log(mu_ind)
lB = Lm.mean(0) - Lm.mean()
lA = Lm.mean(1) - Lm.mean()
check(T, "例2 λ̂B=(−0.0608, 0.1215, −0.0608), λ̂A=(0,0,0)", allclose(lB, [-0.0608, 0.1215, -0.0608]) and allclose(lA, [0, 0, 0], 1e-12), (lB, lA))
# 独立モデルを Poisson GLM として数値当てはめ（ニュートン法、和0ではなく端点制約のダミー）
X = []
for i in range(3):
    for j in range(3):
        X.append([1, i == 1, i == 2, j == 1, j == 2])
X = np.array(X, float)
y = tab.ravel().astype(float)
beta = np.zeros(5)
beta[0] = math.log(y.mean())
for _ in range(50):
    mu = np.exp(X @ beta)
    beta += np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (y - mu))
check(T, "ポアソン回帰（端点制約）の μ̂ = n_i+ n_+j/n（制約によらず同じ）", allclose(np.exp(X @ beta), mu_ind.ravel(), 1e-8), np.exp(X @ beta)[:3])
oe = charts(T, "obs-exp-bar")[0]["params"]
check(T, "chart obs-exp-bar: observed・expected・ddof=4", oe["observed"] == tab.ravel().tolist() and allclose(oe["expected"], mu_ind.ravel(), 1e-12) and oe["ddof"] == 4 and 9 - 1 - oe["ddof"] == 4, (oe["observed"], oe["expected"], oe["ddof"]))
check(T, "sympy: IJ−{1+(I−1)+(J−1)}=(I−1)(J−1)", sp.expand(Ii * Jj - (1 + (Ii - 1) + (Jj - 1)) - (Ii - 1) * (Jj - 1)) == 0, "0")


# ================================================================ hierarchical-loglinear-model（IPF）
def ipf(obs, gens, iters=5000, tol=1e-12):
    """階層モデルの生成元 gens（軸のタプルのリスト）の周辺を合わせる反復比例当てはめ。"""
    obs = np.asarray(obs, float)
    fit = np.ones_like(obs) * obs.sum() / obs.size
    ax_all = tuple(range(obs.ndim))
    for _ in range(iters):
        old = fit.copy()
        for gset in gens:
            other = tuple(x for x in ax_all if x not in gset)
            o = obs.sum(axis=other, keepdims=True)
            f = fit.sum(axis=other, keepdims=True)
            fit = fit * o / f
        if np.max(np.abs(fit - old)) < tol:
            break
    return fit


def n_params(gens, levels):
    terms = set()
    for gset in gens:
        for r_ in range(len(gset) + 1):
            for sub in itertools.combinations(sorted(gset), r_):
                terms.add(sub)
    return sum(int(np.prod([levels[x] - 1 for x in t])) if t else 1 for t in terms)


T = "hierarchical-loglinear-model"
# 例1 自由度
lev = [2, 3, 4]
Aax, Bax, Cax = 0, 1, 2
models_df = {"[A][B][C]": ([(0,), (1,), (2,)], 17), "[AB][C]": ([(0, 1), (2,)], 15), "[AC][BC]": ([(0, 2), (1, 2)], 8), "[AB][AC][BC]": ([(0, 1), (0, 2), (1, 2)], 6)}
for nm, (gs, claim) in models_df.items():
    dfv = 24 - n_params(gs, lev)
    check(T, f"例1 {nm} 自由度={claim}", dfv == claim, dfv)
check(T, "例1 [AC][BC]=4×1×2, [AB][AC][BC]=1×2×3", 4 * 1 * 2 == 8 and 1 * 2 * 3 == 6, (8, 6))
# 一般の自由度の式（sympy）
Is, Js, Ks = sp.symbols("I J K", positive=True, integer=True)
df_formulas = {
    "[A][B][C]": (1 + (Is - 1) + (Js - 1) + (Ks - 1), Is * Js * Ks - Is - Js - Ks + 2),
    "[AB][C]": (1 + (Is - 1) + (Js - 1) + (Ks - 1) + (Is - 1) * (Js - 1), (Is * Js - 1) * (Ks - 1)),
    "[AC][BC]": (1 + (Is - 1) + (Js - 1) + (Ks - 1) + (Is - 1) * (Ks - 1) + (Js - 1) * (Ks - 1), Ks * (Is - 1) * (Js - 1)),
    "[AB][AC][BC]": (1 + (Is - 1) + (Js - 1) + (Ks - 1) + (Is - 1) * (Js - 1) + (Is - 1) * (Ks - 1) + (Js - 1) * (Ks - 1), (Is - 1) * (Js - 1) * (Ks - 1)),
}
for nm, (npar, claim) in df_formulas.items():
    check(T, f"sympy: {nm} の自由度 IJK−パラメータ数 = 表の式", sp.expand(Is * Js * Ks - npar - claim) == 0, claim)
sat = 1 + (Is - 1) + (Js - 1) + (Ks - 1) + (Is - 1) * (Js - 1) + (Is - 1) * (Ks - 1) + (Js - 1) * (Ks - 1) + (Is - 1) * (Js - 1) * (Ks - 1)
check(T, "sympy: 飽和モデルのパラメータ数=IJK（自由度0）", sp.expand(sat - Is * Js * Ks) == 0, "0")
# 例2: 軸 T(0: A/B), R(1: 回復する/しない), S(2: 軽症/重症)
tab3 = np.zeros((2, 2, 2))
tab3[:, :, 0] = [[95, 5], [270, 30]]
tab3[:, :, 1] = [[180, 120], [55, 45]]
Tx, Rx, Sx = 0, 1, 2
mods = {
    "[T][R][S]": [(0,), (1,), (2,)],
    "[TS][R]": [(0, 2), (1,)],
    "[TR][S]": [(0, 1), (2,)],
    "[RS][T]": [(1, 2), (0,)],
    "[TR][TS]": [(0, 1), (0, 2)],
    "[TR][RS]": [(0, 1), (1, 2)],
    "[TS][RS]": [(0, 2), (1, 2)],
    "[TR][TS][RS]": [(0, 1), (0, 2), (1, 2)],
}
G2s, dfs = {}, {}
for nm, gs in mods.items():
    fit = ipf(tab3, gs)
    G2s[nm] = g2(tab3, fit)
    dfs[nm] = 8 - n_params(gs, [2, 2, 2])
    for gset in gs:
        other = tuple(x for x in range(3) if x not in gset)
        assert np.allclose(fit.sum(axis=other), tab3.sum(axis=other), atol=1e-6)
check(T, "IPF の当てはめ値が生成元の周辺を再現（全8モデル）", True, "assert 通過")
table_claims = {"[T][R][S]": (332.848, 4), "[TR][TS]": (106.744, 2), "[TS][RS]": (3.389, 2), "[TR][TS][RS]": (1.049, 1)}
for nm, (gv, dv) in table_claims.items():
    check(T, f"例2 {nm} G²={gv}, df={dv}", close(G2s[nm], gv, 5e-4) and dfs[nm] == dv, (G2s[nm], dfs[nm]))
check(T, "例2 [TS][RS] p=0.184", close(stats.chi2.sf(G2s["[TS][RS]"], 2), 0.184, 5e-4), stats.chi2.sf(G2s["[TS][RS]"], 2))
check(T, "例2 [TR][TS][RS] p=0.306", close(stats.chi2.sf(G2s["[TR][TS][RS]"], 1), 0.306, 5e-4), stats.chi2.sf(G2s["[TR][TS][RS]"], 1))
check(T, "例2 [T][R][S]・[TR][TS] の p はほぼ0", stats.chi2.sf(G2s["[T][R][S]"], 4) < 1e-20 and stats.chi2.sf(G2s["[TR][TS]"], 2) < 1e-20, (stats.chi2.sf(G2s["[T][R][S]"], 4), stats.chi2.sf(G2s["[TR][TS]"], 2)))
simple = ["[T][R][S]", "[TS][R]", "[TR][S]", "[RS][T]", "[TR][TS]", "[TR][RS]", "[TS][RS]"]
fits_ = [nm for nm in simple if stats.chi2.sf(G2s[nm], dfs[nm]) > 0.05]
check(T, "例2 簡単なモデル（2因子項2個以下）で当てはまるのは [TS][RS] だけ", fits_ == ["[TS][RS]"], fits_)
l1 = stats.chi2_contingency(tab3[:, :, 0], correction=False, lambda_="log-likelihood")[0]
l2 = stats.chi2_contingency(tab3[:, :, 1], correction=False, lambda_="log-likelihood")[0]
check(T, "例2 層ごとの G² 2.619, 0.770, 和=[TS][RS]", close(l1, 2.619, 5e-4) and close(l2, 0.770, 5e-4) and close(l1 + l2, G2s["[TS][RS]"], 1e-6), (l1, l2))
dG = G2s["[TS][RS]"] - G2s["[TR][TS][RS]"]
check(T, "例2 入れ子比較 差 2.340, df 1, p=0.126", close(dG, 2.340, 5e-4) and close(stats.chi2.sf(dG, 1), 0.126, 5e-4), (dG, stats.chi2.sf(dG, 1)))
bd = charts(T, "bar-data")[0]["params"]
chart_ok = all(close(G2s[lbl], v, 5e-3) for lbl, v in zip(bd["labels"], bd["values"])) and set(bd["labels"]) == set(mods)
check(T, "chart bar-data: 8モデルの G²（小数第2位）", chart_ok, {k: round(v, 3) for k, v in G2s.items()})
# [TR][TS][RS] はIPFとポアソン回帰（ニュートン法）で一致
cells = list(itertools.product(range(2), range(2), range(2)))
Xh = np.array([[1, i, j, k, i * j, i * k, j * k] for i, j, k in cells], float)
yh = np.array([tab3[i, j, k] for i, j, k in cells])
bh = np.zeros(7)
bh[0] = math.log(yh.mean())
for _ in range(100):
    mu = np.exp(Xh @ bh)
    bh += np.linalg.solve(Xh.T @ (mu[:, None] * Xh), Xh.T @ (yh - mu))
check(T, "[TR][TS][RS]: IPF とポアソン回帰の G² が一致（1e-8）", close(g2(yh, np.exp(Xh @ bh)), G2s["[TR][TS][RS]"], 1e-8), g2(yh, np.exp(Xh @ bh)))
# ピアソン χ² の差は分解しないことの例（G² は分解）
Gs3 = G2s["[T][R][S]"] - G2s["[TS][R]"]
check(T, "入れ子比較: G² の差は非負（全入れ子ペア）", all(G2s[m0] - G2s[m1] >= -1e-9 for m0, m1 in [("[T][R][S]", "[TS][R]"), ("[TS][R]", "[TS][RS]"), ("[TS][RS]", "[TR][TS][RS]"), ("[TR][TS]", "[TR][TS][RS]")]), Gs3)

# ================================================================ sampling-schemes
T = "sampling-schemes"
mu = np.array([[6, 9], [4, 6]], float)
obs = np.array([[3, 10], [5, 6]])
pp = np.prod(stats.poisson.pmf(obs, mu))
check(T, "例1 ポアソン確率の積 2.656e-4", abs(pp - 2.656e-4) < 5e-8, pp)
pt = stats.poisson.pmf(24, 25)
check(T, "例1 P(Po(25)=24)=0.07952", close(pt, 0.07952, 5e-6), pt)
mn = stats.multinomial.pmf(obs.ravel(), 24, (mu / 25).ravel())
check(T, "例1 比 0.003340 = 多項確率 0.003340", close(pp / pt, 0.003340, 5e-7) and close(mn, 0.003340, 5e-7) and close(pp / pt, mn, 1e-12), (pp / pt, mn))
prow = stats.multinomial.pmf([13, 11], 24, [15 / 25, 10 / 25])
check(T, "例1 行合計 (13,11) の確率 0.13674", close(prow, 0.13674, 5e-6), prow)
pb = stats.binom.pmf(3, 13, 0.4) * stats.binom.pmf(5, 11, 0.4)
check(T, "例1 多項/行周辺 0.02443 = Bin(13,.4)(3)·Bin(11,.4)(5)", close(mn / prow, 0.02443, 5e-6) and close(pb, 0.02443, 5e-6) and close(mn / prow, pb, 1e-12), (mn / prow, pb))
check(T, "例1 π_{1|i}=0.4（6/15, 4/10）", close(6 / 15, 0.4, 1e-12) and close(4 / 10, 0.4, 1e-12), 0.4)
pc = stats.binom.pmf(8, 24, 0.4)
check(T, "例1 P(Bin(24,.4)=8)=0.13598", close(pc, 0.13598, 5e-6), pc)
hg = stats.hypergeom.pmf(3, 24, 13, 8)
hg2 = math.comb(13, 3) * math.comb(11, 5) / math.comb(24, 8)
check(T, "例1 0.02443/0.13598=0.17966 = 超幾何", close(pb / pc, 0.17966, 5e-6) and close(hg, 0.17966, 5e-6) and close(pb / pc, hg2, 1e-12), (pb / pc, hg))
# 成功率によらない
vals = []
for q in [0.1, 0.3, 0.7]:
    vals.append(stats.binom.pmf(3, 13, q) * stats.binom.pmf(5, 11, q) / stats.binom.pmf(8, 24, q))
check(T, "条件付き確率は共通成功率によらない（q=0.1/0.3/0.7）", allclose(vals, [hg2] * 3, 1e-12), vals)
# sympy: ポアソン→多項の一般式
m11, m12, m21, m22 = sp.symbols("mu11 mu12 mu21 mu22", positive=True)
k = [3, 10, 5, 6]
mus = [m11, m12, m21, m22]
Mu = sum(mus)
lhs = sp.prod([sp.exp(-m) * m ** kk / sp.factorial(kk) for m, kk in zip(mus, k)]) / (sp.exp(-Mu) * Mu ** 24 / sp.factorial(24))
rhs = sp.factorial(24) / sp.prod([sp.factorial(kk) for kk in k]) * sp.prod([(m / Mu) ** kk for m, kk in zip(mus, k)])
check(T, "sympy: Π Po / Po(総数) = 多項（例1の度数で記号的に）", sp.simplify(lhs / rhs) == 1, "1")
# 乱数: ポアソン → 総数24で条件付け → 多項
Ps = rng.poisson(mu.ravel(), size=(4000000, 4))
selp = Ps.sum(1) == 24
freq = np.mean((Ps[selp] == obs.ravel()).all(1))
check(T, "乱数400万回: ポアソンを総数24で条件付けた頻度≈0.003340（許容 0.0003≈3SE）", abs(freq - 0.00334) < 0.0003, (freq, selp.sum()))
# 自由度
check(T, "sympy: 3モデルの自由度がすべて (I−1)(J−1)", all(sp.expand(e - (Ii - 1) * (Jj - 1)) == 0 for e in [Ii * Jj - (Ii + Jj - 1), (Ii * Jj - 1) - ((Ii - 1) + (Jj - 1)), Ii * (Jj - 1) - (Jj - 1)]), "0")
# 同じ期待度数の MLE: 積多項で同質性の MLE は n_i+ n_+j / n
ptab = np.array([[3, 10], [5, 6]], float)
pi_hat = ptab.sum(0) / ptab.sum()
check(T, "積多項の同質性 MLE n_i+·π̂_j = n_i+ n_+j/n", allclose(ptab.sum(1)[:, None] * pi_hat[None, :], pearson(ptab)[1], 1e-12), pi_hat)
hc = chart_type(T, "pmf")[0]["params"]
check(T, "chart hypergeom N=24, K=13（第1行）, n=8（第1列）, 中心 n1+ n+1/n", hc == {"N": 24, "K": 13, "n": 8}, (hc, 13 * 8 / 24))

# ================================================================ three-way-table
T = "three-way-table"
y = np.array([[80, 20], [40, 10]], float)
o = np.array([[10, 40], [20, 80]], float)
ors = [y[0, 0] * y[1, 1] / (y[0, 1] * y[1, 0]), o[0, 0] * o[1, 1] / (o[0, 1] * o[1, 0])]
check(T, "例1 部分表 OR=1, 1", allclose(ors, [1, 1], 1e-12), ors)
check(T, "例1 通販利用率 0.80, 0.80, 0.20, 0.20", allclose([80 / 100, 40 / 50, 10 / 50, 20 / 100], [0.8, 0.8, 0.2, 0.2], 1e-12), "ok")
mg = y + o
check(T, "例1 周辺表 (90,60;60,90), OR=2.25", mg.tolist() == [[90, 60], [60, 90]] and close(90 * 90 / 3600, 2.25, 1e-12), mg.tolist())
xm = stats.chi2_contingency(mg, correction=False)
check(T, "例1 χ²=12.0, p=0.0005", close(xm[0], 12.0, 5e-2) and close(xm[1], 0.0005, 5e-5), (xm[0], xm[1]))
ca = np.array([y.sum(1), o.sum(1)])
cb = np.array([y.sum(0), o.sum(0)])
check(T, "例1 C×A (100,50;50,100), C×B (120,30;30,120)", ca.tolist() == [[100, 50], [50, 100]] and cb.tolist() == [[120, 30], [30, 120]], (ca.tolist(), cb.tolist()))
t3 = np.stack([y, o], axis=2)
fit_ci = ipf(t3, [(0, 2), (1, 2)])
check(T, "例1 [AC][BC] の当てはめは観測を完全再現（G²=0）", close(g2(t3, fit_ci), 0, 1e-8), g2(t3, fit_ci))
fit_closed = t3.sum(1, keepdims=True) * t3.sum(0, keepdims=True) / t3.sum((0, 1), keepdims=True)
check(T, "条件付き独立の期待度数 n_i+k n_+jk/n_++k = IPF", allclose(fit_closed, fit_ci, 1e-8), "一致")
# 含意関係: ランダムな [A][BC] 分布は条件付き独立かつ周辺独立
ok_all = True
for _ in range(200):
    pa = rng.dirichlet(np.ones(3))
    pbc = rng.dirichlet(np.ones(12)).reshape(3, 4)
    P = pa[:, None, None] * pbc[None, :, :]
    pab = P.sum(2)
    ok_m = np.allclose(pab, pab.sum(1, keepdims=True) * pab.sum(0, keepdims=True))
    ok_c = np.allclose(P, P.sum(1, keepdims=True) * P.sum(0, keepdims=True) / P.sum((0, 1), keepdims=True))
    ok_all &= ok_m and ok_c
check(T, "乱数200組: 同時独立 [A][BC] ⇒ 条件付き独立かつ周辺独立", ok_all, ok_all)
# [AB][C] なら部分表OR=周辺OR
ok_all = True
for _ in range(200):
    pab = rng.dirichlet(np.ones(4)).reshape(2, 2)
    pc_ = rng.dirichlet(np.ones(3))
    P = pab[:, :, None] * pc_[None, None, :]
    orm = pab[0, 0] * pab[1, 1] / (pab[0, 1] * pab[1, 0])
    orsk = P[0, 0] * P[1, 1] / (P[0, 1] * P[1, 0])
    ok_all &= np.allclose(orsk, orm)
check(T, "乱数200組: [AB][C] なら部分表 OR=周辺表 OR", ok_all, ok_all)
check(T, "例1 は条件付き独立だが周辺独立でない（反例）", allclose(ors, [1, 1], 1e-12) and 90 * 90 != 60 * 60, "OR 1,1 vs 2.25")
# 周辺独立だが条件付き独立でない例も作れる
P = np.zeros((2, 2, 2))
P[:, :, 0] = [[0.2, 0.05], [0.05, 0.2]]
P[:, :, 1] = [[0.05, 0.2], [0.2, 0.05]]
pab = P.sum(2)
check(T, "反例: 周辺独立だが条件付き独立でない表が存在", np.allclose(pab, pab.sum(1, keepdims=True) * pab.sum(0, keepdims=True)) and not np.allclose(P[0, 0, 0] * P[1, 1, 0], P[0, 1, 0] * P[1, 0, 0]), pab.tolist())
mos = charts(T, "mosaic")
check(T, "chart mosaic 3枚=若年・高齢・併合", [m["params"]["table"] for m in mos] == [[[80, 20], [40, 10]], [[10, 40], [20, 80]], [[90, 60], [60, 90]]], [m["params"]["table"] for m in mos])

# ================================================================ mantel-haenszel-estimator
T = "mantel-haenszel-estimator"
H = np.array([[15, 10, 9, 16], [20, 20, 12, 28], [8, 12, 4, 16]], float)


def mh(Hs):
    a_, b_, c_, d_ = Hs.T
    nk = Hs.sum(1)
    Rk, Sk = a_ * d_ / nk, b_ * c_ / nk
    Pk, Qk = (a_ + d_) / nk, (b_ + c_) / nk
    R, S = Rk.sum(), Sk.sum()
    t = ((Pk * Rk).sum() / (2 * R ** 2), (Pk * Sk + Qk * Rk).sum() / (2 * R * S), (Qk * Sk).sum() / (2 * S ** 2))
    return dict(nk=nk, Rk=Rk, Sk=Sk, Pk=Pk, Qk=Qk, R=R, S=S, OR=R / S, terms=t, V=sum(t))


r = mh(H)
check(T, "例1 n_k=50,80,40", r["nk"].tolist() == [50, 80, 40], r["nk"])
check(T, "例1 R_k=4.8,7.0,3.2, S_k=1.8,3.0,1.2", allclose(r["Rk"], [4.8, 7.0, 3.2], 1e-12) and allclose(r["Sk"], [1.8, 3.0, 1.2], 1e-12), (r["Rk"], r["Sk"]))
check(T, "例1 OR_MH=15/6=2.500", close(r["R"], 15, 1e-12) and close(r["S"], 6, 1e-12) and close(r["OR"], 2.5, 1e-12), r["OR"])
w = r["Sk"] / r["S"]
ork = H[:, 0] * H[:, 3] / (H[:, 1] * H[:, 2])
check(T, "例1 重み (0.3,0.5,0.2), 層 OR 2.667/2.333/2.667, 加重平均=2.5", allclose(w, [0.3, 0.5, 0.2], 1e-12) and allclose(ork, [2.667, 2.333, 2.667], 5e-4) and close((w * ork).sum(), 2.5, 1e-12), (w, ork))
check(T, "例1 P_k=(0.62,0.60,0.60), Q_k=(0.38,0.40,0.40)", allclose(r["Pk"], [0.62, 0.6, 0.6], 1e-12) and allclose(r["Qk"], [0.38, 0.4, 0.4], 1e-12), (r["Pk"], r["Qk"]))
check(T, "例1 RGB 0.02021+0.05300+0.03283=0.10605", allclose(r["terms"], [0.02021, 0.05300, 0.03283], 5e-6) and close(r["V"], 0.10605, 5e-6), (r["terms"], r["V"]))
se = math.sqrt(r["V"])
check(T, "例1 SE=0.3256, log2.5=0.9163", close(se, 0.3256) and close(math.log(2.5), 0.9163), se)
lo, hi = math.exp(math.log(2.5) - z975 * se), math.exp(math.log(2.5) + z975 * se)
check(T, "例1 95%区間 (1.321, 4.733)", close(lo, 1.321, 5e-4) and close(hi, 4.733, 5e-4), (lo, hi))
lw = np.log(ork)
wv = 1 / (1 / H).sum(1)
woolf = math.exp((wv * lw).sum() / wv.sum())
check(T, "例1 ウルフの方法も 2.500（小数第3位）", close(woolf, 2.5, 5e-4), woolf)
check(T, "MH は層の OR の最小〜最大の間", ork.min() <= r["OR"] <= ork.max(), (ork.min(), ork.max()))
check(T, "層が1つなら ad/bc", close(mh(H[:1])["OR"], ork[0], 1e-12), mh(H[:1])["OR"])
SP = np.array([[95, 5, 270, 30], [180, 120, 55, 45]], float)
r2 = mh(SP)
check(T, "例2 R_k=7.125, 20.25, S_k=3.375, 16.5", allclose(r2["Rk"], [7.125, 20.25], 1e-12) and allclose(r2["Sk"], [3.375, 16.5], 1e-12) and r2["nk"].tolist() == [400, 400], (r2["Rk"], r2["Sk"]))
check(T, "例2 OR_MH=1.377", close(r2["OR"], 1.377, 5e-4), r2["OR"])
pooled = SP.sum(0)
check(T, "例2 併合 OR=0.508", close(pooled[0] * pooled[3] / (pooled[1] * pooled[2]), 0.508, 5e-4), pooled[0] * pooled[3] / (pooled[1] * pooled[2]))
se2 = math.sqrt(r2["V"])
lo, hi = math.exp(math.log(r2["OR"]) - z975 * se2), math.exp(math.log(r2["OR"]) + z975 * se2)
check(T, "例2 RGB SE=0.2081, 95%区間 (0.916, 2.071)", close(se2, 0.2081) and close(lo, 0.916, 5e-4) and close(hi, 2.071, 5e-4), (se2, lo, hi))
# 乱数: 共通 OR の層別データで MH の一致性と RGB 分散
psi = 2.5
base = np.array([0.36, 0.30, 0.20])
nn_ = np.array([[25, 25], [40, 40], [20, 20]])
estims = []
for _ in range(20000):
    rows = []
    for kk in range(3):
        q0 = base[kk]
        q1 = psi * q0 / (1 - q0) / (1 + psi * q0 / (1 - q0))
        x1 = rng.binomial(nn_[kk, 0], q1)
        x0 = rng.binomial(nn_[kk, 1], q0)
        rows.append([x1, nn_[kk, 0] - x1, x0, nn_[kk, 1] - x0])
    rr_ = mh(np.array(rows, float))
    if rr_["S"] > 0 and rr_["R"] > 0:
        estims.append((math.log(rr_["OR"]), rr_["V"]))
estims = np.array(estims)
check(T, "乱数2万回: 共通 OR=2.5 で log OR_MH の平均≈log2.5（許容 0.03）、分散≈RGB の平均（相対許容 8%）",
      abs(estims[:, 0].mean() - math.log(2.5)) < 0.03 and abs(estims[:, 0].var() / estims[:, 1].mean() - 1) < 0.08,
      (estims[:, 0].mean(), estims[:, 0].var(), estims[:, 1].mean()))
bd = charts(T, "bar-data")[0]["params"]
check(T, "chart bar-data: 層 OR と MH 2.5", allclose(bd["values"], list(ork) + [2.5], 5e-5), bd["values"])

# ================================================================ mantel-haenszel-test
T = "mantel-haenszel-test"


def cmh(Hs, cc=False):
    a_, b_, c_, d_ = Hs.T
    nk = Hs.sum(1)
    n1k, n2k, m1k, m2k = a_ + b_, c_ + d_, a_ + c_, b_ + d_
    Ek = n1k * m1k / nk
    Vk = n1k * n2k * m1k * m2k / (nk ** 2 * (nk - 1))
    num = abs(a_.sum() - Ek.sum()) - (0.5 if cc else 0)
    return dict(Ek=Ek, Vk=Vk, dev=a_ - Ek, stat=num ** 2 / Vk.sum())


q = cmh(H)
check(T, "例1 E_k=12,16,6", allclose(q["Ek"], [12, 16, 6], 1e-12), q["Ek"])
check(T, "例1 V_k=3.1837, 4.8608, 2.1538", allclose(q["Vk"], [3.1837, 4.8608, 2.1538]), q["Vk"])
check(T, "例1 a−E=3,4,2, Σa=43, ΣE=34, ΣV=10.1983", allclose(q["dev"], [3, 4, 2], 1e-12) and H[:, 0].sum() == 43 and close(q["Ek"].sum(), 34, 1e-12) and close(q["Vk"].sum(), 10.1983), q["Vk"].sum())
check(T, "例1 χ²_MH=81/10.1983=7.943, p=0.0048", close(q["stat"], 7.943, 5e-4) and close(stats.chi2.sf(q["stat"], 1), 0.0048), (q["stat"], stats.chi2.sf(q["stat"], 1)))
qc = cmh(H, cc=True)
check(T, "例1 連続修正 7.085, p=0.0078", close(qc["stat"], 7.085, 5e-4) and close(stats.chi2.sf(qc["stat"], 1), 0.0078), (qc["stat"], stats.chi2.sf(qc["stat"], 1)))
check(T, "chart caption 7.94 は上側確率 0.005", close(q["stat"], 7.94, 5e-3) and close(stats.chi2.sf(q["stat"], 1), 0.005, 5e-4), stats.chi2.sf(q["stat"], 1))
# 超幾何の平均・分散を scipy で
hv = [(stats.hypergeom.mean(nk, m1k, n1k), stats.hypergeom.var(nk, m1k, n1k)) for nk, m1k, n1k in [(50, 24, 25), (80, 32, 40), (40, 12, 20)]]
check(T, "E[a_k], V[a_k] が超幾何分布（scipy）と一致", allclose([h[0] for h in hv], q["Ek"], 1e-10) and allclose([h[1] for h in hv], q["Vk"], 1e-10), hv)
lay3 = stats.chi2_contingency(H[2].reshape(2, 2), correction=False)[1]
check(T, "例1 病院3 単独では有意でない", lay3 > 0.05, lay3)
q2 = cmh(SP)
check(T, "例2 軽症 a−E=3.75, V=6.0033; 重症 a−E=3.75, V=18.2213", allclose(q2["dev"], [3.75, 3.75], 1e-12) and allclose(q2["Ek"], [91.25, 176.25], 1e-12) and allclose(q2["Vk"], [6.0033, 18.2213]), (q2["Ek"], q2["Vk"]))
check(T, "例2 ΣV=24.2246, χ²_MH=2.322, p=0.128", close(q2["Vk"].sum(), 24.2246) and close(q2["stat"], 2.322, 5e-4) and close(stats.chi2.sf(q2["stat"], 1), 0.128, 5e-4), (q2["Vk"].sum(), q2["stat"]))
# K=1 の式
Ad, Bd, Cd, Dd = sp.symbols("a b c d", positive=True)
Nd = Ad + Bd + Cd + Dd
ek = (Ad + Bd) * (Ad + Cd) / Nd
vk = (Ad + Bd) * (Cd + Dd) * (Ad + Cd) * (Bd + Dd) / (Nd ** 2 * (Nd - 1))
k1 = sp.simplify((Ad - ek) ** 2 / vk - (Nd - 1) * (Ad * Dd - Bd * Cd) ** 2 / ((Ad + Bd) * (Cd + Dd) * (Ad + Cd) * (Bd + Dd)))
check(T, "sympy: K=1 で χ²_MH=(n−1)(ad−bc)²/(n1n2m1m2)", k1 == 0, k1)
pooledH = H.sum(0).reshape(2, 2)
pc_x = stats.chi2_contingency(pooledH, correction=False)[0]
check(T, "例1 は併合しても似た値（併合 χ² と χ²_MH の差 < 1）", abs(pc_x - q["stat"]) < 1, pc_x)
pool2 = SP.sum(0).reshape(2, 2)
pc2 = stats.chi2_contingency(pool2, correction=False)
check(T, "例2 併合表では B が良い強い関連（p<0.001）", pc2[1] < 0.001 and pool2[0, 0] / pool2[0].sum() < pool2[1, 0] / pool2[1].sum(), (pc2[0], pc2[1]))
# 乱数: H0 の下で χ²_MH の棄却率
rej = 0
reps = 20000
for _ in range(reps):
    rows = []
    for kk in range(3):
        x1 = rng.binomial(nn_[kk, 0], base[kk])
        x0 = rng.binomial(nn_[kk, 1], base[kk])
        rows.append([x1, nn_[kk, 0] - x1, x0, nn_[kk, 1] - x0])
    st = cmh(np.array(rows, float))["stat"]
    rej += st > stats.chi2.ppf(0.95, 1)
check(T, "乱数2万回: H0（層ごとに基準リスクが違う）で棄却率≈0.05（許容 0.008）", abs(rej / reps - 0.05) < 0.008, rej / reps)
bd = charts(T, "bar-data")[0]["params"]
pc_c = chart_type(T, "pdf-cdf")[0]["params"]
check(T, "chart bar-data (3,4,2), 合計9・chi2 df=1", bd["values"] == [3, 4, 2] and sum(bd["values"]) == 9 and pc_c == {"df": 1}, (bd["values"], pc_c))

# ================================================================ simpsons-paradox
T = "simpsons-paradox"
rates = [95 / 100, 270 / 300, 180 / 300, 55 / 100, 275 / 400, 325 / 400]
check(T, "例1 回復率 0.950/0.900/0.600/0.550/0.6875/0.8125", allclose(rates, [0.95, 0.9, 0.6, 0.55, 0.6875, 0.8125], 1e-12), rates)
check(T, "例1 全体の度数 A 275/125, B 325/75", (95 + 180, 5 + 120, 270 + 55, 30 + 45) == (275, 125, 325, 75), "ok")
check(T, "例1 層で A が5ポイント高い・全体で B が12.5ポイント高い", close(0.95 - 0.9, 0.05, 1e-12) and close(0.6 - 0.55, 0.05, 1e-12) and close(0.8125 - 0.6875, 0.125, 1e-12), "ok")
check(T, "例1 加重平均 0.2375+0.4500=0.6875, 0.6750+0.1375=0.8125", close(0.25 * 0.95, 0.2375, 1e-12) and close(0.75 * 0.6, 0.45, 1e-12) and close(0.75 * 0.9, 0.675, 1e-12) and close(0.25 * 0.55, 0.1375, 1e-12) and close(0.25 * 0.95 + 0.75 * 0.6, 0.6875, 1e-12) and close(0.75 * 0.9 + 0.25 * 0.55, 0.8125, 1e-12), "ok")
o1, o2 = 95 * 30 / (5 * 270), 180 * 45 / (120 * 55)
op = 275 * 75 / (125 * 325)
check(T, "例2 OR 軽症 2.111, 重症 1.227, 併合 0.508", close(o1, 2.111, 5e-4) and close(o2, 1.227, 5e-4) and close(op, 0.508, 5e-4), (o1, o2, op))
check(T, "例2 MH=27.375/19.875=1.377", close(7.125 + 20.25, 27.375, 1e-12) and close(3.375 + 16.5, 19.875, 1e-12) and close(27.375 / 19.875, 1.377, 5e-4) and close(r2["OR"], 27.375 / 19.875, 1e-12), 27.375 / 19.875)
# 乱数: 母集団の割合でも起こる（構造上の現象）: 本文の割合で大標本
nsim = 400000
trtA = rng.random(nsim) < 0.5
severe = np.where(trtA, rng.random(nsim) < 0.75, rng.random(nsim) < 0.25)
prob = np.select([trtA & ~severe, ~trtA & ~severe, trtA & severe], [0.95, 0.90, 0.60], 0.55)
rec = rng.random(nsim) < prob
check(T, "乱数40万人: 全体の回復率 A≈0.6875, B≈0.8125（許容 0.005）", abs(rec[trtA].mean() - 0.6875) < 0.005 and abs(rec[~trtA].mean() - 0.8125) < 0.005, (rec[trtA].mean(), rec[~trtA].mean()))
# Z が X と無関係なら逆転しない（重みが同じ）
check(T, "重みが治療で同じなら逆転しない（0.5ずつなら A 0.775 > B 0.725）", 0.5 * 0.95 + 0.5 * 0.6 > 0.5 * 0.9 + 0.5 * 0.55, (0.775, 0.725))
bd = charts(T, "bar-data")[0]["params"]
mo = charts(T, "mosaic")[0]["params"]
check(T, "chart bar-data=回復率6本", allclose(bd["values"], rates, 1e-12), bd["values"])
check(T, "chart mosaic=重症度×治療 (100,300;300,100)", mo["table"] == [[100, 300], [300, 100]] and mo["rowLabels"] == ["軽症", "重症"] and mo["colLabels"] == ["治療A", "治療B"], mo["table"])

# ----------------------------------------------------------------
ok = sum(results)
ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
