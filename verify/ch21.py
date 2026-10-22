"""第21章 検算スクリプト（標本調査法）。
小さな母集団の数値例は、全標本を列挙して期待値・分散・MSE を独立に再計算し、本文の公式値・近似値と比べる。
記号的な主張（指示変数の共分散、分散の分解、ラグランジュ乗数法、近似分散の最小化など）は sympy で確認。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数シミュレーションは seed=20261023、許容誤差は各チェックに明記（標準誤差の4倍以上）。
"""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    if claims.size == 1:
        claims = np.full(xs.shape, claims[0])
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol))


def front(term):
    txt = (ROOT / "content" / "ch21" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def S2(v):
    v = np.asarray(v, float)
    return float(((v - v.mean()) ** 2).sum() / (len(v) - 1))


def dist_moments(vals, probs=None):
    """値と確率の列から期待値・分散（厳密に確率で重み付け）"""
    vals = np.asarray(vals, float)
    if probs is None:
        probs = np.full(len(vals), 1 / len(vals))
    probs = np.asarray(probs, float)
    m = float((probs * vals).sum())
    return m, float((probs * (vals - m) ** 2).sum())


# ================= simple-random-sampling =================
T = "simple-random-sampling"
pop = [3, 5, 6, 8, 9, 11]
N, n = 6, 3
check(T, "例1 Ȳ=7, S²=42/5=8.4", np.mean(pop) == 7 and close(S2(pop), 8.4, 1e-12), (np.mean(pop), S2(pop)))
samples = list(combinations(pop, n))
means = [F(sum(s), n) for s in samples]
check(T, "例1 標本は C(6,3)=20 通り", len(samples) == 20, len(samples))
Em = sum(means) / 20
Vm = sum((m - Em) ** 2 for m in means) / 20
check(T, "例1 列挙で E[ȳ]=7（厳密）", Em == 7, Em)
check(T, "例1 列挙で V(ȳ)=1.4（厳密）", Vm == F(7, 5), Vm)
check(T, "例1 公式 (1−3/6)·8.4/3=1.4", close((1 - 3 / 6) * 8.4 / 3, 1.4, 1e-12), (1 - 3 / 6) * 8.4 / 3)
s2s = [F(sum((F(x) - F(sum(s), n)) ** 2 for x in s), n - 1) for s in samples]
check(T, "例1 20個の s² の平均=8.4（E[s²]=S²）", sum(s2s) / 20 == F(42, 5), sum(s2s) / 20)
sig2 = np.var(pop)
check(T, "例1 復元 σ²/n=7/3≈2.3333", close(sig2, 7, 1e-12) and close(sig2 / 3, 2.3333), sig2 / 3)
# bar-data
from collections import Counter
cnt = Counter(round(float(m), 3) for m in means)
ch = charts(T)
labs = [float(v) for v in ch[0]["params"]["labels"]]
vals = ch[0]["params"]["values"]
check(T, "bar-data 標本平均の度数分布が列挙と一致",
      sorted(cnt.keys()) == labs and [cnt[l] for l in labs] == vals and sum(vals) == 20, dict(cnt))
ys = ch[1]["params"]["series"]
nonrep = [(1 - k / 6) * 8.4 / k for k in range(1, 7)]
rep = [7 / k for k in range(1, 7)]
check(T, "line-data 非復元 (1−n/6)·8.4/n", allclose(nonrep, ys[0]["y"], 5e-4), np.round(nonrep, 4))
check(T, "line-data 復元 7/n", allclose(rep, ys[1]["y"], 5e-4), np.round(rep, 4))
# 列挙で n=1..6 の分散も確認
ok = True
for k in range(1, 7):
    ms = [np.mean(s) for s in combinations(pop, k)]
    ok &= close(np.var(ms), nonrep[k - 1], 1e-10)
check(T, "line-data 非復元の値を各 n で全標本列挙と照合", ok, "n=1..6")
# 指示変数の共分散（sympy）
Ns, ns = sp.symbols("N n", positive=True)
f = ns / Ns
cov = ns * (ns - 1) / (Ns * (Ns - 1)) - f ** 2
check(T, "Cov(Z_i,Z_j)=−f(1−f)/(N−1)（sympy）", sp.simplify(cov + f * (1 - f) / (Ns - 1)) == 0, sp.factor(cov))
# 分散の導出: V(ΣZiYi)=f(1−f)N S²（一般の Y で記号確認, N=5）
Y = sp.symbols("Y1:6")
Nn = 5
fs = sp.symbols("f")
lhs = fs * (1 - fs) * sum(y ** 2 for y in Y) - fs * (1 - fs) / (Nn - 1) * sum(Y[i] * Y[j] for i in range(Nn) for j in range(Nn) if i != j)
Yb = sum(Y) / Nn
rhs = fs * (1 - fs) * Nn * sum((y - Yb) ** 2 for y in Y) / (Nn - 1)
check(T, "V(ΣZ_iY_i)=f(1−f)NS²（sympy, N=5 の一般の Y）", sp.expand(lhs - rhs) == 0, "恒等式")
# 比率の S²
P = sp.symbols("P")
Ss = (Ns * P * (1 - P) ** 2 + Ns * (1 - P) * P ** 2) / (Ns - 1)
check(T, "0/1 値で S²=N/(N−1)·P(1−P)（sympy）", sp.simplify(Ss - Ns / (Ns - 1) * P * (1 - P)) == 0, sp.factor(Ss))
# σ² 流儀との一致
sig = (Ns - 1) / Ns * sp.Symbol("S2")
check(T, "(N−n)/(N−1)·σ²/n=(1−n/N)S²/n（sympy）",
      sp.simplify((Ns - ns) / (Ns - 1) * sig / ns - (1 - ns / Ns) * sp.Symbol("S2") / ns) == 0, "恒等式")
# 0/1 母集団で: 1 の個数が超幾何分布、(1−f)p̂(1−p̂)/(n−1) が V(p̂) の不偏推定量
pop01 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
N01, n01 = 10, 4
cnts = Counter(sum(s) for s in combinations(pop01, n01))
tot = math.comb(10, 4)
hyp = all(close(cnts[k] / tot, stats.hypergeom.pmf(k, 10, 3, 4), 1e-12) for k in range(4))
ph = [F(sum(s), n01) for s in combinations(pop01, n01)]
Vp = sum((p - F(3, 10)) ** 2 for p in ph) / tot
Ev = sum((1 - F(n01, N01)) * p * (1 - p) / (n01 - 1) for p in ph) / tot
check(T, "0/1 母集団(N=10,n=4) で1の個数が超幾何分布、(1−f)p̂(1−p̂)/(n−1) が不偏", hyp and Vp == Ev, (Vp, Ev))
# 例2
v = (1 - 400 / 2000) * 0.35 * 0.65 / 399
se = math.sqrt(v)
check(T, "例2 V̂(p̂)≈0.000456", close(v, 0.000456, 5e-7), v)
check(T, "例2 SE≈0.0214", close(se, 0.0214), se)
check(T, "例2 半幅 1.96×0.0214≈0.0419", close(1.96 * 0.0214, 0.0419) and close(1.96 * se, 0.0419), 1.96 * se)
check(T, "例2 fpc なしの SE≈0.0238", close(math.sqrt(0.35 * 0.65 / 400), 0.0238), math.sqrt(0.35 * 0.65 / 400))
n0 = 1.96 ** 2 * 0.25 / 0.03 ** 2
nn = n0 / (1 + n0 / 2000)
check(T, "例3 n0≈1067.1", close(n0, 1067.1, 5e-2), n0)
check(T, "例3 n=n0/(1+n0/N)≈695.8 → 696人", close(nn, 695.8, 5e-2) and math.ceil(nn) == 696, nn)
# 乱数: 非復元SRS の V(ȳ)
popL = rng.normal(50, 10, 200)
reps = 100000
mm = np.array([popL[rng.permutation(200)[:40]].mean() for _ in range(reps)])
theo = (1 - 40 / 200) * S2(popL) / 40
check(T, "乱数 N=200,n=40 で V(ȳ)=(1−f)S²/n（10万回、相対誤差 2%、SE は約0.45%）",
      abs(mm.var() / theo - 1) < 0.02, (mm.var(), theo))

# ================= stratified-sampling =================
T = "stratified-sampling"
h1 = [3, 4, 5, 5, 6, 7]
h2 = [10, 12, 14, 16, 18, 20]
allv = h1 + h2
check(T, "例1 Ȳ1=5,S1²=2,Ȳ2=15,S2²=14",
      np.mean(h1) == 5 and close(S2(h1), 2, 1e-12) and np.mean(h2) == 15 and close(S2(h2), 14, 1e-12),
      (S2(h1), S2(h2)))
check(T, "例1 Ȳ=10, S²≈34.545", np.mean(allv) == 10 and close(S2(allv), 34.545, 5e-4), S2(allv))
SS_tot = 11 * S2(allv)
check(T, "平方和の分解 (N−1)S²=Σ(N_h−1)S_h²+ΣN_h(Ȳ_h−Ȳ)²",
      close(SS_tot, 5 * 2 + 5 * 14 + 6 * 25 + 6 * 25, 1e-9), SS_tot)


def strat_enum(strata, nh):
    W = [len(s) / sum(len(t) for t in strata) for s in strata]
    combos = [list(combinations(s, k)) for s, k in zip(strata, nh)]
    ests = [sum(w * np.mean(c) for w, c in zip(W, cs)) for cs in product(*combos)]
    return len(ests), *dist_moments(ests)


def strat_formula(strata, nh):
    Ntot = sum(len(s) for s in strata)
    return sum((len(s) / Ntot) ** 2 * (1 - k / len(s)) * S2(s) / k for s, k in zip(strata, nh))


cntn, E22, V22 = strat_enum([h1, h2], [2, 2])
vf = strat_formula([h1, h2], [2, 2])
check(T, "例1 公式 0.1667+1.1667=1.3333",
      close(0.25 * (2 / 3) * 1, 0.1667) and close(0.25 * (2 / 3) * 7, 1.1667) and close(vf, 1.3333), vf)
check(T, "例1 225通り列挙で期待値10・分散1.3333", cntn == 225 and close(E22, 10, 1e-12) and close(V22, 1.3333), (E22, V22))
srs = [np.mean(s) for s in combinations(allv, 4)]
Es, Vs = dist_moments(srs)
check(T, "例1 SRS 495通り列挙で分散≈5.7576（公式とも一致）",
      len(srs) == 495 and close(Vs, 5.7576) and close((1 - 4 / 12) * S2(allv) / 4, Vs, 1e-10) and close(Es, 10, 1e-12), Vs)
check(T, "例1 分散比 約1/4.3", close(Vs / V22, 4.3, 5e-2), Vs / V22)
c13, E13, V13 = strat_enum([h1, h2], [1, 3])
check(T, "例1 (1,3) 配分で分散 1.0（列挙・公式）", close(V13, 1.0, 1e-10) and close(strat_formula([h1, h2], [1, 3]), 1.0, 1e-10), V13)
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [5.758,1.333,1.0]", allclose([Vs, V22, V13], ch, 5e-4), ch)
# 例2
Nh = np.array([400, 250, 100])
nh = np.array([20, 15, 10])
yb = np.array([30, 45, 80])
s2h = np.array([64, 100, 400])
W = Nh / Nh.sum()
check(T, "例2 W=(0.5333,0.3333,0.1333)", allclose(W, [0.5333, 0.3333, 0.1333]), W)
yst = (W * yb).sum()
check(T, "例2 ȳst=41.667", close(yst, 41.667, 5e-4), yst)
terms = W ** 2 * (1 - nh / Nh) * s2h / nh
check(T, "例2 各層の項 0.8647, 0.6963, 0.6400", allclose(terms, [0.8647, 0.6963, 0.6400]), terms)
check(T, "例2 合計 2.2010, SE≈1.484", close(terms.sum(), 2.2010) and close(math.sqrt(terms.sum()), 1.484, 5e-4),
      (terms.sum(), math.sqrt(terms.sum())))
check(T, "例2 単純平均 46.111", close((nh * yb).sum() / nh.sum(), 46.111, 5e-4), (nh * yb).sum() / nh.sum())

# ================= proportional-allocation =================
T = "proportional-allocation"
Nh = np.array([500, 300, 200])
Yh = np.array([50, 60, 80])
Sh = np.array([10, 20, 40])
W = Nh / 1000
n = 100
fpc = 0.9
Vprop = fpc / n * (W * Sh ** 2).sum()
check(T, "例1 比例配分 n_h=(50,30,20)", allclose(n * W, [50, 30, 20]), n * W)
check(T, "例1 V_prop=0.009×490=4.41", close((W * Sh ** 2).sum(), 490, 1e-9) and close(Vprop, 4.41, 1e-9), Vprop)
check(T, "例1 V_prop は一般式 ΣW²(1−f)S²/n_h と一致", close((W ** 2 * (1 - 0.1) * Sh ** 2 / (n * W)).sum(), Vprop, 1e-12), Vprop)
Ybar = (W * Yh).sum()
S2pop = (((Nh - 1) * Sh ** 2).sum() + (Nh * (Yh - Ybar) ** 2).sum()) / 999
check(T, "例1 Ȳ=59, S²≈617.52", close(Ybar, 59, 1e-12) and close(S2pop, 617.52, 5e-3), S2pop)
Vsrs = fpc / n * S2pop
check(T, "例1 V_SRS≈5.558", close(Vsrs, 5.558, 5e-4), Vsrs)
check(T, "例1 差 1.148", close(Vsrs - Vprop, 1.148, 5e-4), Vsrs - Vprop)
check(T, "例1 近似 0.009×129=1.161", close((W * (Yh - 59) ** 2).sum(), 129, 1e-9) and close(0.009 * 129, 1.161, 5e-4), 0.009 * 129)
check(T, "例2 ȳst=59.5 と単純平均 59.5 が一致",
      close((W * np.array([51, 58, 83])).sum(), 59.5, 1e-12) and close((50 * 51 + 30 * 58 + 20 * 83) / 100, 59.5, 1e-12), 59.5)
nd = 300 * np.array([1234, 2345, 1421]) / 5000
check(T, "例3 n_h=(74.04,140.70,85.26)、四捨五入(74,141,85)の合計300",
      allclose(nd, [74.04, 140.70, 85.26], 5e-3) and list(np.round(nd).astype(int)) == [74, 141, 85] and np.round(nd).sum() == 300, nd)
Sbar = (W * Sh).sum()
Vopt = Sbar ** 2 / n - (W * Sh ** 2).sum() / 1000
check(T, "比例とネイマンの差=(1/n)ΣW(S_h−S̄)²", close(Vprop - Vopt, (W * (Sh - Sbar) ** 2).sum() / n, 1e-12), Vprop - Vopt)
ch = charts(T)[0]["params"]
check(T, "bar-data [5.558,4.41,3.12] とラベルの配分 (26.3,31.6,42.1)",
      allclose([Vsrs, Vprop, Vopt], ch["values"], 5e-4) and allclose(n * W * Sh / (W * Sh).sum(), [26.3, 31.6, 42.1], 5e-2), ch["values"])
# 自己加重: 比例配分で ȳst=ȳ（任意の標本で）
Wsym = sp.symbols("W1:4", positive=True)
nsym = sp.Symbol("n", positive=True)
ybs = sp.symbols("yb1:4")
lhs = sum(w * y for w, y in zip(Wsym, ybs))
rhs = sum(nsym * w * y for w, y in zip(Wsym, ybs)) / sum(nsym * w for w in Wsym)
check(T, "自己加重 ΣW_hȳ_h=Σn_hȳ_h/n（ΣW=1 のとき, sympy）",
      sp.simplify((lhs - rhs).subs(Wsym[2], 1 - Wsym[0] - Wsym[1])) == 0, "恒等式")
# 列挙: 比例配分でも小母集団で公式が成り立つ（stratified 例1 の (2,2) は比例配分）
check(T, "比例配分の公式 (1−f)/n·ΣW S_h²（層化例1 の (2,2)）= 列挙値 1.3333",
      close((1 - 4 / 12) / 4 * (0.5 * 2 + 0.5 * 14), V22, 1e-12), V22)

# ================= neyman-allocation =================
T = "neyman-allocation"
nhN = n * Nh * Sh / (Nh * Sh).sum()
check(T, "例1 N_hS_h=(5000,6000,8000), n_h=(26.32,31.58,42.11)",
      list(Nh * Sh) == [5000, 6000, 8000] and allclose(nhN, [26.32, 31.58, 42.11], 5e-3), nhN)
check(T, "例1 S̄=19, ΣWS²=490, V_opt=3.61−0.49=3.12", close(Sbar, 19, 1e-12) and close(Vopt, 3.12, 1e-12), Vopt)
Vgen = lambda nh_, Nh_, Sh_: float((((Nh_ / Nh_.sum()) ** 2) * (1 - nh_ / Nh_) * Sh_ ** 2 / nh_).sum())
check(T, "例1 V_opt を一般式に n_h を代入しても 3.12", close(Vgen(nhN, Nh, Sh), 3.12, 1e-10), Vgen(nhN, Nh, Sh))
check(T, "例1 差 1.29=129/100", close(Vprop - Vopt, 1.29, 1e-12), Vprop - Vopt)
check(T, "例1 整数 (26,32,42) で 3.1203", close(Vgen(np.array([26, 32, 42]), Nh, Sh), 3.1203), Vgen(np.array([26, 32, 42]), Nh, Sh))
N2 = np.array([600, 400])
S2h = np.array([10, 30])
n1 = 100 * 6000 / 18000
check(T, "例2 n1=33.3, n2=66.7", close(n1, 33.3, 5e-2) and close(100 - n1, 66.7, 5e-2), n1)
v_opt2 = Vgen(np.array([n1, 100 - n1]), N2, S2h)
v_pr2 = Vgen(np.array([60, 40]), N2, S2h)
check(T, "例2 ネイマン 2.82、比例 3.78", close(v_opt2, 2.82, 5e-3) and close(v_pr2, 3.78, 5e-3), (v_opt2, v_pr2))
ch = charts(T)[0]["params"]
lv = [Vgen(np.array([a, 100 - a]), N2, S2h) for a in ch["x"]]
check(T, "line-data 17点 V(n1) が本文の2層母集団の式と一致（3桁丸め）", allclose(lv, ch["series"][0]["y"], 5e-4), np.round(lv, 3))
grid = np.linspace(1, 99, 98001)
vg = 36 / grid + 144 / (100 - grid) - 0.42
check(T, "例2 数値最小化で最小点 n1=33.33, 最小値 2.82", close(grid[vg.argmin()], 100 / 3, 1e-3) and close(vg.min(), 2.82, 1e-6), (grid[vg.argmin()], vg.min()))
c = np.array([1, 1, 4])
k = 270 / (c * Nh * Sh / np.sqrt(c)).sum()
nc = k * Nh * Sh / np.sqrt(c)
check(T, "例3 N_hS_h/√c=(5000,6000,4000)、n_h=(50,60,40)、Σc n=270",
      allclose(Nh * Sh / np.sqrt(c), [5000, 6000, 4000], 1e-9) and allclose(nc, [50, 60, 40], 1e-9) and close((c * nc).sum(), 270, 1e-9), nc)
check(T, "例3 分散 2.21", close(Vgen(nc, Nh, Sh), 2.21, 1e-10), Vgen(nc, Nh, Sh))
# 費用制約下で数値最適化して同じ解になるか
from scipy.optimize import minimize
cons = {"type": "eq", "fun": lambda z: (c * z).sum() - 270}
res = minimize(lambda z: Vgen(z, Nh, Sh), x0=np.array([90, 60, 30.0]), constraints=[cons], bounds=[(1, 500), (1, 300), (1, 67)], method="SLSQP")
check(T, "例3 SLSQP で費用制約下の最小化も (50,60,40)（数値最適化の許容 0.05）", allclose(res.x, [50, 60, 40], 0.05), res.x)
# ラグランジュ乗数法（sympy）
L = 3
Wv = sp.symbols("W1:4", positive=True)
Sv = sp.symbols("S1:4", positive=True)
nv = sp.symbols("n1:4", positive=True)
lam = sp.Symbol("lambda", positive=True)
Nsym = sp.Symbol("N", positive=True)
Vexpr = sum(Wv[h] ** 2 * (1 - nv[h] / (Wv[h] * Nsym)) * Sv[h] ** 2 / nv[h] for h in range(L))
check(T, "分散の分解 ΣW²(1−n_h/N_h)S²/n_h=ΣW²S²/n_h−(1/N)ΣWS²（sympy）",
      sp.simplify(Vexpr - (sum(Wv[h] ** 2 * Sv[h] ** 2 / nv[h] for h in range(L)) - sum(Wv[h] * Sv[h] ** 2 for h in range(L)) / Nsym)) == 0, "恒等式")
Lag = sum(Wv[h] ** 2 * Sv[h] ** 2 / nv[h] for h in range(L)) + lam * (sum(nv) - nsym)
sol_h = [sp.solve(sp.diff(Lag, nv[h]), nv[h]) for h in range(L)]
check(T, "∂L/∂n_h=0 ⇒ n_h=W_hS_h/√λ（sympy）", all(len(s) == 1 and sp.simplify(s[0] - Wv[h] * Sv[h] / sp.sqrt(lam)) == 0 for h, s in enumerate(sol_h)), sol_h)
lam_sol = sp.solve(sp.Eq(sum(s[0] for s in sol_h), nsym), lam)
nh_sol = [sp.simplify(s[0].subs(lam, lam_sol[0])) for s in sol_h]
SumWS = sum(Wv[h] * Sv[h] for h in range(L))
check(T, "制約から √λ=ΣW_kS_k/n、n_h=nW_hS_h/ΣW_kS_k（sympy）",
      len(lam_sol) == 1 and sp.simplify(sp.sqrt(lam_sol[0]) - SumWS / nsym) == 0 and all(sp.simplify(nh_sol[h] - nsym * Wv[h] * Sv[h] / SumWS) == 0 for h in range(L)), nh_sol)
first = sum(Wv[h] ** 2 * Sv[h] ** 2 / nh_sol[h] for h in range(L))
check(T, "第1項に代入すると (ΣW_hS_h)²/n（sympy）", sp.simplify(first - SumWS ** 2 / nsym) == 0, sp.factor(first))
H = sp.hessian(sum(Wv[h] ** 2 * Sv[h] ** 2 / nv[h] for h in range(L)), nv)
check(T, "目的関数のヘッセ行列は対角で正（凸, sympy）", all(sp.simplify(H[i, j]) == 0 for i in range(L) for j in range(L) if i != j) and all(H[i, i].is_positive for i in range(L)), [H[i, i] for i in range(L)])
cv = sp.symbols("c1:4", positive=True)
Cs = sp.Symbol("C", positive=True)
Lag2 = sum(Wv[h] ** 2 * Sv[h] ** 2 / nv[h] for h in range(L)) + lam * (sum(cv[h] * nv[h] for h in range(L)) - Cs)
sol2 = [sp.solve(sp.diff(Lag2, nv[h]), nv[h]) for h in range(L)]
check(T, "費用つき: n_h=W_hS_h/√(λc_h)（sympy）", all(sp.simplify(s[0] - Wv[h] * Sv[h] / sp.sqrt(lam * cv[h])) == 0 for h, s in enumerate(sol2)), sol2)
check(T, "∂V/∂n_h=−W_h²S_h²/n_h²（sympy）", all(sp.simplify(sp.diff(Vexpr, nv[h]) + Wv[h] ** 2 * Sv[h] ** 2 / nv[h] ** 2) == 0 for h in range(L)), "偏微分")
Sb_ = sp.Symbol("Sbar")
w1, w2 = sp.symbols("w1 w2", positive=True)
s1, s2_, s3 = Sv
ww = [w1, w2, 1 - w1 - w2]
lhsd = sum(ww[h] * Sv[h] ** 2 for h in range(3)) - sum(ww[h] * Sv[h] for h in range(3)) ** 2
rhsd = sum(ww[h] * (Sv[h] - sum(ww[k] * Sv[k] for k in range(3))) ** 2 for h in range(3))
check(T, "ΣWS²−S̄²=ΣW(S_h−S̄)²（ΣW=1, sympy）", sp.expand(lhsd - rhsd) == 0, "恒等式")
# コーシー・シュワルツ: ランダムな配分は V_opt 以上
ok = True
for _ in range(2000):
    z = rng.dirichlet(np.ones(3)) * 100
    ok &= Vgen(z, Nh * 10, Sh) >= Vgen(n * Nh * Sh / (Nh * Sh).sum(), Nh * 10, Sh) - 1e-12
check(T, "ランダムな配分2000通りはどれもネイマン配分以上の分散", ok, "コーシー・シュワルツ")
# 層化の小母集団で列挙: (1,3) はネイマンに近い
check(T, "層化例1でネイマン配分 n1=4√2/(√2+√14)≈1.097 → (1,3) が分散最小の整数配分",
      close(4 * math.sqrt(2) / (math.sqrt(2) + math.sqrt(14)), 1.097, 5e-4) and V13 < V22 and V13 < strat_enum([h1, h2], [3, 1])[2],
      4 * math.sqrt(2) / (math.sqrt(2) + math.sqrt(14)))

# ================= cluster-sampling =================
T = "cluster-sampling"


def cluster_stats(cl):
    allv = np.concatenate(cl)
    A, M = len(cl), len(cl[0])
    Nn = A * M
    Yb = allv.mean()
    S2a = S2(allv)
    cm = np.array([np.mean(c) for c in cl])
    Sb2 = float(((cm - Yb) ** 2).sum() / (A - 1))
    num = 0.0
    for c in cl:
        for j, k in combinations(range(M), 2):
            num += (c[j] - Yb) * (c[k] - Yb)
    rho = 2 * num / ((M - 1) * (Nn - 1) * S2a)
    return dict(A=A, M=M, N=Nn, Yb=Yb, S2=S2a, cm=cm, Sb2=Sb2, rho=rho)


def cluster_enum(cl, a):
    ests = [np.mean([np.mean(cl[i]) for i in s]) for s in combinations(range(len(cl)), a)]
    return dist_moments(ests)


cl1 = [np.array(v, float) for v in ([2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13])]
st = cluster_stats(cl1)
check(T, "例1 集落平均 3,6,9,12、Ȳ=7.5、S²=13、S_b²=15",
      allclose(st["cm"], [3, 6, 9, 12], 1e-12) and close(st["Yb"], 7.5, 1e-12) and close(st["S2"], 13, 1e-12) and close(st["Sb2"], 15, 1e-12), st["Sb2"])
E, V = cluster_enum(cl1, 2)
check(T, "例1 公式 (1−2/4)·15/2=3.75 と 6通り列挙", close(0.5 * 15 / 2, 3.75, 1e-12) and close(V, 3.75, 1e-12) and close(E, 7.5, 1e-12), V)
Vs6 = 0.5 * 13 / 6
srs6 = dist_moments([np.mean(s) for s in combinations(np.concatenate(cl1), 6)])[1]
check(T, "例1 同じ6人の SRS V≈1.0833（列挙とも一致）、比 約3.46", close(Vs6, 1.0833) and close(srs6, Vs6, 1e-10) and close(V / Vs6, 3.46, 5e-3), V / Vs6)
check(T, "例1 ρ≈0.9161", close(st["rho"], 0.9161), st["rho"])
check(T, "例1 (11/9){1+2ρ}=3.4615 が分散比と一致", close(11 / 9 * (1 + 2 * st["rho"]), 3.4615) and close(11 / 9 * (1 + 2 * st["rho"]), V / Vs6, 1e-10),
      11 / 9 * (1 + 2 * st["rho"]))
Vrho = (1 - 2 / 4) / 2 * (st["N"] - 1) / (st["M"] ** 2 * (st["A"] - 1)) * st["S2"] * (1 + (st["M"] - 1) * st["rho"])
check(T, "例1 ρ を使った分散式が 3.75 を再現", close(Vrho, 3.75, 1e-10), Vrho)
cl2 = [np.array(v, float) for v in ([2, 7, 12], [3, 8, 11], [4, 6, 13], [5, 9, 10])]
st2 = cluster_stats(cl2)
E2, V2 = cluster_enum(cl2, 2)
check(T, "例2 集落平均 7, 7.333, 7.667, 8", allclose(st2["cm"], [7, 7.333, 7.667, 8], 5e-4), st2["cm"])
check(T, "例2 S_b²≈0.1852, V≈0.0463（列挙とも一致）", close(st2["Sb2"], 0.1852) and close(V2, 0.0463) and close(0.25 * st2["Sb2"], V2, 1e-12), (st2["Sb2"], V2))
check(T, "例2 SRS の約 0.043 倍", close(V2 / Vs6, 0.043, 5e-4), V2 / Vs6)
check(T, "例2 ρ≈−0.4825、(11/9){1+2ρ} も比と一致", close(st2["rho"], -0.4825) and close(11 / 9 * (1 + 2 * st2["rho"]), V2 / Vs6, 1e-10), st2["rho"])
check(T, "例3 deff=1+29×0.05=2.45、1200/2.45≈490", close(1 + 29 * 0.05, 2.45, 1e-12) and abs(1200 / 2.45 - 490) < 0.5, 1200 / 2.45)
ch = charts(T)[0]["params"]
ok = all(allclose([1 + (M - 1) * r for r in ch["x"]], s["y"], 5e-4) for M, s in zip([5, 10, 20], ch["series"]))
check(T, "line-data 1+(M−1)ρ（M=5,10,20）", ok, "7点×3本")
# 比の一般式: V_cl/V_SRS=(N−1)/(M(A−1))·{1+(M−1)ρ}（ランダムな母集団で）
ok = True
for _ in range(200):
    A_, M_ = rng.integers(3, 7), rng.integers(2, 5)
    cl = [rng.normal(rng.normal(0, 2), 1, M_) for _ in range(A_)]
    s_ = cluster_stats(cl)
    a_ = 2
    Vc = (1 - a_ / A_) * s_["Sb2"] / a_
    Vsr = (1 - a_ / A_) * s_["S2"] / (a_ * M_)
    ok &= close(Vc / Vsr, (s_["N"] - 1) / (M_ * (A_ - 1)) * (1 + (M_ - 1) * s_["rho"]), 1e-9)
check(T, "ランダム母集団200個で V_cl/V_SRS=(N−1)/(M(A−1))·{1+(M−1)ρ}", ok, "恒等式")

# ================= two-stage-sampling =================
T = "two-stage-sampling"
cl = [np.array(v, float) for v in ([2, 4, 6, 8], [5, 7, 9, 11], [3, 6, 9, 12], [10, 11, 13, 14])]
cm = np.array([c.mean() for c in cl])
Si2 = np.array([S2(c) for c in cl])
check(T, "例1 表の Ȳ_i=(5,8,7.5,12), S_i²=(6.667,6.667,15,3.333)", allclose(cm, [5, 8, 7.5, 12], 1e-12) and allclose(Si2, [6.667, 6.667, 15, 3.333], 5e-4), Si2)
Yb = np.concatenate(cl).mean()
Sb2 = float(((cm - Yb) ** 2).sum() / 3)
Sw2 = Si2.mean()
check(T, "例1 Ȳ=8.125, S_b²≈8.3958, S̄_w²=7.9167", close(Yb, 8.125, 1e-12) and close(Sb2, 8.3958) and close(Sw2, 7.9167), (Sb2, Sw2))


def ts_formula(a, m, A=4, M=4):
    return (1 - a / A) * Sb2 / a + (1 - m / M) * Sw2 / (a * m)


def ts_enum(a, m, A=4, M=4):
    ests, Vh, probs = [], [], []
    for s1 in combinations(range(A), a):
        subs = [list(combinations(cl[i], m)) for i in s1]
        for pick in product(*subs):
            ybi = [np.mean(p) for p in pick]
            ests.append(np.mean(ybi))
            if a >= 2 and m >= 2:
                sb2 = S2(ybi)
                si2 = [S2(p) for p in pick]
                Vh.append((1 - a / A) * sb2 / a + a / A * (1 - m / M) / (a * m) * np.mean(si2))
    Eh, Vv = dist_moments(ests)
    return len(ests), Eh, Vv, (np.mean(Vh) if Vh else None)


t1 = ts_formula(2, 2)
check(T, "例1 2.0990+0.9896=3.0885", close((1 - 0.5) * Sb2 / 2, 2.0990) and close(0.5 * Sw2 / 4, 0.9896) and close(t1, 3.0885), t1)
cnt216, Et, Vt, EVh = ts_enum(2, 2)
check(T, "例1 216通り列挙で期待値8.125、分散3.0885", cnt216 == 216 and close(Et, 8.125, 1e-12) and close(Vt, t1, 1e-10), (Et, Vt))
check(T, "例1 V̂ の期待値 3.0885（不偏）", close(EVh, t1, 1e-10), EVh)
v41 = ts_formula(4, 1)
_, _, Vt41, _ = ts_enum(4, 1)
check(T, "例2 4集落×1人 1.4844（列挙とも一致）", close(v41, 1.4844) and close(Vt41, v41, 1e-10), v41)
allv = np.concatenate(cl)
vsrs4 = (1 - 4 / 16) * S2(allv) / 4
srs4 = dist_moments([np.mean(s) for s in combinations(allv, 4)])[1]
check(T, "例2 SRS S²=13.05, V≈2.4469（列挙とも一致）", close(S2(allv), 13.05, 5e-3) and close(vsrs4, 2.4469) and close(srs4, vsrs4, 1e-10), (S2(allv), vsrs4))
v14 = ts_formula(1, 4)
_, _, Vt14, _ = ts_enum(1, 4)
check(T, "例2 1集落全員 (3/4)×8.3958≈6.2969（列挙とも一致）", close(v14, 6.2969) and close(Vt14, v14, 1e-10), v14)
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [1.484,2.447,3.089,6.297]", allclose([v41, vsrs4, t1, v14], ch, 5e-4), ch)
# m=M で集落抽出、a=A で層化（比例配分）に一致
ok = close(ts_formula(2, 4), (1 - 2 / 4) * Sb2 / 2, 1e-12)
strat_v = strat_formula([list(c) for c in cl], [2, 2, 2, 2])
ok &= close(ts_formula(4, 2), strat_v, 1e-12)
check(T, "m=M で集落抽出、a=A で層化（比例配分）の分散に一致", ok, (ts_formula(4, 2), strat_v))
# 全分散の公式の第2項: E[V(·|s1)] の計算
second = np.mean([np.sum([(1 - 2 / 4) * Si2[i] / 2 for i in s]) / 4 for s in combinations(range(4), 2)])
check(T, "第2項 E[(1/a²)Σ(1−m/M)S_i²/m]=(1−m/M)S̄_w²/(am)", close(second, 0.5 * Sw2 / 4, 1e-12), second)
# 他の (a,m) でも列挙と公式が一致
ok = True
for a, m in [(3, 2), (2, 3), (3, 3)]:
    _, _, Vx, EVx = ts_enum(a, m)
    ok &= close(Vx, ts_formula(a, m), 1e-10) and close(EVx, ts_formula(a, m), 1e-10)
check(T, "(a,m)=(3,2),(2,3),(3,3) でも列挙の分散と V̂ の期待値が公式に一致", ok, "3通り")

# ================= systematic-sampling =================
T = "systematic-sampling"


def sys_enum(y, k):
    y = np.asarray(y, float)
    ms = [y[r::k].mean() for r in range(k)]
    return ms, *dist_moments(ms)


y1 = np.arange(1, 13)
ms, E, V = sys_enum(y1, 3)
check(T, "例1 標本平均 5.5, 6.5, 7.5、期待値6.5", allclose(ms, [5.5, 6.5, 7.5], 1e-12) and close(E, 6.5, 1e-12), ms)
check(T, "例1 V=0.6667", close(V, 0.6667), V)
vs = (1 - 4 / 12) * S2(y1) / 4
check(T, "例1 SRS 2.1667、約3倍", close(vs, 2.1667) and close(vs / V, 3.25, 1e-9), vs / V)
y2 = np.array([2, 5, 8] * 4, float)
ms2, E2, V2 = sys_enum(y2, 3)
check(T, "例2 標本平均 2,5,8、Ȳ=5、V=6", allclose(ms2, [2, 5, 8], 1e-12) and close(E2, 5, 1e-12) and close(V2, 6, 1e-12), V2)
vs2 = (1 - 4 / 12) * S2(y2) / 4
check(T, "例2 S²≈6.5455、SRS ≈1.0909、5倍以上", close(S2(y2), 6.5455) and close(vs2, 1.0909) and V2 / vs2 > 5, V2 / vs2)
ch = charts(T)[0]["params"]
check(T, "line-data 2系列が例1・例2の並び", ch["series"][0]["y"] == list(range(1, 13)) and ch["series"][1]["y"] == [2, 5, 8] * 4 and ch["x"] == list(range(1, 13)), "一致")
# 包含確率 1/k、別系列で π_ij=0
pis = np.zeros(12)
pij = np.zeros((12, 12))
for r in range(3):
    idx = list(range(r, 12, 3))
    pis[idx] += 1 / 3
    for i in idx:
        for j in idx:
            pij[i, j] += 1 / 3
check(T, "π_i=1/k、同じ系列 π_ij=1/k、別系列 0", allclose(pis, 1 / 3, 1e-12) and close(pij[0, 3], 1 / 3, 1e-12) and pij[0, 1] == 0, (pis[0], pij[0, 3], pij[0, 1]))

# ================= ratio-estimation / regression-estimation 共通 =================


def est_enum(x, y, n):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Xb, Yb = x.mean(), y.mean()
    out = {"mean": [], "ratio": [], "lr": []}
    for s in combinations(range(len(x)), n):
        xs, ys = x[list(s)], y[list(s)]
        out["mean"].append(ys.mean())
        out["ratio"].append(ys.mean() / xs.mean() * Xb)
        b = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
        out["lr"].append(ys.mean() + b * (Xb - xs.mean()))
    res = {}
    for kk, v in out.items():
        v = np.array(v)
        res[kk] = dict(n=len(v), bias=v.mean() - Yb, mse=((v - Yb) ** 2).mean(), var=v.var())
    return res


T = "ratio-estimation"
x = np.array([2, 3, 4, 5, 6, 8], float)
y = np.array([5, 7, 8, 11, 12, 17], float)
Xb, Yb = x.mean(), y.mean()
R = Yb / Xb
Sx2, Sy2 = S2(x), S2(y)
Sxy = float(((x - Xb) * (y - Yb)).sum() / 5)
r = Sxy / math.sqrt(Sx2 * Sy2)
check(T, "例1 X̄=4.6667, Ȳ=10, R=2.1429", close(Xb, 4.6667) and close(Yb, 10, 1e-12) and close(R, 2.1429), R)
check(T, "例1 S_y²=18.4, S_x²=4.6667, S_xy=9.2, 相関0.9928",
      close(Sy2, 18.4, 1e-10) and close(Sx2, 4.6667) and close(Sxy, 9.2, 1e-10) and close(r, 0.9928), r)
Sd2 = float(((y - R * x) ** 2).sum() / 5)
check(T, "例1 S_d²=0.4（=S_y²−2RS_xy+R²S_x²）", close(Sd2, 0.4, 1e-10) and close(Sy2 - 2 * R * Sxy + R ** 2 * Sx2, Sd2, 1e-10), Sd2)
check(T, "例1 近似分散 (0.5/3)×0.4=0.0667", close(0.5 / 3 * Sd2, 0.0667), 0.5 / 3 * Sd2)
er = est_enum(x, y, 3)
check(T, "例1 20標本列挙で比推定の偏り 0.0314", er["ratio"]["n"] == 20 and close(er["ratio"]["bias"], 0.0314), er["ratio"]["bias"])
bias_apx = 0.5 / 3 / Xb * (R * Sx2 - Sxy)
check(T, "例1 偏りの近似式 0.0286", close(bias_apx, 0.0286), bias_apx)
check(T, "例1 比推定 MSE 0.0785", close(er["ratio"]["mse"], 0.0785), er["ratio"]["mse"])
check(T, "例1 標本平均の分散 (0.5/3)×18.4=3.0667（列挙とも一致）、約1/39",
      close(0.5 / 3 * 18.4, 3.0667) and close(er["mean"]["mse"], 0.5 / 3 * 18.4, 1e-10) and round(er["mean"]["mse"] / er["ratio"]["mse"]) == 39,
      er["mean"]["mse"] / er["ratio"]["mse"])
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [3.067,0.0785,0.1131]（MSE、列挙）",
      close(er["mean"]["mse"], ch[0], 5e-4) and close(er["ratio"]["mse"], ch[1], 5e-5) and close(er["lr"]["mse"], ch[2], 5e-5), [er[k]["mse"] for k in ("mean", "ratio", "lr")])
# 例2
idx = [1, 3, 5]
xs, ys_ = x[idx], y[idx]
Rh = ys_.mean() / xs.mean()
check(T, "例2 x̄=5.3333, ȳ=11.6667, R̂=2.1875, ȳ_R=10.2083",
      close(xs.mean(), 5.3333) and close(ys_.mean(), 11.6667) and close(Rh, 2.1875, 1e-12) and close(Rh * Xb, 10.2083), Rh * Xb)
res_ = ys_ - Rh * xs
check(T, "例2 残差 (0.4375,0.0625,−0.5)、平方和0.4453", allclose(res_, [0.4375, 0.0625, -0.5], 1e-12) and close((res_ ** 2).sum(), 0.4453), (res_ ** 2).sum())
check(T, "例2 分散推定 (0.5/3)(0.4453/2)≈0.0371", close(0.5 / 3 * (res_ ** 2).sum() / 2, 0.0371), 0.5 / 3 * (res_ ** 2).sum() / 2)
# 例3
x3 = np.arange(1, 13, dtype=float)
y3 = np.array([22, 21, 25, 24, 27, 26, 30, 29, 31, 33, 32, 36], float)
r3 = np.corrcoef(x3, y3)[0, 1]
b3 = np.polyfit(x3, y3, 1)
check(T, "例3 相関0.968、切片 約20", close(r3, 0.968, 5e-4) and abs(b3[1] - 20) < 0.5, (r3, b3))
Cx, Cy = math.sqrt(S2(x3)) / x3.mean(), math.sqrt(S2(y3)) / y3.mean()
check(T, "例3 X̄=6.5, Ȳ=28, C_x=0.5547, C_y=0.1647, 右辺≈1.68",
      close(x3.mean(), 6.5, 1e-12) and close(y3.mean(), 28, 1e-12) and close(Cx, 0.5547) and close(Cy, 0.1647) and close(0.5 * Cx / Cy, 1.68, 5e-3), 0.5 * Cx / Cy)
er3 = est_enum(x3, y3, 6)
check(T, "例3 924標本列挙で比推定 MSE 12.893、標本平均 1.773、7倍以上、回帰 0.147",
      er3["ratio"]["n"] == 924 and close(er3["ratio"]["mse"], 12.893, 5e-4) and close(er3["mean"]["mse"], 1.773, 5e-4)
      and er3["ratio"]["mse"] / er3["mean"]["mse"] > 7 and close(er3["lr"]["mse"], 0.147, 5e-4),
      [er3[k]["mse"] for k in ("mean", "ratio", "lr")])
# 近似分散の導出: ȳ_R−Ȳ=(X̄/x̄)(ȳ−Rx̄)、得する条件
xb_, yb_, Xb_ = sp.symbols("xbar ybar Xbar", positive=True)
Rs = sp.Symbol("R", positive=True)
check(T, "ȳ_R−Ȳ=(X̄/x̄)(ȳ−Rx̄)（Ȳ=RX̄, sympy）", sp.simplify(yb_ / xb_ * Xb_ - Rs * Xb_ - Xb_ / xb_ * (yb_ - Rs * xb_)) == 0, "恒等式")
rho_, Sx_, Sy_, Xm, Ym = sp.symbols("rho S_x S_y Xbar Ybar", positive=True)
Sd = Sy_ ** 2 - 2 * (Ym / Xm) * rho_ * Sx_ * Sy_ + (Ym / Xm) ** 2 * Sx_ ** 2
cond = sp.simplify((Sy_ ** 2 - Sd) / (2 * (Ym / Xm) * Sx_ * Sy_))
check(T, "S_d²<S_y² ⇔ ρ>(1/2)C_x/C_y（sympy）", sp.simplify(cond - (rho_ - sp.Rational(1, 2) * (Sx_ / Xm) / (Sy_ / Ym))) == 0, cond)
# 比推定の近似が条件と整合: 例1 は条件を満たす、例3 は満たさない
Cx1, Cy1 = math.sqrt(Sx2) / Xb, math.sqrt(Sy2) / Yb
check(T, "例1 は ρ>C_x/(2C_y) を満たし MSE 改善、例3 は満たさず悪化",
      r > 0.5 * Cx1 / Cy1 and er["ratio"]["mse"] < er["mean"]["mse"] and r3 < 0.5 * Cx / Cy and er3["ratio"]["mse"] > er3["mean"]["mse"], (0.5 * Cx1 / Cy1, 0.5 * Cx / Cy))
# モデルの見方: V(ε)∝x の WLS で原点を通る回帰の BLUE は Σy/Σx
xs_, ys2 = sp.symbols("x1:4", positive=True), sp.symbols("y1:4")
beta = sp.Symbol("beta")
Q = sum((ys2[i] - beta * xs_[i]) ** 2 / xs_[i] for i in range(3))
bsol = sp.solve(sp.diff(Q, beta), beta)[0]
check(T, "y=βx+ε, V(ε)∝x の WLS 解が Σy/Σx=R̂（sympy）", sp.simplify(bsol - sum(ys2) / sum(xs_)) == 0, bsol)
# 偏り O(1/n) のシミュレーションで近似分散を確認
Np = 2000
xp = rng.gamma(4, 2, Np)
yp = 3 * xp + rng.normal(0, 1, Np) * np.sqrt(xp)
Rp = yp.mean() / xp.mean()
Sdp = S2(yp - Rp * xp)
reps = 40000
nn_ = 100
est = np.empty(reps)
for t in range(reps):
    s = rng.choice(Np, nn_, replace=False)
    est[t] = yp[s].mean() / xp[s].mean() * xp.mean()
apx = (1 - nn_ / Np) / nn_ * Sdp
check(T, "乱数 N=2000,n=100 で比推定の分散≈近似分散（4万回、相対誤差 3%、SE 約0.7%）", abs(est.var() / apx - 1) < 0.03, (est.var(), apx))

T = "regression-estimation"
Sy2_3 = S2(y3)
check(T, "例1 X̄=6.5, Ȳ=28, S_y²=21.273, ρ=0.9676", close(Sy2_3, 21.273, 5e-4) and close(r3, 0.9676), (Sy2_3, r3))
vapx = 0.5 / 6 * Sy2_3 * (1 - r3 ** 2)
check(T, "例1 近似分散≈0.1130", close(vapx, 0.1130), vapx)
check(T, "例1 標本平均の分散 1.7727（列挙とも一致）", close(0.5 / 6 * Sy2_3, 1.7727) and close(er3["mean"]["mse"], 0.5 / 6 * Sy2_3, 1e-10), 0.5 / 6 * Sy2_3)
check(T, "例1 924標本で回帰推定 MSE 0.1474、偏り −0.0152", close(er3["lr"]["mse"], 0.1474) and close(er3["lr"]["bias"], -0.0152), (er3["lr"]["mse"], er3["lr"]["bias"]))
check(T, "例1 標本平均の約1/12", round(er3["mean"]["mse"] / er3["lr"]["mse"]) == 12, er3["mean"]["mse"] / er3["lr"]["mse"])
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [1.773,12.893,0.147]", allclose([er3[k]["mse"] for k in ("mean", "ratio", "lr")], ch, 5e-4), ch)
idx = [0, 1, 4, 6, 8, 10]
xs, ysr = x3[idx], y3[idx]
check(T, "例2 選ばれた y=22,21,27,30,31,32", list(ysr) == [22, 21, 27, 30, 31, 32], ysr)
bb = ((xs - xs.mean()) * (ysr - ysr.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
ylr = ysr.mean() + bb * (6.5 - xs.mean())
check(T, "例2 x̄=5.8333, ȳ=27.1667, b=1.1605, ȳ_lr=27.9403",
      close(xs.mean(), 5.8333) and close(ysr.mean(), 27.1667) and close(bb, 1.1605) and close(ylr, 27.9403), (bb, ylr))
check(T, "例2 比推定 30.2714", close(ysr.mean() / xs.mean() * 6.5, 30.2714), ysr.mean() / xs.mean() * 6.5)
resid = ysr - ysr.mean() - bb * (xs - xs.mean())
vh = 0.5 / 6 * (resid ** 2).sum() / 4
check(T, "例2 分散の推定値 0.1532", close(vh, 0.1532), vh)
b0, Sxs, Sys, Sxys = sp.symbols("b0 S_x2 S_y2 S_xy", positive=True)
q = Sys - 2 * b0 * Sxys + b0 ** 2 * Sxs
bmin = sp.solve(sp.diff(q, b0), b0)
check(T, "b0 の2次式の最小点 B=S_xy/S_x²、最小値 S_y²−S_xy²/S_x²=S_y²(1−ρ²)（sympy）",
      len(bmin) == 1 and sp.simplify(bmin[0] - Sxys / Sxs) == 0 and sp.simplify(q.subs(b0, bmin[0]) - (Sys - Sxys ** 2 / Sxs)) == 0
      and sp.simplify((Sys - Sxys ** 2 / Sxs) - Sys * (1 - Sxys ** 2 / (Sxs * Sys))) == 0, bmin)
# b0 固定の推定量は不偏で分散が公式どおり（例1の母集団で列挙）
ok = True
for b0v in [0.0, 1.0, R]:
    v_ = [y3[list(s)].mean() + b0v * (6.5 - x3[list(s)].mean()) for s in combinations(range(12), 6)]
    m_, vv_ = dist_moments(v_)
    ok &= close(m_, 28, 1e-10) and close(vv_, 0.5 / 6 * (Sy2_3 - 2 * b0v * float(np.cov(x3, y3)[0, 1]) + b0v ** 2 * S2(x3)), 1e-10)
check(T, "b0 固定（0,1,R）の推定量は列挙で不偏、分散が (1−f)/n(S_y²−2b0S_xy+b0²S_x²)", ok, "3通り")
check(T, "相関 0.5 で (1−ρ²)=0.75（25%減）", close(1 - 0.5 ** 2, 0.75, 1e-12), 0.75)
# 乱数: 回帰推定の近似分散
yp2 = 50 + 2 * xp + rng.normal(0, 3, Np)
rho_p = np.corrcoef(xp, yp2)[0, 1]
est = np.empty(reps)
for t in range(reps):
    s = rng.choice(Np, nn_, replace=False)
    xs_s, ys_s = xp[s], yp2[s]
    bb_ = np.cov(xs_s, ys_s)[0, 1] / xs_s.var(ddof=1)
    est[t] = ys_s.mean() + bb_ * (xp.mean() - xs_s.mean())
apx = (1 - nn_ / Np) / nn_ * S2(yp2) * (1 - rho_p ** 2)
check(T, "乱数 N=2000,n=100 で回帰推定の分散≈(1−f)/n·S_y²(1−ρ²)（4万回、相対誤差 4%）", abs(est.var() / apx - 1) < 0.04, (est.var(), apx))

# ================= horvitz-thompson-estimator / inclusion-probability =================
T = "horvitz-thompson-estimator"
yv = np.array([6, 9, 11, 14], float)
design = {(0, 1): 0.05, (0, 2): 0.10, (0, 3): 0.15, (1, 2): 0.15, (1, 3): 0.20, (2, 3): 0.35}
pi = np.zeros(4)
pij = np.zeros((4, 4))
for s, p in design.items():
    for i in s:
        pi[i] += p
    pij[s[0], s[1]] += p
    pij[s[1], s[0]] += p
check(T, "例1 π=(0.3,0.4,0.6,0.7)", allclose(pi, [0.3, 0.4, 0.6, 0.7], 1e-12), pi)
ht = {s: sum(yv[i] / pi[i] for i in s) for s in design}
check(T, "例1 {1,2} で 20+22.5=42.5", close(ht[(0, 1)], 42.5, 1e-10), ht[(0, 1)])
check(T, "例1 6通りの推定値 42.5, 38.333, 40.0, 40.833, 42.5, 38.333", allclose(list(ht.values()), [42.5, 38.333, 40.0, 40.833, 42.5, 38.333], 5e-4), list(ht.values()))
Eht, Vht = dist_moments(list(ht.values()), list(design.values()))
check(T, "例1 期待値 40.000（不偏）", close(Eht, 40, 1e-10), Eht)
check(T, "例1 分散 2.9167（列挙）", close(Vht, 2.9167), Vht)
Vf = sum((1 - pi[i]) / pi[i] * yv[i] ** 2 for i in range(4)) + sum((pij[i, j] - pi[i] * pi[j]) / (pi[i] * pi[j]) * yv[i] * yv[j] for i in range(4) for j in range(4) if i != j)
Vsyg = 0.5 * sum((pi[i] * pi[j] - pij[i, j]) * (yv[i] / pi[i] - yv[j] / pi[j]) ** 2 for i in range(4) for j in range(4) if i != j)
check(T, "例1 HT 分散公式・SYG 形とも 2.9167", close(Vf, Vht, 1e-10) and close(Vsyg, Vht, 1e-10), (Vf, Vsyg))
Eyg = sum(p * (pi[s[0]] * pi[s[1]] - pij[s]) / pij[s] * (yv[s[0]] / pi[s[0]] - yv[s[1]] / pi[s[1]]) ** 2 for s, p in design.items())
check(T, "例1 V̂_YG の期待値 2.9167（不偏）", close(Eyg, Vht, 1e-10), Eyg)
check(T, "例2 S²=11.333、4²(1−2/4)·11.333/2=45.333", close(S2(yv), 11.333, 5e-4) and close(16 * 0.5 * S2(yv) / 2, 45.333, 5e-4), 16 * 0.5 * S2(yv) / 2)
srs_tot = dist_moments([4 * yv[list(s)].mean() for s in combinations(range(4), 2)])
check(T, "例2 SRS 列挙でも 45.333", close(srs_tot[1], 45.333, 5e-4), srs_tot)
check(T, "例2 y/π=(20,22.5,18.333,20)、分散 約1/15", allclose(yv / pi, [20, 22.5, 18.333, 20], 5e-4) and 15 <= 45.333 / Vht < 16, 45.333 / Vht)
E4y = sum(p * 4 * yv[list(s)].mean() for s, p in design.items())
check(T, "例2 このデザインで 4ȳ の期待値 43.6（偏る）", close(E4y, 43.6, 1e-10), E4y)
ch = charts(T)[0]["params"]
check(T, "bar-data の推定値とラベルの確率", allclose(ch["values"], list(ht.values()), 5e-4) and "0.05, 0.10, 0.15, 0.15, 0.20, 0.35" in ch["xlabel"], ch["values"])
# 一般の固定サイズデザインで HT 分散式=SYG 形（ランダム例）
ok = True
for _ in range(200):
    Nn = 5
    ss = list(combinations(range(Nn), 2))
    pr = rng.dirichlet(np.ones(len(ss)))
    yy = rng.normal(10, 3, Nn)
    pi_ = np.zeros(Nn)
    pj_ = np.zeros((Nn, Nn))
    for s, p in zip(ss, pr):
        for i in s:
            pi_[i] += p
        pj_[s[0], s[1]] += p
        pj_[s[1], s[0]] += p
    est = [sum(yy[i] / pi_[i] for i in s) for s in ss]
    m_, v_ = dist_moments(est, pr)
    vf = sum((1 - pi_[i]) / pi_[i] * yy[i] ** 2 for i in range(Nn)) + sum((pj_[i, j] - pi_[i] * pi_[j]) / (pi_[i] * pi_[j]) * yy[i] * yy[j] for i in range(Nn) for j in range(Nn) if i != j)
    vs_ = 0.5 * sum((pi_[i] * pi_[j] - pj_[i, j]) * (yy[i] / pi_[i] - yy[j] / pi_[j]) ** 2 for i in range(Nn) for j in range(Nn) if i != j)
    eyg = sum(p * (pi_[s[0]] * pi_[s[1]] - pj_[s]) / pj_[s] * (yy[s[0]] / pi_[s[0]] - yy[s[1]] / pi_[s[1]]) ** 2 for s, p in zip(ss, pr))
    ok &= close(m_, yy.sum(), 1e-9) and close(vf, v_, 1e-8) and close(vs_, v_, 1e-8) and close(eyg, v_, 1e-8)
check(T, "ランダムなデザイン200個(N=5,n=2)で不偏・HT分散式=SYG形=列挙・V̂_YG 不偏", ok, "200例")
# π_i ∝ y_i で分散0
yprop = 50 * pi
v0 = dist_moments([sum(yprop[i] / pi[i] for i in s) for s in design], list(design.values()))[1]
check(T, "y∝π（例1のデザインで y=50π）なら列挙の分散は 0", close(v0, 0, 1e-12), v0)
check(T, "SRS で HT=Nȳ（π=n/N）", close(sum(yv[[0, 2]] / 0.5), 4 * yv[[0, 2]].mean(), 1e-12), "一致")

T = "inclusion-probability"
check(T, "例1 π=(0.3,0.4,0.6,0.7), Σπ=2.0", allclose(pi, [0.3, 0.4, 0.6, 0.7], 1e-12) and close(pi.sum(), 2, 1e-12), pi.sum())
check(T, "例1 π_34=0.35、Σ_{j≠1}π_1j=0.3=(2−1)π_1", close(pij[2, 3], 0.35, 1e-12) and close(pij[0].sum(), pi[0], 1e-12), pij[0].sum())
check(T, "例1 Σ_{j≠i}π_ij=(n−1)π_i が全 i で成立", allclose(pij.sum(1), pi, 1e-12), pij.sum(1))
ch = charts(T)
check(T, "bar-data(1) p(s) が表と一致", allclose(ch[0]["params"]["values"], list(design.values()), 1e-12), ch[0]["params"]["values"])
check(T, "bar-data(2) π_i、単位4=0.15+0.2+0.35=0.7", allclose(ch[1]["params"]["values"], pi, 1e-12) and close(0.15 + 0.2 + 0.35, 0.7, 1e-12), ch[1]["params"]["values"])
check(T, "例2 SRS(10,3): π=0.3, π_ij≈0.0667, π_ij−π_iπ_j=−0.0233",
      close(3 / 10, 0.3, 1e-12) and close(6 / 90, 0.0667) and close(6 / 90 - 0.09, -0.0233), 6 / 90 - 0.09)
# 表の各デザインを列挙で確認
# SRS
Nn, nn2 = 6, 3
pi_ = np.zeros(Nn)
pj_ = np.zeros((Nn, Nn))
ss = list(combinations(range(Nn), nn2))
for s in ss:
    for i in s:
        pi_[i] += 1 / len(ss)
    for i, j in combinations(s, 2):
        pj_[i, j] += 1 / len(ss)
check(T, "表: SRS(6,3) を列挙して π=n/N、π_ij=n(n−1)/(N(N−1))", allclose(pi_, 0.5, 1e-12) and close(pj_[0, 1], 6 / 30, 1e-12), (pi_[0], pj_[0, 1]))
# 層化 層サイズ (4,6)、n_h=(2,3)
strata = [list(range(4)), list(range(4, 10))]
nhs = [2, 3]
cnt_pi = np.zeros(10)
cnt_pj = np.zeros((10, 10))
tot = 0
for picks in product(*[list(combinations(s, k)) for s, k in zip(strata, nhs)]):
    s = sum(picks, ())
    tot += 1
    for i in s:
        cnt_pi[i] += 1
    for i, j in combinations(s, 2):
        cnt_pj[i, j] += 1
cnt_pi /= tot
cnt_pj /= tot
check(T, "表: 層化 π=n_h/N_h、同じ層 n_h(n_h−1)/(N_h(N_h−1))、別の層 π_iπ_j",
      close(cnt_pi[0], 0.5, 1e-12) and close(cnt_pi[5], 0.5, 1e-12) and close(cnt_pj[0, 1], 2 / 12, 1e-12) and close(cnt_pj[4, 5], 6 / 30, 1e-12) and close(cnt_pj[0, 4], 0.25, 1e-12), (cnt_pj[0, 1], cnt_pj[4, 5], cnt_pj[0, 4]))
# 集落 A=4,M=3,a=2 ; 二段 A=4,M=4,a=2,m=2
A_, M_, a_ = 4, 3, 2
ss = list(combinations(range(A_), a_))
pic = np.zeros(A_ * M_)
pjc = np.zeros((A_ * M_, A_ * M_))
for s in ss:
    units = [c * M_ + j for c in s for j in range(M_)]
    for i in units:
        pic[i] += 1 / len(ss)
    for i, j in combinations(units, 2):
        pjc[i, j] += 1 / len(ss)
check(T, "表: 集落 π=a/A、同じ集落 π_ij=a/A", allclose(pic, 0.5, 1e-12) and close(pjc[0, 1], 0.5, 1e-12), (pic[0], pjc[0, 1]))
pit = np.zeros(16)
tot = 0
for s in combinations(range(4), 2):
    for picks in product(*[list(combinations(range(c * 4, c * 4 + 4), 2)) for c in s]):
        tot += 1
        for i in sum(picks, ()):
            pit[i] += 1
pit /= tot
check(T, "表: 二段 π=(a/A)(m/M)", allclose(pit, 0.5 * 0.5, 1e-12), pit[0])
# Cov(Z_i,Z_j)=π_ij−π_iπ_j（例1のデザインで）
Z = np.array([[1 if i in s else 0 for i in range(4)] for s in design])
pr = np.array(list(design.values()))
EZ = pr @ Z
covZ = (Z * pr[:, None]).T @ Z - np.outer(EZ, EZ)
check(T, "例1 デザインで V(Z_i)=π_i(1−π_i)、Cov=π_ij−π_iπ_j", allclose(np.diag(covZ), pi * (1 - pi), 1e-12) and close(covZ[0, 1], pij[0, 1] - pi[0] * pi[1], 1e-12), np.diag(covZ))

# ================= probability-proportional-to-size-sampling =================
T = "probability-proportional-to-size-sampling"
xv = np.array([10, 20, 30, 40], float)
yv = np.array([24, 38, 63, 75], float)
p = xv / xv.sum()
z = yv / p
check(T, "例1 p=(0.1,0.2,0.3,0.4), Y=200, y/p=(240,190,210,187.5)", allclose(p, [0.1, 0.2, 0.3, 0.4], 1e-12) and yv.sum() == 200 and allclose(z, [240, 190, 210, 187.5], 1e-10), z)
Vhh = (p * (z - 200) ** 2).sum() / 2
check(T, "例1 V=(160+20+30+62.5)/2=136.25", allclose(p * (z - 200) ** 2, [160, 20, 30, 62.5], 1e-9) and close(Vhh, 136.25, 1e-10), Vhh)
pairs = list(product(range(4), repeat=2))
ests = [(z[i] + z[j]) / 2 for i, j in pairs]
prs = [p[i] * p[j] for i, j in pairs]
Ehh, Vhh_e = dist_moments(ests, prs)
check(T, "例1 16通り列挙で期待値200、分散136.25", len(pairs) == 16 and close(Ehh, 200, 1e-10) and close(Vhh_e, 136.25, 1e-10), (Ehh, Vhh_e))
Vh_hat = [((z[i] - (z[i] + z[j]) / 2) ** 2 + (z[j] - (z[i] + z[j]) / 2) ** 2) / 2 for i, j in pairs]
check(T, "例1 V̂ の期待値=136.25（不偏）", close(np.dot(prs, Vh_hat), 136.25, 1e-10), np.dot(prs, Vh_hat))
eq_rep = dist_moments([2 * (yv[i] + yv[j]) for i, j in pairs])
eq_nrep = dist_moments([2 * (yv[i] + yv[j]) for i, j in combinations(range(4), 2)])
check(T, "例1 等確率 復元 3228、非復元 2152（列挙）", close(eq_rep[1], 3228, 1e-9) and close(eq_nrep[1], 2152, 1e-9) and close(eq_rep[0], 200, 1e-9), (eq_rep, eq_nrep))
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [3228,2152,136.25]、非復元比 1/15 以下", allclose(ch, [eq_rep[1], eq_nrep[1], Vhh], 1e-6) and eq_nrep[1] / Vhh > 15, eq_nrep[1] / Vhh)
check(T, "例2 村3,村2 で Ŷ=200, V̂=100", close((210 + 190) / 2, 200, 1e-12) and close(((210 - 200) ** 2 + (190 - 200) ** 2) / 2, 100, 1e-12), 100)
check(T, "y=cx なら y/p=Y で分散0", allclose((3 * xv) / p, 3 * xv.sum(), 1e-10), (3 * xv) / p)
Ms = np.array([50, 120, 80, 200, 150])
nsel, m = 2, 10
check(T, "二段で自己加重: n(M_i/M0)(m/M_i)=nm/M0（全集落で等しい）", allclose(nsel * Ms / Ms.sum() * m / Ms, nsel * m / Ms.sum(), 1e-15), nsel * m / Ms.sum())
# 乱数: HH 推定量の不偏性と分散
Np = 50
xp_ = rng.integers(10, 100, Np).astype(float)
yp_ = 2 * xp_ + rng.normal(0, 10, Np)
pp = xp_ / xp_.sum()
zz = yp_ / pp
reps = 200000
draws = rng.choice(Np, size=(reps, 5), p=pp)
hh = zz[draws].mean(1)
Vth = (pp * (zz - yp_.sum()) ** 2).sum() / 5
check(T, "乱数 N=50,n=5（復元PPS, 20万回）で E=Y（4SE以内）・V 相対誤差1%",
      abs(hh.mean() - yp_.sum()) < 4 * math.sqrt(Vth / reps) and abs(hh.var() / Vth - 1) < 0.01, (hh.mean(), yp_.sum(), hh.var(), Vth))

# ================= design-effect =================
T = "design-effect"
deff = 1 + 19 * 0.05
check(T, "例1 deff=1.95、n_eff≈512.8", close(deff, 1.95, 1e-12) and close(1000 / deff, 512.8, 5e-2), 1000 / deff)
se0 = math.sqrt(0.4 * 0.6 / 1000)
check(T, "例1 SE 0.0155 → ×√1.95=0.0216", close(se0, 0.0155) and close(0.0155 * math.sqrt(1.95), 0.0216) and close(se0 * math.sqrt(1.95), 0.0216), (se0, se0 * math.sqrt(1.95)))
check(T, "例1 半幅 0.030 → 0.042", close(1.96 * se0, 0.030, 5e-4) and close(1.96 * se0 * math.sqrt(1.95), 0.042, 5e-4), (1.96 * se0, 1.96 * se0 * math.sqrt(1.95)))
w = np.array([1] * 500 + [3] * 500, float)
dk = len(w) * (w ** 2).sum() / w.sum() ** 2
check(T, "例2 5,000,000/4,000,000=1.25、n_eff=800", close(1000 * (500 + 4500), 5e6, 1e-6) and close((500 + 1500) ** 2, 4e6, 1e-6) and close(dk, 1.25, 1e-12) and close(1000 / dk, 800, 1e-9), dk)
check(T, "例2 1+CV_w²（n 割り）=1.25", close(1 + w.var() / w.mean() ** 2, 1.25, 1e-12), 1 + w.var() / w.mean() ** 2)
check(T, "例3 4.41/5.558=0.793, 3.12/5.558=0.561", close(4.41 / 5.558, 0.793, 5e-4) and close(3.12 / 5.558, 0.561, 5e-4) and close(Vprop / Vsrs, 0.793, 5e-4) and close(Vopt / Vsrs, 0.561, 5e-4), (Vprop / Vsrs, Vopt / Vsrs))
ch = charts(T)[0]["params"]["values"]
check(T, "bar-data [0.561,0.793,1,1.25,1.95]", allclose(ch, [Vopt / Vsrs, Vprop / Vsrs, 1.0, dk, deff], 5e-4), ch)
# Kish の恒等式（sympy, n=3）
wv = sp.symbols("w1:4", positive=True)
mw = sum(wv) / 3
cv2 = sum((wi - mw) ** 2 for wi in wv) / 3 / mw ** 2
check(T, "nΣw²/(Σw)²=1+CV_w²（sympy, n=3）", sp.simplify(3 * sum(wi ** 2 for wi in wv) / sum(wv) ** 2 - 1 - cv2) == 0, "恒等式")
# 交換可能な相関 ρ の M 個の平均の分散
Msym, rs, sg = sp.symbols("M rho sigma", positive=True)
Mv = 5
Sig = sp.Matrix(Mv, Mv, lambda i, j: sg ** 2 if i == j else rs * sg ** 2)
one = sp.ones(Mv, 1) / Mv
vm = sp.simplify((one.T * Sig * one)[0])
check(T, "相関 ρ の M 個の平均の分散=σ²/M{1+(M−1)ρ}（sympy, M=5）", sp.simplify(vm - sg ** 2 / Mv * (1 + (Mv - 1) * rs)) == 0, vm)
# 乱数: 加重平均の分散比
reps = 200000
yy = rng.normal(0, 1, (reps, 1000))
wm = (yy * w).sum(1) / w.sum()
check(T, "乱数 加重平均の分散/単純平均の分散≈1.25（20万回、許容 0.02）", abs(wm.var() / (1 / 1000) - 1.25) < 0.02, wm.var() * 1000)

# ================= nonresponse-bias =================
T = "nonresponse-bias"
check(T, "例1 真の平均 47.2、偏り 0.4×12=4.8", close(0.6 * 52 + 0.4 * 40, 47.2, 1e-12) and close(52 - 47.2, 4.8, 1e-12) and close(0.4 * 12, 4.8, 1e-12), 52 - 47.2)
Wr, Yr, Ynr = sp.symbols("W_r Ybar_r Ybar_nr")
check(T, "Ȳ_r−Ȳ=W_nr(Ȳ_r−Ȳ_nr)（sympy）", sp.expand(Yr - (Wr * Yr + (1 - Wr) * Ynr) - (1 - Wr) * (Yr - Ynr)) == 0, "恒等式")
check(T, "例2 真の平均45、回答者平均 (0.2·30+0.4·60)/0.6=50、偏り5",
      close(0.5 * 30 + 0.5 * 60, 45, 1e-12) and close((0.2 * 30 + 0.4 * 60) / 0.6, 50, 1e-12), (0.2 * 30 + 0.4 * 60) / 0.6)
phi = np.array([0.4, 0.8])
yy2 = np.array([30, 60])
covp = (0.5 * phi * yy2).sum() - phi.mean() * 45
check(T, "例2 Cov(φ,y)=3、3/0.6=5", close(covp, 3, 1e-12) and close(covp / 0.6, 5, 1e-12), covp)
check(T, "例2 重み付け直しで 45", close(0.5 * 30 + 0.5 * 60, 45, 1e-12), 45)
ch = charts(T)[0]["params"]
ok = all(allclose([(1 - rr) * d for rr in ch["x"]], s["y"], 1e-9) for d, s in zip([5, 10, 20], ch["series"]))
check(T, "line-data (1−回答率)×差（差5,10,20）", ok, "8点×3本")
# 乱数: 確率的モデルの近似
Np = 10000
grp = np.repeat([0, 1], Np // 2)
yv_ = np.where(grp == 0, 30, 60) + rng.normal(0, 5, Np)
ph = np.where(grp == 0, 0.4, 0.8)
reps = 2000
biases = np.empty(reps)
for t in range(reps):
    resp = rng.random(Np) < ph
    biases[t] = yv_[resp].mean() - yv_.mean()
apx = (np.mean(ph * yv_) - ph.mean() * yv_.mean()) / ph.mean()
check(T, "乱数 回答傾向 0.4/0.8 の母集団（N=1万、2000回）で偏り≈Cov(φ,y)/φ̄（許容 0.02、SE 約0.003）", abs(biases.mean() - apx) < 0.02, (biases.mean(), apx))
# 層ごとに重み付け直すと偏りが消える
wb = np.empty(reps)
for t in range(reps):
    resp = rng.random(Np) < ph
    wb[t] = 0.5 * yv_[resp & (grp == 0)].mean() + 0.5 * yv_[resp & (grp == 1)].mean() - yv_.mean()
check(T, "乱数 MAR（層内で回答と値が無関係）なら層の重み付け直しで偏り≈0（許容 0.02）", abs(wb.mean()) < 0.02, wb.mean())

# ================= sampling-and-nonsampling-error =================
T = "sampling-and-nonsampling-error"
ch = charts(T)[0]["params"]
nvals = np.array(ch["x"], float)
se_ = np.sqrt(0.25 / nvals)
rm = np.sqrt(se_ ** 2 + 0.03 ** 2)
check(T, "line-data 標準誤差 √(0.25/n)", allclose(se_, ch["series"][0]["y"], 5e-5), np.round(se_, 4))
check(T, "line-data √MSE=√(SE²+0.03²)", allclose(rm, ch["series"][1]["y"], 5e-5), np.round(rm, 4))
check(T, "例1 n=100: SE 0.05、√MSE 0.0583、偏りの割合 26.5%", close(se_[0], 0.05, 1e-12) and close(rm[0], 0.0583) and close(0.0009 / rm[0] ** 2 * 100, 26.5, 5e-2), 0.0009 / rm[0] ** 2)
check(T, "例1 n=2000: SE 0.0112、√MSE 0.0320、割合 87.8%", close(se_[4], 0.0112) and close(rm[4], 0.0320) and close(0.0009 / rm[4] ** 2 * 100, 87.8, 5e-2), 0.0009 / rm[4] ** 2)
dlt = 0.03 / math.sqrt(0.25 / 2000)
cov2000 = stats.norm.cdf(1.96 - dlt) - stats.norm.cdf(-1.96 - dlt)
cov100 = stats.norm.cdf(1.96 - 0.6) - stats.norm.cdf(-1.96 - 0.6)
check(T, "例1 0.03/0.0112=2.683、被覆確率 0.235（n=2000）・0.908（n=100）", close(dlt, 2.683, 5e-4) and close(cov2000, 0.235, 5e-4) and close(cov100, 0.908, 5e-4), (dlt, cov2000, cov100))
check(T, "例2 真の賛成率 0.56、偏り 0.04", close(0.8 * 0.6 + 0.2 * 0.4, 0.56, 1e-12) and close(0.6 - 0.56, 0.04, 1e-12) and close(0.2 * 0.2, 0.04, 1e-12), 0.56)
th, a_, b_ = sp.symbols("theta a b")
check(T, "カバレッジ偏り Ȳ_c−Ȳ=W_u(Ȳ_c−Ȳ_u)（sympy）", sp.expand(Yr - (Wr * Yr + (1 - Wr) * Ynr) - (1 - Wr) * (Yr - Ynr)) == 0, "恒等式")
# 乱数: MSE=V+Bias²、偏りのある CI の被覆率
reps = 400000
ph_ = rng.binomial(2000, 0.53, reps) / 2000
mse = ((ph_ - 0.5) ** 2).mean()
check(T, "乱数 MSE=V+Bias²（p=0.53 を 0.5 の推定に使う, n=2000, 40万回, 相対誤差 1%）", abs(mse / (ph_.var() + (ph_.mean() - 0.5) ** 2) - 1) < 1e-9 and abs(mse / (0.53 * 0.47 / 2000 + 0.0009) - 1) < 0.01, mse)
covsim = np.mean(np.abs(ph_ - 0.5) <= 1.96 * math.sqrt(0.25 / 2000))
check(T, "乱数 偏り 0.03 の 95%CI の被覆率≈0.235（正規近似との差 0.02 以内）", abs(covsim - 0.235) < 0.02, covsim)


# =====================================================================
# 重要度Cの用語（sampling-weight）
# 乱数は seed=20261023+21
# =====================================================================
rngC = np.random.default_rng(20261023 + 21)

T = "sampling-weight"
Nh = np.array([400, 250, 100]); nh = np.array([20, 15, 10]); yb = np.array([30, 45, 80.])
dw_ = Nh / nh
check(T, "ウェイト 20.000, 16.667, 10.000", allclose(dw_, [20, 16.667, 10], 5e-4), dw_)
check(T, "ウェイトの合計 750=N", close((nh * dw_).sum(), 750, 1e-9), (nh * dw_).sum())
wm = (nh * dw_ * yb).sum() / (nh * dw_).sum()
check(T, "重み付き平均 31250/750=41.667（層化抽出の推定値と一致）", close((Nh * yb).sum(), 31250, 1e-9) and close(wm, 41.667, 5e-4) and close(wm, (Nh / Nh.sum() * yb).sum(), 1e-12), wm)
uw = (nh * yb).sum() / nh.sum()
check(T, "重みなし平均 46.111（過大）", close(uw, 46.111, 5e-4) and uw > wm, uw)
sd2 = (nh * dw_ ** 2).sum()
deff = nh.sum() * sd2 / (nh * dw_).sum() ** 2
check(T, "Σd²=13166.7、計画効果 1.053、有効標本 42.7", close(sd2, 13166.7, 0.05) and close(deff, 1.053, 5e-4) and close(nh.sum() / deff, 42.7, .05), (sd2, deff, nh.sum() / deff))
check(T, "無回答調整後の営業ウェイト 250/12=20.833（=16.667×15/12）", close(250 / 12, 20.833, 5e-4) and close(dw_[1] * 15 / 12, 250 / 12, 1e-12), 250 / 12)
check(T, "比例配分なら全員 N/n で、重みなし平均と一致", np.allclose(np.array([24, 15, 6]) / np.array([400, 250, 100]), 45 / 750) and close(np.sum(np.array([24, 15, 6]) * yb) / 45, np.sum(Nh * yb) / 750, 1e-12), "ok")
# 乱数：不等確率抽出（層化）で HT 推定量は不偏、重みなしは偏る
pop = [rngC.normal(m, 8, N) for m, N in zip(yb, Nh)]
Ytot = sum(p.sum() for p in pop)
R = 20000
ht, hj, unw = [], [], []
for _ in range(R):
    smp = [rngC.choice(p, n, replace=False) for p, n in zip(pop, nh)]
    ht.append(sum(d * s.sum() for d, s in zip(dw_, smp)))
    hj.append(ht[-1] / sum(d * n for d, n in zip(dw_, nh)))
    unw.append(np.concatenate(smp).mean())
ht, unw = np.array(ht), np.array(unw)
seht = ht.std() / math.sqrt(R)
check(T, "乱数 2万回：Ŷ=Σd y の平均＝母集団総計（4SE 以内）、重みなし平均は過大", abs(ht.mean() - Ytot) < 4 * seht and unw.mean() - Ytot / Nh.sum() > 3,
      (ht.mean(), Ytot, unw.mean(), Ytot / Nh.sum()))
# キッシュの計画効果：層内で値が同質（分散が等しい）なら、Σd y/Σd の分散 / 単純無作為の分散 ≈ deff
popK = [rngC.normal(0, 1, N) for N in Nh]  # 全層で同じ分布
allK = np.concatenate(popK)
wmK, srsK = [], []
for _ in range(R):
    smp = [rngC.choice(p, n, replace=False) for p, n in zip(popK, nh)]
    wmK.append(sum(d * s.sum() for d, s in zip(dw_, smp)) / 750)
    srsK.append(rngC.choice(allK, 45, replace=False).mean())
ratioK = np.var(wmK) / np.var(srsK)
check(T, "乱数 層の平均が同じとき 分散比≈キッシュの 1.053（有限母集団修正込みの正確値 1.057、±0.05）", abs(ratioK - deff) < .05, ratioK)

# ================= 集計 =================
nok = sum(results)
nng = len(results) - nok
print(f"\n合計: OK {nok} / NG {nng}")
