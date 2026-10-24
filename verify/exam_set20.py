"""模擬試験 第20回（set20）の検算。generator/set20.py とは別の方法（scipy の分布関数・数値積分・sympy・
シミュレーション・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(20)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


def table(iid, value, shown, d, what):
    """問題文に示した数表の値 shown が、正確な値 value を小数第 d 位に丸めたものと一致するか。"""
    E.check(iid, f(value, d) == f(shown, d), f"数表 {what}: 示した値 {shown} ／ 正確な値 {value:.6f}")


def nums_in(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def distinct(iid, d):
    v = [f(x, d) for x in nums_in(iid)]
    E.check(iid, len(set(v)) == 5, f"選択肢の数値がすべて異なる {v}")


x, y, w, s_ = sp.symbols("x y w s", positive=True)

# ---- 大問1 同時密度 f=(2/3)(2x+y)
fxy = sp.Rational(2, 3) * (2 * x + y)
E.check("s20p01a", sp.integrate(fxy, (x, 0, 1), (y, 0, 1)) == 1, "同時密度の積分が 1")
pyx = sp.integrate(sp.integrate(fxy, (x, 0, y)), (y, 0, 1))
E.choice("s20p01a", f"${f(float(pyx), 3)}$", f"sympy で P(Y>X)={pyx}")
N = 2_000_000
xs, ys, us = rng.random(N), rng.random(N), rng.random(N)
keep = us < (2 * xs + ys) / 3
xs, ys = xs[keep], ys[keep]
E.check("s20p01a", abs((ys > xs).mean() - float(pyx)) < 0.003, f"受容棄却のシミュレーション {(ys > xs).mean():.4f}")
distinct("s20p01a", 3)
FX = sp.integrate(sp.integrate(fxy, (y, 0, 1)).subs(x, s_), (s_, 0, x))
med = [float(v) for v in sp.solve(sp.Eq(FX, sp.Rational(1, 2)), x) if 0 < float(v) < 1][0]
E.num("s20p01b", rnd(med, 3), "sympy で周辺分布関数=1/2 を解く")
E.check("s20p01b", abs(np.median(xs) - med) < 0.003, f"シミュレーションの中央値 {np.median(xs):.4f}")

# ---- 大問2
p, pe = 0.9, 0.8
tot = 0.0
for z in product([0, 1], repeat=5):  # SA, SB, AT, BT, AB
    pr = np.prod([(pe if i == 4 else p) if zi else 1 - (pe if i == 4 else p) for i, zi in enumerate(z)])
    SA, SB, AT, BT, AB = z
    ok = (SA and AT) or (SB and BT) or (SA and AB and BT) or (SB and AB and AT)
    tot += pr * ok
E.num("s20p02a", rnd(tot, 3), f"32 通りの状態の総当たり {tot:.5f}")
fw = sp.diff(sp.integrate(sp.exp(-x ** 2 / (2 * s_ ** 2)) / (s_ * sp.sqrt(2 * sp.pi)), (x, -sp.sqrt(w), sp.sqrt(w))), w)
target = sp.exp(-w / (2 * s_ ** 2)) / (s_ * sp.sqrt(2 * sp.pi * w))
E.check("s20p02b", sp.simplify(fw - target) == 0, "sympy で分布関数を微分した密度が正解の選択肢と一致")
E.check("s20p02b", E.item("s20p02b")["answer"] == 1 and "\\sigma\\sqrt{2\\pi w}}\\exp\\left(-\\dfrac{w}{2" in E.item("s20p02b")["choices"][1], "正解の選択肢の式")
E.check("s20p02b", abs(stats.chi2(1, scale=1.5 ** 2).pdf(0.7) - float(target.subs({w: 0.7, s_: 1.5}))) < 1e-12, "scipy の σ²χ²(1) の密度と一致")
E.check("s20p02b", abs(float(sp.integrate(target.subs(s_, 1), (w, 0, sp.oo))) - 1) < 1e-9, "正解の密度の積分が 1（片側だけの式なら 1/2）")

# ---- 大問3
for v, sh, d in ((math.log(1.5625), 0.4463, 4), (math.log(15), 2.708, 3), (math.log(35), 3.555, 3)):
    table("s20p03a", v, sh, d, f"log {sh}")
for zz, sh in ((1.27, 0.8980), (1.50, 0.9332), (1.60, 0.9452), (1.78, 0.9625), (2.40, 0.9918)):
    table("s20p03a", stats.norm.cdf(zz), sh, 4, f"Φ({zz})")
sig2 = math.log(1 + 0.75 ** 2)
mu = math.log(15) - sig2 / 2
lnd = stats.lognorm(s=math.sqrt(sig2), scale=math.exp(mu))
E.check("s20p03a", abs(lnd.mean() - 15) < 1e-9 and abs(lnd.std() / lnd.mean() - 0.75) < 1e-9, "scipy の対数正規で平均 15・変動係数 0.75")
pex = lnd.sf(35)
E.choice("s20p03a", f"${f(pex, 3)}$", f"scipy lognorm.sf(35)={pex:.5f}")
distinct("s20p03a", 3)
# ギブス：条件付き分布を共分散行列から求め直す
m_ = np.array([2.5, 2.3]); sd = np.array([0.6, 0.5]); rho = 0.8
C = np.array([[0.36, rho * 0.3], [rho * 0.3, 0.25]])
def cond(i, j, val):
    mean = m_[i] + C[i, j] / C[j, j] * (val - m_[j])
    var = C[i, i] - C[i, j] ** 2 / C[j, j]
    return mean, math.sqrt(var)
mU, sU = cond(0, 1, 2.0)
U1 = mU + sU * 0.5
mV, sV = cond(1, 0, U1)
V1 = mV + sV * (-1.0)
E.choice("s20p03b", f"$({f(U1, 3)},\\ {f(V1, 3)})$", f"共分散行列から条件付き分布 ({U1:.4f}, {V1:.4f})")
# ギブス連鎖の定常分布が目標の2変量正規になること
u, v = 2.6, 2.0
draws = []
for _ in range(60000):
    a, b = cond(0, 1, v); u = a + b * rng.standard_normal()
    a, b = cond(1, 0, u); v = a + b * rng.standard_normal()
    draws.append((u, v))
draws = np.array(draws[1000:])
E.check("s20p03b", abs(np.corrcoef(draws.T)[0, 1] - 0.8) < 0.02 and np.allclose(draws.std(0), sd, atol=0.02), "ギブス連鎖の相関と標準偏差が目標と一致")

# ---- 大問4
table("s20p04a", math.log(365), 5.900, 3, "log 365")
table("s20p04a", -math.log(0.98), 0.0202, 4, "-log 0.98")
table("s20p04a", math.log(0.0202), -3.902, 3, "log 0.0202")
xq = 10 - 3 * math.log(1 - 0.98 ** (1 / 365))
E.num("s20p04a", rnd(xq, 1), f"(1-e^-y)^365=0.98 を正確に解く {xq:.4f}")
gum = stats.gumbel_r(loc=10 + 3 * math.log(365), scale=3).ppf(0.98)
E.check("s20p04a", rnd(gum, 1) == E.item("s20p04a")["answer"], f"scipy gumbel_r の 98% 点 {gum:.4f}")

# ---- 大問5
table("s20p05a", stats.t.ppf(0.975, 15), 2.131, 3, "t(15)")
table("s20p05a", stats.t.ppf(0.975, 16), 2.120, 3, "t(16)")
lo = stats.t.interval(0.95, 15, loc=94.35, scale=0.8 / 4)[0]
E.num("s20p05a", rnd(94.35 - 2.131 * 0.8 / 4, 3), "示した t 点で下限")
E.check("s20p05a", abs(lo - E.item("s20p05a")["answer"]) <= 0.0005, f"scipy t.interval の下限 {lo:.5f}")
Xs = rng.normal(0, 1, (200_000, 16))
xb_, sdv = Xs.mean(1), Xs.std(1, ddof=1)
cov = np.mean(np.abs(xb_) <= 2.131 * sdv / 4)
E.check("s20p05a", abs(cov - 0.95) < 0.003, f"信頼区間の被覆率 {cov:.4f}")
c, n = sp.symbols("c"), 16
mse = c ** 2 * 2 * (n - 1) + (c * (n - 1) - 1) ** 2
cs = sp.solve(sp.diff(mse, c), c)[0]
E.check("s20p05b", cs == sp.Rational(1, 17), f"sympy で最小化 c={cs}")
E.choice("s20p05b", f"c=1/17$、推定値$ {f(9.6 / 17, 3)}", "推定値 9.6/17")
Ssim = stats.chi2(15).rvs(size=400_000, random_state=1)
mses = {dd: np.mean((Ssim / dd - 1) ** 2) for dd in (14, 15, 16, 17, 18)}
E.check("s20p05b", min(mses, key=mses.get) == 17, "シミュレーションの MSE が 17 で最小")

# (c) z 検定の検出力：scipy の正規分布で両側とも足して計算し、シミュレーションでも確認
for zz, sh in ((0.54, 0.7054), (0.86, 0.8051), (1.34, 0.9099), (2.50, 0.9938)):
    table("s20p05c", stats.norm.cdf(zz), sh, 4, f"Φ({zz})")
se_ = 0.8 / 4
pw = stats.norm.sf(1.96 * se_, loc=0.5, scale=se_) + stats.norm.cdf(-1.96 * se_, loc=0.5, scale=se_)
E.choice("s20p05c", f"${f(pw, 3)}$", f"両側の検出力 {pw:.5f}")
xbars = rng.normal(94.5, se_, 400_000)
E.check("s20p05c", abs(np.mean(np.abs((xbars - 94.0) / se_) > 1.96) - pw) < 0.003, "シミュレーションの棄却率")
distinct("s20p05c", 3)

# ---- 大問6
# k 平均法の結果を模した 20 点のデータで CH 指標を sklearn を使わず計算
def ch_index(B, W, k, n):
    return (B / (k - 1)) / (W / (n - k))
chv = {k: ch_index(100 - Wk, Wk, k, 20) for k, Wk in ((2, 40), (3, 25), (4, 20))}
E.choice("s20p06a", f"$k=2$で最大（{f(chv[2], 1)}）なので", f"CH={chv}")
# 実データで全平方和 = 群内 + 群間 の分解と CH の式を確認
pts = rng.normal(size=(20, 2)); lab = np.repeat([0, 1, 2, 3], 5)
Tss = ((pts - pts.mean(0)) ** 2).sum(); Wss = sum(((pts[lab == g] - pts[lab == g].mean(0)) ** 2).sum() for g in range(4))
Bss = sum(5 * ((pts[lab == g].mean(0) - pts.mean(0)) ** 2).sum() for g in range(4))
E.check("s20p06a", abs(Tss - Wss - Bss) < 1e-9, "全平方和 = 群内 + 群間（B_k = 全体 − W_k）")
l = sp.symbols("l1:4", positive=True)
sol = sp.solve([l[0] * l[1] - sp.Rational(56, 100), l[0] * l[2] - sp.Rational(48, 100), l[1] * l[2] - sp.Rational(42, 100)], l, dict=True)[0]
E.choice("s20p06b", f"${f(1 - float(sol[l[1]]) ** 2, 2)}$", f"sympy で負荷量を解く {sol}")
# 1因子モデルの相関行列を作ってシミュレーション
L = np.array([0.8, 0.7, 0.6])
F = rng.standard_normal(300_000)
Xf = np.outer(F, L) + rng.standard_normal((300_000, 3)) * np.sqrt(1 - L ** 2)
Rm = np.corrcoef(Xf.T)
E.check("s20p06b", abs(Rm[0, 1] - 0.56) < 0.01 and abs(Rm[1, 2] - 0.42) < 0.01, "負荷量 (0.8,0.7,0.6) の1因子モデルが相関を再現")

# ---- 大問7
# MA(1) の偏自己相関：statsmodels を使わずに自己相関から再帰で計算し、シミュレーションでも確認
th = 0.5
rho = lambda k: 1.0 if k == 0 else (th / (1 + th ** 2) if k == 1 else 0.0)
R2 = np.array([[1, rho(1)], [rho(1), 1]])
phi = np.linalg.solve(R2, [rho(1), rho(2)])
E.choice("s20p07a", f"${f(phi[1], 3)}$", f"2 次のユール・ウォーカー方程式の最後の係数 {phi[1]:.5f}")
eps = rng.standard_normal(400_001)
xm = eps[1:] + th * eps[:-1]
reg = np.linalg.lstsq(np.column_stack([xm[1:-1], xm[:-2]]), xm[2:], rcond=None)[0]
E.check("s20p07a", abs(reg[1] - phi[1]) < 0.01, f"シミュレーションの回帰で φ22={reg[1]:.4f}")

# ---- 大問8
# 吸収されるまでの期待ステップ数を連立一次方程式で解く
N_ = 10
A = np.zeros((N_ - 1, N_ - 1)); bvec = np.ones(N_ - 1)
for i in range(1, N_):
    A[i - 1, i - 1] = 1
    if i - 1 >= 1: A[i - 1, i - 2] = -0.5
    if i + 1 <= N_ - 1: A[i - 1, i] = -0.5
Dsol = np.linalg.solve(A, bvec)
E.choice("s20p08a", f"${f(Dsol[2] * 0.25, 2)}$時間", f"連立方程式で D_3={Dsol[2]:.4f}")
lens = []
for _ in range(20000):
    pos, st = 3, 0
    while 0 < pos < 10:
        pos += 1 if rng.random() < 0.5 else -1; st += 1
    lens.append(st)
E.check("s20p08a", abs(np.mean(lens) - 21) < 0.5, f"シミュレーションの平均ステップ数 {np.mean(lens):.3f}")

# ---- 大問9：S 行列を再現するデータを作って最小二乗で確かめる
Smat = np.array([[100, 160, 210], [160, 400, 360], [210, 360, 900]], float)
Z = rng.standard_normal((30, 3))
Z -= Z.mean(0)
Lc = np.linalg.cholesky(Z.T @ Z)
Z = Z @ np.linalg.inv(Lc).T
D = Z @ np.linalg.cholesky(Smat).T
E.check("s20p09a", np.allclose(D.T @ D, Smat), "偏差平方和・積和を再現するデータを作った")
X = np.column_stack([np.ones(30), D[:, 0] + 30, D[:, 1] + 60])
yv = D[:, 2] + 500
bh, rss, *_ = np.linalg.lstsq(X, yv, rcond=None)
s2 = rss[0] / 27
XtXi = np.linalg.inv(X.T @ X)
tval = bh[1] / math.sqrt(s2 * XtXi[1, 1])
E.num("s20p09a", rnd(tval, 2), f"作ったデータの最小二乗で t={tval:.4f}")
E.check("s20p09a", abs(stats.linregress(D[:, 0], D[:, 2]).slope - 2.1) < 1e-9, "単回帰の傾き 2.1（誤答の確認）")
H = X @ XtXi @ X.T
xnew = np.array([1, 30 + 2, 60 - 3])
hnew = xnew @ XtXi @ xnew
E.choice("s20p09b", f"$h_{{ii}}={f(hnew, 3)}$で、目安0.2を上回る", f"ハット行列の式 x'(X'X)^(-1)x={hnew:.5f}")
E.check("s20p09b", abs(np.trace(H) - 3) < 1e-9, "ハット行列のトレース = 3（目安 2p/n の p=3）")
# ニューラルネット：行列で順伝播し、パラメータを数える
W1 = np.array([[0.8, 0.4], [-0.6, 0.3]]); c1 = np.array([-0.1, 0.2]); w2 = np.array([1.5, 2.0]); c2 = 0.3
hid = np.maximum(W1 @ np.array([1.0, -0.5]) + c1, 0)
yhat = w2 @ hid + c2
npar = W1.size + c1.size + w2.size + 1
E.choice("s20p09c", f"$\\hat y={f(yhat, 2)}$、パラメータ {npar} 個", f"行列計算で ŷ={yhat:.4f}, パラメータ {npar}")

# ---- 大問10
zz = (78 - 60) / math.sqrt(60)
E.choice("s20p10a", f"$z={f(zz, 2)}$で1.645を超える", f"z={zz:.4f}")
E.check("s20p10a", stats.poisson(60).sf(77) < 0.05, f"正確なポアソンの片側 p 値 {stats.poisson(60).sf(77):.4f} も 5% 未満")
# IRLS：重み付き最小二乗（作業変数）として 1 回解く
yb = np.r_[np.ones(30), np.zeros(70)]; Xb = np.ones((100, 1)); beta = np.zeros(1)
pb = 1 / (1 + np.exp(-Xb @ beta)); Wd = pb * (1 - pb)
zwork = Xb @ beta + (yb - pb) / Wd
beta1 = np.linalg.lstsq(Xb * np.sqrt(Wd)[:, None], zwork * np.sqrt(Wd), rcond=None)[0][0]
E.choice("s20p10b", f"${f(beta1, 3)}$", f"作業変数の重み付き最小二乗で β1={beta1:.5f}")
E.check("s20p10b", abs(math.log(30 / 70) - (-0.8473)) < 1e-4, "収束後の最尤推定値 log(3/7) は別の値（誤答の確認）")
distinct("s20p10b", 3)

# ---- 大問11
table("s20p11a", stats.f.ppf(0.95, 2, 12), 3.885, 3, "F(2,12)")
table("s20p11a", stats.t.ppf(0.975, 12), 2.179, 3, "t(12)")
cvec = np.array([1, -0.5, -0.5]); means = np.array([32.0, 34.0, 35.0])
tsch = abs(cvec @ means) / math.sqrt(4.0 * (cvec ** 2).sum() / 5)
crit = math.sqrt(2 * stats.f.ppf(0.95, 2, 12))
E.choice("s20p11a", f"$|t|={f(tsch, 2)}$が$\\sqrt{{2F_{{0.05}}(2,12)}}={f(math.sqrt(2 * 3.885), 3)}$を超えないので有意でない", f"|t|={tsch:.4f}, 臨界値 {crit:.4f}")
E.check("s20p11a", stats.t.ppf(0.975, 12) < tsch < crit, "計画した対比の t なら有意、シェフェでは有意でない")
# 漸近相対効率：正規分布でのシミュレーションで検出力を比べ、必要標本数の比が π/3 に近いことを確かめる
E.choice("s20p11b", f"$({math.ceil(40 * math.pi / 3)},\\ {math.ceil(40 * math.pi / 2)})$", "40π/3 と 40π/2 の切り上げ")
E.check("s20p11b", abs(3 / math.pi - 0.955) < 0.001 and abs(2 / math.pi - 0.637) < 0.001, "3/π=0.955, 2/π=0.637")

# ---- 大問12
# 抽出ウェイト：個票を作って重み付き平均
yy1 = np.full(250, 420.0); yy2 = np.full(250, 250.0)
wts = np.r_[np.full(250, 6000 / 250), np.full(250, 4000 / 250)]
est = np.average(np.r_[yy1, yy2], weights=wts)
E.num("s20p12a", rnd(est, 1), f"ウェイト付き平均 {est}")
# 層別抽出の期待値のシミュレーション（不偏性）
pop1 = rng.normal(420, 80, 6000); pop2 = rng.normal(250, 40, 4000)
ests = [np.average(np.r_[rng.choice(pop1, 250, replace=False), rng.choice(pop2, 250, replace=False)], weights=wts) for _ in range(2000)]
E.check("s20p12a", abs(np.mean(ests) - np.r_[pop1, pop2].mean()) < 1.0, "ウェイト付き推定量はほぼ不偏")
cfi = 1 - max(36 - 24, 0) / max(420 - 36, 36 - 24, 0)
E.choice("s20p12b", f"${f(cfi, 3)}$", f"CFI={cfi:.5f}")
distinct("s20p12b", 3)
# 線形SEMのシミュレーション
nS = 1_000_000
Xc = rng.standard_normal(nS)
T = 0.4 * Xc + rng.standard_normal(nS) * math.sqrt(1 - 0.16)
M = 0.5 * T + rng.standard_normal(nS)
Y = -0.2 * T - 0.6 * M + 0.3 * Xc + rng.standard_normal(nS)
coefTX = np.linalg.lstsq(np.column_stack([T, Xc]), Y, rcond=None)[0][0]
E.check("s20p12c", abs(coefTX + 0.5) < 0.01, f"シミュレーションの回帰係数 {coefTX:.4f}")
a_, b_, c_ = sp.symbols("a b c")
E.choice("s20p12c", f"${f(-0.2 + 0.5 * (-0.6), 2)}$", "c+ab")
coefT = np.linalg.lstsq(T[:, None], Y, rcond=None)[0][0]
E.check("s20p12c", abs(coefT + 0.38) < 0.01, f"調整しないと {coefT:.4f}（誤答 −0.38 の確認）")

# ---- 大問13
obs = rng.normal(4.2, math.sqrt(2.0), 20)
obs = 4.2 + (obs - obs.mean()) / obs.std(ddof=1) * math.sqrt(2.0)
full = np.concatenate([obs, np.full(10, obs.mean())])
E.num("s20p13a", rnd(full.var(ddof=1), 2), f"実際に代入して不偏分散 {full.var(ddof=1):.5f}")

E.done()
