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
up = 94.35 + stats.t.ppf(0.975, 15) * 0.8 * math.sqrt(1 + 1 / 16)
E.num("s20p05a", rnd(94.35 + 2.131 * 0.8 * math.sqrt(17 / 16), 3), "示した t 点で予測区間の上限")
E.check("s20p05a", abs(up - E.item("s20p05a")["answer"]) <= 0.001, f"正確な t 点で {up:.5f}（許容範囲内）")
# 予測区間の被覆率をシミュレーションで確認
Xs = rng.normal(0, 1, (200_000, 17))
xb, sdv = Xs[:, :16].mean(1), Xs[:, :16].std(1, ddof=1)
cov = np.mean(np.abs(Xs[:, 16] - xb) <= stats.t.ppf(0.975, 15) * sdv * math.sqrt(17 / 16))
E.check("s20p05a", abs(cov - 0.95) < 0.003, f"予測区間の被覆率 {cov:.4f}")
c, n = sp.symbols("c"), 16
mse = c ** 2 * 2 * (n - 1) + (c * (n - 1) - 1) ** 2
cs = sp.solve(sp.diff(mse, c), c)[0]
E.check("s20p05b", cs == sp.Rational(1, 17), f"sympy で最小化 c={cs}")
E.choice("s20p05b", f"c=1/17$、推定値$ {f(9.6 / 17, 3)}", "推定値 9.6/17")
Ssim = stats.chi2(15).rvs(size=400_000, random_state=1)
mses = {dd: np.mean((Ssim / dd - 1) ** 2) for dd in (14, 15, 16, 17, 18)}
E.check("s20p05b", min(mses, key=mses.get) == 17, "シミュレーションの MSE が 17 で最小")

# ---- 大問6
P = np.array([[0, 0], [1.2, 0.5], [2.5, 1.3], [3.8, 0.9], [0.9, 2.4]])
dtrue = pdist(P)
dshow = np.round(dtrue, 1)
qtxt = E.item("s20p06a")["question"]
E.check("s20p06a", "| A | 0 | 1.3 | 2.8 | 3.9 | 2.6 |" in qtxt and "| E | 2.6 | 1.9 | 1.9 | 3.3 | 0 |" in qtxt, "距離表は2次元の座標から求めたユークリッド距離を丸めたもの")
Zs = linkage(dshow, "single")
hAD = squareform(cophenet(Zs))[0, 3]
E.choice("s20p06a", f"${f(hAD, 1)}$", f"scipy の最短距離法のコーフェン距離 {hAD}")
E.check("s20p06a", squareform(cophenet(linkage(dtrue, "single")))[0, 3] < 1.55, "丸める前の距離でも同じ結合順")
E.check("s20p06a", abs(squareform(cophenet(linkage(dshow, "average")))[0, 3] - 8 / 3) < 1e-9
        and squareform(cophenet(linkage(dshow, "complete")))[0, 3] == 3.9, "群平均法 2.67・最長距離法 3.9（誤答の確認）")
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
for v, sh, d in ((stats.chi2.ppf(0.95, 3), 7.815, 3), (stats.chi2.ppf(0.95, 4), 9.488, 3), (stats.chi2.ppf(0.95, 2), 5.991, 3)):
    table("s20p07a", v, sh, d, f"{sh}")
rk = np.array([0.3, 0.2, 0.1]); nn = 55
Q = nn * (nn + 2) * np.sum(rk ** 2 / (nn - np.arange(1, 4)))
E.choice("s20p07a", f"$Q={f(Q, 2)}$で7.815を超えるので棄却する。予測誤差に自己相関が残っている", f"numpy で Q={Q:.4f}")
E.check("s20p07a", nn * np.sum(rk ** 2) < 7.815 < Q, "ボックス・ピアースでは結論が変わる")
# 白色雑音のもとで Q のシミュレーション分布が χ²(3) に近いこと
sims = rng.standard_normal((20000, nn))
sims -= sims.mean(1, keepdims=True)
den = (sims ** 2).sum(1)
rr = np.stack([(sims[:, k:] * sims[:, :-k]).sum(1) / den for k in (1, 2, 3)], 1)
Qs = nn * (nn + 2) * (rr ** 2 / (nn - np.arange(1, 4))).sum(1)
E.check("s20p07a", abs((Qs > 7.815).mean() - 0.05) < 0.015, f"白色雑音のもとでの棄却率 {(Qs > 7.815).mean():.3f}")
table("s20p07b", stats.chi2.ppf(0.95, 2), 5.991, 3, "χ²(2)")
jb = 55 / 6 * (0.6 ** 2 + (4.2 - 3) ** 2 / 4)
E.choice("s20p07b", f"$JB={f(jb, 2)}$で5.991を超えるので", f"JB={jb:.4f}")
E.check("s20p07b", stats.chi2.sf(jb, 2) < 0.05, f"p 値 {stats.chi2.sf(jb, 2):.4f}")

# ---- 大問8
table("s20p08a", stats.norm.cdf(1.5), 0.9332, 4, "Φ(1.5)")
table("s20p08a", stats.norm.cdf(0.75), 0.7734, 4, "Φ(0.75)")
pr = 2 * stats.norm.sf(6 / math.sqrt(4 * 4))
E.choice("s20p08a", f"${f(pr, 3)}$", f"scipy で 2P(B(4)≤−6)={pr:.5f}")
# 細かいランダムウォークのシミュレーション
steps = 2000
inc = rng.normal(0, math.sqrt(4 * 4 / steps), (20000, steps))
hit = (np.cumsum(inc, 1).min(1) <= -6).mean()
E.check("s20p08a", abs(hit - pr) < 0.012, f"シミュレーションで下限に達する割合 {hit:.4f}（離散化でやや小さめ）")

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
Rx = np.corrcoef(D[:, :2].T)
ry = np.array([np.corrcoef(D[:, 0], D[:, 2])[0, 1], np.corrcoef(D[:, 1], D[:, 2])[0, 1]])
ev, evec = np.linalg.eigh(Rx)
v1 = evec[:, np.argmax(ev)]
Zs_ = (D[:, :2] - D[:, :2].mean(0)) / D[:, :2].std(0)
ys_ = (D[:, 2] - D[:, 2].mean()) / D[:, 2].std()
pc1 = Zs_ @ v1
g = (pc1 @ ys_) / (pc1 @ pc1)
bp = g * v1
E.choice("s20p09c", f"$({f(bp[0], 3)},\\ {f(bp[1], 3)})$", f"データの主成分回帰 ({bp[0]:.4f}, {bp[1]:.4f})")
E.check("s20p09c", np.allclose(np.linalg.solve(Rx, ry), [0.6111111, 0.1111111], atol=1e-6), "最小二乗の標準化係数 (0.611, 0.111)（誤答の確認）")

# ---- 大問10
zz = (78 - 60) / math.sqrt(60)
E.choice("s20p10a", f"$z={f(zz, 2)}$で1.645を超える", f"z={zz:.4f}")
E.check("s20p10a", stats.poisson(60).sf(77) < 0.05, f"正確なポアソンの片側 p 値 {stats.poisson(60).sf(77):.4f} も 5% 未満")
yy = np.array([6, 4, 11, 4]); mu_ = np.array([4, 6, 9, 6])
xcov = np.log(mu_)  # log μ = 0 + 1×x となる説明変数
# ポアソン回帰の最尤推定を数値で解き、当てはめ値が表どおりになるか
import scipy.optimize as so
nll = lambda b: -(yy * (b[0] + b[1] * xcov) - np.exp(b[0] + b[1] * xcov)).sum()
bb = so.minimize(nll, [0.5, 0.8], method="BFGS", options={"gtol": 1e-10}).x
E.check("s20p10b", np.allclose(np.exp(bb[0] + bb[1] * xcov), mu_, atol=1e-4), f"当てはめ値は x=log μ の説明変数での最尤推定値 {np.exp(bb[0] + bb[1] * xcov)}")
Dv = 2 * np.sum(yy * np.log(yy / mu_) - (yy - mu_))
E.choice("s20p10b", f"${f(Dv, 2)}$", f"逸脱度 {Dv:.4f}")
for v, sh in ((math.log(1.5), 0.4055), (math.log(2 / 3), -0.4055), (math.log(11 / 9), 0.2007)):
    table("s20p10b", v, sh, 4, f"log {sh}")
distinct("s20p10b", 2)

# ---- 大問11
for v, sh, d in ((stats.f.ppf(0.95, 2, 4), 6.944, 3), (stats.f.ppf(0.95, 2, 6), 5.143, 3), (stats.f.ppf(0.95, 2, 10), 4.103, 3)):
    table("s20p11a", v, sh, d, f"{sh}")
# 分割区法の自由度を計画から数える（r=3, a=3, b=2）
r_, a_, b_ = 3, 3, 2
E.check("s20p11a", (r_ - 1, a_ - 1, (r_ - 1) * (a_ - 1), b_ - 1, (a_ - 1) * (b_ - 1), a_ * (r_ - 1) * (b_ - 1)) == (2, 2, 4, 1, 2, 6)
        and r_ * a_ * b_ - 1 == 17, "自由度の分解 2+2+4+1+2+6=17")
FA = (48 / 2) / (16 / 4)
E.choice("s20p11a", f"$F={f(FA, 2)}$で、$F_{{0.05}}(2,4)=6.944$を超えないので有意でない", f"F={FA}, p={stats.f.sf(FA, 2, 4):.4f}")
vals = [38, 41, 45, 43, 47, 52]
cnt = sum(1 for cmb in combinations(range(6), 3) if sum(vals[i] for i in cmb) <= 124)
E.choice("s20p11b", f"${f(cnt / 20, 3)}$", f"20 通りの総当たりで {cnt}/20")
perm = np.array([rng.permutation(vals)[:3].sum() for _ in range(100000)])
E.check("s20p11b", abs((perm <= 124).mean() - 0.1) < 0.005, f"ランダムな並べ替え {(perm <= 124).mean():.4f}")

# ---- 大問12
pop_resp, pop_non = 300.0, 360.0
E.num("s20p12a", rnd(0.7 * pop_resp + 0.3 * pop_non, 1), "層の割合で重み付け")
# 期待値のシミュレーション：回答層と無回答層をもつ母集団
popN = 20000
resp = rng.random(popN) < 0.7
yp = np.where(resp, rng.normal(300, 50, popN), rng.normal(360, 50, popN))
ests = []
for _ in range(3000):
    idx = rng.choice(popN, 400, replace=False)
    r1 = idx[resp[idx]]; r0 = idx[~resp[idx]]
    sub = rng.choice(r0, max(1, len(r0) // 4), replace=False)
    ests.append((len(r1) * yp[r1].mean() + len(r0) * yp[sub].mean()) / 400)
E.check("s20p12a", abs(np.mean(ests) - yp.mean()) < 1.0, f"推定量はほぼ不偏（平均 {np.mean(ests):.2f}、母平均 {yp.mean():.2f}）")
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
