"""模擬試験 第5回（set05）の検算。generator/set05.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値積分・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, linalg, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(5)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def distinct_rounded(iid):
    v = nums(iid)
    E.check(iid, len(set(v)) == 5, f"選択肢の数値がすべて異なる {v}")


def nearest(iid, value, what):
    it = E.item(iid)
    v = nums(iid)
    k = int(np.argmin([abs(x - value) for x in v]))
    E.check(iid, k == it["answer"], f"{what}: 最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def phi_ok(table):
    return all(abs(stats.norm.cdf(z) - v) < 6e-5 for z, v in table.items())


# ---- 大問1：スクリーニング
prev, seA, spA, seB, spB = 0.02, 0.90, 0.95, 0.80, 0.98
joint = {}
for d, a, b in product((0, 1), repeat=3):
    pd = prev if d else 1 - prev
    pa = (seA if a else 1 - seA) if d else ((1 - spA) if a else spA)
    pb = (seB if b else 1 - seB) if d else ((1 - spB) if b else spB)
    joint[(d, a, b)] = pd * pa * pb
ppv = joint[(1, 1, 1)] / (joint[(1, 1, 1)] + joint[(0, 1, 1)])
E.choice("s05p01a", f"${f(ppv, 3)}$", "感染と2検査の全組合せの列挙で P(感染|両方陽性)")
nsim = 2_000_000
sim_d = rng.random(nsim) < prev
sim_a = np.where(sim_d, rng.random(nsim) < seA, rng.random(nsim) < 1 - spA)
sim_b = np.where(sim_d, rng.random(nsim) < seB, rng.random(nsim) < 1 - spB)
pos = sim_a & sim_b
E.check("s05p01a", abs(sim_d[pos].mean() - ppv) < 0.012, f"シミュレーションの陽性的中率 {sim_d[pos].mean():.4f}")
distinct_rounded("s05p01a")

E.check("s05p01b", abs(0.98 ** 10 - 0.8171) < 5e-5, "0.98^10 = 0.8171（4桁）")
bin10 = stats.binom(10, 0.02)
exp_per_pool = sum(bin10.pmf(j) * (1 if j == 0 else 11) for j in range(11))
E.num("s05p01b", rnd(100 * exp_per_pool, 1), "二項分布で 1 + 10·1{感染者≧1} の期待値を足し上げ")
pools = rng.random((2_000_000, 10)) < 0.02
sim_tests = 100 * (1 + 10 * pools.any(axis=1)).mean()
E.check("s05p01b", abs(sim_tests - 282.9) < 1.0, f"シミュレーション（標準誤差は約 0.27） {sim_tests:.2f}")

# ---- 大問2：血圧
mu, sig, rho = 128, 20, 0.6
cov = np.array([[sig ** 2, rho * sig ** 2], [rho * sig ** 2, sig ** 2]])
X = rng.multivariate_normal([mu, mu], cov, size=1_000_000)
c = np.corrcoef(X[:, 0], X[:, 1] - X[:, 0])[0, 1]
r_ = sp.Rational(3, 5)
exact = (r_ - 1) * 400 / (20 * sp.sqrt(2 * (1 - r_) * 400))
E.choice("s05p02a", f"${f(float(exact), 3)}$", "sympy で Cov(X1,D)/(sd·sd)")
E.check("s05p02a", abs(c - float(exact)) < 0.003, f"シミュレーションの相関 {c:.4f}")
distinct_rounded("s05p02a")
E.check("s05p02b", abs(stats.chi2.isf(0.05, 9) - 16.919) < 5e-4 and abs(stats.chi2.isf(0.05, 10) - 18.307) < 5e-4, "χ² の上側5%点")
vd = np.array([1, -1]) @ cov @ np.array([1, -1])
cut = vd * stats.chi2.isf(0.05, 9) / 9
E.choice("s05p02b", f"${f(cut, 1)}$", "D の分散を共分散行列から作り scipy の χ² 点で c")
pairs = rng.multivariate_normal([mu, mu], cov, size=(200_000, 10))
S2 = (pairs[:, :, 1] - pairs[:, :, 0]).var(axis=1, ddof=1)
E.check("s05p02b", abs((S2 > cut).mean() - 0.05) < 0.003, f"シミュレーションで P(S²>c)={(S2 > cut).mean():.4f}")
distinct_rounded("s05p02b")

# ---- 大問3：流行の監視
E.check("s05p03a", phi_ok({0.21: 0.5832, 1.50: 0.9332, 1.57: 0.9418, 1.64: 0.9495}), "示した Φ の値")
E.choice("s05p03a", f"${f(stats.norm.sf(59.5, 49, 7), 3)}$", "scipy norm.sf(59.5, 49, 7)")
nearest("s05p03a", stats.poisson.sf(59, 49), "正確なポアソン確率")
distinct_rounded("s05p03a")
# 部分尤度：リスク集合の指示行列で計算（浮動小数）し分数に直す
tt = np.array([2, 3, 4, 5, 7])
xx = np.array([0, 1, 0, 0, 1])
ev = np.array([1, 1, 0, 1, 1])
beta = math.log(0.5)
risk = tt[None, :] >= tt[:, None]
w = np.exp(beta * xx)
contrib = np.where(ev == 1, w / (risk * w[None, :]).sum(axis=1), 1.0)
pl = float(np.prod(contrib))
E.choice("s05p03b", f"${Fraction(pl).limit_denominator(1000)}$", f"リスク集合行列による部分尤度 {pl:.6f}")
nll = lambda b: -np.sum(ev * (b * xx - np.log((risk * np.exp(b * xx)[None, :]).sum(axis=1))))
bh = optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
E.check("s05p03b", -nll(math.log(0.5)) <= -nll(bh) + 1e-12, f"β̂={bh:.3f} で部分尤度が最大（β=log0.5 の値は最大値以下）")

# ---- 大問4：安全性
crit = {(0.025, 10): 20.483, (0.025, 8): 17.535, (0.975, 8): 2.180, (0.975, 10): 3.247}
E.check("s05p04a", all(abs(stats.chi2.isf(a, v) - x) < 5e-4 for (a, v), x in crit.items()), "示した χ² の点")
muU = optimize.brentq(lambda m_: stats.poisson.cdf(4, m_) - 0.025, 1, 30)
E.num("s05p04a", rnd(muU / 2000 * 1000, 2), f"P(Y≦4|μ)=0.025 を数値で解く（μ_U={muU:.4f}）")
lam = 0.5
E.check("s05p04b", abs(math.exp(0.5) - 1.6487) < 5e-5 and abs(math.exp(-0.5) - 0.6065) < 5e-5, "e^±0.5 の値")
nd, R = 2000, 4000
cnt = rng.poisson(lam, size=(R, nd))
xbar = cnt.mean(axis=1)
lt = -np.log((cnt == 0).mean(axis=1))
eff_sim = xbar.var() / lt.var()
L_ = sp.Symbol("l", positive=True)
p0 = sp.exp(-L_)
info01 = sp.diff(p0, L_) ** 2 / (p0 * (1 - p0))
eff_info = float((info01 / (1 / L_)).subs(L_, lam))
E.choice("s05p04b", f"${f(eff_info, 3)}$", "情報量の比（0/1 データ ÷ 件数データ）")
E.check("s05p04b", abs(eff_sim - eff_info) < 0.05, f"シミュレーションの分散比 {eff_sim:.3f}")
distinct_rounded("s05p04b")

# ---- 大問5：2 群の相関係数の比較（正確な artanh と、2 変量正規のシミュレーションで帰無分布を確認）
z_stat = (math.atanh(0.5) - math.atanh(0.3)) / math.sqrt(1 / 100 + 1 / 80)
E.num("s05p05a", rnd(z_stat, 2), "math.atanh で z 変換して標準化")
E.check("s05p05a", abs(math.atanh(0.5) - 0.5493) < 5e-5 and abs(math.atanh(0.3) - 0.3095) < 5e-5, "示した z 変換の値")
zz = []
for _ in range(20000):
    a1 = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 103)
    a2 = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 83)
    zz.append((math.atanh(np.corrcoef(a1.T)[0, 1]) - math.atanh(np.corrcoef(a2.T)[0, 1])) / math.sqrt(1 / 100 + 1 / 80))
E.check("s05p05a", abs(np.std(zz) - 1) < 0.03, f"帰無仮説のもとで統計量の標準偏差 {np.std(zz):.3f}≈1")

# ---- 大問6：用量反応
tab = np.array([[10, 16, 22], [30, 24, 18]])
G = stats.chi2_contingency(tab, correction=False, lambda_="log-likelihood")[0]
E.choice("s05p06a", f"${f(G, 2)}$", "scipy chi2_contingency(lambda_='log-likelihood')")
E.check("s05p06a", all(abs(math.log(a) - b) < 5e-5 for a, b in ((0.625, -0.4700), (1.375, 0.3185), (1.25, 0.2231), (0.75, -0.2877))), "示した対数の値")
distinct_rounded("s05p06a")
zs = stats.norm.ppf([0.3085, 0.9332])
slope, icpt = np.polyfit(np.log([4, 16]), zs, 1)
ed50 = math.exp(-icpt / slope)
E.choice("s05p06b", f"${f(ed50, 2)}$", f"Φ^-1 を scipy で求め log 用量に直線（z={zs.round(4)}）")
distinct_rounded("s05p06b")
Cc = np.array([1, 2, 4, 8, 16, 32.0])
Ee = 20 * Cc / (8 + Cc)
bb_, aa_ = np.polyfit(1 / Cc, 1 / Ee, 1)
E.check("s05p06c", abs(aa_ - 0.05) < 1e-9 and abs(bb_ - 0.40) < 1e-9, "Emax=20, EC50=8 のデータの逆数回帰が切片0.05・傾き0.40")
fit_ = optimize.curve_fit(lambda c_, em, ec: em * c_ / (ec + c_), Cc, Ee, p0=[10, 3])[0]
E.choice("s05p06c", f"${f(fit_[1], 1)}$", "非線形最小二乗で EC50 を推定")
distinct_rounded("s05p06c")

# ---- 大問7：連続時間マルコフ連鎖（生成行列の行列指数関数）
Q = np.array([[-0.2, 0.2], [0.6, -0.6]])  # 状態0=寛解, 1=再燃
Pt = linalg.expm(2 * Q)
E.num("s05p07a", rnd(Pt[1, 1], 3), "scipy.linalg.expm(2Q) の (再燃, 再燃) 成分")
E.check("s05p07a", abs(math.exp(-1.6) - 0.2019) < 5e-5 and abs(math.exp(-1.2) - 0.3012) < 5e-5, "e^-1.6, e^-1.2 の値")

# ---- 大問8：回帰
it = E.item("s05p08a")
E.check("s05p08a", "0.7 日" in it["choices"][it["answer"]] and abs((1.5 - 0.8) - 0.7) < 1e-12, "正解は β2-β1=0.7 の記述")
n_ = 200
grp = rng.integers(0, 3, n_)
age = rng.uniform(30, 80, n_)
yy = 2 + 0.8 * (grp == 1) + 1.5 * (grp == 2) + 0.1 * age + rng.normal(0, 1, n_)
X1 = np.c_[np.ones(n_), grp == 1, grp == 2, age]
X2 = np.c_[np.ones(n_), grp == 0, grp == 1, age]  # 基準を現在喫煙に変更
Xall = np.c_[np.ones(n_), grp == 0, grp == 1, grp == 2, age]
Xnum = np.c_[np.ones(n_), grp, age]
c1 = np.linalg.lstsq(X1, yy, rcond=None)[0]
c2 = np.linalg.lstsq(X2, yy, rcond=None)[0]
E.check("s05p08a", abs(c1[3] - c2[3]) < 1e-9 and np.allclose(X1 @ c1, X2 @ c2), "基準を変えても年齢の係数と当てはめ値は同じ（基準を変える記述は誤り）")
E.check("s05p08a", np.linalg.matrix_rank(Xall) == 4, "3つのダミー+切片は階数落ち（全ダミーの記述は誤り）")
rn = np.linalg.lstsq(Xnum, yy, rcond=None)[0]
E.check("s05p08a", not np.allclose(Xnum @ rn, X1 @ c1), "0,1,2 の数値化は当てはめが変わる（数値化の記述は誤り）")
E.check("s05p08b", abs(stats.norm.cdf(1) - 0.8413) < 5e-5 and abs(stats.norm.pdf(1) - 0.2420) < 5e-5, "Φ(1), φ(1) の値")
Ey = (integrate.quad(lambda v: 1.0 * stats.norm.pdf(v, 2, 1), -np.inf, 1.0)[0]
      + integrate.quad(lambda v: v * stats.norm.pdf(v, 2, 1), 1.0, np.inf)[0])
E.choice("s05p08b", f"${f(Ey, 3)}$", "E[max(Y*,L)] を数値積分")
ys = np.maximum(rng.normal(2, 1, 2_000_000), 1.0)
E.check("s05p08b", abs(ys.mean() - Ey) < 0.003, f"シミュレーション {ys.mean():.4f}")
distinct_rounded("s05p08b")


# ---- 大問9：DAG（線形ガウス SEM の偏相関で条件付き独立を判定）
def pcor(Sig, i, j, given):
    idx = [i, j] + list(given)
    K = np.linalg.inv(Sig[np.ix_(idx, idx)])
    return -K[0, 1] / math.sqrt(K[0, 0] * K[1, 1])


Gi, Si, Ti, Ci = 0, 1, 2, 3
rows = []
for _ in range(20):
    a, b, cc, dd = rng.uniform(0.5, 2, 4) * rng.choice([-1, 1], 4)
    B = np.zeros((4, 4))
    B[Si, Gi], B[Ti, Si], B[Ci, Ti], B[Ci, Gi] = a, b, cc, dd
    A = np.linalg.inv(np.eye(4) - B)
    Sig = A @ A.T
    rows.append({"G⊥T|S": abs(pcor(Sig, Gi, Ti, [Si])) < 1e-9, "S⊥C|G": abs(pcor(Sig, Si, Ci, [Gi])) < 1e-9,
                 "G⊥T": abs(pcor(Sig, Gi, Ti, [])) < 1e-9, "G⊥T|S,C": abs(pcor(Sig, Gi, Ti, [Si, Ci])) < 1e-9,
                 "非独立 S,C|G,T": not abs(pcor(Sig, Si, Ci, [Gi, Ti])) < 1e-9})
truth = {k: all(r[k] for r in rows) for k in rows[0]}
it = E.item("s05p09a")
E.check("s05p09a", sum(truth.values()) == 1 and truth["G⊥T|S"], f"真の記述は G⊥T|S の1つだけ: {truth}")
E.check("s05p09a", "$G$ と $T$ は、$S$ を与えると" in it["choices"][it["answer"]], "正解の選択肢が G⊥T|S")

# ---- 大問10：BIBD（ファノ平面の 7 ブロックを 2 回繰り返した 14 ブロックで数える）
fano = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)]
blocks = fano + fano
rep = [sum(t in bl for bl in blocks) for t in range(7)]
lam_ = {pr: sum(set(pr) <= set(bl) for bl in blocks) for pr in combinations(range(7), 2)}
E.check("s05p10a", len(set(rep)) == 1 and len(set(lam_.values())) == 1, f"実際の BIBD で r={set(rep)}, λ={set(lam_.values())}")
E.choice("s05p10a", f"r={rep[0]},\\ \\lambda={list(lam_.values())[0]}$", "実際の計画で数えた (r, λ)")

# ---- 大問11：調査
M = np.array([200, 120, 300, 80])
cnt_ = np.array([3, 2, 4, 1])
E.num("s05p11a", rnd(40 / 4 * np.sum(M * cnt_ / 20), 0), "numpy で (N/n)ΣM_i ȳ_i")
popM = rng.integers(50, 400, 40)
popP = rng.uniform(0.02, 0.25, 40)
pop = [rng.random(m_) < p_ for m_, p_ in zip(popM, popP)]
true_tot = sum(p_.sum() for p_ in pop)
ests = []
for _ in range(20000):
    hs = rng.choice(40, 4, replace=False)
    ests.append(10 * sum(popM[h] * pop[h][rng.choice(popM[h], 20, replace=False)].mean() for h in hs))
E.check("s05p11a", abs(np.mean(ests) / true_tot - 1) < 0.02, f"二段抽出推定量の平均/真値 = {np.mean(ests) / true_tot:.4f}")
P3 = np.array([[40, 22.0], [46, 22.5], [41, 25.5]])
Z = P3 / np.array([10.0, 2.5])
dAB = np.linalg.norm(Z[0] - Z[1])
dAC = np.linalg.norm(Z[0] - Z[2])
E.check("s05p11b", dAB < dAC and np.linalg.norm(P3[0] - P3[2]) < np.linalg.norm(P3[0] - P3[1]), "標準化前は C、標準化後は B が近い")
E.choice("s05p11b", f"B（距離 ${f(dAB, 2)}$）", "numpy で標準化後の距離")
dcfg = np.array([1.0, 1.8, 1.6, 2.6, 3.2, 2.8])
dhat = optimize.isotonic_regression(dcfg).x
stress = math.sqrt(((dcfg - dhat) ** 2).sum() / (dcfg ** 2).sum())
E.choice("s05p11c", f"${f(stress, 3)}$", f"scipy isotonic_regression による単調回帰 {dhat.round(3)}")
distinct_rounded("s05p11c")

# ---- 大問12：カルマンフィルタ（事前分布と観測の精度による重み付けとして計算）
prior_m, prior_v = 200.0, 75.0 + 25.0
obs, obs_v = 240.0, 60.0
post = (prior_m / prior_v + obs / obs_v) / (1 / prior_v + 1 / obs_v)
E.num("s05p12a", rnd(post, 1), "正規事前×正規尤度の事後平均として")

e_ = np.array([1, 2, 1, -1, -2, -1, 1, -1.0])
E.num("s05p12b", rnd(np.sum(np.diff(e_) ** 2) / np.sum(e_ ** 2), 2), "numpy の diff で DW 比")
E.check("s05p12b", abs(e_.sum()) < 1e-12, "残差の和が 0（切片ありの回帰と整合）")

# ---- 大問13：臨床研究
m0 = stats.binom.pmf(5, 10, 0.2)
m1 = integrate.quad(lambda p: stats.binom.pmf(5, 10, p), 0, 1)[0]
E.choice("s05p13a", f"${f(m1 / m0, 2)}$", "scipy の二項確率と数値積分で周辺尤度の比")
ps_ = sp.Symbol("p")
E.check("s05p13a", sp.integrate(ps_ ** 5 * (1 - ps_) ** 5, (ps_, 0, 1)) == sp.Rational(1, 2772), "∫p^5(1-p)^5dp = 1/2772")
distinct_rounded("s05p13a")


kl = integrate.quad(lambda x_: stats.norm.pdf(x_, 130, 15) * (stats.norm.logpdf(x_, 130, 15) - stats.norm.logpdf(x_, 125, 12)), -np.inf, np.inf)[0]
E.choice("s05p13b", f"${f(kl, 3)}$", "log p − log q を p で数値積分")
xs_ = rng.normal(130, 15, 2_000_000)
E.check("s05p13b", abs(np.mean(stats.norm.logpdf(xs_, 130, 15) - stats.norm.logpdf(xs_, 125, 12)) - kl) < 0.003, "モンテカルロ積分")
E.check("s05p13b", abs(math.log(0.8) + 0.2231) < 5e-5, "log 0.8 の値")
distinct_rounded("s05p13b")
pv = [0.035, 0.002, 0.21, 0.019, 0.047, 0.009, 0.031]
ps = np.sort(pv)
m = len(pv)
adj = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
E.choice("s05p13c", f"{int((adj <= 0.05).sum())} 個", "BH の調整済み p 値（後ろからの累積最小）で数える")
holm = np.maximum.accumulate(np.minimum(1, ps * (m - np.arange(m))))
E.check("s05p13c", int((holm <= 0.05).sum()) == 1 and sum(p <= 0.05 / m for p in pv) == 1, "ホルム法・ボンフェローニ法はともに1個（誤答）")

E.done()
