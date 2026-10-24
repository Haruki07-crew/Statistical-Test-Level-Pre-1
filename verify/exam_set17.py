"""模擬試験 第17回（set17）の検算。generator/set17.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・数値積分・数値最適化）で正解を求め直す。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(17)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def distinct_rounded(iid, d):
    v = nums(iid)
    E.check(iid, len({f(x, d) for x in v}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {v}")


def ans_text(iid):
    it = E.item(iid)
    return it["choices"][it["answer"]]


# ---- 大問2
W = stats.weibull_min(2, scale=90)
E.num("s17p02a", rnd(W.ppf(0.75) - W.ppf(0.25), 1), "scipy weibull_min の ppf")
E.check("s17p02a", f(np.log(0.75), 4) == "-0.2877" and f(np.log(0.25), 4) == "-1.3863", "対数の表")
xs = np.array([1, 2, 3, 4, 5])
ps = np.array([0.10, 0.50, 0.15, 0.20, 0.05])
q1 = np.full(5, 0.2)
q2 = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
k1, k2 = special.rel_entr(ps, q1).sum(), special.rel_entr(ps, q2).sum()
E.check("s17p02b", k2 < k1, f"D(p||q1)={k1:.4f} > D(p||q2)={k2:.4f}")
E.choice("s17p02b", f"$q_2$（$D={f(k2, 3)}$）", "rel_entr で KL")
E.check("s17p02b", f(special.rel_entr(q2, ps).sum(), 3) == "0.107", "逆向きの KL の誤答は 0.107")
lnv = {0.25: -1.3863, 0.5: -0.6931, 0.75: -0.2877, 1.25: 0.2231, 2: 0.6931, 2.5: 0.9163}
E.check("s17p02b", all(f(np.log(u), 4) == f(v, 4) for u, v in lnv.items()), "対数の表")
obs = np.array([4, 20, 6, 8, 2])
ex = 40 * q2
om, em = np.r_[obs[:3], obs[3:].sum()], np.r_[ex[:3], ex[3:].sum()]
res = stats.chisquare(om, em)
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）

# ---- 大問3
over = stats.binom.cdf(2, 103, 0.05)
E.num("s17p03a", rnd(over, 3), "binom.cdf(2, 103, 0.05)")
E.check("s17p03a", f(0.95 ** 101, 6) == "0.005625", "0.95^101 の表示値")
# (b) ベルヌーイの 3 次絶対中心モーメントを定義から（scipy の分布で期待値）
B = stats.bernoulli(0.05)
rho = B.expect(lambda x: abs(x - 0.05) ** 3)
bound = 0.4748 * rho / (B.std() ** 3 * np.sqrt(103))
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）
sup = max(abs(stats.binom.cdf(k, 103, 0.05) - stats.norm.cdf((k - 5.15) / np.sqrt(103 * 0.0475))) for k in range(0, 30))
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）
# (c) ロジスティック回帰を数値最適化で当てはめ、情報行列から標準誤差
D = np.r_[np.ones(400), np.zeros(200)]
y = np.r_[np.ones(32), np.zeros(368), np.ones(6), np.zeros(194)]
X = np.column_stack([np.ones(600), D])
nll = lambda b: -(y * (X @ b) - np.log1p(np.exp(X @ b))).sum()
b = optimize.minimize(nll, np.zeros(2), method="BFGS", options={"gtol": 1e-10}).x
pr = special.expit(X @ b)
Info = X.T @ (X * (pr * (1 - pr))[:, None])
se1 = np.sqrt(np.linalg.inv(Info)[1, 1])
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）

# ---- 大問4
sig = 0.6
Y = np.exp(rng.normal(0, sig, size=2_000_000))
E.choice("s17p04a", f"${f(stats.norm.sf(sig / 2), 3)}$", "norm.sf(σ/2)")
E.check("s17p04a", abs((Y > np.exp(sig ** 2 / 2)).mean() - stats.norm.sf(0.3)) < 0.002, "シミュレーション")
E.check("s17p04a", all(f(stats.norm.cdf(z), 4) == v for z, v in [(0.18, "0.5714"), (0.30, "0.6179"), (0.60, "0.7257")]), "Φ の表")
z = (8.21 - 8.00) / (0.6 / 5)
pz = 2 * stats.norm.sf(z)
E.choice("s17p04b", f"$z={f(z, 2)}$、両側$p$値は{f(pz, 3)}で0.05を超えるので、帰無仮説を棄却しない", "scipy の正規分布で p 値")
E.check("s17p04b", f(stats.norm.cdf(0.35), 4) == "0.6368" and f(stats.norm.cdf(1.75), 4) == "0.9599", "Φ の表")
xsamp = rng.normal(size=10)
xsamp = (xsamp - xsamp.mean()) / xsamp.std(ddof=1) * np.sqrt(0.30) + 7.95
ci = stats.t.interval(0.95, 9, loc=xsamp.mean(), scale=stats.sem(xsamp))
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）

# ---- 大問5
th = sp.symbols("theta", real=True)
pth = sp.exp(th) / (1 + sp.exp(th))
dens = sp.simplify(sp.diff(1 - (1 - pth) ** 2, th))
target = 2 * sp.exp(th) / (1 + sp.exp(th)) ** 3
E.check("s17p05a", sp.simplify(dens - target) == 0, "分布関数を θ で微分した密度と一致")
smp = special.logit(rng.beta(1, 2, size=400_000))
hist, edges = np.histogram(smp, bins=40, range=(-4, 2))
mid = (edges[:-1] + edges[1:]) / 2
emp = hist / (len(smp) * (edges[1] - edges[0]))
E.check("s17p05a", np.max(np.abs(emp - sp.lambdify(th, target)(mid))) < 0.01, "シミュレーションのヒストグラムと一致")
E.choice("s17p05a", r"$\dfrac{2e^{\theta}}{(1+e^{\theta})^{3}}$", "正しい式の選択肢")
logpost = lambda t_: -(t_ - 3 * np.log1p(np.exp(t_)) + 7 * t_ - 20 * np.log1p(np.exp(t_)))
th_hat = optimize.minimize_scalar(logpost, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
E.num("s17p05b", rnd(special.expit(th_hat), 3), "θ の事後密度を数値最大化して p に戻す")
E.check("s17p05b", f((8 - 1) / (23 - 2), 3) != f(special.expit(th_hat), 3), "p の尺度の MAP とは異なる")

# ---- 大問8
tt = (-1.35 + 1) / 0.14
E.choice("s17p08a", f"$t={f(tt, 2)}$で$|t|>2.010$なので、弾力性は$-1$と異なり", "t を直接")
E.check("s17p08a", f(stats.t.isf(0.025, 49), 3) == "2.010" and 2 * stats.t.sf(abs(tt), 49) < 0.05, "t 点と p 値")
# (b) 不均一分散のシミュレーション：OLS は不偏だが通常の SE は過小
xx = np.linspace(1, 10, 60)
bs, ses = [], []
for _ in range(3000):
    yy = 1 + 2 * xx + rng.normal(scale=0.3 * xx ** 1.5)
    Xm = np.column_stack([np.ones(60), xx])
    bb, rss, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
    bs.append(bb[1])
    ses.append(np.sqrt(rss[0] / 58 * np.linalg.inv(Xm.T @ Xm)[1, 1]))
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）

# ---- 大問9
# 中心化移動平均を numpy の畳み込み（重み 1/8,1/4,1/4,1/4,1/8）で
yq = np.array([40, 60, 80, 50, 44, 64, 84, 54], dtype=float)
cma = np.convolve(yq, np.array([1, 2, 2, 2, 1]) / 8, mode="valid")    # 第3四半期から始まる
E.choice("s17p09a", f"${f(yq[2] / cma[0], 2)}$", f"畳み込みのトレンド {cma[0]:.2f}")
E.check("s17p09a", abs(cma[0] - (yq[0:4].mean() + yq[1:5].mean()) / 2) < 1e-12, "2 つの 4 項平均の平均と一致")
distinct_rounded("s17p09a", 2)
relu = lambda u: np.maximum(0, u)
x_in = np.array([1.0, 1.0])
W1 = np.array([[1.0, -2.0], [0.5, 3.0]])
b1 = np.array([0.5, -1.0])
yhat = 2.0 + np.array([1.5, 2.0]) @ relu(W1 @ x_in + b1)
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）

# ---- 大問13
A6 = rng.normal(size=(6, 6))
S6 = A6 @ A6.T
best = max(-optimize.minimize(lambda a: -(a @ S6 @ a) / (a @ a), rng.normal(size=6)).fun for _ in range(5))
E.check("s17p13a", abs(best - np.linalg.eigvalsh(S6).max()) < 1e-6, "制約つき最大値＝最大固有値（数値最適化）")
E.check("s17p13a", "最大固有値 $\\lambda_1$" in ans_text("s17p13a"), "正解の記述")
LA = np.array([[0.58, 0.56], [0.56, 0.42], [-0.28, 0.54]])
LB = np.array([[0.80, 0.10], [0.70, 0.00], [0.10, 0.60]])
T = np.linalg.lstsq(LA, LB, rcond=None)[0]
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）
vm = lambda L: (L ** 2).var(axis=0).sum()
pass  # 作り替え前の小問の検算（2026-10-24 に作り替えたので削除）
# (c) 実際に指示行列を作って対応分析の総イナーシャと非零特異値の数を求める
resp = np.column_stack([rng.integers(0, 3, 500), rng.integers(0, 4, 500), rng.integers(0, 2, 500)])
Z = np.hstack([np.eye(k)[resp[:, j]] for j, k in enumerate((3, 4, 2))])
Pm = Z / Z.sum()
rm, cm = Pm.sum(1), Pm.sum(0)
Sres = (Pm - np.outer(rm, cm)) / np.sqrt(np.outer(rm, cm))
sv = np.linalg.svd(Sres, compute_uv=False)
E.check("s17p13c", abs((sv ** 2).sum() - 2.0) < 1e-9 and (sv > 1e-9).sum() == 6, f"総イナーシャ {(sv ** 2).sum():.4f}、非零の次元 {(sv > 1e-9).sum()}")
E.choice("s17p13c", "総イナーシャ$2.00$、次元数6", "指示行列の SVD")

# ---- 2026-10-24 夕方に作り直した小問の検算（generator とは別の方法）
from itertools import product as _prod
from fractions import Fraction as Fr
# 1(a) 3 部屋の当たりと客の選択とフロント係の選択を全列挙
win = tot = Fr(0)
for car, pick in _prod(range(3), range(3)):
    opts = [d for d in range(3) if d != pick and d != car]
    for o in opts:
        pr = Fr(1, 9) * Fr(1, len(opts))
        sw = [d for d in range(3) if d not in (pick, o)][0]
        tot += pr; win += pr * (sw == car)
E.choice("s17p01a", f"${win.numerator}/{win.denominator}$", f"全列挙で P(変えて当たる)={win}")
# 2(c) scipy の chi2_contingency
O2 = np.array([[30, 45, 25], [20, 15, 15]])
c2 = stats.chi2_contingency(O2, correction=False)
E.choice("s17p02c", f"$X^2={f(c2.statistic, 2)}$（自由度 {c2.dof}）で 5.991 を下回るので独立性は棄却されない。$V={f(math.sqrt(c2.statistic / 150), 3)}$", f"X²={c2.statistic:.4f}, p={c2.pvalue:.3f}")
# 3(b) scipy の poisson
E.num("s17p03b", rnd(stats.poisson.cdf(2, 5.15), 3), f"poisson.cdf(2, 5.15)={stats.poisson.cdf(2, 5.15):.5f}")
E.check("s17p03b", abs(math.exp(-5.15) - 0.005799) < 5e-7, "e^-5.15")
# 3(c) 効用を逆算して選択肢を減らした多項ロジットの確率
Vv = np.log(np.array([0.5, 0.3, 0.2]))
p2_ = np.exp(Vv[:2]) / np.exp(Vv[:2]).sum()
E.choice("s17p03c", f"$({f(p2_[0], 3)},\\ {f(p2_[1], 3)})$", "効用 V=log P を使って 2 択で再計算")
# 4(c) ポアソン過程をシミュレーションして率の差の推定量の標準偏差を確かめる
nX = rng.poisson(7 * 6, 200000) / 6; nY = rng.poisson(3 * 6, 200000) / 6
E.check("s17p04c", abs((nX - nY).std() - math.sqrt(10 / 6)) < 0.01, f"率の差の標準偏差 {(nX - nY).std():.4f} ≈ √(10/6)")
E.num("s17p04c", rnd(4.0 - stats.norm.isf(0.025) * math.sqrt(42 / 36 + 18 / 36), 2), "scipy の正規分布の点")
# 6(a) 定常分布を固有ベクトルで求め、詳細つり合いと閉路の積を確認
P6 = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
w6, V6 = np.linalg.eig(P6.T); pi6 = np.real(V6[:, np.argmin(abs(w6 - 1))]); pi6 /= pi6.sum()
F6 = pi6[:, None] * P6
E.check("s17p06a", np.allclose(pi6, [0.38, 0.34, 0.28]) and not np.allclose(F6, F6.T), f"定常分布 {np.round(pi6, 4)}、流れの行列は非対称")
E.check("s17p06a", F6[0, 1] - F6[1, 0] > 0 and F6[1, 2] - F6[2, 1] > 0 and F6[2, 0] - F6[0, 2] > 0, "正味の流れ A→B、B→C、C→A")
E.choice("s17p06a", f"$\\pi_Ap_{{AB}}={f(F6[0, 1], 3)}$、$\\pi_Bp_{{BA}}={f(F6[1, 0], 3)}$ で詳細つり合いが成り立たず可逆ではない", "固有ベクトル")
# 7(a) 生命表を表から読み取って計算
q7 = E.item("s17p07a")["question"]
rows7 = [[int(x) for x in m_.split("|")] for m_ in re.findall(r"\| \d 年目 \|([^\n]*)\|", q7)]
S7 = np.prod([1 - d_ / (n_ - w_ / 2) for n_, d_, w_ in rows7])
E.check("s17p07a", len(rows7) == 3 and all(rows7[k + 1][0] == rows7[k][0] - rows7[k][1] - rows7[k][2] for k in range(2)), "年初の人数のつながり")
E.num("s17p07a", rnd(S7, 3), f"表から計算 {S7:.5f}")
# 7(b) 有限母集団修正つきの半幅を n で直接探索
hw = lambda n_: 1.96 * math.sqrt((2000 - n_) / (2000 - 1) * 0.25 / n_)
E.num("s17p07b", next(n_ for n_ in range(1, 2000) if hw(n_) <= 0.05), "半幅 ≤ 0.05 の最小の n（(N−n)/(N−1) を使う）")
# 8(b) ダミーだけの計画行列でてこ比
X8 = np.c_[np.ones(52), np.r_[np.ones(4), np.zeros(48)]]
H8 = np.diag(X8 @ np.linalg.inv(X8.T @ X8) @ X8.T)
E.choice("s17p08b", f"$h_{{ii}}={f(H8[0], 3)}$ で、目安 {f(4 / 52, 3)} を上回る", f"ハット行列の対角 {H8[0]:.4f}・{H8[-1]:.4f}")
# 9(b) 同時方程式のデータを作り、2 段階最小二乗が真の弾力性に近く OLS が 0 の方に偏ることを確かめる
n9 = 200000
z1, z2 = rng.normal(size=n9), rng.normal(size=n9)
u = rng.normal(size=n9)
lp = 0.1 * z1 + 0.1 * z2 + 0.08 * u + rng.normal(0, 0.05, n9)     # 需要ショック u が高いと価格も高い
lq = -1.3 * lp + 0.3 * u
Z9 = np.c_[np.ones(n9), z1, z2]
lp_hat = Z9 @ np.linalg.lstsq(Z9, lp, rcond=None)[0]
b2sls = np.cov(lp_hat, lq)[0, 1] / np.var(lp_hat, ddof=1)
bols = np.cov(lp, lq)[0, 1] / np.var(lp, ddof=1)
E.check("s17p09b", abs(b2sls + 1.3) < 0.02 and -1.3 < bols < 0, f"2SLS {b2sls:.3f}、OLS {bols:.3f}（0 の方に偏る）")
E.choice("s17p09b", f"${f(-0.052 / 0.040, 2)}$。需要の多い週に価格を上げる影響で", "与えた値での比")
# 10(a) ラプラス分布の対数尤度を数値最大化
x10 = np.array([-3, -1, 0, 2, 4, 7, 15.])
nll10 = lambda p_: -np.sum(stats.laplace.logpdf(x10, p_[0], np.exp(p_[1])))
o10 = optimize.minimize(nll10, [x10.mean(), 1.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
E.choice("s17p10a", f"$(\\hat\\mu,\\hat b)=({f(o10.x[0], 1)},\\ {f(np.exp(o10.x[1]), 2)})$", f"数値最大化 μ={o10.x[0]:.4f}, b={np.exp(o10.x[1]):.4f}")
# 11(a) scipy の binomtest（両側は「確率が観測値以下の値の和」。p=0.5 では 2 倍と一致）
bt = stats.binomtest(10, 12, 0.5)
E.choice("s17p11a", f"両側 $p$ 値は {f(bt.pvalue, 3)} で、0.05 を下回るので帰無仮説を棄却する", f"binomtest p={bt.pvalue:.5f}")
# 12(a) 入れ子のデータを作り、期待平均平方の関係を確かめる
sims = []
for _ in range(4000):
    hotel = rng.normal(0, math.sqrt(1.2), (3, 4, 1)); y12 = hotel + rng.normal(0, 2, (3, 4, 5))
    ms_b = 5 * ((y12.mean(2) - y12.mean((1, 2))[:, None]) ** 2).sum() / 9
    ms_e = ((y12 - y12.mean(2, keepdims=True)) ** 2).sum() / 48
    sims.append((ms_b, ms_e))
mb, me = np.mean(sims, 0)
E.check("s17p12a", abs(mb - (4 + 5 * 1.2)) < 0.3 and abs(me - 4) < 0.05, f"E[MS_B]≈σ²+5σ_B²={mb:.3f}、E[MS_E]≈σ²={me:.3f}")
E.choice("s17p12a", f"$F={f(36 / 10, 2)}$ で {f(stats.f.isf(0.05, 2, 9), 3)} を下回り、チェーン間の差は有意でない。$\\hat\\sigma_B^2={f((10 - 4) / 5, 2)}$", "F 点と分散成分")
E.check("s17p12a", abs(stats.f.isf(0.05, 2, 48) - 3.191) < 5e-4, "F_0.05(2,48)")
# 13(b) 相関行列を作って 1 因子の解を連立方程式で
l_ = sp.symbols("l1:4", positive=True)
sol13 = sp.solve([l_[0] * l_[1] - sp.Rational(8, 10), l_[0] * l_[2] - sp.Rational(7, 10), l_[1] * l_[2] - sp.Rational(5, 10)], l_, dict=True)[0]
E.choice("s17p13b", f"$l_1^2={f(float(sol13[l_[0]] ** 2), 2)}$ となり、独自分散 $1-l_1^2={f(1 - float(sol13[l_[0]] ** 2), 2)}$ が負", f"sympy {sol13}")

E.done()
