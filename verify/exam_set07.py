"""模擬試験 第7回（set07）の検算。generator/set07.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・反復計算・数値最適化）で正解を求め直し、問題文に示した表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(7)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table(iid, given, true, tol, what):
    E.check(iid, abs(given - true) <= tol, f"問題文の値 {what}={given}（正確な値 {true:.6f}）")


# ---- 大問1 家財保険
# (a) 2件の不正の有無を総当たり
p = [0.10, 0.20]
pr = {o: np.prod([p[i] if o[i] else 1 - p[i] for i in range(2)]) for o in product([0, 1], repeat=2)}
atl = sum(v for o, v in pr.items() if sum(o) >= 1)
E.num("s07p01a", rnd(pr[(1, 1)] / atl, 3), "4通りの総当たりで P(両方|少なくとも1件)")
sim = rng.random((1_000_000, 2)) < np.array(p)
sel = sim.any(1)
E.check("s07p01a", abs(sim[sel].all(1).mean() - 0.0714) < 0.002, f"シミュレーション {sim[sel].all(1).mean():.4f}")
# (b) 混合分布の上側確率：sympy で M_S と混合（0 と Gamma(2,尺度4)）の積率母関数が一致することを確かめ、scipy の gamma.sf で
t = sp.symbols("t")
M = sp.Rational(4, 5) + sp.Rational(1, 5) * (1 - 4 * t) ** -2
u = sp.symbols("u", positive=True)
mgf_g = sp.simplify(sp.integrate(sp.exp(t * u) * u * sp.exp(-u / 4) / 16, (u, 0, sp.oo), conds="none"))
E.check("s07p01b", sp.simplify(sp.Rational(4, 5) + sp.Rational(1, 5) * mgf_g - M) == 0, f"Gamma(2,尺度4) の積率母関数 {mgf_g}")
g = stats.gamma(2, scale=4)
E.choice("s07p01b", f"${f(0.2 * g.sf(8), 4)}$", f"scipy gamma.sf で 0.2×P(G>8)（正確な値 {0.2 * g.sf(8):.5f}）")
Ssim = np.where(rng.random(1_000_000) < 0.2, rng.gamma(2, 4, 1_000_000), 0.0)
E.check("s07p01b", abs((Ssim > 8).mean() - 0.0812) < 0.001, f"シミュレーション P(S>8)={(Ssim > 8).mean():.4f}")
table("s07p01b", 0.1353, np.exp(-2), 0.00005, "e^-2")
m1 = sp.diff(M, t).subs(t, 0)
m2 = sp.diff(M, t, 2).subs(t, 0)
E.check("s07p01c", (float(m1), float(m2 - m1 ** 2)) == (1.6, 16.64), "sympy で M'(0)=1.6、M''(0)-M'(0)^2=16.64（問題文の値）")
# (c) 正確なパーセント点で保険料／支払総額のシミュレーションで超過確率 ≈ 1%
mu, sd = 1.6, np.sqrt(16.64)
P = mu + stats.norm.isf(0.01) * sd / np.sqrt(2500)
E.num("s07p01c", rnd(P, 2), "scipy norm.isf(0.01) で保険料")
K = rng.binomial(2500, 0.2, size=400_000)
T = rng.gamma(2 * K, 4)  # 支払のあった K 件の Gamma(2,4) の和
exceed = (T > 2500 * 1.79).mean()
E.check("s07p01c", 0.007 < exceed < 0.013, f"シミュレーションで P(総額>保険料収入)={exceed:.4f}（≈1%）")
table("s07p01c", 2.326, stats.norm.isf(0.01), 0.0005, "z_0.01")

# ---- 大問2 パレート
# (a) 2件の幾何平均が 2 を超える確率：数値積分（2重）とシミュレーション
par = stats.pareto(b=3, scale=1)
pg = integrate.quad(lambda x1: par.pdf(x1) * min(1.0, par.sf(4 / x1)), 1, np.inf)[0]
E.choice("s07p02a", f"${f(pg, 4)}$", f"P(X2>4/X1) を X1 について数値積分（{pg:.5f}）")
xs2 = par.rvs(size=(1_000_000, 2), random_state=7)
E.check("s07p02a", abs((np.sqrt(xs2.prod(1)) > 2).mean() - pg) < 0.001, f"シミュレーション {(np.sqrt(xs2.prod(1)) > 2).mean():.4f}")
table("s07p02a", 0.6931, np.log(2), 0.00005, "log 2")
# (b) モーメント：scipy の pareto と数値積分
mn, vr = par.mean(), par.var()
E.choice("s07p02b", f"$V[X]={f(vr, 2)}$で、変動係数は${f(np.sqrt(vr) / mn, 3)}$である。$E[X^3]$は発散", f"scipy pareto の平均 {mn} と分散 {vr}")
x3 = sp.symbols("x", positive=True)
E.check("s07p02b", sp.integrate(x3 ** 3 * 3 / x3 ** 4, (x3, 1, sp.oo)) == sp.oo, "sympy で E[X^3]=∞")
par = stats.pareto(b=3, scale=1)

# (c) 過分散：負の二項（分散が平均の 1.5 倍）からデータを発生させ、ピアソン統計量/自由度で φ を推定
mu_ = np.exp(rng.normal(1.5, 0.4, size=(300, 120)))
r_ = mu_ / 0.5                       # NB(r, p) で分散 = mu(1+mu/r) = 1.5 mu
cnt = rng.negative_binomial(r_, r_ / (r_ + mu_))
phi_sim = np.mean(((cnt - mu_) ** 2 / mu_).sum(1) / 117)
E.check("s07p02c", abs(phi_sim - 1.5 * 120 / 117) < 0.03, f"分散が平均の1.5倍のデータで X²/(n-p) ≈ {phi_sim:.3f}")
sec = 0.080 * np.sqrt(175.5 / 117)
zc = 0.24 / sec
E.check("s07p02c", stats.norm.isf(0.025) < zc < stats.norm.isf(0.005), f"補正後の z={zc:.3f} は 5% で有意・1% で非有意")
E.choice("s07p02c", f"\\sqrt{{1.50}}={f(sec, 3)}$、$z\\approx{f(zc, 2)}$", "√φ 倍の標準誤差と z を含む選択肢")
E.check("s07p02c", all("n-p" not in d_ for d_ in E.item("s07p02c")["distractors"]), "φ の割る数（n か n-p か）だけで誤りになる選択肢がない")

# ---- 大問3 運用成績
b0 = stats.binom(12, 0.5)
gam = (0.05 - b0.sf(9)) / b0.pmf(9)
E.num("s07p03a", rnd(gam, 3), "scipy binom で P(X>=10)+γP(X=9)=0.05 を解く")
E.check("s07p03a", abs(b0.sf(9) + gam * b0.pmf(9) - 0.05) < 1e-12, "有意水準がちょうど 0.05")
A = [-3.2, -1.5, 1.1, 2.8]
B = [0.4, 2.3, 5.2, 6.0]
C = [3.6, 4.5, 7.4, 8.9]
kw = stats.kruskal(A, B, C)
E.choice("s07p03b", f"H\\approx{f(kw.statistic, 2)}$で、自由度2の上側5%点5.991を超えるので帰無仮説を棄却する", "scipy.stats.kruskal")
E.check("s07p03b", kw.pvalue < 0.05, f"p 値 {kw.pvalue:.4f} < 0.05")
for q_, df_, val in [(0.95, 2, 5.991), (0.95, 3, 7.815), (0.975, 2, 7.378)]:
    table("s07p03b", val, stats.chi2.ppf(q_, df_), 0.0005, f"χ²({df_}) {q_}点")
c_ = sp.symbols("c")
mse = c_ ** 2 * sp.Rational(16, 36) + (1 - c_) ** 2 * sp.Rational(1, 4)
cs = sp.solve(sp.diff(mse, c_), c_)[0]
E.choice("s07p03c", f"$c={f(float(cs), 3)}$、そのときの MSE は ${f(float(mse.subs(c_, cs)), 4)}$", "sympy で MSE(c) を最小化")
Xb = rng.normal(0.5, 4, size=(200_000, 36)).mean(1)
E.check("s07p03c", abs(np.mean((0.36 * Xb - 0.5) ** 2) - 0.16) < 0.003, "シミュレーションで MSE(0.36) ≈ 0.16")

# ---- 大問4 融資
# (a) 二項分布の最頻値：scipy の確率関数を全部計算して最大を探す
bd = stats.binom(40, 0.06)
pm = bd.pmf(np.arange(41))
kst = int(np.argmax(pm))
E.check("s07p04a", kst == 2, f"scipy で最頻値 k*={kst}")
E.num("s07p04a", rnd(pm[kst], 3), "scipy binom.pmf の最大値")
for e_, v_ in [(37, 0.10133), (38, 0.09525), (39, 0.08953)]:
    table("s07p04a", v_, 0.94 ** e_, 0.000006, f"0.94^{e_}")
# (b) 母比率の検定：statsmodels を使わず、正確な二項分布の正規近似の標準化として求める
bn_ = stats.binom(200, 0.15)
zz = (40 - bn_.mean()) / bn_.std()
E.num("s07p04b", rnd(zz, 2), "scipy binom の平均・標準偏差で標準化")
E.check("s07p04b", stats.norm.isf(0.05) < zz < stats.norm.isf(0.01), f"z={zz:.4f} は片側 5% で有意・1% で非有意（正確な片側 p 値 {bn_.sf(39):.4f}）")
# (c) 反復比例当てはめ（IPF）で [AB][AC] モデルを当てはめる
obs = np.array([[[12, 48], [3, 37]], [[10, 70], [2, 18]]], dtype=float)  # [業種][規模][延滞]
m = np.ones_like(obs)
for _ in range(50):
    m *= (obs.sum(2) / m.sum(2))[:, :, None]
    m *= (obs.sum(1) / m.sum(1))[:, None, :]
E.choice("s07p04c", f"${f(m[0, 0, 0], 2)}$", "IPF で当てはめた期待度数")
E.check("s07p04c", np.allclose(m.sum(2), obs.sum(2)) and np.allclose(m.sum(1), obs.sum(1)), "IPF の周辺度数が観測と一致")

# ---- 大問5 株価指数
# (a) σ の信頼区間：scipy の正確なパーセント点／被覆率のシミュレーション
lo_, hi_ = np.sqrt(15 * 16 / stats.chi2.isf(0.025, 15)), np.sqrt(15 * 16 / stats.chi2.isf(0.975, 15))
E.choice("s07p05a", f"({f(lo_, 2)},\\ {f(hi_, 2)})", "scipy chi2 のパーセント点で区間")
for q_, df_, val in [(0.025, 15, 27.488), (0.975, 15, 6.262), (0.05, 15, 24.996), (0.95, 15, 7.261), (0.025, 16, 28.845), (0.975, 16, 6.908)]:
    table("s07p05a", val, stats.chi2.isf(q_, df_), 0.0005, f"χ²({df_}) 上側{q_}点")
sims = rng.normal(0, 4, size=(100_000, 16)).std(1, ddof=1)
cov = np.mean((np.sqrt(15 * sims ** 2 / 27.488) <= 4) & (4 <= np.sqrt(15 * sims ** 2 / 6.262)))
E.check("s07p05a", abs(cov - 0.95) < 0.005, f"被覆率のシミュレーション {cov:.4f}")
# (b) 分位点回帰：チェック関数を細かい格子で数値最小化
yq = np.array([1.9, -2.5, 3.4, 0.3, -4.1, 5.0, 1.2, -0.8, 2.6])
grid = np.linspace(-6, 6, 120001)
lossg = np.array([np.sum((yq - q) * (0.2 - (yq < q))) for q in grid[::10]])
qbest = grid[::10][np.argmin(lossg)]
E.choice("s07p05b", f"${f(qbest, 1)}$", f"格子探索の最小点 {qbest:.3f}")
E.check("s07p05b", abs(qbest - np.quantile(yq, 0.2, method="inverted_cdf")) < 1e-6, "numpy の逆分布関数型の 0.2 分位点と一致")
# (c) ランダムウォーク
paths = rng.normal(0.05, 1.2, size=(200_000, 25)).sum(1)
qlo, qhi = np.quantile(paths, [0.025, 0.975])
E.check("s07p05c", abs(qlo + 10.51) < 0.1 and abs(qhi - 13.01) < 0.1, f"シミュレーションの 2.5%・97.5% 点 {qlo:.2f}, {qhi:.2f}")
d = stats.norm(25 * 0.05, 1.2 * 5)
E.choice("s07p05c", f"({f(d.ppf(0.025), 2)},\\ {f(d.ppf(0.975), 2)})", "scipy norm の分位点")

# ---- 大問6 社債
Pm = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0, 0, 1]])
Q = Pm[:2, :2]
tvec = np.linalg.inv(np.eye(2) - Q) @ np.ones(2)   # 基本行列 (I-Q)^{-1} の行和
E.choice("s07p06a", f"${f(tvec[0], 2)}$", "基本行列 (I-Q)^{-1}1 で期待年数")
st = np.zeros(20_000, dtype=int)
steps = np.zeros(20_000)
alive = np.ones(20_000, dtype=bool)
while alive.any():
    u = rng.random(alive.sum())
    cum = Pm.cumsum(1)[st[alive]]
    st[alive] = (u[:, None] > cum).sum(1)
    steps[alive] += 1
    alive = st != 2
E.check("s07p06a", abs(steps.mean() - 23.33) < 0.6, f"シミュレーションの平均年数 {steps.mean():.2f}")
prior = np.array([0.80, 0.15, 0.05])
lik = np.array([0.10, 0.40, 0.80])
post = prior * lik / (prior * lik).sum()
Lm = np.array([[-4, 3, 20], [-2, 0.5, 3], [0, 0, 0]])
el = Lm @ post
E.check("s07p06b", int(np.argmin(el)) == 1, f"事後期待損失 {el.round(4)} の最小はヘッジ")
E.choice("s07p06b", f"全額投資${f(el[0], 2)}$、ヘッジつき半額投資${f(el[1], 2)}$", "事後期待損失の値を含む選択肢")
E.check("s07p06b", int(np.argmin(Lm @ prior)) == 0, "事前確率では全額投資が最小（誤答の選択肢が誤りであること）")

# ---- 大問7 中古マンション
tq_ = stats.t.isf(0.025, 56)
lo7, hi7 = np.exp(0.18 - tq_ * 0.05) - 1, np.exp(0.18 + tq_ * 0.05) - 1
E.choice("s07p07a", f"約{f(lo7 * 100, 1)}%高い〜約{f(hi7 * 100, 1)}%高い", f"scipy の t 点 {tq_:.4f} と numpy の exp で区間を変換")
table("s07p07a", 2.003, tq_, 0.0005, "t_0.025(56)")
for x_, v_ in [(0.08, 1.0833), (0.13, 1.1388), (0.18, 1.1972), (0.23, 1.2586), (0.28, 1.3231)]:
    table("s07p07a", v_, np.exp(x_), 0.00005, f"e^{x_}")
Fq = (312 / 18) / (120 / 18)
E.choice("s07p07b", f"$F={f(Fq, 2)}$で、自由度$(18,18)$の上側5%点2.217を超える", "残差平均平方の比")
E.check("s07p07b", stats.f.sf(Fq, 18, 18) < 0.05 and stats.f.sf(Fq, 18, 18) > 0.01, f"scipy の上側確率 {stats.f.sf(Fq, 18, 18):.4f}")
for val, q_, a_, b_ in [(2.217, 0.95, 18, 18), (3.128, 0.99, 18, 18), (2.124, 0.95, 20, 20)]:
    table("s07p07b", val, stats.f.ppf(q_, a_, b_), 0.0006, f"F_{1 - q_:.2f}({a_},{b_})")
# 正規線形モデルのシミュレーションで「群の自由度 = 件数 - 2」を確かめる
xs_ = np.linspace(0, 1, 20)
Xd = np.column_stack([np.ones(20), xs_])
H_ = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
ssr = np.array([np.sum(((np.eye(20) - H_) @ rng.normal(size=20)) ** 2) for _ in range(20000)])
E.check("s07p07b", abs(ssr.mean() - 18) < 0.15, f"残差平方和/σ² の平均 {ssr.mean():.2f} ≈ 18")
aic = {"A": 80 + 6, "B": 73 + 12}
aicc = {k_: aic[k_] + 2 * kk * (kk + 1) / (15 - kk - 1) for k_, kk in [("A", 3), ("B", 6)]}
E.check("s07p07c", min(aic, key=aic.get) == "B" and min(aicc, key=aicc.get) == "A", f"AIC {aic}, AICc {aicc}")
E.choice("s07p07c", f"モデルAが{f(aicc['A'], 2)}、モデルBが{f(aicc['B'], 2)}でモデルAが選ばれる", "AICc の値を含む選択肢")

# ---- 大問8 効果量：群を発生させて平方和を作るのではなく、ω² の定義（母集団の説明割合の推定）をシミュレーションで確かめる
k_, n_ = 4, 10
ssb, ssw = 180.0, 720.0
mse_ = ssw / (k_ * n_ - k_)
om = (ssb - (k_ - 1) * mse_) / (ssb + ssw + mse_)
E.num("s07p08a", rnd(om, 3), "平方和から直接 ω²")
E.check("s07p08a", "**答え** $0.130$" in E.item("s07p08a")["explanation"], "解説の答えの表示が小数第3位まで（0.130）")
true_eff = np.array([-1.0, 0.0, 0.0, 1.0]) * 1.5
pop = np.var(true_eff) / (np.var(true_eff) + 4.0)
oms, etas = [], []
for _ in range(4000):
    yv = true_eff[:, None] + rng.normal(0, 2, size=(4, 10))
    sb = 10 * np.sum((yv.mean(1) - yv.mean()) ** 2)
    sw = np.sum((yv - yv.mean(1, keepdims=True)) ** 2)
    me = sw / 36
    oms.append((sb - 3 * me) / (sb + sw + me))
    etas.append(sb / (sb + sw))
E.check("s07p08a", abs(np.mean(oms) - pop) < abs(np.mean(etas) - pop), f"母集団の割合 {pop:.3f} に対し ω² の平均 {np.mean(oms):.3f}、η² の平均 {np.mean(etas):.3f}")

# ---- 大問9 層化の効果：大きな母集団を作り、比例配分の層化抽出と単純無作為抽出の分散をシミュレーションで比べる
Nh = (40000, 10000)
popA = rng.normal(0, 1, Nh[0]); popA = 130 + 80 * (popA - popA.mean()) / popA.std()
popB = rng.normal(0, 1, Nh[1]); popB = 900 + 600 * (popB - popB.mean()) / popB.std()
allp = np.concatenate([popA, popB])
ratio_exact = (0.8 * popA.var() + 0.2 * popB.var()) / allp.var()
E.num("s07p09a", rnd(ratio_exact, 3), "作った母集団の分散（割る数 N）から直接")
R_ = 20000
srs = np.array([rng.choice(allp, 50, replace=False).mean() for _ in range(R_)])
stv = np.array([0.8 * rng.choice(popA, 40, replace=False).mean() + 0.2 * rng.choice(popB, 10, replace=False).mean() for _ in range(R_)])
E.check("s07p09a", abs(stv.var() / srs.var() - 0.448) < 0.03, f"シミュレーションの分散比 {stv.var() / srs.var():.3f}（許容 ±0.03）")
# ---- 大問10 主成分：元の共分散行列から固有分解
Sg = np.array([[16, 14, 10], [14, 20, 15], [10, 15, 18.0]])
w, V = np.linalg.eigh(Sg)
v1 = V[:, -1] * np.sign(V[0, -1])
v2 = V[:, -2] * np.sign(V[0, -2])
E.check("s07p10a", np.allclose(np.round(v1, 3), [0.517, 0.644, 0.563]) and np.allclose(np.round(w[::-1], 2), [44.34, 6.96, 2.69])
        and np.allclose(np.round(v2, 3), [0.713, 0.040, -0.700], atol=0.0011), "表の固有値・固有ベクトルが共分散行列の固有分解の丸め")
xv = np.array([6, 8, 4.0])
rec = (np.outer(v1, v1) @ xv)[2]     # 第1主成分への射影行列 v1 v1' を掛ける
E.choice("s07p10a", f"${f(rec, 1)}$", "射影行列 v1v1' による近似の10年成分（丸めない固有ベクトル）")

# ---- 大問11
def dof(pp, mm):
    def vech(theta):
        Lam = theta[:pp * mm].reshape(pp, mm)
        Sig = Lam @ Lam.T + np.diag(theta[pp * mm:])
        return Sig[np.triu_indices(pp)]
    th0 = rng.normal(size=pp * mm + pp)
    th0[pp * mm:] = np.abs(th0[pp * mm:]) + 0.5
    eps = 1e-6
    J = np.array([(vech(th0 + eps * np.eye(th0.size)[k]) - vech(th0 - eps * np.eye(th0.size)[k])) / (2 * eps) for k in range(th0.size)]).T
    return pp * (pp + 1) // 2 - np.linalg.matrix_rank(J, tol=1e-6)
dfs = {mm: int(dof(6, mm)) for mm in (1, 2, 3)}
E.check("s07p11a", dfs == {1: 9, 2: 4, 3: 0}, f"ヤコビ行列の階数から数えた自由度 {dfs}")
E.choice("s07p11a", f"自由度{dfs[2]}、最大の因子数{max(mm for mm, v_ in dfs.items() if v_ > 0)}", "正の自由度をもつ最大の m")
Nn = 400_000
Rr = rng.normal(size=Nn)
Hh = -0.5 * Rr + rng.normal(scale=np.sqrt(1 - 0.25), size=Nn)
var_e = 1 - (0.6 ** 2 + 0.2 ** 2 + 2 * 0.6 * -0.2 * -0.5)
Ee = 0.6 * Hh - 0.2 * Rr + rng.normal(scale=np.sqrt(var_e), size=Nn)
cor = np.corrcoef(Rr, Ee)[0, 1]
E.check("s07p11b", abs(cor + 0.5) < 0.005 and abs(Ee.std() - 1) < 0.005, f"シミュレーションの相関 {cor:.4f}（標準化された E）")
E.choice("s07p11b", f"${f(round(cor, 1), 2)}$", "シミュレーションの相関を丸めた値")

E.done()
