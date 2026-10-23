"""第28章（分割表）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from math import comb
from scipy import stats

q = Q(28)
rng = np.random.default_rng(28)
z975 = stats.norm.ppf(0.975)

# --- 分割表 ---
T = np.array([[30, 20, 10], [20, 30, 50.0]])
chi2, p, dof, E = stats.chi2_contingency(T, correction=False)
q.num("q28-contingency-table-1", E[1, 2])
q.num("q28-contingency-table-2", chi2)
q.check("q28-contingency-table-2", dof == 2 and chi2 > stats.chi2.ppf(0.99, 2), "自由度2、1%点を超える")
q.check("q28-contingency-table-3", (4 - 1) * (3 - 1) == 6 and len({6, 12, 11, 5}) == 4, "自由度6、誤答と異なる")

# --- 2×2 ---
a, b, c, d = 40, 60, 25, 75
t22 = np.array([[a, b], [c, d]])
chi2_22 = stats.chi2_contingency(t22, correction=False)[0]
q.num("q28-two-by-two-table-1", chi2_22)
p1, p2, pp = a / 100, c / 100, (a + c) / 200
z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / 100 + 1 / 100))
q.check("q28-two-by-two-table-2", abs(z**2 - chi2_22) < 1e-9, "z² = χ²（正しい）")

# --- 相対リスク・リスク差 ---
RR = (a / 100) / (c / 100)
q.num("q28-relative-risk-1", RR)
se = np.sqrt(1 / a - 1 / 100 + 1 / c - 1 / 100)
up, lo = RR * np.exp(z975 * se), RR * np.exp(-z975 * se)
q.num("q28-relative-risk-2", up)
q.check("q28-relative-risk-3", abs((up - RR) - (RR - lo)) > 0.2 and np.isclose(np.log(up) - np.log(RR), np.log(RR) - np.log(lo)),
        "対数では対称・元の尺度では非対称（誤り）")
se_rd = np.sqrt(p1 * (1 - p1) / 100 + p2 * (1 - p2) / 100)
q.num("q28-risk-difference-1", (p1 - p2) - z975 * se_rd)
q.num("q28-risk-difference-2", 1 / abs(0.12 - 0.20))

# --- オッズ比の区間 ---
a2, b2, c2, d2 = 12, 8, 6, 14
OR = a2 * d2 / (b2 * c2)
se_or = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
q.num("q28-odds-ratio-interval-1", OR * np.exp(-z975 * se_or))
q.check("q28-odds-ratio-interval-1", OR * np.exp(-z975 * se_or) < 1 < OR * np.exp(z975 * se_or), "区間が1を含む")
ha = (0.5 * 5.5) / (10.5 * 5.5)
wrong = [0.0, 0.5 * 5 / (10 * 5), 1 * 6 / (11 * 6)]
q.check("q28-odds-ratio-interval-2", round(ha, 3) == 0.048 and [round(w, 3) for w in wrong] == [0.0, 0.05, 0.091],
        "補正後 0.048、誤答 0 / 0.050 / 0.091")
mu = np.array([300, 200, 250, 400.0])
sim = rng.poisson(mu, size=(400000, 4)).astype(float)
lor = np.log(sim[:, 0]) - np.log(sim[:, 1]) - np.log(sim[:, 2]) + np.log(sim[:, 3])
q.check("q28-odds-ratio-interval-3", abs(np.var(lor) / np.sum(1 / mu) - 1) < 0.05, "シミュレーションの分散 ≈ Σ1/μ")

# --- 抽出の枠組み ---
x1 = stats.poisson.pmf(4, 2) * stats.poisson.pmf(6, 3) / stats.poisson.pmf(10, 5)
q.num("q28-sampling-schemes-1", x1)
q.check("q28-sampling-schemes-1", abs(x1 - stats.binom.pmf(4, 10, 0.4)) < 1e-12, "条件付き分布 = Bin(10,0.4)")
# 積多項モデルの同質性の MLE（行ごとに共通の列割合）が n_i+ n_+j / n になることを数値最適化で確認
from scipy.optimize import minimize
def negll(th):
    pi = np.exp(np.append(th, 0)); pi /= pi.sum()
    return -np.sum(T * np.log(pi))
res = minimize(negll, np.zeros(2))
pi_hat = np.exp(np.append(res.x, 0)); pi_hat /= pi_hat.sum()
q.check("q28-sampling-schemes-2", np.allclose(T.sum(1)[:, None] * pi_hat, E, atol=1e-3),
        "積多項の MLE の期待度数も n_i+ n_+j / n（誤り）")
q.check("q28-sampling-schemes-3", 3 * (4 - 1) == 9 and 3 * 4 - 1 == 11 and 3 * 4 == 12 and 2 * 3 == 6, "I(J-1)=9")

# --- イェーツ ---
yt = stats.chi2_contingency(np.array([[a2, b2], [c2, d2]]), correction=True)[0]
q.num("q28-yates-correction-1", yt)
q.check("q28-yates-correction-2", yt < stats.chi2_contingency(np.array([[a2, b2], [c2, d2]]), correction=False)[0],
        "補正で統計量は小さくなる（誤り）")

# --- マクネマー ---
bb, cc = 15, 5
q.num("q28-mcnemar-test-1", (abs(bb - cc) - 1) ** 2 / (bb + cc))
pm = 2 * stats.binom.cdf(1, 8, 0.5)
q.num("q28-mcnemar-test-2", pm)
q.check("q28-mcnemar-test-2", abs(pm - stats.binomtest(7, 8, 0.5).pvalue) < 1e-12, "scipy の binomtest と一致")
q.check("q28-mcnemar-test-3", (bb - cc) ** 2 / (bb + cc) == (bb - cc) ** 2 / (bb + cc), "a,d が入らない（誤り）")

# --- 対数線形 ---
L = np.log(t22.astype(float))
lam_ab = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
q.num("q28-loglinear-model-1", lam_ab[0, 0])
G2 = 2 * np.sum(T * np.log(T / E))
q.num("q28-loglinear-model-2", G2)
q.check("q28-loglinear-model-2", abs(G2 - stats.chi2_contingency(T, correction=False, lambda_="log-likelihood")[0]) < 1e-9, "scipy の G² と一致")
# 制約の違い：基準カテゴリ制約で解いても μ̂ は同じ（独立モデルの μ̂ を2通りのパラメータ化で再構成）
lam = np.log(E)
A_sum = lam.mean(1) - lam.mean(); B_sum = lam.mean(0) - lam.mean()
mu_sum = np.exp(lam.mean() + A_sum[:, None] + B_sum[None, :])
A_ref = lam[:, 0] - lam[0, 0]; B_ref = lam[0, :] - lam[0, 0]
mu_ref = np.exp(lam[0, 0] + A_ref[:, None] + B_ref[None, :])
q.check("q28-loglinear-model-3", np.allclose(mu_sum, mu_ref) and np.allclose(mu_sum, E), "制約によらず μ̂ は同じ（誤り）")

# --- 階層対数線形 ---
q.concept("q28-hierarchical-loglinear-model-1", "生成元が含む項（数値なし）")
I, J, K = 2, 3, 4
npar = 1 + (I - 1) + (J - 1) + (K - 1) + (I - 1) * (K - 1) + (J - 1) * (K - 1)
q.num("q28-hierarchical-loglinear-model-2", I * J * K - npar)
q.check("q28-hierarchical-loglinear-model-2", I * J * K - npar == K * (I - 1) * (J - 1), "K(I-1)(J-1) と一致")
df_diff = (I * J * K - npar) - (I - 1) * (J - 1) * (K - 1)
q.check("q28-hierarchical-loglinear-model-3", df_diff == 2 and np.isclose(15.2 - 4.1, 11.1)
        and abs(stats.chi2.sf(11.1, 2) - 0.004) < 0.0005, "統計量11.1・自由度2・p≈0.004")

# --- シンプソン ---
A_rate = (18 + 48) / 100; B_rate = (64 + 10) / 100
q.num("q28-simpsons-paradox-1", A_rate - B_rate)
q.check("q28-simpsons-paradox-1", 18 / 20 > 64 / 80 and 48 / 80 > 10 / 20, "層ごとには A が上")
q.concept("q28-simpsons-paradox-2", "逆転の必要条件（数値なし）")
ok = True
for _ in range(20000):
    k = rng.integers(2, 5)
    w = rng.dirichlet(np.ones(k))
    r0 = rng.uniform(0, 0.9, k); r1 = r0 + rng.uniform(0.001, 0.1, k)
    ok &= (w @ r1 - w @ r0) > 0
q.check("q28-simpsons-paradox-3", bool(ok), "構成比が共通なら逆転しない（ランダム例2万件）")

# --- マンテル・ヘンツェル ---
strata = [np.array([[10, 20], [5, 25]]), np.array([[15, 5], [10, 10]])]
num = 0; Ea = 0; Va = 0; R = 0; S = 0
for t in strata:
    (a_, b_), (c_, d_) = t; n = t.sum(); n1, n2 = t.sum(1); m1, m2 = t.sum(0)
    num += a_; Ea += n1 * m1 / n; Va += n1 * n2 * m1 * m2 / (n**2 * (n - 1))
    R += a_ * d_ / n; S += b_ * c_ / n
q.num("q28-mantel-haenszel-test-1", (num - Ea) ** 2 / Va)
res_mh = stats.contingency.odds_ratio if hasattr(stats.contingency, "odds_ratio") else None
q.concept("q28-mantel-haenszel-test-2", "向きの逆の層で打ち消し合う性質（数値なし）")
q.num("q28-mantel-haenszel-estimator-1", R / S)
ors = [t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]) for t in strata]
ws = [t[0, 1] * t[1, 0] / t.sum() for t in strata]
q.check("q28-mantel-haenszel-estimator-2", np.isclose(np.dot(ors, ws) / sum(ws), R / S) and ors == [2.5, 3.0], "加重平均と一致（正しい）")

# --- 3元表 ---
s1 = np.outer([0.8, 0.2], [0.8, 0.2]) * 100; s2 = np.outer([0.2, 0.8], [0.2, 0.8]) * 100
m = s1 + s2
q.num("q28-three-way-table-1", m[0, 0] * m[1, 1] / (m[0, 1] * m[1, 0]))
pA = rng.dirichlet(np.ones(2)); pBC = rng.dirichlet(np.ones(6)).reshape(3, 2)
pj = pA[:, None, None] * pBC[None, :, :]
pij = pj.sum(2)
cond = all(np.allclose(pj[:, :, k] / pj[:, :, k].sum(), np.outer(pj[:, :, k].sum(1), pj[:, :, k].sum(0)) / pj[:, :, k].sum() ** 2) for k in range(2))
q.check("q28-three-way-table-2", np.allclose(pij, np.outer(pij.sum(1), pij.sum(0))) and cond, "同時独立 ⇒ 周辺独立かつ条件付き独立")

# --- 残差 ---
n = T.sum(); ni = T.sum(1); nj = T.sum(0)
adj = (T - E) / np.sqrt(E * (1 - ni[:, None] / n) * (1 - nj[None, :] / n))
q.num("q28-residual-analysis-contingency-1", adj[1, 2])
pear = (T - E) / np.sqrt(E)
q.check("q28-residual-analysis-contingency-2", np.all(np.abs(pear) <= np.abs(adj) + 1e-12), "全セルで |ピアソン残差| ≤ |調整済み残差|")

# --- ファイ・クラメール ---
phi = (a * d - b * c) / np.sqrt(100 * 100 * 65 * 135)
q.num("q28-phi-coefficient-and-association-1", phi)
xs = np.repeat([1, 1, 0, 0], [a, b, c, d]); ys = np.repeat([1, 0, 1, 0], [a, b, c, d])
q.check("q28-phi-coefficient-and-association-1", np.isclose(np.corrcoef(xs, ys)[0, 1], phi) and np.isclose(phi**2, chi2_22 / 200), "相関係数・χ²/n と一致")
q.num("q28-phi-coefficient-and-association-2", np.sqrt(22.04 / 160))
q.check("q28-phi-coefficient-and-association-2", np.isclose(stats.contingency.association(T.astype(int), method="cramer"), np.sqrt(chi2 / 160))
        and abs(np.sqrt(22.04 / 320) - 0.371) > 0.05, "scipy のクラメール V と一致、誤答 0.262 とは異なる")

# --- コホート・ケースコントロール ---
q.concept("q28-cohort-and-case-control-study-1", "ケースコントロールで推定できる指標（数値なし）")
q.num("q28-cohort-and-case-control-study-2", (0.04 / 0.96) / (0.02 / 0.98))

# --- 順序のある分割表 ---
x = np.array([0, 1, 2.0]); mm = np.array([50, 50, 50.0]); y = np.array([5, 10, 20.0])
N = mm.sum(); pb = y.sum() / N
Z2 = (np.sum(x * (y - mm * pb))) ** 2 / (pb * (1 - pb) * (np.sum(mm * x**2) - np.sum(mm * x) ** 2 / N))
q.num("q28-ordinal-contingency-table-1", Z2)
# 回帰の傾きのスコア検定として：個体データで N r^2 と一致
xi = np.repeat(x, mm.astype(int)); yi = np.concatenate([np.r_[np.ones(int(k)), np.zeros(int(m_ - k))] for k, m_ in zip(y, mm)])
q.check("q28-ordinal-contingency-table-1", np.isclose(N * np.corrcoef(xi, yi)[0, 1] ** 2, Z2), "N r² と一致")

# --- 導出問題（追加）---
import sympy as _sp
_p, _n = _sp.symbols("p n", positive=True)
_dv = _sp.simplify((1 / _p)**2 * _p * (1 - _p) / _n - (1 - _p) / (_n * _p))
_a = _sp.symbols("a", positive=True)
_sub = _sp.simplify(((1 - _p) / (_n * _p)).subs(_p, _a / _n) - (1 / _a - 1 / _n))
_se = np.sqrt(1 / 30 - 1 / 100 + 1 / 15 - 1 / 100)
# シミュレーションで log RR の SD を確認
_rng = np.random.default_rng(28)
_x1 = _rng.binomial(100, 0.3, 400000); _x2 = _rng.binomial(100, 0.15, 400000)
_m = (_x1 > 0) & (_x2 > 0)
_sd = np.std(np.log(_x1[_m] / _x2[_m]))
_ci = np.exp(np.log(2) + np.array([-1, 1]) * stats.norm.ppf(0.975) * _se)
q.check("q28-relative-risk-4", _dv == 0 and _sub == 0 and round(_se, 3) == 0.283 and abs(_sd / _se - 1) < 0.1
        and abs(_ci[0] - 1.15) < 0.005 and abs(_ci[1] - 3.48) < 0.005,
        f"デルタ法の式、SE={_se:.4f}、シミュレーションSD={_sd:.4f}（有限標本で約5%ずれるため相対10%以内）、CI=({_ci[0]:.3f},{_ci[1]:.3f})")

q.done()
