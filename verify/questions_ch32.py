"""第32章（シミュレーション）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats, integrate, optimize

q = Q(32)
rng = np.random.default_rng(32)
N = 1_000_000

# --- 一様乱数 ---
q.num("q32-uniform-random-number-1", 9 / 12)
q.check("q32-uniform-random-number-1", abs(stats.uniform(1, 3).var() - 0.75) < 1e-12, "U(1,4) の分散")
q.num("q32-uniform-random-number-2", 0.5 * 0.5 * 0.5)
U = rng.uniform(size=(N, 2))
q.check("q32-uniform-random-number-2", abs(np.mean(U.sum(1) <= 0.5) - 0.125) < 0.002, "シミュレーション")
a1 = np.random.default_rng(123).uniform(size=5); a2 = np.random.default_rng(123).uniform(size=5)
q.check("q32-uniform-random-number-3", np.array_equal(a1, a2), "同じシードで同じ列（正しい）")

# --- 線形合同法 ---
x = 1; seq = []
for _ in range(16):
    x = (5 * x + 3) % 16; seq.append(x)
q.num("q32-linear-congruential-generator-1", seq[2])
periods = []
for x0 in range(16):
    x = x0; seen = {}
    for t in range(40):
        if x in seen:
            periods.append(t - seen[x]); break
        seen[x] = t; x = (5 * x + 3) % 16
q.check("q32-linear-congruential-generator-2", all(p == 16 for p in periods), "全シードで周期16（正しい）")

# --- 逆関数法 ---
q.num("q32-inverse-transform-sampling-1", -np.log(1 - 0.7) / 2)
q.check("q32-inverse-transform-sampling-1", abs(stats.expon(scale=0.5).ppf(0.7) - (-np.log(0.3) / 2)) < 1e-12, "scipy の分位点と一致")
xv = (1 - 0.875) ** (-1 / 3)
q.num("q32-inverse-transform-sampling-2", xv)
q.check("q32-inverse-transform-sampling-2", abs(1 - xv**-3 - 0.875) < 1e-12, "F(2)=0.875")
vals = np.array([1, 2, 3]); F = np.cumsum([0.2, 0.5, 0.3])
q.check("q32-inverse-transform-sampling-3", vals[np.searchsorted(F, 0.75)] == 3, "一般化逆関数で 3")

# --- 棄却法 ---
M = optimize.minimize_scalar(lambda t: -6 * t * (1 - t), bounds=(0, 1), method="bounded")
q.num("q32-rejection-sampling-1", 1 / (-M.fun))
Y = rng.uniform(size=N); V = rng.uniform(size=N)
acc = V <= 6 * Y * (1 - Y) / 1.5
q.check("q32-rejection-sampling-1", abs(acc.mean() - 2 / 3) < 0.002 and abs(Y[acc].mean() - 0.5) < 0.002 and abs(Y[acc].var() - 0.05) < 0.001,
        "シミュレーションで採択率2/3、Be(2,2) の平均・分散")
r = optimize.minimize_scalar(lambda t: -np.sqrt(2 / np.pi) * np.exp(t - t**2 / 2), bounds=(0, 5), method="bounded")
q.num("q32-rejection-sampling-2", -r.fun)
q.check("q32-rejection-sampling-2", abs(-r.fun - np.sqrt(2 * np.e / np.pi)) < 1e-6, "√(2e/π) と一致")
Y = rng.uniform(size=N); V = rng.uniform(size=N)
acc = V <= 6 * Y * (1 - Y) / 5.0
q.check("q32-rejection-sampling-3", abs(acc.mean() - 0.2) < 0.002 and abs(Y[acc].var() - 0.05) < 0.002, "M=5 でも分布は Be(2,2)（誤り）")

# --- ボックス・ミュラー ---
u1, u2 = np.exp(-2), 0.125
q.num("q32-box-muller-method-1", np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))
U1, U2 = rng.uniform(size=N), rng.uniform(size=N)
Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2); Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
R2 = Z1**2 + Z2**2
q.check("q32-box-muller-method-2", abs(R2.mean() - 2) < 0.01 and stats.kstest(R2, "expon", args=(0, 2)).pvalue > 0.001, "R² は平均2の指数分布")
q.num("q32-box-muller-method-3", np.exp(-2))
q.check("q32-box-muller-method-3", abs(stats.chi2.sf(4, 2) - np.exp(-2)) < 1e-12, "χ²(2) の上側確率と一致")

# --- モンテカルロ積分 ---
q.num("q32-monte-carlo-integration-1", 2 * 1.7)
s2 = (np.e**2 - 1) / 2 - (np.e - 1) ** 2
q.num("q32-monte-carlo-integration-2", np.sqrt(s2 / 10000))
q.check("q32-monte-carlo-integration-2", abs(np.var(np.exp(rng.uniform(size=N))) - s2) < 0.002, "シミュレーションの分散")
nreq = int(np.ceil(s2 / 1e-6))
q.num("q32-monte-carlo-integration-3", nreq)
q.check("q32-monte-carlo-integration-3", nreq == 242036, "必要数 242036")

# --- 重点サンプリング ---
q.num("q32-importance-sampling-1", stats.norm.pdf(3) / stats.norm.pdf(3, 3, 1))
def is_var(gpdf, gsample):
    X = gsample(N); w = 1 / gpdf(X); return np.var(X * w)
v_opt = is_var(lambda x: 2 * x, lambda n: np.sqrt(rng.uniform(size=n)))
v_unif = is_var(lambda x: np.ones_like(x), lambda n: rng.uniform(size=n))
v_cube = is_var(lambda x: 3 * x**2, lambda n: rng.uniform(size=n) ** (1 / 3))
v_rev = [integrate.quad(lambda x: x**2 / (2 * (1 - x)), 0, 1 - eps)[0] for eps in (1e-3, 1e-6)]  # ε→0 で発散
v_rev = v_rev[1] - v_rev[0]
q.check("q32-importance-sampling-2", v_opt < 1e-20 and abs(v_unif - 1 / 12) < 0.001 and v_cube > 0.01 and v_rev > 3,
        "g=2x で分散0、他は正（2(1-x) は発散）")
# 裾の軽い提案：f=標準コーシー、g=N(0,1) で E_g[w^2]=∫f²/g が発散
v_tail = integrate.quad(lambda x: stats.cauchy.pdf(x) ** 2 / stats.norm.pdf(x), -30, 30)[0]
q.check("q32-importance-sampling-3", v_tail > 1e50, "裾の軽い g で ∫f²/g が発散（正しい）")

# --- MCMC ---
P = np.array([[0.9, 0.1], [0.3, 0.7]])
w, vec = np.linalg.eig(P.T); pi = np.real(vec[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
q.num("q32-mcmc-1", pi[0])
C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]]); u = np.ones(3) / 3
q.check("q32-mcmc-2", np.allclose(u @ C, u) and u[0] * C[0, 1] != u[1] * C[1, 0], "詳細つり合いなしでも定常（誤り）")

# --- MH ---
q.num("q32-metropolis-hastings-1", min(1, stats.norm.pdf(2) / stats.norm.pdf(1)))
tgt = lambda x: x * np.exp(-x); qd = lambda x: np.exp(-x)
q.num("q32-metropolis-hastings-2", min(1, (tgt(1.5) / qd(1.5)) / (tgt(2) / qd(2))))
# 棄却時に記録しないと分布がずれることをシミュレーションで確認（目標 N(0,1)、ランダムウォーク）
x = 0.0; keep_all, keep_acc = [], []
for _ in range(200000):
    y = x + rng.normal(scale=2.5)
    if rng.uniform() < min(1, np.exp(-(y**2 - x**2) / 2)):
        x = y; keep_acc.append(x)
    keep_all.append(x)
q.check("q32-metropolis-hastings-3", abs(np.var(keep_all) - 1) < 0.05 and np.var(keep_acc) > 1.1, "棄却時を記録しないと分散が膨らむ（誤り）")

# --- ギブス ---
rho = 0.8
q.num("q32-gibbs-sampling-1", 1 - rho**2)
xg, yg = 0.0, 0.0; xs = []
for _ in range(200000):
    xg = rng.normal(rho * yg, np.sqrt(1 - rho**2)); yg = rng.normal(rho * xg, np.sqrt(1 - rho**2)); xs.append(xg)
xs = np.array(xs[1000:])
q.num("q32-gibbs-sampling-2", rho**2)
q.check("q32-gibbs-sampling-2", abs(np.corrcoef(xs[:-1], xs[1:])[0, 1] - 0.64) < 0.01, "シミュレーションの自己相関 ≈ 0.64")
# MH 比が1：2変量正規で x を条件付き分布から提案
xc, yc, xn = 0.3, 1.5, -0.7
joint = lambda a, b: stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).pdf([a, b])
cond = lambda a, b: stats.norm.pdf(a, rho * b, np.sqrt(1 - rho**2))
ratio = joint(xn, yc) * cond(xc, yc) / (joint(xc, yc) * cond(xn, yc))
q.check("q32-gibbs-sampling-3", abs(ratio - 1) < 1e-10, "MH 比 = 1（正しい）")

# --- バーンイン・収束 ---
q.num("q32-burn-in-convergence-1", 10000 / ((1 + 0.8) / (1 - 0.8)))
n, m = 100, 2
W = 1.0; Bv = n / (m - 1) * ((0.9 - 1.0) ** 2 + (1.1 - 1.0) ** 2)
Vh = (n - 1) / n * W + Bv / n
q.num("q32-burn-in-convergence-2", np.sqrt(Vh / W))

# --- ブートストラップ ---
q.num("q32-bootstrap-1", 0.9**10)
sims = rng.integers(0, 10, size=(200000, 10))
q.check("q32-bootstrap-1", abs(np.mean(~(sims == 0).any(1)) - 0.9**10) < 0.003, "シミュレーション")
q.num("q32-bootstrap-2", 5.2 - (5.5 - 5.2))
# B を増やしても n=5 ではブートストラップ SE と真の SE のずれが残る
true_se = 1 / np.sqrt(5); errs = []
for _ in range(400):
    xs5 = rng.normal(size=5)
    idx = rng.integers(0, 5, size=(4000, 5))
    errs.append(xs5[idx].mean(1).std(ddof=1) - true_se)
q.check("q32-bootstrap-3", np.sqrt(np.mean(np.square(errs))) > 0.1, "B=4000 でも誤差が残る（誤り）")

# --- ブートストラップ信頼区間 ---
q.num("q32-bootstrap-confidence-interval-1", 2 * 10 - 12.5)

# --- ジャックナイフ ---
d = np.array([2, 4, 6, 8.0]); n = 4
th_i = np.array([np.delete(d, i).mean() for i in range(n)])
Vj = (n - 1) / n * np.sum((th_i - th_i.mean()) ** 2)
q.num("q32-jackknife-1", np.sqrt(Vj))
q.check("q32-jackknife-1", np.isclose(np.sqrt(Vj), d.std(ddof=1) / 2), "s/√n と一致")
th = d.var(ddof=0); th_v = np.array([np.delete(d, i).var(ddof=0) for i in range(n)])
jk = n * th - (n - 1) * th_v.mean()
q.num("q32-jackknife-2", jk)
q.check("q32-jackknife-2", np.isclose(jk, d.var(ddof=1)), "不偏分散と一致")
# 中央値のジャックナイフ分散は一致しない：n を増やしても真の分散との比が1に近づかない
ratios = []
for _ in range(300):
    z = rng.normal(size=401)
    med_i = np.array([np.median(np.delete(z, i)) for i in range(0, 401, 1)])
    ratios.append((400 / 401) * np.sum((med_i - med_i.mean()) ** 2) / (np.pi / 2 / 401))
q.check("q32-jackknife-3", np.std(ratios) > 0.5, f"中央値のジャックナイフ分散の比が大きくばらつく（sd={np.std(ratios):.2f}）")

# --- 分散減少 ---
cov = np.e - (np.e - 1) ** 2
q.num("q32-variance-reduction-1", (s2 + cov) / 2)
Uu = rng.uniform(size=N)
q.check("q32-variance-reduction-1", abs(np.var((np.exp(Uu) + np.exp(1 - Uu)) / 2) - (s2 + cov) / 2) < 2e-4, "シミュレーションと一致")

# --- モンテカルロ検定 ---
q.num("q32-permutation-simulation-1", (1 + 24) / (999 + 1))

# --- 導出問題（追加）---
_lap_inv = lambda u: np.where(u < 0.5, np.log(2 * u), -np.log(2 * (1 - u)))
_sgn_form = lambda u: np.sign(u - 0.5) * (-np.log(1 - 2 * np.abs(u - 0.5)))
_us = np.linspace(0.001, 0.999, 999)
_v9, _v2 = float(_lap_inv(0.9)), float(_lap_inv(0.2))
q.check("q32-inverse-transform-sampling-4",
        np.allclose(stats.laplace.cdf(_lap_inv(_us)), _us) and np.allclose(_lap_inv(_us), _sgn_form(_us))
        and round(_v9, 3) == 1.609 and round(_v2, 3) == -0.916 and abs(stats.laplace.cdf(0) - 0.5) < 1e-12,
        f"F(F^-1(u))=u、符号つきの式と一致、U=0.9→{_v9:.4f}、U=0.2→{_v2:.4f}")

_ft = lambda x: x**2 * (1 - x)
_r = optimize.minimize_scalar(lambda x: -_ft(x), bounds=(0, 1), method="bounded")
_M = -_r.fun; _Z = integrate.quad(_ft, 0, 1)[0]
_rng = np.random.default_rng(32)
_Y = _rng.uniform(size=1_000_000); _V = _rng.uniform(size=1_000_000)
_acc = _V <= _ft(_Y) / _M
_ks = stats.kstest(_Y[_acc][:20000], stats.beta(3, 2).cdf).pvalue
q.check("q32-rejection-sampling-4", abs(_r.x - 2 / 3) < 1e-5 and abs(_M - 4 / 27) < 1e-9 and abs(_Z - 1 / 12) < 1e-12
        and round(_Z / _M, 4) == 0.5625 and abs(_acc.mean() - 0.5625) < 0.003 and _ks > 0.001,
        f"M=4/27、Z=1/12、採択率 {_Z / _M:.4f}、シミュレーション {_acc.mean():.4f}、Be(3,2) との KS p={_ks:.3f}")

q.done()
