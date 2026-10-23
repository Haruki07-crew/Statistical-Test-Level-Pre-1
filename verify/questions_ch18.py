"""第18章（質的回帰）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, optimize
from scipy.special import expit, logit

q = Q(18)
rng = np.random.default_rng(18)


def irls(X, y, family, offset=None, iters=100):
    """自前の IRLS（logit: ベルヌーイ, log: ポアソン）。"""
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = off + X @ b
        if family == "logit":
            mu = expit(eta); w = mu * (1 - mu); z = eta - off + (y - mu) / w
        else:
            mu = np.exp(eta); w = mu; z = eta - off + (y - mu) / mu
        b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
    return b


# --- ロジスティック回帰 ---
p6 = expit(-2 + 0.5 * 6)
q.num("q18-logistic-regression-1", p6)
xs = sp.Symbol("x"); P = 1 / (1 + sp.exp(-(-2 + sp.Rational(1, 2) * xs)))
slope = float(sp.diff(P, xs).subs(xs, 6))
q.num("q18-logistic-regression-2", slope)
q.check("q18-logistic-regression-2", abs(slope - 0.5 * p6 * (1 - p6)) < 1e-12, "β p(1-p) と一致")
d1 = expit(-2 + 0.5 * 1) - expit(-2 + 0.5 * 0); d2 = expit(-2 + 0.5 * 5) - expit(-2 + 0.5 * 4)
q.check("q18-logistic-regression-3", abs(d1 - d2) > 0.05 and abs(d1 - 0.5) > 0.1, "確率の増分は x で変わり β ではない（誤り）")

# --- オッズ比 ---
a, b_, c, d = 30, 70, 15, 85
OR = a * d / (b_ * c)
q.num("q18-odds-and-odds-ratio-1", OR)
se = np.sqrt(1 / a + 1 / b_ + 1 / c + 1 / d)
lo, hi = np.exp(np.log(OR) - 1.96 * se), np.exp(np.log(OR) + 1.96 * se)
q.num("q18-odds-and-odds-ratio-2", lo)
q.check("q18-odds-and-odds-ratio-2", abs(hi - 4.87) < 0.005, "上限 4.87")
# ロジスティック回帰（群ダミー）で係数が log OR、SE が同じになることも確認
X = np.column_stack([np.ones(200), np.r_[np.ones(100), np.zeros(100)]])
y = np.r_[np.ones(30), np.zeros(70), np.ones(15), np.zeros(85)]
bb = irls(X, y, "logit"); mu = expit(X @ bb)
se_glm = np.sqrt(np.linalg.inv(X.T @ ((mu * (1 - mu))[:, None] * X))[1, 1])
q.check("q18-odds-and-odds-ratio-2", abs(bb[1] - np.log(OR)) < 1e-9 and abs(se_glm - se) < 1e-9, "ロジスティック回帰の係数・SE と一致")
r = np.exp(3 * 0.4)
q.check("q18-odds-and-odds-ratio-3", abs(r - 3.32) < 0.005 and all(abs(w - r) > 0.5 for w in [1.2, np.exp(0.4), 3 * np.exp(0.4)]),
        "正解 e^1.2=3.32、誤答 1.2, 1.49, 4.47 と異なる")

# --- GLM ---
q.concept("q18-generalized-linear-model-1", "GLM の3要素に等分散は含まれない")
lam = 3.0; k = np.arange(0, 200)
ElogY = np.sum(stats.poisson.pmf(k[1:], lam) * np.log(k[1:]))  # y>0 のみでも log E より小さい
q.check("q18-generalized-linear-model-2", ElogY < np.log(lam) and stats.poisson.pmf(0, lam) > 0, "E[log Y] < log E[Y]、y=0 が起こる（誤り）")

# --- リンク関数 ---
mu, y_ = sp.symbols("mu y", positive=True)
logpmf = sp.expand_log(sp.log(sp.exp(-mu) * mu ** y_ / sp.factorial(y_)), force=True)
theta = sp.simplify(sp.diff(logpmf, y_) + sp.diff(sp.log(sp.factorial(y_)), y_))
q.check("q18-link-function-1", sp.simplify(theta - sp.log(mu)) == 0, "y の係数は log μ")
q.num("q18-link-function-2", 1 - np.exp(-np.exp(0)))
q.check("q18-link-function-2", abs(np.log(-np.log(1 - (1 - np.exp(-1))))) < 1e-12, "g(p)=0 を確認")

# --- 指数分散型分布族 ---
th = sp.Symbol("theta"); bth = sp.exp(th)
q.check("q18-exponential-dispersion-family-1", sp.simplify(sp.diff(bth, th).subs(th, sp.log(mu)) - mu) == 0
        and sp.simplify(sp.diff(bth, th, 2).subs(th, sp.log(mu)) - mu) == 0
        and abs(stats.poisson.var(2.7) - 2.7) < 1e-12, "b'=b''=λ")
q.num("q18-exponential-dispersion-family-2", stats.gamma(a=2, scale=4 / 2).var())

# --- プロビット ---
q.num("q18-probit-model-1", stats.norm.cdf(-1 + 0.8 * 2))
q.check("q18-probit-model-1", abs(expit(0.6) - 0.646) < 0.001, "誤答のロジスティック値 0.646")

# --- ロジスティックの最尤推定 ---
y = np.r_[np.ones(12), np.zeros(28)]; X = np.ones((40, 1))
nll = lambda bb: -np.sum(y * bb[0] - np.log1p(np.exp(bb[0])))
opt = optimize.minimize(nll, [0.0])
q.num("q18-logistic-likelihood-estimation-1", opt.x[0])
q.check("q18-logistic-likelihood-estimation-1", abs(logit(0.3) - opt.x[0]) < 1e-4 and abs(1 / np.sqrt(40 * 0.21) - 0.345) < 0.001, "logit(0.3)、SE 0.345")
xsep = np.array([1, 2, 3, 4, 5, 6.]); ysep = np.array([0, 0, 0, 1, 1, 1.])
ll = lambda s: np.sum(ysep * s * (xsep - 3.5) - np.log1p(np.exp(s * (xsep - 3.5))))
vals = [ll(s) for s in [1, 5, 20, 80]]
q.check("q18-logistic-likelihood-estimation-2", all(np.diff(vals) > 0) and vals[-1] < 0 and vals[-1] > -1e-6,
        "完全分離で対数尤度は単調増加し0に近づくだけ（正しい）")

# --- IRLS ---
z = np.log(4) + (6 - 4) / 4
q.num("q18-iteratively-reweighted-least-squares-1", z)
q.check("q18-iteratively-reweighted-least-squares-1", abs(np.log(6) - 1.792) < 0.001, "log y=1.792 と異なる")

# --- 多項ロジット ---
eta = np.array([0, 0.5, -0.2]); pi = np.exp(eta) / np.exp(eta).sum()
q.num("q18-multinomial-logit-model-1", pi[1])
q.check("q18-multinomial-logit-model-1", np.allclose(np.round(pi, 3), [0.288, 0.475, 0.236]) and abs(expit(0.5) - 0.622) < 0.001, "解説の値")
q.num("q18-multinomial-logit-model-2", (4 - 1) * (3 + 1))
q.check("q18-multinomial-logit-model-2", (4 - 1) + 3 == 6, "順序ロジットなら6個")

# --- 順序ロジット ---
g1, g2 = expit(-1 - 0.8), expit(1.5 - 0.8)
q.num("q18-ordered-logit-model-1", g2 - g1)
q.check("q18-ordered-logit-model-1", abs(1 - g2 - 0.332) < 0.001, "P(Y=3)=0.332")

# --- トービット ---
q.num("q18-tobit-model-1", stats.norm.sf(1 / 2))
ys = np.maximum(0, 1 + rng.normal(scale=2, size=400000))
q.check("q18-tobit-model-1", abs(np.mean(ys == 0) - stats.norm.sf(0.5)) < 0.003 and abs(stats.norm.sf(1) - 0.159) < 0.001, "シミュレーションと一致、誤答0.159")

# --- 過分散 ---
q.num("q18-overdispersion-1", np.sqrt(84 / (40 - 3)))
q.check("q18-overdispersion-1", abs(np.sqrt(84 / 40) - 1.45) < 0.005, "n で割った誤答 1.45")
# 準ポアソン：係数は同じ（推定方程式が φ で割っただけ）。負の二項データで確認
X = np.column_stack([np.ones(300), rng.normal(size=300)])
yy = rng.negative_binomial(2, 2 / (2 + np.exp(1 + 0.3 * X[:, 1])))
bp = irls(X, yy.astype(float), "log"); mu = np.exp(X @ bp)
phi = np.sum((yy - mu) ** 2 / mu) / (300 - 2)
se_p = np.sqrt(np.diag(np.linalg.inv(X.T @ (mu[:, None] * X))))
# 準尤度推定方程式 sum (y-mu)x/phi = 0 を数値で解く
qs = optimize.root(lambda bb: X.T @ (yy - np.exp(X @ bb)) / phi, np.zeros(2)).x
q.check("q18-overdispersion-2", np.allclose(qs, bp, atol=1e-8) and phi > 1, f"準尤度の解=ポアソンの解、φ̂={phi:.2f}、SE×√φ")

# --- ポアソン回帰 ---
q.num("q18-poisson-regression-1", np.exp(0.6))
q.num("q18-poisson-regression-2", stats.poisson.pmf(0, np.exp(1.1)))
x01 = np.r_[np.zeros(7), np.ones(5)]; yy = rng.poisson(np.where(x01 == 1, 5, 2)).astype(float)
bp = irls(np.column_stack([np.ones(12), x01]), yy, "log")
q.check("q18-poisson-regression-3", abs(bp[0] - np.log(yy[x01 == 0].mean())) < 1e-9
        and abs(bp[1] - np.log(yy[x01 == 1].mean() / yy[x01 == 0].mean())) < 1e-9, "IRLS の解が log ȳ0, log(ȳ1/ȳ0)")

# --- 逸脱度 ---
y = np.array([2, 5, 0.]); m = np.array([3, 4, 1.])
D = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))
q.num("q18-deviance-1", D)
# 尤度比としても計算
ll = lambda mu_: np.sum(stats.poisson.logpmf(y, mu_))
q.check("q18-deviance-1", abs(2 * (ll(np.where(y > 0, y, 1e-300)) - ll(m)) - D) < 1e-9, "2{ℓ(y;y)-ℓ(μ̂;y)} と一致")
q.num("q18-deviance-2", stats.chi2.sf(45.2 - 38.1, 30 - 28))

# --- オフセット ---
q.num("q18-offset-term-1", 200 * np.exp(-2 + 0.5))
# オフセット付き IRLS で率が再現されるか
t = rng.uniform(50, 300, 2000); xx = rng.integers(0, 2, 2000).astype(float)
yy = rng.poisson(t * np.exp(-2 + 0.5 * xx)).astype(float)
bo = irls(np.column_stack([np.ones(2000), xx]), yy, "log", offset=np.log(t))
q.check("q18-offset-term-1", abs(bo[0] + 2) < 0.02 and abs(bo[1] - 0.5) < 0.02, f"オフセット付きIRLSで係数を再現 {bo.round(3)}")

# --- 導出問題（追加） ---
def _deriv18():
    import numpy as _N
    import sympy as _S
    eta, y = _S.symbols("eta y", real=True)
    p = 1 / (1 + _S.exp(-eta))
    l_orig = y * _S.log(p) + (1 - y) * _S.log(1 - p)
    l_new = y * eta - _S.log(1 + _S.exp(eta))
    ok = all(abs(float((l_orig - l_new).subs({eta: e, y: yy}))) < 1e-12 for e in (-2, 0.3, 1.7) for yy in (0, 1))
    ok &= _S.simplify(_S.diff(l_new, eta) - (y - p)) == 0
    ok &= _S.simplify(_S.diff(l_new, eta, 2) + p * (1 - p)) == 0
    # 数値微分で勾配 X'(y-p)、ヘッセ -X'WX を確認
    rng = _N.random.default_rng(18)
    X = _N.column_stack([_N.ones(30), rng.normal(size=(30, 2))]); yv = rng.integers(0, 2, 30).astype(float)
    b = _N.array([0.2, -0.5, 0.8])
    ll = lambda bb: (yv * (X @ bb) - _N.log1p(_N.exp(X @ bb))).sum()
    pv = 1 / (1 + _N.exp(-(X @ b)))
    g = X.T @ (yv - pv); Hs = -(X.T * (pv * (1 - pv))) @ X
    h = 1e-5; E = _N.eye(3)
    gnum = _N.array([(ll(b + h * E[j]) - ll(b - h * E[j])) / (2 * h) for j in range(3)])
    Hnum = _N.array([[(ll(b + h * E[j] + h * E[k]) - ll(b + h * E[j] - h * E[k]) - ll(b - h * E[j] + h * E[k]) + ll(b - h * E[j] - h * E[k])) / (4 * h * h) for k in range(3)] for j in range(3)])
    ok &= _N.allclose(g, gnum, atol=1e-6) and _N.allclose(Hs, Hnum, atol=1e-3)
    q.check("q18-logistic-regression-4", bool(ok), "sympy: ℓ=yη-log(1+e^η)、dℓ/dη=y-p、d²ℓ/dη²=-p(1-p)、数値微分で X'(y-p)、-X'WX")
_deriv18()

q.done()
