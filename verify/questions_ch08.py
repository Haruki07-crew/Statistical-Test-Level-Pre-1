"""第8章（統計的推定の基礎）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb
from itertools import product
from scipy import stats, special, integrate

q = Q(8)
rng = np.random.default_rng(8)

q.concept("q08-point-estimation-1", "統計量の定義（数値なし）")

# --- 不偏推定量 ---
n, s2 = 10, 5
e = (n - 1) / n * s2
q.check("q08-unbiased-estimator-1", abs(e - 4.5) < 1e-12 and all(abs(w - e) > 0.1 for w in [5, n / (n - 1) * s2, 0.5]),
        "正解 4.5、誤答は異なる")
th, n = 1.0, 4
EM = integrate.quad(lambda m: m * n * m ** (n - 1) / th ** n, 0, th)[0]
q.num("q08-unbiased-estimator-2", th / EM)
n = 5
c4 = np.sqrt(2 / (n - 1)) * special.gamma(n / 2) / special.gamma((n - 1) / 2)
sim = np.std(rng.standard_normal((200000, n)), axis=1, ddof=1).mean()
q.check("q08-unbiased-estimator-3", abs(c4 - 0.940) < 5e-4 and abs(sim - c4) < 3e-3, "E[S]=0.940σ<σ（誤り）")

# --- 一致推定量 ---
q.check("q08-consistent-estimator-1", abs(2 * stats.norm.sf(1) - 0.317) < 5e-4, "P(|X1-μ|>=1)=0.317 で一定（誤り）")
nn = 10 ** 6
vars_ = {"(X1+Xn)/2": 0.5, "mean": 1 / nn, "sum/(n+1)": nn / (nn + 1) ** 2, "median": np.pi / (2 * nn)}
q.check("q08-consistent-estimator-2", vars_["(X1+Xn)/2"] == 0.5 and max(v for k, v in vars_.items() if k != "(X1+Xn)/2") < 1e-5,
        "分散が0に近づかないのは (X1+Xn)/2 だけ")

# --- 有効推定量 ---
q.num("q08-efficient-estimator-1", (1 / 10) / (1 / 2))
nn = 401
med = np.median(rng.standard_normal((40000, nn)), axis=1)
q.num("q08-efficient-estimator-2", 2 / np.pi)
q.check("q08-efficient-estimator-2", abs((1 / nn) / med.var() - 2 / np.pi) < 0.02, "シミュレーションで約0.637")

# --- MSE ---
q.num("q08-mean-squared-error-1", 0.05 + 0.2 ** 2)
n, p = 96, 0.3
bias = (n * p + 2) / (n + 4) - p
var = n * p * (1 - p) / (n + 4) ** 2
q.num("q08-mean-squared-error-2", var + bias ** 2)
q.check("q08-mean-squared-error-2", abs(p * (1 - p) / n - 0.002188) < 1e-6 and var + bias ** 2 < p * (1 - p) / n,
        "不偏推定量 0.002188 より小さい")
c, N = sp.symbols("c n", positive=True)
mse = c ** 2 * 2 * (N - 1) + (c * (N - 1) - 1) ** 2
copt = sp.solve(sp.diff(mse, c), c)[0]
q.check("q08-mean-squared-error-3", sp.simplify(copt - 1 / (N + 1)) == 0, "最小は c=1/(n+1)")
for nv in [5, 10]:
    vals = [float(mse.subs({c: cc, N: nv})) for cc in [1 / (nv + 1), 1 / (nv - 1), 1 / nv, 1 / (nv - 2)]]
    q.check("q08-mean-squared-error-3", int(np.argmin(vals)) == 0 and len(set(np.round(vals, 9))) == 4, f"n={nv} で1番目が最小")

# --- 尤度 ---
q.check("q08-likelihood-function-1", abs(integrate.quad(lambda p: p, 0, 1)[0] - 0.5) < 1e-12, "∫p dp=1/2≠1（誤り）")
x = np.array([2, 0, 3, 1])
L = lambda lam: np.prod(stats.poisson.pmf(x, lam))
q.num("q08-likelihood-function-2", L(2) / L(1))

# --- 最尤推定量 ---
x = np.array([1.2, 3.1, 0.7, 4.0, 3.5])
res = stats.expon.fit(x, floc=0)
q.num("q08-maximum-likelihood-estimator-1", 1 / res[1])
x = np.array([4, 7, 5, 9, 5.0])
mu_hat, sd_hat = stats.norm.fit(x)
q.num("q08-maximum-likelihood-estimator-2", sd_hat ** 2)
q.check("q08-maximum-likelihood-estimator-2", abs(x.var(ddof=1) - 4) < 1e-12, "不偏分散は 4（誤答の値）")
x = rng.uniform(0, 3, 8)
grid = np.linspace(0.01, 6, 600001)
Lg = np.where(grid >= x.max(), grid ** (-8.0), 0)
q.check("q08-maximum-likelihood-estimator-3", abs(grid[np.argmax(Lg)] - x.max()) < 1e-4
        and abs(2 * x.mean() - x.max()) > 1e-6 and abs(9 / 8 * x.max() - x.max()) > 1e-6, "尤度の最大点は max")

# --- スコア ---
lam, X = sp.symbols("lambda x", positive=True)
s = X / lam - 1
k = sp.Symbol("k", integer=True, nonnegative=True)
Es = sp.summation((k / lam - 1) * lam ** k * sp.exp(-lam) / sp.factorial(k), (k, 0, sp.oo))
q.check("q08-score-function-1", sp.simplify(Es) == 0, "E[s]=0（正しい）")
p = sp.Symbol("p", positive=True)
nS, T = sp.symbols("n T", positive=True)
U = T / p - (nS - T) / (1 - p)
q.check("q08-score-function-2", sp.simplify(U - (T - nS * p) / (p * (1 - p))) == 0
        and sp.simplify(nS * p * (1 - p) / (p * (1 - p)) ** 2 - nS / (p * (1 - p))) == 0, "U と V[U]=n/(p(1-p))")

# --- フィッシャー情報量 ---
I = lambda pp: 1 / (pp * (1 - pp))
q.check("q08-fisher-information-1", abs(I(0.2) - 6.25) < 1e-12 and all(abs(w - 6.25) > 0.1 for w in [0.16, 5, 1.25]), "正解 6.25")
q.num("q08-fisher-information-2", 10 * (1 / 2.5))
mu, v = sp.symbols("mu v", positive=True)
logf = -sp.log(2 * sp.pi * v) / 2 - (X - mu) ** 2 / (2 * v)
d2 = sp.diff(logf, v, 2)
Iv = -d2.subs((X - mu) ** 2, v)
Iv = sp.simplify(-(sp.Rational(1, 2) / v ** 2 - v / v ** 3))
q.check("q08-fisher-information-3", sp.simplify(sp.expand(d2) - (1 / (2 * v ** 2) - (X - mu) ** 2 / v ** 3)) == 0
        and sp.simplify(Iv - 1 / (2 * v ** 2)) == 0, "I(σ^2)=1/(2σ^4)")

# --- クラメール・ラオ ---
th = sp.Symbol("theta", positive=True)
lf = -sp.log(th) - X / th
Ith = sp.simplify(-sp.diff(lf, th, 2).subs(X, th))
q.check("q08-cramer-rao-inequality-1", sp.simplify(Ith - 1 / th ** 2) == 0, "I(θ)=1/θ^2")
q.num("q08-cramer-rao-inequality-1", float(1 / (25 * Ith.subs(th, 2))))
gl = sp.exp(-lam)
crb = sp.diff(gl, lam) ** 2 / (50 / lam)
q.num("q08-cramer-rao-inequality-2", float(crb.subs(lam, 1)))
q.check("q08-cramer-rao-inequality-2", abs(float(crb.subs(lam, 1)) - 0.02) > 0.01, "g'^2 を忘れた 0.02 とは異なる")
nn, thv = 5, 1.0
mx = rng.uniform(0, thv, (400000, nn)).max(axis=1) * (nn + 1) / nn
q.check("q08-cramer-rao-inequality-3", abs(mx.var() - 1 / (nn * (nn + 2))) < 2e-3 and 1 / (nn * (nn + 2)) < 1 / nn,
        "不偏推定量の分散 θ²/(n(n+2)) < θ²/n（誤り）")

# --- 十分統計量 ---
okc = True
for pp in [0.1, 0.5, 0.8]:
    pr = {xx: np.prod([pp if a else 1 - pp for a in xx]) for xx in product([0, 1], repeat=4)}
    pt = sum(v for xx, v in pr.items() if sum(xx) == 2)
    okc &= abs(pr[(1, 1, 0, 0)] / pt - 1 / 6) < 1e-12
q.check("q08-sufficient-statistic-1", bool(okc), "p によらず 1/6（正しい）")
q.concept("q08-sufficient-statistic-2", "一様分布の十分統計量（分解定理、数値なし）")

# --- 分解定理 ---
xs = sp.symbols("x1:4", positive=True)
mu_, sg = sp.symbols("mu sigma", positive=True)
expo = sum((xi - mu_) ** 2 for xi in xs)
S1, S2 = sum(xs), sum(xi ** 2 for xi in xs)
q.check("q08-factorization-theorem-1", sp.expand(expo - (S2 - 2 * mu_ * S1 + 3 * mu_ ** 2)) == 0, "指数部は (Σx, Σx²) で書ける")
joint = sp.prod([th * xi ** (th - 1) for xi in xs])
Tp = sp.prod(xs)
q.check("q08-factorization-theorem-2", sp.simplify(sp.powsimp(joint, force=True) - th ** 3 * Tp ** (th - 1)) == 0
        or sp.simplify(sp.expand_power_base(th ** 3 * Tp ** (th - 1), force=True) - joint) == 0, "同時密度=θ^n (Πx)^(θ-1)")

# --- モーメント法 ---
x = np.array([2.0, 7.0, 1.5, 3.0, 2.0])
q.num("q08-method-of-moments-1", 2 * x.mean())
q.check("q08-method-of-moments-1", 2 * x.mean() < x.max(), "推定値 < 最大値 7.0")
a, b = sp.symbols("a b", positive=True)
sol = sp.solve([a * b - 6, a * b ** 2 - 12], [a, b], dict=True)[0]
q.num("q08-method-of-moments-2", float(sol[a]))

# --- MLE の漸近正規性 ---
q.num("q08-asymptotic-normality-of-mle-1", np.sqrt(0.3 * 0.7 / 400))
lf2 = sp.log(lam) - lam * X
Il = sp.simplify(-sp.diff(lf2, lam, 2))
q.check("q08-asymptotic-normality-of-mle-2", sp.simplify(Il - 1 / lam ** 2) == 0, "I(λ)=1/λ^2")
q.num("q08-asymptotic-normality-of-mle-2", 0.4 / np.sqrt(50))
q.concept("q08-asymptotic-normality-of-mle-3", "漸近有効性の定義（数値なし）")

# --- 不変性 ---
q.num("q08-invariance-of-mle-1", np.exp(-1.5))
nn, lv = 4, 1.5
Ee = sum(np.exp(-t / nn) * stats.poisson.pmf(t, nn * lv) for t in range(200))
q.check("q08-invariance-of-mle-2", abs(Ee - np.exp(nn * lv * (np.exp(-1 / nn) - 1))) < 1e-10
        and abs(Ee - 0.2652) < 5e-5 and abs(Ee - np.exp(-lv)) > 0.01, "E[e^{-X̄}]=0.2652≠0.2231（誤り）")

# --- ラオ・ブラックウェル ---
pr = [xx for xx in product([0, 1], repeat=5) if sum(xx) == 3]
q.num("q08-rao-blackwell-theorem-1", np.mean([xx[0] for xx in pr]))

# --- 完備 ---
q.num("q08-complete-statistic-1", (3 / 4) ** 6)
okc = all(abs(sum((3 / 4) ** t * stats.poisson.pmf(t, 4 * lv) for t in range(300)) - np.exp(-lv)) < 1e-10 for lv in [0.5, 1, 2.3])
q.check("q08-complete-statistic-1", okc, "((n-1)/n)^T は e^{-λ} の不偏推定量")

# --- 指数型分布族 ---
q.concept("q08-exponential-family-1", "指数型分布族の判定（台が母数に依存、数値なし）")
eta = sp.Symbol("eta")
A = sp.log(1 + sp.exp(eta))
q.num("q08-exponential-family-2", float(sp.diff(A, eta, 2).subs(eta, sp.log(3))))

# --- 観測情報量 ---
lam_hat = 30 / 10
q.num("q08-observed-fisher-information-1", 30 / lam_hat ** 2)

# --- 順序統計量（用語の見直しで追加） ---
from fractions import Fraction as _F
from math import comb as _comb
_E = _F(2, 5)
q.check("q08-order-statistics-1", abs(stats.beta(2, 3).mean() - 0.4) < 1e-12 and _E == _F(2, 5), "正解 E[X_(2)]=2/5（Beta(2,3)）")
q.check("q08-order-statistics-1", all(w != _E for w in [_F(1, 2), _F(3, 5), _F(1, 5)]), "誤答 1/2, 3/5, 1/5 は正解と異なる")
_p = sum(_comb(5, j) * 0.4 ** j * 0.6 ** (5 - j) for j in range(3, 6))
q.num("q08-order-statistics-2", _p)
q.check("q08-order-statistics-2", abs(stats.beta(3, 3).cdf(0.4) - _p) < 1e-12, "Beta(3,3) の分布関数とも一致")
_xx = sp.Symbol("x", positive=True)
_e2 = sp.integrate(_xx * 6 * (1 - sp.exp(-_xx)) * sp.exp(-2 * _xx), (_xx, 0, sp.oo))
q.num("q08-order-statistics-3", float(_e2))
_sim = np.sort(rng.exponential(1, size=(300000, 3)), axis=1)[:, 1].mean()
q.check("q08-order-statistics-3", _e2 == sp.Rational(5, 6) and abs(_sim - 5 / 6) < 0.01, "密度の積分 5/6、モンテカルロも一致")

# --- 標準誤差（用語の見直しで追加） ---
_sm = [rng.normal(0, 3, size=(20000, n)) for n in (25, 100)]
_se = [s.mean(axis=1).std() for s in _sm]
_sd = [s.std(axis=1, ddof=1).mean() for s in _sm]
q.check("q08-standard-error-1", abs(_se[1] / _se[0] - 0.5) < 0.03 and abs(_sd[1] / _sd[0] - 1) < 0.03, f"n を4倍で SE 比 {_se[1]/_se[0]:.3f}、s の比 {_sd[1]/_sd[0]:.3f}")
q.num("q08-standard-error-2", np.sqrt(6 ** 2 / 40 + 7.5 ** 2 / 50))
q.check("q08-standard-error-2", abs(np.sqrt(36 / 40) + np.sqrt(56.25 / 50) - 2.010) < 0.001, "誤った足し方は 2.010")

# --- 導出問題（追加） ---
def _deriv08():
    import sympy as _S
    m, th = _S.symbols("m theta", positive=True)
    n = _S.symbols("n", positive=True, integer=True)
    F = (m / th) ** n
    f = _S.simplify(_S.diff(F, m))
    ok = _S.simplify(f - n * m ** (n - 1) / th ** n) == 0
    E1 = _S.simplify(_S.integrate(m * f, (m, 0, th)))
    E2 = _S.simplify(_S.integrate(m ** 2 * f, (m, 0, th)))
    ok &= _S.simplify(E1 - n * th / (n + 1)) == 0 and _S.simplify(E2 - n * th ** 2 / (n + 2)) == 0
    VM = _S.simplify(E2 - E1 ** 2)
    ok &= _S.simplify(VM - n * th ** 2 / ((n + 1) ** 2 * (n + 2))) == 0
    ok &= _S.simplify((n + 1) / n * E1 - th) == 0
    ok &= _S.simplify(((n + 1) / n) ** 2 * VM - th ** 2 / (n * (n + 2))) == 0
    q.check("q08-order-statistics-4", bool(ok), "sympy: f_M=nm^{n-1}/θ^n、E[M]=nθ/(n+1)、V[T]=θ²/(n(n+2))")
    lam, x = _S.symbols("lambda x", positive=True)
    lf = x * _S.log(lam) - lam - _S.log(_S.factorial(x))
    d1 = _S.diff(lf, lam); d2 = _S.diff(lf, lam, 2)
    ok2 = _S.simplify(d1 - (x / lam - 1)) == 0 and _S.simplify(d2 + x / lam ** 2) == 0
    I = -d2.subs(x, lam)          # E[X]=λ を代入（d2 は x の1次式）
    ok2 &= _S.simplify(I - 1 / lam) == 0
    k = _S.symbols("k", integer=True, nonnegative=True)
    pmf = _S.exp(-lam) * lam ** k / _S.factorial(k)
    I_sum = _S.simplify(_S.summation((k / lam - 1) ** 2 * pmf, (k, 0, _S.oo)))   # E[score²] でも同じ
    ok2 &= _S.simplify(I_sum - 1 / lam) == 0
    ok2 &= _S.simplify(1 / (n * I) - lam / n) == 0
    q.check("q08-cramer-rao-inequality-4", bool(ok2), "sympy: I(λ)=1/λ（E[score²] でも一致）、下限 λ/n = V[X̄]")
_deriv08()


# --- 4択の追加 ---
def _choice08():
    import numpy as _np
    import sympy as _S
    # 指数分布のスコア関数
    lam, n, T = _S.symbols("lambda n T", positive=True)    # T = Σx_i
    ell = n * _S.log(lam) - lam * T
    U = _S.diff(ell, lam)
    ch = [n * _S.log(lam) - lam * T, -n / lam ** 2, n / lam ** 2, n / lam - T]
    ans = q.qs["q08-score-function-3"]["answer"]
    q.check("q08-score-function-3",
            _S.simplify(ch[ans] - U) == 0 and all(_S.simplify(c - U) != 0 for i, c in enumerate(ch) if i != ans),
            f"sympy: U(λ)={U}、誤答（ℓ, ℓ'', nI）は一致しない")
    q.check("q08-score-function-3",
            _S.simplify(_S.diff(U, lam) - ch[1]) == 0 and _S.solve(U, lam) == [n / T],
            "誤答の由来：ℓ''=-n/λ²、U=0 から λ̂=n/Σx=1/x̄")

    # モーメント法：σ² = m2 - m1²
    x = _np.array([2, 4, 4, 6, 9], dtype=float)
    est = _np.mean(x ** 2) - _np.mean(x) ** 2
    ch = [5.60, 7.00, 2.37, 30.60]
    ans = q.qs["q08-method-of-moments-3"]["answer"]
    q.check("q08-method-of-moments-3",
            abs(ch[ans] - est) < 0.005 and all(abs(c - est) >= 0.005 for i, c in enumerate(ch) if i != ans),
            f"m2-m1²={est:.4f}（= ddof=0 の分散 {x.var():.4f}）、誤答は一致しない")
    q.check("q08-method-of-moments-3",
            abs(x.var(ddof=1) - 7.0) < 1e-12 and abs(_np.sqrt(est) - 2.37) < 0.005 and abs(_np.mean(x ** 2) - 30.6) < 1e-12,
            "誤答の由来：不偏分散 7.00、√5.6=2.37、m2=30.60")

    # ベルヌーイの MLE の漸近分散
    p = _S.symbols("p", positive=True)
    X = _S.symbols("X")
    logf = X * _S.log(p) + (1 - X) * _S.log(1 - p)
    d2 = _S.diff(logf, p, 2)
    I = _S.simplify(-(d2.subs(X, 1) * p + d2.subs(X, 0) * (1 - p)))
    av = float((1 / (100 * I)).subs(p, _S.Rational(1, 5)))
    ch = [0.0400, 0.1600, 0.0016, 0.0025]
    ans = q.qs["q08-asymptotic-normality-of-mle-4"]["answer"]
    q.check("q08-asymptotic-normality-of-mle-4",
            _S.simplify(I - 1 / (p * (1 - p))) == 0 and abs(ch[ans] - av) < 1e-12 and all(abs(c - av) > 1e-6 for i, c in enumerate(ch) if i != ans),
            f"sympy: I(p)={I}、1/(nI)={av}、誤答は一致しない")
_choice08()

q.done()
