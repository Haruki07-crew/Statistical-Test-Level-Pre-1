"""第4章（変数変換）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, integrate

q = Q(4)

# --- 変数変換とヤコビアン ---
# 1: Y=2X+3 の密度。X~N(0,1) で f_X((y-3)/2) の積分が 2 になる（＝誤り）、1/2 を掛けると 1
f = lambda y: stats.norm.pdf((y - 3) / 2)
tot = integrate.quad(f, -np.inf, np.inf)[0]
q.check("q04-change-of-variables-1", abs(tot - 2) < 1e-8 and abs(tot / 2 - 1) < 1e-8,
        "係数なしの積分=2（誤り）、1/2 を掛けると1")
# 2
y = sp.symbols("y", positive=True)
x = sp.exp(-y / 2)
fy = sp.simplify(1 * sp.Abs(sp.diff(x, y)))
p = float(sp.integrate(fy, (y, 1, 4)))
q.num("q04-change-of-variables-2", p)
q.check("q04-change-of-variables-2", abs(p - (stats.expon(scale=2).cdf(4) - stats.expon(scale=2).cdf(1))) < 1e-12,
        "Y はレート1/2の指数分布と一致")
# 3
u, v = sp.symbols("u v", positive=True)
xs, ys = u * v, u * (1 - v)
J = sp.Matrix([[sp.diff(xs, u), sp.diff(xs, v)], [sp.diff(ys, u), sp.diff(ys, v)]]).det()
fuv = sp.simplify(sp.exp(-(xs + ys)) * sp.Abs(J))
q.check("q04-change-of-variables-3", sp.simplify(J + u) == 0 and sp.simplify(fuv - u * sp.exp(-u)) == 0,
        "ヤコビアン=-u、同時密度 u e^{-u}")
fu = sp.integrate(fuv, (v, 0, 1))
fv = sp.integrate(fuv, (u, 0, sp.oo))
q.check("q04-change-of-variables-3", sp.simplify(fu - u * sp.exp(-u)) == 0 and fv == 1
        and sp.simplify(fuv - fu * fv) == 0, "周辺は Ga(2,1) と U(0,1)、積に分解（独立）")

# --- 対数変換 ---
mu, sd = 50, 5
g1 = sp.diff(sp.log(sp.Symbol("x")), sp.Symbol("x")).subs(sp.Symbol("x"), mu)
q.num("q04-log-transformation-1", float(abs(g1) * sd))
data = np.array([1, 4, 16, 64])
gm = 10 ** np.mean(np.log10(data))
q.check("q04-log-transformation-2", abs(gm - 8) < 1e-12, "正解：幾何平均8")
q.check("q04-log-transformation-2",
        abs(data.mean() - 21.25) < 1e-12 and np.median(data) == 10 and abs(stats.hmean(data) - 3.01) < 0.005
        and all(abs(w - gm) > 0.5 for w in [data.mean(), np.median(data), stats.hmean(data)]),
        "誤答（算術平均21.25・中央値10・調和平均3.01）は正解と異なる")

# --- べき乗変換 ---
lam, m, c = sp.symbols("lambda mu c", positive=False)
expo = sp.solve(sp.Eq(2 * (lam - 1) + 3, 0), lam)[0]
q.num("q04-power-transformation-1", float(expo))
mm = sp.symbols("m", positive=True)
var_expr = (expo * mm ** (expo - 1)) ** 2 * (mm ** sp.Rational(3, 2)) ** 2
q.check("q04-power-transformation-1", sp.simplify(sp.diff(var_expr, mm)) == 0, "λ=-1/2 で近似分散が μ によらない")

# --- Box-Cox ---
q.num("q04-box-cox-transformation-1", (9 ** -1 - 1) / -1)
q.concept("q04-box-cox-transformation-2", "ヤコビアン項の必要性（数値を含まない）")

# --- ロジット・ロジスティック ---
q.num("q04-logit-logistic-transformation-1", np.log(0.8 / 0.2))
xx = np.array([0.0, 3.0, 7.0])
odds = lambda t: np.exp(-2 + 0.5 * t)
ratio = odds(xx + 2) / odds(xx)
q.check("q04-logit-logistic-transformation-2", np.allclose(ratio, np.e), "x+2 でオッズ比 e（どの x でも）")
pr = lambda t: 1 / (1 + np.exp(-(-2 + 0.5 * t)))
q.check("q04-logit-logistic-transformation-2",
        not np.allclose(ratio, np.exp(0.5)) and not np.allclose(pr(xx + 2) / pr(xx), np.e)
        and not np.allclose(odds(xx + 2) - odds(xx), 1),
        "誤答（e^0.5倍、確率がe倍、オッズ+1）は成り立たない")
P = sp.Symbol("p")
d = sp.diff(sp.log(P / (1 - P)), P)
se = float(sp.sqrt(d ** 2 * P * (1 - P) / 100).subs(P, sp.Rational(1, 5)))
q.num("q04-logit-logistic-transformation-3", se)

# --- プロビット ---
q.num("q04-probit-transformation-1", stats.norm.ppf(0.975))
xs_ = sp.Symbol("x")
Phi = (1 + sp.erf((-1 + sp.Rational(2, 5) * xs_) / sp.sqrt(2))) / 2
me = float(sp.diff(Phi, xs_).subs(xs_, 5))
q.num("q04-probit-transformation-2", me)
q.check("q04-probit-transformation-2", abs(stats.norm.pdf(1) - 0.2420) < 5e-5, "φ(1)=0.2420")

# --- 確率変数の和（用語の見直しで追加） ---
import numpy as _np
import sympy as _sp
from scipy import stats as _st
_pmf = _np.convolve(_st.binom.pmf(range(6), 5, 0.3), _st.binom.pmf(range(5), 4, 0.5))
_m = (_np.arange(10) * _pmf).sum()
_p = _m / 9
_dev = _np.abs(_pmf - _st.binom.pmf(range(10), 9, _p)).max()
_po = _np.convolve(_st.poisson.pmf(range(40), 1), _st.poisson.pmf(range(40), 2))[:40]
q.check("q04-sum-of-random-variables-1", _dev > 1e-3, f"B(5,0.3)+B(4,0.5) は二項分布でない（最大差 {_dev:.4f}）")
q.check("q04-sum-of-random-variables-1", _np.abs(_po - _st.poisson.pmf(range(40), 3)).max() < 1e-12, "誤答の Po(1)+Po(2)=Po(3) は再生性が成り立つ")
from scipy import integrate as _ig
_conv = [_ig.quad(lambda u: _st.chi2.pdf(u, 2) * _st.chi2.pdf(z - u, 3), 0, z)[0] for z in [1.0, 4.0, 9.0]]
q.check("q04-sum-of-random-variables-1", _np.allclose(_conv, _st.chi2.pdf([1.0, 4.0, 9.0], 5), atol=1e-8), "誤答の χ²(2)+χ²(3)=χ²(5)、N(0,1)+N(1,4)=N(1,5) は再生性")
_z = _sp.symbols("z", positive=True)
q.num("q04-sum-of-random-variables-2", float(_sp.integrate(_z, (_z, _sp.Rational(1, 2), 1)) + _sp.integrate(2 - _z, (_z, 1, _sp.Rational(3, 2)))))
_rng = _np.random.default_rng(4)
_s = _rng.random(400000) + _rng.random(400000)
q.check("q04-sum-of-random-variables-2", abs(_np.mean((_s > 0.5) & (_s < 1.5)) - 0.75) < 0.005, "モンテカルロでも 0.75")
_l1, _l2, _x = _sp.symbols("lambda1 lambda2 x", positive=True)
_f = _sp.integrate(_l1 * _sp.exp(-_l1 * _x) * _l2 * _sp.exp(-_l2 * (_z - _x)), (_x, 0, _z), conds="none")
_target = _l1 * _l2 / (_l2 - _l1) * (_sp.exp(-_l1 * _z) - _sp.exp(-_l2 * _z))
_okd = all(abs(float((_f - _target).subs({_l1: a, _l2: b, _z: c}))) < 1e-12 for a, b, c in [(1, 2, 0.7), (0.5, 3, 2.0), (2.5, 1.2, 1.1)])
_int1 = _sp.simplify(_sp.integrate(_target.subs({_l1: 1, _l2: 3}), (_z, 0, _sp.oo)))
q.check("q04-sum-of-random-variables-3", _okd and _int1 == 1, "畳み込みの結果が最終式と一致し、積分が1")

# --- 導出問題（追加） ---
def _deriv04():
    import sympy as _S
    y_, u_ = _S.symbols("y u", real=True)
    uinv = _S.exp(y_) / (1 + _S.exp(y_))
    ok = _S.simplify(_S.log(uinv / (1 - uinv)) - y_) == 0
    du = _S.diff(uinv, y_)
    ok &= _S.simplify(du - uinv * (1 - uinv)) == 0
    f = _S.exp(y_) / (1 + _S.exp(y_)) ** 2
    ok &= _S.simplify(du - f) == 0 and _S.simplify(f.subs(y_, -y_) - f) == 0
    ok &= _S.integrate(f, (y_, -_S.oo, _S.oo)) == 1
    ok &= _S.simplify(_S.integrate(f.subs(y_, _S.Symbol("z")), (_S.Symbol("z"), -_S.oo, y_)) - 1 / (1 + _S.exp(-y_))) == 0
    q.check("q04-logit-logistic-transformation-4", bool(ok), "sympy: du/dy=u(1-u)=e^y/(1+e^y)^2、積分1、F=1/(1+e^-y)")
    r, th, w = _S.symbols("r theta w", positive=True)
    J = _S.Matrix([r * _S.cos(th), r * _S.sin(th)]).jacobian([r, th]).det()
    ok2 = _S.simplify(J - r) == 0
    g = _S.exp(-(r ** 2) / 2) / (2 * _S.pi) * J
    ok2 &= _S.simplify(_S.integrate(g, (th, 0, 2 * _S.pi), (r, 0, _S.oo)) - 1) == 0
    ok2 &= _S.simplify(_S.integrate(g, (th, 0, 2 * _S.pi)) - r * _S.exp(-r ** 2 / 2)) == 0
    fw = (r * _S.exp(-r ** 2 / 2)).subs(r, _S.sqrt(w)) * _S.diff(_S.sqrt(w), w)
    ok2 &= _S.simplify(fw - _S.exp(-w / 2) / 2) == 0
    q.check("q04-change-of-variables-4", bool(ok2), "sympy: ヤコビアン r、g が r と θ の積、f_W=e^{-w/2}/2")
_deriv04()


# --- 4択の追加 ---
def _choice04():
    import numpy as _np
    import sympy as _S
    y, lam = _S.symbols("y lambda", positive=True)
    # Y=X^2, X~U(0,1)：F_Y(y)=sqrt(y) を微分した密度と比べる
    true_pdf = _S.diff(_S.sqrt(y), y)
    ch = [1 / (2 * _S.sqrt(y)), _S.Integer(1), 2 * _S.sqrt(y), _S.sqrt(y)]
    ans = q.qs["q04-change-of-variables-5"]["answer"]
    q.check("q04-change-of-variables-5",
            _S.simplify(ch[ans] - true_pdf) == 0 and all(_S.simplify(c - true_pdf) != 0 for i, c in enumerate(ch) if i != ans),
            "sympy: d/dy P(X<=sqrt y)=1/(2sqrt y)、誤答は一致しない")
    xs = _np.random.default_rng(4).uniform(0, 1, 400000) ** 2
    emp = _np.mean(xs <= 0.25)
    q.check("q04-change-of-variables-5", abs(emp - float(_S.integrate(ch[ans], (y, 0, _S.Rational(1, 4))))) < 0.005,
            f"シミュレーション P(Y<=0.25)≈{emp:.4f}（正解の密度の積分 0.5 と一致）")

    # Box-Cox の lambda→0 の極限
    lim = _S.limit((y ** lam - 1) / lam, lam, 0)
    ch = [y - 1, 1 - 1 / y, _S.log(y), _S.Integer(0)]
    ans = q.qs["q04-box-cox-transformation-3"]["answer"]
    v = {c: float(c.subs(y, 3)) for c in ch}
    q.check("q04-box-cox-transformation-3",
            _S.simplify(ch[ans] - lim) == 0 and all(abs(float(c.subs(y, 3)) - float(lim.subs(y, 3))) > 1e-6 for i, c in enumerate(ch) if i != ans),
            f"sympy: 極限 {lim}、y=3 で誤答の値は一致しない")
    q.check("q04-box-cox-transformation-3",
            _S.simplify(((y ** 1 - 1) / 1) - ch[0]) == 0 and _S.simplify(((y ** -1 - 1) / -1) - ch[1]) == 0,
            "誤答の由来：λ=1 で y-1、λ=-1 で 1-1/y")
_choice04()

# --- 追加：答えが「正しい」の正誤問題 ---
import numpy as _np
import sympy as _sp
from scipy import stats as _st
# 指数分布：F(x)=1-e^{-x} で P(F(X)≤u)=u（記号計算）
_u = _sp.symbols("u", positive=True)
_xq = -_sp.log(1 - _u)  # F^{-1}(u)
_cdfU = 1 - _sp.exp(-_xq)
# ガンマ分布の乱数で F(X) が一様か（乱数固定）
_rng = _np.random.default_rng(404)
_g = _st.gamma(2.5, scale=1.3)
_U = _g.cdf(_g.rvs(size=20000, random_state=_rng))
_ks = _st.kstest(_U, "uniform").pvalue
# 離散の反例：ベルヌーイ(1/2) の F(X) は {1/2, 1}
_vals = sorted({_st.bernoulli(0.5).cdf(k) for k in (0, 1)})
q.check("q04-change-of-variables-6", _sp.simplify(_cdfU - _u) == 0 and _ks > 0.01 and _vals == [0.5, 1.0],
        f"指数分布で P(U≤u)=u、ガンマ乱数で KS p={_ks:.3f}、ベルヌーイでは F(X)∈{{1/2,1}}")

# V[X-Y]：独立な離散分布の列挙と、相関がある反例
_px = {0: 0.2, 1: 0.5, 3: 0.3}; _py = {-1: 0.4, 2: 0.6}
_E = lambda d: sum(k * p for k, p in d.items())
_V = lambda d: sum(k * k * p for k, p in d.items()) - _E(d) ** 2
_dz = {}
for _a, _pa in _px.items():
    for _b, _pb in _py.items():
        _dz[_a - _b] = _dz.get(_a - _b, 0) + _pa * _pb
_Z = _rng.multivariate_normal([0, 0], [[4, 1.5], [1.5, 1]], size=200000)
_vd = _np.var(_Z[:, 0] - _Z[:, 1])
q.check("q04-sum-of-random-variables-4", abs(_V(_dz) - (_V(_px) + _V(_py))) < 1e-12 and abs(_vd - (4 + 1 - 3)) < 0.05,
        "独立なら V[X-Y]=V[X]+V[Y]（列挙で一致）、Cov=1.5 なら V[X-Y]=4+1-3=2（乱数で確認）")

q.done()
