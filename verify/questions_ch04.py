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

q.done()
