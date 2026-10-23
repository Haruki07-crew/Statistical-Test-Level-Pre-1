"""第19章（回帰分析その他）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import optimize

q = Q(19)
rng = np.random.default_rng(19)

# --- WLS ---
x = np.array([1, 2, 4.]); y = np.array([3, 5, 10.]); w = 1 / x
X = x[:, None]
bw = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))[0]
q.num("q19-weighted-least-squares-1", bw)
q.check("q19-weighted-least-squares-1", abs(bw - y.sum() / x.sum()) < 1e-12 and abs((x @ y) / (x @ x) - 2.524) < 0.001, "Σy/Σx、OLS 2.524")
# 重み 1/分散 と 1/標準偏差 の分散比較（σ_i^2 = x_i^2 の原点回帰）
xs = np.linspace(1, 10, 20); s2 = xs ** 2
var_w = lambda ww: np.sum(ww ** 2 * xs ** 2 * s2) / np.sum(ww * xs ** 2) ** 2
q.check("q19-weighted-least-squares-2", var_w(1 / s2) < var_w(1 / np.sqrt(s2)) - 1e-6, "1/分散の重みのほうが分散が小さい（誤り）")

# --- GLS ---
q.concept("q19-generalized-least-squares-1", "Ω対角ならWLS")
O = np.array([[1, .5], [.5, 4]]); Oi = np.linalg.inv(O); one = np.ones(2); yv = np.array([10, 14.])
mu = one @ Oi @ yv / (one @ Oi @ one)
q.num("q19-generalized-least-squares-2", mu)
q.check("q19-generalized-least-squares-2", abs(1 / (one @ Oi @ one) - 0.9375) < 1e-12, "分散 0.9375σ²")

# --- リッジ ---
A = np.array([[1, .9], [.9, 1]]); c = np.array([1.5, 1.2])
br = np.linalg.solve(A + 0.1 * np.eye(2), c)
q.num("q19-ridge-regression-1", br[1])
q.check("q19-ridge-regression-1", np.allclose(np.round(np.linalg.solve(A, c), 3), [2.211, -0.789]) and abs(br[0] - 1.425) < 1e-9, "OLS (2.211,-0.789)、第1成分 1.425")
Xr = rng.normal(size=(30, 4)); yr = Xr @ [1, 0.2, 0, -0.5] + rng.normal(size=30)
nz = all(np.all(np.abs(np.linalg.solve(Xr.T @ Xr + lam * np.eye(4), Xr.T @ yr)) > 1e-8) for lam in [0.1, 10, 1e3, 1e5])
q.check("q19-ridge-regression-2", nz, "有限のλでは係数はちょうど0にならない（正しい）")

# --- Lasso ---
b = np.array([2.5, -0.6, 1.1]); lam = 0.8
soft = np.sign(b) * np.maximum(np.abs(b) - lam, 0)
# 目的関数を直接最小化（直交計画では成分ごとに 1/2(β-b)^2 + λ|β|）
direct = np.array([optimize.minimize_scalar(lambda t, bj=bj: 0.5 * (t - bj) ** 2 + lam * abs(t), bounds=(-5, 5), method="bounded").x for bj in b])
q.check("q19-lasso-1", np.allclose(soft, [1.7, 0, 0.3]) and np.allclose(direct, soft, atol=1e-4), "正解 (1.7,0,0.3)")
q.check("q19-lasso-1", not np.allclose(b / 1.8, soft) and not np.allclose(np.where(np.abs(b) > lam, b, 0), soft), "誤答（リッジ・ハード閾値）は正解と異なる")


def lasso_cd(X, y, lam, it=2000):
    bb = np.zeros(X.shape[1])
    for _ in range(it):
        for j in range(X.shape[1]):
            r = y - X @ bb + X[:, j] * bb[j]
            z = X[:, j] @ r
            bb[j] = np.sign(z) * max(abs(z) - lam, 0) / (X[:, j] @ X[:, j])
    return bb


Xl = rng.normal(size=(40, 3)); yl = Xl @ [1.0, 0.5, 0.3] + rng.normal(size=40); yl -= yl.mean(); Xl -= Xl.mean(0)
b1 = lasso_cd(Xl, yl, 8.0); Xs = Xl.copy(); Xs[:, 0] *= 100  # 単位を変える
b2 = lasso_cd(Xs, yl, 8.0)
q.check("q19-lasso-2", not np.allclose(Xl @ b1, Xs @ b2, atol=1e-3), "単位を変えると当てはめが変わる（誤り）")

# --- 正則化 ---
q.concept("q19-regularization-1", "L1 で係数がちょうど0")
rss = [np.sum((yr - Xr @ np.linalg.solve(Xr.T @ Xr + l * np.eye(4), Xr.T @ yr)) ** 2) for l in [0, 0.5, 2, 10, 50]]
q.check("q19-regularization-2", all(np.diff(rss) > 0), "λ を大きくすると訓練RSSは増える（誤り）")

# --- エラスティックネット ---
en = optimize.minimize_scalar(lambda t: 0.5 * (t - 2.0) ** 2 + 0.5 * abs(t) + 0.5 * t ** 2, bounds=(-5, 5), method="bounded").x
q.num("q19-elastic-net-1", en)
q.check("q19-elastic-net-1", abs(en - 1.5 / 2) < 1e-4, "公式 (|b|-λ1)/(1+λ2)")

# --- 罰則付き推定量のバイアス ---
L = sp.Symbol("lam", positive=True)
mse = (1 + L ** 2 * 4) / (1 + L) ** 2
lopt = sp.solve(sp.diff(mse, L), L)
q.num("q19-bias-in-penalized-estimation-1", float(mse.subs(L, lopt[0])))
q.check("q19-bias-in-penalized-estimation-1", lopt == [sp.Rational(1, 4)], "最適 λ=1/4")
# シミュレーション：b ~ N(2,1)、b/(1.25) の MSE
bs = 2 + rng.normal(size=400000)
q.check("q19-bias-in-penalized-estimation-1", abs(np.mean((bs / 1.25 - 2) ** 2) - 0.8) < 0.01, "シミュレーションで MSE≈0.8")

# --- 非線形回帰 ---
a_, b_, c_, xx = sp.symbols("a b c x")
models = [a_ * sp.exp(-b_ * xx) + c_, a_ + b_ * xx + c_ * xx ** 2, a_ + b_ * sp.log(xx), a_ * sp.sin(xx) + b_ * sp.cos(xx)]
lin = [all(sp.diff(m, p1, p2) == 0 for p1 in (a_, b_, c_) for p2 in (a_, b_, c_)) for m in models]
q.check("q19-nonlinear-regression-1", lin == [False, True, True, True], "パラメータの2階微分が0でないのは第1のモデルだけ")
q.num("q19-nonlinear-regression-2", 100 / (1 + np.exp(-0.5 * (12 - 10))))

# --- 多項式回帰 ---
xs_ = -4 / (2 * -0.5); yv_ = 3 + 4 * xs_ - 0.5 * xs_ ** 2
q.num("q19-polynomial-regression-1", xs_ + yv_)
xp = np.linspace(-2, 3, 15); yp = 1 + 2 * xp - 0.7 * xp ** 2 + rng.normal(size=15)
Xp = np.column_stack([np.ones(15), xp, xp ** 2])
bp_closed = np.linalg.solve(Xp.T @ Xp, Xp.T @ yp)
bp_iter = optimize.least_squares(lambda bb: yp - Xp @ bb, np.zeros(3)).x
q.check("q19-polynomial-regression-2", np.allclose(bp_closed, bp_iter, atol=1e-6), "閉じた形の解が非線形最小二乗の解と一致（反復不要＝誤り）")

# --- 分位点回帰 ---
rho = lambda u, t: np.where(u >= 0, t * u, (t - 1) * u)
q.num("q19-quantile-regression-1", rho(np.array([2, -1, 0.5, -3.]), 0.9).sum())
yq = rng.normal(size=21)
cq = optimize.minimize_scalar(lambda cc: rho(yq - cc, 0.5).sum(), bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
q.check("q19-quantile-regression-2", abs(cq - np.median(yq)) < 1e-5, "τ=0.5 の切片だけの解＝中央値")

# --- 操作変数 ---
q.num("q19-instrumental-variable-method-1", (25.6 - 22.6) / (12.8 - 12.4))
# シミュレーション：条件を満たす z で IV が一致、Cov(z,ε)≠0 だと一致しない
n = 200000
z = rng.integers(0, 2, n).astype(float); u = rng.normal(size=n)
xv = 1 + 0.8 * z + u + rng.normal(size=n); eps = u + rng.normal(size=n)
yv = 2 + 1.5 * xv + eps
iv = np.cov(z, yv)[0, 1] / np.cov(z, xv)[0, 1]
iv_bad = np.cov(z, yv + 0.5 * z)[0, 1] / np.cov(z, xv)[0, 1]  # z が直接 y に効く＝Cov(z,ε)≠0
q.check("q19-instrumental-variable-method-2", abs(iv - 1.5) < 0.03 and abs(iv_bad - 1.5) > 0.3, f"関連性＋外生性で一致 {iv:.3f}、外生性違反で {iv_bad:.3f}")

# --- 内生性 ---
xstar = rng.normal(scale=2, size=n); xo = xstar + rng.normal(size=n); yy = 1 + 2 * xstar + rng.normal(size=n)
q.num("q19-endogeneity-1", 2 * 4 / 5)
q.check("q19-endogeneity-1", abs(np.cov(xo, yy)[0, 1] / np.var(xo, ddof=1) - 1.6) < 0.02, "シミュレーションで 1.6")
slopes = []
for m in [1000, 100000]:
    xs1 = rng.normal(size=m); e1 = 0.5 * xs1 + rng.normal(size=m)
    slopes.append(np.polyfit(xs1, 1 + 2 * xs1 + e1, 1)[0])
q.check("q19-endogeneity-2", abs(slopes[-1] - 2.5) < 0.02, f"n を増やしても 2 ではなく 2.5 に収束 {slopes}")

# --- 2SLS ---
Z = np.column_stack([np.ones(n), z]); Xm = np.column_stack([np.ones(n), xv])
Xh = Z @ np.linalg.lstsq(Z, Xm, rcond=None)[0]
b2s = np.linalg.lstsq(Xh, yv, rcond=None)[0]
s2_wrong = np.mean((yv - Xh @ b2s) ** 2); s2_right = np.mean((yv - Xm @ b2s) ** 2)
q.check("q19-two-stage-least-squares-1", abs(b2s[1] - 1.5) < 0.03 and abs(s2_wrong / s2_right - 1) > 0.2,
        f"第2段階残差の分散 {s2_wrong:.2f} ≠ 元のXの残差 {s2_right:.2f}（誤り）")

# --- 欠落変数バイアス ---
q.num("q19-omitted-variable-bias-1", 0.5 + 2 * 1.2 / 4)
C = np.array([[4, 1.2], [1.2, 1.0]]); xx2 = rng.multivariate_normal([0, 0], C, size=n)
yo = 0.5 * xx2[:, 0] + 2 * xx2[:, 1] + rng.normal(size=n)
q.check("q19-omitted-variable-bias-1", abs(np.polyfit(xx2[:, 0], yo, 1)[0] - 1.1) < 0.02, "シミュレーションで 1.1")
yo2 = 0.5 * xx2[:, 0] - 2 * xx2[:, 1] + rng.normal(size=n)
q.check("q19-omitted-variable-bias-2", np.polyfit(xx2[:, 0], yo2, 1)[0] < 0.5 - 0.3, "β2<0・正の相関で過小")

# --- 変数選択 ---
aic = lambda se, k: 50 * np.log(se / 50) + 2 * (k + 2)
bic = lambda se, k: 50 * np.log(se / 50) + (k + 2) * np.log(50)
q.check("q19-variable-selection-1", aic(105, 4) < aic(120, 2) and bic(120, 2) < bic(105, 4), "AIC→B、BIC→A")
Xv = rng.normal(size=(30, 5)); yv5 = Xv[:, 0] + rng.normal(size=30)
r2 = []
for k in range(1, 6):
    Xk = np.column_stack([np.ones(30), Xv[:, :k]]); e = yv5 - Xk @ np.linalg.lstsq(Xk, yv5, rcond=None)[0]
    r2.append(1 - e @ e / np.sum((yv5 - yv5.mean()) ** 2))
q.check("q19-variable-selection-2", all(np.diff(r2) >= -1e-12), "R² は単調増加し全部入りが最大（誤り）")

# --- PCR ---
# y が第2主成分（分散小）方向だけに依存する例：上位1成分のPCRでは説明できない
X2 = rng.multivariate_normal([0, 0], [[1, .95], [.95, 1]], size=2000); X2 -= X2.mean(0)
ev, V = np.linalg.eigh(X2.T @ X2); v1, v2 = V[:, 1], V[:, 0]
y2 = X2 @ v2 * 3 + rng.normal(scale=0.1, size=2000)
z1 = X2 @ v1; r2_pc1 = np.corrcoef(z1, y2)[0, 1] ** 2
q.check("q19-principal-component-regression-1", r2_pc1 < 0.01, f"第1主成分だけでは R²={r2_pc1:.4f}（正しい）")

# --- スプライン ---
kn = [2, 4, 6, 8]; xg = np.linspace(0, 10, 50)
B = np.column_stack([xg ** d for d in range(4)] + [np.maximum(xg - k, 0) ** 3 for k in kn])
q.num("q19-spline-regression-1", np.linalg.matrix_rank(B))

# --- ニューラルネットワーク（用語の見直しで追加） ---
_p, _H = 5, 10
q.num("q19-neural-network-model-1", _p * _H + _H + _H + 1)

# --- 導出問題（追加） ---
def _deriv19():
    import numpy as _N
    from scipy import optimize as _op
    rng = _N.random.default_rng(19)
    X = rng.normal(size=(25, 3)); y = rng.normal(size=25); lam = 0.7
    bR = _N.linalg.solve(X.T @ X + lam * _N.eye(3), X.T @ y)
    f = lambda b: ((y - X @ b) ** 2).sum() + lam * (b ** 2).sum()
    res = _op.minimize(f, _N.zeros(3), tol=1e-12)
    ok = _N.allclose(res.x, bR, atol=1e-6)
    # 列が線形従属でも解ける
    Xs = _N.column_stack([X[:, 0], X[:, 0], X[:, 1]])
    ok &= _N.linalg.matrix_rank(Xs.T @ Xs) == 2 and _N.all(_N.linalg.eigvalsh(Xs.T @ Xs + lam * _N.eye(3)) > 0)
    Q, _ = _N.linalg.qr(rng.normal(size=(25, 3)))      # Q'Q=I
    b = Q.T @ y
    ok &= _N.allclose(_N.linalg.solve(Q.T @ Q + lam * _N.eye(3), Q.T @ y), b / (1 + lam))
    beta = _N.array([1.0, -2.0, 0.5])
    Ys = Q @ beta + rng.normal(0, 1, size=(100000, 25))
    ok &= _N.allclose((Ys @ Q / (1 + lam)).mean(0), beta / (1 + lam), atol=0.01)
    q.check("q19-ridge-regression-3", bool(ok), "数値最小化と (X'X+λI)^-1X'y が一致、直交計画で b/(1+λ)、E=β/(1+λ)")
    # Lasso：直交計画で残差平方和の分解と軟しきい値
    bb = _N.array([2.5, -0.6, 1.1, -1.9, 0.3]); lam2 = 0.8
    Q5, _ = _N.linalg.qr(rng.normal(size=(30, 5))); y5 = Q5 @ bb + rng.normal(size=30) * 0.0
    y5 = y5 + (_N.eye(30) - Q5 @ Q5.T) @ rng.normal(size=30)   # 直交補空間の成分を加える（b は変わらない）
    b5 = Q5.T @ y5
    ok2 = _N.allclose(b5, bb)
    beta_t = rng.normal(size=5)
    lhs = 0.5 * ((y5 - Q5 @ beta_t) ** 2).sum()
    rhs = 0.5 * ((y5 - Q5 @ b5) ** 2).sum() + 0.5 * ((beta_t - b5) ** 2).sum()
    ok2 &= abs(lhs - rhs) < 1e-10
    soft = _N.sign(b5) * _N.maximum(_N.abs(b5) - lam2, 0)
    grid = _N.linspace(-4, 4, 800001)
    for j in range(5):
        obj = 0.5 * (grid - b5[j]) ** 2 + lam2 * _N.abs(grid)
        ok2 &= abs(grid[obj.argmin()] - soft[j]) < 2e-5
    ok2 &= soft[1] == 0 and soft[4] == 0
    q.check("q19-lasso-3", bool(ok2), "残差平方和の分解、グリッド最小化と軟しきい値 sign(b)(|b|-λ)+ が一致")
_deriv19()

# --- 4択の追加分 ---
_rng = np.random.default_rng(1904)
_X = _rng.normal(size=(50, 3)); _X[:, 1] = _X[:, 0] + 0.3 * _rng.normal(size=50)
_beta = np.array([1.0, 2.0, -1.5]); _sig = 1.0; _G = _X.T @ _X; _I3 = np.eye(3)
def _bv(lam):
    A = np.linalg.inv(_G + lam * _I3)
    bias = A @ _G @ _beta - _beta; var = _sig ** 2 * np.trace(A @ _G @ A)
    return np.linalg.norm(bias), var
_lams = [0.0, 0.5, 2.0, 10.0, 50.0]; _B, _V = zip(*[_bv(l) for l in _lams])
_mono = all(np.diff(_B) > 0) and all(np.diff(_V) < 0)
_yy = _X @ _beta + _rng.normal(size=50)
_bR = np.linalg.solve(_G + 10 * _I3, _X.T @ _yy)
_Xs = _X * np.array([100.0, 1, 1])     # 1列目の単位を変える
_bRs = np.linalg.solve(_Xs.T @ _Xs + 10 * _I3, _Xs.T @ _yy)
_scale_changes = not np.allclose(_X @ _bR, _Xs @ _bRs)
q.check("q19-ridge-regression-4", _mono and _B[1] > 1e-6 and np.all(np.abs(_bR) > 1e-6) and _scale_changes,
        f"λ↑でバイアス {np.round(_B, 3)} は増加、分散 {np.round(_V, 3)} は減少。λ=10 でも係数は0でない、単位変更で予測値が変わる")
_n = 200000; _x = _rng.normal(size=_n)
_eh = _rng.normal(size=_n) * (0.5 + np.abs(_x))      # 不均一分散（外生）
_slope = lambda xx, yy: np.cov(xx, yy)[0, 1] / np.var(xx, ddof=1)
_b_het = _slope(_x, 1 + 2 * _x + _eh)
_xo = _x + _rng.normal(size=_n)                        # 測定誤差（分散1、真の分散1）
_b_me = _slope(_xo, 1 + 2 * _x + _rng.normal(size=_n))
_z = 0.8 * _x + _rng.normal(size=_n) * 0.6             # 欠落変数
_b_ov = _slope(_x, 1 + 2 * _x + 1.5 * _z + _rng.normal(size=_n))
q.check("q19-endogeneity-3", abs(_b_het - 2) < 0.02 and abs(_b_me - 1) < 0.02 and abs(_b_ov - 3.2) < 0.02,
        f"不均一分散のみ：傾き {_b_het:.3f}≈2（一致）。測定誤差 {_b_me:.3f}≈1、欠落変数 {_b_ov:.3f}≈3.2（偏る）")

q.done()
