"""第19章 検算スクリプト（回帰分析の発展：WLS・GLS・正則化・内生性・非線形・分位点・変数選択）。
数値例は numpy/scipy で独立に再計算し、記号的な主張は sympy で確認する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。丸めた途中値を使う式は許容を明記して広げる。
charts の params は各 md の front matter を yaml で読み、本文の値と照合する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（概ね標準誤差の4倍以上）。
"""
from math import sqrt, log, exp
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from scipy import stats, optimize, integrate

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    # 丁度 5 の丸め境界（0.03125→0.0312 など）を浮動小数点で落とさないよう 1e-12 の余裕
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def r4(v, k=4):
    return [round(float(x), k) for x in np.ravel(v)]


def front(term):
    txt = (ROOT / "content" / "ch19" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def chart(term, cid, k=0):
    return [c for c in charts(term) if c.get("id") == cid][k]


def series(c, i):
    return c["params"]["series"][i]["y"]


def ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


# =====================================================================
# 加重最小二乗法
# =====================================================================
T = "weighted-least-squares"
x = np.arange(1, 7, dtype=float)
y = np.array([3.1, 4.8, 7.6, 8.2, 12.9, 12.4])
w = 1 / x
X = np.column_stack([np.ones(6), x])
W = np.diag(w)
XtWX = X.T @ W @ X
XtWy = X.T @ W @ y
check(T, "例1 X'WX=[[2.45,6],[6,21]]", allclose(XtWX, [[2.45, 6], [6, 21]]), r4(XtWX))
check(T, "例1 X'Wy=(14.73,49)", allclose(XtWy, [14.73, 49]), r4(XtWy))
det = np.linalg.det(XtWX)
check(T, "例1 行列式 15.45", close(det, 15.45), det)
check(T, "例1 分子 21×14.73−6×49=15.33, 2.45×49−6×14.73=31.67",
      close(21 * 14.73 - 6 * 49, 15.33) and close(2.45 * 49 - 6 * 14.73, 31.67), (21 * 14.73 - 6 * 49, 2.45 * 49 - 6 * 14.73))
bW = np.linalg.solve(XtWX, XtWy)
check(T, "例1 WLS (0.9922, 2.0498)", allclose(bW, [0.9922, 2.0498]), r4(bW))
bO = ols(X, y)
check(T, "例1 OLS (1.0267, 2.04)", allclose(bO, [1.0267, 2.04]), r4(bO))
# 変換して OLS
Ws = np.diag(np.sqrt(w))
check(T, "W^{1/2} 変換後の OLS = WLS", allclose(ols(Ws @ X, Ws @ y), bW, 1e-10), r4(ols(Ws @ X, Ws @ y)))
# 重み付き平均による単回帰の式
xw = np.sum(w * x) / w.sum(); yw = np.sum(w * y) / w.sum()
b1w = np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw) ** 2)
check(T, "単回帰の重み付き平均表示が行列解と一致", close(b1w, bW[1], 1e-10) and close(yw - b1w * xw, bW[0], 1e-10), (b1w, yw - b1w * xw))
# 重みの定数倍で不変
bW7 = np.linalg.solve(X.T @ (7 * W) @ X, X.T @ (7 * W) @ y)
check(T, "重みを定数倍しても推定値は不変", allclose(bW7, bW, 1e-10), r4(bW7))
# 分散の比較
A = np.linalg.solve(X.T @ X, X.T)
VO = A @ np.diag(x) @ A.T
VW = np.linalg.inv(XtWX)
check(T, "OLS 傾きのサンドイッチ分散 0.2000σ²", close(VO[1, 1], 0.2), VO[1, 1])
check(T, "WLS 傾きの分散 2.45/15.45=0.1586σ²", close(VW[1, 1], 0.1586) and close(2.45 / 15.45, 0.1586), VW[1, 1])
check(T, "WLS は約21%小さい", abs((1 - VW[1, 1] / VO[1, 1]) - 0.21) < 0.005, 1 - VW[1, 1] / VO[1, 1])
check(T, "V_OLS − V_WLS は非負定値（BLUE）", np.linalg.eigvalsh(VO - VW).min() > -1e-12, np.linalg.eigvalsh(VO - VW))
# sympy: 勾配
b0s, b1s = sp.symbols("b0 b1")
xs_ = [sp.Rational(i) for i in range(1, 7)]
ys_ = [sp.Rational(str(v)) for v in y]
SW = sum((1 / xi) * (yi - b0s - b1s * xi) ** 2 for xi, yi in zip(xs_, ys_))
sol = sp.solve([sp.diff(SW, b0s), sp.diff(SW, b1s)], [b0s, b1s])
check(T, "sympy: ∂S_W/∂β=0 の解が (15.33/15.45, 31.67/15.45)",
      sp.simplify(sol[b0s] - sp.Rational(1533, 1545)) == 0 and sp.simplify(sol[b1s] - sp.Rational(3167, 1545)) == 0,
      (sol[b0s], sol[b1s]))
# 例2
xg = np.array([1., 2, 3]); ng = np.array([5., 10, 20]); yg = np.array([2.0, 3.4, 4.1])
Xg = np.column_stack([np.ones(3), xg])
M = Xg.T @ np.diag(ng) @ Xg; v = Xg.T @ np.diag(ng) @ yg
check(T, "例2 X'WX=[[35,85],[85,225]], X'Wy=(126,324)", allclose(M, [[35, 85], [85, 225]]) and allclose(v, [126, 324]), (r4(M), r4(v)))
bg = np.linalg.solve(M, v)
check(T, "例2 (1.2462, 0.9692)", allclose(bg, [1.2462, 0.9692]), r4(bg))
# 個票：群内のばらつきを任意に付けて OLS
xi = np.repeat(xg, ng.astype(int))
dev = rng.normal(size=xi.size)
for g in xg:
    m = xi == g
    dev[m] -= dev[m].mean()
yi = np.repeat(yg, ng.astype(int)) + dev
check(T, "例2 個票35人の OLS と一致", allclose(ols(np.column_stack([np.ones(35), xi]), yi), bg, 1e-10), r4(ols(np.column_stack([np.ones(35), xi]), yi)))
check(T, "例2 重みなしの傾き 1.05", close(ols(Xg, yg)[1], 1.05), ols(Xg, yg)[1])
# charts
c = chart(T, "line-data")
check(T, "chart: x と観測値", c["params"]["x"] == [1, 2, 3, 4, 5, 6] and allclose(series(c, 0), y), series(c, 0))
check(T, "chart: OLS の直線", allclose(series(c, 1), X @ bO), r4(X @ bO))
check(T, "chart: WLS の直線（厳密な係数で4桁）", allclose(series(c, 2), X @ bW), r4(X @ bW))
c = chart(T, "residual-pattern")
check(T, "chart: residual-pattern params（funnel, 0≤strength≤3）", c["params"]["pattern"] == "funnel" and 0 <= c["params"]["strength"] <= 3, c["params"])
# シミュレーション：E, V
R = 200000
eps = rng.normal(size=(R, 6)) * np.sqrt(x)
beta = np.array([1.0, 2.0])
Y = X @ beta + eps
BW = (np.linalg.solve(XtWX, X.T @ W @ Y.T)).T
BO = (A @ Y.T).T
check(T, "sim: E[β̂_W]=β（±4SE）", np.all(np.abs(BW.mean(0) - beta) < 4 * BW.std(0) / sqrt(R)), r4(BW.mean(0)))
check(T, "sim: V[β̂_W] の傾き = 0.1586（±2%）", abs(BW[:, 1].var() / VW[1, 1] - 1) < 0.02, BW[:, 1].var())
check(T, "sim: V[β̂_OLS] の傾き = 0.2000（±2%）", abs(BO[:, 1].var() / 0.2 - 1) < 0.02, BO[:, 1].var())
sW2 = np.sum(((Y - BW @ X.T) ** 2) * w, axis=1) / 4
check(T, "sim: E[s_W²]=σ²=1（±0.01）", abs(sW2.mean() - 1) < 0.01, sW2.mean())

# =====================================================================
# 一般化最小二乗法
# =====================================================================
T = "generalized-least-squares"
rho = 0.6; n = 5
Om = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
Oi = np.linalg.inv(Om)
Q = (1 - rho ** 2) * Oi
check(T, "(1−ρ²)Ω⁻¹ の対角 (1,1.36,1.36,1.36,1)", allclose(np.diag(Q), [1, 1.36, 1.36, 1.36, 1], 1e-10), r4(np.diag(Q)))
off = Q - np.diag(np.diag(Q))
check(T, "(1−ρ²)Ω⁻¹ は三重対角で隣は −0.6", allclose(np.diag(Q, 1), [-0.6] * 4, 1e-10) and np.abs(off).sum() - np.abs(np.diag(Q, 1)).sum() * 2 < 1e-10, r4(np.diag(Q, 1)))
one = np.ones(n)
l = one @ Oi
check(T, "1'Ω⁻¹=(0.625,0.25,0.25,0.25,0.625)、合計2", allclose(l, [0.625, .25, .25, .25, .625], 1e-10) and close(l.sum(), 2, 1e-10), r4(l))
wg = l / l.sum()
check(T, "GLS の重み (0.3125,0.125,...)", allclose(wg, [0.3125, .125, .125, .125, .3125], 1e-10), r4(wg))
yv = np.array([10.2, 11.0, 10.6, 9.4, 9.8])
mu = wg @ yv
check(T, "μ̂_G=10.125（6.25+3.875）", close(mu, 10.125, 1e-10) and close(0.3125 * 20, 6.25) and close(0.125 * 31, 3.875), mu)
check(T, "単純平均 10.2", close(yv.mean(), 10.2, 1e-10), yv.mean())
check(T, "V[μ̂_G]=σ²/2", close(1 / l.sum(), 0.5, 1e-12), 1 / l.sum())
check(T, "V[単純平均]=1'Ω1/25=0.5233σ²", close(one @ Om @ one / 25, 0.5233), one @ Om @ one / 25)
# プレイス・ウィンステン
P = np.zeros((n, n)); P[0, 0] = sqrt(1 - rho ** 2)
for t in range(1, n):
    P[t, t] = 1; P[t, t - 1] = -rho
check(T, "P'P=(1−ρ²)Ω⁻¹（P は PW 変換、σ_u² 尺度）", allclose(P.T @ P, Q, 1e-10), "ok")
check(T, "PΩP'=(1−ρ²)I", allclose(P @ Om @ P.T, (1 - rho ** 2) * np.eye(n), 1e-10), "ok")
ys = P @ yv; xs = P @ one
check(T, "y*=(8.16,4.88,4.00,3.04,4.16)", allclose(ys, [8.16, 4.88, 4.0, 3.04, 4.16], 1e-10), r4(ys))
check(T, "x*=(0.8,0.4,0.4,0.4,0.4)", allclose(xs, [0.8, .4, .4, .4, .4], 1e-10), r4(xs))
check(T, "原点を通る OLS Σx*y*/Σx*²=10.125", close(xs @ ys / (xs @ xs), 10.125, 1e-10), xs @ ys / (xs @ xs))
# 一般の Ω で コレスキーによる白色化
Xr = np.column_stack([np.ones(8), rng.normal(size=8), rng.normal(size=8)])
Lr = rng.normal(size=(8, 8)); Omr = Lr @ Lr.T + 8 * np.eye(8)
yr = rng.normal(size=8)
Pr = np.linalg.inv(np.linalg.cholesky(Omr))  # Pr Ω Pr' = I
bG = np.linalg.solve(Xr.T @ np.linalg.solve(Omr, Xr), Xr.T @ np.linalg.solve(Omr, yr))
check(T, "白色化して OLS = (X'Ω⁻¹X)⁻¹X'Ω⁻¹y（ランダム例）", allclose(ols(Pr @ Xr, Pr @ yr), bG, 1e-9) and allclose(Pr.T @ Pr, np.linalg.inv(Omr), 1e-9), r4(bG))
Ar = np.linalg.solve(Xr.T @ Xr, Xr.T)
VOr = Ar @ Omr @ Ar.T; VGr = np.linalg.inv(Xr.T @ np.linalg.solve(Omr, Xr))
check(T, "V_OLS − V_GLS は非負定値（ランダム例）", np.linalg.eigvalsh(VOr - VGr).min() > -1e-10, np.linalg.eigvalsh(VOr - VGr).min())
# 例2 等相関
Oeq = 0.7 * np.eye(5) + 0.3
le = one @ np.linalg.inv(Oeq)
check(T, "例2 等相関 r=0.3: 1'Ω⁻¹ の成分は全て 0.4545", allclose(le, [0.4545] * 5), r4(le))
check(T, "例2 GLS 平均＝単純平均", close((le / le.sum()) @ yv, yv.mean(), 1e-10), (le / le.sum()) @ yv)
# X の列が Ω の固有空間に入れば GLS=OLS
ev, U = np.linalg.eigh(Omr)
Xe = U[:, [0, 3]]
check(T, "X の列が Ω の固有ベクトルなら GLS=OLS",
      allclose(np.linalg.solve(Xe.T @ np.linalg.solve(Omr, Xe), Xe.T @ np.linalg.solve(Omr, yr)), ols(Xe, yr), 1e-9), "ok")
c = chart(T, "bar-data")
check(T, "chart: bar-data の重み", allclose(c["params"]["values"], wg, 1e-10), c["params"]["values"])
c = chart(T, "dw-sim")
check(T, "chart: dw-sim φ=0.6 は例1の ρ と一致、n=30", c["params"]["phi"] == 0.6 and c["params"]["n"] == 30, c["params"])
# シミュレーション
R = 200000
E = rng.normal(size=(R, n)) @ np.linalg.cholesky(Om).T
Yt = 10 + E
mG = Yt @ wg; mO = Yt.mean(1)
check(T, "sim: V[μ̂_G]=0.5（±1.5%）", abs(mG.var() / 0.5 - 1) < 0.015, mG.var())
check(T, "sim: V[単純平均]=0.5233（±1.5%）", abs(mO.var() / 0.5233 - 1) < 0.015, mO.var())
check(T, "sim: 両者とも不偏（±4SE）", abs(mG.mean() - 10) < 4 * sqrt(0.5 / R) and abs(mO.mean() - 10) < 4 * sqrt(0.53 / R), (mG.mean(), mO.mean()))

# =====================================================================
# リッジ回帰
# =====================================================================
T = "ridge-regression"
x1 = np.array([-2., -1, 0, 1, 2]); x2 = np.array([-2., 0, -1, 1, 2]); yv = np.array([-3.9, -1.2, -0.6, 1.8, 3.9])
X = np.column_stack([x1, x2])
G = X.T @ X; g = X.T @ yv
check(T, "例1 X'X=[[10,9],[9,10]], X'y=(18.6,18.0)", allclose(G, [[10, 9], [9, 10]]) and allclose(g, [18.6, 18.0], 1e-10), (r4(G), r4(g)))
check(T, "例1 中心化されている", abs(x1.sum()) + abs(x2.sum()) + abs(yv.sum()) < 1e-12, (x1.sum(), x2.sum(), yv.sum()))
check(T, "例1 相関 0.9", close(np.corrcoef(x1, x2)[0, 1], 0.9, 1e-12), np.corrcoef(x1, x2)[0, 1])
bO = np.linalg.solve(G, g)
check(T, "例1 OLS (1.2632, 0.6632)", allclose(bO, [1.2632, 0.6632]), r4(bO))
ridge = lambda lam: np.linalg.solve(G + lam * np.eye(2), g)
check(T, "例1 λ=1: det=40, 分子 (42.6,30.6)", close(np.linalg.det(G + np.eye(2)), 40) and allclose(np.array([[11, -9], [-9, 11]]) @ g, [42.6, 30.6], 1e-10), "ok")
check(T, "例1 λ=1: (1.065, 0.765)", allclose(ridge(1), [1.065, 0.765], 1e-10), r4(ridge(1)))
check(T, "例1 ノルム 1.4267 → 1.3113", close(np.linalg.norm(bO), 1.4267) and close(np.linalg.norm(ridge(1)), 1.3113), (np.linalg.norm(bO), np.linalg.norm(ridge(1))))
check(T, "例1 λ=5: (0.8125, 0.7125)", allclose(ridge(5), [0.8125, 0.7125], 1e-10), r4(ridge(5)))
# 目的関数の最小点
f = lambda b, lam: np.sum((yv - X @ b) ** 2) + lam * b @ b
res = optimize.minimize(f, np.zeros(2), args=(1,), tol=1e-12)
check(T, "数値最適化でも λ=1 の解に一致", allclose(res.x, ridge(1), 1e-5), r4(res.x))
# 例2
d = np.linalg.eigvalsh(G)
check(T, "例2 固有値 1, 19", allclose(d, [1, 19], 1e-10), d)
beta = np.array([1, 0.8])


def ridge_mse(lam, sig2=1.0):
    Ainv = np.linalg.inv(G + lam * np.eye(2))
    bias = -lam * Ainv @ beta
    V = sig2 * Ainv @ G @ Ainv
    return bias @ bias, np.trace(V), bias @ bias + np.trace(V)


check(T, "例2 OLS 分散の和 1/19+1=1.0526", close(ridge_mse(0)[1], 1.0526) and close(1 / 19 + 1, 1.0526), ridge_mse(0))
b2, v2, m2 = ridge_mse(4)
check(T, "例2 λ=4: バイアス² 0.0618, 分散 0.0759, MSE 0.1377", close(b2, 0.0618) and close(v2, 0.0759) and close(m2, 0.1377), (b2, v2, m2))
check(T, "例2 MSE は OLS の約13%", abs(m2 / ridge_mse(0)[2] - 0.13) < 0.005, m2 / ridge_mse(0)[2])
opt = optimize.minimize_scalar(lambda L: ridge_mse(L)[2], bounds=(0.01, 20), method="bounded", options={"xatol": 1e-10})
check(T, "例2 最適 λ≈3.81、MSE 0.1375", abs(opt.x - 3.81) < 0.005 and close(opt.fun, 0.1375), (opt.x, opt.fun))
c = chart(T, "line-data")
lams = c["params"]["x"]
tab = np.array([ridge_mse(L) for L in lams])
check(T, "chart: バイアスの2乗和（4桁）", allclose(series(c, 0), tab[:, 0]), r4(tab[:, 0]))
check(T, "chart: 分散の和（4桁）", allclose(series(c, 1), tab[:, 1]), r4(tab[:, 1]))
check(T, "chart: MSE（4桁）", allclose(series(c, 2), tab[:, 2]), r4(tab[:, 2]))
check(T, "chart: キャプション λ≈3.8", abs(opt.x - 3.8) < 0.05, opt.x)
check(T, "chart: ridge-lasso-path method=ridge", chart(T, "ridge-lasso-path")["params"]["method"] == "ridge", chart(T, "ridge-lasso-path")["params"])
# sympy: 期待値・バイアス・分散の式（一般の 2x2 対称）
a11, a12, a22, lam, B1, B2, s2 = sp.symbols("a11 a12 a22 lambda beta1 beta2 sigma2", real=True)
Gs = sp.Matrix([[a11, a12], [a12, a22]]); Bs = sp.Matrix([B1, B2]); I2 = sp.eye(2)
Ainv = (Gs + lam * I2).inv()
check(T, "sympy: E=(X'X+λI)⁻¹X'Xβ = β−λ(X'X+λI)⁻¹β",
      sp.simplify(Ainv * Gs * Bs - (Bs - lam * Ainv * Bs)) == sp.zeros(2, 1), "ok")
bv = sp.Matrix(sp.symbols("b1 b2"))
yX = sp.Matrix(sp.symbols("g1 g2"))
obj = (bv.T * Gs * bv)[0] - 2 * (bv.T * yX)[0] + lam * (bv.T * bv)[0]
grad = sp.Matrix([sp.diff(obj, t) for t in bv])
check(T, "sympy: 勾配 −2X'y+2X'Xβ+2λβ", sp.simplify(grad - (-2 * yX + 2 * Gs * bv + 2 * lam * bv)) == sp.zeros(2, 1), "ok")
check(T, "sympy: 直交計画 X'X=I で b/(1+λ)", sp.simplify(((I2 + lam * I2).inv() * yX) - yX / (1 + lam)) == sp.zeros(2, 1), "ok")
# 固有方向の縮小
Uq, dq = np.linalg.eigh(G)[1], np.linalg.eigh(G)[0]
shr = Uq.T @ ridge(4) / (Uq.T @ bO)
check(T, "固有方向ごとに d/(d+λ) 倍（λ=4）", allclose(shr, dq / (dq + 4), 1e-10), r4(shr))
# ベイズ：事後平均 = リッジ
sig2v, lamv = 1.3, 2.0
postmean = np.linalg.solve(G / sig2v + np.eye(2) * lamv / sig2v, g / sig2v)
check(T, "β~N(0,(σ²/λ)I) の事後平均 = リッジ", allclose(postmean, ridge(lamv), 1e-10), r4(postmean))
# シミュレーション
R = 200000
Ysim = (X @ beta)[None, :] + rng.normal(size=(R, 5))
Bsim = np.linalg.solve(G + 4 * np.eye(2), X.T @ Ysim.T).T
Eth = np.linalg.solve(G + 4 * np.eye(2), G @ beta)
check(T, "sim: λ=4 の E[β̂_R]（±4SE）", np.all(np.abs(Bsim.mean(0) - Eth) < 4 * Bsim.std(0) / sqrt(R)), (r4(Bsim.mean(0)), r4(Eth)))
msesim = np.mean(np.sum((Bsim - beta) ** 2, 1))
check(T, "sim: λ=4 の MSE 0.1377（±1.5%）", abs(msesim / 0.1377 - 1) < 0.015, msesim)
BOs = np.linalg.solve(G, X.T @ Ysim.T).T
check(T, "sim: OLS の MSE 1.0526（±1.5%）", abs(np.mean(np.sum((BOs - beta) ** 2, 1)) / 1.0526 - 1) < 0.015, np.mean(np.sum((BOs - beta) ** 2, 1)))

# =====================================================================
# Lasso
# =====================================================================
T = "lasso"
soft = lambda b, L: np.sign(b) * np.maximum(np.abs(b) - L, 0)
Xl = 0.5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
yl = np.array([3.0, 0.6, -0.4, -2.2])
check(T, "例1 X'X=I、各列は定数列と直交", allclose(Xl.T @ Xl, np.eye(3), 1e-12) and allclose(Xl.sum(0), [0, 0, 0], 1e-12), "ok")
check(T, "例1 ȳ=0.25", close(yl.mean(), 0.25, 1e-12), yl.mean())
bl = Xl.T @ yl
check(T, "例1 b=X'y=(3.1,2.1,0.3)", allclose(bl, [3.1, 2.1, 0.3], 1e-12), r4(bl))
check(T, "例1 λ=1 Lasso (2.1,1.1,0)", allclose(soft(bl, 1), [2.1, 1.1, 0], 1e-12), r4(soft(bl, 1)))
check(T, "例1 λ=1 リッジ b/2=(1.55,1.05,0.15)", allclose(np.linalg.solve(Xl.T @ Xl + np.eye(3), Xl.T @ yl), [1.55, 1.05, 0.15], 1e-12), "ok")
check(T, "例1 λ=2 Lasso (1.1,0.1,0)", allclose(soft(bl, 2), [1.1, 0.1, 0], 1e-12), r4(soft(bl, 2)))


def cd_lasso(X, y, L, iters=20000):
    b = np.zeros(X.shape[1]); nrm = (X ** 2).sum(0)
    for _ in range(iters):
        bold = b.copy()
        for j in range(X.shape[1]):
            r = y - X @ b + X[:, j] * b[j]
            b[j] = soft(X[:, j] @ r, L) / nrm[j]
        if np.max(np.abs(b - bold)) < 1e-13:
            break
    return b


ylc = yl - yl.mean()
check(T, "例1 座標降下法 λ=1 でも同じ", allclose(cd_lasso(Xl, ylc, 1), [2.1, 1.1, 0], 1e-9), r4(cd_lasso(Xl, ylc, 1)))
check(T, "例1 座標降下法 λ=2 でも同じ", allclose(cd_lasso(Xl, ylc, 2), [1.1, 0.1, 0], 1e-9), r4(cd_lasso(Xl, ylc, 2)))
# 1変数問題を数値で
bb = np.linspace(-3, 3, 25)
grid = np.linspace(-5, 5, 200001)
num = [grid[np.argmin(0.5 * (grid - b0) ** 2 + 1.0 * np.abs(grid))] for b0 in bb]
check(T, "g(β)=½(β−b)²+λ|β| の数値最小点 = ソフト閾値", allclose(num, soft(bb, 1), 1e-4), "25点")
# sympy: 直交計画の分解
bsym = sp.Matrix(sp.symbols("bt1:4")); yy = sp.Matrix(sp.symbols("y1:5"))
Xs = sp.Matrix(Xl.tolist()).applyfunc(sp.nsimplify)
lhs = sp.Rational(1, 2) * ((yy - Xs * bsym).T * (yy - Xs * bsym))[0]
bv_ = Xs.T * yy
rhs = sp.Rational(1, 2) * (yy.T * yy)[0] - (bsym.T * bv_)[0] + sp.Rational(1, 2) * (bsym.T * bsym)[0]
check(T, "sympy: ½‖y−Xβ‖² = ½‖y‖² − β'b + ½‖β‖²（X'X=I）", sp.expand(lhs - rhs) == 0, "ok")
const = sp.Rational(1, 2) * (yy.T * yy)[0] - sp.Rational(1, 2) * (bv_.T * bv_)[0]
check(T, "sympy: = 定数 + Σ½(β_j−b_j)²", sp.expand(lhs - const - sum(sp.Rational(1, 2) * (bsym[j] - bv_[j]) ** 2 for j in range(3))) == 0, "ok")
# 例2
L2 = lambda L: cd_lasso(X, yv, L)
Gi1 = np.linalg.solve(G, np.ones(2))
check(T, "例2 (X'X)⁻¹(1,1)'=(1/19,1/19)", allclose(Gi1, [1 / 19, 1 / 19], 1e-12), r4(Gi1))
check(T, "例2 β̂2=0 になる λ=0.6632×19=12.6", close(bO[1] * 19, 12.6, 1e-10) and close(0.6632 * 19, 12.6, 0.005), bO[1] * 19)
check(T, "例2 λ=10: (0.7368, 0.1368)", allclose(L2(10), [0.7368, 0.1368]) and allclose(bO - 10 / 19, [0.7368, 0.1368]), r4(L2(10)))
check(T, "例2 λ=12.6 以降 β̂1=(18.6−λ)/10（λ=15）", allclose(L2(15), [(18.6 - 15) / 10, 0], 1e-9), r4(L2(15)))
check(T, "例2 λ=12.6 直前は両方正、直後は β2=0", L2(12.5)[1] > 0 and abs(L2(12.7)[1]) < 1e-12, (r4(L2(12.5)), r4(L2(12.7))))
check(T, "例2 λ=18.6 ですべて0、直前は β1>0", allclose(L2(18.6), [0, 0], 1e-12) and L2(18.5)[0] > 0, (r4(L2(18.6)), r4(L2(18.5))))
c = chart(T, "line-data")
bx = np.array(c["params"]["x"], float)
check(T, "chart: OLS（恒等）", allclose(series(c, 0), bx, 1e-12), "ok")
check(T, "chart: ソフト閾値 λ=1", allclose(series(c, 1), soft(bx, 1), 1e-12), series(c, 1))
check(T, "chart: リッジ λ=1 は b/2", allclose(series(c, 2), bx / 2, 1e-12), series(c, 2))
check(T, "chart: ridge-lasso-path method=lasso", chart(T, "ridge-lasso-path")["params"]["method"] == "lasso", "ok")
# ラプラス事前分布の MAP（y=2.3, σ²=1.5, b=0.5 → λ=3 で 0）
check(T, "ラプラス事前分布の MAP = Lasso（λ=σ²/b）: 数値例",
      close(optimize.minimize_scalar(lambda t: (t - 4.3) ** 2 / (2 * 1.5) + abs(t) / 0.5, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x,
            soft(np.array([4.3]), 1.5 / 0.5)[0], 1e-5),
      soft(np.array([4.3]), 3.0)[0])

# =====================================================================
# 罰則付き推定量のバイアス
# =====================================================================
T = "bias-in-penalized-estimation"
L_, b_, s_ = sp.symbols("lambda beta sigma2", positive=True)
mse = (b_ / (1 + L_) - b_) ** 2 + s_ / (1 + L_) ** 2
check(T, "sympy: MSE_j=(σ²+λ²β²)/(1+λ)²", sp.simplify(mse - (s_ + L_ ** 2 * b_ ** 2) / (1 + L_) ** 2) == 0, "ok")
dm = sp.diff(mse, L_)
check(T, "sympy: dMSE/dλ=2(λβ²−σ²)/(1+λ)³", sp.simplify(dm - 2 * (L_ * b_ ** 2 - s_) / (1 + L_) ** 3) == 0, sp.factor(dm))
check(T, "sympy: 中間の分子 2λβ²(1+λ)−2(σ²+λ²β²) = 2(λβ²−σ²)", sp.expand(2 * L_ * b_ ** 2 * (1 + L_) - 2 * (s_ + L_ ** 2 * b_ ** 2) - 2 * (L_ * b_ ** 2 - s_)) == 0, "ok")
check(T, "sympy: λ*=σ²/β², 最小値 σ²β²/(σ²+β²)", sp.solve(dm, L_) == [s_ / b_ ** 2] and sp.simplify(mse.subs(L_, s_ / b_ ** 2) - s_ * b_ ** 2 / (s_ + b_ ** 2)) == 0, sp.solve(dm, L_))
check(T, "sympy: λ=0 での傾き −2σ²", sp.simplify(dm.subs(L_, 0) + 2 * s_) == 0, dm.subs(L_, 0))
check(T, "sympy: バイアス −λβ/(1+λ)、分散 σ²/(1+λ)²", sp.simplify(b_ / (1 + L_) - b_ + L_ * b_ / (1 + L_)) == 0, "ok")
# 例1 表
tab = {0: (0, 0.5, 0.5), 0.25: (0.04, 0.32, 0.36), 0.5: (0.1111, 0.2222, 0.3333), 1: (0.25, 0.125, 0.375), 2: (0.4444, 0.0556, 0.5)}
for Lv, (bb2, vv, mm) in tab.items():
    B2v = (Lv / (1 + Lv)) ** 2; Vv = 0.5 / (1 + Lv) ** 2
    check(T, f"例1 表 λ={Lv}", close(B2v, bb2) and close(Vv, vv) and close(B2v + Vv, mm), r4([B2v, Vv, B2v + Vv]))
check(T, "例1 λ*=0.5, 最小 0.3333、OLS より1/3小さい", close(0.5 * 1 / 1.5, 0.3333) and close(1 - (1 / 3) / 0.5, 1 / 3, 1e-12), 0.5 / 1.5)
mseo = lambda Lv: (0.5 + Lv ** 2) / (1 + Lv) ** 2
check(T, "例1 λ=2 で OLS と同じ、それ以上で悪化", close(mseo(2), 0.5, 1e-12) and mseo(2.5) > 0.5 and mseo(1.9) < 0.5, (mseo(2), mseo(2.5)))
c = chart(T, "line-data")
Ls = np.array(c["params"]["x"], float)
Bc = (Ls / (1 + Ls)) ** 2; Vc = 0.5 / (1 + Ls) ** 2
check(T, "chart: バイアスの2乗（4桁）", allclose(series(c, 0), Bc), r4(Bc))
check(T, "chart: 分散（4桁）", allclose(series(c, 1), Vc), r4(Vc))
check(T, "chart: MSE（4桁）", allclose(series(c, 2), Bc + Vc), r4(Bc + Vc))


# 例2 Lasso の偏り
def lasso_moments(beta, lam=1.0):
    f = lambda b: soft(np.array([b]), lam)[0] * stats.norm.pdf(b, beta)
    f2 = lambda b: (soft(np.array([b]), lam)[0] - beta) ** 2 * stats.norm.pdf(b, beta)
    pts = [-lam, lam]
    E = sum(integrate.quad(f, a, bnd, limit=200)[0] for a, bnd in [(-np.inf, -lam), (-lam, lam), (lam, np.inf)])
    M = sum(integrate.quad(f2, a, bnd, limit=200)[0] for a, bnd in [(-np.inf, -lam), (-lam, lam), (lam, np.inf)])
    return E - beta, M


claims = {0.5: (-0.3315, 0.314), 1: (-0.6095, 0.7249), 2: (-0.9171, 1.5931), 3: (-0.9915, 1.9433), 5: (-1.0000, 1.9999)}
for bt, (cb, cm) in claims.items():
    bb_v, mm_v = lasso_moments(bt)
    tolm = 5e-4 if bt == 0.5 else 5e-5
    check(T, f"例2 Lasso β={bt}: バイアス {cb}, MSE {cm}", close(bb_v, cb) and close(mm_v, cm, tolm), (round(bb_v, 5), round(mm_v, 5)))
check(T, "例2 β=0 の MSE 0.1507", close(lasso_moments(0)[1], 0.1507), lasso_moments(0)[1])
bs = rng.normal(2, 1, 1_000_000)
ls = soft(bs, 1)
check(T, "sim: β=2 のバイアス −0.9171（±0.004）", abs(ls.mean() - 2 + 0.9171) < 0.004, ls.mean() - 2)
# ホーエル・ケナード（一般の X、例として ridge の例2）
check(T, "ホーエル・ケナード：例の X で MSE(λ)<MSE(0) となる λ>0 が存在（λ=0.01）", ridge_mse(0.01)[2] < ridge_mse(0)[2], ridge_mse(0.01)[2])

# =====================================================================
# 正則化
# =====================================================================
T = "regularization"
m_, n_, yb_, lm_ = sp.symbols("mu n ybar lambda", positive=True)
ysym = sp.symbols("y1:11")
objm = sum((yi - m_) ** 2 for yi in ysym) + lm_ * m_ ** 2
solm = sp.solve(sp.diff(objm, m_), m_)[0]
check(T, "sympy: μ̂=nȳ/(n+λ)（n=10）", sp.simplify(solm - sum(ysym) / (10 + lm_)) == 0, solm)
check(T, "例1 μ̂=24/12=2.0", close(10 * 2.4 / 12, 2.0, 1e-12), 10 * 2.4 / 12)
post = (10 * 2.4 / 1) / (10 / 1 + 1 / 0.5)
check(T, "例1 σ²=1, τ²=0.5 の事後平均 2.0（λ=σ²/τ²=2）", close(post, 2.0, 1e-12), post)
# MAP：正規事前で λ=σ²/τ²
bq, sg, tq = sp.symbols("beta sigma tau", positive=True)
yq = sp.Symbol("y")
lp = -(yq - bq) ** 2 / (2 * sg ** 2) - bq ** 2 / (2 * tq ** 2)
pen = sp.expand(-2 * sg ** 2 * lp)
check(T, "sympy: 正規事前の −2σ²log事後 = (y−β)² + (σ²/τ²)β²", sp.simplify(pen - ((yq - bq) ** 2 + sg ** 2 / tq ** 2 * bq ** 2)) == 0, "ok")
lpl = -(yq - bq) ** 2 / (2 * sg ** 2) - bq / sp.Symbol("b", positive=True)
pen2 = sp.expand(-sg ** 2 * lpl)
check(T, "sympy: ラプラス事前の −σ²log事後 = ½(y−β)² + (σ²/b)|β|", sp.simplify(pen2 - ((yq - bq) ** 2 / 2 + sg ** 2 / sp.Symbol("b", positive=True) * bq)) == 0, "ok")
# 例2 6次多項式リッジ
xp = np.linspace(-1, 1, 9)
yp = np.array([0.10, -0.85, -1.05, -0.55, 0.08, 0.62, 1.12, 0.70, -0.05])


def poly_ridge(xt, yt, lam):
    F = np.column_stack([xt ** k for k in range(1, 7)])
    mF = F.mean(0); sF = F.std(0)  # 標準化は母標準偏差（ddof=0）
    Z = (F - mF) / sF
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(6), Z.T @ (yt - yt.mean()))
    pred = lambda xn: yt.mean() + ((np.column_stack([xn ** k for k in range(1, 7)]) - mF) / sF) @ b
    return b, pred


rtab = {0: (2.642, 0.0029, 4.1887), 0.01: (2.184, 0.0039, 0.6814), 0.1: (1.570, 0.0124, 0.1295), 0.3: (1.346, 0.0257, 0.1160),
        1: (0.956, 0.0844, 0.5907), 3: (0.532, 0.2041, 0.8887), 10: (0.222, 0.3336, 0.7485)}
cvs = {}; trs = {}
for lv, (cn, ct, cc) in rtab.items():
    b, pr = poly_ridge(xp, yp, lv)
    tr = np.mean((yp - pr(xp)) ** 2)
    cv = np.mean([(yp[i] - poly_ridge(np.delete(xp, i), np.delete(yp, i), lv)[1](xp[i:i + 1])[0]) ** 2 for i in range(9)])
    cvs[lv] = cv; trs[lv] = tr
    check(T, f"例2 λ={lv}: ノルム {cn}, 訓練 {ct}, LOOCV {cc}", close(np.linalg.norm(b), cn, 5e-4) and close(tr, ct) and close(cv, cc), (round(np.linalg.norm(b), 4), round(tr, 5), round(cv, 5)))
check(T, "例2 LOOCV 最小は λ=0.3", min(cvs, key=cvs.get) == 0.3, min(cvs, key=cvs.get))
check(T, "例2 訓練誤差は λ について単調増加", all(np.diff([trs[k] for k in sorted(trs)]) > 0), "ok")
c = chart(T, "line-data")
lx = c["params"]["x"]
lv_ = [0.01, 0.1, 0.3, 1, 3, 10]
check(T, "chart: x=log10 λ（0.01..10）", allclose(lx, np.log10(lv_)), r4(np.log10(lv_)))
check(T, "chart: 訓練誤差", allclose(series(c, 0), [trs[k] for k in lv_]), r4([trs[k] for k in lv_]))
check(T, "chart: LOOCV", allclose(series(c, 1), [cvs[k] for k in lv_]), r4([cvs[k] for k in lv_]))
c = chart(T, "bias-variance")
check(T, "chart: bias-variance params（degree 9 は範囲 1..12 内）", 1 <= c["params"]["degree"] <= 12 and c["params"]["n"] == 20, c["params"])

# =====================================================================
# 内生性
# =====================================================================
T = "endogeneity"
b1, sx, sw = sp.symbols("beta1 sigma_x sigma_w", positive=True)
plim = b1 - b1 * sw ** 2 / (sx ** 2 + sw ** 2)
check(T, "sympy: β1−β1σw²/(σx²+σw²)=β1σx²/(σx²+σw²)", sp.simplify(plim - b1 * sx ** 2 / (sx ** 2 + sw ** 2)) == 0, "ok")
check(T, "例1 plim=2×4/5=1.6", close(2 * 4 / 5, 1.6, 1e-12), 1.6)
c = chart(T, "line-data")
rr = np.array(c["params"]["x"], float)
check(T, "chart: 希薄化 2/(1+r)", allclose(series(c, 0), 2 / (1 + rr)), r4(2 / (1 + rr)))
check(T, "chart: 真の係数 2", allclose(series(c, 1), [2] * len(rr)), "ok")
check(T, "chart: r=0.25 が例1の 1.6", close(2 / 1.25, 1.6, 1e-12), "ok")
Nn = 1_000_000
xst = rng.normal(0, 2, Nn); xo = xst + rng.normal(0, 1, Nn); yo = 1 + 2 * xst + rng.normal(0, 1, Nn)
sl = np.cov(xo, yo)[0, 1] / xo.var(ddof=1)
se = sqrt((yo.var() - sl ** 2 * xo.var()) / (Nn * xo.var()))
check(T, "sim: n=10^6 の OLS 傾き ≈1.6（±4SE）、本文の 1.6009 も ±4SE 内", abs(sl - 1.6) < 4 * se and abs(1.6009 - 1.6) < 4 * se, (sl, se))
# 同時性
a, bq_, c_, dq_, u, vv_, su, sv = sp.symbols("a b c d u v sigma_u sigma_v", positive=True)
psol = sp.solve([sp.Symbol("q") - (a - bq_ * sp.Symbol("p") + u), sp.Symbol("q") - (c_ + dq_ * sp.Symbol("p") + vv_)], [sp.Symbol("p"), sp.Symbol("q")])
check(T, "sympy: 均衡価格 p=(a−c+u−v)/(b+d)", sp.simplify(psol[sp.Symbol("p")] - (a - c_ + u - vv_) / (bq_ + dq_)) == 0, psol[sp.Symbol("p")])
# Cov を u,v の係数から
pu = sp.diff(psol[sp.Symbol("p")], u); pv = sp.diff(psol[sp.Symbol("p")], vv_)
qu = sp.diff(psol[sp.Symbol("q")], u); qv = sp.diff(psol[sp.Symbol("q")], vv_)
covpq = pu * qu * su ** 2 + pv * qv * sv ** 2
varp = pu ** 2 * su ** 2 + pv ** 2 * sv ** 2
check(T, "sympy: Cov(p,u)=σu²/(b+d)", sp.simplify(pu * su ** 2 - su ** 2 / (bq_ + dq_)) == 0, "ok")
check(T, "sympy: plim β̂=(dσu²−bσv²)/(σu²+σv²)", sp.simplify(covpq / varp - (dq_ * su ** 2 - bq_ * sv ** 2) / (su ** 2 + sv ** 2)) == 0, sp.simplify(covpq / varp))
check(T, "例2 plim=(0.5−1)/2=−0.25", close((0.5 * 1 - 1 * 1) / 2, -0.25, 1e-12), -0.25)
uu = rng.normal(size=Nn); vv = rng.normal(size=Nn)
pp = (10 - 2 + uu - vv) / 1.5; qq = 10 - pp + uu
slq = np.cov(pp, qq)[0, 1] / pp.var(ddof=1)
check(T, "sim: 需要量を価格で回帰した傾き ≈−0.25（±0.003）、本文 −0.2499 も範囲内", abs(slq + 0.25) < 0.003 and abs(-0.2499 + 0.25) < 0.003, slq)
zc = rng.normal(size=Nn)
pp2 = (10 - 2 + uu - vv - zc) / 1.5; qq2 = 10 - pp2 + uu  # 供給 q=2+0.5p+v+z
ivs = np.cov(zc, qq2)[0, 1] / np.cov(zc, pp2)[0, 1]
check(T, "sim: 供給シフト z を操作変数にすると傾き ≈−1（±0.01）", abs(ivs + 1) < 0.01, ivs)
check(T, "chart: iv-scatter params rho=0.6 は範囲内", -0.9 <= chart(T, "iv-scatter")["params"]["rho"] <= 0.9, chart(T, "iv-scatter")["params"])

# =====================================================================
# 操作変数法
# =====================================================================
T = "instrumental-variable-method"
z = np.arange(1, 7, dtype=float); xv = np.array([2.0, 1.5, 3.5, 3.0, 5.0, 4.5]); yv = np.array([3.4, 2.0, 4.6, 3.1, 5.5, 4.3])
check(T, "例1 平均 3.5, 3.25, 3.8167", allclose([z.mean(), xv.mean(), yv.mean()], [3.5, 3.25, 3.8167]), r4([z.mean(), xv.mean(), yv.mean()]))
S = lambda a, b: np.sum((a - a.mean()) * (b - b.mean()))
check(T, "例1 S_zx=11.25, S_zy=6.75, S_xx=9.375, S_xy=7.625", allclose([S(z, xv), S(z, yv), S(xv, xv), S(xv, yv)], [11.25, 6.75, 9.375, 7.625], 1e-10), r4([S(z, xv), S(z, yv), S(xv, xv), S(xv, yv)]))
bols = S(xv, yv) / S(xv, xv); biv = S(z, yv) / S(z, xv)
check(T, "例1 OLS 0.8133, IV 0.6000", close(bols, 0.8133) and close(biv, 0.6, 1e-12), (bols, biv))
check(T, "例1 IV 切片 1.8667、OLS 切片 1.1733", close(yv.mean() - biv * xv.mean(), 1.8667) and close(yv.mean() - bols * xv.mean(), 1.1733), (yv.mean() - biv * xv.mean(), yv.mean() - bols * xv.mean()))
Xm = np.column_stack([np.ones(6), xv]); Zm = np.column_stack([np.ones(6), z])
check(T, "例1 Z'X=[[6,19.5],[21,79.5]], Z'y=(22.9,86.9)", allclose(Zm.T @ Xm, [[6, 19.5], [21, 79.5]], 1e-10) and allclose(Zm.T @ yv, [22.9, 86.9], 1e-10), (r4(Zm.T @ Xm), r4(Zm.T @ yv)))
biv_m = np.linalg.solve(Zm.T @ Xm, Zm.T @ yv)
check(T, "例1 (Z'X)⁻¹Z'y=(1.8667, 0.6000)", allclose(biv_m, [1.8667, 0.6]), r4(biv_m))
check(T, "例1 z と x の相関 0.8783", close(np.corrcoef(z, xv)[0, 1], 0.8783), np.corrcoef(z, xv)[0, 1])
ei = yv - Xm @ biv_m
s2 = ei @ ei / 4
check(T, "例1 s²=0.4733（残差平方和/(n−2)）", close(s2, 0.4733), s2)
Vb = s2 * np.linalg.inv(Zm.T @ Xm) @ Zm.T @ Zm @ np.linalg.inv(Xm.T @ Zm)
check(T, "例1 傾きの標準誤差 0.2558", close(sqrt(Vb[1, 1]), 0.2558), sqrt(Vb[1, 1]))
check(T, "単回帰の分散 σ²/(n s_x² r²) と行列形が一致", close(Vb[1, 1], s2 / (S(xv, xv) * np.corrcoef(z, xv)[0, 1] ** 2), 1e-10), Vb[1, 1])
c = chart(T, "line-data")
xc = np.array(c["params"]["x"]); order = np.argsort(xv)
check(T, "chart: x と y（x の小さい順）", allclose(xc, xv[order]) and allclose(series(c, 0), yv[order]), (xc, series(c, 0)))
check(T, "chart: OLS の直線", allclose(series(c, 1), (yv.mean() - bols * xv.mean()) + bols * xc), r4((yv.mean() - bols * xv.mean()) + bols * xc))
check(T, "chart: IV の直線", allclose(series(c, 2), biv_m[0] + biv_m[1] * xc), r4(biv_m[0] + biv_m[1] * xc))
check(T, "chart: キャプションの傾き 0.8133 と 0.6", close(bols, 0.8133) and close(biv, 0.6), "ok")
check(T, "例2 ワルド (5.4−4.4)/(0.7−0.2)=2.0", close((5.4 - 4.4) / (0.7 - 0.2), 2.0, 1e-12), (5.4 - 4.4) / (0.7 - 0.2))
zb = rng.integers(0, 2, 50).astype(float); xb = rng.normal(size=50) + zb; yb = rng.normal(size=50) + xb
wald = (yb[zb == 1].mean() - yb[zb == 0].mean()) / (xb[zb == 1].mean() - xb[zb == 0].mean())
check(T, "2値 z では s_zy/s_zx = ワルド推定量（ランダム例）", close(S(zb, yb) / S(zb, xb), wald, 1e-10), wald)
# sim: 一致性と分散比
R = 4000; nn = 2000; est = []; esto = []
for _ in range(R):
    zz = rng.normal(size=nn); e = rng.normal(size=nn); xx = 0.8 * zz + 0.6 * e + rng.normal(size=nn) * 0.6
    yy_ = 1 + 0.5 * xx + e
    est.append(S(zz, yy_) / S(zz, xx)); esto.append(S(xx, yy_) / S(xx, xx))
est = np.array(est); esto = np.array(esto)
rzx2 = 0.64 / (0.64 + 0.36 + 0.36)
vth = 1 / (nn * (0.64 + 0.36 + 0.36) * rzx2)
check(T, "sim: IV は β=0.5 に近い（±4SE）、OLS は plim=0.5+0.6/1.36 に偏る",
      abs(est.mean() - 0.5) < 4 * est.std() / sqrt(R) + 0.001 and abs(esto.mean() - (0.5 + 0.6 / 1.36)) < 0.005, (est.mean(), esto.mean()))
check(T, "sim: V[β̂_IV]≈σ²/(n s_x² r²)（±6%）", abs(est.var() / vth - 1) < 0.06, (est.var(), vth))
zz = rng.normal(size=Nn); e = rng.normal(size=Nn) + 0.1 * zz; xx = 0.5 * zz + 0.5 * e + rng.normal(size=Nn)
yy_ = 0.5 * xx + e
check(T, "sim: 外生性の破れ plim=β+Cov(z,ε)/Cov(z,x)（±0.01）", abs(S(zz, yy_) / S(zz, xx) - (0.5 + 0.1 / (0.5 + 0.05))) < 0.01, (S(zz, yy_) / S(zz, xx), 0.5 + 0.1 / 0.55))

# =====================================================================
# 二段階最小二乗法
# =====================================================================
T = "two-stage-least-squares"
z1 = np.array([0, 0, 0, 0, 1, 1, 1, 1.]); z2 = np.array([1, 2, 3, 4, 1, 2, 3, 4.])
xv = np.array([1.2, 1.9, 2.1, 3.0, 2.4, 2.8, 3.9, 4.1]); yv = np.array([3.0, 3.1, 4.2, 4.4, 4.0, 4.9, 5.3, 6.4])
Zm = np.column_stack([np.ones(8), z1, z2]); Xm = np.column_stack([np.ones(8), xv])
g1 = ols(Zm, xv)
check(T, "第1段階 x̂=0.575+1.25z1+0.59z2", allclose(g1, [0.575, 1.25, 0.59], 1e-10), r4(g1))
xh = Zm @ g1
R2 = 1 - np.sum((xv - xh) ** 2) / S(xv, xv)
check(T, "第1段階 R²=0.9665", close(R2, 0.9665), R2)
F1 = (R2 / 2) / ((1 - R2) / 5)
check(T, "第1段階 F=72.12（自由度(2,5)）", close(F1, 72.12, 5e-3), F1)
check(T, "x̂ の値", allclose(xh, [1.165, 1.755, 2.345, 2.935, 2.415, 3.005, 3.595, 4.185], 1e-10), r4(xh))
check(T, "x̂ の偏差平方和 6.606、偏差積和 7.493", close(S(xh, xh), 6.606, 5e-4) and close(S(xh, yv), 7.493, 5e-4), (S(xh, xh), S(xh, yv)))
b1h = S(xh, yv) / S(xh, xh)
check(T, "β̂1=1.1343, ȳ=4.4125, x̂平均 2.675, β̂0=1.3783",
      close(b1h, 1.1343) and close(yv.mean(), 4.4125, 1e-10) and close(xh.mean(), 2.675, 1e-10) and close(yv.mean() - b1h * xh.mean(), 1.3783), (b1h, yv.mean() - b1h * xh.mean()))
check(T, "本文の丸め値 7.493/6.606=1.1343", close(7.493 / 6.606, 1.1343), 7.493 / 6.606)
PZ = Zm @ np.linalg.solve(Zm.T @ Zm, Zm.T)
b2s = np.linalg.solve(Xm.T @ PZ @ Xm, Xm.T @ PZ @ yv)
check(T, "公式 (X'P_ZX)⁻¹X'P_Zy=(1.3783, 1.1343)", allclose(b2s, [1.3783, 1.1343]), r4(b2s))
check(T, "P_Z は対称・べき等", allclose(PZ, PZ.T, 1e-12) and allclose(PZ @ PZ, PZ, 1e-12), "ok")
check(T, "OLS (1.5506, 1.0699)", allclose(ols(Xm, yv), [1.5506, 1.0699]), r4(ols(Xm, yv)))
e2 = yv - Xm @ b2s; s2 = e2 @ e2 / 6
se2 = sqrt(s2 * np.linalg.inv(Xm.T @ PZ @ Xm)[1, 1])
check(T, "σ̂²=0.1856, SE=0.1676（元の x の残差）", close(s2, 0.1856) and close(se2, 0.1676), (s2, se2))
Xh = np.column_stack([np.ones(8), xh]); eh = yv - Xh @ b2s; s2h = eh @ eh / 6
seh = sqrt(s2h * np.linalg.inv(Xh.T @ Xh)[1, 1])
check(T, "第2段階をそのまま：σ̂²=0.0683, SE=0.1017", close(s2h, 0.0683) and close(seh, 0.1017), (s2h, seh))
# 丁度識別で IV と一致（ランダム）
Zr = np.column_stack([np.ones(30), rng.normal(size=30)]); Xr = np.column_stack([np.ones(30), Zr[:, 1] + rng.normal(size=30)]); yr = rng.normal(size=30)
PZr = Zr @ np.linalg.solve(Zr.T @ Zr, Zr.T)
check(T, "丁度識別で 2SLS = (Z'X)⁻¹Z'y（ランダム例）", allclose(np.linalg.solve(Xr.T @ PZr @ Xr, Xr.T @ PZr @ yr), np.linalg.solve(Zr.T @ Xr, Zr.T @ yr), 1e-10), "ok")
Xhr = PZr @ Xr
check(T, "(X̂'X̂)⁻¹X̂'y = (X'P_ZX)⁻¹X'P_Zy（ランダム例）", allclose(ols(Xhr, yr), np.linalg.solve(Xr.T @ PZr @ Xr, Xr.T @ PZr @ yr), 1e-10), "ok")
c = chart(T, "line-data")
check(T, "chart: 観測 x と x̂", allclose(series(c, 0), xv, 1e-12) and allclose(series(c, 1), xh, 1e-10), "ok")
# sim: 過剰識別の一致性
R = 3000; ests = []
for _ in range(R):
    Z1 = rng.normal(size=(1000, 2)); e = rng.normal(size=1000); xx = Z1 @ [0.6, 0.4] + 0.7 * e + rng.normal(size=1000) * 0.5
    yy_ = 1 + 0.8 * xx + e
    Zf = np.column_stack([np.ones(1000), Z1]); Xf = np.column_stack([np.ones(1000), xx])
    xhf = Zf @ ols(Zf, xx)
    ests.append(ols(np.column_stack([np.ones(1000), xhf]), yy_)[1])
ests = np.array(ests)
check(T, "sim: 2SLS（操作変数2個）の平均 ≈0.8（±0.005）", abs(ests.mean() - 0.8) < 0.005, ests.mean())

# =====================================================================
# 欠落変数バイアス
# =====================================================================
T = "omitted-variable-bias"
x1 = np.array([10, 12, 12, 14, 16, 18.]); x2 = np.array([95, 100, 108, 104, 115, 120.]); yv = np.array([2.9, 3.3, 3.8, 3.7, 4.5, 4.9])
bf = ols(np.column_stack([np.ones(6), x1, x2]), yv)
bt = S(x1, yv) / S(x1, x1); dl = S(x1, x2) / S(x1, x1)
check(T, "例1 β̂1=0.0646, β̂2=0.0602", allclose(bf[1:], [0.0646, 0.0602]), r4(bf))
check(T, "例1 β̃1=0.2423", close(bt, 0.2423), bt)
check(T, "例1 δ̂1=2.9538、相関 0.9312", close(dl, 2.9538) and close(np.corrcoef(x1, x2)[0, 1], 0.9312), (dl, np.corrcoef(x1, x2)[0, 1]))
check(T, "例1 恒等式 β̃1=β̂1+β̂2δ̂1（厳密）", close(bf[1] + bf[2] * dl, bt, 1e-12), bf[1] + bf[2] * dl)
check(T, "例1 丸め値 0.0646+0.0602×2.9538=0.2424≈0.2423（丸め誤差内）", abs(0.0646 + 0.0602 * 2.9538 - 0.2423) < 2e-4, 0.0646 + 0.0602 * 2.9538)
check(T, "例1 約3.8倍", abs(bt / bf[1] - 3.8) < 0.06, bt / bf[1])
check(T, "例2 1+0.5×0.6=1.3、ρ=−0.6 で 0.7", close(1 + 0.5 * 0.6, 1.3, 1e-12) and close(1 - 0.3, 0.7, 1e-12), "ok")
c = chart(T, "line-data")
rh = np.array(c["params"]["x"])
check(T, "chart: β1+β2ρ", allclose(series(c, 0), 1 + 0.5 * rh, 1e-12), r4(1 + 0.5 * rh))
check(T, "chart: 真の β1=1", allclose(series(c, 1), [1] * 7), "ok")
# 恒等式ランダム例 & 条件付き期待値
X1r = rng.normal(size=40); X2r = 0.5 * X1r + rng.normal(size=40)
d1 = S(X1r, X2r) / S(X1r, X1r)
R = 100000
Eps = rng.normal(size=(R, 40))
Yr = 1 + 1.0 * X1r + 0.5 * X2r + Eps
tl = ((X1r - X1r.mean()) @ Yr.T) / S(X1r, X1r)
check(T, "sim: X 固定で E[β̃1]=β1+β2δ̂1（±4SE）", abs(tl.mean() - (1 + 0.5 * d1)) < 4 * tl.std() / sqrt(R), (tl.mean(), 1 + 0.5 * d1))
Yr0 = Yr[0]
bfr = ols(np.column_stack([np.ones(40), X1r, X2r]), Yr0)
check(T, "恒等式（ランダム例）", close(bfr[1] + bfr[2] * d1, tl[0], 1e-10), "ok")

# =====================================================================
# 非線形回帰
# =====================================================================
T = "nonlinear-regression"
xn = np.arange(6, dtype=float); yn = np.array([10.3, 6.0, 3.9, 2.1, 1.5, 0.8])
fN = lambda th: th[0] * np.exp(-th[1] * xn)
Jf = lambda th: np.column_stack([np.exp(-th[1] * xn), -th[0] * xn * np.exp(-th[1] * xn)])
th0 = np.array([10, 0.5])
r0 = yn - fN(th0)
check(T, "初期値 (10,0.5) で S=0.1824", close(r0 @ r0, 0.1824), r0 @ r0)
J0 = Jf(th0)
check(T, "J'J=[[1.5781,−8.9486],[−8.9486,181.8803]]", allclose(J0.T @ J0, [[1.5781, -8.9486], [-8.9486, 181.8803]]), r4(J0.T @ J0))
check(T, "J'r=(0.3306, −1.0608)", allclose(J0.T @ r0, [0.3306, -1.0608]), r4(J0.T @ r0))
dlt = np.linalg.solve(J0.T @ J0, J0.T @ r0)
check(T, "更新量 (0.2447, 0.0062)", allclose(dlt, [0.2447, 0.0062]), r4(dlt))
th1 = th0 + dlt
check(T, "1回目 (10.2447, 0.5062)", allclose(th1, [10.2447, 0.5062]), r4(th1))
th = th1.copy()
for _ in range(50):
    J = Jf(th); th = th + np.linalg.solve(J.T @ J, J.T @ (yn - fN(th)))
th2 = th1 + np.linalg.solve(Jf(th1).T @ Jf(th1), Jf(th1).T @ (yn - fN(th1)))
check(T, "2回目でほぼ収束（|Δ|<1e-3）", np.max(np.abs(th2 - th)) < 1e-3, r4(th2 - th, 6))
Sh = np.sum((yn - fN(th)) ** 2)
check(T, "â=10.2452, b̂=0.5062, S=0.1080", allclose(th, [10.2452, 0.5062]) and close(Sh, 0.1080), (r4(th), Sh))
cf = optimize.curve_fit(lambda t, a, b: a * np.exp(-b * t), xn, yn, p0=[10, 0.5])[0]
check(T, "scipy curve_fit でも一致", allclose(cf, th, 1e-6), r4(cf))
s2n = Sh / 4
Jh = Jf(th); sen = np.sqrt(np.diag(s2n * np.linalg.inv(Jh.T @ Jh)))
check(T, "s²=0.0270、SE a=0.1543, b=0.0142", close(s2n, 0.0270) and allclose(sen, [0.1543, 0.0142]), (s2n, r4(sen)))
check(T, "半減期 log2/b̂=1.369", close(log(2) / th[1], 1.369, 5e-4), log(2) / th[1])
gl = ols(np.column_stack([np.ones(6), xn]), np.log(yn))
thl = np.array([exp(gl[0]), -gl[1]])
check(T, "対数線形化 (10.1968, 0.5016)", allclose(thl, [10.1968, 0.5016]), r4(thl))
check(T, "対数線形化の元尺度 S=0.1116", close(np.sum((yn - fN(thl)) ** 2), 0.1116), np.sum((yn - fN(thl)) ** 2))
c = chart(T, "line-data")
check(T, "chart: 観測値", allclose(series(c, 0), yn, 1e-12), "ok")
check(T, "chart: 非線形最小二乗の曲線（3桁）", allclose(series(c, 1), fN(th), 5e-4), r4(fN(th), 3))
check(T, "chart: 対数変換の曲線（3桁）", allclose(series(c, 2), fN(thl), 5e-4), r4(fN(thl), 3))
tt, K, r_, t0, bG_, cG = sp.symbols("t K r t0 b c", positive=True)
lg = K / (1 + sp.exp(-r_ * (tt - t0)))
infl = sp.solve(sp.diff(lg, tt, 2), tt)
check(T, "sympy: ロジスティックの変曲点 t0 で値 K/2", infl == [t0] and sp.simplify(lg.subs(tt, t0) - K / 2) == 0, infl)
gp = K * sp.exp(-bG_ * sp.exp(-cG * tt))
infg = sp.solve(sp.diff(gp, tt, 2), tt)
check(T, "sympy: ゴンペルツの変曲点の値 K/e", len(infg) == 1 and sp.simplify(gp.subs(tt, infg[0]) - K / sp.E) == 0, infg)

# =====================================================================
# 分位点回帰
# =====================================================================
T = "quantile-regression"
rho_t = lambda u, tau: u * (tau - (u < 0))
yq = np.array([3, 5, 6, 8, 9, 12, 15, 22, 30.])
Lq = lambda c, tau: rho_t(yq - c, tau).sum()
check(T, "例1 τ=0.5: c=8,9,12 で 29.0, 28.5, 30.0", allclose([Lq(8, .5), Lq(9, .5), Lq(12, .5)], [29.0, 28.5, 30.0], 1e-10), [Lq(8, .5), Lq(9, .5), Lq(12, .5)])
check(T, "例1 τ=0.8: c=15,22,30 で 27.0, 25.6, 32.0", allclose([Lq(15, .8), Lq(22, .8), Lq(30, .8)], [27.0, 25.6, 32.0], 1e-10), [Lq(15, .8), Lq(22, .8), Lq(30, .8)])
cg = np.linspace(0, 35, 35001)
check(T, "例1 最小点は 9 と 22", close(cg[np.argmin([Lq(cc, .5) for cc in cg])], 9, 1e-9) and close(cg[np.argmin([Lq(cc, .8) for cc in cg])], 22, 1e-9), "ok")
check(T, "例1 内訳 6.4 と 19.2", close(0.8 * 8, 6.4, 1e-12) and close(0.2 * (19 + 17 + 16 + 14 + 13 + 10 + 7), 19.2, 1e-12), "ok")
check(T, "ρ_0.5(u)=|u|/2", allclose(rho_t(np.linspace(-3, 3, 13), 0.5), np.abs(np.linspace(-3, 3, 13)) / 2, 1e-12), "ok")
# 右微分 #{y≤c}−nτ
cc = 10.0; h = 1e-7
check(T, "右微分 = #{y≤c}−nτ（c=10, τ=0.8）", close((Lq(cc + h, .8) - Lq(cc, .8)) / h, np.sum(yq <= cc) - 9 * 0.8, 1e-5), (Lq(cc + h, .8) - Lq(cc, .8)) / h)
# 母集団版
for tau in (0.2, 0.9):
    m = optimize.minimize_scalar(lambda c: integrate.quad(lambda t: rho_t(t - c, tau) * stats.gamma.pdf(t, 2), 0, np.inf)[0], bounds=(0, 10), method="bounded", options={"xatol": 1e-9})
    check(T, f"E[ρτ(Y−c)] の最小点 = τ 分位点（ガンマ(2), τ={tau}）", close(m.x, stats.gamma.ppf(tau, 2), 1e-4), (m.x, stats.gamma.ppf(tau, 2)))
# 例2 LP
xq = np.arange(1, 11, dtype=float); yq2 = np.array([2.1, 3.0, 2.6, 4.4, 3.5, 6.0, 4.2, 7.9, 5.0, 9.3])


def qreg(tau):
    nq = len(xq)
    Xq = np.column_stack([np.ones(nq), xq])
    cvec = np.concatenate([[0, 0], tau * np.ones(nq), (1 - tau) * np.ones(nq)])
    Aeq = np.hstack([Xq, np.eye(nq), -np.eye(nq)])
    res = optimize.linprog(cvec, A_eq=Aeq, b_eq=yq2, bounds=[(None, None)] * 2 + [(0, None)] * (2 * nq), method="highs")
    return res.x[:2], res.fun


qtab = {0.1: (1.4, 0.4, 0, 3), 0.5: (1.3333, 0.7667, 4, 2), 0.9: (1.3667, 0.8167, 8, 2)}
qfit = {}
for tau, (ci, cs, cb, con) in qtab.items():
    bq_v, fv = qreg(tau)
    qfit[tau] = bq_v
    # 主張された直線の損失が LP の最適値と一致（解の一意性に依存しない確認）
    claim_loss = rho_t(yq2 - (ci + cs * xq), tau).sum()
    resid = yq2 - (ci + cs * xq)
    below = int(np.sum(resid < -1e-3)); on = int(np.sum(np.abs(resid) <= 1e-3))
    check(T, f"例2 τ={tau}: 切片 {ci}, 傾き {cs}（LP 解）", allclose(bq_v, [ci, cs]), r4(bq_v))
    check(T, f"例2 τ={tau}: 主張の直線の損失＝LP 最適値", close(claim_loss, fv, 1e-3), (claim_loss, fv))
    check(T, f"例2 τ={tau}: 下 {cb} 点・直線上 {con} 点、下≤nτ≤下+上", below == cb and on == con and below <= 10 * tau <= below + on, (below, on))
bols_q = ols(np.column_stack([np.ones(10), xq]), yq2)
check(T, "例2 OLS 1.2267+0.6497x", allclose(bols_q, [1.2267, 0.6497]), r4(bols_q))
c = chart(T, "line-data", 0)
check(T, "chart1: 観測値", allclose(series(c, 0), yq2, 1e-12), "ok")
for i, tau in enumerate((0.1, 0.5, 0.9), start=1):
    check(T, f"chart1: τ={tau} の直線（LP の厳密解で4桁）", allclose(series(c, i), qfit[tau][0] + qfit[tau][1] * xq), r4(qfit[tau][0] + qfit[tau][1] * xq))
c = chart(T, "line-data", 1)
uq = np.array(c["params"]["x"])
for i, tau in enumerate((0.1, 0.5, 0.9)):
    check(T, f"chart2: チェック関数 τ={tau}", allclose(series(c, i), rho_t(uq, tau), 1e-10), r4(rho_t(uq, tau)))

# =====================================================================
# 多項式回帰
# =====================================================================
T = "polynomial-regression"
xp = np.arange(1, 8, dtype=float); yp = np.array([4.3, 7.2, 9.0, 9.7, 9.6, 7.9, 5.7])
ST = S(yp, yp)
ptab = {1: (23.7414, 0.0547, -0.1344, 14.549), 2: (0.0610, 0.9976, 0.9964, -25.205), 3: (0.0593, 0.9976, 0.9953, -23.399)}
Se_ = {}
for k, (cse, cr2, car2, caic) in ptab.items():
    Xk = np.column_stack([xp ** j for j in range(k + 1)])
    e = yp - Xk @ ols(Xk, yp); se_ = e @ e; Se_[k] = se_
    r2 = 1 - se_ / ST; ar2 = 1 - (se_ / (7 - k - 1)) / (ST / 6); aic = 7 * log(se_ / 7) + 2 * (k + 2)
    check(T, f"次数{k}: S_e {cse}, R² {cr2}, adjR² {car2}, AIC {caic}",
          close(se_, cse) and close(r2, cr2) and close(ar2, car2) and close(aic, caic, 5e-4), r4([se_, r2, ar2, aic]))
check(T, "x と x² の相関 0.9774", close(np.corrcoef(xp, xp ** 2)[0, 1], 0.9774), np.corrcoef(xp, xp ** 2)[0, 1])
u = xp - 4
check(T, "中心化で u と u² は無相関", abs(np.corrcoef(u, u ** 2)[0, 1]) < 1e-12, np.corrcoef(u, u ** 2)[0, 1])
check(T, "Σu=0, Σu²=28, Σu³=0, Σu⁴=196", allclose([u.sum(), (u ** 2).sum(), (u ** 3).sum(), (u ** 4).sum()], [0, 28, 0, 196], 1e-10), "ok")
Xu = np.column_stack([np.ones(7), u, u ** 2])
check(T, "正規方程式の右辺 (53.4, 6.2, 169.0)", allclose(Xu.T @ yp, [53.4, 6.2, 169.0], 1e-10), r4(Xu.T @ yp))
gm = ols(Xu, yp)
check(T, "γ=(9.7524, 0.2214, −0.5310)", allclose(gm, [9.7524, 0.2214, -0.5310]), r4(gm))
check(T, "84γ2=169.0−213.6=−44.6", close(169.0 - 4 * 53.4, -44.6, 1e-10) and close(-44.6 / 84, -0.5310), -44.6 / 84)
bx_ = ols(np.column_stack([np.ones(7), xp, xp ** 2]), yp)
check(T, "展開形 0.3714+4.4690x−0.5310x²", allclose(bx_, [0.3714, 4.4690, -0.5310]), r4(bx_))
xs_ = -bx_[1] / (2 * bx_[2])
check(T, "頂点 x*=4.2085、予測 9.7755", close(xs_, 4.2085) and close(bx_ @ [1, xs_, xs_ ** 2], 9.7755), (xs_, bx_ @ [1, xs_, xs_ ** 2]))
F2 = (Se_[1] - Se_[2]) / (Se_[2] / 4)
check(T, "2次項の部分 F=1554.0（丸め前の S_e で）、p<0.001", close(F2, 1554.0, 0.05) and stats.f.sf(F2, 1, 4) < 0.001, (F2, stats.f.sf(F2, 1, 4)))
check(T, "本文の丸め値 (23.7414−0.0610)/(0.0610/4) は 1552.8（S_e=0.0610 の丸めによる差、許容）", abs((23.7414 - 0.0610) / (0.0610 / 4) - 1554.0) < 2, (23.7414 - 0.0610) / (0.0610 / 4))
F3 = (Se_[2] - Se_[3]) / (Se_[3] / 3)
check(T, "3次項 F=0.0843, p=0.790", close(F3, 0.0843) and close(stats.f.sf(F3, 1, 3), 0.790, 5e-4), (F3, stats.f.sf(F3, 1, 3)))
c = chart(T, "line-data")
check(T, "chart: 観測値", allclose(series(c, 0), yp, 1e-12), "ok")
X1p = np.column_stack([np.ones(7), xp])
check(T, "chart: 1次式（4桁）", allclose(series(c, 1), X1p @ ols(X1p, yp)), r4(X1p @ ols(X1p, yp)))
check(T, "chart: 2次式（3桁）", allclose(series(c, 2), Xu @ gm, 5e-4), r4(Xu @ gm, 3))
check(T, "chart: bias-variance degree=2 は範囲内", 1 <= chart(T, "bias-variance")["params"]["degree"] <= 12, "ok")

# =====================================================================
# 変数選択
# =====================================================================
T = "variable-selection"
X1 = np.array([2, 4, 5, 9, 4, 0, 6, 2, 4, 6.]); X2 = np.array([4, 3, 4, 4, 5, 6, 4, 2, 5, 4.]); X3 = np.array([3, 3, 5, 6, 5, 4, 6, 1, 5, 6.])
yv = np.array([7.0, 8.3, 10.1, 14.2, 11.1, 7.1, 12.7, 4.9, 9.1, 10.3])
nv = 10; STv = S(yv, yv)
check(T, "S_T=70.656", close(STv, 70.656, 5e-4), STv)
cols = {"x1": X1, "x2": X2, "x3": X3}
vtab = {(): (70.656, 23.552, 24.158, 0), ("x1",): (16.328, 10.903, 11.811, 0.7400), ("x2",): (66.389, 24.929, 25.837, -0.0571),
        ("x3",): (15.215, 10.197, 11.105, 0.7577), ("x1", "x2"): (5.115, 1.296, 2.507, 0.9069), ("x1", "x3"): (6.035, 2.950, 4.161, 0.8902),
        ("x2", "x3"): (10.418, 8.410, 9.620, 0.8104), ("x1", "x2", "x3"): (4.983, 3.034, 4.547, 0.8942)}
aics = {}
for key, (cse, caic, cbic, car2) in vtab.items():
    k = len(key)
    Xk = np.column_stack([np.ones(nv)] + [cols[s] for s in key])
    e = yv - Xk @ ols(Xk, yv); se_ = e @ e
    aic = nv * log(se_ / nv) + 2 * (k + 2); bic = nv * log(se_ / nv) + (k + 2) * log(nv); ar2 = 1 - (se_ / (nv - k - 1)) / (STv / (nv - 1))
    aics[key] = aic
    check(T, f"{'+'.join(key) or 'なし'}: S_e {cse}, AIC {caic}, BIC {cbic}, adjR² {car2}",
          close(se_, cse, 5e-4) and close(aic, caic, 5e-4) and close(bic, cbic, 5e-4) and close(ar2, car2), r4([se_, aic, bic, ar2], 4))
check(T, "10log(5.1152/10)=−6.704、+8=1.296", close(10 * log(5.1152 / 10), -6.704, 5e-4) and close(10 * log(5.1152 / 10) + 8, 1.296, 5e-4), 10 * log(5.1152 / 10))
best = min(aics, key=aics.get)
check(T, "総当たり AIC 最良は {x1,x2}", best == ("x1", "x2"), best)
bsel = ols(np.column_stack([np.ones(nv), X1, X2]), yv)
check(T, "ŷ=0.8678+1.0462x1+1.0288x2", allclose(bsel, [0.8678, 1.0462, 1.0288]), r4(bsel))


def forward():
    cur = (); path = [aics[()]]
    while True:
        cands = [tuple(sorted(cur + (v,))) for v in cols if v not in cur]
        if not cands:
            return cur, path
        nb = min(cands, key=lambda k: aics[k])
        if aics[nb] < aics[cur]:
            cur = nb; path.append(aics[cur])
        else:
            return cur, path


def backward():
    cur = ("x1", "x2", "x3"); path = [aics[cur]]
    while cur:
        cands = [tuple(v for v in cur if v != d) for d in cur]
        nb = min(cands, key=lambda k: aics[k])
        if aics[nb] < aics[cur]:
            cur = nb; path.append(aics[cur])
        else:
            return cur, path
    return cur, path


fsel, fpath = forward(); bsel_, bpath = backward()
check(T, "変数増加法は {x1,x3} で止まる（x2 を足すと 3.034 に悪化）", fsel == ("x1", "x3") and aics[("x1", "x2", "x3")] > aics[("x1", "x3")], (fsel, r4(fpath, 3)))
check(T, "変数減少法は {x1,x2} で止まる", bsel_ == ("x1", "x2"), (bsel_, r4(bpath, 3)))
check(T, "BIC・adjR² でも {x1,x2} が最良", min(vtab, key=lambda k: vtab[k][2]) == ("x1", "x2") and max(vtab, key=lambda k: vtab[k][3]) == ("x1", "x2"), "ok")
check(T, "n≥8 で log n>2", log(8) > 2 and log(7) < 2, (log(7), log(8)))
# AIC 条件と F≈2
Se0, Se1 = 16.328, 5.115
check(T, "AIC 低下 ⇔ S_e/S_e'>e^{2/n}（例：x1→x1+x2）", (aics[("x1", "x2")] < aics[("x1",)]) == (Se0 / Se1 > exp(2 / nv)), (Se0 / Se1, exp(2 / nv)))
nb_ = 10000; kk = 3
Fthr = (exp(2 / nb_) - 1) * (nb_ - kk - 2)
check(T, "n 大で AIC の境目は F≈2（n=10000）", abs(Fthr - 2) < 0.01, Fthr)
c = chart(T, "bar-data")
order = [(), ("x1",), ("x2",), ("x3",), ("x1", "x2"), ("x1", "x3"), ("x2", "x3"), ("x1", "x2", "x3")]
check(T, "chart: bar-data の AIC", allclose(c["params"]["values"], [aics[k] for k in order], 5e-4), r4([aics[k] for k in order], 3))
c = chart(T, "line-data")
bestk = [min(aics[k] for k in order if len(k) == j) for j in range(4)]
check(T, "chart: 各変数の数での最良", allclose(series(c, 0), bestk, 5e-4), r4(bestk, 3))
fp = fpath + [aics[("x1", "x2", "x3")]]
check(T, "chart: 変数増加法の経路（停止後は全変数の値で補完）", allclose(series(c, 1), fp, 5e-4), r4(fp, 3))
bp = [aics[()], aics[("x1",)]] + bpath[::-1]
check(T, "chart: 変数減少法の経路（x1 単独は外した後の候補）", allclose(series(c, 2), bp, 5e-4), r4(bp, 3))


# =====================================================================
# 重要度Cの用語（elastic-net, principal-component-regression, spline-regression）
# 乱数は seed=20261023+19
# =====================================================================
rngC = np.random.default_rng(20261023 + 19)
x1C = np.array([-2., -1, 0, 1, 2]); x2C = np.array([-2., 0, -1, 1, 2]); yC = np.array([-3.9, -1.2, -0.6, 1.8, 3.9])
XC = np.column_stack([x1C, x2C]); GC = XC.T @ XC; gC = XC.T @ yC
bOC = np.linalg.solve(GC, gC)

T = "elastic-net"
check(T, "XᵀX=[[10,9],[9,10]]、Xᵀy=(18.6,18.0)、OLS (1.2632,0.6632)", allclose(GC, [[10, 9], [9, 10]], 1e-12) and allclose(gC, [18.6, 18.0], 1e-10) and allclose(bOC, [1.2632, .6632]), r4(bOC))
L1, L2 = 10.0, 5.0
rhs = gC - L1 * np.ones(2)
check(T, "右辺 (8.6, 8.0)、行列 [[15,9],[9,15]]、det=144", allclose(rhs, [8.6, 8.0], 1e-10) and allclose(GC + L2 * np.eye(2), [[15, 9], [9, 15]], 1e-12) and close(np.linalg.det(GC + L2 * np.eye(2)), 144, 1e-9), r4(rhs))
bEN = np.linalg.solve(GC + L2 * np.eye(2), rhs)
check(T, "β̂_EN=(0.3958, 0.2958)、両方正（仮定と整合）", allclose(bEN, [.3958, .2958]) and np.all(bEN > 0), r4(bEN))
fEN = lambda b, l1, l2: 0.5 * np.sum((yC - XC @ b) ** 2) + l1 * np.sum(np.abs(b)) + 0.5 * l2 * b @ b


def cd_en(X, y, l1, l2, iters=100000):
    b = np.zeros(X.shape[1]); nrm = (X ** 2).sum(0)
    for _ in range(iters):
        bo = b.copy()
        for j in range(X.shape[1]):
            r = y - X @ b + X[:, j] * b[j]
            zj = X[:, j] @ r
            b[j] = np.sign(zj) * max(abs(zj) - l1, 0) / (nrm[j] + l2)
        if np.max(np.abs(b - bo)) < 1e-14:
            break
    return b


check(T, "座標降下法（目的関数を直接最小化）でも同じ解", allclose(cd_en(XC, yC, L1, L2), bEN, 1e-9), r4(cd_en(XC, yC, L1, L2)))
gridv = np.linspace(-0.5, 1.5, 401)
fg = np.array([[fEN(np.array([a, b]), L1, L2) for b in gridv] for a in gridv])
check(T, "格子探索で目的関数の最小は解より小さくならない", fg.min() >= fEN(bEN, L1, L2) - 1e-12, (fg.min(), fEN(bEN, L1, L2)))
bL10 = cd_en(XC, yC, 10, 0)
check(T, "同じ λ1=10 の Lasso (0.7368, 0.1368)、比 約5.4", allclose(bL10, [.7368, .1368]) and close(bL10[0] / bL10[1], 5.4, .05), (r4(bL10), bL10[0] / bL10[1]))
check(T, "エラスティックネットの比 約1.34", close(bEN[0] / bEN[1], 1.34, 5e-3), bEN[0] / bEN[1])
check(T, "λ2=0 で Lasso、λ1=0 でリッジ（λ=5 → (0.8125,0.7125)）", allclose(cd_en(XC, yC, 0, 5), [.8125, .7125], 1e-9), r4(cd_en(XC, yC, 0, 5)))
# 直交計画の公式
XlC = 0.5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
ylC = np.array([3.0, 0.6, -0.4, -2.2]); ylC = ylC - ylC.mean()
bo_ = XlC.T @ ylC
okorth = True
for l1, l2 in [(1, 1), (0.5, 2), (2, 0.3), (2.5, 4)]:
    form = np.sign(bo_) * np.maximum(np.abs(bo_) - l1, 0) / (1 + l2)
    num = optimize.minimize(lambda b: 0.5 * np.sum((ylC - XlC @ b) ** 2) + l1 * np.sum(np.abs(b)) + 0.5 * l2 * b @ b, np.zeros(3), method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000)).x
    okorth &= np.allclose(form, num, atol=1e-5) and np.allclose(form, cd_en(XlC, ylC, l1, l2), atol=1e-10)
check(T, "直交計画 XᵀX=I で sign(b)(|b|−λ1)₊/(1+λ2)（4組の (λ1,λ2) で数値最小化と一致）", okorth, "ok")
check(T, "chart ridge-lasso-path method=lasso", chart(T, "ridge-lasso-path")["params"]["method"] == "lasso", chart(T, "ridge-lasso-path")["params"])

T = "principal-component-regression"
ev, V = np.linalg.eigh(GC)
ev, V = ev[::-1], V[:, ::-1]
V = V * np.sign(V[0])  # 第1成分を (1,1)/√2、第2成分を (1,−1)/√2 の向きに
check(T, "固有値 19, 1、v1=(1,1)/√2、v2=(1,−1)/√2", allclose(ev, [19, 1], 1e-10) and allclose(V[:, 0], np.ones(2) / sqrt(2), 1e-10) and allclose(V[:, 1], [1 / sqrt(2), -1 / sqrt(2)], 1e-10), ev)
z1 = XC @ V[:, 0]
g1 = z1 @ yC / (z1 @ z1)
check(T, "γ̂1=1.3621（z'y/z'z と v'X'y/λ が一致）", close(g1, 1.3621) and close(g1, V[:, 0] @ gC / ev[0], 1e-12), g1)
bP = g1 * V[:, 0]
check(T, "β̂_PCR=(0.9632, 0.9632)", allclose(bP, [.9632, .9632]), r4(bP))
bP2 = sum((V[:, k] @ gC / ev[k]) * V[:, k] for k in range(2))
check(T, "m=p なら OLS に一致", allclose(bP2, bOC, 1e-12), r4(bP2))
vO = np.linalg.inv(GC)[0, 0]
vP = (np.outer(V[:, 0], V[:, 0]) / ev[0])[0, 0]
check(T, "分散 10/19=0.5263σ²、(1/2)/19=0.0263σ²、比 1/20", close(vO, .5263) and close(vP, .0263) and close(vP / vO, 1 / 20, 1e-12), (vO, vP))
rssO = np.sum((yC - XC @ bOC) ** 2); rssP = np.sum((yC - XC @ bP) ** 2)
check(T, "残差平方和 0.0284 → 0.2084、y の平方和 35.46", close(rssO, .0284) and close(rssP, .2084) and close(yC @ yC, 35.46, 1e-10), (rssO, rssP))
check(T, "OLS の共分散 σ²Σv_kv_kᵀ/λ_k = σ²(XᵀX)⁻¹", np.allclose(sum(np.outer(V[:, k], V[:, k]) / ev[k] for k in range(2)), np.linalg.inv(GC), atol=1e-12), "ok")
check(T, "リッジは各方向を λ_k/(λ_k+λ) 倍（λ=1）", np.allclose(np.linalg.solve(GC + np.eye(2), gC), sum(ev[k] / (ev[k] + 1) * (V[:, k] @ gC / ev[k]) * V[:, k] for k in range(2)), atol=1e-12), "ok")
# 乱数：β=(1,0.8)、σ=0.3 で PCR の分散と偏り
Rp = 200000
Ysim = XC @ np.array([1, .8]) + rngC.normal(0, .3, (Rp, 5))
BP_ = (Ysim @ z1 / (z1 @ z1))[:, None] * V[:, 0][None, :]
check(T, "乱数 PCR の分散≈0.0263σ²（相対 ±2%）、平均≈(0.9,0.9)（偏り、±0.002）", abs(BP_[:, 0].var() / (vP * .09) - 1) < .02 and np.all(np.abs(BP_.mean(0) - .9) < .002), (BP_[:, 0].var() / .09, BP_.mean(0)))

T = "spline-regression"
xs_ = np.arange(1, 9, dtype=float)
ys_ = np.array([2.0, 3.1, 3.9, 5.2, 5.4, 5.9, 6.1, 6.6])
Xsp = np.column_stack([np.ones(8), xs_, np.maximum(xs_ - 4, 0)])
bsp = ols(Xsp, ys_)
check(T, "f̂=0.9636+1.0318x−0.6655(x−4)₊", allclose(bsp, [.9636, 1.0318, -.6655]), r4(bsp))
check(T, "x>4 の傾き 0.3664（=1.0318−0.6655 を丸め前で計算）", close(bsp[1] + bsp[2], .3664), bsp[1] + bsp[2])
rss_sp = np.sum((ys_ - Xsp @ bsp) ** 2)
bl_ = ols(np.column_stack([np.ones(8), xs_]), ys_)
rss_l = np.sum((ys_ - np.column_stack([np.ones(8), xs_]) @ bl_) ** 2)
check(T, "直線 0.6357x+1.9143、RSS 1.2214 → 0.0616", allclose(bl_, [1.9143, .6357]) and close(rss_l, 1.2214) and close(rss_sp, .0616), (r4(bl_), rss_l, rss_sp))
Fsp = (rss_l - rss_sp) / (rss_sp / 5)
check(T, "部分 F=94.08、p=0.0002", close(Fsp, 94.08, 5e-3) and close(stats.f.sf(Fsp, 1, 5), .0002), (Fsp, stats.f.sf(Fsp, 1, 5)))
check(T, "丸めた RSS での F も 94.1", close((1.2214 - .0616) / (.0616 / 5), 94.14, 5e-3), (1.2214 - .0616) / (.0616 / 5))
bq = ols(np.column_stack([np.ones(8), xs_, xs_ ** 2]), ys_)
rss_q = np.sum((ys_ - np.column_stack([np.ones(8), xs_, xs_ ** 2]) @ bq) ** 2)
check(T, "2次式の RSS 0.2155 > 0.0616", close(rss_q, .2155) and rss_q > rss_sp, rss_q)
fh = lambda x: bsp[0] + bsp[1] * x + bsp[2] * np.maximum(x - 4, 0)
check(T, "x=4 で連続（左右極限の差 0）", close(fh(4 - 1e-9), fh(4 + 1e-9), 1e-8), (fh(4 - 1e-9), fh(4 + 1e-9)))
# 3次の切断べき基底は 2 階微分まで連続、3 階微分が θ だけ跳ぶ（sympy）
xx_, kk_, th_ = sp.symbols("x kappa theta", real=True)
bas = th_ * (xx_ - kk_) ** 3
okc = all(sp.simplify(sp.diff(bas, xx_, r).subs(xx_, kk_)) == 0 for r in range(3)) and sp.diff(bas, xx_, 3) == 6 * th_
check(T, "θ(x−κ)³₊：κ で 0〜2 階微分が 0 から連続、3 階微分が 6θ 跳ぶ（d!θ）", okc, "ok")
check(T, "パラメータ数 d+1+K（3次・K=2 なら 6、切断べき基底の列数）", np.linalg.matrix_rank(np.column_stack([np.ones(20), *[np.linspace(0, 10, 20) ** p for p in (1, 2, 3)], *[np.maximum(np.linspace(0, 10, 20) - k, 0) ** 3 for k in (3, 7)]])) == 6, 6)
c = chart(T, "line-data")
check(T, "chart x=1..8、観測値と当てはめ値", c["params"]["x"] == list(range(1, 9)) and allclose(series(c, 0), ys_, 1e-12) and allclose(series(c, 1), Xsp @ bsp), r4(Xsp @ bsp))

# =====================================================================
ok = sum(results); ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")

# --- 追加用語（用語の見直し）: neural-network-model ---
import re as _re, yaml as _yaml
from pathlib import Path as _Path
T = "neural-network-model"
_fmN = _yaml.safe_load(_re.match(r"^---\n(.*?)\n---\n", (_Path(__file__).resolve().parent.parent / "content/ch19/neural-network-model.md").read_text(encoding="utf-8"), _re.S).group(1))
_x = np.array([1.0, 2.0])
_W = np.array([[0.5, -0.3], [-0.4, 0.6]]); _b = np.array([0.2, -0.1]); _v = np.array([1.0, -2.0]); _c0 = 0.5
_h = np.maximum(0, _W @ _x + _b); _u = _v @ _h + _c0; _f = 1 / (1 + exp(-_u))
check(T, "例 h1=0.1, h2=0.7, u=−0.8, f=1/(1+e^0.8)≈0.3100", allclose(_h, [0.1, 0.7], 1e-12) and close(_u, -0.8, 1e-12) and abs(_f - 0.3100) <= 5e-5, (_h.tolist(), _u, _f))
check(T, "パラメータ数 H(p+1)+(H+1)：p=10, H=5 → 61", 5 * (10 + 1) + (5 + 1) == 61, 61)
# 中間層なし＋シグモイド出力＝ロジスティック回帰（同じ式）
_xx = rng.normal(size=(50, 3)); _vv = rng.normal(size=3)
check(T, "中間層なし・シグモイド出力はロジスティック回帰の確率と一致", allclose(1 / (1 + np.exp(-(_xx @ _vv + 0.3))), stats.logistic.cdf(_xx @ _vv + 0.3), 1e-12), "")
# 活性化が恒等なら2層でも線形
_A1 = rng.normal(size=(4, 3)); _A2 = rng.normal(size=(1, 4))
check(T, "恒等活性化の2層＝1つの線形写像 A2A1", allclose((_A2 @ (_A1 @ _xx.T)).ravel(), (_xx @ (_A2 @ _A1).T).ravel(), 1e-10), "")
# 誤差逆伝播の勾配が数値微分と一致（二乗誤差, 例の重みで y=1）
def _loss(par):
    W = par[:4].reshape(2, 2); b = par[4:6]; v = par[6:8]; c = par[8]
    h = np.maximum(0, W @ _x + b); return 0.5 * (1 / (1 + exp(-(v @ h + c))) - 1.0) ** 2
_p0 = np.concatenate([_W.ravel(), _b, _v, [_c0]])
_d = (_f - 1.0) * _f * (1 - _f)
_gv = _d * _h; _gc = _d; _gh = _d * _v * (_W @ _x + _b > 0); _gW = np.outer(_gh, _x); _gb = _gh
_g = np.concatenate([_gW.ravel(), _gb, _gv, [_gc]])
_num = np.array([(_loss(_p0 + 1e-6 * e) - _loss(_p0 - 1e-6 * e)) / 2e-6 for e in np.eye(9)])
check(T, "誤差逆伝播（連鎖律）の勾配＝数値微分", allclose(_g, _num, 1e-8), r4(_g))
_cN = _fmN["charts"][0]["params"]
check(T, "chart：辺の重みが例の重みと一致", _cN["edges"] == [[0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [3, 4]] and _cN["weights"] == [0.5, -0.4, -0.3, 0.6, 1.0, -2.0] and [_W[0, 0], _W[1, 0], _W[0, 1], _W[1, 1], _v[0], _v[1]] == _cN["weights"], "")

ok = sum(results); ng = len(results) - ok
print(f"\n合計（追加用語を含む）: OK {ok} / NG {ng}")
