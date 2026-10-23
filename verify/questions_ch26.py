"""第26章（その他の多変量解析手法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats
from scipy.optimize import isotonic_regression
from scipy.spatial.distance import pdist, squareform

q = Q(26)
rng = np.random.default_rng(26)


def cmds_B(D):
    n = D.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D ** 2) @ J


def cca(R11, R12, R22):
    M = np.linalg.solve(R11, R12) @ np.linalg.solve(R22, R12.T)
    return np.sqrt(np.sort(np.clip(np.linalg.eigvals(M).real, 0, None))[::-1])

# --- MDS ---
P = rng.normal(size=(8, 3))
ev = np.linalg.eigvalsh(cmds_B(squareform(pdist(P))))
q.check("q26-multidimensional-scaling-1", ev.min() > -1e-9, "ユークリッド距離なら B の固有値 ≥ 0（正しい）")
D3 = np.array([[0, 3, 5], [3, 0, 2], [5, 2, 0.]])
ev3 = np.sort(np.linalg.eigvalsh(cmds_B(D3)))[::-1]
q.num("q26-multidimensional-scaling-2", ev3[0])
q.check("q26-multidimensional-scaling-2", np.allclose(ev3[1:], 0, atol=1e-9), "残りの固有値は0")
lam = np.array([8, 3, 1, 0.])
q.num("q26-multidimensional-scaling-3", lam[:2].sum() / lam[lam > 0].sum())

# --- 古典的MDS ---
pts = np.array([[0, 0], [4, 0], [0, 3.]])
B = cmds_B(squareform(pdist(pts)))
q.num("q26-classical-mds-1", B[0, 0])
c = pts - pts.mean(0)
q.check("q26-classical-mds-1", np.allclose(B, c @ c.T), "B＝中心化座標の内積行列")

# --- 非計量MDS ---
def stress(delta, X):
    d = pdist(X); order = np.argsort(delta, kind="stable")
    dh = np.empty_like(d); dh[order] = isotonic_regression(d[order]).x
    return np.sqrt(((d - dh) ** 2).sum() / (d ** 2).sum())
delta = pdist(rng.normal(size=(7, 2))) + rng.uniform(0, .3, 21)
X = rng.normal(size=(7, 2))
s1, s2, s3 = stress(delta, X), stress(delta ** 2, X), stress(np.log(delta), X)
q.check("q26-nonmetric-mds-1", abs(s1 - s2) < 1e-12 and abs(s1 - s3) < 1e-12 and s1 > 0, f"ストレス {s1:.4f} は単調変換で不変（正しい）")

# --- 正準相関分析 ---
x = rng.normal(size=1000); y = -0.6 * x + 0.8 * rng.normal(size=1000)
r = np.corrcoef(x, y)[0, 1]
rho = cca(np.array([[1.]]), np.array([[r]]), np.array([[1.]]))[0]
q.check("q26-canonical-correlation-analysis-1", r < 0 and abs(rho - abs(r)) < 1e-12, "r<0 でも ρ=|r|（記述は誤り）")
R12 = np.array([[0.6], [0.3]])
q.num("q26-canonical-correlation-analysis-2", cca(np.eye(2), R12, np.array([[1.]]))[0])
R12b = np.array([[.5, .3], [.3, .5]])
rr = cca(np.eye(2), R12b, np.eye(2))
q.num("q26-canonical-correlation-analysis-3", rr[0])
q.check("q26-canonical-correlation-analysis-3", np.allclose(rr, [.8, .2]) and np.allclose(np.linalg.svd(R12b, compute_uv=False), [.8, .2]), "ρ=(0.8,0.2)＝特異値")

# --- 正準変量 ---
n = 200000
Z = rng.normal(size=(n, 5)) @ rng.normal(size=(5, 5))
Xc, Yc = Z[:, :2], Z[:, 2:4] + 0.5 * Z[:, :2]
S = np.cov(np.c_[Xc, Yc], rowvar=False)
S11, S12, S22 = S[:2, :2], S[:2, 2:], S[2:, 2:]
w, A = np.linalg.eig(np.linalg.solve(S11, S12) @ np.linalg.solve(S22, S12.T))
idx = np.argsort(w.real)[::-1]; w = w.real[idx]; A = A.real[:, idx]
Bv = np.linalg.solve(S22, S12.T) @ A
U = Xc @ A; V = Yc @ Bv
Cm = np.corrcoef(np.c_[U, V], rowvar=False)
q.check("q26-canonical-variate-1", abs(Cm[0, 3]) < 1e-8 and abs(Cm[1, 2]) < 1e-8 and abs(Cm[0, 1]) < 1e-8
        and abs(abs(Cm[0, 2]) - np.sqrt(w[0])) < 1e-8, "Corr(U1,V2)=0 ほか（正しい）")
# 交差負荷量 = ρ × 正準負荷量（標本で確認）
cx_u = np.corrcoef(Xc[:, 0], U[:, 0])[0, 1]; cx_v = np.corrcoef(Xc[:, 0], V[:, 0])[0, 1]
q.check("q26-canonical-variate-2", abs(abs(cx_v) - np.sqrt(w[0]) * abs(cx_u)) < 1e-8, "交差負荷量＝ρ×正準負荷量")
q.num("q26-canonical-variate-2", 0.6 * 0.9)
chi = -(100 - 1 - (2 + 2 + 1) / 2) * (np.log(1 - .25) + np.log(1 - .01))
q.num("q26-canonical-variate-3", chi)
q.check("q26-canonical-variate-3", chi > stats.chi2.ppf(.95, 4) and abs(stats.chi2.ppf(.95, 4) - 9.49) < 5e-3, "df=4 の5%点9.49を超える")

# --- コレスポンデンス分析 ---
q.num("q26-correspondence-analysis-1", 24 / 400)
q.num("q26-correspondence-analysis-2", 0.045 / 0.06)
q.check("q26-correspondence-analysis-2", min(4, 3) - 1 == 2, "軸は2本")
N = np.array([[30, 10], [20, 40.]])
Pm = N / N.sum(); rr_ = Pm.sum(1); cc_ = Pm.sum(0)
Sres = (Pm - np.outer(rr_, cc_)) / np.sqrt(np.outer(rr_, cc_))
g = np.linalg.svd(Sres, compute_uv=False)
chi2 = stats.chi2_contingency(N, correction=False)[0]
q.num("q26-correspondence-analysis-3", g[0])
phi = (30 * 40 - 10 * 20) / np.sqrt(40 * 60 * 50 * 50)
q.check("q26-correspondence-analysis-3", abs(g[0] ** 2 - chi2 / 100) < 1e-12 and abs(g[1]) < 1e-12 and abs(g[0] - abs(phi)) < 1e-12,
        "γ1²=χ²/n、軸は1本、|φ| と一致")

# --- 数量化理論 ---
q.concept("q26-quantification-theory-1", "外的基準が質的→II類（分類の知識）")
q.concept("q26-quantification-theory-2", "III類は外的基準なし（分類の知識）")

# --- 数量化I類 ---
yA = np.array([10, 12, 14.]); yB = np.array([20, 22.])
Xd = np.c_[np.ones(5), np.r_[np.zeros(3), np.ones(2)]]
beta = np.linalg.lstsq(Xd, np.r_[yA, yB], rcond=None)[0]   # 基準カテゴリー A をスコア0
sA, sB = 0.0, beta[1]
shift = (3 * sA + 2 * sB) / 5
q.num("q26-quantification-type-1-1", sB - shift)
q.check("q26-quantification-type-1-1", abs(beta[0] + shift - np.r_[yA, yB].mean()) < 1e-12 and abs(3 * (sA - shift) + 2 * (sB - shift)) < 1e-12,
        "基準化後の定数項＝ȳ、重み付き平均0")
q.num("q26-quantification-type-1-2", 1.4 - (-2.1))

# --- 数量化III類 ---
def H_of(F):
    r = F.sum(1); c = F.sum(0)
    return np.diag(c ** -.5) @ F.T @ np.diag(1 / r) @ F @ np.diag(c ** -.5), c
Fr = (rng.uniform(size=(10, 5)) < .5).astype(float); Fr[Fr.sum(1) == 0, 0] = 1; Fr[0] = 1
Hr, cr = H_of(Fr)
evr = np.sort(np.linalg.eigvalsh(Hr))[::-1]
q.check("q26-quantification-type-3-1", abs(evr[0] - 1) < 1e-10 and np.allclose(Hr @ np.sqrt(cr), np.sqrt(cr)), "最大固有値1＝自明解（正しい）")
F = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1.]])
H, _ = H_of(F)
evF = np.sort(np.linalg.eigvalsh(H))[::-1]
q.num("q26-quantification-type-3-2", evF[1])
q.check("q26-quantification-type-3-2", abs(evF[0] - 1) < 1e-12 and abs(evF[1] * evF[2] - 1 / 24) < 1e-12 and abs(np.trace(H) - 1.75) < 1e-12
        and abs(evF[2] - 0.060) < 5e-4, "トレース1.75、積1/24、λ3=0.060")

# --- 多重コレスポンデンス分析 ---
cats = [3, 3, 2, 4]; nresp = 50
Zs = np.hstack([np.eye(k)[rng.integers(0, k, nresp)] for k in cats])
Zs = Zs[:, Zs.sum(0) > 0]
Pz = Zs / Zs.sum(); rz = Pz.sum(1); cz = Pz.sum(0)
inertia = (((Pz - np.outer(rz, cz)) ** 2) / np.outer(rz, cz)).sum()
q.num("q26-multiple-correspondence-analysis-1", inertia)
q.check("q26-multiple-correspondence-analysis-1", Zs.shape[1] == 12, "全カテゴリーが出現（J=12）")

# --- SEM ---
q.num("q26-structural-equation-modeling-1", 6 * 7 // 2 - 13)
q.check("q26-structural-equation-modeling-2", 3 * 4 // 2 - 6 == 0, "df=0（ちょうど識別）")
lam_ = np.array([1, .8])
q.num("q26-structural-equation-modeling-3", (np.outer(lam_, lam_) * 2)[0, 1])

# --- パス解析 ---
q.num("q26-path-analysis-1", 0.3 + 0.5 * 0.4)
R = np.array([[1, .5], [.5, 1]]); rvec = np.array([.45, .5])   # 説明変数 (x, m) と y の相関
cb = np.linalg.solve(R, rvec)   # (c', b)
q.num("q26-path-analysis-2", 0.5 * cb[1])
q.check("q26-path-analysis-2", abs(cb[0] + 0.5 * cb[1] - 0.45) < 1e-12 and abs(cb[0] - 4 / 15) < 1e-12 and abs(cb[1] - 11 / 30) < 1e-12,
        "c'=4/15、b=11/30、総合効果＝r_xy")
q.num("q26-path-analysis-3", 1 - cb @ rvec)
# シミュレーションで残差分散を確認
n = 300000
xs = rng.normal(size=n); ms = 0.5 * xs + np.sqrt(0.75) * rng.normal(size=n)
ys = cb[0] * xs + cb[1] * ms + np.sqrt(1 - cb @ rvec) * rng.normal(size=n)
q.check("q26-path-analysis-3", abs(np.corrcoef(xs, ys)[0, 1] - .45) < .005 and abs(np.corrcoef(ms, ys)[0, 1] - .5) < .005 and abs(ys.var() - 1) < .01,
        "そのモデルで相関 0.45, 0.5、分散1を再現")

# --- 独立成分分析 ---
n = 200000
Sg = rng.normal(size=(n, 2)); Su = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, 2))
kurt = lambda v: stats.kurtosis(v)
th = np.linspace(0, np.pi / 2, 7)
kg = [abs(kurt(Sg @ np.array([np.cos(t), np.sin(t)]))) for t in th]
ku = [kurt(Su @ np.array([np.cos(t), np.sin(t)])) for t in th]
q.check("q26-independent-component-analysis-1", max(kg) < 0.03 and ku[0] < -1.1 and ku[3] > -0.7,
        "正規なら回転しても尖度0で手がかりなし、一様なら回転で尖度が変わる（記述は誤り）")

# --- 適合度指標 ---
q.num("q26-goodness-of-fit-indices-1", np.sqrt(max(30 - 8, 0) / (8 * 300)))

# --- 導出問題（追加） ---
def _deriv26():
    import numpy as _N
    rng = _N.random.default_rng(26)
    n = 7
    X = rng.normal(size=(n, 2)); X -= X.mean(0)
    B = X @ X.T
    c = (X ** 2).sum(1)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    one = _N.ones(n); J = _N.eye(n) - _N.outer(one, one) / n
    ok = _N.allclose(D2, _N.outer(c, one) + _N.outer(one, c) - 2 * B)
    ok &= _N.allclose(J @ one, 0) and _N.allclose(J @ X, X)
    Bh = -0.5 * J @ D2 @ J
    ok &= _N.allclose(Bh, B)
    w, V = _N.linalg.eigh(Bh); idx = _N.argsort(w)[::-1]; w, V = w[idx], V[:, idx]
    Xr = V[:, :2] * _N.sqrt(w[:2])
    D2r = ((Xr[:, None, :] - Xr[None, :, :]) ** 2).sum(-1)
    ok &= _N.allclose(D2r, D2) and _N.all(w[2:] > -1e-10) and _N.allclose(w[2:], 0, atol=1e-10)
    q.check("q26-multidimensional-scaling-4", bool(ok), "D²=c1'+1c'-2B、-½JD²J=B、固有値分解の配置で距離を再現")
_deriv26()

# --- 4択の追加（パス解析・コレスポンデンス分析・正準相関）---
import numpy as _np
_a1, _b1, _a2, _b2, _c = 0.5, 0.4, 0.3, 0.6, 0.1
# 共分散を構造方程式から計算（x の分散1、m1,m2 の分散も1になるよう誤差分散を設定）
_cov_x_m1, _cov_x_m2, _cov_m1m2 = _a1, _a2, _a1 * _a2
_cov_xy = _c + _b1 * _cov_x_m1 + _b2 * _cov_x_m2
_var_y_expl = _c**2 + _b1**2 + _b2**2 + 2 * (_c * _b1 * _a1 + _c * _b2 * _a2 + _b1 * _b2 * _cov_m1m2)
_ind = _a1 * _b1 + _a2 * _b2
_ok = abs(_ind - 0.38) < 1e-12 and abs(_cov_xy - 0.48) < 1e-12 and _var_y_expl < 1
_wr = [(_a1 * _b1, _c + _a1 * _b1), (_cov_xy, _ind), (_a1 + _b1 + _a2 + _b2, _a1 + _b1 + _a2 + _b2 + _c)]
_ok &= [tuple(round(v, 2) for v in w) for w in _wr] == [(0.2, 0.3), (0.48, 0.38), (1.8, 1.9)]
q.check("q26-path-analysis-4", _ok, f"間接 {_ind:.2f}、総合 {_cov_xy:.2f}＝r_xy（説明される y の分散 {_var_y_expl:.3f}<1 で実現可能）")
# コレスポンデンス分析：全イナーシャ＝χ²/n を任意の表で確認し、数値を計算
_rng = _np.random.default_rng(260)
_N = _rng.integers(5, 40, size=(3, 4)).astype(float)
_P = _N / _N.sum(); _r = _P.sum(1); _cc = _P.sum(0)
_Sres = (_P - _np.outer(_r, _cc)) / _np.sqrt(_np.outer(_r, _cc))
_gam = _np.linalg.svd(_Sres, compute_uv=False)
_E = _np.outer(_N.sum(1), _N.sum(0)) / _N.sum()
_chi = ((_N - _E) ** 2 / _E).sum()
_ok = abs((_gam ** 2).sum() - _chi / _N.sum()) < 1e-12 and (_gam > 1e-10).sum() <= 2
_tot = 30 / 200; _con = 0.3 ** 2 / _tot
_ok &= round(_tot, 2) == 0.15 and round(_con, 2) == 0.60 and _tot - 0.09 >= 0 and 0.06 <= 0.09
_ok &= round(0.3 / _np.sqrt(_tot), 2) == 0.77 and round(0.3 ** 2, 2) == 0.09 and 30 != _tot
q.check("q26-correspondence-analysis-4", _ok, f"Σγ²=χ²/n を確認、全イナーシャ {_tot}、寄与率 {_con:.2f}、誤答 0.77・0.09・30")
# 正準相関：乱数データで性質を確認
_n = 400
_Z = _rng.standard_normal((_n, 8)) @ _rng.standard_normal((8, 8))
_Xd, _Yd = _Z[:, :3], _Z[:, 3:]
def _cca(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    S11, S22, S12 = X.T @ X, Y.T @ Y, X.T @ Y
    M = _np.linalg.inv(S11) @ S12 @ _np.linalg.inv(S22) @ S12.T
    return _np.sqrt(_np.sort(_np.clip(_np.linalg.eigvals(M).real, 0, None))[::-1])
_rho = _cca(_Xd, _Yd)
_R = _np.corrcoef(_Z.T)[:3, 3:]
_ok = _rho[0] >= _np.abs(_R).max() - 1e-12 and len(_rho) == 3
_rho_s = _cca(_Xd * _np.array([100, 0.1, 3]), _Yd * _np.array([2, 5, 0.01, 7, 1]))
_ok &= _np.allclose(_rho, _rho_s)
_X1c = _np.column_stack([_np.ones(_n), _Xd])
_beta = _np.linalg.lstsq(_X1c, _Yd[:, 0], rcond=None)[0]
_R2 = 1 - ((_Yd[:, 0] - _X1c @ _beta) ** 2).sum() / ((_Yd[:, 0] - _Yd[:, 0].mean()) ** 2).sum()
_ok &= _R2 < _rho[0] ** 2 - 1e-6
q.check("q26-canonical-correlation-analysis-4", _ok,
        f"ρ1={_rho[0]:.4f} ≥ max|r|={_np.abs(_R).max():.4f}、組は3、尺度不変、R²(y1)={_R2:.4f}<ρ1²={_rho[0]**2:.4f}")

q.done()
