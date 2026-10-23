"""第25章（因子分析・グラフィカルモデル）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from itertools import product
from scipy import stats

q = Q(25)
rng = np.random.default_rng(25)


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def pcorr(S):
    O = np.linalg.inv(S); d = np.sqrt(np.diag(O))
    P = -O / np.outer(d, d); np.fill_diagonal(P, 1)
    return P

# --- 因子分析モデル ---
lam = np.array([0.9, 0.8, 0.6])
Sig = np.outer(lam, lam) + np.diag(1 - lam ** 2)
q.num("q25-factor-analysis-model-1", Sig[0, 2])
# シミュレーションでも：x = λ f + ε
n = 400000
f = rng.normal(size=n); E = rng.normal(size=(n, 3)) * np.sqrt(1 - lam ** 2)
Xs = f[:, None] * lam + E
q.check("q25-factor-analysis-model-1", abs(np.corrcoef(Xs[:, 0], Xs[:, 2])[0, 1] - 0.54) < 0.005, "シミュレーションでも0.54")
q.num("q25-factor-analysis-model-2", 0.5 ** 2 + 0.4 ** 2 + 0.3)
r12, r13, r23 = 0.56, 0.48, 0.42
l1 = np.sqrt(r12 * r13 / r23)
q.num("q25-factor-analysis-model-3", l1)
L = np.array([l1, r12 / l1, r13 / l1])
q.check("q25-factor-analysis-model-3", np.allclose(L, [0.8, 0.7, 0.6]) and abs(L[1] * L[2] - r23) < 1e-12, "λ=(0.8,0.7,0.6) で3つの相関を再現")

# --- 共通因子と独自因子 ---
Lam = rng.normal(size=(5, 2)); Psi = np.diag(rng.uniform(.2, 1, 5))
F = rng.normal(size=(300000, 2)); Eps = rng.normal(size=(300000, 5)) * np.sqrt(np.diag(Psi))
Xs = F @ Lam.T + Eps
C = np.cov(Xs, rowvar=False); LL = Lam @ Lam.T
off = ~np.eye(5, dtype=bool)
q.check("q25-common-factor-unique-factor-1", np.max(np.abs(C[off] - LL[off])) < 0.05, "非対角の共分散＝ΛΛᵀ（正しい）")
q.num("q25-common-factor-unique-factor-2", 1 - 0.25)
q.check("q25-common-factor-unique-factor-2", abs(0.6 + 0.15 + 0.25 - 1) < 1e-12, "分解の合計が1")

# --- 因子負荷量 ---
q.num("q25-factor-loading-1", 1.2 / np.sqrt(4))
lrow = np.array([0.7, 0.1]); Phi = np.array([[1, .4], [.4, 1]])
cands = [lrow @ Phi, lrow, lrow @ np.linalg.inv(Phi), (lrow @ Phi)[::-1]]
want = [(0.74, 0.38), (0.70, 0.10), (0.79, -0.21), (0.38, 0.74)]
q.check("q25-factor-loading-2", all(np.allclose(np.round(c, 2), w) for c, w in zip(cands, want)), "正解 ΛΦ=(0.74,0.38)、誤答は Λ・ΛΦ⁻¹・逆順")
# シミュレーション：斜交因子で x1 と因子の相関
n = 400000
Fo = rng.multivariate_normal([0, 0], Phi, n)
h2 = lrow @ Phi @ lrow
x1 = Fo @ lrow + rng.normal(size=n) * np.sqrt(1 - h2)
q.check("q25-factor-loading-2", np.allclose([np.corrcoef(x1, Fo[:, k])[0, 1] for k in range(2)], [0.74, 0.38], atol=0.005), "シミュレーション一致")
q.num("q25-factor-loading-3", np.array([.8, .3]) @ np.array([.6, .5]))

# --- 共通性 ---
q.num("q25-communality-1", 1 - (0.36 + 0.25))
q.num("q25-communality-2", h2)
Lam = rng.normal(size=(6, 2)); T = rot(0.7)
q.check("q25-communality-3", np.allclose(np.sum((Lam @ T) ** 2, 1), np.sum(Lam ** 2, 1)), "直交回転で共通性は不変（正しい）")

# --- 因子回転 ---
Psi = np.diag(rng.uniform(.2, .8, 6))
q.check("q25-factor-rotation-1", np.allclose((Lam @ T) @ (Lam @ T).T + Psi, Lam @ Lam.T + Psi)
        and not np.allclose(Lam @ T, Lam), "負荷量は変わるが Σ は不変（記述は誤り）")
r = np.array([.8, .4]) @ rot(np.pi / 6)
q.num("q25-factor-rotation-2", r[0])
q.check("q25-factor-rotation-2", abs(r[1] + 0.054) < 5e-4 and abs(r @ r - 0.8) < 1e-12, "第2成分 -0.054、2乗和0.80")
col0 = (Lam ** 2).sum(0); col1 = ((Lam @ T) ** 2).sum(0)
q.check("q25-factor-rotation-3", abs(col0.sum() - col1.sum()) < 1e-12 and not np.allclose(col0, col1), "総和不変・列ごとは変化（正しい）")

# --- バリマックス ---
def varimax_crit(L):
    L2 = L ** 2
    return np.sum(np.mean(L2 ** 2, 0) - np.mean(L2, 0) ** 2)
Lv = np.array([[.8, 0], [.7, 0], [0, .6], [0, .5]])
q.num("q25-varimax-rotation-1", varimax_crit(Lv))
# 単純構造は回転すると基準が下がる（この Λ は局所的に最大）
q.check("q25-varimax-rotation-1", all(varimax_crit(Lv @ rot(t)) < varimax_crit(Lv) for t in np.linspace(0.05, 0.7, 14)),
        "回転すると V は小さくなる")
# バリマックスは直交回転：因子の共分散 TᵀIT=I のまま
q.check("q25-varimax-rotation-2", np.allclose(rot(0.4).T @ rot(0.4), np.eye(2)), "直交行列で因子は無相関のまま（記述は誤り）")

# --- プロマックス ---
q.num("q25-promax-rotation-1", -0.6 * abs(-0.6) ** 2)
A = rng.normal(size=(6, 2)) * 0.5
P = A * np.abs(A) ** 2
U0 = np.linalg.solve(A.T @ A, A.T @ P)
d = np.sqrt(np.diag(np.linalg.inv(U0.T @ U0)))
U = U0 @ np.diag(d)
Phi_p = np.linalg.inv(U.T @ U); Lp = A @ U
q.check("q25-promax-rotation-2", np.allclose(np.diag(Phi_p), 1) and abs(Phi_p[0, 1]) > 1e-3
        and np.allclose(np.diag(Lp @ Phi_p @ Lp.T), np.sum(A ** 2, 1)), "因子相関は0でないが共通性は不変（記述は誤り）")

# --- 因子数 ---
dfm = lambda p, m: ((p - m) ** 2 - (p + m)) / 2
q.num("q25-number-of-factors-1", max(m for m in range(1, 5) if dfm(5, m) >= 0))
pv = {1: 0.001, 2: 0.02, 3: 0.31, 4: 0.64}
q.check("q25-number-of-factors-2", min(m for m in pv if pv[m] >= 0.05) == 3, "初めて棄却されない m=3")

# --- 因子得点 ---
Lf = np.array([.8, .6]); Pf = np.diag([.36, .64]); xv = np.array([1, .5])
Sf = np.outer(Lf, Lf) + Pf
fR = Lf @ np.linalg.solve(Sf, xv)
c = Lf @ np.linalg.solve(Pf, Lf); b = Lf @ np.linalg.solve(Pf, xv)
q.num("q25-factor-score-1", fR)
q.check("q25-factor-score-1", abs(fR - b / (1 + c)) < 1e-12 and abs(b / c - 1.150) < 5e-4, "2つの公式が一致、バートレット 1.150")
ok = True
for _ in range(100):
    Lr = rng.uniform(.2, .9, 4); Pr = np.diag(1 - Lr ** 2); xr = rng.normal(size=4)
    cr = Lr @ np.linalg.solve(Pr, Lr); br = Lr @ np.linalg.solve(Pr, xr)
    fr = Lr @ np.linalg.solve(np.outer(Lr, Lr) + Pr, xr)
    ok &= abs(fr - cr / (1 + cr) * (br / cr)) < 1e-10 and abs(fr) < abs(br / cr)
q.check("q25-factor-score-2", ok, "回帰法＝c/(1+c)×バートレット、絶対値が小さい（正しい）")

# --- 最尤法 ---
nn, p, m = 200, 6, 2
Tst = (nn - 1 - (2 * p + 5) / 6 - 2 * m / 3) * 0.05
q.num("q25-maximum-likelihood-factor-analysis-1", Tst)
q.check("q25-maximum-likelihood-factor-analysis-1", dfm(6, 2) == 4 and Tst > stats.chi2.ppf(.95, 4)
        and abs(stats.chi2.ppf(.95, 4) - 9.49) < 5e-3 and abs(stats.chi2.sf(Tst, 4) - 0.045) < 1e-3, "df=4、9.49 を超え p≈0.045")
# ヘイウッドケース：1因子モデルで r12 r13 / r23 > 1 となる相関行列（正定値）では λ1² > 1、ψ1 < 0
Rh = np.array([[1, .8, .7], [.8, 1, .5], [.7, .5, 1]])
q.check("q25-maximum-likelihood-factor-analysis-2", np.all(np.linalg.eigvalsh(Rh) > 0) and 1 - Rh[0, 1] * Rh[0, 2] / Rh[1, 2] < 0,
        "正定値な相関行列でも1因子解の ψ1 が負になる例（正しい）")

# --- グラフィカルモデル ---
O = np.array([[1, -.4, 0], [-.4, 1.2, -.4], [0, -.4, 1]])
S = np.linalg.inv(O)
q.check("q25-graphical-model-1", O[0, 2] == 0 and abs(S[0, 2]) > 0.05, "ω13=0 でも σ13≠0（記述は誤り）")
Pc = pcorr(S)
def pcorr_given(S, i, j, given):
    idx = [i, j] + list(given); Ss = S[np.ix_(idx, idx)]
    return pcorr(Ss)[0, 1]
q.check("q25-graphical-model-2", abs(pcorr_given(S, 0, 2, [1])) < 1e-12 and abs(S[0, 2]) > 0.05
        and abs(pcorr_given(S, 0, 1, [2])) > 0.05 and abs(pcorr_given(S, 1, 2, [0])) > 0.05, "成り立つのは X1⊥X3|X2 のみ")

# --- 無向グラフィカルモデル ---
O4 = np.eye(4) * 2
for i, j in [(0, 1), (1, 2), (2, 3), (1, 3)]:
    O4[i, j] = O4[j, i] = -0.5
S4 = np.linalg.inv(O4)
# X1 ⊥ (X3,X4) | X2：条件付き共分散 S_{1,34} - S_{1,2} S_22^{-1} S_{2,34} = 0
cond = S4[0, [2, 3]] - S4[0, 1] * S4[1, [2, 3]] / S4[1, 1]
q.check("q25-undirected-graphical-model-1", np.all(np.linalg.eigvalsh(O4) > 0) and np.allclose(cond, 0)
        and abs(pcorr_given(S4, 0, 2, [3])) > 1e-3 and abs(pcorr_given(S4, 2, 3, [1])) > 1e-3 and abs(S4[0, 3]) > 1e-3,
        "正解のみ成立、他は不成立")
q.num("q25-undirected-graphical-model-2", int(np.sum(np.triu(np.abs(O4) < 1e-12, 1))))

# --- 精度行列 ---
q.num("q25-precision-matrix-1", -(-1) / np.sqrt(2 * 2))
Sa = np.array([[1, .5, .25], [.5, 1, .5], [.25, .5, 1]])
Oa = np.linalg.inv(Sa)
q.num("q25-precision-matrix-2", pcorr(Sa)[0, 1])
q.check("q25-precision-matrix-2", np.allclose(Oa, np.array([[1, -.5, 0], [-.5, 1.25, -.5], [0, -.5, 1]]) / 0.75)
        and abs((0.5 - 0.125) / np.sqrt(0.9375 * 0.75) - pcorr(Sa)[0, 1]) < 1e-12, "Ω の形と公式が一致")
cv = Sa[1, 1] - Sa[1, [0, 2]] @ np.linalg.solve(Sa[np.ix_([0, 2], [0, 2])], Sa[[0, 2], 1])
q.num("q25-precision-matrix-3", cv)
q.check("q25-precision-matrix-3", abs(cv - 1 / Oa[1, 1]) < 1e-12, "1/ω22 と一致")

# --- 共分散選択 ---
q.num("q25-covariance-selection-1", 100 * np.log(1.04))
q.check("q25-covariance-selection-1", 100 * np.log(1.04) < stats.chi2.ppf(.95, 2) and abs(stats.chi2.ppf(.95, 2) - 5.99) < 5e-3, "5.99 未満で棄却されない")
s13 = 0.6 * 0.5
Sh = np.array([[1, .6, s13], [.6, 1, .5], [s13, .5, 1]])
q.num("q25-covariance-selection-2", s13)
q.check("q25-covariance-selection-2", abs(np.linalg.inv(Sh)[0, 2]) < 1e-12 and np.all(np.linalg.eigvalsh(Sh) > 0), "Σ̂⁻¹ の (1,3) が0")

# --- DAG ---
# 2値変数で因子分解のパラメータ数を数え、パラメータからつくった分布が確率になることを確認
npar = 1 + 2 + 2 + 4
pa, pb, pc, pd = rng.uniform(size=1), rng.uniform(size=2), rng.uniform(size=2), rng.uniform(size=(2, 2))
tot = 0
for a, b, c_, d_ in product([0, 1], repeat=4):
    fa = pa[0] if a else 1 - pa[0]; fb = pb[a] if b else 1 - pb[a]
    fc = pc[a] if c_ else 1 - pc[a]; fd = pd[b, c_] if d_ else 1 - pd[b, c_]
    tot += fa * fb * fc * fd
q.num("q25-directed-acyclic-graph-1", npar)
q.check("q25-directed-acyclic-graph-1", abs(tot - 1) < 1e-12 and 2 ** 4 - 1 == 15, "9個のパラメータで分布が決まる、飽和は15")
n = 400000
Xc = rng.normal(size=n); Yc = rng.normal(size=n); Zc = Xc + Yc + rng.normal(size=n)
Sc = np.cov(np.c_[Xc, Yc, Zc], rowvar=False)
q.check("q25-directed-acyclic-graph-2", abs(np.corrcoef(Xc, Yc)[0, 1]) < 0.01 and pcorr_given(Sc, 0, 1, [2]) < -0.4,
        "周辺では独立、Z を与えると偏相関 -0.5")
B = np.array([[0, 0, 0], [.5, 0, 0], [0, .4, 0]])
IB = np.linalg.inv(np.eye(3) - B)
Sd = IB @ IB.T
q.num("q25-directed-acyclic-graph-3", Sd[0, 2] / np.sqrt(Sd[0, 0] * Sd[2, 2]))
q.check("q25-directed-acyclic-graph-3", np.allclose(np.diag(Sd), [1, 1.25, 1.2]) and abs(pcorr(Sd)[0, 2]) < 1e-12, "分散 1,1.25,1.2、X2 を与えると独立")

# --- グラフィカル Lasso ---
Xg = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 4))
Sg = np.cov(Xg, rowvar=False)
lam_ = np.max(np.abs(Sg[~np.eye(4, dtype=bool)]))
Od = np.diag(1 / np.diag(Sg))
G = np.linalg.inv(Od) - Sg   # 非対角は -s_jk、対角は0
gam = G / lam_
ok_kkt = np.allclose(np.diag(G), 0) and np.all(np.abs(gam[~np.eye(4, dtype=bool)]) <= 1 + 1e-12)
# 目的関数が対角解のまわりで下がらないことも数値で確認
obj = lambda O_: np.log(np.linalg.det(O_)) - np.trace(Sg @ O_) - lam_ * np.sum(np.abs(O_[~np.eye(4, dtype=bool)]))
worse = all(obj(Od + 1e-3 * (lambda M: (M + M.T) / 2)(rng.normal(size=(4, 4)))) <= obj(Od) + 1e-12 for _ in range(200))
q.check("q25-graphical-lasso-1", ok_kkt and worse, "λ=max|s_jk| で対角解が最適性条件を満たす（正しい）")

# --- 導出問題（追加） ---
def _deriv25():
    import numpy as _N
    import sympy as _S
    x1, x2, x3 = _S.symbols("x1 x2 x3", real=True)
    w = _S.symbols("w11 w12 w13 w22 w23 w33", real=True)
    Om = _S.Matrix([[w[0], w[1], w[2]], [w[1], w[3], w[4]], [w[2], w[4], w[5]]])
    xv = _S.Matrix([x1, x2, x3])
    Qf = _S.expand((xv.T * Om * xv)[0])
    m = -(w[1] * x2 + w[2] * x3) / w[0]
    rest = _S.expand(Qf - w[0] * (x1 - m) ** 2)
    ok = _S.diff(rest, x1) == 0                       # 平方完成の残りに x1 が入らない
    rng = _N.random.default_rng(25)
    for _ in range(3):
        B = rng.normal(size=(4, 4)); S = B @ B.T + 0.2 * _N.eye(4); O = _N.linalg.inv(S)
        coef = S[0, 1:] @ _N.linalg.inv(S[1:, 1:])      # 一般の条件付き分布の公式
        cvar = S[0, 0] - coef @ S[1:, 0]
        ok &= _N.allclose(coef, -O[0, 1:] / O[0, 0]) and abs(cvar - 1 / O[0, 0]) < 1e-10
    q.check("q25-precision-matrix-4", bool(ok), "sympy: 平方完成の残りに x1 なし、Σ12Σ22^-1=-ω1j/ω11、条件付き分散=1/ω11")
_deriv25()

q.done()
