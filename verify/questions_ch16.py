"""第16章（重回帰分析）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats

q = Q(16)
rng = np.random.default_rng(16)


def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b, e


# --- モデル ---
X = np.column_stack([np.ones(30), rng.normal(size=(30, 4))])
q.check("q16-multiple-regression-model-1", X.shape == (30, 5) and (X.T @ X).shape == (5, 5), "X は 30×5")
x = rng.uniform(0, 3, 50); y = 1 + 2 * x - 0.5 * x ** 2 + rng.normal(0, 0.1, 50)
b, _ = ols(np.column_stack([np.ones(50), x, x ** 2]), y)
q.check("q16-multiple-regression-model-2", np.allclose(b, [1, 2, -0.5], atol=0.15), "2次項を説明変数にして OLS で推定できる（誤り）")

# --- 最小二乗 ---
b1 = 36 / 20
q.num("q16-least-squares-method-1", 12 - b1 * 5)
Xr = np.column_stack([np.ones(40), rng.normal(size=(40, 3))]); yr = rng.normal(size=40)
_, e = ols(Xr, yr)
q.check("q16-least-squares-method-2", np.allclose(Xr.T @ e, 0, atol=1e-10), "X^T e = 0（積和も0＝誤り）")

# --- 正規方程式 ---
sol = np.linalg.solve([[10, 4], [4, 8]], [20, 16])
q.num("q16-normal-equations-1", sol[0])
q.check("q16-normal-equations-1", abs(sol[1] - 1.25) < 1e-12, "β2=1.25")
bs = sp.Matrix(sp.symbols("b0:3")); Xs = sp.Matrix(4, 3, sp.symbols("x0:12")); ys = sp.Matrix(sp.symbols("y0:4"))
Sb = ((ys - Xs * bs).T * (ys - Xs * bs))[0]
grad = sp.Matrix([sp.diff(Sb, v) for v in bs])
q.check("q16-normal-equations-2", sp.simplify(grad - (-2 * Xs.T * ys + 2 * Xs.T * Xs * bs)) == sp.zeros(3, 1),
        "勾配 = -2X^Ty + 2X^TXβ（sympy）")

# --- ハット行列 ---
X20 = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
H = X20 @ np.linalg.inv(X20.T @ X20) @ X20.T
q.num("q16-hat-matrix-1", np.trace(H) / 20)
# 残差の分散 σ²(1-h_ii)：シミュレーション
beta = np.array([1, 2, -1, 0.5]); E = rng.normal(size=(20000, 20))
res = (E + X20 @ beta) @ (np.eye(20) - H).T
q.check("q16-hat-matrix-2", np.allclose(res.var(0), 1 - np.diag(H), atol=0.03) and np.ptp(np.diag(H)) > 0.05,
        "V[e_i]=σ²(1-h_ii) で i ごとに異なる（誤り）")
# x_i - x̄ = 3 の点と S_xx=30 をもつデータを作る
xd = np.array([3, -3, 1, -1, 1, -1, 2, -2, 0, 0], dtype=float)  # 平均0, S_xx=9+9+1+1+1+1+4+4=30
X1 = np.column_stack([np.ones(10), xd])
h = np.diag(X1 @ np.linalg.inv(X1.T @ X1) @ X1.T)
q.check("q16-hat-matrix-3", abs((xd ** 2).sum() - 30) < 1e-12 and xd.mean() == 0, "S_xx=30 のデータを構成")
q.num("q16-hat-matrix-3", h[0])

# --- ガウス・マルコフ ---
q.concept("q16-gauss-markov-theorem-1", "定理の仮定（正規性は不要）")
# 非線形な不偏推定量のほうが分散が小さい例：一様誤差での位置推定（切片のみモデル）では中点推定量が標本平均より良い
n = 20; sims = rng.uniform(-1, 1, (20000, n)) + 3.0
mean_v = sims.mean(1).var(); mid_v = ((sims.max(1) + sims.min(1)) / 2).var()
q.check("q16-gauss-markov-theorem-2", mid_v < mean_v and abs(((sims.max(1) + sims.min(1)) / 2).mean() - 3) < 0.01,
        "一様誤差では非線形の中点推定量（不偏）が OLS より分散小（誤り）")

# --- 決定係数 ---
q.num("q16-coefficient-of-determination-1", np.sqrt(150 / 200))
ok = True
for _ in range(100):
    Xa = np.column_stack([np.ones(25), rng.normal(size=(25, 2))]); ya = rng.normal(size=25)
    Xb = np.column_stack([Xa, rng.normal(size=25)])
    ra = 1 - (ols(Xa, ya)[1] ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    rb = 1 - (ols(Xb, ya)[1] ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    ok &= rb >= ra - 1e-12
q.check("q16-coefficient-of-determination-2", ok, "変数を加えても R² は下がらない（100回）")

# --- 自由度調整済み ---
adj = lambda R2, n, p: 1 - (1 - R2) * (n - 1) / (n - p - 1)
q.num("q16-adjusted-r-squared-1", adj(0.6, 25, 4))
q.num("q16-adjusted-r-squared-2", adj(0.1, 11, 3))
ok = True
for _ in range(300):
    Xa = np.column_stack([np.ones(30), rng.normal(size=(30, 2))]); ya = Xa @ [1, 0.3, 0] + rng.normal(size=30)
    z = rng.normal(size=30); Xb = np.column_stack([Xa, z])
    ST = ((ya - ya.mean()) ** 2).sum()
    ea, eb = ols(Xa, ya)[1], ols(Xb, ya)[1]
    Sa, Sb_ = (ea ** 2).sum(), (eb ** 2).sum()
    s2 = Sb_ / (30 - 3 - 1)
    c = np.linalg.inv(Xb.T @ Xb)[3, 3]
    t = ols(Xb, ya)[0][3] / np.sqrt(s2 * c)
    up = adj(1 - Sb_ / ST, 30, 3) > adj(1 - Sa / ST, 30, 2)
    ok &= up == (abs(t) > 1)
q.check("q16-adjusted-r-squared-3", ok, "R̄² が上がる ⇔ |t|>1（300回のシミュレーションで一致）")

# --- t 検定 ---
q.num("q16-regression-coefficient-t-test-1", 1.1 / (2 * 0.2))
q.num("q16-regression-coefficient-t-test-2", 1.1 - 2.120 * 0.4)
q.check("q16-regression-coefficient-t-test-2", abs(stats.t.ppf(0.975, 16) - 2.120) < 5e-4, "t_{0.025}(16)=2.120")
q.check("q16-regression-coefficient-t-test-3", 30 - 4 - 1 == 25 and len({25, 26, 29, 28}) == 4, "自由度 25")

# --- F 検定 ---
q.num("q16-overall-f-test-regression-1", (0.4 / 3) / (0.6 / 20))
q.check("q16-overall-f-test-regression-1", abs(stats.f.ppf(0.95, 3, 20) - 3.10) < 0.005, "F(3,20) 上側5%点 3.10")
q.check("q16-overall-f-test-regression-2", (3, 24 - 3 - 1) == (3, 20), "自由度 (3,20)")

# --- 標準化偏回帰係数 ---
Rm = np.array([[1, 0.4], [0.4, 1]]); ry = np.array([0.6, 0.5])
bstar = np.linalg.solve(Rm, ry)
q.num("q16-standardized-partial-regression-coefficient-1", bstar[0])
q.num("q16-standardized-partial-regression-coefficient-2", bstar @ ry)
# 相関行列を持つ多変量正規データの回帰で確認
C = np.array([[1, 0.4, 0.6], [0.4, 1, 0.5], [0.6, 0.5, 1]])
Z = rng.multivariate_normal(np.zeros(3), C, 200000)
Zs = (Z - Z.mean(0)) / Z.std(0)
bz = np.linalg.lstsq(Zs[:, :2], Zs[:, 2], rcond=None)[0]
q.check("q16-standardized-partial-regression-coefficient-1", np.allclose(bz, bstar, atol=0.01), "標準化データの回帰と一致")

# --- 多重共線性 ---
q.num("q16-multicollinearity-1", np.sqrt(1 / (1 - 0.81)))
# F は有意だが t は有意でない例を作る
found = False
for _ in range(200):
    x1 = rng.normal(size=30); x2 = x1 + rng.normal(0, 0.05, 30)
    ya = 1 + x1 + x2 + rng.normal(0, 2, 30)
    Xc = np.column_stack([np.ones(30), x1, x2]); bc, ec = ols(Xc, ya)
    s2 = (ec ** 2).sum() / 27; se = np.sqrt(s2 * np.diag(np.linalg.inv(Xc.T @ Xc)))
    ST = ((ya - ya.mean()) ** 2).sum(); R2 = 1 - (ec ** 2).sum() / ST
    F = (R2 / 2) / ((1 - R2) / 27)
    if stats.f.sf(F, 2, 27) < 0.05 and all(2 * stats.t.sf(abs(bc[1:] / se[1:]), 27) > 0.05):
        found = True; break
q.check("q16-multicollinearity-2", found, "F 有意・個別 t 非有意の例が実際に作れる（正しい）")

# --- VIF ---
q.num("q16-variance-inflation-factor-1", 1 / (1 - 0.8))
R3 = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
vif1 = np.linalg.inv(R3)[0, 0]
q.num("q16-variance-inflation-factor-2", vif1)
q.check("q16-variance-inflation-factor-2", abs(np.linalg.det(R3) - 0.62) < 1e-12 and abs(vif1 - 0.84 / 0.62) < 1e-12, "det=0.62、余因子0.84")

# --- ダミー ---
g = rng.integers(0, 4, 40)
D4 = np.column_stack([np.ones(40)] + [(g == k).astype(float) for k in range(4)])
D3 = D4[:, :4]
q.check("q16-dummy-variable-1", np.linalg.matrix_rank(D4) == 4 and np.linalg.matrix_rank(D3) == 4, "4個入れるとランク落ち、3個ならフルランク")
q.num("q16-dummy-variable-2", 20 + 1.5 * 5 + 2.5)

# --- 交互作用 ---
x1s, x2s = sp.symbols("x1 x2")
yh = 5 + 2 * x1s + 3 * x2s + sp.Rational(1, 2) * x1s * x2s
q.num("q16-interaction-term-1", float(sp.diff(yh, x1s).subs(x2s, 4)))

# --- 区間 ---
tq = 2.228
lo = 30 - tq * 2 * np.sqrt(1 + 1 / 12 + 25 / 50)
q.num("q16-confidence-and-prediction-interval-regression-1", lo)
q.check("q16-confidence-and-prediction-interval-regression-1", abs(stats.t.ppf(0.975, 10) - tq) < 5e-4
        and abs(30 - tq * 2 * np.sqrt(1 / 12 + 0.5) - 26.60) < 0.005, "t 値 2.228、信頼区間の下限 26.60")
q0 = lambda n_: 1 / n_ + 25 / (50 * n_ / 12)  # S_xx が n に比例して増える
q.check("q16-confidence-and-prediction-interval-regression-2", q0(1e8) < 1e-6 and np.sqrt(1 + q0(1e8)) > 0.999,
        "q0→0、予測区間の √(1+q0)→1（正しい）")

# --- 部分 F ---
q.num("q16-partial-f-test-1", ((120 - 90) / 2) / (90 / 24))
q.check("q16-partial-f-test-1", abs(stats.f.ppf(0.95, 2, 24) - 3.40) < 0.005, "F(2,24) 上側5%点 3.40")

q.done()
