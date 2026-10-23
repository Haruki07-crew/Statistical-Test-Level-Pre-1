"""第17章（回帰診断法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats

q = Q(17)


def fit(X, y):
    """切片列を含む X で OLS。係数・残差・ハット行列を返す。"""
    b = np.linalg.solve(X.T @ X, X.T @ y)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return b, y - X @ b, H


rng = np.random.default_rng(17)

# --- 残差 ---
X = np.column_stack([np.ones(8), rng.normal(size=8), rng.normal(size=8)])
H = X @ np.linalg.inv(X.T @ X) @ X.T
Ve = np.eye(8) - H  # sigma^2=1 のときの V[e]
q.check("q17-residual-1", np.ptp(np.diag(Ve)) > 1e-3 and np.all(np.diag(Ve) < 1), "V[e_i]=1-h_ii は点ごとに異なり1未満（誤り）")
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 3, 5, 4, 8.])
b, e, _ = fit(np.column_stack([np.ones(5), x]), y)
q.num("q17-residual-2", e @ e / 3)
q.check("q17-residual-2", np.allclose(b, [0.5, 1.3]) and np.allclose(e, [0.2, -0.1, 0.6, -1.7, 1.0]), "解説の係数・残差")

# --- 標準化残差 ---
q.num("q17-standardized-residual-1", 2.4 / (1.5 * np.sqrt(1 - 0.36)))
# 上限 sqrt(n-p-1)=3：シミュレーションで超えないことと、極端な配置で近づくことを確認
n, p = 12, 2
mx = 0
for _ in range(2000):
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
    yy = rng.standard_cauchy(n)
    _, e, H = fit(X, yy)
    s = np.sqrt(e @ e / (n - p - 1))
    mx = max(mx, np.max(np.abs(e / (s * np.sqrt(1 - np.diag(H))))))
q.check("q17-standardized-residual-2", mx <= 3 + 1e-9 and mx > 2.9 and np.sqrt(12) > 3 + 0.4 and abs(np.sqrt(10) - 3) > 0.1,
        f"|r_i| の最大 {mx:.4f} ≤ 3、誤答 √12, √10 は3と異なる")

# --- スチューデント化残差 ---
t = 2.5 * np.sqrt(16 / (17 - 2.5 ** 2))
q.num("q17-studentized-residual-1", t)
# 1点除外で実際に当てはめ直して公式を確認
n, p = 20, 2
X = np.column_stack([np.ones(n), rng.normal(size=(n, p))]); yy = rng.normal(size=n); yy[0] += 4
_, e, H = fit(X, yy)
s2 = e @ e / (n - p - 1); r0 = e[0] / np.sqrt(s2 * (1 - H[0, 0]))
_, e1, _ = fit(X[1:], yy[1:]); s1 = np.sqrt(e1 @ e1 / (n - 1 - p - 1))
t0 = e[0] / (s1 * np.sqrt(1 - H[0, 0]))
q.check("q17-studentized-residual-1", abs(t0 - r0 * np.sqrt((n - p - 2) / (n - p - 1 - r0 ** 2))) < 1e-10, "公式 t=r√((n-p-2)/(n-p-1-r²)) を当てはめ直しで確認")
# 自由度：シミュレーションで t(20) に従う
n, p = 25, 3
ts = []
for _ in range(20000):
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p))]); yy = rng.normal(size=n)
    _, e, H = fit(X, yy)
    _, e1, _ = fit(X[1:], yy[1:]); s1 = np.sqrt(e1 @ e1 / (n - 1 - p - 1))
    ts.append(e[0] / (s1 * np.sqrt(1 - H[0, 0])))
ks20 = stats.kstest(ts, "t", args=(20,)).pvalue
q.check("q17-studentized-residual-2", ks20 > 0.01 and n - p - 2 == 20 and n - p - 1 == 21, f"t(20) への KS p={ks20:.3f}")

# --- てこ比 ---
x = np.array([1, 2, 3, 4, 10.])
_, _, H = fit(np.column_stack([np.ones(5), x]), np.zeros(5))
q.num("q17-leverage-1", H[4, 4])
q.num("q17-leverage-2", 2 * (4 + 1) / 40)
X = np.column_stack([np.ones(40), rng.normal(size=(40, 4))])
q.check("q17-leverage-2", abs(np.trace(X @ np.linalg.inv(X.T @ X) @ X.T) - 5) < 1e-9, "tr(H)=p+1=5")
q.check("q17-leverage-3", True, "H=X(X'X)^{-1}X' は y を含まない（誤り）")
# y を変えても H は同じ（fit が y に依存しない H を返す）
_, _, H1 = fit(X, rng.normal(size=40)); _, _, H2 = fit(X, rng.normal(size=40) * 100)
q.check("q17-leverage-3", np.allclose(H1, H2), "y を変えても H は不変")

# --- クックの距離 ---
D = lambda r, h, k: r ** 2 / k * h / (1 - h)
q.num("q17-cooks-distance-1", D(1.8, 0.4, 3))
# 公式 D = r^2/(p+1) h/(1-h) を当てはめ直しの定義で確認
n, p = 15, 2
X = np.column_stack([np.ones(n), rng.normal(size=(n, p))]); yy = rng.normal(size=n)
b, e, H = fit(X, yy); s2 = e @ e / (n - p - 1)
b1, _, _ = fit(X[1:], yy[1:]); dfit = X @ b - X @ b1
r0 = e[0] / np.sqrt(s2 * (1 - H[0, 0]))
q.check("q17-cooks-distance-1", abs(dfit @ dfit / ((p + 1) * s2) - D(r0, H[0, 0], p + 1)) < 1e-10, "公式と定義が一致")
DA, DB = D(3.0, 0.05, 2), D(1.2, 0.6, 2)
q.check("q17-cooks-distance-2", abs(DA - 0.24) < 0.005 and abs(DB - 1.08) < 0.005 and DB > DA, f"D_A={DA:.4f}, D_B={DB:.4f}")
q.check("q17-cooks-distance-2", abs(9 / 2 - 4.5) < 1e-12 and abs(1.44 / 2 - 0.72) < 1e-12
        and abs(4.5 * 0.05 - 0.225) < 1e-12 and abs(0.72 * 0.6 - 0.432) < 1e-12 and abs(DA - DB) > 0.5,
        "誤答（h/(1-h) の掛け忘れなど）は正解と異なる")

# --- DFFITS ---
q.num("q17-dffits-1", 2.2 * np.sqrt(0.3 / 0.7))
q.check("q17-dffits-1", abs(2 * np.sqrt(3 / 30) - 0.632) < 0.001, "解説の目安 0.632")

# --- DFBETAS ---
n, p = 10, 3
X = np.column_stack([np.ones(n), rng.normal(size=(n, p))]); yy = rng.normal(size=n)
b, e, H = fit(X, yy)
dfb = np.array([b - fit(np.delete(X, i, 0), np.delete(yy, i))[0] for i in range(n)])
q.check("q17-dfbetas-1", dfb.shape == (n, p + 1), "1点につき p+1 個（正しい）")

# --- 外れ値と影響点 ---
pts = {"A": (0.3, 0.45), "B": (2.6, 0.05), "C": (2.2, 0.35), "D": (0.8, 0.08)}
match = [k for k, (r, h) in pts.items() if h > 0.2 and abs(r) <= 2 and D(r, h, 2) <= 0.2]
q.check("q17-outlier-and-influential-point-2", match == ["A"], f"条件を満たすのは {match}")
q.check("q17-outlier-and-influential-point-1", D(3.0, 0.05, 2) < 0.5, "残差が大きくても D が小さい例がある（誤り）")

# --- 残差プロット ---
q.concept("q17-residual-plot-1", "らっぱ形＝不均一分散の読み取り")
X = np.column_stack([np.ones(30), rng.uniform(0, 5, 30)]); yy = np.exp(X[:, 1]) + rng.normal(size=30)  # わざと非線形
b, e, _ = fit(X, yy)
q.check("q17-residual-plot-2", abs(np.corrcoef(X @ b, e)[0, 1]) < 1e-10, "非線形でも残差と当てはめ値の相関は0（誤り）")

# --- 正規Q-Q ---
q.num("q17-normal-qq-plot-1", stats.norm.ppf(0.85))
q.check("q17-normal-qq-plot-1", abs(stats.norm.ppf(0.9) - 1.28) < 0.005 and abs(stats.norm.ppf(9 / 11) - 0.91) < 0.005, "解説の別流儀の値")
# 裾の重い t(3) 標本で左端が下・右端が上
z = np.sort(stats.t.ppf((np.arange(1, 201) - 0.5) / 200, 3)); qq = stats.norm.ppf((np.arange(1, 201) - 0.5) / 200)
sl = np.polyfit(qq[50:150], z[50:150], 1)
q.check("q17-normal-qq-plot-2", z[0] < np.polyval(sl, qq[0]) and z[-1] > np.polyval(sl, qq[-1]), "裾が重い分布で左端下・右端上")

# --- 不均一分散 ---
q.concept("q17-heteroscedasticity-1", "OLS の不偏性は E[ε]=0 だけで成り立つ（誤り）")
x = np.array([1, 2, 3, 4.]); X = np.column_stack([np.ones(4), x])
A = np.linalg.inv(X.T @ X) @ X.T
V = A @ np.diag(x ** 2) @ A.T
q.num("q17-heteroscedasticity-2", V[1, 1])
q.check("q17-heteroscedasticity-2", abs(7.5 / 5 - 1.5) < 1e-12 and abs(V[1, 1] - 1.5) > 0.1, "等分散と思った値 1.5 と異なる")

# --- 系列相関 ---
rho, su2 = 0.6, 2
q.num("q17-serial-correlation-1", su2 / (1 - rho ** 2) * rho ** 2)
# シミュレーションで確認
eps = np.zeros(200000); u = rng.normal(scale=np.sqrt(2), size=200000)
for i in range(1, len(eps)):
    eps[i] = rho * eps[i - 1] + u[i]
emp = np.mean(eps[1000:] * np.roll(eps, 2)[1000:])
q.check("q17-serial-correlation-1", abs(emp - 1.125) < 0.05, f"シミュレーションの共分散 {emp:.3f}")
# 正の系列相関の誤差・説明変数で、通常SEの平均が真のSDより小さい
n, R = 50, 3000
xs = np.zeros(n)
for i in range(1, n):
    xs[i] = 0.8 * xs[i - 1] + rng.normal()
X = np.column_stack([np.ones(n), xs])
bs, ses = [], []
for _ in range(R):
    ee = np.zeros(n); uu = rng.normal(size=n)
    for i in range(1, n):
        ee[i] = 0.8 * ee[i - 1] + uu[i]
    b, e, _ = fit(X, 1 + 2 * xs + ee)
    bs.append(b[1]); ses.append(np.sqrt(e @ e / (n - 2) * np.linalg.inv(X.T @ X)[1, 1]))
q.check("q17-serial-correlation-2", abs(np.mean(bs) - 2) < 0.05 and np.mean(ses) < 0.8 * np.std(bs),
        f"不偏（平均 {np.mean(bs):.3f}）、通常SE {np.mean(ses):.3f} < 真のSD {np.std(bs):.3f}")

# --- DW ---
e = np.array([1, 2, 1, -1, -2, -1.])
q.num("q17-durbin-watson-statistic-1", np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
d = 2 * (1 - (-0.35))
q.check("q17-durbin-watson-statistic-2", abs(d - 2.7) < 1e-12 and all(abs(w - d) > 0.5 for w in [2 * (1 - 0.35), abs(2 * -0.35), 2 * -0.35]),
        "正解 2.7、誤答 1.3・0.7・-0.7 と異なる")
dd, dL, dU = 2.85, 1.10, 1.54
q.check("q17-durbin-watson-statistic-3", dL <= 4 - dd <= dU, f"4-d={4 - dd:.2f} は [dL,dU] の中＝判定保留")

# --- BP ---
q.num("q17-breusch-pagan-test-1", stats.chi2.sf(50 * 0.12, 2))
q.check("q17-breusch-pagan-test-1", abs(stats.chi2.sf(6, 2) - np.exp(-3)) < 1e-12, "χ²_2 の上側確率 = e^{-x/2}")

# --- 頑健標準誤差 ---
q.concept("q17-robust-standard-error-1", "係数はOLSのまま、分散推定だけ変える（誤り）")

q.done()
