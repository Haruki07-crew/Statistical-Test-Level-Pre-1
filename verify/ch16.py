"""第16章 検算スクリプト（重回帰分析）。
厳密計算は sympy / fractions、数値は numpy / scipy.stats。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（標準誤差の4倍以上）。
"""
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from scipy import stats

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
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol))


def front(term):
    txt = (ROOT / "content" / "ch16" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Se = float(e @ e)
    n, k = X.shape
    s2 = Se / (n - k)
    se = np.sqrt(s2 * np.diag(XtXi))
    ST = float(((y - y.mean()) ** 2).sum())
    return dict(b=b, e=e, Se=Se, s2=s2, se=se, t=b / se, XtXi=XtXi, ST=ST, R2=1 - Se / ST, n=n, k=k)


# ============ 共通データ ============
x1 = np.array([1, 2, 2, 3, 4, 4, 5, 6], float)
x2 = np.array([3, 2, 4, 3, 5, 4, 6, 5], float)
y = np.array([4, 6, 9, 7, 11, 10, 14, 13], float)
n = 8
one = np.ones(n)
X = np.column_stack([one, x1, x2])
X1 = np.column_stack([one, x1])
X2 = np.column_stack([one, x2])
full = ols(X, y)
m1 = ols(X1, y)
m2 = ols(X2, y)
# 厳密値
Xs = sp.Matrix(X.astype(int))
ys = sp.Matrix(y.astype(int))
XtX = Xs.T * Xs
XtXi_s = XtX.inv()
bs = XtXi_s * Xs.T * ys
es = ys - Xs * bs
Se_s = (es.T * es)[0]
ST_s = sum((v - sp.Rational(74, 8)) ** 2 for v in ys)
Hs = Xs * XtXi_s * Xs.T

# ============ least-squares-method ============
T = "least-squares-method"
check(T, "例1 x̄=3.375, ȳ=9.25", x1.mean() == 3.375 and y.mean() == 9.25, (x1.mean(), y.mean()))
Sxx = ((x1 - x1.mean()) ** 2).sum()
Sxy = ((x1 - x1.mean()) * (y - y.mean())).sum()
check(T, "例1 Sxx=19.875, Sxy=37.25", Sxx == 19.875 and Sxy == 37.25, (Sxx, Sxy))
check(T, "例1 β̂1=1.8742, β̂0=2.9245", close(Sxy / Sxx, 1.8742) and close(9.25 - Sxy / Sxx * 3.375, 2.9245), m1["b"])
check(T, "例1 lstsq と一致", allclose(np.linalg.lstsq(X1, y, rcond=None)[0], m1["b"], 1e-10), m1["b"])
check(T, "例2 X'X, X'y", XtX == sp.Matrix([[8, 27, 32], [27, 111, 120], [32, 120, 140]]) and Xs.T * ys == sp.Matrix([74, 287, 325]), XtX.tolist())
check(T, "例2 β̂=(0.2381,1.0476,1.3690)", allclose(full["b"], [0.2381, 1.0476, 1.3690]), full["b"])
check(T, "例2 Se=4.7738", close(full["Se"], 4.7738), full["Se"])
alt = np.array([0.5, 1.2, 1.2])
S_alt = float(((y - X @ alt) ** 2).sum())
check(T, "例2 (0.5,1.2,1.2) の残差平方和 5.04", close(S_alt, 5.04, 5e-3), S_alt)
d = alt - full["b"]
check(T, "S(β)=Se+(β−β̂)'X'X(β−β̂)（(0.5,1.2,1.2) で）", close(S_alt, full["Se"] + d @ X.T @ X @ d, 1e-10), full["Se"] + d @ X.T @ X @ d)
# ランダムな β でも S≥Se
ok = all(((y - X @ b) ** 2).sum() >= full["Se"] - 1e-12 for b in rng.normal(full["b"], 1, size=(2000, 3)))
check(T, "ランダムな β 2000個で S(β)≥Se", ok, ok)
# sympy で微分 → 正規方程式
b0, b1, b2 = sp.symbols("b0 b1 b2")
bv = sp.Matrix([b0, b1, b2])
S_sym = ((ys - Xs * bv).T * (ys - Xs * bv))[0]
grad = sp.Matrix([sp.diff(S_sym, v) for v in bv])
check(T, "sympy: ∂S/∂β = −2X'y+2X'Xβ", sp.simplify(grad - (-2 * Xs.T * ys + 2 * XtX * bv)) == sp.zeros(3, 1), "一致")
sol = sp.solve(list(grad), [b0, b1, b2])
check(T, "sympy: ∂S/∂β=0 の解 = (5/21,22/21,115/84)", [sol[b0], sol[b1], sol[b2]] == [sp.Rational(5, 21), sp.Rational(22, 21), sp.Rational(115, 84)], sol)
check(T, "sympy: ヘッセ行列 2X'X が正定値（主小行列式）", (sp.hessian(S_sym, bv) == 2 * XtX) and all((2 * XtX)[:k, :k].det() > 0 for k in (1, 2, 3)), "正定値")
# 単回帰の別表現 β1 = r √(Syy/Sxx)
r1y = np.corrcoef(x1, y)[0, 1]
check(T, "β̂1=r_xy√(Syy/Sxx)", close(r1y * np.sqrt(83.5 / Sxx), Sxy / Sxx, 1e-12), r1y * np.sqrt(83.5 / Sxx))
# σ² の MLE と不偏推定量
check(T, "σ² の MLE=Se/n=0.5967、不偏=Se/(n−p−1)=0.9548", close(full["Se"] / 8, 0.59673, 1e-4) and close(full["s2"], 0.9548), (full["Se"] / 8, full["s2"]))
# 図1
ch = charts(T)
p = ch[0]["params"]
Sb1 = [float(((y - X @ np.array([full["b"][0], b, full["b"][2]])) ** 2).sum()) for b in p["x"]]
check(T, "図1 S(β1) の10点", allclose(Sb1, p["series"][0]["y"]), np.round(Sb1, 4))
check(T, "図1 キャプション 最小 β1=1.0476 で 4.7738", close(full["b"][1], 1.0476) and close(full["Se"], 4.7738), (full["b"][1], full["Se"]))
check(T, "図2 regression-residual params", ch[1]["params"] == {"n": 20, "beta0": 1, "beta1": 0.5, "sigma": 1, "outlier": 0}, ch[1]["params"])
# x を y に回帰した直線と y を x に回帰した直線が違う
bxy = Sxy / ((y - y.mean()) ** 2).sum()
check(T, "y on x と x on y の傾きが異なる（1/bxy≠byx, r≠±1）", abs(1 / bxy - Sxy / Sxx) > 0.1, (Sxy / Sxx, 1 / bxy))

# ============ normal-equations ============
T = "normal-equations"
S11 = Sxx
S22 = ((x2 - 4) ** 2).sum()
S12 = ((x1 - x1.mean()) * (x2 - 4)).sum()
S2y = ((x2 - 4) * (y - y.mean())).sum()
check(T, "例1 x̄2=4, 偏差行列 (19.875,12;12,12), 右辺 (37.25,29)", x2.mean() == 4 and (S11, S12, S22, Sxy, S2y) == (19.875, 12, 12, 37.25, 29), (S11, S12, S22, Sxy, S2y))
det = S11 * S22 - S12 ** 2
check(T, "例1 行列式 94.5", det == 94.5, det)
check(T, "例1 クラメル分子 99, 129.375", 12 * 37.25 - 12 * 29 == 99 and 19.875 * 29 - 12 * 37.25 == 129.375, (99, 129.375))
check(T, "例1 β̂1=99/94.5=1.0476, β̂2=129.375/94.5=1.3690", close(99 / 94.5, 1.0476) and close(129.375 / 94.5, 1.3690), (99 / 94.5, 129.375 / 94.5))
check(T, "例1 β̂0=9.25−1.0476×3.375−1.3690×4=0.2381", close(9.25 - 99 / 94.5 * 3.375 - 129.375 / 94.5 * 4, 0.2381), 9.25 - 99 / 94.5 * 3.375 - 129.375 / 94.5 * 4)
check(T, "例2 det(X'X)=756", XtX.det() == 756, XtX.det())
adj = sp.Matrix([[1140, 60, -312], [60, 96, -96], [-312, -96, 159]])
check(T, "例2 (X'X)^{-1}=adj/756", XtXi_s == adj / 756, adj.tolist())
check(T, "例2 β̂=(180,792,1035)/756", bs == sp.Matrix([180, 792, 1035]) / 756, bs.T)
check(T, "例2 残差8個", allclose(full["e"], [-1.3929, 0.9286, 1.1905, -0.4881, -0.2738, 0.0952, 0.3095, -0.3690]), np.round(full["e"], 4))
check(T, "性質1・2 Σe=0, Σx1e=0, Σx2e=0（厳密）", list(Xs.T * es) == [0, 0, 0], list(Xs.T * es))
check(T, "性質3 平均の点を通る", (bs[0] + bs[1] * sp.Rational(27, 8) + bs[2] * 4) == sp.Rational(37, 4), "ȳ=9.25")
yh = Xs * bs
check(T, "性質4 ŷ'e=0（厳密）", (yh.T * es)[0] == 0, 0)
check(T, "性質 残差と説明変数の標本共分散 0", close(np.cov(x1, full["e"])[0, 1], 0, 1e-12) and close(np.cov(x2, full["e"])[0, 1], 0, 1e-12), "0")
# 例3 直交計画
Xo = np.array([[1, -1, -1], [1, 1, -1], [1, -1, 1], [1, 1, 1]], float)
yo = np.array([10, 14, 12, 18], float)
check(T, "例3 X'X=diag(4,4,4), X'y=(54,10,6)", np.array_equal(Xo.T @ Xo, 4 * np.eye(3)) and np.array_equal(Xo.T @ yo, [54, 10, 6]), Xo.T @ yo)
bo = ols(Xo, yo)["b"]
check(T, "例3 β̂=(13.5,2.5,1.5)", allclose(bo, [13.5, 2.5, 1.5], 1e-12), bo)
check(T, "例3 圧力を外しても温度の係数 2.5", close(ols(Xo[:, :2], yo)["b"][1], 2.5, 1e-12), ols(Xo[:, :2], yo)["b"])
# 切片なしでは残差和 0 とは限らない
e_noint = y - x1 * (x1 @ y / (x1 @ x1))
check(T, "切片なしモデルでは Σe≠0（共通データ x1 のみ）", abs(e_noint.sum()) > 0.1, e_noint.sum())
# 完全共線性で X'X 特異
Xc = np.column_stack([one, x1, x2, x1 + x2])
check(T, "完全共線性で rank(X'X)<p+1", np.linalg.matrix_rank(Xc.T @ Xc) == 3, np.linalg.matrix_rank(Xc.T @ Xc))
# 相関行列での正規方程式
Rm = np.corrcoef(np.vstack([x1, x2]))
ry = np.array([np.corrcoef(x1, y)[0, 1], np.corrcoef(x2, y)[0, 1]])
bstar = np.linalg.solve(Rm, ry)
check(T, "Rβ*=r_y の解 = β̂_j s_j/s_y", allclose(bstar, full["b"][1:] * np.array([x1.std(ddof=1), x2.std(ddof=1)]) / y.std(ddof=1), 1e-12), bstar)
ch = charts(T)
check(T, "図 残差の棒 = 例2の残差", ch[0]["params"]["values"] == [-1.3929, 0.9286, 1.1905, -0.4881, -0.2738, 0.0952, 0.3095, -0.3690], ch[0]["params"]["values"])
check(T, "図 キャプション 棒の合計≈0（丸め込み）", close(sum(ch[0]["params"]["values"]), 0, 4e-4), sum(ch[0]["params"]["values"]))

# ============ multiple-regression-model ============
T = "multiple-regression-model"
check(T, "例1 β̂=(5/21,22/21,115/84)", list(bs) == [sp.Rational(5, 21), sp.Rational(22, 21), sp.Rational(115, 84)], list(bs))
check(T, "例1 X は 8×3", X.shape == (8, 3), X.shape)
check(T, "例1 小数 0.2381, 1.0476, 1.3690; 約1.048", allclose([5 / 21, 22 / 21, 115 / 84], [0.2381, 1.0476, 1.3690]) and close(22 / 21, 1.048, 5e-4), [5 / 21, 22 / 21, 115 / 84])
check(T, "例2 単回帰の傾き 1.8742", close(m1["b"][1], 1.8742), m1["b"][1])
check(T, "例2 x1 と x2 の相関 0.777", close(np.corrcoef(x1, x2)[0, 1], 0.777, 5e-4), np.corrcoef(x1, x2)[0, 1])
# 欠落変数バイアス: 単回帰傾き = β1 + β2 * (x2 を x1 に回帰した傾き)
g = S12 / S11
check(T, "例2 単回帰傾き = β̂1+β̂2·(S12/S11)（面積効果の混入）", close(full["b"][1] + full["b"][2] * g, m1["b"][1], 1e-12), full["b"][1] + full["b"][2] * g)
ch = charts(T)
check(T, "図1 scatter regression params", ch[0]["params"] == {"beta0": 1, "beta1": 0.8, "sigma": 1, "n": 50}, ch[0]["params"])
check(T, "図2 regression-residual params", ch[1]["params"] == {"n": 20, "beta0": 1, "beta1": 0.5, "sigma": 1, "outlier": 0}, ch[1]["params"])
# 線形モデル: x² を入れても β について一次 → OLS の式がそのまま使える（多項式回帰の一致）
xx = np.arange(1, 9, dtype=float)
yy = 1 + 0.5 * xx - 0.2 * xx ** 2
Xp = np.column_stack([np.ones(8), xx, xx ** 2])
check(T, "y=β0+β1x+β2x² も OLS で係数を復元（線形モデル）", allclose(ols(Xp, yy)["b"], [1, 0.5, -0.2], 1e-9), ols(Xp, yy)["b"])

# ============ hat-matrix ============
T = "hat-matrix"
I8 = sp.eye(8)
check(T, "共通データ H'=H, H²=H（厳密）", Hs.T == Hs and Hs * Hs == Hs, "成立")
check(T, "HX=X, (I−H)X=O（厳密）", Hs * Xs == Xs and (I8 - Hs) * Xs == sp.zeros(8, 3), "成立")
check(T, "tr H=3, tr(I−H)=5", Hs.trace() == 3 and (I8 - Hs).trace() == 5, (Hs.trace(), (I8 - Hs).trace()))
ev = np.linalg.eigvalsh(np.array(Hs, float))
check(T, "H の固有値は 0 か 1（1 が3個）", allclose(np.sort(ev), [0] * 5 + [1] * 3, 1e-10), np.round(ev, 10))
hd = [Hs[i, i] for i in range(8)]
check(T, "例2 h_ii 8個", allclose([float(v) for v in hd], [0.4484, 0.5079, 0.3651, 0.2579, 0.2262, 0.1746, 0.4762, 0.5437]), [round(float(v), 4) for v in hd])
check(T, "例2 h11=339/756", hd[0] == sp.Rational(339, 756) and (1140 + 2 * 60 * 1 - 2 * 312 * 3 + 96 * 1 - 2 * 96 * 3 + 159 * 9) == 339, hd[0])
check(T, "1/n ≤ h_ii ≤ 1", all(sp.Rational(1, 8) <= v <= 1 for v in hd), "成立")
check(T, "例2 合計 3、平均 0.375", sum(hd) == 3, sum(hd) / 8)
check(T, "例2 1−h88=0.4563, 1−h66=0.8254", close(1 - float(hd[7]), 0.4563) and close(1 - float(hd[5]), 0.8254), (1 - float(hd[7]), 1 - float(hd[5])))
check(T, "例2 H の第1行", allclose([float(v) for v in Hs[0, :]], [0.4484, 0.1825, 0.3651, 0.0992, 0.1071, 0.0159, 0.0238, -0.2421]), [round(float(v), 4) for v in Hs[0, :]])
check(T, "例2 ŷ1=5.3929", close(float((Hs[0, :] * ys)[0]), 5.3929), float((Hs[0, :] * ys)[0]))
check(T, "てこ比が大きいのは店1,2,7,8（平均0.375超）", [i + 1 for i, v in enumerate(hd) if v > sp.Rational(3, 8)] == [1, 2, 7, 8], [i + 1 for i, v in enumerate(hd) if v > sp.Rational(3, 8)])
# 例1 単回帰
xs5 = np.arange(1, 6, dtype=float)
X5 = np.column_stack([np.ones(5), xs5])
H5 = X5 @ np.linalg.inv(X5.T @ X5) @ X5.T
check(T, "例1 h_ii=0.6,0.3,0.2,0.3,0.6、合計2", allclose(np.diag(H5), [0.6, 0.3, 0.2, 0.3, 0.6], 1e-12) and close(np.trace(H5), 2, 1e-12), np.diag(H5))
check(T, "例1 公式 1/n+(x−x̄)²/Sxx と一致", allclose(np.diag(H5), 1 / 5 + (xs5 - 3) ** 2 / 10, 1e-12), "一致")
check(T, "定数のみモデルで H=11'/n", allclose(np.ones((8, 1)) @ np.linalg.inv(np.ones((1, 8)) @ np.ones((8, 1))) @ np.ones((1, 8)), np.full((8, 8), 1 / 8), 1e-12), "1/8")
# 記号: 一般の X（n=4, p=1）で H²=H, tr H=2
a = sp.symbols("a1:5")
Xg = sp.Matrix([[1, ai] for ai in a])
Hg = Xg * (Xg.T * Xg).inv() * Xg.T
check(T, "sympy: 一般の x（n=4,単回帰）で H²=H, trH=2", sp.simplify(Hg * Hg - Hg) == sp.zeros(4, 4) and sp.simplify(Hg.trace()) == 2, "成立")
xbar = sum(a) / 4
check(T, "sympy: h_ii=1/n+(x_i−x̄)²/Sxx", all(sp.simplify(Hg[i, i] - (sp.Rational(1, 4) + (a[i] - xbar) ** 2 / sum((aj - xbar) ** 2 for aj in a))) == 0 for i in range(4)), "成立")
# E[Se]=(n−p−1)σ²、V[e]=σ²(I−H)（シミュレーション）
beta_true = np.array([1.0, 1.0, 1.5])
sig = 1.0
N = 200000
E = rng.normal(0, sig, size=(N, 8))
Ys = X @ beta_true + E
B = Ys @ X @ full["XtXi"]  # 行ごとの β̂
R = Ys - B @ X.T
Se_sim = (R ** 2).sum(1)
check(T, "シミュ: E[Se]=5σ²（20万回、許容±0.03）", close(Se_sim.mean(), 5, 0.03), Se_sim.mean())
Ve = np.cov(R.T)
IH = np.eye(8) - np.array(Hs, float)
check(T, "シミュ: V[e]=σ²(I−H)（20万回、全成分許容±0.01）", np.max(np.abs(Ve - IH)) < 0.01, np.max(np.abs(Ve - IH)))
ks = stats.kstest(Se_sim[:20000], "chi2", args=(5,))
check(T, "シミュ: Se/σ² ~ χ²(5)（KS 検定 2万個、p>0.001）", ks.pvalue > 0.001, ks.pvalue)
ch = charts(T)
check(T, "図1 bar-data = h_ii", ch[0]["params"]["values"] == [0.4484, 0.5079, 0.3651, 0.2579, 0.2262, 0.1746, 0.4762, 0.5437], ch[0]["params"]["values"])
check(T, "図1 キャプション 合計3・平均 0.375・平均点 (3.375,4)", close(sum(ch[0]["params"]["values"]), 3, 5e-4) and x1.mean() == 3.375 and x2.mean() == 4, sum(ch[0]["params"]["values"]))
check(T, "図2 regression-residual params", ch[1]["params"] == {"n": 20, "beta0": 1, "beta1": 0.5, "sigma": 1, "outlier": 4}, ch[1]["params"])

# ============ gauss-markov-theorem ============
T = "gauss-markov-theorem"
check(T, "シミュ: E[β̂]=β（20万回、許容±0.01）", allclose(B.mean(0), beta_true, 0.01), B.mean(0))
VB = np.cov(B.T)
Vt = full["XtXi"]
se_cov = np.sqrt((np.outer(np.diag(Vt), np.diag(Vt)) + Vt ** 2) / N)  # 標本共分散の標準誤差（正規）
check(T, "シミュ: V[β̂]=σ²(X'X)^{-1}（20万回、全成分 許容4SE）", np.all(np.abs(VB - Vt) < 4 * se_cov), np.round(VB, 4).tolist())
# sympy: E[β̂]=β, V[β̂]=σ²(X'X)^{-1} は (X'X)^{-1}X' X = I と A A' から
A_ = XtXi_s * Xs.T
check(T, "sympy: (X'X)^{-1}X'X=I（不偏性）", A_ * Xs == sp.eye(3), "I")
check(T, "sympy: AA'=(X'X)^{-1}（分散）", A_ * A_.T == XtXi_s, "一致")
# 例1
cA = np.array([-1, 0, 0, 0, 1]) / 4
cB = np.array([-1, -1, 0, 1, 1]) / 6
cL = (xs5 - 3) / 10
check(T, "例1 A,B,LS が不偏（Σc=0, Σc x=1）", all(close(c.sum(), 0, 1e-12) and close(c @ xs5, 1, 1e-12) for c in (cA, cB, cL)), "成立")
check(T, "例1 分散 0.125σ², 0.1111σ², 0.1σ²", close(cA @ cA, 0.125, 1e-12) and close(cB @ cB, 0.1111) and close(cL @ cL, 0.1, 1e-12), (cA @ cA, cB @ cB, cL @ cL))
check(T, "例1 効率 0.1/0.125=0.8", close(0.1 / 0.125, 0.8, 1e-12), 0.8)
# D=C−A で DX=O、V=σ²(A A'+DD')
C5 = np.vstack([np.array([1, 0, 0, 0, 0.0]) * 0 + np.linalg.inv(X5.T @ X5)[0] @ X5.T, cA])  # (β0 は LS, β1 は A)
A5 = np.linalg.inv(X5.T @ X5) @ X5.T
D5 = C5 - A5
check(T, "証明: CX=I なら DX=O、CC'=(X'X)^{-1}+DD'", allclose(C5 @ X5, np.eye(2), 1e-12) and allclose(D5 @ X5, np.zeros((2, 2)), 1e-12) and allclose(C5 @ C5.T, np.linalg.inv(X5.T @ X5) + D5 @ D5.T, 1e-12), "成立")
# ランダムな線形不偏推定量で V 差が非負定値
okpsd = True
for _ in range(300):
    Dr = rng.normal(size=(3, 8))
    Dr = Dr @ (np.eye(8) - np.array(Hs, float))  # DX=O
    Cr = full["XtXi"] @ X.T + Dr
    okpsd &= np.min(np.linalg.eigvalsh(Cr @ Cr.T - full["XtXi"])) > -1e-10
check(T, "ランダムな線形不偏推定量300個で V差が非負定値", okpsd, okpsd)
# 例2
Vhat = full["s2"] * full["XtXi"]
check(T, "例2 s²=4.7738/5=0.9548", close(full["s2"], 0.9548), full["s2"])
check(T, "例2 s²(X'X)^{-1}", allclose(Vhat, [[1.4397, 0.0758, -0.3940], [0.0758, 0.1212, -0.1212], [-0.3940, -0.1212, 0.2008]]), np.round(Vhat, 4).tolist())
corr_b = -96 / np.sqrt(96 * 159)
check(T, "例2 corr(β̂1,β̂2)=−96/√(96·159)=−0.777 = −r12", close(corr_b, -0.777, 5e-4) and close(corr_b, -np.corrcoef(x1, x2)[0, 1], 1e-12), corr_b)
# シミュ例1
Ysim = np.array([1, 2]) @ np.array([np.ones(5), xs5]) + rng.normal(size=(200000, 5))
vA = (Ysim @ cA).var()
vB = (Ysim @ cB).var()
vL = (Ysim @ cL).var()
check(T, "シミュ例1: 分散 0.125/0.1111/0.1（20万回、許容±0.0015）", close(vA, .125, .0015) and close(vB, 1 / 9, .0015) and close(vL, .1, .0015), (vA, vB, vL))
check(T, "シミュ例1: 平均はどれも 2（許容±0.005）", all(close((Ysim @ c).mean(), 2, 0.005) for c in (cA, cB, cL)), [(Ysim @ c).mean() for c in (cA, cB, cL)])
ch = charts(T)
p = ch[0]
check(T, "図 mu=2, σ=√0.1=0.3162", p["params"] == {"mu": 2, "sigma": 0.3162} and close(np.sqrt(0.1), 0.3162), p["params"])
check(T, "図 比較 B: σ=1/3=0.3333, A: σ=√0.125=0.3536", p["compare"][0]["params"] == {"mu": 2, "sigma": 0.3333} and p["compare"][1]["params"] == {"mu": 2, "sigma": 0.3536} and close(np.sqrt(.125), .3536), [c["params"] for c in p["compare"]])

# ============ coefficient-of-determination ============
T = "coefficient-of-determination"
check(T, "例1 ST=83.5", ST_s == sp.Rational(167, 2), ST_s)
SR_s = sum((v - sp.Rational(37, 4)) ** 2 for v in yh)
check(T, "平方和分解 ST=SR+Se（厳密）", SR_s + Se_s == ST_s, (SR_s, Se_s))
check(T, "例1 Se=4.7738, SR=78.7262", close(float(Se_s), 4.7738) and close(float(SR_s), 78.7262), (float(Se_s), float(SR_s)))
R2 = float(SR_s / ST_s)
check(T, "例1 R²=0.9428, R=0.9710", close(R2, 0.9428) and close(np.sqrt(R2), 0.9710), (R2, np.sqrt(R2)))
check(T, "R²=r(y,ŷ)²", close(np.corrcoef(y, X @ full["b"])[0, 1] ** 2, R2, 1e-12), np.corrcoef(y, X @ full["b"])[0, 1] ** 2)
check(T, "Σ(y−ȳ)(ŷ−ȳ)=SR", sum((ys[i] - sp.Rational(37, 4)) * (yh[i] - sp.Rational(37, 4)) for i in range(8)) == SR_s, "一致")
check(T, "例2 x1 のみ Se=13.6855, R²=0.8361", close(m1["Se"], 13.6855) and close(m1["R2"], 0.8361), (m1["Se"], m1["R2"]))
check(T, "例2 r=0.9144, r²=0.8361=R²", close(r1y, 0.9144) and close(r1y ** 2, m1["R2"], 1e-12), r1y)
Fv = (R2 / 2) / ((1 - R2) / 5)
check(T, "例3 F=41.23（R² から）", close(Fv, 41.23, 5e-3), Fv)
check(T, "例3 R²/2=0.4714, (1−R²)/5=0.01143（丸める前の R² で）", close(R2 / 2, 0.4714) and close((1 - R2) / 5, 0.01143, 5e-6), (R2 / 2, (1 - R2) / 5))
check(T, "例3 分散分析表からの F と同値", close((float(SR_s) / 2) / (float(Se_s) / 5), Fv, 1e-10), (float(SR_s) / 2) / (float(Se_s) / 5))
Rq, pp_, nn = sp.symbols("R2 p n", positive=True)
STs = sp.symbols("S_T", positive=True)
check(T, "sympy: F(SR,Se)=R²式", sp.simplify((Rq * STs / pp_) / ((1 - Rq) * STs / (nn - pp_ - 1)) - (Rq / pp_) / ((1 - Rq) / (nn - pp_ - 1))) == 0, "一致")
check(T, "変数追加で R² 非減少（x1→x1,x2）", full["R2"] >= m1["R2"] and full["R2"] >= m2["R2"], (m1["R2"], m2["R2"], full["R2"]))
# 切片なしでは分解が崩れる
bn = np.linalg.lstsq(np.column_stack([x1, x2]), y, rcond=None)[0]
yhn = np.column_stack([x1, x2]) @ bn
cross = 2 * ((yhn - y.mean()) * (y - yhn)).sum()
check(T, "切片なしでは交差項≠0", abs(cross) > 1e-3, cross)
ch = charts(T)
check(T, "図1 scatter params", ch[0]["params"] == {"beta0": 1, "beta1": 0.8, "sigma": 1, "n": 50}, ch[0]["params"])
check(T, "図2 bar-data (83.5, 78.7262, 4.7738)", allclose(ch[1]["params"]["values"], [float(ST_s), float(SR_s), float(Se_s)]), ch[1]["params"]["values"])

# ============ adjusted-r-squared ============
T = "adjusted-r-squared"


def adjR2(Se, ST, n, p):
    return 1 - (Se / (n - p - 1)) / (ST / (n - 1))


check(T, "例1 x2 のみ Se=13.4167, R²=0.8393", close(m2["Se"], 13.4167) and close(m2["R2"], 0.8393), (m2["Se"], m2["R2"]))
check(T, "例1 adjR² 0.8088/0.8125/0.9200", close(adjR2(m1["Se"], 83.5, 8, 1), 0.8088) and close(adjR2(m2["Se"], 83.5, 8, 1), 0.8125) and close(adjR2(full["Se"], 83.5, 8, 2), 0.9200), (adjR2(m1["Se"], 83.5, 8, 1), adjR2(m2["Se"], 83.5, 8, 1), adjR2(full["Se"], 83.5, 8, 2)))
check(T, "例1 0.9548/11.9286", close(83.5 / 7, 11.9286), 83.5 / 7)
check(T, "例1 1−(1−0.9428)×7/5=0.9200", close(1 - (1 - 0.9428) * 7 / 5, 0.92, 5e-4), 1 - (1 - 0.9428) * 7 / 5)
check(T, "sympy: 2つの式の一致", sp.simplify(1 - (Rq * 0 + (1 - Rq) * STs / (nn - pp_ - 1)) / (STs / (nn - 1)) - (1 - (1 - Rq) * (nn - 1) / (nn - pp_ - 1))) == 0, "一致")
x3 = np.array([1, 2, 1, 2, 2, 1, 2, 1], float)
m3 = ols(np.column_stack([one, x1, x2, x3]), y)
check(T, "例2 R² 0.9428→0.9442", close(m3["R2"], 0.9442), m3["R2"])
check(T, "例2 adjR² 0.9200→0.9024", close(adjR2(m3["Se"], 83.5, 8, 3), 0.9024), adjR2(m3["Se"], 83.5, 8, 3))
check(T, "例2 x3 の t=0.3145", close(abs(m3["t"][3]), 0.3145), m3["t"][3])
# t²=部分F、|t|>1 ⇔ adjR² 上昇（ランダム 2000 例）
ok = True
cnt = 0
for _ in range(2000):
    nn_ = rng.integers(8, 30)
    k = rng.integers(1, 5)
    Xr = np.column_stack([np.ones(nn_), rng.normal(size=(nn_, k))])
    xr = rng.normal(size=nn_)
    yr = Xr @ rng.normal(size=k + 1) + rng.normal(size=nn_) + rng.normal() * xr
    a0 = ols(Xr, yr)
    a1 = ols(np.column_stack([Xr, xr]), yr)
    Fp = (a0["Se"] - a1["Se"]) / (a1["Se"] / (nn_ - k - 2))
    ok &= close(Fp, a1["t"][-1] ** 2, 1e-8 * max(1, Fp))
    up = adjR2(a1["Se"], a1["ST"], nn_, k + 1) > adjR2(a0["Se"], a0["ST"], nn_, k)
    ok &= (up == (abs(a1["t"][-1]) > 1))
    cnt += up
check(T, "ランダム2000例: t²=部分F、adjR² 上昇 ⇔ |t|>1", ok, f"上昇 {cnt} 例")
check(T, "adjR²≤R²、負になりうる（R²<p/(n−1)）", adjR2(0.95, 1, 20, 2) < 0 and 0.05 < 2 / 19, adjR2(0.95, 1, 20, 2))
check(T, "R²=p/(n−1) で adjR²=0（sympy）", sp.simplify((1 - (1 - pp_ / (nn - 1)) * (nn - 1) / (nn - pp_ - 1))) == 0, 0)
ch = charts(T)
p = ch[0]["params"]
r2m = np.array(p["series"][0]["y"])
adjm = np.array(p["series"][1]["y"])
pred_adj = 1 - (1 - r2m) * 19 / (19 - np.array(p["x"]))
check(T, "図 平均adjR² = 1−(1−平均R²)·19/(19−p)（線形なので平均でも成立、許容±0.002）", allclose(pred_adj, adjm, 0.002), np.round(pred_adj, 4))
# 図の再現シミュ（真の R² の設定は不明なので、adjR² が横ばい・R² の増分 ≈(1−adjR²)/19 を確認）
slope = np.diff(r2m).mean()
check(T, "図 R² の平均増分 ≈ (1−adjR²)/(n−1)=0.0293（許容±0.001）", close(slope, (1 - adjm.mean()) / 19, 0.001), (slope, (1 - adjm.mean()) / 19))
# 独自シミュで定性的に再現: n=20, 2変数有効、無関係変数を追加
Nr = 4000
r2s = np.zeros((Nr, 11))
ads = np.zeros((Nr, 11))
for r in range(Nr):
    Z = rng.normal(size=(20, 12))
    yv = Z[:, 0] + Z[:, 1] + rng.normal(size=20) * 1.6
    for j, pk in enumerate(range(2, 13)):
        o = ols(np.column_stack([np.ones(20), Z[:, :pk]]), yv)
        r2s[r, j] = o["R2"]
        ads[r, j] = adjR2(o["Se"], o["ST"], 20, pk)
check(T, "独自シミュ(4000回): R² は単調増、adjR² は横ばい（変動幅<0.01）", np.all(np.diff(r2s.mean(0)) > 0) and np.ptp(ads.mean(0)) < 0.01, (np.round(r2s.mean(0)[[0, -1]], 3), np.round(ads.mean(0)[[0, -1]], 3)))
check(T, "図 x=2..12, 4000回・n=20 の記述", p["x"] == list(range(2, 13)), p["x"])

# ============ overall-f-test-regression ============
T = "overall-f-test-regression"
F_obs = (float(SR_s) / 2) / (float(Se_s) / 5)
check(T, "例1 VR=39.3631, Ve=0.9548, F=41.23", close(float(SR_s) / 2, 39.3631) and close(float(Se_s) / 5, 0.9548) and close(F_obs, 41.23, 5e-3), (float(SR_s) / 2, F_obs))
check(T, "F0.05(2,5)=5.786, F0.01(2,5)=13.27", close(stats.f.ppf(.95, 2, 5), 5.786, 5e-4) and close(stats.f.ppf(.99, 2, 5), 13.27, 5e-3), (stats.f.ppf(.95, 2, 5), stats.f.ppf(.99, 2, 5)))
pF = stats.f.sf(F_obs, 2, 5)
check(T, "例1 p 値 0.00078（約0.0008）", close(pF, 0.00078, 5e-6), pF)
check(T, "例1 (0.9428/2)/(0.0572/5)=41.23（丸め R² で）", close((0.9428 / 2) / (0.0572 / 5), 41.2, 0.05), (0.9428 / 2) / (0.0572 / 5))
F2 = (0.35 / 4) / (0.65 / 25)
check(T, "例2 F=0.0875/0.026=3.365", close(F2, 3.365, 5e-4), F2)
check(T, "例2 F0.05(4,25)=2.759, p=0.025", close(stats.f.ppf(.95, 4, 25), 2.759, 5e-4) and close(stats.f.sf(F2, 4, 25), 0.025, 5e-4), (stats.f.ppf(.95, 4, 25), stats.f.sf(F2, 4, 25)))
J = sp.ones(8, 8) / 8
check(T, "HJ=J, H−J べき等, tr(H−J)=2, (H−J)(I−H)=O（厳密）", Hs * J == J and (Hs - J) ** 2 == Hs - J and (Hs - J).trace() == 2 and (Hs - J) * (I8 - Hs) == sp.zeros(8, 8), "成立")
check(T, "ST=y'(I−J)y, SR=y'(H−J)y, Se=y'(I−H)y（厳密）", (ys.T * (I8 - J) * ys)[0] == ST_s and (ys.T * (Hs - J) * ys)[0] == SR_s and (ys.T * (I8 - Hs) * ys)[0] == Se_s, "一致")
# H0 下のシミュレーション（上の 20 万回は H0 偽）
Y0 = 3 + rng.normal(size=(200000, 8))
B0 = Y0 @ X @ full["XtXi"]
R0 = Y0 - B0 @ X.T
Se0 = (R0 ** 2).sum(1)
SR0 = ((B0 @ X.T - Y0.mean(1, keepdims=True)) ** 2).sum(1)
F0 = (SR0 / 2) / (Se0 / 5)
rej = (F0 > stats.f.ppf(.95, 2, 5)).mean()
check(T, "シミュ H0: 棄却率 0.05（20万回、許容±0.002）", close(rej, 0.05, 0.002), rej)
check(T, "シミュ H0: SR/σ² の平均 2, SR と Se の相関≈0（許容 ±0.02, ±0.01）", close(SR0.mean(), 2, 0.02) and abs(np.corrcoef(SR0, Se0)[0, 1]) < 0.01, (SR0.mean(), np.corrcoef(SR0, Se0)[0, 1]))
# E[V_R] の式（H0 偽: 先の Ys, β=(1,1,1.5)）
SR1 = ((B @ X.T - Ys.mean(1, keepdims=True)) ** 2).sum(1)
Xc_ = X[:, 1:] - X[:, 1:].mean(0)
EVR = 1 + beta_true[1:] @ Xc_.T @ Xc_ @ beta_true[1:] / 2
check(T, "シミュ: E[V_R]=σ²+β'Xc'Xcβ/p（20万回、許容±0.09≈4SE）", close((SR1 / 2).mean(), EVR, 0.09), ((SR1 / 2).mean(), EVR))
nc = beta_true[1:] @ Xc_.T @ Xc_ @ beta_true[1:]
F1 = (SR1 / 2) / (Se_sim / 5)
pw_sim = (F1 > stats.f.ppf(.95, 2, 5)).mean()
pw = stats.ncf.sf(stats.f.ppf(.95, 2, 5), 2, 5, nc)
check(T, "シミュ: H0 偽で F は非心F（検出力、許容±0.003）", close(pw_sim, pw, 0.003), (pw_sim, pw))
check(T, "単回帰で F=t²（共通データ x1）", close((m1["ST"] - m1["Se"]) / (m1["Se"] / 6), m1["t"][1] ** 2, 1e-10), m1["t"][1] ** 2)
ch = charts(T)
check(T, "図 F(2,5), x 0〜15（F=41.23 は範囲外）", ch[0]["params"] == {"df1": 2, "df2": 5} and ch[0]["x"]["max"] < F_obs, ch[0]["params"])

# ============ regression-coefficient-t-test ============
T = "regression-coefficient-t-test"
check(T, "例1 t0.025(5)=2.571", close(stats.t.ppf(.975, 5), 2.571, 5e-4), stats.t.ppf(.975, 5))
check(T, "例1 SE (1.1999, 0.3482, 0.4481)", allclose(full["se"], [1.1999, 0.3482, 0.4481]), full["se"])
check(T, "例1 t (0.198, 3.009, 3.055)", allclose(full["t"], [0.198, 3.009, 3.055], 5e-4), full["t"])
pv = 2 * stats.t.sf(np.abs(full["t"]), 5)
check(T, "例1 p (0.851, 0.030, 0.028)", allclose(pv, [0.851, 0.030, 0.028], 5e-4), pv)
ci1 = full["b"][1] + np.array([-1, 1]) * stats.t.ppf(.975, 5) * full["se"][1]
check(T, "例1 β1 の CI [0.153, 1.943]", allclose(ci1, [0.153, 1.943], 5e-4), ci1)
t2 = (full["b"][2] - 1) / full["se"][2]
check(T, "例2 t=0.824, p=0.448", close(t2, 0.824, 5e-4) and close(2 * stats.t.sf(t2, 5), 0.448, 5e-4), (t2, 2 * stats.t.sf(t2, 5)))
ci2 = full["b"][2] + np.array([-1, 1]) * stats.t.ppf(.975, 5) * full["se"][2]
check(T, "例2 β2 の CI [0.217, 2.521]", allclose(ci2, [0.217, 2.521], 5e-4), ci2)
check(T, "Cov(β̂, e)=(X'X)^{-1}X'(I−H)=O（厳密）", XtXi_s * Xs.T * (I8 - Hs) == sp.zeros(3, 8), "O")
check(T, "単回帰 t=r√(n−2)/√(1−r²)", close(m1["t"][1], r1y * np.sqrt(6) / np.sqrt(1 - r1y ** 2), 1e-10), m1["t"][1])
# t²=部分F（x1 を外したモデルと）
Fpart = (m2["Se"] - full["Se"]) / (full["Se"] / 5)
check(T, "t²=部分F（x1 を外す）", close(Fpart, full["t"][1] ** 2, 1e-10), (Fpart, full["t"][1] ** 2))
check(T, "c_jj≠1/(X'X)_jj", not close(float(XtXi_s[1, 1]), 1 / 111, 1e-3), (float(XtXi_s[1, 1]), 1 / 111))
# シミュ: t 統計量 ~ t(5)
s_sim = np.sqrt(Se_sim / 5)
tsim = (B[:, 1] - beta_true[1]) / (s_sim * np.sqrt(full["XtXi"][1, 1]))
cover = (np.abs(tsim) <= stats.t.ppf(.975, 5)).mean()
check(T, "シミュ: 95%CI の被覆率 0.95（20万回、許容±0.002）", close(cover, 0.95, 0.002), cover)
ch = charts(T)
check(T, "図 t(5)、F(3.009)≈0.985、上側 ≈0.015", ch[0]["params"] == {"df": 5} and close(stats.t.cdf(3.009, 5), 0.985, 5e-4) and close(stats.t.sf(3.009, 5), 0.015, 5e-4), stats.t.cdf(3.009, 5))

# ============ confidence-and-prediction-interval-regression ============
T = "confidence-and-prediction-interval-regression"
x0 = np.array([1, 3, 4.0])
y0 = x0 @ full["b"]
q0 = sp.Matrix([1, 3, 4]).T * XtXi_s * sp.Matrix([1, 3, 4])
tq = stats.t.ppf(.975, 5)
check(T, "例1 ŷ0=8.857", close(y0, 8.857, 5e-4), y0)
check(T, "例1 q0=108/756=1/7", q0[0] == sp.Rational(1, 7) and (sp.Matrix([1, 3, 4]).T * adj * sp.Matrix([1, 3, 4]))[0] == 108, q0[0])
hw_c = tq * np.sqrt(full["s2"] / 7)
hw_p = tq * np.sqrt(full["s2"] * (1 + 1 / 7))
check(T, "例1 信頼区間 ±0.949 → [7.908, 9.807]", close(hw_c, 0.949, 5e-4) and allclose([y0 - hw_c, y0 + hw_c], [7.908, 9.807], 5e-4), (hw_c, y0 - hw_c, y0 + hw_c))
check(T, "例1 予測区間 ±2.685 → [6.172, 11.542]", close(hw_p, 2.685, 5e-4) and allclose([y0 - hw_p, y0 + hw_p], [6.172, 11.542], 5e-4), (hw_p, y0 - hw_p, y0 + hw_p))
check(T, "例1 幅の比 約2.8倍", close(hw_p / hw_c, 2.8, 0.05), hw_p / hw_c)
x0b = np.array([1, 6, 3.0])
q0b = float(x0b @ full["XtXi"] @ x0b)
check(T, "例2 ŷ0=10.631, q0=1.877", close(x0b @ full["b"], 10.631, 5e-4) and close(q0b, 1.877, 5e-4), (x0b @ full["b"], q0b))
check(T, "例2 半幅 3.441, 4.260", close(tq * np.sqrt(full["s2"] * q0b), 3.441, 5e-4) and close(tq * np.sqrt(full["s2"] * (1 + q0b)), 4.260, 5e-4), (tq * np.sqrt(full["s2"] * q0b), tq * np.sqrt(full["s2"] * (1 + q0b))))
check(T, "平均の点で q0=1/n（最小）", close(np.array([1, 3.375, 4]) @ full["XtXi"] @ np.array([1, 3.375, 4]), 1 / 8, 1e-12), 1 / 8)
check(T, "データ点で q0=h_ii", close(X[0] @ full["XtXi"] @ X[0], float(hd[0]), 1e-12), "一致")
# 単回帰の q0 公式
check(T, "単回帰 q0=1/n+(x0−x̄)²/Sxx", close(np.array([1, 5.0]) @ m1["XtXi"] @ np.array([1, 5.0]), 1 / 8 + (5 - 3.375) ** 2 / Sxx, 1e-12), "一致")
# 図: 単回帰の帯
ch = charts(T)
p = ch[0]["params"]
xg = np.array(p["x"], float)
t6 = stats.t.ppf(.975, 6)
fit = m1["b"][0] + m1["b"][1] * xg
qg = 1 / 8 + (xg - 3.375) ** 2 / Sxx
s1 = np.sqrt(m1["s2"])
cl, cu = fit - t6 * s1 * np.sqrt(qg), fit + t6 * s1 * np.sqrt(qg)
pl, pu = fit - t6 * s1 * np.sqrt(1 + qg), fit + t6 * s1 * np.sqrt(1 + qg)
for lab, arr, ser in [("回帰直線", fit, 0), ("CI下", cl, 1), ("CI上", cu, 2), ("PI下", pl, 3), ("PI上", pu, 4)]:
    check(T, f"図 {lab}（自由度6）", allclose(arr, p["series"][ser]["y"], 5e-4), np.round(arr, 3))
check(T, "図 ラベル 2.925 + 1.874x", close(m1["b"][0], 2.925, 5e-4) and close(m1["b"][1], 1.874, 5e-4), m1["b"])
# シミュ: 予測区間の被覆率
yf = X @ beta_true
y0sim = x0 @ beta_true + rng.normal(size=200000)
yhat0 = B @ x0
cov_p = (np.abs(y0sim - yhat0) <= tq * s_sim * np.sqrt(1 + 1 / 7)).mean()
cov_c = (np.abs(x0 @ beta_true - yhat0) <= tq * s_sim * np.sqrt(1 / 7)).mean()
check(T, "シミュ: 予測区間・信頼区間の被覆率 0.95（20万回、許容±0.002）", close(cov_p, .95, .002) and close(cov_c, .95, .002), (cov_p, cov_c))
check(T, "n→∞ で予測区間幅 → 2z σ（t(10^6) ≈ z）", close(stats.t.ppf(.975, 10 ** 6), stats.norm.ppf(.975), 1e-5), stats.norm.ppf(.975))

# ============ multicollinearity ============
T = "multicollinearity"
u1 = np.arange(1, 7, dtype=float)
u2 = np.array([1.1, 1.9, 3.0, 4.1, 4.9, 6.0])
v = np.array([3.3, 4.8, 7.1, 8.9, 11.0, 12.9])
U = np.column_stack([np.ones(6), u1, u2])
mc = ols(U, v)
r12m = np.corrcoef(u1, u2)[0, 1]
check(T, "例1 r12=0.9989, VIF=454.5", close(r12m, 0.9989) and close(1 / (1 - r12m ** 2), 454.5, 0.05), (r12m, 1 / (1 - r12m ** 2)))
check(T, "例1 β̂=(1.1273,1.1455,0.8182)", allclose(mc["b"], [1.1273, 1.1455, 0.8182]), mc["b"])
Fm = (mc["R2"] / 2) / ((1 - mc["R2"]) / 3)
check(T, "例1 R²=0.9985, F=1021.5, p<0.001", close(mc["R2"], 0.9985) and close(Fm, 1021.5, 0.05) and stats.f.sf(Fm, 2, 3) < 0.001, (mc["R2"], Fm, stats.f.sf(Fm, 2, 3)))
pm = 2 * stats.t.sf(np.abs(mc["t"]), 3)
check(T, "例1 t=1.242 (p=0.302), 0.878 (p=0.444)", allclose(mc["t"][1:], [1.242, 0.878], 5e-4) and allclose(pm[1:], [0.302, 0.444], 5e-4), (mc["t"][1:], pm[1:]))
v2 = v.copy()
v2[1] = 5.1
v2[3] = 8.6
mc2 = ols(U, v2)
check(T, "例2 β̂1=2.8636, β̂2=−0.9545", allclose(mc2["b"][1:], [2.8636, -0.9545]), mc2["b"][1:])
check(T, "例2 β̂1+β̂2: 1.9636→1.9091", close(mc["b"][1:].sum(), 1.9636) and close(mc2["b"][1:].sum(), 1.9091), (mc["b"][1:].sum(), mc2["b"][1:].sum()))
# 2変数の逆行列 (1,1) 成分
s11, s22, s12_ = sp.symbols("S11 S22 S12", positive=True)
inv11 = sp.Matrix([[s11, s12_], [s12_, s22]]).inv()[0, 0]
check(T, "sympy: (S^{-1})_11=1/{S11(1−r12²)}", sp.simplify(inv11 - 1 / (s11 * (1 - s12_ ** 2 / (s11 * s22)))) == 0, inv11)
Mi = sp.Matrix([[s11, s12_], [s12_, s22]]).inv()
check(T, "sympy: corr(β̂1,β̂2)=−r12", sp.simplify(Mi[0, 1] ** 2 / (Mi[0, 0] * Mi[1, 1]) - s12_ ** 2 / (s11 * s22)) == 0 and sp.simplify(Mi[0, 1] + s12_ / (s11 * s22 - s12_ ** 2)) == 0, "−r12")
check(T, "V[β̂j]=σ²/{Sjj(1−Rj²)}（例1 データ）", close(mc["XtXi"][1, 1], 1 / (((u1 - u1.mean()) ** 2).sum() * (1 - r12m ** 2)), 1e-9), mc["XtXi"][1, 1])
check(T, "当てはめ値は安定（例1→例2 で ŷ の最大変化 < 0.3）", np.max(np.abs(U @ mc["b"] - U @ mc2["b"])) < 0.3, np.max(np.abs(U @ mc["b"] - U @ mc2["b"])))
ch = charts(T)
check(T, "図1 bar-data (1.1455, 0.8182, 2.8636, −0.9545)", allclose(ch[0]["params"]["values"], [*mc["b"][1:], *mc2["b"][1:]]), ch[0]["params"]["values"])
check(T, "図1 キャプション 和 1.96, 1.91", close(mc["b"][1:].sum(), 1.96, 5e-3) and close(mc2["b"][1:].sum(), 1.91, 5e-3), "一致")
check(T, "図2 bvnorm rho=0.95, n=200", ch[1]["params"] == {"rho": 0.95, "n": 200}, ch[1]["params"])
# 不偏性は保たれる（シミュ）
Bm = (np.array([1, 1, 1.0]) @ U.T + rng.normal(0, 0.3, size=(200000, 6))) @ U @ mc["XtXi"]
check(T, "シミュ: 共線性下でも E[β̂]=β（20万回、許容±4SE）", np.all(np.abs(Bm.mean(0) - 1) < 4 * Bm.std(0) / np.sqrt(200000)), Bm.mean(0))

# ============ variance-inflation-factor ============
T = "variance-inflation-factor"
r12 = np.corrcoef(x1, x2)[0, 1]
vif = 1 / (1 - r12 ** 2)
check(T, "例1 r12=0.7770, r²=0.6038, VIF=2.524", close(r12, 0.7770) and close(r12 ** 2, 0.6038) and close(vif, 2.524, 5e-4), (r12, r12 ** 2, vif))
check(T, "例1 (0.9548/19.875)×2.524=0.1212 = s²·96/756", close(full["s2"] / 19.875 * vif, 0.1212) and close(full["s2"] * 96 / 756, 0.1212), (full["s2"] / 19.875 * vif, full["s2"] * 96 / 756))
check(T, "例1 √VIF=1.589", close(np.sqrt(vif), 1.589, 5e-4), np.sqrt(vif))
check(T, "VIF=(R^{-1})_jj（共通データ）", allclose(np.diag(np.linalg.inv(Rm)), [vif, vif], 1e-10), np.diag(np.linalg.inv(Rm)))
Rex = np.array([[1, .6, .5], [.6, 1, .8], [.5, .8, 1]])
dinv = np.diag(np.linalg.inv(Rex))
check(T, "例2 (R^{-1})_jj=1.565, 3.261, 2.783", allclose(dinv, [1.565, 3.261, 2.783], 5e-4), dinv)
check(T, "例2 R2²=1−1/3.261=0.693", close(1 - 1 / dinv[1], 0.693, 5e-4), 1 - 1 / dinv[1])
rr = np.array([.6, .5])
R1sq = rr @ np.linalg.inv(np.array([[1, .8], [.8, 1]])) @ rr
check(T, "例2 R1²=0.3611, 1/0.6389=1.565", close(R1sq, 0.3611) and close(1 / (1 - R1sq), 1.565, 5e-4), R1sq)
# ブロック逆行列（記号）
r_a, r_b, r_c = sp.symbols("r_a r_b r_c")
Rsym = sp.Matrix([[1, r_a, r_b], [r_a, 1, r_c], [r_b, r_c, 1]])
rv = sp.Matrix([r_a, r_b])
Rj = sp.Matrix([[1, r_c], [r_c, 1]])
check(T, "sympy: (R^{-1})_11=1/(1−r'R_(1)^{-1}r)", sp.simplify(Rsym.inv()[0, 0] - 1 / (1 - (rv.T * Rj.inv() * rv)[0])) == 0, "一致")
# FWL: β̂j = Σ ũ y/Σ ũ²、Σũ²=Sjj(1−Rj²)
ut = x1 - X2 @ np.linalg.lstsq(X2, x1, rcond=None)[0]
check(T, "FWL: β̂1=Σũy/Σũ²", close(ut @ y / (ut @ ut), full["b"][1], 1e-12), ut @ y / (ut @ ut))
check(T, "Σũ²=S11(1−R1²)", close(ut @ ut, S11 * (1 - r12 ** 2), 1e-10), ut @ ut)
# ランダム 3〜5 変数で VIF = 1/(1−Rj²) = (R^{-1})_jj = Sjj·(X'X)^{-1}_jj
ok = True
for _ in range(200):
    kk = rng.integers(3, 6)
    Z = rng.normal(size=(40, kk)) @ rng.normal(size=(kk, kk))
    Rz = np.corrcoef(Z.T)
    Xz = np.column_stack([np.ones(40), Z])
    XzI = np.linalg.inv(Xz.T @ Xz)
    for j in range(kk):
        oth = np.column_stack([np.ones(40), np.delete(Z, j, 1)])
        Rj2 = ols(oth, Z[:, j])["R2"]
        Sjj = ((Z[:, j] - Z[:, j].mean()) ** 2).sum()
        ok &= close(1 / (1 - Rj2), np.linalg.inv(Rz)[j, j], 1e-6 * (1 / (1 - Rj2))) and close(Sjj * XzI[j + 1, j + 1], 1 / (1 - Rj2), 1e-6 * (1 / (1 - Rj2)))
check(T, "ランダム200例: 1/(1−Rj²)=(R^{-1})_jj=Sjj·c_jj", ok, ok)
check(T, "r12=0.9 で 5.26、0.95 で 10.26", close(1 / (1 - .81), 5.26, 5e-3) and close(1 / (1 - .9025), 10.26, 5e-3), (1 / (1 - .81), 1 / (1 - .9025)))
check(T, "VIF=4 で SE 2倍", np.sqrt(4) == 2, 2)
ch = charts(T)
p = ch[0]["params"]
Rg = np.array(p["x"])
check(T, "図 VIF=1/(1−R_j²)（3桁丸め）", allclose(1 / (1 - Rg ** 2), p["series"][0]["y"], 5e-4), np.round(1 / (1 - Rg ** 2), 3))
check(T, "図 √VIF（3桁丸め）", allclose(1 / np.sqrt(1 - Rg ** 2), p["series"][1]["y"], 5e-4), np.round(1 / np.sqrt(1 - Rg ** 2), 3))
check(T, "図 キャプション R=0.7 で VIF<2、0.95 で ≈10・SE 約3.2倍", 1 / (1 - .49) < 2 and close(1 / (1 - .9025), 10, 0.3) and close(1 / np.sqrt(1 - .9025), 3.2, 0.01), "成立")

# ============ standardized-partial-regression-coefficient ============
T = "standardized-partial-regression-coefficient"
s1_, s2_, sy_ = x1.std(ddof=1), x2.std(ddof=1), y.std(ddof=1)
check(T, "例1 s1=1.6850, s2=1.3093, sy=3.4538", allclose([s1_, s2_, sy_], [1.6850, 1.3093, 3.4538]), (s1_, s2_, sy_))
bst = full["b"][1:] * np.array([s1_, s2_]) / sy_
check(T, "例1 β*=0.5111, 0.5190", allclose(bst, [0.5111, 0.5190]), bst)
r2y = np.corrcoef(x2, y)[0, 1]
check(T, "例1 r12=0.7770, r1y=0.9144, r2y=0.9161", allclose([r12, r1y, r2y], [0.7770, 0.9144, 0.9161]), (r12, r1y, r2y))
check(T, "例1 2変数公式 0.2025/0.3962=0.5111", close(r1y - r12 * r2y, 0.2025) and close(1 - r12 ** 2, 0.3962) and close((r1y - r12 * r2y) / (1 - r12 ** 2), 0.5111), (r1y - r12 * r2y, 1 - r12 ** 2))
check(T, "例1 β2* の2変数公式", close((r2y - r12 * r1y) / (1 - r12 ** 2), bst[1], 1e-12), (r2y - r12 * r1y) / (1 - r12 ** 2))
check(T, "例1 R²=Σβ*r=0.9428", close(bst @ np.array([r1y, r2y]), 0.9428) and close(bst @ np.array([r1y, r2y]), full["R2"], 1e-12), bst @ np.array([r1y, r2y]))
zs = lambda v: (v - v.mean()) / v.std(ddof=1)
mz = ols(np.column_stack([one, zs(x1), zs(x2)]), zs(y))
check(T, "標準化して回帰した係数と一致・切片0", allclose(mz["b"], [0, *bst], 1e-12), mz["b"])
check(T, "標準化しても t 値・R² 不変", allclose(mz["t"][1:], full["t"][1:], 1e-10) and close(mz["R2"], full["R2"], 1e-12), mz["t"][1:])
mu10 = ols(np.column_stack([one, 10 * x1, x2]), y)
check(T, "例2 万円単位: β1=0.10476, s1=16.850, β* 不変", close(mu10["b"][1], 0.10476, 5e-6) and close(10 * s1_, 16.850, 5e-4) and close(mu10["b"][1] * 10 * s1_ / sy_, 0.5111), mu10["b"][1])
check(T, "説明変数1つなら β*=r", close(m1["b"][1] * s1_ / sy_, r1y, 1e-12), r1y)
check(T, "|β*|>1 になりうる（r12=0.9, r1y=0.9, r2y=0.6）", abs((0.9 - 0.9 * 0.6) / (1 - 0.81)) > 1, (0.9 - 0.9 * 0.6) / (1 - 0.81))
ch = charts(T)
check(T, "図 bar-data (1.0476,1.3690,0.5111,0.5190)", allclose(ch[0]["params"]["values"], [*full["b"][1:], *bst]), ch[0]["params"]["values"])
check(T, "図 キャプション 面積のほうが3割ほど大きい（1.369/1.0476）", close(full["b"][2] / full["b"][1], 1.3, 0.05), full["b"][2] / full["b"][1])

# ============ dummy-variable ============
T = "dummy-variable"
dx = np.array([3, 4, 5, 6, 1, 2, 3, 4], float)
dD = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
dy = np.array([6, 8, 9, 11, 7, 8, 10, 11], float)
md = ols(np.column_stack([one, dx, dD]), dy)
check(T, "例1 ŷ=1.75+1.5x+3.5D", allclose(md["b"], [1.75, 1.5, 3.5], 1e-10), md["b"])
check(T, "例1 SE(γ)=0.3, t=11.67, p<0.001", close(md["se"][2], 0.3, 5e-5) and close(md["t"][2], 11.67, 5e-3) and 2 * stats.t.sf(md["t"][2], 5) < 0.001, (md["se"][2], md["t"][2], 2 * stats.t.sf(md["t"][2], 5)))
check(T, "例2 平均売上 駅前 9.0、郊外 8.5、面積平均 2.5 対 4.5", dy[dD == 1].mean() == 9 and dy[dD == 0].mean() == 8.5 and dx[dD == 1].mean() == 2.5 and dx[dD == 0].mean() == 4.5, "一致")
check(T, "例2 0.5+1.5×(4.5−2.5)=3.5", 0.5 + 1.5 * 2 == 3.5, 3.5)
# ダミーだけのモデル = 2標本 t 検定
mdo = ols(np.column_stack([one, dD]), dy)
tt = stats.ttest_ind(dy[dD == 1], dy[dD == 0])
check(T, "ダミーのみ: β0=ȳA, γ=ȳB−ȳA, t=2標本t", close(mdo["b"][0], 8.5, 1e-12) and close(mdo["b"][1], 0.5, 1e-12) and close(mdo["t"][1], tt.statistic, 1e-10), (mdo["b"], tt.statistic))
# 3群で F=一元配置
g = rng.normal(size=12) + np.repeat([0, 1, 2], 4)
Dg = np.column_stack([np.ones(12), np.repeat([0, 1, 0], 4), np.repeat([0, 0, 1], 4)])
mg = ols(Dg, g)
Fg = (mg["R2"] / 2) / ((1 - mg["R2"]) / 9)
check(T, "3群ダミー回帰の F=一元配置分散分析の F", close(Fg, stats.f_oneway(g[:4], g[4:8], g[8:]).statistic, 1e-10), Fg)
check(T, "ダミー変数の罠: k 個＋切片で rank 落ち", np.linalg.matrix_rank(np.column_stack([np.ones(8), dD, 1 - dD])) == 2, "rank 2")
check(T, "基準を変えても当てはめ値は同じ", allclose(ols(np.column_stack([one, dx, 1 - dD]), dy)["b"][[1, 2]], [1.5, -3.5], 1e-10), ols(np.column_stack([one, dx, 1 - dD]), dy)["b"])
ch = charts(T)
p = ch[0]["params"]
xg = np.array(p["x"], float)
check(T, "図 2本の直線 1.75+1.5x, 5.25+1.5x", allclose(1.75 + 1.5 * xg, p["series"][0]["y"], 1e-9) and allclose(5.25 + 1.5 * xg, p["series"][1]["y"], 1e-9), "一致")

# ============ interaction-term ============
T = "interaction-term"
ix = np.array([1, 2, 3, 4, 1, 2, 3, 4], float)
iD = dD
iy = np.array([3, 5, 6, 8, 4, 7, 11, 13], float)
mi = ols(np.column_stack([one, ix, iD, ix * iD]), iy)
check(T, "例1 ŷ=1.5+1.6x−0.5D+1.5xD", allclose(mi["b"], [1.5, 1.6, -0.5, 1.5], 1e-10), mi["b"])
sl0 = np.polyfit(ix[:4], iy[:4], 1)
sl1 = np.polyfit(ix[4:], iy[4:], 1)
check(T, "例1 群別単回帰 傾き 1.6, 3.1、切片 1.5, 1.0", allclose([sl0[0], sl1[0], sl0[1], sl1[1]], [1.6, 3.1, 1.5, 1.0], 1e-10), (sl0, sl1))
pi_ = 2 * stats.t.sf(mi["t"][3], 4)
check(T, "例1 SE=0.3, t=5.0, t0.025(4)=2.776, p=0.0075", close(mi["se"][3], 0.3, 1e-10) and close(mi["t"][3], 5, 1e-9) and close(stats.t.ppf(.975, 4), 2.776, 5e-4) and close(pi_, 0.0075, 5e-5), (mi["se"][3], mi["t"][3], pi_))
mn = ols(np.column_stack([one, ix, iD]), iy)
check(T, "例2 交互作用なし ŷ=−0.375+2.35x+3.25D, Se=6.525", allclose(mn["b"], [-0.375, 2.35, 3.25], 1e-10) and close(mn["Se"], 6.525, 1e-10), (mn["b"], mn["Se"]))
check(T, "例2 交互作用あり Se=0.9", close(mi["Se"], 0.9, 1e-10), mi["Se"])
Fi = (mn["Se"] - mi["Se"]) / (mi["Se"] / 4)
check(T, "例2 F=5.625/0.225=25=t²", close(Fi, 25, 1e-9) and close(Fi, mi["t"][3] ** 2, 1e-9), Fi)
# 中心化で β3 と t 不変
cx = ix - ix.mean()
cD = iD - iD.mean()
mcn = ols(np.column_stack([one, cx, cD, cx * cD]), iy)
check(T, "中心化しても β3・t 不変", close(mcn["b"][3], 1.5, 1e-10) and close(mcn["t"][3], 5, 1e-9), mcn["b"])
check(T, "中心化後の β1 = D 平均での x の効果 1.6+1.5×0.5", close(mcn["b"][1], 1.6 + 1.5 * 0.5, 1e-10), mcn["b"][1])
check(T, "中心化で積と主効果の相関が弱まる", abs(np.corrcoef(cx * cD, cx)[0, 1]) < abs(np.corrcoef(ix * iD, ix)[0, 1]), (np.corrcoef(ix * iD, ix)[0, 1], np.corrcoef(cx * cD, cx)[0, 1]))
xs_, b1_, b3_, x2_ = sp.symbols("x1 beta1 beta3 x2")
check(T, "sympy: ∂E[y]/∂x1=β1+β3x2", sp.diff(b1_ * xs_ + b3_ * xs_ * x2_, xs_) == b1_ + b3_ * x2_, "一致")
ch = charts(T)
p = ch[0]["params"]
xg = np.array(p["x"], float)
check(T, "図 2本の直線 1.5+1.6x, 1.0+3.1x", allclose(1.5 + 1.6 * xg, p["series"][0]["y"], 1e-9) and allclose(1.0 + 3.1 * xg, p["series"][1]["y"], 1e-9), "一致")
check(T, "図 キャプション 傾きの差 1.5、約2倍（3.1/1.6）", close(3.1 - 1.6, 1.5, 1e-12) and close(3.1 / 1.6, 2, 0.1), 3.1 / 1.6)

print()
print(f"OK {sum(results)} / NG {len(results) - sum(results)}")
