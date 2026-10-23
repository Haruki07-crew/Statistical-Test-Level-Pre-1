"""第23章（判別分析）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, optimize

q = Q(23)
rng = np.random.default_rng(23)

# --- 線形判別分析 ---
x = sp.Matrix(sp.symbols("x1 x2"))
S1 = sp.Matrix([[1, 0], [0, 1]]); S2 = sp.Matrix([[2, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
m1 = sp.Matrix([0, 0]); m2 = sp.Matrix([1, 1])
lq = lambda m, S: (-(x - m).T * S.inv() * (x - m) / 2)[0] - sp.log(S.det()) / 2
diff = sp.expand(lq(m2, S2) - lq(m1, S1))
quad = sp.Poly(diff, *x).total_degree()
diff_eq = sp.expand(lq(m2, S1) - lq(m1, S1))
q.check("q23-linear-discriminant-analysis-1", quad == 2 and sp.Poly(diff_eq, *x).total_degree() == 1,
        "Σ が異なると2次、等しいと1次（記述は誤り）")
mu1 = np.array([0, 0.]); mu2 = np.array([2, 1.]); Sig = np.array([[1, .5], [.5, 1]])
xv = np.array([1.5, 0])
z = (mu2 - mu1) @ np.linalg.solve(Sig, xv - (mu1 + mu2) / 2)
q.num("q23-linear-discriminant-analysis-2", z)
# 事前確率つき：ベイズ則を直接計算
f1 = stats.multivariate_normal(mu1, Sig).pdf(xv); f2 = stats.multivariate_normal(mu2, Sig).pdf(xv)
q.check("q23-linear-discriminant-analysis-2", f2 > f1, "等事前確率なら G2（解説）")
q.check("q23-linear-discriminant-analysis-3", abs(z - 1) < 1e-12 and 0.8 * f1 > 0.2 * f2 and abs(np.log(4) - 1.386) < 5e-4
        and z > 0 and z > np.log(0.25), "π=(0.8,0.2) では G1、しきい値 log4=1.386；誤答の基準では G2 になる")

# --- フィッシャー ---
xb1 = np.array([3, 2.]); xb2 = np.array([1, 1.]); S = np.diag([2, .5])
a = np.linalg.solve(S, xb1 - xb2)
par = lambda u, v: abs(u[0] * v[1] - u[1] * v[0]) < 1e-12 and u @ v > 0
cands = [np.array(c, float) for c in [(1, 2), (2, 1), (4, .5), (1, -2)]]
q.check("q23-fisher-linear-discriminant-function-1", [par(a, c) for c in cands] == [True, False, False, False]
        and par(S @ (xb1 - xb2), cands[2]), "正解 (1,2)、(4,0.5) は S を掛けた誤り")
lam = lambda v: (v @ (xb1 - xb2)) ** 2 / (v @ S @ v)
res = optimize.minimize(lambda t: -lam(np.array([np.cos(t[0]), np.sin(t[0])])), [0.3])
q.num("q23-fisher-linear-discriminant-function-2", -res.fun)
q.check("q23-fisher-linear-discriminant-function-2", abs((xb1 - xb2) @ np.linalg.solve(S, xb1 - xb2) - 4) < 1e-12, "D²=4")
pts = rng.normal(size=(1000, 2)) * 2 + 2
y1 = (pts - (xb1 + xb2) / 2) @ a; y3 = (pts - (xb1 + xb2) / 2) @ (3 * a)
q.check("q23-fisher-linear-discriminant-function-3", np.array_equal(y1 > 0, y3 > 0) and abs(lam(a) - lam(3 * a)) < 1e-12,
        "3倍しても λ と符号は不変（正しい）")

# --- マハラノビス距離 ---
q.num("q23-mahalanobis-distance-1", abs(35 - 50) / 10)
Sig = np.array([[4, 2], [2, 4.]])
D = lambda v: np.sqrt(v @ np.linalg.solve(Sig, v))
q.num("q23-mahalanobis-distance-2", D(np.array([2, -2.])))
q.check("q23-mahalanobis-distance-2", abs(D(np.array([2, 2.])) - 1.15) < 5e-3, "(2,2) は 1.15")
q.num("q23-mahalanobis-distance-3", stats.chi2.sf(9, 3))
q.check("q23-mahalanobis-distance-3", abs(2 * stats.norm.sf(3) - 0.0027) < 5e-5, "1変数 3σ 外は 0.0027")

# --- 二次判別 ---
p = 5
q.num("q23-quadratic-discriminant-analysis-1", 2 * p * (p + 1) // 2)
c = optimize.brentq(lambda t: stats.norm.pdf(t, 0, 1) - stats.norm.pdf(t, 0, 2), 0.1, 5)
q.num("q23-quadratic-discriminant-analysis-2", c)
q.check("q23-quadratic-discriminant-analysis-2", stats.norm.pdf(0, 0, 1) > stats.norm.pdf(0, 0, 2)
        and stats.norm.pdf(3, 0, 1) < stats.norm.pdf(3, 0, 2), "中央が G1、両側が G2")

# --- ベイズ判別 ---
q.num("q23-bayes-discriminant-rule-1", 0.15 / (0.15 + 0.07))
thr = 0.5 * 1 / (0.5 * 5)
post1 = 0.5 * 1 / (0.5 * 1 + 0.5 * 0.4)
risk_G1 = 0.5 * 0.4 * 5  # G1 と判別したときの期待損失（G2 の個体を誤る）∝ π2 f2 c(1|2)、f1=1 とおく
risk_G2 = 0.5 * 1 * 1
q.check("q23-bayes-discriminant-rule-2", abs(thr - 0.2) < 1e-12 and risk_G2 < risk_G1 and abs(post1 - 0.714) < 5e-4,
        "期待損失は G2 が小さい（0.5 < 1.0）、しきい値0.2、事後確率0.714")
c = optimize.brentq(lambda t: 0.75 * stats.norm.pdf(t) - 0.25 * stats.norm.pdf(t, 2), -5, 5)
q.num("q23-bayes-discriminant-rule-3", c)
q.check("q23-bayes-discriminant-rule-3", abs(c - (1 + np.log(3) / 2)) < 1e-9, "1+log3/2 と一致")

# --- 誤判別率 ---
q.num("q23-misclassification-rate-1", stats.norm.cdf(-1))
d = np.array([1, -1.]); Sig = np.array([[1, .6], [.6, 1]])
Delta = np.sqrt(d @ np.linalg.solve(Sig, d))
q.num("q23-misclassification-rate-2", stats.norm.cdf(-Delta / 2))
# シミュレーションでも確認
X1 = rng.multivariate_normal(d, Sig, 200000); w = np.linalg.solve(Sig, d)
sim = np.mean((X1 - d / 2) @ w < 0)
q.check("q23-misclassification-rate-2", abs(Delta ** 2 - 5) < 1e-12 and abs(sim - 0.132) < 0.003
        and abs(stats.norm.cdf(-np.sqrt(2) / 2) - 0.240) < 5e-4, "Δ²=5、シミュレーション一致、相関無視なら0.240")
# 見かけの誤判別率 < 真の誤判別率（小標本 LDA のシミュレーション）
pp, n = 5, 10
mu = np.zeros(pp); mu2 = np.r_[1.0, np.zeros(pp - 1)]
app, true = [], []
for _ in range(400):
    A = rng.normal(size=(n, pp)) + mu; B = rng.normal(size=(n, pp)) + mu2
    Sp = (np.cov(A, rowvar=False) + np.cov(B, rowvar=False)) / 2
    wv = np.linalg.solve(Sp, B.mean(0) - A.mean(0)); mid = (A.mean(0) + B.mean(0)) / 2
    app.append((np.sum((A - mid) @ wv > 0) + np.sum((B - mid) @ wv < 0)) / (2 * n))
    s = np.sqrt(wv @ wv)  # 真の Σ=I
    m0 = (mu - mid) @ wv; m1 = (mu2 - mid) @ wv
    true.append((stats.norm.sf(-m0 / s) + stats.norm.cdf(-m1 / s)) / 2)
q.check("q23-misclassification-rate-3", np.mean(app) < np.mean(true) - 0.05,
        f"見かけ {np.mean(app):.3f} < 真 {np.mean(true):.3f}（正しい）")

# --- 混同行列 ---
TP, FN, FP, TN = 40, 10, 20, 130
q.num("q23-confusion-matrix-1", (FP + FN) / 200)
M = np.array([[30, 5, 5], [4, 40, 6], [1, 9, 50]])
cands = [M[1, 1] / M[:, 1].sum(), M[1, 1] / M[1].sum(), M[1, 1] / M.sum(), M[2, 2] / M[2].sum()]
q.check("q23-confusion-matrix-2", abs(cands[0] - 0.741) < 5e-4 and len(set(np.round(cands, 3))) == 4
        and abs(np.trace(M) / M.sum() - 0.8) < 1e-12, "正解 40/54=0.741、誤答はいずれも別の値")

# --- 感度と特異度 ---
q.num("q23-sensitivity-and-specificity-1", TN / (TN + FP))
q.num("q23-sensitivity-and-specificity-2", 0.9 * 0.02 / (0.9 * 0.02 + 0.05 * 0.98))
J = stats.norm.cdf(0.5) + stats.norm.cdf(1.5) - 1
q.num("q23-sensitivity-and-specificity-3", J)
ts = np.linspace(-2, 4, 60001)
Js = stats.norm.cdf(2 - ts) + stats.norm.cdf(ts) - 1
q.check("q23-sensitivity-and-specificity-3", abs(ts[np.argmax(Js)] - 1) < 1e-3 and abs(Js.max() - 0.683) < 5e-4, "最大は t=1 で 0.683")

# --- ROC と AUC ---
s0 = rng.normal(size=100000); s1 = rng.normal(size=100000)
q.check("q23-roc-curve-and-auc-1", abs(np.mean(s1 > s0) - 0.5) < 0.01, "同分布なら AUC≈0.5（完全判別ではない＝誤り）")
pos = np.array([.9, .7, .6]); neg = np.array([.8, .6, .5, .4])
auc = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
q.num("q23-roc-curve-and-auc-2", auc)
U = stats.mannwhitneyu(pos, neg).statistic
q.check("q23-roc-curve-and-auc-2", abs(U / 12 - auc) < 1e-12, "マン・ホイットニーの U/12 と一致")
q.num("q23-roc-curve-and-auc-3", stats.norm.cdf(1.5 / np.sqrt(2)))

# --- 適合率・再現率・F値 ---
P, R = 30 / 40, 30 / 50
q.num("q23-precision-recall-f-measure-1", 2 * P * R / (P + R))
q.check("q23-precision-recall-f-measure-1", abs(2 * P * R / (P + R) - 60 / 90) < 1e-12, "2TP/(2TP+FP+FN) と一致")
q.check("q23-precision-recall-f-measure-2", abs((P + R) / 2 - 0.675) < 1e-12 and abs((P + R) / 2 - 2 * P * R / (P + R)) > 0.005,
        "算術平均 0.675 ≠ F1 0.667（記述は誤り）")

# --- k 近傍法 ---
X = np.array([1, 2, 6, 4, 5, 7, 8.]); y = np.array(list("AAABBBB"))
def knn(x0, k, X=X):
    idx = np.argsort(np.abs(X - x0) if X.ndim == 1 else np.linalg.norm(X - x0, axis=1))[:k]
    lab, cnt = np.unique(y[idx], return_counts=True)
    return lab[np.argmax(cnt)]
q.check("q23-k-nearest-neighbors-1", (knn(5.8, 1), knn(5.8, 3)) == ("A", "B"), "k=1 で A、k=3 で B")
# 単位変更で結果が変わる例（2変数）
X2 = np.array([[0, 0], [0.5, 3], [3, 0.2]]); lab2 = np.array(["A", "B", "B"])
x0 = np.array([1.0, 1.0])
nn = lambda Z, z: lab2[np.argmin(np.linalg.norm(Z - z, axis=1))]
scale = np.array([100, 1.])
q.check("q23-k-nearest-neighbors-2", nn(X2, x0) != nn(X2 * scale, x0 * scale), "単位変更で最近傍が変わる例あり（記述は誤り）")

# --- SVM ---
q.num("q23-support-vector-machine-1", 2 / np.linalg.norm([3, 4]))
pts = [((1, 1), 1), ((0, 2), -1), ((2, .5), -1)]
f = lambda v: v[0] - v[1] + 0.5
q.num("q23-support-vector-machine-2", sum(max(0, 1 - yy * f(v)) for v, yy in pts))
def hard_svm(X, Y):
    cons = [{"type": "ineq", "fun": (lambda t, i=i: Y[i] * (t[:2] @ X[i] + t[2]) - 1)} for i in range(len(Y))]
    r = optimize.minimize(lambda t: 0.5 * t[:2] @ t[:2], [0.1, 0.1, 0], constraints=cons, method="SLSQP", options={"ftol": 1e-12})
    return r.x
X = np.array([[0, 0], [1, 0], [0, 1], [3, 3], [4, 3], [3, 4.]]); Y = np.array([-1, -1, -1, 1, 1, 1.])
t1 = hard_svm(X, Y)
X_mv = X.copy(); X_mv[0] = [-0.5, -0.3]; X_mv[5] = [3.2, 4.5]   # サポートベクトルでない点を外側へ
t2 = hard_svm(X_mv, Y)
marg = Y * (X @ t1[:2] + t1[2])
q.check("q23-support-vector-machine-3", np.allclose(t1, t2, atol=1e-5) and marg[0] > 1.1 and marg[5] > 1.1,
        "非サポートベクトルを動かしても解は同じ（正しい）")

# --- カーネル法 ---
xv = np.array([1, 2.]); yv = np.array([3, 1.])
phi = lambda v: np.array([1, np.sqrt(2) * v[0], np.sqrt(2) * v[1], v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
q.num("q23-kernel-method-1", (xv @ yv + 1) ** 2)
q.check("q23-kernel-method-1", abs(phi(xv) @ phi(yv) - 36) < 1e-9, "6次元の特徴の内積と一致")

# --- 決定木 ---
gini = lambda c: 1 - sum((np.array(c) / sum(c)) ** 2)
q.num("q23-decision-tree-1", gini([10, 10]) - (10 / 20 * gini([8, 2]) + 10 / 20 * gini([2, 8])))

# --- ナイーブベイズ ---
s = 0.4 * 0.8 * 0.6; h = 0.6 * 0.1 * 0.3
q.num("q23-naive-bayes-classifier-1", s / (s + h))

q.done()
