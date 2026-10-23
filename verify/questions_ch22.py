"""第22章（主成分分析）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp

q = Q(22)
rng = np.random.default_rng(22)


def eig_desc(A):
    w, v = np.linalg.eigh(np.asarray(A, float))
    idx = np.argsort(w)[::-1]
    return w[idx], v[:, idx]


def rand_cov(p):
    B = rng.normal(size=(p, p))
    return B @ B.T + 0.1 * np.eye(p)


# --- 主成分分析 ---
S = rand_cov(3)
w, H = eig_desc(S)
q.check("q22-principal-component-analysis-1", abs(H[:, 0] @ S @ H[:, 1]) < 1e-10
        and abs(w[0] * w[1]) > 1e-3, "Cov[Y1,Y2]=0 で λ1λ2≠0（記述は誤り）")
w, H = eig_desc([[5, 2], [2, 2]])
q.check("q22-principal-component-analysis-2", np.allclose(w, [6, 1]), "固有値 6, 1")
q.num("q22-principal-component-analysis-2", abs(H[0, 0]))
s2, rho, lam = sp.symbols("sigma2 rho lambda", positive=True)
Sig = sp.Matrix([[s2, s2 * rho], [s2 * rho, s2]])
ev = Sig.eigenvects()
ok = True
for val, mult, vecs in ev:
    v = vecs[0]
    if sp.simplify(val - s2 * (1 + rho)) == 0:
        ok &= sp.simplify(v[0] - v[1]) == 0
    elif sp.simplify(val - s2 * (1 - rho)) == 0:
        ok &= sp.simplify(v[0] + v[1]) == 0
    else:
        ok = False
q.check("q22-principal-component-analysis-3", ok, "sympy: 固有値 σ²(1±ρ)、固有ベクトル (1,±1)")

# --- 主成分 ---
S = rand_cov(4)
w, H = eig_desc(S)
q.check("q22-principal-component-1", abs(w.sum() - np.trace(S)) < 1e-10
        and np.allclose(H.T @ S @ H, np.diag(w)), "主成分の分散の和＝tr Σ（正しい）")

# --- 固有値と固有ベクトル ---
q.num("q22-eigenvalue-and-eigenvector-1", eig_desc([[3, 1], [1, 3]])[0][0])
A = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 5]], float)
ev = sorted(np.linalg.eigvalsh(A))
cands = [[1, 3, 5], [2, 2, 5], [1, 3, 4], [0, 4, 5]]
q.check("q22-eigenvalue-and-eigenvector-2", [np.allclose(ev, c) for c in cands] == [True, False, False, False],
        "正解は {1,3,5} のみ")
h1 = np.array([1, 1]) / np.sqrt(2); h2 = np.array([1, -1]) / np.sqrt(2)
A = 4 * np.outer(h1, h1) + np.outer(h2, h2)
q.num("q22-eigenvalue-and-eigenvector-3", np.linalg.inv(A)[0, 1])
q.check("q22-eigenvalue-and-eigenvector-3", np.allclose(A, [[2.5, 1.5], [1.5, 2.5]]), "A=[[2.5,1.5],[1.5,2.5]]")

# --- 寄与率 ---
q.num("q22-contribution-ratio-1", 5.2 / (5.2 + 2.1 + 0.7))
cands = [1.2 / 5, 1.2 / 3.8, 3.8 / 5, 1.2 / 2.6]
q.check("q22-contribution-ratio-2", abs(cands[0] - 0.24) < 1e-12 and all(abs(c - cands[0]) > 0.05 for c in cands[1:])
        and abs(cands[1] - 0.316) < 5e-4 and abs(cands[3] - 0.462) < 5e-4, "正解 0.240、誤答の値も一致")
r = 0.7
q.num("q22-contribution-ratio-3", r)
q.check("q22-contribution-ratio-3", abs(eig_desc([[1, r], [r, 1]])[0][0] / 2 - 0.85) < 1e-12, "寄与率 0.85 を再現")

# --- 累積寄与率 ---
w2 = eig_desc([[1, .9], [.9, 1]])[0]
w3 = eig_desc([[1, .9, 0], [.9, 1, 0], [0, 0, 1]])[0]
c1a, c1b = w2[0] / w2.sum(), w3[0] / w3.sum()
q.check("q22-cumulative-contribution-ratio-1", c1b < c1a and abs(c1a - .95) < 1e-12 and abs(c1b - 1.9 / 3) < 1e-12,
        "反例で 0.95→0.633 に下がる（記述は誤り）")
lam = np.array([6, 3, 2, 1.])
C = np.cumsum(lam) / lam.sum()
q.num("q22-cumulative-contribution-ratio-2", int(np.argmax(C >= 0.8)) + 1)
R = 0.5 * np.ones((3, 3)) + 0.5 * np.eye(3)
w = eig_desc(R)[0]
q.num("q22-cumulative-contribution-ratio-3", w[0] / 3)
q.check("q22-cumulative-contribution-ratio-3", np.allclose(w, [2, .5, .5]), "固有値 2, 0.5, 0.5")

# --- 主成分負荷量 ---
q.num("q22-principal-component-loading-1", np.sqrt(2.25) * 0.6)
Sig = np.array([[5, 2], [2, 2.]])
w, H = eig_desc(Sig)
h = H[:, 0] * np.sign(H[0, 0])
# 共分散から直接：Cov[Y1, X2] / sqrt(V[Y1] V[X2])
load = (Sig @ h)[1] / np.sqrt(w[0] * Sig[1, 1])
q.num("q22-principal-component-loading-2", load)
q.check("q22-principal-component-loading-2", np.sqrt(6) / np.sqrt(5) > 1, "σ22 で割らない値は1を超える（解説）")
B = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
Rr = np.corrcoef(B, rowvar=False)
w, H = eig_desc(Rr)
q.check("q22-principal-component-loading-3", np.allclose((H ** 2 * w).sum(axis=1), 1), "負荷量の2乗和はすべての変数で1（正しい）")

# --- 主成分得点 ---
q.num("q22-principal-component-score-1", np.array([.6, .8]) @ (np.array([70, 55]) - np.array([60, 50])))
u = np.array([15 / 10, 2.5 / 5])
q.num("q22-principal-component-score-2", u @ np.array([1, 1]) / np.sqrt(2))
z = np.array([-2, -1, 0, 1, 2.])
cands = [z.var(ddof=1), z.var(ddof=0), z.std(ddof=1), (z ** 2).sum()]
q.check("q22-principal-component-score-3", abs(cands[0] - 2.5) < 1e-12 and abs(cands[1] - 2) < 1e-12
        and abs(cands[2] - 1.58) < 5e-3 and len(set(np.round(cands, 3))) == 4, "正解 2.5、誤答は n 割り・SD・2乗和")

# --- 相関行列からの主成分分析 ---
X = rng.normal(size=(100, 3)) @ rng.normal(size=(3, 3))
X2 = X.copy(); X2[:, 0] /= 100   # cm → m
wa = eig_desc(np.corrcoef(X, rowvar=False))[0]; wb = eig_desc(np.corrcoef(X2, rowvar=False))[0]
ca = eig_desc(np.cov(X, rowvar=False))[0]; cb = eig_desc(np.cov(X2, rowvar=False))[0]
q.check("q22-pca-on-correlation-matrix-1", np.allclose(wa / 3, wb / 3)
        and not np.allclose(ca / ca.sum(), cb / cb.sum()), "相関行列の寄与率は不変、共分散なら変わる（記述は誤り）")
Sig = np.array([[16, 6], [6, 4.]])
D = np.diag(1 / np.sqrt(np.diag(Sig)))
w = eig_desc(D @ Sig @ D)[0]
q.num("q22-pca-on-correlation-matrix-2", w[0] / 2)
wc = eig_desc(Sig)[0]
q.check("q22-pca-on-correlation-matrix-2", abs(wc[0] / wc.sum() - 0.924) < 5e-4 and abs(wc[0] - (10 + np.sqrt(72))) < 1e-9,
        "解説の共分散版 0.924")

# --- スクリープロット ---
lam = np.array([3.1, 1.6, 0.4, 0.35, 0.3, 0.25])
d = -np.diff(lam)
elbow = int(np.argmax(d[:-1] - d[1:])) + 2   # 傾きが最も大きく緩むのは第 elbow 点
q.check("q22-scree-plot-1", abs(lam.sum() - 6) < 1e-12 and elbow == 3 and (elbow - 1) == 2
        and int((lam >= 1).sum()) == 2, "肘は第3点、残すのは2個（カイザーも2）")

# --- カイザー基準 ---
lam = np.array([2.9, 1.3, 0.95, 0.4, 0.3, 0.15])
q.check("q22-kaiser-criterion-1", abs(lam.sum() - 6) < 1e-12, "固有値の和が6")
q.num("q22-kaiser-criterion-1", int((lam >= 1).sum()))
S = np.cov(X, rowvar=False)
q.check("q22-kaiser-criterion-2", not np.isclose(np.trace(S) / 3, 1)
        and not np.isclose(np.trace(S * 1e4) / 3, np.trace(S) / 3), "共分散では固有値の平均が1でなく単位で変わる（記述は誤り）")

# --- 特異値分解 ---
X = np.array([[2, 1], [1, 2], [1, -1.]])
d = np.linalg.svd(X, compute_uv=False)
q.num("q22-singular-value-decomposition-1", d[0])
q.check("q22-singular-value-decomposition-1", abs(d[1] - np.sqrt(3)) < 1e-12, "d2=√3")
Y = rng.normal(size=(6, 4))
q.check("q22-singular-value-decomposition-2", abs((np.linalg.svd(Y, compute_uv=False) ** 2).sum() - (Y ** 2).sum()) < 1e-9,
        "特異値の2乗和＝全成分の2乗和（正しい）")

# --- ラグランジュ乗数法 ---
Sig = np.array([[2, 1], [1, 2.]])
best = max(np.array([np.cos(t), np.sin(t)]) @ Sig @ np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, np.pi, 100001))
a = np.array([1, 1.])
cands = [3, a @ Sig @ a, 2, 1]
q.check("q22-lagrange-multiplier-pca-1", abs(best - 3) < 1e-8 and cands[1] == 6 and len(set(cands)) == 4,
        "単位円上の最大は3、(1,1) をそのまま代入すると6")
a = sp.Matrix(sp.symbols("a1 a2")); l = sp.Symbol("l")
Ssym = sp.Matrix([[2, 1], [1, 2]])
L = (a.T * Ssym * a)[0] - l * ((a.T * a)[0] - 1)
grad = [sp.diff(L, ai) for ai in a]
sols = sp.solve(grad + [(a.T * a)[0] - 1], list(a) + [l], dict=True)
vals = sorted({float((sp.Matrix([s[a[0]], s[a[1]]]).T * Ssym * sp.Matrix([s[a[0]], s[a[1]]]))[0]) for s in sols})
lams = sorted({float(s[l]) for s in sols})
q.check("q22-lagrange-multiplier-pca-2", vals == lams == [1.0, 3.0], "停留点での値＝乗数λ＝固有値、最大は λ1")

# --- バイプロット ---
X = rng.normal(size=(20, 4)); X -= X.mean(axis=0)
U, d, Vt = np.linalg.svd(X, full_matrices=False)
q.check("q22-biplot-1", np.allclose((X @ Vt.T)[:, :2], U[:, :2] * d[:2]), "XV=UD（正しい）")

# --- 低ランク近似 ---
d = np.array([6, 3, 2, 1.])
U, _ = np.linalg.qr(rng.normal(size=(8, 4))); V, _ = np.linalg.qr(rng.normal(size=(4, 4)))
X = U @ np.diag(d) @ V.T
Uu, dd, Vv = np.linalg.svd(X)
X2 = Uu[:, :2] @ np.diag(dd[:2]) @ Vv[:2]
q.num("q22-low-rank-approximation-1", ((X - X2) ** 2).sum() / (X ** 2).sum())

q.done()
