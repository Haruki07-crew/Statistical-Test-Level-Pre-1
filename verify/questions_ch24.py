"""第24章（クラスター分析）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform, pdist, minkowski

q = Q(24)
rng = np.random.default_rng(24)

# 4対象 A,B,C,D の距離行列
DM = np.array([[0, 2, 6, 10], [2, 0, 5, 9], [6, 5, 0, 4], [10, 9, 4, 0.]])
dv = squareform(DM)
Ls = linkage(dv, "single"); Lc = linkage(dv, "complete"); La = linkage(dv, "average")

# --- 階層的クラスタリング ---
X = rng.normal(size=(12, 2))
q.num("q24-hierarchical-clustering-1", linkage(pdist(X), "average").shape[0])
# 入れ子構造：各段のクラスター分けが前の段の細分になっていること
L = linkage(pdist(X), "average")
nested = True
for k in range(2, 12):
    fine = fcluster(L, k, "maxclust"); coarse = fcluster(L, k - 1, "maxclust")
    nested &= all(len(set(coarse[fine == c])) == 1 for c in set(fine))
q.check("q24-hierarchical-clustering-2", nested, "分割は入れ子（正しい）")
def same_tree(L1, L2):
    return all(np.array_equal(fcluster(L1, k, "maxclust"), fcluster(L2, k, "maxclust")) for k in range(1, L1.shape[0] + 2))
single_ok, avg_changed, wavg_changed = True, False, False
for _ in range(300):
    d = pdist(rng.uniform(size=(7, 2)))
    single_ok &= same_tree(linkage(d, "single"), linkage(d ** 2, "single"))
    avg_changed |= not same_tree(linkage(d, "average"), linkage(d ** 2, "average"))
    wavg_changed |= not same_tree(linkage(d, "weighted"), linkage(d ** 2, "weighted"))
q.check("q24-hierarchical-clustering-3", single_ok and avg_changed and wavg_changed,
        "最短距離法は常に同じ木、群平均・重み付き群平均は変わる例あり")

# --- デンドログラム ---
coph = squareform(cophenet(Ls))
q.num("q24-dendrogram-1", coph[0, 2])
q.check("q24-dendrogram-1", np.allclose(Ls[:, 2], [2, 4, 5]), "併合の高さ 2,4,5")
q.check("q24-dendrogram-2", coph[1, 2] == 5 and DM[1, 0] < DM[1, 2], "B と C が隣でも B に最も近いのは A（記述は誤り）")
q.num("q24-dendrogram-3", len(set(fcluster(Lc, 7, "distance"))))
q.check("q24-dendrogram-3", np.allclose(Lc[:, 2], [2, 4, 10]), "最長距離法の高さ 2,4,10")

# --- 最短距離法 ---
pts = np.array([0, 1, 3.2, 6, 10.])[:, None]
L1 = linkage(pdist(pts), "single")
q.num("q24-single-linkage-1", L1[-1, 2])
q.check("q24-single-linkage-1", np.allclose(L1[:, 2], [1, 2.2, 2.8, 4]), "高さ 1,2.2,2.8,4")
# 連鎖効果：2群の間に橋渡しの点を並べると、最短距離法は2分割で片方が1点だけになりやすい
A = rng.normal(size=(30, 2)) * 0.3; B = rng.normal(size=(30, 2)) * 0.3 + [6, 0]
bridge = np.c_[np.linspace(0.8, 5.2, 12), np.zeros(12)]
Z = np.vstack([A, B, bridge])
hs = linkage(pdist(Z), "single")[-1, 2]; hs0 = linkage(pdist(np.vstack([A, B])), "single")[-1, 2]
q.check("q24-single-linkage-2", hs < 1.0 and hs0 > 4,
        f"橋渡しの点があると最短距離法は2群を低い高さ {hs:.2f} で併合（なければ {hs0:.2f}）＝連鎖効果（正しい）")
q.num("q24-single-linkage-3", min(7, 5))

# --- 最長距離法 ---
q.num("q24-complete-linkage-1", max(DM[0, 2], DM[1, 2]))
L2 = linkage(pdist(pts), "complete")
q.num("q24-complete-linkage-2", L2[1, 2])
q.check("q24-complete-linkage-2", np.allclose(L2[:, 2], [1, 2.8, 6, 10]) and abs(L1[1, 2] - 2.2) < 1e-12, "最長は 1,2.8,6,10／最短の2番目は2.2")
ok = True
for _ in range(200):
    d = pdist(rng.normal(size=(rng.integers(3, 10), 3)))
    ok &= abs(linkage(d, "complete")[-1, 2] - d.max()) < 1e-12
q.check("q24-complete-linkage-3", ok, "200例で最後の高さ＝最大距離（正しい）")

# --- 群平均法 ---
q.num("q24-average-linkage-1", La[-1, 2])
cands = [(3 * 4 + 1 * 8) / 4, (4 + 8) / 2, min(4, 8), max(4, 8)]
q.check("q24-average-linkage-2", cands[0] == 5 and len(set(cands)) == 4, "UPGMA=5、WPGMA=6、min=4、max=8")
# scipy でも確認：C_i が3点、C_j が1点で、k との距離が 4（3点とも）と 8
D5 = np.zeros((5, 5))
for a in range(3):
    D5[a, 4] = D5[4, a] = 4
D5[3, 4] = D5[4, 3] = 8
for a in range(3):
    D5[a, 3] = D5[3, a] = 3.5
    for b in range(3):
        if a != b: D5[a, b] = 1
L5 = linkage(squareform(D5), "average")
q.check("q24-average-linkage-2", abs(L5[-1, 2] - 5) < 1e-12, "scipy の群平均法でも5")

# --- ウォード法 ---
ess = lambda P: ((P - P.mean(0)) ** 2).sum()
C1 = np.array([[0, 0], [2, 0.]]); C2 = np.array([[4, 3.]])
q.num("q24-ward-method-1", ess(np.vstack([C1, C2])) - ess(C1) - ess(C2))
delta = lambda ni, nj, a, b: ni * nj / (ni + nj) * (a - b) ** 2
dd = {"PQ": delta(1, 1, 0, 3), "QR": delta(1, 10, 3, 5.5), "PR": delta(1, 10, 0, 5.5)}
q.check("q24-ward-method-2", min(dd, key=dd.get) == "PQ" and len(set(np.round(list(dd.values()), 6))) == 3
        and (5.5 - 3) ** 2 < (3 - 0) ** 2, f"Δ={dd}、重心法なら QR")
x4 = np.array([0, 2, 6, 8.])[:, None]
Lw = linkage(x4, "ward")   # scipy の高さは sqrt(2Δ)
q.num("q24-ward-method-3", Lw[-1, 2] ** 2 / 2)
q.check("q24-ward-method-3", abs(ess(x4) - 40) < 1e-12 and abs(sum(Lw[:, 2] ** 2 / 2) - 40) < 1e-9, "Δの合計＝全平方和40")

# --- 重心法 ---
i, j = np.array([0, 0.]), np.array([np.sqrt(2) * np.cos(np.pi / 4), np.sqrt(2) * np.sin(np.pi / 4)])  # d_ij=2
# k: |k-i|²=4, |k-j|²=10 となる点を求める
from scipy.optimize import fsolve
k = fsolve(lambda v: [v @ v - 4, (v - j) @ (v - j) - 10], [-1, -1])
q.check("q24-centroid-method-1", abs(k @ k - 4) < 1e-9 and abs((k - j) @ (k - j) - 10) < 1e-9, "距離が実現可能")
q.num("q24-centroid-method-1", ((k - (i + j) / 2) ** 2).sum())
q.check("q24-centroid-method-1", abs(0.5 * 4 + 0.5 * 10 - 0.25 * 2 - 6.5) < 1e-12, "更新式でも6.5")
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
Lcen = linkage(tri, "centroid")   # scipy は非2乗のユークリッド距離で高さを出す
q.check("q24-centroid-method-2", Lcen[1, 2] < Lcen[0, 2] and abs(Lcen[1, 2] ** 2 - 0.75) < 1e-9, "正三角形で逆転（正しい）")

# --- ランス・ウィリアムズ ---
def lw(a_i, a_j, b, g, dki, dkj, dij):
    return a_i * dki + a_j * dkj + b * dij + g * abs(dki - dkj)
tests = [(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(50)]
cands = [(.5, .5, 0, .5), (.5, .5, 0, -.5), (.5, .5, -.25, 0), (.5, .5, 0, 0)]
res = [all(abs(lw(*c, a, b, d) - max(a, b)) < 1e-12 for a, b, d in tests) for c in cands]
q.check("q24-lance-williams-formula-1", res == [True, False, False, False], "max になるのは γ=+1/2 のみ")
v = lw(2 / 3, 2 / 3, -1 / 3, 0, 8, 18, 2)
q.num("q24-lance-williams-formula-2", v)
pi_, pj, pk = np.array([0, 0.]), np.array([1, 1.]), np.array([-2, -2.])
q.check("q24-lance-williams-formula-2", abs(pk @ pk - 8) < 1e-12 and abs((pk - pj) @ (pk - pj) - 18) < 1e-12
        and abs(2 * (ess(np.vstack([pi_, pj, pk])) - ess(np.vstack([pi_, pj]))) - v) < 1e-9
        and abs(ess(np.vstack([pi_, pj, pk])) - ess(np.vstack([pi_, pj])) - 8.333) < 5e-4, "2Δ と一致、Δ=8.333")

# --- k-means ---
x = np.array([1, 2, 4, 7, 9, 10.])
def lloyd(x, m, it=50):
    for _ in range(it):
        lab = np.argmin(np.abs(x[:, None] - m[None, :]), axis=1)
        m = np.array([x[lab == j].mean() for j in range(len(m))])
    return m, lab
m1 = np.array([2, 9.]); lab = np.argmin(np.abs(x[:, None] - m1[None, :]), axis=1)
m_new = np.array([x[lab == j].mean() for j in range(2)])
q.num("q24-k-means-method-1", m_new.min())
mf, labf = lloyd(x, m1)
q.check("q24-k-means-method-1", np.allclose(mf, m_new), "1回で収束")
q.num("q24-k-means-method-2", sum(((x[labf == j] - mf[j]) ** 2).sum() for j in range(2)))
R = np.array([[0, 0], [0, 1], [4, 0], [4, 1.]])
def lloyd2(P, m, it=50):
    for _ in range(it):
        lab = np.argmin(((P[:, None, :] - m[None]) ** 2).sum(-1), axis=1)
        m = np.array([P[lab == j].mean(0) for j in range(len(m))])
    W = sum(((P[lab == j] - m[j]) ** 2).sum() for j in range(len(m)))
    return W
q.check("q24-k-means-method-3", abs(lloyd2(R, np.array([[2, 0], [2, 1.]])) - 16) < 1e-12
        and abs(lloyd2(R, np.array([[0, .5], [4, .5]])) - 1) < 1e-12, "局所解 W=16、最適 W=1（記述は誤り）")

# --- 非階層 ---
from itertools import product
parts = {frozenset([frozenset(i for i in range(6) if s[i] == 0), frozenset(i for i in range(6) if s[i] == 1)])
         for s in product([0, 1], repeat=6) if 0 < sum(s) < 6}
q.num("q24-non-hierarchical-clustering-1", len(parts))

# --- 距離 ---
xv = np.array([1, 4, 2.]); yv = np.array([4, 0, 2.])
q.num("q24-distance-measures-1", np.abs(xv - yv).sum())
vals = (minkowski(xv, yv, 1), minkowski(xv, yv, 2), minkowski(xv, yv, 3), np.abs(xv - yv).max())
q.check("q24-distance-measures-2", (abs(vals[0] - 7) < 1e-12 and abs(vals[1] - 5) < 1e-12 and abs(vals[2] - 4.50) < 5e-3 and vals[3] == 4),
        f"d1,d2,d3,d∞={vals}")
q.check("q24-distance-measures-3", (0 - 2) ** 2 > (0 - 1) ** 2 + (1 - 2) ** 2, "三角不等式が破れる（記述は誤り）")

# --- 類似度 ---
bx = np.array([1, 1, 0, 1, 0, 0, 1, 0]); by = np.array([1, 0, 0, 1, 1, 0, 1, 0])
a = np.sum((bx == 1) & (by == 1)); b = np.sum((bx == 1) & (by == 0)); c = np.sum((bx == 0) & (by == 1)); d = np.sum((bx == 0) & (by == 0))
q.num("q24-similarity-measures-1", a / (a + b + c))
from scipy.spatial.distance import jaccard
q.check("q24-similarity-measures-1", abs(1 - jaccard(bx, by) - 0.6) < 1e-12 and (a + d) / 8 == 0.75, "scipy と一致、単純一致0.75")
u, w = np.array([1, 2, 2.]), np.array([2, 0, 1.])
q.num("q24-similarity-measures-2", u @ w / np.linalg.norm(u) / np.linalg.norm(w))

# --- クラスター数 ---
q.num("q24-number-of-clusters-1", (5 - 2) / max(2, 5))
P = rng.normal(size=(15, 2))
Ws = [sum(((P[fcluster(linkage(P, 'ward'), k, 'maxclust') == c] - P[fcluster(linkage(P, 'ward'), k, 'maxclust') == c].mean(0)) ** 2).sum()
          for c in set(fcluster(linkage(P, 'ward'), k, 'maxclust'))) for k in range(1, 16)]
q.check("q24-number-of-clusters-2", Ws[-1] < 1e-12 and all(np.diff(Ws) <= 1e-12), "W(k) は単調減少し k=n で0（記述は誤り）")

# --- 混合モデル ---
g = stats.norm.pdf(1, 0, 1) / (stats.norm.pdf(1, 0, 1) + stats.norm.pdf(1, 3, 1))
q.num("q24-mixture-model-clustering-1", g)

# --- 標準化 ---
q.num("q24-standardization-before-clustering-1", np.hypot(5 / 10, 1 / 0.5))
q.check("q24-standardization-before-clustering-1", abs(np.hypot(5, 1) - 5.10) < 5e-3, "標準化なしは5.10")

# --- 導出問題（追加） ---
def _deriv24():
    import numpy as _N
    import sympy as _S
    na, nb = _S.symbols("n_A n_B", positive=True)
    a1, a2, b1, b2 = _S.symbols("a1 a2 b1 b2", real=True)
    A2 = a1 ** 2 + a2 ** 2; B2 = b1 ** 2 + b2 ** 2
    m1 = (na * a1 + nb * b1) / (na + nb); m2 = (na * a2 + nb * b2) / (na + nb)
    D = na * A2 + nb * B2 - (na + nb) * (m1 ** 2 + m2 ** 2)
    ok = _S.simplify(D - na * nb / (na + nb) * ((a1 - b1) ** 2 + (a2 - b2) ** 2)) == 0
    rng = _N.random.default_rng(24)
    W = lambda P: ((P - P.mean(0)) ** 2).sum()
    for na_, nb_ in [(3, 5), (1, 1), (7, 2)]:
        A = rng.normal(size=(na_, 3)); Bm = rng.normal(size=(nb_, 3)) + 2
        lhs = W(_N.vstack([A, Bm])) - W(A) - W(Bm)
        rhs = na_ * nb_ / (na_ + nb_) * ((A.mean(0) - Bm.mean(0)) ** 2).sum()
        ok &= abs(lhs - rhs) < 1e-10
    q.check("q24-ward-method-4", bool(ok), "sympy: Δ=n_A n_B/(n_A+n_B)‖ā-b̄‖²、乱数のクラスターで直接計算と一致")
_deriv24()

q.done()
