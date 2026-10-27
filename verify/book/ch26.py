"""準1級 教科書 第26章「その他の多変量解析手法」の検算（固有値分解・特異値分解・シミュレーション）。"""
import numpy as np
from bookcheck import Book
from scipy.spatial.distance import pdist, squareform

b = Book("pre1", 26)
D = np.array([[0, 3, 4, 5], [3, 0, 5, 4], [4, 5, 0, 3], [5, 4, 3, 0.]])
J = np.eye(4) - 0.25
B = -0.5 * J @ (D ** 2) @ J
w, V = np.linalg.eigh(B)
i = np.argsort(w)[::-1]
w, V = w[i], V[:, i]
for v, s in zip(w, ["16", "9", "0", "0"]):
    b.num("固有値", v, s, tol=1e-9 if s == "0" else None)
X = V[:, :2] * np.sqrt(w[:2])
b.check("距離を完全に再現", np.allclose(squareform(pdist(X)), D), "")
target = np.array([[-2, 1.5], [-2, -1.5], [2, 1.5], [2, -1.5]])
b.check("座標は 4×3 の長方形（符号の違いを除く）", np.allclose(np.abs(X), np.abs(target)), X.round(3))
R11 = np.array([[1, .4], [.4, 1]])
R22 = np.array([[1, .3], [.3, 1]])
R12 = np.array([[.5, .3], [.4, .2]])
M = np.linalg.inv(R11) @ R12 @ np.linalg.inv(R22) @ R12.T
ev = np.sort(np.linalg.eigvals(M).real)[::-1]
b.num("固有値 1", ev[0], "0.321")
b.num("固有値 2", ev[1], "0.0016")
b.num("正準相関 1", np.sqrt(ev[0]), "0.567")
b.num("正準相関 2", np.sqrt(ev[1]), "0.040")
# シミュレーション：正準相関 = 一次結合の相関の最大
rng = np.random.default_rng(26)
C = np.block([[R11, R12], [R12.T, R22]])
Z = rng.multivariate_normal(np.zeros(4), C, 300000)
ang = np.linspace(0, np.pi, 181)
best = 0
for t1 in ang:
    u = Z[:, :2] @ [np.cos(t1), np.sin(t1)]
    for t2 in ang[::3]:
        best = max(best, abs(np.corrcoef(u, Z[:, 2:] @ [np.cos(t2), np.sin(t2)])[0, 1]))
b.check("一次結合の相関の最大 ≈ 0.567（格子探索）", abs(best - 0.567) < 0.01, best)
T = np.array([[30, 20, 10], [10, 20, 30.]])
n = T.sum()
E = np.outer(T.sum(1), T.sum(0)) / n
chi = ((T - E) ** 2 / E).sum()
b.num("χ²", chi, "20")
b.num("全イナーシャ", chi / n, "0.167")
P = T / n
r, c = P.sum(1), P.sum(0)
S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
sv = np.linalg.svd(S, compute_uv=False)
b.check("特異値の 2 乗の和 = χ²/n、1 次元ですべて", abs((sv ** 2).sum() - chi / n) < 1e-12 and sv[1] < 1e-12, sv)
# 章末問題
D3 = np.full((3, 3), 2.) - 2 * np.eye(3)
J3 = np.eye(3) - 1 / 3
B3 = -0.5 * J3 @ (D3 ** 2) @ J3
b.check("問1 対角 4/3・非対角 −2/3・固有値 2,2,0", np.allclose(np.diag(B3), 4 / 3) and abs(B3[0, 1] + 2 / 3) < 1e-12 and np.allclose(np.sort(np.linalg.eigvalsh(B3)), [0, 2, 2]), "")
b.num("問2 間接", 0.6 * 0.5, "0.3")
b.num("問2 総合", 0.2 + 0.3, "0.5")
b.num("問3", 12 / 200, "0.06")
b.done()
