"""準1級 教科書 第25章「因子分析・グラフィカルモデル」の検算（行列計算・バリマックスの自前実装・シミュレーション）。"""
import numpy as np
from bookcheck import Book

b = Book("pre1", 25)
l = np.array([.9, .8, .7, .6])
b.num("相関 1,2", l[0] * l[1], "0.72")
b.num("相関 3,4", l[2] * l[3], "0.42")
for v, s in zip(l ** 2, ["0.81", "0.64", "0.49", "0.36"]):
    b.num("共通性", v, s)
for v, s in zip(1 - l ** 2, ["0.19", "0.36", "0.51", "0.64"]):
    b.num("独自性", v, s)
rng = np.random.default_rng(25)
f = rng.normal(size=300000)
X = np.outer(f, l) + rng.normal(size=(300000, 4)) * np.sqrt(1 - l ** 2)
b.check("モデルから作ったデータの相関 ≈ λiλj", abs(np.corrcoef(X.T)[0, 1] - 0.72) < 0.005, np.corrcoef(X.T)[0, 1])
L = np.array([[.7, .4], [.75, .35], [.65, .45], [.6, -.5], [.65, -.45], [.55, -.55]])


def varimax(L, it=5000):
    """カイザーの正規化つきバリマックス（収束するまでくり返す）。"""
    h = np.sqrt((L ** 2).sum(1))
    A = L / h[:, None]
    p, k = A.shape
    R = np.eye(k)
    for _ in range(it):
        Lam = A @ R
        u, s, vt = np.linalg.svd(A.T @ (Lam ** 3 - Lam @ np.diag((Lam ** 2).sum(0)) / p))
        R = u @ vt
    return (A @ R) * h[:, None], R


V, Rm = varimax(L)
V[:, 1] *= np.sign(V[3, 1])
V[:, 0] *= np.sign(V[0, 0])
for v, s in zip(V[:, 0], ["0.793", "0.800", "0.787", "0.139", "0.209", "0.068"]):
    b.num("回転後 因子1", v, s)
for v, s in zip(V[:, 1], ["0.143", "0.214", "0.073", "0.769", "0.762", "0.775"]):
    b.num("回転後 因子2", v, s)
V2, _ = varimax(L, 10000)
b.check("収束している（くり返しを倍にしても同じ）", np.allclose(np.abs(V2), np.abs(V), atol=1e-8), "")
b.check("回転は直交行列", np.allclose(Rm @ Rm.T, np.eye(2)), "")
b.check("共通性は変わらない", np.allclose((L ** 2).sum(1), (V ** 2).sum(1)), "")
b.num("変数 1 の共通性", (L[0] ** 2).sum(), "0.65")
b.check("ΛΛ' は回転で不変", np.allclose(L @ L.T, V @ V.T), "")
S = np.array([[1, .6, .36], [.6, 1, .6], [.36, .6, 1]])
P = np.linalg.inv(S)
b.check("精度行列の値", np.allclose(P, [[1.5625, -0.9375, 0], [-0.9375, 2.125, -0.9375], [0, -0.9375, 1.5625]]), P)
b.num("0.6×0.6", 0.36, "0.36")
Y = rng.multivariate_normal(np.zeros(3), S, 400000)
r1 = Y[:, 0] - np.polyfit(Y[:, 1], Y[:, 0], 1)[0] * Y[:, 1]
r3 = Y[:, 2] - np.polyfit(Y[:, 1], Y[:, 2], 1)[0] * Y[:, 1]
b.check("x2 で回帰した残差どうしは無相関（偏相関 0）", abs(np.corrcoef(r1, r3)[0, 1]) < 0.005, np.corrcoef(r1, r3)[0, 1])
# 章末問題
b.num("問1 共通性", 0.36 + 0.25, "0.61")
b.num("問1 独自性", 1 - 0.61, "0.39")
b.num("問2 1,3", 0.8 * 0.3, "0.24")
b.num("問2 2,3", 0.5 * 0.3, "0.15")
O = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2.]])
b.num("問3 偏相関", -O[0, 1] / np.sqrt(O[0, 0] * O[1, 1]), "0.5")
Sc = np.linalg.inv(O)
pc = (Sc[0, 1] - Sc[0, 2] * Sc[1, 2] / Sc[2, 2]) / np.sqrt((Sc[0, 0] - Sc[0, 2] ** 2 / Sc[2, 2]) * (Sc[1, 1] - Sc[1, 2] ** 2 / Sc[2, 2]))
b.check("問3 条件付き分布から計算した偏相関も 0.5", abs(pc - 0.5) < 1e-12, pc)
b.done()
