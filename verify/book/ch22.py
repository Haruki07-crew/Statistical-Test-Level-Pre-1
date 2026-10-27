"""準1級 教科書 第22章「主成分分析」の検算（固有値分解・特異値分解・シミュレーション）。"""
import numpy as np
from bookcheck import Book

b = Book("pre1", 22)
R = np.array([[1, .7, .3, .25], [.7, 1, .35, .3], [.3, .35, 1, .65], [.25, .3, .65, 1]])
w, V = np.linalg.eigh(R)
i = np.argsort(w)[::-1]
w, V = w[i], V[:, i]
V = V * np.sign(V[0])
for v, s in zip(w, ["2.277", "1.078", "0.348", "0.298"]):
    b.num("固有値", v, s)
b.check("図の固有値", np.allclose(b.chart(1)["values"], w.round(3)), "")
for v, s in zip(w / 4 * 100, ["56.9", "26.9", "8.7", "7.4"]):
    b.num("寄与率 %", v, s)
b.num("累積 2 つ %", w[:2].sum() / 4 * 100, "83.9")
for j, s in enumerate(["0.498", "0.523", "0.503", "0.476"]):
    b.num("第1固有ベクトル", V[j, 0], s)
for j, s in enumerate(["0.521", "0.456", "-0.476", "-0.543"]):
    b.num("第2固有ベクトル", V[j, 1], s)
L = V * np.sqrt(w)
b.num("国語と第1の負荷量", L[0, 0], "0.75")
b.num("国語と第2の負荷量", L[0, 1], "0.54")
b.check("負荷量の 2 乗の和 = 1", np.allclose((L ** 2).sum(1), 1), "")
z = np.array([1.2, 0.8, -0.5, -0.9])
sc = z @ V[:, :2]
b.num("第1得点", sc[0], "0.34")
b.num("第2得点", sc[1], "1.72")
# 負荷量 = 相関（シミュレーション）
rng = np.random.default_rng(22)
X = rng.multivariate_normal(np.zeros(4), R, 400000)
Z = X @ V
b.check("負荷量 = 主成分ともとの変数の相関", abs(np.corrcoef(X[:, 0], Z[:, 0])[0, 1] - L[0, 0]) < 0.005, "")
b.check("主成分どうしは無相関", abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) < 0.005, "")
# 分散共分散行列
S = np.array([[100, 5], [5, 1.]])
ws = np.linalg.eigvalsh(S)[::-1]
b.num("寄与率 %", ws[0] / ws.sum() * 100, "99.3")
# SVD
Xc = X[:2000] - X[:2000].mean(0)
U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
b.check("d²/(n−1) = 分散共分散行列の固有値", np.allclose(d ** 2 / 1999, np.sort(np.linalg.eigvalsh(np.cov(Xc.T)))[::-1]), "")
Xr = U[:, :2] * d[:2] @ Vt[:2]
b.check("低ランク近似の誤差 = 捨てた特異値の 2 乗の和", abs(((Xc - Xr) ** 2).sum() - (d[2:] ** 2).sum()) < 1e-6, "")
b.num("2 つ残した誤差", w[2:].sum(), "0.65")
b.num("その割合 %", w[2:].sum() / 4 * 100, "16", tol=0.5)
# 章末問題
b.num("問1 寄与率 ρ=0.6", (1 + 0.6) / 2, "0.8")
b.num("問2", np.sqrt(3.2) * 0.5, "0.894")
b.num("問3", (2.8 + 1.4) / 6, "0.70")
b.done()
