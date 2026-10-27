"""準1級 教科書 第23章「判別分析」の検算（行列計算・シミュレーション）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 23)
m1, m2 = np.array([2., 1.]), np.array([4., 3.])
S = np.array([[1, .5], [.5, 2.]])
Si = np.linalg.inv(S)
b.check("Σ⁻¹ = (1/1.75)[[2,−0.5],[−0.5,1]]", np.allclose(Si, np.array([[2, -0.5], [-0.5, 1]]) / 1.75), "")
a = Si @ (m2 - m1)
c = a @ (m1 + m2) / 2
b.num("a1", a[0], "1.714")
b.num("a2", a[1], "0.571")
b.num("c", c, "6.286")
x = np.array([3.5, 1.5])
sc = a @ x - c
b.num("判別関数の値", sc, "0.571")
d1 = (x - m1) @ Si @ (x - m1)
d2 = (x - m2) @ Si @ (x - m2)
b.num("D1²", d1, "2.286")
b.num("D2²", d2, "1.143")
b.check("(D1²−D2²)/2 = 判別関数", abs((d1 - d2) / 2 - sc) < 1e-12, "")
b.num("log 4", np.log(4), "1.386")
D2 = (m2 - m1) @ Si @ (m2 - m1)
b.num("Δ²", D2, "4.57")
b.num("Δ", np.sqrt(D2), "2.14")
b.num("誤判別率 %", st.norm.cdf(-np.sqrt(D2) / 2) * 100, "14.3")
rng = np.random.default_rng(23)
X1 = rng.multivariate_normal(m1, S, 200000)
X2 = rng.multivariate_normal(m2, S, 200000)
err = (np.mean(X1 @ a - c > 0) + np.mean(X2 @ a - c < 0)) / 2
b.check("誤判別率（シミュレーション）", abs(err - st.norm.cdf(-np.sqrt(D2) / 2)) < 0.003, err)
# フィッシャーの方向：比を最大にする方向を総当たり
ang = np.linspace(0, np.pi, 20001)
dirs = np.column_stack([np.cos(ang), np.sin(ang)])
ratio = (dirs @ (m2 - m1)) ** 2 / np.einsum("ij,jk,ik->i", dirs, S, dirs)
best = dirs[np.argmax(ratio)]
b.check("フィッシャーの方向 ∝ Σ⁻¹(μ2−μ1)", abs(abs(best @ a) / np.linalg.norm(a) - 1) < 1e-6, best)
# 混同行列
TP, FN, FP, TN = 40, 10, 30, 120
b.num("感度", TP / 50, "0.8")
b.num("特異度", TN / 150, "0.8")
b.num("正解率", (TP + TN) / 200, "0.8")
b.num("適合率", TP / 70, "0.571")
b.num("F", 2 * TP / (2 * TP + FP + FN), "0.667")
auc = st.norm.cdf(np.sqrt(D2) / np.sqrt(2))
b.num("AUC", auc, "0.935")
s1, s2 = X1 @ a, X2 @ a
idx = rng.integers(0, 200000, (2, 300000))
b.check("AUC = P(陽性のスコア > 陰性)（シミュレーション）", abs(np.mean(s2[idx[0]] > s1[idx[1]]) - auc) < 0.003, np.mean(s2[idx[0]] > s1[idx[1]]))
# 章末問題
b.num("問1 誤判別率", st.norm.cdf(-1.5), "0.067")
b.num("問1 境界", 1.5 + np.log(9) / 3, "2.23")
b.num("問2 ユークリッド", np.sqrt(5), "2.24")
b.num("問2 マハラノビス", np.sqrt(2), "1.41")
b.num("問3 感度", 18 / 20, "0.9")
b.num("問3 特異度", 71 / 80, "0.888")
b.num("問3 適合率", 18 / 27, "0.667")
b.num("問3 F", 36 / 47, "0.766")
b.done()
