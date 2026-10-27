"""準1級 教科書 第17章「回帰診断法」の検算（行列計算・除いて当てはめ直す直接計算・シミュレーション）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 17)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 20.])
y = np.array([3.1, 5.6, 8.4, 9.7, 12.5, 13.6, 16.8, 17.5, 20.9, 25.0])
b.check("表の y", b.has("| $y$ | 3.1 | 5.6 | 8.4 | 9.7 | 12.5 | 13.6 | 16.8 | 17.5 | 20.9 | 25.0 |"), "")
X = np.column_stack([np.ones(10), x])
fit = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
bh = fit(X, y)
e = y - X @ bh
H = X @ np.linalg.inv(X.T @ X) @ X.T
h = np.diag(H)
s = np.sqrt(e @ e / 8)
r = e / (s * np.sqrt(1 - h))
# s_(i) を、実際に除いて当てはめ直して求める
si = np.array([np.sqrt(((np.delete(y, i) - np.delete(X, i, 0) @ fit(np.delete(X, i, 0), np.delete(y, i))) ** 2).sum() / 7) for i in range(10)])
t = e / (si * np.sqrt(1 - h))
D = r ** 2 * h / (2 * (1 - h))
dff = t * np.sqrt(h / (1 - h))
b.num("傾き", bh[1], "1.15")
b.num("R²", np.corrcoef(x, y)[0, 1] ** 2, "0.82")
b9 = fit(X[:9], y[:9])
b.num("除いた傾き", b9[1], "2.13")
b.num("除いた R²", np.corrcoef(x[:9], y[:9])[0, 1] ** 2, "0.99")
b.num("延ばした値", b9[0] + 20 * b9[1], "43.9")
b.num("残差（x=20）", e[9], "-3.89")
b.num("残差（x=9）", e[8], "4.70")
b.num("残差（x=1）", e[0], "-3.86")
b.num("てこ比", h[9], "0.794")
b.num("標準化残差", r[9], "-2.78")
b.num("スチューデント化残差", t[9], "-14.75")
b.num("クック", D[9], "14.96")
b.check("他のクックはすべて 0.3 未満、最大は x=1", D[:9].max() < 0.3 and int(np.argmax(D[:9])) == 0, D[:9].round(3))
b.num("x=1 のクック", D[0], "0.275")
b.num("DFFITS", dff[9], "-29.0")
b.num("DFFITS の目安", 2 * np.sqrt(2 / 10), "0.89")
yhat_drop = np.array([X[i] @ fit(np.delete(X, i, 0), np.delete(y, i)) for i in range(10)])
b.check("DFFITS の定義（除いた予測値の変化）と一致", np.allclose((X @ bh - yhat_drop) / (si * np.sqrt(h)), dff), "")
b.check("h の合計 2", abs(h.sum() - 2) < 1e-9, "")
# BP と頑健 SE
rng = np.random.default_rng(17)
xx = np.linspace(1, 10, 40)
yy = 2 + 1.5 * xx + rng.normal(0, 0.3 * xx)
XX = np.column_stack([np.ones(40), xx])
bb = fit(XX, yy)
ee = yy - XX @ bb
u = ee ** 2
g = fit(XX, u)
R2 = 1 - ((u - XX @ g) ** 2).sum() / ((u - u.mean()) ** 2).sum()
b.num("BP", 40 * R2, "5.40")
b.num("BP の p", st.chi2.sf(40 * R2, 1), "0.020")
XtXi = np.linalg.inv(XX.T @ XX)
se_ols = np.sqrt(np.diag(ee @ ee / 38 * XtXi))
hc0 = np.sqrt(np.diag(XtXi @ (XX.T @ (XX * u[:, None])) @ XtXi))
hh = np.diag(XX @ XtXi @ XX.T)
hc3 = np.sqrt(np.diag(XtXi @ (XX.T @ (XX * (u / (1 - hh) ** 2)[:, None])) @ XtXi))
b.num("ふつうの SE", se_ols[1], "0.125")
b.num("HC0", hc0[1], "0.105")
b.num("HC3", hc3[1], "0.111")
# DW
e2 = np.array([0.8, 1.1, 0.6, 0.2, -0.3, -0.9, -1.2, -0.6, 0.1, 0.5, 0.9, 0.4])
b.num("DW", (np.diff(e2) ** 2).sum() / (e2 ** 2).sum(), "0.42")
# 章末問題
xp = np.array([2, 4, 6, 8, 30.])
b.num("問1 S_xx", ((xp - 10) ** 2).sum(), "520")
b.num("問1 h", 1 / 5 + 400 / 520, "0.969")
b.num("問2 D", 4 / 2 * 0.3 / 0.7, "0.857")
b.num("問3 DW", 2 * 1.4, "2.8")
b.done()
