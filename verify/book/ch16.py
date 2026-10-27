"""準1級 教科書 第16章「重回帰分析」の検算（行列計算・シミュレーション）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 16)
area = np.array([6.2, 7.5, 5.8, 8.9, 7.1, 9.4, 6.6, 8.0, 7.8, 5.5, 8.6, 6.9])
age = np.array([12, 5, 20, 8, 15, 3, 25, 10, 18, 30, 6, 14.])
price = np.array([31.5, 42.0, 25.8, 48.6, 33.9, 55.2, 26.1, 43.4, 35.0, 20.3, 49.1, 32.8])
b.check("表の床面積", b.has("| 床面積 | " + " | ".join(f"{v:g}" if v % 1 else f"{v:.1f}" for v in area) + " |"), "")
b.check("表の価格", b.has("| 価格 | 31.5 | 42.0 | 25.8 | 48.6 | 33.9 | 55.2 | 26.1 | 43.4 | 35.0 | 20.3 | 49.1 | 32.8 |"), "")
X = np.column_stack([np.ones(12), area, age])
XtX = X.T @ X
bh = np.linalg.solve(XtX, X.T @ price)
e = price - X @ bh
rss = e @ e
s2 = rss / 9
se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
t = bh / se
for v, s in zip(bh, ["4.04", "5.484", "-0.536"]):
    b.num("係数", v, s)
b.num("係数（本文の式）", bh[1], "5.48")
for v, s in zip(se, ["5.12", "0.562", "0.084"]):
    b.num("標準誤差", v, s)
for v, s in zip(t, ["0.79", "9.77", "-6.39"]):
    b.num("t", v, s)
b.num("切片の p", 2 * st.t.sf(abs(t[0]), 9), "0.45")
b.check("残差は X の列と直交", np.allclose(X.T @ e, 0), X.T @ e)
X1 = np.column_stack([np.ones(12), area])
b1 = np.linalg.lstsq(X1, price, rcond=None)[0]
b.num("単回帰の傾き", b1[1], "8.33")
b.num("床面積と築年数の相関", np.corrcoef(area, age)[0, 1], "-0.79")
tss = ((price - price.mean()) ** 2).sum()
R2 = 1 - rss / tss
b.num("s²", s2, "1.97")
b.num("R²", R2, "0.986")
b.num("調整済み R²", 1 - (1 - R2) * 11 / 9, "0.983")
F = ((tss - rss) / 2) / (rss / 9)
b.num("F", F, "316.8")
H = X @ np.linalg.inv(XtX) @ X.T
b.check("H は対称・べき等・トレース 3", np.allclose(H, H.T) and np.allclose(H @ H, H) and abs(np.trace(H) - 3) < 1e-9, "")
b.num("てこ比の平均", np.trace(H) / 12, "0.25")
b.num("物件 10 のてこ比", H[9, 9], "0.43")
b.check("最大のてこ比は物件 10", int(np.argmax(np.diag(H))) == 9, "")
rss1 = ((price - X1 @ b1) ** 2).sum()
b.num("単回帰の RSS", rss1, "97.9")
b.num("RSS", rss, "17.7")
Fp = (rss1 - rss) / (rss / 9)
b.num("部分 F", Fp, "40.8")
b.check("部分 F = t²", abs(Fp - t[2] ** 2) < 1e-9, "")
sb = bh[1:] * np.array([area.std(ddof=1), age.std(ddof=1)]) / price.std(ddof=1)
b.num("標準化（床面積）", sb[0], "0.63")
b.num("標準化（築年数）", sb[1], "-0.41")
r2 = np.corrcoef(area, age)[0, 1] ** 2
b.num("R_j²", r2, "0.629")
b.num("VIF", 1 / (1 - r2), "2.69")
x0 = np.array([1, 7.0, 10.])
yh = x0 @ bh
v = x0 @ np.linalg.inv(XtX) @ x0
tc = st.t.ppf(0.975, 9)
b.num("予測値", yh, "37.07")
b.num("x0'(X'X)^{-1}x0", v, "0.209")
b.num("信頼の下限", yh - tc * np.sqrt(s2 * v), "35.6")
b.num("信頼の上限", yh + tc * np.sqrt(s2 * v), "38.5")
b.num("予測の下限", yh - tc * np.sqrt(s2 * (1 + v)), "33.6")
b.num("予測の上限", yh + tc * np.sqrt(s2 * (1 + v)), "40.6")
# 不偏性・s² の不偏性（シミュレーション）
rng = np.random.default_rng(16)
sims = []
s2s = []
for _ in range(20000):
    yy = X @ bh + rng.normal(0, 1.4, 12)
    bb = np.linalg.solve(XtX, X.T @ yy)
    sims.append(bb)
    s2s.append(((yy - X @ bb) ** 2).sum() / 9)
sims = np.array(sims)
b.check("V[β̂] = σ²(X'X)^{-1}（シミュレーション）", np.allclose(np.cov(sims.T), 1.96 * np.linalg.inv(XtX), rtol=0.05, atol=1e-4), "")
b.check("E[s²] = σ²", abs(np.mean(s2s) - 1.96) < 0.03, np.mean(s2s))
# 章末問題
b.num("問3 F", ((150 - 120) / 2) / (120 / 24), "3.0")
b.num("問3 F_0.05(2,24)", st.f.ppf(0.95, 2, 24), "3.403")
b.num("問4 VIF", 1 / (1 - 0.9), "10")
b.num("問4 √10", np.sqrt(10), "3.16")
b.done()
