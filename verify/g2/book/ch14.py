"""2級 教科書 第14章「回帰分析」の数値の検算（行列の最小二乗で計算し、出力の表と照合する）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("g2", 14)


def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    n, k = X.shape
    coef = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ coef
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    sst = ((y - y.mean()) ** 2).sum()
    R2 = 1 - e @ e / sst
    return dict(b=coef, se=se, t=coef / se, p=2 * st.t.sf(abs(coef / se), n - k), R2=R2, adj=1 - (1 - R2) * (n - 1) / (n - k),
                F=(R2 / (k - 1)) / ((1 - R2) / (n - k)), sse=e @ e, s=np.sqrt(s2))


x = np.array([2, 3, 4, 5, 5, 6, 7, 8, 9, 11.])
y = np.array([48, 58, 50, 60, 66, 57, 72, 64, 79, 76.])
z = np.array([3.5, 2, 4, 2.5, 1.5, 3, 2, 2.5, 1, 2])
r = ols([x], y)
b.num("残差平方和", r["sse"], "256.8")
b.num("s²", r["sse"] / 8, "32.1")
b.num("s", r["s"], "5.67")
b.num("SE(b)", r["se"][1], "0.677")
b.num("T", r["t"][1], "4.75")
b.num("t(8)", st.t.ppf(0.975, 8), "2.306")
b.num("p", r["p"][1], "0.0015")
b.num("下限", r["b"][1] - 2.306 * r["se"][1], "1.65")
b.num("上限", r["b"][1] + 2.306 * r["se"][1], "4.78")
rc = r["b"][1] * 0 + np.corrcoef(x, y)[0, 1]
b.check("相関の検定と同じ T", abs(rc * np.sqrt(8) / np.sqrt(1 - rc ** 2) - r["t"][1]) < 1e-9, "")
# 区間
h = 1 / 10 + (10 - 6) ** 2 / 70
b.num("h", h, "0.329")
yh = r["b"][0] + r["b"][1] * 10
b.num("ŷ0", yh, "75.86")
w1, w2 = 2.306 * r["s"] * np.sqrt(h), 2.306 * r["s"] * np.sqrt(1 + h)
b.num("信頼区間の幅", w1, "7.49")
b.num("予測区間の幅", w2, "15.06")
b.num("信頼の下限", yh - w1, "68.4")
b.num("信頼の上限", yh + w1, "83.3")
b.num("予測の下限", yh - w2, "60.8")
b.num("予測の上限", yh + w2, "90.9")
rng = np.random.default_rng(14)
# 予測区間の被覆（モデルを本文の推定値で作ってシミュレーション）
hit = 0
R = 20000
for _ in range(R):
    yy = 43.71 + 3.214 * x + rng.normal(0, 5.67, 10)
    rr = ols([x], yy)
    y0 = rr["b"][0] + rr["b"][1] * 10
    hit += abs(43.71 + 32.14 + rng.normal(0, 5.67) - y0) <= st.t.ppf(0.975, 8) * rr["s"] * np.sqrt(1 + h)
b.check("予測区間の被覆率 ≈ 95%（シミュレーション）", abs(hit / R - 0.95) < 0.006, hit / R)
# 重回帰
m = ols([x, z], y)
for v, s in zip(m["b"], ["65.81", "2.110", "-6.444"]):
    b.num("係数", v, s)
for v, s in zip(m["se"], ["3.748", "0.314", "0.966"]):
    b.num("標準誤差", v, s)
for v, s in zip(m["t"], ["17.56", "6.71", "-6.67"]):
    b.num("t 値", v, s)
b.check("p 値はすべて 0.001 未満", all(m["p"] < 0.001), m["p"])
b.num("R²", m["R2"], "0.964")
b.num("調整済み R²", m["adj"], "0.954")
b.num("F", m["F"], "94.69")
b.num("調整済み（式の値）", 1 - (1 - 0.964) * 9 / 7, "0.954")
b.num("単回帰の調整済み", 1 - (1 - 0.738) * 9 / 8, "0.705")
b.num("勉強とスマホの相関", np.corrcoef(x, z)[0, 1], "-0.527")
b.num("VIF", 1 / (1 - np.corrcoef(x, z)[0, 1] ** 2), "1.39")
b.num("0.527²", 0.527 ** 2, "0.278")
# ダミー
P = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11.])
S = np.array([30, 27, 25, 22, 20, 52, 49, 46, 44, 41.])
D = np.r_[np.zeros(5), np.ones(5)]
b.num("合わせた傾き", ols([P], S)["b"][1], "2.64")
d = ols([P, D], S)
b.num("切片", d["b"][0], "35.2")
b.num("価格の係数", d["b"][1], "-2.6")
b.num("ダミーの係数", d["b"][2], "34.6")
b.num("約 35 個", d["b"][2], "35", tol=0.5)
# 章末問題
b.num("問1 T", 1.8 / 0.75, "2.4")
b.num("問1 t(18)", st.t.ppf(0.975, 18), "2.101")
b.num("問1 幅", 2.101 * 0.75, "1.58")
b.num("問1 下限", 1.8 - 2.101 * 0.75, "0.22")
b.num("問1 上限", 1.8 + 2.101 * 0.75, "3.38")
b.num("問2 t(27)", st.t.ppf(0.975, 27), "2.052")
b.num("問2 調整済み", 1 - 0.6 * 29 / 27, "0.356")
b.num("問3 F", (0.5 / 4) / (0.5 / 20), "5.0")
b.done()
