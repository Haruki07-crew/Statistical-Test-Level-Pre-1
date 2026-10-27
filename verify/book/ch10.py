"""準1級 教科書 第10章「検定の基礎と検定法の導出」の検算（正確な分布・数値計算）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 10)
rng = np.random.default_rng(10)
b.num("境目 n=10", 1.645 / math.sqrt(10), "0.520")
b.num("検出力", st.norm.sf(1.645 - math.sqrt(10)), "0.935")
B0, B1 = st.binom(10, 0.5), st.binom(10, 0.7)
b.num("P(X≥9)", B0.sf(8), "0.0107")
b.num("P(X≥8)", B0.sf(7), "0.0547")
b.num("P(X=8)", B0.pmf(8), "0.0439")
g = (0.05 - B0.sf(8)) / B0.pmf(8)
b.num("γ", g, "0.893")
b.check("サイズがちょうど 0.05", abs(B0.sf(8) + g * B0.pmf(8) - 0.05) < 1e-12, "")
b.num("無作為化検定の検出力", B1.sf(8) + g * B1.pmf(8), "0.358")
# p 値は一様
z = rng.normal(size=10 ** 6)
pv = st.norm.sf(z)
b.check("p 値は H0 のもとで一様", abs(np.mean(pv <= 0.3) - 0.3) < 0.002, "")
# ポアソン
G = 2 * (30 * math.log(30 / 24) + 18 * math.log(18 / 24))
b.num("共通の率", 48 / 20, "2.4")
b.num("G", G, "3.03")
b.num("χ²_0.05(1)", st.chi2.ppf(0.95, 1), "3.841")
b.num("p", st.chi2.sf(G, 1), "0.082")
# 3 つの検定
w = (0.6 - 0.5) / math.sqrt(0.24 / 100)
s = (0.6 - 0.5) / math.sqrt(0.25 / 100)
lr = 2 * (60 * math.log(1.2) + 40 * math.log(0.8))
b.num("ワルド", w, "2.041")
b.num("ワルド p", 2 * st.norm.sf(w), "0.041")
b.num("スコア", s, "2.000")
b.num("スコア p", 2 * st.norm.sf(s), "0.046")
b.num("尤度比", lr, "4.027")
b.num("√尤度比", math.sqrt(lr), "2.007")
b.num("尤度比 p", st.chi2.sf(lr, 1), "0.045")
# ウィルクス（シミュレーション：正規 3 群の平均の比較、自由度 2）
X = rng.normal(0, 1, (100000, 3, 30))
tot = ((X - X.mean((1, 2), keepdims=True)) ** 2).sum((1, 2))
wit = ((X - X.mean(2, keepdims=True)) ** 2).sum((1, 2))
lrt = 90 * np.log(tot / wit)
b.check("ウィルクス：自由度 2（シミュレーション、各群 30 個なので漸近の誤差を許して ±0.008）", abs(np.mean(lrt > st.chi2.ppf(0.95, 2)) - 0.05) < 0.008, np.mean(lrt > st.chi2.ppf(0.95, 2)))
# 多重検定
b.num("20 回で 1 つ以上", 1 - 0.95 ** 20, "0.64")
p10 = np.array([0.001, 0.004, 0.012, 0.018, 0.030, 0.041, 0.090, 0.210, 0.450, 0.720])


def holm(p, a=0.05):
    m, k = len(p), 0
    for i, q in enumerate(np.sort(p)):
        if q <= a / (m - i):
            k += 1
        else:
            break
    return k


def bh(p, q=0.05):
    m = len(p)
    ks = [i + 1 for i, v in enumerate(np.sort(p)) if v <= (i + 1) / m * q]
    return max(ks) if ks else 0


b.check("ボンフェローニ 2・ホルム 2・BH 4・補正なし 6", ((p10 <= 0.005).sum(), holm(p10), bh(p10), (p10 <= 0.05).sum()) == (2, 2, 4, 6), "")
b.num("ホルムの 2 番目の基準", 0.05 / 9, "0.0056")
b.num("ホルムの 3 番目の基準", 0.05 / 8, "0.00625")
# 標本サイズ
b.num("片側", ((1.645 + 0.8416) * 10 / 5) ** 2, "24.7")
b.num("両側", ((1.96 + 0.8416) * 10 / 5) ** 2, "31.4")
# 章末問題
p5 = np.array([0.004, 0.011, 0.019, 0.032, 0.27])
b.check("問3 ボンフェローニ 1・ホルム 2・BH 4", ((p5 <= 0.01).sum(), holm(p5), bh(p5)) == (1, 2, 4), "")
b.num("問3 基準", 0.05 / 3, "0.0167")
lam, lam0, x, n = sp.symbols("lambda lambda0 x n", positive=True)
xb = sp.symbols("xbar", positive=True)
U = n * xb / lam0 - n
b.check("問4 スコア統計量", sp.simplify(U ** 2 / (n / lam0) - n * (xb - lam0) ** 2 / lam0) == 0, "")
b.done()
