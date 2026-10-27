"""準1級 教科書 第13章「ノンパラメトリック法」の検算（総当たりの正確な分布と scipy）。"""
import itertools
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 13)
d = np.array([1.8, 0.6, 2.4, -0.4, 1.1, 0.9, 3.0, 1.5, -0.7, 12.0])
b.check("本文のデータ", b.has("1.8, 0.6, 2.4, −0.4, 1.1, 0.9, 3.0, 1.5, −0.7, 12.0"), "")
t = st.ttest_1samp(d, 0)
b.num("t", t.statistic, "1.94")
b.num("t の p", t.pvalue, "0.084")
b.num("平均", d.mean(), "2.22")
b.num("sd", d.std(ddof=1), "3.62")
b.num("符号検定 片側", st.binom(10, 0.5).sf(7), "0.055")
b.check("56/1024", st.binom(10, 0.5).sf(7) * 1024 == 56, "")
b.num("符号検定 両側", st.binomtest(8, 10, 0.5).pvalue, "0.109")
r = st.rankdata(np.abs(d))
b.check("負の順位は 1 と 3", sorted(r[d < 0]) == [1, 3], r)
Tm = r[d < 0].sum()
b.num("T−", Tm, "4")
b.num("T+", 55 - Tm, "51")
cnt = sum(1 for s in itertools.product([0, 1], repeat=10) if sum(i + 1 for i, v in enumerate(s) if v) <= 4)
b.check("T−≤4 は 7 通り", cnt == 7, cnt)
b.num("片側", 7 / 1024, "0.0068")
b.num("両側", st.wilcoxon(d, method="exact").pvalue, "0.014")
b.num("E[T+]", 10 * 11 / 4, "27.5")
b.num("V[T+]", 10 * 11 * 21 / 24, "96.25")
# 2 標本
A = np.array([12, 15, 9, 20, 17, 14.])
B = np.array([8, 11, 7, 13, 10, 6, 5.])
allv = np.concatenate([A, B])
rk = st.rankdata(allv)
b.check("A の順位 8,11,5,13,12,10", list(rk[:6]) == [8, 11, 5, 13, 12, 10], rk[:6])
W = rk[:6].sum()
b.num("W", W, "59")
b.num("E[W]", 6 * 14 / 2, "42")
b.num("V[W]", 6 * 7 * 14 / 12, "49")
U = W - 21
b.num("U", U, "38")
b.check("U = X>Y の組の数", U == sum(1 for a in A for c in B if a > c), "")
b.num("P(X>Y)", U / 42, "0.90")
b.num("正確な両側 p", st.mannwhitneyu(A, B, method="exact").pvalue, "0.014")
b.num("z", (W - 42) / 7, "2.43")
b.num("近似 p", 2 * st.norm.sf((W - 42) / 7), "0.015")
obs = A.mean() - B.mean()
b.num("平均の差", obs, "5.93")
c = 0
for idx in itertools.combinations(range(13), 6):
    m = np.zeros(13, bool)
    m[list(idx)] = True
    c += allv[m].mean() - allv[~m].mean() >= obs - 1e-12
b.check("並べ替え：1716 通り中 12 通り", math.comb(13, 6) == 1716 and c == 12, c)
b.num("並べ替えの両側 p", 2 * c / 1716, "0.014")
b.num("t 検定の p", st.ttest_ind(A, B).pvalue, "0.009")
# KW
kw = st.kruskal([23, 27, 31, 25], [30, 35, 29, 33], [38, 36, 41, 34])
b.num("H", kw.statistic, "8.12")
b.num("χ²_0.05(2)", st.chi2.ppf(0.95, 2), "5.99")
# 順位相関
x = np.arange(1, 9)
y = np.array([2, 1, 4, 3, 6, 8, 5, 7])
b.num("Σd²", ((x - y) ** 2).sum(), "14")
b.num("r_s", st.spearmanr(x, y).statistic, "0.833")
C = sum(1 for i in range(8) for j in range(i + 1, 8) if (x[j] - x[i]) * (y[j] - y[i]) > 0)
b.check("一致対 23・不一致対 5", C == 23 and 28 - C == 5, C)
b.num("τ", st.kendalltau(x, y).statistic, "0.643")
# ARE
b.num("3/π", 3 / math.pi, "0.955")
b.num("2/π", 2 / math.pi, "0.637")
b.num("下限 108/125", 108 / 125, "0.864")
rng = np.random.default_rng(13)
R = 20000
pt = pw = 0
for _ in range(R):
    u = rng.laplace(0.35, 1, 30)
    pt += st.ttest_1samp(u, 0).pvalue < 0.05
    pw += st.wilcoxon(u).pvalue < 0.05
b.check("ラプラス分布では符号付き順位検定のほうが検出力が高い（シミュレーション）", pw > pt, (pt / R, pw / R))
# 章末問題
b.num("問1", 1 / 32, "0.031")
b.num("問1 両側の最小", 2 / 32, "0.063")
X1, Y1 = [3, 7, 9], [1, 4, 5, 8]
U1 = sum(1 for a in X1 for c in Y1 if a > c)
rk1 = st.rankdata(X1 + Y1)
b.check("問2 U=8・W=14", U1 == 8 and rk1[:3].sum() == 14, (U1, rk1[:3].sum()))
b.num("問3", st.spearmanr([1, 2, 3, 4, 5], [2, 1, 3, 5, 4]).statistic, "0.8")
b.done()
