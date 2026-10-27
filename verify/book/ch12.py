"""準1級 教科書 第12章「一般の分布に関する検定法」の検算（scipy の正確検定と照合）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 12)
B = st.binom(20, 0.5)
b.num("P(X≥15)", B.sf(14), "0.0207")
b.num("両側", st.binomtest(15, 20, 0.5).pvalue, "0.041")
b.num("Z", 5 / math.sqrt(5), "2.24")
b.num("連続修正なし p", 2 * st.norm.sf(5 / math.sqrt(5)), "0.025")
b.num("連続修正あり p", 2 * st.norm.sf(4.5 / math.sqrt(5)), "0.044")
b.num("2 ポアソンの正確な p", st.binomtest(30, 48, 0.5).pvalue, "0.111")
# 適合度
obs = np.array([48, 74, 44, 20, 14])
tot = 74 + 88 + 60 + 10 * 4 + 4 * 5
b.num("傷の総数", tot, "282")
lam = tot / 200
b.num("λ̂", lam, "1.41")
p = st.poisson(lam).pmf(range(4))
p = np.append(p, 1 - p.sum())
E = 200 * p
for v, s in zip(E, ["48.83", "68.85", "48.54", "22.81", "10.97"]):
    b.num("期待度数", v, s)
c = ((obs - E) ** 2 / E).sum()
G = 2 * (obs * np.log(obs / E)).sum()
b.num("χ²", c, "2.01")
b.num("G", G, "1.96")
b.num("χ²_0.05(3)", st.chi2.ppf(0.95, 3), "7.81")
b.num("p", st.chi2.sf(c, 3), "0.57")
b.num("自由度 4 の p", st.chi2.sf(c, 4), "0.73")
rng = np.random.default_rng(12)
# 推定した母数の分だけ自由度を減らす（シミュレーション：ポアソン 5 カテゴリ、λ を推定）
cnt = 0
R = 20000
for _ in range(R):
    x = rng.poisson(1.41, 200)
    lh = x.mean()
    o = np.array([np.sum(x == k) for k in range(4)] + [np.sum(x >= 4)])
    pp = st.poisson(lh).pmf(range(4))
    pp = np.append(pp, 1 - pp.sum())
    e = 200 * pp
    cnt += ((o - e) ** 2 / e).sum() > st.chi2.ppf(0.95, 3)
b.check("自由度 3 で棄却率 ≈ 5%（シミュレーション）", abs(cnt / R - 0.05) < 0.008, cnt / R)
# 過分散
x = np.array([3, 7, 1, 0, 5, 9, 2, 4, 11, 3, 6, 1])
b.num("平均", x.mean(), "4.33")
b.num("不偏分散", x.var(ddof=1), "11.52")
D = 11 * x.var(ddof=1) / x.mean()
b.num("D", D, "29.2")
b.num("χ²_0.05(11)", st.chi2.ppf(0.95, 11), "19.68")
b.num("p", st.chi2.sf(D, 11), "0.002")
# KS
d = np.array([0.12, 0.31, 0.05, 0.48, 0.27, 0.15, 0.62, 0.09, 0.22, 0.38])
ks = st.kstest(d, "expon", args=(0, 0.3))
b.num("D10", ks.statistic, "0.159")
b.num("正確な棄却限界", st.kstwo.ppf(0.95, 10), "0.409")
b.num("漸近の棄却限界", 1.358 / math.sqrt(10), "0.429")
# 章末問題
b.num("問1 0.9^30", 0.9 ** 30, "0.042")
b.num("問1 Z", -3 / math.sqrt(2.7), "-1.83")
b.num("問1 近似 p", st.norm.cdf(-3 / math.sqrt(2.7)), "0.034")
o3 = np.array([30, 50, 20])
e3 = np.full(3, 100 / 3)
b.num("問3 χ²", ((o3 - e3) ** 2 / e3).sum(), "14.0")
b.num("問3 G", 2 * (o3 * np.log(o3 / e3)).sum(), "13.8")
b.num("問4 D", 9 * 5.4 / 5.0, "9.72")
b.num("問4 χ²_0.05(9)", st.chi2.ppf(0.95, 9), "16.92")
b.done()
