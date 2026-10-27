"""準1級 教科書 第28章「分割表」の検算（scipy・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 28)
a, bb, c, d = 30, 170, 15, 285
b.num("p1", a / 200, "0.15")
b.num("p0", c / 300, "0.05")
b.num("RD", a / 200 - c / 300, "0.10")
b.num("RR", (a / 200) / (c / 300), "3.0")
OR = a * d / (bb * c)
b.num("OR", OR, "3.35")
se = math.sqrt(1 / a + 1 / bb + 1 / c + 1 / d)
b.num("SE", se, "0.331")
b.num("OR 下限", math.exp(math.log(OR) - 1.96 * se), "1.75")
b.num("OR 上限", math.exp(math.log(OR) + 1.96 * se), "6.41")
seR = math.sqrt(1 / a - 1 / 200 + 1 / c - 1 / 300)
b.num("RR 下限", math.exp(math.log(3) - 1.96 * seR), "1.66")
b.num("RR 上限", math.exp(math.log(3) + 1.96 * seR), "5.43")
rng = np.random.default_rng(28)
A = rng.binomial(200, 0.15, 200000)
C = rng.binomial(300, 0.05, 200000)
ok = (A > 0) & (C > 0)
lor = np.log(A[ok] * (300 - C[ok]) / ((200 - A[ok]) * C[ok]))
b.check("log OR の SE の式（シミュレーション）", abs(lor.std() - se) / se < 0.05, lor.std())
T = np.array([[a, bb], [c, d]])
b.num("χ² 修正なし", st.chi2_contingency(T, correction=False)[0], "14.65")
b.num("χ² 修正あり", st.chi2_contingency(T, correction=True)[0], "13.46")
Ex = st.chi2_contingency(T, correction=False)[3]
r = (T - Ex) / np.sqrt(Ex)
b.num("残差（暴露・発症）", r[0, 0], "2.83")
b.num("残差（非暴露・発症）", r[1, 0], "-2.31")
b.num("φ", math.sqrt(st.chi2_contingency(T, correction=False)[0] / 500), "0.171")
xx = np.repeat([1, 1, 0, 0], [a, bb, c, d])
yy = np.repeat([1, 0, 1, 0], [a, bb, c, d])
b.check("φ = 0/1 変数の相関", abs(np.corrcoef(xx, yy)[0, 1] - math.sqrt(st.chi2_contingency(T, correction=False)[0] / 500)) < 1e-12, "")
b.num("マクネマー", (25 - 10) ** 2 / 35, "6.43")
b.num("マクネマー p", st.chi2.sf(225 / 35, 1), "0.011")
b.num("マクネマー 修正", (15 - 1) ** 2 / 35, "5.6")
S = [np.array([[20, 80], [10, 90]]), np.array([[40, 60], [30, 70]])]
b.num("層 1 OR", 20 * 90 / (80 * 10), "2.25")
b.num("層 2 OR", 40 * 70 / (60 * 30), "1.56")
mh = sum(t[0, 0] * t[1, 1] / t.sum() for t in S) / sum(t[0, 1] * t[1, 0] / t.sum() for t in S)
b.num("MH OR", mh, "1.77")
Aobs = sum(t[0, 0] for t in S)
E = sum(t[0].sum() * t[:, 0].sum() / t.sum() for t in S)
V = sum(t[0].sum() * t[1].sum() * t[:, 0].sum() * t[:, 1].sum() / (t.sum() ** 2 * (t.sum() - 1)) for t in S)
b.num("a の合計", Aobs, "60")
b.num("期待値の合計", E, "50")
b.num("分散の合計", V, "17.84")
b.num("CMH", (Aobs - E) ** 2 / V, "5.61")
b.num("CMH p", st.chi2.sf((Aobs - E) ** 2 / V, 1), "0.018")
# 章末問題
OR1 = 40 * 160 / (60 * 40)
s1 = math.sqrt(1 / 40 + 1 / 60 + 1 / 40 + 1 / 160)
b.num("問1 OR", OR1, "2.67")
b.num("問1 SE", s1, "0.270")
b.num("問1 log OR", math.log(OR1), "0.981")
b.num("問1 1.96SE", 1.96 * s1, "0.529")
b.num("問1 下限", math.exp(math.log(OR1) - 1.96 * s1), "1.57")
b.num("問1 上限", math.exp(math.log(OR1) + 1.96 * s1), "4.53")
b.num("問2", 100 / 20, "5")
n1, n2 = 65, 50
num = 10 * 30 / n1 + 8 * 24 / n2
den = 20 * 5 / n1 + 12 * 6 / n2
b.num("問3 分子", num, "8.455")
b.num("問3 分母", den, "2.978")
b.num("問3 MH", num / den, "2.84")
b.done()
