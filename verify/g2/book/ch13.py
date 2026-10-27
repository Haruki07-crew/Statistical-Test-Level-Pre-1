"""2級 教科書 第13章「2標本の検定とカイ二乗検定」の数値の検算（scipy の検定関数と照合）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("g2", 13)
sp2 = (11 * 64 + 9 * 49) / 20
se = np.sqrt(sp2 * (1 / 12 + 1 / 10))
T = 7 / se
b.num("T", T, "2.16")
b.num("p", 2 * st.t.sf(T, 20), "0.043")
r = st.ttest_ind_from_stats(68, 8, 12, 61, 7, 10, equal_var=True)
b.check("scipy（等分散）と一致", abs(r.statistic - T) < 1e-9, r)
a1, a2 = 64 / 12, 49 / 10
b.num("s1²/n1", a1, "5.333")
b.num("s2²/n2", a2, "4.9")
b.num("和", a1 + a2, "10.233")
Tw = 7 / np.sqrt(a1 + a2)
nu = (a1 + a2) ** 2 / (a1 ** 2 / 11 + a2 ** 2 / 9)
b.num("T'", Tw, "2.19")
b.num("ν", nu, "19.9")
b.num("ウェルチの p", 2 * st.t.sf(Tw, nu), "0.041")
rw = st.ttest_ind_from_stats(68, 8, 12, 61, 7, 10, equal_var=False)
b.check("scipy（ウェルチ）と一致", abs(rw.statistic - Tw) < 1e-9 and abs(rw.pvalue - 2 * st.t.sf(Tw, nu)) < 1e-9, rw)
b.num("F", 64 / 49, "1.31")
b.num("F_0.025(11,9)", st.f.ppf(0.975, 11, 9), "3.91")
# 対応のある
bef = np.array([148, 135, 152, 160, 141, 138, 155, 147])
aft = np.array([142, 133, 145, 151, 140, 131, 150, 144])
d = bef - aft
b.check("表の差の行", b.has("| 差（前 − 後） | " + " | ".join(map(str, d)) + " |"), d)
b.num("d̄", d.mean(), "5.0")
b.num("s_d", d.std(ddof=1), "2.78")
Tp = d.mean() / (d.std(ddof=1) / np.sqrt(8))
b.num("T（対応あり）", Tp, "5.09")
b.num("t_0.025(7)", st.t.ppf(0.975, 7), "2.365")
b.num("p", 2 * st.t.sf(Tp, 7), "0.0014")
b.check("scipy（対応あり）と一致", abs(st.ttest_rel(bef, aft).statistic - Tp) < 1e-9, "")
wr = st.ttest_ind(bef, aft)
b.num("対応を無視した T", wr.statistic, "1.26")
b.num("対応を無視した p", wr.pvalue, "0.23")
b.num("前の sd", bef.std(ddof=1), "8.6")
b.num("後の sd", aft.std(ddof=1), "7.2")
b.num("前後の相関", np.corrcoef(bef, aft)[0, 1], "0.95")
# 比率の差
p = 230 / 900
b.num("合わせた比率", p, "0.256")
z = 0.08 / np.sqrt(p * (1 - p) * (1 / 400 + 1 / 500))
b.num("Z", z, "2.73")
b.num("p 値", 2 * st.norm.sf(z), "0.006")
chi2x2 = st.chi2_contingency(np.array([[120, 280], [110, 390]]), correction=False)[0]
b.check("2×2 のカイ二乗 = z²", abs(chi2x2 - z ** 2) < 1e-9, (chi2x2, z ** 2))
# 適合度
O = np.array([15, 22, 18, 25, 16, 24])
b.check("合計 120", O.sum() == 120, "")
chi = ((O - 20) ** 2).sum() / 20
b.num("偏差 2 乗の和", ((O - 20) ** 2).sum(), "90")
b.num("χ²", chi, "4.5")
b.num("上側 5% 点", st.chi2.ppf(0.95, 5), "11.07")
b.num("p", st.chi2.sf(chi, 5), "0.48")
b.check("scipy（適合度）と一致", abs(st.chisquare(O).statistic - chi) < 1e-9, "")
# 独立性
Tab = np.array([[12, 138], [14, 36]])
res = st.chi2_contingency(Tab, correction=False)
for v, s in zip(res[3].ravel(), ["19.5", "130.5", "6.5", "43.5"]):
    b.num("期待度数", v, s)
b.num("χ²", res[0], "13.26")
b.num("χ²_0.05(1)", st.chi2.ppf(0.95, 1), "3.841")
b.num("χ²_0.01(1)", st.chi2.ppf(0.99, 1), "6.635")
b.num("p", res[1], "0.0003")
# 相関
x = np.array([2, 3, 4, 5, 5, 6, 7, 8, 9, 11])
y = np.array([48, 58, 50, 60, 66, 57, 72, 64, 79, 76])
r_ = np.corrcoef(x, y)[0, 1]
Tc = r_ * np.sqrt(8) / np.sqrt(1 - r_ ** 2)
b.num("T（相関）", Tc, "4.75")
b.num("t_0.025(8)", st.t.ppf(0.975, 8), "2.306")
b.num("p", st.pearsonr(x, y).pvalue, "0.0015")
# 章末問題
s = (7 * 4 + 7 * 5.76) / 14
b.num("問1 プール", s, "4.88")
b.num("問1 SE", np.sqrt(s / 4), "1.105")
b.num("問1 T", 2.5 / np.sqrt(s / 4), "2.26")
b.num("問1 t(14)", st.t.ppf(0.975, 14), "2.145")
dd = np.array([8, 3, 5, -1, 5])
b.num("問2 d̄", dd.mean(), "4")
b.num("問2 偏差平方和", ((dd - 4) ** 2).sum(), "44")
b.num("問2 s_d", dd.std(ddof=1), "3.317")
b.num("問2 T", 4 / (dd.std(ddof=1) / np.sqrt(5)), "2.70")
b.num("問2 t_0.05(4)", st.t.ppf(0.95, 4), "2.132")
acc = np.array([10, 8, 12, 9, 11, 14, 6])
b.check("問3 合計 70", acc.sum() == 70, "")
b.num("問3 χ²", ((acc - 10) ** 2).sum() / 10, "4.2")
b.num("問3 χ²_0.05(6)", st.chi2.ppf(0.95, 6), "12.59")
b.num("問4 χ²_0.05(2)", st.chi2.ppf(0.95, 2), "5.991")
b.done()
