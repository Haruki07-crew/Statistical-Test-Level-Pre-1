"""準1級 教科書 第11章「正規分布に関する検定」の検算（scipy・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 11)
rng = np.random.default_rng(11)
# 尤度比 = t
x = rng.normal(0.3, 1, 12)
mu0 = 0
s2h, s20 = ((x - x.mean()) ** 2).mean(), ((x - mu0) ** 2).mean()
T = math.sqrt(12) * (x.mean() - mu0) / x.std(ddof=1)
b.check("Λ^{2/n} = 1/(1+T²/(n−1))", abs(s2h / s20 - 1 / (1 + T ** 2 / 11)) < 1e-12, "")
# 予備検定の問題
R = 100000
xa, ya = rng.normal(0, 2, (R, 10)), rng.normal(0, 1, (R, 30))
pt = st.ttest_ind(xa, ya, axis=1).pvalue
pw = st.ttest_ind(xa, ya, axis=1, equal_var=False).pvalue
F = xa.var(1, ddof=1) / ya.var(1, ddof=1)
pf = 2 * np.minimum(st.f.sf(F, 9, 29), st.f.cdf(F, 9, 29))
two = np.where(pf > 0.05, pt, pw)
b.num("プールした t %", np.mean(pt < 0.05) * 100, "15.5", tol=0.4)
b.num("ウェルチ %", np.mean(pw < 0.05) * 100, "5.2", tol=0.3)
b.num("2 段階 %", np.mean(two < 0.05) * 100, "7.2", tol=0.3)
b.num("F 検定の検出力 %", np.mean(pf < 0.05) * 100, "75", tol=1.5)
xe, ye = rng.normal(0, 2, (R, 20)), rng.normal(0, 1, (R, 20))
b.num("同じ大きさのとき %", np.mean(st.ttest_ind(xe, ye, axis=1).pvalue < 0.05) * 100, "5.3", tol=0.3)
# バートレット
g = [np.array([12.1, 13.4, 11.8, 12.9, 13.0, 12.4]), np.array([14.2, 11.0, 15.8, 12.5, 10.9, 14.6]), np.array([12.6, 12.9, 13.3, 12.2, 12.8, 13.1])]
for v, s in zip([y.var(ddof=1) for y in g], ["0.364", "4.067", "0.150"]):
    b.num("不偏分散", v, s)
B = st.bartlett(*g)
b.num("バートレット", B.statistic, "12.75")
b.num("p", B.pvalue, "0.0017")
b.num("χ²_0.05(2)", st.chi2.ppf(0.95, 2), "5.991")
N, k = 18, 3
sp2 = sum(5 * y.var(ddof=1) for y in g) / (N - k)
C = 1 + 1 / (3 * (k - 1)) * (3 / 5 - 1 / (N - k))
b.check("本文の式 = scipy", abs(((N - k) * math.log(sp2) - sum(5 * math.log(y.var(ddof=1)) for y in g)) / C - B.statistic) < 1e-9, "")
xt, yt = rng.standard_t(5, (R, 20)), rng.standard_t(5, (R, 20))
Ft = xt.var(1, ddof=1) / yt.var(1, ddof=1)
b.num("t5 での F 検定の棄却 %", np.mean((Ft > st.f.ppf(0.975, 19, 19)) | (Ft < st.f.ppf(0.025, 19, 19))) * 100, "16", tol=0.6)
# z 変換
z = math.atanh(0.6)
se = 1 / math.sqrt(27)
b.num("z", z, "0.693")
b.num("SE", se, "0.192")
b.num("下限", math.tanh(z - 1.96 * se), "0.306")
b.num("上限", math.tanh(z + 1.96 * se), "0.790")
zz = (z - math.atanh(0.3)) / se
b.num("検定統計量", zz, "1.99")
b.num("p", 2 * st.norm.sf(zz), "0.046")
r = np.array([np.corrcoef(*rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 30).T)[0, 1] for _ in range(20000)])
b.check("z 変換の分散 ≈ 1/(n−3)（シミュレーション）", abs(np.arctanh(r).var() - 1 / 27) < 0.003, np.arctanh(r).var())
# 検出力
def pw2(n, d=0.5):
    dd = d * math.sqrt(n / 2)
    df = 2 * n - 2
    c = st.t.ppf(0.975, df)
    return st.nct(df, dd).sf(c) + st.nct(df, dd).cdf(-c)
b.check("各群 64 人で 0.8 以上、63 人で未満", pw2(64) >= 0.8 > pw2(63), (pw2(63), pw2(64)))
b.num("63 人の検出力", pw2(63), "0.795")
b.num("正規近似", 2 * ((1.96 + 0.8416) / 0.5) ** 2, "62.8")
# ホテリング
S = np.array([[4, 1], [1, 2.]])
d = np.array([1, 0.5])
T2 = 20 * d @ np.linalg.inv(S) @ d
b.num("T²", T2, "5.71")
Fh = 18 / (2 * 19) * T2
b.num("F", Fh, "2.71")
b.num("F_0.05(2,18)", st.f.ppf(0.95, 2, 18), "3.555")
b.num("p", st.f.sf(Fh, 2, 18), "0.094")
b.num("t1", 1.0 / (2 / math.sqrt(20)), "2.24")
b.num("t1 の p", 2 * st.t.sf(1.0 / (2 / math.sqrt(20)), 19), "0.038")
b.num("t2", 0.5 / (math.sqrt(2) / math.sqrt(20)), "1.58")
# 章末問題
a1, a2 = 10 / 8, 2 / 20
b.num("問1 ν", (a1 + a2) ** 2 / (a1 ** 2 / 7 + a2 ** 2 / 19), "8.15")
z2 = math.atanh(0.4)
b.num("問2 z", z2, "0.424")
b.num("問2 SE", 1 / math.sqrt(47), "0.146")
b.num("問2 下限", math.tanh(z2 - 1.96 / math.sqrt(47)), "0.14")
b.num("問2 上限", math.tanh(z2 + 1.96 / math.sqrt(47)), "0.61")
b.done()
