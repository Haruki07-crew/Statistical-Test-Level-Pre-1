"""準1級 教科書 第9章「区間推定」の検算（正確な二項確率の和・scipy・数値解）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import optimize as op, stats as st

b = Book("pre1", 9)
z = st.norm.ppf(0.975)


def interval(method, k, n):
    ph = k / n
    if method == "wald":
        h = z * np.sqrt(ph * (1 - ph) / n)
        return ph - h, ph + h
    if method == "wilson":
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = z / (1 + z * z / n) * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        return c - h, c + h
    L = np.where(k == 0, 0, st.beta.ppf(0.025, np.maximum(k, 1), n - k + 1))
    U = np.where(k == n, 1, st.beta.ppf(0.975, k + 1, np.maximum(n - k, 1)))
    return L, U


def cov(method, p, n=20):
    k = np.arange(n + 1)
    L, U = interval(method, k, n)
    return (st.binom(n, p).pmf(k) * ((L <= p) & (p <= U))).sum()


w = interval("wald", 3, 20)
b.num("ワルドの幅", z * math.sqrt(0.15 * 0.85 / 20), "0.156")
b.num("ワルド下限", w[0], "-0.006")
b.num("ワルド上限", w[1], "0.306")
wl = interval("wilson", 3, 20)
b.num("ウィルソン下限", wl[0], "0.052")
b.num("ウィルソン上限", wl[1], "0.360")
cp = interval("cp", np.array(3), 20)
b.num("CP 下限", float(cp[0]), "0.032")
b.num("CP 上限", float(cp[1]), "0.379")
b.num("ワルドの被覆 p=0.05 %", cov("wald", 0.05) * 100, "64", tol=0.5)
c = b.chart(1)
for i, m in enumerate(["wald", "wilson", "cp"]):
    b.check(f"図の被覆確率（{m}）が計算と一致", np.allclose(c["series"][i]["y"], [round(cov(m, q), 3) for q in c["x"]], atol=1e-9), "")
b.check("CP はすべての p で ≥ 0.95", all(cov("cp", q) >= 0.95 for q in np.linspace(0.01, 0.99, 99)), "")
# 指数分布のピボット
b.num("χ²_0.025(20)", st.chi2.ppf(0.975, 20), "34.17")
b.num("χ²_0.975(20)", st.chi2.ppf(0.025, 20), "9.591")
lo, hi = 40 / st.chi2.ppf(0.975, 20), 40 / st.chi2.ppf(0.025, 20)
b.num("下限", lo, "1.17")
b.num("上限", hi, "4.17")
rng = np.random.default_rng(9)
e = rng.exponential(2, (200000, 10)).sum(1)
b.check("ピボット区間の被覆 ≈ 95%（シミュレーション）", abs(np.mean((40 / 2 * e / 20 * 0 + 2 * e / st.chi2.ppf(0.975, 20) <= 2) & (2 <= 2 * e / st.chi2.ppf(0.025, 20))) - 0.95) < 0.003, "")
# 尤度比
ll = lambda th: -10 * math.log(th) - 20.0 / th
f = lambda th: 2 * (ll(2) - ll(th)) - st.chi2.ppf(0.95, 1)
b.num("χ²_0.05(1)/2", st.chi2.ppf(0.95, 1) / 2, "1.92")
b.num("尤度比 下限", op.brentq(f, 0.5, 2), "1.14")
b.num("尤度比 上限", op.brentq(f, 2, 10), "3.99")
b.num("ワルド型 下限", 2 - 1.96 * 2 / math.sqrt(10), "0.76")
b.num("ワルド型 上限", 2 + 1.96 * 2 / math.sqrt(10), "3.24")
# 片側
b.num("0/50 の上限", 1 - 0.05 ** (1 / 50), "0.0582")
b.num("3/50", 3 / 50, "0.06")
b.num("−log 0.05", -math.log(0.05), "3", tol=0.05)
# 章末問題
b.num("問1 下限", st.chi2.ppf(0.025, 10) / 4, "0.812")
b.num("問1 上限", st.chi2.ppf(0.975, 10) / 4, "5.12")
b.num("問1 χ²_0.975(10)", st.chi2.ppf(0.025, 10), "3.247")
b.num("問1 χ²_0.025(10)", st.chi2.ppf(0.975, 10), "20.48")
w2 = interval("wald", 10, 100)
b.num("問2 ワルド下限", w2[0], "0.041")
b.num("問2 ワルド上限", w2[1], "0.159")
wl2 = interval("wilson", 10, 100)
b.num("問2 ウィルソン中心", (0.1 + z * z / 200) / (1 + z * z / 100), "0.1148")
b.num("問2 ウィルソン下限", wl2[0], "0.055")
b.num("問2 ウィルソン上限", wl2[1], "0.174")
b.num("問3 正確", 1 - 0.05 ** (1 / 30), "0.0950")
b.done()
