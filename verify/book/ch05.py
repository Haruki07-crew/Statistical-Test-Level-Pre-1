"""準1級 教科書 第5章「離散型分布」の検算（scipy の分布・記号計算・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 5)
rng = np.random.default_rng(5)
# 最頻値
for n, p in [(20, 0.3), (10, 0.5), (7, 0.25)]:
    pm = st.binom(n, p).pmf(range(n + 1))
    b.check(f"最頻値 = floor((n+1)p)（n={n}, p={p}）", int(np.argmax(pm)) == math.floor((n + 1) * p) or abs(pm[math.floor((n + 1) * p)] - pm.max()) < 1e-12, "")
for k, (bs, ps) in enumerate([("0.1351", "0.1353"), ("0.2707", "0.2707"), ("0.2709", "0.2707"), ("0.1806", "0.1804")]):
    b.num(f"Bin(1000,0.002) k={k}", st.binom(1000, 0.002).pmf(k), bs)
    b.num(f"Po(2) k={k}", st.poisson(2).pmf(k), ps)
l1, l2, n_ = 4, 2, 5
cond = [st.poisson(l1).pmf(k) * st.poisson(l2).pmf(n_ - k) / st.poisson(l1 + l2).pmf(n_) for k in range(6)]
b.check("条件付きで Bin(5, 2/3)", np.allclose(cond, st.binom(5, 2 / 3).pmf(range(6))), "")
b.num("さいころの失敗の回数", (5 / 6) / (1 / 6), "5")
g = rng.geometric(1 / 6, 10 ** 6) - 1
b.check("幾何（失敗の回数）の無記憶性（シミュレーション）", abs(np.mean(g[g >= 3] >= 7) - np.mean(g >= 4)) < 0.004, "")
# ガンマ・ポアソン
lam = rng.gamma(2, 2, 10 ** 6)
X = rng.poisson(lam)
b.check("ガンマ・ポアソン混合の平均 4・分散 12", abs(X.mean() - 4) < 0.02 and abs(X.var() - 12) < 0.15, (X.mean(), X.var()))
b.check("混合 = NB(2, 1/3)（P(X=0)）", abs(np.mean(X == 0) - 1 / 9) < 0.002, np.mean(X == 0))
b.num("r", 16 / (12 - 4), "2")
b.num("P(X=0) NB", (1 / 3) ** 2, "0.111")
b.num("P(X=0) Po", math.exp(-4), "0.018")
b.num("10 回以上 NB %", st.nbinom(2, 1 / 3).sf(9) * 100, "7.5")
b.num("10 回以上 Po %", st.poisson(4).sf(9) * 100, "0.8")
b.check("scipy の NB(2,1/3) の平均 4・分散 12", abs(st.nbinom(2, 1 / 3).mean() - 4) < 1e-9 and abs(st.nbinom(2, 1 / 3).var() - 12) < 1e-9, "")
# 超幾何の共分散
N, K = sp.symbols("N K", positive=True)
p = K / N
cov = sp.simplify(K * (K - 1) / (N * (N - 1)) - p ** 2 + p * (1 - p) / (N - 1))
b.check("Cov[I_i, I_j] = −p(1−p)/(N−1)", cov == 0, cov)
nn = sp.symbols("n", positive=True)
V = sp.simplify(nn * p * (1 - p) + nn * (nn - 1) * (-p * (1 - p) / (N - 1)) - nn * p * (1 - p) * (N - nn) / (N - 1))
b.check("V[X] = np(1−p)(N−n)/(N−1)", V == 0, V)
# フィッシャー
h = st.hypergeom(20, 9, 10)
b.num("P(7)", h.pmf(7), "0.0322")
b.num("P(8)", h.pmf(8), "0.0027")
b.num("P(9)", h.pmf(9), "0.0001")
b.num("片側 p", h.sf(6), "0.035")
b.num("両側 p", st.fisher_exact([[7, 3], [2, 8]]).pvalue, "0.070")
b.check("scipy の fisher_exact（片側）と一致", abs(st.fisher_exact([[7, 3], [2, 8]], alternative="greater").pvalue - h.sf(6)) < 1e-12, "")
# 多項
b.num("分散 1", 60 * 0.5 * 0.5, "15")
b.num("分散 2", 60 * 0.3 * 0.7, "12.6")
b.num("共分散", -60 * 0.5 * 0.3, "-9")
b.num("相関", -9 / math.sqrt(15 * 12.6), "-0.65")
M = rng.multinomial(60, [0.5, 0.3, 0.2], size=400000)
b.check("多項（シミュレーション）の共分散 ≈ −9", abs(np.cov(M[:, 0], M[:, 1])[0, 1] + 9) < 0.1, np.cov(M[:, 0], M[:, 1])[0, 1])
# 章末問題
pm = st.binom(20, 0.3).pmf
b.num("問1 (n+1)p", 21 * 0.3, "6.3")
b.num("問1 P(5)", pm(5), "0.179")
b.num("問1 P(6)", pm(6), "0.192")
b.num("問1 P(7)", pm(7), "0.164")
b.num("問2 期待値", 100 * 0.4, "40")
b.num("問2 分散", 100 * 0.4 * 0.6, "24")
b.num("問3 期待値", st.nbinom(5, 0.1).mean(), "45")
b.num("問3 分散", st.nbinom(5, 0.1).var(), "450")
b.check("問4 Cov = −1/3、V[X+Y] = 8/3", abs(-12 / 36 + 1 / 3) < 1e-12 and abs(12 * (1 / 3) * (2 / 3) - 8 / 3) < 1e-12 and abs(5 / 3 + 5 / 3 - 2 / 3 - 8 / 3) < 1e-12, "")
b.done()
