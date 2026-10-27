"""準1級 教科書 第6章「連続型分布と標本分布」の検算（scipy・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import special as sc, stats as st

b = Book("pre1", 6)
rng = np.random.default_rng(6)
b.check("Γ(1/2)² = π", abs(sc.gamma(0.5) ** 2 - math.pi) < 1e-12, "")
b.check("正規の E[Z⁴] = 3", abs(st.norm.moment(4) - 3) < 1e-9, "")
p3 = 1 - math.exp(-2) * (1 + 2 + 2)
b.num("3 件目が 1 時間以内", p3, "0.323")
b.check("ガンマ分布 Ga(3, 1/2) と一致", abs(st.gamma(3, scale=0.5).cdf(1) - p3) < 1e-12, "")
b.check("χ²(k) = Ga(k/2,2)", np.allclose(st.chi2(5).pdf([0.5, 2, 7]), st.gamma(2.5, scale=2).pdf([0.5, 2, 7])), "")
# ベータ
B = st.beta(3, 3)
b.num("Beta(3,3) の分散", B.var(), "0.0357")
b.check("分散 = 1/28 = 9/(36×7)", abs(B.var() - 1 / 28) < 1e-15 and abs(9 / (36 * 7) - 1 / 28) < 1e-15, "")
b.num("一様の分散", 1 / 12, "0.083")
b.num("P(中央値 ≤ 0.3)", B.cdf(0.3), "0.163")
med = np.sort(rng.uniform(size=(200000, 5)), 1)[:, 2]
b.check("シミュレーション：5 個の中央値", abs(med.var() - 1 / 28) < 0.0005 and abs(np.mean(med <= 0.3) - B.cdf(0.3)) < 0.003, "")
g1, g2 = rng.gamma(2, 1, 10 ** 6), rng.gamma(5, 1, 10 ** 6)
b.check("ガンマの割合はベータ（Beta(2,5)）", abs(np.mean(g1 / (g1 + g2) <= 0.3) - st.beta(2, 5).cdf(0.3)) < 0.002, "")
c = rng.normal(size=10 ** 6) / rng.normal(size=10 ** 6)
b.check("正規の比はコーシー", abs(np.mean(c <= 1) - st.cauchy.cdf(1)) < 0.002, "")
b.check("対数正規の平均", abs(st.lognorm(0.5, scale=math.exp(1)).mean() - math.exp(1 + 0.125)) < 1e-12, "")
# 混合
b.num("混合の平均", 0.7 * 0 + 0.3 * 3, "0.9")
b.num("平均の分散", 0.7 * 0.3 * 9, "1.89")
b.num("混合の分散", 1 + 0.7 * 0.3 * 9, "2.89")
mix = np.where(rng.uniform(size=10 ** 6) < 0.7, rng.normal(0, 1, 10 ** 6), rng.normal(3, 1, 10 ** 6))
b.check("シミュレーション", abs(mix.var() - 2.89) < 0.02, mix.var())
# 多変量正規の条件付き
b.num("相関", 2.4 / 6, "0.4")
b.num("条件付き期待値", 10 + 2.4 / 9 * 3, "10.8")
b.num("条件付き分散", 4 - 2.4 ** 2 / 9, "3.36")
b.num("条件付き sd", math.sqrt(3.36), "1.83")
b.num("1−ρ²", 1 - 0.16, "0.84")
X = rng.multivariate_normal([10, 20], [[4, 2.4], [2.4, 9]], size=3 * 10 ** 6)
sel = X[np.abs(X[:, 1] - 23) < 0.05, 0]
b.check("シミュレーション（条件付き）", abs(sel.mean() - 10.8) < 0.05 and abs(sel.var() - 3.36) < 0.1, (sel.mean(), sel.var()))
b.num("χ²_0.05(2)", st.chi2.ppf(0.95, 2), "5.99")
b.num("e^{-5.99/2}", math.exp(-5.99 / 2), "0.05")
S = np.array([[4, 2.4], [2.4, 9]])
d2 = np.einsum("ij,jk,ik->i", X - [10, 20], np.linalg.inv(S), X - [10, 20])
b.check("マハラノビス距離² ～ χ²(2)", abs(np.mean(d2 <= 5.991) - 0.95) < 0.002, np.mean(d2 <= 5.991))
# 標本分布
Z = rng.normal(5, 2, (200000, 8))
xb, s2 = Z.mean(1), Z.var(1, ddof=1)
b.check("x̄ と S² は無相関（正規）", abs(np.corrcoef(xb, s2)[0, 1]) < 0.01, np.corrcoef(xb, s2)[0, 1])
E = rng.exponential(1, (200000, 8))
b.check("指数分布では x̄ と S² は相関する", np.corrcoef(E.mean(1), E.var(1, ddof=1))[0, 1] > 0.3, np.corrcoef(E.mean(1), E.var(1, ddof=1))[0, 1])
T = (xb - 5) / np.sqrt(s2 / 8)
b.check("T ～ t(7)", abs(np.mean(np.abs(T) > st.t.ppf(0.975, 7)) - 0.05) < 0.003, "")
b.check("t² = F(1,ν)", abs(st.t.ppf(0.975, 9) ** 2 - st.f.ppf(0.95, 1, 9)) < 1e-9, "")
b.check("t(1) = コーシー", abs(st.t(1).cdf(2) - st.cauchy.cdf(2)) < 1e-12, "")
# 非心
b.check("非心カイ二乗の期待値 k+λ", abs(st.ncx2(4, 3).mean() - 7) < 1e-9, "")
d = math.sqrt(10)
b.num("非心度", d, "3.16")
tc = st.t.ppf(0.975, 9)
pw = st.nct(9, d).sf(tc) + st.nct(9, d).cdf(-tc)
b.num("検出力", pw, "0.803")
b.num("検出力（2 桁）", pw, "0.80")
sim = rng.normal(1, 1, (200000, 10))
Ts = sim.mean(1) / (sim.std(1, ddof=1) / math.sqrt(10))
b.check("検出力（シミュレーション）", abs(np.mean(np.abs(Ts) > tc) - pw) < 0.004, np.mean(np.abs(Ts) > tc))
b.num("正規で計算", st.norm.sf(1.96 - d), "0.885")
d15 = math.sqrt(15) * 0.8
b.num("n=15 の検出力", st.nct(14, d15).sf(st.t.ppf(0.975, 14)) + st.nct(14, d15).cdf(-st.t.ppf(0.975, 14)), "0.82")
# 章末問題
b.num("問1 Γ(7/2)", sc.gamma(3.5), "3.323")
b.check("問1 15/8 √π", abs(15 / 8 * math.sqrt(math.pi) - sc.gamma(3.5)) < 1e-12, "")
b.num("問2", 1 - math.exp(-1.5) * 2.5, "0.442")
b.num("問3", st.beta(4, 1).mean(), "0.8")
b.num("問4", st.chi2(15).cdf(30), "0.988")
b.done()
