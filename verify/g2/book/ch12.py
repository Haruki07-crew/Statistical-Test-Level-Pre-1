"""2級 教科書 第12章「仮説検定の考え方と1標本の検定」の数値の検算（第1種の過誤・検出力はシミュレーションでも確かめる）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("g2", 12)
rng = np.random.default_rng(12)
b.num("硬貨 z", (59.5 - 50) / 5, "1.9")
b.num("片側 p", st.norm.sf(1.9), "0.029")
b.num("両側 p", 2 * st.norm.sf(1.9), "0.057")
b.check("正確な二項でも片側 < 0.05 < 両側", st.binom(100, 0.5).sf(59) < 0.05 < 2 * st.binom(100, 0.5).sf(59), "")
b.num("z_0.05", st.norm.ppf(0.95), "1.645")
# 検出力
b.num("境目", 100 + 1.645 * 2, "103.29")
b.num("z", (103.29 - 105) / 2, "-0.855")
pw = st.norm.sf((103.29 - 105) / 2)
b.num("検出力", pw, "0.804")
b.num("検出力（2 桁）", pw, "0.80")
b.num("β", 1 - pw, "0.20")
xb = rng.normal(105, 10, (200000, 25)).mean(1)
b.check("検出力（シミュレーション）", abs(np.mean(xb >= 103.29) - 0.804) < 0.004, np.mean(xb >= 103.29))
x0 = rng.normal(100, 10, (200000, 25)).mean(1)
b.check("第1種の過誤 ≈ 0.05（シミュレーション）", abs(np.mean(x0 >= 103.29) - 0.05) < 0.003, np.mean(x0 >= 103.29))
b.num("n=100 の境目", 100 + 1.645, "101.645")
b.check("n=100 の検出力は 0.999 以上", st.norm.sf(101.645 - 105) > 0.999, st.norm.sf(101.645 - 105))
b.num("検出力 0.9 の n", ((1.645 + 1.2816) * 10 / 5) ** 2, "34.3")
b.check("n=35 で検出力 ≥ 0.9、n=34 で未満", st.norm.sf((100 + 1.645 * 10 / np.sqrt(35) - 105) / (10 / np.sqrt(35))) >= 0.9 > st.norm.sf((100 + 1.645 * 10 / np.sqrt(34) - 105) / (10 / np.sqrt(34))), "")
# z 検定
b.num("学校の z", (52.5 - 50) / (6 / 6), "2.5")
b.num("両側 p", 2 * st.norm.sf(2.5), "0.012")
# t 検定
x = np.array([501.2, 499.8, 500.6, 502.1, 498.9, 500.3, 501.5, 499.4, 500.9, 501.3])
T = (x.mean() - 500) / (x.std(ddof=1) / np.sqrt(10))
b.num("T", T, "1.89")
b.num("両側 p", 2 * st.t.sf(T, 9), "0.091")
b.num("片側の境目", st.t.ppf(0.95, 9), "1.833")
b.num("片側 p", st.t.sf(T, 9), "0.046")
r = st.ttest_1samp(x, 500)
b.check("scipy の t 検定と一致", abs(r.statistic - T) < 1e-9 and abs(r.pvalue - 2 * st.t.sf(T, 9)) < 1e-9, r)
# 分散の検定
chi = ((x - x.mean()) ** 2).sum() / 0.64
b.num("χ²", chi, "14.16")
b.num("上側 5% 点", st.chi2.ppf(0.95, 9), "16.92")
b.num("p", st.chi2.sf(chi, 9), "0.117")
# 比率の検定
b.num("√19", np.sqrt(19), "4.36")
z = (28 - 20) / np.sqrt(19)
b.num("Z", z, "1.84")
b.num("p", st.norm.sf(z), "0.033")
b.num("正確な二項 p", st.binom(400, 0.05).sf(27), "0.048")
# 章末問題
b.num("問1 Z", (98.9 - 100) / 0.5, "-2.2")
b.num("問1 p", st.norm.cdf(-2.2), "0.014")
b.num("問2 SE", 5.4 / 3, "1.8")
b.num("問2 T", 4.0 / 1.8, "2.22")
b.num("問2 t(8)", st.t.ppf(0.975, 8), "2.306")
b.num("問3 p̂", 264 / 600, "0.44")
b.num("問3 SE", np.sqrt(0.4 * 0.6 / 600), "0.02")
b.num("問3 Z", (0.44 - 0.40) / 0.02, "2.0")
b.num("問3 p", 2 * st.norm.sf(2.0), "0.046")
b.done()
