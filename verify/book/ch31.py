"""準1級 教科書 第31章「ベイズ法」の検算（共役分布の公式・数値積分・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("pre1", 31)
z = st.norm.ppf(0.975)
ph = 0.7
c = (ph + z * z / 20) / (1 + z * z / 10)
h = z / (1 + z * z / 10) * math.sqrt(ph * (1 - ph) / 10 + z * z / 400)
b.num("ウィルソン下限", c - h, "0.40")
b.num("ウィルソン上限", c + h, "0.89")
post = st.beta(9, 5)
b.num("事後平均", post.mean(), "0.643")
b.num("MAP", 8 / 12, "0.667")
lo, hi = post.ppf([0.025, 0.975])
b.num("信用区間 下", lo, "0.386")
b.num("信用区間 上", hi, "0.861")
b.num("P(p>0.5)", post.sf(0.5), "0.867")
# 事後 ∝ 尤度 × 事前（数値積分で確認）
grid = np.linspace(0, 1, 200001)
un = grid ** 7 * (1 - grid) ** 3 * st.beta(2, 2).pdf(grid)
un /= np.trapezoid(un, grid)
b.check("尤度×事前の正規化 = Beta(9,5)", np.max(np.abs(un - post.pdf(grid))) < 1e-6, "")
b.num("予測：4 人以上", st.betabinom(5, 9, 5).sf(3), "0.439")
b.check("予測分布は二項分布より散らばる", st.betabinom(5, 9, 5).var() > st.binom(5, 9 / 14).var(), "")
# 正規・正規
prec = 1 / 100 + 4 / 64
m = (50 / 100 + 4 * 60 / 64) / prec
b.num("事後平均", m, "58.6")
b.num("事後 sd", math.sqrt(1 / prec), "3.71")
b.num("データの重み", (4 / 64) / prec, "0.86")
# ジェフリーズ
pp = np.linspace(0.01, 0.99, 50)
b.check("二項のジェフリーズ ∝ Beta(1/2,1/2)", np.allclose(np.sqrt(1 / (pp * (1 - pp))) / st.beta(0.5, 0.5).pdf(pp), np.pi), "")
# ベイズファクター
m0 = math.comb(10, 7) * 0.5 ** 10
m1 = integrate.quad(lambda p: math.comb(10, 7) * p ** 7 * (1 - p) ** 3, 0, 1)[0]
b.num("m0", m0, "0.117")
b.num("m1", m1, "0.091")
b.check("m1 = 1/11", abs(m1 - 1 / 11) < 1e-12, "")
b.num("B10", m1 / m0, "0.78")
# 経験ベイズ
y = np.array([12, 7, 9, 15, 5.])
b.num("平均", y.mean(), "9.6")
b.num("不偏分散", y.var(ddof=1), "15.8")
B = 4 / y.var(ddof=1)
b.num("B", B, "0.25")
est = y.mean() + (1 - B) * (y - y.mean())
b.num("15 の推定値", est[3], "13.6")
b.num("5 の推定値", est[4], "6.2")
# 縮小で予測誤差が下がる（シミュレーション）
rng = np.random.default_rng(31)
th = rng.normal(10, math.sqrt(11.8), (100000, 5))
yy = th + rng.normal(0, 2, th.shape)
yb = yy.mean(1, keepdims=True)
Bh = np.clip(4 / yy.var(1, ddof=1, keepdims=True), 0, 1)
eb = yb + (1 - Bh) * (yy - yb)
b.check("経験ベイズのほうが平均二乗誤差が小さい", ((eb - th) ** 2).mean() < ((yy - th) ** 2).mean(), (((eb - th) ** 2).mean(), ((yy - th) ** 2).mean()))
# 章末問題
b.num("問1", 16 / 22, "0.727")
b.num("問2", 22 / 6, "3.67")
b.num("問3 n=1", 2 / 2, "1")
b.num("問3 n=9", 18 / 10, "1.8")
b.done()
