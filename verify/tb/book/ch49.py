"""統計学の教科書 第49章「ノンパラメトリック推定と頑健統計」の検算（数値積分・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("tb", 49)
rng = np.random.default_rng(49)
# 49.1 バンド幅
b.num("シルバーマン n=100", 1.06 * 100 ** -0.2, "0.42")
# MISE 最小の h を数値で（正規分布、正規カーネル）：h* = (4/(3n))^{1/5} σ = 1.06 σ n^{-1/5}
b.check("h* = (4/3)^{1/5} ≈ 1.06", abs((4 / 3) ** 0.2 - 1.06) < 0.002, (4 / 3) ** 0.2)
# 偏りの近似の確認（x=0, h=0.2）
h = 0.2
Ef = st.norm.pdf(0, 0, math.sqrt(1 + h * h))
b.check("偏り ≈ h²/2·f''(0)", abs((Ef - st.norm.pdf(0)) - h * h / 2 * (-st.norm.pdf(0))) < 4e-4, Ef - st.norm.pdf(0))
# 次元の呪い
b.num("5 次元", 100 ** 1.8, "4000", tol=50)
b.check("10 次元 約 40 万", abs(100 ** 2.8 / 4e5 - 1) < 0.01, 100 ** 2.8)
# 49.3 中央値
b.num("中央値の漸近分散", 1 / (4 * st.norm.pdf(0) ** 2), "1.571")
b.num("効率 2/π", 2 / math.pi, "0.637")
med = np.median(rng.standard_normal((20000, 401)), axis=1)
b.check("中央値の分散（シミュレーション）", abs(401 * med.var() - math.pi / 2) < 0.05, 401 * med.var())
# 49.4 フーバー
k = 1.345
def huber_var(f, F):
    Ep = F(k) - F(-k)
    Ep2 = integrate.quad(lambda z: z * z * f(z), -k, k)[0] + k * k * (F(-k) + 1 - F(k))
    return Ep2 / Ep ** 2
v = huber_var(st.norm.pdf, st.norm.cdf)
b.num("フーバーの漸近分散（正規）", v, "1.053")
b.check("効率 95%", abs(1 / v - 0.95) < 1e-4 and b.has("効率 95%"), 1 / v)
f = lambda x: 0.9 * st.norm.pdf(x) + 0.1 * st.norm.pdf(x, 0, 3)
F = lambda x: 0.9 * st.norm.cdf(x) + 0.1 * st.norm.cdf(x, 0, 3)
b.num("混合: 平均", 0.9 + 0.1 * 9, "1.8")
b.num("混合: 中央値", 1 / (4 * f(0) ** 2), "1.80")
b.num("混合: フーバー", huber_var(f, F), "1.30")
# シミュレーション（尺度 1 固定のフーバー）
def huber_fixed(X, it=60):
    m = np.median(X, axis=1)
    for _ in range(it):
        r = X - m[:, None]
        w = np.minimum(1, k / np.maximum(abs(r), 1e-12))
        m = (w * X).sum(1) / w.sum(1)
    return m
N = 400
Z = np.where(rng.uniform(size=(10000, N)) < 0.9, 1, 3) * rng.standard_normal((10000, N))
hv = N * huber_fixed(Z).var()
b.check("混合でのフーバーの分散（シミュレーション）≈ 1.30", abs(hv - 1.30) < 0.06, hv)
# 例題
x = np.array([4.8, 5.1, 5.0, 4.6, 5.2, 5.4, 4.7, 5.0, 5.3, 15.0])
b.num("平均", x.mean(), "6.01")
b.num("中央値", np.median(x), "5.05")
mad = np.median(abs(x - np.median(x)))
b.num("MAD", mad, "0.25")
s = mad / 0.6745
b.num("尺度", s, "0.371")
m = np.median(x)
for _ in range(200):
    w = np.minimum(1, k / np.maximum(abs((x - m) / s), 1e-12))
    m = (w * x).sum() / w.sum()
b.num("フーバー推定量", m, "5.07")
b.num("15.0 の重み", w[-1], "0.05")
b.check("他の重みは 1", np.all(w[:9] == 1), "")
b.num("9 個の平均", x[:9].mean(), "5.01")
# 章末問題
b.num("問1 1000^{-1/5}", 1000 ** -0.2, "0.251")
b.num("問1 h", 1.06 * 2 * 1000 ** -0.2, "0.53")
b.num("問2", math.sqrt(math.pi / 2), "1.253")
b.done()
