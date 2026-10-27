"""準1級 教科書 第7章「極限定理、漸近理論」の検算（正確な分布・シミュレーション）。"""
import math

import numpy as np
import sympy as sp
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 7)
rng = np.random.default_rng(7)
b.num("マルコフの上限", 2 / 10, "0.2")
b.num("指数分布の実際", math.exp(-5), "0.0067")
t = sp.symbols("t")
n = sp.symbols("n", positive=True)
b.check("(1+t²/(2n))^n → e^{t²/2}", sp.limit((1 + t ** 2 / (2 * n)) ** n, n, sp.oo) == sp.exp(t ** 2 / 2), "")
# ベリー・エッセン
p = 0.1
rho = p * (1 - p) * (p ** 2 + (1 - p) ** 2)
s3 = (p * (1 - p)) ** 1.5
b.num("ρ/σ³", rho / s3, "2.73")
b.num("上限 n=100", 0.4748 * rho / s3 / 10, "0.130")
b.num("上限 n=1000", 0.4748 * rho / s3 / math.sqrt(1000), "0.041")


def maxerr(n):
    k = np.arange(n + 1)
    F = st.binom(n, p).cdf(k)
    z = (k - n * p) / math.sqrt(n * p * (1 - p))
    Fl = np.concatenate([[0], F[:-1]])
    return np.maximum(np.abs(F - st.norm.cdf(z)), np.abs(Fl - st.norm.cdf(z))).max()


b.num("実際の誤差 n=100", maxerr(100), "0.083")
b.num("実際の誤差 n=1000", maxerr(1000), "0.027")
# t 統計量（指数分布）
y = rng.exponential(size=(200000, 30))
tt = (y.mean(1) - 1) / (y.std(1, ddof=1) / math.sqrt(30))
b.num("被覆 n=30 %", np.mean(np.abs(tt) <= 1.96) * 100, "91.8", tol=0.3)
# t < −1.96 ⇔ 上限 x̄+1.96s/√n が 1 より小さい ⇔ 区間が母平均より下にずれて外れる
b.num("区間が母平均より下にずれて外れた %", np.mean(tt < -1.96) * 100, "7.3", tol=0.3)
b.num("区間が母平均より上にずれて外れた %", np.mean(tt > 1.96) * 100, "1.0", tol=0.3)
y2 = rng.exponential(size=(100000, 200))
t2 = (y2.mean(1) - 1) / (y2.std(1, ddof=1) / math.sqrt(200))
b.num("被覆 n=200 %", np.mean(np.abs(t2) <= 1.96) * 100, "94.3", tol=0.3)
# デルタ法
b.num("1/√50", 1 / math.sqrt(50), "0.141")
y3 = rng.exponential(2, size=(200000, 50))
b.num("シミュレーション", np.log(y3.mean(1)).std(), "0.142", tol=0.002)
pp = sp.symbols("p", positive=True)
b.check("逆正弦変換：g'(p)² p(1−p) = 1/4", sp.simplify(sp.diff(sp.asin(sp.sqrt(pp)), pp) ** 2 * pp * (1 - pp)) == sp.Rational(1, 4), "")
# 極値
b.num("log 1000 + 1", math.log(1000) + 1, "7.91")
b.num("n=1000 の確率", (1 - math.exp(-1) / 1000) ** 1000, "0.692")
b.num("ガンベルの極限", math.exp(-math.exp(-1)), "0.692")
mx = rng.exponential(size=(20000, 1000)).max(1) - math.log(1000)
b.check("シミュレーション", abs(np.mean(mx <= 1) - 0.692) < 0.01, np.mean(mx <= 1))
# 章末問題
b.num("問1 マルコフ", 50 / 200, "0.25")
b.num("問1 k", 150 / 20, "7.5")
b.num("問1 チェビシェフ", 1 / 7.5 ** 2, "0.0178")
b.num("問2 sd", 5 * math.sqrt(400), "100")
b.num("問2", st.norm.sf(2), "0.0228")
b.check("問3 オッズの漸近分散 p/(n(1−p)³)", sp.simplify(sp.diff(pp / (1 - pp), pp) ** 2 * pp * (1 - pp) - pp / (1 - pp) ** 3) == 0, "")
b.done()
