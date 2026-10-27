"""準1級 教科書 第30章「モデル選択」の検算（同じ乱数のデータで当てはめ直し、規準を計算する）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("pre1", 30)
rng = np.random.default_rng(30)
n = 30
x = np.linspace(0, 1, n)
f = lambda v: np.sin(2 * np.pi * v)
y = f(x) + rng.normal(0, 0.3, n)
xt = np.linspace(0, 1, 1000)
R = {}
for d in range(10):
    c = np.polyfit(x, y, d)
    rss = ((y - np.polyval(c, x)) ** 2).sum()
    k = d + 2
    X = np.vander(x, d + 1)
    H = X @ np.linalg.pinv(X)
    loo = np.mean(((y - H @ y) / (1 - np.diag(H))) ** 2)
    R[d] = dict(rss=rss, aic=n * math.log(rss / n) + 2 * k, bic=n * math.log(rss / n) + k * math.log(n),
                aicc=n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1), loo=loo,
                test=np.mean((f(xt) - np.polyval(c, xt)) ** 2) + 0.09)
ch = b.chart(1)
b.check("図の訓練誤差", np.allclose(ch["series"][0]["y"], [round(R[d]["rss"] / n, 3) for d in range(10)]), [round(R[d]["rss"] / n, 3) for d in range(10)])
b.check("図のテスト誤差", np.allclose(ch["series"][1]["y"], [round(R[d]["test"], 3) for d in range(10)]), [round(R[d]["test"], 3) for d in range(10)])
for d in (1, 3, 5, 7, 9):
    row = "| {} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} |".format(d, R[d]["rss"], R[d]["aic"], R[d]["bic"], R[d]["aicc"], R[d]["loo"], R[d]["test"])
    plain = b.body.replace("**", "").replace("−", "-")
    b.check(f"表の行（次数 {d}）", row in plain, row)
b.check("BIC と AICc は 5 次、AIC と LOOCV は 7 次", min(R, key=lambda d: R[d]["bic"]) == 5 and min(R, key=lambda d: R[d]["aicc"]) == 5 and min(R, key=lambda d: R[d]["aic"]) == 7 and min(R, key=lambda d: R[d]["loo"]) == 7, "")
b.check("テスト誤差の最小は 5〜6 次", min(R, key=lambda d: R[d]["test"]) in (5, 6) and abs(R[5]["test"] - R[6]["test"]) < 1e-3, "")
# LOOCV の公式 = 実際に抜いて当てはめ直す
d = 3
loo_direct = np.mean([(y[i] - np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), d), x[i])) ** 2 for i in range(n)])
b.check("LOOCV の公式 = 抜いて当てはめ直した値", abs(loo_direct - R[3]["loo"]) < 1e-9, (loo_direct, R[3]["loo"]))
# KL
kl = lambda g, fdist: integrate.quad(lambda v: g.pdf(v) * (g.logpdf(v) - fdist.logpdf(v)), -30, 30)[0]
b.check("KL N(0,1)‖N(1,1) = 1/2", abs(kl(st.norm(0, 1), st.norm(1, 1)) - 0.5) < 1e-8, "")
b.num("KL N(0,1)‖N(0,4)", kl(st.norm(0, 1), st.norm(0, 2)), "0.318")
# AIC の偏り ≈ k（シミュレーション：正規分布の平均と分散、k=2）
gap = []
for _ in range(20000):
    z = rng.normal(0, 1, 50)
    m, s2 = z.mean(), z.var()
    ll_in = st.norm(m, math.sqrt(s2)).logpdf(z).sum()
    ll_out = -50 / 2 * (math.log(2 * math.pi * s2) + (1 + m ** 2) / s2)
    gap.append(ll_in - ll_out)
b.check("最大対数尤度の楽観的な偏り ≈ k=2（n=50 なので少し大きい）", 1.9 < np.mean(gap) < 2.4, np.mean(gap))
# 章末問題
b.num("問1 AIC A", 420 + 6, "426")
b.num("問1 AIC B", 413 + 10, "423")
b.num("問1 BIC A", 420 + 3 * math.log(100), "433.8")
b.num("問1 BIC B", 413 + 5 * math.log(100), "436.0")
b.num("問3", 0.6 / 0.6, "1.0")
b.done()
