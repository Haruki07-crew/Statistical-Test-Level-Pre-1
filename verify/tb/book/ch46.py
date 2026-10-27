"""統計学の教科書 第46章「漸近理論の発展」の検算（公式・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("tb", 46)
rng = np.random.default_rng(46)
# 46.2 平均の比のデルタ法
g = np.array([-5 / 4, 1 / 2])
S = np.array([[1.0, 1.0], [1.0, 4.0]])
v = g @ S @ g
b.num("∇g 第1成分", g[0], "−1.25")
b.num("1.25²", 1.25 ** 2, "1.5625")
b.num("漸近分散", v, "1.3125")
b.num("標準偏差", math.sqrt(v), "1.146")
n, R = 200, 20000
L = np.linalg.cholesky(S)
r = np.empty(R)
for k in range(R):
    Z = rng.standard_normal((n, 2)) @ L.T + [2, 5]
    r[k] = Z[:, 1].mean() / Z[:, 0].mean()
sd = math.sqrt(n) * r.std()
b.check("シミュレーションの √n·sd ≈ 1.14", abs(sd - 1.14) < 0.02, sd)
# 46.3 リンデベルグ・フェラー
p = np.tile(np.arange(1, 10) / 10, 23)[:200]
b.num("和の平均", p.sum(), "99.3")
b.num("和の分散", (p * (1 - p)).sum(), "36.55")
Y = (rng.uniform(size=(100000, 200)) < p).sum(1)
z = (Y - p.sum()) / math.sqrt((p * (1 - p)).sum())
b.check("上側5%点を超える割合 約4.5%", abs(np.mean(z > 1.645) - 0.045) < 0.004, np.mean(z > 1.645))
# 46.4 DKW
b.num("DKW 幅 n=100", math.sqrt(math.log(2 / 0.05) / 200), "0.136")
# 46.5 ブートストラップの失敗
b.num("1−(1−1/50)^50", 1 - (1 - 1 / 50) ** 50, "0.636")
b.num("1−e^{-1}", 1 - math.exp(-1), "0.632")
X = rng.uniform(size=50)
hits = np.mean([rng.choice(X, 50).max() == X.max() for _ in range(20000)])
b.check("ブートストラップ最大値が元の最大値に一致する割合 ≈ 0.636", abs(hits - 0.636) < 0.015, hits)
# 46.6 局所対立仮説
for h, s in ((1, "0.26"), (2.5, "0.80")):
    b.num(f"検出力 h={h}", st.norm.sf(1.645 - h), s)
pw = st.norm.sf(1.645 - 2.5 * math.sqrt(2 / math.pi))
b.num("符号検定の検出力", pw, "0.64")
b.num("2/π", 2 / math.pi, "0.64")
# シミュレーション：符号検定 n=400、θ=2.5/20
n = 400
th = 2.5 / math.sqrt(n)
D = rng.standard_normal((20000, n)) + th
zt = D.mean(1) * math.sqrt(n)
zs = ((D > 0).sum(1) - n / 2) / math.sqrt(n / 4)
b.check("z 検定の検出力 ≈ 0.80", abs(np.mean(zt > 1.645) - 0.804) < 0.015, np.mean(zt > 1.645))
b.check("符号検定の検出力 ≈ 0.64", abs(np.mean(zs > 1.645) - pw) < 0.03, np.mean(zs > 1.645))
# 章末問題
b.num("問2 積の漸近分散", np.array([3, 1]) @ np.eye(2) @ np.array([3, 1]), "10")
nn = math.log(40) / (2 * 0.05 ** 2)
b.num("問3 n", nn, "737.8")
b.check("問3 738 以上", math.ceil(nn) == 738, math.ceil(nn))
b.done()
