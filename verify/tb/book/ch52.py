"""統計学の教科書 第52章「高次元統計」の検算（数値積分・座標降下法によるラッソのシミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("tb", 52)
rng = np.random.default_rng(52)


def lasso(X, y, lam, it=300):
    """(1/2n)‖y−Xβ‖² + λ‖β‖₁ を座標降下法で解く。"""
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()
    cs = (X ** 2).sum(0) / n
    for _ in range(it):
        mx = 0
        for j in range(p):
            old = beta[j]
            z = X[:, j] @ r / n + cs[j] * old
            new = np.sign(z) * max(abs(z) - lam, 0) / cs[j]
            if new != old:
                r -= X[:, j] * (new - old)
                beta[j] = new
                mx = max(mx, abs(new - old))
        if mx < 1e-8:
            break
    return beta


# 52.2 集中不等式
b.num("ヘフディング n=1000, t=0.05", 2 * math.exp(-2 * 1000 * 0.05 ** 2), "0.0135")
b.num("√(2 log 1000)", math.sqrt(2 * math.log(1000)), "3.72")
emax = integrate.quad(lambda x: x * 1000 * st.norm.pdf(x) * st.norm.cdf(x) ** 999, -5, 12)[0]
b.num("最大値の期待値", emax, "3.24")
b.check("シミュレーションでも ≈ 3.24", abs(rng.standard_normal((20000, 1000)).max(1).mean() - emax) < 0.01, "")
x = rng.binomial(1, 0.3, (200000, 1000)).mean(1)
b.check("ヘフディングの上界は実際の確率以上", np.mean(abs(x - 0.3) >= 0.05) <= 0.0135, np.mean(abs(x - 0.3) >= 0.05))
b.num("1000 個の 5%", 1000 * 0.05, "50")
b.num("ボンフェローニの臨界値", st.norm.isf(0.025 / 1000), "4.06")
# 52.3 λ の目安とシミュレーション
n, p, s = 100, 500, 5
lam = math.sqrt(2 * math.log(p) / n)
b.num("λ", lam, "0.353")
b.check("約 0.35", b.has("約 0.35"), "")
E, O, RF, sel, fp = [], [], [], [], []
for _ in range(100):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:s] = 1
    y = X @ beta + rng.standard_normal(n)
    bh = lasso(X, y, lam)
    E.append(((bh - beta) ** 2).sum())
    S = np.nonzero(bh)[0]
    sel.append(set(range(s)) <= set(S)); fp.append(int((S >= s).sum()))
    bo = np.linalg.lstsq(X[:, :s], y, rcond=None)[0]
    O.append(((bo - 1) ** 2).sum())
    bf = np.zeros(p); bf[S] = np.linalg.lstsq(X[:, S], y, rcond=None)[0]
    RF.append(((bf - beta) ** 2).sum())
b.check("ラッソの誤差 ≈ 0.83", abs(np.mean(E) - 0.83) < 0.08, np.mean(E))
b.check("オラクル ≈ 0.054", abs(np.mean(O) - 0.054) < 0.01, np.mean(O))
b.num("オラクルの理論値 s/(n−s−1)", s / (n - s - 1), "0.053")
b.check("すべて効く変数を選んだ", np.mean(sel) >= 0.97, np.mean(sel))
b.check("関係ない変数 平均 2.35 個程度", abs(np.mean(fp) - 2.35) < 0.8, np.mean(fp))
b.check("再当てはめ ≈ 0.18", abs(np.mean(RF) - 0.18) < 0.05, np.mean(RF))
b.num("sλ²", s * lam ** 2, "0.62")
b.check("約 15 倍", abs(0.83 / 0.054 - 15) < 0.5, 0.83 / 0.054)
# KKT（最適性）の確認
X = rng.standard_normal((n, p)); beta = np.zeros(p); beta[:s] = 1
y = X @ beta + rng.standard_normal(n)
bh = lasso(X, y, lam)
g = X.T @ (y - X @ bh) / n
nz = bh != 0
b.check("KKT: 非ゼロでは勾配 = λ sign", np.allclose(g[nz], lam * np.sign(bh[nz]), atol=1e-6), "")
b.check("KKT: ゼロでは |勾配| ≤ λ", np.all(abs(g[~nz]) <= lam + 1e-6), "")
# 章末問題
b.num("問1 n", math.log(200) / 0.02, "264.9")
b.num("問2", math.sqrt(2 * math.log(1e6)), "5.26")
b.num("問2 比", math.sqrt(2 * math.log(1e6)) / math.sqrt(2 * math.log(1000)), "1.4")
b.num("問3", 2 * math.sqrt(2 * math.log(10000) / 400), "0.43")
b.done()
