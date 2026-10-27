"""準1級 教科書 第19章「回帰分析その他」の検算（行列計算・シミュレーション）。"""
import numpy as np
from bookcheck import Book

b = Book("pre1", 19)
b.num("単純平均の分散", (1 + 4) / 4, "1.25")
b.num("WLS の分散", 1 / (1 + 1 / 4), "0.8")
# 欠落変数バイアス（第16章の住宅データ）
area = np.array([6.2, 7.5, 5.8, 8.9, 7.1, 9.4, 6.6, 8.0, 7.8, 5.5, 8.6, 6.9])
age = np.array([12, 5, 20, 8, 15, 3, 25, 10, 18, 30, 6, 14.])
price = np.array([31.5, 42.0, 25.8, 48.6, 33.9, 55.2, 26.1, 43.4, 35.0, 20.3, 49.1, 32.8])
X = np.column_stack([np.ones(12), area, age])
bh = np.linalg.lstsq(X, price, rcond=None)[0]
d = np.polyfit(area, age, 1)[0]
b.num("δ", d, "-5.306")
b.num("β1+β2δ", bh[1] + bh[2] * d, "8.33")
b.check("標本上の恒等式（単回帰の傾きと一致）", abs(bh[1] + bh[2] * d - np.polyfit(area, price, 1)[0]) < 1e-9, "")
# IV
rng = np.random.default_rng(19)
n = 2000
z = rng.normal(size=n)
u = rng.normal(size=n)
x = 0.8 * z + 0.6 * u + rng.normal(0, 0.5, n)
y = 1 + 1.0 * x + 1.5 * u + rng.normal(size=n)
b.num("OLS の傾き", np.polyfit(x, y, 1)[0], "1.76")
biv = np.cov(z, y)[0, 1] / np.cov(z, x)[0, 1]
b.num("IV の傾き", biv, "0.98")
xh = np.polyval(np.polyfit(z, x, 1), z)
b.check("2SLS = IV", abs(np.polyfit(xh, y, 1)[0] - biv) < 1e-9, "")
# リッジ（相関の強い 2 変数）
rng = np.random.default_rng(19)
m = 30
x1 = rng.normal(size=m)
x2 = x1 + rng.normal(0, 0.1, m)
yy = 1 + 2 * x1 + 1 * x2 + rng.normal(0, 1, m)
b.num("相関", np.corrcoef(x1, x2)[0, 1], "0.994")
Xc = np.column_stack([x1, x2]) - np.column_stack([x1, x2]).mean(0)
yc = yy - yy.mean()
for lam, s in [(0, ("-0.51", "3.90")), (1, ("1.41", "1.87")), (5, ("1.47", "1.54"))]:
    r = np.linalg.solve(Xc.T @ Xc + lam * np.eye(2), Xc.T @ yc)
    b.num(f"リッジ λ={lam} 係数1", r[0], s[0])
    b.num(f"リッジ λ={lam} 係数2", r[1], s[1])
    b.check(f"合計は約 3（λ={lam}）", abs(r.sum() - 3) < 0.45, r.sum())
for lam, s in [(0, "0.5"), (0.25, "0.36"), (0.5, "0.333"), (1, "0.375")]:
    b.num(f"MSE λ={lam}", 0.5 / (1 + lam) ** 2 + (lam / (1 + lam)) ** 2, s)
lams = np.linspace(0, 3, 3001)
b.check("最小は λ=σ²/β²=0.5", abs(lams[np.argmin(0.5 / (1 + lams) ** 2 + (lams / (1 + lams)) ** 2)] - 0.5) < 1e-3, "")
# 軟しきい値の確認（座標ごとの最小化）
for bb in [4, -2, 0.5]:
    grid = np.linspace(-6, 6, 120001)
    best = grid[np.argmin((bb - grid) ** 2 + 2 * np.abs(grid))]
    b.check(f"Lasso（λ=2）の解 = sign(b)max(|b|−1,0)（b={bb}）", abs(best - np.sign(bb) * max(abs(bb) - 1, 0)) < 1e-3, best)
# 章末問題
w = np.array([1, 0.5, 0.25])
v = np.array([10, 13, 16.])
b.num("問1 平均", (w * v).sum() / w.sum(), "11.71")
b.num("問1 分散", 1 / w.sum(), "0.571")
b.num("問2", 2 + 3 * 0.5, "3.5")
b.done()
