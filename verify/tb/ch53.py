"""教科書 第53章（統計的学習理論）の検算。"""
from math import comb, e, log, log2, sqrt

import numpy as np

from tbcheck import allclose, body, chart, check, close, done

T = "empirical-risk-minimization"
p = chart(T)
rng = np.random.default_rng(53)
ts = np.arange(0, 1, 0.01)
R = 0.1 + 0.8 * np.abs(ts - 0.3)
q95, exc = [], []
for n in p["x"]:
    sup, ex = [], []
    for _ in range(2000):
        x = rng.random(n)
        y = (x > 0.3).astype(int)
        y = np.where(rng.random(n) < 0.1, 1 - y, y)
        Rn = ((x[None, :] > ts[:, None]).astype(int) != y[None, :]).mean(1)
        sup.append(np.abs(Rn - R).max())
        ex.append(R[np.argmin(Rn)] - 0.1)
    q95.append(np.quantile(sup, 0.95))
    exc.append(np.mean(ex))
check(T, "図：上界・一様なずれの 95% 点・超過リスク（同じ乱数で計算し直す）",
      allclose(p["series"][0]["y"], [sqrt((log(100) + log(40)) / (2 * n)) for n in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], q95, 6e-5) and allclose(p["series"][2]["y"], exc, 6e-5), (q95, exc))
check(T, "例 n=500：上界 0.091・2 倍 0.18・95% 点 0.060・超過リスク 0.0005", close(sqrt((log(100) + log(40)) / 1000), 0.091, 5e-4)
      and close(2 * sqrt((log(100) + log(40)) / 1000), 0.18, 0.005) and close(q95[3], 0.060, 5e-4) and close(exc[3], 0.0005, 5e-5))
xx = rng.random(10 ** 6)
yy = (xx > 0.3).astype(int)
yy = np.where(rng.random(10 ** 6) < 0.1, 1 - yy, yy)
check(T, "R(0.5) = 0.26（シミュレーション）", close(((xx > 0.5).astype(int) != yy).mean(), 0.26, 0.002))
check(T, "上界は実際の 95% 点の約 1.5 倍", all(1.3 < a / b < 1.6 for a, b in zip(p["series"][0]["y"], q95)))

T = "vc-dimension"
p = chart(T)
check(T, "図：2ⁿ・ザウアーの上界・直線の付け方 2Σ C(n−1,i)", allclose(p["series"][0]["y"], p["x"], 1e-12)
      and allclose(p["series"][1]["y"], [log2(sum(comb(n, i) for i in range(4))) for n in p["x"]], 6e-4)
      and allclose(p["series"][2]["y"], [log2(2 * sum(comb(n - 1, i) for i in range(3))) for n in p["x"]], 6e-4))
check(T, "例2 n=20：382 通り・2²⁰=1,048,576・ザウアー 1351", 2 * sum(comb(19, i) for i in range(3)) == 382 and 2 ** 20 == 1048576 and sum(comb(20, i) for i in range(4)) == 1351)


def count_lines(pts, m=400000, seed=0):
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, m)
    w = np.c_[np.cos(th), np.sin(th)]
    proj = pts @ w.T
    lo, hi = proj.min(0), proj.max(0)
    b = r.uniform(lo - 0.1, hi + 0.1)
    lab = proj > b
    return len({tuple(c) for c in lab.T})


pts6 = np.random.default_rng(7).random((6, 2))
check(T, "一般の位置の 6 点：直線で実現できる付け方は 2Σ_{i≤2}C(5,i)=32（乱数の直線で数える）", count_lines(pts6) == 32, count_lines(pts6))
pts3 = np.array([[0, 0], [1, 0], [0, 1.0]])
xor = np.array([[0, 0], [1, 1], [1, 0], [0, 1.0]])
def labs(P):
    t = np.random.default_rng(1).uniform(0, 2 * np.pi, 200000)
    b = np.random.default_rng(2).uniform(-2, 2, 200000)
    return {tuple(c) for c in (P @ np.c_[np.cos(t), np.sin(t)].T > b).T}


L4 = labs(xor)
check(T, "例1 直線：3 点は細分できる（8 通り）、XOR の 4 点では (1,1,0,0) が作れない", len(labs(pts3)) == 8 and (True, True, False, False) not in L4 and len(L4) == 14, len(L4))
xs = np.array([1.0, 2.0, 3.0])
ab = [(a, b) for a in np.arange(0, 4, 0.5) for b in np.arange(0, 4, 0.5)]
iv = {tuple((xs >= a) & (xs <= b)) for a, b in ab}
check(T, "例1 区間：3 点の (1,0,1) は作れない（他の 7 通りは作れる）", (True, False, True) not in iv and len(iv) == 7)
# sin の分類器は任意の個数の点 x_i = 10^{-i} を細分できる（Vapnik の構成：ω = π(1 + Σ (1−y_i)10^i)）
xi = np.array([10.0 ** -i for i in range(1, 7)])
ok = True
for code in range(2 ** 6):
    yv = np.array([(code >> i) & 1 for i in range(6)])
    om = np.pi * (1 + sum((1 - yv[i]) * 10 ** (i + 1) for i in range(6)))
    ok &= np.array_equal((np.sin(om * xi) > 0).astype(int), yv)
check(T, "例1 sin(ωx)：6 点の 64 通りをすべて実現（VC 次元は無限）", ok)
eps = lambda n: sqrt(32 * (3 * log(e * n / 3) + log(8 / 0.05)) / n)
check(T, "例3 ε ≈ 0.32（n=10⁴）・0.11（n=10⁵）", close(eps(10 ** 4), 0.32, 0.005) and close(eps(10 ** 5), 0.11, 0.005))
check(T, "ザウアー Σ_{i≤d} C(n,i) ≤ (en/d)^d", all(sum(comb(n, i) for i in range(4)) <= (e * n / 3) ** 3 for n in range(3, 200)))
check(T, "本文の値", all(v in body(T) for v in ("382", "1351", "0.32", "0.11")))
done("第53章（統計的学習理論）")
