"""統計学の教科書 第51章「時系列と状態空間モデル」の検算（逐次計算・固有値・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book

b = Book("tb", 51)
rng = np.random.default_rng(51)
# 例題
q, r = 1.0, 4.0
m, P = 10.0, 4.0
out = []
for y in (12, 11, 15):
    M = P + q
    K = M / (M + r)
    m = m + K * (y - m)
    P = (1 - K) * M
    out.append((M, K, m, P))
b.num("K1", out[0][1], "0.556")
b.num("m1", out[0][2], "11.11")
b.num("P1", out[0][3], "2.22")
b.num("M2", out[1][0], "3.22")
b.num("K2", out[1][1], "0.446")
b.num("m2", out[1][2], "11.06")
b.num("P2", out[1][3], "1.78")
b.num("M3", out[2][0], "2.78")
b.num("K3", out[2][1], "0.410")
b.num("m3", out[2][2], "12.68")
# フィルタ = 事後分布（正規の共役計算）と一致することを確認
mu_post = (10 / 5 + 12 / 4) / (1 / 5 + 1 / 4)
b.check("更新 = 正規分布の事後平均", abs(mu_post - out[0][2]) < 1e-12, "")
# 定常状態
Ms = (1 + math.sqrt(17)) / 2
b.num("定常 M", Ms, "2.56")
b.num("定常 K", Ms / (Ms + r), "0.390")
P = 4.0
for _ in range(200):
    M = P + q; P = M * r / (M + r)
b.check("反復で定常値に収束", abs(P + q - Ms) < 1e-12, "")
# 図
c = b.chart(1)
ys, mus, fs = (np.array(s["y"], float) for s in c["series"])
m, P, f = ys[0], 4.0, [ys[0]]
for t in range(1, len(ys)):
    M = P + 1; K = M / (M + 4); m = m + K * (ys[t] - m); P = M * 4 / (M + 4); f.append(m)
b.check("図のフィルタは観測から再計算した値", np.allclose(f, fs, atol=0.006), "")
b.num("観測の二乗誤差", np.mean((ys - mus) ** 2), "6.0")
b.num("フィルタの二乗誤差", np.mean((np.array(f) - mus) ** 2), "1.5")
# 指数平滑法と一致（定常ゲインで）
Kss = Ms / (Ms + r)
b.check("定常ゲインは指数平滑の α", abs(Kss - 0.39) < 0.001, Kss)
# VAR
A = np.array([[0.5, 0.2], [0.1, 0.4]])
ev = sorted(np.linalg.eigvals(A).real)
b.check("固有値 0.6 と 0.3", np.allclose(ev, [0.3, 0.6]), ev)
b.check("特性多項式 λ²−0.9λ+0.18", abs(np.trace(A) - 0.9) < 1e-12 and abs(np.linalg.det(A) - 0.18) < 1e-12, "")
e = np.array([0, 1.0])
resp = []
for h in range(3):
    e = A @ e; resp.append(e[0])
b.num("IR 1期", resp[0], "0.2")
b.num("IR 2期", resp[1], "0.18")
b.num("IR 3期", resp[2], "0.126")
# 見せかけの回帰
n, R, rej = 100, 10000, 0
for _ in range(R):
    x = rng.standard_normal(n).cumsum(); y = rng.standard_normal(n).cumsum()
    X = np.c_[np.ones(n), x]
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    e2 = y - X @ bb
    se = math.sqrt(e2 @ e2 / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
    rej += abs(bb[1] / se) > 1.984
b.check("見せかけの有意の割合 ≈ 0.76", abs(rej / R - 0.76) < 0.015, rej / R)
b.check("本文の 76%", b.has("約 76%"), "")
# GARCH
w, a_, be = 0.05, 0.1, 0.85
b.num("無条件分散", w / (1 - a_ - be), "1")
b.num("半減期", math.log(0.5) / math.log(0.95), "13.5")
kur = 3 * (1 - (a_ + be) ** 2) / (1 - (a_ + be) ** 2 - 2 * a_ ** 2)
b.num("尖度", kur, "3.77")
T = 2_000_000
z = rng.standard_normal(T)
s2 = 1.0; eps = np.empty(T)
for t in range(T):
    eps[t] = math.sqrt(s2) * z[t]
    s2 = w + a_ * eps[t] ** 2 + be * s2
b.check("シミュレーションの分散 ≈ 1", abs(eps.var() - 1) < 0.02, eps.var())
kk = np.mean(eps ** 4) / eps.var() ** 2
b.check("シミュレーションの尖度 ≈ 3.77", abs(kk - kur) < 0.15, kk)
# 章末問題
M4 = 2 + math.sqrt(20)
b.num("問1 M", M4, "6.47")
b.num("問1 K", M4 / (M4 + 4), "0.618")
ev2 = sorted(np.linalg.eigvals(np.array([[0.9, 0.5], [0.2, 0.6]])).real)
b.check("問2 固有値 0.4 と 1.1", np.allclose(ev2, [0.4, 1.1]), ev2)
b.num("問3 半減期", math.log(0.5) / math.log(0.98), "34.3")
b.done()
