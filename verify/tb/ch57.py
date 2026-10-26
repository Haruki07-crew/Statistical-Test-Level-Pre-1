"""教科書 第57章（生存時間解析の理論）の検算。"""
import numpy as np

from tbcheck import allclose, body, chart, check, close, done

T = "counting-process-martingale"
t = np.array([2, 3, 3, 5, 6, 8, 9.0])
d = np.array([1, 1, 0, 1, 1, 0, 1])
na, var, rows = 0.0, 0.0, []
for ti, di in zip(t, d):
    if di:
        Y = (t >= ti).sum()
        na += 1 / Y
        var += 1 / Y ** 2
        rows.append((ti, Y))
check(T, "例1 イベントの時点とリスク集合 (2,7)(3,6)(5,4)(6,3)(9,1)", rows == [(2, 7), (3, 6), (5, 4), (6, 3), (9, 1)])
check(T, "例1 Λ̂(9)≈1.893・分散≈1.222", close(na, 1.893, 5e-4) and close(var, 1.222, 5e-4), (na, var))
na8 = 1 / 7 + 1 / 6 + 1 / 4 + 1 / 3
km8 = (6 / 7) * (5 / 6) * (3 / 4) * (2 / 3)
check(T, "例1 Λ̂(8)≈0.893・e^{−Λ̂}≈0.409・KM≈0.357・e^{−Λ̂} ≥ KM", close(na8, 0.893, 5e-4) and close(np.exp(-na8), 0.409, 5e-4) and close(km8, 0.357, 5e-4) and np.exp(-na8) >= km8)
rng = np.random.default_rng(1)
Tt = rng.exponential(2, 10 ** 6)
C = rng.uniform(0, 4, 10 ** 6)
X = np.minimum(Tt, C)
D = Tt <= C
m = [((X <= s) & D).mean() - 0.5 * np.minimum(X, s).mean() for s in (1, 2, 3)]
check(T, "例2 N(t)−∫Yλ の平均が 0.0003 以下（t=1,2,3）", all(abs(v) <= 3e-4 for v in m), m)
# 一般に Π(1−d/Y) ≤ exp(−Σ d/Y)
ys = rng.integers(1, 50, (1000, 10)); ds = np.minimum(rng.integers(0, 3, (1000, 10)), ys)
check(T, "e^{−Λ̂} ≥ KM がいつも成り立つ（1−x ≤ e^{−x}）", np.all(np.prod(1 - ds / ys, 1) <= np.exp(-(ds / ys).sum(1)) + 1e-15))

T = "competing-risks"
l1, l2 = 0.1, 0.2
p = chart(T)
F1 = lambda s: l1 / (l1 + l2) * (1 - np.exp(-(l1 + l2) * s))
check(T, "図：累積発生関数と 1−KM（原因 2 を打ち切り）", allclose(p["series"][0]["y"], [F1(s) for s in p["x"]], 6e-5) and allclose(p["series"][1]["y"], [1 - np.exp(-l1 * s) for s in p["x"]], 6e-5))
check(T, "例 t=5：0.259 と 0.393、極限 1/3", close(F1(5), 0.259, 5e-4) and close(1 - np.exp(-0.5), 0.393, 5e-4) and close(F1(1e6), 1 / 3, 1e-9))
T1 = rng.exponential(1 / l1, 10 ** 6)
T2 = rng.exponential(1 / l2, 10 ** 6)
check(T, "シミュレーション：P(T≤5, 原因 1) ≈ 0.259", close(((np.minimum(T1, T2) <= 5) & (T1 < T2)).mean(), 0.259, 0.002))
# アーレン・ヨハンセン推定量と 1−KM をシミュレーションのデータ（打ち切りあり）で比べる
n = 20000
a1, a2, cc = rng.exponential(10, n), rng.exponential(5, n), rng.uniform(0, 30, n)
tt = np.minimum(np.minimum(a1, a2), cc)
cause = np.where(tt == cc, 0, np.where(a1 < a2, 1, 2))
o = np.argsort(tt)
tt, cause = tt[o], cause[o]
Yr = n - np.arange(n)
S, F, KM1 = 1.0, 0.0, 1.0
for i in range(n):
    if tt[i] > 5:
        break
    if cause[i] == 1:
        F += S / Yr[i]
        KM1 *= 1 - 1 / Yr[i]
    if cause[i] > 0:
        S *= 1 - 1 / Yr[i]
check(T, "アーレン・ヨハンセン推定量 ≈ 0.259、1−KM ≈ 0.393（打ち切りのあるデータ、t=5）", close(F, 0.259, 0.01) and close(1 - KM1, 0.393, 0.015), (F, 1 - KM1))
check(T, "本文の値", all(v in body(T) for v in ("0.259", "0.393")))
done("第57章（生存時間解析の理論）")
