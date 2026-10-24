"""2級 第3〜6章（2変数の記述、単回帰、時系列、データの集め方）で2級向けに書いた用語の検算。"""
import numpy as np
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-3):
    return abs(float(x) - claim) <= tol


x = np.array([1, 2, 3, 4, 5.]); y = np.array([3, 5, 6, 8, 8.])
T = "scatter-plot"
r = np.corrcoef(x, y)[0, 1]
check(T, "r = 13/√180 ≈ 0.969", close(r, 0.969, 5e-4) and close(13 / np.sqrt(180), r, 1e-12), r)
xq = np.array([-2, -1, 0, 1, 2.])
check(T, "y=x² では r=0", close(np.corrcoef(xq, xq ** 2)[0, 1], 0, 1e-12), np.corrcoef(xq, xq ** 2)[0, 1])
T = "simple-regression"
Sxx = ((x - 3) ** 2).sum(); Sxy = ((x - 3) * (y - 6)).sum(); Syy = ((y - 6) ** 2).sum()
check(T, "平均 3, 6・Sxx 10・Sxy 13・Syy 18", x.mean() == 3 and y.mean() == 6 and (Sxx, Sxy, Syy) == (10, 13, 18), (Sxx, Sxy, Syy))
b, a = np.polyfit(x, y, 1)
check(T, "b=1.3・a=2.1", close(b, 1.3, 1e-9) and close(a, 2.1, 1e-9), (b, a))
yh = a + b * x; e = y - yh
check(T, "予測値 3.4,4.7,6.0,7.3,8.6・残差 −0.4,0.3,0,0.7,−0.6", np.allclose(yh, [3.4, 4.7, 6, 7.3, 8.6]) and np.allclose(e, [-0.4, 0.3, 0, 0.7, -0.6]), e)
check(T, "残差平方和 1.1・R²≈0.939=r²", close((e ** 2).sum(), 1.1, 1e-9) and close(1 - 1.1 / 18, 0.939, 5e-4) and close(1 - 1.1 / 18, r ** 2, 1e-12), 1 - 1.1 / 18)
check(T, "b = r sy/sx", close(r * y.std(ddof=1) / x.std(ddof=1), 1.3, 1e-12), None)
T = "stratified-scatter"
rng = np.random.default_rng(1)
xs, ys = [], []
rs = []
for g in (0, 1):
    xx = g * 3 + rng.standard_normal(400); yy = g * 3 - 0.5 * (xx - g * 3) + 0.6 * rng.standard_normal(400)
    rs.append(np.corrcoef(xx, yy)[0, 1]); xs += list(xx); ys += list(yy)
rall = np.corrcoef(xs, ys)[0, 1]
check(T, "層の中は負・全体は正（図の既定値 gap 3・within −0.5）", rs[0] < 0 and rs[1] < 0 and rall > 0, (rs, rall))
T = "cross-tabulation"
tab = np.array([[30, 90], [56, 24]])
check(T, "周辺度数 120,80 / 86,114", list(tab.sum(1)) == [120, 80] and list(tab.sum(0)) == [86, 114], None)
check(T, "行パーセント 25%・70%、列パーセント 56/86≈65%、30/86≈35%", close(30 / 120, 0.25) and close(56 / 80, 0.7) and close(56 / 86, 0.65, 0.005) and close(30 / 86, 0.35, 0.005), None)
E = np.outer(tab.sum(1), tab.sum(0)) / 200
check(T, "期待度数 51.6", close(E[0, 0], 51.6, 1e-9), E[0, 0])
chi2 = stats.chi2_contingency(tab, correction=False)
check(T, "独立性の検定で有意（関連あり）", chi2.pvalue < 0.001, (chi2.statistic, chi2.pvalue))
T = "time-series-components"
q = np.array([20, 32, 25, 15, 24, 36, 29, 19, 28, 40, 33, 23.])
ma = np.array([(0.5 * q[t - 2] + q[t - 1] + q[t] + q[t + 1] + 0.5 * q[t + 2]) / 4 for t in range(2, 10)])
check(T, "4項中心化移動平均 23.5〜30.5（1ずつ増加）", np.allclose(ma, np.arange(23.5, 31, 1)), ma)
check(T, "前年同期比 24/20−1 = 20%・同じ四半期は毎年 +4", close(24 / 20 - 1, 0.2, 1e-12) and np.all(q[4:] - q[:-4] == 4), None)
check(T, "15→24 は 60% 増", close(24 / 15 - 1, 0.6, 1e-12), None)
T = "moving-average"
check(T, "(10+32+25+15+12)/4 = 23.5", (10 + 32 + 25 + 15 + 12) / 4 == 23.5, None)
check(T, "3 項移動平均の 2 時点目 ≈ 25.7", close((20 + 32 + 25) / 3, 25.7, 0.05), (20 + 32 + 25) / 3)
ma3 = np.convolve(q, np.ones(3) / 3, "valid")
check(T, "3 項では上下が残る（単調でない）", not np.all(np.diff(ma3) > 0), ma3)
T = "growth-rate"
s = np.array([100, 120, 108, 135.])
g = s[1:] / s[:-1] - 1
check(T, "成長率 +20%・−10%・+25%", np.allclose(g, [0.2, -0.1, 0.25]), g)
gb = 1.35 ** (1 / 3) - 1
check(T, "算術平均 11.7%・平均成長率 10.5%", close(g.mean(), 0.117, 5e-4) and close(gb, 0.105, 5e-4), (g.mean(), gb))
check(T, "100×1.105³≈135、1.117³×100≈139", close(100 * (1 + gb) ** 3, 135, 1e-9) and close(100 * 1.117 ** 3, 139, 0.5), 100 * 1.117 ** 3)
check(T, "図の値 110.52・122.15", close(100 * (1 + gb), 110.52, 0.01) and close(100 * (1 + gb) ** 2, 122.15, 0.01), None)
check(T, "1.1×0.9 = 0.99", close(1.1 * 0.9, 0.99, 1e-12), None)
check(T, "寄与度 6・14 ポイント、寄与率 30%・70%", (66 - 60) / 100 == 0.06 and (54 - 40) / 100 == 0.14 and close(6 / 20, 0.3, 1e-12) and close(14 / 20, 0.7, 1e-12), None)
T = "index-number"
v = np.array([250, 270, 300, 285.])
I = v / v[0] * 100
check(T, "指数 100,108,120,114", np.allclose(I, [100, 108, 120, 114]), I)
I2 = I / I[2] * 100
check(T, "2024=100 で 83.3, 90, 100, 95", np.allclose(I2, [83.33, 90, 100, 95], atol=0.01), I2)
check(T, "120→114 は −5%（付け替え後も同じ）", close(114 / 120 - 1, -0.05, 1e-12) and close(I2[3] / I2[2] - 1, -0.05, 1e-12), None)
T = "price-index"
p0 = np.array([200, 150]); q0 = np.array([10, 20]); p1 = np.array([240, 150]); q1 = np.array([8, 22])
L = (p1 @ q0) / (p0 @ q0) * 100; P = (p1 @ q1) / (p0 @ q1) * 100
check(T, "ラスパイレス 108.0（5400/5000）", (p1 @ q0, p0 @ q0) == (5400, 5000) and close(L, 108, 1e-9), L)
check(T, "パーシェ 約 106.5（5220/4900）", (p1 @ q1, p0 @ q1) == (5220, 4900) and close(P, 106.53, 0.005), P)
check(T, "フィッシャー 約 107.3", close(np.sqrt(L * P), 107.26, 0.01), np.sqrt(L * P))
w = p0 * q0 / (p0 @ q0)
check(T, "重み 0.4・0.6 の加重平均でも 108", np.allclose(w, [0.4, 0.6]) and close((w * p1 / p0).sum() * 100, 108, 1e-9), w)
T = "observational-vs-experimental"
rng = np.random.default_rng(2)
z = rng.standard_normal(200000); X = 1.5 * z + rng.standard_normal(200000); Y = 1.5 * z + rng.standard_normal(200000)
Xr = rng.standard_normal(200000) * X.std()
check(T, "交絡があると相関が出る（理論値 0.692）・ランダムに割り付けると消える", close(np.corrcoef(X, Y)[0, 1], 2.25 / 3.25, 0.01) and abs(np.corrcoef(Xr, Y)[0, 1]) < 0.01, (np.corrcoef(X, Y)[0, 1], np.corrcoef(Xr, Y)[0, 1]))
T = "population-and-sample"
check(T, "標準誤差 √(0.56×0.44/1000) ≈ 0.016", close(np.sqrt(0.56 * 0.44 / 1000), 0.016, 5e-4), np.sqrt(0.56 * 0.44 / 1000))
check(T, "n を 4 倍で標準誤差は半分", close(np.sqrt(0.25 / 4000) / np.sqrt(0.25 / 1000), 0.5, 1e-12), None)

ng = results.count(False)
print(f"第3〜6章（2級）の検算: OK {results.count(True)} / NG {ng}")
raise SystemExit(1 if ng else 0)
