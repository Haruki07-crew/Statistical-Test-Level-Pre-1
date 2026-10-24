"""2級 第1章（データの種類とグラフ）の検算。本文の具体例の数値を計算し直す。"""
import math
import numpy as np

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-4):
    return abs(float(x) - claim) <= tol


# 通学時間 40 人の度数分布（frequency-distribution-histogram・cumulative-frequency で共通）
mid = np.array([5, 15, 25, 35, 45, 55]); f = np.array([4, 9, 13, 8, 4, 2])
T = "frequency-distribution-histogram"
check(T, "度数の合計 40", f.sum() == 40, f.sum())
check(T, "相対度数", np.allclose(f / 40, [0.1, 0.225, 0.325, 0.2, 0.1, 0.05]), f / 40)
check(T, "平均の近似 1050/40=26.25", (mid * f).sum() == 1050 and close((mid * f).sum() / 40, 26.25), (mid * f).sum() / 40)
check(T, "スタージェス 1+log2(40)≈6.3", close(1 + math.log2(40), 6.32, 0.01), 1 + math.log2(40))
cum = np.cumsum(f)
check(T, "中央値（20・21番目）は 20〜30 の階級", cum[1] < 20 and cum[2] >= 21, cum)
T = "cumulative-frequency"
check(T, "累積度数 4,13,26,34,38,40", list(cum) == [4, 13, 26, 34, 38, 40], cum)
check(T, "累積相対度数", np.allclose(cum / 40, [0.1, 0.325, 0.65, 0.85, 0.95, 1.0]), cum / 40)
med = 20 + (0.5 - 0.325) / (0.65 - 0.325) * 10
check(T, "補間した中央値 ≈ 25.4", close(med, 25.4, 0.05), med)
check(T, "40分以上 15%", close(1 - 0.85, 0.15), 1 - cum[3] / 40)

T = "stem-and-leaf-plot"
x = sorted([48, 52, 55, 57, 61, 63, 63, 66, 68, 70, 72, 75, 79, 84, 91])
check(T, "15 個・幹ごとの葉の数 1,3,5,4,1,1", len(x) == 15 and [sum(1 for v in x if v // 10 == s) for s in range(4, 10)] == [1, 3, 5, 4, 1, 1], len(x))
check(T, "中央値 66（8番目）", x[7] == 66, x[7])
check(T, "Q1 57・Q3 75（中央値を除いた半分の中央値）", np.median(x[:7]) == 57 and np.median(x[8:]) == 75, (np.median(x[:7]), np.median(x[8:])))

T = "bar-pie-band-chart"
c = np.array([42, 23, 15, 12, 8])
check(T, "合計 100・累積 42,65,80,92,100", c.sum() == 100 and list(np.cumsum(c)) == [42, 65, 80, 92, 100], np.cumsum(c))
check(T, "傷の中心角 151.2°", close(0.42 * 360, 151.2), 0.42 * 360)

T = "data-types"
check(T, "自転車の相対度数 25/60≈0.417", close(25 / 60, 0.417), 25 / 60)
check(T, "合計 60", 12 + 25 + 8 + 15 == 60, 60)

T = "scales-of-measurement"
F = lambda cdeg: cdeg * 9 / 5 + 32
check(T, "10℃=50°F、25℃=77°F", F(10) == 50 and F(25) == 77, (F(10), F(25)))
check(T, "標準偏差 4℃ = 7.2°F", close(4 * 9 / 5, 7.2), 4 * 9 / 5)
check(T, "変動係数 0.144 と 0.094", close(7.2 / 50, 0.144) and close(7.2 / 77, 0.094, 5e-4), (7.2 / 50, 7.2 / 77))
check(T, "摂氏では 0.4 と 0.16", close(4 / 10, 0.4) and close(4 / 25, 0.16), (0.4, 0.16))
check(T, "20℃と10℃は華氏で 68 と 50（比は 2 でない）", F(20) == 68 and F(10) == 50 and F(20) / F(10) != 2, F(20) / F(10))

T = "distribution-shape"
mu, s = 0, 0.5
mode, median, mean = math.exp(mu - s * s), math.exp(mu), math.exp(mu + s * s / 2)
check(T, "対数正規の最頻値 0.779・中央値 1・平均 1.133", close(mode, 0.779) and median == 1 and close(mean, 1.133), (mode, median, mean))
rng = np.random.default_rng(20261024)
y = rng.lognormal(mu, s, 1_000_000)
check(T, "シミュレーションで 平均>中央値", y.mean() > np.median(y) and close(y.mean(), 1.133, 0.005), (y.mean(), np.median(y)))

ng = results.count(False)
print(f"第1章（2級）の検算: OK {results.count(True)} / NG {ng}")
raise SystemExit(1 if ng else 0)
