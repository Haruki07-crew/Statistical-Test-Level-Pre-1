"""2級 教科書 第2章「1変数データの要約」の数値の検算。四分位数は本文と同じ「半分の中央値」の方法。"""
import numpy as np
from bookcheck import Book

b = Book("g2", 2)


def five(v):
    s = sorted(v)
    n = len(s)
    lo, hi = s[: n // 2], s[(n + 1) // 2:]
    return s[0], float(np.median(lo)), float(np.median(s)), float(np.median(hi)), s[-1]


A = np.array([4, 6, 7, 8, 9, 10, 12, 13, 48])
B = np.array([5, 7, 9, 10, 11, 12, 13, 15, 17])
b.check("本文のデータ A", b.has("4, 6, 7, 8, 9, 10, 12, 13, 48"), list(A))
b.check("本文のデータ B", b.has("5, 7, 9, 10, 11, 12, 13, 15, 17"), list(B))
b.check("図1のデータが本文と一致", b.chart(1)["groups"][0]["data"] == list(A) and b.chart(1)["groups"][1]["data"] == list(B), "")
b.num("A の合計", A.sum(), "117")
b.num("A の平均", A.mean(), "13")
b.num("B の平均", B.mean(), "11")
b.num("A の中央値", np.median(A), "9")
b.num("48 を除いた平均", A[:-1].mean(), "8.625")
b.num("48 を除いた中央値", np.median(A[:-1]), "8.5")
b.check("偏差の合計は 0", abs((A - A.mean()).sum()) < 1e-12, (A - A.mean()).sum())

# 加重平均
b.num("2 クラスの合計点", 30 * 62 + 20 * 72, "3300")
b.num("2 クラスの平均", (30 * 62 + 20 * 72) / 50, "66")
# 幾何平均
g = 1.2 * 0.9 * 1.05
b.num("3 年の倍率", g, "1.134")
b.num("幾何平均", g ** (1 / 3), "1.0428")
b.num("平均の伸び率 %", (g ** (1 / 3) - 1) * 100, "4.3")
# 調和平均
b.num("往復の平均の速さ", 2 / (1 / 60 + 1 / 40), "48")
b.num("往復 240km を 5 時間", 240 / (120 / 60 + 120 / 40), "48")
# 算術 ≥ 幾何 ≥ 調和 の数値例
v = np.array([60.0, 40.0])
b.check("算術≥幾何≥調和（60,40）", v.mean() >= np.sqrt(v.prod()) >= 2 / (1 / v).sum(), (v.mean(), np.sqrt(v.prod()), 2 / (1 / v).sum()))

# 分散の例題
x = np.array([3, 5, 6, 8, 8])
b.num("平均", x.mean(), "6")
b.num("偏差平方和", ((x - x.mean()) ** 2).sum(), "18")
b.num("Σx²", (x ** 2).sum(), "198")
b.num("分散", x.var(ddof=1), "4.5")
b.num("標準偏差", x.std(ddof=1), "2.12")
# 部署の標準偏差
b.num("A の標準偏差", A.std(ddof=1), "13.4")
b.num("B の標準偏差", B.std(ddof=1), "3.8")
b.num("48 を除いた A の標準偏差", A[:-1].std(ddof=1), "3.0")
b.check("標準偏差の比は 3 倍以上", A.std(ddof=1) / B.std(ddof=1) > 3, A.std(ddof=1) / B.std(ddof=1))

# 四分位数
fa, fb = five(A), five(B)
b.num("A の Q1", fa[1], "6.5")
b.num("A の Q3", fa[3], "12.5")
b.num("A の IQR", fa[3] - fa[1], "6")
b.num("B の Q1", fb[1], "8")
b.num("B の Q3", fb[3], "14")
b.check("B の IQR も 6", fb[3] - fb[1] == 6, fb[3] - fb[1])
b.num("A の上の境界", fa[3] + 1.5 * (fa[3] - fa[1]), "21.5")
b.num("A の下の境界", fa[1] - 1.5 * (fa[3] - fa[1]), "-2.5")
b.check("A の外れ値は 48 だけ", [v for v in A if v > 21.5 or v < -2.5] == [48], "")
b.check("B に外れ値はない", all(fb[1] - 9 <= v <= fb[3] + 9 for v in B), "")

# 一次変換（気温）
b.num("華氏の平均", 32 + 1.8 * 20, "68")
b.num("華氏の標準偏差", 1.8 * 3, "5.4")
t = np.array([17.0, 20, 23, 18, 22])
f = 32 + 1.8 * t
b.check("一次変換で標準偏差は |b| 倍（数値例）", abs(f.std(ddof=1) - 1.8 * t.std(ddof=1)) < 1e-12, f.std(ddof=1))

# 標準化
b.num("数学の z", (72 - 60) / 8, "1.5")
b.num("数学の偏差値", 50 + 10 * (72 - 60) / 8, "65")
b.num("英語の z", (80 - 70) / 12.5, "0.8")
b.num("英語の偏差値", 50 + 10 * (80 - 70) / 12.5, "58")
z = (A - A.mean()) / A.std(ddof=1)
b.check("標準化すると平均 0・標準偏差 1", abs(z.mean()) < 1e-12 and abs(z.std(ddof=1) - 1) < 1e-12, (z.mean(), z.std(ddof=1)))
# 変動係数
b.num("A 店の CV", 30 / 200, "0.15")
b.num("B 店の CV", 5 / 20, "0.25")

# 歪度
b.num("A 部署の歪度", (z ** 3).mean(), "1.9")
zb = (B - B.mean()) / B.std(ddof=1)
b.check("B 部署の歪度は 0", abs((zb ** 3).mean()) < 1e-12, (zb ** 3).mean())

# ジニ係数
inc = np.array([2, 3, 5, 10])
b.check("図2の所得が本文と一致", b.chart(2)["values"] == list(inc), b.chart(2)["values"])
cum = np.concatenate([[0], np.cumsum(inc) / inc.sum()])
b.check("累積割合 0, 0.1, 0.25, 0.5, 1", np.allclose(cum, [0, 0.1, 0.25, 0.5, 1]), cum)
area = sum(0.25 * (cum[i] + cum[i + 1]) / 2 for i in range(4))
b.num("曲線の下の面積", area, "0.3375")
b.num("ジニ係数", 1 - 2 * area, "0.325")
pair = np.abs(inc[:, None] - inc[None, :]).sum()
b.num("差の絶対値の合計", pair, "52")
b.num("ジニ係数（別の公式）", pair / (2 * 16 * inc.mean()), "0.325")
b.check("上位 1 世帯が半分", inc.max() / inc.sum() == 0.5, inc.max() / inc.sum())

# 章末問題 1
s = np.array([4, 5, 6, 6, 7, 8, 26])
b.num("歩数の合計", s.sum(), "62")
b.num("歩数の平均", s.mean(), "8.86")
b.num("歩数の中央値", np.median(s), "6")
s2 = np.array([4, 5, 6, 6, 7, 8, 9])
b.num("修正後の合計", s2.sum(), "45")
b.num("修正後の平均", s2.mean(), "6.43")
b.check("修正後の中央値も 6", np.median(s2) == 6, np.median(s2))
b.num("平均の動き", s.mean() - s2.mean(), "2.4")
# 章末問題 2
b.num("調整後の平均", 0.8 * 58 + 20, "66.4")
b.num("調整後の標準偏差", 0.8 * 15, "12")
b.num("調整前の z", (73 - 58) / 15, "1")
b.num("調整後の点数", 0.8 * 73 + 20, "78.4")
b.check("調整後の z も 1", abs((78.4 - 66.4) / 12 - 1) < 1e-12, (78.4 - 66.4) / 12)
# 章末問題 3
d = [12, 15, 16, 18, 19, 20, 21, 23, 24, 26, 41]
fd = five(d)
b.check("5 数要約 12,16,20,24,41", fd == (12, 16.0, 20.0, 24.0, 41), fd)
b.num("IQR", fd[3] - fd[1], "8")
b.num("下の境界", fd[1] - 1.5 * 8, "4")
b.num("上の境界", fd[3] + 1.5 * 8, "36")
b.check("外れ値は 41 だけ、ひげは 26 まで", [v for v in d if v > 36 or v < 4] == [41] and max(v for v in d if v <= 36) == 26, "")
# 章末問題 4
b.num("6 個/時と 12 個/時の調和平均", 2 / (1 / 6 + 1 / 12), "8")
b.check("24 個を 3 時間", 24 / (12 / 6 + 12 / 12) == 8, "")
# 章末問題 5
b.num("身長の CV", 8 / 160, "0.05")
b.num("体重の CV", 6 / 50, "0.12")
b.done()
