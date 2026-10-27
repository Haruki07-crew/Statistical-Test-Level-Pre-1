"""2級 教科書 第1章「データの種類とグラフ」の数値の検算。"""
import math

import numpy as np
from bookcheck import Book

b = Book("g2", 1)

# 1.2 摂氏と華氏
f10, f20 = 10 * 9 / 5 + 32, 20 * 9 / 5 + 32
b.num("10℃ を華氏に", f10, "50")
b.num("20℃ を華氏に", f20, "68")
b.num("華氏での比", f20 / f10, "1.36")

# 1.3 通学手段の相対度数と中心角
cnt = {"徒歩": 6, "自転車": 14, "バス": 8, "電車": 12}
n = sum(cnt.values())
b.check("全体は 40 人", n == 40, n)
for k, shown_p, shown_a in [("徒歩", "15%", "54"), ("自転車", "35%", "126"), ("バス", "20%", "72"), ("電車", "30%", "108")]:
    b.num(f"{k}の相対度数", cnt[k] / n, shown_p)
    b.num(f"{k}の中心角", cnt[k] / n * 360, shown_a)

# 1.4 通学時間 30 人
x = np.array([5, 8, 12, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25, 25, 27, 28, 30, 31, 33, 35, 36, 38, 40, 42, 45, 48, 52, 55, 63, 78])
b.check("本文のデータ列と一致", b.has("5, 8, 12, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25, 25, 27, 28, 30, 31, 33, 35, 36, 38, 40, 42, 45, 48, 52, 55, 63, 78"), len(x))
b.check("30 人", len(x) == 30, len(x))
f = np.histogram(x, bins=range(0, 90, 10))[0]
b.check("度数 2,5,9,6,4,2,1,1", list(f) == [2, 5, 9, 6, 4, 2, 1, 1], list(f))
b.check("図1の度数が表と一致", b.chart(1)["values"] == list(map(int, f)), b.chart(1)["values"])
cum = np.cumsum(f)
b.check("累積度数", list(cum) == [2, 7, 16, 22, 26, 28, 29, 30], list(cum))
rel, crel = f / 30, cum / 30
for i in range(8):
    lo = i * 10
    row = f"| {lo}〜{lo + 10} | {lo + 5} | {f[i]} | {rel[i]:.3f} | {cum[i]} | {crel[i]:.3f} |"
    b.check(f"表の行 {lo}〜{lo + 10}", b.has(row), row)
b.check("図2の累積相対度数が表と一致（小数第3位）", np.allclose(b.chart(2)["y"] if "y" in b.chart(2) else b.chart(2)["series"][0]["y"], [0] + [round(v, 3) for v in crel], atol=1e-9), b.chart(2)["series"][0]["y"])
b.num("スタージェスの公式 n=30", 1 + math.log2(30), "5.9")
m = np.arange(5, 85, 10)
b.num("度数分布表からの平均", (m * f).sum() / 30, "31.7")
b.num("合計", x.sum(), "930")
b.num("平均", x.mean(), "31.0")
srt = np.sort(x)
b.check("15・16 番目は 27・28", (srt[14], srt[15]) == (27, 28), (srt[14], srt[15]))
b.num("中央値", np.median(x), "27.5")
# 幹葉図
for stem in range(8):
    leaves = " ".join(str(v % 10) for v in x if v // 10 == stem)
    b.check(f"幹葉図の幹 {stem}", b.has(f"| {stem} | {leaves} |"), leaves)
# 例題：20 以上 50 未満
b.num("累積相対度数の差", round(crel[4], 3) - round(crel[1], 3), "0.634")
b.num("度数での割合", (cum[4] - cum[1]) / 30, "0.633")
b.check("19 人", cum[4] - cum[1] == 19, cum[4] - cum[1])
b.check("30 分未満は 16 人", cum[2] == 16, cum[2])
b.num("30 分未満の割合", cum[2] / 30, "0.533")

# 1.6 幅の違う階級（貯蓄額）
for c, w, shown in [(30, 1, "30"), (25, 1, "25"), (30, 2, "15"), (15, 4, "3.75")]:
    b.num(f"密度 {c}/{w}", c / w, shown)

# 章末問題 2
fr = [12, 18, 11, 6, 3]
b.check("読書時間 50 人", sum(fr) == 50, sum(fr))
b.num("4〜6 時間までの累積相対度数", sum(fr[:3]) / 50, "0.82")
s = sum(m_ * f_ for m_, f_ in zip([1, 3, 5, 7, 9], fr))
b.num("階級値×度数の合計", s, "190")
b.num("平均の近似", s / 50, "3.8")
# 章末問題 3
b.check("年齢 200 人", 40 + 50 + 60 + 50 == 200, 200)
b.num("60〜100 歳の高さ", 50 / (40 / 20), "25")
b.done()
