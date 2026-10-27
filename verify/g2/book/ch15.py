"""2級 教科書 第15章「実験計画と分散分析」の数値の検算（scipy の f_oneway と、平方和の直接計算で確かめる）。"""
import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("g2", 15)
b.num("3 回の検定の誤り", 1 - 0.95 ** 3, "0.143")
G = [np.array([62, 70, 66, 74, 68.]), np.array([58, 64, 61, 55, 62.]), np.array([71, 77, 69, 75, 73.])]
for g, row in zip(G, ["| A | 62, 70, 66, 74, 68 | 68 |", "| B | 58, 64, 61, 55, 62 | 60 |", "| C | 71, 77, 69, 75, 73 | 73 |"]):
    b.check("表の行と平均", b.has(row) and g.mean() == float(row.split("|")[3]), row)
al = np.concatenate(G)
gm = al.mean()
b.num("全体の平均", gm, "67")
SA = sum(5 * (g.mean() - gm) ** 2 for g in G)
SE = sum(((g - g.mean()) ** 2).sum() for g in G)
ST = ((al - gm) ** 2).sum()
b.num("S_A", SA, "430")
for g, s in zip(G, ["80", "50", "40"]):
    b.num("群内", ((g - g.mean()) ** 2).sum(), s)
b.num("S_E", SE, "170")
b.num("S_T", ST, "600")
b.num("群内の平均平方", SE / 12, "14.17")
F = (SA / 2) / (SE / 12)
b.num("F", F, "15.18")
b.num("F_0.05(2,12)", st.f.ppf(0.95, 2, 12), "3.885")
b.num("p", st.f.sf(F, 2, 12), "0.0005")
b.check("scipy の f_oneway と一致", abs(st.f_oneway(*G).statistic - F) < 1e-9, "")
b.check("2 群なら F = t²", abs(st.f_oneway(G[0], G[1]).statistic - st.ttest_ind(G[0], G[1]).statistic ** 2) < 1e-9, "")
# 乱塊法
Y = np.array([[20, 27, 30, 24], [23, 27, 34, 24], [24, 31, 33, 28.]])
gm = Y.mean()
SA = 4 * ((Y.mean(1) - gm) ** 2).sum()
SB = 3 * ((Y.mean(0) - gm) ** 2).sum()
ST = ((Y - gm) ** 2).sum()
SE = ST - SA - SB
for v, s in zip(Y.mean(1), ["25.25", "27.00", "29.00"]):
    b.num("肥料の平均", v, s)
for v, s in zip(Y.mean(0), ["22.33", "28.33", "32.33", "25.33"]):
    b.num("区域の平均", v, s)
b.num("全体の平均", gm, "27.08")
b.num("S_A", SA, "28.17")
b.num("S_B", SB, "164.25")
b.num("S_E", SE, "10.50")
b.num("S_T", ST, "202.92")
b.num("肥料の平均平方", SA / 2, "14.08")
b.num("区域の平均平方", SB / 3, "54.75")
b.num("誤差の平均平方", SE / 6, "1.75")
FA = (SA / 2) / (SE / 6)
b.num("肥料の F", FA, "8.05")
b.num("区域の F", (SB / 3) / (SE / 6), "31.29")
b.num("F_0.05(2,6)", st.f.ppf(0.95, 2, 6), "5.143")
b.num("p", st.f.sf(FA, 2, 6), "0.020")
b.num("無視したときの誤差", ST - SA, "174.75")
Fw = (SA / 2) / ((ST - SA) / 9)
b.num("無視したときの F", Fw, "0.73")
b.num("無視したときの p", st.f.sf(Fw, 2, 9), "0.51")
b.check("無視したときは scipy の f_oneway と一致", abs(st.f_oneway(*Y).statistic - Fw) < 1e-9, "")
# 二元配置
D = {(0, 0): [10, 12, 11], (0, 1): [14, 15, 13], (1, 0): [12, 13, 14], (1, 1): [20, 19, 21]}
al = np.concatenate([np.array(v, float) for v in D.values()])
gm = al.mean()
cm = {k: np.mean(v) for k, v in D.items()}
b.check("組合せの平均 11・14・13・20", [cm[k] for k in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [11, 14, 13, 20], cm)
b.check("図の平均が表と一致", b.chart(1)["series"][0]["y"] == [11, 14] and b.chart(1)["series"][1]["y"] == [13, 20], "")
Am = [np.mean(D[(i, 0)] + D[(i, 1)]) for i in (0, 1)]
Bm = [np.mean(D[(0, j)] + D[(1, j)]) for j in (0, 1)]
SA2 = 6 * sum((m - gm) ** 2 for m in Am)
SB2 = 6 * sum((m - gm) ** 2 for m in Bm)
SAB = 3 * sum((cm[(i, j)] - Am[i] - Bm[j] + gm) ** 2 for i in (0, 1) for j in (0, 1))
SE2 = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in D.values())
ST2 = ((al - gm) ** 2).sum()
for v, s in [(SA2, "48"), (SB2, "75"), (SAB, "12"), (SE2, "8"), (ST2, "143")]:
    b.num("平方和", v, s)
b.check("分解の合計", abs(SA2 + SB2 + SAB + SE2 - ST2) < 1e-9, "")
b.num("F_0.05(1,8)", st.f.ppf(0.95, 1, 8), "5.318")
b.num("主効果 B", ((14 - 11) + (20 - 13)) / 2, "5")
# 章末問題
b.num("問1 群内", 290 - 90, "200")
b.num("問1 F", (90 / 3) / (200 / 20), "3.0")
b.num("問1 F_0.05(3,20)", st.f.ppf(0.95, 3, 20), "3.098")
P = [np.array([3, 5, 4.]), np.array([6, 8, 7.]), np.array([8, 9, 10.])]
pa = np.concatenate(P)
b.num("問2 全体平均", pa.mean(), "6.67")
b.num("問2 群間", sum(3 * (p.mean() - pa.mean()) ** 2 for p in P), "38")
b.num("問2 群内", sum(((p - p.mean()) ** 2).sum() for p in P), "6")
b.num("問2 全体", ((pa - pa.mean()) ** 2).sum(), "44")
b.num("問2 12.67", sum((p.mean() - pa.mean()) ** 2 for p in P), "12.67")
b.done()
