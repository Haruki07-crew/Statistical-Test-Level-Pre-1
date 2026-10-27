"""準1級 教科書 第20章「分散分析と実験計画法」の検算（scipy・直接計算・シミュレーション）。"""
import itertools

import numpy as np
from bookcheck import Book
from scipy import stats as st

b = Book("pre1", 20)
G = [np.array([62, 65, 60, 63.]), np.array([68, 71, 66, 70.]), np.array([64, 61, 66, 63.]), np.array([72, 75, 70, 74.])]
for g, nm, s in zip(G, "ABCD", ["62.50", "68.75", "63.50", "72.75"]):
    b.check(f"表の {nm}", b.has(f"| {nm} | " + " | ".join(str(int(v)) for v in g) + f" | {s} |"), "")
al = np.concatenate(G)
gm = al.mean()
SA = sum(4 * (g.mean() - gm) ** 2 for g in G)
SE = sum(((g - g.mean()) ** 2).sum() for g in G)
b.num("全体平均", gm, "66.875")
b.num("S_A", SA, "274.25")
b.num("S_E", SE, "55.5")
b.num("V_A", SA / 3, "91.42")
b.num("V_E", SE / 12, "4.625")
F = (SA / 3) / (SE / 12)
b.num("F", F, "19.77")
b.check("scipy と一致", abs(st.f_oneway(*G).statistic - F) < 1e-9, "")
b.num("F_0.05(3,12)", st.f.ppf(0.95, 3, 12), "3.49")
b.num("p", st.f.sf(F, 3, 12), "0.00006")
b.num("σ_A² 推定", (SA / 3 - SE / 12) / 4, "21.70")
s2a = (SA / 3 - SE / 12) / 4
b.num("級内相関", s2a / (s2a + SE / 12), "0.82")
q = st.studentized_range.ppf(0.95, 4, 12)
b.num("q", q, "4.20")
hsd = q * np.sqrt(SE / 12 / 4)
b.num("HSD", hsd, "4.51")
m = [g.mean() for g in G]
sig = {(i, j) for i, j in itertools.combinations(range(4), 2) if abs(m[j] - m[i]) > hsd}
b.check("有意な組は AB・AD・BC・CD", sig == {(0, 1), (0, 3), (1, 2), (2, 3)}, sig)
tk = st.tukey_hsd(*G)
b.check("scipy の tukey_hsd と同じ結論", {(i, j) for i, j in itertools.combinations(range(4), 2) if tk.pvalue[i, j] < 0.05} == sig, "")
b.num("η²", SA / (SA + SE), "0.83")
b.num("ω²", (SA - 3 * SE / 12) / (SA + SE + SE / 12), "0.78")
# 期待平均平方（シミュレーション）
rng = np.random.default_rng(20)
alpha = np.array([-2, 1, 0, 1.])
VA = []
for _ in range(20000):
    gg = [rng.normal(a, 2, 4) for a in alpha]
    aa = np.concatenate(gg)
    VA.append(sum(4 * (g.mean() - aa.mean()) ** 2 for g in gg) / 3)
b.check("E[V_A] = σ² + Σnα²/(k−1)", abs(np.mean(VA) - (4 + 4 * (alpha ** 2).sum() / 3)) < 0.15, (np.mean(VA), 4 + 4 * (alpha ** 2).sum() / 3))
# 2^3
D = np.array(list(itertools.product([-1, 1], repeat=3)))
y = np.array([20, 24, 22, 30, 19, 25, 21, 29.])
eff = {nm: (col * y).sum() / 4 for nm, col in [("A", D[:, 0]), ("B", D[:, 1]), ("C", D[:, 2]), ("AB", D[:, 0] * D[:, 1]), ("AC", D[:, 0] * D[:, 2]), ("BC", D[:, 1] * D[:, 2]), ("ABC", D.prod(1))]}
for nm, s in [("A", "-0.5"), ("B", "3.5"), ("C", "6.5"), ("BC", "1.5")]:
    b.num(f"効果 {nm}", eff[nm], s)
b.check("AB・AC・ABC は ±0.5", all(abs(abs(eff[k]) - 0.5) < 1e-12 for k in ["AB", "AC", "ABC"]), "")
half = D.prod(1) == 1
Dh, yh = D[half], y[half]
b.check("半分の 4 回の収率 24・22・19・29", list(yh) == [24, 22, 19, 29], yh)
for i, (nm, alias, s) in enumerate([("A", "BC", "1.0"), ("B", "AC", "4.0"), ("C", "AB", "6.0")]):
    est = (Dh[:, i] * yh).sum() / 2
    b.num(f"一部実施の {nm}", est, s)
    b.check(f"{nm} + {alias} と一致", abs(est - (eff[nm] + eff[alias])) < 1e-12, "")
# 章末問題
b.num("問1 σ_A²", (30 - 5) / 5, "5")
b.num("問2 q", st.studentized_range.ppf(0.95, 3, 15), "3.67")
b.num("問2 HSD", 3.67 * np.sqrt(8 / 6), "4.24")
b.done()
