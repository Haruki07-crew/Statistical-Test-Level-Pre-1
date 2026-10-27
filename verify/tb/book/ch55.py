"""統計学の教科書 第55章「因果推論の方法」の検算（IPW の表計算・二重頑健のシミュレーション・DID・IV）。"""
import numpy as np
from bookcheck import Book

b = Book("tb", 55)
# 例題：IPW
grp = {(1, 1): (400, 7), (1, 0): (100, 5), (0, 1): (100, 3), (0, 0): (400, 2)}  # (X, T): (人数, 平均)
e = {1: 0.8, 0: 0.2}
n1 = sum(v[0] for (x, t), v in grp.items() if t == 1)
y1 = sum(v[0] * v[1] for (x, t), v in grp.items() if t == 1) / n1
n0 = sum(v[0] for (x, t), v in grp.items() if t == 0)
y0 = sum(v[0] * v[1] for (x, t), v in grp.items() if t == 0) / n0
b.num("処置群の平均", y1, "6.2")
b.num("対照群の平均", y0, "2.6")
b.num("単純な差", y1 - y0, "3.6")
w1 = {x: grp[(x, 1)][0] / e[x] for x in (0, 1)}
w0 = {x: grp[(x, 0)][0] / (1 - e[x]) for x in (0, 1)}
b.check("重みの合計はどちらも 500", all(abs(v - 500) < 1e-9 for v in list(w1.values()) + list(w0.values())), "")
m1 = sum(w1[x] * grp[(x, 1)][1] for x in (0, 1)) / 1000
m0 = sum(w0[x] * grp[(x, 0)][1] for x in (0, 1)) / 1000
b.num("E[Y(1)]", m1, "5")
b.num("E[Y(0)]", m0, "3.5")
b.num("IPW の ATE", m1 - m0, "1.5")
b.check("層別の効果の平均と一致", abs(0.5 * (7 - 5) + 0.5 * (3 - 2) - 1.5) < 1e-12, "")
b.check("傾向スコアが人数と整合", grp[(1, 1)][0] / 500 == 0.8 and grp[(0, 1)][0] / 500 == 0.2, "")


# 二重頑健のシミュレーション（本文の表を再現）
def logit_fit(Z, t, it=50):
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        w += np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z, Z.T @ (t - p))
    return 1 / (1 + np.exp(-Z @ w))


def ols_pred(Z, y, mask):
    return Z @ np.linalg.lstsq(Z[mask], y[mask], rcond=None)[0]


rng = np.random.default_rng(55)
R, n = 500, 2000
keys = ["naive", "reg_ok", "reg_bad", "ipw_ok", "ipw_bad", "dr_both", "dr_ps_ok", "dr_or_ok", "dr_none"]
res = {k: [] for k in keys}
for _ in range(R):
    x = rng.standard_normal(n)
    ee = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.4 * x ** 2)))
    t = (rng.uniform(size=n) < ee).astype(float)
    y = 1 + 2 * t + x + x ** 2 + rng.standard_normal(n)
    Zg = np.c_[np.ones(n), x, x ** 2]; Zb = np.c_[np.ones(n), x]
    eg, eb = logit_fit(Zg, t), logit_fit(Zb, t)
    m1g, m0g = ols_pred(Zg, y, t == 1), ols_pred(Zg, y, t == 0)
    m1b, m0b = ols_pred(Zb, y, t == 1), ols_pred(Zb, y, t == 0)

    def ipw(e_):
        a, c = t / e_, (1 - t) / (1 - e_)
        return (a * y).sum() / a.sum() - (c * y).sum() / c.sum()

    def dr(e_, a, c):
        return np.mean(a - c + t * (y - a) / e_ - (1 - t) * (y - c) / (1 - e_))

    res["naive"].append(y[t == 1].mean() - y[t == 0].mean())
    res["reg_ok"].append(np.mean(m1g - m0g)); res["reg_bad"].append(np.mean(m1b - m0b))
    res["ipw_ok"].append(ipw(eg)); res["ipw_bad"].append(ipw(eb))
    res["dr_both"].append(dr(eg, m1g, m0g)); res["dr_ps_ok"].append(dr(eg, m1b, m0b))
    res["dr_or_ok"].append(dr(eb, m1g, m0g)); res["dr_none"].append(dr(eb, m1b, m0b))
table = {"naive": ("1.083", "0.084"), "reg_ok": ("0.005", "0.048"), "reg_bad": ("0.606", "0.085"),
         "ipw_ok": ("0.014", "0.224"), "ipw_bad": ("0.718", "0.105"), "dr_ps_ok": ("0.011", "0.235"),
         "dr_or_ok": ("0.005", "0.048"), "dr_none": ("0.877", "0.131")}
for k, (bias, sd) in table.items():
    b.num(f"{k} の偏り", np.mean(res[k]) - 2, bias)
    b.num(f"{k} の SD", np.std(res[k]), sd)
b.check("どちらか正しければ偏りは SD/√R の 3 倍以内", all(abs(np.mean(res[k]) - 2) < 3 * np.std(res[k]) / np.sqrt(R) + 0.005 for k in ("dr_ps_ok", "dr_or_ok", "dr_both")), "")
# DID・IV
b.num("DID", (15 - 10) - (10 - 8), "3")
b.num("問2 DID", (66 - 60) - (61 - 58), "3")
b.num("ワルド", 1.5 / (0.7 - 0.2), "3")
b.num("問3", 0.6 / (0.4 - 0.1), "2")
# LATE のシミュレーション（コンプライアーだけ効果 3、常に受診する人は効果 1）
g = np.random.default_rng(7)
N = 400000
typ = g.choice(3, N, p=[0.5, 0.2, 0.3])  # 0: コンプライアー, 1: 常に受診, 2: 決して受診しない
Z = g.integers(0, 2, N)
T = np.where(typ == 1, 1, np.where(typ == 0, Z, 0))
eff = np.where(typ == 0, 3.0, 1.0)
Y = g.standard_normal(N) + typ + eff * T
late = (Y[Z == 1].mean() - Y[Z == 0].mean()) / (T[Z == 1].mean() - T[Z == 0].mean())
b.check("ワルド推定量はコンプライアーの効果 3 を推定", abs(late - 3) < 0.05, late)
# 章末問題
b.num("問1 処置", 1 / 0.25, "4")
b.num("問1 対照", 1 / 0.75, "1.33")
b.done()
