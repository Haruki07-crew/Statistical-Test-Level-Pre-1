"""教科書 第55章（因果推論の方法）の検算。"""
import itertools

import numpy as np

from causal_sim import aipw, gen, ipw, naive, outcome, ps, strat
from did_sim import did22, group_means, panel
from rd_sim import data, global_linear, local_linear
from tbcheck import allclose, body, chart, check, close, done

T = "potential-outcomes-framework"
Y0 = np.array([10, 12, 9, 14, 11, 13, 8, 15.0])
Y1 = np.array([13, 15, 14, 15, 12, 18, 10, 20.0])
est, vh = [], []
for S in itertools.combinations(range(8), 4):
    m = np.zeros(8, bool)
    m[list(S)] = True
    est.append(Y1[m].mean() - Y0[~m].mean())
    vh.append(Y1[m].var(ddof=1) / 4 + Y0[~m].var(ddof=1) / 4)
est = np.array(est)
p = chart(T)
b = np.floor(est).astype(int)
check(T, "図：70 通りの推定値の度数（1 刻み）", p["values"] == [int((b == k).sum()) for k in range(-2, 8)] and len(est) == 70, p["values"])
S1, S0, St = Y1.var(ddof=1), Y0.var(ddof=1), (Y1 - Y0).var(ddof=1)
check(T, "例 効果 3,3,5,1,1,5,2,5・ATE 3.125・不偏・分散 3.694 = 10.268/4+6.0/4−2.982/8・ネイマン 4.067（約 10% 大）",
      list(Y1 - Y0) == [3, 3, 5, 1, 1, 5, 2, 5] and close(est.mean(), 3.125, 1e-12) and close(est.var(), 3.694, 5e-4) and close(S1 / 4 + S0 / 4 - St / 8, est.var(), 1e-12)
      and close(S1, 10.268, 5e-4) and close(S0, 6.0, 1e-12) and close(St, 2.982, 5e-4) and close(np.mean(vh), 4.067, 5e-4) and close(np.mean(vh) / est.var(), 1.10, 0.005))
check(T, "例 1・3・6・8 番目が処置なら 5.0、範囲 −1.25〜7.5", close(Y1[[0, 2, 5, 7]].mean() - Y0[[1, 3, 4, 6]].mean(), 5.0, 1e-12) and est.min() == -1.25 and est.max() == 7.5)

x1, x2, t, y, e = gen()
er = ps(x1, x2, t)
reps = {k: [] for k in ("naive", "s5", "s10", "ipw", "ipw_u", "ipw_w", "reg_w", "dr_rr", "dr_wr", "dr_rw", "dr_ww")}
for r in range(300):
    a1, a2, tt, yy, _ = gen(seed=1000 + r)
    e_r, e_w = ps(a1, a2, tt, True), ps(a1, a2, tt, False)
    m0r, m1r = outcome(a1, a2, tt, yy, True)
    m0w, m1w = outcome(a1, a2, tt, yy, False)
    reps["naive"].append(naive(tt, yy)); reps["s5"].append(strat(tt, yy, e_r)); reps["s10"].append(strat(tt, yy, e_r, 10))
    reps["ipw"].append(ipw(tt, yy, e_r)); reps["ipw_u"].append(ipw(tt, yy, e_r, False)); reps["ipw_w"].append(ipw(tt, yy, e_w))
    reps["reg_w"].append(np.mean(m1w - m0w))
    reps["dr_rr"].append(aipw(tt, yy, e_r, m0r, m1r)); reps["dr_wr"].append(aipw(tt, yy, e_w, m0r, m1r))
    reps["dr_rw"].append(aipw(tt, yy, e_r, m0w, m1w)); reps["dr_ww"].append(aipw(tt, yy, e_w, m0w, m1w))
bias = {k: np.mean(v) - 2 for k, v in reps.items()}
sd = {k: np.std(v) for k, v in reps.items()}

T = "propensity-score"
p = chart(T)
bins = np.minimum((er * 10).astype(int), 9)
check(T, "図：処置群・対照群の傾向スコアの分布", allclose(p["series"][0]["y"], [((bins == k) & (t == 1)).sum() / (t == 1).sum() for k in range(10)], 6e-5)
      and allclose(p["series"][1]["y"], [((bins == k) & (t == 0)).sum() / (t == 0).sum() for k in range(10)], 6e-5))
smd = lambda v: (v[t == 1].mean() - v[t == 0].mean()) / np.sqrt((v[t == 1].var() + v[t == 0].var()) / 2)
q = np.quantile(er, np.linspace(0, 1, 6))
idx = np.clip(np.searchsorted(q, er, side="right") - 1, 0, 4)
ws = []
for j in range(5):
    mm = idx == j
    v, tt = x1[mm], t[mm]
    ws.append((v[tt == 1].mean() - v[tt == 0].mean()) / np.sqrt((v[tt == 1].var() + v[tt == 0].var()) / 2))
check(T, "例 処置 43.2%・単純差 3.79・SMD 0.74 と 0.50・5 層 2.11（層内 SMD 平均 0.11）・10 層 1.98",
      close(t.mean(), 0.432, 5e-4) and close(naive(t, y), 3.79, 0.005) and close(smd(x1), 0.74, 0.005) and close(smd(x2), 0.50, 0.005)
      and close(strat(t, y, er), 2.11, 0.005) and close(np.mean(np.abs(ws)), 0.11, 0.005) and close(strat(t, y, er, 10), 1.98, 0.005))
check(T, "例 300 回の偏り：+1.74・+0.18・+0.07", close(bias["naive"], 1.74, 0.005) and close(bias["s5"], 0.18, 0.005) and close(bias["s10"], 0.07, 0.005), bias)
# バランシング性：真の傾向スコアの細かい層の中では X1 の平均が群で等しい（大きな標本で）
X1b, X2b, Tb, _, Eb = gen(n=400000, seed=99)
cell = (Eb > 0.48) & (Eb < 0.52)
check(T, "バランシング性：e(X)≈0.5 の人の中では X1 の平均が 2 群でほぼ等しい", abs(X1b[cell & (Tb == 1)].mean() - X1b[cell & (Tb == 0)].mean()) < 0.03)

T = "inverse-probability-weighting"
w = np.where(t == 1, 1 / er, 1 / (1 - er))
wsmd = (np.average(x1[t == 1], weights=w[t == 1]) - np.average(x1[t == 0], weights=w[t == 0])) / np.sqrt((x1[t == 1].var() + x1[t == 0].var()) / 2)
check(T, "例 正規化 1.86・HT 1.83・最大の重み 20.2・重みの総和 4975 と 5025・重み付け後の SMD −0.002",
      close(ipw(t, y, er), 1.86, 0.005) and close(ipw(t, y, er, False), 1.83, 0.005) and close(w.max(), 20.2, 0.05)
      and close(w[t == 1].sum(), 4975, 1) and close(w[t == 0].sum(), 5025, 1) and close(wsmd, -0.002, 5e-4))
check(T, "例 300 回：正規化 偏り −0.001・SD 0.094、HT −0.002・0.114、モデルの誤り +1.34",
      close(bias["ipw"], -0.001, 5e-4) and close(sd["ipw"], 0.094, 5e-4) and close(bias["ipw_u"], -0.002, 5e-4) and close(sd["ipw_u"], 0.114, 5e-4)
      and close(bias["ipw_w"], 1.34, 0.005))
p = chart(T)


def ipw_cap(tt, yy, ee, c):
    w1, w0 = np.minimum(tt / ee, c), np.minimum((1 - tt) / (1 - ee), c)
    return (w1 * yy).sum() / w1.sum() - (w0 * yy).sum() / w0.sum()


capres = {c: [] for c in p["x"]}
for r in range(300):
    a1, a2, tt, yy, _ = gen(seed=1000 + r)
    ee = ps(a1, a2, tt)
    for c in p["x"]:
        capres[c].append(ipw_cap(tt, yy, ee, c))
cb = [np.mean(capres[c]) - 2 for c in p["x"]]
cs = [np.std(capres[c]) for c in p["x"]]
cr = [np.sqrt(np.mean((np.array(capres[c]) - 2) ** 2)) for c in p["x"]]
check(T, "図：切り詰めの偏り・SD・RMSE（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], cb, 6e-5) and allclose(p["series"][1]["y"], cs, 6e-5)
      and allclose(p["series"][2]["y"], cr, 6e-5) and p["x"][int(np.argmin(cr))] == 10)

T = "doubly-robust-estimator"
check(T, "例 表：偏り 0.002・0.001・−0.002・−0.082、SD 0.035・0.034・0.126・0.078",
      allclose([bias[k] for k in ("dr_rr", "dr_wr", "dr_rw", "dr_ww")], [0.002, 0.001, -0.002, -0.082], 6e-4)
      and allclose([sd[k] for k in ("dr_rr", "dr_wr", "dr_rw", "dr_ww")], [0.035, 0.034, 0.126, 0.078], 6e-4))
check(T, "例 比較：IPW（誤り）+1.34・回帰（誤り）−0.083", close(bias["ipw_w"], 1.34, 0.005) and close(bias["reg_w"], -0.083, 6e-4))
# 偏りの式（誤差の積）を数値で確かめる：μ₁ をわざと +0.5 ずらし、e も誤らせる
X1c, X2c, Tc, Yc, Ec = gen(n=2_000_000, seed=7)
mu1_true = 1 + 2 + 0.5 * X1c + 1.5 * X1c + X2c + 1.5 * X1c ** 2
e_bad = np.clip(Ec * 0.8 + 0.1, 0.01, 0.99)
lhs = np.mean(mu1_true + 0.5 + Tc * (Yc - mu1_true - 0.5) / e_bad) - np.mean(mu1_true)
rhs = np.mean((Ec - e_bad) * (-0.5) / e_bad)
check(T, "偏り = E[(e°−e)(μ°−μ)/e]（200 万人で数値確認）", close(lhs, rhs, 0.01), (lhs, rhs))

T = "difference-in-differences"
tr, yp = panel()
m1, m0 = group_means(tr, yp)
p = chart(T)
gap = (m1[:3] - m0[:3]).mean()
check(T, "図：2 群の平均と反事実（対照群 + 4.75）", allclose(p["series"][0]["y"], m1, 6e-3) and allclose(p["series"][1]["y"], m0, 6e-3)
      and allclose(p["series"][2]["y"], m0 + gap, 6e-3) and close(gap, 4.75, 0.005))
check(T, "例 差 4.80・4.67・4.80 | 7.82・7.81・7.61", allclose(m1 - m0, [4.80, 4.67, 4.80, 7.82, 7.81, 7.61], 6e-3))
n_, T_ = yp.shape
D = (tr[:, None] * (np.arange(T_) >= 3)[None, :]).astype(float)
dm = lambda a: a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
twfe = (dm(yp) * dm(D)).sum() / (dm(D) ** 2).sum()
check(T, "例 DID 2.99（31.66・25.74・23.91・20.98）・二方向固定効果 2.99・単純比較 7.75・前後比較 5.92",
      close(did22(tr, yp), 2.99, 0.005) and close(twfe, did22(tr, yp), 1e-9) and allclose([m1[3:].mean(), m1[:3].mean(), m0[3:].mean(), m0[:3].mean()], [31.66, 25.74, 23.91, 20.98], 6e-3)
      and close(m1[3:].mean() - m0[3:].mean(), 7.75, 0.005) and close(m1[3:].mean() - m1[:3].mean(), 5.92, 0.005))
tr2, y2 = panel(extra_trend=0.5)
check(T, "例 平行トレンドが崩れると 4.49（偏り 1.5）", close(did22(tr2, y2), 4.49, 0.005) and close(did22(tr2, y2) - did22(tr, yp), 1.5, 1e-9))

T = "regression-discontinuity-design"
x, d, yr = data()
check(T, "例 単純差 29.86・共通傾きの直線 −6.14・局所線形 h=5,10,20,30 で 3.92・3.99・3.01・−0.93",
      close(yr[d == 1].mean() - yr[d == 0].mean(), 29.86, 0.005) and close(global_linear(x, yr), -6.14, 0.005)
      and allclose([local_linear(x, yr, h=h) for h in (5, 10, 20, 30)], [3.92, 3.99, 3.01, -0.93], 0.005))
p = chart(T)
rr = {h: [] for h in p["x"]}
for r in range(300):
    xx, _, yy = data(seed=3000 + r)
    for h in p["x"]:
        rr[h].append(local_linear(xx, yy, h=h))
rb = [abs(np.mean(rr[h]) - 4) for h in p["x"]]
rs = [np.std(rr[h]) for h in p["x"]]
rm = [np.sqrt(np.mean((np.array(rr[h]) - 4) ** 2)) for h in p["x"]]
check(T, "図：偏りの大きさ・SD・RMSE（同じ乱数で計算し直す）、h=10 が最良（偏り −0.22・SD 0.87・RMSE 0.90）、h=30 で −5.38",
      allclose(p["series"][0]["y"], rb, 6e-4) and allclose(p["series"][1]["y"], rs, 6e-4) and allclose(p["series"][2]["y"], rm, 6e-4)
      and p["x"][int(np.argmin(rm))] == 10 and close(np.mean(rr[10]) - 4, -0.22, 0.005) and close(np.mean(rr[30]) - 4, -5.38, 0.005))
check(T, "本文の値", all(v in body(T) for v in ("29.86", "-6.14", "3.92", "3.99", "3.01", "-0.93")))
done("第55章（因果推論の方法）")
