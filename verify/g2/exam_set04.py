"""2級 模擬試験 第4回（g2 set04）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import product
from math import sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(4)
rng = np.random.default_rng(20261027)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


# ---- 大問1
E.check("s04p01a", ans("s04p01a") == "(ア) 名義、(イ) 順序、(ウ) 間隔、(エ) 比例", "尺度の分類")
cum = {110: .15, 120: .35, 130: .62, 140: .80, 150: .93}
E.choice("s04p01b", f"{round((cum[140] - cum[120]) * 200)} 人", "累積の差 × 200")
E.check("s04p01b", opts("s04p01b") == {"90 人", f"{round((cum[130] - cum[120]) * 200)} 人", f"{round((cum[140] - cum[130]) * 200)} 人", "160 人", "45 人"}, "誤答の由来")
# ---- 大問2
A = dict(mn=10, q1=20, md=25, q3=35, mx=50); B = dict(mn=15, q1=22, md=30, q3=33, mx=40)
E.check("s04p02a", (A["q3"] - A["q1"]) > (B["q3"] - B["q1"]) and (B["mx"] - B["mn"]) < (A["mx"] - A["mn"]) and ans("s04p02a") == "四分位範囲は A のほうが大きい", "IQR A 15・B 11、範囲 A 40・B 25")
sk = rng.lognormal(3.2, 0.5, 100000)
E.check("s04p02b", sk.mean() > np.median(sk) and stats.skew(sk) > 0 and ans("s04p02b") == "右に裾が長く、歪度は正と考えられる", "右に裾の長い分布で 平均＞中央値・歪度＞0")
# ---- 大問3
r = {"x1": .65, "x2": .58, "x3": -.70}
E.check("s04p03a", max(r, key=lambda k: abs(r[k])) == "x3" and ans("s04p03a").startswith("駅からの距離 $x_3$、負"), "絶対値最大は x3（負）")
x = np.array([1, 3, 5, 7.]); y = np.array([2, 6, 4, 8.])
E.choice("s04p03b", f(np.cov(x, y)[0, 1], 2), "numpy の cov（n−1）")
E.check("s04p03b", opts("s04p03b") == {f(np.cov(x, y)[0, 1], 2), f(np.cov(x, y, ddof=0)[0, 1], 2), f(np.corrcoef(x, y)[0, 1], 2), "16.00", f(16 / 12, 2)}, "誤答の由来")
# ---- 大問4
xx = rng.uniform(1, 10, 3000); yy = 2 * xx * np.exp(rng.normal(0, 0.3, 3000)); b1, b0 = np.polyfit(xx, yy, 1); res = yy - b0 - b1 * xx; pred = b0 + b1 * xx
lo, hi = pred < np.median(pred), pred >= np.median(pred)
lres = np.log(yy) - np.polyval(np.polyfit(xx, np.log(yy), 1), xx)
E.check("s04p04a", res[hi].std() > 1.5 * res[lo].std() and ans("s04p04a").startswith("誤差の分散が一定でない"), f"乗法的な誤差のデータで残差の標準偏差 小 {res[lo].std():.2f}・大 {res[hi].std():.2f}（扇形）")
# ---- 大問5
E.check("s04p05a", ans("s04p05a") == "10%" and opts("s04p05a") == {f((66 / 60 - 1) * 100, 0) + "%", f((66 / 55 - 1) * 100, 0) + "%", "6%", f((66 / 48 - 1) * 100, 1) + "%", f((55 / 52 - 1) * 100, 1) + "%"}, "前年同期比 10% と誤答の由来")
p0 = np.array([200, 120, 80]); q0 = np.array([5, 10, 15]); p1 = np.array([250, 120, 60]); q1 = np.array([3, 12, 20])
L = p1 @ q0 / (p0 @ q0) * 100; P = p1 @ q1 / (p0 @ q1) * 100
E.choice("s04p05b", f(P, 2), f"パーシェ {P:.4f}")
E.check("s04p05b", opts("s04p05b") == {f(P, 2), f(L, 2), f(np.mean(p1 / p0) * 100, 2), f(p0 @ q1 / (p1 @ q1) * 100, 2), f(sqrt(L * P), 2)}, "誤答の由来")
# ---- 大問6
E.check("s04p06a", ans("s04p06a").startswith("検査すると製品がなくなるので"), "破壊検査は標本調査")
E.check("s04p06b", ans("s04p06b") == "二段抽出法", "学校 → 生徒の 2 段階")
# ---- 大問7
fail = (rng.random(1_000_000) < .1) | (rng.random(1_000_000) < .2)
E.choice("s04p07a", f(1 - .9 * .8, 2), f"シミュレーション {fail.mean():.4f}")
E.check("s04p07a", abs(fail.mean() - .28) < .002 and opts("s04p07a") == {"0.28", "0.30", "0.02", "0.72", "0.26"}, "誤答の値")
out = list(product(range(1, 7), repeat=2)); cond = [o for o in out if o[0] == 6]
pr = Fr(sum(sum(o) >= 10 for o in cond), len(cond))
E.check("s04p07b", ans("s04p07b") == f"{pr.numerator}/{pr.denominator}" and Fr(sum(sum(o) >= 10 for o in out), 36) == Fr(1, 6) and Fr(3, 36) == Fr(1, 12) and Fr(sum(o[1] >= 5 for o in cond), 6) == Fr(1, 3) and Fr(sum(o[1] >= 3 for o in cond), 6) == Fr(2, 3), f"総当たり {pr}・誤答の由来")
prize = rng.choice([1000, 100, 0], 1_000_000, p=[.01, .1, .89])
E.choice("s04p07c", "20 円", f"シミュレーション {prize.mean():.2f}")
E.check("s04p07c", abs(prize.mean() - 20) < 0.5, "シミュレーション")
# ---- 大問8
J = np.array([[.30, .20], [.10, .40]])
E.choice("s04p08a", f"P(Y=1\\mid X=1)={f(J[1, 1] / J[1].sum(), 1)}$ で、$X$ と $Y$ は独立でない", "同時確率 ÷ 周辺確率")
E.check("s04p08a", abs(J[1].sum() * J[:, 1].sum() - J[1, 1]) > 0.05 and f(J[1, 1] / J[:, 1].sum(), 2) == "0.67", "独立でない・誤答 0.67 の由来")
Bn = stats.binom(50, .2)
E.choice("s04p08b", f"期待値 {f(Bn.mean(), 0)}、分散 {f(Bn.var(), 0)}", "scipy の binom")
E.check("s04p08b", opts("s04p08b") == {"期待値 10、分散 8", "期待値 10、分散 10", f"期待値 10、分散 {f(Bn.std(), 2)}", "期待値 0.2、分散 0.16", "期待値 10、分散 40"}, "誤答の由来")
G = stats.geom(.2)
E.choice("s04p08c", f"確率 {f(G.pmf(3), 3)}、期待値 {f(G.mean(), 0)}", "scipy の geom（試行の回数）")
E.check("s04p08c", opts("s04p08c") == {"確率 0.128、期待値 5", f"確率 {f(.2 ** 3, 3)}、期待値 5", f"確率 {f(.8 ** 3, 3)}、期待値 5", "確率 0.128、期待値 4", f"確率 {f(3 * .128, 3)}、期待値 5"}, "誤答の由来")
# ---- 大問9
E.check("s04p09a", ans("s04p09a") == f"平均 {f(400 * .5, 0)}、標準偏差 {f(sqrt(400 * .25), 0)}", "np と √(npq)（「標準偏差 100」と区別するため完全一致で確認）")
E.check("s04p09a", opts("s04p09a") == {"平均 200、標準偏差 10", "平均 200、標準偏差 100", "平均 200、標準偏差 20", "平均 0.5、標準偏差 0.025", f"平均 200、標準偏差 {f(sqrt(200), 1)}"}, "誤答の由来")
E.choice("s04p09b", f(stats.norm.sf(1.95), 4), f"正規近似（正確な値 {stats.poisson(100).sf(119):.4f}）")
E.check("s04p09b", opts("s04p09b") == {f(stats.norm.sf(1.95), 4), f(stats.norm.sf(2.0), 4), f(stats.norm.sf(2.05), 4), f(1 - 0.0256, 4), f(2 * 0.0256, 4)}, "表の値と誤答の由来")
pop = rng.normal(50, 10, 1000); pop = (pop - pop.mean()) / pop.std() * 10 + 50
sm = np.array([rng.choice(pop, 100, replace=False).mean() for _ in range(40000)])
E.choice("s04p09c", f(10 / 10 * sqrt(900 / 999), 3), f"非復元抽出のシミュレーション {sm.std():.4f}")
E.check("s04p09c", abs(sm.std() - 0.949) < 0.01 and opts("s04p09c") == {"0.949", "1.000", f(900 / 999, 3), "0.100", f(0.1 * sqrt(900 / 999), 3)}, "誤答の由来")
# ---- 大問10
E.choice("s04p10a", iv(.2 - 1.96 * .04, .2 + 1.96 * .04, 3), "比率の区間")
E.check("s04p10a", opts("s04p10a") == {iv(.2 - k * w, .2 + k * w, 3) for k, w in ((1.96, .04), (1.645, .04), (1.96, .05), (2.576, .04), (1.96, .0016))}, "誤答の由来")
E.check("s04p10b", ans("s04p10b").startswith("期待値が $\\mu$ に等しく"), "不偏性と一致性")
dev = [np.abs(rng.normal(3, 2, (2000, n)).mean(1) - 3).mean() for n in (10, 1000)]
E.check("s04p10b", dev[1] < dev[0] / 5, f"n を大きくするとずれが小さくなる {dev}")
sp = (7 * 9 + 7 * 7) / 14; se = sqrt(sp * (1 / 8 + 1 / 8))
E.check("s04p10c", abs(stats.t.isf(.025, 14) - 2.145) < 5e-4 and abs(stats.t.isf(.025, 16) - 2.120) < 5e-4 and abs(stats.t.isf(.05, 14) - 1.761) < 5e-4, "表の値")
E.choice("s04p10c", iv(4 - 2.145 * se, 4 + 2.145 * se, 2), f"プールした分散 {sp}")
E.check("s04p10c", opts("s04p10c") == {iv(4 - k * w, 4 + k * w, 2) for k, w in ((2.145, se), (1.96, se), (2.120, se), (1.761, se), (2.145, sqrt(8)))}, "誤答の由来")
E.choice("s04p10d", f"約 {f(10 * 2.576 / 1.96, 1)}", "幅は z に比例")
E.check("s04p10d", opts("s04p10d") == {f"約 {f(10 * 2.576 / 1.96, 1)}", f"約 {f(10 * 1.96 / 2.576, 1)}", "10", f"約 {f(10 * 2.576 / 1.645, 1)}", f"約 {f(10 * .99 / .95, 1)}"}, "誤答の由来")
# ---- 大問11
E.check("s04p11a", ans("s04p11a") == "帰無仮説 $\\mu=50$、対立仮説 $\\mu>50$", "右片側")
pz = 2 * stats.norm.sf(2.2)
E.choice("s04p11b", f"p 値 {f(pz, 4)} で、有意水準 5% では棄却し、1% では棄却しない", f"p={pz:.5f}")
E.check("s04p11b", abs(stats.norm.sf(2.2) - .0139) < 5e-5, "表の値")
E.check("s04p11c", not (2.1 <= 2 <= 5.3) and (2.1 <= 3 <= 5.3) and ans("s04p11c").startswith("「$\\mu=2$」は棄却され"), "区間の外は棄却")
# ---- 大問12
tp = 1.5 / (2 / sqrt(10))
E.choice("s04p12a", f"t ≈ {f(tp, 2)} で、2.262 より大きいので棄却する", f"t={tp:.4f}, p={2 * stats.t.sf(tp, 9):.4f}")
E.check("s04p12a", abs(stats.t.isf(.025, 9) - 2.262) < 5e-4 and f(1.5 / 2, 2) == "0.75" and f(1.5 / .2, 2) == "7.50" and f(1.5 / (4 / sqrt(10)), 2) == "1.19", "表の値と誤答の由来")
tr = .5 * sqrt(16) / sqrt(.75)
E.choice("s04p12b", f"t ≈ {f(tr, 2)} で、2.120 より大きいので棄却する", f"t={tr:.4f}")
rs = np.array([np.corrcoef(rng.standard_normal((2, 18)))[0, 1] for _ in range(100000)])
ts = rs * 4 / np.sqrt(1 - rs ** 2)
E.check("s04p12b", abs(np.mean(np.abs(ts) > 2.120) - 0.05) < 0.003 and f(.5 * sqrt(18) / sqrt(.75), 2) == "2.45" and f(2 / .75, 2) == "2.67" and f(.5 / sqrt(.75), 2) == "0.58", "ρ=0 のとき t が t(16) にしたがう（シミュレーション）・誤答の由来")
cnt = 0
for _ in range(3000):
    d = rng.poisson(1.5, 200); o = np.array([np.sum(d == k) for k in range(4)] + [np.sum(d >= 4)]); lam = d.mean()
    pk = np.r_[stats.poisson(lam).pmf(range(4)), stats.poisson(lam).sf(3)]; e = 200 * pk
    cnt += ((o - e) ** 2 / e).sum() > 7.815
E.check("s04p12c", abs(cnt / 3000 - 0.05) < 0.015 and ans("s04p12c") == "$\\chi^2_{0.05}(3)=7.815$" and abs(stats.chi2.isf(.05, 3) - 7.815) < 5e-4, f"λ を推定した適合度検定で自由度 3 の点を使うと棄却の割合 {cnt / 3000:.3f}")
# ---- 大問13
E.check("s04p13a", ans("s04p13a").startswith("目的変数の変動のうち、64%"), "決定係数の意味")
F = (360 / 1) / (140 / 20)
E.choice("s04p13b", f(F, 1), f"F={F:.3f}")
E.check("s04p13b", opts("s04p13b") == {f(F, 1), f(360 / 140, 2), f(360 / (140 / 21), 1), "0.72", f(360 / 20, 1)} and abs(stats.f.isf(.05, 1, 20) - 4.351) < 5e-4, "誤答の由来と表の値")
reg = {"北": 200, "中": 230, "南": 180}
E.check("s04p13c", reg["中"] - reg["北"] == 30 and reg["南"] - reg["北"] == -20 and ans("s04p13c").startswith("中の平均は北より 30 万円多く"), "基準との差")
# ---- 大問14
E.check("s04p14a", ans("s04p14a") == "4 つの処理の母平均はすべて等しい", "分散分析の帰無仮説")
E.choice("s04p14b", "12", "19 − 3 − 4")
blk = rng.normal(0, 3, 5)[None, :]; trt = np.array([0, 1, 2, 3])[:, None]; Y = 10 + trt + blk + rng.normal(0, 1, (4, 5))
gm = Y.mean(); SST = ((Y - gm) ** 2).sum(); SSA = 5 * ((Y.mean(1) - gm) ** 2).sum(); SSB = 4 * ((Y.mean(0) - gm) ** 2).sum(); SSE = SST - SSA - SSB
resid = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm
E.check("s04p14b", abs(SSE - (resid ** 2).sum()) < 1e-9 and abs(stats.f.isf(.05, 3, 12) - 3.490) < 5e-4, "乱塊法の平方和の分解（誤差は 4×5 の交互作用の形、自由度 3×4＝12）・表の値")
E.done()
