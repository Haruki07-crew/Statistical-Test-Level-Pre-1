"""2級 模擬試験 第10回（g2 set10）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import product
from math import exp, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(10)
rng = np.random.default_rng(20261102)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


E.check("s10p01a", ans("s10p01a").startswith("各階級に入るデータの個数"), "度数分布表から正確にわかるもの")
E.check("s10p01b", ans("s10p01b").startswith("改定の前後で基準と品目が違う"), "基準の改定への注意")
x = rng.normal(50, 10, 20); x2 = x.copy(); x2[x.argmax()] *= 10
q = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
E.check("s10p02a", abs(q(x) - q(x2)) < 1e-9 and np.ptp(x2) != np.ptp(x) and x2.std() != x.std() and ans("s10p02a") == "四分位範囲", "最大値を 10 倍しても IQR は不変")
E.check("s10p02b", sqrt(2 * .5) == 1 and ans("s10p02b") == "0%" and opts("s10p02b") == {"0%", "＋25%", "＋50%", "−25%", "＋100%"}, "倍率の幾何平均 1")
pop = rng.random(200000) < .3; male = rng.random(200000) < .5
E.check("s10p03a", abs(pop[male].mean() - pop[~male].mean()) < .01 and ans("s10p03a").startswith("男性の中での購入者の割合と"), "独立なら男女の購入率が等しい（シミュレーション）")
n_pref = 47; pref = rng.normal(0, 1, n_pref)
sleep = np.concatenate([pref[k] + rng.normal(0, 4, 2000) for k in range(n_pref)]); score = np.concatenate([pref[k] + rng.normal(0, 4, 2000) for k in range(n_pref)])
ms = sleep.reshape(n_pref, -1).mean(1); mt = score.reshape(n_pref, -1).mean(1)
E.check("s10p03b", np.corrcoef(ms, mt)[0, 1] > .8 and np.corrcoef(sleep, score)[0, 1] < .1 and ans("s10p03b").startswith("集計した平均どうしでは"), f"地域の平均の相関 {np.corrcoef(ms, mt)[0, 1]:.2f}、個人の相関 {np.corrcoef(sleep, score)[0, 1]:.3f}")
xx = rng.normal(0, 1, 50); yy = 1 + 2 * xx + rng.normal(0, 1, 50); b, a = np.polyfit(xx, yy, 1); e = yy - a - b * xx
E.check("s10p03c", abs(e.sum()) < 1e-9 and abs((xx * e).sum()) < 1e-9 and ans("s10p03c").startswith("残差の合計は 0 で"), "Σe=0・Σxe=0")
E.check("s10p04a", ans("s10p04a").startswith("前年の 3 月が特別に低かった反動"), "前年同月比の反動")
v = np.r_[np.full(20, 10.0), np.full(20, 20.0)]; ma = np.convolve(v, np.ones(7) / 7, "valid")
E.check("s10p04b", ma[13] == 10 and ma[19] < 20 and ma[20] == 20 and ma[14] > ma[13] and ans("s10p04b").startswith("上がり始めてから少し遅れて"), "後方移動平均は 7 日かけて追いつく")
E.check("s10p05a", ans("s10p05a").startswith("性別・年代の構成は母集団と同じでも"), "割当法は無作為ではない")
E.check("s10p05b", ans("s10p05b").startswith("賛成に誘導する表現で"), "誘導的な質問は非標本誤差")
out = list(product(range(1, 7), repeat=2))
pm = Fr(sum(max(o) <= 4 for o in out), 36)
E.choice("s10p06a", f(float(pm), 3), f"36 通りの総当たり {pm}")
E.check("s10p06a", opts("s10p06a") == {"0.444", f(4 / 6, 3), f(1 - 16 / 36, 3), f(7 / 36, 3), f(1 / 36, 3)} and sum(max(o) == 4 for o in out) == 7, "誤答の由来")
cond = [o for o in out if 6 in o]; pc = Fr(sum(sum(o) % 2 == 0 for o in cond), len(cond))
E.check("s10p06b", ans("s10p06b") == f"{pc.numerator}/{pc.denominator}" and opts("s10p06b") == {"5/11", "1/2", "5/36", "11/36", "6/11"}, f"総当たり {pc}")
X = rng.normal(2, 1, 2_000_000); Y = rng.exponential(3, 2_000_000)
E.check("s10p06c", abs(np.mean(X * Y) - 6) < .02 and ans("s10p06c") == "6", f"シミュレーション {np.mean(X * Y):.3f}")
c = 1 / integrate.quad(lambda t: t, 0, 2)[0]; ex = integrate.quad(lambda t: t * c * t, 0, 2)[0]
E.check("s10p06d", abs(c - .5) < 1e-12 and abs(ex - 4 / 3) < 1e-12 and ans("s10p06d") == "$c=1/2$、$E[X]=4/3$", f"数値積分 c={c}, E={ex:.4f}")
E.check("s10p06d", opts("s10p06d") == {f"$c={cc}$、$E[X]={e}$" for cc, e in (("1/2", "4/3"), ("1", str(Fr(8, 3))), ("1/2", "1"), ("2", str(Fr(2 * 8, 3))), ("1/4", str(Fr(8, 12))))}, "誤答の由来（c=1 なら E=∫x² dx=8/3・c=2 なら 16/3・c=1/4 なら 2/3）")
E.choice("s10p06e", f(exp(-2), 3), f"正確な二項確率 {stats.binom(200, .01).pmf(0):.4f}")
E.check("s10p06e", opts("s10p06e") == {"0.135", "0.010", f(2 * exp(-2), 3), f(1 - exp(-2), 3), "0.990"}, "誤答の由来")
E.choice("s10p06f", f(1 - 2 * .1587, 4), f"正確な値 {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
E.check("s10p06f", opts("s10p06f") == {"0.6826", f(1 - .1587, 4), f(.5 - .1587, 4), "0.1587", "0.9544"}, "誤答の由来")
m = rng.normal(0, 1, (400000, 9)).mean(1) * 3
E.choice("s10p07a", f(2 * .0228, 4), f"シミュレーション {np.mean(np.abs(m) >= 2):.4f}")
E.check("s10p07a", opts("s10p07a") == {"0.0456", "0.0228", f(1 - .0456, 4), "0.25", f(2 * .1587, 4)}, "誤答の由来")
E.choice("s10p07b", f(1 - 2 * .4801, 4), f"正確な二項確率 {stats.binom(400, .5).pmf(200):.4f}")
E.check("s10p07b", abs(stats.norm.sf(.05) - .4801) < 5e-5 and opts("s10p07b") == {"0.0398", "0", "0.5000", "0.4801", f(.5 - .4801, 4)}, "表の値と誤答の由来")
E.choice("s10p07c", "0.1587", f"正確な値 {stats.chi2.sf(60, 50):.4f}")
E.check("s10p07c", stats.chi2(50).var() == 100 and opts("s10p07c") == {"0.1587", f(stats.norm.sf(round(10 / sqrt(50), 2)), 4), f(stats.norm.sf(10 / 100), 4), f(1 - .1587, 4), "0"} and f(stats.norm.sf(1.41), 4) == "0.0793", "分散 2k=100・誤答の由来（分散を k とすると z≈1.41 → 0.0793）")
E.check("s10p08a", ans("s10p08a") == "標本平均 50、標準誤差 2" and abs((53.92 - 46.08) / 2 / 1.96 - 2) < 1e-9, "中央と幅の半分")
se = sqrt(100 / 50 + 25 / 100)
E.choice("s10p08b", iv(6 - 1.96 * se, 6 + 1.96 * se, 2), f"標準誤差 {se}")
E.check("s10p08b", opts("s10p08b") == {iv(6 - 1.96 * se, 6 + 1.96 * se, 2), iv(6 - 1.96 * sqrt(1.5), 6 + 1.96 * sqrt(1.5), 2), iv(6 - 1.96 * (sqrt(2) + .5), 6 + 1.96 * (sqrt(2) + .5), 2), iv(6 - 1.96 * sqrt(125), 6 + 1.96 * sqrt(125), 1), iv(6 - 1.96 * 125 / 150, 6 + 1.96 * 125 / 150, 2)}, "誤答の由来")
E.choice("s10p08c", f((30 + 120) / 400, 3), "合計 ÷ 合計")
E.check("s10p08c", opts("s10p08c") == {"0.375", "0.350", "0.400", "0.300", "0.700"}, "誤答の値")
lo, hi = 135 / 27.49, 135 / 6.262
E.choice("s10p08d", iv(sqrt(lo), sqrt(hi), 2), "母分散の区間の平方根")
E.check("s10p08d", opts("s10p08d") == {iv(sqrt(lo), sqrt(hi), 2), iv(lo, hi, 2), iv(3 - 1.96 * .75, 3 + 1.96 * .75, 2), iv(sqrt(9 / 27.49), sqrt(9 / 6.262), 2), iv(sqrt(144 / 27.49), sqrt(144 / 6.262), 2)}, "誤答の由来")
cov = 0
for _ in range(20):
    s2 = rng.normal(0, 3, (5000, 16)).var(1, ddof=1); cov += np.sum((np.sqrt(15 * s2 / 27.49) <= 3) & (3 <= np.sqrt(15 * s2 / 6.262)))
E.check("s10p08d", abs(cov / 100000 - .95) < .005, f"σ=3 での被覆率 {cov / 100000:.4f}")
E.check("s10p09a", ans("s10p09a") == "0.8", "1 − β")
E.check("s10p09b", ans("s10p09b").startswith("母分散が既知なら z 検定"), "z と t の使い分け")
E.check("s10p09c", ans("s10p09c") == "$|Z|>1.645$" and abs(2 * stats.norm.sf(1.645) - .10) < 1e-3, "両側 10%")
E.check("s10p09d", ans("s10p09d") == "6" and stats.chi2_contingency(rng.integers(5, 20, (3, 4))).dof == 6, "(3−1)(4−1)")
tab = np.array([[30, 10, 20], [10, 30, 20]]); r = stats.chi2_contingency(tab, correction=False); contrib = (tab - r[3]) ** 2 / r[3]
E.check("s10p09e", abs(r[0] - 20) < 1e-9 and np.allclose(contrib[:, 2], 0) and ans("s10p09e").startswith("χ² ＝ 20 で、違いは案 1 と案 2"), f"セルの寄与 {contrib.tolist()}")
E.check("s10p09f", abs(stats.f.isf(.05, 10, 12) - 2.753) < 5e-4 and abs(stats.f.isf(.05, 12, 10) - 2.913) < 5e-4 and 2.5 < 2.753 and ans("s10p09f").startswith("F ＝ 2.5 で、$F_{0.05}(10,12)=2.753$ より小さい"), f"F=2.5, p={stats.f.sf(2.5, 10, 12):.4f}")
Xd = rng.normal(0, 1, (40, 4)); yd = Xd @ [1, 1, 0, 0] + rng.normal(0, 1, 40); res = stats.linregress(Xd[:, 0], yd)
E.check("s10p10a", ans("s10p10a") == str(40 - 4 - 1), "n − p − 1")
Z = rng.normal(0, 1, (5000, 3)); x1 = Z @ [1, 1, 1] / sqrt(3) * 3 + rng.normal(0, 1, 5000)
A = np.c_[np.ones(5000), Z]; e1 = x1 - A @ np.linalg.lstsq(A, x1, rcond=None)[0]; R2 = 1 - e1 @ e1 / ((x1 - x1.mean()) @ (x1 - x1.mean()))
E.check("s10p10b", abs(R2 - .9) < .01 and ans("s10p10b") == "10" and opts("s10p10b") == {"10", "0.1", "0.9", f(1 / .9, 2), "9"}, f"R²≈{R2:.3f} のデータ（VIF ≈ {1 / (1 - R2):.1f}）・誤答の由来")
xs = rng.uniform(10, 30, 40); Sxx = ((xs - xs.mean()) ** 2).sum(); w = lambda x0: sqrt(1 + 1 / 40 + (x0 - xs.mean()) ** 2 / Sxx)
E.check("s10p10c", w(xs.mean()) < min(w(10), w(30), w(0)) and ans("s10p10c") == "20", "予測区間の幅は x̄ で最小")
E.check("s10p11a", ans("s10p11a").startswith("区画に 1〜20 の番号を付け、乱数で"), "乱数による割り付け")
blk = np.repeat(rng.normal(0, 3, 10), 2).reshape(10, 2); trt = np.array([0, 1.0]); Y = 10 + blk + trt + rng.normal(0, 1, (10, 2))
gm = Y.mean(); SSA = 10 * ((Y.mean(0) - gm) ** 2).sum(); SSB = 2 * ((Y.mean(1) - gm) ** 2).sum(); SST = ((Y - gm) ** 2).sum()
F_one = (SSA / 1) / ((SST - SSA) / 18); F_blk = (SSA / 1) / ((SST - SSA - SSB) / 9)
E.check("s10p11b", F_blk > F_one and ans("s10p11b").startswith("ブロックの違いによる変動が誤差から取り除かれ"), f"場所の違いが大きいデータで F 一元配置 {F_one:.2f} → 乱塊法 {F_blk:.2f}")
E.done()
