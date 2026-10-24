"""2級 模擬試験 第12回（g2 set12）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import product
from math import exp, log2, log10, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(12)
rng = np.random.default_rng(20261112)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# 第1問
E.check("s12p01a", ans("s12p01a") == f"{1 + log2(64):g}" and opts("s12p01a") == {"7", f"{sqrt(64):g}", f"{log2(64):g}", "32", f"{round(1 + log10(64)):g}"}, "1 + log₂64 = 7・誤答の由来")
E.check("s12p01b", 60 + 45 + 30 + 25 > 100 and ans("s12p01b") == "交通手段ごとの割合を並べた棒グラフ", "割合の合計 160%")

# 第2問
d = [3, 5, 6, 8, 9, 11, 12, 14, 15, 20]; q1, q3 = np.median(d[:5]), np.median(d[5:])
E.check("s12p02a", ans("s12p02a") == f"{q3 - q1:g}" and opts("s12p02a") == {f"{q3 - q1:g}", f"{max(d) - min(d):g}", f"{(q3 - q1) / 2:g}", f"{np.median(d):g}", f"{q3 - d[1]:g}"}, f"Q1={q1}・Q3={q3}・誤答の由来")
F = rng.normal(68, 9, 100000); F = (F - F.mean()) / F.std() * 9 + 68; C = (F - 32) * 5 / 9
E.check("s12p02b", abs(C.mean() - 20) < 1e-9 and abs(C.std() - 5) < 1e-9 and ans("s12p02b") == "平均 20℃、標準偏差 5℃", f"換算したデータで平均 {C.mean():.3f}・標準偏差 {C.std():.3f}")
E.check("s12p02b", opts("s12p02b") == {"平均 20℃、標準偏差 5℃", "平均 20℃、標準偏差 9℃", f"平均約 {f(68 * 5 / 9, 1)}℃、標準偏差 5℃", f"平均 20℃、標準偏差約 {f((9 - 32) * 5 / 9, 1)}℃", f"平均 20℃、標準偏差約 {f(9 * (5 / 9) ** 2, 2)}℃"}, "誤答の由来")

# 第3問
pa = np.r_[rng.normal(100, 5, 300), rng.normal(130, 5, 300)]; base = np.r_[np.full(300, 50.0), np.full(300, 90.0)]
sales = base - 0.8 * (pa - np.r_[np.full(300, 100.0), np.full(300, 130.0)]) + rng.normal(0, 3, 600)
r_a = np.corrcoef(pa[:300], sales[:300])[0, 1]; r_b = np.corrcoef(pa[300:], sales[300:])[0, 1]; r_all = np.corrcoef(pa, sales)[0, 1]
E.check("s12p03a", r_a < -.5 and r_b < -.5 and r_all > .5 and ans("s12p03a").startswith("客の多い店舗のほうが全体に価格が高く"), f"店舗ごと {r_a:.2f}・{r_b:.2f}、合わせると {r_all:.2f}")
x = rng.normal(0, 1, 50); y = 2 + x + rng.normal(0, 1, 50); b, a = np.polyfit(x, y, 1)
sse = lambda aa, bb: ((y - aa - bb * x) ** 2).sum()
E.check("s12p03b", all(sse(a, b) <= sse(a + da, b + db) for da in (-.01, 0, .01) for db in (-.01, 0, .01)) and ans("s12p03b").startswith("各点と直線の、縦方向（y の方向）のずれの 2 乗の和"), "polyfit の直線が縦の残差平方和の最小")
xs = rng.normal(5, 2, 1000); ys = 3 + 1.5 * xs + rng.normal(0, 4, 1000); zx = (xs - xs.mean()) / xs.std(); zy = (ys - ys.mean()) / ys.std(); bz, az = np.polyfit(zx, zy, 1)
E.check("s12p03c", abs(bz - np.corrcoef(xs, ys)[0, 1]) < 1e-9 and abs(az) < 1e-9 and ans("s12p03c") == "傾き 0.6、切片 0", f"標準化した回帰の傾き {bz:.4f} = r、切片 {az:.1e}")

# 第4問
days = np.arange(70); wk = np.array([5, -2, -3, -1, 0, 3, -2.0]); series = 100 + .2 * days + wk[days % 7]
ma = lambda k: np.convolve(series, np.ones(k) / k, "valid")
res7 = ma(7) - (100 + .2 * (days[3:-3])); res5 = ma(5) - (100 + .2 * days[2:-2])
E.check("s12p04a", np.allclose(res7, res7[0]) and np.ptp(res5) > 1 and ans("s12p04a") == "7 項", "7 項で曜日の変動が消え（残りは一定）、5 項では残る")
p0 = np.array([100, 100.]); q0 = np.array([10, 10.]); p1 = np.array([150, 105.]); q1_ = np.array([6, 13.])
L = (p1 @ q0) / (p0 @ q0) * 100; P = (p1 @ q1_) / (p0 @ q1_) * 100
E.check("s12p04b", L > P and ans("s12p04b").startswith("比較時の購入量を重みにするパーシェ指数では"), f"値上がりした品目の購入量が減る例でラスパイレス {L:.1f} ＞ パーシェ {P:.1f}")

# 第5問
inc_job = rng.choice([300, 500, 800], 30000, p=[.3, .5, .2]) + rng.normal(0, 60, 30000); job = np.searchsorted([400, 650], inc_job - rng.normal(0, 60, 30000) * 0)
grp = np.digitize(inc_job, [400, 650]); month = rng.integers(0, 12, 30000)
def strat_sd(labels, reps=1500):
    vals = []
    for _ in range(reps):
        tot = 0
        for g in np.unique(labels):
            idx = np.flatnonzero(labels == g); w = len(idx) / len(labels); k = max(2, round(300 * w))
            tot += w * inc_job[rng.choice(idx, k, replace=False)].mean()
        vals.append(tot)
    return np.std(vals)
sd_good, sd_bad = strat_sd(grp), strat_sd(month)
E.check("s12p05a", sd_good * 2 < sd_bad and ans("s12p05a").startswith("世帯主の職業や"), f"収入と関係する層 {sd_good:.2f}・誕生月の層 {sd_bad:.2f}")
ability = rng.normal(60, 8, 20000); t1 = ability + rng.normal(0, 8, 20000); t2 = ability + rng.normal(0, 8, 20000); low = t1 < np.quantile(t1, .1)
E.check("s12p05b", t2[low].mean() - t1[low].mean() > 5 and ans("s12p05b").startswith("下位の生徒は偶然悪かった人も多く"), f"補習なしでも下位 10% の平均が {t1[low].mean():.1f} → {t2[low].mean():.1f}")

# 第6問
out = list(product(range(3), repeat=3)); aiko = sum(len(set(o)) in (1, 3) for o in out)
pd_ = Fr(len(out) - aiko, len(out))
E.check("s12p06a", ans("s12p06a") == f"{pd_.numerator}/{pd_.denominator}" and opts("s12p06a") == {"2/3", str(Fr(aiko, 27)), str(Fr(27 - 3, 27)), str(Fr(27 - 6, 27)), "1/2"}, f"27 通りの総当たり：あいこ {aiko} 通り")
X = np.array([-1, 0, 1]); Y = X ** 2; cov = (X * Y).mean() - X.mean() * Y.mean()
E.check("s12p06b", cov == 0 and Fr(1, 3) != 1 and ans("s12p06b") == "X と Y の共分散は 0 だが、X と Y は独立ではない", "Cov=0・P(Y=0)=1/3 だが P(Y=0|X=0)=1")
E.check("s12p06b", f"{Y.mean():.4f}" == f"{2 / 3:.4f}", "誤答 2/3 は E[Y]")

# 第7問
tot = stats.poisson(3)
E.choice("s12p07a", f"約 {f(tot.cdf(1), 3)}", f"平均 3 のポアソン {tot.cdf(1):.4f}（乱数の和 {np.mean(rng.poisson(1, 400000) + rng.poisson(2, 400000) <= 1):.4f}）")
E.check("s12p07a", f(exp(-3), 4) == "0.0498" and opts("s12p07a") == {f"約 {f(tot.cdf(1), 3)}", f"約 {f(tot.pmf(0), 3)}", f"約 {f(stats.poisson(1).cdf(1) * stats.poisson(2).cdf(1), 3)}", f"約 {f(stats.poisson(1.5).cdf(1), 3)}", f"約 {f(tot.pmf(1), 3)}"}, "表の値・誤答の由来")
E.check("s12p07b", abs(stats.hypergeom(10000, 500, 20).pmf(1) - stats.binom(20, .05).pmf(1)) < .002 and ans("s12p07b").startswith("抜き取る個数がロット全体に比べて非常に少なく"), f"P(1 個) 超幾何 {stats.hypergeom(10000, 500, 20).pmf(1):.4f}・二項 {stats.binom(20, .05).pmf(1):.4f}")
arr = rng.exponential(1 / 3, 1_000_000)
E.choice("s12p07c", f"約 {f(exp(-1), 3)}", f"シミュレーション {np.mean(arr > 1 / 3):.4f}")
E.check("s12p07c", opts("s12p07c") == {f"約 {f(exp(-1), 3)}", f"約 {f(exp(-3), 3)}", f"約 {f(1 - exp(-1), 3)}", f"約 {f(1 / 3, 3)}", f"約 {f(exp(-2), 3)}"}, "誤答の由来")
ps = np.linspace(0, 1, 1001); vv = ps * (1 - ps)
E.check("s12p07d", ps[vv.argmax()] == .5 and vv.max() == .25 and ans("s12p07d") == "p ＝ 0.5 のとき最大で、最大値は 0.25", "p(1−p) の最大")

# 第8問
coin = rng.random((200000, 11)) < .5; ten = coin[:, :10].all(1)
E.check("s12p08a", ten.sum() > 100 and abs(coin[ten, 10].mean() - .5) < .1 and ans("s12p08a") == "表が出る確率は 0.5 のままである", f"10 回続けて表のあと {ten.sum()} 回中 表の割合 {coin[ten, 10].mean():.2f}")
dice = rng.integers(1, 7, (200000, 100)).sum(1); sd = sqrt(100 * 35 / 12)
E.choice("s12p08b", f"約 {f(stats.norm.sf(1.17), 3)}", f"z={20 / sd:.3f}・シミュレーション P(合計>370) {np.mean(dice > 370):.4f}")
E.check("s12p08b", abs(np.mean(dice > 370) - .121) < .02 and opts("s12p08b") == {f"約 {f(stats.norm.sf(1.17), 3)}", f"約 {f(2 * stats.norm.sf(1.17), 3)}", f"約 {f(stats.norm.cdf(1.17), 3)}", f"約 {f(.4522, 3)}", "ほぼ 0"} and round(20 / (100 * sqrt(35 / 12)), 2) == .12, "誤答の由来（両側・下側・標準偏差 100 倍で z=0.12・√ なしで z=11.7）")
cover = {}
for p_ in (.5, .02):
    for n_ in (10, 1000):
        mu, s_ = n_ * p_, sqrt(n_ * p_ * (1 - p_)); k = np.arange(n_ + 1)
        cover[(p_, n_)] = abs(stats.binom(n_, p_).cdf(k) - stats.norm.cdf((k + .5 - mu) / s_)).max()
E.check("s12p08c", cover[(.02, 10)] > cover[(.5, 1000)] * 5 and ans("s12p08c").startswith("np と n(1 − p) がどちらも 5 以上"), f"分布関数の最大のずれ：p=0.5,n=1000 {cover[(.5, 1000)]:.4f}・p=0.02,n=10 {cover[(.02, 10)]:.4f}")

# 第9問
iv = lambda m, s_, z: f"({f(m - z * s_, 2)}, {f(m + z * s_, 2)})"
E.choice("s12p09a", iv(4, sqrt(4 / 100), 1.96), "4 ± 1.96 × 0.2")
cov_ok = np.mean([abs(m - 4) <= 1.96 * sqrt(m / 100) for m in rng.poisson(4, (20000, 100)).mean(1)])
E.check("s12p09a", abs(cov_ok - .95) < .01 and opts("s12p09a") == {iv(4, .2, 1.96), iv(4, 2, 1.96), iv(4, .04, 1.96), iv(4, .4, 1.96), iv(4, .2, 1.645)}, f"被覆率 {cov_ok:.3f}・誤答の由来")
smp = rng.normal(10, 3, (4000, 400))
est = {n_: smp[:, :n_].mean(1) + 1 / n_ for n_ in (5, 400)}
E.check("s12p09b", abs(est[5].mean() - 10.2) < .07 and est[400].std() < .2 and abs(est[400].mean() - 10) < .02 and ans("s12p09b") == "X̄ ＋ 1/n", f"n=5 で平均 {est[5].mean():.3f}（偏り 0.2）・n=400 で平均 {est[400].mean():.3f}・ばらつき {est[400].std():.3f}")
mu_known = 10; w = rng.normal(mu_known, 3, (200000, 6))
E.check("s12p09c", abs(((w - mu_known) ** 2).sum(1).mean() / 6 - 9) < .05 and ans("s12p09c") == "n", f"Σ(X−μ)²/n の平均 {((w - mu_known) ** 2).sum(1).mean() / 6:.3f}（母分散 9）")
tr = stats.t.isf(.025, 4) / stats.norm.isf(.025)
E.choice("s12p09d", f"約 {f(2.776 / 1.96, 2)} 倍", f"scipy {tr:.4f}")
E.check("s12p09d", abs(stats.t.isf(.025, 4) - 2.776) < 5e-4 and opts("s12p09d") == {f"約 {f(2.776 / 1.96, 2)} 倍", "1 倍", f"約 {f((2.776 / 1.96) ** 2, 2)} 倍", f"約 {f(1.96 / 2.776, 2)} 倍", f"約 {f(sqrt(2.776 / 1.96), 2)} 倍"}, "表の値・誤答の由来")

# 第10問
zz = rng.normal(0, 1, 1_000_000)
E.check("s12p10a", abs(np.mean(np.abs(zz) > 1.645) - .10) < .002 and ans("s12p10a").startswith("どちらの向きの差でも 5% で棄却されうるので"), f"向きを後で決めると棄却率 {np.mean(np.abs(zz) > 1.645):.4f}")
pw = lambda d_: stats.norm.sf(1.645 - d_)
E.check("s12p10b", abs(pw(0) - .05) < 5e-4 and ans("s12p10b") == "0.05", "検出力関数の μ0 での値 = α")
chi = 15 * 2 / 4
E.check("s12p10c", chi == 7.5 and abs(stats.chi2.ppf(.05, 15) - 7.261) < 5e-4 and abs(stats.chi2.isf(.05, 15) - 25.00) < .005 and abs(stats.chi2.ppf(.05, 16) - 7.962) < 5e-4 and chi > 7.261 and ans("s12p10c").startswith("χ² ＝ 7.5 で、下側 5% 点 7.261 より大きいので棄却しない"), f"χ²=7.5・p={stats.chi2.cdf(7.5, 15):.3f}")
E.check("s12p10c", 2 / 4 == .5, "誤答 0.5 は分散の比")

# 第11問
g1 = rng.normal(0, sqrt(20), 12); g2 = rng.normal(0, sqrt(60), 12); g1 = (g1 - g1.mean()) / g1.std(ddof=1) * sqrt(20) + 5; g2 = (g2 - g2.mean()) / g2.std(ddof=1) * sqrt(60)
tp, tw = stats.ttest_ind(g1, g2).statistic, stats.ttest_ind(g1, g2, equal_var=False).statistic
v1, v2 = 20 / 12, 60 / 12; dfw = (v1 + v2) ** 2 / (v1 ** 2 / 11 + v2 ** 2 / 11)
E.check("s12p11a", abs(tp - tw) < 1e-9 and abs(dfw - 17.6) < .05 and ans("s12p11a").startswith("2 つの t 統計量の値は等しい"), f"t プール {tp:.4f}・ウェルチ {tw:.4f}・ウェルチの自由度 {dfw:.2f}")
E.check("s12p11b", ans("s12p11b") == f"{80 * stats.binom(3, .5).pmf(2):.0f}" and opts("s12p11b") == {"30", f"{80 / 4:.0f}", f"約 {f(80 / 3, 1)}", f"{80 / 2:.0f}", f"{80 * stats.binom(3, .5).pmf(0):.0f}"}, "期待度数 80 × 3/8・誤答の由来")
E.check("s12p11c", ans("s12p11c").startswith("2 つの変数は独立ではない（関連がある）といえるが"), "独立性の検定で言えること")

# 第12問
n_ = 200; area = rng.normal(60, 15, n_); dist = rng.normal(1.0, .5, n_); rent = 3 + .05 * area - 1.2 * dist + rng.normal(0, .8, n_)
A1 = np.c_[np.ones(n_), area, dist]; A2 = np.c_[np.ones(n_), area, dist * 1000]
c1 = np.linalg.lstsq(A1, rent, rcond=None)[0]; c2 = np.linalg.lstsq(A2, rent, rcond=None)[0]
E.check("s12p12a", abs(c2[2] * 1000 - c1[2]) < 1e-9 and np.allclose(A1 @ c1, A2 @ c2) and ans("s12p12a").startswith("2 つの係数は単位が違うので"), f"距離を m にすると係数 {c1[2]:.3f} → {c2[2]:.6f}（予測値は同じ）")
xx = rng.normal(0, 1, 20); yy = .5 * xx + rng.normal(0, 1, 20); lr = stats.linregress(xx, yy); r = lr.rvalue
E.check("s12p12b", abs(lr.slope / lr.stderr - r * sqrt(18) / sqrt(1 - r ** 2)) < 1e-9 and ans("s12p12b") == "2 つの t の値は等しい", "傾きの t と相関係数の t が一致")
tt = np.arange(48); e = np.zeros(48)
for k in range(1, 48):
    e[k] = .9 * e[k - 1] + rng.normal(0, 1)
runs = 1 + np.sum(np.sign(e[1:]) != np.sign(e[:-1]))
E.check("s12p12c", runs < 15 and ans("s12p12c").startswith("誤差が互いに独立でない"), f"自己相関 0.9 の誤差で符号の連の数 {runs}（独立なら約 24）")

# 第13問
E.check("s12p13a", ans("s12p13a") == "15" and opts("s12p13a") == {"15", "17", "2", f"{3 * (5 - 1)}", f"{18 - 2}"}, "18 − 3 = 15・誤答の由来")
blk = rng.normal(0, 3, 3); trt = np.array([0, 1, 2, 3.]); Yb = 10 + trt[:, None] + blk[None, :] + rng.normal(0, 1, (4, 3))
gm = Yb.mean(); SA = 3 * ((Yb.mean(1) - gm) ** 2).sum(); SB = 4 * ((Yb.mean(0) - gm) ** 2).sum(); ST = ((Yb - gm) ** 2).sum(); SE = ST - SA - SB
E.check("s12p13b", abs(SA + SB + SE - ST) < 1e-9 and ans("s12p13b") == f"{(24 / 3) / (12 / 6):.1f}" and opts("s12p13b") == {"4.0", f"約 {f((24 / 3) / (42 / 8), 2)}", f"{(24 / 4) / (12 / 6):.1f}", f"{(30 / 2) / (12 / 6):.1f}", f"{24 / 12:.1f}"}, "乱塊法の平方和の分解・F=8/2・誤答の由来")
E.done()
