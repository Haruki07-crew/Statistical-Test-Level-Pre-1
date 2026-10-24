"""2級 模擬試験 第9回（g2 set09）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from math import atanh, sqrt, tanh
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(9)
rng = np.random.default_rng(20261101)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


E.check("s09p01a", ans("s09p01a") == "(ア) 質的、(イ) 量的・離散、(ウ) 量的・連続、(エ) 質的（順序）", "変数の種類")
E.check("s09p01b", ans("s09p01b") == f"{6 * 20 // 10} 人" and opts("s09p01b") == {"12 人", "6 人", "3 人", "120 人", "26 人"}, "高さ × 幅の比")
E.choice("s09p02a", f"約 {f(20000 / (100 + 50), 0)} 円", "総額 ÷ 総個数")
E.check("s09p02a", abs(stats.hmean([100, 200]) - 20000 / 150) < 1e-9 and opts("s09p02a") == {"約 133 円", "150 円", f"約 {f(sqrt(20000), 0)} 円", "300 円", f"約 {f(20000 / 150 / 2, 0)} 円"}, "調和平均と一致・誤答の由来")
x = 60 + 8 * (65 - 50) / 10
E.choice("s09p02b", f"{f(x, 0)} 点", "偏差値の式を解く")
E.check("s09p02b", opts("s09p02b") == {"72 点", "65 点", f"{60 + 8} 点", f"{60 + 15 * 8} 点", f"{60 - 8} 点"} and 50 + 10 * (52 - 60) / 8 == 40, "誤答の由来")
E.check("s09p03a", ans("s09p03a") == f"{60 * 100 // 80}%" and opts("s09p03a") == {"75%", "60%", f"{60 * 100 // 250}%", f"{80 * 100 // 250}%", "40%"}, "列パーセント 60/80")
R = np.array([[1, .2, .5], [.2, 1, -.4], [.5, -.4, 1]]); P = np.linalg.inv(R); pc = -P[0, 1] / sqrt(P[0, 0] * P[1, 1])
E.choice("s09p03b", f(pc, 2), f"精度行列から {pc:.4f}")
E.check("s09p03b", opts("s09p03b") == {f(pc, 2), "0.20", "0.40", f(.4 / (.75 * .84), 2), "0"}, "誤答の由来")
n = 5; Sx, Sy, Sxy, Sxx = 20, 50, 230, 90
b = (Sxy - Sx * Sy / n) / (Sxx - Sx ** 2 / n); a = Sy / n - b * Sx / n
xs = np.array([2, 3, 4, 5, 6.]); ys = -2 + 3 * xs + np.array([1, -2, 0, 2, -1.])
E.check("s09p04a", (xs.sum(), ys.sum(), xs @ ys, xs @ xs) == (20, 50, 230, 90) and np.allclose(np.polyfit(xs, ys, 1), [3, -2]) and ans("s09p04a") == "$\\hat y=-2+3x$", f"合計が問題文どおりのデータで当てはめ b={b}, a={a}")
E.check("s09p04a", f(Sxy / Sxx, 2) == "2.56" and opts("s09p04a") == {"$\\hat y=-2+3x$", "$\\hat y=10+2.56x$", "$\\hat y=3-2x$", "$\\hat y=10+3x$", "$\\hat y=2+2x$"}, "誤答の由来")
E.check("s09p04b", (70 - 60) / 100 == .10 and (35 - 40) / 100 == -.05 and ans("s09p04b") == "A は ＋10 ポイント、B は −5 ポイント", "寄与度")
real = 1.03 / 1.02 - 1
E.check("s09p04c", ans("s09p04c") == f"＋{f(real * 100, 1)}%" and abs(106.08 / 104 - 1.02) < 1e-12, f"実質 {real:.5f}")
E.check("s09p04c", opts("s09p04c") == {"＋1.0%", f"{f((1.02 / 1.03 - 1) * 100, 1)}%", "＋5.0%", "＋3.0%", f"＋{f(3 - 2.08, 1)}%"}, "誤答の由来")
for N in (1_000_000, 100_000_000):
    pass
fpc = lambda N: sqrt((N - 1000) / (N - 1))
E.check("s09p05a", abs(fpc(10 ** 6) - fpc(10 ** 8)) < 0.001 and ans("s09p05a").startswith("母集団が標本よりずっと大きければ"), f"有限母集団修正 100 万人 {fpc(10 ** 6):.5f}・1 億人 {fpc(10 ** 8):.6f}")
E.check("s09p05b", ans("s09p05b").startswith("標本誤差はないが"), "全数調査の非標本誤差")
lots = "AAAALLLLLL"; seq = list(permutations(range(10), 2))
E.choice("s09p06a", f(sum(lots[i] == "A" and lots[j] == "L" for i, j in seq) / len(seq), 3), "90 通りの総当たり")
E.check("s09p06a", opts("s09p06a") == {"0.267", f(.4 * .6, 2), "0.40", "0.60", f(6 / 9, 3)}, "誤答の由来")
E.check("s09p06b", ans("s09p06b").startswith("$P(A\\mid B)=0$") and opts("s09p06b") >= {"$P(A\\cap B)=0.1$ である", "$P(A\\cup B)=0.6$ である"}, "排反なら独立でない")
X = rng.normal(4, 3, 2_000_000)
E.choice("s09p06c", f(9 + 16, 0), f"シミュレーション {np.mean(X ** 2):.3f}")
E.check("s09p06c", abs(np.mean(X ** 2) - 25) < .05 and opts("s09p06c") == {"25", "7", "16", "13", "9"}, "誤答の由来")
T = rng.normal(50, 2, (400000, 5)).sum(1)
E.check("s09p06d", abs(T.var() - 20) < .2 and ans("s09p06d") == f"分散 20、標準偏差 約 {f(sqrt(20), 2)}", f"シミュレーション 分散 {T.var():.3f}")
E.choice("s09p06e", f(stats.uniform(0, 10).cdf(5) - stats.uniform(0, 10).cdf(2), 1), "scipy の uniform")
E.check("s09p06e", opts("s09p06e") == {"0.3", "0.5", "0.2", "0.8", "3"}, "誤答の値")
N_ = stats.norm(50, 10); iqr = N_.ppf(.75) - N_.ppf(.25)
E.choice("s09p06f", f(iqr, 1), f"scipy の norm で四分位範囲 {iqr:.4f}")
E.check("s09p06f", abs(stats.norm.isf(.25) - .674) < 5e-4 and opts("s09p06f") == {"13.5", f(N_.ppf(.75) - 50, 1), "20.0", "25.0", "10.0"}, "表の値と誤答の由来")
E.check("s09p07a", ans("s09p07a").startswith("前者は自由度 $n$、後者は自由度 $n-1$"), "自由度 n と n−1")
sm = rng.normal(0, 1, (300000, 6)); q1 = (sm ** 2).sum(1); q2 = ((sm - sm.mean(1, keepdims=True)) ** 2).sum(1)
E.check("s09p07a", abs(q1.mean() - 6) < .03 and abs(q2.mean() - 5) < .03, f"シミュレーションの平均 {q1.mean():.3f}（自由度 6）、{q2.mean():.3f}（自由度 5）")
ph = rng.binomial(400, .2, 1_000_000) / 400
E.choice("s09p07b", f(stats.norm.sf(2), 4), f"正確な二項確率 {np.mean(ph > .24):.4f}")
E.check("s09p07b", opts("s09p07b") == {"0.0228", "0.04", f(1 - .0228, 4), f(2 * .0228, 4), "0.24"}, "誤答の由来")
tot = rng.poisson(4, (300000, 100)).sum(1)
E.choice("s09p07c", f(stats.norm.sf(1.5), 4), f"シミュレーション（正確な値）{np.mean(tot > 430):.4f}")
E.check("s09p07c", opts("s09p07c") == {"0.0668", "0.0000", f(stats.norm.sf(30 / 200), 4), f(1 - .0668, 4), f(2 * .0668, 4)}, "誤答の由来")
cover = np.mean([np.any(True) for _ in range(1)])
lo = rng.normal(0, 1, (100000, 25)).mean(1)
E.check("s09p08a", abs(np.mean(np.abs(lo) * 5 > 1.96) * 20 - 1) < 0.05 and ans("s09p08a") == "1 回", f"20 回あたり外れる回数 {np.mean(np.abs(lo) * 5 > 1.96) * 20:.3f}")
s2n = rng.normal(0, 2, (300000, 5)).var(1)
E.check("s09p08b", abs(s2n.mean() - 4 * 4 / 5) < .02 and ans("s09p08b").startswith("$\\dfrac{n-1}{n}\\sigma^2$"), f"n=5、σ²=4 で n 割りの平均 {s2n.mean():.3f}（3.2）")
z0 = atanh(.8)
E.choice("s09p08c", iv(tanh(z0 - .49), tanh(z0 + .49), 2), "z 変換の区間")
E.check("s09p08c", abs(z0 - 1.099) < 5e-4 and opts("s09p08c") == {iv(tanh(z0 - .49), tanh(z0 + .49), 2), iv(.8 - .49, .8 + .49, 2), iv(z0 - .49, z0 + .49, 2), iv(tanh(z0 - 1.96 / sqrt(19)), tanh(z0 + 1.96 / sqrt(19)), 2), iv(tanh(z0 - 1.645 * .25), tanh(z0 + 1.645 * .25), 2)}, "誤答の由来")
E.check("s09p08c", all(abs(tanh(a_) - b_) < 5e-4 for a_, b_ in ((.609, .543), (1.589, .920), (.649, .571), (1.549, .914), (.688, .597), (1.510, .907))), "問題文の tanh の値")
hw = 1.96 * sqrt(400 / 499)
pop = rng.normal(70, 10, 500); pop = (pop - pop.mean()) / pop.std() * 10 + 70
means = np.array([rng.choice(pop, 100, replace=False).mean() for _ in range(30000)])
E.choice("s09p08d", f(hw, 2), f"非復元抽出の標本平均の標準偏差 {means.std():.4f}（理論 {sqrt(400 / 499):.4f}）")
E.check("s09p08d", abs(means.std() - sqrt(400 / 499) * 10 / 10 * sqrt(500 / 499)) < .02 and opts("s09p08d") == {f(hw, 2), "1.96", f(1.96 * 400 / 499, 2), f(sqrt(400 / 499), 2), f(1.96 * 10 / 100 * sqrt(400 / 499), 2)}, "誤答の由来")
rej = 0
for _ in range(20000):
    smp = rng.normal(0, 1, 10); rej += abs(smp.mean() / (smp.std(ddof=1) / sqrt(10))) > stats.t.isf(.025, 9)
E.check("s09p09a", abs(rej / 20000 * 100 - 5) < .5 and ans("s09p09a") == "5 回", f"100 回あたり {rej / 20000 * 100:.2f} 回")
t = (48 - 50) / (4 / 4)
E.check("s09p09b", t == -2 and t < -stats.t.isf(.05, 15) and abs(stats.t.isf(.05, 15) - 1.753) < 5e-4 and ans("s09p09b").startswith("t ＝ −2.0 で、−1.753 より小さい"), f"t={t}, p={stats.t.cdf(t, 15):.4f}")
E.check("s09p09c", abs(stats.t.isf(.01, 9) - 2.821) < 5e-4 and ans("s09p09c") == "$t>2.821$", "t_{0.01}(9)")
E.check("s09p09d", ans("s09p09d") == "2", "3 − 1")
E.check("s09p09e", abs(stats.t.isf(.025, 16) - 2.120) < 5e-4 and 2.0 < 2.120 and ans("s09p09e").startswith("自由度 16 の点 2.120 より小さいので棄却しない"), "2.0 < 2.120")
def zdiff(n, p1=.5, p2=.4):
    p = (p1 + p2) / 2; return (p1 - p2) / sqrt(p * (1 - p) * 2 / n)
E.check("s09p09f", abs(zdiff(400) / zdiff(100) - 2) < 1e-12 and ans("s09p09f").startswith("約 3.0"), "比率が同じなら z は √n に比例")
xx = rng.normal(0, 1, (60, 3)); yy = xx[:, 0] + rng.normal(0, 1, 60)
def r2(X):
    A = np.c_[np.ones(60), X]; e = yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]; return 1 - e @ e / ((yy - yy.mean()) @ (yy - yy.mean()))
adj = lambda R, p: 1 - (1 - R) * 59 / (60 - p - 1)
noise = rng.normal(0, 1, (60, 1))
E.check("s09p10a", r2(np.c_[xx[:, :1], noise]) >= r2(xx[:, :1]) and ans("s09p10a").startswith("決定係数は減らないが"), f"無関係な変数を加えると R² {r2(xx[:, :1]):.4f} → {r2(np.c_[xx[:, :1], noise]):.4f}、調整済み {adj(r2(xx[:, :1]), 1):.4f} → {adj(r2(np.c_[xx[:, :1], noise]), 2):.4f}")
E.check("s09p10b", ans("s09p10b") == str(10 + 2 * 3 + 5) and opts("s09p10b") == {"21", "16", "15", "26", "18"}, "代入と誤答の値")
E.check("s09p10c", -0.3 < 0 < 1.5 and ans("s09p10c").startswith("区間が 0 を含むので"), "区間が 0 を含む")
E.check("s09p11a", ans("s09p11a").startswith("各群のデータが互いに独立で"), "分散分析の前提")
Fs = []
for _ in range(20000):
    g = rng.normal(0, 1, (4, 6)); Fs.append(stats.f_oneway(*g).statistic)
E.check("s09p11b", abs(np.median(Fs) - stats.f.median(3, 20)) < .03 and abs(np.mean(Fs) - 20 / 18) < .05 and ans("s09p11b").startswith("1 の前後の値"), f"帰無仮説のもとの F の平均 {np.mean(Fs):.3f}（理論 20/18）")
E.done()
