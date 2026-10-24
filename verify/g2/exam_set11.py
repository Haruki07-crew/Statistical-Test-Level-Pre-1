"""2級 模擬試験 第11回（g2 set11）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from math import ceil, comb, exp, log, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(11)
rng = np.random.default_rng(20261111)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# 第1問 グラフと累積相対度数
E.check("s11p01a", abs((100 - 96) / 96 - 0.042) < 5e-4 and ans("s11p01a").startswith("縦軸が 0 から始まっていないので変化が大きく見えている"), "増加率 4.2%")
cum = {10: .15, 20: .40, 30: .70, 40: .90, 50: 1.0}
# 階級の中で均等に散らばる（一様）とした累積分布を細かく作り、0.5 になる位置を探す
xs = np.linspace(0, 50, 500001); cx = np.interp(xs, [0, 10, 20, 30, 40, 50], [0, .15, .40, .70, .90, 1.0]); med = xs[np.searchsorted(cx, .5)]
E.choice("s11p01b", f"約 {f(med, 1)} 分", f"補間した累積分布で 0.5 になる位置 {med:.3f}")
E.check("s11p01b", opts("s11p01b") == {f"約 {f(med, 1)} 分", "25.0 分", "20.0 分", f"約 {f(20 + 10 * (.70 - .5) / .30, 1)} 分", "30.0 分"}, "誤答の由来（階級値・下の端・逆向きの補間・上の端）")

# 第2問 散らばり
d = np.array([2, 4, 4, 4, 5, 5, 7, 9.])
E.check("s11p02a", ans("s11p02a") == "2", f"np.std {d.std()}")
E.check("s11p02a", d.std() == 2 and opts("s11p02a") == {f"{d.std():g}", f"{d.var():g}", f"約 {f(d.std(ddof=1), 2)}", f"{np.abs(d - d.mean()).mean():g}", f"{((d - d.mean()) ** 2).sum():g}"}, "誤答の由来（分散・n−1・平均偏差・偏差平方和）")
z = rng.normal(0, 1, 9); x9 = 50 + 6 * (z - z.mean()) / z.std(); x10 = np.r_[x9, 50]
E.choice("s11p02b", f"約 {f(x10.std(), 2)}", f"作ったデータ（平均 {x9.mean():.1f}・標準偏差 {x9.std():.1f}）に 50 を加えた標準偏差 {x10.std():.4f}")
E.check("s11p02b", opts("s11p02b") == {f"約 {f(x10.std(), 2)}", "6.00", f(6 * .9, 2), f"約 {f(6 * sqrt(10 / 9), 2)}", f(x10.var(), 1)}, "誤答の由来")
# 山が 1 つの分布と 2 つの分布で、5 数要約（箱ひげ図）が同じになる例
uni = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10.]); bi = np.array([0, 2.9, 3, 3.1, 5, 6.9, 7, 7.1, 10.])
five = lambda v: (v.min(), np.percentile(v, 25), np.median(v), np.percentile(v, 75), v.max())
E.check("s11p02c", five(uni) == five(bi) and ans("s11p02c") == "分布の山が 1 つか 2 つか", f"5 数要約が同じ {five(uni)}")

# 第3問 相関と回帰
xa = rng.normal(0, 1, 100000); ya = 0.6 * xa + 0.8 * rng.normal(0, 1, 100000); top = xa > np.quantile(xa, .8)
E.check("s11p03a", np.corrcoef(xa, ya)[0, 1] > .55 and np.corrcoef(xa[top], ya[top])[0, 1] < .35 and ans("s11p03a").startswith("合格者は入試の得点が高い人だけに限られ"), f"全体 {np.corrcoef(xa, ya)[0, 1]:.2f}・上位 20% だけ {np.corrcoef(xa[top], ya[top])[0, 1]:.2f}")
fs = rng.multivariate_normal([170, 170], [[36, 18], [18, 36]], 2_000_000); near = np.abs(fs[:, 0] - 182) < .3
E.choice("s11p03b", f"{170 + 0.5 * 6 / 6 * 12:g} cm", f"父 182 cm 前後の息子の平均（シミュレーション）{fs[near, 1].mean():.2f}")
E.check("s11p03b", abs(fs[near, 1].mean() - 176) < .3 and opts("s11p03b") == {"176 cm", "182 cm", "170 cm", f"{170 + .25 * 12:g} cm", f"{170 + 2 * 12:g} cm"}, "誤答の由来")

# 第4問 時系列
q = 100.0
for _ in range(4):
    q *= 1.02
E.choice("s11p04a", f"約 {f(q - 100, 2)}%", f"4 回 1.02 倍 → {q:.4f}")
E.check("s11p04a", opts("s11p04a") == {f"約 {f(q - 100, 2)}%", "8.00%", "2.00%", "0.50%", f"約 {f((1.02 ** 2 - 1) * 100, 2)}%"}, "誤答の由来")
yt = 100 * 1.1 ** np.arange(10); dl = np.diff(np.log(yt))
E.check("s11p04b", np.allclose(dl, log(1.1)) and np.all(np.diff(np.diff(yt)) > 0) and ans("s11p04b") == "右上がりの直線になる", "対数の差が一定（直線）・元の目盛りでは傾きが増える")

# 第5問 調査
pop_int = rng.normal(0, 1, 200000); resp = rng.random(200000) < 0.01 + 0.04 * np.abs(pop_int) ; opin = pop_int > 0.3
E.check("s11p05a", abs(np.abs(pop_int[resp]).mean() - np.abs(pop_int).mean()) > .3 and ans("s11p05a").startswith("回答するかどうかを本人が選ぶ"), f"関心の強さの平均：回答者 {np.abs(pop_int[resp]).mean():.2f}・全体 {np.abs(pop_int).mean():.2f}")
city = rng.normal(4.0, 1, 6000); sub = rng.normal(6.0, 1, 4000); city += 4.0 - city.mean(); sub += 6.0 - sub.mean(); allh = np.r_[city, sub]
E.choice("s11p05b", f"{f(allh.mean(), 1)} 万円", f"市全体の平均 {allh.mean():.3f}")
E.check("s11p05b", opts("s11p05b") == {"4.8 万円", f"{f((4 + 6) / 2, 1)} 万円", f"{f(.4 * 4 + .6 * 6, 1)} 万円", "4.0 万円", "6.0 万円"}, "誤答の由来")

# 第6問 確率と期待値
N = 4_000_000; ill = rng.random(N) < .01
t1 = np.where(ill, rng.random(N) < .9, rng.random(N) < .1); t2 = np.where(ill, rng.random(N) < .9, rng.random(N) < .1)
p2 = ill[t1 & t2].mean(); p1 = .009 / (.009 + .099)
E.choice("s11p06a", "0.45", f"シミュレーション {p2:.4f}・式 {.0081 / (.0081 + .0099):.4f}")
E.check("s11p06a", abs(p2 - .45) < .01 and opts("s11p06a") == {"0.45", f"約 {f(p1, 3)}", f(.9 ** 2, 2), "0.90", f"約 {f(2 * p1, 2)}"}, "誤答の由来")
xv = rng.choice([1, 3, 5], size=400000, p=[.25, .5, .25]); cs = np.linspace(0, 6, 601); mse = [((xv - c) ** 2).mean() for c in cs]
E.check("s11p06b", abs(xv.mean() - 3) < .01 and abs(xv.var() - 2) < .02 and abs(cs[int(np.argmin(mse))] - 3) < .02 and abs(min(mse) - 2) < .02 and ans("s11p06b") == "c ＝ 3 のとき、最小値 2", f"期待値 3・分散 2 の分布で最小の c {cs[int(np.argmin(mse))]:.2f}、最小値 {min(mse):.3f}")
E.check("s11p06b", abs((xv ** 2).mean() - 11) < .05, "c=0 での値は E[X²]=11")
fix = np.mean([sum(i == j for i, j in enumerate(pm)) for pm in permutations(range(4))])
E.check("s11p06c", fix == 1 and ans("s11p06c") == "1 人", f"24 通りの総当たりで自分の傘の人数の平均 {fix}")

# 第7問 確率分布
nmin = min(n for n in range(1, 50) if 1 - .9 ** n >= .5)
E.choice("s11p07a", f"{nmin} 個", f"1 − 0.9^n ≥ 0.5 の最小 n（{1 - .9 ** nmin:.3f}）")
E.check("s11p07a", f(.9 ** 6, 3) == "0.531" and f(.9 ** 7, 3) == "0.478" and opts("s11p07a") == {"7 個", f"{.5 / .1:g} 個", "6 個", f"{1 / .1:g} 個", "8 個"}, "表の値・誤答の由来")
md = stats.expon(scale=10).median()
E.choice("s11p07b", f"約 {f(md, 2)} 分", f"scipy の中央値 {md:.4f}")
E.check("s11p07b", opts("s11p07b") == {f"約 {f(md, 2)} 分", "10 分", "5 分", f"約 {f(10 / log(2), 1)} 分", f"約 {f(.1 * log(2), 4)} 分"}, "誤答の由来")
yy = 2 * rng.normal(50, 10, 2_000_000) + 5
E.choice("s11p07c", "0.1587", f"シミュレーション {np.mean(yy > 125):.4f}")
tab = {0.8: .2119, 1: .1587, 1.25: .1056, 1.41: .0793, 2: .0228}
E.check("s11p07c", all(abs(stats.norm.sf(k) - v) < 5e-5 for k, v in tab.items()) and opts("s11p07c") == {f"{tab[(125 - 105) / 20]:.4f}", f"{tab[(125 - 105) / 10]:.4f}", f"{tab[round((125 - 105) / (10 * sqrt(2)), 2)]:.4f}", f"{tab[(125 - 105) / 25]:.4f}", f"{tab[(125 - 100) / 20]:.4f}"}, "表の値・誤答の由来（SD 10・SD 10√2・SD 25・平均 100）")

# 第8問 標本分布
uv = stats.chi2(3).rvs(1_000_000, random_state=1) + stats.chi2(5).rvs(1_000_000, random_state=2)
E.check("s11p08a", stats.kstest(uv[:20000], "chi2", args=(8,)).pvalue > .01 and abs(uv.mean() - 8) < .02 and ans("s11p08a") == "自由度 8 のカイ二乗分布で、期待値は 8", f"和の平均 {uv.mean():.3f}・分散 {uv.var():.2f}")
mb = rng.exponential(2, (200000, 64)).mean(1)
E.choice("s11p08b", "0.0228", f"シミュレーション {np.mean(mb > 2.5):.4f}・正確な値（ガンマ分布）{stats.gamma(64, scale=2 / 64).sf(2.5):.4f}・正規近似 0.0228")
E.check("s11p08b", abs(np.mean(mb > 2.5) - stats.gamma(64, scale=2 / 64).sf(2.5)) < .002 and abs(stats.gamma(64, scale=2 / 64).sf(2.5) - .0228) < .01 and opts("s11p08b") == {"0.0228", f"{stats.norm.sf(.25):.4f}", f"{2 * .0228:.4f}", f"{1 - .0228:.4f}", f"約 {f(exp(-1.25), 4)}"}, "誤答の由来")
s2 = rng.normal(0, 2, (400000, 11)).var(axis=1, ddof=1)
E.choice("s11p08c", f"{2 * 4 ** 2 / 10:g}", f"シミュレーション V[s²] {s2.var():.3f}")
E.check("s11p08c", abs(s2.var() - 3.2) < .05 and opts("s11p08c") == {"3.2", f"約 {f(32 / 11, 2)}", f"{2 * 10:g}", f"{2 * 4 / 10:g}", f"約 {f(sqrt(3.2), 2)}"}, "誤答の由来")

# 第9問 区間推定
cover = []
for _ in range(20000):
    k = rng.binomial(1000, .4); ph = k / 1000; h = 1.96 * sqrt(ph * (1 - ph) / 1000); cover.append(abs(ph - .4) <= h)
E.check("s11p09a", abs(1.96 * sqrt(.4 * .6 / 1000) - .03) < .001 and abs(np.mean(cover) - .95) < .01 and ans("s11p09a").startswith("本当の支持率は 37〜43% 程度と推定される"), f"±{1.96 * sqrt(.24 / 1000):.4f}・被覆率 {np.mean(cover):.3f}")
nreq = ceil(2 * .25 * (1.96 / .05) ** 2)
E.choice("s11p09b", f"{nreq} 人", f"切り上げ {2 * .25 * (1.96 / .05) ** 2:.2f}")
E.check("s11p09b", 1.96 * sqrt(.5 / nreq) <= .05 < 1.96 * sqrt(.5 / (nreq - 1)) and opts("s11p09b") == {f"{nreq} 人", f"{ceil(.25 * (1.96 / .05) ** 2)} 人", f"{nreq - 1} 人", f"{ceil(.25 * (1.96 / .05) ** 2 * sqrt(2))} 人", f"{2 * nreq} 人"}, "769 で 0.05 以下・768 では超える・誤答の由来")
sv = rng.normal(0, 3, (300000, 5)); ssd = sv.std(axis=1, ddof=1)
E.check("s11p09c", abs(sv.var(axis=1, ddof=1).mean() - 9) < .05 and ssd.mean() < 3 - .1 and ans("s11p09c").startswith("s² は母分散の不偏推定量だが"), f"E[s²]≈{sv.var(axis=1, ddof=1).mean():.3f}・E[s]≈{ssd.mean():.3f} < 3")
ph = .15; se = sqrt(ph * (1 - ph) / 400)
r100 = lambda v: int(Decimal(v).quantize(Decimal(1).scaleb(1), rounding=ROUND_HALF_UP))
lo, hi = 20000 * (ph - 1.96 * se), 20000 * (ph + 1.96 * se)
E.choice("s11p09d", f"約 {r100(lo)} 人から約 {r100(hi)} 人", f"({lo:.0f}, {hi:.0f})")
se_bad = sqrt(ph * (1 - ph) / 20000)
E.check("s11p09d", opts("s11p09d") == {f"約 {r100(lo)} 人から約 {r100(hi)} 人", f"約 {r100(20000 * (ph - 1.645 * se))} 人から約 {r100(20000 * (ph + 1.645 * se))} 人",
                                        f"約 {round(20000 * (ph - 1.96 * se_bad), -2):.0f} 人から約 {round(20000 * (ph + 1.96 * se_bad), -2):.0f} 人", f"{60 - 1.96 * sqrt(400 * ph * (1 - ph)):.0f} 人から {60 + 1.96 * sqrt(400 * ph * (1 - ph)):.0f} 人",
                                        f"{ph - 1.96 * se:.3f} 人から {ph + 1.96 * se:.3f} 人"}, "誤答の由来（1.645・母集団で割る・標本の人数・比率のまま）")

# 第10問 検定の設計
nn = min(n for n in range(1, 200) if stats.norm.sf(1.645 - 5 / (10 / sqrt(n))) >= .9)
E.choice("s11p10a", f"{nn}", f"検出力の式で 0.9 以上になる最小 n（n={nn} で {stats.norm.sf(1.645 - 5 / (10 / sqrt(nn))):.4f}、n={nn - 1} で {stats.norm.sf(1.645 - 5 / (10 / sqrt(nn - 1))):.4f}）")
E.check("s11p10a", abs(stats.norm.isf(.1) - 1.282) < 5e-4 and opts("s11p10a") == {str(nn), str(ceil((1.645 * 2) ** 2)), str(nn - 1), str(ceil(((1.96 + 1.282) * 2) ** 2)), str(round((1.645 + 1.282) * 2))}, "表の値・誤答の由来")
xs_ = np.arange(0, 401); zz = (xs_ - 80) / 8; edge = xs_[zz <= -1.645].max()
E.choice("s11p10b", f"X が {edge} 以下", "z ≤ −1.645 となる最大の整数")
E.check("s11p10b", opts("s11p10b") == {f"X が {edge} 以下", f"X が {edge + 1} 以下", f"X が {int(80 - 1.96 * 8)} 以下", f"X が {int(80 - 1.645 * sqrt(80))} 以下", f"X が {ceil(80 + 1.645 * 8)} 以上"}, "誤答の由来")
alpha = stats.binom(10, .5).sf(8)
E.choice("s11p10c", f"約 {f(alpha, 4)}", f"P(X≥9) = {alpha}（= {comb(10, 9) + comb(10, 10)}/1024）")
E.check("s11p10c", opts("s11p10c") == {f"約 {f(alpha, 4)}", f"約 {f(stats.binom(10, .5).pmf(10), 4)}", f"約 {f(stats.binom(10, .5).pmf(9), 4)}", f"約 {f(stats.binom(10, .5).sf(7), 4)}", "0.05"}, "誤答の由来")
ok = True
for _ in range(20000):
    xb = rng.normal(0, 3); mu0 = rng.normal(0, 3); se_ = 1.0
    ok &= ((xb - mu0) / se_ > 1.645) == (xb - 1.645 * se_ > mu0)
E.check("s11p10d", ok and abs(stats.norm.isf(.05) - 1.645) < 5e-4 and ans("s11p10d").startswith("母平均の 90% 信頼区間（両側）の下限が"), "棄却と 90% 区間の下限の比較が 2 万通りで一致")

# 第11問 クロス表
t2 = np.array([[30, 20], [20, 30]])
E.choice("s11p11a", f"{stats.chi2_contingency(t2, correction=True).statistic:.2f}", "scipy のイェーツの修正つき")
E.check("s11p11a", opts("s11p11a") == {f"{stats.chi2_contingency(t2, correction=True).statistic:.2f}", f"{stats.chi2_contingency(t2, correction=False).statistic:.2f}", f"{4 * (5 - 1) ** 2 / 25:.2f}", f"{(5 - .5) ** 2 / 25:.2f}", f"{stats.chi2.isf(.05, 1):.2f}"}, "誤答の由来")
base = np.array([[20, 30], [30, 20]]) * 1.0; c1 = stats.chi2_contingency(base, correction=False).statistic; c2 = stats.chi2_contingency(2 * base, correction=False).statistic
E.check("s11p11b", abs(c2 - 2 * c1) < 1e-9 and ans("s11p11b").startswith("6.0 になり、3.84 より大きいので棄却される"), f"度数を 2 倍すると {c1:.3f} → {c2:.3f}")

# 第12問 回帰
X3 = rng.normal(0, 1, (2000, 3)); y3 = X3 @ np.array([1, .5, -.8]) + rng.normal(0, 1.2, 2000)
A = np.c_[np.ones(2000), X3]; b3 = np.linalg.lstsq(A, y3, rcond=None)[0]; yh = A @ b3
R2 = 1 - ((y3 - yh) ** 2).sum() / ((y3 - y3.mean()) ** 2).sum()
E.check("s11p12a", abs(np.corrcoef(y3, yh)[0, 1] ** 2 - R2) < 1e-9 and np.corrcoef(y3, yh)[0, 1] > 0 and ans("s11p12a") == "0.80" and opts("s11p12a") == {"0.80", "0.64", f"約 {f(.64 ** 2, 2)}", "±0.80（係数の符号によって決まる）", f"約 {f(.64 / 3, 2)}"}, f"負の係数を含む重回帰でも corr(y, ŷ)² = R²（{R2:.3f}）で相関は正")
season = rng.integers(0, 4, 40); D = np.eye(4)[season]; Xd = np.c_[np.ones(40), D]
E.check("s11p12b", np.linalg.matrix_rank(Xd) == 4 and np.allclose(D.sum(1), 1) and ans("s11p12b").startswith("4 つのダミー変数の和がいつも 1"), "切片と 4 つのダミーで列が 5 本・階数 4（係数が決まらない）")
x1 = rng.normal(0, 1, 30); y1 = 1 + .5 * x1 + rng.normal(0, 1, 30); r = stats.linregress(x1, y1); tt = r.slope / r.stderr
res = y1 - r.intercept - r.slope * x1; Fv = (((r.intercept + r.slope * x1) - y1.mean()) ** 2).sum() / ((res ** 2).sum() / 28)
E.check("s11p12c", abs(Fv - tt ** 2) < 1e-8 and ans("s11p12c") == "9.0", f"単回帰の F={Fv:.4f}、t²={tt ** 2:.4f}")

# 第13問 分散分析
a_, b_, r_ = 3, 2, 4
E.choice("s11p13a", f"{a_ * b_ * (r_ - 1)}", "組合せ × (繰り返し − 1)")
E.check("s11p13a", (a_ * b_ * r_ - 1) - (a_ - 1) - (b_ - 1) - (a_ - 1) * (b_ - 1) == 18 and opts("s11p13a") == {"18", str((a_ - 1) * (b_ - 1)), str(24 - 1 - (a_ - 1) - (b_ - 1)), "23", str(24 - 3)}, "全体から引いても 18・誤答の由来")
E.choice("s11p13b", f"{30 / 120:.2f}", "群間 ÷ 全体")
E.check("s11p13b", opts("s11p13b") == {f"{30 / 120:.2f}", f"約 {f(30 / 90, 2)}", f"{90 / 120:.2f}", f"{(30 / 2) / (90 / 27):.1f}", f"{120 / 30:.1f}"}, "誤答の由来")
E.done()
