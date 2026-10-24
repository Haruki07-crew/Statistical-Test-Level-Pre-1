"""2級 模擬試験 第13回（g2 set13）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, permutations
from math import comb, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(13)
rng = np.random.default_rng(20261113)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# 第1問
blood = rng.choice(4, 100, p=[.4, .2, .3, .1]); codes = [np.array([1, 2, 3, 4]), np.array([3, 1, 4, 2])]
means = [c[blood].mean() for c in codes]
E.check("s13p01a", abs(means[0] - means[1]) > .1 and np.bincount(blood).argmax() == 0 and ans("s13p01a").startswith("最も人数の多い血液型"), f"番号の付け方で平均が変わる {means[0]:.2f} / {means[1]:.2f}")
vals = [3.2, 3.5, 3.7, 4.0, 4.4]
E.check("s13p01b", ans("s13p01b") == f"{np.median(vals):g}" and opts("s13p01b") == {"3.7", "37", f"{np.mean(vals):.2f}", f"{sorted(vals)[1]:g}", "4"}, "葉の単位 0.1・誤答の由来")

# 第2問
def gini(x):
    x = np.sort(x); n = len(x); return (2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())
E.check("s13p02a", abs(gini(np.full(100, 5.0))) < 1e-12 and gini(np.r_[np.zeros(99), 1.0]) > .98 and abs(gini(2 * rng.lognormal(0, 1, 500)) - gini(rng.lognormal(0, 1, 500))) < .1 and ans("s13p02a").startswith("全員の所得が等しいと 0"), "平等で 0、独占で 0.99")
am, gm, hm = (4 + 9) / 2, sqrt(36), 2 / (1 / 4 + 1 / 9)
E.check("s13p02b", am >= gm >= hm and ans("s13p02b") == f"算術平均 {am:g}、幾何平均 {gm:g}、調和平均 約 {f(hm, 2)}" and opts("s13p02b") == {f"算術平均 {am:g}、幾何平均 {gm:g}、調和平均 約 {f(hm, 2)}", "3 つとも 6.5", f"算術平均 {am:g}、幾何平均 約 {f(hm, 2)}、調和平均 {gm:g}", f"算術平均 {am:g}、幾何平均 {4 * 9}、調和平均 約 {f(hm, 2)}", f"算術平均 {am:g}、幾何平均 {gm:g}、調和平均 約 {f((1 / 4 + 1 / 9) / 2, 2)}"}, f"{am}, {gm}, {hm:.3f}・誤答の由来")

# 第3問
xy = rng.multivariate_normal([0, 0], [[9, -6], [-6, 16]], 500000)
E.choice("s13p03a", f"{-6 / 12:g}".replace("-", "−"), f"シミュレーションの相関 {np.corrcoef(xy.T)[0, 1]:.4f}")
E.check("s13p03a", abs(np.corrcoef(xy.T)[0, 1] + .5) < .005 and opts("s13p03a") == {"−0.5", "−6", f"約 {f(-6 / 144, 3)}", f"{-6 / 25:.2f}".replace("-", "−"), f"約 {f(-6 / 9, 2)}"}, "誤答の由来")
hw = rng.multivariate_normal([170, 67.5], [[64, .6 * 8 * 10], [.6 * 8 * 10, 100]], 400000)
b_yx = np.polyfit(hw[:, 0], hw[:, 1], 1); b_xy = np.polyfit(hw[:, 1], hw[:, 0], 1); near = np.abs(hw[:, 1] - 65) < .5
x_inv = (65 - b_yx[1]) / b_yx[0]; x_reg = np.polyval(b_xy, 65)
E.check("s13p03b", abs(hw[near, 0].mean() - x_reg) < .3 and abs(x_inv - hw[near, 0].mean()) > 2 and ans("s13p03b").startswith("x を y で説明する回帰直線を別に求めて"), f"体重 65 前後の人の身長の平均 {hw[near, 0].mean():.1f}・x を y で回帰 {x_reg:.1f}・式を解く {x_inv:.1f}")
E.check("s13p03b", abs((65 + 60) / .75 - 166.67) < .01, "式を解くと 166.7")

# 第4問
E.check("s13p04a", ans("s13p04a") == f"{4 - (0.9 + 1.2 + 0.8):.1f}" and opts("s13p04a") == {"1.1", "1.0", f"約 {f((0.9 + 1.2 + 0.8) / 3, 2)}", f"{3 - 2.9:.1f}", f"{0 - 2.9:.1f}".replace("-", "−")}, "合計 4・誤答の由来")
ser = rng.normal(0, 1, 36); w = np.r_[.5, np.ones(11), .5] / 12; cma = np.convolve(ser, w, "valid")
E.check("s13p04b", len(cma) == 24 and len(np.convolve(ser, np.ones(12) / 12, "valid")) == 25 and ans("s13p04b") == "24 か月分", f"中心化移動平均の個数 {len(cma)}（中心化しないと 25）")
x5 = np.array([4, 6, 2, 8, 5.]); d5 = x5 - x5.mean(); c1 = (d5[:-1] * d5[1:]).sum(); c0 = (d5 ** 2).sum()
E.check("s13p04c", ans("s13p04c") == f(c1 / c0, 2) and opts("s13p04c") == {f(c1 / c0, 2), f(-c1 / c0, 2), f"約 {f((c1 / 4) / (c0 / 5), 2)}", f"{c1:g}".replace("-", "−"), f"{c1 / 5:g}".replace("-", "−")}, f"r1={c1 / c0}・誤答の由来")

# 第5問
popu = rng.random(100000) < .3; has_phone = rng.random(100000) < np.where(popu, .2, .8); opin = np.where(popu, rng.random(100000) < .7, rng.random(100000) < .4)
E.check("s13p05a", abs(opin[has_phone].mean() - opin.mean()) > .03 and ans("s13p05a").startswith("固定電話のない世帯"), f"全体の賛成 {opin.mean():.3f}・名簿の世帯だけ {opin[has_phone].mean():.3f}")
E.check("s13p05b", ans("s13p05b").startswith("「薬を飲んだ」という期待や"), "盲検化の目的")

# 第6問
users = rng.random((400000, 1))
# 指定の重なりをもつ 3 つの事象を、8 つの領域の確率から作る
reg = {"abc": .05, "ab": .15, "ac": .05, "bc": .05, "a": .25, "b": .15, "c": .15}
reg["none"] = 1 - sum(reg.values())
PA = sum(v for k, v in reg.items() if "a" in k and k != "none"); PB = sum(v for k, v in reg.items() if "b" in k and k != "none"); PC = sum(v for k, v in reg.items() if "c" in k and k != "none")
PAB = reg["abc"] + reg["ab"]; PAC = reg["abc"] + reg["ac"]; PBC = reg["abc"] + reg["bc"]
E.check("s13p06a", np.allclose([PA, PB, PC, PAB, PAC, PBC], [.5, .4, .3, .2, .1, .1]) and abs(1 - reg["none"] - .85) < 1e-12 and ans("s13p06a") == "85%" and opts("s13p06a") == {"85%", "120%", "80%", "75%", "15%"}, f"8 つの領域から作った事象で 少なくとも 1 つ {1 - reg['none']:.2f}")
X = rng.normal(0, 1, 1_000_000); Y = 1.5 * X + rng.normal(0, 1, 1_000_000)
cxy = np.cov(X, Y)[0, 1]; cuv = np.cov(2 * X + 1, -Y + 4)[0, 1]
E.check("s13p06b", abs(cuv / cxy + 2) < 1e-9 and ans("s13p06b") == "−6" and opts("s13p06b") == {"−6", "6", "3", "12", "−2"}, f"Cov(U,V)/Cov(X,Y) = {cuv / cxy:.6f}")
u = rng.random(1_000_000) ** (1 / 3)
E.check("s13p06c", abs(np.median(u) - .5 ** (1 / 3)) < .002 and ans("s13p06c") == "約 0.79" and opts("s13p06c") == {"約 0.79", "0.5", f"{.5 ** 3:g}", f"{3 / 4:g}", f"約 {f(sqrt(.5), 2)}"}, f"F=x³ の乱数の中央値 {np.median(u):.4f}")

# 第7問
n_, p_ = 20, .6
E.check("s13p07a", stats.binom(n_, p_).mean() == 12 and abs(stats.binom(n_, p_).var() - 4.8) < 1e-9 and ans("s13p07a") == "n ＝ 20、p ＝ 0.6" and abs(stats.binom(30, .4).var() - 7.2) < 1e-9 and stats.binom(24, .5).var() == 6, "B(20,0.6) は期待値 12・分散 4.8（B(30,0.4) は分散 7.2）")
E.check("s13p07b", abs(stats.expon(scale=5).std() - 5) < 1e-12 and ans("s13p07b") == "5 日" and opts("s13p07b") == {"5 日", f"約 {f(sqrt(5), 2)} 日", "2.5 日", "0.2 日", "25 日"}, "指数分布の標準偏差 = 平均")
E.check("s13p07c", ans("s13p07c") == f"約 {round(1000 * .0228)} 人" and opts("s13p07c") == {"約 23 人", f"約 {round(2000 * .0228)} 人", f"約 {round(1000 * .1587)} 人", f"約 {round(1000 * (1 - .0228))} 人", "50 人"}, f"1000 × 0.0228 = {1000 * .0228}（正確には {1000 * stats.norm.sf(2):.1f}）")

# 第8問
pa = rng.binomial(100, .5, 1_000_000) / 100; pb = rng.binomial(100, .4, 1_000_000) / 100
E.choice("s13p08a", "約 0.076", f"シミュレーション P(A<B) {np.mean(pa < pb):.4f}・P(A≤B) {np.mean(pa <= pb):.4f}（正規近似 0.0764）")
E.check("s13p08a", abs(np.mean(pa < pb) - .0764) < .015 and abs(.1 / sqrt(.0049) - 1.43) < .005 and opts("s13p08a") == {"約 0.076", f"約 {f(.0228, 3)}", f"約 {f(.1562, 3)}", f"約 {f(2 * .0764, 3)}", f"約 {f(1 - .0764, 3)}"} and abs(.1 / (.05 + sqrt(.0024)) - 1.01) < .005, "誤答の由来（SD 0.05 で z=2・SD の和で z=1.01）")
z = rng.normal(0, 1, 2_000_000)
E.check("s13p08b", abs(np.mean(z ** 2 > 3.84) - .05) < .001 and abs(stats.chi2.sf(3.84, 1) - .05) < 5e-4 and ans("s13p08b") == "0.05", f"シミュレーション {np.mean(z ** 2 > 3.84):.4f}")
E.check("s13p08c", stats.chi2(20).mean() > stats.chi2(5).mean() and abs(stats.chi2(100).stats(moments="s")) < abs(stats.chi2(5).stats(moments="s")) and ans("s13p08c").startswith("自由度が大きくなるほど、分布の中心は右に移り"), f"歪度 自由度 5：{float(stats.chi2(5).stats(moments='s')):.2f}・100：{float(stats.chi2(100).stats(moments='s')):.2f}")

# 第9問
th = 10; nn = 5; M = rng.uniform(0, th, (400000, nn)).max(1)
E.check("s13p09a", abs(M.mean() - nn / (nn + 1) * th) < .01 and abs(((nn + 1) / nn * M).mean() - th) < .01 and ans("s13p09a").startswith("M はいつも θ 以下なので小さめに偏る"), f"E[M]≈{M.mean():.3f}（理論 {nn / (nn + 1) * th:.3f}）・(n+1)/n 倍で {((nn + 1) / nn * M).mean():.3f}")
nreq = .21 * (1.96 / .045) ** 2
E.check("s13p09b", round(nreq, -2) == 400 and ans("s13p09b") == "約 400 人" and opts("s13p09b") == {"約 400 人", f"約 {round(.21 / .045 ** 2, -2):.0f} 人", f"約 {round(.21 * (1.645 / .045) ** 2, -1):.0f} 人", f"約 {round(.21 * (1.96 / .0225) ** 2, -2):.0f} 人", f"約 {round(sqrt(nreq)):.0f} 人"}, f"n = {nreq:.1f}・誤答の由来")
pop = rng.normal(40, 15, 200000); smp = rng.choice(pop, (5000, 100))
inside_ind = np.mean(np.abs(pop - 40) <= 1.96 * 15); cover = np.mean(np.abs(smp.mean(1) - pop.mean()) <= 1.96 * smp.std(1, ddof=1) / 10)
E.check("s13p09c", abs(inside_ind - .95) < .01 and abs(cover - .95) < .01 and ans("s13p09c").startswith("前者は個々の人の通勤時間"), f"個々の値が ±1.96×15 に入る割合 {inside_ind:.3f}・±1.96×SE の区間の被覆率 {cover:.3f}")
E.check("s13p09d", stats.norm.isf(.05) < stats.norm.isf(.025) and ans("s13p09d").startswith("90% 信頼区間は、95% 信頼区間の内側"), "1.645 < 1.96")

# 第10問
b20 = stats.binom(20, .5)
E.check("s13p10a", abs(b20.pmf(15) - .0148) < 5e-5 and abs(b20.sf(15) - .0059) < 5e-5 and ans("s13p10a") == f"約 {f(b20.sf(14), 3)}" and opts("s13p10a") == {f"約 {f(b20.sf(14), 3)}", f"約 {f(b20.pmf(15), 3)}", f"約 {f(2 * b20.sf(14), 3)}", f"約 {f(b20.sf(15), 3)}", "0.75"}, f"P(X≥15) = {b20.sf(14):.4f}・表の値・誤答の由来")
E.check("s13p10b", ans("s13p10b").startswith("不良率に差があるとはいえないが"), "棄却されない ≠ 帰無仮説が正しい")
g1 = rng.normal(60, 15, 100000); g2 = rng.normal(60, 15, 100000); g2 = g2 - g2.mean() + g1.mean() + .3
tt = stats.ttest_ind(g2, g1)
E.check("s13p10c", tt.pvalue < .001 and abs(.3 / (15 * sqrt(2 / 100000)) - 4.47) < .01 and stats.ttest_ind(g1 + 0, g1 - g1.mean() + g1.mean() + .1 + 0 * g1).pvalue > .05 and ans("s13p10c").startswith("統計的には有意だが"), f"10 万人ずつ・標準偏差 15・差 0.3 点で p={tt.pvalue:.1e}（z≈4.47。差 0.1 点なら有意にならない）")
zz = rng.normal(2, 1, 1_000_000)
p1s, p2s = np.mean(zz > 1.645), np.mean(np.abs(zz) > 1.96)
E.check("s13p10d", abs(p1s - .639) < .003 and abs(p2s - .516) < .003 and abs(stats.norm.sf(-.355) - .639) < .001 and abs(stats.norm.sf(-.04) - .516) < .001 and ans("s13p10d") == "右片側 約 0.64、両側 約 0.52", f"シミュレーション 右片側 {p1s:.3f}・両側 {p2s:.3f}")

# 第11問
tab = np.array([[50, 30], [40, 80]]); c_n = stats.chi2_contingency(tab, correction=False).statistic; c_pct = stats.chi2_contingency(tab / 2, correction=False).statistic
E.check("s13p11a", abs(c_pct - c_n / 2) < 1e-9 and ans("s13p11a").startswith("統計量は人数に比例するので"), f"200 人の表 {c_n:.2f} → パーセントの表 {c_pct:.2f}")
ex = stats.chi2_contingency(np.array([[12, 30, 18], [25, 10, 5]])).expected_freq
E.check("s13p11b", np.allclose(ex.sum(1), [60, 40]) and np.allclose(ex.sum(0), [37, 40, 23]) and ans("s13p11b").startswith("各行の合計と各列の合計は"), "期待度数の行・列の合計は観測と同じ")

# 第12問
hx = rng.uniform(150, 190, 100); hy = -60 + .75 * hx + rng.normal(0, 6, 100); bb, aa = np.polyfit(hx, hy, 1)
E.check("s13p12a", hx.min() > 140 and aa < 0 and bb > 0 and ans("s13p12a").startswith("身長 0 cm の人の体重の予測値にあたるが"), f"150〜190 cm のデータで切片 {aa:.1f}（x=0 はデータの外）")
def tstat(n):
    x = rng.normal(0, 1, n); y = .15 * x + rng.normal(0, 1, n); r = stats.linregress(x, y); return r.slope / r.stderr
ratio = np.mean([tstat(400) for _ in range(800)]) / np.mean([tstat(100) for _ in range(800)])
E.check("s13p12b", abs(ratio - 2) < .2 and ans("s13p12b") == "約 3.0（約 2 倍）になる", f"同じ条件で n を 4 倍にしたときの t の平均の比 {ratio:.2f}")
x1 = rng.normal(0, 1, 200); x2 = rng.normal(0, 1, 200); x2 = x2 - x1 * (x1 @ (x2 - x2.mean())) / (x1 @ (x1 - x1.mean()))
x2 -= x2.mean(); x1c = x1 - x1.mean(); x2 = x2 - x1c * (x1c @ x2) / (x1c @ x1c)
y = 1 + 2 * x1 - x2 + rng.normal(0, 1, 200)
bm = np.linalg.lstsq(np.c_[np.ones(200), x1, x2], y, rcond=None)[0]; bs = np.polyfit(x1, y, 1)[0]
E.check("s13p12c", abs(np.corrcoef(x1, x2)[0, 1]) < 1e-12 and abs(bm[1] - bs) < 1e-9 and ans("s13p12c") == "y を x₁ だけで説明する単回帰の傾きと等しくなる", f"無相関の x₁, x₂：重回帰 {bm[1]:.6f}・単回帰 {bs:.6f}")

# 第13問
allocs = {frozenset(c) for c in combinations(range(6), 3)}
E.check("s13p13a", len(allocs) == comb(6, 3) == 20 and len(set(permutations(range(6)))) == 720 and ans("s13p13a") == "20 通り" and opts("s13p13a") == {"20 通り", "720 通り", f"{6 * 5 * 4} 通り", f"{2 ** 6} 通り", "6 通り"}, "6C3 = 20・誤答の由来")
m = {"A1B1": 20, "A1B2": 30, "A2B1": 26, "A2B2": 36}
mainA = (m["A2B1"] + m["A2B2"]) / 2 - (m["A1B1"] + m["A1B2"]) / 2; mainB = (m["A1B2"] + m["A2B2"]) / 2 - (m["A1B1"] + m["A2B1"]) / 2
inter = (m["A2B2"] - m["A2B1"]) - (m["A1B2"] - m["A1B1"])
E.check("s13p13b", ans("s13p13b") == f"{mainA:g}" and opts("s13p13b") == {f"{mainA:g}", f"{mainB:g}", f"{inter:g}", f"{m['A2B2'] - m['A1B1']:g}", f"{(m['A2B1'] - m['A1B1']) + (m['A2B2'] - m['A1B2']):g}"}, f"A の主効果 {mainA}・B {mainB}・交互作用 {inter}")
E.done()
