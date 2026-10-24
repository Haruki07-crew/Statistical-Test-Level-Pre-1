"""2級 模擬試験 第5回（g2 set05）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations, product
from math import ceil, exp, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(5)
rng = np.random.default_rng(20261028)


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
E.check("s05p01a", ans("s05p01a") == f"{round(126 / 360 * 200)} 人" and opts("s05p01a") == {"70 人", "126 人", "63 人", "35 人", f"{round(126 / 180 * 200)} 人"}, "中心角 → 割合 → 人数")
m = np.array([5, 15, 25, 35.]); fr = np.array([4, 6, 8, 2]); raw = np.repeat(m, fr)
E.choice("s05p01b", f(raw.std(ddof=1), 2), "階級値を度数だけ並べたデータの標準偏差")
E.check("s05p01b", opts("s05p01b") == {f(raw.std(ddof=1), 2), f(raw.std(), 2), f(raw.var(ddof=1), 2), f(m.std(ddof=1), 2), f(sqrt(((raw - raw.mean()) ** 2).sum() / 4), 2)}, "誤答の由来")
sc = rng.normal(60, 12, 101); sh = sc + 5
q = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
E.check("s05p01c", abs(sc.std() - sh.std()) < 1e-9 and abs(q(sc) - q(sh)) < 1e-9 and abs(np.median(sh) - np.median(sc) - 5) < 1e-9 and ans("s05p01c") == "標準偏差と四分位範囲", "5 点加えると散らばりは不変")
# ---- 大問2
grade = rng.integers(1, 7, 3000); foot = 15 + grade + rng.normal(0, 0.8, 3000); kanji = 40 + 8 * grade + rng.normal(0, 8, 3000)
within = [np.corrcoef(foot[grade == g], kanji[grade == g])[0, 1] for g in range(1, 7)]
E.check("s05p02a", np.corrcoef(foot, kanji)[0, 1] > 0.5 and max(abs(w) for w in within) < 0.1 and ans("s05p02a").startswith("学年（年齢）が両方に影響"), f"全体 r={np.corrcoef(foot, kanji)[0, 1]:.2f}、学年内 |r|<0.1")
x = rng.normal(0, 1, 10); y = rng.normal(0, 1, 10)
while abs(np.corrcoef(x, y)[0, 1]) > 0.15:
    y = rng.normal(0, 1, 10)
x2, y2 = np.r_[x, 8], np.r_[y, 8]
E.check("s05p02b", np.corrcoef(x2, y2)[0, 1] > 0.7 and ans("s05p02b").startswith("1 つの離れた点が"), f"r {np.corrcoef(x, y)[0, 1]:.2f} → 1 点加えて {np.corrcoef(x2, y2)[0, 1]:.2f}")
E.check("s05p02c", ans("s05p02c") == "−0.7" and opts("s05p02c") == {"−0.7", "0.7", "0.49", "−0.49", f(0.49 ** 2, 2)}, "r = −√0.49")
E.check("s05p02d", 80 + 6 * 40 == 320 and ans("s05p02d").startswith("データの範囲外への予測（外挿）"), "外挿")
# ---- 大問3
y = np.array([30, 34, 40, 36, 32, 38, 44, 40.])
w = np.array([.5, 1, 1, 1, .5]) / 4
E.choice("s05p03a", f(w @ y[0:5], 2), "重み 1/8・1/4・1/4・1/4・1/8")
E.check("s05p03a", opts("s05p03a") == {f(w @ y[0:5], 2), f(y[0:4].mean(), 2), f(y[1:5].mean(), 2), f(y[1:4].mean(), 2), "40.00"} and abs((y[0:4].mean() + y[1:5].mean()) / 2 - w @ y[0:5]) < 1e-12, "誤答の由来・2 つの 4 項平均の平均と一致")
E.choice("s05p03b", f(110 * 108 / 100, 1), "110 × 1.08")
E.check("s05p03b", opts("s05p03b") == {"118.8", "118.0", f(108 / 110 * 100, 1), "108.0", "109.0"}, "誤答の由来")
# ---- 大問4
strata = [rng.normal(mu, 2, 5000) for mu in (20, 50, 80)]; pop = np.concatenate(strata)
srs = [rng.choice(pop, 30, replace=False).mean() for _ in range(4000)]
st = [np.mean([rng.choice(s_, 10, replace=False).mean() for s_ in strata]) for _ in range(4000)]
E.check("s05p04a", np.std(st) < np.std(srs) / 3 and ans("s05p04a").startswith("層の間で調べたい値が大きく違い"), f"層の平均が違い層内が似ているとき 標準誤差 層化 {np.std(st):.2f}・単純 {np.std(srs):.2f}")
se = sqrt(.25 / 400)
E.choice("s05p04b", f"約 ±{f(1.96 * se * 100, 1)} ポイント", "1.96 × 0.025")
E.check("s05p04b", opts("s05p04b") == {f"約 ±{f(k * 100, 2 if k < 0.01 else 1)} ポイント" for k in (1.96 * se, se, 2 * 1.96 * se, 1.645 * se, 1.96 * se ** 2)}, "誤答の由来")
# ---- 大問5
hit = (rng.random((1_000_000, 3)) < 0.3).any(1)
E.choice("s05p05a", f(1 - 0.7 ** 3, 3), f"シミュレーション {hit.mean():.4f}")
E.check("s05p05a", opts("s05p05a") == {"0.657", "0.900", "0.027", "0.343", "0.973"}, "誤答の値")
balls = "RRRWW"; seq = list(permutations(range(5), 2))
p2 = sum(balls[j] == "R" for i, j in seq) / len(seq); p12 = sum(balls[i] == "R" and balls[j] == "R" for i, j in seq) / len(seq)
E.check("s05p05b", abs(p2 - .6) < 1e-12 and abs(p12 / p2 - .5) < 1e-12 and ans("s05p05b") == "0.6 と 0.5" and abs((2 / 4 + 3 / 4) / 2 - .625) < 1e-12, f"20 通りの総当たり P(2 回目赤)={p2}, 条件付き {p12 / p2}")
J = {(0, 0): .2, (0, 1): .1, (1, 0): .1, (1, 1): .6}
EX = sum(p * x_ for (x_, _), p in J.items()); EY = sum(p * y_ for (_, y_), p in J.items()); EXY = sum(p * x_ * y_ for (x_, y_), p in J.items())
E.check("s05p05c", ans("s05p05c") == f(EXY - EX * EY, 2), "E[XY] − E[X]E[Y] = 0.11（−0.11 と区別するため完全一致）")
E.check("s05p05c", opts("s05p05c") == {f(EXY - EX * EY, 2), f(EXY, 2), f(EX * EY, 2), f(EX * EY - EXY, 2), f(EX * (1 - EX), 2)}, "誤答の由来")
s2 = np.array([a + b for a, b in product(range(1, 7), repeat=2)])
E.choice("s05p05d", f"期待値 {f(s2.mean(), 0)}、分散 {f(s2.var(), 2)}", "36 通りの総当たり")
E.check("s05p05d", opts("s05p05d") == {"期待値 7、分散 5.83", "期待値 7、分散 2.92", "期待値 3.5、分散 2.92", f"期待値 7、分散 {f(4 * 35 / 12, 2)}", f"期待値 7、分散 {f(s2.std(), 2)}"}, "誤答の由来")
# ---- 大問6
E.choice("s05p06a", f"{f(50 + 1.645 * 10, 2)} 点", "μ + 1.645σ")
E.check("s05p06a", abs(stats.norm(50, 10).isf(.05) - 66.45) < 0.01 and opts("s05p06a") == {"66.45 点", f"{f(50 + 1.96 * 10, 1)} 点", f"{f(50 - 1.645 * 10, 2)} 点", "51.645 点", "60 点"}, "正確な点と誤答の由来")
T = rng.exponential(5, 3_000_000); cond = T[T > 2]
E.choice("s05p06b", f(exp(-0.6), 3), f"シミュレーション {np.mean(cond > 5):.4f}")
E.check("s05p06b", abs(np.mean(cond > 5) - exp(-.6)) < 0.003 and opts("s05p06b") == {f(exp(-.6), 3), f(exp(-1), 3), f(exp(-.4), 3), f(1 - exp(-.6), 3), "0.600"}, "誤答の由来")
# ---- 大問7
E.check("s05p07a", ans("s05p07a").startswith("0 について左右対称で") and stats.t.isf(.025, 5) > stats.t.isf(.025, 50) > 1.96, "自由度が大きいと 1.96 に近づく")
T10 = rng.standard_t(10, 2_000_000)
E.choice("s05p07b", f(2.228 ** 2, 2), f"t(10) の 2 乗が F(1,10) を超える割合 {np.mean(T10 ** 2 > 2.228 ** 2):.4f}")
E.check("s05p07b", abs(stats.f.isf(.05, 1, 10) - 2.228 ** 2) < 0.005 and abs(stats.t.isf(.025, 10) - 2.228) < 5e-4 and opts("s05p07b") == {f(2.228 ** 2, 2), "2.23", f(sqrt(2.228), 2), f(1.812 ** 2, 2), f(1 / 2.228 ** 2, 2)}, "scipy の F 点・誤答の由来")
S2 = rng.normal(0, 1, (400000, 10)).var(1, ddof=1)
E.choice("s05p07c", f(16.92 / 9, 2), f"シミュレーション P(S²>1.88σ²)={np.mean(S2 > 16.92 / 9):.4f}")
E.check("s05p07c", abs(np.mean(S2 > 16.92 / 9) - .05) < .002 and opts("s05p07c") == {f(16.92 / 9, 2), "16.92", f(16.92 / 10, 2), f(3.325 / 9, 2), "1.645"}, "誤答の由来")
# ---- 大問8
ph = 50 / 250
E.choice("s05p08a", f"推定値 {f(ph, 2)}、標準誤差 {f(sqrt(ph * (1 - ph) / 250), 3)}", "√(p̂(1−p̂)/n)")
E.check("s05p08a", opts("s05p08a") == {f"推定値 0.20、標準誤差 {v}" for v in (f(sqrt(.16 / 250), 3), f(.16 / 250, 5), f(sqrt(.2 / 250), 3), f(sqrt(.16), 2), f(sqrt(.25 / 250), 3))}, "誤答の由来")
E.choice("s05p08b", iv(60 - 1.645 * 2, 60 + 1.645 * 2, 2), "z の 90% 区間")
E.check("s05p08b", opts("s05p08b") == {iv(60 - k * w, 60 + k * w, 2) for k, w in ((1.645, 2), (1.96, 2), (1.645, 10), (2.576, 2), (1.645, .4))}, "誤答の由来")
E.check("s05p08c", abs(stats.t.isf(.005, 8) - 3.355) < 5e-4 and abs(stats.t.isf(.025, 8) - 2.306) < 5e-4 and abs(stats.t.isf(.005, 9) - 3.250) < 5e-4, "表の値")
E.choice("s05p08c", iv(30 - 3.355 * 2, 30 + 3.355 * 2, 2), "t の 99% 区間")
E.check("s05p08c", opts("s05p08c") == {iv(30 - k * w, 30 + k * w, 2) for k, w in ((3.355, 2), (2.306, 2), (3.250, 2), (2.576, 2), (3.355, 6))}, "誤答の由来")
E.choice("s05p08d", str(ceil((1.96 * 12 / 2) ** 2)), "(1.96 × 12 / 2)² を切り上げ")
E.check("s05p08d", opts("s05p08d") == {"139", "138", str(ceil((1.645 * 12 / 2) ** 2)), str(ceil(1.96 * 12 / 2)), "36"}, "誤答の由来")
# ---- 大問9
E.check("s05p09a", ans("s05p09a") == "第 2 種の過誤", "効果があるのに有意にならない")
E.check("s05p09b", stats.t.isf(.025, 15) < 2.5 < stats.t.isf(.01, 15) and 0.02 < 2 * stats.t.sf(2.5, 15) < 0.05 and ans("s05p09b") == "0.02 と 0.05 の間", f"両側 p 値 {2 * stats.t.sf(2.5, 15):.4f}")
z = (0.62 - 0.5) / 0.05
E.choice("s05p09c", f"z ＝ {f(z, 1)}、p 値 {f(2 * 0.0082, 4)}", f"z={z}, 正確な両側 p={2 * stats.norm.sf(z):.4f}")
E.check("s05p09c", abs(stats.norm.sf(2.4) - .0082) < 5e-5 and abs(stats.norm.sf(.24) - .4052) < 5e-5 and f(.12 / .5, 2) == "0.24" and f(.12 / .0025, 0) == "48", "表の値と誤答の由来")
zz = (52 - 50) / sqrt(64 / 100 + 36 / 100)
E.choice("s05p09d", f"z ＝ {f(zz, 1)} で、有意水準 5% の両側検定で棄却する", f"z={zz}, p={2 * stats.norm.sf(zz):.4f}")
E.check("s05p09d", f(2 / sqrt(64 + 36), 1) == "0.2" and f(2 / 1.4, 2) == "1.43", "誤答の由来")
o = np.array([90, 50, 40, 20]); gf = stats.chisquare(o, 200 * np.array([.4, .3, .2, .1]))
E.choice("s05p09e", f"χ² ≈ {f(gf.statistic, 2)} で、自由度 3 の点 7.815 より小さいので棄却しない", f"scipy の chisquare χ²={gf.statistic:.4f}, p={gf.pvalue:.3f}")
E.check("s05p09e", abs(stats.chi2.isf(.05, 3) - 7.815) < 5e-4 and ((o - 200 * np.array([.4, .3, .2, .1])) ** 2).sum() == 200, "表の値・誤答 200 の由来")
pre = rng.normal(60, 8, 5); d = np.array([1, 2, 3, 2, 2.]); d = (d - d.mean()) / d.std(ddof=1) + 2; post = pre + d
tp = stats.ttest_rel(post, pre).statistic
E.check("s05p09f", abs(tp - 2 / (1 / sqrt(5))) < 1e-9 and abs(2 / sqrt(64 / 5 + 64 / 5) - 0.395) < 0.001 and ans("s05p09f").startswith("対応のある t 検定では t ≈ 4.47"), f"差の平均 2・標準偏差 1 のデータで ttest_rel {tp:.3f}")
# ---- 大問10
E.choice("s05p10a", f(5 + .8 * 10 + 1.5 * 2 - .2 * 15, 1), "代入")
E.check("s05p10a", opts("s05p10a") == {"13.0", f(5 + 8 + 3 + 3, 1), f(8 + 3 - 3, 1), f(5 + 8 + 3, 1), f(8 + 3 + 3, 1)}, "誤答の由来")
n = 40; z0 = rng.normal(0, 1, n); X = np.c_[z0 + .05 * rng.normal(0, 1, n), z0 + .05 * rng.normal(0, 1, n), z0 + .05 * rng.normal(0, 1, n)]
yy = X.sum(1) + rng.normal(0, 1, n); A = np.c_[np.ones(n), X]; b = np.linalg.lstsq(A, yy, rcond=None)[0]; r = yy - A @ b
s2_ = r @ r / (n - 4); se = np.sqrt(np.diag(s2_ * np.linalg.inv(A.T @ A))); tv = b[1:] / se[1:]
R2 = 1 - r @ r / ((yy - yy.mean()) @ (yy - yy.mean())); F = (R2 / 3) / ((1 - R2) / (n - 4))
E.check("s05p10b", F > stats.f.isf(.01, 3, n - 4) and np.all(np.abs(tv) < stats.t.isf(.025, n - 4)) and ans("s05p10b").startswith("説明変数どうしの相関が強く"), f"相関の強い 3 変数で F={F:.1f}（有意）、t 値 {np.round(tv, 2)}（有意でない）")
b10 = np.linalg.lstsq(A, yy / 10, rcond=None)[0]; r10 = yy / 10 - A @ b10; se10 = np.sqrt(np.diag(r10 @ r10 / (n - 4) * np.linalg.inv(A.T @ A)))
E.check("s05p10c", np.allclose(b10, b / 10) and np.allclose(se10, se / 10) and np.allclose(b10 / se10, b / se) and ans("s05p10c").startswith("係数と標準誤差は 1/10"), "y を 1/10 にして当てはめ直す")
# ---- 大問11
E.check("s05p11a", ans("s05p11a").startswith("対照群がないので"), "対照群の必要性")
SE_ = 30 / 5 * (21 - 3); SA = 30 * 2
E.choice("s05p11b", str(int(SE_ + SA)), "平方和 = 平均平方 × 自由度")
E.check("s05p11b", opts("s05p11b") == {"168", str(int(SE_)), str(SA), "36", str(int(SA + 6 * 21))} and abs(stats.f.isf(.05, 2, 18) - 3.555) < 5e-4, "誤答の由来と表の値")
E.done()
