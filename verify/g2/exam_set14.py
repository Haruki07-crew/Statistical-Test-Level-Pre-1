"""2級 模擬試験 第14回（g2 set14）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import log, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(14)
rng = np.random.default_rng(20261114)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# 第1問
sales = 500 * 1.05 ** np.arange(10); staff = 2000 * 1.01 ** np.arange(10)
E.check("s14p01a", (sales / sales[0] * 100)[0] == (staff / staff[0] * 100)[0] == 100 and abs((sales / sales[0] * 100)[-1] - 155.1) < .1 and abs((staff / staff[0] * 100)[-1] - 109.4) < .1 and ans("s14p01a").startswith("2 つの系列を、それぞれ最初の年を 100 とする指数"), f"指数にすると 155.1 と {(staff / staff[0] * 100)[-1]:.1f} を同じ目盛りで比べられる")
E.check("s14p01b", ans("s14p01b").startswith("分布の形がわかるうえに"), "幹葉図は元の値が残る")
a = np.r_[np.full(50, 3)]; b = np.r_[np.full(25, 1), np.full(25, 5)]
E.check("s14p01c", a.mean() == b.mean() == 3 and np.mean(b >= 4) == .5 and ans("s14p01c").startswith("5 段階は順序尺度なので"), "平均 3 でも全員 3 と 1・5 に分かれる場合がある")

# 第2問
sc = rng.multivariate_normal([50, 50], [[100, 50], [50, 100]], 1_000_000); tot = sc.sum(1)
z_tot = (100 + 20 - tot.mean()) / tot.std()
E.choice("s14p02a", f"約 {f(50 + 10 * z_tot, 1)}", f"シミュレーションの合計の平均 {tot.mean():.1f}・標準偏差 {tot.std():.2f}・偏差値 {50 + 10 * z_tot:.2f}")
E.check("s14p02a", abs(tot.std() - sqrt(300)) < .05 and opts("s14p02a") == {f"約 {f(50 + 20 / sqrt(3), 1)}", "60", "70", f"約 {f(50 + 20 / sqrt(2), 1)}", f"約 {f(50 + 20 / 3, 1)}"}, "誤答の由来（相関 0 なら √200・分散 3 で割る）")
d = np.array([1, 2, 3, 10.])
E.check("s14p02b", ((d - d.mean()) ** 3).sum() == 180 and stats.skew(d) > 0 and d.mean() == 4 and np.median(d) == 2.5 and ans("s14p02b").startswith("偏差の 3 乗の和は 180 で正"), f"Σd³=180・歪度 {stats.skew(d):.3f}")

# 第3問
signs = np.r_[np.ones(23), -np.ones(7)]
E.check("s14p03a", signs.sum() > 0 and ans("s14p03a").startswith("右上と左下では偏差の積が正"), "正の積 23・負の積 7（大きさが同じくらい）")
x = rng.normal(4, 1.5, 300); y = 42 + 5 * x + rng.normal(0, 6, 300); y = y - y.mean() + 62; x = x - x.mean() + 4
b1, b0 = np.polyfit(x, y, 1); c1, c0 = np.polyfit(x - 4, y, 1)
E.check("s14p03b", abs(c0 - 62) < 1e-9 and abs(c1 - b1) < 1e-9 and ans("s14p03b") == "62" and opts("s14p03b") == {"62", "42", "0", "4", f"{5 * 4}"}, f"中心化すると切片 {c0:.3f}（=ȳ）・傾き不変")

# 第4問
p0 = np.array([100, 300.]); q0 = np.array([10, 10.]); q1 = np.array([15, 10.])
LQ = p0 @ q1 / (p0 @ q0) * 100
E.check("s14p04a", ans("s14p04a") == f"{LQ:g}" and opts("s14p04a") == {f"{LQ:g}", f"{(q1 / q0).mean() * 100:g}", f"{q1[0] / q0[0] * 100:g}", f"{q1[1] / q0[1] * 100:g}", f"{(1.5 * 3000 + 1.0 * 1000) / 4000 * 100:g}"}, f"ラスパイレス数量指数 {LQ}・誤答の由来")
t = np.arange(1, 41); ys = 80 + 2.5 * t + np.tile([3, -2, -4, 3], 10) + rng.normal(0, 1, 40); bt = np.polyfit(t, ys, 1)[0]
E.check("s14p04b", abs(bt - 2.5) < .2 and ans("s14p04b") == f"{2.5 * 4:g}" and opts("s14p04b") == {"10", "2.5", f"{80 + 2.5:g}", f"{2.5 * 12:g}", f"{2.5 / 4:g}"}, f"季節の波のある四半期データで傾き {bt:.2f}・1 年で約 {4 * bt:.1f}")

# 第5問
true = 50.0; census = true + rng.normal(0, 1, 2000) + 1.5; sample = true + rng.normal(0, 1, (2000, 1))[:, 0] * 0 + rng.normal(0, 1.0, 2000) * .5
E.check("s14p05a", abs(census.mean() - true) > abs(sample.mean() - true) and ans("s14p05a").startswith("標本調査は対象が少ないぶん"), f"非標本誤差（偏り 1.5）のある全数調査の誤差 {abs(census.mean() - true):.2f}・質を保った標本調査の誤差 {abs(sample.mean() - true):.2f}")
n = 40000; age = rng.uniform(20, 80, n); ex = rng.random(n) < np.clip(1.2 - age / 60, .05, .95); bp = 100 + .6 * age + rng.normal(0, 8, n)
raw = bp[~ex].mean() - bp[ex].mean(); strata = np.digitize(age, [30, 40, 50, 60, 70]); adj = np.mean([bp[(strata == k) & ~ex].mean() - bp[(strata == k) & ex].mean() for k in range(6)])
E.check("s14p05b", raw > 5 and abs(adj) < 2 and ans("s14p05b").startswith("年齢層ごとに分けて"), f"運動に効果がないデータで、そのままの差 {raw:.1f}・年齢層ごとの差の平均 {adj:.2f}")

# 第6問
ok = True
for _ in range(2000):
    pa, pb = rng.uniform(.05, .9, 2); pab = rng.uniform(max(0, pa + pb - 1), min(pa, pb))
    if pab / pb > pa:
        ok &= pab / pa > pb
E.check("s14p06a", ok and ans("s14p06a") == "P(B｜A) ＞ P(B) も成り立つ", "2000 通りの確率で P(A|B)>P(A) なら P(B|A)>P(B)")
u = rng.uniform(0, .5, 1_000_000)
E.check("s14p06b", abs(stats.uniform(0, .5).pdf(.25) - 2) < 1e-12 and abs(np.mean((u >= .1) & (u <= .2)) - .2) < .002 and ans("s14p06b").startswith("密度の値は確率ではなく"), "密度 2・P(0.1≤X≤0.2)=0.2")
J = {(0, 1): .3, (0, 2): .2, (0, 3): .1, (1, 1): .1, (1, 2): .2, (1, 3): .1}
px1 = sum(v for (xx, _), v in J.items() if xx == 1); ey1 = sum(yy * v for (xx, yy), v in J.items() if xx == 1) / px1
ey0 = sum(yy * v for (xx, yy), v in J.items() if xx == 0) / (1 - px1); ey = sum(yy * v for (_, yy), v in J.items())
E.check("s14p06c", abs(ey1 - 2) < 1e-12 and ans("s14p06c") == "2.0" and opts("s14p06c") == {"2.0", f"{ey1 * px1:.1f}", f"{ey:.1f}", f"約 {f(ey0, 2)}", f"{J[(1, 2)] / px1:.1f}"}, f"E[Y|X=1]={ey1}・誤答の由来")

# 第7問
E.check("s14p07a", stats.binom(10, .5).pmf(3) == stats.binom(10, .5).pmf(7) and stats.binom(10, .2).stats(moments="s") > 0 and ans("s14p07a").startswith("p ＝ 0.5 なら左右対称で"), f"B(10,0.2) の歪度 {float(stats.binom(10, .2).stats(moments='s')):.2f}")
m2 = np.minimum(rng.exponential(10, 1_000_000), rng.exponential(10, 1_000_000))
E.check("s14p07b", abs(m2.mean() - 5) < .02 and ans("s14p07b") == "5 時間" and opts("s14p07b") == {"5 時間", "10 時間", "20 時間", f"約 {f(10 / sqrt(2), 2)} 時間", f"約 {f(10 * log(2), 2)} 時間"}, f"シミュレーションの平均 {m2.mean():.3f}")
g = rng.geometric(.2, 1_000_000)
E.check("s14p07c", abs(np.mean(g >= 5) - .8 ** 4) < .002 and ans("s14p07c") == f"約 {f(.8 ** 4, 2)}" and opts("s14p07c") == {f"約 {f(.8 ** 4, 2)}", f"約 {f(.8 ** 5, 2)}", f"約 {f(.8 ** 4 * .2, 2)}", f"約 {f(1 - .8 ** 4, 2)}", "0.2"}, f"シミュレーション P(初成功 ≥ 5) {np.mean(g >= 5):.4f}")

# 第8問
v1 = rng.normal(0, 1, (200000, 10)).var(1, ddof=1); v2 = rng.normal(0, 1, (200000, 10)).var(1, ddof=1)
E.check("s14p08a", abs(np.mean(v1 / v2 > 1) - .5) < .005 and abs(stats.f(9, 9).median() - 1) < 1e-9 and ans("s14p08a") == "0.5", f"シミュレーション P(F>1) {np.mean(v1 / v2 > 1):.4f}・F(9,9) の中央値 1")
sds = {k: (rng.normal(0, 2, (20000, k)).sum(1).std(), rng.normal(0, 2, (20000, k)).mean(1).std()) for k in (4, 64)}
E.check("s14p08b", abs(sds[64][0] / sds[4][0] - 4) < .2 and abs(sds[4][1] / sds[64][1] - 4) < .2 and ans("s14p08b").startswith("合計の標準偏差は √n σ で大きくなり"), f"n 16 倍で合計の SD {sds[64][0] / sds[4][0]:.2f} 倍・平均の SD 1/{sds[4][1] / sds[64][1]:.2f}")
s5 = rng.normal(0, 3, (400000, 5)).var(1, ddof=1)
E.check("s14p08c", abs(np.mean(s5 < 9) - .594) < .004 and abs(stats.chi2.cdf(4, 4) - .594) < 5e-4 and abs(s5.mean() - 9) < .05 and ans("s14p08c") == "約 0.59", f"シミュレーション P(s²<σ²) {np.mean(s5 < 9):.4f}（平均は {s5.mean():.3f} で不偏）")

# 第9問
d12 = 5 / sqrt(1.5 ** 2 + 1.5 ** 2)
E.check("s14p09a", 52.94 > 52.06 and d12 > 1.96 and abs(d12 - 2.36) < .005 and abs(50 - 1.96 * 1.5 - 47.06) < .005 and abs(55 + 1.96 * 1.5 - 57.94) < .005 and ans("s14p09a").startswith("区間が重なっていても、差の標準誤差は約 2.12"), f"区間は重なる・z={d12:.3f}・p={2 * stats.norm.sf(d12):.4f}")
E.check("s14p09a", abs(5 / 3 - 1.67) < .005, "誤答 z=1.67 は標準誤差の和 3 を使った値")
city = rng.normal(3.0, .2, 200000); sub = rng.normal(5.0, .3, 200000); est = .6 * city + .4 * sub
E.choice("s14p09b", f"約 {f(est.std(), 2)} 万円", f"シミュレーションの標準誤差 {est.std():.4f}")
E.check("s14p09b", abs(est.std() - sqrt(.0288)) < .001 and opts("s14p09b") == {f"約 {f(sqrt(.0288), 2)} 万円", f"{.2 + .3:g} 万円", f"{.6 * .2 + .4 * .3:.2f} 万円", f"約 {f(sqrt(.04 + .09), 2)} 万円", f"約 {f(.0288, 3)} 万円"}, "誤答の由来")
def cover(sampler, n=50, reps=8000):
    cm = cv = 0
    for _ in range(reps):
        xx = sampler(n); m, s2 = xx.mean(), xx.var(ddof=1)
        cm += abs(m) <= stats.t.isf(.025, n - 1) * sqrt(s2 / n)
        cv += (n - 1) * s2 / stats.chi2.isf(.025, n - 1) <= 1 <= (n - 1) * s2 / stats.chi2.ppf(.025, n - 1)
    return cm / reps, cv / reps
heavy = lambda n: stats.t(5).rvs(n, random_state=rng) / sqrt(5 / 3)
cm_h, cv_h = cover(heavy)
E.check("s14p09c", cm_h > .93 and cv_h < .85 and ans("s14p09c").startswith("標本が大きければ母平均の区間は"), f"裾の重い分布（t5・分散 1）n=50：平均の区間の被覆率 {cm_h:.3f}・分散の区間 {cv_h:.3f}")
xb = rng.normal(10, 2, (200000, 5)).mean(1)
E.check("s14p09d", abs(xb.mean() - 10) < .01 and xb.std() > .5 and ans("s14p09d").startswith("同じ大きさの標本を何度もとって"), f"標本平均の平均 {xb.mean():.4f}・ばらつき {xb.std():.3f}")

# 第10問
def power_alpha(n, reps=20000):
    se = 10 / sqrt(n); z0 = rng.normal(0, 1, reps); z1 = rng.normal(3 / se, 1, reps)
    return np.mean(z0 > 1.645), np.mean(z1 > 1.645)
a25, p25 = power_alpha(25); a100, p100 = power_alpha(100)
E.check("s14p10a", abs(a25 - .05) < .006 and abs(a100 - .05) < .006 and p100 > p25 and ans("s14p10a").startswith("第 1 種の過誤の確率は 5% のまま変わらず"), f"n=25：α {a25:.3f}・検出力 {p25:.2f}／n=100：α {a100:.3f}・検出力 {p100:.2f}")
rej = 0
for _ in range(4000):
    person = rng.normal(0, 1, 3); vals = (person[:, None] + rng.normal(0, .3, (3, 10))).ravel()
    rej += stats.ttest_1samp(vals, 0).pvalue < .05
E.check("s14p10b", rej / 4000 > .3 and ans("s14p10b").startswith("同じ人の値どうしは似ているので独立ではなく"), f"効果がなくても、3 人 × 10 晩を 30 人とみなすと棄却率 {rej / 4000:.2f}（本来 0.05）")
E.check("s14p10c", abs(stats.norm.sf(1.8) - .0359) < 5e-5 and ans("s14p10c") == f"約 {f(1 - .0359, 3)}" and opts("s14p10c") == {f"約 {f(1 - .0359, 3)}", f"約 {f(.0359, 3)}", f"約 {f(2 * .0359, 3)}", "0.5", "0"}, "P(Z ≥ −1.8) = 0.964・誤答の由来")

# 第11問
rej2 = np.mean([stats.ttest_ind(rng.lognormal(0, 1, 200), rng.lognormal(0, 1, 200)).pvalue < .05 for _ in range(4000)])
E.check("s14p11a", abs(rej2 - .05) < .012 and ans("s14p11a").startswith("標本が大きいので、標本平均の分布は"), f"右に裾の長い分布で各 200：差がないときの棄却率 {rej2:.3f}")
E.check("s14p11b", ans("s14p11b") == f"{(4 * 8 + 8 * 20) / 12:g}" and opts("s14p11b") == {"16", f"{(8 + 20) / 2:g}", f"約 {f((5 * 8 + 9 * 20) / 14, 1)}", f"{8 + 20:g}", f"約 {f((4 * 8 + 8 * 20) / 14, 1)}"}, "プールした分散 192/12・誤答の由来")
E.check("s14p11c", ans("s14p11c").startswith("帰無仮説「2 つの母比率は等しい」のもとでは"), "合わせた比率を使う理由")

# 第12問
def se_slope(xs):
    ses = []
    for _ in range(2000):
        yy = 1 + .5 * xs + rng.normal(0, 1, len(xs)); ses.append(stats.linregress(xs, yy).slope)
    return np.std(ses)
wide = np.linspace(0, 10, 20); narrow = np.linspace(4, 6, 20)
E.check("s14p12a", se_slope(wide) * 3 < se_slope(narrow) and ans("s14p12a").startswith("調べたい範囲の中で、x をできるだけ広く"), f"傾きのばらつき：広い x {se_slope(wide):.3f}・狭い x {se_slope(narrow):.3f}")
def r2(X, y):
    A = np.c_[np.ones(len(y)), X]; r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]; return 1 - r @ r / ((y - y.mean()) ** 2).sum()
okr = True
for _ in range(300):
    rho = rng.uniform(-.9, .9); X = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 200); yy = X @ rng.normal(0, 1, 2) + rng.normal(0, 1, 200)
    okr &= r2(X, yy) >= max(r2(X[:, :1], yy), r2(X[:, 1:], yy)) - 1e-12
E.check("s14p12b", okr and ans("s14p12b") == "0.36 以上である", "300 通りのデータで重回帰の R² ≥ 単回帰の R² の大きいほう")
Fv = .02 / (.98 / 9998)
E.check("s14p12c", Fv > 200 and stats.f.sf(Fv, 1, 9998) < 1e-4 and ans("s14p12c").startswith("説明変数と目的変数に関係があると判断できるが"), f"F = {Fv:.0f}")

# 第13問
gm = np.mean([12, 15, 18])
E.check("s14p13a", ans("s14p13a") == f"{18 - gm:g}" and opts("s14p13a") == {"3", "18", f"{18 - 12:g}", f"{gm:g}", f"{gm - 18:g}".replace("-", "−")}, "効果 18 − 15 = 3・誤答の由来")
true_diff = 0; am = rng.normal(0, 1, 30) + 1.0; pm = rng.normal(0, 1, 30)
E.check("s14p13b", abs((am.mean() - pm.mean()) - true_diff) > .5 and ans("s14p13b").startswith("時間帯による違い"), f"方法に差がなくても時間帯の差 1.0 があると、見かけの差 {am.mean() - pm.mean():.2f}")
E.done()
