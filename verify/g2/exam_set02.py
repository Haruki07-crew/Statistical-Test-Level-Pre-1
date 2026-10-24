"""2級 模擬試験 第2回（g2 set02）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の選択肢の値がその誤り方で出るかを確かめる
（選択肢の並びは正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import exp, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(2)
rng = np.random.default_rng(20261025)


def f(x, d):
    """四捨五入した文字列。負の数は問題文と同じくマイナス記号（−）で書く。"""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


# ---- 大問1 幹葉図
x = np.array([42, 47, 51, 53, 55, 58, 61, 62, 62, 66, 69, 73, 75, 81, 94])
E.choice("s02p01a", f"{int(np.median(x))} 点", f"中央値 {np.median(x)}")
E.check("s02p01a", x.sum() == 949 and opts("s02p01a") == {"62 点", f"{x[6]} 点", f"{f(x.mean(), 1)} 点", f"{(x.min() + x.max()) // 2} 点", "60 点"} and stats.mode(x).mode == 62, "誤答の由来・平均 63.3・最頻値 62")
# ---- 大問2 偏差値
dev = {k: 50 + 10 * (s - m) / sd for k, (m, sd, s) in {"国語": (60, 16, 68), "数学": (70, 5, 74), "英語": (40, 4, 45)}.items()}
E.check("s02p02a", [k for k, v in dev.items() if v > 60] == ["英語"] and ans("s02p02a") == "英語だけ", f"偏差値 {dev}")
# ---- 大問3 ジニ係数
share = np.array([.05, .10, .15, .25, .45]); cs = np.cumsum(share)
area = np.trapezoid(np.r_[0, cs], np.linspace(0, 1, 6))
E.choice("s02p03a", f(1 - 2 * area, 2), f"台形公式（np.trapezoid）で面積 {area:.4f}")
inc = np.repeat(share / 0.2, 1000); g = np.abs(inc[:, None] - inc[None, :]).mean() / (2 * inc.mean())
E.check("s02p03a", abs(g - 0.38) < 0.001 and opts("s02p03a") == {f(1 - 2 * area, 2), f(area, 2), f(0.5 - area, 2), f(1 - area, 2), "0.45"}, f"各グループ内を等しくした所得で平均差の式 {g:.4f}・誤答の由来")
# ---- 大問4 クロス表と偏相関
E.choice("s02p04a", "60% と 56.25%", "行パーセント")
E.check("s02p04a", f(72 / 117 * 100, 1) == "61.5" and f(45 / 117 * 100, 1) == "38.5" and 72 / 200 == 0.36 and 45 / 200 == 0.225 and 35 / 80 == 0.4375 and f(117 / 200 * 100, 1) == "58.5", "誤答の由来（列パーセント・全体パーセント・反対の割合・全体の賛成の割合）")
rxy, rxz, ryz = 0.6, 0.8, 0.7
C = np.array([[1, rxy, rxz], [rxy, 1, ryz], [rxz, ryz, 1]]); P = np.linalg.inv(C)
pc = -P[0, 1] / sqrt(P[0, 0] * P[1, 1])
E.choice("s02p04b", f(pc, 2), f"精度行列から偏相関 {pc:.4f}")
E.check("s02p04b", opts("s02p04b") == {f(pc, 2), "0.60", f(rxy - rxz * ryz, 2), f((rxy - rxz * ryz) / ((1 - rxz ** 2) * (1 - ryz ** 2)), 2), f(rxz * ryz, 2)}, "誤答の由来")
# ---- 大問5 回帰
b = 0.6 * 12 / 4
E.choice("s02p05a", f"傾き {f(b, 1)}、決定係数 {f(0.36, 2)}", "b = r sy/sx、R² = r²")
xs = rng.normal(0, 1, 5000); ys = rng.normal(0, 1, 5000)
xs = (xs - xs.mean()) / xs.std(); ys = ys - ys.mean(); ys -= (ys @ xs) / (xs @ xs) * xs; ys /= ys.std()
X = 5 + 4 * xs; Y = 60 + 12 * (0.6 * xs + 0.8 * ys)
bb, aa = np.polyfit(X, Y, 1)
E.check("s02p05a", abs(bb - 1.8) < 1e-9 and abs(aa - 51) < 1e-9 and abs(np.corrcoef(X, Y)[0, 1] ** 2 - 0.36) < 1e-9, f"要約値どおりのデータで a={aa:.3f}, b={bb:.3f}（s02p05b の直線 51+1.8x とも一致）")
E.check("s02p05b", ans("s02p05b") == f(70 - (51 + 1.8 * 8), 1), "残差 = 実際 − 予測 = 4.6（−4.6 は符号が逆の誤答）")
E.check("s02p05b", opts("s02p05b") == {"4.6", "−4.6", "65.4", "10.0", f(70 - (60 + 0.6 * 3), 1)}, "誤答の由来")
# ---- 大問6 移動平均と寄与率
s = np.array([12, 15, 11, 14, 18, 16, 20.])
E.choice("s02p06a", f(s[1:6].mean(), 1), "2〜6 か月目の平均")
E.check("s02p06a", opts("s02p06a") == {f(s[1:6].mean(), 1), f(s[2:5].mean(), 1), f(s[3:6].mean(), 1), f(s[0:5].mean(), 1), f(s[1:5].mean(), 1)}, "誤答の由来")
prev = np.array([100, 200, 100]); now = np.array([120, 210, 110])
contrib = (now - prev) / prev.sum() / ((now.sum() - prev.sum()) / prev.sum())
E.choice("s02p06b", f"{int(round(contrib[0] * 100))}%", f"寄与率 {contrib}")
E.check("s02p06b", abs(contrib.sum() - 1) < 1e-12 and opts("s02p06b") == {"50%", "20%", "5%", "25%", f(120 / 440 * 100, 1) + "%"}, "寄与率の合計 100%・誤答の由来")
# ---- 大問7 調査の設計
E.check("s02p07a", ans("s02p07a").startswith("非標本誤差であり、標本を大きくしても"), "質問文の誤解は非標本誤差")
true_p = 0.5; misread = 0.2  # 誤解した人は「賛成」と答えやすいとする
for n in (100, 10000):
    pop = rng.random(n) < true_p; mis = rng.random(n) < misread; resp = np.where(mis, True, pop)
E.check("s02p07a", abs(resp.mean() - (true_p + misread * (1 - true_p))) < 0.02, f"n=10000 でも偏り {resp.mean() - true_p:.3f} が残る")
E.choice("s02p07b", "120 人・60 人・20 人", "比例配分")
E.check("s02p07b", [200 * v // 10000 for v in (6000, 3000, 1000)] == [120, 60, 20], "200 × 層の割合")
# ---- 大問8 確率
pA, pB, pAB = 0.4, 0.5, 0.2
E.choice("s02p08a", "P(A\\mid B)=0.4$ で、$A$ と $B$ は独立である", f"P(A|B)={pAB / pB}, P(A)P(B)={pA * pB}")
E.check("s02p08a", abs(pAB / pB - pA) < 1e-12 and abs(pA * pB - pAB) < 1e-12, "独立の条件")
fac = rng.choice(3, 2_000_000, p=[.5, .3, .2]); bad = rng.random(2_000_000) < np.array([.01, .02, .04])[fac]
E.choice("s02p08b", f(0.5 * 0.01 + 0.3 * 0.02 + 0.2 * 0.04, 3), f"全確率の公式（シミュレーション {bad.mean():.4f}）")
E.check("s02p08b", abs(bad.mean() - 0.019) < 0.0005 and opts("s02p08b") == {"0.019", "0.070", f(0.07 / 3, 3), "0.008", f(0.008 / 0.019, 3)}, "誤答の由来")
# ---- 大問9 期待値・分散
Xs = rng.normal(5, 2, 2_000_000); Ys = 3 * Xs - 2
E.choice("s02p09a", "E[Y]=13,\\ V[Y]=36", f"シミュレーション E={Ys.mean():.3f}, V={Ys.var():.3f}")
E.check("s02p09a", abs(Ys.mean() - 13) < 0.01 and abs(Ys.var() - 36) < 0.1, "シミュレーションと一致")
cov = np.array([[4, -2], [-2, 9]]); Z = rng.multivariate_normal([0, 0], cov, 2_000_000)
vs = Z.sum(1).var(); rr = np.corrcoef(Z.T)[0, 1]
E.choice("s02p09b", f"V[X+Y]=9$、相関係数 $-{f(2 / 6, 2)}$", f"シミュレーション V={vs:.3f}, r={rr:.4f}")
E.check("s02p09b", abs(vs - 9) < 0.05 and abs(rr + 1 / 3) < 0.003 and f(2 / 36, 2) == "0.06", "シミュレーションと誤答の由来")
# ---- 大問10 ポアソン・指数
po = stats.poisson(3)
E.choice("s02p10a", f(po.cdf(1), 3), "scipy の poisson")
E.check("s02p10a", opts("s02p10a") == {f(po.cdf(1), 3), f(po.pmf(0), 3), f(po.pmf(1), 3), f(po.sf(1), 3), f(po.pmf(2), 3)} and abs(exp(-3) - 0.0498) < 5e-5, "誤答の由来と表の値")
ex = stats.expon(scale=4)
E.choice("s02p10b", f(ex.sf(2), 3), "scipy の expon（平均 4）")
E.check("s02p10b", opts("s02p10b") == {f(ex.sf(2), 3), f(ex.cdf(2), 3), "0.500", f(ex.sf(4), 3), f(exp(-8), 4)} and abs(exp(-0.5) - 0.6065) < 5e-5, "誤答の由来と表の値")
# ---- 大問11 標本分布
m4 = rng.normal(10, 4, (500000, 4)).mean(1)
E.choice("s02p11a", f(stats.norm.sf(1), 4), f"シミュレーション {np.mean(m4 > 12):.4f}")
E.check("s02p11a", abs(np.mean(m4 > 12) - 0.1587) < 0.003 and opts("s02p11a") == {"0.1587", "0.3085", "0.0228", f(1 - 0.1587, 4), f(0.5 - 0.1587, 4)}, "誤答の由来")
ph = rng.binomial(96, 0.4, 1_000_000) / 96
E.choice("s02p11b", f(stats.norm.sf(2), 4), f"正規近似（正確な二項確率 {np.mean(ph >= 0.5):.4f}）")
E.check("s02p11b", abs(sqrt(0.24 / 96) - 0.05) < 1e-12 and opts("s02p11b") == {"0.0228", f(stats.norm.sf(0.1 / sqrt(0.25 / 96)), 4), f(1 - 0.0228, 4), f(2 * 0.0228, 4), f(stats.norm.sf(1.90), 4)}, "誤答の由来（p=0.5 で z=1.96）")
E.check("s02p11b", abs((0.5 - 1 / 192 - 0.4) / 0.05 - 1.90) < 0.01 and abs(stats.norm.sf(1.9) - 0.0287) < 5e-5, "連続修正をした誤答 z≈1.90・表の値")
E.choice("s02p11c", "0.25", "1/k²（k=2）")
for dist in (stats.uniform(50 - 5 * sqrt(3), 10 * sqrt(3)), stats.norm(50, 5), stats.laplace(50, 5 / sqrt(2))):
    E.check("s02p11c", 1 - (dist.cdf(60) - dist.cdf(40)) <= 0.25, f"{dist.dist.name} でも上限 0.25 以下")
# ---- 大問12 推定
E.check("s02p12a", ans("s02p12a").startswith("$\\dfrac{1}{n-1}"), "不偏分散")
smp = rng.normal(0, 2, (200000, 5)); v1 = smp.var(1, ddof=1).mean(); v0 = smp.var(1).mean()
E.check("s02p12a", abs(v1 - 4) < 0.03 and abs(v0 - 3.2) < 0.03, f"n−1 で割ると平均 {v1:.3f}（σ²=4）、n で割ると {v0:.3f}")
se = sqrt(100 / 100 + 144 / 100)
E.choice("s02p12b", iv(4 - 1.96 * se, 4 + 1.96 * se, 2), f"標準誤差 {se:.4f}")
E.check("s02p12b", opts("s02p12b") == {iv(4 - 1.96 * se, 4 + 1.96 * se, 2), iv(4 - 1.645 * se, 4 + 1.645 * se, 2), iv(4 - 1.96 * 2.2, 4 + 1.96 * 2.2, 2), iv(4 - 1.96 * sqrt(244), 4 + 1.96 * sqrt(244), 1), iv(4 - 1.96 * sqrt(244 / 200), 4 + 1.96 * sqrt(244 / 200), 2)}, "誤答の由来")
c1, c2 = stats.chi2.isf(0.025, 9), stats.chi2.isf(0.975, 9)
E.choice("s02p12c", iv(180 / 19.02, 180 / 2.700, 2), f"表の値（正確な点 {c1:.3f}, {c2:.4f}）")
E.check("s02p12c", abs(c1 - 19.02) < 0.005 and abs(c2 - 2.700) < 5e-4 and abs(stats.chi2.isf(0.05, 9) - 16.92) < 0.005 and abs(stats.chi2.isf(0.95, 9) - 3.325) < 5e-4 and abs(stats.chi2.isf(0.025, 10) - 20.48) < 0.005 and abs(stats.chi2.isf(0.975, 10) - 3.247) < 5e-4, "表の値")
E.check("s02p12c", opts("s02p12c") == {iv(180 / 19.02, 180 / 2.700, 2), iv(200 / 19.02, 200 / 2.700, 2), iv(180 / 20.48, 180 / 3.247, 2), iv(9 * sqrt(20) / 19.02, 9 * sqrt(20) / 2.700, 2), iv(180 / 16.92, 180 / 3.325, 2)}, "誤答の由来")
cover = 0
for _ in range(20):
    sm = rng.normal(0, sqrt(20), (5000, 10)); s2 = sm.var(1, ddof=1)
    cover += np.sum((9 * s2 / 19.02 <= 20) & (20 <= 9 * s2 / 2.700))
E.check("s02p12c", abs(cover / 100000 - 0.95) < 0.005, f"被覆率 {cover / 100000:.4f}")
sd2 = sqrt(0.24 / 200 + 0.25 / 200)
E.choice("s02p12d", iv(0.1 - 1.96 * sd2, 0.1 + 1.96 * sd2, 3), f"標準誤差 {sd2:.5f}")
E.check("s02p12d", opts("s02p12d") == {iv(0.1 - 1.96 * sd2, 0.1 + 1.96 * sd2, 3), iv(0.1 - 1.645 * sd2, 0.1 + 1.645 * sd2, 3), iv(0.1 - 1.96 * sqrt(0.49) / 200, 0.1 + 1.96 * sqrt(0.49) / 200, 3), iv(0.1 - 1.96 * (sqrt(0.24 / 200) + sqrt(0.25 / 200)), 0.1 + 1.96 * (sqrt(0.24 / 200) + sqrt(0.25 / 200)), 3), iv(0.1 - 1.96 * sqrt(0.24 / 200), 0.1 + 1.96 * sqrt(0.24 / 200), 3)}, "誤答の由来")
# ---- 大問13 検定
pw = lambda n, a: stats.norm.sf(stats.norm.isf(a) - 0.5 * sqrt(n))
E.check("s02p13a", pw(40, 0.05) > pw(20, 0.05) and pw(20, 0.01) < pw(20, 0.05) and ans("s02p13a").startswith("標本の大きさを大きくすると"), f"検出力：n=20 {pw(20, .05):.3f}、n=40 {pw(40, .05):.3f}、α=0.01 {pw(20, .01):.3f}")
z = (105 - 100) / (15 / 5)
E.choice("s02p13b", f"z ≈ {f(z, 2)} で、1.645 より大きいので棄却する", f"z={z:.4f}, 片側 p={stats.norm.sf(z):.4f}")
E.check("s02p13b", abs(stats.norm.sf(1.67) - 0.0475) < 5e-5 and f(5 / 15, 2) == "0.33" and f(5 / (15 / 25), 2) == "8.33" and f(2 * 0.0475, 3) == "0.095", "表の値と誤答の由来")
E.check("s02p13c", ans("s02p13c").startswith("帰無仮説が正しいとしたとき"), "p 値の定義")
null = rng.normal(0, 1, 1_000_000)
E.check("s02p13c", abs(np.mean(np.abs(null) >= stats.norm.isf(0.015)) - 0.03) < 0.001, "帰無仮説のもとで |Z| ≥ 観測値 の割合が p 値")
# ---- 大問14 2 群の比較
r = stats.ttest_ind_from_stats(52, 4, 10, 48, sqrt(20), 10, equal_var=True)
E.choice("s02p14a", f(r.statistic, 2), f"scipy の ttest_ind_from_stats t={r.statistic:.4f}, p={r.pvalue:.4f}")
E.check("s02p14a", abs(stats.t.isf(0.025, 18) - 2.101) < 5e-4 and r.statistic > stats.t.isf(0.025, 18) and opts("s02p14a") == {f(r.statistic, 2), f(4 / sqrt(18), 2), f(4 / sqrt(1.8), 2), f(4 / sqrt(0.9), 2), f(4 / 6, 2)}, "誤答の由来・2.101 を超える")
o = np.array([8, 12, 9, 11, 6, 14]); cg = stats.chisquare(o)
E.choice("s02p14b", f"χ² ＝ {f(cg.statistic, 1)} で、棄却しない", f"scipy の chisquare χ²={cg.statistic}, p={cg.pvalue:.3f}")
E.check("s02p14b", abs(stats.chi2.isf(0.05, 5) - 11.07) < 0.005 and abs(stats.chi2.isf(0.05, 1) - 3.841) < 5e-4 and f(((o - 10) ** 2).sum() / 100, 2) == "0.42" and ((o - 10) ** 2).sum() == 42, "表の値・誤答の由来")
E.check("s02p14c", ans("s02p14c").startswith("(i) 対応のある t 検定、(ii) ウェルチの検定、(iii) 自由度 2"), "検定の選び方")
E.check("s02p14c", stats.chi2_contingency(np.array([[30, 20], [25, 25], [15, 35]])).dof == 2, "3×2 表の自由度 2")
# ---- 大問15 回帰の出力
E.check("s02p15a", ans("s02p15a").startswith("気温が同じなら、平日は休日より電力使用量が平均 18.0 kWh 多い"), "ダミーの係数の解釈")
n = 23; t1 = rng.uniform(10, 35, n); wd = (np.arange(n) % 7 < 5).astype(float)
yy = 120 + 2.4 * t1 + 18 * wd + rng.normal(0, 5, n); A = np.c_[np.ones(n), t1, wd]
coef = np.linalg.lstsq(A, yy, rcond=None)[0]
E.check("s02p15a", abs(coef[2] - 18) < 8, f"同じ形のデータで推定すると平日の係数 {coef[2]:.1f}（平日と休日の差）")
E.choice("s02p15b", iv(2.4 - 2.086 * 0.5, 2.4 + 2.086 * 0.5, 2), "推定値 ± t × 標準誤差")
E.check("s02p15b", abs(stats.t.isf(0.025, 20) - 2.086) < 5e-4 and abs(stats.t.isf(0.025, 23) - 2.069) < 5e-4 and abs(stats.t.isf(0.05, 20) - 1.725) < 5e-4 and abs(2 * stats.t.sf(3, 20) - 0.007) < 5e-4 and 2 * stats.t.sf(4.8, 20) < 0.0005 and 2 * stats.t.sf(8, 20) < 0.0005, "表の値・出力の p 値（自由度 20）")
E.check("s02p15b", opts("s02p15b") == {iv(2.4 - k * w, 2.4 + k * w, 2) for k, w in ((2.086, .5), (1.96, .5), (2.069, .5), (1.725, .5), (2.086, 1))}, "誤答の由来")
F = (0.6 / 2) / (0.4 / 20)
E.choice("s02p15c", f(F, 1), f"F={F}, F(0.05;2,20)={stats.f.isf(0.05, 2, 20):.3f}")
E.check("s02p15c", opts("s02p15c") == {f(F, 1), f((0.6 / 2) / (0.4 / 22), 1), "1.5", f((0.6 / 3) / (0.4 / 19), 1), f((0.6 / 2) / (0.4 / 23), 2)} and abs(stats.f.isf(0.05, 2, 20) - 3.49) < 0.005, "誤答の由来・解説の 3.49")
# ---- 大問16 実験
E.check("s02p16a", ans("s02p16a").startswith("教え方以外の条件"), "無作為化の目的")
F16 = (36 / 3) / (64 / 16)
E.choice("s02p16b", f"F ＝ {f(F16, 1)} で、有意でない", f"F={F16}, 上側 5% 点 {stats.f.isf(0.05, 3, 16):.3f}")
E.check("s02p16b", abs(stats.f.isf(0.05, 3, 16) - 3.239) < 5e-4 and f(36 / 64, 2) == "0.56" and f(12 / (100 / 16), 2) == "1.92" and f(12 / (64 / 19), 2) == "3.56", "表の値・誤答の由来")
E.done()
