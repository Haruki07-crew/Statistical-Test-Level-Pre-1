"""2級 模擬試験 第3回（g2 set03）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たり・
数値積分など別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の選択肢の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from math import atanh, exp, sqrt, tanh
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(3)
rng = np.random.default_rng(20261026)


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


# ---- 大問1
nA, nB = 400, 600
cnt = {"A": {"運動": nA * .45, "文化": nA * .30, "無所属": nA * .25}, "B": {"運動": nB * .35, "文化": nB * .40, "無所属": nB * .25}}
st = {"文化部の生徒の人数は、B 高校が A 高校の 2 倍である": cnt["B"]["文化"] == 2 * cnt["A"]["文化"],
      "運動部の生徒の人数は、A 高校のほうが多い": cnt["A"]["運動"] > cnt["B"]["運動"],
      "無所属の生徒の人数は、両校で等しい": cnt["A"]["無所属"] == cnt["B"]["無所属"],
      "B 高校の文化部の割合は、A 高校の 2 倍である": abs(.40 / .30 - 2) < 1e-9}
E.check("s03p01a", [k for k, v in st.items() if v] == [ans("s03p01a")], f"人数 {cnt}")
E.choice("s03p01b", "15", "度数 30 ÷ 幅の比 2")
E.check("s03p01b", opts("s03p01b") == {"15", "30", "60", "1.5", "7.5"} and 30 / 20 == 1.5 and 30 / 4 == 7.5, "誤答の由来")
# ---- 大問2
E.check("s03p02a", ans("s03p02a").startswith(f"変動係数は A が {f(6 / 150, 2)}、B が {f(4 / 50, 2)} で、B のほうが"), "CV A 0.04・B 0.08")
d = 60.0; t = d / 60 + d / 40
E.choice("s03p02b", f(2 * d / t, 0), f"往復の距離 ÷ 時間 = {2 * d / t}")
E.check("s03p02b", opts("s03p02b") == {"48", "50", f(sqrt(2400), 1), "100", "24"}, "誤答の由来（幾何平均 48.99）")
a = rng.normal(0, 1, 10); a = (a - a.mean()) / a.std() * 4 + 50
b = rng.normal(0, 1, 15); b = (b - b.mean()) / b.std() * 5 + 60
both = np.r_[a, b]
E.choice("s03p02c", f"平均 {f(both.mean(), 0)}、分散 {f(both.var(), 1)}", f"実データを作って計算 平均 {both.mean():.3f} 分散 {both.var():.3f}")
E.check("s03p02c", opts("s03p02c") == {"平均 56、分散 45.4", "平均 55、分散 20.5", "平均 56、分散 21.4", "平均 56、分散 24.0", "平均 55、分散 45.4"} and abs((10 * 16 + 15 * 25) / 25 - 21.4) < 1e-9 and abs((10 * 36 + 15 * 16) / 25 - 24) < 1e-9, "誤答の由来")
# ---- 大問3
r = {"A": (81 / 90, 15 / 30, 96 / 120), "B": (19 / 20, 44 / 80, 63 / 100)}
E.check("s03p03a", r["B"][0] > r["A"][0] and r["B"][1] > r["A"][1] and r["A"][2] > r["B"][2] and ans("s03p03a").startswith("軽症・重症のどちらでも B"), f"回復率 {r}")
# ---- 大問4
xs = np.arange(0, 20); ys = np.exp(2.0 + 0.10 * xs)
E.check("s03p04a", np.allclose(ys[1:] / ys[:-1], exp(0.1)) and abs(exp(0.1) - 1.105) < 5e-4 and ans("s03p04a").startswith("約 1.105 倍"), "隣り合う予測値の比は一定で e^0.1")
# ---- 大問5
E.check("s03p05a", ans("s03p05a") == f((1.05 / 1.02 - 1) * 100, 1) + "%", "名目 ÷ 物価 = 2.9%（−2.9% は比を逆にした誤答）")
E.check("s03p05a", opts("s03p05a") == {f((1.05 / 1.02 - 1) * 100, 1) + "%", "3.0%", f((1.05 * 1.02 - 1) * 100, 1) + "%", f((1.02 / 1.05 - 1) * 100, 1) + "%", "7.0%"}, "誤答の由来")
m = np.arange(240); sales = 100 + 0.1 * m + 15 * np.sin(2 * np.pi * m / 12) + rng.normal(0, 3, 240)
acf = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
E.check("s03p05b", acf(sales, 12) > 0.7 and acf(sales, 24) > 0.7 and acf(sales, 6) < 0 and ans("s03p05b") == "12 か月周期の季節変動がある", f"季節変動のある系列の自己相関 ラグ12 {acf(sales, 12):.2f}・ラグ6 {acf(sales, 6):.2f}")
# ---- 大問6
picked = list(range(23, 5001, 50))
E.check("s03p06a", len(picked) == 100 and picked[2] == 123 and ans("s03p06a") == "123 番" and opts("s03p06a") == {"123 番", "73 番", "69 番", "173 番", "150 番"}, f"系統抽出 {picked[:4]}…（100 人）")
pop = rng.random(100000) < 0.4; respond = rng.random(100000) < np.where(pop, 0.45, 0.20)
E.check("s03p06b", abs(pop[respond].mean() - pop.mean()) > 0.1 and ans("s03p06b").startswith("回答した人と回答しなかった人で"), f"賛成の人ほど回答しやすいと、回答者の賛成率 {pop[respond].mean():.2f}（全体 {pop.mean():.2f}）")
# ---- 大問7
pairs = list(combinations("ABCDE", 2))
E.choice("s03p07a", f(sum("A" in p for p in pairs) / len(pairs), 1), "10 組の総当たり")
E.check("s03p07a", opts("s03p07a") == {"0.4", "0.2", "0.1", "0.5", "0.25"}, "誤答の値")
E.choice("s03p07b", f(8 / 14, 3), "遅刻 14 人のうち自転車 8 人")
E.check("s03p07b", opts("s03p07b") == {f(8 / 14, 3), f(8 / 80, 3), f(8 / 200, 3), f(80 / 200, 3), f(14 / 200, 3)}, "誤答の由来")
# ---- 大問8
ex = integrate.quad(lambda x: x * 2 * x, 0, 1)[0]; ph = integrate.quad(lambda x: 2 * x, 0, 0.5)[0]
E.check("s03p08a", abs(ex - 2 / 3) < 1e-12 and abs(ph - 0.25) < 1e-12 and ans("s03p08a") == "$E[X]=2/3$、$P(X\\le 0.5)=0.25$", f"数値積分 E={ex:.4f}, P={ph:.4f}")
# ---- 大問9
U = stats.uniform(0, 10)
E.choice("s03p09a", f"期待値 {f(U.mean(), 0)}、分散 {f(U.var(), 2)}", "scipy の uniform")
E.check("s03p09a", opts("s03p09a") == {"期待値 5、分散 8.33", f"期待値 5、分散 {f(U.std(), 2)}", "期待値 5、分散 100", "期待値 5、分散 10", f"期待値 5、分散 {f(10 / 12, 2)}"}, "誤答の由来")
H = stats.hypergeom(10, 3, 2)
E.choice("s03p09b", f(H.sf(0), 3), "scipy の hypergeom")
E.check("s03p09b", opts("s03p09b") == {f(H.sf(0), 3), f(H.pmf(1), 3), f(H.pmf(2), 3), f(1 - 0.7 ** 2, 3), "0.300"}, "誤答の由来")
S = rng.normal(50, 3, 2_000_000) + rng.normal(30, 4, 2_000_000)
E.choice("s03p09c", f(stats.norm.sf(2), 4), f"シミュレーション {np.mean(S > 90):.4f}")
E.check("s03p09c", abs(np.mean(S > 90) - 0.0228) < 0.0005 and abs(stats.norm.sf(10 / 7) - 0.0764) < 5e-4 and abs(stats.norm.sf(0.4) - 0.3446) < 5e-5 and opts("s03p09c") == {"0.0228", "0.0764", "0.3446", "0.9772", "0.0456"}, "表の値と誤答の由来")
# ---- 大問10
for n in (100, 10000, 1000000):
    h = rng.binomial(n, 0.5)
E.check("s03p10a", ans("s03p10a").startswith("投げる回数を増やすと、表の出る割合"), "大数の法則")
heads = rng.binomial(1, 0.5, (2000, 10000)).cumsum(1)
gap100, gap10000 = np.abs(2 * heads[:, 99] - 100).mean(), np.abs(2 * heads[:, 9999] - 10000).mean()
E.check("s03p10a", gap10000 > gap100 and np.abs(heads[:, 9999] / 10000 - 0.5).mean() < np.abs(heads[:, 99] / 100 - 0.5).mean(), f"表と裏の差の平均 n=100 {gap100:.1f}・n=10000 {gap10000:.1f}（大きくなる）、割合のずれは小さくなる")
W = (rng.standard_normal((500000, 8)) ** 2).sum(1)
E.choice("s03p10b", "期待値 8、分散 16", f"シミュレーション E={W.mean():.3f}, V={W.var():.3f}")
E.check("s03p10b", abs(W.mean() - 8) < 0.03 and abs(W.var() - 16) < 0.2 and stats.chi2(8).var() == 16, "シミュレーションと scipy")
tot = rng.uniform(50 - 2 * sqrt(3), 50 + 2 * sqrt(3), (200000, 100)).sum(1)
E.choice("s03p10c", f(stats.norm.sf(2), 4), f"一様分布の部品でのシミュレーション {np.mean(tot > 5040):.4f}")
E.check("s03p10c", abs(np.mean(tot > 5040) - 0.0228) < 0.002 and abs(stats.norm.sf(0.2) - 0.4207) < 5e-5 and opts("s03p10c") == {"0.0228", "0.4207", "0.9772", "0.0456", "0.0000"}, "表の値と誤答の由来")
# ---- 大問11
x = np.array([4, 7, 8, 9, 12.])
E.choice("s03p11a", f"母平均 {f(x.mean(), 0)}、母分散 {f(x.var(ddof=1), 1)}", "標本平均と不偏分散")
E.check("s03p11a", opts("s03p11a") == {"母平均 8、母分散 8.5", f"母平均 8、母分散 {f(x.var(), 1)}", "母平均 8、母分散 34", f"母平均 8、母分散 {f(x.std(ddof=1), 2)}", "母平均 8、母分散 4.0"}, "誤答の由来")
hit = 0
for _ in range(20):
    sm = rng.normal(50, 5, (5000, 25)).mean(1); hit += np.sum(np.abs(sm - 50) <= 1.96)
E.check("s03p11b", abs(hit / 100000 - 0.95) < 0.005 and ans("s03p11b").startswith("同じ方法で標本を取り直して"), f"区間を 10 万回作ると母平均を含む割合 {hit / 100000:.4f}")
E.choice("s03p11c", f(4.0 * sqrt(100 / 400), 1), "幅は 1/√n に比例")
E.check("s03p11c", opts("s03p11c") == {"2.0", "1.0", "16.0", "8.0", "4.0"}, "誤答の値")
z0 = atanh(0.6)
want = {iv(tanh(z0 - 1.96 * .2), tanh(z0 + 1.96 * .2), 2), iv(0.6 - 1.96 * .2, 0.6 + 1.96 * .2, 2), iv(z0 - 1.96 * .2, z0 + 1.96 * .2, 2),
        iv(tanh(z0 - 1.96 / sqrt(28)), tanh(z0 + 1.96 / sqrt(28)), 2), iv(tanh(z0 - 1.645 * .2), tanh(z0 + 1.645 * .2), 2)}
E.choice("s03p11d", iv(tanh(z0 - 1.96 * .2), tanh(z0 + 1.96 * .2), 2), "z 変換の区間を tanh で戻す")
E.check("s03p11d", opts("s03p11d") == want and abs(z0 - 0.693) < 5e-4, "誤答の由来と tanh⁻¹0.6")
E.check("s03p11d", all(abs(tanh(a_) - b_) < 5e-4 for a_, b_ in ((0.301, 0.292), (1.085, 0.795), (0.323, 0.312), (1.064, 0.787), (0.364, 0.349), (1.022, 0.771))), "問題文の tanh の値")
cov = 0
for _ in range(4000):
    zz = rng.multivariate_normal([0, 0], [[1, .5], [.5, 1]], 28); rr = np.corrcoef(zz.T)[0, 1]
    cov += tanh(atanh(rr) - 0.392) <= 0.5 <= tanh(atanh(rr) + 0.392)
E.check("s03p11d", abs(cov / 4000 - 0.95) < 0.015, f"ρ=0.5 での被覆率 {cov / 4000:.3f}")
# ---- 大問12
E.check("s03p12a", ans("s03p12a") == "$|Z|>2.576$" and abs(stats.norm.isf(0.005) - 2.576) < 5e-4 and abs(stats.norm.isf(0.01) - 2.326) < 5e-4, "両側 1% は上側 0.5% 点")
chi = 9 * 30 / 16
E.choice("s03p12b", f"χ² ＝ {chi} で、16.92 より小さいので棄却しない", f"χ²={chi}, p={stats.chi2.sf(chi, 9):.4f}")
E.check("s03p12b", abs(stats.chi2.isf(.05, 9) - 16.92) < 0.005 and abs(stats.chi2.isf(.95, 9) - 3.325) < 5e-4 and abs(stats.chi2.isf(.05, 10) - 18.31) < 0.005 and 10 * 30 / 16 == 18.75 and 9 * 16 / 30 == 4.8, "表の値と誤答の由来")
cut = 50 + 1.645 * 2
mm = rng.normal(56, 10, (400000, 25)).mean(1)
E.choice("s03p12c", f(stats.norm.sf((cut - 56) / 2), 3), f"シミュレーション {np.mean(mm > cut):.4f}")
E.check("s03p12c", abs(np.mean(mm > cut) - 0.912) < 0.003 and abs(stats.norm.sf(1.355) - 0.0877) < 5e-5 and abs(stats.norm.sf((50 + 1.96 * 2 - 56) / 2) - 0.851) < 5e-4 and abs(stats.norm.sf(1.04) - 0.1492) < 5e-5, "表の値と誤答 0.851 の由来")
# ---- 大問13
tw = stats.ttest_ind_from_stats(36, sqrt(40), 8, 30, sqrt(10), 10, equal_var=False).statistic
tp = stats.ttest_ind_from_stats(36, sqrt(40), 8, 30, sqrt(10), 10, equal_var=True).statistic
E.choice("s03p13a", f(tw, 2), f"scipy（ウェルチ）{tw:.4f}")
E.check("s03p13a", opts("s03p13a") == {f(tw, 2), f(tp, 2), f(6 / sqrt(50), 2), f(6 / (sqrt(5) + 1), 2), "6.00"}, "誤答の由来（プール t は scipy）")
pp = 210 / 400
zp = 0.15 / sqrt(pp * (1 - pp) * 0.01)
E.choice("s03p13b", f(zp, 2), f"z={zp:.4f}")
r2 = stats.chi2_contingency(np.array([[120, 80], [90, 110]]), correction=False)
E.check("s03p13b", abs(r2[0] - zp ** 2) < 1e-9 and opts("s03p13b") == {f(zp, 2), f(0.15 / sqrt(.6 * .4 / 200 + .45 * .55 / 200), 2), f(0.15 / sqrt(pp * (1 - pp) / 200), 2), f(0.15 / (pp * (1 - pp) * 0.01), 1), f(0.15 / sqrt(pp * (1 - pp)), 2)}, "2×2 表のカイ二乗 = z²・誤答の由来")
r3 = stats.chi2_contingency(np.array([[24, 36, 40], [36, 24, 40]]), correction=False)
E.choice("s03p13c", f"χ² ＝ {f(r3[0], 1)} で、自由度 {r3[2]} の点 5.991 より小さいので棄却しない", f"χ²={r3[0]}, df={r3[2]}, p={r3[1]:.3f}")
E.check("s03p13c", abs(stats.chi2.isf(.05, 2) - 5.991) < 5e-4 and abs(4 * 36 / 900 - 0.16) < 1e-12 and 4 * 36 == 144, "表の値と誤答の由来")
# ---- 大問14
n = 60; h_ = rng.normal(170, 7, n); leg = 0.45 * h_ + rng.normal(0, 1, n); w = 0.5 * h_ + 0.3 * leg + rng.normal(0, 5, n)
def se_coef(X, y):
    A = np.c_[np.ones(len(y)), X]; beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = ((y - A @ beta) ** 2).sum() / (len(y) - A.shape[1]); return np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
E.check("s03p14a", np.corrcoef(h_, leg)[0, 1] > 0.9 and se_coef(np.c_[h_, leg], w)[1] > 2 * se_coef(h_[:, None], w)[1] and ans("s03p14a").startswith("係数の標準誤差が大きくなり"), "相関の強い変数を加えると身長の係数の標準誤差が 2 倍以上")
adj = lambda r2_, p_: 1 - (1 - r2_) * 24 / (24 - p_)
E.check("s03p14b", f(adj(0.700, 2), 3) == "0.673" and f(adj(0.705, 3), 3) == "0.663" and ans("s03p14b").startswith("$x_3$ を加えると"), "表の自由度調整済み決定係数を再計算")
xx = rng.uniform(0, 10, 30); yy = 1 + 0.5 * xx + rng.normal(0, 1, 30); b1, b0 = np.polyfit(xx, yy, 1)
s = np.sqrt(((yy - b0 - b1 * xx) ** 2).sum() / 28); Sxx = ((xx - xx.mean()) ** 2).sum()
ci = lambda x0: s * sqrt(1 / 30 + (x0 - xx.mean()) ** 2 / Sxx); pi = lambda x0: s * sqrt(1 + 1 / 30 + (x0 - xx.mean()) ** 2 / Sxx)
E.check("s03p14c", pi(xx.mean()) > ci(xx.mean()) and ci(10) > ci(xx.mean()) and pi(10) > pi(xx.mean()) and ans("s03p14c").startswith("予測区間は信頼区間より広く"), "標準誤差の式で大小と変化を確認")
# ---- 大問15
mean = {("A1", "B1"): 50, ("A1", "B2"): 60, ("A2", "B1"): 55, ("A2", "B2"): 85}
E.check("s03p15a", (mean[("A2", "B2")] - mean[("A2", "B1")]) - (mean[("A1", "B2")] - mean[("A1", "B1")]) == 20 and ans("s03p15a").startswith("肥料 B2 の効果は A1 では 10"), "B2 の効果の差 20")
E.check("s03p15b", ans("s03p15b").startswith("誤差の大きさを評価し"), "反復の目的")
E.done()
