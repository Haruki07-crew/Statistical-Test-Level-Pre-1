"""2級 模擬試験 第8回（g2 set08）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from math import sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(8)
rng = np.random.default_rng(20261031)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


E.check("s08p01a", ans("s08p01a").startswith("国勢調査は日本に住むすべての人"), "国勢調査は全数調査・5 年ごと")
cum = np.cumsum([100, 50, 30, 20]) / 200
E.check("s08p01b", list(cum) == [.5, .75, .9, 1.0] and ans("s08p01b").startswith("A・B・C の 3 つ"), f"累積割合 {cum}")
E.check("s08p02a", f(5 / 110 * 100, 1) == "4.5" and f(7 / 165 * 100, 1) == "4.2" and ans("s08p02a").startswith("1 年生は約 4.5%、大人は約 4.2% で、1 年生"), "変動係数")
ls = 100 - rng.lognormal(2.5, .6, 100000)
E.check("s08p02b", ls.mean() < np.median(ls) and stats.skew(ls) < 0 and ans("s08p02b").startswith("左に裾が長く、平均値は中央値より小さく"), "左に裾の長い分布で 平均＜中央値・歪度＜0")
R = {("国語", "数学"): .15, ("数学", "理科"): -.71, ("国語", "英語"): .62, ("英語", "理科"): .30, ("国語", "理科"): .21, ("英語", "数学"): .33}
E.check("s08p03a", min(R, key=lambda k: abs(R[k])) == ("国語", "数学") and ans("s08p03a") == "国語と数学", "絶対値最小")
rates = {"X": (48 / 80, 14 / 20), "Y": (4 / 20, 20 / 80)}; tot = (52 / 100, 34 / 100)
E.check("s08p03b", all(w > m for m, w in rates.values()) and tot[0] > tot[1] and ans("s08p03b").startswith("どちらの学部でも女性の合格率のほうが高いが"), f"学部別 {rates}、全体 {tot}")
x = np.linspace(-3, 3, 200); y = x ** 2 + rng.normal(0, .3, 200); b, a = np.polyfit(x, y, 1); res = y - a - b * x
E.check("s08p04a", res[np.abs(x) > 2].mean() > 0 > res[np.abs(x) < 1].mean() and ans("s08p04a").startswith("関係が直線的でない可能性"), "2 次の関係に直線を当てはめると残差が U 字形")
wn = rng.normal(0, 1, 2000); ac = [np.corrcoef(wn[:-k], wn[k:])[0, 1] for k in range(1, 15)]
E.check("s08p04b", max(abs(v) for v in ac) < .08 and ans("s08p04b").startswith("時点どうしの関連がほとんどなく"), f"ホワイトノイズの自己相関の最大 {max(abs(v) for v in ac):.3f}")
E.choice("s08p04c", f"{f((sqrt(1.6 * .9) - 1) * 100, 0)}%", "√(1.6 × 0.9) − 1")
E.check("s08p04c", opts("s08p04c") == {"20%", "25%", "50%", f"{f((1.6 * .9 - 1) * 100, 0)}%", "22%"}, "誤答の由来")
schools = [rng.normal(rng.normal(8, 1), .2, 30) for _ in range(200)]; allk = np.concatenate(schools)
clus = [np.concatenate([schools[k] for k in rng.choice(200, 4, replace=False)]).mean() for _ in range(3000)]
srs = [rng.choice(allk, 120, replace=False).mean() for _ in range(3000)]
E.check("s08p05a", np.std(clus) > 3 * np.std(srs) and ans("s08p05a").startswith("同じ学校の児童どうしの睡眠時間がよく似ていて"), f"学校の中が似ているとき 標準誤差 集落 {np.std(clus):.3f}・単純 {np.std(srs):.3f}")
E.check("s08p05b", ans("s08p05b").startswith("督促などで回収率を上げ"), "無回答への対策")
mon = rng.integers(0, 12, (1_000_000, 3)); same = (mon[:, 0] == mon[:, 1]) | (mon[:, 0] == mon[:, 2]) | (mon[:, 1] == mon[:, 2])
E.choice("s08p06a", f(1 - 12 * 11 * 10 / 12 ** 3, 3), f"シミュレーション {same.mean():.4f}")
E.check("s08p06a", opts("s08p06a") == {"0.236", f(1320 / 1728, 3), "0.250", f(1 / 12, 3), f(1 / 144, 3)}, "誤答の由来")
E.choice("s08p06b", f(36 / 90, 2), "利用者 90 人のうち 20 代 36 人")
E.check("s08p06b", opts("s08p06b") == {"0.40", f(90 / 300, 2), f(36 / 300, 2), "0.33", f(18 / 90, 2)}, "誤答の由来")
J = np.array([[.1, .2, .1], [.2, .1, .3]]); py = J.sum(0); EY = py @ [0, 1, 2]
E.choice("s08p06c", f(EY, 1), f"周辺確率 {py}")
E.check("s08p06c", opts("s08p06c") == {f(EY, 1), f(.3 + 2 * .3, 1), "1.0", f(J[1] @ [0, 1, 2], 1), f(J.sum(1) @ [1, 2], 1)}, "誤答の由来")
X = rng.normal(0, 2, 2_000_000); Y = rng.normal(0, 3, 2_000_000); rr = np.corrcoef(X, X + Y)[0, 1]
E.choice("s08p06d", f(4 / (2 * sqrt(13)), 2), f"シミュレーション {rr:.4f}")
E.check("s08p06d", abs(rr - .555) < .003 and opts("s08p06d") == {"0.55", f(4 / 13, 2), "0", f(2 / 3, 2), "1"}, "誤答の由来")
B = stats.binom(10, .3)
E.check("s08p07a", int(np.argmax([B.pmf(k) for k in range(11)])) == 3 and ans("s08p07a") == "3 回", "確率最大は 3 回")
E.choice("s08p07b", f(stats.hypergeom(50, 10, 5).mean(), 1), "scipy の hypergeom")
E.check("s08p07b", opts("s08p07b") == {"1.0", "0.2", "2.0", f(5 * .2 * .8, 1), "5.0"}, "誤答の由来")
E.check("s08p08a", ans("s08p08a") == "9 倍", "√9 = 3")
for d in (stats.norm(0, 1), stats.uniform(-sqrt(3), 2 * sqrt(3)), stats.laplace(0, 1 / sqrt(2)), stats.t(5, scale=sqrt(3 / 5))):
    E.check("s08p08b", d.cdf(3) - d.cdf(-3) >= 1 - 1 / 9, f"{d.dist.name}（標準偏差 1）で ±3 以内 {d.cdf(3) - d.cdf(-3):.4f} ≥ 0.889")
E.check("s08p08b", ans("s08p08b") == "約 88.9%", "1 − 1/9")
ratio = rng.normal(0, 1, (200000, 10)).var(1, ddof=1) / rng.normal(0, 1, (200000, 10)).var(1, ddof=1)
E.choice("s08p08c", "0.05", f"シミュレーション {np.mean(ratio > 3.179):.4f}")
E.check("s08p08c", abs(np.mean(ratio > 3.179) - .05) < .002 and abs(stats.f.isf(.05, 9, 9) - 3.179) < 5e-4, "表の値")
E.choice("s08p09a", f((200 - 5 * 6 ** 2) / 4, 1), "(Σx² − n x̄²)/(n−1)")
E.check("s08p09a", opts("s08p09a") == {"5.0", "4.0", "20.0", "40.0", f(sqrt(5), 2)}, "誤答の由来")
se1 = sqrt(.2475 / 400); se = sqrt(2) * se1
E.choice("s08p09b", iv(.1 - 1.96 * se, .1 + 1.96 * se, 3), f"標準誤差 {se:.5f}")
E.check("s08p09b", opts("s08p09b") == {iv(.1 - k * w, .1 + k * w, 3) for k, w in ((1.96, se), (1.645, se), (1.96, se1), (1.96, 2 * se1), (1.96, 2 * .2475 / 400))}, "誤答の由来")
E.check("s08p09c", abs(stats.chi2.isf(.025, 15) - 27.49) < .005 and abs(stats.chi2.isf(.975, 15) - 6.262) < 5e-4, "表の値")
E.choice("s08p09c", iv(135 / 27.49, 135 / 6.262, 2), "(n−1)s²/χ²")
E.check("s08p09c", opts("s08p09c") == {iv(135 / 27.49, 135 / 6.262, 2), iv(144 / 27.49, 144 / 6.262, 2), iv(9 / 27.49, 9 / 6.262, 2), f"({f(135 / 6.262, 2)}, {f(135 / 27.49, 2)})", iv(9 - 1.96 * sqrt(2 / 15) * 9, 9 + 1.96 * sqrt(2 / 15) * 9, 2)}, "誤答の由来")
E.choice("s08p09d", f(50 + 1.645 * 2, 2), "片側の信頼上限")
mm = rng.normal(50, 10, (200000, 25)).mean(1)
E.check("s08p09d", abs(np.mean(50 <= mm + 1.645 * 2) - .95) < .002 and opts("s08p09d") == {"53.29", f(50 + 1.96 * 2, 2), f(50 + 1.645 * 10, 2), f(50 - 1.645 * 2, 2), f(50 + 1.645 * 10 / 25, 2)}, "片側の被覆率 95%・誤答の由来")
E.check("s08p10a", ans("s08p10a") == "帰無仮説 $\\mu=50$、対立仮説 $\\mu\\neq 50$", "両側")
pw = lambda d: stats.norm.sf(1.96 - d) + stats.norm.cdf(-1.96 - d)
E.check("s08p10b", pw(2) > pw(1) > pw(0.5) and abs(pw(0) - .05) < 1e-4 and ans("s08p10b").startswith("真の平均が 50 から離れているほど、検出力は大きい"), f"検出力 δ=0.5 {pw(.5):.3f}・1 {pw(1):.3f}・2 {pw(2):.3f}")
z = 1.5 / (8 / 8)
E.choice("s08p10c", f"p 値 {f(2 * .0668, 4)} で、棄却しない", f"z={z}, 正確な p={2 * stats.norm.sf(z):.4f}")
E.check("s08p10c", abs(stats.norm.sf(1.5) - .0668) < 5e-5, "表の値")
F = 45 / 12
E.choice("s08p10e", f"F ＝ {f(F, 2)} で、$F_{{0.025}}(9,11)=3.588$ より大きいので棄却する", f"F={F}, 両側 p={2 * stats.f.sf(F, 9, 11):.4f}")
E.check("s08p10e", abs(stats.f.isf(.025, 9, 11) - 3.588) < 5e-4 and abs(stats.f.isf(.025, 11, 9) - 3.912) < 5e-4 and abs(stats.f.isf(.05, 9, 11) - 2.896) < 5e-4 and f(sqrt(F), 2) == "1.94" and f(12 / 45, 2) == "0.27", "表の値と誤答の由来")
XY = rng.multivariate_normal([0, 0], [[100, 80], [80, 100]], 2_000_000); dd = XY[:, 1] - XY[:, 0]
E.choice("s08p10f", f(sqrt(40), 1), f"シミュレーション {dd.std():.3f}")
E.check("s08p10f", abs(dd.std() - sqrt(40)) < .02 and opts("s08p10f") == {"6.3", f(sqrt(200), 1), "20.0", "0", f(sqrt(360), 1)}, "誤答の由来")
pp = .25; zz = .1 / sqrt(pp * (1 - pp) * .01)
E.choice("s08p10g", f"z ≈ {f(zz, 2)} で、有意水準 5% では棄却するが、1% では棄却しない", f"z={zz:.4f}")
E.check("s08p10g", stats.norm.isf(.05) < zz < stats.norm.isf(.01) and abs(stats.norm.isf(.01) - 2.326) < 5e-4, "1.645 < z < 2.326")
E.check("s08p11a", [k for k, p_ in zip("ABC", (.004, .03, .27)) if p_ < .01] == ["A"] and ans("s08p11a") == "A だけ", "p<0.01 は A だけ")
xs = rng.normal(0, 1, 22); xs = (xs - xs.mean()) / np.sqrt(((xs - xs.mean()) ** 2).sum()) * 10
ys = 2 * xs + rng.normal(0, 4, 22); bb, aa = np.polyfit(xs, ys, 1); rs = ys - aa - bb * xs; s_ = np.sqrt((rs ** 2).sum() / 20)
E.check("s08p11b", abs(s_ / np.sqrt((xs ** 2).sum()) - s_ / 10) < 1e-9 and ans("s08p11b") == "標準誤差 0.4、t 値 5.0", "SE = s/√Sxx（Sxx=100 のデータで確認）")
xo = np.r_[rng.normal(5, 1, 20), 20]; yo = np.r_[1 + .5 * xo[:20] + rng.normal(0, .5, 20), 0]
b_with = np.polyfit(xo, yo, 1)[0]; b_wo = np.polyfit(xo[:20], yo[:20], 1)[0]
E.check("s08p11c", abs(b_with - b_wo) > .3 and ans("s08p11c").startswith("その点が結果に大きな影響を与えている"), f"離れた 1 点で傾き {b_wo:.2f} → {b_with:.2f}")
E.check("s08p12a", ans("s08p12a") == "交互作用を誤差と分けて検定することができない", "繰り返しのない二元配置")
E.choice("s08p12b", f(4 * (4 + 6 + 8), 0), "Σ (n−1) s²")
E.check("s08p12b", opts("s08p12b") == {"72", "18", "90", "6", "30"}, "誤答の値")
E.done()
