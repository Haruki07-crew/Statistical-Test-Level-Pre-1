"""2級 模擬試験 第6回（g2 set06）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from math import ceil, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(6)
rng = np.random.default_rng(20261029)


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
mix = np.r_[rng.normal(30, 6, 5000), rng.normal(80, 6, 5000)]
E.check("s06p01a", np.mean(np.abs(mix - mix.mean()) < 5) < 0.02 and ans("s06p01a").startswith("平均値は 2 つの山の間"), f"二峰の平均 {mix.mean():.1f} の近くのデータの割合 {np.mean(np.abs(mix - mix.mean()) < 5):.3f}")
A = sorted([52, 58, 61, 63, 67, 70, 74, 75, 81]); B = sorted([45, 49, 55, 56, 60, 62, 66, 72, 88])
E.choice("s06p01b", f"A 組 {np.median(A):.0f} 点、B 組 {np.median(B):.0f} 点", "5 番目の値")
E.check("s06p01b", opts("s06p01b") == {"A 組 67 点、B 組 60 点", f"A 組 {f(np.mean(A), 1)} 点、B 組 {f(np.mean(B), 1)} 点", f"A 組 {A[3]} 点、B 組 {B[3]} 点", f"A 組 {A[5]} 点、B 組 {B[5]} 点", "A 組 67 点、B 組 62 点"}, "誤答の由来")
# ---- 大問2
x = rng.normal(62, 12, 200); x = (x - x.mean()) / x.std(ddof=1) * 12 + 62; z = (x - 62) / 12; t = 50 + 10 * z
E.check("s06p02a", abs(z.mean()) < 1e-9 and abs(z.std(ddof=1) - 1) < 1e-9 and abs(t.mean() - 50) < 1e-9 and abs(t.std(ddof=1) - 10) < 1e-9 and ans("s06p02a").startswith("標準得点は平均 0・標準偏差 1、偏差値は平均 50・標準偏差 10"), "変換して確認")
E.check("s06p02b", abs((1 - 0.6) - 0.4) < 1e-12 and ans("s06p02b").startswith("所得の多い上位 10% の世帯が、所得全体の 40%"), "上位 10% は 1 − 0.6")
# ---- 大問3
xx = rng.normal(0, 1, 1000); yy = 0.6 * xx + 0.8 * rng.normal(0, 1, 1000)
r0 = np.corrcoef(xx, yy)[0, 1]; r1 = np.corrcoef(-2 * xx + 3, yy)[0, 1]
E.check("s06p03a", abs(r1 + r0) < 1e-12 and ans("s06p03a") == "−0.6" and opts("s06p03a") == {"−0.6", "0.6", "−1.2", "1.8", "−0.9"}, f"u=−2x+3 で r が {r0:.3f} → {r1:.3f}")
lin = np.linspace(0, 10, 20)
E.check("s06p03b", abs(np.corrcoef(lin, 5 - 3 * lin)[0, 1] + 1) < 1e-12 and ans("s06p03b").startswith("すべての点が右下がりの直線"), "直線上の点で r=−1")
b_yx = np.polyfit(xx, yy, 1)[0]; b_xy = np.polyfit(yy, xx, 1)[0]
E.check("s06p03c", abs(b_yx * b_xy - r0 ** 2) < 1e-9 and f(sqrt(.8 * .45), 1) == "0.6" and opts("s06p03c") == {"0.6", "0.36", f((.8 + .45) / 2, 3), "1.25", "0.8"}, f"2 本の傾きの積＝r²（乱数データで {b_yx * b_xy:.4f} と {r0 ** 2:.4f}）")
# ---- 大問4
E.check("s06p04a", ans("s06p04a").startswith("季節の影響を除くと前月より少し減った"), "季節調整値と前年同月比の読み方")
w = np.array([.3, .2, .5]); rel = np.array([1.10, 1.00, 1.04])
E.choice("s06p04b", f(w @ rel * 100, 1), "加重平均")
E.check("s06p04b", opts("s06p04b") == {f(w @ rel * 100, 1), f(rel.mean() * 100, 1), "114.0", f(np.array([.2, .3, .5]) @ rel * 100, 1), "110.0"}, "誤答の由来")
# ---- 大問5
E.check("s06p05a", ans("s06p05a").startswith("18 歳以上の市民の名簿から、乱数"), "単純無作為抽出")
smoke = rng.random(200000) < 0.3; coffee = rng.normal(2 + 1.5 * smoke, 1); heart = rng.random(200000) < np.where(smoke, 0.10, 0.03)
corr_all = np.corrcoef(coffee, heart)[0, 1]; corr_in = [np.corrcoef(coffee[smoke == s_], heart[smoke == s_])[0, 1] for s_ in (0, 1)]
E.check("s06p05b", corr_all > 0.05 and max(abs(c) for c in corr_in) < 0.01 and ans("s06p05b").startswith("喫煙が交絡因子"), f"喫煙が両方に影響すると全体 r={corr_all:.3f}、喫煙で層別すると {np.round(corr_in, 3)}")
# ---- 大問6
perms = list(permutations("ABCDE")); adj = sum(abs(p.index("A") - p.index("B")) == 1 for p in perms) / len(perms)
E.choice("s06p06a", f(adj, 1), f"120 通りの総当たり {adj}")
E.check("s06p06a", opts("s06p06a") == {"0.4", "0.2", "0.1", "0.5", "0.8"}, "誤答の値")
spam = rng.random(2_000_000) < .2; word = rng.random(2_000_000) < np.where(spam, .6, .05)
E.choice("s06p06b", f(.12 / .16, 2), f"シミュレーション {spam[word].mean():.4f}")
E.check("s06p06b", abs(spam[word].mean() - .75) < .003 and opts("s06p06b") == {"0.75", "0.60", "0.12", "0.16", "0.25"}, "誤答の値")
# ---- 大問7
a = 1 - .2 - .5; EX = 1 * .2 + 2 * a + 3 * .5
E.check("s06p07a", ans("s06p07a") == f"$a={f(a, 1)}$、$E[X]={f(EX, 1)}$", f"a={a}, E={EX}")
cov = -.5 * 2 * 3; XY = rng.multivariate_normal([0, 0], [[4, cov], [cov, 9]], 2_000_000); port = .5 * XY[:, 0] + .5 * XY[:, 1]
E.choice("s06p07b", f(.25 * 4 + .25 * 9 + 2 * .25 * cov, 2), f"シミュレーション {port.var():.3f}")
E.check("s06p07b", abs(port.var() - 1.75) < 0.01, "シミュレーション")
E.check("s06p07b", opts("s06p07b") == {"1.75", f(.25 * 4 + .25 * 9, 2), f(.5 * 4 + .5 * 9, 2), f(.25 * 4 + .25 * 9 - 2 * .25 * cov, 2), f(.25 * 4 + .25 * 9 + 2 * .25 * (-.5), 2)}, "誤答の由来（独立 3.25・係数を 2 乗しない 6.50・共分散の符号を逆 4.75・相関係数のまま 3.00）")
# ---- 大問8
Pois = rng.poisson(2, (1_000_000, 3)).sum(1)
E.check("s06p08a", abs(Pois.mean() - 6) < 0.01 and abs(Pois.var() - 6) < 0.03 and ans("s06p08a") == "期待値 6、分散 6", "3 時間の合計のシミュレーション")
E.choice("s06p08b", f"{f(10 / stats.norm.isf(.0228), 0)} mm", "σ = 10 / z(0.0228)")
E.check("s06p08b", abs(stats.norm(60, 5).sf(70) - .0228) < 5e-5 and opts("s06p08b") == {"5 mm", "10 mm", f"{f(10 / 1.96, 1)} mm", "2.28 mm", "20 mm"}, "誤答の由来")
# ---- 大問9
E.check("s06p09a", ans("s06p09a").startswith("母集団が正規分布なら標本の大きさによらず"), "標本平均の分布")
E.choice("s06p09b", f(stats.norm.sf(.9), 4), f"正確な二項確率 {stats.binom(100, .5).cdf(45):.4f}")
E.check("s06p09b", opts("s06p09b") == {f(stats.norm.sf(.9), 4), f(stats.norm.sf(1.0), 4), f(stats.norm.sf(1.1), 4), f(1 - .1841, 4), f(2 * .1841, 4)}, "表の値と誤答の由来")
sm = rng.normal(5, 3, (200000, 16)); T = (sm.mean(1) - 5) / (sm.std(1, ddof=1) / 4)
E.choice("s06p09c", "0.05", f"シミュレーション {np.mean(np.abs(T) > 2.131):.4f}")
E.check("s06p09c", abs(np.mean(np.abs(T) > 2.131) - .05) < .002 and f(2 * stats.norm.sf(2.131), 4) == "0.0331", "誤答 0.0331 の由来")
# ---- 大問10
smp = rng.normal(120, 9, (200000, 36))
E.choice("s06p10a", "推定値 120 g、標準誤差 1.5 g", f"標本平均の標準偏差のシミュレーション {smp.mean(1).std():.3f}")
E.check("s06p10a", abs(smp.mean(1).std() - 1.5) < 0.01 and opts("s06p10a") == {"推定値 120 g、標準誤差 1.5 g", "推定値 120 g、標準誤差 9 g", f"推定値 120 g、標準誤差 {f(9 / 36, 2)} g", f"推定値 120 g、標準誤差 {f(81 / 36, 2)} g", f"推定値 {f(120 / 36, 2)} g、標準誤差 1.5 g"}, "誤答の由来")
X1, X2 = rng.normal(3, 2, (2, 2_000_000)); T1 = (X1 + X2) / 2; T2 = (X1 + 2 * X2) / 3
E.check("s06p10b", abs(T1.mean() - 3) < .005 and abs(T2.mean() - 3) < .005 and abs(T1.var() - 2) < .01 and abs(T2.var() - 4 * 5 / 9) < .01 and ans("s06p10b").startswith("どちらも不偏推定量だが"), f"分散 T1 {T1.var():.3f}（σ²/2=2）、T2 {T2.var():.3f}（5σ²/9=2.22）")
need = (1.96 / .04) ** 2 * .16
E.choice("s06p10c", str(ceil(need)), f"必要な n {need:.2f}")
E.check("s06p10c", opts("s06p10c") == {"385", "384", str(ceil((1.96 / .04) ** 2 * .25)), str(ceil(1.96 / .04 * .16)), str(int((.2 / .04) ** 2 * 4))}, "誤答の由来")
E.check("s06p10d", abs(stats.t.isf(.025, 15) - 2.131) < 5e-4 and abs(stats.t.isf(.025, 16) - 2.120) < 5e-4, "表の値")
E.choice("s06p10d", iv(3 - 2.131, 3 + 2.131, 2), "t 区間（自由度 15）")
E.check("s06p10d", opts("s06p10d") == {iv(3 - k * w, 3 + k * w, 2) for k, w in ((2.131, 1), (1.96, 1), (2.120, 1), (2.131, 4), (2.131, sqrt(2)))}, "誤答の由来")
# ---- 大問11
pw = lambda a: stats.norm.sf(stats.norm.isf(a) - 1.5)
E.check("s06p11a", pw(.01) < pw(.05) and ans("s06p11a").startswith("第 1 種の過誤の確率は小さくなり"), f"検出力 α=0.05 {pw(.05):.3f}、0.01 {pw(.01):.3f}")
E.check("s06p11b", abs(stats.t.isf(.05, 24) - 1.711) < 5e-4 and abs(stats.t.isf(.025, 24) - 2.064) < 5e-4 and 3 / (10 / 5) == 1.5 and ans("s06p11b").startswith("t ＝ 1.5 で、1.711 より小さい"), "表の値と t=1.5")
E.choice("s06p11c", f"\\bar x>{f(100 + 1.645 * 2.5, 2)}", "100 + 1.645 × 15/6")
E.check("s06p11c", opts("s06p11c") == {f"$\\bar x>{f(100 + 1.645 * 2.5, 2)}$", f"$\\bar x>{f(100 + 1.96 * 2.5, 2)}$", f"$\\bar x>{f(100 + 1.645 * 15, 2)}$", f"$\\bar x<{f(100 - 1.645 * 2.5, 2)}$", f"$\\bar x>{f(100 + 1.645 * 15 / 36, 2)}$"}, "誤答の由来")
tab = np.array([[45, 55], [25, 75]]); ch = stats.chi2_contingency(tab, correction=False)
E.choice("s06p11d", f"χ² ≈ {f(ch[0], 2)} で、有意水準 1% でも棄却する", f"χ²={ch[0]:.4f}, p={ch[1]:.4f}")
E.check("s06p11d", abs(stats.chi2.isf(.01, 1) - 6.635) < 5e-4 and abs(stats.chi2.isf(.05, 1) - 3.841) < 5e-4 and f(100 / 35 + 100 / 65, 2) == "4.40" and ((tab - ch[3]) ** 2).sum() == 400, "表の値・誤答の由来")
E.check("s06p11e", ans("s06p11e") == "2 つの不偏分散の比を使う F 検定", "分散の比較は F 検定")
cnt = 0
p6 = np.array([.25, .25, .2, .2, .021, .034]); p6 = p6 / p6.sum() if abs(p6.sum() - 1) > 1e-12 else p6
probs = np.array([.3, .25, .2, .195, .021, .034]); n = 100
for _ in range(4000):
    o = rng.multinomial(n, probs); o5 = np.r_[o[:4], o[4] + o[5]]; e5 = n * np.r_[probs[:4], probs[4] + probs[5]]
    cnt += ((o5 - e5) ** 2 / e5).sum() > stats.chi2.isf(.05, 4)
E.check("s06p11f", abs(cnt / 4000 - .05) < .015 and ans("s06p11f").startswith("期待度数の小さい 2 つのカテゴリをまとめ") and abs(n * (probs[4] + probs[5]) - 5.5) < 1e-9, f"まとめて自由度 4 で検定すると棄却の割合 {cnt / 4000:.3f}（期待度数 2.1 と 3.4 → 5.5）")
# ---- 大問12
tv = {"x1": 4.2 / 1.4, "x2": -.9 / .2}
E.check("s06p12a", all(abs(v) > 2.086 for v in tv.values()) and max(tv, key=lambda k: abs(tv[k])) == "x2" and ans("s06p12a") == "$x_2$（t ＝ −4.5）" and abs(stats.t.isf(.025, 20) - 2.086) < 5e-4, f"t 値 {tv}")
E.choice("s06p12b", f(sqrt(200 / 20), 2), "√(200/20)")
E.check("s06p12b", opts("s06p12b") == {f(sqrt(10), 2), f(sqrt(200 / 23), 2), "10.0", f(sqrt(200), 1), f(sqrt(200 / 22), 2)}, "誤答の由来")
P = rng.uniform(100, 300, 5000); Q = np.exp(8 - 1.2 * np.log(P))
E.check("s06p12c", abs(Q[0] * (1.01 ** -1.2) / Q[0] - 1 + 0.0119) < 1e-3 and abs(2 ** -1.2 - .435) < 1e-3 and ans("s06p12c").startswith("価格が 1% 上がると、販売量はおよそ 1.2% 減る"), "1% 上げると 1.01^(−1.2) ≈ 0.988 倍")
# ---- 大問13
E.check("s06p13a", ans("s06p13a") == "自由度 (2, 9) の F 分布", "(k−1, N−k)")
g = [rng.normal(m_, 1, 4) for m_ in (10, 12, 14)]; g = [x_ - x_.mean() + m_ for x_, m_ in zip(g, (10, 12, 14))]
al = np.concatenate(g); SA = sum(4 * (x_.mean() - al.mean()) ** 2 for x_ in g)
E.choice("s06p13b", f(SA, 0), f"群の平均が 10・12・14 のデータで計算 {SA}")
E.check("s06p13b", opts("s06p13b") == {"32", "8", "16", "128", "4"}, "誤答の値")
E.done()
