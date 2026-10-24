"""2級 模擬試験 第15回（g2 set15）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import combinations, product
from math import exp, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(15)
rng = np.random.default_rng(20261115)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# 第1問
E.check("s15p01a", np.digitize([20], [10, 20, 30])[0] == 2 and ans("s15p01a") == "20 以上 30 未満", "np.digitize（左端を含む）で 20 は 2 番目の境目以上の階級")
tab = np.array([[60, 40], [30, 70]]); width = tab.sum(1) / tab.sum(); height = tab / tab.sum(1, keepdims=True)
E.check("s15p01b", np.allclose(width[:, None] * height, tab / tab.sum()) and ans("s15p01b").startswith("柱の横幅は男性と女性の人数の割合を"), "横幅 × 条件付きの割合 ＝ セルの割合（面積）")

# 第2問
x = np.array([1, 2, 4, 10, 13.]); cs = np.linspace(0, 14, 1401); sabs = [np.abs(x - c).sum() for c in cs]
E.check("s15p02a", abs(cs[int(np.argmin(sabs))] - 4) < .011 and min(sabs) == 20 and ans("s15p02a") == "c ＝ 4（中央値）のとき 20" and np.abs(x - 6).sum() == 22 and np.abs(x - 7).sum() == 23, f"最小の c {cs[int(np.argmin(sabs))]:.2f}・最小値 {min(sabs)}")
z = rng.normal(0, 1, 5); d = 6 + 2 * (z - z.mean()) / z.std()
E.check("s15p02b", abs(d.mean() - 6) < 1e-9 and abs((d ** 2).mean() - 40) < 1e-9 and abs(d.var() - 4) < 1e-9 and ans("s15p02b") == "4" and opts("s15p02b") == {"4", f"{40 - 6:g}", "2", f"{4 * 5 / 4:g}", "40"}, "平均 6・2 乗の平均 40 のデータで分散 4")

# 第3問
yrs = np.arange(30); a = 100 + 5 * yrs + rng.normal(0, 3, 30); b = 70 + .3 * yrs + rng.normal(0, .3, 30)
E.check("s15p03a", np.corrcoef(a, b)[0, 1] > .9 and abs(np.corrcoef(np.diff(a), np.diff(b))[0, 1]) < .5 and ans("s15p03a").startswith("どちらにもトレンドがあるので、前年からの変化"), f"無関係な 2 系列：そのまま {np.corrcoef(a, b)[0, 1]:.2f}・差分 {np.corrcoef(np.diff(a), np.diff(b))[0, 1]:.2f}")
xx = rng.normal(0, 1, 2); yy = rng.normal(0, 1, 2); bb, aa = np.polyfit(xx, yy, 1); r2 = 1 - ((yy - aa - bb * xx) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
E.check("s15p03b", abs(r2 - 1) < 1e-9 and ans("s15p03b").startswith("2 点を通る直線は必ず引けるので"), "乱数の 2 点で R² = 1")

# 第4問
t = np.arange(144); ser = 100 + 2 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 144)
def acf(v, k):
    v = v - v.mean(); return (v[:-k] * v[k:]).sum() / (v ** 2).sum()
raw = [acf(ser, k) for k in (6, 12)]; dif = np.diff(ser); dd = [acf(dif, k) for k in (6, 12)]
E.check("s15p04a", raw[0] > .7 and raw[1] > .7 and dd[0] < -.5 and dd[1] > .5 and ans("s15p04a").startswith("トレンドがあるとどのラグでも自己相関が大きくなり"), f"そのまま ラグ 6：{raw[0]:.2f}・12：{raw[1]:.2f}／差分 ラグ 6：{dd[0]:.2f}・12：{dd[1]:.2f}")
ch = 100 * 1.10 * 1.20 * .90
E.check("s15p04b", ans("s15p04b") == f(ch, 1) and opts("s15p04b") == {f(ch, 1), f"{100 + 10 + 20 - 10:g}", f"約 {f((1.1 + 1.2 + .9) / 3 * 100, 1)}", f"{90:g}", f"{1.1 * 1.2 * 100:g}"}, f"100 × 1.1 × 1.2 × 0.9 = {ch:.1f}・誤答の由来")
E.check("s15p04c", abs(exp(.1) - 1.105) < 5e-4 and ans("s15p04c") == f"約 {f((exp(.1) - 1) * 100, 1)}%" and opts("s15p04c") == {f"約 {f((exp(.1) - 1) * 100, 1)}%", "10.0%", f"約 {f((1 - exp(-.1)) * 100, 1)}%", f"約 {f((10 ** .1 - 1) * 100, 1)}%", "0.1%"}, "e^0.1 − 1 = 10.5%・誤答の由来")

# 第5問
rej_ind = rej_cls = 0
for _ in range(3000):
    cls = rng.normal(0, 1, 10); scores = cls[:, None] + rng.normal(0, 1, (10, 30)); trt = rng.permutation(10) < 5
    rej_ind += stats.ttest_ind(scores[trt].ravel(), scores[~trt].ravel()).pvalue < .05
    rej_cls += stats.ttest_ind(scores[trt].mean(1), scores[~trt].mean(1)).pvalue < .05
E.check("s15p05a", rej_ind / 3000 > .3 and abs(rej_cls / 3000 - .05) < .015 and ans("s15p05a").startswith("同じクラスの生徒は担任や雰囲気が共通で"), f"教材の効果がないとき：300 人として棄却率 {rej_ind / 3000:.2f}・クラス平均で {rej_cls / 3000:.3f}")
E.check("s15p05b", ans("s15p05b").startswith("無作為抽出による標本誤差は確率の理論で"), "標本誤差と非標本誤差")

# 第6問
faces = [("R", "R"), ("W", "W"), ("R", "W")]; seen = [(c, s) for c in faces for s in range(2)]
red = [(c, s) for c, s in seen if c[s] == "R"]; pr = Fr(sum(c[1 - s] == "R" for c, s in red), len(red))
E.check("s15p06a", pr == Fr(2, 3) and ans("s15p06a") == "2/3", f"6 面の総当たり：赤の面 {len(red)}・裏も赤 {sum(c[1 - s] == 'R' for c, s in red)}")
X = rng.choice([1, 4], 1_000_000)
E.check("s15p06b", abs((1 / X).mean() - .625) < .002 and abs(1 / X.mean() - .4) < .002 and ans("s15p06b") == "0.625" and opts("s15p06b") == {"0.625", f"{1 / 2.5:g}", "2.5", f"{1 + .25:g}", "0.25"}, f"シミュレーション E[1/X] {(1 / X).mean():.4f}・1/E[X] {1 / X.mean():.4f}")
e2 = integrate.quad(lambda u: u * u, 0, 1)[0]
E.check("s15p06c", abs(e2 - 1 / 3) < 1e-12 and ans("s15p06c") == "1/3" and opts("s15p06c") == {"1/3", "1/4", "1/2", "1/12", "1"}, "数値積分 1/3")

# 第7問
b = stats.binom(5, .3)
E.check("s15p07a", ans("s15p07a") == f"約 {f(b.sf(3), 3)}" and opts("s15p07a") == {f"約 {f(b.sf(3), 3)}", f"約 {f(b.pmf(4), 3)}", f"約 {f(b.pmf(5), 3)}", f"約 {f(b.sf(2), 3)}", f"約 {f(.3 ** 4, 3)}"}, f"P(X≥4) = {b.sf(3):.5f}・誤答の由来")
h = rng.normal(170, 6, 1_000_000)
E.check("s15p07b", np.mean(h == 170) == 0 and abs(np.mean((h >= 169.5) & (h < 170.5)) - (stats.norm.cdf(.5 / 6) - stats.norm.cdf(-.5 / 6))) < .002 and ans("s15p07b").startswith("0 である"), "連続型の乱数でちょうど 170 は出ない・区間の確率は正")
lam = rng.gamma(1, 3, 100000); cnt = rng.poisson(lam)
E.check("s15p07c", abs(cnt.mean() - 3) < .05 and abs(cnt.var() - 12) < .4 and ans("s15p07c").startswith("ポアソン分布では分散と平均が等しいが"), f"日によって起こりやすさが違う例：平均 {cnt.mean():.2f}・分散 {cnt.var():.2f}")

# 第8問
w = rng.random((400000, 12)).sum(1) - 6
E.check("s15p08a", abs(w.mean()) < .01 and abs(w.var() - 1) < .01 and abs(np.mean(w > 1.96) - .025) < .002 and ans("s15p08a").startswith("期待値 0、分散 1 で"), f"平均 {w.mean():.4f}・分散 {w.var():.4f}・P(W>1.96) {np.mean(w > 1.96):.4f}")
E.check("s15p08b", abs(stats.t.isf(.025, 99) - 1.984) < 5e-4 and abs(1.984 / 1.96 - 1.012) < 5e-4 and ans("s15p08b").startswith("1.984 は 1.96 とほとんど同じなので"), "t(99) の点 1.984")
props = rng.binomial(10000, .5, 200000) / 10000
E.check("s15p08c", abs(props.std() - .005) < 1e-4 and ans("s15p08c") == "0.005" and opts("s15p08c") == {"0.005", f"{sqrt(.25 / 100):g}", f"{.25 / 100:g}", "0.000025", f"{sqrt(10000 * .25):g}"}, f"シミュレーション {props.std():.5f}・誤答の由来")

# 第9問
smp = rng.normal(0, 1, (40000, 101))
E.check("s15p09a", abs(smp.mean(1).var() / np.median(smp, 1).var() - 2 / np.pi) < .03 and ans("s15p09a").startswith("標本平均の分散のほうが小さい"), f"分散の比 {smp.mean(1).var() / np.median(smp, 1).var():.3f}（2/π ≈ 0.637）")
E.check("s15p09b", ans("s15p09b") == "標本平均の値", "幅 2zσ/√n に x̄ は入らない")
E.check("s15p09c", ans("s15p09c") == "区間では √(0.35 × 0.65/400)、検定では √(0.3 × 0.7/400) を使う", "区間は p̂、検定は p₀")
pop = rng.normal(7, 1.5, 200000); like = np.clip(np.abs(pop - 7) / 3, .01, 1); resp = pop[rng.random(200000) < like * (pop > 7) + .2 * like * (pop <= 7)]
smp2 = rng.choice(resp, 2000, replace=False); lo, hi = smp2.mean() - 1.96 * smp2.std(ddof=1) / sqrt(2000), smp2.mean() + 1.96 * smp2.std(ddof=1) / sqrt(2000)
E.check("s15p09d", not (lo <= pop.mean() <= hi) and ans("s15p09d").startswith("回答者は無作為に選ばれていないので"), f"満足した人ほど答える例：区間 ({lo:.2f}, {hi:.2f})・来店者全体の平均 {pop.mean():.2f}")

# 第10問
zz = rng.normal(0, 1, 1_000_000); pv = 2 * stats.norm.sf(np.abs(zz)); chosen_alpha = np.where(pv <= .05, .05, np.where(pv <= .10, .10, .05))
E.check("s15p10a", np.mean(pv <= chosen_alpha) > .09 and ans("s15p10a").startswith("有意水準はデータを見る前に決めておくもので"), f"p を見てから 5% か 10% を選ぶと、差がなくても『有意』の割合 {np.mean(pv <= chosen_alpha):.3f}")
z49, z51 = stats.norm.isf(.049 / 2), stats.norm.isf(.051 / 2)
E.check("s15p10b", abs(z49 - z51) < .02 and ans("s15p10b").startswith("判断は A が有意、B が有意でないと分かれるが"), f"p=0.049 は z={z49:.3f}、p=0.051 は z={z51:.3f}（ほとんど同じ）")
E.check("s15p10c", ans("s15p10c") == "本当に想定した差があるとき、帰無仮説を棄却する確率が 80%", "検出力の定義")

# 第11問
XY = rng.multivariate_normal([0, 0], [[100, -30], [-30, 100]], 1_000_000); dv = (XY[:, 0] - XY[:, 1]).var()
E.check("s15p11a", abs(dv - 260) < 1.5 and ans("s15p11a") == "260" and opts("s15p11a") == {"260", f"{200 - 60:g}", "200", "60", "20"}, f"シミュレーション V[X−Y] {dv:.1f}")
t23 = np.array([[12, 30, 18], [25, 10, 5]]); c1 = stats.chi2_contingency(t23, correction=False); c2 = stats.chi2_contingency(t23.T, correction=False)
E.check("s15p11b", abs(c1.statistic - c2.statistic) < 1e-9 and c1.dof == c2.dof == 2 and ans("s15p11b").startswith("カイ二乗統計量も自由度（2）も変わらない"), f"χ² {c1.statistic:.3f}（転置しても同じ）・自由度 {c1.dof}")
dfs = []
for _ in range(5000):
    v1, v2 = rng.uniform(.1, 100, 2); a1, a2 = v1 / 8, v2 / 13
    dfs.append((a1 + a2) ** 2 / (a1 ** 2 / 7 + a2 ** 2 / 12))
E.check("s15p11c", min(dfs) >= 7 - 1e-9 and max(dfs) <= 19 + 1e-9 and max(dfs) > 18.9 and min(dfs) < 7.2 and ans("s15p11c").startswith("2 つの群の不偏分散によって変わるが、7 以上 19 以下"), f"いろいろな分散でのウェルチの自由度 {min(dfs):.2f}〜{max(dfs):.2f}")

# 第12問
n = 100; x1 = rng.normal(0, 1, n); x2 = .95 * x1 + sqrt(1 - .95 ** 2) * rng.normal(0, 1, n)
preds_in, preds_out, coefs = [], [], []
for _ in range(2000):
    y = 1 + x1 + x2 + rng.normal(0, 1, n); A = np.c_[np.ones(n), x1, x2]; bcoef = np.linalg.lstsq(A, y, rcond=None)[0]
    coefs.append(bcoef[1]); preds_in.append(bcoef @ [1, 1, 1]); preds_out.append(bcoef @ [1, 1, -1])
E.check("s15p12a", np.std(preds_in) * 3 < np.std(preds_out) and np.std(coefs) > 2 * np.std(preds_in) and ans("s15p12a").startswith("データと同じように 2 つの変数が一緒に動く範囲での予測なら"), f"係数のぶれ {np.std(coefs):.2f}・同じ向きの予測のぶれ {np.std(preds_in):.2f}・逆向きの組合せ {np.std(preds_out):.2f}")
reg = rng.choice(3, 300); xx = rng.normal(0, 1, 300); yy = 2 * xx + np.array([0, 4, 7])[reg] + rng.normal(0, 1, 300)
Dn = np.c_[np.ones(300), xx, reg == 1, reg == 2]; Ds = np.c_[np.ones(300), xx, reg == 0, reg == 1]
bn = np.linalg.lstsq(Dn, yy, rcond=None)[0]; bs = np.linalg.lstsq(Ds, yy, rcond=None)[0]
E.check("s15p12b", abs(bs[3] - (bn[2] - bn[3])) < 1e-9 and abs(bs[2] + bn[3]) < 1e-9 and ans("s15p12b") == "−3" and opts("s15p12b") == {"−3", "4", "3", "11", "−7"}, f"基準を南に変えた東の係数 {bs[3]:.3f} ＝ 北基準の 東 − 南 {bn[2] - bn[3]:.3f}")
xs = np.linspace(0, 8, 8001); yh = 10 + 4 * xs - .5 * xs ** 2
E.check("s15p12c", abs(xs[yh.argmax()] - 4) < 1e-9 and abs(yh.max() - 18) < 1e-9 and ans("s15p12c") == "4" and opts("s15p12c") == {"4", f"{4 / .5:g}", f"{4 / 2:g}", "0", "18"}, "格子で最大の x = 4・最大値 18")

# 第13問
Fs = []
for _ in range(20000):
    g = rng.normal(0, 1, (3, 6)); gm = g.mean(); ssa = 6 * ((g.mean(1) - gm) ** 2).sum(); sse = ((g - g.mean(1, keepdims=True)) ** 2).sum()
    Fs.append((ssa / 2) / (sse / 15))
E.check("s15p13a", np.mean(np.array(Fs) < 1) > .5 and stats.f.sf(.6, 2, 15) > .5 and ans("s15p13a").startswith("群間の平均平方が群内の平均平方より小さく"), f"差がないとき F<1 になる割合 {np.mean(np.array(Fs) < 1):.2f}・F=0.6 の p 値（自由度 2, 15）{stats.f.sf(.6, 2, 15):.2f}")
pairs = list(combinations("ABC", 2)); zt = rng.normal(0, 1, (200000, 3))
E.check("s15p13b", len(pairs) == 3 and np.mean((np.abs(zt) > stats.norm.isf(.05 / 3 / 2)).any(1)) <= .052 and 1 - (1 - .05 / 3) ** 3 <= .05 and ans("s15p13b") == "約 0.017" and opts("s15p13b") == {"約 0.017", "0.05", "0.025", f"約 {f(.05 / 6, 3)}", "0.15"}, f"3 組を 0.05/3 で検定したときの全体の過誤 {np.mean((np.abs(zt) > stats.norm.isf(.05 / 3 / 2)).any(1)):.4f}（独立なら正確には {1 - (1 - .05 / 3) ** 3:.4f}）")
E.done()
