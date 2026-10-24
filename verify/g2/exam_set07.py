"""2級 模擬試験 第7回（g2 set07）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たりなど
別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の値がその誤り方で出るかを確かめる
（選択肢は正解の位置をならすために並べ替えてあるので、誤答は集合として比べる）。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import comb, exp, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(7)
rng = np.random.default_rng(20261030)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def opts(iid):
    return {str(c) for c in E.item(iid)["choices"]}


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


def iv(lo, hi, d):
    return f"({f(lo, d)}, {f(hi, d)})"


fr = np.array([3, 9, 14, 10, 4]); cls = ["10〜20", "20〜30", "30〜40", "40〜50", "50〜60"]; cum = np.cumsum(fr)
pos = lambda k: cls[int(np.searchsorted(cum, k))]
E.check("s07p01a", ans("s07p01a") == pos(10) + " kg" == pos(11) + " kg", f"10・11 番目は {pos(10)}（累積 {list(cum)}）")
E.check("s07p01b", ans("s07p01b").startswith("両方の学年について相対度数"), "大きさの違う集団は相対度数で比べる")
E.check("s07p02a", ans("s07p02a") == f"最頻値 35 kg、中央値は {pos(20)} kg の階級" and pos(20) == pos(21), "最頻値の階級値・中央値の階級")
E.choice("s07p02b", f"{f((30 * 62 + 20 * 70 + 50 * 58) / 100, 1)} 点", "人数で重み付け")
E.check("s07p02b", opts("s07p02b") == {"61.6 点", f"{f((62 + 70 + 58) / 3, 1)} 点", "62.0 点", "58.0 点", "70.0 点"}, "誤答の由来")
t = np.arange(31); phone = 5 * t + rng.normal(0, 4, 31); life = 78 + 0.15 * t + rng.normal(0, 0.3, 31)
E.check("s07p03a", np.corrcoef(phone, life)[0, 1] > 0.8 and abs(np.corrcoef(np.diff(phone), np.diff(life))[0, 1]) < 0.4 and ans("s07p03a").startswith("どちらも時間とともに増えている"), f"トレンドのある無関係な系列 r={np.corrcoef(phone, life)[0, 1]:.2f}、差分どうし r={np.corrcoef(np.diff(phone), np.diff(life))[0, 1]:.2f}")
h = rng.normal(165, 8, 500); w = 0.6 * (h - 165) / 8 * 10 + 0.8 * rng.normal(0, 10, 500)
E.check("s07p03b", abs(np.cov(h / 100, w)[0, 1] - np.cov(h, w)[0, 1] / 100) < 1e-9 and abs(np.corrcoef(h / 100, w)[0, 1] - np.corrcoef(h, w)[0, 1]) < 1e-12 and ans("s07p03b") == "共分散は 0.48 になり、相関係数は 0.6 のまま", "単位を変えて計算")
R2 = 1 - 50 / 200
E.check("s07p03c", ans("s07p03c") == f"決定係数 {f(R2, 2)}、相関係数 −{f(sqrt(R2), 2)}", f"R²={R2}, |r|={sqrt(R2):.4f}")
E.choice("s07p04a", f(96 / 120 * 100, 0), "96/120 × 100")
E.check("s07p04a", opts("s07p04a") == {"80", "76", f(96 * 120 / 100, 1), f(120 / 96 * 100, 0), "96"}, "誤答の由来")
E.choice("s07p04b", f(150 / 120, 2), "原系列 ÷ 中心化移動平均")
E.check("s07p04b", opts("s07p04b") == {"1.25", f(120 / 150, 2), "30", "270", "0.25"}, "誤答の由来")
E.check("s07p05a", ans("s07p05a").startswith("$\\mu$ は母数"), "母数と統計量")
roster = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30)
props = [roster[s0::10].mean() for s0 in range(10)]
E.check("s07p05b", max(props) == 1 and min(props) == 0 and ans("s07p05b").startswith("最初に 1 番を選ぶと班長だけが"), "周期 10 の名簿を間隔 10 で抽出すると班長の割合が 1 か 0")
E.choice("s07p06a", f(.3 + .4 - .3 * .4, 2), "加法定理と独立")
E.check("s07p06a", opts("s07p06a") == {"0.58", "0.70", "0.12", "0.42", "0.88"} and abs(.7 * .6 - .42) < 1e-12, "誤答の値")
exempt = rng.random(2_000_000) < .25; pass1 = rng.random(2_000_000) < .4; pass2 = rng.random(2_000_000) < np.where(exempt, .8, .6)
final = np.where(exempt, pass2, pass1 & pass2)
E.choice("s07p06b", f(.75 * .4 * .6 + .25 * .8, 2), f"シミュレーション {final.mean():.4f}")
E.check("s07p06b", abs(final.mean() - .38) < .002 and opts("s07p06b") == {"0.38", "0.24", "0.20", "0.44", "0.60"}, "誤答の値")
X = np.where(rng.random(2_000_000) < .3, 10, 0)
E.check("s07p06c", abs(X.mean() - 3) < .02 and abs(X.var() - 21) < .1 and ans("s07p06c") == "$E[X]=3$、$V[X]=21$", f"シミュレーション E={X.mean():.3f}, V={X.var():.3f}")
U = np.sqrt(rng.random(2_000_000))
E.choice("s07p06d", f(.8 ** 2 - .5 ** 2, 2), f"F(x)=x² の乱数（√U）で {np.mean((U > .5) & (U <= .8)):.4f}")
E.check("s07p06d", opts("s07p06d") == {"0.39", "0.30", "0.60", "0.64", "0.25"}, "誤答の値")
E.check("s07p07a", abs(stats.norm(60, 10).sf(80) - stats.norm(70, 5).sf(80)) < 1e-12 and ans("s07p07a").startswith("両方の組で等しい"), "どちらも z=2")
NB = stats.nbinom(3, .5)
E.choice("s07p07b", f(NB.pmf(2), 4), "scipy の nbinom（失敗 2 回で 3 回目の成功）")
E.check("s07p07b", opts("s07p07b") == {f(NB.pmf(2), 4), f(comb(5, 3) / 32, 4), f(.5 ** 3, 3), f(comb(4, 2) * .5 ** 4, 3), f(.5 ** 5, 4)}, "誤答の由来")
E.check("s07p08a", ans("s07p08a").startswith("大数の法則は「標本平均が母平均に近づく」"), "2 つの定理の違い")
E.choice("s07p08b", f(1 / 5.964, 3), f"scipy の F 点 {stats.f.isf(.95, 4, 10):.4f}")
E.check("s07p08b", abs(stats.f.isf(.05, 10, 4) - 5.964) < 5e-4 and abs(stats.f.isf(.05, 4, 10) - 3.478) < 5e-4 and abs(stats.f.isf(.95, 4, 10) - 1 / 5.964) < 1e-4 and opts("s07p08b") == {"0.168", f(1 / 3.478, 3), "3.478", "5.964", "−3.478"}, "表の値と誤答の由来")
ma = rng.normal(101, 2, (300000, 16)).mean(1); mb = rng.normal(100, 2, (300000, 16)).mean(1)
E.choice("s07p08c", f(1 - .0793, 3), f"シミュレーション {np.mean(ma > mb):.4f}")
E.check("s07p08c", abs(np.mean(ma > mb) - .921) < .003 and abs(stats.norm.sf(1.41) - .0793) < 5e-5 and opts("s07p08c") == {"0.921", "0.079", "0.500", f(stats.norm.sf(-.5), 3), f(stats.norm.sf(-1), 3)}, "表の値と誤答の由来")
E.check("s07p09a", ans("s07p09a").startswith("推定量は標本から母数を推定する式"), "推定量と推定値")
E.choice("s07p09b", iv(2 / 4.295, 2 * 3.855, 2), "F の点で区間")
E.check("s07p09b", abs(stats.f.isf(.025, 10, 8) - 4.295) < 5e-4 and abs(stats.f.isf(.025, 8, 10) - 3.855) < 5e-4 and opts("s07p09b") == {iv(2 / 4.295, 2 * 3.855, 2), iv(2 / 3.855, 2 * 4.295, 2), iv(2 / 4.295, 2 * 4.295, 2), iv(2 / 3.855, 2 * 3.855, 2), iv(1 / 3.855, 3.855, 2)}, "表の値と誤答の由来")
cov = 0
for _ in range(4000):
    a_ = rng.normal(0, sqrt(2), 11).var(ddof=1); b_ = rng.normal(0, 1, 9).var(ddof=1); r = a_ / b_
    cov += r / 4.295 <= 2 <= r * 3.855
E.check("s07p09b", abs(cov / 4000 - .95) < .015, f"真の比 2 での被覆率 {cov / 4000:.3f}")
E.check("s07p09c", -1.2 < 0 < 3.4 and ans("s07p09c").startswith("区間が 0 を含むので、「母平均の差は 0」は棄却されず"), "区間が 0 を含む")
E.check("s07p09d", ans("s07p09d") == f"幅の半分は {f(1.96 * sqrt(.25 / 100), 3)} で、標本比率が 0.5 のときに幅は最大" and max(range(1, 100), key=lambda k: k * (100 - k)) == 50, "幅の半分 0.098・p(1−p) の最大は 0.5")
E.check("s07p10a", ans("s07p10a") == "$Z<-1.645$", "左片側")
E.choice("s07p10b", f"p 値 {f(stats.norm.cdf(-1.8), 4)} で、棄却する", "左片側の p 値")
E.check("s07p10b", abs(stats.norm.sf(1.8) - .0359) < 5e-5, "表の値")
chi = 20 * 50 / 25
E.choice("s07p10c", f"χ² ＝ {f(chi, 0)} で、上側 2.5% 点 34.17 より大きいので棄却する", f"χ²={chi}, 両側 p={2 * stats.chi2.sf(chi, 20):.4f}")
E.check("s07p10c", abs(stats.chi2.isf(.025, 20) - 34.17) < 0.005 and abs(stats.chi2.isf(.975, 20) - 9.591) < 5e-4 and abs(stats.chi2.isf(.05, 20) - 31.41) < 0.005 and 21 * 50 / 25 == 42 and 20 * 25 / 50 == 10, "表の値と誤答の由来")
E.choice("s07p10d", f(200 * stats.poisson(1.5).pmf(0), 1), "200 × P(X=0)")
E.check("s07p10d", abs(exp(-1.5) - .2231) < 5e-5 and opts("s07p10d") == {"44.6", "0.223", f(200 * stats.poisson(1.5).pmf(1), 1), "300", f(200 / 1.5, 1)}, "表の値と誤答の由来")
tab = np.array([[60, 40], [45, 55]]); ch = stats.chi2_contingency(tab, correction=False); p = 105 / 200; z = .15 / sqrt(p * (1 - p) * .02)
E.check("s07p10e", abs(ch[0] - z * z) < 1e-9 and f(ch[0], 2) == "4.51" and f(z, 2) == "2.12" and ans("s07p10e").startswith("カイ二乗統計量 ≈ 4.51 は $z\\approx 2.12$ の 2 乗") and abs(1.96 ** 2 - 3.841) < 0.001, f"χ²={ch[0]:.4f}, z²={z * z:.4f}")
E.check("s07p10f", ans("s07p10f").startswith("性別と好きな季節は独立"), "独立性の検定の帰無仮説")
E.check("s07p11a", ans("s07p11a") == f(120 - (50 + 2 * 40 - 8 * 3), 0), "実際 − 予測 = 14（−14 と区別するため完全一致）")
E.check("s07p11a", opts("s07p11a") == {"14", "−14", "106", f(120 - (2 * 40 - 8 * 3), 0), f(120 - (50 + 80 + 24), 0)}, "誤答の由来")
n = 5000; foot = rng.normal(40, 10, n); comp = np.clip(np.round(foot / 10 + rng.normal(0, 1, n)), 0, None); sales = 50 + 2 * foot - 8 * comp + rng.normal(0, 5, n)
b_simple = np.polyfit(comp, sales, 1)[0]; b_multi = np.linalg.lstsq(np.c_[np.ones(n), foot, comp], sales, rcond=None)[0][2]
E.check("s07p11b", b_simple > 0 > b_multi and ans("s07p11b").startswith("人通りの多い場所ほど競合店も多い"), f"同じ構造のデータで 単回帰 {b_simple:.2f}・重回帰 {b_multi:.2f}")
R2c = 2 * 13.5 / (2 * 13.5 + 27)
E.choice("s07p11c", f(R2c, 2), f"R²={R2c}")
E.check("s07p11c", abs((R2c / 2) / ((1 - R2c) / 27) - 13.5) < 1e-9 and opts("s07p11c") == {"0.50", f(13.5 / 14.5, 2), f(27 / (27 + 29), 2), f(13.5 / (13.5 + 27), 2), f(1 / 13.5, 2)}, "F に戻すと 13.5・誤答の由来")
E.check("s07p12a", ans("s07p12a").startswith("温度の効果は時間によらずほぼ一定"), "平行 → 交互作用なし")
E.check("s07p12b", ans("s07p12b").startswith("少なくとも 1 組の添加剤の母平均に差がある"), "分散分析のあとは多重比較")
E.done()
