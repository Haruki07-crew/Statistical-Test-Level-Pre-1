"""2級 第3〜6章：問題の検算（章ごとに結果を保存する）。"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qhead import Q, chosen  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ng = 0


def run(q):
    global ng
    try:
        q.done()
    except SystemExit as e:
        ng += int(bool(e.code))


# ---------- 第3章 ----------
q = Q(3)
q.check("q03-scatter-plot-1", chosen(q, "q03-scatter-plot-1").startswith("右下がり"), "気温↑で売上↓は負の相関")
xq = np.array([-2, -1, 0, 1, 2.])
q.check("q03-scatter-plot-2", abs(np.corrcoef(xq, xq ** 2)[0, 1]) < 1e-12, "y=x² で r=0（answer=false）")
q.num("q03-scatter-plot-3", 12 / (4 * 5))
rng = np.random.default_rng(7)
xs, ys, rin = [], [], []
for g in range(3):
    x = 2 * g + rng.standard_normal(3000); y = 2 * g - 0.3 * (x - 2 * g) + rng.standard_normal(3000)
    rin.append(np.corrcoef(x, y)[0, 1]); xs += list(x); ys += list(y)
q.check("q03-stratified-scatter-1", all(r < 0 for r in rin) and np.corrcoef(xs, ys)[0, 1] > 0.4 and chosen(q, "q03-stratified-scatter-1").startswith("地域を表す"), f"層の中 {np.round(rin, 2)}・全体 {np.corrcoef(xs, ys)[0, 1]:.2f}")
q.check("q03-stratified-scatter-2", np.corrcoef(xs, ys)[0, 1] > 0 > max(rin), "逆向きになる例（answer=true）")
t = np.array([[40, 60], [20, 80]])
q.num("q03-cross-tabulation-1", np.outer(t.sum(1), t.sum(0))[0, 0] / t.sum())
q.check("q03-cross-tabulation-2", round(56 / 86 * 100) == 65 and chosen(q, "q03-cross-tabulation-2") == "65%" and round(56 / 80 * 100) == 70 and round(86 / 200 * 100) == 43, "56/86≈65%（70%・43%・25% は別の割合）")
q.check("q03-cross-tabulation-3", 30 / 120 != 30 / 86, "行と列で分母が違う（answer=false）")
q.num("q03-covariance-and-correlation-g1", 30 / np.sqrt(25 * 64))
xc = np.array([1, 3, 4, 6, 9.]); yc = np.array([2, 5, 4, 8, 9.]); uc = 2 * xc + 10
q.check("q03-covariance-and-correlation-g2", abs(np.cov(uc, yc)[0, 1] - 2 * np.cov(xc, yc)[0, 1]) < 1e-9 and abs(np.corrcoef(uc, yc)[0, 1] - np.corrcoef(xc, yc)[0, 1]) < 1e-12 and chosen(q, "q03-covariance-and-correlation-g2") == "共分散 24、相関係数 0.6", "u=2x+10 で共分散 2 倍・相関係数は不変")
rs = np.random.default_rng(31); size = rs.normal(0, 1, 5000); trucks = size + 0.5 * rs.normal(0, 1, 5000); loss = size + 0.5 * rs.normal(0, 1, 5000)
rr = np.corrcoef(trucks, loss)[0, 1]; part = [np.corrcoef(trucks[m], loss[m])[0, 1] for m in (size < -0.5, np.abs(size) <= 0.5, size > 0.5)]
q.check("q03-spurious-correlation-g1", rr > 0.7 and max(abs(np.array(part))) < 0.6 and chosen(q, "q03-spurious-correlation-g1").startswith("火事の規模が大きいほど"), f"規模が両方に影響すると r={rr:.2f}、規模で層別すると {np.round(part, 2)}")
q.num("q03-spurious-correlation-g2", (0.5 - 0.6 * 0.5) / np.sqrt((1 - 0.36) * (1 - 0.25)))
q.check("q03-correlation-matrix-g1", chosen(q, "q03-correlation-matrix-g1") == "来客数と雨量" and max([("来客数と気温", .45), ("来客数と雨量", -.82), ("気温と雨量", -.30)], key=lambda t: abs(t[1]))[0] == "来客数と雨量", "絶対値の最大は 0.82")
q.num("q03-partial-correlation-coefficient-g1", (0.6 - 0.5 * 0.4) / np.sqrt(0.75 * 0.84))
q.num("q03-simpsons-paradox-g1", (30 * .8 + 70 * .4) / 100)
q.check("q03-simpsons-paradox-g1", (80 * .7 + 20 * .3) / 100 == 0.62 and .8 > .7 and .4 > .3, "A さんは 0.62（どちらの種類でも B が高いのに全体では A）")
q.num("q03-correlation-matrix-g2", 12 / (4 * 5))
cv3 = np.random.default_rng(31).multivariate_normal([0, 0], [[16, 12], [12, 25]], 400000); q.check("q03-correlation-matrix-g2", abs(np.corrcoef(cv3.T)[0, 1] - 0.6) < 0.005, "シミュレーション")
pr3 = (0.5 - 0.7 * 0.7) / np.sqrt((1 - 0.49) * (1 - 0.49))
q.check("q03-partial-correlation-coefficient-g2", abs(pr3 - 0.0196) < 5e-4 and chosen(q, "q03-partial-correlation-coefficient-g2").startswith("偏相関係数はほぼ 0 で"), f"偏相関 {pr3:.4f}")
q.check("q03-simpsons-paradox-g2", 18 / 20 > 68 / 80 and 40 / 80 > 8 / 20 and 58 / 100 < 76 / 100 and 18 + 40 == 58 and 68 + 8 == 76 and chosen(q, "q03-simpsons-paradox-g2").startswith("X は回復しにくい重症"), "層ごとに X が高く、全体で Y が高い")
Sxx3 = 290 - 10 * 5 ** 2; Syy3 = 700 - 10 * 8 ** 2; Sxy3 = 430 - 10 * 5 * 8; q.num("q03-covariance-and-correlation-g3", Sxy3 / np.sqrt(Sxx3 * Syy3))
q.check("q03-covariance-and-correlation-g3", (Sxx3, Syy3, Sxy3) == (40, 60, 30), "S_xx=40・S_yy=60・S_xy=30")
rg3 = np.random.default_rng(33); gr3 = np.repeat(np.arange(1, 7), 200); shoe = 16 + gr3 + rg3.normal(0, 1, 1200); calc = 40 + 8 * gr3 + rg3.normal(0, 8, 1200)
within3 = [np.corrcoef(shoe[gr3 == g], calc[gr3 == g])[0, 1] for g in range(1, 7)]
q.check("q03-spurious-correlation-g3", np.corrcoef(shoe, calc)[0, 1] > 0.6 and max(abs(np.array(within3))) < 0.2 and chosen(q, "q03-spurious-correlation-g3").startswith("学年ごとに分けて"), f"全体 {np.corrcoef(shoe, calc)[0, 1]:.2f}・学年ごと 最大 {max(abs(np.array(within3))):.2f}")
run(q)

# ---------- 第4章 ----------
q = Q(4)
q.num("q04-simple-regression-1", 50 - 2.5 * 10)
q.num("q04-simple-regression-2", 0.8 * 10 / 5)
x = np.array([2, 4, 6, 8, 10.]); y = np.array([5, 7, 6, 10, 12.])
b, a = np.polyfit(x, y, 1)
q.check("q04-simple-regression-3", abs(b - 0.85) < 1e-9 and abs(a - 2.9) < 1e-9 and "2.9 + 0.85x" in chosen(q, "q04-simple-regression-3"), f"a={a:.3f}, b={b:.3f}")
q.check("q04-simple-regression-3", abs(np.corrcoef(x, y)[0, 1] - 0.92) < 0.005 and abs(3.5 + 0.75 * 6 - 8) < 1e-12 and abs(2.48 + 0.92 * 6 - 8) < 1e-12, "誤答は平均の点を通るが傾きが違う（0.92 は相関係数）")
yh = a + b * x
q.check("q04-simple-regression-4", abs(a + b * x.mean() - y.mean()) < 1e-12 and abs((y - yh).sum()) < 1e-9, "平均の点を通り残差の和 0（answer=true）")
q.check("q04-least-squares-method-g1", chosen(q, "q04-least-squares-method-g1").startswith("各点の $y$ の値と直線の予測値の差（残差）の 2 乗の合計"), "最小にするのは残差平方和")
xs = np.array([1, 2, 3, 4, 5.]); ys = np.array([2, 4, 5, 4, 6.]); bs, as_ = np.polyfit(xs, ys, 1)
q.check("q04-least-squares-method-g1", abs((ys - as_ - bs * xs).sum()) < 1e-9 and all(((ys - as_ - bs * xs) ** 2).sum() <= ((ys - a2 - b2 * xs) ** 2).sum() + 1e-12 for a2 in np.linspace(0, 4, 41) for b2 in np.linspace(0, 2, 41)), "例のデータで残差平方和が最小・残差の和は 0")
n_, sx, sy, sxy, sxx = 5, 20, 50, 215, 90
b_ls = (sxy - n_ * (sx / n_) * (sy / n_)) / (sxx - n_ * (sx / n_) ** 2)
q.num("q04-least-squares-method-g2", sy / n_ - b_ls * sx / n_)
q.check("q04-least-squares-method-g2", abs(b_ls - 1.5) < 1e-12 and abs(sxy / sxx - 2.39) < 0.005, "傾き 1.5・誤り 215/90≈2.39")
y12 = ys.copy(); y12[4] = 12
q.check("q04-least-squares-method-g3", abs(bs - 0.8) < 1e-9 and abs(np.polyfit(xs, y12, 1)[0] - 2.0) < 1e-9, "傾き 0.8 → 2.0（answer=false）")
q.num("q04-coefficient-of-determination-g1", 1 - 100 / 400)
q.check("q04-coefficient-of-determination-g2", chosen(q, "q04-coefficient-of-determination-g2") == "−0.9" and abs(np.sqrt(0.81) - 0.9) < 1e-12, "√0.81=0.9、傾きが負なので −0.9")
q.num("q04-residual-g1", 47 - (30 + 0.5 * 40))
xr = np.array([1, 3, 4, 8, 9.]); yr = np.array([2, 9, 4, 3, 12.]); br, ar = np.polyfit(xr, yr, 1)
q.check("q04-residual-g2", abs((yr - ar - br * xr).sum()) < 1e-9 and np.corrcoef(xr, yr)[0, 1] < 0.6, "当てはまりの悪いデータでも残差の合計は 0（answer=false）")
q.num("q04-log-transformation-g1", (np.exp(0.2) - 1) * 100)
q.check("q04-log-transformation-g1", abs(np.exp(0.2) - 1.2214) < 5e-5, "表の値 e^0.2")
xf = np.linspace(1, 10, 400); yf = 2 + 3 * xf + np.random.default_rng(4).normal(0, 0.3 * xf); bf, af = np.polyfit(xf, yf, 1); ef = yf - af - bf * xf; pf = af + bf * xf
q.check("q04-residual-plot-g1", np.std(ef[pf > np.median(pf)]) > 1.8 * np.std(ef[pf <= np.median(pf)]) and chosen(q, "q04-residual-plot-g1").startswith("誤差の分散が一定でない"), "誤差の標準偏差が x に比例するデータで残差が扇形")
xu = np.linspace(0, 10, 41); yu = 2 + 0.5 * xu ** 2; bu, au = np.polyfit(xu, yu, 1); ru = yu - au - bu * xu
q.check("q04-residual-plot-g2", ru[0] > 0 and ru[-1] > 0 and ru[20] < 0 and abs(ru.sum()) < 1e-9 and chosen(q, "q04-residual-plot-g2").startswith("直線では表せない"), "2 次の関係に直線を当てると残差が U 字形")
q.num("q04-log-transformation-g2", (1.1 ** 0.8 - 1) * 100)
q.check("q04-log-transformation-g2", abs(1.1 ** 0.8 - 1.0792) < 5e-5, "表の値")
q.num("q04-coefficient-of-determination-g3", 1.6 ** 2 * 50 / 200)
q.check("q04-coefficient-of-determination-g3", abs((1.6 * 50) / np.sqrt(50 * 200) - 0.8) < 1e-12, "r=0.8")
x4 = np.array([1, 2, 3, 4.]); y4 = np.array([2, 4, 5, 7.]); b4, a4 = np.polyfit(x4, y4, 1); q.num("q04-residual-g3", ((y4 - a4 - b4 * x4) ** 2).sum())
q.check("q04-residual-g3", abs(b4 - 1.6) < 1e-9 and abs(a4 - 0.5) < 1e-9 and np.allclose(y4 - a4 - b4 * x4, [-0.1, 0.3, -0.3, 0.1]), "傾き 1.6・切片 0.5・残差")
run(q)

# ---------- 第5章 ----------
q = Q(5)
m = np.arange(36); sales = 100 + 2 * m + 20 * np.sin(2 * np.pi * m / 12)
mom = np.diff(sales); yoy = sales[12:] - sales[:-12]
q.check("q05-time-series-components-1", np.allclose(yoy, 24) and np.std(mom) > 5 and chosen(q, "q05-time-series-components-1") == "前年同月比を見る", f"前年同月の差は一定 24（トレンドだけ）、前月との差は季節で上下（標準偏差 {np.std(mom):.1f}）")
ma = np.array([(0.5 * sales[t - 6] + sales[t - 5:t + 6].sum() + 0.5 * sales[t + 6]) / 12 for t in range(6, 30)])
q.check("q05-time-series-components-2", np.allclose(ma, 100 + 2 * np.arange(6, 30)), "12 か月中心化移動平均でトレンドだけが残る（answer=true）")
q.num("q05-growth-rate-1", ((121 / 100) ** 0.5 - 1) * 100)
q.num("q05-growth-rate-2", ((1.44 * 0.81) ** 0.5 - 1) * 100)
q.check("q05-growth-rate-3", (98 - 80) / 200 == 0.09 and (132 - 120) / 200 == 0.06 and (230 - 200) / 200 == 0.15 and chosen(q, "q05-growth-rate-3") == "9 ポイント", "B の寄与度 9 ポイント（A 6、計 15）")
q.check("q05-growth-rate-4", abs(1.1 * 0.9 - 0.99) < 1e-12, "1.1×0.9=0.99（answer=false）")
q.num("q05-index-number-1", (120 / 125 - 1) * 100)
q.num("q05-index-number-2", 132 / 110 * 100)
q.check("q05-index-number-3", 120 - 114 == 6 and abs(114 / 120 - 1 + 0.05) < 1e-12 and "−5%" in chosen(q, "q05-index-number-3"), "6 ポイント・−5%")
p0 = np.array([100, 50]); q0 = np.array([4, 10]); p1 = np.array([120, 50])
L = p1 @ q0 / (p0 @ q0) * 100; w = p0 * q0 / (p0 @ q0)
q.num("q05-price-index-1", L)
q.check("q05-price-index-1", abs((w * p1 / p0).sum() * 100 - L) < 1e-9, "支出の割合で重み付けした価格比の平均と一致")
q.check("q05-price-index-2", chosen(q, "q05-price-index-2") == "比較時の数量", "パーシェは比較時の数量")
q.num("q05-price-index-3", 550 / (110 / 100))
q.num("q05-moving-average-1", (15 + 18 + 13) / 3)
yv = np.array([3., 8, 1, 6, 9])
centered = 0.5 * ((yv[0:4].sum()) / 4 + (yv[1:5].sum()) / 4)
q.check("q05-moving-average-2", abs(centered - np.dot([1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8], yv)) < 1e-12 and chosen(q, "q05-moving-average-2").startswith("両端が 1/8"), "重み 1/8・1/4・1/4・1/4・1/8")
ar = [0.0]; rr5 = np.random.default_rng(5)
for _ in range(5000): ar.append(-0.6 * ar[-1] + rr5.normal())
ar = np.array(ar[1:]); r1_ = np.corrcoef(ar[:-1], ar[1:])[0, 1]
q.check("q05-autocovariance-autocorrelation-g1", r1_ < -0.5 and np.mean(np.sign(np.diff(ar[:-1])) != np.sign(np.diff(ar[1:]))) > 0.6 and chosen(q, "q05-autocovariance-autocorrelation-g1").startswith("前の時点より大きい値の次は小さい値"), f"ラグ 1 の自己相関 {r1_:.2f} の系列で上がり下がりが入れかわる割合 {np.mean(np.sign(np.diff(ar[:-1])) != np.sign(np.diff(ar[1:]))):.2f}")
tq = np.arange(80); yq = 50 + 8 * np.array([1, 0, -1, 0] * 20) + rr5.normal(0, 1, 80); yq -= yq.mean(); acq = lambda k: (yq[:-k] * yq[k:]).sum() / (yq ** 2).sum()
q.check("q05-correlogram-g1", acq(4) > 0.7 and acq(8) > 0.7 and acq(2) < -0.3 and acq(6) < -0.3 and chosen(q, "q05-correlogram-g1") == "1 年（4 四半期）周期の季節変動がある", f"4 期周期の系列でラグ 2・4・6・8 の自己相関 {acq(2):.2f}・{acq(4):.2f}・{acq(6):.2f}・{acq(8):.2f}")
q.num("q05-seasonal-adjustment-g1", 156 / 1.2)
xa5 = np.array([2, 4, 5, 7, 7.]); da5 = xa5 - xa5.mean(); q.num("q05-autocovariance-autocorrelation-g2", (da5[:-1] * da5[1:]).sum() / (da5 ** 2).sum())
tr5 = 0.5 * np.arange(120) + np.random.default_rng(55).normal(0, 3, 120); dt5 = tr5 - tr5.mean(); ac5 = [(dt5[:-k] * dt5[k:]).sum() / (dt5 ** 2).sum() for k in range(1, 13)]
q.check("q05-correlogram-g2", ac5[0] > 0.9 and min(ac5) > 0.6 and all(a > b for a, b in zip(ac5, ac5[1:])) and chosen(q, "q05-correlogram-g2").startswith("長期的に"), f"トレンドのある系列の自己相関 {ac5[0]:.2f} … {ac5[-1]:.2f}")
q.num("q05-time-series-components-g3", 115 - 120 - (-8))
w12 = (np.r_[np.ones(12), 0] + np.r_[0, np.ones(12)]) / 24
q.num("q05-moving-average-g3", len(w12))
q.check("q05-moving-average-g3", abs(w12.sum() - 1) < 1e-12 and w12[0] == w12[-1] == 1 / 24 and np.allclose(w12[1:-1], 1 / 12), "重み 1/24・1/12×11・1/24")
run(q)

# ---------- 第6章 ----------
q = Q(6)
q.concept("q06-observational-vs-experimental-1", "対象者が自分で選んだ要因を比べる観察研究")
for n in (1000, 100000):
    z = rng.standard_normal(n); smoke = z + rng.standard_normal(n) > 0; outcome = z + rng.standard_normal(n)
    diff = outcome[smoke].mean() - outcome[~smoke].mean()
q.check("q06-observational-vs-experimental-2", diff > 0.5, f"効果 0 でも交絡で差 {diff:.2f} が残る（n=100000、answer=false）")
temp = rng.normal(25, 5, 5000); ice = temp + rng.standard_normal(5000) * 3; acc = temp + rng.standard_normal(5000) * 3
q.check("q06-observational-vs-experimental-3", np.corrcoef(ice, acc)[0, 1] > 0.5 and chosen(q, "q06-observational-vs-experimental-3").startswith("気温"), f"共通の要因（気温）で相関 {np.corrcoef(ice, acc)[0, 1]:.2f}")
q.check("q06-population-and-sample-1", chosen(q, "q06-population-and-sample-1").startswith("今日生産した 5000"), "母集団は 5000 個")
pop = rng.random(1_000_000) < 0.4; volunteer = rng.random(1_000_000) < np.where(pop, 0.02, 0.005)
q.check("q06-population-and-sample-2", abs(pop[volunteer].mean() - 0.4) > 0.2 and volunteer.sum() > 10000, f"偏った 1 万件以上でも推定 {pop[volunteer].mean():.2f}（真 0.40、answer=false）")
q.num("q06-population-and-sample-3", 400 * (0.02 / 0.01) ** 2)
q.check("q06-simple-random-sampling-g1", chosen(q, "q06-simple-random-sampling-g1").startswith("学生名簿の全員に番号を付け"), "名簿と乱数")
q.num("q06-simple-random-sampling-g2", 200 / 8000)
rs6 = np.random.default_rng(6); hit6 = np.mean([0 in rs6.choice(8000, 200, replace=False) for _ in range(40000)])
q.check("q06-simple-random-sampling-g2", abs(hit6 - 0.025) < 0.003, f"番号 0 の学生が選ばれた割合 {hit6:.4f}")
q.check("q06-sampling-and-nonsampling-error-g1", chosen(q, "q06-sampling-and-nonsampling-error-g1").startswith("無作為に選んだ標本に、たまたま"), "選ばれ方の偶然 → 標本誤差")
q.concept("q06-sampling-and-nonsampling-error-g2", "全数調査でも非標本誤差は生じる（answer=false）")
q.check("q06-fisher-three-principles-g1", chosen(q, "q06-fisher-three-principles-g1") == "無作為化", "乱数による割り付け")
d_ = np.array([[10, 12.5, 13.5], [14.5, 16, 18], [18, 20.5, 22], [21.5, 24, 26.5]]); gm_ = d_.mean()
ST_ = ((d_ - gm_) ** 2).sum(); SA_ = 4 * ((d_.mean(0) - gm_) ** 2).sum(); SB_ = 3 * ((d_.mean(1) - gm_) ** 2).sum()
q.check("q06-fisher-three-principles-g2", (ST_ - SA_) / 9 > 100 * (ST_ - SA_ - SB_) / 6 and chosen(q, "q06-fisher-three-principles-g2").startswith("区域の違いによる変動を誤差から取り除き"), f"誤差の平均平方 {(ST_ - SA_) / 9:.3f} → {(ST_ - SA_ - SB_) / 6:.3f}")
q.num("q06-stratified-sampling-g1", 100 * 800 / 2500)
q.check("q06-stratified-sampling-g1", 100 * 1200 / 2500 + 100 * 800 / 2500 + 100 * 500 / 2500 == 100, "48 + 32 + 20 = 100")
q.num("q06-systematic-sampling-g1", 17 + 3 * (6000 // 200))
q.check("q06-cluster-sampling-g1", chosen(q, "q06-cluster-sampling-g1") == "集落抽出法", "箱（集落）を選び中は全数")
q.num("q06-two-stage-sampling-g1", 20 / 200 * 25 / 500)
rs2 = np.random.default_rng(62); hit2 = np.mean([(0 in rs2.choice(200, 20, replace=False)) and (0 in rs2.choice(500, 25, replace=False)) for _ in range(200000)])
q.check("q06-two-stage-sampling-g1", abs(hit2 - 0.005) < 0.0006, f"地区 0 の世帯 0 が選ばれた割合 {hit2:.5f}")
q.num("q06-nonresponse-bias-g1", 0.5 * 0.6 / (0.5 * 0.6 + 0.5 * 0.3))
popn = np.random.default_rng(66).random(400000) < 0.5; resp = np.random.default_rng(67).random(400000) < np.where(popn, 0.6, 0.3)
q.check("q06-nonresponse-bias-g1", abs(popn[resp].mean() - 2 / 3) < 0.003, f"シミュレーション {popn[resp].mean():.4f}")
rg6 = np.random.default_rng(61); popA = np.r_[rg6.normal(20, 2, 3000), rg6.normal(50, 2, 3000), rg6.normal(80, 2, 3000)]
srs6 = [popA[rg6.choice(9000, 90, replace=False)].mean() for _ in range(3000)]; st6 = [np.mean([popA[k * 3000 + rg6.choice(3000, 30, replace=False)].mean() for k in range(3)]) for _ in range(3000)]
q.check("q06-stratified-sampling-g2", np.std(st6) * 5 < np.std(srs6) and chosen(q, "q06-stratified-sampling-g2").startswith("層の中では値が似ていて"), f"層の間の差が大きいとき：層化 {np.std(st6):.3f}・単純 {np.std(srs6):.3f}")
roster = np.tile(np.r_[1, np.zeros(9)], 100); props = [roster[s::10].mean() for s in range(10)]
q.check("q06-systematic-sampling-g2", props[0] == 1 and max(props[1:]) == 0 and roster.mean() == 0.1 and chosen(q, "q06-systematic-sampling-g2").startswith("最初の番号によっては"), "開始 1 なら班長 100%、それ以外は 0%（本当は 10%）")
cl_mix = rg6.normal(50, 10, (100, 20)); cl_sim = rg6.normal(0, 10, (100, 1)) + 50 + rg6.normal(0, 2, (100, 20))
se_mix = np.std([cl_mix[rg6.choice(100, 5, replace=False)].mean() for _ in range(3000)]); se_sim = np.std([cl_sim[rg6.choice(100, 5, replace=False)].mean() for _ in range(3000)])
q.check("q06-cluster-sampling-g2", se_mix * 2 < se_sim and chosen(q, "q06-cluster-sampling-g2").startswith("各集落の中が母集団全体と"), f"集落の中が多様 {se_mix:.2f}・集落の中が均質 {se_sim:.2f}")
q.concept("q06-two-stage-sampling-g2", "名簿と費用の利点。精度は単純無作為抽出法よりふつう下がる")
pick, seen6 = [], set()
for r_ in [83, 12, 45, 12, 97, 6, 33]:
    if 1 <= r_ <= 80 and r_ not in seen6 and len(pick) < 3: pick.append(r_); seen6.add(r_)
q.check("q06-simple-random-sampling-g3", pick == [12, 45, 6] and chosen(q, "q06-simple-random-sampling-g3") == "12, 45, 06", f"選ばれる番号 {pick}")
q.check("q06-sampling-and-nonsampling-error-g3", abs(np.sqrt(.25 / 400) / np.sqrt(.25 / 100) - 0.5) < 1e-12 and chosen(q, "q06-sampling-and-nonsampling-error-g3").startswith("標本誤差の大きさはおよそ半分"), "標準誤差は 1/√4 = 1/2")
q.concept("q06-fisher-three-principles-g3", "反復は誤差の大きさを見積もるため")
run(q)
raise SystemExit(1 if ng else 0)
