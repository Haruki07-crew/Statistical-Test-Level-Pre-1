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
run(q)
raise SystemExit(1 if ng else 0)
