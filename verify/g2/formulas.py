"""2級の公式まとめ（content/g2/formulas.md）の検算。分布表は scipy.stats、ほかの公式は数値例・シミュレーションで確かめる。"""
from math import atanh, comb, sqrt, tanh
import numpy as np
from scipy import stats

results = []
rng = np.random.default_rng(20261024)


def check(sec, what, ok, value=""):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {sec} / {what} / 値: {value}")


x = rng.normal(10, 3, 50); y = 2 + 0.7 * x + rng.normal(0, 2, 50)
S = "データの要約"
check(S, "Σ(x−x̄)² = Σx² − n x̄²", abs(((x - x.mean()) ** 2).sum() - ((x ** 2).sum() - 50 * x.mean() ** 2)) < 1e-8)
check(S, "y=a+bx で分散 b² 倍・標準偏差 |b| 倍", abs((5 - 3 * x).var(ddof=1) - 9 * x.var(ddof=1)) < 1e-9 and abs((5 - 3 * x).std(ddof=1) - 3 * x.std(ddof=1)) < 1e-9)
z = (x - x.mean()) / x.std(ddof=1)
check(S, "標準化で平均 0・標準偏差 1", abs(z.mean()) < 1e-12 and abs(z.std(ddof=1) - 1) < 1e-12)
check(S, "幾何平均・調和平均（scipy）", abs(stats.gmean([2, 8]) - 4) < 1e-12 and abs(stats.hmean([60, 40]) - 48) < 1e-9)
v = np.array([100, 200, 300, 400, 1000.]); cs = np.r_[0, np.cumsum(v) / v.sum()]
gini = 1 - 2 * np.trapezoid(cs, np.linspace(0, 1, 6))
check(S, "ジニ係数＝1−2×面積（平均差の式と一致）", abs(gini - np.abs(v[:, None] - v[None, :]).sum() / (2 * 25 * v.mean())) < 1e-12, gini)
a, b = rng.normal(50, 4, 10), rng.normal(60, 5, 15); ab = np.r_[a, b]
within = (10 * a.var() + 15 * b.var()) / 25; between = (10 * (a.mean() - ab.mean()) ** 2 + 15 * (b.mean() - ab.mean()) ** 2) / 25
check(S, "合わせた分散＝群の中＋群の間（n で割る形）", abs(ab.var() - within - between) < 1e-9)
S = "2 変数と単回帰"
Sxx = ((x - x.mean()) ** 2).sum(); Sxy = ((x - x.mean()) * (y - y.mean())).sum(); Syy = ((y - y.mean()) ** 2).sum()
r = np.corrcoef(x, y)[0, 1]; bb, aa = np.polyfit(x, y, 1)
check(S, "r = Sxy/√(SxxSyy)", abs(r - Sxy / sqrt(Sxx * Syy)) < 1e-12)
check(S, "b = Sxy/Sxx = r sy/sx、a = ȳ − b x̄", abs(bb - Sxy / Sxx) < 1e-9 and abs(bb - r * y.std(ddof=1) / x.std(ddof=1)) < 1e-9 and abs(aa - (y.mean() - bb * x.mean())) < 1e-9)
e = y - aa - bb * x
check(S, "残差の和 0・R² = r²", abs(e.sum()) < 1e-9 and abs(1 - (e ** 2).sum() / Syy - r * r) < 1e-12)
check(S, "x を c 倍：r 不変・傾き 1/c", abs(np.corrcoef(4 * x, y)[0, 1] - r) < 1e-12 and abs(np.polyfit(4 * x, y, 1)[0] - bb / 4) < 1e-9)
zz = rng.normal(0, 1, 5000); xx = zz + rng.normal(0, 1, 5000); yy = zz + rng.normal(0, 1, 5000)
R = np.corrcoef([xx, yy, zz]); P = np.linalg.inv(R)
pc_formula = (R[0, 1] - R[0, 2] * R[1, 2]) / sqrt((1 - R[0, 2] ** 2) * (1 - R[1, 2] ** 2))
check(S, "偏相関の式（精度行列から求めた値と一致）", abs(pc_formula + P[0, 1] / sqrt(P[0, 0] * P[1, 1])) < 1e-12, pc_formula)
S = "時系列と指数"
ser = np.array([100, 120, 108, 135, 146.41])
check(S, "平均成長率 (y_n/y_0)^(1/n) − 1 で y_n が再現", abs(100 * (1 + ((ser[-1] / ser[0]) ** 0.25 - 1)) ** 4 - 146.41) < 1e-9)
p0 = np.array([200, 120, 80]); q0 = np.array([5, 10, 15]); p1 = np.array([240, 120, 72])
w = p0 * q0 / (p0 @ q0)
check(S, "ラスパイレス＝基準時の支出割合で重み付けた価格比", abs((p1 @ q0) / (p0 @ q0) - w @ (p1 / p0)) < 1e-12)
yq = np.array([20, 32, 25, 15, 24, 36, 29, 19.])
ma = [(0.5 * yq[t - 2] + yq[t - 1] + yq[t] + yq[t + 1] + 0.5 * yq[t + 2]) / 4 for t in range(2, 6)]
ma2 = [(yq[t - 2:t + 2].mean() + yq[t - 1:t + 3].mean()) / 2 for t in range(2, 6)]
check(S, "4 項中心化移動平均＝隣り合う 4 項平均の平均", np.allclose(ma, ma2))
S = "確率"
check(S, "順列・組合せ", comb(5, 2) == 10 and 5 * 4 == 20)
draws = rng.integers(1, 7, 1_000_000); A = draws % 2 == 0; B = draws >= 4
check(S, "加法定理（さいころのシミュレーション）", abs((A | B).mean() - (A.mean() + B.mean() - (A & B).mean())) < 1e-12)
S = "期待値と分散"
X = rng.normal(0, 2, 1_000_000); Y = -0.3 * X + rng.normal(0, 3, 1_000_000)
cxy = np.cov(X, Y)[0, 1]
check(S, "V[X−Y] = V[X]+V[Y]−2Cov", abs(np.var(X - Y, ddof=1) - (X.var(ddof=1) + Y.var(ddof=1) - 2 * cxy)) < 1e-6)
S = "主な確率分布"
dists = {"二項": (stats.binom(10, .3), 3, 2.1), "ポアソン": (stats.poisson(4), 4, 4), "幾何（試行の回数）": (stats.geom(.25), 4, .75 / .0625),
         "超幾何": (stats.hypergeom(20, 8, 5), 5 * 8 / 20, 5 * .4 * .6 * 15 / 19), "一様": (stats.uniform(2, 6), 5, 36 / 12),
         "指数（レート 2）": (stats.expon(scale=.5), .5, .25), "正規": (stats.norm(3, 2), 3, 4)}
for k, (d, m, v_) in dists.items():
    check(S, f"{k} の期待値と分散", abs(d.mean() - m) < 1e-9 and abs(d.var() - v_) < 1e-9, (d.mean(), d.var()))
check(S, "指数分布 P(X>t)=e^{−λt}", abs(stats.expon(scale=.5).sf(1.3) - np.exp(-2 * 1.3)) < 1e-12)
S = "標本分布"
m = rng.normal(5, 3, (200000, 9)).mean(1)
check(S, "V[X̄] = σ²/n", abs(m.var() - 1) < 0.01, m.var())
check(S, "連続修正 P(X≥60) ≈ P(Z>(59.5−50)/5)", abs(stats.binom(100, .5).sf(59) - stats.norm.sf(1.9)) < 0.001)
check(S, "χ²(k) の期待値 k・分散 2k", stats.chi2(7).mean() == 7 and stats.chi2(7).var() == 14)
check(S, "F_{1−α}(m,n) = 1/F_α(n,m)", abs(stats.f.isf(.95, 5, 8) - 1 / stats.f.isf(.05, 8, 5)) < 1e-9)
sm = rng.normal(0, 2, (100000, 6)); T = sm.mean(1) / (sm.std(1, ddof=1) / sqrt(6))
check(S, "t 統計量は t(n−1)", abs(np.mean(T > stats.t.isf(.05, 5)) - 0.05) < 0.003)
S = "区間推定"
cov = 0
for _ in range(20):
    sm = rng.normal(0, 3, (5000, 12)); s2 = sm.var(1, ddof=1); cov += np.sum((11 * s2 / stats.chi2.isf(.025, 11) <= 9) & (9 <= 11 * s2 / stats.chi2.isf(.975, 11)))
check(S, "母分散の区間の被覆率 95%", abs(cov / 100000 - .95) < 0.004, cov / 100000)
n_need = (1.96 / 0.03) ** 2 * 0.25
check(S, "必要な標本の大きさ", 1.96 * sqrt(.25 / np.ceil(n_need)) <= 0.03, n_need)
check(S, "相関係数の区間（z 変換）", abs(tanh(atanh(.6) - 1.96 / 5) - 0.292) < 5e-4)
S = "検定統計量"
r1 = stats.ttest_ind_from_stats(52, 4, 10, 48, sqrt(20), 10)
sp = (9 * 16 + 9 * 20) / 18
check(S, "等分散の 2 標本 t（scipy と一致）", abs(r1.statistic - 4 / sqrt(sp * .2)) < 1e-12)
tab = np.array([[30, 20, 10], [20, 30, 40]]); res = stats.chi2_contingency(tab, correction=False)
E = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
check(S, "独立性の検定の統計量と自由度", abs(res[0] - ((tab - E) ** 2 / E).sum()) < 1e-12 and res[2] == 2)
S = "回帰分析と分散分析"
n, p = 40, 2; Xd = rng.normal(0, 1, (n, p)); yd = Xd @ [1, .5] + rng.normal(0, 1, n); Ad = np.c_[np.ones(n), Xd]
beta = np.linalg.lstsq(Ad, yd, rcond=None)[0]; res_ = yd - Ad @ beta; SST = ((yd - yd.mean()) ** 2).sum(); R2 = 1 - res_ @ res_ / SST
F = (R2 / p) / ((1 - R2) / (n - p - 1)); Fdirect = ((SST - res_ @ res_) / p) / (res_ @ res_ / (n - p - 1))
check(S, "回帰全体の F（R² の式＝平方和の式）", abs(F - Fdirect) < 1e-9)
g = [rng.normal(mu, 1, 6) for mu in (0, .5, 1, 1.5)]; al = np.concatenate(g)
SA = sum(6 * (gi.mean() - al.mean()) ** 2 for gi in g); SE = sum(((gi - gi.mean()) ** 2).sum() for gi in g)
check(S, "一元配置の F（scipy の f_oneway と一致）", abs((SA / 3) / (SE / 20) - stats.f_oneway(*g).statistic) < 1e-9)
ng = results.count(False)
print(f"公式まとめ（2級）の検算: OK {results.count(True)} / NG {ng}")
raise SystemExit(1 if ng else 0)
