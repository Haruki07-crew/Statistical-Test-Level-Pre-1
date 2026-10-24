"""模擬試験 第21回（set21）の検算。generator/set21.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値最適化）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(21)
rng = np.random.default_rng(20261021)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def ans_idx(iid):
    return E.item(iid)["answer"]


# ---- 大問1：120 通りの並べ方を総当たり
perms = list(permutations(range(5)))
der = sum(all(p[i] != i for i in range(5)) for p in perms)
E.num("s21p01a", f(der / len(perms), 4), f"総当たりで {der}/{len(perms)}")
E.check("s21p01a", f(np.exp(-1), 4) != f(der / 120, 4) and abs(0.8 ** 5 - der / 120) > 0.001, "近似値 e^-1 や独立の近似は正解と区別できる")

# ---- 大問2
n, T = 40, 28
for p in (0.2, 0.5, 0.7, 0.93):
    ev = sum(stats.binom.pmf(t, n, p) * t * (n - t) / (n * (n - 1)) for t in range(n + 1))
    E.check("s21p02a", abs(ev - p * (1 - p)) < 1e-12, f"p={p} で E[T(n−T)/(n(n−1))]=p(1−p)")
    evm = sum(stats.binom.pmf(t, n, p) * (t / n) * (1 - t / n) for t in range(n + 1))
    E.check("s21p02a", abs(evm - p * (1 - p)) > 1e-4, f"p={p} で最尤推定量 p̂(1−p̂) は偏る")
E.choice("s21p02a", f"${f(T * (n - T) / (n * (n - 1)), 4)}$", "推定値")
# (b) 一致性をシミュレーションで：n を大きくしたとき |推定量−p|>0.05 の確率
p = 0.7
res = {}
for nn in (100, 10_000):
    Tn = rng.binomial(nn, p, 20_000)
    X1 = rng.random(20_000) < p
    res[nn] = {"T/n": np.mean(abs(Tn / nn - p) > 0.05), "(T+1)/(n+2)": np.mean(abs((Tn + 1) / (nn + 2) - p) > 0.05),
               "X1": np.mean(abs(X1 - p) > 0.05), "(T+n)/2n": np.mean(abs((Tn + nn) / (2 * nn) - p) > 0.05)}
ok = (res[10_000]["T/n"] < 0.001 and res[10_000]["(T+1)/(n+2)"] < 0.001 and res[10_000]["X1"] > 0.9 and res[10_000]["(T+n)/2n"] > 0.99)
E.check("s21p02b", ok, f"n=10000 で外れる確率 {res[10_000]}")
E.check("s21p02b", "T_n/n" in E.item("s21p02b")["choices"][ans_idx("s21p02b")] and "チェビシェフ" in E.item("s21p02b")["choices"][ans_idx("s21p02b")],
        "正解の選択肢は T_n/n の一致性の記述（他の4つは上のシミュレーションと反例で誤り）")

# ---- 大問3
for x, v in [(0.25, 0.7788), (0.5, 0.6065), (0.75, 0.4724), (1.5, 0.2231)]:
    E.check("s21p03a", abs(np.exp(-x) - v) < 5e-5, f"e^-{x} = {v}")
W = np.minimum.reduce([rng.exponential(20, 1_000_000), rng.exponential(30, 1_000_000), rng.exponential(60, 1_000_000)])
E.check("s21p03a", abs((W > 15).mean() - 0.2231) < 0.002, f"シミュレーション P(W>15)={(W > 15).mean():.4f}")
E.choice("s21p03a", f"${f(stats.expon(scale=20).sf(15) * stats.expon(scale=30).sf(15) * stats.expon(scale=60).sf(15), 3)}$", "scipy の生存関数の積")
dens = lambda x: (4 - x)
Zc = integrate.quad(dens, 0, 4)[0]
med = optimize.brentq(lambda m: integrate.quad(dens, 0, m)[0] / Zc - 0.5, 0, 4)
E.num("s21p03b", f(med, 2), "数値積分した分布関数の 0.5 点")
U = rng.random(1_000_000)
xs = 4 - 4 * np.sqrt(1 - U)  # 逆関数法：F(x)=1-(1-x/4)^2
E.check("s21p03b", abs(np.median(xs) - med) < 0.01, f"逆関数法で発生させた標本の中央値 {np.median(xs):.4f}")

# ---- 大問4
lam1, lam2 = 6, 5
ks = np.arange(0, 200)
kl = np.sum(stats.poisson.pmf(ks, lam1) * (stats.poisson.logpmf(ks, lam1) - stats.poisson.logpmf(ks, lam2)))
E.num("s21p04a", f(kl, 3), f"確率関数の和で KL={kl:.5f}")
E.check("s21p04a", abs(np.log(1.2) - 0.1823) < 5e-5, "log1.2=0.1823")
klr = np.sum(stats.poisson.pmf(ks, lam2) * (stats.poisson.logpmf(ks, lam2) - stats.poisson.logpmf(ks, lam1)))
E.check("s21p04a", f(klr, 3) != f(kl, 3), f"逆向き {klr:.4f} は別の値")
# (b) 2×2 表のカイ二乗（連続修正なし）の平方根と一致
chi = stats.chi2_contingency([[50, 200], [25, 225]], correction=False)[0]
E.num("s21p04b", f(np.sqrt(chi), 2), "連続修正なしのカイ二乗統計量の平方根")
# (c) 系統抽出：開始番号ごとの標本を列挙して包含確率と2次の包含確率
N, k = 12000, 40
samples = [set(range(r, N, k)) for r in range(k)]
incl = np.zeros(N)
for sm in samples:
    for i in sm:
        incl[i] += 1 / k
pair_same = sum(1 for sm in samples if 0 in sm and 40 in sm) / k
pair_diff = sum(1 for sm in samples if 0 in sm and 41 in sm) / k
E.check("s21p04c", np.allclose(incl, 1 / 40) and all(len(sm) == 300 for sm in samples) and len(samples) == 40, "包含確率はすべて 1/40、標本は 40 通り")
E.check("s21p04c", pair_same == 1 / 40 and pair_diff == 0, "2 冊が同時に選ばれる確率は 1/40 か 0")
E.check("s21p04c", "1/«k»" not in E.item("s21p04c")["choices"][ans_idx("s21p04c")] and "$1/40$" in E.item("s21p04c")["choices"][ans_idx("s21p04c")], "正解の選択肢の内容")

# ---- 大問5
r5 = stats.ttest_ind_from_stats(68.5, 8.0, 12, 62.0, 7.0, 10, equal_var=True)
se = 6.5 / r5.statistic
tq = stats.t.ppf(0.975, 20)
E.choice("s21p05a", f"({f(6.5 - tq * se, 2)},\\ {f(6.5 + tq * se, 2)})", "scipy の t 検定の標準誤差と t 点")
E.check("s21p05a", abs(tq - 2.086) < 0.0005 and abs(stats.t.ppf(0.975, 22) - 2.074) < 0.0005, "t 点")
cov = []
if True:
    a = rng.normal(0, 7.5, (100_000, 12))
    b = rng.normal(0, 7.5, (100_000, 10))
    sp2 = (a.var(1, ddof=1) * 11 + b.var(1, ddof=1) * 9) / 20
    h = 2.086 * np.sqrt(sp2 * (1 / 12 + 1 / 10))
    cov = np.abs(a.mean(1) - b.mean(1)) <= h
E.check("s21p05a", abs(cov.mean() - 0.95) < 0.004, f"シミュレーションの被覆率 {cov.mean():.4f}")
tab = np.array([[18, 12], [16, 24]])
lor = np.log(18 * 24 / (12 * 16))
sel = np.sqrt((1 / tab).sum())
lo, hi = np.exp(lor - 1.96 * sel), np.exp(lor + 1.96 * sel)
E.choice("s21p05b", f"({f(lo, 2)},\\ {f(hi, 2)})", "正確な対数と指数で")
for x, v in [(-0.1554, 0.8561), (1.7772, 5.9133), (0.3344, 1.3971), (1.2874, 3.6234)]:
    E.check("s21p05b", abs(np.exp(x) - v) < 5e-5, f"e^{x}={v}")
E.check("s21p05b", abs(np.log(2.25) - 0.8109) < 5e-5, "log2.25")
# ブートストラップ風の確認：対数オッズ比の標準誤差をシミュレーション（多項）で
sim = rng.multinomial(70, tab.flatten() / 70, 200_000).reshape(-1, 2, 2) + 0.0
sim = sim[(sim > 0).all(axis=(1, 2))]
lsim = np.log(sim[:, 0, 0] * sim[:, 1, 1] / (sim[:, 0, 1] * sim[:, 1, 0]))
E.check("s21p05b", abs(lsim.std() - sel) < 0.03, f"シミュレーションの log OR の標準偏差 {lsim.std():.4f}（式 {sel:.4f}）")

# ---- 大問6：データを作って実際に交互作用モデルを当てはめる
xg = rng.uniform(40, 95, 400)
Dg = (rng.random(400) < 0.4).astype(float)
yg = 1.20 + 0.020 * xg + 0.90 * Dg - 0.008 * Dg * xg  # 誤差なしで係数が復元される
Xm = np.column_stack([np.ones(400), xg, Dg, Dg * xg])
bh = np.linalg.lstsq(Xm, yg, rcond=None)[0]
pred = lambda d, x: bh @ [1, x, d, d * x]
dif = pred(1, 70) - pred(0, 70)
slope = pred(1, 71) - pred(1, 70)
E.choice("s21p06a", f"差{f(dif, 2)}、傾き{f(slope, 3)}", "当てはめた式で 70 点の差と推薦の傾き")
# (b) 不均一分散のシミュレーション：OLS は不偏だが通常の SE は過小
xs6 = np.linspace(1, 10, 50)
Xs = np.column_stack([np.ones(50), xs6])
bs, ses = [], []
for _ in range(20_000):
    e = rng.normal(0, 0.3 * xs6)
    y = 1 + 0.5 * xs6 + e
    b, rs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    bs.append(b[1])
    s2 = rs[0] / 48
    ses.append(np.sqrt(s2 * np.linalg.inv(Xs.T @ Xs)[1, 1]))
bs = np.array(bs)
E.check("s21p06b", abs(bs.mean() - 0.5) < 0.003 and np.mean(ses) < bs.std() * 0.95,
        f"不均一分散でも傾きは不偏（平均 {bs.mean():.4f}）、通常の SE の平均 {np.mean(ses):.4f} < 実際の標準偏差 {bs.std():.4f}")
E.check("s21p06b", "頑健な標準誤差" in E.item("s21p06b")["choices"][ans_idx("s21p06b")] and "不偏のまま" in E.item("s21p06b")["choices"][ans_idx("s21p06b")], "正解の選択肢の内容")

# ---- 大問7
pz = stats.norm.sf(64.5, 60, 15 / 6)
E.choice("s21p07a", f"${f(pz, 3)}$", "scipy norm.sf(64.5, 60, 2.5)")
for x, v in [(0.3, 0.6179), (0.72, 0.7642), (1.8, 0.9641)]:
    E.check("s21p07a", abs(stats.norm.cdf(x) - v) < 5e-5, f"Φ({x})={v}")
for rr in (0.6, -0.6):
    w, V = np.linalg.eigh(np.array([[1, rr], [rr, 1]]))
    E.check("s21p07b", np.allclose(sorted(w), [0.4, 1.6]) and abs(max(w) / 2 - 0.8) < 1e-12, f"r={rr} の固有値 {w}、寄与率 80%")
w, V = np.linalg.eigh(np.array([[1, 0.6], [0.6, 1]]))
E.check("s21p07b", np.allclose(np.abs(V[:, 0]), [1 / np.sqrt(2)] * 2) and V[0, 0] * V[1, 0] < 0, "第2主成分の固有ベクトルは差の方向 (1,−1)/√2")
E.choice("s21p07b", "第1主成分の寄与率は$80\\%$である。第2主成分", "正解の文")

# ---- 大問8
pmf = {0: 0.4, 1: 0.3, 2: 0.2, 4: 0.1}
xs8 = np.repeat(list(pmf), [int(v * 10_000) for v in pmf.values()])
E.choice("s21p08a", f"${f(stats.skew(xs8), 2)}$", "比率どおりの 10000 個のデータで scipy の歪度")
g = stats.geom(0.3)
cond = (g.cdf(5) - g.cdf(3)) / g.sf(3)
sim = rng.geometric(0.3, 2_000_000)
rest = sim[sim > 3] - 3
E.choice("s21p08b", f"確率{f(cond, 2)}、期待値{f(rest.mean(), 2)}回", f"scipy の条件付き確率と、シミュレーションの残り回数の平均 {rest.mean():.3f}")

# ---- 大問9
P = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.4, 0.6]])
wv, vv = np.linalg.eig(P.T)
pi = np.real(vv[:, np.argmin(abs(wv - 1))])
pi = pi / pi.sum()
E.num("s21p09a", f(1 / pi[2], 2), f"固有ベクトルによる定常分布 {pi}")
# 初到達時間の連立方程式で直接：上級から出発して上級に戻るまで
Q = P[:2, :2]
m = np.linalg.solve(np.eye(2) - Q, np.ones(2))  # 初級・中級から上級までの平均日数
ret = 1 + P[2, 0] * m[0] + P[2, 1] * m[1]
E.check("s21p09a", abs(ret - 1 / pi[2]) < 1e-12, f"初到達時間から求めた平均再帰時間 {ret:.4f}")

# ---- 大問10
a_ = stats.binom.sf(3, 5, 0.6)
b_ = stats.binom.cdf(3, 5, 0.9)
E.choice("s21p10a", f"第1種{f(a_, 3)}、第2種{f(b_, 3)}", "scipy の二項分布")

# ---- 大問11
SA, SB, SAB, SE = 180, 420, 60, 540
E.choice("s21p11a", f"\\eta^2={f(SA / (SA + SB + SAB + SE), 2)}$、偏$\\eta^2={f(SA / (SA + SE), 2)}", "定義から")
# 経験ベイズ：周辺尤度の数値最大化ではなくモーメント法の式を独立に。さらにシミュレーションで縮小推定の MSE が小さいことを確認
sig2n = 100 / 25
tau2 = 13 - sig2n
post_mean = (70 / sig2n + 60 / tau2) / (1 / sig2n + 1 / tau2)  # 正規・正規の事後平均（精度の重み付き平均）
E.choice("s21p11b", f"${f(post_mean, 2)}$", "精度の重み付き平均として事後平均")
th = rng.normal(60, 3, (50_000, 8))
yb = th + rng.normal(0, 2, th.shape)
mb = yb.mean(1, keepdims=True)
t2 = np.maximum(yb.var(1, ddof=1, keepdims=True) - 4, 0.01)
eb = mb + (1 - 4 / (4 + t2)) * (yb - mb)
E.check("s21p11b", ((eb - th) ** 2).mean() < ((yb - th) ** 2).mean(), f"経験ベイズの MSE {((eb - th) ** 2).mean():.3f} < 生の平均点の MSE {((yb - th) ** 2).mean():.3f}")
E.check("s21p11a", SA + SB + SAB + SE == 1200, "平方和の合計")

# ---- 大問12
c = 1.2 / 2.0
p0 = stats.norm.cdf(-c)
Ey = stats.norm.cdf(c) * 1.2 + 2.0 * stats.norm.pdf(c)
E.choice("s21p12a", f"P(y=0)={f(p0, 3)}$、$E[y]={f(Ey, 2)}", "scipy の Φ と φ")
ys = np.maximum(0, 1.2 + rng.normal(0, 2.0, 2_000_000))
E.check("s21p12a", abs((ys == 0).mean() - 0.2743) < 0.002 and abs(ys.mean() - 1.537) < 0.005, f"シミュレーション P={(ys == 0).mean():.4f}、E={ys.mean():.4f}")
E.check("s21p12a", abs(stats.norm.cdf(0.6) - 0.7257) < 5e-5 and abs(stats.norm.pdf(0.6) - 0.3332) < 5e-5, "Φ(0.6)・φ(0.6)")
obs = rng.normal(0, 1, 40)
obs = (obs - obs.mean()) / obs.std(ddof=1) * np.sqrt(1.2) + 2.0
full = np.r_[obs, np.full(10, obs.mean())]
E.num("s21p12b", f(full.var(ddof=1), 3), "実際に平均で埋めたデータの不偏分散")
yq = np.array([0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
grid = np.linspace(0, 7, 70_001)
loss = [np.sum((yq - cc) * (0.8 - (yq - cc < 0))) for cc in grid]
cbest = grid[int(np.argmin(loss))]
E.choice("s21p12c", f"${f(cbest, 1)}$", f"格子探索でチェック関数の和の最小点 {cbest:.4f}")
E.check("s21p12c", f(np.quantile(yq, 0.8, method="inverted_cdf"), 1) == "4.0", "経験分布の 80% 分位点（逆関数）")

# ---- 大問13
Dm = np.array([[0, 1.0, 3.5, 6.0, 5.5], [1.0, 0, 2.0, 5.0, 4.8], [3.5, 2.0, 0, 4.0, 4.2], [6.0, 5.0, 4.0, 0, 2.5], [5.5, 4.8, 4.2, 2.5, 0]])
Ls = linkage(squareform(Dm), method="single")
lab = fcluster(Ls, 3.0, criterion="distance")
groups = sorted(sorted("ABCDE"[i] for i in range(5) if lab[i] == g) for g in set(lab))
E.check("s21p13a", groups == [["A", "B", "C"], ["D", "E"]], f"scipy 単連結法を高さ 3.0 で切る {groups}")
Lc = linkage(squareform(Dm), method="complete")
labc = fcluster(Lc, 3.0, criterion="distance")
groupsc = sorted(sorted("ABCDE"[i] for i in range(5) if labc[i] == g) for g in set(labc))
E.check("s21p13a", groupsc == [["A", "B"], ["C"], ["D", "E"]], f"最長距離法なら {groupsc}（選択肢 0）")
E.choice("s21p13a", "{A,B,C}、{D,E}", "正解の分け方")
# SVM：判別関数 ±1 の直線上の点を作って距離を測る
w = np.array([1.2, 0.5])
b = -2.0
x_plus = (1 - b) * w / (w @ w)
x_minus = (-1 - b) * w / (w @ w)
E.num("s21p13b", f(np.linalg.norm(x_plus - x_minus), 2), "f=+1 と f=−1 の直線上の最近点どうしの距離")
dist_lines = abs(sp.Rational(2)) / sp.sqrt(sp.Rational(144, 100) + sp.Rational(25, 100))
E.check("s21p13b", abs(float(dist_lines) - np.linalg.norm(x_plus - x_minus)) < 1e-12, "sympy の平行線の距離と一致")

E.done()
