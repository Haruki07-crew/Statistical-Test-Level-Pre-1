"""模擬試験 第18回（set18）の検算。generator/set18.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値積分）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(18)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table_ok(iid, pairs, d, what):
    """問題文に示した値（丸め）が正確な値と一致するか。pairs = [(示した値, 正確な値)]"""
    bad = [(a, b) for a, b in pairs if f(b, d) != f(a, d)]
    E.check(iid, not bad, f"{what}（不一致 {bad}）")


def nearest(iid, value):
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c).replace("−", "-")).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1 遺伝
nAA, nAa, naa = 50, 80, 70
n = 200
# (a) 多項尤度を数値最大化
negll = lambda p: -(nAA * math.log(p * p) + nAa * math.log(2 * p * (1 - p)) + naa * math.log((1 - p) ** 2))
p_hat = optimize.minimize_scalar(negll, bounds=(1e-6, 1 - 1e-6), method="bounded").x
E.choice("s18p01a", f"={f(p_hat, 2)}$", "多項尤度の数値最大化で p̂")
# 十分性：T が同じ別のデータで尤度比が p によらないこと
L = lambda a, b, c, p: (p * p) ** a * (2 * p * (1 - p)) ** b * ((1 - p) ** 2) ** c
r1 = [L(50, 80, 70, p) / L(60, 60, 80, p) for p in (0.2, 0.45, 0.7)]   # T=180 で同じ
E.check("s18p01a", max(r1) / min(r1) - 1 < 1e-9, "T=2nAA+nAa が等しいデータの尤度比が p によらない（十分性）")
r2 = [L(50, 80, 70, p) / L(70, 60, 70, p) for p in (0.2, 0.45, 0.7)]   # A をもつ個体数は同じ130だが T が違う
E.check("s18p01a", max(r2) / min(r2) - 1 > 1e-3, "A をもつ個体数が同じでも T が違えば尤度比は p による（個体数は十分でない）")
# (b) G 統計量を scipy で
obs = np.array([nAA, nAa, naa])
exp = n * np.array([p_hat ** 2, 2 * p_hat * (1 - p_hat), (1 - p_hat) ** 2])
G = 2 * np.sum(obs * np.log(obs / exp))
G2 = stats.power_divergence(obs, exp, ddof=1, lambda_="log-likelihood")
E.choice("s18p01b", f"G={f(G, 2)}$、自由度1で、${f(stats.chi2.isf(0.01, 1), 2)}$を超えるので棄却する", "正確な対数で G と自由度1の1%点")
E.check("s18p01b", abs(G2.statistic - G) < 1e-9 and G2.pvalue < 0.01, f"scipy power_divergence(ddof=1) p値 {G2.pvalue:.4f}")
table_ok("s18p01b", [(0.2107, math.log(50 / 40.5)), (-0.2131, math.log(80 / 99)), (0.1459, math.log(70 / 60.5))], 4, "示した対数の値")
table_ok("s18p01b", [(3.84, stats.chi2.isf(.05, 1)), (6.63, stats.chi2.isf(.01, 1)), (7.88, stats.chi2.isf(.005, 1)),
                     (5.99, stats.chi2.isf(.05, 2)), (9.21, stats.chi2.isf(.01, 2)), (10.60, stats.chi2.isf(.005, 2))], 2, "χ² 表")
# (c) EM：E ステップを条件付き確率の総当たりで、M ステップを完全データ尤度の数値最大化で
th0 = 0.5
probs = {"BB": th0 ** 2, "Bb": 2 * th0 * (1 - th0)}
eBB = 160 * probs["BB"] / sum(probs.values())
eBb = 160 * probs["Bb"] / sum(probs.values())
q = lambda t: -((2 * eBB + eBb) * math.log(t) + (eBb + 2 * 40) * math.log(1 - t))
th1 = optimize.minimize_scalar(q, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-10}).x
E.num("s18p01c", float(f(th1, 3)), "M ステップを数値最大化")
th = 0.5
for _ in range(500):
    a_ = 160 * th * th / (th * th + 2 * th * (1 - th))
    b_ = 160 - a_
    th = (2 * a_ + b_) / 400
E.check("s18p01c", abs(th - (1 - math.sqrt(0.2))) < 1e-6, f"反復の収束先 {th:.4f} = 1−√0.2（解説の主張）")

# ---- 大問2
# (a) コーシーの標本平均をシミュレーションと特性関数で
xs = stats.cauchy(scale=10).rvs(size=(200_000, 25), random_state=rng).mean(axis=1)
pr = np.mean(np.abs(xs) <= 10)
E.check("s18p02a", abs(pr - 0.5) < 0.005, f"シミュレーション P(|X̄|≤10) ≈ {pr:.4f}")
t, g, nn = sp.symbols("t gamma n", positive=True)
E.check("s18p02a", sp.simplify(sp.exp(-g * t / nn) ** nn - sp.exp(-g * t)) == 0, "特性関数の積が元と同じ")
E.choice("s18p02a", f"${f(2 * stats.cauchy(scale=10).cdf(10) - 1, 3)}$", "scipy cauchy の分布関数")
table_ok("s18p02a", [(1.3734, math.atan(5)), (1.5308, math.atan(25)), (0.0400, math.atan(0.04))], 4, "arctan の値")
# (b) ワイブル（scipy weibull_min）の四分位範囲
w = stats.weibull_min(2, scale=40)
E.num("s18p02b", float(f(w.ppf(0.75) - w.ppf(0.25), 1)), "scipy weibull_min の ppf")
table_ok("s18p02b", [(0.2877, math.log(4 / 3)), (1.3863, math.log(4))], 4, "対数の値")

# ---- 大問3
# (a) シミュレーション：Z~N(6,2²) から W=(1+0.5Z)²（Z>-2 の範囲で単調。P(Z<-2) は無視できる）
Z = rng.normal(6, 2, 1_000_000)
W = (1 + 0.5 * Z) ** 2
E.check("s18p03a", abs(np.median(W) - 16) < 0.05 and abs(np.mean(W > 25) - 0.1587) < 0.002,
        f"シミュレーション 中央値 {np.median(W):.3f}、P(W>25) {np.mean(W > 25):.4f}")
E.choice("s18p03a", f"{f(16, 1)}$g、$P(W>25)={f(stats.norm.sf(1), 3)}$", "scipy norm.sf(1)")
E.check("s18p03a", stats.norm.cdf(-4) < 1e-4, "1+0.5Z<0（変換の範囲外）となる確率は無視できる")
table_ok("s18p03a", [(0.6915, stats.norm.cdf(.5)), (0.8413, stats.norm.cdf(1)), (0.9772, stats.norm.cdf(2))], 4, "Φ の値")
# (b) 予測区間の係数をシミュレーションと正確な t 点で
k = stats.t.isf(0.025, 9) * math.sqrt(1.1)
E.choice("s18p03b", f"${f(k, 2)}$", "scipy の t 点")
X = rng.normal(0, 1, (200_000, 11))
xb, s_ = X[:, :10].mean(1), X[:, :10].std(1, ddof=1)
cov = np.mean(np.abs(X[:, 10] - xb) <= float(f(k, 2)) * s_)
E.check("s18p03b", abs(cov - 0.95) < 0.003, f"シミュレーションの被覆確率 {cov:.4f}")
table_ok("s18p03b", [(2.262, stats.t.isf(.025, 9)), (2.228, stats.t.isf(.025, 10))], 3, "t 表")
# (c) F 検定
Fv = 4.8 / 1.6
pv = 2 * stats.f.sf(Fv, 9, 12)
E.check("s18p03c", pv > 0.05 and stats.f.sf(Fv, 9, 12) < 0.05, f"両側 p 値 {pv:.4f}>0.05、片側なら <0.05")
E.choice("s18p03c", f"F={f(Fv, 2)}$で、${f(stats.f.isf(.025, 9, 12), 2)}$を超えないので帰無仮説を棄却しない", "scipy の F 点")
table_ok("s18p03c", [(2.80, stats.f.isf(.05, 9, 12)), (3.44, stats.f.isf(.025, 9, 12)), (3.87, stats.f.isf(.025, 12, 9))], 2, "F 表")

# ---- 大問4
# (a) ρ を定義から数値計算
p = 0.8
rho = sum(pr_ * abs(b - p) ** 3 for b, pr_ in ((1, p), (0, 1 - p)))
sig = math.sqrt(stats.bernoulli(p).var())
be = 0.4748 * rho / (sig ** 3 * math.sqrt(400))
E.choice("s18p04a", f"${f(be, 3)}$", "定義から ρ と σ を計算")
dev = max(abs(stats.binom.cdf(k_, 400, p) - stats.norm.cdf((k_ - 320) / 8)) for k_ in range(250, 390))
E.check("s18p04a", dev <= be, f"実際の最大誤差 {dev:.4f} は上界以下")
# (b) 端点を正確に逆変換
se = 1 / math.sqrt(150 * .8 * .2)
lo, hi = math.log(4) - 1.96 * se, math.log(4) + 1.96 * se
expit = lambda z: 1 / (1 + math.exp(-z))
E.choice("s18p04b", f"({f(expit(lo), 3)},\\ {f(expit(hi), 3)})", "正確な値で逆変換")
table_ok("s18p04b", [(v, math.exp(-x)) for x, v in [(0.986, .3731), (1.051, .3496), (1.322, .2666), (1.450, .2346), (1.722, .1787), (1.786, .1676)]], 4, "e^{-x} の表")
table_ok("s18p04b", [(1.3863, math.log(4)), (0.2041, se)], 4, "log4 と標準誤差")

# ---- 大問5
# (a) 切断ポアソンの期待値を和で
pk = stats.poisson(1.5).pmf(np.arange(1, 60))
E.choice("s18p05a", f"${f(np.sum(np.arange(1, 60) * pk) / pk.sum(), 2)}$", "確率関数の和で E[X|X≥1]")
table_ok("s18p05a", [(0.2231, math.exp(-1.5))], 4, "e^{-1.5}")
# (b) 開始点を全列挙
y = np.array([5, 8, 12, 6, 9, 11, 7, 10, 13, 4, 9, 14])
ms = [y[j::3].mean() for j in range(3)]
E.num("s18p05b", float(f(np.var(ms), 2)), "3 通りの標本平均の分散（numpy var, ddof=0）")
srs = [np.mean(c) for c in itertools.combinations(y, 4)]
E.check("s18p05b", abs(np.var(srs) - (1 - 4 / 12) * np.var(y, ddof=1) / 4) < 1e-9 and np.var(srs) < np.var(ms),
        f"単純無作為抽出の分散 {np.var(srs):.4f}（全組合せ）より大きい")

# ---- 大問6
tv = (3.24 - 3) / 0.10
E.check("s18p06a", stats.t.sf(tv, 23) * 2 < 0.05, f"両側 p 値 {2 * stats.t.sf(tv, 23):.4f}")
E.choice("s18p06a", "正の相対成長", "β>3 は正の相対成長")
table_ok("s18p06a", [(2.069, stats.t.isf(.025, 23)), (2.807, stats.t.isf(.005, 23)), (2.064, stats.t.isf(.025, 24))], 3, "t 表")
E.choice("s18p06b", f"{f(stats.norm.ppf(8.5 / 10), 2)}で、右の裾が正規分布より重い", "scipy norm.ppf(0.85)")
table_ok("s18p06b", [(0.842, stats.norm.ppf(.8)), (0.908, stats.norm.ppf(9 / 11)), (1.036, stats.norm.ppf(.85)), (1.282, stats.norm.ppf(.9))], 3, "分位点の表（0.818=9/11）")
tq = stats.t(3)   # 裾の重い分布では右上の点が直線より上に出る
E.check("s18p06b", tq.ppf(0.85) / tq.std() > 0 and stats.t(3).ppf(0.99) > stats.norm.ppf(0.99) * tq.std() * 0 + stats.norm.ppf(0.99),
        "裾の重い t(3) の上側分位点は正規分布より大きい")
# (c) 折れ線回帰：データを作って当てはめ直す
ages = np.arange(0, 8, 0.5)
Xd = np.column_stack([np.ones_like(ages), ages, np.maximum(ages - 3, 0)])
yd = Xd @ np.array([8.0, 12.0, -9.0])
bhat = np.linalg.lstsq(Xd, yd, rcond=None)[0]
pred = np.array([1, 5, 2]) @ bhat
E.choice("s18p06c", f"{f(bhat[1] + bhat[2], 1)}cm/年、{f(pred, 1)}cm", "最小二乗で当てはめ直した傾きと予測")

# ---- 大問7 トービット：数値積分
mu, sg = 2.0, 4.0
Ey = integrate.quad(lambda v: max(v, 0) * stats.norm.pdf(v, mu, sg), -40, 40, points=[0])[0]
E.choice("s18p07a", f"${f(Ey, 2)}$", "数値積分 E[max(0,y*)]")
ysim = np.maximum(rng.normal(mu, sg, 1_000_000), 0)
E.check("s18p07a", abs(ysim.mean() - Ey) < 0.01, f"シミュレーション {ysim.mean():.4f}")
table_ok("s18p07a", [(0.6915, stats.norm.cdf(.5)), (0.3521, stats.norm.pdf(.5))], 4, "Φ と φ")

# ---- 大問8 分割区法：自由度を項の数え上げで
r_, a, b = 3, 2, 3
cells = r_ * a * b
df = {"blk": r_ - 1, "A": a - 1, "E1": (r_ - 1) * (a - 1), "B": b - 1, "AB": (a - 1) * (b - 1)}
df["E2"] = cells - 1 - sum(df.values())
E.check("s18p08a", df["E1"] == 2 and df["E2"] == 8, f"自由度 {df}")
Fa = (30.0 / 1) / (4.0 / 2)
E.choice("s18p08a", f"F={f(Fa, 1)}$で、$F_{{0.05}}(1,2)={f(stats.f.isf(.05, 1, 2), 2)}$を超えないので棄却しない", "scipy の F 点")
E.check("s18p08a", stats.f.sf(Fa, 1, 2) > 0.05, f"p 値 {stats.f.sf(Fa, 1, 2):.4f}")
table_ok("s18p08a", [(18.51, stats.f.isf(.05, 1, 2)), (5.32, stats.f.isf(.05, 1, 8)), (4.96, stats.f.isf(.05, 1, 10))], 2, "F 表")
E.check("s18p08a", abs(6 + 30 + 4 + 24 + 3 + 12 - 79) < 1e-9, "平方和の計")

# ---- 大問9 ATT：個体レベルのデータを作り、回帰（層×処置）の予測で処置群の反実仮想を作る
rows = [(1, 1, 50.0, 30), (1, 0, 30.0, 10), (0, 1, 18.0, 10), (0, 0, 10.0, 50)]
Wv = np.concatenate([[w_] * k_ for w_, t_, y_, k_ in rows])
Tv = np.concatenate([[t_] * k_ for w_, t_, y_, k_ in rows])
Yv = np.concatenate([[y_] * k_ for w_, t_, y_, k_ in rows])
Xm = np.column_stack([np.ones_like(Wv), Wv, Tv, Wv * Tv]).astype(float)
bb = np.linalg.lstsq(Xm, Yv, rcond=None)[0]
treated = Tv == 1
X1 = Xm[treated].copy()
X0 = X1.copy()
X0[:, 2] = 0
X0[:, 3] = 0
E.num("s18p09a", float(f(np.mean(X1 @ bb - X0 @ bb), 1)), "飽和回帰で処置群の Y(1)−Y(0) を平均")

# ---- 大問10
# (a) 混合分布の所属確率（scipy の正規密度）
d1, d2 = 0.6 * stats.norm.pdf(15, 12, 3), 0.4 * stats.norm.pdf(15, 20, 3)
E.choice("s18p10a", f"${f(d1 / (d1 + d2), 3)}$", "scipy norm.pdf で事後確率")
table_ok("s18p10a", [(2.4327, math.exp(0.889))], 4, "e^{0.889}")
E.check("s18p10a", abs((25 - 9) / 18 - 0.889) < 5e-4, "指数 16/18 ≈ 0.889")
# (b) KS
A = [11, 13, 14, 16, 18, 21]
B = [15, 17, 19, 20, 22, 24]
ks = stats.ks_2samp(A, B, method="exact")
E.choice("s18p10b", f"D=3/6$で、p値${f(ks.pvalue, 3)}$が0.05より大きいので棄却しない", "scipy ks_2samp(exact)")
cnt = {}
for comb in itertools.combinations(range(12), 6):
    fa = fb = dm = 0
    for i in range(12):
        if i in comb:
            fa += 1
        else:
            fb += 1
        dm = max(dm, abs(fa - fb))
    cnt[dm] = cnt.get(dm, 0) + 1
tot = sum(cnt.values())
pv_ = {kk: sum(v for d, v in cnt.items() if d >= kk) / tot for kk in (3, 4, 5)}
table_ok("s18p10b", [(0.474, pv_[3]), (0.143, pv_[4]), (0.026, pv_[5])], 3, "P(D≥k/6) を 924 通りの総当たりで")
# (c) ファイ係数を 0-1 データの相関で
xa = np.array([1] * 15 + [1] * 5 + [0] * 10 + [0] * 20)
ya = np.array([1] * 15 + [0] * 5 + [1] * 10 + [0] * 20)
E.num("s18p10c", float(f(np.corrcoef(xa, ya)[0, 1], 3)), "0-1 変数の相関係数（numpy corrcoef）")

# ---- 大問11
# (a) 判別関数を numpy で
Sm = np.array([[4.0, 2.0], [2.0, 5.0]])
m1v, m2v = np.array([40.0, 30.0]), np.array([36.0, 24.0])
zz = (m1v - m2v) @ np.linalg.solve(Sm, np.array([42.0, 24.5]) - (m1v + m2v) / 2)
E.choice("s18p11a", f"z={f(zz, 2)}$で負なので種2", "numpy solve")
# マハラノビス距離でも種2に近いこと
dm1 = (np.array([42, 24.5]) - m1v) @ np.linalg.solve(Sm, np.array([42, 24.5]) - m1v)
dm2 = (np.array([42, 24.5]) - m2v) @ np.linalg.solve(Sm, np.array([42, 24.5]) - m2v)
E.check("s18p11a", dm2 < dm1, f"マハラノビス距離² 種1 {dm1:.3f}、種2 {dm2:.3f}")
# (b) 共分散選択：精度行列の(1,2)成分を0にする制約つき最尤を数値最適化
Sr = np.array([[1, .62, .80], [.62, 1, .60], [.80, .60, 1]])


def nll(x):
    s12 = x[0]
    Sg = np.array([[1, s12, .80], [s12, 1, .60], [.80, .60, 1]])
    if np.any(np.linalg.eigvalsh(Sg) <= 0):
        return 1e9
    return np.log(np.linalg.det(Sg)) + np.trace(Sr @ np.linalg.inv(Sg))


# 対角と辺のある成分を標本値に固定した上で (1,2) を動かし、精度行列の (1,2) が 0 になる点
root = optimize.brentq(lambda s12: np.linalg.inv(np.array([[1, s12, .8], [s12, 1, .6], [.8, .6, 1]]))[0, 1], 0.2, 0.7)
E.choice("s18p11b", f"${f(root, 2)}$", "精度行列の(1,2)成分が0になる相関を数値で解く")
# 一般の制約つき最尤（全成分を動かす）でも同じになることを、Ω の (1,2)=0 制約下の最適化で確かめる


def nll_prec(v):
    o11, o22, o33, o13, o23 = v
    Om = np.array([[o11, 0, o13], [0, o22, o23], [o13, o23, o33]])
    ev = np.linalg.eigvalsh(Om)
    if np.any(ev <= 0):
        return 1e9
    return -np.log(np.linalg.det(Om)) + np.trace(Sr @ Om)


res = optimize.minimize(nll_prec, x0=[2, 2, 3, -1, -0.5], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
Sig = np.linalg.inv(np.array([[res.x[0], 0, res.x[3]], [0, res.x[1], res.x[4]], [res.x[3], res.x[4], res.x[2]]]))
r12_fit = Sig[0, 1] / math.sqrt(Sig[0, 0] * Sig[1, 1])
E.check("s18p11b", abs(r12_fit - 0.48) < 1e-3, f"精度行列を直接最尤推定した相関 {r12_fit:.4f}")

# ---- 大問12
# (a) 反射原理をシミュレーションで（細かい格子）
steps, T = 2000, 9
inc = rng.normal(0, math.sqrt(4 * T / steps), (20_000, steps))
mx = np.cumsum(inc, axis=1).max(axis=1)
sim = np.mean(mx >= 6)
E.check("s18p12a", abs(sim - 0.3174) < 0.015, f"シミュレーション {sim:.4f}（離散化でやや小さめ）")
E.num("s18p12a", float(f(2 * stats.norm.sf(6, 0, 6), 3)), "scipy 2·P(B(9)≥6)")
table_ok("s18p12a", [(0.5675, stats.norm.cdf(.17)), (0.6293, stats.norm.cdf(.33)), (0.8413, stats.norm.cdf(1))], 4, "Φ の値")
# (b) 正確な関数値で
zb = math.sqrt(-2 * math.log(0.3)) * math.cos(2 * math.pi * 0.2)
E.choice("s18p12b", f"${f(2 * zb, 2)}$", "math の log・cos で計算")
table_ok("s18p12b", [(-1.2040, math.log(.3)), (-1.6094, math.log(.2)), (0.3090, math.cos(.4 * math.pi)),
                     (0.9511, math.sin(.4 * math.pi)), (-0.3090, math.cos(.6 * math.pi))], 4, "log・cos・sin の値")
u1 = rng.random(500_000)
u2 = rng.random(500_000)
zs = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
E.check("s18p12b", abs(zs.mean()) < 0.01 and abs(zs.std() - 1) < 0.01, "ボックス・ミュラー法の出力が標準正規（平均0・標準偏差1）")

E.done()
