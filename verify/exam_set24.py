"""模擬試験 第24回（set24）の検算。generator/set24.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値最適化）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(24)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


def nearest(iid, value):
    """選択肢の最初の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1
# (a) ハザードから個人ごとに解約月を発生させるシミュレーションと、分布を直接組み立てた値
h = [0.20, 0.15, 0.10, 0.08]
pmf, alive = [], Fraction(1)
for hk in h:
    hk = Fraction(str(hk))
    pmf.append(alive * hk)
    alive *= 1 - hk
E.choice("s24p01a", f"${f(float(pmf[3]), 4)}$", "分数で P(T=4) を組み立てた")
U = rng.random((400_000, 4))
dead = U < np.array(h)
first = np.where(dead.any(1), dead.argmax(1) + 1, 99)
E.check("s24p01a", abs((first == 4).mean() - float(pmf[3])) < 0.0015, f"シミュレーション P(T=4)={(first == 4).mean():.4f}")
# (b) 打ち切りつき指数モデル：観測情報量から log λ の分散、正確な正規点で上限
d1, T1, d2, T2 = 30, 500, 24, 800
lam = sp.symbols("lam", positive=True)
ll = lambda d, T: d * sp.log(lam) - lam * T
var_log = []
for d, T in [(d1, T1), (d2, T2)]:
    lhat = sp.solve(sp.diff(ll(d, T), lam), lam)[0]
    info = -sp.diff(ll(d, T), lam, 2).subs(lam, lhat)
    var_log.append(float((1 / info) / lhat ** 2))       # デルタ法で log λ̂ の分散
se = math.sqrt(sum(var_log))
up = math.exp(math.log((d1 / T1) / (d2 / T2)) + stats.norm.ppf(0.975) * se)
E.num("s24p01b", rnd(up, 2), f"観測情報量とデルタ法・正確な点で上限 {up:.4f}")
for x, v in [(0.16, 1.1735), (0.54, 1.7160), (0.75, 2.1170), (1.06, 2.8864), (1.23, 3.4212)]:
    E.check("s24p01b", abs(math.exp(x) - v) < 6e-5, f"e^{x} の表の値")
E.check("s24p01b", abs(math.log(2) - 0.6931) < 5e-5, "log 2 の値")
# シミュレーション：打ち切りつきデータで log λ̂ の分散 ≈ 1/d
lam0, cens = 0.06, 15.0
tt = rng.exponential(1 / lam0, size=(20000, 60))
obs = np.minimum(tt, cens)
dd = (tt <= cens).sum(1)
lh = dd / obs.sum(1)
E.check("s24p01b", abs(np.var(np.log(lh)) * dd.mean() - 1) < 0.05, f"シミュレーションで V[log λ̂]·d ≈ {np.var(np.log(lh)) * dd.mean():.3f}")

# ---- 大問7
s_ = sp.symbols("s")
lt = 48
GK = sp.Rational(6, 10) * s_ + sp.Rational(3, 10) * s_ ** 2 + sp.Rational(1, 10) * s_ ** 3
GS = sp.exp(lt * (GK - 1))
m1 = sp.diff(GS, s_).subs(s_, 1)
m2 = sp.diff(GS, s_, 2).subs(s_, 1)
sdS = float(sp.sqrt(m2 + m1 - m1 ** 2))
E.num("s24p07b", rnd(sdS, 2), f"複合ポアソンの確率母関数を微分 sd={sdS:.4f}")
Nn = rng.poisson(lt, 200_000)
tot = np.array([rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1], size=q).sum() for q in Nn[:40000]])
E.check("s24p07b", abs(tot.std() - sdS) < 0.1, f"シミュレーション sd {tot.std():.3f}")

# ---- 大問10
tab = np.array([[30, 70], [20, 180]])
vid = np.repeat([1, 1, 0, 0], tab.ravel())
bz = np.repeat([1, 0, 1, 0], tab.ravel())
ll0 = stats.bernoulli(50 / 300).logpmf(bz).sum()
E.choice("s24p10b", f"${f(1 - (-112.0) / ll0, 3)}$", f"scipy の対数尤度 log L0={ll0:.4f}")
E.check("s24p10b", abs(math.log(1 / 6) + 1.7918) < 5e-5 and abs(math.log(5 / 6) + 0.1823) < 5e-5, "対数の値")
# 切片だけのロジスティック回帰を数値最大化しても同じ
opt = optimize.minimize_scalar(lambda a: -(bz * a - np.log1p(np.exp(a))).sum())
E.check("s24p10b", abs(-opt.fun - ll0) < 1e-6, "切片のみロジスティック回帰の最大対数尤度と一致")

# ---- 大問14：構造方程式のシミュレーション
Nn = 2_000_000
Zc = rng.normal(size=Nn)
Xc = 0.6 * Zc + rng.normal(size=Nn)
eY = rng.normal(size=Nn)
Yc = 0.3 * Xc + 0.5 * Zc + eY
slope = np.cov(Xc, Yc)[0, 1] / Xc.var()
coef = np.linalg.lstsq(np.column_stack([np.ones(Nn), Xc, Zc]), Yc, rcond=None)[0]
Ydo = 0.3 * (Xc + 1) + 0.5 * Zc + eY
E.check("s24p14a", abs(slope - 0.708 / 1.36) < 0.003, f"単回帰の傾き {slope:.4f}")
E.check("s24p14a", abs(coef[1] - 0.3) < 0.003 and abs((Ydo - Yc).mean() - 0.3) < 1e-9, f"重回帰の係数 {coef[1]:.4f}、介入による変化 0.3")
E.choice("s24p14a", f"約{f(slope, 2)}で、$Z$も説明変数に加えた重回帰の$X$の係数0.3が因果効果", "シミュレーションの傾き")

# ---- 作り替えた小問・検算を書き直した小問（2026-10-24）
# 大問2 (a) 混合分布のシミュレーションと全分散の公式
promo = rng.random(2_000_000) < 0.3
orders = np.where(promo, rng.normal(50, 10, promo.size), rng.normal(20, 5, promo.size))
v2a = 0.3 * 100 + 0.7 * 25 + 0.3 * 0.7 * (50 - 20) ** 2
E.choice("s24p02a", f"${f(math.sqrt(v2a), 2)}$", "全分散（平均の差の 2 乗 × p(1−p) の形）")
E.check("s24p02a", abs(orders.std() - math.sqrt(v2a)) < 0.03, f"シミュレーションの標準偏差 {orders.std():.3f}")
# (b) 回答者 800 人分のウェイトを並べて直接計算
w2b = np.repeat([50.0, 200.0], [400, 400])
deff = len(w2b) * (w2b ** 2).sum() / w2b.sum() ** 2
E.num("s24p02b", round(len(w2b) / deff), f"ウェイトの配列から deff={deff:.4f}")
E.check("s24p02b", abs(w2b.sum() ** 2 / (w2b ** 2).sum() - len(w2b) / deff) < 1e-9, "有効標本サイズ (Σw)²/Σw² と一致")

# 大問3 (a) scipy で log 比の分布、シミュレーション
for zz_, pv_ in ((0.35, 0.6368), (0.49, 0.6879), (0.69, 0.7549)):
    E.check("s24p03a", abs(stats.norm.cdf(zz_) - pv_) < 5e-5, f"Φ({zz_}) の表の値")
y1, y2 = np.exp(rng.normal(0, 1, 2_000_000)), np.exp(rng.normal(0, 1, 2_000_000))
p3a = stats.norm.sf(math.log(2) / math.sqrt(2))
E.check("s24p03a", abs(np.mean(y1 >= 2 * y2) - p3a) < 0.002, f"シミュレーション {np.mean(y1 >= 2 * y2):.4f}、正確な値 {p3a:.4f}")
E.choice("s24p03a", f"${f(1 - 0.6879, 3)}$", "z=0.49 で表を使った値")
# (b) 相関のある 2 変量正規でマハラノビス距離の 2 乗の分布
Sg = np.array([[1.0, 0.7], [0.7, 2.0]])
Xm = rng.multivariate_normal([0, 0], Sg, 1_000_000)
D2 = np.einsum("ij,jk,ik->i", Xm, np.linalg.inv(Sg), Xm)
E.check("s24p03b", abs(np.mean(D2 <= 5.99) - 0.95) < 0.002 and abs(np.mean(D2 <= 3.84) - stats.chi2.cdf(3.84, 2)) < 0.002,
        f"シミュレーション：D²≤5.99 の割合 {np.mean(D2 <= 5.99):.4f}、D²≤3.84 の割合 {np.mean(D2 <= 3.84):.4f}")
E.check("s24p03b", abs(math.exp(-1.92) - 0.1466) < 5e-5 and abs(stats.chi2.isf(0.05, 2) - 5.99) < 0.005, "e^{-1.92} と χ²(2) の点")
E.choice("s24p03b", f"$c=5.99$。$c=3.84$の楕円に入るのは約{f(stats.chi2.cdf(3.84, 2), 3)}", "scipy chi2(2)")

# 大問4 (a) 長い期間のシミュレーションで比の和と比の平均
days = 2_000_000
big_ = rng.random(days) < 0.5
Iv = np.where(big_, 9000, 1000)
Sv = rng.binomial(Iv, np.where(big_, 0.01, 0.05))
E.num("s24p04a", rnd(Sv.sum() / Iv.sum(), 3), f"200 万日のシミュレーションの比の和 {Sv.sum() / Iv.sum():.5f}")
E.check("s24p04a", abs((Sv / Iv).mean() - 0.03) < 0.0005, f"比の平均は {(Sv / Iv).mean():.4f}（解説の 0.03）")
# (b) scipy の chisquare
obs4 = np.array([14, 18, 22, 25, 24, 19, 18])
cs4 = stats.chisquare(obs4)
E.check("s24p04b", abs(stats.chi2.isf(0.05, 6) - 12.59) < 0.005 and abs(stats.chi2.isf(0.05, 7) - 14.07) < 0.005, "χ² の点")
E.choice("s24p04b", f"統計量は{f(cs4.statistic, 2)}、自由度6で12.59を下回る", f"scipy chisquare（p 値 {cs4.pvalue:.3f}）")

# 大問5 (a) 配分を総当たりして半幅を最小化
hw5 = {na: 1.96 * math.sqrt(3600 / na + 900 / (180 - na)) for na in range(1, 180)}
best5 = min(hw5, key=hw5.get)
E.check("s24p05a", best5 == 120, f"総当たりで最小の n_A={best5}")
E.num("s24p05a", rnd(hw5[best5], 2), "総当たりの最小の半幅")
# (b) scipy の ttest_ind_from_stats（等分散／ウェルチ）と、プールした検定の第 1 種の誤りのシミュレーション
tp5 = stats.ttest_ind_from_stats(20, 30, 10, 0, 12, 30, equal_var=True)
tw5 = stats.ttest_ind_from_stats(20, 30, 10, 0, 12, 30, equal_var=False)
E.check("s24p05b", tp5.pvalue < 0.05 < tw5.pvalue, f"プール p={tp5.pvalue:.4f}、ウェルチ p={tw5.pvalue:.4f}")
E.check("s24p05b", abs(stats.t.ppf(0.975, 38) - 2.024) < 5e-4 and abs(stats.t.ppf(0.975, 10) - 2.228) < 5e-4, "t の点")
a5 = rng.normal(0, 30, (100_000, 10))
b5 = rng.normal(0, 12, (100_000, 30))
sp5 = np.sqrt((9 * a5.var(1, ddof=1) + 29 * b5.var(1, ddof=1)) / 38 * (1 / 10 + 1 / 30))
rej5 = np.mean(np.abs(a5.mean(1) - b5.mean(1)) / sp5 > stats.t.ppf(0.975, 38))
E.check("s24p05b", rej5 > 0.1, f"等平均のもとでプールした t 検定が棄却する割合 {rej5:.3f}（名目 0.05）")
E.choice("s24p05b", f"プールした$t={f(20 / math.sqrt((9 * 900 + 29 * 144) / 38 * (1 / 10 + 1 / 30)), 2)}$は5%で有意、ウェルチの$t={f(tw5.statistic, 2)}$は有意でない", "scipy の t 統計量")

# 大問6 (a) T(T−1)/n² の不偏性を正確な和で、分散の比較をシミュレーションで
lam6, n6 = 3.0, 10
Tk = np.arange(0, 200)
pT = stats.poisson.pmf(Tk, n6 * lam6)
E.check("s24p06a", abs((Tk * (Tk - 1) / n6 ** 2 * pT).sum() - lam6 ** 2) < 1e-9 and abs(((Tk / n6) ** 2 * pT).sum() - lam6 ** 2 - lam6 / n6) < 1e-9,
        "E[T(T−1)/n²]=λ²、E[X̄²]=λ²+λ/n を正確な和で確認")
X6 = rng.poisson(lam6, (200_000, n6))
u1 = X6.sum(1) * (X6.sum(1) - 1) / n6 ** 2
u2 = (X6 * (X6 - 1)).mean(1)
E.check("s24p06a", abs(u2.mean() - 9) < 0.03 and u1.var() < u2.var(), f"どちらも不偏で、分散は T(T−1)/n² {u1.var():.3f} < 平均 {u2.var():.3f}")
xs6 = np.array([2, 5, 1, 3, 4, 2, 6, 3, 2, 2])
E.choice("s24p06a", f"T(T-1)/n^2={f(xs6.sum() * (xs6.sum() - 1) / 100, 2)}$が一様最小分散不偏推定量", "データから計算")
# (b) パーセンタイル法：表の 2.5% 点と 97.5% 点（実際のブートストラップで手順も確認）
E.choice("s24p06b", "$(41.0,\\58.5)$", "パーセンタイル法の定義")
samp6 = rng.poisson(48, 40)
boot = np.median(rng.choice(samp6, (2000, 40)), axis=1)
E.check("s24p06b", np.quantile(boot, 0.025) <= np.median(samp6) <= np.quantile(boot, 0.975), "乱数データのブートストラップで、パーセンタイル区間が標本中央値を含む（手順の確認）")

# 大問7 (a) シミュレーションと正確な値
draw7 = rng.integers(0, 6, (400_000, 10))
ndist = np.array([len(set(r_)) for r_ in draw7[:100_000]])
E.num("s24p07a", rnd(6 * (1 - 0.1615), 3), "6(1−(5/6)^10)（問題文の値で）")
E.check("s24p07a", abs(ndist.mean() - 6 * (1 - (5 / 6) ** 10)) < 0.01 and abs((5 / 6) ** 10 - 0.1615) < 5e-5, f"シミュレーション {ndist.mean():.4f}")

# 大問8 (a) 要約統計量を満たすデータを作って numpy で傾きの区間
n8 = 12
x8 = rng.normal(size=n8)
x8 = (x8 - x8.mean()) / np.sqrt(((x8 - x8.mean()) ** 2).sum()) * np.sqrt(20) + 3
X8 = np.column_stack([np.ones(n8), x8])
e8 = rng.normal(size=n8)
e8 = e8 - X8 @ np.linalg.lstsq(X8, e8, rcond=None)[0]
e8 = e8 / np.sqrt(e8 @ e8) * np.sqrt(40)
y8 = 2 + 1.5 * x8 + e8
bh8 = np.linalg.lstsq(X8, y8, rcond=None)[0]
s28 = ((y8 - X8 @ bh8) ** 2).sum() / (n8 - 2)
se8 = math.sqrt(s28 * np.linalg.inv(X8.T @ X8)[1, 1])
t10 = stats.t.ppf(0.975, 10)
E.check("s24p08a", abs(bh8[1] - 1.5) < 1e-9 and abs(s28 - 4) < 1e-9 and abs(t10 - 2.228) < 5e-4, "作ったデータの当てはめが問題文と一致")
E.choice("s24p08a", f"({f(1.5 - t10 * se8, 2)},\\{f(1.5 + t10 * se8, 2)})", "行列計算と正確な t 点で区間")
# (b) ブルーシュ・ペーガン：補助回帰の nR² と、帰無仮説のもとでの分布のシミュレーション
E.choice("s24p08b", f"{12}\\times0.42={f(12 * 0.42, 2)}$で、自由度1の3.84を上回る", "nR² と自由度 1")
bps = []
for _ in range(20000):
    yy8 = 2 + 1.5 * x8 + rng.normal(0, 2, n8)
    ee = yy8 - X8 @ np.linalg.lstsq(X8, yy8, rcond=None)[0]
    g = ee ** 2
    gh = X8 @ np.linalg.lstsq(X8, g, rcond=None)[0]
    bps.append(n8 * (1 - ((g - gh) ** 2).sum() / ((g - g.mean()) ** 2).sum()))
E.check("s24p08b", np.mean(np.array(bps) > 5.04) < 0.06, f"等分散のもとで nR²>5.04 となる割合 {np.mean(np.array(bps) > 5.04):.4f}（小さい）")

# 大問9 直交計画で C_p を全 64 通りの部分集合で計算
b9 = np.array([2.1, -0.9, 1.5, 0.3, -1.8, 0.6])
rss_full = 50.0   # 任意（C_p の比較には差だけが効く）
best9 = min(itertools.chain.from_iterable(itertools.combinations(range(6), k) for k in range(7)),
            key=lambda S_: rss_full + (b9 ** 2).sum() - (b9[list(S_)] ** 2).sum() + 2 * len(S_))
E.check("s24p09a", tuple(best9) == (0, 2, 4) and [j for j in range(6) if abs(b9[j]) > 1.96] == [0], f"C_p 最小の組 {[j + 1 for j in best9]}")
E.choice("s24p09a", "$C_p$：$x_1,x_3,x_5$、$t$検定：$x_1$", "全部分集合の C_p")

# 大問10 (a) リスク比とオッズ比、列を固定した抽出のシミュレーション
E.choice("s24p10a", f"リスク比{f((30 / 100) / (20 / 200), 2)}、オッズ比{f(30 * 180 / (70 * 20), 2)}。列の合計を固定した抽出では、オッズ比は同じ意味で推定できるが、リスク比は推定できない", "定義どおり")
# 拡散 50・非拡散 250 を母集団から別々に抽出しても、オッズ比は同じ値に近づき、見かけの「拡散する割合」は抽出数で決まる
pop_v = np.repeat([1, 1, 0, 0], [3000, 7000, 2000, 18000])
pop_s = np.repeat([1, 0, 1, 0], [3000, 7000, 2000, 18000])
ors, rrs = [], []
for _ in range(2000):
    cs_ = np.concatenate([rng.choice(pop_v[pop_s == 1], 500), rng.choice(pop_v[pop_s == 0], 500)])
    ss_ = np.repeat([1, 0], [500, 500])
    a_, b_ = ((cs_ == 1) & (ss_ == 1)).sum(), ((cs_ == 1) & (ss_ == 0)).sum()
    c_, d_ = ((cs_ == 0) & (ss_ == 1)).sum(), ((cs_ == 0) & (ss_ == 0)).sum()
    ors.append(np.log(a_ * d_ / (b_ * c_)))
    rrs.append((a_ / (a_ + b_)) / (c_ / (c_ + d_)))
E.check("s24p10a", abs(math.exp(np.mean(ors)) - 30 * 180 / (70 * 20)) < 0.1 and abs(np.median(rrs) - 3.0) > 0.5,
        f"列固定の抽出で OR≈{math.exp(np.mean(ors)):.3f}（母集団 3.857）、見かけのリスク比 {np.median(rrs):.3f}（母集団 3.00 と違う）")

# 大問11 (a) scipy の kendalltau と総当たり
kt = stats.kendalltau([1, 2, 3, 4, 5, 6, 7], [2, 1, 4, 3, 6, 7, 5])
E.choice("s24p11a", f"一致対17、不一致対4で、$\\tau={f(kt.statistic, 3)}$", "scipy kendalltau")
E.check("s24p11a", f(stats.spearmanr([1, 2, 3, 4, 5, 6, 7], [2, 1, 4, 3, 6, 7, 5])[0], 3) == "0.821", "スピアマン 0.821（解説）")
# (b) 期待コストを numpy で
tp11 = np.array([0.95, 0.88, 0.75, 0.50])
fp11 = np.array([0.30, 0.15, 0.05, 0.01])
cost11 = 0.10 * (1 - tp11) * 5 + 0.90 * fp11 * 1
E.choice("s24p11b", f"$T_{int(np.argmin(cost11)) + 1}$：期待コストが約{f(cost11.min(), 3)}で最小", "numpy で期待コスト")
E.check("s24p11b", int(np.argmax(tp11 - fp11)) == 1 and int(np.argmin(0.1 * (1 - tp11) + 0.9 * fp11)) == 3, "ユーデン指数は T2、誤判定率は T4 が最良（誤答の根拠）")

# 大問12 (a) ジャカード係数（scipy の距離）
A12 = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0], bool)
B12 = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0], bool)
from scipy.spatial.distance import jaccard, hamming  # noqa: E402
E.num("s24p12a", rnd(1 - jaccard(A12, B12), 3), "scipy の jaccard 距離から")
E.check("s24p12a", abs((1 - hamming(A12, B12)) - 0.75) < 1e-12, "単純一致係数 0.75（解説）")
# (b) 二重中心化した行列の固有値
D12 = np.array([[0, 2, 5], [2, 0, 2], [5, 2, 0]], float)
J12 = np.eye(3) - 1 / 3
B12m = -0.5 * J12 @ (D12 ** 2) @ J12
ev12 = np.sort(np.linalg.eigvalsh(B12m))[::-1]
E.check("s24p12b", np.allclose(ev12, [12.5, 0, -1.5]) and abs(B12m[1, 1] + 1) < 1e-12 and abs(B12m[0, 2] + 6.5) < 1e-12, f"固有値 {np.round(ev12, 6)}、b_QQ={B12m[1, 1]}")
E.choice("s24p12b", "負の固有値があり、この非類似度はユークリッド空間の点の距離としては正確に表せない", "固有値の符号")

# 大問13 データを作って分散分析し、平方和と η²
from itertools import product  # noqa: E402
SS13 = {"A": 48, "B": 12, "C": 3, "AB": 9, "AC": 1, "BC": 2, "ABC": 1, "E": 24}
E.check("s24p13a", sum(SS13.values()) == 100, "平方和の合計")
E.choice("s24p13a", f"\\eta^2={f(48 / 100, 3)}$、偏$\\eta^2={f(48 / 72, 3)}$、$F={f(48 / 3, 1)}", "定義どおり")
# 2^3×2 回のデータで、偏 η² = F·1/(F·1+8) が成り立つことを確認
runs = np.array(list(product([-1, 1], repeat=3)) * 2, float)
y13 = 0.5 + 0.8 * runs[:, 0] + 0.3 * runs[:, 1] + rng.normal(0, 0.4, 16)
Xd = np.column_stack([np.ones(16), runs, runs[:, 0] * runs[:, 1], runs[:, 0] * runs[:, 2], runs[:, 1] * runs[:, 2], runs.prod(1)])
coef13 = np.linalg.lstsq(Xd, y13, rcond=None)[0]
ssA = 16 * coef13[1] ** 2
ssE = ((y13 - Xd @ coef13) ** 2).sum()
F13 = ssA / (ssE / 8)
E.check("s24p13a", abs(ssA / (ssA + ssE) - F13 / (F13 + 8)) < 1e-12, "乱数データで 偏η² = F/(F+8)")

E.done()
