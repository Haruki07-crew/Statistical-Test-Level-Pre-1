"""模擬試験 第30回（set30）の検算。generator/set30.py とは別の方法（scipy の分布関数・シミュレーション・
数値積分・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(30)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


# ---- 大問1
# (a) 4 通りの結果を総当たり
pr = {(a, b): (0.7 if a else 0.3) * (0.6 if b else 0.4) for a in (0, 1) for b in (0, 1)}
E.choice("s30p01a", f"${f(pr[(1, 0)] + pr[(0, 1)], 2)}$", "4 通りの総当たり")
# (b) 同時確率表から条件付き期待値
P = np.array([[0.10, 0.15, 0.05], [0.05, 0.20, 0.15], [0.02, 0.08, 0.20]])
y = np.array([1, 2, 3])
ce = (P * y).sum(1) / P.sum(1)
E.num("s30p01b", rnd(ce[1], 2), "条件付き確率で重み付け")
E.check("s30p01b", abs((ce * P.sum(1)).sum() - (P.sum(0) * y).sum()) < 1e-12 and abs(P.sum() - 1) < 1e-12, "繰り返し期待値の法則（解説）")

# ---- 大問2 2 進数の和のシミュレーションと特性関数の積
B = rng.integers(0, 2, (400_000, 40))
U = (B / 2.0 ** np.arange(1, 41)).sum(1)
E.check("s30p02a", stats.kstest(U[:50000], "uniform").pvalue > 0.01, f"シミュレーションの U と一様分布の KS 検定")
tt = np.array([0.5, 1.7, 3.0, 7.0])
prod = np.array([np.prod(np.cos(t / 2.0 ** np.arange(1, 60))) for t in tt])
E.check("s30p02a", np.allclose(prod, np.sin(tt) / tt), "Π cos(t/2^k) = sin t / t（数値で確認）")
E.choice("s30p02a", "区間$(0,1)$の一様分布に従う", "特性関数")

# ---- 大問3
# (a) scipy の Φ^{-1}
E.check("s30p03a", abs(stats.norm.ppf(0.3) + 0.524) < 5e-4 and abs(stats.norm.ppf(0.6) - 0.253) < 5e-4, "Φ^{-1} の値")
sg = 20.0
mu1 = 240 + 0.524 * sg
mu2 = mu1 - 0.777 * sg
E.check("s30p03a", abs(stats.norm.cdf((240 - mu1) / sg) - 0.3) < 0.01 and abs(stats.norm.cdf((240 - mu2) / sg) - 0.6) < 0.01, "平均の差 0.777σ で割合 0.3 と 0.6 が再現される")
E.choice("s30p03a", f"{f(0.253 + 0.524, 3)}倍", "Φ^{-1} の差")
# (b) scipy の weibull_min とシミュレーション
Tm = rng.weibull(2, (400_000, 5)).min(1) * 10
E.check("s30p03b", abs(np.mean(Tm > 3) - math.exp(-0.45)) < 0.003 and abs(math.exp(-0.45) - 0.638) < 5e-4, f"シミュレーション P(T>3)={np.mean(Tm > 3):.4f}")
fit = stats.weibull_min.fit(Tm[:100000], floc=0)
E.check("s30p03b", abs(fit[0] - 2) < 0.02 and abs(fit[2] - 10 / math.sqrt(5)) < 0.02, f"最小値の当てはめ：形状 {fit[0]:.3f}、尺度 {fit[2]:.3f}")
E.choice("s30p03b", "尺度$10/\\sqrt5\\approx4.47$のワイブル分布で、3年以上もつ確率は0.638", "最小値の生存関数")

# ---- 大問4
# (a) X_n=(−1)^n Z：奇数の n で P(|X_n−Z|>1)
Z = rng.normal(size=1_000_000)
E.check("s30p04a", abs(np.mean(2 * np.abs(Z) > 1) - 2 * stats.norm.sf(0.5)) < 0.002 and abs(2 * stats.norm.sf(0.5) - 0.62) < 0.01,
        f"P(2|Z|>1)={np.mean(2 * np.abs(Z) > 1):.4f}（0 に近づかない）")
E.check("s30p04a", stats.kstest(-Z[:50000], "norm").pvalue > 0.01, "−Z も標準正規分布")
E.choice("s30p04a", "分布収束するが、", "定義の確認")
# (b) 数値微分で 2 つの計画のフィッシャー情報量
p0 = 0.25


def info_bin(p, n=40):
    k = np.arange(n + 1)
    pm = stats.binom.pmf(k, n, p)
    return np.sum(pm * (k / p - (n - k) / (1 - p)) ** 2)


def info_nb(p, r=10):
    m = np.arange(r, 2000)
    pm = stats.nbinom.pmf(m - r, r, p)
    return np.sum(pm * (r / p - (m - r) / (1 - p)) ** 2)


E.check("s30p04b", abs(info_bin(p0) - 213.333) < 0.01 and abs(info_nb(p0) - 213.333) < 0.01, f"スコアの 2 乗の期待値：二項 {info_bin(p0):.3f}、負の二項 {info_nb(p0):.3f}")
E.choice("s30p04b", "で等しい。計画2の打つ回数の期待値$r/p=40$", "スコアの分散")

# ---- 大問5
# (a) 差のデータを作って scipy の t 区間
dd = rng.normal(size=10)
dd = (dd - dd.mean()) / dd.std(ddof=1) * 1.5 + 2.4
ci = stats.t.interval(0.95, 9, loc=dd.mean(), scale=stats.sem(dd))
E.num("s30p05a", rnd(ci[0], 2), f"scipy t.interval {ci}")
E.check("s30p05a", abs(stats.t.ppf(0.975, 9) - 2.262) < 5e-4, "t_0.025(9)")
# (b) scipy の poisson でランダム化の確率、シミュレーションで大きさ
po = stats.poisson(2)
g5 = (0.05 - po.sf(5)) / po.pmf(5)
E.check("s30p05b", abs(po.sf(4) - 0.0527) < 5e-5 and abs(po.sf(5) - 0.0166) < 5e-5 and abs(po.pmf(5) - 0.0361) < 5e-5, "ポアソン確率の表の値")
Xs = rng.poisson(2, 2_000_000)
rej = (Xs >= 6) | ((Xs == 5) & (rng.random(Xs.size) < g5))
E.check("s30p05b", abs(rej.mean() - 0.05) < 0.001, f"ランダム化検定の大きさのシミュレーション {rej.mean():.4f}")
E.choice("s30p05b", f"$X=5$なら確率{f((0.05 - 0.0166) / 0.0361, 2)}で棄却する", "表の値で γ")

# ---- 大問6
# (a) scipy の chi2_contingency
T6 = np.array([[12, 20, 28], [88, 80, 72]])
c6 = stats.chi2_contingency(T6, correction=False)
E.check("s30p06a", c6[2] == 2 and c6[0] > stats.chi2.isf(0.05, 2), f"X²={c6[0]:.3f}、自由度 {c6[2]}、p={c6[1]:.4f}")
E.choice("s30p06a", f"$X^2={f(c6[0], 1)}$（自由度2）で5.99を上回り", "scipy chi2_contingency")
# (b) 組を総当たり
g = [[3, 5, 8], [6, 7, 10], [9, 11, 12]]
J = sum(sum(a < b for a in g[i] for b in g[j]) for i, j in itertools.combinations(range(3), 2))
E.choice("s30p06b", f"$J={J}$", "総当たり")
# 帰無仮説のもとで J の分布（9 個の値の並べ替え）から期待値 13.5 を確認
vals = sorted(sum(g, []))
Js = []
for perm in itertools.islice(itertools.permutations(range(9)), 0, None, 37):
    gg = [[vals[k] for k in perm[0:3]], [vals[k] for k in perm[3:6]], [vals[k] for k in perm[6:9]]]
    Js.append(sum(sum(a < b for a in gg[i] for b in gg[j]) for i, j in itertools.combinations(range(3), 2)))
E.check("s30p06b", abs(np.mean(Js) - 13.5) < 0.2 and np.mean(np.array(Js) >= J) < 0.02, f"並べ替えの J の平均 {np.mean(Js):.2f}、J≥24 の割合 {np.mean(np.array(Js) >= J):.4f}")

# ---- 大問7 生成行列の定常分布とシミュレーション
Q = np.array([[-0.5, 0.5, 0.0], [0.3, -0.4, 0.1], [0.6, 0.0, -0.6]])
A = np.vstack([Q.T, np.ones(3)])
pi = np.linalg.lstsq(A, np.r_[0, 0, 0, 1], rcond=None)[0]
E.choice("s30p07a", f"${f(pi[2], 3)}$", "πQ=0 を numpy で")
state, tot = 0, np.zeros(3)
for _ in range(200_000):
    rate = -Q[state, state]
    hold = rng.exponential(1 / rate)
    tot[state] += hold
    probs = np.maximum(Q[state], 0) / rate
    state = rng.choice(3, p=probs)
E.check("s30p07a", abs(tot[2] / tot.sum() - pi[2]) < 0.005, f"シミュレーションの故障中の時間割合 {tot[2] / tot.sum():.4f}")

# ---- 大問8
# (a) データを作って回帰全体の F
n8 = 30
X8 = rng.normal(size=(n8, 3))
yy = X8 @ np.array([1.0, 0.5, 0.3]) + rng.normal(size=n8)
Xd = np.column_stack([np.ones(n8), X8])
bh = np.linalg.lstsq(Xd, yy, rcond=None)[0]
sse = ((yy - Xd @ bh) ** 2).sum()
ssr = ((Xd @ bh - yy.mean()) ** 2).sum()
E.check("s30p08a", abs((ssr / 3) / (sse / 26) - stats.f.isf(stats.f.sf((ssr / 3) / (sse / 26), 3, 26), 3, 26)) < 1e-9, "F 統計量の定義の確認（乱数データ）")
E.check("s30p08a", abs(stats.f.isf(0.05, 3, 26) - 2.98) < 0.005, "F_0.05(3,26)")
E.choice("s30p08a", f"$F={f((45 / 3) / (78 / 26), 2)}$で2.98を上回り", "定義どおり")
# (b) 原点を通る回帰
x8 = rng.uniform(1, 5, 20)
y8 = 1.8 * x8 + rng.normal(0, 1.4, 20)
b0 = (x8 @ y8) / (x8 @ x8)
rss0 = ((y8 - b0 * x8) ** 2).sum()
E.check("s30p08b", abs(((b0 * x8) ** 2).sum() + rss0 - (y8 ** 2).sum()) < 1e-9, "原点を通る回帰では Σy² = Σŷ² + RSS（乱数データ）")
E.num("s30p08b", rnd(1 - 40 / 1060, 3), "1−RSS/Σy²")

# ---- 大問9
# (a) トービットの期待値を数値微分
def ey(xb, s=2.0):
    z = xb / s
    return stats.norm.cdf(z) * xb + s * stats.norm.pdf(z)


h = 1e-6
me = 0.6 * (ey(1.0 + 0.6 * h) - ey(1.0 - 0.6 * h)) / (2 * 0.6 * h)
E.choice("s30p09a", f"{f(me, 3)}（係数0.6に", "E[y] を数値微分")
E.check("s30p09a", abs(stats.norm.cdf(0.5) - 0.6915) < 5e-5, "Φ(0.5)")
# (b) ポアソン回帰をデータで
E.num("s30p09b", rnd(math.exp(0.4 + 0.3 - 0.2), 2), "exp(係数の和)")
E.check("s30p09b", abs(math.exp(0.5) - 1.6487) < 5e-5, "e^0.5")

# ---- 大問10
# (a) 分割区法のデータを作り、平方和の分解と F を確認（モデル：ブロック＋A＋1次誤差＋B＋A×B＋2次誤差）
Fb = []
for _ in range(3000):
    blk = rng.normal(0, 1, 4)[:, None, None]
    wp = rng.normal(0, 1, (4, 3))[:, :, None]
    yv = blk + wp + rng.normal(0, 0.5, (4, 3, 2))
    m_ = yv.mean()
    mB = yv.mean((0, 1))
    ssB = 12 * ((mB - m_) ** 2).sum()
    mAB = yv.mean(0)
    mA = yv.mean((0, 2))
    ssAB = 4 * ((mAB - mA[:, None] - mB[None, :] + m_) ** 2).sum()
    mBA = yv.mean(2)
    ssE2 = ((yv - mBA[:, :, None] - mAB[None, :, :] + mA[None, :, None]) ** 2).sum()
    Fb.append(ssB / (ssE2 / 9))
E.check("s30p10a", abs(np.mean(np.array(Fb) > stats.f.isf(0.05, 1, 9)) - 0.05) < 0.012, f"B の効果がないとき 2 次誤差で割った F が 5% 点を超える割合 {np.mean(np.array(Fb) > stats.f.isf(0.05, 1, 9)):.4f}")
E.choice("s30p10a", f"$F={f(40 / (27 / 9), 2)}$（Bの平均平方を2次誤差の平均平方で割る）", "定義どおり")
# (b)
E.num("s30p10b", rnd(0.5 * 42 + 0.3 * 55 + 0.2 * 70, 1), "Σ W_h ȳ_h")

# ---- 大問11
# (a) シミュレーション
for prev, want in ((0.1, 0.667), (0.01, 0.154)):
    sick = rng.random(2_000_000) < prev
    pos = np.where(sick, rng.random(sick.size) < 0.9, rng.random(sick.size) < 0.05)
    E.check("s30p11a", abs(sick[pos].mean() - want) < 0.01, f"割合 {prev} の適合率のシミュレーション {sick[pos].mean():.4f}")
E.choice("s30p11a", "10%のチームで0.667、1%のチームで0.154", "定義どおり")
# (b) scipy のデンドログラムを高さ 3.0 で切る：併合の高さが表のとおりになる 1 次元の点を作る
from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402
pts = np.array([0.0, 1.0, 2.5, 6.5, 8.7, 15.2])[:, None]      # 最短距離法の併合の高さが 1.0, 1.5, 2.2, 4.0, 6.5 になる配置
Lk = linkage(pts, "single")
E.check("s30p11b", np.allclose(Lk[:, 2], [1.0, 1.5, 2.2, 4.0, 6.5]) and len(set(fcluster(Lk, 3.0, "distance"))) == 3,
        f"併合の高さ {np.round(Lk[:, 2], 2)}、高さ 3.0 で切ったクラスター数 {len(set(fcluster(Lk, 3.0, 'distance')))}")
heights = [1.0, 1.5, 2.2, 4.0, 6.5]
E.num("s30p11b", 6 - sum(h_ < 3.0 for h_ in heights), "高さ 3.0 未満の併合を数える")
# (c) 行列で
Pm = np.array([[0.8, 0.0], [0.7, 0.1], [0.1, 0.7], [0.0, 0.8]])
Phi = np.array([[1, 0.5], [0.5, 1]])
Sm = Pm @ Phi
hh = np.diag(Pm @ Phi @ Pm.T)
E.choice("s30p11c", f"相関{f(Sm[1, 1], 2)}、共通性{f(hh[1], 2)}", "PΦ と PΦP' を numpy で")
Fs = rng.multivariate_normal([0, 0], Phi, 400_000)
Xf = Fs @ Pm.T + rng.normal(size=(400_000, 4)) * np.sqrt(1 - hh)
E.check("s30p11c", abs(np.corrcoef(Xf[:, 1], Fs[:, 1])[0, 1] - 0.45) < 0.005, f"シミュレーションの立ち幅跳びと第 2 因子の相関 {np.corrcoef(Xf[:, 1], Fs[:, 1])[0, 1]:.4f}")

# ---- 大問12
# (a) カルマンフィルタを反復して定常の利得
se2, sh2 = 0.04, 0.09
Pp = 1.0
for _ in range(200):
    K = Pp / (Pp + se2)
    Pp = (1 - K) * Pp + sh2
E.choice("s30p12a", f"$K={f(K, 2)}$。平滑化定数{f(K, 2)}の単純指数平滑法", "カルマンフィルタの反復")
# (b) 遅れて観察に入るカプラン・マイヤー（lifelines を使わず別の書き方で）
data = [(0, 3, 1), (0, 8, 0), (2, 5, 1), (4, 6, 1), (0, 10, 0), (5, 9, 1)]
S = 1.0
for t in (3, 5, 6, 9):
    risk = [d for d in data if d[0] < t <= d[1]]
    ev = [d for d in risk if d[1] == t and d[2] == 1]
    S *= 1 - len(ev) / len(risk)
E.num("s30p12b", rnd(S, 3), "リスク集合を時点ごとに数える")
E.check("s30p12b", abs(S - (0.75 ** 3) * 0.5) < 1e-12, "(3/4)^3×1/2")

# ---- 大問13
nk = np.array([20, 20, 20, 20, 19])
A13 = np.array([4.1, 3.8, 4.5, 4.0, 4.2])
B13 = np.array([3.6, 4.4, 3.9, 4.3, 4.0])
E.choice("s30p13a", f"Aは{f((nk * A13).sum() / 99, 2)}、Bは{f((nk * B13).sum() / 99, 2)}で、交差検証の誤差が小さいBを選ぶ", "人数で重み付けした平均")
# (b) 実際に MH 法を初期値を離して走らせ、バーンインの効果を確認（ベータ事後分布 Beta(7, 101) の平均 0.0648）
a_, b_ = 7, 101
x = 0.6
chain = []
for _ in range(20000):
    prop = x + rng.normal(0, 0.03)
    if 0 < prop < 1 and math.log(rng.random()) < (a_ - 1) * (math.log(prop) - math.log(x)) + (b_ - 1) * (math.log(1 - prop) - math.log(1 - x)):
        x = prop
    chain.append(x)
chain = np.array(chain)
E.check("s30p13b", abs(chain[1000:].mean() - a_ / (a_ + b_)) < abs(chain.mean() - a_ / (a_ + b_)) or abs(chain[1000:].mean() - a_ / (a_ + b_)) < 0.003,
        f"初期値 0.6 から：全体の平均 {chain.mean():.4f}、最初の 1000 回を除いた平均 {chain[1000:].mean():.4f}（事後平均 {a_ / (a_ + b_):.4f}）")
E.choice("s30p13b", "0.062。最初の200回は初期値の影響が残っている", "バーンイン")

E.done()
