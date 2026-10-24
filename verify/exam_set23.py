"""模擬試験 第23回（set23）の検算。generator/set23.py とは別の方法（全列挙・scipy の分布関数・数値積分・
数値最適化・シミュレーション・sympy）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(23)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def near(a, b, tol):
    return abs(a - b) <= tol


def nearest(iid, value):
    """選択肢の最初の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1：正方形上の一様分布をシミュレーションし、格子での数値積分でも確かめる
X1, Y1 = rng.uniform(0, 60, 1_000_000), rng.uniform(0, 60, 1_000_000)
ov = np.mean(np.abs(X1 - Y1) < 12)
g_ = np.linspace(0, 60, 1201)
area = np.mean(np.abs(g_[:, None] - g_[None, :]) < 12)
E.num("s23p01a", rnd(1 - (48 / 60) ** 2, 3), f"シミュレーション {ov:.4f}、格子 {area:.4f}")
E.check("s23p01a", abs(ov - 0.36) < 0.002 and abs(area - 0.36) < 0.003, "シミュレーションと格子の値が 0.360 に近い")

# ---- 大問2
# (a) 作問に使ったデータから ℓ_A, ℓ_B*, Σln y を求め直し、scipy の lognorm の対数尤度でヤコビアンを確かめる
y = np.array([1650, 1820, 1950, 2380, 1720, 2100, 2750, 1880, 1560, 2290.])
lA = stats.norm.logpdf(y, y.mean(), y.std()).sum()
ly = np.log(y)
lBs = stats.norm.logpdf(ly, ly.mean(), ly.std()).sum()
lB = stats.lognorm.logpdf(y, s=ly.std(), scale=math.exp(ly.mean())).sum()
E.check("s23p02a", near(lA, -72.84, 0.005) and near(lBs, 3.58, 0.005) and near(ly.sum(), 75.91, 0.005),
        f"示した値 ℓ_A={lA:.3f}, ℓ_B*={lBs:.3f}, Σln y={ly.sum():.3f}")
E.check("s23p02a", near(lB, lBs - ly.sum(), 1e-8), "lognorm の y についての対数尤度 = ℓ_B* − Σln y")
aicA_raw, aicB_raw = -2 * lA + 4, -2 * lB + 4
E.check("s23p02a", aicB_raw < aicA_raw, f"生のデータでも AIC_B {aicB_raw:.2f} < AIC_A {aicA_raw:.2f}")
E.choice("s23p02a", f"${f(-2 * (3.58 - 75.91) + 4, 2)}$", "示した値からの AIC_B")
# (b) 2 変量正規の上側の象限確率を scipy の多変量正規で
mvn = stats.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
orth = mvn.cdf([0, 0])        # 対称性より P(U<0,V<0)=P(U>0,V>0)
E.choice("s23p02b", f"${f(orth, 3)}$", f"scipy の多変量正規分布 {orth:.5f}")

# ---- 大問3：モーメント法の方程式を sympy で解く
a_, b_ = sp.symbols("a b", real=True)
sol = sp.solve([(a_ + b_) / 2 - sp.Rational(40, 100), (b_ - a_) ** 2 / 12 - sp.Rational(12, 100)], [a_, b_], dict=True)
sol = [d_ for d_ in sol if d_[b_] > d_[a_]][0]
E.choice("s23p03a", f"$(\\hat a,\\hat b)=({f(float(sol[a_]), 2)},\\ {f(float(sol[b_]), 2)})$", f"sympy の解 {sol}")
U3 = rng.uniform(float(sol[a_]), float(sol[b_]), (20000, 20))
E.check("s23p03a", abs(U3.var(1).mean() * 20 / 19 - 0.12) < 0.003, "解いた区間の一様分布の分散が 0.12 と整合")

# ---- 大問4
# (a) ロジスティック分布（scipy）
E.num("s23p04a", rnd(stats.logistic.sf(2), 3), f"scipy logistic の上側確率 {stats.logistic.sf(2):.5f}")
U4 = rng.random(1_000_000)
E.check("s23p04a", abs(np.mean(np.log(U4 / (1 - U4)) > 2) - stats.logistic.sf(2)) < 0.002, "一様乱数をロジット変換したシミュレーション")
# (b) 標本 90% 点と X̄ log10 の標準偏差をシミュレーション
Xs4 = rng.exponential(2.0, (20000, 400))
q_s = np.quantile(Xs4, 0.9, axis=1)
q_m = Xs4.mean(1) * math.log(10)
E.check("s23p04b", abs(q_s.std() - 0.30) < 0.012 and abs(q_m.std() - 0.2303) < 0.006, f"シミュレーション (i) {q_s.std():.4f}、(ii) {q_m.std():.4f}")
E.choice("s23p04b", f"(i) ${f(math.sqrt(0.09 / (400 * stats.expon(scale=2).pdf(stats.expon(scale=2).ppf(0.9)) ** 2)), 3)}$、(ii) ${f(math.log(10) * 2 / 20, 3)}$", "scipy の密度と分位点")

# ---- 大問5
# (a) 混合分布のモーメントを数値積分
pdf = lambda x: 0.9 * stats.norm.pdf(x, 0, 1) + 0.1 * stats.norm.pdf(x, 0, 3)
m2 = integrate.quad(lambda x: x ** 2 * pdf(x), -np.inf, np.inf)[0]
m4 = integrate.quad(lambda x: x ** 4 * pdf(x), -np.inf, np.inf)[0]
E.choice("s23p05a", f"${f(m4 / m2 ** 2, 2)}$", "数値積分による尖度")
# (b) 混合分布の雑音を発生させて移動平均をとり、標本自己相関を計算
N = 400000
eps = np.where(rng.random(N) < 0.9, rng.normal(0, 1, N), rng.normal(0, 3, N))
Y = (eps[2:] + 2 * eps[1:-1] + eps[:-2]) / 4
Yc = Y - Y.mean()
acf = [np.dot(Yc[:-k], Yc[k:]) / np.dot(Yc, Yc) for k in (1, 2, 3)]
E.check("s23p05b", abs(acf[0] - 2 / 3) < 0.01 and abs(acf[1] - 1 / 6) < 0.01 and abs(acf[2]) < 0.01, f"標本自己相関 {np.round(acf, 4)}")
E.choice("s23p05b", f"({f(2 / 3, 3)},\\ {f(1 / 6, 3)},\\ 0)", "理論値")

# ---- 大問6
# (a) scipy の t 点と、3 区間が同時に含む割合のシミュレーション
tq = stats.t.isf(0.05 / 6, 5)
for shown, a6 in ((2.571, 0.025), (2.912, 0.05 / 3), (3.534, 0.05 / 6)):
    E.check("s23p06a", abs(stats.t.isf(a6, 5) - shown) < 5e-4, f"t の点 {shown}")
E.num("s23p06a", rnd(tq * 2.4 / math.sqrt(6), 2), f"scipy の t 点 {tq:.4f}")
Z6 = rng.normal(0, 1, (100_000, 3, 6))
m6, s6 = Z6.mean(2), Z6.std(2, ddof=1)
cover = np.all(np.abs(m6) <= tq * s6 / math.sqrt(6), axis=1).mean()
E.check("s23p06a", cover >= 0.95, f"3 区間が同時に含む割合 {cover:.4f}（95% 以上）")
# (b) ホッジス・レーマン推定量：全組の差の中央値
A6, B6 = np.array([30.1, 31.4, 32.0]), np.array([32.5, 33.0, 34.6, 35.2])
hl = np.median((B6[None, :] - A6[:, None]).ravel())
E.choice("s23p06b", f"${f(hl, 2)}$", f"numpy の中央値 {hl:.4f}")
U6 = lambda d_: np.sum((B6 - d_)[None, :] > A6[:, None]) + 0.5 * np.sum(np.isclose((B6 - d_)[None, :], A6[:, None]))
E.check("s23p06b", U6(hl) == 6, "Δ=HL で B−Δ と A の組の比較の数がちょうど期待値 6（=12/2）")

# ---- 大問7
# (a) 欠測値を変えて誤差平方和を数値最小化
Yt = np.array([[52, 55, 50], [58, 60, 0], [49, 53, 47], [56, 59, 54.]])
def sse(x):
    M = Yt.copy(); M[1, 2] = x
    R = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    return np.sum(R ** 2)
xm = optimize.minimize_scalar(sse, bounds=(40, 70), method="bounded").x
E.num("s23p07a", rnd(xm, 2), f"誤差平方和の数値最小化 x={xm:.5f}")
# (b) scipy の chisquare（推定した母数 2 個で ddof=2）
cs = stats.chisquare([3, 27, 30, 28, 12], [10, 23, 34, 23, 10.], ddof=2)
E.choice("s23p07b", f"統計量は {f(cs.statistic, 2)}、自由度 2 の 5.991 を上回るので、正規分布に従うという帰無仮説を棄却する", f"X²={cs.statistic:.4f}, p={cs.pvalue:.4f}")
E.check("s23p07b", cs.pvalue < 0.05 < stats.chi2.sf(cs.statistic, 3) and all(abs(stats.chi2.isf(0.05, k_) - v_) < 5e-4 for k_, v_ in ((2, 5.991), (3, 7.815), (4, 9.488))), "自由度 3・4 では棄却されない、数表の値")

# ---- 大問8：尤度比を数値で評価して単調性を調べる
xg = np.linspace(-30, 30, 6001)
lr_c = stats.cauchy.pdf(xg, loc=1) / stats.cauchy.pdf(xg, loc=0)
E.check("s23p08a", np.any(np.diff(lr_c) < -1e-9) and np.any(np.diff(lr_c) > 1e-9), "コーシー：尤度比が増えたり減ったりする（単調でない）")
xp = np.linspace(0.01, 30, 3000)
mono = [np.all(np.diff(stats.expon.pdf(xp, scale=2) / stats.expon.pdf(xp, scale=1)) >= -1e-12),
        np.all(np.diff(stats.norm.pdf(xg, 1) / stats.norm.pdf(xg, 0)) >= -1e-12)]
E.check("s23p08a", all(mono), "指数分布・正規分布の尤度比は単調")
E.choice("s23p08a", "コーシー分布", "単調でないのはコーシー分布")

# ---- 大問9：推移回数から最尤推定（多項尤度の数値最大化）して予測
Cn = np.array([[60, 15, 5], [0, 40, 10], [0, 0, 20.]])
Ph = np.zeros((3, 3))
for i in range(3):
    nz = Cn[i] > 0
    k_ = nz.sum()
    def nll(z):
        p_ = np.exp(np.r_[z, 0.0]); p_ /= p_.sum(); return -np.sum(Cn[i][nz] * np.log(p_))
    z0 = optimize.minimize(nll, np.zeros(k_ - 1)).x if k_ > 1 else np.zeros(0)
    p_ = np.exp(np.r_[z0, 0.0]); p_ /= p_.sum(); Ph[i][nz] = p_
pred = np.array([0.5, 0.3, 0.2]) @ Ph
E.choice("s23p09a", f"${f(pred[2], 3)}$", f"多項尤度の数値最大化による推移行列 {np.round(Ph, 4)}")

# ---- 大問10
# (a) データ（作問に使った値）で原点を通る回帰
x10 = np.arange(1, 7.); y10 = np.array([2.1, 4.3, 5.8, 8.4, 9.9, 12.3])
E.check("s23p10a", np.isclose((x10 * x10).sum(), 91) and np.isclose((x10 * y10).sum(), 185.0) and np.isclose((y10 * y10).sum(), 376.40), "問題文の和がデータと一致")
b10, res10 = np.linalg.lstsq(x10[:, None], y10, rcond=None)[:2]
E.num("s23p10a", rnd(math.sqrt(res10[0] / 5 / 91), 4), f"lstsq で β={b10[0]:.5f}")
# (b) 問題文の残差を読み取って DW を計算
q10 = E.item("s23p10b")["question"]
e10 = np.array([float(v_) for v_ in re.search(r"\$\$(.*?)\$\$", q10).group(1).replace("\\ ", "").split(",")])
dw = np.sum(np.diff(e10) ** 2) / np.sum(e10 ** 2)
E.choice("s23p10b", f"$d={f(dw, 2)}$。2 より大きく下回り、隣り合う月の残差に正の", f"DW={dw:.4f}")
# (c) 2 つの候補で θ1 を最小二乗、θ2 の連続な最適値も確認
t10, yy10 = np.array([7, 14, 28.]), np.array([22, 31, 38.])
def prof(th):
    g10 = 1 - np.exp(-th * t10); a10 = (yy10 @ g10) / (g10 @ g10); return a10, np.sum((yy10 - a10 * g10) ** 2)
(aA, rA), (aB, rB) = prof(0.10), prof(0.15)
opt10 = optimize.minimize_scalar(lambda th: prof(th)[1], bounds=(0.02, 0.5), method="bounded").x
E.check("s23p10c", rA < rB and abs(opt10 - 0.10) < abs(opt10 - 0.15), f"RSS 0.10:{rA:.3f}, 0.15:{rB:.3f}、連続の最適 θ2={opt10:.4f}")
E.choice("s23p10c", f"$\\theta_2=0.10$、$\\hat\\theta_1={f(aA, 1)}$", "正確な e の値で計算")
E.check("s23p10c", all(abs(math.exp(-a_) - v_) < 5e-5 for a_, v_ in ((0.7, 0.4966), (1.4, 0.2466), (2.8, 0.0608), (1.05, 0.3499), (2.1, 0.1225), (4.2, 0.0150))), "示した e の値")

# ---- 大問11
# (a) scipy.stats.logrank（打ち切りは CensoredData）
A = stats.CensoredData(uncensored=[5, 7, 11], right=[9])
B = stats.CensoredData(uncensored=[8, 12], right=[10, 14])
lr = stats.logrank(A, B)
E.num("s23p11a", rnd(lr.statistic ** 2, 2), f"scipy logrank（z={lr.statistic:.4f}）の 2 乗")
E.check("s23p11a", lr.pvalue > 0.05, f"p 値 {lr.pvalue:.3f} > 0.05")
# (b) 完全分離：β1 を大きくすると対数尤度が 0 に近づき、上限に達しない
x11 = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.55, 0.70]); y11 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
ll11 = lambda b1: -np.sum(y11 * np.logaddexp(0, -b1 * (x11 - 0.35)) + (1 - y11) * np.logaddexp(0, b1 * (x11 - 0.35)))
vals = [ll11(b) for b in (10, 50, 100, 200)]
E.check("s23p11b", all(np.diff(vals) > 0) and vals[-1] < 0 and vals[-1] > -1e-3, f"対数尤度 {np.round(vals, 6)} が 0 に向かって増え続ける")
E.check("s23p11b", x11[y11 == 0].max() < x11[y11 == 1].min(), "0 と 1 が完全に分かれている")
E.choice("s23p11b", "有限の最尤推定値が存在しない", "完全分離")

# ---- 大問12
# (a) 母集団を作って単純無作為抽出の総額推定の標準偏差をシミュレーション
pop = rng.gamma(9, 120 / 9, 400); pop = 120 + (pop - pop.mean()) / pop.std(ddof=1) * 40
tots = np.array([400 * rng.choice(pop, 25, replace=False).mean() for _ in range(40000)])
E.check("s23p12a", abs(tots.std() / 3098.4 - 1) < 0.02, f"シミュレーションの標準偏差 {tots.std():.1f}")
E.num("s23p12a", rnd(400 * math.sqrt((1 - 25 / 400) * 40 ** 2 / 25), 0), "式の値")
# (b) 事後分布を格子で数値計算
mu_g = np.linspace(10, 50, 40001)
post = stats.norm.pdf(mu_g, 30, 4) * stats.norm.pdf(34, mu_g, 6 / 3)
post /= post.sum()
pp12 = post[mu_g > 32].sum()
E.check("s23p12b", abs(pp12 - stats.norm.sf(32, 33.2, math.sqrt(3.2))) < 1e-4, f"格子で求めた事後確率 {pp12:.5f}")
nearest("s23p12b", pp12)
E.check("s23p12b", all(abs(stats.norm.cdf(z_) - v_) < 5e-5 for z_, v_ in ((0.5, 0.6915), (0.6, 0.7257), (0.67, 0.7486), (1.0, 0.8413))), "示した Φ の値")

# ---- 大問13
# (a) 問題文の表を読んで一個抜き k 近傍法
q13 = E.item("s23p13a")["question"]
xs13 = np.array([float(v_) for v_ in re.search(r"\| \$x\$ \|([^\n]*)\|", q13).group(1).split("|")])
lb13 = np.array([v_.strip() for v_ in re.search(r"\| ラベル \|([^\n]*)\|", q13).group(1).split("|")])
def loo13(k):
    err = 0
    for i in range(8):
        m_ = np.arange(8) != i
        d_ = np.abs(xs13[m_] - xs13[i]); o_ = np.argsort(d_, kind="stable")
        assert d_[o_[k - 1]] < d_[o_[k]] - 1e-9
        v_, c_ = np.unique(lb13[m_][o_[:k]], return_counts=True)
        err += v_[np.argmax(c_)] != lb13[i]
    return err / 8
E.choice("s23p13a", f"$k=1$：${f(loo13(1), 3)}$、$k=3$：${f(loo13(3), 3)}$", "一個抜きの総当たり")
# (b) クーポン収集をシミュレーション
def coupon():
    seen, k = set(), 0
    while len(seen) < 4:
        seen.add(rng.integers(4)); k += 1
    return k
cc = np.mean([coupon() for _ in range(100_000)])
E.check("s23p13b", abs(cc - 25 / 3) < 0.05, f"シミュレーション {cc:.3f}")
E.choice("s23p13b", f"${f(sum(4 / k_ for k_ in range(1, 5)), 2)}$", "4×調和数")

# ---- 大問14
# (a) 親の数から母数を数える
parents = {"E": [], "L": ["E"], "D": ["E", "L"], "C": ["D"]}
npar = sum(2 ** len(p_) for p_ in parents.values())
E.choice("s23p14a", f"グラフにもとづくと {npar}、制約なしでは {2 ** 4 - 1}", "2^(親の数) の和")
# (b) 座標から重心法を直接計算（scipy の centroid は平方距離ではないので、平方で比べる）
from scipy.cluster.hierarchy import linkage
P14 = np.array([[0, 0], [2, 0], [1, 1.8]])
Lk = linkage(P14, method="centroid")
E.check("s23p14b", np.allclose(Lk[:, 2] ** 2, [4.0, 3.24]), f"scipy の重心法の併合の高さの 2 乗 {np.round(Lk[:, 2] ** 2, 4)}（逆転）")
E.choice("s23p14b", f"2 回目は高さ {f(Lk[1, 2] ** 2, 2)} で併合する。2 回目の方が低い", "scipy linkage")

E.done()
