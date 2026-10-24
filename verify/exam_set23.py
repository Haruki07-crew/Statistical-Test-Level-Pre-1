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


# ---- 大問1：道路網の全 32 状態を列挙し、S から T へのつながりを探索で調べる
p = [0.9, 0.8, 0.7, 0.8, 0.9]
edges = [("S", "A"), ("S", "B"), ("A", "B"), ("A", "T"), ("B", "T")]
tot = 0.0
for st in product([0, 1], repeat=5):
    pr = math.prod(pi if s else 1 - pi for pi, s in zip(p, st))
    seen, stack = {"S"}, ["S"]
    while stack:
        u = stack.pop()
        for (a, b), s in zip(edges, st):
            if s and u in (a, b):
                v = b if u == a else a
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    tot += pr * ("T" in seen)
E.num("s23p01a", rnd(tot, 3), "32 状態の列挙と到達可能性の探索")
sim = rng.random((300000, 5)) < np.array(p)
c = sim[:, 2]
ok = np.where(c, (sim[:, 0] | sim[:, 1]) & (sim[:, 3] | sim[:, 4]), (sim[:, 0] & sim[:, 3]) | (sim[:, 1] & sim[:, 4]))
E.check("s23p01a", abs(ok.mean() - tot) < 0.002, f"シミュレーション {ok.mean():.4f}")

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
# (b) scipy の lognorm（平均 2000、変動係数 0.2 を満たす母数）
s2 = math.log(1 + 0.2 ** 2)
dist = stats.lognorm(s=math.sqrt(s2), scale=2000 * math.exp(-s2 / 2))
E.check("s23p02b", near(dist.mean(), 2000, 1e-6) and near(dist.std() / dist.mean(), 0.2, 1e-9), "母数が平均と変動係数を再現")
nearest("s23p02b", dist.cdf(1400))
E.check("s23p02b", all(near(stats.norm.cdf(z), v, 5e-5) for z, v in [(1.5, 0.9332), (1.7, 0.9554), (1.8, 0.9641), (1.9, 0.9713)])
        and near(math.log(1.04), 0.0392, 5e-5) and near(math.log(0.7), -0.3567, 5e-5) and near(math.sqrt(0.0392), 0.1980, 5e-5),
        "示した Φ・ln・√ の値")

# ---- 大問3：sympy でフィッシャー情報量、シミュレーションで最尤推定量の標準偏差
th, h = sp.symbols("theta h", positive=True)
logf = sp.log(h / th) - h ** 2 / (2 * th)
dens = h / th * sp.exp(-h ** 2 / (2 * th))
I1 = sp.simplify(-sp.integrate(sp.diff(logf, th, 2) * dens, (h, 0, sp.oo)))
E.check("s23p03a", sp.simplify(I1 - 1 / th ** 2) == 0, f"1 個あたりの情報量 {I1}")
n, sh2 = 25, 60.5
mle = optimize.minimize_scalar(lambda t: -(-n * math.log(t) - sh2 / (2 * t)), bounds=(0.1, 10), method="bounded").x
se = math.sqrt(1 / float((n * I1).subs(th, mle)))
E.num("s23p03a", rnd(se, 2), "数値最適化の最尤推定値と sympy の情報量")
H = stats.rayleigh(scale=math.sqrt(1.21)).rvs((40000, n), random_state=rng)
est = (H ** 2).sum(1) / (2 * n)
E.check("s23p03a", abs(est.std() - 1.21 / 5) < 0.005, f"シミュレーションの θ̂ の標準偏差 {est.std():.4f}")

# ---- 大問4
# (a) 数値積分とシミュレーション
pD, _ = integrate.quad(lambda x: 1 - min(1.0, 0.2 / x), 0.2, 1)
nearest("s23p04a", pD)
XY = rng.random(400000) * rng.random(400000)
E.check("s23p04a", abs((XY > 0.2).mean() - pD) < 0.003 and near(math.log(0.2), -1.6094, 5e-5), f"シミュレーション {(XY > 0.2).mean():.4f}・ln0.2")
# (b) 分布関数 (s/s0)^m（べき分布）での最小値の中央値を n と 10n で正確に計算し比をとる
m = 10
med = lambda nn: (-math.expm1(math.log(0.5) / nn)) ** (1 / m)
ratio = med(100000) / med(10000)
nearest("s23p04b", ratio)
E.check("s23p04b", near(10 ** -0.1, 0.7943, 5e-5) and near(10 ** -0.05, 0.8913, 5e-5) and near(10 ** -0.5, 0.3162, 5e-5), "示した 10 のべき")

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
x = np.array([33.1, 31.4, 34.6, 29.2, 32.8, 35.3])
E.check("s23p06a", near(x.mean(), 32.733, 5e-4) and near(x.std(ddof=1), 2.212, 5e-4), "示した平均と s")
t = stats.t.ppf(0.975, 5)
E.check("s23p06a", near(t, 2.571, 5e-4) and near(stats.t.ppf(0.95, 5), 2.015, 5e-4), "t 点")
E.num("s23p06a", rnd(x.mean() - t * x.std(ddof=1) * math.sqrt(1 + 1 / 6), 2), "丸めない値と scipy の t 点")
# 予測区間の被覆率のシミュレーション
Z = rng.normal(0, 1, (100000, 7))
lo = Z[:, :6].mean(1) - t * Z[:, :6].std(1, ddof=1) * math.sqrt(7 / 6)
hi = Z[:, :6].mean(1) + t * Z[:, :6].std(1, ddof=1) * math.sqrt(7 / 6)
cov = ((Z[:, 6] > lo) & (Z[:, 6] < hi)).mean()
E.check("s23p06a", abs(cov - 0.95) < 0.004, f"予測区間の被覆率 {cov:.4f}")
res = stats.wilcoxon(x - 30, alternative="greater", method="exact")
E.choice("s23p06b", f"${f(res.pvalue, 3)}$", "scipy wilcoxon（exact）")

# ---- 大問7
# (a) ±1 の計画行列で、日の指示変数がどの効果の対比と一致・直交するかを調べる
runs = ["(1)", "a", "b", "c", "ab", "ac", "bc", "abc"]
lvl = {r: {k: (1 if k in r else -1) for k in "abc"} for r in runs}
effects = {"A": "a", "B": "b", "C": "c", "AB": "ab", "AC": "ac", "BC": "bc", "ABC": "abc"}
col = lambda e: np.array([math.prod(lvl[r][k] for k in effects[e]) for r in runs])


def confounded(block):
    d = np.array([1 if r in block else -1 for r in runs])
    return [e for e in effects if abs(np.dot(d, col(e))) > 0]


cands = {0: ["(1)", "a", "b", "ab"], 1: ["(1)", "ab", "c", "abc"], 2: ["(1)", "a", "bc", "abc"], 3: ["(1)", "ab", "ac", "bc"], 4: ["(1)", "a", "b", "c"]}
good = [k for k, b in cands.items() if confounded(b) == ["ABC"]]
E.check("s23p07a", good == [E.item("s23p07a")["answer"]], f"日の効果が ABC だけと重なる候補 {good}（他: {[confounded(b) for b in cands.values()]}）")
# (b) scipy の二項分布
pv = stats.binom.sf(2, 20, 0.05)
E.num("s23p07b", rnd(pv, 3), "scipy binom.sf")
E.check("s23p07b", near(0.95 ** 20, 0.3585, 5e-5) and near(0.95 ** 19, 0.3774, 5e-5) and near(0.95 ** 18, 0.3972, 5e-5), "示した 0.95 のべき")
E.check("s23p07b", near(stats.binomtest(3, 20, 0.05, alternative="greater").pvalue, pv, 1e-12), "binomtest と一致")

# ---- 大問8：scipy の χ² とシミュレーション
c = stats.chi2.ppf(0.05, 10) * 40 / 2
power = stats.chi2.cdf(2 * c / 20, 10)
nearest("s23p08a", power)
T0 = rng.exponential(40, (200000, 5)).sum(1)
T1 = rng.exponential(20, (200000, 5)).sum(1)
E.check("s23p08a", abs((T0 <= c).mean() - 0.05) < 0.002 and abs((T1 <= c).mean() - power) < 0.004, f"シミュレーション α={(T0 <= c).mean():.4f}, 検出力={(T1 <= c).mean():.4f}")
E.check("s23p08a", near(stats.chi2.ppf(0.05, 10), 3.940, 5e-4) and near(stats.chi2.ppf(0.05, 5), 1.145, 5e-4)
        and near(stats.chi2.cdf(7.88, 10), 0.3594, 5e-5) and near(stats.chi2.cdf(1.97, 10), 0.0034, 5e-5)
        and near(stats.chi2.cdf(2.29, 5), 0.1923, 5e-5), "示した χ² の値")

# ---- 大問9：状態 3 を吸収状態にした行列の 3 乗で初到達確率、シミュレーションでも確認
P = np.array([[0.7, 0.25, 0.05], [0, 0.6, 0.4], [0.6, 0, 0.4]])
Pa = P.copy()
Pa[2] = [0, 0, 1]
first3 = np.linalg.matrix_power(Pa, 3)[0, 2] - np.linalg.matrix_power(Pa, 2)[0, 2]
E.choice("s23p09a", f"${f(first3, 4)}$", "吸収連鎖の 3 段階と 2 段階の差")
cum = rng.random((200000, 3))
st = np.zeros(200000, dtype=int)
hit = np.full(200000, -1)
for k in range(3):
    nxt = np.where(cum[:, k] < P[st, 0], 0, np.where(cum[:, k] < P[st, 0] + P[st, 1], 1, 2))
    hit = np.where((hit < 0) & (nxt == 2), k + 1, hit)
    st = nxt
E.check("s23p09a", abs((hit == 3).mean() - first3) < 0.003, f"シミュレーション {(hit == 3).mean():.4f}")

# ---- 大問10
# (a) 回帰のデータを作り、行列計算で平均応答の標準誤差を求める
xs = np.array([10, 15, 20, 22, 25, 28, 32, 35, 38, 40, 45, 50], float)
xs = 30 + (xs - xs.mean()) * math.sqrt(2000 / ((xs - xs.mean()) ** 2).sum())
X = np.c_[np.ones(12), xs]
x0 = np.array([1, 40.0])
var0 = x0 @ np.linalg.inv(X.T @ X) @ x0
hw = stats.t.ppf(0.975, 10) * 1.2 * math.sqrt(var0)
E.choice("s23p10a", f"${f(hw, 3)}$", "行列 x0'(X'X)^{-1}x0 と scipy の t 点")
E.check("s23p10a", near(stats.t.ppf(0.975, 10), 2.228, 5e-4) and near(stats.t.ppf(0.975, 11), 2.201, 5e-4) and near(stats.t.ppf(0.95, 10), 1.812, 5e-4), "t 点")
# (b) 定義どおり（標準化残差の 2 乗 × h/(1-h) ÷ p）に計算して最大を選ぶ
cook = {}
for k, (e, hh) in {"A": (3.0, 0.08), "B": (1.2, 0.45), "C": (2.0, 0.20)}.items():
    rr = e / (1.2 * math.sqrt(1 - hh))
    cook[k] = rr ** 2 / 2 * hh / (1 - hh)
kmax = max(cook, key=cook.get)
E.choice("s23p10b", f"橋 {kmax}、$D={f(cook[kmax], 2)}$", "定義からのクックの距離の最大")
# (c) 重み行列とバイアスベクトルの形から数える
shapes = [(5, 8), (8,), (8, 1), (1,)]
E.choice("s23p10c", f"${sum(int(np.prod(s_)) for s_ in shapes)}$", "パラメータ配列の要素数の合計")

# ---- 大問11
# (a) scipy.stats.logrank（打ち切りは CensoredData）
A = stats.CensoredData(uncensored=[5, 7, 11], right=[9])
B = stats.CensoredData(uncensored=[8, 12], right=[10, 14])
lr = stats.logrank(A, B)
E.num("s23p11a", rnd(lr.statistic ** 2, 2), f"scipy logrank（z={lr.statistic:.4f}）の 2 乗")
E.check("s23p11a", lr.pvalue > 0.05, f"p 値 {lr.pvalue:.3f} > 0.05")
# (b) scipy のガンマ分布（形状 1/φ、尺度 μφ）
mu = math.exp(0.50 + 0.405 + 0.10 * 4)
g = stats.gamma(a=1 / 0.16, scale=mu * 0.16)
E.choice("s23p11b", f"平均 ${f(g.mean(), 2)}$、標準偏差 ${f(g.std(), 2)}$", "scipy gamma の平均と標準偏差")
E.check("s23p11b", near(math.exp(1.305), 3.687, 5e-4) and near(math.exp(0.9), 2.460, 5e-4), "示した e の値")

# ---- 大問12
# (a) 費用一定のもとで分散を最小にする配分を数値最適化し、合計 120 に縮める
Nh, Sh, ch = np.array([600, 300, 100.]), np.array([2, 3, 8.]), np.array([4, 9, 16.])
W = Nh / Nh.sum()
var = lambda nn: np.sum(W ** 2 * Sh ** 2 / nn)
cons = {"type": "eq", "fun": lambda nn: np.dot(ch, nn) - 1000}
sol = optimize.minimize(var, x0=np.array([50, 30, 10.]), constraints=[cons], bounds=[(1, None)] * 3, method="SLSQP")
alloc = sol.x / sol.x.sum() * 120
E.num("s23p12a", rnd(alloc[2], 0), f"数値最適化の配分 {np.round(alloc, 3)}")
# (b) ジェフリーズ事前分布 √I(p) を数値積分で正規化して事後平均を計算
nn_, yy = 22, 4
prior = lambda q: math.sqrt(1 / (q * (1 - q)))
lik = lambda q: q ** yy * (1 - q) ** (nn_ - yy)
num = integrate.quad(lambda q: q * lik(q) * prior(q), 0, 1)[0]
den = integrate.quad(lambda q: lik(q) * prior(q), 0, 1)[0]
E.choice("s23p12b", f"${f(num / den, 3)}$", "数値積分による事後平均")
E.check("s23p12b", rnd(alloc[2], 0) == nn_, "(b) の点検数は (a) の答えと同じ")

# ---- 大問13
# (a) 2 つの正規密度が等しくなる点を数値的に求める
d = lambda v: stats.norm.pdf(v, 3, 0.5) - stats.norm.pdf(v, 2, 1)
r1 = optimize.brentq(d, 2.0, 3.0)
r2 = optimize.brentq(d, 3.5, 6.0)
E.choice("s23p13a", f"${f(r1, 2)}<X<{f(r2, 2)}$", "brentq による境界")
E.check("s23p13a", d(3.3) > 0 and d(1.0) < 0 and d(5.0) < 0 and near(math.log(2), 0.6931, 5e-5), "境界の間で健全の密度が大きい・ln2")
# (b) scipy の poisson で条件付き期待値
k = np.arange(1, 60)
zt = np.sum(k * stats.poisson.pmf(k, 0.8)) / stats.poisson.sf(0, 0.8)
E.choice("s23p13b", f"${f(zt, 3)}$", "scipy poisson の和")
E.check("s23p13b", near(math.exp(-0.8), 0.4493, 5e-5), "e^-0.8")

# ---- 大問14
# (a) 1 因子モデルを最小二乗で当てはめ（数値最適化）
R = {(0, 1): 0.72, (0, 2): 0.63, (1, 2): 0.56}
obj = lambda lam: sum((lam[i] * lam[j] - r) ** 2 for (i, j), r in R.items())
lam = optimize.minimize(obj, x0=[0.5, 0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14}).x
E.choice("s23p14a", f"\\lambda_3={f(lam[2], 2)}$、$\\psi_3={f(1 - lam[2] ** 2, 2)}", "数値最適化による負荷量")
E.check("s23p14a", obj(lam) < 1e-10, "相関をちょうど再現")
# (b) 点の座標がなくても定義どおりに：各クラスターの平均距離を辞書から計算
dist_ = {"A": [1.2, 1.8], "B": [2.6, 3.0, 3.4], "C": [2.2, 3.6]}
a_ = np.mean(dist_["A"])
b_ = min(np.mean(v) for kk, v in dist_.items() if kk != "A")
E.num("s23p14b", rnd((b_ - a_) / max(a_, b_), 2), "シルエット係数の定義")

E.done()
