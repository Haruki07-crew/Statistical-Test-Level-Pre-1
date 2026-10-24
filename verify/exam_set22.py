"""模擬試験 第22回（set22）の検算。generator/set22.py とは別の方法（scipy の分布関数・全列挙・
シミュレーション・sympy・数値積分）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import permutations, combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(22)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [[float(v) for v in re.findall(r"-?\d+\.?\d*", str(c).replace("−", "-"))] for c in E.item(iid)["choices"]]


def near(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1
# (a) 20 箱の全順列を数え上げる代わりに、破損箱の位置の組（C(20,3) 通り）を全列挙
cnt = sum(1 for c in combinations(range(20), 3) if min(c) == 2)
p = Fraction(cnt, math.comb(20, 3))
E.num("s22p01a", rnd(float(p), 3), "破損箱の位置の全列挙で最初の破損が3番目")
sim = rng.permuted(np.tile(np.r_[np.ones(3), np.zeros(17)], (200000, 1)), axis=1)
first = sim.argmax(axis=1)
E.check("s22p01a", abs((first == 2).mean() - float(p)) < 0.003, f"シミュレーション {(first == 2).mean():.4f}")
# (b) scipy の二項分布
a = stats.binom.sf(1, 20, 0.05)
b = stats.binom.cdf(1, 20, 0.20)
E.choice("s22p01b", f"\\alpha={f(a, 3)}$、$\\beta={f(b, 3)}", "scipy binom の α・β")
E.check("s22p01b", near(0.95 ** 19, 0.3774, 5e-5) and near(0.8 ** 19, 0.01441, 5e-6), "示した 0.95^19, 0.8^19")
# (c) 半幅条件を満たす最小の n を総当たり
z = stats.norm.ppf(0.975)
n = next(k for k in range(1, 5000) if 1.96 * math.sqrt(0.15 * 0.85 / k) <= 0.03)
E.num("s22p01c", n, "半幅 ≤ 0.03 となる最小の n を総当たり")
E.check("s22p01c", abs(z - 1.96) < 0.001, f"z_0.025={z:.4f}")

# ---- 大問2
pa = stats.poisson.sf(5, 3.0)
E.num("s22p02a", rnd(pa, 3), "scipy poisson.sf(5, 3)")
E.check("s22p02a", abs(math.exp(-3) - 0.0498) < 5e-5, "e^-3 = 0.0498")
E.check("s22p02a", abs((rng.poisson(1.5, (300000, 2)).sum(1) > 5).mean() - pa) < 0.003, "2日分の需要のシミュレーション")
# (b) 期待損失を需要分布の仮定のもとで数値的に最小化して分位点が τ=0.8 になることを確かめる（正規分布で例示）
Y = rng.normal(150, 20, 400000)
qs = np.arange(100, 200, 0.5)
loss = [np.mean(400 * np.maximum(Y - q, 0) + 100 * np.maximum(q - Y, 0)) for q in qs]
qopt = qs[int(np.argmin(loss))]
E.check("s22p02b", abs(stats.norm.cdf(qopt, 150, 20) - 0.8) < 0.01, f"期待損失の最小点の分布関数値 {stats.norm.cdf(qopt, 150, 20):.3f} ≈ 0.8")
E.choice("s22p02b", f"${121 + 44 * 1}$", "τ=0.8 の推定式で x=1")

# ---- 大問3
# (a) 数値計算でハザード
S = lambda t: 0.6 * math.exp(-t) + 0.4 * math.exp(-t / 3)
dens = lambda t: -(S(t + 1e-6) - S(t - 1e-6)) / 2e-6
h = dens(3) / S(3)
E.choice("s22p03a", f"${f(h, 3)}$", "数値微分による f/S")
E.check("s22p03a", abs(math.exp(-1) - 0.3679) < 5e-5, "e^-1 = 0.3679")
# (b) scipy の順序統計量（ベータ分布）と数値積分
vmed = 120 ** 2 * stats.beta(3, 3).var()
vmean = 120 ** 2 / 12 / 5
E.choice("s22p03b", f"${f(vmed / vmean, 3)}$", "beta(3,3) の分散から比")
U = rng.uniform(0, 120, (200000, 5))
r_sim = np.median(U, axis=1).var() / U.mean(axis=1).var()
E.check("s22p03b", abs(r_sim - vmed / vmean) < 0.05, f"シミュレーションの比 {r_sim:.3f}")

# ---- 大問4
a6 = 6.0
ER = integrate.quad(lambda r: r * 2 * r / a6 ** 2, 0, a6)[0]
E.choice("s22p04a", f"${f(ER, 2)}$", "数値積分 E[R]")
th = rng.uniform(0, 2 * np.pi, 400000)
rad = a6 * np.sqrt(rng.uniform(0, 1, 400000))
pts = rng.uniform(-a6, a6, (800000, 2))
inside = pts[(pts ** 2).sum(1) <= a6 ** 2]
E.check("s22p04a", abs(np.sqrt((inside ** 2).sum(1)).mean() - 4.0) < 0.02, "円内一様点の棄却法シミュレーション")
rs = np.array([0.9, 2.4, 3.9, 5.4])
fpdf = lambda r: 2 * r / a6 ** 2
gpdf = lambda r: 1 / a6
E.choice("s22p04b", f"${f(np.mean(rs * fpdf(rs) / gpdf(rs)), 2)}$", "重要度サンプリング推定量を直接計算")
uu = rng.uniform(0, a6, 400000)
E.check("s22p04b", abs(np.mean(uu * fpdf(uu) / gpdf(uu)) - 4.0) < 0.02, "多数の提案値で不偏性を確認")

# ---- 大問5
pT = stats.lognorm(s=0.5, scale=math.exp(1.6)).sf(15)
E.choice("s22p05a", f"${f(1 - stats.norm.cdf(2.22), 3)}$", "Φ(2.22) から")
E.check("s22p05a", abs(pT - 0.013) < 0.0012, f"scipy lognorm.sf(15)={pT:.4f}（丸めなし）")
E.check("s22p05a", abs(math.log(15) - 2.708) < 5e-4 and abs(math.log10(15) - 1.176) < 5e-4
        and abs(stats.norm.cdf(2.22) - 0.9868) < 5e-5 and abs(stats.norm.cdf(0.85) - 0.8023) < 5e-5, "示した対数と Φ")
# (b) 遷移率行列の定常分布を数値的に解く
lam, mu = 20.0, 10.0
Q = np.zeros((4, 4))
for i in range(4):
    if i < 3:
        Q[i, i + 1] = lam
    if i > 0:
        Q[i, i - 1] = i * mu
    Q[i, i] = -Q[i].sum()
A = np.vstack([Q.T, np.ones(4)])
pi = np.linalg.lstsq(A, np.r_[np.zeros(4), 1], rcond=None)[0]
E.num("s22p05b", rnd(lam * (1 - pi[3]), 2), "生成行列から定常分布を解いて λ(1−π3)")
# 離散事象シミュレーション
t, busy, acc, arr = 0.0, [], 0, 0
T_END = 20000.0
nxt = rng.exponential(1 / lam)
while nxt < T_END:
    busy = [e for e in busy if e > nxt]
    arr += 1
    if len(busy) < 3:
        busy.append(nxt + rng.exponential(1 / mu))
        acc += 1
    nxt += rng.exponential(1 / lam)
E.check("s22p05b", abs(acc / T_END - lam * (1 - pi[3])) < 0.15, f"シミュレーションの受付率 {acc / T_END:.3f}/時")
# (c) シミュレーションで V[√N]
Nn = rng.poisson(20, 1000000)
E.choice("s22p05c", "$0.250$", "デルタ法の値")
E.check("s22p05c", abs(np.sqrt(Nn).var() - 0.25) < 0.02, f"シミュレーションの V[√N]={np.sqrt(Nn).var():.4f}")

# ---- 大問6
A6 = np.array([18, 22, 26, 30, 41])
B6 = np.array([24, 33, 36, 44, 48])
ks = stats.ks_2samp(A6, B6).statistic
E.choice("s22p06a", f"${f(ks, 1)}$", "scipy ks_2samp の統計量")
v = [4.0, 9.0, 16.0]
# 分散がちょうどその値になる正規データを作って scipy.bartlett に入れる
groups = []
for vv in v:
    x = rng.normal(0, 1, 6)
    x = (x - x.mean()) / x.std(ddof=1) * math.sqrt(vv)
    groups.append(x)
bt = stats.bartlett(*groups).statistic
E.choice("s22p06b", f"統計量は${f(bt, 2)}$で、$5.99$", "scipy.stats.bartlett")
E.check("s22p06b", abs(stats.chi2.ppf(0.95, 2) - 5.99) < 0.005 and abs(stats.chi2.ppf(0.95, 3) - 7.81) < 0.005, "χ² の点")
E.check("s22p06b", all(abs(math.log(x) - y) < 5e-5 for x, y in [(4, 1.3863), (9, 2.1972), (16, 2.7726), (29 / 3, 2.2687)]), "示した対数")

# ---- 大問7
xx = sp.symbols("x")
sol = sp.solve(sp.Eq(2.0 + 0.5 * xx, 2.0 + 0.5 * xx + 1.5 - 0.30 * xx), xx)
E.choice("s22p07a", f"${f(float(sol[0]), 1)}$", "sympy で 2 直線の交点")
res = np.array([-1.1, -0.9, -0.8, -0.6, -0.4, -0.3, -0.1, 0.4, 1.3, 2.5])
tq = stats.norm.ppf((np.arange(1, 11) - 0.5) / 10)
E.check("s22p07b", abs(tq[-1] - 1.645) < 0.001 and abs(tq[0] + 1.645) < 0.001, "理論分位点 ±1.645")
E.check("s22p07b", res[0] > tq[0] and res[-1] > tq[-1] and stats.skew(res) > 0, f"両端とも直線より上・標本歪度 {stats.skew(res):.2f}>0")
E.check("s22p07b", E.item("s22p07b")["answer"] == 0 and "右に裾の長い" in E.item("s22p07b")["choices"][0], "正解は右に歪んだ分布の記述")
E.check("s22p07b", abs(stats.norm.ppf(0.9) - 1.282) < 0.0005, "0.90 分位点 1.282")

# ---- 大問8
y = np.array([12, 8, 20, 10])
xg = np.array([0, 0, 1, 1])
off = np.log(np.array([5000, 5000, 8000, 7000]))
# ニュートン法でポアソン回帰を当てはめる
beta = np.zeros(2)
X = np.c_[np.ones(4), xg]
for _ in range(50):
    mu8 = np.exp(X @ beta + off)
    beta = beta + np.linalg.solve(X.T @ (mu8[:, None] * X), X.T @ (y - mu8))
mu8 = np.exp(X @ beta + off)
E.check("s22p08a", np.allclose(mu8, [10, 10, 16, 14], atol=1e-6), f"当てはめ値 {np.round(mu8, 4)}")
Dv = 2 * np.sum(y * np.log(y / mu8) - (y - mu8))
E.choice("s22p08a", f"$D={f(Dv, 2)}$で、自由度2の5%点5.99を下回り", "当てはめ直した逸脱度")
tab = np.array([[10, 190], [30, 170]])
xs = np.r_[np.ones(200), np.zeros(200)]
ys = np.r_[np.ones(10), np.zeros(190), np.ones(30), np.zeros(170)]
phi = np.corrcoef(xs, ys)[0, 1]
E.num("s22p08b", rnd(phi, 3), "0/1 データの相関係数")
chi = stats.chi2_contingency(tab, correction=False)[0]
E.check("s22p08b", abs(chi - 400 * phi ** 2) < 1e-9, f"χ²={chi:.3f} = nφ²")

# ---- 大問9
cnt = {}
for w in (0, 1):
    for k in (0, 1):
        cnt[(w, k)] = (0.2 if w else 0.8) * (0.1 if k else 0.9)
pd = sum(v for (w, k), v in cnt.items() if w or k)
p1 = sum(v for (w, k), v in cnt.items() if w) / pd
p2 = cnt[(1, 1)] / sum(v for (w, k), v in cnt.items() if k)
E.choice("s22p09a", f"${f(p1, 3)}$と${f(p2, 3)}$", "同時確率の全列挙")

# ---- 大問10
yv = np.arange(10, 33, 2)
means = [yv[i::3].mean() for i in range(3)]
E.num("s22p10a", rnd(np.mean((np.array(means) - yv.mean()) ** 2), 2), "3通りの系統標本の全列挙")
srs = [np.mean(c) for c in combinations(yv, 4)]
E.check("s22p10a", abs(np.var(srs) - 8.6667) < 1e-3, f"単純無作為抽出の全組合せの分散 {np.var(srs):.4f}")

# ---- 大問11
R = np.full((3, 3), 0.6) + 0.4 * np.eye(3)
w, V = np.linalg.eigh(R)
v1 = V[:, -1]
zv = np.array([1.2, 0.3, -0.6])
E.choice("s22p11a", f"${f(np.sum((zv - (v1 @ zv) * v1) ** 2), 2)}$", "numpy の固有ベクトルで射影")
E.check("s22p11a", abs(w[-1] - 2.2) < 1e-9, "最大固有値 2.2")
Lp = np.array([0.7, 0.2])
Phi = np.array([[1, 0.5], [0.5, 1]])
h2 = Lp @ Phi @ Lp
stc = Lp @ Phi
E.choice("s22p11b", f"共通性${f(h2, 2)}$、因子構造$({f(stc[0], 2)},\\ {f(stc[1], 2)})$", "行列計算 ΛΦΛ'・ΛΦ")
# シミュレーション：因子と誤差を発生させ、相関を確かめる
F = rng.multivariate_normal([0, 0], Phi, 300000)
xsim = F @ Lp + rng.normal(0, math.sqrt(1 - h2), 300000)
E.check("s22p11b", abs(np.corrcoef(xsim, F[:, 0])[0, 1] - 0.8) < 0.01, "シミュレーションで Cor(x,f1)≈0.80")
from scipy.optimize import isotonic_regression  # noqa: E402
d = np.array([1.0, 1.6, 1.4, 2.2, 2.8, 3.0])
dh = isotonic_regression(d).x
E.num("s22p11c", rnd(math.sqrt(np.sum((d - dh) ** 2) / np.sum(d ** 2)), 3), "scipy の単調回帰（isotonic_regression）でストレス")

# ---- 大問12
om = sp.symbols("omega")
fsp = 1 / (1 - 2 * sp.Rational(3, 5) * sp.cos(om) + sp.Rational(9, 25))
ratio = fsp.subs(om, 0) / fsp.subs(om, sp.pi)
E.choice("s22p12a", f"比は${int(ratio)}$で、長い周期", "sympy でスペクトル比")
# AR(1) を発生させてピリオドグラムで低周波が強いことを確認
e = rng.normal(0, 1, 200000)
xa = np.empty_like(e)
xa[0] = e[0]
for i in range(1, len(e)):
    xa[i] = 0.6 * xa[i - 1] + e[i]
E.check("s22p12a", abs(np.corrcoef(xa[1:], xa[:-1])[0, 1] - 0.6) < 0.01, "シミュレーションの自己相関 0.6")

# ---- 大問13：分割区データを作って statsmodels なしで平方和を分解し、主区誤差で F を計算する仕組みを確認
E.check("s22p13a", abs(stats.f.ppf(0.95, 2, 6) - 5.14) < 0.005 and abs(stats.f.ppf(0.95, 2, 9) - 4.26) < 0.005
        and abs(stats.f.ppf(0.95, 2, 15) - 3.68) < 0.005, "F の点")
dfE1 = (4 - 1) * (3 - 1)
dfE2 = 24 - 1 - (3 + 2 + dfE1 + 1 + 2)
E.check("s22p13a", dfE1 == 6 and dfE2 == 9, "自由度 6・9")
E.choice("s22p13a", f"$F={f((36 / 2) / (24 / dfE1), 2)}$で$F_{{0.05}}(2,6)", "主区誤差で F")
# 期待平均平方の確認：主区誤差のある模型で、A の効果 0 のとき E[MS_A] ≈ E[MS_E1] になることをシミュレーション
msA, msE1, msE2 = [], [], []
for _ in range(3000):
    blk = rng.normal(0, 1, 4)[:, None, None]
    wp = rng.normal(0, 1.5, (4, 3))[:, :, None]
    eps = rng.normal(0, 1, (4, 3, 2))
    yy = blk + wp + eps
    g = yy.mean()
    mb, ma = yy.mean((1, 2)), yy.mean((0, 2))
    mab = yy.mean(2)
    ssA = 8 * np.sum((ma - g) ** 2)
    ssE1 = 2 * np.sum((mab - mb[:, None] - ma[None, :] + g) ** 2)
    msA.append(ssA / 2)
    msE1.append(ssE1 / 6)
E.check("s22p13a", abs(np.mean(msA) / np.mean(msE1) - 1) < 0.05, f"A の効果 0 で E[MS_A]/E[MS_E1]={np.mean(msA) / np.mean(msE1):.3f}≈1")

E.done()
