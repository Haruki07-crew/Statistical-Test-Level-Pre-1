"""模擬試験 第20回（set20）の検算。generator/set20.py とは別の方法（scipy の分布関数・数値積分・sympy・
シミュレーション・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(20)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


def table(iid, value, shown, d, what):
    """問題文に示した数表の値 shown が、正確な値 value を小数第 d 位に丸めたものと一致するか。"""
    E.check(iid, f(value, d) == f(shown, d), f"数表 {what}: 示した値 {shown} ／ 正確な値 {value:.6f}")


def nums_in(iid):
    it = E.item(iid)
    return [float(re.search(r"-?\d+\.?\d*", str(c).replace("−", "-")).group()) for c in it["choices"]]


def distinct(iid, d):
    """数値の選択肢が丸めて互いに区別できるか。"""
    v = [f(x, d) for x in nums_in(iid)]
    E.check(iid, len(set(v)) == 5, f"選択肢の数値がすべて異なる {v}")


x, y = sp.symbols("x y", positive=True)

# ---- 大問1 同時密度 f=(2/3)(2x+y)
fxy = sp.Rational(2, 3) * (2 * x + y)
E.check("s20p01a", sp.integrate(fxy, (x, 0, 1), (y, 0, 1)) == 1, "同時密度の積分が 1")
fx = sp.integrate(fxy, (y, 0, 1))
pc = sp.integrate(fxy.subs(x, sp.Rational(1, 5)), (y, sp.Rational(1, 2), 1)) / fx.subs(x, sp.Rational(1, 5))
E.choice("s20p01a", f"${f(float(pc), 3)}$", "sympy で条件付き確率")
# シミュレーション（受容棄却で同時分布から発生し、X≈0.2 の近傍で条件付け）
N = 2_000_000
xs, ys, us = rng.random(N), rng.random(N), rng.random(N)
keep = us < (2 * xs + ys) / 3
xs, ys = xs[keep], ys[keep]
near = np.abs(xs - 0.2) < 0.01
E.check("s20p01a", abs((ys[near] > 0.5).mean() - float(pc)) < 0.01, f"近傍条件付けのシミュレーション {(ys[near] > 0.5).mean():.4f}")
distinct("s20p01a", 3)
rho_sim = np.corrcoef(xs, ys)[0, 1]
m = {k: sp.integrate(v * fxy, (x, 0, 1), (y, 0, 1)) for k, v in {"x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y}.items()}
rho = (m["xy"] - m["x"] * m["y"]) / sp.sqrt((m["xx"] - m["x"] ** 2) * (m["yy"] - m["y"] ** 2))
E.num("s20p01b", rnd(float(rho), 3), "sympy の重積分で相関係数")
E.check("s20p01b", abs(rho_sim - float(rho)) < 0.005, f"シミュレーションの相関 {rho_sim:.4f}")

# ---- 大問2
p = [0.95, 0.90, 0.85]
tot = 0.0
for z in product([0, 1], repeat=3):
    pr = np.prod([pi if zi else 1 - pi for pi, zi in zip(p, z)])
    if sum(z) >= 2:
        tot += pr
E.num("s20p02a", rnd(tot, 3), "8 通りの総当たり")
X = rng.uniform(1, 3, 2_000_000)
Y = rng.uniform(1, 4, 2_000_000)
pint = integrate.quad(lambda xx: (min(5.5 - xx, 4) - 1) / 6, 1, 3)[0]  # x を固定して y の区間の長さを積分
E.choice("s20p02b", f"${f(pint, 4)}$", f"数値積分 {pint:.6f}")
E.check("s20p02b", abs((X + Y <= 5.5).mean() - 0.8125) < 0.002, "シミュレーション")
distinct("s20p02b", 4)

# ---- 大問3
table("s20p03a", math.exp(-4.5), 0.01111, 5, "e^-4.5")
table("s20p03a", math.exp(-1.5), 0.2231, 4, "e^-1.5")
pg = stats.gamma(3, scale=200).sf(900)
E.choice("s20p03a", f"${f(pg, 3)}$", f"scipy gamma(3, 尺度200).sf(900)={pg:.5f}")
distinct("s20p03a", 3)
for v, sh, d in ((1.2 ** 8, 4.300, 3), (1.2 ** 9, 5.160, 3), (1.2 ** 10, 6.192, 3), (math.exp(-1.6), 0.2019, 4), (math.exp(-2.0), 0.1353, 4)):
    table("s20p03b", v, sh, d, f"{sh}")
post = stats.gamma(10, scale=1 / 20)
acc = min(1.0, post.pdf(0.6) / post.pdf(0.5))
E.choice("s20p03b", f"${f(acc, 3)}$", f"scipy の事後ガンマ密度の比 {acc:.5f}")
# 事後分布が形状10・レート20 であることを数値積分で確認
lik = lambda t: t ** 8 * math.exp(-16 * t) * stats.gamma(2, scale=1 / 4).pdf(t)
Z = integrate.quad(lik, 0, np.inf)[0]
E.check("s20p03b", abs(integrate.quad(lambda t: t * lik(t), 0, np.inf)[0] / Z - 0.5) < 1e-8, "事後平均 10/20=0.5（数値積分）")
distinct("s20p03b", 3)

# ---- 大問4
table("s20p04a", math.log(365), 5.900, 3, "log 365")
table("s20p04a", -math.log(0.98), 0.0202, 4, "-log 0.98")
table("s20p04a", math.log(0.0202), -3.902, 3, "log 0.0202")
# 近似を使わない正確な分位点
xq = 10 - 3 * math.log(1 - 0.98 ** (1 / 365))
E.num("s20p04a", rnd(xq, 1), f"(1-e^-y)^365=0.98 を正確に解く {xq:.4f}")
gum = stats.gumbel_r(loc=10 + 3 * math.log(365), scale=3).ppf(0.98)
E.check("s20p04a", rnd(gum, 1) == E.item("s20p04a")["answer"], f"scipy gumbel_r の 98% 点 {gum:.4f}")

# ---- 大問5
for k, lo, hi in ((15, 6.262, 27.49), (16, 6.908, 28.85)):
    table("s20p05a", stats.chi2.ppf(0.025, k), lo, 3, f"χ²({k}) 下側2.5%")
    table("s20p05a", stats.chi2.ppf(0.975, k), hi, 2, f"χ²({k}) 上側2.5%")
E.num("s20p05a", rnd(math.sqrt(2.4 / 6.262), 3), "示した点で上限")
E.check("s20p05a", rnd(math.sqrt(2.4 / stats.chi2.ppf(0.025, 15)), 3) == 0.619, "正確な χ² 点でも 0.619")
# (b) MSE(c) を sympy で最小化し、シミュレーションで確認
c, n = sp.symbols("c"), 16
mse = c ** 2 * 2 * (n - 1) + (c * (n - 1) - 1) ** 2
cs = sp.solve(sp.diff(mse, c), c)[0]
E.check("s20p05b", cs == sp.Rational(1, 17), f"sympy で最小化 c={cs}")
E.choice("s20p05b", f"c=1/17$、推定値$ {f(2.4 / 17, 3)}", "推定値 2.4/17")
Ssim = stats.chi2(15).rvs(size=400_000, random_state=1)  # σ²=1
mses = {d: np.mean((Ssim / d - 1) ** 2) for d in (14, 15, 16, 17, 18)}
E.check("s20p05b", min(mses, key=mses.get) == 17, f"シミュレーションの MSE {({k: round(v, 4) for k, v in mses.items()})}")

# ---- 大問6
for v, sh, d in ((stats.f.ppf(0.95, 2, 10), 4.103, 3), (stats.f.ppf(0.95, 2, 11), 3.982, 3), (stats.t.ppf(0.975, 11), 2.201, 3)):
    table("s20p06a", v, sh, d, f"{sh}")
dbar = np.array([1.2, 0.8])
S = np.array([[4.0, -1.5], [-1.5, 2.25]])
T2 = 12 * dbar @ np.linalg.solve(S, dbar)
Fst = (12 - 2) / (2 * 11) * T2
E.choice("s20p06a", f"$T^2={f(T2, 2)}$、$F={f(Fst, 2)}$で、$F_{{0.05}}(2,10)=4.103$を超えるので棄却する", "numpy で T²・F")
E.check("s20p06a", np.all(np.linalg.eigvalsh(S) > 0), "S は正定値")
E.check("s20p06a", stats.f.sf(Fst, 2, 10) < 0.05 and max(abs(dbar / np.sqrt(np.diag(S) / 12))) < stats.t.ppf(0.975, 11), "F は有意、成分ごとの t は有意でない")
pts = {"A": (0, 0), "B": (1, 1), "C": (3, 0), "D": (2, 3)}
d2 = lambda a, b: sum((np.array(a) - np.array(b)) ** 2)
E.check("s20p06b", min(d2(pts[a], pts[b]) for a, b in [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]) == 2, "最小距離は A-B（最初の併合）")
cen = np.mean([pts["A"], pts["B"]], axis=0)
E.choice("s20p06b", f"${f(d2(cen, pts['C']), 1)}$", "座標から重心と C の距離の2乗を直接計算")
# 表がユークリッド距離の2乗になりうること（古典的MDSの内積行列が半正定値・階数2）
names = "ABCD"
D2 = np.array([[d2(pts[a], pts[b]) for b in names] for a in names], dtype=float)
J = np.eye(4) - 1 / 4
Bm = -0.5 * J @ D2 @ J
ev = np.linalg.eigvalsh(Bm)
E.check("s20p06b", ev.min() > -1e-9 and np.sum(ev > 1e-9) == 2, f"距離表は2次元のユークリッド距離の2乗（固有値 {np.round(ev, 3)}）")
qtxt = E.item("s20p06b")["question"]
E.check("s20p06b", "| A | 0 | 2 | 9 | 13 |" in qtxt and "| C | 9 | 5 | 0 | 10 |" in qtxt, "問題文の距離表が座標と一致")
L = np.array([[0.8, 0.3], [0.7, 0.4], [-0.2, 0.7], [0.6, 0.1]])
th = 0.7
R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
Lr = L @ R
uniq = 1 - (L ** 2).sum(1)
E.check("s20p06c", abs(uniq[1] - 0.35) < 1e-12 and np.allclose(1 - (Lr ** 2).sum(1), uniq), "独自分散 0.35、回転で不変")
E.check("s20p06c", abs((L @ L.T)[0, 1] - 0.68) < 1e-12 and np.allclose(Lr @ Lr.T, L @ L.T), "モデル上の相関 0.68（≠0.56）、回転で不変")
E.check("s20p06c", abs((L[:, 0] ** 2).sum() / 4 - 0.3825) < 1e-12 and abs((Lr[:, 0] ** 2).sum() - (L[:, 0] ** 2).sum()) > 0.01, "寄与率 38.25%（≠76.5%）、因子の寄与は回転で変わる")
E.check("s20p06c", (L[1] ** 2).sum() < 1, "PM2.5 の共通性 0.65 < 1（不適解でない）")
E.choice("s20p06c", "独自分散は$0.35$", "正しい記述の選択肢")

# ---- 大問7
xs7 = np.array([3, 7, 2, 9, 4, 1, 8, 5, 2, 11, 3, 5])
D = ((xs7 - xs7.mean()) ** 2).sum() / xs7.mean()
table("s20p07a", stats.chi2.ppf(0.95, 11), 19.68, 2, "χ²(11) 上側5%")
table("s20p07a", stats.chi2.ppf(0.99, 11), 24.72, 2, "χ²(11) 上側1%")
pv = stats.chi2.sf(D, 11)
E.check("s20p07a", 0.01 < pv < 0.05, f"p 値 {pv:.4f} は 1%〜5%")
E.choice("s20p07a", f"$D={f(D, 1)}$で、5%水準では帰無仮説を棄却するが、1%水準では棄却しない", "結論の選択肢")
# 帰無仮説のもとでのシミュレーションでも p 値が 1〜5%
sim = rng.poisson(5, size=(200_000, 12))
Ds = ((sim - sim.mean(1, keepdims=True)) ** 2).sum(1) / np.maximum(sim.mean(1), 1e-9)
E.check("s20p07a", 0.01 < (Ds >= D).mean() < 0.05, f"シミュレーションの p 値 {(Ds >= D).mean():.4f}")

# ---- 大問8
phi1, phi2 = 0.5, 0.3
# AR(2) の自己共分散を十分長い MA(∞) 表現から計算
psi = [1.0, phi1]
for k in range(2, 2000):
    psi.append(phi1 * psi[-1] + phi2 * psi[-2])
psi = np.array(psi)
g0 = (psi ** 2).sum()
g2 = (psi[:-2] * psi[2:]).sum()
E.choice("s20p08a", f"${f(g2 / g0, 3)}$", f"MA(∞) 表現から ρ2={g2 / g0:.6f}")
roots = np.roots([-phi2, -phi1, 1])  # 1 - φ1 z - φ2 z^2 = 0
E.check("s20p08a", np.all(np.abs(roots) > 1), f"特性方程式の根の絶対値 {np.abs(roots)} > 1（定常）")
distinct("s20p08a", 3)
seq = "+++---++++----++-----++++---++"
R = 1 + sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
# 15 個ずつの並べ方の総当たり（C(30,15) は大きいので、連の数の正確な分布を組合せの式で）
from math import comb
n1 = n2 = 15
def prun(r):
    if r % 2 == 0:
        k = r // 2
        return 2 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1) / comb(30, 15)
    k = (r - 1) // 2
    return (comb(n1 - 1, k) * comb(n2 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(n2 - 1, k)) / comb(30, 15)
dist = {r: prun(r) for r in range(2, 31)}
ER = sum(r * p_ for r, p_ in dist.items())
VR = sum(r * r * p_ for r, p_ in dist.items()) - ER ** 2
E.check("s20p08b", abs(sum(dist.values()) - 1) < 1e-12 and abs(ER - 16) < 1e-9, f"連の数の正確な分布：E[R]={ER:.4f}")
z = (R - ER) / math.sqrt(VR)
E.choice("s20p08b", f"$z={f(z, 2)}$で、連が少なすぎるので棄却する", f"正確な分布の平均・分散から z={z:.4f}（R={R}）")
E.check("s20p08b", sum(p_ for r, p_ in dist.items() if r <= R) < 0.025, "正確な片側確率も 2.5% 未満")

# ---- 大問9 M/M/2：生成行列から定常分布を数値で解く
K = 200
lam, mu = 3.0, 2.0
Q = np.zeros((K, K))
for i in range(K):
    if i + 1 < K:
        Q[i, i + 1] = lam
    if i > 0:
        Q[i, i - 1] = mu * min(i, 2)
    Q[i, i] = -Q[i].sum()
A = np.vstack([Q.T, np.ones(K)])
bb = np.zeros(K + 1); bb[-1] = 1
pi = np.linalg.lstsq(A, bb, rcond=None)[0]
Lq = sum((k - 2) * pi[k] for k in range(2, K))
E.choice("s20p09a", f"${f(Lq / lam * 60, 1)}$分", f"生成行列の定常分布から Wq={Lq / lam * 60:.3f} 分")
# 事象駆動シミュレーション
t, free_at, waits = 0.0, [0.0, 0.0], []
for _ in range(300_000):
    t += rng.exponential(1 / lam)
    j = int(np.argmin(free_at))
    start = max(t, free_at[j])
    waits.append(start - t)
    free_at[j] = start + rng.exponential(1 / mu)
wsim = np.mean(waits[10000:]) * 60
E.check("s20p09a", abs(wsim - 38.57) < 1.5, f"シミュレーションの平均待ち時間 {wsim:.2f} 分")

# ---- 大問10
E.num("s20p10a", rnd(((1800 - 1200) / 3) / (1200 / 35), 2), "部分 F")
table("s20p10a", stats.f.ppf(0.95, 3, 35), 2.874, 3, "F(3,35) 5%")
# 外部スチューデント化残差を、実際に回帰データを作って確かめる
Xd = np.column_stack([np.ones(40), rng.normal(size=(40, 4))])
H = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)
yv = Xd @ np.array([500, 20, 5, 30, -3]) + rng.normal(0, 6, 40)
e_ = yv - H @ yv
i0 = 7
# 残差 i0 とてこ比を問題の値に合わせるのは難しいので、公式（再当てはめなしの式）を実データで確認し、問題の値に当てはめる
mask = np.arange(40) != i0
b_i = np.linalg.lstsq(Xd[mask], yv[mask], rcond=None)[0]
s2_i = ((yv[mask] - Xd[mask] @ b_i) ** 2).sum() / (40 - 5 - 1)
s2_formula = ((e_ ** 2).sum() - e_[i0] ** 2 / (1 - H[i0, i0])) / (40 - 5 - 1)
E.check("s20p10b", abs(s2_i - s2_formula) < 1e-9, "1 点を除いた誤差分散の公式を再当てはめで確認")
t_ext = 14 / math.sqrt((1200 - 14 ** 2 / 0.7) / 34 * 0.7)
r_int = 14 / math.sqrt(1200 / 35 * 0.7)
E.choice("s20p10b", f"${f(r_int * math.sqrt(34 / (35 - r_int ** 2)), 2)}$", "内部→外部の変換式 r√((n−p−1)/(n−p−r²))")
E.check("s20p10b", abs(t_ext - r_int * math.sqrt(34 / (35 - r_int ** 2))) < 1e-12, "2 つの式が一致")
distinct("s20p10b", 2)
# IV：データを発生させて推定量が真値に近いことと、OLS が上にずれることを確認（式の向きの確認）
nn = 400_000
zz = rng.normal(size=nn); uu = rng.normal(size=nn)
lp = 0.8 * zz + 0.5 * uu + rng.normal(size=nn)
ly = -0.3 * lp + uu
cz = np.cov(zz, ly)[0, 1] / np.cov(zz, lp)[0, 1]
E.check("s20p10c", abs(cz + 0.3) < 0.02 and np.cov(lp, ly)[0, 1] / lp.var() > -0.3, f"IV={cz:.3f}≈真値 −0.3、OLS は上向きに偏る")
# 2SLS を行列で
Z = np.column_stack([np.ones(nn), zz]); Xm = np.column_stack([np.ones(nn), lp])
Xh = Z @ np.linalg.lstsq(Z, Xm, rcond=None)[0]
b2s = np.linalg.lstsq(Xh, ly, rcond=None)[0][1]
E.check("s20p10c", abs(b2s - cz) < 1e-8, "2SLS と IV 推定量が一致")
E.choice("s20p10c", f"${f(-0.024 / 0.080, 2)}$", "問題の共分散で Cov(z,y)/Cov(z,x)")

# ---- 大問11
for v, sh, d in ((math.exp(0.8), 2.226, 3), (math.exp(2.3), 9.974, 3), (math.exp(2.8), 16.44, 2)):
    table("s20p11a", v, sh, d, f"{sh}")
eta = 0.2 * 5 + 0.8
cum = lambda th: 1 / (1 + math.exp(-(th - eta)))
probs = [cum(-0.5), cum(1.0) - cum(-0.5), 1 - cum(1.0)]
E.check("s20p11a", abs(sum(probs) - 1) < 1e-12 and min(probs) > 0, "3 カテゴリーの確率が正で和が 1")
E.choice("s20p11a", f"${f(probs[2], 3)}$", f"正確な e^x で P(Y=3)={probs[2]:.5f}")
# 潜在変数表現 Y*=βx+ε（ε はロジスティック）からのシミュレーション
ys_ = eta + rng.logistic(size=1_000_000)
E.check("s20p11a", abs((ys_ > 1.0).mean() - probs[2]) < 0.002, "潜在変数のシミュレーション")
distinct("s20p11a", 3)

# ---- 大問12 ラテン方格：実データを作って平方和の分解と自由度を確認
for v, sh, d in ((stats.f.ppf(0.95, 3, 6), 4.757, 3), (stats.f.ppf(0.95, 3, 9), 3.863, 3), (stats.f.ppf(0.95, 3, 12), 3.490, 3)):
    table("s20p12a", v, sh, d, f"{sh}")
k = 4
sq = np.array([[(i + j) % k for j in range(k)] for i in range(k)])
yl = rng.normal(size=(k, k))
gm = yl.mean()
ssr = k * ((yl.mean(1) - gm) ** 2).sum(); ssc = k * ((yl.mean(0) - gm) ** 2).sum()
sst_ = k * sum((yl[sq == t].mean() - gm) ** 2 for t in range(k)); stot = ((yl - gm) ** 2).sum()
# 加法モデルの最小二乗当てはめの残差平方和と残差自由度
Xl = [np.ones(16)]
for t in range(1, k):
    Xl += [np.repeat(np.arange(k) == t, k).astype(float), np.tile(np.arange(k) == t, k).astype(float), (sq.ravel() == t).astype(float)]
Xl = np.column_stack(Xl)
res = yl.ravel() - Xl @ np.linalg.lstsq(Xl, yl.ravel(), rcond=None)[0]
E.check("s20p12a", abs((res ** 2).sum() - (stot - ssr - ssc - sst_)) < 1e-9 and 16 - np.linalg.matrix_rank(Xl) == 6, "誤差平方和は引き算で求まり、自由度は 6")
Fl = (45 / 3) / ((141 - 30 - 48 - 45) / 6)
E.choice("s20p12a", f"$F={f(Fl, 2)}$で、$F_{{0.05}}(3,6)=4.757$を超えるので有意である", "F=5.00")
E.check("s20p12a", stats.f.sf(Fl, 3, 6) < 0.05, f"p 値 {stats.f.sf(Fl, 3, 6):.4f}")

# ---- 大問13
nmin = next(nv for nv in range(1, 8000) if 1.96 * math.sqrt((1 - nv / 8000) * 160 ** 2 / nv) <= 10)
E.num("s20p13a", nmin, "条件を満たす最小の n を総当たり")
# 数量化I類：仮想データを作って、基準化後スコアが重み付き平均 0 の最小二乗解になることを確認
house = np.array([1] * 4 + [0] * 6)          # 戸建て 0.4
pers = np.array([0, 1, 2, 2, 0, 0, 1, 1, 2, 2])  # 1人 0.3・2人 0.3・3人以上 0.4
E.check("s20p13b", house.mean() == 0.4 and [np.mean(pers == j) for j in range(3)] == [0.3, 0.3, 0.4], "仮想データの構成比")
yq = 250 + 80 * house + np.array([0, 60, 130])[pers]
scores_h = {1: 80 - 80 * house.mean(), 0: -80 * house.mean()}
w = np.array([np.mean(pers == j) for j in range(3)])
sp_ = np.array([0, 60, 130]) - (w * np.array([0, 60, 130])).sum()
pred = yq.mean() + np.array([scores_h[h] for h in house]) + sp_[pers]
E.check("s20p13b", np.allclose(pred, yq) and abs(yq.mean() - 352) < 1e-9, f"基準化後も予測値は同じ・定数項 = 平均 {yq.mean()}")
E.choice("s20p13b", f"定数項{f(yq.mean(), 0)}、2人のスコア{f(sp_[1], 0)}", "仮想データで基準化")
# IPW：個票を作って計算
rows = []
for n_, t_, yv_ in ((240, 1, 380), (160, 0, 420)):
    rows += [("h", t_, yv_)] * n_
for n_, t_, yv_ in ((120, 1, 250), (480, 0, 270)):
    rows += [("a", t_, yv_)] * n_
grp = np.array([r_[0] for r_ in rows]); T_ = np.array([r_[1] for r_ in rows]); Y_ = np.array([r_[2] for r_ in rows], float)
ps = {g: T_[grp == g].mean() for g in ("h", "a")}
e_i = np.array([ps[g] for g in grp])
ate = np.mean(T_ * Y_ / e_i) - np.mean((1 - T_) * Y_ / (1 - e_i))
E.num("s20p13c", rnd(ate, 1), f"個票 1000 件から IPW {ate:.4f}")
# 重み付き回帰（Hajek 型）でも同じ
w1 = T_ / e_i; w0 = (1 - T_) / (1 - e_i)
E.check("s20p13c", abs((w1 @ Y_) / w1.sum() - (w0 @ Y_) / w0.sum() - ate) < 1e-9, "ハーイェク型の IPW とも一致")

# ---- 大問14
Qv = np.array([4.12, 4.20, 4.05, 4.18, 4.15]); Uv = np.array([0.0016, 0.0018, 0.0015, 0.0017, 0.0019])
Tt = Uv.mean() + (1 + 1 / 5) * Qv.var(ddof=1)
E.num("s20p14a", rnd(math.sqrt(Tt), 3), f"numpy でルービンのルール SE={math.sqrt(Tt):.5f}")

E.done()
