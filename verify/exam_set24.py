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

# ---- 大問2
B = 1_000_000
cost = np.array([0.5, 0.3, 0.2]) * B
clicks = cost / np.array([80, 40, 160])
E.choice("s24p02a", f"${f(B / clicks.sum(), 1)}$", "総費用÷総クリック数を直接計算")
# (b) 費用制約のもとで分散を数値最小化
N = np.array([40000, 10000])
S = np.array([12, 20])
c = np.array([100, 900])
W = N / N.sum()
C = 540000
obj = lambda n2: (W[0] ** 2 * S[0] ** 2 / ((C - c[1] * n2) / c[0]) + W[1] ** 2 * S[1] ** 2 / n2)
res = optimize.minimize_scalar(obj, bounds=(1, C / c[1] - 1), method="bounded", options={"xatol": 1e-8})
E.num("s24p02b", round(res.x), f"分散を数値最小化した n2={res.x:.3f}")
n2s = np.arange(300, 370)
best = n2s[np.argmin([obj(k) for k in n2s])]
E.check("s24p02b", best == 333, f"整数の総当たりでも n2={best}")

# ---- 大問3
phi = {0.8: 0.7881, 1.2: 0.8849, 1.6: 0.9452, 2.4: 0.9918, 0.6: 0.7257, 0.72: 0.7642}
E.check("s24p03a", all(abs(stats.norm.cdf(k) - v) < 6e-5 for k, v in phi.items()), "Φ の表の値")
Wv = rng.normal(10, 2.5, 1_000_000)
Yv = (0.5 * Wv + 1) ** 2
E.check("s24p03a", abs((Yv > 64).mean() - stats.norm.sf(1.6)) < 0.001, f"逆変換したシミュレーション {(Yv > 64).mean():.4f}")
E.choice("s24p03a", f"${f(stats.norm.sf((2 * (8 - 1) - 10) / 2.5), 4)}$", "scipy norm.sf")
sig = 1.2
Ylog = np.exp(rng.normal(0.7, sig, 1_000_000))   # μ は任意（0.7）
frac = (Ylog > Ylog.mean()).mean()
E.check("s24p03b", abs(frac - stats.norm.sf(sig / 2)) < 0.002, f"シミュレーションで平均を上回る割合 {frac:.4f}")
E.choice("s24p03b", f"${f(stats.lognorm(sig, scale=math.exp(-2.0)).sf(stats.lognorm(sig, scale=math.exp(-2.0)).mean()), 4)}$",
         "scipy lognorm の平均と生存関数（μ=-2 でも同じ）")

# ---- 大問4
mx, my, sx, sy, rho, n = 50.0, 1.5, 10.0, 0.4, 0.8, 25
a_, b_ = sp.symbols("a b", positive=True)
g = b_ / a_
grad = sp.Matrix([sp.diff(g, a_), sp.diff(g, b_)]).subs({a_: mx, b_: my})
Sig = sp.Matrix([[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]]) / n
se_r = float(sp.sqrt((grad.T * Sig * grad)[0]))
E.num("s24p04a", rnd(se_r * 100, 3), f"sympy の勾配ベクトルで SE={se_r * 100:.5f}")
cov = [[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]]
XY = rng.multivariate_normal([mx, my], cov, size=(100_000, n))
Rh = XY[:, :, 1].mean(1) / XY[:, :, 0].mean(1) * 100
E.check("s24p04a", abs(Rh.std() - 0.0963) < 0.002, f"シミュレーションの SD {Rh.std():.4f}")
pm = stats.binom(20, 0.1).pmf(range(6))
E.check("s24p04b", all(f(pm[k], 4) in str(E.item("s24p04b")["question"]) for k in range(6)), "二項確率の表の値")
pv = stats.binomtest(5, 20, 0.1, alternative="greater").pvalue
E.choice("s24p04b", f"${f(pv, 3)}$で、有意水準5%で棄却するが1%", f"scipy binomtest p={pv:.5f}")
E.check("s24p04b", 0.01 < pv < 0.05, "5% で有意・1% で有意でない")

# ---- 大問5
v1, v2 = 30 ** 2 / 12, 12 ** 2 / 10
nu = (v1 + v2) ** 2 / (v1 ** 2 / 11 + v2 ** 2 / 9)
lo = 32 - stats.t.ppf(0.975, math.floor(nu)) * math.sqrt(v1 + v2)
E.num("s24p05a", rnd(lo, 2), f"scipy t.ppf(0.975,{math.floor(nu)}) で下限 {lo:.4f}")
E.check("s24p05a", all(abs(stats.t.ppf(0.975, k) - v) < 6e-4 for k, v in [(14, 2.145), (15, 2.131), (20, 2.086), (21, 2.080)]), "t 表の値")
# 被覆率のシミュレーション（ウェルチ区間がほぼ 95%）
A = rng.normal(0, 30, (60_000, 12))
Bm = rng.normal(0, 12, (60_000, 10))
va, vb = A.var(1, ddof=1) / 12, Bm.var(1, ddof=1) / 10
nn = np.floor((va + vb) ** 2 / (va ** 2 / 11 + vb ** 2 / 9))
hw = stats.t.ppf(0.975, nn) * np.sqrt(va + vb)
cover = (np.abs(A.mean(1) - Bm.mean(1)) <= hw).mean()
E.check("s24p05a", abs(cover - 0.95) < 0.006, f"ウェルチ区間の被覆率 {cover:.4f}")
tc = stats.t.ppf(0.975, 14)
tb = 5.4 / (4.2 / tc)
pt = 2 * stats.t.sf(tb, 14)
E.choice("s24p05b", f"\\approx{f(tb, 2)}$で、両側p値は0.01と0.02の間", f"t={tb:.4f}, p={pt:.4f}")
E.check("s24p05b", 0.01 < pt < 0.02, "両側 p 値が 0.01〜0.02")
E.check("s24p05b", all(abs(stats.t.isf(q, 14) - v) < 6e-4 for q, v in [(0.05, 1.761), (0.01, 2.624), (0.005, 2.977)]), "t 表の値")

# ---- 大問6
xs = np.array([3.2, 4.0, 5.5, 3.6, 8.1, 4.4, 6.0, 5.2])
nll = lambda p: np.inf if (p[1] <= 0 or p[0] > xs.min()) else -stats.expon(loc=p[0], scale=p[1]).logpdf(xs).sum()
grid = [(t, b) for t in np.linspace(2.5, 3.2, 71) for b in np.linspace(1.0, 3.0, 201)]
tb_best = min(grid, key=nll)
E.check("s24p06a", abs(tb_best[0] - 3.2) < 1e-9 and abs(tb_best[1] - 1.8) < 0.011, f"格子で最大化した (θ,β)={tb_best}")
E.choice("s24p06a", f"=({f(tb_best[0], 1)},\\{f(tb_best[1], 1)})$", "格子探索の最尤推定値")
E.check("s24p06a", abs(stats.expon.fit(xs)[0] - 3.2) < 1e-9 and abs(stats.expon.fit(xs)[1] - 1.8) < 1e-9, "scipy expon.fit と一致")
x = stats.expon(loc=3.2, scale=1.8).ppf(0.75)
E.choice("s24p06b", f"${f(x, 2)}$", "scipy expon.ppf(0.75)")
E.check("s24p06b", abs(math.log(0.25) + 1.3863) < 5e-5 and abs(math.log(0.75) + 0.2877) < 5e-5, "対数の値")

# ---- 大問7
k = 6
Ek = sum(Fraction(k, k - j) for j in range(k))
E.num("s24p07a", float(Ek), f"分数の和 {Ek}")
cnt = []
for _ in range(100_000):
    seen, m = set(), 0
    while len(seen) < k:
        seen.add(int(rng.integers(k)))
        m += 1
    cnt.append(m)
E.check("s24p07a", abs(np.mean(cnt) - float(Ek)) < 0.06, f"シミュレーション平均 {np.mean(cnt):.3f}")
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

# ---- 大問8：要約統計量を満たすデータを作って numpy で当てはめる
n = 12
x = rng.normal(size=n)
x = (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).sum()) * np.sqrt(20) + 3
X = np.column_stack([np.ones(n), x])
e = rng.normal(size=n)
e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
e = e / np.sqrt(e @ e) * np.sqrt(40)
y = 2 + 1.5 * x + e
bh = np.linalg.lstsq(X, y, rcond=None)[0]
s2 = ((y - X @ bh) ** 2).sum() / (n - 2)
x0 = np.array([1, 5.0])
sem = np.sqrt(s2 * x0 @ np.linalg.inv(X.T @ X) @ x0)
t10 = stats.t.ppf(0.975, 10)
E.check("s24p08a", abs(bh[0] - 2) < 1e-9 and abs(bh[1] - 1.5) < 1e-9 and abs(s2 - 4) < 1e-9, "作ったデータの当てはめが問題文と一致")
E.choice("s24p08a", f"({f(x0 @ bh - t10 * sem, 2)},\\{f(x0 @ bh + t10 * sem, 2)})", "行列計算と正確な t 点で区間")
E.check("s24p08a", abs(t10 - 2.228) < 5e-4 and abs(stats.t.ppf(0.975, 11) - 2.201) < 5e-4, "t 表の値")
# (b) 1 点除いて当てはめ直す定義と公式の一致（乱数データ）を確かめてから、表の値で計算
H = X @ np.linalg.inv(X.T @ X) @ X.T
i = 3
bi = np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0]
cook_def = ((X @ bh - X @ bi) ** 2).sum() / (2 * s2)
ei, hi = (y - X @ bh)[i], H[i, i]
E.check("s24p08b", abs(cook_def - ei ** 2 * hi / (2 * s2 * (1 - hi) ** 2)) < 1e-10, "当てはめ直しの定義と公式が一致")
D = {kk: ee ** 2 * hh / (2 * 4.0 * (1 - hh) ** 2) for kk, (ee, hh) in {"A": (3.6, 0.096), "B": (-1.5, 0.533), "C": (2.4, 0.283)}.items()}
E.check("s24p08b", max(D, key=D.get) == "B" and D["B"] > 4 / 12 and D["C"] > 4 / 12 and D["A"] < 4 / 12, f"D={D}")
E.choice("s24p08b", f"約{f(D['B'], 2)}）で、目安$4/n$を超えるのはBとC", "B が最大・B と C が超える")

# ---- 大問9：直交計画を作り、目的関数を係数ごとに細かい格子で最小化
b = np.array([2.4, -0.6, 1.1, 0.3, -1.8, 0.9])
Q, _ = np.linalg.qr(rng.normal(size=(30, 6)))
yv = Q @ b + (np.eye(30) - Q @ Q.T) @ rng.normal(size=30)
E.check("s24p09a", np.allclose(Q.T @ yv, b), "X'y=b となるデータ")
grid = np.linspace(-3, 3, 600001)
est = []
for j in range(6):
    vals = (b[j] - grid) ** 2 + 1.2 * np.abs(grid) + 0.5 * grid ** 2
    est.append(grid[np.argmin(vals)])
est = np.array(est)
full = lambda be: ((yv - Q @ be) ** 2).sum() + 1.2 * np.abs(be).sum() + 0.5 * (be ** 2).sum()
E.check("s24p09a", all(full(est) <= full(est + d) + 1e-9 for d in rng.normal(0, 0.01, (200, 6))), "全体の目的関数でも局所的に最小")
nz = int((np.abs(est) > 1e-4).sum())
E.choice("s24p09a", f"0でない係数は{nz}個、$\\hat\\beta_1={f(est[0], 2)}$", f"格子最小化 {np.round(est, 4)}")

# ---- 大問10
tab = np.array([[30, 70], [20, 180]])
v = np.repeat([1, 1, 0, 0], tab.ravel())
bz = np.repeat([1, 0, 1, 0], tab.ravel())
ph = np.corrcoef(v, bz)[0, 1]
E.choice("s24p10a", f"${f(ph, 3)}$", "0-1 データを展開した相関係数")
chi = stats.chi2_contingency(tab, correction=False)[0]
E.check("s24p10a", abs(chi / 300 - ph ** 2) < 1e-12, "χ²/n = φ²")
ll0 = stats.bernoulli(50 / 300).logpmf(bz).sum()
E.choice("s24p10b", f"${f(1 - (-112.0) / ll0, 3)}$", f"scipy の対数尤度 log L0={ll0:.4f}")
E.check("s24p10b", abs(math.log(1 / 6) + 1.7918) < 5e-5 and abs(math.log(5 / 6) + 0.1823) < 5e-5, "対数の値")
# 切片だけのロジスティック回帰を数値最大化しても同じ
opt = optimize.minimize_scalar(lambda a: -(bz * a - np.log1p(np.exp(a))).sum())
E.check("s24p10b", abs(-opt.fun - ll0) < 1e-6, "切片のみロジスティック回帰の最大対数尤度と一致")

# ---- 大問11
hum = [2, 4, 6, 8, 11, 19]
bot = [13, 15, 17, 20, 24, 29]
ks = stats.ks_2samp(hum, bot, method="exact")
E.check("s24p11a", abs(ks.statistic - 5 / 6) < 1e-12 and abs(ks.pvalue - 0.0260) < 5e-5, f"scipy ks_2samp D={ks.statistic:.4f} p={ks.pvalue:.4f}")
cntD = {}
for cmb in itertools.combinations(range(12), 6):
    sset = set(cmb)
    fa = fb = Dm = 0
    for ii in range(12):
        if ii in sset:
            fa += 1
        else:
            fb += 1
        Dm = max(Dm, abs(fa - fb))
    cntD[Dm] = cntD.get(Dm, 0) + 1
tail = lambda d: sum(v for kk, v in cntD.items() if kk >= d) / 924
E.check("s24p11a", [f(tail(d), 4) for d in (6, 5, 4, 3)] == ["0.0022", "0.0260", "0.1429", "0.4740"], "924 通りの総当たりで P(D≥d)")
grid_x = np.arange(0, 31) + 0.5
Fh = np.array([(np.array(hum) <= t).mean() for t in grid_x])
Fb = np.array([(np.array(bot) <= t).mean() for t in grid_x])
E.check("s24p11a", (Fh >= Fb).all(), "一般利用者の経験分布関数がつねに上（ボットは大きい側）")
E.choice("s24p11a", "$D=5/6$で、p値は0.0260なので有意水準5%で棄却する", "正解の記述")
fdiff = lambda z: stats.norm.pdf(z, 3, 2) - stats.norm.pdf(z, 0, 1)
r1 = optimize.brentq(fdiff, 0, 3)
r2 = optimize.brentq(fdiff, -6, -1)
zz = np.linspace(-8, 8, 16001)
botreg = fdiff(zz) > 0
E.check("s24p11b", ((zz > r1) | (zz < r2) == botreg)[np.abs(zz - r1) > 1e-3][np.abs(zz[np.abs(zz - r1) > 1e-3] - r2) > 1e-3].all(), "密度の比較でボット領域が2根の外側")
E.choice("s24p11b", f"$z>{f(r1, 2)}$または$z<{f(r2, 2)}$", f"brentq の根 {r1:.4f}, {r2:.4f}")

# ---- 大問12
V = np.column_stack([np.ones(3) / np.sqrt(3), np.array([1, 0, -1]) / np.sqrt(2), np.array([1, -2, 1]) / np.sqrt(6)])
Sg = V @ np.diag([4.0, 1.0, 0.25]) @ V.T
xv = np.array([1.5, 0.5, 2.5])
d2 = xv @ np.linalg.solve(Sg, xv)
E.num("s24p12a", rnd(d2, 2), f"Σ を組み立てて x'Σ^-1 x={d2:.5f}")
E.check("s24p12a", np.allclose(V.T @ V, np.eye(3)) and np.allclose(np.linalg.eigvalsh(Sg), [0.25, 1, 4]), "固有ベクトルが正規直交で固有値が一致")
Rfull = np.eye(4)
Cx = np.array([[0.5, 0.3], [0.2, 0.4]])
Rfull[:2, 2:] = Cx
Rfull[2:, :2] = Cx.T
E.check("s24p12b", np.linalg.eigvalsh(Rfull).min() > 0, "4 変数の相関行列が正定値")
Mm = np.linalg.inv(Rfull[:2, :2]) @ Rfull[:2, 2:] @ np.linalg.inv(Rfull[2:, 2:]) @ Rfull[2:, :2]
rho1 = math.sqrt(max(np.linalg.eigvals(Mm).real))
E.choice("s24p12b", f"${f(rho1, 3)}$", f"R_xx^-1 R_xy R_yy^-1 R_yx の固有値 ρ1={rho1:.4f}")
Z = rng.multivariate_normal(np.zeros(4), Rfull, size=200_000)
from numpy.linalg import svd  # noqa: E402
Zs = (Z - Z.mean(0)) / Z.std(0)
Xs, Ys = Zs[:, :2], Zs[:, 2:]
Lx = np.linalg.cholesky(np.cov(Xs.T))
Ly = np.linalg.cholesky(np.cov(Ys.T))
Kmat = np.linalg.inv(Lx) @ np.cov(Xs.T, Ys.T)[:2, 2:] @ np.linalg.inv(Ly).T
E.check("s24p12b", abs(svd(Kmat)[1][0] - rho1) < 0.005, f"シミュレーションの第1正準相関 {svd(Kmat)[1][0]:.4f}")

# ---- 大問13：±1 の列で ABC の符号を作り、日の区分と効果の列の直交性を確かめる
runs = list(itertools.product([-1, 1], repeat=3))
name = lambda r_: "".join(ch for ch, v_ in zip("abc", r_) if v_ == 1) or "(1)"
abc = {name(r_): r_[0] * r_[1] * r_[2] for r_ in runs}
day1 = sorted(nm for nm, sg in abc.items() if sg == abc["(1)"])
E.check("s24p13a", day1 == sorted(["(1)", "ab", "ac", "bc"]), f"(1) と同じ ABC 符号の組 {day1}")
blk = np.array([1 if abc[name(r_)] == abc["(1)"] else -1 for r_ in runs])
cols = {}
for kk in range(1, 4):
    for comb in itertools.combinations(range(3), kk):
        cols["".join("ABC"[j] for j in comb)] = np.array([np.prod([r_[j] for j in comb]) for r_ in runs])
conf = [kk for kk, col in cols.items() if abs(col @ blk) == 8]
orth = [kk for kk, col in cols.items() if col @ blk == 0]
E.check("s24p13a", conf == ["ABC"] and len(orth) == 6, f"日と交絡する効果 {conf}、直交する効果 {orth}")
E.choice("s24p13a", "\\{(1),\\ab,\\ac,\\bc\\}$、日と交絡するのはABC", "正解の組")

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

E.done()
