"""模擬試験 第28回（set28）の検算。generator/set28.py とは別の方法（scipy の分布関数・数値積分・
シミュレーション・sympy・全列挙・座標からの直接計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
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

E = Exam(28)
rng = np.random.default_rng(20261028)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    out = []
    for c in E.item(iid)["choices"]:
        out.append([float(v) for v in re.findall(r"-?\d+\.?\d*", str(c))])
    return out


def nearest(iid, value):
    """選択肢の（最初の）数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    vs = [n[0] for n in nums(iid)]
    k = int(np.argmin([abs(x - value) for x in vs]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def table(iid, cond, what):
    E.check(iid, bool(cond), "数表の値: " + what)


# ---- 大問1
table("s28p01a", abs(math.exp(-0.5) - 0.6065) < 5e-5 and abs(math.exp(-1) - 0.3679) < 5e-5, "e^-0.5=0.6065, e^-1=0.3679")
y_b = stats.poisson(0.5 * 2.0).pmf(0)
E.num("s28p01a", rnd(290 * 0.3679, 1), "示した e^-1 で 290×P(X=0)")
E.check("s28p01a", rnd(290 * y_b, 1) == E.item("s28p01a")["answer"], f"scipy の正確な P(X=0) でも同じ丸め値 {290 * y_b:.4f}")
# (b) sympy で微分してデルタ法、シミュレーションでも確かめる
table("s28p01b", abs(math.exp(-0.6) - 0.5488) < 5e-5, "e^-0.6=0.5488")
l = sp.symbols("l", positive=True)
se = float(sp.Abs(sp.diff(sp.exp(-l), l)).subs(l, sp.Rational(3, 5)) * sp.sqrt(sp.Rational(3, 5) / 40))
E.choice("s28p01b", f"${f(se, 4)}$", "sympy の微分によるデルタ法")
X = rng.poisson(0.6, size=(200_000, 40)).mean(1)
sim_sd = np.exp(-X).std()
E.check("s28p01b", abs(sim_sd - se) < 0.003, f"シミュレーションの e^(-Xbar) の標準偏差 {sim_sd:.4f} ≈ {se:.4f}")
# (c) ガンマ分布（平均0.6, sd0.3）で Λ を発生させるシミュレーション＋全分散の公式（sympy）
shape, scale = (0.6 / 0.3) ** 2, 0.3 ** 2 / 0.6
Lam = rng.gamma(shape, scale, size=1_000_000)
S = rng.poisson(10 * Lam)
E.check("s28p01c", abs(S.var() - 15) < 0.15, f"シミュレーションの V[S] = {S.var():.3f}")
Lsym = sp.symbols("L")
vS = 10 * sp.Rational(6, 10) + 100 * sp.Rational(9, 100)
E.choice("s28p01c", f"${f(float(vS), 1)}$", "全分散の公式（有理数で）")
nearest("s28p01c", S.var())

# ---- 大問2
table("s28p02a", abs(math.log(0.9) - -0.10536) < 5e-6, "ln0.9=-0.10536")
w = stats.weibull_min(2, scale=4000)
E.num("s28p02a", rnd(w.ppf(0.10), 0), f"scipy weibull_min.ppf(0.10) = {w.ppf(0.10):.3f}")
table("s28p02b", abs(math.log(0.25) - -1.3863) < 5e-5 and abs(math.log(0.75) - -0.2877) < 5e-5, "ln0.25, ln0.75")
E.choice("s28p02b", f"${f(w.ppf(0.75), 0)}$", "scipy weibull_min.ppf(0.75)")
u = rng.uniform(size=400_000)
tt = 4000 * np.sqrt(-np.log(1 - u))
E.check("s28p02b", abs(np.mean(tt <= w.ppf(0.10)) - 0.10) < 0.003, "逆関数法の乱数がワイブル分布に従う（10%点）")

# ---- 大問3
# (a) 5 台の状態を全列挙
pr = {"p": 0.9, "c": 0.95, "v": 0.8}
comps = ["p", "p", "c", "v", "v"]
R = 0.0
for st in itertools.product([0, 1], repeat=5):
    prob = np.prod([pr[c] if s_ else 1 - pr[c] for c, s_ in zip(comps, st)])
    if (st[0] or st[1]) and st[2] and (st[3] or st[4]):
        R += prob
E.choice("s28p03a", f"${f(R, 4)}$", "32 通りの全列挙")
# (b) 数値積分とシミュレーション
P, _ = integrate.dblquad(lambda y, x: 1.0, 0, 0.1, lambda x: 0, lambda x: min(0.1, 0.0025 / x) if x > 0 else 0.1)
P /= 0.01
E.choice("s28p03b", f"${f(P, 4)}$", f"二重積分 P = {P:.6f}")
Wsim = rng.uniform(0, 0.1, 1_000_000) * rng.uniform(0, 0.1, 1_000_000)
E.check("s28p03b", abs(np.mean(Wsim <= 0.0025) - P) < 0.002, "シミュレーション")
table("s28p03b", rnd(0.25 - 0.25 * -1.3863, 4) == rnd(P, 4), "ln0.25 の丸め値でも同じ答え")

# ---- 大問4
A = np.array([512.4, 498.7, 505.1, 520.3, 509.8, 501.6, 515.2, 497.9, 507.5, 511.0])
Bv = np.array([511.2, 497.9, 505.4, 518.8, 508.9, 501.2, 514.1, 498.1, 506.8, 510.0])
it = E.item("s28p04a")
E.check("s28p04a", "511.2" in E.items["s28p04a"][0]["context"], "表の B の値が generator と一致")
res = stats.ttest_rel(A, Bv)
E.num("s28p04a", rnd(res.statistic, 2), f"scipy ttest_rel t = {res.statistic:.4f}")
table("s28p04a", abs(stats.t.ppf(0.975, 9) - 2.262) < 5e-4, "t_0.025(9)=2.262")
p = stats.binomtest(int(np.sum(A - Bv > 0)), 10, 0.5, alternative="greater").pvalue
E.choice("s28p04b", f"${f(p, 4)}$", "scipy binomtest（片側）")

# ---- 大問5
table("s28p05a", abs(math.log(0.05) - -2.9957) < 5e-5 and abs(math.log(0.025) - -3.6889) < 5e-5, "ln0.05, ln0.025")
table("s28p05a", abs(math.exp(-0.04993) - 0.95130) < 5e-6 and abs(math.exp(-0.06148) - 0.94037) < 5e-6, "e^-0.04993, e^-0.06148")
cp = stats.beta.ppf(0.95, 1, 60)  # クロッパー・ピアソン片側上限（x=0）
E.choice("s28p05a", f"${f(cp, 4)}$", f"beta.ppf(0.95,1,60) = {cp:.6f}")
E.check("s28p05a", abs(stats.binom(60, cp).pmf(0) - 0.05) < 1e-10, "上限で 0 個の確率が 5%")
table("s28p05b", abs(math.log(0.45) - -0.7985) < 5e-5 and abs(stats.chi2.ppf(0.95, 1) - 3.84) < 5e-3, "ln0.45, χ²_0.05(1)")
x = rng.exponential(1, 10)
x = x / x.sum() * 4500
ll = lambda th: stats.expon(scale=th).logpdf(x).sum()
opt = optimize.minimize_scalar(lambda th: -ll(th), bounds=(10, 5000), method="bounded")
lr = 2 * (ll(opt.x) - ll(1000))
E.num("s28p05b", rnd(lr, 2), f"数値最適化した対数尤度から -2lnΛ = {lr:.4f}")

# ---- 大問6
xs, th, n = sp.symbols("x theta n", positive=True)
nn = 10
dens = nn * xs ** (nn - 1) / th ** nn
m1 = sp.integrate(xs * dens, (xs, 0, th))
m2 = sp.integrate(xs ** 2 * dens, (xs, 0, th))
v2 = sp.simplify(sp.Rational(nn + 1, nn) ** 2 * (m2 - m1 ** 2))
v1 = sp.simplify(4 * (th ** 2 / 12) / nn)
ratio = float(sp.simplify(v1 / v2))
E.choice("s28p06a", f"${f(ratio, 2)}$", f"sympy の積分で分散の比 {ratio}")
U = rng.uniform(0, 1, size=(200_000, 10))
E.check("s28p06a", abs((2 * U.mean(1)).var() / (1.1 * U.max(1)).var() - 4) < 0.1, "シミュレーションの分散の比 ≈ 4")
fc = 4.0
for y in (5.0, 3.6):
    fc = fc + 0.3 * (y - fc)  # 誤差修正形
E.choice("s28p06b", f"${f(fc, 2)}$", "誤差修正形 F+α(y−F) で2回更新")

# ---- 大問7
V = np.array([[1.5 ** 2, 1.5], [1.5, 1.8 ** 2]])
E.check("s28p07a", np.all(np.linalg.eigvalsh(V) > 0), "共分散行列が正定値")
c = np.array([1, -1])
tv = (c @ np.array([14.0, 9.5])) / math.sqrt(c @ V @ c)
E.choice("s28p07a", f"${f(tv, 2)}$", "対比ベクトル c'Vc から t")
table("s28p07a", abs(stats.t.ppf(0.975, 17) - 2.110) < 5e-4, "t_0.025(17)=2.110")
# (b) 無作為なデータで PRESS の恒等式を確かめ、その式で答えを出す
Xd = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
yd = Xd @ np.array([1, 2, 3]) + rng.normal(size=20)
H = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)
e = yd - H @ yd
bl = np.linalg.lstsq(np.delete(Xd, 6, 0), np.delete(yd, 6), rcond=None)[0]
E.check("s28p07b", abs((yd[6] - Xd[6] @ bl) - e[6] / (1 - H[6, 6])) < 1e-9, "1点除外の当てはめ直しと e/(1−h) が一致")
E.num("s28p07b", rnd(48.0 - (48.0 - 51.2) / (1 - 0.30), 2), "PRESS 残差の式")
# (c) フーバーの ψ(u)/u を直接
psi = lambda u, k=1.345: np.clip(u, -k, k)
uu = -6.4 / 2.0
E.choice("s28p07c", f"${f(psi(uu) / uu, 3)}$", "ψ(u)/u")

# ---- 大問8
tabs = [(8, 92, 18, 82), (4, 76, 11, 69)]
num = 0.0
var = 0.0
for a, b, c_, d in tabs:
    hg = stats.hypergeom(a + b + c_ + d, a + c_, a + b)  # 全体, 不良の数, 新方式の数
    num += a - hg.mean()
    var += hg.var()
cmh = num ** 2 / var
E.num("s28p08a", rnd(cmh, 2), f"scipy 超幾何の平均・分散から CMH = {cmh:.4f}")
# (b) 4 セルの平均に交互作用つき回帰を当てはめる
Xc = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], float)  # 定数, 導入後, ラインA, 交互作用
yc = np.array([3.8, 3.3, 4.0, 2.5])
coef = np.linalg.solve(Xc, yc)
E.choice("s28p08b", f"${f(coef[3], 1)}$", "交互作用項の係数（差の差）")
# (c)
for v_, shown in [(-0.35, 0.7047), (-0.5852, 0.5570), (-0.1148, 0.8915), (0.1148, 1.1216), (0.5852, 1.7954)]:
    table("s28p08c", abs(math.exp(v_) - shown) < 5e-5, f"e^{v_}={shown}")
zq = stats.norm.ppf(0.975)
lo, hi = np.exp(-0.35 - zq * 0.12), np.exp(-0.35 + zq * 0.12)
E.choice("s28p08c", f"({f(lo, 3)},\\ {f(hi, 3)})", "正確な z で区間を作り指数変換")

# ---- 大問9
table("s28p09a", abs(stats.studentized_range.ppf(0.95, 3, 15) - 3.67) < 5e-3 and abs(stats.t.ppf(0.975, 15) - 2.131) < 5e-4, "q(3,15)=3.67, t(15)=2.131")
hsd = 3.67 * math.sqrt(121.5 / 15 / 6)
E.choice("s28p09a", f"${f(hsd, 2)}$", "HSD")
# シミュレーション：HSD を使うと3群が同じ母平均のとき全体の誤りが約5%
sims = rng.normal(0, math.sqrt(8.1), size=(100_000, 3, 6))
mm = sims.mean(2)
mse = ((sims - mm[:, :, None]) ** 2).sum((1, 2)) / 15
rngs = mm.max(1) - mm.min(1)
crit = stats.studentized_range.ppf(0.95, 3, 15) * np.sqrt(mse / 6)
E.check("s28p09a", abs(np.mean(rngs > crit) - 0.05) < 0.004, "シミュレーションで全体の第1種の誤り ≈ 5%")

# ---- 大問10
Nh = np.array([2000, 3000, 5000]); sh = np.array([4.0, 6.0, 5.0]); nh = np.array([20, 30, 50])
Vst = np.sum(Nh * sh ** 2) / Nh.sum() / 100  # 比例配分の式 Σ W s² / n
E.num("s28p10a", rnd(math.sqrt(Vst), 3), "比例配分の式 ΣW_h s_h²/n")
E.check("s28p10a", np.array_equal(nh, 100 * Nh / Nh.sum()), "抜取数が比例配分")

# ---- 大問11
lf = lambda x_: stats.norm(54, 4).logpdf(x_) - stats.norm(50, 2).logpdf(x_)
r1 = optimize.brentq(lf, 30, 50); r2 = optimize.brentq(lf, 50, 60)
E.choice("s28p11a", f"$x<{f(r1, 2)}$ または $x>{f(r2, 2)}$", f"scipy の対数密度の差の根 {r1:.4f}, {r2:.4f}")
E.check("s28p11a", lf(30) > 0 and lf(70) > 0 and lf(50) < 0, "両側で不良の密度が大きい")
table("s28p11a", abs(math.log(2) - 0.6931) < 5e-5, "ln2=0.6931")
Ci = np.array([0.0, 0.0]); pts_i = np.array([[-1.0, 0.5], [1.0, -0.5]])
assert np.allclose(pts_i.mean(0), Ci)
Gj = np.array([0.0, 3.0]); Gk = np.array([6.0, 1.0])
d2 = lambda a, b: float(np.sum((a - b) ** 2))
E.check("s28p11b", d2(Gk, Ci) == 37 and d2(Gk, Gj) == 40 and d2(Ci, Gj) == 9, "座標で与えた平方距離が実現できる（ユークリッド距離になる）")
newc = np.vstack([pts_i, Gj]).mean(0)
E.choice("s28p11b", f"${f(d2(Gk, newc), 2)}$", "座標から新しい重心までの平方距離")
lam = np.array([0.9, 0.8, 0.6]); z = np.array([1.2, 0.5, -0.4])
Sig = np.outer(lam, lam) + np.diag(1 - lam ** 2)
fr = lam @ np.linalg.solve(Sig, z)
E.choice("s28p11c", f"${f(fr, 3)}$", f"numpy で λ'Σ^-1 z = {fr:.5f}")
E.check("s28p11c", np.allclose(np.diag(Sig), 1), "Σ が相関行列（対角 1）")

# ---- 大問12
lam_, mu_ = 10.0, 60 / 4.5
rho = lam_ / mu_
k = np.arange(0, 2000)
pi = (1 - rho) * rho ** k
Lq = np.sum(np.maximum(k - 1, 0) * pi)
E.num("s28p12a", rnd(Lq, 2), f"定常分布から Σ(k−1)π_k = {Lq:.6f}")

E.done()
