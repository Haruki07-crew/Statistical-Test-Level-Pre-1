"""模擬試験 第17回（set17）の検算。generator/set17.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・数値積分）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(17)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    it = E.item(iid)
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    v = nums(iid)
    k = int(np.argmin([abs(x - value) for x in v]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def distinct_rounded(iid, d):
    v = nums(iid)
    E.check(iid, len({f(x, d) for x in v}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {v}")


# ---- 大問1：原子確率から全部の事象の確率を総当たりで計算
atoms = {(1, 1, 1): 0.15, (1, 1, 0): 0.05, (1, 0, 1): 0.09, (0, 1, 1): 0.15,
         (1, 0, 0): 0.11, (0, 1, 0): 0.15, (0, 0, 1): 0.21, (0, 0, 0): 0.09}
P = lambda cond: sum(p for w, p in atoms.items() if cond(w))
pA, pB, pC = P(lambda w: w[0]), P(lambda w: w[1]), P(lambda w: w[2])
pair_ok = all(abs(P(lambda w, i=i, j=j: w[i] and w[j]) - [pA, pB, pC][i] * [pA, pB, pC][j]) < 1e-12
              for i, j in [(0, 1), (0, 2), (1, 2)])
mutual = abs(P(lambda w: all(w)) - pA * pB * pC) < 1e-12
union = P(lambda w: any(w))
E.check("s17p01a", abs(sum(atoms.values()) - 1) < 1e-12 and pair_ok and not mutual, "対ごとに独立・相互独立でない（選択肢1が正しい、選択肢0・2は誤り）")
E.check("s17p01a", abs(P(lambda w: all(w)) / P(lambda w: w[0] and w[1]) - pC) > 0.1, "P(C|A∩B)≠P(C)（選択肢3は誤り）")
E.check("s17p01a", abs(union - (1 - (1 - pA) * (1 - pB) * (1 - pC))) > 0.02, f"和事象 {union:.2f} は独立の公式の値と違う（選択肢4は誤り）")
E.check("s17p01a", "0.88" in E.item("s17p01a")["choices"][4] and "0.60" in E.item("s17p01a")["choices"][3], "選択肢の数値が表と整合")

# ---- 大問2：人工データで部分F・Cp・外部スチューデント化残差の公式を確かめ、与えた RSS から計算
n = 40
X = np.column_stack([np.ones(n), rng.normal(size=(n, 7))])
y = X @ rng.normal(size=8) + rng.normal(scale=3, size=n)


def rss(cols):
    b, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
    return float(((y - X[:, cols] @ b) ** 2).sum())


small, big = list(range(4)), list(range(6))
Fd = ((rss(small) - rss(big)) / 2) / (rss(big) / (n - 6))
# 尤度比の形の同値な式：F = ((RSS0/RSS1)-1)*(n-6)/2
Fd2 = (rss(small) / rss(big) - 1) * (n - 6) / 2
E.check("s17p02a", abs(Fd - Fd2) < 1e-9, "人工データで部分 F の式の同値性")
F_given = (Fr(520) / Fr(400) - 1) * Fr(34, 2)
E.num("s17p02a", rnd(float(F_given), 2), "分数で (RSS1/RSS3−1)(n−6)/2")
E.check("s17p02a", abs(stats.f.isf(0.05, 2, 34) - 3.28) < 0.005, "F_0.05(2,34)=3.28")
# Cp：分数で厳密に
mods = {"M1": (3, 520), "M2": (4, 430), "M3": (5, 400), "M4": (7, 368)}
s2 = Fr(368, 40 - 8)
cps = {k: Fr(r_) / s2 - 40 + 2 * (p + 1) for k, (p, r_) in mods.items()}
best = min(cps, key=cps.get)
E.check("s17p02b", best == "M3" and cps["M4"] == 8, f"Cp 最小は {best}、全変数モデルの Cp=k=8")
E.choice("s17p02b", f"M3（$C_p={f(float(cps['M3']), 2)}$）", "分数で計算した Cp")
# 人工データで「1点除いた RSS = RSS − e²/(1−h)」を確かめる
cols = list(range(6))
Xs = X[:, cols]
H = Xs @ np.linalg.inv(Xs.T @ Xs) @ Xs.T
e = y - H @ y
i = 7
rss_del = rss_wo = None
mask = np.arange(n) != i
b, *_ = np.linalg.lstsq(Xs[mask], y[mask], rcond=None)
rss_wo = float(((y[mask] - Xs[mask] @ b) ** 2).sum())
E.check("s17p02c", abs(rss_wo - (float((e ** 2).sum()) - e[i] ** 2 / (1 - H[i, i]))) < 1e-8, "人工データで削除公式を確認")
# 人工データで、外部スチューデント化残差が「平均シフトの t 値」と一致することも確認
D = np.column_stack([Xs, (np.arange(n) == i).astype(float)])
bD, *_ = np.linalg.lstsq(D, y, rcond=None)
resD = y - D @ bD
s2D = (resD ** 2).sum() / (n - 7)
seD = np.sqrt(s2D * np.linalg.inv(D.T @ D)[-1, -1])
t_shift = bD[-1] / seD
t_ext_sim = e[i] / np.sqrt(rss_wo / (n - 7) * (1 - H[i, i]))
E.check("s17p02c", abs(t_shift - t_ext_sim) < 1e-8, "外部スチューデント化残差＝平均シフトモデルの t 値")
t_ext = 8 / np.sqrt(float(Fr(400) - Fr(64) / Fr(4, 5)) / 33 * 0.8)
E.num("s17p02c", rnd(t_ext, 2), "削除公式から直接")

# ---- 大問3
x = np.array([0, 2, 1, 1, 3, 0, 1, 2, 1, 1])
Dst = x.var(ddof=1) * (len(x) - 1) / x.mean()
c9, c9lo, c10 = stats.chi2.isf(0.05, 9), stats.chi2.isf(0.95, 9), stats.chi2.isf(0.05, 10)
E.check("s17p03a", [f(c9, 2), f(c9lo, 2), f(c10, 2)] == ["16.92", "3.33", "18.31"], "χ² の点（scipy）")
E.choice("s17p03a", f"$D={f(Dst, 2)}$で、$\\chi^2_{{0.05}}(9)=16.92$", "numpy で D を計算")
E.check("s17p03a", Dst < c9 and stats.chi2.sf(Dst, 9) > 0.05, f"上側 p 値 {stats.chi2.sf(Dst, 9):.3f} > 0.05")
# シミュレーション：ポアソン(1.2) 10 個の D の上側 5% 点 ≈ 16.9（χ²(9) 近似の確認）
sim = rng.poisson(1.2, size=(200_000, 10))
mm = sim.mean(1)
ok = mm > 0
Dsim = ((sim[ok] - mm[ok, None]) ** 2).sum(1) / mm[ok]
E.check("s17p03a", np.mean(Dsim >= Dst) > 0.3, f"シミュレーションでも D≥{Dst:.2f} の確率 {np.mean(Dsim >= Dst):.3f} は大きい")
# (b) T=12 のもとで X1 の条件付き分布を、多項分布の総当たり（n=10 だと重いので二項分布の定義）で
t, nn = 12, 10
cond = stats.binom.pmf(0, t, 1 / nn)
E.choice("s17p03b", f"${f(cond, 3)}$", "binom.pmf(0, 12, 0.1)")
# 不偏性：λ をいくつか変えて E[(1-1/n)^T] = e^{-λ}（T~Poisson(nλ)）を級数で確かめる
for lam in (0.5, 1.2, 2.0):
    k = np.arange(0, 400)
    ev = float((stats.poisson.pmf(k, nn * lam) * (1 - 1 / nn) ** k).sum())
    E.check("s17p03b", abs(ev - np.exp(-lam)) < 1e-10, f"λ={lam} で不偏（{ev:.6f}）")
E.check("s17p03b", f(np.exp(-1.2), 4) == "0.3012", "e^{-1.2}=0.3012")
distinct_rounded("s17p03b", 3)
# (c) ガンマ分布の生存関数
g = stats.gamma(3, scale=1 / 1.2).sf(2)
E.num("s17p03c", rnd(g, 3), "scipy gamma(3, scale=1/1.2).sf(2)")
E.check("s17p03c", f(np.exp(-2.4), 4) == "0.0907", "e^{-2.4}=0.0907")
S3 = rng.exponential(1 / 1.2, size=(400_000, 3)).sum(1)
E.check("s17p03c", abs((S3 > 2).mean() - 0.570) < 0.004, f"シミュレーション {(S3 > 2).mean():.4f}")

# ---- 大問4
ph, nn, z = 0.8, 50, 1.96
h = lambda p: (ph - p) ** 2 - z ** 2 * p * (1 - p) / nn
up = optimize.brentq(h, ph, 0.9999)
lo = optimize.brentq(h, 1e-6, ph)
E.num("s17p04a", rnd(up, 3), "2 次不等式の根を brentq で")
# 被覆の性質の確認：区間の端点 p0 では z 検定の統計量がちょうど 1.96
E.check("s17p04a", abs(abs(ph - up) / np.sqrt(up * (1 - up) / nn) - 1.96) < 1e-9, "上限ではスコア統計量が 1.96")
# (b) 数値微分によるデルタ法 + シミュレーション
gfun = lambda p: p / (1 - p)
d = (gfun(ph + 1e-6) - gfun(ph - 1e-6)) / 2e-6
se = abs(d) * np.sqrt(ph * (1 - ph) / nn)
E.choice("s17p04b", f"${f(se, 3)}$", "数値微分によるデルタ法")
distinct_rounded("s17p04b", 3)

# ---- 大問5
res = stats.fisher_exact([[5, 1], [1, 5]], alternative="greater")
E.choice("s17p05a", f"${f(res.pvalue, 3)}$", "scipy fisher_exact(alternative='greater')")
distinct_rounded("s17p05a", 3)
# 総当たり：12 人から新レシピ 6 人を選ぶ全組み合わせで、「また注文したい」6 人のうち何人入るか
from itertools import combinations  # noqa: E402
cnt = sum(1 for c in combinations(range(12), 6) if sum(1 for i in c if i < 6) >= 5)
E.check("s17p05a", cnt == 37, f"総当たりで 5 人以上になる組み合わせ {cnt}/924")

# ---- 大問6
q = stats.studentized_range.ppf(0.95, 4, 20)
q24 = stats.studentized_range.ppf(0.95, 4, 24)
t20 = stats.t.isf(0.025, 20)
E.check("s17p06a", f(q, 2) == "3.96" and f(q24, 2) == "3.90" and f(t20, 3) == "2.086", f"q(4,20)={q:.4f}, q(4,24)={q24:.4f}, t={t20:.4f}")
means = {"A": 7.20, "B": 6.10, "C": 4.60, "D": 6.65}
hsd = 3.96 * np.sqrt(1.5 / 6)
sig = sorted(a + b for a, b in combinations("ABCD", 2) if abs(means[a] - means[b]) > hsd)
E.check("s17p06a", sig == ["AC", "CD"], f"HSD={hsd:.3f} で有意な組 {sig}")
E.choice("s17p06a", f"HSDは${f(hsd, 2)}$で、有意な組はAとC、CとDの2組", "表の q で HSD")
# シミュレーション：帰無仮説のもとで最大の差が HSD を超える確率 ≈ 5%
Ysim = rng.normal(0, np.sqrt(1.5), size=(100_000, 4, 6))
mu = Ysim.mean(2)
msesim = ((Ysim - mu[..., None]) ** 2).sum((1, 2)) / 20
rng_ = mu.max(1) - mu.min(1)
E.check("s17p06a", abs(np.mean(rng_ > 3.96 * np.sqrt(msesim / 6)) - 0.05) < 0.004, "テューキー法の FWER ≈ 5%（シミュレーション）")
# (b) p 値を scipy で求め直し、ホルム法とボンフェローニ法を独自に実装
pv = {a + b: 2 * stats.t.sf(abs(means[a] - means[b]) / np.sqrt(2 * 1.5 / 6), 20) for a, b in combinations("ABCD", 2)}
shown = {"AB": 0.1355, "AC": 0.0015, "AD": 0.4458, "BC": 0.0466, "BD": 0.4458, "CD": 0.0089}
E.check("s17p06b", all(f(pv[k], 4) == f(v, 4) for k, v in shown.items()), "表の p 値（scipy）")
srt = sorted(pv.items(), key=lambda kv: kv[1])
adj = np.maximum.accumulate([min(1, (6 - i) * p) for i, (_, p) in enumerate(srt)])  # ホルムの調整済み p 値
holm = sorted(k for (k, _), a in zip(srt, adj) if a <= 0.05)
bonf = sorted(k for k, p in pv.items() if 6 * p <= 0.05)
E.check("s17p06b", holm == ["AC", "CD"] and bonf == ["AC"], f"調整済み p 値でホルム {holm}、ボンフェローニ {bonf}")
E.check("s17p06b", E.item("s17p06b")["answer"] == 1, "正解は選択肢1（ホルム2組・ボンフェローニ1組）")

# ---- 大問7
F7 = 6.4 ** 2 / 4.0 ** 2
fa, fb, fc = stats.f.isf(0.025, 12, 15), stats.f.isf(0.05, 12, 15), stats.f.isf(0.025, 13, 16)
E.check("s17p07a", [f(fa, 2), f(fb, 2), f(fc, 2)] == ["2.96", "2.48", "2.85"], f"F の点 {fa:.4f} {fb:.4f} {fc:.4f}")
pv7 = 2 * stats.f.sf(F7, 12, 15)
E.check("s17p07a", pv7 > 0.05 and stats.f.sf(F7, 12, 15) < 0.05, f"両側 p 値 {pv7:.3f}>0.05、片側では <0.05")
E.choice("s17p07a", f"$F={f(F7, 2)}$は$F_{{0.025}}(12,15)=2.96$を超えないので、帰無仮説を棄却しない", "scipy で判定")

# ---- 大問8
th = sp.symbols("theta", real=True)
pth = sp.exp(th) / (1 + sp.exp(th))
Fp = 1 - (1 - pth) ** 2          # Be(1,2) の分布関数 F(p)=1-(1-p)^2 を θ で表す
dens = sp.simplify(sp.diff(Fp, th))
target = 2 * sp.exp(th) / (1 + sp.exp(th)) ** 3
E.check("s17p08a", sp.simplify(dens - target) == 0, "分布関数を θ で微分した密度と一致")
E.check("s17p08a", abs(integrate.quad(sp.lambdify(th, target), -50, 50)[0] - 1) < 1e-8, "密度の積分が 1")
smp = special.logit(rng.beta(1, 2, size=400_000))
hist, edges = np.histogram(smp, bins=40, range=(-4, 2), density=False)
mid = (edges[:-1] + edges[1:]) / 2
emp = hist / (len(smp) * (edges[1] - edges[0]))
E.check("s17p08a", np.max(np.abs(emp - sp.lambdify(th, target)(mid))) < 0.01, "シミュレーションのヒストグラムと一致")
E.choice("s17p08a", r"$\dfrac{2e^{\theta}}{(1+e^{\theta})^{3}}$", "正しい式の選択肢")
post = stats.beta(8, 15)
ep2 = integrate.quad(lambda p: p ** 2 * post.pdf(p), 0, 1)[0]
E.num("s17p08b", rnd(ep2, 3), "数値積分で E[p²|データ]")
lik = lambda p: stats.binom.pmf(7, 20, p) * 2 * (1 - p)
E.check("s17p08b", abs(integrate.quad(lambda p: p ** 2 * lik(p), 0, 1)[0] / integrate.quad(lik, 0, 1)[0] - ep2) < 1e-10, "事前×尤度から直接")

# ---- 大問9
Q = np.array([[0.7, 0.2], [0.3, 0.4]])
N = np.linalg.inv(np.eye(2) - Q)
E.num("s17p09a", rnd(N[0].sum(), 2), "基本行列の行和")
Pm = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0, 0, 1]])
st = np.zeros(200_000, dtype=int)
months = np.zeros(200_000)
alive = np.ones(200_000, bool)
cum = Pm.cumsum(1)
while alive.any():
    months[alive] += 1
    u = rng.random(alive.sum())
    st[alive] = (u[:, None] > cum[st[alive]]).sum(1)
    alive = st != 2
E.check("s17p09a", abs(months.mean() - 20 / 3) < 0.05, f"シミュレーション {months.mean():.3f}")

# ---- 大問10
L = special.expit
p2 = L(1.0 - 0.8) - L(-1.2 - 0.8)
nearest("s17p10a", p2)
E.choice("s17p10a", f"${f(p2, 3)}$", "expit で正確に")
tab = {-2.0: 0.1353, -1.2: 0.3012, -0.4: 0.6703, 0.2: 1.2214, 1.0: 2.7183, 1.8: 6.0496}
E.check("s17p10a", all(f(np.exp(k), 4) == f(v, 4) for k, v in tab.items()), "e^u の表")
distinct_rounded("s17p10a", 3)
rm = np.sqrt(max((72.4 - 24) / (24 * 400), 0))
E.choice("s17p10b", f"${f(rm, 3)}$", "RMSEA を直接")
# 非心度パラメータの推定 λ̂=χ²−df から RMSEA=√(λ̂/(df(N−1))) と同じ
E.check("s17p10b", abs(np.sqrt((72.4 - 24) / 24 / 400) - rm) < 1e-12, "非心度による表現と一致")
distinct_rounded("s17p10b", 3)

# ---- 大問11
S = np.array([[0.040, 0.030, -0.048], [0.030, 0.100, -0.040], [-0.048, -0.040, 0.250]])
E.check("s17p11a", np.all(np.linalg.eigvalsh(S) > 0), "分散共分散行列が正定値")
# 平方根行列で標本を作り、2 段階最小二乗法を実行（標本共分散がちょうど S になるように白色化）
m = 52
Zr = rng.normal(size=(m, 3))
Zr -= Zr.mean(0)
Zr = Zr @ np.linalg.inv(np.linalg.cholesky(np.cov(Zr.T)).T) @ np.linalg.cholesky(S).T
z_, lp, lq = Zr.T
E.check("s17p11a", np.allclose(np.cov(Zr.T), S), "作った標本の共分散が表と一致")
A1 = np.column_stack([np.ones(m), z_])
lp_hat = A1 @ np.linalg.lstsq(A1, lp, rcond=None)[0]
b2 = np.linalg.lstsq(np.column_stack([np.ones(m), lp_hat]), lq, rcond=None)[0][1]
E.choice("s17p11a", f"${f(b2, 2)}$", "2 段階最小二乗法を実行")
distinct_rounded("s17p11a", 2)

# ---- 大問12
R = np.array([[1, 0.72, 0.80], [0.72, 1, 0.85], [0.80, 0.85, 1]])
E.check("s17p12a", np.all(np.linalg.eigvalsh(R) > 0), "相関行列が正定値")
Pinv = np.linalg.inv(R)
pc = -Pinv[0, 1] / np.sqrt(Pinv[0, 0] * Pinv[1, 1])
E.num("s17p12a", rnd(pc, 2), "相関行列の逆行列から偏相関")
# ARMA(1,1)：ψ 重みから自己共分散を数値計算
phi, tht = 0.6, 0.3
psi = np.array([1.0] + [(phi + tht) * phi ** (k - 1) for k in range(1, 400)])
gam = lambda h: float((psi[:-h or None] * psi[h:]).sum()) if h else float((psi ** 2).sum())
rho2 = gam(2) / gam(0)
E.choice("s17p12b", f"${f(rho2, 3)}$", "ψ 重みから ρ(2)")
xs = np.zeros(400_000)
eps = rng.normal(size=400_000)
for k in range(1, len(xs)):
    xs[k] = phi * xs[k - 1] + eps[k] + tht * eps[k - 1]
xs = xs[1000:]
r2 = np.corrcoef(xs[:-2], xs[2:])[0, 1]
E.check("s17p12b", abs(r2 - rho2) < 0.01, f"シミュレーションの標本自己相関 {r2:.4f}")
distinct_rounded("s17p12b", 3)

# ---- 大問13
reg = 38.0 + 12.0 * (2.4 - 2.2)
E.choice("s17p13a", f"${f(reg, 1)}$", "回帰推定量")
# 回帰推定量は「標本の回帰直線の X̄ における予測値」と同じ：切片 a=ȳ−b x̄
a = 38.0 - 12.0 * 2.2
E.check("s17p13a", abs((a + 12.0 * 2.4) - reg) < 1e-12, "回帰直線の X̄ での値と一致")
distinct_rounded("s17p13a", 1)
Qj = np.array([6.2, 6.5, 6.0, 6.4, 6.4])
Uj = np.array([0.040, 0.045, 0.038, 0.042, 0.035])
Tt = Uj.mean() + (1 + 1 / 5) * Qj.var(ddof=1)
E.num("s17p13b", rnd(np.sqrt(Tt), 3), "numpy var(ddof=1) でルービンのルール")

# ---- 大問14
ev = np.array([2.9, 1.4, 0.8, 0.5, 0.3, 0.1])
# 固有値をもつ相関行列を実際に作れるか（ランダム直交行列で回し、対角を 1 にできることまでは要求しない）
cumr = np.cumsum(ev) / ev.sum()
E.check("s17p14a", abs(ev.sum() - 6.0) < 1e-12 and (ev >= 1).sum() == 2 and int(np.argmax(cumr >= 0.8)) + 1 == 3, f"カイザー 2 個、累積 {np.round(cumr, 3)} で 3 個")
E.check("s17p14a", E.item("s17p14a")["answer"] == 2, "正解は選択肢2")
Lam = np.array([[0.70, 0.30], [0.20, 0.60]])
c, s_ = 0.8, 0.6
T = np.array([[c, -s_], [s_, c]])
E.check("s17p14b", np.allclose(T @ T.T, np.eye(2)), "T は直交行列")
Ls = Lam @ T
E.check("s17p14b", np.allclose(Lam @ Lam.T, Ls @ Ls.T), "回転でモデル上の相関（ΛΛᵀ）が変わらない")
E.choice("s17p14b", f"負荷量$({f(Ls[0, 0], 2)},\\ {f(Ls[0, 1], 2)})$、共通性${f((Ls[0] ** 2).sum(), 2)}$", "行列積で回転")

E.done()
