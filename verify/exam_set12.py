"""模擬試験 第12回（set12）の検算。generator/set12.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算）で正解を求め直し、問題文に示した数表・関数値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(12)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value, what=""):
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']} {what}")


def distinct(iid, d):
    xs = nums(iid)
    E.check(iid, len({round(x, d) for x in xs}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {xs}")


# ---- 大問1 ガラス瓶（同時確率表を配列にして計算）
J = np.array([[0.50, 0.10], [0.15, 0.10], [0.05, 0.10]])
xv = np.array([0, 1, 2])
yv = np.array([0, 1])
E.check("s12p01a", abs(J.sum() - 1) < 1e-12 and (J >= 0).all(), "同時確率の和が 1")
cond = J[:, 1] / J[:, 1].sum()
E.choice("s12p01a", f"${f(cond @ xv, 2)}$", "条件付き分布のベクトルで E[X|Y=1]")
distinct("s12p01a", 2)
# (b) 同時分布から多数の標本を発生させて相関係数（加重共分散でも）
X, Y = np.meshgrid(xv, yv, indexing="ij")
w = J.ravel()
cov = np.cov(np.vstack([X.ravel(), Y.ravel()]), aweights=w, bias=True)
rho = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
E.num("s12p01b", rnd(rho, 3), "重みつき分散共分散行列の相関")
idx = rng.choice(6, size=1_000_000, p=w)
rs = np.corrcoef(X.ravel()[idx], Y.ravel()[idx])[0, 1]
E.check("s12p01b", abs(rs - rho) < 0.005, f"シミュレーションの相関 {rs:.4f}")

# ---- 大問2 反応時間
E.check("s12p02a", abs(math.exp(-3) - 0.0498) < 5e-5, "e^{-3}=0.0498")
E.num("s12p02a", rnd(stats.gamma(a=3, scale=0.5).sf(1.5), 3), "scipy gamma(形状3,尺度0.5).sf(1.5)")
T = rng.exponential(0.5, size=(1_000_000, 3)).sum(axis=1)
E.check("s12p02a", abs((T > 1.5).mean() - 0.4232) < 0.003, f"シミュレーション {(T > 1.5).mean():.4f}")
# (b) sympy で分布関数を積分して微分
u, x, y = sp.symbols("u x y", positive=True)
Fu = sp.integrate(sp.integrate(3 * sp.exp(-3 * x), (x, 0, u * y / (1 - u))) * sp.exp(-y), (y, 0, sp.oo), conds="none")
fu = sp.simplify(sp.diff(Fu, u))
E.check("s12p02b", sp.simplify(fu - 3 / (1 + 2 * u) ** 2) == 0, f"sympy の密度 {fu}")
E.choice("s12p02b", r"\dfrac{3}{(1+2u)^2}", "密度の式")
Xs = rng.exponential(1 / 3, 1_000_000)
Ys = rng.exponential(1.0, 1_000_000)
U = Xs / (Xs + Ys)
E.check("s12p02b", abs((U < 0.5).mean() - 0.75) < 0.003, f"P(U<0.5) シミュレーション {(U < 0.5).mean():.4f} ≈ 3u/(1+2u)=0.75")
for wrong in (r"\dfrac{3}{1+2u}", r"\dfrac{6}{(1+2u)^3}"):
    g = sp.sympify(wrong.replace(r"\dfrac{", "(").replace("}{", ")/(").replace("u}", "u)").replace("^3}", "**3)").replace("2u", "2*u"))
    E.check("s12p02b", sp.integrate(g, (u, 0, 1)) != 1, f"誤答 {wrong} は密度にならない")

# ---- 大問3 多項分布
n, p = 120, np.array([0.15, 0.60, 0.25])
Sig = n * (np.diag(p) - np.outer(p, p))
a = np.array([1, 0, -1])
vD = a @ Sig @ a
E.num("s12p03a", rnd(vD, 1), "多項分布の共分散行列から a'Σa")
M = rng.multinomial(n, p, size=400_000)
Dsim = M[:, 0] - M[:, 2]
E.check("s12p03a", abs(Dsim.var() - 46.8) < 0.5, f"シミュレーションの分散 {Dsim.var():.2f}")
E.choice("s12p03b", f"${f(vD / 15 ** 2, 3)}$", "V/k^2")
distinct("s12p03b", 3)
E.check("s12p03b", (np.abs(Dsim + 12) >= 15).mean() <= vD / 225, f"実際の確率 {(np.abs(Dsim + 12) >= 15).mean():.4f} は上界以下")
E.check("s12p03b", abs(stats.norm.cdf(2.19) - 0.9857) < 5e-5, "Φ(2.19)=0.9857（解説）")

# ---- 大問4 一様分布の上端
th, nn = sp.symbols("theta n", positive=True)
t = sp.symbols("t", positive=True)
dens = nn * t ** (nn - 1) / th ** nn
m1 = sp.simplify(sp.integrate(t * dens, (t, 0, th), conds="none"))
m2 = sp.simplify(sp.integrate(t ** 2 * dens, (t, 0, th), conds="none"))
vT2 = sp.simplify(((nn + 1) / nn) ** 2 * (m2 - m1 ** 2))
ratio = sp.simplify(vT2 / (4 * th ** 2 / 12 / nn)).subs(nn, 10)
E.choice("s12p04a", f"${f(float(ratio), 3)}$", "sympy で分散比")
Us = rng.uniform(0, 1, size=(400_000, 10))
T1 = 2 * Us.mean(axis=1)
T2 = 11 / 10 * Us.max(axis=1)
E.check("s12p04a", abs(T2.var() / T1.var() - 0.25) < 0.01, f"シミュレーションの分散比 {T2.var() / T1.var():.4f}")
distinct("s12p04a", 3)
# (b) ブートストラップをシミュレーション
E.check("s12p04b", f(0.9 ** 9, 4) == "0.3874" and f(0.9 ** 10, 4) == "0.3487", "0.9^9, 0.9^10 の値")
B = rng.integers(0, 10, size=(400_000, 10))
pb = (B == 9).any(axis=1).mean()
E.check("s12p04b", abs(pb - 0.6513) < 0.003, f"再標本の最大値が一致する割合 {pb:.4f}")
E.choice("s12p04b", f"${f(1 - stats.binom(10, 0.1).pmf(0), 3)}$", "1-二項(10,0.1)の P(0)")
distinct("s12p04b", 3)

# ---- 大問5 異物
xs = np.array([2, 0, 5, 1, 3, 8, 0, 2, 6, 3])
Dst = ((xs - xs.mean()) ** 2).sum() / xs.mean()
E.check("s12p05a", abs(stats.chi2.ppf(0.95, 9) - 16.92) < 0.005 and abs(stats.chi2.ppf(0.95, 10) - 18.31) < 0.005
        and abs(stats.chi2.ppf(0.05, 9) - 3.33) < 0.005, "χ² の点 16.92 / 18.31 / 3.33")
ch = E.item("s12p05a")["choices"]
E.check("s12p05a", E.item("s12p05a")["answer"] == 0 and f(Dst, 2) in ch[0] and "棄却し過分散" in ch[0]
        and stats.chi2.sf(Dst, 9) < 0.05, f"D={Dst:.3f}、p 値 {stats.chi2.sf(Dst, 9):.4f} < 0.05 で選択肢 0")
E.check("s12p05a", f(xs.var(ddof=1), 2) in ch[3] and f(xs.var(ddof=1) / xs.mean(), 2) in ch[2] and "平均" in ch[1] and "下側" in ch[4],
        "誤りの選択肢の値・記述")
# ポアソンの仮定のもとで D の分布をシミュレーションして χ²(9) 近似を確かめる
P = rng.poisson(3, size=(200_000, 10))
mP = P.mean(axis=1)
ok = mP > 0
Dp = ((P[ok] - mP[ok, None]) ** 2).sum(axis=1) / mP[ok]
E.check("s12p05a", (Dp >= Dst).mean() < 0.05, f"シミュレーションの p 値 {(Dp >= Dst).mean():.4f}")
# (b) 総当たり
for v, s_ in ((math.log(0.05), -2.9957), (math.log(0.025), -3.6889), (math.log(0.01), -4.6052), (math.log(0.99), -0.010050)):
    E.check("s12p05b", abs(v - s_) < 6e-5 if abs(s_) > 1 else abs(v - s_) < 6e-7, f"ln の値 {s_}")
nmin = next(k for k in range(1, 2000) if stats.beta.ppf(0.95, 1, k) <= 0.01)
E.choice("s12p05b", f"${nmin}$", "クロッパー・ピアソン上限 beta.ppf(0.95,1,n) ≤ 0.01 となる最小の n")

# ---- 大問6 測定法
d = np.array([12, 5, -3, 10, 8, 15, -2, 7, 11, 4])
A_ = rng.normal(100, 30, 10)
res = stats.ttest_rel(A_ + d, A_)
E.num("s12p06a", rnd(res.statistic, 2), "scipy ttest_rel")
E.num("s12p06a", rnd(stats.ttest_1samp(d, 0).statistic, 2), "scipy ttest_1samp")
E.choice("s12p06b", f"${f(stats.binomtest(8, 10, 0.5).pvalue, 3)}$", "scipy binomtest 両側")
distinct("s12p06b", 3)

# ---- 大問7 M/M/1 をシミュレーション（リンドレーの漸化式）
lam, mu = 6 / 60, 1 / 8  # 1分あたり
Nq = 2_000_000
inter = rng.exponential(1 / lam, Nq)
serv = rng.exponential(1 / mu, Nq)
Wq = np.empty(Nq)
Wq[0] = 0.0
for i in range(1, Nq):
    v = Wq[i - 1] + serv[i - 1] - inter[i]
    Wq[i] = v if v > 0 else 0.0
wsim = Wq[100_000:].mean()
E.check("s12p07a", abs(wsim - 32) < 1.5, f"シミュレーションの平均待ち時間 {wsim:.2f} 分")
wq_exact = lam / (mu * (mu - lam))
E.choice("s12p07a", f"{f(wq_exact, 1)} 分", "λ/(μ(μ-λ))")

# ---- 大問8 回帰
E.check("s12p08a", abs(stats.f.ppf(0.95, 2, 35) - 3.27) < 0.005, f"F(2,35) 5% 点 {stats.f.ppf(0.95, 2, 35):.4f}")
Fv = (Fr("139.2") - Fr("120.0")) / 2 / (Fr("120.0") / 35)
E.choice("s12p08a", f"${f(float(Fv), 2)}$", "分数で F")
distinct("s12p08a", 2)
# (b) 1点削除の残差平方和の式をランダムなデータで確かめ、与えた値から正確に計算
Xr = np.column_stack([np.ones(40), rng.normal(size=(40, 4))])
yr = Xr @ rng.normal(size=5) + rng.normal(size=40)
H = Xr @ np.linalg.inv(Xr.T @ Xr) @ Xr.T
e = yr - H @ yr
i = 7
Xd, yd = np.delete(Xr, i, 0), np.delete(yr, i, 0)
rss_d = ((yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0]) ** 2).sum()
E.check("s12p08b", abs(rss_d - ((e ** 2).sum() - e[i] ** 2 / (1 - H[i, i]))) < 1e-9, "1点削除の残差平方和の式")
# 削除した当てはめでの予測残差を使う形（定義）と、問題文の式による t_i が一致するか
beta_d = np.linalg.lstsq(Xd, yd, rcond=None)[0]
s_i = math.sqrt(rss_d / (40 - 5 - 1))
lev_d = Xr[i] @ np.linalg.inv(Xd.T @ Xd) @ Xr[i]
t_def = (yr[i] - Xr[i] @ beta_d) / (s_i * math.sqrt(1 + lev_d))
E.check("s12p08b", abs(t_def - e[i] / (s_i * math.sqrt(1 - H[i, i]))) < 1e-9, "削除予測残差による定義と e/(s_(i)√(1-h)) が一致")
t_ext = 4.5 / (math.sqrt((120 - 4.5 ** 2 / 0.75) / 34) * math.sqrt(0.75))
E.choice("s12p08b", f"${f(t_ext, 2)}$", "丸めなしで t_i")
distinct("s12p08b", 2)
# (c) 2段階最小二乗を積率行列で
Szz, Szx, Szy = 2.00, 0.60, 0.84
pi_hat = Szx / Szz
b2sls = (pi_hat * Szy) / (pi_hat ** 2 * Szz)
E.choice("s12p08c", f"${f(b2sls, 2)}$", "2段階最小二乗")
zz = rng.normal(0, math.sqrt(2), 1_000_000)
uu = rng.normal(size=1_000_000)
xx = 0.3 * zz + uu
yy = 1.4 * xx - 0.3 * uu + rng.normal(size=1_000_000)
biv = np.cov(zz, yy)[0, 1] / np.cov(zz, xx)[0, 1]
E.check("s12p08c", abs(biv - 1.4) < 0.02, f"内生性のあるシミュレーションで IV 推定 {biv:.3f}（真値 1.4）")

# ---- 大問9 順序ロジット
for v, s_ in ((math.exp(3), 20.086), (math.exp(0.5), 1.6487), (math.exp(-1), 0.3679), (math.exp(-3.5), 0.0302)):
    E.check("s12p09a", abs(v - s_) < 6e-4 if s_ > 10 else abs(v - s_) < 6e-5, f"指数の値 {s_}")
th1, th2, b, xx9 = -1.0, 1.5, 0.8, 2.5
pr = np.diff(np.concatenate([[0], special.expit(np.array([th1, th2]) - b * xx9), [1]]))
E.choice("s12p09a", f"${f(pr[1], 3)}$", "expit の差")
E.check("s12p09a", abs(pr.sum() - 1) < 1e-12 and (pr > 0).all(), "3 カテゴリの確率")
distinct("s12p09a", 3)

# ---- 大問10 変量効果：F 比から級内相関 (F-1)/(F-1+r)
Fr10 = (2.40 / 4) / (1.50 / 15)
E.num("s12p10a", rnd((Fr10 - 1) / (Fr10 - 1 + 4), 3), "(F-1)/(F-1+r)")
# シミュレーションで分散成分の推定量の不偏性を確認
sA, se_ = math.sqrt(0.125), math.sqrt(0.10)
ysim = rng.normal(0, sA, (100_000, 5, 1)) + rng.normal(0, se_, (100_000, 5, 4))
msa = 4 * ysim.mean(axis=2).var(axis=1, ddof=1)
mse = ysim.var(axis=2, ddof=1).mean(axis=1)
E.check("s12p10a", abs(((msa - mse) / 4).mean() - 0.125) < 0.003, f"(MSA-MSE)/r の平均 {((msa - mse) / 4).mean():.4f}")

# ---- 大問11
yv11 = np.array([12, 5, 20])
piv = np.array([0.30, 0.10, 0.50])
E.choice("s12p11a", f(np.sum(yv11 * (1 / piv)), 1), "重み 1/π との内積")
distinct("s12p11a", 1)
Q = np.array([812, 806, 818, 809, 815])
Uk = np.array([140, 150, 145, 138, 152])
E.check("s12p11b", "812 | 806 | 818 | 809 | 815" in E.item("s12p11b")["question"], "表の値が検算と同じ")
Tt = Uk.mean() + (1 + 1 / 5) * Q.var(ddof=1)
E.num("s12p11b", rnd(math.sqrt(Tt), 2), "numpy の var(ddof=1) でルービンのルール")

# ---- 大問12
S = np.array([[5.0, 2.0], [2.0, 2.0]])
val, vec = np.linalg.eigh(S)
v1 = vec[:, -1] * np.sign(vec[0, -1])
E.choice("s12p12a", f"${f(v1 @ np.array([1.5, -1.0]), 3)}$", "numpy eigh の固有ベクトル")
E.check("s12p12a", abs(math.sqrt(5) - 2.2361) < 5e-5 and np.allclose(val, [1, 6]), "√5 と固有値")
distinct("s12p12a", 3)
p_, k_ = 8, 2
# 自由度：共分散の要素数 − (負荷 + 独自分散 − 回転) を数値ヤコビアンの階数で確かめる
L0 = rng.normal(size=(p_, k_))
psi0 = rng.uniform(0.3, 1, p_)


def sig_vec(par):
    L = par[: p_ * k_].reshape(p_, k_)
    ps = par[p_ * k_:]
    Sg = L @ L.T + np.diag(ps)
    return Sg[np.triu_indices(p_)]


par0 = np.concatenate([L0.ravel(), psi0])
Jm = np.array([(sig_vec(par0 + 1e-6 * np.eye(len(par0))[j]) - sig_vec(par0 - 1e-6 * np.eye(len(par0))[j])) / 2e-6 for j in range(len(par0))]).T
rank = np.linalg.matrix_rank(Jm, tol=1e-6)
E.choice("s12p12b", str(p_ * (p_ + 1) // 2 - rank), f"共分散構造のヤコビアンの階数 {rank} から自由度")
# パス解析：構造方程式から相関行列
Bm = np.array([[0, 0, 0], [0.6, 0, 0], [-0.2, 0.5, 0]])  # 行が従属変数 (R,B,S)
IB = np.linalg.inv(np.eye(3) - Bm)
var_e = np.zeros(3)
var_e[0] = 1
var_e[1] = 1 - 0.6 ** 2
# S の誤差分散は V[S]=1 となるよう決める
bS = Bm[2]
Cpart = IB[:2, :2] @ np.diag(var_e[:2]) @ IB[:2, :2].T
var_e[2] = 1 - bS[:2] @ Cpart @ bS[:2]
C = IB @ np.diag(var_e) @ IB.T
E.check("s12p12c", np.allclose(np.diag(C), 1), "標準化されている")
E.choice("s12p12c", f"${f(C[1, 2], 2)}$", "構造方程式から計算した r_BS")
distinct("s12p12c", 2)

# ---- 大問13
tabs = [np.array([[4, 196], [2, 48]]), np.array([[6, 54], [30, 90]])]
# MH 推定量を「層ごとのオッズ比の重みつき平均（重み b_k c_k / n_k）」として計算
wts = np.array([tb[0, 1] * tb[1, 0] / tb.sum() for tb in tabs])
ors = np.array([tb[0, 0] * tb[1, 1] / (tb[0, 1] * tb[1, 0]) for tb in tabs])
E.choice("s12p13a", f"${f((wts * ors).sum() / wts.sum(), 3)}$", "層別オッズ比の重みつき平均")
distinct("s12p13a", 3)
# (b) 未観測の U（0/1）を含む構造モデルを具体的に作り、観測された確率を再現することを確かめてから、
#     介入 do(X=x) のもとでの P(Y=1) を U について直接平均して求める（フロントドアの式は使わない）。
a = {1: 0.9, 0: 0.1}                       # P(U=1 | X=x)（観測と矛盾しないように選んだ値）
px = {1: 0.4, 0: 0.6}
q = sum(px[x_] * a[x_] for x_ in (0, 1))   # P(U=1)
pm = {1: 0.10, 0: 0.50}                    # P(M=1 | X=x)：M は X だけで決まる
dy = {1: 0.10 / (a[1] - a[0]), 0: 0.02 / (a[1] - a[0])}
yu = {}
for m_, base in ((1, 0.20), (0, 0.04)):
    yu[(m_, 0)] = base - a[0] * dy[m_]      # P(Y=1 | M=m, U=0)
    yu[(m_, 1)] = yu[(m_, 0)] + dy[m_]      # P(Y=1 | M=m, U=1)
obs = {(m_, x_): (1 - a[x_]) * yu[(m_, 0)] + a[x_] * yu[(m_, 1)] for m_ in (0, 1) for x_ in (0, 1)}
target = {(1, 1): 0.30, (1, 0): 0.20, (0, 1): 0.06, (0, 0): 0.04}
E.check("s12p13b", all(abs(obs[k_] - target[k_]) < 1e-12 for k_ in target) and all(0 < v < 1 for v in yu.values()),
        "U つき構造モデルが観測の P(Y|M,X) と P(X) を再現する")
# モンテカルロで介入をシミュレーション
Nn = 2_000_000
Usim = rng.random(Nn) < q
res_do = []
for x_ in (0, 1):
    Msim = rng.random(Nn) < pm[x_]
    py_ = np.where(Msim, np.where(Usim, yu[(1, 1)], yu[(1, 0)]), np.where(Usim, yu[(0, 1)], yu[(0, 0)]))
    res_do.append((rng.random(Nn) < py_).mean())
exact = [sum((pm[x_] if m_ else 1 - pm[x_]) * ((1 - q) * yu[(m_, 0)] + q * yu[(m_, 1)]) for m_ in (0, 1)) for x_ in (0, 1)]
E.num("s12p13b", rnd(exact[1] - exact[0], 3), "構造モデルで U について平均した介入効果")
E.check("s12p13b", abs((res_do[1] - res_do[0]) - (exact[1] - exact[0])) < 0.002, f"介入のシミュレーション {res_do[1] - res_do[0]:.4f}")

E.done()
