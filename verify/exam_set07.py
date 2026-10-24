"""模擬試験 第7回（set07）の検算。generator/set07.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・反復計算・数値最適化）で正解を求め直し、問題文に示した表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(7)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table(iid, given, true, tol, what):
    E.check(iid, abs(given - true) <= tol, f"問題文の値 {what}={given}（正確な値 {true:.6f}）")


# ---- 大問1 家財保険
# (a) 2件の不正の有無を総当たり
p = [0.10, 0.20]
pr = {o: np.prod([p[i] if o[i] else 1 - p[i] for i in range(2)]) for o in product([0, 1], repeat=2)}
atl = sum(v for o, v in pr.items() if sum(o) >= 1)
E.num("s07p01a", rnd(pr[(1, 1)] / atl, 3), "4通りの総当たりで P(両方|少なくとも1件)")
sim = rng.random((1_000_000, 2)) < np.array(p)
sel = sim.any(1)
E.check("s07p01a", abs(sim[sel].all(1).mean() - 0.0714) < 0.002, f"シミュレーション {sim[sel].all(1).mean():.4f}")
# (b) sympy で積率母関数を微分／混合分布のモーメントを scipy で
t = sp.symbols("t")
M = sp.Rational(4, 5) + sp.Rational(1, 5) * (1 - 4 * t) ** -2
m1 = sp.diff(M, t).subs(t, 0)
m2 = sp.diff(M, t, 2).subs(t, 0)
E.choice("s07p01b", f"${f(float(m2 - m1 ** 2), 2)}$", "sympy で M''(0)-M'(0)^2")
g = stats.gamma(2, scale=4)
mix_var = 0.2 * g.moment(2) - (0.2 * g.mean()) ** 2
E.check("s07p01b", abs(mix_var - float(m2 - m1 ** 2)) < 1e-9, "混合（0 と Gamma(2, 尺度4)）のモーメントと一致")
# (c) 正確なパーセント点で保険料／支払総額のシミュレーションで超過確率 ≈ 1%
mu, sd = 1.6, np.sqrt(16.64)
P = mu + stats.norm.isf(0.01) * sd / np.sqrt(2500)
E.num("s07p01c", rnd(P, 2), "scipy norm.isf(0.01) で保険料")
K = rng.binomial(2500, 0.2, size=400_000)
T = rng.gamma(2 * K, 4)  # 支払のあった K 件の Gamma(2,4) の和
exceed = (T > 2500 * 1.79).mean()
E.check("s07p01c", 0.007 < exceed < 0.013, f"シミュレーションで P(総額>保険料収入)={exceed:.4f}（≈1%）")
table("s07p01c", 2.326, stats.norm.isf(0.01), 0.0005, "z_0.01")

# ---- 大問2 パレート
x, y, al, th = sp.symbols("x y alpha theta", positive=True)
fx = al * th ** al / x ** (al + 1)
G = 1 - sp.integrate(fx, (x, th * sp.exp(y), sp.oo))   # P(Y<=y)
gy = sp.simplify(sp.diff(G, y))
E.check("s07p02a", sp.simplify(gy - al * sp.exp(-al * y)) == 0, f"P(Y≤y) を微分した密度 {gy}")
E.choice("s07p02a", "\\alphae^{-\\alphay}", "正解の式を含む選択肢")
ys = np.log(stats.pareto(b=2.5).rvs(size=200_000, random_state=1))
E.check("s07p02a", abs(ys.mean() - 1 / 2.5) < 0.005, "シミュレーションで log X の平均 ≈ 1/α（指数分布）")
par = stats.pareto(b=3, scale=1)
v = par.ppf(0.99)
es = integrate.quad(lambda z: z * par.pdf(z), v, np.inf)[0] / par.sf(v)
E.num("s07p02b", rnd(es, 2), f"scipy の分位点 {v:.4f} と数値積分で E[X|X>v]")
table("s07p02b", 4.642, 100 ** (1 / 3), 0.0005, "100^(1/3)")

# ---- 大問3 運用成績
b0 = stats.binom(12, 0.5)
gam = (0.05 - b0.sf(9)) / b0.pmf(9)
E.num("s07p03a", rnd(gam, 3), "scipy binom で P(X>=10)+γP(X=9)=0.05 を解く")
E.check("s07p03a", abs(b0.sf(9) + gam * b0.pmf(9) - 0.05) < 1e-12, "有意水準がちょうど 0.05")
A = [-3.2, -1.5, 1.1, 2.8]
B = [0.4, 2.3, 5.2, 6.0]
C = [3.6, 4.5, 7.4, 8.9]
kw = stats.kruskal(A, B, C)
E.choice("s07p03b", f"H\\approx{f(kw.statistic, 2)}$で、自由度2の上側5%点5.991を超えるので帰無仮説を棄却する", "scipy.stats.kruskal")
E.check("s07p03b", kw.pvalue < 0.05, f"p 値 {kw.pvalue:.4f} < 0.05")
for q_, df_, val in [(0.95, 2, 5.991), (0.95, 3, 7.815), (0.975, 2, 7.378)]:
    table("s07p03b", val, stats.chi2.ppf(q_, df_), 0.0005, f"χ²({df_}) {q_}点")
c_ = sp.symbols("c")
mse = c_ ** 2 * sp.Rational(16, 36) + (1 - c_) ** 2 * sp.Rational(1, 4)
cs = sp.solve(sp.diff(mse, c_), c_)[0]
E.choice("s07p03c", f"$c={f(float(cs), 3)}$、そのときの MSE は ${f(float(mse.subs(c_, cs)), 4)}$", "sympy で MSE(c) を最小化")
Xb = rng.normal(0.5, 4, size=(200_000, 36)).mean(1)
E.check("s07p03c", abs(np.mean((0.36 * Xb - 0.5) ** 2) - 0.16) < 0.003, "シミュレーションで MSE(0.36) ≈ 0.16")

# ---- 大問4 ウィルソン区間：スコア検定を受容する p の範囲を数値的に求める
n, xx = 50, 2
ph = xx / n
h = lambda q: abs(ph - q) - 1.96 * np.sqrt(q * (1 - q) / n)
lo = optimize.brentq(h, 1e-9, ph)
hi = optimize.brentq(h, ph, 0.999)
E.choice("s07p04a", f"({f(lo, 3)},\\ {f(hi, 3)})", "スコア検定の受容域の端を brentq で求める")
cover = np.mean([(lambda k: optimize.brentq(lambda q: abs(k / n - q) - 1.96 * np.sqrt(q * (1 - q) / n), 1e-12, max(k / n, 1e-12)) <= 0.04 if k > 0 else True)(k) for k in rng.binomial(n, 0.04, size=2000)])
E.check("s07p04a", cover > 0.9, f"p=0.04 のとき下限が真値以下になる割合 {cover:.3f}")

# ---- 大問5 融資
res = stats.fisher_exact([[4, 2], [1, 7]], alternative="greater")
E.num("s07p05a", rnd(res.pvalue, 3), "scipy.stats.fisher_exact(alternative='greater')")
E.check("s07p05a", abs(stats.hypergeom(14, 5, 6).sf(3) - res.pvalue) < 1e-12, "超幾何分布の上側確率と一致")
# (b) 反復比例当てはめ（IPF）で [AB][AC] モデルを当てはめる
obs = np.array([[[12, 48], [3, 37]], [[10, 70], [2, 18]]], dtype=float)  # [業種][規模][延滞]
m = np.ones_like(obs)
for _ in range(50):
    m *= (obs.sum(2) / m.sum(2))[:, :, None]
    m *= (obs.sum(1) / m.sum(1))[:, None, :]
E.choice("s07p05b", f"${f(m[0, 0, 0], 2)}$", "IPF で当てはめた期待度数")
E.check("s07p05b", np.allclose(m.sum(2), obs.sum(2)) and np.allclose(m.sum(1), obs.sum(1)), "IPF の周辺度数が観測と一致")

# ---- 大問6 株価指数
xs = [-1.2, -0.4, 0.1, 0.5, 1.9]
ks = stats.kstest(xs, "norm")
norm_ = lambda s_: re.sub(r"\s+", "", str(s_))
hits = [i for i, c in enumerate(E.item("s07p06a")["choices"])
        if norm_(c) == norm_(f"$D\\approx{f(ks.statistic, 3)}$ で、0.563 を超えないので帰無仮説を棄却しない")]
E.check("s07p06a", hits == [E.item("s07p06a")["answer"]], f"scipy.stats.kstest の D={ks.statistic:.4f} で棄却しないとする選択肢 {hits}")
E.check("s07p06a", ks.statistic < stats.kstwo(5).ppf(0.95), "正確な限界値でも棄却しない")
for xv, Fv in zip(xs, [0.1151, 0.3446, 0.5398, 0.6915, 0.9713]):
    table("s07p06a", Fv, stats.norm.cdf(xv), 0.00006, f"Φ({xv})")
table("s07p06a", 0.563, stats.kstwo(5).ppf(0.95), 0.0006, "KS の n=5 の 5% 点")
paths = rng.normal(0.05, 1.2, size=(200_000, 25)).sum(1)
qlo, qhi = np.quantile(paths, [0.025, 0.975])
E.check("s07p06b", abs(qlo + 10.51) < 0.1 and abs(qhi - 13.01) < 0.1, f"シミュレーションの 2.5%・97.5% 点 {qlo:.2f}, {qhi:.2f}")
d = stats.norm(25 * 0.05, 1.2 * 5)
E.choice("s07p06b", f"({f(d.ppf(0.025), 2)},\\ {f(d.ppf(0.975), 2)})", "scipy norm の分位点")

# ---- 大問7 社債
Pm = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0, 0, 1]])
E.choice("s07p07a", f"${f(np.linalg.matrix_power(Pm, 2)[0, 2], 4)}$", "numpy で推移行列の2乗")
st = np.zeros(200_000, dtype=int)
for _ in range(2):
    u = rng.random(st.size)
    cum = Pm.cumsum(1)[st]
    st = (u[:, None] > cum).sum(1)
E.check("s07p07a", abs((st == 2).mean() - 0.046) < 0.002, f"シミュレーション {(st == 2).mean():.4f}")
L = stats.logistic.cdf
p2 = L(2.5 - 0.8 * 2.5) - L(0.5 - 0.8 * 2.5)
E.choice("s07p07b", f"${f(p2, 3)}$", "scipy logistic.cdf で累積確率の差")
table("s07p07b", 4.4817, np.exp(1.5), 0.00006, "e^1.5")
table("s07p07b", 0.6065, np.exp(-0.5), 0.00006, "e^-0.5")
prior = np.array([0.80, 0.15, 0.05])
lik = np.array([0.10, 0.40, 0.80])
post = prior * lik / (prior * lik).sum()
Lm = np.array([[-4, 3, 20], [-2, 0.5, 3], [0, 0, 0]])
el = Lm @ post
E.check("s07p07c", int(np.argmin(el)) == 1, f"事後期待損失 {el.round(4)} の最小はヘッジ")
E.choice("s07p07c", f"全額投資${f(el[0], 2)}$、ヘッジつき半額投資${f(el[1], 2)}$", "事後期待損失の値を含む選択肢")
elp = Lm @ prior
E.check("s07p07c", int(np.argmin(elp)) == 0, "事前確率では全額投資が最小（誤答の選択肢が誤りであること）")

# ---- 大問8 中古マンション
E.choice("s07p08a", f"約{f((np.exp(0.18) - 1) * 100, 1)}%高い", "numpy で e^0.18 - 1")
it = E.item("s07p08a")
E.check("s07p08a", "16.5" in it["choices"][3] and "でない物件の価格は約16.5%高い" in re.sub(r"\s+", "", it["choices"][3]),
        "選択肢4は「5分以内でない物件が高い」で向きが誤り")
for given, true, w in [(1.1972, np.exp(0.18), "e^0.18"), (0.8353, np.exp(-0.18), "e^-0.18"), (1.5136, 10 ** 0.18, "10^0.18")]:
    table("s07p08a", given, true, 0.00006, w)
xv = np.array([40, 50, 60, 70, 90.0])
yv = np.array([29, 36, 41, 49, 58.0])
X = np.column_stack([np.ones(5), xv])
beta = np.linalg.lstsq(X, yv, rcond=None)[0]
e = yv - X @ beta
XtXi = np.linalg.inv(X.T @ X)
V_hc0 = XtXi @ (X.T * e ** 2) @ X @ XtXi
E.num("s07p08b", rnd(np.sqrt(V_hc0[1, 1]), 4), "サンドイッチ行列 (X'X)^-1 X'diag(e²)X (X'X)^-1（丸めない残差）")
E.check("s07p08b", np.allclose(np.round(e, 3), [-0.757, 0.405, -0.432, 1.730, -0.946]), "表の残差が最小二乗残差の丸め")
n8, P8 = 60, 4
cp = {k: (n8 - P8) * sse / 1140 - n8 + 2 * pk for k, (sse, pk) in {"M1": (1480, 2), "M2": (1160, 3), "M3": (1140, 4)}.items()}
E.choice("s07p08c", f"M2が${f(cp['M2'], 2)}$", "C_p=(n-P)SSE_p/SSE_full - n + 2p で計算")
E.check("s07p08c", min(cp, key=cp.get) == "M2", f"C_p {cp} の最小は M2")

# ---- 大問9 リッジ：目的関数を数値最小化
R = np.array([[1, 0.8], [0.8, 1]])
gy_ = np.array([0.7, 0.6])
obj = lambda b: b @ R @ b - 2 * b @ gy_ + 0.3 * b @ b   # ||y-Xb||^2 + λ||b||^2（定数を除く）
bb = optimize.minimize(obj, np.zeros(2), tol=1e-12).x
E.choice("s07p09a", f"({f(bb[0], 3)},\\ {f(bb[1], 3)})", "scipy.optimize.minimize で罰則つき残差平方和を最小化")
E.check("s07p09a", np.allclose(bb, np.linalg.solve(R + 0.3 * np.eye(2), gy_), atol=1e-5), "正規方程式の解と一致")

# ---- 大問10 テューキー
q = stats.studentized_range.ppf(0.95, 3, 21)
table("s07p10a", 3.565, q, 0.0006, "q(3,21;0.05)")
table("s07p10a", 2.080, stats.t.isf(0.025, 21), 0.0005, "t_0.025(21)")
hsd = q * np.sqrt(25 / 8)
means = {"A": 63.5, "B": 56.5, "C": 58.0}
sig = {(a, b) for a in means for b in means if a < b and abs(means[a] - means[b]) > hsd}
E.check("s07p10a", sig == {("A", "B")}, f"正確な q で有意な組 {sig}")
pv = stats.studentized_range.sf(abs(63.5 - 58.0) / np.sqrt(25 / 8), 3, 21)
E.check("s07p10a", pv > 0.05, f"A と C の調整済み p 値 {pv:.3f} > 0.05")
E.check("s07p10a", E.item("s07p10a")["answer"] == 1 and f(hsd, 2) in E.item("s07p10a")["choices"][1],
        "正解の選択肢が基準値 6.30・A-B のみ")

# ---- 大問11 回帰推定量：標本の要約値をちょうど満たす 8 戸のデータを作って直線を当てはめる
Z = rng.normal(size=(8, 2))
Z -= Z.mean(0)
S = np.cov(Z.T)
target = np.array([[40.0, 46.0], [46.0, 60.0]])
Lz = np.linalg.cholesky(S)
Lt = np.linalg.cholesky(target)
D = Z @ np.linalg.inv(Lz).T @ Lt.T + np.array([24.0, 30.5])
E.check("s07p11a", np.allclose(np.cov(D.T), target) and np.allclose(D.mean(0), [24.0, 30.5]), "作ったデータが要約値と一致")
coef = np.polyfit(D[:, 0], D[:, 1], 1)
E.num("s07p11a", rnd(np.polyval(coef, 26.0), 2), "最小二乗直線の X̄=26 での予測値")

# ---- 大問12 主成分：元の共分散行列から固有分解
Sg = np.array([[16, 14, 10], [14, 20, 15], [10, 15, 18.0]])
w, V = np.linalg.eigh(Sg)
v1 = V[:, -1] * np.sign(V[0, -1])
E.check("s07p12a", np.allclose(np.round(v1, 3), [0.517, 0.644, 0.563]) and np.allclose(np.round(w[::-1], 2), [44.34, 6.96, 2.69]),
        "表の固有値・第1固有ベクトルが共分散行列の固有分解の丸め")
v2 = V[:, -2] * np.sign(V[0, -2])
E.check("s07p12a", np.allclose(np.round(v2, 3), [0.713, 0.040, -0.700], atol=0.0011), "第2固有ベクトルの丸め")
E.choice("s07p12a", f"${f(v1 @ np.array([6, 8, 4.0]), 2)}$", "丸めない固有ベクトルとの内積")

# ---- 大問13
# (a) 因子モデルの写像 (Λ, Ψ) → vech(ΛΛ'+Ψ) のヤコビ行列の階数から自由度を数える
def dof(pp, mm):
    def vech(theta):
        Lam = theta[:pp * mm].reshape(pp, mm)
        Sig = Lam @ Lam.T + np.diag(theta[pp * mm:])
        return Sig[np.triu_indices(pp)]
    th0 = rng.normal(size=pp * mm + pp)
    th0[pp * mm:] = np.abs(th0[pp * mm:]) + 0.5
    eps = 1e-6
    J = np.array([(vech(th0 + eps * np.eye(th0.size)[k]) - vech(th0 - eps * np.eye(th0.size)[k])) / (2 * eps) for k in range(th0.size)]).T
    return pp * (pp + 1) // 2 - np.linalg.matrix_rank(J, tol=1e-6)
dfs = {mm: dof(6, mm) for mm in (1, 2, 3)}
E.check("s07p13a", dfs == {1: 9, 2: 4, 3: 0}, f"ヤコビ行列の階数から数えた自由度 {dfs}")
E.choice("s07p13a", f"自由度{dfs[2]}、最大の因子数{max(mm for mm, v_ in dfs.items() if v_ > 0)}", "正の自由度をもつ最大の m")
# (b) パスモデルからデータを発生させ、R と E の相関（=総合効果）を求める
Nn = 400_000
Rr = rng.normal(size=Nn)
Hh = -0.5 * Rr + rng.normal(scale=np.sqrt(1 - 0.25), size=Nn)
var_e = 1 - (0.6 ** 2 + 0.2 ** 2 + 2 * 0.6 * -0.2 * -0.5)
Ee = 0.6 * Hh - 0.2 * Rr + rng.normal(scale=np.sqrt(var_e), size=Nn)
cor = np.corrcoef(Rr, Ee)[0, 1]
E.check("s07p13b", abs(cor + 0.5) < 0.005 and abs(Ee.std() - 1) < 0.005, f"シミュレーションの相関 {cor:.4f}（標準化された E）")
E.choice("s07p13b", f"${f(round(cor, 1), 2)}$", "シミュレーションの相関を丸めた値")

E.done()
