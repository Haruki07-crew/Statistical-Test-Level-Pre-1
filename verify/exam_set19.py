"""模擬試験 第19回（set19）の検算。generator/set19.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(19)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value):
    it = E.item(iid)
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.6f}）が正解 {it['answer']}")


def table_ok(iid, given, true, tol, what):
    E.check(iid, abs(given - true) <= tol, f"{what}: 問題文 {given} / scipy {true:.5f}")


# ---- 大問1
# (a) 事象を列挙した同時分布から条件付き確率
joint = {("店頭", 1): 0.30 * 0.04, ("Web", 1): 0.50 * 0.10, ("提携", 1): 0.20 * 0.06}
pd = sum(joint.values())
E.choice("s19p01a", f"${f(joint[('Web', 1)] / pd, 3)}$", "同時確率の表から P(Web|延滞)")
sim = rng.choice(3, size=2_000_000, p=[0.3, 0.5, 0.2])
late = rng.random(sim.size) < np.array([0.04, 0.10, 0.06])[sim]
E.check("s19p01a", abs((sim[late] == 1).mean() - 0.6757) < 0.005, f"シミュレーション {(sim[late] == 1).mean():.4f}")
# (b) sympy で logit を微分して分散を作る／シミュレーション
p, n = sp.symbols("p n", positive=True)
g = sp.log(p / (1 - p))
var = sp.simplify(sp.diff(g, p) ** 2 * p * (1 - p) / n)
E.num("s19p01b", rnd(float(sp.sqrt(var.subs({p: sp.Rational(32, 400), n: 400}))), 3), "sympy で g'(p)^2 p(1-p)/n")
E.num("s19p01b", rnd(np.sqrt(1 / 32 + 1 / 368), 3), "√(1/x+1/(n−x))")
xs = rng.binomial(400, 0.08, size=400_000)
xs = xs[(xs > 0) & (xs < 400)]
lg = np.log(xs / (400 - xs))
E.check("s19p01b", abs(lg.std() - 0.1843) < 0.008, f"シミュレーションの標準偏差 {lg.std():.4f}（有限標本なので漸近値 0.1843 より少し大きい）")

# ---- 大問2
data = np.array([1, 0, 2, 0, 1, 3, 0, 1, 0, 2, 1, 1])
D = ((data - data.mean()) ** 2).sum() / data.mean()
table_ok("s19p02a", 19.68, stats.chi2.ppf(0.95, 11), 0.006, "χ²_{0.05}(11)")
table_ok("s19p02a", 21.03, stats.chi2.ppf(0.95, 12), 0.006, "χ²_{0.05}(12)")
table_ok("s19p02a", 4.575, stats.chi2.ppf(0.05, 11), 0.0006, "χ²_{0.95}(11)")
E.choice("s19p02a", f"$D={f(D, 1)}$で19.68より小さいので、帰無仮説を棄却しない", "np で D を計算し上側点と比較")
E.check("s19p02a", stats.chi2.sf(D, 11) > 0.05, f"p 値 {stats.chi2.sf(D, 11):.3f} > 0.05")
E.check("s19p02a", abs(np.var(data, ddof=1) / data.mean() - 0.91) < 0.005, "誤答 s²/x̄=0.91")
# (b) 定義どおり E[I(X1=0)|T=12] を多項分布の総当たりではなく二項分布の pmf で、さらにシミュレーションで不偏性
n2, T = 12, int(data.sum())
umvu = stats.binom.pmf(0, T, 1 / n2)
E.choice("s19p02b", f"${f(umvu, 3)}$", "scipy binom.pmf(0, T, 1/n)")
lam = 0.9
Xs = rng.poisson(lam, size=(300_000, n2))
est = (1 - 1 / n2) ** Xs.sum(axis=1)
E.check("s19p02b", abs(est.mean() - np.exp(-lam)) < 0.002, f"不偏性のシミュレーション {est.mean():.4f} vs {np.exp(-lam):.4f}")
cond = Xs[Xs.sum(axis=1) == 12]
E.check("s19p02b", abs((cond[:, 0] == 0).mean() - umvu) < 0.01, f"T=12 で条件付けた X1=0 の割合 {(cond[:, 0] == 0).mean():.4f}")
E.check("s19p02b", abs(np.exp(-1) - 0.3679) < 5e-5, "e^{-1}=0.3679")

# ---- 大問3
lam3, dl = 0.2, 0.05
dd = sp.symbols("d", positive=True)
t = sp.symbols("t", positive=True)
L3, DL = sp.Rational(1, 5), sp.Rational(1, 20)
FD = sp.integrate(L3 * sp.exp(-L3 * t), (t, -sp.log(dd) / DL, sp.oo))
fD = sp.simplify(sp.diff(FD, dd))
E.check("s19p03a", sp.simplify(fD - 4 * dd ** 3) == 0, f"sympy で分布関数を微分 {fD}")
Ts = rng.exponential(1 / lam3, size=1_000_000)
Dsim = np.exp(-dl * Ts)
E.check("s19p03a", stats.kstest(Dsim[:20000], stats.beta(4, 1).cdf).pvalue > 0.01, "シミュレーションが Beta(4,1) と整合（KS 検定）")
E.choice("s19p03a", "$4d^{3}$", "正解の式")
EDs = integrate.quad(lambda x: np.exp(-dl * x) * lam3 * np.exp(-lam3 * x), 0, np.inf)[0]
ED2 = integrate.quad(lambda x: np.exp(-2 * dl * x) * lam3 * np.exp(-lam3 * x), 0, np.inf)[0]
E.num("s19p03b", rnd(100 * np.sqrt(ED2 - EDs ** 2), 2), "数値積分で E[D], E[D²]")
E.num("s19p03b", rnd(100 * stats.beta(4, 1).std(), 2), "Beta(4,1) の標準偏差")
E.check("s19p03b", abs(100 * Dsim.std() - 16.33) < 0.05, f"シミュレーション {100 * Dsim.std():.3f}")

# ---- 大問4
# (a) 回帰推定量（直接計算）
E.num("s19p04a", rnd(28.0 - 0.55 * (42.0 - 40.0), 2), "ȳ − b(x̄ − X̄)")
# (b) 相関行列の逆行列から偏相関（generator の公式とは別の方法）
R = np.array([[1, 0.42, 0.70], [0.42, 1, 0.50], [0.70, 0.50, 1]])
Pm = np.linalg.inv(R)
pc = -Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1])
E.num("s19p04b", rnd(pc, 3), "精度行列から偏相関")
E.check("s19p04b", np.all(np.linalg.eigvalsh(R) > 0), "相関行列が正定値")
# (c) 共分散行列から 2SLS を行列で計算し、シミュレーションでも一致性を確かめる
sdC = 55 / (0.70 * 10)
S = np.array([[100, 55, 80], [55, sdC ** 2, 52], [80, 52, 100]])  # X, C, Z
E.check("s19p04c", np.all(np.linalg.eigvalsh(S) > 0), "X, C, Z の分散共分散行列が正定値（b の相関とも整合）")
b_iv = (S[2, 1] / S[2, 2]) / (S[2, 0] / S[2, 2])
E.choice("s19p04c", f"${f(b_iv, 2)}$", "誘導形 ÷ 第1段階")
m = 400_000
xstar = rng.normal(0, 1, m)
z = xstar + rng.normal(0, 1, m)
x = xstar + rng.normal(0, 1, m)
c = 0.65 * xstar + rng.normal(0, 1, m)
Zm = np.column_stack([np.ones(m), z])
xhat = Zm @ np.linalg.lstsq(Zm, x, rcond=None)[0]
b2 = np.linalg.lstsq(np.column_stack([np.ones(m), xhat]), c, rcond=None)[0][1]
bols = np.polyfit(x, c, 1)[0]
E.check("s19p04c", abs(b2 - 0.65) < 0.01 and bols < 0.4, f"シミュレーション：2SLS {b2:.3f}≈0.65、OLS {bols:.3f} は希薄化")

# ---- 大問5
n5 = 60
F = ((13.20 - 11.00) / 2) / (11.00 / 55)
E.choice("s19p05a", f"${f(F, 2)}$", "部分 F")
table_ok("s19p05a", 3.165, stats.f.ppf(0.95, 2, 55), 0.0006, "F_{0.05}(2,55)")
# 部分 F を R² ではなく尤度比と同じ形で：シミュレーションで帰無分布が F(2,55)
Xd = np.column_stack([np.ones(n5), rng.normal(size=(n5, 4))])
Fs = []
for _ in range(4000):
    y = Xd[:, :3] @ np.array([1, 0.5, -0.3]) + rng.normal(size=n5)
    r1 = np.sum((y - Xd[:, :3] @ np.linalg.lstsq(Xd[:, :3], y, rcond=None)[0]) ** 2)
    r2 = np.sum((y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]) ** 2)
    Fs.append(((r1 - r2) / 2) / (r2 / 55))
E.check("s19p05a", abs(np.mean(np.array(Fs) > 3.165) - 0.05) < 0.012, f"帰無仮説のもとでの棄却率 {np.mean(np.array(Fs) > 3.165):.3f}")
# (b) 外部スチューデント化残差：1 点を除いた誤差分散を公式 s_(i)^2 = ((n-p)s^2 - e^2/(1-h))/(n-p-1) から
s2 = 11.00 / 55
s2i = ((55) * s2 - 1.20 ** 2 / (1 - 0.25)) / 54
ti = 1.20 / np.sqrt(s2i * (1 - 0.25))
E.num("s19p05b", rnd(ti, 2), "1 点を除いた誤差分散から直接")
# 実データで公式を確認（任意の回帰で外部スチューデント化残差の 2 通りの計算が一致）
y = Xd @ np.array([1, 0.5, -0.3, 0.2, 0.1]) + rng.normal(size=n5)
H = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
e = y - H @ y
i = 7
keep = np.arange(n5) != i
bb = np.linalg.lstsq(Xd[keep], y[keep], rcond=None)[0]
s2_del = np.sum((y[keep] - Xd[keep] @ bb) ** 2) / (n5 - 1 - 5)
t_direct = e[i] / np.sqrt(s2_del * (1 - H[i, i]))
r_i = e[i] / np.sqrt(np.sum(e ** 2) / 55 * (1 - H[i, i]))
t_formula = r_i * np.sqrt(54 / (55 - r_i ** 2))
E.check("s19p05b", abs(t_direct - t_formula) < 1e-9, "問題文の t_i の式が 1 点除外の再当てはめと一致")
# (c) Cp
sig2 = 10.60 / 53
cps = [rss / sig2 - 60 + 2 * pp for rss, pp in ((13.20, 3), (11.00, 5), (10.60, 7))]
E.choice("s19p05c", f"M1: {f(cps[0], 1)}、M2: {f(cps[1], 1)}、M3: {f(cps[2], 1)}。M{int(np.argmin(cps)) + 1} を選ぶ", "Cp を直接計算")

# ---- 大問6
bad = np.array([42, 55, 61, 70])
good = np.array([58, 66, 72, 75, 81, 88])
yy = np.r_[np.ones(4), np.zeros(6)]
sc = np.r_[bad, good]
# しきい値を動かして ROC を作り台形公式で面積
ths = np.r_[-np.inf, np.sort(sc), np.inf]
tpr = [np.mean(bad <= th) for th in ths]
fpr = [np.mean(good <= th) for th in ths]
auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
E.num("s19p06a", rnd(auc, 3), "ROC 曲線の台形公式")
U = stats.mannwhitneyu(good, bad, alternative="greater").statistic
E.num("s19p06a", rnd(U / 24, 3), "scipy mannwhitneyu の U / 24")
# (b) scipy の expit で境界を数値的に解く
from scipy.optimize import brentq  # noqa: E402
from scipy.special import expit  # noqa: E402
xb = brentq(lambda x: expit(3.0 - 0.08 * x) - 0.2, 0, 100)
nearest("s19p06b", xb)
E.check("s19p06b", abs(np.log(4) - 1.3863) < 5e-5 and abs(np.log(5) - 1.6094) < 5e-5, "log4, log5 の値")
E.check("s19p06b", expit(3.0 - 0.08 * 54.8) > 0.2 > expit(3.0 - 0.08 * 54.9), "54.8 と 54.9 の間で 0.2 を下回る")

# ---- 大問7
P = np.array([[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0, 0, 1]])
P2 = np.linalg.matrix_power(P, 2)
E.choice("s19p07a", f"${f(P2[0, 2], 3)}$", "推移確率行列の 2 乗")
st = np.zeros(200_000, dtype=int)
for _ in range(2):
    u = rng.random(st.size)
    cum = P.cumsum(axis=1)[st]
    st = (u[:, None] > cum).sum(axis=1)
E.check("s19p07a", abs((st == 2).mean() - 0.046) < 0.002, f"シミュレーション {(st == 2).mean():.4f}")

# ---- 大問8
pr = stats.hypergeom(50, 3, 10).sf(0)
E.choice("s19p08a", f"${f(pr, 3)}$", "scipy hypergeom")
E.check("s19p08a", abs(1 - sp.binomial(47, 10) / sp.binomial(50, 10) - pr) < 1e-12, "二項係数の比で確認")
pv = {"A": 0.022, "B": 0.004, "C": 0.120, "D": 0.019, "E": 0.011}
try:
    from statsmodels.stats.multitest import multipletests
    rej = multipletests(list(pv.values()), alpha=0.05, method="holm")[0]
    nrej = int(rej.sum())
    how = "statsmodels multipletests(holm)"
except Exception:
    # 調整済み p 値 max_{j≤i}(m−j+1)p_(j) を作って 0.05 と比べる（別の手順）
    ps = np.sort(list(pv.values()))
    adj = np.maximum.accumulate((5 - np.arange(5)) * ps)
    nrej = int((adj <= 0.05).sum())
    how = "調整済み p 値"
E.choice("s19p08b", f"{nrej} 支店", how)

# ---- 大問9
table_ok("s19p09a", 5.629, stats.chi2.ppf(0.025, 14), 0.0006, "χ² 上側0.975(14)")
table_ok("s19p09a", 6.571, stats.chi2.ppf(0.05, 14), 0.0006, "χ² 上側0.95(14)")
table_ok("s19p09a", 23.68, stats.chi2.ppf(0.95, 14), 0.006, "χ² 上側0.05(14)")
table_ok("s19p09a", 26.12, stats.chi2.ppf(0.975, 14), 0.006, "χ² 上側0.025(14)")
table_ok("s19p09a", 6.262, stats.chi2.ppf(0.025, 15), 0.0006, "χ² 上側0.975(15)")
table_ok("s19p09a", 7.261, stats.chi2.ppf(0.05, 15), 0.0006, "χ² 上側0.95(15)")
table_ok("s19p09a", 25.00, stats.chi2.ppf(0.95, 15), 0.006, "χ² 上側0.05(15)")
table_ok("s19p09a", 27.49, stats.chi2.ppf(0.975, 15), 0.006, "χ² 上側0.025(15)")
lo = np.sqrt(14 * 0.64 / stats.chi2.ppf(0.975, 14))
hi = np.sqrt(14 * 0.64 / stats.chi2.ppf(0.025, 14))
E.choice("s19p09a", f"({f(lo, 3)},\\ {f(hi, 3)})", "scipy の χ² 分位点で区間")
# 被覆率のシミュレーション
smp = rng.normal(0, 1.0, size=(100_000, 15))
s2s = smp.var(axis=1, ddof=1)
cov = np.mean((np.sqrt(14 * s2s / 26.12) < 1) & (1 < np.sqrt(14 * s2s / 5.629)))
E.check("s19p09a", abs(cov - 0.95) < 0.004, f"被覆率 {cov:.4f}")
# (b)
table_ok("s19p09b", 3.565, stats.studentized_range.ppf(0.95, 3, 21), 0.0015, "q(3,21)")
table_ok("s19p09b", 4.165, stats.studentized_range.ppf(0.95, 3, 7), 0.0015, "q(3,7)")
table_ok("s19p09b", 2.080, stats.t.ppf(0.975, 21), 0.0006, "t_{0.025}(21)")
crit = stats.studentized_range.ppf(0.95, 3, 21) * np.sqrt(0.50 / 8)
means = {"駅前": 5.10, "商業施設": 4.30, "住宅地": 4.00}
sig = [(a, b) for a, b in combinations(means, 2) if abs(means[a] - means[b]) > crit]
E.check("s19p09b", sig == [("駅前", "住宅地")], f"scipy の q で有意な組 {sig}（基準 {crit:.4f}）")
E.choice("s19p09b", f"基準値 {f(crit, 2)} で、駅前と住宅地の 1 組だけが有意", "scipy の q で基準値")

# ---- 大問10
sat = [4.6, 4.2, 3.9, 3.8, 3.5, 3.3, 3.0]
gro = [5.1, 6.0, 3.2, 4.4, 1.5, 2.8, 2.0]
E.num("s19p10a", rnd(stats.spearmanr(sat, gro).statistic, 3), "scipy spearmanr")

# ---- 大問11
ev = np.array([2.70, 1.20, 0.95, 0.55, 0.40, 0.20])
cum = ev.cumsum() / ev.sum()
k_kaiser = int((ev >= 1).sum())
k80 = int(np.argmax(cum >= 0.80) + 1)
E.check("s19p11a", (k_kaiser, k80) == (2, 3) and abs(ev.sum() - 6) < 1e-12, f"カイザー {k_kaiser} 個、80% で {k80} 個")
E.check("s19p11a", E.item("s19p11a")["answer"] == 2 and "80.83" in E.item("s19p11a")["choices"][3], "正解の位置と誤答の値")
# 固有値に合う相関行列が存在すること（ランダムな直交行列で構成し、対角が 1 になるよう調整できるかではなく、
# 固有値の多数化条件：相関行列の固有値は対角 1 のベクトルを優越する。シューア・ホーンの定理から存在）
E.check("s19p11a", all(ev[:k].sum() >= k - 1e-12 for k in range(1, 7)), "シューア・ホーンの条件（固有値が (1,…,1) を優越）")
# (b) 母数を記号で数えて自由度を求め、RMSEA を計算
L_ = sp.Matrix(6, 2, lambda i, j: sp.Symbol(f"l{i}") if (i < 3) == (j == 0) else 0)
phi = sp.Symbol("phi")
Th = sp.diag(*[sp.Symbol(f"u{i}") for i in range(6)])
Sig = L_ * sp.Matrix([[1, phi], [phi, 1]]) * L_.T + Th
params = sorted(Sig.free_symbols, key=str)
dfm = 21 - len(params)
J = sp.Matrix([Sig[i, j] for i in range(6) for j in range(i, 6)]).jacobian(params)
vals = {s_: 0.7 for s_ in params}
vals[phi] = 0.4
rankJ = J.subs(vals).rank()
E.check("s19p11b", dfm == 8 and rankJ == len(params), f"母数 {len(params)} 個（識別可能：ヤコビアンの階数 {rankJ}）、自由度 {dfm}")
E.num("s19p11b", rnd(np.sqrt(max((20.0 - dfm) / (dfm * 300), 0)), 3), "数えた自由度で RMSEA")

# ---- 大問12
# 長いシミュレーションと、MA(∞) 表現の係数から自己相関を求める
phi_, th_ = 0.6, 0.3
psi = np.r_[1, (phi_ + th_) * phi_ ** np.arange(0, 400)]
g0 = np.sum(psi ** 2)
g1 = np.sum(psi[:-1] * psi[1:])
g2 = np.sum(psi[:-2] * psi[2:])
E.choice("s19p12a", f"\\rho(1)={f(g1 / g0, 3)},\\ \\rho(2)={f(g2 / g0, 3)}", "MA(∞) 表現の係数から自己共分散")
eps = rng.normal(size=400_001)
X = np.zeros(400_000)
for tt in range(1, X.size):
    X[tt] = phi_ * X[tt - 1] + eps[tt + 1] + th_ * eps[tt]
X = X[1000:]
r1 = np.corrcoef(X[:-1], X[1:])[0, 1]
r2 = np.corrcoef(X[:-2], X[2:])[0, 1]
E.check("s19p12a", abs(r1 - 0.732) < 0.01 and abs(r2 - 0.439) < 0.01, f"シミュレーション ρ1={r1:.3f}, ρ2={r2:.3f}")

# ---- 大問13
try:
    from statsmodels.stats.contingency_tables import StratifiedTable
    st_ = StratifiedTable([np.array([[80, 20], [60, 40]]), np.array([[20, 30], [15, 35]])])
    mh = st_.oddsratio_pooled
    how = "statsmodels StratifiedTable"
except Exception:
    tabs = [np.array([[80, 20], [60, 40]]), np.array([[20, 30], [15, 35]])]
    mh = sum(t_[0, 0] * t_[1, 1] / t_.sum() for t_ in tabs) / sum(t_[0, 1] * t_[1, 0] / t_.sum() for t_ in tabs)
    how = "表から直接"
E.choice("s19p13a", f"${f(mh, 2)}$", how)
# (b) 事後分布で θ² を数値積分、ベータ二項分布でも
post = stats.beta(5, 95)
pred = integrate.quad(lambda th: th ** 2 * post.pdf(th), 0, 1)[0]
E.choice("s19p13b", f"${f(pred, 5)}$", "事後分布で θ² を数値積分")
E.check("s19p13b", abs(stats.betabinom(2, 5, 95).pmf(2) - pred) < 1e-12, "scipy betabinom(2,5,95).pmf(2) と一致")
th_s = rng.beta(2, 48, size=3_000_000)
y_s = rng.binomial(50, th_s)
sel = th_s[y_s == 3]
E.check("s19p13b", abs(np.mean(sel ** 2) - pred) < 0.0001, f"事前→データの条件付けのシミュレーション {np.mean(sel ** 2):.5f}")

E.done()
