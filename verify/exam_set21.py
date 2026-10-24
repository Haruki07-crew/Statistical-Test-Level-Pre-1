"""模擬試験 第21回（set21）の検算。generator/set21.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・実際のデータでの当てはめ）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats
from scipy.cluster.hierarchy import linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(21)
rng = np.random.default_rng(20261021)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [[float(v) for v in re.findall(r"-?\d+\.?\d*", str(c))] for c in E.item(iid)["choices"]]


def nearest(iid, value, pos=0):
    """選択肢の pos 番目の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    xs = [n[pos] for n in nums(iid)]
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1
# (a) 状態（理解・未理解）と選んだ選択肢を全列挙して条件付き確率を分数で
pC_U, pC = Fraction(0), Fraction(0)
for und, outcome in product([True, False], range(4)):  # outcome 0 を正解の選択肢とする
    pu = Fraction(3, 5) if und else Fraction(2, 5)
    po = (Fraction(95, 100) if outcome == 0 else Fraction(5, 100) / 3) if und else Fraction(1, 4)
    if outcome == 0:
        pC += pu * po
        if und:
            pC_U += pu * po
E.num("s21p01a", f(float(pC_U / pC), 3), "全列挙した同時確率の比")
sim_u = rng.random(1_000_000) < 0.6
sim_c = np.where(sim_u, rng.random(1_000_000) < 0.95, rng.integers(0, 4, 1_000_000) == 0)
E.check("s21p01a", abs(sim_u[sim_c].mean() - 0.8507) < 0.002, f"シミュレーション {sim_u[sim_c].mean():.4f}")
# (b) 事後分布で θ² を数値積分し、シミュレーションでも
prior_k = lambda t: t ** 2 * (1 - t)
lik = lambda t: t ** 6 * (1 - t) ** 2
Z = integrate.quad(lambda t: prior_k(t) * lik(t), 0, 1)[0]
Et2 = integrate.quad(lambda t: t * t * prior_k(t) * lik(t), 0, 1)[0] / Z
E.choice("s21p01b", f"${f(Et2, 3)}$", "事前×尤度を数値積分して E[θ²|データ]")
th = rng.beta(3, 2, 2_000_000)
ys = rng.binomial(8, th)
keep = th[ys == 6]
nxt = (rng.random(keep.size) < keep) & (rng.random(keep.size) < keep)
E.check("s21p01b", abs(nxt.mean() - 0.4945) < 0.005, f"事前から生成して y=6 で条件づけたシミュレーション {nxt.mean():.4f}")

# ---- 大問2
n, T = 40, 28
# (a) 不偏性を二項分布の全項の和で確かめる
for p in (0.2, 0.5, 0.7, 0.93):
    ev = sum(stats.binom.pmf(t, n, p) * t * (n - t) / (n * (n - 1)) for t in range(n + 1))
    E.check("s21p02a", abs(ev - p * (1 - p)) < 1e-12, f"p={p} で E[T(n−T)/(n(n−1))]=p(1−p)")
    evm = sum(stats.binom.pmf(t, n, p) * (t / n) * (1 - t / n) for t in range(n + 1))
    E.check("s21p02a", abs(evm - p * (1 - p)) > 1e-4, f"p={p} で最尤推定量 p̂(1−p̂) は偏る")
E.choice("s21p02a", f"${f(T * (n - T) / (n * (n - 1)), 4)}$", "推定値")
# (b) sympy で微分、シミュレーションでも
pp = sp.symbols("p", positive=True)
g = sp.log(pp / (1 - pp))
se = float(sp.sqrt(sp.diff(g, pp) ** 2 * pp * (1 - pp) / n).subs(pp, sp.Rational(7, 10)))
E.num("s21p02b", f(se, 3), "sympy で g'(p)²p(1−p)/n")
ph = rng.binomial(n, 0.7, 400_000) / n
ph = ph[(ph > 0) & (ph < 1)]
sd_sim = np.log(ph / (1 - ph)).std()
E.check("s21p02b", abs(sd_sim - se) < 0.03, f"シミュレーションの標準偏差 {sd_sim:.4f}（デルタ法 {se:.4f}）")
# (c) 2 次方程式を数値で解く
z = 1.96
coef = [1 + z ** 2 / n, -(2 * 0.7 + z ** 2 / n), 0.7 ** 2]
roots = sorted(np.roots(coef).real)
E.choice("s21p02c", f"({f(roots[0], 3)},\\ {f(roots[1], 3)})", "スコア検定の2次不等式の根")
E.check("s21p02c", abs(stats.norm.isf(0.025) - 1.96) < 0.001 and abs(stats.norm.isf(0.05) - 1.645) < 0.001, "z の値")
cover = []
for lo, hi in [roots]:
    pt = np.linspace(lo + 1e-6, hi - 1e-6, 5)
    cover = [abs(0.7 - q) / np.sqrt(q * (1 - q) / n) <= z + 1e-9 for q in pt]
E.check("s21p02c", all(cover), "区間内の p はスコア検定で棄却されない")

# ---- 大問3
for x, v in [(0.25, 0.7788), (0.5, 0.6065), (0.75, 0.4724), (1.5, 0.2231)]:
    E.check("s21p03a", abs(np.exp(-x) - v) < 5e-5, f"e^-{x} = {v}")
W = np.minimum.reduce([rng.exponential(20, 1_000_000), rng.exponential(30, 1_000_000), rng.exponential(60, 1_000_000)])
E.check("s21p03a", abs((W > 15).mean() - 0.2231) < 0.002, f"シミュレーション P(W>15)={(W > 15).mean():.4f}")
E.choice("s21p03a", f"${f(stats.expon(scale=20).sf(15) * stats.expon(scale=30).sf(15) * stats.expon(scale=60).sf(15), 3)}$", "scipy の生存関数の積")

# ---- 大問4
s_ = sp.symbols("s")
GY = sp.exp(120 * ((sp.Rational(19, 20) + sp.Rational(1, 20) * s_) - 1))
m1 = sp.diff(GY, s_).subs(s_, 1)
m2 = sp.diff(GY, s_, 2).subs(s_, 1)
var = sp.simplify(m2 + m1 - m1 ** 2)
E.check("s21p04a", sp.simplify(GY - sp.exp(6 * (s_ - 1))) == 0 and var == 6, f"sympy で合成 G_N(G_B(s)) と分散 {var}")
Ns = rng.poisson(120, 500_000)
Ys = rng.binomial(Ns, 0.05)
E.check("s21p04a", abs(Ys.var() - 6) < 0.06, f"シミュレーションの分散 {Ys.var():.3f}")
E.choice("s21p04a", "e^{6(s-1)}$、$V[Y]=6$", "正解の組")
y = np.array([2, 9, 4, 11, 5, 1, 8, 6, 10, 4])
D = ((y - y.mean()) ** 2).sum() / y.mean()
E.check("s21p04b", abs(stats.chi2.isf(0.05, 9) - 16.92) < 0.005 and abs(stats.chi2.isf(0.05, 10) - 18.31) < 0.005, "χ² の上側 5% 点")
E.check("s21p04b", stats.chi2.sf(D, 9) < 0.05 < stats.chi2.sf(D, 10), f"D={D:.3f} は自由度9で棄却、10で棄却しない")
E.choice("s21p04b", f"統計量は{f(D, 2)}。自由度9の上側5%点16.92を超えるので帰無仮説を棄却し、過分散", "正解の文")
# ポアソンからの D の分布をシミュレーションで（自由度 9 の近似の確認）
sims = rng.poisson(6, (200_000, 10))
Ds = ((sims - sims.mean(1, keepdims=True)) ** 2).sum(1) / np.maximum(sims.mean(1), 1e-9)
E.check("s21p04b", abs((Ds > 16.92).mean() - 0.05) < 0.01, f"ポアソンのもとで D>16.92 の割合 {(Ds > 16.92).mean():.4f}")

# ---- 大問5
res = stats.fisher_exact([[6, 1], [2, 5]], alternative="greater")
E.choice("s21p05a", f"${f(res.pvalue, 3)}$", "scipy fisher_exact(greater)")
E.check("s21p05a", abs(stats.fisher_exact([[6, 1], [2, 5]]).pvalue - 0.1026) < 0.0005, "両側 p 値（誤答）")
tabs = [np.array([[6, 1], [2, 5]]), np.array([[18, 12], [16, 24]])]
# MH 推定量を、共通オッズ比 ψ のもとでの MH 推定方程式 Σ(a d − ψ b c)/n = 0 を数値で解いて求める
from scipy.optimize import brentq  # noqa: E402
eq = lambda psi: sum((t[0, 0] * t[1, 1] - psi * t[0, 1] * t[1, 0]) / t.sum() for t in tabs)
E.num("s21p05b", f(brentq(eq, 0.1, 50), 2), "MH 推定方程式の根")

# ---- 大問6
# 部分 F：R² の形で計算（TSS は相殺されるので任意の値でよい）
TSS = 30.0
R1, R4 = 1 - 12.60 / TSS, 1 - 10.36 / TSS
Fv = ((R4 - R1) / 2) / ((1 - R4) / 36)
E.choice("s21p06a", f"$F={f(Fv, 2)}$、$3.26$を超えるので棄却する", "決定係数の差の形で F")
for (d1, d2, v) in [(2, 36, 3.26), (3, 36, 2.87), (2, 38, 3.24)]:
    E.check("s21p06a", abs(stats.f.isf(0.05, d1, d2) - v) < 0.005, f"F_0.05({d1},{d2})={v}")
# (b) 削除公式を実データで確かめ（当てはめ直し）、その後問題の値に適用
X = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
yy = X @ [1, 0.5, 0.2, 0.1] + rng.normal(size=40)
H = X @ np.linalg.solve(X.T @ X, X.T)
e = yy - H @ yy
i = 7
Xi, yi = np.delete(X, i, 0), np.delete(yy, i)
bi = np.linalg.lstsq(Xi, yi, rcond=None)[0]
s2_refit = ((yi - Xi @ bi) ** 2).sum() / (39 - 4)
s2_formula = ((e ** 2).sum() - e[i] ** 2 / (1 - H[i, i])) / (40 - 4 - 1)
E.check("s21p06b", abs(s2_refit - s2_formula) < 1e-10, "削除公式が当てはめ直しと一致（乱数データ）")
s2i = (10.36 - 1.25 ** 2 / 0.85) / 35
E.num("s21p06b", f(1.25 / np.sqrt(s2i * 0.85), 2), "外部スチューデント化残差")
# 外部スチューデント化残差と内部の関係式でも
r_int = 1.25 / np.sqrt(10.36 / 36 * 0.85)
E.check("s21p06b", abs(r_int * np.sqrt(35 / (36 - r_int ** 2)) - 1.25 / np.sqrt(s2i * 0.85)) < 1e-12, "t_i = r_i√((n−k−1)/(n−k−r_i²)) と一致")
# (c) Cp
rss = {"M1": (12.60, 2), "M2": (10.80, 3), "M3": (11.90, 3), "M4": (10.36, 4)}
s2 = 10.36 / 36
cp = {m: v / s2 - 40 + 2 * k for m, (v, k) in rss.items()}
best = min(cp, key=cp.get)
E.choice("s21p06c", f"{best}、$C_p={f(cp[best], 2)}$", f"Cp {cp}")

# ---- 大問7
Sig = np.array([[225, 0.4 * 150], [0.4 * 150, 100]])
w = np.array([1, 2])
E.choice("s21p07a", f"${f(np.sqrt(w @ Sig @ w), 2)}$", "w'Σw の平方根")
Xs = rng.multivariate_normal([0, 0], Sig, 1_000_000)
E.check("s21p07a", abs((Xs @ w).std() - 29.41) < 0.1, f"シミュレーション {(Xs @ w).std():.3f}")
ev = np.array([2.85, 0.92, 0.55, 0.40, 0.28])
# 相関行列として実現できるか（シューア・ホーンの定理：(1,…,1) が固有値に優越されるか）
cs = np.cumsum(np.sort(ev)[::-1])
E.check("s21p07b", all(cs[k] >= k + 1 - 1e-12 for k in range(5)) and abs(cs[-1] - 5) < 1e-12, "固有値の組をもつ相関行列が存在する")
# 実際にその固有値をもつ相関行列を作る（回転を数値で探す）
from scipy.optimize import minimize  # noqa: E402


def corr_from(theta):
    A = np.zeros((5, 5))
    A[np.triu_indices(5, 1)] = theta
    Qm = np.linalg.qr(np.eye(5) + (A - A.T))[0]
    return Qm @ np.diag(ev) @ Qm.T


obj = lambda th: ((np.diag(corr_from(th)) - 1) ** 2).sum()
best_r = min((minimize(obj, rng.normal(size=10), method="BFGS") for _ in range(20)), key=lambda r_: r_.fun)
Rm = corr_from(best_r.x)
E.check("s21p07b", best_r.fun < 1e-10 and np.allclose(np.sort(np.linalg.eigvalsh(Rm))[::-1], ev), f"固有値 {ev} の相関行列を構成（対角の誤差 {best_r.fun:.1e}）")
kaiser = int((ev >= 1).sum())
E.check("s21p07b", kaiser == 1 and f(100 * ev[0] / ev.sum(), 1) == "57.0", "カイザー基準で1つ、寄与率 57.0%")
E.check("s21p07b", ev[:2].sum() / 5 < 0.8 <= ev[:3].sum() / 5, "累積寄与率80%には3つ必要（選択肢3は誤り）")
E.choice("s21p07b", "第1主成分だけで、その寄与率は$57.0\\%$", "正解の文")

# ---- 大問8
P = np.array([[0.6, 0.4, 0], [0.2, 0.5, 0.3], [0, 0, 1]])
Q = P[:2, :2]
Nf = np.linalg.inv(np.eye(2) - Q)
E.num("s21p08a", f(Nf.sum(1)[0], 1), "基本行列 (I−Q)^{-1} の行和")
steps = []
for _ in range(100_000):
    st, k = 0, 0
    while st != 2:
        st = rng.choice(3, p=P[st])
        k += 1
    steps.append(k)
E.check("s21p08a", abs(np.mean(steps) - 7.5) < 0.08, f"シミュレーション {np.mean(steps):.3f}")

# ---- 大問9
rr, nn_ = 0.42, 27
# 回帰の傾きの t 値と一致すること：単回帰 y=a+bx の傾きの t は r√(n−2)/√(1−r²)
xx = rng.normal(size=nn_)
yv = rng.normal(size=nn_)
# 標本相関がちょうど 0.42 になるよう yv を作る
xc = (xx - xx.mean()) / xx.std()
u = yv - yv.mean()
u = u - (u @ xc) / (xc @ xc) * xc
u = u / u.std()
yv = rr * xc + np.sqrt(1 - rr ** 2) * u
lr = stats.linregress(xc, yv)
E.check("s21p09a", abs(lr.rvalue - rr) < 1e-12, "構成したデータの相関 0.42")
E.choice("s21p09a", f"${f(lr.slope / lr.stderr, 2)}$", "scipy linregress の傾きの t 値")
E.check("s21p09a", abs(stats.t.isf(0.025, 25) - 2.060) < 0.0005, "t_0.025(25)=2.060")
# (b) 2 段階最小二乗をデータで（群内のばらつきを加えても群平均が表と一致するように）
nz = 500
Z = np.repeat([1, 0], nz)
xg = np.where(Z == 1, 6.0, 4.0) + np.tile(rng.normal(size=nz) - 0, 2)
xg = xg - np.where(Z == 1, xg[Z == 1].mean() - 6.0, xg[Z == 0].mean() - 4.0)
yg = np.where(Z == 1, 68.0, 64.6) + np.tile(rng.normal(size=nz), 2) * 5
yg = yg - np.where(Z == 1, yg[Z == 1].mean() - 68.0, yg[Z == 0].mean() - 64.6)
Z1 = np.column_stack([np.ones(2 * nz), Z])
xhat = Z1 @ np.linalg.lstsq(Z1, xg, rcond=None)[0]
b2 = np.linalg.lstsq(np.column_stack([np.ones(2 * nz), xhat]), yg, rcond=None)[0][1]
E.choice("s21p09b", f"${f(b2, 2)}$", "2 段階最小二乗の第 2 段の傾き")

# ---- 大問10
E.check("s21p10a", abs(stats.studentized_range.ppf(0.95, 3, 21) - 3.565) < 0.001, "q_0.05(3,21)=3.565")
E.check("s21p10a", abs(stats.t.isf(0.025, 21) - 2.080) < 0.0005, "t_0.025(21)=2.080")
hsd = stats.studentized_range.ppf(0.95, 3, 21) * np.sqrt(48 / 8)
means = {"A": 62, "B": 70, "C": 73}
sig = [(a, b) for a in means for b in means if a < b and abs(means[a] - means[b]) > hsd]
E.check("s21p10a", sig == [("A", "C")], f"有意な対 {sig}")
E.choice("s21p10a", f"臨界差{f(hsd, 2)}。有意なのはCとAの対だけ", "scipy の q 点で臨界差")
# シミュレーション：帰無仮説のもとで最大の差が HSD を超える確率 ≈ 5%
sim = rng.normal(0, 1, (100_000, 3, 8))
mse_s = sim.var(axis=2, ddof=1).mean(1)
rngs = np.ptp(sim.mean(2), axis=1) / np.sqrt(mse_s / 8)
E.check("s21p10a", abs((rngs > 3.565).mean() - 0.05) < 0.004, f"範囲が q を超える割合 {(rngs > 3.565).mean():.4f}")
pv = {"ア": 0.038, "イ": 0.003, "ウ": 0.20, "エ": 0.012, "オ": 0.009}
# ホルム法を調整済み p 値で：p_adj(k) = max_{j≤k} (m−j+1) p_(j)
order = sorted(pv, key=pv.get)
adj, run = {}, 0
for j, kname in enumerate(order):
    run = max(run, (5 - j) * pv[kname])
    adj[kname] = min(1, run)
rej = sorted(k for k, v in adj.items() if v <= 0.05)
E.check("s21p10b", rej == sorted(["イ", "エ", "オ"]), f"ホルムの調整済み p 値 {adj}")
names = ["".join(re.findall("[アイウエオ]", c)) for c in E.item("s21p10b")["choices"]]
E.check("s21p10b", names.index("".join(sorted(rej, key="アイウエオ".index))) == E.item("s21p10b")["answer"], f"棄却集合 {rej} がちょうど一致する選択肢が正解")

# ---- 大問11
def pc(eta):
    return 1 / (1 + np.exp(-eta))


p2 = pc(1.5 - 0.8) - pc(-0.5 - 0.8)
nearest("s21p11a", p2)
for x, v in [(-2.3, 0.1003), (-1.5, 0.2231), (-0.7, 0.4966), (-0.3, 0.7408), (0.5, 1.6487), (1.3, 3.6693)]:
    E.check("s21p11a", abs(np.exp(x) - v) < 5e-5, f"e^{x}={v}")
E.choice("s21p11a", f"${f(p2, 3)}$", "正確な指数関数で")
pos = [0.82, 0.64, 0.55, 0.31]
neg = [0.70, 0.48, 0.40, 0.25, 0.18, 0.12]
U = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
E.num("s21p11b", f(U / 24, 3), "マン・ホイットニーの U / (n1 n2)")
# ROC 曲線の台形の面積
sc = np.array(pos + neg)
lab = np.array([1] * 4 + [0] * 6)
thr = np.r_[np.inf, np.sort(sc)[::-1]]
tpr = [(sc[lab == 1] >= t).mean() for t in thr]
fpr = [(sc[lab == 0] >= t).mean() for t in thr]
E.num("s21p11b", f(np.trapezoid(tpr, fpr), 3), "ROC 曲線の台形公式")

# ---- 大問12
N = 800
nmin = next(m for m in range(1, N + 1) if 1.96 * np.sqrt((N - m) / (N - 1) * 0.25 / m) <= 0.05)
E.num("s21p12a", nmin, "n を 1 から順に調べた最小値")
E.check("s21p12a", 1.96 * np.sqrt((N - nmin + 1) / (N - 1) * 0.25 / (nmin - 1)) > 0.05, "1 つ少ないと条件を満たさない")
est = np.array([61.2, 62.0, 60.8, 61.6, 61.4])
w2 = np.array([0.90, 0.85, 0.95, 0.88, 0.92])
Tt = w2.mean() + (1 + 1 / 5) * est.var(ddof=1)
E.choice("s21p12b", f"${f(np.sqrt(Tt), 3)}$", "numpy でルービンのルール")

# ---- 大問13
pts = np.array([[0, 0], [2, 0], [1, 3], [5, 1]])
D2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
tab = E.items["s21p13a"][0]["context"]
shown = [int(v) for v in re.findall(r"\|\s*(\d+)\s*(?=\|)", tab)]
E.check("s21p13a", sorted(shown) == sorted(D2.flatten().tolist()), "問題の距離行列は座標から作った平方ユークリッド距離（ユークリッド距離として実現できる）")
Lk = linkage(pts, method="centroid")
E.choice("s21p13a", f"${f(Lk[-1, 2] ** 2, 2)}$", f"scipy linkage(centroid) の最後の併合距離の2乗 {Lk[:, 2] ** 2}")
E.check("s21p13a", np.allclose(Lk[:, 2] ** 2, [4, 9, 16]), "併合距離の列 4, 9, 16")

E.done()
