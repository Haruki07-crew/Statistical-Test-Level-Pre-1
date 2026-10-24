"""模擬試験 第8回（set08）の検算。generator/set08.py とは別の方法（総当たり・scipy の分布関数・
sympy の積分・シミュレーション・数値最適化・別のアルゴリズム）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(8)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return str(abs(v) if v == 0 else v)


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def table_ok(iid, pairs, tol, what):
    bad = [(a, b) for a, b in pairs if abs(a - b) > tol]
    E.check(iid, not bad, f"{what}（ずれ {bad}）")


# ---- 大問1
# (a) 衝突しない確率の正確な積と、シミュレーション
exact = 1 - np.prod([1 - i / 1000 for i in range(20)])
nearest("s08p01a", exact)
keys = rng.integers(0, 1000, size=(200_000, 20))
srt = np.sort(keys, axis=1)
sim = (np.diff(srt, axis=1) == 0).any(axis=1).mean()
E.check("s08p01a", abs(sim - exact) < 0.005, f"シミュレーションの衝突確率 {sim:.4f}（正確 {exact:.4f}）")
E.choice("s08p01a", f"${f(1 - np.exp(-0.19), 3)}$", "指数近似 1−e^{−0.19}")
table_ok("s08p01a", [(np.exp(-0.02), 0.9802), (np.exp(-0.19), 0.8270), (np.exp(-0.2), 0.8187), (np.exp(-0.38), 0.6839)], 5e-5, "e^x の値")
# (b) 3 番目の順序統計量は Beta(3,3)
E.num("s08p01b", rnd(stats.beta(3, 3).cdf(0.4), 3), "scipy の Beta(3,3) の分布関数")
U = np.sort(rng.random((400_000, 5)), axis=1)[:, 2]
E.check("s08p01b", abs((U <= 0.4).mean() - 0.31744) < 0.003, f"シミュレーション {(U <= 0.4).mean():.4f}")

# ---- 大問2
# (a) sympy で同時密度を変数変換して周辺化
r_, s_ = sp.symbols("r s", positive=True)
l1, l2 = sp.Rational(1, 2), sp.Rational(1, 4)
fR = sp.simplify(sp.integrate(l1 * sp.exp(-l1 * r_ * s_) * l2 * sp.exp(-l2 * s_) * s_, (s_, 0, sp.oo)))
E.check("s08p02a", sp.simplify(fR - 2 / (2 * r_ + 1) ** 2) == 0, f"sympy で f_R = {fR}")
E.choice("s08p02a", "\\dfrac{2}{(2r+1)^2}", "正解の式を含む選択肢がただ1つ")
T1 = rng.exponential(2, 400_000)
T2 = rng.exponential(4, 400_000)
E.check("s08p02a", abs((T1 / T2 <= 1).mean() - 2 / 3) < 0.003, f"シミュレーションで P(R≤1) = {(T1 / T2 <= 1).mean():.4f} ≈ 2/3")
for k, wrong in [(0, 2 * r_ / (2 * r_ + 1)), (1, 2 / (r_ + 2) ** 2), (3, sp.Rational(1, 2) / (2 * r_ + 1)), (4, sp.Rational(1, 8) * sp.exp(-sp.Rational(3, 4) * r_))]:
    E.check("s08p02a", sp.simplify(wrong - fR) != 0, f"選択肢 {k} は導いた密度と異なる")
# (b) 推移率行列の零空間から定常分布
lm, mu, K = 3.0, 4.0, 3
Q = np.zeros((K + 1, K + 1))
for i in range(K + 1):
    if i < K:
        Q[i, i + 1] = lm
    if i > 0:
        Q[i, i - 1] = mu
    Q[i, i] = -Q[i].sum()
A = np.vstack([Q.T, np.ones(K + 1)])
pi = np.linalg.lstsq(A, np.r_[np.zeros(K + 1), 1], rcond=None)[0]
E.choice("s08p02b", f"${f(pi[K], 3)}$", "生成行列 Q の定常分布 πQ=0")
# 離散事象シミュレーション
t, n_sys, arr, lost = 0.0, 0, 0, 0
nxt_a, nxt_d = rng.exponential(1 / lm), np.inf
for _ in range(400_000):
    if nxt_a < nxt_d:
        t = nxt_a
        arr += 1
        if n_sys == K:
            lost += 1
        else:
            n_sys += 1
            if n_sys == 1:
                nxt_d = t + rng.exponential(1 / mu)
        nxt_a = t + rng.exponential(1 / lm)
    else:
        t = nxt_d
        n_sys -= 1
        nxt_d = t + rng.exponential(1 / mu) if n_sys > 0 else np.inf
E.check("s08p02b", abs(lost / arr - pi[K]) < 0.005, f"シミュレーションの廃棄率 {lost / arr:.4f}")
# (c) ポアソン確率の偶数項の和
pe = sum(stats.poisson(0.8).pmf(k) for k in range(0, 60, 2))
E.choice("s08p02c", f"${f(pe, 3)}$", "偶数 k の確率を直接足す")
table_ok("s08p02c", [(np.exp(-0.8), 0.4493), (np.exp(-1.6), 0.2019)], 5e-5, "e^x の値")

# ---- 大問3
Sig = np.array([[16.0, 6.0], [6.0, 9.0]])
res = optimize.minimize_scalar(lambda w: np.array([w, 1 - w]) @ Sig @ np.array([w, 1 - w]), bounds=(-2, 3), method="bounded",
                               options={"xatol": 1e-10})
E.num("s08p03a", rnd(res.fun, 2), f"数値最適化（w={res.x:.5f}）")
X = rng.multivariate_normal([0, 0], Sig, size=1_000_000)
vsim = (X @ np.array([res.x, 1 - res.x])).var()
E.check("s08p03a", abs(vsim - res.fun) < 0.05, f"シミュレーションの分散 {vsim:.4f}")

# ---- 大問4
# (a) scipy でベルヌーイの 3 次絶対中心モーメント
xv = np.array([0, 1])
pv = np.array([0.9, 0.1])
rho3 = (np.abs(xv - 0.1) ** 3 * pv).sum()
sd = stats.bernoulli(0.1).std()
E.num("s08p04a", rnd(0.4748 * rho3 / (sd ** 3 * np.sqrt(400)), 3), "定義から E|X−p|^3 と σ を計算")
Sn = stats.binom(400, 0.1)
ks = np.arange(0, 401)
gap = max(np.max(np.abs(Sn.cdf(ks) - stats.norm.cdf((ks - 40) / 6))), np.max(np.abs(Sn.cdf(ks - 1) - stats.norm.cdf((ks - 40) / 6))))
E.check("s08p04a", gap <= 0.0649, f"実際の最大のずれ {gap:.4f} は上界以下")
# (b) 対数尤度を数値で最大化
nll = lambda l: -(stats.poisson(6 * l).logpmf(18) + stats.expon(scale=1 / l).logpdf(np.full(5, 0.5)).sum())
lh = optimize.minimize_scalar(nll, bounds=(0.1, 10), method="bounded", options={"xatol": 1e-10}).x
E.choice("s08p04b", f"${f(lh, 2)}$", f"数値最大化 λ̂ = {lh:.5f}（間隔の合計 2.5 になる値で代用）")
# (c) 二項分布で信頼係数
cov = {k: 1 - 2 * stats.binom(10, 0.5).cdf(k - 1) for k in range(1, 6)}
best = max(k for k in cov if cov[k] >= 0.95)
E.check("s08p04c", best == 2 and E.item("s08p04c")["answer"] == 1, f"95% 以上で最も狭いのは k={best}（信頼係数 {cov}）")
Xs = rng.lognormal(0, 1, size=(200_000, 10))
Xs.sort(axis=1)
c2 = ((Xs[:, 1] <= 1) & (1 <= Xs[:, 8])).mean()
E.check("s08p04c", abs(c2 - cov[2]) < 0.003, f"歪んだ分布でのシミュレーションの被覆率 {c2:.4f}")

# ---- 大問5
# (a) scipy.stats.bartlett と同じ式を不偏分散から計算（正確な対数）
ni, s2 = 11, np.array([4.0, 9.0, 16.0])
Nn, kk = 33, 3
sp2 = ((ni - 1) * s2).sum() / (Nn - kk)
B = ((Nn - kk) * np.log(sp2) - (ni - 1) * np.log(s2).sum()) / (1 + (3 / (ni - 1) - 1 / (Nn - kk)) / (3 * (kk - 1)))
E.choice("s08p05a", f"${f(B, 2)}$", f"正確な対数で B = {B:.4f}")
g = [rng.normal(size=ni) for _ in range(3)]
g = [(x - x.mean()) / x.std(ddof=1) * np.sqrt(v) for x, v in zip(g, s2)]
E.check("s08p05a", abs(stats.bartlett(*g).statistic - B) < 1e-9, "分散を合わせたデータで scipy.stats.bartlett と一致")
E.check("s08p05a", stats.chi2.sf(B, 2) > 0.05 and abs(stats.chi2.ppf(0.95, 2) - 5.99) < 0.005, "5% で有意でない・χ²(2) の 5% 点")
table_ok("s08p05a", [(np.log(4), 1.3863), (np.log(9), 2.1972), (np.log(16), 2.7726), (np.log(29 / 3), 2.2687)], 5e-5, "対数の値")
# (b) scipy の permutation_test
old, new = np.array([12.4, 11.8, 13.1]), np.array([10.9, 11.5, 12.0])
pt = stats.permutation_test((old, new), lambda a, b: a.mean() - b.mean(), permutation_type="independent",
                            alternative="greater", n_resamples=np.inf)
E.choice("s08p05b", f"${f(pt.pvalue, 3)}$", f"scipy permutation_test p = {pt.pvalue}")

# ---- 大問6
# (a) 乱数データで単位変換を実際に行う
n = 50
X1 = rng.uniform(0.5, 3, n)
X2 = rng.integers(1, 20, n).astype(float)
yy = 1 + 0.8 * X1 + 0.05 * X2 + rng.normal(0, 0.3, n)


def ols(Xm, y):
    b = np.linalg.lstsq(Xm, y, rcond=None)[0]
    e = y - Xm @ b
    s2_ = e @ e / (len(y) - Xm.shape[1])
    se = np.sqrt(np.diag(s2_ * np.linalg.inv(Xm.T @ Xm)))
    return b, se, 1 - e @ e / ((y - y.mean()) ** 2).sum()


bA, seA, R2A = ols(np.column_stack([np.ones(n), X1, X2]), yy)
bB, seB, R2B = ols(np.column_stack([np.ones(n), 1000 * X1, X2]), yy)
okA = (np.isclose(bB[1], bA[1] / 1000) and np.isclose(seB[1], seA[1] / 1000) and np.isclose(bB[1] / seB[1], bA[1] / seA[1])
       and np.isclose(R2A, R2B) and np.isclose(bA[0], bB[0]) and np.isclose(bA[2], bB[2]))
E.check("s08p06a", okA and E.item("s08p06a")["answer"] == 1, "乱数データで単位を変えて当てはめ直すと選択肢 1 の通り")
# (b) 残差が問題文どおりになる y を作り、実際に 1 点除いて当てはめ直す
xx = np.arange(1, 7, dtype=float)
ee = np.array([0.1, 0.2, -0.1, -0.2, -0.6, 0.6])
yv = 2.0 + 0.5 * xx + ee
Xd = np.column_stack([np.ones(6), xx])
bfull = np.linalg.lstsq(Xd, yv, rcond=None)[0]
E.check("s08p06b", np.allclose(bfull, [2.0, 0.5]) and np.allclose(yv - Xd @ bfull, ee), "作った y で傾き 0.5・残差が問題文どおり")
bdel = np.linalg.lstsq(Xd[:5], yv[:5], rcond=None)[0]
E.num("s08p06b", rnd(bdel[1], 2), f"ページ 6 を除いて再当てはめ（傾き {bdel[1]:.5f}）")
# (c) X'X の固有値が 9,4,1 になる X を作り、ハット行列のトレースを直接計算
Qm, _ = np.linalg.qr(rng.normal(size=(30, 3)))
Xr = Qm @ np.diag([3.0, 2.0, 1.0])
H = Xr @ np.linalg.inv(Xr.T @ Xr + 2 * np.eye(3)) @ Xr.T
E.choice("s08p06c", f"${f(np.trace(H), 2)}$", f"tr H(λ) = {np.trace(H):.5f}")

# ---- 大問7
# (a) 起点 4 通りの標本平均を列挙
pop = np.array([10, 20, 30, 40] * 3, float)
means = np.array([pop[s::4].mean() for s in range(4)])
E.choice("s08p07a", f"${f(means.var(), 1)}$", "起点 4 通りの標本平均の分散")
srs = np.array([np.mean(c) for c in combinations(pop, 3)])
E.check("s08p07a", f(srs.var(), 1) in E.item("s08p07a")["choices"][0], f"単純無作為抽出の分散（全組合せ {srs.var():.3f}）は選択肢 0")
# (b) ポアソン件数の「1 件以上」をシミュレーション
cnt = rng.poisson(np.exp(-1.5), 1_000_000)
E.check("s08p07b", abs((cnt > 0).mean() - 0.2) < 0.002, f"件数 Po(e^-1.5) で 1 件以上の割合 {(cnt > 0).mean():.4f}")
E.choice("s08p07b", f"${f(1 - np.exp(-np.exp(-1.5)), 3)}$", "1−exp(−e^{−1.5})")
table_ok("s08p07b", [(np.exp(-2), 0.1353), (np.exp(-1.5), 0.2231), (np.exp(-0.1353), 0.8735), (np.exp(-0.2231), 0.8000),
                     (1 / (1 + np.exp(1.5)), 0.1824)], 5e-5, "関数値")

# ---- 大問8
# (a) 潜在結果を明示した母集団を作り ATT を直接計算
#     層ごとに Y(1)-Y(0) の平均が問題文の差になるように作る（交絡は Z だけ）
Np = 1_000_000
Z = rng.random(Np) < 0.4
pX = np.where(Z, 0.7 * 0.4 / 0.4, 0.3 * 0.4 / 0.6)   # P(X=1|Z) をベイズで
Xt = rng.random(Np) < pX
Y0 = np.where(Z, 1.8, 1.0) + rng.normal(0, 0.1, Np)
Y1 = Y0 + np.where(Z, -0.6, -0.2)
att = (Y1 - Y0)[Xt].mean()
E.check("s08p08a", abs(Z[Xt].mean() - 0.7) < 0.003, f"作った母集団で P(Z=高|X=1) = {Z[Xt].mean():.4f}")
E.num("s08p08a", rnd(att, 2), f"潜在結果から直接 ATT = {att:.4f}")
# (b) ±1 コードの回帰で交互作用係数
runs = {"1": 120, "a": 96, "b": 112, "ab": 80, "c": 110, "ac": 90, "bc": 104, "abc": 70}
rows, y9 = [], []
for key, val in runs.items():
    a_, b_, c_ = [1 if ch in key else -1 for ch in "abc"]
    rows.append([1, a_, b_, c_, a_ * b_, a_ * c_, b_ * c_, a_ * b_ * c_])
    y9.append(val)
coef = np.linalg.solve(np.array(rows, float), np.array(y9, float))
E.num("s08p08b", 2 * coef[4], "飽和モデルの回帰係数 ×2")

# ---- 大問9
# (a) 2 本の直線の距離を点の射影で直接計算
w = np.array([3.0, 4.0])
p1 = w * (1 - (-10)) / (w @ w)     # f=1 上の点
p2 = w * (-1 - (-10)) / (w @ w)    # f=-1 上の点
E.choice("s08p09a", f"マージンの幅 ${f(np.linalg.norm(p1 - p2), 2)}$、点 $(1,\\ 2)$ は攻撃側", "2 直線上の点の距離と f の符号")
# (b) ベクトルで直接計算
P = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
Qv = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
jac = ((P & Qv).sum()) / ((P | Qv).sum())
smc = (P == Qv).mean()
E.check("s08p09b", (P & Qv).sum() == 2 and (P & (1 - Qv)).sum() == 2 and ((1 - P) & Qv).sum() == 1, "作った 0/1 ベクトルが a=2,b=2,c=1,d=5")
E.choice("s08p09b", f"ジャカード係数 ${f(jac, 2)}$、単純一致係数 ${f(smc, 2)}$", "0/1 ベクトルから直接")

# ---- 大問10
# (a) 共分散選択モデルを尤度の数値最大化で当てはめる
S = np.array([[1, 0.6, 0.45], [0.6, 1, 0.5], [0.45, 0.5, 1]])


def negll(th):
    a, b, c, d, e = th   # 精度行列 K（(R,T) 成分は 0）
    Km = np.array([[a, d, 0], [d, b, e], [0, e, c]])
    ev = np.linalg.eigvalsh(Km)
    if ev.min() <= 0:
        return 1e9
    return -np.log(np.linalg.det(Km)) + np.trace(Km @ S)


opt = optimize.minimize(negll, [2, 2, 2, -0.5, -0.5], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
a, b, c, d, e = opt.x
Sfit = np.linalg.inv(np.array([[a, d, 0], [d, b, e], [0, e, c]]))
E.choice("s08p10a", f"${f(Sfit[0, 2] / np.sqrt(Sfit[0, 0] * Sfit[2, 2]), 2)}$", f"尤度の数値最大化で当てはめ相関 {Sfit[0, 2]:.5f}")
# (b) 状態空間（MA の誤差を順に再帰）で 1 期先予測
xs = [104, 110, 107]
fc = 100.0
for x in xs:
    eps = x - fc
    fc = x - 0.6 * eps
E.num("s08p10b", rnd(fc, 2), "X̂_{t+1} = X_t + θ ε_t の再帰")
# 長い系列のシミュレーションで予測式の最適性（誤差がホワイトノイズの分散）を確認
epsv = rng.normal(size=200_001)
Xsim = np.cumsum(epsv[1:] - 0.6 * epsv[:-1])
fcs = np.zeros_like(Xsim)
for i in range(1, len(Xsim)):
    fcs[i] = 0.4 * Xsim[i - 1] + 0.6 * fcs[i - 1]
E.check("s08p10b", abs(np.var(Xsim[1000:] - fcs[1000:]) - 1) < 0.02, "指数平滑の予測誤差の分散 ≈ 1（ε の分散）")

# ---- 大問11
pv_ = stats.poisson(2).sf(5)
E.choice("s08p11a", f"${f(pv_, 3)}$", f"scipy poisson.sf(5, 2) = {pv_:.5f}")
table_ok("s08p11a", [(np.exp(-2), 0.1353), (stats.norm.cdf(2.83), 0.9977)], 6e-5, "関数値")
# (b) 部分尤度を数値最大化
data = [(2, 1, 0), (3, 1, 1), (5, 0, 0), (6, 1, 1)]   # (時間, 事象, z)


def npl(beta):
    ll = 0.0
    for t_, ev, z in data:
        if ev:
            risk = [zz for tt, _, zz in data if tt >= t_]
            ll += beta * z - np.log(sum(np.exp(beta * zz) for zz in risk))
    return -ll


bh = optimize.minimize_scalar(npl, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
E.choice("s08p11b", f"${f(np.exp(bh), 3)}$", f"部分尤度の数値最大化 β̂ = {bh:.5f}")

# ---- 大問12：全初期値で周期を総当たり
for (a_, c_), idx in [((7, 3), 0), ((5, 4), 1), ((3, 5), 2), ((5, 3), 3), ((9, 6), 4)]:
    full = True
    for x0 in range(16):
        seen, x = set(), x0
        while x not in seen:
            seen.add(x)
            x = (a_ * x + c_) % 16
        if len(seen) != 16 or x != x0:
            full = False
    E.check("s08p12a", full == (idx == E.item("s08p12a")["answer"]), f"(a,c)=({a_},{c_}) は全初期値で最大周期: {full}")
E.done()
