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
# (a) sympy で指数分布の 4 次中心モーメントを積分し、scipy の正規分布関数で近似確率を計算
xx, th = sp.symbols("x theta", positive=True)
dens = sp.exp(-xx / th) / th
mu4 = sp.simplify(sp.integrate((xx - th) ** 4 * dens, (xx, 0, sp.oo)))
E.check("s08p04a", sp.simplify(mu4 - 9 * th ** 4) == 0, f"指数分布の μ4 = {mu4}（σ²=θ²）")
sd_rel = np.sqrt(float((mu4 - th ** 4) / th ** 4) / 200)
papprox = stats.norm.cdf(0.2 / sd_rel) - stats.norm.cdf(-0.2 / sd_rel)
nearest("s08p04a", papprox)
E.choice("s08p04a", f"${f(papprox, 3)}$", f"正規近似 P = {papprox:.5f}")
# 大きな n のシミュレーションで √n(S²−σ²)/σ² の分散が μ4/σ⁴−1=8 に近いこと（n=2000、5000 回）
Xe = rng.exponential(1.0, size=(5000, 2000))
v_s2 = (2000 * Xe.var(axis=1, ddof=1)).var() / 2000
E.check("s08p04a", abs(v_s2 - 8) < 0.6, f"シミュレーションの n·V[S²]/σ⁴ = {v_s2:.3f}（理論 8、許容 ±0.6）")
table_ok("s08p04a", [(stats.norm.cdf(z), v) for z, v in [(0.85, 0.8023), (0.94, 0.8264), (1.0, 0.8413), (2.0, 0.9772)]], 5e-5, "Φ の値")
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
# (a) scipy.stats.levene（平均を中心にした版）
lev = stats.levene([10, 12, 11, 13], [9, 15, 12, 12], [8, 16, 10, 14], center="mean")
E.choice("s08p05a", f"${f(lev.statistic, 2)}$", f"scipy levene(center='mean') = {lev.statistic:.4f}")
E.check("s08p05a", lev.pvalue > 0.05 and abs(stats.f.ppf(0.95, 2, 9) - 4.26) < 0.005, "5% で有意でない・F(2,9) の 5% 点")
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
# (b) 中心モーメントが問題文どおりになる 3 点分布を数値的に作り、そのモーメントから JB を計算
sol = optimize.root(lambda v: [v[3] + v[4] + v[5] - 1,
                               v[3] * v[0] + v[4] * v[1] + v[5] * v[2],
                               v[3] * v[0] ** 2 + v[4] * v[1] ** 2 + v[5] * v[2] ** 2 - 0.25,
                               v[3] * v[0] ** 3 + v[4] * v[1] ** 3 + v[5] * v[2] ** 3 - 0.075,
                               v[3] * v[0] ** 4 + v[4] * v[1] ** 4 + v[5] * v[2] ** 4 - 0.275,
                               v[1]], [-0.6, 0.0, 1.0, 0.3, 0.5, 0.2])
x3, p3 = sol.x[:3], sol.x[3:]
E.check("s08p06b", sol.success and np.all(p3 > 0), f"モーメントが (0.25, 0.075, 0.275) の分布が存在（値 {np.round(x3, 4)}, 確率 {np.round(p3, 4)}）")
m2_, m3_, m4_ = [(p3 * x3 ** k).sum() for k in (2, 3, 4)]
Sk, Ku = m3_ / m2_ ** 1.5, m4_ / m2_ ** 2
E.num("s08p06b", rnd(60 / 6 * (Sk ** 2 + (Ku - 3) ** 2 / 4), 2), "その分布のモーメントから JB")
dat = rng.normal(size=60)
jb_sc = stats.jarque_bera(dat).statistic
mm = [((dat - dat.mean()) ** k).mean() for k in (2, 3, 4)]
E.check("s08p06b", abs(jb_sc - 10 * ((mm[1] / mm[0] ** 1.5) ** 2 + (mm[2] / mm[0] ** 2 - 3) ** 2 / 4)) < 1e-9,
        "乱数データで scipy.stats.jarque_bera と公式（n で割るモーメント）が一致")
E.check("s08p06b", abs(stats.chi2.ppf(0.95, 2) - 5.99) < 0.005, "χ²(2) の 5% 点")
# (c) X'X の固有値が 9,4,1 になる X を作り、ハット行列のトレースを直接計算
Qm, _ = np.linalg.qr(rng.normal(size=(30, 3)))
Xr = Qm @ np.diag([3.0, 2.0, 1.0])
H = Xr @ np.linalg.inv(Xr.T @ Xr + 2 * np.eye(3)) @ Xr.T
E.choice("s08p06c", f"${f(np.trace(H), 2)}$", f"tr H(λ) = {np.trace(H):.5f}")

# ---- 大問7
# (a) 母集団を作り、枠内から抽出するシミュレーションで RMSE
cov_ = rng.normal(30, 10, 80_000)
exc_ = rng.normal(20, 10, 20_000)
cov_ = (cov_ - cov_.mean()) / cov_.std() * 10 + 30
exc_ = exc_ - exc_.mean() + 20
target = np.r_[cov_, exc_].mean()
ests = rng.choice(cov_, size=(20_000, 400), replace=True).mean(1)
rmse = np.sqrt(((ests - target) ** 2).mean())
E.check("s08p07a", abs(target - 28) < 1e-9, "全利用者の平均 28")
nearest("s08p07a", rmse)
E.choice("s08p07a", f"${f(np.sqrt(2 ** 2 + 100 / 400), 2)}$", "√(偏り² + 分散)")
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
# (b) 日のずれを入れた応答時間を作り、各分け方で 7 つの効果の推定値が日のずれで変わるかを最小二乗で確かめる
runs8 = ["(1)", "a", "b", "ab", "c", "ac", "bc", "abc"]
Xd = np.array([[1 if ch in r8 else -1 for ch in "abc"] for r8 in runs8], float)
cols = {"A": Xd[:, 0], "B": Xd[:, 1], "C": Xd[:, 2], "AB": Xd[:, 0] * Xd[:, 1], "AC": Xd[:, 0] * Xd[:, 2],
        "BC": Xd[:, 1] * Xd[:, 2], "ABC": Xd[:, 0] * Xd[:, 1] * Xd[:, 2]}
Mfull = np.column_stack([np.ones(8)] + list(cols.values()))
ybase = rng.normal(100, 10, 8)


def affected(day1):
    shift = np.array([15.0 if r8 in day1 else 0.0 for r8 in runs8])
    e0 = np.linalg.solve(Mfull, ybase)[1:]
    e1 = np.linalg.solve(Mfull, ybase + shift)[1:]
    return [k for k, a, b in zip(cols, e0, e1) if abs(a - b) > 1e-9]


cands = {0: (["(1)", "a", "b", "ab"], "C"), 1: (["(1)", "ab", "c", "abc"], "AB"), 2: (["(1)", "a", "b", "c"], "ABC"),
         3: (["(1)", "ab", "ac", "bc"], None), 4: (["(1)", "ab", "ac", "bc"], "ABC")}
good = []
for idx, (day1, claim) in cands.items():
    aff = affected(day1)
    ok_ = aff == ([claim] if claim else []) and all(e not in aff for e in ["A", "B", "C", "AB", "AC", "BC"])
    if ok_:
        good.append(idx)
    print(f"    分け方 {idx}: 日のずれで変わる効果 {aff}")
E.check("s08p08b", good == [E.item("s08p08b")["answer"]], f"目的に合い主張も正しい選択肢 {good}")
E.check("s08p08b", all(f"{{{','.join(r8 for r8 in d1)}}}" in re.sub(r"[\\\s]", "", E.item("s08p08b")["choices"][k])
                        for k, (d1, _) in cands.items()), "選択肢の 4 条件が検算の分け方と一致")

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
# (b) 季節差分 AR(1) を長い系列でシミュレーションし、回帰で予測式を確かめてから値を計算
nT = 200_000
eps = rng.normal(size=nT)
Yd = np.zeros(nT)
for i in range(1, nT):
    Yd[i] = 0.4 * Yd[i - 1] + eps[i]
Xl = np.zeros(nT)
Xl[:7] = [120, 135, 128, 126, 140, 95, 90.0]
for i in range(7, nT):
    Xl[i] = Xl[i - 7] + Yd[i]
yy_ = Xl[8:] - Xl[1:-7]
xx_ = Xl[7:-1] - Xl[:-8]
E.check("s08p10b", abs(np.polyfit(xx_, yy_, 1)[0] - 0.4) < 0.01, "シミュレーションで X_{t+1}−X_{t−6} の X_t−X_{t−7} への回帰の傾き ≈ 0.4")
w = [120, 135, 128, 126, 140, 95, 90, 133]
E.num("s08p10b", rnd(w[1] + 0.4 * (w[7] - w[0]), 1), "X̂_{T+1} = X_{T−6} + φ(X_T − X_{T−7})")

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
