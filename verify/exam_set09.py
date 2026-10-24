"""模擬試験 第9回（set09）の検算。generator/set09.py とは別の方法（scipy の分布関数・シミュレーション・
総当たり・行列計算・当てはめ直し）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(9)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nearest(iid, value):
    """選択肢の最初の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def table_ok(iid, pairs, d=4, what="問題文の関数値"):
    ok = all(abs(rnd(true, d) - given) < 1e-12 for true, given in pairs)
    E.check(iid, ok, what + "が正しい丸め値")


# ---- 大問1 鳴き声の識別
# (a) 3近傍法（scipy の距離行列で計算）
from scipy.spatial.distance import cdist  # noqa: E402
F = np.array([[3900, .95], [4150, .90], [4300, .45], [3600, .55], [4050, .20], [3950, 1.00], [4560, .50]])
lab = np.array(list("BBACCBA"))
new = np.array([[4000, .50]])


def knn(scale):
    dist = cdist(F / scale, new / scale)[:, 0]
    nn = lab[np.argsort(dist)[:3]]
    vals, cnt = np.unique(nn, return_counts=True)
    assert cnt.max() >= 2
    return vals[np.argmax(cnt)]


E.choice("s09p01a", f"(i){knn(np.array([1, 1]))}、(ii){knn(np.array([400, 0.2]))}", "cdist による3近傍の多数決")
E.check("s09p01a", knn(np.array([1, 1])) == "B" and knn(np.array([400, 0.2])) == "A", "そのままでは B、標準化で A")
# (b) 野外の種構成で鳴き声を発生させ、判定 A のうち真に A の割合
cm = np.array([[84, 12, 4], [18, 165, 17], [9, 21, 270]])
prior = np.array([0.05, 0.35, 0.60])
L = cm / cm.sum(1, keepdims=True)
post = (prior * L[:, 0]) / (prior @ L[:, 0])
E.choice("s09p01b", f"${f(post[0], 3)}$", "行列計算によるベイズの定理")
sp_ = rng.choice(3, size=2_000_000, p=prior)
judgeA = rng.random(sp_.size) < L[sp_, 0]
E.check("s09p01b", abs(np.mean(sp_[judgeA] == 0) - 0.459) < 0.005, f"シミュレーション {np.mean(sp_[judgeA] == 0):.4f} ≈ 0.459")

# ---- 大問2 試合形式
# (a) シリーズの全結果を総当たり
from itertools import product  # noqa: E402
p6 = 0.0
for seq in product([1, 0], repeat=7):
    a = b = 0
    for g, w in enumerate(seq, 1):
        a += w
        b += 1 - w
        if a == 4 or b == 4:
            break
    if g == 6:
        p6 += 0.6 ** sum(seq) * 0.4 ** (7 - sum(seq))   # 残りの試合は周辺化される
E.choice("s09p02a", f"${f(p6, 3)}$", "7試合の全結果 2^7 通りの総当たり")
E.check("s09p02a", abs(p6 - (stats.nbinom.pmf(2, 4, 0.6) + stats.nbinom.pmf(2, 4, 0.4))) < 1e-12, "scipy nbinom（4勝までの負け数=2）と一致")
# (b) テニス（吸収的マルコフ連鎖を連立方程式で解く＋シミュレーション）
p = 0.6
states = [(i, j) for i in range(4) for j in range(4)] + ["AdS", "AdR"]
idx = {s: k for k, s in enumerate(states)}


def nxt(s, win):
    if s == "AdS":
        return "W" if win else (3, 3)
    if s == "AdR":
        return (3, 3) if win else "L"
    i, j = s
    if win:
        i += 1
    else:
        j += 1
    if i == 4 and j <= 2:
        return "W"
    if j == 4 and i <= 2:
        return "L"
    if i == 4 and j == 3:
        return "AdS"
    if j == 4 and i == 3:
        return "AdR"
    return (i, j)


A = np.eye(len(states))
bvec = np.zeros(len(states))
for s in states:
    for win, pr in ((True, p), (False, 1 - p)):
        t = nxt(s, win)
        if t == "W":
            bvec[idx[s]] += pr
        elif t != "L":
            A[idx[s], idx[t]] -= pr
sol = np.linalg.solve(A, bvec)
E.choice("s09p02b", f"${f(sol[idx[(1, 2)]], 3)}$", "全状態の吸収確率の連立方程式")
wins, M = 0, 100_000
for _ in range(M):
    s = (1, 2)
    while s not in ("W", "L"):
        s = nxt(s, rng.random() < p)
    wins += s == "W"
E.check("s09p02b", abs(wins / M - 0.515) < 0.006, f"シミュレーション {wins / M:.4f}")

# ---- 大問3 切断データ
table_ok("s09p03a", [(stats.norm.pdf(0.5), 0.3521), (stats.norm.cdf(0.5), 0.6915)])
tn = stats.truncnorm((20 - 18) / 4, np.inf, loc=18, scale=4)
E.choice("s09p03a", f"${f(tn.mean(), 1)}$", "scipy truncnorm の平均")
xs = rng.normal(18, 4, 2_000_000)
E.check("s09p03a", abs(xs[xs > 20].mean() - 22.56) < 0.02, f"シミュレーション {xs[xs > 20].mean():.3f}")

# ---- 大問4 気象
table_ok("s09p04a", [(stats.chi2.sf(x, df), g) for x, df, g in
                     [(2.5, 1, 0.1138), (3.125, 1, 0.0771), (6.25, 1, 0.0124),
                      (2.5, 2, 0.2865), (3.125, 2, 0.2096), (6.25, 2, 0.0439)]])
U, V = rng.normal(0, 4, (2, 2_000_000))
pm_ = np.mean(np.hypot(U, V) > 10)
E.check("s09p04a", abs(pm_ - 0.0439) < 0.001, f"シミュレーション P(W>10)={pm_:.4f}")
E.choice("s09p04a", f"${f(stats.rayleigh(scale=4).sf(10), 4)}$", "scipy のレイリー分布")
ex = stats.expon(scale=12)
table_ok("s09p04b", [(np.exp(-5), 0.0067)])
E.choice("s09p04b", f"(ア)${f(ex.mean() / 60, 3)}$、(イ)${f(ex.var() / 48 ** 2, 4)}$、(ウ)${f(ex.sf(60), 4)}$",
         "scipy expon の平均・分散・上側確率")
R = np.array([[1, -0.3, 0.5], [-0.3, 1, -0.6], [0.5, -0.6, 1]])
w_, v_ = np.linalg.eigh(R)
o = np.argsort(w_)[::-1]
w_, v_ = w_[o], v_[:, o] * np.sign(v_[0, o])
E.check("s09p04c", np.allclose(np.round(w_, 3), [1.944, 0.707, 0.349]) and
        np.allclose(np.round(v_, 3), [[.519, .787, .333], [-.570, .609, -.551], [.637, -.096, -.765]]),
        "問題文の固有値・固有ベクトルが相関行列 R の丸め値")
# 主成分得点を作り、湿度を第1・第2主成分に回帰した決定係数
Z = rng.multivariate_normal(np.zeros(3), R, size=400_000)
Y = Z @ v_
Xr = np.column_stack([np.ones(len(Y)), Y[:, :2]])
res = Z[:, 1] - Xr @ np.linalg.lstsq(Xr, Z[:, 1], rcond=None)[0]
r2 = 1 - res.var() / Z[:, 1].var()
E.check("s09p04c", abs(r2 - 0.894) < 0.003, f"シミュレーションで湿度を PC1・PC2 に回帰した R² {r2:.4f}")
E.choice("s09p04c", f"${f(w_[0] * v_[1, 0] ** 2 + w_[1] * v_[1, 1] ** 2, 3)}$", "正確な固有分解から")

# ---- 大問5 野鳥
Ns = np.arange(98, 2000)
lik = stats.hypergeom(Ns, 64, 45).pmf(11)
E.num("s09p05a", int(Ns[np.argmax(lik)]), "N=98..1999 の尤度を総当たり")
E.check("s09p05a", lik[Ns == 261][0] > lik[Ns == 262][0] and lik[Ns == 261][0] > lik[Ns == 260][0], "L(261) が両隣より大きい")
table_ok("s09p05b", [(np.exp(x), g) for x, g in [(1.3, 3.6693), (-1.3, 0.2725), (0.3, 1.3499), (-0.3, 0.7408)]])
cdf = lambda u: stats.logistic.cdf(u)
E.num("s09p05b", rnd(cdf(2.0 - 1.7) - cdf(0.4 - 1.7), 3), "scipy logistic の分布関数の差")
# (c) 産卵数と孵化を直接シミュレーション＋全確率
pN = {2: 0.2, 3: 0.5, 4: 0.3}
p1 = sum(v * stats.binom.pmf(1, k, 0.8) for k, v in pN.items())
E.choice("s09p05c", f"${f(p1, 3)}$", "N で条件づけた二項確率の和")
Nn = rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3], size=1_000_000)
Yh = rng.binomial(Nn, 0.8)
E.check("s09p05c", abs(np.mean(Yh == 1) - 0.1197) < 0.002, f"シミュレーション {np.mean(Yh == 1):.4f}")

# ---- 大問6 種子
xk = np.array([1.2, 2.5, 4.1, 6.8, 12.0])
table_ok("s09p06a", [(stats.expon(scale=5).cdf(x), g) for x, g in zip(xk, [0.2134, 0.3935, 0.5596, 0.7433, 0.9093])])
ks = stats.kstest(xk, stats.expon(scale=5).cdf)
E.check("s09p06a", abs(stats.kstwo.ppf(0.95, 5) - 0.563) < 0.001, f"n=5 の 5% 棄却限界 {stats.kstwo.ppf(0.95, 5):.4f}")
E.choice("s09p06a", f"$D={f(ks.statistic, 3)}$で、0.563より小さいので棄却しない", "scipy kstest")
nmin = next(n for n in range(1, 5000) if 1.96 * np.sqrt(0.16 / n) <= 0.03)
E.num("s09p06b", nmin, "n を 1 から増やして幅の半分を確認")

# ---- 大問7 観測所
fw = 1 - stats.binom.pmf(0, 20, 0.05)
fb = 1 - stats.binom.pmf(0, 20, 0.0025)
table_ok("s09p07a", [(0.95 ** 20, 0.3585), (0.9975 ** 20, 0.9512)])
E.choice("s09p07a", f"(ア)${f(fw, 3)}$、(イ)${f(fb, 3)}$", "二項分布 P(少なくとも1件)")
d = np.array([0.5, -3.0])
S = np.array([[0.8, -2.0], [-2.0, 25.0]])
T2 = 20 * d @ np.linalg.solve(S, d)
Fst = (20 - 2) / (2 * 19) * T2
table_ok("s09p07b", [(stats.f.ppf(0.95, 2, 18), 3.555), (stats.f.ppf(0.95, 2, 19), 3.522)], 3, "F の上側5%点")
table_ok("s09p07b", [(stats.f.ppf(0.95, 18, 2), 19.44)], 2, "F(18,2) の上側5%点")
E.choice("s09p07b", f"$T^2={f(T2, 2)}$、$F={f(Fst, 2)}$で、有意水準5%で棄却する", "numpy solve による T²")
E.check("s09p07b", stats.f.sf(Fst, 2, 18) < 0.05, f"p 値 {stats.f.sf(Fst, 2, 18):.4f}")
P = np.array([[0, 0], [6, 0], [1, 4], [5, 5]], float)
D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
J = np.eye(4) - np.ones((4, 4)) / 4
B = -0.5 * J @ D2 @ J
E.choice("s09p07c", f"${f(B[0, 3], 2)}$", "行列 -JD²J/2 による二重中心化")
it = E.item("s09p07c")
E.check("s09p07c", f"| 1 | 0 | {int(D2[0,1])} | {int(D2[0,2])} | {int(D2[0,3])} |" in it["question"], "問題文の距離の2乗が座標と一致")

# ---- 大問8 相関（相関行列から多変量正規データを発生させても確認）
Rm = np.array([[1, .5, .4], [.5, 1, -.3], [.4, -.3, 1]])   # (Y, X1, X2)
E.check("s09p08a", np.all(np.linalg.eigvalsh(Rm) > 0), "与えた相関行列が正定値")
a = np.array([0, 1, 1.0])
e1 = np.array([1, 0, 0.0])
rz = (e1 @ Rm @ a) / np.sqrt(a @ Rm @ a)
E.choice("s09p08a", f"${f(rz, 3)}$", "二次形式 a'Ra による相関")
Dm = rng.multivariate_normal(np.zeros(3), Rm, size=300_000)
E.check("s09p08a", abs(np.corrcoef(Dm[:, 0], Dm[:, 1] + Dm[:, 2])[0, 1] - 0.761) < 0.005, "シミュレーション")
bstd = np.linalg.solve(Rm[1:, 1:], Rm[1:, 0])
E.num("s09p08b", rnd(Rm[0, 1:] @ bstd, 3), "正規方程式を numpy で解いた R²")
Ds = (Dm - Dm.mean(0)) / Dm.std(0)
fit = np.linalg.lstsq(Ds[:, 1:], Ds[:, 0], rcond=None)[0]
E.check("s09p08b", abs(1 - np.var(Ds[:, 0] - Ds[:, 1:] @ fit) - 0.582) < 0.005, "シミュレーションの R²")
lam = 0.2
br = np.linalg.solve(Ds[:, 1:].T @ Ds[:, 1:] / len(Ds) + lam * np.eye(2), Ds[:, 1:].T @ Ds[:, 0] / len(Ds))
ex_r = np.linalg.solve(Rm[1:, 1:] + lam * np.eye(2), Rm[1:, 0])
E.choice("s09p08c", f"$({f(ex_r[0], 3)},\\{f(ex_r[1], 3)})$", "(R+λI)^{-1} r を numpy で")
E.check("s09p08c", np.allclose(br, ex_r, atol=0.01), f"シミュレーションデータのリッジ推定値 {br.round(3)}")

# ---- 大問9 河川
xv = np.array([2, 3, 4, 5, 6, 10.0])
e = np.array([0.4, -0.6, 0.5, -0.1, -0.375, 0.175])
y = 1 + 0.5 * xv + e
b_all = np.polyfit(xv, y, 1)
E.check("s09p09a", np.allclose(y - np.polyval(b_all, xv), e), "与えた残差が最小二乗の残差になっている")
b_6 = np.polyfit(xv[:5], y[:5], 1)
E.num("s09p09a", rnd(b_6[0] - b_all[0], 3), "6回目を除いて当てはめ直した傾きの差")
k = np.array([3, 4, 5])
ll = np.array([-20.0, -18.2, -16.5])
aic = -2 * ll + 2 * k
aicc = aic + 2 * k * (k + 1) / (12 - k - 1)
E.check("s09p09b", np.argmin(aic) == 2 and np.argmin(aicc) == 0, f"AIC {aic}, AICc {aicc.round(2)}")
E.choice("s09p09b", "AICはM3、AIC$_\\mathrm{c}$はM1を選ぶ", "argmin")

# ---- 大問10 林業
M_ = np.array([120, 80, 150])
yb_ = np.array([0.62, 0.55, 0.70])
E.num("s09p10a", rnd(30 / 3 * np.sum(M_ * yb_), 0), "N/n Σ M_i ȳ_i")
# 二段抽出の不偏性をシミュレーションで確認（人工の森林）
Mi = rng.integers(60, 160, 30)
trees = [rng.gamma(4, 0.15, m) for m in Mi]
tot = sum(t.sum() for t in trees)
ests = []
for _ in range(20_000):
    sel = rng.choice(30, 3, replace=False)
    ests.append(30 / 3 * sum(Mi[i] * rng.choice(trees[i], 5, replace=False).mean() for i in sel))
E.check("s09p10a", abs(np.mean(ests) / tot - 1) < 0.01, f"推定量の平均/真の総量 {np.mean(ests) / tot:.4f}")
# (b) ラテン方格：平方和が与えた値になる 5×5 データを作り、statsmodels を使わず最小二乗で分解
sst, ssr, ssc, ssa = 156.0, 36.0, 20.0, 64.0
dfe = 4 * 3
Fv = (ssa / 4) / ((sst - ssr - ssc - ssa) / dfe)
table_ok("s09p10b", [(stats.f.ppf(0.95, 4, 12), 3.259), (stats.f.ppf(0.95, 12, 4), 5.912),
                     (stats.f.ppf(0.95, 4, 16), 3.007), (stats.f.ppf(0.95, 4, 20), 2.866)], 3, "F の上側5%点")
# 自由度の確認：5×5 ラテン方格の計画行列のランクから誤差自由度を数える
sq = np.array([[(i + j) % 5 for j in range(5)] for i in range(5)])
X = np.column_stack([np.ones(25)] + [np.eye(5)[np.repeat(np.arange(5), 5)][:, 1:]] +
                    [np.eye(5)[np.tile(np.arange(5), 5)][:, 1:]] + [np.eye(5)[sq.ravel()][:, 1:]])
E.check("s09p10b", 25 - np.linalg.matrix_rank(X) == 12, "計画行列のランクから誤差自由度 12")
E.choice("s09p10b", f"$F={f(Fv, 2)}$で、$3.259$を超えるので産地による差があるといえる", "平方和の分解")
E.check("s09p10b", stats.f.sf(Fv, 4, 12) < 0.05, f"p 値 {stats.f.sf(Fv, 4, 12):.4f}")

# ---- 大問11 相対危険
x1, n1, x0, n0 = 14, 200, 35, 250
lrr = np.log((x1 / n1) / (x0 / n0))
se = np.sqrt((1 - x1 / n1) / x1 + (1 - x0 / n0) / x0)
lo, hi = np.exp(lrr + np.array([-1, 1]) * stats.norm.ppf(0.975) * se)
E.choice("s09p11a", f"$({f(lo, 3)},\\{f(hi, 3)})$", "デルタ法の分散 (1-p)/(np) から直接")
it = E.item("s09p11a")
lines = it["question"].splitlines()
exps = [float(t) for t in re.findall(r"-?\d\.\d+", next(s for s in lines if s.startswith("| $x$")))]
vals = [float(t) for t in re.findall(r"-?\d\.\d+", next(s for s in lines if s.startswith("| $e^{x}$")))]
E.check("s09p11a", len(exps) == len(vals) == 6, "e^x の表が 6 組")
table_ok("s09p11a", list(zip(np.exp(exps), vals)), 4, "e^x の表")
X1 = rng.binomial(n1, 0.07, 200_000)
X0 = rng.binomial(n0, 0.14, 200_000)
ok = (X1 > 0) & (X0 > 0)
l_ = np.log((X1[ok] / n1) / (X0[ok] / n0))
s_ = np.sqrt(1 / X1[ok] - 1 / n1 + 1 / X0[ok] - 1 / n0)
cover = np.mean(np.abs(l_ - np.log(0.5)) < 1.96 * s_)
E.check("s09p11a", 0.93 < cover < 0.97, f"区間の被覆確率 {cover:.3f}")

# ---- 大問12 マラソン（重要度重みで事後予測確率）
table_ok("s09p12a", [(stats.norm.cdf(z), g) for z, g in
                     [(0.25, 0.5987), (0.51, 0.6950), (0.56, 0.7123), (1.00, 0.8413), (1.34, 0.9099)]])
mus = rng.normal(180, 6, 2_000_000)
wts = stats.norm.pdf(172, mus, 8 / 2)
nxt_rec = mus + rng.normal(0, 8, mus.size)
pp = np.sum(wts * (nxt_rec < 170)) / np.sum(wts)
E.check("s09p12a", abs(pp - 0.303) < 0.004, f"重要度重み付きシミュレーション {pp:.4f}")
nearest("s09p12a", pp)
E.choice("s09p12a", f"${f(stats.norm.sf(rnd((174.4615 - 170) / np.sqrt(1 / (1 / 36 + 4 / 64) + 64), 2)), 3)}$", "丸めた z と scipy の Φ")

E.done()
