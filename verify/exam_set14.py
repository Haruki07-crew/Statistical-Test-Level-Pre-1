"""模擬試験 第14回（set14）の検算。generator/set14.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値計算）で正解を求め直し、問題文に示した表の値も確かめる。

使い方: .venv/bin/python verify/exam_set14.py
"""
import itertools
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(14)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    out = []
    for c in E.item(iid)["choices"]:
        m = re.findall(r"-?\d+\.\d+|-?\d+", str(c))
        out.append(float(m[0]) if m else float("nan"))
    return out


def nearest(iid, value, what):
    v = nums(iid)
    k = int(np.nanargmin([abs(x - value) for x in v]))
    E.check(iid, k == E.item(iid)["answer"], f"{what}: 最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']}")


def distinct(iid, d):
    v = [f(x, d) for x in nums(iid)]
    E.check(iid, len(set(v)) == len(v), f"選択肢の数値が丸めても重ならない {v}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1：4 通りの結果を列挙して独立性を確かめる
out = list(itertools.product([0, 1], repeat=2))  # (赤, 青) 1=表
P = lambda ev: sum(1 for w in out if ev(w)) / 4
A = lambda w: w[0] == 1
B = lambda w: w[1] == 1
C = lambda w: w[0] == w[1]
pair_ind = all(abs(P(lambda w: x(w) and y(w)) - P(x) * P(y)) < 1e-12 for x, y in [(A, B), (A, C), (B, C)])
triple = P(lambda w: A(w) and B(w) and C(w))
st = [triple == P(A) * P(B) * P(C),                                        # 0 相互独立
      not abs(P(lambda w: A(w) and C(w)) - P(A) * P(C)) < 1e-12,              # 1 A と C は独立でない
      pair_ind and triple != P(A) * P(B) * P(C),                              # 2
      abs(P(lambda w: A(w) and B(w) and C(w)) - P(A) * P(lambda w: B(w) and C(w))) < 1e-12,  # 3
      abs(P(lambda w: A(w) or B(w) or C(w)) - 7 / 8) < 1e-12]                 # 4
E.check("s14p01a", st == [False, False, True, False, False], f"列挙で各記述の真偽 {st}")
E.choice("s14p01a", "\\frac{1}{4}", "P(A∩B∩C)=1/4 の記述")

# ---- 大問2
# (a) 示した Φ と ln2 の確認、正確な値とモンテカルロ
for z, v in {0.82: 0.7939, 1.16: 0.8770, 1.40: 0.9192, 1.64: 0.9495, 1.98: 0.9761}.items():
    E.check("s14p02a", close(stats.norm.cdf(z), v, 5e-5), f"Φ({z})={v}")
E.check("s14p02a", close(np.log(2), 0.6931, 5e-5), "log 2 = 0.6931")
exact = stats.lognorm(s=np.sqrt(0.18), scale=np.exp(0.2)).sf(2)
nearest("s14p02a", exact, "scipy の対数正規 sf(2)（積の分布）")
M = np.exp(rng.normal(0.1, 0.3, (2, 1_000_000))).prod(axis=0)
E.check("s14p02a", close((M > 2).mean(), 0.123, 0.003), f"シミュレーション {(M > 2).mean():.4f}")
distinct("s14p02a", 3)
# (b) 数値積分で D の密度から P(D>5)、シミュレーション
E.check("s14p02b", close(np.exp(-0.5), 0.6065, 5e-5) and close(np.exp(-1 / 3), 0.7165, 5e-5), "e^{-0.5}, e^{-1/3} の値")
pd = integrate.dblquad(lambda y, x: stats.expon(scale=10).pdf(x) * stats.expon(scale=15).pdf(y), 0, np.inf,
                       lambda x: 0, lambda x: max(x - 5, 0))[0]
nearest("s14p02b", pd, "二重積分で P(X_A - X_B > 5)")
XA, XB = rng.exponential(10, 2_000_000), rng.exponential(15, 2_000_000)
E.check("s14p02b", close((XA - XB > 5).mean(), 0.2426, 0.002), f"シミュレーション {(XA - XB > 5).mean():.4f}")
distinct("s14p02b", 3)

# ---- 大問3
# (a) 共分散行列から直接
S = np.array([[36, 0.25 * 48], [0.25 * 48, 64]])
w = np.array([1, 1])
corr = (S @ w)[0] / np.sqrt(S[0, 0] * w @ S @ w)
E.num("s14p03a", float(f(corr, 2)), "行列計算 Cov(X1,T)/√(V1 V_T)")
# (b) sympy で連立方程式を解く
a, b = sp.symbols("a b", positive=True)
sol = sp.solve([a / (a + b) - sp.Rational(76, 100), a * b / ((a + b) ** 2 * (a + b + 1)) - sp.Rational(64, 10000)], [a, b], dict=True)
E.num("s14p03b", float(f(float(sol[0][a]), 1)), "sympy で平均・分散の式を連立")
E.check("s14p03b", close(stats.beta(float(sol[0][a]), float(sol[0][b])).var(), 0.0064, 1e-9), "scipy のベータ分布の分散が 0.0064")
# (c) scipy の χ² 点で区間、表の値の確認
for k, q, v in [(9, 0.025, 2.70), (9, 0.975, 19.02), (10, 0.025, 3.25), (10, 0.975, 20.48)]:
    E.check("s14p03c", close(stats.chi2.ppf(q, k), v, 0.006), f"χ² 点 ({k}, 下側{q}) = {v}")
up = np.sqrt(9 * 25 / stats.chi2.ppf(0.025, 9))
E.choice("s14p03c", f"${f(up, 2)}$", "scipy の点で上限")
sims = rng.normal(0, 5, (200_000, 10)).std(axis=1, ddof=1)
cover = np.mean((np.sqrt(9 * sims ** 2 / 19.02) <= 5) & (5 <= np.sqrt(9 * sims ** 2 / 2.70)))
E.check("s14p03c", close(cover, 0.95, 0.005), f"区間の被覆率シミュレーション {cover:.4f}")

# ---- 大問4
# (a) 最大値の検出力をシミュレーションと解析式（c を数値で求める）
c = 0.95 ** (1 / 5)
E.check("s14p04a", close(1.2 ** 5, 2.48832, 1e-9), "1.2^5 の値")
Mx = rng.uniform(0, 1.2, (1_000_000, 5)).max(axis=1)
pw = (Mx >= c).mean()
nearest("s14p04a", pw, "シミュレーションの検出力")
M0 = rng.uniform(0, 1, (1_000_000, 5)).max(axis=1)
E.check("s14p04a", close((M0 >= c).mean(), 0.05, 0.001), "θ=1 で有意水準 5%")
distinct("s14p04a", 3)
# (b) 定義どおりに P(|X_n|>ε) と E[X_n^2] を sympy で
n = sp.symbols("n", positive=True)
EX2 = n ** 2 * (1 / n)
EX = n * (1 / n)
E.check("s14p04b", sp.limit(1 / n, n, sp.oo) == 0 and sp.limit(EX2, n, sp.oo) == sp.oo and sp.simplify(EX) == 1,
        "P(X_n≠0)→0、E[X_n^2]→∞、E[X_n]=1")
E.check("s14p04b", E.item("s14p04b")["answer"] == 2, "正しい記述は「確率収束するが平均二乗収束しない」(選択肢2)")

# ---- 大問5：吸収マルコフ連鎖を線形方程式で解く
Q = np.zeros((5, 5))
bvec = np.zeros(5)
for i in range(1, 6):
    if i + 1 <= 5:
        Q[i - 1, i] = 0.45
    else:
        bvec[i - 1] = 0.45
    if i - 1 >= 1:
        Q[i - 1, i - 2] = 0.55
h = np.linalg.solve(np.eye(5) - Q, bvec)
E.num("s14p05a", float(f(h[2], 3)), "吸収確率の連立方程式")
walk = 0
for _ in range(100_000):
    x = 3
    while 0 < x < 6:
        x += 1 if rng.random() < 0.45 else -1
    walk += x == 6
E.check("s14p05a", close(walk / 100_000, 0.354, 0.005), f"シミュレーション {walk / 100_000:.4f}")

# ---- 大問6
# (a) scipy の spearmanr
j1 = list(range(1, 9))
j2 = [2, 1, 4, 3, 6, 8, 5, 7]
E.num("s14p06a", float(f(stats.spearmanr(j1, j2)[0], 3)), "scipy spearmanr")
# (b) F 点の確認と、混合モデルのシミュレーションで MS_A/MS_AB の帰無分布
for (d1, d2, q, v) in [(3, 12, .95, 3.49), (3, 20, .95, 3.10), (3, 32, .95, 2.90), (3, 12, .90, 2.61)]:
    E.check("s14p06b", close(stats.f.ppf(q, d1, d2), v, 0.006), f"F_{1 - q:.2f}({d1},{d2})={v}")
Fs, Fe = [], []
for _ in range(4000):
    bj = rng.normal(0, 1, 5)
    ab = rng.normal(0, 1.0, (4, 5))
    y = bj[None, :, None] + ab[:, :, None] + rng.normal(0, 1, (4, 5, 2))
    gm = y.mean()
    ma, mb, mab = y.mean(axis=(1, 2)), y.mean(axis=(0, 2)), y.mean(axis=2)
    ssa = 10 * ((ma - gm) ** 2).sum()
    ssab = 2 * ((mab - ma[:, None] - mb[None, :] + gm) ** 2).sum()
    sse = ((y - mab[:, :, None]) ** 2).sum()
    Fs.append((ssa / 3) / (ssab / 12))
    Fe.append((ssa / 3) / (sse / 20))
Fs, Fe = np.array(Fs), np.array(Fe)
E.check("s14p06b", close((Fs > 3.49).mean(), 0.05, 0.012) and (Fe > 3.10).mean() > 0.15,
        f"H0 のもとで MS_A/MS_AB は名目 5%（{(Fs > 3.49).mean():.3f}）、MS_A/MS_E は過大（{(Fe > 3.10).mean():.3f}）")
E.choice("s14p06b", f"F={9 / 3:.2f}$で、$F_{{0.05}}(3,12)=3.49$を超えないので有意でない", "MS_A/MS_AB と F(3,12)")
# (c) scipy の F 分布で両側 p 値
Fv = 0.48 ** 2 / 0.30 ** 2
pv = 2 * stats.f.sf(Fv, 10, 15)
E.check("s14p06c", pv > 0.05 and stats.f.sf(Fv, 10, 15) < 0.05, f"両側 p 値 {pv:.4f} > 0.05（片側なら < 0.05）")
E.check("s14p06c", close(stats.f.ppf(.975, 10, 15), 3.06, 0.005) and close(stats.f.ppf(.95, 10, 15), 2.54, 0.005), "F 点の値")
E.choice("s14p06c", f"F={f(Fv, 2)}$で、$3.06$を超えないので帰無仮説を棄却しない", "F=2.56 で棄却しない")

# ---- 大問7
R = np.array([[1, 0.6], [0.6, 1]])
vif = np.linalg.inv(R)[0, 0]
E.choice("s14p07a", f"\\mathrm{{VIF}}={f(vif, 2)},\\ \\text{{標準誤差は}}\\ {f(np.sqrt(vif), 2)}", "逆相関行列の対角成分")
Xs = rng.multivariate_normal([0, 0], R, 200_000)
Xs = (Xs - Xs.mean(0)) / Xs.std(0)
Xi = Xs.copy()
Xi[:, 1] = rng.normal(size=200_000)
se_ratio = np.sqrt(np.linalg.inv(Xs.T @ Xs)[0, 0] / np.linalg.inv(Xi.T @ Xi)[0, 0])
E.check("s14p07a", close(se_ratio, 1.25, 0.01), f"シミュレーションの標準誤差の比 {se_ratio:.4f}")
# (b) 相関行列の固有分解で第 1 主成分を作り、射影で係数を戻す
r = np.array([0.5, 0.45])
ev, V = np.linalg.eigh(R)
v1 = V[:, np.argmax(ev)]
gam = (v1 @ r) / ev.max()
beta = gam * v1
E.choice("s14p07b", f"({f(beta[0], 3)},\\ {f(beta[1], 3)})", "固有分解による主成分回帰")

# ---- 大問8
# (a) 同時確率の総当たり
num = stats.poisson(1.5).pmf(2) * stats.poisson(1.0).pmf(1)
den = sum(stats.poisson(1.5).pmf(x) * stats.poisson(1.0).pmf(3 - x) for x in range(4))
E.num("s14p08a", float(f(num / den, 3)), "同時確率の比")
# (b) Pearson φ と z の再計算、e^0.24
phi = 333.0 / (152 - 4)
z = 0.24 / (0.08 * np.sqrt(phi))
E.check("s14p08b", close(np.exp(0.24), 1.271, 5e-4), "e^0.24 = 1.271")
E.check("s14p08b", abs(z) > 1.96 and E.item("s14p08b")["answer"] == 0, f"準ポアソンの z={z:.3f} > 1.96 で有意（選択肢0）")
E.choice("s14p08b", f"倍の${f(0.08 * np.sqrt(phi), 2)}$にする。$z={f(z, 2)}$となり、有意である", "SE×√φ の記述")

# ---- 大問9：DFFITS をデータを作って当てはめ直しで確かめる（外部化の公式を使わない）
n_, p_ = 36, 5
# 公式の値を、h と r を満たす人工データでの削除計算と照合する
from numpy.linalg import lstsq  # noqa: E402
found = False
for seed in range(200):
    g = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_), g.normal(size=(n_, p_))])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    i = int(np.argmin(abs(np.diag(H) - 0.35)))
    if abs(H[i, i] - 0.35) < 0.02:
        found = True
        break
y = g.normal(size=n_)
e = y - H @ y
s2 = e @ e / (n_ - p_ - 1)
ri = e[i] / np.sqrt(s2 * (1 - H[i, i]))
mask = np.arange(n_) != i
bi = lstsq(X[mask], y[mask], rcond=None)[0]
s2i = ((y[mask] - X[mask] @ bi) ** 2).sum() / (n_ - p_ - 2)
dff_direct = (H[i] @ y - X[i] @ bi) / np.sqrt(s2i * H[i, i])
ti = ri * np.sqrt((n_ - p_ - 2) / (n_ - p_ - 1 - ri ** 2))
E.check("s14p09a", found and close(dff_direct, ti * np.sqrt(H[i, i] / (1 - H[i, i])), 1e-8),
        "人工データで削除計算した DFFITS が t_i√(h/(1-h)) と一致（公式の確認）")
t_ = 2.6 * np.sqrt(29 / (30 - 2.6 ** 2))
E.num("s14p09a", float(f(t_ * np.sqrt(0.35 / 0.65), 2)), "丸めない計算")
# (b) d 分離：線形ガウス SEM で偏相関を計算
coef = {"X": {"Z": 0.7}, "Y": {"Z": 0.5, "X": 0.6}, "S": {"X": 0.4, "Y": 0.8}}
order = ["Z", "X", "Y", "S"]
Bm = np.zeros((4, 4))
for k, pa in coef.items():
    for q, cf in pa.items():
        Bm[order.index(k), order.index(q)] = cf
Ainv = np.linalg.inv(np.eye(4) - Bm)
Sig = Ainv @ Ainv.T


def pcorr(a_, b_, cond):
    idx = [order.index(x) for x in [a_, b_] + cond]
    Pm = np.linalg.inv(Sig[np.ix_(idx, idx)])
    return -Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1])


tests = [abs(pcorr("Z", "Y", ["X"])) < 1e-10, abs(pcorr("Z", "S", ["X"])) < 1e-10, abs(pcorr("Z", "S", [])) < 1e-10,
         abs(pcorr("Z", "S", ["X", "Y"])) < 1e-10, abs(pcorr("X", "S", ["Y"])) < 1e-10]
E.check("s14p09b", tests == [False, False, False, True, False], f"線形SEMの偏相関が0になるもの {tests}")

# ---- 大問10
# (a) 全標本の列挙
ep = np.array([10, 12, 15, 11, 13, 16, 12, 14, 18, 13, 15, 19])
ms = [ep[j::3].mean() for j in range(3)]
E.choice("s14p10a", f"${f(np.mean((np.array(ms) - ep.mean()) ** 2), 2)}$", "3 標本の列挙")
distinct("s14p10a", 2)
# (b) 正準相関を一般の公式（固有値）で
Ryy = np.array([[1.0]])
Rxx = np.array([[1, 0.3], [0.3, 1]])
Rxy = np.array([[0.6], [0.4]])
Mcc = np.linalg.inv(Ryy) @ Rxy.T @ np.linalg.inv(Rxx) @ Rxy
cc = np.sqrt(np.linalg.eigvals(Mcc).real.max())
E.choice("s14p10b", f"${f(cc, 3)}$", "正準相関の固有値問題")
# (c) DF 臨界値をシミュレーションで確認し、結論を判定
taus = []
for _ in range(4000):
    yv = np.cumsum(rng.normal(size=201))
    dy, yl = np.diff(yv), yv[:-1]
    Xd = np.column_stack([np.ones(200), yl])
    bb, res = np.linalg.lstsq(Xd, dy, rcond=None)[:2]
    s2d = res[0] / (200 - 2)
    taus.append(bb[1] / np.sqrt(s2d * np.linalg.inv(Xd.T @ Xd)[1, 1]))
q5 = np.quantile(taus, 0.05)
E.check("s14p10c", close(q5, -2.86, 0.08), f"定数項ありDF分布の5%点（シミュレーション）{q5:.3f} ≈ -2.86")
tau = -0.08 / 0.03
E.check("s14p10c", tau > -2.86 and tau < -1.645 and E.item("s14p10c")["answer"] == 1,
        f"τ={tau:.3f} は DF 点を下回らず（棄却できない）、正規の点なら棄却 → 選択肢1")

# ---- 大問11
# (a) numpy の固有値
Rm = np.array([[1, .6, .5], [.6, 1, .4], [.5, .4, 1]])
evs = np.sort(np.linalg.eigvalsh(Rm))[::-1]
E.check("s14p11a", close(evs[0], 2.004, 0.0005), f"最大固有値 {evs[0]:.5f} ≈ 2.004")
E.choice("s14p11a", f"{f(100 * evs[:2].sum() / 3, 1)}\\%", "numpy eigvalsh で累積寄与率")
distinct("s14p11a", 1)
# (b) 同時分布を直接作って条件付き確率
pj = {}
for cl, pr, a1, a2 in [(1, 0.05, 0.8, 0.6), (0, 0.95, 0.1, 0.2)]:
    for x1 in (0, 1):
        for x2 in (0, 1):
            pj[(cl, x1, x2)] = pr * (a1 if x1 else 1 - a1) * (a2 if x2 else 1 - a2)
E.num("s14p11b", float(f(pj[(1, 1, 0)] / (pj[(1, 1, 0)] + pj[(0, 1, 0)]), 3)), "同時分布表から条件付き確率")

# ---- 大問12
# (a) 個人データを再現して lifelines 風に手計算（打ち切りを個別に処理）
times = [(2, 1), (7, 0), (9, 1), (12, 0), (14, 1)] + [(30, 0)] * 5   # (時間, やめた=1)
Sg, gs = 1.0, 0.0
for t in sorted({t for t, d in times if d}):
    nr = sum(1 for tt, _ in times if tt >= t)
    dd = sum(1 for tt, d in times if tt == t and d)
    Sg *= 1 - dd / nr
    gs += dd / (nr * (nr - dd))
E.check("s14p12a", close(Sg, 0.65625, 1e-12), f"KM 推定値 {Sg}")
E.choice("s14p12a", f"${f(Sg * np.sqrt(gs), 3)}$", "個人データからのグリーンウッド")
distinct("s14p12a", 3)
# (b) 総当たりの代わりに大規模シミュレーションと厳密式
E.check("s14p12b", close(0.9 ** 10, 0.3487, 5e-5), "0.9^10 の値")
dist = np.array([len(set(rng.integers(0, 10, 10))) for _ in range(200_000)])
E.check("s14p12b", close(dist.mean(), 6.513, 0.02), f"シミュレーションの平均 {dist.mean():.4f}")
from sympy.functions.combinatorial.numbers import stirling  # noqa: E402
ex = sp.Rational(sum(sp.binomial(10, jj) * sp.factorial(jj) * stirling(10, jj) * jj for jj in range(1, 11)), 10 ** 10)
E.choice("s14p12b", f"${f(float(ex), 2)}$", "第2種スターリング数による厳密な期待値")

E.done()
