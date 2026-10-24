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


# ---- 大問1：吸収マルコフ連鎖（状態＝次に投げる人）を解き、シミュレーションでも確かめる
p_hit = 0.25
# 状態 P の手番・Q の手番。x_P = p + (1-p) x_Q、x_Q = (1-p) x_P を連立
Am = np.array([[1, -(1 - p_hit)], [-(1 - p_hit), 1]])
xP, xQ = np.linalg.solve(Am, [p_hit, 0])
nearest("s14p01a", xP, "連立方程式で先手が勝つ確率")
wins = 0
for _ in range(200_000):
    turn = 0
    while not (rng.random() < 0.5 and rng.random() < 0.5):
        turn ^= 1
    wins += turn == 0
E.check("s14p01a", close(wins / 200_000, 4 / 7, 0.004), f"シミュレーション {wins / 200_000:.4f}（許容 ±0.004）")
distinct("s14p01a", 3)

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
# (a) 多項分布の共分散行列（scipy）とシミュレーション
mn = stats.multinomial(20, [0.30, 0.15, 0.55])
Cm = mn.cov()
E.num("s14p03a", float(f(Cm[0, 1] / np.sqrt(Cm[0, 0] * Cm[1, 1]), 3)), "scipy multinomial の共分散行列")
smp = rng.multinomial(20, [0.30, 0.15, 0.55], 400_000)
rs = np.corrcoef(smp[:, 0], smp[:, 1])[0, 1]
E.check("s14p03a", close(rs, -0.275, 0.005), f"シミュレーションの相関 {rs:.4f}（許容 ±0.005）")
# (b) sympy で連立方程式を解く
a, b = sp.symbols("a b", positive=True)
sol = sp.solve([a / (a + b) - sp.Rational(76, 100), a * b / ((a + b) ** 2 * (a + b + 1)) - sp.Rational(64, 10000)], [a, b], dict=True)
E.num("s14p03b", float(f(float(sol[0][a]), 1)), "sympy で平均・分散の式を連立")
E.check("s14p03b", close(stats.beta(float(sol[0][a]), float(sol[0][b])).var(), 0.0064, 1e-9), "scipy のベータ分布の分散が 0.0064")
# (c) クロッパー＝ピアソン：ベータ分布の分位点と、二項分布の裾確率を数値で解く
lo_beta = stats.beta.ppf(0.025, 12, 1)
from scipy.optimize import brentq  # noqa: E402
lo_root = brentq(lambda th: stats.binom.sf(11, 12, th) - 0.025, 0.01, 0.999)
E.check("s14p03c", close(lo_beta, lo_root, 1e-8), f"ベータ分位点 {lo_beta:.5f} と二項の裾確率の解 {lo_root:.5f} が一致")
E.choice("s14p03c", f"${f(lo_beta, 3)}$", "Beta(12,1) の 2.5% 点")
for v, val in [(0.025 ** (1 / 12), 0.7354), (0.05 ** (1 / 12), 0.7791), (0.025 ** (1 / 11), 0.7151)]:
    E.check("s14p03c", close(v, val, 5e-5), f"示した値 {val}")
distinct("s14p03c", 3)

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
E.check("s14p04b", E.item("s14p04b")["answer"] == 0, "正しい記述は「確率収束するが平均二乗収束しない」（並べ替え後の位置）")

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
# (a) 切断分布を直接足し合わせる
xs = np.arange(1, 200)
pm = stats.poisson(1.5).pmf(xs)
E.num("s14p08a", float(f((xs * pm).sum() / pm.sum(), 3)), "x≥1 の確率関数で直接和をとる")
E.check("s14p08a", close(np.exp(-1.5), 0.2231, 5e-5), "e^{-1.5} の値")
# (b) scipy の負の二項分布（n=θ, p=θ/(θ+μ)）、示した e^x の値
for xv, val in [(0.25, 1.284), (0.34, 1.405), (0.49, 1.632), (-1.632, 0.1955)]:
    E.check("s14p08b", close(np.exp(xv), val, 5e-4), f"e^{xv} = {val}")
mu_ = np.exp(0.10 + 0.24 + 0.30 * 0.5)
p0 = stats.nbinom(2.0, 2.0 / (2.0 + mu_)).pmf(0)
nearest("s14p08b", p0, "scipy nbinom の P(Y=0)")
nbv = stats.nbinom(2.0, 2.0 / (2.0 + mu_))
E.check("s14p08b", close(nbv.var(), mu_ + mu_ ** 2 / 2.0, 1e-9), "scipy の分散が μ+μ²/θ（パラメータの対応の確認）")
distinct("s14p08b", 3)

# ---- 大問9：外部スチューデント化残差の公式を削除計算で確かめ、ボンフェローニの点を scipy で
n_, p_ = 36, 5
from numpy.linalg import lstsq  # noqa: E402
g = np.random.default_rng(7)
X = np.column_stack([np.ones(n_), g.normal(size=(n_, p_))])
y = g.normal(size=n_)
H = X @ np.linalg.inv(X.T @ X) @ X.T
e = y - H @ y
s2 = e @ e / (n_ - p_ - 1)
i = 0
ri = e[i] / np.sqrt(s2 * (1 - H[i, i]))
mask = np.arange(n_) != i
bi = lstsq(X[mask], y[mask], rcond=None)[0]
s2i = ((y[mask] - X[mask] @ bi) ** 2).sum() / (n_ - p_ - 2)
t_direct = e[i] / np.sqrt(s2i * (1 - H[i, i]))
E.check("s14p09a", close(t_direct, ri * np.sqrt((n_ - p_ - 2) / (n_ - p_ - 1 - ri ** 2)), 1e-8),
        "人工データで削除計算した外部スチューデント化残差が公式と一致")
t_ = 2.6 * np.sqrt(29 / (30 - 2.6 ** 2))
for k, q, v in [(29, 0.975, 2.045), (29, 1 - 0.05 / 72, 3.54), (30, 1 - 0.05 / 72, 3.52)]:
    E.check("s14p09a", close(stats.t.ppf(q, k), v, 0.006), f"t 点 ({k}, {q:.6f}) = {v}")
padj = min(1, 72 * stats.t.sf(t_, 29))
E.check("s14p09a", padj > 0.05 and 2 * stats.t.sf(t_, 29) < 0.05 and E.item("s14p09a")["answer"] == 4,
        f"ボンフェローニ調整 p 値 {padj:.3f} > 0.05（未調整は < 0.05）→ 選択肢1")
E.choice("s14p09a", f"t_i={f(t_, 2)}$で、$t_{{0.05/72}}(29)=3.54$を超えないので", "t_i の値と補正した点")
dff = t_ * np.sqrt(0.35 / 0.65)
E.check("s14p09a", dff > 2 * np.sqrt(6 / 36) and 0.35 > 2 * 6 / 36, f"DFFITS {dff:.3f} が目安を超える（選択肢3は誤り）")
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
# (a) 層別抽出をシミュレーション（層内は正規の母集団を作り、比例配分で繰り返し抽出）
popA = rng.normal(20, 5, 8000)
popB = rng.normal(40, 15, 2000)
popA = (popA - popA.mean()) / popA.std(ddof=1) * 5 + 20
popB = (popB - popB.mean()) / popB.std(ddof=1) * 15 + 40
ests = []
for _ in range(20_000):
    ya = rng.choice(popA, 160, replace=True)
    yb = rng.choice(popB, 40, replace=True)
    ests.append(0.8 * ya.mean() + 0.2 * yb.mean())
sd_sim = np.std(ests)
nearest("s14p10a", sd_sim, "復元抽出（有限母集団修正なしに対応）のシミュレーションの標準偏差")
E.check("s14p10a", close(sd_sim, 0.5701, 0.01), f"シミュレーション {sd_sim:.4f}（許容 ±0.01）")
distinct("s14p10a", 3)
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
# (b) 正規分布の対数尤度比（2 次判別の式で共分散共通）から直接、判別関数値を求める
mu1, mu2 = np.array([7, 5.0]), np.array([4, 4.0])
Sg2 = np.array([[2, 1], [1, 2.0]])
xv = np.array([5.7, 6.0])
llr = stats.multivariate_normal(mu1, Sg2).logpdf(xv) - stats.multivariate_normal(mu2, Sg2).logpdf(xv)
E.choice("s14p11b", f"L={f(llr, 2)}$で、一般プレイヤー", "2 群の対数尤度比（= 線形判別関数）")
dM = [float((xv - m) @ np.linalg.solve(Sg2, xv - m)) for m in (mu1, mu2)]
E.check("s14p11b", dM[1] < dM[0], f"マハラノビス距離の 2 乗 {dM[0]:.3f}（群1）> {dM[1]:.3f}（群2）で群2 に近い")

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
