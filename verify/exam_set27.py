"""模擬試験 第27回（set27）の検算。generator/set27.py とは別の方法（scipy の分布関数・シミュレーション・
総当たり・数値積分・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(27)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


# ---- 大問1
# (a) scipy の期待度数
tab = np.array([[9, 3], [2, 6]])
ex = stats.contingency.expected_freq(tab)
E.check("s27p01a", f(ex.min(), 1) == "3.6" and (ex < 5).sum() == 2, f"期待度数 {ex.ravel()}")
E.choice("s27p01a", f"最小の期待度数は{f(ex.min(), 1)}（地区Bの反対）で、期待度数が5未満のセルが4つのうち2つ", "scipy expected_freq")
E.check("s27p01a", stats.fisher_exact(tab).pvalue > 0, f"フィッシャーの正確検定は計算できる（p={stats.fisher_exact(tab).pvalue:.4f}）")
# (b) ベータ事後分布の分位点
post = stats.beta(31, 21)
E.check("s27p01b", f(post.ppf(0.025), 3) == "0.461" and f(post.ppf(0.975), 3) == "0.724", f"Beta(31,21) の分位点 {post.ppf([0.025, 0.975])}")
E.check("s27p01b", f(stats.beta(30, 20).ppf(0.025), 3) == "0.462" and f(stats.beta(30, 20).ppf(0.975), 3) == "0.730", "Beta(30,20) の分位点（誤答の値）")
E.choice("s27p01b", f"事後分布$\\mathrm{{Beta}}(31,21)$、信用区間$({f(post.ppf(0.025), 3)},\\{f(post.ppf(0.975), 3)})$。データと事前分布のもとで", "scipy beta.ppf")
th = rng.beta(1, 1, 2_000_000)
yy = rng.binomial(50, th)
sel = th[yy == 30]
E.check("s27p01b", abs(np.mean((sel > 0.461) & (sel < 0.724)) - 0.95) < 0.01, f"事前分布から θ を引き y=30 のものだけ残すと区間に入る割合 {np.mean((sel > 0.461) & (sel < 0.724)):.4f}")

# ---- 大問2
# (a) シミュレーション
T2 = rng.normal(30, 4, 1_000_000) + rng.normal(20, 3, 1_000_000)
E.check("s27p02a", abs(np.mean(T2 > 60) - stats.norm.sf(2)) < 0.001, f"シミュレーション {np.mean(T2 > 60):.4f}")
for z_, v_ in ((1.0, 0.8413), (1.43, 0.9236), (2.0, 0.9772)):
    E.check("s27p02a", abs(stats.norm.cdf(z_) - v_) < 5e-5, f"Φ({z_})")
E.choice("s27p02a", f"${f(stats.norm.sf(10 / 5), 4)}$", "scipy norm.sf")
# (b) TOST：δ を格子で動かし、2 つの片側検定がともに棄却される最小の δ
d, se = 1.2, 0.8
z95 = stats.norm.ppf(0.95)
grid = np.arange(0, 5, 0.0001)
ok = [dl for dl in grid if (d + dl) / se > z95 and (dl - d) / se > z95]
E.num("s27p02b", rnd(ok[0], 2), f"格子で TOST が成立する最小の δ={ok[0]:.4f}")
E.check("s27p02b", abs(ok[0] - (d + z95 * se)) < 2e-4, "90% 信頼区間の上端と一致")
# (c) 一様分布の最大値の正確な確率とシミュレーション
M = rng.uniform(0, 600, (100_000, 200)).max(1)
E.num("s27p02c", rnd(math.exp(-1), 3), "極限の指数分布 e^{-1}")
E.check("s27p02c", abs(np.mean(M <= 597) - (597 / 600) ** 200) < 0.004 and abs((597 / 600) ** 200 - math.exp(-1)) < 0.002,
        f"シミュレーション {np.mean(M <= 597):.4f}、正確な値 {(597 / 600) ** 200:.4f}")

# ---- 大問3
# (a) sympy を使わず数値積分で周辺密度と条件付き期待値
fy = lambda y: integrate.quad(lambda x: 6 * x, 0, y)[0]
ex_cond = lambda y: integrate.quad(lambda x: x * 6 * x, 0, y)[0] / fy(y)
E.check("s27p03a", all(abs(fy(y) - 3 * y * y) < 1e-10 and abs(ex_cond(y) - 2 * y / 3) < 1e-10 for y in (0.3, 0.7, 0.95)), "数値積分で f_Y=3y²、E[X|Y=y]=2y/3")
E.check("s27p03a", abs(integrate.dblquad(lambda x, y: 6 * x, 0, 1, 0, lambda y: y)[0] - 1) < 1e-10, "同時密度の積分が 1")
E.choice("s27p03a", "f_Y(y)=3y^2$、$E[X\\mid Y=y]=\\dfrac{2y}{3}$", "数値積分")
# (b) シミュレーション
d1 = rng.random((400_000, 60)) < 0.2
d2 = rng.random((400_000, 60)) < 0.3
first = (d1 | d2).argmax(1) + 1
E.check("s27p03b", abs(first.mean() - 1 / 0.44) < 0.01, f"シミュレーション {first.mean():.4f}")
E.choice("s27p03b", f"${f(1 / 0.44, 2)}$", "1/(1−0.8×0.7)")

# ---- 大問4
# (a) 相関のある (N,S) を作り、積の平均を確認
cov4 = np.array([[100 ** 2, -50], [-50, 0.5 ** 2]])
NS = rng.multivariate_normal([1000, 2.0], cov4, 2_000_000)
E.check("s27p04a", abs((NS[:, 0] * NS[:, 1]).mean() - 1950) < 0.5, f"シミュレーションの E[NS]={(NS[:, 0] * NS[:, 1]).mean():.2f}")
E.num("s27p04a", round(1000 * 2.0 - 50), "E[N]E[S]+Cov")
# (b) 対数尤度を数値最大化し、Σr² が十分統計量であることを確認
rs = np.array([1.2, 0.8, 2.0, 1.5, 1.0])
sg2 = np.linspace(0.1, 3, 290001)
ll = np.log(rs).sum() - len(rs) * np.log(sg2) - (rs ** 2).sum() / (2 * sg2)
E.choice("s27p04b", f"十分統計量$\\sum r_i^2$、$\\hat\\sigma^2={f(sg2[np.argmax(ll)], 3)}$", "格子で最大化")
fit = stats.rayleigh.fit(rs, floc=0)
E.check("s27p04b", abs(fit[1] ** 2 - 0.933) < 1e-3 and abs(math.sqrt(math.pi / 2) - 1.2533) < 5e-5, f"scipy rayleigh.fit の尺度² {fit[1] ** 2:.4f}")
# (c) 推移行列の固有ベクトルとシミュレーション
nodes = ["駅", "城", "温泉街", "美術館"]
edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
A = np.zeros((4, 4))
for i, j in edges:
    A[i, j] = A[j, i] = 1
P = A / A.sum(1, keepdims=True)
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmin(np.abs(w - 1))])
pi = pi / pi.sum()
E.check("s27p04c", np.allclose(pi, A.sum(1) / A.sum()) and np.allclose(pi[:, None] * P, (pi[:, None] * P).T), f"定常分布 {np.round(pi, 4)}、詳細つり合い")
E.check("s27p04c", np.allclose(np.linalg.matrix_power(P, 60)[1], pi), "60 ステップ後に初期地点によらず収束（非周期的）")
E.choice("s27p04c", f"{f(pi[2], 1)}。定常分布は各地点から出る散策路の本数（次数）に比例", "固有ベクトル")

# ---- 大問5
# (a) 二相抽出のシミュレーション（母集団：市街地 60%、賛成率 0.55／0.80）
est = []
for _ in range(20000):
    urb = rng.random(500) < 0.6
    wu = urb.mean()
    pu_ = rng.binomial(40, 0.55) / 40
    pm_ = rng.binomial(40, 0.80) / 40
    est.append(wu * pu_ + (1 - wu) * pm_)
E.check("s27p05a", abs(np.mean(est) - 0.65) < 0.002, f"二相抽出の推定量の平均 {np.mean(est):.4f}（母集団の値 0.65、不偏）")
E.num("s27p05a", rnd(300 / 500 * 0.55 + 200 / 500 * 0.80, 3), "w_h ȳ_h の和")
# (b) 最小の n を総当たり
nmin = next(n for n in range(1, 1000) if 1 - 0.98 ** n >= 0.95)
E.choice("s27p05b", f"${nmin}$", "総当たりで最小の n")
E.check("s27p05b", f(0.98 ** 148, 4) == "0.0503" and abs(math.log(0.05) + 2.9957) < 5e-5 and abs(math.log(0.98) + 0.0202) < 5e-5, "0.98^148=0.0503（解説）と対数の値")

# ---- 大問6
# (a) データを作り、2 つのモデルの決定係数から調整済み決定係数と部分 F を statsmodels を使わずに確認
n6 = 30
adj = lambda r2, p: 1 - (1 - r2) * (n6 - 1) / (n6 - p - 1)
F6 = ((0.64 - 0.62) / 2) / ((1 - 0.64) / 25)
E.check("s27p06a", adj(0.62, 2) > adj(0.64, 4) and F6 < stats.f.isf(0.05, 2, 25) and abs(stats.f.isf(0.05, 2, 25) - 3.39) < 0.005,
        f"調整済み {adj(0.62, 2):.4f} > {adj(0.64, 4):.4f}、F={F6:.3f} < {stats.f.isf(0.05, 2, 25):.3f}")
E.choice("s27p06a", f"調整済み決定係数はAが{f(adj(0.62, 2), 3)}、Bが{f(adj(0.64, 4), 3)}でAの方が大きい。部分$F={f(F6, 2)}$は3.39を下回り", "定義どおり")
# 部分 F が 1 より大きいとき、そのときに限り調整済み決定係数が増えることをいくつかの値で確認
E.check("s27p06a", all((adj(0.62 + dr, 4) > adj(0.62, 2)) == (((dr / 2) / ((1 - 0.62 - dr) / 25)) > 1) for dr in np.linspace(0.001, 0.2, 200)), "部分 F>1 ⇔ 調整済み決定係数が増える")
# (b) 残差を作ったデータで最小二乗法を当てはめて確認
x6 = np.array([2, 3, 5, 6, 8, 9, 11, 12.0])
e6 = np.array([1.2, -0.8, 0.5, -1.5, 0.9, -1.5, 1.6, -0.4])
y6 = 14 + 1.6 * x6 + e6
bh = np.polyfit(x6, y6, 1)
E.check("s27p06b", np.allclose(y6 - np.polyval(bh, x6), e6), "y=14+1.6x+e を当てはめ直すと残差が表の値に一致")
known = [i for i in range(8) if x6[i] not in (9, 11)]
Amat = np.array([[1, 1], [9, 11.0]])
uv = np.linalg.solve(Amat, [-e6[known].sum(), -(x6[known] * e6[known]).sum()])
E.choice("s27p06b", f"${f(uv[0], 2)}$", "連立方程式を numpy で解く")

# ---- 大問7 2 値説明変数の GLM をニュートン法で当てはめる
x7 = np.repeat([1, 1, 0, 0], [30, 70, 15, 85]).astype(float)
y7 = np.repeat([1, 0, 1, 0], [30, 70, 15, 85]).astype(float)
X7 = np.column_stack([np.ones(200), x7])
b = np.zeros(2)
for _ in range(30):
    p_ = 1 / (1 + np.exp(-X7 @ b))
    b += np.linalg.solve(X7.T @ (X7 * (p_ * (1 - p_))[:, None]), X7.T @ (y7 - p_))
g = np.array([math.log(0.15), 0.0])
for _ in range(50):
    mu = np.exp(X7 @ g)
    W = mu / (1 - mu)
    z = X7 @ g + (y7 - mu) / mu
    g = np.linalg.solve(X7.T @ (X7 * W[:, None]), X7.T @ (W * z))
E.check("s27p07a", abs(b[1] - math.log((0.3 / 0.7) / (0.15 / 0.85))) < 1e-8 and abs(g[1] - math.log(2)) < 1e-8, f"ロジット {b[1]:.4f}、対数リンク {g[1]:.4f}")
E.choice("s27p07a", f"\\hat\\beta_1={f(b[1], 3)}$（$e^{{\\hat\\beta_1}}$はオッズ比{f(math.exp(b[1]), 2)}）、$\\hat\\gamma_1={f(g[1], 3)}$", "ニュートン法（IRLS）")

# ---- 大問8
# (a) 条件付き分布を作って分位点回帰の直線と一致するか（x ごとの分位点）
xs8 = np.array([1, 2, 3, 4, 5])
qs = {}
for xv in xs8:
    yv = rng.normal(0, 1, 400_000)
    # 分位点が a_τ + b_τ x になるような分布：中央値 2+x、尺度が x に比例（非対称）
    yv = np.where(yv < 0, (2 + xv) + yv * (1 + 0.5 * xv) / 1.2816, (2 + xv) + yv * (2 + xv) / 1.2816)
    qs[xv] = np.quantile(yv, [0.1, 0.5, 0.9])
E.check("s27p08a", all(abs(qs[xv][0] - (1 + 0.5 * xv)) < 0.02 and abs(qs[xv][2] - (4 + 2 * xv)) < 0.03 for xv in xs8),
        "分位点が 1+0.5x、2+x、4+2x になるデータで確認")
E.choice("s27p08a", f"$x=4$で消費額の10%点から90%点までの幅は{f((4 + 2 * 4) - (1 + 0.5 * 4), 1)}（千円）で、$x=1$の{f((4 + 2) - (1 + 0.5), 1)}", "分位点の差")
# (b) 10 個の独立な標準正規の最大値の期待値
emax = integrate.quad(lambda t: t * 10 * stats.norm.pdf(t) * stats.norm.cdf(t) ** 9, -10, 10)[0]
sims8 = (1 + 0.1 * rng.normal(size=(200_000, 10))).min(1)
E.check("s27p08b", abs(emax - 1.539) < 5e-4 and abs(sims8.mean() - (1 - 0.1 * emax)) < 0.002, f"E[max]={emax:.4f}、最小テスト誤差の平均 {sims8.mean():.4f}")
E.choice("s27p08b", f"1.00-1.539\\times0.10={f(1 - 0.1539, 2)}$で、選んだモデルのテスト誤差0.85は本当の予測誤差を小さく見積もっている", "期待値")

# ---- 大問9
# (a) 252 通りを総当たりして連の数を数える
runs = []
for pos in itertools.combinations(range(10), 5):
    seq = ["L"] * 10
    for p_ in pos:
        seq[p_] = "H"
    runs.append(1 + sum(seq[i] != seq[i - 1] for i in range(1, 10)))
runs = np.array(runs)
obs = "HHHLLLLLHH"
r_obs = 1 + sum(obs[i] != obs[i - 1] for i in range(1, 10))
E.check("s27p09a", r_obs == 3 and len(runs) == 252, f"観測した連の数 {r_obs}")
E.choice("s27p09a", f"${f(np.mean(runs <= 3), 3)}$。5%水準で棄却し", f"総当たり（P(R≤3)={np.mean(runs <= 3):.4f}）")
E.check("s27p09a", f(np.mean(runs <= 5), 3) == "0.357" and f(np.mean(runs == 3), 3) == "0.032" and abs(runs.mean() - 6) < 1e-12, "誤答の値と平均 6")
# (b) 定義どおりのホルム法と BH 法（statsmodels を使わない別実装）
ps = np.array([0.004, 0.011, 0.028, 0.032, 0.20, 0.64])
order = np.argsort(ps)
holm = np.zeros(6, bool)
for k, idx in enumerate(order):
    if ps[idx] <= 0.05 / (6 - k):
        holm[idx] = True
    else:
        break
adj_bh = np.minimum.accumulate((ps[order] * 6 / np.arange(1, 7))[::-1])[::-1]
bh = adj_bh <= 0.05
E.choice("s27p09b", f"ホルム法{holm.sum()}個、ベンジャミニ・ホッホベルグ法{bh.sum()}個", "調整済み p 値で BH")

# ---- 大問10 L9 直交表を作り、列 1・2 の交互作用が列 3・4 に現れることを確認
L9 = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
cols_ok = all(len(set(map(tuple, L9[:, [i, j]]))) == 9 for i in range(4) for j in range(4) if i < j)
# A×B の 4 自由度の対比（A と B の水準の組の平均の、主効果で説明できない部分）が列 3・4 の対比の張る空間に入るか
D = np.zeros((9, 9))
for k, (a, b, _, _) in enumerate(L9):
    D[k, 3 * a + b] = 1
main = np.column_stack([np.ones(9)] + [(L9[:, 0] == lv).astype(float) for lv in (1, 2)] + [(L9[:, 1] == lv).astype(float) for lv in (1, 2)])
c34 = np.column_stack([(L9[:, c] == lv).astype(float) for c in (2, 3) for lv in (1, 2)])
full = np.column_stack([main, c34])
E.check("s27p10a", cols_ok and np.linalg.matrix_rank(full) == 9 and np.linalg.matrix_rank(np.column_stack([main, D])) == 9,
        "L9 の各 2 列は直交し、主効果 A・B（5 次元）に列 3・4（4 次元）を加えると 9 通りの組をすべて表せる（A×B は列 3・4 に現れる）")
E.choice("s27p10a", "A×Bを推定したいなら第3列と第4列は空けておく必要があり", "交互作用の列")

# ---- 大問11
# (a) 固有値から
ev = np.array([2.6, 1.3, 0.95, 0.6, 0.35, 0.2])
cum = np.cumsum(ev) / ev.sum()
E.choice("s27p11a", f"カイザー基準（固有値1以上）では{(ev >= 1).sum()}個、累積寄与率80%以上の基準では{int(np.argmax(cum >= 0.8)) + 1}個", "numpy")
# (b) 回帰法：E[f|x] = λ'Σ^{-1}x、バートレット法：(λ'Ψ^{-1}λ)^{-1} λ'Ψ^{-1}x を行列で
lam = np.array([0.8, 0.7, 0.6])
Psi = np.diag(1 - lam ** 2)
Sig = np.outer(lam, lam) + Psi
xv = np.array([1.0, 0.5, 1.2])
freg = lam @ np.linalg.solve(Sig, xv)
fbar = (lam @ np.linalg.solve(Psi, xv)) / (lam @ np.linalg.solve(Psi, lam))
E.choice("s27p11b", f"回帰法{f(freg, 2)}、バートレット法{f(fbar, 2)}", "行列で計算")
fs = rng.normal(size=400_000)
Xs = np.outer(fs, lam) + rng.normal(size=(400_000, 3)) * np.sqrt(1 - lam ** 2)
coef = np.linalg.lstsq(Xs, fs, rcond=None)[0]
E.check("s27p11b", abs(coef @ xv - freg) < 0.01, f"f を x に回帰したシミュレーションの予測 {coef @ xv:.4f}（回帰法）")

# ---- 大問12 個票に展開して相関係数
T12 = np.array([[20, 10, 5], [10, 15, 10], [5, 10, 15]])
u, v = [], []
for i in range(3):
    for j in range(3):
        u += [i + 1] * T12[i, j]
        v += [j + 1] * T12[i, j]
r12 = np.corrcoef(u, v)[0, 1]
E.num("s27p12a", rnd((T12.sum() - 1) * r12 ** 2, 2), f"個票の相関係数 r={r12:.4f}")
E.check("s27p12a", stats.chi2.sf((T12.sum() - 1) * r12 ** 2, 1) < 0.001, f"p 値 {stats.chi2.sf((T12.sum() - 1) * r12 ** 2, 1):.5f}")

# ---- 大問13 ルービンのルール
Q = np.array([52.1, 50.8, 53.0, 51.5, 52.6])
B13 = Q.var(ddof=1)
T13 = 4.0 + (1 + 1 / 5) * B13
E.num("s27p13a", rnd((1 + 1 / 5) * B13 / T13, 3), f"B={B13:.4f}、T={T13:.4f}")

# ---- 大問14 データを作って回帰で確認
C = np.array([[1, 0.5, 0.4], [0.5, 1, 0.5], [0.4, 0.5, 1]])
Z = rng.multivariate_normal(np.zeros(3), C, 1_000_000)
a_ = np.linalg.lstsq(Z[:, [0]], Z[:, 1], rcond=None)[0][0]
bc = np.linalg.lstsq(Z[:, [0, 1]], Z[:, 2], rcond=None)[0]
E.check("s27p14a", abs(a_ - 0.5) < 0.005 and abs(bc[1] - 0.4) < 0.005 and abs(bc[0] - 0.2) < 0.005, f"シミュレーションの回帰 a={a_:.3f}、c'={bc[0]:.3f}、b={bc[1]:.3f}")
bb = np.linalg.solve(C[:2, :2], C[:2, 2])
E.num("s27p14a", rnd(0.5 * bb[1], 2), "正規方程式を numpy で解いた b と a の積")

E.done()
