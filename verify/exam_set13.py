"""模擬試験 第13回（set13）の検算。generator/set13.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・線形代数）で正解を求め直し、問題文に示した分布の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(13)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums_of(iid):
    out = []
    for c in E.item(iid)["choices"]:
        m = re.search(r"-?\d+\.?\d*", str(c))
        out.append(float(m.group()) if m else None)
    return out


def nearest(iid, value, what="最も近い選択肢"):
    nums = nums_of(iid)
    k = int(np.argmin([abs(x - value) if x is not None else 1e9 for x in nums]))
    E.check(iid, k == E.item(iid)["answer"], f"{what} {k}（値 {value:.6f}）が正解 {E.item(iid)['answer']}")


def distinct(iid, d):
    """選択肢の数値が丸めの桁で互いに異なる。"""
    nums = [x for x in nums_of(iid) if x is not None]
    E.check(iid, len({round(x, d) for x in nums}) == len(nums), f"選択肢の数値が互いに異なる {nums}")


def tab(name, val, true, tol):
    ok = abs(val - true) <= tol
    print(f"   表の値 {name}: 問題 {val} / 計算 {true:.5f} {'OK' if ok else 'NG'}")
    return ok


# ---- 大問1 ランダム化回答法
# (a) 回答者を 1 人ずつシミュレーション（該当・サイコロ・回答）して P(該当 | はい)
M = 2_000_000
T = rng.random(M) < 0.15
die = rng.integers(1, 7, size=M)
ans = np.where(die <= 4, T, die == 5)
sim = T[ans].mean()
exact = Fr(15, 100) * Fr(5, 6) / (Fr(15, 100) * Fr(5, 6) + Fr(85, 100) * Fr(1, 6))
E.choice("s13p01a", f"${f(float(exact), 3)}$", "分数で厳密に計算した P(該当|はい)")
E.check("s13p01a", abs(sim - float(exact)) < 0.003, f"シミュレーション {sim:.4f} ≈ {float(exact):.4f}")
distinct("s13p01a", 3)
# (b) 推定量の分散を二項分布の厳密な分散から（p を p̂ で置き換え）／シミュレーションで不偏性
ph = 186 / 600
var_pi = stats.binom(600, ph).var() / 600 ** 2 * (1 / (2 / 3)) ** 2
E.num("s13p01b", float(f(np.sqrt(var_pi), 4)), "scipy binom の分散 ×(3/2)^2 の平方根")
pi_true = 0.2
Ys = rng.binomial(600, (2 / 3) * pi_true + 1 / 6, size=400_000)
est = (Ys / 600 - 1 / 6) / (2 / 3)
E.check("s13p01b", abs(est.mean() - pi_true) < 0.0005, f"π=0.2 のシミュレーションで推定量の平均 {est.mean():.5f}（不偏）")
p_ = (2 / 3) * pi_true + 1 / 6
E.check("s13p01b", abs(est.std() - 1.5 * np.sqrt(p_ * (1 - p_) / 600)) < 0.0003, f"推定量の標準偏差 {est.std():.5f} が式と一致")

# ---- 大問2 世帯
P = np.array([[0.38, 0.22], [0.14, 0.06], [0.16, 0.04]])
xs = np.array([0, 1, 2])
E.check("s13p02a", abs(P.sum() - 1) < 1e-12, "同時確率の合計が 1")
cond = P[:, 1] / P[:, 1].sum()
E.choice("s13p02a", f"${f(float(cond @ xs), 4)}$", "条件付き分布のベクトル計算")
distinct("s13p02a", 4)
# (b) 同時分布から 200 万世帯を発生させて標本相関、さらに厳密値
idx = rng.choice(6, size=2_000_000, p=P.ravel())
X = idx // 2
Y = idx % 2
rs = np.corrcoef(X, Y)[0, 1]
xx, yy = np.meshgrid(xs, [0, 1], indexing="ij")
mx, my = (P * xx).sum(), (P * yy).sum()
cov = (P * (xx - mx) * (yy - my)).sum()
rho = cov / np.sqrt((P * (xx - mx) ** 2).sum() * (P * (yy - my) ** 2).sum())
E.choice("s13p02b", f"${f(rho, 3)}$", "中心化した積率で相関係数")
E.check("s13p02b", abs(rs - rho) < 0.003, f"シミュレーションの相関 {rs:.4f}")
distinct("s13p02b", 3)

# ---- 大問3 パレート
xm, a = 200, 3
par = stats.pareto(a, scale=xm)
Elog = integrate.quad(lambda x: np.log(x) * par.pdf(x), xm, np.inf)[0]
gm_exact = np.exp(Elog)
E.check("s13p03a", tab("e^(1/3)", 1.3956, np.exp(1 / 3), 5e-5) and tab("e^(1/4)", 1.2840, np.exp(0.25), 5e-5)
        and tab("e^(1/2)", 1.6487, np.exp(0.5), 5e-5) and tab("2^(1/3)", 1.2599, 2 ** (1 / 3), 5e-5), "e^x と 2^(1/3) の値")
E.choice("s13p03a", f"${f(xm * 1.3956, 1)}$", "示した e^{1/3} を使った幾何平均")
nearest("s13p03a", gm_exact, "数値積分の幾何平均に最も近い選択肢")
Ys = np.log(par.rvs(size=500_000, random_state=1) / xm)
E.check("s13p03a", stats.kstest(Ys, stats.expon(scale=1 / a).cdf).pvalue > 0.01, "log(X/x_m) が指数分布(レート3)と矛盾しない（KS 検定）")
distinct("s13p03a", 1)
E.check("s13p03b", par.mean() == 300 and abs(par.var() - 30000) < 1e-6, "scipy pareto の平均 300・分散 30000（問題文の値）")
cheb = par.var() / (900 - par.mean()) ** 2
E.choice("s13p03b", f"${f(cheb, 3)}$", "scipy の分散からチェビシェフの上界")
E.check("s13p03b", par.sf(900) <= cheb, f"真の値 {par.sf(900):.4f} が上界以下")
distinct("s13p03b", 3)

# ---- 大問4 世論調査
obs = np.array([336, 396, 468])
res = stats.chisquare(obs, 1200 * np.array([0.30, 0.34, 0.36]))
ok = tab("chi2_0.05(1)", 3.841, stats.chi2.ppf(0.95, 1), 5e-4) and tab("chi2_0.05(2)", 5.991, stats.chi2.ppf(0.95, 2), 5e-4) \
    and tab("chi2_0.05(3)", 7.815, stats.chi2.ppf(0.95, 3), 5e-4)
E.check("s13p04a", ok, "カイ二乗分布の 5% 点")
E.check("s13p04a", "棄却しない" in E.item("s13p04a")["choices"][E.item("s13p04a")["answer"]] and f(res.statistic, 2) in E.item("s13p04a")["choices"][2],
        f"scipy chisquare: 統計量 {res.statistic:.4f}、p 値 {res.pvalue:.4f} > 0.05")
E.check("s13p04a", res.pvalue > 0.05, "p 値 > 0.05 で棄却しない")
# (b) 多項分布の共分散行列から
pv = np.array([0.42, 0.38, 0.20])
Cov = 1200 * (np.diag(pv) - np.outer(pv, pv))
w = np.array([1, -1, 0])
E.num("s13p04b", float(f(np.sqrt(w @ Cov @ w), 2)), "多項分布の共分散行列 n(diag(p)−pp') による w'Σw")
sim = rng.multinomial(1200, pv, size=400_000)
sd_sim = (sim[:, 0] - sim[:, 1]).std()
E.check("s13p04b", abs(sd_sim - 30.95) < 0.15, f"シミュレーションの標準偏差 {sd_sim:.3f}")
# (c) 共分散行列に標本比率を入れてデルタ法なしの直接計算
ph = np.array([516, 444, 240]) / 1200
C = (np.diag(ph) - np.outer(ph, ph)) / 1200
se = np.sqrt(w @ C @ w)
z = stats.norm.ppf(0.975)
E.check("s13p04c", tab("z_0.025", 1.960, z, 5e-4) and tab("z_0.05", 1.645, stats.norm.ppf(0.95), 5e-4)
        and tab("z_0.005", 2.576, stats.norm.ppf(0.995), 5e-4), "標準正規分布の点")
E.choice("s13p04c", f"${f(0.06 - 1.96 * se, 4)}$", "標本比率の多項共分散行列から下限")
distinct("s13p04c", 4)

# ---- 大問5
d_t = 3.1 / (4.2 / np.sqrt(12))
E.choice("s13p05a", f"${f(d_t, 2)}$", "numpy の √12 で t 統計量")
# 平均 3.1・標準偏差 4.2 になるデータを作って scipy の ttest_1samp / ttest_rel
z0 = rng.standard_normal(12)
z0 = (z0 - z0.mean()) / z0.std(ddof=1)
dd = 3.1 + 4.2 * z0
after = rng.normal(20, 3, 12)
before = after + dd
E.check("s13p05a", abs(stats.ttest_rel(before, after).statistic - d_t) < 1e-9, "scipy ttest_rel と一致")
distinct("s13p05a", 2)
pv5 = np.array([0.004, 0.011, 0.019, 0.022, 0.41])
# ホルムの調整済み p 値：max_{j<=k} min(1, (m−j+1) p_(j))（手続きを止める規則と同値な別表現）
order = np.argsort(pv5)
adj = np.maximum.accumulate(np.minimum(1, (5 - np.arange(5)) * pv5[order]))
rej = np.zeros(5, bool)
rej[order] = adj <= 0.05
names = [n for n, r_ in zip("ABCDE", rej) if r_]
E.check("s13p05b", names == ["A", "B"], f"調整済み p 値によるホルム法で棄却 {names}")
E.check("s13p05b", E.item("s13p05b")["choices"][E.item("s13p05b")["answer"]] == "A と B", "正解の選択肢が A と B")
bon = [n for n, r_ in zip("ABCDE", np.minimum(1, 5 * pv5) <= 0.05) if r_]
E.check("s13p05b", bon == ["A"], f"ボンフェローニでは {bon}（誤答 0 と一致）")

# ---- 大問6 マンテル・ヘンツェル
tabs = [np.array([[180, 120], [150, 150]]), np.array([[50, 150], [80, 520]])]
# MH 推定量を「層ごとのオッズ比の重み付き平均（重み b_k c_k / n_k）」として計算
ors = np.array([t_[0, 0] * t_[1, 1] / (t_[0, 1] * t_[1, 0]) for t_ in tabs])
wts = np.array([t_[0, 1] * t_[1, 0] / t_.sum() for t_ in tabs])
mh = (wts * ors).sum() / wts.sum()
E.choice("s13p06a", f"${f(mh, 2)}$", "層別オッズ比の重み付き平均としての MH オッズ比")
# 共通オッズ比の条件付き最尤推定（非心超幾何）も近い値になることを確認
def cll(psi):
    s_ = 0.0
    for t_ in tabs:
        n1, m1, N = t_[0].sum(), t_[:, 0].sum(), t_.sum()
        s_ += stats.nchypergeom_fisher(N, n1, m1, psi).logpmf(t_[0, 0])
    return -s_
from scipy.optimize import minimize_scalar  # noqa: E402
cmle = minimize_scalar(cll, bounds=(0.5, 5), method="bounded").x
E.check("s13p06a", abs(cmle - mh) < 0.05, f"条件付き最尤推定 {cmle:.4f} も近い値")
distinct("s13p06a", 2)

# ---- 大問7 吸収マルコフ連鎖：基本行列 N=(I−Q)^{-1}
Q = sp.Matrix([[sp.Rational(87, 100), sp.Rational(5, 100)], [sp.Rational(35, 100), sp.Rational(50, 100)]])
Nf = (sp.eye(2) - Q).inv()
tE = float((Nf * sp.Matrix([1, 1]))[0])
E.num("s13p07a", float(f(tE, 2)), "基本行列 (I−Q)^{-1} の行和（sympy）")
# シミュレーション
Pm = np.array([[0.87, 0.05, 0.08], [0.35, 0.50, 0.15]])
st = np.zeros(200_000, dtype=int)
steps = np.zeros(200_000)
alive = np.ones(200_000, bool)
while alive.any():
    u = rng.random(alive.sum())
    cur = st[alive]
    cum = Pm[cur].cumsum(axis=1)
    nxt = (u[:, None] > cum).sum(axis=1)
    steps[alive] += 1
    st[alive] = nxt
    alive[alive] = nxt < 2
E.check("s13p07a", abs(steps.mean() - tE) < 0.08, f"シミュレーションの平均 {steps.mean():.3f}")

# ---- 大問8
F05, F01 = stats.f.ppf(0.95, 2, 35), stats.f.ppf(0.99, 2, 35)
E.check("s13p08a", tab("F_0.05(2,35)", 3.267, F05, 5e-4) and tab("F_0.01(2,35)", 5.268, F01, 5e-4), "F 分布の点")
Fv = ((2.40 - 1.80) / 2) / (1.80 / 35)
E.check("s13p08a", f"F={f(Fv, 2)}" in re.sub(r"\s", "", E.item("s13p08a")["choices"][2]) and Fv > F01, f"F={Fv:.4f} > 1% 点")
E.check("s13p08a", stats.f.sf(Fv, 2, 35) < 0.01, f"p 値 {stats.f.sf(Fv, 2, 35):.4f}")
# (b) 実際にデータを作って Cp を計算（RSS をちょうど合わせた人工データ）
n = 40
Xd = np.column_stack([np.ones(n), rng.standard_normal((n, 4))])
H = Xd @ np.linalg.pinv(Xd)
e = (np.eye(n) - H) @ rng.standard_normal(n)
e *= np.sqrt(1.80 / (e @ e))
Xr = Xd[:, :3]
# y = Xd b + e とし、縮小モデルの RSS が 2.40 になるように b の後半を調整
Hr = Xr @ np.linalg.pinv(Xr)
v = (np.eye(n) - Hr) @ Xd[:, 3:] @ np.array([1.0, 0.5])
# 縮小モデルの残差は (I−Hr)(Xd b + e) = c v + (I−Hr) e。e ⟂ 列空間(Xd) なので (I−Hr)e = e、v ⟂ e
cc = np.sqrt((2.40 - 1.80) / (v @ v))
y = Xd[:, :3] @ np.array([1.5, 0.2, -0.1]) + cc * Xd[:, 3:] @ np.array([1.0, 0.5]) + e
rss = lambda A: float(((np.eye(n) - A @ np.linalg.pinv(A)) @ y) @ ((np.eye(n) - A @ np.linalg.pinv(A)) @ y))
rf, rr = rss(Xd), rss(Xr)
E.check("s13p08b", abs(rf - 1.80) < 1e-9 and abs(rr - 2.40) < 1e-9, f"人工データの RSS {rf:.4f} / {rr:.4f}")
cp = rr / (rf / (n - 5)) - n + 2 * 3
E.num("s13p08b", float(f(cp, 2)), "人工データの回帰から Cp")
E.check("s13p08b", abs(cp - (3 + (Fv - 1) * 2)) < 1e-9, "Cp = p + q(F−1) の関係とも一致")
E.check("s13p08c", tab("chi2_0.05(2)", 5.991, stats.chi2.ppf(0.95, 2), 5e-4) and tab("chi2_0.01(2)", 9.210, stats.chi2.ppf(0.99, 2), 5e-4)
        and tab("chi2_0.05(4)", 9.488, stats.chi2.ppf(0.95, 4), 5e-4) and tab("chi2_0.05(1)", 3.841, stats.chi2.ppf(0.95, 1), 5e-4), "カイ二乗分布の点")
lm = 40 * 0.21
E.check("s13p08c", stats.chi2.sf(lm, 2) < 0.05 and stats.chi2.sf(lm, 2) > 0.01, f"nR²={lm}, p 値 {stats.chi2.sf(lm, 2):.4f}（5% で棄却・1% で棄却しない）")
E.choice("s13p08c", "頑健な標準誤差", "正解の選択肢が対処の記述")
# 帰無仮説（等分散）のもとで nR² が自由度 2 のカイ二乗分布に近いことをシミュレーションで確認
cnt = 0
NS = 20000
Z = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
Pz = Z @ np.linalg.pinv(Z)
for _ in range(NS):
    e2 = rng.standard_normal(n) ** 2
    fit = Pz @ e2
    r2 = ((fit - e2.mean()) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    cnt += n * r2 > 5.991
E.check("s13p08c", abs(cnt / NS - 0.05) < 0.015, f"等分散のもとで nR² > 5.991 となる割合 {cnt / NS:.4f}（自由度 2 で約 5%）")

# ---- 大問9
E.check("s13p09a", tab("chi2_0.05(3)", 7.815, stats.chi2.ppf(0.95, 3), 5e-4) and tab("chi2_0.01(3)", 11.345, stats.chi2.ppf(0.99, 3), 5e-4)
        and tab("chi2_0.01(2)", 9.210, stats.chi2.ppf(0.99, 2), 5e-4), "カイ二乗分布の点")
lr = 1012.6 - 1001.9
pv9 = stats.chi2.sf(lr, 3)
E.check("s13p09a", 0.01 < pv9 < 0.05, f"LR={lr:.1f}、自由度 3 の p 値 {pv9:.4f}")
E.choice("s13p09a", "自由度は3で、有意水準5%では棄却するが1%では棄却しない", "正解の記述")
vot = [0.81, 0.66, 0.58, 0.40]
non = [0.62, 0.45, 0.40, 0.33, 0.21]
# しきい値を動かして ROC 曲線の点を作り、台形公式で面積を求める
thr = sorted(set(vot + non), reverse=True)
roc = [(0.0, 0.0)] + [(np.mean([u >= c for u in non]), np.mean([v_ >= c for v_ in vot])) for c in thr]
auc = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(roc, roc[1:]))
E.num("s13p09b", float(f(auc, 3)), "ROC 曲線の点から台形公式で面積")
U = stats.mannwhitneyu(vot, non).statistic / 20
E.check("s13p09b", abs(U - auc) < 1e-12, f"マン・ホイットニーの U/20 = {U}")

# ---- 大問10
q = stats.studentized_range.ppf(0.95, 3, 27)
E.check("s13p10a", tab("q_0.05(3,27)", 3.506, q, 1e-3) and tab("t_0.025(27)", 2.052, stats.t.ppf(0.975, 27), 5e-4), "q と t の点")
hsd = q * np.sqrt(4.5 / 10)
sig = [pr for pr, dlt in (("BA", 2.0), ("CA", 2.9), ("CB", 0.9)) if dlt >= hsd]
E.check("s13p10a", sig == ["CA"], f"scipy の q から HSD={hsd:.4f}、有意 {sig}")
E.choice("s13p10a", f"臨界値は{f(3.506 * np.sqrt(0.45), 2)}で、CとAの組だけ", "正解の記述")
# 正解の主張に対応するシミュレーション：3 群とも同じ平均のとき最大差が HSD を超える確率 ≈ 5%
sims = rng.normal(0, np.sqrt(4.5), size=(100_000, 3, 10))
mm = sims.mean(axis=2)
mse_s = ((sims - mm[:, :, None]) ** 2).sum(axis=(1, 2)) / 27
rate = ((mm.max(1) - mm.min(1)) >= 3.506 * np.sqrt(mse_s / 10)).mean()
E.check("s13p10a", abs(rate - 0.05) < 0.004, f"帰無仮説のもとの誤り率 {rate:.4f} ≈ 0.05")
# (b) 2 段階最小二乗法を個票の人工データで再現
Nn = 500
z_ = np.r_[np.ones(Nn), np.zeros(Nn)]
d_ = np.r_[np.r_[np.ones(340), np.zeros(160)], np.r_[np.ones(55), np.zeros(445)]]
yv = np.r_[np.full(Nn, 5.46), np.full(Nn, 3.72)] + 0.0
X1 = np.column_stack([np.ones(2 * Nn), z_])
dhat = X1 @ np.linalg.lstsq(X1, d_, rcond=None)[0]
b2 = np.linalg.lstsq(np.column_stack([np.ones(2 * Nn), dhat]), yv, rcond=None)[0][1]
E.num("s13p10b", float(f(b2, 2)), "個票を作って 2 段階最小二乗法")
E.check("s13p10b", abs(np.mean(d_[:Nn]) - 0.68) < 1e-12 and abs(np.mean(d_[Nn:]) - 0.11) < 1e-12, "人工データの受講率が表と一致")
mp, mn = 6.26, (4590 - 395 * 6.26) / 605
E.check("s13p10b", abs((mp - mn) - 2.76) < 0.006, f"単純比較の差 {mp - mn:.4f} ≈ 2.76（問題文の「約」の値と整合）")

# ---- 大問11 HT
xs11 = np.array([120000, 50000, 30000])
ys11 = np.array([84, 33, 16])
pis = 3 * xs11 / 2_000_000
E.check("s13p11a", (pis < 1).all(), "包含確率がすべて 1 未満")
E.choice("s13p11a", f"${f((ys11 / pis).sum(), 0)}$", "ベクトル計算 Σ y/π")
# 同じ包含確率をもつ系統 PPS 抽出で HT が不偏になることを小さな母集団で確認
pop_x = np.array([120, 50, 30, 200, 90, 160, 70, 80, 110, 90], float)
pop_y = rng.poisson(pop_x * 0.6)
pi_ = 3 * pop_x / pop_x.sum()
cum = np.r_[0, np.cumsum(pi_)]
starts = (np.arange(20000) + 0.5) / 20000
ests = []
for s0 in starts:
    pts = s0 + np.arange(3)
    sel = np.searchsorted(cum, pts, side="right") - 1
    ests.append((pop_y[sel] / pi_[sel]).sum())
E.check("s13p11a", abs(np.mean(ests) - pop_y.sum()) < 0.01 * pop_y.sum(), f"系統 PPS での HT の平均 {np.mean(ests):.1f} ≈ 真の総数 {pop_y.sum()}")
distinct("s13p11a", 0)

# ---- 大問12
pts = {"A": (-1.4, 0.7), "B": (-0.6, 1.0), "C": (-1.0, -0.2), "D": (0.4, -0.4), "E": (1.2, -1.4), "F": (1.4, 0.4)}
Pp = np.array(list(pts.values()))
cen = np.array([pts["B"], pts["C"]])
lab1 = np.argmin(((Pp[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
cen2 = np.array([Pp[lab1 == k].mean(0) for k in range(2)])
lab2 = np.argmin(((Pp[:, None, :] - cen2[None]) ** 2).sum(-1), axis=1)
changed = [k for k, a_, b_ in zip(pts, lab1, lab2) if a_ != b_]
E.check("s13p12a", changed == ["C", "F"], f"厳密な中心で 2 回目に所属が変わるのは {changed}")
E.check("s13p12a", np.allclose(cen2, [[-0.2, 0.7], [0.2, -2 / 3]]), "1 回目の更新後の中心 (−0.2, 0.7), (0.2, −2/3)")
E.check("s13p12a", sorted(np.bincount(lab2)) == [3, 3], "2 回目後のクラスターの大きさは 3 と 3")
wss = lambda lab: sum(((Pp[lab == k] - Pp[lab == k].mean(0)) ** 2).sum() for k in range(2))
E.check("s13p12a", wss(lab2) < wss(lab1), f"クラスター内平方和は減る {wss(lab1):.3f} → {wss(lab2):.3f}")
E.check("s13p12a", E.item("s13p12a")["answer"] == 2, "正解は選択肢 2（C と F が入れ替わる）")
Om = np.array([[2.0, -0.8, 0, 0.5], [-0.8, 2.5, -0.9, 0], [0, -0.9, 2.2, -0.7], [0.5, 0, -0.7, 1.8]])
E.check("s13p12b", np.all(np.linalg.eigvalsh(Om) > 0), "精度行列が正定値")
S = np.linalg.inv(Om)
# 条件付き分布の共分散（シューア補）から X1, X3 | X2, X4 の共分散が 0 になることを確認
i, j = [0, 2], [1, 3]
condS = S[np.ix_(i, i)] - S[np.ix_(i, j)] @ np.linalg.inv(S[np.ix_(j, j)]) @ S[np.ix_(j, i)]
E.check("s13p12b", abs(condS[0, 1]) < 1e-12, "Σ のシューア補で X1,X3 | X2,X4 の条件付き共分散が 0（正解の記述が正しい）")
E.check("s13p12b", abs(S[0, 2]) > 0.01, f"σ13={S[0, 2]:.4f} ≠ 0（選択肢 0 は誤り）")
i, j = [0, 3], [1, 2]
condS = S[np.ix_(i, i)] - S[np.ix_(i, j)] @ np.linalg.inv(S[np.ix_(j, j)]) @ S[np.ix_(j, i)]
pc = condS[0, 1] / np.sqrt(condS[0, 0] * condS[1, 1])
E.check("s13p12b", pc < 0, f"X1,X4 | X2,X3 の偏相関 {pc:.4f} < 0（選択肢 1 は誤り）")
i, j = [1, 3], [0, 2]
condS = S[np.ix_(i, i)] - S[np.ix_(i, j)] @ np.linalg.inv(S[np.ix_(j, j)]) @ S[np.ix_(j, i)]
E.check("s13p12b", abs(condS[0, 1]) < 1e-12, "X2,X4 | X1,X3 は条件付き独立（選択肢 4 は誤り）")
E.check("s13p12b", np.all(np.abs(S[np.triu_indices(4, 1)]) > 0.01) and np.sum(np.abs(Om[np.triu_indices(4, 1)]) > 0) == 4,
        "Σ の非対角はすべて 0 でないが辺は 4 本（選択肢 3 は誤り）")
E.check("s13p12b", np.allclose(np.round(np.linalg.inv(np.round(S, 3)), 1), Om, atol=0.15), "丸めた Σ の逆行列が Ω にほぼ一致")

# ---- 大問13 AR(2)：ユール・ウォーカーを行列で解き、さらに AR(2) の理論自己相関に戻す
R = np.array([[1, 0.48], [0.48, 1]])
phi = np.linalg.solve(R, [0.48, 0.06])
E.num("s13p13a", float(f(phi[0], 3)), "numpy.linalg.solve で Yule–Walker")
# 推定した係数の AR(2) の理論自己相関を差分方程式で再計算し、長い系列の標本偏自己相関とも比べる
rho1 = phi[0] / (1 - phi[1])
rho2 = phi[0] * rho1 + phi[1]
roots = np.roots([-phi[1], -phi[0], 1])
E.check("s13p13a", abs(rho1 - 0.48) < 1e-9 and abs(rho2 - 0.06) < 1e-9 and np.all(np.abs(roots) > 1), f"理論自己相関 {rho1:.4f}, {rho2:.4f}、特性根の絶対値 > 1（定常）")
g = np.random.default_rng(5)
N_ = 300_000
eps = g.standard_normal(N_ + 200)
xsim = np.zeros(N_ + 200)
for t_ in range(2, N_ + 200):
    xsim[t_] = phi[0] * xsim[t_ - 1] + phi[1] * xsim[t_ - 2] + eps[t_]
xsim = xsim[200:] - xsim[200:].mean()
c0 = xsim @ xsim
s1, s2 = (xsim[1:] @ xsim[:-1]) / c0, (xsim[2:] @ xsim[:-2]) / c0
pacf2 = (s2 - s1 ** 2) / (1 - s1 ** 2)
E.check("s13p13a", abs(pacf2 - phi[1]) < 0.01, f"大標本の偏自己相関（ラグ2）{pacf2:.4f} ≈ φ2 {phi[1]:.4f}")

# ---- 大問14 ベータ・二項
post = stats.beta(15, 20)
pred = integrate.quad(lambda t: t ** 2 * post.pdf(t), 0, 1)[0]
E.num("s13p14a", float(f(pred, 3)), "事後分布で θ² を数値積分")
bb = stats.betabinom(2, 15, 20).pmf(2)
E.check("s13p14a", abs(bb - pred) < 1e-12, f"scipy betabinom(2,15,20).pmf(2) = {bb:.6f}")
th = rng.beta(6, 9, size=3_000_000)
k = rng.binomial(20, th)
sel = th[k == 9]
nxt = rng.binomial(2, sel)
E.check("s13p14a", abs((nxt == 2).mean() - pred) < 0.004, f"事前から生成して 9/20 の条件付きシミュレーション {(nxt == 2).mean():.4f}")

E.done()
