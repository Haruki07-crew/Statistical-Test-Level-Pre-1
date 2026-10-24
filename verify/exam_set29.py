"""模擬試験 第29回（set29）の検算。generator/set29.py とは別の方法（scipy の分布関数・シミュレーション・
数値積分・総当たり・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import integrate, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(29)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd(x, d):
    return float(f(x, d))


# ---- 大問1
# (a) ベイズの定理を確率のまま計算
post = 0.10 * 0.80 / (0.10 * 0.80 + 0.90 * 0.10)
E.choice("s29p01a", f"${f(post, 3)}$", "ベイズの定理（確率の形）")
sim = rng.random(2_000_000) < 0.10
pos = np.where(sim, rng.random(sim.size) < 0.80, rng.random(sim.size) < 0.10)
E.check("s29p01a", abs(sim[pos].mean() - post) < 0.003, f"シミュレーション {sim[pos].mean():.4f}")
# (b) 半正規分布
hn = stats.halfnorm(scale=0.3)
E.choice("s29p01b", f"$E|X|={f(hn.mean(), 3)}$、$P(|X|>0.3)={f(2 * (1 - 0.8413), 4)}$", "scipy halfnorm の平均と、表の Φ(1) を使った両側確率")
E.check("s29p01b", abs(hn.sf(0.3) - 0.3174) < 2e-4, f"正確な P(|X|>σ)={hn.sf(0.3):.5f}")
E.check("s29p01b", abs(math.sqrt(2 / math.pi) - 0.7979) < 5e-5 and abs(stats.norm.cdf(1) - 0.8413) < 5e-5, "√(2/π) と Φ(1)")

# ---- 大問2
# (a) 行列で相関行列
S2 = np.array([[64, 30, 24], [30, 25, 15], [24, 15, 36]], float)
Dm = np.diag(1 / np.sqrt(np.diag(S2)))
R2 = Dm @ S2 @ Dm
E.num("s29p02a", rnd(R2[1, 2], 2), "D^{-1/2} Σ D^{-1/2}")
# (b) シミュレーション：y に独立な誤差を足して決定係数
x = rng.normal(size=1_000_000)
y = x + rng.normal(size=x.size)              # 真の R²=0.5
ystar = y + rng.normal(0, math.sqrt(0.5), x.size)   # 信頼性 V[y]/V[y*]=2/2.5=0.8
r2s = np.corrcoef(x, ystar)[0, 1] ** 2
E.num("s29p02b", rnd(0.5 * 0.8, 2), "R²×信頼性")
E.check("s29p02b", abs(r2s - 0.4) < 0.003 and abs(np.polyfit(x, ystar, 1)[0] - 1) < 0.005, f"シミュレーションの決定係数 {r2s:.4f}、傾きは変わらない")

# ---- 大問3 シミュレーションと正確な和
T = rng.exponential(20, 2_000_000)
K = np.floor(T / 7)
q = math.exp(-0.35)
E.check("s29p03a", abs(K.mean() - q / (1 - q)) < 0.01 and abs(np.mean(K == 0) - (1 - q)) < 0.002 and abs(q - 0.7047) < 5e-5,
        f"シミュレーションの E[K]={K.mean():.4f}、P(K=0)={np.mean(K == 0):.4f}")
E.choice("s29p03a", f"$E[K]={f(0.7047 / 0.2953, 2)}$", "q/(1−q)")

# ---- 大問4 ベータ二項分布（scipy の betabinom）
bb = stats.betabinom(4, 1, 3)
E.choice("s29p04a", f"$P(X=0)={f(bb.pmf(0), 3)}$、$V[X]={f(bb.var(), 2)}$。二項分布（$P(X=0)={f(stats.binom.pmf(0, 4, 0.25), 3)}$", "scipy betabinom")
th = rng.beta(1, 3, 1_000_000)
X4 = rng.binomial(4, th)
E.check("s29p04a", abs(np.mean(X4 == 0) - 3 / 7) < 0.002 and abs(X4.var() - 1.2) < 0.01, f"シミュレーション P(X=0)={np.mean(X4 == 0):.4f}、分散 {X4.var():.4f}")

# ---- 大問5
# (a) 等相関の正規乱数で平均の標準偏差
C5 = 100 * (0.1 * np.ones((25, 25)) + 0.9 * np.eye(25))
Z5 = rng.multivariate_normal(np.zeros(25), C5, 200_000)
E.num("s29p05a", rnd(math.sqrt(np.ones(25) @ C5 @ np.ones(25) / 625), 2), "1'Σ1/n² を行列で")
E.check("s29p05a", abs(Z5.mean(1).std() - 3.688) < 0.02, f"シミュレーション {Z5.mean(1).std():.4f}")
# (b) シミュレーションで V[s²]、MLE の MSE
X5 = rng.normal(0, 1, (400_000, 10))
s2 = X5.var(1, ddof=1)
mle = X5.var(1)
E.check("s29p05b", abs(s2.var() - 2 / 9) < 0.003 and abs(np.mean((mle - 1) ** 2) - 0.19) < 0.003, f"V[s²]={s2.var():.4f}、MLE の MSE={np.mean((mle - 1) ** 2):.4f}")
# フィッシャー情報量を数値微分で：σ² についての 1 個あたりの情報量 1/(2σ⁴)
sig2 = 1.0
ll = lambda v, xx: -0.5 * np.log(2 * np.pi * v) - xx ** 2 / (2 * v)
h = 1e-4
xx = rng.normal(0, 1, 2_000_000)
info = -np.mean((ll(sig2 + h, xx) - 2 * ll(sig2, xx) + ll(sig2 - h, xx)) / h ** 2)
E.check("s29p05b", abs(info - 0.5) < 0.01, f"数値微分のフィッシャー情報量 {info:.4f}（1/(2σ⁴)=0.5）")
E.choice("s29p05b", "V[s^2]=0.222$、下限$0.200$、最尤推定量の平均二乗誤差0.190。最尤推定量は下限を下回るが", "シミュレーション")

# ---- 大問6 尤度比の区間を数値で解く
G = lambda l: 2 * (4 * math.log(4 / l) - 4 + l)
from scipy.optimize import brentq  # noqa: E402
lo6, hi6 = brentq(lambda l: G(l) - 3.84, 0.5, 4), brentq(lambda l: G(l) - 3.84, 4, 30)
E.check("s29p06a", abs(lo6 - 1.24) < 0.01 and abs(hi6 - 9.29) < 0.01, f"尤度にもとづく区間 ({lo6:.3f}, {hi6:.3f})")
E.check("s29p06a", abs(stats.chi2.ppf(0.025, 8) / 2 - 1.09) < 0.005 and abs(stats.chi2.ppf(0.975, 10) / 2 - 10.24) < 0.005, "正確な区間 (1.09, 10.24)")
E.choice("s29p06a", f"約$({f(lo6, 1)},\\{f(hi6, 1)})$。ワルド型", "brentq")

# ---- 大問7
# (a) 非劣性の z と、95% 区間の下限
z7 = (-1.0 + 3.0) / 0.8
E.choice("s29p07a", f"$z=(-1.0+3.0)/0.8={f(z7, 2)}>1.96$で棄却し、非劣性が示された", "定義どおり")
E.check("s29p07a", (-1.0 - 1.96 * 0.8) > -3.0, "95% 区間の下限 −2.568 が −3 より上")
# (b) numpy の逆行列で T²、F の点
d7 = np.array([4.0, -3.0])
S7 = np.array([[100, 48], [48, 64.0]])
T2 = 10 * d7 @ np.linalg.solve(S7, d7)
F7 = 37 / 76 * T2
E.check("s29p07b", F7 > stats.f.isf(0.05, 2, 37) and abs(stats.f.isf(0.05, 2, 37) - 3.25) < 0.005 and abs(stats.t.ppf(0.975, 38) - 2.024) < 5e-4, f"T²={T2:.4f}、F={F7:.4f}")
E.choice("s29p07b", f"$T^2={f(T2, 2)}$、$F={f(F7, 2)}$で有意。1変数ずつの$t$は{f(4 / math.sqrt(10), 2)}と{f(3 / math.sqrt(6.4), 2)}", "numpy solve")

# ---- 大問8
tab8 = np.array([[6, 14], [14, 6]])
E.choice("s29p08a", f"$X^2={f(stats.chi2_contingency(tab8, correction=False)[0], 1)}$で3.84を上回り、2つの職場で睡眠時間の分布の位置が違う", "scipy chi2_contingency")
med = stats.median_test(np.r_[np.arange(6) + 100, np.arange(14) - 100], np.r_[np.arange(14) + 100, np.arange(6) - 100], correction=False)
E.check("s29p08a", abs(med.statistic - 6.4) < 1e-9, f"scipy median_test の統計量 {med.statistic:.3f}")

# ---- 大問9 ランダムウォークのシミュレーション
hits = []
for _ in range(100_000):
    s_ = 0
    while -2 < s_ < 3:
        s_ += 1 if rng.random() < 0.5 else -1
    hits.append(s_ == 3)
E.check("s29p09a", abs(np.mean(hits) - 0.4) < 0.005, f"シミュレーション {np.mean(hits):.4f}")
E.choice("s29p09a", "0.4。$S_t$はマルチンゲールなので", "任意抽出定理")

# ---- 大問10
# (a) 右に歪んだ残差の QQ プロットの形（対数正規の分位点と正規の分位点の差）
pp = (np.arange(1, 200) - 0.5) / 199
qn = stats.norm.ppf(pp)
qs = stats.lognorm(0.8).ppf(pp)
qs = (qs - qs.mean()) / qs.std()
resid = qs - np.polyval(np.polyfit(qn, qs, 1), qn)
E.check("s29p10a", resid[0] > 0 and resid[-1] > 0 and resid[100] < 0, "右に歪んだ分布の QQ プロットは両端が直線より上（下に凸）")
E.choice("s29p10a", "右に歪んでいる", "定義の確認")
# (b) 比例オッズ：累積確率から各段階の確率
Fl = lambda u: 1 / (1 + math.exp(-u))
p3 = lambda x_: 1 - Fl(1.5 - 0.8 * x_)
odds_ratio = [(1 - Fl(c - 0.8)) / Fl(c - 0.8) / ((1 - Fl(c)) / Fl(c)) for c in (-1.0, 1.5)]
E.check("s29p10b", all(abs(o - math.exp(0.8)) < 1e-12 for o in odds_ratio) and abs(math.exp(0.7) - 2.0138) < 5e-5, f"累積オッズ比 {odds_ratio}")
E.choice("s29p10b", f"{f(p3(1), 3)}。「悪い以上」と「ふつう以上」のどちらの区切りでも、運動習慣がない人の累積オッズは", "logistic の式")
# (c) 主成分回帰の係数の分散をシミュレーションで
lam10 = np.array([2.20, 0.75, 0.05])
Zp = rng.normal(size=(60, 3)) * np.sqrt(lam10)
Zp = Zp - Zp.mean(0)
Zp = Zp / np.sqrt((Zp ** 2).sum(0) / 59) * np.sqrt(lam10)        # 標本分散を固有値に合わせる
Zp = np.linalg.qr(Zp)[0] * np.sqrt(59 * lam10)                    # 列を直交化（主成分得点）
gam = []
for _ in range(20000):
    yy = Zp @ np.array([1.0, 0.5, 0.2]) + rng.normal(size=60)
    gam.append(np.linalg.lstsq(Zp, yy, rcond=None)[0])
gam = np.array(gam)
E.check("s29p10c", np.allclose(gam.var(0) * 59, 1 / lam10, rtol=0.05), f"係数の分散 × (n−1) {np.round(gam.var(0) * 59, 3)}（1/λ={np.round(1 / lam10, 3)}）")
E.choice("s29p10c", f"すべて入れると{f((1 / lam10).sum(), 2)}、第3主成分を除くと{f((1 / lam10[:2]).sum(), 2)}。除くと分散は大きく減るが", "1/λ の和")

# ---- 大問11
# (a) 7 処理・ブロックの大きさ 3 の BIBD（ファノ平面）を作って r と λ を数える
blocks = [(0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)]
r_ = [sum(t in b for b in blocks) for t in range(7)]
lam_ = {sum(i in b and j in b for b in blocks) for i, j in itertools.combinations(range(7), 2)}
E.check("s29p11a", set(r_) == {3} and lam_ == {1}, f"反復数 {set(r_)}、出会う回数 {lam_}")
E.num("s29p11a", rnd(1 * 7 / (3 * 3), 3), "λt/(rk)")
# (b) 層別の総数
E.num("s29p11b", round(3000 * 6 / 100 + 4000 * 12 / 100 + 2000 * 40 / 200), "Σ N_h p_h")

# ---- 大問12
# (a) scipy の正規密度で事後確率（e^{3.12} の丸めの影響も確認）
la = stats.norm.logpdf(136, 140, 10) + stats.norm.logpdf(6.6, 6.8, 0.5) + math.log(0.3)
lb = stats.norm.logpdf(136, 120, 10) + stats.norm.logpdf(6.6, 5.6, 0.5) + math.log(0.7)
pA = 1 / (1 + math.exp(lb - la))
E.choice("s29p12a", f"${f(pA, 3)}$", "scipy norm.logpdf")
E.check("s29p12a", abs(math.exp(3.12) - 22.65) < 0.005, "e^{3.12}")
# (b) 実際の点で群平均距離を計算して更新式と比べる
P_ = rng.normal(size=(2, 2))
Q_ = rng.normal(size=(3, 2)) + 3
R_ = rng.normal(size=(1, 2)) + 1
dm = lambda A, B: np.mean([np.linalg.norm(a - b) for a in A for b in B])
E.check("s29p12b", abs(dm(R_, np.vstack([P_, Q_])) - (2 * dm(R_, P_) + 3 * dm(R_, Q_)) / 5) < 1e-12, "乱数の点で群平均距離の更新式を確認")
E.num("s29p12b", rnd((2 * 6 + 3 * 9) / 5, 2), "更新式")
# (c) 相関比を数値最大化（スコアを格子で探す）
a12 = np.array([8, 22, 30])
b12 = np.array([32, 18, 10])
yv = np.r_[np.ones(60), np.zeros(60)]
cat = np.r_[np.repeat([0, 1, 2], a12), np.repeat([0, 1, 2], b12)]
best = 0
for s1 in np.linspace(-1, 2, 61):
    for s2_ in np.linspace(-1, 2, 61):
        sc = np.array([0, s1, s2_])[cat]
        if sc.std() > 0:
            best = max(best, np.corrcoef(sc, yv)[0, 1] ** 2)
X2 = stats.chi2_contingency(np.vstack([a12, b12]), correction=False)[0]
E.check("s29p12c", abs(best - X2 / 120) < 1e-3, f"格子探索の最大の相関比 {best:.4f}、X²/n={X2 / 120:.4f}")
E.choice("s29p12c", f"最大の相関比は$X^2/n={f(X2 / 120, 3)}$", "scipy chi2_contingency")

# ---- 大問13
# (a) AR(2) の自己相関をシミュレーションと特性根で
roots = np.roots([1, -1.0, 0.5])
E.check("s29p13a", np.all(np.abs(roots) < 1) and abs(2 * np.pi / abs(np.angle(roots[0])) - 8) < 1e-9, f"特性根 {roots}（|根|<1、周期 8）")
xt = np.zeros(400_000)
eps = rng.normal(size=xt.size)
for t in range(2, xt.size):
    xt[t] = xt[t - 1] - 0.5 * xt[t - 2] + eps[t]
xt = xt[1000:]
ac = [np.corrcoef(xt[:-k], xt[k:])[0, 1] for k in (1, 2)]
E.check("s29p13a", abs(ac[0] - 2 / 3) < 0.01 and abs(ac[1] - 1 / 6) < 0.01, f"シミュレーションの自己相関 {np.round(ac, 4)}")
E.choice("s29p13a", "周期は約8時間。$\\rho_1=0.667$、$\\rho_2=0.167$", "特性根と自己相関")
# (b) シミュレーション
heavy = rng.random(2_000_000) < 0.4
yv13 = np.where(heavy, 5, 1)
resp = rng.random(heavy.size) < np.where(heavy, 0.5, 0.9)
E.num("s29p13b", rnd((0.54 * 1 + 0.2 * 5) / 0.74, 2), "回答者の構成比から")
E.check("s29p13b", abs(yv13[resp].mean() - 2.081) < 0.005, f"シミュレーションの回答者平均 {yv13[resp].mean():.4f}")

# ---- 大問14
# (a) 正規・正規モデルの事後平均をシミュレーション（θ を引いて、観測平均が 10 付近のものの θ の平均）
thA = rng.normal(6, 2, 4_000_000)
ybA = rng.normal(thA, math.sqrt(16 / 8))
selA = np.abs(ybA - 10) < 0.05
thB = rng.normal(6, 2, 4_000_000)
ybB = rng.normal(thB, math.sqrt(16 / 64))
selB = np.abs(ybB - 10) < 0.05
E.check("s29p14a", abs(thA[selA].mean() - 8.667) < 0.05 and abs(thB[selB].mean() - 9.765) < 0.05, f"シミュレーションの事後平均 A {thA[selA].mean():.3f}、B {thB[selB].mean():.3f}")
E.choice("s29p14a", "Aは8.67、Bは9.76。人数の少ないAほど", "重み付き平均")
# (b) ジャックナイフ
x14 = np.array([3.0, 5, 6, 10])
th14 = x14.var()
loo = np.array([np.delete(x14, i).var() for i in range(4)])
corr = th14 - 3 * (loo.mean() - th14)
E.num("s29p14b", rnd(corr, 2), "1 つずつ除いて計算")
E.check("s29p14b", abs(corr - x14.var(ddof=1)) < 1e-12, "補正値 = 不偏分散（解説）")

E.done()
