"""模擬試験 第3回 の検算。generator/set03.py とは別の方法（scipy の分布関数・数値積分・
シミュレーション・sympy・線形代数・データを実際に作っての当てはめ）で正解を求め直す。"""
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(3)
rng = np.random.default_rng(20261024)


def r3(x):
    return f"{x:.3f}"


# ---------- s03p01：混合正規・条件付き独立・相関 ----------
sfA = stats.norm.sf(55, 60, 5)
sfB = stats.norm.sf(55, 45, 8)
p1 = 0.3 * sfA + 0.7 * sfB
E.choice("s03p01a", r3(p1), "P(Y≥55) を scipy の正規分布で直接計算")
N = 2_000_000
zA = rng.random(N) < 0.3
mu = np.where(zA, 60.0, 45.0)
sd = np.where(zA, 5.0, 8.0)
Y1 = rng.normal(mu, sd)
Y2 = rng.normal(mu, sd)
E.check("s03p01a", abs((Y1 >= 55).mean() - p1) < 0.002, f"シミュレーション P(Y1≥55)={(Y1 >= 55).mean():.4f}")
cond = (0.3 * sfA ** 2 + 0.7 * sfB ** 2) / p1
E.choice("s03p01b", r3(cond), "P(Y2≥55 | Y1≥55) を scipy で直接計算")
sim_c = ((Y1 >= 55) & (Y2 >= 55)).sum() / (Y1 >= 55).sum()
E.check("s03p01b", abs(sim_c - cond) < 0.004, f"シミュレーション {sim_c:.4f}")
E.check("s03p01b", abs(cond - p1) > 0.1, "無条件の独立とは明らかに違う（条件付き独立≠独立）")
w = sp.Rational(3, 10)
EY = w * 60 + (1 - w) * 45
EY2 = w * (60 ** 2 + 25) + (1 - w) * (45 ** 2 + 64)
EY1Y2 = w * 60 ** 2 + (1 - w) * 45 ** 2        # 土壌を与えると独立なので E[Y1Y2|Z]=μ_Z^2
rho = (EY1Y2 - EY ** 2) / (EY2 - EY ** 2)
E.num("s03p01c", round(float(rho), 3), "sympy で E[Y1Y2] から相関係数を厳密計算（3桁に丸め）")
E.check("s03p01c", abs(np.corrcoef(Y1, Y2)[0, 1] - float(rho)) < 0.005, f"シミュレーションの相関 {np.corrcoef(Y1, Y2)[0, 1]:.4f}")

# ---------- s03p02：ハザード・カプラン＝マイヤー ----------
Hq = lambda t: integrate.quad(lambda u: 0.08 * u, 0, t)[0]
E.choice("s03p02a", r3(math.exp(-Hq(5)) / math.exp(-Hq(3))), "ハザードを数値積分して S(5)/S(3)")
# 逆関数法で寿命を発生させて確認（S(t)=exp(-0.04 t^2) → T = sqrt(-log U / 0.04)）
Tm = np.sqrt(-np.log(rng.random(2_000_000)) / 0.04)
E.check("s03p02a", abs((Tm > 5).sum() / (Tm > 3).sum() - math.exp(-0.64)) < 0.003, f"シミュレーション {(Tm > 5).sum() / (Tm > 3).sum():.4f}")
data = [(1.5, 1), (2.0, 0), (2.8, 1), (3.1, 1), (3.5, 0), (4.2, 1), (4.6, 0), (5.0, 1), (5.5, 0), (6.0, 0)]
tt = np.array([t for t, _ in data])
dd = np.array([d for _, d in data])
S = 1.0
for t in np.unique(tt[dd == 1]):
    if t > 4.5:
        break
    S *= 1 - ((tt == t) & (dd == 1)).sum() / (tt >= t).sum()     # リスク集合を「t 以上」で数える
E.choice("s03p02b", r3(S), "リスク集合 {T ≥ t} を数え直したカプラン＝マイヤー")

# ---------- s03p03：ポアソン・2標本の比較・ポアソン回帰 ----------
x = np.repeat([0, 1, 2, 3, 4], [38, 40, 26, 12, 4])
lam_hat = x.mean()
E.num("s03p03a", round(len(x) * 0.3012 * lam_hat ** 2 / 2, 1), "与えた e^{-1.2} での期待度数（1桁に丸め）")
E.check("s03p03a", abs(len(x) * stats.poisson.pmf(2, lam_hat) - 26.0) < 0.05, f"scipy の pmf での期待度数 {len(x) * stats.poisson.pmf(2, lam_hat):.4f}")
# scipy.stats.chisquare（ddof=1 で推定パラメータ分の自由度を引く）
# 条件付き二項：scipy の binom の平均・標準偏差から z、正確な二項検定でも結論を確認
cb = stats.binom(243, 120 / 210)
zc = (144 - cb.mean()) / cb.std()
E.choice("s03p03b", f"$z={zc:.2f}$、棄却しない", "scipy.stats.binom の平均・標準偏差で標準化")
E.check("s03p03b", stats.binomtest(144, 243, 120 / 210).pvalue > 0.05, f"正確な二項検定の両側 p={stats.binomtest(144, 243, 120 / 210).pvalue:.3f}")
simA = rng.poisson(1.15 * 120, 200000)
simB = rng.poisson(1.15 * 90, 200000)
zs = (simA - (simA + simB) * 4 / 7) / np.sqrt((simA + simB) * 12 / 49)
E.check("s03p03b", abs((np.abs(zs) > 1.96).mean() - 0.05) < 0.005, f"H0 の下での棄却率 {(np.abs(zs) > 1.96).mean():.4f}")
b0, b1 = sp.Rational(-190, 100), sp.Rational(-45, 100)
muP = float(sp.exp(sp.log(8) + b0 + b1))
E.choice("s03p03c", r3(muP), "sympy で exp(log T + β0 + β1) を計算")

# ---------- s03p04：指数分布 ----------
n, Ssum = 40, 300.0
res = optimize.minimize_scalar(lambda l: -(n * math.log(l) - l * Ssum), bounds=(1e-4, 2), method="bounded")
E.choice("s03p04a", r3(math.exp(-10 * res.x)), "対数尤度の数値最大化から不変性で")
th = sp.symbols("theta", positive=True)
se = sp.sqrt(sp.diff(60 / th, th) ** 2 * th ** 2 / n).subs(th, sp.Rational(15, 2))
E.num("s03p04b", round(float(se), 2), "sympy で微分してデルタ法（2桁に丸め）")
sm = 60 / rng.exponential(7.5, size=(200000, 40)).mean(axis=1)
E.check("s03p04b", abs(sm.std() - float(se)) < 0.1, f"シミュレーションの標準偏差 {sm.std():.4f}（デルタ法 {float(se):.4f}）")
lo, hi = 2 * Ssum / stats.chi2.ppf(0.975, 80), 2 * Ssum / stats.chi2.ppf(0.025, 80)
E.choice("s03p04c", f"({lo:.2f},\\ {hi:.2f})", "scipy の χ² 分位点から直接")
sums = rng.exponential(7.5, size=(100000, 40)).sum(axis=1)
cov = ((2 * sums / stats.chi2.ppf(0.975, 80) <= 7.5) & (7.5 <= 2 * sums / stats.chi2.ppf(0.025, 80))).mean()
E.check("s03p04c", abs(cov - 0.95) < 0.005, f"被覆率シミュレーション {cov:.4f}")

# ---------- s03p05：符号付き順位検定・ネイマン・ピアソン ----------
d = np.array([7, -3, 12, 5, -1, 9, 4, -6, 11, 2])
r = stats.wilcoxon(d, alternative="greater", method="approx", correction=False)
z = stats.norm.isf(r.pvalue)
E.choice("s03p05a", f"$W^+={int(r.statistic)}$、$z={z:.2f}$、棄却する", "scipy.stats.wilcoxon（正規近似・連続修正なし）")
E.check("s03p05a", r.pvalue < 0.05, f"片側 p={r.pvalue:.4f} < 0.05")
# 尤度比の大きい x から順に棄却域に入れ、サイズ 5% 以下で最大の（非ランダム化）棄却域を総当たりで探す
xs_ = np.arange(16)
lr = stats.binom.pmf(xs_, 15, 0.7) / stats.binom.pmf(xs_, 15, 0.5)
order = xs_[np.argsort(-lr)]
best = None
for k in range(1, 16):
    R = order[:k]
    if stats.binom.pmf(R, 15, 0.5).sum() <= 0.05:
        best = R
E.check("s03p05b", sorted(best) == list(range(12, 16)), f"尤度比の大きい順に集めた棄却域 {sorted(best)}")
E.choice("s03p05b", "$X\\ge" + f"{min(best)}$、検出力 {stats.binom.pmf(best, 15, 0.7).sum():.3f}", "棄却域と検出力（scipy の二項分布）")

# ---------- s03p06：2ステップの経路 ----------
P = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]])
P2 = P @ P
E.choice("s03p06a", r3(P[0, 1] * P[1, 2] / P2[0, 2]), "行列の積 P² を使って条件付き確率")
s1 = np.array([rng.choice(3, p=P[0]) for _ in range(100000)])
s2 = np.array([rng.choice(3, p=P[k]) for k in s1])
E.check("s03p06a", abs((s1[s2 == 2] == 1).mean() - 0.09 / 0.19) < 0.01, f"シミュレーション {(s1[s2 == 2] == 1).mean():.4f}")

# ---------- s03p07：調整済み R²・てこ比・除外変数バイアス ----------
# 実際にデータを作って確認する：調整済み R² は残差の不偏分散 / y の不偏分散
Xd = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
yd = Xd @ np.array([1.0, 0.8, 0.5]) + rng.normal(size=30)
bh = np.linalg.lstsq(Xd, yd, rcond=None)[0]
ed = yd - Xd @ bh
R2d = 1 - (ed ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
adj_def = 1 - ((ed ** 2).sum() / 27) / (((yd - yd.mean()) ** 2).sum() / 29)
E.check("s03p07a", abs(adj_def - (1 - (1 - R2d) * 29 / 27)) < 1e-12, "定義（不偏分散の比）と公式の一致を実データで確認")
E.choice("s03p07a", r3(1 - (1 - 0.62) * 29 / 27), "公式に R²=0.62 を代入")
# てこ比：平均 10・偏差平方和 250・x17=22 となる 30 個の x を実際に作り、ハット行列の対角成分を計算
xr = rng.normal(size=29)
xr = xr - xr.mean()
others_mean = 10 - 12.0 / 29
ss_needed = 250 - 12.0 ** 2 - 29 * (others_mean - 10) ** 2
xr = others_mean + xr * math.sqrt(ss_needed / (xr ** 2).sum())
xv = np.append(xr, 22.0)
E.check("s03p07b", abs(xv.mean() - 10) < 1e-12 and abs(((xv - 10) ** 2).sum() - 250) < 1e-9, "作った x の平均 10・偏差平方和 250")
Xh = np.column_stack([np.ones(30), xv])
Hh = Xh @ np.linalg.inv(Xh.T @ Xh) @ Xh.T
E.num("s03p07b", round(Hh[-1, -1], 3), "ハット行列 X(X'X)^{-1}X' の対角成分（3桁に丸め）")
# 除外変数バイアス：重回帰係数と補助回帰から単回帰の傾きが正確に再現されることを実データで確認
x1 = rng.normal(5, 1, 30)
x2 = 150 - 4 * x1 + rng.normal(0, 3, 30)
yy = 20 + 1.8 * x1 + 0.05 * x2 + rng.normal(0, 1, 30)
bm = np.linalg.lstsq(np.column_stack([np.ones(30), x1, x2]), yy, rcond=None)[0]
dl = np.polyfit(x1, x2, 1)[0]
bs = np.polyfit(x1, yy, 1)[0]
E.check("s03p07c", abs(bs - (bm[1] + bm[2] * dl)) < 1e-10, "単回帰の傾き = β1 + β2δ を実データで確認")
E.choice("s03p07c", f"{1.8 + 0.05 * (-4.0):.2f}。施肥量の多い区画ほど", "公式に与えた値を代入")

# ---------- s03p08：AIC/BIC ----------
mll = {"M1": (236.8, 2), "M2": (232.0, 3), "M3": (229.9, 4)}
aic = min(mll, key=lambda m: mll[m][0] + 2 * mll[m][1])
bic = min(mll, key=lambda m: mll[m][0] + math.log(200) * mll[m][1])
E.choice("s03p08a", f"AIC は {aic}、BIC は {bic} を選ぶ", "log(200) を math.log で計算して比較")

# ---------- s03p09：正規・正規モデル・乱塊法 ----------
# 事後分布を格子上で数値計算（事前密度 × 尤度を正規化）
g = np.linspace(1.0, 4.0, 300001)
logpost = stats.norm.logpdf(g, 2.0, 0.3) + stats.norm.logpdf(2.6, g, 0.8 / 4)
wpost = np.exp(logpost - logpost.max())
wpost /= wpost.sum()
pm = (g * wpost).sum()
ps = math.sqrt(((g - pm) ** 2 * wpost).sum())
E.choice("s03p09a", f"({pm - 1.96 * ps:.2f},\\ {pm + 1.96 * ps:.2f})", "格子近似で事後平均と事後標準偏差")
a_eff = np.array([52.0, 55.5, 49.0, 53.5]) - 52.5
bvec = rng.normal(size=5)
bvec -= bvec.mean()
bvec *= math.sqrt(70.0 / 4 / (bvec ** 2).sum())
e = rng.normal(size=(4, 5))
e = e - e.mean(axis=1, keepdims=True) - e.mean(axis=0, keepdims=True) + e.mean()
e *= math.sqrt(47.5 / (e ** 2).sum())
Ymat = 52.5 + a_eff[:, None] + bvec[None, :] + e
E.check("s03p09b", abs(((Ymat - Ymat.mean()) ** 2).sum() - 230.0) < 1e-8, "作ったデータの総平方和が 230")
yv = Ymat.ravel()
rows = np.repeat(np.arange(4), 5)
cols = np.tile(np.arange(5), 4)
Xfull = np.column_stack([np.ones(20)] + [(rows == k).astype(float) for k in range(1, 4)] + [(cols == k).astype(float) for k in range(1, 5)])
Xblk = np.column_stack([np.ones(20)] + [(cols == k).astype(float) for k in range(1, 5)])
sse = lambda Xm: ((yv - Xm @ np.linalg.lstsq(Xm, yv, rcond=None)[0]) ** 2).sum()
Fv = ((sse(Xblk) - sse(Xfull)) / 3) / (sse(Xfull) / 12)
E.num("s03p09b", round(Fv, 2), "ダミー変数の回帰で縮小モデルとの比較から F（2桁に丸め）")

# ---------- s03p10：比推定 ----------
yr = Fr(54) / Fr(15, 2) * Fr(3200, 400)
E.num("s03p10a", round(float(yr), 1), "分数で比推定量")

# ---------- s03p11：混同行列と F 値 ----------
# 日ごとのラベルを作り直して、集合の数え上げで F = 2TP/(2TP+FP+FN)
truth = np.array([1] * 18 + [0] * 12 + [1] * 6 + [0] * 329)
pred = np.array([1] * 30 + [0] * 335)
tp = ((truth == 1) & (pred == 1)).sum()
F1 = 2 * tp / (2 * tp + ((truth == 0) & (pred == 1)).sum() + ((truth == 1) & (pred == 0)).sum())
E.choice("s03p11a", r3(F1), "F = 2TP/(2TP+FP+FN) で計算")

# ---------- s03p12：最長距離法 ----------
Dm = np.array([[0, 2, 6, 10, 9], [2, 0, 5, 8, 7.5], [6, 5, 0, 4, 3], [10, 8, 4, 0, 3.5], [9, 7.5, 3, 3.5, 0]])
Z = linkage(squareform(Dm), method="complete")
E.choice("s03p12a", f"{Z[-1, 2]:.1f}", "scipy の linkage(complete)")
E.check("s03p12a", abs(linkage(squareform(Dm), "single")[-1, 2] - 5.0) < 1e-12 and abs(linkage(squareform(Dm), "average")[-1, 2] - 7.5833) < 1e-3,
        "誤答（単連結 5.0・群平均 7.58）の確認")

# ---------- s03p13：1因子モデル ----------
L = np.array([0.8, 0.7, 0.6, 0.5])
Sg = np.outer(L, L) + np.diag(1 - L ** 2)
E.choice("s03p13a", f"{Sg[0, 2]:.2f}", "Σ=λλ'+Ψ の (国語,数学) 成分")
C = np.zeros((3, 3))
C[:2, :2] = Sg[:2, :2]
C[:2, 2] = C[2, :2] = L[:2]
C[2, 2] = 1
cc = C[:2, :2] - np.outer(C[:2, 2], C[2, :2]) / C[2, 2]
E.check("s03p13a", abs(cc[0, 1]) < 1e-12, "f を固定した偏相関は 0（選択肢 4 は誤り）")
E.check("s03p13a", abs(L[0] ** 2 - 0.80) > 0.1 and abs(np.sum(L ** 2) / 4 - 0.65) > 0.1 and abs(1 - L[1] ** 2 - 0.30) > 0.1,
        "他の記述（共通性 0.80・寄与率 0.65・独自性 0.30）は誤り")

# ---------- s03p14：シンプソンのパラドックス ----------
tab = {"X": (18, 20, 56, 70), "Y": (42, 80, 9, 20)}
within = all(a / na > b / nb for a, na, b, nb in tab.values())
tot_a = sum(v[0] for v in tab.values()) / sum(v[1] for v in tab.values())
tot_b = sum(v[2] for v in tab.values()) / sum(v[3] for v in tab.values())
E.check("s03p14a", within and tot_a < tot_b, f"各校では受講者が高いが合計では逆転（{tot_a:.3f} < {tot_b:.3f}）")
E.check("s03p14a", tab["Y"][1] / 100 > 0.5 and tab["Y"][0] / tab["Y"][1] < tab["X"][0] / tab["X"][1], "受講者は合格率の低い Y 校に偏る（正解の記述の根拠）")
E.check("s03p14a", abs(tot_a - (18 / 20 + 42 / 80) / 2) > 0.05, "合計は単純平均ではない（選択肢 3 は誤り）")
E.check("s03p14a", 9 / 20 < 42 / 80, "Y 校の中では受講者のほうが高い（選択肢 5 は誤り）")
it14 = E.item("s03p14a")
E.check("s03p14a", it14["answer"] == 1 and "学校で層別" in it14["choices"][1], "正解は層別比較を勧める記述")

E.done()
