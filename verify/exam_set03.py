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


# ---------- s03p01：混合正規・ベイズの定理・相関 ----------
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
post = 0.3 * sfA / p1
E.choice("s03p01b", r3(post), "ベイズの定理を scipy の確率で直接計算")
sim_post = zA[Y1 >= 55].mean()
E.check("s03p01b", abs(sim_post - post) < 0.003, f"シミュレーション P(A | Y1≥55)={sim_post:.4f}")
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

# ---------- s03p03：ポアソン・適合度・ポアソン回帰 ----------
x = np.repeat([0, 1, 2, 3, 4], [38, 40, 26, 12, 4])
lam_hat = x.mean()
E.num("s03p03a", round(len(x) * 0.3012 * lam_hat ** 2 / 2, 1), "与えた e^{-1.2} での期待度数（1桁に丸め）")
E.check("s03p03a", abs(len(x) * stats.poisson.pmf(2, lam_hat) - 26.0) < 0.05, f"scipy の pmf での期待度数 {len(x) * stats.poisson.pmf(2, lam_hat):.4f}")
# scipy.stats.chisquare（ddof=1 で推定パラメータ分の自由度を引く）
pr = np.array([0.3012 * lam_hat ** k / math.factorial(k) for k in range(3)])
expd = len(x) * np.append(pr, 1 - pr.sum())
obs = np.array([38, 40, 26, 16])
cs = stats.chisquare(obs, expd, ddof=1)
E.choice("s03p03b", f"統計量 {cs.statistic:.3f}、自由度 2", "scipy.stats.chisquare（ddof=1）")
E.check("s03p03b", cs.pvalue > 0.05 and abs(stats.chi2.sf(cs.statistic, 2) - cs.pvalue) < 1e-12, f"自由度 2 の p 値 {cs.pvalue:.3f} > 0.05")
E.check("s03p03b", len(x) * stats.poisson.sf(3, lam_hat) < 5, "4 件以上のセルの期待度数は 5 未満（併合が必要）")
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

# ---------- s03p05：符号付き順位検定・サンプルサイズ ----------
d = np.array([7, -3, 12, 5, -1, 9, 4, -6, 11, 2])
r = stats.wilcoxon(d, alternative="greater", method="approx", correction=False)
z = stats.norm.isf(r.pvalue)
E.choice("s03p05a", f"$W^+={int(r.statistic)}$、$z={z:.2f}$、棄却する", "scipy.stats.wilcoxon（正規近似・連続修正なし）")
E.check("s03p05a", r.pvalue < 0.05, f"片側 p={r.pvalue:.4f} < 0.05")
# 与えた z 値で検出力を直接計算し、0.9 以上になる最小の n を総当たりで探す
power = lambda m: 1 - stats.norm.cdf(1.645 - math.sqrt(m) * 5 / 10)
nmin = next(m for m in range(1, 500) if (1.645 + 1.282 - math.sqrt(m) * 0.5) <= 0)
E.num("s03p05b", nmin, "与えた分位点で条件を満たす最小の n を総当たり")
E.check("s03p05b", power(nmin) >= 0.9 > power(nmin - 1), f"scipy で検出力 n={nmin}: {power(nmin):.4f}、n={nmin - 1}: {power(nmin - 1):.4f}")

# ---------- s03p06：定常分布 ----------
P = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]])
vals, vecs = np.linalg.eig(P.T)
pi = np.real(vecs[:, np.argmin(abs(vals - 1))])
pi /= pi.sum()
E.choice("s03p06a", r3(pi[2]), "P^T の固有値 1 の固有ベクトル")
E.check("s03p06a", np.allclose(np.linalg.matrix_power(P, 200)[0], pi), "P^200 の行が定常分布に収束")

# ---------- s03p07：調整済み R²・クックの距離・除外変数バイアス ----------
# 実際にデータを作って確認する：調整済み R² は残差の不偏分散 / y の不偏分散
Xd = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
yd = Xd @ np.array([1.0, 0.8, 0.5]) + rng.normal(size=30)
bh = np.linalg.lstsq(Xd, yd, rcond=None)[0]
ed = yd - Xd @ bh
R2d = 1 - (ed ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
adj_def = 1 - ((ed ** 2).sum() / 27) / (((yd - yd.mean()) ** 2).sum() / 29)
E.check("s03p07a", abs(adj_def - (1 - (1 - R2d) * 29 / 27)) < 1e-12, "定義（不偏分散の比）と公式の一致を実データで確認")
E.choice("s03p07a", r3(1 - (1 - 0.62) * 29 / 27), "公式に R²=0.62 を代入")
# クックの距離：定義（1点を除いたときの係数の変化）と公式の一致を実データで確認
H = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)
s2 = (ed ** 2).sum() / 27
i = 7
bi = np.linalg.lstsq(np.delete(Xd, i, 0), np.delete(yd, i), rcond=None)[0]
Ddef = (bh - bi) @ (Xd.T @ Xd) @ (bh - bi) / (3 * s2)
ri = ed[i] / math.sqrt(s2 * (1 - H[i, i]))
E.check("s03p07b", abs(Ddef - ri ** 2 / 3 * H[i, i] / (1 - H[i, i])) < 1e-10, "クックの距離の定義と公式の一致を実データで確認")
E.num("s03p07b", round(float(Fr(24, 10) ** 2 / 3 * Fr(28, 72)), 2), "分数で公式を計算（2桁に丸め）")
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

# ---------- s03p11：マハラノビス距離と誤判別確率 ----------
m1, m2 = np.array([85.0, 6.0]), np.array([70.0, 2.0])
Sig = np.array([[100.0, 12.0], [12.0, 4.0]])
D2 = float((m1 - m2) @ np.linalg.solve(Sig, m1 - m2))
# 誤判別確率を判別関数の分布から直接（scipy の多変量正規でシミュレーション）
Xs = stats.multivariate_normal(m1, Sig).rvs(size=400000, random_state=7)
wv = np.linalg.solve(Sig, m1 - m2)
mis = ((Xs - (m1 + m2) / 2) @ wv < 0).mean()
E.check("s03p11a", abs(mis - stats.norm.sf(math.sqrt(D2) / 2)) < 0.003, f"シミュレーションの誤判別率 {mis:.4f}")
E.choice("s03p11a", f"$\\Delta^2={D2:.2f}$、誤判別確率 {stats.norm.sf(round(math.sqrt(D2) / 2, 2)):.3f}", "np.linalg.solve と scipy の正規分布")

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

# ---------- s03p14：マクネマー検定 ----------
# 条件付きで c ~ Bin(b+c, 1/2) の正確検定と、χ² 版の統計量
b, c = 6, 19
chi = (c - b) ** 2 / (b + c)
E.choice("s03p14a", f"統計量 {chi:.2f}、棄却する", "不一致ペアから直接")
E.check("s03p14a", stats.binomtest(c, b + c, 0.5).pvalue < 0.05 and stats.chi2.sf(chi, 1) < 0.05,
        f"正確な二項検定 p={stats.binomtest(c, b + c, 0.5).pvalue:.4f}、χ² 近似 p={stats.chi2.sf(chi, 1):.4f}")

E.done()
