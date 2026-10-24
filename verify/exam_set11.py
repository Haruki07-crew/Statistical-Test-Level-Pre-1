"""模擬試験 第11回（set11）の検算。generator/set11.py とは別の方法（scipy の分布関数・シミュレーション・
数値積分・数値最適化・総当たり・行列計算）で正解を求め直し、問題文に示した表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(11)
rng = np.random.default_rng(20261111)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def ans_text(iid):
    it = E.item(iid)
    return it["choices"][it["answer"]]


def distinct_rounded(iid):
    """数値の選択肢がすべて異なる（丸めたあとで正解と重ならない）。"""
    v = nums(iid)
    E.check(iid, len(set(v)) == 5, f"選択肢の数値がすべて異なる {v}")


# ---- 大問1 古典的テスト理論
# (a) 共分散行列を行列計算で：(T, E1, E2) の共分散行列 → S=2T+E1+E2 の分散
C = np.diag([80.0, 20.0, 20.0])
a_true, a_S = np.array([2, 0, 0]), np.array([2, 1, 1])
relS = (a_true @ C @ a_true) / (a_S @ C @ a_S)
E.choice("s11p01a", f"${f(relS, 3)}$", "線形結合の分散を共分散行列で計算")
T = rng.normal(60, np.sqrt(80), 400_000)
S = 2 * T + rng.normal(0, np.sqrt(20), (2, 400_000)).sum(0)
E.check("s11p01a", abs(np.var(2 * T) / np.var(S) - relS) < 0.004, f"シミュレーション {np.var(2 * T) / np.var(S):.4f}")
distinct_rounded("s11p01a")
# (b) 2変量正規の条件付き分布（scipy）とシミュレーション
cov = np.array([[80.0, 80.0], [80.0, 100.0]])
m_c = 60 + cov[0, 1] / cov[1, 1] * (72 - 60)
v_c = cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1]
p_b = stats.norm.cdf(70, m_c, np.sqrt(v_c))
E.num("s11p01b", rnd(p_b, 3), "scipy norm.cdf で条件付き分布")
E.check("s11p01b", abs(stats.norm.cdf(0.10) - 0.5398) < 5e-5, "Φ(0.10)=0.5398")
Tm = rng.normal(60, np.sqrt(80), 4_000_000)
Xm = Tm + rng.normal(0, np.sqrt(20), 4_000_000)
sel = np.abs(Xm - 72) < 0.25
E.check("s11p01b", abs((Tm[sel] < 70).mean() - 0.540) < 0.01, f"X≈72 の受験者のシミュレーション {(Tm[sel] < 70).mean():.4f}")

# ---- 大問2 採用選考
# (a) 3人の判定の 8 通りを総当たり
num = den = 0.0
for good, pr in ((True, 0.3), (False, 0.7)):
    q = 0.8 if good else 0.25
    for v in product([0, 1], repeat=3):
        pv = pr * np.prod([q if x else 1 - q for x in v])
        if sum(v) >= 2:
            den += pv
            num += pv if good else 0
E.num("s11p02a", rnd(num / den, 3), "判定の組を総当たり")
# (b) マン＝ホイットニーの U / (n1 n2)
st, lv = [72, 65, 80, 58, 69], [55, 61, 65, 50, 70]
U = stats.mannwhitneyu(st, lv).statistic
E.choice("s11p02b", f"${f(U / 25, 2)}$", f"scipy mannwhitneyu の U={U}")
sc = np.r_[st, lv]
lab = np.r_[np.ones(5), np.zeros(5)]
ths = np.r_[np.inf, np.sort(np.unique(sc))[::-1], -np.inf]
tpr = [((sc >= t) & (lab == 1)).sum() / 5 for t in ths]
fpr = [((sc >= t) & (lab == 0)).sum() / 5 for t in ths]
E.check("s11p02b", abs(np.trapezoid(tpr, fpr) - U / 25) < 1e-12, "ROC 曲線を台形で積分した面積と一致")
# (c) 事後分布での E[p^2]（数値積分）とシミュレーション
post = stats.beta(13, 5)
Ep2 = integrate.quad(lambda p: p ** 2 * stats.beta(6, 2).pdf(p) * stats.binom.pmf(7, 10, p), 0, 1)[0] / \
    integrate.quad(lambda p: stats.beta(6, 2).pdf(p) * stats.binom.pmf(7, 10, p), 0, 1)[0]
E.choice("s11p02c", f"${f(Ep2, 3)}$", "事前×尤度の数値積分で E[p^2]")
ps = post.rvs(size=1_000_000, random_state=rng)
both = (rng.random(ps.size) < ps) & (rng.random(ps.size) < ps)
E.check("s11p02c", abs(both.mean() - Ep2) < 0.003, f"2人の結果のシミュレーション {both.mean():.4f}")
distinct_rounded("s11p02c")

# ---- 大問3 対応時間
l1, l2 = 0.5, 0.2
fy = lambda y: integrate.quad(lambda u: l1 * np.exp(-l1 * u) * l2 * np.exp(-l2 * (y - u)), 0, y)[0]
surv = 1 - integrate.quad(fy, 0, 10)[0]
E.choice("s11p03a", f"${f(surv, 3)}$", "畳み込みを数値積分")
Y = rng.exponential(2, 2_000_000) + rng.exponential(5, 2_000_000)
E.check("s11p03a", abs((Y > 10).mean() - surv) < 0.002, f"シミュレーション {(Y > 10).mean():.4f}")
E.check("s11p03a", abs(np.exp(-2) - 0.1353) < 5e-5 and abs(np.exp(-5) - 0.0067) < 5e-5, "e^-2, e^-5 の値")
distinct_rounded("s11p03a")
varY = stats.expon(scale=2).var() + stats.expon(scale=5).var()
n = 1
while varY / n > 0.05:
    n += 1
E.choice("s11p03b", f"${n}$", "チェビシェフの上界が 0.05 以下になる n を探索")
distinct_rounded("s11p03b")

# ---- 大問4 満足度調査
p = np.array([0.15, 0.35, 0.25, 0.15, 0.10])
Sig = 200 * (np.diag(p) - np.outer(p, p))
a = np.array([1, 1, 0, -1, -1])
E.choice("s11p04a", f"${f(a @ Sig @ a, 1)}$", "多項分布の共分散行列で a'Σa")
D = rng.multinomial(200, p, size=300_000) @ a
E.check("s11p04a", abs(D.var() - 137.5) < 1.5, f"シミュレーション {D.var():.2f}")
distinct_rounded("s11p04a")
chi, pv, dof, _ = stats.chi2_contingency([[60, 25, 35], [40, 25, 15]], correction=False)
ok = (f(chi, 2) in ans_text("s11p04b") and "自由度 2" in ans_text("s11p04b") and "棄却しない" in ans_text("s11p04b") and dof == 2 and pv > 0.05)
E.check("s11p04b", ok, f"scipy chi2_contingency: χ²={chi:.4f}, 自由度 {dof}, p={pv:.4f}")
E.check("s11p04b", all(abs(stats.chi2.ppf(0.95, d) - v) < 5e-4 for d, v in ((1, 3.841), (2, 5.991), (5, 11.070))), "χ² の 5% 点")
others = [c for i, c in enumerate(E.item("s11p04b")["choices"]) if i != E.item("s11p04b")["answer"]]
E.check("s11p04b", all(("自由度 2" not in c) or (f(chi, 2) not in c) for c in others), "他の選択肢は統計量か自由度が誤り")

# ---- 大問5 ラッシュモデル
nll = lambda th: -(16 * (th - 0.5) - 20 * np.log1p(np.exp(th - 0.5)))
th = optimize.minimize_scalar(nll, bounds=(-5, 8), method="bounded", options={"xatol": 1e-10}).x
h = 1e-4
info = (nll(th + h) - 2 * nll(th) + nll(th - h)) / h ** 2
E.num("s11p05a", rnd(th + 1.96 / np.sqrt(info), 2), f"数値最適化と数値微分 θ̂={th:.5f}, I={info:.4f}")
E.check("s11p05a", abs(np.log(4) - 1.3863) < 5e-5, "log 4 = 1.3863")

# ---- 大問6 研修効果
b = np.array([62, 55, 70, 48, 66, 59, 73, 51])
aa = np.array([67, 63, 68, 54, 69, 68, 77, 58])
res = stats.ttest_rel(aa, b)
lo = res.confidence_interval(0.95).low
E.num("s11p06a", rnd(lo, 2), "scipy ttest_rel の信頼区間")
E.check("s11p06a", abs(stats.t.ppf(0.975, 7) - 2.365) < 5e-4 and abs(stats.t.ppf(0.975, 8) - 2.306) < 5e-4, "t の点")
pvals = {"国語": 0.014, "数学": 0.004, "英語": 0.045, "理科": 0.011, "社会": 0.036}
keys = sorted(pvals, key=pvals.get)
adj, run = {}, 0
for k_, key in enumerate(keys):
    run = max(run, (5 - k_) * pvals[key])
    adj[key] = min(1, run)
rej = [k for k in keys if adj[k] <= 0.05]
E.check("s11p06b", ans_text("s11p06b") == "・".join(rej), f"ホルム調整 p 値 {adj} → {rej}")

# ---- 大問7 フリードマン検定
A7 = [62, 58, 67, 55, 60, 69]
B7 = [66, 70, 64, 61, 72, 66]
C7 = [71, 65, 73, 63, 68, 75]
fr = stats.friedmanchisquare(A7, B7, C7)
E.num("s11p07a", rnd(fr.statistic, 2), "scipy friedmanchisquare")
E.check("s11p07a", fr.pvalue > 0.05 and abs(stats.chi2.ppf(0.95, 2) - 5.991) < 5e-4, f"p 値 {fr.pvalue:.4f} > 0.05（解説の結論）")

# ---- 大問8 賃金の回帰
# (a) 実際にデータを作って2つのモデルを当てはめ、部分F検定の式が残差平方和だけで決まることを確かめたうえで、与えた値で計算
Xd = np.c_[np.ones(60), rng.normal(size=(60, 5))]
yd = Xd @ rng.normal(size=6) + rng.normal(size=60)
rs = lambda X: np.sum((yd - X @ np.linalg.lstsq(X, yd, rcond=None)[0]) ** 2)
r1, r2 = rs(Xd[:, :3]), rs(Xd)
Fsim = ((r1 - r2) / 3) / (r2 / 54)
R2a, R2b = 1 - r1 / np.sum((yd - yd.mean()) ** 2), 1 - r2 / np.sum((yd - yd.mean()) ** 2)
E.check("s11p08a", abs(Fsim - ((R2b - R2a) / 3) / ((1 - R2b) / 54)) < 1e-9, "決定係数による部分F の式と一致（自由度 3, 54）")
Fv = (0.90 / 3) * 54 / 3.90
E.choice("s11p08a", f"${f(Fv, 2)}$", "分数で計算した F")
E.check("s11p08a", abs(stats.f.ppf(0.95, 3, 54) - 2.78) < 0.005 and stats.f.sf(Fv, 3, 54) < 0.05, "F(3,54) の 5% 点と p 値")
distinct_rounded("s11p08a")
lm = 60 * 0.09
E.check("s11p08b", stats.chi2.sf(lm, 2) > 0.05 and stats.chi2.sf(lm, 1) < 0.05 and f(lm, 2) in ans_text("s11p08b")
        and "棄却しない" in ans_text("s11p08b") and "自由度 2" in ans_text("s11p08b") and "証明" not in ans_text("s11p08b"),
        f"χ²(2) の p 値 {stats.chi2.sf(lm, 2):.4f}")
E.check("s11p08b", abs(stats.f.ppf(0.95, 2, 57) - 3.16) < 0.005, "F(2,57) の 5% 点 3.16")
# (c) 2群の平均を正確に再現するデータを作り、2段階最小二乗を行列で計算
nz = 500
Z = np.r_[np.ones(nz), np.zeros(nz)]
Sx = np.r_[13.6 + rng.normal(0, 2, nz), 13.1 + rng.normal(0, 2, nz)]
Sx[:nz] += 13.6 - Sx[:nz].mean()
Sx[nz:] += 13.1 - Sx[nz:].mean()
yy = np.r_[6.452 + rng.normal(0, 0.3, nz), 6.397 + rng.normal(0, 0.3, nz)]
yy[:nz] += 6.452 - yy[:nz].mean()
yy[nz:] += 6.397 - yy[nz:].mean()
Zm = np.c_[np.ones(2 * nz), Z]
Xm2 = np.c_[np.ones(2 * nz), Sx]
Shat = Zm @ np.linalg.lstsq(Zm, Sx, rcond=None)[0]
b2sls = np.linalg.lstsq(np.c_[np.ones(2 * nz), Shat], yy, rcond=None)[0][1]
at = ans_text("s11p08c")
E.check("s11p08c", f(b2sls, 3) in at and "無相関" in at and "直接" not in at, f"2段階最小二乗 {b2sls:.5f}")
E.check("s11p08c", sum(f(b2sls, 3) in c for c in E.item("s11p08c")["choices"]) == 2, "同じ推定値の他の選択肢は条件が誤り（除外制約の逆）")

# ---- 大問9 離職
E.choice("s11p09a", f"${f(stats.norm.cdf(-1.4 + 0.025 * 60) - stats.norm.cdf(-1.4 + 0.025 * 20), 3)}$", "scipy norm.cdf")
E.check("s11p09a", all(abs(stats.norm.cdf(z) - v) < 5e-5 for z, v in ((0.1, 0.5398), (0.9, 0.8159), (1.0, 0.8413))), "Φ の値")
distinct_rounded("s11p09a")
P = np.array([[0, 0.6, 0.4], [1, 0, 0], [1, 0, 0]])
w, V = np.linalg.eig(P.T)
pi = np.real(V[:, np.argmin(np.abs(w - 1))])
pi = pi / pi.sum()
P2 = P @ P
P20, P21 = np.linalg.matrix_power(P, 20), np.linalg.matrix_power(P, 21)
period = np.gcd.reduce([n_ for n_ in range(1, 13) if np.linalg.matrix_power(P, n_)[0, 0] > 0])
at = ans_text("s11p09b")
E.check("s11p09b", np.allclose(pi, [0.5, 0.3, 0.2]) and period == 2 and not np.allclose(P20, P21)
        and "(0.5,0.3,0.2)" in at.replace(" ", "") and "収束しない" in at,
        f"固有ベクトルで定常分布 {pi.round(4)}、周期 {period}、P^20≠P^21")
E.check("s11p09b", abs(P2[1, 1] - 0.6) < 1e-12 and abs(np.mean([np.linalg.matrix_power(P, n_)[0, 0] for n_ in range(1, 2001)]) - 0.5) < 1e-3,
        "2年後に企画に戻る確率は 0.6（選択肢5は誤り）、営業にいる時間平均は 0.5")

# ---- 大問10 採点者の変量効果
Fr = (104 / 4) / (108 / 15)
E.choice("s11p10a", f"${f((Fr - 1) / (Fr + 4 - 1), 3)}$", "F 比による式 (F-1)/(F+n-1)")
sims = []
for _ in range(20000):
    yv = rng.normal(0, 2.0, (5, 1)) + rng.normal(0, 3.0, (5, 4))
    msa = 4 * yv.mean(1).var(ddof=1)
    mse = ((yv - yv.mean(1, keepdims=True)) ** 2).sum() / 15
    sims.append((msa - mse) / 4)
E.check("s11p10a", abs(np.mean(sims) - 4.0) < 0.1, f"(MSA-MSE)/n が σA²=4 の不偏推定（シミュレーション {np.mean(sims):.3f}）")
distinct_rounded("s11p10a")

# ---- 大問11 学校調査
# 包含確率を全標本の列挙で数え、HT 推定量の不偏性も架空の母集団で確かめる
from itertools import combinations
small = list(range(17))
samples = list(combinations(small, 4))
pi_small = sum(1 for sm in samples if 0 in sm) / len(samples)
E.check("s11p11a", abs(pi_small - 4 / 17) < 1e-12, f"列挙による小規模校の包含確率 {pi_small:.6f} = 4/17")
ht_v = (27 + 31 + 22) * 1.0 + sum(v / pi_small for v in [6, 9, 4, 5])
E.choice("s11p11a", f"${f(ht_v, 0)}$", "列挙した包含確率で Σ y/π")
ypop = rng.integers(0, 12, 17)
mean_ht = np.mean([30 + ypop[list(sm)].sum() / pi_small for sm in samples])
E.check("s11p11a", abs(mean_ht - (30 + ypop.sum())) < 1e-9, "全標本の平均が真の総数に一致（不偏）")
distinct_rounded("s11p11a")
q = np.array([8.2, 8.6, 8.0, 8.5, 8.7])
u = np.array([0.090, 0.094, 0.086, 0.092, 0.088])
Tv = u.mean() + (1 + 1 / 5) * q.var(ddof=1)
E.num("s11p11b", rnd(np.sqrt(Tv), 3), "numpy var(ddof=1) でルービンのルール")

# ---- 大問12 適性検査
ev = np.linalg.eigvalsh(np.array([[30.0, 12], [12, 20]]))
E.choice("s11p12a", f"${f(ev.max() / ev.sum(), 3)}$", "numpy eigvalsh")
distinct_rounded("s11p12a")
pts = np.array([[2, 3], [3, 4], [4, 2], [8, 7], [9, 9], [7, 8]], dtype=float)
cen, lab = kmeans2(pts, np.array([[2.0, 3], [4, 2]]), iter=20, minit="matrix")
wss = sum(((pts[lab == k] - cen[k]) ** 2).sum() for k in range(2))
E.choice("s11p12b", f"${f(wss, 1)}$", f"scipy kmeans2 のラベル {lab.tolist()}")
distinct_rounded("s11p12b")

# ---- 大問13 AR(2)
phi = np.linalg.solve(np.array([[1, 0.8], [0.8, 1]]), np.array([0.8, 0.5]))
roots = np.roots([-phi[1], -phi[0], 1])
stat_ok = np.all(np.abs(roots) > 1)
at = ans_text("s11p13a")
E.check("s11p13a", f(phi[0], 3) in at and f(phi[1], 3) in at and "満たす" in at and "満たさない" not in at and stat_ok,
        f"連立方程式 φ={phi.round(5)}、特性根の絶対値 {np.abs(roots).round(4)}")
x_ = rng.normal(size=600_000)
xs = np.zeros_like(x_)
for t in range(2, x_.size):
    xs[t] = phi[0] * xs[t - 1] + phi[1] * xs[t - 2] + x_[t]
xs = xs[1000:]
rho1 = np.corrcoef(xs[1:], xs[:-1])[0, 1]
rho2 = np.corrcoef(xs[2:], xs[:-2])[0, 1]
E.check("s11p13a", abs(rho1 - 0.8) < 0.01 and abs(rho2 - 0.5) < 0.01, f"推定したモデルの自己相関（シミュレーション）{rho1:.4f}, {rho2:.4f}")

# ---- 大問14 LOOCV
xv = np.array([2, 4, 5, 6, 8, 11.0])
yv = np.array([54, 61, 58, 66, 67, 70.0])
errs = []
for i in range(6):
    m = np.arange(6) != i
    co = np.polyfit(xv[m], yv[m], 1)
    errs.append(yv[i] - np.polyval(co, xv[i]))
E.num("s11p14a", rnd(np.mean(np.square(errs)), 1), "6回の当てはめ直しによる LOOCV")
co = np.polyfit(xv, yv, 1)
e_full = yv - np.polyval(co, xv)
X1 = np.c_[np.ones(6), xv]
hdiag = np.diag(X1 @ np.linalg.inv(X1.T @ X1) @ X1.T)
ctx = [p for p in E.data["problems"] if p["id"] == "s11p14"][0]["context"]
shown = all(f(v, 3) in ctx for v in np.r_[e_full, hdiag])
E.check("s11p14a", shown and f(co[0], 2) in ctx, "表の残差・てこ比・傾きが実データの値を丸めたもの")

E.done()
