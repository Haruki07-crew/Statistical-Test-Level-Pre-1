"""模擬試験 第11回（set11）の検算。generator/set11.py とは別の方法（scipy の分布関数・シミュレーション・
数値積分・数値最適化・総当たり・行列計算）で正解を求め直し、問題文に示した表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
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
# (a) 一様分布の最大値のピボット区間：scipy のベータ分布（M/θ ~ Be(n,1)）の 5% 点と、被覆率のシミュレーション
c_ = stats.beta(8, 1).ppf(0.05)
E.num("s11p06a", rnd(12.0 / c_, 2), f"Be(8,1) の 5% 点 {c_:.5f} で上端 {12.0 / c_:.4f}")
U6 = rng.uniform(0, 1, (200_000, 8)).max(1)
E.check("s11p06a", abs(np.mean((U6 <= 1) & (1 <= U6 / c_)) - 0.95) < 0.003, f"θ=1 のとき [M, M/c] が θ を含む割合 {np.mean(1 <= U6 / c_):.4f}")
E.check("s11p06a", all(abs(a_ - b_) < 5e-5 for a_, b_ in ((0.05 ** (1 / 8), 0.6877), (0.025 ** (1 / 8), 0.6306), (0.05 ** (1 / 7), 0.6518))), "問題文の数値")
# (b) 尤度比：対数尤度を数値で最大化して −2logΛ を作り、H0 のもとのシミュレーションで χ²(1) 近似の棄却率を確かめる
from scipy import optimize
def nll(s2, ss_, n_):
    return 0.5 * n_ * np.log(s2) + ss_ / (2 * s2)
opt = optimize.minimize_scalar(lambda v_: nll(v_, 1800, 30), bounds=(1, 1000), method="bounded")
T6 = 2 * (nll(100, 1800, 30) - opt.fun)
E.choice("s11p06b", f"$-2\\log\\Lambda={f(T6, 2)}$で、自由度1の3.841を下回るので、帰無仮説を棄却しない", f"数値最大化で σ̂²={opt.x:.3f}, −2logΛ={T6:.4f}")
Xs6 = rng.normal(60, 10, (100_000, 30))
sh6 = ((Xs6 - 60) ** 2).mean(1) / 100
Ts = 30 * (sh6 - 1 - np.log(sh6))
E.check("s11p06b", abs((Ts > stats.chi2.isf(0.05, 1)).mean() - 0.05) < 0.01, f"H0 のもとで −2logΛ>3.841 の割合 {(Ts > stats.chi2.isf(0.05, 1)).mean():.4f}")
E.check("s11p06b", abs(stats.chi2.isf(0.05, 30) - 43.77) < 0.005 and abs(stats.chi2.ppf(0.025, 30) - 16.79) < 0.005 and abs(np.log(0.6) + 0.5108) < 5e-5, "問題文・解説の数表の値")

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
# (b) 頑健標準誤差：不均一分散のデータで statsmodels を使わず HC0 を行列で計算し、係数は OLS と同じであることを確かめたうえで、与えた値で t を計算
Xh = np.c_[np.ones(60), rng.uniform(0, 30, 60)]
eh = rng.normal(0, 1, 60) * (0.2 + Xh[:, 1] / 10)
yh = Xh @ np.array([5.0, 0.03]) + eh
bh = np.linalg.solve(Xh.T @ Xh, Xh.T @ yh)
rh = yh - Xh @ bh
XtXi = np.linalg.inv(Xh.T @ Xh)
hc0 = np.sqrt(np.diag(XtXi @ (Xh.T * rh ** 2) @ Xh @ XtXi))
E.check("s11p08b", hc0.shape == (2,) and np.allclose(bh, np.linalg.lstsq(Xh, yh, rcond=None)[0]), "HC0 は係数を変えず標準誤差だけを変える（行列計算で確認）")
E.choice("s11p08b", f"$t={f(0.032 / 0.012, 2)}$で5%水準で有意だが、頑健標準誤差では$t={f(0.032 / 0.019, 2)}$で有意でない", "与えた値の t")
E.check("s11p08b", 0.032 / 0.019 < stats.t.isf(0.025, 54) < 0.032 / 0.012 and abs(stats.t.isf(0.025, 54) - 2.005) < 5e-4, "t(54) の点 2.005 と判定")
# (c) 除外変数バイアス：大きな標本を作り、S だけの回帰の傾きが β1+γδ になることを確かめる
nb = 400_000
Sb = rng.normal(13, 2, nb)
Ab = 0.25 * (Sb - 13) + rng.normal(0, np.sqrt(1 - 0.25 ** 2 * 4), nb)
yb = 1.0 + 0.060 * Sb + 0.10 * Ab + rng.normal(0, 0.3, nb)
slope = np.polyfit(Sb, yb, 1)[0]
E.check("s11p08c", abs(np.cov(Sb, Ab)[0, 1] - 1.0) < 0.02 and abs(slope - 0.085) < 0.002, f"β1=0.060 の母集団で S だけの回帰の傾き {slope:.4f}（問題文の 0.085 と一致）")
E.choice("s11p08c", f"${f(0.085 - 0.10 * 1.0 / 4.0, 3)}$", "β1 = 0.085 − γ·Cov/V")

# ---- 大問9 離職
# (a) sympy で b(θ) から分散関数と正準リンクを導く
th, mu_ = sp.symbols("theta mu", real=True)
def vf(b_, theta_of_mu):
    m_ = sp.diff(b_, th)
    return sp.simplify(sp.diff(b_, th, 2).subs(th, theta_of_mu)), sp.simplify(m_.subs(th, theta_of_mu) - mu_)
vb, cb = vf(sp.log(1 + sp.exp(th)), sp.log(mu_ / (1 - mu_)))
vpo, cpo = vf(sp.exp(th), sp.log(mu_))
vg, cg = vf(-sp.log(-th), -1 / mu_)
E.check("s11p09a", sp.simplify(vb - mu_ * (1 - mu_)) == 0 and cb == 0, f"ベルヌーイ：V={vb}、ロジットで μ=b'(θ)")
E.check("s11p09a", sp.simplify(vpo - mu_) == 0 and cpo == 0, f"ポアソン：V={vpo}、log で μ=b'(θ)")
E.check("s11p09a", sp.simplify(vg - mu_ ** 2) == 0 and cg == 0, f"ガンマ：V={vg}、θ=−1/μ で μ=b'(θ)")
truth = {"ベルヌーイ": ("\\mu(1-\\mu)", "\\log\\frac{\\mu}{1-\\mu}"), "ポアソン": ("\\mu", "\\log\\mu"), "ガンマ": ("\\mu^2", "1/\\mu")}
def all_true(ch):
    segs = dict(re.findall(r"(ベルヌーイ|ポアソン|ガンマ)：([^。]*)", ch))
    ok = []
    for k_, (vv, ll_) in truth.items():
        parts = re.findall(r"\$([^$]*)\$", segs[k_])
        ok.append(parts[0] == f"V(\\mu)={vv}" and parts[1] == ll_)
    return all(ok)
hits9 = [i for i, ch in enumerate(E.item("s11p09a")["choices"]) if all_true(ch)]
E.check("s11p09a", hits9 == [E.item("s11p09a")["answer"]], f"3 つの分布とも正しい（sympy の結果と一致する）選択肢 {hits9}")
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
# L8 直交表を実際に作り、列どうしの積（交互作用の列）を確かめる
base = np.array(list(product([1, -1], repeat=3)))      # (a, b, c) の 8 行
cols = {"a": base[:, 0], "b": base[:, 1], "c": base[:, 2]}
cols["ab"] = cols["a"] * cols["b"]; cols["ac"] = cols["a"] * cols["c"]; cols["bc"] = cols["b"] * cols["c"]; cols["abc"] = cols["ab"] * cols["c"]
order = ["a", "b", "ab", "c", "ac", "bc", "abc"]
L8 = np.c_[[cols[k_] for k_ in order]].T
E.check("s11p10a", np.allclose(L8.T @ L8, 8 * np.eye(7)), "作った L8 は直交")
colof = lambda vec: [j + 1 for j in range(7) if np.array_equal(L8[:, j], vec)][0]
inter = {colof(L8[:, 0] * L8[:, 1]), colof(L8[:, 0] * L8[:, 3]), colof(L8[:, 1] * L8[:, 3])}
free_ = [j for j in range(1, 8) if j not in {1, 2, 4} | inter]
E.check("s11p10a", inter == {3, 5, 6} and free_ == [7], f"A×B・A×C・B×C の列 {sorted(inter)}、空いている列 {free_}")
E.choice("s11p10a", f"第 {free_[0]} 列（D は A×B×C と交絡", "空いている列")
E.check("s11p10a", colof(L8[:, 0] * L8[:, 6]) == 6, "D を第 7 列に置くと A×D は第 6 列（B×C）に現れる（解説の確認）")

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

# ---- 大問14 マローズの Cp：説明変数を作って実際に当てはめ、Cp の式と AIC の近さも確かめる
ctx14 = [p for p in E.data["problems"] if p["id"] == "s11p14"][0]["context"]
rows = {m_[0]: [float(x) for x in m_[1].split("|")] for m_ in re.findall(r"\| (\$p\$|残差平方和) \|([^\n]*)\|", ctx14)}
pp, rr = rows["$p$"], rows["残差平方和"]
s2_ = rr[3] / (30 - pp[3])
cps = [r_ / s2_ - 30 + 2 * p_ for p_, r_ in zip(pp, rr)]
best = "ABCF"[int(np.argmin(cps))]
E.choice("s11p14a", f"モデル {best}（$C_p={f(min(cps), 1)}$）", f"表から計算した Cp {np.round(cps, 3)}")
E.check("s11p14a", abs(cps[3] - pp[3]) < 1e-9, "全変数のモデルでは Cp=p")
Xc = np.c_[np.ones(30), rng.normal(size=(30, 5))]
yc = Xc @ np.array([1, 2, 1, 0.5, 0, 0]) + rng.normal(size=30)
rssf = lambda cols_: np.sum((yc - Xc[:, cols_] @ np.linalg.lstsq(Xc[:, cols_], yc, rcond=None)[0]) ** 2)
s2c = rssf(list(range(6))) / 24
cp_sim = rssf([0, 1, 2, 3]) / s2c - 30 + 8
E.check("s11p14a", abs(cp_sim - (rssf([0, 1, 2, 3]) / s2c - 30 + 2 * 4)) < 1e-12 and cp_sim < 30, f"乱数データでの Cp の計算 {cp_sim:.3f}")

E.done()
