"""第23章 検算スクリプト（判別分析）。
共通データ G1=(1,2),(4,5),(1,5),(2,4)、G2=(3,5),(6,8),(6,5),(5,6) について、
併合共分散・判別関数・D²・Φ(−D/2)・見かけ／一つ抜きの誤判別率・(4,4) の距離・AUC・kNN を numpy/scipy で再計算する。
SVM の6点の例は主問題・双対問題とも scipy.optimize（SLSQP）で解く。
等分散正規でベイズ判別が線形になる導出、D1²−D2²=2z、フィッシャー比の停留条件は sympy で確認。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, optimize


def cross2(u, v):
    return float(u[0] * v[1] - u[1] * v[0])

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)
Phi = stats.norm.cdf


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-12


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    if claims.size == 1:
        claims = np.full(xs.shape, claims[0])
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def front(term):
    txt = (ROOT / "content" / "ch23" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def chart(term, cid, idx=0):
    return [c for c in front(term).get("charts", []) if c["id"] == cid][idx]


# ---------------------------------------------------------------- 共通データ
G1 = np.array([[1, 2], [4, 5], [1, 5], [2, 4]], float)
G2 = np.array([[3, 5], [6, 8], [6, 5], [5, 6]], float)
X = np.vstack([G1, G2])
y = np.array([1] * 4 + [2] * 4)


def lda_fit(A, B):
    m1, m2 = A.mean(0), B.mean(0)
    W = (A - m1).T @ (A - m1) + (B - m2).T @ (B - m2)
    S = W / (len(A) + len(B) - 2)
    a = np.linalg.solve(S, m2 - m1)
    return m1, m2, S, a


def lda_z(x, m1, m2, a):
    return a @ (np.asarray(x, float) - (m1 + m2) / 2)


m1, m2, S, a = lda_fit(G1, G2)
Si = np.linalg.inv(S)
d = m2 - m1
D2 = d @ Si @ d
D = math.sqrt(D2)
T = "linear-discriminant-analysis"
check(T, "平均 (2,4)・(5,6)", allclose(np.r_[m1, m2], [2, 4, 5, 6], 1e-12), np.r_[m1, m2])
check(T, "各群の偏差平方和積和行列が [[6,3],[3,6]]",
      allclose((G1 - m1).T @ (G1 - m1), [6, 3, 3, 6], 1e-12) and allclose((G2 - m2).T @ (G2 - m2), [6, 3, 3, 6], 1e-12),
      ((G1 - m1).T @ (G1 - m1)).tolist())
check(T, "併合 S=[[2,1],[1,2]]（分母 n1+n2−2=6）", allclose(S, [2, 1, 1, 2], 1e-12), S.tolist())
check(T, "S⁻¹=(1/3)[[2,−1],[−1,2]]、|S|=3", allclose(Si * 3, [2, -1, -1, 2], 1e-12) and close(np.linalg.det(S), 3, 1e-12), (Si * 3).tolist())
check(T, "係数 a=(4,1)/3", allclose(a * 3, [4, 1], 1e-12), a * 3)
check(T, "中点 (3.5,5)", allclose((m1 + m2) / 2, [3.5, 5], 1e-12), (m1 + m2) / 2)
xs_test = rng.normal(size=(20, 2)) * 5
check(T, "z(x)=(4x1+x2−19)/3（乱数20点で一致）",
      allclose([lda_z(x, m1, m2, a) for x in xs_test], (4 * xs_test[:, 0] + xs_test[:, 1] - 19) / 3, 1e-10), "max差<1e-10")
z44 = lda_z([4, 4], m1, m2, a)
check(T, "例2 z(4,4)=1/3≈0.333>0 → G2", close(z44, 1 / 3, 1e-12) and z44 > 0, z44)
check(T, "例2 log9≈2.197、0.333<2.197 → G1", close(math.log(9), 2.197, 5e-4) and z44 < math.log(9), math.log(9))
zs = np.array([lda_z(x, m1, m2, a) for x in X])
check(T, "例3 (4,5) の z=2/3、(3,5) の z=−2/3", close(zs[1], 2 / 3, 1e-12) and close(zs[4], -2 / 3, 1e-12), (zs[1], zs[4]))
pred = np.where(zs > 0, 2, 1)
app_err = np.mean(pred != y)
wrong = [tuple(X[i].astype(int)) for i in np.where(pred != y)[0]]
check(T, "例3 見かけの誤判別 2/8=0.25（(4,5),(3,5)）", close(app_err, 0.25, 1e-12) and wrong == [(4, 5), (3, 5)], wrong)
check(T, "例3 D²=14/3、Φ(−D/2)≈0.140", close(D2, 14 / 3, 1e-12) and close(Phi(-D / 2), 0.140, 5e-4), (D2, Phi(-D / 2)))
check(T, "例1 (4,1)/3 は平均の差 (3,2) と向きが違う", abs(cross2(a, d)) > 1e-6, cross2(a, d))

# LDA の導出（sympy）：等分散共分散なら log f2/f1 は1次式で z に一致
s11, s12, s22, x1, x2, u1, u2, v1, v2 = sp.symbols("s11 s12 s22 x1 x2 u1 u2 v1 v2", real=True)
Sig = sp.Matrix([[s11, s12], [s12, s22]])
SigI = Sig.inv()
xv = sp.Matrix([x1, x2]); mu1 = sp.Matrix([u1, u2]); mu2 = sp.Matrix([v1, v2])
q = lambda m, Mi: ((xv - m).T * Mi * (xv - m))[0]
logratio = -q(mu2, SigI) / 2 + q(mu1, SigI) / 2
zsym = ((mu2 - mu1).T * SigI * (xv - (mu1 + mu2) / 2))[0]
check(T, "sympy: 共通Σで log f2/f1 = z(x)（一般の 2×2 Σ）", sp.simplify(logratio - zsym) == 0, "差=0")
H = sp.hessian(logratio, (x1, x2))
check(T, "sympy: log f2/f1 のヘッセ行列=0（2次の項が消える）", sp.simplify(H) == sp.zeros(2, 2), "H=0")
D1sq, D2sq = q(mu1, SigI), q(mu2, SigI)
check("mahalanobis-distance", "sympy: D1²−D2² = 2z(x)", sp.simplify(D1sq - D2sq - 2 * zsym) == 0, "差=0")

# 一般 p=3 で数値確認
for _ in range(3):
    M = rng.normal(size=(3, 3)); Sg = M @ M.T + np.eye(3)
    mA, mB, xx = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    lr = stats.multivariate_normal(mB, Sg).logpdf(xx) - stats.multivariate_normal(mA, Sg).logpdf(xx)
    zz = (mB - mA) @ np.linalg.solve(Sg, xx - (mA + mB) / 2)
    check(T, "p=3 乱数: log f2/f1 = z（数値）", close(lr, zz, 1e-9), (lr, zz))

# 事前確率でしきい値 log(π1/π2)：π2 を変えたときの境界の平行移動（乱数）
pi1, pi2 = 0.9, 0.1
pts = rng.normal(size=(2000, 2)) * 3 + 4
post_rule = pi2 * stats.multivariate_normal(m2, S).pdf(pts) > pi1 * stats.multivariate_normal(m1, S).pdf(pts)
z_rule = np.array([lda_z(p_, m1, m2, a) for p_ in pts]) > math.log(pi1 / pi2)
check(T, "π1=0.9: π2f2>π1f1 ⇔ z>log9（乱数2000点で全一致）", bool(np.all(post_rule == z_rule)), int(np.sum(post_rule != z_rule)))

# ---------------------------------------------------------------- 誤判別率（一つ抜き）
T = "misclassification-rate"
loo_pred, loo_scores = [], []
for i in range(8):
    mask = np.arange(8) != i
    Xi, yi = X[mask], y[mask]
    mm1, mm2, SS, aa = lda_fit(Xi[yi == 1], Xi[yi == 2])
    zi = lda_z(X[i], mm1, mm2, aa)
    loo_scores.append(zi)
    loo_pred.append(2 if zi > 0 else 1)
loo_pred = np.array(loo_pred)
loo_wrong = [tuple(X[i].astype(int)) for i in np.where(loo_pred != y)[0]]
check(T, "例2 一つ抜き法: (4,5),(3,5) の2個だけ誤判別、2/8=0.25", loo_wrong == [(4, 5), (3, 5)], (loo_wrong, np.round(loo_scores, 4)))
check(T, "例2 (4,5) を除いたルールでの判別得点 2.222>0", close(loo_scores[1], 2.222, 5e-4), loo_scores[1])
check(T, "例2 代入法 Φ(−2.160/2)=Φ(−1.080)≈0.140", close(D, 2.160, 5e-4) and close(D / 2, 1.080, 5e-4) and close(Phi(-D / 2), 0.140, 5e-4), Phi(-D / 2))
check(T, "例1 Φ(−1)≈0.1587", close(Phi(-1), 0.1587), Phi(-1))
c = math.log(9)
p21, p12 = Phi(-1 - c / 2), Phi(-1 + c / 2)
check(T, "例1 c/Δ≈1.099", close(c / 2, 1.099, 5e-4), c / 2)
check(T, "例1 P(2|1)=Φ(−2.099)≈0.0179", close(p21, 0.0179), p21)
check(T, "例1 P(1|2)=Φ(0.099)≈0.5393（Φ(0.0986) の値）", close(p12, 0.5393), p12)
tot = 0.9 * p21 + 0.1 * p12
check(T, "例1 全体 0.9×0.0179+0.1×0.5393≈0.0701", close(tot, 0.0701), tot)
check(T, "試験 Φ(−0.5)≈0.3085、Φ(−1.5)≈0.0668", close(Phi(-0.5), 0.3085) and close(Phi(-1.5), 0.0668), (Phi(-0.5), Phi(-1.5)))
# しきい値 c の P(2|1), P(1|2) の式を乱数で確認（Δ=2, c=log9, p=2, 20万点、許容 0.003）
mu_a, mu_b = np.zeros(2), np.array([2.0, 0.0])
n = 200000
xa = rng.normal(size=(n, 2)) + mu_a
xb = rng.normal(size=(n, 2)) + mu_b
za = (mu_b - mu_a) @ (xa - (mu_a + mu_b) / 2).T
zb = (mu_b - mu_a) @ (xb - (mu_a + mu_b) / 2).T
check(T, "乱数: Δ=2,c=0 で P(2|1)≈Φ(−1)（20万、許容0.003）", abs(np.mean(za > 0) - Phi(-1)) < 0.003, np.mean(za > 0))
check(T, "乱数: c=log9 で P(2|1)≈0.0179・P(1|2)≈0.5393（許容0.003）",
      abs(np.mean(za > c) - p21) < 0.003 and abs(np.mean(zb <= c) - p12) < 0.003, (np.mean(za > c), np.mean(zb <= c)))
check(T, "乱数: 群1で z~N(−Δ²/2, Δ²)（平均−2・分散4、許容0.03）", abs(za.mean() + 2) < 0.03 and abs(za.var() - 4) < 0.03, (za.mean(), za.var()))
ch = chart(T, "line-data")
xg = np.array(ch["params"]["x"])
check(T, "charts line-data Φ(−Δ/2)（9点、5e-5）", allclose(Phi(-xg / 2), ch["params"]["series"][0]["y"]), np.round(Phi(-xg / 2), 4))
check(T, "caption Δ=2 で 0.159、Δ=4 で 0.023", close(Phi(-1), 0.159, 5e-4) and close(Phi(-2), 0.023, 5e-4), (Phi(-1), Phi(-2)))
rc = chart(T, "roc-curve")["params"]
check(T, "charts roc-curve (μ0,μ1,σ,t)=(0,2,1,1)、両群の誤判別 1−0.841=0.159",
      rc == {"mu0": 0, "mu1": 2, "sigma": 1, "threshold": 1} and close(1 - Phi(1), 0.159, 5e-4) and close(Phi(1), 0.841, 5e-4), rc)

# ---------------------------------------------------------------- マハラノビス距離
T = "mahalanobis-distance"
x0 = np.array([4.0, 4.0])
D1s = (x0 - m1) @ Si @ (x0 - m1)
D2s = (x0 - m2) @ Si @ (x0 - m2)
check(T, "例1 D1²=8/3≈2.667、D1≈1.633", close(D1s, 8 / 3, 1e-12) and close(math.sqrt(D1s), 1.633, 5e-4), (D1s, math.sqrt(D1s)))
check(T, "例1 D2²=2、D2≈1.414 → 群2", close(D2s, 2, 1e-12) and close(math.sqrt(D2s), 1.414, 5e-4) and D2s < D1s, (D2s, math.sqrt(D2s)))
e1, e2 = np.linalg.norm(x0 - m1), np.linalg.norm(x0 - m2)
check(T, "例1 ユークリッド 2 と 2.236 → 群1", close(e1, 2, 1e-12) and close(e2, 2.236, 5e-4) and e1 < e2, (e1, e2))
check(T, "例1 D1²−D2²=2/3=2z", close(D1s - D2s, 2 / 3, 1e-12) and close(D1s - D2s, 2 * z44, 1e-12), D1s - D2s)
v_ = rng.normal(size=(10, 2))
check(T, "2×2 の v'S⁻¹v=(2v1²−2v1v2+2v2²)/3（乱数10本）",
      allclose([v @ Si @ v for v in v_], (2 * v_[:, 0] ** 2 - 2 * v_[:, 0] * v_[:, 1] + 2 * v_[:, 1] ** 2) / 3, 1e-12), "一致")
Sg = np.array([[1, 0.8], [0.8, 1]])
Sgi = np.linalg.inv(Sg)
check(T, "例2 Σ⁻¹=(1/0.36)[[1,−0.8],[−0.8,1]]", allclose(Sgi * 0.36, [1, -0.8, -0.8, 1], 1e-12), Sgi)
A_, B_ = np.array([1.5, 1.5]), np.array([1.5, -1.5])
dA, dB = A_ @ Sgi @ A_, B_ @ Sgi @ B_
check(T, "例2 A の D²=0.9/0.36=2.5、B の D²=8.1/0.36=22.5", close(dA, 2.5, 1e-10) and close(dB, 22.5, 1e-10), (dA, dB))
check(T, "例2 ユークリッド √4.5≈2.121", close(math.sqrt(4.5), 2.121, 5e-4), math.sqrt(4.5))
check(T, "例2 χ²₂ 上側確率 e^{−c/2}: A≈0.287、B≈1.3e−5",
      close(stats.chi2.sf(2.5, 2), math.exp(-1.25), 1e-12) and close(math.exp(-1.25), 0.287, 5e-4)
      and close(stats.chi2.sf(22.5, 2), 1.3e-5, 5e-7), (stats.chi2.sf(2.5, 2), stats.chi2.sf(22.5, 2)))
check(T, "例2 χ²₂ 上側1%点 9.210、B だけ超える", close(stats.chi2.ppf(0.99, 2), 9.210, 5e-4) and dA < 9.21 < dB, stats.chi2.ppf(0.99, 2))
# D² ~ χ²_p（乱数、p=3）
M = rng.normal(size=(3, 3)); Sg3 = M @ M.T + np.eye(3); mu3 = rng.normal(size=3)
Xs = rng.multivariate_normal(mu3, Sg3, size=100000)
d2s = np.einsum("ij,jk,ik->i", Xs - mu3, np.linalg.inv(Sg3), Xs - mu3)
ks = stats.kstest(d2s, "chi2", args=(3,))
check(T, "乱数: N_3 で D²~χ²₃（10万点、KS p>0.01、平均≈3 許容0.03）", ks.pvalue > 0.01 and abs(d2s.mean() - 3) < 0.03, (ks.pvalue, d2s.mean()))
# 楕円の半径 c√λ
w, V = np.linalg.eigh(Sg)
pt = 2 * math.sqrt(w[1]) * V[:, 1]
check(T, "等高線: 固有ベクトル方向の半径 c√λ で D=c（c=2）", close(math.sqrt(pt @ Sgi @ pt), 2, 1e-12), math.sqrt(pt @ Sgi @ pt))
# 正則な線形変換で不変
Aff = np.array([[2.0, 1.0], [0.5, 3.0]])
t_ = (Aff @ (x0 - m1)) @ np.linalg.inv(Aff @ S @ Aff.T) @ (Aff @ (x0 - m1))
check(T, "正則な線形変換で D² 不変", close(t_, D1s, 1e-10), t_)
mc = chart(T, "mahalanobis-contour")["params"]
check(T, "charts mahalanobis-contour の params がスライダー範囲内", -0.9 <= mc["rho"] <= 0.9 and 0.5 <= mc["sx"] <= 3, mc)

# ---------------------------------------------------------------- フィッシャー
T = "fisher-linear-discriminant-function"
lam = lambda v: (v @ d) ** 2 / (v @ S @ v)
rows = [((1, 0), 9, 2, 4.5), ((0, 1), 4, 2, 2.0), ((1, 1), 25, 6, 4.167), ((4, 1), 196, 42, 4.667)]
for v, num, den, lv in rows:
    v = np.array(v, float)
    check(T, f"例1 表 a={tuple(v.astype(int))}: {num}/{den}={lv}",
          close((v @ d) ** 2, num, 1e-12) and close(v @ S @ v, den, 1e-12) and close(lam(v), lv, 5e-4), lam(v))
check(T, "例1 最大値 14/3=D²", close(lam(np.array([4.0, 1])), D2, 1e-12), D2)
res = optimize.minimize_scalar(lambda th: -lam(np.array([math.cos(th), math.sin(th)])), bounds=(0, math.pi), method="bounded")
vopt = np.array([math.cos(res.x), math.sin(res.x)])
check(T, "数値最大化: 最適方向 ∝ (4,1)、最大値 14/3", abs(cross2(vopt, [4, 1])) < 1e-4 and close(-res.fun, 14 / 3, 1e-8), (vopt / vopt[1], -res.fun))
check(T, "d=(3,2) の向きの λ=169/38≈4.447", close(lam(d), 169 / 38, 1e-12) and close(169 / 38, 4.447, 5e-4), lam(d))
yp1, yp2 = G1 @ [4, 1], G2 @ [4, 1]
check(T, "例2 y=4x1+x2: G1=6,21,9,12（平均12）、G2=17,32,29,26（平均26）",
      list(yp1) == [6, 21, 9, 12] and list(yp2) == [17, 32, 29, 26] and yp1.mean() == 12 and yp2.mean() == 26, (yp1, yp2))
ss1, ss2 = ((yp1 - 12) ** 2).sum(), ((yp2 - 26) ** 2).sum()
check(T, "例2 偏差平方和 126・126、プール分散 42、λ=14²/42", ss1 == 126 and ss2 == 126 and (ss1 + ss2) / 6 == 42 and close(196 / 42, 4.667, 5e-4), (ss1, ss2))
check(T, "例2 境界 y=19（中点）", (12 + 26) / 2 == 19, 19)
# B/W 比との比例
n1 = n2 = 4
Bm = n1 * n2 / (n1 + n2) * np.outer(d, d)
Wm = 6 * S
vv = rng.normal(size=(10, 2))
ratio = np.array([(v @ Bm @ v) / (v @ Wm @ v) / lam(v) for v in vv])
check(T, "a'Ba/a'Wa = λ(a)×n1n2/{(n1+n2)(n1+n2−2)}（定数倍）", allclose(ratio, n1 * n2 / 8 / 6, 1e-12), ratio[0])
# 停留条件（sympy）
a1, a2, d1s, d2s_ = sp.symbols("a1 a2 d1 d2", real=True)
av = sp.Matrix([a1, a2]); dv = sp.Matrix([d1s, d2s_])
lam_s = (av.T * dv)[0] ** 2 / (av.T * Sig * av)[0]
grad = sp.Matrix([sp.diff(lam_s, t) for t in (a1, a2)])
sol = SigI * dv
gsub = sp.simplify(grad.subs({a1: sol[0], a2: sol[1]}))
check(T, "sympy: a=Σ⁻¹d で勾配=0", gsub == sp.zeros(2, 1), "勾配=0")
lmax = sp.simplify(lam_s.subs({a1: sol[0], a2: sol[1]}) - (dv.T * SigI * dv)[0])
check(T, "sympy: 最大値 = d'Σ⁻¹d", lmax == 0, "差=0")
# 3群の一般化固有値問題：判別軸の本数 min(K−1,p)
K, p = 3, 4
means = rng.normal(size=(K, p)); Bk = sum(np.outer(mm - means.mean(0), mm - means.mean(0)) for mm in means)
check(T, "K=3,p=4: B の階数 = min(K−1,p)=2", np.linalg.matrix_rank(Bk) == 2, np.linalg.matrix_rank(Bk))
fp = chart(T, "fisher-projection")["params"]
check(T, "charts fisher-projection の params がスライダー範囲内", 0 <= fp["angle"] <= 180 and -0.9 <= fp["rho"] <= 0.9, fp)

# ---------------------------------------------------------------- ベイズ判別
T = "bayes-discriminant-rule"
thr = math.log(19)
xc = 115 + thr / 0.3
check(T, "例1 係数 (130−100)/10²=0.3、log19≈2.944", close(math.log(19), 2.944, 5e-4), thr)
lr_ = lambda xx: stats.norm.logpdf(xx, 130, 10) - stats.norm.logpdf(xx, 100, 10)
check(T, "例1 log f2/f1 = 0.3(x−115)（数点）", allclose([lr_(t) for t in (90, 115, 140)], [0.3 * (t - 115) for t in (90, 115, 140)], 1e-10), "一致")
check(T, "例1 境界 115+9.815=124.815", close(thr / 0.3, 9.815, 5e-4) and close(xc, 124.815, 5e-4), xc)
post120 = 1 / (1 + 19 * math.exp(-1.5))
post_direct = 0.05 * stats.norm.pdf(120, 130, 10) / (0.05 * stats.norm.pdf(120, 130, 10) + 0.95 * stats.norm.pdf(120, 100, 10))
check(T, "例1 e^{−1.5}≈0.2231、P(患者|120)≈0.1909", close(math.exp(-1.5), 0.2231) and close(post120, 0.1909) and close(post_direct, post120, 1e-12), post120)
p21b = 1 - Phi((xc - 100) / 10)
p12b = Phi((xc - 130) / 10)
totb = 0.95 * p21b + 0.05 * p12b
check(T, "例1 P(2|1)≈0.0065、P(1|2)≈0.3020", close(p21b, 0.0065) and close(p12b, 0.3020), (p21b, p12b))
check(T, "例1 全体誤判別 ≈0.0213", close(totb, 0.0213), totb)
tot115 = 0.95 * (1 - Phi(1.5)) + 0.05 * Phi(-1.5)
check(T, "例1 中点115 の全体誤判別 0.0668", close(tot115, 0.0668), tot115)
grid = np.linspace(100, 150, 50001)
errs = 0.95 * (1 - Phi((grid - 100) / 10)) + 0.05 * Phi((grid - 130) / 10)
check(T, "例1 しきい値の格子探索で最小点≈124.815（誤判別最小）", abs(grid[np.argmin(errs)] - xc) < 2e-3, grid[np.argmin(errs)])
check(T, "例2 等しい事前確率 P(G2|x)=1/(1+e^{−1/3})≈0.5826", close(1 / (1 + math.exp(-1 / 3)), 0.5826), 1 / (1 + math.exp(-1 / 3)))
pp = 1 / (1 + 9 * math.exp(-1 / 3))
pp_direct = 0.1 * stats.multivariate_normal(m2, S).pdf(x0) / (0.1 * stats.multivariate_normal(m2, S).pdf(x0) + 0.9 * stats.multivariate_normal(m1, S).pdf(x0))
check(T, "例2 π=(0.9,0.1) で ≈0.1343（密度から直接計算と一致）", close(pp, 0.1343) and close(pp, pp_direct, 1e-12), pp)
check(T, "例2 損失10倍: log0.9≈−0.105、1/3>−0.105 → 不良品", close(math.log(0.9), -0.105, 5e-4) and 1 / 3 > math.log(0.9), math.log(0.9))
# sympy: 事後確率のロジスティック形
zz_, p1_, p2_ = sp.symbols("z pi1 pi2", positive=True)
post = p2_ * sp.exp(zz_) / (p2_ * sp.exp(zz_) + p1_)  # f2/f1=e^z
check(T, "sympy: P(G2|x)=1/(1+(π1/π2)e^{−z})", sp.simplify(post - 1 / (1 + p1_ / p2_ * sp.exp(-zz_))) == 0, "差=0")
# 異なる Σ の一般式（数値、p=2）
for _ in range(3):
    M1 = rng.normal(size=(2, 2)); M2 = rng.normal(size=(2, 2))
    S1_ = M1 @ M1.T + np.eye(2); S2_ = M2 @ M2.T + np.eye(2)
    mA, mB, xx = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    lhs = math.log(0.3) + stats.multivariate_normal(mB, S2_).logpdf(xx) - math.log(0.7) - stats.multivariate_normal(mA, S1_).logpdf(xx)
    dd1 = (xx - mA) @ np.linalg.solve(S1_, xx - mA); dd2 = (xx - mB) @ np.linalg.solve(S2_, xx - mB)
    rhs = -dd2 / 2 + dd1 / 2 - 0.5 * math.log(np.linalg.det(S2_) / np.linalg.det(S1_)) + math.log(0.3 / 0.7)
    check(T, "正規の対数事後比の式（異なる Σ、乱数）", close(lhs, rhs, 1e-10), (lhs, rhs))
ba = chart(T, "bayes-area")["params"]
check(T, "charts bayes-area: prior 0.05、感度 Φ(0.5185)≈0.698、特異度 Φ(2.4815)≈0.993",
      ba["prior"] == 0.05 and close(1 - p12b, ba["sensitivity"], 5e-4) and close(1 - p21b, ba["specificity"], 5e-4), (1 - p12b, 1 - p21b))
db = chart(T, "discriminant-boundary")["params"]
check(T, "charts discriminant-boundary の params がスライダー範囲内", 0.05 <= db["prior"] <= 0.95 and 0.25 <= db["ratio"] <= 4, db)

# ---------------------------------------------------------------- 二次判別
T = "quadratic-discriminant-analysis"
g = lambda xx: (stats.norm.logpdf(xx, 2, 2) - stats.norm.logpdf(xx, 0, 1))
gt = lambda xx: -(xx - 2) ** 2 / 8 - math.log(2) + xx ** 2 / 2
check(T, "例 g2−g1=−(x−2)²/8−log2+x²/2（数点）", allclose([g(t) for t in (-3, 0, 1.7)], [gt(t) for t in (-3, 0, 1.7)], 1e-12), "一致")
xs_ = sp.symbols("x")
check(T, "sympy: 8倍して 3x²+4x−4−8log2", sp.expand(8 * (-(xs_ - 2) ** 2 / 8 - sp.log(2) + xs_ ** 2 / 2)) == sp.expand(3 * xs_ ** 2 + 4 * xs_ - 4 - 8 * sp.log(2)), "一致")
check(T, "8log2≈5.545", close(8 * math.log(2), 5.545, 5e-4), 8 * math.log(2))
roots = np.sort(np.roots([3, 4, -4 - 8 * math.log(2)]))
check(T, "根 ≈ −2.571, 1.238", allclose(roots, [-2.571, 1.238], 5e-4), roots)
check(T, "g2−g1(−3)≈0.682>0、g2−g1(−2)≈−0.693<0", close(g(-3), 0.682, 5e-4) and close(g(-2), -0.693, 5e-4), (g(-3), g(-2)))
r1, r2 = roots
e1q = Phi(r1) + 1 - Phi(r2)
e2q = Phi((r2 - 2) / 2) - Phi((r1 - 2) / 2)
check(T, "誤判別 群1→群2≈0.113、群2→群1≈0.340、平均≈0.227", close(e1q, 0.113, 5e-4) and close(e2q, 0.340, 5e-4) and close((e1q + e2q) / 2, 0.227, 5e-4), (e1q, e2q, (e1q + e2q) / 2))
e_mid = ((1 - Phi(1)) + Phi((1 - 2) / 2)) / 2
check(T, "中点 x=1 の1境界の場合 0.234", close(e_mid, 0.234, 5e-4), e_mid)
# QDA が最適（しきい値1つの全探索より小さい）
tt = np.linspace(-1, 4, 5001)
best1 = np.min(((1 - Phi(tt)) + Phi((tt - 2) / 2)) / 2)
check(T, "1つの境界の最良値よりも二次判別が小さい", (e1q + e2q) / 2 < best1, best1)
check(T, "パラメータ数 K·p(p+1)/2（p=2,K=2 で 6）", 2 * 2 * 3 / 2 == 6, 6)
qd = chart(T, "discriminant-boundary")["params"]
check(T, "charts discriminant-boundary の params がスライダー範囲内", 0.25 <= qd["ratio"] <= 4 and 0 <= qd["delta"] <= 4, qd)

# ---------------------------------------------------------------- 混同行列
T = "confusion-matrix"
TP, FN, FP, TN = 80, 20, 90, 810
check(T, "例1 正解率0.890・誤判別率0.110", close((TP + TN) / 1000, 0.89, 1e-12) and close((FP + FN) / 1000, 0.11, 1e-12), (TP + TN) / 1000)
check(T, "例1 感度0.800・特異度0.900", close(TP / (TP + FN), 0.8, 1e-12) and close(TN / (FP + TN), 0.9, 1e-12), "OK")
check(T, "例1 適合率 80/170≈0.4706、陰性的中率 810/830≈0.9759", close(80 / 170, 0.4706) and close(810 / 830, 0.9759), (80 / 170, 810 / 830))
ppv = 0.8 * 0.1 / (0.8 * 0.1 + 0.1 * 0.9)
check(T, "ベイズの定理 0.08/0.17≈0.4706", close(ppv, 80 / 170, 1e-12), ppv)
cm = np.zeros((2, 2), int)
for yt, yp in zip(y, pred):
    cm[0 if yt == 2 else 1, 0 if yp == 2 else 1] += 1
check(T, "例2 判別分析の混同行列 [[3,1],[1,3]]・感度0.75・特異度0.75", cm.tolist() == [[3, 1], [1, 3]], cm.tolist())
mo = chart(T, "mosaic")["params"]
check(T, "charts mosaic table=[[80,20],[90,810]]", mo["table"] == [[80, 20], [90, 810]], mo["table"])
bc = chart(T, "bayes-area")["params"]
check(T, "charts bayes-area (0.1,0.8,0.9) が例1と一致", (bc["prior"], bc["sensitivity"], bc["specificity"]) == (0.1, 0.8, 0.9)
      and close(TP / 1000, bc["prior"] * bc["sensitivity"], 1e-12) and close(FP / 1000, (1 - bc["prior"]) * (1 - bc["specificity"]), 1e-12), bc)

# ---------------------------------------------------------------- 感度・特異度
T = "sensitivity-and-specificity"
for t, se, sp_, J, lrp in [(0.5, 0.9332, 0.6915, 0.6247, 3.025), (1.0, 0.8413, 0.8413, 0.6827, 5.303), (1.5, 0.6915, 0.9332, 0.6247, 10.350)]:
    se_, spc = Phi(2 - t), Phi(t)
    check(T, f"例1 t={t}: 感度{se}・特異度{sp_}・J{J}・LR+{lrp}",
          close(se_, se) and close(spc, sp_) and close(se_ + spc - 1, J) and close(se_ / (1 - spc), lrp, 5e-4), (se_, spc, se_ + spc - 1, se_ / (1 - spc)))
tg = np.linspace(-2, 4, 60001)
check(T, "J は中点 t=1 で最大", abs(tg[np.argmax(Phi(2 - tg) + Phi(tg) - 1)] - 1) < 1e-3, tg[np.argmax(Phi(2 - tg) + Phi(tg) - 1)])
check(T, "LR+ はしきい値とともに増加", bool(np.all(np.diff(Phi(2 - tg) / (1 - Phi(tg))) > 0)), "単調増加")
LRp, LRm = 0.8 / 0.1, 0.2 / 0.9
check(T, "例2 LR+=8、LR−≈0.2222", close(LRp, 8, 1e-12) and close(LRm, 0.2222), (LRp, LRm))
po = LRp / 9
check(T, "例2 陽性: 検査後オッズ 8/9、確率 8/17≈0.4706", close(po, 8 / 9, 1e-12) and close(po / (1 + po), 0.4706), po / (1 + po))
no = LRm / 9
npost = no / (1 + no)
direct = 0.1 * 0.2 / (0.1 * 0.2 + 0.9 * 0.9)
check(T, "例2 陰性: オッズ≈0.0247、確率≈0.0241（直接計算と一致）", close(no, 0.0247) and close(npost, 0.0241) and close(npost, direct, 1e-12), (no, npost))
sc = chart(T, "roc-curve")["params"]
check(T, "charts roc-curve (0,2,1,1) が例1と一致", sc == {"mu0": 0, "mu1": 2, "sigma": 1, "threshold": 1}, sc)
sb = chart(T, "bayes-area")["params"]
check(T, "charts bayes-area (0.1,0.8,0.9) が例2と一致", (sb["prior"], sb["sensitivity"], sb["specificity"]) == (0.1, 0.8, 0.9), sb)

# ---------------------------------------------------------------- 適合率・再現率・F値
T = "precision-recall-f-measure"
for (tp, fn, fp, tn), (P_, R_, F_, Acc) in [((80, 20, 90, 810), (0.4706, 0.8, 0.5926, 0.890)), ((8, 2, 99, 891), (0.0748, 0.8, 0.1368, 0.899))]:
    Pc, Rc = tp / (tp + fp), tp / (tp + fn)
    Fc = 2 * Pc * Rc / (Pc + Rc)
    check(T, f"例1 表 TP={tp}: P={P_}・R={R_}・F={F_}・正解率={Acc}",
          close(Pc, P_) and close(Rc, R_) and close(Fc, F_) and close(2 * tp / (2 * tp + fp + fn), Fc, 1e-12) and close((tp + tn) / 1000, Acc, 1e-12), (Pc, Fc))
    pr = (tp + fn) / 1000
    check(T, f"例1 TP={tp} の行が感度0.8・特異度0.9 の期待度数", close(tp, 1000 * pr * 0.8, 1e-9) and close(fp, 1000 * (1 - pr) * 0.1, 1e-9), pr)
check(T, "例1 160/270≈0.5926、16/117≈0.1368", close(160 / 270, 0.5926) and close(16 / 117, 0.1368), (160 / 270, 16 / 117))
check(T, "例1 全員陰性で正解率0.99", close(990 / 1000, 0.99, 1e-12), 0.99)
Pp, Rr = 80 / 170, 0.8
F2 = 5 * Pp * Rr / (4 * Pp + Rr)
check(T, "例1 F2≈0.7018（丸めた P=0.4706 でも同値）", close(F2, 0.7018) and close(5 * 0.4706 * 0.8 / (4 * 0.4706 + 0.8), 0.7018), F2)
check(T, "P=1,R=0.01: 相加平均≈0.5、F≈0.02", close(1.01 / 2, 0.505, 1e-12) and close(2 * 0.01 / 1.01, 0.0198, 5e-4), 2 * 0.01 / 1.01)
pi_ = 0.1


def prf(t):
    R = 1 - Phi(t - 2)
    P = pi_ * R / (pi_ * R + (1 - pi_) * (1 - Phi(t)))
    return R, P, 2 * P * R / (P + R)


for t, (R0, P0, F0) in [(1, (0.8413, 0.3708, 0.5147)), (1.75, (0.5987, 0.6241, 0.6112)), (2, (0.5, 0.7095, 0.5866))]:
    Rv, Pv, Fv = prf(t)
    check(T, f"例2 t={t}: R={R0}・P={P0}・F={F0}", close(Rv, R0) and close(Pv, P0) and close(Fv, F0), (Rv, Pv, Fv))
lp = chart(T, "line-data")["params"]
tx = np.array(lp["x"])
Rs, Ps, Fs = np.array([prf(t) for t in tx]).T
check(T, "charts line-data 再現率（17点、5e-5）", allclose(Rs, lp["series"][0]["y"]), np.round(Rs, 4))
check(T, "charts line-data 適合率（17点、5e-5）", allclose(Ps, lp["series"][1]["y"]), np.round(Ps, 4))
check(T, "charts line-data F値（17点、5e-5）", allclose(Fs, lp["series"][2]["y"]), np.round(Fs, 4))
resF = optimize.minimize_scalar(lambda t: -prf(t)[2], bounds=(0, 3), method="bounded")
check(T, "caption F値は t=1.7 付近で最大（約0.612）", abs(resF.x - 1.7) < 0.05 and close(-resF.fun, 0.612, 5e-4), (resF.x, -resF.fun))
check(T, "F 最大のしきい値 > ユーデン最大の t=1", resF.x > 1, resF.x)
pb = chart(T, "bayes-area")["params"]
check(T, "charts bayes-area (0.1,0.8,0.9) が例1と一致", (pb["prior"], pb["sensitivity"], pb["specificity"]) == (0.1, 0.8, 0.9), pb)
# 適合率の式（p=1%）
check(T, "適合率の式 p=0.01: 0.008/(0.008+0.099)≈0.0748", close(0.8 * 0.01 / (0.8 * 0.01 + 0.1 * 0.99), 0.0748), 0.8 * 0.01 / (0.8 * 0.01 + 0.1 * 0.99))

# ---------------------------------------------------------------- ROC・AUC
T = "roc-curve-and-auc"
s_neg = G1 @ [4, 1]; s_pos = G2 @ [4, 1]
check(T, "例1 得点 陰性 6,9,12,21・陽性 17,26,29,32", sorted(s_neg) == [6, 9, 12, 21] and sorted(s_pos) == [17, 26, 29, 32], (sorted(s_neg), sorted(s_pos)))
ths = [32, 29, 26, 21, 17, 12, 9, 6]
tpr = [np.mean(s_pos >= t) for t in ths]; fpr = [np.mean(s_neg >= t) for t in ths]
check(T, "例1 表の感度", allclose(tpr, [0.25, 0.5, 0.75, 0.75, 1, 1, 1, 1], 1e-12), tpr)
check(T, "例1 表の偽陽性率", allclose(fpr, [0, 0, 0, 0.25, 0.25, 0.5, 0.75, 1], 1e-12), fpr)
fx = np.r_[0, fpr]; ty = np.r_[0, tpr]
auc_area = np.sum(np.diff(fx) * ty[1:])
pairs = np.mean([(sp1 > sn) + 0.5 * (sp1 == sn) for sp1 in s_pos for sn in s_neg])
U = stats.mannwhitneyu(s_pos, s_neg, alternative="greater").statistic
check(T, "例1 AUC 面積=0.9375=15/16=U/16（scipy の U=15）", close(auc_area, 0.9375, 1e-12) and close(pairs, 15 / 16, 1e-12) and U == 15, (auc_area, pairs, U))
ld = chart(T, "line-data")["params"]
check(T, "charts line-data 経験ROC の点列が表と一致", ld["x"] == list(fx) and ld["series"][0]["y"] == list(ty), list(zip(fx, ty)))
check(T, "charts line-data 対角線 y=x", ld["series"][1]["y"] == ld["x"], ld["series"][1]["y"])
check(T, "caption 15/16=0.9375", close(15 / 16, 0.9375, 1e-12), 15 / 16)
check(T, "例2 Φ(2/√2)=Φ(1.414)≈0.9214", close(Phi(math.sqrt(2)), 0.9214) and close(math.sqrt(2), 1.414, 5e-4), Phi(math.sqrt(2)))
check(T, "例2 Φ(2.160/√2)=Φ(1.528)≈0.9367", close(D / math.sqrt(2), 1.528, 5e-4) and close(Phi(D / math.sqrt(2)), 0.9367), Phi(D / math.sqrt(2)))
# AUC=∫TPR dFPR = P(S1>S0)（数値積分）
from scipy import integrate
for (m0, s0, m1_, s1) in [(0, 1, 2, 1), (0, 1, 1, 2), (1, 0.5, 1.5, 1.5)]:
    val = integrate.quad(lambda t: (1 - stats.norm.cdf(t, m1_, s1)) * stats.norm.pdf(t, m0, s0), -np.inf, np.inf)[0]
    check(T, f"積分 ∫P(S1>t)f0(t)dt = Φ(Δμ/√(σ0²+σ1²))（{m0},{s0},{m1_},{s1}）", close(val, Phi((m1_ - m0) / math.hypot(s0, s1)), 1e-8), val)
u = np.linspace(0, 1, 200001)
roc_y = 1 - Phi(stats.norm.ppf(1 - u) - 2)  # TPR as a function of FPR=u
check(T, "ROC 曲線の台形積分 ≈ Φ(√2)（N(0,1) vs N(2,1)）", close(np.trapezoid(roc_y, u), Phi(math.sqrt(2)), 1e-5), np.trapezoid(roc_y, u))
check(T, "LDA 得点の AUC=Φ(Δ/√2)：N(∓Δ²/2,Δ²) から（Δ=2）", close(Phi((2 - (-2)) / math.sqrt(8)), Phi(2 / math.sqrt(2)), 1e-12), Phi(math.sqrt(2)))
a0 = rng.normal(0, 1, 3000); a1 = rng.normal(2, 1, 3000)
emp = stats.mannwhitneyu(a1, a0).statistic / 9e6
check(T, "乱数: 経験 AUC ≈0.9214（各3000、許容0.01）", abs(emp - Phi(math.sqrt(2))) < 0.01, emp)
check(T, "AUC は単調変換で不変（exp 変換）", stats.mannwhitneyu(np.exp(a1), np.exp(a0)).statistic / 9e6 == emp, "不変")
rp = chart(T, "roc-curve")["params"]
check(T, "charts roc-curve params (0,1.5,1,0.75) がスライダー範囲内", rp["mu0"] == 0 and 0 <= rp["mu1"] <= 4 and -2 <= rp["threshold"] <= 4 and rp["sigma"] == 1, rp)

# ---------------------------------------------------------------- kNN
T = "k-nearest-neighbors"
d2 = ((X - x0) ** 2).sum(1)
order = np.lexsort((np.arange(8), d2))
tab = [(tuple(X[i].astype(int)), y[i], d2[i]) for i in order]
claim = [((4, 5), 1, 1), ((3, 5), 2, 2), ((2, 4), 1, 4), ((6, 5), 2, 5), ((5, 6), 2, 5), ((1, 5), 1, 10), ((1, 2), 1, 13), ((6, 8), 2, 20)]
check(T, "例 距離の2乗の順位表", [(p_, int(g_), int(dd)) for p_, g_, dd in tab] == claim, tab)
check(T, "例 距離 1, 1.414, 2, 2.236, 2.236, 3.162, 3.606, 4.472", allclose(np.sqrt(np.sort(d2)), [1, 1.414, 2, 2.236, 2.236, 3.162, 3.606, 4.472], 5e-4), np.round(np.sqrt(np.sort(d2)), 3))


def knn(Xtr, ytr, x, k):
    dd = ((Xtr - x) ** 2).sum(1)
    srt = np.sort(dd)
    tie = k < len(dd) and srt[k - 1] == srt[k]
    idx = np.argsort(dd, kind="stable")[:k]
    votes = np.bincount(ytr[idx], minlength=3)
    return (1 if votes[1] > votes[2] else 2), votes[2] / k, tie


for k, gg, pv in [(1, 1, None), (3, 1, 1 / 3), (5, 2, 0.6)]:
    gk, p2, tie = knn(X, y, x0, k)
    check(T, f"例 k={k} → 群{gg}" + (f"（群2の割合 {pv:.3f}）" if pv else ""), gk == gg and not tie and (pv is None or close(p2, pv, 1e-12)), (gk, p2, tie))
from itertools import combinations


def knn_all(Xtr, ytr, x, k):
    """第k近傍に同順位があるとき、同順位の点の選び方をすべて試して得られる判別結果の集合"""
    dd = ((Xtr - x) ** 2).sum(1)
    kth = np.sort(dd)[k - 1]
    inside = np.where(dd < kth)[0]
    tied = np.where(dd == kth)[0]
    outs = set()
    for comb in combinations(tied, k - len(inside)):
        idx = np.r_[inside, list(comb)].astype(int)
        v = np.bincount(ytr[idx], minlength=3)
        outs.add(1 if v[1] > v[2] else 2)
    return outs, len(tied) > k - len(inside)


loo_sets, ties = [], []
for i in range(8):
    mask = np.arange(8) != i
    outs, tie = knn_all(X[mask], y[mask], X[i], 3)
    loo_sets.append(outs); ties.append(tie)
wr = [tuple(X[i].astype(int)) for i in range(8) if loo_sets[i] == {3 - y[i]}]
uniq = all(len(o) == 1 for o in loo_sets)
check(T, "例 k=3 一つ抜き: (4,5),(3,5) だけ誤判別、2/8=0.25（第3近傍の同順位はどの選び方でも結果が同じ）",
      wr == [(4, 5), (3, 5)] and uniq, (wr, f"同順位あり {sum(ties)} 点、結果は一意={uniq}"))
check(T, "k=1 の再代入誤判別率 0", np.mean([knn(X, y, X[i], 1)[0] != y[i] for i in range(8)]) == 0, 0)
kb = chart(T, "knn-boundary")["params"]
check(T, "charts knn-boundary k=5（奇数、範囲内）", kb["k"] % 2 == 1 and 1 <= kb["k"] <= 31, kb)

# ---------------------------------------------------------------- SVM
T = "support-vector-machine"
Xs6 = np.array([[0, 0], [1, 0], [0, 1], [2, 2], [3, 2], [2, 3]], float)
ys6 = np.array([-1, -1, -1, 1, 1, 1], float)
cons = [{"type": "ineq", "fun": (lambda v, i=i: ys6[i] * (v[:2] @ Xs6[i] + v[2]) - 1)} for i in range(6)]
pr_ = optimize.minimize(lambda v: 0.5 * (v[0] ** 2 + v[1] ** 2), x0=[1, 1, -3], constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
w_, b_ = pr_.x[:2], pr_.x[2]
check(T, "主問題（SLSQP）: w=(2/3,2/3)、b=−5/3（許容1e-6）", pr_.success and allclose(w_, [2 / 3, 2 / 3], 1e-6) and close(b_, -5 / 3, 1e-6), (w_, b_))
check(T, "‖w‖=(2/3)√2≈0.9428、幅 2/‖w‖=3/√2≈2.121",
      close(np.linalg.norm(w_), 0.9428, 1e-4) and close(2 / np.linalg.norm(w_), 2.121, 5e-4) and close(3 / math.sqrt(2), 2.121, 5e-4), 2 / np.linalg.norm(w_))
check(T, "直線 x1+x2=1 と 4 の距離 3/√2", close(abs(4 - 1) / math.sqrt(2), 2 / (2 / 3 * math.sqrt(2)), 1e-12), 3 / math.sqrt(2))
wt, bt = np.array([2 / 3, 2 / 3]), -5 / 3
mg = ys6 * (Xs6 @ wt + bt)
check(T, "y_i f(x_i): (1,0),(0,1),(2,2) で1、(0,0),(3,2),(2,3) で 5/3", allclose(mg, [5 / 3, 1, 1, 1, 5 / 3, 5 / 3], 1e-12), mg)
Kmat = (ys6[:, None] * Xs6) @ (ys6[:, None] * Xs6).T
du = optimize.minimize(lambda al: 0.5 * al @ Kmat @ al - al.sum(), x0=np.full(6, 0.1),
                       jac=lambda al: Kmat @ al - 1,
                       constraints=[{"type": "eq", "fun": lambda al: al @ ys6, "jac": lambda al: ys6}],
                       bounds=[(0, None)] * 6, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
al = du.x
check(T, "双対（SLSQP）: α=(0, 2/9, 2/9, 4/9, 0, 0)（許容1e-5）", du.success and allclose(al, [0, 2 / 9, 2 / 9, 4 / 9, 0, 0], 1e-5), np.round(al, 6))
check(T, "w=Σα_i y_i x_i=(2/3,2/3)、Σα_i y_i=0", allclose((al * ys6) @ Xs6, [2 / 3, 2 / 3], 1e-5) and abs(al @ ys6) < 1e-8, (al * ys6) @ Xs6)
check(T, "サポートベクトルは (1,0),(0,1),(2,2) の3点", list(np.where(al > 1e-6)[0]) == [1, 2, 3], np.where(al > 1e-6)[0])
alpha = 2 / 9
check(T, "対称な形 3α=2/3 → α=2/9、‖w‖²=Σα=8/9", close(3 * alpha, 2 / 3, 1e-12) and close(al.sum(), 8 / 9, 1e-5) and close(wt @ wt, 8 / 9, 1e-12), al.sum())
# 双対解は一意（(1,0),(0,1),(2,2) で w を作る連立方程式）
Msv = np.array([[-1, 0, 2], [0, -1, 2], [-1, -1, 1]], float)  # w1, w2, Σαy
sol3 = np.linalg.solve(Msv, [2 / 3, 2 / 3, 0])
check(T, "SV 3点の α は連立方程式から一意に (2/9,2/9,4/9)", allclose(sol3, [2 / 9, 2 / 9, 4 / 9], 1e-12), sol3)
f = lambda p_: wt @ p_ + bt
check(T, "新しい点 (1.5,1.5): f=1/3>0、(2,0): f=−1/3<0", close(f(np.array([1.5, 1.5])), 1 / 3, 1e-12) and close(f(np.array([2.0, 0])), -1 / 3, 1e-12), (f(np.array([1.5, 1.5])), f(np.array([2.0, 0]))))
check(T, "x1+x2 の範囲: 群−1 最大1、群+1 最小4", Xs6[ys6 < 0].sum(1).max() == 1 and Xs6[ys6 > 0].sum(1).min() == 4, "OK")
# ソフトマージン＝ヒンジ損失＋L2（C 大でハードマージンに一致）
Cc = 100.0
# 変数 (w1,w2,b,ξ1..ξ6) のソフトマージン二次計画
cons_s = [{"type": "ineq", "fun": (lambda v, i=i: ys6[i] * (v[:2] @ Xs6[i] + v[2]) - 1 + v[3 + i])} for i in range(6)]
so = optimize.minimize(lambda v: 0.5 * (v[0] ** 2 + v[1] ** 2) + Cc * v[3:].sum(), x0=np.r_[1, 1, -3, np.zeros(6)],
                       constraints=cons_s, bounds=[(None, None)] * 3 + [(0, None)] * 6, method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
hinge = lambda v: 0.5 * (v[0] ** 2 + v[1] ** 2) + Cc * np.maximum(0, 1 - ys6 * (Xs6 @ v[:2] + v[2])).sum()
check(T, "ソフトマージン（C=100、SLSQP）: ξ=0 で w=(2/3,2/3), b=−5/3（許容1e-6）、目的関数＝ヒンジ損失＋L2",
      allclose(so.x[:3], [2 / 3, 2 / 3, -5 / 3], 1e-6) and allclose(so.x[3:], 0, 1e-6) and close(hinge(so.x[:3]), so.fun, 1e-6), so.x[:3])
sm = chart(T, "svm-margin")["params"]
check(T, "charts svm-margin の params がスライダー範囲内", -0.5 <= sm["gap"] <= 2, sm)

# ================================================================ 重要度Cの用語
def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


T = "kernel-method"
xk = np.array([1.0, 2.0])
yk = np.array([3.0, 1.0])
phi = lambda v: np.array([v[0] ** 2, math.sqrt(2) * v[0] * v[1], v[1] ** 2])
check(T, "x'y=5、(x'y)²=25", close(xk @ yk, 5, 1e-12) and close((xk @ yk) ** 2, 25, 1e-12), (xk @ yk) ** 2)
check(T, "φ(x)=(1,2√2,4)、φ(y)=(9,3√2,1)、内積 9+12+4=25", allclose(phi(xk), [1, 2 * math.sqrt(2), 4], 1e-12) and allclose(phi(yk), [9, 3 * math.sqrt(2), 1], 1e-12)
      and close(phi(xk) @ phi(yk), 25, 1e-10), r(phi(xk) @ phi(yk)))
a1s, a2s, b1s, b2s = sp.symbols("a1 a2 b1 b2", real=True)
check(T, "sympy：(a'b)² = φ(a)'φ(b)（恒等式）", sp.expand((a1s * b1s + a2s * b2s) ** 2 - (a1s ** 2 * b1s ** 2 + 2 * a1s * a2s * b1s * b2s + a2s ** 2 * b2s ** 2)) == 0, "恒等的に0")
d2k = ((xk - yk) ** 2).sum()
check(T, "‖x−y‖²=5、σ=1 で e^{−2.5}≈0.0821、σ=2 で e^{−0.625}≈0.5353", close(d2k, 5, 1e-12) and close(math.exp(-d2k / 2), 0.0821) and close(math.exp(-d2k / 8), 0.5353), (r(math.exp(-d2k / 2)), r(math.exp(-d2k / 8))))
check(T, "γ=1/(2σ²) の流儀で同じ値", close(math.exp(-(1 / (2 * 4)) * d2k), math.exp(-d2k / 8), 1e-15), "一致")
okpsd = True
for _ in range(200):
    P = rng.normal(size=(8, 2))
    Dg = ((P[:, None] - P[None]) ** 2).sum(2)
    for Kmat in (np.exp(-Dg / 2), (P @ P.T + 1) ** 3):
        okpsd &= np.linalg.eigvalsh(Kmat).min() > -1e-8
check(T, "ガウス・多項式カーネルのグラム行列は半正定値（乱数200組、許容 −1e-8）", okpsd, "成立")
Pbad = rng.normal(size=(8, 2))
Kbad = -((Pbad[:, None] - Pbad[None]) ** 2).sum(2)
check(T, "よくある間違い：K=−‖x−y‖² は半正定値でない（カーネルにならない）", np.linalg.eigvalsh(Kbad).min() < -1e-6, r(np.linalg.eigvalsh(Kbad).min()))
x1d = np.array([-0.5, 0.5, -2, 2])
g1d = np.array([0, 0, 1, 1])
thr_ok = any(((x1d > t) == g1d).all() or ((x1d < t) == g1d).all() for t in np.linspace(-3, 3, 601))
check(T, "1次元 ±0.5 vs ±2 はしきい値1本で分けられず、x²=1 で分けられる", (not thr_ok) and (((x1d ** 2) > 1) == g1d).all(), "成立")

T = "decision-tree"
gini = lambda c: 1 - sum((x / sum(c)) ** 2 for x in c)
ent = lambda c: -sum((x / sum(c)) * math.log2(x / sum(c)) for x in c if x > 0)
check(T, "親ノード ジニ 0.5、エントロピー 1（2クラス半々で最大）", close(gini([5, 5]), 0.5, 1e-12) and close(ent([5, 5]), 1, 1e-12)
      and all(gini([k, 10 - k]) <= 0.5 + 1e-12 and ent([k, 10 - k]) <= 1 + 1e-12 for k in range(11)), "")
gA = 0.5 * gini([4, 1]) + 0.5 * gini([1, 4])
check(T, "分け方A：子 0.32、加重平均 0.32、減少量 0.18", close(gini([4, 1]), 0.32, 1e-12) and close(gA, 0.32, 1e-12) and close(0.5 - gA, 0.18, 1e-12), r(0.5 - gA))
gB = 0.3 * gini([3, 0]) + 0.7 * gini([2, 5])
check(T, "分け方B：右 20/49≈0.408、加重平均 0.286、減少量 0.214", close(gini([2, 5]), 20 / 49, 1e-12) and close(20 / 49, 0.408, 5e-4)
      and close(gB, 0.286, 5e-4) and close(0.5 - gB, 0.214, 5e-4), (r(gini([2, 5])), r(gB), r(0.5 - gB)))
iA = 1 - ent([4, 1])
iB = 1 - 0.7 * ent([2, 5])
check(T, "情報利得 A: 1−0.722=0.278、B: 1−0.7×0.863≈0.396 → B を選ぶ", close(ent([4, 1]), 0.722, 5e-4) and close(iA, 0.278, 5e-4)
      and close(ent([2, 5]), 0.863, 5e-4) and close(iB, 0.396, 5e-4) and iB > iA and 0.5 - gB > 0.5 - gA, (r(iA), r(iB)))

T = "naive-bayes-classifier"
sp_ = 0.4 * 0.5 * 0.2
hm_ = 0.6 * 0.05 * 0.01
check(T, "迷惑 0.04、通常 0.0003、事後確率 0.04/0.0403≈0.9926", close(sp_, 0.04, 1e-15) and close(hm_, 0.0003, 1e-15) and close(sp_ / (sp_ + hm_), 0.9926), r(sp_ / (sp_ + hm_)))
sp2, hm2 = sp_ * 0.1, hm_ * 0.4
check(T, "「会議」を加えると 0.004/(0.004+0.00012)≈0.9709", close(sp2, 0.004, 1e-15) and close(hm2, 0.00012, 1e-15) and close(sp2 / (sp2 + hm2), 0.9709), r(sp2 / (sp2 + hm2)))
lp = math.log(0.4) + math.log(0.5) + math.log(0.2) + math.log(0.1)
lh = math.log(0.6) + math.log(0.05) + math.log(0.01) + math.log(0.4)
check(T, "対数の和で比べても同じ事後確率（ロジスティック変換）", close(1 / (1 + math.exp(lh - lp)), sp2 / (sp2 + hm2), 1e-12), "一致")
check(T, "ラプラス平滑化 (0+1)/(30+2)=0.03125", close((0 + 1) / (30 + 2), 0.03125, 1e-15), 1 / 32)
# 正規分布のナイーブベイズ = 共分散行列が対角の QDA（乱数で確認）
mu_a, mu_b = np.array([0.0, 1.0]), np.array([1.5, -0.5])
sd_a, sd_b = np.array([1.0, 2.0]), np.array([0.7, 1.2])
pts = rng.normal(size=(500, 2)) * 2
nb = lambda x, m, s: stats.norm.logpdf(x, m, s).sum(1)
qd = lambda x, m, s: stats.multivariate_normal(m, np.diag(s ** 2)).logpdf(x)
check(T, "正規分布のナイーブベイズ = 対角共分散の二次判別（乱数500点、許容1e-10）",
      allclose(nb(pts, mu_a, sd_a) - nb(pts, mu_b, sd_b), qd(pts, mu_a, sd_a) - qd(pts, mu_b, sd_b), 1e-10), "一致")

# ---------------------------------------------------------------- まとめ
ok = sum(results)
ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
