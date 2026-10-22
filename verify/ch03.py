"""第3章 検算スクリプト（期待値・分散・相関など多変数の基本量）。
厳密計算は fractions.Fraction / sympy、数値は numpy / scipy.stats。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from fractions import Fraction as F
import math
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


rng = np.random.default_rng(20261023)
N = 1_000_000

# ============ coefficient-of-variation ============
T = "coefficient-of-variation"
lam = 2.5
m, v = stats.expon(scale=1 / lam).stats("mv")
check(T, "指数分布 平均1/λ, SD1/λ, CV=1", close(m, 1 / lam) and close(np.sqrt(v), 1 / lam) and close(np.sqrt(v) / m, 1), float(np.sqrt(v) / m))
m, v = stats.poisson(lam).stats("mv")
check(T, "ポアソン CV=1/√λ, 分散/平均=1", close(np.sqrt(v) / m, 1 / np.sqrt(lam)) and close(v / m, 1), float(np.sqrt(v) / m))
for k, th in [(4, 1), (4, 2.7), (9, 0.5)]:
    m, v = stats.gamma(k, scale=th).stats("mv")
    check(T, f"ガンマ(k={k},θ={th}) 平均kθ, SD√kθ, CV=1/√k", close(m, k * th) and close(np.sqrt(v), np.sqrt(k) * th) and close(np.sqrt(v) / m, 1 / np.sqrt(k)), float(np.sqrt(v) / m))
c1, c2 = 5.7 / 170, 9 / 60
check(T, "例1 CV身長≈0.0335, CV体重=0.15, 比≈4.5", close(c1, 0.0335) and close(c2, 0.15) and abs(c2 / c1 - 4.5) < 0.05, (round(c1, 5), c2, round(c2 / c1, 3)))
check(T, "例2 k=4 で CV=0.5", close(1 / math.sqrt(4), 0.5), 0.5)
x = rng.gamma(4, 1, N)
cv = lambda a: a.std() / a.mean()
check(T, "尺度不変・位置で変化（シミュ, 許容1e-9 / 差>0.01）", abs(cv(x) - cv(3 * x)) < 1e-9 and abs(cv(x) - cv(x + 5)) > 0.01, (round(cv(x), 4), round(cv(3 * x), 4), round(cv(x + 5), 4)))
mu_s, sig_s, n = sp.symbols("mu sigma n", positive=True)
cvn = (sig_s / sp.sqrt(n)) / mu_s
check(T, "X̄ の CV=σ/(√n μ)、n を4倍で半分", sp.simplify(cvn.subs(n, 4 * n) / cvn) == sp.Rational(1, 2), sp.simplify(cvn.subs(n, 4 * n) / cvn))

# ============ conditional-expectation ============
T = "conditional-expectation"
P = {(0, 0): F(3, 10), (0, 1): F(2, 10), (1, 0): F(1, 10), (1, 1): F(4, 10)}
px = {a: P[(a, 0)] + P[(a, 1)] for a in (0, 1)}
EYx = {a: P[(a, 1)] / px[a] for a in (0, 1)}
EY = P[(0, 1)] + P[(1, 1)]
check(T, "例1 P(X=0)=P(X=1)=0.5", px[0] == px[1] == F(1, 2), (px[0], px[1]))
check(T, "例1 E[Y|X=0]=0.4, E[Y|X=1]=0.8", EYx[0] == F(2, 5) and EYx[1] == F(4, 5), (float(EYx[0]), float(EYx[1])))
check(T, "例1 繰り返し期待値 0.6=E[Y]", px[0] * EYx[0] + px[1] * EYx[1] == EY == F(3, 5), float(EY))
# 例2 シミュレーション（購入額は平均3の指数分布で代用）
M = 200_000
Nn = rng.poisson(20, M)
S = np.array([rng.exponential(3, k).sum() for k in Nn[:M]])
check(T, "例2 E[S]=60（シミュ M=2e5, 許容0.2）", abs(S.mean() - 60) < 0.2, round(S.mean(), 4))
check(T, "例3 E[Y|X=182]=176", 170 + 0.5 * 6 / 6 * (182 - 170) == 176, 176)
# 2変量正規の条件付き期待値（シミュ）
cov = [[36, 18], [18, 36]]
XY = rng.multivariate_normal([170, 170], cov, 2_000_000)
sel = np.abs(XY[:, 0] - 182) < 0.25
check(T, "例3 シミュ E[Y|X≈182]≈176（許容0.1）", abs(XY[sel, 1].mean() - 176) < 0.1, round(XY[sel, 1].mean(), 4))
b = np.polyfit(XY[:, 1], XY[:, 0], 1)[0]
check(T, "逆回帰 E[X|Y] の傾き ρσX/σY=0.5（許容0.005）", abs(b - 0.5) < 0.005, round(b, 4))
Xu = rng.random(N); Yu = rng.random(N) * Xu
xs, ys = sp.symbols("x y", positive=True)
EY4 = sp.integrate(sp.integrate(ys / xs, (ys, 0, xs)), (xs, 0, 1))
check(T, "例4 E[Y]=1/4（sympy厳密 + シミュ許容0.002）", EY4 == sp.Rational(1, 4) and abs(Yu.mean() - 0.25) < 0.002, (EY4, round(Yu.mean(), 4)))
# 最良予測の分解 E[(Y-h)^2]=E[(Y-m)^2]+E[(m-h)^2]
h = 0.3 + 0.2 * Xu
lhs = np.mean((Yu - h) ** 2); rhs = np.mean((Yu - Xu / 2) ** 2) + np.mean((Xu / 2 - h) ** 2)
check(T, "MSE 分解（シミュ, 許容2e-4）", abs(lhs - rhs) < 2e-4, (round(lhs, 5), round(rhs, 5)))
# E[Y|X]=E[Y] でも独立でない例: Y = X*eps, E[eps]=0
eps = rng.standard_normal(N); Xn = rng.random(N) + 0.5; Yn = Xn * eps
lo, hi = Xn < 0.8, Xn > 1.2
check(T, "E[Y|X]=E[Y] だが分散は X に依存（シミュ）", abs(Yn[lo].mean()) < 0.01 and abs(Yn[hi].mean()) < 0.01 and Yn[hi].var() > 2 * Yn[lo].var(), (round(Yn[lo].var(), 3), round(Yn[hi].var(), 3)))

# ============ conditional-variance ============
T = "conditional-variance"
VYx = {a: EYx[a] * (1 - EYx[a]) for a in (0, 1)}
EV = px[0] * VYx[0] + px[1] * VYx[1]
VE = px[0] * (EYx[0] - EY) ** 2 + px[1] * (EYx[1] - EY) ** 2
check(T, "例1 V[Y|X=0]=0.24, V[Y|X=1]=0.16", VYx[0] == F(24, 100) and VYx[1] == F(16, 100), (float(VYx[0]), float(VYx[1])))
check(T, "例1 E[V]=0.20, V[E]=0.04, 和0.24=V[Y]", EV == F(1, 5) and VE == F(1, 25) and EV + VE == EY * (1 - EY) == F(24, 100), (float(EV), float(VE), float(EY * (1 - EY))))
VS = 4 * 20 + 9 * 20
check(T, "例2 V[S]=260, √260≈16.125, =λ(σ²+μ²)", VS == 260 and close(math.sqrt(260), 16.125, 5e-4) and 20 * (4 + 9) == 260, round(math.sqrt(260), 4))
# シミュ: 購入額 gamma(shape=9/4, scale=4/3) → 平均3, SD2
S2 = np.array([rng.gamma(9 / 4, 4 / 3, k).sum() for k in Nn])
check(T, "例2 シミュ E[S]≈60, V[S]≈260（M=2e5, 許容 0.2 / 3）", abs(S2.mean() - 60) < 0.2 and abs(S2.var() - 260) < 3, (round(S2.mean(), 3), round(S2.var(), 2)))
check(T, "例3 36(1-0.25)=27, √27≈5.196", 36 * 0.75 == 27 and close(math.sqrt(27), 5.196, 5e-4), round(math.sqrt(27), 4))
sel2 = np.abs(XY[:, 0] - 160) < 0.25
check(T, "例3 シミュ V[Y|X≈160]≈27（許容0.6）", abs(XY[sel2, 1].var() - 27) < 0.6, round(XY[sel2, 1].var(), 3))
VY4 = sp.integrate(xs**2 / 12, (xs, 0, 1)) + sp.Rational(1, 4) * (sp.Rational(1, 3) - sp.Rational(1, 4))
check(T, "例4 1/36+1/48=7/144≈0.0486（sympy + シミュ許容5e-4）", VY4 == sp.Rational(7, 144) and close(7 / 144, 0.0486) and abs(Yu.var() - 7 / 144) < 5e-4, (VY4, round(Yu.var(), 5)))
# 混合分布の分散
w, m1, s1, m2, s2 = 0.3, 0, 1, 4, 1.5
g = rng.random(N) < w
Z = np.where(g, rng.normal(m1, s1, N), rng.normal(m2, s2, N))
th = w * s1**2 + (1 - w) * s2**2 + w * (1 - w) * (m1 - m2) ** 2
check(T, "混合正規の分散 = 群内 + w(1-w)(μ1-μ2)²（シミュ許容0.02）", abs(Z.var() - th) < 0.02, (round(Z.var(), 4), th))
# 過分散（ガンマ混合ポアソン）
lamg = rng.gamma(2, 2, N); Yp = rng.poisson(lamg)
check(T, "ガンマ混合ポアソンで分散>平均（理論 4+8=12 vs 4, 許容0.15）", abs(Yp.var() - 12) < 0.15 and abs(Yp.mean() - 4) < 0.02, (round(Yp.mean(), 3), round(Yp.var(), 3)))
# 特定 x で V[Y|X=x]>V[Y] の例: 例4 で x=1 → 1/12 > 7/144
check(T, "特定 x で V[Y|X=x]>V[Y]（例4, x=1: 1/12>7/144）", F(1, 12) > F(7, 144), (1 / 12, 7 / 144))

# ============ correlation-matrix ============
T = "correlation-matrix"
Sig = np.array([[4, 1.2, -2], [1.2, 9, 3], [-2, 3, 16]])
Dm = np.diag(1 / np.sqrt(np.diag(Sig)))
R = Dm @ Sig @ Dm
Rclaim = np.array([[1, 0.2, -0.25], [0.2, 1, 0.25], [-0.25, 0.25, 1]])
check(T, "例1 R=D^{-1/2}ΣD^{-1/2}（0.2, -0.25, 0.25）", np.allclose(R, Rclaim), R.round(4).tolist())
check(T, "例1 Σ, R は半正定値", np.linalg.eigvalsh(Sig).min() > 0 and np.linalg.eigvalsh(R).min() > 0, np.linalg.eigvalsh(R).round(4).tolist())
Dh = np.diag(np.sqrt(np.diag(Sig)))
check(T, "Σ = D^{1/2} R D^{1/2} で復元", np.allclose(Dh @ R @ Dh, Sig), True)
ok = True
for p in (3, 5, 8):
    for rho in (-0.1, 0.3, 0.7):
        E = np.sort(np.linalg.eigvalsh((1 - rho) * np.eye(p) + rho * np.ones((p, p))))
        ok &= np.isclose(E[-1], 1 + (p - 1) * rho) and np.allclose(E[:-1], 1 - rho) if rho > 0 else np.isclose(E[0], 1 + (p - 1) * rho) and np.allclose(E[1:], 1 - rho)
check(T, "等相関行列の固有値 1+(p-1)ρ, 1-ρ", ok, ok)
E3 = np.linalg.eigvalsh(1.6 * np.eye(3) - 0.6 * np.ones((3, 3)))
check(T, "例2 全相関-0.6 の最小固有値 -0.2 < 0", np.isclose(E3.min(), -0.2), E3.round(4).tolist())
check(T, "ρ≥-1/(p-1) 境界（p=4, ρ=-1/3 で最小固有値0）", np.isclose(np.linalg.eigvalsh((4 / 3) * np.eye(4) - (1 / 3) * np.ones((4, 4))).min(), 0), 0)
Xs = rng.multivariate_normal([0, 0, 0], Sig, N)
Rs = np.corrcoef(Xs.T)
check(T, "標準化変数の共分散 = R（シミュ許容0.005）", np.abs(Rs - Rclaim).max() < 0.005, Rs.round(4).tolist())

# ============ covariance-and-correlation ============
T = "covariance-and-correlation"
EX = px[1]; EXY = P[(1, 1)]
Cv = EXY - EX * EY
rho = float(Cv) / math.sqrt(float(EX * (1 - EX) * EY * (1 - EY)))
check(T, "例1 E[X]=0.5,E[Y]=0.6,E[XY]=0.4, Cov=0.1", (EX, EY, EXY, Cv) == (F(1, 2), F(3, 5), F(2, 5), F(1, 10)), float(Cv))
check(T, "例1 V[X]=0.25,V[Y]=0.24, ρ≈0.4082", EX * (1 - EX) == F(1, 4) and EY * (1 - EY) == F(6, 25) and close(rho, 0.4082), round(rho, 5))
xv = [F(-1), F(0), F(1)]
EX2 = sum(xv) / 3; EX3 = sum(t**3 for t in xv) / 3
check(T, "例2 Y=X²: E[X]=0, E[XY]=E[X³]=0 → Cov=0", EX2 == 0 and EX3 == 0, 0)
check(T, "例2 独立でない: P(X=0,Y=0)=1/3 ≠ P(X=0)P(Y=0)=1/9", F(1, 3) != F(1, 3) * F(1, 3), (F(1, 3), F(1, 9)))
xd = np.array([1, 2, 3, 4, 5.]); yd = np.array([50, 55, 65, 70, 85.])
sxy = ((xd - xd.mean()) * (yd - yd.mean())).sum(); sxx = ((xd - 3) ** 2).sum(); syy = ((yd - 65) ** 2).sum()
r = sxy / math.sqrt(sxx * syy)
check(T, "例3 平均3,65, Sxy=85, Sxx=10, Syy=750", (xd.mean(), yd.mean(), sxy, sxx, syy) == (3, 65, 85, 10, 750), (sxy, sxx, syy))
check(T, "例3 r≈0.9815（np.corrcoef と一致）", close(r, 0.9815) and np.isclose(r, np.corrcoef(xd, yd)[0, 1]), round(r, 5))
check(T, "線形変換: ρ(aX+b,cY+d)=sign(ac)ρ", np.isclose(np.corrcoef(-2 * xd + 1, 3 * yd - 4)[0, 1], -r) and np.isclose(np.cov(-2 * xd + 1, 3 * yd - 4)[0, 1], -6 * np.cov(xd, yd)[0, 1]), True)
Xr = rng.standard_normal(N); Yr = 0.5 * Xr + rng.standard_normal(N)
check(T, "|ρ|≤1 / Y=aX+b で ρ=±1", abs(np.corrcoef(Xr, Yr)[0, 1]) <= 1 and np.isclose(np.corrcoef(Xr, -3 * Xr + 2)[0, 1], -1), True)
check(T, "r=0.8 vs 0.4 → r²=0.64, 0.16", close(0.8**2, 0.64) and close(0.4**2, 0.16), (0.64, 0.16))

# ============ expectation-variance-properties ============
T = "expectation-variance-properties"
d = [F(i) for i in range(1, 7)]
E1 = sum(d) / 6; E2 = sum(t * t for t in d) / 6; V1 = E2 - E1**2
check(T, "例1 E=3.5, E[X²]=91/6, V=35/12≈2.9167", E1 == F(7, 2) and E2 == F(91, 6) and V1 == F(35, 12) and close(V1, 2.9167), float(V1))
Yd = [2 * t + 3 for t in d]; EYd = sum(Yd) / 6; VYd = sum(t * t for t in Yd) / 6 - EYd**2
check(T, "例1 Y=2X+3: E=10, V=35/3≈11.667", EYd == 10 and VYd == F(35, 3) and close(VYd, 11.667, 5e-4), float(VYd))
sums = [a + b for a in d for b in d]; diffs = [a - b for a in d for b in d]
Vs = sum(t * t for t in sums) / 36 - (sum(sums) / 36) ** 2; Vd = sum(t * t for t in diffs) / 36 - (sum(diffs) / 36) ** 2
check(T, "例2 V[和]=V[差]=35/6≈5.8333, E 7 と 0", Vs == Vd == F(35, 6) and sum(sums) / 36 == 7 and sum(diffs) / 36 == 0 and close(Vs, 5.8333), float(Vs))
vp = 0.25 * 4 + 0.25 * 9 + 2 * 0.25 * (-1.2)
check(T, "例3 V[0.5X+0.5Y]=2.65", close(vp, 2.65), vp)
a_ = np.array([1., -2, 0.5]); Sg = np.array([[2, 0.3, -0.4], [0.3, 1, 0.2], [-0.4, 0.2, 1.5]])
Xg = rng.multivariate_normal([0, 0, 0], Sg, N)
form = sum(a_[i]**2 * Sg[i, i] for i in range(3)) + 2 * sum(a_[i] * a_[j] * Sg[i, j] for i in range(3) for j in range(i + 1, 3))
check(T, "一般の V[Σa_iX_i] 公式（シミュ許容0.03）", abs((Xg @ a_).var() - form) < 0.03, (round((Xg @ a_).var(), 4), form))
Xb = rng.exponential(2, (200_000, 10)).mean(axis=1)
check(T, "V[X̄]=σ²/n（指数 σ²=4, n=10 → 0.4, 許容0.005）", abs(Xb.var() - 0.4) < 0.005 and abs(Xb.mean() - 2) < 0.005, round(Xb.var(), 4))
check(T, "V[2X]=4V[X] vs V[X1+X2]=2V[X]（サイコロ）", 4 * V1 == F(35, 3) and Vs == 2 * V1, (float(4 * V1), float(Vs)))

# ============ mean-vector ============
T = "mean-vector"
A = np.array([[1, 1], [1, -1]])
check(T, "例1 Aμ=(130,-10)", (A @ np.array([60, 70])).tolist() == [130, -10], (A @ np.array([60, 70])).tolist())
Mn = rng.multinomial(10, [0.5, 0.3, 0.2], N)
check(T, "例2 多項分布 E=(5,3,2)（シミュ許容0.01）, 和=10", np.abs(Mn.mean(axis=0) - [5, 3, 2]).max() < 0.01 and (Mn.sum(axis=1) == 10).all(), Mn.mean(axis=0).round(4).tolist())
mu = np.array([1., 2]); S22 = np.array([[2, 0.6], [0.6, 1]]); B = np.array([[1, 0.5], [0.5, 3]])
Xq = rng.multivariate_normal(mu, S22, N)
qf = np.einsum("ij,jk,ik->i", Xq, B, Xq).mean(); th = np.trace(B @ S22) + mu @ B @ mu
check(T, "E[XᵀBX]=tr(BΣ)+μᵀBμ（シミュ許容0.05）", abs(qf - th) < 0.05, (round(qf, 4), th))
check(T, "E[X1X2]=μ1μ2+σ12（シミュ許容0.01）", abs((Xq[:, 0] * Xq[:, 1]).mean() - (2 + 0.6)) < 0.01, round((Xq[:, 0] * Xq[:, 1]).mean(), 4))
Xm = Xq @ np.array([[1, 2], [0, -1]]).T + np.array([3, 4])
check(T, "E[AX+b]=Aμ+b（相関ありでも, シミュ許容0.01）", np.abs(Xm.mean(axis=0) - (np.array([[1, 2], [0, -1]]) @ mu + [3, 4])).max() < 0.01, Xm.mean(axis=0).round(4).tolist())

# ============ partial-correlation-coefficient ============
T = "partial-correlation-coefficient"
pc = lambda rxy, rxz, ryz: (rxy - rxz * ryz) / math.sqrt((1 - rxz**2) * (1 - ryz**2))
check(T, "例1 分子0.04, (1-.49)(1-.64)=0.1836, r≈0.0934", close(0.6 - 0.56, 0.04) and close(0.51 * 0.36, 0.1836) and close(pc(0.6, 0.7, 0.8), 0.0934), round(pc(0.6, 0.7, 0.8), 5))
check(T, "例2 0.26/√0.3276≈0.4543", close(0.36 * 0.91, 0.3276) and close(pc(0.5, 0.8, 0.3), 0.4543), round(pc(0.5, 0.8, 0.3), 5))
R1 = np.array([[1, 0.6, 0.7], [0.6, 1, 0.8], [0.7, 0.8, 1]])
Ri = np.linalg.inv(R1)
check(T, "例3 r^11≈1.9780, r^22≈2.8022, r^12≈-0.2198", close(Ri[0, 0], 1.9780) and close(Ri[1, 1], 2.8022) and close(Ri[0, 1], -0.2198), (round(Ri[0, 0], 5), round(Ri[1, 1], 5), round(Ri[0, 1], 5)))
check(T, "例3 -r^12/√(r^11 r^22)≈0.0934", close(-Ri[0, 1] / math.sqrt(Ri[0, 0] * Ri[1, 1]), 0.0934), round(-Ri[0, 1] / math.sqrt(Ri[0, 0] * Ri[1, 1]), 5))
Xp = rng.multivariate_normal([0, 0, 0], R1, N)
ex = Xp[:, 0] - np.polyval(np.polyfit(Xp[:, 2], Xp[:, 0], 1), Xp[:, 2])
ey = Xp[:, 1] - np.polyval(np.polyfit(Xp[:, 2], Xp[:, 1], 1), Xp[:, 2])
check(T, "残差どうしの相関 = 公式（シミュ許容0.005）", abs(np.corrcoef(ex, ey)[0, 1] - pc(0.6, 0.7, 0.8)) < 0.005, round(np.corrcoef(ex, ey)[0, 1], 4))
check(T, "補足 r_XZ=2/√5≈0.8944, r_YZ=1.5/√3.25≈0.8321, r_XY≈0.7442, 積=r_XY",
      close(2 / math.sqrt(5), 0.8944) and close(1.5 / math.sqrt(3.25), 0.8321) and close(3 / math.sqrt(16.25), 0.7442)
      and math.isclose(2 / math.sqrt(5) * 1.5 / math.sqrt(3.25), 3 / math.sqrt(16.25)), round(3 / math.sqrt(16.25), 5))
# 偏相関の t 検定 = 回帰係数の t 統計量（df=n-2-k, k=1）
nn = 40
W = rng.multivariate_normal([0, 0, 0], R1, nn)
rx = np.corrcoef(W.T)
rp = pc(rx[0, 1], rx[0, 2], rx[1, 2])
tp = rp * math.sqrt(nn - 3) / math.sqrt(1 - rp**2)
Xd = np.column_stack([np.ones(nn), W[:, 0], W[:, 2]])
beta, res, *_ = np.linalg.lstsq(Xd, W[:, 1], rcond=None)
s2 = res[0] / (nn - 3); se = math.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
check(T, "偏相関の t=r√(n-2-k)/√(1-r²) は回帰係数の t と一致", math.isclose(tp, beta[1] / se, rel_tol=1e-9), (round(tp, 5), round(beta[1] / se, 5)))

# ============ quantile-function ============
T = "quantile-function"
pp = sp.symbols("p", positive=True); lm = sp.symbols("lambda", positive=True)
Qe = -sp.log(1 - pp) / lm
check(T, "指数 Q(p)=-log(1-p)/λ で F(Q(p))=p", sp.simplify(1 - sp.exp(-lm * Qe) - pp) == 0, Qe)
q5, q9 = stats.expon(scale=2).ppf([0.5, 0.9])
check(T, "例1 Q(0.5)=2log2≈1.3863, Q(0.9)=2log10≈4.6052", close(q5, 1.3863) and close(q9, 4.6052) and close(2 * math.log(2), 1.3863) and close(2 * math.log(10), 4.6052), (round(q5, 5), round(q9, 5)))
check(T, "例1 中央値 < 平均2", q5 < 2, q5)
die = stats.randint(1, 7)
check(T, "例2 サイコロ F(2)=1/3, F(3)=0.5, Q(0.5)=3", close(die.cdf(2), 1 / 3) and close(die.cdf(3), 0.5) and die.ppf(0.5) == 3, die.ppf(0.5))
check(T, "例3 Φ⁻¹(0.975)≈1.9600, Φ⁻¹(0.95)≈1.6449", close(stats.norm.ppf(0.975), 1.96) and close(stats.norm.ppf(0.95), 1.6449), (round(stats.norm.ppf(0.975), 5), round(stats.norm.ppf(0.95), 5)))
check(T, "対称分布 Q(p)=-Q(1-p)（t, df=7）", math.isclose(stats.t(7).ppf(0.1), -stats.t(7).ppf(0.9)), round(stats.t(7).ppf(0.1), 5))
U = rng.random(N); Xi = -np.log(1 - U) / 0.5
ks = stats.kstest(Xi[:100000], stats.expon(scale=2).cdf)
check(T, "逆関数法 Q(U)~F（KS検定 p>0.01, 平均許容0.01）", ks.pvalue > 0.01 and abs(Xi.mean() - 2) < 0.01, (round(ks.pvalue, 4), round(Xi.mean(), 4)))
check(T, "対数正規の中央値 e^μ（μ=0.7, σ=1.3）", math.isclose(stats.lognorm(1.3, scale=math.exp(0.7)).median(), math.exp(0.7)), round(math.exp(0.7), 5))
Xn2 = stats.norm(1, 2)
check(T, "減少関数 g(x)=-x: Q_Y(p)=g(Q_X(1-p))", math.isclose(stats.norm(-1, 2).ppf(0.2), -Xn2.ppf(0.8)), round(stats.norm(-1, 2).ppf(0.2), 5))

# ============ skewness-kurtosis ============
T = "skewness-kurtosis"
def sk(dist):
    s, k = dist.stats("sk"); return float(s), float(k) + 3
tab = [("正規", stats.norm(), 0, 3), ("一様", stats.uniform(), 0, 1.8), ("指数", stats.expon(), 2, 9),
       ("Po(3)", stats.poisson(3), 1 / math.sqrt(3), 3 + 1 / 3), ("ガンマk=2.5", stats.gamma(2.5), 2 / math.sqrt(2.5), 3 + 6 / 2.5),
       ("t(ν=7)", stats.t(7), 0, 3 + 6 / 3)]
for name, dist, s0, k0 in tab:
    s, k = sk(dist)
    check(T, f"表 {name}: 歪度 {s0:.4f}, 尖度 {k0:.4f}", close(s, s0) and close(k, k0), (round(s, 4), round(k, 4)))
xv = np.array([0, 5.]); pv = np.array([0.8, 0.2])
mu_ = (xv * pv).sum(); var_ = ((xv - mu_) ** 2 * pv).sum(); m3 = ((xv - mu_) ** 3 * pv).sum(); m4 = ((xv - mu_) ** 4 * pv).sum()
check(T, "例1 μ=1, σ²=4, m3=12, β1=1.5, m4=52, β2=3.25", (mu_, var_, m3, m3 / 8, m4, m4 / 16) == (1, 4, 12, 1.5, 52, 3.25), (mu_, var_, m3, m4))
s, k = sk(stats.poisson(4))
check(T, "例2 Po(4) 歪度0.5, 超過尖度0.25", close(s, 0.5) and close(k - 3, 0.25), (s, k - 3))
X_, m_ = sp.symbols("X mu")
check(T, "E[(X-μ)³]=E[X³]-3μE[X²]+2μ³（μ=E[X] を代入した展開）",
      sp.expand((X_ - m_) ** 3 - (X_**3 - 3 * m_ * X_**2 + 3 * m_**2 * X_ - m_**3)) == 0, "E[3μ²X]=3μ³ → -μ³+3μ³=2μ³")
Xg2 = rng.gamma(2, 1, N)
check(T, "aX+b (a>0) で歪度・尖度不変（シミュ厳密一致 1e-8）", math.isclose(stats.skew(Xg2), stats.skew(3 * Xg2 + 7), abs_tol=1e-8) and math.isclose(stats.kurtosis(Xg2), stats.kurtosis(3 * Xg2 + 7), abs_tol=1e-8), round(stats.skew(Xg2), 4))
check(T, "グラフ: ガンマk=2 と同平均・分散の正規 N(2, 1.414²)", close(math.sqrt(stats.gamma(2).var()), 1.414, 5e-4), round(math.sqrt(2), 4))
# キュムラント: ポアソンは全キュムラント λ
check(T, "キュムラント表現 κ3/κ2^1.5, κ4/κ2²（Po(4): 4/8, 4/16）", close(4 / 4**1.5, s) and close(4 / 16, k - 3), True)

# ============ spurious-correlation ============
T = "spurious-correlation"
a, b = 2, 1.5
Zs = rng.standard_normal(N); X2 = a * Zs + rng.standard_normal(N); Y2 = b * Zs + rng.standard_normal(N)
check(T, "Cov=abV[Z]=3（シミュ許容0.02）", abs(np.cov(X2, Y2)[0, 1] - 3) < 0.02, round(np.cov(X2, Y2)[0, 1], 4))
rho2 = a * b / math.sqrt((a * a + 1) * (b * b + 1))
check(T, "例2 ρ=3/√16.25≈0.7442（シミュ許容0.003）", close(rho2, 0.7442) and close((4 + 1) * (2.25 + 1), 16.25) and abs(np.corrcoef(X2, Y2)[0, 1] - rho2) < 0.003, (round(rho2, 5), round(np.corrcoef(X2, Y2)[0, 1], 4)))
Rm = np.corrcoef([X2, Y2, Zs])
pcs = pc(Rm[0, 1], Rm[0, 2], Rm[1, 2])
check(T, "Z を除いた偏相関≈0（シミュ許容0.003）", abs(pcs) < 0.003, round(pcs, 5))
Zs2 = rng.standard_normal(N); X3 = 1.5 * Zs2 + rng.standard_normal(N); Y3 = -1.5 * Zs2 + rng.standard_normal(N)
check(T, "a,b 異符号で負の相関、a=0 で無相関（シミュ）", np.corrcoef(X3, Y3)[0, 1] < -0.6 and abs(np.corrcoef(rng.standard_normal(N), Y3)[0, 1]) < 0.005, round(np.corrcoef(X3, Y3)[0, 1], 4))

# ============ variance-covariance-matrix ============
T = "variance-covariance-matrix"
S2m = np.array([[4, 1.2], [1.2, 9]])
ASA = A @ S2m @ A.T
check(T, "例1 AΣAᵀ=[[15.4,-5],[-5,10.6]]", np.allclose(ASA, [[15.4, -5], [-5, 10.6]]), ASA.tolist())
ev = np.linalg.eigvalsh(S2m)
check(T, "例1 det=34.56, 固有値≈3.7269, 9.2731, 和13, 積34.56", close(np.linalg.det(S2m), 34.56) and close(ev[0], 3.7269) and close(ev[1], 9.2731) and close(ev.sum(), 13) and close(ev.prod(), 34.56), ev.round(5).tolist())
Xv = rng.multivariate_normal([0, 0], S2m, N)
check(T, "例1 シミュで V[和], V[差], Cov（許容0.05）", np.abs(np.cov((Xv @ A.T).T) - ASA).max() < 0.05, np.cov((Xv @ A.T).T).round(3).tolist())
ws = sp.symbols("w")
vw = 4 * ws**2 + 9 * (1 - ws) ** 2 + sp.Rational(12, 5) * ws * (1 - ws)
wst = sp.solve(sp.diff(vw, ws), ws)[0]
check(T, "例2 w*=7.8/10.6≈0.7358, 最小分散≈3.2604", wst == sp.Rational(78, 106) and close(float(wst), 0.7358) and close(float(vw.subs(ws, wst)), 3.2604) and close(34.56 / 10.6, 3.2604), (float(wst), float(vw.subs(ws, wst))))
pr = np.array([0.5, 0.3, 0.2])
Sm = 10 * (np.diag(pr) - np.outer(pr, pr))
check(T, "例3 多項 Σ=[[2.5,-1.5,-1],[-1.5,2.1,-0.6],[-1,-0.6,1.6]]", np.allclose(Sm, [[2.5, -1.5, -1], [-1.5, 2.1, -0.6], [-1, -0.6, 1.6]]), Sm.round(4).tolist())
check(T, "例3 行和0, det=0, シミュ共分散と一致（許容0.02）", np.allclose(Sm.sum(axis=1), 0) and abs(np.linalg.det(Sm)) < 1e-12 and np.abs(np.cov(Mn.T) - Sm).max() < 0.02, round(np.linalg.det(Sm), 12))
Yv = Xv @ np.array([[0.3, 1], [1, -0.2]]).T + rng.standard_normal((N, 2))
Cov_xy = S2m @ np.array([[0.3, 1], [1, -0.2]]).T
Ac = np.array([[1, 1]]); Bc = np.array([[2, -1]])
emp = np.cov(np.column_stack([Xv @ Ac.T, Yv @ Bc.T]).T)[0, 1]
# 理論値 20.36、標準誤差 ≈ √(V[AX]V[BY]+Cov²)/√N ≈ 0.03 → 許容 0.15（約4.5SE）
check(T, "Cov[AX,BY]=A Cov[X,Y] Bᵀ（シミュ許容0.15）", abs(emp - (Ac @ Cov_xy @ Bc.T)[0, 0]) < 0.15, (round(emp, 4), (Ac @ Cov_xy @ Bc.T)[0, 0]))
Cs = np.cov(np.column_stack([Xv, Yv]).T)[:2, 2:]
check(T, "標本でも Cov[AX,BY]=A Cov[X,Y] Bᵀ が厳密に成立（1e-8）", math.isclose(emp, (Ac @ Cs @ Bc.T)[0, 0], abs_tol=1e-8), round((Ac @ Cs @ Bc.T)[0, 0], 4))

# ============ weighted-geometric-harmonic-mean ============
T = "weighted-geometric-harmonic-mean"
gm = 1.32 ** (1 / 3)
check(T, "例1 1.1·1.5·0.8=1.32, GM≈1.0970（9.70%）", close(1.1 * 1.5 * 0.8, 1.32) and close(gm, 1.0970) and close(stats.gmean([1.1, 1.5, 0.8]), gm, 1e-12), round(gm, 5))
check(T, "例1 相加平均13.33%, 1.1333³≈1.456", close(40 / 3, 13.33, 5e-3) and close((1 + 0.4 / 3) ** 3, 1.456, 5e-4), round((1 + 0.4 / 3) ** 3, 5))
check(T, "例2 2/(1/60+1/40)=48", F(2) / (F(1, 60) + F(1, 40)) == 48 and close(stats.hmean([60, 40]), 48), 48)
check(T, "例3 加重平均66, (2,8): AM5 GM4 HM3.2, GM²=AM·HM", F(30 * 60 + 20 * 75, 50) == 66 and close(stats.gmean([2, 8]), 4) and close(stats.hmean([2, 8]), 3.2) and close(16, 5 * 3.2), 66)
check(T, "+50%,-50% → 0.75倍", close(1.5 * 0.5, 0.75), 0.75)
ok = True
for _ in range(2000):
    xx = rng.uniform(0.01, 10, rng.integers(2, 8)); ww = rng.dirichlet(np.ones(len(xx)))
    am = (ww * xx).sum(); gmw = np.exp((ww * np.log(xx)).sum()); hm = 1 / (ww / xx).sum()
    ok &= am >= gmw - 1e-12 and gmw >= hm - 1e-12
check(T, "加重 AM≥GM≥HM（乱数2000組）", ok, ok)
check(T, "等しい値で AM=GM=HM", np.isclose(stats.gmean([3, 3, 3]), 3) and np.isclose(stats.hmean([3, 3, 3]), 3), 3)
aa, bb = sp.symbols("a b", positive=True)
check(T, "(a+b)/2-√(ab)=(√a-√b)²/2 / 2数で GM²=AM·HM", sp.simplify((aa + bb) / 2 - sp.sqrt(aa * bb) - (sp.sqrt(aa) - sp.sqrt(bb)) ** 2 / 2) == 0
      and sp.simplify(aa * bb - (aa + bb) / 2 * 2 / (1 / aa + 1 / bb)) == 0, True)

ok_n = sum(results); ng_n = len(results) - ok_n
print(f"\n合計: OK {ok_n} / NG {ng_n}")
