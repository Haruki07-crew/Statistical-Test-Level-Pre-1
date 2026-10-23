"""第3章 分布の特性値：問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

from itertools import product
import numpy as np
import sympy as sp
from scipy import stats

q = Q(3)
x, y, w = sp.symbols("x y w", real=True)

# --- 期待値と分散の性質 ---
q.num("q03-expectation-variance-properties-1", (-2) ** 2 * 4)
a = np.array([2, -3]); S = np.array([[4, -2], [-2, 9]])
q.num("q03-expectation-variance-properties-2", a @ S @ a)
# 3: 小さな離散分布・n=3 で E[S^2] を厳密に列挙
vals = np.array([0.0, 1.0, 5.0]); pr = np.array([0.5, 0.3, 0.2])
mu = vals @ pr; sig2 = (vals - mu) ** 2 @ pr
n = 3
ES2 = 0.0
for idx in product(range(3), repeat=n):
    xs = vals[list(idx)]; p = np.prod(pr[list(idx)])
    ES2 += p * np.mean((xs - xs.mean()) ** 2)
cands = [(n - 1) / n * sig2, sig2, sig2 / n, n / (n - 1) * sig2]
q.check("q03-expectation-variance-properties-3", abs(ES2 - cands[0]) < 1e-12 and all(abs(ES2 - c) > 1e-6 for c in cands[1:]),
        f"n=3 の厳密列挙で E[S^2]={ES2:.6f}=(n-1)/n σ²、誤答とは不一致")

# --- 分位点関数 ---
q.num("q03-quantile-function-1", stats.expon(scale=2).ppf(0.5))
F = stats.binom(3, 0.5).cdf
Qv = min(k for k in range(4) if F(k) >= 0.5 - 1e-15)
q.check("q03-quantile-function-2", Qv == 1 and stats.binom(3, 0.5).ppf(0.5) == 1, f"inf 定義で Q(0.5)={Qv}（scipy の ppf も1）、誤答 1.5・2・0.5 とは不一致")

# --- 変動係数 ---
U = stats.uniform(0, 7.3)
q.num("q03-coefficient-of-variation-1", U.std() / U.mean())
X = stats.gamma(a=2, scale=3)
cv = X.std() / X.mean(); cv10 = X.std() / (X.mean() + 10); cvs = (100 * X.std()) / (100 * X.mean())
q.check("q03-coefficient-of-variation-2", abs(cv - cv10) > 0.1 and abs(cv - cvs) < 1e-12,
        f"例：CV={cv:.3f}、X+10 では {cv10:.3f}（変わる。answer=false）、定数倍では不変")

# --- 共分散と相関係数 ---
xs = np.array([-1, 0, 1]); ps = np.ones(3) / 3; ys = xs**2
cov = (xs * ys) @ ps - (xs @ ps) * (ys @ ps)
pj = ps[1]; pm = ps[1] * ps[1]  # P(X=0,Y=0)=P(X=0), P(Y=0)=P(X=0)
q.check("q03-covariance-and-correlation-1", abs(cov) < 1e-12 and abs(pj - pm) > 0.1,
        "Y=X^2 で Cov=0 だが P(X=0,Y=0)=1/3 ≠ P(X=0)P(Y=0)=1/9（answer=false）")
C = np.array([[1, 0.6], [0.6, 1]]); A = np.diag([2, -3])
CU = A @ C @ A.T; rUV = CU[0, 1] / np.sqrt(CU[0, 0] * CU[1, 1])
q.check("q03-covariance-and-correlation-2", abs(rUV + 0.6) < 1e-12 and all(abs(rUV - c) > 1e-6 for c in [0.6, -3.6, -0.36]),
        f"ρ_UV={rUV}、誤答とは不一致")
S = np.array([[4, 1], [1, 1]]); B = np.array([[1, 1], [1, -1]])
CB = B @ S @ B.T
q.num("q03-covariance-and-correlation-3", CB[0, 1] / np.sqrt(CB[0, 0] * CB[1, 1]))

# --- 擬似相関 ---
L = np.array([[2, 1, 0], [1, 0, 1]])  # (Z, e1, e2) の係数
CS = L @ L.T
q.num("q03-spurious-correlation-1", CS[0, 1] / np.sqrt(CS[0, 0] * CS[1, 1]))
rng = np.random.default_rng(0)
Z = rng.normal(size=200000); X1 = Z + rng.normal(size=200000); Y1 = Z + rng.normal(size=200000)
r = np.corrcoef(X1, Y1)[0, 1]
# Z を除いた偏相関（残差どうし）
eX = X1 - np.polyval(np.polyfit(Z, X1, 1), Z); eY = Y1 - np.polyval(np.polyfit(Z, Y1, 1), Z)
rp = np.corrcoef(eX, eY)[0, 1]
q.check("q03-spurious-correlation-2", r > 0.4 and abs(rp) < 0.01, f"直接の関係なしでも相関 {r:.3f}、Z を除くと {rp:.4f}（answer=false）")

# --- 偏相関係数 ---
def pc(rxy, rxz, ryz):
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
def pc_inv(rxy, rxz, ryz):
    R = np.array([[1, rxy, rxz], [rxy, 1, ryz], [rxz, ryz, 1]])
    Ri = np.linalg.inv(R)
    return -Ri[0, 1] / np.sqrt(Ri[0, 0] * Ri[1, 1]), np.linalg.eigvalsh(R).min()
v1 = pc(0.6, 0.7, 0.8); v1b, e1 = pc_inv(0.6, 0.7, 0.8)
q.num("q03-partial-correlation-coefficient-1", v1)
q.check("q03-partial-correlation-coefficient-1", abs(v1 - v1b) < 1e-12 and e1 > 0, "逆行列の式と一致・相関行列は正定値")
v2 = pc(0, 0.5, -0.5); v2b, e2 = pc_inv(0, 0.5, -0.5)
q.check("q03-partial-correlation-coefficient-2", abs(v2 - 1 / 3) < 1e-12 and abs(v2b - v2) < 1e-12 and e2 > 0 and all(abs(v2 - c) > 1e-6 for c in [-1 / 3, 0, 0.25]),
        f"r_XY·Z={v2:.6f}=1/3、誤答とは不一致")
v3 = pc(0.3, 0.8, 0.8); _, e3 = pc_inv(0.3, 0.8, 0.8)
q.check("q03-partial-correlation-coefficient-3", v3 < 0 and abs(v3 - (-0.34 / 0.36)) < 1e-12 and e3 > 0,
        f"反例 r=0.3,0.8,0.8 で偏相関 {v3:.3f}<0（正定値な相関行列。answer=false）")

# --- 条件付き期待値 ---
P = np.array([[0.10, 0.20, 0.10], [0.30, 0.10, 0.20]]); yv = np.array([1, 2, 3])
q.check("q03-conditional-expectation-1", abs(P.sum() - 1) < 1e-12, "表の合計が1")
q.num("q03-conditional-expectation-1", (P[1] / P[1].sum()) @ yv)
EY = sum(sp.Rational(1, 6) * N * sp.Rational(1, 2) for N in range(1, 7))
# 直接列挙
EYd = sum(sp.Rational(1, 6) * sum(k * sp.binomial(N, k) * sp.Rational(1, 2**N) for k in range(N + 1)) for N in range(1, 7))
q.num("q03-conditional-expectation-2", float(EYd))
q.check("q03-conditional-expectation-2", EY == EYd, "繰り返し期待値と直接列挙が一致")
f = x + y
fX = sp.integrate(f, (y, 0, 1))
m = sp.simplify(sp.integrate(y * f, (y, 0, 1)) / fX)
target = (3 * x + 2) / (6 * x + 3)
wrongs = [(3 * x + 2) / 6, sp.Rational(1, 2), (x + 1) / (2 * x + 1)]
q.check("q03-conditional-expectation-3", sp.simplify(m - target) == 0 and all(sp.simplify(m - wv) != 0 for wv in wrongs),
        f"E[Y|X=x]={sp.factor(m)}、誤答とは不一致")

# --- 条件付き分散 ---
VY = sum(sp.Rational(1, 6) * sum(k**2 * sp.binomial(N, k) * sp.Rational(1, 2**N) for k in range(N + 1)) for N in range(1, 7)) - EYd**2
VYf = sp.Rational(7, 2) / 4 + sp.Rational(35, 12) / 4
q.num("q03-conditional-variance-2", float(VY))
q.check("q03-conditional-variance-2", VY == VYf, f"直接列挙 {VY} と全分散の公式が一致")
q.check("q03-conditional-variance-1", VY >= sp.Rational(7, 2) / 4, "数値例：V[Y]=1.604 ≥ E[V[Y|N]]=0.875")
p = sp.symbols("p")
pmf = [sp.integrate(sp.binomial(10, k) * p**k * (1 - p) ** (10 - k), (p, 0, 1)) for k in range(11)]
EY3 = sum(k * pmf[k] for k in range(11)); VY3 = sum(k**2 * pmf[k] for k in range(11)) - EY3**2
q.num("q03-conditional-variance-3", float(VY3))
q.check("q03-conditional-variance-3", all(v == sp.Rational(1, 11) for v in pmf), "Y は 0..10 の一様分布（解説の確認）")

# --- 歪度と尖度 ---
b2 = {"unif": stats.uniform.stats(moments="k") + 3, "expon": stats.expon.stats(moments="k") + 3,
      "t10": stats.t(10).stats(moments="k") + 3, "po1": stats.poisson(1).stats(moments="k") + 3}
q.check("q03-skewness-kurtosis-1", b2["unif"] < 3 and all(b2[k] > 3 for k in ["expon", "t10", "po1"]),
        "β2: " + ", ".join(f"{k}={float(v):.2f}" for k, v in b2.items()) + " → 一様分布のみ3未満")
q.num("q03-skewness-kurtosis-2", float(stats.bernoulli(0.2).stats(moments="s")))

# --- 加重・幾何・調和平均 ---
q.num("q03-weighted-geometric-harmonic-mean-1", (stats.gmean([1.2, 0.9, 1.05]) - 1) * 100)
q.num("q03-weighted-geometric-harmonic-mean-2", stats.hmean([30, 60, 90]))

# --- 期待値ベクトル ---
q.num("q03-mean-vector-1", np.array([1, 2, -1]) @ np.array([2, -1, 3]) + 4)

# --- 分散共分散行列 ---
Sg = np.array([[4, 1], [1, 9]])
q.num("q03-variance-covariance-matrix-1", np.array([1, -2]) @ Sg @ np.array([1, -2]))
mats = [np.array([[1, 2], [2, 1]]), np.array([[1, 1], [1, 1]]), np.array([[4, -2], [-2, 9]]), np.array([[2, 0], [0, 5]])]
mins = [np.linalg.eigvalsh(M).min() for M in mats]
q.check("q03-variance-covariance-matrix-2", mins[0] < 0 and all(e >= -1e-12 for e in mins[1:]),
        f"最小固有値 {[round(float(e), 4) for e in mins]} → 半正定値でないのは選択肢0のみ")
V = 4 * w**2 + 9 * (1 - w) ** 2 + 2 * w * (1 - w)
ws = sp.solve(sp.diff(V, w), w)[0]
vmin = V.subs(w, ws)
one = np.ones(2); Si = np.linalg.inv(Sg)
q.num("q03-variance-covariance-matrix-3", float(vmin))
q.check("q03-variance-covariance-matrix-3", ws == sp.Rational(8, 11) and abs(1 / (one @ Si @ one) - float(vmin)) < 1e-12,
        "w=8/11、最小分散 1/(1'Σ^{-1}1) の公式とも一致")

# --- 相関係数行列 ---
Sc = np.array([[4, -3, 1], [-3, 9, 0.6], [1, 0.6, 1]])
d = np.sqrt(np.diag(Sc)); R = Sc / np.outer(d, d)
q.check("q03-correlation-matrix-1", np.linalg.eigvalsh(Sc).min() > 0 and abs(R[0, 1] + 0.5) < 1e-12 and abs(R[0, 2] - 0.5) < 1e-12,
        "Σ は正定値、ρ12=-0.5、ρ13=0.5（解説の確認）")
q.num("q03-correlation-matrix-1", R[1, 2])
Rbad = np.full((3, 3), -0.6) + np.eye(3) * 1.6
q.check("q03-correlation-matrix-2", np.linalg.eigvalsh(Rbad).min() < 0 and abs(np.ones(3) @ Rbad @ np.ones(3) + 0.6) < 1e-12,
        "最小固有値 -0.2<0、1'R1=-0.6 → ありえない（answer=false）")

q.done()
