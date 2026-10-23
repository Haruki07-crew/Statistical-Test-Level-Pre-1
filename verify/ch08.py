"""第8章 検算スクリプト（点推定・最尤法・十分統計量・情報量）。
記号計算は sympy（スコア・情報量・CR 下限・MLE の導出）、数値は scipy.stats、
主張値との比較は小数第4位（許容 5e-5、表示桁が少ないものは丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の数倍）。
"""
from math import comb, exp, sqrt, log, factorial, pi
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def zero(expr):
    return sp.simplify(expr) == 0


def series_ok(xs, ys, f, tol):
    """図のデータ列 ys が f(x) を表示桁で丸めた値と一致するか。最大誤差を返す。"""
    err = max(abs(f(x) - y) for x, y in zip(xs, ys))
    return err <= tol, err


def Emom(expr, x, moms):
    """x の多項式 expr の期待値。moms[k]=E[x^k]。"""
    poly = sp.Poly(sp.expand(expr), x)
    return sp.simplify(sum(c * moms[m[0]] for m, c in zip(poly.monoms(), poly.coeffs())))


rng = np.random.default_rng(20261023)
Phi = stats.norm.cdf
x = sp.Symbol("x", real=True)
n, m = sp.symbols("n m", positive=True, integer=True)
p = sp.Symbol("p", positive=True)
lam, mu, v, sig, th = sp.symbols("lambda mu v sigma theta", positive=True)
mu_r = sp.Symbol("mu", real=True)

# 各分布の log f と x のモーメント（スコア・情報量の記号検算に共通で使う）
dists = {
    "Bern(p)": (x * sp.log(p) + (1 - x) * sp.log(1 - p), p, {0: 1, 1: p, 2: p, 3: p, 4: p}, 1 / (p * (1 - p))),
    "Bin(m,p)": (x * sp.log(p) + (m - x) * sp.log(1 - p), p,
                 {0: 1, 1: m * p, 2: m * p * (1 - p) + m**2 * p**2}, m / (p * (1 - p))),
    "Po(λ)": (x * sp.log(lam) - lam - sp.log(sp.factorial(x)), lam, {0: 1, 1: lam, 2: lam + lam**2}, 1 / lam),
    "Ex(λ)": (sp.log(lam) - lam * x, lam, {0: 1, 1: 1 / lam, 2: 2 / lam**2}, 1 / lam**2),
}

# ================= score-function =================
T = "score-function"
for name, (lf, par, moms, info) in dists.items():
    s = sp.diff(lf, par)
    if name == "Po(λ)":
        s = sp.expand(s)
    Es = Emom(s, x, moms)
    Es2 = Emom(sp.expand(s**2), x, moms) if name != "Bin(m,p)" else None
    check(T, f"{name} E[s]=0", zero(Es), Es)
    if Es2 is not None:
        check(T, f"{name} E[s²]=I={info}", zero(Es2 - info), sp.simplify(Es2))
s_b = sp.diff(dists["Bern(p)"][0], p)
check(T, "Bern スコア = (x-p)/{p(1-p)}", zero(s_b - (x - p) / (p * (1 - p))), sp.simplify(s_b))
check(T, "Po スコア = (x-λ)/λ", zero(sp.diff(x * sp.log(lam) - lam, lam) - (x - lam) / lam), "")
check(T, "N スコア = (x-μ)/σ²", zero(sp.diff(-(x - mu_r)**2 / (2 * sig**2), mu_r) - (x - mu_r) / sig**2), "")
check(T, "Ex スコア = 1/λ - x", zero(sp.diff(sp.log(lam) - lam * x, lam) - (1 / lam - x)), "")
# 例1
U = lambda l: 15 / l - 5
check(T, "例1 U(2)=2.5, U(3)=0, U(4)=-1.25", U(2) == 2.5 and U(3) == 0 and U(4) == -1.25, (U(2), U(3), U(4)))
xs = [1 + 0.25 * i for i in range(21)]
ys = [10, 7, 5, 3.571, 2.5, 1.667, 1, 0.455, 0, -0.385, -0.714, -1, -1.25, -1.471, -1.667, -1.842, -2, -2.143, -2.273, -2.391, -2.5]
ok, err = series_ok(xs, ys, U, 5e-4)
check(T, "図 U(λ)=15/λ-5 の21点（許容5e-4）", ok, err)
# 例2
s1, s0 = 0.7 / 0.21, -0.3 / 0.21
check(T, "例2 s(1)=3.3333, s(0)=-1.4286", close(s1, 3.3333) and close(s0, -1.4286), (s1, s0))
check(T, "例2 E[s]=0, E[s²]=4.7619=1/0.21", close(0.3 * s1 + 0.7 * s0, 0, 1e-12) and close(0.3 * s1**2 + 0.7 * s0**2, 4.7619) and close(1 / 0.21, 4.7619),
      0.3 * s1**2 + 0.7 * s0**2)
check(T, "例2 内訳 0.3·3.3333²=3.3333, 0.7·1.4286²=1.4286", close(0.3 * s1**2, 3.3333) and close(0.7 * s0**2, 1.4286), "")
# シミュ: 指数分布スコアの平均0・分散1/λ²
X = rng.exponential(1 / 2.0, 1_000_000)
sc = 1 / 2.0 - X
check(T, "シミュ Ex(2) スコア 平均≈0（許容0.002）, 分散≈0.25（許容0.002）", abs(sc.mean()) < 0.002 and abs(sc.var() - 0.25) < 0.002, (sc.mean(), sc.var()))
# スコア検定統計量の漸近 χ²1（ポアソン λ0=3, n=50）
Xs = rng.poisson(3, (200000, 50))
Ust = (Xs.sum(axis=1) / 3 - 50) ** 2 / (50 / 3)
Tsum = stats.poisson(150)
kk_ = np.arange(0, 400)
exact_rej = Tsum.pmf(kk_)[(kk_ / 3 - 50) ** 2 / (50 / 3) > 3.8415].sum()
check(T, "シミュ スコア検定 U²/(nI)（Po λ0=3, n=50）棄却率が正確計算と一致（許容0.002）", abs((Ust > 3.8415).mean() - exact_rej) < 0.002, ((Ust > 3.8415).mean(), exact_rej))
big = stats.poisson(3 * 20000); kb = np.arange(50000, 70000)
rej_big = big.pmf(kb)[(kb / 3 - 20000) ** 2 / (20000 / 3) > 3.8415].sum()
check(T, "スコア検定の棄却率 → 0.05（n=20000 の正確計算, 許容0.002；n=50 は離散性で 0.046 程度）", abs(rej_big - 0.05) < 0.002, rej_big)

# ================= fisher-information =================
T = "fisher-information"
for name, (lf, par, moms, info) in dists.items():
    d2 = sp.diff(lf, par, 2)
    I2 = -Emom(d2, x, moms)
    check(T, f"{name} -E[∂²logf]={info}", zero(I2 - info), sp.simplify(I2))
lfN = -sp.log(2 * sp.pi * sig**2) / 2 - (x - mu_r)**2 / (2 * sig**2)
check(T, "N(μ) I=1/σ²", zero(-sp.diff(lfN, mu_r, 2) - 1 / sig**2), -sp.diff(lfN, mu_r, 2))
y = sp.Symbol("y", real=True)  # y = x-μ
cm = {0: 1, 1: 0, 2: v, 3: 0, 4: 3 * v**2}
lfv = -sp.log(2 * sp.pi) / 2 - sp.log(v) / 2 - y**2 / (2 * v)
sv = sp.diff(lfv, v)
check(T, "N(σ²) ∂/∂v = -1/(2v)+(x-μ)²/(2v²)", zero(sv - (-1 / (2 * v) + y**2 / (2 * v**2))), sv)
check(T, "N(σ²) ∂²/∂v² = 1/(2v²)-(x-μ)²/v³", zero(sp.diff(lfv, v, 2) - (1 / (2 * v**2) - y**2 / v**3)), sp.diff(lfv, v, 2))
check(T, "N(σ²) -E[∂²]=1/(2σ⁴)", zero(-Emom(sp.diff(lfv, v, 2), y, cm) - 1 / (2 * v**2)), -Emom(sp.diff(lfv, v, 2), y, cm))
check(T, "N(σ²) E[s²]=1/(2σ⁴)", zero(Emom(sp.expand(sv**2), y, cm) - 1 / (2 * v**2)), "")
# 交差項
lf2 = -sp.log(2 * sp.pi * v) / 2 - (x - mu_r)**2 / (2 * v)
cross = sp.diff(lf2, mu_r, v).subs(x, y + mu_r)
check(T, "N(μ,σ²) 交差項 E[-(X-μ)/v²]=0（対角）", zero(Emom(sp.expand(cross), y, cm)), cross)
# 母数の取り換え：指数を平均 θ で
lft = -sp.log(th) - x / th
I_th = -Emom(sp.diff(lft, th, 2), x, {0: 1, 1: th, 2: 2 * th**2})
check(T, "指数を平均θで I(θ)=1/θ²（直接計算）", zero(I_th - 1 / th**2), I_th)
check(T, "取り換え I(λ)(dλ/dθ)² = λ²/θ⁴ = 1/θ²（λ=1/θ）", zero((1 / lam**2).subs(lam, 1 / th) * sp.diff(1 / th, th)**2 - 1 / th**2), "")
# 図1：ℓ(p)-ℓ(0.3)
xs = [0.05 + 0.025 * i for i in range(25)]
y10 = [-3.238, -2.208, -1.537, -1.064, -0.72, -0.467, -0.282, -0.151, -0.064, -0.015, 0, -0.014, -0.056, -0.124, -0.216, -0.332, -0.472, -0.635, -0.823, -1.036, -1.274, -1.541, -1.838, -2.167, -2.532]
y40 = [-12.95, -8.832, -6.147, -4.258, -2.881, -1.867, -1.127, -0.602, -0.256, -0.062, 0, -0.058, -0.225, -0.495, -0.864, -1.328, -1.887, -2.541, -3.291, -4.142, -5.098, -6.165, -7.351, -8.669, -10.13]
dl = lambda q, k: k * (3 * log(q / 0.3) + 7 * log((1 - q) / 0.7))
ok, err = series_ok(xs, y10, lambda q: dl(q, 1), 5e-4)
check(T, "図1 n=10 の ℓ(p)-ℓ(0.3)（許容5e-4）", ok, err)
ok, err = series_ok(xs, y40, lambda q: dl(q, 4), 5e-3)
check(T, "図1 n=40 の ℓ(p)-ℓ(0.3)（許容5e-3）", ok, err)
xs = [0.02 + 0.04 * i for i in range(25)]
yI = [51.02, 17.73, 11.111, 8.306, 6.775, 5.828, 5.198, 4.762, 4.456, 4.244, 4.105, 4.026, 4, 4.026, 4.105, 4.244, 4.456, 4.762, 5.198, 5.828, 6.775, 8.306, 11.111, 17.73, 51.02]
ok, err = series_ok(xs, yI, lambda q: 1 / (q * (1 - q)), 5e-3)
check(T, "図2 I(p)=1/{p(1-p)} の25点（許容5e-3）", ok, err)
check(T, "例1 nI=476.19, 下限0.0021, SE 0.0458", close(100 / 0.21, 476.19, 5e-3) and close(0.21 / 100, 0.0021) and close(sqrt(0.0021), 0.0458), sqrt(0.0021))
check(T, "例2 Po: nI=5, 下限0.2, SE 0.4472", 20 / 4 == 5 and close(sqrt(0.2), 0.4472), sqrt(0.2))
check(T, "例2 Ex: nI=200, 下限0.005, SE 0.0707", close(50 / 0.25, 200, 1e-12) and close(sqrt(0.005), 0.0707), sqrt(0.005))
check(T, "例3 nI=10/32=0.3125, 下限3.2", close(10 / 32, 0.3125, 1e-12) and close(32 / 10, 3.2, 1e-12), 10 / 32)
# シミュ：加法性 V[U]=nI（ベルヌーイ p=0.3, n=100）
B = rng.binomial(100, 0.3, 500000)
Ub = (B - 100 * 0.3) / 0.21
check(T, "シミュ Bern n=100 全体スコアの分散≈476.19（相対許容1%）", abs(Ub.var() / 476.19 - 1) < 0.01, Ub.var())

# ================= cramer-rao-inequality =================
T = "cramer-rao-inequality"
check(T, "例1 Bern/Po/N の CR 下限 = 推定量の分散", zero(1 / (n / (p * (1 - p))) - p * (1 - p) / n) and zero(1 / (n / lam) - lam / n) and zero(1 / (n / sig**2) - sig**2 / n), "")
# スコアが 1次式
xb = sp.Symbol("xbar", real=True)
Ubern = sp.diff(n * xb * sp.log(p) + (n - n * xb) * sp.log(1 - p), p)
check(T, "例1 Bern U = n/(p(1-p))·(X̄-p)", zero(Ubern - n / (p * (1 - p)) * (xb - p)), sp.simplify(Ubern))
Upo = sp.diff(n * xb * sp.log(lam) - n * lam, lam)
check(T, "例1 Po U = (n/λ)(X̄-λ)", zero(Upo - n / lam * (xb - lam)), "")
check(T, "例1 p=0.3,n=100 下限 0.0021", close(0.21 / 100, 0.0021), 0.0021)
g = 1 / lam
crb = sp.diff(g, lam)**2 / (n / lam**2)
check(T, "例2 g=1/λ の下限 = 1/(nλ²)", zero(crb - 1 / (n * lam**2)), sp.simplify(crb))
Uex = sp.diff(n * sp.log(lam) - lam * n * xb, lam)
check(T, "例2 U = n/λ - ΣX = -n(X̄-1/λ)", zero(Uex + n * (xb - 1 / lam)), Uex)
S = sp.Symbol("S", positive=True)
nn = sp.Symbol("nn", positive=True)
dens = lam**nn * S**(nn - 1) * sp.exp(-lam * S) / sp.gamma(nn)
E1 = sp.simplify(sp.integrate(dens / S, (S, 0, sp.oo), conds="none"))
E2 = sp.simplify(sp.integrate(dens / S**2, (S, 0, sp.oo), conds="none"))
check(T, "例2 E[1/S]=λ/(n-1)", zero(sp.gammasimp(E1 - lam / (nn - 1))), E1)
check(T, "例2 E[1/S²]=λ²/((n-1)(n-2))", zero(sp.gammasimp(E2 - lam**2 / ((nn - 1) * (nn - 2)))), E2)
VT = sp.simplify((nn - 1)**2 * lam**2 / ((nn - 1) * (nn - 2)) - lam**2)
check(T, "例2 V[(n-1)/S]=λ²/(n-2)", zero(VT - lam**2 / (nn - 2)), VT)
check(T, "例2 n=10: 0.125λ² vs 0.1λ², 効率0.8", close(1 / 8, 0.125, 1e-12) and close(8 / 10, 0.8, 1e-12), "")
xs = list(range(3, 31))
yT = [1, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429, 0.125, 0.1111, 0.1, 0.0909, 0.0833, 0.0769, 0.0714, 0.0667, 0.0625, 0.0588, 0.0556, 0.0526, 0.05, 0.0476, 0.0455, 0.0435, 0.0417, 0.04, 0.0385, 0.037, 0.0357]
yC = [0.3333, 0.25, 0.2, 0.1667, 0.1429, 0.125, 0.1111, 0.1, 0.0909, 0.0833, 0.0769, 0.0714, 0.0667, 0.0625, 0.0588, 0.0556, 0.0526, 0.05, 0.0476, 0.0455, 0.0435, 0.0417, 0.04, 0.0385, 0.037, 0.0357, 0.0345, 0.0333]
ok1, e1 = series_ok(xs, yT, lambda k: 1 / (k - 2), 5e-5)
ok2, e2 = series_ok(xs, yC, lambda k: 1 / k, 5e-5)
check(T, "図 1/(n-2) と 1/n（n=3..30, 許容5e-5）", ok1 and ok2, (e1, e2))
lam0, n0 = 2.0, 10
Sx = rng.gamma(n0, 1 / lam0, 1_000_000)
Tt = (n0 - 1) / Sx
check(T, "シミュ Ex(2), n=10: E[(n-1)/S]≈2（許容0.003）, V≈4/8=0.5（許容0.005）", abs(Tt.mean() - 2) < 0.003 and abs(Tt.var() - 0.5) < 0.005, (Tt.mean(), Tt.var()))
# 例3 一様
t_ = sp.Symbol("t", positive=True)
fmax = n * t_**(n - 1) / th**n
EX = sp.simplify(sp.integrate(t_ * fmax, (t_, 0, th)))
EX2 = sp.simplify(sp.integrate(t_**2 * fmax, (t_, 0, th)))
Vu = sp.simplify(((n + 1) / n)**2 * (EX2 - EX**2))
check(T, "例3 V[(n+1)/n·X(n)]=θ²/(n(n+2))", zero(Vu - th**2 / (n * (n + 2))), Vu)
# CR 証明 Cov(T,U)=g'(θ)：ポアソン T=X̄, U=(n/λ)(X̄-λ) → Cov=1
Xp = rng.poisson(3, (300000, 10))
Xbar = Xp.mean(axis=1)
Up = 10 / 3 * (Xbar - 3)
check(T, "シミュ Po(3), n=10: Cov(X̄,U)≈g'(λ)=1（許容0.01）", abs(np.cov(Xbar, Up)[0, 1] - 1) < 0.01, np.cov(Xbar, Up)[0, 1])

# ================= efficient-estimator =================
T = "efficient-estimator"
check(T, "N(μ) スコア Σ(Xi-μ)/σ² = n/σ²·(X̄-μ)（n=4）", zero(sp.diff(sum(-(a - mu_r)**2 / (2 * sig**2) for a in sp.symbols("z1:5", real=True)), mu_r) - 4 / sig**2 * (sum(sp.symbols("z1:5", real=True)) / 4 - mu_r)), "")
check(T, "例1 下限 2σ⁴/n=3.2", close(2 * 16 / 10, 3.2, 1e-12), 3.2)
check(T, "例1 μ既知 V[T]=σ⁴/n²·2n=3.2", close(16 / 100 * 20, 3.2, 1e-12), 16 / 100 * 20)
check(T, "例1 V[S²]=32/9≈3.5556, 効率0.9", close(32 / 9, 3.5556) and close(3.2 / (32 / 9), 0.9, 1e-12), 32 / 9)
Z = rng.normal(0, 2, (400000, 10))
S2 = Z.var(axis=1, ddof=1)
Tk = (Z**2).mean(axis=1)
check(T, "シミュ σ²=4,n=10: V[S²]≈3.5556（許容0.04）, V[T(μ既知)]≈3.2（許容0.04）", abs(S2.var() - 32 / 9) < 0.04 and abs(Tk.var() - 3.2) < 0.04, (S2.var(), Tk.var()))
check(T, "例2 √0.1=0.3162, π/20=0.1571, √(π/20)=0.3963", close(sqrt(0.1), 0.3162) and close(pi / 20, 0.1571) and close(sqrt(pi / 20), 0.3963), sqrt(pi / 20))
check(T, "例2 2/π=0.6366, π/2≈1.57", close(2 / pi, 0.6366) and close(pi / 2, 1.57, 5e-3), 2 / pi)
check(T, "効率0.9 → 1/0.9≈1.11", close(1 / 0.9, 1.11, 5e-3), 1 / 0.9)
Zm = rng.normal(0, 1, (20000, 401))
rel = 1 / 401 / np.median(Zm, axis=1).var()
check(T, "シミュ n=401 の (1/n)/V[中央値]≈2/π（許容0.03）", abs(rel - 2 / pi) < 0.03, rel)

# ================= exponential-family =================
T = "exponential-family"
eta = sp.Symbol("eta", real=True)
A_b = sp.log(1 + sp.exp(eta))
pe = sp.exp(eta) / (1 + sp.exp(eta))
check(T, "Bern A'(η)=p", zero(sp.diff(A_b, eta) - pe), "")
check(T, "Bern A''(η)=p(1-p)", zero(sp.diff(A_b, eta, 2) - pe * (1 - pe)), "")
check(T, "Bern A=-log(1-p)", zero(A_b + sp.log(1 - pe)), "")
# ベルヌーイの書き換え
pp_ = sp.Rational(3, 10)
ok = all(zero(pp_**xx * (1 - pp_)**(1 - xx) - sp.exp(xx * sp.log(pp_ / (1 - pp_)) + sp.log(1 - pp_))) for xx in [0, 1])
check(T, "Bern p^x(1-p)^(1-x)=exp{xη+log(1-p)}", ok, "")
# ポアソン
A_p = sp.exp(eta)
check(T, "Po A'=A''=e^η=λ", zero(sp.diff(A_p, eta) - sp.diff(A_p, eta, 2)), "")
# 指数分布 A=-log(-η)
etn = sp.Symbol("eta_n", negative=True)
A_e = -sp.log(-etn)
check(T, "Ex A'(η)=-1/η=1/λ, A''=1/η²=1/λ²", zero(sp.diff(A_e, etn) + 1 / etn) and zero(sp.diff(A_e, etn, 2) - 1 / etn**2), "")
check(T, "Ex 正規化 ∫exp(ηx)dx = e^{A}", zero(sp.integrate(sp.exp(etn * x), (x, 0, sp.oo)) - sp.exp(A_e)), "")
# 正規 σ² 既知 A=σ²η²/2
A_n = sig**2 * eta**2 / 2
check(T, "N(σ²既知) A(μ/σ²)=μ²/(2σ²), A'=μ, A''=σ²", zero(A_n.subs(eta, mu_r / sig**2) - mu_r**2 / (2 * sig**2)) and zero(sp.diff(A_n, eta).subs(eta, mu_r / sig**2) - mu_r) and zero(sp.diff(A_n, eta, 2) - sig**2), "")
# 二項
A_bin = m * sp.log(1 + sp.exp(eta))
check(T, "Bin A'=mp, A''=mp(1-p)", zero(sp.diff(A_bin, eta) - m * pe) and zero(sp.diff(A_bin, eta, 2) - m * pe * (1 - pe)), "")
# ガンマの自然母数
al, be = sp.symbols("alpha beta", positive=True)
lg = sp.expand_log(sp.log(be**al / sp.gamma(al) * x**(al - 1) * sp.exp(-be * x)), force=True)
check(T, "Gamma log f の log x と x の係数 = (α-1, -β)", zero(sp.diff(lg, x) - ((al - 1) / x - be)), "")
# 正規2母数
lN = -(x - mu_r)**2 / (2 * v)
check(T, "N 2母数 x と x² の係数 = (μ/σ², -1/(2σ²))", zero(sp.expand(lN).coeff(x, 1) - mu_r / v) and zero(sp.expand(lN).coeff(x, 2) + 1 / (2 * v)), "")
# キュムラント母関数（ポアソン）
s_ = sp.Symbol("s", real=True)
k = sp.Symbol("k", integer=True, nonnegative=True)
lam_e = sp.exp(eta)
mgf = sp.exp(lam_e * (sp.exp(s_) - 1))  # ポアソンの MGF
check(T, "Po log E[e^{sT}] = A(η+s)-A(η)", zero(sp.log(mgf) - (sp.exp(eta + s_) - sp.exp(eta))) or zero(sp.expand(sp.log(mgf), force=True) - sp.expand(sp.exp(eta + s_) - sp.exp(eta))), "")
# 例1
e1 = log(0.3 / 0.7)
check(T, "例1 η=-0.8473, A=0.3567", close(e1, -0.8473) and close(log(1 + exp(e1)), 0.3567) and close(-log(0.7), 0.3567), (e1, -log(0.7)))
pf = float(pe.subs(eta, e1))
check(T, "例1 A'=0.3, A''=0.21", close(pf, 0.3, 1e-12) and close(pf * (1 - pf), 0.21, 1e-12), pf * (1 - pf))
check(T, "例1 I(η)=I(p)(dp/dη)²=0.21", close((1 / 0.21) * 0.21**2, 0.21, 1e-12), "")
check(T, "例2 η̂=log3=1.0986", close(log(3), 1.0986), log(3))
xs = [-4 + 0.5 * i for i in range(17)]
yA = [0.0181, 0.0298, 0.0486, 0.0789, 0.1269, 0.2014, 0.3133, 0.4741, 0.6931, 0.9741, 1.3133, 1.7014, 2.1269, 2.5789, 3.0486, 3.5298, 4.0181]
yA1 = [0.018, 0.0293, 0.0474, 0.0759, 0.1192, 0.1824, 0.2689, 0.3775, 0.5, 0.6225, 0.7311, 0.8176, 0.8808, 0.9241, 0.9526, 0.9707, 0.982]
yA2 = [0.0177, 0.0285, 0.0452, 0.0701, 0.105, 0.1491, 0.1966, 0.235, 0.25, 0.235, 0.1966, 0.1491, 0.105, 0.0701, 0.0452, 0.0285, 0.0177]
sig_ = lambda e: 1 / (1 + exp(-e))
r = [series_ok(xs, yA, lambda e: log(1 + exp(e)), 5e-5), series_ok(xs, yA1, sig_, 5e-5), series_ok(xs, yA2, lambda e: sig_(e) * (1 - sig_(e)), 5e-5)]
check(T, "図 A, A', A''（17点×3, 許容5e-5）", all(o for o, _ in r), [e for _, e in r])
# シミュ：ポアソン T の分散 = A''
Xq = rng.poisson(3, 1_000_000)
check(T, "シミュ Po(3): E[T]≈A'=3, V[T]≈A''=3（許容0.01）", abs(Xq.mean() - 3) < 0.01 and abs(Xq.var() - 3) < 0.02, (Xq.mean(), Xq.var()))

# ================= likelihood-function =================
T = "likelihood-function"
L3, L5 = 0.3**3 * 0.7**7, 0.5**10
check(T, "例1 L(0.3)=0.0022236", close(L3, 0.0022236, 5e-8), L3)
check(T, "例1 L(0.5)=0.0009766", close(L5, 0.0009766, 5e-8), L5)
check(T, "例1 比 2.2769", close(L3 / L5, 2.2769), L3 / L5)
check(T, "例1 ℓ(0.3)=-6.1086, ℓ(0.5)=-6.9315, 差0.8228", close(log(L3), -6.1086) and close(log(L5), -6.9315) and close(log(L3 / L5), 0.8228), (log(L3), log(L5)))
lfac = log(2 * 24 * 6 * 1 * 120)
check(T, "例2 Σlog xi! = log 34560 = 10.4505", close(lfac, 10.4505) and close(sum(log(factorial(k)) for k in [2, 4, 3, 1, 5]), 10.4505), lfac)
l3 = 15 * log(3) - 15 - lfac
l2 = 15 * log(2) - 10 - lfac
check(T, "例2 ℓ(3)=-8.9713, ℓ(2)=-10.0532, 差1.082", close(l3, -8.9713) and close(l2, -10.0532) and close(l3 - l2, 1.082, 5e-4), (l3, l2))
check(T, "例2 scipy 対数pmfの和と一致", close(stats.poisson.logpmf([2, 4, 3, 1, 5], 3).sum(), l3, 1e-9), stats.poisson.logpmf([2, 4, 3, 1, 5], 3).sum())
xs = [0.02 + 0.04 * i for i in range(25)]
yR = [0.0031, 0.063, 0.2151, 0.4294, 0.6538, 0.8412, 0.9605, 1, 0.9643, 0.869, 0.7357, 0.5861, 0.4392, 0.3086, 0.2023, 0.1226, 0.0679, 0.0337, 0.0146, 0.0053, 0.0015, 0.0003, 0, 0, 0]
yl = [-5.769, -2.765, -1.537, -0.845, -0.425, -0.173, -0.04, 0, -0.036, -0.14, -0.307, -0.534, -0.823, -1.176, -1.598, -2.099, -2.69, -3.389, -4.224, -5.236, -6.49, -8.107, -10.326, -13.771, -21.336]
ok, err = series_ok(xs, yR, lambda q: q**3 * (1 - q)**7 / L3, 5e-5)
check(T, "図1 相対尤度 25点（許容5e-5）", ok, err)
ok, err = series_ok(xs, yl, lambda q: dl(q, 1), 5e-4)
check(T, "図2 ℓ(p)-ℓ(0.3) 25点（許容5e-4）", ok, err)
check(T, "図1 p=0.5 で約0.44倍", close(L5 / L3, 0.44, 5e-3), L5 / L3)
check(T, "二項係数をつけても比は同じ", close(stats.binom.pmf(3, 10, 0.3) / stats.binom.pmf(3, 10, 0.5), L3 / L5, 1e-9), "")
check(T, "正規 ℓ の行（σ=2, μ=1, データ）が scipy と一致", close(-5 / 2 * log(2 * pi * 4) - sum((a - 1)**2 for a in [0.5, 1.2, 3, -1, 2]) / 8, stats.norm.logpdf([0.5, 1.2, 3, -1, 2], 1, 2).sum(), 1e-9), "")

# ================= maximum-likelihood-estimator =================
T = "maximum-likelihood-estimator"
t_s = sp.Symbol("t", positive=True)
sol = sp.solve(sp.diff(t_s * sp.log(p) + (n - t_s) * sp.log(1 - p), p), p)
check(T, "Bern 尤度方程式 → p̂=t/n", sol == [t_s / n], sol)
sol = sp.solve(sp.diff(-n * lam + t_s * sp.log(lam), lam), lam)
check(T, "Po 尤度方程式 → λ̂=t/n", sol == [t_s / n], sol)
check(T, "Po ℓ''<0", sp.simplify(sp.diff(-n * lam + t_s * sp.log(lam), lam, 2)).is_negative, "")
sol = sp.solve(sp.diff(n * sp.log(lam) - lam * t_s, lam), lam)
check(T, "Ex 尤度方程式 → λ̂=n/Σx", sol == [n / t_s], sol)
Qs = sp.Symbol("Q", positive=True)
sol = sp.solve(sp.diff(-n / 2 * sp.log(v) - Qs / (2 * v), v), v)
check(T, "N 尤度方程式 → v̂=Q/n", sol == [Qs / n], sol)
sol = sp.solve(sp.diff(-n / 2 * sp.log(th) - Qs / (2 * th), th), th)
check(T, "Ex 平均θ流儀 θ̂=x̄", sp.solve(sp.diff(-n * sp.log(th) - t_s / th, th), th) == [t_s / n], "")
# 二項
sol = sp.solve(sp.diff(t_s * sp.log(p) + (m * n - t_s) * sp.log(1 - p), p), p)
check(T, "Bin(m,p) p̂=Σx/(mn)=X̄/m", sol == [t_s / (m * n)], sol)
check(T, "例1 λ̂=3, ℓ'(3)=0", 15 / 5 == 3 and -5 + 15 / 3 == 0, 3)
dat = np.array([4.8, 5.6, 5.1, 4.5, 5.5])
ss = ((dat - dat.mean())**2).sum()
check(T, "例2 μ̂=5.1, 偏差平方和0.86", close(dat.mean(), 5.1, 1e-9) and close(ss, 0.86, 1e-9), ss)
check(T, "例2 σ̂²=0.172, σ̂=0.4147, 不偏分散0.215", close(ss / 5, 0.172, 1e-9) and close(sqrt(ss / 5), 0.4147) and close(ss / 4, 0.215, 1e-9), sqrt(ss / 5))
check(T, "例2 scipy norm.fit と一致", close(stats.norm.fit(dat)[1], sqrt(ss / 5), 1e-6), stats.norm.fit(dat))
w = np.array([2.5, 0.8, 4.1, 1.6, 3.0, 0.6])
check(T, "例3 x̄=2.1, λ̂=0.4762", close(w.mean(), 2.1, 1e-9) and close(1 / w.mean(), 0.4762), 1 / w.mean())
u = np.array([2.1, 7.4, 5.0, 3.3, 9.2])
check(T, "例3 一様 θ̂=9.2, 2x̄=10.8", u.max() == 9.2 and close(2 * u.mean(), 10.8, 1e-9), 2 * u.mean())
xs = [1 + 0.25 * i for i in range(21)]
yM = [-6.479, -4.382, -2.897, -1.835, -1.082, -0.565, -0.235, -0.055, 0, -0.049, -0.188, -0.403, -0.685, -1.025, -1.418, -1.857, -2.338, -2.856, -3.408, -3.991, -4.603]
ok, err = series_ok(xs, yM, lambda l: 15 * log(l / 3) - 5 * (l - 3), 5e-4)
check(T, "図 ℓ(λ)-ℓ(3) 21点（許容5e-4）", ok, err)
Zs = rng.normal(0, 1, (300000, 5))
check(T, "シミュ E[σ̂²]=(n-1)/n σ²=0.8（n=5, 許容0.005）", abs(Zs.var(axis=1).mean() - 0.8) < 0.005, Zs.var(axis=1).mean())

# ================= asymptotic-normality-of-mle =================
T = "asymptotic-normality-of-mle"
mm_ = sp.Symbol("mu_", positive=True)
gp = sp.diff(1 / mm_, mm_).subs(mm_, 1 / lam)
check(T, "指数 g'(1/λ)=-λ², λ⁴/(nλ²)=λ²/n", zero(gp + lam**2) and zero(gp**2 / (n * lam**2) - lam**2 / n), gp)
se = 0.5 / sqrt(50)
check(T, "例1 SE=0.0707, CI (0.3614, 0.6386)", close(se, 0.0707) and close(0.5 - 1.96 * se, 0.3614) and close(0.5 + 1.96 * se, 0.6386), (0.5 - 1.96 * se, 0.5 + 1.96 * se))
se2 = sqrt(0.35 * 0.65 / 200)
check(T, "例2 SE=0.0337, CI (0.2839, 0.4161)", close(se2, 0.0337) and close(0.35 - 1.96 * se2, 0.2839) and close(0.35 + 1.96 * se2, 0.4161), (0.35 - 1.96 * se2, 0.35 + 1.96 * se2))
check(T, "例3 4×0.0707=0.2828=2/√50", close(4 * 0.0707, 0.2828) and close(2 / sqrt(50), 0.2828), 2 / sqrt(50))
Xe = rng.exponential(1 / 0.5, (200000, 50))
lh = 1 / Xe.mean(axis=1)
zz = np.sqrt(50) * (lh - 0.5) / 0.5
check(T, "シミュ 指数 λ=0.5,n=50: sd(λ̂)≈0.0707（実際は有限n補正で0.0737程度, 許容0.005）", abs(lh.std() - 0.0707) < 0.005, lh.std())
check(T, "シミュ 指数 n=1000: √n(λ̂-λ)/λ の分散≈1（許容0.02）", abs((np.sqrt(1000) * (1 / rng.exponential(2, (20000, 1000)).mean(axis=1) - 0.5) / 0.5).var() - 1) < 0.02, "")
U_ = rng.uniform(0, 1, (200000, 100)).max(axis=1)
check(T, "シミュ U(0,1) n=100: n(θ-X(n)) の平均≈1, P(>1)≈e^{-1}（許容0.01）", abs((100 * (1 - U_)).mean() - 1) < 0.01 and abs((100 * (1 - U_) > 1).mean() - exp(-1)) < 0.01, (100 * (1 - U_)).mean())

# ================= consistent-estimator =================
T = "consistent-estimator"
check(T, "例1 チェビシェフ上限 0.84, 0.084", close(0.21 / (0.0025 * 100), 0.84, 1e-12) and close(0.21 / (0.0025 * 1000), 0.084, 1e-12), "")
pn1 = 2 * stats.norm.sf(0.05 / sqrt(0.21 / 100))
pn2 = 2 * stats.norm.sf(0.05 / sqrt(0.21 / 1000))
check(T, "例1 正規近似 0.2752（n=100）, 0.0006（n=1000）", close(pn1, 0.2752) and close(pn2, 0.0006), (pn1, pn2))
check(T, "例2 0.95^n: 0.5987, 0.0769, 0.0059", close(0.95**10, 0.5987) and close(0.95**50, 0.0769) and close(0.95**100, 0.0059), (0.95**10, 0.95**50, 0.95**100))
ph = rng.binomial(100, 0.3, 1_000_000) / 100
check(T, "シミュ参考 n=100 の正確な P(|p̂-0.3|≥0.05)≤0.84", (np.abs(ph - 0.3) >= 0.05 - 1e-12).mean() <= 0.84, (np.abs(ph - 0.3) >= 0.05 - 1e-12).mean())
Um = rng.uniform(0, 10, (200000, 10)).max(axis=1)
check(T, "シミュ U(0,10), n=10: P(X(n)≤9.5)≈0.5987（許容0.004）", abs((Um <= 9.5).mean() - 0.5987) < 0.004, (Um <= 9.5).mean())
check(T, "σ̂² のバイアス -σ²/n", zero((n - 1) / n * sig**2 - sig**2 + sig**2 / n), "")

# ================= factorization-theorem =================
T = "factorization-theorem"
check(T, "例1 L(2)/L(3)=(2/3)^15 e^5≈0.3389", close((2 / 3)**15 * exp(5), 0.3389), (2 / 3)**15 * exp(5))
A_ = np.array([2, 4, 3, 1, 5]); B_ = np.array([3, 3, 3, 3, 3])
rA = np.exp(stats.poisson.logpmf(A_, 2).sum() - stats.poisson.logpmf(A_, 3).sum())
rB = np.exp(stats.poisson.logpmf(B_, 2).sum() - stats.poisson.logpmf(B_, 3).sum())
check(T, "例1 A・B の L(2)/L(3) が同じ", close(rA, rB, 1e-12) and close(rA, 0.3389), (rA, rB))
xs = [0.5 + 0.25 * i for i in range(23)]
y15 = [0, 0.0001, 0.0015, 0.0125, 0.0552, 0.1596, 0.3389, 0.5682, 0.7907, 0.9463, 1, 0.9518, 0.8288, 0.6684, 0.5042, 0.3587, 0.2422, 0.1561, 0.0966, 0.0575, 0.0331, 0.0185, 0.01]
y10 = [0.0017, 0.0285, 0.1449, 0.3867, 0.686, 0.9182, 1, 0.9304, 0.7645, 0.5681, 0.3885, 0.2479, 0.149, 0.0851, 0.0465, 0.0244, 0.0124, 0.0061, 0.0029, 0.0014, 0.0006, 0.0003, 0.0001]
r1 = series_ok(xs, y15, lambda l: (l / 3)**15 * exp(-5 * (l - 3)), 5e-5)
r2 = series_ok(xs, y10, lambda l: (l / 2)**10 * exp(-5 * (l - 2)), 5e-5)
check(T, "図 合計15と合計10の正規化尤度（許容5e-5）", r1[0] and r2[0], (r1[1], r2[1]))
check(T, "例2 x̄=5, Σ(x-x̄)²=4, s²=1.3333, σ̂²=1", 20 / 4 == 5 and 104 - 4 * 25 == 4 and close(4 / 3, 1.3333) and 4 / 4 == 1, "")
check(T, "例3 Σx=12.6, 6/12.6=0.4762", close(w.sum(), 12.6, 1e-9) and close(6 / w.sum(), 0.4762), 6 / w.sum())
# 正規の展開
xi = sp.symbols("x1:5", real=True)
check(T, "Σ(xi-μ)²=Σxi²-2μΣxi+nμ²", zero(sum((a - mu_r)**2 for a in xi) - (sum(a**2 for a in xi) - 2 * mu_r * sum(xi) + 4 * mu_r**2)), "")
# ガンマ分解
xg = sp.symbols("y1:4", positive=True)
lhs = sp.prod([be**al / sp.gamma(al) * a**(al - 1) * sp.exp(-be * a) for a in xg])
rhs = be**(3 * al) / sp.gamma(al)**3 * sp.prod(xg)**(al - 1) * sp.exp(-be * sum(xg))
check(T, "ガンマ 同時密度 = β^{nα}/Γ(α)^n (Πx)^{α-1} e^{-βΣx}（n=3）", zero(sp.expand_power_base(sp.powsimp(lhs / rhs, force=True), force=True) - 1) or zero(sp.simplify(sp.log(lhs) - sp.log(rhs))), "")
# 離散型証明：ポアソン n=3, t=4 で P(X=x|T=t) が λ によらない
cond = [stats.poisson.pmf([1, 2, 1], l).prod() / stats.poisson.pmf(4, 3 * l) for l in [0.5, 1, 3]]
check(T, "離散型 P(X=x|T=t) が λ によらない（Po, x=(1,2,1)）", np.allclose(cond, cond[0]) and close(cond[0], factorial(4) / (1 * 2 * 1) / 3**4, 1e-12), cond[0])

# ================= sufficient-statistic =================
T = "sufficient-statistic"
check(T, "例1 p(1-p)²: 0.128, 0.125, 0.032", close(0.2 * 0.8**2, 0.128, 1e-12) and close(0.125, 0.5**3, 1e-12) and close(0.8 * 0.2**2, 0.032, 1e-12), "")
xs = [0.02 + 0.04 * i for i in range(25)]
yS = [0.0192, 0.053, 0.081, 0.1035, 0.121, 0.1338, 0.1424, 0.147, 0.1481, 0.1461, 0.1413, 0.1341, 0.125, 0.1143, 0.1023, 0.0895, 0.0763, 0.063, 0.05, 0.0378, 0.0266, 0.0169, 0.009, 0.0034, 0.0004]
ok, err = series_ok(xs, yS, lambda q: q * (1 - q)**2, 5e-5)
check(T, "図 p(1-p)² の25点（許容5e-5）", ok, err)
check(T, "例1 条件付き 1/3（どの p でも）", all(close(q * (1 - q)**2 / (3 * q * (1 - q)**2), 1 / 3, 1e-12) for q in xs), 1 / 3)
nb, tb = 5, 2
check(T, "Bern P(X=x|T=t)=1/C(n,t)", zero(p**tb * (1 - p)**(nb - tb) / (sp.binomial(nb, tb) * p**tb * (1 - p)**(nb - tb)) - sp.Rational(1, comb(nb, tb))), "")
xx_ = [2, 0, 1]
expr = sp.prod([sp.exp(-lam) * lam**a / sp.factorial(a) for a in xx_]) / (sp.exp(-3 * lam) * (3 * lam)**3 / sp.factorial(3))
check(T, "Po P(X=x|T=t)=t!/Πxi! (1/n)^t（x=(2,0,1)）", zero(expr - sp.factorial(3) / (2 * 1 * 1) * sp.Rational(1, 27)), sp.simplify(expr))
# シミュ：ベルヌーイ n=3, T=1 で (1,0,0) の条件付き割合 1/3
Bs = rng.random((600000, 3)) < 0.2
sel = Bs[Bs.sum(axis=1) == 1]
check(T, "シミュ p=0.2 で P((1,0,0)|T=1)≈1/3（許容0.005）", abs(sel[:, 0].mean() - 1 / 3) < 0.005, sel[:, 0].mean())

# ================= rao-blackwell-theorem =================
T = "rao-blackwell-theorem"
nl = sp.Symbol("n", positive=True)
Ephi2 = sp.exp(nl * lam * ((1 - 1 / nl)**2 - 1))
check(T, "E[φ²]=exp{nλ((1-1/n)²-1)}=e^{-2λ+λ/n}", zero(sp.expand(sp.log(Ephi2), force=True) - (-2 * lam + lam / nl)), "")
Ephi = sp.exp(nl * lam * ((1 - 1 / nl) - 1))
check(T, "E[(1-1/n)^T]=e^{-λ}（不偏）", zero(sp.simplify(Ephi - sp.exp(-lam))), sp.simplify(Ephi))
check(T, "V[φ]=e^{-2λ}(e^{λ/n}-1)", zero(sp.exp(-2 * lam + lam / nl) - sp.exp(-lam)**2 - sp.exp(-2 * lam) * (sp.exp(lam / nl) - 1)), "")
crb_rb = sp.diff(sp.exp(-lam), lam)**2 / (nl / lam)
check(T, "CR 下限 λe^{-2λ}/n", zero(crb_rb - lam * sp.exp(-2 * lam) / nl), crb_rb)
v1 = exp(-1) * (1 - exp(-1)); v2 = exp(-2) * (exp(0.1) - 1); v3 = exp(-2) / 10
check(T, "例2 0.2325, 0.01423, 0.01353", close(v1, 0.2325) and close(v2, 0.01423, 5e-6) and close(v3, 0.01353, 5e-6), (v1, v2, v3))
check(T, "例2 分散比 約16分の1", 15.5 <= v1 / v2 <= 16.5, v1 / v2)
check(T, "例2 (4/5)^15=0.0352, e^{-3}=0.0498", close(0.8**15, 0.0352) and close(exp(-3), 0.0498), 0.8**15)
xs = [0.2 * i for i in range(1, 21)]
yr = [0.14841, 0.22099, 0.24762, 0.24743, 0.23254, 0.21048, 0.18579, 0.16113, 0.13798, 0.11702, 0.09853, 0.08249, 0.06876, 0.05711, 0.04731, 0.0391, 0.03226, 0.02658, 0.02187, 0.01798]
yi = [0.01354, 0.01834, 0.01862, 0.01682, 0.01423, 0.01157, 0.00914, 0.00707, 0.00539, 0.00406, 0.00302, 0.00223, 0.00164, 0.00119, 0.00087, 0.00063, 0.00045, 0.00032, 0.00023, 0.00016]
yc = [0.01341, 0.01797, 0.01807, 0.01615, 0.01353, 0.01089, 0.00851, 0.00652, 0.00492, 0.00366, 0.0027, 0.00198, 0.00143, 0.00104, 0.00074, 0.00053, 0.00038, 0.00027, 0.00019, 0.00013]
rr = [series_ok(xs, yr, lambda l: exp(-l) * (1 - exp(-l)), 5e-6),
      series_ok(xs, yi, lambda l: exp(-2 * l) * (exp(l / 10) - 1), 5e-6),
      series_ok(xs, yc, lambda l: l * exp(-2 * l) / 10, 5e-6)]
check(T, "図 3系列×20点（許容5e-6）", all(o for o, _ in rr), [e for _, e in rr])
Pm = rng.poisson(1, (1_000_000, 10))
d0 = (Pm[:, 0] == 0).astype(float)
phi = 0.9 ** Pm.sum(axis=1)
check(T, "シミュ λ=1,n=10: E[1{X1=0}]≈E[φ]≈e^{-1}（許容0.002）", abs(d0.mean() - exp(-1)) < 0.002 and abs(phi.mean() - exp(-1)) < 0.001, (d0.mean(), phi.mean()))
check(T, "シミュ V[1{X1=0}]≈0.2325（許容0.002）, V[φ]≈0.01423（許容0.0003）", abs(d0.var() - v1) < 0.002 and abs(phi.var() - v2) < 0.0003, (d0.var(), phi.var()))
# 例1 Bern: E[X1|T]=T/n
Bn = rng.random((400000, 8)) < 0.3
Tn = Bn.sum(axis=1)
ok = all(abs(Bn[Tn == t, 0].mean() - t / 8) < 0.01 for t in range(1, 6))
check(T, "シミュ Bern n=8: E[X1|T=t]≈t/n（t=1..5, 許容0.01）", ok, "")

# ================= mean-squared-error =================
T = "mean-squared-error"
d = sp.Symbol("d", positive=True)
mse = 2 * (n - 1) / d**2 + ((n - 1) / d - 1)**2
dstar = sp.solve(sp.diff(mse, d), d)
check(T, "例1 MSE(Q/d) を最小にする d = n+1", dstar == [n + 1], dstar)
vals = {dd: (18 / dd**2, (9 / dd - 1)**2) for dd in range(6, 17)}
check(T, "例1 表 d=9: 0.2222/0/0.2222", close(vals[9][0], 0.2222) and vals[9][1] == 0, vals[9])
check(T, "例1 表 d=10: 0.18/0.01/0.19", close(vals[10][0], 0.18, 1e-12) and close(vals[10][1], 0.01, 1e-12) and close(sum(vals[10]), 0.19, 1e-12), vals[10])
check(T, "例1 表 d=11: 0.1488/0.0331/0.1818", close(vals[11][0], 0.1488) and close(vals[11][1], 0.0331) and close(sum(vals[11]), 0.1818), vals[11])
check(T, "例1 2/(n-1) > (2n-1)/n² > 2/(n+1)（n=10 と記号的に）", 2 / 9 > 19 / 100 > 2 / 11 and zero(mse.subs(d, n) - (2 * n - 1) / n**2) and zero(mse.subs(d, n + 1) - 2 / (n + 1)) and zero(mse.subs(d, n - 1) - 2 / (n - 1)), "")
xs = list(range(6, 17))
yb = [0.25, 0.0816, 0.0156, 0, 0.01, 0.0331, 0.0625, 0.0947, 0.1276, 0.16, 0.1914]
yv = [0.5, 0.3673, 0.2812, 0.2222, 0.18, 0.1488, 0.125, 0.1065, 0.0918, 0.08, 0.0703]
ym = [0.75, 0.449, 0.2969, 0.2222, 0.19, 0.1818, 0.1875, 0.2012, 0.2194, 0.24, 0.2617]
rr = [series_ok(xs, yb, lambda dd: (9 / dd - 1)**2, 5e-5), series_ok(xs, yv, lambda dd: 18 / dd**2, 5e-5),
      series_ok(xs, ym, lambda dd: 18 / dd**2 + (9 / dd - 1)**2, 5e-5)]
check(T, "図1 3系列×11点（許容5e-5）", all(o for o, _ in rr), [e for _, e in rr])
m1 = lambda q: q * (1 - q) / 10
m2 = lambda q: 10 * q * (1 - q) / 144 + ((10 * q + 1) / 12 - q)**2
check(T, "例2 p=0.5: 0.025, 0.017361", close(m1(0.5), 0.025, 1e-12) and close(m2(0.5), 0.017361, 5e-7), m2(0.5))
check(T, "例2 p=0.1: 0.009, 0.00625+0.004444=0.0107", close(m1(0.1), 0.009, 1e-12) and close(10 * 0.09 / 144, 0.00625, 1e-12) and close((2 / 12 - 0.1)**2, 0.004444, 5e-7) and close(m2(0.1), 0.0107), m2(0.1))
# 例2 の MSE を二項分布から厳密計算
kk = np.arange(11)
ok = all(close((stats.binom.pmf(kk, 10, q) * ((kk + 1) / 12 - q)**2).sum(), m2(q), 1e-12) for q in [0.1, 0.3, 0.5])
check(T, "例2 MSE(p̂2) の式が二項分布の厳密計算と一致", ok, "")
xs = [0.02 + 0.04 * i for i in range(25)]
y1 = [0.002, 0.0056, 0.009, 0.012, 0.0148, 0.0172, 0.0192, 0.021, 0.0224, 0.0236, 0.0244, 0.0248, 0.025, 0.0248, 0.0244, 0.0236, 0.0224, 0.021, 0.0192, 0.0172, 0.0148, 0.012, 0.009, 0.0056, 0.002]
y2 = [0.0078, 0.0093, 0.0107, 0.012, 0.0131, 0.0141, 0.015, 0.0157, 0.0163, 0.0168, 0.0171, 0.0173, 0.0174, 0.0173, 0.0171, 0.0168, 0.0163, 0.0157, 0.015, 0.0141, 0.0131, 0.012, 0.0107, 0.0093, 0.0078]
rr = [series_ok(xs, y1, m1, 5e-5), series_ok(xs, y2, m2, 5e-5)]
check(T, "図2 2系列×25点（許容5e-5）", all(o for o, _ in rr), [e for _, e in rr])
c = sp.Symbol("c", real=True)
msec = c**2 * sig**2 / n + (c - 1)**2 * mu**2
cs = sp.solve(sp.diff(msec, c), c)
check(T, "例3 c*=μ²/(μ²+σ²/n)", len(cs) == 1 and zero(cs[0] - mu**2 / (mu**2 + sig**2 / n)), cs)
check(T, "例3 μ=1,σ=2,n=4: c*=0.5, MSE=0.5, c=1 で 1", close(float(cs[0].subs({mu: 1, sig: 2, n: 4})), 0.5, 1e-12) and close(float(msec.subs({c: 0.5, mu: 1, sig: 2, n: 4})), 0.5, 1e-12) and close(float(msec.subs({c: 1, mu: 1, sig: 2, n: 4})), 1, 1e-12), "")
Zq = rng.normal(0, 1, (400000, 10))
Q = ((Zq - Zq.mean(axis=1, keepdims=True))**2).sum(axis=1)
sim = [((Q / dd - 1)**2).mean() for dd in (9, 10, 11)]
check(T, "シミュ n=10,σ²=1: MSE(Q/9,Q/10,Q/11)≈0.2222/0.19/0.1818（許容0.003）", all(abs(a - b) < 0.003 for a, b in zip(sim, [2 / 9, 0.19, 2 / 11])), sim)

# ================= method-of-moments =================
T = "method-of-moments"
Xbar_, s2_ = sp.symbols("Xbar s2", positive=True)
solg = sp.solve([al * be - Xbar_, al * be**2 - s2_], [al, be], dict=True)
check(T, "ガンマ α̂=X̄²/σ̂², β̂=σ̂²/X̄", len(solg) == 1 and zero(solg[0][al] - Xbar_**2 / s2_) and zero(solg[0][be] - s2_ / Xbar_), solg)
check(T, "例1 α̂=2, β̂=2, レート0.5", 16 / 8 == 2 and 8 / 4 == 2 and 1 / 2 == 0.5, "")
check(T, "図 Gamma(2, 尺度2) 平均4 分散8", stats.gamma.mean(2, scale=2) == 4 and stats.gamma.var(2, scale=2) == 8, "")
a_, b_ = sp.symbols("a b", real=True)
sig_h = sp.Symbol("sh", positive=True)
solu = sp.solve([(a_ + b_) / 2 - Xbar_, (b_ - a_) / (2 * sp.sqrt(3)) - sig_h], [a_, b_], dict=True)
check(T, "U(a,b) â=X̄-√3σ̂, b̂=X̄+√3σ̂", zero(solu[0][a_] - (Xbar_ - sp.sqrt(3) * sig_h)) and zero(solu[0][b_] - (Xbar_ + sp.sqrt(3) * sig_h)), solu)
ua, ub = sp.symbols("ua ub", real=True)
check(T, "U(a,b) V=(b-a)²/12", zero(sp.integrate(x**2 / (ub - ua), (x, ua, ub)) - ((ua + ub) / 2)**2 - (ub - ua)**2 / 12), "")
check(T, "例2 x̄=5.4, 2x̄=10.8, (6/5)·9.2=11.04", close(u.mean(), 5.4, 1e-9) and close(6 / 5 * 9.2, 11.04, 1e-9), "")
check(T, "例2 V[2X̄]=θ²/(3n)", zero(4 * th**2 / 12 / n - th**2 / (3 * n)), "")
s2u = u.var()
check(T, "例3 σ̂²=6.78, â=0.8900, b̂=9.9100", close(s2u, 6.78, 1e-9) and close(5.4 - sqrt(3 * s2u), 0.8900) and close(5.4 + sqrt(3 * s2u), 9.9100), (5.4 - sqrt(3 * s2u), 5.4 + sqrt(3 * s2u)))
G = rng.gamma(2, 2, (2000, 2000))
ah = G.mean(axis=1)**2 / G.var(axis=1)
check(T, "シミュ Gamma(2,2), n=2000: α̂ の平均≈2（一致性, 許容0.02）", abs(ah.mean() - 2) < 0.02, ah.mean())
Uu = rng.uniform(0, 10, (300000, 5))
e1_ = 2 * Uu.mean(axis=1); e2_ = 1.2 * Uu.max(axis=1)
check(T, "シミュ U(0,10),n=5: V[2X̄]≈6.6667（許容0.06）, V[1.2X(5)]≈2.8571（許容0.03）", abs(e1_.var() - 20 / 3) < 0.06 and abs(e2_.var() - 100 / 35) < 0.03, (e1_.var(), e2_.var()))

# ================= unbiased-estimator =================
T = "unbiased-estimator"
Xs_ = sp.symbols("X1:5", real=True)
expand_ok = zero(sum((a - sum(Xs_) / 4)**2 for a in Xs_) - (sum(a**2 for a in Xs_) - 4 * (sum(Xs_) / 4)**2))
check(T, "Σ(Xi-X̄)²=ΣXi²-nX̄²（n=4）", expand_ok, "")
check(T, "n(σ²+μ²)-n(σ²/n+μ²)=(n-1)σ²", zero(n * (sig**2 + mu**2) - n * (sig**2 / n + mu**2) - (n - 1) * sig**2), "")
check(T, "例1 s²=7.5, σ̂²=6.0, 比4/5", close(np.var([102, 98, 101, 99, 105], ddof=1), 7.5, 1e-12) and close(np.var([102, 98, 101, 99, 105]), 6.0, 1e-12), "")
check(T, "例2 E[X(n)]=nθ/(n+1)", zero(EX - n * th / (n + 1)), EX)
check(T, "例2 n=5,θ=10: 6.6667, 2.8571", close(100 / 15, 6.6667) and close(100 / 35, 2.8571), (100 / 15, 100 / 35))
# 正規性不要：指数分布で S² の不偏性
E_ = rng.exponential(2.0, (400000, 5))
check(T, "シミュ 非正規（Ex 平均2, σ²=4）n=5: E[S²]≈4（許容0.03）, E[σ̂²]≈3.2（許容0.03）", abs(E_.var(axis=1, ddof=1).mean() - 4) < 0.03 and abs(E_.var(axis=1).mean() - 3.2) < 0.03, (E_.var(axis=1, ddof=1).mean(), E_.var(axis=1).mean()))
P_ = rng.poisson(4, (400000, 6))
check(T, "シミュ Po(4) n=6: E[X̄]≈E[S²]≈4（許容0.02）", abs(P_.mean() - 4) < 0.02 and abs(P_.var(axis=1, ddof=1).mean() - 4) < 0.02, P_.var(axis=1, ddof=1).mean())
Zn = rng.normal(0, 1, (400000, 5))
check(T, "シミュ E[S]<σ（n=5, 理論0.9400）", abs(Zn.std(axis=1, ddof=1).mean() - 0.9400) < 0.003, Zn.std(axis=1, ddof=1).mean())
check(T, "図 S²~Gamma(2,尺度0.5) 平均1, σ̂²~Gamma(2,尺度0.4) 平均0.8（σ²=1,n=5）", close(stats.gamma.mean(2, scale=0.5), 1, 1e-12) and close(stats.gamma.mean(2, scale=0.4), 0.8, 1e-12) and
      close(stats.kstest(Zn.var(axis=1, ddof=1)[:20000], "gamma", args=(2, 0, 0.5)).statistic, 0, 0.015), "")
n1, n2 = 4, 7
Y1 = rng.normal(0, 3, (300000, n1)); Y2 = rng.normal(5, 3, (300000, n2))
pool = ((n1 - 1) * Y1.var(axis=1, ddof=1) + (n2 - 1) * Y2.var(axis=1, ddof=1)) / (n1 + n2 - 2)
check(T, "シミュ プールした分散の期待値≈9（許容0.05）", abs(pool.mean() - 9) < 0.05, pool.mean())

# ================= invariance-of-mle =================
T = "invariance-of-mle"
check(T, "例1 e^{-3}=0.0498", close(exp(-3), 0.0498), exp(-3))
Px = rng.poisson(3, (400000, 10))
check(T, "例1 E[e^{-X̄}]≠e^{-λ}（λ=3,n=10 で理論 exp(30(e^{-0.1}-1)) と比較）", abs(np.exp(-Px.mean(axis=1)).mean() - exp(30 * (exp(-0.1) - 1))) < 5e-4 and abs(exp(30 * (exp(-0.1) - 1)) - exp(-3)) > 1e-3, (exp(30 * (exp(-0.1) - 1)), exp(-3)))
check(T, "例2 e^{-2.5}=0.0821", close(exp(-2.5), 0.0821), exp(-2.5))
check(T, "例3 0.4286, log=-0.8473", close(0.3 / 0.7, 0.4286) and close(log(3 / 7), -0.8473), log(3 / 7))
xs = [0.05 + 0.075 * i for i in range(27)]
yo = [0.0345, 0.2705, 0.5811, 0.8238, 0.959, 0.9999, 0.9749, 0.9102, 0.8258, 0.7346, 0.6449, 0.561, 0.485, 0.4177, 0.3589, 0.3079, 0.2642, 0.2267, 0.1946, 0.1673, 0.1441, 0.1242, 0.1073, 0.0929, 0.0805, 0.07, 0.0609]
Lo = lambda o: (o / (1 + o))**3 * (1 / (1 + o))**7 / L3
ok, err = series_ok(xs, yo, Lo, 5e-5)
check(T, "図 オッズで書いた相対尤度 27点（許容5e-5）", ok, err)
from scipy.optimize import minimize_scalar
om = minimize_scalar(lambda o: -Lo(o), bounds=(0.01, 2), method="bounded").x
check(T, "図 頂上 θ=0.3/0.7≈0.4286（数値最大化）", close(om, 3 / 7, 1e-4), om)
# 誘導尤度：θ² の MLE = θ̂²（正規平均）
check(T, "σ̂=√σ̂²（例: 0.172→0.4147）", close(sqrt(0.172), 0.4147), sqrt(0.172))

# ================= point-estimation =================
T = "point-estimation"
d5 = np.array([102, 98, 101, 99, 105])
check(T, "例1 x̄=101, 偏差平方和30, s²=7.5", d5.mean() == 101 and ((d5 - 101)**2).sum() == 30 and close(d5.var(ddof=1), 7.5, 1e-12), "")
check(T, "例1 SE=√1.5=1.2247", close(sqrt(7.5) / sqrt(5), 1.2247) and close(sqrt(1.5), 1.2247), sqrt(1.5))
check(T, "例2 中央値101, 30/5=6.0", np.median(d5) == 101 and 30 / 5 == 6.0, "")
check(T, "図 10/√10=3.1623", close(10 / sqrt(10), 3.1623), 10 / sqrt(10))
Zp = rng.normal(50, 10, (200000, 10)).mean(axis=1)
check(T, "シミュ 標本平均の SD≈3.1623（許容0.02）", abs(Zp.std() - 3.1623) < 0.02, Zp.std())


# ######################################################################
# ================= 重要度Cの用語 =================
# 主張値は本文の表示桁で丸めて一致するか（cc：表示桁の 0.5 単位 + extra）。
# 乱数は専用の Generator（seed=20261023+章番号）を使い、許容誤差を各チェックに明記。
def _dec(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s else 0


def cc(term, what, x, s, extra=0.0):
    tol = 0.5 * 10 ** (-_dec(s)) * (1 + 1e-9) + extra
    check(term, f"{what} = {s}", abs(float(x) - float(s)) <= tol, round(float(x), _dec(s) + 3))


def mcC(term, what, est, target, tol):
    check(term, f"{what}（シミュ、許容±{tol}）", abs(float(est) - target) <= tol, round(float(est), 5))

rngC = np.random.default_rng(20261023 + 8)

# ================= complete-statistic =================
T = "complete-statistic"
pp, nn = sp.symbols("p n", positive=True)
tt = sp.symbols("t", integer=True, nonnegative=True)
n0 = 10
ET = sum(sp.binomial(n0, k) * pp**k * (1 - pp)**(n0 - k) * k * (n0 - k) for k in range(n0 + 1))
check(T, "E[T(n-T)]=n(n-1)p(1-p)（n=10, sympy）", sp.expand(ET - n0 * (n0 - 1) * pp * (1 - pp)) == 0, "")
for n1 in (3, 7):
    E1 = sum(sp.binomial(n1, k) * pp**k * (1 - pp)**(n1 - k) * k * (n1 - k) for k in range(n1 + 1))
    check(T, f"E[T(n-T)]=n(n-1)p(1-p)（n={n1}）", sp.expand(E1 - n1 * (n1 - 1) * pp * (1 - pp)) == 0, "")
cc(T, "τ̂=3·7/(10·9)", 3 * 7 / 90, "0.2333")
cc(T, "代入 0.3×0.7", 0.21, "0.21")
Eplug = sum(sp.binomial(n0, k) * pp**k * (1 - pp)**(n0 - k) * sp.Rational(k, n0) * (1 - sp.Rational(k, n0)) for k in range(n0 + 1))
check(T, "E[p̂(1-p̂)]=p(1-p)(1-1/n)（n=10）", sp.expand(Eplug - pp * (1 - pp) * (1 - sp.Rational(1, n0))) == 0, "")
cc(T, "p=0.5 で 0.25(1-1/10)", 0.25 * 0.9, "0.225")
r = sp.symbols("r", positive=True)
gs = sp.symbols("g0:4")
lhs = sum(gs[k] * sp.binomial(3, k) * pp**k * (1 - pp)**(3 - k) for k in range(4))
rhs = (1 - pp)**3 * sum(gs[k] * sp.binomial(3, k) * r**k for k in range(4))
check(T, "E[g(T)]=(1-p)ⁿΣg(t)C(n,t)rᵗ（n=3, r=p/(1-p)）", sp.simplify(lhs - rhs.subs(r, pp / (1 - pp))) == 0, "")
X = rngC.normal(1.7, 1, (200000, 2))
mcC(T, "N(θ,1) で E[X1-X2]=0", (X[:, 0] - X[:, 1]).mean(), 0.0, 0.01)
check(T, "X1-X2 は0でない（SD≈√2）", abs((X[:, 0] - X[:, 1]).std() - sqrt(2)) < 0.01, (X[:, 0] - X[:, 1]).std())
# UMVUE τ̂ の分散 > クラメール・ラオ下限（p=0.3, n=10）
k = np.arange(11); pk = stats.binom.pmf(k, 10, 0.3)
tau = k * (10 - k) / 90
vt = (pk * tau**2).sum() - (pk * tau).sum()**2
cr = (1 - 2 * 0.3)**2 * 0.3 * 0.7 / 10
check(T, "τ̂ は不偏（p=0.3, n=10）", abs((pk * tau).sum() - 0.21) < 1e-12, (pk * tau).sum())
check(T, "V[τ̂] > CR下限 (1-2p)²p(1-p)/n（p=0.3, n=10）", vt > cr, (vt, cr))

# ================= observed-fisher-information =================
T = "observed-fisher-information"
xs = np.array([-0.4, 0.8, 1.2, 2.1, 7.5])
th, xx = sp.symbols("theta x", real=True)
lf = -sp.log(sp.pi) - sp.log(1 + (xx - th)**2)
u = sp.symbols("u", real=True)
check(T, "-d²/dθ² log f = 2(1-u²)/(1+u²)²（sympy）",
      sp.simplify((-sp.diff(lf, th, 2)).subs(xx, th + u) - 2 * (1 - u**2) / (1 + u**2)**2) == 0, "")
from scipy.optimize import minimize_scalar
nll = lambda t: np.sum(np.log1p((xs - t)**2))
res = minimize_scalar(nll, bounds=(0, 3), method="bounded", options={"xatol": 1e-10})
grid = np.linspace(-5, 12, 17001)
check(T, "θ̂ は全域最大（格子 -5〜12）", abs(grid[np.argmin([nll(g) for g in grid])] - res.x) < 2e-3, res.x)
cc(T, "θ̂", res.x, "1.109")
uu = xs - res.x
J = np.sum(2 * (1 - uu**2) / (1 + uu**2)**2)
cc(T, "J(θ̂)", J, "3.184")
cc(T, "SE 1/√J", 1 / sqrt(J), "0.560")
Ic = __import__("scipy.integrate", fromlist=["quad"]).quad(lambda x: (2 * x / (1 + x**2))**2 * stats.cauchy.pdf(x), -np.inf, np.inf)[0]
cc(T, "コーシー位置 I=1/2（数値積分）", Ic, "0.5")
cc(T, "nI", 5 * Ic, "2.5")
cc(T, "SE 1/√2.5", 1 / sqrt(2.5), "0.632")
h = 1e-4
Jnum = (nll(res.x + h) - 2 * nll(res.x) + nll(res.x - h)) / h**2
check(T, "J(θ̂) は対数尤度の数値2階微分と一致", abs(Jnum - J) < 1e-4, Jnum)
C = rngC.standard_cauchy((200000, 5))
EJ = np.mean(np.sum(2 * (1 - C**2) / (1 + C**2)**2, 1))
mcC(T, "E_θ[J(θ)]=nI=2.5（真値で評価）", EJ, 2.5, 0.01)
lam = sp.symbols("lambda", positive=True)
xsym = sp.symbols("x1:5", positive=True)
lp = sum(x * sp.log(lam) - lam for x in xsym)
lh = sum(xsym) / 4
Jp = (-sp.diff(lp, lam, 2)).subs(lam, lh)
check(T, "ポアソン J(λ̂)=Σx/λ̂²=n/x̄（n=4）", sp.simplify(Jp - 4 / lh) == 0, "")


print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")

# --- 追加用語（用語の見直し）: order-statistics ---
T = "order-statistics"
check(T, "例1 直列: 最小値は平均200の指数, P(>100)=e^-0.5≈0.6065", close(exp(-0.5), 0.6065) and close(stats.expon(scale=1000).sf(100) ** 5, exp(-0.5), 1e-12), round(exp(-0.5), 5))
check(T, "例1 並列: (1−e^-1)^5≈0.1009, 0.6321, 平均≈2283.3", close((1 - exp(-1)) ** 5, 0.1009) and close(1 - exp(-1), 0.6321) and close(1000 * sum(1 / j for j in range(1, 6)), 2283.3, 0.05), round((1 - exp(-1)) ** 5, 5))
L = rng.exponential(1000, (400_000, 5))
check(T, "例1 シミュ E[min]≈200（許容2）, E[max]≈2283.3（許容8）, P(max≤1000)（許容0.002）", abs(L.min(axis=1).mean() - 200) < 2 and abs(L.max(axis=1).mean() - 2283.3) < 8 and abs((L.max(axis=1) <= 1000).mean() - 0.1009) < 0.002, (round(L.min(axis=1).mean(), 2), round(L.max(axis=1).mean(), 2)))
xo = sp.symbols("x", positive=True)
def f_k(nv, kv, Fx, fx):
    return sp.factorial(nv) / (sp.factorial(kv - 1) * sp.factorial(nv - kv)) * Fx ** (kv - 1) * (1 - Fx) ** (nv - kv) * fx
f2 = sp.expand(f_k(4, 2, xo, 1))
check(T, "例2 f_(2)=12x(1−x)², E=0.4, V=0.04", sp.simplify(f2 - 12 * xo * (1 - xo) ** 2) == 0 and sp.integrate(xo * f2, (xo, 0, 1)) == sp.Rational(2, 5)
      and sp.integrate(xo**2 * f2, (xo, 0, 1)) - sp.Rational(4, 25) == sp.Rational(1, 25), "")
check(T, "例2 P(X_(2)≤0.5)=11/16=0.6875（積分・二項和・Beta）", sp.integrate(f2, (xo, 0, sp.Rational(1, 2))) == sp.Rational(11, 16) and sum(comb(4, j) for j in range(2, 5)) == 11 and close(stats.beta(2, 3).cdf(0.5), 0.6875, 1e-12), "")
ok = True
for nv, kv in [(4, 2), (5, 1), (5, 5), (7, 3), (10, 6)]:
    ok &= close(stats.beta(kv, nv - kv + 1).mean(), kv / (nv + 1), 1e-12) and close(stats.beta(kv, nv - kv + 1).var(), kv * (nv - kv + 1) / ((nv + 1) ** 2 * (nv + 2)), 1e-12)
    ok &= np.allclose([float(f_k(nv, kv, xo, 1).subs(xo, xv_)) for xv_ in (0.1, 0.37, 0.8)], stats.beta(kv, nv - kv + 1).pdf([0.1, 0.37, 0.8]), rtol=1e-12)
check(T, "U(0,1) の X_(k) ~ Beta(k,n−k+1), E=k/(n+1), V 公式（5組）", ok, "")
Us = np.sort(rng.uniform(size=(400_000, 4)), axis=1)
check(T, "例2 シミュ E[X_(2)]≈0.4, V≈0.04, P(≤0.5)≈0.6875（許容0.002）", abs(Us[:, 1].mean() - 0.4) < 0.002 and abs(Us[:, 1].var() - 0.04) < 0.002 and abs((Us[:, 1] <= 0.5).mean() - 0.6875) < 0.002, round(Us[:, 1].mean(), 4))
F1 = Phi(1)
pm = 1 - (3 * F1**2 - 2 * F1**3)
check(T, "例3 3F²(1−F)+F³=3F²−2F³, P(med>1)≈0.0675, 単独≈0.1587", close(F1, 0.8413) and close(pm, 0.0675) and close(1 - F1, 0.1587) and close(3 * F1**2 * (1 - F1) + F1**3, 3 * F1**2 - 2 * F1**3, 1e-12), round(pm, 5))
Ms = np.median(rng.standard_normal((400_000, 3)), axis=1)
check(T, "例3 シミュ P(med>1)（許容0.002）", abs((Ms > 1).mean() - 0.0675) < 0.002, round((Ms > 1).mean(), 4))
R = Us[:, 3] - Us[:, 0]
check(T, "範囲 R ~ Beta(n−1,2), E[R]=(n−1)/(n+1)（n=4: 0.6, KS p>0.001）", abs(R.mean() - 0.6) < 0.002 and stats.kstest(R[:50000], stats.beta(3, 2).cdf).pvalue > 0.001, round(R.mean(), 4))
xv, yv = sp.symbols("x y", positive=True)
fj = 4 * 3 * (yv - xv) ** 2
fR = sp.integrate(fj.subs(yv, xv + sp.Symbol("r", positive=True)), (xv, 0, 1 - sp.Symbol("r", positive=True)))
rr = sp.Symbol("r", positive=True)
check(T, "同時密度 n(n−1)(y−x)^{n−2} から R の密度 = 12r²(1−r)（Beta(3,2)）", sp.simplify(fR - 12 * rr**2 * (1 - rr)) == 0, sp.factor(fR))
check(T, "U(0,θ) 最大値 E=nθ/(n+1), 不偏 (n+1)/n·X_(n)（θ=1,n=10: 10/11≈0.9091, 範囲 9/11≈0.8182）", close(stats.beta(10, 1).mean(), 10 / 11, 1e-12) and close(10 / 11, 0.9091) and close(9 / 11, 0.8182), "")
sp_ = np.diff(np.sort(rng.exponential(1, (200_000, 5)), axis=1), axis=1)
check(T, "指数の間隔 X_(j+1)−X_(j) の平均 1/(n−j)（n=5, 許容0.01）", np.allclose(sp_.mean(axis=0), [1 / 4, 1 / 3, 1 / 2, 1], atol=0.01), sp_.mean(axis=0).round(3).tolist())
check(T, "係数 n!/((k−1)!(n−k)!) = n·C(n−1,k−1)（n=4,k=2 → 12）", all(factorial(nv) // (factorial(kv - 1) * factorial(nv - kv)) == nv * comb(nv - 1, kv - 1) for nv in range(1, 9) for kv in range(1, nv + 1)) and factorial(4) // (factorial(1) * factorial(2)) == 12, 12)

print()
print(f"合計（追加用語を含む）: OK {sum(results)} / NG {len(results) - sum(results)}")

# --- 追加用語（用語の見直し）: standard-error ---
import re as _re, yaml as _yaml
from pathlib import Path as _Path
T = "standard-error"
_rngS = np.random.default_rng(20261023 + 801)
def _clS(x, s):
    d = len(s.split(".")[1]) if "." in s else 0
    return abs(float(x) - float(s)) <= 0.5 * 10 ** (-d) + 1e-9
_fmS = _yaml.safe_load(_re.match(r"^---\n(.*?)\n---\n", (_Path(__file__).resolve().parent.parent / "content/ch08/standard-error.md").read_text(encoding="utf-8"), _re.S).group(1))
_s2, _n = sp.symbols("sigma2 n", positive=True)
check(T, "V[X̄]=(1/n²)·nσ²=σ²/n（sympy）", sp.simplify(_n * _s2 / _n**2 - _s2 / _n) == 0, "")
_t24 = stats.t.ppf(0.975, 24)
check(T, "例1 SE=1000/√25=200、t_0.025(24)=2.064、区間 [4587.2, 5412.8]、100日で 100",
      1000 / sqrt(25) == 200 and _clS(_t24, "2.064") and _clS(5000 - 2.064 * 200, "4587.2") and _clS(5000 + 2.064 * 200, "5412.8") and 1000 / sqrt(100) == 100, _t24)
check(T, "例2 比率 SE √(0.3·0.7/400)≈0.0229", 120 / 400 == 0.3 and _clS(sqrt(0.3 * 0.7 / 400), "0.0229"), sqrt(0.21 / 400))
check(T, "例2 平均の差 SE √(100/25+144/36)=√8≈2.828", 100 / 25 + 144 / 36 == 8 and _clS(sqrt(8), "2.828"), sqrt(8))
_lam = sp.Symbol("lambda", positive=True); _xx = sp.Symbol("x", integer=True, nonnegative=True)
_I = -sp.diff(_xx * sp.log(_lam) - _lam, _lam, 2).subs(_xx, _lam)
check(T, "例2 ポアソン I(λ)=1/λ、SE=√(λ/n)、n=50, x̄=4 → 0.2828", sp.simplify(_I - 1 / _lam) == 0 and _clS(sqrt(4 / 50), "0.2828"), sqrt(0.08))
# シミュレーション：標本平均の SD ≈ σ/√n
_xb = _rngS.normal(5000, 1000, (100000, 25)).mean(1)
check(T, "シミュレーション：n=25, σ=1000 の X̄ の標準偏差≈200（許容 2）", abs(_xb.std() - 200) < 2, round(_xb.std(), 2))
_ph = _rngS.binomial(400, 0.3, 200000) / 400
check(T, "シミュレーション：p̂ の標準偏差≈0.0229（許容 0.0003）", abs(_ph.std() - sqrt(0.21 / 400)) < 3e-4, round(_ph.std(), 5))
check(T, "n を4倍で SE 半分、2倍で 1/√2≈0.707倍", sqrt(1 / 4) == 0.5 and _clS(1 / sqrt(2), "0.707"), "")
_cS = _fmS["charts"][0]["params"]
check(T, "chart：σ=10 の σ/√n（n=1,4,…,100）", _cS["x"] == [k * k for k in range(1, 11)] and _cS["series"][0]["y"] == [round(10 / k, 4) for k in range(1, 11)], _cS["series"][0]["y"])

# --- 統合候補からの論点の追記: fisher-information ---
_lam, _xs = sp.symbols("lam xs", positive=True)
_n8 = sp.Symbol("n", positive=True)
_ellP = _xs * sp.log(_lam) - _n8 * _lam  # ポアソン対数尤度（xs=Σx_i、定数省略）
_J = -sp.diff(_ellP, _lam, 2)
_lhat = _xs / _n8
check("fisher-information", "ポアソン：J(λ̂)=Σx/λ̂²=n/x̄=nI(λ̂)（sympy）", sp.simplify(_J.subs(_lam, _lhat) - _n8 ** 2 / _xs) == 0 and sp.simplify(_n8 * (1 / _lam) - _J.subs(_xs, _n8 * _lam)) == 0, _J)

print()
print(f"合計（追加用語を含む）: OK {sum(results)} / NG {len(results) - sum(results)}")
