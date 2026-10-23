"""第15章（確率過程の基礎）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb, factorial, exp
from scipy import stats
from scipy.integrate import quad

q = Q(15)

# --- 確率過程：周辺が同じでも共分散が違う ---
q.check("q15-stochastic-process-1", min(1, 4) != np.sqrt(1 * 4), "W と √t Z は周辺同じで共分散 1 と 2（誤り）")

# --- 計数過程 ---
v = stats.poisson.sf(2, 3)
q.num("q15-counting-process-1", v)
q.check("q15-counting-process-1", abs(stats.gamma.cdf(2, 3, scale=1 / 1.5) - v) < 1e-12, "ガンマ分布の cdf と一致")

# --- ポアソン過程 ---
q.num("q15-poisson-process-1", stats.poisson.pmf(4, 6))
lam = 1.7  # λ に依らないことも確認
cond = stats.poisson.pmf(2, lam) * stats.poisson.pmf(4, 3 * lam) / stats.poisson.pmf(6, 4 * lam)
q.num("q15-poisson-process-2", cond)
q.check("q15-poisson-process-2", abs(cond - stats.binom.pmf(2, 6, 0.25)) < 1e-12, "Bin(6,1/4) と一致（λ によらない）")
rng = np.random.default_rng(0)
n2 = rng.poisson(6, 400000); n5 = n2 + rng.poisson(9, 400000)
cov = np.cov(n2, n5)[0, 1]
q.check("q15-poisson-process-3", abs(cov - 6) < 0.1 and all(abs(w - 6) > 1 for w in [15, 0, 9]), "Cov=λmin=6（シミュレーションでも）")

# --- 到着時間間隔 ---
q.num("q15-interarrival-time-1", stats.expon.sf(0.5, scale=1 / 4))
E = stats.expon(scale=1 / 2)
c = E.sf(25 / 60) / E.sf(10 / 60)
q.check("q15-interarrival-time-2", abs(c - np.exp(-0.5)) < 1e-12 and abs(round(c, 3) - 0.607) < 1e-9
        and all(abs(w - c) > 0.1 for w in [np.exp(-5 / 6), np.exp(-1 / 3), 1 - np.exp(-0.5)]), "無記憶性で e^{-0.5}、誤答は異なる")
val = stats.gamma.cdf(1, 2, scale=1 / 2)
q.num("q15-interarrival-time-3", val)
q.check("q15-interarrival-time-3", abs(quad(lambda t: 4 * t * np.exp(-2 * t), 0, 1)[0] - val) < 1e-10, "密度の積分と一致")

# --- 合成・間引き ---
q.num("q15-superposition-thinning-poisson-1", (2 / 5) ** 2)
q.num("q15-superposition-thinning-poisson-2", np.exp(-3))
q.check("q15-superposition-thinning-poisson-2", abs(sum(stats.poisson.pmf(n, 10) * 0.7 ** n for n in range(200)) - np.exp(-3)) < 1e-12,
        "来店数で条件づけた計算と一致")

# --- ランダムウォーク ---
q.num("q15-random-walk-1", stats.binom.pmf(4, 6, 0.6))
V = 100 * (1 - (0.55 - 0.45) ** 2)
q.check("q15-random-walk-2", abs(V - 99) < 1e-9 and abs(V - 4 * 100 * 0.55 * 0.45) < 1e-9
        and all(abs(w - V) > 0.5 for w in [24.75, 100, 9.95]), "V=99、誤答は異なる")

# --- 破産問題（吸収連鎖で直接解く） ---
def ruin(p, N):
    A = np.eye(N + 1); b = np.zeros(N + 1); Dm = np.eye(N + 1); db = np.zeros(N + 1)
    for i in range(1, N):
        A[i, i + 1] -= p; A[i, i - 1] -= 1 - p
        Dm[i, i + 1] -= p; Dm[i, i - 1] -= 1 - p; db[i] = 1
    b[N] = 1
    return np.linalg.solve(A, b), np.linalg.solve(Dm, db)
Pw, Dw = ruin(0.5, 10)
q.num("q15-gambler-ruin-problem-1", Dw[3])
Pw2, _ = ruin(0.4, 10)
q.num("q15-gambler-ruin-problem-2", Pw2[5])
Pbig, _ = ruin(0.6, 200)
q.num("q15-gambler-ruin-problem-3", 1 - Pbig[3])

# --- ブラウン運動 ---
q.num("q15-brownian-motion-1", stats.norm.sf(3, scale=2))
Cm = np.array([[1, 1], [1, 3]])  # Cov(W1,W3)
a = np.array([-2, 1])
q.num("q15-brownian-motion-2", a @ Cm @ a)

# --- マルチンゲール ---
S, n = sp.symbols("S n")
EM = sp.Rational(1, 2) * ((S + 1) ** 2 - (n + 1)) + sp.Rational(1, 2) * ((S - 1) ** 2 - (n + 1))
q.check("q15-martingale-1", sp.simplify(EM - (S ** 2 - n)) == 0, "E[M_{n+1}|F_n]=M_n（sympy）")

# --- 出生死滅 ---
w = np.array([1, 2 / 1, 2 * 2 / (1 * 2)])
q.num("q15-birth-death-process-1", (w / w.sum())[2])
# 生成行列で確認
G = np.array([[-2, 2, 0], [1, -3, 2], [0, 2, -2]], dtype=float)
u, s_, vt = np.linalg.svd(G.T); pi = np.abs(vt[-1]); pi /= pi.sum()
q.check("q15-birth-death-process-1", np.allclose(pi, w / w.sum()), "πG=0 の解と一致")
K = 60
wt = np.array([np.prod([2 / (k + 1) for k in range(m)]) / 1 ** m for m in range(K)])
q.num("q15-birth-death-process-2", 1 / wt.sum())
q.check("q15-birth-death-process-2", np.allclose(wt / wt.sum(), stats.poisson.pmf(np.arange(K), 2)), "Po(2) と一致")

# --- ユール過程（純出生）：前進方程式を数値で解く ---
from scipy.linalg import expm
Kmax = 80
Gy = np.zeros((Kmax, Kmax))
for i in range(Kmax - 1):
    nn = i + 1
    Gy[i, i] = -0.5 * nn; Gy[i, i + 1] = 0.5 * nn
Pt = expm(2 * Gy)[0]
q.num("q15-pure-birth-process-1", Pt[2])
q.check("q15-pure-birth-process-1", abs(Pt[2] - np.exp(-1) * (1 - np.exp(-1)) ** 2) < 1e-9, "幾何分布の式と一致")

# --- M/D/1 ---
lq = (0.5 ** 2 * 0 + 0.5 ** 2) / (2 * (1 - 0.5))
q.num("q15-queueing-theory-1", lq)
q.check("q15-queueing-theory-1", abs((0.5 ** 2 * 1 + 0.25) / (2 * 0.5) - 0.5 ** 2 / 0.5) < 1e-12, "M/M/1 なら 0.5（ρ²/(1-ρ)）")

# --- M/M/1 ---
lam, mu = 4, 5
q.num("q15-mm1-queue-1", (lam / mu) / (mu - lam) * 60)
q.num("q15-mm1-queue-2", sum((1 - 0.8) * 0.8 ** k for k in range(3, 2000)))
r = (1 / (5 - 4.5)) / (1 / (5 - 4))
q.check("q15-mm1-queue-3", abs(r - 2) < 1e-12 and all(abs(w - r) > 0.1 for w in [1.125, 1.5, 4])
        and abs(4.5 * 2 / (4 * 1) - 2.25) < 1e-12, "W は2倍、L は2.25倍")

# --- 利用率 ---
q.num("q15-traffic-intensity-1", 8 * 6 / 60)
rho = 12 / (3 * 5)
q.check("q15-traffic-intensity-2", abs(rho - 0.8) < 1e-12 and all(abs(w - rho) > 0.1 for w in [2.4, 0.267, 1.25]), "ρ=0.8")

# --- リトル ---
q.num("q15-littles-law-1", 30 / 2)
# M/M/1 で L=λW を確認（式の確認）
q.check("q15-littles-law-1", abs((0.8 / 0.2) - 4 * (1 / (5 - 4))) < 1e-12, "M/M/1 で L=λW")

# --- 複合ポアソン ---
sim = np.array([rng.normal(2, 1, k).sum() for k in rng.poisson(3, 100000)])
q.num("q15-compound-poisson-process-1", 3 * (1 + 4))
q.check("q15-compound-poisson-process-1", abs(sim.var() - 15) < 0.3, "シミュレーションでも分散≈15")

# --- 増分（非斉次ポアソン） ---
lam_t = lambda t: 1 + t
m1 = quad(lam_t, 0, 1)[0]; m2 = quad(lam_t, 5, 6)[0]
q.check("q15-independent-stationary-increments-1", abs(m1 - m2) > 1, "非斉次では同じ長さでも増分の平均が異なる（定常でない）")

# --- M/M/2 ---
lam, mu, c_ = 2, 1.5, 2
a_ = lam / mu; rho_ = a_ / c_
pi0 = 1 / (sum(a_ ** k / factorial(k) for k in range(c_)) + a_ ** c_ / (factorial(c_) * (1 - rho_)))
Cerl = a_ ** c_ / (factorial(c_) * (1 - rho_)) * pi0
q.num("q15-mmc-queue-1", Cerl)
# 出生死滅の定常分布から直接 P(N>=2)
Kq = 400
wq = [1.0]
for nn in range(1, Kq):
    wq.append(wq[-1] * lam / (min(nn, c_) * mu))
wq = np.array(wq) / sum(wq)
q.check("q15-mmc-queue-1", abs(wq[c_:].sum() - Cerl) < 1e-9 and abs(pi0 - 0.2) < 1e-12, "定常分布から P(N≥2) と一致、π0=0.2")

# --- 反射原理 ---
val = 2 * stats.norm.sf(1)
q.num("q15-reflection-principle-1", val)
nsteps, npath = 2000, 20000
inc = rng.normal(0, np.sqrt(4 / nsteps), (npath, nsteps)).astype(np.float32)
mx = np.cumsum(inc, axis=1).max(axis=1)
q.check("q15-reflection-principle-1", abs((mx >= 2).mean() - val) < 0.02, "シミュレーション（離散化）でも ≈0.317")

# --- 導出問題（追加） ---
def _deriv15():
    import numpy as _N
    import sympy as _S
    p = _S.symbols("p", positive=True); i, N = _S.symbols("i N", integer=True, positive=True)
    qq = 1 - p; r = qq / p
    h = (1 - r ** i) / (1 - r ** N)
    resid = _S.expand(_S.powsimp(_S.expand_power_base(_S.numer(_S.together(p * h.subs(i, i + 1) + qq * h.subs(i, i - 1) - h)), force=True), force=True))
    ok = all(abs(float(resid.subs({p: pv, i: iv, N: 10}))) < 1e-12 for pv in (0.3, 0.4, 0.7) for iv in range(1, 10))
    ok &= _S.simplify(h.subs(i, 0)) == 0 and _S.simplify(h.subs(i, N) - 1) == 0
    # 線形方程式を直接解いた値と一致（p=0.4, N=10）
    pp, NN = 0.4, 10
    A = _N.zeros((NN + 1, NN + 1)); bb = _N.zeros(NN + 1)
    A[0, 0] = 1; A[NN, NN] = 1; bb[NN] = 1
    for k in range(1, NN):
        A[k, k] = 1; A[k, k + 1] = -pp; A[k, k - 1] = -(1 - pp)
    hv = _N.linalg.solve(A, bb)
    rr = (1 - pp) / pp
    ok &= _N.allclose(hv, [(1 - rr ** k) / (1 - rr ** NN) for k in range(NN + 1)])
    ok &= _S.limit(((1 - _S.Symbol("x") ** 3) / (1 - _S.Symbol("x") ** 10)), _S.Symbol("x"), 1) == _S.Rational(3, 10)
    q.check("q15-gambler-ruin-problem-4", bool(ok), "sympy: 漸化式と境界条件を満たす、連立方程式の解と一致、r→1 で i/N")
    rho = _S.symbols("rho", positive=True); lam, mu = _S.symbols("lambda mu", positive=True)
    n = _S.symbols("n", integer=True, nonnegative=True)
    ok2 = True
    for rv in (_S.Rational(1, 5), _S.Rational(1, 2), _S.Rational(4, 5)):   # ρ<1 の具体値で級数を厳密に計算
        ok2 &= _S.summation(rv ** n, (n, 0, _S.oo)) == 1 / (1 - rv)
        ok2 &= _S.summation(n * (1 - rv) * rv ** n, (n, 0, _S.oo)) == rv / (1 - rv)
    W = (rho / (1 - rho) / lam).subs(rho, lam / mu)
    ok2 &= _S.simplify(W - 1 / (mu - lam)) == 0
    # 数値：λ=4, μ=5 の生成行列から定常分布を解いて比較（状態を 400 で打ち切り）
    K = 400; Q = _N.zeros((K, K))
    for k in range(K - 1):
        Q[k, k + 1] = 4.0; Q[k + 1, k] = 5.0
    Q -= _N.diag(Q.sum(1))
    Aq = _N.vstack([Q.T, _N.ones(K)]); bq = _N.zeros(K + 1); bq[-1] = 1
    pv = _N.linalg.lstsq(Aq, bq, rcond=None)[0]
    ok2 &= _N.allclose(pv[:20], 0.2 * 0.8 ** _N.arange(20), atol=1e-10) and abs((_N.arange(K) * pv).sum() - 4) < 1e-6
    q.check("q15-mm1-queue-4", bool(ok2), "sympy: p0=1-ρ、L=ρ/(1-ρ)、W=1/(μ-λ)、生成行列の定常解と一致")
_deriv15()

# --- 4択の追加分 ---
_rng = np.random.default_rng(15)
_W2 = _rng.normal(0, np.sqrt(2), 400000); _W5 = _W2 + _rng.normal(0, np.sqrt(3), 400000)
_rho = 2 / np.sqrt(10); _sim = np.corrcoef(_W2, _W5)[0, 1]
_wrong = [2 / 5, np.sqrt(3 / 5), 3 / 5]
q.check("q15-brownian-motion-3",
        round(_rho, 3) == 0.632 and abs(_sim - _rho) < 0.005 and [round(w, 3) for w in _wrong] == [0.4, 0.775, 0.6]
        and all(abs(w - _rho) > 0.02 for w in _wrong),
        f"ρ=√(2/5)={_rho:.4f}（シミュレーション {_sim:.4f}）。誤答 {[round(w, 4) for w in _wrong]}")
_ok = True
for _r in (0.5, 3.0):
    _w = np.array([_r ** n / factorial(n) for n in range(60)])
    # 出生死滅の積公式 π_n ∝ Π λ/(kμ) を直接計算（λ=_r, μ=1）
    _pr = np.cumprod([1.0] + [_r / k for k in range(1, 60)]); _pr /= _pr.sum()
    _ok &= np.allclose(_pr, stats.poisson.pmf(range(60), _r))
    _ok &= not np.allclose(_pr, stats.poisson.pmf(range(60), 1 / _r))
    if _r < 1:
        _ok &= not np.allclose(_pr, (1 - _r) * _r ** np.arange(60))
# ρ=3 でも正規化定数が有限（定常分布が存在）
_ok &= np.isfinite(sum(3.0 ** n / factorial(n) for n in range(150)))
# 平衡方程式 λπ_{n-1}=nμπ_n の確認（ρ=3）
_pp = stats.poisson.pmf(np.arange(1, 30), 3.0); _pm = stats.poisson.pmf(np.arange(0, 29), 3.0)
_ok &= np.allclose(3.0 * _pm, np.arange(1, 30) * _pp)
q.check("q15-birth-death-process-3", bool(_ok),
        "ρ=0.5, 3 で積公式の定常分布＝Poisson(ρ)、Poisson(1/ρ)・幾何分布とは不一致、ρ≥1 でも存在")

q.done()
