"""第27章（時系列解析）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, linalg

q = Q(27)
rng = np.random.default_rng(27)


def acf_ma(theta, h):
    """MA(q) の自己共分散（sigma^2=1）。theta は [1, θ1, ..., θq]。"""
    th = np.asarray(theta, float)
    if h >= len(th):
        return 0.0
    return float(np.sum(th[: len(th) - h] * th[h:]))


def arma_acov(phi, theta, nlag, sigma2=1.0, nterm=2000):
    """ARMA の自己共分散を MA(∞) 係数から数値計算。"""
    psi = np.zeros(nterm)
    for j in range(nterm):
        v = theta[j] if j < len(theta) else 0.0
        for i, ph in enumerate(phi, start=1):
            if j - i >= 0:
                v += ph * psi[j - i]
        psi[j] = v
    return [sigma2 * float(np.sum(psi[: nterm - h] * psi[h:])) for h in range(nlag + 1)]


def pacf_from_rho(rho, k):
    R = linalg.toeplitz([1.0] + list(rho[: k - 1]))
    return float(np.linalg.solve(R, rho[:k])[-1])


# --- 時系列 ---
y = np.array([10, 14, 12, 8, 11, 15.0])
w = np.array([0.5, 1, 1, 1, 0.5]) / 4
q.num("q27-time-series-1", float(w @ y[0:5]))
q.concept("q27-time-series-2", "乗法モデルの使いどころ（数値なし）")

# --- 弱定常 ---
Z = rng.normal(size=200000)
q.check("q27-weak-stationarity-1", abs(np.mean(Z)) < 0.01 and abs(np.var(Z) - 1) < 0.01,
        "X_t=Z の平均0・自己共分散1が t によらない（弱定常＝正しい）")
sig2 = 1.5
q.num("q27-weak-stationarity-2", 10 * sig2 - 6 * sig2)
eps = rng.normal(scale=np.sqrt(sig2), size=(200000, 10))
X = eps.cumsum(axis=1)
q.check("q27-weak-stationarity-2", abs(np.var(X[:, 9]) - np.var(X[:, 9] - X[:, 3]) - 6) < 0.15,
        "シミュレーションで差 ≈ 6")
A, B, lam, t, h, s2 = sp.symbols("A B lambda t h sigma2")
expr = sp.expand((A * sp.cos(lam * t) + B * sp.sin(lam * t)) * (A * sp.cos(lam * (t + h)) + B * sp.sin(lam * (t + h))))
cov = expr.subs({A * B: 0}).subs({A**2: s2, B**2: s2})
cov = sp.simplify(cov)
q.check("q27-weak-stationarity-3", sp.simplify(cov - s2 * sp.cos(lam * h)) == 0
        and sp.simplify(cov - s2 * sp.cos(lam * (2 * t + h))) != 0
        and sp.simplify(cov - 2 * s2 * sp.cos(lam * h)) != 0, "共分散 = σ²cos(λh)、誤答とは一致しない")

# --- 強定常 ---
q.check("q27-strict-stationarity-1", np.isinf(stats.t(2).var()) and stats.t(2).mean() == 0,
        "t(2) は分散無限（弱定常でない）、i.i.d. なので強定常＝正しい")

# --- 自己共分散・自己相関 ---
x = np.array([1, 3, 4, 6, 5, 7.0]); d = x - x.mean()
q.num("q27-autocovariance-autocorrelation-1", (d[:-1] * d[1:]).sum() / (d**2).sum())
th = [1, 0, -1]
g0, g1, g2 = acf_ma(th, 0), acf_ma(th, 1), acf_ma(th, 2)
q.check("q27-autocovariance-autocorrelation-2", (g1 / g0, g2 / g0) == (0.0, -0.5)
        and g2 != -1 * g0 and g2 / g0 != 0.5, "(ρ1,ρ2)=(0,-0.5)")
eigs = [np.linalg.eigvalsh(linalg.toeplitz([1, 0.8] + [0] * (k - 2))).min() for k in range(2, 8)]
q.check("q27-autocovariance-autocorrelation-3", min(eigs) < 0 and abs(1.6 - 0.6 * 3 + 0.2) < 1e-12,
        "自己共分散行列が負の固有値をもつ（非負定値でない＝存在しない＝誤り）")

# --- コレログラム ---
q.num("q27-correlogram-1", 1.96 * np.sqrt((1 + 2 * 0.4**2) / 400))
q.check("q27-correlogram-2", 0.3**2 < 0.1, "AR(1) φ=0.3 なら ρ2=0.09 で急減（ゆっくり減るのは単位根）")

# --- 偏自己相関 ---
q.num("q27-partial-autocorrelation-1", pacf_from_rho([0.6, 0.5], 2))
p22 = pacf_from_rho([0.7, 0.49], 2)
q.check("q27-partial-autocorrelation-2", abs(p22) < 1e-12 and abs(p22 - 0.49) > 0.1, "φ22=0（0.49 ではない＝誤り）")
r = arma_acov([], [1, 0.5], 3)
rho = [r[1] / r[0], r[2] / r[0]]
v = pacf_from_rho(rho, 2)
q.num("q27-partial-autocorrelation-3", v)
th_ = 0.5
q.check("q27-partial-autocorrelation-3", abs(v - (-(-th_) ** 2 * (1 - th_**2) / (1 - th_**6))) < 1e-12, "MA(1) の PACF 公式と一致")

# --- ホワイトノイズ ---
rr = np.array([0.15, -0.10, 0.08]); n = 100
Qlb = n * (n + 2) * np.sum(rr**2 / (n - np.arange(1, 4)))
q.num("q27-white-noise-1", Qlb)
q.check("q27-white-noise-1", Qlb < stats.chi2.ppf(0.95, 3) and abs(stats.chi2.ppf(0.95, 3) - 7.81) < 0.01, "7.81 より小さく棄却しない")
q.concept("q27-white-noise-2", "WN は無相関まで（独立は要求しない）")
e = rng.normal(size=1_000_001)
Xp = e[1:] * e[:-1]
c1 = np.mean(Xp[1:] * Xp[:-1])
c2 = np.cov(Xp[1:] ** 2, Xp[:-1] ** 2)[0, 1]
q.check("q27-white-noise-3", abs(np.var(Xp) - 1) < 0.02 and abs(c1) < 0.01 and abs(c2 - 2) < 0.2,
        "分散1・ラグ1共分散0・2乗の共分散≈2（無相関だが独立でない）")

# --- AR ---
q.num("q27-ar-model-1", 4 / (1 - 0.36))
q.check("q27-ar-model-1", abs(arma_acov([0.6], [1], 0, sigma2=4)[0] - 6.25) < 1e-9 and abs(4 / 0.4 - 6.25) > 1, "MA(∞) 和でも 6.25")
ac = arma_acov([0.5, 0.3], [1], 2)
q.num("q27-ar-model-2", ac[2] / ac[0])
q.num("q27-ar-model-3", ac[0])
q.check("q27-ar-model-3", abs(1 / (1 - 0.25 - 0.09) - ac[0]) > 0.5, "誤った公式 1/0.66 とは一致しない")

# --- MA ---
q.num("q27-ma-model-1", -0.6 / 1.36)
th = [1, 0.5, -0.3]
g1 = acf_ma(th, 1)
q.check("q27-ma-model-2", abs(g1 - 0.35) < 1e-12 and all(abs(g1 - c) > 1e-9 for c in [0.5, 0.65, -0.3])
        and abs(acf_ma(th, 2) + 0.3) < 1e-12, "γ(1)=0.35、γ(2)=-0.3")
T = sp.Symbol("theta", real=True)
f = T / (1 + T**2)
crit = sp.solve(sp.diff(f, T), T)
q.check("q27-ma-model-3", sorted(crit) == [-1, 1] and f.subs(T, 1) == sp.Rational(1, 2)
        and sp.limit(f, T, sp.oo) == 0, "|θ/(1+θ²)| の最大は θ=±1 で 1/2（正しい）")

# --- ARMA ---
ac = arma_acov([0.5], [1, 0.4], 2)
q.num("q27-arma-model-1", ac[1] / ac[0])
ac = arma_acov([0.5], [1, -0.5], 3)
q.check("q27-arma-model-2", abs(ac[0] - 1) < 1e-9 and abs(ac[1]) < 1e-9 and abs(ac[2]) < 1e-9, "共通根で約分＝WN")
ph, tt, s = sp.symbols("phi theta sigma2")
g0s, g1s = sp.symbols("g0 g1")
sol = sp.solve([sp.Eq(g0s, ph * g1s + s + tt * (ph + tt) * s), sp.Eq(g1s, ph * g0s + tt * s)], [g0s, g1s])
q.check("q27-arma-model-3", sp.simplify(sol[g0s] - (1 + 2 * ph * tt + tt**2) * s / (1 - ph**2)) == 0
        and abs(arma_acov([0.6], [1, 0.3], 0)[0] - (1 + 2 * 0.18 + 0.09) / (1 - 0.36)) < 1e-9, "連立方程式の解と数値例が一致")

# --- ARIMA ---
roots = np.roots(np.polymul([-0.4, 1], [-1, 1]))  # (1-0.4z)(1-z) を高次から
q.check("q27-arima-model-1", np.any(np.isclose(np.abs(roots), 1.0)) and np.any(np.isclose(roots, 2.5)),
        "AR 多項式に単位根 z=1 と z=2.5 → ARIMA(1,1,1)")
q.num("q27-arima-model-2", 0.3 * 50 + 0.7 * 46)
q.check("q27-arima-model-2", abs(50 - 0.7 * (50 - 46) - 47.2) < 1e-12, "直接計算でも 47.2")
q.num("q27-arima-model-3", 20 + 4 * 0.5 + stats.norm.ppf(0.975) * 2)

# --- ユール・ウォーカー ---
r1, r2 = 0.6, 0.45
p1, p2 = np.linalg.solve([[1, r1], [r1, 1]], [r1, r2])
q.num("q27-yule-walker-equations-1", p1)
q.num("q27-yule-walker-equations-2", 5 * (1 - p1 * r1 - p2 * r2))
q.check("q27-yule-walker-equations-3", abs(np.linalg.solve([[1.0]], [0.37])[0] - 0.37) < 1e-12, "AR(1) の解は r1（正しい）")

# --- 定常条件 ---
def ar2_stationary(p1, p2):
    rts = np.roots([-p2, -p1, 1])
    return bool(np.all(np.abs(rts) > 1))
res = [ar2_stationary(*c) for c in [(0.5, 0.4), (0.7, 0.4), (-0.3, 0.8), (1.2, -1.1)]]
q.check("q27-stationarity-condition-1", res == [True, False, False, False], "定常なのは (0.5,0.4) だけ")
rts = np.roots([0.5, -1.2, 1])
q.num("q27-stationarity-condition-2", np.abs(rts[0]))
q.check("q27-stationarity-condition-2", np.allclose(np.abs(rts), np.sqrt(2)) and np.all(np.iscomplex(rts)), "複素共役根 |z|=√2")
ac = [acf_ma([1, 3], h) for h in range(3)]
q.check("q27-stationarity-condition-3", ac == [10.0, 3.0, 0.0], "MA は γ(h) が t によらず定常（誤り）")

# --- 反転可能性 ---
q.num("q27-invertibility-1", 2.5**2 * 1)
q.check("q27-invertibility-1", np.isclose((1 + 0.16) * 6.25, 1 + 6.25) and np.isclose(0.4 * 6.25, 2.5), "自己共分散が一致")
def ma2_invertible(t1, t2):
    return bool(np.all(np.abs(np.roots([t2, t1, 1])) > 1 + 1e-12))
res = [ma2_invertible(*c) for c in [(-0.5, 0.06), (-1.5, 0.5), (2.5, 1), (0.2, -0.8)]]
q.check("q27-invertibility-2", res == [True, False, False, False], "反転可能なのは1つめだけ")

# --- 単位根 ---
tau = -0.08 / 0.025
q.check("q27-unit-root-1", np.isclose(tau, -3.2) and tau < -2.86 and tau > -3.41, "τ=-3.2 < -2.86 で棄却")
q.num("q27-unit-root-2", 16 / np.sqrt(16 * 25))
poly = sp.Poly(1 - sp.Rational(13, 10) * sp.Symbol("z") + sp.Rational(3, 10) * sp.Symbol("z") ** 2)
zz = sp.Symbol("z")
q.check("q27-unit-root-3", poly.eval(1) == 0 and sp.expand((1 - zz) * (1 - sp.Rational(3, 10) * zz) - poly.as_expr()) == 0,
        "φ(1)=0、φ(z)=(1-z)(1-0.3z)（正しい）")

# --- スペクトル ---
fAR = lambda lam, phi: 1 / (2 * np.pi * (1 - 2 * phi * np.cos(lam) + phi**2))
q.num("q27-spectrum-1", fAR(0, 0.5) / fAR(np.pi, 0.5))
grid = np.linspace(0, np.pi, 1001)
q.check("q27-spectrum-2", np.argmax(fAR(grid, -0.8)) == len(grid) - 1, "φ=-0.8 では λ=π で最大（誤り）")

# --- 季節調整 ---
raw = np.array([3, -1, -4, 4.0]); S = raw - raw.mean()
q.num("q27-seasonal-adjustment-1", 120 - S[1])
q.concept("q27-seasonal-adjustment-2", "SARIMA の次数の読み取り（数値なし）")

# --- 予測 ---
mu = 3 / 0.3
q.num("q27-forecasting-1", mu + 0.7**3 * (14 - mu))
xh = 14.0
for _ in range(3):
    xh = 3 + 0.7 * xh
q.check("q27-forecasting-1", abs(xh - 11.372) < 1e-9, "逐次代入でも 11.372")
q.num("q27-forecasting-2", 2 * sum(0.7 ** (2 * j) for j in range(3)))
q.check("q27-forecasting-2", np.isclose(2 * (1 - 0.7**6) / (1 - 0.49), 3.4602), "等比数列の和の公式と一致")

# --- 状態空間モデル ---
Pp = 2 + 1; K = Pp / (Pp + 3)
q.num("q27-state-space-model-1", 10 + K * (14 - 10))
qq = 2.0
Pbar = (qq + np.sqrt(qq**2 + 4 * qq)) / 2
q.num("q27-state-space-model-2", Pbar / (Pbar + 1))
P = 5.0
for _ in range(200):  # σε²=1, ση²=2 で反復して定常値に収束
    Ppred = P + 2; Kt = Ppred / (Ppred + 1); P = (1 - Kt) * Ppred
q.check("q27-state-space-model-2", abs(Kt - Pbar / (Pbar + 1)) < 1e-12 and abs(Ppred - Pbar) < 1e-12, "反復の極限と一致")

# --- 導出問題（追加）---
import sympy as _sp
_c, _ph, _s2, _mu, _g0 = _sp.symbols("c phi sigma2 mu g0")
_mu_sol = _sp.solve(_sp.Eq(_mu, _c + _ph * _mu), _mu)[0]
_g0_sol = _sp.solve(_sp.Eq(_g0, _ph**2 * _g0 + _s2), _g0)[0]
_ok = _sp.simplify(_mu_sol - _c / (1 - _ph)) == 0 and _sp.simplify(_g0_sol - _s2 / (1 - _ph**2)) == 0
_vals = {_c: 3, _ph: _sp.Rational(-1, 2), _s2: 3}
_m, _v = _mu_sol.subs(_vals), _g0_sol.subs(_vals)
_g2 = (_sp.Rational(-1, 2))**2 * _v
# MA(∞) 表現の和でも同じ γ(0)、γ(2)
_psi = (-0.5) ** np.arange(200)
_ok2 = abs(3 * np.sum(_psi**2) - 4) < 1e-9 and abs(3 * np.sum(_psi[:-2] * _psi[2:]) - 1) < 1e-9
q.check("q27-ar-model-4", _ok and _m == 2 and _v == 4 and _g2 == 1 and _ok2, "μ=c/(1-φ)、γ(0)=σ²/(1-φ²)、数値 μ=2, γ(0)=4, γ(2)=1（MA(∞) の和とも一致）")

_th = _sp.symbols("theta")
_roots = sorted(_sp.solve(_sp.Eq(_th / (1 + _th**2), _sp.Rational(-3, 10)), _th))
_inv = [r for r in _roots if abs(r) < 1]
_s2hat = 2 / (1 + _inv[0]**2)
# γ(h) を係数の積の和で直接計算（θ=-1/3, σ²=1.8）
_coef = np.array([1, -1 / 3]); _g = [1.8 * np.sum(_coef[: 2 - h] * _coef[h:]) if h < 2 else 0.0 for h in range(3)]
q.check("q27-ma-model-4", _roots == [-3, _sp.Rational(-1, 3)] and _inv == [_sp.Rational(-1, 3)] and _s2hat == _sp.Rational(9, 5)
        and abs(_g[0] - 2) < 1e-12 and abs(_g[1] / _g[0] + 0.3) < 1e-12 and _g[2] == 0,
        "解は -3 と -1/3、反転可能な θ=-1/3、σ²=1.8 で γ(0)=2、ρ(1)=-0.3")

# --- 4択の追加（偏自己相関によるモデルの見分け・ユール・ウォーカー）---
import numpy as _np
_th = [1.0, 1.0, 0.5]
_g = [sum(_th[i] * _th[i + h] for i in range(3 - h)) for h in range(3)]
_rho = [1.0, _g[1] / _g[0], _g[2] / _g[0]] + [0.0] * 10
def _pacf(rho, K):
    out = []
    for k in range(1, K + 1):
        R = _np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        out.append(_np.linalg.solve(R, rho[1:k + 1])[-1])
    return _np.array(out)
_pa = _pacf(_rho, 10)
_ok = _np.allclose(_np.round(_rho[1:6], 3), [0.667, 0.222, 0, 0, 0]) and _np.allclose(_np.round(_pa[:5], 3), [0.667, -0.4, 0.127, 0.067, -0.127])
_ok &= all(abs(v) > 1e-4 for v in _pa[5:10]) and abs(_pa[9]) < abs(_pa[0])        # 偏自己相関は切れない
# AR(2)（PACF がラグ3以降0）・AR(1)（ρ(3)≠0）・ホワイトノイズ（ρ(1)=0）とは合わない
_ok &= abs(_pa[2]) > 0.05 and abs(_rho[1] ** 3) > 0.05 and _rho[1] != 0
q.check("q27-partial-autocorrelation-4", _ok, f"MA(2) θ=(1,0.5) の ACF {_np.round(_rho[1:6],3)}、PACF {_np.round(_pa[:6],3)}")
_r1, _r2 = 0.5, 0.4
_phi = _np.linalg.solve(_np.array([[1, _r1], [_r1, 1]]), _np.array([_r1, _r2]))
_ok = _np.allclose(_np.round(_phi, 2), [0.40, 0.20])
_wr = [(_r1, _r2), (_r1, _r2 - _r1 ** 2), (_phi[1], _phi[0])]
_ok &= [tuple(_np.round(w, 2)) for w in _wr] == [(0.5, 0.4), (0.5, 0.15), (0.2, 0.4)]
_ok &= all(not _np.allclose(_np.round(w, 2), [0.40, 0.20]) for w in _wr)
q.check("q27-yule-walker-equations-4", _ok, f"φ̂ = {_np.round(_phi, 4)}、誤答 (0.50,0.40)・(0.50,0.15)・(0.20,0.40) と異なる")

q.done()
