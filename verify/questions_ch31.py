"""第31章（ベイズ法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb
from scipy import stats, integrate, optimize

q = Q(31)
rng = np.random.default_rng(31)

# --- 事前分布 ---
a = 0.3 * 10
q.num("q31-prior-distribution-1", a)
grid = np.linspace(0.001, 0.999, 999)
prior = np.where(grid < 0.5, 1.0, 0.0)
post = prior * grid**80 * (1 - grid) ** 20
q.check("q31-prior-distribution-2", post[grid >= 0.5].sum() == 0, "事前0の範囲は事後も0（尤度が θ=0.8 付近に集中しても）")

# --- 事後分布 ---
L = np.array([0.2**3 * 0.8, 0.5**4]); P = L * 0.5 / np.sum(L * 0.5)
q.num("q31-posterior-distribution-1", P[1])
lam, al, be, n_, s_ = sp.symbols("lambda alpha beta n s", positive=True)
ker = sp.simplify(lam**n_ * sp.exp(-lam * s_) * lam ** (al - 1) * sp.exp(-be * lam))
q.check("q31-posterior-distribution-2", sp.simplify(ker / (lam ** (al + n_ - 1) * sp.exp(-(be + s_) * lam))) == 1, "核が Ga(α+n, β+Σx)")
xs = rng.binomial(1, 0.3, 20)
ab = np.array([1.0, 1.0])
for x in xs:
    ab += [x, 1 - x]
q.check("q31-posterior-distribution-3", np.allclose(ab, [1 + xs.sum(), 1 + 20 - xs.sum()]), "逐次更新＝一括更新")

# --- 共役 ---
q.concept("q31-conjugate-prior-1", "ポアソンの共役はガンマ（定義）")
x = np.array([1, -2, 1.0])
q.num("q31-conjugate-prior-2", (2 + len(x) / 2) / (3 + np.sum(x**2) / 2))
# 数値積分でも確認
tau = np.linspace(1e-4, 20, 200001)
dens = stats.gamma.pdf(tau, 2, scale=1 / 3) * np.prod([stats.norm.pdf(xi, 0, 1 / np.sqrt(tau)) for xi in x], axis=0)
q.check("q31-conjugate-prior-2", abs(np.sum(tau * dens) / np.sum(dens) - 3.5 / 6) < 1e-4, "数値積分の事後平均と一致")
g = np.array([3, 0, 5, 2])
q.num("q31-conjugate-prior-3", (2 + 4) / (2 + 4 + 2 + g.sum()))

# --- ベータ二項 ---
post = stats.beta(9, 6)
q.num("q31-beta-binomial-model-1", post.mean())
mode = optimize.minimize_scalar(lambda t: -post.pdf(t), bounds=(0.01, 0.99), method="bounded").x
q.num("q31-beta-binomial-model-2", mode)
w = 10 / 15
q.num("q31-beta-binomial-model-3", w)
q.check("q31-beta-binomial-model-3", np.isclose(w * 0.7 + (1 - w) * 0.4, 0.6), "加重平均で 0.6")

# --- ガンマ・ポアソン ---
gp = stats.gamma(18, scale=1 / 6)
q.num("q31-gamma-poisson-model-1", gp.mean())
q.num("q31-gamma-poisson-model-2", gp.std())
modes = optimize.minimize_scalar(lambda t: -gp.pdf(t), bounds=(0.01, 10), method="bounded").x
q.check("q31-gamma-poisson-model-3", modes < gp.mean() and np.isclose(gp.mean() - modes, 1 / 6, atol=1e-4), "MAP < 平均、差 1/β'")

# --- 正規・正規 ---
k0, kx = 1.0, 16 / 4
mun = (k0 * 10 + kx * 10.5) / (k0 + kx); tn2 = 1 / (k0 + kx)
q.num("q31-normal-normal-model-1", mun)
q.num("q31-normal-normal-model-2", mun + stats.norm.ppf(0.975) * np.sqrt(tn2))
nmin = next(n for n in range(1, 1000) if 1 / np.sqrt(1 + n / 4) <= 0.25)
q.num("q31-normal-normal-model-3", nmin)

# --- 事後平均・MAP ---
d = optimize.minimize_scalar(lambda d: integrate.quad(lambda t: abs(t - d) * stats.beta.pdf(t, 3, 9), 0, 1, points=[d])[0],
                             bounds=(0, 1), method="bounded").x
q.check("q31-posterior-mean-map-1", abs(d - stats.beta(3, 9).median()) < 1e-3 and abs(d - stats.beta(3, 9).mean()) > 0.005,
        "絶対損失の最小は事後中央値（平均とは異なる）")
q.num("q31-posterior-mean-map-2", 3 / 12 - 2 / 10)
phi = np.linspace(-5, 2, 700001)
dphi = np.exp(3 * phi) * np.exp(-6 * np.exp(phi))
mode_phi = phi[np.argmax(dphi)]
q.check("q31-posterior-mean-map-3", abs(mode_phi - np.log(0.5)) < 1e-4 and abs(np.log(1 / 3) - mode_phi) > 0.3,
        "log 変換後の最頻値 log0.5 ≠ log(1/3)（誤り）")

# --- 信用区間 ---
U = stats.beta(1, 10).ppf(0.95)
q.num("q31-credible-interval-1", U)
q.concept("q31-credible-interval-2", "信用区間の解釈（数値なし）")
B = stats.beta(1, 10)
eq = (B.ppf(0.025), B.ppf(0.975))
q.check("q31-credible-interval-3", np.isclose(B.cdf(0.259), 0.95, atol=1e-3) and abs(eq[0] - 0.003) < 0.0005 and abs(eq[1] - 0.308) < 0.0005
        and B.cdf(0.308) > 0.97 and (eq[1] - eq[0]) > U and abs(B.cdf(0.309) - B.cdf(0.05) - 0.95) > 0.03,
        "HPD=[0,0.259]、等裾=[0.003,0.308]、他の誤答は95%でない")

# --- 予測分布 ---
q.num("q31-predictive-distribution-1", 9 * 10 / (15 * 16))
q.check("q31-predictive-distribution-1", abs(integrate.quad(lambda t: t**2 * stats.beta.pdf(t, 9, 6), 0, 1)[0] - 0.375) < 1e-10, "数値積分と一致")
q.num("q31-predictive-distribution-2", stats.norm.ppf(0.975) * np.sqrt(4 + tn2))
p0 = integrate.quad(lambda l: np.exp(-l) * gp.pdf(l), 0, 50)[0]
q.num("q31-predictive-distribution-3", p0)
q.check("q31-predictive-distribution-3", abs(p0 - stats.nbinom.pmf(0, 18, 6 / 7)) < 1e-12, "負の二項分布と一致")

# --- 無情報 ---
mu = np.linspace(-20, 30, 500001); xb = rng.normal(5, 3, 25)
lik = np.exp(-np.sum((xb[:, None] - mu[None, :]) ** 2, axis=0) / (2 * 9) + np.sum((xb - xb.mean()) ** 2) / 18)
q.check("q31-noninformative-prior-1", np.isfinite(np.trapezoid(lik, mu)) and np.trapezoid(lik, mu) > 0, "平坦事前でも事後は正規化できる（誤り）")
q.num("q31-noninformative-prior-2", 5 + stats.norm.ppf(0.975) * 3 / 5)

# --- ジェフリーズ ---
lmb, xx = sp.symbols("lambda x", positive=True)
logf = xx * sp.log(lmb) - lmb
I = -sp.diff(logf, lmb, 2).subs(xx, lmb)  # E[X]=λ を代入
q.check("q31-jeffreys-prior-1", sp.simplify(sp.sqrt(I) - lmb ** sp.Rational(-1, 2)) == 0, "√I(λ) = λ^(-1/2)")
q.num("q31-jeffreys-prior-2", 3.5 / 11)

# --- ベイズファクター ---
B10 = (0.7**7 * 0.3**3) / 0.5**10
q.num("q31-bayes-factor-1", B10)
q.check("q31-bayes-factor-1", np.isclose(B10, stats.binom.pmf(7, 10, 0.7) / stats.binom.pmf(7, 10, 0.5)), "尤度比と一致")
po = B10 * 0.25
q.num("q31-bayes-factor-2", po / (1 + po))

# --- 周辺尤度 ---
m2 = integrate.quad(lambda t: comb(3, 2) * t**2 * (1 - t), 0, 1)[0]
q.num("q31-marginal-likelihood-1", m2)
q.check("q31-marginal-likelihood-1", all(abs(integrate.quad(lambda t: stats.binom.pmf(k, 3, t), 0, 1)[0] - 0.25) < 1e-10 for k in range(4)),
        "どの x でも 1/(n+1)")
m1 = integrate.quad(lambda t: stats.binom.pmf(7, 10, t), 0, 1)[0]
q.num("q31-marginal-likelihood-2", m1 / stats.binom.pmf(7, 10, 0.5))

# --- 階層ベイズ ---
Bj = 25 / (25 + 25)
q.num("q31-hierarchical-bayes-model-1", Bj * 50 + (1 - Bj) * 62)
Bs = [(100 / nj) / (100 / nj + 25) for nj in [2, 5, 20, 100]]
q.check("q31-hierarchical-bayes-model-2", all(np.diff(Bs) < 0), "n_j が小さいほど縮小係数が大きい")

# --- 経験ベイズ ---
tau2 = 10 - 4; Bh = 4 / (4 + tau2)
q.num("q31-empirical-bayes-1", Bh * 15 + (1 - Bh) * 20)

# --- ベイズ決定 ---
q.num("q31-bayes-decision-theory-1", min(0.3 * 100, 0.7 * 20))
q.check("q31-bayes-decision-theory-1", 0.7 * 20 < 0.3 * 100, "廃棄がベイズ解")

# 校閲（review-ch27-33）で解説に加えた値
from scipy import stats as _st
_B = _st.beta(1, 10)
q.check("q31-credible-interval-3", abs(_B.cdf(0.309) - _B.cdf(0.05) - 0.574) < 0.001, "[0.050,0.309] の確率は約57%")
# --- 導出問題（追加）---
_mu, _m0, _k0, _kx, _xb = sp.symbols("mu mu0 kappa0 kappa_x xbar", real=True)
_expo = -(_kx * (_mu - _xb)**2 + _k0 * (_mu - _m0)**2) / 2
_kn = _k0 + _kx; _mn = (_k0 * _m0 + _kx * _xb) / _kn
_diff = sp.simplify(sp.expand(_expo - (-_kn * (_mu - _mn)**2 / 2)))
_ok = sp.diff(_diff, _mu) == 0  # 差は μ を含まない
_vals = {_k0: sp.Rational(1, 4), _kx: 1, _m0: 50, _xb: 56}
_post_mean, _post_var = _mn.subs(_vals), (1 / _kn).subs(_vals)
# 数値積分でも確認（σ²=9, n=9 のデータ平均 56 は尤度 N(56, 1) と同じ）
_dens = lambda m: stats.norm.pdf(56, m, 1) * stats.norm.pdf(m, 50, 2)
_Zc = integrate.quad(_dens, 30, 80)[0]
_Em = integrate.quad(lambda m: m * _dens(m), 30, 80)[0] / _Zc
_Vm = integrate.quad(lambda m: (m - _Em)**2 * _dens(m), 30, 80)[0] / _Zc
q.check("q31-normal-normal-model-4", _ok and _post_mean == sp.Rational(274, 5) and _post_var == sp.Rational(4, 5)
        and abs(_Em - 54.8) < 1e-6 and abs(_Vm - 0.8) < 1e-6, f"平方完成、事後 N(54.8, 0.8)、数値積分 {_Em:.4f}, {_Vm:.4f}")

_lam = sp.symbols("lambda", positive=True)
_a, _b = sp.symbols("a b", positive=True)
_k = sp.symbols("k", integer=True, nonnegative=True)
_I = sp.integrate(_lam**(_a + _k - 1) * sp.exp(-(_b + 1) * _lam), (_lam, 0, sp.oo), conds="none")
_pk = _b**_a / (sp.factorial(_k) * sp.gamma(_a)) * _I
_nb = sp.gamma(_a + _k) / (sp.factorial(_k) * sp.gamma(_a)) * (_b / (_b + 1))**_a * (1 / (_b + 1))**_k
_ok = sp.simplify(sp.powsimp(_pk / _nb, force=True)) == 1
_p1 = float(_nb.subs({_a: 6, _b: 2, _k: 1}))
_nbd = stats.nbinom(6, 2 / 3)
q.check("q31-gamma-poisson-model-4", _ok and abs(_p1 - 128 / 729) < 1e-12 and round(_p1, 3) == 0.176
        and abs(_nbd.pmf(1) - _p1) < 1e-12 and abs(_nbd.mean() - 3) < 1e-12 and abs(_nbd.var() - 4.5) < 1e-12
        and abs(stats.poisson.pmf(1, 3) - 0.149) < 0.001,
        f"積分が負の二項分布に一致、P(Y=1)={_p1:.4f}、平均3・分散4.5、誤答 Po(3) は 0.149")

# --- 4択の追加（ガンマ・ポアソン・予測分布）---
import numpy as _np
from scipy import stats as _st
from scipy.special import beta as _B, comb as _comb
_x = _np.array([6, 3, 5, 4]); _a, _b = 4, 2
_post = _st.gamma(_a + _x.sum(), scale=1 / (_b + len(_x)))
_lam = _np.linspace(1e-3, 15, 30001)
_dens = _st.poisson.pmf(_x[:, None], _lam).prod(0) * _st.gamma.pdf(_lam, _a, scale=1 / _b)
_pm_num = (_lam * _dens).sum() / _dens.sum()
_ok = abs(_post.mean() - _pm_num) < 1e-4 and round(_post.mean(), 2) == 3.67
_wr = [(_a + _x.sum() - 1) / (_b + len(_x)), _x.mean(), (_a + _x.sum()) / (1 / _b + len(_x))]
_ok &= [round(w, 2) for w in _wr] == [3.50, 4.50, 4.89]
q.check("q31-gamma-poisson-model-5", _ok, f"事後平均 {_post.mean():.4f}（数値積分 {_pm_num:.4f}）、誤答 MAP・MLE・尺度取り違え")
_a2, _b2 = 1 + 4, 1 + 2
_pred = _comb(3, 1) * _B(_a2 + 1, _b2 + 2) / _B(_a2, _b2)
_th = _np.linspace(0, 1, 200001)
_num = (_st.binom.pmf(1, 3, _th) * _st.beta.pdf(_th, _a2, _b2)).mean()
_ok = abs(_pred - 0.25) < 1e-12 and abs(_num - _pred) < 1e-4 and abs(_st.betabinom.pmf(1, 3, _a2, _b2) - _pred) < 1e-12
_wr = [_st.binom.pmf(1, 3, 5 / 8), _st.binom.pmf(1, 3, 2 / 3), 1 - 5 / 8]
_ok &= [round(w, 3) for w in _wr] == [0.264, 0.222, 0.375]
q.check("q31-predictive-distribution-4", _ok, f"予測確率 {_pred:.4f}（ベータ二項）、誤答 {[round(w,4) for w in _wr]}")

q.done()
