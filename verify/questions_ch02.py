"""第2章 確率分布と母関数：問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, integrate

q = Q(2)
x, y, t, s, m, u = sp.symbols("x y t s m u", real=True)
lam1, lam2, lam = sp.symbols("lambda1 lambda2 lambda", positive=True)

# --- 累積分布関数 ---
# 1: サイコロの分布関数は x=1 で左連続でない
F = lambda v: np.sum(np.arange(1, 7) <= v) / 6
q.check("q02-cumulative-distribution-function-1", F(1) == 1 / 6 and abs(F(1 - 1e-9) - 0) < 1e-12,
        "F(1)=1/6 だが左極限は0 → 左連続ではない（answer=false）")
Fm = lambda v: 0.0 if v < 0 else 0.2 + 0.8 * (1 - np.exp(-v))
q.num("q02-cumulative-distribution-function-2", Fm(1) - Fm(0))
q.check("q02-cumulative-distribution-function-2", abs(Fm(1) - 0.506) > 0.1, "誤答 F(1)≈0.706 とは区別される")
FM = (m / 2) ** 3
fM = sp.diff(FM, m)
EM = sp.integrate(m * fM, (m, 0, 2))
EM2 = sp.integrate(1 - FM, (m, 0, 2))
mc = np.random.default_rng(0).uniform(0, 2, (200000, 3)).max(axis=1).mean()
q.check("q02-cumulative-distribution-function-3", sp.simplify(fM - 3 * m**2 / 8) == 0 and EM == sp.Rational(3, 2) and EM2 == EM and abs(mc - 1.5) < 0.01,
        f"f_M=3m^2/8, E[M]={EM}（生存関数の積分・モンテカルロ {mc:.4f} とも一致）")

# --- 生存関数 ---
ET = sp.integrate((1 + t) ** -3, (t, 0, sp.oo))
ET2 = sp.integrate(t * 3 * (1 + t) ** -4, (t, 0, sp.oo))
q.num("q02-survival-function-1", float(ET))
q.check("q02-survival-function-1", ET == ET2, "密度から計算しても一致")
sv = stats.geom.sf(5, 0.2)  # scipy の geom は試行回数（1,2,...）
ch = [0.328, 0.410, 0.672, 0.262]
q.check("q02-survival-function-2", abs(sv - ch[0]) < 0.0005 and all(abs(sv - c) > 0.05 for c in ch[1:]),
        f"S(5)={sv:.5f}、誤答 {0.8**4:.3f},{1-0.8**5:.3f},{0.8**6:.3f} は一致しない")

# --- ハザード関数 ---
tp = sp.symbols("tp", positive=True)
h = sp.diff(-sp.log(sp.exp(-tp ** sp.Rational(1, 2))), tp)
dh = sp.diff(h, tp)
q.check("q02-hazard-function-1", sp.simplify(h - tp ** sp.Rational(-1, 2) / 2) == 0 and all(dh.subs(tp, v) < 0 for v in [0.01, 0.5, 1, 5, 100]),
        "h(t)=0.5 t^(-1/2)、導関数が負 → 単調減少（選択肢0）")
H = sp.integrate(sp.Rational(2, 100) * u, (u, 0, t))
S_ = sp.exp(-H)
val = float(S_.subs(t, 15) / S_.subs(t, 10))
q.num("q02-hazard-function-2", val)
wb = stats.weibull_min(c=2, scale=10)  # h=0.02t ⇔ ワイブル形状2・尺度10
q.check("q02-hazard-function-2", abs(wb.sf(15) / wb.sf(10) - val) < 1e-12, "scipy のワイブル分布（形状2・尺度10）でも一致")
ST = sp.exp(-lam1 * tp) * sp.exp(-lam2 * tp)
hT = sp.simplify(-sp.diff(sp.log(ST), tp))
rng = np.random.default_rng(1)
mins = np.minimum(rng.exponential(1 / 1.5, 300000), rng.exponential(1 / 0.5, 300000))
q.check("q02-hazard-function-3", sp.simplify(hT - (lam1 + lam2)) == 0 and abs(mins.mean() - 1 / 2.0) < 0.01,
        f"h_T=λ1+λ2（記号計算）、λ=(1.5,0.5) のシミュレーション平均 {mins.mean():.4f}≈1/2")

# --- 同時確率関数 ---
P = np.array([[0.10, 0.20, 0.10], [0.15, 0.25, 0.20]])
q.check("q02-joint-probability-function-1", abs(P.sum() - 1) < 1e-12, "表の合計が1")
q.num("q02-joint-probability-function-1", sum(P[i, j] for i in range(2) for j in range(3) if i + j >= 2))
tot = sum(xx + yy for xx in (1, 2) for yy in (1, 2, 3))
cands = [sp.Rational(1, 21), sp.Rational(1, 12), sp.Rational(1, 9), sp.Rational(1, 18)]
q.check("q02-joint-probability-function-2", sp.Rational(1, tot) == cands[0] and all(c != cands[0] for c in cands[1:]),
        f"合計 {tot} → c=1/{tot}、誤答は一致しない")

# --- 周辺分布 ---
indep = np.full((2, 2), 0.25); same = np.array([[0.5, 0], [0, 0.5]])
q.check("q02-marginal-distribution-1", np.allclose(indep.sum(0), same.sum(0)) and np.allclose(indep.sum(1), same.sum(1)) and not np.allclose(indep, same),
        "周辺が同じで同時分布が違う反例（answer=false）")
q.num("q02-marginal-distribution-2", stats.binom.pmf(3, 10, 0.3))
# 多項分布の周辺を直接和で確認
from math import factorial
mp = sum(factorial(10) / (factorial(a) * factorial(3) * factorial(7 - a)) * 0.5**a * 0.3**3 * 0.2**(7 - a) for a in range(8))
q.check("q02-marginal-distribution-2", abs(mp - stats.binom.pmf(3, 10, 0.3)) < 1e-12, "多項分布を他の成分で和をとっても一致")
cov = np.array([[4, 0.5 * 2 * 3], [0.5 * 2 * 3, 9]])
ys = np.random.default_rng(2).multivariate_normal([1, 2], cov, 400000)[:, 1]
q.check("q02-marginal-distribution-3", abs(ys.mean() - 2) < 0.02 and abs(ys.var() - 9) < 0.1 and abs(9 * (1 - 0.25) - 6.75) < 1e-12,
        f"Y の標本平均 {ys.mean():.3f}・分散 {ys.var():.3f} ≈ N(2,9)、誤答 6.75 は条件付き分散")

# --- 周辺確率関数 ---
pxy = {(a, b): sp.Rational(a + 2 * b, 12) for a in range(3) for b in range(2)}
q.check("q02-marginal-probability-function-1", sum(pxy.values()) == 1, "同時確率の合計が1")
q.num("q02-marginal-probability-function-1", float(pxy[(1, 0)] + pxy[(1, 1)]))

# --- 条件付き確率関数 ---
P2 = np.array([[0.05, 0.15, 0.20], [0.30, 0.10, 0.20]])
q.check("q02-conditional-probability-function-1", abs(P2.sum() - 1) < 1e-12, "表の合計が1")
q.num("q02-conditional-probability-function-1", P2[1, 2] / P2[1].sum())
cond = np.array([stats.poisson.pmf(k, 2) * stats.poisson.pmf(5 - k, 3) for k in range(6)]) / stats.poisson.pmf(5, 5)
b4 = stats.binom.pmf(range(6), 5, 0.4); b6 = stats.binom.pmf(range(6), 5, 0.6)
q.check("q02-conditional-probability-function-2", np.allclose(cond, b4) and not np.allclose(cond, b6) and not np.allclose(cond, stats.poisson.pmf(range(6), 2)),
        "条件付き分布が Bin(5,0.4) と一致、Bin(5,0.6)・Po(2) とは不一致")

# --- 同時確率密度関数 ---
I = sp.integrate(sp.integrate(x * y, (y, 0, x)), (x, 0, 1))
q.num("q02-joint-probability-density-function-1", float(1 / I))
fX = sp.integrate(8 * x * y, (y, 0, x)); fY = sp.integrate(8 * x * y, (x, y, 1))
q.check("q02-joint-probability-density-function-2", sp.simplify(fX * fY - 8 * x * y) != 0,
        f"f_X={fX}, f_Y={sp.factor(fY)} の積は 8xy と一致しない → 独立でない（answer=false）")
pv = integrate.dblquad(lambda xx, yy: np.exp(-xx - yy), 0, np.inf, 0, lambda yy: 2 * yy)[0]
q.num("q02-joint-probability-density-function-3", pv)

# --- 周辺確率密度関数 ---
EY = sp.integrate(y * fY, (y, 0, 1))
q.num("q02-marginal-probability-density-function-1", float(EY))
q.check("q02-marginal-probability-density-function-1", sp.integrate(fY, (y, 0, 1)) == 1 and EY == sp.Rational(8, 15), "f_Y の積分が1、E[Y]=8/15")

# --- 条件付き確率密度関数 ---
fyx = sp.simplify(8 * x * y / fX)
EYx = sp.integrate(y * fyx, (y, 0, x))
q.num("q02-conditional-probability-density-function-1", float(EYx.subs(x, sp.Rational(6, 10))))
mu = 60 + 0.6 * 8 / 10 * (65 - 50); var = 64 * (1 - 0.36)
# 数値でも：同時分布からの条件付き（共分散行列による公式）
S2 = np.array([[100, 0.6 * 10 * 8], [0.6 * 10 * 8, 64]])
mu2 = 60 + S2[1, 0] / S2[0, 0] * 15; var2 = S2[1, 1] - S2[1, 0] ** 2 / S2[0, 0]
q.check("q02-conditional-probability-density-function-2", abs(mu - 67.2) < 1e-9 and abs(var - 40.96) < 1e-9 and abs(mu2 - mu) < 1e-9 and abs(var2 - var) < 1e-9
        and abs(60 + 0.6 * 10 / 8 * 15 - 71.25) < 1e-9 and abs(64 * 0.36 - 23.04) < 1e-9,
        "条件付き N(67.2, 40.96)、誤答 71.25・64・23.04 はいずれも別の値")

# --- モーメント母関数 ---
Mt = sp.exp(3 * t + 2 * t**2)
mean = sp.diff(Mt, t).subs(t, 0); var = sp.diff(Mt, t, 2).subs(t, 0) - mean**2
q.check("q02-moment-generating-function-1", mean == 3 and var == 4, f"M から平均 {mean}・分散 {var} → N(3,4)（選択肢0）")
K = sp.log((1 - 2 * t) ** -3)
q.num("q02-moment-generating-function-2", float(sp.diff(K, t, 2).subs(t, 0)))
q.check("q02-moment-generating-function-2", abs(stats.gamma(a=3, scale=2).var() - 12) < 1e-12 and abs(stats.chi2(6).var() - 12) < 1e-12,
        "ガンマ(形状3・尺度2)・χ²(6) の分散も12")
xp = sp.symbols("xp", positive=True)
Mexp = sp.integrate(sp.exp(t * xp) * lam * sp.exp(-lam * xp), (xp, 0, sp.oo), conds="none")
n = 4
Mgam = sp.integrate(sp.exp(t * xp) * lam**n * xp**(n - 1) * sp.exp(-lam * xp) / sp.factorial(n - 1), (xp, 0, sp.oo), conds="none")
q.check("q02-moment-generating-function-3", sp.simplify(Mexp - lam / (lam - t)) == 0 and sp.simplify(Mgam - (lam / (lam - t)) ** n) == 0,
        "指数分布の M=λ/(λ-t)、ガンマ(形状n=4)の M=(λ/(λ-t))^n と一致")

# --- 確率母関数 ---
G = sp.exp(2 * (s - 1))  # Po(2) で G''(1)=E[X(X-1)]=4 ≠ E[X^2]=6
q.check("q02-probability-generating-function-1", sp.diff(G, s, 2).subs(s, 1) == 4 and 4 != 2 + 4,
        "Po(2) で G''(1)=4=E[X(X-1)] ≠ E[X^2]=6（answer=false）")
G2 = sp.expand((sp.Rational(2, 10) + sp.Rational(5, 10) * s + sp.Rational(3, 10) * s**2) ** 2)
q.num("q02-probability-generating-function-2", float(G2.coeff(s, 2)))
GS = sp.exp(4 * ((sp.Rational(7, 10) + sp.Rational(3, 10) * s) - 1))
q.num("q02-probability-generating-function-3", float(GS.subs(s, 0)))
rng = np.random.default_rng(3)
Ns = rng.poisson(4, 300000); Ss = rng.binomial(Ns, 0.3)
q.check("q02-probability-generating-function-3", abs((Ss == 0).mean() - np.exp(-1.2)) < 0.005 and sp.simplify(GS - sp.exp(sp.Rational(6, 5) * (s - 1))) == 0,
        f"G_S が Po(1.2) の母関数、シミュレーション P(S=0)={(Ss==0).mean():.4f}")

# --- 特性関数 ---
# コーシー分布の特性関数を数値積分で確認（|φ|≤1 で存在）
phi1 = 2 * integrate.quad(stats.cauchy.pdf, 0, np.inf, weight="cos", wvar=1.0)[0]
q.check("q02-characteristic-function-1", abs(phi1 - np.exp(-1)) < 1e-4, f"コーシー分布で φ(1)={phi1:.5f}≈e^(-1)（存在する。answer=true）")
phiX = sp.exp(sp.I * t - 2 * t**2)
phiY = sp.exp(2 * sp.I * t) * phiX.subs(t, 3 * t)
target = sp.exp(5 * sp.I * t - 18 * t**2)
wrong = [sp.exp(5 * sp.I * t - 6 * t**2), sp.exp(3 * sp.I * t - 18 * t**2), sp.exp(5 * sp.I * t - 36 * t**2)]
q.check("q02-characteristic-function-2", sp.simplify(phiY - target) == 0 and all(sp.simplify(phiY - w) != 0 for w in wrong),
        "φ_Y=exp(5it-18t^2)、誤答は一致しない")

# --- 確率関数と確率密度関数（用語の見直しで追加） ---
import sympy as _sp
from fractions import Fraction as _F
from scipy import stats as _st
q.check("q02-probability-mass-and-density-function-1", abs(_st.uniform(0, 0.5).pdf(0.25) - 2) < 1e-12, "反例：U(0,0.5) の密度は 2 > 1")
_c = 1 / (1 / (1 - _F(1, 3)))  # 等比級数の和 3/2 の逆数
_tail = 1 - _c - _c * _F(1, 3)
_s = sum(float(_c) * (1 / 3) ** x for x in range(2, 200))
_wrong = [_c * _F(1, 9), sum(_F(1, 3) ** x for x in range(2, 200)), _c + _c * _F(1, 3)]
q.check("q02-probability-mass-and-density-function-2", _tail == _F(1, 9) and abs(_s - 1 / 9) < 1e-12, "正解 1/9")
q.check("q02-probability-mass-and-density-function-2", _wrong[0] == _F(2, 27) and abs(float(_wrong[1]) - 1 / 6) < 1e-12 and _wrong[2] == _F(8, 9), "誤答 2/27, 1/6, 8/9 の由来と正解との不一致")
_x, _cc = _sp.symbols("x c", positive=True)
_csol = _sp.solve(_sp.integrate(_cc * _x * (2 - _x), (_x, 0, 2)) - 1, _cc)[0]
q.check("q02-probability-mass-and-density-function-3", _csol == _sp.Rational(3, 4), "c = 3/4")
q.num("q02-probability-mass-and-density-function-3", float(_sp.integrate(_csol * _x * (2 - _x), (_x, 0, _sp.Rational(1, 2)))))

q.done()
