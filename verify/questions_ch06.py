"""第6章（連続型分布と標本分布）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, integrate, special

q = Q(6)
rng = np.random.default_rng(20261023)

# --- 正規 ---
q.num("q06-normal-distribution-1", stats.norm.sf(65, 50, 10))
pxy = stats.norm.sf(0, 4, np.sqrt(13))
q.num("q06-normal-distribution-2", pxy)
q.check("q06-normal-distribution-2", abs(stats.norm.sf(0, 4, np.sqrt(5)) - pxy) > 0.01, "分散5（引き算）の誤りとは値が異なる")

# --- 標準正規 ---
q.check("q06-standard-normal-distribution-1", abs(stats.norm.moment(4) - 3) < 1e-10, "E[Z^4]=3")
t = sp.Symbol("t")
q.check("q06-standard-normal-distribution-1", sp.diff(sp.exp(t ** 2 / 2), t, 4).subs(t, 0) == 3, "MGF の4階微分も3")
q.num("q06-standard-normal-distribution-2", stats.norm.cdf(2) - stats.norm.cdf(-0.5))

# --- 一様 ---
q.num("q06-uniform-distribution-1", stats.uniform(2, 6).var())
m = sp.Symbol("m")
em = sp.integrate(m * sp.diff(m ** 4, m), (m, 0, 1))
q.num("q06-uniform-distribution-2", float(em))
q.check("q06-uniform-distribution-2", abs(rng.random((200000, 4)).max(axis=1).mean() - 0.8) < 0.003, "シミュレーションでも約0.8")

# --- 指数（レート） ---
q.num("q06-exponential-distribution-1", stats.expon(scale=1 / 0.5).median())
e = stats.expon(scale=10)
cp = e.sf(15) / e.sf(5)
q.check("q06-exponential-distribution-2", abs(cp - np.exp(-1)) < 1e-12, "正解 e^{-1}")
q.check("q06-exponential-distribution-2", all(abs(w - cp) > 0.1 for w in [np.exp(-1.5), np.exp(-0.5), 1 - np.exp(-1)]),
        "誤答は正解と異なる")
pmin = np.prod([stats.expon(scale=1 / r).sf(2) for r in (0.1, 0.3, 0.4)])
q.num("q06-exponential-distribution-3", pmin)
q.check("q06-exponential-distribution-3", abs(pmin - stats.expon(scale=1 / 0.8).sf(2)) < 1e-12, "最小値はレート0.8の指数分布")

# --- ガンマ（形状・尺度） ---
g = stats.gamma(3, scale=2)
q.check("q06-gamma-distribution-1", abs(g.var() - 12) < 1e-12, "正解 12")
grate = stats.gamma(3, scale=1 / 2)
q.check("q06-gamma-distribution-1", abs(grate.var() - 0.75) < 1e-12 and abs(grate.mean() - 1.5) < 1e-12
        and abs(g.mean() - 6) < 1e-12, "誤答はレート読みの分散0.75・期待値1.5、期待値6")
ps = stats.gamma(4, scale=0.5).cdf(1)
q.num("q06-gamma-distribution-2", ps)
q.check("q06-gamma-distribution-2", abs(ps - stats.poisson.sf(3, 2)) < 1e-12, "Po(2) で4回以上の確率と一致")
a, b, x = sp.symbols("alpha beta x", positive=True)
dens = x ** (a - 1) * sp.exp(-x / b) / (sp.gamma(a) * b ** a)
ok3 = True
for av, bv in [(3, 2), (2.5, 0.7), (6, 1.5)]:
    num = integrate.quad(lambda xx: stats.gamma.pdf(xx, av, scale=bv) / xx, 0, np.inf)[0]
    ok3 &= abs(num - 1 / ((av - 1) * bv)) < 1e-7
sym = sp.simplify(sp.gamma(a - 1) * b ** (a - 1) / (sp.gamma(a) * b ** a) - 1 / ((a - 1) * b))
q.check("q06-gamma-distribution-3", bool(ok3) and sym == 0, "E[1/X]=1/((α-1)β)（数値積分と記号計算）")

# --- ベータ ---
q.num("q06-beta-distribution-1", stats.beta(2, 3).var())
cut = 0.3
pw = integrate.quad(lambda yy: stats.gamma.pdf(yy, 3) * stats.gamma.cdf(cut / (1 - cut) * yy, 2), 0, np.inf)[0]
q.check("q06-beta-distribution-2", abs(pw - stats.beta(2, 3).cdf(cut)) < 1e-8, "P(W≤0.3) が Be(2,3) と一致")
q.check("q06-beta-distribution-2", all(abs(stats.beta(2, 3).cdf(c2) - w) > 0.05 for c2 in [0.5] for w in [stats.beta(3, 2).cdf(c2), c2, stats.gamma(5).cdf(c2)]),
        "誤答（Be(3,2)・U(0,1)・Ga(5,1)）とは値が異なる")

# --- コーシー ---
xbar = stats.cauchy.rvs(size=(2000, 1000), random_state=rng).mean(axis=1)
q.check("q06-cauchy-distribution-1", stats.kstest(xbar, "cauchy").pvalue > 0.01 and np.mean(np.abs(xbar) > 1) > 0.4,
        "n=1000 の標本平均も標準コーシー（0に集中しない）＝誤り")
q.num("q06-cauchy-distribution-2", stats.cauchy.sf(np.sqrt(3)))

# --- 対数正規 ---
ln = stats.lognorm(s=0.5, scale=np.e)
q.num("q06-lognormal-distribution-1", ln.mean())
q.check("q06-lognormal-distribution-2", abs(ln.median() - np.e) < 1e-12, "中央値 e^1")
from scipy import optimize
md = optimize.minimize_scalar(lambda xx: -ln.pdf(xx), bounds=(0.1, 10), method="bounded").x
q.check("q06-lognormal-distribution-2", abs(md - np.exp(0.75)) < 1e-4 and abs(ln.mean() - np.exp(1.125)) < 1e-12
        and len({round(v, 6) for v in [np.e, np.exp(1.125), np.exp(0.75), 1.0]}) == 4,
        "誤答は期待値 e^1.125・最頻値 e^0.75・1 で正解と異なる")

# --- 2変量正規 ---
mx, my, sx, sy, r = 50, 60, 10, 15, 0.6
cov = np.array([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])
cm = my + cov[1, 0] / cov[0, 0] * (65 - mx)
cv = cov[1, 1] - cov[1, 0] ** 2 / cov[0, 0]
q.num("q06-bivariate-normal-distribution-1", cm)
q.num("q06-bivariate-normal-distribution-2", stats.norm.sf(80, cm, np.sqrt(cv)))
q.check("q06-bivariate-normal-distribution-2", abs(np.sqrt(cv) - 12) < 1e-12, "条件付き標準偏差12")
mvn = stats.multivariate_normal([0, 0], [[1, 0], [0, 4]])
pts = rng.normal(size=(5, 2))
q.check("q06-bivariate-normal-distribution-3",
        np.allclose(mvn.pdf(pts), stats.norm.pdf(pts[:, 0]) * stats.norm.pdf(pts[:, 1], 0, 2)), "ρ=0 で密度が積に分解")

# --- 多変量正規 ---
mu = np.array([1, 2, 3]); S = np.array([[4, 1, 0], [1, 2, -1], [0, -1, 3]]); av = np.array([1, -1, 2])
mw, vw = av @ mu, av @ S @ av
q.check("q06-multivariate-normal-distribution-1", mw == 5 and vw == 20, "平均5・分散20")
q.num("q06-multivariate-normal-distribution-1", stats.norm.sf(10, mw, np.sqrt(vw)))
Xs = rng.multivariate_normal(mu, S, size=200000)
d = Xs - mu
D2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
q.check("q06-multivariate-normal-distribution-2", stats.kstest(D2[:20000], "chi2", args=(3,)).pvalue > 0.01
        and abs(D2.mean() - 3) < 0.03 and abs(D2.var() - 6) < 0.15, "D^2 は χ²(3)（平均3・分散6）")
q.check("q06-multivariate-normal-distribution-2", abs(3 ** 2 * 2 - 6) > 1 and stats.chi2(2).mean() != 3,
        "誤答：3χ²(1) は分散18、χ²(2) は平均2")

# --- 混合正規 ---
w = 0.3
m1 = w * 0 + (1 - w) * 4
m2 = w * (1 + 0) + (1 - w) * (1 + 16)
q.num("q06-normal-mixture-distribution-1", m2 - m1 ** 2)
num_var = integrate.quad(lambda xx: (xx - m1) ** 2 * (w * stats.norm.pdf(xx) + (1 - w) * stats.norm.pdf(xx, 4)), -20, 30)[0]
q.check("q06-normal-mixture-distribution-1", abs(num_var - 4.36) < 1e-8, "密度の数値積分でも4.36")
q.check("q06-normal-mixture-distribution-2", abs((0.09 + 0.49) - num_var) > 1, "線形結合の分散0.58と混合の分散4.36は異なる＝誤り")

# --- カイ二乗 ---
q.check("q06-chi-squared-distribution-1", stats.chi2(8).var() == 16, "正解16")
q.check("q06-chi-squared-distribution-1", all(w_ != 16 for w_ in [stats.chi2(8).mean(), stats.chi2(8).std(), 64]), "誤答は期待値8・標準偏差4・64")
pc = stats.chi2(9).sf(9 * 6 / 4)
q.num("q06-chi-squared-distribution-2", pc)
q.check("q06-chi-squared-distribution-2", abs(stats.chi2(10).sf(13.5) - 0.197) < 5e-4, "自由度を10と取り違えた P(χ²(10)>13.5)=0.197")
y = sp.Symbol("y", positive=True)
Phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2
fy = sp.simplify(sp.diff(Phi(sp.sqrt(y)) - Phi(-sp.sqrt(y)), y))
target = sp.exp(-y / 2) / sp.sqrt(2 * sp.pi * y)
q.check("q06-chi-squared-distribution-3", sp.simplify(fy - target) == 0
        and all(abs(float(target.subs(y, yv)) - stats.chi2.pdf(yv, 1)) < 1e-12 for yv in [0.3, 1, 2.5, 7]),
        "密度が χ²(1) と一致")

# --- 非心カイ二乗（λ=Σμ²、scipy の nc と同じ） ---
mus = np.array([1, 2, 0])
q.num("q06-noncentral-chi-squared-distribution-1", stats.ncx2(3, (mus ** 2).sum()).var())
W = (rng.normal(size=(400000, 3)) + mus) ** 2
q.check("q06-noncentral-chi-squared-distribution-1", abs(W.sum(axis=1).var() - 26) < 0.4, "シミュレーションでも約26")

# --- t ---
q.num("q06-t-distribution-1", stats.t.ppf(0.975, 10))
q.check("q06-t-distribution-2", abs(stats.t(6).var() - 1.5) < 1e-12, "正解1.5")
q.check("q06-t-distribution-2", all(abs(w_ - 1.5) > 0.1 for w_ in [1, 1.2, 3]), "誤答は正解と異なる")

# --- t 統計量 ---
tval = (103.5 - 100) / (6.0 / np.sqrt(16))
q.num("q06-t-statistic-1", tval)
q.check("q06-t-statistic-1", abs(stats.t.ppf(0.975, 15) - 2.131) < 5e-4 and tval > stats.t.ppf(0.975, 15), "両側5%点2.131、棄却")
res = stats.ttest_ind_from_stats(24.0, 3.0, 8, 20.5, np.sqrt(7.0), 10, equal_var=True)
q.num("q06-t-statistic-2", res.statistic)

# --- 非心 t ---
nu, dl = 10, 1
form = dl * np.sqrt(nu / 2) * special.gamma((nu - 1) / 2) / special.gamma(nu / 2)
q.num("q06-noncentral-t-distribution-1", stats.nct(nu, dl).mean())
q.check("q06-noncentral-t-distribution-1", abs(form - stats.nct(nu, dl).mean()) < 1e-9
        and abs(special.gamma(4.5) - 11.632) < 5e-4, "公式と scipy が一致、Γ(4.5)=11.632")

# --- F ---
q.check("q06-f-distribution-1", abs(stats.t.ppf(0.975, 10) ** 2 - stats.f.ppf(0.95, 1, 10)) < 1e-9
        and abs(stats.f.ppf(0.95, 1, 10) - 4.965) < 5e-4, "t_{0.025}(10)^2 = F_{0.05}(1,10)=4.965")
q.num("q06-f-distribution-2", stats.f.ppf(0.05, 5, 10))
q.check("q06-f-distribution-2", abs(stats.f.ppf(0.95, 10, 5) - 4.735) < 5e-4, "F_{0.05}(10,5)=4.735")

# --- 非心 F ---
q.num("q06-noncentral-f-distribution-1", stats.ncf(3, 12, 4).mean())
q.check("q06-noncentral-f-distribution-1", abs(stats.f(3, 12).mean() - 1.2) < 1e-12, "中心 F の期待値1.2")

# --- 導出問題（追加） ---
def _deriv06():
    import numpy as _N
    import sympy as _S
    sx, sy = _S.symbols("sigma_X sigma_Y", positive=True)
    rho, b = _S.symbols("rho beta", real=True)
    bsol = _S.solve(rho * sx * sy - b * sx ** 2, b)[0]
    ok = _S.simplify(bsol - rho * sy / sx) == 0
    VW = sy ** 2 - 2 * b * rho * sx * sy + b ** 2 * sx ** 2
    ok &= _S.simplify(VW.subs(b, bsol) - sy ** 2 * (1 - rho ** 2)) == 0
    # 一般の条件付き分布の公式（Σ12/Σ22, Σ11-Σ12²/Σ22）と一致することを数値で
    mx, my, s1, s2, r = 50.0, 60.0, 10.0, 15.0, 0.6
    S = _N.array([[s1 ** 2, r * s1 * s2], [r * s1 * s2, s2 ** 2]])
    x = 65.0
    cm = my + S[1, 0] / S[0, 0] * (x - mx); cv = S[1, 1] - S[1, 0] ** 2 / S[0, 0]
    ok &= abs(cm - (my + r * s2 / s1 * (x - mx))) < 1e-12 and abs(cv - s2 ** 2 * (1 - r ** 2)) < 1e-9
    q.check("q06-bivariate-normal-distribution-4", bool(ok), "sympy: β=ρσY/σX、V[W]=σY²(1-ρ²)、一般公式と一致")
_deriv06()


# --- 4択の追加 ---
def _choice06():
    import numpy as _np
    from scipy import stats as _st
    # 1次結合 W=2X1-X2、X ~ N(3,4)
    a = _np.array([2, -1]); mu = _np.array([3, 3]); S = _np.diag([4, 4])
    m, v = float(a @ mu), float(a @ S @ a)
    ch = [(3, 12), (3, 20), (9, 20), (3, 4)]
    ans = q.qs["q06-normal-distribution-3"]["answer"]
    q.check("q06-normal-distribution-3",
            ch[ans] == (m, v) and all(c != (m, v) for i, c in enumerate(ch) if i != ans),
            f"a'μ={m}, a'Σa={v}、誤答は一致しない")
    rng = _np.random.default_rng(6)
    w = 2 * rng.normal(3, 2, 400000) - rng.normal(3, 2, 400000)
    q.check("q06-normal-distribution-3", abs(w.mean() - 3) < 0.03 and abs(w.var() - 20) < 0.2,
            f"シミュレーション 平均≈{w.mean():.3f}, 分散≈{w.var():.3f}")

    # 2変量正規の条件付き分散（シューア補元）
    sx, sy, r = 4.0, 10.0, 0.8
    Sig = _np.array([[sx ** 2, r * sx * sy], [r * sx * sy, sy ** 2]])
    cv = Sig[1, 1] - Sig[1, 0] ** 2 / Sig[0, 0]
    ch = [6, 64, 20, 36]
    ans = q.qs["q06-bivariate-normal-distribution-5"]["answer"]
    q.check("q06-bivariate-normal-distribution-5",
            abs(ch[ans] - cv) < 1e-9 and all(abs(c - cv) > 1e-6 for i, c in enumerate(ch) if i != ans),
            f"Σyy-Σyx²/Σxx={cv}、誤答（SD 6, σ²ρ² 64, σ²(1-ρ) 20）は一致しない")
    q.check("q06-bivariate-normal-distribution-5",
            abs(_np.sqrt(cv) - 6) < 1e-9 and abs(sy ** 2 * r ** 2 - 64) < 1e-9 and abs(sy ** 2 * (1 - r) - 20) < 1e-9,
            "誤答の由来：条件付きSD／σ²ρ²／σ²(1-ρ)")

    # F 分布の性質（m=3, n=10 で確かめる。一般の形は explanation のとおり）
    mm, nn = 3, 10
    ps = [0.05, 0.3, 0.5, 0.9]
    t0 = all(abs(_st.f.ppf(1 - p, nn, mm) - 1 / _st.f.ppf(p, mm, nn)) < 1e-9 for p in ps)   # 1/X ~ F(n,m)
    t1 = abs(_st.f.mean(mm, nn) - 1) < 1e-9                                                  # 期待値 1？
    t2 = abs(_st.f.ppf(0.95, mm, nn) - _st.f.ppf(0.95, nn, mm)) < 1e-9                       # 同じ分布？
    # U/V = (m/n)X の分布が F(m,n) と一致するか
    t3 = all(abs(_st.f.cdf(x * nn / mm, mm, nn) - _st.f.cdf(x, mm, nn)) < 1e-9 for x in [0.5, 1, 2])
    truth = [t0, t1, t2, t3]
    ans = q.qs["q06-f-distribution-3"]["answer"]
    q.check("q06-f-distribution-3", truth[ans] and sum(truth) == 1,
            f"scipy（m=3,n=10）: 各記述の真偽 {truth}、E={_st.f.mean(mm, nn):.3f}、"
            f"F0.05(3,10)={_st.f.ppf(0.95, 3, 10):.2f}, F0.05(10,3)={_st.f.ppf(0.95, 10, 3):.2f}")
_choice06()

# --- 追加：答えが「正しい」の正誤問題 ---
import numpy as _np
from scipy import stats as _st, integrate as _ig
def _conv(d1, d2, z):
    return _ig.quad(lambda x: d1.pdf(x) * d2.pdf(z - x), 0, z, limit=200)[0]
_a1, _a2, _be = 1.5, 2.3, 2.0
_g1, _g2, _gs = _st.gamma(_a1, scale=_be), _st.gamma(_a2, scale=_be), _st.gamma(_a1 + _a2, scale=_be)
_same = max(abs(_conv(_g1, _g2, z) - _gs.pdf(z)) for z in (0.5, 2, 5, 10, 20)) < 1e-7
# 反例：尺度が違うと、平均・分散を合わせたガンマ分布とも一致しない
_h1, _h2 = _st.gamma(1.5, scale=1.0), _st.gamma(2.3, scale=3.0)
_m = 1.5 * 1 + 2.3 * 3; _v = 1.5 * 1 + 2.3 * 9
_gm = _st.gamma(_m**2 / _v, scale=_v / _m)
_diff = max(abs(_conv(_h1, _h2, z) - _gm.pdf(z)) for z in (0.5, 2, 5, 10, 20))
q.check("q06-gamma-distribution-4", _same and _diff > 1e-3,
        f"尺度共通なら畳み込み＝Ga(3.8, 2) の密度、尺度が違うとモーメントを合わせたガンマとも差 {_diff:.4f}")

q.done()
