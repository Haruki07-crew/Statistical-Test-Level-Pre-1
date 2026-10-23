"""第29章 検算スクリプト（欠測データと生存時間解析）。
EM アルゴリズム（打ち切り指数分布・混合正規）、打ち切り・切断データの尤度、欠測メカニズム（MCAR/MAR/MNAR）、
完全ケース分析、単一代入、多重代入とルービンの規則、カプラン・マイヤー推定量、グリーンウッドの公式、
ログランク検定（scipy.stats.logrank でも）、比例ハザードモデル、部分尤度（ニュートン法・スコア・ワルド・尤度比）、
ハザード比を numpy/scipy/sympy で独立に再計算し、本文と照合する。
charts の params は各 md の front matter を yaml で読み、本文の式から再計算して照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4、第2位なら 5e-3）。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch29" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def chart(term, cid, k=0):
    return [c for c in front(term).get("charts", []) if c.get("id") == cid or c.get("type") == cid][k]


def sliders_ok(c):
    p = c["params"]
    return all(s["min"] <= p[s["param"]] <= s["max"] for s in c.get("sliders", []))


z975 = stats.norm.ppf(0.975)

# ================================================================ em-algorithm
T = "em-algorithm"
lam = 1.0
lams, Es, lls = [], [], []
for k in range(9):
    lams.append(lam)
    Es.append(27 + 2 / lam)
    lls.append(4 * math.log(lam) - 27 * lam)
    lam = 6 / (27 + 2 / lam)
check(T, "例1 表 λ(k) k=0..5", allclose(lams[:6], [1, 0.2069, 0.1636, 0.1530, 0.1497, 0.1487]), np.round(lams[:6], 4))
check(T, "例1 表 E[Σt] k=0..5", allclose(Es[:6], [29, 36.667, 39.222, 40.074, 40.358, 40.453], 5e-4), np.round(Es[:6], 3))
check(T, "例1 表 対数尤度 k=0..5", allclose(lls[:6], [-27, -11.888, -11.659, -11.640, -11.638, -11.638], 5e-4), np.round(lls[:6], 3))
check(T, "例1 対数尤度が単調増加", all(np.diff(lls) >= 0), np.round(lls, 4))
lam = 1.0
for _ in range(500):
    lam = 6 / (27 + 2 / lam)
check(T, "例1 収束値 = 4/27 = 0.1481", close(lam, 4 / 27, 1e-12) and close(4 / 27, 0.1481), lam)
L = sp.symbols("lambda", positive=True)
g = 6 / (27 + 2 / L)
gd = sp.simplify(sp.diff(g, L).subs(L, sp.Rational(4, 27)))
check(T, "例1 反復写像の微分 g'(4/27)=1/3（sympy）", gd == sp.Rational(1, 3), gd)
errs = np.abs(np.array(lams) - 4 / 27)
ratio = errs[3:8] / errs[2:7]
check(T, "例1 ずれの比が約1/3（k=3..7）", np.all(np.abs(ratio - 1 / 3) < 0.03), np.round(ratio, 4))
# 観測データ尤度の最大化
res = optimize.minimize_scalar(lambda l: -(4 * math.log(l) - 27 * l), bounds=(1e-4, 2), method="bounded")
check(T, "例1 観測データ対数尤度の数値最大化 = 4/27", close(res.x, 4 / 27, 1e-5), res.x)
check(T, "例1 誤った推定 6/27=0.222, 4/17=0.235", close(6 / 27, 0.222, 5e-4) and close(4 / 17, 0.235, 5e-4), (6 / 27, 4 / 17))
# 無記憶性 E[T|T>c]=c+1/λ（数値積分）
c0, l0 = 4.0, 0.3
num = optimize.quad if False else None
from scipy.integrate import quad
eT = quad(lambda t: t * l0 * math.exp(-l0 * t), c0, np.inf)[0] / math.exp(-l0 * c0)
check(T, "E[T|T>c]=c+1/λ（数値積分, c=4, λ=0.3）", close(eT, c0 + 1 / l0, 1e-8), eT)
c = chart(T, "line-data")
ys = c["params"]["series"][0]["y"]
check(T, "chart line-data EM の反復値（9点）", allclose(ys, np.round(lams, 4)), ys)
check(T, "chart line-data 最尤推定量 0.1481", allclose(c["params"]["series"][1]["y"], [round(4 / 27, 4)] * 9), c["params"]["series"][1]["y"][0])

# 例2 混合正規
x = np.array([1.0, 1.8, 2.5, 3.6, 4.4, 5.0])


def estep(w, m1, m2):
    a = w * stats.norm.pdf(x, m1, 1)
    b = (1 - w) * stats.norm.pdf(x, m2, 1)
    return a / (a + b), float(np.log(a + b).sum())


gam, ll0 = estep(0.5, 1, 5)
check(T, "例2 x=2.5 の負担率 1/(1+e^-2)=0.881", close(gam[2], 1 / (1 + math.exp(-2)), 1e-12) and close(gam[2], 0.881, 5e-4), gam[2])
check(T, "例2 負担率6点", allclose(gam, [0.9997, 0.9918, 0.8808, 0.0832, 0.0037, 0.0003]), np.round(gam, 4))
check(T, "例2 負担率の合計 2.9595", close(gam.sum(), 2.9595), gam.sum())
w1 = gam.mean()
s1 = (gam * x).sum()
s2 = ((1 - gam) * x).sum()
m1 = s1 / gam.sum()
m2 = s2 / (1 - gam).sum()
check(T, "例2 w=0.4932", close(w1, 0.4932), w1)
check(T, "例2 Σγx=5.3043, Σ(1-γ)x=12.9957, Σ(1-γ)=3.0405",
      close(s1, 5.3043, 2e-4) and close(s2, 12.9957, 2e-4) and close((1 - gam).sum(), 3.0405), (s1, s2, (1 - gam).sum()))
check(T, "例2 μ1=1.792, μ2=4.274", close(m1, 1.792, 5e-4) and close(m2, 4.274, 5e-4), (m1, m2))
_, ll1 = estep(w1, m1, m2)
check(T, "例2 対数尤度 −12.051 → −10.187", close(ll0, -12.051, 5e-4) and close(ll1, -10.187, 5e-4), (ll0, ll1))
w, a1, a2 = 0.5, 1.0, 5.0
for _ in range(2000):
    gg, _ = estep(w, a1, a2)
    w, a1, a2 = gg.mean(), (gg * x).sum() / gg.sum(), ((1 - gg) * x).sum() / (1 - gg).sum()
_, llc = estep(w, a1, a2)
check(T, "例2 収束 w=0.495, μ1=1.892, μ2=4.186, ℓ=−10.166",
      close(w, 0.495, 5e-4) and close(a1, 1.892, 5e-4) and close(a2, 4.186, 5e-4) and close(llc, -10.166, 5e-4), (w, a1, a2, llc))
c = chart(T, "pdf")
p = c["params"]
check(T, "chart normmix の params = 収束値（小数第4位）",
      close(p["w"], w) and close(p["mu1"], a1) and close(p["mu2"], a2) and p["sigma1"] == 1 and p["sigma2"] == 1, p)
# EM の単調性（混合正規、乱数初期値20組）
mono = True
for _ in range(20):
    w, a1, a2 = rng.uniform(0.1, 0.9), rng.uniform(0, 6), rng.uniform(0, 6)
    prev = -np.inf
    for _ in range(50):
        gg, ll = estep(w, a1, a2)
        mono &= ll >= prev - 1e-10
        prev = ll
        w, a1, a2 = gg.mean(), (gg * x).sum() / gg.sum(), ((1 - gg) * x).sum() / (1 - gg).sum()
check(T, "EM で対数尤度が減らない（乱数初期値20組×50回）", mono, mono)

# ================================================================ censored-data
T = "censored-data"
yv = np.array([2, 3, 5, 7, 4, 6.0])
dv = np.array([1, 1, 1, 1, 0, 0])
lh = dv.sum() / yv.sum()
check(T, "例1 d=4, Σy=27, λ̂=0.1481", dv.sum() == 4 and yv.sum() == 27 and close(lh, 0.1481), lh)
res = optimize.minimize_scalar(lambda l: -np.sum(dv * np.log(l * np.exp(-l * yv)) + (1 - dv) * (-l * yv)), bounds=(1e-4, 2), method="bounded")
check(T, "尤度 Πf^δ S^(1−δ) の数値最大化 = d/Σy", close(res.x, 4 / 27, 1e-5), res.x)
ls = sp.symbols("l", positive=True)
logL = 4 * sp.log(ls) - 27 * ls
check(T, "2階微分 −d/λ²（sympy）", sp.simplify(sp.diff(logL, ls, 2) + 4 / ls**2) == 0, sp.diff(logL, ls, 2))
check(T, "例1 平均 6.75, メディアン 4.679", close(1 / lh, 6.75, 5e-3) and close(math.log(2) / lh, 4.679, 5e-4), (1 / lh, math.log(2) / lh))
se = lh / 2
ci = lh * np.exp(np.array([-1, 1]) * 1.96 / 2)
check(T, "例1 SE=0.0741, 区間 [0.0556, 0.3947]", close(se, 0.0741) and allclose(ci, [0.0556, 0.3947]), (se, ci))
check(T, "例1 誤った扱い 4/17=0.235, 6/27=0.222", close(4 / 17, 0.235, 5e-4) and close(6 / 27, 0.222, 5e-4), None)
v = math.exp(-0.3) - math.exp(-0.6)
check(T, "例2 S(2)−S(4)=0.192", close(v, 0.192, 5e-4), v)
# h^δ S = f^δ S^(1−δ)
f = lambda t: 0.3 * math.exp(-0.3 * t)
S = lambda t: math.exp(-0.3 * t)
check(T, "f^δ S^(1−δ) = h^δ S（数値）", all(abs(f(t) ** d * S(t) ** (1 - d) - (f(t) / S(t)) ** d * S(t)) < 1e-14 for t in yv for d in (0, 1)), True)
c = chart(T, "survival-hazard")
check(T, "chart expon rate=0.1481 とスライダー範囲", close(c["params"]["rate"], 4 / 27) and sliders_ok(c), c["params"])
check(T, "chart km-curve スライダー範囲", sliders_ok(chart(T, "km-curve")), chart(T, "km-curve")["params"])
# 乱数: λ̂=d/Σy の一致性と SE≈λ/√d
lt, nsim, ns = 0.2, 4000, 200
est, sev = [], []
for _ in range(nsim):
    tt = rng.exponential(1 / lt, ns)
    cc = rng.exponential(1 / 0.1, ns)
    yy = np.minimum(tt, cc)
    dd = (tt <= cc).sum()
    est.append(dd / yy.sum())
    sev.append(dd / yy.sum() / math.sqrt(dd))
est = np.array(est)
check(T, "乱数 打ち切りありの λ̂ の平均≈0.2（許容 0.003）, SD≈平均SE（許容 10%）",
      abs(est.mean() - lt) < 0.003 and abs(est.std() / np.mean(sev) - 1) < 0.1, (est.mean(), est.std(), np.mean(sev)))

# ================================================================ truncated-data
T = "truncated-data"
lt_ = optimize.brentq(lambda l: l / (1 - math.exp(-l)) - 1.6, 0.01, 5)
check(T, "例1 λ/(1−e^−λ)=1.6 の解 1.027", close(lt_, 1.027, 5e-4), lt_)
lsym = sp.symbols("l", positive=True)
xb = sp.Rational(8, 5)
ll = xb * sp.log(lsym) - lsym - sp.log(1 - sp.exp(-lsym))
eq = sp.diff(ll, lsym)
check(T, "対数尤度の微分 = x̄/λ − 1 − e^−λ/(1−e^−λ)（sympy）",
      sp.simplify(eq - (xb / lsym - 1 - sp.exp(-lsym) / (1 - sp.exp(-lsym)))) == 0, eq)
check(T, "例1 過大評価 約56%, e^−λ̂=0.358", close(1.6 / lt_ - 1, 0.56, 5e-3) and close(math.exp(-lt_), 0.358, 5e-4), (1.6 / lt_ - 1, math.exp(-lt_)))
c = chart(T, "line-data")
xs = np.arange(7)
pp = stats.poisson.pmf(xs, lt_)
pt = np.where(xs >= 1, pp / (1 - math.exp(-lt_)), 0)
check(T, "chart ポアソン分布 λ=1.027（7点, 小数第4位）", allclose(c["params"]["series"][0]["y"], np.round(pp, 4), 1e-4), np.round(pp, 4))
check(T, "chart 0切断ポアソン（7点）", allclose(c["params"]["series"][1]["y"], np.round(pt, 4), 1e-4), np.round(pt, 4))
check(T, "ゼロ切断ポアソンの期待値 λ/(1−e^−λ)", close((xs_ := np.arange(1, 200)) @ (stats.poisson.pmf(xs_, 1.3) / (1 - math.exp(-1.3))), 1.3 / (1 - math.exp(-1.3)), 1e-10), None)
# 左側切断の指数: λ̂ = d/Σ(y−a)
a_ = np.array([0, 1, 2, 0.5, 3.0])
y_ = np.array([3, 4, 6, 2, 5.0])
d_ = np.array([1, 0, 1, 1, 0])
nll = lambda l: -np.sum(d_ * np.log(l) - l * y_ + l * a_)
r_ = optimize.minimize_scalar(nll, bounds=(1e-4, 5), method="bounded")
check(T, "左側切断の指数分布 λ̂=d/Σ(y−a)（数値最大化）", close(r_.x, d_.sum() / (y_ - a_).sum(), 1e-5), r_.x)
# 乱数: ゼロ切断ポアソンの最尤推定
sim = rng.poisson(1.027, 400000)
sim = sim[sim >= 1]
lhat = optimize.brentq(lambda l: l / (1 - math.exp(-l)) - sim.mean(), 0.01, 5)
check(T, "乱数 ゼロ切断データの平均≈1.6, λ̂≈1.027（許容 0.01）", abs(sim.mean() - 1.6) < 0.01 and abs(lhat - 1.027) < 0.01, (sim.mean(), lhat))

# ================================================================ missing-data-mechanism / mar / mnar / mcar
k = stats.norm.pdf(0) / stats.norm.cdf(0)
T = "missing-data-mechanism"
check(T, "φ(0)/Φ(0)=0.7979", close(k, 0.7979), k)
eL, eH = 500 - 60 * k, 500 + 60 * k
check(T, "MAR E[Y|X≤0]=452.13, E[Y|X>0]=547.87", close(eL, 452.13, 5e-3) and close(eH, 547.87, 5e-3), (eL, eH))
mar_mean = (0.4 * eL + 0.1 * eH) / 0.5
check(T, "MAR 回答者平均 471.28", close(mar_mean, 471.28, 5e-3), mar_mean)
yL, yH = 500 - 100 * k, 500 + 100 * k
mnar_mean = (0.4 * yL + 0.1 * yH) / 0.5
check(T, "MNAR E[Y|Y≤500]=420.21, 579.79, 回答者平均 452.13", close(yL, 420.21, 5e-3) and close(yH, 579.79, 5e-3) and close(mnar_mean, 452.13, 5e-3), (yL, yH, mnar_mean))
c = chart(T, "bar-data")
check(T, "chart bar-data [500, 500, 471.28, 452.13]", allclose(c["params"]["values"], [500, 500, round(mar_mean, 2), round(mnar_mean, 2)], 1e-9), c["params"]["values"])
# 乱数: 3つの仕組み（n=40万、許容 0.6）
N = 400000
X = rng.standard_normal(N)
Y = 500 + 100 * (0.6 * X + 0.8 * rng.standard_normal(N))
u = rng.uniform(size=N)
r_mcar = u < 0.5
r_mar = u < np.where(X <= 0, 0.8, 0.2)
r_mnar = u < np.where(Y <= 500, 0.8, 0.2)
mm = [Y[r_mcar].mean(), Y[r_mar].mean(), Y[r_mnar].mean()]
check(T, "乱数 n=40万 観測者平均 ≈ 500/471.28/452.13（許容 0.6）", allclose(mm, [500, 471.28, 452.13], 0.6), np.round(mm, 2))
rew = 0.5 * Y[r_mar & (X <= 0)].mean() + 0.5 * Y[r_mar & (X > 0)].mean()
check(T, "乱数 MAR で X の層ごとに重みを戻すと ≈500（許容 0.6）", abs(rew - 500) < 0.6, rew)
for cid in ["missing-scatter"]:
    for term in ["missing-data-mechanism", "mar", "mcar", "mnar"]:
        p = chart(term, cid)["params"]
        check(term, f"chart missing-scatter mechanism/rho/n", p["mechanism"] == {"missing-data-mechanism": "mar"}.get(term, term) and p["rho"] == 0.6 and p["n"] == 200, p)

T = "mar"
check(T, "例1 回答者平均 471.28, 重み戻し 500.00", close((400 * 452.13 + 100 * 547.87) / 500, 471.28, 5e-3) and close(0.5 * 452.13 + 0.5 * 547.87, 500, 5e-3), None)
check(T, "例1 真の平均 0.5×452.13+0.5×547.87=500, 過小評価 約29万円", close(500 - 471.28, 28.72, 5e-3), 500 - 471.28)
check(T, "chart bar-data [500, 471.28, 500]", allclose(chart(T, "bar-data")["params"]["values"], [500, 471.28, 500], 1e-9), None)
# 乱数: 欠測が X だけに依存なら回帰係数は偏らない
b_cc = np.polyfit(X[r_mar], Y[r_mar], 1)
check(T, "乱数 MAR(X依存) の完全ケース回帰 傾き≈60, 切片≈500（許容 0.6）", abs(b_cc[0] - 60) < 0.6 and abs(b_cc[1] - 500) < 0.6, b_cc)

T = "mnar"
check(T, "例1 全体の回答率 0.5", close(0.5 * 0.8 + 0.5 * 0.2, 0.5, 1e-12), 0.5)
check(T, "例1 回答者平均 452.13", close((0.8 * yL + 0.2 * yH) / 1.0, 452.13, 5e-3), (0.8 * yL + 0.2 * yH))
nonresp = (0.2 * yL + 0.8 * yH)
check(T, "例1 無回答者の平均 547.87, δ=95.74", close(nonresp, 547.87, 5e-3) and close(nonresp - 452.13, 95.74, 5e-3), (nonresp, nonresp - mnar_mean))
check(T, "例1 δ=0,50,100 → 452.13, 477.13, 502.13", allclose([452.13 + 0.5 * d for d in (0, 50, 100)], [452.13, 477.13, 502.13], 1e-9), None)
check(T, "0.5×452.13+0.5×547.87=500（δ=95.74 で真値）", close(0.5 * mnar_mean + 0.5 * nonresp, 500, 1e-9), 0.5 * mnar_mean + 0.5 * nonresp)
c = chart(T, "line-data")
check(T, "chart line-data 452.13+0.5δ（7点）と真の平均", allclose(c["params"]["series"][0]["y"], [452.13 + 0.5 * d for d in c["params"]["x"]], 1e-9) and c["params"]["series"][1]["y"] == [500] * 7, c["params"]["series"][0]["y"])
nr = Y[~r_mnar].mean()
check(T, "乱数 無回答者の平均 ≈547.87（許容 0.6）", abs(nr - 547.87) < 0.6, nr)

T = "mcar"
check(T, "例1 SE 7.071 → 7.906, 比 1.118=1/√0.8", close(100 / math.sqrt(200), 7.071, 5e-4) and close(100 / math.sqrt(160), 7.906, 5e-4) and close(math.sqrt(200 / 160), 1.118, 5e-4) and close(1 / math.sqrt(0.8), 1.118, 5e-4), None)
se_w = math.sqrt(100 / 160 + 121 / 40)
tw = 3.8 / se_w
dfw = (100 / 160 + 121 / 40) ** 2 / ((100 / 160) ** 2 / 159 + (121 / 40) ** 2 / 39)
pw = 2 * stats.t.sf(tw, dfw)
check(T, "例2 SE=1.910, t=1.989", close(se_w, 1.910, 5e-4) and close(tw, 1.989, 5e-4), (se_w, tw))
check(T, "例2 ウェルチ p≈0.052（df=%.1f）" % dfw, abs(pw - 0.052) < 0.0015, pw)
r_sc = stats.ttest_ind_from_stats(45.0, 11, 40, 41.2, 10, 160, equal_var=False)
check(T, "例2 scipy ttest_ind_from_stats（Welch）と一致", close(r_sc.statistic, tw, 1e-10) and close(r_sc.pvalue, pw, 1e-10), (r_sc.statistic, r_sc.pvalue))
c = chart(T, "line-data")
check(T, "chart line-data 1/√(1−p)（7点）", allclose(c["params"]["series"][0]["y"], [round(1 / math.sqrt(1 - q), 3) for q in c["params"]["x"]], 1e-9), c["params"]["series"][0]["y"])
check(T, "欠測率50% で 1.414", close(1 / math.sqrt(0.5), 1.414, 5e-4), None)

# ================================================================ complete-case-analysis
T = "complete-case-analysis"
check(T, "例1 0.9^5=0.590, 0.9^10=0.349, 0.95^10=0.599", close(0.9**5, 0.590, 5e-4) and close(0.9**10, 0.349, 5e-4) and close(0.95**10, 0.599, 5e-4), (0.9**5, 0.9**10, 0.95**10))
check(T, "例2 471.28", close((400 * 452.13 + 100 * 547.87) / 500, 471.28, 5e-3), None)
c = chart(T, "line-data")
xk = np.array(c["params"]["x"])
check(T, "chart line-data (1−p)^k p=0.05, 0.10（11点×2）",
      allclose(c["params"]["series"][0]["y"], np.round(0.95**xk, 3), 1e-9) and allclose(c["params"]["series"][1]["y"], np.round(0.9**xk, 3), 1e-9), None)
M = rng.uniform(size=(200000, 5)) < 0.1
check(T, "乱数 5変数独立欠測10% で完全ケース割合≈0.590（許容 0.004）", abs((~M.any(1)).mean() - 0.9**5) < 0.004, (~M.any(1)).mean())

# ================================================================ single-imputation
T = "single-imputation"
xo = np.array([1, 2, 3, 4, 5, 7, 10.0])
yo = np.array([45, 52, 48, 60, 55, 68, 75.0])
check(T, "例1 観測平均 57.571, s_r²=117.619", close(yo.mean(), 57.571, 5e-4) and close(yo.var(ddof=1), 117.619, 5e-4), (yo.mean(), yo.var(ddof=1)))
ymean = np.concatenate([yo, [yo.mean()] * 3])
xall = np.concatenate([xo, [6, 8, 9]])
check(T, "例1 平均値代入の分散 78.413 = 6×s²/9", close(ymean.var(ddof=1), 78.413, 5e-4) and close(6 * yo.var(ddof=1) / 9, ymean.var(ddof=1), 1e-9), ymean.var(ddof=1))
r_o = np.corrcoef(xo, yo)[0, 1]
r_m = np.corrcoef(xall, ymean)[0, 1]
check(T, "例1 相関 0.950 → 0.795", close(r_o, 0.950, 5e-4) and close(r_m, 0.795, 5e-4), (r_o, r_m))
b1, b0 = np.polyfit(xo, yo, 1)
check(T, "例1 回帰 ŷ=42.386+3.322x", close(b0, 42.386, 5e-4) and close(b1, 3.322, 5e-4), (b0, b1))
pred = b0 + b1 * np.array([6, 8, 9.0])
check(T, "例1 代入値 62.317, 68.960, 72.282", allclose(pred, [62.317, 68.960, 72.282], 5e-4), pred)
yreg = np.concatenate([yo, pred])
check(T, "例1 回帰代入後 平均 60.656, 分散 108.801, 相関 0.964",
      close(yreg.mean(), 60.656, 5e-4) and close(yreg.var(ddof=1), 108.801, 5e-4) and close(np.corrcoef(xall, yreg)[0, 1], 0.964, 5e-4),
      (yreg.mean(), yreg.var(ddof=1), np.corrcoef(xall, yreg)[0, 1]))
check(T, "chart bar-data [117.62, 78.41, 108.8]", allclose(chart(T, "bar-data")["params"]["values"], [round(yo.var(ddof=1), 2), round(ymean.var(ddof=1), 2), round(yreg.var(ddof=1), 2)], 1e-9), chart(T, "bar-data")["params"]["values"])
# 一般式 (r−1)s²/(n−1)
z = rng.normal(size=13)
zf = np.concatenate([z, [z.mean()] * 7])
check(T, "平均値代入の分散=(r−1)s_r²/(n−1)（乱数 r=13, n=20）", close(zf.var(ddof=1), 12 * z.var(ddof=1) / 19, 1e-12), None)

# ================================================================ rubins-rules
T = "rubins-rules"


def rubin(Q, U):
    Q, U = np.asarray(Q, float), np.asarray(U, float)
    m = len(Q)
    qb, ub, B = Q.mean(), U.mean(), Q.var(ddof=1)
    Tt = ub + (1 + 1 / m) * B
    r = (1 + 1 / m) * B / ub
    nu = (m - 1) * (1 + 1 / r) ** 2
    return qb, ub, B, Tt, r, nu


Q = [52.1, 53.4, 51.6, 52.8, 53.1]
U = [1.21, 1.30, 1.18, 1.25, 1.26]
qb, ub, B, Tt, r, nu = rubin(Q, U)
check(T, "例1 Q̄=52.6, Ū=1.24, 和 263.0/6.20", close(qb, 52.6, 1e-9) and close(ub, 1.24, 1e-9) and close(sum(Q), 263.0, 1e-9) and close(sum(U), 6.2, 1e-9), (qb, ub))
dev = np.array(Q) - qb
check(T, "例1 偏差・平方和 2.18, B=0.545", allclose(dev, [-0.5, 0.8, -1.0, 0.2, 0.5], 1e-9) and close((dev**2).sum(), 2.18, 1e-9) and close(B, 0.545, 1e-9), B)
check(T, "例1 T=1.894, (1+1/m)B=0.654, SE=1.376", close(Tt, 1.894, 1e-9) and close(1.2 * B, 0.654, 1e-9) and close(math.sqrt(Tt), 1.376, 5e-4), (Tt, math.sqrt(Tt)))
check(T, "例1 r=0.527, ν=33.55", close(r, 0.527, 5e-4) and close(nu, 33.55, 5e-3), (r, nu))
tq = stats.t.ppf(0.975, nu)
ci = qb + np.array([-1, 1]) * tq * math.sqrt(Tt)
check(T, "例1 t_0.025(33.55)=2.033, 区間 [49.80, 55.40]", close(tq, 2.033, 5e-4) and allclose(ci, [49.80, 55.40], 5e-3), (tq, ci))
lamb = 1.2 * B / Tt
check(T, "例1 λ=0.345", close(lamb, 0.345, 5e-4), lamb)
t2 = (qb - 50) / math.sqrt(Tt)
p2 = 2 * stats.t.sf(t2, nu)
check(T, "例2 t=1.889, p=0.068", close(t2, 1.889, 5e-4) and close(p2, 0.068, 5e-4), (t2, p2))
check(T, "例2 √Ū=1.114, z=2.335（p<0.05）", close(math.sqrt(ub), 1.114, 5e-4) and close(2.6 / math.sqrt(ub), 2.335, 5e-4) and 2 * stats.norm.sf(2.6 / math.sqrt(ub)) < 0.05, 2.6 / math.sqrt(ub))
Q3 = np.log([1.8, 2.1, 1.9])
U3 = np.array([0.20, 0.21, 0.19]) ** 2
qb3, ub3, B3, T3, r3, nu3 = rubin(Q3, U3)
check(T, "例3 log OR 0.5878, 0.7419, 0.6419", allclose(Q3, [0.5878, 0.7419, 0.6419]), Q3)
check(T, "例3 Q̄=0.6572, Ū=0.04007, B=0.00612", close(qb3, 0.6572) and close(ub3, 0.04007, 5e-6) and close(B3, 0.00612, 5e-6), (qb3, ub3, B3))
check(T, "例3 T=0.04822, √T=0.2196", close(T3, 0.04822, 5e-6) and close(math.sqrt(T3), 0.2196), (T3, math.sqrt(T3)))
tq3 = stats.t.ppf(0.975, nu3)
check(T, "例3 r=0.204, ν=69.9, t=1.994", close(r3, 0.204, 5e-4) and close(nu3, 69.9, 0.05) and close(tq3, 1.994, 5e-4), (r3, nu3, tq3))
ci3 = np.exp(qb3 + np.array([-1, 1]) * tq3 * math.sqrt(T3))
check(T, "例3 e^Q̄=1.929, 区間 [1.245, 2.990], OR平均 1.933", close(math.exp(qb3), 1.929, 5e-4) and allclose(ci3, [1.245, 2.990], 5e-4) and close(np.mean([1.8, 2.1, 1.9]), 1.933, 5e-4), (math.exp(qb3), ci3))
c = chart(T, "bar-data")
check(T, "chart bar-data [Ū, 1.2B, T]", allclose(c["params"]["values"], [ub, 1.2 * B, Tt], 1e-9), c["params"]["values"])
# 乱数: 適切な多重代入（正規平均, MCAR）で Rubin の区間の被覆率≈0.95
cover, nrep, n0, m0 = 0, 2000, 60, 20
for _ in range(nrep):
    yy = rng.normal(10, 2, n0)
    obs = rng.uniform(size=n0) > 0.4
    yob = yy[obs]
    r0 = len(yob)
    Qs, Us = [], []
    for _ in range(m0):
        s2 = (r0 - 1) * yob.var(ddof=1) / rng.chisquare(r0 - 1)
        mu = rng.normal(yob.mean(), math.sqrt(s2 / r0))
        yimp = yy.copy()
        yimp[~obs] = rng.normal(mu, math.sqrt(s2), (~obs).sum())
        Qs.append(yimp.mean())
        Us.append(yimp.var(ddof=1) / n0)
    q_, u_, b_, t_, rr, nn = rubin(Qs, Us)
    cover += abs(q_ - 10) <= stats.t.ppf(0.975, nn) * math.sqrt(t_)
check(T, "乱数 適切な多重代入＋ルービンの規則の被覆率≈0.95（2000回, 許容 0.02）", abs(cover / nrep - 0.95) < 0.02, cover / nrep)

# ================================================================ multiple-imputation
T = "multiple-imputation"
qb, ub, B, Tt, r, nu = rubin([10.2, 10.8, 10.5], [0.40, 0.42, 0.38])
check(T, "例1 Q̄=10.5, Ū=0.40, B=0.09", close(qb, 10.5, 1e-9) and close(ub, 0.40, 1e-9) and close(B, 0.09, 1e-9), (qb, ub, B))
check(T, "例1 T=0.52, √T=0.721", close(Tt, 0.52, 1e-9) and close(math.sqrt(Tt), 0.721, 5e-4), math.sqrt(Tt))
check(T, "例1 r=0.30, ν=2×4.333²=37.56", close(r, 0.30, 1e-9) and close(1 + 1 / r, 4.333, 5e-4) and close(nu, 37.56, 5e-3), nu)
tq = stats.t.ppf(0.975, nu)
ci = qb + np.array([-1, 1]) * tq * math.sqrt(Tt)
check(T, "例1 t_0.025(37.56)=2.025, 区間 [9.040, 11.960]", close(tq, 2.025, 5e-4) and allclose(ci, [9.040, 11.960], 5e-4), (tq, ci))
check(T, "例1 √0.40=0.632", close(math.sqrt(0.4), 0.632, 5e-4), None)
check(T, "例2 RE m=5: 0.943, m=20: 0.985", close(1 / (1 + 0.3 / 5), 0.943, 5e-4) and close(1 / (1 + 0.3 / 20), 0.985, 5e-4), None)
c = chart(T, "line-data")
mx = np.array(c["params"]["x"], float)
ok = all(allclose(s["y"], np.round(1 / (1 + gm / mx), 4), 1e-9) for s, gm in zip(c["params"]["series"], [0.1, 0.3, 0.5]))
check(T, "chart line-data RE=(1+γ/m)^−1（γ=0.1,0.3,0.5 ×6点）", ok, [s["y"] for s in c["params"]["series"]])

# ================================================================ kaplan-meier-estimator
T = "kaplan-meier-estimator"
tkm = np.array([2, 3, 4, 4, 6, 7, 9, 11, 12, 14.0])
dkm = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0])


def km_table(t, d):
    out = []
    S = 1.0
    gsum = 0.0
    for u in np.unique(t[d == 1]):
        n = (t >= u).sum()
        e = ((t == u) & (d == 1)).sum()
        S *= 1 - e / n
        gsum += e / (n * (n - e)) if n > e else np.inf
        out.append((u, n, e, 1 - e / n, S, e / (n * (n - e)) if n > e else np.inf, gsum))
    return out


tab = km_table(tkm, dkm)
check(T, "例1 表 t, n_j, d_j", [(r_[0], r_[1], r_[2]) for r_ in tab] == [(2, 10, 1), (4, 8, 2), (6, 6, 1), (9, 4, 1), (12, 2, 1)], [(r_[0], r_[1], r_[2]) for r_ in tab])
check(T, "例1 表 1−d/n", allclose([r_[3] for r_ in tab], [0.9, 0.75, 0.8333, 0.75, 0.5]), [r_[3] for r_ in tab])
check(T, "例1 表 Ŝ", allclose([r_[4] for r_ in tab], [0.9, 0.675, 0.5625, 0.4219, 0.2109]), [r_[4] for r_ in tab])
sres = stats.ecdf(stats.CensoredData(uncensored=tkm[dkm == 1], right=tkm[dkm == 0]))
check(T, "scipy.stats.ecdf（打ち切りあり）の生存関数と一致", allclose(sres.sf.evaluate([2, 4, 6, 9, 12]), [r_[4] for r_ in tab], 1e-12), sres.sf.evaluate([2, 4, 6, 9, 12]))
check(T, "例1 中央値 9（Ŝ≤0.5 の最小 t）", [r_[0] for r_ in tab if r_[4] <= 0.5][0] == 9, None)
check(T, "例2 誤った扱い 0.600, 0.333 と正しい 0.5625",
      close(1 - 4 / 10, 0.6, 1e-9) and close(2 / 6, 0.333, 5e-4) and (tkm[(dkm == 1) & (tkm <= 6)].size == 4) and (tkm[dkm == 1] > 6).sum() == 2 and 2 / 6 < 0.5625 < 0.6, None)
c = chart(T, "line-data")
xs_c = c["params"]["x"]
ys_c = c["params"]["series"][0]["y"]


def km_at(tq_, right_limit):
    S = 1.0
    for r_ in tab:
        if r_[0] < tq_ or (right_limit and r_[0] == tq_):
            S = r_[4]
    return S


# 階段データ：同じ x の2点目は右極限、1点目は左極限
exp_y = []
for i, xv in enumerate(xs_c):
    right = i > 0 and xs_c[i - 1] == xv
    exp_y.append(round(km_at(xv, right or xv in (0, 14)), 4))
check(T, "chart line-data 階段（12点）が Ŝ と一致", allclose(ys_c, exp_y, 1e-9), exp_y)
check(T, "chart km-curve スライダー範囲", sliders_ok(chart(T, "km-curve")), chart(T, "km-curve")["params"])
# 打ち切りなしなら経験生存関数
tt = rng.exponential(5, 30)
S_ = np.prod([1 - 1 / (tt >= u).sum() for u in np.sort(tt) if u <= 4])
check(T, "打ち切りなしなら経験生存関数に一致（乱数30点）", close(S_, (tt > 4).mean(), 1e-12), (S_, (tt > 4).mean()))
# 乱数: KM の一致性
nn_ = 4000
tt = rng.exponential(10, nn_)
cc = rng.exponential(25, nn_)
yy = np.minimum(tt, cc)
dd = (tt <= cc).astype(int)
ek = stats.ecdf(stats.CensoredData(uncensored=yy[dd == 1], right=yy[dd == 0]))
check(T, "乱数 n=4000 指数（平均10）+打ち切り で Ŝ(10)≈e^−1（許容 0.02）", abs(ek.sf.evaluate(10) - math.exp(-1)) < 0.02, ek.sf.evaluate(10))

# ================================================================ greenwood-formula
T = "greenwood-formula"
gw = [r_[5] for r_ in tab]
cs = [r_[6] for r_ in tab]
Sv = np.array([r_[4] for r_ in tab])
SE = Sv * np.sqrt(cs)
check(T, "例1 表 d/(n(n−d))", allclose(gw, [0.0111, 0.0417, 0.0333, 0.0833, 0.5]), np.round(gw, 4))
check(T, "例1 表 累積和", allclose(cs, [0.0111, 0.0528, 0.0861, 0.1694, 0.6694]), np.round(cs, 4))
check(T, "例1 表 SE", allclose(SE, [0.0949, 0.1551, 0.1651, 0.1737, 0.1726]), np.round(SE, 4))
lo = np.clip(Sv - 1.96 * SE, 0, 1)
hi = np.clip(Sv + 1.96 * SE, 0, 1)
check(T, "例1 表 95%区間（0,1で切る, 小数第3位）", allclose(lo, [0.714, 0.371, 0.239, 0.082, 0], 5e-4) and allclose(hi, [1, 0.979, 0.886, 0.762, 0.549], 5e-4), (np.round(lo, 4), np.round(hi, 4)))
check(T, "t=6 の途中式 0.08611, √=0.2934, SE=0.1651", close(cs[2], 0.08611, 5e-6) and close(math.sqrt(cs[2]), 0.2934) and close(SE[2], 0.1651), None)
sse = sres.sf.confidence_interval(0.95)  # scipy の linear 区間（グリーンウッド）
check(T, "scipy ecdf の linear 信頼区間（グリーンウッド）と一致", allclose(sse.low.evaluate([2, 4, 6, 9, 12]), lo, 1e-3) and allclose(sse.high.evaluate([2, 4, 6, 9, 12]), hi, 1e-3),
      (sse.low.evaluate([2, 4, 6, 9, 12]), sse.high.evaluate([2, 4, 6, 9, 12])))
# log-log 区間 t=6
se_ll = math.sqrt(cs[2]) / abs(math.log(Sv[2]))
llci = sorted(Sv[2] ** np.exp(np.array([-1, 1]) * 1.96 * se_ll))
check(T, "t=6 log-log 区間 [0.209, 0.809]", allclose(llci, [0.209, 0.809], 5e-4), llci)
sll = sres.sf.confidence_interval(0.95, method="log-log")
check(T, "scipy ecdf の log-log 区間と一致（t=6）", allclose([sll.low.evaluate(6), sll.high.evaluate(6)], llci, 1e-3), (sll.low.evaluate(6), sll.high.evaluate(6)))
# log-log 尺度の分散（デルタ法）を sympy で
Ssym, vs = sp.symbols("S v", positive=True)
gfun = sp.log(-sp.log(Ssym))
dvar = sp.simplify(sp.diff(gfun, Ssym) ** 2 * Ssym**2 * vs)
check(T, "log(−log Ŝ) の分散 = Σ/(log Ŝ)²（sympy, Var Ŝ=S²Σ）", sp.simplify(dvar - vs / sp.log(Ssym) ** 2) == 0, dvar)
# 打ち切りなしで S(1−S)/n
tt = np.arange(1, 21, dtype=float)
nj = np.arange(20, 0, -1)
Sx = np.prod(1 - 1 / nj[:7])
gsx = Sx**2 * np.sum(1 / (nj[:7] * (nj[:7] - 1)))
check(T, "打ち切りなしなら Ŝ²Σ = Ŝ(1−Ŝ)/n（n=20, 7件後）", close(gsx, Sx * (1 - Sx) / 20, 1e-12), (gsx, Sx * (1 - Sx) / 20))
c = chart(T, "line-data")
xs_c = c["params"]["x"]
ser = c["params"]["series"]


def step(vals, xv, right):
    v = 1.0
    for r_, val in zip(tab, vals):
        if r_[0] < xv or (right and r_[0] == xv):
            v = val
    return v


ok = True
exp_all = [[], [], []]
for i, xv in enumerate(xs_c):
    right = (i > 0 and xs_c[i - 1] == xv) or xv in (0, 14)
    for j, vals in enumerate([Sv, lo, hi]):
        exp_all[j].append(round(step(vals, xv, right), 4))
check(T, "chart line-data KM・95%下限・上限（12点×3）", all(allclose(s["y"], e, 1e-9) for s, e in zip(ser, exp_all)), exp_all)
# 乱数: グリーンウッドの SE ≈ Ŝ の実際の SD
ests, ses = [], []
for _ in range(3000):
    t1 = rng.exponential(10, 100)
    c1 = rng.uniform(0, 30, 100)
    y1 = np.minimum(t1, c1)
    d1 = t1 <= c1
    S, gs = 1.0, 0.0
    for u in np.sort(y1[d1]):
        if u > 8:
            break
        n = (y1 >= u).sum()
        S *= 1 - 1 / n
        gs += 1 / (n * (n - 1))
    ests.append(S)
    ses.append(S * math.sqrt(gs))
check(T, "乱数 グリーンウッド SE の平均 ≈ Ŝ(8) の SD（3000回, 許容 5%）", abs(np.mean(ses) / np.std(ests) - 1) < 0.05, (np.mean(ses), np.std(ests)))

# ================================================================ log-rank-test
T = "log-rank-test"
tA = np.array([3, 5, 6, 8, 9, 10.0]); dA = np.array([1, 1, 1, 0, 1, 1])
tB = np.array([4, 7, 11, 12, 13, 15.0]); dB = np.array([0, 1, 1, 0, 1, 0])
tt = np.concatenate([tA, tB]); dd = np.concatenate([dA, dB]); gA = np.r_[np.ones(6), np.zeros(6)].astype(bool)
rows = []
for u in np.unique(tt[dd == 1]):
    risk = tt >= u
    n = risk.sum(); na = (risk & gA).sum()
    d = ((tt == u) & (dd == 1)).sum(); da = ((tt == u) & (dd == 1) & gA).sum()
    E = d * na / n
    V = d * (na / n) * (1 - na / n) * (n - d) / (n - 1) if n > 1 else 0.0
    rows.append((u, n, na, n - na, d, da, E, V))
check(T, "例1 表 t, n, n_A, n_B, d, d_A", [tuple(int(v) for v in r_[:6]) for r_ in rows] ==
      [(3, 12, 6, 6, 1, 1), (5, 10, 5, 5, 1, 1), (6, 9, 4, 5, 1, 1), (7, 8, 3, 5, 1, 0), (9, 6, 2, 4, 1, 1), (10, 5, 1, 4, 1, 1), (11, 4, 0, 4, 1, 0), (13, 2, 0, 2, 1, 0)],
      [tuple(int(v) for v in r_[:6]) for r_ in rows])
check(T, "例1 表 E_A", allclose([r_[6] for r_ in rows], [0.5, 0.5, 0.4444, 0.375, 0.3333, 0.2, 0, 0]), np.round([r_[6] for r_ in rows], 4))
check(T, "例1 表 V", allclose([r_[7] for r_ in rows], [0.25, 0.25, 0.2469, 0.2344, 0.2222, 0.16, 0, 0]), np.round([r_[7] for r_ in rows], 4))
OA = sum(r_[5] for r_ in rows); EA = sum(r_[6] for r_ in rows); VV = sum(r_[7] for r_ in rows)
check(T, "例1 合計 O_A=5, E_A=2.3528, V=1.3635", OA == 5 and close(EA, 2.3528) and close(VV, 1.3635), (OA, EA, VV))
chi = (OA - EA) ** 2 / VV
check(T, "例1 (O−E)²=7.0077, χ²=5.140", close((OA - EA) ** 2, 7.0077, 1e-4) and close(chi, 5.140, 5e-4), chi)
pl = stats.chi2.sf(chi, 1)
check(T, "例1 p=0.023", close(pl, 0.023, 5e-4), pl)
lr = stats.logrank(stats.CensoredData(uncensored=tA[dA == 1], right=tA[dA == 0]), stats.CensoredData(uncensored=tB[dB == 1], right=tB[dB == 0]))
check(T, "scipy.stats.logrank と一致（統計量²=χ², p）", close(lr.statistic**2, chi, 1e-8) and close(lr.pvalue, pl, 1e-8), (lr.statistic, lr.pvalue))
EB = 8 - EA
simp = (5 - EA) ** 2 / EA + (3 - EB) ** 2 / EB
check(T, "例1 O_B=3, E_B=5.6472, 簡便法 4.219", close(EB, 5.6472) and close(simp, 4.219, 5e-4) and simp < chi, (EB, simp))
# スコア検定（コックス, β=0）= ログランク
xg = gA.astype(float)


def cox_UI(beta, t, d, x):
    U = I = ll = 0.0
    for i in np.where(d == 1)[0]:
        R = t >= t[i]
        w = np.exp(beta * x[R])
        xb = (w * x[R]).sum() / w.sum()
        x2 = (w * x[R] ** 2).sum() / w.sum()
        U += x[i] - xb
        I += x2 - xb**2
        ll += beta * x[i] - math.log(w.sum())
    return U, I, ll


U0, I0, _ = cox_UI(0.0, tt, dd, xg)
check(T, "タイなしなら コックスのスコア U(0)=O−E, I(0)=V", close(U0, OA - EA, 1e-12) and close(I0, VV, 1e-12), (U0, I0))
c = chart(T, "line-data")
xs_c = c["params"]["x"]


def km_curve(t, d):
    out = []
    S = 1.0
    for u in np.unique(t[d == 1]):
        n = (t >= u).sum(); e = ((t == u) & (d == 1)).sum()
        S *= 1 - e / n
        out.append((u, S))
    return out


def km_eval(curve, xv, right):
    v = 1.0
    for u, S in curve:
        if u < xv or (right and u == xv):
            v = S
    return v


ok = True
for s, (t_, d_) in zip(c["params"]["series"], [(tA, dA), (tB, dB)]):
    cur = km_curve(t_, d_)
    e = [round(km_eval(cur, xv, (i > 0 and xs_c[i - 1] == xv) or xv in (0, 15)), 4) for i, xv in enumerate(xs_c)]
    ok &= allclose(s["y"], e, 1e-9)
check(T, "chart line-data A群・B群の KM 曲線（18点×2）", ok, None)
check(T, "chart km-curve スライダー範囲", sliders_ok(chart(T, "km-curve")), chart(T, "km-curve")["params"])
# 乱数: H0 でのログランクの大きさ
rej = 0
for _ in range(2000):
    a = rng.exponential(10, 40); b = rng.exponential(10, 40)
    ca = rng.uniform(0, 30, 40); cb = rng.uniform(0, 30, 40)
    ya, yb = np.minimum(a, ca), np.minimum(b, cb)
    da, db = a <= ca, b <= cb
    res = stats.logrank(stats.CensoredData(uncensored=ya[da], right=ya[~da]), stats.CensoredData(uncensored=yb[db], right=yb[~db]))
    rej += res.pvalue < 0.05
check(T, "乱数 H0 で棄却率≈0.05（2000回, 許容 0.012）", abs(rej / 2000 - 0.05) < 0.012, rej / 2000)

# ================================================================ partial-likelihood
T = "partial-likelihood"
tp = np.array([1, 2, 3, 4, 5.0]); dp = np.array([1, 1, 1, 0, 1]); xp = np.array([1, 0, 1, 0, 0.0])
b = sp.symbols("beta", real=True)
u_ = sp.exp(b)
Lsym = sp.exp(2 * b) / ((2 * u_ + 3) * (u_ + 3) * (u_ + 2))
_, _, ll_num = cox_UI(0.7, tp, dp, xp)
check(T, "L(β) の閉じた式 = 部分尤度の定義（β=0.7）", close(float(sp.log(Lsym).subs(b, 0.7)), ll_num, 1e-12), ll_num)
check(T, "log L(0) = −4.0943", close(math.log(1 / 60), -4.0943), math.log(1 / 60))
U0, I0, _ = cox_UI(0.0, tp, dp, xp)
check(T, "U(0)=1.0167, I(0)=0.6497", close(U0, 1.0167) and close(I0, 0.6497) and close(I0, 0.24 + 0.1875 + 2 / 9, 1e-12), (U0, I0))
lsym = sp.log(Lsym)
Us = sp.diff(lsym, b); Is = -sp.diff(lsym, b, 2)
check(T, "スコア・情報量の式 = sympy の微分（β=0, 1.2）", all(close(float(Us.subs(b, v)), cox_UI(v, tp, dp, xp)[0], 1e-12) and close(float(Is.subs(b, v)), cox_UI(v, tp, dp, xp)[1], 1e-12) for v in (0, 1.2)), None)
bk, its = 0.0, []
for _ in range(10):
    U_, I_, _ = cox_UI(bk, tp, dp, xp)
    bk = bk + U_ / I_
    its.append(bk)
check(T, "ニュートン法 β(1)=1.5648, β(2)=1.4359, β(3)=1.4388", allclose(its[:3], [1.5648, 1.4359, 1.4388]), np.round(its[:3], 4))
bh = its[-1]
check(T, "β̂=1.4388, HR=4.215", close(bh, 1.4388) and close(math.exp(bh), 4.215, 5e-4), (bh, math.exp(bh)))
Ih = cox_UI(bh, tp, dp, xp)[1]
check(T, "I(β̂)=0.6547, SE=1.236, z=1.164, p=0.244", close(Ih, 0.6547) and close(1 / math.sqrt(Ih), 1.236, 5e-4) and close(bh * math.sqrt(Ih), 1.164, 5e-4) and close(2 * stats.norm.sf(bh * math.sqrt(Ih)), 0.244, 5e-4),
      (Ih, 1 / math.sqrt(Ih), bh * math.sqrt(Ih), 2 * stats.norm.sf(bh * math.sqrt(Ih))))
sc = U0**2 / I0
check(T, "スコア検定 1.591, p=0.207", close(sc, 1.591, 5e-4) and close(stats.chi2.sf(sc, 1), 0.207, 5e-4), (sc, stats.chi2.sf(sc, 1)))
llh = cox_UI(bh, tp, dp, xp)[2]
lrt = 2 * (llh - math.log(1 / 60))
check(T, "尤度比検定 log L(β̂)=−3.3621, 1.465, p=0.226", close(llh, -3.3621) and close(lrt, 1.465, 5e-4) and close(stats.chi2.sf(lrt, 1), 0.226, 5e-4), (llh, lrt, stats.chi2.sf(lrt, 1)))
c = chart(T, "line-data")
xb_ = c["params"]["x"]
check(T, "chart line-data log L(β)（8点）", allclose(c["params"]["series"][0]["y"], [round(cox_UI(v, tp, dp, xp)[2], 4) for v in xb_], 1e-9), [round(cox_UI(v, tp, dp, xp)[2], 4) for v in xb_])
tr = tp ** 3 + 2
bt = 0.0
for _ in range(20):
    U_, I_, _ = cox_UI(bt, tr, dp, xp)
    bt += U_ / I_
check(T, "時間を単調変換しても β̂ は不変", close(bt, bh, 1e-10), bt)
# 乱数: 比例ハザードの β の推定
Nn = 3000
xx = rng.integers(0, 2, Nn).astype(float)
tt_ = rng.exponential(1 / (0.1 * np.exp(0.7 * xx)))
cc_ = rng.exponential(20, Nn)
yy_ = np.minimum(tt_, cc_); dd_ = (tt_ <= cc_).astype(int)
order = np.argsort(-yy_)
ys, ds, xs2 = yy_[order], dd_[order], xx[order]
bb = 0.0
for _ in range(15):
    w = np.exp(bb * xs2)
    S0 = np.cumsum(w); S1 = np.cumsum(w * xs2)
    m_ = S1 / S0
    U_ = np.sum(ds * (xs2 - m_)); I_ = np.sum(ds * (m_ - m_**2))
    bb += U_ / I_
check(T, "乱数 n=3000 で β̂≈0.7（許容 0.1≈2.5SE）", abs(bb - 0.7) < 0.1, (bb, 1 / math.sqrt(I_)))

# ================================================================ cox-proportional-hazards-model
T = "cox-proportional-hazards-model"
check(T, "例1 HR=e^0.405=1.499, z=2.70", close(math.exp(0.405), 1.499, 5e-4) and close(0.405 / 0.15, 2.70, 5e-3), math.exp(0.405))
ci = np.exp(0.405 + np.array([-1, 1]) * 1.96 * 0.15)
check(T, "例1 区間 [1.117, 2.012]", allclose(ci, [1.117, 2.012], 5e-4), ci)
check(T, "例2 0.8^2=0.640, 0.8^0.5=0.894", close(0.8**2, 0.64, 1e-9) and close(0.8**0.5, 0.894, 5e-4), None)
check(T, "例3 e^0.03=1.030, e^0.3=1.350", close(math.exp(0.03), 1.030, 5e-4) and close(math.exp(0.3), 1.350, 5e-4), None)
c = chart(T, "line-data")
tg = np.array(c["params"]["x"], float)
S0 = np.exp(-(tg / 10) ** 1.5)
check(T, "chart line-data S0^HR（HR=0.5,1,2 ×11点）", all(allclose(s["y"], np.round(S0**h, 4), 1e-9) for s, h in zip(c["params"]["series"], [0.5, 1, 2])), None)
cw = chart(T, "survival-hazard")
check(T, "chart weibull shape=1.5, scale=10 とスライダー範囲", cw["params"] == {"shape": 1.5, "scale": 10} and sliders_ok(cw), cw["params"])
H0 = sp.symbols("H", positive=True); bsym = sp.symbols("b", real=True)
check(T, "S(t|x)=exp(−H0 e^βx)=S0^exp(βx)（sympy, H0>0）", sp.simplify(sp.exp(-H0 * sp.exp(bsym)) - sp.exp(-H0) ** sp.exp(bsym)) == 0, None)
check(T, "log(−log S) の差 = β（HR=2 で log2, t=1..20）", np.allclose(np.log(-np.log(S0[1:] ** 2)) - np.log(-np.log(S0[1:])), math.log(2)), None)

# ================================================================ hazard-ratio
T = "hazard-ratio"
l0, l1 = 20 / 400, 12 / 480
check(T, "例1 λ0=0.050, λ1=0.025, HR=0.50", close(l0, 0.05, 1e-12) and close(l1, 0.025, 1e-12) and close(l1 / l0, 0.5, 1e-12), (l0, l1))
check(T, "例1 中央値 13.86, 27.73, 比 0.5", close(math.log(2) / l0, 13.86, 5e-3) and close(math.log(2) / l1, 27.73, 5e-3), (math.log(2) / l0, math.log(2) / l1))
se = math.sqrt(1 / 12 + 1 / 20)
ci = 0.5 * np.exp(np.array([-1, 1]) * 1.96 * se)
check(T, "例1 SE=0.3651, 区間 [0.244, 1.023], z=−1.898", close(se, 0.3651) and allclose(ci, [0.244, 1.023], 5e-4) and close(math.log(0.5) / se, -1.898, 5e-4), (se, ci, math.log(0.5) / se))
check(T, "例2 RR=1.80, OR=2.25", close(0.36 / 0.2, 1.8, 1e-9) and close((0.36 / 0.64) / (0.2 / 0.8), 2.25, 1e-9), None)
c = chart(T, "line-data")
tg = np.array(c["params"]["x"], float)
h0 = 1.5 / 10 * (tg / 10) ** 0.5
check(T, "chart line-data h0 と 2h0（ワイブル形状1.5・尺度10, 11点×2）",
      allclose(c["params"]["series"][0]["y"], np.round(h0, 4), 1e-9) and allclose(c["params"]["series"][1]["y"], np.round(2 * h0, 4), 1e-9), None)
# 乱数: SE(log HR) ≈ √(1/d1+1/d0)（漸近式なので期待件数を10倍の 120・200 にして確認）
d0 = rng.poisson(200, 20000); d1 = rng.poisson(120, 20000)
lh_ = np.log((d1 / 4800) / (d0 / 4000))
se10 = math.sqrt(1 / 120 + 1 / 200)
check(T, "乱数 log HR の SD ≈ √(1/d1+1/d0)（件数120・200, 2万回, 許容 3%）", abs(np.std(lh_) / se10 - 1) < 0.03, (np.std(lh_), se10))

# ================================================================ 重要度Cの用語
# ---------------------------------------------------------------- nelson-aalen-estimator
T = "nelson-aalen-estimator"
tt_ = np.array([2, 3, 4, 4, 6, 7, 9, 11, 12, 14])
ev_ = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0])
rows_ = []
H_, S_, V_ = 0.0, 1.0, 0.0
for u_ in sorted(set(tt_[ev_ == 1])):
    nj_ = int((tt_ >= u_).sum())
    dj_ = int(((tt_ == u_) & (ev_ == 1)).sum())
    H_ += dj_ / nj_
    S_ *= 1 - dj_ / nj_
    V_ += dj_ / nj_ ** 2
    rows_.append((u_, nj_, dj_, dj_ / nj_, H_, math.exp(-H_), S_, -math.log(S_), V_))
R_ = np.array(rows_, float)
check(T, "表の t・n_j・d_j（2,4,6,9,12 / 10,8,6,4,2 / 1,2,1,1,1）",
      R_[:, 0].tolist() == [2, 4, 6, 9, 12] and R_[:, 1].tolist() == [10, 8, 6, 4, 2] and R_[:, 2].tolist() == [1, 2, 1, 1, 1], R_[:, :3].tolist())
check(T, "表の d_j/n_j", allclose(R_[:, 3], [0.1, 0.25, 0.1667, 0.25, 0.5]), np.round(R_[:, 3], 4).tolist())
check(T, "表の Ĥ(t_(j))", allclose(R_[:, 4], [0.1, 0.35, 0.5167, 0.7667, 1.2667]), np.round(R_[:, 4], 4).tolist())
check(T, "表の e^{−Ĥ}", allclose(R_[:, 5], [0.9048, 0.7047, 0.5965, 0.4646, 0.2818]), np.round(R_[:, 5], 4).tolist())
check(T, "表の Ŝ_KM（kaplan-meier-estimator の値 0.9,0.675,0.5625,0.4219,0.2109 と一致）",
      allclose(R_[:, 6], [0.9, 0.675, 0.5625, 0.4219, 0.2109]), np.round(R_[:, 6], 4).tolist())
km_ = (ROOT / "content" / "ch29" / "kaplan-meier-estimator.md").read_text(encoding="utf-8")
check(T, "kaplan-meier-estimator.md と同じデータ列・同じ推定値", "2,\\ 3^+,\\ 4,\\ 4,\\ 6,\\ 7^+,\\ 9,\\ 11^+,\\ 12,\\ 14^+" in km_
      and all(s in km_ for s in ["0.6750", "0.5625", "0.4219", "0.2109"]), None)
check(T, "t=12 の分散の各項 0.01, 0.03125, 0.02778, 0.0625, 0.25、和 0.3815、SE 0.618",
      allclose(R_[:, 2] / R_[:, 1] ** 2, [0.01, 0.03125, 0.02778, 0.0625, 0.25], 5e-6) and close(V_, 0.3815) and close(math.sqrt(V_), 0.618, 5e-4), (V_, math.sqrt(V_)))
check(T, "e^{−Ĥ} ≥ Ŝ_KM（全時刻）、差は t=12 で最大", bool(np.all(R_[:, 5] >= R_[:, 6])) and int(np.argmax(R_[:, 5] - R_[:, 6])) == 4, np.round(R_[:, 5] - R_[:, 6], 4).tolist())
check(T, "log(1−x) < −x（0<x<1, sympy で差の導関数が負・x→0 で 0）",
      (lambda xx: sp.simplify(sp.diff(sp.log(1 - xx) + xx, xx) + xx / (1 - xx)) == 0)(sp.symbols("x", positive=True)), None)
check(T, "累積ハザードは 1 を超えうる（1.267）", close(H_, 1.2667) and H_ > 1, H_)
c_ = chart(T, "line-data")
xp_ = c_["params"]["x"]
na_y = [s for s in c_["params"]["series"] if s["label"].startswith("ネルソン")][0]["y"]
lk_y = [s for s in c_["params"]["series"] if s["label"].startswith("−log")][0]["y"]
exp_x = [0] + [v for t in [2, 4, 6, 9, 12] for v in (t, t)] + [14]
exp_na = [0, 0] + [float(h) for h in np.round(R_[:, 4], 4) for _ in range(2)]
exp_lk = [0, 0] + [float(h) for h in np.round(R_[:, 7], 4) for _ in range(2)]
check(T, "chart line-data x=[0,2,2,4,4,...,12,12,14]", xp_ == exp_x, xp_)
check(T, "chart ネルソン・アーレンの階段 y", allclose(na_y, exp_na, 1e-9), na_y)
check(T, "chart −log(KM) の階段 y（0.1054,0.393,0.5754,0.863,1.5562）", allclose(lk_y, exp_lk, 1e-9), lk_y)
# 乱数：指数分布（λ=0.1）＋一様打ち切りで Ĥ(t)≈λt、分散推定 Σd/n² ≈ 実際の分散
lam_ = 0.1
Hs_, Vs_ = [], []
for _ in range(4000):
    tv = rng.exponential(1 / lam_, 200)
    cv = rng.uniform(0, 30, 200)
    ob = np.minimum(tv, cv)
    dd = tv <= cv
    o = np.argsort(ob)
    ob, dd = ob[o], dd[o]
    nr = 200 - np.arange(200)
    m = (ob <= 10) & dd
    Hs_.append(np.sum(1 / nr[m]))
    Vs_.append(np.sum(1 / nr[m] ** 2))
Hs_ = np.array(Hs_)
check(T, "乱数4000回（指数 λ=0.1, n=200, 打ち切りあり）：Ĥ(10) の平均≈1（許容 0.01）、分散≈平均の Σd/n²（相対 8%）",
      abs(Hs_.mean() - 1) < 0.01 and abs(Hs_.var() / np.mean(Vs_) - 1) < 0.08, (round(Hs_.mean(), 4), round(Hs_.var(), 5), round(np.mean(Vs_), 5)))

# ---------------------------------------------------------------- ignorability
T = "ignorability"
check(T, "回答者 A 90 人・B 50 人（回答率 0.9, 0.5）、はい 27・30（割合 0.3, 0.6）",
      close(100 * 0.9, 90, 1e-12) and close(100 * 0.5, 50, 1e-12) and close(90 * 0.3, 27, 1e-12) and close(50 * 0.6, 30, 1e-12), None)
check(T, "回答者全体の割合 57/140≈0.4071", close(57 / 140, 0.4071), 57 / 140)
check(T, "グループ別 MLE 0.3, 0.6、全体 0.5×0.3+0.5×0.6=0.450", close(27 / 90, 0.3, 1e-12) and close(30 / 50, 0.6, 1e-12) and close(0.5 * 0.3 + 0.5 * 0.6, 0.45, 1e-12), None)
# 尤度の分解：θ の MLE が φ によらない（数値最大化で確認）
def negll_(par, phiA, phiB):
    tA, tB = par
    return -(27 * math.log(tA) + 63 * math.log(1 - tA) + 30 * math.log(tB) + 20 * math.log(1 - tB)
             + 90 * math.log(phiA) + 10 * math.log(1 - phiA) + 50 * math.log(phiB) + 50 * math.log(1 - phiB))
ests_ = [optimize.minimize(negll_, [0.5, 0.5], args=(pa, pb), bounds=[(1e-6, 1 - 1e-6)] * 2).x for pa, pb in [(0.9, 0.5), (0.5, 0.5), (0.2, 0.8)]]
check(T, "観測データ尤度の θ の最尤値は φ の値によらず (0.3, 0.6)", all(allclose(e, [0.3, 0.6], 1e-4) for e in ests_), np.round(ests_, 5).tolist())
# 乱数：MAR 下で多数回
gA = rng.random((20000, 100)) < 0.3
gB = rng.random((20000, 100)) < 0.6
rA = rng.random((20000, 100)) < 0.9
rB = rng.random((20000, 100)) < 0.5
cc_mean = ((gA & rA).sum(1) + (gB & rB).sum(1)) / (rA.sum(1) + rB.sum(1))
ml_mean = 0.5 * (gA & rA).sum(1) / rA.sum(1) + 0.5 * (gB & rB).sum(1) / rB.sum(1)
check(T, "乱数2万回：回答者平均は≈0.407 に偏り、尤度に基づく推定は≈0.45（許容 0.003）",
      abs(cc_mean.mean() - (0.27 + 0.30) / 1.4) < 0.003 and abs(ml_mean.mean() - 0.45) < 0.003, (round(cc_mean.mean(), 4), round(ml_mean.mean(), 4)))

# ================================================================ 集計
n_ok = sum(results)
n_ng = len(results) - n_ok
print(f"\n合計: OK {n_ok} / NG {n_ng}")

# --- 重要度A への加筆: complete-case-analysis ---
T = "complete-case-analysis"
_m = np.r_[np.full(500, 452.13), np.full(500, 547.87)]
_sd = 60.0
_reps = 4000; _b0 = np.empty(_reps); _b1 = np.empty(_reps); _mcc = np.empty(_reps); _mw = np.empty(_reps)
_x = np.r_[np.zeros(500), np.ones(500)]
for _r in range(_reps):
    _y = _m + rng.normal(0, _sd, 1000)
    _obs = np.r_[rng.permutation(np.r_[np.ones(400, bool), np.zeros(100, bool)]), rng.permutation(np.r_[np.ones(100, bool), np.zeros(400, bool)])]
    _bb = np.polyfit(_x[_obs], _y[_obs], 1); _b1[_r], _b0[_r] = _bb
    _mcc[_r] = _y[_obs].mean()
    _mw[_r] = 0.5 * _y[_obs & (_x == 0)].mean() + 0.5 * _y[_obs & (_x == 1)].mean()
check(T, "例3 β̂0=452.13, β̂1=95.74（期待値, 乱数 4000回, 許容0.5）", abs(_b0.mean() - 452.13) < 0.5 and abs(_b1.mean() - 95.74) < 0.5 and close(547.87 - 452.13, 95.74, 5e-3), (round(_b0.mean(), 3), round(_b1.mean(), 3)))
check(T, "例3 完全ケース平均≈471.28（偏る）, 重みづけし直すと≈500（許容0.5）", abs(_mcc.mean() - 471.28) < 0.5 and abs(_mw.mean() - 500) < 0.5 and close(0.5 * 452.13 + 0.5 * 547.87, 500.0, 5e-3), (round(_mcc.mean(), 3), round(_mw.mean(), 3)))
check(T, "例4 √(1000/590)≈1.302", close(math.sqrt(1000 / 590), 1.302, 5e-4), round(math.sqrt(1000 / 590), 5))
_R = np.array([[1, 0.9, 0.9], [0.9, 1, 0], [0.9, 0, 1]])
check(T, "例5 det=−0.62, 最小固有値≈−0.273", close(np.linalg.det(_R), -0.62, 1e-9) and close(np.linalg.eigvalsh(_R)[0], -0.273, 5e-4), np.round(np.linalg.eigvalsh(_R), 5))

n_ok = sum(results)
n_ng = len(results) - n_ok
print(f"\n合計（重要度A への加筆を含む）: OK {n_ok} / NG {n_ng}")
