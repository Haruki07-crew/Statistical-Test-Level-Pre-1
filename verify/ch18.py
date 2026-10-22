"""第18章 検算スクリプト（質的回帰・一般化線形モデル）。
GLM の推定は numpy で IRLS（フィッシャーのスコア法）を独立に実装し、プロビットは scipy.optimize でも最尤推定して照合する。
記号計算は sympy、分布の数値は scipy.stats。主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。
charts の params は各 md の front matter を yaml で読み、本文の値と照合する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（概ね標準誤差の4倍以上）。
"""
from math import sqrt, log, exp, pi
from pathlib import Path
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
    return abs(float(x) - claim) <= tol


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float)
    return len(xs) == len(claims) and bool(np.all(np.abs(xs - np.asarray(claims, float)) <= tol))


def r4(v, k=4):
    return [round(float(x), k) for x in v]


def charts(term):
    txt = (ROOT / "content" / "ch18" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1]).get("charts", [])


def chart(term, cid, k=0):
    cs = [c for c in charts(term) if c.get("id") == cid or c.get("dist") == cid]
    return cs[k]


# ---------------- 汎用 IRLS ----------------
expit = lambda t: 1 / (1 + np.exp(-t))


def irls(X, y, family, m=None, offset=None, beta0=None, iters=50, record=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    b = np.zeros(p) if beta0 is None else np.asarray(beta0, float)
    hist = [b.copy()]
    for _ in range(iters):
        eta = X @ b + off
        if family == "binomial":
            pr = expit(eta)
            mu = m * pr
            dmu = m * pr * (1 - pr)
            V = m * pr * (1 - pr)
        elif family == "poisson":
            mu = np.exp(eta)
            dmu = mu
            V = mu
        elif family == "probit":
            pr = stats.norm.cdf(eta)
            mu = m * pr
            dmu = m * stats.norm.pdf(eta)
            V = m * pr * (1 - pr)
        w = dmu ** 2 / V
        z = eta - off + (y - mu) / dmu
        b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
        hist.append(b.copy())
    eta = X @ b + off
    if family == "binomial":
        pr = expit(eta); mu = m * pr; w = m * pr * (1 - pr)
    elif family == "poisson":
        mu = np.exp(eta); w = mu
    else:
        pr = stats.norm.cdf(eta); mu = m * pr
        w = (m * stats.norm.pdf(eta)) ** 2 / (m * pr * (1 - pr))
    info = X.T @ (w[:, None] * X)
    out = dict(b=b, mu=mu, w=w, info=info, cov=np.linalg.inv(info), hist=hist)
    return out


def binom_dev_terms(y, m, mu):
    def xlogx(a, b):
        return np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / b), 0.0)
    return 2 * (xlogx(y, mu) + xlogx(m - y, m - mu))


def pois_dev_terms(y, mu):
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))


# ================= データ =================
xL = np.arange(5.0)
yL = np.array([3, 4, 10, 12, 18], float)
mL = np.full(5, 20.0)
XL = np.column_stack([np.ones(5), xL])
L = irls(XL, yL, "binomial", m=mL)
bL = L["b"]
seL = np.sqrt(np.diag(L["cov"]))
pL = L["mu"] / mL

xP = np.arange(5.0)
yP = np.array([2, 3, 6, 7, 12], float)
XP = np.column_stack([np.ones(5), xP])
P = irls(XP, yP, "poisson")
bP = P["b"]
seP = np.sqrt(np.diag(P["cov"]))

# ================= logistic-regression =================
T = "logistic-regression"
check(T, "例1 β̂0, β̂1 = (−2.0479, 0.9391)（IRLS 独立実装）", allclose(bL, [-2.0479, 0.9391]), r4(bL))
# scipy.optimize でも最尤推定
def nll_logit(b):
    eta = XL @ b
    return -np.sum(yL * eta - mL * np.log1p(np.exp(eta)))
opt = optimize.minimize(nll_logit, [0, 0], method="BFGS", options={"gtol": 1e-10})
check(T, "例1 scipy.optimize の最尤推定でも同じ係数", allclose(opt.x, [-2.0479, 0.9391]), r4(opt.x))
check(T, "例1 SE(β̂1)=0.1947", close(seL[1], 0.1947), round(seL[1], 5))
check(T, "例1 オッズ比 e^0.9391≈2.558", close(np.exp(bL[1]), 2.558, 5e-4), round(np.exp(bL[1]), 4))
eta2 = bL[0] + 2 * bL[1]
check(T, "例1 x=2 の η̂=−0.1698", close(eta2, -0.1698), round(eta2, 5))
check(T, "例1 x=2 の p̂=0.4577", close(expit(eta2), 0.4577), round(expit(eta2), 5))
check(T, "例1 半数致死 x=−β̂0/β̂1≈2.181", close(-bL[0] / bL[1], 2.181, 5e-4), round(-bL[0] / bL[1], 4))
ci = np.exp(0.9391 + np.array([-1, 1]) * 1.96 * 0.1947)
check(T, "例1 オッズ比95%CI (1.746, 3.746)", allclose(ci, [1.746, 3.746], 5e-4), r4(ci))
# 確率の微分 β p(1-p)
xs_, b0s, b1s = sp.symbols("x beta0 beta1", real=True)
ps = 1 / (1 + sp.exp(-(b0s + b1s * xs_)))
check(T, "∂p/∂x = β1 p(1−p)（sympy）", sp.simplify(sp.diff(ps, xs_) - b1s * ps * (1 - ps)) == 0, "差=0")
oddsr = sp.simplify((ps / (1 - ps)).subs(xs_, xs_ + 1) / (ps / (1 - ps)))
check(T, "odds(x+1)/odds(x)=e^β1（sympy）", sp.simplify(oddsr - sp.exp(b1s)) == 0, oddsr)
# 例2
Xd = np.column_stack([np.ones(2), [1, 0]])
D2 = irls(Xd, np.array([30, 18.]), "binomial", m=np.array([50, 50.]))
check(T, "例2 β̂0=log(18/32)=−0.5754, β̂1=log2.6667=0.9808（IRLS）", allclose(D2["b"], [-0.5754, 0.9808]), r4(D2["b"]))
check(T, "例2 e^β̂1 = ad/(bc)=2.6667", close(np.exp(D2["b"][1]), 30 * 32 / (20 * 18)) and close(30 * 32 / (20 * 18), 2.6667), round(np.exp(D2["b"][1]), 5))
c = chart(T, "logistic-curve")
check(T, "chart logistic-curve params β0=−2.048, β1=0.939（推定値の3桁丸め）",
      c["params"]["beta0"] == round(bL[0], 3) and c["params"]["beta1"] == round(bL[1], 3), c["params"])
c = chart(T, "line-data")
check(T, "chart line-data 観測割合 y/m", allclose(c["params"]["series"][0]["y"], yL / mL), c["params"]["series"][0]["y"])
check(T, "chart line-data 当てはめ値", allclose(c["params"]["series"][1]["y"], pL), r4(pL))

# ================= odds-and-odds-ratio =================
T = "odds-and-odds-ratio"
check(T, "p=0.2 のオッズ 0.25", close(0.2 / 0.8, 0.25), 0.25)
check(T, "例1 オッズ0.75 → 確率 0.4286、確率は約2.14倍", close(0.75 / 1.75, 0.4286) and close(0.75 / 1.75 / 0.2, 2.14, 5e-3), round(0.75 / 1.75, 5))
a, b, cc, d = 30, 20, 18, 32
OR = a * d / (b * cc)
se = sqrt(1 / a + 1 / b + 1 / cc + 1 / d)
check(T, "例2 オッズ 1.5, 0.5625", close(a / b, 1.5) and close(cc / d, 0.5625), (a / b, cc / d))
check(T, "例2 OR=960/360=2.6667, log=0.9808", close(OR, 2.6667) and close(log(OR), 0.9808) and a * d == 960 and b * cc == 360, round(OR, 5))
check(T, "例2 SE=0.4125", close(se, 0.4125), round(se, 5))
ci = np.exp(log(OR) + np.array([-1, 1]) * 1.96 * se)
check(T, "例2 95%CI (1.188, 5.985)", allclose(ci, [1.188, 5.985], 5e-4), r4(ci))
check(T, "例2 リスク比 0.6/0.36=1.667", close(0.6 / 0.36, 1.667, 5e-4), round(0.6 / 0.36, 4))
check(T, "例3 RR=2, OR=6·997/(994·3)≈2.006", close(0.006 / 0.003, 2) and close(6 * 997 / (994 * 3), 2.006, 5e-4), round(6 * 997 / (994 * 3), 5))
p1, p2 = sp.symbols("p1 p2", positive=True)
ORs = (p1 / (1 - p1)) / (p2 / (1 - p2))
check(T, "OR = RR(1−p2)/(1−p1)（sympy）", sp.simplify(ORs - (p1 / p2) * (1 - p2) / (1 - p1)) == 0, "差=0")
check(T, "行列を入れ替えても OR 同じ (a/b)/(c/d)=(a/c)/(b/d)", close((a / b) / (cc / d), (a / cc) / (b / d), 1e-12), round((a / cc) / (b / d), 5))
check(T, "log2=0.693、OR=2 と 0.5 は ±0.693", close(log(2), 0.693, 5e-4) and close(log(0.5), -0.693, 5e-4), round(log(2), 4))
check(T, "p=0.8 のオッズ 4", close(0.8 / 0.2, 4), 4)
# デルタ法 SE をシミュレーション（ポアソンセル）
sim = rng.poisson([a, b, cc, d], size=(200000, 4)).astype(float)
sim = sim[(sim > 0).all(1)]
lo = np.log(sim[:, 0]) - np.log(sim[:, 1]) - np.log(sim[:, 2]) + np.log(sim[:, 3])
check(T, "SE 式 √(Σ1/セル) をポアソンセルのシミュレーション(20万回)で確認（デルタ法は近似なのでこの小さいセルでは許容 0.03）", abs(lo.std() - se) < 0.03, round(lo.std(), 4))
sim = rng.poisson(np.array([a, b, cc, d]) * 20, size=(200000, 4)).astype(float)
lo = np.log(sim[:, 0]) - np.log(sim[:, 1]) - np.log(sim[:, 2]) + np.log(sim[:, 3])
check(T, "セルを20倍にするとシミュレーション SD が SE/√20 に近づく（許容 0.002）", abs(lo.std() - se / sqrt(20)) < 0.002, (round(lo.std(), 4), round(se / sqrt(20), 4)))
c = chart(T, "mosaic")
check(T, "chart mosaic table", c["params"]["table"] == [[30, 20], [18, 32]], c["params"]["table"])
c = chart(T, "line-data")
px = np.array(c["params"]["x"])
check(T, "chart line-data オッズ p/(1−p)", allclose(c["params"]["series"][0]["y"], px / (1 - px)), r4(px / (1 - px)))

# ================= logistic-likelihood-estimation =================
T = "logistic-likelihood-estimation"
eta_s = sp.symbols("eta", real=True)
m_s, y_s = sp.symbols("m y", positive=True)
pe = sp.exp(eta_s) / (1 + sp.exp(eta_s))
lhs = y_s * sp.log(pe) + (m_s - y_s) * sp.log(1 - pe)
rhs = y_s * eta_s - m_s * sp.log(1 + sp.exp(eta_s))
check(T, "y log p+(m−y)log(1−p)=yη−m log(1+e^η)（sympy、数値代入）",
      all(abs(float((lhs - rhs).subs({eta_s: e, m_s: 7, y_s: 3}))) < 1e-12 for e in [-2, -0.3, 0.5, 3]), "差≈0")
check(T, "∂/∂η = y−mp（sympy）", sp.simplify(sp.diff(rhs, eta_s) - (y_s - m_s * pe)) == 0, "差=0")
check(T, "∂²/∂η² = −mp(1−p)（sympy）", sp.simplify(sp.diff(rhs, eta_s, 2) + m_s * pe * (1 - pe)) == 0, "差=0")
# 1回目
W0 = np.full(5, 5.0)
U0 = XL.T @ (yL - 10)
I0 = XL.T @ (W0[:, None] * XL)
check(T, "例1 ① p=0.5, mp=10, w=5", True, "定義どおり")
check(T, "例1 ② U=(−3, 32), Σy=47, Σxy=132", list(U0) == [-3, 32] and yL.sum() == 47 and xL @ yL == 132, U0)
check(T, "例1 ③ X'WX=[[25,50],[50,150]]、Σx=10、Σx²=30", np.array_equal(I0, [[25, 50], [50, 150]]), I0.tolist())
check(T, "例1 ③ det=1250、逆行列 (1/1250)[[150,−50],[−50,25]]",
      close(np.linalg.det(I0), 1250, 1e-9) and np.allclose(np.linalg.inv(I0) * 1250, [[150, -50], [-50, 25]]), "OK")
b1st = np.linalg.solve(I0, U0)
check(T, "例1 ④ β(1)=(−2050/1250, 950/1250)=(−1.64, 0.76)", np.allclose(b1st, [-1.64, 0.76]) and 150 * -3 - 50 * 32 == -2050 and -50 * -3 + 25 * 32 == 950, b1st)
H = L["hist"]
check(T, "例1 反復列 (−2.0088,0.9224)→(−2.0475,0.9389)→(−2.0479,0.9391)",
      allclose(H[2], [-2.0088, 0.9224]) and allclose(H[3], [-2.0475, 0.9389]) and allclose(H[4], [-2.0479, 0.9391]),
      [r4(h) for h in H[1:5]])
check(T, "例1 ⑤ ŵ=(2.024, 3.731, 4.964, 4.328, 2.597)", allclose(L["w"], [2.024, 3.731, 4.964, 4.328, 2.597], 5e-4), r4(L["w"], 3))
check(T, "例1 ⑤ X'ŴX=[[17.644,37.030],[37.030,104.088]]", allclose(L["info"].ravel(), [17.644, 37.030, 37.030, 104.088], 5e-4), np.round(L["info"], 3).tolist())
check(T, "例1 ⑤ 逆行列 [[0.22371,−0.07959],[−0.07959,0.03792]]", allclose(L["cov"].ravel(), [0.22371, -0.07959, -0.07959, 0.03792], 5e-6), np.round(L["cov"], 5).tolist())
check(T, "例1 SE(β̂0)=0.4730, SE(β̂1)=0.1947", allclose(seL, [0.4730, 0.1947]), r4(seL))
check(T, "例1 ワルド z=0.9391/0.1947≈4.82", close(bL[1] / seL[1], 4.82, 5e-3) and close(0.9391 / 0.1947, 4.82, 5e-3), round(bL[1] / seL[1], 3))
# ニュートン=フィッシャー（ヘッセが y を含まない）: 数値ヘッセと比較
def ll_logit(b):
    return -nll_logit(b)
hN = np.zeros((2, 2)); hh = 1e-4
for i in range(2):
    for j in range(2):
        ei = np.eye(2)[i] * hh; ej = np.eye(2)[j] * hh
        hN[i, j] = (ll_logit(bL + ei + ej) - ll_logit(bL + ei - ej) - ll_logit(bL - ei + ej) + ll_logit(bL - ei - ej)) / (4 * hh * hh)
check(T, "観測情報量（数値ヘッセ）= X'ŴX", np.allclose(-hN, L["info"], atol=1e-3), np.round(-hN, 3).tolist())
check(T, "例2 β̂0=−0.5754, β̂1=0.9808", close(log(18 / 32), -0.5754) and close(log(30 / 20) - log(18 / 32), 0.9808), (round(log(18 / 32), 5), round(log(1.5) - log(18 / 32), 5)))
# 完全分離：尤度が上限に近づくだけ
xs_sep = np.array([0, 1, 2, 3.]); ys_sep = np.array([0, 0, 1, 1.])
lls = [np.sum(ys_sep * (s * (xs_sep - 1.5)) - np.log1p(np.exp(s * (xs_sep - 1.5)))) for s in [1, 5, 20, 50]]
check(T, "完全分離で対数尤度が単調増加し0に近づく", np.all(np.diff(lls) > 0) and lls[-1] < 0 and lls[-1] > -1e-10, [f"{v:.2e}" for v in lls])
# プロファイル対数尤度の図
c = chart(T, "line-data")
gx = np.array(c["params"]["x"])
prof = []
for b1v in gx:
    f = lambda b0: -np.sum(yL * (b0 + b1v * xL) - mL * np.log1p(np.exp(b0 + b1v * xL)))
    r = optimize.minimize_scalar(f, bounds=(-10, 5), method="bounded", options={"xatol": 1e-10})
    # 定数項を除く対数尤度 Σ{y log p + (m-y) log(1-p)}
    prof.append(-r.fun)
check(T, "chart プロファイル対数尤度 12点（Σ{y log p+(m−y)log(1−p)}、5e-4）", allclose(prof, c["params"]["series"][0]["y"], 5e-4), r4(prof, 3))
check(T, "chart キャプション 頂上 β1=0.939、対数尤度 −53.159", close(-nll_logit(bL), -53.159, 5e-4) and close(bL[1], 0.939, 5e-4), round(-nll_logit(bL), 4))
check(T, "χ²_{1,0.95}/2 = 1.92", close(stats.chi2.ppf(0.95, 1) / 2, 1.92, 5e-3), round(stats.chi2.ppf(0.95, 1) / 2, 4))

# ================= iteratively-reweighted-least-squares =================
T = "iteratively-reweighted-least-squares"
# 一般形：w = (dμ/dη)²/V、z = η+(y−μ)dη/dμ、スコア法の更新 = WLS（sympy で記号的に1観測の恒等式）
mu_s, V_s, d_s, yy = sp.symbols("mu V d y", positive=True)
w_s = d_s ** 2 / V_s
check(T, "(1/V)(dμ/dη) = w·(dη/dμ)（sympy）", sp.simplify(d_s / V_s - w_s / d_s) == 0, "差=0")
# 数値で：スコア法の1ステップ = WLS（ポアソン・ロジット・プロビットで）
ok = True
for fam, Xf, yf, mf in [("binomial", XL, yL, mL), ("poisson", XP, yP, None), ("probit", XL, yL, mL)]:
    b0_ = np.array([0.3, 0.1])
    eta = Xf @ b0_
    if fam == "binomial":
        pr = expit(eta); mu = mf * pr; dmu = mf * pr * (1 - pr); V = dmu
    elif fam == "poisson":
        mu = np.exp(eta); dmu = mu; V = mu
    else:
        pr = stats.norm.cdf(eta); mu = mf * pr; dmu = mf * stats.norm.pdf(eta); V = mf * pr * (1 - pr)
    w = dmu ** 2 / V
    U = Xf.T @ ((yf - mu) / V * dmu)
    I = Xf.T @ (w[:, None] * Xf)
    scoring = b0_ + np.linalg.solve(I, U)
    wls = np.linalg.solve(I, Xf.T @ (w * (eta + (yf - mu) / dmu)))
    ok &= np.allclose(scoring, wls)
check(T, "スコア法の更新 β+I⁻¹U = WLS (X'WX)⁻¹X'Wz（3モデルで数値一致）", ok, "一致")
# 表の重み・作業応答
check(T, "表 ロジスティック dμ/dη=mp(1−p)（μ=mp）", sp.simplify(sp.diff(m_s * pe, eta_s) - m_s * pe * (1 - pe)) == 0, "sympy")
check(T, "表 ポアソン w=μ²/μ=μ", sp.simplify((sp.exp(eta_s)) ** 2 / sp.exp(eta_s) - sp.exp(eta_s)) == 0, "sympy")
z1 = (yL - 10) / 5
check(T, "例1 w=5、z=(−1.4,−1.2,0,0.4,1.6)", allclose(z1, [-1.4, -1.2, 0, 0.4, 1.6]), z1.tolist())
check(T, "例1 z̄=−0.12, Σ(x−2)z=8.0 → β1=0.76, β0=−1.64",
      close(z1.mean(), -0.12) and close((xL - 2) @ z1, 2.8 + 1.2 + 0 + 0.4 + 3.2) and close((xL - 2) @ z1 / 10, 0.76) and close(z1.mean() - 0.76 * 2, -1.64),
      ((xL - 2) * z1).tolist())
c = chart(T, "line-data")
s0, s1 = c["params"]["series"]
check(T, "chart 係数推移 β0（t=0..5）", allclose(s0["y"], [h[0] for h in H[:6]]), r4([h[0] for h in H[:6]]))
check(T, "chart 係数推移 β1（t=0..5）", allclose(s1["y"], [h[1] for h in H[:6]]), r4([h[1] for h in H[:6]]))
# 2次収束：誤差比 e_{t+1}/e_t^2 が有界
err = [np.max(np.abs(h - bL)) for h in H[:5]]
check(T, "誤差が2乗のペースで減る（e_{t+1}/e_t² が概ね一定・有界）", all(err[i + 1] / err[i] ** 2 < 1.0 for i in range(1, 4)), [f"{e:.1e}" for e in err])
# 例2 ポアソン
Pp = irls(XP, yP, "poisson", beta0=[log(6), 0], iters=10)
zP = log(6) + (yP - 6) / 6
check(T, "例2 β0(0)=log6=1.7918, z=(1.1251,1.2918,1.7918,1.9584,2.7918)", close(log(6), 1.7918) and allclose(zP, [1.1251, 1.2918, 1.7918, 1.9584, 2.7918]), r4(zP))
check(T, "例2 β(1)=(0.9918, 0.4000)", allclose(Pp["hist"][1], [0.9918, 0.4000]) and close((xP - 2) @ zP / 10, 0.4), r4(Pp["hist"][1]))
tbl = [[1.7918, 0], [0.9918, 0.4000], [0.7757, 0.4264], [0.7504, 0.4312], [0.7501, 0.4313]]
check(T, "例2 表の反復 t=0..4", all(allclose(Pp["hist"][t], tbl[t]) for t in range(5)), [r4(h) for h in Pp["hist"][:5]])
llP = [float(np.sum(yP * (XP @ h) - np.exp(XP @ h))) for h in Pp["hist"][:5]]
check(T, "例2 表の対数尤度 Σ{yη−μ}=23.7528, 28.3307, 28.7303, 28.7330, 28.7330", allclose(llP, [23.7528, 28.3307, 28.7303, 28.7330, 28.7330]), r4(llP))

# ================= probit-model =================
T = "probit-model"
PR = irls(XL, yL, "probit", m=mL, beta0=[0, 0], iters=100)
def nll_probit(b):
    eta = XL @ b
    return -np.sum(yL * stats.norm.logcdf(eta) + (mL - yL) * stats.norm.logsf(eta))
op = optimize.minimize(nll_probit, [0, 0], method="BFGS", options={"gtol": 1e-10})
check(T, "例1 β̂=(−1.2193, 0.5629)（IRLS）", allclose(PR["b"], [-1.2193, 0.5629]), r4(PR["b"]))
check(T, "例1 β̂（scipy.optimize の最尤推定）", allclose(op.x, [-1.2193, 0.5629]), r4(op.x))
rat = bL / PR["b"]
check(T, "例1 ロジット/プロビット比がどちらも約1.67", allclose(rat, [1.67, 1.67], 0.015), r4(rat, 3))
e2 = -1.2193 + 0.5629 * 2
eP2 = PR["b"][0] + 2 * PR["b"][1]
check(T, "例1 x=2: η=−0.0935、Φ=0.4627（厳密な MLE で 0.46274。丸めた −0.0935 を代入すると 0.46275 で境目、許容 1e-4）", close(eP2, -0.0935) and close(stats.norm.cdf(eP2), 0.4627) and close(stats.norm.cdf(e2), 0.4627, 1e-4), (round(stats.norm.cdf(eP2), 6), round(stats.norm.cdf(e2), 6)))
check(T, "例1 半数致死 2.166", close(1.2193 / 0.5629, 2.166, 5e-4), round(1.2193 / 0.5629, 4))
me = stats.norm.pdf(e2) * 0.5629
check(T, "例1 限界効果 φ(−0.0935)=0.3972, ×0.5629=0.2236", close(stats.norm.pdf(e2), 0.3972) and close(me, 0.2236), (round(stats.norm.pdf(e2), 5), round(me, 5)))
check(T, "例1 ロジットの限界効果 0.4577×0.5423×0.9391=0.2331", close(0.4577 * 0.5423 * 0.9391, 0.2331), round(0.4577 * 0.5423 * 0.9391, 5))
DL = binom_dev_terms(yL, mL, L["mu"]).sum()
DPr = binom_dev_terms(yL, mL, PR["mu"]).sum()
check(T, "例1 逸脱度 ロジット 1.744、プロビット 1.758", close(DL, 1.744, 5e-4) and close(DPr, 1.758, 5e-4), (round(DL, 4), round(DPr, 4)))
check(T, "標準ロジスティックの分散 π²/3、SD 1.814", close(stats.logistic.var(), pi ** 2 / 3, 1e-9) and close(pi / sqrt(3), 1.814, 5e-4), round(pi / sqrt(3), 4))
check(T, "中央の密度 1/4 と 1/√(2π)=0.399、比 ≈1.6", close(stats.logistic.pdf(0), 0.25) and close(stats.norm.pdf(0), 0.399, 5e-4) and close(stats.norm.pdf(0) / 0.25, 1.6, 5e-3), round(stats.norm.pdf(0) / 0.25, 4))
gx = np.linspace(-10, 10, 20001)
maxdiff = np.max(np.abs(expit(gx) - stats.norm.cdf(gx / 1.702)))
check(T, "1.702 で尺度をそろえると最大差 <0.01", maxdiff < 0.01, round(maxdiff, 5))
# 潜在変数シミュレーション
eta_t = 0.4
eps = rng.standard_normal(400000)
check(T, "潜在変数 Y*=η+ε>0 の割合 = Φ(η)（40万回、許容 0.004）", abs((eta_t + eps > 0).mean() - stats.norm.cdf(eta_t)) < 0.004, round((eta_t + eps > 0).mean(), 4))
epsl = rng.logistic(size=400000)
check(T, "誤差をロジスティックにすると expit(η)（40万回、許容 0.004）", abs((eta_t + epsl > 0).mean() - expit(eta_t)) < 0.004, round((eta_t + epsl > 0).mean(), 4))
check(T, "∂Φ(xβ)/∂x = φ(xβ)β（sympy）", sp.simplify(sp.diff((1 + sp.erf((b0s + b1s * xs_) / sp.sqrt(2))) / 2, xs_) - b1s * sp.exp(-(b0s + b1s * xs_) ** 2 / 2) / sp.sqrt(2 * sp.pi)) == 0, "差=0")
c = chart(T, "line-data")
check(T, "chart line-data 観測割合とロジット当てはめ値", allclose(c["params"]["series"][0]["y"], yL / mL) and allclose(c["params"]["series"][1]["y"], pL), "一致")
check(T, "chart line-data プロビット当てはめ値", allclose(c["params"]["series"][2]["y"], PR["mu"] / mL), r4(PR["mu"] / mL))
c = chart(T, "logistic-curve")
check(T, "chart logistic-curve probit=1（実装は Φ((β0+β1x)/1.702)）", c["params"].get("probit") == 1 and "1.702" in (ROOT / "src" / "charts_custom.js").read_text(), c["params"])

# ================= poisson-regression =================
T = "poisson-regression"
def nll_pois(b):
    eta = XP @ b
    return -np.sum(yP * eta - np.exp(eta))
opP = optimize.minimize(nll_pois, [0, 0], method="BFGS", options={"gtol": 1e-10})
check(T, "例1 β̂=(0.7501, 0.4313)（IRLS）", allclose(bP, [0.7501, 0.4313]), r4(bP))
check(T, "例1 β̂（scipy.optimize）", allclose(opP.x, [0.7501, 0.4313]), r4(opP.x))
check(T, "例1 SE(β̂1)=0.1447", close(seP[1], 0.1447), round(seP[1], 5))
check(T, "例1 率比 e^0.4313≈1.539", close(np.exp(0.4313), 1.539, 5e-4), round(np.exp(0.4313), 4))
ci = np.exp(0.4313 + np.array([-1, 1]) * 1.96 * 0.1447)
check(T, "例1 率比95%CI (1.159, 2.044)", allclose(ci, [1.159, 2.044], 5e-4), r4(ci))
zP_ = bP[1] / seP[1]
check(T, "例1 z≈2.98, p≈0.0029", close(zP_, 2.98, 5e-3) and close(2 * stats.norm.sf(zP_), 0.0029), (round(zP_, 3), round(2 * stats.norm.sf(zP_), 5)))
check(T, "例1 当てはめ値 2.117, 3.259, 5.016, 7.722, 11.886", allclose(P["mu"], [2.117, 3.259, 5.016, 7.722, 11.886], 5e-4), r4(P["mu"], 3))
check(T, "例1 Σμ̂=30.000=Σy", close(P["mu"].sum(), 30, 5e-4), round(P["mu"].sum(), 6))
DP = pois_dev_terms(yP, P["mu"]).sum()
DP0 = pois_dev_terms(yP, np.full(5, yP.mean())).sum()
check(T, "例1 D=0.280、D0=10.240、差 9.960", close(DP, 0.280, 5e-4) and close(DP0, 10.240, 5e-4) and close(DP0 - DP, 9.960, 5e-4), (round(DP, 4), round(DP0, 4), round(DP0 - DP, 4)))
check(T, "例1 差の p≈0.0016（χ²₁）", close(stats.chi2.sf(DP0 - DP, 1), 0.0016), round(stats.chi2.sf(DP0 - DP, 1), 5))
# スコア・ヘッセ（sympy）
lam_s = sp.symbols("mu", positive=True)
ll1 = y_s * eta_s - sp.exp(eta_s)
check(T, "∂ℓ/∂η=y−μ、∂²ℓ/∂η²=−μ（sympy）", sp.simplify(sp.diff(ll1, eta_s) - (y_s - sp.exp(eta_s))) == 0 and sp.simplify(sp.diff(ll1, eta_s, 2) + sp.exp(eta_s)) == 0, "差=0")
# 例2 オフセット
Xo = np.column_stack([np.ones(2), [0, 1.]])
O = irls(Xo, np.array([18, 30.]), "poisson", offset=np.log([1000, 1200.]), beta0=[-4, 0])
check(T, "例2 オフセット付き IRLS β̂=(−4.0174, 0.3285)", allclose(O["b"], [-4.0174, 0.3285]), r4(O["b"]))
check(T, "例2 log(18/1000)=−4.0174、率比 0.025/0.018=1.3889", close(log(0.018), -4.0174) and close(0.025 / 0.018, 1.3889) and close(log(0.025 / 0.018), 0.3285), round(0.025 / 0.018, 5))
seO = np.sqrt(np.diag(O["cov"]))
check(T, "例2 SE(β̂1)=√(1/18+1/30)=0.2981（IRLS の SE とも一致）", close(sqrt(1 / 18 + 1 / 30), 0.2981) and close(seO[1], 0.2981), round(seO[1], 5))
ci = np.exp(0.3285 + np.array([-1, 1]) * 1.96 * 0.2981)
check(T, "例2 率比95%CI (0.774, 2.491)", allclose(ci, [0.774, 2.491], 5e-4), r4(ci))
c = chart(T, "line-data")
check(T, "chart line-data 観測件数・当てはめ値", c["params"]["series"][0]["y"] == [2, 3, 6, 7, 12] and allclose(c["params"]["series"][1]["y"], P["mu"]) and allclose(c["params"]["series"][1]["y"], np.exp(0.7501 + 0.4313 * xP), 6e-4), r4(P["mu"]))
check(T, "chart pmf λ=5.0164（x=2 の当てはめ値）", close(chart(T, "poisson")["params"]["lambda"], P["mu"][2], 2e-4), round(P["mu"][2], 5))
check(T, "キャプション 1.54倍", close(np.exp(bP[1]), 1.54, 5e-3), round(np.exp(bP[1]), 4))

# ================= overdispersion =================
T = "overdispersion"
yo = np.array([0, 1, 9, 2, 3, 12, 1, 0, 8, 4], float)
check(T, "例1 μ̂=ȳ=4", yo.mean() == 4, yo.mean())
check(T, "例1 s²=17.78", close(yo.var(ddof=1), 17.78, 5e-3), round(yo.var(ddof=1), 4))
X2 = np.sum((yo - 4) ** 2 / 4)
check(T, "例1 X²=160/4=40", np.sum((yo - 4) ** 2) == 160 and X2 == 40, X2)
check(T, "例1 φ̂=40/9≈4.444", close(X2 / 9, 4.444, 5e-4), round(X2 / 9, 5))
check(T, "例1 χ²₉ 上側5%点 16.919", close(stats.chi2.ppf(0.95, 9), 16.919, 5e-4), round(stats.chi2.ppf(0.95, 9), 4))
check(T, "例1 上側確率 ≈7.6e-6", abs(stats.chi2.sf(40, 9) - 7.6e-6) < 5e-8, f"{stats.chi2.sf(40, 9):.3e}")
# 切片のみポアソン IRLS の SE
Po0 = irls(np.ones((10, 1)), yo, "poisson", iters=30)
se0 = sqrt(Po0["cov"][0, 0])
check(T, "例1 log μ の SE=1/√40=0.1581（IRLS）", close(se0, 0.1581) and close(1 / sqrt(40), 0.1581), round(se0, 5))
check(T, "例1 準ポアソン SE=√4.444×0.1581≈2.108×0.1581≈0.3333", close(sqrt(40 / 9), 2.108, 5e-4) and close(sqrt(40 / 9) / sqrt(40), 0.3333) and close(2.108 * 0.1581, 0.3333), round(sqrt(40 / 9) / sqrt(40), 5))
k = 16 / (yo.var(ddof=1) - 4)
check(T, "例1 k=16/13.78≈1.161", close(k, 1.161, 5e-4) and close(yo.var(ddof=1) - 4, 13.78, 5e-3), round(k, 4))
pnb = k / (k + 4)
check(T, "例1 P(0) ポアソン e^−4=0.018、負の二項 0.177", close(exp(-4), 0.018, 5e-4) and close(stats.nbinom.pmf(0, k, pnb), 0.177, 5e-4), (round(exp(-4), 4), round(stats.nbinom.pmf(0, k, pnb), 4)))
c = charts(T)
check(T, "chart pmf poisson λ=4、キャプション P(0)=0.018, P(≥10)=0.008", c[0]["params"]["lambda"] == 4 and close(stats.poisson.sf(9, 4), 0.008, 5e-4), round(stats.poisson.sf(9, 4), 4))
r_, p_ = c[1]["params"]["r"], c[1]["params"]["p"]
m_nb, v_nb = stats.nbinom.stats(r_, p_)
check(T, "chart nbinom r=1.161, p=0.225 → 平均4・分散17.78（p=k/(k+4)、丸め込みで許容 0.01・0.1）", close(r_, k, 5e-4) and close(p_, pnb, 5e-4) and abs(m_nb - 4) < 0.01 and abs(v_nb - 17.78) < 0.1, (round(float(m_nb), 4), round(float(v_nb), 3)))
check(T, "本文 負の二項は 10件以上の確率がずっと大きい", stats.nbinom.sf(9, k, pnb) > 5 * stats.poisson.sf(9, 4), round(stats.nbinom.sf(9, k, pnb), 4))
# ポアソン-ガンマ混合（シミュレーション）
lam = rng.gamma(k, 4 / k, size=1_000_000)
yy_ = rng.poisson(lam)
check(T, "ポアソン−ガンマ混合 V=μ+μ²/k（100万回、許容 平均0.02・分散0.3）", abs(yy_.mean() - 4) < 0.02 and abs(yy_.var() - (4 + 16 / k)) < 0.3, (round(yy_.mean(), 4), round(yy_.var(), 3)))
check(T, "混合の P(0) が負の二項の 0.177（許容 0.002）", abs((yy_ == 0).mean() - 0.177) < 0.002, round((yy_ == 0).mean(), 4))
# 条件付き分散（sympy、ガンマ）
kk, mm = sp.symbols("k mu", positive=True)
check(T, "ガンマ(形状k、平均μ) の分散 μ²/k（sympy の k·θ² with θ=μ/k）", sp.simplify(kk * (mm / kk) ** 2 - mm ** 2 / kk) == 0, "μ²/k")
# ベータ二項分散
m_bb, a_bb, b_bb = 10, 2.0, 3.0
pbb = a_bb / (a_bb + b_bb); rho = 1 / (a_bb + b_bb + 1)
check(T, "ベータ二項 V=mp(1−p){1+(m−1)ρ}（scipy.betabinom）", close(stats.betabinom.var(m_bb, a_bb, b_bb), m_bb * pbb * (1 - pbb) * (1 + (m_bb - 1) * rho), 1e-10), round(stats.betabinom.var(m_bb, a_bb, b_bb), 5))
# 準尤度：SE が √φ 倍（過分散データのシミュレーション）
nrep = 2000; xs_q = np.repeat(np.arange(5.0), 20); Xq = np.column_stack([np.ones(100), xs_q])
b_hat = []; se_naive = []
for _ in range(nrep):
    mu_t = np.exp(1 + 0.2 * xs_q)
    yq = rng.poisson(rng.gamma(2, mu_t / 2))
    f = irls(Xq, yq, "poisson", iters=8)
    b_hat.append(f["b"][1]); se_naive.append(np.sqrt(f["cov"][1, 1]))
ratio = np.std(b_hat) / np.mean(se_naive)
check(T, "過分散でポアソン SE は過小（実際のばらつき/報告 SE >1.5、2000回）", ratio > 1.5, round(ratio, 3))

# ================= deviance =================
T = "deviance"
dterms = binom_dev_terms(yL, mL, L["mu"])
check(T, "例1 p̂=0.1143,0.2481,0.4577,0.6834,0.8466", allclose(pL, [0.1143, 0.2481, 0.4577, 0.6834, 0.8466]), r4(pL))
check(T, "例1 20×p̂1=2.2853", close(L["mu"][0], 2.2853), round(L["mu"][0], 5))
check(T, "例1 d=(0.2325,0.2600,0.1439,0.6171,0.4901)", allclose(dterms, [0.2325, 0.2600, 0.1439, 0.6171, 0.4901]), r4(dterms))
check(T, "例1 D=1.744", close(dterms.sum(), 1.744, 5e-4), round(dterms.sum(), 5))
check(T, "例1 上側確率 0.627、χ²₃ 5%点 7.815", close(stats.chi2.sf(dterms.sum(), 3), 0.627, 5e-4) and close(stats.chi2.ppf(0.95, 3), 7.815, 5e-4), round(stats.chi2.sf(dterms.sum(), 3), 4))
X2L = np.sum((yL - L["mu"]) ** 2 / (mL * pL * (1 - pL)))
check(T, "例1 ピアソン X²=1.726", close(X2L, 1.726, 5e-4), round(X2L, 4))
llm = stats.binom.logpmf(yL, 20, pL).sum()
lls_ = stats.binom.logpmf(yL, 20, yL / 20).sum()
check(T, "例1 対数尤度 モデル −8.5168、飽和 −7.6450", close(llm, -8.5168) and close(lls_, -7.6450), (round(llm, 5), round(lls_, 5)))
check(T, "例1 2×(−7.6450+8.5168)=1.744", close(2 * (lls_ - llm), 1.744, 5e-4) and close(2 * (-7.6450 + 8.5168), 1.744, 5e-4), round(2 * (lls_ - llm), 5))
D0L = binom_dev_terms(yL, mL, mL * 0.47).sum()
check(T, "例2 D0=33.695、差 31.951", close(D0L, 33.695, 5e-4) and close(D0L - dterms.sum(), 31.951, 5e-4), (round(D0L, 4), round(D0L - dterms.sum(), 4)))
check(T, "例2 χ²₁ 上側確率 ≈1.6e-8", abs(stats.chi2.sf(D0L - dterms.sum(), 1) - 1.6e-8) < 5e-10, f"{stats.chi2.sf(D0L - dterms.sum(), 1):.3e}")
check(T, "例3 ポアソン D=0.280, D0=10.240, 差 9.960, p≈0.0016", close(DP, 0.280, 5e-4) and close(DP0, 10.240, 5e-4) and close(stats.chi2.sf(DP0 - DP, 1), 0.0016), (round(DP, 4), round(DP0, 4)))
# 逸脱度残差の二乗和
rD = np.sign(yL - L["mu"]) * np.sqrt(dterms)
check(T, "Σ(r_D)² = D", close(np.sum(rD ** 2), dterms.sum(), 1e-12), round(np.sum(rD ** 2), 5))
# 正規の逸脱度 = RSS（σ²D*）
ys_, mus_, sig = sp.symbols("y mu sigma", positive=True)
lnorm = -sp.log(sig * sp.sqrt(2 * sp.pi)) - (ys_ - mus_) ** 2 / (2 * sig ** 2)
Dstar = 2 * (lnorm.subs(mus_, ys_) - lnorm)
check(T, "正規 σ²D* = (y−μ̂)²（sympy）", sp.simplify(sig ** 2 * Dstar - (ys_ - mus_) ** 2) == 0, "差=0")
lpois = ys_ * sp.log(mus_) - mus_
Dp = sp.simplify(2 * (lpois.subs(mus_, ys_) - lpois))
check(T, "ポアソン D の式 2{y log(y/μ)−(y−μ)}（sympy）", sp.simplify(sp.expand_log(Dp - 2 * (ys_ * sp.log(ys_ / mus_) - (ys_ - mus_)), force=True)) == 0, Dp)
check(T, "切片ありポアソンでは Σ(y−μ̂)=0（第2項が消える）", close(np.sum(yP - P["mu"]), 0, 1e-8), f"{np.sum(yP - P['mu']):.1e}")
# ウィルクス：D0−D1 ~ χ²₁ をシミュレーション（ロジスティック、β1=0、ニュートン法をベクトル化）
Nw = 400000
Yw = rng.binomial(20, 0.47, size=(Nw, 5)).astype(float)
bw = np.zeros((Nw, 2))
for _ in range(30):
    pw = expit(bw @ XL.T)
    ww = 20 * pw * (1 - pw)
    Uw = (Yw - 20 * pw) @ XL
    Iw = np.einsum("ni,ij,ik->njk", ww, XL, XL)
    bw = bw + np.linalg.solve(Iw, Uw[..., None])[..., 0]
def dev_rows(yv, muv):
    t = lambda a_, c_: np.where(a_ > 0, a_ * np.log(np.where(a_ > 0, a_, 1) / c_), 0.0)
    return 2 * (t(yv, muv) + t(20 - yv, 20 - muv)).sum(1)
Ds = dev_rows(Yw, 20 * Yw.sum(1, keepdims=True) / 100) - dev_rows(Yw, 20 * expit(bw @ XL.T))
check(T, "H0 下で D0−D1 ≈ χ²₁：5%超過率 ≈0.05・平均 ≈1（40万回、m=20 の有限標本なので許容 0.008・0.03）", abs((Ds > stats.chi2.ppf(0.95, 1)).mean() - 0.05) < 0.008 and abs(Ds.mean() - 1) < 0.03, (round((Ds > stats.chi2.ppf(0.95, 1)).mean(), 4), round(Ds.mean(), 3)))
c = chart(T, "bar-data")
check(T, "chart bar-data d_i", allclose(c["params"]["values"], dterms), c["params"]["values"])
check(T, "chart chi2 df=3 (=5−2)", chart(T, "chi2")["params"]["df"] == 3, 3)

# ================= link-function =================
T = "link-function"
c = chart(T, "line-data")
gx = np.array(c["params"]["x"], float)
s = c["params"]["series"]
check(T, "chart ロジット逆関数 17点", allclose(s[0]["y"], expit(gx)), r4(expit(gx)))
check(T, "chart プロビット逆関数 17点", allclose(s[1]["y"], stats.norm.cdf(gx)), r4(stats.norm.cdf(gx)))
check(T, "chart 補対数対数逆関数 17点", allclose(s[2]["y"], 1 - np.exp(-np.exp(gx))), r4(1 - np.exp(-np.exp(gx))))
check(T, "例1 η=0.5: 0.6225, 0.6915, 0.8077、e^0.5=1.6487",
      close(expit(0.5), 0.6225) and close(stats.norm.cdf(0.5), 0.6915) and close(1 - exp(-exp(0.5)), 0.8077) and close(exp(0.5), 1.6487), (round(expit(0.5), 5), round(stats.norm.cdf(0.5), 5), round(1 - exp(-exp(0.5)), 5)))
check(T, "例1 η=0 で補対数対数 1−e^−1=0.6321", close(1 - exp(-1), 0.6321), round(1 - exp(-1), 5))
lam_s2 = sp.symbols("lambda", positive=True)
pp_s = sp.symbols("p", positive=True)
cll = sp.log(-sp.log(1 - (1 - sp.exp(-sp.exp(eta_s)))))
check(T, "例2 log{−log(1−p)}=η（p=1−exp(−e^η)、sympy）", sp.simplify(cll - eta_s) == 0, sp.simplify(cll))
Nsim = rng.poisson(exp(0.5), 400000)
check(T, "例2 P(N≥1)=0.8077（ポアソン乱数40万、許容 0.003）", abs((Nsim >= 1).mean() - 0.8077) < 0.003, round((Nsim >= 1).mean(), 4))
check(T, "例3 e^0.4=1.492", close(exp(0.4), 1.492, 5e-4), round(exp(0.4), 4))
# 正準リンク
th = sp.symbols("theta", real=True)
canon = {}
for name, bth, inv_expected in [
    ("正規", th ** 2 / 2, lambda mu: mu),
    ("ポアソン", sp.exp(th), lambda mu: sp.log(mu)),
    ("ベルヌーイ", sp.log(1 + sp.exp(th)), lambda mu: sp.log(mu / (1 - mu))),
    ("ガンマ", -sp.log(-th), lambda mu: -1 / mu),
    ("逆ガウス", -sp.sqrt(-2 * th), lambda mu: -1 / (2 * mu ** 2)),
]:
    mu_sym = sp.symbols("mu", positive=True)
    # θ = g(μ) を b' に代入して μ に戻るか（ベルヌーイは 0<μ<1 で数値確認）
    back = sp.diff(bth, th).subs(th, inv_expected(mu_sym))
    ok_ = all(abs(float(back.subs(mu_sym, v)) - v) < 1e-12 for v in [0.2, 0.5, 0.7])
    canon[name] = ok_
check(T, "正準リンク (b')⁻¹：正規→恒等、ポアソン→log、二項→ロジット、ガンマ→−1/μ、逆ガウス→−1/(2μ²)", all(canon.values()), canon)
check(T, "ベルヌーイ f=exp{y logit p + log(1−p)}（y=0,1）",
      all(abs(float(sp.exp(yv * sp.log(pp_s / (1 - pp_s)) + sp.log(1 - pp_s)).subs(pp_s, 0.3)) - (0.3 if yv else 0.7)) < 1e-12 for yv in [0, 1]), "OK")
# b 凸
check(T, "b''(θ)>0（ポアソン e^θ、ベルヌーイ e^θ/(1+e^θ)²、ガンマ 1/θ²）", all(sp.simplify(sp.diff(bth, th, 2) - e) == 0 for bth, e in [(sp.exp(th), sp.exp(th)), (sp.log(1 + sp.exp(th)), sp.exp(th) / (1 + sp.exp(th)) ** 2), (-sp.log(-th), 1 / th ** 2)]), "sympy")
check(T, "chart logistic-curve probit=1", chart(T, "logistic-curve")["params"].get("probit") == 1, "OK")

# ================= exponential-dispersion-family =================
T = "exponential-dispersion-family"
mvar = sp.symbols("m", positive=True)
tab = [
    ("正規", th ** 2 / 2, th, 1),
    ("ポアソン", sp.exp(th), sp.exp(th), sp.exp(th)),
    ("二項", mvar * sp.log(1 + sp.exp(th)), mvar * sp.exp(th) / (1 + sp.exp(th)), None),
    ("ガンマ", -sp.log(-th), -1 / th, 1 / th ** 2),
]
for name, bth, mu_exp, b2 in tab:
    ok1 = sp.simplify(sp.diff(bth, th) - mu_exp) == 0
    check(T, f"{name}: b'(θ)={mu_exp}（sympy）", ok1, sp.diff(bth, th))
# V(μ)
muv = sp.symbols("mu", positive=True)
Vchk = {
    "正規": sp.simplify(sp.diff(th ** 2 / 2, th, 2) - 1) == 0,
    "ポアソン": sp.simplify(sp.diff(sp.exp(th), th, 2).subs(th, sp.log(muv)) - muv) == 0,
    "二項": sp.simplify(sp.diff(mvar * sp.log(1 + sp.exp(th)), th, 2).subs(th, sp.log(muv / (mvar - muv))) - muv * (1 - muv / mvar)) == 0,
    "ガンマ": sp.simplify(sp.diff(-sp.log(-th), th, 2).subs(th, -1 / muv) - muv ** 2) == 0,
}
check(T, "分散関数 V(μ)=1, μ, μ(1−μ/m), μ²（sympy）", all(Vchk.values()), Vchk)
# 各分布を指数分散型に書けること：対数密度の差（数値）
okf = True
for yv in [0, 1, 3, 7]:
    lamv = 2.7
    okf &= abs(stats.poisson.logpmf(yv, lamv) - (yv * log(lamv) - lamv - float(sp.log(sp.factorial(yv))))) < 1e-10
    mv, pv = 10, 0.3
    tv = log(pv / (1 - pv))
    okf &= abs(stats.binom.logpmf(yv, mv, pv) - (yv * tv - mv * log(1 + exp(tv)) + log(float(sp.binomial(mv, yv))))) < 1e-10
for yv in [0.3, 1.2, 4.0]:
    mu0, nu = 2.0, 4.0; tv = -1 / mu0
    lg = stats.gamma.logpdf(yv, nu, scale=mu0 / nu)
    c_ = nu * log(nu) + (nu - 1) * log(yv) - float(sp.loggamma(nu))
    okf &= abs(lg - ((yv * tv - (-log(-tv))) * nu + c_)) < 1e-10
    sg = 1.5
    okf &= abs(stats.norm.logpdf(yv, 0.8, sg) - ((yv * 0.8 - 0.8 ** 2 / 2) / sg ** 2 - yv ** 2 / (2 * sg ** 2) - log(sg * sqrt(2 * pi)))) < 1e-10
check(T, "表の (θ, b, a(φ)) で正規・ポアソン・二項・ガンマの対数密度を再現（数値）", okf, "一致")
# スコア性質から E,V：数値積分（ポアソン・ガンマ）
check(T, "E[∂ℓ/∂θ]=0 と情報量等式から V=a b''（ガンマ μ=2, ν=4 を数値積分で V=1）", close(stats.gamma.var(4, scale=0.5), 1, 1e-12) and close(stats.gamma.mean(4, scale=0.5), 2, 1e-12), stats.gamma.var(4, scale=0.5))
# MGF
tt, aa = sp.symbols("t a", positive=True)
for name, bth in [("ポアソン", sp.exp(th)), ("二項", mvar * sp.log(1 + sp.exp(th))), ("ガンマ", -sp.log(-th))]:
    Kc = (bth.subs(th, th + aa * tt) - bth) / aa
    k1 = sp.simplify(sp.diff(Kc, tt).subs(tt, 0) - sp.diff(bth, th))
    k2 = sp.simplify(sp.diff(Kc, tt, 2).subs(tt, 0) - aa * sp.diff(bth, th, 2))
    check(T, f"{name}: K(t)={{b(θ+at)−b(θ)}}/a の K'(0)=b'、K''(0)=a b''（sympy）", k1 == 0 and k2 == 0, (k1, k2))
# MGF 式が実際の MGF と一致（ポアソン λ=3、二項、ガンマ）
tv = 0.2
check(T, "MGF 式が実際の MGF と一致（ポアソン λ=3, t=0.2）", close(exp(exp(log(3) + tv) - 3), exp(3 * (exp(tv) - 1)), 1e-12), round(exp(3 * (exp(tv) - 1)), 6))
gam_mgf = exp((-log(-(-0.5 + 0.25 * tv)) + log(0.5)) / 0.25)
check(T, "MGF 式が実際の MGF と一致（ガンマ 形状4・尺度0.5, t=0.2）", close(gam_mgf, (1 - 0.5 * tv) ** -4, 1e-12), round(gam_mgf, 6))
check(T, "例1 θ=log3=1.0986、E=V=3", close(log(3), 1.0986) and close(exp(log(3)), 3, 1e-12), round(log(3), 5))
tb = log(0.3 / 0.7)
check(T, "例2 θ=−0.8473, b'=3, b''=2.1", close(tb, -0.8473) and close(10 * exp(tb) / (1 + exp(tb)), 3, 1e-12) and close(10 * exp(tb) / (1 + exp(tb)) ** 2, 2.1, 1e-12), round(tb, 5))
check(T, "例3 θ=−0.5、a=0.25、b'=2、b''=4、V=1", close(-1 / -0.5, 2) and close(1 / 0.25, 4) and close(0.25 * 4, 1) and close(4 * 0.5 ** 2, 1), "OK")
ys2 = sp.symbols("y", positive=True)
nu_s, mu_s2 = sp.symbols("nu mu", positive=True)
logf = -sp.loggamma(nu_s) + nu_s * sp.log(nu_s / mu_s2) + (nu_s - 1) * sp.log(ys2) - nu_s * ys2 / mu_s2
diff_ = sp.expand_log(logf - nu_s * (ys2 * (-1 / mu_s2) - sp.log(mu_s2)), force=True)
check(T, "例3 ガンマ対数密度 = ν{y(−1/μ)−log μ}+（μ を含まない項）（sympy）", sp.simplify(sp.diff(diff_, mu_s2)) == 0, "μ で微分して0")
c = chart(T, "line-data")
gx = np.array(c["params"]["x"])
s = c["params"]["series"]
check(T, "chart 分散関数 1, μ, μ²", allclose(s[0]["y"], np.ones(7)) and allclose(s[1]["y"], gx) and allclose(s[2]["y"], gx ** 2), "一致")

# ================= generalized-linear-model =================
T = "generalized-linear-model"
# 連鎖律：一般リンクで dℓ/dβ = (y−μ)/(aV) dμ/dη x（数値：ロジスティック + プロビットリンクで）
bb = np.array([0.2, 0.3])
def ll_probit(b):
    return -nll_probit(b)
gnum = np.array([(ll_probit(bb + np.eye(2)[j] * 1e-6) - ll_probit(bb - np.eye(2)[j] * 1e-6)) / 2e-6 for j in range(2)])
eta = XL @ bb; pr = stats.norm.cdf(eta); mu = mL * pr; V = mL * pr * (1 - pr); dmu = mL * stats.norm.pdf(eta)
gform = XL.T @ ((yL - mu) / V * dmu)
check(T, "スコア方程式の一般形（プロビットリンクで数値微分と一致）", np.allclose(gnum, gform, atol=1e-5), np.round(gform, 4))
check(T, "正準リンクで dμ/dη=V(μ)、スコア=X'(y−μ)（ロジスティックで数値微分と一致）",
      np.allclose([(ll_logit(bb + np.eye(2)[j] * 1e-6) - ll_logit(bb - np.eye(2)[j] * 1e-6)) / 2e-6 for j in range(2)], XL.T @ (yL - mL * expit(XL @ bb)), atol=1e-5), "一致")
# 正規＋恒等 = OLS
Xn = np.column_stack([np.ones(6), [1, 2, 3, 4, 5, 6.]]); yn = np.array([1.2, 1.9, 3.2, 3.8, 5.1, 6.3])
check(T, "正規＋恒等の MLE = OLS（正規方程式）", np.allclose(np.linalg.lstsq(Xn, yn, rcond=None)[0], optimize.minimize(lambda b: np.sum((yn - Xn @ b) ** 2), [0, 0]).x, atol=1e-5), "一致")
check(T, "例1 β̂=(0.7501, 0.4313)、e^0.4313≈1.539", allclose(bP, [0.7501, 0.4313]) and close(exp(0.4313), 1.539, 5e-4), r4(bP))
check(T, "例1 当てはめ値 2.117…11.886", allclose(P["mu"], [2.117, 3.259, 5.016, 7.722, 11.886], 5e-4), r4(P["mu"], 3))
check(T, "例1 Σμ̂=30.000, Σxμ̂=84.000=Σxy", close(P["mu"].sum(), 30, 5e-4) and close(xP @ P["mu"], 84, 5e-4) and xP @ yP == 84, (round(P["mu"].sum(), 5), round(xP @ P["mu"], 5)))
check(T, "例1 x=5 の予測 ≈18.30（厳密な MLE で 18.2956。丸めた係数では 18.2945）", close(np.exp(bP[0] + 5 * bP[1]), 18.30, 5e-3), round(float(np.exp(bP[0] + 5 * bP[1])), 4))
e3 = bL[0] + 3 * bL[1]
check(T, "例2 x=3: η=0.7693、p=0.6834、20p=13.67", close(e3, 0.7693, 1e-4) and close(expit(e3), 0.6834) and close(20 * expit(e3), 13.67, 5e-3), (round(e3, 5), round(20 * expit(e3), 4)))
check(T, "例2 V(μ)=μ(1−μ/20)=20p(1−p)、割合では p(1−p)/20", sp.simplify((muv * (1 - muv / 20)).subs(muv, 20 * pp_s) - 20 * pp_s * (1 - pp_s)) == 0, "sympy")
c = chart(T, "line-data")
check(T, "chart line-data（ポアソン当てはめ値）", c["params"]["series"][0]["y"] == [2, 3, 6, 7, 12] and allclose(c["params"]["series"][1]["y"], P["mu"], 2e-4), c["params"]["series"][1]["y"])
c = chart(T, "logistic-curve")
check(T, "chart logistic-curve β0=−2.048, β1=0.939", c["params"]["beta0"] == round(bL[0], 3) and c["params"]["beta1"] == round(bL[1], 3), c["params"])

# ================= multinomial-logit-model =================
T = "multinomial-logit-model"
def mnl(x):
    e = np.array([np.ones_like(x), np.exp(-1 + 0.5 * x), np.exp(-2 + 1.0 * x)])
    return e / e.sum(0)
pi0 = mnl(np.array([0.0]))[:, 0]
check(T, "例1 x=0: e^−1=0.3679, e^−2=0.1353, 合計 1.5032", close(exp(-1), 0.3679) and close(exp(-2), 0.1353) and close(1 + exp(-1) + exp(-2), 1.5032), round(1 + exp(-1) + exp(-2), 5))
check(T, "例1 x=0: (0.6652, 0.2447, 0.0900)", allclose(pi0, [0.6652, 0.2447, 0.0900]), r4(pi0))
check(T, "例1 x=2: 3つとも 1/3", allclose(mnl(np.array([2.0]))[:, 0], [1 / 3] * 3, 1e-12), "1/3")
check(T, "例1 オッズ比 e^0.5=1.649, e^1=2.718, e^0.5（プレミアム対スタンダード）", close(exp(0.5), 1.649, 5e-4) and close(exp(1), 2.718, 5e-4), (round(exp(0.5), 4), round(exp(1), 4)))
xx = np.linspace(-5, 10, 150001)
xmax = xx[np.argmax(mnl(xx)[1])]
check(T, "例1 スタンダードの確率は x=2 で最大（以後減少）", abs(xmax - 2) < 1e-3, round(xmax, 4))
c = chart(T, "line-data")
gx = np.array(c["params"]["x"], float)
pr_ = mnl(gx)
check(T, "chart 3プランの確率 9点×3", all(allclose(c["params"]["series"][k]["y"], pr_[k]) for k in range(3)), [r4(pr_[k]) for k in range(3)])
check(T, "chart 各 x で合計1（4桁丸め誤差内）", np.all(np.abs(np.sum([c["params"]["series"][k]["y"] for k in range(3)], 0) - 1) < 2e-4), "OK")
# 基準カテゴリを変えても確率は同じ / K=2 でロジスティック
B = np.array([[0, 0], [-1, 0.5], [-2, 1.0]])
B2 = B - B[1]
x_ = 0.7
pA = np.exp(B @ [1, x_]); pA /= pA.sum(); pB = np.exp(B2 @ [1, x_]); pB /= pB.sum()
check(T, "基準をスタンダードに変えても確率同じ", np.allclose(pA, pB), r4(pA))
check(T, "K=2 でロジスティック回帰に一致", close(np.exp(-1 + 0.5 * x_) / (1 + np.exp(-1 + 0.5 * x_)), expit(-1 + 0.5 * x_), 1e-14), "一致")
check(T, "パラメータ数 (K−1)(p+1)：K=3,p=1 で 4", (3 - 1) * (1 + 1) == B[1:].size, 4)
# IIA：選択肢を除いても比は不変
pAB = np.exp(B[:2] @ [1, x_]); pAB /= pAB.sum()
check(T, "IIA：プレミアムを除いても π2/π1 は不変", close(pA[1] / pA[0], pAB[1] / pAB[0], 1e-12), round(pA[1] / pA[0], 5))
# 最尤推定の確認（シミュレーション）：ソフトマックスで生成→MLE で再現
nn = 20000; xsim = rng.normal(0, 1.5, nn)
probs = mnl(xsim).T
u = rng.random(nn)[:, None]
ysim = (u > np.cumsum(probs, 1)).sum(1)
Y = np.eye(3)[ysim]
def nll_mnl(v):
    Bv = np.vstack([[0, 0], v.reshape(2, 2)])
    eta = np.column_stack([np.ones(nn), xsim]) @ Bv.T
    return -(np.sum(Y * eta) - np.sum(np.log(np.exp(eta).sum(1))))
om = optimize.minimize(nll_mnl, np.zeros(4), method="BFGS")
check(T, "多項ロジット MLE がシミュレーションの真値 (−1,0.5,−2,1) を再現（2万件、許容 0.1）", np.all(np.abs(om.x - [-1, 0.5, -2, 1]) < 0.1), r4(om.x, 3))

# ================= ordered-logit-model =================
T = "ordered-logit-model"
def olog(x, a1=-1, a2=1, be=0.8):
    g1 = expit(a1 - be * x); g2 = expit(a2 - be * x)
    return np.array([g1, g2 - g1, 1 - g2])
o0 = olog(np.array([0.0]))[:, 0]
check(T, "例1 x=0: γ1=1/(1+e)=0.2689, γ2=0.7311", close(expit(-1), 0.2689) and close(expit(1), 0.7311), (round(expit(-1), 5), round(expit(1), 5)))
check(T, "例1 x=0: (0.2689, 0.4621, 0.2689)", allclose(o0, [0.2689, 0.4621, 0.2689]), r4(o0))
check(T, "例1 x=2: γ1=0.0691, γ2=0.3543", close(expit(-2.6), 0.0691) and close(expit(-0.6), 0.3543), (round(expit(-2.6), 5), round(expit(-0.6), 5)))
o2 = olog(np.array([2.0]))[:, 0]
check(T, "例1 x=2: (0.0691, 0.2852, 0.6457)", allclose(o2, [0.0691, 0.2852, 0.6457]), r4(o2))
check(T, "例1 累積オッズ e^−1=0.3679→0.0743、e^1=2.7183→0.5488", close(exp(-1), 0.3679) and close(exp(-2.6), 0.0743) and close(exp(1), 2.7183) and close(exp(-0.6), 0.5488), (round(exp(-2.6), 5), round(exp(-0.6), 5)))
check(T, "例1 比 e^−1.6=0.2019（両方の閾値で同じ）、逆数 4.953",
      close(exp(-1.6), 0.2019) and close(exp(-2.6) / exp(-1), exp(-0.6) / exp(1), 1e-14) and close(exp(1.6), 4.953, 5e-4), (round(exp(-1.6), 5), round(exp(1.6), 4)))
xx = np.linspace(-5, 5, 100001)
check(T, "「普通」のピークは x=(α1+α2)/(2β)=0", abs(xx[np.argmax(olog(xx)[1])]) < 1e-3, round(xx[np.argmax(olog(xx)[1])], 4))
c = chart(T, "line-data")
gx = np.array(c["params"]["x"], float)
oo = olog(gx)
check(T, "chart 3カテゴリの確率 9点×3", all(allclose(c["params"]["series"][k]["y"], oo[k]) for k in range(3)), [r4(oo[k]) for k in range(3)])
# 比例オッズ性（sympy）
x1s, x2s, al, bes = sp.symbols("x1 x2 alpha beta", real=True)
odds = lambda x: sp.exp(al - bes * x)  # γ/(1−γ) = e^{α−βx}
check(T, "累積オッズ比 = exp{−(x1−x2)β}、α によらない（sympy）", sp.simplify(odds(x1s) / odds(x2s) - sp.exp(-(x1s - x2s) * bes)) == 0, "差=0")
gam_s = 1 / (1 + sp.exp(-(al - bes * x1s)))
check(T, "logit γ = α − xβ（sympy）", sp.simplify(sp.log(gam_s / (1 - gam_s)) - (al - bes * x1s)) == 0 or
      all(abs(float((sp.log(gam_s / (1 - gam_s)) - (al - bes * x1s)).subs({al: a_, bes: b_, x1s: x__}))) < 1e-12 for a_, b_, x__ in [(-1, .8, 2), (1, .8, -1), (0.3, -1, 0.5)]), "OK")
# 潜在変数シミュレーション
eps = rng.logistic(size=400000)
ystar = 0.8 * 1.0 + eps
freq = np.array([(ystar <= -1).mean(), ((ystar > -1) & (ystar <= 1)).mean(), (ystar > 1).mean()])
check(T, "潜在変数 Y*=0.8x+ε を閾値 −1, 1 で区切ると x=1 の確率（40万回、許容 0.004）", np.all(np.abs(freq - olog(np.array([1.0]))[:, 0]) < 0.004), (r4(freq), r4(olog(np.array([1.0]))[:, 0])))
check(T, "パラメータ数 (K−1)+p=3 < (K−1)(p+1)=4（K=3,p=1）", (3 - 1) + 1 == 3 and (3 - 1) * 2 == 4, "3<4")


# =====================================================================
# 重要度Cの用語（tobit-model, offset-term）
# 乱数は seed=20261023+18
# =====================================================================
rngC = np.random.default_rng(20261023 + 18)

T = "tobit-model"
Phi5, phi5 = stats.norm.cdf(.5), stats.norm.pdf(.5)
check(T, "Φ(0.5)=0.6915、P(y=0)=0.3085、φ(0.5)=0.3521", close(Phi5, .6915) and close(1 - Phi5, .3085) and close(phi5, .3521), (Phi5, phi5))
Ey = Phi5 * 1 + 2 * phi5
check(T, "E[y]=1.3956", close(Ey, 1.3956), Ey)
Eyp = 1 + 2 * phi5 / Phi5
check(T, "E[y|y>0]=2.0183", close(Eyp, 2.0183), Eyp)
check(T, "限界効果 0.6915×0.5=0.3457", close(Phi5 * .5, .3457), Phi5 * .5)
from scipy.integrate import quad as quadC
Eint = quadC(lambda u: max(u, 0) * stats.norm.pdf(u, 1, 2), -30, 30, points=[0])[0]
check(T, "E[max(0,Y*)] を数値積分（Y*~N(1,2²)）でも 1.3956", close(Eint, Ey, 1e-8), Eint)
mu_s, s_s, xb = sp.symbols("mu sigma x", positive=True)
z_s = mu_s / s_s
Phi_s = lambda t: (1 + sp.erf(t / sp.sqrt(2))) / 2
phi_s = lambda t: sp.exp(-t ** 2 / 2) / sp.sqrt(2 * sp.pi)
EY_s = Phi_s(z_s) * mu_s + s_s * phi_s(z_s)
check(T, "∂E[y]/∂(xᵀβ)=Φ(z)（sympy）→ 限界効果 Φ(z)β_j", sp.simplify(sp.diff(EY_s, mu_s) - Phi_s(z_s)) == 0, "差=0")
ys_ = rngC.normal(1, 2, 1_000_000)
yo = np.maximum(ys_, 0)
check(T, "乱数 100万点：P(y=0)、E[y]、E[y|y>0]（許容 0.003, 0.006, 0.006）",
      abs((yo == 0).mean() - .3085) < .003 and abs(yo.mean() - Ey) < .006 and abs(yo[yo > 0].mean() - Eyp) < .006,
      ((yo == 0).mean(), yo.mean(), yo[yo > 0].mean()))
# 最尤推定（尤度の式）でパラメータを復元、正の値だけの OLS は 0 方向に偏る
nT = 5000
xt = rngC.uniform(-2, 2, nT)
yst = 0.5 + 1.0 * xt + rngC.normal(0, 1, nT)
yt = np.maximum(yst, 0)


def nll_tobit(th):
    b0, b1, ls = th
    sg = np.exp(ls)
    m = b0 + b1 * xt
    pos = yt > 0
    return -(np.sum(stats.norm.logsf(m[~pos] / sg)) + np.sum(stats.norm.logpdf((yt[pos] - m[pos]) / sg) - ls))


rT = optimize.minimize(nll_tobit, [0, 0.5, 0], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
check(T, "乱数 n=5000：本文の尤度の最尤推定で (0.5, 1.0, σ=1) を復元（±0.08）", np.all(np.abs(np.r_[rT.x[:2], np.exp(rT.x[2])] - [.5, 1, 1]) < .08), r4(np.r_[rT.x[:2], np.exp(rT.x[2])]))
pos = yt > 0
bpos = np.polyfit(xt[pos], yt[pos], 1)[0]
ball = np.polyfit(xt, yt, 1)[0]
check(T, "乱数 正の値だけ・0を含めた OLS の傾きはともに 1 より 0 側", 0 < bpos < .9 and 0 < ball < .9, (bpos, ball))

T = "offset-term"
tO = np.array([50, 100, 200, 400], float)
yO = np.array([6, 10, 14, 24], float)


def pois_fit(X, y, off):
    b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(100):
        mu = np.exp(X @ b + off)
        b = b + np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (y - mu))
    mu = np.exp(X @ b + off)
    dev = 2 * np.sum(y * np.log(y / mu) - (y - mu))
    return b, mu, dev, np.linalg.inv(X.T @ (mu[:, None] * X))


b1o, mu1o, dev1o, _ = pois_fit(np.ones((4, 1)), yO, np.log(tO))
check(T, "e^β̂0=54/750=0.072、β̂0=−2.6311", close(np.exp(b1o[0]), .072, 1e-12) and close(b1o[0], -2.6311), b1o)
check(T, "当てはめ値 3.6, 7.2, 14.4, 28.8", allclose(mu1o, [3.6, 7.2, 14.4, 28.8], 1e-10), r4(mu1o))
check(T, "逸脱度 3.1598（自由度3）", close(dev1o, 3.1598), dev1o)
X2o = np.column_stack([np.ones(4), np.log(tO)])
b2o, mu2o, dev2o, cov2o = pois_fit(X2o, yO, np.zeros(4))
check(T, "γ̂=0.6553（SE 0.1910）、逸脱度 0.0972", close(b2o[1], .6553) and close(sqrt(cov2o[1, 1]), .1910) and close(dev2o, .0972), (b2o, sqrt(cov2o[1, 1]), dev2o))
rO = optimize.minimize(lambda b: -np.sum(yO * (b[0] + b[1] * np.log(tO)) - np.exp(b[0] + b[1] * np.log(tO))), [0, 1], method="BFGS", options=dict(gtol=1e-10))
check(T, "γ̂ を scipy.optimize の最尤推定でも確認", close(rO.x[1], b2o[1], 1e-5), rO.x)
check(T, "逸脱度の差 3.0626、χ²(1) で p=0.0801 >0.05", close(dev1o - dev2o, 3.0626) and close(stats.chi2.sf(dev1o - dev2o, 1), .0801), (dev1o - dev2o, stats.chi2.sf(dev1o - dev2o, 1)))
check(T, "切片のみのとき e^β̂0=Σy/Σt（スコア方程式 Σ(y−t e^β)=0）", close(np.sum(yO - tO * np.exp(b1o[0])), 0, 1e-9), "ok")
check(T, "log μ=log t+xβ ⇔ μ/t=exp(xβ)（ばく露 2 倍で期待件数 2 倍）", close(np.exp(np.log(200) + b1o[0]) / np.exp(np.log(100) + b1o[0]), 2, 1e-12), "ok")
# 乱数：比例モデルから生成すると γ̂≈1
tt_ = rngC.uniform(10, 500, 3000)
yy_o = rngC.poisson(0.072 * tt_)
Xs_ = np.column_stack([np.ones(3000), np.log(tt_)])
bsim = np.array([np.log(yy_o.sum() / tt_.sum()), 1.0])
for _ in range(100):
    mu_ = np.exp(Xs_ @ bsim)
    bsim = bsim + np.linalg.solve(Xs_.T @ (mu_[:, None] * Xs_), Xs_.T @ (yy_o - mu_))
check(T, "乱数 比例モデルから生成すると log t の係数≈1（±0.06）", abs(bsim[1] - 1) < .06, bsim)

# ================= 集計 =================
n_ok = sum(results)
n_ng = len(results) - n_ok
print(f"\n合計: OK {n_ok} / NG {n_ng}")
