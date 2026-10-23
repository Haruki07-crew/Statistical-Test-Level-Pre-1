"""第29章（不完全データの統計処理）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats, integrate, optimize

q = Q(29)
rng = np.random.default_rng(29)
z975 = stats.norm.ppf(0.975)

# --- 欠測メカニズム ---
q.concept("q29-missing-data-mechanism-1", "場面から MNAR を判断（数値なし）")
q.concept("q29-missing-data-mechanism-2", "MCAR ⊂ MAR（定義の包含）")

# --- MCAR ---
q.num("q29-mcar-1", 10 / np.sqrt(200 * 0.64))
q.concept("q29-mcar-2", "MCAR はデータで証明できない（数値なし）")

# --- MAR ---
q.concept("q29-mar-1", "場面から MAR を判断（数値なし）")
cc = (0.5 * 0.8 * 10 + 0.5 * 0.4 * 20) / (0.5 * 0.8 + 0.5 * 0.4)
q.num("q29-mar-2", cc)
N = 400000
X = rng.integers(0, 2, N); Y = np.where(X == 0, 10, 20) + rng.normal(size=N)
R = rng.uniform(size=N) > np.where(X == 0, 0.2, 0.6)
q.check("q29-mar-2", abs(Y[R].mean() - cc) < 0.05 and abs(0.5 * Y[R & (X == 0)].mean() + 0.5 * Y[R & (X == 1)].mean() - 15) < 0.05,
        "シミュレーション：CC平均≈13.33、X 別に戻すと≈15")

# --- MNAR ---
q.concept("q29-mnar-1", "MAR と MNAR は観測データで区別できない（数値なし）")
q.num("q29-mnar-2", stats.truncnorm(-np.inf, 1).mean())
q.check("q29-mnar-2", abs(-stats.norm.pdf(1) / stats.norm.cdf(1) - stats.truncnorm(-np.inf, 1).mean()) < 1e-12, "公式と一致")

# --- 完全ケース ---
q.num("q29-complete-case-analysis-1", 0.9**5)
x = rng.normal(size=N); y = 1 + 2 * x + rng.normal(size=N)
r = rng.uniform(size=N) < stats.norm.cdf(1.5 * x)
b = np.polyfit(x[r], y[r], 1)
q.check("q29-complete-case-analysis-2", abs(b[0] - 2) < 0.02 and abs(b[1] - 1) < 0.02 and abs(y[r].mean() - y.mean()) > 0.5,
        "X 依存の欠測で回帰係数は偏らない（平均は偏る）")

# --- 単一代入 ---
q.num("q29-single-imputation-1", 39 * 25 / 49)
obs = rng.normal(size=40); obs = (obs - obs.mean()) / obs.std(ddof=1) * 5 + 50
filled = np.r_[obs, np.full(10, obs.mean())]
q.check("q29-single-imputation-1", abs(filled.var(ddof=1) - 975 / 49) < 1e-9, "実データで再現")
n = 2000; xx = rng.normal(size=n); yy = 0.5 * xx + rng.normal(size=n)
miss = rng.uniform(size=n) < 0.4
bb = np.polyfit(xx[~miss], yy[~miss], 1)
yimp = yy.copy(); yimp[miss] = np.polyval(bb, xx[miss])
q.check("q29-single-imputation-2", np.corrcoef(xx, yimp)[0, 1] > np.corrcoef(xx, yy)[0, 1] + 0.05, "回帰代入で相関が強くなりすぎる")

# --- 多重代入 ---
ms = [m for m in range(1, 50) if 1 / (1 + 0.4 / m) >= 0.95]
q.num("q29-multiple-imputation-1", ms[0])
q.check("q29-multiple-imputation-1", abs(1 / 1.08 - 0.926) < 0.001, "m=5 で 0.926")
# 固定パラメータの代入：B が過小になることを確認（平均の推定、n_obs=30, 欠測 30）
def mi_var(proper, reps=3000, m=20):
    covs = []
    for _ in range(reps):
        yo = rng.normal(size=30)
        qs, us = [], []
        for _ in range(m):
            mu, sd = yo.mean(), yo.std(ddof=1)
            if proper:
                sd = sd * np.sqrt(29 / rng.chisquare(29)); mu = rng.normal(yo.mean(), sd / np.sqrt(30))
            yc = np.r_[yo, rng.normal(mu, sd, 30)]
            qs.append(yc.mean()); us.append(yc.var(ddof=1) / 60)
        T = np.mean(us) + (1 + 1 / m) * np.var(qs, ddof=1)
        covs.append(abs(np.mean(qs)) < 1.96 * np.sqrt(T))
    return np.mean(covs)
cov_fixed, cov_proper = mi_var(False, 1500), mi_var(True, 1500)
q.check("q29-multiple-imputation-2", cov_fixed < 0.92 and cov_proper > 0.93, f"固定代入の被覆率 {cov_fixed:.3f} が名目より低い（誤り）")

# --- ルービンの規則 ---
Qh = np.array([10.2, 9.8, 10.5, 10.1, 9.9]); U = np.array([0.40, 0.42, 0.38, 0.41, 0.39]); m = 5
B = Qh.var(ddof=1); Ub = U.mean(); T = Ub + (1 + 1 / m) * B
q.num("q29-rubins-rules-1", T)
rr = (1 + 1 / m) * B / Ub
q.num("q29-rubins-rules-2", (m - 1) * (1 + 1 / rr) ** 2)
q.check("q29-rubins-rules-3", abs((Ub + B) - T) > 0.01, "U+B は T と一致しない（誤り）")

# --- EM ---
mu = 0.0; seq = []
for _ in range(200):
    mu = (64 + 2 * mu) / 10; seq.append(mu)
q.num("q29-em-algorithm-1", seq[1])
q.check("q29-em-algorithm-1", abs(seq[-1] - 8) < 1e-12, "不動点 8")
# 単調性：混合正規の EM で対数尤度が減らないことを確認
data = np.r_[rng.normal(-2, 1, 150), rng.normal(2, 1, 150)]
pi_, m1, m2 = 0.5, -0.5, 0.5; lls = []
for _ in range(60):
    p1 = pi_ * stats.norm.pdf(data, m1); p2 = (1 - pi_) * stats.norm.pdf(data, m2)
    lls.append(np.sum(np.log(p1 + p2)))
    w = p1 / (p1 + p2); pi_ = w.mean(); m1 = np.sum(w * data) / w.sum(); m2 = np.sum((1 - w) * data) / (1 - w).sum()
q.check("q29-em-algorithm-2", np.all(np.diff(lls) >= -1e-9), "混合正規の EM で対数尤度が単調非減少")
lam = 0.5
lam1 = 3 / (2 + 4 + (3 + 1 / lam))
q.num("q29-em-algorithm-3", lam1)
for _ in range(500):
    lam = 3 / (2 + 4 + 3 + 1 / lam)
q.check("q29-em-algorithm-3", abs(lam - 2 / 9) < 1e-9 and abs(stats.expon(scale=2).mean() - 2) < 1e-12, "収束先 2/9、E[T|T>3]=3+1/λ")

# --- 打ち切り ---
yv = np.array([2, 5, 3, 8, 4.0]); dv = np.array([1, 1, 0, 1, 0])
nll = lambda l: -(dv.sum() * np.log(l) - l * yv.sum())
lh = optimize.minimize_scalar(nll, bounds=(1e-4, 5), method="bounded").x
q.num("q29-censored-data-1", 1 / lh)
q.check("q29-censored-data-1", abs(1 / lh - 22 / 3) < 1e-3 and abs(np.mean([2, 5, 8]) - 5) < 1e-12, "数値最大化と一致")
q.concept("q29-censored-data-2", "打ち切りの種類（数値なし）")
q.concept("q29-censored-data-3", "打ち切りの寄与は S(y)（数値なし）")

# --- 切断 ---
v, _ = integrate.quad(lambda t: t * stats.norm.pdf(t) / 0.5, 0, np.inf)
q.num("q29-truncated-data-1", v)

# --- KM・グリーンウッド・NA ---
times = np.array([2, 3, 4, 4, 6, 7, 9, 10.0]); ev = np.array([1, 0, 1, 1, 0, 1, 1, 0])
def km(t):
    S = 1.0; gw = 0.0; H = 0.0
    for u in np.unique(times[ev == 1]):
        if u > t: break
        nj = np.sum(times >= u); dj = np.sum((times == u) & (ev == 1))
        S *= 1 - dj / nj; gw += dj / (nj * (nj - dj)) if nj > dj else 0; H += dj / nj
    return S, S**2 * gw, H
S5, V5, _ = km(5)
q.num("q29-kaplan-meier-estimator-1", S5)
q.num("q29-kaplan-meier-estimator-2", km(9)[0])
naive = np.mean(times[ev == 1] > 5)
q.check("q29-kaplan-meier-estimator-3", naive < S5 and np.isclose(naive, 0.4), "除外すると 0.4 < 0.583（過小＝誤り）")
q.num("q29-greenwood-formula-1", np.sqrt(V5))
H7 = km(7)[2]
q.num("q29-nelson-aalen-estimator-1", H7)
q.check("q29-nelson-aalen-estimator-1", np.exp(-H7) > km(7)[0], "exp(-H) ≥ KM")

# --- ログランク ---
A = [(2, 1), (5, 1), (7, 0), (9, 1)]; Bg = [(1, 1), (3, 1), (4, 1), (6, 0)]
allv = [(t, e, 0) for t, e in A] + [(t, e, 1) for t, e in Bg]
O = E = V = 0.0
for t in sorted({t for t, e, g in allv if e}):
    nA = sum(1 for s, e, g in allv if g == 0 and s >= t); nn = sum(1 for s, e, g in allv if s >= t)
    dA = sum(1 for s, e, g in allv if g == 0 and s == t and e); d = sum(1 for s, e, g in allv if s == t and e)
    O += dA; E += d * nA / nn; V += d * (nA / nn) * (1 - nA / nn) * (nn - d) / (nn - 1) if nn > 1 else 0
q.num("q29-log-rank-test-1", E)
q.num("q29-log-rank-test-2", (O - E) ** 2 / V)
q.concept("q29-log-rank-test-3", "交差するハザードで打ち消し合う性質（数値なし）")

# --- コックス ---
q.num("q29-cox-proportional-hazards-model-1", np.exp(0.8))
q.check("q29-cox-proportional-hazards-model-1", abs(2 * np.exp(0.4) - np.exp(0.8)) > 0.5, "誤答 2e^0.4 と異なる")
q.num("q29-cox-proportional-hazards-model-2", 0.8**2)
q.concept("q29-cox-proportional-hazards-model-3", "切片は h0 に吸収（数値なし）")

# --- 部分尤度 ---
pdat = [(2, 1, 1), (3, 1, 0), (4, 0, 0), (6, 1, 1)]
def logpl(beta, data):
    s = 0.0
    for t, e, xv in data:
        if e:
            risk = [x2 for t2, e2, x2 in data if t2 >= t]
            s += beta * xv - np.log(np.sum(np.exp(beta * np.array(risk))))
    return s
h = 1e-6
U0 = (logpl(h, pdat) - logpl(-h, pdat)) / (2 * h)
q.num("q29-partial-likelihood-1", U0)
bh1 = optimize.minimize_scalar(lambda b: -logpl(b, pdat), bounds=(-10, 10), method="bounded").x
pdat2 = [(np.log(t), e, xv) for t, e, xv in pdat]
bh2 = optimize.minimize_scalar(lambda b: -logpl(b, pdat2), bounds=(-10, 10), method="bounded").x
q.check("q29-partial-likelihood-2", abs(bh1 - bh2) < 1e-6, "log 変換しても β̂ は同じ（正しい）")

# --- ハザード比 ---
q.num("q29-hazard-ratio-1", (np.log(2) / 18) / (np.log(2) / 12))
q.num("q29-hazard-ratio-2", 0.667 * np.exp(z975 * np.sqrt(1 / 40 + 1 / 60)))

# --- 無視可能性 ---
q.concept("q29-ignorability-1", "MAR＋パラメータの分離（定義）")

# 校閲（review-ch27-33）で加えた値
from scipy import stats as _st
q.check("q29-mnar-2", abs(_st.norm.pdf(1) - 0.2420) < 5e-5 and abs(_st.norm.cdf(1) - 0.8413) < 5e-5
        and abs(-0.2420 / 0.8413 - (-0.288)) < 0.0005, "与えた φ(1)=0.2420、Φ(1)=0.8413 で -0.288")
_se = (7 / 12) * (1 / 56 + 2 / 24) ** 0.5
q.check("q29-greenwood-formula-1", abs(_se - 0.18556) < 1e-5 and abs(0.185 - 0.186) <= 0.0015, "SE=0.18556、途中丸めの 0.185 も許容")
q.done()
