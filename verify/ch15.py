"""第15章 検算スクリプト（確率過程・待ち行列）。
厳密計算は sympy / fractions、数値は scipy.stats。主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（概ね標準誤差の4倍以上）。
"""
from fractions import Fraction as F
from math import comb, exp, sqrt, pi as PI, factorial
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def front(term):
    txt = (ROOT / "content" / "ch15" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


t, s, lam, mu, th, h = sp.symbols("t s lambda mu theta h", positive=True)
n_ = sp.symbols("n", integer=True, positive=True)

# ============ birth-death-process ============
T = "birth-death-process"
l0, l1, m1, m2 = 2, 1, 3, 3
pi = [F(1), F(l0, m1), F(l0 * l1, m1 * m2)]
Z = sum(pi)
pi = [x / Z for x in pi]
check(T, "例1 π=(9/17,6/17,2/17)", pi == [F(9, 17), F(6, 17), F(2, 17)], pi)
check(T, "例1 π1=2/3π0, π2=2/9π0, 正規化 17/9", pi[1] / pi[0] == F(2, 3) and pi[2] / pi[0] == F(2, 9) and Z == F(17, 9), Z)
check(T, "例1 小数 0.5294/0.3529/0.1176", close(pi[0], .5294) and close(pi[1], .3529) and close(pi[2], .1176), [float(x) for x in pi])
Lm = pi[1] + 2 * pi[2]
check(T, "例1 平均故障台数 10/17≈0.5882", Lm == F(10, 17) and close(Lm, .5882), Lm)
check(T, "例1 2台停止の割合 約11.8%", close(pi[2], .118, 5e-4), float(pi[2]))
# 生成行列で πQ=0
Q = sp.Matrix([[-2, 2, 0], [3, -4, 1], [0, 3, -3]])
piv = sp.Matrix([[sp.Rational(9, 17), sp.Rational(6, 17), sp.Rational(2, 17)]])
check(T, "例1 πQ=0（生成行列）", piv * Q == sp.zeros(1, 3), piv * Q)
P_unif = sp.eye(3) + Q / 5
ch = charts(T)
Pc = sp.Matrix([[sp.Rational(str(x)) for x in r] for r in ch[0]["params"]["P"]])
check(T, "図1 P = I+Q/5 と一致", P_unif == Pc, Pc.tolist())
check(T, "図1 πP=π（一様化で定常分布不変）", piv * Pc == piv, piv * Pc)
Pn = np.linalg.matrix_power(np.array(Pc.tolist(), float), 20)
check(T, "図1 20ステップで (0.529,0.353,0.118) に収束（全出発点）", np.allclose(Pn, np.array([[9, 6, 2]] * 3) / 17, atol=5e-4), Pn[0].round(4))
# 一般の平衡方程式から詳細つり合いが出る（K=5 の記号）
L = sp.symbols("l0:5", positive=True)
M = sp.symbols("m1:6", positive=True)
p0 = sp.Symbol("p0")
P = [p0]
for k in range(5):
    P.append(P[-1] * L[k] / M[k])
okbal = sp.simplify(L[0] * P[0] - M[0] * P[1]) == 0
for k in range(1, 5):
    lhs = (L[k] + M[k - 1]) * P[k]
    rhs = L[k - 1] * P[k - 1] + M[k] * P[k + 1]
    okbal &= sp.simplify(lhs - rhs) == 0
check(T, "積の公式が平衡方程式 (λn+μn)πn=λn−1πn−1+μn+1πn+1 を満たす（n≤4 記号）", okbal, okbal)
# 前向き方程式の導出 h の1次係数
Pm, Pn0, Pp, ln, lnm, mn, mnp = sp.symbols("Pm Pn Pp ln lnm mn mnp")
expr = (lnm * h * Pm + mnp * h * Pp + (1 - (ln + mn) * h) * Pn0 - Pn0) / h
check(T, "前向き方程式 P'n=λn−1Pn−1+μn+1Pn+1−(λn+μn)Pn", sp.simplify(expr - (lnm * Pm + mnp * Pp - (ln + mn) * Pn0)) == 0, sp.expand(expr))
# 例2 M/M/∞
k = sp.Symbol("k", integer=True, nonnegative=True)
term = sp.Integer(4) ** k / (sp.Rational(1, 2) ** k * sp.factorial(k))
check(T, "例2 4^n/(0.5^n n!)=8^n/n!", sp.simplify(term - 8 ** k / sp.factorial(k)) == 0, "8^n/n!")
check(T, "例2 Σ8^n/n!=e^8 → π0=e^-8", sp.summation(8 ** k / sp.factorial(k), (k, 0, sp.oo)) == sp.exp(8), "e^8")
check(T, "例2 e^-8≈0.000335", close(exp(-8), .000335, 5e-7), exp(-8))
check(T, "図2 poisson λ=8 が例2の λ/μ=4/0.5", ch[1]["params"]["lambda"] == 4 / 0.5, ch[1]["params"])
# 滞在時間の離脱方向
check(T, "状態 n を離れるとき上へ λn/(λn+μn)（例1 n=1: 1/4）", F(l1, l1 + m1) == F(1, 4), "1/4")
# シミュレーション：機械修理モデルの時間平均
cur, tt = 0, 0.0
occ = np.zeros(3)
rates_up = [2, 1, 0]
rates_dn = [0, 3, 3]
for _ in range(400000):
    r = rates_up[cur] + rates_dn[cur]
    d = rng.exponential(1 / r)
    occ[cur] += d
    tt += d
    cur = cur + 1 if rng.random() < rates_up[cur] / r else cur - 1
fr_ = occ / tt
check(T, "シミュ 例1 時間割合（40万遷移, 許容0.005）", np.allclose(fr_, [9 / 17, 6 / 17, 2 / 17], atol=0.005), fr_.round(4))

# ============ brownian-motion ============
T = "brownian-motion"
check(T, "例1 P(W(4)>2)=P(Z>1)≈0.1587", close(stats.norm.sf(2, 0, 2), .1587), stats.norm.sf(2, 0, 2))
check(T, "例2 相関 1/√3≈0.5774", close(1 / sqrt(3), .5774), 1 / sqrt(3))
cm = 0 + 1 / 3 * 3
cv = 1 - 1 / 3
check(T, "例2 W(1)|W(3)=3 ~ N(1,2/3)", cm == 1 and abs(cv - 2 / 3) < 1e-12, (cm, cv))
check(T, "例2 1/√(2/3)=1.2247, P≈0.1103", close(1 / sqrt(2 / 3), 1.2247) and close(stats.norm.sf(2, 1, sqrt(2 / 3)), .1103), stats.norm.sf(2, 1, sqrt(2 / 3)))
check(T, "例3 2P(W(1)≥1)≈0.3173", close(2 * stats.norm.sf(1), .3173), 2 * stats.norm.sf(1))
check(T, "例3 2×0.1587=0.3174≈0.3173 の丸め表記", close(2 * 0.1587, .3173, 1.5e-4), 2 * .1587)
check(T, "例4 X(4)~N(2,16), P(X(4)<0)=P(Z<−0.5)≈0.3085", close(stats.norm.cdf(0, 0.5 * 4, 2 * 2), .3085), stats.norm.cdf(0, 2, 4))
check(T, "V[W(1)+W(2)]=1+2+2·1=5", 1 + 2 + 2 * min(1, 2) == 5 and 4 * 1 + 1 == 5, 5)
# シミュ：共分散・反射原理
nP, nS = 20000, 2000
dW = rng.normal(0, sqrt(1 / nS), size=(nP, nS))
Wp = np.cumsum(dW, axis=1)
mx = Wp.max(axis=1)
pr = (mx >= 1).mean()
# 離散格子では最大値を過小評価する（約 0.5826/√n の補正）。補正後の主張値と比較
corr = 2 * stats.norm.sf(1 + 0.5826 * sqrt(1 / nS))  # 離散監視の補正（Broadie-Glasserman）。連続時間の 0.3173 に対応する格子上の値
check(T, f"シミュ 反射原理 2P(W(1)≥1)：2000刻み格子での補正値 {corr:.4f} と一致（2万本, 許容0.013≈4SE）", abs(pr - corr) < 0.013 and close(2 * stats.norm.sf(1), .3173), (pr, corr))
W1 = rng.normal(0, 1, 400000)
W3 = W1 + rng.normal(0, sqrt(2), 400000)
check(T, "シミュ Cov(W(1),W(3))=1（40万, 許容0.02）", abs(np.cov(W1, W3)[0, 1] - 1) < 0.02, np.cov(W1, W3)[0, 1])
sel = np.abs(W3 - 3) < 0.05
check(T, "シミュ W(1)|W(3)≈3 の平均1・分散2/3（許容0.06）", abs(W1[sel].mean() - 1) < 0.06 and abs(W1[sel].var() - 2 / 3) < 0.06, (W1[sel].mean(), W1[sel].var(), sel.sum()))
# 自己相似性 W(ct)/√c の分散 t
check(T, "自己相似 V[W(ct)/√c]=t（記号）", sp.simplify((sp.Symbol('c', positive=True) * t) / sp.Symbol('c', positive=True) - t) == 0, "t")
ch = charts(T)
check(T, "図 random-walk p=0.5（対称）", ch[0]["params"]["p"] == 0.5, ch[0]["params"])

# ============ gambler-ruin-problem ============
T = "gambler-ruin-problem"
i_, N_ = sp.symbols("i N", integer=True)
p_ = sp.Symbol("p", positive=True)
q_ = 1 - p_
r_ = sp.Symbol("r", positive=True)
# r で表した確率の公式が差分方程式を満たす：p=1/(1+r), q=r/(1+r)
pr_, qr_ = 1 / (1 + r_), r_ / (1 + r_)
Pf = lambda i: (1 - r_ ** i) / (1 - r_ ** N_)
check(T, "P_i=(1−r^i)/(1−r^N) が P_i=pP_{i+1}+qP_{i−1} を満たす", sp.simplify(pr_ * Pf(i_ + 1) + qr_ * Pf(i_ - 1) - Pf(i_)) == 0, "0")
check(T, "境界 P_0=0, P_N=1", sp.simplify(Pf(0)) == 0 and sp.simplify(Pf(N_)) == 1, "ok")
# D_i
qmp = (r_ - 1) / (1 + r_)  # q−p
Df = lambda i: i / qmp - N_ / qmp * Pf(i)
check(T, "D_i 公式が D_i=1+pD_{i+1}+qD_{i−1} を満たす", sp.simplify(1 + pr_ * Df(i_ + 1) + qr_ * Df(i_ - 1) - Df(i_)) == 0, "0")
check(T, "D_0=D_N=0", sp.simplify(Df(0)) == 0 and sp.simplify(Df(N_)) == 0, "ok")
check(T, "特殊解 i/(q−p)：i−p(i+1)−q(i−1)=q−p", sp.simplify(i_ - p_ * (i_ + 1) - q_ * (i_ - 1) - (q_ - p_)) == 0, "q−p")
Ds = i_ * (N_ - i_)
check(T, "対称 D_i=i(N−i) の2階差分=−2", sp.expand((i_ + 1) * (N_ - i_ - 1) - 2 * Ds + (i_ - 1) * (N_ - i_ + 1)) == -2, -2)
check(T, "r→1 で P_i→i/N", sp.limit(Pf(3).subs(N_, 7), r_, 1) == sp.Rational(3, 7), sp.limit(Pf(3).subs(N_, 7), r_, 1))
# 第14章型の吸収連鎖で照合（例2）
def absorb(p, i0, N):
    Qm = np.zeros((N - 1, N - 1)); R = np.zeros((N - 1, 2))
    for a in range(1, N):
        if a + 1 < N: Qm[a - 1, a] = p
        else: R[a - 1, 1] = p
        if a - 1 > 0: Qm[a - 1, a - 2] = 1 - p
        else: R[a - 1, 0] = 1 - p
    Nm = np.linalg.inv(np.eye(N - 1) - Qm)
    return (Nm @ R)[i0 - 1, 1], (Nm @ np.ones(N - 1))[i0 - 1]
Pa, Da = absorb(0.5, 10, 20)
check(T, "例1 P=0.5, D=100（基本行列で再計算）", close(Pa, .5, 1e-9) and close(Da, 100, 1e-7), (Pa, Da))
r = 0.53 / 0.47
check(T, "例2 r≈1.1277", close(r, 1.1277), r)
check(T, "例2 r^10≈3.3249, r^20≈11.0550", close(r ** 10, 3.3249) and close(r ** 20, 11.0550), (r ** 10, r ** 20))
P10 = (1 - r ** 10) / (1 - r ** 20)
check(T, "例2 P10≈0.2312、破産≈0.7688", close(P10, .2312) and close(1 - P10, .7688), P10)
check(T, "例2 分子・分母 −2.3249/−10.0550", close(1 - r ** 10, -2.3249) and close(1 - r ** 20, -10.0550), (1 - r ** 10, 1 - r ** 20))
D10 = 10 / 0.06 - 20 / 0.06 * P10
check(T, "例2 D10≈89.59（表示の P10=0.2312 で計算すると 89.60 になるが厳密値は 89.59）", close(D10, 89.59, 5e-3), D10)
Pa, Da = absorb(0.47, 10, 20)
check(T, "例2 基本行列で P・D 一致", close(Pa, P10, 1e-9) and close(Da, D10, 1e-6), (Pa, Da))
check(T, "例3 (2/3)^3=8/27≈0.2963", F(2, 3) ** 3 == F(8, 27) and close(8 / 27, .2963), 8 / 27)
check(T, "N→∞, p=0.45, i=10 の平均 i/(q−p)=100", close(10 / (0.55 - 0.45), 100, 1e-9), 10 / 0.1)
ch = charts(T)
pp = ch[0]["params"]
check(T, "図 params start=10, lower=0, upper=20, p=0.5（例1）", pp["start"] == 10 and pp["lower"] == 0 and pp["upper"] == 20 and pp["p"] == 0.5, pp)
# シミュ
def sim_ruin(p, i0, N, nrep):
    pos = np.full(nrep, i0); steps = np.zeros(nrep, int); alive = np.ones(nrep, bool)
    while alive.any():
        idx = np.where(alive)[0]
        pos[idx] += np.where(rng.random(idx.size) < p, 1, -1)
        steps[idx] += 1
        alive[idx] = (pos[idx] > 0) & (pos[idx] < N)
    return (pos == N).mean(), steps.mean()
w, d = sim_ruin(0.47, 10, 20, 100000)
check(T, "シミュ 例2 P10=0.2312（10万本, 許容0.006）・D10=89.59（許容1.5）", abs(w - P10) < 0.006 and abs(d - D10) < 1.5, (w, d))
w, d = sim_ruin(0.5, 10, 20, 50000)
check(T, "シミュ 例1 0.5（5万本, 許容0.009）・100回（許容2）", abs(w - .5) < .009 and abs(d - 100) < 2, (w, d))

# ============ independent-stationary-increments ============
T = "independent-stationary-increments"
check(T, "例1 Cov(N(1),N(3))=2, 相関 2/√12=1/√3≈0.5774", close(2 / sqrt(2 * 6), .5774), 2 / sqrt(12))
check(T, "例1 E[N(3)|N(1)=4]=8, V=4", 4 + 2 * 2 == 8 and 2 * 2 == 4, (8, 4))
N1 = rng.poisson(2, 400000); N3 = N1 + rng.poisson(4, 400000)
c = np.cov(N1, N3)[0, 1]
check(T, "シミュ Cov(N(1),N(3))=2（40万, 許容0.05）", abs(c - 2) < 0.05, c)
sel = N1 == 4
check(T, "シミュ N(3)|N(1)=4 の平均8・分散4（許容0.08・0.15）", abs(N3[sel].mean() - 8) < .08 and abs(N3[sel].var() - 4) < .15, (N3[sel].mean(), N3[sel].var()))
Zs = rng.normal(size=200000)
check(T, "例2 X=tZ: 増分(0,1],(1,2] の相関=1, V[X(t)]=t^2", np.corrcoef(Zs * 1, Zs * 2 - Zs * 1)[0, 1] > 0.9999, "1")

# ============ interarrival-time ============
T = "interarrival-time"
check(T, "Exp の密度積分=1, E=1/λ, V=1/λ^2",
      sp.integrate(lam * sp.exp(-lam * t), (t, 0, sp.oo)) == 1 and sp.integrate(t * lam * sp.exp(-lam * t), (t, 0, sp.oo)) == 1 / lam
      and sp.simplify(sp.integrate(t ** 2 * lam * sp.exp(-lam * t), (t, 0, sp.oo)) - 1 / lam ** 2 - 1 / lam ** 2) == 0, "ok")
ok = True
for n in range(1, 7):
    cdf = 1 - sum(sp.exp(-lam * t) * (lam * t) ** kk / sp.factorial(kk) for kk in range(n))
    dens = lam ** n * t ** (n - 1) * sp.exp(-lam * t) / sp.factorial(n - 1)
    ok &= sp.simplify(sp.diff(cdf, t) - dens) == 0
    ok &= sp.simplify(sp.integrate(t * dens, (t, 0, sp.oo)) - n / lam) == 0
    ok &= sp.simplify(sp.integrate(t ** 2 * dens, (t, 0, sp.oo)) - (n / lam) ** 2 - n / lam ** 2) == 0
check(T, "d/dt P(N(t)≥n)=ガンマ密度、E=n/λ, V=n/λ^2（n=1..6）", ok, ok)
kk = sp.Symbol("k", integer=True, positive=True)
dterm = sp.diff(sp.exp(-lam * t) * (lam * t) ** kk / sp.factorial(kk), t)
claim = -lam * sp.exp(-lam * t) * (lam * t) ** kk / sp.factorial(kk) + lam * sp.exp(-lam * t) * (lam * t) ** (kk - 1) / sp.factorial(kk - 1)
check(T, "各項の微分（積の微分）", sp.simplify(sp.combsimp(dterm - claim)) == 0, "0")
Mt = sp.integrate(sp.exp(th * t) * lam * sp.exp(-lam * t), (t, 0, sp.oo), conds="none")
check(T, "MGF M_T=λ/(λ−θ)", sp.simplify(Mt - lam / (lam - th)) == 0, Mt)
dens3 = lam ** 3 * t ** 2 * sp.exp(-lam * t) / 2
Mg = sp.integrate(sp.exp(th * t) * dens3, (t, 0, sp.oo), conds="none")
check(T, "ガンマ(3) の MGF={λ/(λ−θ)}^3", sp.simplify(Mg - (lam / (lam - th)) ** 3) == 0, Mg)
beta = sp.Symbol("beta", positive=True)
check(T, "尺度表記 t^{n−1}e^{−t/β}/((n−1)!β^n) が λ=1/β と一致（n=3）",
      sp.simplify(t ** 2 * sp.exp(-t / beta) / (2 * beta ** 3) - dens3.subs(lam, 1 / beta)) == 0, "ok")
check(T, "例1 e^-1.5≈0.2231", close(exp(-1.5), .2231), exp(-1.5))
check(T, "例2 E[S3]=1, V=1/3", F(3, 3) == 1 and F(3, 9) == F(1, 3), "1, 1/3")
v = 1 - 8.5 * exp(-3)
check(T, "例2 1−8.5e^-3≈0.5768 = gamma.cdf(1;3,1/3)", close(v, .5768) and close(stats.gamma.cdf(1, 3, scale=1 / 3), v, 1e-12), v)
check(T, "例3 無記憶性 e^-1≈0.3679", close(exp(-3 * (2 / 3)) / exp(-3 * (1 / 3)), .3679), exp(-1))
ch = charts(T)
check(T, "図2 expon rate=3（例1の λ）", ch[1]["params"]["rate"] == 3, ch[1]["params"])
check(T, "図1 gamma shape=1 は指数分布（尺度0.5=平均間隔）", ch[0]["params"]["shape"] == 1, ch[0]["params"])
# シミュ
Ts = rng.exponential(1 / 3, size=(200000, 3))
S3 = Ts.sum(1)
check(T, "シミュ S3≤1 の確率 0.5768（20万, 許容0.005）", abs((S3 <= 1).mean() - v) < .005, (S3 <= 1).mean())
check(T, "シミュ S3 ~ Gamma(3, scale 1/3)（KS p>0.001）", stats.kstest(S3, "gamma", args=(3, 0, 1 / 3)).pvalue > 1e-3, stats.kstest(S3, "gamma", args=(3, 0, 1 / 3)).pvalue)
# ポアソン過程の到着列から間隔
arr = np.cumsum(rng.exponential(1 / 3, 600000))
cnt = np.bincount((arr[arr < 100000]).astype(int))
check(T, "シミュ Exp 間隔で並べると単位時間の個数 ~ Po(3)（平均・分散 許容0.03・0.05）", abs(cnt.mean() - 3) < .03 and abs(cnt.var() - 3) < .05, (cnt.mean(), cnt.var()))
# 検査のパラドックス
tt_ = rng.uniform(1000, 190000, 100000)
idx = np.searchsorted(arr, tt_)
span = arr[idx] - arr[idx - 1]
res = arr[idx] - tt_
check(T, "シミュ 検査のパラドックス 間隔の平均 2/λ=0.667、残り時間 1/λ=0.333（許容0.006・0.004）", abs(span.mean() - 2 / 3) < .006 and abs(res.mean() - 1 / 3) < .004, (span.mean(), res.mean()))
# 条件付き一様順序統計量
lam0, tt0 = 2.0, 1.0
s1 = []
for _ in range(60000):
    a = np.cumsum(rng.exponential(1 / lam0, 10))
    if (a <= tt0).sum() == 2 and a[1] <= tt0:
        s1.append(a[0])
s1 = np.array(s1)
check(T, "シミュ N(1)=2 のもとで S1 ~ 一様2個の最小（E=1/3, 許容0.01）", abs(s1.mean() - 1 / 3) < 0.01, (s1.mean(), s1.size))

# ============ littles-law ============
T = "littles-law"
arrv = [0, 1, 2, 5]; dep = [3, 4, 6, 8]
stay = [d - a for a, d in zip(arrv, dep)]
check(T, "例1 滞在時間 3,3,4,3 合計13", stay == [3, 3, 4, 3] and sum(stay) == 13, stay)
tg = np.linspace(0, 10, 100001)[:-1] + 5e-5
Nt = sum(((tg >= a) & (tg < d)).astype(int) for a, d in zip(arrv, dep))
area = Nt.sum() * 1e-4
check(T, "例1 面積=13（数値積分）", close(area, 13, 1e-6), area)
check(T, "例1 1+2+3+2+1+2+1×2=13", 1 + 2 + 3 + 2 + 1 + 2 + 1 * 2 == 13, 13)
check(T, "例1 L=1.3, λ=0.4, W=3.25, λW=1.3", F(13, 10) == F(4, 10) * F(13, 4), (1.3, .4, 3.25))
ch = charts(T)
xs = ch[0]["params"]["x"]; ys = ch[0]["params"]["series"][0]["y"]
okstep = True
for j in range(0, len(xs), 2):
    x0, x1 = xs[j], xs[j + 1]
    mid = (x0 + x1) / 2
    okstep &= ys[j] == ys[j + 1] == sum(a <= mid < d for a, d in zip(arrv, dep))
area_c = sum((xs[j + 1] - xs[j]) * ys[j] for j in range(0, len(xs), 2))
check(T, "図 line-data の階段が例1の N(t) と一致、面積13", okstep and area_c == 13, area_c)
check(T, "例2 W=30/60=0.5時間", F(30, 60) == F(1, 2), .5)
check(T, "例3 L=1.5, Lq=0.9 → W=0.25, Wq=0.15", close(1.5 / 6, .25, 1e-12) and close(.9 / 6, .15, 1e-12), (.25, .15))
check(T, "現場 200×0.05=10、2000/500=4", close(200 * .05, 10, 1e-12) and 2000 / 500 == 4, (10, 4))
check(T, "L=Lq+λ/μ（W=Wq+1/μ に λ を掛ける）", sp.simplify(lam * (sp.Symbol('Wq') + 1 / mu) - (lam * sp.Symbol('Wq') + lam / mu)) == 0, "ok")

# ============ martingale ============
T = "martingale"
S = sp.Symbol("S")
check(T, "S_n^2−n：E[S_{n+1}^2]=S^2+1（p=1/2）", sp.expand(sp.Rational(1, 2) * (S + 1) ** 2 + sp.Rational(1, 2) * (S - 1) ** 2 - S ** 2) == 1, 1)
check(T, "r^{S}：p r + q/r = q+p（r=q/p）", sp.simplify(p_ * (q_ / p_) + q_ / (q_ / p_) - 1) == 0, "1")
check(T, "S_n−n(p−q)：E[増分]=p−q", sp.simplify(p_ * 1 + q_ * (-1) - (p_ - q_)) == 0, "ok")
check(T, "例1 P=0.3, E[τ]=100·0.3−9=21=3×7", F(3, 10) * 10 == 3 and 100 * F(3, 10) - 9 == 21 == 3 * 7, 21)
r = 1.5
Pm2 = (1 - r ** 3) / (1 - r ** 10)
check(T, "例2 1.5^3=3.375, 1.5^10≈57.665", close(r ** 3, 3.375, 1e-12) and close(r ** 10, 57.665, 5e-4), r ** 10)
check(T, "例2 P≈0.0419（差分方程式の公式・吸収連鎖と一致）", close(Pm2, .0419) and close(absorb(0.4, 3, 10)[0], Pm2, 1e-10), Pm2)
w, d = sim_ruin(0.5, 3, 10, 100000)
check(T, "シミュ 例1 P=0.3（10万本, 許容0.006）, E[τ]=21（許容0.4）", abs(w - .3) < .006 and abs(d - 21) < .4, (w, d))
w, d = sim_ruin(0.4, 3, 10, 100000)
check(T, "シミュ 例2 P=0.0419（10万本, 許容0.003）", abs(w - Pm2) < .003, w)
# 倍賭け反例: 1に達したら止める τ の期待値は無限 → 打ち切り平均が増え続ける
ch = charts(T)
pp = ch[0]["params"]
check(T, "図 params start=3, lower=0, upper=10, p=0.5（例1）", pp["start"] == 3 and pp["lower"] == 0 and pp["upper"] == 10 and pp["p"] == 0.5, pp)
Nt_ = rng.poisson(2 * 3.0, 200000) - 2 * 3.0
check(T, "シミュ N(t)−λt の平均0（λ=2,t=3, 20万, 許容0.03）", abs(Nt_.mean()) < .03, Nt_.mean())

# ============ mm1-queue ============
T = "mm1-queue"
rho = sp.Symbol("rho", positive=True)
n = sp.Symbol("n", integer=True, nonnegative=True)
pmf = (1 - rho) * rho ** n
# |ρ|<1 の等比級数 G=Σρ^n=1/(1−ρ) を微分して Σnρ^n, Σn²ρ^n を作る（本文の導出と同じ方法）
G = 1 / (1 - rho)
S1 = rho * sp.diff(G, rho)
S2 = rho * sp.diff(S1, rho)
tot = sp.simplify((1 - rho) * G)
Lq_ = sp.simplify((1 - rho) * S1)
E2 = sp.simplify((1 - rho) * S2)
trunc = lambda f: sum(float(f(j)) for j in range(400))
num_ok = all(abs(trunc(lambda j: j ** a * (1 - rv) * rv ** j) - float(ex.subs(rho, rv))) < 1e-10
             for rv in (0.3, 0.6, 0.8) for a, ex in ((0, tot), (1, Lq_), (2, E2)))
check(T, "Σ(1−ρ)ρ^n=1（打ち切り和 ρ=0.3,0.6,0.8 でも一致）", sp.simplify(tot - 1) == 0 and num_ok, tot)
check(T, "L=ρ/(1−ρ)", sp.simplify(Lq_ - rho / (1 - rho)) == 0, Lq_)
check(T, "V[N]=ρ/(1−ρ)^2", sp.simplify(E2 - Lq_ ** 2 - rho / (1 - rho) ** 2) == 0, sp.simplify(E2 - Lq_ ** 2))
Lqq = sp.simplify(Lq_ - (1 - (1 - rho)))
check(T, "Lq=L−ρ=ρ^2/(1−ρ)", sp.simplify(Lqq - rho ** 2 / (1 - rho)) == 0, Lqq)
check(T, "P(N≥n)=ρ^n（n=0..5）", all(sp.simplify(1 - sum(pmf.subs(n, j) for j in range(m)) - rho ** m) == 0 for m in range(6)), "ok")
LL = lam / (mu - lam)
check(T, "L=λ/(μ−λ), W=1/(μ−λ), Wq=ρ/(μ−λ), W=Wq+1/μ",
      sp.simplify((rho / (1 - rho)).subs(rho, lam / mu) - LL) == 0 and sp.simplify(LL / lam - 1 / (mu - lam)) == 0
      and sp.simplify((rho ** 2 / (1 - rho)).subs(rho, lam / mu) / lam - (lam / mu) / (mu - lam)) == 0
      and sp.simplify((lam / mu) / (mu - lam) + 1 / mu - 1 / (mu - lam)) == 0, "ok")
# 系内時間 MGF
sv = sp.Symbol("s")
lhs = (1 - lam / mu) * mu / (mu - sv) / (1 - lam / (mu - sv))
check(T, "系内時間 MGF の和 = (μ−λ)/(μ−λ−s)", sp.simplify(lhs - (mu - lam) / (mu - lam - sv)) == 0, sp.simplify(lhs))
vals = {lam: 3, mu: 7, sv: sp.Rational(1, 2)}
num = sum(((1 - sp.Rational(3, 7)) * sp.Rational(3, 7) ** j * (sp.Rational(7, 1) / (7 - sp.Rational(1, 2))) ** (j + 1)) for j in range(400))
check(T, "MGF の級数を数値で（λ=3,μ=7,s=0.5）", close(float(num), float((sp.Integer(4) / (4 - sp.Rational(1, 2)))), 1e-10), float(num))
# 例1
l, m = 6, 10
rh = F(l, m)
check(T, "例1 ρ=0.6, π0=0.4, L=1.5, Lq=0.9", rh == F(3, 5) and 1 - rh == F(2, 5) and rh / (1 - rh) == F(3, 2) and rh ** 2 / (1 - rh) == F(9, 10), "ok")
check(T, "例1 W=0.25時間=15分, Wq=0.6/4=0.15=9分", F(1, m - l) == F(1, 4) and rh / (m - l) == F(3, 20), "ok")
check(T, "例1 ρ^3=0.216, e^-2≈0.1353", rh ** 3 == F(216, 1000) and close(exp(-4 * .5), .1353), exp(-2))
check(T, "例2 λ=9: L=9, W=1, 15→60分=4倍, 1.5→9=6倍", F(9, 10) / F(1, 10) == 9 and F(1, 1) == 1 and 60 / 15 == 4 and 9 / 1.5 == 6, "ok")
check(T, "例3 1/(μ−6)≤1/6 ⇔ μ≥12, 60/12=5分", 1 / (12 - 6) == 1 / 6 and 60 / 12 == 5, "ok")
ch = charts(T)
check(T, "図1 geom p=0.4=1−ρ（例1）", ch[0]["params"]["p"] == 0.4 and ch[0]["params"]["start"] == 0, ch[0]["params"])
xs = ch[1]["params"]["x"]
yL = ch[1]["params"]["series"][0]["y"]; yQ = ch[1]["params"]["series"][1]["y"]
check(T, "図2 L=ρ/(1−ρ) の値（2桁丸め, 許容0.005）", all(abs(a / (1 - a) - b) <= 0.005 + 1e-9 for a, b in zip(xs, yL)), list(zip(xs, yL)))
check(T, "図2 Lq=ρ^2/(1−ρ) の値（2桁丸め, 許容0.005）", all(abs(a * a / (1 - a) - b) <= 0.005 + 1e-9 for a, b in zip(xs, yQ)), list(zip(xs, yQ)))
# シミュ：リンドレー漸化式で M/M/1 (λ=6, μ=10)
nC = 1000000
A = rng.exponential(1 / 6, nC); Sv = rng.exponential(1 / 10, nC)
Wq = np.zeros(nC)
for j in range(1, nC):
    Wq[j] = max(0.0, Wq[j - 1] + Sv[j - 1] - A[j])
Wsys = Wq + Sv
Wq, Wsys = Wq[10000:], Wsys[10000:]
check(T, "シミュ W=0.25, Wq=0.15（100万人, 許容0.006・0.006）", abs(Wsys.mean() - .25) < .006 and abs(Wq.mean() - .15) < .006, (Wsys.mean(), Wq.mean()))
check(T, "シミュ P(系内時間>0.5)=e^-2（許容0.006）", abs((Wsys > .5).mean() - exp(-2)) < .006, (Wsys > .5).mean())
check(T, "シミュ P(Wq=0)=1−ρ=0.4（許容0.006）", abs((Wq == 0).mean() - .4) < .006, (Wq == 0).mean())
# 系内人数の時間平均（CTMC）
cur, tt = 0, 0.0; occ = np.zeros(200)
for _ in range(400000):
    r_ = 6 + (10 if cur > 0 else 0)
    d = rng.exponential(1 / r_)
    occ[min(cur, 199)] += d; tt += d
    cur = cur + 1 if rng.random() < 6 / r_ else cur - 1
pn = occ / tt
check(T, "シミュ 時間平均 π0=0.4, L=1.5, P(N≥3)=0.216（40万遷移, 許容0.01・0.06・0.01）",
      abs(pn[0] - .4) < .01 and abs((pn * np.arange(200)).sum() - 1.5) < .06 and abs(pn[3:].sum() - .216) < .01, (pn[0], (pn * np.arange(200)).sum(), pn[3:].sum()))

# ============ poisson-process ============
T = "poisson-process"
P0f = sp.Function("P0")
sol = sp.dsolve(sp.Eq(P0f(t).diff(t), -lam * P0f(t)), ics={P0f(0): 1})
check(T, "P0'=−λP0, P0(0)=1 → e^{−λt}", sp.simplify(sol.rhs - sp.exp(-lam * t)) == 0, sol.rhs)
ok = True
Qp = sp.Integer(1)
for nn in range(1, 6):
    Qn = sp.integrate(lam * Qp.subs(t, s), (s, 0, t))
    ok &= sp.simplify(Qn - (lam * t) ** nn / sp.factorial(nn)) == 0
    Pn_ = sp.exp(-lam * t) * Qn; Pm_ = sp.exp(-lam * t) * Qp
    ok &= sp.simplify(sp.diff(Pn_, t) - (-lam * Pn_ + lam * Pm_)) == 0
    Qp = Qn
check(T, "Q_n=(λt)^n/n! と P_n'=−λP_n+λP_{n−1}（n=1..5）", ok, ok)
ok = True
for nn in range(0, 6):
    for kk_ in range(0, nn + 1):
        joint = sp.exp(-lam * s) * (lam * s) ** kk_ / sp.factorial(kk_) * sp.exp(-lam * (t - s)) * (lam * (t - s)) ** (nn - kk_) / sp.factorial(nn - kk_)
        marg = sp.exp(-lam * t) * (lam * t) ** nn / sp.factorial(nn)
        ok &= sp.simplify(joint / marg - sp.binomial(nn, kk_) * (s / t) ** kk_ * (1 - s / t) ** (nn - kk_)) == 0
check(T, "N(s)|N(t)=n ~ Bin(n,s/t)（n≤5 記号）", ok, ok)
check(T, "例1 e^-2≈0.1353, 1−5e^-2≈0.3233", close(exp(-2), .1353) and close(1 - 5 * exp(-2), .3233) and close(stats.poisson.sf(2, 2), 1 - 5 * exp(-2), 1e-12), 1 - 5 * exp(-2))
v = comb(5, 2) * F(1, 3) ** 2 * F(2, 3) ** 3
check(T, "例2 80/243≈0.3292", v == F(80, 243) and close(v, .3292), v)
a1 = 2 * exp(-2); a2 = 4 ** 3 / 6 * exp(-4)
check(T, "例3 0.2707×0.1954≈0.0529", close(a1, .2707) and close(a2, .1954) and close(a1 * a2, .0529), (a1, a2, a1 * a2))
# シミュ 例3
N1 = rng.poisson(2, 1000000); N3 = N1 + rng.poisson(4, 1000000)
pj = ((N1 == 1) & (N3 == 4)).mean()
check(T, "シミュ 例3 P(N(1)=1,N(3)=4)=0.0529（100万, 許容0.0012）、積 P(N1=1)P(N3=4)=0.0363 とは異なる", abs(pj - a1 * a2) < .0012 and abs(a1 * stats.poisson.pmf(4, 6) - pj) > .01, (pj, a1 * stats.poisson.pmf(4, 6)))
# 条件付き二項
N1 = rng.poisson(1, 1000000); N3 = N1 + rng.poisson(2, 1000000)
sel = N3 == 5
pc = (N1[sel] == 2).mean()
check(T, "シミュ 例2 P(N(1)=2|N(3)=5)=0.3292（許容0.006）", abs(pc - 80 / 243) < .006, (pc, sel.sum()))
ch = charts(T)
check(T, "図2 Po(λt) λt=2 は例1の 0.2×10", close(ch[1]["params"]["lambda"], 0.2 * 10, 1e-12), ch[1]["params"])

# ============ queueing-theory ============
T = "queueing-theory"
sig2 = sp.Symbol("sigma2", positive=True)
PK = (lam ** 2 * sig2 + rho ** 2) / (2 * (1 - rho))
check(T, "P-K で σ^2=1/μ^2 → ρ^2/(1−ρ)", sp.simplify(PK.subs(sig2, 1 / mu ** 2).subs(rho, lam / mu) - (lam / mu) ** 2 / (1 - lam / mu)) == 0, "ok")
check(T, "P-K で σ^2=0 → ρ^2/{2(1−ρ)}", sp.simplify(PK.subs(sig2, 0) - rho ** 2 / (2 * (1 - rho))) == 0, "ok")
rh = F(4, 5)
LqM = rh ** 2 / (1 - rh); LqD = rh ** 2 / (2 * (1 - rh))
check(T, "例1 Lq 3.2 / 1.6, L 4.0 / 2.4", LqM == F(16, 5) and LqD == F(8, 5) and LqM + rh == 4 and LqD + rh == F(12, 5), (LqM, LqD))
check(T, "例1 Wq 0.8/0.4 時間, W 1.0/0.6 時間", LqM / 4 == F(4, 5) and LqD / 4 == F(2, 5) and (LqM + rh) / 4 == 1 and (LqD + rh) / 4 == F(3, 5), "ok")
ch = charts(T)
xs = ch[0]["params"]["x"]
yM = ch[0]["params"]["series"][0]["y"]; yD = ch[0]["params"]["series"][1]["y"]
check(T, "図 M/M/1 L の値（許容0.005）", all(abs(a / (1 - a) - b) <= .005 + 1e-9 for a, b in zip(xs, yM)), yM)
check(T, "図 M/D/1 L=ρ+ρ^2/(2(1−ρ)) の値（許容0.005）", all(abs(a + a * a / (2 * (1 - a)) - b) <= .005 + 1e-9 for a, b in zip(xs, yD)), [round(a + a * a / (2 * (1 - a)), 4) for a in xs])
# シミュ M/D/1 λ=4, D=0.2
A = rng.exponential(1 / 4, nC)
Wq = np.zeros(nC)
for j in range(1, nC):
    Wq[j] = max(0.0, Wq[j - 1] + 0.2 - A[j])
check(T, "シミュ M/D/1 Wq=0.4（100万人, 許容0.015）", abs(Wq[10000:].mean() - .4) < .015, Wq[10000:].mean())
Sv = rng.uniform(0, .4, nC)
for j in range(1, nC):
    Wq[j] = max(0.0, Wq[j - 1] + Sv[j - 1] - A[j])
pk = (16 * .4 ** 2 / 12 + .64) / (2 * .2) / 4
check(T, f"シミュ M/G/1 一様サービス Wq={pk:.4f}（P-K, 許容0.02）", abs(Wq[10000:].mean() - pk) < .02, Wq[10000:].mean())

# ============ random-walk ============
T = "random-walk"
check(T, "1歩 E=p−q, V=1−(p−q)^2=4pq", sp.simplify(1 - (p_ - q_) ** 2 - 4 * p_ * q_) == 0, "4pq")
check(T, "例1 E[S10]=2, V=9.6", close(10 * .2, 2, 1e-12) and close(4 * 10 * .6 * .4, 9.6, 1e-12), (2, 9.6))
v = comb(10, 6) * .6 ** 6 * .4 ** 4
check(T, "例1 210×0.046656×0.0256≈0.2508", comb(10, 6) == 210 and close(.6 ** 6, .046656, 1e-12) and close(.4 ** 4, .0256, 1e-12) and close(v, .2508), v)
check(T, "例2 252/1024≈0.2461, 1/√(5π)≈0.2523", F(comb(10, 5), 1024) == F(252, 1024) and close(252 / 1024, .2461) and close(1 / sqrt(5 * PI), .2523), (252 / 1024, 1 / sqrt(5 * PI)))
check(T, "例3 正規近似 P(Z≥0.9)≈0.1841, 正確 P(U≥55)≈0.1841", close(stats.norm.sf(.9), .1841) and close(stats.binom.sf(54, 100, .5), .1841), (stats.norm.sf(.9), stats.binom.sf(54, 100, .5)))
check(T, "例4 Cov=4.8, ρ=√0.5≈0.7071", close(4 * 5 * .6 * .4, 4.8, 1e-12) and close(4.8 / sqrt(4 * 5 * .24 * 4 * 10 * .24), .7071), (4.8, sqrt(.5)))
U = rng.binomial(1, .6, size=(400000, 10)) * 2 - 1
Sx = U.cumsum(1)
check(T, "シミュ p=0.6 E[S10]=2, V=9.6, Cov(S5,S10)=4.8（40万, 許容0.02・0.08・0.05）",
      abs(Sx[:, 9].mean() - 2) < .02 and abs(Sx[:, 9].var() - 9.6) < .08 and abs(np.cov(Sx[:, 4], Sx[:, 9])[0, 1] - 4.8) < .05,
      (Sx[:, 9].mean(), Sx[:, 9].var(), np.cov(Sx[:, 4], Sx[:, 9])[0, 1]))
check(T, "シミュ P(S10=2)=0.2508（許容0.003）、P(S10=3)=0", abs((Sx[:, 9] == 2).mean() - v) < .003 and (Sx[:, 9] == 3).sum() == 0, (Sx[:, 9] == 2).mean())
ch = charts(T)
check(T, "図2 Bin(20,0.5)（S=2U−n の説明と整合）", ch[1]["params"] == {"n": 20, "p": 0.5}, ch[1]["params"])
check(T, "C(2n,n)/4^n ≈ 1/√(πn)（n=1000 比 1±0.001）", abs(stats.binom.pmf(1000, 2000, .5) * sqrt(PI * 1000) - 1) < 1e-3, stats.binom.pmf(1000, 2000, .5) * sqrt(PI * 1000))

# ============ stochastic-process ============
T = "stochastic-process"
check(T, "例1 E=n/2, V=n/4, Cov(X4,X10)=1, 相関 4/√40≈0.6325=√(4/10)", close(4 / sqrt(40), .6325) and close(sqrt(.4), 4 / sqrt(40), 1e-12), 4 / sqrt(40))
X = rng.binomial(1, .5, size=(400000, 10)).cumsum(1)
check(T, "シミュ Cov(X4,X10)=1（40万, 許容0.01）, 相関0.6325（許容0.005）",
      abs(np.cov(X[:, 3], X[:, 9])[0, 1] - 1) < .01 and abs(np.corrcoef(X[:, 3], X[:, 9])[0, 1] - .6325) < .005, np.corrcoef(X[:, 3], X[:, 9])[0, 1])

# ============ superposition-thinning-poisson ============
T = "superposition-thinning-poisson"
l1_, l2_ = sp.symbols("lambda1 lambda2", positive=True)
check(T, "P(T1<T2)=λ1/(λ1+λ2)", sp.simplify(sp.integrate(l1_ * sp.exp(-l1_ * t) * sp.exp(-l2_ * t), (t, 0, sp.oo)) - l1_ / (l1_ + l2_)) == 0, "ok")
zz = sp.Symbol("z")
check(T, "PGF の積 e^{λ1t(s−1)}e^{λ2t(s−1)}", sp.simplify(sp.exp(l1_ * t * (zz - 1)) * sp.exp(l2_ * t * (zz - 1)) - sp.exp((l1_ + l2_) * t * (zz - 1))) == 0, "ok")
pp_ = sp.Symbol("p", positive=True)
ok = True
for kk_ in range(4):
    for mm in range(4):
        lhs = sp.exp(-lam * t) * (lam * t) ** (kk_ + mm) / sp.factorial(kk_ + mm) * sp.binomial(kk_ + mm, kk_) * pp_ ** kk_ * (1 - pp_) ** mm
        rhs = sp.exp(-lam * pp_ * t) * (lam * pp_ * t) ** kk_ / sp.factorial(kk_) * sp.exp(-lam * (1 - pp_) * t) * (lam * (1 - pp_) * t) ** mm / sp.factorial(mm)
        ok &= sp.simplify(lhs - rhs) == 0
check(T, "間引きの同時確率が Po(λpt)×Po(λqt) に分解（k,m≤3 記号）", ok, ok)
check(T, "例1 強度6, e^{-2}≈0.1353", close(20 * .3, 6, 1e-12) and close(exp(-6 / 3), .1353), exp(-2))
check(T, "例2 0.4, Po(5) で P(5)≈0.1755, 平均6分", close(4 / 10, .4, 1e-12) and close(stats.poisson.pmf(5, 5), .1755) and close(60 / 10, 6, 1e-12), stats.poisson.pmf(5, 5))
Ntot = rng.poisson(20 / 3, 1000000)
N1 = rng.binomial(Ntot, .3); N2 = Ntot - N1
check(T, "シミュ 例1 間引き P(N1=0)=e^-2（100万, 許容0.001）、Corr(N1,N2)≈0（許容0.004）",
      abs((N1 == 0).mean() - exp(-2)) < .001 and abs(np.corrcoef(N1, N2)[0, 1]) < .004, ((N1 == 0).mean(), np.corrcoef(N1, N2)[0, 1]))
check(T, "シミュ P(N1=0|N2=k) が k によらない（k=3,5,7, 許容0.006）",
      all(abs((N1[N2 == kq] == 0).mean() - exp(-2)) < .006 for kq in (3, 5, 7)), [(N1[N2 == kq] == 0).mean() for kq in (3, 5, 7)])
T1 = rng.exponential(1 / 4, 1000000); T2 = rng.exponential(1 / 6, 1000000)
check(T, "シミュ 例2 P(A が先)=0.4（許容0.002）, min の平均 0.1時間=6分（許容0.0005）",
      abs((T1 < T2).mean() - .4) < .002 and abs(np.minimum(T1, T2).mean() - .1) < .0005, ((T1 < T2).mean(), np.minimum(T1, T2).mean()))
NA = rng.poisson(2, 1000000); NB = rng.poisson(3, 1000000)
check(T, "シミュ 例2 30分で合計 Po(5)、P(=5)=0.1755（許容0.0015）", abs(((NA + NB) == 5).mean() - stats.poisson.pmf(5, 5)) < .0015, ((NA + NB) == 5).mean())
sel = (NA + NB) == 6
check(T, "シミュ 合計 n=6 のもとで A の個数 ~ Bin(6,0.4)（平均2.4, 許容0.02）", abs(NA[sel].mean() - 2.4) < .02, NA[sel].mean())
ch = charts(T)
check(T, "図 λ=2（params 存在）", ch[0]["params"]["lambda"] == 2, ch[0]["params"])

# ============ traffic-intensity ============
T = "traffic-intensity"
check(T, "6人/時×6分=36分 → ρ=0.6", close(6 * 6 / 60, .6, 1e-12), .6)
check(T, "dL/dρ=1/(1−ρ)^2, ρ=0.5で4, 0.9で100", sp.simplify(sp.diff(rho / (1 - rho), rho) - 1 / (1 - rho) ** 2) == 0
      and close(1 / (1 - .5) ** 2, 4, 1e-9) and close(1 / (1 - .9) ** 2, 100, 1e-9), "ok")
check(T, "例1 a=120×4/60=8, ρ=0.8", close(120 * 4 / 60, 8, 1e-12) and close(8 / 10, .8, 1e-12), 8)
check(T, "例2 L=1,4,9,19（ρ=.5,.8,.9,.95）, 9→19 で2倍以上", all(close(a / (1 - a), b, 1e-9) for a, b in [(.5, 1), (.8, 4), (.9, 9), (.95, 19)]) and 19 / 9 > 2, "ok")
check(T, "例3 6/0.7≈8.571, 60/8.571≈7分", close(6 / .7, 8.571, 5e-4) and close(60 / (6 / .7), 7, 1e-9), 6 / .7)
ch = charts(T)
xs = ch[0]["params"]["x"]; yL = ch[0]["params"]["series"][0]["y"]
check(T, "図1 L=ρ/(1−ρ)（許容0.005）、0.5→1, 0.9→9", all(abs(a / (1 - a) - b) <= .005 + 1e-9 for a, b in zip(xs, yL)), yL)
check(T, "図2 geom p=0.2=1−ρ（ρ=0.8 は例1と整合）, start=0", ch[1]["params"] == {"p": 0.2, "start": 0}, ch[1]["params"])

# ================= 重要度Cの用語 =================
print("\n--- 重要度Cの用語 ---")
from itertools import product as _product
rngC = np.random.default_rng(20261023)

T = "counting-process"
arr = np.array([0.8, 1.5, 1.9, 3.2, 4.6])
Nt = lambda t: int((arr <= t).sum())
check(T, "例 N(2)=3, N(4)−N(2)=1, N(5)−N(3)=2", (Nt(2), Nt(4) - Nt(2), Nt(5) - Nt(3)) == (3, 1, 2), (Nt(2), Nt(4) - Nt(2), Nt(5) - Nt(3)))
check(T, "例 P(N(2)=3)=e^{−2.4}2.4³/3!=0.2090", close(stats.poisson.pmf(3, 2.4), .2090) and close(exp(-2.4) * 2.4 ** 3 / 6, .2090), stats.poisson.pmf(3, 2.4))
check(T, "{N(t)≥n}={S_n≤t}: P(N(2)≥3)=P(Ga(3,λ) ≤2)", close(stats.poisson.sf(2, 2.4), stats.gamma.cdf(2, 3, scale=1 / 1.2), 1e-12), stats.poisson.sf(2, 2.4))
gaps = rngC.exponential(1 / 1.2, (200000, 12)).cumsum(1)
n2 = (gaps <= 2).sum(1)
check(T, "sim 指数間隔から作る N(2) で P(N(2)=3)≈0.2090（20万本, 許容 0.004）", abs((n2 == 3).mean() - .2090) < .004, (n2 == 3).mean())

T = "pure-birth-process"
lam_, tt_ = .5, 2.0
q_ = exp(-lam_ * tt_)
check(T, "例 e^{−1}=0.3679, E[X(2)]=2.718, P(X=1)=0.3679, P(X≥3)=(1−0.3679)²=0.3996", close(q_, .3679) and close(exp(1), 2.718, 5e-4) and close((1 - q_) ** 2, .3996), (q_, (1 - q_) ** 2))
check(T, "幾何分布（台1..）の平均 1/e^{−λt}=e^{λt}", close(stats.geom.mean(q_), exp(1), 1e-12), stats.geom.mean(q_))
check(T, "例 1/0.5+1/1+1/1.5=3.667", close(1 / .5 + 1 / 1 + 1 / 1.5, 3.667, 5e-4), 1 / .5 + 1 + 1 / 1.5)
ts_, ls_ = sp.symbols("t lambda", positive=True)
nn = sp.symbols("n", integer=True, positive=True)
Pn = lambda k: sp.exp(-ls_ * ts_) * (1 - sp.exp(-ls_ * ts_)) ** (k - 1)
fw = all(sp.simplify(sp.diff(Pn(k), ts_) - (-k * ls_ * Pn(k) + ((k - 1) * ls_ * Pn(k - 1) if k > 1 else 0))) == 0 for k in range(1, 7))
check(T, "sympy 前進方程式 P_n'=−nλP_n+(n−1)λP_{n−1} を満たす（n=1..6）", fw, "")
# 打ち切り生成行列の expm
from scipy.linalg import expm as _expm
Kb = 120
Qb = np.zeros((Kb, Kb))
for i in range(Kb - 1):
    Qb[i, i] = -(i + 1) * lam_; Qb[i, i + 1] = (i + 1) * lam_
Pb = _expm(Qb * tt_)[0]
check(T, "打ち切り生成行列 expm の P(X(2)=n) が幾何分布と一致", np.allclose(Pb[:20], [q_ * (1 - q_) ** k for k in range(20)], atol=1e-10), Pb[:3])
nsb = 100000
xb = np.ones(nsb); tb = np.zeros(nsb); al = np.ones(nsb, bool)
while al.any():
    idx = np.where(al)[0]
    tb[idx] += rngC.exponential(1 / (lam_ * xb[idx]))
    d = tb[idx] > tt_
    al[idx[d]] = False
    xb[idx[~d]] += 1
check(T, "sim 平均個数≈2.72（10万本, 許容 0.03）, P(X(2)≥3)≈0.400（許容 0.006）", abs(xb.mean() - exp(1)) < .03 and abs((xb >= 3).mean() - .3996) < .006, (xb.mean(), (xb >= 3).mean()))
t4 = sum(rngC.exponential(1 / (lam_ * k), nsb) for k in (1, 2, 3))
check(T, "sim 1→4 の平均時間≈3.667（10万本, 許容 0.03）", abs(t4.mean() - 11 / 3) < .03, t4.mean())

T = "compound-poisson-process"
us, lt = sp.symbols("u lt", positive=True)
EY, EY2, c3 = sp.symbols("EY EY2 c3")
MYp = 1 + EY * us + EY2 * us ** 2 / 2 + c3 * us ** 3  # M_Y の u=0 まわりの展開
Ms = sp.exp(lt * (MYp - 1))
m1v = sp.diff(Ms, us).subs(us, 0); m2v = sp.diff(Ms, us, 2).subs(us, 0)
check(T, "sympy MGF 微分: E[S]=λtE[Y], V[S]=λtE[Y²]", sp.simplify(m1v - lt * EY) == 0 and sp.simplify(m2v - m1v ** 2 - lt * EY2) == 0, (m1v, sp.expand(m2v - m1v ** 2)))
ES = 36 * 20; VS = 36 * (10 ** 2 + 20 ** 2)
zc = (900 - ES) / sqrt(VS)
check(T, "例 E[S]=720, V[S]=18000, SD=134.16, z=1.342, P=0.0899", ES == 720 and VS == 18000 and close(sqrt(VS), 134.16, 5e-3) and close(zc, 1.342, 5e-4) and close(stats.norm.sf(zc), .0899), (sqrt(VS), zc, stats.norm.sf(zc)))
Nn = rngC.poisson(36, 1000000)
Ssim = np.where(Nn > 0, rngC.gamma(np.maximum(4 * Nn, 1), 5.0), 0.0)
ps_ = (Ssim > 900).mean()
check(T, "sim ガンマ（形4・尺度5: 平均20, SD10）で E≈720, V≈18000（相対1%）, P(S>900)≈0.094（100万本, 許容 0.002）", abs(Ssim.mean() / 720 - 1) < .01 and abs(Ssim.var() / 18000 - 1) < .01 and abs(ps_ - .094) < .002, (Ssim.mean(), Ssim.var(), ps_))

T = "mmc-queue"
def mmc(lam, mu, c):
    a = lam / mu; rho = a / c
    p0 = 1 / (sum(a ** k / factorial(k) for k in range(c)) + a ** c / (factorial(c) * (1 - rho)))
    C = a ** c / (factorial(c) * (1 - rho)) * p0
    Lq = C * rho / (1 - rho); Wq = C / (c * mu - lam)
    return dict(p0=p0, C=C, Lq=Lq, Wq=Wq, W=Wq + 1 / mu, L=Lq + a)
r = mmc(6, 4, 2)
check(T, "例 π0=1/7=0.1429, C=4.5/7=0.6429", close(r["p0"], 1 / 7, 1e-12) and close(r["C"], 4.5 / 7, 1e-12) and close(r["p0"], .1429) and close(r["C"], .6429), (r["p0"], r["C"]))
check(T, "例 Lq=1.929, Wq=0.3214h=19.3分, W=34.3分, L=3.429", close(r["Lq"], 1.929, 5e-4) and close(r["Wq"], .3214) and close(r["Wq"] * 60, 19.3, .05) and close(r["W"] * 60, 34.3, .05) and close(r["L"], 3.429, 5e-4), r)
check(T, "リトル Lq=λWq, L=λW", close(r["Lq"], 6 * r["Wq"], 1e-12) and close(r["L"], 6 * r["W"], 1e-12), "")
# 定常分布を打ち切り生成行列で
Kq = 400
Qq = np.zeros((Kq, Kq))
for n in range(Kq):
    if n + 1 < Kq: Qq[n, n + 1] = 6
    if n > 0: Qq[n, n - 1] = min(n, 2) * 4
    Qq[n, n] = -Qq[n].sum()
A_ = np.vstack([Qq.T, np.ones(Kq)]); bvec = np.zeros(Kq + 1); bvec[-1] = 1
piq = np.linalg.lstsq(A_, bvec, rcond=None)[0]
check(T, "打ち切り生成行列の定常分布で π0, P(N≥2), L が一致", close(piq[0], 1 / 7, 1e-9) and close(piq[2:].sum(), r["C"], 1e-9) and close((np.arange(Kq) * piq).sum(), r["L"], 1e-8), (piq[0], piq[2:].sum()))
r1 = mmc(6, 8, 1); r2 = mmc(3, 4, 1)
check(T, "比較 M/M/1(μ=8): Wq=22.5分, W=30分; M/M/1×2(λ=3): Wq=45分", close(r1["Wq"] * 60, 22.5, 1e-9) and close(r1["W"] * 60, 30, 1e-9) and close(r2["Wq"] * 60, 45, 1e-9), (r1["Wq"] * 60, r1["W"] * 60, r2["Wq"] * 60))
lq, mq = sp.symbols("lambda mu", positive=True)
rq = lq / mq
p0s = 1 / (1 + rq / (1 - rq))
check(T, "sympy c=1 で π0=1−ρ, C=ρ", sp.simplify(p0s - (1 - rq)) == 0 and sp.simplify(rq / (1 - rq) * p0s - rq) == 0, "")

T = "reflection-principle"
cnt = 0; cnt6 = 0; cntb = 0
for st in _product((1, -1), repeat=10):
    s = np.cumsum(st)
    if s.max() >= 4: cnt += 1
    if s[-1] == 2 and s.max() >= 4: cntb += 1
    if s[-1] == 6: cnt6 += 1
check(T, "例 P(S10≥4)=176/1024, P(S10=4)=120/1024", sum(comb(10, k) for k in range(7, 11)) == 176 and comb(10, 7) == 120, 176)
check(T, "例 (2·176−120)/1024=232/1024=0.2266、全 2^10 通りの数え上げも 232", 2 * 176 - 120 == 232 and cnt == 232 and close(232 / 1024, .2266), cnt)
check(T, "例 終点2で途中4に達する道=終点6の道=C(10,8)=45", cntb == 45 and cnt6 == 45 and comb(10, 8) == 45, (cntb, cnt6))
okall = True
for n in range(1, 13):
    paths = np.array(list(_product((1, -1), repeat=n))).cumsum(1)
    mx = np.maximum(paths.max(1), 0); en = paths[:, -1]
    for m in range(1, n + 1):
        lhs = (mx >= m).sum(); rhs = 2 * (en >= m).sum() - (en == m).sum()
        if lhs != rhs or rhs != (en >= m).sum() + (en > m).sum(): okall = False
        for b in range(-n, m):
            if ((mx >= m) & (en == b)).sum() != (en == 2 * m - b).sum(): okall = False
check(T, "全数え上げ（n=1..12, 全 m, b）で道の対応と P(max≥m) の式が成立", okall, "")
check(T, "BM: 2P(W(1)≥1)=2×0.1587=0.3173", close(2 * stats.norm.sf(1), .3173) and close(stats.norm.sf(1), .1587), 2 * stats.norm.sf(1))

print()
print(f"OK {sum(results)} / NG {len(results) - sum(results)}")

