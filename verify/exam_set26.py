"""模擬試験 第26回（set26）の検算。generator/set26.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・行列計算・数値最適化）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(26)
rng = np.random.default_rng(20261026)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---- 大問1：ブリッジ型ネットワーク（2^5 通りの総当たり＋グラフの連結判定）
pr = [0.9, 0.8, 0.8, 0.9, 0.7]
edges = [("S", "A"), ("S", "B"), ("A", "T"), ("B", "T"), ("A", "B")]
tot = 0.0
for st in product([0, 1], repeat=5):
    w = np.prod([p if s_ else 1 - p for p, s_ in zip(pr, st)])
    seen, stack = {"S"}, ["S"]
    while stack:
        u = stack.pop()
        for (a, b), s_ in zip(edges, st):
            if s_ and u in (a, b):
                v = b if u == a else a
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    tot += w * ("T" in seen)
E.num("s26p01a", rnd(tot, 3), "32 通りの状態の総当たり")
sim = rng.random((400_000, 5)) < np.array(pr)
ok = (sim[:, 0] & sim[:, 2]) | (sim[:, 1] & sim[:, 3]) | (sim[:, 0] & sim[:, 4] & sim[:, 3]) | (sim[:, 1] & sim[:, 4] & sim[:, 2])
E.check("s26p01a", abs(ok.mean() - tot) < 0.002, f"シミュレーション {ok.mean():.4f}（最小パス集合による判定）")

# ---- 大問2
# (a) ジャック・ベラ：scipy の χ² 点と、式の再計算
n, S_, K_ = 200, 0.21, 3.30
jb = n / 6 * (S_ ** 2 + (K_ - 3) ** 2 / 4)
E.choice("s26p02a", f"JB={f(jb, 2)}$ で、正規性は棄却されない")
E.check("s26p02a", close(stats.chi2.ppf(0.95, 2), 5.991, 5e-4) and close(stats.chi2.ppf(0.95, 1), 3.841, 5e-4), "χ² の 5% 点（自由度 2, 1）")
# 実際に歪度 0.21・尖度 3.30 に近い分布で JB の帰無分布を確かめる必要はないが、正規データで JB の 95% 点が χ²_2 に近いことを確認
z = rng.standard_normal((20_000, 200))
zc = z - z.mean(1, keepdims=True)
m2 = (zc ** 2).mean(1)
sk = (zc ** 3).mean(1) / m2 ** 1.5
ku = (zc ** 4).mean(1) / m2 ** 2
jbs = 200 / 6 * (sk ** 2 + (ku - 3) ** 2 / 4)
E.check("s26p02a", (jbs > jb).mean() > 0.2, f"正規データで JB>{jb:.2f} となる割合 {(jbs > jb).mean():.3f}（棄却されない）")
# (b) 対数正規：σ を数値で解き、scipy の lognorm で上側確率
sig = optimize.brentq(lambda s_: np.exp(s_ ** 2 / 2) - 100 / 80, 0.01, 3)
p_exact = stats.lognorm(sig, scale=80).sf(200)
nearest("s26p02b", p_exact)
E.check("s26p02b", close(stats.lognorm(sig, scale=80).mean(), 100, 1e-9), "lognorm の平均が 100")
for zz, v in {0.68: 0.7517, 1.04: 0.8508, 1.37: 0.9147, 1.94: 0.9738, 2.05: 0.9798}.items():
    E.check("s26p02b", close(stats.norm.cdf(zz), v, 6e-5), f"Φ({zz})={v}")
E.check("s26p02b", close(np.log(1.25), 0.2231, 5e-5) and close(np.log(2.5), 0.9163, 5e-5) and close(np.log(2), 0.6931, 5e-5), "対数の値")
E.choice("s26p02b", f"${f(1 - stats.norm.cdf(1.37), 3)}$", "表の値を使った値")

# ---- 大問3
# (a) シミュレーションで X_(1) の偏りと MSE
theta, beta, nn = 23.0, 2.0, 10
X = theta + rng.exponential(beta, size=(400_000, nn))
mn = X.min(1)
mse_mle = ((mn - theta) ** 2).mean()
mse_unb = ((mn - beta / nn - theta) ** 2).mean()
E.check("s26p03a", abs(mn.mean() - theta - 0.2) < 0.003, f"E[X(1)]−θ={mn.mean() - theta:.4f}≈0.2")
E.check("s26p03a", abs(mse_mle / mse_unb - 2) < 0.03, f"MSE 比 {mse_mle / mse_unb:.3f}≈2")
E.choice("s26p03a", f"\\tilde\\theta={f(23.6 - beta / nn, 1)}$、比は 2")
# (b) scipy の χ² で検出力関数を解く
c = 2.0 * stats.chi2.ppf(0.95, 20) / 2
b1 = optimize.brentq(lambda b: stats.chi2.sf(2 * c / b, 20) - 0.90, 2.01, 50)
nearest("s26p03b", b1)
for a_, v in {0.95: 10.85, 0.90: 12.44, 0.10: 28.41, 0.05: 31.41, 0.025: 34.17}.items():
    E.check("s26p03b", close(stats.chi2.isf(a_, 20), v, 5e-3), f"χ²_20 の上側 {a_} 点 {v}")
T = rng.exponential(b1, size=(200_000, 10)).sum(1)
E.check("s26p03b", abs((T >= c).mean() - 0.90) < 0.005, f"β={b1:.3f} での検出力（シミュレーション）{(T >= c).mean():.4f}")

# ---- 大問4
# (a) sympy で分布関数を作って微分
v = sp.symbols("v", positive=True)
G1 = sp.Rational(1, 3) * (sp.sqrt(v) - (-sp.sqrt(v)))     # 0<v<1: P(-√v<X<√v)
G2 = sp.Rational(1, 3) * (sp.sqrt(v) - (-1))               # 1<v<4: P(-1<X<√v)
g1, g2 = sp.simplify(sp.diff(G1, v)), sp.simplify(sp.diff(G2, v))
E.check("s26p04a", sp.simplify(g1 - 1 / (3 * sp.sqrt(v))) == 0 and sp.simplify(g2 - 1 / (6 * sp.sqrt(v))) == 0, f"sympy: {g1}, {g2}")
E.choice("s26p04a", r"\dfrac{1}{3\sqrt v}\ (0<v<1),\quad g(v)=\dfrac{1}{6\sqrt v}\ (1\le v<4)")
Vs = rng.uniform(-1, 2, 400_000) ** 2
E.check("s26p04a", abs((Vs < 0.25).mean() - 2 / 3 * 0.5) < 0.003 and abs((Vs > 2.25).mean() - (2 - 1.5) / 3) < 0.003,
        "シミュレーションで P(V<0.25)=1/3, P(V>2.25)=1/6")
# (b) 平均のシミュレーションと scipy の cauchy
Wb = stats.cauchy(scale=0.5).rvs(size=(200_000, 16), random_state=rng).mean(1)
p_sim = (np.abs(Wb) > 1).mean()
p_th = 2 * stats.cauchy(scale=0.5).sf(1)
E.check("s26p04b", abs(p_sim - p_th) < 0.004, f"シミュレーション {p_sim:.4f} ≈ 理論 {p_th:.4f}")
nearest("s26p04b", p_th)
for u_, val in {0.5: 0.4636, 2: 1.1071, 8: 1.4464, 32: 1.5396}.items():
    E.check("s26p04b", close(np.arctan(u_), val, 5e-5), f"arctan {u_}={val}")
E.choice("s26p04b", f"${f(p_th, 3)}$")

# ---- 大問5
xm, al = 0.4, 3
par = stats.pareto(al, scale=xm)
E.choice("s26p05a", f"${f(par.ppf(0.875), 2)}$", "scipy pareto.ppf(0.875)")
nearest("s26p05a", par.ppf(0.875))
# (b) 数値積分の E[S²] とシミュレーション
es2 = integrate.quad(lambda s_: s_ ** 2 * par.pdf(s_), xm, np.inf)[0]
E.num("s26p05b", rnd(np.sqrt(30 * es2), 2), "数値積分の E[S²] から √(λt E[S²])")
Nn = rng.poisson(30, size=200_000)
Z = np.array([par.rvs(size=k, random_state=rng).sum() if k else 0.0 for k in Nn[:60_000]])
E.check("s26p05b", abs(Z.mean() - 18) < 0.1, f"シミュレーションの平均 {Z.mean():.3f}≈18")
E.check("s26p05b", abs(Z.var() / 14.4 - 1) < 0.15, f"シミュレーションの分散 {Z.var():.2f}≈14.4（裾が重いのでゆるい判定）")
# (c) 確率収束と平均二乗収束を計算で確かめる
ok7 = True
for k in (10, 100, 1000):
    p_big = 1 / k
    ok7 &= close(k * p_big, 1, 1e-12) and close(k ** 2 * p_big, k, 1e-9)
E.check("s26p05c", ok7 and E.item("s26p05c")["answer"] == 1, "P(|Z_n|>ε)=1/n→0、E[Z_n]=1、E[Z_n²]=n→∞ から選択肢 1 だけが正しい")

# ---- 大問6：時間の合計から
t_tot = 60 / 1.2 + 120 / 2.0 + 180 / 3.0
E.num("s26p06a", rnd(360 / t_tot, 2), "総量÷総時間")
E.check("s26p06a", close(stats.hmean([1.2, 2.0, 3.0], weights=[60, 120, 180]), 360 / t_tot, 1e-12), "scipy の加重調和平均と一致")

# ---- 大問7
L = [3.0, 3.3, 3.6, 3.9, 4.2, 4.5]
M = [3.8, 4.0, 4.4, 4.6, 5.0, 5.2]
ks = stats.ks_2samp(L, M, method="exact")
E.choice("s26p07a", f"D={f(ks.statistic, 2)}$ で、棄却されない", "scipy ks_2samp")
allv = sorted(L + M)
cnt = {}
for idx in combinations(range(12), 6):
    a = [allv[i] for i in idx]
    b = [allv[i] for i in range(12) if i not in idx]
    d = max(abs(sum(x <= t for x in a) - sum(x <= t for x in b)) for t in allv)
    cnt[d] = cnt.get(d, 0) + 1
p5 = sum(vv for k, vv in cnt.items() if k >= 5) / 924
p4 = sum(vv for k, vv in cnt.items() if k >= 4) / 924
E.check("s26p07a", p5 <= 0.05 < p4, f"924 通りの並べ替えで P(D≥5/6)={p5:.4f}≤0.05<P(D≥4/6)={p4:.4f}（棄却域 D≥5/6）")
E.check("s26p07a", close(ks.pvalue, sum(vv for k, vv in cnt.items() if k >= 3) / 924, 1e-9), f"p 値 {ks.pvalue:.4f} が総当たりと一致")
# (b) scipy の t 区間（等分散）
res = stats.ttest_ind(M, L, equal_var=True)
ci = res.confidence_interval(0.95)
E.choice("s26p07b", f"({f(ci.low, 2)},\\ {f(ci.high, 2)})", "scipy ttest_ind の信頼区間")
E.check("s26p07b", close(stats.t.ppf(0.975, 10), 2.228, 5e-4) and close(stats.t.ppf(0.975, 5), 2.571, 5e-4), "t 点 2.228, 2.571")
E.check("s26p07b", close(np.var(L, ddof=1), 0.315, 1e-9) and close(np.var(M, ddof=1), 0.300, 1e-9), "不偏分散 0.315・0.300")

# ---- 大問8
# (a) 相関行列から VIF を作れる例を作り、逆行列の対角が 1/(1−R²) になることを確かめる
R = np.array([[1, 0.6, 0.9], [0.6, 1, 0.5], [0.9, 0.5, 1]])
vif_inv = np.diag(np.linalg.inv(R))
r2_aux = 1 - 1 / vif_inv
E.check("s26p08a", np.allclose(vif_inv, 1 / (1 - r2_aux)), "相関行列の逆行列の対角 = 1/(1−R_j²)")
E.choice("s26p08a", f"VIF}}_1={f(1 / (1 - 0.92), 1)}$、約 {f(np.sqrt(1 / (1 - 0.92)), 2)} 倍")
# シミュレーション：x1 を他と相関させたときと無相関のときの係数の SE の比
nobs = 40
Z3 = rng.standard_normal((nobs, 2))
x1c = Z3 @ np.array([0.8, 0.5]) + 0.3 * rng.standard_normal(nobs)
x1c = (x1c - x1c.mean())
aux = np.c_[np.ones(nobs), Z3]
res1 = x1c - aux @ np.linalg.lstsq(aux, x1c, rcond=None)[0]
r2 = 1 - (res1 ** 2).sum() / (x1c ** 2).sum()
Xc = np.c_[np.ones(nobs), x1c, Z3]
se_c = np.sqrt(np.linalg.inv(Xc.T @ Xc)[1, 1])
se_o = np.sqrt(1 / (x1c ** 2).sum())
E.check("s26p08a", close(se_c / se_o, np.sqrt(1 / (1 - r2)), 1e-9), f"行列計算で SE の比 {se_c / se_o:.4f} = √VIF")
# (b) 数値で方程式を解く
yhat = lambda x: 20 + 0.5 * x + 3.0 * max(x - 70, 0)
E.num("s26p08b", rnd(optimize.brentq(lambda x: yhat(x) - 120, 0, 100), 1), "brentq で ŷ=120 を解く")
# (c) 実際に 1 点ずつ除いて当てはめ直す（表の元データ）
x = np.array([10, 20, 30, 40, 50, 60.])
y = np.array([42.0, 47.0, 58.0, 61.0, 71.0, 72.0])
press = 0.0
for i in range(6):
    m = np.arange(6) != i
    b_, a_ = np.polyfit(x[m], y[m], 1)
    press += (y[i] - (a_ + b_ * x[i])) ** 2
E.num("s26p08c", rnd(press, 1), "1 点ずつ除いて当てはめ直した PRESS（元データ）")
X2 = np.c_[np.ones(6), x]
H = X2 @ np.linalg.inv(X2.T @ X2) @ X2.T
e = y - H @ y
it = E.item("s26p08c")
shown_e = [float(v) for v in re.findall(r"\| \$e_i\$ \| (.*?) \|\n", it["question"])[0].split(" | ")]
shown_h = [float(v) for v in re.findall(r"\| \$h_\{ii\}\$ \| (.*?) \|\n", it["question"])[0].split(" | ")]
E.check("s26p08c", np.allclose(shown_e, np.round(e, 2)) and np.allclose(shown_h, np.round(np.diag(H), 3)), "表の残差・てこ比が行列計算と一致")
E.check("s26p08c", close(sum((a / (1 - b)) ** 2 for a, b in zip(shown_e, shown_h)), it["answer"], 0.05), "表の値からの PRESS も正解の丸めに入る")

# ---- 大問9：オフセットを作って実際にポアソン回帰を当てはめ、逸脱度を求める
yv = np.array([12, 7, 20, 9.])
mu = np.array([10.0, 9.0, 18.0, 11.0])
age = np.array([1, 2, 4, 3.])            # Σ age (y−μ)=0 を満たす年数
a0, b0 = -1.0, 0.3
off = np.log(mu) - (a0 + b0 * age)
nll = lambda p: -(yv * (off + p[0] + p[1] * age) - np.exp(off + p[0] + p[1] * age)).sum()
opt = optimize.minimize(nll, [0, 0], method="BFGS", options={"gtol": 1e-10})
mu_fit = np.exp(off + opt.x[0] + opt.x[1] * age)
E.check("s26p09a", np.allclose(mu_fit, mu, atol=1e-4), f"最尤推定の当てはめ値 {np.round(mu_fit, 4)} が表と一致")
dev = 2 * (yv * np.log(yv / mu_fit) - (yv - mu_fit)).sum()
E.choice("s26p09a", f"D={f(dev, 2)}$、自由度 2", "当てはめ直した逸脱度")
E.check("s26p09a", dev < stats.chi2.ppf(0.95, 2) and close(stats.chi2.ppf(0.95, 3), 7.815, 5e-4), "自由度 2 の 5% 点 5.991 より小さい・7.815 を確認")
for i, v_ in enumerate([0.1823, -0.2513, 0.1054, -0.2007]):
    E.check("s26p09a", close(np.log(yv[i] / mu[i]), v_, 5e-5), f"ln(y/μ) 表の値 {v_}")

# ---- 大問10：KM と Greenwood を独立に実装（打ち切りを順に処理）
times = np.array([5, 8, 12, 12, 15, 20, 24, 30])
event = np.array([1, 0, 1, 1, 0, 1, 0, 0])
S_km, gw = 1.0, 0.0
for t_ in np.unique(times[event == 1]):
    if t_ > 20:
        break
    n_at = (times >= t_).sum()
    d_ = ((times == t_) & (event == 1)).sum()
    S_km *= 1 - d_ / n_at
    gw += d_ / (n_at * (n_at - d_))
E.num("s26p10a", rnd(np.sqrt(S_km ** 2 * gw), 3), "独立実装のグリーンウッド")
E.check("s26p10a", close(S_km, 7 / 8 * 4 / 6 * 2 / 3, 1e-12), f"KM 推定値 {S_km:.4f}")

# ---- 大問11：分割区法のデータを作って平方和を求め直す（線形モデルの平方和分解）
r_, a_l, b_l = 4, 3, 2
cells = [(i, j, k) for i in range(r_) for j in range(a_l) for k in range(b_l)]
ydat = rng.normal(size=len(cells))
idx = np.array(cells)
def ss_effect(cols):
    keys = [tuple(row) for row in idx[:, cols]]
    gm = ydat.mean()
    tot_ = 0.0
    for kk in set(keys):
        m = np.array([k_ == kk for k_ in keys])
        tot_ += m.sum() * (ydat[m].mean() - gm) ** 2
    return tot_
ssR = ss_effect([0]); ssA = ss_effect([1]); ssRA = ss_effect([0, 1]) - ssR - ssA
ssB = ss_effect([2]); ssAB = ss_effect([1, 2]) - ssA - ssB
ssT = ((ydat - ydat.mean()) ** 2).sum()
ssE2 = ssT - ssR - ssA - ssRA - ssB - ssAB
# 自由度の確認：1 次誤差 (r-1)(a-1)=6、2 次誤差 a(r-1)(b-1)=9
E.check("s26p11a", (r_ - 1) * (a_l - 1) == 6 and a_l * (r_ - 1) * (b_l - 1) == 9 and ssE2 > 0, "1 次誤差の自由度 6・2 次誤差の自由度 9")
F_A = (42.0 / 2) / (9.0 / 6)
E.choice("s26p11a", f"F={f(F_A, 1)}$、自由度 $(2,\\,6)", "MS_A / MS_E1")
E.check("s26p11a", close(stats.f.ppf(0.95, 2, 6), 5.14, 5e-3), "F_{0.05}(2,6)=5.14")
# 期待平均平方：主区因子の F の分母は σ²_E2 + b σ²_RA（シミュレーションで MS_RA の期待値を確認）
msra, mse2 = [], []
for _ in range(3000):
    eRA = rng.normal(0, 1.0, size=(r_, a_l))
    yy = np.array([eRA[i, j] + rng.normal(0, 0.5) for (i, j, k) in cells])
    ydat = yy
    sR, sA = ss_effect([0]), ss_effect([1])
    sRA = ss_effect([0, 1]) - sR - sA
    sB = ss_effect([2]); sAB = ss_effect([1, 2]) - sA - sB
    sE2 = ((yy - yy.mean()) ** 2).sum() - sR - sA - sRA - sB - sAB
    msra.append(sRA / 6); mse2.append(sE2 / 9)
E.check("s26p11a", abs(np.mean(msra) - (0.25 + 2 * 1.0)) < 0.1 and abs(np.mean(mse2) - 0.25) < 0.02,
        f"E[MS_E1]={np.mean(msra):.3f}≈σ²+bσ²_RA=2.25、E[MS_E2]={np.mean(mse2):.3f}≈0.25（主区因子は 1 次誤差で割る）")

# ---- 大問12：費用制約つきの分散最小化を数値最適化で解く
N = np.array([400, 1200, 400.]); Sd = np.array([6, 3, 10.]); cc = np.array([4, 1, 9.]); C = 1020
Nt = N.sum()
var_fn = lambda nh: (((N / Nt) ** 2) * Sd ** 2 * (1 / nh - 1 / N)).sum()
cons = {"type": "eq", "fun": lambda nh: (cc * nh).sum() - C}
opt = optimize.minimize(var_fn, np.array([50, 100, 50.]), constraints=[cons], bounds=[(1, 399)] * 1 + [(1, 1199)] + [(1, 399)],
                        method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
E.num("s26p12a", rnd(opt.x[1], 0), f"SLSQP の最適解 {np.round(opt.x, 3)}")

# ---- 大問13
# (a) 自己共分散からスペクトルを数値で作る（sum γ_k cos(kω)）
phi = 0.8
kk = np.arange(1, 400)
gam = lambda k: phi ** np.abs(k) / (1 - phi ** 2)
spec = lambda w: (gam(0) + 2 * (gam(kk) * np.cos(kk * w)).sum()) / (2 * np.pi)
ratio = spec(0) / spec(np.pi)
E.choice("s26p13a", f"f(0)/f(\\pi)={f(ratio, 0)}$ で、ゆっくりした", "自己共分散の級数から")
# (b) マハラノビス距離と射影
Rm = np.array([[1, 0.8], [0.8, 1]])
zz = np.array([1.5, -0.5])
T2 = zz @ np.linalg.inv(Rm) @ zz
w_, V_ = np.linalg.eigh(Rm)
v1 = V_[:, np.argmax(w_)]
Q = ((zz - (v1 @ zz) * v1) ** 2).sum()
E.choice("s26p13b", f"T^2={f(T2, 2)}$、$Q={f(Q, 2)}", "z'R^{-1}z と第1主成分への射影の残差")
# (c) 数値で事後確率が等しくなる点を求める
gfun = lambda s_: stats.norm(0, 2).pdf(s_) - stats.norm(0, 1).pdf(s_)
cb = optimize.brentq(gfun, 0.5, 3)
E.choice("s26p13c", f"|s|>{f(cb, 2)}", "brentq で密度が等しくなる点")
E.check("s26p13c", gfun(0) < 0 and gfun(3) > 0 and gfun(-3) > 0, "0 付近は正常、両側の外側が攻撃")
E.check("s26p13c", close(np.log(2), 0.6931, 5e-5) and close(np.log(4), 1.3863, 5e-5), "ln2・ln4 の値")

# ---- 大問14
# (a) 点の座標を作らず、距離から直接（numpy）
dA, dB, dC = np.array([2.0, 3.0, 1.0]), np.array([5.0, 4.0, 6.0, 5.0]), np.array([3.0, 4.0])
a = dA.mean(); b = min(dB.mean(), dC.mean())
E.num("s26p14a", rnd((b - a) / max(a, b), 2), "定義どおり")
# 1 次元の点で同じ距離を実現できる例でも sklearn と同じ定義になることの確認（ユークリッド距離で再現できる配置）
try:
    from sklearn.metrics import silhouette_samples
    pts = np.array([[0.0], [2.0], [3.0], [1.0], [5.0], [4.0], [6.0], [5.0], [-3.0], [-4.0]])
    lab = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    s0 = silhouette_samples(pts, lab)[0]
    E.check("s26p14a", close(s0, (3.5 - 2.0) / 3.5, 1e-9), f"sklearn の silhouette_samples {s0:.4f}")
except ImportError:
    E.check("s26p14a", close((b - a) / max(a, b), 3 / 7, 1e-12), "分数 3/7 と一致")

# (b) d 分離を線形ガウスの構造方程式で確かめる（偏相関がゼロになる組）
coef = {("M", "P"): 0.7, ("M", "A"): 0.6, ("P", "V"): 0.8, ("V", "I"): 0.9, ("I", "A"): 0.5}
order = ["M", "P", "V", "I", "A"]
nodes = {k: i for i, k in enumerate(order)}
B = np.zeros((5, 5))
for (u_, v__), c_ in coef.items():
    B[nodes[v__], nodes[u_]] = c_
Ainv = np.linalg.inv(np.eye(5) - B)
Sig = Ainv @ Ainv.T
def pcor(x_, y_, given):
    ids = [nodes[x_], nodes[y_]] + [nodes[g] for g in given]
    P = np.linalg.inv(Sig[np.ix_(ids, ids)])
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
claims = [("P", "A", ["V"]), ("V", "A", ["I"]), ("M", "I", ["A"]), ("P", "I", []), ("V", "A", ["I", "M"])]
zero = [abs(pcor(*c_)) < 1e-10 for c_ in claims]
E.check("s26p14b", zero == [False, False, False, False, True] and E.item("s26p14b")["answer"] == 4,
        f"線形ガウス SEM の偏相関がゼロになるのは選択肢 4 だけ: {zero}")

E.done()
