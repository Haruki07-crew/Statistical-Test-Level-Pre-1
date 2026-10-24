"""模擬試験 第25回（set25）の検算。generator/set25.py とは別の方法（scipy の分布関数・数値積分・数値最適化・
シミュレーション・全列挙・行列計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(25)
rng = np.random.default_rng(20261025)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    """各選択肢に含まれる数値のリスト（$ の中の最初の数値）。"""
    out = []
    for c in E.item(iid)["choices"]:
        out.append([float(x) for x in re.findall(r"-?\d+\.?\d*", str(c))])
    return out


def nearest(iid, value, pos=0):
    """選択肢の pos 番目の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    v = [n[pos] for n in nums(iid)]
    k = int(np.argmin([abs(x - value) for x in v]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---- 大問1 限界希釈
E.check("s25p01a", close(math.log(58 / 96), -0.5039, 5e-5), "ln(58/96) = -0.5039")
lam = -math.log(58 / 96)
pc = stats.poisson.pmf(1, lam) / stats.poisson.sf(0, lam)
E.num("s25p01a", rnd(pc, 3), "scipy の poisson で P(X=1)/P(X≥1)")
sim = rng.poisson(lam, 2_000_000)
E.check("s25p01a", close((sim == 1).sum() / (sim >= 1).sum(), pc, 0.002), "シミュレーションでも一致")
# (b) ゼロ切断ポアソンの対数尤度を数値最大化
cnt = {1: 28, 2: 16, 3: 5, 4: 1}
xs = np.repeat(list(cnt), list(cnt.values()))


def nll(l):
    return -np.sum(stats.poisson.logpmf(xs, l) - np.log1p(-np.exp(-l)))


res = optimize.minimize_scalar(nll, bounds=(0.1, 5), method="bounded")
nearest("s25p01b", res.x)
for l, g in {0.8: 1.4528, 0.9: 1.5166, 1.0: 1.5820, 1.1: 1.6489, 1.2: 1.7172}.items():
    E.check("s25p01b", close(l / (1 - math.exp(-l)), g, 5e-5), f"表の λ/(1-e^-λ) λ={l}")

# ---- 大問2 天秤
# (a) 正規近似（scipy）とシミュレーション
E.num("s25p02a", rnd(2 * stats.norm.sf(0.1, 0, math.sqrt(48 * 0.1 ** 2 / 12)), 3), "scipy norm で 2P(S>0.1)")
S = rng.uniform(-0.05, 0.05, size=(200_000, 48)).sum(axis=1)
E.check("s25p02a", close((np.abs(S) > 0.1).mean(), 0.617, 0.005), f"シミュレーション {(np.abs(S) > 0.1).mean():.4f}")
for z, v in {0.5: 0.6915, 1.0: 0.8413, 2.5: 0.9938}.items():
    E.check("s25p02a", close(stats.norm.cdf(z), v, 5e-5), f"Φ({z})")
# (b) 混合分布の分散をシミュレーションと数値積分で
k = rng.random(2_000_000) < 0.6
eps = np.where(k, rng.normal(30, 20, k.size), rng.normal(-20, 40, k.size))
dens = lambda x: 0.6 * stats.norm.pdf(x, 30, 20) + 0.4 * stats.norm.pdf(x, -20, 40)
m1 = integrate.quad(lambda x: x * dens(x), -np.inf, np.inf)[0]
m2 = integrate.quad(lambda x: x * x * dens(x), -np.inf, np.inf)[0]
E.choice("s25p02b", f"${f(math.sqrt(m2 - m1 ** 2), 1)}$", "数値積分の標準偏差")
E.check("s25p02b", close(eps.std(), 38.47, 0.1), f"シミュレーションの標準偏差 {eps.std():.3f}")
# (c) JB をデータの歪度・尖度から直接（scipy の jarque_bera は標本モーメントで計算するので、同じ b1,b2 をもつ式で）
n, b1, b2 = 120, 0.30, 4.50
jb = n / 6 * (b1 ** 2 + (b2 - 3) ** 2 / 4)
it = E.item("s25p02c")
ans = it["choices"][it["answer"]]
E.check("s25p02c", f(jb, 2) in ans and "重い" in ans and jb > stats.chi2.ppf(0.95, 2), f"JB={jb:.3f}、棄却、裾が重い")
others = [c for i, c in enumerate(it["choices"]) if i != it["answer"]]
E.check("s25p02c", all(("軽い" in c) or (f(jb, 2) not in c) for c in others), "他の選択肢は値か解釈が誤り")
E.check("s25p02c", close(stats.chi2.ppf(0.95, 2), 5.991, 5e-4) and close(stats.chi2.ppf(0.95, 1), 3.841, 5e-4), "χ² の点")

# ---- 大問3 安定性
val = integrate.dblquad(lambda x, y: 0.1 * math.exp(-0.1 * x) / 24, 0, 24, lambda y: 0, lambda y: y)[0]
E.choice("s25p03a", f"${f(val, 3)}$", "二重積分で P(X≤Y)")
X = rng.exponential(10, 1_000_000)
Y = rng.uniform(0, 24, 1_000_000)
E.check("s25p03a", close((X <= Y).mean(), val, 0.002), f"シミュレーション {(X <= Y).mean():.4f}")
E.check("s25p03a", close(math.exp(-2.4), 0.0907, 5e-5) and close(math.exp(-1.2), 0.3012, 5e-5), "e^-2.4, e^-1.2")
ln = stats.lognorm(s=0.4, scale=math.exp(3.0))
E.choice("s25p03b", f"${f(ln.std() / ln.mean(), 3)}$", "scipy lognorm の std/mean")
E.check("s25p03b", close(math.exp(0.16), 1.1735, 5e-5), "e^0.16")

# ---- 大問4 ラベル（全列挙）
perms = list(permutations(range(4)))
none = sum(all(p[i] != i for i in range(4)) for p in perms) / len(perms)
E.choice("s25p04a", f"${f(none, 3)}$", "4! 通りの全列挙")

# ---- 大問5 培地
tab = np.array([[18, 182], [10, 240]])
chi2 = stats.chi2_contingency(tab, correction=False)[0]
E.num("s25p05a", rnd(math.sqrt(chi2), 2), "連続修正なしのカイ二乗統計量の平方根（プールした z と一致）")
# (b) ウルフの区間を scipy の正規分位点と対数オッズ比の分散で
a, b, c, d = 18, 182, 10, 240
lor = math.log(a * d / (b * c))
se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
zq = stats.norm.ppf(0.975)
lo, hi = math.exp(lor - zq * se), math.exp(lor + zq * se)
E.choice("s25p05b", f"$({f(lo, 3)},\\ {f(hi, 3)})$", "対数オッズ比の正規近似（丸めなし）")
E.check("s25p05b", close(math.log(4320 / 1820), 0.8644, 5e-5), "ln(4320/1820)")
for x, v in {0.068: 1.0704, 0.091: 1.0953, 0.196: 1.2165, 1.533: 4.6321, 1.637: 5.1397, 1.661: 5.2646}.items():
    E.check("s25p05b", close(math.exp(x), v, 5e-5), f"e^{x}")
try:
    from statsmodels.stats.contingency_tables import Table2x2
    ci = Table2x2(tab).oddsratio_confint(method="normal")
    E.check("s25p05b", close(ci[0], lo, 1e-6) and close(ci[1], hi, 1e-6), "statsmodels の Table2x2 と一致")
except ImportError:
    pass
# (c) 事後予測：事後分布での数値積分とシミュレーション
post = stats.gamma(a=5, scale=1 / 9)
p0 = integrate.quad(lambda l: math.exp(-l) * post.pdf(l), 0, np.inf)[0]
E.choice("s25p05c", f"${f(p0, 3)}$", "e^{-λ} を事後分布で数値積分")
E.check("s25p05c", close(stats.nbinom.pmf(0, 5, 9 / 10), p0, 1e-12), "負の二項の事後予測分布の P(0) と一致")
lam_s = post.rvs(500_000, random_state=1)
E.check("s25p05c", close((rng.poisson(lam_s) == 0).mean(), p0, 0.003), "シミュレーション")
E.check("s25p05c", close(math.exp(-0.556), 0.5735, 5e-5) and close(math.exp(-0.6), 0.5488, 5e-5), "e^-0.556, e^-0.6")

# ---- 大問6 2ロット：同じ要約統計量をもつデータを作って scipy で区間を計算
def make(nn, m, s_):
    z = rng.normal(size=nn)
    z = (z - z.mean()) / z.std(ddof=1)
    return m + s_ * z


A, B = make(8, 99.6, 1.2), make(10, 98.4, 1.0)
r_ = stats.ttest_ind(A, B, equal_var=True)
ci = r_.confidence_interval(0.95)
E.num("s25p06a", rnd(ci.low, 3), "scipy ttest_ind の信頼区間")
E.check("s25p06a", close(stats.t.ppf(0.975, 16), 2.120, 5e-4) and close(stats.t.ppf(0.975, 18), 2.101, 5e-4), "t 点")

# ---- 大問7 分注機：最大値/θ ~ Beta(5,1)
cc = 2.0 * 0.95 ** 0.2
E.check("s25p07a", close(stats.beta(5, 1).sf(cc / 2.0), 0.05, 1e-12), "有意水準 5%")
pw = stats.beta(5, 1).sf(cc / 2.5)
E.choice("s25p07a", f"${f(pw, 3)}$", "Beta(5,1) で検出力")
M = rng.uniform(0, 2.5, size=(400_000, 5)).max(axis=1)
E.check("s25p07a", close((M > cc).mean(), pw, 0.003), "シミュレーション")
E.check("s25p07a", close(0.95 ** 0.2, 0.98979, 5e-6), "0.95^{1/5}")

# ---- 大問8 インキュベーター：推移行列の固有ベクトル
P = np.array([[0.5, 0.5, 0, 0], [0.2, 0.4, 0.4, 0], [0, 0.3, 0.4, 0.3], [0, 0, 0.6, 0.4]])
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi /= pi.sum()
E.choice("s25p08a", f"${f(pi[3], 3)}$", "推移行列の固有値1の左固有ベクトル")
E.check("s25p08a", np.allclose(np.linalg.matrix_power(P, 200)[0], pi), "P^200 の行と一致")
# (b) DF 検定：τ と 5% 点（MacKinnon の近似式で n=100 の点を確かめる）
tau = -0.12 / 0.045
T = 100
cv = -2.8621 - 2.738 / T - 8.36 / T ** 2   # MacKinnon (1996) 定数項ありの 5% 点の応答曲面
E.check("s25p08b", close(cv, -2.89, 0.01), f"DF の 5% 点 {cv:.3f} ≈ -2.89")
E.check("s25p08b", close(stats.t.ppf(0.05, 98), -1.661, 5e-4), "t(98) の下側 5% 点")
it = E.item("s25p08b")
E.check("s25p08b", tau > cv and "棄却できない" in it["choices"][it["answer"]] and f(tau, 2) in it["choices"][it["answer"]],
        f"τ={tau:.3f} > {cv:.3f} で棄却できない")
# 単位根のもとでの τ の分布をシミュレーションして 5% 点を確かめる
sims = []
for _ in range(4000):
    e = rng.normal(size=101)
    x = np.cumsum(e)
    dx, xl = np.diff(x), x[:-1]
    Xm = np.column_stack([np.ones(100), xl])
    beta, rss, *_ = np.linalg.lstsq(Xm, dx, rcond=None)
    s2 = rss[0] / 98
    sev = math.sqrt(s2 * np.linalg.inv(Xm.T @ Xm)[1, 1])
    sims.append(beta[1] / sev)
q5 = np.quantile(sims, 0.05)
E.check("s25p08b", close(q5, -2.89, 0.08) and tau > q5, f"シミュレーションの 5% 点 {q5:.3f}")

# ---- 大問9 検量線
it = E.item("s25p09a")
E.check("s25p09a", "不均一分散" in it["choices"][it["answer"]] and "関数形" in it["choices"][it["answer"]], "正解の選択肢の内容")
# (b) 正規モデルの最大対数尤度を scipy で（σ̂²=RSS/n の正規分布で残差を評価するのと同じ値になる）
n = 12
aic = {}
for deg, rss in {1: 0.048, 2: 0.021, 3: 0.019}.items():
    sig = math.sqrt(rss / n)
    # 残差の2乗和が rss になる任意の残差ベクトルで対数尤度は同じ
    res_ = np.full(n, math.sqrt(rss / n))
    ll = stats.norm.logpdf(res_, 0, sig).sum()
    aic[deg] = -2 * ll + 2 * (deg + 2) - n * (math.log(2 * math.pi) + 1)
best = min(aic, key=aic.get)
E.check("s25p09b", best == 2, f"AIC* {aic}")
E.choice("s25p09b", f"={f(aic[2], 2)}$", "scipy の対数尤度から AIC*")
for q_, v in {0.004: -5.5215, 0.00175: -6.3481, 0.019 / 12: -6.4482}.items():
    E.check("s25p09b", close(math.log(q_), v, 5e-5), f"ln {q_}")
# (c) VIF を補助回帰で
x = np.repeat([2, 4, 6, 8, 10, 12], 2).astype(float)


def vif(a1, a2):
    Xm = np.column_stack([np.ones_like(a2), a2])
    beta, rss, *_ = np.linalg.lstsq(Xm, a1, rcond=None)
    r2 = 1 - rss[0] / ((a1 - a1.mean()) ** 2).sum()
    return 1 / (1 - r2)


v0, v1 = vif(x, x ** 2), vif(x - 7, (x - 7) ** 2)
E.check("s25p09c", close(np.corrcoef(x, x ** 2)[0, 1], 0.979, 5e-4), "相関係数 0.979")
it = E.item("s25p09c")
ch = nums("s25p09c")
k = int(np.argmin([abs(a - v0) + abs(b_ - v1) for a, b_ in ch]))
E.check("s25p09c", k == it["answer"], f"補助回帰の VIF {v0:.3f}, {v1:.3f} に最も近い組が正解")

# ---- 大問10 スプライン
tt = np.array([30, 37, 41])
yy = 2.0 + 0.30 * tt - 0.75 * np.maximum(tt - 37, 0)
slope = (yy[2] - yy[1]) / (41 - 37)
E.choice("s25p10a", f"予測値 ${f(yy[2], 2)}$、傾き ${f(slope, 2)}$", "折れ点の後の2点から傾き")

# ---- 大問11 分割区：計画行列の階数で自由度を数える
days, A_, B_ = 4, 3, 2
rows = [(d_, a_, b_) for d_ in range(days) for a_ in range(A_) for b_ in range(B_)]


def dummies(f_):
    keys = sorted(set(f_(r) for r in rows))
    return np.array([[1.0 if f_(r) == k_ else 0.0 for k_ in keys] for r in rows])


rk = lambda *ms: np.linalg.matrix_rank(np.column_stack(ms))
one = np.ones((len(rows), 1))
Dd, Da, Db = dummies(lambda r: r[0]), dummies(lambda r: r[1]), dummies(lambda r: r[2])
Dda, Dab = dummies(lambda r: (r[0], r[1])), dummies(lambda r: (r[1], r[2]))
e1 = rk(one, Dda) - rk(one, Dd, Da)
full = rk(one, Dda, Db, Dab)
e2 = len(rows) - full
it = E.item("s25p11a")
E.check("s25p11a", f"自由度 {e1}）、培地は 2 次誤差（自由度 {e2}）" in it["choices"][it["answer"]], f"1次誤差 {e1}、2次誤差 {e2}")
# (b) 逸脱度残差を scipy の対数確率から
y, mu = 12, 7.5
dev = 2 * (stats.poisson.logpmf(y, y) - stats.poisson.logpmf(y, mu))
E.num("s25p11b", rnd(math.sqrt(dev), 2), "2{log p(y;y) − log p(y;μ)} の平方根")
E.check("s25p11b", close(math.log(1.6), 0.4700, 5e-5), "ln1.6")

# ---- 大問12 非回答：シミュレーションで母集団を作り、同じ設計で推定する
est = np.average([72.0, 60.0], weights=[240, 160])
E.num("s25p12a", rnd(72.0 - est, 1), "層の大きさで重み付けした平均との差")

# ---- 大問13 CH 指標（scipy なしで F 比として計算し直す）
n, T = 30, 300.0
chv = {kk: ((T - W) / (kk - 1)) * (n - kk) / W for kk, W in {2: 150, 3: 90, 4: 75, 5: 66}.items()}
kb = max(chv, key=chv.get)
it = E.item("s25p13a")
E.check("s25p13a", kb == 3 and f"k={kb}" in it["choices"][it["answer"]], f"CH {chv}")
E.choice("s25p13a", f"指標の値 ${f(chv[kb], 1)}$", "最大の CH 指標の値")

# ---- 大問14
R = np.array([[1, 0.6, 0.5], [0.6, 1, 0.3], [0.5, 0.3, 1]])
E.check("s25p14a", np.all(np.linalg.eigvalsh(R) > 0), "相関行列が正定値")
E.choice("s25p14a", f"${f(1 - 1 / np.linalg.inv(R)[0, 0], 3)}$", "1 − 1/(R^{-1})_{11}")
# (b) 正準相関の逐次検定を、正準相関係数をもつ相関行列を作って固有値から求め直す
Rb = np.eye(4)
Rb[0, 2] = Rb[2, 0] = 0.6
Rb[1, 3] = Rb[3, 1] = 0.2
R11, R12, R22 = Rb[:2, :2], Rb[:2, 2:], Rb[2:, 2:]
ev = np.sort(np.linalg.eigvals(np.linalg.inv(R11) @ R12 @ np.linalg.inv(R22) @ R12.T).real)[::-1]
fac = 60 - 1 - (2 + 2 + 1) / 2
s1 = -fac * np.log(1 - ev).sum()
s2 = -fac * np.log(1 - ev[1])
ok = s1 > stats.chi2.ppf(0.95, 4) and s2 < stats.chi2.ppf(0.95, 1)
it = E.item("s25p14b")
E.check("s25p14b", ok and f(s1, 2) in it["choices"][it["answer"]] and f(s2, 2) in it["choices"][it["answer"]]
        and "1 つ" in it["choices"][it["answer"]], f"統計量 {s1:.3f}, {s2:.3f}")
for q_, v in {0.4: -0.9163, 0.64: -0.4463, 0.8: -0.2231, 0.96: -0.0408}.items():
    E.check("s25p14b", close(math.log(q_), v, 5e-5), f"ln {q_}")
E.check("s25p14b", close(stats.chi2.ppf(0.95, 4), 9.488, 5e-4), "χ²_4 の 5% 点")

E.done()
