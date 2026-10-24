"""模擬試験 第5回（set05）の検算。generator/set05.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・定義からの数値計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(5)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value, what):
    """選択肢の数値のうち value に最も近いものが正解で、2番目に近いものとは十分離れているか。"""
    it = E.item(iid)
    v = nums(iid)
    dist = sorted((abs(x - value), k) for k, x in enumerate(v))
    ok = dist[0][1] == it["answer"] and dist[1][0] > 3 * dist[0][0] + 1e-9
    E.check(iid, ok, f"{what}: 最も近い選択肢 {dist[0][1]}（値 {value:.5f}）が正解 {it['answer']}")


def distinct_rounded(iid):
    v = nums(iid)
    E.check(iid, len(set(v)) == 5, f"選択肢の数値がすべて異なる {v}")


def phi_ok(table):
    return all(abs(stats.norm.cdf(z) - v) < 6e-5 for z, v in table.items())


# ---- 大問1：スクリーニング
# (a) 感染と2つの検査結果の全組合せを列挙して P(感染 | A+ かつ B+)
prev, seA, spA, seB, spB = 0.02, 0.90, 0.95, 0.80, 0.98
joint = {}
for d, a, b in product((0, 1), repeat=3):
    pd = prev if d else 1 - prev
    pa = (seA if a else 1 - seA) if d else ((1 - spA) if a else spA)
    pb = (seB if b else 1 - seB) if d else ((1 - spB) if b else spB)
    joint[(d, a, b)] = pd * pa * pb
ppv = joint[(1, 1, 1)] / (joint[(1, 1, 1)] + joint[(0, 1, 1)])
E.choice("s05p01a", f"${f(ppv, 3)}$", "全組合せの列挙で P(感染|両方陽性)")
sim_d = rng.random(2_000_000) < prev
sim_a = np.where(sim_d, rng.random(sim_d.size) < seA, rng.random(sim_d.size) < 1 - spA)
sim_b = np.where(sim_d, rng.random(sim_d.size) < seB, rng.random(sim_d.size) < 1 - spB)
pos = sim_a & sim_b
E.check("s05p01a", abs(sim_d[pos].mean() - ppv) < 0.01, f"シミュレーションの陽性的中率 {sim_d[pos].mean():.4f}")
distinct_rounded("s05p01a")
# (b) プール内の感染者数を二項分布で足し上げて期待検査回数（正確な 0.98^10 を使い、示した値との差も確認）
E.check("s05p01b", abs(0.98 ** 10 - 0.8171) < 5e-5, "0.98^10 = 0.8171（4桁）")
bin10 = stats.binom(10, 0.02)
exp_per_pool = sum(bin10.pmf(j) * (1 if j == 0 else 11) for j in range(11))
E.num("s05p01b", rnd(100 * exp_per_pool, 1), "二項分布の確率で 1 + 10·1{感染者≧1} の期待値を足し上げ")
pools = rng.random((2_000_000, 10)) < 0.02
sim_tests = 100 * (1 + 10 * pools.any(axis=1)).mean()
E.check("s05p01b", abs(sim_tests - 282.9) < 1.0, f"シミュレーション（標準誤差は約 0.27） {sim_tests:.2f}")
# (c) チェビシェフの上界が 0.10 以下になる最小の n を p=1/2（最悪）で探す
n = 1
while 0.25 / (n * 0.02 ** 2) > 0.10 + 1e-15:
    n += 1
E.choice("s05p01c", f"${n}$", "最悪の p=1/2 で上界≦0.10 となる最小の n を探索")
nn = 1
while stats.norm.ppf(0.95) * math.sqrt(0.25 / nn) > 0.02:
    nn += 1
E.check("s05p01c", nn in (1691, 1692) and n != nn, f"正規近似の n={nn} は誤答の選択肢（正解と別）")

# ---- 大問2：血圧
mu, sig, rho = 128, 20, 0.6
cov = np.array([[sig ** 2, rho * sig ** 2], [rho * sig ** 2, sig ** 2]])
X = rng.multivariate_normal([mu, mu], cov, size=1_000_000)
c = np.corrcoef(X[:, 0], X[:, 1] - X[:, 0])[0, 1]
sym_r = sp.symbols("r")
exact = sp.nsimplify((sym_r - 1) / sp.sqrt(2 * (1 - sym_r))).subs(sym_r, sp.Rational(3, 5))
E.choice("s05p02a", f"${f(float(exact), 3)}$", "sympy で Cov/√(Var·Var)")
E.check("s05p02a", abs(c - float(exact)) < 0.003, f"シミュレーションの相関 {c:.4f}")
distinct_rounded("s05p02a")
E.check("s05p02b", phi_ok({0.60: 0.7257, 0.75: 0.7734, 1.00: 0.8413, 1.75: 0.9599}), "示した Φ の値が正しい")
cm = mu + cov[0, 1] / cov[0, 0] * (168 - mu)
cs = math.sqrt(cov[1, 1] - cov[0, 1] ** 2 / cov[0, 0])
pr = stats.norm.cdf(140, cm, cs)
E.choice("s05p02b", f"${f(pr, 3)}$", "条件付き分布を共分散行列から作り scipy で P(X2<140)")
band = X[np.abs(X[:, 0] - 168) < 0.5]
E.check("s05p02b", abs((band[:, 1] < 140).mean() - pr) < 0.02, f"X1≈168 の標本での割合 {(band[:, 1] < 140).mean():.3f}（{len(band)} 件）")
distinct_rounded("s05p02b")

# ---- 大問3：潜伏期間と回復
E.check("s05p03a", abs(math.exp(-2) - 0.1353) < 5e-5 and abs(math.exp(-4) - 0.0183) < 5e-5, "e^-2, e^-4 の値")
dens = lambda s: integrate.quad(lambda u: stats.expon(scale=5).pdf(u) * stats.expon(scale=2.5).pdf(s - u), 0, s)[0]
tail = 1 - integrate.quad(dens, 0, 10)[0]
E.num("s05p03a", rnd(tail, 3), "畳み込みを数値積分")
S = rng.exponential(5, 1_000_000) + rng.exponential(2.5, 1_000_000)
E.check("s05p03a", abs((S > 10).mean() - tail) < 0.002, f"シミュレーション {(S > 10).mean():.4f}")


def km_redistribute(times, events, t):
    """Efron の「右への再配分」で KM を求める（積の公式とは別の計算）。"""
    order = sorted(range(len(times)), key=lambda i: (times[i], -events[i]))
    mass = {i: 1 / len(times) for i in order}
    for pos_, i in enumerate(order):
        if not events[i]:
            later = order[pos_ + 1:]
            if later:
                share = mass[i] / len(later)
                for j in later:
                    mass[j] += share
                mass[i] = 0
    return 1 - sum(mass[i] for i in order if events[i] and times[i] <= t)


tm = [3, 4, 5, 5, 7, 8, 10, 12]
ev = [1, 0, 1, 1, 0, 1, 1, 0]
E.choice("s05p03b", f"${f(km_redistribute(tm, ev, 9), 3)}$", "右への再配分アルゴリズムで S(9)")
distinct_rounded("s05p03b")

# ---- 大問4：安全性
E.check("s05p04a", abs(0.05 ** (1 / 150) - 0.98023) < 5e-6, "0.05^(1/150) の値")
up = stats.beta.ppf(0.95, 1, 150)  # x=0 の Clopper-Pearson 上側限界 = Beta(1, n) の 95% 点
E.num("s05p04a", rnd(up, 4), "Beta(1,150) の 95% 点（クロッパー・ピアソン）")
E.check("s05p04a", abs(stats.binom.cdf(0, 150, up) - 0.05) < 1e-9, "上限で P(X≦0)=0.05")
E.check("s05p04a", rnd(3 / 150, 4) != rnd(up, 4), "3 の法則の値は正解と区別される")
lam = 0.5
E.check("s05p04b", abs(math.exp(0.5) - 1.6487) < 5e-5 and abs(math.exp(-0.5) - 0.6065) < 5e-5, "e^±0.5 の値")
# 漸近分散をシミュレーションで比べる（n=2000 日を多数回）
nd, R = 2000, 4000
cnt = rng.poisson(lam, size=(R, nd))
xbar = cnt.mean(axis=1)
lt = -np.log((cnt == 0).mean(axis=1))
eff_sim = xbar.var() / lt.var()
# フィッシャー情報量から：0/1 データの情報量 = (dp0/dλ)^2/(p0(1-p0)) と、ポアソンの情報量 1/λ の比
p0 = sp.exp(-sp.Symbol("l"))
info01 = (sp.diff(p0, "l")) ** 2 / (p0 * (1 - p0))
eff_info = float((info01 / (1 / sp.Symbol("l"))).subs("l", lam))
E.choice("s05p04b", f"${f(eff_info, 3)}$", "情報量の比（0/1 データ ÷ 件数データ）")
E.check("s05p04b", abs(eff_sim - eff_info) < 0.05, f"シミュレーションの分散比 {eff_sim:.3f}")

# ---- 大問5：クロスオーバー試験
z = rng.standard_normal(10)
z = (z - z.mean()) / z.std(ddof=1)
d = -4.2 + 5.5 * z
tt = stats.ttest_1samp(d, 0).statistic
E.choice("s05p05a", f"${f(tt, 2)}$", "要約統計量を再現したデータで scipy ttest_1samp")
E.check("s05p05a", abs(stats.t.ppf(0.975, 9) - 2.262) < 5e-4, "t_0.025(9)=2.262")
t_ind = stats.ttest_ind_from_stats(131.8, 9.0, 10, 136.0, 10.0, 10, equal_var=False).statistic
E.check("s05p05a", f"${f(t_ind, 2)}$" == E.item("s05p05a")["choices"][0], "独立2標本扱いの値が誤答の選択肢")
distinct_rounded("s05p05a")
pv = [0.035, 0.002, 0.21, 0.019, 0.047, 0.009, 0.031]
# BH の調整済み p 値（後ろからの累積最小）で 0.05 以下を数える
o = np.argsort(pv)
ps = np.array(pv)[o]
m = len(pv)
adj = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
k_bh = int((adj <= 0.05).sum())
E.choice("s05p05b", f"{k_bh} 個", "BH の調整済み p 値で数える")
holm = np.maximum.accumulate(np.minimum(1, ps * (m - np.arange(m))))
E.check("s05p05b", int((holm <= 0.05).sum()) == 1 and sum(p <= 0.05 / m for p in pv) == 1, "ホルム法・ボンフェローニ法はともに1個（誤答）")

# ---- 大問6：罹患率
# 条件付き二項分布のスコア統計量を、期待度数 (28, 42) のカイ二乗適合度統計量の平方根として求める
chi = stats.chisquare([40, 30], [28, 42]).statistic
E.choice("s05p06a", f"${f(math.sqrt(chi), 2)}$", "2セルのカイ二乗適合度統計量の平方根")
distinct_rounded("s05p06a")


def negll(b):
    mu1 = 4000 * np.exp(b[0] + b[1])
    mu0 = 6000 * np.exp(b[0])
    return -(40 * np.log(mu1) - mu1 + 30 * np.log(mu0) - mu0)


res = optimize.minimize(negll, [math.log(0.005), 0.5], method="BFGS")
b0, b1 = res.x
# 観測情報量を数値微分で
h = 1e-4
H = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        e_i, e_j = np.eye(2)[i] * h, np.eye(2)[j] * h
        H[i, j] = (negll(res.x + e_i + e_j) - negll(res.x + e_i - e_j) - negll(res.x - e_i + e_j) + negll(res.x - e_i - e_j)) / (4 * h * h)
se_num = math.sqrt(np.linalg.inv(H)[1, 1])
E.num("s05p06b", rnd(se_num, 3), "対数尤度を数値最適化し、ヘッセ行列の逆行列から")
E.check("s05p06b", abs(b1 - math.log(2)) < 1e-4, f"β1 の最尤推定値 {b1:.5f} = log 2")

# ---- 大問7：入院のマルコフ連鎖
P = np.array([[0.80, 0.05, 0.14, 0.01], [0.20, 0.70, 0.00, 0.10], [0, 0, 1, 0], [0, 0, 0, 1]])
Pn = np.linalg.matrix_power(P, 2000)
E.choice("s05p07a", f"${f(Pn[1, 2], 2)}$", "推移行列の 2000 乗で I から D に吸収される確率")
st = np.full(20000, 1)
for _ in range(3000):
    u = rng.random(st.size)
    cum = P[st].cumsum(axis=1)
    st = (u[:, None] > cum).sum(axis=1)
E.check("s05p07a", abs((st == 2).mean() - Pn[1, 2]) < 0.015, f"シミュレーション {(st == 2).mean():.3f}")
distinct_rounded("s05p07a")

# ---- 大問8：重回帰
Rm = np.array([[1, 0.5, 0.6], [0.5, 1, 0.4], [0.6, 0.4, 1]])
Ryx, Rxx = Rm[0:1, 1:], Rm[1:, 1:]
ev_cc = np.linalg.eigvals(np.linalg.inv(Rxx) @ Ryx.T @ Ryx)  # 正準相関の2乗（Σ_yy=1）
E.choice("s05p08a", f"${f(math.sqrt(max(ev_cc.real)), 3)}$", "正準相関の固有値問題")
w = rng.multivariate_normal([0, 0, 0], Rm, size=400_000)
coef = np.linalg.lstsq(np.c_[np.ones(len(w)), w[:, 1:]], w[:, 0], rcond=None)[0]
E.check("s05p08a", abs(np.corrcoef(w[:, 1:] @ coef[1:], w[:, 0])[0, 1] - 0.6637) < 0.003, "シミュレーションの重相関")
vif = np.linalg.inv(Rxx)[0, 0]
E.choice("s05p08b", f"${f(math.sqrt(vif), 3)}$", "説明変数の相関行列の逆行列の対角成分 → √VIF")
distinct_rounded("s05p08b")
lm = 50 * 0.15
crit = {k: stats.chi2.isf(0.05, k) for k in (2, 3, 49)}
E.check("s05p08c", abs(crit[2] - 5.991) < 5e-4 and abs(crit[3] - 7.815) < 5e-4 and abs(crit[49] - 66.339) < 5e-4, "χ² の上側5%点")
it = E.item("s05p08c")
E.check("s05p08c", "自由度 2" in it["choices"][it["answer"]] and "棄却される" in it["choices"][it["answer"]] and lm > crit[2],
        "正解は自由度2で棄却する記述で、nR²=7.5 > 5.991")
E.check("s05p08c", lm < crit[3] and lm < crit[49], "誤答の記述（自由度3・49 で棄却しない）は自由度が誤り")

# ---- 大問9：DAG と欠落変数
# 線形ガウスの構造方程式に無作為な係数を入れ、偏相関が 0 になるかで条件付き独立を判定する
def pcor(Sig, i, j, given):
    idx = [i, j] + list(given)
    K = np.linalg.inv(Sig[np.ix_(idx, idx)])
    return -K[0, 1] / math.sqrt(K[0, 0] * K[1, 1])


Gi, Si, Ti, Ci = 0, 1, 2, 3
ok_all = []
for _ in range(20):
    a, b, cc, dd = rng.uniform(0.5, 2, 4) * rng.choice([-1, 1], 4)
    B = np.zeros((4, 4))
    B[Si, Gi], B[Ti, Si], B[Ci, Ti], B[Ci, Gi] = a, b, cc, dd
    A = np.linalg.inv(np.eye(4) - B)
    Sig = A @ A.T
    stmts = [abs(pcor(Sig, Si, Ci, [Gi])) < 1e-9, abs(pcor(Sig, Gi, Ti, [])) < 1e-9,
             abs(pcor(Sig, Gi, Ti, [Si])) < 1e-9, abs(pcor(Sig, Gi, Ti, [Si, Ci])) < 1e-9,
             not abs(pcor(Sig, Si, Ci, [Gi, Ti])) < 1e-9]
    ok_all.append(stmts)
truth_orig = [all(col) for col in zip(*ok_all)]  # 元の順序での真偽
labels = ["S⊥C|G", "G⊥T", "G⊥T|S", "G⊥T|S,C", "S,C は G,T を与えても非独立"]
it = E.item("s05p09a")
true_text = [lab for lab, t_ in zip(labels, truth_orig) if t_]
E.check("s05p09a", truth_orig == [False, False, True, False, False], f"真の記述は1つだけ: {true_text}")
E.check("s05p09a", "$G$ と $T$ は、$S$ を与えると" in it["choices"][it["answer"]], "正解の選択肢が G⊥T|S")
nobs = 1_000_000
cv = np.array([[100, 40], [40, 144]])
xw = rng.multivariate_normal([20, 50], cv, size=nobs)
y = 4 - 0.020 * xw[:, 0] - 0.030 * xw[:, 1] + rng.normal(0, 0.3, nobs)
slope = np.polyfit(xw[:, 0], y, 1)[0]
E.choice("s05p09b", f"${f(slope, 3)}$", f"シミュレーションで単回帰の傾き {slope:.5f}")
distinct_rounded("s05p09b")

# ---- 大問10：変量効果
s2a, s2 = sp.symbols("s2a s2")
sol = sp.solve([sp.Eq(s2 + 8 * s2a, sp.Rational(260, 5)), sp.Eq(s2, sp.Rational(504, 42))], [s2a, s2])
icc = sol[s2a] / (sol[s2a] + sol[s2])
E.num("s05p10a", rnd(float(icc), 3), "sympy で期待平均平方の方程式を解く")
# 推定方法の妥当性：σA²=5, σ²=12 で生成したデータに同じ推定をすると平均的に 5 と 12 に近い
est = []
for _ in range(4000):
    a = rng.normal(0, math.sqrt(5), 6)
    yy = a[:, None] + rng.normal(0, math.sqrt(12), (6, 8))
    msa = 8 * ((yy.mean(axis=1) - yy.mean()) ** 2).sum() / 5
    mse = ((yy - yy.mean(axis=1, keepdims=True)) ** 2).sum() / 42
    est.append(((msa - mse) / 8, mse))
est = np.array(est).mean(axis=0)
E.check("s05p10a", abs(est[0] - 5) < 0.3 and abs(est[1] - 12) < 0.2, f"分散成分推定量の不偏性 {est.round(3)}")

# ---- 大問11：k 平均と HT 推定
pts = np.array([[20, 25], [22, 27], [24, 23], [30, 33], [32, 35], [26, 29]], float)
cen, lab = kmeans2(pts, pts[[0, 2]].copy(), iter=50, minit="matrix")
cen = cen[np.argsort(cen[:, 0])]
E.choice("s05p11a", f"$({f(cen[1][0], 1)},\\ {f(cen[1][1], 1)})$", "scipy kmeans2（初期中心 A, C）の2つ目の中心")
E.check("s05p11a", np.allclose(cen[0], [22, 25]), f"1つ目の中心 {cen[0]}")
# HT：抽出設計の不偏性の確認として、小さな母集団で PPS 系統抽出の HT 推定の期待値＝母合計
y_ = [12, 30, 8, 45]
pi_ = [0.10, 0.25, 0.05, 0.40]
E.num("s05p11b", rnd(sum(np.array(y_) * (1 / np.array(pi_))), 1), "重み 1/π を掛けた和（numpy）")
popy = rng.integers(1, 60, 40).astype(float)
size = popy + rng.integers(1, 20, 40)
pis = 4 * size / size.sum()
assert pis.max() < 1
ests = []
for _ in range(20000):
    start = rng.random()
    cum = np.cumsum(pis)
    sel = np.searchsorted(cum, start + np.arange(4))
    ests.append((popy[sel] / pis[sel]).sum())
E.check("s05p11b", abs(np.mean(ests) / popy.sum() - 1) < 0.01, "HT 推定量が母合計の不偏推定になっている（PPS 系統抽出のシミュレーション）")

# ---- 大問12：ユール・ウォーカー
Rmat = np.array([[1, 0.6], [0.6, 1]])
phi = np.linalg.solve(Rmat, [0.6, 0.2])
E.choice("s05p12a", f"=({f(phi[0], 2)},\\ {f(phi[1], 2)})", "ユール・ウォーカー方程式を行列で解く")
# 推定した AR(2) の理論自己相関が r1, r2 を再現するか（長い系列で）
xs = np.zeros(400_000)
eps = rng.standard_normal(xs.size)
for t_ in range(2, xs.size):
    xs[t_] = phi[0] * xs[t_ - 1] + phi[1] * xs[t_ - 2] + eps[t_]
xs = xs[1000:]
ac1 = np.corrcoef(xs[1:], xs[:-1])[0, 1]
ac2 = np.corrcoef(xs[2:], xs[:-2])[0, 1]
E.check("s05p12a", abs(ac1 - 0.6) < 0.01 and abs(ac2 - 0.2) < 0.01, f"シミュレーションの自己相関 {ac1:.3f}, {ac2:.3f}")

# ---- 大問13：ベイズと LOOCV
grid = np.linspace(-20, 40, 600_001)
post = stats.norm.pdf(grid, 4, 3) * stats.norm.pdf(10, grid, 2)
post /= post.sum()
cdf = post.cumsum()
lo, hi = grid[np.searchsorted(cdf, 0.025)], grid[np.searchsorted(cdf, 0.975)]
pmean = (grid * post).sum()
psd = math.sqrt(((grid - pmean) ** 2 * post).sum())
E.choice("s05p13a", f"$({f(pmean - 1.96 * psd, 2)},\\ {f(pmean + 1.96 * psd, 2)})$", "格子上で事後分布を数値計算")
E.check("s05p13a", abs(lo - (pmean - 1.96 * psd)) < 0.01 and abs(hi - (pmean + 1.96 * psd)) < 0.01, f"数値の分位点 ({lo:.3f}, {hi:.3f})")
x = np.array([1, 2, 3, 4, 5.0])
yv = np.array([3, 7, 8, 8, 13.0])
errs = []
for i in range(5):
    msk = np.arange(5) != i
    bb, aa = np.polyfit(x[msk], yv[msk], 1)
    errs.append(yv[i] - (aa + bb * x[i]))
E.num("s05p13b", rnd(np.mean(np.square(errs)), 2), "1組ずつ除いて実際に当てはめ直す")
bb, aa = np.polyfit(x, yv, 1)
E.check("s05p13b", abs(bb - 2.1) < 1e-9 and abs(aa - 1.5) < 1e-9, "問題文の当てはめ直線 1.5+2.1x")

E.done()
