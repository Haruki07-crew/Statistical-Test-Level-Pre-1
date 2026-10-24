"""模擬試験 第16回（set16）の検算。generator/set16.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値積分・数値最適化・実際のデータを作っての当てはめ）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations, groupby, product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(16)
rng = np.random.default_rng(20261016)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table_ok(pairs, tol):
    return all(abs(a - b) <= tol for a, b in pairs)


# ---- 大問1
# (a) 着氷の有無と 3 つのセンサーの状態を総当たり
c, q, p = 0.05, 0.4, 0.02
tot = 0.0
for ice in (0, 1):
    pi = q if ice else p
    for st in product((0, 1), repeat=3):  # 1 = 故障
        if sum(st) >= 2:
            tot += (c if ice else 1 - c) * np.prod([pi if s_ else 1 - pi for s_ in st])
E.num("s16p01a", rnd(tot, 4), "着氷と3センサーの全状態を総当たり")
ice = rng.random(2_000_000) < c
fail = rng.random((2_000_000, 3)) < np.where(ice, q, p)[:, None]
E.check("s16p01a", abs((fail.sum(1) >= 2).mean() - tot) < 0.0005, f"シミュレーション {(fail.sum(1) >= 2).mean():.5f}")
# (b) 事後確率が等しくなる点を数値的に解く
post = lambda x: 0.1 * stats.norm.pdf(x, 0, 3) - 0.9 * stats.norm.pdf(x, 0, 1)
cq = optimize.brentq(post, 0.5, 6)
E.choice("s16p01b", f"${f(cq, 2)}$", f"π_F f_F = π_N f_N を brentq で解く c={cq:.5f}")
E.check("s16p01b", post(cq + 0.1) > 0 > post(cq - 0.1) and abs(np.log(3) - 1.0986) < 5e-5, "外側で不具合の方が大きい、ln3 の値")
lab = rng.random(2_000_000) < 0.1
xs2 = np.where(lab, rng.normal(0, 3, 2_000_000), rng.normal(0, 1, 2_000_000))
err = lambda c_: ((np.abs(xs2) > c_) != lab).mean()
E.check("s16p01b", err(cq) < min(err(1.57), err(2.22), err(1.93), err(3.14)), f"誤判別率 c={cq:.2f}: {err(cq):.4f} が他の候補より小さい")

# ---- 大問2
ls, lm = 0.0012, 0.0008
E.choice("s16p02a", f"${f(stats.gamma.sf(1000, 3, scale=1 / (ls + lm)), 3)}$", "scipy gamma(3, scale=500).sf(1000)")
T3 = rng.exponential(1 / (ls + lm), size=(1_000_000, 3)).sum(1)
E.check("s16p02a", abs((T3 > 1000).mean() - 0.6767) < 0.002, f"シミュレーション {(T3 > 1000).mean():.4f}")
E.check("s16p02a", abs(np.exp(-2) - 0.1353) < 5e-5, "e^{-2} の値")
# (b) 同時密度 f(t, J=s) を数値積分
num = integrate.quad(lambda t: ls * np.exp(-ls * t) * np.exp(-lm * t), 300, 500)[0]
den = stats.expon(scale=1 / (ls + lm)).sf(300)
E.choice("s16p02b", f"${f(num / den, 3)}$", "シールの故障時刻の密度×モーターの生存を数値積分")
T1 = rng.exponential(1 / ls, 3_000_000)
T2 = rng.exponential(1 / lm, 3_000_000)
T = np.minimum(T1, T2)
sel = T > 300
E.check("s16p02b", abs(((T <= 500) & (T1 < T2))[sel].mean() - 0.1978) < 0.002, f"シミュレーション {((T <= 500) & (T1 < T2))[sel].mean():.4f}")
E.check("s16p02b", table_ok([(np.exp(-0.4), 0.6703), (np.exp(-0.6), 0.5488), (np.exp(-0.24), 0.7866)], 5e-5), "e^x の値")

# ---- 大問3
# (a) データを実際に作り、もとの y の尤度を数値で最大化した値（σ² と係数を最尤推定）で比べる
# 問題文の SSE と Σln y をもつデータは作りにくいので、プロファイル対数尤度の式を任意のデータで数値最大化と照合してから適用する
xd = rng.uniform(1, 3, 20)
yd = np.exp(1.5 + 0.2 * xd + rng.normal(0, 0.3, 20))


def bc(y, lam):
    return np.log(y) if lam == 0 else (y ** lam - 1) / lam


def full_ll(lam):
    z = bc(yd, lam)
    def nll(par):
        a, b, ls2 = par
        r_ = z - a - b * xd
        return -(stats.norm.logpdf(r_, 0, np.exp(ls2 / 2)).sum() + (lam - 1) * np.log(yd).sum())
    return -optimize.minimize(nll, [0, 0, 0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}).fun


def prof(lam):
    z = bc(yd, lam)
    X = np.c_[np.ones(20), xd]
    r_ = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    return -10 * np.log(r_ @ r_ / 20) + (lam - 1) * np.log(yd).sum()


diffs = [full_ll(l_) - prof(l_) for l_ in (0, 0.5, 1)]
E.check("s16p03a", np.ptp(diffs) < 1e-4, f"数値最大化した尤度とプロファイル式の差が λ によらない定数（{diffs[0]:.4f}）")
ll = {l_: -10 * np.log(sse / 20) + (l_ - 1) * 30.0 for l_, sse in ((0, 2.40), (0.5, 9.40), (1, 40.0))}
E.choice("s16p03a", f"$({f(ll[0], 2)},\\ {f(ll[0.5], 2)},\\ {f(ll[1], 2)})$で、$\\lambda=1$", "math.log で直接計算")
E.check("s16p03a", max(ll, key=ll.get) == 1 and table_ok([(np.log(0.12), -2.1203), (np.log(0.47), -0.7550), (np.log(2.0), 0.6931)], 5e-5), "λ=1 が最大、ln の値")
# (b) 連の検定
seq = re.search(r"`([+-]+)`", E.item("s16p03b")["question"]).group(1)
n1, n2 = seq.count("+"), seq.count("-")
R = len([k for k, _ in groupby(seq)])
N = n1 + n2
Rs = []
for pos in combinations(range(N), n2):
    arr = ["+"] * N
    for i in pos:
        arr[i] = "-"
    Rs.append(len([k for k, _ in groupby(arr)]))
Rs = np.array(Rs)
z = (R - Rs.mean()) / Rs.std()
E.choice("s16p03b", f"$z={f(z, 2)}$で$|z|>1.96$なので棄却する。連が少なく", "全配置の総当たりで E[R]・V[R] を求めて z")
E.check("s16p03b", (n1, n2, R) == (14, 6, 5) and (Rs <= R).mean() < 0.025, f"正確な下側確率 {(Rs <= R).mean():.4f} も 2.5% 未満")

# ---- 大問4
xs = np.array([0, 3, 1, 6, 2, 0, 4, 1, 7, 6])
Tt = xs.sum()
E.num("s16p04a", rnd((Tt - stats.poisson(20).mean()) / stats.poisson(20).std(), 2), "scipy poisson(20) の平均と標準偏差で標準化")
E.check("s16p04a", stats.poisson(20).sf(Tt - 1) < 0.05 and stats.norm.sf(2.236) < 0.05, f"正確な片側 p 値 {stats.poisson(20).sf(Tt - 1):.4f} も 5% 未満")
# (b) 個票を作りロジスティック回帰を数値最大化
xb = np.r_[np.ones(40), np.zeros(60)]
yb = np.r_[np.ones(18), np.zeros(22), np.ones(12), np.zeros(48)]
def nll_lr(b):
    eta = b[0] + b[1] * xb
    return -(yb * eta - np.log1p(np.exp(eta))).sum()
bhat = optimize.minimize(nll_lr, [0.0, 0.0], method="BFGS", options={"gtol": 1e-10}).x
E.choice("s16p04b", f"${f(bhat[1], 2)}$", f"個票の対数尤度を数値最大化 β=({bhat[0]:.4f}, {bhat[1]:.4f})")
E.check("s16p04b", table_ok([(np.log(3.273), 1.186), (np.log(2.25), 0.811), (np.log(1.5), 0.405), (np.log(0.818), -0.201)], 5e-4), "表の対数の値")
qd = 0.95
ns_ = np.arange(2, 120)
wts = stats.poisson.pmf(ns_, 4.0) * stats.binom.pmf(2, ns_, qd)
E.num("s16p04c", rnd((ns_ * wts).sum() / wts.sum(), 2), "ベイズの定理で N の条件付き分布を作って期待値")
Nsim = rng.poisson(4.0, 3_000_000)
Ksim = rng.binomial(Nsim, qd)
E.check("s16p04c", abs(Nsim[Ksim == 2].mean() - 2.20) < 0.005, f"シミュレーション {Nsim[Ksim == 2].mean():.4f}")

# ---- 大問5
t = np.array([0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1])
nll = lambda th: -stats.weibull_min(2, scale=np.sqrt(th)).logpdf(t).sum()
th_hat = optimize.minimize_scalar(nll, bounds=(0.1, 10), method="bounded", options={"xatol": 1e-10}).x
E.choice("s16p05a", f"${f(th_hat, 2)}$", "scipy の weibull_min（尺度 √θ）の対数尤度を数値最大化")
E.check("s16p05a", abs(stats.weibull_min(2, scale=np.sqrt(2.0)).pdf(1.3) - 2 * 1.3 / 2.0 * np.exp(-1.3 ** 2 / 2.0)) < 1e-12,
        "問題文の密度は scipy の形状2・尺度√θ のワイブル分布と一致")
# (b) 文の正誤をシミュレーションで確かめる：一致性（n を増やすと R̂ が R に集中）、有限 n での偏り、漸近分散の違い
th = 2.0
R = np.exp(-1 / th)
for n_ in (8, 2000):
    Y = rng.exponential(th, size=(20_000, n_))
    Rh = np.exp(-1 / Y.mean(1))
    if n_ == 8:
        bias8 = Rh.mean() - R
    else:
        conc = (np.abs(Rh - R) < 0.03).mean()
        avar = n_ * Rh.var()
E.check("s16p05b", conc > 0.95, f"n=2000 で |R̂-R|<0.03 の割合 {conc:.4f}（一致性）")
E.check("s16p05b", abs(bias8) > 0.005, f"n=8 での R̂ の偏り {bias8:.4f}（不偏でない：選択肢 1 は誤り）")
E.check("s16p05b", abs(avar - (R / th ** 2) ** 2 * th ** 2) < 0.01 and abs(avar - th ** 2) > 1, f"√n(R̂-R) の分散 {avar:.4f} はデルタ法の値で θ²=4 と異なる（選択肢 4 は誤り）")
E.check("s16p05b", stats.normaltest(rng.exponential(th, size=(20_000, 8)).mean(1)).pvalue < 1e-6, "n=8 の標本平均は正規分布でない（選択肢 2 は誤り）")
E.choice("s16p05b", "連続写像定理により$\\hat R_n$も$R$に確率収束する", "正しい記述")

# ---- 大問6
tval = (41.0 - 40.0) / (1.6 / np.sqrt(10))
E.choice("s16p06a", f"$t={f(tval, 2)}$で、自由度9の上側5%点1.833を超える", "t 統計量")
E.check("s16p06a", stats.t.sf(tval, 9) < 0.05 < 2 * stats.t.sf(tval, 9), "片側 p<0.05、両側 p>0.05")
E.check("s16p06a", table_ok([(stats.t.isf(0.05, 9), 1.833), (stats.t.isf(0.025, 9), 2.262), (stats.t.isf(0.05, 10), 1.812)], 5e-4), "t 分布の点")
# (b) 2 変量正規から r の分布をシミュレーションし、z 変換の区間の被覆率を確かめ、正確な tanh で区間を求める
zr = np.arctanh(0.70)
lo_, hi_ = np.tanh(zr - stats.norm.isf(0.025) / np.sqrt(7)), np.tanh(zr + stats.norm.isf(0.025) / np.sqrt(7))
E.choice("s16p06b", f"$({f(lo_, 2)},\\ {f(hi_, 2)})$", "numpy の arctanh・tanh と scipy の正規分布の点")
cov_ = np.array([[1, 0.5], [0.5, 1]])
XY = rng.multivariate_normal([0, 0], cov_, size=(100_000, 10))
xc = XY - XY.mean(1, keepdims=True)
rs = (xc[..., 0] * xc[..., 1]).sum(1) / np.sqrt((xc[..., 0] ** 2).sum(1) * (xc[..., 1] ** 2).sum(1))
hw_ = 1.96 / np.sqrt(7)
cover = (np.abs(np.arctanh(rs) - np.arctanh(0.5)) <= hw_).mean()
E.check("s16p06b", abs(cover - 0.95) < 0.006 and abs(np.arctanh(rs).var() * 7 - 1) < 0.08, f"n=10 で z 区間の被覆率 {cover:.4f}、z の分散×7 = {np.arctanh(rs).var() * 7:.3f}")
xs_ = [0.127, 0.174, 0.247, 0.497, 1.238, 1.487, 1.560, 1.608, 1.739]
tab_ = [0.1263, 0.1723, 0.2421, 0.4598, 0.8449, 0.9028, 0.9154, 0.9229, 0.9401]
E.check("s16p06b", table_ok(list(zip(np.tanh(xs_), tab_)), 5e-5) and abs(np.log(17 / 3) - 1.7346) < 5e-5 and abs(stats.t.isf(0.025, 8) - 2.306) < 5e-4, "問題文の tanh・ln・t の値")
txt = E.item("s16p06b")["question"]
E.check("s16p06b", all(f"{x:.3f}" in txt for x in xs_) and all(f"{y:.4f}" in txt for y in tab_), "問題文の表に表の値がすべてある")

# ---- 大問7：複合ポアソンを実際に発生させて平均と分散、予算
Nn = rng.poisson(300, 200_000)
k_, th_ = (8 / 6) ** 2, 36 / 8   # 平均 8・標準偏差 6 のガンマ分布
Ssim = rng.gamma(Nn * k_, th_)                    # 独立なガンマの和はガンマ
E.check("s16p07a", abs(Ssim.var() / 30000 - 1) < 0.02 and abs(Ssim.mean() - 2400) < 2, f"シミュレーション 平均 {Ssim.mean():.1f}・分散 {Ssim.var():.0f}")
ES = 50 * 6 * 8
VS = 50 * 6 * (6 ** 2 + 8 ** 2)
E.num("s16p07a", rnd(stats.norm(ES, np.sqrt(VS)).isf(0.05) - (stats.norm.isf(0.05) - 1.645) * np.sqrt(VS), 0), "scipy norm.isf（表の 1.645 に合わせる）")
E.check("s16p07a", abs(stats.norm.isf(0.05) - 1.645) < 5e-4, "上側 5% 点")

# ---- 大問8
# (a) 実際の計画行列を作り、平均の信頼区間の被覆率をシミュレーション
Xd = np.c_[np.ones(40), rng.normal(size=(40, 4))]
XtXi = np.linalg.inv(Xd.T @ Xd)
hw = 2.030 * np.sqrt(430 / 35 * 0.10)
E.choice("s16p08a", f"$({f(52.0 - hw, 2)},\\ {f(52.0 + hw, 2)})$", "区間を直接計算")
E.check("s16p08a", abs(stats.t.isf(0.025, 35) - 2.030) < 5e-4, "t(35) の点")
x0 = np.r_[1, rng.normal(size=4)]
h0 = x0 @ XtXi @ x0
beta = np.array([30, 2, 3, -1, 1.5])
Y = Xd @ beta + rng.normal(0, 3.5, size=(100_000, 40))
B = Y @ Xd @ XtXi
res = Y - B @ Xd.T
s2 = (res ** 2).sum(1) / 35
cov = np.abs(B @ x0 - x0 @ beta) <= stats.t.isf(0.025, 35) * np.sqrt(s2 * h0)
E.check("s16p08a", abs(cov.mean() - 0.95) < 0.003, f"平均の区間の被覆率 {cov.mean():.4f}")
# (b) 標準偏差が当てはめ値に比例する誤差でデータを作り、重み 1/ŷ² の WLS が最も効率がよいことを確かめる
fit = np.array([20, 40, 60])
rsd = np.array([2.1, 4.0, 6.1])
E.check("s16p08b", np.ptp(rsd / fit) < 0.01 and np.ptp(rsd ** 2 / fit) > 0.3, f"SD/ŷ={rsd / fit}、分散/ŷ={rsd ** 2 / fit}")
xw = np.linspace(1, 10, 60)
Xw = np.c_[np.ones(60), xw]
mu_w = Xw @ np.array([2.0, 6.0])
Yw = mu_w + rng.normal(size=(20_000, 60)) * 0.1 * mu_w
def wls(w):
    W = Xw * w[:, None]
    return np.linalg.solve(Xw.T @ W, W.T @ Yw.T)[1]
v_ols, v_1, v_2 = wls(np.ones(60)).var(), wls(1 / mu_w).var(), wls(1 / mu_w ** 2).var()
E.check("s16p08b", v_2 < v_1 < v_ols and abs(wls(np.ones(60)).mean() - 6) < 0.01, f"傾きの分散 OLS {v_ols:.5f}・重み1/ŷ {v_1:.5f}・重み1/ŷ² {v_2:.5f}、OLS は不偏")
E.choice("s16p08b", "重み$1/\\hat y_i^2$の重み付き最小二乗法を使う（または目的変数を対数変換する）", "正しい記述")

# ---- 大問9
# (a) d 分離をモラル化で判定してバックドア基準を確かめる
edges = [("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y"), ("X", "C"), ("Y", "C")]


def desc(v):
    out, st_ = set(), [v]
    while st_:
        a_ = st_.pop()
        for s_, t_ in edges:
            if s_ == a_ and t_ not in out:
                out.add(t_)
                st_.append(t_)
    return out


def dsep(xn, yn, zs, es):
    anc = set([xn, yn]) | zs
    ch = True
    while ch:
        ch = False
        for s_, t_ in es:
            if t_ in anc and s_ not in anc:
                anc.add(s_)
                ch = True
    sub = [(s_, t_) for s_, t_ in es if s_ in anc and t_ in anc]
    und = {frozenset(e_) for e_ in sub}
    for v_ in anc:
        pa = [s_ for s_, t_ in sub if t_ == v_]
        und |= {frozenset((a_, b_)) for a_ in pa for b_ in pa if a_ != b_}
    seen, st_ = {xn}, [xn]
    while st_:
        a_ = st_.pop()
        for e_ in und:
            if a_ in e_:
                (b_,) = e_ - {a_}
                if b_ not in seen and b_ not in zs:
                    seen.add(b_)
                    st_.append(b_)
    return yn not in seen


def backdoor(zs):
    return not (zs & desc("X")) and dsep("X", "Y", zs, [e_ for e_ in edges if e_[0] != "X"])


okb = [backdoor(z_) for z_ in [set(), {"Z", "M"}, {"Z", "C"}, {"M", "C"}, {"Z"}]]
E.check("s16p09a", okb == [False, False, False, False, True], f"各選択肢のバックドア基準の判定 {okb}")
# 線形構造方程式でデータを作り、{Z} で調整した回帰が総合効果を、他の集合は偏った値を返すことを確かめる
n9 = 400_000
Zs = rng.normal(size=n9)
Xs = 0.8 * Zs + rng.normal(size=n9)
Ms = 0.5 * Xs + rng.normal(size=n9)
Ys = -0.6 * Ms + 0.7 * Zs + rng.normal(size=n9)
Cs = Xs + Ys + rng.normal(size=n9)
coef = lambda *cols: np.linalg.lstsq(np.c_[np.ones(n9), Xs, *cols], Ys, rcond=None)[0][1]
E.check("s16p09a", abs(coef(Zs) + 0.3) < 0.01 and abs(coef() + 0.3) > 0.1 and abs(coef(Zs, Ms) + 0.3) > 0.1 and abs(coef(Zs, Cs) + 0.3) > 0.1,
        f"総合効果 -0.3：{{Z}} {coef(Zs):.3f}、空 {coef():.3f}、{{Z,M}} {coef(Zs, Ms):.3f}、{{Z,C}} {coef(Zs, Cs):.3f}")
E.choice("s16p09a", "$\\{Z\\}$", "正解の選択肢")
# (b) 測定誤差のあるデータを作って重回帰し、補正した値が真の β に戻ること
n9 = 1_000_000
Zc = rng.normal(size=n9)
Xt = np.sqrt(0.35) * Zc + rng.normal(0, 1.0, n9)          # Z で説明できない分散 1.0
Xo = Xt + rng.normal(0, 0.5, n9)                   # 誤差分散 0.25
Yc = 1.0 - 0.25 * Xt + 0.3 * Zc + rng.normal(0, 0.5, n9)
Dm = np.c_[np.ones(n9), Xo, Zc]
bh = np.linalg.lstsq(Dm, Yc, rcond=None)[0][1]
rres = Xo - np.c_[np.ones(n9), Zc] @ np.linalg.lstsq(np.c_[np.ones(n9), Zc], Xo, rcond=None)[0]
E.check("s16p09b", abs(bh + 0.20) < 0.003 and abs(rres.var() - 1.25) < 0.01 and abs(Xo.var() - 1.60) < 0.02,
        f"作ったデータ：係数 {bh:.4f}、残差分散 {rres.var():.3f}、全分散 {Xo.var():.3f}（問題文の値）")
E.check("s16p09b", abs(bh * rres.var() / (rres.var() - 0.25) + 0.25) < 0.005, "補正すると真の β=-0.25 に戻る")
E.choice("s16p09b", f"${f(-0.20 / ((1.25 - 0.25) / 1.25), 2)}$", "信頼性比で割る")

# ---- 大問10
qc = stats.studentized_range.isf(0.05, 4, 20)
E.check("s16p10a", abs(qc - 3.96) < 0.005 and abs(stats.t.isf(0.025, 20) - 2.086) < 5e-4, f"q(4,20)={qc:.4f}、t(20)")
mm = {"A": 12.0, "B": 14.6, "C": 15.2, "D": 18.2}
hsd = 3.96 * np.sqrt(4.20 / 6)
sig = [(i, j) for i, j in combinations("ABCD", 2) if abs(mm[i] - mm[j]) > hsd]
E.choice("s16p10a", f"HSD$={f(hsd, 2)}$、有意な対は{len(sig)}組", f"有意な対 {sig}")
Ysim = rng.normal(0, np.sqrt(4.2), size=(100_000, 4, 6))
mse_ = Ysim.var(axis=2, ddof=1).mean(1)
rngm = np.ptp(Ysim.mean(2), axis=1)
E.check("s16p10a", abs((rngm > 3.96 * np.sqrt(mse_ / 6)).mean() - 0.05) < 0.003, f"族全体の誤り率 {(rngm > 3.96 * np.sqrt(mse_ / 6)).mean():.4f}")

# ---- 大問11
# (a) 母集団を作り、比推定量と標本平均の分散をシミュレーションで比べる（近似式の確認）
Np = 800
xp = rng.gamma(6.25, 1.6, Np)
yp = 1.5 * xp + rng.normal(0, 1.0, Np) * np.sqrt(xp) * 0.9
Sy2, Sx2, Sxy = yp.var(ddof=1), xp.var(ddof=1), np.cov(xp, yp)[0, 1]
Rp = yp.mean() / xp.mean()
approx = (Sy2 - 2 * Rp * Sxy + Rp ** 2 * Sx2) / Sy2
idx = np.argsort(rng.random((40_000, Np)), axis=1)[:, :50]
ys, xs_ = yp[idx], xp[idx]
yr = ys.mean(1) / xs_.mean(1) * xp.mean()
E.check("s16p11a", abs(yr.var() / ys.mean(1).var() - approx) < 0.03, f"シミュレーションの分散比 {yr.var() / ys.mean(1).var():.4f} と近似式 {approx:.4f}")
E.num("s16p11a", rnd(np.array([1, -1.5]) @ np.array([[25.0, 18.0], [18.0, 16.0]]) @ np.array([1, -1.5]) / 25.0, 2), "V[y-R x] を分散共分散行列の二次形式で")
# (b)
pt = stats.contingency.relative_risk(32, 400, 56, 400)
E.check("s16p11b", abs(pt.relative_risk - 0.5714) < 1e-3, f"scipy relative_risk {pt.relative_risk:.4f}")
E.choice("s16p11b", f"{int(np.ceil(1 / (56 / 400 - 32 / 400)))}台", "1/(リスク差) を切り上げ")
sim0 = rng.binomial(1, 0.14, (2000, 17)).sum(1)
sim1 = rng.binomial(1, 0.08, (2000, 17)).sum(1)
E.check("s16p11b", abs((sim0 - sim1).mean() - 1.02) < 0.05, f"17 台あたり減る早期摩耗の平均 {(sim0 - sim1).mean():.3f} ≈ 1")

# ---- 大問12
# (a) 負荷量と整合する相関行列を作り、すべての主成分の和で厳密な相関を、上位 2 つで近似を確認
Lm = np.array([[0.90, 0.30], [0.80, -0.40], [0.20, 0.95]])
E.choice("s16p12a", f"${f((Lm[0] * Lm[1]).sum(), 2)}$", "負荷量ベクトルの内積")
Rm = Lm @ Lm.T
np.fill_diagonal(Rm, 1.0)
ev_, V_ = np.linalg.eigh(Rm)
E.check("s16p12a", ev_.min() > 0, f"負荷量と矛盾しない相関行列が作れる（固有値 {np.round(ev_, 3)}）")
Lfull = V_ * np.sqrt(ev_)
E.check("s16p12a", abs((Lfull[0] * Lfull[1]).sum() - Rm[0, 1]) < 1e-12, "全主成分の負荷量の内積は相関係数に一致")
# (b) 実際の点を作り、併合後の重心から C の重心までの距離の 2 乗を直接計算
A_ = rng.normal(size=(2, 2)); A_ += np.array([0, 0]) - A_.mean(0)
B_ = rng.normal(size=(3, 2)); B_ += np.array([2, 0]) - B_.mean(0)
C_ = np.array([[3.0, 3.0]])
d2 = lambda u_, v_: float(((u_.mean(0) - v_.mean(0)) ** 2).sum())
E.check("s16p12b", [round(d2(A_, B_), 9), round(d2(A_, C_), 9), round(d2(B_, C_), 9)] == [4, 18, 10], "作った点の重心間距離が表と一致")
E.choice("s16p12b", f"${f(d2(np.vstack([A_, B_]), C_), 2)}$", "併合後の重心から直接計算")

# ---- 大問13：ARMA(1,1) の理論自己相関を φ・θ の連立方程式として数値で解き、シミュレーションで標本自己相関の比を確かめる
def arma_acf(phi_, th_):
    g0 = (1 + 2 * phi_ * th_ + th_ ** 2) / (1 - phi_ ** 2)
    g1 = (1 + phi_ * th_) * (phi_ + th_) / (1 - phi_ ** 2)
    return g1 / g0, phi_ * g1 / g0
sol = optimize.fsolve(lambda p: np.array(arma_acf(*p)) - [0.55, 0.40], [0.5, 0.0])
E.num("s16p13a", rnd(sol[0], 3), f"ARMA(1,1) の理論自己相関の連立方程式の解 φ={sol[0]:.5f}, θ={sol[1]:.5f}")
E.check("s16p13a", abs(sol[1]) < 1 and abs(sol[1] - (-0.260)) < 5e-4, "θ は反転可能で解説の値と一致")
e_ = rng.normal(size=(2000, 2300))
yy = np.zeros_like(e_)
for t_ in range(1, e_.shape[1]):
    yy[:, t_] = sol[0] * yy[:, t_ - 1] + e_[:, t_] + sol[1] * e_[:, t_ - 1]
yy = yy[:, 300:] - yy[:, 300:].mean(1, keepdims=True)
ac = lambda k: (yy[:, k:] * yy[:, :-k]).sum() / (yy ** 2).sum()
E.check("s16p13a", abs(ac(1) - 0.55) < 0.01 and abs(ac(3) / ac(2) - sol[0]) < 0.02, f"シミュレーション r1={ac(1):.4f}, r3/r2={ac(3) / ac(2):.4f}")

# ---- 大問14
u_ = sp.symbols("u")
var = sp.integrate(u_ ** 4, (u_, 0, 1)) - sp.Rational(1, 9)
cv = sp.integrate(u_ ** 2 * (1 - u_) ** 2, (u_, 0, 1)) - sp.Rational(1, 9)
ratio = ((var + cv) / 2) / (var / 2)
E.choice("s16p14a", f"${f(float(ratio), 3)}$", f"sympy の積分で比 {ratio}")
E.check("s16p14a", sp.integrate(u_ ** 2 * (1 - u_) ** 2, (u_, 0, 1)) == sp.Rational(1, 30), "問題文の E[U^2(1-U)^2]=1/30")
Us = rng.random((200_000, 50))
th1 = (rng.random((200_000, 100)) ** 2).mean(1)
th2 = ((Us ** 2 + (1 - Us) ** 2) / 2).mean(1)
E.check("s16p14a", abs(th2.var() / th1.var() - 0.125) < 0.005, f"シミュレーションの分散比 {th2.var() / th1.var():.4f}")

E.done()
