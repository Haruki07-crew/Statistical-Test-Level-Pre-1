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
# (b) マハラノビス距離の比較で判別（線形判別と同値）し、h の値を numpy の逆行列で計算
mN, mF, S = np.array([1.0, 2.0]), np.array([3.0, 5.0]), np.array([[2.0, 1.0], [1.0, 3.0]])
x = np.array([4.2, 2.0])
Si = np.linalg.inv(S)
dN, dF = (x - mN) @ Si @ (x - mN), (x - mF) @ Si @ (x - mF)
w = Si @ (mF - mN)
h = w @ x - w @ (mN + mF) / 2
E.check("s16p01b", dF < dN and abs((dN - dF) / 2 - h) < 1e-12, f"マハラノビス距離² 正常 {dN:.3f}・故障 {dF:.3f}、(dN-dF)/2={h:.3f}")
E.choice("s16p01b", f"$\\boldsymbol w=({f(w[0], 1)},\\ {f(w[1], 1)})$、$h=+{f(h, 2)}$で「故障」", "numpy の逆行列で係数と h")
E.check("s16p01b", ((x - mN) ** 2).sum() < ((x - mF) ** 2).sum(), "ユークリッド距離では正常側（共分散を無視すると判定が変わる）")

# ---- 大問2
ls, lm = 0.0012, 0.0008
Ts = rng.exponential(1 / ls, (2_000_000, 2))
Tm = rng.exponential(1 / lm, (2_000_000, 2))
life = np.minimum(Ts, Tm).min(1)
E.choice("s16p02a", f"${f(stats.expon(scale=1 / (2 * (ls + lm))).sf(250), 3)}$", "scipy expon（率 2(λs+λm)）の生存関数")
E.check("s16p02a", abs((life > 250).mean() - np.exp(-1)) < 0.001, f"4 つの指数時間の最小値のシミュレーション {(life > 250).mean():.4f}")
E.check("s16p02a", table_ok([(np.exp(-1), 0.3679), (np.exp(-0.5), 0.6065), (np.exp(-0.25), 0.7788)], 5e-5), "e^x の値")
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
E.choice("s16p04a", f"${f(2 * (np.sqrt(Tt) - np.sqrt(20)), 2)}$", "2(√T-√20)")
Tsim = rng.poisson(20, 1_000_000)
E.check("s16p04a", abs(np.sqrt(Tsim).var() - 0.25) < 0.01, f"Po(20) の √T の分散 {np.sqrt(Tsim).var():.4f} ≈ 1/4")
E.check("s16p04a", stats.poisson(20).sf(Tt - 1) < 0.05, f"正確な片側 p 値 {stats.poisson(20).sf(Tt - 1):.4f} も 5% 未満")
a90 = np.exp((stats.norm.ppf(0.9) + 1.6) / 2.5)
E.choice("s16p04b", f"${f(a90, 2)}$ mm", "scipy の正確な Φ^{-1}(0.9) で逆算")
E.check("s16p04b", abs(stats.norm.cdf(-1.6 + 2.5 * np.log(3.167)) - 0.9) < 5e-4, "求めた長さで検出確率が 0.9")
E.check("s16p04b", table_ok([(np.exp(1.1528), 3.1670), (np.exp(1.5188), 4.5667), (np.exp(1.298), 3.6620), (np.exp(-0.1272), 0.8806),
                             (stats.norm.ppf(0.9), 1.282), (stats.norm.ppf(0.95), 1.645), (np.log(9), 2.197)], 5e-4) and abs(10 ** 1.1528 - 14.22) < 0.005, "表の値")
qd = 0.969
E.check("s16p04c", abs(stats.norm.cdf(-1.6 + 2.5 * 1.386) - qd) < 5e-4, "4 mm の検出確率 Φ(1.865)≈0.969")
ns_ = np.arange(2, 120)
wts = stats.poisson.pmf(ns_, 4.0) * stats.binom.pmf(2, ns_, qd)
E.num("s16p04c", rnd((ns_ * wts).sum() / wts.sum(), 2), "ベイズの定理で N の条件付き分布を作って期待値")
Nsim = rng.poisson(4.0, 3_000_000)
Ksim = rng.binomial(Nsim, qd)
E.check("s16p04c", abs(Nsim[Ksim == 2].mean() - 2.124) < 0.005, f"シミュレーション {Nsim[Ksim == 2].mean():.4f}")

# ---- 大問5
t = np.array([0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1])
nll = lambda th: -stats.weibull_min(2, scale=np.sqrt(th)).logpdf(t).sum()
th_hat = optimize.minimize_scalar(nll, bounds=(0.1, 10), method="bounded", options={"xatol": 1e-10}).x
E.choice("s16p05a", f"${f(th_hat, 2)}$", "scipy の weibull_min（尺度 √θ）の対数尤度を数値最大化")
E.check("s16p05a", abs(stats.weibull_min(2, scale=np.sqrt(2.0)).pdf(1.3) - 2 * 1.3 / 2.0 * np.exp(-1.3 ** 2 / 2.0)) < 1e-12,
        "問題文の密度は scipy の形状2・尺度√θ のワイブル分布と一致")
# (b) ρ/σ³ を数値積分（Y=T² の分布は T の密度から）、n を総当たり
th = 2.0
Ey = integrate.quad(lambda u: u ** 2 * stats.weibull_min(2, scale=np.sqrt(th)).pdf(u), 0, np.inf)[0]
Vy = integrate.quad(lambda u: (u ** 2 - Ey) ** 2 * stats.weibull_min(2, scale=np.sqrt(th)).pdf(u), 0, np.inf)[0]
g3 = lambda u: abs(u ** 2 - Ey) ** 3 * stats.weibull_min(2, scale=np.sqrt(th)).pdf(u)
rho = integrate.quad(g3, 0, np.sqrt(Ey))[0] + integrate.quad(g3, np.sqrt(Ey), np.inf)[0]
ratio = rho / Vy ** 1.5
E.check("s16p05b", abs(ratio - 2.415) < 5e-4 and abs(ratio - (12 / np.e - 2)) < 1e-6, f"数値積分で ρ/σ³={ratio:.6f}")
nb = next(n for n in range(1, 10_000) if 0.4748 * 2.415 / np.sqrt(n) <= 0.05)
E.num("s16p05b", nb, "n=1,2,… と総当たり")

# ---- 大問6
tval = (41.0 - 40.0) / (1.6 / np.sqrt(10))
E.choice("s16p06a", f"$t={f(tval, 2)}$で、自由度9の上側5%点1.833を超える", "t 統計量")
E.check("s16p06a", stats.t.sf(tval, 9) < 0.05 < 2 * stats.t.sf(tval, 9), "片側 p<0.05、両側 p>0.05")
E.check("s16p06a", table_ok([(stats.t.isf(0.05, 9), 1.833), (stats.t.isf(0.025, 9), 2.262), (stats.t.isf(0.05, 10), 1.812)], 5e-4), "t 分布の点")
U = 41.0 + stats.t.isf(0.05, 9) * 1.6 * np.sqrt(1.1)
E.choice("s16p06b", f"${f(U, 2)}$ m", "scipy の正確な t 点で予測上限")
Xs = rng.normal(0, 1, size=(300_000, 11))
xb, sd = Xs[:, :10].mean(1), Xs[:, :10].std(1, ddof=1)
cover = Xs[:, 10] <= xb + 1.833 * sd * np.sqrt(1.1)
E.check("s16p06b", abs(cover.mean() - 0.95) < 0.002, f"シミュレーションで次の 1 回が上限以下の割合 {cover.mean():.4f}")

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
# (b) QQ プロットの理論分位点と、右に歪んだ分布（指数分布）でのパターンをシミュレーションで確認
E.check("s16p08b", abs(stats.norm.ppf(39.5 / 40) - 2.24) < 0.005, f"Φ^(-1)(0.9875)={stats.norm.ppf(39.5 / 40):.4f}")
zs = np.sort((rng.exponential(1, (20_000, 40)) - 1), axis=1)
E.check("s16p08b", np.median(zs[:, -1]) > 2.24 and np.median(zs[:, 0]) > -2.24, "右に歪んだ分布では両端の点が直線より上")
E.choice("s16p08b", "右に裾が長い（右に歪んでいる）", "解釈の選択肢")

# ---- 大問9
# (a) 構造モデルからデータを発生させ、潜在結果で ATT を直接求める
n9 = 2_000_000
Z = rng.random(n9) < 0.4
X = rng.random(n9) < np.where(Z, 0.7, 0.3)
u = rng.random(n9)
Y1 = u < np.where(Z, 0.20, 0.08)
Y0 = u < np.where(Z, 0.35, 0.12)
att = (Y1[X].astype(float) - Y0[X]).mean()
E.check("s16p09a", abs(att + 0.107) < 0.002, f"潜在結果のシミュレーションで ATT {att:.4f}")
pz1x1 = 0.4 * 0.7 / (0.4 * 0.7 + 0.6 * 0.3)
E.choice("s16p09a", f"${f(-0.15 * pz1x1 - 0.04 * (1 - pz1x1), 3)}$", "ベイズの定理で P(Z|X=1) を出して重み付け")
# 調整集合 {Z} がバックドア基準を満たすことを d 分離で確認
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


E.check("s16p09a", not ({"Z"} & desc("X")) and dsep("X", "Y", {"Z"}, [e_ for e_ in edges if e_[0] != "X"])
        and not dsep("X", "Y", set(), [e_ for e_ in edges if e_[0] != "X"]), "{Z} はバックドア基準を満たし、空集合は満たさない")
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

# ---- 大問10：混合模型からデータを発生させ、A の効果がないとき V_A/V_AB と V_A/V_E の第 1 種の誤り率を比べる
a, b, rr = 4, 3, 2
nsim = 40_000
G = rng.normal(0, 1.5, (nsim, 1, b, 1))           # 装置の変量効果
AB = rng.normal(0, 1.5, (nsim, a, b, 1))          # 交互作用（変量）
Yd = G + AB + rng.normal(0, 1.0, (nsim, a, b, rr))
m_ij = Yd.mean(3)
m_i = Yd.mean((2, 3))
m_j = Yd.mean((1, 3))
m = Yd.mean((1, 2, 3))
VA = b * rr * ((m_i - m[:, None]) ** 2).sum(1) / (a - 1)
VAB = rr * ((m_ij - m_i[:, :, None] - m_j[:, None, :] + m[:, None, None]) ** 2).sum((1, 2)) / ((a - 1) * (b - 1))
VE = ((Yd - m_ij[..., None]) ** 2).sum((1, 2, 3)) / (a * b * (rr - 1))
rej_ok = (VA / VAB > stats.f.isf(0.05, 3, 6)).mean()
rej_bad = (VA / VE > stats.f.isf(0.05, 3, 12)).mean()
E.check("s16p10a", abs(rej_ok - 0.05) < 0.005 and rej_bad > 0.2, f"第1種の誤り率：V_A/V_AB {rej_ok:.4f}、V_A/V_E {rej_bad:.3f}")
E.check("s16p10a", table_ok([(stats.f.isf(0.05, 3, 6), 4.76), (stats.f.isf(0.05, 3, 12), 3.49), (stats.f.isf(0.05, 3, 2), 19.16), (stats.f.isf(0.05, 6, 12), 3.00)], 0.005), "F の点")
E.choice("s16p10a", f"$F={f(48 / 12, 2)}$で$F_{{0.05}}(3,6)=4.76$を超えない", "V_A/V_AB")

# ---- 大問11
y = np.array([1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6, 2.8, 3.1, 3.3, 3.6, 3.9])
ests = [y[st::3].mean() for st in range(3)]
E.num("s16p11a", rnd(np.var(ests) if abs(np.mean(ests) - y.mean()) < 1e-12 else -1, 4), "3 通りの標本を numpy で作り分散（母平均と一致を確認）")
srs = np.array([y[list(cmb)].mean() for cmb in combinations(range(12), 4)])
E.check("s16p11a", abs(srs.var() - 0.1468) < 5e-4 and np.var(ests) < srs.var(), f"単純無作為抽出の全 495 通りの分散 {srs.var():.4f} より小さい")
ORs = 30 * 110 / (70 * 90)
E.choice("s16p11b", f"オッズ比は約{f(ORs, 2)}である。相対危険はこの表から直接は推定できない", "交差比")
# コホートの母集団を作り、症例と対照を抽出したときのオッズ比が母集団のオッズ比に一致することを確かめる
expo = rng.random(4_000_000) < 0.3
dis = rng.random(4_000_000) < np.where(expo, 0.01, 0.02)
case = np.flatnonzero(dis)
ctrl = rng.choice(np.flatnonzero(~dis), 2 * len(case), replace=False)
a_, b_ = expo[case].sum(), (~expo[case]).sum()
c_, d_ = expo[ctrl].sum(), (~expo[ctrl]).sum()
or_pop = (0.01 / 0.99) / (0.02 / 0.98)
E.check("s16p11b", abs(a_ * d_ / (b_ * c_) - or_pop) < 0.03 and abs((a_ / (a_ + c_)) / (b_ / (b_ + d_)) - 0.5) > 0.05,
        f"抽出したオッズ比 {a_ * d_ / (b_ * c_):.3f} ≈ 母集団 {or_pop:.3f}、列の割合の比は相対危険 0.5 にならない")

# ---- 大問12
ev = np.array([3.10, 1.40, 0.70, 0.40, 0.25, 0.15])
Q = np.linalg.qr(rng.normal(size=(6, 6)))[0]
evc = np.sort(np.linalg.eigvalsh(Q @ np.diag(ev) @ Q.T))[::-1]
cum = np.cumsum(evc) / evc.sum()
k1, k2 = int((evc > 1).sum()), int(np.argmax(cum >= 0.8) + 1)
E.choice("s16p12a", f"(ア) {k1}　(イ) {k2}　(ウ) ${f(100 * (1 - cum[k2 - 1]), 1)}\\%$", "固有値分解から基準を計算")
pi1, pi2 = 0.7, 0.3
g = pi2 * stats.norm.pdf(3.8, 5, 1) / (pi1 * stats.norm.pdf(3.8, 2, 1) + pi2 * stats.norm.pdf(3.8, 5, 1))
E.choice("s16p12b", f"${f(g, 3)}$", "scipy の正規密度で負担率")
E.check("s16p12b", table_ok([(np.exp(-1.62), 0.1979), (np.exp(-0.72), 0.4868), (np.exp(-1.8), 0.1653), (np.exp(-1.2), 0.3012)], 5e-5), "e^x の値")
comp = rng.random(4_000_000) < pi2
xx = np.where(comp, rng.normal(5, 1, 4_000_000), rng.normal(2, 1, 4_000_000))
near_ = np.abs(xx - 3.8) < 0.01
E.check("s16p12b", abs(comp[near_].mean() - g) < 0.01, f"混合分布のシミュレーション {comp[near_].mean():.4f}")

# ---- 大問13
nsim = 2_000_000
eps = rng.normal(0, 0.05, size=(nsim, 2))
d1 = 0.02 + 0.6 * 0.15 + eps[:, 0]
d2_ = 0.02 + 0.6 * d1 + eps[:, 1]
E.check("s16p13a", abs((2.25 + d1 + d2_).mean() - 2.446) < 0.0005, f"シミュレーションの平均 {(2.25 + d1 + d2_).mean():.5f}")
d1s = sp.Rational(2, 100) + sp.Rational(6, 10) * (sp.Rational(225, 100) - sp.Rational(210, 100))
d2s = sp.Rational(2, 100) + sp.Rational(6, 10) * d1s
E.num("s16p13a", rnd(float(sp.Rational(225, 100) + d1s + d2s), 3), f"sympy の有理数で厳密に {sp.Rational(225, 100) + d1s + d2s}")

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
