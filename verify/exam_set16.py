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


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def nearest(iid, value):
    """選択肢の最初の数値のうち value に最も近いものが正解か。"""
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1
# (a) 着氷の有無と 3 つのセンサーの状態 2×2^3 通りを総当たり
c, q, p = 0.05, 0.4, 0.02
tot = 0.0
for ice in (0, 1):
    pi = q if ice else p
    for st in product((0, 1), repeat=3):  # 1 = 故障
        pr = (c if ice else 1 - c) * np.prod([pi if s_ else 1 - pi for s_ in st])
        if sum(st) >= 2:
            tot += pr
E.num("s16p01a", rnd(tot, 4), "着氷と3センサーの全状態を総当たり")
sim_ice = rng.random(2_000_000) < c
fail = rng.random((2_000_000, 3)) < np.where(sim_ice, q, p)[:, None]
E.check("s16p01a", abs((fail.sum(1) >= 2).mean() - tot) < 0.0005, f"シミュレーション {(fail.sum(1) >= 2).mean():.5f}")
# (b) ROC 曲線を実際に描いて台形の面積
fs = np.array([0.82, 0.64, 0.55, 0.91])
ns = np.array([0.31, 0.58, 0.47, 0.64, 0.22])
ths = np.r_[np.inf, np.sort(np.unique(np.r_[fs, ns]))[::-1], -np.inf]
tpr = [(fs >= t).mean() for t in ths]
fpr = [(ns >= t).mean() for t in ths]
auc = np.trapezoid(tpr, fpr)
E.num("s16p01b", rnd(auc, 3), "ROC 曲線の台形の面積")
u = stats.mannwhitneyu(fs, ns, alternative="two-sided").statistic
E.check("s16p01b", abs(u / 20 - auc) < 1e-12, f"マン・ホイットニーの U/(mn)={u / 20}")

# ---- 大問2
lam = 1 / 500
E.choice("s16p02a", f"${f(stats.gamma.sf(1000, 3, scale=500), 3)}$", "scipy gamma(3, scale=500).sf(1000)")
T3 = rng.exponential(500, size=(1_000_000, 3)).sum(1)
E.check("s16p02a", abs((T3 > 1000).mean() - 0.6767) < 0.002, f"シミュレーション {(T3 > 1000).mean():.4f}")
E.check("s16p02a", abs(np.exp(-2) - 0.1353) < 5e-5, "e^{-2} の値")
nearest("s16p02a", 0.1353 * 5)
joint = lambda t1, t2: lam ** 2 * np.exp(-lam * t2) if 0 < t1 < t2 else 0.0
num = integrate.quad(lambda t1: joint(t1, 800), 0, 200)[0]
den = integrate.quad(lambda t1: joint(t1, 800), 0, 800)[0]
E.choice("s16p02b", f"${f(num / den, 3)}$", "同時密度を数値積分して条件付き確率")
# シミュレーション：T2 が 800 付近の標本で T1<=200 の割合
X = rng.exponential(500, size=(4_000_000, 2))
T1, T2 = X[:, 0], X.sum(1)
sel = np.abs(T2 - 800) < 5
E.check("s16p02b", abs((T1[sel] <= 200).mean() - 0.25) < 0.01, f"シミュレーション {(T1[sel] <= 200).mean():.4f}（{sel.sum()} 件）")
E.check("s16p02b", all(abs(np.exp(-x) - v) < 5e-5 for x, v in [(0.4, 0.6703), (1.2, 0.3012), (1.6, 0.2019)]), "e^x の値")

# ---- 大問3
a, b, cc = 30, 20, 40
fS = lambda s_: integrate.quad(lambda x: stats.uniform(-a, 2 * a).pdf(s_ - x) * stats.uniform(-b, 2 * b).pdf(x), -b, b, points=[s_ - a, s_ + a])[0]
tail = integrate.quad(fS, cc, a + b)[0]
E.num("s16p03a", rnd(2 * tail, 4), "畳み込みの密度を数値積分")
Ssim = rng.uniform(-a, a, 4_000_000) + rng.uniform(-b, b, 4_000_000)
E.check("s16p03a", abs((np.abs(Ssim) > cc).mean() - 1 / 24) < 0.0005, f"シミュレーション {(np.abs(Ssim) > cc).mean():.5f}")
seq = E.item("s16p03b")["question"]
seq = re.search(r"`([+-]+)`", seq).group(1)
n1, n2 = seq.count("+"), seq.count("-")
R = len([k for k, _ in groupby(seq)])
N = n1 + n2
# 帰無仮説のもとの連の数の分布を全配置の総当たりで求める
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
x = np.array([0, 3, 1, 6, 2, 0, 4, 1, 7, 6])
D = ((x - x.mean()) ** 2).sum() / x.mean()
E.choice("s16p04a", f"$D={f(D, 2)}$で、自由度9の上側5%点$16.92$を超える", "numpy で分散指数")
E.check("s16p04a", abs(stats.chi2.isf(0.05, 9) - 16.92) < 0.005 and abs(stats.chi2.ppf(0.05, 9) - 3.33) < 0.005
        and abs(stats.chi2.isf(0.05, 10) - 18.31) < 0.005, "問題文のカイ二乗の点")
# 帰無仮説（ポアソン）のもとでの D の分布をシミュレーションして上側確率
Ps = rng.poisson(3, size=(200_000, 10))
Ds = ((Ps - Ps.mean(1, keepdims=True)) ** 2).sum(1) / np.maximum(Ps.mean(1), 1e-12)
E.check("s16p04a", (Ds >= D).mean() < 0.05, f"シミュレーションの p 値 {(Ds >= D).mean():.4f} < 0.05")
pod = stats.norm.cdf(-1.572 + 2.0 * np.log(4))
E.choice("s16p04b", f"${f(pod, 3)}$", "scipy norm.cdf(β0+β1 ln 4)")
E.check("s16p04b", abs(stats.norm.cdf(1.2) - 0.8849) < 5e-5 and abs(stats.norm.cdf(0.37) - 0.6443) < 5e-5
        and abs(np.exp(-1.2) - 0.3012) < 5e-5 and abs(np.log(4) - 1.386) < 5e-4 and abs(np.log10(4) - 0.602) < 5e-4, "表の値")
qd = 0.8849
ns_ = np.arange(2, 80)
w = stats.poisson.pmf(ns_, 2.5) * stats.binom.pmf(2, ns_, qd)
E.num("s16p04c", rnd((ns_ * w).sum() / w.sum(), 2), "ベイズの定理で N の条件付き分布を作って期待値")
Nsim = rng.poisson(2.5, 3_000_000)
Ksim = rng.binomial(Nsim, qd)
E.check("s16p04c", abs(Nsim[Ksim == 2].mean() - 2.28775) < 0.005, f"シミュレーション {Nsim[Ksim == 2].mean():.4f}")

# ---- 大問5
t = np.array([0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1])
nll = lambda th: -stats.weibull_min(2, scale=np.sqrt(th)).logpdf(t).sum()
th_hat = optimize.minimize_scalar(nll, bounds=(0.1, 10), method="bounded", options={"xatol": 1e-10}).x
E.choice("s16p05a", f"${f(th_hat, 2)}$", "scipy の weibull_min（尺度 √θ）の対数尤度を数値最大化")
E.check("s16p05a", abs(stats.weibull_min(2, scale=np.sqrt(2.0)).pdf(1.3) - 2 * 1.3 / 2.0 * np.exp(-1.3 ** 2 / 2.0)) < 1e-12,
        "問題文の密度は scipy の形状2・尺度√θ のワイブル分布と一致")
th, tt, nn = sp.symbols("theta t n", positive=True)
logf = sp.log(2 * tt / th) - tt ** 2 / th
info = sp.simplify(-sp.diff(logf, th, 2).subs(tt ** 2, th))  # E[T^2]=θ
gR = sp.exp(-1 / th)
seR = sp.Abs(sp.diff(gR, th)) * sp.sqrt(1 / (8 * info))
E.check("s16p05b", sp.simplify(info - 1 / th ** 2) == 0, f"sympy で 1 個のフィッシャー情報量 {info}")
E.num("s16p05b", rnd(float(seR.subs(th, 2)), 3), "sympy でデルタ法")
# シミュレーション（θ=2, n=8）で R̂ の標準偏差（小標本なので大まかに比べる）
sims = (rng.weibull(2, size=(200_000, 8)) * np.sqrt(2.0)) ** 2
Rh = np.exp(-1 / sims.mean(1))
E.check("s16p05b", abs(Rh.std() - 0.107) < 0.02, f"シミュレーションの標準偏差 {Rh.std():.4f}（漸近近似と大まかに一致）")
E.check("s16p05b", abs(np.exp(-0.5) - 0.6065) < 5e-5, "e^{-0.5}")

# ---- 大問6
tval = (41.0 - 40.0) / (1.6 / np.sqrt(10))
E.choice("s16p06a", f"$t={f(tval, 2)}$で、自由度9の上側5%点1.833を超える", "t 統計量")
E.check("s16p06a", stats.t.sf(tval, 9) < 0.05 < 2 * stats.t.sf(tval, 9), "片側 p<0.05、両側 p>0.05")
E.check("s16p06a", abs(stats.t.isf(0.05, 9) - 1.833) < 5e-4 and abs(stats.t.isf(0.025, 9) - 2.262) < 5e-4
        and abs(stats.t.isf(0.05, 10) - 1.812) < 5e-4, "t 分布の点")
ss = 9 * 1.6 ** 2
lo, hi = np.sqrt(ss / stats.chi2.isf(0.025, 9)), np.sqrt(ss / stats.chi2.ppf(0.025, 9))
E.choice("s16p06b", f"$({f(lo, 2)},\\ {f(hi, 2)})$", "scipy の正確なカイ二乗点で区間")
E.check("s16p06b", all(abs(v - w_) < 0.005 for v, w_ in [(stats.chi2.ppf(0.025, 9), 2.70), (stats.chi2.isf(0.025, 9), 19.02),
        (stats.chi2.ppf(0.05, 9), 3.33), (stats.chi2.isf(0.05, 9), 16.92), (stats.chi2.ppf(0.025, 10), 3.25), (stats.chi2.isf(0.025, 10), 20.48)]),
        "表のカイ二乗の点")
S = rng.normal(0, 1.6, size=(200_000, 10)).std(axis=1, ddof=1)
cover = (np.sqrt(9 * S ** 2 / 19.02) <= 1.6) & (1.6 <= np.sqrt(9 * S ** 2 / 2.70))
E.check("s16p06b", abs(cover.mean() - 0.95) < 0.003, f"被覆率 {cover.mean():.4f}")

# ---- 大問7：出生死滅過程の定常分布を数値で（状態を 3000 で打ち切る）
lam7, mu7, K = 3.0, 2.0, 3000
w7 = np.ones(K)
for n in range(1, K):
    w7[n] = w7[n - 1] * lam7 / (min(n, 2) * mu7)
pn = w7 / w7.sum()
Lq = sum(max(n - 2, 0) * pn[n] for n in range(K))
E.num("s16p07a", rnd(Lq / lam7, 2), "出生死滅過程の定常分布から L_q/λ")
# 事象駆動のシミュレーション（先着順・2 窓口）
arr = np.cumsum(rng.exponential(1 / lam7, 400_000))
svc = rng.exponential(1 / mu7, 400_000)
free = [0.0, 0.0]
waits = np.empty(len(arr))
for i, (ta, sv) in enumerate(zip(arr, svc)):
    k = 0 if free[0] <= free[1] else 1
    st = max(ta, free[k])
    waits[i] = st - ta
    free[k] = st + sv
E.check("s16p07a", abs(waits[20_000:].mean() - 9 / 14) < 0.03, f"シミュレーションの平均待ち {waits[20_000:].mean():.4f}")

# ---- 大問8
# (a) 決定係数の形（任意の総平方和）で部分 F
sst = 1000.0
R1, R2 = 1 - 520 / sst, 1 - 430 / sst
F = ((R2 - R1) / 2) / ((1 - R2) / 35)
E.choice("s16p08a", f"$F={f(F, 2)}$で、$F_{{0.05}}(2,35)=3.27$を超える", "決定係数の差による部分 F")
E.check("s16p08a", abs(stats.f.isf(0.05, 2, 35) - 3.27) < 0.005 and abs(stats.f.isf(0.05, 2, 37) - 3.25) < 0.005
        and stats.f.sf(F, 2, 35) < 0.05, "F 分布の点と p 値")
# (b) 実際のデータで「1 台を除いて当てはめ直す」計算を行い、公式 t=r√((n-p-2)/(n-p-1-r²)) を確かめてから適用
Xd = np.c_[np.ones(40), rng.normal(size=(40, 4))]
yd = Xd @ np.array([1, 2, -1, 0.5, 0.3]) + rng.normal(size=40)
yd[16] += 4.0
H = Xd @ np.linalg.solve(Xd.T @ Xd, Xd.T)
e = yd - H @ yd
s2 = e @ e / 35
ri = e[16] / np.sqrt(s2 * (1 - H[16, 16]))
mask = np.arange(40) != 16
bd = np.linalg.lstsq(Xd[mask], yd[mask], rcond=None)[0]
ed = yd[mask] - Xd[mask] @ bd
ti = e[16] / np.sqrt(ed @ ed / 34 * (1 - H[16, 16]))
E.check("s16p08b", abs(ti - ri * np.sqrt(34 / (35 - ri ** 2))) < 1e-9, f"実データで除いて当てはめ直した t={ti:.4f} と公式が一致")
E.choice("s16p08b", f"${f(2.80 * np.sqrt(34 / (35 - 2.80 ** 2)), 2)}$", "確かめた公式に r=2.80 を代入")

# ---- 大問9
# (a) d 分離をモラル化で判定してバックドア基準を確かめる
edges = [("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y"), ("X", "C"), ("Y", "C")]


def ancestors(nodes, es):
    out = set(nodes)
    ch = True
    while ch:
        ch = False
        for u_, v_ in es:
            if v_ in out and u_ not in out:
                out.add(u_)
                ch = True
    return out


def dsep(xs, ys, zs, es):
    an = ancestors(set(xs) | set(ys) | set(zs), es)
    sub = [(u_, v_) for u_, v_ in es if u_ in an and v_ in an]
    und = {frozenset(e_) for e_ in sub}
    for v_ in an:
        pa = [u_ for u_, w_ in sub if w_ == v_]
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                und.add(frozenset((pa[i], pa[j])))
    seen, stack = set(xs), list(xs)
    while stack:
        u_ = stack.pop()
        for e_ in und:
            if u_ in e_:
                (v_,) = e_ - {u_}
                if v_ not in seen and v_ not in zs:
                    seen.add(v_)
                    stack.append(v_)
    return not (seen & set(ys))


def descendants(v, es):
    out, stack = set(), [v]
    while stack:
        u_ = stack.pop()
        for a_, b_ in es:
            if a_ == u_ and b_ not in out:
                out.add(b_)
                stack.append(b_)
    return out


def backdoor(zs):
    if set(zs) & descendants("X", edges):
        return False
    es = [(u_, v_) for u_, v_ in edges if u_ != "X"]
    return dsep({"X"}, {"Y"}, set(zs), es)


cands = [set(), {"Z", "M"}, {"Z", "C"}, {"M", "C"}, {"Z"}]
ok = [backdoor(z_) for z_ in cands]
E.check("s16p09a", ok == [False, False, False, False, True], f"各選択肢のバックドア基準の判定 {ok}")
# (b) 交絡 U のある個人データを作り、W を操作変数とした 2 段階最小二乗
n9 = 400_000
W = rng.integers(0, 2, n9)
U = rng.normal(size=n9)
Xi = 2.4 + 0.8 * W + 0.7 * U + rng.normal(size=n9)
Yi = 1.9 - 0.25 * Xi + 0.4 * U + rng.normal(size=n9) * 0.3
Z1 = np.c_[np.ones(n9), W]
xhat = Z1 @ np.linalg.lstsq(Z1, Xi, rcond=None)[0]
b2 = np.linalg.lstsq(np.c_[np.ones(n9), xhat], Yi, rcond=None)[0][1]
bols = np.polyfit(Xi, Yi, 1)[0]
E.check("s16p09b", abs(b2 + 0.25) < 0.01 and bols > 0, f"シミュレーション：2SLS {b2:.4f}（真値 -0.25）、OLS {bols:.3f} は正に偏る")
E.check("s16p09b", abs(Xi[W == 1].mean() - 3.2) < 0.01 and abs(Yi[W == 1].mean() - 1.10) < 0.01 and abs(Yi[W == 0].mean() - 1.30) < 0.01,
        "作ったデータの群平均が問題文の値に一致")
E.choice("s16p09b", f"${f((1.10 - 1.30) / (3.2 - 2.4), 2)}$", "群平均からワルド推定量")

# ---- 大問10
qc = stats.studentized_range.isf(0.05, 4, 20)
E.check("s16p10a", abs(qc - 3.96) < 0.005 and abs(stats.t.isf(0.025, 20) - 2.086) < 5e-4, f"q(4,20)={qc:.4f}、t(20)")
mm = {"A": 12.0, "B": 14.6, "C": 15.2, "D": 18.2}
hsd = 3.96 * np.sqrt(4.20 / 6)
sig = [(i, j) for i, j in combinations("ABCD", 2) if abs(mm[i] - mm[j]) > hsd]
E.choice("s16p10a", f"HSD$={f(hsd, 2)}$、有意な対は{len(sig)}組", f"有意な対 {sig}")
# 全平均が等しい場合のシミュレーションで、HSD の族全体の誤り率が 5% になることを確かめる
Ysim = rng.normal(0, np.sqrt(4.2), size=(100_000, 4, 6))
mse = Ysim.var(axis=2, ddof=1).mean(1)
rngm = np.ptp(Ysim.mean(2), axis=1)
E.check("s16p10a", abs((rngm > 3.96 * np.sqrt(mse / 6)).mean() - 0.05) < 0.003, f"族全体の誤り率 {(rngm > 3.96 * np.sqrt(mse / 6)).mean():.4f}")

# ---- 大問11
nreq = next(n for n in range(1, 801) if 1.96 * np.sqrt((800 - n) / 799 * 0.25 / n) <= 0.05)
E.num("s16p11a", nreq, "n=1,2,… と総当たり")
tabs = [np.array([[10, 90], [20, 80]]), np.array([[30, 70], [45, 55]])]
ors = [t_[0, 0] * t_[1, 1] / (t_[0, 1] * t_[1, 0]) for t_ in tabs]
wts = [t_[0, 1] * t_[1, 0] / t_.sum() for t_ in tabs]
mh = sum(o * w_ for o, w_ in zip(ors, wts)) / sum(wts)
E.choice("s16p11b", f"${f(mh, 3)}$", "層別オッズ比を b_k c_k/n_k で重み付けた平均")
# 共通オッズ比の条件付き最尤推定（非心超幾何）とも近いことを確かめる
def cll(lpsi):
    psi = np.exp(lpsi)
    tot_ = 0.0
    for t_ in tabs:
        m1, n1_, k_ = t_[0].sum(), t_[1].sum(), t_[:, 0].sum()
        tot_ += stats.nchypergeom_fisher(m1 + n1_, m1, k_, psi).logpmf(t_[0, 0])
    return -tot_


cmle = np.exp(optimize.minimize_scalar(cll, bounds=(-3, 3), method="bounded").x)
nearest("s16p11b", cmle)

# ---- 大問12
ev = np.array([3.10, 1.40, 0.70, 0.40, 0.25, 0.15])
# 固有値を指定した相関行列を実際に作って主成分分析する（ランダム直交行列で回し、対角を 1 にする反復）
Q = np.linalg.qr(rng.normal(size=(6, 6)))[0]
Cm = Q @ np.diag(ev) @ Q.T
evc = np.sort(np.linalg.eigvalsh(Cm))[::-1]
cum = np.cumsum(evc) / evc.sum()
k1 = int((evc > 1).sum())
k2 = int(np.argmax(cum >= 0.8) + 1)
lost = 100 * (1 - cum[k2 - 1])
E.choice("s16p12a", f"(ア) {k1}　(イ) {k2}　(ウ) ${f(lost, 1)}\\%$", "固有値分解から基準を計算")
E.check("s16p12a", abs(np.trace(Cm) - 6) < 1e-12, "固有値の和が変数の数 6（相関行列の対角和）に一致")
# (b) 実際の点を作り、併合後の重心から C の重心までの距離の 2 乗を直接計算
A_ = rng.normal(size=(2, 2)); A_ += np.array([0, 0]) - A_.mean(0)
B_ = rng.normal(size=(3, 2)); B_ += np.array([2, 0]) - B_.mean(0)
C_ = np.array([[3.0, 3.0]])
d2 = lambda u_, v_: float(((u_.mean(0) - v_.mean(0)) ** 2).sum())
E.check("s16p12b", [round(d2(A_, B_), 9), round(d2(A_, C_), 9), round(d2(B_, C_), 9)] == [4, 18, 10], "作った点の重心間距離が表と一致")
E.choice("s16p12b", f"${f(d2(np.vstack([A_, B_]), C_), 2)}$", "併合後の重心から直接計算")

# ---- 大問13：ノイズを入れた将来の経路をシミュレーションして平均
nsim = 2_000_000
eps = rng.normal(0, 0.05, size=(nsim, 2))
d1 = 0.02 + 0.6 * 0.15 + eps[:, 0]
d2_ = 0.02 + 0.6 * d1 + eps[:, 1]
yT2 = 2.25 + d1 + d2_
E.check("s16p13a", abs(yT2.mean() - 2.446) < 0.0005, f"シミュレーションの平均 {yT2.mean():.5f}")
yv, dv = sp.symbols("y d")
d1s = sp.Rational(2, 100) + sp.Rational(6, 10) * (sp.Rational(225, 100) - sp.Rational(210, 100))
d2s = sp.Rational(2, 100) + sp.Rational(6, 10) * d1s
E.num("s16p13a", rnd(float(sp.Rational(225, 100) + d1s + d2s), 3), f"sympy の有理数で厳密に {sp.Rational(225, 100) + d1s + d2s}")

# ---- 大問14
u_ = sp.symbols("u")
var = sp.integrate(u_ ** 4, (u_, 0, 1)) - sp.Rational(1, 9)
cov = sp.integrate(u_ ** 2 * (1 - u_) ** 2, (u_, 0, 1)) - sp.Rational(1, 9)
ratio = ((var + cov) / 2) / (var / 2)
E.choice("s16p14a", f"${f(float(ratio), 3)}$", f"sympy の積分で比 {ratio}")
E.check("s16p14a", sp.integrate(u_ ** 2 * (1 - u_) ** 2, (u_, 0, 1)) == sp.Rational(1, 30), "問題文の E[U^2(1-U)^2]=1/30")
Us = rng.random((200_000, 50))
th1 = (rng.random((200_000, 100)) ** 2).mean(1)
th2 = ((Us ** 2 + (1 - Us) ** 2) / 2).mean(1)
E.check("s16p14a", abs(th2.var() / th1.var() - 0.125) < 0.005, f"シミュレーションの分散比 {th2.var() / th1.var():.4f}")

E.done()
