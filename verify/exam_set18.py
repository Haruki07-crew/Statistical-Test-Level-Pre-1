"""模擬試験 第18回（set18）の検算。generator/set18.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・数値積分）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import itertools
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(18)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table_ok(iid, pairs, d, what):
    """問題文に示した値（丸め）が正確な値と一致するか。pairs = [(示した値, 正確な値)]"""
    bad = [(a, b) for a, b in pairs if f(b, d) != f(a, d)]
    E.check(iid, not bad, f"{what}（不一致 {bad}）")


def nearest(iid, value):
    it = E.item(iid)
    nums = [float(re.search(r"-?\d+\.?\d*", str(c).replace("−", "-")).group()) for c in it["choices"]]
    k = int(np.argmin([abs(x - value) for x in nums]))
    E.check(iid, k == it["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {it['answer']}")


# ---- 大問1 遺伝
nAA, nAa, naa = 50, 80, 70
n = 200
# (a) 多項尤度を数値最大化
negll = lambda p: -(nAA * math.log(p * p) + nAa * math.log(2 * p * (1 - p)) + naa * math.log((1 - p) ** 2))
p_hat = optimize.minimize_scalar(negll, bounds=(1e-6, 1 - 1e-6), method="bounded").x
E.choice("s18p01a", f"\\dfrac{{p}}{{1-p}}$、$T=2n_{{AA}}+n_{{Aa}}=180$、$\\hat p={f(p_hat, 2)}$", "多項尤度の数値最大化で p̂")
# 指数型分布族：対数尤度が T の 1 次式で、係数が log(p/(1-p)) になること（sympy）
pp, T_, nn_ = sp.symbols("p T n", positive=True)
loglik = T_ * sp.log(pp) + (2 * nn_ - T_) * sp.log(1 - pp)
E.check("s18p01a", abs(float(sp.diff(loglik, T_).subs(pp, 0.3)) - math.log(0.3 / 0.7)) < 1e-12, "T の係数が自然母数 log(p/(1-p))")
# 十分性：T が同じ別のデータで尤度比が p によらないこと
L = lambda a, b, c, p: (p * p) ** a * (2 * p * (1 - p)) ** b * ((1 - p) ** 2) ** c
r1 = [L(50, 80, 70, p) / L(60, 60, 80, p) for p in (0.2, 0.45, 0.7)]   # T=180 で同じ
E.check("s18p01a", max(r1) / min(r1) - 1 < 1e-9, "T=2nAA+nAa が等しいデータの尤度比が p によらない（十分性）")
r2 = [L(50, 80, 70, p) / L(70, 60, 70, p) for p in (0.2, 0.45, 0.7)]   # A をもつ個体数は同じ130だが T が違う
E.check("s18p01a", max(r2) / min(r2) - 1 > 1e-3, "A をもつ個体数が同じでも T が違えば尤度比は p による（個体数は十分でない）")
# (b) G 統計量を scipy で
obs = np.array([nAA, nAa, naa])
exp = n * np.array([p_hat ** 2, 2 * p_hat * (1 - p_hat), (1 - p_hat) ** 2])
G = 2 * np.sum(obs * np.log(obs / exp))
G2 = stats.power_divergence(obs, exp, ddof=1, lambda_="log-likelihood")
E.choice("s18p01b", f"G={f(G, 2)}$、自由度1で、${f(stats.chi2.isf(0.01, 1), 2)}$を超えるので棄却する", "正確な対数で G と自由度1の1%点")
E.check("s18p01b", abs(G2.statistic - G) < 1e-9 and G2.pvalue < 0.01, f"scipy power_divergence(ddof=1) p値 {G2.pvalue:.4f}")
table_ok("s18p01b", [(0.2107, math.log(50 / 40.5)), (-0.2131, math.log(80 / 99)), (0.1459, math.log(70 / 60.5))], 4, "示した対数の値")
table_ok("s18p01b", [(3.84, stats.chi2.isf(.05, 1)), (6.63, stats.chi2.isf(.01, 1)), (7.88, stats.chi2.isf(.005, 1)),
                     (5.99, stats.chi2.isf(.05, 2)), (9.21, stats.chi2.isf(.01, 2)), (10.60, stats.chi2.isf(.005, 2))], 2, "χ² 表")
# (c) EM：E ステップを条件付き確率の総当たりで、M ステップを完全データ尤度の数値最大化で
th0 = 0.5
probs = {"BB": th0 ** 2, "Bb": 2 * th0 * (1 - th0)}
eBB = 160 * probs["BB"] / sum(probs.values())
eBb = 160 * probs["Bb"] / sum(probs.values())
q = lambda t: -((2 * eBB + eBb) * math.log(t) + (eBb + 2 * 40) * math.log(1 - t))
th1 = optimize.minimize_scalar(q, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-10}).x
E.num("s18p01c", float(f(th1, 3)), "M ステップを数値最大化")
th = 0.5
for _ in range(500):
    a_ = 160 * th * th / (th * th + 2 * th * (1 - th))
    b_ = 160 - a_
    th = (2 * a_ + b_) / 400
E.check("s18p01c", abs(th - (1 - math.sqrt(0.2))) < 1e-6, f"反復の収束先 {th:.4f} = 1−√0.2（解説の主張）")

# ---- 大問2
# (a) ハプロタイプの同時確率を 0-1 の 2 変数として周辺化
H = np.array([[0.40, 0.20], [0.10, 0.30]])   # 行 A,a ／ 列 B,b
pA_, pB_ = H[0].sum(), H[:, 0].sum()
D_ = H[0, 0] - pA_ * pB_
E.choice("s18p02a", f"p_A={f(pA_, 2)}$、$p_B={f(pB_, 2)}$、$D={f(D_, 2)}$で、2つの遺伝子座は独立でない", "行和・列和から周辺確率と D")
E.check("s18p02a", abs(D_ - (H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])) < 1e-12 and abs(D_) > 0, "D=pAB·pab−pAb·paB と一致し 0 でない")
# (b) ワイブル（scipy weibull_min）の四分位範囲
w = stats.weibull_min(2, scale=40)
E.num("s18p02b", float(f(w.ppf(0.75) - w.ppf(0.25), 1)), "scipy weibull_min の ppf")
table_ok("s18p02b", [(0.2877, math.log(4 / 3)), (1.3863, math.log(4))], 4, "対数の値")

# ---- 大問3
# ---- 大問3
# (a) シミュレーション：Z~N(6,2²) から W=(1+0.5Z)²（Z>-2 の範囲で単調。P(Z<-2) は無視できる）
Z = rng.normal(6, 2, 1_000_000)
W = (1 + 0.5 * Z) ** 2
E.check("s18p03a", abs(np.median(W) - 16) < 0.05 and abs(np.mean(W > 25) - 0.1587) < 0.002,
        f"シミュレーション 中央値 {np.median(W):.3f}、P(W>25) {np.mean(W > 25):.4f}")
E.choice("s18p03a", f"{f(16, 1)}$g、$P(W>25)={f(stats.norm.sf(1), 3)}$", "scipy norm.sf(1)")
E.check("s18p03a", stats.norm.cdf(-4) < 1e-4, "1+0.5Z<0（変換の範囲外）となる確率は無視できる")
table_ok("s18p03a", [(0.6915, stats.norm.cdf(.5)), (0.8413, stats.norm.cdf(1)), (0.9772, stats.norm.cdf(2))], 4, "Φ の値")
# (b) 検出力：非心 t の正規近似（問題の指定）と、正確な非心 t
ncp = 2 * math.sqrt(16) / 4
pw = stats.norm.cdf(round(ncp - stats.t.isf(0.05, 15), 2))
E.choice("s18p03b", f"${f(pw, 3)}$", "scipy norm.cdf で正規近似")
nearest("s18p03b", stats.nct.sf(stats.t.isf(0.05, 15), 15, ncp))
table_ok("s18p03b", [(1.753, stats.t.isf(.05, 15)), (2.131, stats.t.isf(.025, 15)), (1.645, stats.norm.isf(.05))], 3, "t・z の点")
table_ok("s18p03b", [(0.5517, stats.norm.cdf(.13)), (0.5987, stats.norm.cdf(.25)), (0.6406, stats.norm.cdf(.36)), (0.8944, stats.norm.cdf(1.25))], 4, "Φ の値")
# ---- 大問4
# (a) ρ を定義から数値計算
p = 0.8
rho = sum(pr_ * abs(b - p) ** 3 for b, pr_ in ((1, p), (0, 1 - p)))
sig = math.sqrt(stats.bernoulli(p).var())
be = 0.4748 * rho / (sig ** 3 * math.sqrt(400))
E.choice("s18p04a", f"${f(be, 3)}$", "定義から ρ と σ を計算")
dev = max(abs(stats.binom.cdf(k_, 400, p) - stats.norm.cdf((k_ - 320) / 8)) for k_ in range(250, 390))
E.check("s18p04a", dev <= be, f"実際の最大誤差 {dev:.4f} は上界以下")
# (b) 端点を正確に逆変換
se = 1 / math.sqrt(150 * .8 * .2)
lo, hi = math.log(4) - 1.96 * se, math.log(4) + 1.96 * se
expit = lambda z: 1 / (1 + math.exp(-z))
E.choice("s18p04b", f"({f(expit(lo), 3)},\\ {f(expit(hi), 3)})", "正確な値で逆変換")
table_ok("s18p04b", [(v, math.exp(-x)) for x, v in [(0.986, .3731), (1.051, .3496), (1.322, .2666), (1.450, .2346), (1.722, .1787), (1.786, .1676)]], 4, "e^{-x} の表")
table_ok("s18p04b", [(1.3863, math.log(4)), (0.2041, se)], 4, "log4 と標準誤差")

# ---- 大問5
# (a) 切断ポアソンの期待値を和で
pk = stats.poisson(1.5).pmf(np.arange(1, 60))
E.choice("s18p05a", f"${f(np.sum(np.arange(1, 60) * pk) / pk.sum(), 2)}$", "確率関数の和で E[X|X≥1]")
table_ok("s18p05a", [(0.2231, math.exp(-1.5))], 4, "e^{-1.5}")
# (b) 層別抽出：母集団を作って抽出を繰り返し、推定分散の式が真の分散と一致するか確かめ、式の値を独立に計算
N1, N2, n1, n2 = 200, 300, 20, 30
v = (0.4 ** 2) * (1 - n1 / N1) * 16 / n1 + (0.6 ** 2) * (1 - n2 / N2) * 10 / n2
E.num("s18p05b", float(f(math.sqrt(v), 2)), "層別推定量の分散の式（独立に計算）")
pop1 = rng.normal(12, 4, N1)
pop2 = rng.normal(5, math.sqrt(10), N2)
ests = [0.4 * rng.choice(pop1, n1, replace=False).mean() + 0.6 * rng.choice(pop2, n2, replace=False).mean() for _ in range(40000)]
vtrue = 0.16 * (1 - n1 / N1) * pop1.var(ddof=1) / n1 + 0.36 * (1 - n2 / N2) * pop2.var(ddof=1) / n2
E.check("s18p05b", abs(np.var(ests) / vtrue - 1) < 0.03, f"非復元抽出のシミュレーション分散 {np.var(ests):.4f} ≈ 式 {vtrue:.4f}")

# ---- 大問6
# ---- 大問6
tv = (3.24 - 3) / 0.10
E.check("s18p06a", stats.t.sf(tv, 23) * 2 < 0.05, f"両側 p 値 {2 * stats.t.sf(tv, 23):.4f}")
E.choice("s18p06a", "正の相対成長", "β>3 は正の相対成長")
E.check("s18p06b", E.item("s18p06b")["answer"] == 3 and "誤差の分散が一定でない" in E.item("s18p06b")["choices"][3], "扇形＝不均一分散（定義の確認）")
xs_ = rng.uniform(10, 30, 4000)
ys_ = 0.01 * xs_ ** 3 * np.exp(rng.normal(0, 0.15, 4000))
b_ = np.polyfit(xs_, ys_, 1)
res_ = ys_ - np.polyval(b_, xs_)
fit_ = np.polyval(b_, xs_)
lo_, hi_ = np.abs(res_[fit_ < np.quantile(fit_, .3)]).mean(), np.abs(res_[fit_ > np.quantile(fit_, .7)]).mean()
lres = np.log(ys_) - np.polyval(np.polyfit(np.log(xs_), np.log(ys_), 1), np.log(xs_))
E.check("s18p06b", hi_ > 1.5 * lo_ and abs(np.abs(lres[fit_ > np.quantile(fit_, .7)]).mean() / np.abs(lres[fit_ < np.quantile(fit_, .3)]).mean() - 1) < 0.2,
        f"体重∝体長³で乗法誤差のとき直線回帰の残差は扇形（{lo_:.2f}→{hi_:.2f}）、両対数では一様")

# ---- 大問7
# (a) ニュートン・ラフソンを sympy の微分で
bb = sp.symbols("b")
ll = 30 * bb - 100 * sp.log(1 + sp.exp(bb))
b1_ = float(0 - sp.diff(ll, bb).subs(bb, 0) / sp.diff(ll, bb, 2).subs(bb, 0))
E.choice("s18p07a", f"${f(b1_, 2)}$", "sympy で U/I")
E.check("s18p07a", abs(float(sp.nsolve(sp.diff(ll, bb), bb, -1)) - math.log(3 / 7)) < 1e-9, "収束先 = log(3/7)")
table_ok("s18p07a", [(-0.8473, math.log(3 / 7))], 4, "log(3/7)")
# (b) 正確な指数関数で順伝播
sg_ = lambda u: 1 / (1 + math.exp(-u))
h1_, h2_ = sg_(1.0 * 2 - 2.0), sg_(-1.0 * 2 + 1.0)
ph = sg_(2.0 * h1_ - 3.0 * h2_ + 0.5)
E.choice("s18p07b", f"${f(ph, 3)}$", "math.exp で順伝播")
table_ok("s18p07b", [(2.7183, math.e), (0.5001, math.exp(-.693)), (1.9997, math.exp(.693)), (0.6065, math.exp(-.5)), (0.0302, math.exp(-3.5))], 4, "指数の値")

# ---- 大問8
# (a) L8：回帰で A の係数と平方和
A_ = np.array([1, 1, 1, 1, 2, 2, 2, 2]); B_ = np.array([1, 1, 2, 2, 1, 1, 2, 2]); C_ = np.array([1, 2, 1, 2, 1, 2, 1, 2])
y8 = np.array([20, 22, 25, 27, 26, 30, 31, 35], float)
X8 = np.column_stack([np.ones(8), A_ == 2, B_ == 2, C_ == 2]).astype(float)
beta8 = np.linalg.lstsq(X8, y8, rcond=None)[0]
ssA = 8 * np.mean([(y8[A_ == k].mean() - y8.mean()) ** 2 for k in (1, 2)]) * 1.0
ssA = sum(4 * (y8[A_ == k].mean() - y8.mean()) ** 2 for k in (1, 2))
E.choice("s18p08a", f"主効果{f(beta8[1], 1)}、平方和{f(ssA, 1)}", "直交計画の回帰係数と水準平均からの平方和")
E.check("s18p08a", all(abs(np.dot(c1 - 1.5, c2 - 1.5)) < 1e-12 for c1, c2 in [(A_, B_), (A_, C_), (B_, C_)]), "第1・2・4列が直交")
# (b) 非心 F：scipy ncf の平均とシミュレーション
EF_ = stats.ncf(2, 21, 8 * 8 / 16).mean()
E.num("s18p08b", float(f(EF_, 2)), "scipy ncf(2,21,4).mean()")
sims = []
for _ in range(20000):
    g_ = [rng.normal(m_, 4, 8) for m_ in (-2, 0, 2)]
    gm = np.concatenate(g_).mean()
    msa = 8 * sum((x.mean() - gm) ** 2 for x in g_) / 2
    mse = sum(((x - x.mean()) ** 2).sum() for x in g_) / 21
    sims.append(msa / mse)
E.check("s18p08b", abs(np.mean(sims) - EF_) < 0.06, f"分散分析のシミュレーションの F の平均 {np.mean(sims):.3f}")

# ---- 大問9 バックドア基準：networkx を使わず d 分離を総当たりの経路列挙で確かめる
edges = [("Z", "X"), ("W", "X"), ("W", "Y"), ("W", "A"), ("A", "Y"), ("X", "Y"), ("X", "M"), ("M", "Y"), ("X", "K"), ("Y", "K")]
nodes = {u for e in edges for u in e}
ch_ = {u: {b for a, b in edges if a == u} for u in nodes}
pa_ = {u: {a for a, b in edges if b == u} for u in nodes}


def desc(u):
    out, st = set(), [u]
    while st:
        v = st.pop()
        for w in ch_[v]:
            if w not in out:
                out.add(w); st.append(w)
    return out


def paths(u, v, seen=()):
    if u == v:
        yield [u]; return
    for w in ch_[u] | pa_[u]:
        if w not in seen and w != u:
            for rest in paths(w, v, seen + (u,)):
                yield [u] + rest


def blocked(path, Zs):
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]
        collider = b in ch_[a] and b in ch_[c]
        if collider:
            if b not in Zs and not (desc(b) & Zs):
                return True
        elif b in Zs:
            return True
    return False


def backdoor(Zs):
    if desc("X") & Zs:
        return False
    return all(blocked(p, Zs) for p in paths("X", "Y") if p[1] in pa_["X"])


cands = [{"Z"}, {"A"}, {"W"}, {"W", "M"}, {"W", "K"}]
ok_ = [backdoor(c) for c in cands]
E.check("s18p09a", ok_ == [False, False, True, False, False], f"経路の総当たりで判定 {ok_}")
E.check("s18p09a", backdoor({"W", "Z"}) and backdoor({"W", "A"}), "解説の主張：{W,Z} と {W,A} も基準を満たす")

# ---- 大問10
# (a) 座標を作って重心間の平方距離を直接計算
Pp, Qp, Rp = np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 4.0])
dd = lambda a, b: float(((a - b) ** 2).sum())
E.check("s18p10a", (dd(Pp, Qp), dd(Pp, Rp), dd(Qp, Rp)) == (4.0, 25.0, 17.0), "表の平方距離を実現する配置がある")
E.choice("s18p10a", f"${f(dd(Rp, (Pp + Qp) / 2), 1)}$", "重心と R の平方距離")
# (b) KS
A = [11, 13, 14, 16, 18, 21]
B = [15, 17, 19, 20, 22, 24]
ks = stats.ks_2samp(A, B, method="exact")
E.choice("s18p10b", f"D=3/6$で、p値${f(ks.pvalue, 3)}$が0.05より大きいので棄却しない", "scipy ks_2samp(exact)")
cnt = {}
for comb in itertools.combinations(range(12), 6):
    fa = fb = dm = 0
    for i in range(12):
        if i in comb:
            fa += 1
        else:
            fb += 1
        dm = max(dm, abs(fa - fb))
    cnt[dm] = cnt.get(dm, 0) + 1
tot = sum(cnt.values())
pv_ = {kk: sum(v for d, v in cnt.items() if d >= kk) / tot for kk in (3, 4, 5)}
table_ok("s18p10b", [(0.474, pv_[3]), (0.143, pv_[4]), (0.026, pv_[5])], 3, "P(D≥k/6) を 924 通りの総当たりで")
# (c) ファイ係数を 0-1 データの相関で
xa = np.array([1] * 15 + [1] * 5 + [0] * 10 + [0] * 20)
ya = np.array([1] * 15 + [0] * 5 + [1] * 10 + [0] * 20)
E.num("s18p10c", float(f(np.corrcoef(xa, ya)[0, 1], 3)), "0-1 変数の相関係数（numpy corrcoef）")


# ---- 大問11
# (a) scipy の正規密度で 2 次判別
g_ = stats.norm.logpdf(34, 40, 2) - stats.norm.logpdf(34, 44, 4)
E.choice("s18p11a", f"g={f(g_, 2)}$で負なので種2", "scipy norm.logpdf")
E.check("s18p11a", abs(34 - 40) < abs(34 - 44), "平均の距離では種1に近い（解説の主張）")
table_ok("s18p11a", [(0.6931, math.log(2))], 4, "log2")
# (b) 因子得点：同時正規の条件付き期待値を行列で直接計算
lam_ = np.array([0.9, 0.8, 0.6]); Psi = np.diag(1 - lam_ ** 2)
Sig = np.outer(lam_, lam_) + Psi
fhat = lam_ @ np.linalg.solve(Sig, np.array([1.2, 0.8, -0.5]))
E.choice("s18p11b", f"${f(fhat, 2)}$", "λ'Σ^(-1)x を numpy で")
fs = rng.normal(size=400000)
Xs = np.outer(fs, lam_) + rng.normal(size=(400000, 3)) * np.sqrt(1 - lam_ ** 2)
coef = np.linalg.lstsq(Xs, fs, rcond=None)[0]
E.check("s18p11b", abs(coef @ np.array([1.2, 0.8, -0.5]) - fhat) < 0.01, "f を x に回帰したシミュレーション（回帰法の名前の由来）")

# ---- 大問12
# (a) ユール過程をシミュレーション（指数時間で出生）
cnt = 0
R_ = 100000
for _ in range(R_):
    nind, tt = 1, 0.0
    while True:
        tt += rng.exponential(1 / (0.5 * nind))
        if tt > 2:
            break
        nind += 1
    cnt += nind >= 4
E.check("s18p12a", abs(cnt / R_ - (1 - math.exp(-1)) ** 3) < 0.005, f"シミュレーション {cnt / R_:.4f}")
E.num("s18p12a", float(f(stats.geom(math.exp(-1)).sf(3), 3)), "scipy geom(e^-1).sf(3)")
table_ok("s18p12a", [(0.3679, math.exp(-1))], 4, "e^-1")
# (b) 正確な関数値で
zb = math.sqrt(-2 * math.log(0.3)) * math.cos(2 * math.pi * 0.2)
E.choice("s18p12b", f"${f(0.30 + 0.10 * zb, 3)}$", "math の log・cos で計算")
table_ok("s18p12b", [(-1.2040, math.log(.3)), (-1.6094, math.log(.2)), (0.3090, math.cos(.4 * math.pi)),
                     (0.9511, math.sin(.4 * math.pi)), (-0.3090, math.cos(.6 * math.pi))], 4, "log・cos・sin の値")
u1 = rng.random(500_000)
u2 = rng.random(500_000)
zs = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
E.check("s18p12b", abs(zs.mean()) < 0.01 and abs(zs.std() - 1) < 0.01, "ボックス・ミュラー法の出力が標準正規（平均0・標準偏差1）")

E.done()
