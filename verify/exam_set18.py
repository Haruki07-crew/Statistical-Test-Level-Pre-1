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

# ---- 大問3
# (a) シミュレーション：Z~N(6,2²) から W=(1+0.5Z)²（Z>-2 の範囲で単調。P(Z<-2) は無視できる）
Z = rng.normal(6, 2, 1_000_000)
W = (1 + 0.5 * Z) ** 2
E.check("s18p03a", abs(np.median(W) - 16) < 0.05 and abs(np.mean(W > 25) - 0.1587) < 0.002,
        f"シミュレーション 中央値 {np.median(W):.3f}、P(W>25) {np.mean(W > 25):.4f}")
E.choice("s18p03a", f"{f(16, 1)}$g、$P(W>25)={f(stats.norm.sf(1), 3)}$", "scipy norm.sf(1)")
E.check("s18p03a", stats.norm.cdf(-4) < 1e-4, "1+0.5Z<0（変換の範囲外）となる確率は無視できる")
table_ok("s18p03a", [(0.6915, stats.norm.cdf(.5)), (0.8413, stats.norm.cdf(1)), (0.9772, stats.norm.cdf(2))], 4, "Φ の値")
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

# ---- 大問11
# (a) 閾値ごとの感度・特異度を numpy の比較で数え、ユーデン指数を比べる
pres_ = np.array([2, 4, 5, 6, 7, 8, 9, 9]); absn_ = np.array([1, 1, 2, 3, 3, 4, 5, 7])
J_ = {c: np.mean(pres_ >= c) + np.mean(absn_ < c) - 1 for c in (3, 5, 7)}
E.check("s18p11a", max(J_, key=J_.get) == 5 and sorted(J_.values())[-1] > sorted(J_.values())[-2], f"ユーデン指数 {J_}")
E.choice("s18p11a", f"c=5$：感度{f(np.mean(pres_ >= 5), 3)}、特異度{f(np.mean(absn_ < 5), 3)}、$J={f(J_[5], 3)}$が最大", "numpy で数え上げ")
# (b) 負荷量行列 ΛΛ' の (殻長, 殻幅) 成分、シミュレーションでも
Lam = np.array([[0.8, 0.3], [0.7, 0.4], [0.3, 0.8], [0.6, 0.5]])
Rm = Lam @ Lam.T
E.choice("s18p11b", f"再現相関{f(Rm[0, 2], 2)}、残差{f(0.50 - Rm[0, 2], 2)}", "ΛΛ' を numpy で")
Fs_ = rng.normal(size=(400_000, 2))
Xs_ = Fs_ @ Lam.T + rng.normal(size=(400_000, 4)) * np.sqrt(1 - (Lam ** 2).sum(axis=1))
E.check("s18p11b", abs(np.corrcoef(Xs_[:, 0], Xs_[:, 2])[0, 1] - 0.48) < 0.005, f"因子モデルのシミュレーションで相関 {np.corrcoef(Xs_[:, 0], Xs_[:, 2])[0, 1]:.4f}")

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

# ---- 作り替えた小問（2026-10-24）
# 大問1 (b) 複合帰無仮説：scipy binom で検出力関数を格子上で計算し、帰無仮説の範囲での最大値を求める
qs = np.linspace(0, 0.2, 2001)
pow_ = stats.binom.sf(4, 10, qs)
E.check("s18p01b", np.all(np.diff(pow_) >= 0) and abs(pow_.max() - stats.binom.sf(4, 10, 0.2)) < 1e-15, "検出力関数は q について増加し、帰無仮説の範囲での最大は境界 q=0.2")
table_ok("s18p01b", [(0.0016, stats.binom.sf(4, 10, 0.1)), (0.0328, stats.binom.sf(4, 10, 0.2)), (0.6230, stats.binom.sf(4, 10, 0.5))], 4, "P(X≥5) の値")
E.choice("s18p01b", f"有意水準 {f(pow_.max(), 3)}、検出力 {f(stats.binom.sf(4, 10, 0.5), 3)}", "scipy binom.sf")
simX = rng.binomial(10, 0.2, 400_000)
E.check("s18p01b", abs(np.mean(simX >= 5) - 0.0328) < 0.002, f"q=0.2 のシミュレーションで棄却率 {np.mean(simX >= 5):.4f}")

# 大問2 (b) 比の見かけの相関：近似式を対数の分散（対数正規なら log(1+C²)）で、シミュレーションでも
cx_, cy_, cz_ = 0.10, 0.20, 0.20
E.num("s18p02b", float(f(cz_ ** 2 / math.sqrt((cx_ ** 2 + cz_ ** 2) * (cy_ ** 2 + cz_ ** 2)), 2)), "近似式を直接計算")
lv = lambda c: math.log(1 + c * c)
rho_ln = lv(cz_) / math.sqrt((lv(cx_) + lv(cz_)) * (lv(cy_) + lv(cz_)))
# 対数正規の X,Y,Z（変動係数が指定の値）では比そのものの相関もモーメントから正確に計算できる
sg2 = {k: lv(c) for k, c in (("x", cx_), ("y", cy_), ("z", cz_))}
M_ = lambda s2, k: math.exp(k * k * s2 / 2)   # E[V^k]（対数の平均 0）
cov_r = M_(sg2["x"], 1) * M_(sg2["y"], 1) * (M_(sg2["z"], -2) - M_(sg2["z"], -1) ** 2)
var_r = lambda s2: M_(s2, 2) * M_(sg2["z"], -2) - M_(s2, 1) ** 2 * M_(sg2["z"], -1) ** 2
rho_ex = cov_r / math.sqrt(var_r(sg2["x"]) * var_r(sg2["y"]))
X_, Y_, Z_ = (np.exp(rng.normal(-v / 2, math.sqrt(v), 2_000_000)) for v in (sg2["x"], sg2["y"], sg2["z"]))
rsim = np.corrcoef(X_ / Z_, Y_ / Z_)[0, 1]
E.check("s18p02b", abs(rho_ln - 0.6325) < 0.01 and abs(rho_ex - 0.6325) < 0.015 and abs(rsim - rho_ex) < 0.005,
        f"対数正規で：対数の相関 {rho_ln:.4f}、比の正確な相関 {rho_ex:.4f}、シミュレーション {rsim:.4f}（近似式 0.6325）")

# 大問3 (b) 母分散のカイ二乗検定
table_ok("s18p03b", [(6.262, stats.chi2.ppf(0.025, 15)), (27.488, stats.chi2.isf(0.025, 15)), (24.996, stats.chi2.isf(0.05, 15))], 3, "χ²(15) の点")
X2 = 15 * 6.9 / 4
pv2 = 2 * min(stats.chi2.sf(X2, 15), stats.chi2.cdf(X2, 15))
E.check("s18p03b", 0.05 < pv2 < 0.1 and stats.chi2.sf(X2, 15) < 0.05, f"両側 p 値 {pv2:.4f}（0.05 より大きい）、片側 p 値 {stats.chi2.sf(X2, 15):.4f}（0.05 より小さい）")
E.choice("s18p03b", f"統計量は{f(X2, 3)}で、6.262と27.488の間にあるので棄却しない", "scipy chi2 で両側判定")
sims2 = rng.normal(0, 2, (200_000, 16)).var(axis=1, ddof=1) * 15 / 4
E.check("s18p03b", abs(np.mean((sims2 < 6.262) | (sims2 > 27.488)) - 0.05) < 0.003, "帰無仮説のもとのシミュレーションで両側の棄却率 ≈ 0.05")

# 大問4 (a) ボレル・カンテリ：(i) n=N+1..M に一度も起きない確率 = Π(1-1/n) = N/M → 0、(ii) N 日以降に一度でも起きる確率 ≤ Σ1/n² → 0
N0 = 1000
prod_i = np.prod(1 - 1 / np.arange(N0 + 1, 10 ** 6 + 1))
prod_ii = 1 - np.prod(1 - 1 / np.arange(N0 + 1, 10 ** 6 + 1) ** 2)
E.check("s18p04a", abs(prod_i - N0 / 10 ** 6) < 1e-9 and prod_ii < 1.01 / N0,
        f"(i) 1000 日目以降 100 万日目まで一度も起きない確率 {prod_i:.2e}（→0、無限回起こる）、(ii) 1000 日目以降に一度でも起きる確率 {prod_ii:.2e}（→0）")
E.choice("s18p04a", "(i)確率収束するが概収束しない、(ii)確率収束し概収束もする", "ボレル・カンテリの補題")
# (b) 定数打ち切り：シミュレーションで 2T/θ ~ χ²(16) を確かめ、区間を scipy の点で
table_ok("s18p04b", [(6.908, stats.chi2.ppf(.025, 16)), (28.845, stats.chi2.isf(.025, 16)), (12.401, stats.chi2.ppf(.025, 24)), (39.364, stats.chi2.isf(.025, 24))], 3, "χ² の点")
th_ = 600
life = np.sort(rng.exponential(th_, (200_000, 12)), axis=1)
Tsim = life[:, :8].sum(axis=1) + 4 * life[:, 7]
piv = 2 * Tsim / th_
E.check("s18p04b", abs(piv.mean() - 16) < 0.05 and abs(piv.var() / 32 - 1) < 0.02 and abs(np.mean(piv < stats.chi2.ppf(.025, 16)) - 0.025) < 0.002,
        f"定数打ち切りのシミュレーションで 2T/θ の平均 {piv.mean():.3f}、分散 {piv.var():.2f}（χ²(16) は 16, 32）")
cov_ = np.mean((2 * Tsim / stats.chi2.isf(.025, 16) <= th_) & (th_ <= 2 * Tsim / stats.chi2.ppf(.025, 16)))
E.check("s18p04b", abs(cov_ - 0.95) < 0.003, f"区間の被覆率 {cov_:.4f}")
E.choice("s18p04b", f"({f(9600 / stats.chi2.isf(.025, 16), 0)},\\{f(9600 / stats.chi2.ppf(.025, 16), 0)})", "scipy chi2 の点で区間")

# 大問5 (a) 超幾何分布
hg = stats.hypergeom(50, 10, 5)
E.choice("s18p05a", f"${f(hg.pmf(2), 4)}$", "scipy hypergeom.pmf")
E.check("s18p05a", f(stats.binom.pmf(2, 5, 0.2), 4) == "0.2048" and f(hg.sf(1), 4) == "0.2581", "誤答の値（二項近似・2 尾以上）")
draw = np.array([np.sum(rng.permutation(50)[:5] < 10) for _ in range(100_000)])
E.check("s18p05a", abs(np.mean(draw == 2) - hg.pmf(2)) < 0.004, f"非復元抽出のシミュレーション {np.mean(draw == 2):.4f}")
# (b) 二相抽出の比推定量
from fractions import Fraction as Fr
E.num("s18p05b", float(400 * Fr(45, 10) / Fr(300, 10) * Fr(320, 10)), "分数で正確に計算")
E.check("s18p05b", 30.0 < 32.0, "第 2 相の x の平均が第 1 相より低い（解説の主張：比で補正すると大きくなる）")

# 大問6 (a) 2 本の回帰の傾きの積 = r²：データを作って確かめる
xx = rng.normal(30, 4, 40)
yy = 2.5 * xx + rng.normal(0, 6, 40)
byx_ = np.polyfit(xx, yy, 1)[0]
bxy_ = np.polyfit(yy, xx, 1)[0]
E.check("s18p06a", abs(byx_ * bxy_ - np.corrcoef(xx, yy)[0, 1] ** 2) < 1e-12, "任意のデータで b_yx·b_xy = r²")
E.choice("s18p06a", f"r={f(math.sqrt(2.5 * 0.30), 3)}$、$R^2={f(2.5 * 0.30, 2)}$", "√(積)")
# (b) 体重 ∝ 体長³ に直線を当てはめると、残差が両端で正・中央で負（分散一定の誤差）
L_ = rng.uniform(10, 30, 6000)
W_ = 0.01 * L_ ** 3 + rng.normal(0, 5, 6000)
res6 = W_ - np.polyval(np.polyfit(L_, W_, 1), L_)
lo6, mid6, hi6 = res6[L_ < 13].mean(), res6[(L_ > 18) & (L_ < 22)].mean(), res6[L_ > 27].mean()
res6q = W_ - np.polyval(np.polyfit(L_, W_, 2), L_)
E.check("s18p06b", lo6 > 0 and hi6 > 0 and mid6 < 0 and abs(res6q[(L_ > 18) & (L_ < 22)].mean()) < 0.5 * abs(mid6),
        f"直線の残差の平均 小 {lo6:.1f}・中 {mid6:.1f}・大 {hi6:.1f}（弓なり）、2 次の項を加えると中央の偏り {res6q[(L_ > 18) & (L_ < 22)].mean():.2f}")
E.choice("s18p06b", "関数形の誤り", "弓なり＝関数形")

# 大問7 (a) 準二項：標準誤差を √φ 倍。ベータ二項のシミュレーションで、二項の標準誤差が過小になることも確かめる
E.choice("s18p07a", f"z={f(0.48 / (0.20 * math.sqrt(2.5)), 2)}$で、1.96を下回るので有意でない", "√φ 倍した標準誤差で z")
rho_bb = (2.5 - 1) / (20 - 1)   # φ = 1+(m-1)ρ となる級内相関
a_bb = 0.5 * (1 / rho_bb - 1)
props = rng.binomial(20, rng.beta(a_bb, a_bb, (20_000, 30))) / 20
ratio_ = props.mean(axis=1).var() / (0.25 / 600)
E.check("s18p07a", abs(ratio_ - 2.5) < 0.1, f"ベータ二項（φ=2.5）のシミュレーションで、発芽率の平均の分散は二項の {ratio_:.3f} 倍")
# (b) 前進選択：部分 F をループで
RSS = {(): 400, (1,): 250, (2,): 300, (3,): 330, (1, 2): 200, (1, 3): 240, (1, 2, 3): 180}
sel = ()
while True:
    cand = [v for v in (1, 2, 3) if v not in sel]
    if not cand:
        break
    Fs = {v: (RSS[sel] - RSS[tuple(sorted(sel + (v,)))]) / (RSS[tuple(sorted(sel + (v,)))] / (30 - len(sel) - 2)) for v in cand}
    best = max(Fs, key=Fs.get)
    if Fs[best] < 4.0:
        break
    sel = tuple(sorted(sel + (best,)))
E.check("s18p07b", sel == (1, 2), f"前進選択の結果 {sel}")
E.choice("s18p07b", f"$x_1,x_2$（2段階目の$x_2$の$F={f(50 / (200 / 27), 2)}$、3段階目の$x_3$の$F={f(20 / (180 / 26), 2)}$で止まる）", "部分 F をループで")

# 大問9 中間変数：g 式（Σ_m P(m|t)P(y|t,m)）で総合効果、層別の標準化値も
cnt = {(1, "L"): (150, 84), (1, "S"): (50, 14), (0, "L"): (50, 30), (0, "S"): (150, 45)}
def g(t):
    nt = sum(cnt[(t, m)][0] for m in "LS")
    return sum(cnt[(t, m)][0] / nt * cnt[(t, m)][1] / cnt[(t, m)][0] for m in "LS")
tot9 = g(1) - g(0)
std9 = sum(0.5 * (cnt[(1, m)][1] / cnt[(1, m)][0] - cnt[(0, m)][1] / cnt[(0, m)][0]) for m in "LS")
E.choice("s18p09a", f"$+{f(tot9, 3)}$（大きさで層別せずに", "g 式（中間変数を周辺化）で総合効果")
E.check("s18p09a", f(std9, 3) == "-0.030" and all(cnt[(1, m)][1] / cnt[(1, m)][0] < cnt[(0, m)][1] / cnt[(0, m)][0] for m in "LS"), "層別すると両層で新しい餌が低く、標準化値 −0.030（シンプソンの逆転）")
# 構造モデルからのシミュレーション：T を無作為化 → M|T → Y|T,M。do(T) の差が粗い差と一致
Tm = rng.integers(0, 2, 1_000_000)
Mm = rng.random(1_000_000) < np.where(Tm == 1, 0.75, 0.25)
py = np.where(Tm == 1, np.where(Mm, 0.56, 0.28), np.where(Mm, 0.60, 0.30))
Ym = rng.random(1_000_000) < py
E.check("s18p09a", abs((Ym[Tm == 1].mean() - Ym[Tm == 0].mean()) - 0.115) < 0.004, f"構造モデルのシミュレーションで介入効果 {Ym[Tm == 1].mean() - Ym[Tm == 0].mean():.4f}")

# 大問10 (a) scipy の階層的クラスタリングとコーフェン相関
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
Dm = np.array([[0, 2, 6, 10], [2, 0, 5, 9], [6, 5, 0, 4], [10, 9, 4, 0]], float)
cv = squareform(Dm)
c_s = cophenet(linkage(cv, "single"), cv)[0]
c_a = cophenet(linkage(cv, "average"), cv)[0]
c_c = cophenet(linkage(cv, "complete"), cv)[0]
E.choice("s18p10a", f"最短距離法{f(c_s, 3)}、群平均法{f(c_a, 3)}で、群平均法の方が", "scipy linkage + cophenet")
E.check("s18p10a", f(c_c, 3) == "0.791" and f(stats.spearmanr(cv, cophenet(linkage(cv, "single")))[0], 3) == "0.845", "誤答の値（最長距離法・順位の相関）")
# (b) scipy spearmanr（同順位は中間順位）
aa = [3, 8, 8, 15, 22, 30, 41, 41]
bb = [5, 2, 9, 9, 17, 14, 26, 33]
rs_ = stats.spearmanr(aa, bb)[0]
d2 = np.sum((stats.rankdata(aa) - stats.rankdata(bb)) ** 2)
E.choice("s18p10b", f"r_S={f(rs_, 3)}$（同順位があるので、順位差の2乗和を使う簡略式の値{f(1 - 6 * d2 / (8 * 63), 3)}", "scipy spearmanr")
# (c) 集団寄与危険割合：湖 1000 の集団で数え上げ
r0_ = 0.1
risk_all = 0.6 * r0_ + 0.4 * 3 * r0_
E.num("s18p10c", float(f((risk_all - r0_) / risk_all, 3)), "全体の危険と非曝露の危険から（定義）")

E.done()
