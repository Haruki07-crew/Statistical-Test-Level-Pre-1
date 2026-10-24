"""模擬試験 第13回（set13）の検算。generator/set13.py とは別の方法（scipy の分布関数・シミュレーション・
sympy・総当たり・線形代数）で正解を求め直し、問題文に示した分布の値も確かめる。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(13)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums_of(iid):
    out = []
    for c in E.item(iid)["choices"]:
        m = re.search(r"-?\d+\.?\d*", str(c))
        out.append(float(m.group()) if m else None)
    return out


def nearest(iid, value, what="最も近い選択肢"):
    nums = nums_of(iid)
    k = int(np.argmin([abs(x - value) if x is not None else 1e9 for x in nums]))
    E.check(iid, k == E.item(iid)["answer"], f"{what} {k}（値 {value:.6f}）が正解 {E.item(iid)['answer']}")


def distinct(iid, d):
    """選択肢の数値が丸めの桁で互いに異なる。"""
    nums = [x for x in nums_of(iid) if x is not None]
    E.check(iid, len({round(x, d) for x in nums}) == len(nums), f"選択肢の数値が互いに異なる {nums}")


def tab(name, val, true, tol):
    ok = abs(val - true) <= tol
    print(f"   表の値 {name}: 問題 {val} / 計算 {true:.5f} {'OK' if ok else 'NG'}")
    return ok


# ---- 大問1 ランダム化回答法
# (a) 回答者を 1 人ずつシミュレーション（該当・サイコロ・回答）して P(該当 | はい)
M = 2_000_000
T = rng.random(M) < 0.15
die = rng.integers(1, 7, size=M)
ans = np.where(die <= 4, T, die == 5)
sim = T[ans].mean()
exact = Fr(15, 100) * Fr(5, 6) / (Fr(15, 100) * Fr(5, 6) + Fr(85, 100) * Fr(1, 6))
E.choice("s13p01a", f"${f(float(exact), 3)}$", "分数で厳密に計算した P(該当|はい)")
E.check("s13p01a", abs(sim - float(exact)) < 0.003, f"シミュレーション {sim:.4f} ≈ {float(exact):.4f}")
distinct("s13p01a", 3)
# (b) 推定量の分散を二項分布の厳密な分散から（p を p̂ で置き換え）／シミュレーションで不偏性
ph = 186 / 600
var_pi = stats.binom(600, ph).var() / 600 ** 2 * (1 / (2 / 3)) ** 2
E.num("s13p01b", float(f(np.sqrt(var_pi), 4)), "scipy binom の分散 ×(3/2)^2 の平方根")
pi_true = 0.2
Ys = rng.binomial(600, (2 / 3) * pi_true + 1 / 6, size=400_000)
est = (Ys / 600 - 1 / 6) / (2 / 3)
E.check("s13p01b", abs(est.mean() - pi_true) < 0.0005, f"π=0.2 のシミュレーションで推定量の平均 {est.mean():.5f}（不偏）")
p_ = (2 / 3) * pi_true + 1 / 6
E.check("s13p01b", abs(est.std() - 1.5 * np.sqrt(p_ * (1 - p_) / 600)) < 0.0003, f"推定量の標準偏差 {est.std():.5f} が式と一致")

# ---- 大問2 期日前投票所
lam = 0.2
# 条件付き密度を数値積分で：P(X<=3 | Y=10) = ∫_0^3 f(x,10)dx / ∫_0^10 f(x,10)dx
fxy = lambda x, y: lam ** 2 * np.exp(-lam * y)
cp = integrate.quad(lambda x: fxy(x, 10), 0, 3)[0] / integrate.quad(lambda x: fxy(x, 10), 0, 10)[0]
E.choice("s13p02a", f"${f(cp, 3)}$", "同時密度の数値積分による条件付き確率")
# シミュレーション：Y が 10 付近の標本で X<=3 の割合
Xs = rng.exponential(1 / lam, 3_000_000)
Ys = Xs + rng.exponential(1 / lam, 3_000_000)
sel = np.abs(Ys - 10) < 0.1
E.check("s13p02a", abs((Xs[sel] <= 3).mean() - 0.3) < 0.01, f"シミュレーション {(Xs[sel] <= 3).mean():.4f}")
E.check("s13p02a", tab("e^-0.6", 0.5488, np.exp(-0.6), 5e-5) and tab("e^-2", 0.1353, np.exp(-2), 5e-5), "e^x の値")
distinct("s13p02a", 3)
sk = stats.gamma(2, scale=1 / lam).stats(moments="s")
E.choice("s13p02b", f"${f(float(sk), 3)}$", "scipy gamma(形状2) の歪度")
E.check("s13p02b", abs(stats.skew(Ys) - np.sqrt(2)) < 0.02, f"シミュレーションの歪度 {stats.skew(Ys):.4f}")
E.check("s13p02b", abs(stats.expon(scale=5).stats(moments="v") - 25) < 1e-9 and abs(sp.integrate(
    (sp.Symbol("x") - 5) ** 3 * sp.Rational(1, 5) * sp.exp(-sp.Symbol("x") / 5), (sp.Symbol("x"), 0, sp.oo)) - 250) == 0,
    "指数分布の分散 25・3 次中心モーメント 250（問題文の式）")
E.check("s13p02b", tab("sqrt50", 7.0711, np.sqrt(50), 5e-5), "√50")
distinct("s13p02b", 3)

# ---- 大問3
U = rng.uniform(-6, 6, (4_000_000, 2))
D = np.hypot(U[:, 0], U[:, 1])
D = D[D < 6]
med_sim = np.median(D)
E.choice("s13p03a", f"${f(6 / np.sqrt(2), 2)}$", "F(r)=r²/R² を scipy brentq で解いた中央値と同じ")
from scipy.optimize import brentq  # noqa: E402
m_ = brentq(lambda r_: integrate.quad(lambda u: 2 * u / 36, 0, r_)[0] - 0.5, 0, 6)
nearest("s13p03a", m_, "数値積分で求めた中央値に最も近い選択肢")
E.check("s13p03a", abs(med_sim - m_) < 0.01, f"円内一様の点のシミュレーションの中央値 {med_sim:.4f}")
distinct("s13p03a", 2)
par = stats.pareto(3, scale=200)
exact_max = par.cdf(4000) ** 4000
E.check("s13p03b", tab("e^-0.5", 0.6065, np.exp(-0.5), 5e-5) and tab("e^-0.025", 0.9753, np.exp(-0.025), 5e-5), "e^x の値")
nearest("s13p03b", exact_max, "正確な値 F(x)^n に最も近い選択肢")
E.choice("s13p03b", f"${f(stats.invweibull(3, scale=200 * 4000 ** (1 / 3)).cdf(4000), 3)}$", "scipy のフレシェ分布 invweibull(3, scale=x_m n^{1/3})")
mx = par.rvs(size=(2000, 4000), random_state=7).max(axis=1)
E.check("s13p03b", abs((mx <= 4000).mean() - exact_max) < 0.03, f"最大値のシミュレーション {(mx <= 4000).mean():.3f}")
distinct("s13p03b", 3)

# ---- 大問4 世論調査
# (a) 多項分布の差の分散を sympy で、z を条件付き二項分布で求め直し、帰無仮説のもとのシミュレーションで分散を確認
pa_, pb_ = sp.symbols("p_A p_B", positive=True)
nn = sp.Symbol("n", positive=True)
var_diff = (pa_ * (1 - pa_) + pb_ * (1 - pb_) - 2 * (-pa_ * pb_)) / nn
E.check("s13p04a", sp.simplify(var_diff.subs(pb_, pa_) - 2 * pa_ / nn) == 0, "H0 のもとで V[p̂A−p̂B]=(pA+pB)/n（sympy）")
zc = (516 - 972 * 0.5) / np.sqrt(stats.binom(972, 0.5).var())
E.choice("s13p04a", f"$z={f(zc, 2)}$ で、1.96 を超えないので帰無仮説を棄却しない", f"A か B の回答者に条件付けた二項分布の z={zc:.4f}")
pb2 = stats.binomtest(516, 972, 0.5).pvalue
E.check("s13p04a", 0.05 < pb2 < 0.07, f"正確な二項検定の両側 p 値 {pb2:.4f} > 0.05（結論が同じ）")
cnt = rng.multinomial(1200, [0.405, 0.405, 0.19], size=400_000)
dsim = (cnt[:, 0] - cnt[:, 1]) / 1200
E.check("s13p04a", abs(dsim.var() / (0.81 / 1200) - 1) < 0.01, f"多項分布のシミュレーションで分散比 {dsim.var() / (0.81 / 1200):.4f} ≈ 1")
E.check("s13p04a", tab("z_0.025", 1.96, stats.norm.ppf(0.975), 5e-3) and tab("z_0.05", 1.645, stats.norm.ppf(0.95), 5e-4), "正規分布の点")
# (b) scipy の負の二項分布（失敗回数）＋ n で発信回数の標準偏差、シミュレーション
nb = stats.nbinom(1200, 0.15)
E.num("s13p04b", float(f(nb.std(), 1)), "scipy nbinom(1200, 0.15) の標準偏差（失敗回数 + 定数）")
Tsim = rng.negative_binomial(1200, 0.15, size=200_000) + 1200
E.check("s13p04b", abs(Tsim.std() - nb.std()) < 1.5 and abs(Tsim.mean() - 8000) < 2, f"シミュレーション 平均 {Tsim.mean():.1f}・標準偏差 {Tsim.std():.2f}")
# (c) 二項分布の正確な値と、示した Φ の値
E.check("s13p04c", tab("Φ(1.20)", 0.8849, stats.norm.cdf(1.20), 5e-5) and tab("Φ(1.22)", 0.8888, stats.norm.cdf(1.22), 5e-5)
        and tab("Φ(2.00)", 0.9772, stats.norm.cdf(2.00), 5e-5), "Φ の値")
al = stats.norm.sf(0.55, 0.5, np.sqrt(0.25 / 400))
be = stats.norm.cdf(0.55, 0.58, np.sqrt(0.58 * 0.42 / 400))
zb = round((0.58 - 0.55) / np.sqrt(0.58 * 0.42 / 400), 2)
E.choice("s13p04c", f"\\alpha={f(al, 4)}$、$\\beta={f(stats.norm.sf(zb), 4)}", f"scipy norm で α、表の桁 z={zb} に丸めて β")
E.check("s13p04c", abs(be - 0.1112) < abs(be - 0.1151) and abs(be - 0.1112) < 0.001, f"丸めない β={be:.4f} も正解の値に最も近い")
ex_a = stats.binom(400, 0.5).sf(219)
ex_b = stats.binom(400, 0.58).cdf(219)
E.check("s13p04c", abs(ex_a - al) < 0.01 and abs(ex_b - be) < 0.03, f"二項分布の正確な値 α={ex_a:.4f}, β={ex_b:.4f} と近い")

# ---- 大問5 予備検定の問題：シミュレーションで手順全体の第一種の過誤と、ウェルチ検定の水準を比べる
n1_, n2_, sd1, sd2 = 8, 12, 6.8, 3.9
R = 40000
a1 = rng.normal(0, sd1, (R, n1_))
a2 = rng.normal(0, sd2, (R, n2_))
v1_, v2_ = a1.var(1, ddof=1), a2.var(1, ddof=1)
Fst = v1_ / v2_
pF = 2 * np.minimum(stats.f.sf(Fst, n1_ - 1, n2_ - 1), stats.f.cdf(Fst, n1_ - 1, n2_ - 1))
p_pool = stats.ttest_ind(a1, a2, axis=1, equal_var=True).pvalue
p_welch = stats.ttest_ind(a1, a2, axis=1, equal_var=False).pvalue
two = np.where(pF < 0.05, p_welch, p_pool) < 0.05
E.check("s13p05a", abs(two.mean() - 0.05) > 0.01, f"母分散が異なるとき二段階の手順の誤り率 {two.mean():.4f}（5% からずれる）")
E.check("s13p05a", abs((p_welch < 0.05).mean() - 0.05) < 0.006, f"ウェルチ検定の誤り率 {(p_welch < 0.05).mean():.4f} ≈ 5%")
E.check("s13p05a", (p_pool[pF >= 0.05] < 0.05).mean() > 0.06, f"予備検定を通った後のプールした t 検定の誤り率 {(p_pool[pF >= 0.05] < 0.05).mean():.4f} > 5%（選択肢 0・4 は誤り）")
b1_ = rng.normal(0, 5, (R, n1_)); b2_ = rng.normal(0, 5, (R, n2_))
E.check("s13p05a", abs((stats.ttest_ind(b1_, b2_, axis=1, equal_var=False).pvalue < 0.05).mean() - 0.05) < 0.006, "等分散でもウェルチ検定の水準は約 5%（選択肢 2 は誤り）")
E.check("s13p05a", E.item("s13p05a")["answer"] == 1, "正解は選択肢 1")
Fp = 6.8 ** 2 / 3.9 ** 2
E.check("s13p05a", abs(2 * stats.f.sf(Fp, 7, 11) - 0.10) < 0.005, f"問題文の F 検定の両側 p 値 {2 * stats.f.sf(Fp, 7, 11):.4f} ≈ 0.10")

# ---- 大問6 ケース・コントロール：母集団を仮定して抽出をシミュレーションし、オッズ比は保たれ投票率は保たれないことを確認
a, b, c, d = 160, 240, 100, 300
E.check("s13p06a", abs(a * d / (b * c) - 2.0) < 1e-12 and abs((a / c) / (b / d) - 2.0) < 1e-12, "行・列どちらの向きでもオッズ比 2.0")
# 母集団：投票率 0.55、はがき受領で投票オッズ 2 倍、受領率 0.3 の例
pr0 = sp.Symbol("p0", positive=True)
sol = sp.nsolve(0.3 * (2 * pr0 / (1 - pr0)) / (1 + 2 * pr0 / (1 - pr0)) + 0.7 * pr0 - 0.55, pr0, 0.5)
p0v = float(sol)
p1v = 2 * p0v / (1 - p0v) / (1 + 2 * p0v / (1 - p0v))
Nsim = 2_000_000
card = rng.random(Nsim) < 0.3
vote = rng.random(Nsim) < np.where(card, p1v, p0v)
iv = rng.choice(np.where(vote)[0], 400, replace=False)
ia = rng.choice(np.where(~vote)[0], 400, replace=False)
A_, B_ = card[iv].sum(), 400 - card[iv].sum()
C_, D_ = card[ia].sum(), 400 - card[ia].sum()
E.check("s13p06a", abs(np.log(A_ * D_ / (B_ * C_)) - np.log(2)) < 0.35, f"ケース・コントロール抽出のオッズ比 {A_ * D_ / (B_ * C_):.2f} ≈ 母集団の 2")
E.check("s13p06a", abs(A_ / (A_ + C_) - p1v) > 0.03, f"抽出表の行比率 {A_ / (A_ + C_):.3f} は真の投票率 {p1v:.3f} と一致しない")
E.check("s13p06a", abs(p1v / p0v - 2.0) > 0.3, f"投票率 55% では相対リスク {p1v / p0v:.3f} はオッズ比 2 から離れる")
E.check("s13p06a", E.item("s13p06a")["answer"] == 1, "正解はオッズ比の記述")

# ---- 大問7 状態の分類：到達可能性の推移閉包と、行列のべき乗の極限
Pm = np.array([[0.90, 0.04, 0.03, 0.03], [0.10, 0.85, 0.02, 0.03], [0, 0, 0.80, 0.20], [0, 0, 0, 1]])
E.check("s13p07a", np.allclose(Pm.sum(1), 1), "行和 1")
Rch = (Pm > 0).astype(int)
for _ in range(4):
    Rch = ((Rch + Rch @ Rch) > 0).astype(int)
closed = [i for i in range(4) if all(Rch[j, i] for j in range(4) if Rch[i, j])]
E.check("s13p07a", closed == [3], f"到達先から必ず戻れる（再帰的）状態は {closed}（状態 4 のみ）")
Pn = np.linalg.matrix_power(Pm, 2000)
E.check("s13p07a", np.allclose(Pn[:, 3], 1, atol=1e-9), "P^2000 の第 4 列がすべて 1（確率 1 で吸収）")
w_, vec = np.linalg.eig(Pm.T)
E.check("s13p07a", np.sum(np.abs(w_ - 1) < 1e-9) == 1, "固有値 1 は 1 個（定常分布は (0,0,0,1) のみ）")
E.check("s13p07a", E.item("s13p07a")["answer"] == 2, "正解は選択肢 2")

# ---- 大問8
# (a) 交互作用：個票を作って最小二乗で β1+β3 とその分散を推定（共分散行列が問題の値に比例することを確認）
Xd = np.column_stack([np.ones(40), np.r_[np.linspace(0, 6, 20), np.linspace(0, 6, 20)], np.r_[np.zeros(20), np.ones(20)]])
Xd = np.column_stack([Xd, Xd[:, 1] * Xd[:, 2]])
XtXi = np.linalg.inv(Xd.T @ Xd)
cvec = np.array([0, 1, 0, 1])
E.check("s13p08a", abs(XtXi[1, 3] + XtXi[1, 1]) < 1e-12, "完全交互作用モデルでは Cov(β1,β3) = −V(β1)（問題の値の組と整合）")
V = np.array([[0.00016, -0.00016], [-0.00016, 0.00036]])
se_ = np.sqrt(np.array([1, 1]) @ V @ np.array([1, 1]))
E.choice("s13p08a", f"推定値 ${f(0.05 - 0.03, 3)}$、標準誤差 ${f(se_, 4)}$", "行列計算 c'Vc")
# (b) 内部スチューデント化残差：V[e_k]=σ²(1−h_kk) をシミュレーションで確かめ、e/√(RSS(1−h)/(n−p)) で計算
Xh = np.column_stack([np.ones(40), rng.standard_normal((40, 4))])
Hh = Xh @ np.linalg.inv(Xh.T @ Xh) @ Xh.T
Eps = rng.standard_normal((100_000, 40))
Res = Eps - Eps @ Hh.T
E.check("s13p08b", np.allclose(Res.var(0), 1 - np.diag(Hh), atol=0.02), "シミュレーションで V[e_k]=σ²(1−h_kk)（σ²=1）")
E.num("s13p08b", float(f(0.48 / np.sqrt(1.80 * (1 - 0.38) / (40 - 5)), 2)), "e_k/√(RSS(1−h)/(n−p))")
# (c) 楽観度：シミュレーションで E[新しい観測の平均二乗誤差] − E[RSS/n] = 2pσ²/n を確認し、推定値を計算
Xd5 = np.column_stack([np.ones(40), rng.standard_normal((40, 4))])
Hm = Xd5 @ np.linalg.pinv(Xd5)
tr_, te_ = [], []
for _ in range(20000):
    e1 = rng.standard_normal(40)
    e2 = rng.standard_normal(40)
    res_ = e1 - Hm @ e1
    tr_.append(res_ @ res_ / 40)
    te_.append(((e2 - Hm @ e1) ** 2).mean())
E.check("s13p08c", abs(np.mean(te_) - np.mean(tr_) - 2 * 5 / 40) < 0.01, f"σ²=1 のシミュレーションで差 {np.mean(te_) - np.mean(tr_):.4f} ≈ 2p/n=0.25")
E.num("s13p08c", float(f(1.80 / 40 * (40 + 5) / (40 - 5), 4)), "別の形 (RSS/n)(n+p)/(n−p)")

# ---- 大問9
E.check("s13p09a", tab("chi2_0.05(3)", 7.815, stats.chi2.ppf(0.95, 3), 5e-4) and tab("chi2_0.01(3)", 11.345, stats.chi2.ppf(0.99, 3), 5e-4)
        and tab("chi2_0.01(2)", 9.210, stats.chi2.ppf(0.99, 2), 5e-4), "カイ二乗分布の点")
lr = 1012.6 - 1001.9
pv9 = stats.chi2.sf(lr, 3)
E.check("s13p09a", 0.01 < pv9 < 0.05, f"LR={lr:.1f}、自由度 3 の p 値 {pv9:.4f}")
E.choice("s13p09a", "自由度は3で、有意水準5%では棄却するが1%では棄却しない", "正解の記述")
# (b) ナイーブベイズ：特徴の同時分布を条件付き独立で作り、全確率から直接事後確率
pri = {"v": 0.6, "a": 0.4}
pp = {"v": (0.6, 0.8, 0.4), "a": (0.2, 0.3, 0.25)}
from itertools import product  # noqa: E402
joint = {}
for cl in "va":
    for fs in product([0, 1], repeat=3):
        pr_ = pri[cl]
        for fv, pj in zip(fs, pp[cl]):
            pr_ *= pj if fv else 1 - pj
        joint[(cl,) + fs] = pr_
E.check("s13p09b", abs(sum(joint.values()) - 1) < 1e-12, "同時分布の合計 1")
post = joint[("v", 0, 1, 1)] / (joint[("v", 0, 1, 1)] + joint[("a", 0, 1, 1)])
E.choice("s13p09b", f"${f(post, 3)}$", "同時分布の表から条件付き確率")
distinct("s13p09b", 3)

# ---- 大問10
E.check("s13p10a", tab("t_0.025(27)", 2.052, stats.t.ppf(0.975, 27), 5e-4) and tab("t_0.025(9)", 2.262, stats.t.ppf(0.975, 9), 5e-4), "t の点")
cc = np.array([-0.5, -0.5, 1.0])
mm = np.array([3.2, 5.2, 6.1])
tc = (cc @ mm) / np.sqrt(4.5 * (cc @ cc) / 10)
E.choice("s13p10a", f"$t={f(tc, 2)}$で、2.052を超えるので有意", "ベクトル計算の対比 t")
E.check("s13p10a", stats.t.sf(tc, 27) * 2 < 0.05, f"両側 p 値 {2 * stats.t.sf(tc, 27):.4f}")
# (b) 個票を作って重み付きでない通常の最小二乗を個票に当てはめると、集計値の WLS と一致
xs = np.repeat([2, 4, 6, 8], [10, 40, 30, 20]).astype(float)
ys = np.repeat([2.0, 3.6, 3.8, 6.0], [10, 40, 30, 20]) + 0.0
dev = rng.standard_normal(100)
for k_, (a0, b0) in enumerate([(0, 10), (10, 50), (50, 80), (80, 100)]):
    dd_ = dev[a0:b0] - dev[a0:b0].mean()
    ys[a0:b0] += dd_
b_ind = np.polyfit(xs, ys, 1)[0]
E.num("s13p10b", float(f(b_ind, 3)), "個票（各機関の平均が表と一致）への通常の最小二乗の傾き")
Wm = np.diag([10, 40, 30, 20.0])
Xg = np.column_stack([np.ones(4), [2, 4, 6, 8]])
bw = np.linalg.solve(Xg.T @ Wm @ Xg, Xg.T @ Wm @ np.array([2.0, 3.6, 3.8, 6.0]))[1]
E.check("s13p10b", abs(bw - b_ind) < 1e-9, f"行列の WLS (X'WX)^{{-1}}X'Wy = {bw:.5f}")

# ---- 大問11 無回答の追跡調査：母集団を作って二相の推定量の不偏性をシミュレーション
E.choice("s13p11a", f"${f(0.6 * 0.70 + 0.4 * 0.40, 3)}$", "層の重み付き平均")
Npop = 200_000
resp = rng.random(Npop) < 0.6
intent = np.where(resp, rng.random(Npop) < 0.70, rng.random(Npop) < 0.40)
ests, naive = [], []
for _ in range(3000):
    smp = rng.choice(Npop, 1000, replace=False)
    r_ = resp[smp]
    yr_ = intent[smp][r_].mean()
    nm_idx = np.where(~r_)[0]
    fu = rng.choice(nm_idx, max(1, len(nm_idx) // 4), replace=False)
    ym_ = intent[smp][fu].mean()
    ests.append(r_.mean() * yr_ + (1 - r_.mean()) * ym_)
    naive.append(yr_)
E.check("s13p11a", abs(np.mean(ests) - intent.mean()) < 0.003 and abs(np.mean(naive) - intent.mean()) > 0.1,
        f"二相の推定量の平均 {np.mean(ests):.4f} ≈ 真値 {intent.mean():.4f}、回答者のみ {np.mean(naive):.4f} は偏る")
distinct("s13p11a", 3)

# ---- 大問12
Dm = np.array([[0, 2, 6, 10, 9], [2, 0, 5, 9, 8], [6, 5, 0, 4, 7], [10, 9, 4, 0, 3], [9, 8, 7, 3, 0]], float)
lab = np.array([0, 0, 1, 1, 1])
silv = []
for i_ in range(5):
    own = (lab == lab[i_]) & (np.arange(5) != i_)
    a_ = Dm[i_, own].mean()
    b_ = Dm[i_, lab != lab[i_]].mean()
    silv.append((b_ - a_) / max(a_, b_))
E.choice("s13p12a", f"$s(C)={f(silv[2], 3)}$、$s(D)={f(silv[3], 3)}$", f"マスクを使った行列計算のシルエット係数 {np.round(silv, 4)}")
E.check("s13p12a", len(set(E.item("s13p12a")["choices"])) == 5, "選択肢の組が互いに異なる")
# (b) 斜交因子：因子・独自因子を発生させて共通部分の分散
Phi = np.array([[1, 0.4], [0.4, 1]])
Fz = rng.multivariate_normal([0, 0], Phi, 1_000_000)
comm = Fz @ np.array([0.6, 0.3])
h2 = np.array([0.6, 0.3]) @ Phi @ np.array([0.6, 0.3])
E.num("s13p12b", float(f(h2, 3)), "l'Φl の行列計算")
E.check("s13p12b", abs(comm.var() - h2) < 0.003, f"シミュレーションの共通部分の分散 {comm.var():.4f}")

# ---- 大問13 AR(1) のスペクトル：自己共分散のフーリエ和で数値的に求める
phi = 0.6
gam = lambda k: phi ** abs(k) / (1 - phi ** 2)
fs = lambda w: (gam(0) + 2 * sum(gam(k) * np.cos(w * k) for k in range(1, 400))) / (2 * np.pi)
ratio = fs(0) / fs(np.pi)
E.choice("s13p13a", f"$f(0)/f(\\pi)={f(ratio, 0)}$で、周期の長い", f"自己共分散の和から f(0)/f(π)={ratio:.4f}")
E.check("s13p13a", abs(fs(1.0) - 1 / (2 * np.pi * (1 - 2 * phi * np.cos(1.0) + phi ** 2))) < 1e-9, "問題文のスペクトル密度の式と一致（σ²=1）")
E.check("s13p13a", E.item("s13p13a")["answer"] == 3, "正解は選択肢 3")

# ---- 大問14
N_, n_, x_ = 5000, 800, 344
hw = 1.96 * np.sqrt(stats.hypergeom(N_, int(N_ * 0.43), n_).var() / n_ ** 2)
E.choice("s13p14a", f"${f(hw, 4)}$", "超幾何分布（母集団支持率 0.43）の分散から半幅")
distinct("s13p14a", 4)
# (b) 経験ベイズ：周辺尤度ではなくモーメント法の定義どおりに numpy で
yy = np.array([0.42, 0.50, 0.38, 0.55, 0.40])
t2 = yy.var(ddof=1) - 0.0025
Bv = 0.0025 / (0.0025 + t2)
E.num("s13p14b", float(f(yy.mean() + (1 - Bv) * (yy[3] - yy.mean()), 3)), "numpy の不偏分散から縮小推定")
# 事後平均の式を sympy で（正規×正規の積を完成平方）
th = sp.Symbol("th")
mu_, s2_, t2_ = sp.Rational(45, 100), sp.Rational(25, 10000), sp.nsimplify(t2)
logpost = -(sp.Rational(55, 100) - th) ** 2 / (2 * s2_) - (th - mu_) ** 2 / (2 * t2_)
pm = sp.solve(sp.diff(logpost, th), th)[0]
E.num("s13p14b", float(f(float(pm), 3)), "sympy で事後密度の最大点（正規なので事後平均）")
E.done()
