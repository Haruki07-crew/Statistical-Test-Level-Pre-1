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
# (b) 調和平均：同じ件数を処理する時間を直接計算
m_ = 120
E.choice("s25p02b", f"${f(2 * m_ / (m_ / 40 + m_ / 60), 1)}$", "総件数 ÷ 総時間")
# (c) フィッシャーの z：2 変量正規のシミュレーションで ρ=0.9 のもとの分布を確かめ、統計量を計算
z_ = (np.arctanh(0.80) - np.arctanh(0.90)) * math.sqrt(25)
E.check("s25p02c", abs(np.arctanh(0.8) - 1.0986) < 5e-5 and abs(np.arctanh(0.9) - 1.4722) < 5e-5, "示した tanh⁻¹ の値")
XY = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=(40000, 28))
xc = XY - XY.mean(1, keepdims=True)
rr = (xc[..., 0] * xc[..., 1]).sum(1) / np.sqrt((xc[..., 0] ** 2).sum(1) * (xc[..., 1] ** 2).sum(1))
zz = (np.arctanh(rr) - np.arctanh(0.9)) * 5
E.check("s25p02c", abs(zz.std() - 1) < 0.03 and 0.035 < np.mean(zz < -1.645) < 0.047, f"ρ=0.9 のもとの統計量の標準偏差 {zz.std():.3f}、棄却率 {np.mean(zz < -1.645):.4f}")
E.choice("s25p02c", f"統計量は ${f(z_, 2)}$ で、$-1.645$ を下回るので帰無仮説を棄却", f"z={z_:.4f}")

# ---- 大問3 安定性
val = integrate.dblquad(lambda x, y: 0.1 * math.exp(-0.1 * x) / 24, 0, 24, lambda y: 0, lambda y: y)[0]
E.choice("s25p03a", f"${f(val, 3)}$", "二重積分で P(X≤Y)")
X = rng.exponential(10, 1_000_000)
Y = rng.uniform(0, 24, 1_000_000)
E.check("s25p03a", close((X <= Y).mean(), val, 0.002), f"シミュレーション {(X <= Y).mean():.4f}")
E.check("s25p03a", close(math.exp(-2.4), 0.0907, 5e-5) and close(math.exp(-1.2), 0.3012, 5e-5), "e^-2.4, e^-1.2")
# (b) 10^U の期待値を数値積分とシミュレーションで
ey = integrate.quad(lambda u: 10 ** u, 0, 1)[0]
E.choice("s25p03b", f"$E[Y]={f(ey, 2)}$、$10^{{E[U]}}={f(10 ** 0.5, 2)}$", f"数値積分 {ey:.5f}")
E.check("s25p03b", abs(np.mean(10 ** rng.random(1_000_000)) - ey) < 0.01, "シミュレーション")

# ---- 大問4 ラベル（全列挙）
perms = list(permutations(range(4)))
none = sum(all(p[i] != i for i in range(4)) for p in perms) / len(perms)
E.choice("s25p04a", f"${f(none, 3)}$", "4! 通りの全列挙")

# ---- 大問5 培地
O5 = np.array([[18, 42], [8, 52]])
g5 = stats.chi2_contingency(O5, correction=False, lambda_="log-likelihood")
E.choice("s25p05a", f"$G={f(g5.statistic, 2)}$ で、自由度 1 の 3.841 を上回るので、独立という帰無仮説を棄却する", f"scipy chi2_contingency(G) {g5.statistic:.4f}, 自由度 {g5.dof}")
E.check("s25p05a", all(abs(math.log(o_ / e_) - v_) < 5e-5 for o_, e_, v_ in ((18, 13, 0.3254), (42, 47, -0.1125), (8, 13, -0.4855), (52, 47, 0.1011))), "示した ln の値")
x5 = np.r_[np.ones(60), np.zeros(60)]; y5 = np.r_[np.ones(18), np.zeros(42), np.ones(8), np.zeros(52)]
ph = np.corrcoef(x5, y5)[0, 1]
X2 = stats.chi2_contingency(O5, correction=False).statistic
E.choice("s25p05b", f"$\\phi={f(ph, 3)}$、$n\\phi^2={f(120 * ph * ph, 2)}$（ピアソン", f"0/1 変数の相関 {ph:.5f}、X²={X2:.4f}")
E.check("s25p05b", abs(120 * ph * ph - X2) < 1e-9, "nφ² = X²")
post = stats.beta(5, 25)
mode = optimize.minimize_scalar(lambda p_: -post.logpdf(p_), bounds=(0.001, 0.999), method="bounded").x
E.choice("s25p05c", f"事後平均 ${f(post.mean(), 3)}$、MAP ${f(mode, 3)}$", f"scipy beta の平均と密度の最大点 {mode:.5f}")

# ---- 大問6 対数の t 区間を指数変換
lo6, hi6 = np.exp(stats.t.interval(0.95, 8, loc=math.log(100), scale=0.10 / 3))
E.choice("s25p06a", f"$({f(lo6, 1)},\\ {f(hi6, 1)})$ で、生菌数の中央値の区間", f"scipy の t 区間を指数変換 ({lo6:.3f}, {hi6:.3f})")
E.check("s25p06a", all(abs(math.exp(a_) - b_) < 5e-5 for a_, b_ in ((-0.0769, 0.9260), (0.0769, 1.0799), (-0.0653, 0.9368), (0.0653, 1.0675))) and abs(stats.t.isf(0.025, 8) - 2.306) < 5e-4, "示した値")
Lg = rng.normal(math.log(100), 0.10, (100000, 9))
lo_, hi_ = np.exp(Lg.mean(1) - 2.306 * Lg.std(1, ddof=1) / 3), np.exp(Lg.mean(1) + 2.306 * Lg.std(1, ddof=1) / 3)
E.check("s25p06a", abs(np.mean((lo_ <= 100) & (100 <= hi_)) - 0.95) < 0.004, "中央値 100 を含む割合（被覆率）")

# ---- 大問7 ポアソンの最強力検定
c7 = next(c for c in range(1, 40) if stats.poisson.sf(c - 1, 6) <= 0.05)
E.choice("s25p07a", f"$c={c7}$、検出力 {f(stats.poisson.sf(c7 - 1, 12), 3)}", f"scipy の poisson c={c7}")
q7 = E.item("s25p07a")["question"]
E.check("s25p07a", all(f"{stats.poisson.sf(t - 1, m_):.4f}" in q7 for t in (9, 10, 11, 12) for m_ in (6, 12)), "表の値")

# ---- 大問8
# (a) 推移行列の固有ベクトルと、再帰時間のシミュレーション
P8 = np.array([[0.4, 0.6], [0.2, 0.8]])
w8, V8 = np.linalg.eig(P8.T); pi8 = np.real(V8[:, np.argmin(abs(w8 - 1))]); pi8 /= pi8.sum()
st = 0; last = 0; gaps = []; run = 0; runs = []
u8 = rng.random(400_000)
for t_, u_ in enumerate(u8):
    st = (1 if u_ < 0.6 else 0) if st == 0 else (0 if u_ < 0.2 else 1)
    if st == 0:
        gaps.append(t_ - last); last = t_
        if run: runs.append(run); run = 0
    else:
        run += 1
E.check("s25p08a", abs(np.mean(gaps[1:]) - 1 / pi8[0]) < 0.05 and abs(np.mean(runs) - 5) < 0.1, f"シミュレーションの再帰時間 {np.mean(gaps[1:]):.3f}、使用の連続日数 {np.mean(runs):.3f}")
E.choice("s25p08a", f"(i) {f(1 / pi8[0], 2)} 日、(ii) {f(1 / 0.2, 2)} 日", f"定常分布 {pi8}")
# (b) AR(1) をシミュレーション
e8 = rng.normal(0, 0.3, 400_000); x8 = np.zeros_like(e8)
for t_ in range(1, len(e8)):
    x8[t_] = 0.8 * x8[t_ - 1] + e8[t_]
k8 = next(k for k in range(1, 20) if 0.8 ** k < 0.5)
E.check("s25p08b", abs(x8[1000:].std() - 0.5) < 0.005, f"シミュレーションの標準偏差 {x8[1000:].std():.4f}")
E.choice("s25p08b", f"標準偏差 {f(math.sqrt(0.09 / 0.36), 2)}、$k={k8}$", "式とラグの探索")

# ---- 大問9 検量線
# (a) てこ比を計画行列から求め、クックの距離を計算
x9 = np.repeat([2, 4, 6, 8, 10, 12], 2).astype(float)
X9 = np.c_[np.ones(12), x9]; H9 = X9 @ np.linalg.inv(X9.T @ X9) @ X9.T
h6_, h12_ = H9[4, 4], H9[10, 10]
E.check("s25p09a", abs(h6_ - 0.090) < 5e-4 and abs(h12_ - 0.262) < 5e-4 and abs(H9[0, 0] - h12_) < 1e-12, f"計画行列のてこ比 濃度6:{h6_:.4f} 濃度12:{h12_:.4f}")
D9 = {k_: r_ ** 2 / 2 * h_ / (1 - h_) for k_, (r_, h_) in {"P": (2.5, 0.090), "Q": (1.6, 0.262), "R": (0.5, 0.262)}.items()}
kb = max(D9, key=D9.get)
E.choice("s25p09a", f"{kb} で、$D={f(D9[kb], 2)}$", f"D={D9}")
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

# ---- 大問10 スプライン：計画行列の階数で自由度を数える
tt = np.linspace(20, 60, 41)
B1 = np.c_[np.ones_like(tt), tt, np.maximum(tt - 30, 0), np.maximum(tt - 45, 0)]
seg_ = [(tt < 30), (tt >= 30) & (tt < 45), (tt >= 45)]
B2 = np.column_stack([m_ * 1.0 for m_ in seg_] + [m_ * tt for m_ in seg_])
E.choice("s25p10a", f"(i) {np.linalg.matrix_rank(B1)}、(ii) {np.linalg.matrix_rank(B2)}", "計画行列の階数")

# ---- 大問11
# (a) 群の平均と大きさがそろうデータを作り scipy の f_oneway の群間平方和で確かめる
grp = []
for n_, m_ in ((4, 12.0), (6, 15.0), (5, 13.0)):
    e_ = rng.normal(0, 2, n_); e_ -= e_.mean(); grp.append(m_ + e_)
allv = np.concatenate(grp); gm_ = allv.mean()
ssb_ = sum(len(g_) * (g_.mean() - gm_) ** 2 for g_ in grp)
E.choice("s25p11a", f"$F={f(ssb_ / 2 / 4.0, 2)}$ で、3.885 を下回り", f"群間平方和 {ssb_:.4f}")
E.check("s25p11a", abs(stats.f.isf(0.05, 2, 12) - 3.885) < 5e-4, "F 点")
# (b) ポアソン回帰をニュートン法で
x11 = np.r_[np.zeros(10), np.ones(10)]
y11 = np.r_[np.full(10, 6.0) + np.array([1, -1, 2, -2, 0, 1, -1, 0, 0, 0]), np.full(10, 4.5) + np.array([0.5, -0.5, 1.5, -1.5, 0, 0, 0, 0, 0, 0])]
X11 = np.c_[np.ones(20), x11]; bb = np.zeros(2)
for _ in range(50):
    mu11 = np.exp(X11 @ bb); bb = bb + np.linalg.solve(X11.T @ (mu11[:, None] * X11), X11.T @ (y11 - mu11))
E.check("s25p11b", abs(y11[:10].mean() - 6.0) < 1e-12 and abs(y11[10:].mean() - 4.5) < 1e-12, "群の平均")
E.num("s25p11b", rnd(bb[1], 3), f"ニュートン法 β={bb}")

# ---- 大問12 PPS：包含確率 2x/X となる非復元の抽出を全列挙で作り、HT 推定量の不偏性を確かめる
xs12 = np.array([100, 200, 300, 400.]); pi12 = 2 * xs12 / xs12.sum()
from itertools import combinations as comb_
pairs = list(comb_(range(4), 2))
# 各組の確率 p_ij を、包含確率が pi になるように線形方程式で決める（6 組・4 本の式＋和が 1）
A12 = np.array([[1.0 if i in pr else 0.0 for pr in pairs] for i in range(4)] + [[1.0] * 6])
p12 = optimize.linprog(np.zeros(6), A_eq=A12, b_eq=np.r_[pi12, 1], bounds=[(0.01, 1)] * 6).x
E.check("s25p12a", np.all(p12 > 0) and np.allclose(A12 @ p12, np.r_[pi12, 1]), f"包含確率 {pi12} を実現する抽出確率 {np.round(p12, 4)}")
yall = np.array([12, 30, 48, 70.])
est = [sum(yall[i] / pi12[i] for i in pr) for pr in pairs]
E.check("s25p12a", abs(np.dot(p12, est) - yall.sum()) < 1e-9, "HT 推定量の期待値が総数に一致（不偏）")
E.num("s25p12a", rnd(30 / pi12[1] + 70 / pi12[3], 1), "選ばれた 2 ロットでの値")

# ---- 大問13 k-means を 1 ステップずつ
P13 = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [4, 5.]])
c13 = np.array([[1, 1], [2, 1.]])
lab13 = ((P13[:, None] - c13[None]) ** 2).sum(2).argmin(1)
c13b = np.array([P13[lab13 == k_].mean(0) for k_ in range(2)])
lab13b = ((P13[:, None] - c13b[None]) ** 2).sum(2).argmin(1)
moved = [f"S{i + 1}" for i in range(6) if lab13[i] != lab13b[i]]
E.check("s25p13a", moved == ["S2"], f"2 回目に移る株 {moved}")
E.choice("s25p13a", f"新しい $c_2=({f(c13b[1, 0], 2)},\\ {f(c13b[1, 1], 2)})$。2 回目の割り当てで {moved[0]} がクラスター 1 に移る", "k-means の手順")

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
