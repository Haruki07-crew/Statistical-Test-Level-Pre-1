"""模擬試験 第6回（set06）の検算。generator/set06.py とは別の方法（scipy の分布関数・数値積分・数値微分・
シミュレーション・総当たり・行列計算）で正解を求め直す。"""
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(6)
rng = np.random.default_rng(20261106)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.search(r"-?\d+\.?\d*", str(c)).group()) for c in E.item(iid)["choices"]]


def nearest(iid, value):
    """選択肢の数値のうち value に最も近いものが正解か。"""
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']}")


# ---- 大問1 生存関数 S(t)=(θ/(θ+t))^α
al, th = 2.5, 10.0
t = sp.symbols("t", positive=True)
S = (sp.Integer(10) / (10 + t)) ** sp.Rational(5, 2)
fdens = -sp.diff(S, t)
h6 = float((fdens / S).subs(t, 6))
E.choice("s06p01a", f"${f(h6, 3)}$", "sympy で f=-S' を作り f/S を t=6 で評価")
E.check("s06p01a", abs(stats.lomax(c=al, scale=th).sf(6) - float(S.subs(t, 6))) < 1e-12, "scipy lomax の生存関数がモデルと一致")
E.check("s06p01a", abs(stats.lomax(c=al, scale=th).pdf(6) / stats.lomax(c=al, scale=th).sf(6) - h6) < 1e-12, "scipy lomax の pdf/sf")
# (b) 平均残存期間を数値積分とシミュレーションで
Lx = stats.lomax(c=al, scale=th)
mrl = integrate.quad(lambda u: Lx.sf(6 + u), 0, np.inf)[0] / Lx.sf(6)
E.num("s06p01b", rnd(mrl, 2), "∫S(6+u)du / S(6) を数値積分")
smp = Lx.rvs(size=4_000_000, random_state=rng)
ex = smp[smp > 6] - 6
E.check("s06p01b", abs(np.median(ex) - (16 * (2 ** (1 / 2.5) - 1))) < 0.1, f"シミュレーションの残存期間の中央値 {np.median(ex):.3f} が理論値と一致（裾が重く平均は不安定なので中央値で確認）")
# (c) 比例ハザード：基準ハザードを具体的に置いて（ワイブル型）数値積分で S1(12) を求める
k_w = 1.3
lam_w = (-np.log(0.55)) / 12 ** k_w   # S0(12)=0.55 となる累積ハザード H0(t)=lam t^k
h0 = lambda u: lam_w * k_w * u ** (k_w - 1)
H1 = integrate.quad(lambda u: h0(u) * np.exp(-0.693), 0, 12)[0]
E.num("s06p01c", rnd(np.exp(-H1), 3), "ハザード h0(t)e^β を数値積分して exp(-H1(12))")
E.check("s06p01c", abs(np.exp(-0.693) - 0.500) < 5e-4, "e^{-0.693} の値")

# ---- 大問2 ガンマの比はベータ
th2 = 1.7
X = rng.gamma(2, th2, size=1_000_000)
Y = rng.gamma(3, th2, size=1_000_000)
U, V = X + Y, Y / (X + Y)
ks = stats.kstest(V, stats.beta(3, 2).cdf).statistic
ksu = stats.kstest(U, stats.gamma(5, scale=th2).cdf).statistic
cor = np.corrcoef(U, V)[0, 1]
cor2 = np.corrcoef(U ** 2, V ** 2)[0, 1]
E.check("s06p02a", ks < 0.003 and ksu < 0.003 and abs(cor) < 0.005 and abs(cor2) < 0.005,
        f"シミュレーション：V~Beta(3,2)（KS {ks:.4f}）、U~Ga(5,θ)（KS {ksu:.4f}）、相関 {cor:.4f}/{cor2:.4f}")
u_, v_ = sp.symbols("u v", positive=True)
th_ = sp.symbols("theta", positive=True)
fxy = lambda x, y: x * sp.exp(-x / th_) / th_ ** 2 * y ** 2 * sp.exp(-y / th_) / (2 * th_ ** 3)
J = sp.Matrix([u_ * (1 - v_), u_ * v_]).jacobian([u_, v_]).det()
fuv = sp.simplify(fxy(u_ * (1 - v_), u_ * v_) * sp.Abs(J))
target = stats.gamma  # noqa
prod_ = (u_ ** 4 * sp.exp(-u_ / th_) / (sp.factorial(4) * th_ ** 5)) * (12 * v_ ** 2 * (1 - v_))
E.check("s06p02a", sp.simplify(fuv - prod_) == 0, "sympy：同時密度 = Ga(5,θ) の密度 × Beta(3,2) の密度")
E.check("s06p02a", "Beta}(3,2)" in E.item("s06p02a")["choices"][3] and "独立である" in E.item("s06p02a")["choices"][3]
        and "尺度 $\\theta$" in E.item("s06p02a")["choices"][3], "正解の選択肢の文が導出と一致")
E.choice("s06p02b", f"${f(stats.beta(3, 2).sf(0.6), 3)}$", "scipy beta(3,2).sf(0.6)")
E.check("s06p02b", abs((V >= 0.6).mean() - stats.beta(3, 2).sf(0.6)) < 0.002, "シミュレーションの割合")

# ---- 大問3 推奨度スコア：行列形と、多項分布のシミュレーションで標準誤差を確認
a = np.array([1, 0, -1])
cnt = np.array([188, 140, 72])
ph = cnt / 400
Sig = np.diag(ph) - np.outer(ph, ph)
se = np.sqrt(a @ Sig @ a / 400)
E.num("s06p03a", rnd(a @ ph - stats.norm.ppf(0.975) * se, 3), "行列形 a'(diag(p)-pp')a/m と正確な 97.5% 点")
sim = rng.multinomial(400, ph, size=400_000) @ a / 400
E.check("s06p03a", abs(sim.std() - se) < 0.0003, f"シミュレーションの標準偏差 {sim.std():.5f} ≈ {se:.5f}")

# ---- 大問4 幾何分布
data = np.full(50, 4)  # 合計 200（十分統計量だけが効く）
nll = lambda q: -stats.geom(q).logpmf(data).sum()
res = optimize.minimize_scalar(nll, bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-10})
ph4 = res.x
# 期待フィッシャー情報量を定義どおり級数和で数値計算し、τ(p)=1/p の下限 τ'(p)^2/(n I(p)) を求める
ks_ = np.arange(1, 4000)
d2 = -1 / ph4 ** 2 - (ks_ - 1) / (1 - ph4) ** 2
I1 = -(stats.geom(ph4).pmf(ks_) * d2).sum()
E.num("s06p04a", rnd((1 / ph4 ** 2) ** 2 / (50 * I1), 3), f"数値最適化の MLE {ph4:.6f} と級数和の情報量 {I1:.4f}")
E.check("s06p04a", abs(stats.geom(0.25).var() / 50 - 0.24) < 1e-12, "標本平均の分散 V[X]/n が下限に一致（有効）")
G2 = 2 * (stats.geom(0.25).logpmf(data).sum() - stats.geom(0.2).logpmf(data).sum())
E.choice("s06p04b", f"$G^2={f(G2, 2)}$で、$3.841$未満なので$H_0$を棄却しない", f"scipy geom の対数尤度で G²={G2:.4f}")
E.check("s06p04b", abs(np.log(1.25) - 0.22314) < 5e-6 and abs(np.log(0.9375) + 0.06454) < 5e-6
        and abs(stats.chi2.ppf(0.95, 1) - 3.841) < 5e-4 and abs(stats.chi2.ppf(0.90, 1) - 2.706) < 5e-4, "問題文の対数・カイ二乗の点")
# スコア統計量・ワルド統計量を sympy の対数尤度の微分から求め直し、誤答の選択肢と一致し正解と異なることを確認
q_ = sp.symbols("q", positive=True)
ll = 50 * sp.log(q_) + 150 * sp.log(1 - q_)
U0 = sp.diff(ll, q_).subs(q_, sp.Rational(1, 5))
Iexp = 50 / (sp.Rational(1, 5) ** 2 * sp.Rational(4, 5))
score = float(U0 ** 2 / Iexp)
wald = float((sp.Rational(1, 4) - sp.Rational(1, 5)) ** 2 * (-sp.diff(ll, q_, 2)).subs(q_, sp.Rational(1, 4)))
E.check("s06p04b", f(score, 2) in E.item("s06p04b")["choices"][3] and f(wald, 2) in E.item("s06p04b")["choices"][1]
        and score < 3.841 and wald < 3.841 and f(G2, 2) not in (f(score, 2), f(wald, 2)),
        f"スコア統計量 {score:.3f}（選択肢3）・ワルド統計量 {wald:.3f}（選択肢1）は尤度比 {G2:.3f} と異なる")

# ---- 大問5
# (a) コーシー分布の標本中央値：順序統計量の正確な分布をシミュレーションで確認し、漸近式も scipy で
fm = stats.cauchy(scale=2).pdf(0)
se_med = 1 / (2 * np.sqrt(100) * fm)
E.choice("s06p05a", f"${f(se_med, 3)}$", "scipy cauchy.pdf(0) で 1/(2√n f)")
med = np.median(stats.cauchy(scale=2).rvs(size=(100_000, 100), random_state=rng), axis=1)
E.check("s06p05a", abs(med.std() - 0.3142) < 0.01, f"シミュレーションの中央値の標準偏差 {med.std():.4f}")
mn = stats.cauchy(scale=2).rvs(size=(20_000, 100), random_state=rng).mean(1)
E.check("s06p05a", abs(np.subtract(*np.quantile(mn, [0.75, 0.25])) / 2 - 2) < 0.1, "標本平均は元と同じ尺度 2 のコーシー分布（四分位範囲の半分が 2）")
# ブートストラップ：5^5 通りを総当たり
xs = [2, 3, 4, 6, 10]
means = [np.mean(c) for c in product(xs, repeat=5)]
E.choice("s06p05b", f"${f(np.var(means), 2)}$", "5^5=3125 通りの総当たり")

# ---- 大問6
E.check("s06p06a", abs(stats.t.ppf(0.975, 38) - 2.024) < 5e-4 and abs(stats.t.ppf(0.95, 38) - 1.686) < 5e-4, "t 分布の点")
rc = optimize.brentq(lambda r_: stats.t.sf(r_ * np.sqrt(38) / np.sqrt(1 - r_ ** 2), 38) * 2 - 0.05, 0.01, 0.99)
nearest("s06p06a", rc)
E.choice("s06p06a", f"${f(rc, 3)}$", f"p 値が 0.05 になる r を数値的に解く（{rc:.5f}）")
zz = []
for _ in range(40000):
    d_ = rng.normal(size=(40, 2))
    zz.append(abs(np.corrcoef(d_.T)[0, 1]))
E.check("s06p06a", abs(np.mean(np.array(zz) >= rc) - 0.05) < 0.004, f"無相関のシミュレーションで |r|≥境界の割合 {np.mean(np.array(zz) >= rc):.4f}")
xx = sp.symbols("x")
g = 0.4 * (200 + 6 * xx - sp.Rational(12, 100) * xx ** 2) - xx
xs = sp.solve(sp.diff(g, xx), xx)[0]
E.num("s06p06b", rnd(float(xs), 2), "sympy で微分して解く")
E.check("s06p06b", sp.diff(g, xx, 2) < 0, "2 階微分が負（最大）")

# ---- 大問7
grp = [np.array([12, 15, 18]), np.array([10, 13, 14]), np.array([8, 9, 11])]
Jv = sum((grp[i][:, None] > grp[j][None, :]).sum() for i in range(3) for j in range(i + 1, 3))
# 帰無仮説のもとでの J の正確な分布を、9 店の値の割り当てを総当たりして平均・分散を求める
allv = np.concatenate(grp)
Js = []
import itertools as _it
for c1 in _it.combinations(range(9), 3):
    rest = [k_ for k_ in range(9) if k_ not in c1]
    for c2 in _it.combinations(rest, 3):
        c3 = [k_ for k_ in rest if k_ not in c2]
        gg = [allv[list(c1)], allv[list(c2)], allv[c3]]
        Js.append(sum((gg[i][:, None] > gg[j][None, :]).sum() for i in range(3) for j in range(i + 1, 3)))
Js = np.array(Js)
Zj = (Jv - Js.mean()) / Js.std()
E.choice("s06p07a", f"${f(Zj, 2)}$", f"割り当て 1680 通りの総当たりで E[J]={Js.mean():.2f}, V[J]={Js.var():.3f}, J={Jv}")
E.check("s06p07a", (Js >= Jv).mean() < 0.05, f"正確な片側 p 値 {(Js >= Jv).mean():.4f} も 5% 未満")
# (b) 2^{4-1}：全 16 個の効果の列を作り、AB と同じ列になる効果（別名）を総当たりで探す。分解能は定義関係の最短の語の長さ
A_ = np.tile([-1, 1], 4); B_ = np.tile([-1, -1, 1, 1], 2); C_ = np.repeat([-1, 1], 4); D_ = A_ * B_ * C_
y7 = np.array([52, 61, 58, 55, 63, 57, 54, 68.0])
cols = {"A": A_, "B": B_, "C": C_, "D": D_}
words = {}
for k_ in range(1, 5):
    for w in _it.combinations("ABCD", k_):
        words["".join(w)] = np.prod([cols[c] for c in w], axis=0)
alias_AB = sorted(w for w, v in words.items() if w != "AB" and np.array_equal(v, words["AB"]))
defining = [w for w, v in words.items() if np.all(v == 1)]
resolution = min(len(w) for w in defining)
E.check("s06p07b", alias_AB == ["CD"] and defining == ["ABCD"] and resolution == 4,
        f"総当たり：AB の別名 {alias_AB}、定義関係 {defining}、分解能 {resolution}")
# 飽和モデル（A,B,C,D,AB,AC,AD）を最小二乗で当てはめ、AB の係数の 2 倍を効果とする
Xf = np.column_stack([np.ones(8), A_, B_, C_, D_, A_ * B_, A_ * C_, A_ * D_])
coef7 = np.linalg.lstsq(Xf, y7, rcond=None)[0]
E.choice("s06p07b", f"推定値は${f(2 * coef7[5], 1)}$。ABは交互作用CDと別名の関係にあり、この計画の分解能はIVである", "最小二乗の AB 係数の 2 倍")

# ---- 大問8 2 本のポアソン過程を別々に発生させて重ね合わせる
Ns = 2_000_000
ta = rng.exponential(60 / 6, size=Ns)    # アプリの次の到着（分）
tw = rng.exponential(60 / 18, size=Ns)   # Web の次の到着（分）
ev = (np.minimum(ta, tw) >= 5) & (ta < tw)
E.check("s06p08a", abs(ev.mean() - 0.25 * np.exp(-2)) < 0.0005, f"シミュレーション {ev.mean():.5f}")
E.choice("s06p08a", f"${f(integrate.quad(lambda u: (1 / 10) * np.exp(-u / 10) * np.exp(-u * 18 / 60), 5, np.inf)[0], 3)}$",
         "アプリが先に来て時刻が5分以上の確率を積分で")
E.check("s06p08a", all(abs(np.exp(-x_) - v_) < 5e-5 for x_, v_ in [(0.5, 0.6065), (1.5, 0.2231), (2, 0.1353)]), "e の値")

# ---- 大問9
# (a) 誤差を正規として対数モデルからシミュレーションし、期待値の比を求める
eps = rng.normal(0, 0.5, size=2_000_000)
Q0 = np.exp(3 - 1.8 * np.log(100) + eps).mean()
Q1 = np.exp(3 - 1.8 * np.log(90) + 0.6 + eps).mean()
E.num("s06p09a", rnd(0.9 ** -1.8 * np.exp(0.6), 2), "正確なべき乗と指数")
E.check("s06p09a", abs(Q1 / Q0 - 2.2026) < 0.01 and abs(0.9 ** -1.8 - 1.2088) < 5e-5 and abs(np.exp(0.6) - 1.8221) < 5e-5,
        f"シミュレーションの期待値の比 {Q1 / Q0:.4f} と問題文の値")
# (b) 実際のデータを作り、第 6 週を除いて当てはめ直した残差標準偏差から外部スチューデント化残差を求める。
#     別法：第 6 週だけ 1 のダミー変数を加えた回帰でのダミーの t 値（外部スチューデント化残差に一致する）
d6 = np.array([-5, -3, -1, 1, 3, 5.0])
x6 = 300 + 10 * d6
y6b = 50 - 0.3 * x6 + np.array([2.0, -1.6, -0.4, -0.4, -1.6, 2.0])
Xd = np.column_stack([np.ones(6), x6])
bh = np.linalg.lstsq(Xd, y6b, rcond=None)[0]
res = y6b - Xd @ bh
E.check("s06p09b", np.allclose(res, [2.0, -1.6, -0.4, -0.4, -1.6, 2.0]), "作ったデータの残差が問題文と一致")
b5 = np.linalg.lstsq(Xd[:5], y6b[:5], rcond=None)[0]
s6_ = np.sqrt(((y6b[:5] - Xd[:5] @ b5) ** 2).sum() / 3)
H = Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
t6v = res[5] / (s6_ * np.sqrt(1 - H[5, 5]))
E.num("s06p09b", rnd(t6v, 2), f"実際に当てはめ直した s_(6)={s6_:.4f} とハット行列の h66={H[5, 5]:.4f}")
Xdum = np.column_stack([Xd, np.eye(6)[:, 5]])
bd, rss, *_ = np.linalg.lstsq(Xdum, y6b, rcond=None)
s2d = ((y6b - Xdum @ bd) ** 2).sum() / 3
tdum = bd[2] / np.sqrt(s2d * np.linalg.inv(Xdum.T @ Xdum)[2, 2])
E.check("s06p09b", abs(tdum - t6v) < 1e-9, f"ダミー変数の t 値 {tdum:.4f} と一致")
E.check("s06p09b", abs(stats.t.ppf(0.975, 3) - 3.182) < 5e-4, "t(3) の上側 2.5% 点")

# ---- 大問10 回答者ごとにウェイトを付けた加重平均として計算（ウェイトは名簿の構成比/回答者の構成比）
nh = np.array([60, 200, 140])
yh = np.array([3.2, 3.6, 4.1])
Wp = np.array([0.3, 0.5, 0.2])
yi = np.repeat(yh, nh)
wi = np.repeat(Wp / (nh / nh.sum()), nh)
E.choice("s06p10a", f"${f((wi * yi).sum() / wi.sum(), 3)}$", "個票の加重平均")
E.check("s06p10a", abs(wi.sum() - 400) < 1e-9, "ウェイトの合計が回答者数に一致")

# ---- 大問11
eta = np.array([0.0, -1.2 + 0.8 * 2, -0.5 - 0.3 * 2])
pr = np.exp(eta) / np.exp(eta).sum()
E.choice("s06p11a", f"${f(pr[1], 3)}$", "正確な exp によるソフトマックス")
E.check("s06p11a", abs(np.exp(0.4) - 1.4918) < 5e-5 and abs(np.exp(-1.1) - 0.3329) < 5e-5, "問題文の指数の値")

# ---- 大問12
# (a) 特異値 6,3,2,1 をもつ中心化行列を実際に作り、階数2近似の残差から割合を求める
Uq, _ = np.linalg.qr(rng.normal(size=(30, 4)))
Uq -= Uq.mean(0)
Uq, _ = np.linalg.qr(Uq)
Vq, _ = np.linalg.qr(rng.normal(size=(4, 4)))
Xm = Uq @ np.diag([6, 3, 2, 1.0]) @ Vq.T
uu, ss, vt = np.linalg.svd(Xm, full_matrices=False)
X2 = uu[:, :2] @ np.diag(ss[:2]) @ vt[:2]
expl = 1 - ((Xm - X2) ** 2).sum() / (Xm ** 2).sum()
E.check("s06p12a", np.allclose(Xm.mean(0), 0, atol=1e-12) and np.allclose(ss, [6, 3, 2, 1]), "列平均0・特異値 6,3,2,1 の行列を作成")
E.choice("s06p12a", f"${f(100 * expl, 1)}\\%$", "階数2近似の残差平方和から")
ev = np.linalg.eigvalsh(np.cov(Xm.T))[::-1]
E.check("s06p12a", abs(ev[:2].sum() / ev.sum() - expl) < 1e-12, "共分散行列の固有値による累積寄与率と一致")
# (b) 混合正規の負担率を scipy の正規密度で（与えた φ の値も確認）
num1 = 0.6 * stats.norm(2, 1).pdf(3.2); num2 = 0.4 * stats.norm(5, 1).pdf(3.2)
E.choice("s06p12b", f"${f(num1 / (num1 + num2), 3)}$", "scipy norm.pdf による事後確率")
E.check("s06p12b", abs(stats.norm.pdf(1.2) - 0.1942) < 5e-5 and abs(stats.norm.pdf(1.8) - 0.0790) < 5e-5, "φ の値")
lab = rng.random(2_000_000) < 0.6
xx_ = np.where(lab, rng.normal(2, 1, 2_000_000), rng.normal(5, 1, 2_000_000))
near = np.abs(xx_ - 3.2) < 0.02
E.check("s06p12b", abs(lab[near].mean() - 0.787) < 0.02, f"シミュレーションで x≈3.2 の会員のクラスター1の割合 {lab[near].mean():.3f}")
# (c) 個票を作ってジニ不純度を直接計算
yv = np.array([1] * 20 + [0] * 20 + [1] * 10 + [0] * 50)
open_ = np.array([1] * 40 + [0] * 60)
gini = lambda v: 1 - np.mean(v) ** 2 - (1 - np.mean(v)) ** 2
dec = gini(yv) - sum(np.mean(open_ == k_) * gini(yv[open_ == k_]) for k_ in (0, 1))
E.choice("s06p12c", f"${f(dec, 3)}$", "個票から不純度の減少量")

# ---- 大問13
tab = np.array([[30, 90], [20, 160]])
cy = stats.chi2_contingency(tab, correction=True)
E.choice("s06p13a", f"${f(cy.statistic, 2)}$", "scipy chi2_contingency(correction=True)")
q13 = E.item("s06p13a")["question"]
E.check("s06p13a", "| 計 | 50 | 250 | 300 |" in q13, "問題文の表の列合計が 50, 250, 300")
E.check("s06p13a", cy.statistic > 3.841 and abs(stats.chi2_contingency(tab, correction=False).statistic - 10) < 1e-9, "判定と補正なしの値")
# (b) 幾何分布の無記憶性：scipy geom の生存関数で条件付き確率、シミュレーションでも
gq = stats.geom(0.2)
E.choice("s06p13b", f"${f((gq.cdf(8) - gq.cdf(5)) / gq.sf(5), 3)}$", "scipy geom で (F(8)-F(5))/S(5)")
first = (rng.random((400_000, 60)) < 0.2).argmax(1) + 1
cond = first[first > 5]
E.check("s06p13b", abs((cond <= 8).mean() - 0.488) < 0.004, f"シミュレーション {(cond <= 8).mean():.4f}")
# (c) 合流点で条件づけた偏相関：シミュレーションで残差どうしの相関
Nn = 1_000_000
Xs_ = rng.normal(size=Nn); Ys_ = rng.normal(size=Nn); Cs_ = Xs_ + Ys_ + rng.normal(size=Nn)
rx = Xs_ - np.polyfit(Cs_, Xs_, 1) @ np.vstack([Cs_, np.ones(Nn)])
ry = Ys_ - np.polyfit(Cs_, Ys_, 1) @ np.vstack([Cs_, np.ones(Nn)])
pc = np.corrcoef(rx, ry)[0, 1]
E.check("s06p13c", abs(pc + 0.5) < 0.005, f"シミュレーションの偏相関 {pc:.4f}")
Sg = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 3.0]])
P_ = np.linalg.inv(Sg)
E.choice("s06p13c", f"${f(-P_[0, 1] / np.sqrt(P_[0, 0] * P_[1, 1]), 2)}$", "精度行列から偏相関 -p12/√(p11 p22)")

E.done()
