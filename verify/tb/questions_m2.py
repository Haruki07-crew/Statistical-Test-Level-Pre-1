"""教科書 修士2年（第52〜56章）：問題の検算（章ごとに結果を保存する）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qhead import Q, chosen  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

ng = 0


def run(q):
    global ng
    try:
        q.done()
    except SystemExit as e:
        ng += int(bool(e.code))


rng = np.random.default_rng(5200)

# ---------- 第52章 ----------
q = Q(52)
q.num("q52-concentration-inequality-1", 2 * np.exp(-2 * 400 * 0.05 ** 2))
xb = rng.random((200000, 400)).mean(1)
q.check("q52-concentration-inequality-1", (np.abs(xb - 0.5) >= 0.05).mean() <= 0.2707, "一様分布で実際の確率は上界以下")
em = lambda p: integrate.quad(lambda v: v * p * stats.norm.pdf(v) * stats.norm.cdf(v) ** (p - 1), -10, 12)[0]
q.check("q52-concentration-inequality-2", em(10000) < np.sqrt(2 * np.log(10000)) and abs(em(10000) - 3.85) < 0.005 and em(10000) / em(100) < 1.6
        and chosen(q, "q52-concentration-inequality-2") == "$\\sqrt{2\\log p}$ 程度", f"E max: p=100 で {em(100):.3f}、p=10000 で {em(10000):.3f}")
q.num("q52-high-dimensional-regression-1", 29 / (60 - 29 - 1))
errs = []
for _ in range(3000):
    X = rng.normal(size=(60, 29)); y = rng.normal(size=60)
    errs.append((np.linalg.lstsq(X, y, rcond=None)[0] ** 2).sum())
q.check("q52-high-dimensional-regression-1", abs(np.mean(errs) - 29 / 30) < 0.05, f"シミュレーション {np.mean(errs):.3f}")
X = rng.normal(size=(30, 50)); y = rng.normal(size=30); b = np.linalg.pinv(X) @ y
q.check("q52-high-dimensional-regression-2", np.allclose(X @ b, y), "p>n で残差 0（answer=true）")
q.num("q52-lasso-theory-1", np.sign(1.3) * max(abs(1.3) - 0.5, 0))
t52 = np.sqrt(2 * np.log(2 * 1000 / 0.05))
zm = np.abs(rng.normal(size=(20000, 1000))).max(1)
q.check("q52-concentration-inequality-3", abs(t52 - 4.60) < 0.005 and (zm > t52).mean() <= 0.05 and "4.60" in q.qs["q52-concentration-inequality-3"]["explanation"], f"t={t52:.3f}・超える割合 {(zm > t52).mean():.4f}")
Xf = rng.normal(size=(40, 6)); tr_ = np.trace(np.linalg.inv(Xf.T @ Xf))
ers = [((np.linalg.lstsq(Xf, rng.normal(size=40), rcond=None)[0]) ** 2).sum() for _ in range(20000)]
q.check("q52-high-dimensional-regression-3", abs(np.mean(ers) / tr_ - 1) < 0.03, f"固定した X で E‖β̂−β‖² / tr((XᵀX)⁻¹) = {np.mean(ers) / tr_:.3f}")
q.num("q52-lasso-theory-2", np.sqrt(2 * np.log(1000) / 400))
run(q)

# ---------- 第53章 ----------
q = Q(53)
q.num("q53-empirical-risk-minimization-1", np.sqrt((np.log(50) + np.log(40)) / 4000))
q.concept("q53-empirical-risk-minimization-2", "超過リスクの分解の定義から（近似誤差は F の拡大で減り、推定誤差の上界 log|F| は増える）")
xs = np.array([1.0, 2.0, 3.0])
iv2 = {tuple((xs[:2] >= a) & (xs[:2] <= b)) for a in np.arange(0, 4, 0.5) for b in np.arange(0, 4, 0.5)}
iv3 = {tuple((xs >= a) & (xs <= b)) for a in np.arange(0, 4, 0.5) for b in np.arange(0, 4, 0.5)}
q.check("q53-vc-dimension-1", len(iv2) == 4 and (True, False, True) not in iv3 and chosen(q, "q53-vc-dimension-1") == "2", "2 点は細分、3 点の (1,0,1) は不可")
q.concept("q53-empirical-risk-minimization-3", "3 つの差への分解と、ERM の定義・sup の定義だけで成り立つ（第53章の検算のシミュレーションで、超過リスクが 2×一様なずれ以下であることを確認済み）")
from math import comb  # noqa: E402
q.num("q53-vc-dimension-2", sum(comb(10, i) for i in range(4)))
run(q)

# ---------- 第54章 ----------
q = Q(54)
q.num("q54-variational-inference-1", np.sqrt(1 - 0.8 ** 2))
# 平均場の最適解の分散が 1−ρ² であることを KL の数値最小化で確かめる
from scipy.optimize import minimize  # noqa: E402
Sg = np.array([[1, 0.8], [0.8, 1.0]]); Si = np.linalg.inv(Sg)
kl = lambda t: 0.5 * (np.trace(Si @ np.diag(np.exp(t))) - 2 + np.log(np.linalg.det(Sg) / np.exp(t).prod()))
q.check("q54-variational-inference-1", np.allclose(np.exp(minimize(kl, [0.0, 0.0]).x), 0.36, atol=1e-4), "KL の数値最小化で分散 0.36")
# ELBO ≤ log p(x)：正規・正規モデルでいろいろな q について確かめる
xo = 1.3  # x | θ ~ N(θ,1)、θ ~ N(0,1) → p(x)=N(0,2)
logpx = stats.norm.logpdf(xo, 0, np.sqrt(2))
ok = True
for m_, s_ in [(0, 1), (0.65, np.sqrt(0.5)), (2, 0.3), (-1, 2)]:
    elbo = -0.5 * np.log(2 * np.pi) - 0.5 * ((xo - m_) ** 2 + s_ ** 2) - 0.5 * np.log(2 * np.pi) - 0.5 * (m_ ** 2 + s_ ** 2) + 0.5 * np.log(2 * np.pi * np.e * s_ ** 2)
    ok &= elbo <= logpx + 1e-12
elbo_opt = -0.5 * np.log(2 * np.pi) - 0.5 * ((xo - 0.65) ** 2 + 0.5) - 0.5 * np.log(2 * np.pi) - 0.5 * (0.65 ** 2 + 0.5) + 0.5 * np.log(2 * np.pi * np.e * 0.5)
q.check("q54-variational-inference-2", ok and abs(elbo_opt - logpx) < 1e-12, "ELBO ≤ log p(x)、事後分布で等号（answer=true）")
from hmc_sim import hmc  # noqa: E402
smp, acc = hmc(n=4000)
q.check("q54-hamiltonian-monte-carlo-1", acc < 1 and abs(np.corrcoef(smp.T)[0, 1] - 0.95) < 0.01 and chosen(q, "q54-hamiltonian-monte-carlo-1").startswith("時間を逆にたどれて"),
        f"採択率 {acc:.3f}（1 未満なので採択の判定は必要）・目標の相関を再現")
q.num("q54-gaussian-process-regression-1", 1 - 1 / 1.25)
f1 = rng.normal(size=10 ** 6); y1 = f1 + rng.normal(0, 0.5, 10 ** 6)
q.check("q54-gaussian-process-regression-1", abs((f1 - y1 / 1.25).var() - 0.2) < 0.002, "同時正規分布のシミュレーション")
Xo = np.array([0.2, 0.8]); yo = np.array([1.0, -1.0])
def gp(xs_, l):
    K = np.exp(-(Xo[:, None] - Xo[None, :]) ** 2 / (2 * l * l)) + 0.01 * np.eye(2)
    ks = np.exp(-(xs_ - Xo) ** 2 / (2 * l * l))
    return ks @ np.linalg.solve(K, yo), 1 - ks @ np.linalg.solve(K, ks)
m_s, v_s = gp(0.5, 0.02)
q.check("q54-gaussian-process-regression-2", abs(m_s) < 1e-6 and v_s > 0.999 and chosen(q, "q54-gaussian-process-regression-2").startswith("観測点の間で"), "ℓ=0.02 では間の点で平均 0・分散 1 に戻る")
q.num("q54-waic-1", -2 * (-50.3 - 3.2))
q.num("q54-dirichlet-process-1", sum(2 / (2 + i) for i in range(10)))
cnt = []
for _ in range(20000):
    c_ = 0
    for i in range(10):
        c_ += rng.random() < 2 / (2 + i)
    cnt.append(c_)
q.check("q54-dirichlet-process-1", abs(np.mean(cnt) - 4.040) < 0.02, "新しいクラスターを作る事象のシミュレーション")
xo3 = 0.4
lp3 = stats.norm.logpdf(xo3, 0, np.sqrt(2))
m3_, s3_ = -0.3, 1.1
elbo3 = -np.log(2 * np.pi) - 0.5 * ((xo3 - m3_) ** 2 + s3_ ** 2) - 0.5 * (m3_ ** 2 + s3_ ** 2) + 0.5 * np.log(2 * np.pi * np.e * s3_ ** 2)
kl3 = np.log(np.sqrt(0.5) / s3_) + (s3_ ** 2 + (m3_ - xo3 / 2) ** 2) / (2 * 0.5) - 0.5
q.check("q54-variational-inference-3", abs(elbo3 + kl3 - lp3) < 1e-10, "正規・正規モデルで log p(x) = ELBO + KL")
kk = lambda a, b: np.exp(-(a - b) ** 2 / (2 * 0.3 ** 2))
fs = rng.multivariate_normal([0, 0], [[1, kk(0.5, 0.2)], [kk(0.5, 0.2), 1]], 10 ** 6); ys3 = fs[:, 1] + rng.normal(0, 0.5, 10 ** 6)
bslope = np.cov(fs[:, 0], ys3)[0, 1] / ys3.var()
resid = fs[:, 0] - bslope * ys3
q.check("q54-gaussian-process-regression-3", abs(bslope - kk(0.5, 0.2) / 1.25) < 0.003 and abs(resid.var() - (1 - kk(0.5, 0.2) ** 2 / 1.25)) < 0.003, "同時正規のシミュレーションで平均の係数と分散")
th_, pm_, ep_ = 1.0, 0.0, 0.5
p1_ = pm_ - ep_ / 2 * th_; th1_ = th_ + ep_ * p1_; p2_ = p1_ - ep_ / 2 * th1_
q.num("q54-hamiltonian-monte-carlo-2", 0.5 * th1_ ** 2 + 0.5 * p2_ ** 2)
q.check("q54-hamiltonian-monte-carlo-2", abs(p1_ + 0.25) < 1e-12 and abs(th1_ - 0.875) < 1e-12 and abs(p2_ + 0.46875) < 1e-12, "途中の値 −0.25・0.875・−0.46875")
q.num("q54-dirichlet-process-2", 2 / (2 + 8))
q.check("q54-waic-2", chosen(q, "q54-waic-2").startswith("正則なモデルでデータが多ければ"), "第54章の WAIC の例で WAIC 62.79 と AIC 62.52 が近いことを確認済み")
run(q)

# ---------- 第55章 ----------
q = Q(55)
q.num("q55-potential-outcomes-framework-1", 8 / 10 + 6 / 10)
q.check("q55-potential-outcomes-framework-2", chosen(q, "q55-potential-outcomes-framework-2") == "正値性（重なり）", "定義")
from causal_sim import expit, gen  # noqa: E402
rr = np.random.default_rng(9)
U = rr.normal(size=200000); Xo2 = rr.normal(size=200000)
T = (rr.random(200000) < expit(0.8 * Xo2 + 1.0 * U)).astype(float)
Y = 2 * T + Xo2 + U + rr.normal(size=200000)
from causal_sim import logit_fit  # noqa: E402
eh = logit_fit(np.c_[np.ones(200000), Xo2], T)
w1, w0 = T / eh, (1 - T) / (1 - eh)
est = (w1 * Y).sum() / w1.sum() - (w0 * Y).sum() / w0.sum()
q.check("q55-propensity-score-1", est - 2 > 0.5, f"観測されない U があると、観測した X だけの傾向スコアでは偏る（推定 {est:.2f}、真の値 2）（answer=false）")
q.num("q55-propensity-score-2", 4 / np.sqrt((100 + 36) / 2))
q.num("q55-inverse-probability-weighting-1", 1 / (1 - 0.9))
d4 = [(1, 10, 0.5), (1, 8, 0.8), (0, 6, 0.5), (0, 5, 0.2)]
q.num("q55-inverse-probability-weighting-2", np.mean([t * y / e - (1 - t) * y / (1 - e) for t, y, e in d4]))
ha = sum(t * y / e for t, y, e in d4) / sum(t / e for t, y, e in d4) - sum((1 - t) * y / (1 - e) for t, y, e in d4) / sum((1 - t) / (1 - e) for t, y, e in d4)
q.check("q55-inverse-probability-weighting-2", abs(ha - 3.615) < 5e-4, f"正規化した型 {ha:.4f}")
q.check("q55-doubly-robust-estimator-1", chosen(q, "q55-doubly-robust-estimator-1").startswith("傾向スコアのモデルか結果のモデルの少なくとも一方"), "第55章の検算（4 つの場合の偏り）と同じ")
q.num("q55-difference-in-differences-1", (62 - 50) - (51 - 45))
q.check("q55-difference-in-differences-2", chosen(q, "q55-difference-in-differences-2").startswith("施策がなければ"), "定義")
q.check("q55-regression-discontinuity-design-1", chosen(q, "q55-regression-discontinuity-design-1").startswith("ちょうど 60 点付近"), "定義")
from causal_sim import aipw, outcome, ps as ps_  # noqa: E402
x1_, x2_, t_, y_, e_ = gen(n=300000, seed=31)
m0_, m1_ = outcome(x1_, x2_, t_, y_)
mu1_true = (1 + 2 + 0.5 * x1_ + 1.5 * x1_ + x2_ + 1.5 * x1_ ** 2).mean()
q.check("q55-potential-outcomes-framework-3", abs(m1_.mean() - mu1_true) < 0.02, f"E[E[Y|T=1,X]] {m1_.mean():.3f} ≈ E[Y(1)] {mu1_true:.3f}")
cellb = (e_ > 0.58) & (e_ < 0.62)
q.check("q55-propensity-score-3", abs(t_[cellb].mean() - 0.6) < 0.01 and abs(x1_[cellb & (t_ == 1)].mean() - x1_[cellb & (t_ == 0)].mean()) < 0.03, "e(X)≈0.6 の人の処置の割合 0.6・X1 の平均が 2 群でほぼ等しい")
q.check("q55-inverse-probability-weighting-3", abs((t_ * y_ / e_).mean() - mu1_true) < 0.03, f"E[TY/e(X)] {(t_ * y_ / e_).mean():.3f} ≈ E[Y(1)]")
from did_sim import did22, panel  # noqa: E402
tr_, yp_ = panel()
tr2_, yp2_ = panel(extra_trend=0.5)
q.check("q55-difference-in-differences-3", abs(did22(tr_, yp_) - 3) < 0.05 and abs(did22(tr2_, yp2_) - did22(tr_, yp_) - 1.5) < 1e-9, "平行トレンドで ATT=3 を再現、崩れると 1.5 偏る")
q.num("q55-doubly-robust-estimator-2", 5 - 3 + 1 * (7 - 5) / 0.4 - 0)
q.num("q55-regression-discontinuity-design-2", 2.0 / 0.4)
run(q)

# ---------- 第56章 ----------
q = Q(56)
q.num("q56-conformal-prediction-1", int(np.ceil(100 * 0.9)))
cov = []
for _ in range(20000):
    s = rng.normal(size=100)
    qh = np.sort(np.abs(s[:99]))[89]
    cov.append(abs(s[99]) <= qh)
q.check("q56-conformal-prediction-1", abs(np.mean(cov) - 0.9) < 0.006, f"被覆率 {np.mean(cov):.4f}")
from conformal_sim import run as crun  # noqa: E402
c, _, mc, _, _ = crun(reps=500)
q.check("q56-conformal-prediction-2", c[0] > 0.99 and c[-1] < 0.72 and abs(mc - 0.9) < 0.01, f"左端 {c[0]:.3f}・右端 {c[-1]:.3f}・全体 {mc:.3f}（answer=false）")
q.num("q56-semiparametric-efficiency-1", np.sqrt((1 / 0.5 + 1 / 0.5) / 400))
q.check("q56-semiparametric-efficiency-1", abs(np.sqrt(1 / 200 + 1 / 200) - 0.1) < 1e-12, "平均の差の標準偏差と一致")
q.check("q56-double-machine-learning-1", chosen(q, "q56-double-machine-learning-1").startswith("$Y$ と $D$ のそれぞれ"), "第56章の検算（直交しない推定は −0.51 偏る）と同じ")
ranks = []
for _ in range(20000):
    sc = rng.exponential(size=20)
    ranks.append((sc[:19] < sc[19]).sum() + 1)
q.check("q56-conformal-prediction-3", abs(np.mean(np.array(ranks) <= 18) - 0.9) < 0.006 and np.ptp(np.bincount(ranks)[1:] / 20000) < 0.01, "順位が一様・P(順位 ≤ 18) = 18/20")
q.num("q56-semiparametric-efficiency-2", 1 / 0.2 + 1 / 0.8)
tt = (rng.random((20000, 400)) < 0.2); yy2 = rng.normal(size=(20000, 400)) + 0.0 * tt
dm = np.array([yy2[i][tt[i]].mean() - yy2[i][~tt[i]].mean() for i in range(20000)])
q.check("q56-semiparametric-efficiency-2", abs(400 * dm.var() - 6.25) < 0.25, f"20% が処置の実験で n·V[平均の差] = {400 * dm.var():.2f}")
vv = np.array([0.5, -1.0, 1.5, -0.5]); uu_ = np.array([1.0, -0.8, 2.0, -0.2])
q.num("q56-double-machine-learning-2", (vv * uu_).sum() / (vv * vv).sum())
q.check("q56-double-machine-learning-2", abs(np.linalg.lstsq(vv[:, None], uu_, rcond=None)[0][0] - 1.1733) < 1e-4, "原点を通る回帰の係数と一致")
run(q)

# ---------- 第57章 ----------
q = Q(57)
t7 = np.array([1, 2, 4, 4, 7.0]); d7 = np.array([1, 0, 1, 1, 1])
ev = sorted(set(t7[d7 == 1]))
q.num("q57-counting-process-martingale-1", sum(((t7 == e) & (d7 == 1)).sum() / (t7 >= e).sum() for e in ev if e <= 4))
km7 = np.prod([1 - ((t7 == e) & (d7 == 1)).sum() / (t7 >= e).sum() for e in ev if e <= 4])
q.check("q57-counting-process-martingale-2", np.exp(-(1 / 5 + 2 / 3)) >= km7, f"e^(-Λ̂)={np.exp(-(1 / 5 + 2 / 3)):.4f} ≥ KM={km7:.4f}（answer=true）")
q.num("q57-competing-risks-1", 0.3 / 0.4)
a1_, a2_ = rng.exponential(1 / 0.3, 10 ** 6), rng.exponential(1 / 0.1, 10 ** 6)
q.check("q57-competing-risks-1", abs((a1_ < a2_).mean() - 0.75) < 0.002, "シミュレーション")
q.check("q57-competing-risks-2", chosen(q, "q57-competing-risks-2").startswith("他の原因で亡くなる人がいない仮想の世界"), "第57章の検算（1−KM 0.393 > 累積発生関数 0.259）と同じ")
run(q)
print(f"修士2年（第52〜57章）の問題の検算: NG のある章 {ng}")
sys.exit(1 if ng else 0)
