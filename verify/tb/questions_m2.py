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
run(q)

# ---------- 第53章 ----------
q = Q(53)
q.num("q53-empirical-risk-minimization-1", np.sqrt((np.log(50) + np.log(40)) / 4000))
q.concept("q53-empirical-risk-minimization-2", "超過リスクの分解の定義から（近似誤差は F の拡大で減り、推定誤差の上界 log|F| は増える）")
xs = np.array([1.0, 2.0, 3.0])
iv2 = {tuple((xs[:2] >= a) & (xs[:2] <= b)) for a in np.arange(0, 4, 0.5) for b in np.arange(0, 4, 0.5)}
iv3 = {tuple((xs >= a) & (xs <= b)) for a in np.arange(0, 4, 0.5) for b in np.arange(0, 4, 0.5)}
q.check("q53-vc-dimension-1", len(iv2) == 4 and (True, False, True) not in iv3 and chosen(q, "q53-vc-dimension-1") == "2", "2 点は細分、3 点の (1,0,1) は不可")
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
run(q)
print(f"修士2年（第52〜56章）の問題の検算: NG のある章 {ng}")
sys.exit(1 if ng else 0)
