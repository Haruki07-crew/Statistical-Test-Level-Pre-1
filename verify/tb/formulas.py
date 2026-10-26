"""教科書の公式まとめ（content/tb/formulas.md）の検算。各公式を数値例・シミュレーション・別の計算で確かめる。"""
from math import comb, e

import numpy as np
from scipy import integrate, stats

from tbcheck import check, close, done, CONTENT

rng = np.random.default_rng(20261026)
text = (CONTENT / "formulas.md").read_text(encoding="utf-8")

S = "測度論的確率論"
check(S, "裾の公式（ガンマ分布）", close(integrate.quad(lambda t: stats.gamma(3, scale=2).sf(t), 0, np.inf)[0], 6, 1e-6))
x = rng.normal(size=10 ** 6)
check(S, "dQ/dP = e^{μx−μ²/2}（μ=0.7）で E_P[X dQ/dP] = 0.7", close((x * np.exp(0.7 * x - 0.245)).mean(), 0.7, 0.01))
u = rng.random(10 ** 6)
check(S, "単調収束：min(X, n) の期待値が E[X] に増加（指数分布）", all(np.diff([integrate.quad(lambda t: min(t, n) * np.exp(-t), 0, 50)[0] for n in (1, 2, 4, 8, 16)]) > 0)
      and close(integrate.quad(lambda t: min(t, 16) * np.exp(-t), 0, 60)[0], 1, 1e-6))

S = "漸近理論"
z = rng.normal(size=(20000, 400))
check(S, "X̄−μ = O_p(n^{-1/2})：√n(X̄) の SD ≈ 1", close((np.sqrt(400) * z.mean(1)).std(), 1, 0.02))
g = lambda a, b: b / a
mux, muy = 2.0, 3.0
Sg = np.array([[1.0, 0.6], [0.6, 2.25]])
grad = np.array([-muy / mux ** 2, 1 / mux])
xy = rng.multivariate_normal([mux, muy], Sg, (4000, 400))
R = xy[:, :, 1].mean(1) / xy[:, :, 0].mean(1)
check(S, "多変量デルタ法 ∇gᵀΣ∇g（比 Ȳ/X̄）", close(400 * R.var(), grad @ Sg @ grad, 0.05), (400 * R.var(), grad @ Sg @ grad))
p1, p2, n = 0.4, 0.35, 1000
mm = rng.multinomial(n, [p1, p2, 0.25], 200000) / n
check(S, "多項分布の差の分散 {p1+p2−(p1−p2)²}/n・共分散 −p1p2/n", close((mm[:, 0] - mm[:, 1]).var(), (p1 + p2 - (p1 - p2) ** 2) / n, 2e-5) and close(np.cov(mm[:, 0], mm[:, 1])[0, 1], -p1 * p2 / n, 5e-6))
check(S, "DKW：2e^{−2nε²} は KS の正確な裾以上（n=100, ε=0.12）", 2 * np.exp(-2 * 100 * 0.0144) >= stats.kstwo(100).sf(0.12))

S = "推定の理論・決定理論"
lp = stats.laplace.rvs(size=(20000, 2000), random_state=rng)
check(S, "中央値の漸近分散 1/(4f²)（ラプラスで 1、n=2000 のシミュレーション）", close(2000 * np.median(lp, 1).var(), 1.0, 0.06), 2000 * np.median(lp, 1).var())
Rk = lambda p, n, a, b: (n * p * (1 - p) + (a - b * p) ** 2) / (n + b) ** 2
xb = rng.binomial(12, 0.35, 10 ** 6)
check(S, "二項の推定量のリスクの式（n=12, α=2, β=3, p=0.35）", close((((xb + 2) / 15 - 0.35) ** 2).mean(), Rk(0.35, 12, 2, 3), 1e-4))
check(S, "ミニマックスのリスク 1/(4(√n+1)²)（n=25）", close(Rk(0.2, 25, 2.5, 5), 1 / (4 * 6 ** 2), 1e-12) and close(Rk(0.7, 25, 2.5, 5), 1 / 144, 1e-12))
X10 = rng.normal(size=(200000, 10))
check(S, "ジェームズ・スタインのリスク 2（θ=0）", close((((1 - 8 / (X10 ** 2).sum(1))[:, None] * X10) ** 2).sum(1).mean(), 2, 0.03))
check(S, "一様分布の UMVUE の分散 θ²/(n(n+2))（n=6）", close(((7 / 6 * rng.random((400000, 6)).max(1)) ).var(), 1 / 48, 3e-4))
f = lambda n, d: (2 * (n - 1) + (n - 1 - d) ** 2) / d ** 2
ds = np.linspace(3, 15, 12001)
check(S, "SS/d の MSE は d=n+1 で最小（n=8）", close(ds[np.argmin(f(8, ds))], 9, 2e-3))
xs_ = np.c_[np.ones(6), np.arange(1, 7.0)]
ys_ = np.array([1.2, 2.3, 2.9, 4.8, 4.1, 7.0])
bh = np.linalg.solve(xs_.T @ xs_, xs_.T @ ys_)
ee = ys_ - xs_ @ bh
XtXi = np.linalg.inv(xs_.T @ xs_)
hc0 = XtXi @ (xs_.T * ee ** 2) @ xs_ @ XtXi
hc0b = XtXi @ sum(ee[i] ** 2 * np.outer(xs_[i], xs_[i]) for i in range(6)) @ XtXi
check(S, "HC0 の式（行列の 2 つの書き方が一致）", np.allclose(hc0, hc0b))

S = "ノンパラメトリック推定と頑健統計"
check(S, "正規基準 1.06 ≈ (4/3)^{1/5}", close((4 / 3) ** 0.2, 1.06, 0.001))
k = 1.345
P = stats.norm.cdf(k) - stats.norm.cdf(-k)
E2 = P - 2 * k * stats.norm.pdf(k) + 2 * k * k * stats.norm.sf(k)
check(S, "フーバー k=1.345 の効率 95%", close(P * P / E2, 0.95, 5e-4))
hk = 0.02
Kh = lambda v: stats.norm.pdf(v / hk) / hk
m2 = integrate.quad(lambda v: Kh(v) ** 2 * stats.norm.pdf(v), -8, 8, points=[0])[0]
m1 = integrate.quad(lambda v: Kh(v) * stats.norm.pdf(v), -8, 8, points=[0])[0]
exact = (m2 - m1 ** 2) / 200
approx = 1 / (2 * np.sqrt(np.pi)) * stats.norm.pdf(0) / (200 * hk)
check(S, "KDE の分散 ≈ R(K)f(x)/(nh)（x=0, h=0.02, n=200 の正確な分散と比 3% 以内）", abs(exact / approx - 1) < 0.03, (exact, approx))
check(S, "KDE の偏り ≈ h²/2·f''(0)（h=0.1）", close(stats.norm.pdf(0, 0, np.sqrt(1 + 0.01)) - stats.norm.pdf(0), 0.01 / 2 * (-stats.norm.pdf(0)), 2e-4))

S = "混合モデルと相関のあるデータ"
uu = rng.normal(0, 2, (200000, 1)); yy = uu + rng.normal(0, 3, (200000, 5))
check(S, "BLUP の縮小係数 = E[u|ȳ] の傾き（n=5, σu²=4, σ²=9）", close(np.cov(uu[:, 0], yy.mean(1))[0, 1] / yy.mean(1).var(), 20 / 29, 0.005))
check(S, "デザイン効果 1+(m−1)ρ", close((uu[:, 0] + rng.normal(0, 3, (200000, 5)).mean(1)).var() / (13 / 5), 1 + 4 * 4 / 13, 0.02))

S = "時系列と状態空間モデル"
q = 0.5
P_ = 1.0
for _ in range(500):
    Pp = P_ + q; K = Pp / (Pp + 1); P_ = (1 - K) * Pp
check(S, "定常ゲイン (−q+√(q²+4q))/2（q=0.5）", close(K, (-q + np.sqrt(q * q + 4 * q)) / 2, 1e-12))
w, a, b = 0.05, 0.1, 0.85
check(S, "GARCH の予測の式を漸化式で確かめる", close(w + (a + b) * (w + (a + b) * 3.0), 1 + (a + b) ** 2 * (3.0 - 1), 1e-12) and close(w / (1 - a - b), 1, 1e-12))
A = np.array([[0.5, 0.2], [0.1, 0.6]]); c = np.array([1.0, 0.5])
yv = np.zeros(2)
for _ in range(500):
    yv = c + A @ yv
check(S, "VAR(1) の平均 (I−A)⁻¹c", np.allclose(yv, np.linalg.solve(np.eye(2) - A, c)))

S = "高次元統計と学習理論"
check(S, "ヘフディングの上界は正確な二項の裾以上（n=300, t=0.06）", 2 * np.exp(-2 * 300 * 0.0036) >= stats.binom.cdf(np.floor(300 * 0.44), 300, 0.5) + stats.binom.sf(np.ceil(300 * 0.56) - 1, 300, 0.5))
em = integrate.quad(lambda v: v * 500 * stats.norm.pdf(v) * stats.norm.cdf(v) ** 499, -10, 12)[0]
check(S, "E max ≤ √(2 log p)（p=500）", em < np.sqrt(2 * np.log(500)), em)
errs = []
for _ in range(2000):
    Xr = rng.normal(size=(40, 15)); errs.append((np.linalg.lstsq(Xr, rng.normal(size=40), rcond=None)[0] ** 2).sum())
check(S, "最小二乗の誤差 σ²p/(n−p−1)（n=40, p=15）", close(np.mean(errs), 15 / 24, 0.03))
check(S, "ザウアーの補題の 2 つ目の不等号（d=4）", all(sum(comb(n, i) for i in range(5)) <= (e * n / 4) ** 4 for n in range(4, 300)))

S = "ベイズ計算"
xo = 0.9
logpx = stats.norm.logpdf(xo, 0, np.sqrt(2))
m_, s_ = 0.2, 0.9
elbo = -np.log(2 * np.pi) - 0.5 * ((xo - m_) ** 2 + s_ ** 2) - 0.5 * (m_ ** 2 + s_ ** 2) + 0.5 * np.log(2 * np.pi * e * s_ ** 2)
kl = np.log(np.sqrt(0.5) / s_) + (s_ ** 2 + (m_ - xo / 2) ** 2) / (2 * 0.5) - 0.5
check(S, "log p(x) = ELBO + KL（正規・正規モデル）", close(elbo + kl, logpx, 1e-10))
yv1 = 1.5
check(S, "ガウス過程の事後平均・分散（1 点、k=1、σn²=0.25）", close(yv1 / 1.25, 1.2, 1e-12) and close(1 - 1 / 1.25, 0.2, 1e-12))

S = "因果推論と予測"
from causal_sim import aipw, gen, ipw, outcome, ps  # noqa: E402
x1, x2, t, y, ee_ = gen(n=200000, seed=8)
er = ps(x1, x2, t)
m0, m1 = outcome(x1, x2, t, y)
check(S, "IPW と AIPW が真の値 2 の近く（20 万人）", close(ipw(t, y, er), 2, 0.05) and close(aipw(t, y, er, m0, m1), 2, 0.02))
check(S, "識別の式 E[μ₁(X)−μ₀(X)] も 2 の近く", close(np.mean(m1 - m0), 2, 0.02))
check(S, "差の差の式の例（62−50）−（51−45）=6", (62 - 50) - (51 - 45) == 6)
check(S, "コンフォーマルの k/(n+1) は保証の範囲に入る（n=9〜1000）", all(0.9 <= np.ceil((n + 1) * 0.9) / (n + 1) <= 0.9 + 1 / (n + 1) + 1e-12 for n in range(9, 1001)))
check(S, "無作為化実験の効率限界 = 平均の差の分散（e=0.5、σ²=1、n=400）", close(np.sqrt(4 / 400), np.sqrt(1 / 200 + 1 / 200), 1e-12))

import re
import yaml
plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
titles = {t["title"] for c in plan["chapters"] for t in c["terms"]}
refs = re.findall(r"→ ([^\n]+)", text)
check("全体", f"→ の参照 {len(refs)} 件", all(r.strip() in titles for r in refs), [r for r in refs if r.strip() not in titles])
done("教科書の公式まとめ")
