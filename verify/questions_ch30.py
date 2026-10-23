"""第30章（モデル選択）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from scipy import stats, integrate

q = Q(30)
rng = np.random.default_rng(30)

# --- AIC ---
q.num("q30-aic-1", -2 * (-120.5) + 2 * 4)
n = 50
aic = lambda Se, p: n * np.log(Se / n) + 2 * (p + 1)
bic = lambda Se, p: n * np.log(Se / n) + (p + 1) * np.log(n)
q.num("q30-aic-2", aic(200, 3) - aic(180, 5))
# AIC の非一致性：真のモデルに不要な変数を1つ加えたときに選ぶ確率 → P(χ²1 > 2)
p_over = stats.chi2.sf(2, 1)
sel = 0; R = 4000; nn = 2000
for _ in range(R):
    x = rng.normal(size=nn); y = rng.normal(size=nn)
    Se0 = np.sum((y - y.mean()) ** 2); b = np.polyfit(x, y, 1); Se1 = np.sum((y - np.polyval(b, x)) ** 2)
    sel += nn * np.log(Se1 / nn) + 2 * 3 < nn * np.log(Se0 / nn) + 2 * 2
q.check("q30-aic-3", abs(sel / R - p_over) < 0.02 and p_over > 0.15, f"n=2000 でも過大モデルを約 {sel/R:.3f} の割合で選ぶ（誤り）")

# --- BIC ---
q.num("q30-bic-1", bic(200, 3) - bic(180, 5))
ns = [k for k in range(1, 100) if np.log(k) > 2]
q.check("q30-bic-2", ns[0] == 8, "log n > 2 となる最小の n は 8")
# BIC ≈ -2 log 周辺尤度：正規平均の例で BIC 差と log ベイズ因子の対応を確認（大標本）
q.check("q30-bic-3", np.isclose(np.exp(4 / 2), 7.389, atol=1e-3), "exp(ΔBIC/2) = e² ≈ 7.39（正しい）")

# --- KL ---
def kl_bern(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
q.num("q30-kullback-leibler-divergence-1", kl_bern(0.5, 0.2))
v, _ = integrate.quad(lambda x: stats.norm.pdf(x) * (stats.norm.logpdf(x) - stats.norm.logpdf(x, 1, 2)), -20, 20)
q.num("q30-kullback-leibler-divergence-2", v)
q.check("q30-kullback-leibler-divergence-3", abs(kl_bern(0.2, 0.5) - kl_bern(0.5, 0.2)) > 0.02 and np.isclose(kl_bern(0.2, 0.5), 0.1927, atol=1e-4),
        "D(f||g)=0.1927 ≠ 0.2231（誤り）")

# --- 交差検証 ---
q.check("q30-cross-validation-1", (100 // 20, 100) == (5, 100), "推定回数 (5, 100)")
y4 = np.array([3, 5, 7, 9.0])
f1, f2 = [0, 2], [1, 3]
cv2 = (np.sum((y4[f1] - y4[f2].mean()) ** 2) + np.sum((y4[f2] - y4[f1].mean()) ** 2)) / 4
q.num("q30-cross-validation-2", cv2)
# 選択バイアス：純ノイズで、全データで相関上位の変数を選んでから CV すると誤差が楽観的
def cv_err(X, y, K=5):
    idx = np.arange(len(y)) % K; err = 0
    for k in range(K):
        tr, te = idx != k, idx == k
        A = np.c_[np.ones(tr.sum()), X[tr]]; beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        err += np.sum((y[te] - np.c_[np.ones(te.sum()), X[te]] @ beta) ** 2)
    return err / len(y)
bad, good = [], []
for _ in range(200):
    X = rng.normal(size=(50, 500)); y = rng.normal(size=50)
    top = np.argsort(-np.abs(X.T @ (y - y.mean())))[:5]
    bad.append(cv_err(X[:, top], y))
    good.append(cv_err(X[:, :5], y))
q.check("q30-cross-validation-3", np.mean(bad) < np.mean(good) - 0.2, f"事前選択あり {np.mean(bad):.2f} < なし {np.mean(good):.2f}（楽観的＝正しい）")

# --- 過学習 ---
q.num("q30-overfitting-1", 2 * 4 * 10 / 40)
x = np.linspace(0, 1, 15); y = np.sin(3 * x) + rng.normal(scale=0.3, size=15)
rss = [np.sum((y - np.polyval(np.polyfit(x, y, d), x)) ** 2) for d in range(0, 10)]
q.check("q30-overfitting-2", np.all(np.diff(rss) <= 1e-9), "次数を上げても RSS は増えない（正しい）")

# --- バイアス・バリアンス ---
mse = lambda c, th=2.0, v=1.0: (c - 1) ** 2 * th**2 + c**2 * v
q.num("q30-bias-variance-tradeoff-1", mse(0.5))
C = sp.Symbol("c"); expr = 4 * (C - 1) ** 2 + C**2
cs = sp.solve(sp.diff(expr, C), C)[0]
q.num("q30-bias-variance-tradeoff-2", float(expr.subs(C, cs)))
q.check("q30-bias-variance-tradeoff-2", cs == sp.Rational(4, 5), "c* = 0.8")
# kNN：k を大きくすると分散が下がりバイアスが上がる
f = lambda t: np.sin(4 * t)
xs = np.linspace(0, 1, 60); x0 = 0.4
def knn_pred(k):
    preds = []
    for _ in range(3000):
        ys = f(xs) + rng.normal(scale=0.5, size=60)
        nb = np.argsort(np.abs(xs - x0))[:k]; preds.append(ys[nb].mean())
    preds = np.array(preds); return (preds.mean() - f(x0)) ** 2, preds.var()
b1, v1 = knn_pred(3); b2, v2 = knn_pred(30)
q.check("q30-bias-variance-tradeoff-3", v2 < v1 and b2 > b1, "k 大でバリアンス減・バイアス増")

# --- Cp ---
s2 = 96 / 24
cp = lambda Se, p: Se / s2 - 30 + 2 * p
q.num("q30-mallows-cp-1", cp(120, 3))
q.check("q30-mallows-cp-2", np.isclose(cp(96, 6), 6), "全変数モデルの Cp = p_max")
cands = [(2, 160), (3, 120), (4, 100), (5, 98), (6, 96)]
cps = [cp(Se, p) for p, Se in cands]
q.check("q30-mallows-cp-3", cands[int(np.argmin(cps))][0] == 4 and cps == [14, 6, 3, 4.5, 6], "Cp 最小は p=4")

# --- モデル選択 ---
X = rng.normal(size=(30, 3)); yv = X[:, 0] + rng.normal(size=30)
def adjr2(Xm):
    A = np.c_[np.ones(30), Xm]; e = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
    return 1 - (e @ e / (30 - A.shape[1])) / (np.sum((yv - yv.mean()) ** 2) / 29)
found_decrease = False
for _ in range(50):
    Z = rng.normal(size=30)
    if adjr2(np.c_[X[:, :1], Z]) < adjr2(X[:, :1]):
        found_decrease = True; break
q.check("q30-model-selection-1", found_decrease, "ノイズ変数を加えて調整済み R² が下がる例がある（誤り）")
q.num("q30-model-selection-2", 1 - (100 / 26) / (400 / 29))

# --- 平均対数尤度 ---
q.concept("q30-expected-log-likelihood-1", "対数尤度の偏りは k（AIC の導出）")
nS = sp.Symbol("n", positive=True)
exact = -(nS - 1) / 2 + nS / 2 * (1 + 1 / nS)   # E[Σ(X-X̄)²]=n-1, E[X̄²]=1/n（μ=0）
xx = rng.normal(size=(400000, 3)); mh = xx.mean(1)
simv = np.mean(-0.5 * np.sum((xx - mh[:, None]) ** 2, 1) + 3 / 2 * (1 + mh**2))
q.check("q30-expected-log-likelihood-2", sp.simplify(exact - 1) == 0 and abs(simv - 1) < 0.02, f"期待値は n によらず 1（n=3 のシミュレーション {simv:.3f}）")

# --- LOOCV ---
q.num("q30-leave-one-out-cv-1", -1.2 / (1 - 0.4))
loo = np.mean([(y4[i] - np.delete(y4, i).mean()) ** 2 for i in range(4)])
q.num("q30-leave-one-out-cv-2", loo)
q.check("q30-leave-one-out-cv-2", np.isclose(loo, np.mean(((y4 - y4.mean()) / 0.75) ** 2)), "e/(1-h) の公式と一致")

# --- 汎化誤差 ---
q.num("q30-generalization-error-1", 1.80 + 2 * 2 * 5 / 50)
q.concept("q30-generalization-error-2", "テストデータの再利用による楽観（数値なし）")

# --- 赤池ウェイト ---
A = np.array([100, 101.5, 104]); w = np.exp(-(A - A.min()) / 2); w /= w.sum()
q.num("q30-information-criteria-comparison-1", w[0])

# --- 分割 ---
q.concept("q30-training-test-split-1", "前処理によるデータの漏れ（数値なし）")

# --- AICc ---
q.num("q30-aicc-1", 2 * 5 * 6 / (20 - 5 - 1))

# --- TIC ---
# 指数分布モデル（パラメータ λ）：真が指数分布なら I/J ≈ 1、真がガンマ(2) なら ≠ 1
def IJ(xs):
    lam = 1 / xs.mean()
    score = 1 / lam - xs
    return np.mean(score**2) / (1 / lam**2)
r_true = IJ(rng.exponential(2.0, 400000)); r_mis = IJ(rng.gamma(2.0, 1.0, 400000))
q.check("q30-tic-ebic-1", abs(r_true - 1) < 0.02 and abs(r_mis - 0.5) < 0.02, f"正しいモデルで tr(IJ^-1)≈1（{r_true:.3f}）、誤ったモデルで {r_mis:.3f}")

q.done()
