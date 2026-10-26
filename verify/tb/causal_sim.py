"""第55章（因果推論の方法）の例で共通に使う観察研究の模擬データと推定量。verify/tb/ch55.py から使う。

X1, X2 ~ N(0,1)、処置の確率 e(x) = expit(−0.3 + 0.8X1 + 0.6X2)、
結果 Y = 1 + (2 + 0.5X1)T + 1.5X1 + X2 + 1.5X1² + ε（ε ~ N(0,1)）。真の平均処置効果 ATE = 2。
"""
import numpy as np

expit = lambda z: 1 / (1 + np.exp(-z))


def gen(n=5000, seed=55):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    e = expit(-0.3 + 0.8 * x1 + 0.6 * x2)
    t = (rng.random(n) < e).astype(float)
    y = 1 + (2 + 0.5 * x1) * t + 1.5 * x1 + x2 + 1.5 * x1 ** 2 + rng.normal(size=n)
    return x1, x2, t, y, e


def logit_fit(Z, t, it=30):
    b = np.zeros(Z.shape[1])
    for _ in range(it):
        p = expit(Z @ b)
        b += np.linalg.solve((Z.T * (p * (1 - p))) @ Z, Z.T @ (t - p))
    return expit(Z @ b)


def ps(x1, x2, t, right=True):
    Z = np.c_[np.ones_like(x1), x1, x2] if right else np.c_[np.ones_like(x1), x2]
    return logit_fit(Z, t)


def outcome(x1, x2, t, y, right=True):
    """群ごとの回帰で μ₁(x)・μ₀(x) を予測する（正しいモデルは X1² を含む）。"""
    Z = np.c_[np.ones_like(x1), x1, x2, x1 ** 2] if right else np.c_[np.ones_like(x1), x1, x2]
    mu = []
    for g in (0, 1):
        b = np.linalg.lstsq(Z[t == g], y[t == g], rcond=None)[0]
        mu.append(Z @ b)
    return mu[0], mu[1]


def naive(t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def ipw(t, y, e, normalized=True):
    w1, w0 = t / e, (1 - t) / (1 - e)
    if normalized:
        return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()
    return np.mean(w1 * y - w0 * y)


def strat(t, y, e, k=5):
    q = np.quantile(e, np.linspace(0, 1, k + 1))
    idx = np.clip(np.searchsorted(q, e, side="right") - 1, 0, k - 1)
    est = 0.0
    for j in range(k):
        m = idx == j
        est += m.mean() * (y[m & (t == 1)].mean() - y[m & (t == 0)].mean())
    return est


def aipw(t, y, e, m0, m1):
    return np.mean(m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e))
