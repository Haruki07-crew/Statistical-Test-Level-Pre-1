"""ダブル機械学習のページの例：部分線形モデル Y = θD + g(X) + ε、D = m(X) + v（θ = 1）。verify/tb/ch56.py から使う。

g(x) = 2 sin(2x₁) + x₂²、m(x) = tanh(2x₁) + 0.5x₂、X ~ U(−2,2)²、ε, v ~ N(0,1)。学習器は k 近傍回帰（k = 25）。
"""
import numpy as np

TH = 1.0


def gen(rng, n):
    x = rng.uniform(-2, 2, (n, 2))
    m = np.tanh(2 * x[:, 0]) + 0.5 * x[:, 1]
    d = m + rng.normal(size=n)
    y = TH * d + 2 * np.sin(2 * x[:, 0]) + x[:, 1] ** 2 + rng.normal(size=n)
    return x, d, y


def knn(xtr, ytr, xte, k=25, same=False):
    d2 = ((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(2)
    idx = np.argsort(d2, axis=1)[:, :k]
    return ytr[idx].mean(1)


def ols_linear(x, d, y):
    Z = np.c_[np.ones(len(d)), d, x]
    return np.linalg.lstsq(Z, y, rcond=None)[0][1]


def dml(x, d, y, cross=True, K=2, rng=None, half=False):
    n = len(d)
    lhat, mhat = np.empty(n), np.empty(n)
    if cross:
        fold = rng.permutation(n) % K
        for f in range(K):
            tr, te = fold != f, fold == f
            lhat[te] = knn(x[tr], y[tr], x[te])
            mhat[te] = knn(x[tr], d[tr], x[te])
    else:
        lhat, mhat = knn(x, y, x), knn(x, d, x)
    u, v = y - lhat, d - mhat
    if half:  # D を残差化しない（直交化しない）推定：Σ D(Y − ℓ̂(X)) / Σ D²
        return (d * u).sum() / (d * d).sum(), np.nan
    th = (v * u).sum() / (v * v).sum()
    psi = (u - th * v) * v
    se = np.sqrt(np.mean(psi ** 2) / np.mean(v * v) ** 2 / n)
    return th, se


def run(reps=200, n=1000, seed=5600):
    rng = np.random.default_rng(seed)
    out = {"ols": [], "dml": [], "dml_se": [], "nocross": [], "half": []}
    for _ in range(reps):
        x, d, y = gen(rng, n)
        out["ols"].append(ols_linear(x, d, y))
        th, se = dml(x, d, y, True, rng=rng)
        out["dml"].append(th); out["dml_se"].append(se)
        out["nocross"].append(dml(x, d, y, False)[0])
        out["half"].append(dml(x, d, y, True, rng=rng, half=True)[0])
    return {k: np.array(v) for k, v in out.items()}
