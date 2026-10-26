"""高次元の回帰・ラッソの理論のページの例のシミュレーション（座標降下法のラッソ）。verify/tb/ch52.py から使う。"""
import numpy as np


def lasso(X, y, lam, b0=None, it=500, tol=1e-9):
    """(1/2n)‖y − Xb‖² + lam‖b‖₁ を座標降下法で最小化する（切片なし）。"""
    n, p = X.shape
    b = np.zeros(p) if b0 is None else b0.copy()
    cn = (X ** 2).sum(0) / n
    r = y - X @ b
    for _ in range(it):
        mx = 0.0
        for j in range(p):
            old = b[j]
            zj = X[:, j] @ r / n + cn[j] * old
            new = np.sign(zj) * max(abs(zj) - lam, 0) / cn[j]
            if new != old:
                r -= X[:, j] * (new - old)
                b[j] = new
                mx = max(mx, abs(new - old))
        if mx < tol:
            break
    return b


def errors(p, n=100, s=5, beta=1.0, sigma=1.0, reps=100, seed=52):
    """推定誤差 ‖b̂ − β‖² の平均：最小二乗（p ≥ n は最小ノルム解）・ラッソ（λ = σ√(2 log p / n)）・真の変数だけの最小二乗。"""
    rng = np.random.default_rng(seed + p)
    out = np.zeros(3)
    lam = sigma * np.sqrt(2 * np.log(p) / n)
    for _ in range(reps):
        X = rng.normal(size=(n, p))
        b = np.zeros(p)
        b[:s] = beta
        y = X @ b + rng.normal(0, sigma, n)
        bo = np.linalg.pinv(X) @ y
        bl = lasso(X, y, lam)
        bor = np.zeros(p)
        bor[:s] = np.linalg.lstsq(X[:, :s], y, rcond=None)[0]
        out += [((bo - b) ** 2).sum(), ((bl - b) ** 2).sum(), ((bor - b) ** 2).sum()]
    return out / reps
