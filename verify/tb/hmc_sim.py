"""ハミルトニアン・モンテカルロ法のページの例（相関 0.95 の 2 変量正規分布）。verify/tb/ch54.py から使う。"""
import numpy as np

RHO = 0.95
S = np.array([[1, RHO], [RHO, 1.0]])
P = np.linalg.inv(S)  # 精度行列：U(θ) = θᵀPθ/2


def rwmh(n=20000, step=0.35, seed=54):
    rng = np.random.default_rng(seed)
    th = np.zeros(2)
    out, acc = np.empty((n, 2)), 0
    for i in range(n):
        prop = th + step * rng.normal(size=2)
        if np.log(rng.random()) < -0.5 * (prop @ P @ prop - th @ P @ th):
            th, acc = prop, acc + 1
        out[i] = th
    return out, acc / n


def hmc(n=20000, eps=0.12, L=15, seed=54):
    rng = np.random.default_rng(seed)
    th = np.zeros(2)
    out, acc = np.empty((n, 2)), 0
    for i in range(n):
        p0 = rng.normal(size=2)
        q, p = th.copy(), p0 - 0.5 * eps * (P @ th)
        for k in range(L):
            q = q + eps * p
            if k < L - 1:
                p = p - eps * (P @ q)
        p = p - 0.5 * eps * (P @ q)
        h0 = 0.5 * th @ P @ th + 0.5 * p0 @ p0
        h1 = 0.5 * q @ P @ q + 0.5 * p @ p
        if np.log(rng.random()) < h0 - h1:
            th, acc = q, acc + 1
        out[i] = th
    return out, acc / n


def acf(x, k):
    x = x - x.mean()
    return 1.0 if k == 0 else (x[:-k] @ x[k:]) / (x @ x)


def ess(x, kmax=1000):
    s = 0.0
    for k in range(1, kmax):
        r = acf(x, k)
        if r < 0.05:
            break
        s += r
    return len(x) / (1 + 2 * s)
