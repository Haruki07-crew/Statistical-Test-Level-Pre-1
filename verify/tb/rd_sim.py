"""回帰不連続デザインのページの例：入学試験の点数 x（閾値 60 点以上で奨学金）と 1 年後の成績 y。verify/tb/ch55.py から使う。"""
import numpy as np


def data(seed=557, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(30, 90, n)
    d = (x >= 60).astype(float)
    y = 40 + 0.4 * (x - 60) + 0.001 * (x - 60) ** 3 + 4.0 * d + rng.normal(0, 5, n)
    return x, d, y


def local_linear(x, y, c=60.0, h=10.0):
    """閾値の両側で三角カーネルの重み付き直線を当て、閾値での値の差を返す。"""
    est = []
    for side in (x >= c, x < c):
        m = side & (np.abs(x - c) <= h)
        w = 1 - np.abs(x[m] - c) / h
        X = np.c_[np.ones(m.sum()), x[m] - c]
        W = X.T * w
        est.append(np.linalg.solve(W @ X, W @ y[m])[0])
    return est[0] - est[1]


def global_linear(x, y, c=60.0):
    d = (x >= c).astype(float)
    X = np.c_[np.ones_like(x), d, x - c]
    return np.linalg.lstsq(X, y, rcond=None)[0][1]
