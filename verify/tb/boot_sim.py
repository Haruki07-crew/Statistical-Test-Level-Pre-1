"""ブートストラップの理論のページの例。verify/tb/ch46.py から使う。

(1) 指数分布（平均 1）の n 個の標本平均の 95% 信頼区間：正規近似・パーセンタイル法・ブートストラップ t 法の被覆率。
(2) 一様分布 U(0,θ) の最大値でブートストラップが失敗する例。
"""
import numpy as np
from scipy import stats


def coverage(n=20, reps=2000, B=999, seed=46):
    rng = np.random.default_rng(seed)
    cov = np.zeros(3)
    for _ in range(reps):
        x = rng.exponential(1, n)
        m, s = x.mean(), x.std(ddof=1)
        bs = x[rng.integers(0, n, (B, n))]
        bm = bs.mean(1)
        bt = (bm - m) / (bs.std(1, ddof=1) / np.sqrt(n))
        z = stats.t.isf(0.025, n - 1)
        ci = [(m - z * s / np.sqrt(n), m + z * s / np.sqrt(n)),
              tuple(np.quantile(bm, [0.025, 0.975])),
              (m - np.quantile(bt, 0.975) * s / np.sqrt(n), m - np.quantile(bt, 0.025) * s / np.sqrt(n))]
        cov += [lo <= 1 <= hi for lo, hi in ci]
    return cov / reps


def max_fail(n=50, B=20000, seed=461):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    bmax = x[rng.integers(0, n, (B, n))].max(1)
    return (bmax == x.max()).mean(), 1 - (1 - 1 / n) ** n
