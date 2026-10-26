"""差の差（DID）のページの例：200 店舗 × 6 期のパネルデータ（前半 3 期は処置前、後半 3 期で処置群に効果 3）。verify/tb/ch55.py から使う。"""
import numpy as np


def panel(seed=556, n=200, T=6, effect=3.0, extra_trend=0.0):
    rng = np.random.default_rng(seed)
    treat = (np.arange(n) < n // 2).astype(float)
    a = rng.normal(0, 3, n) + 5 * treat        # 店舗の固定効果（処置群はもともと 5 高い）
    lam = np.array([20, 21, 22.5, 23, 24.2, 25.0])  # 共通の時間効果
    t = np.arange(T)
    post = (t >= 3).astype(float)
    y = a[:, None] + lam[None, :] + effect * treat[:, None] * post[None, :] + extra_trend * treat[:, None] * t[None, :] + rng.normal(0, 1, (n, T))
    return treat, y


def group_means(treat, y):
    return y[treat == 1].mean(0), y[treat == 0].mean(0)


def did22(treat, y):
    m1, m0 = group_means(treat, y)
    return (m1[3:].mean() - m1[:3].mean()) - (m0[3:].mean() - m0[:3].mean())
