"""カルマンフィルタのページの図（ローカルレベルモデル）のデータと、フィルタ・平滑化の計算。verify/tb/ch51.py から使う。"""
import numpy as np


def simulate(T=40, se2=1.0, sh2=0.25, seed=51):
    rng = np.random.default_rng(seed)
    mu = 10 + np.cumsum(rng.normal(0, np.sqrt(sh2), T))
    y = mu + rng.normal(0, np.sqrt(se2), T)
    return mu, y


def filt(y, se2=1.0, sh2=0.25, a0=10.0, P0=1.0):
    a, P = a0, P0
    ap_, Pp_, af, Pf, ll = [], [], [], [], 0.0
    for yt in y:
        ap, Pp = a, P + sh2
        F = Pp + se2
        v = yt - ap
        K = Pp / F
        a, P = ap + K * v, (1 - K) * Pp
        ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        ap_.append(ap); Pp_.append(Pp); af.append(a); Pf.append(P)
    return np.array(ap_), np.array(Pp_), np.array(af), np.array(Pf), ll


def smooth(af, Pf, Pp):
    s = af.copy()
    for t in range(len(af) - 2, -1, -1):
        J = Pf[t] / Pp[t + 1]
        s[t] = af[t] + J * (s[t + 1] - af[t])
    return s
