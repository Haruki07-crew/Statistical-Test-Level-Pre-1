"""カーネル回帰の境界の偏りの次数を数値積分で調べる（問題の検算用）。"""
import numpy as np
from scipy import integrate, stats

m = lambda z: np.sin(2 * np.pi * z)


def bias0(h):
    K = lambda t: stats.norm.pdf(t / h) / h
    a0, a1, a2 = (integrate.quad(lambda t: K(t) * t ** j, 0, 1)[0] for j in (0, 1, 2))
    b0, b1 = (integrate.quad(lambda t: K(t) * t ** j * m(t), 0, 1)[0] for j in (0, 1))
    return b0 / a0, (a2 * b0 - a1 * b1) / (a0 * a2 - a1 ** 2)


def boundary_ratio():
    r1, r2 = bias0(0.04), bias0(0.02)
    return r1[0] / r2[0], r1[1] / r2[1]
