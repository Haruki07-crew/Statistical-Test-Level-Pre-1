"""模擬試験 第20回（set20）の数値。values() が {大問id: {名前: 値}} を返す。

題材：エネルギー・電力需要・再生可能エネルギー・環境測定。
問題文に示した分布の値（Φ・χ²・F・t の点、e^x、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from itertools import combinations
from math import sqrt


def r(x, d):
    """四捨五入（銀行丸めを避ける）。数値で返す。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return float(v) if d > 0 else int(v)


def s(x, d):
    """四捨五入した文字列（末尾の 0 も残す）。"""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def p01():
    """同時密度 f(x,y)=(2/3)(2x+y)（0<x<1, 0<y<1）"""
    c = Fr(2, 3)
    p_yx = c * Fr(2, 3)                 # (2/3)∫_0^1 ∫_0^y (2x+y) dx dy = (2/3)∫ 2y^2 dy
    med = (-2 + sqrt(52)) / 8           # 4x^2+2x-3=0
    medY = -1 + sqrt(2.5)
    return {
        "p_yx": s(p_yx, 3), "p_xy": s(1 - p_yx, 3), "w_half": "0.500", "w_noc": s(2 / 3, 3), "w_no2": s(1 / 3, 3),
        "med": r(med, 3), "med_s": s(med, 3), "med5": s(med, 5), "disc": s(sqrt(52), 4),
        "meanX": s(11 / 18, 3), "medY": s(medY, 3), "w_nomarg": s(sqrt(0.5), 3),
    }


def p02():
    p, pe = 0.9, 0.8
    par = 1 - (1 - p) ** 2
    ser = p * p
    r_e1 = par ** 2
    r_e0 = 1 - (1 - ser) ** 2
    R = pe * r_e1 + (1 - pe) * r_e0
    return {
        "p": p, "pe": pe, "qe": s(1 - pe, 1), "par": s(par, 2), "ser": s(ser, 2), "r_e1": s(r_e1, 4),
        "omser": s(1 - ser, 2), "omser2": s((1 - ser) ** 2, 4), "r_e0": s(r_e0, 4),
        "t1": s(pe * r_e1, 5), "t0": s((1 - pe) * r_e0, 5), "R": r(R, 3), "R_s": s(R, 3), "R5": s(R, 5),
    }


def p03():
    m, cv = 15, 0.75
    ln1c, ln_m, ln35 = 0.4463, 2.708, 3.555     # log(1.5625), log 15, log 35（問題文に示す）
    phi = {1.27: 0.8980, 1.50: 0.9332, 1.60: 0.9452, 1.78: 0.9625, 2.40: 0.9918}
    sig = sqrt(ln1c)
    mu = ln_m - ln1c / 2
    z = r((ln35 - mu) / sig, 2)
    z_cv = r((ln35 - (ln_m - cv ** 2 / 2)) / cv, 2)
    z_nohalf = r((ln35 - ln_m) / sig, 2)
    z_norm = r((35 - m) / (m * cv), 2)
    z_var = r((ln35 - mu) / ln1c, 2)
    for zz in (z, z_cv, z_nohalf, z_norm, z_var):
        assert zz in phi, zz
    # (b) ギブス・サンプリング
    muU, muV, sU, sV, rho = 2.5, 2.3, 0.6, 0.5, 0.8
    U0, V0, z1, z2 = 2.6, 2.0, 0.5, -1.0
    k = sqrt(1 - rho ** 2)

    def step(bu, bv, cu, cv_, uold=None):
        u1 = muU + bu * (V0 - muV) + cu * z1
        base = U0 if uold else u1
        v1 = muV + bv * (base - muU) + cv_ * z2
        return u1, v1
    U1, V1 = step(rho * sU / sV, rho * sV / sU, sU * k, sV * k)
    U1a, V1a = step(rho * sU / sV, rho * sV / sU, sU, sV)
    U1b, V1b = step(rho, rho, sU * k, sV * k)
    U1c, V1c = step(rho * sU / sV, rho * sV / sU, sU * k * k, sV * k * k)
    _, V1d = step(rho * sU / sV, rho * sV / sU, sU * k, sV * k, uold=True)
    return {
        "m": m, "cv": cv, "cv2": s(cv ** 2, 4), "opc": s(1 + cv ** 2, 4), "ln1c": s(ln1c, 4), "ln_m": s(ln_m, 3),
        "ln35": s(ln35, 3), "sig": s(sig, 4), "half": s(ln1c / 2, 5), "mu": s(mu, 5), "num": s(ln35 - mu, 5),
        "z": s(z, 2), "zraw": s((ln35 - mu) / sig, 4), "phiz": s(phi[z], 4), "ans": s(1 - phi[z], 3), "ans4": s(1 - phi[z], 4),
        "ph127": s(phi[1.27], 4), "ph150": s(phi[1.50], 4), "ph160": s(phi[1.60], 4), "ph178": s(phi[1.78], 4), "ph240": s(phi[2.40], 4),
        "w_cv": s(1 - phi[z_cv], 3), "w_nohalf": s(1 - phi[z_nohalf], 3), "w_norm": s(1 - phi[z_norm], 3), "w_var": s(1 - phi[z_var], 3),
        "z_cv": s(z_cv, 2), "z_nohalf": s(z_nohalf, 2), "z_norm": s(z_norm, 2), "sdm": s(m * cv, 2),
        "muU": muU, "muV": muV, "sU": sU, "sV": sV, "rho": rho, "U0": U0, "V0": V0, "z1": z1, "z2": s(z2, 1),
        "k": s(k, 1), "bU": s(rho * sU / sV, 2), "bV": s(rho * sV / sU, 4), "csU": s(sU * k, 2), "csV": s(sV * k, 2),
        "dV0": s(V0 - muV, 1), "tU1": s(rho * sU / sV * (V0 - muV), 3), "tU2": s(sU * k * z1, 2),
        "U1": s(U1, 3), "dU1": s(U1 - muU, 3), "tV1": s(rho * sV / sU * (U1 - muU), 3), "tV2": s(sV * k * z2, 2), "V1": s(V1, 3),
        "U1a": s(U1a, 3), "V1a": s(V1a, 3), "U1b": s(U1b, 3), "V1b": s(V1b, 3), "U1c": s(U1c, 3), "V1c": s(V1c, 3),
        "V1d": s(V1d, 3),
    }


def p04():
    loc, theta, n = 10, 3, 365
    ln365, mlog98, ln0202 = 5.900, 0.0202, -3.902   # ln 365, -ln 0.98, ln 0.0202（問題文に示す）
    ln002 = -3.912                                   # ln 0.02（解説の誤答例に使う）
    y = ln365 - ln0202
    x = loc + theta * y
    return {
        "loc": loc, "theta": theta, "n": n, "ln365": s(ln365, 3), "mlog98": s(mlog98, 4), "ln0202": s(ln0202, 3),
        "ln002": s(ln002, 3), "ln002a": s(-ln002, 3), "y": s(y, 3), "x": r(x, 1), "x3": s(x, 3), "xs": s(x, 1),
        "w_noloc": s(theta * y, 1), "w_day": s(loc - theta * ln002, 1),
    }


def p05():
    n, xb, sd = 16, 94.35, 0.80
    s2 = sd ** 2
    S = (n - 1) * s2
    t15, t16, z = 2.131, 2.120, 1.960
    half = t15 * sd / sqrt(n)
    lo = xb - half
    # (c) ルビーン検定（平均からの絶対偏差の一元配置）
    k, m = 3, 5
    zb = [0.30, 0.50, 0.85]
    ssw = 0.84
    zg = sum(zb) / k
    ssb = m * sum((v - zg) ** 2 for v in zb)
    Fl = (ssb / (k - 1)) / (ssw / (k * m - k))
    F212 = 3.885
    # (c) 母標準偏差既知の両側 z 検定の検出力
    delta, sig = 0.5, 0.80
    shift = delta / (sig / sqrt(n))
    phi = {0.54: 0.7054, 0.86: 0.8051, 1.34: 0.9099, 2.50: 0.9938}
    za, zb_ = r(shift - 1.96, 2), r(shift - 1.645, 2)
    zs, zn = r(delta / sig - 1.96, 2), r(shift, 2)
    assert za in phi and zb_ in phi and -zs in phi and zn in phi
    pw = phi[za]
    return {
        "delta": delta, "sig": s(sig, 2), "shift": s(shift, 2), "za": s(za, 2), "pw": s(pw, 3), "pw4": s(pw, 4),
        "ph054": phi[0.54], "ph086": phi[0.86], "ph134": phi[1.34], "ph250": phi[2.50],
        "w_one": s(phi[zb_], 3), "w_beta": s(1 - pw, 3), "w_nosqrt": s(1 - phi[-zs], 3), "w_nocrit": s(phi[zn], 3), "zs": s(zs, 2),
        **p05a(S),
        "n": n, "xb": s(xb, 2), "sd": s(sd, 2), "s2": s(s2, 2), "df": n - 1, "S": s(S, 1), "sqn": s(sqrt(n), 0),
        "t15": t15, "t16": t16, "z": s(z, 2), "se": s(sd / sqrt(n), 2), "half": s(half, 4), "lo": r(lo, 3), "lo_s": s(lo, 3),
        "lo5": s(lo, 4), "hi_s": s(xb + half, 3),
        "w_t16": s(xb - t16 * sd / sqrt(n), 3), "w_z": s(xb - z * sd / sqrt(n), 3), "w_pi": s(xb - t15 * sd * sqrt(1 + 1 / n), 3),
        "n1": n + 1, "nm1": n - 1, "nm2": n - 2, "np2": n + 2,
        "e17": s(S / (n + 1), 3), "e16": s(S / n, 3), "e15": s(S / (n - 1), 3), "e14": s(S / (n - 2), 3),
        "e18": s(S / (n + 2), 3),
        "m17": s(2 / (n + 1), 4), "m15": s(2 / (n - 1), 4), "m16": s((2 * n - 1) / n ** 2, 4),
        "k": k, "m": m, "z1": s(zb[0], 2), "z2": s(zb[1], 2), "z3": s(zb[2], 2), "zg": s(zg, 2),
        "dz1": s(zb[0] - zg, 2), "dz2": s(zb[1] - zg, 2), "dz3": s(zb[2] - zg, 2), "ssb": s(ssb, 3), "ssw": s(ssw, 2),
        "dfb": k - 1, "dfw": k * m - k, "msb": s(ssb / (k - 1), 4), "msw": s(ssw / (k * m - k), 2),
        "Fl": s(Fl, 2), "F212": F212, "w_nodf": s(ssb / (ssw / (k * m - k)), 2), "w_ss": s(ssb / ssw, 2), "w_inv": s(1 / Fl, 2),
    }


def p05a(S):
    """(a) σ の 95% 片側信頼上限。カイ二乗分布の点は問題文の値。"""
    c15l, c15ll, c15u, c16l = 7.261, 6.262, 25.00, 7.962
    U2 = S / c15l
    return {"c15l": c15l, "c15ll": c15ll, "c15u": s(c15u, 2), "c16l": c16l, "U2": s(U2, 4), "U2s": s(U2, 3),
            "U": r(sqrt(U2), 3), "U_s": s(sqrt(U2), 3), "U5": s(sqrt(U2), 4),
            "w_up": s(sqrt(S / c15u), 3), "w_two": s(sqrt(S / c15ll), 3), "w_16": s(sqrt(S / c16l), 3)}


def p06():
    n, T = 20, 100.0
    W = {2: 40.0, 3: 25.0, 4: 20.0}
    ch = {k: ((T - w) / (k - 1)) / (w / (n - k)) for k, w in W.items()}
    bw = {k: (T - w) / w for k, w in W.items()}
    chk = {k: ((T - w) / k) / (w / (n - k)) for k, w in W.items()}
    best = max(ch, key=ch.get)
    assert best == 2 and max(bw, key=bw.get) == 4 and max(chk, key=chk.get) == 3
    r12, r13, r23 = 0.56, 0.48, 0.42
    l1sq = r12 * r13 / r23
    l2sq = r12 * r23 / r13
    l3sq = r13 * r23 / r12
    return {
        "n": n, "T": s(T, 0), "W2": s(W[2], 0), "W3": s(W[3], 0), "W4": s(W[4], 0),
        "B2": s(T - W[2], 0), "B3": s(T - W[3], 0), "B4": s(T - W[4], 0),
        "ch2": s(ch[2], 1), "ch3": s(ch[3], 1), "ch4": s(ch[4], 1), "bw4": s(bw[4], 1), "chk3": s(chk[3], 1),
        "r12": r12, "r13": r13, "r23": r23, "l1sq": s(l1sq, 2), "l2sq": s(l2sq, 2), "l3sq": s(l3sq, 2),
        "l1": s(sqrt(l1sq), 1), "l2": s(sqrt(l2sq), 1), "l3": s(sqrt(l3sq), 1),
        **p06b(),
    }


def p06b():
    """(b) 精度行列から無向グラフ。周辺共分散と偏相関は解説に使う。"""
    K = [[1.5, -0.6, 0, 0], [-0.6, 2.0, -0.5, -0.4], [0, -0.5, 1.8, -0.7], [0, -0.4, -0.7, 1.6]]
    import numpy as np
    Ka = np.array(K)
    assert np.all(np.linalg.eigvalsh(Ka) > 0)
    Sg = np.linalg.inv(Ka)
    fm = lambda v: ("-" if v < 0 else "") + s(abs(v), 1)
    mat = "\\begin{pmatrix}" + "\\\\".join("&".join(fm(v) if v else "0" for v in row) for row in K) + "\\end{pmatrix}"
    return {"Kmat": mat, "s13": s(Sg[0, 2], 3), "pc23": s(-K[1][2] / sqrt(K[1][1] * K[2][2]), 3)}


def p07():
    th = 0.5
    r1 = th / (1 + th ** 2)
    phi22 = -r1 ** 2 / (1 - r1 ** 2)
    return {"th": th, "r1": s(r1, 2), "r1sq": s(r1 ** 2, 2), "om": s(1 - r1 ** 2, 2), "phi22": s(phi22, 3), "phi22_4": s(phi22, 4),
            "w_r1sq": s(r1 ** 2, 3), "w_noden": s(-r1 ** 2, 3), "w_r1": s(r1, 3), "w_zero": "0"}


def p08():
    N, k, dt = 10, 3, 0.25
    steps = k * (N - k)
    return {"N": N, "k": k, "Nk": N - k, "dt": dt, "mins": 15, "steps": steps, "hours": s(steps * dt, 2),
            "w_k2": s(k ** 2 * dt, 2), "w_nk2": s((N - k) ** 2 * dt, 2), "w_Nk": s(N * k * dt, 2), "w_half": s(steps * dt / 2, 3)}


def p09():
    n = 30
    S11, S22, S12, Syy, S1y, S2y = 100.0, 400.0, 160.0, 900.0, 210.0, 360.0
    det = S11 * S22 - S12 ** 2
    b1 = (S22 * S1y - S12 * S2y) / det
    b2 = (S11 * S2y - S12 * S1y) / det
    rss = Syy - b1 * S1y - b2 * S2y
    s2 = rss / (n - 3)
    c11 = S22 / det
    se1 = sqrt(s2 * c11)
    t1 = b1 / se1
    dx1, dx2 = 2.0, -3.0
    q = (S22 * dx1 ** 2 - 2 * S12 * dx1 * dx2 + S11 * dx2 ** 2) / det
    h = 1 / n + q
    h_ind = 1 / n + dx1 ** 2 / S11 + dx2 ** 2 / S22
    h_sign = 1 / n + (S22 * dx1 ** 2 + 2 * S12 * dx1 * dx2 + S11 * dx2 ** 2) / det
    # (c) ニューラルネット
    z1, z2 = 1.0, -0.5
    W1 = [[0.8, 0.4], [-0.6, 0.3]]
    c1 = [-0.1, 0.2]
    w2, c2 = [1.5, 2.0], 0.3
    a = [W1[i][0] * z1 + W1[i][1] * z2 + c1[i] for i in range(2)]
    hdn = [max(v, 0.0) for v in a]
    out = w2[0] * hdn[0] + w2[1] * hdn[1] + c2
    out_lin = w2[0] * a[0] + w2[1] * a[1] + c2
    npar = 2 * 2 + 2 + 2 + 1
    # (旧) エラスティックネット（Z'Z=I）
    b = [0.9, -0.25, 0.4]
    l1, l2 = 0.6, 0.5
    def soft(v, t):
        return math.copysign(max(abs(v) - t, 0.0), v)
    en = [soft(v, l1 / 2) / (1 + l2) for v in b]
    la = [soft(v, l1 / 2) for v in b]
    ri = [v / (1 + l2) for v in b]
    nh = [soft(v, l1) / (1 + l2) for v in b]
    d2 = [soft(v, l1 / 2) / (1 + 2 * l2) for v in b]
    fmt = lambda v: "(" + ",\\ ".join("0" if abs(x) < 1e-12 else s(x, 3) for x in v) + ")"
    return {
        "n": n, "S11": s(S11, 0), "S22": s(S22, 0), "S12": s(S12, 0), "Syy": s(Syy, 0), "S1y": s(S1y, 0), "S2y": s(S2y, 0),
        "det": s(det, 0), "b1": s(b1, 4), "b2": s(b2, 4), "b1n": s(S22 * S1y - S12 * S2y, 0), "b2n": s(S11 * S2y - S12 * S1y, 0),
        "reg1": s(b1 * S1y, 1), "reg2": s(b2 * S2y, 1), "rss": s(rss, 1), "dfe": n - 3, "s2": s(s2, 4),
        "c11": s(c11, 6), "se1": s(se1, 4), "t1": r(t1, 2), "t1_s": s(t1, 2), "t1_4": s(t1, 4),
        "w_simple": s(b1 / sqrt(s2 / S11), 2), "w_df": s(b1 / sqrt(rss / (n - 2) * c11), 2), "slope": s(S1y / S11, 2),
        "dx1": s(dx1, 0), "dx2": s(dx2, 0), "qa": s(S22 * dx1 ** 2, 0), "qb": s(-2 * S12 * dx1 * dx2, 0), "qc": s(S11 * dx2 ** 2, 0),
        "q": s(q, 5), "inv_n": s(1 / n, 5), "h": s(h, 3), "h5": s(h, 5), "cut": s(2 * 3 / n, 1),
        "w_ind": s(h_ind, 3), "w_non": s(q, 3), "w_sign": s(h_sign, 3),
        "eb1": s(b[0], 1), "eb2": s(b[1], 2), "eb3": s(b[2], 1), "l1": l1, "l2": l2, "thr": s(l1 / 2, 1), "opl": s(1 + l2, 1),
        "nz1": s(z1, 1), "nz2": s(z2, 1), "w11": W1[0][0], "w12": W1[0][1], "w21": W1[1][0], "w22": W1[1][1],
        "c11b": c1[0], "c12b": c1[1], "v1": w2[0], "v2": w2[1], "c2": c2, "a1": s(a[0], 2), "a2": s(a[1], 2),
        "h1": s(hdn[0], 2), "h2": s(hdn[1], 0), "out": s(out, 2), "out_lin": s(out_lin, 2), "out_nob": s(out - c2, 2),
        "npar": npar, "np_nob": 2 * 2 + 2, "np_nohb": 2 * 2 + 2 + 1,
        "en": fmt(en), "la": fmt(la), "ri": fmt(ri), "nh": fmt(nh), "d2": fmt(d2),
        "en1": s(en[0], 3), "en3": s(en[2], 4), "ab1": s(abs(b[0]) - l1 / 2, 1), "ab3": s(abs(b[2]) - l1 / 2, 1),
    }


def p10():
    xs = [3, 5, 9, 13, 4, 6, 10, 1, 8, 7, 3, 9]
    c11, c11b, c12 = 19.68, 21.92, 21.03
    xbar_ = sum(xs) / len(xs)
    ss_ = sum(v * v for v in xs) - len(xs) * xbar_ ** 2
    D = ss_ / xbar_
    assert c11 < D < c12 < c11b
    disp = {"xrow": " | ".join(map(str, xs)), "sx2": sum(v * v for v in xs), "xbar": s(xbar_, 1), "xbar2": s(xbar_ ** 2, 2),
            "ss": s(ss_, 0), "D": s(D, 2), "D4": s(D, 4), "c11": c11, "c11b": c11b, "c12": c12,
            "s2": s(ss_ / 11, 2), "idx": s(ss_ / 11 / xbar_, 2)}
    lam0, months, tot = 5, 12, sum(xs)
    nI, yI = 100, 30
    p0 = 0.5
    step = (yI - nI * p0) / (nI * p0 * (1 - p0))
    mle = math.log(yI / (nI - yI))
    b0, b1, x = -3.0, 0.5, 4
    eta = b0 + b1 * x
    e1, em1, eem = 2.718, 0.3679, 0.6922      # e^1, e^-1, e^-0.3679
    pr = 1 - eem
    return {
        **disp, "tot": tot,
        "nI": nI, "yI": yI, "nyI": nI - yI, "sres": s(yI - nI * p0, 0), "sw": s(nI * p0 * (1 - p0), 0), "beta1": s(step, 3),
        "mleI": s(mle, 3), "w_now": s((yI - nI * p0) / nI, 3), "w_sign": s(-step, 3), "w_p": s((yI - nI * p0) / (nI * p0), 3),
        "p1I": s(1 / (1 + math.exp(-step)), 3),
        "b0": s(b0, 1), "b1": s(b1, 1), "x": x, "eta": s(eta, 1), "e1": e1, "em1": em1, "eem": eem,
        "pr": s(pr, 3), "pr4": s(pr, 4), "w_logit": s(1 / (1 + e1), 3), "w_mean": s(em1, 3), "w_comp": s(eem, 3), "w_once": s(1 - em1, 3),
    }


def p11():
    k, m, mse, dfe = 3, 5, 4.0, 12
    mA, mB, mC = 32.0, 34.0, 35.0
    L = mA - (mB + mC) / 2
    csum = 1 + 0.25 + 0.25
    se = sqrt(mse * csum / m)
    tL = abs(L) / se
    F212, t12 = 3.885, 2.179
    crit = sqrt((k - 1) * F212)
    assert t12 < tL < crit
    se_bad = sqrt(mse / m)
    assert abs(L) / se_bad > crit
    # (b) 漸近相対効率
    nt = 40
    nw = math.ceil(nt * math.pi / 3)
    ns = math.ceil(nt * math.pi / 2)
    return {
        "k": k, "m": m, "mse": s(mse, 1), "dfe": dfe, "mA": s(mA, 1), "mB": s(mB, 1), "mC": s(mC, 1),
        "L": s(L, 1), "aL": s(abs(L), 1), "csum": s(csum, 1), "se": s(se, 4), "tL": s(tL, 2), "tL4": s(tL, 4),
        "F212": F212, "t12": t12, "crit": s(crit, 3), "crit2": s((k - 1) * F212, 2), "tL2": s(tL ** 2, 2),
        "w_se": s(abs(L) / se_bad, 2), "se_bad": s(se_bad, 4), "w_var": s(abs(L) / (mse * csum / m), 2),
        "nt": nt, "nw": nw, "ns": ns, "nw_raw": s(nt * math.pi / 3, 2), "ns_raw": s(nt * math.pi / 2, 2),
        "w_mw": math.floor(nt * 3 / math.pi), "w_ms": math.floor(nt * 2 / math.pi), "w_half": 2 * nt,
        "are_w": s(3 / math.pi, 3), "are_s": s(2 / math.pi, 3),
    }


def p12():
    sN1, sN2, sn1, sn2, sy1, sy2 = 6000, 4000, 250, 250, 420.0, 250.0
    w1_, w2_ = sN1 / sn1, sN2 / sn2
    west = (sn1 * w1_ * sy1 + sn2 * w2_ * sy2) / (sn1 * w1_ + sn2 * w2_)
    N1, S1, c1, N2, S2, c2, n = 6000, 200, 9, 4000, 100, 1, 500
    a1 = N1 * S1 / sqrt(c1)
    a2 = N2 * S2 / sqrt(c2)
    n1 = n * a1 / (a1 + a2)
    ney = n * N1 * S1 / (N1 * S1 + N2 * S2)
    prop = n * N1 / (N1 + N2)
    byc = n * (N1 * S1 / c1) / (N1 * S1 / c1 + N2 * S2 / c2)
    chi, df, chi0, df0 = 36.0, 24, 420.0, 36
    cfi = 1 - (chi - df) / (chi0 - df0)
    nfi = 1 - chi / chi0
    tli = (chi0 / df0 - chi / df) / (chi0 / df0 - 1)
    aXT, a, b, c, d = 0.4, 0.5, -0.6, -0.2, 0.3
    total = c + a * b
    naive = total + d * aXT
    return {
        "sN1": sN1, "sN2": sN2, "sn1": sn1, "sn2": sn2, "sy1": s(sy1, 0), "sy2": s(sy2, 0), "sw1": s(w1_, 0), "sw2": s(w2_, 0),
        "west": r(west, 1), "west_s": s(west, 1), "wsum": s(sn1 * w1_ + sn2 * w2_, 0), "wnum": s(sn1 * w1_ * sy1 + sn2 * w2_ * sy2, 0),
        "w_unw": s((sy1 + sy2) / 2, 1), "w_inv": s((sn1 / w1_ * sy1 + sn2 / w2_ * sy2) / (sn1 / w1_ + sn2 / w2_), 1),
        "N1": N1, "S1": S1, "c1": c1, "N2": N2, "S2": S2, "c2": c2, "nn": n, "sc1": s(sqrt(c1), 0),
        "a1": s(a1, 0), "a2": s(a2, 0), "n1": r(n1, 0), "n1raw": s(n1, 1), "n2": r(n - n1, 0),
        "w_ney": s(ney, 0), "w_prop": s(prop, 0), "w_byc": s(byc, 0),
        "chi": s(chi, 0), "df": df, "chi0": s(chi0, 0), "df0": df0, "nc": s(chi - df, 0), "nc0": s(chi0 - df0, 0),
        "cfi": s(cfi, 3), "cfi5": s(cfi, 5), "nfi": s(nfi, 3), "tli": s(tli, 3), "w_cfi0": s(1 - (chi - df) / chi0, 3),
        "w_ratio": s(1 - (chi / df) / (chi0 / df0), 3),
        "aXT": aXT, "a": a, "b": b, "c": c, "d": d, "ab": s(a * b, 2), "total": s(total, 2), "bias": s(d * aXT, 2),
        "naive": s(naive, 2), "w_bias": s(total - d * aXT, 2), "c_s": s(c, 2),
    }


def p13():
    n, r_, xa, xo, yo, b = 30, 20, 5.0, 5.6, 5.2, 0.90
    xm = (n * xa - r_ * xo) / (n - r_)
    ym = yo + b * (xm - xo)
    m = (r_ * yo + (n - r_) * ym) / n
    assert abs(m - (yo + b * (xa - xo))) < 1e-12
    return {"n": n, "r": r_, "miss": n - r_, "xa": s(xa, 1), "xo": s(xo, 1), "yo": s(yo, 1), "b": s(b, 2),
            "xm": s(xm, 1), "ym": s(ym, 2), "m": r(m, 2), "m_s": s(m, 2), "m4": s(m, 4)}


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s20p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
