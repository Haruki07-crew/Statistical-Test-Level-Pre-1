"""模擬試験 第4回（set04）の数値。values() が {大問id: {名前: 値}} を返す。

題材：製造・半導体・エネルギー・電力。
問題文に示した分布の値（Φ・t の点、e^x、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
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
    # (a) サプライヤーの事後確率（二項尤度とベイズの定理）
    wA, wB, qA, qB = 0.7, 0.3, 0.02, 0.06
    nS, xS = 20, 2
    pw98, pw94 = 0.6951, 0.3283  # 0.98^18, 0.94^18（問題文に示す）
    LA = qA ** 2 * pw98
    LB = qB ** 2 * pw94
    jA, jB = wA * LA, wB * LB
    post = jB / (jA + jB)
    post_nopow = wB * qB ** 2 / (wA * qA ** 2 + wB * qB ** 2)
    post_noprior = LB / (LA + LB)
    # (b) 負の二項分布：モーメント法で r と p を決め、欠陥 0 個の確率（歩留まり）
    m, v = 0.6, 0.72
    p = m / v
    rr = m * p / (1 - p)
    rr_i = int(round(rr))
    assert abs(rr - rr_i) < 1e-9
    y0 = p ** rr_i
    epois = 0.5488  # e^{-0.6}（問題文に示す）
    # (c) 3×2 分割表の調整済み標準化残差（夜勤×不良）
    n_sh = 400
    d = [18, 22, 40]
    D = sum(d)
    N = n_sh * 3
    E = n_sh * D / N
    adj_den2 = E * (1 - n_sh / N) * (1 - D / N)
    adj = (d[2] - E) / sqrt(adj_den2)
    pear = (d[2] - E) / sqrt(E)
    return {
        "wA": wA, "wB": wB, "qA": qA, "qB": qB, "nS": nS, "xS": xS, "nx": nS - xS, "pw98": pw98, "pw94": pw94,
        "qA2": s(qA ** 2, 4), "qB2": s(qB ** 2, 4), "LA": s(LA, 8), "LB": s(LB, 8), "jA": s(jA, 8), "jB": s(jB, 8), "jAB": s(jA + jB, 8),
        "LAe": s(LA * 1e6, 2), "LBe": s(LB * 1e6, 2), "jAe": s(jA * 1e6, 2), "jBe": s(jB * 1e6, 2), "jABe": s((jA + jB) * 1e6, 2),
        "post1": r(post, 3), "post14": s(post, 4), "post_nopow": s(post_nopow, 3), "post_noprior": s(post_noprior, 3),
        "m": m, "v": v, "p": s(p, 4), "pfrac": "\\frac{5}{6}", "q": s(1 - p, 4), "rr": rr_i,
        "vm": s(v - m, 2), "m2": s(m * m, 2),
        "y0": s(y0, 3), "y04": s(y0, 4), "epois": epois, "y_pois": s(epois, 3), "y_p": s(p, 3),
        "y_p2": s(p ** 2, 3), "y_p4": s(p ** 4, 3),
        "n_sh": n_sh, "d1": d[0], "d2": d[1], "d3": d[2], "g1": n_sh - d[0], "g2": n_sh - d[1], "g3": n_sh - d[2],
        "D": D, "N": N, "G": N - D, "E": s(E, 3), "oe": s(d[2] - E, 3),
        "rowf": s(1 - n_sh / N, 4), "colf": s(1 - D / N, 4), "den2": s(adj_den2, 3), "den": s(sqrt(adj_den2), 4),
        "adj": r(adj, 2), "adj4": s(adj, 4), "pear": s(pear, 2),
    }


def p02():
    c = 8
    v0 = 12
    e225 = 0.1054                     # e^{-2.25}（問題文に示す）
    assert abs(v0 ** 2 / c ** 2 - 2.25) < 1e-12
    pq = e225 * (1 - e225)
    se = 0.005
    nraw = pq / se ** 2
    return {
        "c": c, "c2": c * c, "half_c2": c * c // 2, "k3": 3 * c * c // 2, "v0": v0, "v02": v0 * v0, "ex": "2.25",
        "e225": e225, "pq": s(pq, 5), "se": se, "se2": s(se ** 2, 6), "nraw": s(nraw, 1),
        "n_ok": math.ceil(nraw), "n_half": math.ceil(0.25 / se ** 2), "n_nosq": math.ceil(sqrt(pq) / se),
        "n_196": math.ceil(1.96 ** 2 * pq / se ** 2), "n_p": math.ceil(e225 / se ** 2),
    }


def p03():
    # (a) 確率母関数の積から P(N=2)
    q, nb = 0.1, 2
    lam = 0.4
    e04 = 0.6703  # e^{-0.4}
    b = [(1 - q) ** 2, 2 * q * (1 - q), q ** 2]               # 二項 B(2, 0.1)
    po = [1, lam, lam ** 2 / 2]                              # e^{-λ} を除いたポアソンの係数
    inner = b[0] * po[2] + b[1] * po[1] + b[2] * po[0]
    P2 = e04 * inner
    # (b) ウィルソンの上限
    n, x, z = 50, 2, 1.96
    ph = x / n
    z2 = z * z
    a1 = z2 / (2 * n)
    v1 = ph * (1 - ph) / n
    v2 = z2 / (4 * n * n)
    rt = sqrt(v1 + v2)
    up = (ph + a1 + z * rt) / (1 + z2 / n)
    wald = ph + z * sqrt(v1)
    up_noden = ph + a1 + z * rt
    up_164 = (ph + 1.645 ** 2 / (2 * n) + 1.645 * sqrt(v1 + 1.645 ** 2 / (4 * n * n))) / (1 + 1.645 ** 2 / n)
    return {
        "q": q, "q1": s(1 - q, 1), "lam": lam, "e04": e04, "b0": s(b[0], 2), "b1": s(b[1], 2), "b2": s(b[2], 2),
        "po2": s(po[2], 2), "t1": s(b[0] * po[2], 4), "t2": s(b[1] * po[1], 4), "t3": s(b[2] * po[0], 4),
        "inner": s(inner, 4), "P2": r(P2, 4), "P26": s(P2, 6),
        "P2_only": s(e04 * po[2], 4), "P2_b": s(b[2], 4),
        "n": n, "x": x, "z": z, "ph": s(ph, 2), "z2": s(z2, 4), "a1": s(a1, 6), "v1": s(v1, 6), "v2": s(v2, 8),
        "vs": s(v1 + v2, 8), "rt": s(rt, 6), "zrt": s(z * rt, 6), "num": s(ph + a1 + z * rt, 6), "den": s(1 + z2 / n, 6),
        "up": r(up, 3), "up5": s(up, 5), "wald": s(wald, 3), "up_noden": s(up_noden, 3), "up_164": s(up_164, 3),
    }


def p04():
    # (a) 一様分布 2 つの和（三角分布）
    hw, c = 1.0, 1.5
    tail = (2 * hw - c) ** 2 / (8 * hw ** 2)      # P(S>c)
    ex2 = 2 * tail
    sdS = sqrt(2 * (2 * hw) ** 2 / 12)
    zc = r(c / sdS, 2)
    phi184 = 0.9671
    assert zc == 1.84
    nrm = 2 * (1 - phi184)
    # (b) ウェルチの検定
    nA, mA, sA = 8, 101.2, 2.0
    nB, mB, sB = 12, 98.9, 4.5
    a, b = sA ** 2 / nA, sB ** 2 / nB
    se = sqrt(a + b)
    t = (mA - mB) / se
    nu = (a + b) ** 2 / (a ** 2 / (nA - 1) + b ** 2 / (nB - 1))
    nu_n = (a + b) ** 2 / (a ** 2 / nA + b ** 2 / nB)
    sp2 = ((nA - 1) * sA ** 2 + (nB - 1) * sB ** 2) / (nA + nB - 2)
    tp = (mA - mB) / sqrt(sp2 * (1 / nA + 1 / nB))
    pair = lambda tt, nn: f"$t={tt},\\ \\nu\\approx {nn}$"
    return {
        "hw": s(hw, 0), "c": c, "rest": s(2 * hw - c, 1), "tail": s(tail, 5), "ex2": s(ex2, 4), "ex2_5": s(ex2, 5),
        "varS": "\\frac{2}{3}", "sdS": s(sdS, 4), "zraw": s(c / sdS, 4), "zc": s(zc, 2), "phi184": phi184,
        "c_ok": s(ex2, 4), "c_nrm": s(nrm, 4), "c_one": s(tail, 4), "c_unif": s((2 * 2 * hw - 2 * c) / (4 * hw), 4), "c_half": s(2 * ex2, 4),
        "nA": nA, "mA": mA, "sA": sA, "nB": nB, "mB": mB, "sB": sB, "vA": s(sA ** 2, 2), "vB": s(sB ** 2, 2),
        "a": s(a, 4), "b": s(b, 4), "ab": s(a + b, 4), "se": s(se, 4), "diff": s(mA - mB, 1),
        "t": s(t, 2), "nu": s(nu, 1), "ab2": s((a + b) ** 2, 4), "a2": s(a ** 2 / (nA - 1), 4), "b2": s(b ** 2 / (nB - 1), 4),
        "dA": nA - 1, "dB": nB - 1, "dfp": nA + nB - 2, "sp2": s(sp2, 2), "tp": s(tp, 2), "nu_n": s(nu_n, 1),
        "ch_ok": pair(s(t, 2), s(nu, 1)), "ch_pool": pair(s(tp, 2), nA + nB - 2), "ch_wdf": pair(s(t, 2), nA + nB - 2),
        "ch_min": pair(s(t, 2), min(nA, nB) - 1), "ch_n": pair(s(t, 2), s(nu_n, 1)),
    }


def p05():
    # (a) マルコフの不等式
    mu, a = 0.75, 3
    e075 = 0.4724
    pois = 1 - e075 * (1 + mu + mu ** 2 / 2)
    # (b) ラオ・ブラックウェル化した e^{-λ} の不偏推定量
    n, T = 12, 9
    rb = ((n - 1) / n) ** T
    return {
        "mu": mu, "a": a, "e075": e075, "bound": s(mu / a, 3), "b_sq": s(mu / a ** 2, 3), "b_am1": s(mu / (a - 1), 3),
        "b_pois": s(pois, 3), "b_1m": s(1 - mu / a, 3), "pois_in": s(1 + mu + mu ** 2 / 2, 5), "pois4": s(pois, 4),
        "n": n, "T": T, "n1": n - 1, "lamh": s(T / n, 2), "rb": s(rb, 3), "rb5": s(rb, 5), "frac": s((n - 1) / n, 4),
        "w_mle": s(e075, 3), "w_t1": s(((n - 1) / n) ** (T - 1), 3), "w_lin": s(1 - T / n, 3), "w_n": s(((n - 1) / n) ** n, 3),
    }


def p06():
    # (a) G 統計量による適合度検定
    hrs = [1000, 1500, 2500]
    obs = [18, 12, 20]
    n = sum(obs)
    H = sum(hrs)
    exp = [n * h / H for h in hrs]
    ln = {1.8: 0.5878, 0.8: -0.2231}
    ratios = [o / e for o, e in zip(obs, exp)]
    for q in ratios:
        assert round(q, 6) in ln, q
    terms = [o * ln[round(q, 6)] for o, q in zip(obs, ratios)]
    G = 2 * sum(terms)
    X2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    assert 5.991 < G < 7.815
    # (b) プロビットモデルの逆推定
    b0, b1 = -6.0, 0.025
    z10, z05, ln9 = 1.2816, 1.645, 2.1972
    x10 = (-z10 - b0) / b1
    pr = [x10, (z10 - b0) / b1, -b0 / b1, (-ln9 - b0) / b1, (-z05 - b0) / b1]
    assert len({s(v, 1) for v in pr}) == 5
    return {
        "h1": hrs[0], "h2": hrs[1], "h3": hrs[2], "H": H, "o1": obs[0], "o2": obs[1], "o3": obs[2], "n": n,
        "e1": s(exp[0], 0), "e2": s(exp[1], 0), "e3": s(exp[2], 0), "ln18": s(ln[1.8], 4), "ln08": s(ln[0.8], 4),
        "t1": s(terms[0], 4), "t2": s(terms[1], 4), "t3": s(terms[2], 4), "tsum": s(sum(terms), 4),
        "G": s(G, 2), "Ghalf": s(G / 2, 2), "X2": s(X2, 2), "c2": 5.991, "c3": 7.815, "c201": "9.210",
        "b0": b0, "b1": b1, "z10": z10, "z05": z05, "ln9": ln9, "num10": s(-z10 - b0, 4),
        "x10": s(x10, 1), "x10_4": s(x10, 3), "x_pos": s(pr[1], 1), "x50": s(pr[2], 1), "x_logit": s(pr[3], 1), "x_05": s(pr[4], 1),
    }


def p07():
    lam, mu = 0.05, 0.5   # 1 台あたりの故障率、修理率（1 時間あたり）
    a1 = 2 * lam / mu
    a2 = a1 * lam / mu
    p0 = 1 / (1 + a1 + a2)
    p2 = a2 * p0
    # 修理担当 2 人なら π2/π1 = λ/(2μ)
    b2 = a1 * lam / (2 * mu)
    p2_two = b2 / (1 + a1 + b2)
    ind = (lam / (lam + mu)) ** 2
    return {"lam": lam, "mu": mu, "lam2": s(2 * lam, 1), "a1": s(a1, 1), "a2": s(a2, 2), "rat": s(lam / mu, 1),
            "den": s(1 + a1 + a2, 2), "p0": s(p0, 4), "p1": s(a1 * p0, 4), "p2": r(p2, 4), "p26": s(p2, 6),
            "p2_two": s(p2_two, 4), "ind": s(ind, 4)}


def p08():
    A = [412, 455, 498]
    B = [305, 360, 389, 421, 440]
    allv = sorted(A + B)
    rk = [allv.index(x) + 1 for x in A]
    W = sum(rk)
    from itertools import combinations
    sums = [sum(c) for c in combinations(range(1, 9), 3)]
    tot = len(sums)
    ge = sum(1 for x in sums if x >= W)
    gt = sum(1 for x in sums if x > W)
    EW = 3 * 9 / 2
    VW = 3 * 5 * 9 / 12
    z = (W - EW) / sqrt(VW)
    assert r(z, 2) == 1.64
    return {"a1": A[0], "a2": A[1], "a3": A[2], "b1": B[0], "b2": B[1], "b3": B[2], "b4": B[3], "b5": B[4],
            "rk1": rk[0], "rk2": rk[1], "rk3": rk[2], "W": W, "tot": tot, "ge": ge, "gt": gt,
            "p_ok": s(ge / tot, 3), "p_gt": s(gt / tot, 3), "p_two": s(2 * ge / tot, 3), "p_low": s(1 - ge / tot, 3),
            "EW": s(EW, 1), "VW": s(VW, 2), "sdW": s(sqrt(VW), 3), "z": s(z, 2), "phi164": 0.9495, "p_norm": s(Decimal("1") - Decimal("0.9495"), 3)}


def p09():
    n, xbar, Sxx, b0, b1, sres = 20, 30, 160, -300, 25, 12
    x0 = 36
    t18, t19, t18_05 = 2.101, 2.093, 1.734
    y0 = b0 + b1 * x0
    k = 1 + 1 / n + (x0 - xbar) ** 2 / Sxx
    sepred = sres * sqrt(k)
    up = y0 + t18 * sepred
    km = 1 / n + (x0 - xbar) ** 2 / Sxx
    # (b) DFFITS
    xi, ti = 38, 2.6
    h = 1 / n + (xi - xbar) ** 2 / Sxx
    dff = ti * sqrt(h / (1 - h))
    h_no = (xi - xbar) ** 2 / Sxx
    p = 2
    thr = 2 * sqrt(p / n)
    # (c) AR(1) 残差の平均の分散と有効標本サイズ
    phi, nd = 0.6, 120
    fac = (1 + phi) / (1 - phi)
    neff_ex = nd * nd / sum(phi ** abs(i - j) for i in range(nd) for j in range(nd))   # 有限 n の正確な値（解説の補足）
    return {
        "n": n, "xbar": xbar, "Sxx": Sxx, "b0": b0, "b1": b1, "sres": sres, "x0": x0, "dx0": x0 - xbar, "dx02": (x0 - xbar) ** 2,
        "t18": t18, "t19": t19, "t18_05": t18_05, "y0": y0, "n_inv": s(1 / n, 2), "q": s((x0 - xbar) ** 2 / Sxx, 3),
        "k": s(k, 3), "sqk": s(sqrt(k), 4), "sepred": s(sepred, 4), "hw": s(t18 * sepred, 3),
        "up": r(up, 1), "up4": s(up, 3), "up_mean": s(y0 + t18 * sres * sqrt(km), 1),
        "up_t19": s(y0 + t19 * sepred, 1), "up_z": s(y0 + 1.96 * sepred, 1), "up_1s": s(y0 + t18_05 * sepred, 1),
        "xi": xi, "ti": ti, "dxi": xi - xbar, "dxi2": (xi - xbar) ** 2, "h": s(h, 2), "hr": s(h / (1 - h), 4), "sqhr": s(sqrt(h / (1 - h)), 4),
        "dff": s(dff, 2), "d_noinv": s(ti * sqrt(h_no / (1 - h_no)), 2), "d_sqh": s(ti * sqrt(h), 2),
        "d_nosq": s(ti * sqrt((1 / n + (xi - xbar) / Sxx) / (1 - 1 / n - (xi - xbar) / Sxx)), 2),
        "d_div": s(ti * sqrt(h) / (1 - h), 2), "h_no": s(h_no, 2), "h_nosq": s(1 / n + (xi - xbar) / Sxx, 2),
        "p": p, "thr": s(thr, 3), "hbar": s(p / n, 2),
        "phi": phi, "nd": nd, "fac": s(fac, 1), "sumr": s(phi / (1 - phi), 1), "neff": s(nd / fac, 0), "neff_ex": s(neff_ex, 1),
        "n_1p": s(nd / (1 + phi), 0), "n_1m": s(nd * (1 - phi), 0), "n_r2": s(nd * (1 - phi ** 2), 1), "n_rev": s(nd * fac, 0),
        "phi2": s(phi ** 2, 2),
    }


def p10():
    # (a) 有限母集団修正つきの必要標本数
    Nn, S, d, z = 50, 40, 15, 1.96
    n0 = (z * S / d) ** 2
    n = n0 / (1 + n0 / Nn)
    n_alt = n0 / (1 + (n0 - 1) / Nn)
    assert math.ceil(n) == math.ceil(n_alt)
    n164 = (1.645 * S / d) ** 2
    nsq = z * S / d
    # (b) 原点を通る回帰、分散 ∝ x
    x = [10, 20, 40, 50, 80]
    yy = [32, 58, 130, 148, 252]
    Sx, Sy = sum(x), sum(yy)
    wls = Sy / Sx
    ols0 = sum(a * b for a, b in zip(x, yy)) / sum(a * a for a in x)
    mr = sum(b / a for a, b in zip(x, yy)) / len(x)
    xb, yb = Sx / len(x), Sy / len(x)
    ols1 = sum((a - xb) * (b - yb) for a, b in zip(x, yy)) / sum((a - xb) ** 2 for a in x)
    wx = sum(a * a * b for a, b in zip(x, yy)) / sum(a ** 3 for a in x)
    vals = [wls, ols0, mr, ols1, wx]
    assert len({s(v, 3) for v in vals}) == 5, [s(v, 3) for v in vals]
    # (c) HC0 型の頑健な標準誤差
    xh = [10, 20, 30, 40, 50]
    eh = [3, -6, -3, 12, -6]
    xm = sum(xh) / 5
    assert sum(eh) == 0 and sum(a * b for a, b in zip(xh, eh)) == 0
    dd = [(a - xm) ** 2 for a in xh]
    Sxx_h = sum(dd)
    meat = sum(a * b * b for a, b in zip(dd, eh))
    hc0 = sqrt(meat) / Sxx_h
    s2 = sum(b * b for b in eh) / 3
    ols_se = sqrt(s2 / Sxx_h)
    return {
        "Nn": Nn, "S": S, "d": d, "z": z, "zSd": s(z * S / d, 4), "n0": s(n0, 2), "n0N": s(n0 / Nn, 4), "nraw": s(n, 2),
        "nreq": math.ceil(n), "n_nofpc": math.ceil(n0), "n_164": math.ceil(n164 / (1 + n164 / Nn)), "n_nosq": math.ceil(nsq / (1 + nsq / Nn)),
        "n_mult": math.ceil(n0 * (1 - n0 / Nn)),
        "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "x5": x[4], "yy1": yy[0], "yy2": yy[1], "yy3": yy[2], "yy4": yy[3], "yy5": yy[4],
        "Sx": Sx, "Sy": Sy, "wls": s(wls, 3), "ols0": s(ols0, 3), "mr": s(mr, 3), "ols1": s(ols1, 3), "wx": s(wx, 3),
        "Sxy": sum(a * b for a, b in zip(x, yy)), "Sxx": sum(a * a for a in x),
        "h1": xh[0], "h2": xh[1], "h3": xh[2], "h4": xh[3], "h5": xh[4], "e1": eh[0], "e2": eh[1], "e3": eh[2], "e4": eh[3], "e5": eh[4],
        "hxm": s(xm, 0), "dd1": s(dd[0], 0), "dd2": s(dd[1], 0), "dd4": s(dd[3], 0), "dd5": s(dd[4], 0), "Sxxh": s(Sxx_h, 0), "Sxxh2": s(Sxx_h ** 2, 0),
        "m1": s(dd[0] * eh[0] ** 2, 0), "m2": s(dd[1] * eh[1] ** 2, 0), "m4": s(dd[3] * eh[3] ** 2, 0), "m5": s(dd[4] * eh[4] ** 2, 0),
        "meat": s(meat, 0), "vhc": s(meat / Sxx_h ** 2, 3), "hc0": s(hc0, 3), "hc0_4": s(hc0, 4), "ss": sum(b * b for b in eh), "s2": s(s2, 0),
        "ols_se": s(ols_se, 3), "hc1": s(hc0 * sqrt(5 / 3), 3), "hc_bad": s(sqrt(meat / Sxx_h), 3),
    }


def p11():
    y = {"a": 20, "b": 26, "c": 18, "abc": 30}
    hiA = (y["a"] + y["abc"]) / 2
    loA = (y["b"] + y["c"]) / 2
    eff = hiA - loA
    contrast = y["a"] + y["abc"] - y["b"] - y["c"]
    return {"ya": y["a"], "yb": y["b"], "yc": y["c"], "yabc": y["abc"], "hiA": s(hiA, 1), "loA": s(loA, 1),
            "eff": s(eff, 1), "contrast": s(contrast, 0), "eff4": s(contrast / 4, 1), "effc": s(contrast, 1)}


def p12():
    # (a) 相関行列の主成分得点
    lam = [2.6, 0.8, 0.4, 0.2]
    e1 = [0.7, 0.5, 0.5, 0.1]
    assert abs(sum(x * x for x in e1) - 1) < 1e-12
    mean = [400, 2.0, 50, 10]
    sdv = [5, 0.1, 2, 0.5]
    obs = [406, 2.08, 49.2, 11.0]
    zz = [(o - m) / d for o, m, d in zip(obs, mean, sdv)]
    sc = sum(a * b for a, b in zip(e1, zz))
    sc_raw = sum(a * (o - m) for a, o, m in zip(e1, obs, mean))
    sc_abs = sum(a * abs(b) for a, b in zip(e1, zz))
    sq26 = 1.6125
    # (b) 行プロファイル間のカイ二乗距離
    tab = [[30, 10, 10], [10, 20, 20], [20, 30, 50]]
    Nt = sum(map(sum, tab))
    col = [sum(rw[j] for rw in tab) / Nt for j in range(3)]
    p1 = [v / sum(tab[0]) for v in tab[0]]
    p2 = [v / sum(tab[1]) for v in tab[1]]
    d2 = sum((a - b) ** 2 / c for a, b, c in zip(p1, p2, col))
    eu = sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    mul = sqrt(sum((a - b) ** 2 * c for a, b, c in zip(p1, p2, col)))
    avg = sqrt(sum((a - b) ** 2 / ((a + b) / 2) for a, b in zip(p1, p2)))
    vals = [sqrt(d2), eu, d2, mul, avg]
    assert len({s(v, 3) for v in vals}) == 5
    # (c) コストつきベイズ判別
    mG, mD, sd = 50, 58, 4
    piD, cm, cf = 0.05, 10, 1
    ln19r, ln19, ln10 = 0.6419, 2.9444, 2.3026   # ln 1.9, ln 19, ln 10（問題文に示す）
    assert abs((1 - piD) * cf / (piD * cm) - 1.9) < 1e-12
    mid = (mG + mD) / 2
    k = sd ** 2 / (mD - mG)
    thr = mid + k * ln19r            # ln{(π_G c_f)/(π_D c_m)} = ln 1.9
    thr_prior = mid + k * ln19
    thr_cost = mid - k * ln10
    thr_rev = mid - k * ln19r
    assert len({s(v, 2) for v in (mid, thr, thr_prior, thr_cost, thr_rev)}) == 5
    return {
        "l1": lam[0], "l2": lam[1], "l3": lam[2], "l4": lam[3], "e11": e1[0], "e12": e1[1], "e13": e1[2], "e14": e1[3],
        "m1": mean[0], "m2": s(mean[1], 1), "m3": mean[2], "m4": mean[3], "d1": sdv[0], "d2": sdv[1], "d3": sdv[2], "d4": sdv[3],
        "o1": obs[0], "o2": s(obs[1], 2), "o3": s(obs[2], 1), "o4": s(obs[3], 1),
        "z1": s(zz[0], 1), "z2": s(zz[1], 1), "z3": s(zz[2], 1), "z4": s(zz[3], 1),
        "p1": s(e1[0] * zz[0], 2), "p2": s(e1[1] * zz[1], 2), "p3": s(e1[2] * zz[2], 2), "p4": s(e1[3] * zz[3], 2),
        "sc": s(sc, 2), "sc_raw": s(sc_raw, 2), "sc_load": s(sc * sq26, 2), "sc_div": s(sc / sq26, 2), "sc_abs": s(sc_abs, 2),
        "sq26": sq26, "cum2": s(100 * (lam[0] + lam[1]) / 4, 0),
        "t11": tab[0][0], "t12": tab[0][1], "t13": tab[0][2], "t21": tab[1][0], "t22": tab[1][1], "t23": tab[1][2],
        "t31": tab[2][0], "t32": tab[2][1], "t33": tab[2][2], "r1": sum(tab[0]), "r2": sum(tab[1]), "r3": sum(tab[2]),
        "k1": sum(rw[0] for rw in tab), "k2": sum(rw[1] for rw in tab), "k3": sum(rw[2] for rw in tab), "Nt": Nt,
        "c1": s(col[0], 1), "c2": s(col[1], 1), "c3": s(col[2], 1),
        "p11": s(p1[0], 1), "p12": s(p1[1], 1), "p13": s(p1[2], 1), "p21": s(p2[0], 1), "p22": s(p2[1], 1), "p23": s(p2[2], 1),
        "q1": s((p1[0] - p2[0]) ** 2 / col[0], 4), "q2": s((p1[1] - p2[1]) ** 2 / col[1], 4), "q3": s((p1[2] - p2[2]) ** 2 / col[2], 4),
        "chi_d2": s(d2, 4), "chi_d": s(sqrt(d2), 3), "w_eu": s(eu, 3), "w_d2": s(d2, 3), "w_mul": s(mul, 3), "w_avg": s(avg, 3),
        "mG": mG, "mD": mD, "sd": sd, "sd2": sd * sd, "piD": piD, "piG": s(1 - piD, 2), "cm": cm, "cf": cf,
        "ln19r": s(ln19r, 4), "ln19": s(ln19, 4), "ln10": s(ln10, 4), "mid": s(mid, 0), "k": s(k, 0),
        "thr": s(thr, 2), "thr4": s(thr, 4), "thr_mid": s(mid, 2), "thr_prior": s(thr_prior, 2), "thr_cost": s(thr_cost, 2), "thr_rev": s(thr_rev, 2),
        "lr": s((1 - piD) * cf / (piD * cm), 2), "dm": mD - mG, "sum_m2": mD ** 2 - mG ** 2, "twos2": 2 * sd * sd,
    }


def p13():
    # 層 z=1（老朽）, z=0（新しい）。n[z][x] と事故件数 y[z][x]
    n = {1: {1: 50, 0: 150}, 0: {1: 200, 0: 100}}
    y = {1: {1: 10, 0: 45}, 0: {1: 10, 0: 8}}
    N = sum(n[z][x] for z in n for x in (0, 1))
    e = {z: n[z][1] / (n[z][1] + n[z][0]) for z in n}
    E1 = sum(y[z][1] / e[z] for z in n) / N
    E0 = sum(y[z][0] / (1 - e[z]) for z in n) / N
    ace = E1 - E0
    n1 = sum(n[z][1] for z in n)
    n0 = sum(n[z][0] for z in n)
    naive = sum(y[z][1] for z in n) / n1 - sum(y[z][0] for z in n) / n0
    E1s = sum(y[z][1] / (1 - e[z]) for z in n) / N
    E0s = sum(y[z][0] / e[z] for z in n) / N
    risk = {z: {x: y[z][x] / n[z][x] for x in (0, 1)} for z in n}
    att = sum(n[z][1] / n1 * (risk[z][1] - risk[z][0]) for z in n)
    unw = sum(risk[z][1] - risk[z][0] for z in n) / 2
    vals = [ace, naive, E1s - E0s, att, unw]
    assert len({s(v, 3) for v in vals}) == 5, [s(v, 3) for v in vals]
    fr = lambda v: "\\frac{2}{3}" if abs(v - 2 / 3) < 1e-12 else ("\\frac{1}{3}" if abs(v - 1 / 3) < 1e-12 else s(v, 2))
    return {
        "n11": n[1][1], "n10": n[1][0], "n01": n[0][1], "n00": n[0][0], "y11": y[1][1], "y10": y[1][0], "y01": y[0][1], "y00": y[0][0],
        "N1": n[1][1] + n[1][0], "N0": n[0][1] + n[0][0], "N": N, "e1": s(e[1], 2), "e0": fr(e[0]), "q1": s(1 - e[1], 2), "q0": fr(1 - e[0]),
        "w11": s(y[1][1] / e[1], 0), "w01": s(y[0][1] / e[0], 0), "w10": s(y[1][0] / (1 - e[1]), 0), "w00": s(y[0][0] / (1 - e[0]), 0),
        "s1": s(sum(y[z][1] / e[z] for z in n), 0), "s0": s(sum(y[z][0] / (1 - e[z]) for z in n), 0),
        "E1": s(E1, 3), "E0": s(E0, 3), "ace": s(ace, 3), "naive": s(naive, 3), "swap": s(E1s - E0s, 3), "att": s(att, 3), "unw": s(unw, 3),
        "r11": s(risk[1][1], 2), "r10": s(risk[1][0], 2), "r01": s(risk[0][1], 2), "r00": s(risk[0][0], 2),
        "nv1": s(sum(y[z][1] for z in n) / n1, 3), "nv0": s(sum(y[z][0] for z in n) / n0, 3), "pz1": s((n[1][1] + n[1][0]) / N, 1), "pz0": s((n[0][1] + n[0][0]) / N, 1),
        "ws1": s(sum(n[z][1] / e[z] for z in n), 0),
    }


def values():
    return {"s04p01": p01(), "s04p02": p02(), "s04p03": p03(), "s04p04": p04(), "s04p05": p05(),
            "s04p06": p06(), "s04p07": p07(), "s04p08": p08(), "s04p09": p09(), "s04p10": p10(),
            "s04p11": p11(), "s04p12": p12(), "s04p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
