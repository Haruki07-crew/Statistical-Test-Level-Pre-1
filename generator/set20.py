"""模擬試験 第20回（set20）の数値。values() が {大問id: {名前: 値}} を返す。

題材：エネルギー・電力需要・再生可能エネルギー・環境測定。
問題文に示した分布の値（χ²・F・t の点、e^x、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
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
    x0 = Fr(1, 5)
    # (a) P(Y>1/2 | X=x0) = ∫_{1/2}^1 (2x0+y)dy / ∫_0^1 (2x0+y)dy
    num = 2 * x0 * Fr(1, 2) + (1 - Fr(1, 4)) / 2      # = x0 + 3/8
    den = 2 * x0 + Fr(1, 2)
    cond = num / den
    joint_only = c * num                               # 周辺密度で割り忘れ
    uncond = c * (Fr(1, 2) * 1 + Fr(3, 8))             # P(Y>1/2)=(2/3)∫_{1/2}^1 (1+y)dy
    y0 = Fr(1, 5)
    wrongvar = (Fr(3, 4) + y0 * Fr(1, 2)) / (1 + y0)   # P(X>1/2 | Y=0.2)
    # (b) 相関係数
    EX = c * (Fr(2, 3) + Fr(1, 4))
    EY = c * (Fr(1, 2) + Fr(1, 3))
    EXY = c * (Fr(2, 6) + Fr(1, 6))
    EX2 = c * (Fr(2, 4) + Fr(1, 6))
    EY2 = c * (Fr(1, 3) + Fr(1, 4))
    cov = EXY - EX * EY
    vx = EX2 - EX ** 2
    vy = EY2 - EY ** 2
    rho = float(cov) / sqrt(float(vx) * float(vy))
    rho_nc = float(EXY) / sqrt(float(EX2) * float(EY2))
    return {
        "x0": s(x0, 1), "y0": s(y0, 1), "fx0": s(c * den, 4), "fx0f": "\\frac{2}{3}\\times" + s(den, 1),
        "num": s(num, 3), "den": s(den, 1), "cond": s(cond, 3), "cond4": s(cond, 4),
        "c_joint": s(joint_only, 3), "c_uncond": s(uncond, 3), "c_comp": s(1 - cond, 3), "c_wrongvar": s(wrongvar, 3),
        "EX": f"\\frac{{{EX.numerator}}}{{{EX.denominator}}}", "EY": f"\\frac{{{EY.numerator}}}{{{EY.denominator}}}",
        "EXY": f"\\frac{{{EXY.numerator}}}{{{EXY.denominator}}}", "EX2": f"\\frac{{{EX2.numerator}}}{{{EX2.denominator}}}",
        "EY2": f"\\frac{{{EY2.numerator}}}{{{EY2.denominator}}}",
        "cov": f"-\\frac{{{-cov.numerator}}}{{{cov.denominator}}}", "vx": f"\\frac{{{vx.numerator}}}{{{vx.denominator}}}",
        "vy": f"\\frac{{{vy.numerator}}}{{{vy.denominator}}}",
        "cov_d": s(cov, 6), "vx_d": s(vx, 5), "vy_d": s(vy, 5), "sdx": s(sqrt(vx), 5), "sdy": s(sqrt(vy), 5),
        "rho": r(rho, 3), "rho_s": s(rho, 3), "rho5": s(rho, 4), "rho_nc": s(rho_nc, 3),
    }


def p02():
    p1, p2, p3 = 0.95, 0.90, 0.85
    pairs = p1 * p2 + p1 * p3 + p2 * p3
    triple = p1 * p2 * p3
    vote = pairs - 2 * triple
    exact2 = p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
    # (b) X~U(1,3), Y~U(1,4) の和が 5.5 以下
    a1, b1, a2, b2, tt = 1, 3, 1, 4, 5.5
    L1, L2 = b1 - a1, b2 - a2
    u = tt - a1 - a2                     # 3.5
    # 余事象：x'+y'>u の三角形（脚の長さ L1+L2-u）
    leg = L1 + L2 - u
    comp = leg ** 2 / 2 / (L1 * L2)
    ok = 1 - comp
    return {
        "p1": p1, "p2": p2, "p3": p3, "pa12": s(p1 * p2, 4), "pa13": s(p1 * p3, 4), "pa23": s(p2 * p3, 4),
        "pairs": s(pairs, 4), "triple": s(triple, 5), "two_triple": s(2 * triple, 4),
        "vote": r(vote, 3), "vote_s": s(vote, 3), "exact2": s(exact2, 5),
        "w_any": s(1 - (1 - p1) * (1 - p2) * (1 - p3), 5), "w_all": s(triple, 3), "w_pairs1": s(pairs - triple, 4),
        "a1": a1, "b1": b1, "a2": a2, "b2": b2, "tt": tt, "L1": L1, "L2": L2, "u": u, "leg": leg,
        "area": L1 * L2, "tri": s(leg ** 2 / 2, 3),
        "ok": s(ok, 4), "comp": s(comp, 4), "w_unif": s((tt - a1 - a2) / (L1 + L2), 4),
        "w_noclip": s((u * L1 - L1 ** 2 / 2) / (L1 * L2), 4), "w_nohalf": s(1 - leg ** 2 / (L1 * L2), 4),
        "noclip_area": s(u * L1 - L1 ** 2 / 2, 1),
    }


def p03():
    mean, k, t = 200, 3, 900
    lt = t / mean                         # 4.5
    e45, e15 = 0.01111, 0.2231            # e^{-4.5}, e^{-1.5}（問題文に示す値）
    poly = 1 + lt + lt ** 2 / 2
    ans = e45 * poly
    # (b) メトロポリス法
    n, T = 8, 16.0                        # 100日単位の寿命合計
    a0, b0 = 2, 4                         # 事前ガンマ（形状 2、レート 4）
    cur, prop = 0.5, 0.6
    shape = n + a0 - 1                    # θ の指数 9
    rate = T + b0                         # 20
    pw8, pw9, pw10 = 4.300, 5.160, 6.192  # 1.2^8, 1.2^9, 1.2^10
    e16, e20 = 0.2019, 0.1353             # e^{-1.6}, e^{-2.0}
    acc = pw9 * e20
    return {
        "mean": mean, "k": k, "t": t, "lt": s(lt, 1), "lt2": s(lt ** 2 / 2, 3), "poly": s(poly, 3),
        "e45": e45, "e15": e15, "ans": s(ans, 3), "ans5": s(ans, 5),
        "w_k2": s(e45 * (1 + lt), 3), "w_nohalf": s(e45 * (1 + lt + lt ** 2), 3),
        "w_comp": s(1 - ans, 3), "w_exp": s(e15, 3), "mean3": mean * k,
        "n": n, "Td": int(T * 100), "T": s(T, 0), "a0": a0, "b0": b0, "cur": s(cur, 2), "prop": s(prop, 2),
        "shape": shape, "rate": s(rate, 0), "shape1": shape + 1, "diff": s(prop - cur, 1),
        "rdiff": s(rate * (prop - cur), 1), "ratio": s(prop / cur, 1),
        "pw8": s(pw8, 3), "pw9": s(pw9, 3), "pw10": s(pw10, 3), "e16": s(e16, 4), "e20": s(e20, 4),
        "acc": s(acc, 3), "acc5": s(acc, 5),
        "w_noprior": s(pw8 * e16, 3), "w_shape": s(pw10 * e20, 3), "w_rev": "1", "w_rej": s(1 - acc, 3),
        "rev": s(1 / acc, 2),
    }


def p04():
    loc, theta, n = 10, 3, 365
    ln365, mlog98, ln0202 = 5.900, 0.0202, -3.902   # ln 365, -ln 0.98, ln 0.0202（問題文に示す）
    ln002 = -3.912                                   # ln 0.02（解説の誤答例に使う）
    y = ln365 - ln0202
    x = loc + theta * y
    return {
        "loc": loc, "theta": theta, "n": n, "ln365": s(ln365, 3), "mlog98": s(mlog98, 4), "ln0202": s(ln0202, 3),
        "ln002": s(ln002, 3), "ln002a": s(-ln002, 3), "y": s(y, 3), "ty": s(theta * y, 3), "x": r(x, 1), "x3": s(x, 3), "xs": s(x, 1),
        "w_noloc": s(theta * y, 1), "w_day": s(loc - theta * ln002, 1),
    }


def p05():
    n, sd = 16, 0.40
    s2 = sd ** 2
    S = (n - 1) * s2
    lo15, hi15 = 6.262, 27.49        # χ²(15) の下側2.5%点・上側2.5%点
    lo16, hi16 = 6.908, 28.85        # χ²(16)
    up = sqrt(S / lo15)
    low = sqrt(S / hi15)
    # (b) MSE 最小の割る数
    return {
        "n": n, "sd": s(sd, 2), "s2": s(s2, 2), "df": n - 1, "S": s(S, 1),
        "lo15": lo15, "hi15": hi15, "lo16": lo16, "hi16": hi16,
        "up2": s(S / lo15, 5), "up": r(up, 3), "up_s": s(up, 3), "up5": s(up, 5),
        "low": s(low, 3), "w_df16": s(sqrt(n * s2 / lo16), 3), "w_var": s(S / lo15, 3),
        "n1": n + 1, "nm1": n - 1, "nm2": n - 2, "np2": n + 2,
        "e17": s(S / (n + 1), 3), "e16": s(S / n, 3), "e15": s(S / (n - 1), 3), "e14": s(S / (n - 2), 3),
        "e18": s(S / (n + 2), 3),
        "m17": s(2 / (n + 1), 4), "m15": s(2 / (n - 1), 4), "m16": s((2 * n - 1) / n ** 2, 4),
    }


def p06():
    # (a) ホテリングの T²
    n, p = 12, 2
    d1, d2 = 1.2, 0.8
    s11, s12, s22 = 4.0, -1.5, 2.25
    det = s11 * s22 - s12 ** 2
    q = (s22 * d1 ** 2 - 2 * s12 * d1 * d2 + s11 * d2 ** 2) / det
    q_pos = (s22 * d1 ** 2 + 2 * s12 * d1 * d2 + s11 * d2 ** 2) / det   # 共分散の符号を逆に
    T2 = n * q
    F = (n - p) / (p * (n - 1)) * T2
    T2pos = n * q_pos
    Fpos = (n - p) / (p * (n - 1)) * T2pos
    F210, F211, t11 = 4.103, 3.982, 2.201
    t1 = d1 / sqrt(s11 / n)
    t2 = d2 / sqrt(s22 / n)
    # (b) 重心法
    pts = {"A": (0, 0), "B": (1, 1), "C": (3, 0), "D": (2, 3)}
    def d2f(a, b):
        return (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2
    dAB, dAC, dBC, dAD, dBD, dCD = (d2f(*k) for k in ("AB", "AC", "BC", "AD", "BD", "CD"))
    cen = 0.5 * dAC + 0.5 * dBC - 0.25 * dAB
    cenD = 0.5 * dAD + 0.5 * dBD - 0.25 * dAB
    # (c) 2因子モデル
    L = {"NO2": (0.8, 0.3), "PM": (0.7, 0.4), "O3": (-0.2, 0.7), "SO2": (0.6, 0.1)}
    comPM = L["PM"][0] ** 2 + L["PM"][1] ** 2
    corr = L["NO2"][0] * L["PM"][0] + L["NO2"][1] * L["PM"][1]
    v1 = sum(v[0] ** 2 for v in L.values())
    return {
        "n": n, "p": p, "d1": s(d1, 1), "d2": s(d2, 1), "s11": s(s11, 2), "s12": s(s12, 2), "s22": s(s22, 2),
        "det": s(det, 2), "is11": s(s22, 2), "is12": s(-s12, 2), "is22": s(s11, 2),
        "qa": s(s22 * d1 ** 2, 2), "qb": s(-2 * s12 * d1 * d2, 2), "qc": s(s11 * d2 ** 2, 2),
        "qnum": s(q * det, 2), "q": s(q, 4),
        "T2": s(T2, 2), "F": s(F, 2), "Fc": s((n - p) / (p * (n - 1)), 4), "nmp": n - p, "nm1": n - 1,
        "T2pos": s(T2pos, 2), "Fpos": s(Fpos, 2), "T2non": s(q, 2), "Fnon": s((n - p) / (p * (n - 1)) * q, 2),
        "Fhalf": s(T2 / p, 2), "F210": F210, "F211": F211, "t11": t11, "t1": s(t1, 2), "t2": s(t2, 2),
        "dAB": dAB, "dAC": dAC, "dBC": dBC, "dAD": dAD, "dBD": dBD, "dCD": dCD,
        "cen": s(cen, 1), "cenD": s(cenD, 1), "w_avg": s((dAC + dBC) / 2, 1), "w_min": s(min(dAC, dBC), 1),
        "w_max": s(max(dAC, dBC), 1), "w_half": s(0.5 * dAC + 0.5 * dBC - 0.5 * dAB, 1),
        "hAC": s(0.5 * dAC, 1), "hBC": s(0.5 * dBC, 1), "qAB": s(0.25 * dAB, 1),
        "l11": s(L["NO2"][0], 1), "l12": s(L["NO2"][1], 1), "l21": s(L["PM"][0], 1), "l22": s(L["PM"][1], 1),
        "l31": s(L["O3"][0], 1), "l32": s(L["O3"][1], 1), "l41": s(L["SO2"][0], 1), "l42": s(L["SO2"][1], 1),
        "comPM": s(comPM, 2), "uniqPM": s(1 - comPM, 2), "sumPM": s(L["PM"][0] + L["PM"][1], 1),
        "corr": s(corr, 2), "corr1": s(L["NO2"][0] * L["PM"][0], 2),
        "v1": s(v1, 2), "v1half": s(v1 / 2 * 100, 1), "v1q": s(v1 / 4 * 100, 2),
    }


def p07():
    x = [3, 7, 2, 9, 4, 1, 8, 5, 2, 11, 3, 5]
    n = len(x)
    xb = sum(x) / n
    ss = sum((v - xb) ** 2 for v in x)
    D = ss / xb
    s2 = ss / (n - 1)
    c05, c01 = 19.68, 24.72          # χ²(11)
    assert c05 < D < c01
    return {
        "xs": " | ".join(str(v) for v in x), "n": n, "sum": sum(x), "xb": s(xb, 1), "ss": s(ss, 0),
        "D": s(D, 1), "df": n - 1, "c05": c05, "c01": c01, "s2": s(s2, 2), "ratio": s(s2 / xb, 2),
        "w_D11": s(s2 / xb, 2),
    }


def p08():
    ph1, ph2 = 0.5, 0.3
    r1 = ph1 / (1 - ph2)
    r2 = ph1 * r1 + ph2
    r3 = ph1 * r2 + ph2 * r1
    # (b) 連の検定
    seq = "+++---++++----++-----++++---++"
    runs = 1 + sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    n1, n2 = seq.count("+"), seq.count("-")
    N = n1 + n2
    ER = 2 * n1 * n2 / N + 1
    VR = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (N ** 2 * (N - 1))
    z = (runs - ER) / sqrt(VR)
    return {
        "ph1": ph1, "ph2": ph2, "one_m": s(1 - ph2, 1), "r1": s(r1, 4), "r1_3": s(r1, 3), "r2": s(r2, 3),
        "r2_4": s(r2, 4), "pr1": s(ph1 * r1, 4), "r3": s(r3, 3),
        "w_r2a": s(ph1 ** 2 + ph2, 3), "w_r2b": s(ph2, 3), "w_r2c": s(r1, 3), "w_r2d": s(r1 ** 2, 3),
        "seq": " ".join(seq), "N": N, "n1": n1, "n2": n2, "R": runs, "ER": s(ER, 0), "VR": s(VR, 4),
        "VRnum": 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2), "VRden": N ** 2 * (N - 1),
        "sdR": s(sqrt(VR), 4), "z": s(z, 2), "w_zvar": s((runs - ER) / VR, 2),
        "w_zE": s((runs - n1 * n2 / N) / sqrt(VR), 2), "nn2": 2 * n1 * n2, "nn": n1 * n2,
    }


def p09():
    lam, mu, c = 3.0, 2.0, 2
    a = lam / mu
    rho = lam / (c * mu)
    tail = a ** c / (math.factorial(c) * (1 - rho))
    P0 = 1 / (1 + a + tail)
    Pw = tail * P0
    Wq = Pw / (c * mu - lam)
    Lq = Pw * rho / (1 - rho)
    W = Wq + 1 / mu
    Wq_mm1 = rho / (c * mu - lam)          # 速さ2倍の1台とみなした M/M/1
    return {
        "lam": s(lam, 0), "mu": s(mu, 0), "c": c, "a": s(a, 1), "rho": s(rho, 2), "mins": 30,
        "a2": s(a ** 2, 2), "a2h": s(a ** 2 / 2, 3), "tail": s(tail, 1), "den": s(1 + a + tail, 1),
        "P0": "\\frac{1}{7}", "P0d": s(P0, 4), "P1": s(a * P0, 4), "Pw": s(Pw, 4), "Pw3": s(Pw, 3),
        "Lq": s(Lq, 3), "Wq_h": s(Wq, 4), "Wq": s(Wq * 60, 1), "cmu": s(c * mu, 0),
        "w_mm1": s(Wq_mm1 * 60, 1), "w_W": s(W * 60, 1), "w_div": s(Pw / (c * mu) * 60, 1),
        "w_rho": s(rho / (1 - rho) / lam * 60, 1),
    }


def p10():
    n = 40
    rss_f, rss_r = 1200.0, 1800.0
    p_f, p_r = 5, 2
    q = p_f - p_r
    dfe = n - p_f
    F = ((rss_r - rss_f) / q) / (rss_f / dfe)
    # (b) 外部スチューデント化残差
    e, h = 14.0, 0.30
    s2 = rss_f / dfe
    sd = sqrt(s2)
    r_int = e / (sd * sqrt(1 - h))
    s2i = (rss_f - e ** 2 / (1 - h)) / (dfe - 1)
    t_ext = e / (sqrt(s2i) * sqrt(1 - h))
    # (c) 操作変数法
    czy, czx, cxy, vx, vz = -0.024, 0.080, 0.010, 0.050, 0.100
    return {
        "n": n, "rss_f": s(rss_f, 0), "rss_r": s(rss_r, 0), "p_f": p_f, "p_r": p_r, "q": q, "dfe": dfe,
        "dfr": n - p_r, "drss": s(rss_r - rss_f, 0), "num": s((rss_r - rss_f) / q, 1), "mse": s(rss_f / dfe, 4),
        "F": r(F, 2), "F_s": s(F, 2), "F4": s(F, 4), "w_F36": s(((rss_r - rss_f) / q) / (rss_f / (n - 4)), 2),
        "w_Fnoq": s((rss_r - rss_f) / (rss_f / dfe), 2), "Fc": 2.874,
        "e": s(e, 1), "h": s(h, 2), "omh": s(1 - h, 2), "s2": s(s2, 4), "sd": s(sd, 4), "sq1h": s(sqrt(1 - h), 4),
        "r_int": s(r_int, 2), "e2h": s(e ** 2 / (1 - h), 0), "num_i": s(rss_f - e ** 2 / (1 - h), 0),
        "dfe1": dfe - 1, "s2i": s(s2i, 4), "sdi": s(sqrt(s2i), 4), "den_i": s(sqrt(s2i) * sqrt(1 - h), 4),
        "t_ext": s(t_ext, 2), "w_raw": s(e / sd, 2), "w_1h": s(e / (sd * (1 - h)), 2),
        "w_1ph": s(e / (sd * sqrt(1 + h)), 2),
        "czy": czy, "czx": czx, "cxy": cxy, "vx": vx, "vz": vz,
        "iv": s(czy / czx, 2), "ols": s(cxy / vx, 2), "w_inv": s(czx / czy, 2), "w_vx": s(czy / vx, 2),
        "w_vz": s(czy / vz, 2), "fs": s(czx / vz, 2), "rf": s(czy / vz, 2),
    }


def p11():
    th1, th2 = -0.5, 1.0
    b_d, b_m = 0.2, 0.8
    dist, meet = 5, 1
    eta = b_d * dist + b_m * meet
    e23, e08, e28 = 9.974, 2.226, 16.44   # e^{2.3}, e^{0.8}, e^{2.8}
    P1 = 1 / (1 + e23)                    # L(-2.3)
    P2c = 1 / (1 + e08)                   # L(-0.8)
    p3 = 1 - P2c
    return {
        "th1": th1, "th2": th2, "b_d": b_d, "b_m": b_m, "dist": dist, "eta": s(eta, 1),
        "g1": s(th1 - eta, 1), "g2": s(th2 - eta, 1), "e23": e23, "e08": e08, "e28": e28,
        "P1": s(P1, 4), "P2c": s(P2c, 4), "p2": s(P2c - P1, 3), "p3": s(p3, 3), "p3_4": s(p3, 4),
        "w_p2": s(P2c - P1, 3), "w_cum": s(P2c, 3), "w_sign": s(1 / (1 + e28), 3), "w_nomeet": s(0.5, 3),
        "g2s": s(th2 + eta, 1),
    }


def p12():
    k = 4
    ss_r, ss_c, ss_t, ss_tot = 30.0, 48.0, 45.0, 141.0
    ss_e = ss_tot - ss_r - ss_c - ss_t
    df_t = k - 1
    df_e = (k - 1) * (k - 2)
    F = (ss_t / df_t) / (ss_e / df_e)
    F36, F39, F312 = 4.757, 3.863, 3.490
    assert F > F36
    w9 = (ss_t / df_t) / (ss_e / 9)                     # 誤差の自由度を 9 と数えた
    w1 = (ss_t / df_t) / ((ss_e + ss_r + ss_c) / 12)    # 行と列を誤差に入れた
    w_rb = (ss_t / df_t) / ((ss_e + ss_c) / 9)          # 行（住宅）だけをブロックにした
    return {
        "k": k, "kk": k * k, "ss_r": s(ss_r, 1), "ss_c": s(ss_c, 1), "ss_t": s(ss_t, 1), "ss_tot": s(ss_tot, 1),
        "ss_e": s(ss_e, 1), "df_t": df_t, "df_e": df_e, "df_tot": k * k - 1, "ms_t": s(ss_t / df_t, 1),
        "ms_e": s(ss_e / df_e, 1), "F": s(F, 2), "F36": F36, "F39": F39, "F312": F312,
        "w9": s(w9, 2), "w1": s(w1, 2), "w_raw": s(ss_t / ss_e, 2), "w_rb": s(w_rb, 2),
        "ss_e1": s(ss_e + ss_r + ss_c, 1), "ss_rb": s(ss_e + ss_c, 1), "ms_e9": s(ss_e / 9, 1),
    }


def p13():
    N, S, e, z = 8000, 160, 10, 1.96
    n0 = (z * S / e) ** 2
    n = n0 / (1 + n0 / N)
    nreq = math.ceil(n)
    # (b) 数量化I類
    const, b_house = 250, 80
    b2, b3 = 60, 130
    w_house, w_apt = 0.4, 0.6
    w1, w2, w3 = 0.3, 0.3, 0.4
    mh = w_house * b_house
    mp = w2 * b2 + w3 * b3
    const_n = const + mh + mp
    # (c) IPW
    g = {"house": dict(n=400, n1=240, y1=380, n0=160, y0=420), "apt": dict(n=600, n1=120, y1=250, n0=480, y0=270)}
    Ntot = sum(v["n"] for v in g.values())
    e_h = g["house"]["n1"] / g["house"]["n"]
    e_a = g["apt"]["n1"] / g["apt"]["n"]
    t1h = g["house"]["n1"] * g["house"]["y1"] / e_h
    t1a = g["apt"]["n1"] * g["apt"]["y1"] / e_a
    t0h = g["house"]["n0"] * g["house"]["y0"] / (1 - e_h)
    t0a = g["apt"]["n0"] * g["apt"]["y0"] / (1 - e_a)
    m1 = (t1h + t1a) / Ntot
    m0 = (t0h + t0a) / Ntot
    ate = m1 - m0
    naive1 = (g["house"]["n1"] * g["house"]["y1"] + g["apt"]["n1"] * g["apt"]["y1"]) / (g["house"]["n1"] + g["apt"]["n1"])
    naive0 = (g["house"]["n0"] * g["house"]["y0"] + g["apt"]["n0"] * g["apt"]["y0"]) / (g["house"]["n0"] + g["apt"]["n0"])
    # 傾向スコアをかけてしまう誤り
    wr1 = (g["house"]["n1"] * g["house"]["y1"] * e_h + g["apt"]["n1"] * g["apt"]["y1"] * e_a) / (g["house"]["n1"] * e_h + g["apt"]["n1"] * e_a)
    return {
        "N": N, "S": S, "e": e, "z": z, "zS": s(z * S / e, 2), "n0": s(n0, 2), "n0N": s(n0 / N, 5),
        "nraw": s(n, 2), "nreq": nreq, "w_nofpc": math.ceil(n0),
        "const": const, "b_house": b_house, "b2": b2, "b3": b3,
        "w_house": w_house, "w_apt": w_apt, "w1": w1, "w2": w2, "w3": w3,
        "mh": s(mh, 0), "mp": s(mp, 0), "const_n": s(const_n, 0),
        "sc_house": s(b_house - mh, 0), "sc_apt": s(-mh, 0), "sc1": s(-mp, 0), "sc2": s(b2 - mp, 0), "sc3": s(b3 - mp, 0),
        "w_un_mp": s((0 + b2 + b3) / 3, 2), "w_un_sc2": s(b2 - (0 + b2 + b3) / 3, 1),
        "w_un_const": s(const + b_house / 2 + (0 + b2 + b3) / 3, 1),
        "w_c_noh": s(const + mp, 0), "w_c_neg": s(const - mh - mp, 0),
        "Nh": g["house"]["n"], "Na": g["apt"]["n"], "Ntot": Ntot,
        "h1": g["house"]["n1"], "hy1": g["house"]["y1"], "h0": g["house"]["n0"], "hy0": g["house"]["y0"],
        "a1": g["apt"]["n1"], "ay1": g["apt"]["y1"], "a0": g["apt"]["n0"], "ay0": g["apt"]["y0"],
        "e_h": s(e_h, 1), "e_a": s(e_a, 1), "ome_h": s(1 - e_h, 1), "ome_a": s(1 - e_a, 1),
        "t1h": s(t1h, 0), "t1a": s(t1a, 0), "t0h": s(t0h, 0), "t0a": s(t0a, 0),
        "m1": s(m1, 1), "m0": s(m0, 1), "ate": r(ate, 1), "ate_s": s(ate, 1),
        "naive1": s(naive1, 2), "naive0": s(naive0, 2), "naive": s(naive1 - naive0, 2),
        "std": s((g["house"]["n"] * (g["house"]["y1"] - g["house"]["y0"]) + g["apt"]["n"] * (g["apt"]["y1"] - g["apt"]["y0"])) / Ntot, 1),
        "wr1": s(wr1, 1),
    }


def p14():
    Q = [4.12, 4.20, 4.05, 4.18, 4.15]
    U = [0.0016, 0.0018, 0.0015, 0.0017, 0.0019]
    m = len(Q)
    qb = sum(Q) / m
    W = sum(U) / m
    B = sum((q - qb) ** 2 for q in Q) / (m - 1)
    T = W + (1 + 1 / m) * B
    se = sqrt(T)
    Bm = sum((q - qb) ** 2 for q in Q) / m
    return {
        "Q1": s(Q[0], 2), "Q2": s(Q[1], 2), "Q3": s(Q[2], 2), "Q4": s(Q[3], 2), "Q5": s(Q[4], 2),
        "U1": s(U[0], 4), "U2": s(U[1], 4), "U3": s(U[2], 4), "U4": s(U[3], 4), "U5": s(U[4], 4),
        "m": m, "qb": s(qb, 2), "W": s(W, 4), "sqd": s(sum((q - qb) ** 2 for q in Q), 4), "B": s(B, 5),
        "fac": s(1 + 1 / m, 1), "fB": s((1 + 1 / m) * B, 5), "T": s(T, 5), "se": r(se, 3), "se_s": s(se, 3),
        "se5": s(se, 5), "w_nofac": s(sqrt(W + B), 3), "w_W": s(sqrt(W), 3), "w_Bm": s(sqrt(W + (1 + 1 / m) * Bm), 3),
        "d1": s(Q[0] - qb, 2), "d2": s(Q[1] - qb, 2), "d3": s(Q[2] - qb, 2), "d4": s(Q[3] - qb, 2), "d5": s(Q[4] - qb, 2),
    }


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14]
    return {f"s20p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
