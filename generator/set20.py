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
    f = sqrt(1 + 1 / n)
    up = xb + t15 * sd * f
    return {
        "n": n, "xb": s(xb, 2), "sd": s(sd, 2), "s2": s(s2, 2), "df": n - 1, "S": s(S, 1),
        "t15": t15, "t16": t16, "z": s(z, 2), "f": s(f, 5), "half": s(t15 * sd * f, 5), "up": r(up, 3), "up_s": s(up, 3),
        "up5": s(up, 5), "lo_s": s(xb - t15 * sd * f, 3),
        "w_ci": s(xb + t15 * sd / sqrt(n), 3), "w_t16": s(xb + t16 * sd * f, 3), "w_z": s(xb + z * sd * f, 3),
        "n1": n + 1, "nm1": n - 1, "nm2": n - 2, "np2": n + 2,
        "e17": s(S / (n + 1), 3), "e16": s(S / n, 3), "e15": s(S / (n - 1), 3), "e14": s(S / (n - 2), 3),
        "e18": s(S / (n + 2), 3),
        "m17": s(2 / (n + 1), 4), "m15": s(2 / (n - 1), 4), "m16": s((2 * n - 1) / n ** 2, 4),
    }


def p06():
    names = "ABCDE"
    D = [[0, 1.3, 2.8, 3.9, 2.6], [1.3, 0, 1.5, 2.6, 1.9], [2.8, 1.5, 0, 1.4, 1.9],
         [3.9, 2.6, 1.4, 0, 3.3], [2.6, 1.9, 1.9, 3.3, 0]]
    out = {f"d{names[i]}{names[j]}": s(D[i][j], 1) for i in range(5) for j in range(5) if i < j}
    r12, r13, r23 = 0.56, 0.48, 0.42
    l1sq = r12 * r13 / r23
    l2sq = r12 * r23 / r13
    l3sq = r13 * r23 / r12
    out.update({
        "h_single": "1.5", "h_complete": "3.9", "h_avg": s(8 / 3, 2), "h_final": "1.9", "h_cd": "1.4",
        "avg_num": s(2.8 + 3.9 + 2.6 + 2.6 + 1.5 + 2.6 - 2.6 - 2.6 + 0, 1),
        "r12": r12, "r13": r13, "r23": r23, "l1sq": s(l1sq, 2), "l2sq": s(l2sq, 2), "l3sq": s(l3sq, 2),
        "l1": s(sqrt(l1sq), 1), "l2": s(sqrt(l2sq), 1), "l3": s(sqrt(l3sq), 1),
        "u2": s(1 - l2sq, 2), "w_1ml": s(1 - sqrt(l2sq), 2), "w_1mr": s(1 - r12, 2), "num2": s(r12 * r23, 4),
    })
    return out


def p07():
    n = 55
    rk = [0.30, 0.20, 0.10]
    terms = [rk[k] ** 2 / (n - k - 1) for k in range(3)]
    lb = n * (n + 2) * sum(terms)
    bp = n * sum(v ** 2 for v in rk)
    c3, c4 = 7.815, 9.488
    assert bp < c3 < lb < c4
    sk, ku = 0.6, 4.2
    jb = n / 6 * (sk ** 2 + (ku - 3) ** 2 / 4)
    c2 = 5.991
    return {
        "n": n, "r1": s(rk[0], 2), "r2": s(rk[1], 2), "r3": s(rk[2], 2), "nn2": n * (n + 2),
        "a1": s(terms[0], 7), "a2": s(terms[1], 7), "a3": s(terms[2], 7), "asum": s(sum(terms), 7),
        "lb": s(lb, 2), "bp": s(bp, 2), "c3": c3, "c4": c4, "w_sq": s(sum(v ** 2 for v in rk), 2),
        "nm1": n - 1, "nm2": n - 2, "nm3": n - 3,
        "sk": sk, "ku": ku, "ek": s(ku - 3, 1), "sk2": s(sk ** 2, 2), "ek2": s((ku - 3) ** 2 / 4, 2), "n6": s(n / 6, 4),
        "jb": s(jb, 2), "c2": c2, "w_raw": s(n / 6 * (sk ** 2 + ku ** 2 / 4), 2), "w_non": s(sk ** 2 + (ku - 3) ** 2 / 4, 2),
    }


def p08():
    sig2, t, a = 4.0, 4.0, 6.0
    sd = sqrt(sig2 * t)
    zz = a / sd
    zw = a / (sqrt(sig2) * t)
    phi = {1.50: 0.9332, 0.75: 0.7734}
    assert r(zz, 2) in phi and r(zw, 2) in phi
    ans = 2 * (1 - phi[r(zz, 2)])
    return {
        "sig2": s(sig2, 0), "sig": s(sqrt(sig2), 0), "t": s(t, 0), "a": s(a, 0), "sd": s(sd, 0), "z": s(zz, 2),
        "ph15": phi[1.50], "ph075": phi[0.75], "tail": s(1 - phi[1.50], 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_one": s(1 - phi[1.50], 3), "w_t": s(2 * (1 - phi[0.75]), 3), "w_t1": s(1 - phi[0.75], 3), "w_comp": s(1 - ans, 3),
        "zw": s(zw, 2),
    }


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
    r12, r1y, r2y = S12 / sqrt(S11 * S22), S1y / sqrt(S11 * Syy), S2y / sqrt(S22 * Syy)
    lam1, lam2 = 1 + r12, 1 - r12
    pcr = (r1y + r2y) / (2 * lam1)
    ols1 = (r1y - r12 * r2y) / (1 - r12 ** 2)
    ols2 = (r2y - r12 * r1y) / (1 - r12 ** 2)
    pc2 = (r1y - r2y) / (2 * lam2)
    return {
        "n": n, "S11": s(S11, 0), "S22": s(S22, 0), "S12": s(S12, 0), "Syy": s(Syy, 0), "S1y": s(S1y, 0), "S2y": s(S2y, 0),
        "det": s(det, 0), "b1": s(b1, 4), "b2": s(b2, 4), "b1n": s(S22 * S1y - S12 * S2y, 0), "b2n": s(S11 * S2y - S12 * S1y, 0),
        "reg1": s(b1 * S1y, 1), "reg2": s(b2 * S2y, 1), "rss": s(rss, 1), "dfe": n - 3, "s2": s(s2, 4),
        "c11": s(c11, 6), "se1": s(se1, 4), "t1": r(t1, 2), "t1_s": s(t1, 2), "t1_4": s(t1, 4),
        "w_simple": s(b1 / sqrt(s2 / S11), 2), "w_df": s(b1 / sqrt(rss / (n - 2) * c11), 2), "slope": s(S1y / S11, 2),
        "dx1": s(dx1, 0), "dx2": s(dx2, 0), "qa": s(S22 * dx1 ** 2, 0), "qb": s(-2 * S12 * dx1 * dx2, 0), "qc": s(S11 * dx2 ** 2, 0),
        "qn": s(q * det, 0), "q": s(q, 5), "inv_n": s(1 / n, 5), "h": s(h, 3), "h5": s(h, 5), "cut": s(2 * 3 / n, 1),
        "w_ind": s(h_ind, 3), "w_non": s(q, 3), "w_sign": s(h_sign, 3),
        "r12": s(r12, 1), "r1y": s(r1y, 1), "r2y": s(r2y, 1), "lam1": s(lam1, 1), "lam2": s(lam2, 1),
        "pcr": s(pcr, 3), "pcr4": s(pcr, 4), "gam": s((r1y + r2y) / sqrt(2) / lam1, 4), "vr": s((r1y + r2y) / sqrt(2), 4),
        "ols1": s(ols1, 3), "ols2": s(ols2, 3), "w_nolam": s((r1y + r2y) / 2, 3), "w_nonorm": s((r1y + r2y) / lam1, 3),
        "pc2": s(pc2, 3), "mpc2": s(-pc2, 3), "rsum": s(r1y + r2y, 1),
    }


def p10():
    lam0, months, tot = 5, 12, 78
    mu0 = lam0 * months
    zz = (tot - mu0) / sqrt(mu0)
    y = [6, 4, 11, 4]
    mu = [4, 6, 9, 6]
    ln15, ln23, ln119 = 0.4055, -0.4055, 0.2007   # log 1.5, log(2/3), log(11/9)
    lnr = [ln15, ln23, ln119, ln23]
    tsum = sum(yi * l for yi, l in zip(y, lnr))
    D = 2 * tsum
    return {
        "lam0": lam0, "months": months, "tot": tot, "mu0": mu0, "sq": s(sqrt(mu0), 3), "z": s(zz, 2), "z4": s(zz, 4),
        "mbar": s(tot / months, 1), "w_mon": s((tot / months - lam0) / sqrt(lam0), 2), "w_var": s((tot - mu0) / mu0, 2),
        "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3], "m1": mu[0], "m2": mu[1], "m3": mu[2], "m4": mu[3],
        "ln15": s(ln15, 4), "ln23": s(ln23, 4), "ln119": s(ln119, 4),
        "k1": s(y[0] * ln15, 4), "k2": s(y[1] * ln23, 4), "k3": s(y[2] * ln119, 4), "k4": s(y[3] * ln23, 4),
        "tsum": s(tsum, 4), "D": s(D, 2), "w_half": s(tsum, 2), "w_sign": s(-D, 2), "w_log10": s(D / math.log(10), 2),
        "w_noy": s(2 * sum(lnr), 2), "ysum": sum(y), "musum": sum(mu),
    }


def p11():
    ss = dict(blk=12.0, A=48.0, E1=16.0, B=30.0, AB=4.0, E2=9.0)
    df = dict(blk=2, A=2, E1=4, B=1, AB=2, E2=6)
    msA = ss["A"] / df["A"]
    FA = msA / (ss["E1"] / df["E1"])
    F_e2 = msA / (ss["E2"] / df["E2"])
    ms_pool = (ss["E1"] + ss["E2"]) / (df["E1"] + df["E2"])
    F_pool = msA / ms_pool
    F24, F26, F210 = 6.944, 5.143, 4.103
    assert FA < F24 and F_e2 > F26 and F_pool > F210 and FA > F26
    up = [38, 41, 45]
    no = [43, 47, 52]
    allv = up + no
    obs = sum(up)
    ntot = math.comb(6, 3)
    cnt = sum(1 for c in combinations(allv, 3) if sum(c) <= obs)
    cnt_lt = sum(1 for c in combinations(allv, 3) if sum(c) < obs)
    cnt_ge = sum(1 for c in combinations(allv, 3) if sum(c) >= obs)
    return {
        **{f"ss_{k}": s(v, 1) for k, v in ss.items()}, **{f"df_{k}": v for k, v in df.items()},
        "ss_tot": s(sum(ss.values()), 1), "df_tot": sum(df.values()),
        "ms_A": s(msA, 1), "ms_E1": s(ss["E1"] / 4, 1), "ms_E2": s(ss["E2"] / 6, 1), "ms_pool": s(ms_pool, 1),
        "FA": s(FA, 2), "F_e2": s(F_e2, 2), "F_pool": s(F_pool, 2), "F_raw": s(ss["A"] / ss["E1"], 2),
        "F24": F24, "F26": F26, "F210": F210,
        "u1": up[0], "u2": up[1], "u3": up[2], "n1": no[0], "n2": no[1], "n3": no[2], "obs": obs, "obs_no": sum(no),
        "mu_up": s(obs / 3, 2), "mu_no": s(sum(no) / 3, 2), "cnt": cnt, "ntot": ntot,
        "p": s(cnt / ntot, 3), "w_lt": s(cnt_lt / ntot, 3), "w_two": s(2 * cnt / ntot, 3), "w_perm": s(cnt / 120, 3),
        "w_ge": s(cnt_ge / ntot, 3),
    }


def p12():
    n, n1, ybar1, n2, m2, ybar2 = 400, 280, 300.0, 120, 30, 360.0
    est = (n1 * ybar1 + n2 * ybar2) / n
    chi, df, chi0, df0 = 36.0, 24, 420.0, 36
    cfi = 1 - (chi - df) / (chi0 - df0)
    nfi = 1 - chi / chi0
    tli = (chi0 / df0 - chi / df) / (chi0 / df0 - 1)
    aXT, a, b, c, d = 0.4, 0.5, -0.6, -0.2, 0.3
    total = c + a * b
    naive = total + d * aXT
    return {
        "n": n, "n1": n1, "ybar1": s(ybar1, 0), "n2": n2, "m2": m2, "ybar2": s(ybar2, 0),
        "w1": s(n1 / n, 1), "w2": s(n2 / n, 1), "est": r(est, 1), "est_s": s(est, 1),
        "w_resp": s(ybar1, 1), "w_avg": s((ybar1 + ybar2) / 2, 1), "w_sub": s((n1 * ybar1 + m2 * ybar2) / (n1 + m2), 1),
        "chi": s(chi, 0), "df": df, "chi0": s(chi0, 0), "df0": df0, "nc": s(chi - df, 0), "nc0": s(chi0 - df0, 0),
        "cfi": s(cfi, 3), "cfi5": s(cfi, 5), "nfi": s(nfi, 3), "tli": s(tli, 3), "w_cfi0": s(1 - (chi - df) / chi0, 3),
        "w_ratio": s(1 - (chi / df) / (chi0 / df0), 3),
        "aXT": aXT, "a": a, "b": b, "c": c, "d": d, "ab": s(a * b, 2), "total": s(total, 2), "bias": s(d * aXT, 2),
        "naive": s(naive, 2), "w_bias": s(total - d * aXT, 2), "c_s": s(c, 2),
    }


def p13():
    n, r_, s2 = 30, 20, 2.0
    v = (r_ - 1) * s2 / (n - 1)
    return {"n": n, "r": r_, "miss": n - r_, "s2": s(s2, 1), "rm1": r_ - 1, "nm1": n - 1, "ss": s((r_ - 1) * s2, 1),
            "v": r(v, 2), "v_s": s(v, 2), "v4": s(v, 4), "ratio": s((r_ - 1) / (n - 1), 4),
            "w_n": s(r_ * s2 / n, 2), "w_sd": s(sqrt(v), 2)}


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s20p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
