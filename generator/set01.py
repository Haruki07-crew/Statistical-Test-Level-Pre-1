"""模擬試験 第1回（set01）の数値。values() が {大問id: {名前: 値}} を返す。

問題文に示した分布の値（Φ・χ²・F の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) 包除原理
    pa, pb, pc = 0.03, 0.04, 0.02
    pab, pac, pbc, pabc = 0.012, 0.006, 0.008, 0.004
    s1 = pa + pb + pc
    s2 = pab + pac + pbc
    union = s1 - s2 + pabc
    # (b) 超幾何
    N, D, n = 40, 4, 5
    p = D / N
    var_h = n * p * (1 - p) * (N - n) / (N - 1)
    var_b = n * p * (1 - p)
    var_fpc_wrong = n * p * (1 - p) * (N - n) / N
    # (c) 二項分布の正規近似（連続修正）
    m, pr, k = 400, 0.02, 12
    mu = m * pr
    var = m * pr * (1 - pr)
    sd = sqrt(var)
    phi = {0.45: 0.6736, 1.25: 0.8944, 1.43: 0.9236, 1.61: 0.9463}
    z_cc = r((k - 0.5 - mu) / sd, 2)
    z_no = r((k - mu) / sd, 2)
    z_up = r((k + 0.5 - mu) / sd, 2)
    z_var = r((k - 0.5 - mu) / var, 2)
    for z in (z_cc, z_no, z_up, z_var):
        assert z in phi, z
    c_ok = 1 - phi[z_cc]
    return {
        "pa": pa, "pb": pb, "pc": pc, "pab": pab, "pac": pac, "pbc": pbc, "pabc": pabc,
        "s1": s(s1, 2), "s2": s(s2, 3), "union": r(union, 3), "union_s": s(union, 3),
        "u_no3": s(s1 - s2, 3), "u_sum": s(s1, 2), "u_ind": s(1 - (1 - pa) * (1 - pb) * (1 - pc), 4),
        "N": N, "D": D, "n": n, "p": s(p, 1), "q": s(1 - p, 1), "fpc": s((N - n) / (N - 1), 4),
        "fpcN": N - n, "fpcD": N - 1,
        "npq": s(var_b, 2), "varH": s(var_h, 3), "varB": s(var_b, 3), "varW": s(var_fpc_wrong, 3),
        "meanH": s(n * p, 3), "sdH": s(sqrt(var_h), 3), "varH4": s(var_h, 4),
        "m": m, "pr": pr, "k": k, "km": k - 0.5, "kp": k + 0.5, "mu": s(mu, 1), "var": s(var, 2), "sd": s(sd, 1),
        "zcc": s(z_cc, 2), "zno": s(z_no, 2), "zup": s(z_up, 2), "zvar": s(z_var, 2),
        "zno_raw": s((k - mu) / sd, 4),
        "phiA": s(phi[0.45], 4), "phiB": s(phi[1.25], 4), "phiC": s(phi[1.43], 4), "phiD": s(phi[1.61], 4),
        "c_ok": s(c_ok, 3), "c_no": s(1 - phi[z_no], 3), "c_up": s(1 - phi[z_up], 3),
        "c_var": s(1 - phi[z_var], 3), "c_low": s(phi[z_cc], 3), "c_ok4": s(c_ok, 4),
    }


def p02():
    rr, pp = 2, 0.8
    qq = 1 - pp
    g1 = rr * qq / pp                       # G'(1)
    g2 = rr * (rr + 1) * qq ** 2 / pp ** 2  # G''(1)
    varN = g2 + g1 - g1 ** 2
    muX, sdX = 20, 10
    varS = g1 * sdX ** 2 + varN * muX ** 2
    # (c) ガンマ・ポアソンの事後平均
    alpha, theta = 2, 0.25
    t, y = 5, 4
    post = (alpha + y) / (1 / theta + t)
    return {
        "r": rr, "p": s(pp, 1), "q": s(qq, 1), "g1": s(g1, 3), "g2": s(g2, 3), "g1sq": s(g1 ** 2, 3),
        "varN": s(varN, 3), "v_mean": s(g1, 3), "v_E2": s(g2 + g1, 3), "v_g2m": s(g2 - g1 ** 2, 3),
        "v_plus": s(g2 + g1 + g1 ** 2, 3),
        "muX": muX, "sdX": sdX, "varX": sdX ** 2, "muX2": muX ** 2,
        "t1": s(g1 * sdX ** 2, 1), "t2": s(varN * muX ** 2, 1), "varS": s(varS, 1),
        "sdS": r(sqrt(varS), 2), "sdS4": s(sqrt(varS), 4),
        "sdS_wrong": s(sqrt(g1 * sdX ** 2 + g1 * muX ** 2), 2), "sdS_wrong2": s(sqrt(g1 * sdX ** 2), 2),
        "alpha": alpha, "theta": theta, "rate": s(1 / theta, 0), "pmean": s(alpha * theta, 1),
        "t": t, "y": y, "ay": alpha + y, "bt": s(1 / theta + t, 0),
        "post": s(post, 3), "mle": s(y / t, 3), "prior": s(alpha * theta, 3),
        "wrong_rate": s((alpha + y) / (theta + t), 3), "wrong_noalpha": s(y / (1 / theta + t), 3),
        "postscale": s(theta / (1 + t * theta), 4),
    }


def p03():
    n, sy2 = 10, 25.6
    mle = n / sy2
    unb = (n - 1) / sy2
    return {"n": n, "sy2": sy2, "n1": n - 1, "mle": s(mle, 4), "unb": r(unb, 3), "unb4": s(unb, 4),
            "w_n2": s((n - 2) / sy2, 4), "w_p1": s((n + 1) / sy2, 4)}


def p04():
    n1, m1, s1 = 12, 14.2, 6.0
    n2, m2, s2 = 10, 8.5, 5.0
    diff = m1 - m2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    inv = 1 / n1 + 1 / n2
    se = sqrt(sp2 * inv)
    t = diff / se
    t_welch = diff / sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    sp2n = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2)
    t_n = diff / sqrt(sp2n * inv)
    t_nosqrt = diff / (sqrt(sp2) * inv)
    t_sum = diff / (s1 / sqrt(n1) + s2 / sqrt(n2))
    # (b) 例数設計
    sig, delta, za, zb = 6, 4, 1.645, 1.282
    nn = 2 * sig ** 2 * (za + zb) ** 2 / delta ** 2
    return {
        "n1": n1, "m1": m1, "s1": s1, "n2": n2, "m2": m2, "s2": s2, "diff": s(diff, 1),
        "ss1": s((n1 - 1) * s1 ** 2, 0), "ss2": s((n2 - 1) * s2 ** 2, 0), "df": n1 + n2 - 2,
        "sp2": s(sp2, 3), "inv": s(inv, 4), "se": s(se, 4), "seq": s(sp2 * inv, 4),
        "t": s(t, 2), "t_w": s(t_welch, 2), "t_n": s(t_n, 2), "t_ns": s(t_nosqrt, 2), "t_sum": s(t_sum, 2),
        "sig": sig, "delta": delta, "za": za, "zb": zb, "zab": s(za + zb, 3), "zab2": s((za + zb) ** 2, 4),
        "nraw": s(nn, 2), "nreq": math.ceil(nn), "n_one": math.ceil(sig ** 2 * (za + zb) ** 2 / delta ** 2),
        "n_two": math.ceil(2 * sig ** 2 * (1.960 + zb) ** 2 / delta ** 2), "nlow": math.floor(nn),
    }


def p05():
    k, m = 3, 6
    N = k * m
    ssa, sst = 48.0, 138.0
    sse = sst - ssa
    dfa, dfe = k - 1, N - k
    msa, mse = ssa / dfa, sse / dfe
    F = msa / mse
    # (b) 分散比の信頼区間
    na, va, nb, vb = 11, 0.50, 16, 0.20
    fa, fb, f05 = 3.06, 3.52, 2.54  # F_{0.025}(10,15), F_{0.025}(15,10), F_{0.05}(10,15)
    ratio = va / vb
    lo = ratio / fa
    return {
        "k": k, "m": m, "N": N, "ssa": s(ssa, 1), "sst": s(sst, 1), "sse": s(sse, 1),
        "dfa": dfa, "dfe": dfe, "dft": N - 1, "msa": s(msa, 1), "mse": s(mse, 1), "F": s(F, 2),
        "F17": s(msa / (sse / (N - 1)), 2), "Ftot": s(msa / (sst / dfe), 2), "Fraw": s(ssa / sse, 2),
        "Fc": 3.68, "Fc17": 3.59, "Fcrev": 19.43,
        "na": na, "va": va, "nb": nb, "vb": vb, "dfa1": na - 1, "dfb1": nb - 1,
        "fa": fa, "fb": fb, "f05": f05, "ratio": s(ratio, 1),
        "lo": r(lo, 2), "lo4": s(lo, 4), "hi": s(ratio * fb, 2),
        "lo_w": s(ratio / fb, 2), "lo_05": s(ratio / f05, 2),
    }


def p06():
    obs = [38, 82, 54, 26]
    n = sum(obs)
    lam, e = 1.35, 0.2592
    p0 = e
    p1 = p0 * lam
    p2 = p1 * lam / 2
    p3 = 1 - p0 - p1 - p2
    exp = [r(n * q, 2) for q in (p0, p1, p2, p3)]  # 問題文の指示どおり期待度数を小数第2位に丸める
    terms = [(o - x) ** 2 / x for o, x in zip(obs, exp)]
    chi = sum(terms)
    assert 5.991 < chi < 7.815, chi
    # (b) ウォード法
    na, nb = 3, 2
    ca, cb = (2, 4), (5, 8)
    d2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
    w = na * nb / (na + nb)
    return {
        "n": n, "o0": obs[0], "o1": obs[1], "o2": obs[2], "o3": obs[3], "lam": lam, "e": e,
        "p1": s(p1, 4), "p2": s(p2, 4), "p3": s(p3, 4),
        "e0": s(exp[0], 2), "e1": s(exp[1], 2), "e2": s(exp[2], 2), "e3": s(exp[3], 2),
        "c0": s(terms[0], 3), "c1": s(terms[1], 3), "c2": s(terms[2], 3), "c3": s(terms[3], 3),
        "chi": s(chi, 2), "chi1": s(chi, 1),
        "na": na, "nb": nb, "ax": ca[0], "ay": ca[1], "bx": cb[0], "by": cb[1],
        "dx": cb[0] - ca[0], "dy": cb[1] - ca[1], "d2": d2, "dist": s(sqrt(d2), 1), "w": s(w, 1),
        "ward": s(w * d2, 1), "ward_nw": s(d2, 1), "ward_prod": s(na * nb * d2, 1),
        "ward_d": s(w * sqrt(d2), 1), "ward_half": s((na + nb) / 2 * d2, 1),
    }


def p07():
    gg, gd, gf = 0.90, 0.08, 0.02
    dd, df = 0.75, 0.25
    rd = gd / (1 - dd)          # π_D / π_G
    rf = gf + df * rd           # π_F / π_G
    pg = 1 / (1 + rd + rf)
    cd, cf = 2, 10
    cost = (rd * cd + rf * cf) * pg
    return {"gg": gg, "gd": gd, "gf": gf, "dd": dd, "df": df, "cd": cd, "cf": cf,
            "rd": s(rd, 2), "rf": s(rf, 2), "den": s(1 + rd + rf, 2),
            "pg": s(pg, 4), "pd": s(rd * pg, 4), "pf": s(rf * pg, 4), "dur": s(1 / (1 - dd), 0),
            "num": s(rd * cd + rf * cf, 2), "cost": r(cost, 2), "cost4": s(cost, 4),
            "w_row": s(gd * cd + gf * cf, 2)}


def p08():
    n, p = 30, 3
    sst, sse = 500.0, 150.0
    r2 = 1 - sse / sst
    F = (r2 / p) / ((1 - r2) / (n - p - 1))
    h, rr = 0.40, 2.0
    pp = p + 1
    cook = rr ** 2 / pp * h / (1 - h)
    return {
        "n": n, "p": p, "pp": pp, "sst": s(sst, 0), "sse": s(sse, 0), "ssr": s(sst - sse, 0), "r2": s(r2, 1),
        "dfe": n - p - 1, "msr": s((sst - sse) / p, 4), "mse": s(sse / (n - p - 1), 4),
        "F": s(F, 2), "F_np": s((r2 / p) / ((1 - r2) / (n - p)), 2), "F_nodiv": s(r2 / ((1 - r2) / (n - p - 1)), 2),
        "F_swap": s(((1 - r2) / p) / (r2 / (n - p - 1)), 2), "F_pp": s((r2 / pp) / ((1 - r2) / (n - p - 1)), 2),
        "Fc": 2.98,
        "h": h, "rr": rr, "rr2": s(rr ** 2, 0), "hr": s(h / (1 - h), 4), "hbar": s(pp / n, 3), "h2": s(2 * pp / n, 3),
        "cook": s(cook, 3), "cook_np": s(rr ** 2 * h / (1 - h), 3), "cook_sq": s(rr ** 2 / pp * h / (1 - h) ** 2, 3),
        "cook_p": s(rr ** 2 / p * h / (1 - h), 3), "cook_r": s(rr / pp * h / (1 - h), 3),
    }


def p09():
    b, se = 0.85, 0.30
    ex = {0.262: 1.300, 0.357: 1.429, 0.550: 1.733, 0.850: 2.340, 1.150: 3.158, 1.344: 3.834, 1.438: 4.212}
    lo, hi = r(b - 1.96 * se, 3), r(b + 1.96 * se, 3)
    lo9, hi9 = r(b - 1.645 * se, 3), r(b + 1.645 * se, 3)
    assert lo9 in ex and hi9 in ex, (lo9, hi9)
    lo1, hi1 = r(b - se, 3), r(b + se, 3)
    # (b) マクネマー
    a_, b_, c_, d_ = 80, 25, 10, 35
    n = a_ + b_ + c_ + d_
    mc = (b_ - c_) ** 2 / (b_ + c_)
    mc_cc = (abs(b_ - c_) - 1) ** 2 / (b_ + c_)
    z = (b_ - c_) / sqrt(b_ + c_)
    r1, r2_, k1, k2 = a_ + b_, c_ + d_, a_ + c_, b_ + d_
    pear = n * (a_ * d_ - b_ * c_) ** 2 / (r1 * r2_ * k1 * k2)
    return {
        "b": b, "se": se, "hw": s(1.96 * se, 3), "lo": s(lo, 3), "hi": s(hi, 3),
        "eA": s(ex[0.262], 3), "eB": s(ex[0.357], 3), "eC": s(ex[0.550], 3), "eD": s(ex[0.850], 3),
        "eE": s(ex[1.150], 3), "eF": s(ex[1.344], 3), "eG": s(ex[1.438], 3),
        "ci_ok": f"({s(ex[lo], 2)},\\ {s(ex[hi], 2)})", "ci_90": f"({s(ex[lo9], 2)},\\ {s(ex[hi9], 2)})",
        "ci_1se": f"({s(ex[lo1], 2)},\\ {s(ex[hi1], 2)})", "ci_noexp": f"({s(lo, 2)},\\ {s(hi, 2)})",
        "ci_lin": f"({s(ex[0.850] - 1.96 * se, 2)},\\ {s(ex[0.850] + 1.96 * se, 2)})",
        "or": s(ex[0.850], 2),
        "a": a_, "bb": b_, "c": c_, "d": d_, "n": n, "bc": b_ + c_, "bmc2": (b_ - c_) ** 2,
        "mc": s(mc, 2), "mc_cc": s(mc_cc, 2), "z": s(z, 2), "pear": s(pear, 2),
        "pA": s(r1 / n, 3), "pB": s(k1 / n, 3),
    }


def p10():
    bo = [2.4, -0.6, 1.1]
    lam = 1.6
    th = lam / 2

    def soft(x, t):
        return (1 if x > 0 else -1) * max(abs(x) - t, 0.0)

    def fmt(v):
        out = []
        for x in v:
            x = 0.0 if abs(x) < 1e-12 else x
            out.append("0" if x == 0 else s(x, 2))
        return "(" + ",\\ ".join(out) + ")"
    las = [soft(x, th) for x in bo]
    return {
        "b1": bo[0], "b2": bo[1], "b3": bo[2], "lam": lam, "th": th,
        "ok": fmt(las), "lam_full": fmt([soft(x, lam) for x in bo]),
        "ridge": fmt([x / (1 + lam) for x in bo]), "hard": fmt([x if abs(x) > th else 0 for x in bo]),
        "shift": fmt([x - th for x in bo]), "l1": s(las[0], 1), "l3": s(las[2], 1),
    }


def p11():
    rho, pdim = 0.5, 3
    lam1 = 1 + (pdim - 1) * rho
    lam2 = 1 - rho
    pct = lambda x: s(100 * x, 1) + "\\%"
    # (b) 判別
    m1, m2 = (10, 20), (14, 26)
    S11, S12, S22 = 4, 3, 9
    d = (m2[0] - m1[0], m2[1] - m1[1])
    det = S11 * S22 - S12 ** 2
    q1, q2, q3 = S22 * d[0] ** 2, 2 * S12 * d[0] * d[1], S11 * d[1] ** 2
    D2 = (q1 - q2 + q3) / det
    D = sqrt(D2)
    phi = {1.15: 0.8749, 1.33: 0.9082, 1.41: 0.9207, 2.31: 0.9896, 2.67: 0.9962}
    D2diag = d[0] ** 2 / S11 + d[1] ** 2 / S22
    z_ok, z_diag, z_full, z_q, z_sq = r(D / 2, 2), r(sqrt(D2diag) / 2, 2), r(D, 2), r(D2 / 4, 2), r(D2 / 2, 2)
    for z in (z_ok, z_diag, z_full, z_q, z_sq):
        assert z in phi, z
    return {
        "rho": rho, "pdim": pdim, "lam1": s(lam1, 1), "lam2": s(lam2, 1), "tr": pdim,
        "c_ok": pct(lam1 / pdim), "c_rho": pct(rho), "c_two": pct((1 + rho) / 2), "c_eq": pct(1 / pdim),
        "c_bad": pct((1 + pdim * rho) / pdim),
        "m11": m1[0], "m12": m1[1], "m21": m2[0], "m22": m2[1], "d1": d[0], "d2": d[1],
        "s11": S11, "s12": S12, "s22": S22, "det": det, "q1": q1, "q2": q2, "q3": q3,
        "D2": s(D2, 3), "D": s(D, 3), "Dh": s(D / 2, 3), "D2diag": s(D2diag, 1),
        "zok": s(z_ok, 2),
        "f1": s(phi[1.15], 4), "f2": s(phi[1.33], 4), "f3": s(phi[1.41], 4), "f4": s(phi[2.31], 4), "f5": s(phi[2.67], 4),
        "e_ok": s(1 - phi[z_ok], 3), "e_diag": s(1 - phi[z_diag], 3), "e_full": s(1 - phi[z_full], 3),
        "e_q": s(1 - phi[z_q], 3), "e_sq": s(1 - phi[z_sq], 3),
    }


def p12():
    c, phi, s2 = 8, 0.6, 16
    xT = 26
    mu = c / (1 - phi)
    f2 = mu + phi ** 2 * (xT - mu)
    v2 = s2 * (1 + phi ** 2)
    up = f2 + 1.96 * sqrt(v2)
    return {"c": c, "phi": phi, "s2": s2, "xT": xT, "mu": s(mu, 0), "dev": s(xT - mu, 0),
            "phi2": s(phi ** 2, 2), "f2": s(f2, 2), "v2": s(v2, 2), "sd2": s(sqrt(v2), 4),
            "hw": s(1.96 * sqrt(v2), 4), "up": r(up, 2), "up_s": s(up, 2), "up4": s(up, 4),
            "up_1": s(f2 + 1.96 * sqrt(s2), 2), "up_stat": s(f2 + 1.96 * sqrt(s2 / (1 - phi ** 2)), 2),
            "up_phi": s(f2 + 1.96 * sqrt(s2 * (1 + phi)), 2)}


def p13():
    Nh = [6000, 3000, 1000]
    Sh = [2, 4, 10]
    n = 200
    NS = [a * b for a, b in zip(Nh, Sh)]
    ney = n * NS[2] / sum(NS)
    NS2 = [a * b * b for a, b in zip(Nh, Sh)]
    # (b) AIC
    lA, kA = -410.0, 3
    lB, kB = -409.2, 4
    lC, kC = -407.4, 6
    aic = lambda l, k: -2 * l + 2 * k
    nobs = 500
    return {
        "N1": Nh[0], "N2": Nh[1], "N3": Nh[2], "S1": Sh[0], "S2": Sh[1], "S3": Sh[2], "nn": n,
        "NS1": NS[0], "NS2": NS[1], "NS3": NS[2], "NSsum": sum(NS), "ney": s(ney, 1),
        "a_ok": s(ney, 0), "a_prop": s(n * Nh[2] / sum(Nh), 0), "a_s": s(n * Sh[2] / sum(Sh), 0),
        "a_s2": s(n * NS2[2] / sum(NS2), 0), "a_eq": s(n / 3, 0),
        "lA": s(lA, 1), "lB": s(lB, 1), "lC": s(lC, 1), "kA": kA, "kB": kB, "kC": kC, "nobs": nobs,
        "aA": s(aic(lA, kA), 1), "aB": s(aic(lB, kB), 1), "aC": s(aic(lC, kC), 1),
        "aB3": s(aic(lB, kB - 1), 1), "aCk": s(-2 * lC + kC, 1),
        "bicA": s(-2 * lA + kA * math.log(nobs), 1), "m2A": s(-2 * lA, 1), "m2B": s(-2 * lB, 1), "m2C": s(-2 * lC, 1),
    }


def values():
    return {"s01p01": p01(), "s01p02": p02(), "s01p03": p03(), "s01p04": p04(), "s01p05": p05(),
            "s01p06": p06(), "s01p07": p07(), "s01p08": p08(), "s01p09": p09(), "s01p10": p10(),
            "s01p11": p11(), "s01p12": p12(), "s01p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
