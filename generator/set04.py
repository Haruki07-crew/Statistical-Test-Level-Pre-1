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
    # (a) 負の二項分布：モーメント法で r と p を決め、欠陥 0 個の確率（歩留まり）
    m, v = 0.6, 0.72
    p = m / v
    rr = m * p / (1 - p)
    rr_i = int(round(rr))
    assert abs(rr - rr_i) < 1e-9
    y0 = p ** rr_i
    epois = 0.5488  # e^{-0.6}（問題文に示す）
    # (b) 3×2 分割表の調整済み標準化残差（夜勤×不良）
    n_sh = 400
    d = [18, 22, 40]
    D = sum(d)
    N = n_sh * 3
    E = n_sh * D / N
    adj_den2 = E * (1 - n_sh / N) * (1 - D / N)
    adj = (d[2] - E) / sqrt(adj_den2)
    pear = (d[2] - E) / sqrt(E)
    return {
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
    c, a = 8, None  # a は記号のまま
    U = 0.70
    ln03, ln07 = -1.2040, -0.3567
    v_ok = c * sqrt(-ln03)
    return {
        "c": c, "c2": c * c, "half_c2": c * c // 2, "k3": 3 * c * c // 2, "U": s(U, 2), "U1": s(1 - U, 2),
        "ln03": s(ln03, 4), "ln07": s(ln07, 4), "sq03": s(sqrt(-ln03), 4),
        "v_ok": s(v_ok, 2), "v_u": s(c * sqrt(-ln07), 2), "v_nosq": s(c * (-ln03), 2),
        "v_pow": s(c * (-ln03) ** 2, 2), "v_in": s(sqrt(c * (-ln03)), 2),
    }


def p03():
    # (a) 直列・並列の信頼度とベイズの定理
    pt = 0.96          # 変圧器が正常な確率
    pf = 0.85          # 各フィーダーが正常な確率
    qf2 = (1 - pf) ** 2
    rel = pt * (1 - qf2)
    fail = 1 - rel
    post = (1 - pt) / fail
    # (b) 線形ハザード
    h0, h1 = 0.02, 0.004
    t1, t2 = 10, 15
    H = lambda t: h0 * t + h1 * t * t / 2
    dH = H(t2) - H(t1)
    ex = {0.30: 0.7408, 0.35: 0.7047, 0.40: 0.6703, 0.75: 0.4724}
    assert r(dH, 2) in ex and r(H(t2), 2) in ex
    hz10 = h0 + h1 * t1
    hz15 = h0 + h1 * t2
    assert r(hz10 * (t2 - t1), 2) in ex and r(hz15 * (t2 - t1), 2) in ex
    # (c) 故障ゼロの実証試験
    p0, ln005, ln098, ln0025 = 0.02, -2.9957, -0.0202, -3.6889
    nraw = ln005 / ln098
    return {
        "pt": pt, "pf": pf, "qt": s(1 - pt, 2), "qf": s(1 - pf, 2), "qf2": s(qf2, 4), "par": s(1 - qf2, 4),
        "rel": s(rel, 4), "fail": s(fail, 4), "post": r(post, 3), "post4": s(post, 4),
        "post_feed": s(pt * qf2 / fail, 4), "both": s((1 - pt) * qf2 / fail, 4),
        "h0": h0, "h1": h1, "h1h": s(h1 / 2, 3), "t1": t1, "t2": t2, "Ht1": s(H(t1), 2), "Ht2": s(H(t2), 2),
        "dH": s(dH, 2), "S": r(ex[r(dH, 2)], 3), "S4": s(ex[r(dH, 2)], 4),
        "e30": s(ex[0.30], 4), "e35": s(ex[0.35], 4), "e40": s(ex[0.40], 4), "e75": s(ex[0.75], 4),
        "S_unc": s(ex[r(H(t2), 2)], 3), "hz10": s(hz10, 2), "hz15": s(hz15, 2),
        "S_h10": s(ex[r(hz10 * 5, 2)], 3), "S_h15": s(ex[r(hz15 * 5, 2)], 3),
        "p0": p0, "ln005": s(ln005, 4), "ln098": s(ln098, 4), "ln0025": s(ln0025, 4),
        "nraw": s(nraw, 2), "nreq": math.ceil(nraw), "n3": s(3 / p0, 0), "n_two": math.ceil(ln0025 / ln098),
        "n_low": math.floor(nraw),
    }


def p04():
    # (a) 2変量正規の条件付き分布
    mx, my, sx, sy, rho = 100, 96, 4, 5, 0.8
    x0, up = 104, 102
    cm = my + rho * sy / sx * (x0 - mx)
    cs = sy * sqrt(1 - rho ** 2)
    phi = {0.40: 0.6554, 0.67: 0.7486, 0.89: 0.8133, 1.15: 0.8749, 1.20: 0.8849}
    z_ok = r((up - cm) / cs, 2)
    z_unc = r((up - my) / sy, 2)
    z_sd = r((up - cm) / sy, 2)
    z_1r = r((up - cm) / (sy * sqrt(1 - rho)), 2)
    cm_w = my + rho * sx / sy * (x0 - mx)
    z_sl = r((up - cm_w) / cs, 2)
    for z in (z_ok, z_unc, z_sd, z_1r, z_sl):
        assert z in phi, z
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
        "mx": mx, "my": my, "sx": sx, "sy": sy, "rho": rho, "x0": x0, "up": up,
        "slope": s(rho * sy / sx, 1), "dx": x0 - mx, "cm": s(cm, 1), "rho2": s(rho ** 2, 2), "cs": s(cs, 1),
        "zraw": s((up - cm) / cs, 4), "zok": s(z_ok, 2),
        "f40": s(phi[0.40], 4), "f67": s(phi[0.67], 4), "f89": s(phi[0.89], 4), "f115": s(phi[1.15], 4), "f120": s(phi[1.20], 4),
        "c_ok": s(1 - phi[z_ok], 3), "c_unc": s(1 - phi[z_unc], 3), "c_sd": s(1 - phi[z_sd], 3),
        "c_1r": s(1 - phi[z_1r], 3), "c_sl": s(1 - phi[z_sl], 3), "cm_w": s(cm_w, 2), "cs_1r": s(sy * sqrt(1 - rho), 3),
        "nA": nA, "mA": mA, "sA": sA, "nB": nB, "mB": mB, "sB": sB, "vA": s(sA ** 2, 2), "vB": s(sB ** 2, 2),
        "a": s(a, 4), "b": s(b, 4), "ab": s(a + b, 4), "se": s(se, 4), "diff": s(mA - mB, 1),
        "t": s(t, 2), "nu": s(nu, 1), "ab2": s((a + b) ** 2, 4), "a2": s(a ** 2 / (nA - 1), 4), "b2": s(b ** 2 / (nB - 1), 4),
        "dA": nA - 1, "dB": nB - 1, "dfp": nA + nB - 2, "sp2": s(sp2, 2), "tp": s(tp, 2),
        "ch_ok": pair(s(t, 2), s(nu, 1)), "ch_pool": pair(s(tp, 2), nA + nB - 2), "ch_wdf": pair(s(t, 2), nA + nB - 2),
        "ch_min": pair(s(t, 2), min(nA, nB) - 1), "ch_n": pair(s(t, 2), s(nu_n, 1)),
    }


def p05():
    from fractions import Fraction as Fr
    n = 8
    mse1 = Fr(1, 3 * n)                   # θ² の係数：MSE(2X̄)
    mse2 = Fr(1, n * (n + 2))             # V((n+1)/n X(n))
    mse_mle = Fr(2, (n + 1) * (n + 2))    # MSE(X(n))
    var_max = Fr(n, (n + 1) ** 2 * (n + 2))
    ratio = mse1 / mse2
    # (b) チェビシェフ
    th, eps, al = 10, 1, 0.05
    n_ch = th ** 2 / (3 * eps ** 2 * al)
    return {
        "n": n, "n1": n + 1, "n2": n + 2, "n3": 3 * n, "nn2": n * (n + 2),
        "ratio": s(ratio, 2), "r_mle": s(mse1 / mse_mle, 2), "r_var": s(Fr(1, 12 * n) / mse2, 2),
        "r_max": s(mse1 / var_max, 2), "r_inv": s(1 / ratio, 2), "mle_c": f"\\frac{{2}}{{{(n + 1) * (n + 2)}}}",
        "th": th, "eps": eps, "al": al, "th2": th * th, "nch_raw": s(n_ch, 2), "nch": math.ceil(n_ch),
        "n_clt": math.ceil(1.96 ** 2 * th ** 2 / 3 / eps ** 2), "n_var1": math.ceil(th ** 2 / (12 * eps ** 2 * al)),
        "n_th2": math.ceil(th ** 2 / (eps ** 2 * al)), "n_95": math.ceil(th ** 2 / (3 * eps ** 2 * 0.95)),
        "clt_raw": s(1.96 ** 2 * th ** 2 / 3, 2),
    }


def p06():
    yA, tA, yB, tB = 18, 1200, 9, 1500
    lam = (yA + yB) / (tA + tB)
    rA, rB = yA / tA, yB / tB
    se = sqrt(lam * (1 / tA + 1 / tB))
    z = (rA - rB) / se
    z_un = (rA - rB) / sqrt(rA / tA + rB / tB)
    z_cnt = (yA - yB) / sqrt(yA + yB)
    z_A = (rA - rB) / sqrt(rA * (1 / tA + 1 / tB))
    pa = tA / (tA + tB)
    eA = (yA + yB) * pa
    z_cc = (abs(yA - eA) - 0.5) / sqrt((yA + yB) * pa * (1 - pa))
    # (b) ポアソン回帰（オフセット）の率比の信頼区間
    lnRR = {2.5: 0.9163, 2.0: 0.6931}
    b1 = lnRR[2.5]
    seb = sqrt(1 / yA + 1 / yB)
    ex = {}

    def ci(center, half):
        return r(center - half, 3), r(center + half, 3)
    c_ok = ci(b1, 1.96 * r(seb, 4))
    c_no = ci(lnRR[2.0], 1.96 * r(seb, 4))
    c_ns = ci(b1, 1.96 * (1 / yA + 1 / yB))
    c_90 = ci(b1, 1.645 * r(seb, 4))
    table = {-0.107: 0.8985, 0.116: 1.1230, 0.245: 1.2776, 0.590: 1.8040, 1.243: 3.4661,
             1.493: 4.4502, 1.588: 4.8937, 1.716: 5.5622}
    for cc in (c_ok, c_no, c_ns, c_90):
        for x in cc:
            assert x in table, x
    fm = lambda cc: f"$({s(table[cc[0]], 2)},\\ {s(table[cc[1]], 2)})$"
    lin = (2.5 - 1.96 * r(seb, 4), 2.5 + 1.96 * r(seb, 4))
    tab_keys = sorted(table)
    return {
        "yA": yA, "tA": tA, "yB": yB, "tB": tB, "y": yA + yB, "t": tA + tB, "lam": s(lam, 2),
        "rA": s(rA, 3), "rB": s(rB, 3), "dr": s(rA - rB, 3), "inv": s(1 / tA + 1 / tB, 6), "var": s(lam * (1 / tA + 1 / tB), 7),
        "se": s(se, 6), "z": s(z, 2), "z4": s(z, 4), "z_un": s(z_un, 2), "z_cnt": s(z_cnt, 2), "z_A": s(z_A, 2), "z_cc": s(z_cc, 2),
        "pa": s(pa, 4), "eA": s(eA, 0), "bvar": s((yA + yB) * pa * (1 - pa), 4),
        "rr": s(rA / rB, 1), "b1": s(b1, 4), "ln2": s(lnRR[2.0], 4), "ln25": s(b1, 4),
        "seb2": s(1 / yA + 1 / yB, 4), "seb": s(seb, 4), "half": s(1.96 * r(seb, 4), 4),
        "lo": s(c_ok[0], 3), "hi": s(c_ok[1], 3),
        "tab_x": " | ".join(s(k, 3) for k in tab_keys), "tab_e": " | ".join(s(table[k], 4) for k in tab_keys),
        "tab_sep": "|".join(["---"] * len(tab_keys)),
        "ci_ok": fm(c_ok), "ci_no": fm(c_no), "ci_ns": fm(c_ns), "ci_90": fm(c_90),
        "ci_lin": f"$({s(lin[0], 2)},\\ {s(lin[1], 2)})$",
        "elo": s(table[c_ok[0]], 2), "ehi": s(table[c_ok[1]], 2),
    }


def p07():
    lam, ms = 12, 4  # 1時間あたり到着、平均処理時間（分）
    mu = 60 / ms
    rho = lam / mu
    L = rho / (1 - rho)
    W = L / lam * 60
    Wq = W - ms
    return {"lam": lam, "ms": ms, "mu": s(mu, 0), "rho": s(rho, 1), "L": s(L, 1), "W": s(W, 1),
            "Wq": r(Wq, 1), "Wq_s": s(Wq, 1), "Lq": s(rho * L, 1), "Wh": s(W / 60, 4)}


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
    # (c) ユール・ウォーカー
    r1, r2 = 0.6, 0.5
    phi1 = r1 * (1 - r2) / (1 - r1 ** 2)
    phi2 = (r2 - r1 ** 2) / (1 - r1 ** 2)
    pr = lambda a, b: f"$\\hat\\phi_1={s(a, 3)},\\ \\hat\\phi_2={s(b, 3)}$"
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
        "r1": r1, "r2": r2, "r12": s(r1 ** 2, 2), "den": s(1 - r1 ** 2, 2), "num1": s(r1 * (1 - r2), 2), "num2": s(r2 - r1 ** 2, 2),
        "phi1": s(phi1, 3), "phi2": s(phi2, 3), "phi1_5": s(phi1, 5), "phi2_5": s(phi2, 5),
        "c_ok": pr(phi1, phi2), "c_raw": pr(r1, r2), "c_ar1": pr(r1, r2 - r1 ** 2), "c_swap": pr(phi2, phi1),
        "c_den": pr(r1 * (1 - r2) / (1 - r1), (r2 - r1 ** 2) / (1 - r1)),
    }


def p10():
    # (a) HT 推定量
    y = [120, 45, 300, 80]
    pi = [0.20, 0.08, 0.50, 0.16]
    Nn = 50
    ht = sum(a / b for a, b in zip(y, pi))
    srs = Nn * sum(y) / len(y)
    hh = ht / len(y)
    prod = sum(a * b for a, b in zip(y, pi))
    rat = sum(y) / sum(pi)
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
    return {
        "Nn": Nn, "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3], "pi1": s(pi[0], 2), "pi2": s(pi[1], 2), "pi3": s(pi[2], 2), "pi4": s(pi[3], 2),
        "q1": s(y[0] / pi[0], 1), "q2": s(y[1] / pi[1], 1), "q3": s(y[2] / pi[2], 1), "q4": s(y[3] / pi[3], 1),
        "ht": s(ht, 1), "srs": s(srs, 1), "hh": s(hh, 1), "prod": s(prod, 1), "rat": s(rat, 1), "ybar": s(sum(y) / 4, 2), "ysum": sum(y),
        "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "x5": x[4], "yy1": yy[0], "yy2": yy[1], "yy3": yy[2], "yy4": yy[3], "yy5": yy[4],
        "Sx": Sx, "Sy": Sy, "wls": s(wls, 3), "ols0": s(ols0, 3), "mr": s(mr, 3), "ols1": s(ols1, 3), "wx": s(wx, 3),
        "Sxy": sum(a * b for a, b in zip(x, yy)), "Sxx": sum(a * a for a in x),
    }


def p11():
    a, rr = 6, 4
    msa, mse = 15.2, 2.6
    sa2 = (msa - mse) / rr
    icc = sa2 / (sa2 + mse)
    return {"a": a, "rr": rr, "N": a * rr, "dfa": a - 1, "dfe": a * (rr - 1), "msa": msa, "mse": mse,
            "dms": s(msa - mse, 1), "sa2": s(sa2, 3), "tot": s(sa2 + mse, 3), "icc": r(icc, 3), "icc4": s(icc, 4),
            "w_nodiv": s((msa - mse) / (msa - mse + mse), 3), "w_msa": s(msa / (msa + mse), 3), "w_a": s(((msa - mse) / a) / ((msa - mse) / a + mse), 3)}


def p12():
    # (a) 共分散行列の主成分
    v1, v2, c12 = 10, 4, 4
    tr = v1 + v2
    det = v1 * v2 - c12 ** 2
    disc = sqrt((tr / 2) ** 2 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    rcor = c12 / sqrt(v1 * v2)
    pct = lambda x: s(100 * x, 1) + "\\%"
    # (b) 正準相関（1 変数対 2 変数）＝重相関係数
    ry1, ry2, r12 = 0.6, 0.5, 0.4
    R2 = (ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12) / (1 - r12 ** 2)
    # (c) コストつきベイズ判別
    mG, mD, sd = 50, 58, 4
    piD, cm, cf = 0.05, 20, 1
    ln095, ln19, ln20 = -0.0513, 2.9444, 2.9957
    mid = (mG + mD) / 2
    k = sd ** 2 / (mD - mG)
    thr = mid + k * ln095            # ln{(π_G c_f)/(π_D c_m)} = ln 0.95
    thr_prior = mid + k * ln19
    thr_cost = mid - k * ln20
    thr_rev = mid - k * ln095
    return {
        "v1": v1, "v2": v2, "c12": c12, "tr": tr, "det": det, "half": s(tr / 2, 0), "disc2": s((tr / 2) ** 2 - det, 0), "disc": s(disc, 0),
        "l1": s(l1, 0), "l2": s(l2, 0), "rcor": s(rcor, 4),
        "c_ok": pct(l1 / tr), "c_cor": pct((1 + rcor) / 2), "c_var": pct(v1 / tr), "c_r": pct(rcor), "c_tot": pct(l1 / (tr + 2 * c12)),
        "ry1": ry1, "ry2": ry2, "r12": r12, "num": s(ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12, 2), "den": s(1 - r12 ** 2, 2),
        "R2": s(R2, 4), "R": s(sqrt(R2), 3), "w_max": s(ry1, 3), "w_ind": s(sqrt(ry1 ** 2 + ry2 ** 2), 3), "w_R2": s(R2, 3),
        "w_noden": s(sqrt(ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12), 3),
        "mG": mG, "mD": mD, "sd": sd, "sd2": sd * sd, "piD": piD, "piG": s(1 - piD, 2), "cm": cm, "cf": cf,
        "ln095": s(ln095, 4), "ln19": s(ln19, 4), "ln20": s(ln20, 4), "mid": s(mid, 0), "k": s(k, 0),
        "thr": s(thr, 2), "thr4": s(thr, 4), "thr_mid": s(mid, 2), "thr_prior": s(thr_prior, 2), "thr_cost": s(thr_cost, 2), "thr_rev": s(thr_rev, 2),
        "lr": s((1 - piD) * cf / (piD * cm), 2), "dm": mD - mG, "sum_m2": mD ** 2 - mG ** 2, "twos2": 2 * sd * sd,
    }


def p13():
    px = 0.4
    pm = {1: 0.8, 0: 0.3}
    py = {(1, 1): 0.05, (0, 1): 0.20, (1, 0): 0.10, (0, 0): 0.25}  # (m, x)
    pxv = {1: px, 0: 1 - px}
    inner = {m: sum(py[(m, xx)] * pxv[xx] for xx in (0, 1)) for m in (0, 1)}
    do = {x: pm[x] * inner[1] + (1 - pm[x]) * inner[0] for x in (0, 1)}
    ace = do[1] - do[0]
    naive = {x: pm[x] * py[(1, x)] + (1 - pm[x]) * py[(0, x)] for x in (0, 1)}
    pM1 = sum(pxv[x] * pm[x] for x in (0, 1))
    pxm = {(x, 1): pxv[x] * pm[x] / pM1 for x in (0, 1)}
    pxm.update({(x, 0): pxv[x] * (1 - pm[x]) / (1 - pM1) for x in (0, 1)})
    pyM = {m: sum(py[(m, x)] * pxm[(x, m)] for x in (0, 1)) for m in (0, 1)}
    w_cond = (pm[1] - pm[0]) * (pyM[1] - pyM[0])
    w_med = (py[(1, 1)] - py[(1, 0)]) * pM1 + (py[(0, 1)] - py[(0, 0)]) * (1 - pM1)
    w_step = inner[1] - inner[0]
    vals = [ace, naive[1] - naive[0], w_cond, w_step, w_med]
    assert len({s(v, 3) for v in vals}) == 5, [s(v, 3) for v in vals]
    return {
        "px": px, "px0": s(1 - px, 1), "pm1": pm[1], "pm0": pm[0], "y11": py[(1, 1)], "y01": s(py[(0, 1)], 2), "y10": s(py[(1, 0)], 2), "y00": py[(0, 0)],
        "in1": s(inner[1], 2), "in0": s(inner[0], 2), "do1": s(do[1], 3), "do0": s(do[0], 3),
        "ace": s(ace, 3), "naive": s(naive[1] - naive[0], 3), "n1": s(naive[1], 3), "n0": s(naive[0], 3),
        "w_cond": s(w_cond, 3), "w_step": s(w_step, 3), "w_med": s(w_med, 3),
        "qm1": s(1 - pm[1], 1), "qm0": s(1 - pm[0], 1),
    }


def values():
    return {"s04p01": p01(), "s04p02": p02(), "s04p03": p03(), "s04p04": p04(), "s04p05": p05(),
            "s04p06": p06(), "s04p07": p07(), "s04p08": p08(), "s04p09": p09(), "s04p10": p10(),
            "s04p11": p11(), "s04p12": p12(), "s04p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
