"""模擬試験 第21回（set21）の数値。values() が {大問id: {名前: 値}} を返す。

題材：学校・入試・学習アプリ・図書館。
問題文に示した分布の値（Φ・t の点、e^x・log など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from math import exp, factorial, sqrt


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
    # 5 人の答案を無作為に返却して誰も自分の答案を受け取らない確率（包除原理）
    n = 5
    ans = sum((-1) ** k / factorial(k) for k in range(n + 1))
    return {
        "n": n, "fact": factorial(n), "der": round(ans * factorial(n)),
        "ans": r(ans, 4), "ans_s": s(ans, 4), "ans6": s(ans, 6),
        "e1": s(exp(-1), 4), "ind": s((1 - 1 / n) ** n, 4), "one": s(1 - ans, 4),
    }


def p02():
    n, T = 40, 28
    ph = T / n
    pq = ph * (1 - ph)
    unb = T * (n - T) / (n * (n - 1))
    return {
        "n": n, "T": T, "nT": n - T, "ph": s(ph, 1), "qh": s(1 - ph, 1), "pq": s(pq, 2), "n1": n - 1,
        "Tnum": T * (n - T), "uden": n * (n - 1),
        "unb": s(unb, 4), "unb5": s(unb, 5), "mle": s(pq, 4), "shr": s(pq * (n - 1) / n, 4),
        "vph": s(pq / n, 4), "both": s(T * (n - T) / (n - 1) ** 2, 4),
    }


def p03():
    m1, m2, m3, t = 20, 30, 60, 15
    rate = 1 / m1 + 1 / m2 + 1 / m3  # 0.1
    e = {0.25: 0.7788, 0.5: 0.6065, 0.75: 0.4724, 1.5: 0.2231}
    ans = e[1.5]
    d_max = 1 - (1 - e[0.75]) * (1 - e[0.5]) * (1 - e[0.25])
    # (b) 密度 c(L-x) (0<=x<=L) の中央値
    L = 4
    med = L - L / sqrt(2)  # x^2 - 2Lx + L^2/2 = 0 の小さい解
    return {
        "m1": m1, "m2": m2, "m3": m3, "t": t, "rate": s(rate, 1), "mean": s(1 / rate, 0),
        "e025": s(e[0.25], 4), "e05": s(e[0.5], 4), "e075": s(e[0.75], 4), "e15": s(e[1.5], 4),
        "ans": s(ans, 3), "d_short": s(e[0.75], 3), "d_mid": s(e[0.5], 3), "d_cmp": s(1 - ans, 3),
        "d_max": s(d_max, 3), "qprod": s((1 - e[0.75]) * (1 - e[0.5]) * (1 - e[0.25]), 4),
        "L": L, "cinv": L ** 2 // 2, "twoL": 2 * L, "L2h": L ** 2 // 2,
        "med": r(med, 2), "med_s": s(med, 2), "med4": s(med, 4), "big": s(L + L / sqrt(2), 2),
        "xmean": s(L / 3, 2), "d_mid2": s(L / 2, 2), "sq": s(L / sqrt(2), 4),
    }


def p04():
    # (a) ポアソン分布どうしのカルバック・ライブラー情報量
    l1, l2, ln12 = 6, 5, 0.1823
    kl = l1 * ln12 - l1 + l2
    kl_rev = l2 * (-ln12) - l2 + l1
    # (b) 2 つの分館の延滞率の差（プールした比率）
    n1, x1, n2, x2 = 250, 50, 250, 25
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se0 = sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    se_u = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "l1": l1, "l2": l2, "ln12": ln12, "t1": s(l1 * ln12, 4), "kl": r(kl, 3), "kl_s": s(kl, 3), "kl4": s(kl, 4),
        "kl_rev": s(kl_rev, 3), "t2": s(l2 * ln12, 4),
        "n1": n1, "x1": x1, "n2": n2, "x2": x2, "p1": s(p1, 2), "p2": s(p2, 2), "pp": s(pp, 2), "qq": s(1 - pp, 2),
        "dp": s(p1 - p2, 2), "ppq": s(pp * (1 - pp), 4), "inv": s(1 / n1 + 1 / n2, 3), "var0": s(se0 ** 2, 5),
        "se0": s(se0, 6), "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4), "z_u": s((p1 - p2) / se_u, 2),
        "se_u": s(se_u, 5),
        "N": 12000, "ns": 300, "k": 12000 // 300,
    }


def p05():
    # (a) 等分散を仮定した 2 標本 t 区間
    n1, m1, s1 = 12, 68.5, 8.0
    n2, m2, s2 = 10, 62.0, 7.0
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    sp = sqrt(sp2)
    f = sqrt(1 / n1 + 1 / n2)
    se = sp * f
    t20, t22, z = 2.086, 2.074, 1.96
    d = m1 - m2
    se_u = sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    se_w = sp * (1 / n1 + 1 / n2)
    iv = lambda h: (s(d - h, 2), s(d + h, 2))
    ok, zz, d22, uu, ww = iv(t20 * se), iv(z * se), iv(t22 * se), iv(t20 * se_u), iv(t20 * se_w)
    # (b) オッズ比のウルフの信頼区間
    a, b, c, dd = 18, 12, 16, 24
    OR = a * dd / (b * c)
    lnOR = 0.8109
    v = 1 / a + 1 / b + 1 / c + 1 / dd
    seL = r(sqrt(v), 4)
    h = 1.96 * seL
    lo, hi = r(lnOR - h, 4), r(lnOR + h, 4)
    vv = r(v, 4)
    hv = 1.96 * vv
    lo2, hi2 = r(lnOR - hv, 4), r(lnOR + hv, 4)
    E = {x: r(exp(x), 4) for x in (lo, hi, lo2, hi2)}
    sym = 1.96 * OR * seL
    nolog = 1.96 * seL
    return {
        "n1": n1, "m1": m1, "s1": s1, "n2": n2, "m2": m2, "s2": s2, "df": n1 + n2 - 2, "t20": t20, "t22": t22, "z": z,
        "n11": n1 - 1, "n21": n2 - 1, "v1": s(s1 ** 2, 0), "v2": s(s2 ** 2, 0),
        "ssum": s((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2, 0),
        "sp2": s(sp2, 2), "sp": s(sp, 4), "f": s(f, 4), "se": s(se, 4), "h": s(t20 * se, 3), "d": s(d, 1),
        "ok_l": ok[0], "ok_u": ok[1], "z_l": zz[0], "z_u": zz[1], "d22_l": d22[0], "d22_u": d22[1],
        "u_l": uu[0], "u_u": uu[1], "w_l": ww[0], "w_u": ww[1],
        "a": a, "b": b, "c": c, "dd": dd, "OR": s(OR, 2), "lnOR": lnOR, "v": s(v, 4), "seL": s(seL, 4), "hL": s(h, 4),
        "lo": s(lo, 4), "hi": s(hi, 4), "elo": s(E[lo], 4), "ehi": s(E[hi], 4),
        "lo2": s(lo2, 4), "hi2": s(hi2, 4), "elo2": s(E[lo2], 4), "ehi2": s(E[hi2], 4), "hv": s(hv, 4),
        "ci_l": s(E[lo], 2), "ci_u": s(E[hi], 2), "lg_l": s(lo, 2), "lg_u": s(hi, 2),
        "sq_l": s(E[lo2], 2), "sq_u": s(E[hi2], 2), "sy_l": s(OR - sym, 2), "sy_u": s(OR + sym, 2),
        "nl_l": s(OR - nolog, 2), "nl_u": s(OR + nolog, 2),
    }


def p06():
    b0, b1, b2, b3, x = 1.20, 0.020, 0.90, -0.008, 70
    diff = b2 + b3 * x
    slope = b1 + b3
    return {
        "b0": s(b0, 2), "b1": s(b1, 3), "b2": s(b2, 2), "b3": s(b3, 3), "x": x, "b3x": s(b3 * x, 2),
        "diff": s(diff, 2), "slope": s(slope, 3), "d_sign": s(b2 - b3 * x, 2), "s_sign": s(b1 - b3, 3),
        "g_gen": s(b0 + b1 * x, 2), "g_rec": s(b0 + b2 + slope * x, 2), "b02": s(b0 + b2, 2),
    }


def p07():
    mu0, sig, n, xbar = 60, 15, 36, 64.5
    se = sig / sqrt(n)
    z = (xbar - mu0) / se
    phi = {0.3: 0.6179, 0.72: 0.7642, 1.8: 0.9641}
    z_sd = (xbar - mu0) / sig
    z_var = (xbar - mu0) / (sig ** 2 / n)
    assert round(z, 2) == 1.8 and round(z_sd, 2) == 0.3 and round(z_var, 2) == 0.72
    p = 1 - phi[1.8]
    rho = 0.6
    return {
        "mu0": mu0, "sig": sig, "n": n, "xbar": xbar, "se": s(se, 1), "dx": s(xbar - mu0, 1), "z": s(z, 1),
        "phi03": s(phi[0.3], 4), "phi072": s(phi[0.72], 4), "phi18": s(phi[1.8], 4),
        "p": s(p, 3), "p4": s(p, 4), "d_two": s(2 * p, 3), "d_sd": s(1 - phi[0.3], 3), "d_var": s(1 - phi[0.72], 3),
        "d_cmp": s(phi[1.8], 3), "zsd": s(z_sd, 1), "zvar": s(z_var, 2), "varn": s(sig ** 2 / n, 2),
        "rho": rho, "l1": s(1 + rho, 1), "l2": s(1 - rho, 1), "c1": s(100 * (1 + rho) / 2, 0), "c_r": s(100 * rho, 0),
        "c_sq": s(100 * (1 + rho) ** 2 / 4, 0), "l1sq": s(1 + rho ** 2, 2), "l2sq": s(1 - rho ** 2, 2),
    }


def p08():
    xs = [0, 1, 2, 4]
    ps = [0.4, 0.3, 0.2, 0.1]
    mu = sum(x * p for x, p in zip(xs, ps))
    m2 = sum((x - mu) ** 2 * p for x, p in zip(xs, ps))
    m3 = sum((x - mu) ** 3 * p for x, p in zip(xs, ps))
    sd = sqrt(m2)
    sk = m3 / sd ** 3
    raw3 = sum(x ** 3 * p for x, p in zip(xs, ps))
    out = {
        "mu": s(mu, 1), "m2": s(m2, 2), "m3": s(m3, 3), "sd": s(sd, 5), "sd3": s(sd ** 3, 5), "sk": s(sk, 2), "sk4": s(sk, 4),
        "d_m3": s(m3, 2), "d_m2": s(m3 / m2, 2), "d_raw": s(raw3 / sd ** 3, 2), "d_4": s(m3 / m2 ** 2, 2), "raw3": s(raw3, 1),
    }
    for i, (x, p) in enumerate(zip(xs, ps)):
        dv = x - mu
        out[f"x{i}"] = x
        out[f"p{i}"] = p
        out[f"dv{i}"] = s(dv, 1)
        out[f"sq{i}"] = s(dv ** 2 * p, 3)
        out[f"c{i}"] = s(dv ** 3 * p, 4)
    # (b) 幾何分布の無記憶性
    p, k, past = 0.3, 2, 3
    within = 1 - (1 - p) ** k
    out.update({
        "p": p, "q": s(1 - p, 1), "k": k, "past": past, "within": s(within, 2), "q2": s((1 - p) ** k, 2),
        "ex": s(1 / p, 2), "d_tot": s(past + 1 / p, 2), "d_fail": s((1 - p) / p, 2),
        "d_all": s(1 - (1 - p) ** (past + k), 2), "d_exact": s((1 - p) * p, 2),
    })
    return out


def p09():
    # 初級 1・中級 2・上級 3 の出生死滅型の連鎖
    a12, a21, a23, a32 = 0.3, 0.2, 0.3, 0.4
    r2 = a12 / a21          # π2/π1
    r3 = r2 * a23 / a32     # π3/π1
    tot = 1 + r2 + r3
    pi3 = r3 / tot
    pi1 = 1 / tot
    return {
        "p11": s(1 - a12, 1), "p12": a12, "p21": a21, "p22": s(1 - a21 - a23, 1), "p23": a23, "p32": a32, "p33": s(1 - a32, 1),
        "r2": s(r2, 1), "r3": s(r3, 3), "tot": s(tot, 3), "pi1": s(pi1, 4), "pi2": s(r2 / tot, 4), "pi3": s(pi3, 4),
        "ret": r(1 / pi3, 2), "ret_s": s(1 / pi3, 2), "ret4": s(1 / pi3, 4), "d_pi1": s(1 / pi1, 2), "d_stay": s(1 / a32, 2),
        "ret_frac": f"{tot:.3f}/{r3:.3f}",
    }


def p10():
    n, c, p0, p1 = 5, 4, 0.6, 0.9
    pw = {(0.6, 4): 0.1296, (0.6, 5): 0.07776, (0.9, 4): 0.6561, (0.9, 5): 0.59049}
    P4 = lambda p: 5 * pw[(p, 4)] * (1 - p)
    P5 = lambda p: pw[(p, 5)]
    alpha = P4(p0) + P5(p0)
    power = P4(p1) + P5(p1)
    beta = 1 - power
    return {
        "n": n, "c": c, "p0": p0, "p1": p1, "q0": s(1 - p0, 1), "q1": s(1 - p1, 1),
        "w64": pw[(0.6, 4)], "w65": pw[(0.6, 5)], "w94": pw[(0.9, 4)], "w95": pw[(0.9, 5)],
        "a4": s(P4(p0), 4), "a5": s(P5(p0), 5), "alpha": s(alpha, 3), "alpha5": s(alpha, 5),
        "b4": s(P4(p1), 5), "b5": s(P5(p1), 5), "power": s(power, 5), "beta": s(beta, 3), "beta5": s(beta, 5),
        "d_ax": s(P5(p0), 3), "d_bx": s(1 - P5(p1), 3), "d_pow": s(power, 3), "d_1a": s(1 - alpha, 3),
    }


def p11():
    SA, SB, SAB, SE = 180, 420, 60, 540
    ST = SA + SB + SAB + SE
    # (b) 経験ベイズ
    J, nn, sw, vm, mbar, yx = 8, 25, 10, 13, 60, 70
    se2 = sw ** 2 / nn
    tau2 = vm - se2
    B = se2 / (se2 + tau2)
    eb = mbar + (1 - B) * (yx - mbar)
    B_wrong = se2 / (se2 + vm)
    return {
        "SA": SA, "SB": SB, "SAB": SAB, "SE": SE, "ST": ST, "eta": s(SA / ST, 2), "peta": s(SA / (SA + SE), 2),
        "d_ae": s(SA / SE, 2), "d_int": s(SA / (SA + SAB + SE), 2), "SAE": SA + SE, "SAABE": SA + SAB + SE,
        "J": J, "nn": nn, "sw": sw, "vm": vm, "mbar": mbar, "yx": yx, "se2": s(se2, 0), "tau2": s(tau2, 0),
        "B": s(B, 4), "B1": s(1 - B, 4), "shr": s(se2 + tau2, 0), "eb": s(eb, 2), "eb4": s(eb, 4),
        "d_rev": s(mbar + B * (yx - mbar), 2), "d_vm": s(mbar + (1 - B_wrong) * (yx - mbar), 2),
        "d_half": s((mbar + yx) / 2, 2), "d_raw": s(yx, 2), "Bw": s(B_wrong, 4), "dy": yx - mbar, "vmse": vm + int(se2),
    }


def p12():
    # (a) トービット
    xb, sig, Phi, phi = 1.2, 2.0, 0.7257, 0.3332
    p0 = 1 - Phi
    Ey = Phi * xb + sig * phi
    Epos = xb + sig * phi / Phi
    # (b) 平均値代入
    n, miss, s2 = 50, 10, 1.20
    nobs = n - miss
    v_imp = (nobs - 1) * s2 / (n - 1)
    # (c) 分位点回帰
    y = [0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    tau = 0.8
    nt = len(y) * tau
    k = int(nt) + 1  # nτ が整数でないので y_(k) がただ1つの最小点
    return {
        "xb": xb, "sig": sig, "Phi": Phi, "phi": phi, "z": s(xb / sig, 1), "p0": s(p0, 3), "p0_4": s(p0, 4),
        "t1": s(Phi * xb, 4), "t2": s(sig * phi, 4), "Ey": s(Ey, 2), "Ey4": s(Ey, 4), "Epos": s(Epos, 2),
        "d_noterm": s(Phi * xb, 2), "d_p": s(Phi, 3), "ratio": s(phi / Phi, 4),
        "n": n, "miss": miss, "nobs": nobs, "s2": s(s2, 2), "nobs1": nobs - 1, "n1": n - 1,
        "num": s((nobs - 1) * s2, 2), "vimp": r(v_imp, 3), "vimp_s": s(v_imp, 3), "vimp5": s(v_imp, 5),
        "ratio_v": s(v_imp / s2, 3), "d_n": s((nobs - 1) * s2 / n, 3),
        "y": "、".join(f"{v:.1f}" for v in y), "ny": len(y), "tau": tau, "nt": s(nt, 1),
        "k": k, "qk": s(y[k - 1], 1), "q_low": s(y[k - 2], 1), "q_high": s(y[k], 1), "ymean": s(sum(y) / len(y), 2),
        "ymed": s(sorted(y)[len(y) // 2], 1), "km1": k - 1, "nk": len(y) - k,
        "dl": s(tau * (k - 1) - (1 - tau) * 0, 1),
    }


def p13():
    w1, w2, b = 1.2, 0.5, -2.0
    nw = sqrt(w1 ** 2 + w2 ** 2)
    return {
        "w1": w1, "w2": w2, "b": b, "w1s": s(w1 ** 2, 2), "w2s": s(w2 ** 2, 2), "nw2": s(nw ** 2, 2), "nw": s(nw, 1),
        "margin": r(2 / nw, 2), "margin_s": s(2 / nw, 2), "margin4": s(2 / nw, 4), "d_half": s(1 / nw, 2),
        "d_sq": s(2 / nw ** 2, 2),
    }


def values():
    return {
        "s21p01": p01(), "s21p02": p02(), "s21p03": p03(), "s21p04": p04(), "s21p05": p05(),
        "s21p06": p06(), "s21p07": p07(), "s21p08": p08(), "s21p09": p09(), "s21p10": p10(),
        "s21p11": p11(), "s21p12": p12(), "s21p13": p13(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
