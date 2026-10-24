"""模擬試験 第19回（set19）の数値。values() が {大問id: {名前: 値}} を返す。

題材：銀行・与信・決済・家計。
問題文に示した分布の値（χ² の点・分布関数の値・t 点・指数や対数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import combinations
from math import comb, sqrt


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
    # (a) 2つの比率の差の検定（プールした割合、連続修正なし）
    n1, x1, n2, x2 = 400, 16, 500, 45
    p1, p2 = x1 / n1, x2 / n2
    pb = (x1 + x2) / (n1 + n2)
    inv = 1 / n1 + 1 / n2
    var0 = pb * (1 - pb) * inv
    se = sqrt(var0)
    z = (p2 - p1) / se
    se_un = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    se_add = sqrt(pb * (1 - pb) / n1) + sqrt(pb * (1 - pb) / n2)
    z_yates = ((p2 - p1) - 0.5 * inv) / se
    # (b) 結果に依存した抽出（延滞者は全件、非延滞者は 10%）のロジスティック回帰の切片補正
    b0, b1, x = 2.0, 0.06, 60
    f1, f0 = 1.0, 0.10
    eta = b0 - b1 * x                    # -1.6
    e16 = 4.9530                         # e^{1.6}
    odds_s = 1 / e16
    odds_p = odds_s * f0 / f1
    p_ok = odds_p / (1 + odds_p)
    p_raw = odds_s / (1 + odds_s)
    odds_rev = odds_s * f1 / f0
    return {
        "n1": n1, "x1": x1, "n2": n2, "x2": x2, "p1": s(p1, 2), "p2": s(p2, 2), "xs": x1 + x2, "ns": n1 + n2,
        "pb": s(pb, 5), "qb": s(1 - pb, 5), "inv": s(inv, 4), "var0": s(var0, 8), "se": s(se, 5), "dp": s(p2 - p1, 2),
        "z": s(z, 2), "z4": s(z, 4), "z_un": s((p2 - p1) / se_un, 2), "z_add": s((p2 - p1) / se_add, 2),
        "z_yates": s(z_yates, 2), "chi": s(z ** 2, 2), "se_un": s(se_un, 5),
        "b0": s(b0, 1), "b1": s(b1, 2), "x": x, "eta": s(eta, 1), "e16": s(e16, 4), "f0": s(f0, 2),
        "odds_s": s(odds_s, 5), "odds_p": s(odds_p, 6), "den": s(1 + 10 * e16, 2), "ten_e": s(10 * e16, 2),
        "p_ok": s(p_ok, 4), "p_ok5": s(p_ok, 5), "p_raw": s(p_raw, 3), "p_raw10": s(p_raw * f0, 4),
        "p_odds": s(odds_p, 4), "p_rev": s(odds_rev / (1 + odds_rev), 3), "ln10": 2.3026,
    }


def p02():
    n, S = 10, 40.0
    eff = Fraction(n - 2, n)
    lam0, lam1 = 0.1, 0.2
    q05 = 10.85                         # χ²(20) の下側 5% 点
    cut = q05 / (2 * lam0)
    F = {5.43: 0.0005, 10.85: 0.050, 21.70: 0.643, 31.41: 0.950, 43.40: 0.998}
    arg = r(2 * lam1 * cut, 2)
    assert arg in F
    power = F[arg]
    ln2 = 0.6931
    klA = -ln2 + lam1 / lam0 - 1
    klB = ln2 + 0.05 / lam0 - 1
    return {
        "n": n, "S": s(S, 0), "nm1": n - 1, "nm2": n - 2, "eff": s(eff, 3), "w_nm1": s(Fraction(n - 1, n), 3),
        "w_nm2": s(Fraction(n - 2, n - 1), 3), "w_inv": s(1 / eff, 3), "one": "1.000",
        "lam0": lam0, "lam1": lam1, "df": 2 * n, "q05": q05, "cut": s(cut, 2), "stat": s(2 * lam0 * S, 1),
        "arg": s(arg, 2), "power": s(power, 3), "w_one": s(1 - power, 3), "size": s(0.05, 3),
        "w_ratio": s(F[5.43], 4), "w_two": s(F[43.40], 3), "F543": s(F[5.43], 4), "F1085": s(F[10.85], 3),
        "F2170": s(F[21.70], 3), "F3141": s(F[31.41], 3), "F4340": s(F[43.40], 3), "cut2": s(q05 / lam0, 1),
        "ln2": ln2, "klA": s(klA, 3), "klB": s(klB, 3), "klA_s": s(ln2 + 1, 3), "klB_s": s(-ln2 - 0.5, 3),
    }


def p03():
    lam, delta = Fraction(1, 5), Fraction(1, 20)
    k = lam / delta
    ed = lam / (lam + delta)
    ed2 = lam / (lam + 2 * delta)
    vd = ed2 - ed ** 2
    vpv = 100 ** 2 * vd
    sd = sqrt(float(vpv))
    w_plus = lam / (lam - delta)
    return {
        "lam": s(lam, 1), "mean": int(1 / lam), "delta": s(delta, 2), "k": int(k), "km1": int(k - 1),
        "inv": s(delta / lam, 2), "invm1": s(delta / lam - 1, 2),
        "ed": s(ed, 1), "ed2_f": "2/3", "ed_sq": "16/25", "vd_f": "2/75",
        "vpv": s(vpv, 2), "sd": r(sd, 2), "sd_s": s(sd, 2), "sd4": s(sd, 4),
        "w_var": s(float(vpv), 2), "w_nosq": s(sqrt(float(100 ** 2 * ed2)), 2), "w_plus": s(w_plus, 3),
        "pv_mean": s(100 * ed, 0),
    }


def p04():
    # (a) 非回答者の追跡調査（二相抽出）
    n, nr, ybr, m, ybm = 400, 240, 9.0, 40, 6.0
    rr = nr / n
    est = rr * ybr + (1 - rr) * ybm
    # (b) 多重対応分析
    cats = [3, 4, 2]
    Q, J = len(cats), sum(cats)
    # (c) 分位点回帰のチェック損失
    xs = [20, 30, 40, 50, 60]
    ys = [9, 12, 16, 17, 24]
    tau = 0.9
    lines = {"A": 2, "B": 4, "C": 5}

    def loss(a, kind):
        u = [y - (a + 0.3 * x) for x, y in zip(xs, ys)]
        if kind == "check":
            return sum(tau * v if v >= 0 else (tau - 1) * v for v in u)
        if kind == "rev":
            return sum((1 - tau) * v if v >= 0 else -tau * v for v in u)
        if kind == "sq":
            return sum(v * v for v in u)
        if kind == "abs":
            return sum(0.5 * abs(v) for v in u)
        return sum(tau * abs(v) for v in u)

    def txt(kind):
        v = {k: loss(a, kind) for k, a in lines.items()}
        best = min(v, key=v.get)
        return "、".join(f"{k}: {s(v[k], 1)}" for k in "ABC") + f"。{best} を選ぶ"
    res = {k: [y - (a + 0.3 * x) for x, y in zip(xs, ys)] for k, a in lines.items()}
    vok = {k: loss(a, "check") for k, a in lines.items()}
    assert min(vok, key=vok.get) == "C"
    out = {
        "n": n, "nr": nr, "nn": n - nr, "ybr": s(ybr, 1), "m": m, "ybm": s(ybm, 1), "rr": s(rr, 1), "rn": s(1 - rr, 1),
        "est": r(est, 2), "est_s": s(est, 2), "bias": s(ybr - est, 2), "w_m": s((nr * ybr + m * ybm) / (nr + m), 2),
        "Q": Q, "J": J, "c1": cats[0], "c2": cats[1], "c3": cats[2], "in_ok": s((J - Q) / Q, 1), "dim_ok": J - Q,
        "in_jq": s(J / Q, 1), "in_nq": s(J - Q, 1), "in_j": s((J - Q) / J, 3), "dim_j1": J - 1,
        "tau": tau, "tau1": s(1 - tau, 1),
        "c_ok": txt("check"), "c_rev": txt("rev"), "c_sq": txt("sq"), "c_abs": txt("abs"), "c_tabs": txt("tauabs"),
        "LA": s(vok["A"], 1), "LB": s(vok["B"], 1), "LC": s(vok["C"], 1),
    }
    for i, (x, y) in enumerate(zip(xs, ys)):
        out[f"x{i}"] = x
        out[f"y{i}"] = y
        for k in "ABC":
            out[f"r{k}{i}"] = s(res[k][i], 0)
    return out


def p05():
    n, s11, s22, s12, s1y, s2y, s2 = 30, 40, 90, 30, 22, 27, 0.25
    det = s11 * s22 - s12 ** 2
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    v11 = s22 / det
    se1 = sqrt(s2 * v11)
    t = b1 / se1
    se_simple = sqrt(s2 / s11)
    return {
        "n": n, "s11": s11, "s22": s22, "s12": s12, "s1y": s1y, "s2y": s2y, "s2": s2, "dfe": n - 3, "det": det,
        "n1": s22 * s1y, "n2": s12 * s2y, "n3": s11 * s2y, "n4": s12 * s1y, "b1n": s22 * s1y - s12 * s2y,
        "b2n": s11 * s2y - s12 * s1y,
        "b1": s(b1, 4), "b2": s(b2, 4), "v11": s(v11, 5), "var1": s(s2 * v11, 6), "se1": s(se1, 5),
        "t": s(t, 2), "t4": s(t, 4), "w_simple": s(b1 / se_simple, 2), "w_slope": s((s1y / s11) / se_simple, 2),
        "w_elem": s(b1 / sqrt(s2 * s11 / det), 2), "w_nosq": s(b1 / (s2 * v11), 1), "se_simple": s(se_simple, 5),
        "b_simple": s(s1y / s11, 2),
    }


def p06():
    e = [0.8, 1.1, 0.6, -0.2, -0.9, -1.2, -0.4, 0.2]
    ss = sum(v * v for v in e)
    dd = [e[i] - e[i - 1] for i in range(1, len(e))]
    sd = sum(v * v for v in dd)
    d = sd / ss
    r1 = sum(e[i] * e[i - 1] for i in range(1, len(e))) / ss
    t1, t2 = 0.5, -0.3
    g0 = 1 + t1 ** 2 + t2 ** 2
    rho1 = (t1 + t1 * t2) / g0
    rho2 = t2 / g0

    def tri(a, b, c):
        return f"\\rho(1)={s(a, 3)},\\ \\rho(2)={s(b, 3)},\\ \\rho(3)={s(c, 3)}"
    return {
        "e": "、".join(f"{v:.1f}" for v in e), "ne": len(e), "ss": s(ss, 2), "sd": s(sd, 2),
        "dd": "、".join(f"{v:.1f}" for v in dd), "d": r(d, 2), "d_s": s(d, 2), "d4": s(d, 4),
        "r1": s(r1, 3), "w_rd": s(1 - d / 2, 3), "w_inv": s(ss / sd, 2), "w_half": s(sd / ss / 2, 2),
        "t1": t1, "t2": s(t2, 1), "mt2": s(-t2, 1), "g0": s(g0, 2), "t1t2": s(t1 * t2, 2), "num1": s(t1 + t1 * t2, 2),
        "rho1": s(rho1, 3), "rho2": s(rho2, 3),
        "c_ok": tri(rho1, rho2, 0), "c_th": tri(t1, t2, 0), "c_nox": tri(t1 / g0, rho2, 0),
        "c_nog": tri(t1 + t1 * t2, t2, 0), "c_dec": tri(rho1, rho2, rho1 * rho2),
    }


def p07():
    pts = {"P1": (0.0, 0.0, "正常"), "P2": (0.5, 1.0, "延滞"), "P3": (-0.5, 0.5, "延滞"),
           "P4": (1.0, 0.0, "正常"), "P5": (0.0, 1.2, "延滞"), "P6": (0.6, 0.3, "正常")}
    new = (0.2, 0.4)
    d2 = {k: (v[0] - new[0]) ** 2 + (v[1] - new[1]) ** 2 for k, v in pts.items()}
    order = sorted(d2, key=d2.get)
    assert order == ["P6", "P1", "P2", "P3", "P5", "P4"]
    out = {"nz1": new[0], "nz2": new[1]}
    for k, v in pts.items():
        out[f"{k}a"] = s(v[0], 1)
        out[f"{k}b"] = s(v[1], 1)
        out[f"{k}y"] = v[2]
        out[f"{k}d"] = s(d2[k], 2)
    # (b) 歪度
    p, n = 0.02, 100
    sk1 = (1 - 2 * p) / sqrt(p * (1 - p))
    skn = sk1 / sqrt(n)
    out.update({"p": p, "q": s(1 - p, 2), "pq": s(p * (1 - p), 4), "spq": s(sqrt(p * (1 - p)), 2),
                "q2p": s(1 - 2 * p, 2), "nl": n, "sk1": s(sk1, 3), "skn": s(skn, 3), "w_n": s(sk1 / n, 3),
                "w_x": s(sk1 * sqrt(n), 3), "w_num": s(1 - 2 * p, 3)})
    # (c) ベイズ決定
    al, be, nn, y = 2, 48, 50, 4
    ap, bp = al + y, be + nn - y
    gain, lossv = 6, 90
    pm = Fraction(ap, ap + bp)

    def prof(th):
        return gain - (gain + lossv) * th
    out.update({
        "al": al, "be": be, "nn": nn, "y": y, "ny": nn - y, "ap": ap, "bp": bp, "apb": ap + bp, "gain": gain, "loss": lossv,
        "gl": gain + lossv, "pm": s(pm, 2), "prof": s(prof(pm), 2), "thstar": s(Fraction(gain, gain + lossv), 4),
        "w_mle": s(prof(Fraction(y, nn)), 2), "w_map": s(prof(Fraction(ap - 1, ap + bp - 2)), 2),
        "w_prior": s(prof(Fraction(al, al + be)), 2), "w_bad": s(prof(Fraction(ap, ap + be + nn)), 2),
        "mle": s(Fraction(y, nn), 2), "map": s(Fraction(ap - 1, ap + bp - 2), 4), "badb": be + nn,
        "prior_m": s(Fraction(al, al + be), 2),
    })
    return out


def p08():
    nper = 4
    der = sum(Fraction((-1) ** k, math.factorial(k)) for k in range(nper + 1))
    out = {"np": nper, "der": s(der, 3), "w_ind": s((1 - 1 / nper) ** nper, 3), "w_e": s(0.3679, 3),
           "w_trunc": s(Fraction(1, 2), 3), "w_comp": s(1 - der, 3), "fact": math.factorial(nper),
           "nder": int(der * math.factorial(nper))}
    N, D, n = 50, 3, 10
    p0 = comb(N - D, n) / comb(N, n)
    p1 = D * comb(N - D, n - 1) / comb(N, n)
    ans = 1 - p0
    binom = 1 - (1 - D / N) ** n
    out.update({
        "N": N, "D": D, "n": n, "ND": N - D,
        "p0_num": 40 * 39 * 38, "p0_den": 50 * 49 * 48, "p0": s(p0, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_binom": s(binom, 3), "w_mean": s(n * D / N, 3), "w_p1": s(p1, 3), "w_p0": s(p0, 3),
        "q94": s((1 - D / N) ** n, 4),
    })
    return out


def p09():
    n1, m1, v1, n2, m2, v2 = 10, 5.20, 0.90, 15, 4.40, 0.30
    a, b = v1 / n1, v2 / n2
    se = sqrt(a + b)
    nu = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    tt = {9: 2.262, 13: 2.160, 23: 2.069}
    nuf = int(nu)
    assert nuf == 13
    diff = m1 - m2

    def iv(c, h):
        return f"({s(c - h, 3)},\\ {s(c + h, 3)})"
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    se_p = sqrt(sp2 * (1 / n1 + 1 / n2))
    out = {
        "n1": n1, "m1": s(m1, 2), "v1": s(v1, 2), "n2": n2, "m2": s(m2, 2), "v2": s(v2, 2), "a": s(a, 2), "b": s(b, 2),
        "ab": s(a + b, 2), "se": s(se, 5), "a2": s(a ** 2 / (n1 - 1), 7), "b2": s(b ** 2 / (n2 - 1), 7),
        "ab2": s((a + b) ** 2, 4), "nu": s(nu, 2), "nuf": nuf, "t9": s(tt[9], 3), "t13": s(tt[13], 3), "t23": s(tt[23], 3),
        "diff": s(diff, 2), "half": s(tt[13] * se, 4),
        "ci_ok": iv(diff, tt[13] * se), "ci_pool": iv(diff, tt[23] * se_p), "ci_23": iv(diff, tt[23] * se),
        "ci_z": iv(diff, 1.96 * se), "ci_9": iv(diff, tt[9] * se), "sp2": s(sp2, 4),
    }
    P = [[Fraction(90, 100), Fraction(6, 100), Fraction(4, 100)], [Fraction(5, 10), Fraction(4, 10), Fraction(1, 10)],
         [Fraction(6, 10), Fraction(0), Fraction(4, 10)]]
    pi = (Fraction(60, 71), Fraction(6, 71), Fraction(5, 71))
    for j in range(3):
        assert sum(pi[i] * P[i][j] for i in range(3)) == pi[j]
    names = ["nn", "nm", "ns", "mn", "mm", "ms", "sn", "sm", "ss"]
    for k, v in zip(names, [x for rw in P for x in rw]):
        out["P" + k] = s(v, 2)
    out.update({"piN": "60/71", "piM": "6/71", "piS": "5/71", "ret": s(1 / pi[2], 1), "w_pi": s(pi[2], 3),
                "w_stay": s(1 / (1 - P[2][2]), 2), "w_geo": s(1 / P[0][2], 1), "w_m1": s(1 / pi[2] - 1, 1)})
    return out


def p10():
    A = [8, 11, 12]
    B = [5, 7, 9]
    allv = A + B
    tot = sum(allv)
    obs = sum(A)
    sums = [sum(c) for c in combinations(allv, 3)]
    ge = sum(1 for v in sums if v >= obs)
    gt = sum(1 for v in sums if v > obs)
    return {"A": "、".join(map(str, A)), "B": "、".join(map(str, B)), "obs": obs, "tot": tot, "obsB": tot - obs,
            "mA": s(obs / 3, 2), "mB": s((tot - obs) / 3, 2), "dobs": s(obs / 3 - (tot - obs) / 3, 2),
            "ncomb": len(sums), "ge": ge, "p": r(ge / len(sums), 2), "p_s": s(ge / len(sums), 2),
            "w_gt": s(gt / len(sums), 2), "w_two": s(2 * ge / len(sums), 2)}


def p11():
    n, mu, sd = 48, 1.0, 4.0
    sr = mu / sd
    var = (1 + sr ** 2 / 2) / n
    se = sqrt(var)
    return {"n": n, "mu": s(mu, 1), "sd": s(sd, 1), "sr": s(sr, 2), "sr2h": s(sr ** 2 / 2, 5), "var": s(var, 7),
            "se": r(se, 3), "se_s": s(se, 3), "se4": s(se, 4), "w_mean": s(sqrt(1 / n), 3)}


def p12():
    l1, l2 = 2.004, 0.613
    e11, e21 = 0.580, 0.559
    a1, a2 = l1 * e11 ** 2, l2 * e21 ** 2
    tot = a1 + a2
    return {"l1": s(l1, 3), "l2": s(l2, 3), "e11": s(e11, 3), "e12": s(0.613, 3), "e13": s(0.536, 3),
            "e21": s(e21, 3), "e22": s(0.179, 3), "e23": s(-0.810, 3), "e11sq": s(e11 ** 2, 4), "e21sq": s(e21 ** 2, 4),
            "a1": s(a1, 4), "a2": s(a2, 4), "ans": r(tot, 3), "ans_s": s(tot, 3), "ans4": s(tot, 4),
            "w_pc1": s(a1, 3), "w_e2": s(e11 ** 2 + e21 ** 2, 3), "w_noq": s(l1 * e11 + l2 * e21, 3)}


def p13():
    y = {"A": [9, 13, 11], "B": [7, 10, 7], "C": [12, 15, 15], "D": [8, 10, 9]}
    a, b = 4, 3
    allv = [v for row in y.values() for v in row]
    gm = sum(allv) / len(allv)
    rm = {k: sum(v) / b for k, v in y.items()}
    cm = [sum(y[k][j] for k in y) / a for j in range(b)]
    sst = sum((v - gm) ** 2 for v in allv)
    ssa = b * sum((m - gm) ** 2 for m in rm.values())
    ssb = a * sum((m - gm) ** 2 for m in cm)
    sse = sst - ssa - ssb
    dfa, dfb, dfe = a - 1, b - 1, (a - 1) * (b - 1)
    F = (ssa / dfa) / (sse / dfe)
    F1 = (ssa / dfa) / ((sst - ssa) / (a * b - a))
    out = {"gm": s(gm, 1), "ssa": s(ssa, 1), "ssb": s(ssb, 1), "sst": s(sst, 1), "sse": s(sse, 1), "dfa": dfa, "dfb": dfb,
           "dfe": dfe, "msa": s(ssa / dfa, 1), "mse": s(sse / dfe, 4), "F": r(F, 2), "F_s": s(F, 2), "F1": s(F1, 2),
           "dfe1": a * b - a, "sse1": s(sst - ssa, 1), "fcrit": 4.757, "Fb": s((ssb / dfb) / (sse / dfe), 2),
           "msb": s(ssb / dfb, 1)}
    for k, v in y.items():
        for j, x in enumerate(v):
            out[f"y{k}{j}"] = x
        out[f"m{k}"] = s(rm[k], 1)
        out[f"d{k}"] = s(rm[k] - gm, 1)
    for j, m in enumerate(cm):
        out[f"c{j}"] = s(m, 1)
        out[f"dc{j}"] = s(m - gm, 1)
    return out


def p14():
    a, b, c, d = 80, 120, 80, 320
    orr = a * d / (b * c)
    lnor = 0.9808
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    h = 1.96 * se
    ex = {0.6066: 1.834, 1.3550: 3.877, 0.6667: 1.948, 1.2949: 3.651}
    lo, hi = r(lnor - h, 4), r(lnor + h, 4)
    assert lo in ex and hi in ex, (lo, hi)
    h90 = 1.645 * se
    lo9, hi9 = r(lnor - h90, 4), r(lnor + h90, 4)
    assert lo9 in ex and hi9 in ex, (lo9, hi9)
    se_or = orr * se

    def iv(x, y):
        return f"({s(x, 3)},\\ {s(y, 3)})"
    return {
        "a": a, "b": b, "c": c, "d": d, "na": a + b, "nc": c + d, "or": s(orr, 4), "or3": s(orr, 3), "lnor": lnor,
        "sum": s(1 / a + 1 / b + 1 / c + 1 / d, 6), "se": s(se, 5), "h": s(h, 4), "lo": s(lo, 4), "hi": s(hi, 4),
        "elo": ex[lo], "ehi": ex[hi], "elo9": ex[lo9], "ehi9": ex[hi9], "lo9": s(lo9, 4), "hi9": s(hi9, 4),
        "ci_ok": iv(ex[lo], ex[hi]), "ci_90": iv(ex[lo9], ex[hi9]), "ci_log": iv(lo, hi),
        "ci_raw": iv(orr - 1.96 * se_or, orr + 1.96 * se_or), "se_or": s(se_or, 4),
        "pc": s(a / (a + b), 2), "pk": s(c / (c + d), 2), "w_rr": s((a / (a + b)) / (c / (c + d)), 2),
        "ci_half": iv(math.exp(lnor - 1.96 * sqrt(1 / a + 1 / b)), math.exp(lnor + 1.96 * sqrt(1 / a + 1 / b))),
    }


def values():
    return {
        "s19p01": p01(), "s19p02": p02(), "s19p03": p03(), "s19p04": p04(), "s19p05": p05(), "s19p06": p06(),
        "s19p07": p07(), "s19p08": p08(), "s19p09": p09(), "s19p10": p10(), "s19p11": p11(), "s19p12": p12(),
        "s19p13": p13(), "s19p14": p14(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
