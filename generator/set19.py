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
    # (a) 1標本の比率の片側検定の標本の大きさ
    p0, p1, za, zb, z2 = 0.05, 0.08, 1.645, 0.842, 1.96
    s0, s1 = sqrt(p0 * (1 - p0)), sqrt(p1 * (1 - p1))
    num = za * s0 + zb * s1
    root = num / (p1 - p0)
    nreq = math.ceil(root ** 2)
    # (b) 結果に依存した抽出（延滞者は全件、非延滞者は 10%）のロジスティック回帰の切片補正
    b0, b1, x = 2.0, 0.06, 60
    f1, f0 = 1.0, 0.10
    eta = b0 - b1 * x
    e16 = 4.9530
    odds_s = 1 / e16
    odds_p = odds_s * f0 / f1
    p_ok = odds_p / (1 + odds_p)
    p_raw = odds_s / (1 + odds_s)
    odds_rev = odds_s * f1 / f0
    return {
        "p0": p0, "p1": p1, "za": za, "zb": zb, "z2": z2, "s0": s(s0, 5), "s1": s(s1, 5), "pq0": s(p0 * (1 - p0), 4),
        "pq1": s(p1 * (1 - p1), 4), "t0": s(za * s0, 5), "t1": s(zb * s1, 5), "num": s(num, 5), "dp": s(p1 - p0, 2),
        "root": s(root, 3), "sq": s(root ** 2, 1), "nreq": nreq,
        "w_p1": math.ceil(((za + zb) * s1 / (p1 - p0)) ** 2), "w_p0": math.ceil(((za + zb) * s0 / (p1 - p0)) ** 2),
        "w_two": math.ceil(((z2 * s0 + zb * s1) / (p1 - p0)) ** 2), "w_nob": math.ceil((za * s0 / (p1 - p0)) ** 2),
        "b0": s(b0, 1), "b1": s(b1, 2), "x": x, "eta": s(eta, 1), "e16": s(e16, 4), "f0": s(f0, 2),
        "odds_s": s(odds_s, 5), "odds_p": s(odds_p, 6), "den": s(1 + 10 * e16, 2),
        "p_ok": s(p_ok, 4), "p_ok5": s(p_ok, 5), "p_raw": s(p_raw, 3), "p_raw10": s(p_raw * f0, 4),
        "p_odds": s(odds_p, 4), "p_rev": s(odds_rev / (1 + odds_rev), 3), "ln10": 2.3026,
    }


def p02():
    # (a) 分散指数検定
    xs = [1, 0, 2, 0, 1, 3, 0, 1, 0, 2, 1, 1]
    n = len(xs)
    T = sum(xs)
    m = T / n
    sx2 = sum(v * v for v in xs)
    ss = sx2 - n * m ** 2
    D = ss / m
    # (b) ゼロ切断ポアソン
    ncard, ntx = 40, 60
    xb = ntx / ncard
    cand = {0.5: 0.6065, 0.667: 0.5132, 0.874: 0.4173, 1.165: 0.3119, 1.5: 0.2231}
    g = {k: k / (1 - v) for k, v in cand.items()}
    best = min(g, key=lambda k: abs(g[k] - xb))
    assert best == 0.874 and abs(g[best] - xb) < 0.001
    # (c) 条件付き独立でない2つのルール
    pf, both_f, both_l = 0.002, 0.75, 0.004
    sa_f, sb_f, sa_l, sb_l = 0.90, 0.80, 0.02, 0.03
    post = pf * both_f / (pf * both_f + (1 - pf) * both_l)
    post_ind = pf * sa_f * sb_f / (pf * sa_f * sb_f + (1 - pf) * sa_l * sb_l)
    post_a = pf * sa_f / (pf * sa_f + (1 - pf) * sa_l)
    out = {
        "data": "、".join(str(v) for v in xs), "n": n, "T": T, "m": s(m, 1), "sx2": sx2, "ss": s(ss, 1),
        "nm2": s(n * m ** 2, 1), "D": s(D, 1), "df": n - 1, "ratio": s(ss / (n - 1) / m, 2), "D_raw": s(sx2 / m, 1),
        "chi11": 19.68, "chi12": 21.03, "chi11_lo": 4.575,
        "ncard": ncard, "ntx": ntx, "xb": s(xb, 1),
        "pf": pf, "both_f": both_f, "both_l": both_l, "sa_f": s(sa_f, 2), "sb_f": s(sb_f, 2), "sa_l": s(sa_l, 2), "sb_l": s(sb_l, 2),
        "nf": s(pf * both_f, 4), "nl": s((1 - pf) * both_l, 6), "dsum": s(pf * both_f + (1 - pf) * both_l, 6),
        "post": s(post, 3), "post4": s(post, 4), "w_ind": s(post_ind, 3), "w_a": s(post_a, 3), "w_like": s(both_f, 3),
        "w_noprior": s(both_f / (both_f + both_l), 3), "prodl": s(sa_l * sb_l, 4), "prodf": s(sa_f * sb_f, 2),
    }
    for i, (k, v) in enumerate(sorted(cand.items())):
        out[f"lc{i}"] = s(k, 3)
        out[f"ec{i}"] = s(v, 4)
        out[f"gc{i}"] = s(g[k], 3)
    out["lam"] = s(best, 3)
    return out


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
    # (a) 費用を考えた最適配分
    N1, S1, c1, N2, S2, c2, ntot = 8000, 40, 1, 500, 400, 16, 370
    w1, w2 = N1 * S1 / sqrt(c1), N2 * S2 / sqrt(c2)
    n2 = ntot * w2 / (w1 + w2)
    ney2 = ntot * N2 * S2 / (N1 * S1 + N2 * S2)
    prop2 = ntot * N2 / (N1 + N2)
    cost2 = ntot * N2 * S2 / c2 / (N1 * S1 / c1 + N2 * S2 / c2)
    # (b) 多重対応分析
    cats = [3, 4, 2]
    Q, J = len(cats), sum(cats)
    return {
        "N1": N1, "S1": S1, "c1": c1, "N2": N2, "S2": S2, "c2": c2, "ntot": ntot, "sc2": int(sqrt(c2)),
        "w1": int(w1), "w2": int(w2), "wsum": int(w1 + w2), "n2": r(n2, 0), "n1": ntot - r(n2, 0), "n2_s": s(n2, 1),
        "w_ney": s(ney2, 1), "w_prop": s(prop2, 1), "w_cost": s(cost2, 1),
        "Q": Q, "J": J, "c1q": cats[0], "c2q": cats[1], "c3q": cats[2], "in_ok": s((J - Q) / Q, 1), "dim_ok": J - Q,
        "in_jq": s(J / Q, 1), "in_nq": s(J - Q, 1), "in_j": s((J - Q) / J, 3), "dim_j1": J - 1,
    }


def p05():
    # (a) 平均応答の信頼区間
    n, xbar, sxx, sd, x0, t18 = 20, 60, 2000, 0.30, 80, 2.101
    k = 1 / n + (x0 - xbar) ** 2 / sxx
    se_m = sd * sqrt(k)
    half = t18 * se_m
    # (b) BIC からモデルの事後確率
    bic1, bic2, e21 = 412.6, 416.8, 8.166
    post = e21 / (1 + e21)
    return {
        "n": n, "xbar": xbar, "sxx": sxx, "sd": s(sd, 2), "x0": x0, "t18": s(t18, 3), "dfe": n - 2,
        "k1": s(1 / n, 2), "k2": s((x0 - xbar) ** 2 / sxx, 2), "k": s(k, 2), "sqk": s(sqrt(k), 2), "se_m": s(se_m, 3),
        "half": s(half, 3), "half4": s(half, 4), "w_pred": s(t18 * sd * sqrt(1 + k), 3), "w_z": s(1.96 * se_m, 3),
        "w_nodev": s(t18 * sd * sqrt(1 / n), 3), "w_lin": s(t18 * sd * sqrt(1 / n + (x0 - xbar) / sxx), 3),
        "bic1": bic1, "bic2": bic2, "dbic": s(bic2 - bic1, 1), "e21": e21, "em21": 0.1225, "e42": 66.69,
        "post": s(post, 3), "w_m2": s(1 - post, 3), "w_no2": s(66.69 / 67.69, 3), "w_ratio": s(bic2 / (bic1 + bic2), 3),
        "w_exp": s(1 - 0.1225, 3),
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
    # (b) エラスティックネット（直交計画）
    b = [1.20, -0.50, 0.25]
    l1, l2 = 0.6, 0.5

    def soft(v, t):
        return math.copysign(max(abs(v) - t, 0), v)

    def tri(v):
        return "(" + ",\\ ".join(s(x + 0.0, 3) for x in v) + ")"
    en = [soft(v, l1 / 2) / (1 + l2) for v in b]
    out.update({"b1": s(b[0], 2), "b2": s(b[1], 2), "b3": s(b[2], 2), "l1": l1, "l2": l2, "h1": s(l1 / 2, 1), "d1": s(1 + l2, 1),
                "en": tri(en), "w_lasso": tri([soft(v, l1 / 2) for v in b]), "w_ridge": tri([v / (1 + l2) for v in b]),
                "w_full": tri([soft(v, l1) / (1 + l2) for v in b]), "w_two": tri([soft(v, l1 / 2) / (1 + 2 * l2) for v in b]),
                "s1": s(soft(b[0], l1 / 2), 2), "s2": s(soft(b[1], l1 / 2), 2), "en1": s(en[0], 3), "en2": s(en[1], 3)})
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
        "mle": s(Fraction(y, nn), 2), "prior_m": s(Fraction(al, al + be), 2),
    })
    return out


def p08():
    N, D, n = 50, 3, 10
    p0 = comb(N - D, n) / comb(N, n)
    p1 = D * comb(N - D, n - 1) / comb(N, n)
    ans = 1 - p0
    binom = 1 - (1 - D / N) ** n
    out = {
        "N": N, "D": D, "n": n, "ND": N - D,
        "p0_num": 40 * 39 * 38, "p0_den": 50 * 49 * 48, "p0": s(p0, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_binom": s(binom, 3), "w_mean": s(n * D / N, 3), "w_p1": s(p1, 3), "w_p0": s(p0, 3),
        "q94": s((1 - D / N) ** n, 4),
    }
    # (b) 符号検定
    before = [12, 9, 15, 7, 10, 8, 14, 11, 6, 13, 9, 10]
    after = [9, 7, 11, 7, 8, 9, 10, 8, 5, 9, 10, 7]
    d = [a - b for a, b in zip(before, after)]
    npos = sum(1 for v in d if v > 0)
    nneg = sum(1 for v in d if v < 0)
    m = npos + nneg
    cnt = sum(comb(m, k) for k in range(npos, m + 1))
    pv = cnt / 2 ** m
    cnt12 = sum(comb(12, k) for k in range(npos, 13))
    for i, (a, b) in enumerate(zip(before, after)):
        out[f"bf{i}"] = a
        out[f"af{i}"] = b
    out.update({"npos": npos, "nneg": nneg, "nz": len(d) - m, "m": m, "c9": comb(m, 9), "c10": comb(m, 10), "c11": comb(m, 11),
                "cnt": cnt, "pow": 2 ** m, "sp": r(pv, 3), "sp_s": s(pv, 3), "sp4": s(pv, 4),
                "w_12": s(cnt12 / 2 ** 12, 3), "w_two": s(2 * pv, 3), "w_eq": s(comb(m, npos) / 2 ** m, 3),
                "w_gt": s(sum(comb(m, k) for k in range(npos + 1, m + 1)) / 2 ** m, 3)})
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
        "ci_z": iv(diff, 1.96 * se), "ci_9": iv(diff, tt[9] * se),
    }
    P = [[Fraction(90, 100), Fraction(6, 100), Fraction(4, 100)], [Fraction(5, 10), Fraction(4, 10), Fraction(1, 10)],
         [Fraction(6, 10), Fraction(0), Fraction(4, 10)]]
    names = ["nn", "nm", "ns", "mn", "mm", "ms", "sn", "sm", "ss"]
    for k, v in zip(names, [x for rw in P for x in rw]):
        out["P" + k] = s(v, 2)
    two = sum(P[0][k] * P[k][2] for k in range(3))
    out.update({"q1": s(P[0][0] * P[0][2], 3), "q2": s(P[0][1] * P[1][2], 3), "q3": s(P[0][2] * P[2][2], 3), "two": s(two, 3),
                "w_twice": s(2 * P[0][2], 3), "w_sq": s(P[0][2] ** 2, 4), "w_noss": s(two - P[0][2] * P[2][2], 3),
                "w_ind": s(1 - (1 - P[0][2]) ** 2, 4)})
    return out


def p10():
    tabs = [(80, 20, 60, 40), (20, 30, 15, 35)]
    diff, var = 0.0, 0.0
    out = {}
    stat_sum = 0.0
    for i, (a, b, c, d) in enumerate(tabs, 1):
        n = a + b + c + d
        r1, r2, k1, k2 = a + b, c + d, a + c, b + d
        e = r1 * k1 / n
        v = r1 * r2 * k1 * k2 / (n ** 2 * (n - 1))
        diff += a - e
        var += v
        stat_sum += n * (a * d - b * c) ** 2 / (r1 * r2 * k1 * k2)
        out.update({f"a{i}": a, f"b{i}": b, f"c{i}": c, f"d{i}": d, f"n{i}": n, f"r{i}": r1, f"k{i}": k1, f"l{i}": k2, f"q{i}": r2,
                    f"e{i}": s(e, 1), f"v{i}": s(v, 3), f"dd{i}": s(a - e, 1)})
    cmh = diff ** 2 / var
    cmh_cc = (abs(diff) - 0.5) ** 2 / var
    varN = sum((a + b) * (c + d) * (a + c) * (b + d) / (a + b + c + d) ** 3 for a, b, c, d in tabs)
    A = sum(t[0] for t in tabs); B = sum(t[1] for t in tabs); C = sum(t[2] for t in tabs); D = sum(t[3] for t in tabs)
    Nn = A + B + C + D
    pooled = Nn * (A * D - B * C) ** 2 / ((A + B) * (C + D) * (A + C) * (B + D))
    out.update({"diff": s(diff, 1), "var": s(var, 3), "cmh": s(cmh, 2), "cmh4": s(cmh, 4), "w_cc": s(cmh_cc, 2),
                "w_N": s(diff ** 2 / varN, 2), "w_pool": s(pooled, 2), "w_sum": s(stat_sum, 2), "diff2": s(diff ** 2, 2)})
    return out


def p11():
    # (a) 混合正規の尖度
    w1, s1, w2, s2 = 0.9, 1.0, 0.1, 3.0
    v = w1 * s1 ** 2 + w2 * s2 ** 2
    m4 = 3 * (w1 * s1 ** 4 + w2 * s2 ** 4)
    kurt = m4 / v ** 2
    # (b) シャープレシオ
    n, mu, sd = 48, 1.0, 4.0
    sr = mu / sd
    var = (1 + sr ** 2 / 2) / n
    se = sqrt(var)
    return {"w1": w1, "w2": w2, "s1": s(s1, 1), "s2": s(s2, 1), "v": s(v, 1), "v2": s(v ** 2, 2), "m4": s(m4, 1),
            "m4b": s(w1 * s1 ** 4 + w2 * s2 ** 4, 1), "kurt": s(kurt, 2), "w_ex": s(kurt - 3, 2), "w_norm": "3.00",
            "w_no3": s(m4 / 3 / v ** 2, 2), "w_v": s(m4 / v, 2),
            "n": n, "mu": s(mu, 1), "sd": s(sd, 1), "sr": s(sr, 2), "sr2h": s(sr ** 2 / 2, 5), "var": s(var, 7),
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


def values():
    return {f"s19p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 14)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
