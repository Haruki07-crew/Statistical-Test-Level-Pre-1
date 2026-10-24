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
    # (b) 口座数（曝露量）の異なる3グループのポアソン件数から共通の率の最尤推定
    tt, xx = [20, 50, 30], [6, 9, 9]
    lam = sum(xx) / sum(tt)
    se_l = sqrt(lam / sum(tt))
    avg = sum(x_ / t_ for x_, t_ in zip(xx, tt)) / 3

    def pr(a_, b_):
        return f"\\hat\\lambda={s(a_, 3)},\\ \\text{{標準誤差}}={s(b_, 3)}"
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
        "t1": tt[0], "t2": tt[1], "t3": tt[2], "x1": xx[0], "x2": xx[1], "x3": xx[2], "sx": sum(xx), "st": sum(tt),
        "lamh": s(lam, 2), "info": s(sum(tt) / lam, 1), "se_l": s(se_l, 4),
        "c_ok": pr(lam, se_l), "c_avg": pr(avg, sqrt(avg / sum(tt))), "c_n": pr(lam, sqrt(lam / 3)),
        "c_sq": pr(lam, lam / sqrt(sum(tt))), "c_var": pr(lam, lam / sum(tt)), "avg": s(avg, 2),
        "pf": pf, "both_f": both_f, "both_l": both_l, "sa_f": s(sa_f, 2), "sb_f": s(sb_f, 2), "sa_l": s(sa_l, 2), "sb_l": s(sb_l, 2),
        "nf": s(pf * both_f, 4), "nl": s((1 - pf) * both_l, 6), "dsum": s(pf * both_f + (1 - pf) * both_l, 6),
        "post": s(post, 3), "post4": s(post, 4), "w_ind": s(post_ind, 3), "w_a": s(post_a, 3), "w_like": s(both_f, 3),
        "w_noprior": s(both_f / (both_f + both_l), 3), "prodl": s(sa_l * sb_l, 4), "prodf": s(sa_f * sb_f, 2),
    }
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
    # (a) 回帰推定量に必要な標本の大きさ
    n0, rho = 400, 0.8
    nreg = n0 * (1 - rho ** 2)
    # (b) 合成信頼性
    ld = [0.8, 0.7, 0.6]
    sl = sum(ld)
    th = [1 - v ** 2 for v in ld]
    omega = sl ** 2 / (sl ** 2 + sum(th))
    return {
        "n0": n0, "rho": rho, "rho2": s(rho ** 2, 2), "f": s(1 - rho ** 2, 2), "nreg": r(nreg, 0),
        "w_1r": r(n0 * (1 - rho), 0), "w_r2": r(n0 * rho ** 2, 0), "w_sq": r(n0 * sqrt(1 - rho ** 2), 0),
        "w_div": r(n0 / (1 - rho ** 2), 0),
        "l1": ld[0], "l2": ld[1], "l3": ld[2], "sl": s(sl, 1), "sl2": s(sl ** 2, 2), "th1": s(th[0], 2), "th2": s(th[1], 2),
        "th3": s(th[2], 2), "sth": s(sum(th), 2), "den": s(sl ** 2 + sum(th), 2), "omega": s(omega, 3), "omega4": s(omega, 4),
        "w_ave": s(sum(v * v for v in ld) / 3, 3), "w_lin": s(sl / (sl + sum(th)), 3), "w_mean": s(sl / 3, 3),
        "w_sq2": s(sl ** 2 / (sl ** 2 + sum(v * v for v in th)), 3),
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
    # (b) リッジ推定量の偏りと分散（直交計画）
    beta, sig2 = 1.0, 0.25
    lam_s = sig2 / beta ** 2

    def mse(lm):
        return (sig2 + lm ** 2 * beta ** 2) / (1 + lm) ** 2
    out.update({"beta": s(beta, 1), "sig2": sig2, "lam": s(lam_s, 2), "mse": s(mse(lam_s), 3), "w_ols": s(sig2, 3),
                "w_inv": s(beta ** 2 / sig2, 2), "mse_inv": s(mse(beta ** 2 / sig2), 3),
                "w_var": s(sig2 / (1 + lam_s) ** 2, 3), "w_bias": s(lam_s ** 2 * beta ** 2 / (1 + lam_s) ** 2, 3),
                "d15": s((1 + lam_s) ** 2, 4), "num": s(sig2 + lam_s ** 2 * beta ** 2, 4), "lb2": s(lam_s ** 2 * beta ** 2, 4)})
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
    x = 16
    rt = sqrt(x)
    h = 1.96 / 2
    lo, hi = (rt - h) ** 2, (rt + h) ** 2

    def iv(a_, b_):
        return f"({s(a_, 2)},\\ {s(b_, 2)})"
    out = {"x": x, "rt": s(rt, 0), "h": s(h, 2), "rlo": s(rt - h, 2), "rhi": s(rt + h, 2),
           "ci_ok": iv(lo, hi), "ci_wald": iv(x - 1.96 * rt, x + 1.96 * rt), "ci_no2": iv((rt - 1.96) ** 2, (rt + 1.96) ** 2),
           "ci_raw": iv(rt - h, rt + h), "ci_se": iv((rt - 1.96 / (2 * rt)) ** 2, (rt + 1.96 / (2 * rt)) ** 2)}
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
    S = [[5.1, 3.1, 1.6], [3.1, 5.1, 1.6], [1.6, 1.6, 2.6]]
    e = [Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)]
    Se = [sum(Fraction(str(S[i][j])) * e[j] for j in range(3)) for i in range(3)]
    lam = Se[0] / e[0]
    assert all(Se[i] == lam * e[i] for i in range(3))
    tr = sum(S[i][i] for i in range(3))
    out = {f"S{i}{j}": S[i][j] for i in range(3) for j in range(3)}
    out.update({"Se0": s(Se[0] * 3, 0), "Se2": s(Se[2] * 3, 0), "lam": s(lam, 0), "tr": s(tr, 1), "ans": r(float(lam) / tr, 3),
                "ans_s": s(float(lam) / tr, 3), "ans4": s(float(lam) / tr, 4), "w_raw": s(81 / tr, 2),
                "w_sum": s(float(lam) / sum(sum(rw) for rw in S), 3), "w_diag": s(5.1 / tr, 3)})
    return out


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
