"""模擬試験 第25回（set25）の数値。values() が {大問id: {名前: 値}} を返す。

題材は製薬の品質試験・バイオ実験・研究所。問題文に示した分布の値（Φ・t・χ² の点、e^x、ln x など）は、
ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。負の数は − ではなく - のまま。"""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def m(x, d):
    """数式用：負の数のマイナスを残した文字列（s と同じ）。"""
    return s(x, d)


def p01():
    W, W0 = 96, 58
    W1 = W - W0
    ln_ratio = -0.5039                      # 問題文に示す ln(58/96)
    lam = -ln_ratio
    p1_cond = lam * W0 / W1                 # λe^{-λ}/(1-e^{-λ})、e^{-λ}=W0/W
    p1_unc = lam * W0 / W                   # 条件づけ忘れ
    # (b) ゼロ切断ポアソン
    cnt = {1: 28, 2: 16, 3: 5, 4: 1}
    n = sum(cnt.values())
    tot = sum(k * v for k, v in cnt.items())
    xbar = tot / n
    g = {0.8: 1.4528, 0.9: 1.5166, 1.0: 1.5820, 1.1: 1.6489, 1.2: 1.7172}   # λ/(1-e^{-λ})（小数第4位）
    lam_hat = min(g, key=lambda l: abs(g[l] - xbar))
    assert lam_hat == 1.0
    # 誤答：λ(1-e^{-λ})=x̄ の解（二分法）と、固定点反復を1回で止めた値
    lo, hi = 1.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid * (1 - math.exp(-mid)) < xbar:
            lo = mid
        else:
            hi = mid
    w_inv = (lo + hi) / 2
    w_iter = xbar * (1 - math.exp(-xbar))
    return {
        "W": W, "W0": W0, "W1": W1, "lnr": s(ln_ratio, 4), "lam": s(lam, 4),
        "e0": f"{W0}/{W}", "p1c": r(p1_cond, 3), "p1c_s": s(p1_cond, 3), "p1c4": s(p1_cond, 4),
        "p1u": s(p1_unc, 3), "lamfrac": s(lam * W0 / W, 4),
        "n1": cnt[1], "n2": cnt[2], "n3": cnt[3], "n4": cnt[4], "n": n, "tot": tot, "xbar": s(xbar, 2),
        "g08": s(g[0.8], 4), "g09": s(g[0.9], 4), "g10": s(g[1.0], 4), "g11": s(g[1.1], 4), "g12": s(g[1.2], 4),
        "c_shift": s(xbar - 1, 2), "c_ok": s(lam_hat, 2), "c_iter": s(w_iter, 2), "c_naive": s(xbar, 2),
        "c_inv": s(w_inv, 2),
    }


def p02():
    # (a) 丸め誤差の和
    mm, h, thr = 48, 0.1, 0.1
    var1 = h ** 2 / 12
    varS = mm * var1
    sdS = sqrt(varS)
    z = r(thr / sdS, 2)
    phi = {0.50: 0.6915, 1.00: 0.8413, 2.50: 0.9938}
    assert z in phi
    ans_a = 2 * (1 - phi[z])
    z_half = r(thr / sqrt(mm * (h / 2) ** 2 / 12), 2)       # 幅を 0.05 とした誤り
    assert z_half in phi
    # (b) 偏りの異なる2台の天秤（全分散の公式）
    w1, w2, b1, b2, sd1, sd2 = 0.6, 0.4, 30.0, -20.0, 20.0, 40.0
    mean = w1 * b1 + w2 * b2
    ev = w1 * sd1 ** 2 + w2 * sd2 ** 2
    ve = w1 * w2 * (b1 - b2) ** 2
    ve2 = w1 * (b1 - mean) ** 2 + w2 * (b2 - mean) ** 2
    assert abs(ve - ve2) < 1e-9
    tot = ev + ve
    ex2 = w1 * (sd1 ** 2 + b1 ** 2) + w2 * (sd2 ** 2 + b2 ** 2)
    # (c) ジャック・ベラ
    n, sk, ku = 120, 0.30, 4.50
    jb = n / 6 * (sk ** 2 + (ku - 3) ** 2 / 4)
    jb_n6 = sk ** 2 + (ku - 3) ** 2 / 4
    jb_no4 = n / 6 * (sk ** 2 + (ku - 3) ** 2)
    jb_no3 = n / 6 * (sk ** 2 + ku ** 2 / 4)
    return {
        "mm": mm, "h": h, "hh": s(h / 2, 2), "thr": thr, "var1": s(var1, 6), "varS": s(varS, 2), "sdS": s(sdS, 1),
        "z": s(z, 2), "phi050": s(phi[0.50], 4), "phi100": s(phi[1.00], 4), "phi250": s(phi[2.50], 4),
        "tail": s(1 - phi[z], 4), "ans_a": r(ans_a, 3), "ans_a_s": s(ans_a, 3),
        "one_side": s(1 - phi[z], 3), "w_half": s(2 * (1 - phi[z_half]), 3), "z_half": s(z_half, 2),
        "w1": w1, "w2": w2, "b1": s(b1, 0), "b2": s(b2, 0), "sd1": s(sd1, 0), "sd2": s(sd2, 0),
        "v1": s(sd1 ** 2, 0), "v2": s(sd2 ** 2, 0), "mean": s(mean, 0), "ev": s(ev, 0), "ve": s(ve, 0),
        "bd": s(b1 - b2, 0), "bd2": s((b1 - b2) ** 2, 0), "dev1": s(b1 - mean, 0), "dev2": s(b2 - mean, 0),
        "tot": s(tot, 0), "sdT": s(sqrt(tot), 1), "sdT4": s(sqrt(tot), 3),
        "w_ev": s(sqrt(ev), 1), "w_ve": s(sqrt(ve), 1), "w_avg": s(w1 * sd1 + w2 * sd2, 1), "w_ex2": s(sqrt(ex2), 1),
        "ex2": s(ex2, 0),
        "n": n, "sk": s(sk, 2), "ku": s(ku, 2), "n6": s(n / 6, 0), "sk2": s(sk ** 2, 2),
        "ku3": s((ku - 3) ** 2 / 4, 4), "ku3sq": s((ku - 3) ** 2, 2), "inner": s(jb_n6, 4),
        "jb": s(jb, 2), "jb_n6": s(jb_n6, 2), "jb_no4": s(jb_no4, 2), "jb_no3": s(jb_no3, 2),
        "chi2": "5.991", "chi1": "3.841",
    }


def p03():
    lam, c = 0.1, 24
    lc = lam * c
    e24, e12 = 0.0907, 0.3012
    ans = 1 - (1 - e24) / lc
    w_mean = 1 - e12                       # Y を平均 12 に置き換えた
    w_end = 1 - e24                        # Y を 24 に置き換えた
    w_exp = lam / (lam + 1 / 12)           # Y を平均 12 の指数分布とした
    # (b) 対数正規の変動係数
    s2, e016 = 0.16, 1.1735
    cv = sqrt(e016 - 1)
    return {
        "lam": lam, "mean": s(1 / lam, 0), "c": c, "lc": s(lc, 1), "e24": s(e24, 4), "e12": s(e12, 4),
        "one_e": s(1 - e24, 4), "frac": s((1 - e24) / lc, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_comp": s(1 - ans, 3), "w_mean": s(w_mean, 3), "w_end": s(w_end, 3), "w_exp": s(w_exp, 3),
        "s2": s2, "sd": s(sqrt(s2), 1), "e016": s(e016, 4), "cv2": s(e016 - 1, 4), "cv": s(cv, 3), "cv4": s(cv, 4),
        "w_sqrt": s(sqrt(e016), 3), "c_s2": s(s2, 3), "c_sd": s(sqrt(s2), 3), "cvpct": s(cv * 100, 1),
    }


def p04():
    k = 4
    terms = [(-1) ** j / math.factorial(j) for j in range(k + 1)]
    p0 = sum(terms)
    n_der = round(p0 * math.factorial(k))
    p1 = n_der_1 = 8 / 24                  # ちょうど1本正しい：4×D3/4! = 4×2/24
    assert abs(p1 - k * 2 / math.factorial(k)) < 1e-12
    return {
        "k": k, "kf": math.factorial(k), "nder": n_der, "ans": s(p0, 3), "ans_frac": f"{n_der}/{math.factorial(k)}",
        "w_ind": s((3 / 4) ** 4, 3), "w_e": s(math.exp(-1), 3), "w_one": s(n_der_1, 3), "w_atl": s(1 - p0, 3),
        "S1": "1", "S2": s(1 / 2, 1), "S3": s(1 / 6, 4), "S4": s(1 / 24, 4),
    }


def p09():
    n = 12
    rss = {1: 0.0480, 2: 0.0210, 3: 0.0190}
    lnv = {1: -5.5215, 2: -6.3481, 3: -6.4482}         # ln(RSS/n)（小数第4位）
    for kdeg in rss:
        assert abs(math.log(rss[kdeg] / n) - lnv[kdeg]) < 6e-5
    k = {d: d + 2 for d in rss}                       # 回帰係数 d+1 個と誤差分散
    aic = {d: n * lnv[d] + 2 * k[d] for d in rss}
    best = min(aic, key=aic.get)
    assert best == 2
    aic_k = {d: n * lnv[d] + k[d] for d in rss}      # 罰則を 2 倍し忘れ
    assert min(aic_k, key=aic_k.get) == 3
    aic_nov = n * lnv[2] + 2 * (k[2] - 1)
    # (c) VIF
    xs = [2, 4, 6, 8, 10, 12] * 2
    mx = sum(xs) / len(xs)
    x2 = [x * x for x in xs]
    mx2 = sum(x2) / len(x2)
    sxy = sum((a - mx) * (b - mx2) for a, b in zip(xs, x2))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - mx2) ** 2 for b in x2)
    rr = r(sxy / sqrt(sxx * syy), 3)
    vif = 1 / (1 - rr ** 2)
    return {
        "n": n, "rss1": s(rss[1], 4), "rss2": s(rss[2], 4), "rss3": s(rss[3], 4),
        "q1": s(rss[1] / n, 4), "q2": s(rss[2] / n, 5), "q3": s(rss[3] / n, 7),
        "l1": s(lnv[1], 4), "l2": s(lnv[2], 4), "l3": s(lnv[3], 4),
        "nl1": s(n * lnv[1], 3), "nl2": s(n * lnv[2], 4), "nl3": s(n * lnv[3], 4),
        "k1": k[1], "k2": k[2], "k3": k[3], "pen1": 2 * k[1], "pen2": 2 * k[2], "pen3": 2 * k[3],
        "a1": s(aic[1], 2), "a2": s(aic[2], 2), "a3": s(aic[3], 2), "a3k": s(aic_k[3], 2), "a2k": s(aic_k[2], 2),
        "a2nov": s(aic_nov, 2),
        "rr": s(rr, 3), "rr2": s(rr ** 2, 6), "one_r2": s(1 - rr ** 2, 6), "vif": s(vif, 1), "vif2": s(vif, 2),
        "w_1r": s(1 / (1 - rr), 1), "w_odds": s(rr ** 2 / (1 - rr ** 2), 1), "mx": s(mx, 0),
    }


def p14():
    r12, r13, r23 = 0.6, 0.5, 0.3
    num = r12 ** 2 + r13 ** 2 - 2 * r12 * r13 * r23
    den = 1 - r23 ** 2
    smc = num / den
    # (b) 正準相関の逐次検定（バートレットのカイ二乗近似）
    N, p, q, c1, c2 = 60, 2, 2, 0.60, 0.20
    ln = {0.64: -0.4463, 0.96: -0.0408, 0.4: -0.9163, 0.8: -0.2231}
    for k, v in ln.items():
        assert abs(math.log(k) - v) < 1e-4
    fac = N - 1 - (p + q + 1) / 2
    st1 = -fac * (ln[0.64] + ln[0.96])
    st2 = -fac * ln[0.96]
    st1_r = -fac * (ln[0.4] + ln[0.8])
    st2_r = -fac * ln[0.8]
    st1_n = -N * (ln[0.64] + ln[0.96])
    assert st1 > 9.488 and st2 < 3.841 and st1_r > 9.488 and st2_r > 3.841
    return {
        "r12": r12, "r13": r13, "r23": r23, "num": s(num, 2), "den": s(den, 2), "smc": s(smc, 3), "smc4": s(smc, 4),
        "cross": s(2 * r12 * r13 * r23, 2), "sq": s(r12 ** 2 + r13 ** 2, 2),
        "w_sum": s(r12 ** 2 + r13 ** 2, 3), "w_max": s(r12 ** 2, 3), "w_num": s(num, 3), "w_uniq": s(1 - smc, 3),
        "N": N, "p": p, "q": q, "c1": s(c1, 2), "c2": s(c2, 2), "c1s": s(1 - c1 ** 2, 2), "c2s": s(1 - c2 ** 2, 2),
        "l064": s(ln[0.64], 4), "l096": s(ln[0.96], 4), "l04": s(ln[0.4], 4), "l08": s(ln[0.8], 4),
        "fac": s(fac, 1), "lsum": s(ln[0.64] + ln[0.96], 4), "st1": s(st1, 2), "st2": s(st2, 2),
        "st1r": s(st1_r, 2), "st2r": s(st2_r, 2), "st1n": s(st1_n, 2), "df1": p * q, "df2": (p - 1) * (q - 1),
    }


# ---- 2026-10-24 夕方：公開済みのセットと同じ型だった 16 問を作り直した分
def n02b():
    v1, v2 = 40, 60
    hm = 2 / (1 / v1 + 1 / v2)
    return {"v1": v1, "v2": v2, "inv1": s(1 / v1, 4), "inv2": s(1 / v2, 4), "hm": s(hm, 1), "hm4": s(hm, 4), "am": s((v1 + v2) / 2, 1),
            "gm": s(sqrt(v1 * v2), 1), "w_inv": s((60 / v1 + 60 / v2) / 2, 2), "w_sum": s(v1 + v2, 1)}


def n02c():
    n, r_, r0 = 28, 0.80, 0.90
    z1, z0 = 1.0986, 1.4722
    d = z1 - z0
    zs = d * sqrt(n - 3)
    assert zs < -1.645 and zs > -1.96
    return {"n": n, "r": s(r_, 2), "r0": s(r0, 2), "z1": s(z1, 4), "z0": s(z0, 4), "nm3": n - 3, "rt": s(sqrt(n - 3), 0), "d": s(d, 4),
            "zstat": s(zs, 2), "zstat4": s(zs, 4), "w_n": s(d * sqrt(n), 2), "w_wald": s((r_ - r0) / ((1 - r0 ** 2) / sqrt(n)), 2)}


def n03b():
    ln10 = 2.3026
    ey = 9 / ln10
    return {"ey": s(ey, 2), "ey4": s(ey, 4), "g": s(3.1623, 2), "w_mid": "5.50", "w_ln": "9.00"}


def p05():
    a, b, c, d = 18, 42, 8, 52
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    n = r1 + r2
    O = [[a, b], [c, d]]; E = [[r1 * c1 / n, r1 * c2 / n], [r2 * c1 / n, r2 * c2 / n]]
    lg = [[r(math.log(O[i][j] / E[i][j]), 4) for j in range(2)] for i in range(2)]
    G = 2 * sum(O[i][j] * lg[i][j] for i in range(2) for j in range(2))
    X = sum((O[i][j] - E[i][j]) ** 2 / E[i][j] for i in range(2) for j in range(2))
    assert G > 3.841
    num = a * d - b * c; den = sqrt(r1 * r2 * c1 * c2); phi = num / den
    out = {"a": a, "b": b, "c": c, "d": d, "r1": r1, "r2": r2, "c1": c1, "c2": c2, "n": n,
           "e11": s(E[0][0], 0), "e12": s(E[0][1], 0), "e21": s(E[1][0], 0), "e22": s(E[1][1], 0),
           "l11": s(lg[0][0], 4), "l12": s(lg[0][1], 4), "l21": s(lg[1][0], 4), "l22": s(lg[1][1], 4),
           "G": s(G, 2), "G4": s(G, 4), "w_half": s(G / 2, 2), "w_abs": s(2 * sum(abs(O[i][j] * lg[i][j]) for i in range(2) for j in range(2)), 2), "X": s(X, 2),
           "num": num, "den": s(den, 1), "phi": s(phi, 3), "phi4": s(phi, 4), "nphi2": s(n * phi * phi, 2), "nphi24": s(n * phi * phi, 4),
           "w_phi2": s(phi * phi, 3), "w_G": s(G, 2), "w_or": s(a * d / (b * c), 3), "w_neg": s(-phi, 3)}
    # (c) ベータ・二項
    al, be, nn, k = 2, 8, 20, 3
    A, B = al + k, be + nn - k
    out.update({"al": al, "be": be, "nn": nn, "k": k, "A": A, "B": B, "pm": s(A / (A + B), 3), "pm4": s(A / (A + B), 4),
                "map": s((A - 1) / (A + B - 2), 3), "map4": s((A - 1) / (A + B - 2), 4), "mle": s(k / nn, 3), "prior": s(al / (al + be), 3),
                "w_add": s((al + k) / (al + k + be + nn), 3)})
    assert len({out["pm"], out["map"], out["mle"], out["prior"], out["w_add"]}) == 5
    return out


def p06():
    lm, ls, n = 4.605, 0.10, 9
    t8, t9 = 2.306, 2.262
    se = ls / sqrt(n)
    hw = t8 * se
    hwz = 1.96 * se
    em, ep, emz, epz = 0.9260, 1.0799, 0.9368, 1.0675
    assert abs(math.exp(-r(hw, 4)) - em) < 5e-5 and abs(math.exp(r(hw, 4)) - ep) < 5e-5
    assert abs(math.exp(-r(hwz, 4)) - emz) < 5e-5 and abs(math.exp(r(hwz, 4)) - epz) < 5e-5
    return {"lm": s(lm, 3), "ls": s(ls, 2), "t8": t8, "t9": t9, "se": s(se, 4), "hw": s(hw, 4), "hw3": s(hw, 4), "hwz": s(hwz, 4),
            "em": s(em, 4), "ep": s(ep, 4), "emz": s(emz, 4), "epz": s(epz, 4),
            "lo": s(100 * em, 1), "hi": s(100 * ep, 1), "w_s_lo": s(100 - 100 * hw, 1), "w_s_hi": s(100 + 100 * hw, 1),
            "w_z_lo": s(100 * emz, 1), "w_z_hi": s(100 * epz, 1)}


def p07():
    n, l0, l1 = 3, 2, 4
    from math import exp, factorial
    sf = lambda m_, t: 1 - sum(exp(-m_) * m_ ** k / factorial(k) for k in range(t))
    out = {"n": n, "l0": l0, "l1": l1}
    for t in (9, 10, 11, 12):
        out[f"p0_{t}"] = s(sf(n * l0, t), 4); out[f"p1_{t}"] = s(sf(n * l1, t), 4)
    c = next(t for t in range(1, 40) if sf(n * l0, t) <= 0.05)
    assert c == 11
    out.update({"pow": s(sf(n * l1, c), 3), "w10": s(sf(n * l1, 10), 3), "w12": s(sf(n * l1, 12), 3),
                "size": s(sf(n * l0, c), 3), "beta": s(1 - sf(n * l1, c), 3)})
    return out


def p08():
    a, b = 0.6, 0.2
    p0 = b / (a + b)
    out = {"a": s(a, 1), "b": s(b, 1), "q": s(1 - a, 1), "p0": s(p0, 2), "rec": s(1 / p0, 2), "soj": s(1 / b, 2),
           "w_rec1": s((a + b) / a, 2), "w_soj": s(1 / a, 2), "w_cyc": s(1 / a + 1 / b, 2), "w_soj2": s(1 / (1 - b), 2)}
    assert abs((1 - a) * 1 + a * (1 + 1 / b) - 1 / p0) < 1e-12
    phi, se = 0.8, 0.3
    var = se * se / (1 - phi * phi)
    out.update({"phi": s(phi, 1), "se": s(se, 2), "se2": s(se * se, 2), "om": s(1 - phi * phi, 2), "var": s(var, 2), "sd": s(sqrt(var), 2),
                "w_sd": s(se / sqrt(1 - phi), 2), **{f"r{k}": s(phi ** k, 4) for k in range(1, 5)}})
    assert phi ** 3 > 0.5 > phi ** 4
    return out


def n09a():
    h6, h12 = 0.090, 0.262        # 濃度 2〜12 を 2 回ずつ測った計画で、1/12+(x−7)²/140 を丸めた値
    assert abs(1 / 12 + 1 / 140 - h6) < 5e-4 and abs(1 / 12 + 25 / 140 - h12) < 5e-4
    pts = {"A": (2.5, h6), "B": (1.6, h12), "C": (0.5, h12)}
    D = {k: rr * rr / 2 * h / (1 - h) for k, (rr, h) in pts.items()}
    assert max(D, key=D.get) == "B"
    out = {}
    for k, (rr, h) in pts.items():
        out["r" + k] = s(rr, 1); out["h" + k] = s(h, 3); out["d" + k] = s(D[k], 2); out["d" + k + "4"] = s(D[k], 4)
    out.update({"w_A": s(D["A"] * 2, 2), "w_B": s(D["B"] * 2, 2)})
    return out


def p10():
    return {}


def p11():
    n = [4, 6, 5]; m_ = [12.0, 15.0, 13.0]; mse = 4.0
    N = sum(n); gm = sum(a * b for a, b in zip(n, m_)) / N
    ssb = sum(a * (b - gm) ** 2 for a, b in zip(n, m_))
    F = ssb / 2 / mse
    um = sum(m_) / 3
    out = {"n1": n[0], "n2": n[1], "n3": n[2], "m1": s(m_[0], 1), "m2": s(m_[1], 1), "m3": s(m_[2], 1), "mse": s(mse, 1),
           "dfe": N - 3, "fc": 3.885, "N": N, "gm": s(gm, 4), "ssb": s(ssb, 4), "F": s(F, 2), "F4": s(F, 4),
           "w_unw": s(sum(a * (b - um) ** 2 for a, b in zip(n, m_)) / 2 / mse, 2), "w_df": s(ssb / 3 / mse, 2),
           "w_nodf": s(ssb / mse, 2), "w_eq": s(sum(5 * (b - um) ** 2 for b in m_) / 2 / mse, 2)}
    assert F < 3.885 and ssb / mse > 3.885
    assert len({out["F"], out["w_unw"], out["w_df"], out["w_nodf"], out["w_eq"]}) == 5
    # (b) ポアソン回帰
    ya, yb = 6.0, 4.5
    lna, lnb = 1.7918, 1.5041
    b1 = lnb - lna
    out.update({"m": 10, "ya": s(ya, 1), "yb": s(yb, 1), "lna": s(lna, 4), "lnb": s(lnb, 4), "b1": r(b1, 3), "b1_s": s(b1, 3), "b14": s(b1, 4),
                "rr": s(yb / ya, 2), "diff": s(yb - ya, 1)})
    assert abs(math.log(ya) - lna) < 5e-5 and abs(math.log(yb) - lnb) < 5e-5
    return out


def p12():
    x = [100, 200, 300, 400]; X = sum(x)
    y2, y4 = 30, 70
    pi2, pi4 = 2 * x[1] / X, 2 * x[3] / X
    ht = y2 / pi2 + y4 / pi4
    return {"x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "X": X, "y2": y2, "y4": y4, "pi2": s(pi2, 1), "pi4": s(pi4, 1),
            "t2": s(y2 / pi2, 1), "t4": s(y4 / pi4, 1), "ht": r(ht, 1), "ht_s": s(ht, 1),
            "w_once": s(y2 / (x[1] / X) + y4 / (x[3] / X), 1), "w_srs": s(4 * (y2 + y4) / 2, 1)}


def p13():
    P = [(1, 1), (2, 1), (1, 2), (5, 4), (6, 5), (4, 5)]
    c = [(1, 1), (2, 1)]
    d2 = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
    lab = [min(range(2), key=lambda k: d2(p, c[k])) for p in P]
    cen = lambda lb: [tuple(sum(P[i][j] for i in range(6) if lb[i] == k) / lb.count(k) for j in range(2)) for k in range(2)]
    wss = lambda cc, lb: sum(d2(P[i], cc[lb[i]]) for i in range(6))
    c1 = cen(lab)
    lab2 = [min(range(2), key=lambda k: d2(p, c1[k])) for p in P]
    c2 = cen(lab2)
    assert lab == [0, 1, 0, 1, 1, 1] and lab2 == [0, 0, 0, 1, 1, 1]
    other = [P[i] for i in (3, 4, 5)]
    wu = sum(p[0] for p in other) / 3; wv = sum(p[1] for p in other) / 3
    return {"c2u": s(c1[1][0], 2), "c2v": s(c1[1][1], 2), "d2": s(d2(P[1], c1[1]), 3), "w_u": s(wu, 2), "w_v": s(wv, 2),
            "w0": s(wss(c, lab), 2), "w1": s(wss(c1, lab), 2), "w2": s(wss(c2, lab2), 2)}


def values():
    v = {"s25p01": p01(), "s25p02": p02(), "s25p03": p03(), "s25p04": p04(), "s25p05": p05(), "s25p06": p06(), "s25p07": p07(),
         "s25p08": p08(), "s25p09": p09(), "s25p10": p10(), "s25p11": p11(), "s25p12": p12(), "s25p13": p13(), "s25p14": p14()}
    v["s25p02"].update({"nb_" + k: x for k, x in n02b().items()}); v["s25p02"].update({"nc_" + k: x for k, x in n02c().items()})
    v["s25p03"].update({"nd_" + k: x for k, x in n03b().items()}); v["s25p09"].update({"ne_" + k: x for k, x in n09a().items()})
    return v


if __name__ == "__main__":
    import pprint
    pprint.pprint(values())
