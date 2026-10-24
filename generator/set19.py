"""模擬試験 第19回（set19）の数値。values() が {大問id: {名前: 値}} を返す。

題材：銀行・与信・決済・家計。
問題文に示した分布の値（χ² 点・F 点・スチューデント化範囲の点・対数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
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
    # (a) 全確率の公式とベイズの定理
    w = {"shop": 0.30, "web": 0.50, "ally": 0.20}
    d = {"shop": 0.04, "web": 0.10, "ally": 0.06}
    joint = {k: w[k] * d[k] for k in w}
    tot = sum(joint.values())
    post = joint["web"] / tot
    # (b) ロジットの標準誤差（デルタ法）
    n, x = 400, 32
    ph = x / n
    npq = n * ph * (1 - ph)
    se_logit = 1 / sqrt(npq)
    se_p = sqrt(ph * (1 - ph) / n)
    return {
        "ws": s(w["shop"], 2), "ww": s(w["web"], 2), "wa": s(w["ally"], 2), "ds": s(d["shop"], 2), "dw": s(d["web"], 2), "da": s(d["ally"], 2),
        "js": s(joint["shop"], 3), "jw": s(joint["web"], 3), "ja": s(joint["ally"], 3), "tot": s(tot, 3),
        "post": s(post, 3), "post4": s(post, 4),
        "c_tot": s(tot, 3), "c_joint": s(joint["web"], 3), "c_cond": s(d["web"], 3), "c_prior": s(w["web"], 3),
        "n": n, "x": x, "nx": n - x, "ph": s(ph, 2), "qh": s(1 - ph, 2), "npq": s(npq, 2), "sqnpq": s(sqrt(npq), 4),
        "se_logit": r(se_logit, 3), "se_logit_s": s(se_logit, 3), "se_logit4": s(se_logit, 4),
        "se_p": s(se_p, 4), "se_nosqrt": s(1 / npq, 4), "se_pq": s(1 / (ph * (1 - ph)), 2),
    }


def p02():
    xs = [1, 0, 2, 0, 1, 3, 0, 1, 0, 2, 1, 1]
    n = len(xs)
    T = sum(xs)
    m = T / n
    sx2 = sum(v * v for v in xs)
    ss = sx2 - n * m ** 2
    D = ss / m
    chi11, chi12, chi11_lo = 19.68, 21.03, 4.575
    e_m1 = 0.3679  # e^{-1}
    umvu = (1 - 1 / n) ** T
    return {
        "data": "、".join(str(v) for v in xs), "n": n, "T": T, "m": s(m, 1), "sx2": sx2, "ss": s(ss, 1),
        "nm2": s(n * m ** 2, 1), "D": s(D, 1), "df": n - 1, "ratio": s(ss / (n - 1) / m, 2), "D_raw": s(sx2 / m, 1),
        "chi11": chi11, "chi12": chi12, "chi11_lo": chi11_lo,
        "nz": xs.count(0), "e_m1": e_m1,
        "umvu": s(umvu, 3), "umvu4": s(umvu, 4), "frac": "11/12", "base": s(1 - 1 / n, 6),
        "mle": s(e_m1, 3), "emp": s(xs.count(0) / n, 3), "w_n1": s((1 - 1 / (n - 1)) ** T, 3),
        "w_np1": s((n / (n + 1)) ** T, 3),
    }


def p03():
    lam, delta = Fraction(1, 5), Fraction(1, 20)
    k = lam / delta                      # 4
    ed = lam / (lam + delta)             # 4/5
    ed2 = lam / (lam + 2 * delta)        # 2/3
    vd = ed2 - ed ** 2                   # 2/75
    vpv = 100 ** 2 * vd
    sd = sqrt(float(vpv))
    # 誤答
    w_plus = lam / (lam - delta)         # M(+δ) を使った
    return {
        "lam": s(lam, 1), "mean": int(1 / lam), "delta": s(delta, 2), "k": int(k), "km1": int(k - 1),
        "inv": s(delta / lam, 2), "invm1": s(delta / lam - 1, 2),
        "ed": s(ed, 1), "ed2_f": "2/3", "ed_sq": "16/25", "vd_f": "2/75", "vd": s(vd, 6),
        "vpv": s(vpv, 2), "sd": r(sd, 2), "sd_s": s(sd, 2), "sd4": s(sd, 4),
        "w_var": s(float(vpv), 2),
        "w_nosq": s(sqrt(float(100 ** 2 * ed2)), 2), "w_plus": s(w_plus, 3),
        "pv_mean": s(100 * ed, 0),
    }


def p04():
    # (a) 回帰推定
    N, n = 20000, 200
    ybar, xbar, Xbar, b = 28.0, 42.0, 40.0, 0.55
    yreg = ybar + b * (Xbar - xbar)
    yratio = ybar * Xbar / xbar
    # (b) 偏相関
    rcr, rcx, rrx = 0.42, 0.70, 0.50
    num = rcr - rcx * rrx
    den = sqrt((1 - rcx ** 2) * (1 - rrx ** 2))
    pc = num / den
    # (c) 操作変数
    vx, cxc, czc, czx, vz = 100.0, 55.0, 52.0, 80.0, 100.0
    iv = czc / czx
    return {
        "N": N, "n": n, "ybar": s(ybar, 1), "xbar": s(xbar, 1), "Xbar": s(Xbar, 1), "b": b,
        "dx": s(Xbar - xbar, 1), "adj": s(b * (Xbar - xbar), 2), "yreg": r(yreg, 2), "yreg_s": s(yreg, 2),
        "yratio": s(yratio, 2), "ysign": s(ybar - b * (Xbar - xbar), 2),
        "rcr": s(rcr, 2), "rcx": s(rcx, 2), "rrx": s(rrx, 2), "num": s(num, 2), "prod": s(rcx * rrx, 2),
        "a1": s(1 - rcx ** 2, 2), "a2": s(1 - rrx ** 2, 2), "aa": s((1 - rcx ** 2) * (1 - rrx ** 2), 4),
        "den": s(den, 5), "pc": r(pc, 3), "pc_s": s(pc, 3), "pc4": s(pc, 4),
        "pc_nosq": s(num / ((1 - rcx ** 2) * (1 - rrx ** 2)), 3), "pc_semi": s(num / sqrt(1 - rrx ** 2), 3),
        "vx": s(vx, 0), "cxc": s(cxc, 0), "czc": s(czc, 0), "czx": s(czx, 0), "vz": s(vz, 0),
        "ols": s(cxc / vx, 2), "iv": s(iv, 2), "rf": s(czc / vz, 2), "fs": s(czx / vz, 2), "inv": s(czx / czc, 2),
    }


def p05():
    n = 60
    rss1, rss2, rss3 = 13.20, 11.00, 10.60
    p1, p2, p3 = 3, 5, 7
    q = p2 - p1
    F = ((rss1 - rss2) / q) / (rss2 / (n - p2))
    F_noq = (rss1 - rss2) / (rss2 / (n - p2))
    F_n = ((rss1 - rss2) / q) / (rss2 / n)
    F_small = ((rss1 - rss2) / q) / (rss1 / (n - p1))
    F_q4 = ((rss1 - rss2) / 4) / (rss2 / (n - p2))
    f_crit = 3.165   # F_{0.05}(2,55)
    # (b) 外部スチューデント化残差
    e, h = 1.20, 0.25
    s2 = rss2 / (n - p2)
    r2 = e ** 2 / (s2 * (1 - h))
    ri = sqrt(r2)
    t2 = r2 * (n - p2 - 1) / (n - p2 - r2)
    ti = sqrt(t2)
    # (c) マローズの Cp
    sig2 = rss3 / (n - p3)
    cp = [rss / sig2 - n + 2 * p for rss, p in ((rss1, p1), (rss2, p2), (rss3, p3))]
    cp_noint = [rss / sig2 - n + 2 * (p - 1) for rss, p in ((rss1, p1), (rss2, p2), (rss3, p3))]
    cp_own = [rss / (rss / (n - p)) - n + 2 * p for rss, p in ((rss1, p1), (rss2, p2), (rss3, p3))]
    cp_minus = [rss / sig2 - n - 2 * p for rss, p in ((rss1, p1), (rss2, p2), (rss3, p3))]
    sig2n = rss3 / n
    cp_n = [rss / sig2n - n + 2 * p for rss, p in ((rss1, p1), (rss2, p2), (rss3, p3))]

    def trip(v):
        return "、".join(f"M{i + 1}: {s(x, 1)}" for i, x in enumerate(v))
    return {
        "n": n, "rss1": s(rss1, 2), "rss2": s(rss2, 2), "rss3": s(rss3, 2), "p1": p1, "p2": p2, "p3": p3,
        "q": q, "dfe2": n - p2, "drss": s(rss1 - rss2, 2), "msd": s((rss1 - rss2) / q, 2), "mse2": s(s2, 2),
        "F": s(F, 2), "F_noq": s(F_noq, 2), "F_n": s(F_n, 2), "F_small": s(F_small, 2), "F_q4": s(F_q4, 2),
        "f_crit": f_crit, "mse1": s(rss1 / (n - p1), 4), "dfe1": n - p1, "mse_n": s(rss2 / n, 4),
        "e": s(e, 2), "h": s(h, 2), "omh": s(1 - h, 2), "r2": s(r2, 1), "ri": s(ri, 4),
        "nmp": n - p2, "nmp1": n - p2 - 1, "den_t": s(n - p2 - r2, 1), "t2": s(t2, 4),
        "ti": r(ti, 2), "ti_s": s(ti, 2), "ti4": s(ti, 4), "ri2": s(ri, 2),
        "t_wrong_df": s(sqrt(r2 * (n - p2) / (n - p2 - 1 - r2)), 2),
        "sig2": s(sig2, 2), "dfe3": n - p3,
        "cp": trip(cp), "cp_noint": trip(cp_noint), "cp_own": trip(cp_own), "cp_minus": trip(cp_minus),
        "cp_n": trip(cp_n), "sig2n": s(sig2n, 4),
        "cp1": s(cp[0], 1), "cp2": s(cp[1], 1), "cp3": s(cp[2], 1),
        "cpr1": s(rss1 / sig2, 1), "cpr2": s(rss2 / sig2, 1), "cpr3": s(rss3 / sig2, 1),
    }


def p06():
    bad = [42, 55, 61, 70]
    good = [58, 66, 72, 75, 81, 88]
    wins = sum((g > b) + 0.5 * (g == b) for b in bad for g in good)
    pairs = len(bad) * len(good)
    per = [sum(g > b for g in good) for b in bad]
    auc = wins / pairs
    # (b) ロジスティック回帰
    b0, b1 = 3.0, 0.08
    ln4, ln5 = 1.3863, 1.6094
    s_ok = (b0 + ln4) / b1        # logit(0.2) = -ln 4
    s_rev = (b0 - ln4) / b1       # logit(0.8) を使った
    s_lnp = (b0 + ln5) / b1       # ln(0.2) = -ln 5 を使った（オッズでなく確率の対数）
    s_noint = ln4 / b1            # 切片を忘れた
    s_lin = (b0 - 0.2) / b1       # 線形確率モデルとして解いた
    return {
        "bad": "、".join(map(str, bad)), "good": "、".join(map(str, good)), "nb": len(bad), "ng": len(good),
        "pairs": pairs, "w1": per[0], "w2": per[1], "w3": per[2], "w4": per[3], "wins": int(wins),
        "auc": r(auc, 3), "auc_s": s(auc, 3), "auc_rev": s(1 - auc, 3),
        "b0": s(b0, 1), "b1": s(b1, 2), "ln4": ln4, "ln5": ln5, "num": s(b0 + ln4, 4),
        "s_ok": s(s_ok, 1), "s_rev": s(s_rev, 1), "s_lnp": s(s_lnp, 1), "s_noint": s(s_noint, 1), "s_lin": s(s_lin, 1),
        "s_ok2": s(s_ok, 2), "or10": s(0.4493, 4),
    }


def p07():
    P = {"A": {"A": 0.90, "B": 0.08, "D": 0.02}, "B": {"A": 0.10, "B": 0.80, "D": 0.10}, "D": {"A": 0, "B": 0, "D": 1}}
    two = sum(P["A"][k] * P[k]["D"] for k in "ABD")
    return {
        "aa": s(P["A"]["A"], 2), "ab": s(P["A"]["B"], 2), "ad": s(P["A"]["D"], 2), "ba": s(P["B"]["A"], 2), "bb": s(P["B"]["B"], 2), "bd": s(P["B"]["D"], 2),
        "t1": s(P["A"]["A"] * P["A"]["D"], 3), "t2": s(P["A"]["B"] * P["B"]["D"], 3), "t3": s(P["A"]["D"], 3),
        "two": s(two, 3), "w_twice": s(2 * P["A"]["D"], 3), "w_noabs": s(two - P["A"]["D"], 3),
        "w_ind": s(1 - (1 - P["A"]["D"]) ** 2, 4), "w_sq": s(P["A"]["D"] ** 2, 4),
    }


def p08():
    N, D, n = 50, 3, 10
    p0 = comb(N - D, n) / comb(N, n)
    p1 = D * comb(N - D, n - 1) / comb(N, n)
    ans = 1 - p0
    binom = 1 - (1 - D / N) ** n
    # (b) Holm
    pv = {"A": 0.022, "B": 0.004, "C": 0.120, "D": 0.019, "E": 0.011}
    alpha, m = 0.05, len(pv)
    srt = sorted(pv.items(), key=lambda kv: kv[1])
    holm = 0
    for i, (_, p) in enumerate(srt):
        if p <= alpha / (m - i):
            holm += 1
        else:
            break
    holm_cont = sum(1 for i, (_, p) in enumerate(srt) if p <= alpha / (m - i))
    bonf = sum(1 for p in pv.values() if p <= alpha / m)
    raw = sum(1 for p in pv.values() if p <= alpha)
    dbl = sum(1 for i, (_, p) in enumerate(srt) if (m - i) * p <= alpha / m)
    assert (holm, holm_cont, bonf, raw, dbl) == (2, 3, 1, 4, 0)
    return {
        "N": N, "D": D, "n": n, "ND": N - D,
        "p0_num": 40 * 39 * 38, "p0_den": 50 * 49 * 48, "p0": s(p0, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_binom": s(binom, 3), "w_mean": s(n * D / N, 3), "w_p1": s(p1, 3), "w_p0": s(p0, 3),
        "q94": s((1 - D / N) ** n, 4),
        "pA": s(pv["A"], 3), "pB": s(pv["B"], 3), "pC": s(pv["C"], 3), "pD": s(pv["D"], 3), "pE": s(pv["E"], 3),
        "th1": s(alpha / 5, 4), "th2": s(alpha / 4, 4), "th3": s(alpha / 3, 4), "th4": s(alpha / 2, 4),
    }


def p09():
    n, s2 = 15, 0.64
    c_hi, c_lo = 26.12, 5.629      # χ²_{0.025}(14), χ²_{0.975}(14)
    c90_hi, c90_lo = 23.68, 6.571  # χ²_{0.05}(14), χ²_{0.95}(14)
    c15_hi, c15_lo = 27.49, 6.262  # χ²_{0.025}(15), χ²_{0.975}(15)
    ss = (n - 1) * s2
    lo, hi = sqrt(ss / c_hi), sqrt(ss / c_lo)

    def iv(a, b):
        return f"({s(a, 3)},\\ {s(b, 3)})"
    se_n = 1.96 * sqrt(s2) / sqrt(2 * (n - 1))
    # (b) テューキー
    k, m, mse = 3, 8, 0.50
    q, q7, t21 = 3.565, 4.165, 2.080
    hsd = q * sqrt(mse / m)
    lsd = t21 * sqrt(2 * mse / m)
    hsd2 = q * sqrt(2 * mse / m)
    hsd_nosq = q * mse / m
    hsd7 = q7 * sqrt(mse / m)
    means = {"st": 5.10, "co": 4.30, "re": 4.00}
    return {
        "n": n, "s2": s2, "df": n - 1, "ss": s(ss, 2), "c_hi": c_hi, "c_lo": c_lo, "c90_hi": c90_hi, "c90_lo": c90_lo,
        "c15_hi": c15_hi, "c15_lo": c15_lo, "c15_95": 7.261, "c15_05": "25.00",
        "v_lo": s(ss / c_hi, 4), "v_hi": s(ss / c_lo, 4),
        "ci_ok": iv(lo, hi), "ci_var": iv(ss / c_hi, ss / c_lo), "ci_norm": iv(sqrt(s2) - se_n, sqrt(s2) + se_n),
        "ci_n": iv(sqrt(n * s2 / c15_hi), sqrt(n * s2 / c15_lo)), "ci_90": iv(sqrt(ss / c90_hi), sqrt(ss / c90_lo)),
        "k": k, "m": m, "N": k * m, "mse": s(mse, 2), "dfe": k * m - k, "q": q, "q7": q7, "t21": s(t21, 3),
        "mst": s(means["st"], 2), "mco": s(means["co"], 2), "mre": s(means["re"], 2),
        "d_sc": s(means["st"] - means["co"], 2), "d_cr": s(means["co"] - means["re"], 2), "d_sr": s(means["st"] - means["re"], 2),
        "se_q": s(sqrt(mse / m), 4), "hsd": s(hsd, 2), "hsd4": s(hsd, 4), "lsd": s(lsd, 2), "hsd2": s(hsd2, 2),
        "hsd_nosq": s(hsd_nosq, 2), "hsd7": s(hsd7, 2), "msem": s(mse / m, 4),
    }


def p10():
    sat = {"A": 4.6, "B": 4.2, "C": 3.9, "D": 3.8, "E": 3.5, "F": 3.3, "G": 3.0}
    gro = {"A": 5.1, "B": 6.0, "C": 3.2, "D": 4.4, "E": 1.5, "F": 2.8, "G": 2.0}

    def rank_desc(dct):
        o = sorted(dct, key=lambda k: -dct[k])
        return {k: i + 1 for i, k in enumerate(o)}
    r1, r2 = rank_desc(sat), rank_desc(gro)
    n = len(sat)
    d2 = sum((r1[k] - r2[k]) ** 2 for k in sat)
    rs = 1 - 6 * d2 / (n * (n ** 2 - 1))
    keys = "ABCDEFG"
    out = {"n": n, "d2": d2, "den": n * (n ** 2 - 1), "six": 6 * d2, "rs": r(rs, 3), "rs_s": s(rs, 3), "rs4": s(rs, 4),
           "w_no6": s(1 - d2 / (n * (n ** 2 - 1)), 3), "w_np1": s(1 - 6 * d2 / (n * (n ** 2 + 1)), 3)}
    for k in keys:
        out[f"s{k}"] = s(sat[k], 1)
        out[f"g{k}"] = s(gro[k], 1)
        out[f"r{k}"] = r1[k]
        out[f"q{k}"] = r2[k]
        out[f"d{k}"] = r1[k] - r2[k]
    return out


def p11():
    ev = [2.70, 1.20, 0.95, 0.55, 0.40, 0.20]
    tot = sum(ev)
    cum = []
    c = 0
    for v in ev:
        c += v
        cum.append(c / tot * 100)
    # (b) RMSEA
    pvar = 6
    moments = pvar * (pvar + 1) // 2
    npar = 6 + 6 + 1
    df = moments - npar
    chi2, n = 20.0, 301
    rmsea = sqrt(max((chi2 - df) / (df * (n - 1)), 0))
    df9 = df + 1
    return {
        **{f"l{i + 1}": s(v, 2) for i, v in enumerate(ev)}, "tot": s(tot, 2),
        **{f"c{i + 1}": s(v, 2) for i, v in enumerate(cum)},
        "r1": s(ev[0] / tot * 100, 1), "r3": s(ev[2] / tot * 100, 2),
        "pvar": pvar, "moments": moments, "npar": npar, "df": df, "chi2": s(chi2, 1), "n": n, "n1": n - 1,
        "num": s(chi2 - df, 1), "den": df * (n - 1), "ratio": s((chi2 - df) / (df * (n - 1)), 4),
        "rmsea": r(rmsea, 3), "rmsea_s": s(rmsea, 3), "rmsea4": s(rmsea, 4),
        "w_df9": s(sqrt((chi2 - df9) / (df9 * (n - 1))), 3), "df9": df9,
        "w_nosq": s((chi2 - df) / (df * (n - 1)), 4), "w_chi": s(sqrt(chi2 / (df * (n - 1))), 3),
    }


def p12():
    phi, th = 0.6, 0.3
    rho1 = (1 + phi * th) * (phi + th) / (1 + 2 * phi * th + th ** 2)
    rho2 = phi * rho1
    thn = -th
    rho1n = (1 + phi * thn) * (phi + thn) / (1 + 2 * phi * thn + thn ** 2)

    def pr(a, b):
        return f"\\rho(1)={s(a, 3)},\\ \\rho(2)={s(b, 3)}"
    return {
        "phi": phi, "th": th, "g0num": s(1 + 2 * phi * th + th ** 2, 2), "g1num": s((1 + phi * th) * (phi + th), 3),
        "a1": s(1 + phi * th, 2), "a2": s(phi + th, 1), "pt2": s(2 * phi * th, 2), "th2": s(th ** 2, 2),
        "omp2": s(1 - phi ** 2, 2),
        "rho1": s(rho1, 3), "rho1_4": s(rho1, 4), "rho2": s(rho2, 3),
        "c_ok": pr(rho1, rho2), "c_ar": pr(phi, phi ** 2), "c_sum": pr(phi + th, phi * (phi + th)),
        "c_sq": pr(rho1, rho1 ** 2), "c_neg": pr(rho1n, phi * rho1n),
    }


def p13():
    # 層1: 延滞30日未満、層2: 30日以上。(解消, 未解消)
    t1 = {"sa": (80, 20), "sn": (60, 40)}
    t2 = {"sa": (20, 30), "sn": (15, 35)}

    def parts(t):
        a, b = t["sa"]
        c, d = t["sn"]
        n = a + b + c + d
        return a, b, c, d, n
    a1, b1, c1, d1, n1 = parts(t1)
    a2, b2, c2, d2, n2 = parts(t2)
    num = a1 * d1 / n1 + a2 * d2 / n2
    den = b1 * c1 / n1 + b2 * c2 / n2
    mh = num / den
    or1, or2 = a1 * d1 / (b1 * c1), a2 * d2 / (b2 * c2)
    A, B, C, D = a1 + a2, b1 + b2, c1 + c2, d1 + d2
    crude = A * D / (B * C)
    # (b) ベータ・二項
    al, be, nn, y = 2, 48, 50, 3
    ap, bp = al + y, be + nn - y
    pred = Fraction(ap * (ap + 1), (ap + bp) * (ap + bp + 1))
    return {
        "a1": a1, "b1": b1, "c1": c1, "d1": d1, "n1": n1, "a2": a2, "b2": b2, "c2": c2, "d2": d2, "n2": n2,
        "ad1": s(a1 * d1 / n1, 1), "bc1": s(b1 * c1 / n1, 1), "ad2": s(a2 * d2 / n2, 1), "bc2": s(b2 * c2 / n2, 1),
        "num": s(num, 1), "den": s(den, 1), "mh": s(mh, 2), "mh4": s(mh, 4),
        "or1": s(or1, 2), "or2": s(or2, 2), "crude": s(crude, 2), "avg": s((or1 + or2) / 2, 2), "inv": s(1 / mh, 2),
        "A": A, "B": B, "C": C, "D": D,
        "al": al, "be": be, "nn": nn, "y": y, "ny": nn - y, "ap": ap, "bp": bp, "apb": ap + bp,
        "pmean": s(Fraction(ap, ap + bp), 2), "ap1": ap + 1, "apb1": ap + bp + 1,
        "pnum": ap * (ap + 1), "pden": (ap + bp) * (ap + bp + 1),
        "pred": s(pred, 5), "pred6": s(pred, 6), "w_plug": s(Fraction(ap, ap + bp) ** 2, 5),
        "w_prior": s(Fraction(al, al + be) ** 2, 5), "w_mle": s(Fraction(y, nn) ** 2, 5),
        "w_bad": s(Fraction(ap * (ap + 1), (ap + be + nn) * (ap + be + nn + 1)), 5),
        "bad_b": be + nn,
    }


def values():
    return {
        "s19p01": p01(), "s19p02": p02(), "s19p03": p03(), "s19p04": p04(), "s19p05": p05(), "s19p06": p06(),
        "s19p07": p07(), "s19p08": p08(), "s19p09": p09(), "s19p10": p10(), "s19p11": p11(), "s19p12": p12(),
        "s19p13": p13(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
