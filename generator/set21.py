"""模擬試験 第21回（set21）の数値。values() が {大問id: {名前: 値}} を返す。

題材：学校・入試・学習アプリ・図書館。
問題文に示した分布の値（χ²・F・t・q の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
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
    # (a) 知っていたか当て推量か（ベイズの定理）
    pk, slip_ok, k = 0.6, 0.95, 4
    g = 1 / k
    num = pk * slip_ok
    den = num + (1 - pk) * g
    post = num / den
    noslip = pk / (pk + (1 - pk) * g)
    # (b) ベータ事前分布と2問続けて正解する事後予測確率
    a, b, n, y = 3, 2, 8, 6
    A, B = a + y, b + n - y
    pred = A * (A + 1) / ((A + B) * (A + B + 1))
    pm = A / (A + B)
    return {
        "pk": pk, "pk1": s(1 - pk, 1), "ok": slip_ok, "k": k, "g": s(g, 2),
        "num": s(num, 2), "gq": s((1 - pk) * g, 2), "den": s(den, 2),
        "post": r(post, 3), "post_s": s(post, 3), "post5": s(post, 5), "noslip": s(noslip, 3),
        "a": a, "b": b, "n": n, "y": y, "ny": n - y, "A": A, "B": B, "AB": A + B, "A1": A + 1, "AB1": A + B + 1,
        "pnum": A * (A + 1), "pden": (A + B) * (A + B + 1),
        "pred": s(pred, 3), "pred4": s(pred, 4), "pm": s(pm, 4), "pm2": s(pm ** 2, 3),
        "mle2": s((y / n) ** 2, 3), "prior2": s(a * (a + 1) / ((a + b) * (a + b + 1)), 3), "pm3": s(pm, 3),
    }


def p02():
    n, T = 40, 28
    ph = T / n
    pq = ph * (1 - ph)
    # (a) p(1-p) の不偏推定量
    unb = T * (n - T) / (n * (n - 1))
    # (b) デルタ法：ロジットの標準誤差
    npq = n * pq
    se_logit = 1 / sqrt(npq)
    # (c) ウィルソンの区間
    z, z90 = 1.96, 1.645

    def wilson(zz):
        c = (T + zz ** 2 / 2) / (n + zz ** 2)
        h = zz * sqrt(pq / n + zz ** 2 / (4 * n ** 2)) / (1 + zz ** 2 / n)
        return c, h
    c, h = wilson(z)
    c90, h90 = wilson(z90)
    se = sqrt(pq / n)
    wald = (ph - z * se, ph + z * se)
    cc = (ph - z * se - 1 / (2 * n), ph + z * se + 1 / (2 * n))
    nc = (ph - h, ph + h)
    z2 = z ** 2
    return {
        "n": n, "T": T, "nT": n - T, "ph": s(ph, 1), "qh": s(1 - ph, 1), "pq": s(pq, 2), "n1": n - 1,
        "Tnum": T * (n - T), "uden": n * (n - 1),
        "unb": s(unb, 4), "unb5": s(unb, 5), "mle": s(pq, 4), "shr": s(pq * (n - 1) / n, 4),
        "vph": s(pq / n, 4), "both": s(T * (n - T) / (n - 1) ** 2, 4),
        "npq": s(npq, 1), "se_logit": r(se_logit, 3), "se_logit_s": s(se_logit, 3), "se_logit5": s(se_logit, 5),
        "se_p": s(se, 4), "se_inv": s(sqrt(npq), 3),
        "z": z, "z90": z90, "z2": s(z2, 4), "z2h": s(z2 / 2, 4),
        "cnum": s(T + z2 / 2, 4), "cden": s(n + z2, 4), "cen": s(c, 4),
        "t1": s(pq / n, 5), "t2": s(z2 / (4 * n ** 2), 8), "tsum": s(pq / n + z2 / (4 * n ** 2), 8),
        "tsq": s(sqrt(pq / n + z2 / (4 * n ** 2)), 5), "hden": s(1 + z2 / n, 5), "half": s(h, 4),
        "wl": s(c - h, 3), "wu": s(c + h, 3),
        "dl": s(wald[0], 3), "du": s(wald[1], 3), "ccl": s(cc[0], 3), "ccu": s(cc[1], 3),
        "ncl": s(nc[0], 3), "ncu": s(nc[1], 3), "w90l": s(c90 - h90, 3), "w90u": s(c90 + h90, 3),
    }


def p03():
    m1, m2, m3, t = 20, 30, 60, 15
    rate = 1 / m1 + 1 / m2 + 1 / m3  # 0.1
    e = {0.25: 0.7788, 0.5: 0.6065, 0.75: 0.4724, 1.5: 0.2231}
    ans = e[1.5]
    d_max = 1 - (1 - e[0.75]) * (1 - e[0.5]) * (1 - e[0.25])
    return {
        "m1": m1, "m2": m2, "m3": m3, "t": t, "rate": s(rate, 1), "mean": s(1 / rate, 0),
        "e025": s(e[0.25], 4), "e05": s(e[0.5], 4), "e075": s(e[0.75], 4), "e15": s(e[1.5], 4),
        "ans": s(ans, 3), "d_short": s(e[0.75], 3), "d_mid": s(e[0.5], 3), "d_cmp": s(1 - ans, 3),
        "d_max": s(d_max, 3), "q1": s(1 - e[0.75], 4), "q2": s(1 - e[0.5], 4), "q3": s(1 - e[0.25], 4),
        "qprod": s((1 - e[0.75]) * (1 - e[0.5]) * (1 - e[0.25]), 4),
    }


def p04():
    lam, p = 120, 0.05
    mu = lam * p
    v_bin = lam * p * (1 - p)
    v_rev = p * (lam + lam ** 2) - (p * lam) ** 2
    v_prod = lam + p * (1 - p)
    data = [2, 9, 4, 11, 5, 1, 8, 6, 10, 4]
    nd = len(data)
    xbar = sum(data) / nd
    ss = sum((x - xbar) ** 2 for x in data)
    D = ss / xbar
    s2 = ss / (nd - 1)
    ratio = s2 / xbar
    nD = nd * s2 / xbar
    c9, c10 = 16.92, 18.31
    assert c9 < D < c10 and nD > c10 and ratio < c9
    return {
        "lam": lam, "p": p, "q": s(1 - p, 2), "mu": s(mu, 0), "v_bin": s(v_bin, 1), "v_rev": s(v_rev, 0),
        "v_prod": s(v_prod, 2), "Ebv": s(p * (1 - p), 4), "t1": s(lam * p * (1 - p), 1), "t2": s(lam * p ** 2, 1),
        "data": "、".join(str(x) for x in data), "nd": nd, "sumx": sum(data), "xbar": s(xbar, 0),
        "ss": s(ss, 0), "D": s(D, 2), "s2": s(s2, 2), "ratio": s(ratio, 2), "nD": s(nD, 2),
        "c9": c9, "c10": c10, "df": nd - 1,
    }


def p05():
    # 学校 A（少人数）：受講 7 人中 6 人合格、非受講 7 人中 2 人合格
    a1, b1, c1, d1 = 6, 1, 2, 5
    n1 = a1 + b1 + c1 + d1
    K = a1 + c1        # 合格者数
    m = a1 + b1        # 受講者数
    tot = comb(n1, m)
    probs = {x: comb(K, x) * comb(n1 - K, m - x) for x in range(0, m + 1)}
    p_ge6 = (probs[6] + probs[7]) / tot
    p_eq6 = probs[6] / tot
    p_ge5 = (probs[5] + probs[6] + probs[7]) / tot
    p_two = sum(v for v in probs.values() if v <= probs[6]) / tot
    # 学校 B
    a2, b2, c2, d2 = 18, 12, 16, 24
    n2 = a2 + b2 + c2 + d2
    num = a1 * d1 / n1 + a2 * d2 / n2
    den = b1 * c1 / n1 + b2 * c2 / n2
    mh = num / den
    crude = (a1 + a2) * (d1 + d2) / ((b1 + b2) * (c1 + c2))
    return {
        "a1": a1, "b1": b1, "c1": c1, "d1": d1, "n1": n1, "K": K, "m": m, "nK": n1 - K,
        "a2": a2, "b2": b2, "c2": c2, "d2": d2, "n2": n2, "m2": a2 + b2, "u2": c2 + d2, "u1": c1 + d1,
        "tot": tot, "C86": comb(8, 6), "C61": comb(6, 1), "C87": comb(8, 7), "C60": comb(6, 0),
        "P6n": probs[6], "P7n": probs[7], "Pn": probs[6] + probs[7],
        "p": s(p_ge6, 3), "p4": s(p_ge6, 4), "d_eq": s(p_eq6, 3), "d_two": s(p_two, 3),
        "d_ge5": s(p_ge5, 3), "d_cmp": s(1 - p_ge6, 3), "P5n": probs[5], "Pn5": probs[5] + probs[6] + probs[7],
        "ad1": s(a1 * d1 / n1, 4), "ad2": s(a2 * d2 / n2, 4), "bc1": s(b1 * c1 / n1, 4), "bc2": s(b2 * c2 / n2, 4),
        "num": s(num, 4), "den": s(den, 4), "mh": r(mh, 2), "mh_s": s(mh, 2), "mh4": s(mh, 4),
        "crude": s(crude, 2), "orA": s(a1 * d1 / (b1 * c1), 2), "orB": s(a2 * d2 / (b2 * c2), 2),
        "orAvg": s((a1 * d1 / (b1 * c1) + a2 * d2 / (b2 * c2)) / 2, 2),
        "adA": a1 * d1, "bcA": b1 * c1, "adB": a2 * d2, "bcB": b2 * c2,
    }


def p06():
    n = 40
    rss = {1: 12.60, 2: 10.80, 3: 11.90, 4: 10.36}
    k = {1: 2, 2: 3, 3: 3, 4: 4}          # 切片を含む係数の個数
    dfe = n - k[4]
    sig2 = rss[4] / dfe
    # (a) 部分 F 検定（M1 対 M4）
    q = k[4] - k[1]
    diff = rss[1] - rss[4]
    F = (diff / q) / sig2
    F_noq = diff / sig2
    F_q3 = (diff / 3) / sig2
    F_red = (diff / q) / (rss[1] / (n - k[1]))
    F_nodf = (diff / q) / rss[4]
    # (b) 外部スチューデント化残差
    e, h = 1.25, 0.15
    e2h = e ** 2 / (1 - h)
    s2i = (rss[4] - e2h) / (dfe - 1)
    ti = e / (sqrt(s2i) * sqrt(1 - h))
    ri = e / (sqrt(sig2) * sqrt(1 - h))
    # (c) マローズの Cp
    cp = {m: rss[m] / sig2 - n + 2 * k[m] for m in rss}
    cp_noint = {m: rss[m] / sig2 - n + 2 * (k[m] - 1) for m in rss}
    sig2n = rss[4] / n
    cp_n = {m: rss[m] / sig2n - n + 2 * k[m] for m in rss}
    cp_no2 = {m: rss[m] / sig2 - n + k[m] for m in rss}
    assert min(cp, key=cp.get) == 2 and min(cp_noint, key=cp_noint.get) == 2 and min(cp_n, key=cp_n.get) == 2
    assert min(cp_no2, key=cp_no2.get) == 4
    out = {
        "n": n, "r1": s(rss[1], 2), "r2": s(rss[2], 2), "r3": s(rss[3], 2), "r4": s(rss[4], 2), "dfe": dfe,
        "sig2": s(sig2, 5), "q": q, "diff": s(diff, 2), "dq": s(diff / q, 2),
        "F": s(F, 2), "F4": s(F, 4), "F_noq": s(F_noq, 2), "F_q3": s(F_q3, 2), "F_red": s(F_red, 2), "F_nodf": s(F_nodf, 2),
        "f236": 3.26, "f336": 2.87, "f238": 3.24, "df1": n - k[1], "rss1df": s(rss[1] / (n - k[1]), 5),
        "e": e, "h": h, "h1": s(1 - h, 2), "e2": s(e ** 2, 4), "e2h": s(e2h, 5), "rem": s(rss[4] - e2h, 5),
        "dfe1": dfe - 1, "s2i": s(s2i, 5), "si": s(sqrt(s2i), 5), "sh": s(sqrt(1 - h), 5),
        "den_t": s(sqrt(s2i) * sqrt(1 - h), 5), "ti": r(ti, 2), "ti_s": s(ti, 2), "ti4": s(ti, 4),
        "ri": s(ri, 2), "sig": s(sqrt(sig2), 5),
        "cp2": s(cp[2], 2), "cp4": s(cp[4], 2), "cp2_noint": s(cp_noint[2], 2), "cp2_n": s(cp_n[2], 2),
        "cp4_no2": s(cp_no2[4], 2), "sig2n": s(sig2n, 3),
    }
    for m in rss:
        out[f"q{m}"] = s(rss[m] / sig2, 3)
        out[f"cpv{m}"] = s(cp[m], 2)
        out[f"k{m}"] = k[m]
        out[f"nk{m}"] = n - 2 * k[m]
    return out


def p07():
    sx, sy, rho, w = 15, 10, 0.4, 2
    cov = rho * sx * sy
    v = sx ** 2 + w ** 2 * sy ** 2 + 2 * w * cov
    ev = [2.85, 0.92, 0.55, 0.40, 0.28]
    tot = sum(ev)
    return {
        "sx": sx, "sy": sy, "rho": rho, "w": w, "cov": s(cov, 0), "vx": sx ** 2, "vy": sy ** 2,
        "w2vy": w ** 2 * sy ** 2, "covterm": s(2 * w * cov, 0), "v": s(v, 0), "sd": s(sqrt(v), 2),
        "d_nocov": s(sqrt(sx ** 2 + w ** 2 * sy ** 2), 2), "d_w1": s(sqrt(sx ** 2 + w * sy ** 2 + w * cov), 2),
        "w1vy": w * sy ** 2, "wcov": s(w * cov, 0), "v_w1": s(sx ** 2 + w * sy ** 2 + w * cov, 0),
        "v_once": s(sx ** 2 + w ** 2 * sy ** 2 + w * cov, 0), "d_once": s(sqrt(sx ** 2 + w ** 2 * sy ** 2 + w * cov), 2),
        "d_sum": s(sx + w * sy, 2), "v_nocov": s(sx ** 2 + w ** 2 * sy ** 2, 0),
        "l1": s(ev[0], 2), "l2": s(ev[1], 2), "l3": s(ev[2], 2), "l4": s(ev[3], 2), "l5": s(ev[4], 2), "ltot": s(tot, 2),
        "c1": s(100 * ev[0] / tot, 1), "c12": s(100 * (ev[0] + ev[1]) / tot, 1), "c123": s(100 * sum(ev[:3]) / tot, 1),
    }


def p08():
    # 状態 1（初級）→2（中級）→3（上級：吸収）
    p11, p12 = 0.6, 0.4
    p21, p22, p23 = 0.2, 0.5, 0.3
    # m1 = 1 + p11 m1 + p12 m2,  m2 = 1 + p21 m1 + p22 m2
    a11, a12 = 1 - p11, -p12
    a21, a22 = -p21, 1 - p22
    det = a11 * a22 - a12 * a21
    m1 = (1 * a22 - a12 * 1) / det
    m2 = (a11 * 1 - a21 * 1) / det
    return {
        "p11": p11, "p12": p12, "p21": p21, "p22": p22, "p23": p23,
        "a11": s(a11, 1), "a22": s(a22, 1), "m1": r(m1, 1), "m1_s": s(m1, 1), "m2": s(m2, 1),
        "c1": s(1 / a11, 1), "c2": s(p12 / a11, 0), "rhs": s(1 + p21 * (1 / a11), 1), "lhs": s(a22 - p21 * (p12 / a11), 1),
        "d_23": s(1 / p23, 2), "d_m2": s(m2, 1), "d_geo": s(1 / p12 + 1 / p23, 2),
    }


def p09():
    rr, n = 0.42, 27
    t = rr * sqrt(n - 2) / sqrt(1 - rr ** 2)
    # (b) 操作変数（抽選）
    x1, x0, y1, y0, ols = 6.0, 4.0, 68.0, 64.6, 2.6
    iv = (y1 - y0) / (x1 - x0)
    return {
        "r": rr, "n": n, "n2": n - 2, "r2": s(rr ** 2, 4), "r21": s(1 - rr ** 2, 4), "sq": s(sqrt(1 - rr ** 2), 4),
        "sqn2": s(sqrt(n - 2), 0), "num": s(rr * sqrt(n - 2), 2), "t": s(t, 2), "t4": s(t, 4), "tc": 2.060,
        "d_n": s(rr * sqrt(n) / sqrt(1 - rr ** 2), 2), "d_nosq": s(rr * sqrt(n - 2) / (1 - rr ** 2), 2),
        "d_noden": s(rr * sqrt(n - 2), 2), "d_nonum": s(rr / sqrt(1 - rr ** 2), 2),
        "x1": s(x1, 1), "x0": s(x0, 1), "y1": s(y1, 1), "y0": s(y0, 1), "ols": s(ols, 2),
        "dy": s(y1 - y0, 1), "dx": s(x1 - x0, 1), "iv": s(iv, 2), "d_itt": s(y1 - y0, 2),
        "d_inv": s((x1 - x0) / (y1 - y0), 2), "d_ratio": s(y1 / x1, 2),
    }


def p10():
    means = {"A": 62, "B": 70, "C": 73}
    n, mse, dfe, qv, tv = 8, 48, 21, 3.565, 2.080
    hsd = qv * sqrt(mse / n)
    big = qv * sqrt(2 * mse / n)
    lsd = tv * sqrt(2 * mse / n)
    nosq = qv * mse / n
    half = qv / sqrt(2) * sqrt(mse / n)
    dBA, dCA, dCB = means["B"] - means["A"], means["C"] - means["A"], means["C"] - means["B"]
    assert dBA < hsd < dCA and dCB < hsd and lsd < dBA and half < dBA and big > dCA and nosq > dCA
    # (b) ホルム法
    pv = {"ア": 0.038, "イ": 0.003, "ウ": 0.20, "エ": 0.012, "オ": 0.009}
    return {
        "mA": means["A"], "mB": means["B"], "mC": means["C"], "n": n, "mse": mse, "dfe": dfe, "q": qv, "tv": tv,
        "k": 3, "N": 3 * n, "msen": s(mse / n, 0), "sq": s(sqrt(mse / n), 5),
        "hsd": s(hsd, 2), "hsd4": s(hsd, 4), "big": s(big, 2), "lsd": s(lsd, 2), "nosq": s(nosq, 2), "half": s(half, 2),
        "dBA": dBA, "dCA": dCA, "dCB": dCB, "sq2": s(sqrt(2 * mse / n), 4),
        "pa": pv["ア"], "pi": pv["イ"], "pu": pv["ウ"], "pe": pv["エ"], "po": pv["オ"],
        "a5": s(0.05 / 5, 4), "a4": s(0.05 / 4, 4), "a3": s(0.05 / 3, 4), "a2": s(0.05 / 2, 4), "a10": s(0.05 / 10, 4),
    }


def p11():
    th1, th2, beta = -0.5, 1.5, 0.8
    ex = {-2.3: 0.1003, -1.5: 0.2231, -0.7: 0.4966, -0.3: 0.7408, 0.5: 1.6487, 1.3: 3.6693}

    def cum(eta):
        # P = e^eta/(1+e^eta) = 1/(1+e^{-eta})
        return 1 / (1 + ex[round(-eta, 1)])
    c1, c2 = cum(th1 - beta), cum(th2 - beta)
    s1, s2 = cum(th1 + beta), cum(th2 + beta)
    z1, z2 = cum(th1), cum(th2)
    # (b) AUC
    pos = [0.82, 0.64, 0.55, 0.31]
    neg = [0.70, 0.48, 0.40, 0.25, 0.18, 0.12]
    wins = sum(1 for a in pos for b in neg if a > b)
    per = [sum(1 for b in neg if a > b) for a in pos]
    return {
        "th1": th1, "th2": th2, "beta": beta, "eta1": s(th1 - beta, 1), "eta2": s(th2 - beta, 1),
        "e_m23": s(ex[-2.3], 4), "e_m15": s(ex[-1.5], 4), "e_m07": s(ex[-0.7], 4), "e_m03": s(ex[-0.3], 4),
        "e_05": s(ex[0.5], 4), "e_13": s(ex[1.3], 4),
        "c1": s(c1, 4), "c2": s(c2, 4), "ans": s(c2 - c1, 3), "ans4": s(c2 - c1, 4),
        "d_p1": s(c1, 3), "d_cum": s(c2, 3), "d_sign": s(s2 - s1, 3), "d_x0": s(z2 - z1, 3), "d_p3": s(1 - c2, 3),
        "pos": "、".join(f"{x:.2f}" for x in pos), "neg": "、".join(f"{x:.2f}" for x in neg),
        "np": len(pos), "nn": len(neg), "pairs": len(pos) * len(neg), "wins": wins,
        "w1": per[0], "w2": per[1], "w3": per[2], "w4": per[3],
        "auc": r(wins / (len(pos) * len(neg)), 3), "auc_s": s(wins / (len(pos) * len(neg)), 3),
        "auc4": s(wins / (len(pos) * len(neg)), 4),
    }


def p12():
    N, e, z, p = 800, 0.05, 1.96, 0.5
    n0 = z ** 2 * p * (1 - p) / e ** 2
    nreq = N * n0 / (N - 1 + n0)
    ans = int(-(-nreq // 1))  # 切り上げ
    n_simple = n0 / (1 + n0 / N)
    # (b) ルービンのルール
    est = [61.2, 62.0, 60.8, 61.6, 61.4]
    se2 = [0.90, 0.85, 0.95, 0.88, 0.92]
    m = len(est)
    qbar = sum(est) / m
    W = sum(se2) / m
    B = sum((x - qbar) ** 2 for x in est) / (m - 1)
    T = W + (1 + 1 / m) * B
    Bm = sum((x - qbar) ** 2 for x in est) / m
    return {
        "N": N, "e": e, "z": z, "n0": s(n0, 2), "N1": N - 1, "nnum": s(N * n0, 0), "nden": s(N - 1 + n0, 2),
        "nreq": s(nreq, 2), "ans": ans, "n0up": int(-(-n0 // 1)), "nsimple": s(n_simple, 2),
        "est": "、".join(f"{x:.1f}" for x in est), "se2": "、".join(f"{x:.2f}" for x in se2), "m": m,
        "qbar": s(qbar, 1), "W": s(W, 2), "ssB": s(B * (m - 1), 2), "B": s(B, 2), "BT": s((1 + 1 / m) * B, 2),
        "T": s(T, 2), "seT": s(sqrt(T), 3), "seT4": s(sqrt(T), 4),
        "d_W": s(sqrt(W), 3), "d_WB": s(sqrt(W + B), 3), "d_Bm": s(sqrt(W + B / m), 3),
        "d_Bn": s(sqrt(W + (1 + 1 / m) * Bm), 3), "WBm": s(W + B / m, 2), "WB": s(W + B, 2), "Bmv": s(Bm, 2),
        "TBn": s(W + (1 + 1 / m) * Bm, 3), "d_T": s(T, 3),
    }


def p13():
    # 平方ユークリッド距離（座標は問題文に示さない）：A(0,0) B(2,0) C(1,3) D(5,1)
    pts = {"A": (0, 0), "B": (2, 0), "C": (1, 3), "D": (5, 1)}
    d2 = lambda u, v: (pts[u][0] - pts[v][0]) ** 2 + (pts[u][1] - pts[v][1]) ** 2
    dAB, dAC, dAD, dBC, dBD, dCD = d2("A", "B"), d2("A", "C"), d2("A", "D"), d2("B", "C"), d2("B", "D"), d2("C", "D")
    dC_AB = 0.5 * dAC + 0.5 * dBC - 0.25 * dAB
    dD_AB = 0.5 * dAD + 0.5 * dBD - 0.25 * dAB
    assert dC_AB < min(dD_AB, dCD)
    dD_ABC = 2 / 3 * dD_AB + 1 / 3 * dCD - 2 / 9 * dC_AB
    wrong_noterm = 2 / 3 * dD_AB + 1 / 3 * dCD
    wrong_half = 0.5 * dD_AB + 0.5 * dCD - 0.25 * dC_AB
    single = min(dAD, dBD, dCD)
    complete = max(dAD, dBD, dCD)
    return {
        "dAB": dAB, "dAC": dAC, "dAD": dAD, "dBC": dBC, "dBD": dBD, "dCD": dCD,
        "dC_AB": s(dC_AB, 0), "dD_AB": s(dD_AB, 0), "ans": s(dD_ABC, 2), "ans0": s(dD_ABC, 0),
        "d_noterm": s(wrong_noterm, 2), "d_half": s(wrong_half, 2), "d_single": s(single, 2), "d_complete": s(complete, 2),
        "t1": s(2 / 3 * dD_AB, 4), "t2": s(1 / 3 * dCD, 4), "t3": s(2 / 9 * dC_AB, 0),
        "u1": s(0.5 * dAC, 0), "u2": s(0.5 * dBC, 0), "u3": s(0.25 * dAB, 0),
        "v1": s(0.5 * dAD, 0), "v2": s(0.5 * dBD, 0),
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
