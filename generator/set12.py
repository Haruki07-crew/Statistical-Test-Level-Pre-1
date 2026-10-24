"""模擬試験 第12回（set12）の数値。values() が {大問id: {名前: 値}} を返す。

題材：化学・材料・食品・醸造の品質管理。
問題文に示した関数値（e^x、ln x、χ² の点など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
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
    # ガラス瓶：気泡の数 X（0,1,2）と傷の有無 Y（0,1）の同時確率
    P = {(0, 0): 0.50, (0, 1): 0.10, (1, 0): 0.15, (1, 1): 0.10, (2, 0): 0.05, (2, 1): 0.10}
    px = {x: P[(x, 0)] + P[(x, 1)] for x in range(3)}
    py = {y: sum(P[(x, y)] for x in range(3)) for y in range(2)}
    # (a) E[X | Y=1]
    num1 = sum(x * P[(x, 1)] for x in range(3))
    ex_y1 = num1 / py[1]
    ex = sum(x * px[x] for x in range(3))
    ex_y0 = sum(x * P[(x, 0)] for x in range(3)) / py[0]
    pge1_y1 = (P[(1, 1)] + P[(2, 1)]) / py[1]
    # (b) 相関係数
    ex2 = sum(x * x * px[x] for x in range(3))
    vx = ex2 - ex ** 2
    ey = py[1]
    vy = ey * (1 - ey)
    exy = sum(x * y * P[(x, y)] for x in range(3) for y in range(2))
    cov = exy - ex * ey
    prod = vx * vy
    sq = r(sqrt(prod), 4)
    rho = cov / sq
    out = {f"p{x}{y}": s(P[(x, y)], 2) for x in range(3) for y in range(2)}
    out.update({
        "px0": s(px[0], 2), "px1": s(px[1], 2), "px2": s(px[2], 2), "py0": s(py[0], 2), "py1": s(py[1], 2),
        "num1": s(num1, 2), "ex_y1": s(ex_y1, 2), "ex": s(ex, 2), "ex_y0": s(ex_y0, 2), "pge1": s(pge1_y1, 2),
        "joint1": s(num1, 2),
        "ex2": s(ex2, 2), "ex_sq": s(ex ** 2, 4), "vx": s(vx, 4), "ey": s(ey, 2), "vy": s(vy, 2),
        "exy": s(exy, 2), "exey": s(ex * ey, 3), "cov": s(cov, 3), "prod": s(prod, 6), "sq": s(sq, 4),
        "rho": r(rho, 3), "rho_s": s(rho, 3), "rho4": s(rho, 4),
        "rho_e": s(cov / (vx * vy), 3), "rho_noc": s(exy / sq, 3),
    })
    return out


def p02():
    # (a) 3段階の反応時間（各段階 平均 0.5 時間の指数分布）の合計がガンマ分布
    rate, k, t = 2, 3, 1.5
    lam = rate * t                      # ポアソンの平均 3
    e3 = 0.0498                         # 問題文に示す e^{-3}
    terms = [1, lam, lam ** 2 / 2]      # 1 + 3 + 4.5
    sum3 = sum(terms)
    ans = sum3 * e3
    wrong2 = (1 + lam) * e3             # k=2 項までで止めた
    wrong_exp = e3                      # 1段階だけの指数分布
    # (b) X ~ Exp(率 3), Y ~ Exp(率 1) で U = X/(X+Y)
    l1, l2 = 3, 1
    return {
        "rate": rate, "k": k, "t": t, "lam": s(lam, 0), "e3": s(e3, 4), "sum3": s(sum3, 1),
        "ans": r(ans, 3), "ans_s": s(ans, 3), "ans4": s(ans, 4),
        "wrong2": s(wrong2, 3), "wrong_k4": s((sum3 + lam ** 3 / 6) * e3, 3),
        "l1": l1, "l2": l2, "dl": l1 - l2,
    }


def p03():
    # 酒米の等級：多項分布
    n, p1, p2, p3 = 120, 0.15, 0.60, 0.25
    v1 = p1 * (1 - p1)
    v3 = p3 * (1 - p3)
    cv = p1 * p3
    var = n * (v1 + v3 + 2 * cv)
    var_nocov = n * (v1 + v3)
    var_minus = n * (v1 + v3 - 2 * cv)
    mean = n * (p1 - p3)
    # (b) チェビシェフ
    kk = 15
    bound = var / kk ** 2
    sd = sqrt(var)
    z = r(kk / sd, 2)
    phi_z = 0.9857   # Φ(2.19)（解説のみで使う）
    return {
        "n": n, "p1": s(p1, 2), "p2": s(p2, 2), "p3": s(p3, 2),
        "v1": s(v1, 4), "v3": s(v3, 4), "cv": s(cv, 4), "cv2": s(2 * cv, 3),
        "inner": s(v1 + v3 + 2 * cv, 2), "var": r(var, 1), "var_s": s(var, 1),
        "var_nocov": s(var_nocov, 1), "var_minus": s(var_minus, 1),
        "inner_alt": s(p1 + p3 - (p1 - p3) ** 2, 2), "p13": s(p1 + p3, 2), "dsq": s((p1 - p3) ** 2, 2),
        "mean": s(mean, 0), "kk": kk, "kk2": kk ** 2,
        "bound": s(bound, 3), "b_nocov": s(var_nocov / kk ** 2, 3), "b_sd": s(sd / kk, 3),
        "b_half": s(bound / 2, 3), "b_norm": s(2 * (1 - phi_z), 3), "sd": s(sd, 3), "z": s(z, 2),
        "phiz": s(phi_z, 4),
    }


def p04():
    n = 10
    ratio = 3 / (n + 2)
    noscale = 3 * n ** 2 / ((n + 1) ** 2 * (n + 2))
    wrongvar = 1 / (n + 2)
    mse_max = 6 * n / ((n + 1) * (n + 2))
    # (b) ブートストラップ
    q10 = 0.9 ** 10
    q10s = r(q10, 4)
    q9 = r(0.9 ** 9, 4)
    return {
        "n": n, "n1": n + 1, "n2": n + 2, "ratio": s(ratio, 3), "noscale": s(noscale, 3),
        "wrongvar": s(wrongvar, 3), "mse_max": s(mse_max, 3), "inv": s(1 / ratio, 3),
        "v1den": 3 * n, "v2den": n * (n + 2),
        "q10": s(q10s, 4), "q9": s(q9, 4), "boot": s(1 - q10s, 3), "boot4": s(1 - q10s, 4),
        "once": s(10 * 0.1 * q9, 3), "limit": s(0.632, 3), "single": s(0.1, 3), "comp": s(q10s, 3),
    }


def p05():
    x = [2, 0, 5, 1, 3, 8, 0, 2, 6, 3]
    n = len(x)
    tot = sum(x)
    m = tot / n
    ss = sum((v - m) ** 2 for v in x)
    D = ss / m
    # (b) 不良ゼロのときの検査個数
    ln05, ln99, ln025, ln01 = -2.9957, -0.010050, -3.6889, -4.6052
    nn = ln05 / ln99
    import math
    n_ok = math.ceil(nn)
    n_two = math.ceil(ln025 / ln99)
    n_99 = math.ceil(ln01 / ln99)
    return {
        "xs": ", ".join(str(v) for v in x), "n": n, "tot": tot, "m": s(m, 1), "ss": s(ss, 0),
        "D": s(D, 2), "Dvar": s(ss / (n - 1) / m, 2), "svar": s(ss / (n - 1), 2), "df": n - 1, "df1": n,
        "c9": "16.92", "c10": "18.31", "c9l": "3.33",
        "ln05": "-2.9957", "ln99": "-0.010050", "ln025": "-3.6889", "ln01": "-4.6052",
        "nraw": s(nn, 2), "n_ok": n_ok, "n_trunc": n_ok - 1, "n_r3": 300, "n_two": n_two, "n_99": n_99,
        "nraw_two": s(ln025 / ln99, 2), "nraw_99": s(ln01 / ln99, 2),
    }


def p06():
    d = [12, 5, -3, 10, 8, 15, -2, 7, 11, 4]
    n = len(d)
    tot = sum(d)
    m = tot / n
    sq = sum(v * v for v in d)
    ss = sq - n * m ** 2
    s2 = ss / (n - 1)
    sd = r(sqrt(s2), 3)
    se = r(sd / sqrt(n), 3)
    t = m / se
    # 符号検定
    from math import comb
    npos = sum(1 for v in d if v > 0)
    tail = sum(comb(n, k) for k in range(npos, n + 1))
    tail9 = sum(comb(n, k) for k in range(npos + 1, n + 1))
    return {
        "ds": ", ".join(str(v) for v in d), "n": n, "tot": tot, "m": s(m, 1), "sq": sq, "nm2": s(n * m ** 2, 1),
        "ss": s(ss, 1), "s2": s(s2, 3), "sd": s(sd, 3), "sqrtn": "3.162", "se": s(se, 3),
        "t": r(t, 2), "t_s": s(t, 2), "t4": s(t, 4),
        "t_mle": s(m / (sqrt(ss / n) / sqrt(n)), 2), "t_nosqrt": s(m / sd, 2),
        "npos": npos, "c8": comb(n, 8), "c9": comb(n, 9), "c10": comb(n, 10), "tail": tail, "tail9": tail9,
        "p_one": s(tail / 1024, 3), "p_two": s(2 * tail / 1024, 3), "p_eq": s(comb(n, 8) / 1024, 3),
        "p9": s(tail9 / 1024, 3), "p9two": s(2 * tail9 / 1024, 3), "p_one4": s(tail / 1024, 4),
    }


def p07():
    lam, mu_min = 6, 8          # 1時間に6件、平均分析時間 8 分
    mu = 60 / mu_min            # 7.5 件/時
    rho = lam / mu
    lq = rho ** 2 / (1 - rho)
    wq = lq / lam * 60
    w = wq + mu_min
    return {
        "lam": lam, "mu_min": mu_min, "mu": s(mu, 1), "rho": s(rho, 1), "lq": s(lq, 1), "L": s(rho / (1 - rho), 1),
        "wq": s(wq, 1), "w": s(w, 1), "w_lqmu": s(lq / mu * 60, 1), "w_rs": s(rho * mu_min, 1),
        "w_bad": s(lq / (lam * (1 - rho)) * 60, 1), "wq_h": s(lq / lam, 4), "lamrho": s(lam * (1 - rho), 1),
    }


def p08():
    n, k_full, k_red = 40, 4, 2
    rss_f, rss_r = 120.0, 139.2
    q = k_full - k_red
    df_f = n - k_full - 1
    mse = rss_f / df_f
    mse_s = r(mse, 4)
    num = (rss_r - rss_f) / q
    F = num / mse_s
    F_df = num / (rss_f / (n - k_full))
    F_noq = (rss_r - rss_f) / mse_s
    F_red = num / (rss_r / (n - k_red - 1))
    # (b) 外部スチューデント化残差
    e, h = 4.5, 0.25
    s_full = r(sqrt(mse), 4)
    sq1h = 0.8660
    e2h = e ** 2 / (1 - h)
    s2i = (rss_f - e2h) / (df_f - 1)
    s2i_s = r(s2i, 4)
    si = r(sqrt(s2i_s), 4)
    den = r(si * sq1h, 4)
    t_ext = e / den
    r_int = e / (s_full * sq1h)
    r_raw = e / s_full
    r_noroot = e / (s_full * (1 - h))
    s2i_bad = (rss_f - e ** 2) / (df_f - 1)
    t_bad = e / (sqrt(s2i_bad) * sq1h)
    # (c) 操作変数
    czy, czx, cxy, vx, vz = 0.84, 0.60, 1.10, 1.00, 2.00
    iv = czy / czx
    return {
        "n": n, "kf": k_full, "kr": k_red, "rss_f": s(rss_f, 1), "rss_r": s(rss_r, 1), "q": q, "df_f": df_f,
        "dss": s(rss_r - rss_f, 1), "num": s(num, 1), "mse": s(mse_s, 4), "F": s(F, 2), "F_df": s(F_df, 2),
        "F_noq": s(F_noq, 2), "F_red": s(F_red, 2), "F_ratio": s((rss_r - rss_f) / rss_f, 2),
        "dfr": n - k_full, "dfred": n - k_red - 1, "mse_df": s(rss_f / (n - k_full), 4), "mse_red": s(rss_r / (n - k_red - 1), 4),
        "Fcrit": "3.27",
        "e": e, "h": h, "omh": s(1 - h, 2), "sq1h": s(sq1h, 4), "s_full": s(s_full, 4), "e2": s(e ** 2, 2),
        "e2h": s(e2h, 1), "rss_i": s(rss_f - e2h, 1), "df_i": df_f - 1, "s2i": s(s2i_s, 4), "si": s(si, 4),
        "den": s(den, 4), "t_ext": s(t_ext, 2), "t_ext4": s(t_ext, 4), "r_int": s(r_int, 2), "r_raw": s(r_raw, 2),
        "r_noroot": s(r_noroot, 2), "t_bad": s(t_bad, 2), "s2i_bad": s(s2i_bad, 4),
        "sfull_den": s(r(s_full * sq1h, 4), 4),
        "czy": s(czy, 2), "czx": s(czx, 2), "cxy": s(cxy, 2), "vx": s(vx, 2), "vz": s(vz, 2),
        "iv": s(iv, 2), "ols": s(cxy / vx, 2), "rf": s(czy / vz, 2), "fs": s(czx / vz, 2), "ivinv": s(czx / czy, 2),
    }


def p09():
    th1, th2, beta, x = -1.0, 1.5, 0.8, 2.5
    bx = beta * x
    e3, e05 = 20.086, 1.6487
    em1, em35 = 0.3679, 0.0302
    c1 = 1 / (1 + e3)
    c2 = 1 / (1 + e05)
    c1s, c2s = r(c1, 4), r(c2, 4)
    p2 = c2s - c1s
    # 符号を逆にした誤り θ + βx
    w1 = r(1 / (1 + em1), 4)
    w2 = r(1 / (1 + em35), 4)
    return {
        "th1": s(th1, 1), "th2": s(th2, 1), "beta": s(beta, 1), "x": s(x, 1), "bx": s(bx, 1),
        "a1": s(th1 - bx, 1), "a2": s(th2 - bx, 1), "e3": s(e3, 3), "e05": s(e05, 4), "em1": s(em1, 4),
        "em35": s(em35, 4), "e1": "2.7183", "em05": "0.6065", "em3": "0.0498",
        "c1": s(c1s, 4), "c2": s(c2s, 4), "p2": s(p2, 3), "p2_4": s(p2, 4), "p3": s(1 - c2s, 3), "p1": s(c1s, 3),
        "cum2": s(c2s, 3), "w1": s(w1, 4), "w2": s(w2, 4), "wrong": s(w2 - w1, 3),
        "d1": s(1 + e3, 3), "d2": s(1 + e05, 4),
    }


def p10():
    a, rr = 5, 4
    ssa, sse = 2.40, 1.50
    dfa, dfe = a - 1, a * (rr - 1)
    msa, mse = ssa / dfa, sse / dfe
    sa2 = (msa - mse) / rr
    icc = sa2 / (sa2 + mse)
    sa2_a = (msa - mse) / a
    icc_a = sa2_a / (sa2_a + mse)
    sa2_nosub = msa / rr
    icc_nosub = sa2_nosub / (sa2_nosub + mse)
    return {
        "a": a, "rr": rr, "ssa": s(ssa, 2), "sse": s(sse, 2), "dfa": dfa, "dfe": dfe, "msa": s(msa, 2),
        "mse": s(mse, 2), "sa2": s(sa2, 3), "tot": s(sa2 + mse, 3), "icc": r(icc, 3), "icc_s": s(icc, 3),
        "icc4": s(icc, 4), "icc_a": s(icc_a, 3), "icc_nosub": s(icc_nosub, 3), "sa2_a": s(sa2_a, 3),
        "fratio": s(sa2 / mse, 2), "F": s(msa / mse, 1),
    }


def p11():
    ys = [12, 5, 20]
    ps = [0.30, 0.10, 0.50]
    ht = sum(y / p for y, p in zip(ys, ps))
    N = 12
    out = {
        "y1": ys[0], "y2": ys[1], "y3": ys[2], "pi1": s(ps[0], 2), "pi2": s(ps[1], 2), "pi3": s(ps[2], 2),
        "w1": s(ys[0] / ps[0], 1), "w2": s(ys[1] / ps[1], 1), "w3": s(ys[2] / ps[2], 1),
        "N": N, "ht": s(ht, 1), "mul": s(sum(y * p for y, p in zip(ys, ps)), 1), "avg": s(ht / 3, 1),
        "nmean": s(N * sum(ys) / 3, 1), "sumratio": s(sum(ys) / (sum(ps) / 3), 1), "sumy": sum(ys),
        "pbar": s(sum(ps) / 3, 1),
    }
    # (b) Rubin のルール
    Q = [812, 806, 818, 809, 815]
    U = [140, 150, 145, 138, 152]
    m = len(Q)
    qb = sum(Q) / m
    ub = sum(U) / m
    B = sum((q - qb) ** 2 for q in Q) / (m - 1)
    T = ub + (1 + 1 / m) * B
    out.update({
        "Qs": ", ".join(map(str, Q)), "Us": ", ".join(map(str, U)), "m": m, "qb": s(qb, 0), "ub": s(ub, 0),
        "ssq": s(sum((q - qb) ** 2 for q in Q), 0), "B": s(B, 1), "fac": s(1 + 1 / m, 1), "fB": s((1 + 1 / m) * B, 1),
        "T": s(T, 1), "se": r(sqrt(T), 2), "se_s": s(sqrt(T), 2), "se4": s(sqrt(T), 4),
        "se_nofac": s(sqrt(ub + B), 2), "se_u": s(sqrt(ub), 2), "se_bm": s(sqrt(ub + (1 + 1 / m) * B * (m - 1) / m), 2),
        "Bm": s(B * (m - 1) / m, 1),
    })
    return out


def p12():
    # (a) 2変量の主成分
    s11, s12, s22 = 5, 2, 2
    tr, det = s11 + s22, s11 * s22 - s12 ** 2
    disc = tr ** 2 - 4 * det
    l1 = (tr + sqrt(disc)) / 2
    l2 = (tr - sqrt(disc)) / 2
    d1, d2 = 1.5, -1.0
    sq5 = 2.2361
    score = (2 * d1 + 1 * d2) / sq5
    score2 = (1 * d1 - 2 * d2) / sq5
    # (b) 因子分析の自由度
    p, k = 8, 2
    df = ((p - k) ** 2 - (p + k)) // 2
    # (c) パス解析
    a, b, c = 0.6, 0.5, -0.2
    return {
        "s11": s11, "s12": s12, "s22": s22, "tr": tr, "det": det, "disc": disc, "sqd": s(sqrt(disc), 0),
        "l1": s(l1, 0), "l2": s(l2, 0), "d1": s(d1, 1), "d2": s(d2, 1), "sq5": s(sq5, 4),
        "score": s(score, 3), "score2": s(score2, 3), "unnorm": s(2 * d1 + d2, 3), "contrib": s(l1 / tr, 3),
        "eqw": s((d1 + d2) / sqrt(2), 3), "numer": s(2 * d1 + d2, 1),
        "p": p, "k": k, "df": df, "pp1": p * (p + 1) // 2, "pk": p * k, "rot": k * (k - 1) // 2,
        "df_norot": p * (p + 1) // 2 - p * k - p, "df_nouniq": p * (p + 1) // 2 - p * k - k * (k - 1) // 2,
        "df_nouniq_rot": p * (p + 1) // 2 - p * k, "df_ind": p * (p - 1) // 2,
        "pa": s(a, 1), "pb": s(b, 1), "pc": s(c, 1), "rbs": s(b + c * a, 2), "ca": s(c * a, 2),
        "total": s(c + a * b, 2), "rbs_m": s(b - c * a, 2), "ab": s(a * b, 2),
    }


def p13():
    A = (4, 196, 2, 48)    # 新：腐敗あり・なし、旧：腐敗あり・なし
    B = (6, 54, 30, 90)
    def st(a, b, c, d):
        n = a + b + c + d
        return a * d / n, b * c / n, a * d / (b * c), n
    sa, sb = st(*A), st(*B)
    mh = (sa[0] + sb[0]) / (sa[1] + sb[1])
    a, b, c, d = (A[i] + B[i] for i in range(4))
    crude = a * d / (b * c)
    noN = (A[0] * A[3] + B[0] * B[3]) / (A[1] * A[2] + B[1] * B[2])
    out = {
        "Aa": A[0], "Ab": A[1], "Ac": A[2], "Ad": A[3], "Ba": B[0], "Bb": B[1], "Bc": B[2], "Bd": B[3],
        "An": sa[3], "Bn": sb[3], "Aad": A[0] * A[3], "Abc": A[1] * A[2], "Bad": B[0] * B[3], "Bbc": B[1] * B[2],
        "Aadn": s(sa[0], 3), "Abcn": s(sa[1], 3), "Badn": s(sb[0], 3), "Bbcn": s(sb[1], 3),
        "numer": s(sa[0] + sb[0], 3), "denom": s(sa[1] + sb[1], 3),
        "mh": s(mh, 3), "crude": s(crude, 3), "noN": s(noN, 3), "mean": s((sa[2] + sb[2]) / 2, 3),
        "inv": s(1 / mh, 3), "orA": s(sa[2], 3), "orB": s(sb[2], 3),
        "ta": a, "tb": b, "tc": c, "td": d,
    }
    # (b) フロントドア調整
    px1 = 0.4
    pm = {1: 0.10, 0: 0.50}            # P(M=1 | X=x)
    py = {(1, 1): 0.30, (1, 0): 0.20, (0, 1): 0.06, (0, 0): 0.04}   # P(Y=1 | M=m, X=x)
    inner = {m: py[(m, 1)] * px1 + py[(m, 0)] * (1 - px1) for m in (0, 1)}
    do = {x: pm[x] * inner[1] + (1 - pm[x]) * inner[0] for x in (0, 1)}
    ace = do[1] - do[0]
    obs = {x: pm[x] * py[(1, x)] + (1 - pm[x]) * py[(0, x)] for x in (0, 1)}
    out.update({
        "px1": s(px1, 1), "px0": s(1 - px1, 1), "pm1": s(pm[1], 2), "pm0": s(pm[0], 2),
        "pm1c": s(1 - pm[1], 2), "pm0c": s(1 - pm[0], 2),
        "y11": s(py[(1, 1)], 2), "y10": s(py[(1, 0)], 2), "y01": s(py[(0, 1)], 2), "y00": s(py[(0, 0)], 2),
        "in1": s(inner[1], 3), "in0": s(inner[0], 3), "do1": s(do[1], 4), "do0": s(do[0], 4),
        "ace": r(ace, 3), "ace_s": s(ace, 3), "ace4": s(ace, 4),
        "obs1": s(obs[1], 3), "obs0": s(obs[0], 3), "naive": s(obs[1] - obs[0], 3),
        "t11": s(py[(1, 1)] * px1, 3), "t10": s(py[(1, 0)] * (1 - px1), 3),
        "t01": s(py[(0, 1)] * px1, 3), "t00": s(py[(0, 0)] * (1 - px1), 3),
    })
    return out


def values():
    return {
        "s12p01": p01(), "s12p02": p02(), "s12p03": p03(), "s12p04": p04(), "s12p05": p05(),
        "s12p06": p06(), "s12p07": p07(), "s12p08": p08(), "s12p09": p09(), "s12p10": p10(),
        "s12p11": p11(), "s12p12": p12(), "s12p13": p13(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
