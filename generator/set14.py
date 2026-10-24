"""模擬試験 第14回（set14）の数値。values() が {大問id: {名前: 値}} を返す。

題材：スポーツ・エンタメ・ゲーム・メディア。
問題文に示した値（e^x、χ² や F の点、atanh の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
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


def frac(f):
    """Fraction を LaTeX の分数に。"""
    f = Fr(f)
    if f.denominator == 1:
        return str(f.numerator)
    return f"\\frac{{{f.numerator}}}{{{f.denominator}}}"


# ---------------------------------------------------------------- 大問1 ボードゲームのコイン
def p01():
    # 公平なコイン2枚：A=赤が表、B=青が表、C=表裏が一致
    half = Fr(1, 2)
    pab = half * half            # P(A∩B)=P(両方表)
    pabc = Fr(1, 4)              # A∩B∩C = 両方表
    prod3 = half ** 3
    union = 3 * half - 3 * pab + pabc   # 包除原理
    union_ind = 1 - (1 - half) ** 3
    return {"half": frac(half), "pab": frac(pab), "pabc": frac(pabc), "prod3": frac(prod3),
            "union": frac(union), "union_ind": frac(union_ind)}


# ---------------------------------------------------------------- 大問2 スピードランのレース
def p02():
    ma, mb, mc = 10, 15, 30
    la, lb, lc = Fr(1, ma), Fr(1, mb), Fr(1, mc)
    L = la + lb + lc
    t = 4
    e08 = 0.4493                 # e^{-0.8}（問題文に示す値）
    e04 = 0.6703                 # e^{-0.4}
    assert float(L * t) == 0.8 and float(la * t) == 0.4
    share = la / L
    ok = float(share) * (1 - e08)
    # (b) D = X_A - X_B、P(D > 5)
    d = 5
    e05 = 0.6065                 # e^{-0.5}
    e13 = 0.7165                 # e^{-1/3}
    assert float(la * d) == 0.5 and abs(float(lb * d) - 1 / 3) < 1e-12
    pb = lb / (la + lb)          # P(X_A > X_B)
    okb = float(pb) * e05
    return {
        "ma": ma, "mb": mb, "mc": mc, "la": frac(la), "lb": frac(lb), "lc": frac(lc), "L": frac(L),
        "Ld": s(float(L), 1), "Lt": s(float(L * t), 1), "lat": s(float(la * t), 1), "t": t,
        "e08": e08, "e04": e04, "share": s(float(share), 1), "minp": s(1 - e08, 4),
        "ok": s(ok, 3), "ok5": s(ok, 5), "w_share": s(float(share), 3), "w_min": s(1 - e08, 3),
        "w_a": s(1 - e04, 3), "w_ab": s(float(share) * (1 - e04), 3), "meanmin": s(float(1 / L), 0),
        "d": d, "e05": e05, "e13": e13, "lab": frac(la + lb), "c0": frac(la * lb / (la + lb)),
        "pb": frac(pb), "pbd": s(float(pb), 1), "pa": s(float(la / (la + lb)), 1),
        "okb": s(okb, 3), "okb5": s(okb, 5), "w_pb": s(float(pb), 3), "w_e": s(e05, 3),
        "w_swap": s(float(la / (la + lb)) * e05, 3), "w_rate": s(float(pb) * e13, 3),
        "lad": s(float(la * d), 1),
    }


# ---------------------------------------------------------------- 大問3 バスケットボール
def p03():
    s1, s2, rho = 6, 8, 0.25
    cov12 = rho * s1 * s2
    cov = s1 ** 2 + cov12
    var = s1 ** 2 + s2 ** 2 + 2 * cov12
    sd = r(sqrt(var), 3)
    corr = cov / (s1 * sd)
    # (b) フリースローの加算推定量 (X+1)/(n+2) の MSE
    n, p = 10, 0.8
    v = n * p * (1 - p) / (n + 2) ** 2
    m = (n * p + 1) / (n + 2)
    bias = m - p
    mse = v + bias ** 2
    mse_mle = p * (1 - p) / n
    # (c) 信頼区間の幅から必要な本数
    z, e = 1.96, 0.03
    z2 = r(z * z, 4)
    nraw = z2 * p * (1 - p) / e ** 2
    import math
    return {
        "s1": s1, "s2": s2, "rho": rho, "v1": s1 ** 2, "v2": s2 ** 2, "cov12": s(cov12, 0),
        "cov": s(cov, 0), "var": s(var, 0), "sd": s(sd, 3), "den": s(s1 * sd, 3),
        "corr": r(corr, 2), "corr4": s(corr, 4),
        "w_nocov": s(s1 ** 2 / (s1 * sqrt(s1 ** 2 + s2 ** 2)), 2), "w_var": s(cov / var, 2),
        "n": n, "p": p, "q": s(1 - p, 1), "np": s(n * p, 0), "np1": s(n * p + 1, 0), "n2": n + 2, "n22": (n + 2) ** 2,
        "npq": s(n * p * (1 - p), 1), "vt": s(v, 6), "m": s(m, 2), "bias": s(bias, 2), "bias2": s(bias ** 2, 4),
        "mse": r(mse, 4), "mse6": s(mse, 6), "mse_s": s(mse, 4), "mse_mle": s(mse_mle, 3),
        "w_nobias": s(v, 4), "w_mvar": s(p * (1 - p) / (n + 2) + bias ** 2, 4),
        "z": z, "e": e, "z2": s(z2, 4), "pq": s(p * (1 - p), 2), "e2": s(e ** 2, 4),
        "nraw": s(nraw, 2), "nreq": math.ceil(nraw),
        "n_wc": math.ceil(z2 * 0.25 / e ** 2), "n_wcraw": s(z2 * 0.25 / e ** 2, 2),
        "n_164": math.ceil(r(1.645 ** 2, 4) * p * (1 - p) / e ** 2),
        "n_nosq": math.ceil(z * p * (1 - p) / e ** 2), "n_e": math.ceil(z2 * p * (1 - p) / e),
    }


# ---------------------------------------------------------------- 大問4 ソーシャルゲームの運営
def p04():
    # (a) 乱択検定 B(20, 0.1)
    n, p0 = 20, 0.1
    pge5, pge4, p4 = 0.0432, 0.1330, 0.0898     # 問題文に示す値
    assert abs(pge4 - (pge5 + p4)) < 1e-12
    a = 0.05
    gam = (a - pge5) / p4
    # (b) 極値分布
    m, nn = 30, 200
    ln = 5.298                  # ln 200
    g = 2.250                   # -ln(-ln 0.9)
    c = m * (ln + g)
    return {
        "n": n, "p0": p0, "pge5": pge5, "pge4": pge4, "p4": p4, "a": a,
        "num": s(a - pge5, 4), "gam": s(gam, 3), "gam5": s(gam, 5),
        "w_nodiv": s(a - pge5, 3), "w_ge4": s((a - pge5) / pge4, 3), "w_nosub": s(a / p4, 3),
        "w_rev": s((pge4 - a) / p4, 3), "size": s(pge5 + gam * p4, 4),
        "m": m, "nn": nn, "ln": ln, "g": g, "lg": s(ln + g, 3), "c": s(c, 1), "c2": s(c, 2),
        "w_sign": s(m * (ln - g), 1), "w_loc": s(m * ln, 1), "w_noscale": s(m * ln + g, 1), "w_one": s(m * g, 1),
    }


# ---------------------------------------------------------------- 大問5 カジノのミニゲーム
def p05():
    p, q = 0.45, 0.55
    i, N = 3, 6
    rr = Fr(55, 45)
    r3 = rr ** 3
    r6 = rr ** 6
    pw = 1 / (1 + r3)                      # (1-r^3)/(1-r^6) = 1/(1+r^3)
    r3s = r(float(r3), 4)
    pw_disp = 1 / (1 + r3s)
    return {
        "p": p, "q": q, "i": i, "N": N, "rr": s(float(rr), 4), "rfr": frac(rr),
        "r3": s(r3s, 4), "r6": s(float(r6), 4), "den": s(1 + r3s, 4),
        "pw": r(pw, 3), "pw5": s(pw_disp, 5), "pws": s(pw, 3), "pfr": frac(pw),
        "w_half": s(i / N, 3), "w_inv": s(1 / (1 + float(1 / rr) ** 3), 3),
    }




# ---------------------------------------------------------------- 大問6 体操競技の採点
J2 = [2, 1, 4, 3, 6, 8, 5, 7]   # 審判2の順位（審判1は A〜H に 1〜8）


def p06():
    n = len(J2)
    d = [i + 1 - j for i, j in enumerate(J2)]
    sd2 = sum(x * x for x in d)
    rs = 1 - Fr(6 * sd2, n * (n * n - 1))
    out = {"nj": n, "sd2": sd2, "six": 6 * sd2, "den": n * (n * n - 1), "rs": r(float(rs), 3), "rs5": s(float(rs), 5),
           "rs_s": s(float(rs), 3), "w_nosix": s(1 - sd2 / (n * (n * n - 1)), 3), "w_n2": s(1 - 6 * sd2 / (n * (n * n + 1)), 3)}
    for i, v in enumerate(J2):
        out[f"j{i + 1}"] = v
        out[f"d{i + 1}"] = d[i]
    # (b) ラテン方格
    k = 4
    sst, ssr, ssc, sstr = 70.0, 20.0, 8.0, 30.0
    sse = sst - ssr - ssc - sstr
    dfe = (k - 1) * (k - 2)
    mstr = sstr / (k - 1)
    mse = sse / dfe
    F = mstr / mse
    sse_nc = sst - ssr - sstr
    dfe_nc = k * k - 1 - 2 * (k - 1)
    out.update({"k": k, "sst": s(sst, 1), "ssr": s(ssr, 1), "ssc": s(ssc, 1), "sstr": s(sstr, 1), "sse": s(sse, 1),
                "dfk": k - 1, "dfe": dfe, "dft": k * k - 1, "mstr": s(mstr, 1), "mse": s(mse, 1), "F": s(F, 2),
                "msr": s(ssr / (k - 1), 3), "Fss": s(sstr / sse, 2), "Fnc": s(mstr / (sse_nc / dfe_nc), 2),
                "Fdf9": s(mstr / (sse / 9), 2), "Frow": s(mstr / (ssr / (k - 1)), 2),
                "ssenc": s(sse_nc, 1), "dfenc": dfe_nc, "msenc": s(sse_nc / dfe_nc, 3),
                "f36": 4.76, "f39": 3.86})
    # (c) 等分散の F 検定
    n1, s1, n2, s2 = 11, 0.48, 16, 0.30
    Fv = s1 ** 2 / s2 ** 2
    out.update({"n1": n1, "sP": s1, "n2": n2, "sQ": s2, "v1": s(s1 ** 2, 4), "v2": s(s2 ** 2, 2), "Fv": s(Fv, 2),
                "df1": n1 - 1, "df2": n2 - 1, "Fsd": s(s1 / s2, 2),
                "Fmean": s((s1 ** 2 / n1) / (s2 ** 2 / n2), 2), "f025": 3.06, "f05": 2.54})
    return out


# ---------------------------------------------------------------- 大問7 サッカー選手の年俸
def p07():
    b0, bg, bm, bf = 2.10, 0.045, 0.30, 0.55
    vec = lambda a, b, c, d: f"({a},\\ {b},\\ {c},\\ {d})"
    # 基準を FW に：切片 b0+bf、D_DF の係数 -bf、D_MF の係数 bm-bf
    out = {"b0": s(b0, 2), "bg": s(bg, 3), "bm": s(bm, 2), "bf": s(bf, 2),
           "nb0": s(b0 + bf, 2), "ndf": s(-bf, 2), "nmf": s(bm - bf, 2),
           "d_ok": vec(s(b0 + bf, 2), s(bg, 3), s(-bf, 2), s(bm - bf, 2)),
           "d_int": vec(s(b0, 2), s(bg, 3), s(-bf, 2), s(bm - bf, 2)),
           "d_mf": vec(s(b0 + bf, 2), s(bg, 3), s(-bf, 2), s(bm, 2)),
           "d_sign": vec(s(b0 + bf, 2), s(bg, 3), s(bf, 2), s(bf - bm, 2)),
           "d_add": vec(s(b0 + bf, 2), s(bg, 3), s(-bf, 2), s(-bm - bf, 2))}
    # (b) リッジ回帰（標準化した 2 変数）
    rho, r1, r2, lam = 0.6, 0.50, 0.45, 0.4
    a = 1 + lam
    det = a * a - rho ** 2
    rb1 = (a * r1 - rho * r2) / det
    rb2 = (a * r2 - rho * r1) / det
    det0 = 1 - rho ** 2
    ob1 = (r1 - rho * r2) / det0
    ob2 = (r2 - rho * r1) / det0
    pair = lambda x, y: f"(\\hat\\beta_1,\\ \\hat\\beta_2)=({s(x, 3)},\\ {s(y, 3)})"
    out.update({"rho": rho, "r1": r1, "r2": r2, "lam": lam, "a": s(a, 1), "det": s(det, 2), "det0": s(det0, 2),
                "a2": s(a * a, 2), "rho2": s(rho ** 2, 2),
                "n1": s(a * r1 - rho * r2, 2), "n2": s(a * r2 - rho * r1, 2), "rb1": s(rb1, 5), "rb2": s(rb2, 5),
                "ar1": s(a * r1, 2), "rr2": s(rho * r2, 2), "ar2": s(a * r2, 2), "rr1": s(rho * r1, 2),
                "ob1": s(ob1, 4), "ob2": s(ob2, 4),
                "c_ok": pair(rb1, rb2), "c_ols": pair(ob1, ob2), "c_div": pair(ob1 / a, ob2 / a),
                "c_r": pair(r1 / a, r2 / a), "c_mul": pair(rb1 * a, rb2 * a)})
    return out


# ---------------------------------------------------------------- 大問8 サッカーの試合
def p08():
    lh, la = 1.5, 1.0
    tot, k = 3, 2
    pr = lh / (lh + la)
    from math import comb
    ans = comb(tot, k) * pr ** k * (1 - pr) ** (tot - k)
    # (b) ロジスティック回帰の限界効果
    b, p = 0.05, 0.30
    me = b * p * (1 - p)
    eb = 1.0513
    return {"lh": lh, "la": la, "lsum": s(lh + la, 1), "tot": tot, "k": k, "pr": s(pr, 1), "qr": s(1 - pr, 1),
            "comb": comb(tot, k), "ans": r(ans, 3), "ans_s": s(ans, 3),
            "w_uncond": s(1.5 ** 2 / 2 * 0.2231, 3), "w_k3": s(pr ** 2 * (1 - pr), 3),
            "e25": 0.0821, "pp": s(pr ** 2, 2),
            "b": b, "p": p, "pq": s(p * (1 - p), 2), "me": s(me, 4), "eb": eb,
            "w_4": s(b / 4, 4), "w_bp": s(b * p, 4), "w_b": s(b, 4), "w_eb": s(eb - 1, 4),
            "orr": s(eb, 4)}


# ---------------------------------------------------------------- 大問9 eスポーツの練習と成績
XS = [2, 4, 6, 8, 10]
ES = [0.6, -1.0, -0.3, 1.2, -0.5]
B0, B1 = 1.0, 0.85


def p09():
    n = len(XS)
    xb = sum(XS) / n
    dev = [x - xb for x in XS]
    sxx = sum(d * d for d in dev)
    assert abs(sum(ES)) < 1e-12 and abs(sum(d * e for d, e in zip(dev, ES))) < 1e-12
    ys = [r(B0 + B1 * x + e, 2) for x, e in zip(XS, ES)]
    terms = [d * d * e * e for d, e in zip(dev, ES)]
    num = sum(terms)
    v = num / sxx ** 2
    se = sqrt(v)
    s2 = sum(e * e for e in ES) / (n - 2)
    se_cl = sqrt(s2 / sxx)
    out = {"nobs": n, "xb": s(xb, 0), "sxx": s(sxx, 0), "sxx2": s(sxx ** 2, 0), "num": s(num, 2), "v": s(v, 4),
           "se": r(se, 3), "se5": s(se, 5), "se_s": s(se, 3), "b0": B0, "b1": B1,
           "s2": s(s2, 5), "se_cl": s(se_cl, 3), "w_n": s(sqrt(num / sxx ** 2 * n / (n - 2)), 3),
           "w_nosq": s(sqrt(num / sxx), 3), "sse": s(sum(e * e for e in ES), 2)}
    for i, (x, y, e, d, t) in enumerate(zip(XS, ys, ES, dev, terms), 1):
        out.update({f"x{i}": x, f"y{i}": s(y, 1), f"e{i}": s(e, 1), f"dv{i}": s(d, 0), f"t{i}": s(t, 2)})
    return out


# ---------------------------------------------------------------- 大問10 動画配信サービス
def p10():
    N = [12000, 6000, 2000]
    n_all = 200
    Nt = sum(N)
    W = [x / Nt for x in N]
    nh = [round(n_all * w) for w in W]
    ybar = [5, 15, 40]
    sd = [3, 6, 15]
    mean = sum(w * y for w, y in zip(W, ybar))
    terms = [w * w * s_ * s_ / m for w, s_, m in zip(W, sd, nh)]
    var = sum(terms)
    se = sqrt(var)
    out = {"Nt": Nt, "nall": n_all, "mean": s(mean, 1), "var": s(var, 4), "se": s(se, 3), "se5": s(se, 5),
           "w_var": s(var, 3), "w_ney": s(sum(w * s_ for w, s_ in zip(W, sd)) / sqrt(n_all), 3),
           "w_w1": s(sqrt(sum(w * s_ * s_ / m for w, s_, m in zip(W, sd, nh))), 3),
           "w_now": s(sqrt(sum(s_ * s_ / m for s_, m in zip(sd, nh))), 3),
           "sumws2": s(sum(w * s_ * s_ for w, s_ in zip(W, sd)), 1)}
    for i in range(3):
        out.update({f"N{i + 1}": N[i], f"W{i + 1}": s(W[i], 1), f"n{i + 1}": nh[i], f"y{i + 1}": ybar[i],
                    f"s{i + 1}": sd[i], f"W2{i + 1}": s(W[i] ** 2, 2), f"vt{i + 1}": s(terms[i], 4),
                    f"wy{i + 1}": s(W[i] * ybar[i], 1), f"ws{i + 1}": s(W[i] * sd[i] ** 2, 1)})
    # (b) RMSEA と CFI
    chi, df, nobs = 85.2, 32, 301
    chib, dfb = 620.0, 45
    rmsea = sqrt(max(chi - df, 0) / (df * (nobs - 1)))
    cfi = 1 - (chi - df) / (chib - dfb)
    pair = lambda a, b: f"\\mathrm{{RMSEA}}={a},\\ \\mathrm{{CFI}}={b}"
    out.update({"chi": s(chi, 1), "df": df, "nobs": nobs, "chib": s(chib, 1), "dfb": dfb,
                "cd": s(chi - df, 1), "dn": df * (nobs - 1), "rin": s((chi - df) / (df * (nobs - 1)), 7),
                "rmsea": s(rmsea, 3), "rmsea5": s(rmsea, 5), "cbd": s(chib - dfb, 1), "cfi": s(cfi, 3), "cfi5": s(cfi, 5),
                "r_nodf": s(sqrt(chi / (df * (nobs - 1))), 3), "cfi_nodf": s(1 - chi / chib, 3),
                "r_nosq": s((chi - df) / (df * (nobs - 1)), 4),
                "c_ok": pair(s(rmsea, 3), s(cfi, 3)), "c_r": pair(s(sqrt(chi / (df * (nobs - 1))), 3), s(cfi, 3)),
                "c_c": pair(s(rmsea, 3), s(1 - chi / chib, 3)),
                "c_rc": pair(s(sqrt(chi / (df * (nobs - 1))), 3), s(1 - chi / chib, 3)),
                "c_sq": pair(s((chi - df) / (df * (nobs - 1)), 4), s(cfi, 3))})
    # (c) ARMA(1,1)
    ph, th = 0.5, 0.4
    num_ = (1 + ph * th) * (ph + th)
    den_ = 1 + 2 * ph * th + th * th
    r1 = num_ / den_
    r2 = ph * r1
    pr = lambda a, b: f"(\\rho_1,\\ \\rho_2)=({s(a, 3)},\\ {s(b, 3)})"
    out.update({"ph": ph, "th": th, "pt": s(ph * th, 1), "a1": s(1 + ph * th, 1), "a2": s(ph + th, 1),
                "an": s(num_, 2), "ad": s(den_, 2), "ar1": s(r1, 5), "ar2": s(r2, 5),
                "a_ok": pr(r1, r2), "a_ar": pr(ph, ph ** 2), "a_sum": pr(ph + th, ph * (ph + th)),
                "a_sq": pr(r1, r1 ** 2), "a_add": pr(ph + th / (1 + th * th), ph * (ph + th / (1 + th * th)))})
    return out


# ---------------------------------------------------------------- 大問11 オンライン対戦ゲーム
def p11():
    r12, r13, r23 = 0.6, 0.5, 0.4
    det = 1 + 2 * r12 * r13 * r23 - r12 ** 2 - r13 ** 2 - r23 ** 2
    l1 = 2.004                  # 問題文に示す最大固有値
    ssum = 3 - l1
    prod = r(det / l1, 6)
    disc = r(ssum ** 2 - 4 * prod, 6)
    sq = r(sqrt(disc), 5)
    l2 = (ssum + sq) / 2
    l3 = (ssum - sq) / 2
    pct = lambda x: s(100 * x, 1) + "\\%"
    out = {"r12": r12, "r13": r13, "r23": r23, "det": s(det, 2), "l1": l1, "ssum": s(ssum, 3), "prod": s(prod, 6),
           "ssum2": s(ssum ** 2, 6), "disc": s(disc, 6), "sq": s(sq, 5), "l2": s(l2, 4), "l3": s(l3, 4),
           "cum": s((l1 + l2) / 3, 4), "c_ok": pct((l1 + l2) / 3), "c_1": pct(l1 / 3), "c_eq": pct((l1 + ssum / 2) / 3),
           "c_2": pct(l2 / 3), "c_all": pct((l1 + ssum) / 3), "rr": s(2 * r12 * r13 * r23, 2),
           "q12": s(r12 ** 2, 2), "q13": s(r13 ** 2, 2), "q23": s(r23 ** 2, 2)}
    # (b) ジニ不純度
    n, c = 40, 10
    nl, cl = 12, 8
    nr, cr = n - nl, c - cl
    gini = lambda k, m: 1 - Fr(k, m) ** 2 - Fr(m - k, m) ** 2
    g0, gl, gr = gini(c, n), gini(cl, nl), gini(cr, nr)
    wg = Fr(nl, n) * gl + Fr(nr, n) * gr
    dec = g0 - wg
    ent = None
    out.update({"n": n, "c": c, "nl": nl, "cl": cl, "nr": nr, "cr": cr,
                "g0": s(float(g0), 4), "gl": s(float(gl), 4), "gr": s(float(gr), 4),
                "wl": s(float(Fr(nl, n) * gl), 4), "wr": s(float(Fr(nr, n) * gr), 4), "wg": s(float(wg), 4),
                "dec": r(float(dec), 3), "dec5": s(float(dec), 5), "dec_s": s(float(dec), 3),
                "w_nowt": s(float(g0 - (gl + gr) / 2), 3), "w_sum": s(float(g0 - gl - gr), 3),
                "p0": s(c / n, 2), "pl": frac(Fr(cl, nl)), "prr": frac(Fr(cr, nr))})
    return out


# ---------------------------------------------------------------- 大問12 ゲームアプリの継続
LR = [  # (日, n_A, d_A, n_B, d_B)
    (2, 10, 1, 10, 3),
    (5, 9, 0, 7, 2),
    (9, 8, 1, 4, 1),
    (14, 6, 1, 3, 2),
]


def p12():
    out = {}
    EA = V = VB = Fr(0)
    OA = dtot = 0
    for i, (day, na, da, nb, db) in enumerate(LR, 1):
        n, d = na + nb, da + db
        e = Fr(d * na, n)
        v = Fr(d * na * nb * (n - d), n * n * (n - 1))
        vb = Fr(d * na * nb, n * n)
        EA += e
        V += v
        VB += vb
        OA += da
        dtot += d
        out.update({f"day{i}": day, f"na{i}": na, f"da{i}": da, f"nb{i}": nb, f"db{i}": db, f"n{i}": n, f"d{i}": d,
                    f"e{i}": s(float(e), 4), f"v{i}": s(float(v), 4)})
    EAs = r(float(EA), 4)
    Vs = r(float(V), 4)
    diff = OA - EAs
    chi = diff ** 2 / Vs
    EB = dtot - EAs
    OB = dtot - OA
    chi_e = diff ** 2 / EAs + (OB - EB) ** 2 / EB
    chi_b = diff ** 2 / r(float(VB), 4)
    zz = abs(diff) / sqrt(Vs)
    out.update({"EA": s(EAs, 4), "OA": OA, "V": s(Vs, 4), "diff": s(diff, 4), "diff2": s(diff ** 2, 4),
                "chi": s(chi, 2), "chi4": s(chi, 4), "chi_e": s(chi_e, 2), "chi_b": s(chi_b, 2), "z": s(zz, 2),
                "EB": s(EB, 4), "OB": OB, "VB": s(float(VB), 4), "dtot": dtot, "c1": 3.841, "c2": 5.991})
    # (b) ブートストラップ
    n = 10
    q10 = 0.3487            # 0.9^10（問題文に示す値）
    out.update({"bn": n, "q10": q10, "bs": s(n * (1 - q10), 2), "bs_out": s(n * q10, 2),
                "bs_e": s(n * (1 - 0.3679), 2), "bs_one": s(n * 0.9, 2), "bs_all": s(n, 2), "inc": s(1 - q10, 4)})
    return out


def values():
    return {"s14p01": p01(), "s14p02": p02(), "s14p03": p03(), "s14p04": p04(), "s14p05": p05(),
            "s14p06": p06(), "s14p07": p07(), "s14p08": p08(), "s14p09": p09(), "s14p10": p10(),
            "s14p11": p11(), "s14p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
