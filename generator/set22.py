"""模擬試験 第22回（set22）の数値。values() が {大問id: {名前: 値}} を返す。

題材：物流倉庫・EC配送・在庫管理・コールセンター。
問題文に示した分布の値（指数関数の値、q 分布・t 分布・F 分布の点）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) 20 箱中 3 箱が破損。無作為な順に開けて 3 箱目で初めて破損箱
    N, D = 20, 3
    G = N - D
    pa = (G / N) * ((G - 1) / (N - 1)) * (D / (N - 2))
    with_rep = (G / N) ** 2 * (D / N)
    atleast = 1 - (G / N) * ((G - 1) / (N - 1)) * ((G - 2) / (N - 2))
    # (b) 破損率の推定に必要な標本数
    p0, e, z = 0.15, 0.03, 1.96
    nraw = z ** 2 * p0 * (1 - p0) / e ** 2
    n_half = z ** 2 * 0.25 / e ** 2
    n_645 = 1.645 ** 2 * p0 * (1 - p0) / e ** 2
    n_width = z ** 2 * p0 * (1 - p0) / (2 * e) ** 2
    ceil = lambda x: int(-(-x // 1))
    return {
        "N": N, "D": D, "G": G, "G1": G - 1, "N1": N - 1, "N2": N - 2,
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "num": G * (G - 1) * D, "den": N * (N - 1) * (N - 2),
        "with_rep": s(with_rep, 4), "atleast": s(atleast, 4),
        "p0": s(p0, 2), "q0": s(1 - p0, 2), "e": s(e, 2), "z": s(z, 2), "z2": s(z ** 2, 4),
        "pq": s(p0 * (1 - p0), 4), "e2": s(e ** 2, 4), "nraw": s(nraw, 1), "nreq": ceil(nraw),
        "n_half": ceil(n_half), "n_halfraw": s(n_half, 1), "n_645": ceil(n_645), "n_width": ceil(n_width),
        "nlow": int(nraw),
    }


def p02():
    # (a) リードタイム 2 日の需要 Po(3) が在庫 5 を超える確率
    lam, L, stock = 1.5, 2, 5
    mu = lam * L
    em3 = 0.0498                       # 問題文に示す e^{-3}
    terms = [1.0]
    for k in range(1, stock + 1):
        terms.append(terms[-1] * mu / k)
    ssum = sum(terms)
    pa = 1 - em3 * ssum
    # 誤答：P(D>=5)、1日分 Po(1.5)
    p_ge5 = 1 - em3 * (ssum - terms[-1])
    # (b) UMVU of e^{-λ}
    n, T, zero_days = 10, 12, 2
    umvu = ((n - 1) / n) ** T
    em12 = 0.3012                      # 問題文に示す e^{-1.2}
    # (c) メトロポリス法。目標 Po(1.2)、±1 提案、現在 x=3
    lh = T / n
    x = 3
    up = min(1.0, lh / (x + 1))
    down = min(1.0, x / lh)
    move = 0.5 * up + 0.5 * down
    inv = 0.5 * min(1.0, (x + 1) / lh) + 0.5 * min(1.0, lh / x)
    nofact = 0.5 * min(1.0, lh) + 0.5 * min(1.0, 1 / lh)
    return {
        "lam": s(lam, 1), "L": L, "stock": stock, "mu": s(mu, 0), "em3": s(em3, 4), "stock1": stock + 1,
        "t0": s(terms[0], 0), "t1": s(terms[1], 0), "t2": s(terms[2], 1), "t3": s(terms[3], 1),
        "t4": s(terms[4], 3), "t5": s(terms[5], 3), "tsum": s(ssum, 1), "cdf": s(em3 * ssum, 5),
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "p_ge5": s(p_ge5, 3),
        "n": n, "T": T, "zero_days": zero_days, "n1": n - 1,
        "umvu": s(umvu, 3), "umvu4": s(umvu, 4), "mle": s(em12, 3), "em12": s(em12, 4), "lh": s(lh, 1),
        "prop0": s(zero_days / n, 3), "w_T1": s(((n - 1) / n) ** (T - 1), 3), "w_rev": s((1 - 1 / T) ** n, 3),
        "ratio": s((n - 1) / n, 1),
        "x": x, "xp": x + 1, "xm": x - 1,
        "up": s(up, 2), "down_raw": s(x / lh, 1), "move": s(move, 2), "stay": s(1 - move, 2),
        "inv": s(inv, 2), "upraw": s(lh / (x + 1), 2), "onlyup": s(up, 2), "nofact": s(nofact, 2),
    }


def p03():
    # (a) 混合分布の積率母関数
    w1, m1 = 0.6, 1.0           # 指数分布（平均 1）
    w2, k2, th2 = 0.4, 2, 3.0   # ガンマ分布（形状 2、尺度 3）
    E1 = w1 * m1 + w2 * k2 * th2
    E2 = w1 * 2 * m1 ** 2 + w2 * k2 * (k2 + 1) * th2 ** 2
    V = E2 - E1 ** 2
    within = w1 * m1 ** 2 + w2 * k2 * th2 ** 2
    indep = w1 ** 2 * m1 ** 2 + w2 ** 2 * k2 * th2 ** 2
    # (b) 一様分布の和
    a1, b1, a2, b2, c = 10, 30, 5, 15, 38
    leg = (b1 + b2) - c
    area = leg ** 2 / 2
    tot = (b1 - a1) * (b2 - a2)
    pb = area / tot
    # (c) テューキー法
    means = {"A": 32.0, "B": 28.2, "C": 27.0, "D": 25.0}
    nn, mse, dfe, q, t = 6, 9.0, 20, 3.96, 2.086
    hsd = q * sqrt(mse / nn)
    lsd = t * sqrt(2 * mse / nn)
    hsd2 = q * sqrt(2 * mse / nn)
    return {
        "w1": s(w1, 1), "w2": s(w2, 1), "k2": k2, "th2": s(th2, 0), "k2p": k2 + 1,
        "E1": s(E1, 1), "E2": s(E2, 1), "E1sq": s(E1 ** 2, 1), "V": s(V, 1),
        "Vw": s(within, 1), "Vind": s(indep, 2), "M2mM1": s(E2 - E1, 1),
        "d1": s(w1 * m1, 1), "d2": s(w2 * k2 * th2, 1), "dd1": s(w1 * 2, 1), "dd2": s(w2 * k2 * (k2 + 1) * th2 ** 2, 1),
        "g2c": s(k2 * (k2 + 1) * th2 ** 2, 0), "vg": s(k2 * th2 ** 2, 0), "g1c": s(k2 * th2, 0),
        "a1": a1, "b1": b1, "a2": a2, "b2": b2, "c": c, "smin": a1 + a2, "smax": b1 + b2, "xlo": c - b2,
        "ylo": c - b1, "leg": leg, "area": s(area, 1), "tot": tot, "pb": s(pb, 4),
        "pb_noh": s(leg ** 2 / tot, 4), "pb_uni": s((b1 + b2 - c) / (b1 + b2 - a1 - a2), 4), "pb_comp": s(1 - pb, 4),
        "pb_400": s(area / 400, 4),
        "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1), "mD": s(means["D"], 1),
        "nn": nn, "N": 4 * nn, "mse": s(mse, 1), "dfe": dfe, "q": s(q, 2), "t": s(t, 3),
        "msen": s(mse / nn, 1), "se1": s(sqrt(mse / nn), 4), "hsd": s(hsd, 2), "lsd": s(lsd, 2), "hsd2": s(hsd2, 2),
        "lsd1": s(t * sqrt(mse / nn), 2), "qs": s(q * sqrt(mse), 2),
        "dAB": s(means["A"] - means["B"], 1), "dAC": s(means["A"] - means["C"], 1), "dAD": s(means["A"] - means["D"], 1),
        "dBC": s(means["B"] - means["C"], 1), "dBD": s(means["B"] - means["D"], 1), "dCD": s(means["C"] - means["D"], 1),
    }


def p04():
    # (a) 平均 6 分の指数分布 4 件の和が 30 分以内
    mean, k, tt = 6, 4, 30
    m = tt / mean
    em5 = 0.00674
    terms = [1.0]
    for j in range(1, k + 1):
        terms.append(terms[-1] * m / j)
    s3 = sum(terms[:k])
    pa = 1 - em5 * s3
    p_off = 1 - em5 * (s3 + terms[k])
    em125 = 0.2865048  # 誤答用（指数分布 1 個とみなした場合）。問題文には示さない
    # (b) M/M/2
    lam, mu_h, c = 15, 60 / mean, 2
    a = lam / mu_h
    rho = a / c
    tail = a ** 2 / 2 / (1 - rho)
    pi0 = 1 / (1 + a + tail)
    pw = tail * pi0
    # 誤答：2 本の M/M/1（各 λ/2）や M/M/1 の ρ
    mm1_rho = lam / mu_h if lam / mu_h < 1 else None
    # (c) デルタ法
    lc = 15
    return {
        "mean": mean, "k": k, "tt": tt, "km1": k - 1, "m": s(m, 0), "em5": s(em5, 5),
        "q0": s(terms[0], 0), "q1": s(terms[1], 0), "q2": s(terms[2], 1), "q3": s(terms[3], 4),
        "s3": s(s3, 4), "cdf": s(em5 * s3, 4), "pa": s(pa, 3), "pa4": s(pa, 4), "q4": s(terms[4], 4),
        "p_off": s(p_off, 3), "p_comp": s(em5 * s3, 3), "p_norm": s(0.691462, 3), "p_exp": s(1 - em125, 3),
        "emean": k * mean, "esd": s(sqrt(k) * mean, 0),
        "lam": lam, "mu": s(mu_h, 0), "c": c, "a": s(a, 1), "rho": s(rho, 2), "a2h": s(a ** 2 / 2, 3),
        "tail": s(tail, 1), "den": s(1 + a + tail, 1), "pi0": s(pi0, 4), "pi1": s(a * pi0, 4),
        "pw": r(pw, 3), "pw_s": s(pw, 3), "pw4": s(pw, 4), "pw_mm1": s(rho, 3),
        "pw_noq": s(1 - pi0 - a * pi0, 3),
        "lc": lc, "sq": s(sqrt(lc), 2), "dv": s(0.25, 3), "dv_nosq": s(1 / (2 * sqrt(lc)), 3),
        "dv_nolam": s(1 / (4 * lc), 4), "dv_sd": s(sqrt(lc), 2), "dv_raw": s(lc, 1),
        "gp": s(1 / (2 * sqrt(lc)), 4), "gp2": s(1 / (4 * lc), 5),
    }


def p05():
    # (a) ホルム法
    pv = {"A": 0.011, "B": 0.21, "C": 0.004, "D": 0.024, "E": 0.018}
    m, alpha = 5, 0.05
    srt = sorted(pv.items(), key=lambda kv: kv[1])
    holm = 0
    for i, (_, p) in enumerate(srt):
        if p <= alpha / (m - i):
            holm += 1
        else:
            break
    nonstop = sum(1 for i, (_, p) in enumerate(srt) if p <= alpha / (m - i))
    bonf = sum(1 for p in pv.values() if p <= alpha / m)
    raw = sum(1 for p in pv.values() if p <= alpha)
    # (b) 連の検定
    seq = "NNNNSSSNNNNNSSSSNNNS"
    n1 = seq.count("S")
    n2 = seq.count("N")
    N = len(seq)
    R = 1 + sum(1 for i in range(1, N) if seq[i] != seq[i - 1])
    ER = 1 + 2 * n1 * n2 / N
    VR = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))
    z = (R - ER) / sqrt(VR)
    VRn = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * N)
    runsS = sum(1 for i in range(N) if seq[i] == "S" and (i == 0 or seq[i - 1] != "S"))
    return {
        "pA": s(pv["A"], 3), "pB": s(pv["B"], 2), "pC": s(pv["C"], 3), "pD": s(pv["D"], 3), "pE": s(pv["E"], 3),
        "m": m, "a1": s(alpha / 5, 3), "a2": s(alpha / 4, 4), "a3": s(alpha / 3, 4), "a4": s(alpha / 2, 3),
        "holm": holm, "nonstop": nonstop, "bonf": bonf, "raw": raw,
        "seq": " ".join(seq), "n1": n1, "n2": n2, "NN": N, "R": R, "n12": 2 * n1 * n2,
        "ER": s(ER, 1), "VRnum": 2 * n1 * n2 * (2 * n1 * n2 - N), "VRden": N ** 2 * (N - 1), "VR": s(VR, 4),
        "sdR": s(sqrt(VR), 4), "z": s(z, 2), "z_n": s((R - ER) / sqrt(VRn), 2), "z_e": s((R - (ER - 1)) / sqrt(VR), 2),
        "z_v": s((R - ER) / VR, 2), "z_s": s((runsS - ER) / sqrt(VR), 2), "runsS": runsS, "diffR": s(R - ER, 1),
    }


def p06():
    n, p = 12, 2
    xbar = (63.0, 21.0)
    mu0 = (60.0, 20.0)
    S = ((16.0, 4.0), (4.0, 4.0))
    d = (xbar[0] - mu0[0], xbar[1] - mu0[1])
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    q = (S[1][1] * d[0] ** 2 - 2 * S[0][1] * d[0] * d[1] + S[0][0] * d[1] ** 2) / det
    T2 = n * q
    F = (n - p) / (p * (n - 1)) * T2
    qdiag = d[0] ** 2 / S[0][0] + d[1] ** 2 / S[1][1]
    Fdiag = (n - p) / (p * (n - 1)) * n * qdiag
    Fn = (n - p) / (p * n) * T2
    Fnon = (n - p) / (p * (n - 1)) * q
    return {
        "n": n, "p": p, "x1": s(xbar[0], 1), "x2": s(xbar[1], 1), "m1": s(mu0[0], 0), "m2": s(mu0[1], 0),
        "s11": s(S[0][0], 0), "s12": s(S[0][1], 0), "s22": s(S[1][1], 0), "d1": s(d[0], 0), "d2": s(d[1], 0),
        "det": s(det, 0), "qnum": s(q * det, 0), "q": s(q, 4), "T2": s(T2, 2), "np": n - p, "pn1": p * (n - 1),
        "F": s(F, 2), "Fc": "4.10", "Fdiag": s(Fdiag, 2), "Fn": s(Fn, 2), "Fnon": s(Fnon, 2), "qdiag": s(qdiag, 4),
        "T2diag": s(n * qdiag, 2),
    }


def p07():
    n = 40
    rss1, k1 = 520.0, 3   # 切片 + 2 変数
    rss2, k2 = 430.0, 6   # + 地域ダミー 3 個
    q = k2 - k1
    dfe = n - k2
    F = ((rss1 - rss2) / q) / (rss2 / dfe)
    F35 = ((rss1 - rss2) / q) / (rss2 / (dfe + 1))
    Fnoq = (rss1 - rss2) / (rss2 / dfe)
    F1 = ((rss1 - rss2) / q) / (rss1 / (n - k1))
    # (b) 外部スチューデント化残差
    e, h = 9.0, 0.30
    s2 = rss2 / dfe
    sd = sqrt(s2)
    sqh = sqrt(1 - h)
    rint = e / (sd * sqh)
    s2i = (rss2 - e ** 2 / (1 - h)) / (dfe - 1)
    sdi = sqrt(s2i)
    t = e / (sdi * sqh)
    return {
        "n": n, "rss1": s(rss1, 0), "rss2": s(rss2, 0), "k1": k1, "k2": k2, "q": q, "dfe": dfe, "dfe1": n - k1,
        "drss": s(rss1 - rss2, 0), "msq": s((rss1 - rss2) / q, 1), "mse": s(rss2 / dfe, 4),
        "F": r(F, 2), "F_s": s(F, 2), "F4": s(F, 4), "F35": s(F35, 2), "Fnoq": s(Fnoq, 2), "F1": s(F1, 2), "Fc": "2.88",
        "e": s(e, 1), "h": s(h, 2), "omh": s(1 - h, 2), "s2": s(s2, 4), "sd": s(sd, 4), "sqh": s(sqh, 4),
        "e2h": s(e ** 2 / (1 - h), 4), "s2inum": s(rss2 - e ** 2 / (1 - h), 4), "dfe_1": dfe - 1,
        "s2i": s(s2i, 4), "sdi": s(sdi, 4), "den_t": s(sdi * sqh, 4),
        "t": s(t, 2), "t4": s(t, 4), "rint": s(rint, 2), "e_s": s(e / sd, 2), "e_s1h": s(e / (sd * (1 - h)), 2),
        "e_si": s(e / sdi, 2),
    }


def p08():
    a1, a2, b, x = -2.0, 0.5, 0.8, 2
    l1 = a1 + b * x
    l2 = a2 + b * x
    e04 = 1.4918   # e^{0.4}
    em21 = 0.1225  # e^{-2.1}
    P1 = 1 / (1 + e04)
    P2 = 1 / (1 + em21)
    pmid = P2 - P1
    # 誤答：符号を逆（α - βx）にした場合など（値は誤答用に正確な指数で計算する）
    import math
    lg = lambda z: 1 / (1 + math.exp(-z))
    wrong_sign = lg(a2 - b * x) - lg(a1 - b * x)
    return {
        "a1": s(a1, 1), "a2": s(a2, 1), "b": s(b, 1), "x": x, "l1": s(l1, 1), "l2": s(l2, 1),
        "e04": s(e04, 4), "em21": s(em21, 4), "den1": s(1 + e04, 4), "den2": s(1 + em21, 4),
        "P1": s(P1, 4), "P2": s(P2, 4), "pmid": s(pmid, 3), "pmid4": s(pmid, 4),
        "P1s": s(P1, 3), "P2s": s(P2, 3), "P3s": s(1 - P2, 3), "wsign": s(wrong_sign, 3),
        "nl1": s(-l1, 1),
    }


def p09():
    sz, sx, sy = 5.0, 20.0, 30.0
    rzx, rzy, rxy = 0.5, -0.4, -0.5
    czx = rzx * sz * sx
    czy = rzy * sz * sy
    cxy = rxy * sx * sy
    iv = czy / czx
    ols = cxy / sx ** 2
    g1 = czx / sz ** 2
    pi = czy / sz ** 2
    return {
        "sz": s(sz, 0), "sx": s(sx, 0), "sy": s(sy, 0), "rzx": s(rzx, 1), "rzy": s(rzy, 1), "rxy": s(rxy, 1),
        "czx": s(czx, 0), "czy": s(czy, 0), "cxy": s(cxy, 0), "vz": s(sz ** 2, 0), "vx": s(sx ** 2, 0),
        "iv": s(iv, 2), "iv_abs": s(-iv, 1), "ols": s(ols, 2), "g1": s(g1, 1), "pi": s(pi, 1), "rf": s(pi, 2),
        "inv": s(czx / czy, 2), "rr": s(rzy / rzx, 2),
    }


def p10():
    Npop, n = 500, 20
    Xbar, xbar, ybar = 120.0, 115.0, 110.0
    sxy, sxx = 360.0, 400.0
    b = sxy / sxx
    yreg = ybar + b * (Xbar - xbar)
    tot = Npop * yreg
    yrat = ybar / xbar * Xbar
    return {
        "Npop": Npop, "n": n, "Xbar": s(Xbar, 0), "xbar": s(xbar, 0), "ybar": s(ybar, 0), "sxy": s(sxy, 0),
        "sxx": s(sxx, 0), "b": s(b, 1), "dx": s(Xbar - xbar, 0), "adj": s(b * (Xbar - xbar), 1),
        "yreg": s(yreg, 1), "tot": r(tot, 0), "tot_s": s(tot, 0), "tot_rat": s(Npop * yrat, 0),
        "tot_srs": s(Npop * ybar, 0), "tot_rev": s(Npop * (ybar - b * (Xbar - xbar)), 0),
        "tot_b1": s(Npop * (ybar + (Xbar - xbar)), 0), "tot_x": s(Npop * Xbar, 0), "yrat": s(yrat, 2),
    }


def p11():
    ev = [2.30, 1.20, 0.80, 0.45, 0.25]
    p = len(ev)
    cum = []
    c = 0
    for v in ev:
        c += v
        cum.append(c / p)
    kaiser = sum(1 for v in ev if v >= 1)
    k80 = next(i + 1 for i, v in enumerate(cum) if v >= 0.8)
    # (b) 直交回転
    l = (0.6, 0.5)
    cs, sn = 0.8, 0.6
    rot = (l[0] * cs + l[1] * sn, -l[0] * sn + l[1] * cs)
    rotT = (l[0] * cs - l[1] * sn, l[0] * sn + l[1] * cs)
    h2 = l[0] ** 2 + l[1] ** 2
    h2r = rot[0] ** 2 + rot[1] ** 2
    return {
        "ev1": s(ev[0], 2), "ev2": s(ev[1], 2), "ev3": s(ev[2], 2), "ev4": s(ev[3], 2), "ev5": s(ev[4], 2), "p": p,
        "c1": s(cum[0] * 100, 0), "c2": s(cum[1] * 100, 0), "c3": s(cum[2] * 100, 0), "c4": s(cum[3] * 100, 0),
        "kaiser": kaiser, "k80": k80,
        "l1": s(l[0], 1), "l2": s(l[1], 1), "cs": s(cs, 1), "sn": s(sn, 1),
        "r1": s(rot[0], 2), "r2": s(rot[1], 2), "rt1": s(rotT[0], 2), "rt2": s(rotT[1], 2),
        "h2": s(h2, 2), "u2": s(1 - h2, 2), "h2r": s(h2r, 2), "r1sq": s(rot[0] ** 2, 4), "r2sq": s(rot[1] ** 2, 4),
        "u_sum": s(1 - (rot[0] + rot[1]), 2), "lsum": s(rot[0] + rot[1], 2),
        "a11": s(l[0] * cs, 2), "a12": s(l[1] * sn, 2), "a21": s(-l[0] * sn, 2), "a22": s(l[1] * cs, 2),
    }


def p12():
    ybar = 20.0
    reg = {"都市": (60, 0.0), "郊外": (30, 3.0), "離島": (10, 12.0)}
    tim = {"なし": (70, 0.0), "あり": (30, 2.0)}
    nreg = sum(v[0] for v in reg.values())
    mreg = sum(v[0] * v[1] for v in reg.values()) / nreg
    mtim = sum(v[0] * v[1] for v in tim.values()) / nreg
    sc = {k: v[1] - mreg for k, v in reg.items()}
    st = {k: v[1] - mtim for k, v in tim.items()}
    pred = ybar + sc["郊外"] + st["あり"]
    ureg = sum(v[1] for v in reg.values()) / 3
    utim = sum(v[1] for v in tim.values()) / 2
    return {
        "ybar": s(ybar, 1), "nU": reg["都市"][0], "nS": reg["郊外"][0], "nI": reg["離島"][0],
        "bS": s(reg["郊外"][1], 1), "bI": s(reg["離島"][1], 1), "nN": tim["なし"][0], "nY": tim["あり"][0],
        "bY": s(tim["あり"][1], 1), "nall": nreg,
        "mreg": s(mreg, 1), "mtim": s(mtim, 1), "scU": s(sc["都市"], 1), "scS": s(sc["郊外"], 1), "scI": s(sc["離島"], 1),
        "stN": s(st["なし"], 1), "stY": s(st["あり"], 1), "pred": s(pred, 1), "c0": s(ybar - mreg - mtim, 1),
        "w_raw": s(ybar + reg["郊外"][1] + tim["あり"][1], 1),
        "w_unw": s(ybar + reg["郊外"][1] - ureg + tim["あり"][1] - utim, 1),
        "w_cat": s(ybar + sc["郊外"] + st["なし"], 1), "w_noint": s(sc["郊外"] + st["あり"], 1),
        "ureg": s(ureg, 1), "utim": s(utim, 1),
    }


def p13():
    f1, f2 = 0.5, 0.3
    rho1 = f1 / (1 - f2)
    rho2 = f1 * rho1 + f2
    pacf2 = (rho2 - rho1 ** 2) / (1 - rho1 ** 2)
    return {
        "f1": s(f1, 1), "f2": s(f2, 1), "fs": s(f1 + f2, 1), "fd": s(f2 - f1, 1), "omf2": s(1 - f2, 1),
        "rho1": s(rho1, 3), "rho2": s(rho2, 3), "pacf2": s(pacf2, 3), "rho1sq": s(rho1 ** 2, 4),
        "rho2_4": s(rho2, 4), "pnum": s(rho2 - rho1 ** 2, 4), "pden": s(1 - rho1 ** 2, 4),
        "rho1_plus": s(f1 / (1 + f2), 3),
    }


def p14():
    # 層 A, B：(新・誤, 新・正, 旧・誤, 旧・正)
    A = (10, 190, 30, 270)
    B = (6, 394, 8, 192)
    nA, nB = sum(A), sum(B)
    adA, bcA = A[0] * A[3] / nA, A[1] * A[2] / nA
    adB, bcB = B[0] * B[3] / nB, B[1] * B[2] / nB
    mh = (adA + adB) / (bcA + bcB)
    a, b, c, d = (A[i] + B[i] for i in range(4))
    crude = a * d / (b * c)
    orA = A[0] * A[3] / (A[1] * A[2])
    orB = B[0] * B[3] / (B[1] * B[2])
    return {
        "Aa": A[0], "Ab": A[1], "Ac": A[2], "Ad": A[3], "Ba": B[0], "Bb": B[1], "Bc": B[2], "Bd": B[3],
        "nA": nA, "nB": nB, "nAn": A[0] + A[1], "nAo": A[2] + A[3], "nBn": B[0] + B[1], "nBo": B[2] + B[3],
        "adA": s(adA, 2), "bcA": s(bcA, 2), "adB": s(adB, 2), "bcB": s(bcB, 4),
        "sad": s(adA + adB, 2), "sbc": s(bcA + bcB, 4),
        "mh": r(mh, 2), "mh_s": s(mh, 2), "mh4": s(mh, 4), "crude": s(crude, 2), "orA": s(orA, 2), "orB": s(orB, 2),
        "avg": s((orA + orB) / 2, 2), "inv": s(1 / mh, 2),
        "ta": a, "tb": b, "tc": c, "td": d,
    }


def values():
    return {"s22p01": p01(), "s22p02": p02(), "s22p03": p03(), "s22p04": p04(), "s22p05": p05(),
            "s22p06": p06(), "s22p07": p07(), "s22p08": p08(), "s22p09": p09(), "s22p10": p10(),
            "s22p11": p11(), "s22p12": p12(), "s22p13": p13(), "s22p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
