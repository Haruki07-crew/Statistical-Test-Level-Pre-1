"""模擬試験 第8回（set08）の数値。values() が {大問id: {名前: 値}} を返す。

題材：Web・IT 運用・通信・ソフトウェア品質。
問題文に示した関数値（F・χ²・e^x・ロジスティック関数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from math import comb, sqrt


def r(x, d):
    """四捨五入（銀行丸めを避ける）。数値で返す。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return float(v) if d > 0 else int(v)


def s(x, d):
    """四捨五入した文字列（末尾の 0 も残す。−0 は 0 にする）。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return str(v)


def distinct(*xs):
    assert len(set(xs)) == len(xs), xs


def p01():
    # (a) 誕生日問題型のハッシュ衝突（指数近似）
    N, k = 1000, 20
    pairs = k * (k - 1) // 2
    ex = {0.19: 0.8270, 0.20: 0.8187, 0.38: 0.6839, 0.02: 0.9802}
    lam = pairs / N
    assert abs(lam - 0.19) < 1e-12
    ok = 1 - ex[0.19]
    opts = [s(k / N, 3), s(ok, 3), s(1 - ex[0.20], 3), s(lam, 3), s(1 - ex[0.38], 3)]
    distinct(*opts)
    # (b) M/M/2
    lm, mu, c = 3, 2, 2
    a = lm / mu
    rho = lm / (c * mu)
    tail = a ** 2 / (2 * (1 - rho))
    p0 = 1 / (1 + a + tail)
    pw = tail * p0
    p2 = a ** 2 / 2 * p0
    bopts = [s(p2, 3), s(rho ** 2, 3), s(pw, 3), s(rho, 3), s(1 - p0, 3)]
    distinct(*bopts)
    return {
        "N": N, "k": k, "pairs": pairs, "lam": s(lam, 2), "e019": s(ex[0.19], 4), "e020": s(ex[0.20], 4),
        "e038": s(ex[0.38], 4), "e002": s(ex[0.02], 4),
        "o_kn": opts[0], "o_ok": opts[1], "o_sq": opts[2], "o_union": opts[3], "o_noh": opts[4],
        "lm": lm, "mu": mu, "a": s(a, 1), "rho": s(rho, 2), "a2": s(a ** 2, 2), "a22": s(a ** 2 / 2, 3),
        "tail": s(tail, 1), "den": s(1 + a + tail, 1), "p0d": s(p0, 4),
        "pw4": s(pw, 4), "wq": s(pw / (c * mu - lm), 3), "cmu": c * mu, "omr": s(1 - rho, 2),
        "b_p2": bopts[0], "b_r2": bopts[1], "b_ok": bopts[2], "b_rho": bopts[3], "b_busy": bopts[4],
    }


def p02():
    # (a) 一様遅延の差（到着順の入れ替わり）
    L, gap = 10, 2
    ok = (1 - gap / L) ** 2 / 2
    opts = [s(ok / 2, 2), s(ok, 2), s((L - gap) / (2 * L), 2), s(0.5, 2), s(2 * ok, 2)]
    distinct(*opts)
    # (b) アーラン分布（5 個目の到着）
    rate, kk, t = 2, 5, 1.5
    m = rate * t
    e3 = 0.0498
    terms = [m ** j / math.factorial(j) for j in range(kk)]
    ssum = sum(terms)
    ans = 1 - e3 * ssum
    # (c) 偶数回の切り替え
    lf = 0.8
    ev = {0.8: 0.4493, 1.6: 0.2019}
    even = (1 + ev[1.6]) / 2
    copts = [s(ev[0.8], 3), s(even, 3), s(0.5, 3), s((1 - ev[1.6]) / 2, 3), s((1 + ev[0.8]) / 2, 3)]
    distinct(*copts)
    return {
        "L": L, "gap": gap, "ratio": s(gap / L, 1), "omr": s(1 - gap / L, 1), "omr2": s((1 - gap / L) ** 2, 2),
        "rest": L - gap, "area": s((L - gap) ** 2 / 2, 0), "sq": L * L,
        "o_q": opts[0], "o_ok": opts[1], "o_lin": opts[2], "o_half": opts[3], "o_two": opts[4],
        "rate": rate, "kk": kk, "t": t, "m": s(m, 0), "e3": s(e3, 4),
        "t2": s(terms[2], 1), "t3": s(terms[3], 1), "t4": s(terms[4], 3),
        "tsum": s(ssum, 3), "prod": s(e3 * ssum, 6), "erl": r(ans, 3), "erl_s": s(ans, 3), "erl6": s(ans, 6),
        "w_eq": s(e3 * terms[4], 3), "w_le": s(1 - e3 * (ssum + m ** 5 / 120), 3),
        "lf": lf, "e08": s(ev[0.8], 4), "e16": s(ev[1.6], 4), "twol": s(2 * lf, 1),
        "c_zero": copts[0], "c_ok": copts[1], "c_half": copts[2], "c_odd": copts[3], "c_wrong": copts[4],
        "even4": s(even, 4),
    }


def p03():
    s1, s2, rho = 4.0, 3.0, 0.5
    cov = rho * s1 * s2
    den = s1 ** 2 + s2 ** 2 - 2 * cov
    w = (s2 ** 2 - cov) / den
    vmin = (s1 ** 2 * s2 ** 2 - cov ** 2) / den
    w0 = s2 ** 2 / (s1 ** 2 + s2 ** 2)
    v0 = s1 ** 2 * s2 ** 2 / (s1 ** 2 + s2 ** 2)
    return {
        "s1": s(s1, 0), "s2": s(s2, 0), "rho": rho, "v1": s(s1 ** 2, 0), "v2": s(s2 ** 2, 0), "cov": s(cov, 0),
        "den": s(den, 0), "w": s(w, 4), "w_frac": f"\\frac{{{s(s2 ** 2 - cov, 0)}}}{{{s(den, 0)}}}",
        "vnum": s(s1 ** 2 * s2 ** 2 - cov ** 2, 0), "vmin": r(vmin, 2), "vmin4": s(vmin, 4), "vmin_s": s(vmin, 2),
        "w0": s(w0, 2), "v0": s(v0, 2), "v_half": s((s1 ** 2 + s2 ** 2 + 2 * cov) / 4, 2),
        "p1v1": s(s1 ** 2 * s2 ** 2, 0), "cov2": s(cov ** 2, 0), "w1": s(1 - w, 4),
    }


def p04():
    # (a) 最大値（極値）の近似
    n, mean, thr = 1000, 50, 400
    lt = thr / mean
    e8, e0335 = 0.000335, 0.7153
    ne = n * e8
    assert abs(ne - 0.335) < 1e-12
    ans = 1 - e0335
    # (b) e^{-λ} の UMVUE
    cnt = [0, 2, 1, 0, 0, 3, 1, 0, 1, 0]
    nn, T = len(cnt), sum(cnt)
    pw = {8: 0.43047, 9: 0.38742}
    e08 = 0.4493
    bopts = [s(pw[9], 2), s(pw[8], 2), s(e08, 2), s(cnt.count(0) / nn, 2), s(T / nn, 2)]
    distinct(*bopts)
    # (c) ウィルソンの区間
    x, m, z = 4, 50, 1.96
    p = x / m
    z2 = z * z
    cen = (p + z2 / (2 * m)) / (1 + z2 / m)
    half = z / (1 + z2 / m) * sqrt(p * (1 - p) / m + z2 / (4 * m * m))
    z1 = 1.645
    cen1 = (p + z1 ** 2 / (2 * m)) / (1 + z1 ** 2 / m)
    half1 = z1 / (1 + z1 ** 2 / m) * sqrt(p * (1 - p) / m + z1 ** 2 / (4 * m * m))
    wald = p + z * sqrt(p * (1 - p) / m)
    copts = [s(wald, 3), s(cen + half, 3), s(cen, 3), s(cen - half, 3), s(cen1 + half1, 3)]
    distinct(*copts)
    return {
        "n": n, "mean": mean, "thr": thr, "lt": s(lt, 0), "e8": s(e8, 6), "ne": s(ne, 3), "e0335": s(e0335, 4),
        "gum": r(ans, 3), "gum_s": s(ans, 3), "gum4": s(ans, 4), "w_union": s(ne, 3), "w_comp": s(e0335, 3),
        "cnt_row": " | ".join(str(c) for c in cnt), "nn": nn, "T": T, "nz": cnt.count(0), "nm1": nn - 1,
        "ratio": s((nn - 1) / nn, 1), "pw8": s(pw[8], 5), "pw9": s(pw[9], 5), "e08": s(e08, 4), "lhat": s(T / nn, 1),
        "b_9": bopts[0], "b_ok": bopts[1], "b_mle": bopts[2], "b_frac": bopts[3], "b_lam": bopts[4],
        "x": x, "m": m, "z": z, "z2": s(z2, 4), "p": s(p, 2), "z2n": s(z2 / m, 6), "z22n": s(z2 / (2 * m), 6),
        "den": s(1 + z2 / m, 6), "cnum": s(p + z2 / (2 * m), 6), "cen": s(cen, 4),
        "var1": s(p * (1 - p) / m, 6), "var2": s(z2 / (4 * m * m), 6), "rt": s(sqrt(p * (1 - p) / m + z2 / (4 * m * m)), 5),
        "fac": s(z / (1 + z2 / m), 5), "half": s(half, 4), "up4": s(cen + half, 4), "sew": s(sqrt(p * (1 - p) / m), 5),
        "c_wald": copts[0], "c_ok": copts[1], "c_cen": copts[2], "c_low": copts[3], "c_z1": copts[4],
    }


def p05():
    # (a) ホテリングの T²
    n, pdim = 20, 2
    d = (1.0, 0.5)
    S11, S12, S22 = 4.0, 1.0, 1.0
    det = S11 * S22 - S12 ** 2
    q = (S22 * d[0] ** 2 - 2 * S12 * d[0] * d[1] + S11 * d[1] ** 2) / det
    T2 = n * q
    F = (n - pdim) / (pdim * (n - 1)) * T2
    Fc, Fc19, chi2c, tc = 3.55, 3.52, 5.99, 2.093
    assert F < Fc and F < Fc19 and T2 > chi2c
    t1 = d[0] / sqrt(S11 / n)
    t2 = d[1] / sqrt(S22 / n)
    assert t1 > tc and t2 > tc
    qw = S11 * d[0] ** 2 + 2 * S12 * d[0] * d[1] + S22 * d[1] ** 2
    # (b) 符号検定
    nn, pos = 8, 6
    tail = sum(comb(nn, k) for k in range(pos, nn + 1))
    pv = tail / 2 ** nn
    p_eq = comb(nn, pos) / 2 ** nn
    p_up = sum(comb(nn, k) for k in range(pos + 1, nn + 1)) / 2 ** nn
    sopts = [s(p_up, 3), s(p_eq, 3), s(pv, 3), s(2 * pv, 3), s(1 - p_up, 3)]
    distinct(*sopts)
    return {
        "n": n, "pdim": pdim, "n1": n - 1, "np": n - pdim, "d1": s(d[0], 1), "d2": s(d[1], 1),
        "S11": s(S11, 0), "S12": s(S12, 0), "S22": s(S22, 0), "det": s(det, 0),
        "qnum": s(q * det, 2), "q": s(q, 4), "T2": s(T2, 2), "F": s(F, 2), "Fk": f"\\frac{{{n - pdim}}}{{{pdim * (n - 1)}}}",
        "Fc": Fc, "Fc19": Fc19, "chi2c": chi2c, "tc": tc, "t1": s(t1, 3), "t2": s(t2, 3),
        "T2w": s(n * qw, 1), "qw": s(qw, 2),
        "nn": nn, "pos": pos, "neg": nn - pos, "c6": comb(nn, 6), "c7": comb(nn, 7), "c8": comb(nn, 8),
        "tail": tail, "pow2": 2 ** nn, "pv4": s(pv, 4),
        "s_up": sopts[0], "s_eq": sopts[1], "s_ok": sopts[2], "s_two": sopts[3], "s_low": sopts[4],
    }


def p06():
    # (a) ダミー変数の基準の変更
    b0, bs, bf = 2.10, 0.30, 0.12
    ok = (b0 + bs, -bs, bf - bs)
    fmt = lambda t: f"切片 ${s(t[0], 2)}$、Chrome ${s(t[1], 2)}$、Firefox ${s(t[2], 2)}$"
    opts = [fmt((b0, -bs, bf - bs)), fmt((b0 + bs, bs, bf)), fmt(ok), fmt((b0 + bs, -bs, bf)), fmt((b0 + bs, -bs, -(bs + bf)))]
    distinct(*opts)
    # (b) ホワイトの頑健標準誤差（HC0）
    x = [1, 2, 3, 4, 5, 6]
    e = [0.1, 0.2, -0.1, -0.2, -0.6, 0.6]
    n = len(x)
    xb = sum(x) / n
    xc = [xi - xb for xi in x]
    assert abs(sum(e)) < 1e-12 and abs(sum(a * b for a, b in zip(xc, e))) < 1e-12
    Sxx = sum(v * v for v in xc)
    num = sum(a * a * b * b for a, b in zip(xc, e))
    hc0 = sqrt(num) / Sxx
    sse = sum(v * v for v in e)
    ols = sqrt(sse / (n - 2) / Sxx)
    # (c) リッジ回帰の有効自由度
    d2, lam = [9, 4, 1], 2
    df = sum(v / (v + lam) for v in d2)
    copts = [s(sum(lam / (v + lam) for v in d2), 2), s(sum(sqrt(v) / (sqrt(v) + lam) for v in d2), 2), s(df, 2),
             s(len(d2) / (1 + lam), 2), s(len(d2), 2)]
    distinct(*copts)
    return {
        "b0": s(b0, 2), "bs": s(bs, 2), "bf": s(bf, 2), "nb0": s(b0 + bs, 2), "nbf": s(bf - bs, 2), "nbc": s(-bs, 2),
        "o0": opts[0], "o1": opts[1], "o2": opts[2], "o3": opts[3], "o4": opts[4],
        "x_row": " | ".join(str(v) for v in x), "e_row": " | ".join(s(v, 1) for v in e), "nx": n, "xb": s(xb, 1),
        "xc_row": " | ".join(s(v, 1) for v in xc), "w_row": " | ".join(s(a * a * b * b, 4) for a, b in zip(xc, e)),
        "Sxx": s(Sxx, 1), "Sxx2": s(Sxx ** 2, 2), "num": s(num, 3), "vhc": s(num / Sxx ** 2, 6),
        "hc0": r(hc0, 3), "hc0_s": s(hc0, 3), "hc04": s(hc0, 4), "sse": s(sse, 2), "s2": s(sse / (n - 2), 3),
        "ols": s(ols, 3), "hc1": s(hc0 * sqrt(n / (n - 2)), 3), "w_noroot": s(num / Sxx ** 2, 4), "w_s": s(sqrt(num / Sxx), 3),
        "d1": d2[0], "d2": d2[1], "d3": d2[2], "lam": lam, "f1": s(d2[0] / (d2[0] + lam), 4), "f2": s(d2[1] / (d2[1] + lam), 4),
        "f3": s(d2[2] / (d2[2] + lam), 4), "df4": s(df, 4),
        "c_comp": copts[0], "c_sv": copts[1], "c_ok": copts[2], "c_eq": copts[3], "c_ols": copts[4],
    }


def p07():
    # (a) 回帰推定量
    xbar, ybar, Xbar, sxy, sx2 = 40, 12.0, 45, 20.0, 80.0
    b = sxy / sx2
    yreg = ybar + b * (Xbar - xbar)
    # (b) 順序ロジット
    k1, k2, eta = -1.0, 1.5, 0.5
    Lg = {-1.5: 0.1824, 1.0: 0.7311, -0.5: 0.3775, 2.0: 0.8808}
    F1, F2 = Lg[k1 - eta], Lg[k2 - eta]
    mid = F2 - F1
    opts = [s(F1, 3), s(1 - F2, 3), s(Lg[k2 + eta] - Lg[k1 + eta], 3), s(mid, 3), s(F2, 3)]
    distinct(*opts)
    return {"xbar": xbar, "ybar": s(ybar, 1), "Xbar": Xbar, "sxy": s(sxy, 1), "sx2": s(sx2, 1), "b": s(b, 2),
            "dx": Xbar - xbar, "adj": s(b * (Xbar - xbar), 2), "yreg": r(yreg, 2), "yreg_s": s(yreg, 2),
            "ratio_est": s(ybar * Xbar / xbar, 2), "w_rev": s(ybar + b * (xbar - Xbar), 2),
            "k1": s(k1, 1), "k2": s(k2, 1), "eta": s(eta, 1), "z1": s(k1 - eta, 1), "z2": s(k2 - eta, 1),
            "L15": s(Lg[-1.5], 4), "L10": s(Lg[1.0], 4), "L05": s(Lg[-0.5], 4), "L20": s(Lg[2.0], 4),
            "o_low": opts[0], "o_high": opts[1], "o_sign": opts[2], "o_ok": opts[3], "o_cum": opts[4],
            "mid4": s(mid, 4)}


def p08():
    y = {"1": 120, "a": 96, "b": 112, "ab": 80, "c": 110, "ac": 90, "bc": 104, "abc": 70}
    hiB = ((y["ab"] - y["b"]) + (y["abc"] - y["bc"])) / 2
    loB = ((y["a"] - y["1"]) + (y["ac"] - y["c"])) / 2
    ab = (hiB - loB) / 2
    contrast = y["1"] - y["a"] - y["b"] + y["ab"] + y["c"] - y["ac"] - y["bc"] + y["abc"]
    assert abs(contrast / 4 - ab) < 1e-12
    return {**{f"y{k}": v for k, v in y.items()}, "hiB": s(hiB, 1), "loB": s(loB, 1), "ab": r(ab, 1), "ab_s": s(ab, 1),
            "contrast": contrast, "ab8": s(contrast / 8, 2), "ss": s(contrast ** 2 / 8, 1),
            "dA_hi1": y["ab"] - y["b"], "dA_hi2": y["abc"] - y["bc"], "dA_lo1": y["a"] - y["1"], "dA_lo2": y["ac"] - y["c"]}


def p09():
    # (a) ジニ不純度の減少
    N, A = 100, 20
    nl, al = 25, 18
    nr, ar = N - nl, A - al
    g = lambda a, n: 2 * (a / n) * (1 - a / n)
    gp, gl, gr = g(A, N), g(al, nl), g(ar, nr)
    wch = nl / N * gl + nr / N * gr
    dec = gp - wch
    err = lambda a, n: min(a, n - a) / n
    dec_err = err(A, N) - (nl / N * err(al, nl) + nr / N * err(ar, nr))
    opts = [s(wch, 3), s(dec, 3), s(gp, 3), s(gp - (gl + gr) / 2, 3), s(dec_err, 3)]
    distinct(*opts)
    # (b) ジャカード係数と単純一致係数
    a_, b_, c_, d_ = 2, 2, 1, 5
    J = a_ / (a_ + b_ + c_)
    SM = (a_ + d_) / (a_ + b_ + c_ + d_)
    pair = lambda u, v: f"ジャカード係数 ${s(u, 2)}$、単純一致係数 ${s(v, 2)}$"
    bopts = [pair(SM, J), pair(J, a_ / (a_ + b_ + c_ + d_)), pair(J, SM), pair(a_ / (a_ + b_), SM), pair(a_ / (a_ + b_ + c_ + d_), SM)]
    distinct(*bopts)
    return {
        "N": N, "A": A, "nl": nl, "al": al, "nr": nr, "ar": ar, "pp": s(A / N, 1), "pl": s(al / nl, 2), "pr": s(ar / nr, 4),
        "gp": s(gp, 2), "gl": s(gl, 4), "gr": s(gr, 4), "wl": s(nl / N, 2), "wr": s(nr / N, 2), "wch": s(wch, 4), "dec4": s(dec, 4),
        "o_wch": opts[0], "o_ok": opts[1], "o_gp": opts[2], "o_unw": opts[3], "o_err": opts[4],
        "a": a_, "b": b_, "c": c_, "d": d_, "J": s(J, 2), "SM": s(SM, 2),
        "q0": bopts[0], "q1": bopts[1], "q2": bopts[2], "q3": bopts[3], "q4": bopts[4],
    }


def p10():
    # (a) 1 因子モデルの因子得点（回帰法）
    l = [0.8, 0.7, 0.6]
    z = [1.5, 1.0, 0.5]
    psi = [1 - v * v for v in l]
    w = [a / b for a, b in zip(l, psi)]
    q = sum(a * b for a, b in zip(l, w))
    num = sum(a * b for a, b in zip(w, z))
    reg = num / (1 + q)
    bart = num / q
    lz = sum(a * b for a, b in zip(l, z))
    opts = [s(bart, 2), s(lz, 2), s(reg, 2), s(sum(z) / 3, 2), s(lz / sum(v * v for v in l), 2)]
    distinct(*opts)
    # (b) ARMA(1,1) の自己相関
    phi, th = 0.6, 0.4
    rho1 = (1 + phi * th) * (phi + th) / (1 + 2 * phi * th + th ** 2)
    rho2 = phi * rho1
    return {
        "l1": l[0], "l2": l[1], "l3": l[2], "z1": z[0], "z2": z[1], "z3": z[2],
        "psi1": s(psi[0], 2), "psi2": s(psi[1], 2), "psi3": s(psi[2], 2),
        "w1": s(w[0], 4), "w2": s(w[1], 4), "w3": s(w[2], 4), "q": s(q, 4), "num": s(num, 4), "q1": s(1 + q, 4),
        "a_bart": opts[0], "a_lz": opts[1], "a_ok": opts[2], "a_mean": opts[3], "a_ls": opts[4], "reg4": s(reg, 4),
        "phi": phi, "th": th, "n1": s(1 + phi * th, 2), "n2": s(phi + th, 1), "dd": s(1 + 2 * phi * th + th ** 2, 2),
        "rho1": s(rho1, 4), "rho2": r(rho2, 3), "rho2_s": s(rho2, 3), "rho24": s(rho2, 4),
        "w_ar": s(phi ** 2, 3), "w_ma": s(th / (1 + th ** 2), 3), "rho1_3": s(rho1, 3),
    }


def p11():
    # ログランク検定。A=旧版、B=新版。(時間, 事象=1/打ち切り=0)
    A = [(2, 1), (3, 1), (5, 0), (6, 1)]
    B = [(4, 1), (7, 1), (8, 0), (9, 0)]
    times = sorted({t for t, e in A + B if e})
    O = E = V = 0.0
    rows = []
    for t in times:
        na = sum(1 for u, _ in A if u >= t)
        nb = sum(1 for u, _ in B if u >= t)
        da = sum(1 for u, e in A if u == t and e)
        db = sum(1 for u, e in B if u == t and e)
        d, n = da + db, na + nb
        ea = d * na / n
        v = na * nb * d * (n - d) / (n * n * (n - 1)) if n > 1 else 0.0
        O += da
        E += ea
        V += v
        rows.append((t, na, nb, da, db, ea, v))
    chi = (O - E) ** 2 / V
    Eb = sum(x[3] + x[4] for x in rows) - E
    Ob = sum(x[4] for x in rows)
    approx = (O - E) ** 2 / E + (Ob - Eb) ** 2 / Eb
    opts = [s((O - E) ** 2 / E, 2), s(O - E, 2), s((O - E) / sqrt(V), 2), s(approx, 2), s(chi, 2)]
    distinct(*opts)
    out = {"O": s(O, 0), "E": s(E, 4), "V": s(V, 4), "OE": s(O - E, 4), "chi4": s(chi, 4),
           "Eb": s(Eb, 4), "Ob": s(Ob, 0),
           "o_e": opts[0], "o_diff": opts[1], "o_z": opts[2], "o_app": opts[3], "o_ok": opts[4], "c1": 3.84}
    for t, na, nb, da, db, ea, v in rows:
        out.update({f"na{t}": na, f"nb{t}": nb, f"da{t}": da, f"db{t}": db, f"ea{t}": s(ea, 4), f"v{t}": s(v, 4)})
    return out


def p12():
    # (a) 線形合同法の最大周期（総当たりで確かめた組だけを使う）
    m = 16

    def period(a, c, x0=1):
        seen, x, k = {}, x0, 0
        while x not in seen:
            seen[x] = k
            x = (a * x + c) % m
            k += 1
        return k - seen[x]
    cand = [(7, 3), (5, 4), (3, 5), (5, 3), (9, 6)]
    per = [period(a, c) for a, c in cand]
    assert [p == m for p in per] == [False, False, False, True, False], per
    # (b) KS 統計量
    u = [0.10, 0.45, 0.58, 0.66, 0.79, 0.95]
    n = len(u)
    dp = max((i + 1) / n - v for i, v in enumerate(u))
    dm = max(v - i / n for i, v in enumerate(u))
    D = max(dp, dm)
    only_up = max(abs((i + 1) / n - v) for i, v in enumerate(u))
    pp = max(abs((i + 1) / (n + 1) - v) for i, v in enumerate(u))
    crit = 0.519
    opts = [s(dp, 3), s(only_up, 3), s(pp, 3), s(D, 3), s(crit, 3)]
    distinct(*opts)
    return {"m": m, "p73": per[0], "p54": per[1], "p35": per[2], "p53": per[3], "p96": per[4],
            "u_row": ",\\ ".join(s(v, 2) for v in u), "n": n,
            "up_row": " | ".join(s((i + 1) / n - v, 3) for i, v in enumerate(u)),
            "dn_row": " | ".join(s(v - i / n, 3) for i, v in enumerate(u)),
            "dp": s(dp, 3), "dm": s(dm, 3), "D": s(D, 3), "crit": s(crit, 3),
            "o_dp": opts[0], "o_abs": opts[1], "o_pp": opts[2], "o_ok": opts[3], "o_crit": opts[4]}


def values():
    return {"s08p01": p01(), "s08p02": p02(), "s08p03": p03(), "s08p04": p04(), "s08p05": p05(),
            "s08p06": p06(), "s08p07": p07(), "s08p08": p08(), "s08p09": p09(), "s08p10": p10(),
            "s08p11": p11(), "s08p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
