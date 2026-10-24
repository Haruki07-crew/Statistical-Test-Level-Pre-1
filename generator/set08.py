"""模擬試験 第8回（set08）の数値。values() が {大問id: {名前: 値}} を返す。

題材：Web・IT 運用・通信・ソフトウェア品質。
問題文に示した関数値（Φ・χ²・e^x・対数など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
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
    # (b) 5 台中 3 台目の応答（一様分布の順序統計量）
    n, kq, t = 5, 3, 0.4
    terms = [comb(n, j) * t ** j * (1 - t) ** (n - j) for j in range(kq, n + 1)]
    q = sum(terms)
    return {
        "N": N, "k": k, "pairs": pairs, "lam": s(lam, 2), "e019": s(ex[0.19], 4), "e020": s(ex[0.20], 4),
        "e038": s(ex[0.38], 4), "e002": s(ex[0.02], 4),
        "o_kn": opts[0], "o_ok": opts[1], "o_sq": opts[2], "o_union": opts[3], "o_noh": opts[4],
        "n": n, "kq": kq, "t": t, "qt": s(1 - t, 1), "q3": s(terms[0], 5), "q4": s(terms[1], 5), "q5": s(terms[2], 5),
        "quo": r(q, 3), "quo_s": s(q, 3), "quo5": s(q, 5), "w_one": s(1 - (1 - t) ** n, 3), "w_all": s(t ** n, 3),
        "w_eq": s(terms[0], 3), "w_ge4": s(1 - sum(comb(n, j) * t ** j * (1 - t) ** (n - j) for j in range(0, 4)), 3),
    }


def p02():
    # (a) 指数分布に従う遅延の比 R=T1/T2 の密度
    l1, l2 = 0.5, 0.25
    # (b) M/M/1/K の呼損
    lm, mu, K = 3, 4, 3
    rho = lm / mu
    block = rho ** K * (1 - rho) / (1 - rho ** (K + 1))
    bopts = [s(rho ** K * (1 - rho), 3), s(block, 3), s(rho ** K * (1 - rho) / (1 - rho ** K), 3), s(1 / (K + 1), 3), s(rho ** K, 3)]
    distinct(*bopts)
    probs = [rho ** j * (1 - rho) / (1 - rho ** (K + 1)) for j in range(K + 1)]
    # (c) 偶数回の切り替え
    lf = 0.8
    ev = {0.8: 0.4493, 1.6: 0.2019}
    even = (1 + ev[1.6]) / 2
    copts = [s(ev[0.8], 3), s(even, 3), s(0.5, 3), s((1 - ev[1.6]) / 2, 3), s((1 + ev[0.8]) / 2, 3)]
    distinct(*copts)
    return {
        "l1": l1, "l2": l2, "m1": s(1 / l1, 0), "m2": s(1 / l2, 0), "lp": s(l1 * l2, 3),
        "lm": lm, "mu": mu, "K": K, "K1": K + 1, "rho": s(rho, 2), "rhoK": s(rho ** K, 6), "rhoK1": s(rho ** (K + 1), 8),
        "den": s(1 - rho ** (K + 1), 8), "num": s(rho ** K * (1 - rho), 8), "block4": s(block, 4),
        "pr0": s(probs[0], 4), "pr1": s(probs[1], 4), "pr2": s(probs[2], 4), "pr3": s(probs[3], 4),
        "thru": s(lm * (1 - block), 3),
        "b_inf": bopts[0], "b_ok": bopts[1], "b_kden": bopts[2], "b_uni": bopts[3], "b_rk": bopts[4],
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
    # (a) ベリー・エッシーンの上界
    p, n, C = 0.1, 400, 0.4748
    rho3 = p * (1 - p) * ((1 - p) ** 2 + p ** 2)
    sd = sqrt(p * (1 - p))
    be = C * rho3 / (sd ** 3 * sqrt(n))
    # (b) 2 種類のデータを合わせた最尤推定
    x, tA, m, tsum = 18, 6, 5, 2.5
    mle = (x + m) / (tA + tsum)
    bopts = [s(m / tsum, 2), s((x / tA + m / tsum) / 2, 2), s(mle, 2), s(x / tA, 2), s((x + m) / tA, 2)]
    distinct(*bopts)
    # (c) 中央値の信頼区間
    nn = 10
    cdf = lambda k: sum(comb(nn, j) for j in range(k + 1))
    cov = {k: 1 - 2 * cdf(k - 1) / 2 ** nn for k in range(1, 6)}
    assert cov[2] >= 0.95 > cov[3]
    return {
        "p": p, "q": s(1 - p, 1), "n": n, "C": C, "rho3": s(rho3, 4), "sd": s(sd, 1), "sd3": s(sd ** 3, 3), "rn": s(sqrt(n), 0),
        "inner": s((1 - p) ** 2 + p ** 2, 2), "pq": s(p * (1 - p), 2), "be": r(be, 3), "be_s": s(be, 3), "be5": s(be, 5),
        "w_var": s(C * rho3 / (sd ** 2 * sqrt(n)), 3), "w_n": s(C * rho3 / (sd ** 3 * n), 4),
        "x": x, "tA": tA, "m": m, "tsum": tsum, "xm": x + m, "tt": s(tA + tsum, 1),
        "b_int": bopts[0], "b_avg": bopts[1], "b_ok": bopts[2], "b_cnt": bopts[3], "b_wrong": bopts[4], "mle4": s(mle, 4),
        "nn": nn, "c0": cdf(0), "c1": cdf(1), "c2": cdf(2), "pow": 2 ** nn,
        "cov1": s(cov[1], 4), "cov2": s(cov[2], 4), "cov3": s(cov[3], 4), "cov4": s(cov[4], 4), "cov5": s(cov[5], 4),
    }


def p05():
    # (a) バートレット検定
    ni, s2 = 11, [4.0, 9.0, 16.0]
    k = len(s2)
    N = ni * k
    sp2 = sum(s2) / k
    ln = {4.0: 1.3863, 9.0: 2.1972, 16.0: 2.7726, "sp": 2.2687}
    num = (N - k) * ln["sp"] - (ni - 1) * sum(ln[v] for v in s2)
    Cc = 1 + (1 / (3 * (k - 1))) * (k / (ni - 1) - 1 / (N - k))
    B = num / Cc
    num_n = N * ln["sp"] - ni * sum(ln[v] for v in s2)
    opts = [s(B / 2.302585, 2), s(B, 2), s(num, 2), s(num_n, 2), s(5.99, 2)]
    distinct(*opts)
    # (b) 並べ替え検定（0.1 秒単位の整数で扱う）
    old, new = [124, 118, 131], [109, 115, 120]
    allv = old + new
    obs = sum(old)
    cnt_ge = cnt_gt = cnt_le = 0
    for c in combinations(range(6), 3):
        sm = sum(allv[i] for i in c)
        cnt_ge += sm >= obs
        cnt_gt += sm > obs
        cnt_le += sm <= sum(new)
    tot = comb(6, 3)
    pv = cnt_ge / tot
    popts = [s(cnt_gt / tot, 3), s(pv, 3), s((cnt_ge + 1) / tot, 3), s((cnt_ge + cnt_le) / tot, 3), s(cnt_ge / 720, 3)]
    distinct(*popts)
    f1 = lambda v: s(v / 10, 1)
    return {
        "ni": ni, "k": k, "N": N, "df": ni - 1, "Nk": N - k, "s21": s(s2[0], 0), "s22": s(s2[1], 0), "s23": s(s2[2], 0),
        "sp2": s(sp2, 4), "ln4": s(ln[4.0], 4), "ln9": s(ln[9.0], 4), "ln16": s(ln[16.0], 4), "lnsp": s(ln["sp"], 4),
        "lsum": s(sum(ln[v] for v in s2), 4), "t1": s((N - k) * ln["sp"], 3), "t2": s((ni - 1) * sum(ln[v] for v in s2), 3),
        "num": s(num, 3), "C": s(Cc, 4), "B4": s(B, 4),
        "o_log": opts[0], "o_ok": opts[1], "o_noc": opts[2], "o_n": opts[3], "o_crit": opts[4], "chi": 5.99,
        "o1": f1(old[0]), "o2": f1(old[1]), "o3": f1(old[2]), "n1": f1(new[0]), "n2": f1(new[1]), "n3": f1(new[2]),
        "osum": f1(obs), "nsum": f1(sum(new)), "dbar": s((obs - sum(new)) / 30, 3), "tot": tot, "cge": cnt_ge,
        "p_gt": popts[0], "p_ok": popts[1], "p_extra": popts[2], "p_two": popts[3], "p_perm": popts[4],
    }


def p06():
    # (b) 1 点を除いたときの傾きの変化（DFBETA）
    x = [1, 2, 3, 4, 5, 6]
    e = [0.1, 0.2, -0.1, -0.2, -0.6, 0.6]
    b1 = 0.50
    n = len(x)
    xb = sum(x) / n
    xc = [xi - xb for xi in x]
    assert abs(sum(e)) < 1e-12 and abs(sum(a * b for a, b in zip(xc, e))) < 1e-12
    Sxx = sum(v * v for v in xc)
    i = 5
    h = 1 / n + xc[i] ** 2 / Sxx
    dfb = xc[i] * e[i] / (Sxx * (1 - h))
    b1d = b1 - dfb
    # (c) リッジ回帰の有効自由度
    d2, lam = [9, 4, 1], 2
    df = sum(v / (v + lam) for v in d2)
    copts = [s(sum(lam / (v + lam) for v in d2), 2), s(sum(sqrt(v) / (sqrt(v) + lam) for v in d2), 2), s(df, 2),
             s(len(d2) / (1 + lam), 2), s(len(d2), 2)]
    distinct(*copts)
    return {
        "x_row": " | ".join(str(v) for v in x), "e_row": " | ".join(s(v, 1) for v in e), "nx": n, "xb": s(xb, 1),
        "b1": s(b1, 2), "Sxx": s(Sxx, 1), "xci": s(xc[i], 1), "ei": s(e[i], 1), "xci2": s(xc[i] ** 2, 2),
        "h": s(h, 4), "h_frac": "\\frac{1}{6}+\\frac{6.25}{17.5}", "omh": s(1 - h, 4), "dfb": s(dfb, 4), "dfb_num": s(xc[i] * e[i], 2),
        "dfb_den": s(Sxx * (1 - h), 4), "b1d": r(b1d, 2), "b1d_s": s(b1d, 2),
        "w_noh": s(b1 - xc[i] * e[i] / Sxx, 2), "w_sign": s(b1 + dfb, 2),
        "d1": d2[0], "d2": d2[1], "d3": d2[2], "lam": lam, "f1": s(d2[0] / (d2[0] + lam), 4), "f2": s(d2[1] / (d2[1] + lam), 4),
        "f3": s(d2[2] / (d2[2] + lam), 4), "df4": s(df, 4),
        "c_comp": copts[0], "c_sv": copts[1], "c_ok": copts[2], "c_eq": copts[3], "c_ols": copts[4],
    }


def p07():
    # (a) 系統抽出の分散
    pattern = [10, 20, 30, 40]
    pop = pattern * 3
    N, kint = len(pop), 4
    n = N // kint
    Y = sum(pop) / N
    means = [sum(pop[st::kint]) / n for st in range(kint)]
    vsys = sum((m - Y) ** 2 for m in means) / kint
    S2 = sum((y - Y) ** 2 for y in pop) / (N - 1)
    sig2 = sum((y - Y) ** 2 for y in pop) / N
    vsrs = (1 - n / N) * S2 / n
    opts = [s(vsrs, 1), s(sig2 / n, 1), s(vsys, 1), s(S2 / n, 1), s(0.0, 1)]
    distinct(*opts)
    # (b) 補対数対数リンク
    b0, b1 = -2.0, 0.5
    ev = {-2.0: 0.1353, -1.5: 0.2231, "a": 0.8735, "b": 0.8000}
    p_new = 1 - ev["b"]
    p_old = round(1 - ev["a"], 4)
    lg = 0.1824
    bopts = [s(p_old, 3), s(lg, 3), s(p_new, 3), s(ev[-1.5], 3), s(ev["b"], 3)]
    distinct(*bopts)
    return {
        "pop_row": " | ".join(str(v) for v in pop), "N": N, "kint": kint, "n": n, "Y": s(Y, 0),
        "means": "、".join(s(m, 0) for m in means), "vsys": s(vsys, 1), "S2": s(S2, 2), "sig2": s(sig2, 0), "vsrs": s(vsrs, 2),
        "o_srs": opts[0], "o_sig": opts[1], "o_ok": opts[2], "o_S2": opts[3], "o_zero": opts[4],
        "b0": s(b0, 1), "b1": s(b1, 1), "lin": s(b0 + b1, 1), "e20": s(ev[-2.0], 4), "e15": s(ev[-1.5], 4),
        "ea": s(ev["a"], 4), "eb": s(ev["b"], 4), "lg": s(lg, 4),
        "c_old": bopts[0], "c_lg": bopts[1], "c_ok": bopts[2], "c_inner": bopts[3], "c_comp": bopts[4],
        "rr": "e^{0.5}",
    }


def p08():
    # (a) 処置群の平均処置効果（ATT）
    pz1, pz = 0.7, 0.4                  # P(高|導入), P(高)
    px = 0.4
    pz0 = (pz - pz1 * px) / (1 - px)    # P(高|未導入)
    y = {(1, 1): 1.2, (0, 1): 1.8, (1, 0): 0.8, (0, 0): 1.0}
    d_hi, d_lo = y[(1, 1)] - y[(0, 1)], y[(1, 0)] - y[(0, 0)]
    att = pz1 * d_hi + (1 - pz1) * d_lo
    ate = pz * d_hi + (1 - pz) * d_lo
    e1 = pz1 * y[(1, 1)] + (1 - pz1) * y[(1, 0)]
    e0 = pz0 * y[(0, 1)] + (1 - pz0) * y[(0, 0)]
    # (b) 2^3 要因実験
    yy = {"1": 120, "a": 96, "b": 112, "ab": 80, "c": 110, "ac": 90, "bc": 104, "abc": 70}
    hiB = ((yy["ab"] - yy["b"]) + (yy["abc"] - yy["bc"])) / 2
    loB = ((yy["a"] - yy["1"]) + (yy["ac"] - yy["c"])) / 2
    ab = (hiB - loB) / 2
    contrast = yy["1"] - yy["a"] - yy["b"] + yy["ab"] + yy["c"] - yy["ac"] - yy["bc"] + yy["abc"]
    assert abs(contrast / 4 - ab) < 1e-12
    return {"pz1": pz1, "qz1": s(1 - pz1, 1), "pz": pz, "qz": s(1 - pz, 1), "px": px, "pz0": s(pz0, 1),
            "y11": s(y[(1, 1)], 1), "y01": s(y[(0, 1)], 1), "y10": s(y[(1, 0)], 1), "y00": s(y[(0, 0)], 1),
            "dhi": s(d_hi, 1), "dlo": s(d_lo, 1), "att": r(att, 2), "att_s": s(att, 2), "ate": s(ate, 2),
            "e1": s(e1, 2), "e0": s(e0, 2), "naive": s(e1 - e0, 2),
            **{f"y{k}": v for k, v in yy.items()}, "hiB": s(hiB, 1), "loB": s(loB, 1), "ab": r(ab, 1), "ab_s": s(ab, 1),
            "contrast": contrast, "ab8": s(contrast / 8, 2), "ss": s(contrast ** 2 / 8, 1),
            "dA_hi1": yy["ab"] - yy["b"], "dA_hi2": yy["abc"] - yy["bc"], "dA_lo1": yy["a"] - yy["1"], "dA_lo2": yy["ac"] - yy["c"]}


def p09():
    # (a) 線形 SVM
    w, b = (3, 4), -10
    nw = sqrt(w[0] ** 2 + w[1] ** 2)
    pt = (1, 2)
    f = w[0] * pt[0] + w[1] * pt[1] + b
    assert f == 1
    side = lambda v: "攻撃側（$+1$）" if v > 0 else "正常側（$-1$）"
    fmt = lambda m, v: f"マージンの幅 ${s(m, 2)}$、点 $(1,\\ 2)$ は{side(v)}"
    opts = [fmt(1 / nw, f), fmt(2 / nw, -f), fmt(2 / nw, f), fmt(2 / (w[0] + w[1]), f), fmt(2 / nw ** 2, f)]
    distinct(*opts)
    # (b) ジャカード係数と単純一致係数
    a_, b_, c_, d_ = 2, 2, 1, 5
    J = a_ / (a_ + b_ + c_)
    SM = (a_ + d_) / (a_ + b_ + c_ + d_)
    pair = lambda u, v: f"ジャカード係数 ${s(u, 2)}$、単純一致係数 ${s(v, 2)}$"
    bopts = [pair(SM, J), pair(J, a_ / (a_ + b_ + c_ + d_)), pair(J, SM), pair(a_ / (a_ + b_), SM), pair(a_ / (a_ + b_ + c_ + d_), SM)]
    distinct(*bopts)
    return {
        "w1": w[0], "w2": w[1], "bb": b, "nw": s(nw, 0), "marg": s(2 / nw, 2), "f": f,
        "o0": opts[0], "o1": opts[1], "o2": opts[2], "o3": opts[3], "o4": opts[4],
        "a": a_, "b": b_, "c": c_, "d": d_, "J": s(J, 2), "SM": s(SM, 2),
        "q0": bopts[0], "q1": bopts[1], "q2": bopts[2], "q3": bopts[3], "q4": bopts[4],
    }


def p10():
    # (a) 共分散選択（連鎖）
    rRC, rCT, rRT = 0.6, 0.5, 0.45
    fit = rRC * rCT
    pc = (rRT - rRC * rCT) / sqrt((1 - rRC ** 2) * (1 - rCT ** 2))
    opts = [s(0.0, 2), s(pc, 2), s(fit, 2), s(rRT, 2), s((rRC + rCT) / 2, 2)]
    distinct(*opts)
    # (b) ARIMA(0,1,1) の予測（指数平滑）
    th = -0.6
    alpha = 1 + th
    f = 100.0
    xs = [104, 110, 107]
    fc = [f]
    for x in xs:
        f = alpha * x + (1 - alpha) * f
        fc.append(f)
    return {"rRC": rRC, "rCT": rCT, "rRT": rRT, "fit": s(fit, 2), "pc": s(pc, 4),
            "o_zero": opts[0], "o_pc": opts[1], "o_ok": opts[2], "o_obs": opts[3], "o_avg": opts[4],
            "th": th, "mth": s(-th, 1), "alpha": s(alpha, 1), "x1": xs[0], "x2": xs[1], "x3": xs[2], "f1": s(fc[0], 0),
            "f2": s(fc[1], 2), "f3": s(fc[2], 2), "f4": r(fc[3], 2), "f4s": s(fc[3], 2), "f43": s(fc[3], 3),
            "e1": s(xs[0] - fc[0], 2), "e2": s(xs[1] - fc[1], 2), "e3": s(xs[2] - fc[2], 2),
            "w_naive": s(xs[2], 2), "w_alpha": s(0.6 * xs[2] + 0.4 * fc[2], 2)}


def p11():
    # (a) 正確なポアソン検定
    lam0, x = 2, 6
    e2 = 0.1353
    pm = [lam0 ** j / [1, 1, 2, 6, 24, 120, 720][j] for j in range(x + 1)]
    cdf5 = e2 * sum(pm[:x])
    pv = 1 - cdf5
    p6 = e2 * pm[x]
    phi = 0.9977
    opts = [s(pv - p6, 3), s(p6, 3), s(pv, 3), s(2 * pv, 3), s(1 - phi, 3)]
    distinct(*opts)
    # (b) コックスモデル：最大点は e^β = 1/√2
    hr = 1 / sqrt(2)
    import math
    bopts = [s(0.5, 3), s(hr, 3), s(1.0, 3), s(sqrt(2), 3), s(math.log(hr), 3)]
    distinct(*bopts)
    return {"lam0": lam0, "x": x, "e2": s(e2, 4), "psum": s(sum(pm[:x]), 4), "cdf5": s(cdf5, 5), "pv5": s(pv, 5),
            "p6": s(p6, 4), "z": s((x - lam0) / sqrt(lam0), 2), "phi": s(phi, 4),
            "o_gt": opts[0], "o_eq": opts[1], "o_ok": opts[2], "o_two": opts[3], "o_norm": opts[4],
            "b_half": bopts[0], "b_ok": bopts[1], "b_one": bopts[2], "b_inv": bopts[3], "b_beta": bopts[4]}


def p12():
    # 線形合同法の最大周期（総当たりで確かめた組だけを使う）
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
    return {"m": m, "p73": per[0], "p54": per[1], "p35": per[2], "p53": per[3], "p96": per[4]}


def values():
    return {"s08p01": p01(), "s08p02": p02(), "s08p03": p03(), "s08p04": p04(), "s08p05": p05(),
            "s08p06": p06(), "s08p07": p07(), "s08p08": p08(), "s08p09": p09(), "s08p10": p10(),
            "s08p11": p11(), "s08p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
