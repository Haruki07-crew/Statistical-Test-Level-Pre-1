"""模擬試験 第29回（set29）の数値。values() が {大問id: {名前: 値}} を返す。

題材：健康診断・栄養・スポーツ医学・睡眠。
問題文に示した数表・関数の値（e^x・log・t 点・q 点など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) 尤度比によるベイズ更新
    prev = 0.20
    sens = {"hi": 0.60, "mid": 0.30, "lo": 0.10}   # P(結果 | OSA)
    fp = {"hi": 0.08, "mid": 0.32, "lo": 0.60}     # P(結果 | OSA でない)
    lr = sens["hi"] / fp["hi"]
    odds0 = prev / (1 - prev)
    odds1 = odds0 * lr
    post = odds1 / (1 + odds1)
    pos_mid = prev * (sens["hi"] + sens["mid"]) / (prev * (sens["hi"] + sens["mid"]) + (1 - prev) * (fp["hi"] + fp["mid"]))
    # (b) 決定木の情報利得（エントロピーは log2）
    N, D = 40, 10
    nL, dL = 16, 8
    nR, dR = N - nL, D - dL
    l3, l11 = 1.5850, 3.4594
    h_root = 0.5 + 0.75 * (2 - l3)                 # -(1/4)log2(1/4)-(3/4)log2(3/4)
    h_L = 1.0
    h_R = (2 + l3) - (11 / 12) * l11               # log2 12 - (11/12) log2 11
    wavg = nL / N * h_L + nR / N * h_R
    gain = h_root - wavg
    gain_unw = h_root - (h_L + h_R) / 2
    # (c) ニューラルネットワーク
    W1 = [[0.8, 1.2, 0.5], [-0.6, 0.4, 1.0]]
    b1 = [-0.5, 0.2]
    w2 = [1.5, -0.8]
    b2 = -1.0
    x = [1, 1.0, -0.5]
    u = [sum(a * b for a, b in zip(W1[k], x)) + b1[k] for k in range(2)]
    h = [max(0.0, v) for v in u]
    z = w2[0] * h[0] + w2[1] * h[1] + b2
    z_lin = w2[0] * u[0] + w2[1] * u[1] + b2
    e_z, e_lin = 0.4169, 0.2794                     # e^{-0.875}, e^{-1.275}
    assert abs(z - 0.875) < 1e-12 and abs(z_lin - 1.275) < 1e-12
    p_ok = 1 / (1 + e_z)
    p_lin = 1 / (1 + e_lin)
    npar = 3 * 2 + 2 + 2 * 1 + 1
    return {
        "prev": s(prev, 2), "se_hi": s(sens["hi"], 2), "se_mid": s(sens["mid"], 2), "se_lo": s(sens["lo"], 2),
        "fp_hi": s(fp["hi"], 2), "fp_mid": s(fp["mid"], 2), "fp_lo": s(fp["lo"], 2),
        "lr": s(lr, 1), "odds0": s(odds0, 2), "odds1": s(odds1, 3), "post": s(post, 3), "post4": s(post, 4),
        "w_sens": s(sens["hi"], 3), "w_lr": s(lr / (1 + lr), 3), "w_joint": s(prev * sens["hi"], 3), "w_mid": s(pos_mid, 3),
        "num": s(prev * sens["hi"], 3), "den2": s((1 - prev) * fp["hi"], 3), "den": s(prev * sens["hi"] + (1 - prev) * fp["hi"], 3),
        "N": N, "D": D, "nL": nL, "dL": dL, "nR": nR, "dR": dR, "l3": s(l3, 4), "l11": s(l11, 4),
        "h_root": s(h_root, 4), "h_R": s(h_R, 4), "wavg": s(wavg, 4), "gain": r(gain, 3), "gain_s": s(gain, 3), "gain4": s(gain, 4),
        "gain_unw": s(gain_unw, 3), "wL": s(nL / N, 1), "wR": s(nR / N, 1), "l3q": s(0.75 * (2 - l3), 4), "l11q": s(11 / 12 * l11, 4),
        "w11": s(W1[0][0], 1), "w12": s(W1[0][1], 1), "w13": s(W1[0][2], 1),
        "w21": s(W1[1][0], 1), "w22": s(W1[1][1], 1), "w23": s(W1[1][2], 1),
        "b11": s(b1[0], 1), "b12": s(b1[1], 1), "v1": s(w2[0], 1), "v2": s(w2[1], 1), "b2": s(b2, 1),
        "x1": x[0], "x2": s(x[1], 1), "x3": s(x[2], 1), "u1": s(u[0], 2), "u2": s(u[1], 2),
        "z": s(z, 3), "zlin": s(z_lin, 3), "ez": s(e_z, 4), "elin": s(e_lin, 4),
        "npar": npar, "npar_nob": 3 * 2 + 2, "p_ok": s(p_ok, 3), "p_lin": s(p_lin, 3), "p_noz": s(z, 3), "p_ok4": s(p_ok, 4),
    }


def p02():
    # (a) 3食のエネルギーの分散共分散行列
    sd = [150, 200, 250]
    rho = {(0, 1): 0.1, (0, 2): -0.3, (1, 2): 0.1}
    V = [[sd[i] * sd[j] * (1 if i == j else rho[(min(i, j), max(i, j))]) for j in range(3)] for i in range(3)]
    varT = sum(V[i][j] for i in range(3) for j in range(3))
    covBT = sum(V[0])
    corr = covBT / (sd[0] * sqrt(varT))
    corr_ind = V[0][0] / (sd[0] * sqrt(sum(V[i][i] for i in range(3))))
    # (b) 主成分負荷量から固有値と寄与率
    L1 = [0.94, 0.81, 0.81, 0.65, 0.60]
    L2 = [-0.21, 0.28, 0.15, -0.68, 0.49]
    lam1 = sum(a * a for a in L1)
    lam2 = sum(a * a for a in L2)
    return {
        "vB": V[0][0], "vL": V[1][1], "vD": V[2][2], "cBL": s(V[0][1], 0), "cBD": s(V[0][2], 0), "cLD": s(V[1][2], 0),
        "sumv": V[0][0] + V[1][1] + V[2][2], "sumc": s(V[0][1] + V[0][2] + V[1][2], 0), "varT": s(varT, 0), "sdT": s(sqrt(varT), 2),
        "covBT": s(covBT, 0), "corr": r(corr, 3), "corr_s": s(corr, 3), "corr4": s(corr, 4), "sdB": sd[0],
        "corr_ind": s(corr_ind, 3), "sdTind": s(sqrt(sum(V[i][i] for i in range(3))), 2),
        "l11": s(L1[0], 2), "l12": s(L1[1], 2), "l13": s(L1[2], 2), "l14": s(L1[3], 2), "l15": s(L1[4], 2),
        "l21": s(L2[0], 2), "l22": s(L2[1], 2), "l23": s(L2[2], 2), "l24": s(L2[3], 2), "l25": s(L2[4], 2),
        "lam1": s(lam1, 4), "lam2": s(lam2, 4), "c1": s(lam1 / 5 * 100, 1), "c_cum": s((lam1 + lam2) / 5 * 100, 1),
        "c_nosq": s(sum(L1) / 5 * 100, 1), "c_two": s(lam1 / (lam1 + lam2) * 100, 1), "c_max": s(L1[0] ** 2 * 100, 1),
        "sumL1": s(sum(L1), 2),
    }


def p03():
    W = [0.7, 0.3]
    n = [200, 150]
    m = [9.8, 11.2]
    sdh = [3.0, 3.5]
    est = sum(w * a for w, a in zip(W, m))
    terms = [w * w * b * b / k for w, b, k in zip(W, sdh, n)]
    var = sum(terms)
    se = sqrt(var)
    w_now2 = sqrt(sum(w * b * b / k for w, b, k in zip(W, sdh, n)))
    w_sd = sqrt(sum(w * w * b / k for w, b, k in zip(W, sdh, n)))
    w_now = sqrt(sum(b * b / k for b, k in zip(sdh, n)))
    return {
        "W1": s(W[0], 1), "W2": s(W[1], 1), "n1": n[0], "n2": n[1], "m1": s(m[0], 1), "m2": s(m[1], 1),
        "sd1": s(sdh[0], 1), "sd2": s(sdh[1], 1), "v1": s(sdh[0] ** 2, 2), "v2": s(sdh[1] ** 2, 2),
        "W1sq": s(W[0] ** 2, 2), "W2sq": s(W[1] ** 2, 2), "est": s(est, 2),
        "t1": s(terms[0], 5), "t2": s(terms[1], 5), "var": s(var, 4), "se": s(se, 3), "se4": s(se, 4),
        "w_now2": s(w_now2, 3), "w_sd": s(w_sd, 3), "w_var": s(var, 3), "w_now": s(w_now, 3),
    }


def p04():
    # (a) フィッシャーの正確検定
    nA, xA, nB, xB = 8, 1, 8, 5
    K = xA + xB
    N = nA + nB
    tot = comb(N, nA)
    pr = [comb(K, k) * comb(N - K, nA - k) for k in range(0, K + 1)]
    p_le = (pr[0] + pr[1]) / tot
    # (b) ガンマ分布（形状 3・尺度 5）
    k, theta, t = 3, 5, 20
    lam_t = t / theta
    e4 = 0.01832
    surv = e4 * (1 + lam_t + lam_t ** 2 / 2)
    # (c) ログランク
    A = [(3, 1), (6, 0), (8, 1), (12, 0), (15, 0)]
    B = [(2, 1), (4, 1), (5, 0), (7, 1), (9, 1)]
    times = sorted({tt for tt, d in A + B if d})
    O = E = Vv = 0.0
    rows = []
    for tt in times:
        na = sum(1 for u, _ in A if u >= tt)
        nb = sum(1 for u, _ in B if u >= tt)
        da = sum(1 for u, d in A if u == tt and d)
        db = sum(1 for u, d in B if u == tt and d)
        dd, nn = da + db, na + nb
        e = na * dd / nn
        v = na * nb * dd * (nn - dd) / (nn ** 2 * (nn - 1))
        rows.append((tt, na, nb, da, db, e, v))
        O += da
        E += e
        Vv += v
    chi = (O - E) ** 2 / Vv
    EB = len(times) - E
    OB = len(times) - O
    pear = (O - E) ** 2 / E + (OB - EB) ** 2 / EB
    out = {
        "nA": nA, "xA": xA, "nB": nB, "xB": xB, "K": K, "N": N, "NK": N - K, "tot": tot,
        "pr0": pr[0], "pr1": pr[1], "p_ok": s(p_le, 3), "p_ok4": s(p_le, 4), "p_eq": s(pr[1] / tot, 3),
        "p_zero": s(pr[0] / tot, 3), "p_two": s(2 * p_le, 3), "p_rev": s(1 - p_le, 3), "pr01": pr[0] + pr[1],
        "k": k, "theta": theta, "t": t, "lt": s(lam_t, 0), "lt2": s(lam_t ** 2 / 2, 0), "e4": s(e4, 5),
        "sum3": s(1 + lam_t + lam_t ** 2 / 2, 0), "surv": r(surv, 3), "surv_s": s(surv, 3), "surv5": s(surv, 5),
        "mean": k * theta, "w_exp": s(0.01832 * 1, 3), "w_two": s(e4 * (1 + lam_t), 3),
        "O": s(O, 0), "E": s(E, 3), "OE": s(O - E, 3), "V": s(Vv, 4), "chi": s(chi, 2), "chi4": s(chi, 3),
        "OEsq": s((O - E) ** 2, 3), "EB": s(EB, 3), "OB": s(OB, 0), "pear": s(pear, 2), "oneterm": s((O - E) ** 2 / E, 2),
        "nodiv": s((O - E) ** 2, 2), "zabs": s(abs(O - E) / sqrt(Vv), 2),
    }
    for i, (tt, na, nb, da, db, e, v) in enumerate(rows, 1):
        out.update({f"r{i}t": tt, f"r{i}a": na, f"r{i}b": nb, f"r{i}da": da, f"r{i}db": db,
                    f"r{i}e": s(e, 4), f"r{i}v": s(v, 4)})
    return out


def p05():
    n, a, b, c = 50, 0.10, 0.30, 0.104
    lam = n * (c - a) / (b - a)
    em1 = 0.3679
    ok = 1 - em1
    return {"n": n, "a": s(a, 2), "b": s(b, 2), "c": s(c, 3), "ca": s(c - a, 3), "ba": s(b - a, 1),
            "lam": s(lam, 1), "em1": s(em1, 4), "ok": s(ok, 3), "pone": s((c - a) / (b - a), 2),
            "w_comp": s(em1, 3), "w_nodiv": s(1 - 0.8187, 3), "w_one": s((c - a) / (b - a), 3), "w_half": s(1 - 0.1353, 3),
            "exact": s(1 - (1 - (c - a) / (b - a)) ** n, 3), "e02": s(0.8187, 4), "e2": s(0.1353, 4)}


def p06():
    n = 16
    m1, s1, m2, s2, rho = 5.9, 0.8, 6.3, 0.9, 0.75
    d = m2 - m1
    vd = s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2
    sdd = sqrt(vd)
    se = sdd / sqrt(n)
    t = d / se
    tr = r(t, 2)
    sp2 = (s1 ** 2 + s2 ** 2) / 2
    t_ind = d / sqrt(sp2 * 2 / n)
    t_nosqrt = d / sdd
    t_plus = d / (sqrt(s1 ** 2 + s2 ** 2 + 2 * rho * s1 * s2) / sqrt(n))
    t15 = 1.753
    # (b) 尤度比統計量
    q = tr ** 2 / (n - 1)
    lnv = 0.3792          # log(1.4611)
    lr = n * lnv
    wald = n * tr ** 2 / (n - 1)
    score = n * tr ** 2 / (n - 1 + tr ** 2)
    return {
        "n": n, "m1": s(m1, 1), "s1": s(s1, 1), "m2": s(m2, 1), "s2": s(s2, 1), "rho": s(rho, 2), "d": s(d, 1),
        "s1sq": s(s1 ** 2, 2), "s2sq": s(s2 ** 2, 2), "cov2": s(2 * rho * s1 * s2, 2), "vd": s(vd, 2), "sdd": s(sdd, 4),
        "se": s(se, 4), "t": s(tr, 2), "t15": s(t15, 3), "t30": "1.697", "df": n - 1, "sqn": s(sqrt(n), 0),
        "t_ind": s(t_ind, 2), "sp2": s(sp2, 3), "t_nosqrt": s(t_nosqrt, 2), "t_plus": s(t_plus, 2),
        "tsq": s(tr ** 2, 4), "q": s(q, 4), "q1": s(1 + q, 4), "lnv": s(lnv, 4), "lr": s(lr, 2), "lr4": s(lr, 4),
        "w_tsq": s(tr ** 2, 2), "w_n1": s((n - 1) * lnv, 2), "wald": s(wald, 2), "score": s(score, 2),
        "chi1": "3.841", "nd2": s(n / 2, 0),
    }


def p07():
    x = [2, 0, 5, 1, 3, 6, 1, 0, 4, 9]
    n = len(x)
    mean = sum(x) / n
    ss = sum((v - mean) ** 2 for v in x)
    sx2 = sum(v * v for v in x)
    D = ss / mean
    # (b) 睡眠段階のマルコフ連鎖（W, L, Dp）
    P = [[0.5, 0.5, 0.0], [0.1, 0.6, 0.3], [0.0, 0.4, 0.6]]
    # 初到達確率：D に入らない経路だけ
    a1 = [P[0][0], P[0][1]]                     # 1 歩目 W, L
    a2W = a1[0] * P[0][0] + a1[1] * P[1][0]
    a2L = a1[0] * P[0][1] + a1[1] * P[1][1]
    f3 = a2W * P[0][2] + a2L * P[1][2]
    f2 = a1[0] * P[0][2] + a1[1] * P[1][2]
    f1 = P[0][2]
    # 3 ステップ推移確率（途中で D に入ってもよい）
    def mm(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    P3 = mm(mm(P, P), P)
    return {
        "xs": "、".join(str(v) for v in x), "n": n, "sum": sum(x), "mean": s(mean, 1), "sx2": sx2,
        "nm2": s(n * mean ** 2, 1), "ss": s(ss, 1), "D": r(D, 2), "D_s": s(D, 2), "D4": s(D, 4), "df": n - 1, "chi9": "16.919",
        "var_u": s(ss / (n - 1), 3), "disp": s(ss / (n - 1) / mean, 3),
        "pWW": s(P[0][0], 1), "pWL": s(P[0][1], 1), "pWD": s(P[0][2], 1), "pLW": s(P[1][0], 1), "pLL": s(P[1][1], 1),
        "pLD": s(P[1][2], 1), "pDW": s(P[2][0], 1), "pDL": s(P[2][1], 1), "pDD": s(P[2][2], 1),
        "a2W": s(a2W, 2), "a2L": s(a2L, 2), "f3": s(f3, 3), "f2": s(f2, 3), "f1": s(f1, 3), "P3": s(P3[0][2], 3),
        "cum3": s(f1 + f2 + f3, 3), "noW": s(a1[1] * P[1][1] * P[1][2], 3),
        "aWL": s(a1[0] * P[0][1], 2), "aLL": s(a1[1] * P[1][1], 2),
    }


def p08():
    n = 60
    root = 0.95130      # 0.05^{1/60}
    root2 = 0.94037     # 0.025^{1/60}
    up = 1 - root
    return {"n": n, "root": s(root, 5), "root2": s(root2, 5), "up": s(up, 3), "up4": s(up, 5),
            "w3": s(3 / n, 3), "w2s": s(1 - root2, 3), "w_wald": "0.000", "w_div": s(0.05 / n, 3), "ln05": "-2.9957",
            "ln05n": s(-2.9957 / n, 5)}


def p09():
    c = 0.5
    ln2 = 0.6931
    cdf = c + c * ln2      # P(W<=c) = c - c ln c
    ok = 1 - cdf
    return {"c": s(c, 1), "ln2": s(ln2, 4), "cdf": s(cdf, 4), "ok": s(ok, 3), "ok4": s(ok, 4),
            "w_ind": s((1 - c) ** 2, 3), "w_cln": s(c * ln2, 3), "w_unif": s(1 - c, 3), "w_comp": s(cdf, 3), "cln": s(c * ln2, 4)}


def p10():
    # (a) 対数変換した中性脂肪の回帰
    b0, b1, b2 = 3.80, 0.040, 0.15
    bmi, alc = 25, 3
    sig = 0.40
    mu = b0 + b1 * bmi + b2 * alc
    e_mu, e08, e16, e40 = 190.57, 1.0833, 1.1735, 1.4918
    mean = e_mu * e08
    # (b) 外れ値のボンフェローニ判定
    n, pcoef = 60, 3
    tmax = 3.40
    # (c) マローズの Cp
    rss = {"M1": 14.6, "M2": 11.8, "M3": 11.24, "M4": 11.00}
    pp = {"M1": 2, "M2": 3, "M3": 4, "M4": 5}
    s2 = rss["M4"] / (n - 5)
    cp = {k: rss[k] / s2 - n + 2 * pp[k] for k in rss}
    cp_noint = {k: rss[k] / s2 - n + 2 * (pp[k] - 1) for k in rss}
    s2n = rss["M4"] / n
    cp_n = {k: rss[k] / s2n - n + 2 * pp[k] for k in rss}
    best = min(cp, key=cp.get)
    assert best == "M3"
    out = {
        "b0": s(b0, 2), "b1": s(b1, 3), "b2": s(b2, 2), "bmi": bmi, "alc": alc, "sig": s(sig, 2), "sig2h": s(sig ** 2 / 2, 2),
        "sig2": s(sig ** 2, 2), "mu": s(mu, 2), "mu2": s(mu + sig ** 2 / 2, 2), "emu": s(e_mu, 2), "e08": s(e08, 4),
        "e16": s(e16, 4), "e40": s(e40, 4), "mean": s(mean, 0), "mean1": s(mean, 1), "w_med": s(e_mu, 0),
        "w_s2": s(e_mu * e16, 0), "w_sig": s(e_mu * e40, 0), "w_minus": s(e_mu / e08, 0), "b1x": s(b1 * bmi, 2), "b2x": s(b2 * alc, 2),
        "n": n, "pc": pcoef, "dfr": n - pcoef - 1, "tmax": s(tmax, 2), "tb": "3.532", "t025": "2.003", "t1s": "3.304",
        "alpha2n": s(0.05 / (2 * n), 6), "alphan": s(0.05 / n, 6),
        "s2": s(s2, 2), "rssF": s(rss["M4"], 2), "dfF": n - 5,
    }
    for k in rss:
        out.update({f"rss{k}": s(rss[k], 2), f"p{k}": pp[k], f"cp{k}": s(cp[k], 1), f"cpn{k}": s(cp_noint[k], 1),
                    f"cpN{k}": s(cp_n[k], 1), f"q{k}": s(rss[k] / s2, 1)})
    return out


def p11():
    n, y = 200, 30
    p0 = 0.5
    score = y - n * p0
    info = n * p0 * (1 - p0)
    b1 = 0 + score / info
    ybar = y / n
    # (b) ジェフリーズ事前分布
    a, b = y + 0.5, n - y + 0.5
    return {
        "n": n, "y": y, "score": s(score, 0), "info": s(info, 0), "b1": r(b1, 2), "b1s": s(b1, 2),
        "ybar": s(ybar, 2), "logit": "-1.7346", "w_ols": s(ybar - 0.5, 2), "w_inv": s(info / score, 2),
        "a": s(a, 1), "bb": s(b, 1), "ab": s(a + b, 0), "jeff": s(a / (a + b), 3), "jeff4": s(a / (a + b), 4),
        "unif": s((y + 1) / (n + 2), 3), "mle": s(ybar, 3), "beta22": s((y + 2) / (n + 4), 3),
        "mode": s((a - 1) / (a + b - 2), 3),
    }


def p12():
    ns = [8, 10, 10, 12]
    ms = [6.2, 3.9, 5.4, 2.6]
    mse, q, tcrit = 5.0, 3.809, 2.028
    names = "ABCD"
    dfe = sum(ns) - 4
    out = {"mse": s(mse, 1), "q": s(q, 3), "tcrit": s(tcrit, 3), "dfe": dfe, "N": sum(ns), "qr2": s(q / sqrt(2), 4)}
    cnt = cnt_l = cnt_q = 0
    for i, a in enumerate(names):
        out[f"n{a}"] = ns[i]
        out[f"m{a}"] = s(ms[i], 1)
    for i in range(4):
        for j in range(i + 1, 4):
            se = sqrt(mse * (1 / ns[i] + 1 / ns[j]))
            hsd = q / sqrt(2) * se
            diff = abs(ms[i] - ms[j])
            key = names[i] + names[j]
            out[f"se{key}"] = s(se, 4)
            out[f"h{key}"] = s(hsd, 3)
            out[f"l{key}"] = s(tcrit * se, 3)
            out[f"d{key}"] = s(diff, 1)
            out[f"j{key}"] = "有意" if diff > hsd else "有意でない"
            out[f"jl{key}"] = "有意" if diff > tcrit * se else "有意でない"
            cnt += diff > hsd
            cnt_l += diff > tcrit * se
            cnt_q += diff > q * se
    out.update({"cnt": cnt, "cnt_l": cnt_l, "cnt_q": cnt_q})
    return out


def p13():
    chi, df, N = 96.4, 40, 151
    rm = sqrt(max(chi - df, 0) / (df * (N - 1)))
    rmN = sqrt(max(chi - df, 0) / (df * N))
    return {"chi": s(chi, 1), "df": df, "N": N, "N1": N - 1, "num": s(chi - df, 1), "den": df * (N - 1),
            "ratio": s((chi - df) / (df * (N - 1)), 4), "rm": r(rm, 3), "rm_s": s(rm, 3), "rm4": s(rm, 4), "rmN": s(rmN, 4),
            "chidf": s(chi / df, 2), "w_nosqrt": s((chi - df) / (df * (N - 1)), 4)}


def p14():
    n, sig = 50, 6.0
    se = sig / sqrt(2 * n)
    r1, r2, nn = 0.60, 0.42, 200
    phi22 = (r2 - r1 ** 2) / (1 - r1 ** 2)
    return {"n": n, "sig": s(sig, 1), "se": s(se, 2), "w_mean": s(sig / sqrt(n), 2), "w_var": s(sig ** 2 * sqrt(2 / n), 2),
            "w_v2": s(sig ** 2 / (2 * n), 2), "w_rel": s(sig * sqrt(2 / n), 2), "sig2": s(sig ** 2, 0), "two_n": 2 * n,
            "r1": s(r1, 2), "r2": s(r2, 2), "nn": nn, "r1sq": s(r1 ** 2, 2), "phi22": s(phi22, 3), "num": s(r2 - r1 ** 2, 2),
            "den": s(1 - r1 ** 2, 2), "band": s(2 / sqrt(nn), 3), "w_nodiv": s(r2 - r1 ** 2, 3)}


def values():
    return {"s29p01": p01(), "s29p02": p02(), "s29p03": p03(), "s29p04": p04(), "s29p05": p05(),
            "s29p06": p06(), "s29p07": p07(), "s29p08": p08(), "s29p09": p09(), "s29p10": p10(),
            "s29p11": p11(), "s29p12": p12(), "s29p13": p13(), "s29p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
