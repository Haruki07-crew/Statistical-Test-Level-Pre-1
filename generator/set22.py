"""模擬試験 第22回（set22）の数値。values() が {大問id: {名前: 値}} を返す。

題材：物流倉庫・EC配送・在庫管理・コールセンター。
問題文に示した関数の値（指数・対数・正規分布の値、χ² 分布・F 分布の点）は、ここでも同じ丸めた値を使って答えを計算する。
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


def ceil(x):
    return int(-(-x // 1))


def p01():
    # (a) 20 箱中 3 箱が破損。無作為な順に開けて 3 箱目で初めて破損箱
    N, D = 20, 3
    G = N - D
    pa = (G / N) * ((G - 1) / (N - 1)) * (D / (N - 2))
    with_rep = (G / N) ** 2 * (D / N)
    atleast = 1 - (G / N) * ((G - 1) / (N - 1)) * ((G - 2) / (N - 2))
    # (b) 抜取検査：20 箱を調べ破損 2 箱以上で不合格
    n, c = 20, 2
    p_good, p_bad = 0.05, 0.20
    q19g, q19b = 0.3774, 0.01441          # 問題文に示す 0.95^19, 0.8^19
    acc_g = (1 - p_good) * q19g + n * p_good * q19g     # P(X<=1 | 0.05)
    acc_b = (1 - p_bad) * q19b + n * p_bad * q19b       # P(X<=1 | 0.20)
    alpha = 1 - acc_g
    beta = acc_b
    alpha0 = 1 - (1 - p_good) * q19g                   # X=1 の項を忘れた
    beta0 = (1 - p_bad) * q19b
    # (c) 破損率の推定に必要な標本数
    p0, e, z = 0.15, 0.03, 1.96
    nraw = z ** 2 * p0 * (1 - p0) / e ** 2
    n_half = z ** 2 * 0.25 / e ** 2
    n_645 = 1.645 ** 2 * p0 * (1 - p0) / e ** 2
    n_width = z ** 2 * p0 * (1 - p0) / (2 * e) ** 2
    return {
        **{'lb_' + k_: v_ for k_, v_ in p01b().items()},
        "N": N, "D": D, "G": G, "G1": G - 1, "G2": G - 2, "N1": N - 1, "N2": N - 2,
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "num": G * (G - 1) * D, "den": N * (N - 1) * (N - 2),
        "with_rep": s(with_rep, 4), "atleast": s(atleast, 4),
        "n": n, "c": c, "pg": s(p_good, 2), "pb": s(p_bad, 2), "qg": s(1 - p_good, 2), "qb": s(1 - p_bad, 1),
        "q19g": s(q19g, 4), "q19b": s(q19b, 5), "npg": s(n * p_good, 1), "npb": s(n * p_bad, 1),
        "acc_g": s(acc_g, 4), "acc_b": s(acc_b, 4), "alpha": s(alpha, 3), "beta": s(beta, 3),
        "alpha4": s(alpha, 4), "beta4": s(beta, 4), "acc_g3": s(acc_g, 3), "rej_b3": s(1 - beta, 3),
        "alpha0": s(alpha0, 3), "beta0": s(beta0, 3), "fg": s((1 - p_good) * q19g, 4), "fb": s((1 - p_bad) * q19b, 5),
        "gg": s(n * p_good * q19g, 4), "gb": s(n * p_bad * q19b, 5),
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
    p_ge5 = 1 - em3 * (ssum - terms[-1])
    # (b) 分位点回帰による発注量
    cu, co = 400, 100
    lines = {0.20: (80, 20), 0.25: (84, 22), 0.50: (100, 30), 0.75: (116, 40), 0.80: (121, 44)}
    tau = cu / (cu + co)
    q = lambda t, x: lines[t][0] + lines[t][1] * x
    return {
        "lam": s(lam, 1), "L": L, "stock": stock, "mu": s(mu, 0), "em3": s(em3, 4), "stock1": stock + 1,
        "t0": s(terms[0], 0), "t1": s(terms[1], 0), "t2": s(terms[2], 1), "t3": s(terms[3], 1),
        "t4": s(terms[4], 3), "t5": s(terms[5], 3), "tsum": s(ssum, 1), "cdf": s(em3 * ssum, 5),
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "p_ge5": s(p_ge5, 3),
        "cu": cu, "co": co, "cucoh": cu + co, "tau": s(tau, 1),
        "a20": lines[0.20][0], "b20": lines[0.20][1], "a25": lines[0.25][0], "b25": lines[0.25][1],
        "a50": lines[0.50][0], "b50": lines[0.50][1], "a75": lines[0.75][0], "b75": lines[0.75][1],
        "a80": lines[0.80][0], "b80": lines[0.80][1],
        "q80": q(0.80, 1), "q20": q(0.20, 1), "q25": q(0.25, 1), "q50": q(0.50, 1), "q75": q(0.75, 1),
        "lc": 20, "dv": s(0.25, 3), "dv_nosq": s(1 / (2 * sqrt(20)), 3), "dv_nolam": s(1 / (4 * 20), 4),
        "dv_sd": s(sqrt(20), 2), "dv_raw": s(20, 1), "gp": s(1 / (2 * sqrt(20)), 4), "gp2": s(1 / (4 * 20), 4),
        "q80x0": q(0.80, 0), "tau_rev": s(co / (cu + co), 1), "tau_ratio": s(co / cu, 2), "tau_1m": s(1 - co / cu, 2),
    }


def p03():
    # (a) 指数分布の混合のハザード
    w1, m1, w2, m2, t = 0.6, 1.0, 0.4, 3.0, 3
    em3, em1 = 0.0498, 0.3679     # e^{-3}, e^{-1}
    S = w1 * em3 + w2 * em1
    f = w1 / m1 * em3 + w2 / m2 * em1
    h = f / S
    h0 = w1 / m1 + w2 / m2
    mean = w1 * m1 + w2 * m2
    # (b) 一様分布 [0, θ] の 5 個の標本中央値と標本平均
    th, n = 120, 5
    k = (n + 1) // 2
    a, b = k, n - k + 1
    vbeta = a * b / ((a + b) ** 2 * (a + b + 1))
    vmed = th ** 2 * vbeta
    vmean = th ** 2 / 12 / n
    ratio = vmed / vmean
    asym = (1 / (4 * n * (1 / th) ** 2)) / vmean
    return {
        "w1": s(w1, 1), "w2": s(w2, 1), "m1": s(m1, 0), "m2": s(m2, 0), "t": t, "em3": s(em3, 4), "em1": s(em1, 4),
        "S1": s(w1 * em3, 5), "S2": s(w2 * em1, 5), "S": s(S, 5), "f1": s(w1 / m1 * em3, 5), "f2": s(w2 / m2 * em1, 5),
        "f": s(f, 5), "h": s(h, 3), "h4": s(h, 4), "h0": s(h0, 3), "f3": s(f, 3), "S3": s(S, 3), "invmean": s(1 / mean, 3),
        "mean": s(mean, 1), "w2m2": s(w2 / m2, 4), "t3": s(t / m2, 0),
        "th": th, "n": n, "k": k, "ba": a, "bb": b, "bab": a + b, "bab1": a + b + 1,
        "vbeta": s(vbeta, 5), "vmed": s(vmed, 2), "vmean": s(vmean, 1), "ratio": s(ratio, 3),
        "ratio_inv": s(1 / ratio, 3), "asym": s(asym, 3), "pi2": s(1.5708, 3), "th2": th ** 2,
        "vbeta_frac": "1/28", "vmean_frac": "1/60",
    }


def p04():
    # (a) 半径 a の円内に一様に分布する配送先までの距離
    a = 6.0
    ER = 2 * a / 3
    # (b) 重要度サンプリング：提案は [0,a] の一様分布、重み w(r)=f(r)/g(r)=2r/a
    rs = [0.9, 2.4, 3.9, 5.4]
    n = len(rs)
    w = [2 * x / a for x in rs]
    is_est = sum(x * wi for x, wi in zip(rs, w)) / n
    sn_est = sum(x * wi for x, wi in zip(rs, w)) / sum(w)
    naive = sum(rs) / n
    inv_w = sum(x * (a / (2 * x)) for x in rs) / n
    return {
        "a": s(a, 0), "ER": s(ER, 2), "Ehalf": s(a / 2, 2), "Erms": s(a / sqrt(2), 2), "E34": s(3 * a / 4, 2),
        "E13": s(a / 3, 2), "a2": s(a ** 2, 0), "Er2": s(a ** 2 / 2, 0),
        "r1": s(rs[0], 1), "r2": s(rs[1], 1), "r3": s(rs[2], 1), "r4": s(rs[3], 1), "nis": n,
        "w1": s(w[0], 2), "w2": s(w[1], 2), "w3": s(w[2], 2), "w4": s(w[3], 2),
        "rw1": s(rs[0] * w[0], 3), "rw2": s(rs[1] * w[1], 3), "rw3": s(rs[2] * w[2], 3), "rw4": s(rs[3] * w[3], 3),
        "srw": s(sum(x * wi for x, wi in zip(rs, w)), 3), "sw": s(sum(w), 1),
        "is": s(is_est, 2), "is4": s(is_est, 4), "sn": s(sn_est, 2), "naive": s(naive, 2), "invw": s(inv_w, 2),
    }


def p05():
    # (a) 対数正規分布の後処理時間
    mu, sig, c = 1.6, 0.5, 15
    ln15 = 2.708
    z = r((ln15 - mu) / sig, 2)
    phi = 0.9868                          # Φ(2.22)
    pa = 1 - phi
    zvar = (ln15 - mu) / sig ** 2
    log10_15 = 1.176
    zlog10 = (log10_15 - mu) / sig
    # (b) アーランの損失式（M/M/3/3）
    lam, mean_min, cc = 20, 6, 3
    muh = 60 / mean_min
    aa = lam / muh
    terms = [1.0]
    for k in range(1, cc + 1):
        terms.append(terms[-1] * aa / k)
    B = terms[-1] / sum(terms)
    acc = lam * (1 - B)
    # 誤答用：M/M/3 の待つ確率（待たせる方式と取り違え）
    rho = aa / cc
    tail = terms[-1] / (1 - rho)
    pwait = tail / (sum(terms[:-1]) + tail)
    # (c) フィッシャーの方法による p 値の統合
    pv = [0.08, 0.12, 0.06]
    lnp = [-2.5257, -2.1203, -2.8134]
    Xf = -2 * sum(lnp)
    # (c) デルタ法
    lc = lam
    return {
        "mu": s(mu, 1), "sig": s(sig, 1), "c": c, "ln15": s(ln15, 3), "z": s(z, 2), "zraw": s((ln15 - mu) / sig, 3),
        "phi": s(phi, 4), "pa": s(pa, 3), "pa4": s(pa, 4), "pa2": s(2 * pa, 3), "pcomp": s(phi, 3),
        "pvar": "0.000", "zvar": s(zvar, 2), "log10": s(log10_15, 3), "zlog": s(zlog10, 2), "plog": s(0.8023, 3),
        "phi085": "0.8023", "diff": s(ln15 - mu, 3), "med": s(4.953, 2),
        "lam": lam, "mean": mean_min, "cc": cc, "muh": s(muh, 0), "aa": s(aa, 0),
        "e0": s(terms[0], 0), "e1": s(terms[1], 0), "e2": s(terms[2], 0), "e3": s(terms[3], 4), "esum": s(sum(terms), 4),
        "B": s(B, 4), "acc": r(acc, 2), "acc_s": s(acc, 2), "acc4": s(acc, 4), "omB": s(1 - B, 4),
        "acc_wait": s(lam * (1 - pwait), 2), "lost": s(lam * B, 2),
        "pv1": s(pv[0], 2), "pv2": s(pv[1], 2), "pv3": s(pv[2], 2), "lp1": s(lnp[0], 4), "lp2": s(lnp[1], 4),
        "lp3": s(lnp[2], 4), "lsum": s(sum(lnp), 4), "Xf": s(Xf, 2), "Xh": s(Xf / 2, 2), "bon": s(min(pv) * 3, 2),
        "pmin": s(min(pv), 2), "c6": "12.59", "c3f": "7.81",
        "lc": lc, "dv": s(0.25, 3), "dv_nosq": s(1 / (2 * sqrt(lc)), 3),
        "dv_nolam": s(1 / (4 * lc), 4), "dv_sd": s(sqrt(lc), 2), "dv_raw": s(lc, 1),
        "gp": s(1 / (2 * sqrt(lc)), 4), "gp2": s(1 / (4 * lc), 4),
    }


def p06():
    # (a) 2 標本コルモゴロフ・スミルノフ統計量
    A = [18, 22, 26, 30, 41]
    B = [24, 33, 36, 44, 48]
    pts = sorted(A + B)
    FA = lambda x: sum(1 for v in A if v <= x) / len(A)
    FB = lambda x: sum(1 for v in B if v <= x) / len(B)
    D = max(abs(FA(x) - FB(x)) for x in pts)
    DB = max(abs(FA(x) - FB(x)) for x in B)
    # (a') 同順位のある順位和検定
    TA = [1, 2, 2, 3]
    TB = [2, 3, 4, 4, 5]
    allv = sorted(TA + TB)
    NN = len(allv)
    rank = {}
    for v in set(allv):
        idx = [i + 1 for i, x in enumerate(allv) if x == v]
        rank[v] = sum(idx) / len(idx)
    n1, n2 = len(TA), len(TB)
    W = sum(rank[v] for v in TA)
    EW = n1 * (NN + 1) / 2
    ties = [allv.count(v) for v in set(allv)]
    corr = sum(tt ** 3 - tt for tt in ties) / (NN * (NN - 1))
    VW = n1 * n2 / 12 * ((NN + 1) - corr)
    VW0 = n1 * n2 * (NN + 1) / 12
    zt = (W - EW) / sqrt(VW)
    # 中間順位を使わず、同じ値に小さい順位から A を先に割り当てた場合
    order = sorted([(v, 0) for v in TA] + [(v, 1) for v in TB])
    W_naive = sum(i + 1 for i, (v, g) in enumerate(order) if g == 0)
    # (b) 対応のある t 統計量（標準偏差と相関から差の標準偏差を求める）
    npd, mo, mn, so, sn, rc = 16, 62.0, 57.0, 8.0, 7.0, 0.75
    vd = so ** 2 + sn ** 2 - 2 * rc * so * sn
    sd_d = sqrt(vd)
    t_pair = (mo - mn) / (sd_d / sqrt(npd))
    t_unp = (mo - mn) / sqrt((so ** 2 + sn ** 2) / npd)
    t_plus = (mo - mn) / (sqrt(so ** 2 + sn ** 2 + 2 * rc * so * sn) / sqrt(npd))
    t_var = (mo - mn) / (vd / sqrt(npd))
    t_nosq = (mo - mn) / sd_d
    return {
        **{'ks_' + k_: v_ for k_, v_ in p06a().items()},
        "A1": A[0], "A2": A[1], "A3": A[2], "A4": A[3], "A5": A[4],
        "B1": B[0], "B2": B[1], "B3": B[2], "B4": B[3], "B5": B[4],
        "D": s(D, 1), "DB": s(DB, 1),
        "TA": ", ".join(str(v) for v in TA), "TB": ", ".join(str(v) for v in TB), "n1": n1, "n2": n2, "NN": NN,
        "W": s(W, 1), "EW": s(EW, 0), "VW": s(VW, 4), "VW0": s(VW0, 4), "corr": s(corr, 1), "tsum3": sum(tt ** 3 - tt for tt in ties),
        "NN1": NN * (NN - 1), "zt": s(zt, 2), "zt0": s((W - EW) / sqrt(VW0), 2), "zN": s((W - EW) / sqrt(n1 * n2 * NN / 12), 2),
        "zv": s((W - EW) / VW, 2), "znaive": s((W_naive - EW) / sqrt(VW0), 2), "Wnaive": W_naive, "sdW": s(sqrt(VW), 4),
        "dW": s(W - EW, 1), "n12": n1 * n2, "zt3": s(zt, 3),
        "npd": npd, "mo": s(mo, 0), "mn": s(mn, 0), "so": s(so, 0), "sn": s(sn, 0), "rc": s(rc, 2), "dm": s(mo - mn, 0),
        "so2": s(so ** 2, 0), "sn2": s(sn ** 2, 0), "cov": s(rc * so * sn, 0), "cov2": s(2 * rc * so * sn, 0), "vd": s(vd, 0),
        "sdd": s(sd_d, 4), "sqn": s(sqrt(npd), 0), "sed": s(sd_d / sqrt(npd), 4), "tp": s(t_pair, 2), "tp4": s(t_pair, 4),
        "tu": s(t_unp, 2), "tplus": s(t_plus, 2), "tvar": s(t_var, 2), "tnosq": s(t_nosq, 2), "dfp": npd - 1,
    }


def p07():
    b0, b1, g, d = 2.0, 0.50, 1.5, -0.30
    xs = -g / d
    # (b) 追加変数プロット
    ex = [-2, -1, 0, 1, 2]
    ey = [-1.6, -0.4, 0.2, 0.6, 1.2]
    sxy = sum(a_ * b_ for a_, b_ in zip(ex, ey))
    sxx = sum(a_ ** 2 for a_ in ex)
    syy = sum(b_ ** 2 for b_ in ey)
    bav = sxy / sxx
    return {
        **{'ct_' + k_: v_ for k_, v_ in p07a().items()},
        "b0": s(b0, 1), "b1": s(b1, 2), "g": s(g, 1), "d": s(d, 2), "dabs": s(-d, 2),
        "slopeA": s(b1 + d, 2), "b0A": s(b0 + g, 1), "xs": s(xs, 1),
        "w1": s(g / b1, 1), "w2": s(g / (b1 + d), 1), "w3": s(-xs, 1),
        "x8": 8, "yA8": s(b0 + g + (b1 + d) * 8, 1), "yB8": s(b0 + b1 * 8, 1),
        "ex": ", ".join(str(v) for v in ex), "ey": ", ".join(s(v, 1) for v in ey), "avsxy": s(sxy, 1), "avsxx": sxx,
        "avsyy": s(syy, 2), "bav": s(bav, 2), "binv": s(sxy / syy, 2), "rav": s(sxy / sqrt(sxx * syy), 2),
        "bcov": s(sxy / len(ex), 2), "bsimple": "0.90",
        "p1": s(ex[0] * ey[0], 1), "p2": s(ex[1] * ey[1], 1), "p4": s(ex[3] * ey[3], 1), "p5": s(ex[4] * ey[4], 1),
    }


def p08():
    # (a) オフセットつきポアソン回帰の逸脱度
    y = [12, 8, 20, 10]
    m = [10, 10, 16, 14]
    lr = [0.1823, -0.2231, 0.2231, -0.3365]   # ln(y/μ)
    terms = [yi * l for yi, l in zip(y, lr)]
    D = 2 * (sum(terms) - sum(yi - mi for yi, mi in zip(y, m)))
    X2 = sum((yi - mi) ** 2 / mi for yi, mi in zip(y, m))
    # (b') グッドマン・クラスカルの γ
    T = [[30, 5, 1], [20, 8, 3], [10, 6, 5]]
    Cc = Dd = 0
    for i in range(3):
        for j in range(3):
            for k in range(i + 1, 3):
                for l in range(3):
                    if l > j:
                        Cc += T[i][j] * T[k][l]
                    elif l < j:
                        Dd += T[i][j] * T[k][l]
    gam = (Cc - Dd) / (Cc + Dd)
    # (b) ファイ係数
    a, b, c, dd = 10, 190, 30, 170
    n = a + b + c + dd
    phi = (a * dd - b * c) / sqrt((a + b) * (c + dd) * (a + c) * (b + dd))
    return {
        **{'dr_' + k_: v_ for k_, v_ in p08a().items()},
        "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3], "m1": m[0], "m2": m[1], "m3": m[2], "m4": m[3],
        "l1": s(lr[0], 4), "l2": s(lr[1], 4), "l3": s(lr[2], 4), "l4": s(lr[3], 4),
        "tt1": s(terms[0], 4), "tt2": s(terms[1], 4), "tt3": s(terms[2], 4), "tt4": s(terms[3], 4),
        "tsum": s(sum(terms), 4), "D": s(D, 2), "Dh": s(D / 2, 2), "X2": s(X2, 2),
        "T11": T[0][0], "T12": T[0][1], "T13": T[0][2], "T21": T[1][0], "T22": T[1][1], "T23": T[1][2],
        "T31": T[2][0], "T32": T[2][1], "T33": T[2][2], "C": Cc, "Dd": Dd, "CmD": Cc - Dd, "CpD": Cc + Dd,
        "gam": r(gam, 3), "gam_s": s(gam, 3), "gam4": s(gam, 4), "gam_neg": s(-gam, 3),
        "a": a, "b": b, "c": c, "dd": dd, "n": n, "r1": a + b, "r2": c + dd, "c1": a + c, "c2": b + dd,
        "ad": a * dd, "bc": b * c, "adbc": a * dd - b * c, "prod": (a + b) * (c + dd) * (a + c) * (b + dd),
        "sq": s(sqrt((a + b) * (c + dd) * (a + c) * (b + dd)), 0),
        "phi": r(phi, 3), "phi_s": s(phi, 3), "phi4": s(phi, 4), "chi2": s(n * phi ** 2, 2), "phi2": s(phi ** 2, 4),
    }


def p09():
    pw, pk = 0.2, 0.1
    pdel = 1 - (1 - pw) * (1 - pk)
    pw_d = pw / pdel
    return {"pw": s(pw, 1), "pk": s(pk, 1), "pdel": s(pdel, 2), "pw_d": s(pw_d, 3), "pw_dk": s(pw, 3),
            "qw": s(1 - pw, 1), "qk": s(1 - pk, 1), "pw_add": s(pw / (pw + pk), 3)}


def p10():
    # パレットを集落とする集落抽出（大きさの異なる集落）の比推定
    Npal, n = 200, 5
    M = [20, 30, 25, 15, 10]
    a = [2, 3, 1, 2, 2]
    p = sum(a) / sum(M)
    Mbar = sum(M) / n
    res = [ai - p * mi for ai, mi in zip(a, M)]
    sr2 = sum(e ** 2 for e in res) / (n - 1)
    f = n / Npal
    V = (1 - f) / (n * Mbar ** 2) * sr2
    se = sqrt(V)
    se_bin = sqrt(p * (1 - p) / sum(M))
    se_nofpc = sqrt(sr2 / (n * Mbar ** 2))
    se_n = sqrt((1 - f) / (n * Mbar ** 2) * sum(e ** 2 for e in res) / n)
    return {
        "Npal": Npal, "n": n, "M1": M[0], "M2": M[1], "M3": M[2], "M4": M[3], "M5": M[4],
        "a1": a[0], "a2": a[1], "a3": a[2], "a4": a[3], "a5": a[4], "sM": sum(M), "sa": sum(a),
        "p": s(p, 2), "Mbar": s(Mbar, 0), "e1": s(res[0], 1), "e2": s(res[1], 1), "e3": s(res[2], 1), "e4": s(res[3], 1),
        "e5": s(res[4], 1), "ssr": s(sum(e ** 2 for e in res), 2), "sr2": s(sr2, 4), "f": s(f, 3), "omf": s(1 - f, 3),
        "Mbar2": s(Mbar ** 2, 0), "V": s(V, 8), "se": r(se, 4), "se_s": s(se, 4), "se6": s(se, 6),
        "se_bin": s(se_bin, 4), "se_nofpc": s(se_nofpc, 4), "se_n": s(se_n, 4), "nM2": s(n * Mbar ** 2, 0),
    }


def p11():
    # (a) 等相関の 3 指標、第 1 主成分による再構成
    z = (1.2, 0.3, -0.6)
    rr = 0.6
    sz = sum(z)
    score = sz / sqrt(3)
    recon = sz / 3
    res = [v - recon for v in z]
    rss = sum(v ** 2 for v in res)
    tot = sum(v ** 2 for v in z)
    res_un = [v - sz for v in z]
    # (b) バリマックス基準
    L = [(0.8, 0.1), (0.7, 0.2), (0.1, 0.8), (0.2, 0.7)]
    p = len(L)

    def vcol(col):
        sq = [l[col] ** 2 for l in L]
        m2 = sum(sq) / p
        m4 = sum(v ** 2 for v in sq) / p
        return m4 - m2 ** 2, m2, m4
    v1, m21, m41 = vcol(0)
    v2, _, _ = vcol(1)
    Vrot = v1 + v2
    c = 1 / sqrt(2)
    L0 = [((a + b) * c, (a - b) * c) for a, b in L]
    sq0 = [[l[j] ** 2 for l in L0] for j in (0, 1)]
    V0 = sum(sum(v ** 2 for v in col) / p - (sum(col) / p) ** 2 for col in sq0)
    nosub = 2 * m41
    lin = 2 * (sum(l[0] ** 2 for l in L) / p - (sum(l[0] for l in L) / p) ** 2)
    # (c) 構造方程式モデルの自由度と RMSEA
    pobs, qfree, chi, nobs = 6, 13, 20.0, 101
    mom = pobs * (pobs + 1) // 2
    dfm = mom - qfree
    rmsea = sqrt(max(chi - dfm, 0) / (dfm * (nobs - 1)))
    df_w = pobs * (pobs - 1) // 2 - qfree
    rm_w = sqrt(max(chi - df_w, 0) / (df_w * (nobs - 1)))
    rm_nosub = sqrt(chi / (dfm * (nobs - 1)))
    return {
        **{'mh_' + k_: v_ for k_, v_ in p11a().items()},
        "z1": s(z[0], 1), "z2": s(z[1], 1), "z3": s(z[2], 1), "rr": s(rr, 1), "ev1": s(1 + 2 * rr, 1),
        "sz": s(sz, 1), "score": s(score, 4), "recon": s(recon, 1), "e1": s(res[0], 1), "e2": s(res[1], 1),
        "e3": s(res[2], 1), "rss": s(rss, 2), "tot": s(tot, 2), "sc2": s(score ** 2, 2),
        "w_nosq": s(tot - score, 2), "w_un": s(sum(v ** 2 for v in res_un), 2),
        "L11": s(L[0][0], 1), "L12": s(L[0][1], 1), "L21": s(L[1][0], 1), "L22": s(L[1][1], 1),
        "L31": s(L[2][0], 1), "L32": s(L[2][1], 1), "L41": s(L[3][0], 1), "L42": s(L[3][1], 1),
        "sq1": "0.64, 0.49, 0.01, 0.04", "m21": s(m21, 3), "m41": s(m41, 5), "m21sq": s(m21 ** 2, 6),
        "v1": s(v1, 6), "Vrot": s(Vrot, 3), "Vrot6": s(Vrot, 6), "V0": s(V0, 3), "v1s": s(v1, 3),
        "nosub": s(nosub, 3), "lin": s(lin, 3), "pv": p,
        "pobs": pobs, "qfree": qfree, "chi": s(chi, 1), "nobs": nobs, "mom": mom, "dfm": dfm, "n1": nobs - 1,
        "rnum": s(chi - dfm, 1), "rden": dfm * (nobs - 1), "rratio": s((chi - dfm) / (dfm * (nobs - 1)), 4),
        "rmsea": r(rmsea, 3), "rmsea_s": s(rmsea, 3), "rmsea4": s(rmsea, 4), "dfw": df_w, "rm_w": s(rm_w, 3),
        "rm_nosub": s(rm_nosub, 3),
    }


def p12():
    mu, th = 50.0, 0.6
    xs = [53.0, 47.0, 54.0]
    e, eps = 0.0, []
    for x in xs:
        e = x - mu - th * e
        eps.append(e)
    f1 = mu + th * eps[-1]
    f2 = mu
    naive = mu + th * (xs[-1] - mu)
    return {"mu": s(mu, 0), "th": s(th, 1), "x1": s(xs[0], 0), "x2": s(xs[1], 0), "x3": s(xs[2], 0),
            "e1": s(eps[0], 1), "e2": s(eps[1], 1), "e3": s(eps[2], 2), "the1": s(th * eps[0], 1), "the2": s(th * eps[1], 2),
            "f1": s(f1, 2), "f2": s(f2, 2), "naive": s(naive, 2), "ar2": s(mu + th * (f1 - mu), 2),
            "nai2": s(mu + th * (naive - mu), 2), "wrong": s(mu + th * eps[1], 2), "the3": s(th * eps[2], 3),
            "v1": "1", "v2": s(1 + th ** 2, 2)}


def p13():
    lv = [20, 30, 40]
    m = [20.0, 24.0, 30.0]
    n, mse, dfe = 5, 8.0, 12
    c1, c2 = (-1, 0, 1), (1, -2, 1)
    L1 = sum(c * x for c, x in zip(c1, m))
    L2 = sum(c * x for c, x in zip(c2, m))
    ss1 = n * L1 ** 2 / sum(c * c for c in c1)
    ss2 = n * L2 ** 2 / sum(c * c for c in c2)
    g = sum(m) / 3
    ssa = n * sum((x - g) ** 2 for x in m)
    return {"l1": lv[0], "l2": lv[1], "l3": lv[2], "m1": s(m[0], 1), "m2": s(m[1], 1), "m3": s(m[2], 1), "n": n,
            "mse": s(mse, 1), "dfe": dfe, "N": 3 * n, "L1": s(L1, 1), "L2": s(L2, 1), "ss1": s(ss1, 1), "ss2": s(ss2, 2),
            "ssa": s(ssa, 2), "F1": s(ss1 / mse, 2), "F2": s(ss2 / mse, 2), "ss_noc": s(n * L1 ** 2, 1), "F_noc": s(n * L1 ** 2 / mse, 2),
            "ss_non": s(L1 ** 2 / 2, 1), "F_non": s(L1 ** 2 / 2 / mse, 2), "Fa": s(ssa / 2 / mse, 2), "L1sq": s(L1 ** 2, 0),
            "f112": "4.75"}


def p14():
    mu = (100, 80, 60)
    S = ((100, 40, 30), (40, 64, 24), (30, 24, 36))
    x3, c = 72, 210
    ms = mu[0] + mu[1]
    vs = S[0][0] + S[1][1] + 2 * S[0][1]
    cs = S[0][2] + S[1][2]
    cm = ms + cs / S[2][2] * (x3 - mu[2])
    cv = vs - cs ** 2 / S[2][2]
    phi = {0.77: 0.7794, 0.94: 0.8264, 1.32: 0.9066, 1.92: 0.9726}
    z = r((c - cm) / sqrt(cv), 2)
    pa = 1 - phi[z]
    z_unc = r((c - ms) / sqrt(vs), 2)
    z_nov = r((c - cm) / sqrt(vs), 2)
    vs0 = S[0][0] + S[1][1]
    z_nocov = r((c - cm) / sqrt(vs0 - cs ** 2 / S[2][2]), 2)
    return {"m1": mu[0], "m2": mu[1], "m3": mu[2], "s11": S[0][0], "s22": S[1][1], "s33": S[2][2], "s12": S[0][1],
            "s13": S[0][2], "s23": S[1][2], "x3": x3, "c": c, "ms": ms, "vs": vs, "cs": cs, "slope": s(cs / S[2][2], 1),
            "cm": s(cm, 0), "cs2": s(cs ** 2 / S[2][2], 0), "cv": s(cv, 0), "sd": s(sqrt(cv), 2), "z": s(z, 2),
            "zraw": s((c - cm) / sqrt(cv), 3), "pa": s(pa, 3), "p_unc": s(1 - phi[z_unc], 3), "p_nov": s(1 - phi[z_nov], 3),
            "p_nocov": s(1 - phi[z_nocov], 3), "p_comp": s(phi[z], 3), "dx3": x3 - mu[2],
            "f077": "0.7794", "f094": "0.8264", "f132": "0.9066", "f192": "0.9726", "zunc": s(z_unc, 2), "znov": s(z_nov, 2),
            "znocov": s(z_nocov, 2), "vs0": vs0}


def p01b():
    """(b) ロジット変換による比率の区間。e^x は問題文の表（x を小数第 3 位に丸めた値）。"""
    import math
    n, k = 200, 12
    ph = k / n
    L = -2.752                                  # log(12/188) を問題文で示す値
    sq = 3.3586                                 # √(n p̂ (1-p̂)) = √11.28
    se = 1 / sq
    ex = lambda x: math.exp(r(x, 3))
    pr = lambda x: ex(x) / (1 + ex(x))
    sep = sqrt(ph * (1 - ph) / n)
    cands = {"ok": (L - 1.96 * se, L + 1.96 * se), "90": (L - 1.645 * se, L + 1.645 * se), "narrow": (L - 1.96 * sep, L + 1.96 * sep)}
    ci = lambda a, b: f"({s(a, 3)},\\ {s(b, 3)})"
    out = {k_: ci(pr(a), pr(b)) for k_, (a, b) in cands.items()}
    out["odds"] = ci(ex(cands["ok"][0]), ex(cands["ok"][1]))
    out["wald"] = ci(ph - 1.96 * sep, ph + 1.96 * sep)
    assert len(set(out.values())) == 5
    xs = sorted(r(v, 3) for ab in cands.values() for v in ab)
    return {"n": n, "k": k, "nk": n - k, "ph": s(ph, 2), "qh": s(1 - ph, 2), "npq": s(n * ph * (1 - ph), 2), "sq": s(sq, 4),
            "L3": s(L, 3), "se": s(se, 4), "sep": s(sep, 4), "lo3": s(cands["ok"][0], 3), "hi3": s(cands["ok"][1], 3),
            "elo": s(math.exp(r(cands["ok"][0], 3)), 4), "ehi": s(math.exp(r(cands["ok"][1], 3)), 4),
            "pL": s(pr(cands["ok"][0]), 4), "pU": s(pr(cands["ok"][1]), 4),
            "tabx": " | ".join(s(x, 3) for x in xs), "tabsep": "---|" * len(xs), "taby": " | ".join(s(math.exp(x), 4) for x in xs),
            **{"ci_" + k_: v for k_, v in out.items()}}


def p06a():
    """(a) 2 標本コルモゴロフ・スミルノフ統計量。"""
    A, B = [1, 2, 2, 3], [2, 3, 4, 4, 5]
    xs = sorted(set(A + B))
    FA = [sum(v <= x for v in A) / len(A) for x in xs]
    FB = [sum(v <= x for v in B) / len(B) for x in xs]
    d = [abs(a - b) for a, b in zip(FA, FB)]
    D = max(d)
    nA, nB = len(A), len(B)
    crit = 1.36 * sqrt((nA + nB) / (nA * nB))
    assert D < crit and D > 1.36 / sqrt(nA + nB) and max(b - a for a, b in zip(FA, FB)) <= 0
    j = " | "
    return {"xa": ", ".join(map(str, A)), "xb": ", ".join(map(str, B)), "nA": nA, "nB": nB, "nAB": nA + nB, "nAnB": nA * nB,
            "D": s(D, 2), "crit": s(crit, 2), "w_crit": s(1.36 / sqrt(nA + nB), 2), "xmax": xs[d.index(D)],
            "ecx": j.join(map(str, xs)), "ecsep": "---|" * len(xs), "eca": j.join(s(v, 2) for v in FA), "ecb": j.join(s(v, 2) for v in FB),
            "ecd": j.join(s(v, 2) for v in d)}


def p07a():
    """(a) 交互作用のあるモデルで x を平均で中心化したときの係数。"""
    b0, b1, b2, b3, xbar = 2.0, 0.50, 1.5, -0.30, 6.0
    g0, g2 = b0 + b1 * xbar, b2 + b3 * xbar
    assert g2 < 0
    return {"b0": s(b0, 1), "b1": s(b1, 2), "b2": s(b2, 1), "b3": s(b3, 2), "b3s": s(b3, 2), "xbar": s(xbar, 1),
            "g0": s(g0, 1), "g2": s(g2, 2), "g2abs": s(-g2, 2), "w_g2": s(b2 - b3 * xbar, 2)}


def p08a():
    """(a) センター 4 の逸脱度残差とピアソン残差（log は問題文の 4 桁の値）。"""
    y4, m4, lg = 10, 14, -0.3365
    inner = y4 * lg - (y4 - m4)
    rD = -sqrt(2 * inner)
    rP = (y4 - m4) / 3.7417
    return {"inner": s(inner, 3), "dev2": s(2 * inner, 2), "sqd": s(sqrt(2 * inner), 4), "rD": s(rD, 2), "rD4": s(rD, 4),
            "rP": s(rP, 2), "rP4": s(rP, 4), "w_rD1": s(-sqrt(2 * y4 * -lg), 2), "w_rD2": s(-sqrt(inner), 2),
            "w_rP1": s((y4 - m4) / m4, 2), "w_rP2": s((y4 - m4) / sqrt(y4), 2)}


def p11a():
    """(a) 等相関の相関行列で、主成分得点からマハラノビス距離の 2 乗。"""
    rho, z = 0.6, (1.2, 0.3, -0.6)
    l1, l2 = 1 + 2 * rho, 1 - rho
    t1sq = sum(z) ** 2 / 3
    zz = sum(v * v for v in z)
    res = zz - t1sq
    D2 = t1sq / l1 + res / l2
    out = {"D2": s(D2, 2), "w_euc": s(zz, 2), "w_res": s(res / l2, 2), "w_mul": s(t1sq * l1 + res * l2, 2), "w_sqrt": s(t1sq / sqrt(l1) + res / sqrt(l2), 2)}
    assert len(set(out.values())) == 5
    return {"rho": s(rho, 1), "lam1": s(l1, 1), "lam2": s(l2, 1), "zvec": "(" + ",\\ ".join(s(v, 1) for v in z) + ")",
            "t1sq": s(t1sq, 2), "zz": s(zz, 2), "res": s(res, 2), "p1": s(t1sq / l1, 4), "p2": s(res / l2, 4), "D24": s(D2, 4), **out}


def values():
    return {"s22p01": p01(), "s22p02": p02(), "s22p03": p03(), "s22p04": p04(), "s22p05": p05(),
            "s22p06": p06(), "s22p07": p07(), "s22p08": p08(), "s22p09": p09(), "s22p10": p10(),
            "s22p11": p11(), "s22p12": p12(), "s22p13": p13(), "s22p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
