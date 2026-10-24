"""模擬試験 第28回（set28）の数値。values() が {大問id: {名前: 値}} を返す。

題材：半導体・電子部品・精密機器の工程管理。
問題文に示した分布の値（e^x・ln・t・q・χ² の点など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。"""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def p01():
    # (a) ポアソン歩留まり
    D = 0.5            # 欠陥密度（個/cm^2）
    AA, nA = 1.0, 600  # 品種A：面積と1枚あたりチップ数
    AB, nB = 2.0, 290
    eA, eB = 0.6065, 0.3679   # e^{-0.5}, e^{-1.0}（問題文の値）
    goodB = nB * eB
    goodA = nA * eA
    # (b) デルタ法
    n, T = 40, 24
    lam = T / n
    e06 = 0.5488             # e^{-0.6}
    se_lam = sqrt(lam / n)
    se_y = e06 * se_lam
    var_y = e06 ** 2 * lam / n
    bin_se = sqrt(e06 * (1 - e06) / n)
    nosqrt = e06 * lam / n
    # (c) 全分散の公式
    m, muL, sdL = 10, 0.6, 0.3
    vS = m * muL + m ** 2 * sdL ** 2
    return {
        "D": D, "AA": s(AA, 1), "nA": nA, "AB": s(AB, 1), "nB": nB, "eA": s(eA, 4), "eB": s(eB, 4),
        "lamA": s(D * AA, 1), "lamB": s(D * AB, 1),
        "goodB": r(goodB, 1), "goodB_s": s(goodB, 1), "goodB3": s(goodB, 3), "goodA": s(goodA, 1),
        "goodB_lin": s(nB * (1 - D * AB) if D * AB < 1 else 0, 1), "goodB_A": s(nB * eA, 1),
        "n": n, "T": T, "lam": s(lam, 1), "e06": s(e06, 4), "lamn": s(lam / n, 3),
        "se_lam": s(se_lam, 4), "se_y": s(se_y, 4), "se_y5": s(se_y, 5), "var_y": s(var_y, 4),
        "bin_se": s(bin_se, 4), "nosqrt": s(nosqrt, 4), "e06sq": s(e06 ** 2, 4),
        "m": m, "muL": s(muL, 1), "sdL": s(sdL, 1), "varL": s(sdL ** 2, 2),
        "mmu": s(m * muL, 1), "m2": m ** 2, "m2v": s(m ** 2 * sdL ** 2, 1),
        "vS": s(vS, 1), "vS_pois": s(m * muL, 1), "vS_m": s(m * muL + m * sdL ** 2, 1),
        "vS_only": s(m ** 2 * sdL ** 2, 1), "vS_sd": s(m * muL + m ** 2 * sdL, 1),
    }


def p02():
    eta, mm = 4000, 2
    ln09 = -0.10536
    b10 = eta * sqrt(-ln09)
    b10_noroot = eta * (-ln09)
    # (b) 逆関数法
    u = 0.75
    ln025, ln075 = -1.3863, -0.2877
    t_ok = eta * sqrt(-ln025)
    t_u = eta * sqrt(-ln075)
    t_nosq = eta * (-ln025)
    t_sq = eta * ln025 ** 2
    t_in = sqrt(eta * (-ln025))
    return {
        "eta": eta, "m": mm, "ln09": s(ln09, 5), "mln09": s(-ln09, 5), "rt": s(sqrt(-ln09), 6),
        "b10": r(b10, 0), "b10_s": s(b10, 2), "b10_noroot": s(b10_noroot, 0),
        "b10_50": s(eta * sqrt(math.log(2)), 0),
        "u": u, "ln025": s(ln025, 4), "ln075": s(ln075, 4), "mln025": s(-ln025, 4),
        "rt025": s(sqrt(-ln025), 5),
        "t_ok": s(t_ok, 0), "t_u": s(t_u, 0), "t_nosq": s(t_nosq, 0), "t_sq": s(t_sq, 0), "t_in": s(t_in, 1),
    }


def p03():
    pp, pc, pv = 0.90, 0.95, 0.80   # ポンプ・制御器・バルブが正常に働く確率
    rp = 1 - (1 - pp) ** 2
    rv = 1 - (1 - pv) ** 2
    R = rp * pc * rv
    # (b) 一様分布の積
    a = 0.1
    w0 = 0.0025
    t = w0 / a ** 2
    ln_t = -1.3863     # ln 0.25
    P = t - t * ln_t
    assert abs(t - 0.25) < 1e-12
    return {
        "pp": s(pp, 2), "pc": s(pc, 2), "pv": s(pv, 2), "qp": s(1 - pp, 2), "qv": s(1 - pv, 2),
        "rp": s(rp, 2), "rv": s(rv, 2), "R": s(R, 4),
        "R_series": s(pp ** 2 * pc * pv ** 2, 4), "R_v1": s(rp * pc * pv, 4), "R_p1": s(pp * pc * rv, 4),
        "R_union": s(1 - ((1 - pp) ** 2 + (1 - pc) + (1 - pv) ** 2), 4),
        "a": s(a, 1), "a2": s(a ** 2, 2), "w0": s(w0, 4), "t": s(t, 2), "ln_t": s(ln_t, 4),
        "tlnt": s(-t * ln_t, 4), "P": s(P, 4), "P_t": s(t, 4), "P_sq": s(t ** 2, 4), "P_sqrt": s(sqrt(t), 4),
    }


def p04():
    A = [512.4, 498.7, 505.1, 520.3, 509.8, 501.6, 515.2, 497.9, 507.5, 511.0]
    d = [1.2, 0.8, -0.3, 1.5, 0.9, 0.4, 1.1, -0.2, 0.7, 1.0]
    B = [round(a - x, 1) for a, x in zip(A, d)]
    n = len(d)
    sd_ = sum(d)
    sd2 = sum(x * x for x in d)
    mean = sd_ / n
    ss = sd2 - sd_ ** 2 / n
    var = ss / (n - 1)
    sdv = sqrt(var)
    t = mean / (sdv / sqrt(n))
    # 符号検定
    pos = sum(1 for x in d if x > 0)
    neg = n - pos
    p8 = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    out = {"n": n, "n1": n - 1, "sd": s(sd_, 1), "sd2": s(sd2, 2), "mean": s(mean, 2),
           "ss": s(ss, 3), "var": s(var, 4), "sdv": s(sdv, 4), "sem": s(sdv / sqrt(n), 4),
           "t": r(t, 2), "t_s": s(t, 2), "t3": s(t, 3),
           "t_n": s(mean / (sqrt(ss / n) / sqrt(n)), 2),
           "pos": pos, "neg": neg,
           "c8": math.comb(n, 8), "c9": math.comb(n, 9), "c10": 1, "csum": math.comb(n, 8) + math.comb(n, 9) + 1,
           "p": s(p8, 4), "p2": s(2 * p8, 4), "pe": s(math.comb(n, pos) / 2 ** n, 4),
           "p9": s((math.comb(n, 9) + 1) / 2 ** n, 4), "pc": s(1 - p8, 4)}
    for i in range(n):
        out[f"a{i}"] = s(A[i], 1)
        out[f"b{i}"] = s(B[i], 1)
        out[f"d{i}"] = s(d[i], 1)
    return out


def p05():
    # (a) 故障0の正確な片側上限
    n = 60
    ln05, ln025, ln95 = -2.9957, -3.6889, -0.05129
    x = ln05 / n
    x_s = r(x, 5)
    ex = 0.95130   # e^{-0.04993}
    assert abs(math.exp(x_s) - ex) < 5e-6
    pu = 1 - ex
    x2 = r(ln025 / n, 5)
    ex2 = 0.94037  # e^{-0.06148}
    assert abs(math.exp(x2) - ex2) < 5e-6
    x3 = r(ln95 / n, 6)
    # (b) 指数分布の平均の尤度比検定
    m, theta0, xbar = 10, 1000, 450
    ratio = xbar / theta0
    ln_ratio = -0.7985  # ln 0.45
    lr = 2 * m * (ratio - 1 - ln_ratio)
    return {
        "n": n, "ln05": s(ln05, 4), "ln025": s(ln025, 4), "x": s(x_s, 5), "mx": s(-x_s, 5), "ex": s(ex, 5),
        "x2": s(x2, 5), "ex2": s(ex2, 5),
        "pu": s(pu, 4), "pu_r3": s(3 / n, 4), "pu_1n": s(1 / n, 4), "pu_two": s(1 - ex2, 4),
        "pu_95": s(1 - math.exp(x3), 4), "ln95": s(ln95, 5),
        "m": m, "theta0": theta0, "xbar": xbar, "sumx": m * xbar, "ratio": s(ratio, 2),
        "ln_ratio": s(ln_ratio, 4), "inner": s(ratio - 1 - ln_ratio, 4),
        "lr": r(lr, 2), "lr_s": s(lr, 2), "lr_half": s(lr / 2, 2), "chi1": 3.84,
        "lr_m": s(m * (ratio - 1 - ln_ratio), 2),
    }


def p06():
    n = 10
    v1 = 1 / (3 * n)            # V[2 Xbar]/θ²
    v2 = 1 / (n * (n + 2))      # V[(n+1)/n X(n)]/θ²
    vmle = n / ((n + 1) ** 2 * (n + 2))
    # (b) 指数平滑
    alpha, f0, y1, y2 = 0.3, 4.0, 5.0, 3.6
    f1 = alpha * y1 + (1 - alpha) * f0
    f2 = alpha * y2 + (1 - alpha) * f1
    g1 = (1 - alpha) * y1 + alpha * f0
    g2 = (1 - alpha) * y2 + alpha * g1
    return {
        "n": n, "n2": n + 2, "n1": n + 1, "v1": f"\\frac{{1}}{{{3 * n}}}", "v2": f"\\frac{{1}}{{{n * (n + 2)}}}",
        "ratio": s(v1 / v2, 2), "r_n1": s((n + 1) / 3, 2), "r_n": s(n / 3, 2), "r_mle": s(v1 / vmle, 2),
        "r_one": s(1, 2), "den3n": 3 * n, "dennn": n * (n + 2),
        "alpha": s(alpha, 1), "beta": s(1 - alpha, 1), "f0": s(f0, 1), "y1": s(y1, 1), "y2": s(y2, 1),
        "f1": s(f1, 2), "f2": s(f2, 2), "g2": s(g2, 2), "avg": s((f0 + y1 + y2) / 3, 2),
        "f1a": s(alpha * y1, 2), "f1b": s((1 - alpha) * f0, 2), "f2a": s(alpha * y2, 2), "f2b": s((1 - alpha) * f1, 3),
    }


def p07():
    n, k = 20, 2
    b1, b2 = 14.0, 9.5
    se1, se2, cov = 1.5, 1.8, 1.5
    v1, v2 = se1 ** 2, se2 ** 2
    vd = v1 + v2 - 2 * cov
    t = (b1 - b2) / sqrt(vd)
    t_nocov = (b1 - b2) / sqrt(v1 + v2)
    t_plus = (b1 - b2) / sqrt(v1 + v2 + 2 * cov)
    t_sum = (b1 - b2) / (se1 + se2)
    t_once = (b1 - b2) / sqrt(v1 + v2 - cov)
    df = n - k - 1
    tc = 2.110
    # (b) PRESS 残差
    y7, yhat7, h7 = 48.0, 51.2, 0.30
    e7 = y7 - yhat7
    ep = e7 / (1 - h7)
    pred = y7 - ep
    # (c) Huber の重み
    rr, sig, kk = -6.4, 2.0, 1.345
    u = abs(rr) / sig
    w = min(1, kk / u)
    w_raw = kk / abs(rr)
    w_sq = (kk / u) ** 2
    c_bi = 4.685
    w_bi = (1 - (u / c_bi) ** 2) ** 2
    return {
        "n": n, "k": k, "df": df, "b1": s(b1, 1), "b2": s(b2, 1), "se1": s(se1, 1), "se2": s(se2, 1),
        "cov": s(cov, 1), "v1": s(v1, 2), "v2": s(v2, 2), "cov2": s(2 * cov, 1), "vd": s(vd, 2),
        "sdd": s(sqrt(vd), 4), "diff": s(b1 - b2, 1), "t": s(t, 2), "t_nocov": s(t_nocov, 2),
        "t_plus": s(t_plus, 2), "t_sum": s(t_sum, 2), "t_once": s(t_once, 2), "tc": s(tc, 3),
        "corr": s(cov / (se1 * se2), 3), "vsum": s(v1 + v2, 2),
        "y7": s(y7, 1), "yhat7": s(yhat7, 1), "h7": s(h7, 2), "e7": s(e7, 1), "h1": s(1 - h7, 2),
        "ep": s(ep, 4), "pred": r(pred, 2), "pred_s": s(pred, 2), "pred_wrong": s(y7 - e7, 2),
        "pred_h": s(y7 - e7 * (1 - h7), 2),
        "r": s(rr, 1), "sig": s(sig, 1), "kk": s(kk, 3), "u": s(u, 1), "w": s(w, 3), "w4": s(w, 4),
        "w_raw": s(w_raw, 3), "w_sq": s(w_sq, 3), "w_bi": s(w_bi, 3), "w_one": s(1, 3), "cbi": s(c_bi, 3),
    }


def p08():
    # (a) CMH 検定：層ごとに (新・不良, 新・良, 旧・不良, 旧・良)
    tabs = [(8, 92, 18, 82), (4, 76, 11, 69)]
    rows = []
    num = 0.0
    var = 0.0
    for a, b, c, d in tabs:
        n1, n2 = a + b, c + d
        m1, m2 = a + c, b + d
        N = n1 + n2
        E = n1 * m1 / N
        V = n1 * n2 * m1 * m2 / (N ** 2 * (N - 1))
        rows.append((a, b, c, d, n1, n2, m1, m2, N, E, V))
        num += a - E
        var += V
    Es = [r(x[9], 3) for x in rows]
    Vs = [r(x[10], 4) for x in rows]
    # 表示した丸めた途中の値から計算しても同じ答えになることを確かめる
    num_s = sum(x[0] for x in rows) - sum(Es)
    cmh = num ** 2 / var
    cmh_s = num_s ** 2 / sum(Vs)
    assert r(cmh, 2) == r(cmh_s, 2), (cmh, cmh_s)
    # 合併した表（層を無視）のカイ二乗
    A = sum(x[0] for x in rows); B = sum(x[1] for x in rows); C = sum(x[2] for x in rows); Dd = sum(x[3] for x in rows)
    out = {"cmh": r(cmh, 2), "cmh_s": s(cmh, 2), "num": s(num_s, 3), "num2": s(num_s ** 2, 4),
           "vsum": s(sum(Vs), 4), "chi1": 3.84}
    for i, (a, b, c, d, n1, n2, m1, m2, N, E, V) in enumerate(rows, 1):
        out.update({f"a{i}": a, f"b{i}": b, f"c{i}": c, f"d{i}": d, f"n1{i}": n1, f"n2{i}": n2,
                    f"m1{i}": m1, f"m2{i}": m2, f"N{i}": N, f"E{i}": s(E, 3), f"V{i}": s(V, 4),
                    f"N{i}m1": N - 1, f"N{i}sq": N ** 2})
    out["asum"] = A
    out["Esum"] = s(sum(Es), 3)
    # (b) 差の差
    a0, a1, b0, b1 = 4.0, 2.5, 3.8, 3.3
    did = (a1 - a0) - (b1 - b0)
    out.update({"pa0": s(a0, 1), "pa1": s(a1, 1), "pb0": s(b0, 1), "pb1": s(b1, 1),
                "dA": s(a1 - a0, 1), "dB": s(b1 - b0, 1), "did": s(did, 1),
                "post": s(a1 - b1, 1), "pre": s(a0 - b0, 1), "did_rev": s((a1 - a0) + (b1 - b0), 1)})
    # (c) ポアソン回帰の率比の信頼区間
    beta, se, z = -0.35, 0.12, 1.96
    lo, hi = beta - z * se, beta + z * se
    ex = {-0.35: 0.7047, -0.5852: 0.5570, -0.1148: 0.8915, 0.5852: 1.7954, 0.1148: 1.1216}
    assert abs(lo - -0.5852) < 1e-9 and abs(hi - -0.1148) < 1e-9
    rr = ex[-0.35]
    out.update({"beta": s(beta, 2), "se": s(se, 2), "z": s(z, 2), "zse": s(z * se, 4), "lo": s(lo, 4), "hi": s(hi, 4),
                "e_b": s(ex[-0.35], 4), "e_lo": s(ex[-0.5852], 4), "e_hi": s(ex[-0.1148], 4),
                "e_plo": s(ex[0.1148], 4), "e_phi": s(ex[0.5852], 4),
                "ci_ok": f"({s(ex[-0.5852], 3)},\\ {s(ex[-0.1148], 3)})",
                "ci_lin": f"({s(rr - z * se, 3)},\\ {s(rr + z * se, 3)})",
                "ci_raw": f"({s(lo, 3)},\\ {s(hi, 3)})",
                "ci_inv": f"({s(ex[0.1148], 3)},\\ {s(ex[0.5852], 3)})",
                "ci_delta": f"({s(rr - z * rr * se, 3)},\\ {s(rr + z * rr * se, 3)})",
                "ok_lo": s(ex[-0.5852], 3), "ok_hi": s(ex[-0.1148], 3),
                "red_lo": s((1 - ex[-0.1148]) * 100, 1), "red_hi": s((1 - ex[-0.5852]) * 100, 1)})
    return out


def p09():
    k, n = 3, 6
    means = [102.3, 98.1, 96.5]
    sse = 121.5
    dfe = k * n - k
    mse = sse / dfe
    q = 3.67
    t = 2.131
    hsd = q * sqrt(mse / n)
    lsd = t * sqrt(2 * mse / n)
    return {
        "k": k, "n": n, "N": k * n, "m1": s(means[0], 1), "m2": s(means[1], 1), "m3": s(means[2], 1),
        "sse": s(sse, 1), "dfe": dfe, "mse": s(mse, 1), "q": s(q, 2), "tq": s(t, 3),
        "msen": s(mse / n, 2), "rt": s(sqrt(mse / n), 4),
        "hsd": s(hsd, 2), "lsd": s(lsd, 2), "hsd_2": s(q * sqrt(2 * mse / n), 2), "hsd_non": s(q * sqrt(mse), 2),
        "hsd_nosqrt": s(q * mse / n, 2),
        "d12": s(means[0] - means[1], 1), "d13": s(means[0] - means[2], 1), "d23": s(means[1] - means[2], 1),
    }


def p10():
    Nh = [2000, 3000, 5000]
    N = sum(Nh)
    n = 100
    sh = [4.0, 6.0, 5.0]
    W = [x / N for x in Nh]
    nh = [n * w for w in W]
    terms = [w * w * sv * sv / m for w, sv, m in zip(W, sh, nh)]
    V = sum(terms)
    se = sqrt(V)
    # 誤り：単純平均の分散で比べる / W を掛けない
    v_now = sum(sv * sv / m for sv, m in zip(sh, nh))
    v_w1 = sum(w * sv * sv / m for w, sv, m in zip(W, sh, nh))
    out = {"N": N, "n": n, "V": s(V, 4), "se": r(se, 3), "se_s": s(se, 3), "se4": s(se, 4)}
    for i in range(3):
        out.update({f"N{i}": Nh[i], f"W{i}": s(W[i], 1), f"n{i}": int(nh[i]), f"s{i}": s(sh[i], 1),
                    f"s2{i}": s(sh[i] ** 2, 0), f"t{i}": s(terms[i], 3)})
    out["se_now"] = s(sqrt(v_now), 3)
    out["se_w1"] = s(sqrt(v_w1), 3)
    return out


def p11():
    # (a) 1次元の2次判別
    m0, s0, m1, s1 = 50.0, 2.0, 54.0, 4.0
    ln2 = 0.6931
    # 4(x-50)^2 - (x-54)^2 > 32 ln 2  →  3x^2 - 292x + (7084 - 32 ln2) > 0
    a_, b_, c0 = 3, -292, 4 * m0 ** 2 - m1 ** 2
    c = c0 - 32 * ln2
    disc = b_ ** 2 - 4 * a_ * c
    xl = (-b_ - sqrt(disc)) / (2 * a_)
    xh = (-b_ + sqrt(disc)) / (2 * a_)
    disc0 = b_ ** 2 - 4 * a_ * c0
    yl = (-b_ - sqrt(disc0)) / (2 * a_)
    yh = (-b_ + sqrt(disc0)) / (2 * a_)
    # (b) 重心法のランス・ウィリアムズ更新
    ni, nj = 2, 1
    dki, dkj, dij = 37.0, 40.0, 9.0
    ai, aj, bb = ni / (ni + nj), nj / (ni + nj), -ni * nj / (ni + nj) ** 2
    dnew = ai * dki + aj * dkj + bb * dij
    d_nob = ai * dki + aj * dkj
    d_half = 0.5 * dki + 0.5 * dkj - 0.25 * dij
    # (c) 因子得点（回帰法）
    lam = [0.9, 0.8, 0.6]
    z = [1.2, 0.5, -0.4]
    psi = [1 - l * l for l in lam]
    q = [l * l / p for l, p in zip(lam, psi)]
    g = [l * zz / p for l, zz, p in zip(lam, z, psi)]
    f_reg = sum(g) / (1 + sum(q))
    f_bart = sum(g) / sum(q)
    lz = sum(l * zz for l, zz in zip(lam, z))
    return {
        "m0": s(m0, 0), "s0": s(s0, 0), "m1": s(m1, 0), "s1": s(s1, 0), "v0": s(s0 ** 2, 0), "v1": s(s1 ** 2, 0),
        "ln2": s(ln2, 4), "c32": s(32 * ln2, 4), "c0": s(c0, 0), "c": s(c, 4), "disc": s(disc, 4),
        "rdisc": s(sqrt(disc), 4), "xl": s(xl, 2), "xh": s(xh, 2), "mid": s((m0 + m1) / 2, 2),
        "disc0": s(disc0, 0), "yl": s(yl, 2), "yh": s(yh, 2),
        "ni": ni, "nj": nj, "dki": s(dki, 0), "dkj": s(dkj, 0), "dij": s(dij, 0),
        "dnew": s(dnew, 2), "d_nob": s(d_nob, 2), "d_half": s(d_half, 2), "d_min": s(min(dki, dkj), 2),
        "d_max": s(max(dki, dkj), 2), "t_i": s(ai * dki, 3), "t_j": s(aj * dkj, 3), "t_b": s(bb * dij, 1),
        "l1": s(lam[0], 1), "l2": s(lam[1], 1), "l3": s(lam[2], 1), "z1": s(z[0], 1), "z2": s(z[1], 1), "z3": s(z[2], 1),
        "p1": s(psi[0], 2), "p2": s(psi[1], 2), "p3": s(psi[2], 2),
        "q1": s(q[0], 4), "q2": s(q[1], 4), "q3": s(q[2], 4), "qs": s(sum(q), 4), "qs1": s(1 + sum(q), 4),
        "g1": s(g[0], 4), "g2": s(g[1], 4), "g3": s(g[2], 4), "gs": s(sum(g), 4),
        "f_reg": s(f_reg, 3), "f_reg4": s(f_reg, 4), "f_bart": s(f_bart, 3), "f_lz": s(lz, 3),
        "f_ls": s(lz / sum(l * l for l in lam), 3), "f_mean": s(sum(z) / 3, 3),
    }


def p12():
    lam = 10.0          # 到着率（枚/時）
    mean_s = 4.5        # 平均計測時間（分）
    mu = 60 / mean_s    # 処理率（枚/時）
    rho = lam / mu
    L = rho / (1 - rho)
    W = L / lam * 60            # 平均滞在時間（分）
    Wq = W - mean_s             # 平均待ち時間（分）
    Lq = lam * Wq / 60
    return {
        "lam": s(lam, 0), "ms": s(mean_s, 1), "mu": s(mu, 4), "rho": s(rho, 2), "L": s(L, 2),
        "W": s(W, 1), "Wh": s(W / 60, 2), "Wq": s(Wq, 1), "Wqh": s(Wq / 60, 3),
        "Lq": r(Lq, 2), "Lq_s": s(Lq, 2), "Lq_f": s(rho ** 2 / (1 - rho), 4),
        "L_wrong": s(L, 2), "Lq_min": s(lam * Wq, 1),
    }


def values():
    return {"s28p01": p01(), "s28p02": p02(), "s28p03": p03(), "s28p04": p04(), "s28p05": p05(),
            "s28p06": p06(), "s28p07": p07(), "s28p08": p08(), "s28p09": p09(), "s28p10": p10(),
            "s28p11": p11(), "s28p12": p12()}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=150)
