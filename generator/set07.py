"""模擬試験 第7回（set07）の数値。values() が {大問id: {名前: 値}} を返す。題材は金融・保険・経済指標・不動産。

問題文に示した値（e^x、べき乗、χ² 表・正規分布表の点、固有ベクトルの丸めた成分など）は、
ここでも同じ丸めた値を使って答えを計算する（一般電卓で問題文の値だけから同じ答えになるように）。
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


def distinct(*xs):
    assert len(set(xs)) == len(xs), xs


def p01():
    # (a) 2年連続の延滞からのベイズ更新（層が与えられたとき各年の延滞は条件付き独立）
    pH, dH, dN = 0.08, 0.30, 0.05
    pN = 1 - pH
    num = pH * dH ** 2
    den = num + pN * dN ** 2
    post = num / den
    post1 = pH * dH / (pH * dH + pN * dN)
    # (b) 多項分布の2つの度数の相関
    n, p1, p2, p3 = 200, 0.90, 0.07, 0.03
    cov = -n * p2 * p3
    v2, v3 = n * p2 * (1 - p2), n * p3 * (1 - p3)
    rho = cov / sqrt(v2 * v3)
    rho_nodenom = -sqrt(p2 * p3)
    c_rho = [s(rho, 3), "0.000", s(cov, 3), s(rho_nodenom, 3), s(-rho, 3)]
    distinct(*c_rho)
    # (c) デフォルト0件のときの片側95%上側信頼限界（正確法）
    m = 60
    q05, q025, q95 = 0.95130, 0.94037, 0.99915  # 0.05^(1/60), 0.025^(1/60), 0.95^(1/60)
    up = 1 - q05
    c_up = [s(up, 4), s(3 / m, 4), s(1 - q025, 4), s(1 - q95, 4), "0.0000"]
    distinct(*c_up)
    return {
        "pH": pH, "pN": s(pN, 2), "dH": s(dH, 2), "dN": s(dN, 2),
        "dH2": s(dH ** 2, 2), "dN2": s(dN ** 2, 4), "num": s(num, 4), "den2": s(pN * dN ** 2, 4), "den": s(den, 4),
        "post": r(post, 3), "post_s": s(post, 3), "post4": s(post, 4), "post1": s(post1, 3),
        "n": n, "p1": s(p1, 2), "p2": s(p2, 2), "p3": s(p3, 2),
        "cov": s(cov, 2), "v2": s(v2, 2), "v3": s(v3, 2), "v23": s(v2 * v3, 4),
        "rho": c_rho[0], "rho_zero": c_rho[1], "rho_cov": c_rho[2], "rho_nd": c_rho[3], "rho_pos": c_rho[4],
        "q2": s(1 - p2, 2), "q3": s(1 - p3, 2), "p23": s(p2 * p3, 4), "q23": s((1 - p2) * (1 - p3), 4),
        "m": m, "q05": s(q05, 5), "q025": s(q025, 5), "q95": s(q95, 5),
        "up": c_up[0], "up3": c_up[1], "up2": c_up[2], "up95": c_up[3], "up0": c_up[4],
    }


def p02():
    # (a) 線形の死力 mu(t) = a + b (t - 40)（t >= 40）
    a, b = 0.002, 0.0004
    x0, x1 = 50, 60
    H = a * (x1 - x0) + b * ((x1 - 40) ** 2 - (x0 - 40) ** 2) / 2
    H40 = a * (x1 - 40) + b * (x1 - 40) ** 2 / 2
    Hend = (a + b * (x1 - 40)) * (x1 - x0)
    ex = {0.08: 0.9231, 0.12: 0.8869, 0.10: 0.9048}
    assert abs(H - 0.08) < 1e-12 and abs(H40 - 0.12) < 1e-12 and abs(Hend - 0.10) < 1e-12
    surv = ex[0.08]
    c = [s(surv, 3), s(ex[0.12], 3), s(ex[0.10], 3), s(1 - H, 3), s(1 - surv, 3)]
    distinct(*c)
    # (b) 1年定期保険の保険料（中心極限定理）
    n, q, B, z = 5000, 0.02, 500, 2.326
    EN = n * q
    VN = n * q * (1 - q)
    sdN = sqrt(VN)
    prem = B * q + z * B * sdN / n
    return {
        "a": a, "b": b, "x0": x0, "x1": x1, "H1": s(a * (x1 - x0), 2), "H2": s(b * ((x1 - 40) ** 2 - (x0 - 40) ** 2) / 2, 2),
        "H": s(H, 2), "H40": s(H40, 2), "Hend": s(Hend, 2), "mu60": s(a + b * (x1 - 40), 3),
        "e08": s(ex[0.08], 4), "e12": s(ex[0.12], 4), "e10": s(ex[0.10], 4),
        "surv": c[0], "s40": c[1], "send": c[2], "slin": c[3], "dead": c[4],
        "n": n, "q": q, "B": B, "z": z, "EN": s(EN, 0), "VN": s(VN, 0), "sdN": s(sdN, 4),
        "ES": s(B * EN, 0), "sdS": s(B * sdN, 2), "load": s(z * B * sdN / n, 4), "pure": s(B * q, 0),
        "prem": r(prem, 2), "prem4": s(prem, 4),
        "w_var": s(B * q + z * B * VN / n, 2), "w_nq": s(B * q + z * B * sqrt(EN) / n, 2),
        "w_1645": s(B * q + 1.645 * B * sdN / n, 2), "w_sqrtn": s(B * q + z * B * sdN / sqrt(n), 2),
    }


def p03():
    # (a) ドルコスト平均法の平均取得単価（調和平均）
    prices = [10000, 12500, 8000, 10000]
    amt = 30000
    units = [amt / p for p in prices]
    hm = len(prices) / sum(1 / p for p in prices)
    am = sum(prices) / len(prices)
    # (b) 正規・正規モデル
    m0, t0 = 2.0, 1.5          # 事前 N(2, 1.5^2)
    n, xbar, sig = 9, 5.0, 6.0
    vd = sig ** 2 / n
    prec = 1 / t0 ** 2 + 1 / vd
    vpost = 1 / prec
    mpost = (m0 / t0 ** 2 + xbar / vd) / prec
    sdp = sqrt(vpost)
    z = 1.96
    ci = lambda c, h: f"({s(c - h, 2)},\\ {s(c + h, 2)})"
    mswap = (m0 / vd + xbar / t0 ** 2) / prec
    out = {
        "ci_ok": ci(mpost, z * sdp), "ci_freq": ci(xbar, z * sqrt(vd)), "ci_swap": ci(mswap, z * sdp),
        "ci_var": ci(mpost, z * vpost), "ci_90": ci(mpost, 1.645 * sdp),
    }
    distinct(*out.values())
    out.update({
        "pr1": prices[0], "pr2": prices[1], "pr3": prices[2], "pr4": prices[3], "amt": amt,
        "u1": s(units[0], 1), "u2": s(units[1], 1), "u3": s(units[2], 2), "u4": s(units[3], 1),
        "usum": s(sum(units), 2), "tot": amt * 4, "hm": r(hm, 1), "hm4": s(hm, 4), "am": s(am, 0),
        "invsum": s(sum(1 / p for p in prices), 6),
        "m0": s(m0, 1), "t0": s(t0, 1), "t02": s(t0 ** 2, 2), "n": n, "xbar": s(xbar, 1), "sig": s(sig, 1),
        "vd": s(vd, 1), "pr0": s(1 / t0 ** 2, 4), "prd": s(1 / vd, 2), "prec": s(prec, 4), "vpost": s(vpost, 2),
        "mpost": s(mpost, 2), "sdp": s(sdp, 1), "hw": s(z * sdp, 3), "mnum": s(m0 / t0 ** 2 + xbar / vd, 4),
        "w0": s((1 / t0 ** 2) / prec, 2), "w1": s((1 / vd) / prec, 2), "mswap": s(mswap, 2),
    })
    return out


def p04():
    # (b) パレート分布の下限母数の不偏推定
    alpha, n, xmin = 3, 10, 1.18
    na = n * alpha
    unb = (na - 1) / na * xmin
    c = [s(unb, 3), s(xmin, 3), s((n - 1) / n * xmin, 3), s((na + 1) / na * xmin, 3), s((alpha - 1) / alpha * xmin, 3)]
    distinct(*c)
    return {"alpha": alpha, "n": n, "xmin": s(xmin, 2), "na": na, "na1": na - 1,
            "unb": c[0], "mle": c[1], "w_n": c[2], "w_p": c[3], "w_a": c[4], "ratio": s((na - 1) / na, 4)}


def p05():
    # (a) ネイマン・ピアソン：X ~ Bin(12, p)、H0: p=0.5 vs H1: p=0.75
    n = 12
    tail = lambda c: sum(comb(n, k) for k in range(c, n + 1))
    a10, a9 = tail(10) / 2 ** n, tail(9) / 2 ** n
    pw = {10: 0.05631, 11: 0.04224, 12: 0.03168}   # 0.75^10, 0.75^11, 0.75^12（問題文の値）
    t10 = comb(12, 10) * pw[10] * 0.25 ** 2
    t11 = comb(12, 11) * pw[11] * 0.25
    t12 = pw[12]
    power = t10 + t11 + t12
    p75_9 = 0.07508  # 0.75^9
    t9 = comb(12, 9) * p75_9 * 0.25 ** 3
    # (b) クラスカル・ウォリス
    A = [-3.2, -1.5, 1.1, 2.8]
    Bv = [0.4, 2.3, 5.2, 6.0]
    C = [3.6, 4.5, 7.4, 8.9]
    allv = sorted(A + Bv + C)
    rank = {v: i + 1 for i, v in enumerate(allv)}
    R = [sum(rank[v] for v in g) for g in (A, Bv, C)]
    N = 12
    Ssq = sum(x * x / 4 for x in R)
    Hs = 12 / (N * (N + 1)) * Ssq - 3 * (N + 1)
    Hw = 12 / (N * (N + 1)) * Ssq - 3 * N
    d = {"n": n, "c4096": 2 ** n, "t10c": tail(10), "t9c": tail(9), "a10": s(a10, 4), "a9": s(a9, 4),
         "pw10": s(pw[10], 5), "pw11": s(pw[11], 5), "pw12": s(pw[12], 5), "pw9": s(p75_9, 5),
         "t10": s(t10, 5), "t11": s(t11, 5), "t12": s(t12, 5), "power": r(power, 3), "power4": s(power, 4),
         "power9": s(power + t9, 3),
         "R1": R[0], "R2": R[1], "R3": R[2], "Ssq": s(Ssq, 1), "H": s(Hs, 2), "Hw": s(Hw, 2),
         "H4": s(Hs, 4), "coef": s(12 / (N * (N + 1)), 6)}
    for i, v in enumerate(A):
        d[f"a{i}"] = v
    for i, v in enumerate(Bv):
        d[f"b{i}"] = v
    for i, v in enumerate(C):
        d[f"c{i}"] = v
    return d


def p06():
    # (a) 観測期間の違う2支店のポアソン件数の比較（総件数で条件付けた二項分布）
    xa, ta, xb, tb = 30, 2, 27, 3
    tot = xa + xb
    p0 = ta / (ta + tb)
    E = tot * p0
    V = tot * p0 * (1 - p0)
    z = (xa - E) / sqrt(V)
    zh = (xa - tot / 2) / sqrt(tot / 4)
    zp = (xa - E) / sqrt(E)
    phi195 = 0.9744
    assert s(z, 2) == "1.95"
    # (b) オフセットつきポアソン回帰
    b0, b1, expo = -1.20, 0.45, 8
    e = {-0.75: 0.4724, -1.20: 0.3012, 0.45: 1.5683}
    c = [s(expo * e[-0.75], 2), s(e[-0.75], 2), s(expo * e[-1.20], 2), s(expo * e[0.45], 2), s(e[-0.75] / expo, 2)]
    distinct(*c)
    return {"xa": xa, "ta": ta, "xb": xb, "tb": tb, "tot": tot, "p0": s(p0, 1), "q0": s(1 - p0, 1),
            "E": s(E, 1), "V": s(V, 2), "sd": s(sqrt(V), 4), "diff": s(xa - E, 1), "z": s(z, 2), "z4": s(z, 4),
            "zh": s(zh, 2), "zp": s(zp, 2), "phi": s(phi195, 4), "pv2": s(2 * (1 - phi195), 4), "pv1": s(1 - phi195, 4),
            "ra": s(xa / ta, 1), "rb": s(xb / tb, 1),
            "b0": s(b0, 2), "b1": s(b1, 2), "expo": expo, "lin": s(b0 + b1, 2),
            "em075": s(e[-0.75], 4), "em120": s(e[-1.20], 4), "ep045": s(e[0.45], 4),
            "mu": c[0], "mu_nooff": c[1], "mu_nob1": c[2], "mu_nob0": c[3], "mu_div": c[4]}


def p07():
    # 吸収確率：正常 C, 延滞 D, 完済 P, デフォルト F
    cc, cd, cp = 0.85, 0.05, 0.10
    dc, dd, df = 0.40, 0.30, 0.30
    # a_C = cc a_C + cd a_D ; a_D = dc a_C + dd a_D + df
    k = cd / (1 - cc)              # a_C = k a_D
    aD = df / (1 - dd - dc * k)
    aC = k * aD
    first = cd / (cd + cp)
    exitD = df / (df + dc)
    c = [s(aC, 3), s(aD, 3), s(first * exitD, 3), s(first, 3), s(cd * df, 3)]
    distinct(*c)
    return {"cc": s(cc, 2), "cd": s(cd, 2), "cp": s(cp, 2), "dc": s(dc, 2), "dd": s(dd, 2), "df": s(df, 2),
            "k": s(k, 4), "kfrac": "1/3", "one_cc": s(1 - cc, 2), "one_dd": s(1 - dd, 2), "dck": s(dc * k, 4),
            "denD": s(1 - dd - dc * k, 4),
            "aC": c[0], "aD": c[1], "w_path": c[2], "w_first": c[3], "w_prod": c[4],
            "aC_frac": "3/17", "aD_frac": "9/17", "exitD": s(exitD, 3)}


def p08():
    # (a) VIF
    R2 = 0.36
    vif = 1 / (1 - R2)
    # (b) ブルーシュ・ペーガン
    n, R2bp = 60, 0.21
    LM = n * R2bp
    # (c) 1個抜き交差検証（単回帰 5 件）
    x = [40, 50, 60, 70, 90]
    y = [29, 36, 41, 49, 58]
    nn = len(x)
    xb, yb = sum(x) / nn, sum(y) / nn
    Sxx = sum((xi - xb) ** 2 for xi in x)
    Sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    b1 = Sxy / Sxx
    b0 = yb - b1 * xb
    e = [yi - (b0 + b1 * xi) for xi, yi in zip(x, y)]
    h = [1 / nn + (xi - xb) ** 2 / Sxx for xi in x]
    er = [r(v, 3) for v in e]
    hr = [r(v, 3) for v in h]
    pr = [ei / (1 - hi) for ei, hi in zip(er, hr)]
    cv = sum(v * v for v in pr) / nn
    mse_tr = sum(v * v for v in er) / nn
    cv_noh = sum((ei / (1 - hi)) for ei, hi in zip(er, hr))
    d = {"R2": R2, "vif": s(vif, 4), "se": s(sqrt(vif), 2), "one_r2": s(1 - R2, 2), "se_inv": s(sqrt(1 - R2), 2),
         "vif_r2": s(1 / R2, 4), "se_r2": s(sqrt(1 / R2), 4),
         "n": n, "R2bp": R2bp, "LM": s(LM, 1), "LMsq": s(sqrt(LM), 2),
         "Sxx": s(Sxx, 0), "xbar": s(xb, 0), "b0": s(b0, 3), "b1": s(b1, 4),
         "cv": r(cv, 2), "cv4": s(cv, 4), "mse_tr": s(mse_tr, 4), "cv_h": s(sum((ei / (1 - hi)) ** 2 for ei, hi in zip(er, hr)), 4),
         "cv_wrong_sq": s(sum(ei ** 2 / (1 - hi) for ei, hi in zip(er, hr)) / nn, 4),
         "cv_plus": s(sum((ei / (1 + hi)) ** 2 for ei, hi in zip(er, hr)) / nn, 4)}
    for i in range(nn):
        d[f"x{i}"] = x[i]
        d[f"y{i}"] = y[i]
        d[f"e{i}"] = s(er[i], 3)
        d[f"h{i}"] = s(hr[i], 3)
        d[f"o{i}"] = s(1 - hr[i], 3)
        d[f"pr{i}"] = s(pr[i], 4)
        d[f"pq{i}"] = s(pr[i] ** 2, 4)
    _ = cv_noh
    return d


def p09():
    # (a) 欠落変数バイアス
    b1, b2 = 0.60, 0.03
    varY, covYW = 25.0, 150.0
    delta = covYW / varY
    plim = b1 + b2 * delta
    c = [s(plim, 3), s(b1, 3), s(b1 - b2 * delta, 3), s(b1 + b2 * covYW, 3), s(b1 + b2 / delta, 3)]
    distinct(*c)
    # (b) ユール・ウォーカー推定 AR(2)
    r1, r2 = 0.6, 0.5
    phi1 = r1 * (1 - r2) / (1 - r1 ** 2)
    phi2 = (r2 - r1 ** 2) / (1 - r1 ** 2)
    pair = lambda a, b: f"\\hat\\phi_1={s(a, 3)},\\ \\hat\\phi_2={s(b, 3)}"
    cc = [pair(phi1, phi2), pair(r1, r2), pair(r1, r2 - r1 ** 2), pair(r1, 0), pair(phi2, phi1)]
    distinct(*cc)
    return {"b1": s(b1, 2), "b2": s(b2, 2), "varY": s(varY, 0), "covYW": s(covYW, 0), "delta": s(delta, 0),
            "bias": s(b2 * delta, 2),
            "plim": c[0], "w_none": c[1], "w_minus": c[2], "w_cov": c[3], "w_inv": c[4],
            "r1": r1, "r2": r2, "r1sq": s(r1 ** 2, 2), "den": s(1 - r1 ** 2, 2), "n1": s(r1 * (1 - r2), 2),
            "n2": s(r2 - r1 ** 2, 2), "phi1": s(phi1, 3), "phi2": s(phi2, 3), "phi1_5": s(phi1, 5), "phi2_5": s(phi2, 5),
            "yw_ok": cc[0], "yw_naive": cc[1], "yw_sub": cc[2], "yw_ar1": cc[3], "yw_swap": cc[4]}


def p10():
    # (a) 変量効果の一元配置：級内相関
    k, m = 5, 6
    ssa, sse = 150.0, 250.0
    dfa, dfe = k - 1, k * (m - 1)
    msa, mse = ssa / dfa, sse / dfe
    sa2 = (msa - mse) / m
    icc = sa2 / (sa2 + mse)
    # (b) シンプソンのパラドックス
    X = {"hi": (100, 1), "lo": (400, 32)}
    Y = {"hi": (400, 8), "lo": (100, 10)}
    rate = lambda t: 100 * t[1] / t[0]
    tx = 100 * (X["hi"][1] + X["lo"][1]) / 500
    ty = 100 * (Y["hi"][1] + Y["lo"][1]) / 500
    stdx = (rate(X["hi"]) + rate(X["lo"])) / 2
    stdy = (rate(Y["hi"]) + rate(Y["lo"])) / 2
    return {"k": k, "m": m, "N": k * m, "ssa": s(ssa, 0), "sse": s(sse, 0), "dfa": dfa, "dfe": dfe,
            "msa": s(msa, 1), "mse": s(mse, 1), "sa2": s(sa2, 4), "icc": r(icc, 3), "icc4": s(icc, 4),
            "tot_var": s(sa2 + mse, 4), "w_nom": s((msa - mse) / (msa - mse + mse), 4), "w_msa": s((msa - mse) / msa, 4),
            "xhn": X["hi"][0], "xhd": X["hi"][1], "xln": X["lo"][0], "xld": X["lo"][1],
            "yhn": Y["hi"][0], "yhd": Y["hi"][1], "yln": Y["lo"][0], "yld": Y["lo"][1],
            "rxh": s(rate(X["hi"]), 1), "rxl": s(rate(X["lo"]), 1), "ryh": s(rate(Y["hi"]), 1), "ryl": s(rate(Y["lo"]), 1),
            "tx": s(tx, 1), "ty": s(ty, 1), "stdx": s(stdx, 2), "stdy": s(stdy, 2),
            "xd": X["hi"][1] + X["lo"][1], "yd": Y["hi"][1] + Y["lo"][1],
            "rr_hi": s(rate(X["hi"]) / rate(Y["hi"]), 1), "rr_hi_w": s(rate(Y["hi"]) / rate(X["hi"]), 1)}


def p11():
    # 金利変化の主成分得点（共分散行列の固有ベクトル、問題文の丸めた値で計算）
    lam = [44.34, 6.96, 2.69]
    v1 = [0.517, 0.645, 0.563]
    v2 = [0.713, 0.040, -0.700]
    x = [5, 7, 3]
    sc1 = sum(a * b for a, b in zip(v1, x))
    sc2 = sum(a * b for a, b in zip(v2, x))
    c = [s(sc1, 2), s(sc2, 2), s(sc1 / sqrt(lam[0]), 2), s(sum(x), 2), s(sc1 * lam[0], 1)]
    distinct(*c)
    return {"l1": s(lam[0], 2), "l2": s(lam[1], 2), "l3": s(lam[2], 2),
            "v11": s(v1[0], 3), "v12": s(v1[1], 3), "v13": s(v1[2], 3),
            "v21": s(v2[0], 3), "v22": s(v2[1], 3), "v23": s(v2[2], 3),
            "x1": x[0], "x2": x[1], "x3": x[2],
            "t1": s(v1[0] * x[0], 3), "t2": s(v1[1] * x[1], 3), "t3": s(v1[2] * x[2], 3),
            "sc": c[0], "sc2": c[1], "sc_std": c[2], "sc_sum": c[3], "sc_lam": c[4], "sc3": s(sc1, 3),
            "sqrtl1": s(sqrt(lam[0]), 3), "contrib": s(lam[0] / sum(lam), 3)}


def p12():
    # (a) 最尤因子分析の自由度
    p = 6
    df = lambda m: ((p - m) ** 2 - (p + m)) // 2
    d1, d2, d3 = df(1), df(2), df(3)
    assert (d1, d2, d3) == (9, 4, 0)
    nmom = p * (p + 1) // 2
    # (b) パス解析
    a, b, c = -0.5, 0.6, -0.2
    ind = a * b
    tot = ind + c
    ch = [s(tot, 2), s(ind, 2), s(c, 2), s(a + b + c, 2), s(a * b * c, 2)]
    distinct(*ch)
    return {"p": p, "d1": d1, "d2": d2, "d3": d3, "nmom": nmom, "par2": p * 2 + p - 1, "par2raw": p * 2 + p,
            "rot": 1, "nmom3": nmom, "par3": p * 3 + p - 3, "d2_rot": nmom - (p * 2 + p),
            "d2_sign": ((p - 2) ** 2 + (p + 2)) // 2, "m_sign": p - 1,
            "a": s(a, 1), "b": s(b, 1), "c": s(c, 1), "ind": ch[1], "tot": ch[0], "w_dir": ch[2], "w_sum": ch[3], "w_prod": ch[4]}


def p13():
    # PPS（評価額に比例）抽出のホーヴィッツ・トンプソン推定
    Ntot, X, n = 200, 2900, 4
    x = [15, 30, 24, 45]
    y = [21, 40, 30, 63]
    pi = [n * xi / X for xi in x]
    ht = sum(yi / p for yi, p in zip(y, pi))
    ratios = [yi / xi for xi, yi in zip(x, y)]
    d = {"Ntot": Ntot, "X": X, "n": n, "fac": s(X / n, 0), "rsum": s(sum(ratios), 6),
         "ht": r(ht / 100, 2), "ht_m": s(ht, 2), "ht_oku": s(ht / 100, 4),
         "w_mean": s(Ntot * sum(y) / n / 100, 2), "w_ratio": s(X * sum(y) / sum(x) / 100, 2),
         "ybar": s(sum(y) / n, 2), "ysum": sum(y), "xsum": sum(x)}
    for i in range(n):
        d[f"x{i}"] = x[i]
        d[f"y{i}"] = y[i]
        d[f"pi{i}"] = s(pi[i], 5)
        d[f"rt{i}"] = s(ratios[i], 4)
    return d


def values():
    return {
        "s07p01": p01(), "s07p02": p02(), "s07p03": p03(), "s07p04": p04(), "s07p05": p05(),
        "s07p06": p06(), "s07p07": p07(), "s07p08": p08(), "s07p09": p09(), "s07p10": p10(),
        "s07p11": p11(), "s07p12": p12(), "s07p13": p13(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
