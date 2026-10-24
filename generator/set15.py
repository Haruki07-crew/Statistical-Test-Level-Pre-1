"""模擬試験 第15回（set15）の数値。values() が {大問id: {名前: 値}} を返す。

題材：医療機器・臨床検査・看護・介護。
問題文に示した分布の値（χ² の上側確率・パーセント点、F・t・q の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
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
    # (a) 同時密度 f(x,y)=x+y（0<x<1, 0<y<1）。X=搬送時間, Y=分析時間（時間）
    x0, y0 = Fraction(1, 4), Fraction(1, 2)
    fx = x0 + Fraction(1, 2)                                  # 周辺密度 f_X(x0)
    num = x0 * (1 - y0) + (1 - y0 ** 2) / 2                   # ∫_{y0}^{1} (x0+y) dy
    cond = num / fx
    marg = (1 - y0 ** 2) / 2 + (1 - y0) / 2                   # P(Y>y0)=∫(y+1/2)dy
    joint_le = x0 ** 2 / 2 * (1 - y0) + x0 * (1 - y0 ** 2) / 2  # P(X<=x0, Y>y0)
    FX = x0 ** 2 / 2 + x0 / 2
    # (b) 標本標準偏差が 1.5σ を超える確率（χ²_9 の表）
    n, k = 10, 1.5
    chi = {2.25: 0.9869, 13.5: 0.1413, 20.25: 0.0164, 22.5: 0.0074}
    c_ok = (n - 1) * k ** 2
    c_n = n * k ** 2
    c_nosq = (n - 1) * k
    c_raw = k ** 2
    for c in (c_ok, c_n, c_nosq, c_raw):
        assert c in chi, c
    # (c) σ の 95% 信頼区間
    S = 2.6
    lo, hi = 2.70, 19.02                                      # χ²_9 の下側・上側 2.5% 点
    ss = (n - 1) * S ** 2
    vlo, vhi = ss / hi, ss / lo
    ss_n = n * S ** 2
    z_hw = 1.96 * S / sqrt(2 * (n - 1))
    return {
        "x0": "1/4", "y0": "1/2", "fx": "3/4",
        "num": s(float(num), 3), "cond": s(float(cond), 3), "c_marg": s(float(marg), 3),
        "c_joint": s(float(num), 3), "c_le": s(float(joint_le / FX), 3), "c_comp": s(float(1 - cond), 3),
        "t1": s(float(x0 * (1 - y0)), 3), "t2": s(float((1 - y0 ** 2) / 2), 3),
        "jle": s(float(joint_le), 6), "FX": s(float(FX), 5),
        "n": n, "df": n - 1, "k": k, "k2": s(k ** 2, 2), "cok": s(c_ok, 2), "cn": s(c_n, 1), "cnosq": s(c_nosq, 1),
        "craw": s(c_raw, 2),
        "q1": s(chi[2.25], 4), "q2": s(chi[13.5], 4), "q3": s(chi[20.25], 4), "q4": s(chi[22.5], 4),
        "b_ok": s(chi[c_ok], 4), "b_n": s(chi[c_n], 4), "b_nosq": s(chi[c_nosq], 4),
        "b_comp": s(1 - chi[c_ok], 4), "b_raw": s(chi[c_raw], 4),
        "S": S, "S2": s(S ** 2, 2), "ss": s(ss, 2), "lo": s(lo, 2), "hi": s(hi, 2),
        "vlo": s(vlo, 3), "vhi": s(vhi, 3), "sdlo": s(sqrt(vlo), 2), "sdhi": s(sqrt(vhi), 2),
        "vlo4": s(vlo, 4), "vhi4": s(vhi, 4), "sdlo4": s(sqrt(vlo), 4), "sdhi4": s(sqrt(vhi), 4),
        "w_nlo": s(sqrt(ss_n / hi), 2), "w_nhi": s(sqrt(ss_n / lo), 2), "ssn": s(ss_n, 1),
        "w_vlo": s(vlo, 2), "w_vhi": s(vhi, 2),
        "w_slo": s((n - 1) * S / hi, 2), "w_shi": s((n - 1) * S / lo, 2),
        "w_zlo": s(S - z_hw, 2), "w_zhi": s(S + z_hw, 2), "zhw": s(z_hw, 3),
    }


def p02():
    # (a) 3 社の輸液ポンプのアラーム：ベイズの定理
    w = [0.5, 0.3, 0.2]
    f = [0.10, 0.20, 0.35]
    joint = [a * b for a, b in zip(w, f)]
    pf = sum(joint)
    post = joint[2] / pf
    lik_norm = f[2] / sum(f)
    notf = (w[2] * (1 - f[2])) / sum(a * (1 - b) for a, b in zip(w, f))
    # (b) 3 回目の誤報が 10 回目のアラーム：負の二項分布
    p, k, m = 0.2, 3, 10
    prob = math.comb(m - 1, k - 1) * p ** k * (1 - p) ** (m - k)
    return {
        "wA": w[0], "wB": w[1], "wC": w[2], "fA": s(f[0], 2), "fB": s(f[1], 2), "fC": s(f[2], 2),
        "jA": s(joint[0], 3), "jB": s(joint[1], 3), "jC": s(joint[2], 3), "pf": s(pf, 3),
        "post": s(post, 3), "post4": s(post, 4), "c_prior": s(w[2], 3), "c_joint": s(joint[2], 3),
        "c_lik": s(lik_norm, 3), "c_not": s(notf, 3),
        "p": s(p, 1), "q": s(1 - p, 1), "k": k, "m": m, "km1": k - 1, "mm1": m - 1, "mk": m - k,
        "comb": math.comb(m - 1, k - 1), "c103": math.comb(m, k), "pk": s(p ** k, 3), "q7": s((1 - p) ** (m - k), 7),
        "ans": r(prob, 4), "ans6": s(prob, 6),
        "w_c10": s(math.comb(m, k) * p ** k * (1 - p) ** (m - k), 4),
        "w_c93": s(math.comb(m - 1, k) * p ** k * (1 - p) ** (m - k), 4),
        "w_bin": s(math.comb(m, k) * p ** k * (1 - p) ** (m - k), 4),
    }


def p03():
    # (a) 8 病棟の転倒件数の分散指数検定
    x = [2, 7, 1, 5, 8, 3, 0, 6]
    k = len(x)
    xb = sum(x) / k
    ssd = sum((v - xb) ** 2 for v in x)
    D = ssd / xb
    c7, c7h, c8 = 14.07, 16.01, 15.51
    assert c7 < D < c8
    ratio = (ssd / (k - 1)) / xb
    # (b) ランダム化検定
    mu0 = 6
    e6 = 0.002479
    p0 = e6
    p1 = mu0 * e6
    p2 = mu0 ** 2 / 2 * e6
    cdf1 = p0 + p1
    alpha = 0.05
    gamma = (alpha - cdf1) / p2
    return {
        "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "x5": x[4], "x6": x[5], "x7": x[6], "x8": x[7],
        "k": k, "km1": k - 1, "sumx": sum(x), "xb": s(xb, 1), "ssd": s(ssd, 1), "D": s(D, 2), "D1": s(D, 1),
        "c7": s(c7, 2), "c7h": s(c7h, 2), "c8": s(c8, 2), "ratio": s(ratio, 2), "s2": s(ssd / (k - 1), 3),
        "dev2": " + ".join(f"{(v - xb) ** 2:g}" for v in x),
        "mu0": mu0, "lam": 2, "months": 3, "e6": s(e6, 6),
        "p0": s(p0, 6), "p1": s(p1, 6), "p2": s(p2, 6), "cdf1": s(cdf1, 6), "cdf2": s(cdf1 + p2, 6),
        "num": s(alpha - cdf1, 6), "gamma": r(gamma, 2), "gamma4": s(gamma, 4),
        "w_ratio": s(alpha / (cdf1 + p2), 4), "w_comp": s(1 - gamma, 2),
    }


def p04():
    # (a) マルコフの不等式
    mu, a = 12, 60
    # (b) ガンマ分布のモーメント推定
    n, xb, v = 40, 12.6, 50.4
    k = xb ** 2 / v
    th = v / xb
    vu = v * n / (n - 1)
    return {
        "mu": mu, "a": a, "c_ok": s(mu / a, 3), "c_sq": s((mu / a) ** 2, 3), "c_dev": s(mu / (a - mu), 3),
        "c_comp": s(1 - mu / a, 3), "c_exp": s(0.0067, 4), "e5": s(0.0067, 4),
        "n": n, "xb": xb, "v": v, "xb2": s(xb ** 2, 2), "k": r(k, 2), "k4": s(k, 4), "th": s(th, 2),
        "vu": s(vu, 3), "w_unb": s(xb ** 2 / vu, 2), "w_inv": s(xb / v, 2), "w_th": s(th, 2),
    }


def p05():
    # (a) 偏相関係数
    rwg, rwa, rga = 0.50, -0.60, -0.55
    num = rwg - rwa * rga
    den2 = (1 - rwa ** 2) * (1 - rga ** 2)
    pc = num / sqrt(den2)
    # (b) 部分 F 検定
    n = 60
    rss1, p1 = 4.20, 1
    rss3, p3 = 3.40, 3
    q = p3 - p1
    dfe = n - p3 - 1
    mse = rss3 / dfe
    F = ((rss1 - rss3) / q) / mse
    # (c) マローズの Cp（σ² はフルモデルの残差平均平方）
    models = [("M1", "年齢", 4.20, 2), ("M2", "年齢・握力", 3.46, 3), ("M3", "年齢・下肢筋力", 3.70, 3),
              ("M4", "年齢・握力・下肢筋力", 3.40, 4)]
    cp = {m: rss / mse - n + 2 * kk for m, _, rss, kk in models}
    cp_noint = {m: rss / mse - n + 2 * (kk - 1) for m, _, rss, kk in models}
    mse_n = rss3 / n
    cp_n = {m: rss / mse_n - n + 2 * kk for m, _, rss, kk in models}
    best = min(cp, key=cp.get)
    assert best == "M2" and min(cp_noint, key=cp_noint.get) == "M2" and min(cp_n, key=cp_n.get) == "M2"
    out = {
        "rwg": s(rwg, 2), "rwa": s(rwa, 2), "rga": s(rga, 2), "prod": s(rwa * rga, 2), "num": s(num, 2),
        "rwa2": s(rwa ** 2, 2), "rga2": s(rga ** 2, 4), "den2": s(den2, 4), "den": s(sqrt(den2), 4),
        "pc": s(pc, 3), "c_raw": s(rwg, 3), "c_num": s(num, 3), "c_nosqrt": s(num / den2, 3),
        "c_lin": s(num / sqrt((1 - abs(rwa)) * (1 - abs(rga))), 3),
        "n": n, "rss1": s(rss1, 2), "rss3": s(rss3, 2), "q": q, "dfe": dfe, "drss": s(rss1 - rss3, 2),
        "mnum": s((rss1 - rss3) / q, 2), "mse": s(mse, 6), "F": s(F, 2), "F4": s(F, 4), "f05": 3.16,
        "F_q1": s((rss1 - rss3) / mse, 2), "F_57": s(((rss1 - rss3) / q) / (rss3 / (n - p3)), 2),
        "F_rss1": s(((rss1 - rss3) / q) / (rss1 / dfe), 2), "F_inv": s(mse / ((rss1 - rss3) / q), 2),
        "mse_n": s(mse_n, 5),
    }
    for m, lab, rss, kk in models:
        out[f"{m}rss"] = s(rss, 2)
        out[f"{m}k"] = kk
        out[f"{m}ratio"] = s(rss / mse, 3)
        out[f"{m}cp"] = s(cp[m], 2)
        out[f"{m}cpn"] = s(cp_noint[m], 2)
        out[f"{m}cpN"] = s(cp_n[m], 2)
    return out


def p06():
    # 外部スチューデント化残差
    n, e, h, s2 = 12, 1.60, 0.36, 0.64
    sd = sqrt(s2)
    ri = e / (sd * sqrt(1 - h))
    s2i = ((n - 2) * s2 - e ** 2 / (1 - h)) / (n - 3)
    ti = e / (sqrt(s2i) * sqrt(1 - h))
    s2i_w = ((n - 2) * s2 - e ** 2 / (1 - h)) / (n - 2)
    t_w = e / (sqrt(s2i_w) * sqrt(1 - h))
    t_noh = e / sqrt(s2i)
    return {
        "n": n, "e": s(e, 2), "h": s(h, 2), "s2": s(s2, 2), "sd": s(sd, 1), "omh": s(1 - h, 2),
        "sqomh": s(sqrt(1 - h), 1), "ri": s(ri, 2), "ns2": s((n - 2) * s2, 2), "e2": s(e ** 2, 2),
        "e2h": s(e ** 2 / (1 - h), 2), "diff": s((n - 2) * s2 - e ** 2 / (1 - h), 2), "n3": n - 3, "n2": n - 2,
        "s2i": s(s2i, 5), "si": s(sqrt(s2i), 4), "ti": s(ti, 2), "ti4": s(ti, 4), "den": s(sqrt(s2i) * sqrt(1 - h), 5),
        "t9": 2.262, "c_raw": s(e / sd, 2), "c_int": s(ri, 2), "c_noh": s(t_noh, 2), "c_n2": s(t_w, 2),
        "alt": s(ri * sqrt((n - 3) / (n - 2 - ri ** 2)), 4), "ri2": s(ri ** 2, 2),
    }


def p07():
    # (a) スピアマンの順位相関
    ra = [1, 2, 3, 4, 5, 6, 7, 8]
    rb = [2, 1, 4, 3, 6, 8, 5, 7]
    n = len(ra)
    d2 = sum((a - b) ** 2 for a, b in zip(ra, rb))
    rs = 1 - 6 * d2 / (n * (n * n - 1))
    # (b) 比例オッズモデル
    th1, th2, beta = 1.2, 2.8, 1.5
    ex = {0.3: 1.3499, -1.3: 0.2725, -1.2: 0.3012, -2.8: 0.0608}
    P1 = 1 / (1 + ex[0.3])     # x=1: logit = 1.2-1.5 = -0.3 -> P = 1/(1+e^{0.3})
    P2 = 1 / (1 + ex[-1.3])    # logit = 1.3 -> P = 1/(1+e^{-1.3})
    P1x0 = 1 / (1 + ex[-1.2])
    P2x0 = 1 / (1 + ex[-2.8])
    out = {
        "n": n, "d2": d2, "nn": n * (n * n - 1), "rs": r(rs, 3), "rs4": s(rs, 4), "six": 6 * d2,
        "w_n2": s(1 - 6 * d2 / (n * n * n), 3), "n3": n ** 3,
        "dlist": ",\\ ".join(str(a - b) for a, b in zip(ra, rb)),
        "d2list": "+".join(str((a - b) ** 2) for a, b in zip(ra, rb)),
        "th1": th1, "th2": th2, "beta": beta, "l1": s(th1 - beta, 1), "l2": s(th2 - beta, 1),
        "e03": s(ex[0.3], 4), "em13": s(ex[-1.3], 4), "em12": s(ex[-1.2], 4), "em28": s(ex[-2.8], 4),
        "P1": s(P1, 4), "P2": s(P2, 4), "ans": s(P2 - P1, 3), "ans4": s(P2 - P1, 4),
        "c_cum": s(P2, 3), "c_p1": s(P1, 3), "c_p3": s(1 - P2, 3), "c_x0": s(P2x0 - P1x0, 3),
        "P1x0": s(P1x0, 4), "P2x0": s(P2x0, 4),
    }
    for i in range(n):
        out[f"a{i + 1}"] = ra[i]
        out[f"b{i + 1}"] = rb[i]
    return out


def p08():
    # 吸収までの期待年数
    a11, a12, a13 = 0.85, 0.10, 0.05
    a21, a22, a23 = 0.20, 0.60, 0.20
    # m1 = 1 + a11 m1 + a12 m2, m2 = 1 + a21 m1 + a22 m2
    # m2 = (1 + a21 m1)/(1-a22)
    c0 = 1 / (1 - a22)
    c1 = a21 / (1 - a22)
    m1 = (1 + a12 * c0) / (1 - a11 - a12 * c1)
    m2 = c0 + c1 * m1
    # 誤り：要支援から自立へ戻らないとした
    m2_w = 1 / (1 - a22)
    m1_w = (1 + a12 * m2_w) / (1 - a11)
    return {
        "a11": s(a11, 2), "a12": s(a12, 2), "a13": s(a13, 2), "a21": s(a21, 2), "a22": s(a22, 2), "a23": s(a23, 2),
        "omb": s(1 - a22, 1), "c0": s(c0, 1), "c1": s(c1, 1), "a12c0": s(a12 * c0, 2), "a12c1": s(a12 * c1, 2),
        "lhs": s(1 - a11 - a12 * c1, 2), "rhs": s(1 + a12 * c0, 2),
        "m1": r(m1, 1), "m1s": s(m1, 2), "m2": s(m2, 2), "w_m1": s(m1_w, 2), "w_once": s(1 / a13, 1),
    }


def p09():
    px1, px0 = 0.70, 0.30
    py1, py0 = 0.12, 0.18
    iv = (py1 - py0) / (px1 - px0)
    return {
        "px1": s(px1, 2), "px0": s(px0, 2), "py1": s(py1, 2), "py0": s(py0, 2),
        "dy": s(py1 - py0, 2), "dx": s(px1 - px0, 2), "iv": s(iv, 3),
        "c_itt": s(py1 - py0, 3), "c_mul": s((py1 - py0) * (px1 - px0), 3), "c_sign": s(-iv, 3),
        "c_px1": s((py1 - py0) / px1, 3),
    }


def p10():
    m = {"A": 2.10, "B": 2.85, "C": 3.30}
    n, mse, k = 6, 0.36, 3
    dfe = k * (n - 1)
    q, t = 3.67, 2.131
    hsd = q * sqrt(mse / n)
    lsd = t * sqrt(2 * mse / n)
    hsd2 = q * sqrt(2 * mse / n)
    hsd_nosqrt = q * mse / n
    hsd_N = q * sqrt(mse / (k * n))
    d = {"BA": m["B"] - m["A"], "CA": m["C"] - m["A"], "CB": m["C"] - m["B"]}

    def sig(c):
        return sorted(key for key, v in d.items() if v > c)
    assert sig(hsd) == ["CA"]
    assert sig(hsd2) == []
    assert sig(lsd) == ["BA", "CA"]
    assert sig(hsd_nosqrt) == ["BA", "CA", "CB"]
    assert sig(hsd_N) == ["BA", "CA"]
    return {
        "mA": s(m["A"], 2), "mB": s(m["B"], 2), "mC": s(m["C"], 2), "n": n, "mse": s(mse, 2), "dfe": dfe,
        "q": s(q, 2), "t": s(t, 3), "msen": s(mse / n, 2), "sqmsen": s(sqrt(mse / n), 5),
        "hsd": s(hsd, 2), "hsd4": s(hsd, 4), "lsd": s(lsd, 2), "hsd2": s(hsd2, 2), "hsdns": s(hsd_nosqrt, 2),
        "hsdN": s(hsd_N, 2), "dBA": s(d["BA"], 2), "dCA": s(d["CA"], 2), "dCB": s(d["CB"], 2), "kn": k * n,
    }


def p11():
    # (a) 比例配分の層別抽出
    Nh = [600, 300, 100]
    Sh = [8, 12, 20]
    mh = [70, 60, 50]
    n = 100
    N = sum(Nh)
    W = [a / N for a in Nh]
    nh = [n * w for w in W]
    v_st = sum(w * w * sv * sv / m for w, sv, m in zip(W, Sh, nh))
    mu = sum(w * m for w, m in zip(W, mh))
    within = sum(w * sv * sv for w, sv in zip(W, Sh))
    between = sum(w * (m - mu) ** 2 for w, m in zip(W, mh))
    v_srs = (within + between) / n
    v_w2 = sum(w * w * sv * sv for w, sv in zip(W, Sh)) / n
    se_avgsd = sum(w * sv for w, sv in zip(W, Sh)) / sqrt(n)
    # (b) ルービンのルール
    Q = [64.2, 65.0, 63.6, 64.8, 64.4]
    U = [1.30, 1.25, 1.35, 1.28, 1.32]
    m = len(Q)
    qb = sum(Q) / m
    Wv = sum(U) / m
    B = sum((x - qb) ** 2 for x in Q) / (m - 1)
    T = Wv + (1 + 1 / m) * B
    out = {
        "N1": Nh[0], "N2": Nh[1], "N3": Nh[2], "N": N, "S1": Sh[0], "S2": Sh[1], "S3": Sh[2],
        "m1": mh[0], "m2": mh[1], "m3": mh[2], "n": n, "n1": int(nh[0]), "n2": int(nh[1]), "n3": int(nh[2]),
        "W1": s(W[0], 1), "W2": s(W[1], 1), "W3": s(W[2], 1),
        "t1": s(W[0] * Sh[0] ** 2, 1), "t2": s(W[1] * Sh[1] ** 2, 1), "t3": s(W[2] * Sh[2] ** 2, 1),
        "within": s(within, 1), "vst": s(v_st, 3), "se": s(sqrt(v_st), 2), "se4": s(sqrt(v_st), 4),
        "c_var": s(v_st, 2), "c_srs": s(sqrt(v_srs), 2), "c_w2": s(sqrt(v_w2), 2), "c_avg": s(se_avgsd, 2),
        "mu": s(mu, 0), "between": s(between, 1), "vsrs": s(v_srs, 3),
        "m": m, "qb": s(qb, 2), "Wv": s(Wv, 3), "B": s(B, 3), "T": s(T, 3), "ans": r(sqrt(T), 2),
        "ans4": s(sqrt(T), 4), "ssq": s(sum((x - qb) ** 2 for x in Q), 2), "Bfac": s((1 + 1 / m) * B, 3),
        "w_noadj": s(sqrt(Wv + B), 2), "w_W": s(sqrt(Wv), 2), "w_T": s(T, 2),
        "sumU": s(sum(U), 2), "devs": ",\\ ".join(s(x - qb, 1) for x in Q), "sumQ": s(sum(Q), 1),
    }
    for i in range(m):
        out[f"Q{i + 1}"] = s(Q[i], 1)
        out[f"U{i + 1}"] = s(U[i], 2)
    return out


def p12():
    L = [(0.80, 0.20), (0.75, 0.10), (0.70, 0.30), (0.20, 0.85), (0.10, 0.65)]
    p = len(L)
    ss1 = sum(a * a for a, _ in L)
    ss2 = sum(b * b for _, b in L)
    # (b) RMSEA
    chi2, df, N = 13.6, 4, 301
    rmsea = sqrt(max(chi2 - df, 0) / (df * (N - 1)))
    out = {
        "ss1": s(ss1, 4), "ss2": s(ss2, 4), "p": p, "ans": s(ss2 / p, 3), "c_f1": s(ss1 / p, 3),
        "c_share": s(ss2 / (ss1 + ss2), 3), "c_sum": s(sum(b for _, b in L) / p, 3), "c_ss": s(ss2, 3),
        "comm": s(ss1 + ss2, 4),
        "chi2": chi2, "df": df, "N": N, "Nm1": N - 1, "diff": s(chi2 - df, 1), "dN": df * (N - 1),
        "inner": s((chi2 - df) / (df * (N - 1)), 3), "rmsea": s(rmsea, 3), "rmsea4": s(rmsea, 4),
        "c_nosub": s(sqrt(chi2 / (df * (N - 1))), 3), "c_nosqrt": s((chi2 - df) / (df * (N - 1)), 3),
        "c_nodf": s(sqrt((chi2 - df) / (N - 1)), 3), "c_ratio": s(chi2 / df, 3),
    }
    for i, (a, b) in enumerate(L):
        out[f"l{i + 1}a"] = s(a, 2)
        out[f"l{i + 1}b"] = s(b, 2)
        out[f"sq{i + 1}"] = s(b * b, 4)
    return out


def p13():
    P, Q, R = (130, 6.0), (140, 6.2), (128, 7.6)
    sd = (15, 1.0)

    def dist(u, v, sc=(1, 1), sq=False):
        d2 = sum(((a - b) / c) ** 2 for a, b, c in zip(u, v, sc))
        return d2 if sq else sqrt(d2)
    raw = {"PQ": dist(P, Q), "PR": dist(P, R), "QR": dist(Q, R)}
    std = {"PQ": dist(P, Q, sd), "PR": dist(P, R, sd), "QR": dist(Q, R, sd)}
    var = {"PQ": dist(P, Q, (225, 1.0)), "PR": dist(P, R, (225, 1.0)), "QR": dist(Q, R, (225, 1.0))}
    assert min(raw, key=raw.get) == "PR" and min(std, key=std.get) == "PQ" and min(var, key=var.get) == "PQ"
    return {
        "Pa": P[0], "Pb": P[1], "Qa": Q[0], "Qb": Q[1], "Ra": R[0], "Rb": R[1], "sda": sd[0], "sdb": s(sd[1], 1),
        "zPQa": s((Q[0] - P[0]) / sd[0], 4), "zPRa": s((R[0] - P[0]) / sd[0], 4), "zQRa": s((R[0] - Q[0]) / sd[0], 4),
        "sPQ": s(std["PQ"], 2), "sPQ3": s(std["PQ"], 3), "sPR3": s(std["PR"], 3), "sQR3": s(std["QR"], 3),
        "qPQ": s(((Q[0] - P[0]) / sd[0]) ** 2, 4), "qPR": s(((R[0] - P[0]) / sd[0]) ** 2, 4),
        "qQR": s(((R[0] - Q[0]) / sd[0]) ** 2, 4), "hPQ": s((Q[1] - P[1]) ** 2, 2), "hPR": s((R[1] - P[1]) ** 2, 2),
        "hQR": s((R[1] - Q[1]) ** 2, 2), "sPR": s(std["PR"], 2), "sQR": s(std["QR"], 2),
        "s2PQ": s(std["PQ"] ** 2, 4), "s2PR": s(std["PR"] ** 2, 4), "s2QR": s(std["QR"] ** 2, 4),
        "rPQ": s(raw["PQ"], 2), "c_man": s(abs(Q[0] - P[0]) / sd[0] + abs(Q[1] - P[1]) / sd[1], 2),
        "dPQa": Q[0] - P[0], "dPRa": R[0] - P[0], "dQRa": R[0] - Q[0],
        "dPQb": s(Q[1] - P[1], 1), "dPRb": s(R[1] - P[1], 1), "dQRb": s(R[1] - Q[1], 1), "rPR": s(raw["PR"], 2), "rQR": s(raw["QR"], 2),
        "vPQ": s(var["PQ"], 2), "c_sq": s(std["PQ"] ** 2, 2),
    }


def p14():
    # (a) マンテル・ヘンツェル
    t1 = (10, 40, 20, 30)
    t2 = (6, 54, 9, 51)
    num = den = 0.0
    parts = {}
    for i, (a, b, c, d) in enumerate((t1, t2), 1):
        n = a + b + c + d
        parts[f"n{i}"] = n
        parts[f"ad{i}"] = s(a * d / n, 2)
        parts[f"bc{i}"] = s(b * c / n, 2)
        parts[f"or{i}"] = s(a * d / (b * c), 3)
        num += a * d / n
        den += b * c / n
    mh = num / den
    A, B, C, D = (t1[i] + t2[i] for i in range(4))
    crude = A * D / (B * C)
    # (b) ベータ・二項の予測
    a0, b0, nn, x = 8, 2, 20, 17
    pred = (a0 + x) * (a0 + x + 1) / ((a0 + b0 + nn) * (a0 + b0 + nn + 1))
    out = {
        "a1": t1[0], "b1": t1[1], "c1": t1[2], "d1": t1[3], "a2": t2[0], "b2": t2[1], "c2": t2[2], "d2": t2[3],
        "num": s(num, 2), "den": s(den, 2), "mh": r(mh, 2), "mh4": s(mh, 4), "crude": s(crude, 3),
        "A": A, "B": B, "C": C, "D": D,
        "a0": a0, "b0": b0, "nn": nn, "x": x, "ax": a0 + x, "bx": b0 + nn - x, "abn": a0 + b0 + nn,
        "ax1": a0 + x + 1, "abn1": a0 + b0 + nn + 1, "pnum": (a0 + x) * (a0 + x + 1),
        "pden": (a0 + b0 + nn) * (a0 + b0 + nn + 1), "pred": r(pred, 3), "pred5": s(pred, 5),
        "pmean": s((a0 + x) / (a0 + b0 + nn), 4), "w_sq": s(((a0 + x) / (a0 + b0 + nn)) ** 2, 4),
        "w_mle": s((x / nn) ** 2, 4), "w_prior": s((a0 / (a0 + b0)) ** 2, 4),
    }
    out.update(parts)
    return out


def values():
    return {"s15p01": p01(), "s15p02": p02(), "s15p03": p03(), "s15p04": p04(), "s15p05": p05(),
            "s15p06": p06(), "s15p07": p07(), "s15p08": p08(), "s15p09": p09(), "s15p10": p10(),
            "s15p11": p11(), "s15p12": p12(), "s15p13": p13(), "s15p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
