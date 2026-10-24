"""模擬試験 第9回（set09）の数値。values() が {大問id: {名前: 値}} を返す。

題材：農業・環境・生態・気象・スポーツ。
問題文に示した関数値（e^x、Φ、χ² の上側確率、べき乗、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # 鳴き声の自動識別：検証用データの混同行列（行＝真の種、列＝判定）
    cm = {"A": (84, 12, 4), "B": (18, 165, 17), "C": (9, 21, 270)}
    nA, nB, nC = (sum(cm[k]) for k in "ABC")
    tp = cm["A"][0]
    predA = cm["A"][0] + cm["B"][0] + cm["C"][0]
    prec = tp / predA
    rec = tp / nA
    f1 = 2 * prec * rec / (prec + rec)
    acc = (cm["A"][0] + cm["B"][1] + cm["C"][2]) / (nA + nB + nC)
    # (b) 野外での種の割合が違うときの事後確率
    piA, piB, piC = 0.05, 0.35, 0.60
    lA, lB, lC = tp / nA, cm["B"][0] / nB, cm["C"][0] / nC
    num = piA * lA
    den = num + piB * lB + piC * lC
    post = num / den
    eq = lA / (lA + lB + lC)
    return {
        "a1": cm["A"][0], "a2": cm["A"][1], "a3": cm["A"][2],
        "b1": cm["B"][0], "b2": cm["B"][1], "b3": cm["B"][2],
        "c1": cm["C"][0], "c2": cm["C"][1], "c3": cm["C"][2],
        "nA": nA, "nB": nB, "nC": nC, "tp": tp, "predA": predA, "fp": predA - tp, "fn": nA - tp,
        "prec": s(prec, 4), "rec": s(rec, 2), "f1": r(f1, 3), "f1_4": s(f1, 4),
        "f1num": 2 * tp, "f1den": predA + nA, "acc": s(acc, 4), "amean": s((prec + rec) / 2, 4),
        "piA": piA, "piB": piB, "piC": piC,
        "lA": s(lA, 2), "lB": s(lB, 2), "lC": s(lC, 2),
        "jA": s(piA * lA, 3), "jB": s(piB * lB, 4), "jC": s(piC * lC, 3), "den": s(den, 4),
        "post": s(post, 3), "post4": s(post, 4), "w_prec": s(prec, 3), "w_rec": s(lA, 3), "w_joint": s(num, 3),
        "w_eq": s(eq, 3),
    }


def p02():
    # サッカーの勝ち点：1試合 勝 3・分 1・負 0、試合は独立
    n, pw, pd = 38, 0.45, 0.25
    pl = round(1 - pw - pd, 2)
    vw = n * pw * (1 - pw)
    vd = n * pd * (1 - pd)
    cov = -n * pw * pd
    var = 9 * vw + vd + 2 * 3 * cov
    mean = n * (3 * pw + pd)
    # (b) チェビシェフと正規近似
    k = 16
    sd = sqrt(var)
    cheb = var / k ** 2
    z = r(k / sd, 2)
    phi = {1.97: 0.9756}
    assert z in phi, z
    norm2 = 2 * (1 - phi[z])
    return {
        "n": n, "pw": pw, "pd": pd, "pl": pl,
        "vw": s(vw, 3), "vd": s(vd, 3), "cov": s(cov, 3), "v9": s(9 * vw, 4), "c6": s(6 * cov, 3),
        "var": r(var, 2), "var_s": s(var, 2), "w_nocov": s(9 * vw + vd, 2),
        "mean": s(mean, 1), "per": s(3 * pw + pd, 2), "ex2": s(9 * pw + pd, 2), "pervar": s(9 * pw + pd - (3 * pw + pd) ** 2, 2),
        "k": k, "sd": s(sd, 3), "cheb": s(cheb, 3), "z": s(z, 2), "phi": s(phi[z], 4),
        "norm2": s(norm2, 3), "norm1": s(1 - phi[z], 3), "w_cheb_sd": s(sd / k, 3), "w_cheb_low": s(1 - cheb, 3),
        "inner": s(1 - norm2, 3), "k2": k ** 2,
    }


def p03():
    # (a) ハザード h(t)=a+bt の苗木の枯死
    a, b = 0.02, 0.01
    t1, t2 = 4, 8
    H = lambda t: a * t + b * t ** 2 / 2
    Hw = lambda t: a * t + b * t ** 2      # 積分の 1/2 を落とした
    ex = {0.24: 0.7866, 0.32: 0.7261, 0.48: 0.6188, 0.56: 0.5712}
    dH = r(H(t2) - H(t1), 2)
    dHw = r(Hw(t2) - Hw(t1), 2)
    hconst = r((a + b * t1) * (t2 - t1), 2)
    H8 = r(H(t2), 2)
    for x in (dH, dHw, hconst, H8):
        assert x in ex, x
    # (b) カプラン・マイヤー
    times = [(1, 1), (2, 1), (2, 1), (3, 0), (4, 1), (5, 0), (5, 1), (6, 0), (7, 1), (8, 0)]
    km = 1.0
    steps = []
    for t in sorted({t for t, d in times if d == 1}):
        if t > 5:
            break
        nrisk = sum(1 for u, _ in times if u >= t)
        d = sum(1 for u, dd in times if u == t and dd == 1)
        km *= (nrisk - d) / nrisk
        steps.append((t, nrisk, d))
    # 打ち切りを除いた誤り：打ち切り 3 本を除いた 7 本のうち 5 年を超えて生きたもの
    unc = [(t, d) for t, d in times if d == 1]
    naive_drop = sum(1 for t, d in unc if t > 5) / len(unc)
    naive_all = sum(1 for t, d in times if t > 5) / len(times)
    return {
        "a": a, "b": b, "t1": t1, "t2": t2, "hb2": s(b / 2, 3),
        "H1": s(H(t1), 2), "H2": s(H(t2), 2), "dH": s(dH, 2),
        "e24": s(ex[0.24], 4), "e32": s(ex[0.32], 4), "e48": s(ex[0.48], 4), "e56": s(ex[0.56], 4),
        "ans": s(ex[dH], 4), "w_unc": s(ex[H8], 4), "w_half": s(ex[dHw], 4), "w_const": s(ex[hconst], 4),
        "w_die": s(1 - ex[dH], 4), "h1": s(a + b * t1, 2),
        "Hw1": s(Hw(t1), 2), "Hw2": s(Hw(t2), 2),
        "km": r(km, 3), "km4": s(km, 4),
        "s1": s(steps[0][2] and (steps[0][1] - steps[0][2]) / steps[0][1], 4),
        "w_drop": s(naive_drop, 3), "w_all": s(naive_all, 3),
    }


def p04():
    # (a) 風速：東西・南北成分が独立に N(0, σ^2)
    sig, w = 4, 10
    thr = w ** 2 / sig ** 2
    tab1 = {2.5: 0.1138, 3.125: 0.0771, 6.25: 0.0124}
    tab2 = {2.5: 0.2865, 3.125: 0.2096, 6.25: 0.0439}
    ans = tab2[thr]
    # (b) 主成分：別の地点の分散共分散行列
    s11, s22, s12 = 9, 3, 4
    tr = s11 + s22
    det = s11 * s22 - s12 ** 2
    disc = sqrt(tr ** 2 / 4 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    rr = s12 / sqrt(s11 * s22)
    return {
        "sig": sig, "w": w, "thr": s(thr, 2), "sig2": sig ** 2, "w2": w ** 2,
        "t1a": s(tab1[2.5], 4), "t1b": s(tab1[3.125], 4), "t1c": s(tab1[6.25], 4),
        "t2a": s(tab2[2.5], 4), "t2b": s(tab2[3.125], 4), "t2c": s(tab2[6.25], 4),
        "ans": s(ans, 4), "w_df1": s(tab1[thr], 4), "w_nosq": s(tab2[w / sig], 4), "w_comp": s(1 - ans, 4),
        "w_half": s(tab2[w ** 2 / (2 * sig ** 2)], 4), "thr2": s(w ** 2 / (2 * sig ** 2), 3),
        "s11": s11, "s22": s22, "s12": s12, "tr": tr, "det": det, "half": s(tr / 2, 0), "disc2": s(tr ** 2 / 4 - det, 0),
        "disc": s(disc, 0), "l1": s(l1, 0), "l2": s(l2, 0),
        "cr": s(l1 / tr, 3), "w_var": s(s11 / tr, 3), "w_corr": s((1 + rr) / 2, 3), "rr": s(rr, 4),
        "w_sqrt": s(sqrt(l1) / (sqrt(l1) + sqrt(l2)), 3), "w_pc2": s(l2 / tr, 3),
    }


def p05():
    # (a) 標識再捕獲
    M, n, m = 64, 45, 11
    lp = M * n / m
    nhat = math.floor(lp)
    # (b) オフセットつきポアソン回帰
    b0, b1, b2, t = 0.40, 0.65, 0.30, 2.5
    ex = {0.40: 1.4918, 0.65: 1.9155, 0.95: 2.5857, 1.05: 2.8577, 1.35: 3.8574, 3.85: 46.993}
    lin = round(b0 + b1 + b2, 2)
    mu = t * ex[lin]
    return {
        "M": M, "n": n, "m": m, "lp": s(lp, 2), "nhat": nhat, "Mn": M * n, "w_ceil": nhat + 1,
        "lower": M + n - m, "Mm": M - m,
        "b0": s(b0, 2), "b1": s(b1, 2), "b2": s(b2, 2), "t": t, "lin": s(lin, 2),
        "e040": s(ex[0.40], 4), "e065": s(ex[0.65], 4), "e095": s(ex[0.95], 4), "e105": s(ex[1.05], 4),
        "e135": s(ex[1.35], 4), "e385": s(ex[3.85], 3),
        "mu": s(mu, 2), "w_nooff": s(ex[lin], 2), "w_lin": s(t * lin, 2), "w_t": s(ex[3.85], 2),
        "w_nomorn": s(t * ex[1.05], 2), "lin_t": s(lin + t, 2),
    }


def p06():
    # (a) 発芽率の正確な二項検定
    n, p0, p1 = 12, 0.6, 0.9
    pw6 = {10: 0.0060466, 11: 0.0036280, 12: 0.0021768}
    pw9 = {10: 0.3486784, 11: 0.3138106, 12: 0.2824295}
    q0, q1 = 1 - p0, 1 - p1
    P11_0 = n * pw6[11] * q0 + pw6[12]
    P10_0 = P11_0 + 66 * pw6[10] * q0 ** 2
    P11_1 = n * pw9[11] * q1 + pw9[12]
    P10_1 = P11_1 + 66 * pw9[10] * q1 ** 2
    assert P11_0 <= 0.05 < P10_0
    # (b) 病害ゼロ件の片側上側信頼限界
    ln05, ln025, ln99 = -2.9957, -3.6889, -0.010050
    nraw = ln05 / ln99
    nreq = math.ceil(nraw)
    n2 = math.ceil(ln025 / ln99)
    return {
        "n": n, "p0": p0, "p1": p1,
        "a10": s(pw6[10], 7), "a11": s(pw6[11], 7), "a12": s(pw6[12], 7),
        "c10": s(pw9[10], 7), "c11": s(pw9[11], 7), "c12": s(pw9[12], 7),
        "P12_0": s(pw6[12], 4), "P11only0": s(n * pw6[11] * q0, 4), "P11_0": s(P11_0, 3), "P11_0f": s(P11_0, 4),
        "P10_0": s(P10_0, 3), "P10_0f": s(P10_0, 4), "P10only0": s(66 * pw6[10] * q0 ** 2, 4),
        "P11_1": s(P11_1, 3), "P11_1f": s(P11_1, 4), "P10_1": s(P10_1, 3),
        "P11only1": s(n * pw9[11] * q1, 4), "P12_1": s(pw9[12], 4),
        "beta": s(1 - P11_1, 3), "P12_0s": s(pw6[12], 3), "P12_1s": s(pw9[12], 3),
        "ln05": ln05, "ln025": ln025, "ln99": s(ln99, 6), "nraw": s(nraw, 2), "nreq": nreq,
        "n2": n2, "n2raw": s(ln025 / ln99, 2), "n3": 300,
    }


def p07():
    # (a) フィッシャーの z 変換
    n, rr, rho0 = 52, 0.78, 0.60
    z_r, z_0 = 1.0454, 0.6931
    diff = z_r - z_0
    z = diff * sqrt(n - 3)
    z_n = diff * sqrt(n)
    z_raw = (rr - rho0) * sqrt(n - 3)
    t0 = rr * sqrt(n - 2) / sqrt(1 - rr ** 2)
    # (b) 落雷回数の比較
    xa, ta, xb, tb = 18, 3, 42, 4
    tot = xa + xb
    pa = ta / (ta + tb)
    ea = tot * pa
    va = tot * pa * (1 - pa)
    zl = (xa - ea) / sqrt(va)
    phi201 = 0.9778
    zh = (xa - tot / 2) / sqrt(tot / 4)
    zp = (xa - ea) / sqrt(ea)
    zv = (xa - ea) / va
    assert r(abs(zl), 2) == 2.01
    # (c) 古典的多次元尺度法
    P = [(0, 0), (6, 0), (1, 4), (5, 5)]
    D2 = [[(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for b in P] for a in P]
    rows = [sum(row) / 4 for row in D2]
    gm = sum(sum(row) for row in D2) / 16
    i, j = 0, 3
    b14 = -0.5 * (D2[i][j] - rows[i] - rows[j] + gm)
    return {
        "n": n, "r": rr, "rho0": rho0, "zr": z_r, "z0": s(z_0, 4), "diff": s(diff, 4), "n3": n - 3,
        "sq": s(sqrt(n - 3), 0), "z": s(z, 2), "z_n": s(z_n, 2), "z_raw": s(z_raw, 2), "t0": s(t0, 2),
        "xa": xa, "ta": ta, "xb": xb, "tb": tb, "tot": tot, "pa": "3/7", "ea": s(ea, 3), "va": s(va, 3),
        "sda": s(sqrt(va), 3), "zl": s(zl, 2), "phi": s(phi201, 4), "pv": s(2 * (1 - phi201), 3),
        "pv1": s(1 - phi201, 3), "zh": s(zh, 2), "zp": s(zp, 2), "zv": s(zv, 2),
        "d12": D2[0][1], "d13": D2[0][2], "d14": D2[0][3], "d23": D2[1][2], "d24": D2[1][3], "d34": D2[2][3],
        "rs1": sum(D2[0]), "rs4": sum(D2[3]), "rm1": s(rows[0], 2), "rm4": s(rows[3], 2),
        "tot2": sum(sum(row) for row in D2), "gm": s(gm, 3),
        "inner": s(D2[i][j] - rows[i] - rows[j] + gm, 3),
        "b14": s(b14, 2), "b14f": s(b14, 4), "w_nocent": s(-0.5 * D2[i][j], 2), "w_sign": s(-b14, 2),
        "w_nohalf": s(-(D2[i][j] - rows[i] - rows[j] + gm), 2),
        "w_gm": s(-0.5 * (D2[i][j] - rows[i] - rows[j] - gm), 2),
    }


def p08():
    p = 0.6
    q = 1 - p
    D = p ** 2 / (p ** 2 + q ** 2)
    P3040 = p * D
    P1540 = p * P3040
    ans = p * D + q * P1540
    # デュースを無視：先に 4 ポイント（あと 3 点）を取れば勝ち、相手はあと 2 点で勝ち
    nodeuce = p ** 3 + 3 * p ** 3 * q
    Dp = p
    alt = p * Dp + q * p * p * Dp
    return {
        "p": p, "q": s(q, 1), "p2": s(p ** 2, 2), "q2": s(q ** 2, 2), "den": s(p ** 2 + q ** 2, 2),
        "D": s(D, 4), "P3040": s(P3040, 4), "P1540": s(P1540, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "t1": s(p * D, 4), "t2": s(q * P1540, 4),
        "w_D": s(D, 3), "w_nod": s(nodeuce, 3), "w_alt": s(alt, 3), "w_opp": s(1 - ans, 3),
    }


def p09():
    d = (20, 30, 10)
    v = (20, 40, 50)
    T = sum(di / vi for di, vi in zip(d, v))
    hm = sum(d) / T
    return {
        "d1": d[0], "d2": d[1], "d3": d[2], "v1": v[0], "v2": v[1], "v3": v[2], "D": sum(d),
        "T1": s(d[0] / v[0], 2), "T2": s(d[1] / v[1], 2), "T3": s(d[2] / v[2], 2), "T": s(T, 2),
        "hm": s(hm, 2), "w_am": s(sum(v) / 3, 2), "w_wam": s(sum(di * vi for di, vi in zip(d, v)) / sum(d), 2),
        "w_hm": s(3 / sum(1 / vi for vi in v), 2), "w_gm": s((v[0] * v[1] * v[2]) ** (1 / 3), 2),
    }


def p10():
    # (a) 交互作用のある回帰
    b1, b3, se1, se3, c13 = 0.42, 0.18, 0.10, 0.14, -0.0100
    var = se1 ** 2 + se3 ** 2 + 2 * c13
    se = sqrt(var)
    t = (b1 + b3) / se
    t_nocov = (b1 + b3) / sqrt(se1 ** 2 + se3 ** 2)
    t_b3 = b3 / se3
    t_sum = (b1 + b3) / (se1 + se3)
    # (b) 省略変数バイアス
    g1, g2, cxs, vx, vs = 0.50, 0.30, -4.0, 5.0, 8.0
    dlt = cxs / vx
    short = g1 + g2 * dlt
    rev = g1 + g2 * cxs / vs
    return {
        "b1": s(b1, 2), "b3": s(b3, 2), "se1": s(se1, 2), "se3": s(se3, 2), "c13": s(c13, 4),
        "bs": s(b1 + b3, 2), "v1": s(se1 ** 2, 4), "v3": s(se3 ** 2, 4), "c2": s(2 * c13, 4),
        "var": s(var, 4), "se": s(se, 4), "t": r(t, 2), "t4": s(t, 4),
        "w_nocov": s(t_nocov, 2), "w_b3": s(t_b3, 2), "w_sum": s(t_sum, 2),
        "g1": s(g1, 2), "g2": s(g2, 2), "cxs": s(cxs, 1), "vx": s(vx, 1), "vs": s(vs, 1),
        "dlt": s(dlt, 2), "bias": s(g2 * dlt, 2), "short": s(short, 2), "w_rev": s(rev, 2),
        "w_none": s(g1, 2), "w_sign": s(g1 - g2 * dlt, 2), "w_bias": s(g2 * dlt, 2), "rdl": s(cxs / vs, 2),
    }


def p11():
    x = [2, 3, 4, 5, 6, 10]
    e = [0.4, -0.6, 0.5, -0.1, -0.375, 0.175]
    n = len(x)
    xb = sum(x) / n
    sxx = sum((xi - xb) ** 2 for xi in x)
    h = [1 / n + (xi - xb) ** 2 / sxx for xi in x]
    hr = h  # 計算は正確な値（問題文には小数第4位までを示す）
    cv = sum((ei / (1 - hi)) ** 2 for ei, hi in zip(e, hr)) / n
    mse = sum(ei ** 2 for ei in e) / n
    cv_plus = sum((ei / (1 + hi)) ** 2 for ei, hi in zip(e, hr)) / n
    cv_lin = sum((ei / (1 - hi)) for ei, hi in zip(e, hr)) ** 2 / n
    cv_noSq = sum(ei ** 2 / (1 - hi) for ei, hi in zip(e, hr)) / n
    d = {f"x{i+1}": xi for i, xi in enumerate(x)}
    d.update({f"e{i+1}": s(ei, 3) if abs(ei * 100 - round(ei * 100)) > 1e-9 else s(ei, 1 if abs(ei * 10 - round(ei * 10)) < 1e-9 else 2)
              for i, ei in enumerate(e)})
    d.update({f"h{i+1}": s(hi, 4) for i, hi in enumerate(hr)})
    d.update({f"u{i+1}": s(ei / (1 - hi), 4) for i, (ei, hi) in enumerate(zip(e, hr))})
    d.update({f"uu{i+1}": s((ei / (1 - hi)) ** 2, 4) for i, (ei, hi) in enumerate(zip(e, hr))})
    d.update({f"om{i+1}": s(1 - hi, 4) for i, hi in enumerate(hr)})
    d.update({
        "n": n, "xb": s(xb, 0), "sxx": s(sxx, 0), "dev6": s(x[-1] - xb, 0), "dev62": s((x[-1] - xb) ** 2, 0),
        "h6a": s(1 / n, 4), "h6b": s((x[-1] - xb) ** 2 / sxx, 4), "h6": s(h[-1], 3),
        "w_noinv": s((x[-1] - xb) ** 2 / sxx, 3), "w_nosq": s(1 / n + (x[-1] - xb) / sxx, 3),
        "w_only": s(1 / n, 3), "w_raw": s(1 / n + (x[-1] - xb) ** 2 / sum(xi ** 2 for xi in x), 3),
        "sx2": sum(xi ** 2 for xi in x), "thr": s(2 * 2 / n, 3),
        "sumu2": s(sum((ei / (1 - hi)) ** 2 for ei, hi in zip(e, hr)), 4),
        "cv": r(cv, 3), "cv4": s(cv, 4), "mse": s(mse, 3), "w_plus": s(cv_plus, 3), "w_nosqcv": s(cv_noSq, 3),
        "sse": s(sum(ei ** 2 for ei in e), 4),
    })
    return d


def p12():
    # (a) 面積比例確率抽出とホルビッツ・トンプソン推定量
    A = 50.0
    nn = 4
    area = [3.0, 1.5, 2.0, 4.0]
    y = [180, 95, 130, 250]
    pi = [nn * a / A for a in area]
    ht = sum(yi / p for yi, p in zip(y, pi))
    ratio = sum(y) / sum(area) * A
    Nn = 20
    srs = Nn * sum(y) / nn
    # (b) 変量効果の一元配置
    a, rep, msa, mse = 6, 5, 38.4, 6.0
    sa = (msa - mse) / rep
    icc = sa / (sa + mse)
    return {
        "A": s(A, 0), "nn": nn, "N": Nn,
        "ar1": s(area[0], 1), "ar2": s(area[1], 1), "ar3": s(area[2], 1), "ar4": s(area[3], 1),
        "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3],
        "pi1": s(pi[0], 2), "pi2": s(pi[1], 2), "pi3": s(pi[2], 2), "pi4": s(pi[3], 2),
        "q1": s(y[0] / pi[0], 2), "q2": s(y[1] / pi[1], 2), "q3": s(y[2] / pi[2], 2), "q4": s(y[3] / pi[3], 2),
        "ht": r(ht, 0), "ht2": s(ht, 2), "w_ratio": s(ratio, 1), "w_srs": s(srs, 0), "w_nopi": s(ht * nn, 0),
        "a": a, "rep": rep, "msa": s(msa, 1), "mse": s(mse, 1), "dfa": a - 1, "dfe": a * (rep - 1),
        "diff": s(msa - mse, 1), "sa": s(sa, 2), "icc": s(icc, 3), "tot": s(sa + mse, 2),
        "w1_sa": s(msa - mse, 1), "w1_icc": s((msa - mse) / (msa - mse + mse), 3),
        "w2_sa": s((msa - mse) / a, 2), "w2_icc": s((msa - mse) / a / ((msa - mse) / a + mse), 3),
        "w3_icc": s(sa / msa, 3),
        "w4_sa": s(msa / rep, 2), "w4_icc": s(msa / rep / (msa / rep + mse), 3),
    }


def p13():
    # テーピングと足首の捻挫（コホート研究）
    n1, x1, n0, x0 = 200, 14, 250, 35
    rr = (x1 / n1) / (x0 / n0)
    lnrr = -0.6931
    se = sqrt(1 / x1 - 1 / n1 + 1 / x0 - 1 / n0)
    se_w = sqrt(1 / x1 + 1 / x0)
    z, z1 = 1.96, 1.645
    expo = lambda v: math.exp(v)
    pts = {}

    def lim(c):
        lo, hi = r(lnrr - c, 4), r(lnrr + c, 4)
        pts[lo] = r(expo(lo), 4)
        pts[hi] = r(expo(hi), 4)
        return lo, hi

    ok = lim(r(z * r(se, 4), 4))
    nm = lim(r(z * r(se_w, 4), 4))
    on = lim(r(z1 * r(se, 4), 4))
    keys = sorted(pts)
    d = {
        "n1": n1, "x1": x1, "n0": n0, "x0": x0, "m1": n1 - x1, "m0": n0 - x0,
        "p1": s(x1 / n1, 2), "p0": s(x0 / n0, 2), "rr": s(rr, 2), "lnrr": s(lnrr, 4),
        "v": s(se ** 2, 4), "se": s(se, 4), "hw": s(z * r(se, 4), 4), "z": z,
        "lo": s(ok[0], 4), "hi": s(ok[1], 4), "elo": s(pts[ok[0]], 3), "ehi": s(pts[ok[1]], 3),
        "w1lo": s(pts[nm[0]], 3), "w1hi": s(pts[nm[1]], 3),
        "w2lo": s(rr - z * se, 3), "w2hi": s(rr + z * se, 3),
        "w3lo": s(pts[on[0]], 3), "w3hi": s(pts[on[1]], 3),
        "w4lo": s(ok[0], 3), "w4hi": s(ok[1], 3), "sew": s(se_w, 4),
        "inv1": s(1 / x1, 4), "inv2": s(1 / n1, 4), "inv3": s(1 / x0, 4), "inv4": s(1 / n0, 4),
    }
    for i, k in enumerate(keys):
        d[f"ex{i}"] = s(k, 4)
        d[f"ev{i}"] = s(pts[k], 4)
    return d


def p14():
    m0, tau, sig, n, xbar, c = 180, 6, 8, 4, 172, 170
    prec = 1 / tau ** 2 + n / sig ** 2
    w = (n / sig ** 2) / prec
    pm = m0 + w * (xbar - m0)
    pv = 1 / prec
    predv = pv + sig ** 2
    phi = {0.25: 0.5987, 0.51: 0.6950, 0.56: 0.7123, 1.00: 0.8413, 1.34: 0.9099}
    z_ok = r(abs(c - pm) / sqrt(predv), 2)
    z_post = r(abs(c - pm) / sqrt(pv), 2)
    z_plug = r(abs(c - pm) / sig, 2)
    z_prior = r(abs(c - m0) / sqrt(tau ** 2 + sig ** 2), 2)
    z_data = r(abs(c - xbar) / sig, 2)
    for zz in (z_ok, z_post, z_plug, z_prior, z_data):
        assert zz in phi, zz
    P = lambda zz: 1 - phi[zz]
    return {
        "m0": m0, "tau": tau, "tau2": tau ** 2, "sig": sig, "sig2": sig ** 2, "n": n, "xbar": xbar, "c": c,
        "pr0": s(1 / tau ** 2, 4), "pr1": s(n / sig ** 2, 4), "prec": s(prec, 4), "w": s(w, 4),
        "pm": s(pm, 2), "pv": s(pv, 2), "predv": s(predv, 2), "predsd": s(sqrt(predv), 3), "postsd": s(sqrt(pv), 3),
        "z": s(z_ok, 2), "zraw": s((c - pm) / sqrt(predv), 3),
        "phi025": s(phi[0.25], 4), "phi051": s(phi[0.51], 4), "phi056": s(phi[0.56], 4),
        "phi100": s(phi[1.00], 4), "phi134": s(phi[1.34], 4),
        "ans": s(P(z_ok), 3), "w_post": s(P(z_post), 3), "w_plug": s(P(z_plug), 3),
        "w_prior": s(P(z_prior), 3), "w_data": s(P(z_data), 3),
        "zpost": s(z_post, 2), "zplug": s(z_plug, 2), "zprior": s(z_prior, 2), "zdata": s(z_data, 2),
        "num": s(m0 / tau ** 2 + xbar * n / sig ** 2, 4),
    }


def values():
    return {f"s09p{i:02d}": f() for i, f in enumerate(
        [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14], start=1)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
