"""模擬試験 第18回（set18）の数値。values() が {大問id: {名前: 値}} を返す。

題材：生物学・遺伝・生態調査・水産。
問題文に示した分布の値（χ²・F・t・q の点、e^x・ln・cos など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # 共優性の遺伝子座：遺伝子型の数
    nAA, nAa, naa = 50, 80, 70
    n = nAA + nAa + naa
    cntA = 2 * nAA + nAa
    p = cntA / (2 * n)
    q = 1 - p
    se = sqrt(p * q / (2 * n))
    se_n = sqrt(p * q / n)
    # (b) ハーディ・ワインベルグ平衡の尤度比検定（G 統計量）
    eAA, eAa, eaa = n * p * p, n * 2 * p * q, n * q * q
    lnv = {"l1": 0.2107, "l2": -0.2131, "l3": 0.1459}  # ln(50/40.5), ln(80/99), ln(70/60.5)
    half = nAA * lnv["l1"] + nAa * lnv["l2"] + naa * lnv["l3"]
    G = 2 * half
    chi = {"c11": 3.84, "c12": 6.63, "c13": 7.88, "c21": 5.99, "c22": 9.21, "c23": 10.60}
    assert chi["c12"] < G < chi["c13"] and G < chi["c22"], G
    # メンデル比 1:2:1 を帰無仮説にした G（誤答）
    eM = [n / 4, n / 2, n / 4]
    G_m = 2 * sum(o * math.log(o / e) for o, e in zip((nAA, nAa, naa), eM))
    # (c) 優性の遺伝子座の EM アルゴリズム 1 回
    nD, nR = 160, 40
    nn = nD + nR
    p0 = 0.5
    q0 = 1 - p0
    den = p0 ** 2 + 2 * p0 * q0
    eBB = nD * p0 ** 2 / den
    eBb = nD * 2 * p0 * q0 / den
    p1 = (2 * eBB + eBb) / (2 * nn)
    q_mle = sqrt(nR / nn)
    return {
        "nAA": nAA, "nAa": nAa, "naa": naa, "n": n, "n2": 2 * n, "cntA": cntA,
        "p": s(p, 2), "q": s(q, 2), "pq": s(p * q, 4), "var": s(p * q / (2 * n), 8),
        "se": r(se, 4), "se_s": s(se, 4), "se6": s(se, 6), "se_n": s(se_n, 4),
        "eAA": s(eAA, 1), "eAa": s(eAa, 1), "eaa": s(eaa, 1), "p2": s(p * p, 4), "pq2": s(2 * p * q, 4), "q2": s(q * q, 4),
        "l1": s(lnv["l1"], 4), "l2": s(lnv["l2"], 4), "l3": s(lnv["l3"], 4),
        "t1": s(nAA * lnv["l1"], 3), "t2": s(nAa * lnv["l2"], 3), "t3": s(naa * lnv["l3"], 3),
        "half": s(half, 2), "G": s(G, 2), "Gm": s(G_m, 2),
        **{k: s(v, 2) for k, v in chi.items()},
        "nD": nD, "nR": nR, "nn": nn, "nn2": 2 * nn, "p0": s(p0, 1), "pp0": s(p0 ** 2, 2), "pq0": s(2 * p0 * q0, 2),
        "den0": s(den, 2), "eBB": s(eBB, 2), "eBb": s(eBb, 2), "eBB_f": "160/3", "eBb_f": "320/3",
        "p1": r(p1, 3), "p1_s": s(p1, 3), "p1_4": s(p1, 4), "two_eBB": s(2 * eBB, 2), "num1": s(2 * eBB + eBb, 2),
        "q_mle": s(q_mle, 4), "p_mle": s(1 - q_mle, 4), "p_mle3": s(1 - q_mle, 3),
        "p1_wrong": s((eBB + eBb) / nn, 3),
    }


def p02():
    lam, h = 4.5, 0.6
    mu = lam * h
    mu0 = lam * (1 - h)
    ex = {2.7: 0.0672, 4.5: 0.0111, 1.8: 0.1653}
    k = 2
    pk = ex[2.7] * mu ** k / math.factorial(k)
    p_n = ex[4.5] * lam ** k / 2
    p_1 = ex[2.7] * mu
    p_wrong = ex[1.8] * mu0 ** k / 2
    p_cum = ex[2.7] * (1 + mu + mu ** 2 / 2)
    cond = k + mu0
    return {
        "lam": lam, "h": h, "h0": s(1 - h, 1), "mu": s(mu, 1), "mu0": s(mu0, 1), "mu2": s(mu ** 2, 2),
        "e27": s(ex[2.7], 4), "e45": s(ex[4.5], 4), "e18": s(ex[1.8], 4), "k": k,
        "pk": s(pk, 3), "pk5": s(pk, 5), "p_n": s(p_n, 3), "p_1": s(p_1, 3), "p_w": s(p_wrong, 3), "p_cum": s(p_cum, 3),
        "cond": r(cond, 1), "cond_s": s(cond, 1), "c_div": s(k / h, 2), "c_add": s(k + lam, 1),
    }


def p03():
    # (a) 一様分布の体長から体重へ
    a, b, c = 20, 30, 100  # W = L^3 / c
    wa, wb = a ** 3 / c, b ** 3 / c
    # (b) 標本分散のカイ二乗分布
    n = 10
    chi = {"c95_9": 16.92, "c975_9": 19.02, "c95_10": 18.31, "c975_10": 20.48}
    k_ok = chi["c95_9"] / (n - 1)
    k_n = chi["c95_9"] / n
    k_10n = chi["c95_10"] / n
    k_10 = chi["c95_10"] / (n - 1)
    k_25 = chi["c975_9"] / (n - 1)
    ks = [k_ok, k_n, k_10n, k_10, k_25]
    assert len({s(x, 2) for x in ks}) == 5
    # (c) 等分散の F 検定
    n1, v1, n2, v2 = 10, 4.8, 13, 1.6
    F = v1 / v2
    f = {"f05": 2.80, "f025": 3.44, "f025r": 3.87}
    assert f["f05"] < F < f["f025"]
    return {
        "a": a, "b": b, "c": c, "wa": s(wa, 0), "wb": s(wb, 0), "wr": s(wb - wa, 0),
        "n": n, "n1df": n - 1,
        **{k: s(v, 2) for k, v in chi.items()},
        "k_ok": s(k_ok, 2), "k_ok4": s(k_ok, 4), "k_n": s(k_n, 2), "k_10n": s(k_10n, 2), "k_10": s(k_10, 2), "k_25": s(k_25, 2),
        "m1": n1, "v1": s(v1, 1), "m2": n2, "v2": s(v2, 1), "d1": n1 - 1, "d2": n2 - 1,
        "F": s(F, 2), "Fsd": s(sqrt(F), 2),
        **{k: s(v, 2) for k, v in f.items()},
    }


def p04():
    n, x = 150, 120
    p = x / n
    se_logit = 1 / sqrt(n * p * (1 - p))
    se_p = sqrt(p * (1 - p) / n)
    ln4 = 1.3863
    se4 = r(se_logit, 4)
    lo, hi = r(ln4 - 1.96 * se4, 3), r(ln4 + 1.96 * se4, 3)
    lo9, hi9 = r(ln4 - 1.645 * se4, 3), r(ln4 + 1.645 * se4, 3)
    lon, hin = r(ln4 - 1.96 * se_p, 3), r(ln4 + 1.96 * se_p, 3)
    em = {0.986: 0.3731, 1.051: 0.3496, 1.322: 0.2666, 1.450: 0.2346, 1.722: 0.1787, 1.786: 0.1676}
    for z in (lo, hi, lo9, hi9, lon, hin):
        assert z in em, z
    inv = lambda z: 1 / (1 + em[z])
    ci = lambda a, b: f"({s(a, 3)},\\ {s(b, 3)})"
    return {
        "n": n, "x": x, "p": s(p, 1), "npq": s(n * p * (1 - p), 0),
        "se": r(se_logit, 3), "se_s": s(se_logit, 3), "se4": s(se4, 4), "se_p": s(se_p, 4), "se_p3": s(se_p, 3),
        "se_wrong": s(1 / sqrt(n), 3),
        "ln4": s(ln4, 4), "hw": s(1.96 * se4, 4), "lo": s(lo, 3), "hi": s(hi, 3),
        "e1": s(em[0.986], 4), "e2": s(em[1.051], 4), "e3": s(em[1.322], 4), "e4": s(em[1.450], 4),
        "e5": s(em[1.722], 4), "e6": s(em[1.786], 4),
        "pl": s(inv(lo), 4), "pu": s(inv(hi), 4),
        "ci_ok": ci(inv(lo), inv(hi)), "ci_90": ci(inv(lo9), inv(hi9)), "ci_naive": ci(inv(lon), inv(hin)),
        "ci_wald": ci(p - 1.96 * se_p, p + 1.96 * se_p), "ci_flip": ci(1 - inv(hi), 1 - inv(lo)),
    }


def p05():
    # (a) 負の二項分布
    th, k, m = 0.2, 3, 5
    b = lambda nn, j: math.comb(nn, j) * th ** j * (1 - th) ** (nn - j)
    ok = sum(b(m, j) for j in range(k, m + 1))
    terms = [math.comb(y - 1, k - 1) * th ** k * (1 - th) ** (y - k) for y in range(k, m + 1)]
    assert abs(sum(terms) - ok) < 1e-12
    only5 = terms[-1]
    ge2 = sum(b(m, j) for j in range(2, m + 1))
    eq3 = b(m, 3)
    fail5 = sum(b(m + k, j) for j in range(k, m + k + 1))
    vals = [only5, eq3, ok, fail5, ge2]
    assert len({s(x, 3) for x in vals}) == 5
    # (b) 層別抽出
    N1, N2, n1, n2 = 200, 300, 20, 30
    y1, y2, v1, v2 = 12.0, 5.0, 16.0, 10.0
    N = N1 + N2
    W1, W2 = N1 / N, N2 / N
    t1 = W1 ** 2 * (1 - n1 / N1) * v1 / n1
    t2 = W2 ** 2 * (1 - n2 / N2) * v2 / n2
    var = t1 + t2
    var_nf = W1 ** 2 * v1 / n1 + W2 ** 2 * v2 / n2
    return {
        "th": th, "k": k, "m": m, "q": s(1 - th, 1),
        "b3": s(b(m, 3), 4), "b4": s(b(m, 4), 4), "b5": s(b(m, 5), 5),
        "f3": s(terms[0], 4), "f4": s(terms[1], 4), "f5": s(terms[2], 5),
        "ok": s(ok, 3), "ok5": s(ok, 5), "only5": s(only5, 3), "eq3": s(eq3, 3), "ge2": s(ge2, 3), "fail5": s(fail5, 3),
        "N1": N1, "N2": N2, "n1": n1, "n2": n2, "y1": s(y1, 1), "y2": s(y2, 1), "v1": s(v1, 0), "v2": s(v2, 0),
        "W1": s(W1, 1), "W2": s(W2, 1), "W1s": s(W1 ** 2, 2), "W2s": s(W2 ** 2, 2),
        "f1": s(1 - n1 / N1, 1), "f2": s(1 - n2 / N2, 1), "mean": s(W1 * y1 + W2 * y2, 1),
        "t1": s(t1, 4), "t2": s(t2, 4), "var": s(var, 4), "se": r(sqrt(var), 2), "se4": s(sqrt(var), 4),
        "se_nf": s(sqrt(var_nf), 2), "var_nf": s(var_nf, 3), "se_var": s(var, 2),
    }


def p06():
    n, b, se, b0 = 25, 3.24, 0.10, 3
    t = (b - b0) / se
    tc = {"t23": 2.069, "t23h": 2.807, "t24": 2.064}
    assert tc["t23"] < t < tc["t23h"]
    # (b) 外部スチューデント化残差
    pp, rr = 2, 2.8
    ext = rr * sqrt((n - pp - 1) / (n - pp - rr ** 2))
    ext_w = rr * sqrt((n - pp) / (n - pp - 1 - rr ** 2))
    return {
        "n": n, "b": s(b, 2), "se": s(se, 2), "b0": b0, "df": n - 2, "t": s(t, 2), "t0": s(b / se, 1),
        "t_nose": s(b - b0, 2), **{k: s(v, 3) for k, v in tc.items()},
        "pp": pp, "rr": s(rr, 1), "rr2": s(rr ** 2, 2), "np": n - pp, "np1": n - pp - 1,
        "den": s(n - pp - rr ** 2, 2), "ratio": s((n - pp - 1) / (n - pp - rr ** 2), 4),
        "sq": s(sqrt((n - pp - 1) / (n - pp - rr ** 2)), 4),
        "ext": r(ext, 2), "ext_s": s(ext, 2), "ext4": s(ext, 4), "ext_w": s(ext_w, 2),
    }


def p07():
    # (a) マンテル・ヘンツェル
    A = (30, 20, 20, 30)   # 大型 感染/非感染, 小型 感染/非感染
    B = (12, 18, 14, 36)
    def nums(t):
        a, b, c, d = t
        n = a + b + c + d
        return a * d / n, b * c / n, n
    adA, bcA, nA = nums(A)
    adB, bcB, nB = nums(B)
    mh = (adA + adB) / (bcA + bcB)
    orA = A[0] * A[3] / (A[1] * A[2])
    orB = B[0] * B[3] / (B[1] * B[2])
    P = tuple(x + y for x, y in zip(A, B))
    orP = P[0] * P[3] / (P[1] * P[2])
    nos = (A[0] * A[3] + B[0] * B[3]) / (A[1] * A[2] + B[1] * B[2])
    vals = [(orA + orB) / 2, mh, nos, orP, orA]
    assert len({s(x, 2) for x in vals}) == 5
    # (b) ロジスティック回帰
    b0, b1 = -3.6, 0.12
    e12, em12 = 3.320, 0.3012
    x50 = -b0 / b1
    p20 = em12 / (1 + em12)
    return {
        **{f"A{i}": v for i, v in enumerate(A)}, **{f"B{i}": v for i, v in enumerate(B)},
        "nA": nA, "nB": nB, "adA": s(adA, 2), "bcA": s(bcA, 2), "adB": s(adB, 2), "bcB": s(bcB, 2),
        "sad": s(adA + adB, 2), "sbc": s(bcA + bcB, 2), "mh": s(mh, 2), "mh4": s(mh, 4),
        "orA": s(orA, 2), "orB": s(orB, 3), "orM": s((orA + orB) / 2, 2), "orP": s(orP, 2), "nos": s(nos, 2),
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "b0": b0, "b1": b1, "mb0": s(-b0, 1), "e12": s(e12, 3), "em12": s(em12, 4), "x50": s(x50, 0), "b10": s(10 * b1, 1),
        "slope": s(b1 * 0.25, 2), "lp20": s(b0 + b1 * 20, 1), "p20": s(p20, 3),
    }


def p08():
    k, n = 3, 8
    means = {"A": 52.0, "B": 47.5, "C": 44.0}
    mse, q, t = 16.0, 3.56, 2.080
    dfe = k * n - k
    hsd = q * sqrt(mse / n)
    lsd = t * sqrt(2 * mse / n)
    w2 = q * sqrt(2 * mse / n)
    w3 = q * sqrt(mse)
    w4 = q * sqrt(mse / n) / sqrt(2)
    dAB, dAC, dBC = means["A"] - means["B"], means["A"] - means["C"], means["B"] - means["C"]
    assert dBC < dAB < hsd < dAC and dAB < w2 < dAC and dAC < w3 and lsd < dAB and dBC < lsd and w4 < dAB and dBC < w4
    return {
        "k": k, "n": n, "N": k * n, "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1),
        "mse": s(mse, 1), "dfe": dfe, "q": s(q, 2), "t": s(t, 3), "msen": s(mse / n, 1), "sq": s(sqrt(mse / n), 4),
        "hsd": s(hsd, 2), "lsd": s(lsd, 2), "w2": s(w2, 2), "w3": s(w3, 2), "w4": s(w4, 2),
        "dAB": s(dAB, 1), "dAC": s(dAC, 1), "dBC": s(dBC, 1),
    }


def p09():
    x1, x0, y1, y0 = 52.0, 40.0, 31.2, 28.2
    ols = 0.40
    iv = (y1 - y0) / (x1 - x0)
    vals = [iv, ols, y1 - y0, (x1 - x0) / (y1 - y0), (y1 - y0) / x1]
    assert len({s(v, 2) for v in vals}) == 5
    return {
        "x1": s(x1, 1), "x0": s(x0, 1), "y1": s(y1, 1), "y0": s(y0, 1), "ols": s(ols, 2),
        "dx": s(x1 - x0, 1), "dy": s(y1 - y0, 1), "iv": s(iv, 2), "inv": s((x1 - x0) / (y1 - y0), 2),
        "red": s(y1 - y0, 2), "per": s((y1 - y0) / x1, 2),
    }


def p10():
    temp = [14.2, 16.8, 18.1, 19.5, 21.0, 22.4, 23.9, 25.3]
    spp = [23, 15, 19, 31, 27, 42, 33, 38]
    n = len(temp)
    rk = lambda v: [sorted(v).index(x) + 1 for x in v]
    rt, rs = rk(temp), rk(spp)
    d2 = sum((a - b) ** 2 for a, b in zip(rt, rs))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sgn = (rt[i] - rt[j]) * (rs[i] - rs[j])
            conc += sgn > 0
            disc += sgn < 0
    tau = (conc - disc) / (n * (n - 1) / 2)
    vals = [rho, 1 - d2 / (n * (n * n - 1)), 1 - 6 * d2 / (n * n * (n - 1)), tau, 6 * d2 / (n * (n * n - 1))]
    assert len({s(v, 3) for v in vals}) == 5
    # (b) 群平均法
    D = {"AB": 2, "AC": 6, "AD": 10, "AE": 9, "BC": 5, "BD": 8, "BE": 7, "CD": 4, "CE": 6, "DE": 3}
    ab_c = (D["AC"] + D["BC"]) / 2
    ab_d = (D["AD"] + D["BD"]) / 2
    ab_e = (D["AE"] + D["BE"]) / 2
    ab_de = (D["AD"] + D["AE"] + D["BD"] + D["BE"]) / 4
    c_de = (D["CD"] + D["CE"]) / 2
    assert c_de < ab_c
    fin = (D["AC"] + D["BC"] + D["AD"] + D["AE"] + D["BD"] + D["BE"]) / 6
    wpgma = (ab_c + ab_de) / 2
    out = {
        **{f"t{i}": s(v, 1) for i, v in enumerate(temp)}, **{f"s{i}": v for i, v in enumerate(spp)},
        **{f"rt{i}": v for i, v in enumerate(rt)}, **{f"rs{i}": v for i, v in enumerate(rs)},
        **{f"d{i}": a - b for i, (a, b) in enumerate(zip(rt, rs))},
        "n": n, "sd2": d2, "d2list": "+".join(str((a - b) ** 2) for a, b in zip(rt, rs)), "den": n * (n * n - 1), "num": 6 * d2,
        "rho": s(rho, 3), "rho4": s(rho, 4), "r_no6": s(vals[1], 3), "r_den": s(vals[2], 3), "tau": s(tau, 3),
        "r_inv": s(vals[4], 3), "conc": conc, "disc": disc,
        **{k: v for k, v in D.items()},
        "ab_c": s(ab_c, 1), "ab_d": s(ab_d, 1), "ab_e": s(ab_e, 1), "ab_de": s(ab_de, 1), "c_de": s(c_de, 1),
        "fin": s(fin, 1), "fin_sum": D["AC"] + D["BC"] + D["AD"] + D["AE"] + D["BD"] + D["BE"],
        "wpgma": s(wpgma, 1), "single": s(D["BC"], 1), "complete": s(D["AD"], 1),
    }
    return out


def p11():
    m1, m2 = (40.0, 30.0), (36.0, 24.0)
    S = ((4.0, 2.0), (2.0, 5.0))
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    Si = ((S[1][1] / det, -S[0][1] / det), (-S[0][1] / det, S[0][0] / det))
    d = (m1[0] - m2[0], m1[1] - m2[1])
    w = (Si[0][0] * d[0] + Si[0][1] * d[1], Si[1][0] * d[0] + Si[1][1] * d[1])
    mid = ((m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2)
    x = (42.0, 24.5)
    dx = (x[0] - mid[0], x[1] - mid[1])
    z = w[0] * dx[0] + w[1] * dx[1]
    z_S = (S[0][0] * d[0] + S[0][1] * d[1]) * dx[0] + (S[1][0] * d[0] + S[1][1] * d[1]) * dx[1]
    z_E = d[0] * dx[0] + d[1] * dx[1]
    z_D = d[0] / S[0][0] * dx[0] + d[1] / S[1][1] * dx[1]
    assert z < 0 and z_S > 0 and z_E > 0 and z_D > 0
    # (b) 1 因子モデルの因子得点（回帰法）
    lam = (0.9, 0.8, 0.6)
    psi = tuple(1 - l * l for l in lam)
    xs = (1.2, 0.8, -0.5)
    lp = [l / p for l, p in zip(lam, psi)]
    num = sum(a * b for a, b in zip(lp, xs))
    den = sum(l * l / p for l, p in zip(lam, psi))
    reg = num / (1 + den)
    bart = num / den
    lx = sum(l * v for l, v in zip(lam, xs))
    vals = [sum(xs) / 3, lx / sum(l * l for l in lam), reg, bart, lx]
    assert len({s(v, 2) for v in vals}) == 5
    return {
        "m11": s(m1[0], 0), "m12": s(m1[1], 0), "m21": s(m2[0], 0), "m22": s(m2[1], 0),
        "s11": s(S[0][0], 0), "s12": s(S[0][1], 0), "s22": s(S[1][1], 0), "det": s(det, 0),
        "d1": s(d[0], 0), "d2": s(d[1], 0), "w1": s(w[0], 1), "w2": s(w[1], 1),
        "mid1": s(mid[0], 0), "mid2": s(mid[1], 0), "x1": s(x[0], 0), "x2": s(x[1], 1),
        "dx1": s(dx[0], 1), "dx2": s(dx[1], 1),
        "z": s(z, 2), "zS": s(z_S, 1), "zE": s(z_E, 2), "zD": s(z_D, 2),
        "Sd1": s(S[0][0] * d[0] + S[0][1] * d[1], 0), "Sd2": s(S[1][0] * d[0] + S[1][1] * d[1], 0),
        "l1": lam[0], "l2": lam[1], "l3": lam[2], "p1": s(psi[0], 2), "p2": s(psi[1], 2), "p3": s(psi[2], 2),
        "x1f": s(xs[0], 1), "x2f": s(xs[1], 1), "x3f": s(xs[2], 1),
        "lp1": s(lp[0], 4), "lp2": s(lp[1], 4), "lp3": s(lp[2], 4),
        "q1": s(lam[0] ** 2 / psi[0], 4), "q2": s(lam[1] ** 2 / psi[1], 4), "q3": s(lam[2] ** 2 / psi[2], 4),
        "num": s(num, 4), "den": s(den, 4), "den1": s(1 + den, 4),
        "reg": s(reg, 2), "reg4": s(reg, 4), "bart": s(bart, 2), "lx": s(lx, 2), "mean": s(sum(xs) / 3, 2),
        "lsum": s(lx / sum(l * l for l in lam), 2),
    }


def p12():
    pu, qd, i0, N = 0.6, 0.4, 3, 8
    rr = qd / pu
    ruin = (rr ** i0 - rr ** N) / (1 - rr ** N)
    ruin_inf = rr ** i0
    # (b) ボックス・ミュラー法
    u1, u2 = 0.30, 0.20
    ln1, ln2, c1, s1, c2 = -1.2040, -1.6094, 0.3090, 0.9511, -0.3090
    z = sqrt(-2 * ln1) * c1
    mu, sg = 0.05, 0.10
    g = mu + sg * z
    g_swap = mu + sg * sqrt(-2 * ln2) * c2
    g_nosq = mu + sg * (-2 * ln1) * c1
    g_sin = mu + sg * sqrt(-2 * ln1) * s1
    g_no2 = mu + sg * sqrt(-ln1) * c1
    vals = [g_swap, g_no2, g, g_nosq, g_sin]
    assert len({s(v, 3) for v in vals}) == 5
    return {
        "pu": pu, "qd": qd, "i0": i0, "N": N, "rr": "2/3", "r3": s(rr ** i0, 4), "r8": s(rr ** N, 4),
        "num": s(rr ** i0 - rr ** N, 4), "den": s(1 - rr ** N, 4),
        "ruin": r(ruin, 3), "ruin_s": s(ruin, 3), "ruin4": s(ruin, 4), "ruin_inf": s(ruin_inf, 3),
        "succ": s(1 - ruin, 3),
        "u1": s(u1, 2), "u2": s(u2, 2), "ln1": s(ln1, 4), "ln2": s(ln2, 4), "c1": s(c1, 4), "s1": s(s1, 4), "c2": s(c2, 4),
        "m2ln": s(-2 * ln1, 3), "sqv": s(sqrt(-2 * ln1), 4), "z": s(z, 4), "mu": mu, "sg": s(sg, 2),
        "g": s(g, 3), "g5": s(mu + sg * r(z, 4), 5), "g_swap": s(g_swap, 3), "g_nosq": s(g_nosq, 3), "g_sin": s(g_sin, 3), "g_no2": s(g_no2, 3),
    }


def values():
    return {"s18p01": p01(), "s18p02": p02(), "s18p03": p03(), "s18p04": p04(), "s18p05": p05(),
            "s18p06": p06(), "s18p07": p07(), "s18p08": p08(), "s18p09": p09(), "s18p10": p10(),
            "s18p11": p11(), "s18p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
