"""模擬試験 第24回（set24）の数値。values() が {大問id: {名前: 値}} を返す。

題材：広告・SNS・メディア分析・出版。
問題文に示した関数の値（指数・対数・正規分布の値、t 分布の点、二項確率）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from math import sqrt, floor


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
    # (a) 離散時間ハザード
    h = [0.20, 0.15, 0.10, 0.08]
    s1, s2, s3 = 1 - h[0], (1 - h[0]) * (1 - h[1]), (1 - h[0]) * (1 - h[1]) * (1 - h[2])
    pa = s3 * h[3]
    surv4 = s3 * (1 - h[3])
    wrong_sum = (1 - sum(h[:3])) * h[3]
    # (b) 指数モデルのハザード比
    d1, T1, d2, T2 = 30, 500, 24, 800
    lam1, lam2 = d1 / T1, d2 / T2
    hr = lam1 / lam2
    ln2 = 0.6931                      # 問題文に示す ln 2
    se2 = 1 / d1 + 1 / d2
    se = sqrt(se2)
    up_exp = ln2 + 1.96 * r(se, 4)
    up_x = r(up_exp, 2)
    etab = {1.23: 3.4212, 0.16: 1.1735, 0.75: 2.1170, 1.06: 2.8864, 0.54: 1.7160}
    assert up_x in etab
    up = etab[up_x]
    lo_x = r(ln2 - 1.96 * r(se, 4), 2)
    assert lo_x in etab
    # 誤り：対数をとらずに HR ± 1.96 SE
    up_lin = hr + 1.96 * r(se, 4)
    # 誤り：SE を sqrt(1/d1)+sqrt(1/d2)
    se_sum = sqrt(1 / d1) + sqrt(1 / d2)
    up_sum_x = r(ln2 + 1.96 * se_sum, 2)
    return {
        "h1": s(h[0], 2), "h2": s(h[1], 2), "h3": s(h[2], 2), "h4": s(h[3], 2),
        "q1": s(1 - h[0], 2), "q2": s(1 - h[1], 2), "q3": s(1 - h[2], 2), "q4": s(1 - h[3], 2),
        "S1": s(s1, 2), "S2": s(s2, 2), "S3": s(s3, 3), "pa": s(pa, 4), "pa5": s(pa, 5),
        "surv4": s(surv4, 4), "atrisk4": s(s3, 4), "w_sum": s(wrong_sum, 4), "hsum3": s(sum(h[:3]), 2),
        "d1": d1, "T1": T1, "d2": d2, "T2": T2, "lam1": s(lam1, 2), "lam2": s(lam2, 2), "hr": s(hr, 1),
        "ln2": s(ln2, 4), "inv1": s(1 / d1, 5), "inv2": s(1 / d2, 5), "se2": s(se2, 5), "se": s(se, 4),
        "mult": s(1.96 * r(se, 4), 4), "up_exp": s(up_exp, 4), "up_x": s(up_x, 2), "lo_x": s(lo_x, 2),
        "e_up": s(up, 4), "e_lo": s(etab[lo_x], 4), "up": r(up, 2), "up_s": s(up, 2), "lo_s": s(etab[lo_x], 2),
        "e016": "1.1735", "e054": "1.7160", "e075": "2.1170", "e106": "2.8864", "e123": "3.4212",
        "up_lin": s(up_lin, 2), "up_sum_x": s(up_sum_x, 2),
    }


def p02():
    # (a) 加重調和平均
    w = [0.5, 0.3, 0.2]
    c = [80, 40, 160]
    inv = [wi / ci for wi, ci in zip(w, c)]
    hm = 1 / sum(inv)
    am_w = sum(wi * ci for wi, ci in zip(w, c))
    am = sum(c) / 3
    hm_simple = 3 / sum(1 / ci for ci in c)
    import math
    gm_w = math.exp(sum(wi * math.log(ci) for wi, ci in zip(w, c)))
    # (b) 費用を考慮した最適配分
    N1, S1, c1 = 40000, 12, 100
    N2, S2, c2 = 10000, 20, 900
    C = 540000
    a1, a2 = N1 * S1 / sqrt(c1), N2 * S2 / sqrt(c2)
    den = N1 * S1 * sqrt(c1) + N2 * S2 * sqrt(c2)
    n1, n2 = C * a1 / den, C * a2 / den
    # 誤り：費用を無視したネイマン配分で同じ予算を使い切る n_h ∝ N_h S_h
    k = C / (N1 * S1 * c1 + N2 * S2 * c2)
    n2_ney = k * N2 * S2
    # 誤り：√c ではなく c で割る
    b1, b2 = N1 * S1 / c1, N2 * S2 / c2
    k2 = C / (b1 * c1 + b2 * c2)
    n2_c = k2 * b2
    return {
        "w1": "50", "w2": "30", "w3": "20", "c1": c[0], "c2": c[1], "c3": c[2],
        "f1": s(inv[0], 5), "f2": s(inv[1], 4), "f3": s(inv[2], 5), "fsum": s(sum(inv), 3),
        "hm": s(hm, 1), "am_w": s(am_w, 1), "am": s(am, 1), "hm_simple": s(hm_simple, 1), "gm_w": s(gm_w, 1),
        "N1": N1, "S1": S1, "cc1": c1, "N2": N2, "S2": S2, "cc2": c2, "C": C, "C_s": "540{,}000",
        "N1s": "40{,}000", "N2s": "10{,}000",
        "sq1": int(sqrt(c1)), "sq2": int(sqrt(c2)), "a1": int(a1), "a2": s(a2, 1),
        "t1": "4{,}800{,}000", "t2": "6{,}000{,}000", "den": "10{,}800{,}000",
        "n1": s(n1, 0), "n2": r(n2, 0), "cost_tot": f"{int(r(n1, 0) * c1 + r(n2, 0) * c2):,}".replace(",", "{,}"), "n2_raw": s(n2, 2), "n2_ney": s(n2_ney, 1), "n2_c": s(n2_c, 1),
        "cost1": int(r(n1, 0) * c1), "cost2": int(r(n2, 0) * c2),
    }


def p03():
    # (a) Box-Cox λ=0.5
    lam, mu, sig, y0 = 0.5, 10.0, 2.5, 64
    w0 = (sqrt(y0) - 1) / lam
    z = (w0 - mu) / sig
    phi = {1.6: 0.9452, 2.4: 0.9918, 0.8: 0.7881, 1.2: 0.8849}
    assert r(z, 1) == 1.6
    ans = 1 - phi[1.6]
    z_sqrt = (sqrt(y0) - mu) / sig          # 変換を √y だけにした → -0.8
    z_nom1 = (sqrt(y0) / lam - mu) / sig    # -1 を忘れた → 2.4
    z_nolam = (sqrt(y0) - 1 - mu) / sig     # /λ を忘れた → -1.2
    assert (r(z_sqrt, 1), r(z_nom1, 1), r(z_nolam, 1)) == (-0.8, 2.4, -1.2)
    # (b) 対数正規：平均を上回る割合 = 1-Φ(σ/2)
    sg = 1.2
    phib = {0.6: 0.7257, 0.72: 0.7642, 1.2: 0.8849}
    ansb = 1 - phib[0.6]
    return {
        "lam": s(lam, 1), "mu": s(mu, 1), "sig": s(sig, 1), "y0": y0, "sy0": int(sqrt(y0)),
        "w0": s(w0, 1), "z": s(z, 1), "phi16": "0.9452", "phi24": "0.9918", "phi08": "0.7881", "phi12": "0.8849",
        "ans_a": s(ans, 4), "a_sqrt": s(phi[0.8], 4), "a_nom1": s(1 - phi[2.4], 4), "a_nolam": s(phi[1.2], 4),
        "a_comp": s(phi[1.6], 4), "z_sqrt": s(z_sqrt, 1), "z_nom1": s(z_nom1, 1), "z_nolam": s(z_nolam, 1),
        "w_nom1": s(sqrt(y0) / lam, 0), "w_nolam": s(sqrt(y0) - 1, 0),
        "sg": s(sg, 1), "sg2h": s(sg ** 2 / 2, 2), "sgh": s(sg / 2, 1), "phi06": "0.7257", "phi072": "0.7642",
        "ans_b": s(ansb, 4), "b_half": "0.5000", "b_var": s(1 - phib[0.72], 4), "b_sig": s(1 - phib[1.2], 4),
        "b_comp": s(phib[0.6], 4),
    }


def p04():
    # (a) 比のデルタ法
    n, mx, my, sx, sy, rho = 25, 50.0, 1.50, 10.0, 0.40, 0.80
    sxy = rho * sx * sy
    R = my / mx
    t1 = sy ** 2
    t2 = 2 * R * sxy
    t3 = R ** 2 * sx ** 2
    inner = t1 - t2 + t3
    var = inner / (n * mx ** 2)
    se_pct = sqrt(var) * 100
    se_nocov = sqrt((t1 + t3) / (n * mx ** 2)) * 100
    se_y = sy / sqrt(n) / mx * 100
    # (b) 正確な二項検定 B(20, 0.1)
    from math import comb
    pm = [round(comb(20, k) * 0.1 ** k * 0.9 ** (20 - k), 4) for k in range(6)]
    cum4 = sum(pm[:5])
    pval = 1 - cum4
    p_eq = pm[5]
    p_gt = 1 - sum(pm)
    return {
        "n": n, "mx": s(mx, 1), "my": s(my, 2), "sx": s(sx, 1), "sy": s(sy, 2), "rho": s(rho, 2),
        "sxy": s(sxy, 2), "R": s(R, 2), "Rpct": s(R * 100, 1), "t1": s(t1, 2), "t2": s(t2, 3), "t3": s(t3, 2),
        "inner": s(inner, 3), "den": "62{,}500", "var_pct": s(var * 1e4, 5), "inner_pct": s(inner * 1e4, 0), "sd_raw": s(sqrt(var), 7),
        "se_pct": r(se_pct, 3), "se_pct_s": s(se_pct, 3), "se_pct5": s(se_pct, 5),
        "se_nocov": s(se_nocov, 3), "se_y": s(se_y, 3),
        "N": 20, "p0": "0.1", "x": 5,
        "pm0": s(pm[0], 4), "pm1": s(pm[1], 4), "pm2": s(pm[2], 4), "pm3": s(pm[3], 4), "pm4": s(pm[4], 4), "pm5": s(pm[5], 4),
        "cum4": s(cum4, 4), "pval": s(pval, 4), "pval3": s(pval, 3), "p_eq": s(p_eq, 3), "p_gt": s(p_gt, 3),
        "p_two": s(2 * pval, 3), "p_low": s(sum(pm), 3),
    }


def p05():
    # (a) ウェルチの信頼区間
    n1, m1, s1 = 12, 182.0, 30.0
    n2, m2, s2 = 10, 150.0, 12.0
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se2 = v1 + v2
    se = sqrt(se2)
    nu = se2 ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    df = floor(nu)
    ttab = {14: 2.145, 15: 2.131, 20: 2.086, 21: 2.080}
    t = ttab[df]
    diff = m1 - m2
    lo = diff - t * r(se, 4)
    lo15 = diff - ttab[15] * r(se, 4)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    se_p = sqrt(sp2 * (1 / n1 + 1 / n2))
    lo_pool = diff - ttab[20] * se_p
    # (b) 区間から t を逆算（各 8 本、等分散、自由度 14）
    L, U = 1.2, 9.6
    mid, half = (L + U) / 2, (U - L) / 2
    tc = 2.145
    se_b = half / tc
    tb = mid / r(se_b, 4)
    t_half = mid / half
    t_z = mid / (half / 1.96)
    return {
        "n1": n1, "m1": s(m1, 0), "s1": s(s1, 0), "n2": n2, "m2": s(m2, 0), "s2": s(s2, 0),
        "v1": s(v1, 1), "v2": s(v2, 1), "se2": s(se2, 1), "se": s(se, 4), "diff": s(diff, 0),
        "v1sq": s(v1 ** 2 / (n1 - 1), 2), "v2sq": s(v2 ** 2 / (n2 - 1), 2), "nu_num": s(se2 ** 2, 2),
        "nu_den": s(v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1), 2), "nu": s(nu, 3), "df": df,
        "t": s(t, 3), "t15": "2.131", "t20": "2.086", "t21": "2.080", "t14": "2.145",
        "mult": s(t * r(se, 4), 4), "lo": r(lo, 2), "lo_s": s(lo, 2), "up_s": s(diff + t * r(se, 4), 2),
        "lo15": s(lo15, 2), "sp2": s(sp2, 1), "se_p": s(se_p, 3), "lo_pool": s(lo_pool, 2),
        "L": s(L, 1), "U": s(U, 1), "mid": s(mid, 1), "half": s(half, 1), "se_b": s(se_b, 4),
        "tb": s(tb, 2), "t_half": s(t_half, 2), "t_z": s(t_z, 2),
        "t01": "2.624", "t005": "2.977", "t05": "1.761",
    }


def p06():
    xs = [3.2, 4.0, 5.5, 3.6, 8.1, 4.4, 6.0, 5.2]
    n = len(xs)
    tot = sum(xs)
    mean = tot / n
    mn = min(xs)
    beta = mean - mn
    ss = sum((x - mean) ** 2 for x in xs)
    sd = sqrt(ss / (n - 1))
    # (b) 逆関数法
    u = 0.75
    ln025, ln075 = -1.3863, -0.2877
    x = mn - r(beta, 1) * ln025
    x_u = mn - r(beta, 1) * ln075
    x_noth = -r(beta, 1) * ln025
    x_rate = mn - ln025 / r(beta, 1)
    x_mean = mn - mean * ln025
    return {
        "x1": "3.2", "x2": "4.0", "x3": "5.5", "x4": "3.6", "x5": "8.1", "x6": "4.4", "x7": "6.0", "x8": "5.2",
        "n": n, "tot": s(tot, 1), "mean": s(mean, 1), "mn": s(mn, 1), "beta": s(beta, 1),
        "mm_th": s(mean - sd, 2), "mm_be": s(sd, 2), "b_unb": s(beta * n / (n - 1), 2), "nn1": n - 1,
        "u": s(u, 2), "ln025": "-1.3863", "ln075": "-0.2877", "prod": s(r(beta, 1) * 1.3863, 4),
        "xg": s(x, 2), "x_u": s(x_u, 2), "x_noth": s(x_noth, 2), "x_rate": s(x_rate, 2), "x_mean": s(x_mean, 2),
    }


def p07():
    # (a) 6 種類のカードのコンプリート
    k = 6
    terms = [k / (k - j) for j in range(k)]
    E = sum(terms)
    H = sum(1 / j for j in range(1, k + 1))
    # (b) 複合ポアソン
    lam, t = 6, 8
    pk = {1: 0.6, 2: 0.3, 3: 0.1}
    EK = sum(kk * p for kk, p in pk.items())
    EK2 = sum(kk * kk * p for kk, p in pk.items())
    VK = EK2 - EK ** 2
    m = lam * t * EK
    v = lam * t * EK2
    sd = sqrt(v)
    return {
        "k": k, "E": r(E, 2), "E_s": s(E, 2), "H": s(H, 2), "E_pois": k * k,
        "e1": s(terms[0], 0), "e2": s(terms[1], 1), "e3": s(terms[2], 1), "e4": s(terms[3], 0), "e5": s(terms[4], 0), "e6": s(terms[5], 0),
        "lam": lam, "t": t, "lt": lam * t, "p1": "0.6", "p2": "0.3", "p3": "0.1",
        "EK": s(EK, 1), "EK2": s(EK2, 1), "VK": s(VK, 2), "m": s(m, 0), "v": s(v, 1),
        "sd": r(sd, 2), "sd_s": s(sd, 2), "sd4": s(sd, 4), "sd_vk": s(sqrt(lam * t * VK), 2),
        "sd_pois": s(sqrt(m), 2), "sd_ek2": s(sqrt(lam * t * EK ** 2), 2),
    }


def p08():
    n, xbar, Sxx, b0, b1, s2 = 12, 3.0, 20.0, 2.0, 1.5, 4.0
    x0 = 5.0
    y0 = b0 + b1 * x0
    g = 1 / n + (x0 - xbar) ** 2 / Sxx
    se_m = sqrt(s2 * g)
    t10, t11, z = 2.228, 2.201, 1.96
    hw = t10 * r(se_m, 4)
    se_p = sqrt(s2 * (1 + g))
    hw_p = t10 * r(se_p, 4)
    hw11 = t11 * r(se_m, 4)
    se_nox = sqrt(s2 / n)
    hw_nox = t10 * se_nox
    hw_z = z * r(se_m, 4)
    iv = lambda h: f"({s(y0 - h, 2)},\\ {s(y0 + h, 2)})"
    # (b) クックの距離
    pts = {"A": (3.6, 0.096), "B": (-1.5, 0.533), "C": (2.4, 0.283)}
    p = 2
    D = {k: e * e * h / (p * s2 * (1 - h) ** 2) for k, (e, h) in pts.items()}
    D1 = {k: e * e * h / (p * s2 * (1 - h)) for k, (e, h) in pts.items()}
    D0 = {k: e * e * h / (p * s2) for k, (e, h) in pts.items()}
    Dp1 = {k: e * e * h / (1 * s2 * (1 - h) ** 2) for k, (e, h) in pts.items()}
    cut = 4 / n
    out = {
        "n": n, "xbar": s(xbar, 1), "Sxx": s(Sxx, 1), "b0": s(b0, 1), "b1": s(b1, 1), "s2": s(s2, 1), "rss": s(s2 * (n - 2), 1),
        "df": n - 2, "x0": s(x0, 1), "y0": s(y0, 1), "g": s(g, 5), "gfrac": s(1 / n, 5), "gx": s((x0 - xbar) ** 2 / Sxx, 1),
        "se_m": s(se_m, 4), "hw": s(hw, 4), "ci": iv(hw), "ci_p": iv(hw_p), "ci_11": iv(hw11), "ci_nox": iv(hw_nox), "ci_z": iv(hw_z),
        "se_p": s(se_p, 4), "hw_p": s(hw_p, 3), "t10": "2.228", "t11": "2.201", "z": "1.96", "dx2": s((x0 - xbar) ** 2, 1),
        "cut": s(cut, 3), "cutfrac": "4/12",
    }
    for k, (e, h) in pts.items():
        out[f"e{k}"] = s(e, 1)
        out[f"h{k}"] = s(h, 3)
        out[f"q{k}"] = s(1 - h, 3)
        out[f"num{k}"] = s(e * e * h, 5)
        out[f"dd{k}"] = s(p * s2 * (1 - h) ** 2, 5)
        out[f"D{k}"] = s(D[k], 4)
        out[f"D{k}2"] = s(D[k], 2)
    out.update({"D1B": s(D1["B"], 2), "D0C": s(D0["C"], 2), "Dp1B": s(Dp1["B"], 2), "DA2": s(D["A"], 2)})
    assert max(D, key=D.get) == "B" and D["A"] < cut < D["C"] < D["B"]
    assert max(D1, key=D1.get) == "B" and max(D1.values()) < cut
    assert max(D0, key=D0.get) == "C" and max(D0.values()) < cut
    assert min(Dp1.values()) > cut
    return out


def p09():
    b = [2.4, -0.6, 1.1, 0.3, -1.8, 0.9]
    l1, l2 = 1.2, 0.5
    th = l1 / 2

    def soft(x, t):
        return (abs(x) - t) * (1 if x > 0 else -1) if abs(x) > t else 0.0
    en = [soft(x, th) / (1 + l2) for x in b]
    nz = sum(1 for x in en if x != 0)
    lasso = [soft(x, th) for x in b]
    thr_l1 = [soft(x, l1) / (1 + l2) for x in b]
    two = [soft(x, th) / (1 + 2 * l2) for x in b]
    ridge = [x / (1 + l2) for x in b]
    out = {f"b{i+1}": s(x, 1) for i, x in enumerate(b)}
    out.update({
        "l1": s(l1, 1), "l2": s(l2, 1), "th": s(th, 1), "dv": s(1 + l2, 1),
        "nz": nz, "en1": s(en[0], 2), "en3": s(en[2], 3), "en5": s(en[4], 2), "en6": s(en[5], 2),
        "la1": s(lasso[0], 2), "nz_la": sum(1 for x in lasso if x != 0),
        "l1_1": s(thr_l1[0], 2), "nz_l1": sum(1 for x in thr_l1 if x != 0),
        "two1": s(two[0], 2), "nz_two": sum(1 for x in two if x != 0),
        "ri1": s(ridge[0], 2), "nz_ri": 6,
        "sb1": s(abs(b[0]) - th, 1), "sb3": s(abs(b[2]) - th, 1), "sb5": s(abs(b[4]) - th, 1), "sb6": s(abs(b[5]) - th, 1),
    })
    return out


def p10():
    a, b, c, d = 30, 70, 20, 180
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    num = a * d - b * c
    den = r1 * r2 * c1 * c2
    phi = num / sqrt(den)
    chi = n * phi ** 2
    Q = num / (a * d + b * c)
    OR = a * d / (b * c)
    # (b)
    lnp, lnq = -1.7918, -0.1823
    ll0 = c1 * lnp + c2 * lnq
    ll1 = -112.0
    R2 = 1 - ll1 / ll0
    ratio = ll1 / ll0
    k = 3
    adj = 1 - (ll1 - k) / ll0
    lr = 2 * (ll1 - ll0)
    import math
    ll_half = n * math.log(0.5)
    R2_half = 1 - ll1 / ll_half
    return {
        "a": a, "b": b, "c": c, "d": d, "n": n, "r1": r1, "r2": r2, "c1": c1, "c2": c2,
        "ad": a * d, "bc": b * c, "num": num, "den": "250{,}000{,}000", "sqden": s(sqrt(den), 2),
        "phi": s(phi, 3), "phi4": s(phi, 4), "phi2": s(phi ** 2, 3), "chi": s(chi, 1),
        "Q": s(Q, 3), "OR": s(OR, 2), "pd": s(a / r1 - c / r2, 3),
        "p_buzz": "1/6", "lnp": s(lnp, 4), "lnq": s(lnq, 4), "t0a": s(c1 * lnp, 2), "t0b": s(c2 * lnq, 3),
        "ll0": s(ll0, 3), "ll1": s(ll1, 1), "R2": s(R2, 3), "ratio": s(ratio, 3), "adj": s(adj, 3), "k": k,
        "lr": s(lr, 2), "R2_half": s(R2_half, 3), "ll_half": s(ll_half, 2),
    }


def p11():
    hum = [2, 4, 6, 8, 11, 19]
    bot = [13, 15, 17, 20, 24, 29]
    # (b) 2 次判別
    import math
    mu_b, sb = 3.0, 2.0
    ln2 = 0.6931
    cst = (9 + 8 * ln2) / 3
    disc = 1 + cst
    rt = sqrt(disc)
    z1, z2 = -1 + rt, -1 - rt
    cst_no = 9 / 3
    rt_no = sqrt(1 + cst_no)
    cst_v = (9 + 16 * ln2) / 3
    rt_v = sqrt(1 + cst_v)
    return {
        "h1": hum[0], "h2": hum[1], "h3": hum[2], "h4": hum[3], "h5": hum[4], "h6": hum[5],
        "g1": bot[0], "g2": bot[1], "g3": bot[2], "g4": bot[3], "g5": bot[4], "g6": bot[5],
        "P1": "0.0022", "P56": "0.0260", "P46": "0.1429", "P36": "0.4740", "P56x2": "0.0520",
        "mub": s(mu_b, 0), "sb": s(sb, 0), "ln2": s(ln2, 4), "ln2x8": s(8 * ln2, 4), "c9": s(9 + 8 * ln2, 4),
        "cst": s(cst, 4), "disc": s(disc, 4), "rt": s(rt, 4), "z1": s(z1, 2), "z2": s(z2, 2),
        "z1_4": s(z1, 4), "z2_4": s(z2, 4), "mid": s(mu_b / 2, 2),
        "zn1": s(-1 + rt_no, 2), "zn2": s(-1 - rt_no, 2), "zv1": s(-1 + rt_v, 2), "zv2": s(-1 - rt_v, 2),
    }


def p12():
    # (a) 主成分得点からマハラノビス距離
    lam = [4.0, 1.0, 0.25]
    x = [1.5, 0.5, 2.5]
    s1 = x[0] + x[1] + x[2]
    s2 = x[0] - x[2]
    s3 = x[0] - 2 * x[1] + x[2]
    t1sq, t2sq, t3sq = s1 ** 2 / 3, s2 ** 2 / 2, s3 ** 2 / 6
    d2 = t1sq / lam[0] + t2sq / lam[1] + t3sq / lam[2]
    eu = sum(v * v for v in x)
    d_sqrtl = t1sq / sqrt(lam[0]) + t2sq / sqrt(lam[1]) + t3sq / sqrt(lam[2])
    d_two = t1sq / lam[0] + t2sq / lam[1]
    # (b) 正準相関
    C = [[0.5, 0.3], [0.2, 0.4]]
    M = [[C[0][0] ** 2 + C[0][1] ** 2, C[0][0] * C[1][0] + C[0][1] * C[1][1]],
         [C[0][0] * C[1][0] + C[0][1] * C[1][1], C[1][0] ** 2 + C[1][1] ** 2]]
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] ** 2
    disc = tr ** 2 - 4 * det
    l1 = (tr + sqrt(disc)) / 2
    l2 = (tr - sqrt(disc)) / 2
    return {
        "L1": s(lam[0], 2), "L2": s(lam[1], 2), "L3": s(lam[2], 2), "x1": s(x[0], 1), "x2": s(x[1], 1), "x3": s(x[2], 1),
        "s1": s(s1, 1), "s2": s(s2, 1), "s3": s(s3, 1), "t1sq": s(t1sq, 2), "t2sq": s(t2sq, 2), "t3sq": s(t3sq, 2),
        "t1": s(s1 / sqrt(3), 4), "t2": s(s2 / sqrt(2), 4), "t3": s(s3 / sqrt(6), 4),
        "q1": s(t1sq / lam[0], 4), "q2": s(t2sq / lam[1], 2), "q3": s(t3sq / lam[2], 2), "eu": s(eu, 2),
        "d2": r(d2, 2), "d2_s": s(d2, 2), "d2_4": s(d2, 4), "d_sqrtl": s(d_sqrtl, 2), "d_two": s(d_two, 2),
        "c11": s(C[0][0], 1), "c12": s(C[0][1], 1), "c21": s(C[1][0], 1), "c22": s(C[1][1], 1),
        "m11": s(M[0][0], 2), "m12": s(M[0][1], 2), "m22": s(M[1][1], 2), "tr": s(tr, 2), "det": s(det, 4),
        "disc": s(disc, 4), "sqd": s(sqrt(disc), 4), "l1": s(l1, 4), "l2": s(l2, 4),
        "rho1": s(sqrt(l1), 3), "rho2": s(sqrt(l2), 3), "l1_3": s(l1, 3), "sqtr": s(sqrt(tr), 3),
        "diag": s(sqrt(C[0][0] ** 2 + C[1][1] ** 2), 3),
    }


def p13():
    return {"day1": "(1),\\ ab,\\ ac,\\ bc", "day2": "a,\\ b,\\ c,\\ abc", "abblk": "(1),\\ ab,\\ c,\\ abc",
            "low": "(1),\\ a,\\ b,\\ c"}


def p14():
    a, b, c = 0.6, 0.3, 0.5
    vz, ve = 1.0, 1.0
    vx = a * a * vz + ve
    cxz = a * vz
    cxy = b * vx + c * cxz
    naive = cxy / vx
    return {
        "a": s(a, 1), "b": s(b, 1), "c": s(c, 1), "vx": s(vx, 2), "cxz": s(cxz, 1), "bvx": s(b * vx, 3), "ccxz": s(c * cxz, 1),
        "cxy": s(cxy, 3), "naive": s(naive, 2), "naive4": s(naive, 4), "bias": s(naive - b, 2), "w_v1": s(b + c * a, 2),
        "a2": s(a * a, 2),
    }


def values():
    return {"s24p01": p01(), "s24p02": p02(), "s24p03": p03(), "s24p04": p04(), "s24p05": p05(),
            "s24p06": p06(), "s24p07": p07(), "s24p08": p08(), "s24p09": p09(), "s24p10": p10(),
            "s24p11": p11(), "s24p12": p12(), "s24p13": p13(), "s24p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
