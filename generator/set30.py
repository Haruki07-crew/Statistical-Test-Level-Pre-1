"""模擬試験 第30回（set30）の数値。values() が {大問id: {名前: 値}} を返す。

問題文に示した分布の値（t 点・χ² 点・スチューデント化範囲の点、e^x・対数・べき乗根など）は、
ここでも同じ丸めた値を使って答えを計算する。
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


# ---- 大問1：棚替え前後の日販（対応のある t 検定・符号検定）
BEFORE = [52.0, 61.0, 48.0, 70.0, 45.5, 59.0, 63.0, 51.0, 56.5, 60.0]
DIFF = [12.0, -3.0, 8.0, 15.0, 5.0, -2.0, 9.0, 11.0, 4.0, 6.0]
AFTER = [b + d for b, d in zip(BEFORE, DIFF)]


def p01():
    n = len(DIFF)
    dbar = sum(DIFF) / n
    ss = sum((d - dbar) ** 2 for d in DIFF)
    sd2 = ss / (n - 1)
    sd = sqrt(sd2)
    se = sd / sqrt(n)
    t = dbar / se
    t_n = dbar / (sqrt(ss / n) / sqrt(n))       # 分散を n で割った
    t_nosqrt = dbar / sd                         # √n を掛け忘れ
    t_n1 = dbar / (sd / sqrt(n - 1))             # √(n-1) で割った
    # 2 標本 t（対応を無視）
    mb, ma = sum(BEFORE) / n, sum(AFTER) / n
    vb = sum((x - mb) ** 2 for x in BEFORE) / (n - 1)
    va = sum((x - ma) ** 2 for x in AFTER) / (n - 1)
    t2 = (ma - mb) / sqrt((vb + va) / 2 * (2 / n))
    vals = [t, t_n, t_nosqrt, t_n1, t2]
    assert len({s(v, 2) for v in vals}) == 5
    # (b) 符号検定
    k = sum(1 for d in DIFF if d > 0)
    tail = sum(comb(n, j) for j in range(k, n + 1))
    p1 = tail / 2 ** n
    return {
        **{f"b{i}": s(x, 1) for i, x in enumerate(BEFORE)},
        **{f"a{i}": s(x, 1) for i, x in enumerate(AFTER)},
        **{f"d{i}": s(x, 1) for i, x in enumerate(DIFF)},
        "n": n, "n1": n - 1, "sumd": s(sum(DIFF), 1), "dbar": s(dbar, 2), "ss": s(ss, 2),
        "sd2": s(sd2, 4), "sd": s(sd, 4), "se": s(se, 4), "rootn": s(sqrt(n), 4),
        "t": s(t, 2), "t4": s(t, 4), "t_n": s(t_n, 2), "t_nosqrt": s(t_nosqrt, 2), "t_n1": s(t_n1, 2),
        "t2": s(t2, 2), "mb": s(mb, 2), "ma": s(ma, 2), "vb": s(vb, 2), "va": s(va, 2),
        "tcrit": "1.833",
        "k": k, "nneg": n - k, "c10": comb(n, k), "c9": comb(n, k + 1), "c8": comb(n, k + 2),
        "tail": tail, "pow": 2 ** n,
        "p1": r(p1, 4), "p1s": s(p1, 4), "p1raw": s(p1, 7), "p_eq": s(comb(n, k) / 2 ** n, 4),
        "p2": s(2 * p1, 4),
    }


def p02():
    x0 = 0.2
    # f(y|x0) = (x0 + y) / (x0 + 1/2)
    num = x0 * 0.5 + (1 - 0.25) / 2
    den = x0 + 0.5
    cond = num / den
    marg = 0.5 * 0.5 + (1 - 0.25) / 2          # P(Y>1/2) = ∫(y+1/2)dy
    joint = num                                   # 分母で割り忘れ
    return {"x0": x0, "num": s(num, 3), "den": s(den, 1), "cond": r(cond, 3), "cond4": s(cond, 4),
            "marg": s(marg, 3), "joint": s(joint, 3), "x0half": s(x0 * 0.5, 2)}


def p03():
    vz, a, ve1, b, ve2 = 16, 3, 81, -2, 36
    cov = a * b * vz
    vx = a ** 2 * vz + ve1
    vy = b ** 2 * vz + ve2
    rho = cov / sqrt(vx * vy)
    w_var = cov / vx            # 片方の分散で割った
    w_noerr = cov / sqrt(a ** 2 * vz * b ** 2 * vz)   # 誤差の分散を足し忘れた
    w_err = cov / sqrt(ve1 * ve2)
    vals = [rho, w_var, w_noerr, 0.0, cov / vy]
    assert len({s(v, 2) for v in vals}) == 5, vals
    return {"vz": vz, "sz": 4, "a": a, "ve1": ve1, "b": b, "bab": abs(b), "ve2": ve2, "cov": cov,
            "vx": vx, "vy": vy, "sx": s(sqrt(vx), 0), "sy": s(sqrt(vy), 0),
            "a2vz": a ** 2 * vz, "b2vz": b ** 2 * vz,
            "rho": s(rho, 2), "w_var": s(w_var, 2), "w_noerr": s(w_noerr, 2), "w_vy": s(cov / vy, 2),
            "w_err": s(w_err, 2)}


def p04():
    lam = 0.5     # 人/分
    t = 6
    k = 5
    m = lam * t   # 3
    e3 = 0.0498
    terms = [1, 3, 4.5, 4.5, 3.375]      # m^j/j!, j=0..4
    ssum = sum(terms)
    ok = 1 - e3 * ssum
    w4 = 1 - e3 * sum(terms[:4])          # N≥4
    w6 = 1 - e3 * (ssum + 3 ** 5 / 120)   # N≥6
    eq5 = e3 * 3 ** 5 / 120               # P(N=5)
    vals = [ok, w4, w6, eq5, 1 - ok]
    assert len({s(v, 3) for v in vals}) == 5
    # (b) M/M/1
    svc = 1.6
    mu = 1 / svc
    rho = lam / mu
    L = rho / (1 - rho)
    W = L / lam
    Wq = W - svc
    Lq = rho ** 2 / (1 - rho)
    vals2 = [Wq, W, Lq / mu, L, Lq]
    assert len({s(v, 2) for v in vals2}) == 5, vals2
    return {"lam": lam, "t": t, "k": k, "m": s(m, 0), "e3": e3, "ssum": s(ssum, 3),
            "prod": s(e3 * ssum, 6), "ok": s(ok, 3), "ok6": s(ok, 6), "w4": s(w4, 3), "w6": s(w6, 3),
            "eq5": s(eq5, 3), "comp": s(1 - ok, 3), "t4": s(3 ** 5 / 120, 3),
            "svc": svc, "mu": s(mu, 3), "rho": s(rho, 1), "L": s(L, 1), "W": s(W, 1), "Wq": s(Wq, 1),
            "Lq": s(Lq, 1), "Lqmu": s(Lq / mu, 2), "mean_gap": s(1 / lam, 0)}


AREA = [1.0, 1.5, 2.0, 2.5]
SALES = [34, 41, 63, 70]


def p05():
    A = sum(AREA)
    X = sum(SALES)
    th = X / A
    se = sqrt(th / A)
    naive = sum(x / a for x, a in zip(SALES, AREA)) / 4
    se_wrong = sqrt(th) / sqrt(4)   # 店舗数で割った
    # (b) 新店
    a_new, x_new = 2.0, 80
    th0 = (X + x_new) / (A + a_new)
    th2 = x_new / a_new
    ln1, ln2 = -0.0741, 0.2231       # ln(13/14), ln(5/4)
    g = 2 * (X * ln1 + x_new * ln2)
    assert abs(th / th0 - 13 / 14) < 1e-12 and abs(th2 / th0 - 1.25) < 1e-12
    return {**{f"ar{i}": s(a, 1) for i, a in enumerate(AREA)}, **{f"x{i}": x for i, x in enumerate(SALES)},
            "A": s(A, 1), "X": X, "th": s(th, 4), "th2d": s(th, 2), "thA": s(th / A, 4),
            "se": r(se, 2), "se4": s(se, 4), "naive": s(naive, 2), "se_wrong": s(se_wrong, 2),
            "anew": s(a_new, 1), "xnew": x_new, "XX": X + x_new, "AA": s(A + a_new, 1), "th0": s(th0, 0),
            "th2": s(th2, 0), "ln1": ln1, "ln2": ln2, "t1": s(X * ln1, 4), "t2": s(x_new * ln2, 3),
            "g": s(g, 2), "ghalf": s(g / 2, 2), "chi1": "3.841", "chi2": "5.991",
            }


def p06():
    # (a) フィッシャーの正確検定
    n_new, n_old, s_new, s_old = 8, 7, 6, 1
    N = n_new + n_old
    K = s_new + s_old
    tot = comb(N, n_new)
    pr = {x: comb(K, x) * comb(N - K, n_new - x) for x in range(0, K + 1) if n_new - x <= N - K}
    tail = sum(v for x, v in pr.items() if x >= s_new)
    p = tail / tot
    p_eq = pr[s_new] / tot
    # 両側（確率が観測値以下の表の和）
    p_two = sum(v for v in pr.values() if v <= pr[s_new]) / tot
    # (b) クロッパー・ピアソンの片側上限
    n = 60
    root = 0.9513   # 0.05^(1/60)
    up = 1 - root
    return {"nnew": n_new, "nold": n_old, "snew": s_new, "sold": s_old, "fnew": n_new - s_new,
            "fold": n_old - s_old, "N": N, "K": K, "NK": N - K, "tot": tot,
            "c76": comb(K, 6), "c82": comb(N - K, 2), "c77": comb(K, 7), "c81": comb(N - K, 1),
            "num6": pr[6], "num7": pr[7], "tail": tail,
            "p": r(p, 4), "p6": s(p, 6), "p_eq": s(p_eq, 4), "p_two": s(p_two, 4),
            "n60": n, "root": root, "up": r(up, 4), "up4": s(up, 4), "rule3": s(3 / n, 4),
            }


def p07():
    beta = -0.0025
    p_now = 480
    el = beta * p_now
    p_star = -1 / beta
    # (b) 内部スチューデント化残差
    s_res, e, h = 0.12, 0.30, 0.36
    rstd = e / (s_res * sqrt(1 - h))
    vals_b = [rstd, e / s_res, e / (s_res * (1 - h)), e * sqrt(1 - h) / s_res, e / (s_res * sqrt(h))]
    assert len({s(v, 2) for v in vals_b}) == 5, vals_b
    # (c) 同時決定バイアス
    delta, vu, vv = 100, 0.04, 900
    vp = delta ** 2 * vu + vv
    cpu = delta * vu
    plim = beta + cpu / vp
    w_v = beta + cpu / vv
    w_sign = beta - cpu / vp
    w_du = beta + cpu / (delta ** 2 * vu)
    vals_c = [plim, beta, w_v, w_sign, w_du]
    assert len({s(v * 1000, 2) for v in vals_c}) == 5, vals_c
    return {"beta": beta, "betak": s(beta * 1000, 1), "pnow": p_now, "el": s(el, 1), "pstar": s(p_star, 0),
            "p2star": s(-2 / beta, 0), "elinv": s(1 / el, 2),
            "nobs": 40, "sres": s_res, "e": s(e, 2), "h": s(h, 2), "omh": s(1 - h, 2), "romh": s(sqrt(1 - h), 1),
            "den": s(s_res * sqrt(1 - h), 3), "rstd": s(rstd, 2), "rraw": s(e / s_res, 2),
            "rnosq": s(e / (s_res * (1 - h)), 2), "rmul": s(e * sqrt(1 - h) / s_res, 2), "rh": s(e / (s_res * sqrt(h)), 2),
            "delta": delta, "vu": vu, "su": s(sqrt(vu), 1), "vv": vv, "sv": s(sqrt(vv), 0), "d2vu": s(delta ** 2 * vu, 0),
            "vp": s(vp, 0), "cpu": s(cpu, 0), "bias": s(cpu / vp * 1000, 2),
            "plim": s(plim * 1000, 2), "w_beta": s(beta * 1000, 2), "w_v": s(w_v * 1000, 2),
            "w_sign": s(w_sign * 1000, 2), "w_du": s(w_du * 1000, 2), "biasraw": s(cpu / vp, 6),
            }


# 大問8 の分割表：層ごとに (クーポンあり購入, あり非購入, なし購入, なし非購入)
STRATA = {"大型店": (40, 60, 15, 35), "小型店": (16, 34, 14, 86)}


def p08():
    out = {}
    num = 0.0
    var = 0.0
    var_n = 0.0
    sumE = 0.0
    for key, (a, b, c, d) in zip(["L", "S"], STRATA.values()):
        n1, n0 = a + b, c + d
        m1, m0 = a + c, b + d
        N = n1 + n0
        Ea = n1 * m1 / N
        Va = n1 * n0 * m1 * m0 / (N ** 2 * (N - 1))
        Vn = n1 * n0 * m1 * m0 / N ** 3
        out.update({f"{key}a": a, f"{key}b": b, f"{key}c": c, f"{key}d": d, f"{key}n1": n1, f"{key}n0": n0,
                    f"{key}m1": m1, f"{key}m0": m0, f"{key}N": N, f"{key}E": s(Ea, 4), f"{key}V": s(Va, 4),
                    f"{key}dev": s(a - Ea, 4), f"{key}N1": N - 1})
        num += a - Ea
        var += Va
        var_n += Vn
        sumE += Ea
    # 表示した4桁の値から計算する
    dL, dS = float(out["Ldev"]), float(out["Sdev"])
    VL, VS = float(out["LV"]), float(out["SV"])
    num_d = dL + dS
    var_d = VL + VS
    mh = num_d ** 2 / var_d
    mh_cc = (abs(num_d) - 0.5) ** 2 / var_d
    mh_n = num ** 2 / var_n
    # 層を無視した併合表のカイ二乗（連続修正なし）
    A = sum(t[0] for t in STRATA.values()); B = sum(t[1] for t in STRATA.values())
    C = sum(t[2] for t in STRATA.values()); D = sum(t[3] for t in STRATA.values())
    NN = A + B + C + D
    chi_pool = NN * (A * D - B * C) ** 2 / ((A + B) * (C + D) * (A + C) * (B + D))
    sep = dL ** 2 / VL + dS ** 2 / VS
    out.update({"num": s(num_d, 4), "num2": s(num_d ** 2, 4), "var": s(var_d, 4), "mh": r(mh, 2), "mh4": s(mh, 4),
                "mh_cc": s(mh_cc, 2), "mh_n": s(mh_n, 2), "chi_pool": s(chi_pool, 2), "sep": s(sep, 2),
                "pA": A, "pB": B, "pC": C, "pD": D, "pN": NN})
    return out


def p09():
    b0, b1, d = -2.0, 0.08, 10
    eta = b0 + b1 * d
    e12 = 3.3201
    p = 1 / (1 + e12)
    me = b1 * p * (1 - p)
    vals = [me, b1 * p, b1 * (1 - p), b1, p * (1 - p)]
    assert len({s(v, 4) for v in vals}) == 5
    # (b) 一様分布の積
    t = 0.25
    ln4 = 1.3863
    cdf = t + t * ln4         # t - t ln t
    surv = 1 - cdf
    vals2 = [surv, cdf, 0.75 ** 2, 0.25, 0.75]
    assert len({s(v, 3) for v in vals2}) == 5
    return {"b0": s(b0, 1), "b1": b1, "d": d, "eta": s(eta, 1), "etaabs": s(-eta, 1), "e12": e12,
            "p": s(p, 4), "q": s(1 - p, 4), "pq": s(p * (1 - p), 4), "me": s(me, 4), "me_pt": s(me * 100, 2),
            "w_bp": s(b1 * p, 4), "w_bq": s(b1 * (1 - p), 4), "w_b": s(b1, 4), "w_pq": s(p * (1 - p), 4),
            "onepe": s(1 + e12, 4),
            "tt": t, "ln4": ln4, "tln": s(t * ln4, 4), "cdf": s(cdf, 4), "surv": s(surv, 3), "surv4": s(surv, 4),
            "w_cdf": s(cdf, 3), "w_ind": s(0.75 ** 2, 3), "w_u": "0.250", "w_u2": "0.750"}


def p10():
    means = {"A": 52.0, "B": 46.5, "C": 44.0}
    n, k, mse = 6, 3, 16.0
    dfe = k * (n - 1)
    q = 3.67
    t = 2.131
    hsd = q * sqrt(mse / n)
    hsd2 = q * sqrt(2 * mse / n)
    lsd = t * sqrt(2 * mse / n)
    hsd_nosq = q * sqrt(mse)
    sig = lambda thr: [p for p, (x, y) in {"A–B": ("A", "B"), "A–C": ("A", "C"), "B–C": ("B", "C")}.items()
                       if abs(means[x] - means[y]) > thr]
    assert sig(hsd) == ["A–C"] and sig(hsd2) == [] and sig(lsd) == ["A–B", "A–C"] and sig(hsd_nosq) == []
    return {"mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1), "n": n, "k": k,
            "mse": s(mse, 1), "dfe": dfe, "q": q, "tq": t, "msen": s(mse / n, 4), "rmsen": s(sqrt(mse / n), 4),
            "hsd": s(hsd, 2), "hsd2": s(hsd2, 2), "lsd": s(lsd, 2), "hsdns": s(hsd_nosq, 2),
            "dAB": s(means["A"] - means["B"], 1), "dAC": s(means["A"] - means["C"], 1),
            "dBC": s(means["B"] - means["C"], 1), "r2msen": s(sqrt(2 * mse / n), 4)}


JK_Y = [60, 75, 90, 55]
JK_X = [10, 12, 16, 8]


def p11():
    N1, n1, N2, n2 = 60, 12, 240, 16
    sum1, sum2 = 180.0, 96.0
    w1, w2 = N1 / n1, N2 / n2
    tot = w1 * sum1 + w2 * sum2
    mean = tot / (N1 + N2)
    naive = (sum1 + sum2) / (n1 + n2)
    eq = (sum1 / n1 + sum2 / n2) / 2
    # (b) ジャックナイフ
    n = len(JK_Y)
    R = sum(JK_Y) / sum(JK_X)
    loo = [(sum(JK_Y) - y) / (sum(JK_X) - x) for y, x in zip(JK_Y, JK_X)]
    loo_s = [float(s(v, 4)) for v in loo]
    R_s = float(s(R, 4))
    mloo = sum(loo_s) / n
    rj = n * R_s - (n - 1) * mloo
    bias = (n - 1) * (mloo - R_s)
    mean_ratio = sum(y / x for y, x in zip(JK_Y, JK_X)) / n
    return {"N1": N1, "n1": n1, "N2": N2, "n2": n2, "sum1": s(sum1, 0), "sum2": s(sum2, 0),
            "w1": s(w1, 0), "w2": s(w2, 0), "t1": s(w1 * sum1, 0), "t2": s(w2 * sum2, 0), "tot": s(tot, 0),
            "NN": N1 + N2, "mean": r(mean, 2), "means": s(mean, 2), "naive": s(naive, 2), "eq": s(eq, 2),
            "m1": s(sum1 / n1, 1), "m2": s(sum2 / n2, 1),
            **{f"y{i}": y for i, y in enumerate(JK_Y)}, **{f"x{i}": x for i, x in enumerate(JK_X)},
            "sy": sum(JK_Y), "sx": sum(JK_X), "R": s(R, 4),
            **{f"l{i}": s(v, 4) for i, v in enumerate(loo)},
            **{f"ly{i}": sum(JK_Y) - y for i, y in enumerate(JK_Y)}, **{f"lx{i}": sum(JK_X) - x for i, x in enumerate(JK_X)},
            "mloo": s(mloo, 4), "rj": r(rj, 2), "rj4": s(rj, 4), "bias": s(bias, 4), "nR": s(n * R_s, 4),
            "n1m": s((n - 1) * mloo, 4), "mratio": s(mean_ratio, 2), "jn": n, "jn1": n - 1,
            }


def p12():
    pts = {"A": (0, 0), "B": (1, 1), "C": (3, 1), "D": (4, 4)}
    d2 = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
    D = {a + b: d2(pts[a], pts[b]) for a, b in ["AB", "AC", "AD", "BC", "BD", "CD"]}
    # 1 段目：A と B（距離 2）
    dABC = 0.5 * D["AC"] + 0.5 * D["BC"] - 0.25 * D["AB"]
    dABD = 0.5 * D["AD"] + 0.5 * D["BD"] - 0.25 * D["AB"]
    # 2 段目：{A,B} と C（6.5 < 10 = CD）
    assert dABC < D["CD"] < dABD
    fin = 2 / 3 * dABD + 1 / 3 * D["CD"] - 2 / 9 * dABC
    w_nominus = 2 / 3 * dABD + 1 / 3 * D["CD"]
    w_half = 0.5 * dABD + 0.5 * D["CD"] - 0.25 * dABC
    w_avg = (D["AD"] + D["BD"] + D["CD"]) / 3
    w_single = min(D["AD"], D["BD"], D["CD"])
    vals = [fin, w_nominus, w_half, w_avg, w_single]
    assert len({s(v, 2) for v in vals}) == 5
    return {**{f"d{k}": v for k, v in D.items()}, "dABC": s(dABC, 1), "dABD": s(dABD, 1),
            "fin": s(fin, 2), "w_nominus": s(w_nominus, 2), "w_half": s(w_half, 2), "w_avg": s(w_avg, 2),
            "w_single": s(w_single, 2), "f1": s(2 / 3 * dABD, 4), "f2": s(D["CD"] / 3, 4), "f3": s(2 / 9 * dABC, 4)}


# 大問13 (b)：会員の継続月数（+ は打ち切り）
PLAN_A = [(2, 1), (3, 1), (4.5, 0), (5, 1), (6, 0)]
PLAN_B = [(3.5, 0), (4, 1), (6.5, 0), (7, 1), (8, 0)]


def logrank(A, B, keep_censored=False):
    times = sorted({t for t, e in A + B if e})
    O = E = V = 0.0
    rows = []
    for t in times:
        if keep_censored:
            nA = sum(1 for u, e in A if u >= t or e == 0)
            nB = sum(1 for u, e in B if u >= t or e == 0)
        else:
            nA = sum(1 for u, e in A if u >= t)
            nB = sum(1 for u, e in B if u >= t)
        dA = sum(1 for u, e in A if u == t and e)
        dB = sum(1 for u, e in B if u == t and e)
        d, n = dA + dB, nA + nB
        e_ = nA * d / n
        v = nA * nB * d * (n - d) / (n ** 2 * (n - 1)) if n > 1 else 0.0
        O += dA; E += e_; V += v
        rows.append((t, nA, nB, dA, dB, e_, v))
    return O, E, V, rows


def p13():
    # (a) SEM の自由度
    p = 6
    mom = p * (p + 1) // 2
    free = 4 + 6 + 1 + 1 + 1
    df = mom - free
    # (b) ログランク
    O, E, V, rows = logrank(PLAN_A, PLAN_B)
    Es = [float(s(x[5], 4)) for x in rows]
    Vs = [float(s(x[6], 4)) for x in rows]
    Esum, Vsum = sum(Es), sum(Vs)
    chi = (O - Esum) ** 2 / Vsum
    totd = sum(x[3] + x[4] for x in rows)
    EB = totd - Esum
    w_pois = (O - Esum) ** 2 / Esum + (totd - O - EB) ** 2 / EB
    w_EA = (O - Esum) ** 2 / Esum
    w_nosq = (O - Esum) / Vsum
    O2, E2, V2, _ = logrank(PLAN_A, PLAN_B, keep_censored=True)
    w_keep = (O2 - E2) ** 2 / V2
    vals = [chi, w_pois, w_EA, w_nosq, w_keep]
    assert len({s(v, 2) for v in vals}) == 5, vals
    out = {"p": p, "mom": mom, "free": free, "df": df, "w_all6": mom - (free + 2), "w_p2": p * p - free,
           "w_offdiag": p * (p - 1) // 2 - free, "w_nopath": mom - (free - 1),
           "O": int(O), "Esum": s(Esum, 4), "Vsum": s(Vsum, 4), "OE": s(O - Esum, 4), "OE2": s((O - Esum) ** 2, 4),
           "chi": s(chi, 2), "w_pois": s(w_pois, 2), "w_EA": s(w_EA, 2), "w_nosq": s(w_nosq, 2),
           "w_keep": s(w_keep, 2), "EB": s(EB, 4), "OB": int(totd - O), "totd": totd}
    for i, (t, nA, nB, dA, dB, e_, v) in enumerate(rows):
        out.update({f"rt{i}": s(t, 0), f"rnA{i}": nA, f"rnB{i}": nB, f"rdA{i}": dA, f"rdB{i}": dB,
                    f"rE{i}": s(e_, 4), f"rV{i}": s(v, 4)})
    return out


def values():
    return {"s30p01": p01(), "s30p02": p02(), "s30p03": p03(), "s30p04": p04(), "s30p05": p05(),
            "s30p06": p06(), "s30p07": p07(), "s30p08": p08(), "s30p09": p09(), "s30p10": p10(),
            "s30p11": p11(), "s30p12": p12(), "s30p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
