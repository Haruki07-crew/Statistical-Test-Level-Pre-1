"""模擬試験 第12回（set12）の数値。values() が {大問id: {名前: 値}} を返す。

題材：化学・材料・食品・醸造の品質管理。
問題文に示した関数値（Φ・arctan・ln・χ² の点など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP


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
    # 吸光度の比：独立な N(0,σ²) の比はコーシー分布。標本平均もコーシー分布
    n = 10
    pi = 3.1416
    at1 = 0.7854          # arctan 1 = π/4
    at_s = 1.2645         # arctan √10
    at_n = 1.4711         # arctan 10
    phi1, phi316 = 0.8413, 0.9992
    return {
        "n": n, "pi": s(pi, 4), "at_s": s(at_s, 4), "at_n": s(at_n, 4), "phi1": s(phi1, 4), "phi316": s(phi316, 4),
        "ans": s(2 * at1 / pi, 3), "c_sqrt": s(2 * at_s / pi, 3), "c_n": s(2 * at_n / pi, 3),
        "c_norm": s(2 * phi1 - 1, 3), "c_clt": s(2 * phi316 - 1, 3),
        "c_sqrt4": s(2 * at_s / pi, 4), "c_n4": s(2 * at_n / pi, 4),
    }


def p02():
    # (a) 抜取検査 n=20, c=1, p=0.05
    n, c, p = 20, 1, 0.05
    q19, q20 = 0.3774, 0.3585
    p0 = q20
    p1 = n * p * q19
    acc = p0 + p1
    # (b) 加算推定量 (S+1)/(m+2) の平均二乗誤差
    m = 48
    bias = (m * p + 1) / (m + 2) - p
    vr = m * p * (1 - p) / (m + 2) ** 2
    mse = vr + bias ** 2
    mse_u = p * (1 - p) / m
    bias_w = 1 / (m + 2)
    return {
        "n": n, "c": c, "p": s(p, 2), "q19": s(q19, 4), "q20": s(q20, 4), "p0": s(p0, 4), "p1": s(p1, 4),
        "acc": s(acc, 4), "risk": s(1 - acc, 3), "risk4": s(1 - acc, 4), "o_p0": s(p0, 3), "o_p1": s(p1, 3),
        "o_only0": s(1 - p0, 3), "o_acc": s(acc, 3),
        "m": m, "m2": m + 2, "mp": s(m * p, 1), "mp1": s(m * p + 1, 1), "mean_est": s((m * p + 1) / (m + 2), 3),
        "bias": s(bias, 3), "mpq": s(m * p * (1 - p), 2), "m22": (m + 2) ** 2,
        "vr": s(vr * 1e4, 2), "b2": s(bias ** 2 * 1e4, 2), "mse": s(mse * 1e4, 2), "mse_u": s(mse_u * 1e4, 2),
        "mse_w": s((vr + bias_w ** 2) * 1e4, 2), "pq": s(p * (1 - p), 4),
    }


def p03():
    m1, m2, m3 = 1.0, 1.5, 3.0
    var = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    sq = 0.7071
    sig3 = var * sq
    skew = mu3 / sig3
    # (b) 非心カイ二乗
    d = [1.0, 0.5, -0.5]
    k = len(d)
    lam = sum(v * v for v in d)
    return {
        "m1": s(m1, 1), "m2": s(m2, 1), "m3": s(m3, 1), "var": s(var, 1), "mu3": s(mu3, 1), "sq": s(sq, 4),
        "t2": s(3 * m1 * m2, 1), "t3": s(2 * m1 ** 3, 1),
        "sig3": s(sig3, 5), "skew": r(skew, 2), "skew_s": s(skew, 2), "skew4": s(skew, 4),
        "sk_raw": s(m3 / sig3, 2), "sk_var": s(mu3 / var, 2), "sk_no2": s((m3 - 3 * m1 * m2) / sig3, 2),
        "k": k, "d1": s(d[0], 1), "d2": s(d[1], 1), "d3": s(d[2], 1), "lam": s(lam, 1),
        "V": s(2 * (k + 2 * lam), 0), "V_c": s(2 * k, 0), "V_2": s(2 * k + 2 * lam, 0), "E": s(k + lam, 1),
        "E2": s((k + lam) ** 2, 2), "fourlam": s(4 * lam, 0),
    }


def p04():
    ln05, ln09, ln01, ln025 = -2.9957, -0.10536, -2.3026, -3.6889
    raw = ln05 / ln09
    return {
        "ln05": s(ln05, 4), "ln09": s(ln09, 5), "ln01": s(ln01, 4), "ln025": s(ln025, 4),
        "raw": s(raw, 2), "n_ok": math.ceil(raw), "n_tr": math.ceil(raw) - 1, "n_01": math.ceil(ln01 / ln09),
        "n_025": math.ceil(ln025 / ln09), "n_r3": 30, "raw01": s(ln01 / ln09, 2), "raw025": s(ln025 / ln09, 2),
    }


def p05():
    vy = 1 / 3 - 1 / 4
    cov = 1 / 4 - 1 / 2
    vx = 1.0
    cstar = -cov / vx
    vcv = vy - cov ** 2 / vx
    ratio = vcv / vy
    rho = cov / math.sqrt(vx * vy)
    wrong_sign = (vy + cstar ** 2 * vx - 2 * cstar * cov) / vy
    c1 = (vy + vx + 2 * cov) / vy
    return {
        "ratio": s(ratio, 3), "rho2": s(rho ** 2, 3), "one_m_rho": s(1 - abs(rho), 3), "wsign": s(wrong_sign, 3),
        "c1": s(c1, 3), "rho": s(rho, 3),
    }


def p06():
    n = 12
    s2 = [0.0484, 0.0225, 0.0729]
    c11u, c11l, c12u, c12l = 21.92, 3.816, 23.34, 4.404
    ss = (n - 1) * s2[0]
    lo, hi = math.sqrt(ss / c11u), math.sqrt(ss / c11l)
    t11, sq1312 = 2.201, 1.0408
    s_ = 0.22
    out = {
        "t11": s(t11, 3), "sq1312": s(sq1312, 4), "s": s(s_, 2), "pi": s(t11 * s_ * sq1312, 3), "pi4": s(t11 * s_ * sq1312, 4),
        "ts": s(t11 * s_, 4), "ci": s(t11 * s_ / 3.4641, 3), "piz": s(1.96 * s_ * sq1312, 3), "pinos": s(t11 * s_, 3),
        "piv": s(t11 * 0.0484 * sq1312, 3), "sq12": "3.4641",
        "n": n, "df": n - 1, "s2a": s(s2[0], 4), "s2b": s(s2[1], 4), "s2c": s(s2[2], 4), "sa": s(math.sqrt(s2[0]), 2),
        "c11u": s(c11u, 2), "c11l": s(c11l, 3), "c12u": s(c12u, 2), "c12l": s(c12l, 3), "ss": s(ss, 4),
        "v_lo": s(ss / c11u, 5), "v_hi": s(ss / c11l, 5),
        "lo": s(lo, 3), "hi": s(hi, 3),
        "vlo3": s(ss / c11u, 3), "vhi3": s(ss / c11l, 3),
        "nlo": s(math.sqrt(n * s2[0] / c11u), 3), "nhi": s(math.sqrt(n * s2[0] / c11l), 3),
        "dlo": s(math.sqrt(ss / c12u), 3), "dhi": s(math.sqrt(ss / c12l), 3),
    }
    # (b) バートレット検定（示した対数の値を使う）
    k = 3
    lnp, lna, lnb, lnc = -3.0379, -3.0283, -3.7942, -2.6187
    sp2 = sum(s2) / k
    sumln = lna + lnb + lnc
    M = (k * (n - 1)) * lnp - (n - 1) * sumln
    C = 1 + (1 / (3 * (k - 1))) * (k / (n - 1) - 1 / (k * (n - 1)))
    B = M / r(C, 4)
    out.update({
        "k": k, "Nk": k * (n - 1), "sp2": s(sp2, 6), "lnp": s(lnp, 4), "lna": s(lna, 4), "lnb": s(lnb, 4),
        "lnc": s(lnc, 4), "sumln": s(sumln, 4), "t1": s(k * (n - 1) * lnp, 4), "t2": s((n - 1) * sumln, 4),
        "M": s(M, 4), "C": s(r(C, 4), 4), "B": r(B, 2), "B_s": s(B, 2), "B4": s(B, 4), "B_noc": s(M, 2),
        "chi2": "5.991",
    })
    return out


def p07():
    xa, na, xb, nb = 18, 300, 15, 500
    pa, pb = xa / na, xb / nb
    asa, asb = 0.2475, 0.1741
    diff = asa - asb
    v = 1 / (4 * na) + 1 / (4 * nb)
    se = r(math.sqrt(v), 5)
    z = diff / se
    # (b) メディアン検定
    a, b, c, d = 6, 2, 2, 6
    n = a + b + c + d
    chi = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    yates = n * (abs(a * d - b * c) - n / 2) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    return {
        "xa": xa, "na": na, "xb": xb, "nb": nb, "pa": s(pa, 2), "pb": s(pb, 2), "asa": s(asa, 4), "asb": s(asb, 4),
        "diff": s(diff, 4), "v": s(v, 8), "va": s(1 / (4 * na), 8), "vb": s(1 / (4 * nb), 4), "se": s(se, 5),
        "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4),
        "a": a, "b": b, "c": c, "d": d, "nn": n, "adbc": a * d - b * c, "chi": s(chi, 2), "yates": s(yates, 2),
        "chi_n1": s((n - 1) * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d)), 2),
        "chi_half": s(chi / 2, 2), "chi_nodiv": s(4 * (a - 4) ** 2, 2),
    }


def p08():
    a, sig2, T = 0.5, 0.04, 4
    sd = math.sqrt(sig2 * T)
    z = a / sd
    phi = {1.25: 0.8944, 0.63: 0.7357, 2.50: 0.9938}
    ans = 2 * (1 - phi[r(z, 2)])
    return {
        "a": s(a, 1), "sig2": s(sig2, 2), "sig": s(math.sqrt(sig2), 1), "T": T, "sd": s(sd, 1), "z": s(z, 2),
        "phiA": s(phi[1.25], 4), "phiB": s(phi[0.63], 4), "phiC": s(phi[2.50], 4),
        "ans": s(ans, 3), "half": s(1 - phi[1.25], 3), "comp": s(1 - ans, 3),
        "wT": s(2 * (1 - phi[0.63]), 3), "wnoT": s(2 * (1 - phi[2.50]), 3), "tail4": s(1 - phi[1.25], 4),
    }


def p09():
    b0, b1, b2, knot, t = 50.0, 1.2, -1.5, 10, 16
    pred = b0 + b1 * t + b2 * (t - knot)
    # (b) てこ比の和
    n, p = 10, 3
    hsum9 = 2.30
    h10 = p - hsum9
    return {
        "b0": s(b0, 1), "b1": s(b1, 1), "b2": s(b2, 1), "knot": knot, "t": t, "x": 1100 + 10 * t,
        "xk": 1100 + 10 * knot, "tk": t - knot, "pred": s(pred, 1), "p_noplus": s(b0 + b1 * t, 1),
        "p_all": s(b0 + b1 * t + b2 * t, 1), "p_b2only": s(b0 + b1 * knot + b2 * (t - knot), 1),
        "p_sign": s(b0 + b1 * t - b2 * (t - knot), 1), "slope2": s(b1 + b2, 1), "b1t": s(b1 * t, 1),
        "b2t": s(b2 * (t - knot), 1),
        "n": n, "p": p, "hsum9": s(hsum9, 2), "h10": r(h10, 2), "h10_s": s(h10, 2), "thr": s(2 * p / n, 1),
        "q85": "1.036", "q818": "0.908", "q90": "1.282", "q95": "1.645",
    }


def p10():
    y = [3, 7, 5, 9]
    mu = [3.81, 5.00, 6.57, 8.62]                  # 速度 1,2,3,4 で当てはめた値（小数第2位まで）
    lg = [-0.23902, 0.33647, -0.27308, 0.04314]   # ln(y/mu)
    terms = [yy * l for yy, l in zip(y, lg)]
    D = 2 * sum(terms)
    pear = sum((yy - m) ** 2 / m for yy, m in zip(y, mu))
    swap = 2 * sum(-m * l for m, l in zip(mu, lg))
    out = {f"y{i + 1}": y[i] for i in range(4)}
    out.update({f"mu{i + 1}": f"{mu[i]:.2f}" for i in range(4)})
    out.update({f"l{i + 1}": f"{lg[i]:.5f}" for i in range(4)})
    out.update({f"tt{i + 1}": f"{terms[i]:.5f}" for i in range(4)})
    out.update({"sumt": s(sum(terms), 5), "D": s(D, 2), "D4": s(D, 4), "half": s(sum(terms), 2), "pear": s(pear, 2),
                "swap": s(swap, 2), "neg": s(-D, 2), "chi2": "5.991", "musum": s(sum(mu), 2)})
    return out


def p11():
    ms = {"blk": 3.0, "A": 12.0, "E1": 2.0, "B": 9.0, "AB": 1.5, "E2": 0.5}
    return {
        "msA": s(ms["A"], 1), "msE1": s(ms["E1"], 1), "msB": s(ms["B"], 1), "msAB": s(ms["AB"], 1),
        "msE2": s(ms["E2"], 1), "msblk": s(ms["blk"], 1),
        "F": s(ms["A"] / ms["E1"], 1), "F_e2": s(ms["A"] / ms["E2"], 1), "F_ab": s(ms["A"] / ms["AB"], 1),
        "F_pool": s(ms["A"] / (ms["E1"] + ms["E2"]), 1), "F_B": s(ms["B"] / ms["E2"], 1), "Fcrit": "5.14",
    }


def p12():
    y = [3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10]
    N, k = 12, 3
    n = N // k
    means = [sum(y[i::k]) / n for i in range(k)]
    mu = sum(y) / N
    vsys = sum((m - mu) ** 2 for m in means) / k
    S2 = sum((v - mu) ** 2 for v in y) / (N - 1)
    vsrs = (1 - n / N) * S2 / n
    out = {
        "ys": ", ".join(map(str, y)), "N": N, "k": k, "n": n, "m1": s(means[0], 2), "m2": s(means[1], 2),
        "m3": s(means[2], 2), "mu": s(mu, 2), "vsys": s(vsys, 3), "S2": s(S2, 4), "vsrs": s(vsrs, 3),
        "v_k1": s(sum((m - mu) ** 2 for m in means) / (k - 1), 3), "v_nofpc": s(S2 / n, 3),
        "dev1": s(means[0] - mu, 2), "dev2": s(means[1] - mu, 2), "dev3": s(means[2] - mu, 2),
        "sumsq": s(sum((m - mu) ** 2 for m in means), 4), "sumsq3": s(sum((m - mu) ** 2 for m in means), 3),
    }
    # (b) ネルソン・エーレン
    h = 1 / 8 + 2 / 6 + 1 / 3
    out.update({"H": r(h, 3), "H_s": s(h, 3), "H4": s(h, 4), "S_na": s(math.exp(-h), 3)})
    return out


def p13():
    ev = [2.4, 1.3, 0.7, 0.4]
    p = 5
    l5 = p - sum(ev)
    cum2 = (ev[0] + ev[1]) / p
    # (b) 1因子モデル
    r12, r13, r23 = 0.56, 0.48, 0.42
    l1sq = r12 * r13 / r23
    # (c) ストレス
    d = [1.0, 2.0, 2.5, 3.0, 1.5, 2.0]
    dh = [1.0, 1.8, 2.5, 3.0, 1.7, 2.0]
    num = sum((a - b) ** 2 for a, b in zip(d, dh))
    den = sum(a * a for a in d)
    return {
        "e1": s(ev[0], 1), "e2": s(ev[1], 1), "e3": s(ev[2], 1), "e4": s(ev[3], 1), "e5": s(l5, 1), "p": p,
        "cum2": s(100 * cum2, 1), "cum3": s(100 * (cum2 + ev[2] / p), 1), "cum_w": s(100 * (ev[0] + ev[1]) / sum(ev), 1),
        "cum1": s(100 * ev[0] / p, 1), "cum4": s(100 * (1 - l5 / p), 1), "sum4": s(sum(ev), 1), "top2": s(ev[0] + ev[1], 1),
        "r12": s(r12, 2), "r13": s(r13, 2), "r23": s(r23, 2), "l1sq": s(l1sq, 2), "l1": s(math.sqrt(l1sq), 1),
        "psi": s(1 - l1sq, 2), "w_l": s(1 - math.sqrt(l1sq), 2), "w_r": s(1 - r12, 2), "w_nodiv": s(1 - r12 * r13, 2),
        "prod": s(r12 * r13, 4),
        "num": s(num, 2), "den": s(den, 2), "stress": s(math.sqrt(num / den), 3), "stress4": s(math.sqrt(num / den), 4),
        "ratio": s(num / den, 5), "st_sq": s(num / den, 3), "st_raw": s(math.sqrt(num), 3),
        "st_mean": s(num / 6, 3), "st_rms": s(math.sqrt(num / 6), 3),
    }


def p14():
    a, b, c, d = 30, 12, 22, 63   # 症例：食べた・食べない、対照：食べた・食べない
    orr = a * d / (b * c)
    rr_wrong = (a / (a + c)) / (b / (b + d))
    out = {"a": a, "b": b, "c": c, "d": d, "ncase": a + b, "nctrl": c + d, "ad": a * d, "bc": b * c,
           "OR": r(orr, 2), "OR_s": s(orr, 2), "OR4": s(orr, 4), "RRw": s(rr_wrong, 2), "ORinv": s(1 / orr, 2),
           "oc": s(a / b, 3), "ot": s(c / d, 3)}
    # (b) フロントドア調整
    px1 = 0.4
    pm = {1: 0.10, 0: 0.50}
    py = {(1, 1): 0.30, (1, 0): 0.20, (0, 1): 0.06, (0, 0): 0.04}
    inner = {m: py[(m, 1)] * px1 + py[(m, 0)] * (1 - px1) for m in (0, 1)}
    do = {x: pm[x] * inner[1] + (1 - pm[x]) * inner[0] for x in (0, 1)}
    ace = do[1] - do[0]
    obs = {x: pm[x] * py[(1, x)] + (1 - pm[x]) * py[(0, x)] for x in (0, 1)}
    out.update({
        "px1": s(px1, 1), "px0": s(1 - px1, 1), "pm1": s(pm[1], 2), "pm0": s(pm[0], 2),
        "pm1c": s(1 - pm[1], 2), "pm0c": s(1 - pm[0], 2),
        "y11": s(py[(1, 1)], 2), "y10": s(py[(1, 0)], 2), "y01": s(py[(0, 1)], 2), "y00": s(py[(0, 0)], 2),
        "in1": s(inner[1], 3), "in0": s(inner[0], 3), "do1": s(do[1], 4), "do0": s(do[0], 4), "do1v": r(do[1], 3), "do1s": s(do[1], 3),
        "ace": r(ace, 3), "ace_s": s(ace, 3), "ace4": s(ace, 4),
        "obs1": s(obs[1], 3), "obs0": s(obs[0], 3), "naive": s(obs[1] - obs[0], 3),
        "t11": s(py[(1, 1)] * px1, 3), "t10": s(py[(1, 0)] * (1 - px1), 3),
        "t01": s(py[(0, 1)] * px1, 3), "t00": s(py[(0, 0)] * (1 - px1), 3),
    })
    return out


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14]
    return {f"s12p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
