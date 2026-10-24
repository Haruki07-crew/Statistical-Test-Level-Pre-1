"""模擬試験 第27回（set27）の数値。values() が {大問id: {名前: 値}} を返す。

題材：地方自治体・観光・地域経済・住民調査。
問題文に示した分布の値（Φ・t の点、e^x、ln など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
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


# ---------------------------------------------------------------- 大問1 住民意向調査
def p01():
    n, x, p0 = 400, 218, 0.5
    ph = x / n
    se0 = sqrt(p0 * (1 - p0) / n)
    z = (ph - p0) / se0
    phi180 = 0.9641
    z_non = (ph - p0) / sqrt(p0 * (1 - p0))          # √n を入れ忘れ
    z_var = (ph - p0) / (p0 * (1 - p0) / sqrt(n))    # p0(1-p0) の平方根をとらない
    # (b) 少なくとも1人
    p = 0.02
    ln05, ln98 = -2.9957, -0.0202
    ratio = ln05 / ln98
    nmin = math.ceil(ratio)
    return {
        "n": n, "x": x, "ph": s(ph, 3), "diff": s(ph - p0, 3), "se0": s(se0, 3), "z": s(z, 2),
        "zc1": 1.645, "zc2": 1.96, "phi": s(phi180, 4), "pval": s(1 - phi180, 4), "pval2": s(2 * (1 - phi180), 4),
        "z_non": s(z_non, 2), "z_var": s(z_var, 2),
        "p": p, "ln05": s(ln05, 4), "ln98": s(ln98, 4), "ratio": s(ratio, 2), "nmin": nmin, "nlow": nmin - 1,
        "rule3": int(3 / p), "w_np": math.ceil(0.95 / p), "w_ln95": math.ceil(-0.0513 / ln98),
        "pow148": s(0.98 ** (nmin - 1), 4), "pow149": s(0.98 ** nmin, 4),
    }


# ---------------------------------------------------------------- 大問2 観光客の消費額
def p02():
    # (a) 正規混合
    w1, m1, s1 = 0.7, 6, 2
    w2, m2, s2 = 0.3, 14, 4
    mean = w1 * m1 + w2 * m2
    within = w1 * s1 ** 2 + w2 * s2 ** 2
    between = w1 * w2 * (m2 - m1) ** 2
    var = within + between
    ex2 = w1 * (m1 ** 2 + s1 ** 2) + w2 * (m2 ** 2 + s2 ** 2)
    lin = w1 ** 2 * s1 ** 2 + w2 ** 2 * s2 ** 2
    # (b) 片側 t 信頼下限
    n, xb, sd = 9, 9.2, 2.4
    se = sd / sqrt(n)
    tt = {(0.05, 8): 1.860, (0.05, 9): 1.833, (0.025, 8): 2.306, (0.025, 9): 2.262}
    lo = xb - tt[(0.05, 8)] * se
    # (c) デルタ法（積）
    N, seN, X, seX = 135, 9, 7.6, 0.35
    tot = N * X                       # 千万円
    t1 = X ** 2 * seN ** 2
    t2 = N ** 2 * seX ** 2
    se_tot = sqrt(t1 + t2)            # 千万円
    se_exact = sqrt(t1 + t2 + seN ** 2 * seX ** 2)
    se_lin = (seN / N + seX / X) * tot
    return {
        "w1": w1, "w2": w2, "mu1": m1, "mu2": m2, "sd1": s1, "sd2": s2, "v1": s1 ** 2, "v2": s2 ** 2,
        "mean": s(mean, 1), "within": s(within, 2), "between": s(between, 2), "var": s(var, 2),
        "dm": m2 - m1, "dm2": (m2 - m1) ** 2, "w12": s(w1 * w2, 2), "ex2": s(ex2, 2), "mean2": s(mean ** 2, 2),
        "c_lin": s(lin, 2), "c_within": s(within, 2), "c_between": s(between, 2), "c_var": s(var, 2), "c_ex2": s(ex2, 2),
        "sdmix": s(sqrt(var), 2),
        "n": n, "xb": xb, "sd": sd, "nm1": n - 1, "se": s(se, 1), "t8": s(tt[(0.05, 8)], 3), "t9": s(tt[(0.05, 9)], 3),
        "t8b": s(tt[(0.025, 8)], 3), "t9b": s(tt[(0.025, 9)], 3), "tse": s(tt[(0.05, 8)] * se, 4),
        "lo": r(lo, 2), "lo_s": s(lo, 2), "lo4": s(lo, 4), "lo_t9": s(xb - tt[(0.05, 9)] * se, 2),
        "lo_z": s(xb - 1.645 * se, 2), "lo_two": s(xb - tt[(0.025, 8)] * se, 2), "lo_sd": s(xb - tt[(0.05, 8)] * sd, 2),
        "N": N, "seN": seN, "X": X, "seX": seX, "tot": s(tot, 1), "tot_oku": s(tot / 10, 2),
        "X2": s(X ** 2, 2), "seN2": seN ** 2, "N2": N ** 2, "seX2": s(seX ** 2, 4),
        "t1": s(t1, 2), "t2": s(t2, 4), "var_tot": s(t1 + t2, 4), "se_tot": s(se_tot, 3),
        "ans": r(se_tot / 10, 2), "ans_s": s(se_tot / 10, 2), "ans4": s(se_tot / 10, 4),
        "se_exact": s(se_exact / 10, 3), "se_lin": s(se_lin / 10, 2), "se_onlyN": s(sqrt(t1) / 10, 2),
        "se_onlyX": s(sqrt(t2) / 10, 2),
    }


# ---------------------------------------------------------------- 大問3 観光案内所
def p03():
    lam = 4
    sizes = {1: 0.5, 2: 0.3, 3: 0.2}
    eg = sum(k * p for k, p in sizes.items())
    eg2 = sum(k * k * p for k, p in sizes.items())
    mean = lam * eg
    var = lam * eg2
    # (b) 幾何分布の最小値
    p1, p2 = 0.2, 0.3
    q = (1 - p1) * (1 - p2)
    pm = 1 - q
    et = 1 / pm
    # (c) ガンマ・ポアソンの事後予測
    a0, b0 = 2, 1          # 形状, レート
    months, y = 6, 5
    a1, b1 = a0 + y, b0 + months
    pred0 = (b1 / (b1 + 1)) ** a1
    e_m1, e_m56 = 0.3679, 0.4346    # e^{-1}, e^{-5/6}
    return {
        "lam": lam, "q1": sizes[1], "q2": sizes[2], "q3": sizes[3],
        "eg": s(eg, 1), "eg2": s(eg2, 1), "mean": s(mean, 2), "var": s(var, 2),
        "c_mean": s(mean, 2), "c_comp": s(lam * (eg2 - eg ** 2), 2), "c_sq": s(lam * eg ** 2, 2),
        "c_m2": s(var + mean ** 2, 2), "mean2": s(mean ** 2, 2), "vg": s(eg2 - eg ** 2, 2),
        "p1": p1, "p2": p2, "qq": s(q, 2), "pm": s(pm, 2), "et": s(et, 2),
        "c_sum": s(1 / (p1 + p2), 2), "c_add": s(1 / p1 + 1 / p2, 2), "c_max": s(1 / max(p1, p2), 2),
        "c_fail": s(q / pm, 2), "p1m": s(1 - p1, 1), "p2m": s(1 - p2, 1),
        "a0": a0, "b0": b0, "months": months, "y": y, "a1": a1, "b1": b1, "b1p": b1 + 1,
        "pmean": s(a0 / b0, 0), "postmean": s(a1 / b1, 0),
        "pred0": r(pred0, 3), "pred0_s": s(pred0, 3), "pred0_4": s(pred0, 4), "frac": s(b1 / (b1 + 1), 3),
        "w_plug": s(e_m1, 3), "w_mle": s(e_m56, 3), "w_prior": s((b0 / (b0 + 1)) ** a0, 3),
        "w_scale": s((1 / (1 / b1 + 1)) ** a1, 3), "em1": s(e_m1, 4), "em56": s(e_m56, 4),
    }


# ---------------------------------------------------------------- 大問4 周遊バス
def p04():
    n = 8
    eff = (n + 2) / 3
    w_n3 = n / 3
    w_n13 = (n + 1) / 3
    w_raw = (n + 1) ** 2 * (n + 2) / (3 * n ** 2)
    w_mse = (n + 1) * (n + 2) / (6 * n)
    A = {"A": 0.1, "B": 0.4, "buy": 0.2, "out": 0.3}
    B = {"A": 0.3, "B": 0.1, "buy": 0.4, "out": 0.2}
    det = (1 - A["A"]) * (1 - B["B"]) - A["B"] * B["A"]
    hA = (A["buy"] * (1 - B["B"]) + A["B"] * B["buy"]) / det
    hB = (B["buy"] * (1 - A["A"]) + B["A"] * A["buy"]) / det
    two = A["buy"] + A["A"] * A["buy"] + A["B"] * B["buy"]
    return {
        "n": n, "np2": n + 2, "np1": n + 1, "eff": s(eff, 2), "w_n3": s(w_n3, 2), "w_n13": s(w_n13, 2),
        "w_raw": s(w_raw, 2), "w_mse": s(w_mse, 2), "n3": 3 * n, "nn2": n * (n + 2),
        "aa": A["A"], "ab": A["B"], "abuy": A["buy"], "aout": A["out"],
        "ba": B["A"], "bb": B["B"], "bbuy": B["buy"], "bout": B["out"],
        "caa": s(1 - A["A"], 1), "cbb": s(1 - B["B"], 1), "det": s(det, 2),
        "hA": s(hA, 3), "hA4": s(hA, 4), "hB": s(hB, 3), "hB4": s(hB, 4), "two": s(two, 3),
        "naive": s(A["buy"] / (A["buy"] + A["out"]), 3), "one": s(A["buy"], 3),
        "numA": s(A["buy"] * (1 - B["B"]) + A["B"] * B["buy"], 2),
    }


# ---------------------------------------------------------------- 大問5 宿泊施設の調査
def p05():
    Npop, n = 300, 12
    Xbar = 42.0
    ybar, xbar = 36.0, 40.0
    sy2, sx2, sxy = 225.0, 196.0, 189.0
    R = ybar / xbar
    f = n / Npop
    yr = R * Xbar
    sd2 = sy2 - 2 * R * sxy + R ** 2 * sx2
    var = (1 - f) / n * sd2
    se = sqrt(var)
    # (b) 変動係数
    mA, sA, mB, sB, add = 120, 30, 15, 6, 5
    return {
        "Npop": Npop, "n": n, "Xbar": s(Xbar, 1), "ybar": s(ybar, 1), "xbar": s(xbar, 1),
        "sy2": s(sy2, 0), "sx2": s(sx2, 0), "sxy": s(sxy, 0), "R": s(R, 2), "R2": s(R ** 2, 2), "f": s(f, 2),
        "fpc": s(1 - f, 2), "yr": s(yr, 1), "t2R": s(2 * R * sxy, 1), "tR2": s(R ** 2 * sx2, 2), "sd2": s(sd2, 2),
        "coef": s((1 - f) / n, 2), "var": s(var, 4), "se": r(se, 2), "se_s": s(se, 2), "se4": s(se, 4),
        "w_srs": s(sqrt((1 - f) / n * sy2), 2), "w_nofpc": s(sqrt(sd2 / n), 2),
        "w_sign": s(sqrt((1 - f) / n * (sy2 + 2 * R * sxy + R ** 2 * sx2)), 2),
        "mA": mA, "sA": sA, "mB": mB, "sB": sB, "cvA": s(sA / mA, 2), "cvB": s(sB / mB, 2),
        "icvA": s(mA / sA, 1), "icvB": s(mB / sB, 1), "add": add, "mBadd": mB + add, "cvBadd": s(sB / (mB + add), 2),
    }


# ---------------------------------------------------------------- 大問6 観光客数の回帰
def p06():
    x = [2, 3, 5, 6, 8, 9, 11, 12]
    a, b = 14.0, 1.6
    e = [1.2, -0.8, 0.5, -1.5, 0.9, -1.5, 1.6, -0.4]
    y = [a + b * xi + ei for xi, ei in zip(x, e)]
    n = len(x)
    xb = sum(x) / n
    yb = sum(y) / n
    Sxx = sum((xi - xb) ** 2 for xi in x)
    Sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    Syy = sum((yi - yb) ** 2 for yi in y)
    bh = Sxy / Sxx
    rss = Syy - bh * Sxy
    s2 = rss / (n - 2)
    se = sqrt(s2 / Sxx)
    t = bh / se
    t_n = bh / sqrt(rss / n / Sxx)
    t_n1 = bh / sqrt(rss / (n - 1) / Sxx)
    t_syy = bh / sqrt(Syy / (n - 2) / Sxx)
    t_nosqrt = bh / (s2 / Sxx)
    known = [(xi, ei) for xi, ei in zip(x, e) if xi not in (9, 11)]
    se_known = sum(ei for _, ei in known)
    sxe_known = sum(xi * ei for xi, ei in known)
    return {
        "n": n, "xb": s(xb, 0), "yb": s(yb, 2), "Sxx": s(Sxx, 0), "Sxy": s(Sxy, 1), "Syy": s(Syy, 2),
        "b": s(bh, 2), "bSxy": s(bh * Sxy, 2), "rss": s(rss, 2), "s2": s(s2, 4), "s2Sxx": s(s2 / Sxx, 6),
        "se": s(se, 4), "t": s(t, 2), "t_n": s(t_n, 2), "t_n1": s(t_n1, 2), "t_syy": s(t_syy, 2),
        "t_nosqrt": s(t_nosqrt, 2), "df": n - 2, "a": s(yb - bh * xb, 2),
        **{f"x{i}": xi for i, xi in enumerate(x)},
        **{f"e{i}": s(ei, 1) for i, ei in enumerate(e)},
        "sek": s(se_known, 1), "sxek": s(sxe_known, 1), "msek": s(-se_known, 1), "msxek": s(-sxe_known, 1),
        "e9": s(e[5], 2), "e11": s(e[6], 2), "c_e9": s(e[5], 2), "c_e11": s(e[6], 2), "c_half": s(-se_known / 2, 2),
        "c_flip": s(-e[5], 2), "c_sum": s(se_known, 2), "two_u": s(11 * (-se_known) - (-sxe_known), 1),
    }


# ---------------------------------------------------------------- 大問7 移住意向のロジスティック回帰
def p07():
    b0, b1, b2, b3 = -1.20, 0.85, -0.40, 0.55
    ex = {0.15: 1.1618, 0.55: 1.7333, 0.85: 2.3396, 1.00: 2.7183, 1.40: 4.0552}
    return {
        "b0": s(b0, 2), "b1": s(b1, 2), "b2": s(b2, 2), "b3": s(b3, 2), "b13": s(b1 + b3, 2),
        "e015": s(ex[0.15], 4), "e055": s(ex[0.55], 4), "e085": s(ex[0.85], 4), "e100": s(ex[1.00], 4),
        "e140": s(ex[1.40], 4),
        "or1": s(ex[round(b1 + b3, 2)], 2), "or0": s(ex[b1], 2), "or_int": s(ex[b3], 2),
        "or_all": s(ex[round(b1 + b2 + b3, 2)], 2), "or_child": s(ex[round(b2 + b3, 2)], 2),
    }


# ---------------------------------------------------------------- 大問8 観光需要の予測モデル
def p08():
    S, beta, sig2 = 50, 0.8, 4.0
    lam = sig2 / beta ** 2
    shrink = S / (S + lam)
    # (b) バイアス・分散
    s2 = 16.0
    bias = [3.0, 1.4, 0.55]
    var = [1.5, 4.0, 9.5]
    epe = [s2 + bb ** 2 + vv for bb, vv in zip(bias, var)]
    return {
        "S": S, "beta": beta, "sig2": s(sig2, 1), "beta2": s(beta ** 2, 2), "lam": s(lam, 2),
        "w_sd": s(sqrt(sig2) / beta, 2), "w_nosq": s(sig2 / beta, 2), "w_inv": s(beta ** 2 / sig2, 2),
        "w_S": s(sig2 * S / beta ** 2, 1), "bl": s(shrink * beta, 3), "shrink": s(shrink, 4), "Slam": s(S + lam, 2),
        "es2": s(s2, 0), "bi1": s(bias[0], 2), "bi2": s(bias[1], 2), "bi3": s(bias[2], 2),
        "va1": s(var[0], 1), "va2": s(var[1], 1), "va3": s(var[2], 1),
        "bs1": s(bias[0] ** 2, 2), "bs2": s(bias[1] ** 2, 2), "bs3": s(bias[2] ** 2, 4),
        "epe1": s(epe[0], 2), "epe2": s(epe[1], 2), "epe3": s(epe[2], 2),
        "w_nosq1": s(s2 + bias[0] + var[0], 2), "w_noise2": s(bias[1] ** 2 + var[1], 2),
        "w_var1": s(s2 + var[0], 2),
    }


# ---------------------------------------------------------------- 大問9 観光施設のリニューアル効果
def p09():
    m, up, down, tie = 12, 9, 2, 1
    n = up + down
    p_one = sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n
    p_tie = sum(comb(m, k) for k in range(up, m + 1)) / 2 ** m
    p_eq = comb(n, up) / 2 ** n
    p_strict = sum(comb(n, k) for k in range(up + 1, n + 1)) / 2 ** n
    mu0, mu1, sig, nn = 120, 132, 45, 81
    se = sig / sqrt(nn)
    d = (mu1 - mu0) / se
    zc = 1.645
    phi = {0.755: 0.7749, 0.44: 0.6700, 1.378: 0.9159}
    z1 = d - zc
    z2 = d - 1.96
    z3 = (mu1 - mu0) / sig - zc
    pw = phi[round(z1, 3)]
    return {
        "m": m, "up": up, "down": down, "tie": tie, "nn_": n, "two_n": 2 ** n,
        "c9": comb(n, 9), "c10": comb(n, 10), "c11": comb(n, 11), "cnum": sum(comb(n, k) for k in range(up, n + 1)),
        "p": s(p_one, 3), "p4": s(p_one, 4), "c_tie": s(p_tie, 3), "c_two": s(2 * p_one, 3), "c_eq": s(p_eq, 3),
        "c_strict": s(p_strict, 3),
        "mu0": mu0, "mu1": mu1, "sig": sig, "nsur": nn, "se": s(se, 0), "delta": mu1 - mu0, "d": s(d, 2),
        "zc": zc, "z1": s(z1, 3), "z2": s(z2, 2), "z3": s(-z3, 3),
        "phi1": s(phi[0.755], 4), "phi2": s(phi[0.44], 4), "phi3": s(phi[1.378], 4),
        "pw": s(pw, 3), "c_two_s": s(phi[0.44], 3), "c_beta": s(1 - pw, 3), "c_sig": s(1 - phi[1.378], 3),
        "c_one": "1.000",
    }


# ---------------------------------------------------------------- 大問11 地域指標の多変量解析
def p11():
    L = {"T": (0.90, 0.23), "H": (0.86, 0.21), "F": (0.81, -0.36), "A": (0.09, -0.96)}
    lam1 = sum(v[0] ** 2 for v in L.values())
    lam2 = sum(v[1] ** 2 for v in L.values())
    f, a = L["F"], L["A"]
    r2 = f[0] * a[0] + f[1] * a[1]
    nf, na = sqrt(f[0] ** 2 + f[1] ** 2), sqrt(a[0] ** 2 + a[1] ** 2)
    return {
        **{f"{k}1": s(v[0], 2) for k, v in L.items()}, **{f"{k}2": s(v[1], 2) for k, v in L.items()},
        "lam1": s(lam1, 2), "lam2": s(lam2, 2), "cum": s((lam1 + lam2) / 4 * 100, 1),
        "p1": s(f[0] * a[0], 4), "p2": s(f[1] * a[1], 4), "r2": s(r2, 3), "r2_4": s(r2, 4),
        "c_pc1": s(f[0] * a[0], 3), "c_pc2": s(f[1] * a[1], 3),
        "c_cos": s(r2 / (nf * na), 3), "c_sign": s(f[0] * a[0] - f[1] * a[1], 3),
        "comF": s(nf ** 2, 4), "comA": s(na ** 2, 4),
    }


# ---------------------------------------------------------------- 大問12 2地区の層別表（CMH）
def p12():
    st = [(30, 10, 45, 35), (19, 11, 25, 25)]   # a b c d
    out = {}
    sa = se = sv = vw = 0.0
    for i, (a, b, c, d) in enumerate(st, 1):
        n = a + b + c + d
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        e = r1 * c1 / n
        v = r1 * r2 * c1 * c2 / (n ** 2 * (n - 1))
        out.update({f"a{i}": a, f"b{i}": b, f"c{i}": c, f"d{i}": d, f"n{i}": n, f"r1{i}": r1, f"r2{i}": r2,
                    f"k1{i}": c1, f"k2{i}": c2, f"e{i}": s(e, 4), f"v{i}": s(v, 4),
                    f"vnum{i}": r1 * r2 * c1 * c2, f"vden{i}": n ** 2 * (n - 1), f"n{i}m1": n - 1})
        sa += a
        se += e
        sv += v
        vw += r1 * r2 * c1 * c2 / n ** 3
    cmh = (sa - se) ** 2 / sv
    A, B, C, D = (sum(x[k] for x in st) for k in range(4))
    N = A + B + C + D
    chi_pool = N * (A * D - B * C) ** 2 / ((A + B) * (C + D) * (A + C) * (B + D))
    out.update({
        "sa": int(sa), "se": s(se, 4), "sv": s(sv, 4), "dev": s(sa - se, 4), "dev2": s((sa - se) ** 2, 4),
        "cmh": r(cmh, 2), "cmh_s": s(cmh, 2), "cmh4": s(cmh, 4),
        "cmh_cc": s((abs(sa - se) - 0.5) ** 2 / sv, 2), "cmh_n": s((sa - se) ** 2 / vw, 2),
        "chi_pool": s(chi_pool, 2), "chi05": 3.841,
    })
    return out


# ---------------------------------------------------------------- 大問13 移住者の定着（カプラン・マイヤー）
def p13():
    data = [(2, 1), (3, 0), (5, 1), (7, 1), (8, 0), (10, 1), (12, 0), (14, 1), (18, 0), (20, 0)]
    S = 1.0
    out = {}
    med = None
    k = 0
    for t, d in data:
        if not d:
            continue
        nrisk = sum(1 for u, _ in data if u >= t)
        S *= (nrisk - 1) / nrisk
        k += 1
        out.update({f"t{k}": t, f"r{k}": nrisk, f"S{k}": s(S, 4), f"q{k}": f"\\frac{{{nrisk - 1}}}{{{nrisk}}}"})
        if med is None and S <= 0.5:
            med = t
    times = sorted(t for t, _ in data)
    naive = (times[4] + times[5]) / 2
    ev = sorted(t for t, d in data if d)
    out.update({"med": med, "naive": s(naive, 0) if naive == int(naive) else s(naive, 1),
                "ev_med": ev[(len(ev) - 1) // 2] if len(ev) % 2 else s((ev[len(ev) // 2 - 1] + ev[len(ev) // 2]) / 2, 1),
                "nev": len(ev)})
    return out


# ---------------------------------------------------------------- 大問14 住民の地域愛着（SEM）
def p14():
    cm, dm, cb, db = 48.6, 24, 612.0, 36
    cfi = 1 - max(cm - dm, 0) / max(cb - db, cm - dm, 0)
    nfi = 1 - cm / cb
    tli = (cb / db - cm / dm) / (cb / db - 1)
    return {
        "cm": s(cm, 1), "dm": dm, "cb": s(cb, 1), "db": db, "nm": s(cm - dm, 1), "nb": s(cb - db, 1),
        "cfi": r(cfi, 3), "cfi_s": s(cfi, 3), "cfi4": s(cfi, 4), "nfi": s(nfi, 3), "tli": s(tli, 3),
        "rb": s(cb / db, 2), "rm": s(cm / dm, 3),
    }


def values():
    return {
        "s27p01": p01(), "s27p02": p02(), "s27p03": p03(), "s27p04": p04(), "s27p05": p05(),
        "s27p06": p06(), "s27p07": p07(), "s27p08": p08(), "s27p09": p09(),
        "s27p11": p11(), "s27p12": p12(), "s27p13": p13(), "s27p14": p14(),
    }


if __name__ == "__main__":
    import pprint
    v = values()
    for k in ("s27p02", "s27p03", "s27p05", "s27p07", "s27p08", "s27p13", "s27p01"):
        pprint.pprint(v[k], width=200)
