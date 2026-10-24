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
    """四捨五入した文字列（末尾の 0 も残す）。負の数は LaTeX のマイナスにしない（$ の中で使う）。"""
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
    z_var = (ph - p0) / (p0 * (1 - p0) / sqrt(n))    # 分散 p(1-p) の平方根をとらない
    # (b) ジェフリーズ事前分布
    m, y = 12, 9
    a, b = y + 0.5, m - y + 0.5
    mapv = (a - 1) / (a + b - 2)
    mean = a / (a + b)
    return {
        "n": n, "x": x, "ph": s(ph, 3), "diff": s(ph - p0, 3), "se0": s(se0, 3), "z": s(z, 2),
        "zc1": 1.645, "zc2": 1.96, "phi": s(phi180, 4), "pval": s(1 - phi180, 4), "pval2": s(2 * (1 - phi180), 4),
        "z_non": s(z_non, 2), "z_var": s(z_var, 2),
        "m": m, "y": y, "my": m - y, "a": s(a, 1), "b": s(b, 1), "am1": s(a - 1, 1), "abm2": s(a + b - 2, 0),
        "ab": s(a + b, 0),
        "map": s(mapv, 3), "mean": s(mean, 3), "mle": s(y / m, 3), "u_mean": s((y + 1) / (m + 2), 3),
        "w_half": s((y + 0.5) / m, 3), "map4": s(mapv, 4),
    }


# ---------------------------------------------------------------- 大問2 観光客の消費額
def p02():
    mu, sg = 1.8, 0.7
    ex = {1.8: 6.0496, 2.045: 7.7299, 2.15: 8.5849, 2.29: 9.8749, 2.5: 12.1825}
    half = mu + sg ** 2 / 2
    assert abs(half - 2.045) < 1e-12
    # (b) ウェルチ
    n1, m1, s1 = 10, 9.4, 3.0
    n2, m2, s2 = 16, 6.8, 1.6
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    dfi = math.floor(df)
    tt = {9: 2.262, 12: 2.179, 13: 2.160, 24: 2.064}
    assert dfi == 12 and round(df) == 12
    diff = m1 - m2
    lo = diff - tt[dfi] * se
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    se_p = sqrt(sp2 * (1 / n1 + 1 / n2))
    lo_p = diff - tt[24] * se_p
    lo_9 = diff - tt[9] * se
    lo_z = diff - 1.96 * se
    # (c) デルタ法（積）
    N, seN, X, seX = 135, 9, 7.6, 0.35
    tot = N * X                       # 千万円
    t1 = X ** 2 * seN ** 2
    t2 = N ** 2 * seX ** 2
    se_tot = sqrt(t1 + t2)            # 千万円
    se_exact = sqrt(t1 + t2 + seN ** 2 * seX ** 2)
    se_lin = (seN / N + seX / X) * tot
    return {
        "mu": mu, "sg": sg, "sg2": s(sg ** 2, 2), "half": s(sg ** 2 / 2, 3),
        "e180": s(ex[1.8], 4), "e2045": s(ex[2.045], 4), "e215": s(ex[2.15], 4), "e229": s(ex[2.29], 4), "e250": s(ex[2.5], 4),
        "c_med": s(ex[1.8], 2), "c_mean": s(ex[2.045], 2), "c_sd2": s(ex[2.15], 2), "c_nohalf": s(ex[2.29], 2),
        "c_plus": s(ex[2.5], 2),
        "n1": n1, "m1": m1, "s1": s(s1, 1), "n2": n2, "m2": m2, "s2": s(s2, 1),
        "v1": s(v1, 4), "v2": s(v2, 4), "vsum": s(v1 + v2, 4), "se": s(se, 4),
        "df": s(df, 2), "dfi": dfi, "num": s((v1 + v2) ** 2, 4), "den1": s(v1 ** 2 / (n1 - 1), 4),
        "den2": s(v2 ** 2 / (n2 - 1), 5), "den": s(v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1), 4),
        "n1m": n1 - 1, "n2m": n2 - 1, "dfp": n1 + n2 - 2,
        "t9": s(tt[9], 3), "t12": s(tt[12], 3), "t13": s(tt[13], 3), "t24": s(tt[24], 3),
        "diff": s(diff, 1), "tse": s(tt[dfi] * se, 4), "lo": r(lo, 2), "lo4": s(lo, 4), "lo_s": s(lo, 2),
        "lo_p": s(lo_p, 2), "lo_9": s(lo_9, 2), "lo_z": s(lo_z, 2), "sp2": s(sp2, 4), "se_p": s(se_p, 4),
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
    return {
        "lam": lam, "q1": sizes[1], "q2": sizes[2], "q3": sizes[3],
        "eg": s(eg, 1), "eg2": s(eg2, 1), "mean": s(mean, 2), "var": s(var, 2),
        "c_mean": s(mean, 2), "c_comp": s(lam * (eg2 - eg ** 2), 2), "c_sq": s(lam * eg ** 2, 2),
        "c_m2": s(var + mean ** 2, 2), "mean2": s(mean ** 2, 2), "vg": s(eg2 - eg ** 2, 2),
        "p1": p1, "p2": p2, "qq": s(q, 2), "pm": s(pm, 2), "et": s(et, 2),
        "c_sum": s(1 / (p1 + p2), 2), "c_add": s(1 / p1 + 1 / p2, 2), "c_max": s(1 / max(p1, p2), 2),
        "c_fail": s(q / pm, 2), "p1m": s(1 - p1, 1), "p2m": s(1 - p2, 1),
    }


# ---------------------------------------------------------------- 大問4 周遊バス
def p04():
    segs = [(12, 24), (9, 36), (10, 20)]
    D = sum(d for d, _ in segs)
    hrs = [d / v for d, v in segs]
    H = sum(hrs)
    hm = D / H
    am = sum(v for _, v in segs) / 3
    wam = sum(d * v for d, v in segs) / D
    # (b) 一様分布の推定量
    n = 8
    eff = (n + 2) / 3
    w_n3 = n / 3
    w_n13 = (n + 1) / 3
    w_raw = (n + 1) ** 2 * (n + 2) / (3 * n ** 2)
    w_mse = (n + 1) * (n + 2) / (6 * n)
    # (c) 吸収確率
    A = {"A": 0.1, "B": 0.4, "buy": 0.2, "out": 0.3}
    B = {"A": 0.3, "B": 0.1, "buy": 0.4, "out": 0.2}
    det = (1 - A["A"]) * (1 - B["B"]) - A["B"] * B["A"]
    hA = (A["buy"] * (1 - B["B"]) + A["B"] * B["buy"]) / det
    hB = (B["buy"] * (1 - A["A"]) + B["A"] * A["buy"]) / det
    two = A["buy"] + A["A"] * A["buy"] + A["B"] * B["buy"]
    return {
        "d1": segs[0][0], "v1": segs[0][1], "d2": segs[1][0], "v2": segs[1][1], "d3": segs[2][0], "v3": segs[2][1],
        "h1": s(hrs[0], 2), "h2": s(hrs[1], 2), "h3": s(hrs[2], 2), "D": D, "H": s(H, 2),
        "hm": r(hm, 1), "hm_s": s(hm, 1), "hm3": s(hm, 3), "am": s(am, 2), "wam": s(wam, 2),
        "n": n, "np2": n + 2, "np1": n + 1, "eff": s(eff, 2), "w_n3": s(w_n3, 2), "w_n13": s(w_n13, 2),
        "w_raw": s(w_raw, 2), "w_mse": s(w_mse, 2), "v2x": f"\\frac{{\\theta^2}}{{{3 * n}}}",
        "vT": f"\\frac{{\\theta^2}}{{{n * (n + 2)}}}", "n3": 3 * n, "nn2": n * (n + 2),
        "aa": A["A"], "ab": A["B"], "abuy": A["buy"], "aout": A["out"],
        "ba": B["A"], "bb": B["B"], "bbuy": B["buy"], "bout": B["out"],
        "caa": s(1 - A["A"], 1), "cbb": s(1 - B["B"], 1), "det": s(det, 2),
        "hA": s(hA, 3), "hA4": s(hA, 4), "hB": s(hB, 3), "hB4": s(hB, 4), "two": s(two, 3),
        "naive": s(A["buy"] / (A["buy"] + A["out"]), 3), "one": s(A["buy"], 3),
        "numA": s(A["buy"] * (1 - B["B"]) + A["B"] * B["buy"], 2),
    }


# ---------------------------------------------------------------- 大問5 住民調査の設計
def p05():
    N1, S1, c1 = 40000, 12, 400
    N2, S2, c2 = 10000, 18, 2500
    C = 1_000_000
    sc1, sc2 = sqrt(c1), sqrt(c2)
    w1, w2 = N1 * S1 / sc1, N2 * S2 / sc2
    k = C / (N1 * S1 * sc1 + N2 * S2 * sc2)
    n1, n2 = k * w1, k * w2
    ntot = n1 + n2
    ney2 = ntot * N2 * S2 / (N1 * S1 + N2 * S2)
    prop2 = ntot * N2 / (N1 + N2)
    # 比例配分で費用を使い切る場合
    prop_cost_n = C / (c1 * N1 / (N1 + N2) + c2 * N2 / (N1 + N2))
    # (b) 少なくとも1人
    p = 0.02
    ln05, ln98 = -2.9957, -0.0202
    nmin = math.ceil(ln05 / ln98)
    return {
        "N1": N1, "S1": S1, "c1": c1, "N2": N2, "S2": S2, "c2": c2, "C": "1{,}000{,}000", "Ctxt": "100 万",
        "sc1": int(sc1), "sc2": int(sc2), "w1": s(w1, 0), "w2": s(w2, 0),
        "f1": s(N1 * S1 * sc1, 0), "f2": s(N2 * S2 * sc2, 0), "fsum": s(N1 * S1 * sc1 + N2 * S2 * sc2, 0),
        "n1": s(n1, 1), "n2raw": s(n2, 2), "n2": r(n2, 0), "ntot": s(ntot, 1), "cost_chk": s(c1 * n1 + c2 * n2, 0),
        "ney2": s(ney2, 1), "prop2": s(prop2, 1), "prop_cost2": s(prop_cost_n * N2 / (N1 + N2), 1),
        "p": p, "ln05": s(ln05, 4), "ln98": s(ln98, 4), "ratio": s(ln05 / ln98, 2), "nmin": nmin,
        "nlow": nmin - 1, "rule3": s(3 / p, 0), "w_div": s(0.95 / p, 1), "w_inv": s(-0.0513 / ln98, 2),
        "c_rule3": int(3 / p), "c_low": nmin - 1, "c_wrong": math.ceil(0.95 / p), "c_ln95": math.ceil(-0.0513 / ln98),
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
    # (b) 欠けた残差（x=9 と x=11）
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
        "c_flip": s(-e[5], 2), "c_sum": s(se_known, 2),
    }


# ---------------------------------------------------------------- 大問7 移住意向のロジスティック回帰
def p07():
    n, l0, l1, k = 600, -398.4, -352.7, 4
    mcf = 1 - l1 / l0
    return {
        "n": n, "l0": s(l0, 1), "l1": s(l1, 1), "k": k, "ratio": s(l1 / l0, 4), "mcf": s(mcf, 3),
        "c_ratio": s(l1 / l0, 3), "c_adj": s(1 - (l1 - k) / l0, 3), "c_inv": s(l0 / l1 - 1, 3),
        "c_half": s(mcf / 2, 3), "lr": s(2 * (l1 - l0), 1), "kp1": k + 1,
    }


# ---------------------------------------------------------------- 大問8 観光需要予測モデル
def p08():
    b = [2.4, -0.9, 0.5]
    l1, l2 = 0.6, 0.5

    def soft(v, t):
        return math.copysign(max(abs(v) - t, 0.0), v)

    en = [soft(v, l1) / (1 + l2) for v in b]
    lasso = [soft(v, l1) for v in b]
    ridge = [v / (1 + l2) for v in b]
    rev = [soft(v / (1 + l2), l1) for v in b]
    two = [soft(v, l1) / (1 + 2 * l2) for v in b]

    def vec(v):
        return "(" + ",\\ ".join(s(x + 0.0, 2) if abs(x) > 1e-12 else "0" for x in v) + ")"
    # (b) バイアス・分散
    sig2 = 16.0
    bias = [3.0, 1.4, 0.55]
    var = [1.5, 4.0, 9.5]
    epe = [sig2 + bb ** 2 + vv for bb, vv in zip(bias, var)]
    return {
        "b1": b[0], "b2": b[1], "b3": b[2], "l1": l1, "l2": l2, "l2p": s(1 + l2, 1),
        "en": vec(en), "lasso": vec(lasso), "ridge": vec(ridge), "rev": vec(rev), "two": vec(two),
        "st1": s(soft(b[0], l1), 1), "st2": s(soft(b[1], l1), 1),
        "sig2": s(sig2, 0), "bi1": s(bias[0], 2), "bi2": s(bias[1], 2), "bi3": s(bias[2], 2),
        "va1": s(var[0], 1), "va2": s(var[1], 1), "va3": s(var[2], 1),
        "bs1": s(bias[0] ** 2, 2), "bs2": s(bias[1] ** 2, 2), "bs3": s(bias[2] ** 2, 4),
        "epe1": s(epe[0], 2), "epe2": s(epe[1], 2), "epe3": s(epe[2], 2),
        "w_nosq1": s(sig2 + bias[0] + var[0], 2), "w_noise2": s(bias[1] ** 2 + var[1], 2),
        "w_var1": s(sig2 + var[0], 2),
    }


# ---------------------------------------------------------------- 大問9 観光施設のリニューアル効果
def p09():
    m, up, down, tie = 12, 9, 2, 1
    n = up + down
    p_one = sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n
    p_tie = sum(comb(m, k) for k in range(up, m + 1)) / 2 ** m
    p_eq = comb(n, up) / 2 ** n
    p_strict = sum(comb(n, k) for k in range(up + 1, n + 1)) / 2 ** n
    # (b) 検出力
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
    r1 = f[0] * a[0]
    rpc2 = f[1] * a[1]
    nf, na = sqrt(f[0] ** 2 + f[1] ** 2), sqrt(a[0] ** 2 + a[1] ** 2)
    cos = r2 / (nf * na)
    wrong_sign = f[0] * a[0] - f[1] * a[1]
    return {
        **{f"{k}1": s(v[0], 2) for k, v in L.items()}, **{f"{k}2": s(v[1], 2) for k, v in L.items()},
        "lam1": s(lam1, 2), "lam2": s(lam2, 2), "cum": s((lam1 + lam2) / 4 * 100, 1),
        "p1": s(r1, 4), "p2": s(rpc2, 4), "r2": s(r2, 3), "r2_4": s(r2, 4), "c_pc1": s(r1, 3), "c_pc2": s(rpc2, 3),
        "c_cos": s(cos, 3), "c_sign": s(wrong_sign, 3), "nf": s(nf, 4), "na": s(na, 4),
        "comF": s(nf ** 2, 4), "comA": s(na ** 2, 4),
    }


# ---------------------------------------------------------------- 大問12 2地区の層別表（CMH）
def p12():
    st = [(30, 10, 45, 35), (19, 11, 25, 25)]   # a b c d
    out = {}
    sa = se = sv = 0.0
    for i, (a, b, c, d) in enumerate(st, 1):
        n = a + b + c + d
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        e = r1 * c1 / n
        v = r1 * r2 * c1 * c2 / (n ** 2 * (n - 1))
        vw = r1 * r2 * c1 * c2 / n ** 3
        out.update({f"a{i}": a, f"b{i}": b, f"c{i}": c, f"d{i}": d, f"n{i}": n, f"r1{i}": r1, f"r2{i}": r2,
                    f"k1{i}": c1, f"k2{i}": c2, f"e{i}": s(e, 4), f"v{i}": s(v, 4), f"vw{i}": s(vw, 4),
                    f"vnum{i}": r1 * r2 * c1 * c2, f"vden{i}": n ** 2 * (n - 1), f"n{i}m1": n - 1})
        sa += a
        se += e
        sv += v
        out.setdefault("_vw", 0.0)
        out["_vw"] += vw
    vw = out.pop("_vw")
    cmh = (sa - se) ** 2 / sv
    cmh_cc = (abs(sa - se) - 0.5) ** 2 / sv
    cmh_n = (sa - se) ** 2 / vw
    # 層を併合した表のカイ二乗
    A, B, C, D = (sum(x[k] for x in st) for k in range(4))
    N = A + B + C + D
    chi_pool = N * (A * D - B * C) ** 2 / ((A + B) * (C + D) * (A + C) * (B + D))
    out.update({
        "sa": sa, "se": s(se, 4), "sv": s(sv, 4), "dev": s(sa - se, 4), "dev2": s((sa - se) ** 2, 4),
        "cmh": r(cmh, 2), "cmh_s": s(cmh, 2), "cmh4": s(cmh, 4), "cmh_cc": s(cmh_cc, 2), "cmh_n": s(cmh_n, 2),
        "chi_pool": s(chi_pool, 2), "chi05": 3.841,
    })
    return out


# ---------------------------------------------------------------- 大問13 移住者の定着（ログランク）
def p13():
    A = [(6, 1), (9, 0), (11, 1), (15, 0), (18, 0), (24, 0)]
    B = [(1, 1), (3, 1), (4, 1), (8, 0), (10, 1), (13, 1)]
    ts = sorted({t for t, d in A + B if d})
    rows = []
    O = E = V = 0.0
    for t in ts:
        nA = sum(1 for u, _ in A if u >= t)
        nB = sum(1 for u, _ in B if u >= t)
        dA = sum(1 for u, d in A if u == t and d)
        dB = sum(1 for u, d in B if u == t and d)
        n, d = nA + nB, dA + dB
        e = d * nA / n
        v = d * nA * nB * (n - d) / (n ** 2 * (n - 1))
        rows.append((t, nA, nB, dA, dB, e, v))
        O += dA
        E += e
        V += v
    OB = sum(d for _, d in B)
    EB = sum(d for _, d in A + B) - E
    chi = (O - E) ** 2 / V
    chi_oe = (O - E) ** 2 / E + (OB - EB) ** 2 / EB
    out = {"OA": int(O), "EA": s(E, 4), "VA": s(V, 4), "OB": OB, "EB": s(EB, 4), "dev": s(O - E, 4),
           "dev2": s((O - E) ** 2, 4), "chi": r(chi, 2), "chi_s": s(chi, 2), "chi4": s(chi, 4),
           "chi_oe": s(chi_oe, 2), "chi05": 3.841, "nev": len(ts)}
    for i, (t, nA, nB, dA, dB, e, v) in enumerate(rows, 1):
        out.update({f"t{i}": t, f"nA{i}": nA, f"nB{i}": nB, f"dA{i}": dA, f"dB{i}": dB,
                    f"eA{i}": s(e, 4), f"vA{i}": s(v, 4)})
    return out


# ---------------------------------------------------------------- 大問14 住民の地域愛着（SEM）
def p14():
    cm, dm, cb, db = 48.6, 24, 612.0, 36
    cfi = 1 - max(cm - dm, 0) / max(cb - db, cm - dm, 0)
    nfi = 1 - cm / cb
    tli = (cb / db - cm / dm) / (cb / db - 1)
    cfi_wrong = 1 - (cm / dm) / (cb / db)
    return {
        "cm": s(cm, 1), "dm": dm, "cb": s(cb, 1), "db": db, "nm": s(cm - dm, 1), "nb": s(cb - db, 1),
        "cfi": r(cfi, 3), "cfi_s": s(cfi, 3), "cfi4": s(cfi, 4), "nfi": s(nfi, 3), "tli": s(tli, 3),
        "cfi_w": s(cfi_wrong, 3), "rb": s(cb / db, 2), "rm": s(cm / dm, 3),
    }


def values():
    return {
        "s27p01": p01(), "s27p02": p02(), "s27p03": p03(), "s27p04": p04(), "s27p05": p05(),
        "s27p06": p06(), "s27p07": p07(), "s27p08": p08(), "s27p09": p09(),
        "s27p11": p11(), "s27p12": p12(), "s27p13": p13(), "s27p14": p14(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
