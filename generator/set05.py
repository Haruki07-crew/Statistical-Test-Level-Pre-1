"""模擬試験 第5回（set05）の数値。values() が {大問id: {名前: 値}} を返す。

題材：医療・疫学・製薬・公衆衛生。
問題文に示した分布の値（Φ・χ²・q の点、e^x・log など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。-0.000 は 0.000 にする。"""
    q = Decimal(1).scaleb(-d)
    out = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def p01():
    prev = 0.02
    seA, spA, seB, spB = 0.90, 0.95, 0.80, 0.98
    # (a) 直列検査（両方陽性で陽性）。感染の有無を与えたとき2つの検査は条件付き独立
    sens = seA * seB
    fp = (1 - spA) * (1 - spB)
    num = prev * sens
    fpm = (1 - prev) * fp
    ppv = num / (num + fpm)
    ppvA = prev * seA / (prev * seA + (1 - prev) * (1 - spA))
    sens_par = 1 - (1 - seA) * (1 - seB)
    fp_par = 1 - spA * spB
    ppv_par = prev * sens_par / (prev * sens_par + (1 - prev) * fp_par)
    ppv_fpw = num / (num + (1 - prev) * fp_par)
    ppv_noprev = sens / (sens + fp)
    # (b) プール検査（ドーフマン法）。検査は誤りなし
    k, N = 10, 1000
    q10 = 0.8171  # 0.98^10（問題文に示す値）
    pools = N // k
    ppos = 1 - q10
    tests = pools * (1 + k * ppos)
    return {
        "prev": prev, "seA": seA, "spA": spA, "seB": seB, "spB": spB,
        "sens": s(sens, 2), "fpA": s(1 - spA, 2), "fpB": s(1 - spB, 2), "fp": s(fp, 3),
        "num": s(num, 4), "fpm": s(fpm, 5), "den": s(num + fpm, 5), "ppv": s(ppv, 3), "ppv4": s(ppv, 4),
        "ppvA": s(ppvA, 3), "ppv_par": s(ppv_par, 3), "ppv_fpw": s(ppv_fpw, 3), "ppv_noprev": s(ppv_noprev, 3),
        "fp_par": s(fp_par, 3),
        "k": k, "N": N, "q10": q10, "pools": pools, "ppos": s(ppos, 4), "kppos": s(k * ppos, 3),
        "tests": r(tests, 1), "tests_s": s(tests, 1), "per": s(tests / N, 4),
        "tests_wrong": s(pools * (1 + k * prev), 1),
    }


def p02():
    mu, sig, rho = 128, 20, 0.6
    # (a) 初回値と変化量の相関
    corr = -sqrt((1 - rho) / 2)
    # (b) 変化量の標本分散の上側5%点
    n = 10
    vd = 2 * (1 - rho) * sig ** 2
    c9, c10 = 16.919, 18.307  # χ²_0.05(9), χ²_0.05(10)
    cut = vd * c9 / (n - 1)
    return {
        "mu": mu, "sig": sig, "rho": rho, "var": sig ** 2, "cov": s(rho * sig ** 2, 0),
        "corr": s(corr, 3), "corr_w0": "0", "corr_w1": s(-(1 - rho) / 2, 3), "corr_w2": s(-sqrt(1 - rho), 3),
        "corr_w3": s(-corr, 3), "vd": s(vd, 0), "cxd": s((rho - 1) * sig ** 2, 0), "half": s((1 - rho) / 2, 1),
        "n": n, "df": n - 1, "c9": c9, "c10": c10,
        "cut": s(cut, 1), "cut_sig": s(sig ** 2 * c9 / (n - 1), 1), "cut_ind": s(2 * sig ** 2 * c9 / (n - 1), 1),
        "cut_n": s(vd * c9 / n, 1), "cut_df": s(vd * c10 / (n - 1), 1), "vd2": s(2 * sig ** 2, 0),
    }


def p03():
    # (a) 誤った注意報の確率を 5% 以下にする最小の基準 k（ポアソンの連続修正つき正規近似）
    lam = 49
    sd = sqrt(lam)
    z05, z025 = 1.645, 1.960  # 標準正規分布の上側 5%・2.5% 点（問題文に示す値）
    b_ok = lam + 0.5 + z05 * sd
    b_nocc = lam + z05 * sd
    k_ok = math.ceil(b_ok)
    from math import exp, factorial
    pois_sf = lambda k: 1 - sum(exp(-lam) * lam ** j / factorial(j) for j in range(k))
    # (b) コックスの部分尤度（ハザード比 0.5 を仮定）
    hr = 0.5
    pts = [("A", 2, 0, 1), ("B", 3, 1, 1), ("C", 4, 0, 0), ("D", 5, 0, 1), ("E", 7, 1, 1)]
    from fractions import Fraction as Fr
    w = lambda x: Fr(1, 2) if x else Fr(1)

    def pl(h, censor_as_event=False, keep_censored=False, keep_failed=False):
        L = Fr(1)
        for nm, t, x, e in pts:
            if not (e or censor_as_event):
                continue
            risk = [q for q in pts if q[1] >= t or (keep_failed) or (keep_censored and q[3] == 0)]
            ww = (lambda xx: h if xx else Fr(1))
            L *= ww(x) / sum(ww(q[2]) for q in risk)
        return L
    ok = pl(Fr(1, 2))
    out = {"lam": lam, "sd": s(sd, 0), "z05": z05, "z025": z025, "b_ok": s(b_ok, 3), "b_nocc": s(b_nocc, 3),
           "k_ok": k_ok, "k_nocc": math.ceil(b_nocc), "k_floor": math.floor(b_nocc),
           "k_two": math.ceil(lam + z025 * sd), "k_twocc": math.ceil(lam + 0.5 + z025 * sd),
           "ex61": s(pois_sf(61), 3), "ex62": s(pois_sf(62), 3),
           "hr": hr, "L_ok": str(ok), "L_b0": str(pl(Fr(1))), "L_keepc": str(pl(Fr(1, 2), keep_censored=True)),
           "L_cev": str(pl(Fr(1, 2), censor_as_event=True)), "L_inv": str(pl(Fr(2))),
           "L_ok_dec": s(float(ok), 4)}
    return out

def p04():
    # (a) ポアソン率の正確な信頼区間（カイ二乗分布）
    y, T = 4, 2000
    c_hi, c_lo, c_hi8, c_lo10 = 20.483, 2.180, 17.535, 3.247  # χ²_0.025(10), χ²_0.975(8), χ²_0.025(8), χ²_0.975(10)
    up = c_hi / 2 / T * 1000
    lo = c_lo / 2 / T * 1000
    # (b) 0件の日の割合だけから推定したときの漸近相対効率
    lam = 0.5
    e_p, e_m = 1.6487, 0.6065  # e^{0.5}, e^{-0.5}
    eff = lam / (e_p - 1)
    return {
        "y": y, "T": T, "df_hi": 2 * (y + 1), "df_lo": 2 * y, "c_hi": c_hi, "c_lo": c_lo, "c_hi8": c_hi8, "c_lo10": c_lo10,
        "half_hi": s(c_hi / 2, 4), "up": r(up, 2), "up4": s(up, 4), "lo": s(lo, 2), "rate": s(1000 * y / T, 1),
        "up_df8": s(c_hi8 / 2 / T * 1000, 2), "up_norm": s((y + 1.96 * sqrt(y)) / T * 1000, 2),
        "lam": lam, "ep": e_p, "em": e_m, "ep1": s(e_p - 1, 4),
        "eff": s(eff, 3), "eff4": s(eff, 4), "eff_inv": s((e_p - 1) / lam, 3), "eff_p0": s(e_m, 3),
        "eff_le": s(lam * e_m, 3), "eff_nog": s(lam / (e_m * (1 - e_m)), 3),
    }

def p05():
    # 2 群の相関係数の差のフィッシャーの z 変換による検定
    n1, r1, n2, r2 = 103, 0.5, 83, 0.3
    z1, z2 = 0.5493, 0.3095  # artanh(0.5), artanh(0.3)
    se = sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    return {"n1": n1, "r1": r1, "n2": n2, "r2": r2, "z1": z1, "z2": z2, "dz": s(z1 - z2, 4),
            "d1": n1 - 3, "d2": n2 - 3, "v": s(1 / (n1 - 3) + 1 / (n2 - 3), 4), "se": s(se, 2),
            "z": r(z, 2), "z4": s(z, 4), "z_raw": s((r1 - r2) / se, 2),
            "z_n": s((z1 - z2) / sqrt(1 / n1 + 1 / n2), 2), "z_n1": s((z1 - z2) / sqrt(1 / (n1 - 1) + 1 / (n2 - 1)), 2)}

def p06():
    # (a) 二項分布による正確な片側検定の p 値
    bn, bx, bp0 = 10, 7, 0.4
    pm = lambda j: math.comb(bn, j) * bp0 ** j * (1 - bp0) ** (bn - j)
    pv = sum(pm(j) for j in range(bx, bn + 1))
    se0 = sqrt(bp0 * (1 - bp0) / bn)
    bz = (bx / bn - bp0) / se0
    Phi = lambda z: 0.5 * (1 + math.erf(z / sqrt(2)))
    out = {"bn": bn, "bx": bx, "bp0": bp0, "t7": s(pm(7), 6), "t8": s(pm(8), 6), "t9": s(pm(9), 6), "t10": s(pm(10), 6),
           "bp4": s(pv, 4), "bp_ok": s(pv, 3), "bp_eq": s(pm(bx), 3), "bp_gt": s(pv - pm(bx), 3),
           "bp_comp": s(1 - pv, 3), "bz": s(bz, 2), "bp_norm": s(1 - Phi(bz), 3),
           "bp_cc": s(1 - Phi((bx - 0.5 - bn * bp0) / (bn * se0)), 3)}
    # (b) プロビット：用量 4 mg で 30.85%（Φ(-0.5)）、16 mg で 93.32%（Φ(1.5)）
    d1, d2, z1, z2 = 4, 16, -0.5, 1.5
    p1, p2 = 0.3085, 0.9332
    # log2 で考えると z は log2(d) の一次式：z = (log2 d - 2.5)
    l1, l2 = math.log2(d1), math.log2(d2)
    slope = (z2 - z1) / (l2 - l1)
    l50 = l1 - z1 / slope
    ed50 = 2 ** l50
    lin = d1 + (d2 - d1) * (0.5 - p1) / (p2 - p1)
    loglin = 2 ** (l1 + (l2 - l1) * (0.5 - p1) / (p2 - p1))
    out.update({
        "dd1": d1, "dd2": d2, "pp1": s(100 * p1, 2), "pp2": s(100 * p2, 2), "zz1": z1, "zz2": z2, "pr1": p1, "pr2": p2,
        "ed50": s(ed50, 2), "ed_geo": s(sqrt(d1 * d2), 2), "ed_ari": s((d1 + d2) / 2, 2),
        "ed_lin": s(lin, 2), "ed_loglin": s(loglin, 2), "l50": s(l50, 1), "rt2": s(sqrt(2), 4),
    })
    # (c) Emax モデルのラインウィーバー・バーク直線化
    a, b = 0.05, 0.40
    out.update({"ia": s(a, 2), "ib": s(b, 2), "emax": s(1 / a, 1), "ec50": s(b / a, 1),
                "ec_w1": s(b, 2), "ec_w2": s(1 / a / 2, 1), "ec_w3": s(1 / b, 1), "ec_w4": s(a / b, 3)})
    return out


def p07():
    a, b = 0.2, 0.6  # 寛解→再燃のレート、再燃→寛解のレート（1年あたり）
    t = 2
    e16, e12 = 0.2019, 0.3012  # e^{-1.6}, e^{-1.2}
    pi = a / (a + b)
    pt = pi + (1 - pi) * e16
    return {"a": a, "b": b, "t": t, "e16": e16, "e12": e12, "ab": s(a + b, 1), "abt": s((a + b) * t, 1),
            "pi": s(pi, 2), "pi1": s(1 - pi, 2), "pt": r(pt, 3), "pt4": s(pt, 4), "term": s((1 - pi) * e16, 6),
            "w_stay": s(e12, 3), "w_pi": s(pi, 3)}


def p08():
    # (a) ダミー変数
    b0, b1, b2, b3 = 2.0, 0.8, 1.5, 0.10
    out = {"b0": s(b0, 1), "b1": s(b1, 1), "b2": s(b2, 1), "b3": s(b3, 2), "b21": s(b2 - b1, 1)}
    # (c) トービットモデル（検出限界で左側打ち切り）
    mu, sig, L = 2.0, 1.0, 1.0
    Phi1, phi1 = 0.8413, 0.2420  # Φ(1), φ(1)
    a = (L - mu) / sig
    Pc = 1 - Phi1  # Φ(-1)
    Ey = L * Pc + mu * Phi1 + sig * phi1
    out.update({"mu": s(mu, 1), "sig": s(sig, 1), "L": s(L, 1), "Phi1": Phi1, "phi1": phi1, "alpha": s(a, 0),
                "Pc": s(Pc, 4), "t1": s(L * Pc, 4), "t2": s(mu * Phi1, 4), "t3": s(sig * phi1, 4),
                "Ey": s(Ey, 3), "Ey4": s(Ey, 4), "Ey_mu": s(mu, 3), "Ey_nophi": s(L * Pc + mu * Phi1, 3),
                "Ey_trunc": s(mu + sig * phi1 / Phi1, 3), "Ey_sign": s(L * Pc + mu * Phi1 - sig * phi1, 3)})
    return out

def p10():
    # 釣り合い型不完備ブロック計画
    t, k, b = 7, 3, 14
    rr = b * k // t
    lam = rr * (k - 1) // (t - 1)
    assert b * k == rr * t and lam * (t - 1) == rr * (k - 1)
    pair = lambda a, c: f"r={a},\\ \\lambda={c}"
    return {"t": t, "k": k, "b": b, "bk": b * k, "r": rr, "lam": lam, "rk1": rr * (k - 1),
            "c_ok": pair(rr, lam), "c_nok": pair(rr, rr // (t - 1)), "c_swap": pair(lam, rr),
            "c_bk": pair(b * k, b * k * (k - 1) // (t - 1)), "c_k": pair(rr, rr * (k - 1) // k)}

def p11():
    # (a) 二段抽出
    Nh, nh, m = 40, 4, 20
    M = [200, 120, 300, 80]
    cnt = [3, 2, 4, 1]
    est_i = [Mi * c / m for Mi, c in zip(M, cnt)]
    tot = Nh / nh * sum(est_i)
    out = {"Nh": Nh, "nh": nh, "m": m, "ratio": s(Nh / nh, 0), "sum_i": s(sum(est_i), 0), "tot": r(tot, 0),
           "tot_s": s(tot, 0), "w_noM": s(Nh / nh * sum(cnt), 0), "w_mean": s(sum(cnt) / (nh * m) * sum(M), 1)}
    for i in range(4):
        out.update({f"M{i}": M[i], f"c{i}": cnt[i], f"q{i}": s(cnt[i] / m, 2), f"t{i}": s(est_i[i], 0)})
    # (b) 標準化と距離
    A, B, C = (40, 22.0), (46, 22.5), (41, 25.5)
    sa, sb = 10.0, 2.5

    def dist(p, q, za=1.0, zb=1.0):
        return sqrt(((p[0] - q[0]) / za) ** 2 + ((p[1] - q[1]) / zb) ** 2)
    out.update({"Aa": A[0], "Ab": s(A[1], 1), "Ba": B[0], "Bb": s(B[1], 1), "Ca": C[0], "Cb": s(C[1], 1),
                "sa": s(sa, 0), "sb": s(sb, 1),
                "rAB": s(dist(A, B), 2), "rAC": s(dist(A, C), 2), "zAB": s(dist(A, B, sa, sb), 2), "zAC": s(dist(A, C, sa, sb), 2),
                "zAB2": s(dist(A, B, sa, sb) ** 2, 2), "zAC2": s(dist(A, C, sa, sb) ** 2, 2),
                "uAB": s((B[0] - A[0]) / sa, 1), "vAB": s((B[1] - A[1]) / sb, 1),
                "uAC": s((C[0] - A[0]) / sa, 1), "vAC": s((C[1] - A[1]) / sb, 1)})
    assert dist(A, C) < dist(A, B) and dist(A, B, sa, sb) < dist(A, C, sa, sb)
    # (c) 非計量 MDS のストレス
    d = [1.0, 1.8, 1.6, 2.6, 3.2, 2.8]
    # 単調回帰（隣り合う違反を平均でまとめる）
    blocks = [[v] for v in d]
    changed = True
    while changed:
        changed = False
        for j in range(len(blocks) - 1):
            if sum(blocks[j]) / len(blocks[j]) > sum(blocks[j + 1]) / len(blocks[j + 1]):
                blocks[j] = blocks[j] + blocks.pop(j + 1)
                changed = True
                break
    dh = [sum(bk) / len(bk) for bk in blocks for _ in bk]
    num = sum((a - b) ** 2 for a, b in zip(d, dh))
    den = sum(a * a for a in d)
    dbar = sum(d) / len(d)
    den2 = sum((a - dbar) ** 2 for a in d)
    out.update({"num": s(num, 2), "den": s(den, 2), "st": s(sqrt(num / den), 3), "st4": s(sqrt(num / den), 4),
                "st_nosq": s(num / den, 3), "st2": s(sqrt(num / den2), 3), "st_one": s(sqrt(0.02 / den), 3),
                "st_min": s(sqrt((0.2 ** 2 + 0.4 ** 2) / den), 3), "ratio_st": s(num / den, 5)})
    for j in range(6):
        out[f"d{j}"], out[f"h{j}"] = s(d[j], 1), s(dh[j], 1)
        out[f"sq{j}"] = s((d[j] - dh[j]) ** 2, 2)
        out[f"dd{j}"] = s(d[j] ** 2, 2)
    return out

def p12():
    m0, P0, q, h = 200.0, 75.0, 25.0, 60.0
    y = 240.0
    Pp = P0 + q
    K = Pp / (Pp + h)
    mt = m0 + K * (y - m0)
    Pt = (1 - K) * Pp
    K_noq = P0 / (P0 + h)
    K_inv = h / (Pp + h)
    out = {"m0": s(m0, 0), "P0": s(P0, 0), "q": s(q, 0), "h": s(h, 0), "y": s(y, 0), "Pp": s(Pp, 0),
           "K": s(K, 3), "res": s(y - m0, 0), "mt": r(mt, 1), "mt_s": s(mt, 1), "Pt": s(Pt, 1),
           "m_noq": s(m0 + K_noq * (y - m0), 1), "m_inv": s(m0 + K_inv * (y - m0), 1), "K_noq": s(K_noq, 4)}
    # (b) 1 週を除いたときの傾きの変化（DFBETA）
    x = [2, 4, 6, 8, 10]
    y = [40, 36, 30, 28, 14]

    def fit(xs, ys):
        n = len(xs)
        xb, yb = sum(xs) / n, sum(ys) / n
        sxx = sum((u - xb) ** 2 for u in xs)
        sxy = sum((u - xb) * (v - yb) for u, v in zip(xs, ys))
        return xb, yb, sxx, sxy, sxy / sxx, yb - sxy / sxx * xb
    xb, yb, sxx, sxy, bb, aa = fit(x, y)
    xb4, yb4, sxx4, sxy4, bb4, _ = fit(x[:4], y[:4])
    i = 4
    ei = y[i] - (aa + bb * x[i])
    h = 1 / len(x) + (x[i] - xb) ** 2 / sxx
    dfb = bb - bb4
    assert abs(dfb - (x[i] - xb) * ei / (sxx * (1 - h))) < 1e-12
    out.update({"xb": s(xb, 0), "yb": s(yb, 1), "sxx": s(sxx, 0), "sxy": s(sxy, 0), "bb": s(bb, 1), "aa": s(aa, 1),
                "xb4": s(xb4, 0), "yb4": s(yb4, 1), "sxx4": s(sxx4, 0), "sxy4": s(sxy4, 0), "bb4": s(bb4, 1),
                "dfb": r(dfb, 2), "dfb_s": s(dfb, 2), "ei": s(ei, 1), "hi": s(h, 1), "omh": s(1 - h, 1),
                "dxi": s(x[i] - xb, 0), "dfb_noh": s((x[i] - xb) * ei / sxx, 2), "dfb_rev": s(-dfb, 2)})
    for j in range(5):
        out[f"x{j}"], out[f"y{j}"] = x[j], y[j]
    return out

def p13():
    # (a) ベイズ因子
    n, x, p0 = 10, 5, 0.2
    comb = math.comb(n, x)
    beta_int = 1 / 2772  # ∫ p^5 (1-p)^5 dp（問題文に示す値）
    pp = p0 ** x * (1 - p0) ** (n - x)
    m0 = comb * pp
    m1 = comb * beta_int
    bf = m1 / m0
    lr_max = (0.5 ** n) / pp
    out = {"n": n, "x": x, "p0": p0, "comb": comb, "bint": 2772, "pp": s(pp, 10), "m0": s(m0, 6),
           "m1": s(m1, 6), "bf": s(bf, 2), "bf4": s(bf, 4), "bf_inv": s(1 / bf, 2),
           "bf_lr": s(lr_max, 2), "bf_one": s(1 / m0, 2), "bf_nocomb": s(beta_int / m0, 3)}
    # (b) 正規分布どうしのカルバック・ライブラー情報量
    mp, sp_, mq, sq = 130.0, 15.0, 125.0, 12.0
    lg = -0.2231  # log(12/15) = log 0.8
    kl = lg + (sp_ ** 2 + (mp - mq) ** 2) / (2 * sq ** 2) - 0.5
    kl_rev = -lg + (sq ** 2 + (mp - mq) ** 2) / (2 * sp_ ** 2) - 0.5
    out.update({"mp": s(mp, 0), "spp": s(sp_, 0), "mq": s(mq, 0), "sq": s(sq, 0), "lg": lg,
                "spsq": s(sp_ ** 2, 0), "dm2": s((mp - mq) ** 2, 0), "sq2x2": s(2 * sq ** 2, 0), "frac": s((sp_ ** 2 + (mp - mq) ** 2) / (2 * sq ** 2), 4),
                "kl": s(kl, 3), "kl4": s(kl, 4), "kl_rev": s(kl_rev, 3), "kl_nohalf": s(kl + 0.5, 3),
                "kl_nolog": s(kl - lg, 3), "kl_sign": s(kl - 2 * lg, 3)})
    # (c) BH 法
    names = ["頭痛の強さ", "睡眠の質", "不安得点", "疲労感", "食欲", "めまい", "集中力"]
    ps = [0.035, 0.002, 0.21, 0.019, 0.047, 0.009, 0.031]
    m, qq = len(ps), 0.05
    srt = sorted(ps)
    th = [(i + 1) * qq / m for i in range(m)]
    ok = [pv <= t_ for pv, t_ in zip(srt, th)]
    out.update({"m": m, "q": qq, "kbh": max(i + 1 for i in range(m) if ok[i]),
                "kstep": next(i for i in range(m) if not ok[i]), "kind": sum(ok),
                "kraw": sum(pv < qq for pv in ps), "kbon": sum(pv <= qq / m for pv in ps)})
    for i, (nm, pv) in enumerate(zip(names, ps)):
        out[f"nm{i}"] = nm
        out[f"pv{i}"] = s(pv, 3)
    for i in range(m):
        out[f"sp{i + 1}"] = s(srt[i], 3)
        out[f"th{i + 1}"] = s(th[i], 5)
    return out

def values():
    return {"s05p01": p01(), "s05p02": p02(), "s05p03": p03(), "s05p04": p04(), "s05p05": p05(),
            "s05p06": p06(), "s05p07": p07(), "s05p08": p08(), "s05p10": p10(), "s05p11": p11(),
            "s05p12": p12(), "s05p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
