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
    # (a) ポアソン分布の連続修正つき正規近似
    lam, k = 49, 60
    sd = sqrt(lam)
    phi = {0.21: 0.5832, 1.50: 0.9332, 1.57: 0.9418, 1.64: 0.9495}
    z_ok = r((k - 0.5 - lam) / sd, 2)
    z_no = r((k - lam) / sd, 2)
    z_up = r((k + 0.5 - lam) / sd, 2)
    z_var = r((k - 0.5 - lam) / lam, 2)
    for z in (z_ok, z_no, z_up, z_var):
        assert z in phi, z
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
    out = {"lam": lam, "k": k, "sd": s(sd, 0), "km": s(k - 0.5, 1), "kp": s(k + 0.5, 1),
           "zok": s(z_ok, 2), "zno_raw": s((k - lam) / sd, 4), "zno": s(z_no, 2), "zup": s(z_up, 2),
           "fA": s(phi[0.21], 4), "fB": s(phi[1.50], 4), "fC": s(phi[1.57], 4), "fD": s(phi[1.64], 4),
           "c_ok": s(1 - phi[z_ok], 3), "c_ok4": s(1 - phi[z_ok], 4), "c_no": s(1 - phi[z_no], 3), "c_up": s(1 - phi[z_up], 3),
           "c_var": s(1 - phi[z_var], 3), "c_low": s(phi[z_ok], 3),
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
    # バートレット検定
    k, n = 3, 11
    v = [4.0, 9.0, 14.0]
    ln = {4.0: 1.3863, 9.0: 2.1972, 14.0: 2.6391}
    nu = n - 1
    N = k * n
    sp2 = sum(nu * x for x in v) / (N - k)
    assert sp2 in ln
    M = (N - k) * ln[sp2] - sum(nu * ln[x] for x in v)
    C = 1 + (1 / (3 * (k - 1))) * (k / nu - 1 / (N - k))
    B = M / C
    return {"k": k, "n": n, "nu": nu, "N": N, "dfp": N - k, "v1": s(v[0], 1), "v2": s(v[1], 1), "v3": s(v[2], 1),
            "l4": s(ln[4.0], 4), "l9": s(ln[9.0], 4), "l14": s(ln[14.0], 4), "sp2": s(sp2, 1),
            "Mt1": s((N - k) * ln[sp2], 3), "Mt2": s(sum(nu * ln[x] for x in v), 3), "lsum": s(sum(ln[x] for x in v), 4),
            "M": s(M, 3), "C": s(C, 5), "Cin": s(k / nu - 1 / (N - k), 5), "B": r(B, 2), "B4": s(B, 4),
            "crit": 5.991, "M2": s(M, 2)}

def p06():
    # (a) 3 用量群の有効率の一様性を G 統計量で
    ng = 40
    resp = [10, 16, 22]
    tot = sum(resp)
    pbar = tot / (3 * ng)
    er, en = ng * pbar, ng * (1 - pbar)
    non = [ng - y for y in resp]
    ln = {10: -0.4700, 16: 0.0, 22: 0.3185, 30: 0.2231, 24: 0.0, 18: -0.2877}  # log(O/E)（問題文に示す値）
    terms_r = [y * ln[y] for y in resp]
    terms_n = [y * ln[y] for y in non]
    half = sum(terms_r) + sum(terms_n)
    G = 2 * half
    X2 = sum((y - er) ** 2 / er for y in resp) + sum((y - en) ** 2 / en for y in non)
    out = {
        "ng": ng, "r0": resp[0], "r1": resp[1], "r2": resp[2], "n0": non[0], "n1": non[1], "n2": non[2],
        "tot": tot, "pbar": s(pbar, 1), "er": s(er, 0), "en": s(en, 0),
        "lnA": s(ln[10], 4), "lnB": s(ln[22], 4), "lnC": s(ln[30], 4), "lnD": s(ln[18], 4),
        "g0": s(terms_r[0], 4), "g2": s(terms_r[2], 4), "g3": s(terms_n[0], 4), "g5": s(terms_n[2], 4),
        "half": s(half, 4), "G": s(G, 2), "G4": s(G, 4), "X2": s(X2, 2), "Ghalf": s(half, 2),
        "Grow": s(2 * sum(terms_r), 2), "Gneg": s(-G, 2), "crit": 5.991,
    }
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
    # (b) 1 人を除いたときの傾きの変化（DFBETA）
    x = [1, 2, 3, 4, 5]
    y = [3, 4, 8, 7, 13]
    n = len(x)
    xb, yb = sum(x) / n, sum(y) / n
    sxx = sum((u - xb) ** 2 for u in x)
    sxy = sum((u - xb) * (v - yb) for u, v in zip(x, y))
    bb = sxy / sxx
    aa = yb - bb * xb
    fit = [aa + bb * u for u in x]
    e = [v - f_ for v, f_ in zip(y, fit)]
    i = 4
    h = 1 / n + (x[i] - xb) ** 2 / sxx
    dfb = (x[i] - xb) * e[i] / (sxx * (1 - h))
    out = {"b0": s(b0, 1), "b1": s(b1, 1), "b2": s(b2, 1), "b3": s(b3, 2), "b21": s(b2 - b1, 1),
           "xb": s(xb, 0), "sxx": s(sxx, 0), "bb": s(bb, 1), "aa": s(aa, 1), "hi": s(h, 1), "omh": s(1 - h, 1),
           "dxi": s(x[i] - xb, 0), "ei": s(e[i], 1), "dfb": s(dfb, 2), "b_new": r(bb - dfb, 2), "b_new_s": s(bb - dfb, 2),
           "b_noh": s(bb - (x[i] - xb) * e[i] / sxx, 2), "b_plus": s(bb + dfb, 2)}
    for j in range(n):
        out.update({f"x{j}": x[j], f"y{j}": y[j], f"f{j}": s(fit[j], 1), f"e{j}": s(e[j], 1)})
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
    return {"m0": s(m0, 0), "P0": s(P0, 0), "q": s(q, 0), "h": s(h, 0), "y": s(y, 0), "Pp": s(Pp, 0),
            "K": s(K, 3), "res": s(y - m0, 0), "mt": r(mt, 1), "mt_s": s(mt, 1), "Pt": s(Pt, 1),
            "m_noq": s(m0 + K_noq * (y - m0), 1), "m_inv": s(m0 + K_inv * (y - m0), 1), "K_noq": s(K_noq, 4)}


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
    # (b) AIC と AICc
    N = 20
    mods = {"A": (3, -40.0), "B": (6, -36.2)}
    for kname, (kk, ll) in mods.items():
        aic = -2 * ll + 2 * kk
        corr = 2 * kk * (kk + 1) / (N - kk - 1)
        out.update({f"k{kname}": kk, f"l{kname}": s(ll, 1), f"m2{kname}": s(-2 * ll, 1), f"aic{kname}": s(aic, 1),
                    f"cor{kname}": s(corr, 4), f"aicc{kname}": s(aic + corr, 2), f"den{kname}": N - kk - 1,
                    f"nm{kname}": 2 * kk * (kk + 1)})
    aA, aB = -2 * mods["A"][1] + 6, -2 * mods["B"][1] + 12
    cA, cB = aA + 24 / 16, aB + 84 / 13
    assert aB < aA and cA < cB
    out.update({"N": N, "gap": s(cB - cA, 2)})
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
