"""模擬試験 第15回（set15）の数値。values() が {大問id: {名前: 値}} を返す。

題材：医療機器・臨床検査・看護・介護。
問題文に示した分布の値（F・t・χ² の点、φ(z)、e^x、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) 2 つのスクリーニング票の並列判定（どちらか陽性なら陽性）
    seA, spA, seB, spB = 0.80, 0.90, 0.70, 0.95
    se_par = 1 - (1 - seA) * (1 - seB)
    sp_par = spA * spB
    se_ser = seA * seB
    sp_ser = 1 - (1 - spA) * (1 - spB)
    # (b) 症例対照研究のオッズ比の信頼区間（ウルフの方法）
    a, b, c, d = 24, 36, 24, 96          # 症例：使用 a・不使用 b、対照：使用 c・不使用 d
    OR = a * d / (b * c)
    lnor = 0.981                         # log(8/3)（問題文で与える）
    vsum = 1 / a + 1 / b + 1 / c + 1 / d
    se_r = r(sqrt(vsum), 4)
    L, U = r(lnor - 1.96 * se_r, 3), r(lnor + 1.96 * se_r, 3)
    L90, U90 = r(lnor - 1.645 * se_r, 3), r(lnor + 1.645 * se_r, 3)
    Lv, Uv = r(lnor - 1.96 * r(vsum, 4), 3), r(lnor + 1.96 * r(vsum, 4), 3)
    ex = {0.298: 1.347, 1.664: 5.280, 0.408: 1.504, 1.554: 4.730, 0.743: 2.102, 1.219: 3.384}
    for z in (L, U, L90, U90, Lv, Uv):
        assert z in ex, z
    sym = 1.96 * OR * se_r
    return {
        "seA": s(seA, 2), "spA": s(spA, 2), "seB": s(seB, 2), "spB": s(spB, 2),
        "fnA": s(1 - seA, 2), "fnB": s(1 - seB, 2), "fn2": s((1 - seA) * (1 - seB), 2),
        "se_par": s(se_par, 3), "sp_par": s(sp_par, 3), "se_ser": s(se_ser, 3), "sp_ser": s(sp_ser, 3),
        "se_avg": s((seA + seB) / 2, 3), "sp_avg": s((spA + spB) / 2, 3),
        "a": a, "b": b, "c": c, "d": d, "na": a + b, "nc": c + d, "OR": s(OR, 3), "lnor": s(lnor, 3),
        "vsum": s(vsum, 4), "se": s(se_r, 4), "hw": s(1.96 * se_r, 4),
        "L": s(L, 3), "U": s(U, 3), "eL": s(ex[L], 2), "eU": s(ex[U], 2),
        "L90": s(L90, 3), "U90": s(U90, 3), "eL90": s(ex[L90], 2), "eU90": s(ex[U90], 2),
        "Lv": s(Lv, 3), "Uv": s(Uv, 3), "eLv": s(ex[Lv], 2), "eUv": s(ex[Uv], 2),
        "y1": s(ex[0.298], 3), "y2": s(ex[0.408], 3), "y3": s(ex[0.743], 3), "y4": s(ex[1.219], 3),
        "y5": s(ex[1.554], 3), "y6": s(ex[1.664], 3),
        "Ls": s(L, 2), "Us": s(U, 2),
        "symL": s(OR - sym, 2), "symU": s(OR + sym, 2), "symhw": s(sym, 3),
    }


def p02():
    # (a) 区分的に一定のハザード（1/月）
    h1, h2, h3 = 0.06, 0.02, 0.01
    H_1_12 = h2 * 3 + h3 * 8
    e = {0.14: 0.8694, 0.20: 0.8187, 0.12: 0.8869}
    ans = 1 - e[round(H_1_12, 2)]
    # (b) カプラン・マイヤーとグリーンウッドの公式
    rows = [(2, 10, 1), (4, 8, 2), (7, 5, 1)]   # (時刻, リスク集合, 死亡数)
    S = 1.0
    gw = 0.0
    parts = []
    for t, n, dd in rows:
        S *= (n - dd) / n
        gw += dd / (n * (n - dd))
        parts.append(dd / (n * (n - dd)))
    var = S ** 2 * gw
    return {
        "h1": s(h1, 2), "h2": s(h2, 2), "h3": s(h3, 2), "H": s(H_1_12, 2), "H3": s(h2 * 3, 2), "H8": s(h3 * 8, 2),
        "e14": s(e[0.14], 4), "e20": s(e[0.20], 4), "e12": s(e[0.12], 4),
        "ans": r(ans, 3), "ans4": s(ans, 4), "w_unc": s(1 - e[0.20], 4), "w_surv": s(e[0.14], 4),
        "w_flat": s(1 - e[0.12], 4), "H0_12": s(h1 + h2 * 3 + h3 * 8, 2),
        "S1": s(9 / 10, 3), "S2": s(9 / 10 * 6 / 8, 3), "S3": s(S, 3),
        "g1": s(parts[0], 6), "g2": s(parts[1], 6), "g3": s(parts[2], 6), "gsum": s(gw, 6),
        "S2sq": s(S ** 2, 4), "var": s(var, 6), "se": r(sqrt(var), 3), "se5": s(sqrt(var), 5),
        "w_bin": s(sqrt(S * (1 - S) / 10), 3), "w_nosq": s(sqrt(S * gw), 3),
    }


def p03():
    # (a) 対数正規分布の変動係数
    s2 = 0.20
    e02, e01 = 1.2214, 1.1052
    cv = sqrt(e02 - 1)
    # (b) 比のデルタ法
    nT, mT, sT = 20, 80.0, 16.0
    nR, mR, sR = 20, 100.0, 20.0
    R = mT / mR
    cvT2, cvR2 = (sT / mT) ** 2, (sR / mR) ** 2
    se = R * sqrt(cvT2 / nT + cvR2 / nR)
    return {
        "s2": s(s2, 2), "e02": s(e02, 4), "e01": s(e01, 4), "em1": s(e02 - 1, 4), "cv": s(cv, 3), "cv4": s(cv, 4),
        "c_sig": s(sqrt(s2), 3), "c_nosqrt": s(e02 - 1, 3), "c_half": s(sqrt(e01 - 1), 3), "c_s2": s(s2, 3),
        "nT": nT, "mT": s(mT, 0), "sT": s(sT, 0), "nR": nR, "mR": s(mR, 0), "sR": s(sR, 0), "R": s(R, 2),
        "cvT": s(sT / mT, 2), "cvR": s(sR / mR, 2), "cvT2n": s(cvT2 / nT, 4), "cvR2n": s(cvR2 / nR, 4),
        "inner": s(cvT2 / nT + cvR2 / nR, 4), "root": s(sqrt(cvT2 / nT + cvR2 / nR), 5),
        "ans": s(se, 4), "w_T": s(R * sqrt(cvT2 / nT), 4), "w_noR": s(sqrt(cvT2 / nT + cvR2 / nR), 4),
        "w_add": s(R * (sqrt(cvT2 / nT) + sqrt(cvR2 / nR)), 4), "w_sd": s(R * sqrt(cvT2 + cvR2), 4),
    }


def p04():
    nA, nB = 10, 8
    # (a) F 分布
    k = 1.5
    Ftab = {(9, 7, 2.25): 0.1488, (9, 7, 1.5): 0.3033, (7, 9, 2.25): 0.1278, (10, 8, 2.25): 0.1315}
    # (b) プールした t 区間
    mA, sA, mB, sB = 0.32, 0.20, 0.10, 0.16
    df = nA + nB - 2
    sp2 = ((nA - 1) * sA ** 2 + (nB - 1) * sB ** 2) / df
    se = sqrt(sp2 * (1 / nA + 1 / nB))
    t025, t05 = 2.120, 1.746
    diff = mA - mB

    def iv(hw):
        return s(diff - hw, 3), s(diff + hw, 3)
    ok = iv(t025 * se)
    z = iv(1.96 * se)
    one = iv(t05 * se)
    nof = iv(t025 * sqrt(sp2))
    wel = iv(t025 * sqrt(sA ** 2 / nA + sB ** 2 / nB))
    # (c) 単位の変換
    b0, b1, se1, R2 = 3.6, 0.98, 0.02, 0.97
    return {
        "nA": nA, "nB": nB, "dfA": nA - 1, "dfB": nB - 1, "k": k, "k2": s(k * k, 2),
        "F1": s(Ftab[(9, 7, 2.25)], 4), "F2": s(Ftab[(9, 7, 1.5)], 4), "F3": s(Ftab[(7, 9, 2.25)], 4),
        "F4": s(Ftab[(10, 8, 2.25)], 4), "F1c": s(1 - Ftab[(9, 7, 2.25)], 4),
        "mA": s(mA, 2), "sA": s(sA, 2), "mB": s(mB, 2), "sB": s(sB, 2), "df": df, "diff": s(diff, 2),
        "ssA": s((nA - 1) * sA ** 2, 2), "ssB": s((nB - 1) * sB ** 2, 4), "sp2": s(sp2, 4),
        "inv": s(1 / nA + 1 / nB, 3), "se2": s(sp2 * (1 / nA + 1 / nB), 7), "se": s(se, 5),
        "t025": s(t025, 3), "t05": s(t05, 3), "hw": s(t025 * se, 5),
        "okL": ok[0], "okU": ok[1], "zL": z[0], "zU": z[1], "oneL": one[0], "oneU": one[1],
        "nofL": nof[0], "nofU": nof[1], "welL": wel[0], "welU": wel[1],
        "b0": s(b0, 1), "b1": s(b1, 2), "se1": s(se1, 2), "R2": s(R2, 2), "t1": s(b1 / se1, 1),
        "b0m": s(b0 / 18, 2), "b1y": s(b1 / 18, 3), "b1x": s(b1 * 18, 2),
    }


def p05():
    # (a) 順序統計量
    n, mu = 5, 36
    e2 = mu / n + mu / (n - 1)
    # (b) 指数分布の平均の UMP 検定
    lo10, hi10, lo5 = 3.940, 18.31, 1.145
    c = mu * lo10 / 2
    return {
        "n": n, "mu": mu, "n1": n - 1, "t1": s(mu / n, 1), "t2": s(mu / (n - 1), 1), "e2": r(e2, 1),
        "e2s": s(e2, 1), "w_2n": s(2 * mu / n, 1), "w_min": s(mu / n, 1),
        "lo10": s(lo10, 3), "hi10": s(hi10, 2), "lo5": s(lo5, 3), "twon": 2 * n,
        "c": s(c, 1), "c_up": s(mu * hi10 / 2, 1), "c_no2": s(mu * lo10, 1), "c_df5": s(mu * lo5 / 2, 1),
        "c_norm": s(n * mu - 1.645 * mu * sqrt(n), 1),
    }


def p06():
    p = 0.05
    l5, l95 = -0.6931, -0.0513
    k = l5 / l95
    kk = math.ceil(k)
    return {"p": s(p, 2), "q": s(1 - p, 2), "l5": s(l5, 4), "l95": s(l95, 4), "kraw": s(k, 2),
            "ans": kk, "q13": s((1 - p) ** (kk - 1), 4), "q14": s((1 - p) ** kk, 4), "km1": kk - 1,
            "done": 20, "w_sub": kk - 20 if kk > 20 else 0, "w_tot": kk + 20}


def p07():
    # (a) 2 病棟の感染率の差の z 検定
    xA, nA, xB, nB = 14, 200, 4, 160
    pA, pB = xA / nA, xB / nB
    pp = (xA + xB) / (nA + nB)
    inv = 1 / nA + 1 / nB
    se0 = sqrt(pp * (1 - pp) * inv)
    z = (pA - pB) / se0
    se1 = sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    z1 = (pA - pB) / se1
    zbad = (pA - pB) / sqrt(pp * (1 - pp) / (nA + nB))
    assert z < 1.96 < z1 and z > 1.645
    # (b) 経験ベイズ
    y = [12, 15, 9, 14, 10]
    sig2 = 4.0
    m = len(y)
    yb = sum(y) / m
    s2u = sum((v - yb) ** 2 for v in y) / (m - 1)
    tau2 = s2u - sig2
    B = sig2 / (sig2 + tau2)
    est = yb + (1 - B) * (y[1] - yb)
    s2n = sum((v - yb) ** 2 for v in y) / m
    Bn = sig2 / s2n
    return {
        "xA": xA, "nA": nA, "xB": xB, "nB": nB, "pA": s(pA, 3), "pB": s(pB, 3), "dp": s(pA - pB, 3),
        "pp": s(pp, 3), "qq": s(1 - pp, 3), "ppq": s(pp * (1 - pp), 4), "inv": s(inv, 5), "se0": s(se0, 5),
        "z": s(z, 2), "z1": s(z1, 2), "zbad": s(zbad, 2), "se1": s(se1, 5), "rr": s(pA / pB, 1),
        "xs": xA + xB, "ns": nA + nB,
        "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3], "y5": y[4], "sig2": s(sig2, 0), "m": m,
        "yb": s(yb, 0), "ss": s(s2u * (m - 1), 0), "s2u": s(s2u, 1), "tau2": s(tau2, 1), "vtot": s(sig2 + tau2, 1),
        "B": s(B, 4), "omB": s(1 - B, 4), "dev": s(y[1] - yb, 0), "est": r(est, 2), "est4": s(est, 4),
        "w_rev": s(yb + B * (y[1] - yb), 2), "w_n": s(yb + (1 - Bn) * (y[1] - yb), 2), "s2n": s(s2n, 1),
    }


def p08():
    up = 0.4
    down = {1: 0.2, 2: 0.4, 3: 0.6}
    w = [1.0]
    for k in (1, 2, 3):
        w.append(w[-1] * up / down[k])
    tot = sum(w)
    pi = [v / tot for v in w]
    return {"up": s(up, 1), "d1": s(down[1], 1), "d2": s(down[2], 1), "d3": s(down[3], 1),
            "st0": s(1 - up, 1), "st1": s(1 - up - down[1], 1), "st2": s(1 - up - down[2], 1), "st3": s(1 - down[3], 1),
            "w1": s(w[1], 0), "w2": s(w[2], 0), "w3": "4/3", "tot": "19/3", "pi0": "3/19", "pi3f": "4/19",
            "ans": r(pi[3], 3), "ans4": s(pi[3], 4), "w_eq": s(0.25, 3)}


def p09():
    # (a) 二元配置の効果量
    SA, SB, SAB, SE = 48.0, 6.0, 4.0, 42.0
    ST = SA + SB + SAB + SE
    dfA, dfE = 2, 18
    mse = SE / dfE
    peta = SA / (SA + SE)
    # (b) ヨンクヒール・タプストラ検定
    g = [[1, 3, 4, 7], [2, 6, 8, 9], [5, 10, 11, 12]]

    def U(x, y):
        return sum(1 for a in x for b in y if a < b)
    u12, u13, u23 = U(g[0], g[1]), U(g[0], g[2]), U(g[1], g[2])
    J = u12 + u13 + u23
    ns = [len(v) for v in g]
    N = sum(ns)
    EJ = (N * N - sum(n * n for n in ns)) / 4
    VJ = (N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns)) / 72
    z = (J - EJ) / sqrt(VJ)
    out = {
        "SA": s(SA, 1), "SB": s(SB, 1), "SAB": s(SAB, 1), "SE": s(SE, 1), "ST": s(ST, 1),
        "dfA": dfA, "dfB": 1, "dfAB": 2, "dfE": dfE, "dfT": 23, "mse": s(mse, 3),
        "peta": s(peta, 3), "c_eta": s(SA / ST, 3), "c_omega": s((SA - dfA * mse) / (ST + mse), 3),
        "c_noE": s(SA / (SA + SB + SAB), 3), "c_F": s((SA / dfA) / mse, 2), "SAE": s(SA + SE, 1),
        "u12": u12, "u13": u13, "u23": u23, "J": J, "N": N, "EJ": s(EJ, 0), "VJ": s(VJ, 3),
        "sdJ": s(sqrt(VJ), 4), "z": r(z, 2), "z4": s(z, 4), "N2": N * N, "sumn2": sum(n * n for n in ns),
        "vnum": N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns), "w_noc": s((J - EJ) / VJ, 2),
    }
    for i, grp in enumerate(g):
        for j, v in enumerate(grp):
            out[f"g{i + 1}{j + 1}"] = v
    return out


def p10():
    # (a) 1 次スプライン
    b0, b1, b2, kn = 120.0, -0.50, -0.80, 65
    x = 80
    yhat = b0 + b1 * x + b2 * (x - kn)
    # (b) DFBETAS
    bb, bi, si, s_all, sxx, n = -0.90, -0.84, 8.0, 8.5, 4000, 40
    den = si / sqrt(sxx)
    dfb = (bb - bi) / den
    # (c) AIC と AICc
    mods = [("L", 300.0, 3), ("S", 295.6, 4), ("Q", 291.0, 6)]
    aic = {m: v + 2 * k for m, v, k in mods}
    aicc = {m: v + 2 * k + 2 * k * (k + 1) / (n - k - 1) for m, v, k in mods}
    assert min(aic, key=aic.get) == "Q" and min(aicc, key=aicc.get) == "S"
    out = {
        "b0": s(b0, 1), "b1": s(b1, 2), "b2": s(b2, 2), "kn": kn, "x": x, "xk": x - kn,
        "t1": s(b1 * x, 1), "t2": s(b2 * (x - kn), 1), "yhat": s(yhat, 1), "slope2": s(b1 + b2, 2),
        "c_nos": s(b0 + b1 * x, 1), "c_x": s(b0 + b1 * x + b2 * x, 1), "c_rep": s(b0 + b1 * kn + b2 * (x - kn), 1),
        "c_sign": s(b0 + b1 * x - b2 * (x - kn), 1),
        "bb": s(bb, 2), "bi": s(bi, 2), "si": s(si, 1), "sa": s(s_all, 1), "sxx": sxx, "n": n,
        "sq": s(sqrt(sxx), 3), "den": s(den, 5), "db": s(bb - bi, 2), "dfb": s(dfb, 3),
        "c_sall": s((bb - bi) / (s_all / sqrt(sxx)), 3), "c_pos": s(-dfb, 3), "c_raw": s(bb - bi, 3),
        "c_nosq": s((bb - bi) / (si / sxx), 1), "cut": s(2 / sqrt(n), 3),
    }
    for m, v, k in mods:
        out[f"{m}l"] = s(v, 1)
        out[f"{m}k"] = k
        out[f"{m}aic"] = s(aic[m], 1)
        out[f"{m}pen"] = s(2 * k * (k + 1) / (n - k - 1), 3)
        out[f"{m}aicc"] = s(aicc[m], 2)
        out[f"{m}den"] = n - k - 1
        out[f"{m}num"] = 2 * k * (k + 1)
    return out


def p11():
    b1, b2 = 0.40, 0.10
    e6, e5, sq = 1.8221, 1.6487, 1.3499
    return {"b1": s(b1, 2), "b2": s(b2, 2), "lin": s(b1 + 2 * b2, 2), "e6": s(e6, 4), "e5": s(e5, 4),
            "sq": s(sq, 4), "c_ok": s(e6, 2), "c_lin": s(1 + b1 + 2 * b2, 2), "c_one": s(e5, 2), "c_sq": s(sq, 2),
            "b12": s(b1 + b2, 2)}


def p12():
    Nf, nf = 40, 4
    M = [50, 80, 30, 60]
    x = [6, 4, 7, 5]
    m = 10
    yi = [Mi * xi / m for Mi, xi in zip(M, x)]
    tot = Nf / nf * sum(yi)
    return {"Nf": Nf, "nf": nf, "m": m, "M1": M[0], "M2": M[1], "M3": M[2], "M4": M[3],
            "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3],
            "y1": s(yi[0], 0), "y2": s(yi[1], 0), "y3": s(yi[2], 0), "y4": s(yi[3], 0), "ysum": s(sum(yi), 0),
            "fac": s(Nf / nf, 0), "ans": r(tot, 0),
            "w_ign": s(Nf / nf * sum(x), 0), "w_noN": s(sum(yi), 0),
            "w_avg": s(Nf * (sum(M) / nf) * (sum(x) / (nf * m)), 0),
            "Mbar": s(sum(M) / nf, 0), "pbar": s(sum(x) / (nf * m), 3),
            "p1": s(x[0] / m, 1), "p2": s(x[1] / m, 1), "p3": s(x[2] / m, 1), "p4": s(x[3] / m, 1)}


def p13():
    pi1, mu1, sd1 = 0.7, 5.6, 0.4
    pi2, mu2, sd2 = 0.3, 7.2, 0.8
    x = 6.4
    phi = {1.0: 0.2420, 2.0: 0.0540}
    z1, z2 = abs(x - mu1) / sd1, abs(x - mu2) / sd2
    f1, f2 = phi[round(z1, 1)] / sd1, phi[round(z2, 1)] / sd2
    w1, w2 = pi1 * f1, pi2 * f2
    g2 = w2 / (w1 + w2)
    return {"pi1": s(pi1, 1), "mu1": s(mu1, 1), "sd1": s(sd1, 1), "pi2": s(pi2, 1), "mu2": s(mu2, 1),
            "sd2": s(sd2, 1), "x": s(x, 1), "z1": s(z1, 1), "z2": s(z2, 1), "phi1": s(phi[1.0], 4), "phi2": s(phi[2.0], 4),
            "f1": s(f1, 4), "f2": s(f2, 4), "w1": s(w1, 5), "w2": s(w2, 5), "wsum": s(w1 + w2, 5),
            "ans": s(g2, 3), "ans4": s(g2, 4), "c_comp": s(1 - g2, 3),
            "c_nosd": s(pi2 * phi[1.0] / (pi2 * phi[1.0] + pi1 * phi[2.0]), 3),
            "c_nopi": s(f2 / (f1 + f2), 3), "c_none": s(phi[1.0] / (phi[1.0] + phi[2.0]), 3)}


def p14():
    l1, l2, ph = 0.50, 0.30, 0.40
    h = l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * ph
    # (b) クラスカルのストレス
    d = [1.0, 1.4, 1.2, 2.1, 3.2, 2.8]
    dh = [1.0, 1.3, 1.3, 2.1, 3.0, 3.0]
    num = sum((a - b) ** 2 for a, b in zip(d, dh))
    den = sum(a * a for a in d)
    st = sqrt(num / den)
    out = {"l1": s(l1, 2), "l2": s(l2, 2), "ph": s(ph, 2), "l1s": s(l1 ** 2, 2), "l2s": s(l2 ** 2, 2),
           "cross": s(2 * l1 * l2 * ph, 2), "h": s(h, 3), "c_orth": s(l1 ** 2 + l2 ** 2, 3),
           "c_ph2": s(l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * ph ** 2, 3), "c_u": s(1 - h, 3),
           "c_one": s((l1 + l2) ** 2, 3),
           "num": s(num, 2), "den": s(den, 2), "ratio": s(num / den, 6), "st": s(st, 3), "st4": s(st, 4),
           "c_nosqrt": s(num / den, 4), "c_num": s(num, 3), "c_rms": s(sqrt(num / len(d)), 3),
           "c_rt": s(sqrt(num), 3),
           "sq": " + ".join(s((a - b) ** 2, 2) for a, b in zip(d, dh)),
           "dsq": " + ".join(s(a * a, 2) for a in d)}
    for i in range(6):
        out[f"d{i + 1}"] = s(d[i], 1)
        out[f"h{i + 1}"] = s(dh[i], 1)
    return out


def values():
    return {"s15p01": p01(), "s15p02": p02(), "s15p03": p03(), "s15p04": p04(), "s15p05": p05(),
            "s15p06": p06(), "s15p07": p07(), "s15p08": p08(), "s15p09": p09(), "s15p10": p10(),
            "s15p11": p11(), "s15p12": p12(), "s15p13": p13(), "s15p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
