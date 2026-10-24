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
    # (a) 年齢区分 X と HbA1c の欠測 R の同時確率表
    tab = {(0, 0): 0.20, (0, 1): 0.05, (1, 0): 0.25, (1, 1): 0.15, (2, 0): 0.15, (2, 1): 0.20}
    px = {x: tab[(x, 0)] + tab[(x, 1)] for x in range(3)}
    pr1 = sum(tab[(x, 1)] for x in range(3))
    cond = tab[(2, 1)] / px[2]
    prod = px[2] * pr1
    ex = sum(x * px[x] for x in range(3))
    out = {f"t{x}{y}": s(v, 2) for (x, y), v in tab.items()}
    out.update({"px0": s(px[0], 2), "px1": s(px[1], 2), "px2": s(px[2], 2), "pr1": s(pr1, 2), "pr0": s(1 - pr1, 2),
                "cond": s(cond, 3), "prod": s(prod, 2), "c_rev": s(tab[(2, 1)] / pr1, 2), "ex": s(ex, 2),
                "c0": s(cond_x0 := tab[(0, 1)] / px[0], 2)})
    return out


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
    p = [0.05, 0.10, 0.20, 0.30, 0.35]
    v = sum(q * (1 - q) for q in p)
    pb = sum(p) / len(p)
    out = {f"p{i + 1}": s(q, 2) for i, q in enumerate(p)}
    out.update({"n": len(p), "terms": " + ".join(s(q * (1 - q), 4) for q in p), "ans": r(v, 3), "ans4": s(v, 4),
                "mean": s(sum(p), 2), "pb": s(pb, 2), "w_bin": s(len(p) * pb * (1 - pb), 3),
                "w_sum": s(sum(q * q for q in p), 4)})
    return out


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
        "w_rev": s(yb + B * (y[1] - yb), 2), "w_n": s(yb + (1 - Bn) * (y[1] - yb), 2), "s2n": s(s2n, 1), "devsq": "+".join(s((v - yb) ** 2, 0) for v in y),
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
    # (b) 符号付き順位検定の正確な p 値
    d = [12, 3, -6, 15, 8, 10]
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    rank = [0] * len(d)
    for rk, i in enumerate(order, 1):
        rank[i] = rk
    wp = sum(rk for rk, v in zip(rank, d) if v > 0)
    n = len(d)
    from itertools import product
    cnt = sum(1 for signs in product([0, 1], repeat=n) if sum(rk for rk, sg in zip(range(1, n + 1), signs) if sg) >= wp)
    p = cnt / 2 ** n
    out = {f"d{i + 1}": d[i] for i in range(n)}
    out.update({f"r{i + 1}": rank[i] for i in range(n)})
    out.update({"n": n, "wp": wp, "wm": n * (n + 1) // 2 - wp, "tot": n * (n + 1) // 2, "cnt": cnt, "N": 2 ** n,
                "p": r(p, 3), "p6": s(p, 6), "w_two": s(2 * p, 4)})
    return out


def p10():
    # (a) DFBETAS
    bb, bi, si, s_all, sxx, n = -0.90, -0.84, 8.0, 8.5, 4000, 40
    den = si / sqrt(sxx)
    dfb = (bb - bi) / den
    # (b) エラスティックネット（直交計画）
    b = [2.0, -0.6, 0.3]
    l1, l2 = 0.5, 1.0

    def soft(v, t):
        return math.copysign(max(abs(v) - t, 0.0), v)

    def fmt(v):
        return "(" + ",\\ ".join(s(x + 0.0, 2) if abs(x) > 1e-12 else "0.00" for x in v) + ")"
    en = [soft(v, l1) / (1 + l2) for v in b]
    lasso = [soft(v, l1) for v in b]
    ridge = [v / (1 + l2) for v in b]
    r_then = [soft(v / (1 + l2), l1) for v in b]
    nozero = [(1 if v > 0 else -1) * (abs(v) - l1) / (1 + l2) for v in b]
    # (c) 偏りと分散
    s2 = 9.0
    A = (2.0, 1.0)
    B = (0.5, 4.5)
    epe = lambda m: s2 + m[0] ** 2 + m[1]
    return {
        "bb": s(bb, 2), "bi": s(bi, 2), "si": s(si, 1), "sa": s(s_all, 1), "sxx": sxx, "n": n,
        "sq": s(sqrt(sxx), 3), "den": s(den, 5), "db": s(bb - bi, 2), "dfb": s(dfb, 3),
        "c_sall": s((bb - bi) / (s_all / sqrt(sxx)), 3), "c_pos": s(-dfb, 3), "c_raw": s(bb - bi, 3),
        "c_nosq": s((bb - bi) / (si / sxx), 1), "cut": s(2 / sqrt(n), 3),
        "b1": s(b[0], 1), "b2": s(b[1], 1), "b3": s(b[2], 1), "l1": s(l1, 1), "l2": s(l2, 1), "opl": s(1 + l2, 1),
        "en": fmt(en), "lasso": fmt(lasso), "ridge": fmt(ridge), "rthen": fmt(r_then), "nozero": fmt(nozero),
        "st1": s(soft(b[0], l1), 1), "st2": s(soft(b[1], l1), 1), "st3": s(soft(b[2], l1), 1),
        "s2": s(s2, 0), "bA": s(A[0], 1), "vA": s(A[1], 1), "bB": s(B[0], 1), "vB": s(B[1], 1),
        "bA2": s(A[0] ** 2, 2), "bB2": s(B[0] ** 2, 2), "eA": s(epe(A), 2), "eB": s(epe(B), 2),
        "e_nosqB": s(s2 + B[0] + B[1], 2), "e_nosqA": s(s2 + A[0] + A[1], 2),
        "e_nos2B": s(B[0] ** 2 + B[1], 2), "e_nos2A": s(A[0] ** 2 + A[1], 2), "e_rawA": s(A[0] + A[1], 2),
        "e_rawB": s(B[0] + B[1], 2), "e_novB": s(s2 + B[0] ** 2, 2), "e_novA": s(s2 + A[0] ** 2, 2),
    }


def p11():
    b1, b2 = 0.40, 0.10
    e6, e5, sq = 1.8221, 1.6487, 1.3499
    return {"b1": s(b1, 2), "b2": s(b2, 2), "lin": s(b1 + 2 * b2, 2), "e6": s(e6, 4), "e5": s(e5, 4),
            "sq": s(sq, 4), "c_ok": s(e6, 2), "c_lin": s(1 + b1 + 2 * b2, 2), "c_one": s(e5, 2), "c_sq": s(sq, 2),
            "b12": s(b1 + b2, 2)}


def p12():
    N = [3000, 1000]
    S = [10, 12]
    c = [1, 4]
    n = 200
    w = [a * b / sqrt(cc) for a, b, cc in zip(N, S, c)]
    n2 = n * w[1] / sum(w)
    ney = n * N[1] * S[1] / sum(a * b for a, b in zip(N, S))
    return {"N1": N[0], "N2": N[1], "S1": S[0], "S2": S[1], "c1": c[0], "c2": c[1], "n": n,
            "w1": s(w[0], 0), "w2": s(w[1], 0), "wsum": s(sum(w), 0), "n2raw": s(n2, 2), "ans": r(n2, 0),
            "ney": s(ney, 1), "prop": s(n * N[1] / sum(N), 0), "w_c": s(n * N[1] * S[1] / c[1] / (N[0] * S[0] / c[0] + N[1] * S[1] / c[1]), 1),
            "sc2": s(sqrt(c[1]), 0)}


def p13():
    da = [2.0, 3.0, 4.0]
    dbb = [5.0, 7.0]
    dc = [4.0, 6.0, 5.0]
    a = sum(da) / len(da)
    mb, mc = sum(dbb) / len(dbb), sum(dc) / len(dc)
    b = min(mb, mc)
    sil = (b - a) / max(a, b)
    a_self = sum(da) / (len(da) + 1)
    return {"a1": s(da[0], 1), "a2": s(da[1], 1), "a3": s(da[2], 1), "b1": s(dbb[0], 1), "b2": s(dbb[1], 1),
            "c1": s(dc[0], 1), "c2": s(dc[1], 1), "c3": s(dc[2], 1), "a": s(a, 1), "mb": s(mb, 1), "mc": s(mc, 1),
            "b": s(b, 1), "sil": s(sil, 3), "c_far": s((mb - a) / max(a, mb), 3), "c_self": s((b - a_self) / max(a_self, b), 3),
            "c_diva": s((b - a) / a, 3), "c_min": s((min(dc + dbb) - a) / max(a, min(dc + dbb)), 3), "aself": s(a_self, 2)}


def p14():
    lam = [0.8, 0.7, 0.6]
    x = [1.2, 0.4, -0.6]
    psi = [1 - l * l for l in lam]
    num = sum(l * v / p for l, v, p in zip(lam, x, psi))
    den = sum(l * l / p for l, p in zip(lam, psi))
    bart = num / den
    thom = num / (1 + den)
    unw = sum(l * v for l, v in zip(lam, x)) / sum(l * l for l in lam)
    # (b) 正準相関
    R12 = [[0.5, 0.3], [0.2, 0.4]]
    M = [[sum(R12[i][k] * R12[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    disc = tr * tr - 4 * det
    lam1 = (tr + sqrt(disc)) / 2
    out = {f"l{i + 1}": s(lam[i], 1) for i in range(3)}
    out.update({f"x{i + 1}": s(x[i], 1) for i in range(3)})
    out.update({f"psi{i + 1}": s(psi[i], 2) for i in range(3)})
    out.update({f"u{i + 1}": s(lam[i] * x[i] / psi[i], 4) for i in range(3)})
    out.update({f"v{i + 1}": s(lam[i] ** 2 / psi[i], 4) for i in range(3)})
    out.update({"num": s(num, 4), "den": s(den, 4), "bart": s(bart, 3), "bart4": s(bart, 4), "thom": s(thom, 3),
                "unw": s(unw, 3), "lx": s(sum(l * v for l, v in zip(lam, x)), 3), "mx": s(sum(x) / 3, 3), "den1": s(1 + den, 4),
                "r11": s(R12[0][0], 1), "r12": s(R12[0][1], 1), "r21": s(R12[1][0], 1), "r22": s(R12[1][1], 1),
                "m11": s(M[0][0], 2), "m12": s(M[0][1], 2), "m22": s(M[1][1], 2), "tr": s(tr, 2), "det": s(det, 4),
                "disc": s(disc, 4), "sqd": s(sqrt(disc), 5), "lam1": s(lam1, 5), "rho": r(sqrt(lam1), 3), "rho4": s(sqrt(lam1), 4),
                "w_lam": s(lam1, 3), "w_max": s(0.5, 1)})
    return out


def p04_extra():
    t, df = -3.2, 27
    pr = t / sqrt(t * t + df)
    return {"tH": s(t, 1), "dfr": df, "n30": 30, "t2": s(t * t, 2), "t2df": s(t * t + df, 2), "rt": s(sqrt(t * t + df), 4),
            "pr": s(pr, 3), "pr4": s(pr, 4), "c_df29": s(t / sqrt(t * t + 29), 3), "c_nosq": s(t / sqrt(abs(t) + df), 3),
            "c_r2": s(t * t / (t * t + df), 3), "c_nodf": s(t / sqrt(df), 3)}


def p07b():
    x, n = 2, 60
    a, b = x + 0.5, n - x + 0.5
    return {"x": x, "n": n, "a": s(a, 1), "b": s(b, 1), "ab": s(a + b, 0), "jm": r(a / (a + b), 4), "jm5": s(a / (a + b), 5),
            "w_unif": s((x + 1) / (n + 2), 4), "w_mle": s(x / n, 4), "w_half": s((x + 0.5) / (n + 0.5), 4)}


def values():
    return {"s15p01": p01(), "s15p02": p02(), "s15p03": p03(), "s15p04": {**p04(), **p04_extra()}, "s15p05": p05(),
            "s15p06": p06(), "s15p07": {**p07(), **p07b()}, "s15p08": p08(), "s15p09": p09(), "s15p10": p10(),
            "s15p11": p11(), "s15p12": p12(), "s15p13": p13(), "s15p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
