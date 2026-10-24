"""模擬試験 第26回（set26）の数値。values() が {大問id: {名前: 値}} を返す。

題材：通信ネットワーク・サイバーセキュリティ・データセンター。
問題文に示した分布の値（Φ・χ²・t・F の点、対数、arctan など）は、ここでも同じ丸めた値を使って答えを計算する。
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


# 問題文に示す標準正規分布の分布関数の値
PHI = {0.68: 0.7517, 1.04: 0.8508, 1.36: 0.9131, 1.37: 0.9147, 1.94: 0.9738, 2.05: 0.9798}


def p01():
    # ブリッジ型ネットワーク：S-A(p1) S-B(p2) A-T(p3) B-T(p4) A-B(p5)
    p1, p2, p3, p4, p5 = 0.9, 0.8, 0.8, 0.9, 0.7
    up = (1 - (1 - p1) * (1 - p2)) * (1 - (1 - p3) * (1 - p4))
    down = 1 - (1 - p1 * p3) * (1 - p2 * p4)
    rel = p5 * up + (1 - p5) * down
    return {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
        "q1": s(1 - p1, 1), "q2": s(1 - p2, 1), "q3": s(1 - p3, 1), "q4": s(1 - p4, 1), "q5": s(1 - p5, 1),
        "sa": s(1 - (1 - p1) * (1 - p2), 2), "at": s(1 - (1 - p3) * (1 - p4), 2), "up": s(up, 4),
        "path1": s(p1 * p3, 2), "path2": s(p2 * p4, 2), "fail1": s(1 - p1 * p3, 2), "fail2": s(1 - p2 * p4, 2),
        "down": s(down, 4), "t_up": s(p5 * up, 5), "t_down": s((1 - p5) * down, 5),
        "rel": r(rel, 3), "rel_s": s(rel, 3), "rel5": s(rel, 5),
        "w_down": s(down, 3), "w_up": s(up, 3),
    }


def p02():
    # (a) ジャック・ベラ
    n, sk, ku = 200, 0.21, 3.30
    t1 = sk ** 2
    t2 = (ku - 3) ** 2 / 4
    jb = n / 6 * (t1 + t2)
    jb_ex = n / 6 * (sk ** 2 + ku ** 2 / 4)       # 尖度 3.30 を超過尖度とみた
    jb_no4 = n / 6 * (sk ** 2 + (ku - 3) ** 2)     # 4 で割り忘れ
    jb_no6 = n * (t1 + t2)                          # 6 で割り忘れ
    jb_24 = n / 24 * (t1 + t2)                      # 6 と 24 の取り違え
    c2 = 5.991
    assert jb < c2 and jb_no4 < c2 and jb_24 < c2 and jb_ex > c2 and jb_no6 > c2
    # (b) 対数正規分布
    med, mean, thr = 80, 100, 200
    ln125, ln25, ln2 = 0.2231, 0.9163, 0.6931
    sig2 = 2 * ln125
    sig = sqrt(sig2)
    z = r(ln25 / r(sig, 4), 2)
    z_var = r(ln25 / r(sig2, 4), 2)            # 標準偏差の代わりに分散で割った
    sig_half = sqrt(ln125)                      # σ² = ln1.25（2 倍し忘れ）
    z_half = r(ln25 / r(sig_half, 4), 2)
    z_mean = r(ln2 / r(sig, 4), 2)              # 中心を平均 100 にした
    for zz in (z, z_var, z_half, z_mean):
        assert zz in PHI, zz
    pr = 1 - PHI[z]
    return {
        "n": n, "sk": s(sk, 2), "ku": s(ku, 2), "t1": s(t1, 4), "t2": s(t2, 4), "tsum": s(t1 + t2, 4),
        "n6": s(n / 6, 3), "jb": s(jb, 2), "jb_ex": s(jb_ex, 2), "jb_no4": s(jb_no4, 2), "jb_no6": s(jb_no6, 2),
        "jb_24": s(jb_24, 2), "c2": c2, "c1": 3.841, "kum3": s(ku - 3, 2),
        "med": med, "mean": mean, "thr": thr, "ratio": s(mean / med, 2), "ln125": s(ln125, 4), "ln25": s(ln25, 4),
        "ln2": s(ln2, 4), "sig2": s(sig2, 4), "sig": s(sig, 4), "z": s(z, 2), "zraw": s(ln25 / r(sig, 4), 4),
        "pr": s(pr, 4), "pr3": s(pr, 3), "phi_z": s(PHI[z], 4),
        "w_var": s(1 - PHI[z_var], 3), "w_half": s(1 - PHI[z_half], 3), "w_mean": s(1 - PHI[z_mean], 3),
        "w_low": s(PHI[z], 3), "z_var": s(z_var, 2), "z_half": s(z_half, 2), "z_mean": s(z_mean, 2),
        "sig_half": s(sig_half, 4),
        "pA": s(PHI[0.68], 4), "pB": s(PHI[1.04], 4), "pC": s(PHI[1.37], 4), "pD": s(PHI[1.94], 4), "pE": s(PHI[2.05], 4),
    }


def p03():
    # (a) 位置母数つき指数分布
    beta, n, xmin = 2.0, 10, 23.6
    bias = beta / n
    unb = xmin - bias
    mse_mle = 2 * beta ** 2 / n ** 2
    mse_unb = beta ** 2 / n ** 2
    # (b) 一様最強力検定と検出力
    b0, m = 2.0, 10
    df = 2 * m
    c05, c10, c90, c95, c025 = 31.41, 28.41, 12.44, 10.85, 34.17   # χ²_20 の上側 5%,10%,90%,95%,2.5% 点
    crit = b0 * c05 / 2
    beta1 = 2 * crit / c90
    return {
        "beta": s(beta, 1), "n": n, "xmin": s(xmin, 1), "bias": s(bias, 1), "unb": s(unb, 1),
        "beta2": s(beta ** 2, 0), "n2": n ** 2, "var1": s(beta ** 2 / n ** 2, 2), "bias2": s(bias ** 2, 2),
        "mse_mle": s(mse_mle, 2), "mse_unb": s(mse_unb, 2), "w_unb_beta": s(xmin - beta, 1), "w_unb_plus": s(xmin + bias, 1),
        "b0": s(b0, 0), "m": m, "df": df, "c05": c05, "c10": c10, "c90": c90, "c95": c95, "c025": c025,
        "crit": s(crit, 2), "beta1": s(beta1, 2), "beta1_4": s(beta1, 4),
        "w95": s(b0 * c05 / c95, 2), "w10": s(b0 * c10 / c90, 2), "w025": s(b0 * c025 / c90, 2), "winv": s(b0 * c90 / c05, 2),
    }


def p04():
    # (b) コーシー分布の標本平均
    gam, n, thr = 0.5, 16, 1.0
    pi = 3.1416
    at = {2: 1.1071, 8: 1.4464, 32: 1.5396, 0.5: 0.4636}
    ok = 1 - 2 / pi * at[2]
    w_clt = 1 - 2 / pi * at[8]        # 尺度を γ/√n とした
    w_n = 1 - 2 / pi * at[32]         # 尺度を γ/n とした
    w_one = 1 / 2 - at[2] / pi        # 片側だけ
    w_inv = 1 - 2 / pi * at[0.5]      # x/γ を γ/x と取り違えた
    return {
        "gam": s(gam, 1), "n": n, "thr": s(thr, 0), "pi": s(pi, 4),
        "at2": s(at[2], 4), "at8": s(at[8], 4), "at32": s(at[32], 4), "at05": s(at[0.5], 4),
        "ratio": s(thr / gam, 0), "ok": s(ok, 3), "ok4": s(ok, 4), "two_over_pi": s(2 / pi * at[2], 4),
        "w_clt": s(w_clt, 3), "w_n": s(w_n, 3), "w_one": s(w_one, 3), "w_inv": s(w_inv, 3),
        "sc_clt": s(gam / sqrt(n), 3), "sc_n": s(gam / n, 4),
    }


def p05():
    lam, t = 3, 10
    xm, alpha = 0.4, 3
    # (a) 逆関数法
    u = 0.875
    x = xm * (1 - u) ** (-1 / alpha)
    x_u = xm * u ** (-1 / alpha)
    # (b) 複合ポアソン過程
    es = alpha * xm / (alpha - 1)
    es2 = alpha * xm ** 2 / (alpha - 2)
    vs = es2 - es ** 2
    lt = lam * t
    mean = lt * es
    var = lt * es2
    sd = sqrt(var)
    return {
        "lam": lam, "t": t, "xm": s(xm, 1), "alpha": alpha, "u": s(u, 3), "omu": s(1 - u, 3),
        "x": s(x, 2), "x_u": s(x_u, 3), "x_pow": s(xm * (1 - u) ** (-alpha), 1), "x_noxm": s((1 - u) ** (-1 / alpha), 2),
        "x_root": s(xm * (1 - u) ** (1 / alpha), 2), "cube": s((1 - u) ** (-1 / alpha), 0),
        "es": s(es, 1), "es2": s(es2, 2), "xm2": s(xm ** 2, 2), "vs": s(vs, 2), "es_sq": s(es ** 2, 2), "lt": lt,
        "mean": s(mean, 0), "var": s(var, 1), "sd": r(sd, 2), "sd4": s(sd, 4),
        "w_vs": s(sqrt(lt * vs), 2), "w_vsvar": s(lt * vs, 1), "w_meanvar": s(sqrt(lt * es ** 2), 2),
    }


def p06():
    sizes = [60, 120, 180]
    speeds = [1.2, 2.0, 3.0]
    times = [a / b for a, b in zip(sizes, speeds)]
    tot, ttot = sum(sizes), sum(times)
    hm = tot / ttot
    am = sum(speeds) / 3
    wam = sum(a * b for a, b in zip(sizes, speeds)) / tot
    uhm = 3 / sum(1 / b for b in speeds)
    return {
        "a1": sizes[0], "a2": sizes[1], "a3": sizes[2], "v1": s(speeds[0], 1), "v2": s(speeds[1], 1), "v3": s(speeds[2], 1),
        "t1": s(times[0], 0), "t2": s(times[1], 0), "t3": s(times[2], 0), "tot": tot, "ttot": s(ttot, 0),
        "hm": r(hm, 2), "hm4": s(hm, 4), "am": s(am, 2), "wam": s(wam, 2), "uhm": s(uhm, 2),
    }


def p07():
    L = [3.0, 3.3, 3.6, 3.9, 4.2, 4.5]
    M = [3.8, 4.0, 4.4, 4.6, 5.0, 5.2]
    allv = sorted(L + M)
    FL = lambda x: sum(v <= x for v in L) / 6
    FM = lambda x: sum(v <= x for v in M) / 6
    D = max(abs(FL(x) - FM(x)) for x in allv)
    D_M = max(abs(FL(x) - FM(x)) for x in M)
    D_neg = max(FM(x) - FL(x) for x in allv)
    med_diff = (M[2] + M[3]) / 2 - (L[2] + L[3]) / 2
    # (b) プールした分散による t 区間
    n1 = n2 = 6
    mL, mM = sum(L) / 6, sum(M) / 6
    ssL = sum((v - mL) ** 2 for v in L)
    ssM = sum((v - mM) ** 2 for v in M)
    vL, vM = ssL / 5, ssM / 5
    sp2 = (ssL + ssM) / (n1 + n2 - 2)
    se = sqrt(sp2 * (1 / n1 + 1 / n2))
    t10, t5, z = 2.228, 2.571, 1.960
    diff = mM - mL
    ci = lambda hw: f"({s(diff - hw, 2)},\\ {s(diff + hw, 2)})"
    sp2n = (ssL + ssM) / (n1 + n2)
    return {
        "L": ",\\ ".join(s(v, 1) for v in L), "M": ",\\ ".join(s(v, 1) for v in M),
        "D": s(D, 2), "D_M": s(D_M, 2), "D_neg": s(D_neg, 2), "D_cnt": s(D * 6, 0), "med_diff": s(med_diff, 2),
        "dcrit": "5/6", "dcrit_d": s(5 / 6, 3), "pval": s(0.474025974, 3),
        "mL": s(mL, 2), "mM": s(mM, 2), "vL": s(vL, 3), "vM": s(vM, 3), "ssL": s(ssL, 3), "ssM": s(ssM, 3),
        "sp2": s(sp2, 4), "inv": s(1 / n1 + 1 / n2, 4), "se2": s(sp2 * (1 / n1 + 1 / n2), 4), "se": s(se, 4),
        "t10": t10, "t5": t5, "z": s(z, 3), "diff": s(diff, 2), "hw": s(t10 * se, 4),
        "ci": ci(t10 * se), "ci_z": ci(z * se), "ci_t5": ci(t5 * se), "ci_nof": ci(t10 * sqrt(sp2)),
        "ci_n": ci(t10 * sqrt(sp2n * (1 / n1 + 1 / n2))), "sp2n": s(sp2n, 4),
        "tab_v": " | ".join(s(v, 1) for v in allv),
        "tab_L": " | ".join(str(round(FL(v) * 6)) for v in allv),
        "tab_M": " | ".join(str(round(FM(v) * 6)) for v in allv),
        "tab_d": " | ".join(str(round(abs(FL(v) - FM(v)) * 6)) for v in allv),
    }


def p08():
    # (a) VIF
    r2 = {1: 0.92, 2: 0.60, 3: 0.88}
    vif1 = 1 / (1 - r2[1])
    # (b) 区分線形スプライン
    b0, b1, b2, k, sla = 20, 0.5, 3.0, 70, 120
    yk = b0 + b1 * k
    xs = k + (sla - yk) / (b1 + b2)
    xs_b2 = k + (sla - yk) / b2
    xs_lin = (sla - b0) / b1
    # (c) PRESS
    x = [10, 20, 30, 40, 50, 60]
    y = [42.0, 47.5, 58.0, 60.5, 71.0, 72.0]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    bb = sxy / sxx
    aa = my - bb * mx
    e = [b - (aa + bb * a) for a, b in zip(x, y)]
    h = [1 / n + (a - mx) ** 2 / sxx for a in x]
    e_r = [r(v, 2) for v in e]
    h_r = [r(v, 3) for v in h]
    pr = [ev / (1 - hv) for ev, hv in zip(e_r, h_r)]
    press = sum(v ** 2 for v in pr)
    sse = sum(v ** 2 for v in e_r)
    w_h = sum((ev * (1 - hv)) ** 2 for ev, hv in zip(e_r, h_r))
    w_nosq = sum((ev / (1 - hv) ** 2) ** 2 for ev, hv in zip(e_r, h_r))
    out = {
        "r1": s(r2[1], 2), "r2": s(r2[2], 2), "r3": s(r2[3], 2), "n": 40,
        "vif1": s(vif1, 1), "sevif": s(sqrt(vif1), 2), "w_inv": s(1 / r2[1], 2), "w_invse": s(sqrt(1 / r2[1]), 2),
        "w_1mr": s(1 - r2[1], 2), "w_1mrse": s(sqrt(1 - r2[1]), 2), "w_r": s(1 / (1 - r2[1] ** 2), 2),
        "w_rse": s(sqrt(1 / (1 - r2[1] ** 2)), 2), "omr": s(1 - r2[1], 2),
        "vif2": s(1 / (1 - r2[2]), 1), "vif3": s(1 / (1 - r2[3]), 2),
        "b0": b0, "b1": s(b1, 1), "b2": s(b2, 1), "k": k, "sla": sla, "yk": s(yk, 0), "slope": s(b1 + b2, 1),
        "gap": s(sla - yk, 0), "xs": r(xs, 1), "xs4": s(xs, 3), "xs_b2": s(xs_b2, 1), "xs_lin": s(xs_lin, 0),
        "press": r(press, 1), "press1": s(press, 1), "press_s": s(press, 2), "sse": s(sse, 2), "w_h": s(w_h, 2), "w_nosq": s(w_nosq, 2),
        "a_hat": s(aa, 2), "b_hat": s(bb, 2),
    }
    for i in range(n):
        out[f"x{i}"] = x[i]
        out[f"e{i}"] = s(e_r[i], 2)
        out[f"h{i}"] = s(h_r[i], 3)
        out[f"omh{i}"] = s(1 - h_r[i], 3)
        out[f"pr{i}"] = s(pr[i], 4)
        out[f"pq{i}"] = s(pr[i] ** 2, 4)
    out["_e"] = e_r
    out["_h"] = h_r
    out["_y"] = y
    return out


def p09():
    y = [12, 7, 20, 9]
    mu = [10.0, 9.0, 18.0, 11.0]
    assert sum(y) == sum(mu)
    lnr = {0: 0.1823, 1: -0.2513, 2: 0.1054, 3: -0.2007}   # ln(y/μ) を小数第4位に丸めた表の値
    terms = [y[i] * lnr[i] for i in range(4)]
    D = 2 * sum(terms)
    X2 = sum((a - b) ** 2 / b for a, b in zip(y, mu))
    out = {"D": s(D, 2), "D4": s(D, 4), "half": s(D / 2, 2), "X2": s(X2, 2), "negD": s(-D, 2), "df": 2, "c2": 5.991, "c3": 7.815,
           "sumy": sum(y), "summu": s(sum(mu), 0)}
    for i in range(4):
        out[f"y{i}"] = y[i]
        out[f"mu{i}"] = s(mu[i], 1)
        out[f"l{i}"] = s(lnr[i], 4)
        out[f"tm{i}"] = s(terms[i], 4)
    out["tsum"] = s(sum(terms), 4)
    return out


def p10():
    # 時刻（月）と事象（1=故障, 0=打ち切り）
    data = [(5, 1), (8, 0), (12, 1), (12, 1), (15, 0), (20, 1), (24, 0), (30, 0)]
    steps = [(8, 1), (6, 2), (3, 1)]   # (リスク集合の大きさ, 故障数) 時刻 5, 12, 20
    S = 1.0
    g = 0.0
    for nn, d in steps:
        S *= (nn - d) / nn
        g += d / (nn * (nn - d))
    var = S ** 2 * g
    se = sqrt(var)
    w_binom = sqrt(S * (1 - S) / 8)
    return {
        "S": s(S, 4), "S2": s(S ** 2, 4), "g1": s(1 / (8 * 7), 4), "g2": s(2 / (6 * 4), 4), "g3": s(1 / (3 * 2), 4),
        "g": s(g, 4), "var": s(var, 4), "se": r(se, 3), "se4": s(se, 4), "w_binom": s(w_binom, 3),
        "w_var": s(var, 3), "w_noS": s(sqrt(g), 3),
        "_data": data,
    }


def p11():
    ssR, ssA, ssE1, ssB, ssAB, ssE2 = 18.0, 42.0, 9.0, 12.0, 4.0, 9.0
    dfR, dfA, dfE1, dfB, dfAB, dfE2 = 3, 2, 6, 1, 2, 9
    msA = ssA / dfA
    F = msA / (ssE1 / dfE1)
    F2 = msA / (ssE2 / dfE2)
    Fp = msA / ((ssE1 + ssE2) / (dfE1 + dfE2))
    return {
        "ssR": s(ssR, 1), "ssA": s(ssA, 1), "ssE1": s(ssE1, 1), "ssB": s(ssB, 1), "ssAB": s(ssAB, 1), "ssE2": s(ssE2, 1),
        "ssT": s(ssR + ssA + ssE1 + ssB + ssAB + ssE2, 1),
        "dfR": dfR, "dfA": dfA, "dfE1": dfE1, "dfB": dfB, "dfAB": dfAB, "dfE2": dfE2, "dfT": 23, "dfEp": dfE1 + dfE2,
        "msA": s(msA, 1), "msE1": s(ssE1 / dfE1, 1), "msE2": s(ssE2 / dfE2, 1), "msEp": s((ssE1 + ssE2) / 15, 1),
        "F": s(F, 1), "F2": s(F2, 1), "Fp": s(Fp, 1), "Fraw": s(ssA / ssE1, 2),
        "f26": 5.14, "f29": 4.26, "f215": 3.68,
    }


def p12():
    N = [400, 1200, 400]
    S = [6, 3, 10]
    c = [4, 1, 9]
    C = 1020
    rc = [sqrt(v) for v in c]
    a = [N[i] * S[i] / rc[i] for i in range(3)]
    b = [N[i] * S[i] * rc[i] for i in range(3)]
    nh = [C * a[i] / sum(b) for i in range(3)]
    cost = sum(nh[i] * c[i] for i in range(3))
    assert abs(cost - C) < 1e-9
    # 誤り：費用を無視したネイマン配分の比率で、同じ予算を使い切る
    ney = [N[i] * S[i] for i in range(3)]
    k = C / sum(ney[i] * c[i] for i in range(3))
    ney_n = [k * v for v in ney]
    prop = [C * N[i] / sum(N[j] * c[j] for j in range(3)) for i in range(3)]
    return {
        "N1": N[0], "N2": N[1], "N3": N[2], "S1": S[0], "S2": S[1], "S3": S[2], "c1": c[0], "c2": c[1], "c3": c[2], "C": C,
        "rc1": s(rc[0], 0), "rc2": s(rc[1], 0), "rc3": s(rc[2], 0),
        "a1": s(a[0], 0), "a2": s(a[1], 0), "a3": s(a[2], 2), "b1": s(b[0], 0), "b2": s(b[1], 0), "b3": s(b[2], 0),
        "sumb": s(sum(b), 0), "n1": s(nh[0], 2), "n2": r(nh[1], 0), "n2s": s(nh[1], 2), "n3": s(nh[2], 2),
        "w_ney": s(ney_n[1], 1), "w_prop": s(prop[1], 1),
    }


def p13():
    # (a) AR(1) のスペクトル
    phi = 0.8
    ratio = ((1 + phi) / (1 - phi)) ** 2
    # (b) 主成分による異常度
    rho = 0.8
    l1, l2 = 1 + rho, 1 - rho
    z1, z2 = 1.5, -0.5
    t1 = (z1 + z2) / sqrt(2)
    t2 = (z1 - z2) / sqrt(2)
    T2 = t1 ** 2 / l1 + t2 ** 2 / l2
    Q = t2 ** 2
    # (c) 2次判別
    ln2, ln4 = 0.6931, 1.3863
    c2 = 8 * ln2 / 3
    c = sqrt(c2)
    c_ln4 = sqrt(8 * ln4 / 3)
    c_sd4 = sqrt(ln4 / (1 / 2 - 1 / 32))
    return {
        "phi": s(phi, 1), "opp": s(1 + phi, 1), "omp": s(1 - phi, 1), "ratio": s(ratio, 0), "rt": s((1 + phi) / (1 - phi), 0),
        "w_inv": s(1 / ratio, 4), "w_1mp": s(1 / (1 - phi) ** 2, 0),
        "rho": s(rho, 1), "l1": s(l1, 1), "l2": s(l2, 1), "z1": s(z1, 1), "z2": s(z2, 1),
        "zs": s(z1 + z2, 1), "zd": s(z1 - z2, 1), "t1sq": s(t1 ** 2, 2), "t2sq": s(t2 ** 2, 2),
        "T2": s(T2, 2), "T2a": s(t1 ** 2 / l1, 4), "T2b": s(t2 ** 2 / l2, 4), "Q": s(Q, 2),
        "w_nodiv": s(t1 ** 2 + t2 ** 2, 2), "w_mul": s(t1 ** 2 * l1 + t2 ** 2 * l2, 2), "w_first": s(t1 ** 2 / l1, 2),
        "w_Q": s(t1 ** 2, 2),
        "ln2": s(ln2, 4), "ln4": s(ln4, 4), "c2": s(c2, 4), "c": s(c, 2), "c4": s(c, 4), "c_ln4": s(c_ln4, 2), "c_sd4": s(c_sd4, 2),
        "coef": s(1 / 2 - 1 / 8, 3),
    }


def p14():
    dA = [2.0, 3.0, 1.0]
    dB = [5.0, 4.0, 6.0, 5.0]
    dC = [3.0, 4.0]
    a = sum(dA) / len(dA)
    bB = sum(dB) / len(dB)
    bC = sum(dC) / len(dC)
    b = min(bB, bC)
    sil = (b - a) / max(a, b)
    return {
        "dA1": s(dA[0], 1), "dA2": s(dA[1], 1), "dA3": s(dA[2], 1),
        "dB1": s(dB[0], 1), "dB2": s(dB[1], 1), "dB3": s(dB[2], 1), "dB4": s(dB[3], 1),
        "dC1": s(dC[0], 1), "dC2": s(dC[1], 1),
        "a": s(a, 1), "bB": s(bB, 1), "bC": s(bC, 1), "b": s(b, 1),
        "sil": r(sil, 2), "sil4": s(sil, 4), "w_far": s((bB - a) / bB, 2), "w_div": s((b - a) / a, 2),
    }


def values():
    out = {}
    for i, f in enumerate([p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14], start=1):
        v = f()
        out[f"s26p{i:02d}"] = {k: val for k, val in v.items() if not k.startswith("_")}
    return out


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
