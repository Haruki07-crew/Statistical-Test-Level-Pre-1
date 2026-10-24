"""模擬試験 第24回（set24）の数値。values() が {大問id: {名前: 値}} を返す。

題材：広告・SNS・メディア分析・出版。
問題文に示した関数の値（指数・対数・正規分布の値、t 分布の点、二項確率）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
import math
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


def ceil(x):
    return int(-(-x // 1))


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
    # (a) 販促日と通常日の混合：全分散の公式
    pp, m1, s1, m0, s0 = 0.3, 50, 10, 20, 5
    E = pp * m1 + (1 - pp) * m0
    ev = pp * s1 ** 2 + (1 - pp) * s0 ** 2
    ve = pp * m1 ** 2 + (1 - pp) * m0 ** 2 - E ** 2
    vals = [sqrt(ev + ve), sqrt(ev), pp * s1 + (1 - pp) * s0, sqrt(ve), sqrt(pp ** 2 * s1 ** 2 + (1 - pp) ** 2 * s0 ** 2)]
    assert len({s(v, 2) for v in vals}) == 5
    # (b) 抽出ウェイトによる設計効果（キッシュ）
    N1, n1, N2, n2 = 20000, 400, 80000, 400
    w1, w2 = N1 / n1, N2 / n2
    sw = n1 * w1 + n2 * w2
    sw2 = n1 * w1 ** 2 + n2 * w2 ** 2
    n = n1 + n2
    deff = n * sw2 / sw ** 2
    neff = sw ** 2 / sw2
    p1, p2 = 0.60, 0.30
    est_w = (N1 * p1 + N2 * p2) / (N1 + N2)
    est_u = (n1 * p1 + n2 * p2) / n
    return {
        "pp": s(pp, 1), "qq": s(1 - pp, 1), "m1": m1, "s1": s1, "m0": m0, "s0": s0, "E": s(E, 0),
        "ev": s(ev, 1), "ve": s(ve, 1), "V": s(ev + ve, 1), "sd": s(vals[0], 2), "w_ev": s(vals[1], 2), "w_avg": s(vals[2], 2),
        "w_ve": s(vals[3], 2), "w_ind": s(vals[4], 2), "e2": s(pp * m1 ** 2 + (1 - pp) * m0 ** 2, 0),
        "N1s": "20{,}000", "N2s": "80{,}000", "nn1": n1, "nn2": n2, "nn": n, "ww1": int(w1), "ww2": int(w2),
        "sw": "100{,}000", "sw2": "17{,}000{,}000", "sw_sq": "10^{10}", "deff": s(deff, 2), "neff": r(neff, 0), "neff_s": s(neff, 2),
        "neff_raw": s(neff, 0), "n_deff_wrong": s(n * deff, 0), "pp1": s(p1, 2), "pp2": s(p2, 2), "est_w": s(est_w, 2), "est_u": s(est_u, 2),
        "w1sq": "2{,}500", "w2sq": "40{,}000",
    }


def p03():
    # (a) 独立な対数正規の比：log(Y1/Y2) ~ N(0, 2σ²)
    sg = 1.0
    ph = {0.35: 0.6368, 0.49: 0.6879, 0.69: 0.7549}
    z = r(math.log(2) / (sg * sqrt(2)), 2)
    z1 = r(math.log(2) / sg, 2)
    z2 = r(math.log(2) / (2 * sg), 2)
    assert (z, z1, z2) == (0.49, 0.69, 0.35)
    # (b) 2 変量正規のマハラノビス距離の 2 乗は自由度 2 のカイ二乗分布
    c1, c2 = 3.84, 5.99
    e192 = 0.1466
    return {
        "sg": s(sg, 1), "ln2": "0.6931", "sq2": "1.4142", "z": s(z, 2), "z4": s(math.log(2) / (sg * sqrt(2)), 4),
        "ph35": s(ph[0.35], 4), "ph49": s(ph[0.49], 4), "ph69": s(ph[0.69], 4),
        "ok": s(1 - ph[z], 3), "w1": s(1 - ph[z1], 3), "w2": s(1 - ph[z2], 3), "w_ph": s(ph[z], 3),
        "c1": s(c1, 2), "c2": s(c2, 2), "e192": s(e192, 4), "in384": s(1 - e192, 3),
    }


def p04():
    # (a) 比の和の極限（大数の強法則と連続写像）
    I_, pr_ = [1000, 9000], [0.05, 0.01]
    ES = 0.5 * I_[0] * pr_[0] + 0.5 * I_[1] * pr_[1]
    EI = 0.5 * I_[0] + 0.5 * I_[1]
    lim = ES / EI
    avg = 0.5 * pr_[0] + 0.5 * pr_[1]
    # (b) 曜日ごとのシェア数の一様性（適合度検定）
    obs = [14, 18, 22, 25, 24, 19, 18]
    e = sum(obs) / 7
    X2 = sum((o - e) ** 2 / e for o in obs)
    X2o = sum((o - e) ** 2 / o for o in obs)
    return {
        "I1": I_[0], "I2": I_[1], "q1": pr_[0], "q2": pr_[1], "S1": s(I_[0] * pr_[0], 0), "S2": s(I_[1] * pr_[1], 0),
        "ES": s(ES, 0), "EI": s(EI, 0), "lim": r(lim, 3), "lim_s": s(lim, 3), "avg": s(avg, 2),
        "obs": " | ".join(map(str, obs)), "tot": sum(obs), "e": s(e, 0),
        "dev2": " + ".join(f"{(o - e) ** 2:.0f}" for o in obs), "sdev2": s(sum((o - e) ** 2 for o in obs), 0),
        "X2": s(X2, 2), "X2o": s(X2o, 2), "c6": "12.59", "c7": "14.07",
    }


def p05():
    # (a) 母標準偏差が既知の 2 群：半幅を最小にする配分
    sa, sb, N = 60, 30, 180
    na = N * sa / (sa + sb)
    nb = N - na
    hw = 1.96 * sqrt(sa ** 2 / na + sb ** 2 / nb)
    hw_eq = 1.96 * sqrt(sa ** 2 / (N / 2) + sb ** 2 / (N / 2))
    # (b) 事前の等分散検定と 2 標本 t（プール vs ウェルチ）
    n1, sd1, n2, sd2, d = 10, 30, 30, 12, 20
    sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2)
    sep = sqrt(sp2 * (1 / n1 + 1 / n2))
    v1, v2 = sd1 ** 2 / n1, sd2 ** 2 / n2
    sew = sqrt(v1 + v2)
    dfw = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return {
        "sa": sa, "sb": sb, "N": N, "na": s(na, 0), "nb": s(nb, 0), "hw": r(hw, 2), "hw_s": s(hw, 2), "hw4": s(hw, 4),
        "hw_eq": s(hw_eq, 2), "vsum": s(sa ** 2 / na + sb ** 2 / nb, 1),
        "n1": n1, "sd1": sd1, "n2": n2, "sd2": sd2, "d": d, "sp2": s(sp2, 2), "sp": s(sqrt(sp2), 2), "sep": s(sep, 3),
        "tp": s(d / sep, 2), "v1": s(v1, 1), "v2": s(v2, 1), "sew": s(sew, 3), "tw": s(d / sew, 2), "dfw": s(dfw, 2),
        "dfw0": s(dfw, 0), "t38": "2.024", "t10": "2.228", "Fr": s(sd1 ** 2 / sd2 ** 2, 2), "F95": "2.21",
    }


def p06():
    # (a) ポアソン分布の λ² の一様最小分散不偏推定量
    xs = [2, 5, 1, 3, 4, 2, 6, 3, 2, 2]
    n = len(xs)
    T = sum(xs)
    xb = T / n
    umvue = T * (T - 1) / n ** 2
    naive = xb ** 2
    avgf = sum(v * (v - 1) for v in xs) / n
    # (b) ブートストラップのパーセンタイル区間
    med, q025, q05, q95, q975, bm, bsd = 48.0, 41.0, 42.5, 56.0, 58.5, 48.8, 4.6
    return {
        "xs": "、".join(map(str, xs)), "n": n, "T": T, "xb": s(xb, 1), "umvue": s(umvue, 2), "naive": s(naive, 2), "avgf": s(avgf, 2),
        "TT1": T * (T - 1), "n2": n * n, "bias": s(xb / n, 2),
        "med": s(med, 1), "q025": s(q025, 1), "q05": s(q05, 1), "q95": s(q95, 1), "q975": s(q975, 1), "bm": s(bm, 1), "bsd": s(bsd, 1),
        "nlo": s(med - 1.96 * bsd, 2), "nhi": s(med + 1.96 * bsd, 2), "blo": s(2 * med - q975, 1), "bhi": s(2 * med - q025, 1),
    }


def p07():
    # (a) 10 冊で集まるカードの種類数の期待値（指示変数の和）
    k, m = 6, 10
    q = 0.1615
    Ed = k * (1 - q)
    # (b) 複合ポアソン
    lam, t = 6, 8
    pk = {1: 0.6, 2: 0.3, 3: 0.1}
    EK = sum(kk * p for kk, p in pk.items())
    EK2 = sum(kk * kk * p for kk, p in pk.items())
    VK = EK2 - EK ** 2
    mm = lam * t * EK
    v = lam * t * EK2
    sd = sqrt(v)
    return {
        "k": k, "m": m, "q": s(q, 4), "Ed": r(Ed, 3), "Ed_s": s(Ed, 3), "Ed4": s(Ed, 4), "w_m": s(m / k, 2),
        "lam": lam, "t": t, "lt": lam * t, "p1": "0.6", "p2": "0.3", "p3": "0.1",
        "EK": s(EK, 1), "EK2": s(EK2, 1), "VK": s(VK, 2), "mm": s(mm, 0), "v": s(v, 1),
        "sd": r(sd, 2), "sd_s": s(sd, 2), "sd4": s(sd, 4), "sd_vk": s(sqrt(lam * t * VK), 2),
        "sd_pois": s(sqrt(mm), 2), "sd_ek2": s(sqrt(lam * t * EK ** 2), 2),
    }


def p08():
    n, xbar, Sxx, b0, b1, s2 = 12, 3.0, 20.0, 2.0, 1.5, 4.0
    se = sqrt(s2 / Sxx)
    t10, t11, z = 2.228, 2.201, 1.96
    iv = lambda h: f"({s(b1 - h, 2)},\\ {s(b1 + h, 2)})"
    hw = t10 * r(se, 4)
    # (b) ブルーシュ・ペーガン検定（残差の 2 乗を x に回帰）
    R2a = 0.42
    bp = n * R2a
    return {
        "n": n, "xbar": s(xbar, 1), "Sxx": s(Sxx, 1), "b0": s(b0, 1), "b1": s(b1, 1), "s2": s(s2, 1), "rss": s(s2 * (n - 2), 1),
        "df": n - 2, "se": s(se, 4), "se2": s(s2 / Sxx, 1), "hw": s(hw, 4), "ci": iv(hw), "ci_11": iv(t11 * r(se, 4)),
        "ci_s2": iv(t10 * s2 / sqrt(Sxx)), "ci_sxx": iv(t10 * sqrt(s2) / Sxx), "ci_z": iv(z * r(se, 4)),
        "t10": "2.228", "t11": "2.201",
        "R2a": s(R2a, 2), "bp": s(bp, 2), "bp_n2": s((n - 2) * R2a, 2), "c1": "3.84", "c2": "5.99",
    }


def p09():
    # 直交計画での変数選択：C_p（AIC）は |b_j| > √2 σ、t 検定（5%）は |b_j| > 1.96σ
    b = [2.1, -0.9, 1.5, 0.3, -1.8, 0.6]
    sig = 1.0
    cp = [j + 1 for j, v in enumerate(b) if v * v > 2 * sig ** 2]
    tt = [j + 1 for j, v in enumerate(b) if abs(v) > 1.96 * sig]
    assert cp == [1, 3, 5] and tt == [1]
    return {**{f"b{j + 1}": s(v, 1) for j, v in enumerate(b)}, "sq2": "1.414",
            "b2s": "、".join(s(v * v, 2) for v in b)}


def p10():
    a, b, c, d = 30, 70, 20, 180
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    OR = a * d / (b * c)
    RR = (a / r1) / (c / r2)
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
        "ad": a * d, "bc": b * c, "OR": s(OR, 2), "RR": s(RR, 2), "pa": s(a / r1, 2), "pc": s(c / r2, 2),
        "oa": s(a / b, 3), "oc": s(c / d, 3), "w_or_col": s((a / c) / (b / d), 2),
        "p_buzz": "1/6", "lnp": s(lnp, 4), "lnq": s(lnq, 4), "t0a": s(c1 * lnp, 2), "t0b": s(c2 * lnq, 3),
        "ll0": s(ll0, 3), "ll1": s(ll1, 1), "R2": s(R2, 3), "ratio": s(ratio, 3), "adj": s(adj, 3), "k": k,
        "lr": s(lr, 2), "R2_half": s(R2_half, 3), "ll_half": s(ll_half, 2),
    }


def p11():
    # (a) 2 つの検出方式の順位のケンドールの順位相関
    r1 = [1, 2, 3, 4, 5, 6, 7]
    r2 = [2, 1, 4, 3, 6, 7, 5]
    C = D = 0
    for i in range(7):
        for j in range(i + 1, 7):
            sgn = (r1[i] - r1[j]) * (r2[i] - r2[j])
            C += sgn > 0
            D += sgn < 0
    tau = (C - D) / 21
    d2 = sum((u - v) ** 2 for u, v in zip(r1, r2))
    rs = 1 - 6 * d2 / (7 * 48)
    # (b) ROC の動作点と期待コスト
    pi, cfn, cfp = 0.10, 5, 1
    pts = [(0.95, 0.30), (0.88, 0.15), (0.75, 0.05), (0.50, 0.01)]
    cost = [pi * (1 - tp) * cfn + (1 - pi) * fp * cfp for tp, fp in pts]
    err = [pi * (1 - tp) + (1 - pi) * fp for tp, fp in pts]
    nopi = [(1 - tp) * cfn + fp * cfp for tp, fp in pts]
    J = [tp - fp for tp, fp in pts]
    am = lambda v: v.index(min(v)) + 1
    assert (am(cost), am(err), am(nopi), J.index(max(J)) + 1) == (3, 4, 1, 2)
    return {
        "ra": " | ".join(map(str, r1)), "rb": " | ".join(map(str, r2)), "C": C, "D": D, "tau": s(tau, 3), "rs": s(rs, 3),
        "w_c": s(C / 21, 3), "w_cd": s(C / (C + D), 3), "d2": d2,
        "pi": s(pi, 2), "qi": s(1 - pi, 2), "cfn": cfn, "cfp": cfp,
        **{f"tp{k + 1}": s(pts[k][0], 2) for k in range(4)}, **{f"fp{k + 1}": s(pts[k][1], 2) for k in range(4)},
        **{f"fn{k + 1}": s(1 - pts[k][0], 2) for k in range(4)},
        **{f"a{k + 1}": s(pi * (1 - pts[k][0]) * cfn, 3) for k in range(4)}, **{f"b{k + 1}": s((1 - pi) * pts[k][1] * cfp, 3) for k in range(4)},
        **{f"cost{k + 1}": s(cost[k], 3) for k in range(4)}, **{f"err{k + 1}": s(err[k], 3) for k in range(4)},
        **{f"nopi{k + 1}": s(nopi[k], 2) for k in range(4)}, **{f"J{k + 1}": s(J[k], 2) for k in range(4)},
    }


def p12():
    # (a) 2 値データのジャカード係数
    A = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    B = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    a = sum(u and v for u, v in zip(A, B))
    b = sum(u and not v for u, v in zip(A, B))
    c = sum(v and not u for u, v in zip(A, B))
    d = len(A) - a - b - c
    jac = a / (a + b + c)
    sm = (a + d) / len(A)
    # (b) 古典的多次元尺度法：三角不等式を満たさない非類似度
    D2 = [[0, 4, 25], [4, 0, 4], [25, 4, 0]]
    rm = [sum(rw) / 3 for rw in D2]
    gm = sum(rm) / 3
    Bm = [[-0.5 * (D2[i][j] - rm[i] - rm[j] + gm) for j in range(3)] for i in range(3)]
    return {
        "A": " | ".join(map(str, A)), "B": " | ".join(map(str, B)), "na": a, "nb": b, "nc": c, "nd": d, "m": len(A),
        "jac": r(jac, 3), "jac_s": s(jac, 3), "sm": s(sm, 3), "w_ab": s(a / len(A), 3),
        "rm1": "29/3", "rm2": "8/3", "gm": "22/3", "b11": s(Bm[0][0], 1), "b22": s(Bm[1][1], 1), "b13": s(Bm[0][2], 1),
        "b12": s(Bm[0][1], 1), "ev1": "12.5", "ev2": "0", "ev3": "-1.5",
    }


def p13():
    # 2^3 要因実験（各条件 2 回）の効果量 η² と偏 η²
    SS = {"A": 48, "B": 12, "C": 3, "AB": 9, "AC": 1, "BC": 2, "ABC": 1, "E": 24}
    tot = sum(SS.values())
    eta = SS["A"] / tot
    peta = SS["A"] / (SS["A"] + SS["E"])
    F = SS["A"] / (SS["E"] / 8)
    psum = sum(SS[k] / (SS[k] + SS["E"]) for k in SS if k != "E")
    return {**{f"S{k}": v for k, v in SS.items()}, "tot": tot, "eta": s(eta, 3), "peta": s(peta, 3), "F": s(F, 1),
            "msE": s(SS["E"] / 8, 1), "petaB": s(SS["B"] / (SS["B"] + SS["E"]), 3), "psum": s(psum, 2),
            "w_ss": s(SS["A"] / (tot - SS["E"]), 3), "w_f": s(F / (F + 8), 3)}


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
