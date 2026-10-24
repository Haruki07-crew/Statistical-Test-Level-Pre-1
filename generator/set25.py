"""模擬試験 第25回（set25）の数値。values() が {大問id: {名前: 値}} を返す。

題材は製薬の品質試験・バイオ実験・研究所。問題文に示した分布の値（Φ・t・χ² の点、e^x、ln x など）は、
ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。負の数は − ではなく - のまま。"""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def m(x, d):
    """数式用：負の数のマイナスを残した文字列（s と同じ）。"""
    return s(x, d)


# ---------------------------------------------------------------- 大問1 限界希釈法
def p01():
    W, W0 = 96, 58
    W1 = W - W0
    ln_ratio = -0.5039                      # 問題文に示す ln(58/96)
    lam = -ln_ratio
    p1_cond = lam * W0 / W1                 # λe^{-λ}/(1-e^{-λ})、e^{-λ}=W0/W
    p1_unc = lam * W0 / W                   # 条件づけ忘れ
    # (b) ゼロ切断ポアソン
    cnt = {1: 28, 2: 16, 3: 5, 4: 1}
    n = sum(cnt.values())
    tot = sum(k * v for k, v in cnt.items())
    xbar = tot / n
    g = {0.8: 1.4528, 0.9: 1.5166, 1.0: 1.5820, 1.1: 1.6489, 1.2: 1.7172}   # λ/(1-e^{-λ})（小数第4位）
    lam_hat = min(g, key=lambda l: abs(g[l] - xbar))
    assert lam_hat == 1.0
    # 誤答：λ(1-e^{-λ})=x̄ の解（二分法）と、固定点反復を1回で止めた値
    lo, hi = 1.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid * (1 - math.exp(-mid)) < xbar:
            lo = mid
        else:
            hi = mid
    w_inv = (lo + hi) / 2
    w_iter = xbar * (1 - math.exp(-xbar))
    return {
        "W": W, "W0": W0, "W1": W1, "lnr": s(ln_ratio, 4), "lam": s(lam, 4),
        "e0": f"{W0}/{W}", "p1c": r(p1_cond, 3), "p1c_s": s(p1_cond, 3), "p1c4": s(p1_cond, 4),
        "p1u": s(p1_unc, 3), "lamfrac": s(lam * W0 / W, 4),
        "n1": cnt[1], "n2": cnt[2], "n3": cnt[3], "n4": cnt[4], "n": n, "tot": tot, "xbar": s(xbar, 2),
        "g08": s(g[0.8], 4), "g09": s(g[0.9], 4), "g10": s(g[1.0], 4), "g11": s(g[1.1], 4), "g12": s(g[1.2], 4),
        "c_shift": s(xbar - 1, 2), "c_ok": s(lam_hat, 2), "c_iter": s(w_iter, 2), "c_naive": s(xbar, 2),
        "c_inv": s(w_inv, 2),
    }


# ---------------------------------------------------------------- 大問2 天秤の測定誤差
def p02():
    # (a) 丸め誤差の和
    mm, h, thr = 48, 0.1, 0.1
    var1 = h ** 2 / 12
    varS = mm * var1
    sdS = sqrt(varS)
    z = r(thr / sdS, 2)
    phi = {0.50: 0.6915, 1.00: 0.8413, 2.50: 0.9938}
    assert z in phi
    ans_a = 2 * (1 - phi[z])
    z_half = r(thr / sqrt(mm * (h / 2) ** 2 / 12), 2)       # 幅を 0.05 とした誤り
    assert z_half in phi
    # (b) 混合正規の尖度
    w1, w2, v1, v2 = 0.9, 0.1, 1.0, 4.0
    m4 = 3 * (w1 * v1 ** 2 + w2 * v2 ** 2)
    var = w1 * v1 + w2 * v2
    kurt = m4 / var ** 2
    k_var = m4 / var                                            # 分散で割った
    m4_wrong = 3 * (w1 * v1 + w2 * v2)                          # σ⁴ の代わりに σ² を使った
    k_m4w = m4_wrong / var ** 2
    # (c) ジャック・ベラ
    n, sk, ku = 120, 0.30, 4.50
    jb = n / 6 * (sk ** 2 + (ku - 3) ** 2 / 4)
    jb_n6 = sk ** 2 + (ku - 3) ** 2 / 4
    jb_no4 = n / 6 * (sk ** 2 + (ku - 3) ** 2)
    jb_no3 = n / 6 * (sk ** 2 + ku ** 2 / 4)
    return {
        "mm": mm, "h": h, "hh": s(h / 2, 2), "thr": thr, "var1": s(var1, 6), "varS": s(varS, 2), "sdS": s(sdS, 1),
        "z": s(z, 2), "phi050": s(phi[0.50], 4), "phi100": s(phi[1.00], 4), "phi250": s(phi[2.50], 4),
        "tail": s(1 - phi[z], 4), "ans_a": r(ans_a, 3), "ans_a_s": s(ans_a, 3),
        "one_side": s(1 - phi[z], 3), "w_half": s(2 * (1 - phi[z_half]), 3), "z_half": s(z_half, 2),
        "w1": w1, "w2": w2, "v1": s(v1, 0), "v2": s(v2, 0), "sd2": s(sqrt(v2), 0),
        "m4": s(m4, 2), "var": s(var, 1), "var2": s(var ** 2, 2), "kurt": s(kurt, 2), "kurt4": s(kurt, 4),
        "k_ex": s(kurt - 3, 2), "k_var": s(k_var, 2), "k_m4w": s(k_m4w, 2), "m4w": s(m4_wrong, 1),
        "n": n, "sk": s(sk, 2), "ku": s(ku, 2), "n6": s(n / 6, 0), "sk2": s(sk ** 2, 2),
        "ku3": s((ku - 3) ** 2 / 4, 4), "ku3sq": s((ku - 3) ** 2, 2), "inner": s(jb_n6, 4),
        "jb": s(jb, 2), "jb_n6": s(jb_n6, 2), "jb_no4": s(jb_no4, 2), "jb_no3": s(jb_no3, 2),
        "chi2": "5.991", "chi1": "3.841",
    }


# ---------------------------------------------------------------- 大問3 製剤の安定性試験
def p03():
    lam, c = 0.1, 24
    lc = lam * c
    e24, e12 = 0.0907, 0.3012
    ans = 1 - (1 - e24) / lc
    w_mean = 1 - e12                       # Y を平均 12 に置き換えた
    w_end = 1 - e24                        # Y を 24 に置き換えた
    w_exp = lam / (lam + 1 / 12)           # Y を平均 12 の指数分布とした
    # (b) 対数正規の変動係数
    s2, e016 = 0.16, 1.1735
    cv = sqrt(e016 - 1)
    return {
        "lam": lam, "mean": s(1 / lam, 0), "c": c, "lc": s(lc, 1), "e24": s(e24, 4), "e12": s(e12, 4),
        "one_e": s(1 - e24, 4), "frac": s((1 - e24) / lc, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "w_comp": s(1 - ans, 3), "w_mean": s(w_mean, 3), "w_end": s(w_end, 3), "w_exp": s(w_exp, 3),
        "s2": s2, "sd": s(sqrt(s2), 1), "e016": s(e016, 4), "cv2": s(e016 - 1, 4), "cv": s(cv, 3), "cv4": s(cv, 4),
        "w_sqrt": s(sqrt(e016), 3), "c_s2": s(s2, 3), "c_sd": s(sqrt(s2), 3), "cvpct": s(cv * 100, 1),
    }


# ---------------------------------------------------------------- 大問4 ラベルの貼り直し
def p04():
    k = 4
    terms = [(-1) ** j / math.factorial(j) for j in range(k + 1)]
    p0 = sum(terms)
    n_der = round(p0 * math.factorial(k))
    p1 = n_der_1 = 8 / 24                  # ちょうど1本正しい：4×D3/4! = 4×2/24
    assert abs(p1 - k * 2 / math.factorial(k)) < 1e-12
    return {
        "k": k, "kf": math.factorial(k), "nder": n_der, "ans": s(p0, 3), "ans_frac": f"{n_der}/{math.factorial(k)}",
        "w_ind": s((3 / 4) ** 4, 3), "w_e": s(math.exp(-1), 3), "w_one": s(n_der_1, 3), "w_atl": s(1 - p0, 3),
        "S1": "1", "S2": s(1 / 2, 1), "S3": s(1 / 6, 4), "S4": s(1 / 24, 4),
    }


# ---------------------------------------------------------------- 大問5 2種類の培地の汚染率
def p05():
    nA, xA, nB, xB = 200, 18, 250, 10
    pA, pB = xA / nA, xB / nB
    pp = (xA + xB) / (nA + nB)
    se = sqrt(pp * (1 - pp) * (1 / nA + 1 / nB))
    z = (pA - pB) / se
    se_u = sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    # (b) ウルフの区間
    a, b, c, d = xA, nA - xA, xB, nB - xB
    orr = a * d / (b * c)
    lor = r(math.log(orr), 4)
    v = 1 / a + 1 / b + 1 / c + 1 / d
    se_l = r(sqrt(v), 4)
    lo, hi = lor - 1.96 * se_l, lor + 1.96 * se_l
    lo90, hi90 = lor - 1.645 * se_l, lor + 1.645 * se_l
    se_ac = r(sqrt(1 / a + 1 / c), 4)
    loac, hiac = lor - 1.96 * se_ac, lor + 1.96 * se_ac
    etab = {0.068: 1.0704, 0.091: 1.0953, 0.196: 1.2165, 1.533: 4.6321, 1.637: 5.1397, 1.661: 5.2646}
    for x in (lo, hi, lo90, hi90, loac, hiac):
        assert r(x, 3) in etab, x
    E = lambda x: etab[r(x, 3)]
    raw_lo, raw_hi = orr - 1.96 * se_l, orr + 1.96 * se_l
    # (c) ジェフリーズ事前分布
    n0 = 20
    a_post, b_post = 0.5, 0.5 + n0
    mean = a_post / (a_post + b_post)
    return {
        "nA": nA, "xA": xA, "nB": nB, "xB": xB, "pA": s(pA, 2), "pB": s(pB, 2), "diff": s(pA - pB, 2),
        "xs": xA + xB, "ns": nA + nB, "pp": s(pp, 5), "qq": s(1 - pp, 5), "inv": s(1 / nA + 1 / nB, 3),
        "se": s(se, 5), "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4), "z_u": s((pA - pB) / se_u, 2), "se_u": s(se_u, 5),
        "a": a, "b": b, "c": c, "d": d, "or": s(orr, 3), "lor": s(lor, 4), "v": s(v, 4), "se_l": s(se_l, 4),
        "ia": s(1 / a, 5), "ib": s(1 / b, 5), "ic": s(1 / c, 5), "id": s(1 / d, 5), "v5": s(v, 5),
        "half": s(1.96 * se_l, 4), "se_ac": s(se_ac, 4), "lo": s(lo, 4), "hi": s(hi, 4),
        "e068": s(etab[0.068], 4), "e091": s(etab[0.091], 4), "e196": s(etab[0.196], 4),
        "e1533": s(etab[1.533], 4), "e1637": s(etab[1.637], 4), "e1661": s(etab[1.661], 4),
        "ci_lo": s(E(lo), 2), "ci_hi": s(E(hi), 2),
        "c90_lo": s(E(lo90), 2), "c90_hi": s(E(hi90), 2),
        "cac_lo": s(E(loac), 2), "cac_hi": s(E(hiac), 2),
        "nx_lo": s(lo, 2), "nx_hi": s(hi, 2), "raw_lo": s(raw_lo, 2), "raw_hi": s(raw_hi, 2),
        "n0": n0, "ap": "0.5", "bp": s(b_post, 1), "pmean": s(mean, 3), "pmean4": s(mean, 4),
        "w_unif": s(1 / (n0 + 2), 3), "w_mode": s((a_post - 1) / (a_post + b_post - 2), 3),
        "w_swap": s(1 - mean, 3), "abm2": s(a_post + b_post - 2, 0), "ab": s(a_post + b_post, 0),
    }


# ---------------------------------------------------------------- 大問6 2ロットの含量
def p06():
    n1, m1, s1 = 8, 99.6, 1.2
    n2, m2, s2 = 10, 98.4, 1.0
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
    inv = 1 / n1 + 1 / n2
    se = sqrt(sp2 * inv)
    t16, t18, z = 2.120, 2.101, 1.960
    lo = (m1 - m2) - t16 * se
    return {
        "n1": n1, "m1": m1, "s1": s1, "n2": n2, "m2": m2, "s2": s2, "df": df, "df2": n1 + n2,
        "ss1": s((n1 - 1) * s1 ** 2, 2), "ss2": s((n2 - 1) * s2 ** 2, 2), "sp2": s(sp2, 4), "inv": s(inv, 3),
        "se": s(se, 4), "diff": s(m1 - m2, 1), "t16": s(t16, 3), "t18": s(t18, 3), "z": s(z, 3),
        "half": s(t16 * r(se, 4), 4), "lo": r(lo, 3), "lo4": s((m1 - m2) - r(t16 * r(se, 4), 4), 4),
        "hi": s((m1 - m2) + t16 * se, 3),
        "w18": s((m1 - m2) - t18 * se, 3), "wz": s((m1 - m2) - z * se, 3),
    }


# ---------------------------------------------------------------- 大問7 分注機の作動遅延
def p07():
    th0, th1, n, alpha = 2.0, 2.5, 5, 0.05
    root = 0.98979                         # 0.95^{1/5}
    cc = th0 * root
    ratio = th0 / th1
    pw = 1 - (1 - alpha) * ratio ** n
    return {
        "th0": s(th0, 1), "th1": s(th1, 1), "n": n, "root": s(root, 5), "cc": s(cc, 4),
        "ratio": s(ratio, 1), "rpow": s(ratio ** n, 5), "beta": s((1 - alpha) * ratio ** n, 4),
        "pw": s(pw, 3), "pw4": s(pw, 4),
        "w_beta": s((1 - alpha) * ratio ** n, 3), "w_noa": s(1 - ratio ** n, 3),
        "w_one": s(1 - cc / th1, 3), "w_noroot": s(1 - (th0 * (1 - alpha) / th1) ** n, 3),
        "c_noroot": s(th0 * (1 - alpha), 2),
    }


# ---------------------------------------------------------------- 大問8 インキュベーターの運用
def p08():
    up = [0.5, 0.4, 0.3]                   # k→k+1（k=0,1,2）
    dn = [0.2, 0.3, 0.6]                   # k→k-1（k=1,2,3）
    w = [1.0]
    for k in range(3):
        w.append(w[-1] * up[k] / dn[k])
    tot = sum(w)
    pi = [x / tot for x in w]
    wr = [1.0]
    for k in range(3):
        wr.append(wr[-1] * dn[k] / up[k])
    pr = wr[3] / sum(wr)
    mean_use = sum(k * pi[k] for k in range(4))
    stay = [1 - up[0], 1 - up[1] - dn[0], 1 - up[2] - dn[1], 1 - dn[2]]
    # 整数比 6:15:20:10
    ints = [round(x * 6) for x in w]
    assert all(abs(x * 6 - i) < 1e-9 for x, i in zip(w, ints))
    # (b) DF 検定
    g, se = -0.12, 0.045
    tau = g / se
    return {
        "u0": up[0], "u1": up[1], "u2": up[2], "d1": dn[0], "d2": dn[1], "d3": dn[2],
        "s0": s(stay[0], 1), "s1": s(stay[1], 1), "s2": s(stay[2], 1), "s3": s(stay[3], 1),
        "w1": s(w[1], 1), "w2": s(w[2], 4), "w3": s(w[3], 4), "i0": ints[0], "i1": ints[1], "i2": ints[2], "i3": ints[3],
        "isum": sum(ints), "ans": s(pi[3], 3), "ans4": s(pi[3], 4),
        "w_unif": s(0.25, 3), "w_rev": s(pr, 3), "w_two": s(pi[2], 3), "w_rate": s(mean_use / 3, 3),
        "mean_use": s(mean_use, 3),
        "g": s(g, 2), "se": s(se, 3), "tau": s(tau, 2), "rho": s(1 + g, 2), "wrong_t": s((1 + g) / se, 2),
        "dfc": "-2.89", "tc": "-1.661", "tc2": "1.985",
    }


# ---------------------------------------------------------------- 大問9 HPLC の検量線
def p09():
    n = 12
    rss = {1: 0.0480, 2: 0.0210, 3: 0.0190}
    lnv = {1: -5.5215, 2: -6.3481, 3: -6.4482}         # ln(RSS/n)（小数第4位）
    for kdeg in rss:
        assert abs(math.log(rss[kdeg] / n) - lnv[kdeg]) < 6e-5
    k = {d: d + 2 for d in rss}                       # 回帰係数 d+1 個と誤差分散
    aic = {d: n * lnv[d] + 2 * k[d] for d in rss}
    best = min(aic, key=aic.get)
    assert best == 2
    aic_k = {d: n * lnv[d] + k[d] for d in rss}      # 罰則を 2 倍し忘れ
    assert min(aic_k, key=aic_k.get) == 3
    aic_nov = n * lnv[2] + 2 * (k[2] - 1)
    # (c) VIF
    xs = [2, 4, 6, 8, 10, 12] * 2
    mx = sum(xs) / len(xs)
    x2 = [x * x for x in xs]
    mx2 = sum(x2) / len(x2)
    sxy = sum((a - mx) * (b - mx2) for a, b in zip(xs, x2))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - mx2) ** 2 for b in x2)
    rr = r(sxy / sqrt(sxx * syy), 3)
    vif = 1 / (1 - rr ** 2)
    return {
        "n": n, "rss1": s(rss[1], 4), "rss2": s(rss[2], 4), "rss3": s(rss[3], 4),
        "q1": s(rss[1] / n, 4), "q2": s(rss[2] / n, 5), "q3": s(rss[3] / n, 7),
        "l1": s(lnv[1], 4), "l2": s(lnv[2], 4), "l3": s(lnv[3], 4),
        "nl1": s(n * lnv[1], 3), "nl2": s(n * lnv[2], 4), "nl3": s(n * lnv[3], 4),
        "k1": k[1], "k2": k[2], "k3": k[3], "pen1": 2 * k[1], "pen2": 2 * k[2], "pen3": 2 * k[3],
        "a1": s(aic[1], 2), "a2": s(aic[2], 2), "a3": s(aic[3], 2), "a3k": s(aic_k[3], 2), "a2k": s(aic_k[2], 2),
        "a2nov": s(aic_nov, 2),
        "rr": s(rr, 3), "rr2": s(rr ** 2, 6), "one_r2": s(1 - rr ** 2, 6), "vif": s(vif, 1), "vif2": s(vif, 2),
        "w_1r": s(1 / (1 - rr), 1), "w_odds": s(rr ** 2 / (1 - rr ** 2), 1), "mx": s(mx, 0),
    }


# ---------------------------------------------------------------- 大問10 酵素活性と温度
def p10():
    b0, b1, b2, knot, t = 2.0, 0.30, -0.75, 37, 41
    pred = b0 + b1 * t + b2 * max(t - knot, 0)
    return {
        "b0": s(b0, 1), "b1": s(b1, 2), "b2": s(b2, 2), "knot": knot, "t": t, "dt": t - knot,
        "lin": s(b1 * t, 2), "hinge": s(b2 * (t - knot), 2),
        "pred": s(pred, 2), "slope": s(b1 + b2, 2),
        "w_nohinge": s(b0 + b1 * t, 2), "w_full": s(b0 + b1 * t + b2 * t, 2),
    }


# ---------------------------------------------------------------- 大問11 培養条件の実験
def p11():
    days, a, b = 4, 3, 2
    total = days * a * b - 1
    df_blk, df_a = days - 1, a - 1
    df_e1 = (days - 1) * (a - 1)
    df_b, df_ab = b - 1, (a - 1) * (b - 1)
    df_e2 = a * days * (b - 1) - df_b - df_ab
    assert df_e2 == total - df_blk - df_a - df_e1 - df_b - df_ab
    df_pool = df_e1 + df_e2
    df_rep = a * b * (days - 1)                     # 日を無視した繰り返しのある二元配置
    # (b) 逸脱度残差
    y, mu, ln16 = 12, 7.5, 0.4700
    assert abs(math.log(y / mu) - ln16) < 1e-4
    dev = 2 * (y * ln16 - (y - mu))
    dres = sqrt(dev)
    pear = (y - mu) / sqrt(mu)
    return {
        "days": days, "a": a, "b": b, "total": total, "dfb": df_blk, "dfa": df_a, "dfe1": df_e1,
        "dfB": df_b, "dfab": df_ab, "dfe2": df_e2, "dfpool": df_pool, "dfrep": df_rep, "N": days * a * b,
        "ainc": a * days,
        "y": y, "mu": s(mu, 1), "ln16": s(ln16, 4), "ylog": s(y * ln16, 2), "ymu": s(y - mu, 1),
        "devi": s(dev, 2), "dres": r(dres, 2), "dres4": s(dres, 4), "pear": s(pear, 2),
        "w_noterm": s(sqrt(2 * y * ln16), 2), "w_no2": s(sqrt(dev / 2), 2), "ratio": s(y / mu, 1),
    }


# ---------------------------------------------------------------- 大問12 安全意識調査
def p12():
    N, nR, yR, nF, yF = 400, 240, 72.0, 20, 60.0
    nNR = N - nR
    wgt = nNR / nF
    est = (nR * yR + nF * wgt * yF) / N
    bias = yR - est
    return {
        "N": N, "nR": nR, "yR": s(yR, 1), "nNR": nNR, "nF": nF, "yF": s(yF, 1), "wgt": s(wgt, 0),
        "rr": s(nR / N, 1), "nr": s(nNR / N, 1), "est": s(est, 1), "bias": r(bias, 1), "bias_s": s(bias, 1),
        "gap": s(yR - yF, 1), "sumR": s(nR * yR, 0), "sumF": s(nF * wgt * yF, 0),
        "w_half": s(yR - (yR + yF) / 2, 1), "w_full": s(yR - yF, 1),
    }


# ---------------------------------------------------------------- 大問13 微生物株のクラスタリング
def p13():
    dA = {"B": 2.0, "C": 3.0, "D": 5.0, "E": 7.0, "F": 4.5, "G": 6.5}
    a_ = (dA["B"] + dA["C"]) / 2
    bDE = (dA["D"] + dA["E"]) / 2
    bFG = (dA["F"] + dA["G"]) / 2
    b_ = min(bDE, bFG)
    sil = (b_ - a_) / max(a_, b_)
    near = min(dA["D"], dA["E"], dA["F"], dA["G"])
    return {
        **{f"d{k}": s(v, 1) for k, v in dA.items()},
        "a": s(a_, 1), "bDE": s(bDE, 1), "bFG": s(bFG, 1), "b": s(b_, 1),
        "sil": r(sil, 2), "sil4": s(sil, 4),
        "w_near": s((near - a_) / near, 2), "w_far": s((bDE - a_) / bDE, 2),
        "w_all": s((sum(dA[k] for k in "DEFG") / 4 - a_) / (sum(dA[k] for k in "DEFG") / 4), 2),
    }


# ---------------------------------------------------------------- 大問14 錠剤の品質特性
def p14():
    r12, r13, r23 = 0.6, 0.5, 0.3
    num = r12 ** 2 + r13 ** 2 - 2 * r12 * r13 * r23
    den = 1 - r23 ** 2
    smc = num / den
    # (b) 正準相関（各組内は無相関）
    R = [[0.4, 0.3], [-0.2, 0.4]]
    M = [[sum(R[i][k] * R[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    disc = (tr / 2) ** 2 - det
    lam1 = tr / 2 + sqrt(disc)
    lam2 = tr / 2 - sqrt(disc)
    return {
        "r12": r12, "r13": r13, "r23": r23, "num": s(num, 2), "den": s(den, 2), "smc": s(smc, 3), "smc4": s(smc, 4),
        "cross": s(2 * r12 * r13 * r23, 2), "sq": s(r12 ** 2 + r13 ** 2, 2),
        "w_sum": s(r12 ** 2 + r13 ** 2, 3), "w_max": s(r12 ** 2, 3), "w_num": s(num, 3), "w_uniq": s(1 - smc, 3),
        "R11": R[0][0], "R12": R[0][1], "R21": R[1][0], "R22": R[1][1],
        "M11": s(M[0][0], 2), "M12": s(M[0][1], 2), "M22": s(M[1][1], 2),
        "tr": s(tr, 2), "det": s(det, 4), "half": s(tr / 2, 3), "disc": s(disc, 6), "sdisc": s(sqrt(disc), 5),
        "lam1": s(lam1, 5), "lam2": s(lam2, 5), "rho1": s(sqrt(lam1), 3), "rho2": s(sqrt(lam2), 3),
        "w_lam": s(lam1, 3), "w_max": s(0.4, 3), "w_tr": s(tr, 3), "w_sqtr": s(sqrt(tr), 3),
    }


def values():
    return {"s25p01": p01(), "s25p02": p02(), "s25p03": p03(), "s25p04": p04(), "s25p05": p05(),
            "s25p06": p06(), "s25p07": p07(), "s25p08": p08(), "s25p09": p09(), "s25p10": p10(),
            "s25p11": p11(), "s25p12": p12(), "s25p13": p13(), "s25p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
