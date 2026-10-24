"""模擬試験 第17回（set17）の数値。values() が {大問id: {名前: 値}} を返す。

題材：飲食・ホテル・サービス業の顧客と需要。
問題文に示した分布の値（χ²・F・q の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from math import comb, sqrt

from scipy import stats


def r(x, d):
    """四捨五入（銀行丸めを避ける）。数値で返す。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return float(v) if d > 0 else int(v)


def s(x, d):
    """四捨五入した文字列（末尾の 0 も残す）。負の数は LaTeX の負号のまま。"""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- 大問1 注文の組み合わせ
def p01():
    # 8 通りの組み合わせの割合（A 前菜, B デザート, C ドリンク）
    atoms = {"abc": 0.15, "ab": 0.05, "ac": 0.09, "bc": 0.15, "a": 0.11, "b": 0.15, "c": 0.21, "none": 0.09}
    assert abs(sum(atoms.values()) - 1) < 1e-12
    pa = atoms["abc"] + atoms["ab"] + atoms["ac"] + atoms["a"]
    pb = atoms["abc"] + atoms["ab"] + atoms["bc"] + atoms["b"]
    pc = atoms["abc"] + atoms["ac"] + atoms["bc"] + atoms["c"]
    pab = atoms["abc"] + atoms["ab"]
    pac = atoms["abc"] + atoms["ac"]
    pbc = atoms["abc"] + atoms["bc"]
    pabc = atoms["abc"]
    union = 1 - atoms["none"]
    union_ind = 1 - (1 - pa) * (1 - pb) * (1 - pc)
    return {
        **{f"t_{k}": s(v, 2) for k, v in atoms.items()},
        "pa": s(pa, 2), "pb": s(pb, 2), "pc": s(pc, 2),
        "pab": s(pab, 2), "pac": s(pac, 2), "pbc": s(pbc, 2), "pabc": s(pabc, 2),
        "papb": s(pa * pb, 2), "papc": s(pa * pc, 2), "pbpc": s(pb * pc, 2), "papbpc": s(pa * pb * pc, 3),
        "qa": s(1 - pa, 1), "qb": s(1 - pb, 1), "qc": s(1 - pc, 1),
        "cgab": s(pabc / pab, 2), "union": s(union, 2), "union_ind": s(union_ind, 2),
    }


# ---------------------------------------------------------------- 大問2 客室単価の重回帰
def p02():
    n = 40
    models = {  # 名前: (説明変数の個数, RSS)
        "M1": (3, 520.0), "M2": (4, 430.0), "M3": (5, 400.0), "M4": (7, 368.0),
    }
    p1, rss1 = models["M1"]
    p2, rss2 = models["M3"]          # 部分 F 検定は M1 と M3 を比べる
    p3, rss3 = models["M4"]          # 全変数モデル
    df2 = n - p2 - 1
    F = ((rss1 - rss2) / (p2 - p1)) / (rss2 / df2)
    s2full = rss3 / (n - p3 - 1)

    def cp(name, sig2=s2full, pen=1, minus_n=True):
        p, rss = models[name]
        return rss / sig2 - (n if minus_n else 0) + 2 * (p + pen)

    cps = {k: cp(k) for k in models}
    best = min(cps, key=cps.get)
    assert best == "M3"
    # 誤りの Cp（いずれも M3 で）
    cp_nointer = cp("M3", pen=0)
    s2n = rss3 / n
    cps_n = {k: cp(k, sig2=s2n) for k in models}
    assert min(cps_n, key=cps_n.get) == "M3"
    cp_non = cp("M3", minus_n=False)
    # (c) 外部スチューデント化残差
    e, h = 8.0, 0.20
    s2 = rss2 / df2
    r_int = e / sqrt(s2 * (1 - h))
    rss_i = rss2 - e ** 2 / (1 - h)
    s2_i = rss_i / (df2 - 1)
    t_ext = e / sqrt(s2_i * (1 - h))
    t_from_r = r_int * sqrt((df2 - 1) / (df2 - r_int ** 2))
    assert abs(t_ext - t_from_r) < 1e-12
    t_wrong_noh = e / sqrt(s2_i)        # 1-h を掛け忘れ
    return {
        "n": n, "rss1": s(rss1, 1), "rss2": s(models["M2"][1], 1), "rss3": s(rss2, 1), "rss4": s(rss3, 1),
        "drss": s(rss1 - rss2, 1), "dp": p2 - p1, "df2": df2, "msd": s((rss1 - rss2) / (p2 - p1), 1),
        "mse2": s(rss2 / df2, 4), "F": r(F, 2), "F_s": s(F, 2), "Fc": s(stats.f.isf(0.05, 2, df2), 2),
        "F_df35": s(((rss1 - rss2) / 2) / (rss2 / (n - p2)), 2), "F_nodiv": s((rss1 - rss2) / (rss2 / df2), 2),
        "F_rss1": s(((rss1 - rss2) / 2) / (rss1 / (n - p1 - 1)), 2),
        "df4": n - p3 - 1, "dfa": n - p1 - 1, "s2full": s(s2full, 1),
        "cp1": s(cps["M1"], 2), "cp2": s(cps["M2"], 2), "cp3": s(cps["M3"], 2), "cp4": s(cps["M4"], 2),
        "r1": s(rss1 / s2full, 4), "r2": s(models["M2"][1] / s2full, 4), "r3": s(rss2 / s2full, 4), "r4": s(rss3 / s2full, 4),
        **{f"k{i}": models[f"M{i}"][0] + 1 for i in range(1, 5)},
        **{f"nk{i}": n - 2 * (models[f"M{i}"][0] + 1) for i in range(1, 5)},
        "cp_nointer": s(cp_nointer, 2), "cp_n": s(cps_n["M3"], 2), "cp_non": s(cp_non, 2), "s2n": s(s2n, 1),
        "e": s(e, 1), "h": s(h, 2), "omh": s(1 - h, 2), "e2": s(e ** 2, 1), "e2h": s(e ** 2 / (1 - h), 1),
        "rss_i": s(rss_i, 1), "df2m1": df2 - 1, "s2_i": s(s2_i, 4), "s_i": s(sqrt(s2_i), 4),
        "den_ext": s(sqrt(s2_i * (1 - h)), 4), "s_2": s(sqrt(s2), 4), "den_int": s(sqrt(s2 * (1 - h)), 4),
        "r_int": s(r_int, 2), "r_int4": s(r_int, 4), "t_ext": r(t_ext, 2), "t_ext_s": s(t_ext, 2), "t_ext4": s(t_ext, 4),
        "t_noh": s(t_wrong_noh, 2), "rint2": s(r_int ** 2, 4),
    }


# ---------------------------------------------------------------- 大問3 予約なしの来店組数
def p03():
    x = [0, 2, 1, 1, 3, 0, 1, 2, 1, 1]
    n = len(x)
    T = sum(x)
    m = T / n
    ss = sum((v - m) ** 2 for v in x)
    D = ss / m
    c9, c9lo, c10 = 16.92, 3.33, 18.31
    assert c9lo < D < c9
    # (b) P(X=0)=e^{-λ} の推定
    e12 = 0.3012
    umvue = (1 - 1 / n) ** T
    zero_frac = sum(1 for v in x if v == 0) / n
    # (c) 3 組目までの時間
    lam, t = 1.2, 2
    mu = lam * t
    e24 = 0.0907
    poly = 1 + mu + mu ** 2 / 2
    p3 = e24 * poly
    return {
        "x": ", ".join(str(v) for v in x), "n": n, "T": T, "m": s(m, 1), "sx2": sum(v * v for v in x),
        "nm2": s(n * m ** 2, 1), "ss": s(ss, 1), "D": s(D, 2), "D_raw": s(ss, 2), "Dratio": s(ss / (n - 1) / m, 2),
        "s2": s(ss / (n - 1), 4), "df": n - 1, "df10": n,
        "c9": s(c9, 2), "c9lo": s(c9lo, 2), "c10": s(c10, 2),
        "e12": s(e12, 4), "um": s(umvue, 3), "um4": s(umvue, 4), "um_p1": s((1 - 1 / n) ** (T + 1), 3),
        "um_m1": s((1 - 1 / n) ** (T - 1), 3), "mle": s(e12, 3), "zf": s(zero_frac, 3), "q": s(1 - 1 / n, 1),
        "lam": s(lam, 1), "t": t, "mu": s(mu, 1), "e24": s(e24, 4), "mu2h": s(mu ** 2 / 2, 2), "poly": s(poly, 2),
        "p3": r(p3, 3), "p3_s": s(p3, 3), "p3_4": s(p3, 4),
        "p3_no2": s(e24 * (1 + mu), 3), "p3_up3": s(e24 * (poly + mu ** 3 / 6), 3), "p3_exp": s(e24, 3),
    }


# ---------------------------------------------------------------- 大問4 リピーターの割合
def p04():
    n, x, z = 50, 40, 1.96
    ph = x / n
    z2 = z * z
    den = 1 + z2 / n
    cen = (ph + z2 / (2 * n)) / den
    half = z * sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)) / den
    up = cen + half
    lo = cen - half
    wald_up = ph + z * sqrt(ph * (1 - ph) / n)
    # (b) オッズのデルタ法
    odds = ph / (1 - ph)
    vp = ph * (1 - ph) / n
    g1 = 1 / (1 - ph) ** 2
    var_odds = g1 ** 2 * vp
    se_odds = sqrt(var_odds)
    return {
        "n": n, "x": x, "z": s(z, 2), "ph": s(ph, 1), "qh": s(1 - ph, 1), "z2": s(z2, 4),
        "z2n": s(z2 / n, 6), "z2_2n": s(z2 / (2 * n), 6), "den": s(den, 6), "num_c": s(ph + z2 / (2 * n), 6),
        "cen": s(cen, 4), "pq_n": s(ph * (1 - ph) / n, 4), "z2_4n2": s(z2 / (4 * n * n), 8),
        "rad": s(ph * (1 - ph) / n + z2 / (4 * n * n), 8), "srad": s(sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)), 5),
        "half": s(half, 4), "up": r(up, 3), "up_s": s(up, 3), "up4": s(up, 4), "lo": s(lo, 3),
        "wald_up": s(wald_up, 3), "sep": s(sqrt(vp), 4),
        "odds": s(odds, 0), "vp": s(vp, 4), "g1": s(g1, 0), "g1sq": s(g1 ** 2, 0), "var_odds": s(var_odds, 3),
        "se_odds": s(se_odds, 3), "se_lin": s(g1 * vp ** 0.5 / g1 * (1 / (1 - ph)), 3),
        "se_g1nosq": s(sqrt(g1 * vp), 3), "se_logit": s(sqrt(1 / (n * ph * (1 - ph))), 3), "se_p": s(sqrt(vp), 3),
    }


# ---------------------------------------------------------------- 大問5 新レシピの試食会
def p05():
    N, K, nn = 12, 6, 6          # 参加者 12 人、「また注文したい」6 人、新レシピ 6 人
    a = 5                        # 新レシピで「また注文したい」
    tot = comb(N, nn)
    pk = {k: comb(K, k) * comb(N - K, nn - k) for k in range(0, 7)}
    p_one = (pk[5] + pk[6]) / tot
    return {
        "N": N, "K": K, "nn": nn, "a": a, "b": nn - a, "c": K - a, "d": N - K - (nn - a),
        "tot": tot, "c65": comb(6, 5), "c61": comb(6, 1), "pk5": pk[5], "pk6": pk[6], "pk4": pk[4],
        "c64": comb(6, 4), "c62": comb(6, 2),
        "p_one": s(p_one, 3), "p_one4": s(p_one, 4), "p_5": s(pk[5] / tot, 3), "p_6": s(pk[6] / tot, 3),
        "p_two": s(2 * p_one, 3), "p_ge4": s((pk[4] + pk[5] + pk[6]) / tot, 3), "sum56": pk[5] + pk[6],
        "sum456": pk[4] + pk[5] + pk[6],
    }


# ---------------------------------------------------------------- 大問6 4つのコース料理
def p06():
    means = {"A": 7.20, "B": 6.10, "C": 4.60, "D": 6.65}
    n, k, mse = 6, 4, 1.50
    dfe = k * n - k
    q, q24, t20 = 3.96, 3.90, 2.086
    hsd = q * sqrt(mse / n)
    lsd = t20 * sqrt(2 * mse / n)
    hsd2 = q * sqrt(2 * mse / n)
    hsd_nosqrt = q * mse / n
    hsd_non = q * sqrt(mse)
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    diffs = {a + b: abs(means[a] - means[b]) for a, b in pairs}
    sig = lambda thr: sorted(kk for kk, v in diffs.items() if v > thr)
    assert sig(hsd) == ["AC", "CD"], sig(hsd)
    assert sig(lsd) == ["AC", "BC", "CD"], sig(lsd)
    assert sig(hsd_nosqrt) == ["AB", "AC", "BC", "CD"]
    assert sig(hsd2) == [] and sig(hsd_non) == []
    # (b) 各組の t 検定の両側 p 値（MSE を使う、自由度 20）。統計ソフトの出力として小数第4位まで示す
    se = sqrt(2 * mse / n)
    pv = {kk: r(2 * stats.t.sf(v / se, dfe), 4) for kk, v in diffs.items()}
    order = sorted(pv, key=pv.get)
    m = len(pv)
    holm = []
    for i, kk in enumerate(order):
        if pv[kk] <= 0.05 / (m - i):
            holm.append(kk)
        else:
            break
    bonf = [kk for kk in pv if pv[kk] <= 0.05 / m]
    assert sorted(holm) == ["AC", "CD"] and bonf == ["AC"], (holm, bonf, pv)
    out = {f"m{kk}": s(v, 2) for kk, v in means.items()}
    out.update({f"d{kk}": s(v, 2) for kk, v in diffs.items()})
    out.update({f"p{kk}": s(v, 4) for kk, v in pv.items()})
    out.update({
        "n": n, "k": k, "mse": s(mse, 2), "dfe": dfe, "N": k * n, "q": s(q, 2), "q24": s(q24, 2), "t20": s(t20, 3),
        "smn": s(sqrt(mse / n), 2), "mn": s(mse / n, 2), "hsd": s(hsd, 2), "lsd": s(lsd, 2), "hsd2": s(hsd2, 2),
        "hsd_ns": s(hsd_nosqrt, 2), "hsd_non": s(hsd_non, 2), "s2mn": s(se, 4),
        "th1": s(0.05 / 6, 5), "th2": s(0.05 / 5, 4), "th3": s(0.05 / 4, 4),
    })
    return out


# ---------------------------------------------------------------- 大問7 2店舗の待ち時間
def p07():
    n1, s1, n2, s2 = 13, 6.4, 16, 4.0
    F = s1 ** 2 / s2 ** 2
    f025, f05 = r(stats.f.isf(0.025, n1 - 1, n2 - 1), 2), r(stats.f.isf(0.05, n1 - 1, n2 - 1), 2)
    f025b = r(stats.f.isf(0.025, n1, n2), 2)
    assert f05 < F < f025 and f025b > F
    return {
        "n1": n1, "s1": s(s1, 1), "n2": n2, "s2": s(s2, 1), "v1": s(s1 ** 2, 2), "v2": s(s2 ** 2, 2),
        "F": s(F, 2), "Fsd": s(s1 / s2, 2), "df1": n1 - 1, "df2": n2 - 1,
        "f025": s(f025, 2), "f05": s(f05, 2), "f025b": s(f025b, 2),
    }


# ---------------------------------------------------------------- 大問8 アップグレード提案の受諾率
def p08():
    a0, b0 = 1, 2
    m, y = 20, 7
    a1, b1 = a0 + y, b0 + m - y
    pred2 = a1 / (a1 + b1) * (a1 + 1) / (a1 + b1 + 1)
    pm = a1 / (a1 + b1)
    return {
        "a0": a0, "b0": b0, "m": m, "y": y, "a1": a1, "b1": b1, "ab1": a1 + b1, "ab2": a1 + b1 + 1, "a2": a1 + 1,
        "pred": r(pred2, 3), "pred_s": s(pred2, 3), "pred4": s(pred2, 4), "num": a1 * (a1 + 1), "den": (a1 + b1) * (a1 + b1 + 1),
        "pm": s(pm, 4), "pm2": s(pm ** 2, 3), "plug": s((y / m) ** 2, 4), "pm1": s(pm, 3), "my": m - y,
    }


# ---------------------------------------------------------------- 大問9 会員の利用状態
def p09():
    P = [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]]
    q11, q12, q21, q22 = P[0][0], P[0][1], P[1][0], P[1][1]
    det = (1 - q11) * (1 - q22) - q12 * q21
    n11 = (1 - q22) / det
    n12 = q12 / det
    n21 = q21 / det
    n22 = (1 - q11) / det
    m1 = n11 + n12
    m2 = n21 + n22
    return {
        "p11": q11, "p12": q12, "p13": P[0][2], "p21": q21, "p22": q22, "p23": P[1][2],
        "a11": s(1 - q11, 1), "a22": s(1 - q22, 1), "det": s(det, 2), "m1": r(m1, 2), "m1_s": s(m1, 2),
        "m1_4": s(m1, 4), "m2": s(m2, 2), "n11": s(n11, 2), "n12": s(n12, 2), "naive": s(1 / P[0][2], 2),
        "m1_minus1": s(m1 - 1, 2),
    }


# ---------------------------------------------------------------- 大問10 顧客満足度調査
def p10():
    th1, th2, beta = -1.2, 1.0, 0.8
    e = {-2.0: 0.1353, -1.2: 0.3012, -0.4: 0.6703, 0.2: 1.2214, 1.0: 2.7183, 1.8: 6.0496}
    L = lambda z: e[round(z, 1)] / (1 + e[round(z, 1)])
    c1 = L(th1 - beta)
    c2 = L(th2 - beta)
    ans = c2 - c1
    wrong_sign = L(th2 + beta) - L(th1 + beta)
    nonmem = L(th2) - L(th1)
    # (b) RMSEA
    N, chi2, df = 401, 72.4, 24
    rmsea = sqrt(max(chi2 - df, 0) / (df * (N - 1)))
    return {
        "th1": s(th1, 1), "th2": s(th2, 1), "beta": s(beta, 1), "z1": s(th1 - beta, 1), "z2": s(th2 - beta, 1),
        "em20": s(e[-2.0], 4), "em12": s(e[-1.2], 4), "em04": s(e[-0.4], 4), "e02": s(e[0.2], 4),
        "e10": s(e[1.0], 4), "e18": s(e[1.8], 4), "e02p1": s(1 + e[0.2], 4), "em20p1": s(1 + e[-2.0], 4),
        "c1": s(c1, 4), "c2": s(c2, 4), "ans": s(ans, 3), "ans4": s(ans, 4), "wsign": s(wrong_sign, 3),
        "p3": s(1 - c2, 3), "cum2": s(c2, 3), "nonmem": s(nonmem, 3),
        "N": N, "chi2": s(chi2, 1), "dfm": df, "N1": N - 1, "dchi": s(chi2 - df, 1), "dN": df * (N - 1),
        "rm2": s((chi2 - df) / (df * (N - 1)), 7), "rmsea": s(rmsea, 3),
        "rm_nodf": s(sqrt(chi2 / (df * (N - 1))), 3), "rm_nosqrt": s((chi2 - df) / (df * (N - 1)), 3),
        "rm_ratio": s(chi2 / df, 3), "rm_nodfden": s(sqrt((chi2 - df) / (N - 1)), 3),
    }


# ---------------------------------------------------------------- 大問11 客室需要と価格（操作変数）
def p11():
    vz, vp, vq = 0.040, 0.100, 0.250
    czp, czq, cpq = 0.030, -0.048, -0.040
    ols = cpq / vp
    iv = czq / czp
    return {
        "vz": s(vz, 3), "vp": s(vp, 3), "vq": s(vq, 3), "czp": s(czp, 3), "czq": s(czq, 3), "cpq": s(cpq, 3),
        "ols": s(ols, 2), "iv": s(iv, 2), "inv": s(czp / czq, 2), "czq_vp": s(czq / vp, 2), "rf": s(czq / vz, 2),
        "fs": s(czp / vz, 2),
    }


# ---------------------------------------------------------------- 大問12 海辺のカフェ
def p12():
    r12, r1t, r2t = 0.72, 0.80, 0.85
    num = r12 - r1t * r2t
    den2 = (1 - r1t ** 2) * (1 - r2t ** 2)
    pc = num / sqrt(den2)
    # (b) ARMA(1,1)
    phi, th = 0.6, 0.3
    rho1 = (1 + phi * th) * (phi + th) / (1 + 2 * phi * th + th ** 2)
    rho2 = phi * rho1
    return {
        "r12": s(r12, 2), "r1t": s(r1t, 2), "r2t": s(r2t, 2), "prod": s(r1t * r2t, 2), "num": s(num, 2),
        "a1": s(1 - r1t ** 2, 2), "a2": s(1 - r2t ** 2, 4), "den2": s(den2, 4), "den": s(sqrt(den2), 4),
        "pc": r(pc, 2), "pc_s": s(pc, 2), "pc4": s(pc, 4), "pc_nosqrt": s(num / den2, 2),
        "phi": s(phi, 1), "th": s(th, 1), "pt": s(phi * th, 2), "ppt": s(phi + th, 1), "n1": s(1 + phi * th, 2),
        "d1": s(1 + 2 * phi * th + th ** 2, 2), "rnum": s((1 + phi * th) * (phi + th), 3),
        "rho1": s(rho1, 4), "rho1_3": s(rho1, 3), "rho2": s(rho2, 3), "rho2_4": s(rho2, 4),
        "ar2": s(phi ** 2, 3), "naive": s(phi * (phi + th), 3),
        "wd": s(phi * (1 + phi * th) * (phi + th) / (1 + th ** 2), 3), "th2": s(th ** 2, 2), "tpt": s(2 * phi * th, 2),
    }


# ---------------------------------------------------------------- 大問13 宿泊客アンケート
def p13():
    N, n = 2000, 100
    X, xb, yb, b = 2.4, 2.2, 38.0, 12.0
    reg = yb + b * (X - xb)
    ratio = yb / xb * X
    # (b) 多重代入
    Q = [6.2, 6.5, 6.0, 6.4, 6.4]
    U = [0.040, 0.045, 0.038, 0.042, 0.035]
    m = len(Q)
    qb = sum(Q) / m
    W = sum(U) / m
    B = sum((q - qb) ** 2 for q in Q) / (m - 1)
    T = W + (1 + 1 / m) * B
    return {
        "N": N, "n": n, "X": s(X, 1), "xb": s(xb, 1), "yb": s(yb, 1), "b": s(b, 1), "dx": s(X - xb, 1),
        "adj": s(b * (X - xb), 1), "reg": s(reg, 1), "ratio": s(ratio, 1), "rev": s(yb - b * (X - xb), 1),
        "bx": s(b * X, 1), "ratio_r": s(yb / xb, 4),
        **{f"q{i+1}": s(v, 1) for i, v in enumerate(Q)}, **{f"u{i+1}": s(v, 3) for i, v in enumerate(U)},
        **{f"dv{i+1}": s(v - qb, 1) for i, v in enumerate(Q)}, "m1": m - 1,
        "m": m, "qb": s(qb, 2), "W": s(W, 3), "ssq": s(sum((q - qb) ** 2 for q in Q), 2), "B": s(B, 3),
        "fac": s(1 + 1 / m, 1), "fB": s((1 + 1 / m) * B, 3), "T": s(T, 3), "se": r(sqrt(T), 3), "se_s": s(sqrt(T), 3),
        "se4": s(sqrt(T), 4), "se_nof": s(sqrt(W + B), 3), "se_bm": s(sqrt(W + (1 + 1 / m) * B * (m - 1) / m), 3),
        "se_w": s(sqrt(W), 3), "Bm": s(B * (m - 1) / m, 3),
    }


# ---------------------------------------------------------------- 大問14 口コミ評価項目
def p14():
    ev = [2.9, 1.4, 0.8, 0.5, 0.3, 0.1]
    p = len(ev)
    cum = [sum(ev[:k]) / p for k in range(1, p + 1)]
    kaiser = sum(1 for v in ev if v >= 1)
    k80 = next(i + 1 for i, c in enumerate(cum) if c >= 0.8)
    assert kaiser == 2 and k80 == 3
    # (b) 回転
    l1 = (0.70, 0.30)
    c, sn = 0.8, 0.6
    rot = (l1[0] * c + l1[1] * sn, -l1[0] * sn + l1[1] * c)
    rotT = (l1[0] * c - l1[1] * sn, l1[0] * sn + l1[1] * c)
    comm = l1[0] ** 2 + l1[1] ** 2
    return {
        **{f"ev{i+1}": s(v, 1) for i, v in enumerate(ev)}, "p": p,
        "cum1": s(cum[0], 3), "cum2": s(cum[1], 3), "cum3": s(cum[2], 3), "s2": s(sum(ev[:2]), 1), "s3": s(sum(ev[:3]), 1),
        "c12": s(ev[0] / (ev[0] + ev[1]), 3),
        "l11": s(l1[0], 2), "l12": s(l1[1], 2), "c": s(c, 1), "sn": s(sn, 1),
        "r1": s(rot[0], 2), "r2": s(rot[1], 2), "rt1": s(rotT[0], 2), "rt2": s(rotT[1], 2),
        "comm": s(comm, 2), "uniq": s(1 - comm, 2), "rsum": s(rot[0] + rot[1], 2),
        "e1": s(l1[0] * c, 2), "e2": s(l1[1] * sn, 2), "e3": s(l1[0] * sn, 2), "e4": s(l1[1] * c, 2),
        "r1sq": s(rot[0] ** 2, 4), "r2sq": s(rot[1] ** 2, 4), "w1": s(l1[0] * c, 2), "w2": s(l1[1] * c, 2),
        "wcomm": s((l1[0] * c) ** 2 + (l1[1] * c) ** 2, 2),
    }


def values():
    return {
        "s17p01": p01(), "s17p02": p02(), "s17p03": p03(), "s17p04": p04(), "s17p05": p05(), "s17p06": p06(),
        "s17p07": p07(), "s17p08": p08(), "s17p09": p09(), "s17p10": p10(), "s17p11": p11(), "s17p12": p12(),
        "s17p13": p13(), "s17p14": p14(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
