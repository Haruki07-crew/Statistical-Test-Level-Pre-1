"""模擬試験 第23回（set23）の数値。values() が {大問id: {名前: 値}} を返す。

題材：建設・土木・インフラ点検・防災。
2026-10-24 夕方：公開済みのセットと同じ型が 21 問あったため、p02a・p05・p11a 以外を作り直した。
問題文に示した数表の値（Φ・t・χ² の点、ln・e^x・10^x の値）は、ここでも同じ丸めた値を使って答えを計算する。
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


def p02():
    # (a) AIC：y の正規モデルと ln y の正規モデル（対数正規）
    n = 10
    lA, lB, S = -72.84, 3.58, 75.91        # 問題文に示す値
    aicA = -2 * lA + 2 * 2
    aicBlog = -2 * lB + 2 * 2
    aicB = aicBlog + 2 * S
    # (b) 平均と変動係数から対数正規の母数
    m, cv, load = 2000, 0.20, 1400
    ln104, ln07, sig = 0.0392, -0.3567, 0.1980   # ln(1.04), ln(0.7), sqrt(0.0392)
    half = ln104 / 2
    z = r((ln07 + half) / sig, 2)
    z_noshift = r(ln07 / sig, 2)
    z_plus = r((ln07 - half) / sig, 2)
    z_norm = r((load - m) / (cv * m), 2)
    phi = {1.50: 0.9332, 1.70: 0.9554, 1.80: 0.9641, 1.90: 0.9713}
    for zz in (z, z_noshift, z_plus, z_norm):
        assert -zz in phi, zz
    pr = lambda zz: 1 - phi[-zz]
    return {
        "n": n, "lA": lA, "lB": s(lB, 2), "S": S, "aicA": s(aicA, 2), "m2lA": s(-2 * lA, 2), "m2lB": s(-2 * lB, 2),
        "aicBlog": s(aicBlog, 2), "twoS": s(2 * S, 2), "aicB": s(aicB, 2), "dAIC": s(aicA - aicB, 2),
        "aicB_sign": s(aicBlog - 2 * S, 2), "aicB_one": s(aicBlog + S, 2),
        "m": m, "cv": s(cv, 2), "cv2": s(cv ** 2, 2), "load": load, "ln104": ln104, "ln07": ln07, "sig": sig,
        "half": s(half, 4), "num": s(ln07 + half, 4), "zraw": s((ln07 + half) / sig, 4), "z": s(z, 2), "za": s(-z, 2),
        "phiz": s(phi[-z], 4), "ans": s(pr(z), 3), "ans4": s(pr(z), 4),
        "z_noshift": s(z_noshift, 2), "c_noshift": s(pr(z_noshift), 3),
        "c_comp": s(phi[-z], 3), "z_plus": s(z_plus, 2), "c_plus": s(pr(z_plus), 3),
        "z_norm": s(z_norm, 2), "c_norm": s(pr(z_norm), 3), "sdnorm": s(cv * m, 0),
        "ph150": s(phi[1.50], 4), "ph170": s(phi[1.70], 4), "ph180": s(phi[1.80], 4), "ph190": s(phi[1.90], 4),
    }


def p05():
    # (a) 正規分布の尺度混合の尖度
    w, s1, s2 = 0.9, 1, 3
    m2 = w * s1 ** 2 + (1 - w) * s2 ** 2
    m4 = 3 * (w * s1 ** 4 + (1 - w) * s2 ** 4)
    kurt = m4 / m2 ** 2
    # (b) 重み (1,2,1)/4 の移動平均 → MA(2)
    a = (1, 2, 1)
    g0 = sum(x * x for x in a)
    g1 = a[0] * a[1] + a[1] * a[2]
    g2 = a[0] * a[2]
    return {
        "w": w, "w2": s(1 - w, 1), "s1": s1, "s2": s2, "v1": s1 ** 2, "v2": s2 ** 2, "q1": s1 ** 4, "q2": s2 ** 4,
        "m2": s(m2, 1), "m4": s(m4, 1), "inner": s(w * s1 ** 4 + (1 - w) * s2 ** 4, 1), "m2sq": s(m2 ** 2, 2),
        "kurt": s(kurt, 2), "ex": s(kurt - 3, 2), "w_no3": s(m4 / 3 / m2 ** 2, 2), "w_var": s(m4 / m2, 2),
        "g0": g0, "g1": g1, "g2": g2, "r1": s(g1 / g0, 3), "r2": s(g2 / g0, 3),
        "ar2": s((g1 / g0) ** 2, 3), "ar3": s((g1 / g0) ** 3, 3),
        "b1": s(g1 / 16, 3), "b2": s(g2 / 16, 3), "h1": s(2 / g0, 3),
    }


def p11():
    # (a) ログランク検定
    A = [(5, 1), (7, 1), (9, 0), (11, 1)]
    B = [(8, 1), (10, 0), (12, 1), (14, 0)]
    ev = sorted({t for t, d in A + B if d})
    rows = []
    O = E = V = 0.0
    for t in ev:
        na = sum(1 for u, _ in A if u >= t)
        nb = sum(1 for u, _ in B if u >= t)
        da = sum(1 for u, d in A if u == t and d)
        db = sum(1 for u, d in B if u == t and d)
        nn, dd = na + nb, da + db
        e = na * dd / nn
        v = na * nb * dd * (nn - dd) / (nn ** 2 * (nn - 1)) if nn > 1 else 0.0
        rows.append((t, na, nb, da, db, e, v))
        O += da
        E += e
        V += v
    chi = (O - E) ** 2 / V
    out = {"O": int(O), "E": s(E, 4), "OE": s(O - E, 4), "OE2": s((O - E) ** 2, 4), "V": s(V, 4), "chi": r(chi, 2), "chi4": s(chi, 4),
           "chi_E": s((O - E) ** 2 / E, 2)}
    EB = sum(r_[2] * (r_[3] + r_[4]) / (r_[1] + r_[2]) for r_ in rows)
    OB = sum(d for _, d in B)
    out["EB"] = s(EB, 4)
    out["OB"] = OB
    out["chi_pear"] = s((O - E) ** 2 / E + (OB - EB) ** 2 / EB, 2)
    for i, (t, na, nb, da, db, e, v) in enumerate(rows):
        out[f"t{i}"] = t
        out[f"na{i}"] = na
        out[f"nb{i}"] = nb
        out[f"da{i}"] = da
        out[f"db{i}"] = db
        out[f"e{i}"] = s(e, 4)
        out[f"v{i}"] = s(v, 4)
    # (b) ガンマ回帰（対数リンク）
    b0, b1, b2, x2, phi = 0.40, 0.405, 0.10, 4, 0.16
    eta = b0 + b1 + b2 * x2
    e_eta, e_no = 3.337, 2.226     # e^{1.205}, e^{0.800}
    mu = e_eta
    out.update({"b0": s(b0, 2), "b1": s(b1, 3), "b2": s(b2, 2), "x2": x2, "phi": s(phi, 2), "eta": s(eta, 3),
                "e_eta": s(e_eta, 3), "e_no": s(e_no, 3), "eta_no": s(b0 + b2 * x2, 3), "rphi": s(sqrt(phi), 1),
                "mu": s(mu, 2), "sdY": s(sqrt(phi) * mu, 2), "varY": s(phi * mu ** 2, 3), "mu2": s(mu ** 2, 3),
                "w_phi": s(phi * mu, 2), "w_pois": s(sqrt(phi * mu), 2), "w_sqrt": s(sqrt(mu), 2),
                "mu_no": s(e_no, 2), "sd_no": s(sqrt(phi) * e_no, 2), "eb1": "1.50"})
    return out

def p01():
    m = 12
    k = 60 - m
    ans = 1 - (k / 60) ** 2
    return {"m": m, "k": k, "k2": s((k / 60) ** 2, 4), "ans": r(ans, 3), "ans_s": s(ans, 3), "w_lin": s(2 * m / 60, 3)}


def p02b():
    rho = 0.5
    ok = 0.25 + math.asin(rho) / (2 * math.pi)
    return {"rho": s(rho, 1), "ok": s(ok, 3), "ind": "0.250", "lin": s(0.25 + rho / 4, 3),
            "no2": s(0.25 + math.asin(rho) / math.pi, 3), "half": "0.500"}


def p03():
    xb, v, mn, mx = 0.40, 0.12, -0.15, 0.95
    hw = sqrt(3 * v)
    out = {"xb": s(xb, 2), "v": s(v, 2), "mn": s(mn, 2), "mx": s(mx, 2), "wd": s(2 * hw, 2), "hw": s(hw, 2),
           "ok_a": s(xb - hw, 2), "ok_b": s(xb + hw, 2)}
    for k_, h in (("w1", sqrt(v)), ("w2", sqrt(12 * v)), ("w3", sqrt(v / 3))):
        out[k_ + "_a"] = s(xb - h, 2); out[k_ + "_b"] = s(xb + h, 2)
    assert xb - hw < mn and mx < xb + hw
    return out


def p04():
    l, el = 2, 7.389
    ans = 1 / (1 + el)
    # (b) 指数分布の 90% 点の推定
    th, n = 2.0, 400
    ln10 = 2.3026
    q = th * ln10
    fq = (1 / th) * 0.1
    se1 = sqrt(0.9 * 0.1 / (n * fq ** 2))
    se2 = ln10 * th / sqrt(n)
    return {"l": l, "el": el, "ans": r(ans, 3), "ans_s": s(ans, 3), "ans4": s(ans, 4), "w_inv": s(1 / el, 3),
            "th": s(th, 1), "n": n, "q": s(q, 3), "fq": s(fq, 2), "rn": s(sqrt(n), 0),
            "se1": s(se1, 3), "se1_4": s(se1, 4), "se2": s(se2, 3), "se2_4": s(se2, 4), "are": s((se2 / se1) ** 2, 2),
            "w1": s(sqrt(0.09 / (n * (1 / th) ** 2)), 3), "w2": s(th / sqrt(n), 3), "w3": s(sqrt(0.9 / (n * fq ** 2)), 3)}


def p06():
    n, sd, df = 6, 2.4, 5
    t1, t2, t3, rn = 2.571, 2.912, 3.534, 2.449
    hw = t3 * sd / rn
    A, B = [30.1, 31.4, 32.0], [32.5, 33.0, 34.6, 35.2]
    d = sorted(r(b - a, 1) for a in A for b in B)
    hl = (d[5] + d[6]) / 2
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2
    out = {"n": n, "s": s(sd, 1), "df": df, "t1": t1, "t2": t2, "t3": t3, "rn": rn, "hw": r(hw, 2), "hw_s": s(hw, 2), "hw4": s(hw, 4),
           "w1": s(t1 * sd / rn, 2), "w2": s(t2 * sd / rn, 2),
           "a": "、".join(s(v, 1) for v in A), "b": "、".join(s(v, 1) for v in B),
           "dlist": ",\\ ".join(s(v, 1) for v in d), "d6": s(d[5], 1), "d7": s(d[6], 1), "ok": s(hl, 2),
           "w_med": s(med(B) - med(A), 2), "w_mean": s(sum(B) / 4 - sum(A) / 3, 2), "w_lo": s(d[5], 2), "w_hi": s(d[6], 2)}
    assert len({out["ok"], out["w_med"], out["w_mean"], out["w_lo"], out["w_hi"]}) == 5
    return out


def p07():
    Y = [[52, 55, 50], [58, 60, None], [49, 53, 47], [56, 59, 54]]
    a, b = 4, 3
    T = sum(v for v in Y[1] if v is not None)
    B = sum(row[2] for row in Y if row[2] is not None)
    G = sum(v for row in Y for v in row if v is not None)
    xh = (a * T + b * B - G) / ((a - 1) * (b - 1))
    out = {"T": T, "B": B, "G": G, "num": a * T + b * B - G, "xh": r(xh, 2), "xh_s": s(xh, 2), "xh4": s(xh, 4),
           "w_swap": s((b * T + a * B - G) / ((a - 1) * (b - 1)), 2), "w_row": s(T / 2, 2)}
    for i, row in enumerate(Y):
        for j, v in enumerate(row):
            if v is not None:
                out[f"y{i + 1}{j + 1}"] = v
    # (b) 正規分布への適合度
    O = [3, 27, 30, 28, 12]; E = [10.0, 23.0, 34.0, 23.0, 10.0]
    X = sum((o - e) ** 2 / e for o, e in zip(O, E))
    assert 5.991 < X < 7.815
    out.update({f"o{i + 1}": O[i] for i in range(5)}); out.update({f"e{i + 1}": s(E[i], 1) for i in range(5)})
    out.update({"X": s(X, 2), "X4": s(X, 4), "w_abs": s(sum(abs(o - e) for o, e in zip(O, E)), 2),
                "terms": "+".join(f"\\frac{{({o}-{s(e, 1)})^2}}{{{s(e, 1)}}}" for o, e in zip(O, E))})
    return out


def p08():
    return {}


def p09():
    C = [[60, 15, 5], [0, 40, 10], [0, 0, 20]]
    q = [0.5, 0.3, 0.2]
    rs = [sum(rw) for rw in C]
    P = [[c / r_ for c in rw] for rw, r_ in zip(C, rs)]
    nxt = [sum(q[i] * P[i][j] for i in range(3)) for j in range(3)]
    two = [sum(nxt[i] * P[i][j] for i in range(3)) for j in range(3)]
    out = {"q1": s(q[0], 1), "q2": s(q[1], 1), "q3": s(q[2], 1), "q3c": s(q[2], 3), "r1": rs[0], "r2": rs[1], "r3": rs[2],
           "p13": s(P[0][2], 4), "p23": s(P[1][2], 1), "ok": s(nxt[2], 3), "ok4": s(nxt[2], 5),
           "w_pool": s(sum(rw[2] for rw in C) / sum(rs), 3), "w_noabs": s(q[0] * P[0][2] + q[1] * P[1][2], 3), "w_2step": s(two[2], 3)}
    for i in range(3):
        for j in range(3):
            out[f"c{i + 1}{j + 1}"] = C[i][j]
    assert len({out["ok"], out["w_pool"], out["w_noabs"], out["w_2step"], out["q3c"]}) == 5
    return out


def p10():
    x = [1, 2, 3, 4, 5, 6]; y = [2.1, 4.3, 5.8, 8.4, 9.9, 12.3]
    n = len(x)
    sxx = sum(v * v for v in x); sxy = sum(a * b for a, b in zip(x, y)); syy = sum(v * v for v in y)
    b = sxy / sxx; rss = syy - b * sxy; s2 = rss / (n - 1); se = sqrt(s2 / sxx)
    out = {"n": n, "sxx": sxx, "sxy": s(sxy, 1), "syy": s(syy, 2), "b5": s(b, 5), "rss": s(rss, 4), "s2": s(s2, 5),
           "se": r(se, 4), "se_s": s(se, 4), "se5": s(se, 5), "w_n2": s(sqrt(rss / (n - 2) / sxx), 4)}
    # (b) ダービン・ワトソン比
    e = [1.2, 0.9, 1.1, 0.3, -0.4, -1.0, -0.8, -1.3]
    dif = [e[t] - e[t - 1] for t in range(1, 8)]
    num = sum(d * d for d in dif); den = sum(v * v for v in e)
    r1 = sum(e[t] * e[t - 1] for t in range(1, 8)) / den
    out.update({"elist": ",\\ ".join(s(v, 1) for v in e), "dlist": ",\\ ".join(s(d, 1) for d in dif), "num": s(num, 2), "den": s(den, 2),
                "dw": s(num / den, 2), "dw4": s(num / den, 4), "r1": s(r1, 3), "apx": s(2 * (1 - r1), 3),
                "w_inv": s(den / num, 2), "w_noSq": s(sum(abs(d) for d in dif) / den, 2)})
    # (c) 非線形回帰
    t = [7, 14, 28]; yy = [22, 31, 38]
    ex = {0.7: 0.4966, 1.4: 0.2466, 2.8: 0.0608, 1.05: 0.3499, 2.1: 0.1225, 4.2: 0.0150}
    res = {}
    for th in (0.10, 0.15):
        g = [1 - ex[round(th * tt, 2)] for tt in t]
        a_ = sum(u * v for u, v in zip(yy, g)) / sum(v * v for v in g)
        res[th] = (g, a_, sum((u - a_ * v) ** 2 for u, v in zip(yy, g)))
    best = min(res, key=lambda k_: res[k_][2])
    assert best == 0.10
    ga, ta, ra = res[0.10]; gb, tb, rb = res[0.15]
    out.update({"t1": t[0], "t2": t[1], "t3": t[2], "y1": yy[0], "y2": yy[1], "y3": yy[2], "a": "0.10", "b": "0.15",
                "ga1": s(ga[0], 4), "ga2": s(ga[1], 4), "ga3": s(ga[2], 4), "gb1": s(gb[0], 4), "gb2": s(gb[1], 4), "gb3": s(gb[2], 4),
                "ta": s(ta, 2), "tb": s(tb, 2), "ra": s(ra, 2), "rb": s(rb, 2), "ok": s(ta, 1), "wb": s(tb, 1),
                "w_ratio": s(sum(u / v for u, v in zip(yy, ga)) / 3, 1), "w_last": s(yy[2] / ga[2], 1)})
    assert len({out["ok"], out["wb"], out["w_ratio"], out["w_last"]}) == 4
    return out


def p11b():
    xs = [0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.55, 0.70]
    ys = [0, 0, 0, 0, 1, 1, 1, 1]
    m1 = sum(x for x, y in zip(xs, ys) if y) / 4; m0 = sum(x for x, y in zip(xs, ys) if not y) / 4
    return {"xs": " | ".join(s(v, 2) for v in xs), "sep": "---|" * len(xs), "ys": " | ".join(map(str, ys)), "d": s(m1 - m0, 4)}


def p12():
    N, n, yb, sd = 400, 25, 120.0, 40.0
    f = n / N
    inner = sqrt((1 - f) * sd * sd / n)
    se = N * inner
    out = {"N": N, "n": n, "yb": s(yb, 1), "s": s(sd, 1), "tot": s(N * yb, 0), "f": s(f, 4), "inner": s(inner, 4),
           "se1": s(se, 1), "se": r(se, 0), "se_s": s(se, 0), "w_nofpc": s(N * sd / sqrt(n), 0)}
    # (b) 正規・正規モデル
    m0, t0, sig, nn, xb, c = 30.0, 4.0, 6.0, 9, 34.0, 32.0
    pp, pd = 1 / t0 ** 2, nn / sig ** 2
    ppost = pp + pd
    mpost = (pp * m0 + pd * xb) / ppost
    sdp = sqrt(1 / ppost)
    z = (c - mpost) / sdp
    ph = {0.5: 0.6915, 0.6: 0.7257, 0.67: 0.7486, 1.0: 0.8413}
    zr = r(-z, 2)
    ok = ph[zr]
    out.update({"m0": s(m0, 1), "t0": s(t0, 1), "sig": s(sig, 1), "nn": nn, "xb": s(xb, 1), "c": s(c, 1),
                "pp": s(pp, 4), "pd": s(pd, 2), "ppost": s(ppost, 4), "vpost": s(1 / ppost, 1), "sdpost": s(sdp, 4),
                "mpost": s(mpost, 1), "z": s(z, 2), "ok": s(ok, 3), "ok4": s(ok, 4), "se_d": s(sig / sqrt(nn), 1),
                "w_prior": s(1 - ph[0.5], 3), "w_data": s(ph[1.0], 3), "w_sd": s(ph[r((mpost - c) / (sig / sqrt(nn)), 1)], 3),
                "w_comp": s(1 - ok, 3)})
    assert zr == 0.67
    return out


def p13():
    xs = [1.0, 1.6, 2.1, 2.75, 3.15, 3.65, 4.4, 5.2]
    lab = list("HHHDHDDD")
    def loo(k):
        wrong = []
        for i in range(8):
            ds = sorted((abs(xs[j] - xs[i]), j) for j in range(8) if j != i)
            assert ds[k - 1][0] + 1e-9 < ds[k][0]
            votes = [lab[j] for _, j in ds[:k]]
            pred = max(set(votes), key=votes.count)
            if pred != lab[i]:
                wrong.append(i)
        return wrong
    w1, w3 = loo(1), loo(3)
    nm = lambda idx: "、".join(f"$x={s(xs[i], 2)}$（{lab[i]}）" for i in idx)
    out = {"xs": " | ".join(s(v, 2) for v in xs), "sep": "---|" * 8, "labs": " | ".join(lab),
           "n1": len(w1), "n3": len(w3), "e1": s(len(w1) / 8, 3), "e3": s(len(w3) / 8, 3), "list1": nm(w1), "list3": nm(w3),
           "w3": s(3 / 8, 3), "w1": s(2 / 8, 3)}
    assert (len(w1), len(w3)) == (3, 2)
    # (b) クーポン収集
    ok = 4 * (1 + 1 / 2 + 1 / 3 + 1 / 4)
    out.update({"ok": s(ok, 2), "ok4": s(ok, 4), "w_ln": s(4 * math.log(4), 2), "w_n2": "16.00", "w_drop": s(ok - 4, 2)})
    return out


def p14():
    ok = 1 + 2 + 4 + 2
    cy = 1.8
    single = 1 + cy * cy
    ward = (2 * single + 2 * single - 4) / 3
    assert cy * cy < 4 < single
    return {"ok": ok, "w_edges": 4, "w_noroot": ok - 1, "w_full": 2 * (1 + 2 + 4 + 2), "cy": s(cy, 1),
            "ok2": s(cy * cy, 2), "w_single": s(single, 2), "w_ward": s(ward, 2)}


def values():
    v = {"s23p01": p01(), "s23p02": p02(), "s23p03": p03(), "s23p04": p04(), "s23p05": p05(),
         "s23p06": p06(), "s23p07": p07(), "s23p08": p08(), "s23p09": p09(), "s23p10": p10(),
         "s23p11": p11(), "s23p12": p12(), "s23p13": p13(), "s23p14": p14()}
    v["s23p02"].update({"nb_" + k: x for k, x in p02b().items()})
    v["s23p11"].update({"nl_" + k: x for k, x in p11b().items()})
    return v


if __name__ == "__main__":
    import pprint
    pprint.pprint(values())
