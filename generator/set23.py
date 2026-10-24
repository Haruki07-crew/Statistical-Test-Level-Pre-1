"""模擬試験 第23回（set23）の数値。values() が {大問id: {名前: 値}} を返す。

題材：建設・土木・インフラ点検・防災。
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


def p01():
    # ブリッジ型の道路網：区間1 S-A, 2 S-B, 3 A-B, 4 A-T, 5 B-T
    p1, p2, p3, p4, p5 = 0.9, 0.8, 0.7, 0.8, 0.9
    up = (1 - (1 - p1) * (1 - p2)) * (1 - (1 - p4) * (1 - p5))
    down = 1 - (1 - p1 * p4) * (1 - p2 * p5)
    ans = p3 * up + (1 - p3) * down
    return {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "q3": s(1 - p3, 1),
        "u1": s(1 - (1 - p1) * (1 - p2), 2), "u2": s(1 - (1 - p4) * (1 - p5), 2), "up": s(up, 4),
        "a14": s(p1 * p4, 2), "a25": s(p2 * p5, 2), "b14": s(1 - p1 * p4, 2), "b25": s(1 - p2 * p5, 2), "down": s(down, 4),
        "t_up": s(p3 * up, 5), "t_down": s(r((1 - p3) * down, 5), 5), "ans": r(ans, 3), "ans5": s(ans, 5),
        "w_up": s(up, 3), "w_down": s(down, 3),
    }


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


def p03():
    n, sh2 = 25, 60.5
    th = sh2 / (2 * n)
    se = th / sqrt(n)
    return {"n": n, "sh2": sh2, "twon": 2 * n, "th": s(th, 2), "rootn": s(sqrt(n), 0), "se": r(se, 2), "se4": s(se, 4),
            "w_n": s(th / n, 4), "w_sqrt2": s(th / sqrt(2 * n), 4), "w_sig": s(sqrt(th) / sqrt(n), 4),
            "info": s(n / th ** 2, 4), "sig": s(sqrt(th), 2)}


def p04():
    # (a) 一様分布の積 D=XY
    d, lnd = 0.2, -1.6094
    cdf = d - d * lnd
    ans = 1 - cdf
    # (b) 最弱リンクの寸法効果
    m, k = 10, 10
    t1, t05, t2 = 0.7943, 0.8913, 0.3162   # 10^{-0.1}, 10^{-0.05}, 10^{-0.5}
    return {
        "d": d, "lnd": lnd, "dlnd": s(-d * lnd, 5), "cdf": s(cdf, 5), "ans": s(ans, 3), "ans5": s(ans, 5),
        "c_cdf": s(cdf, 3), "c_indep": s((1 - d) ** 2, 3), "c_part": s(1 + d * lnd, 3), "c_unif": s(1 - d, 3),
        "m": m, "k": k, "t1": s(t1, 4), "t05": s(t05, 4), "t2": s(t2, 4),
        "ratio": s(t1, 3), "w05": s(t05, 3), "w2": s(t2, 3), "w_inv": s(1 / k, 3), "w_one": s(1, 3),
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


def p06():
    x = [33.1, 31.4, 34.6, 29.2, 32.8, 35.3]
    n = len(x)
    mean = sum(x) / n
    sd = sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
    mean_s, sd_s = r(mean, 3), r(sd, 3)
    t = 2.571   # t_{0.025}(5)
    fac = sqrt(1 + 1 / n)
    lo = mean_s - t * sd_s * fac
    lo_ci = mean_s - t * sd_s / sqrt(n)
    lo_z = mean_s - 1.960 * sd_s * fac
    lo_t1 = mean_s - 2.015 * sd_s * fac
    # (b) 符号付き順位検定（正確な片側 p 値）
    from itertools import combinations
    d = [round(v - 30, 1) for v in x]
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = {i: k + 1 for k, i in enumerate(order)}
    wplus = sum(rank[i] for i in range(n) if d[i] > 0)
    cnt = sum(1 for k in range(n + 1) for c in combinations(range(1, n + 1), k) if sum(c) >= wplus)
    eq = sum(1 for k in range(n + 1) for c in combinations(range(1, n + 1), k) if sum(c) == wplus)
    ge19 = sum(1 for k in range(n + 1) for c in combinations(range(1, n + 1), k) if sum(c) >= wplus - 1)
    pos = sum(1 for v in d if v > 0)
    sign_p = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    out = {
        "n": n, "n1": n - 1, "sum": s(sum(x), 1), "mean": s(mean_s, 3), "sd": s(sd_s, 3), "t": t, "t1": 2.015,
        "fac": s(fac, 4), "half": s(t * sd_s * fac, 4), "lo": r(lo, 2), "lo4": s(lo, 4),
        "lo_ci": s(lo_ci, 2), "lo_z": s(lo_z, 2), "lo_t1": s(lo_t1, 2), "ss": s(sd_s ** 2 * (n - 1), 3),
        "wplus": wplus, "cnt": cnt, "tot": 2 ** n, "p": s(cnt / 2 ** n, 3), "p5": s(cnt / 2 ** n, 5),
        "p_eq": s(eq / 2 ** n, 3), "p_two": s(2 * cnt / 2 ** n, 3), "p_ge19": s(ge19 / 2 ** n, 3), "wm1": wplus - 1,
        "p_sign": s(sign_p, 3), "pos": pos, "sign_cnt": sum(math.comb(n, k) for k in range(pos, n + 1)),
        "nrank": n * (n + 1) // 2,
    }
    for i, v in enumerate(x):
        out[f"x{i + 1}"] = v
        out[f"d{i + 1}"] = s(d[i], 1)
        out[f"k{i + 1}"] = rank[i]
    return out


def p07():
    # (b) 二項検定の正確な片側 p 値
    n, x, p0 = 20, 3, 0.05
    q20, q19, q18 = 0.3585, 0.3774, 0.3972
    P0 = q20
    P1 = n * p0 * q19
    P2 = math.comb(n, 2) * p0 ** 2 * q18
    pv = 1 - P0 - P1 - P2
    return {"n": n, "x": x, "p0": p0, "q20": q20, "q19": q19, "q18": q18, "c2": math.comb(n, 2),
            "P0": s(P0, 4), "P1": s(P1, 4), "P2": s(P2, 5), "sumP": s(P0 + P1 + P2, 5), "pv": r(pv, 3), "pv5": s(pv, 5),
            "np0": s(n * p0, 0), "p02": s(p0 ** 2, 4), "w_le2": s(P0 + P1 + P2, 3)}


def p08():
    th0, th1, n = 40, 20, 5
    df = 2 * n
    q05 = 3.940          # χ²_{10} の下側 5% 点
    q05_5 = 1.145        # χ²_{5} の下側 5% 点
    F788, F197, F5_229 = 0.3594, 0.0034, 0.1923
    c = th0 * q05 / 2
    x = 2 * c / th1
    assert abs(x - 7.88) < 1e-9
    assert abs(q05 * th1 / th0 - 1.97) < 1e-9
    c5 = th0 * q05_5 / 2
    assert abs(2 * c5 / th1 - 2.29) < 1e-9
    return {"th0": th0, "th1": th1, "n": n, "df": df, "q05": s(q05, 3), "q05_5": s(q05_5, 3),
            "F788": s(F788, 4), "F197": s(F197, 4), "F5": s(F5_229, 4), "c": s(c, 1), "x": s(x, 2),
            "ans": s(F788, 3), "w_comp": s(1 - F788, 3), "w_inv": s(F197, 3), "w_df": s(F5_229, 3), "alpha": "0.050",
            "c5": s(c5, 1), "x197": s(q05 * th1 / th0, 2), "x229": s(2 * c5 / th1, 2)}


def p09():
    from fractions import Fraction as Fr
    P = [[Fr(7, 10), Fr(25, 100), Fr(5, 100)], [Fr(0), Fr(6, 10), Fr(4, 10)], [Fr(6, 10), Fr(0), Fr(4, 10)]]
    p = lambda i, j: P[i][j]
    # 3年目に初めて状態3（経路で列挙）
    paths = {"113": p(0, 0) * p(0, 0) * p(0, 2), "123": p(0, 0) * p(0, 1) * p(1, 2), "223": p(0, 1) * p(1, 1) * p(1, 2)}
    f3 = sum(paths.values())
    f1 = p(0, 2)
    f2 = p(0, 0) * p(0, 2) + p(0, 1) * p(1, 2)

    def mul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    P3 = mul(mul(P, P), P)[0][2]
    return {"p11": s(p(0, 0), 2), "p12": s(p(0, 1), 2), "p13": s(p(0, 2), 2), "p22": s(p(1, 1), 2), "p23": s(p(1, 2), 2),
            "p31": s(p(2, 0), 2), "p33": s(p(2, 2), 2),
            "a113": s(paths["113"], 4), "a123": s(paths["123"], 4), "a223": s(paths["223"], 4),
            "f3": s(f3, 4), "f1": s(f1, 4), "f2": s(f2, 4), "cum": s(f1 + f2 + f3, 4), "P3": s(P3, 4),
            "only123": s(paths["123"] + paths["223"], 4), "extra": s(P3 - f3, 4)}


def p10():
    n, xbar, sxx, s2 = 12, 30, 2000, 1.44
    sd = sqrt(s2)
    x0 = 40
    t10, t11, t05 = 2.228, 2.201, 1.812
    a = 1 / n + (x0 - xbar) ** 2 / sxx
    hw = t10 * sd * sqrt(a)
    # (b) クックの距離
    pp = 2
    br = {"A": (3.0, 0.08), "B": (1.2, 0.45), "C": (2.0, 0.20)}
    D = {k: e * e * h / (pp * s2 * (1 - h) ** 2) for k, (e, h) in br.items()}
    Dw = {k: e * e * h / (pp * s2 * (1 - h)) for k, (e, h) in br.items()}
    best = max(D, key=D.get)
    assert best == "B"
    # (c) ニューラルネットワーク
    kin, kh = 5, 8
    out = {
        "n": n, "xbar": xbar, "sxx": sxx, "s2": s2, "sd": s(sd, 1), "x0": x0, "dx2": (x0 - xbar) ** 2, "df": n - 2,
        "t10": t10, "t11": t11, "t05": t05, "inv_n": s(1 / n, 4), "lev": s((x0 - xbar) ** 2 / sxx, 4), "a": s(a, 4),
        "ra": s(sqrt(a), 4), "hw": s(hw, 3), "hw4": s(hw, 4), "pw": s(t10 * sd * sqrt(1 + a), 3), "pa": s(1 + a, 4),
        "w_t11": s(t11 * sd * sqrt(a), 3), "w_t05": s(t05 * sd * sqrt(a), 3), "w_non": s(t10 * sd * sqrt((x0 - xbar) ** 2 / sxx), 3),
        "pp": pp, "ps2": s(pp * s2, 2),
        "kin": kin, "kh": kh, "w1": kin * kh, "b1": kh, "w2": kh, "b2": 1,
        "nn": kin * kh + kh + kh + 1, "nn_nob": kin * kh + kh, "nn_ob": kin * kh + kh + 1, "nn_nob2": kin * kh + kh + kh,
        "nn_first": kin * kh,
    }
    for k, (e, h) in br.items():
        out[f"e{k}"] = s(e, 1)
        out[f"h{k}"] = s(h, 2)
        out[f"D{k}"] = s(D[k], 2)
        out[f"Dw{k}"] = s(Dw[k], 2)
        out[f"num{k}"] = s(e * e * h, 3)
        out[f"den{k}"] = s(pp * s2 * (1 - h) ** 2, 4)
        out[f"om{k}"] = s(1 - h, 2)
        out[f"r2{k}"] = s(e * e / (s2 * (1 - h)), 3)
        out[f"noh{k}"] = s(e * e / (pp * s2), 2)
    return out


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


def p12():
    N = (600, 300, 100)
    S = (2, 3, 8)
    c = (4, 9, 16)
    n = 120
    w = [Ni * Si / sqrt(ci) for Ni, Si, ci in zip(N, S, c)]
    nh = n * w[2] / sum(w)
    ney = [Ni * Si for Ni, Si in zip(N, S)]
    wc = [Ni * Si / ci for Ni, Si, ci in zip(N, S, c)]
    # (b) ジェフリーズ事前分布
    m, y = r(nh, 0), 4
    a, b = y + 0.5, m - y + 0.5
    return {
        "N1": N[0], "N2": N[1], "N3": N[2], "S1": S[0], "S2": S[1], "S3": S[2], "c1": c[0], "c2": c[1], "c3": c[2], "n": n,
        "rc1": int(sqrt(c[0])), "rc2": int(sqrt(c[1])), "rc3": int(sqrt(c[2])),
        "w1": s(w[0], 0), "w2": s(w[1], 0), "w3": s(w[2], 0), "wsum": s(sum(w), 0), "nh_raw": s(nh, 2), "nh": r(nh, 0),
        "ney_raw": s(n * ney[2] / sum(ney), 2), "ney": r(n * ney[2] / sum(ney), 0), "neysum": sum(ney), "ney3": ney[2],
        "prop": s(n * N[2] / sum(N), 0), "wc_raw": s(n * wc[2] / sum(wc), 2), "wc": r(n * wc[2] / sum(wc), 0),
        "m": m, "y": y, "a": s(a, 1), "b": s(b, 1), "ab": s(a + b, 0), "post": s(a / (a + b), 3), "post4": s(a / (a + b), 4),
        "mle": s(y / m, 3), "unif": s((y + 1) / (m + 2), 3), "mode": s((a - 1) / (a + b - 2), 3), "w15": s((y + 1.5) / (m + 3), 3),
        "my": m - y,
    }


def p13():
    # (a) 分散の異なる2群の判別（2次判別）
    mu1, s1, mu2, s2 = 3.0, 0.5, 2.0, 1.0
    ln2 = 0.6931
    # 2(x-3)^2 - 0.5(x-2)^2 = ln2 → 1.5x^2 - 10x + 16 - ln2 = 0
    A = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
    B = -2 * mu1 / (2 * s1 ** 2) + 2 * mu2 / (2 * s2 ** 2)
    C = mu1 ** 2 / (2 * s1 ** 2) - mu2 ** 2 / (2 * s2 ** 2) - ln2
    disc = B * B - 4 * A * C
    lo, hi = (-B - sqrt(disc)) / (2 * A), (-B + sqrt(disc)) / (2 * A)
    C0 = C + ln2
    d0 = B * B - 4 * A * C0
    lo0, hi0 = (-B - sqrt(d0)) / (2 * A), (-B + sqrt(d0)) / (2 * A)
    # σ を σ² の代わりに使った誤り
    A2 = 1 / (2 * s1) - 1 / (2 * s2)
    B2 = -2 * mu1 / (2 * s1) + 2 * mu2 / (2 * s2)
    C2 = mu1 ** 2 / (2 * s1) - mu2 ** 2 / (2 * s2) - ln2
    d2 = B2 * B2 - 4 * A2 * C2
    lo2, hi2 = (-B2 - sqrt(d2)) / (2 * A2), (-B2 + sqrt(d2)) / (2 * A2)
    # (b) ゼロ切断ポアソン
    lam, em = 0.8, 0.4493
    zt = lam / (1 - em)
    return {
        "mu1": s(mu1, 1), "s1": s(s1, 1), "mu2": s(mu2, 1), "s2": s(s2, 1), "v1": s(s1 ** 2, 2), "v2": s(s2 ** 2, 1),
        "ln2": ln2, "A": s(A, 1), "B": s(-B, 0), "C0": s(C0, 0), "C": s(C, 4), "disc": s(disc, 4), "rd": s(sqrt(disc), 4),
        "lo": s(lo, 2), "hi": s(hi, 2), "lo4": s(lo, 4), "hi4": s(hi, 4), "mid": s((mu1 + mu2) / 2, 2),
        "lo0": s(lo0, 2), "hi0": s(hi0, 2), "lo2": s(lo2, 2), "hi2": s(hi2, 2), "twoA": s(2 * A, 0),
        "lam": lam, "em": em, "one_em": s(1 - em, 4), "zt": s(zt, 3), "zt4": s(zt, 4),
        "w_lam1": s(lam + 1, 3), "w_mul": s(lam * (1 - em), 3), "w_div": s(lam / em, 3), "w_lam": s(lam, 3),
    }


def p14():
    r12, r13, r23 = 0.72, 0.63, 0.56
    l3sq = r13 * r23 / r12
    l3 = sqrt(l3sq)
    l2sq = r12 * r23 / r13
    l1sq = r12 * r13 / r23
    avg = (r13 + r23) / 2
    # (b) シルエット係数
    own = (1.2, 1.8)
    cb = (2.6, 3.0, 3.4)
    cc = (2.2, 3.6)
    a = sum(own) / len(own)
    mb, mc = sum(cb) / len(cb), sum(cc) / len(cc)
    b = min(mb, mc)
    sil = (b - a) / max(a, b)
    return {
        "r12": r12, "r13": r13, "r23": r23, "num": s(r13 * r23, 4), "l3sq": s(l3sq, 2), "l3": s(l3, 2), "psi3": s(1 - l3sq, 2),
        "l1sq": s(l1sq, 2), "l2sq": s(l2sq, 2), "l1": s(sqrt(l1sq), 2), "l2": s(sqrt(l2sq), 2),
        "w_l": s(l3sq, 2), "w_psi_l": s(1 - l3, 2), "w_avg": s(avg, 3), "w_avgpsi": s(1 - avg ** 2, 3),
        "w_wrong": s(sqrt(l2sq), 2), "w_wrongpsi": s(1 - l2sq, 2),
        "o1": own[0], "o2": own[1], "b1": cb[0], "b2": cb[1], "b3": cb[2], "c1": cc[0], "c2": cc[1],
        "a": s(a, 2), "mb": s(mb, 2), "mc": s(mc, 2), "bb": s(b, 2), "sil": r(sil, 2), "sil4": s(sil, 4),
        "w_min": s((min(cb + cc) - a) / max(a, min(cb + cc)), 4), "w_mb": s((mb - a) / mb, 4),
    }


def values():
    return {"s23p01": p01(), "s23p02": p02(), "s23p03": p03(), "s23p04": p04(), "s23p05": p05(),
            "s23p06": p06(), "s23p07": p07(), "s23p08": p08(), "s23p09": p09(), "s23p10": p10(),
            "s23p11": p11(), "s23p12": p12(), "s23p13": p13(), "s23p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
