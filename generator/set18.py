"""模擬試験 第18回（set18）の数値。values() が {大問id: {名前: 値}} を返す。

題材：生物学・遺伝・生態調査・水産。
問題文に示した分布の値（χ²・F・t の点、Φ・φ、e^x・ln・cos など）は、ここでも同じ丸めた値を使って答えを計算する。
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


def distinct(vals, d):
    assert len({s(v, d) for v in vals}) == len(vals), vals


def p01():
    # 共優性の遺伝子座：遺伝子型の数
    nAA, nAa, naa = 50, 80, 70
    n = nAA + nAa + naa
    cntA = 2 * nAA + nAa
    p = cntA / (2 * n)
    q = 1 - p
    # (b) 複合帰無仮説 q<=0.2 の二項検定（数値は問題文に直接書く。verify で検算）
    # (c) 優性の遺伝子座の EM アルゴリズム 1 回
    nD, nR = 160, 40
    nn = nD + nR
    p0 = 0.5
    q0 = 1 - p0
    den = p0 ** 2 + 2 * p0 * q0
    eBB = nD * p0 ** 2 / den
    eBb = nD * 2 * p0 * q0 / den
    p1 = (2 * eBB + eBb) / (2 * nn)
    q_mle = sqrt(nR / nn)
    return {
        "nAA": nAA, "nAa": nAa, "naa": naa, "n": n, "n2": 2 * n, "cntA": cntA,
        "p": s(p, 2), "q": s(q, 2), "p_ind": s((nAA + nAa) / n, 2), "nind": nAA + nAa, "p_hom": s(sqrt(nAA / n), 2),
        "p_n": s(cntA / n, 2),
        "nD": nD, "nR": nR, "nn": nn, "nn2": 2 * nn, "p0": s(p0, 1), "pp0": s(p0 ** 2, 2), "pq0": s(2 * p0 * q0, 2),
        "den0": s(den, 2), "eBB": s(eBB, 2), "eBb": s(eBb, 2), "eBB_f": "160/3", "eBb_f": "320/3",
        "p1": r(p1, 3), "p1_s": s(p1, 3), "p1_4": s(p1, 4), "two_eBB": s(2 * eBB, 2), "num1": s(2 * eBB + eBb, 2),
        "q_mle": s(q_mle, 4), "p_mle": s(1 - q_mle, 4),
        "p1_wrong": s((eBB + eBb) / nn, 3),
    }


def p02():
    # (a) 2 遺伝子座のハプロタイプ頻度と連鎖不平衡
    h = {"AB": 0.40, "Ab": 0.20, "aB": 0.10, "ab": 0.30}
    pA = h["AB"] + h["Ab"]
    pB = h["AB"] + h["aB"]
    D = h["AB"] - pA * pB
    assert abs(D - (h["AB"] * h["ab"] - h["Ab"] * h["aB"])) < 1e-12
    # (b) 共通の分母をもつ比の見かけの相関
    cx, cy, cz = 0.10, 0.20, 0.20
    a, b = cx ** 2 + cz ** 2, cy ** 2 + cz ** 2
    rho = cz ** 2 / sqrt(a * b)
    return {
        "hAB": s(h["AB"], 2), "hAb": s(h["Ab"], 2), "haB": s(h["aB"], 2), "hab": s(h["ab"], 2),
        "pA": s(pA, 2), "pB": s(pB, 2), "pApB": s(pA * pB, 2), "D": s(D, 2), "mD": s(-D, 2),
        "prod1": s(h["AB"] * h["ab"], 2), "prod2": s(h["Ab"] * h["aB"], 2), "ratio": s(h["AB"] / (pA * pB), 2),
        "pBA": s(h["AB"] / pA, 3),
        "cx": s(cx, 2), "cy": s(cy, 2), "cz": s(cz, 2), "cz2": s(cz ** 2, 2), "a": s(a, 2), "b": s(b, 2),
        "ans": r(rho, 2), "ans4": s(rho, 4), "ans_s": s(rho, 2),
    }


def p03():
    # (a) ボックス・コックス変換（λ=0.5）で正規 N(6, 2^2)
    lam, mu, sg = 0.5, 6.0, 2.0
    phi = {0.5: 0.6915, 1.0: 0.8413, 2.0: 0.9772}
    med = (1 + lam * mu) ** 2
    w0 = 25
    z0 = (sqrt(w0) - 1) / lam
    zs = (z0 - mu) / sg
    assert zs in phi
    p_ok = 1 - phi[zs]
    med_nl = (mu + 1) ** 2
    z_nl = sqrt(w0) - 1
    p_nl = phi[(mu - z_nl) / sg]
    med_n1 = (lam * mu) ** 2
    z_n1 = sqrt(w0) / lam
    p_n1 = 1 - phi[(z_n1 - mu) / sg]
    p_var = 1 - phi[(z0 - mu) / sg ** 2]
    # (b) 母分散のカイ二乗検定（両側）
    n, s2, s0 = 16, 6.9, 4
    lo, hi, u5 = 6.262, 27.488, 24.996
    X = (n - 1) * s2 / s0
    assert lo < X < hi and X > u5
    return {
        "lam": lam, "mu": s(mu, 0), "sg": s(sg, 0), "var": s(sg ** 2, 0),
        "ph05": s(phi[0.5], 4), "ph1": s(phi[1.0], 4), "ph2": s(phi[2.0], 4),
        "med": s(med, 1), "sqmed": s(1 + lam * mu, 0), "w0": w0, "z0": s(z0, 0), "zs": s(zs, 1),
        "p_ok": s(p_ok, 3), "p_ok4": s(p_ok, 4), "med_nl": s(med_nl, 1), "p_nl": s(p_nl, 3), "med_n1": s(med_n1, 1),
        "p_n1": s(p_n1, 3), "p_var": s(p_var, 3), "p_tail": s(phi[zs], 3),
        "n": n, "df": n - 1, "s2": s(s2, 1), "s0": s0, "lo": s(lo, 3), "hi": s(hi, 3), "u5": s(u5, 3),
        "X": s(X, 3), "X4": s(X, 3), "p2": "0.08",
        "w_n": s(n * s2 / s0, 1), "w_ratio": s(s2 / s0, 3),
    }


def p04():
    # (b) 定数打ち切り（r 本目で終了）の指数分布の平均の信頼区間
    n, rr, T = 12, 8, 4800
    c16l, c16u, c24l, c24u = 6.908, 28.845, 12.401, 39.364
    th = T / rr
    lo, hi = 2 * T / c16u, 2 * T / c16l
    w24 = (2 * T / c24u, 2 * T / c24l)
    wsw = (T / c16u, T / c16l)
    wz = (th - 1.96 * th / sqrt(rr), th + 1.96 * th / sqrt(rr))
    wm = (2 * th / c16u, 2 * th / c16l)
    distinct([lo, w24[0], wsw[0], wz[0], wm[0]], 0)
    return {
        "n": n, "r": rr, "nr": n - rr, "T": T, "T2": 2 * T, "th": s(th, 0),
        "c16l": s(c16l, 3), "c16u": s(c16u, 3), "c24l": s(c24l, 3), "c24u": s(c24u, 3),
        "lo": s(lo, 0), "hi": s(hi, 0),
        "w24_lo": s(w24[0], 0), "w24_hi": s(w24[1], 0), "w_sw_lo": s(wsw[0], 0), "w_sw_hi": s(wsw[1], 0),
        "w_z_lo": s(wz[0], 0), "w_z_hi": s(wz[1], 0), "w_m_lo": s(wm[0], 0), "w_m_hi": s(wm[1], 0),
    }


def p05():
    # (a) 超幾何分布
    N, K, m, k = 50, 10, 5, 2
    c1, c2, c3 = math.comb(K, k), math.comb(N - K, m - k), math.comb(N, m)
    ok = c1 * c2 / c3
    pp = K / N
    binom = math.comb(m, k) * pp ** k * (1 - pp) ** (m - k)
    ge = sum(math.comb(K, j) * math.comb(N - K, m - j) for j in range(2, m + 1)) / c3
    vals = [ok, binom, ge, k / m, 1 - ok]
    distinct(vals, 4)
    # (b) 二相抽出の比推定
    NQ, n1, n2, x1, x2, y2 = 400, 100, 20, 32.0, 30.0, 4.5
    R = y2 / x2
    ybar = R * x1
    tot = NQ * ybar
    return {
        "N": N, "K": K, "m": m, "k": k, "NK": N - K, "mk": m - k, "p": s(pp, 1),
        "c1": c1, "c2": c2, "c3": c3, "ok": s(ok, 4), "ok4": s(ok, 4), "binom": s(binom, 4),
        "w_ge": s(ge, 4), "w_ratio": s(k / m, 4), "w_comp": s(1 - ok, 4),
        "NQ": NQ, "n1": n1, "n2": n2, "x1": s(x1, 1), "x2": s(x2, 1), "y2": s(y2, 1),
        "R": s(R, 2), "ybar": s(ybar, 1), "tot": r(tot, 0), "tot_s": s(tot, 0), "w_y2": s(NQ * y2, 0),
    }


def p06():
    # (a) 2 本の回帰直線の傾きの積 = r^2
    n, byx, bxy = 40, 2.5, 0.30
    r2 = byx * bxy
    rr = sqrt(r2)
    wr2 = bxy / byx
    distinct([rr, r2, sqrt(wr2)], 3)
    return {
        "n": n, "byx": s(byx, 1), "bxy": s(bxy, 2), "r2": s(r2, 2), "r": s(rr, 3),
        "w_r": s(sqrt(wr2), 3), "w_r2": s(wr2, 2), "inv": s(1 / bxy, 2),
    }


def p07():
    # (a) 準二項モデルによる標準誤差の補正
    b, se, phi = 0.48, 0.20, 2.5
    z1 = b / se
    se2 = se * sqrt(phi)
    z2 = b / se2
    w_mul = b / (se * phi)
    w_div = b / (se / sqrt(phi))
    distinct([z1, z2, w_mul, w_div], 2)
    # (b) 前進選択の部分 F
    n = 30
    R = {"0": 400, "1": 250, "2": 300, "3": 330, "12": 200, "13": 240, "123": 180}
    F = lambda a, c, p: (R[a] - R[c]) / (R[c] / (n - p))
    F1, F2, F3 = F("0", "1", 2), F("0", "2", 2), F("0", "3", 2)
    F12, F13, F123 = F("1", "12", 3), F("1", "13", 3), F("12", "123", 4)
    assert F1 == max(F1, F2, F3) >= 4 and F12 >= 4 > F13 and F12 > F13 and F123 < 4
    return {
        "b": s(b, 2), "se": s(se, 2), "phi": s(phi, 1), "z1": s(z1, 2), "se2": s(se2, 3),
        "z2": s(z2, 2), "z24": s(z2, 3), "w_mul": s(w_mul, 2), "w_div": s(w_div, 2),
        **{"r" + k: v for k, v in R.items()},
        "F1": s(F1, 1), "F2": s(F2, 2), "F3": s(F3, 2), "F12": s(F12, 2), "F13": s(F13, 3), "F123": s(F123, 2),
    }


def p08():
    # (a) L8 直交表
    colA = [1, 1, 1, 1, 2, 2, 2, 2]
    colB = [1, 1, 2, 2, 1, 1, 2, 2]
    colC = [1, 2, 1, 2, 1, 2, 1, 2]
    y = [20, 22, 25, 27, 26, 30, 31, 35]
    T1 = sum(v for v, a in zip(y, colA) if a == 1)
    T2 = sum(v for v, a in zip(y, colA) if a == 2)
    eff = (T2 - T1) / 4
    ss = (T2 - T1) ** 2 / 8
    # (b) 非心 F 分布の期待値
    k, nn, sig = 3, 8, 4.0
    al = (-2.0, 0.0, 2.0)
    lam = nn * sum(a * a for a in al) / sig ** 2
    nu1, nu2 = k - 1, k * nn - k
    EF = nu2 * (nu1 + lam) / (nu1 * (nu2 - 2))
    lam_n = sum(a * a for a in al) / sig ** 2
    lam_s = nn * sum(a * a for a in al) / sig
    vals = [EF, nu2 * (nu1 + lam_n) / (nu1 * (nu2 - 2)), nu2 * (nu1 + lam_s) / (nu1 * (nu2 - 2)), nu2 / (nu2 - 2)]
    distinct(vals, 2)
    out = {
        "T1": T1, "T2": T2, "dT": T2 - T1, "eff": s(eff, 1), "ss": s(ss, 1), "ss4": s((T2 - T1) ** 2 / 4, 1),
        "ss16": s((T2 - T1) ** 2 / 16, 1), "half": s(eff / 2, 2), "ssh": s(ss / 4, 2),
        "k": k, "nn": nn, "sig": s(sig, 0), "s2": s(sig ** 2, 0), "a1": s(al[0], 0), "a3": s(al[2], 0),
        "sa2": s(sum(a * a for a in al), 0), "lam": s(lam, 1), "nu1": nu1, "nu2": nu2,
        "EF": r(EF, 2), "EF_s": s(EF, 2), "EF4": s(EF, 4), "EF_n": s(vals[1], 2), "EF_s2": s(vals[2], 2), "EF0": s(vals[3], 2),
        "lam_n": s(lam_n, 1), "lam_s": s(lam_s, 0),
    }
    for i, v in enumerate(y):
        out[f"y{i}"] = v
    return out


def p09():
    # 無作為化した餌 → 放流時の大きさ（中間変数）→ 生残
    a1n, a1s, b1n, b1s = 150, 84, 50, 14
    a0n, a0s, b0n, b0s = 50, 30, 150, 45
    n1, n0 = a1n + b1n, a0n + b0n
    r1, r0 = (a1s + b1s) / n1, (a0s + b0s) / n0
    wa = (a1n + a0n) / (n1 + n0)
    adj = (wa * a1s / a1n + (1 - wa) * b1s / b1n) - (wa * a0s / a0n + (1 - wa) * b0s / b0n)
    tot = r1 - r0
    assert tot > 0 > adj and wa == 0.5
    return {
        "n": n1 + n0, "n1": n1, "n0": n0, "a1n": a1n, "a1s": a1s, "b1n": b1n, "b1s": b1s,
        "a0n": a0n, "a0s": a0s, "b0n": b0n, "b0s": b0s, "r1": s(r1, 3), "r0": s(r0, 3),
        "tot": "$+" + s(tot, 3) + "$", "adj": "$" + s(adj, 3) + "$",
        "p1a": s(a1s / a1n, 2), "p0a": s(a0s / a0n, 2), "p1b": s(b1s / b1n, 2), "p0b": s(b0s / b0n, 2),
        "da": s(a1s / a1n - a0s / a0n, 2), "db": s(b1s / b1n - b0s / b0n, 2),
    }


def p10():
    # (a) コーフェン相関
    d = {"ab": 2, "ac": 6, "ad": 10, "bc": 5, "bd": 9, "cd": 4}
    keys = list(d)
    x = [d[k] for k in keys]

    def corr(u, v):
        mu, mv = sum(u) / len(u), sum(v) / len(v)
        suv = sum((a - mu) * (b - mv) for a, b in zip(u, v))
        return suv / sqrt(sum((a - mu) ** 2 for a in u) * sum((b - mv) ** 2 for b in v))

    def coph(hh):
        return [d["ab"] if k == "ab" else d["cd"] if k == "cd" else hh for k in keys]

    cross = [d["ac"], d["ad"], d["bc"], d["bd"]]
    hs, ha, hc = min(cross), sum(cross) / 4, max(cross)
    cs, ca, cc = corr(x, coph(hs)), corr(x, coph(ha)), corr(x, coph(hc))

    def rank(v):
        sv = sorted(v)
        return [(sv.index(a) + 1 + len(sv) - sv[::-1].index(a)) / 2 for a in v]

    w_s = corr(rank(x), rank(coph(hs)))
    distinct([cs, ca, cc, w_s], 3)
    # (b) 同順位のある順位相関
    a = [3, 8, 8, 15, 22, 30, 41, 41]
    b = [5, 2, 9, 9, 17, 14, 26, 33]
    ra, rb = rank(a), rank(b)
    rs = corr(ra, rb)
    sd2 = sum((u - v) ** 2 for u, v in zip(ra, rb))
    rs0 = 1 - 6 * sd2 / (8 * 63)

    def rmin(v):
        sv = sorted(v)
        return [sv.index(a) + 1 for a in v]

    distinct([rs, rs0, corr(a, b), corr(rmin(a), rmin(b))], 3)
    fmt = lambda v: str(int(v)) if v == int(v) else str(v)
    # (c) 集団寄与危険割合
    RR, pe = 3, 0.40
    num = pe * (RR - 1)
    paf = num / (1 + num)
    return {
        **{"d" + k: v for k, v in d.items()}, "hs": s(hs, 0), "ha": s(ha, 1),
        "cs": s(cs, 3), "ca": s(ca, 3), "cc": s(cc, 3), "w_s": s(w_s, 3),
        "a": " | ".join(map(str, a)), "b": " | ".join(map(str, b)),
        "ra": ",\\ ".join(map(fmt, ra)), "rb": ",\\ ".join(map(fmt, rb)), "sd2": s(sd2, 0),
        "rs": s(rs, 3), "rs0": s(rs0, 3), "w_pear": s(corr(a, b), 3), "w_int": s(corr(rmin(a), rmin(b)), 3),
        "RR": RR, "pe": s(pe, 2), "num": s(num, 2), "den": s(1 + num, 2), "paf": r(paf, 3), "paf4": s(paf, 4),
        "paf_s": s(paf, 3), "pct": s(paf * 100, 0), "w_ex": s((RR - 1) / RR, 3),
    }


def p11():
    # (a) 環境DNA の検出スコアと閾値
    pres = [2, 4, 5, 6, 7, 8, 9, 9]
    absn = [1, 1, 2, 3, 3, 4, 5, 7]
    out = {}
    best = None
    for c in (3, 5, 7):
        se = sum(v >= c for v in pres) / len(pres)
        sp_ = sum(v < c for v in absn) / len(absn)
        J = se + sp_ - 1
        out[f"se{c}"], out[f"sp{c}"], out[f"J{c}"] = s(se, 3), s(sp_, 3), s(J, 3)
        out[f"fp{c}"] = s(1 - sp_, 3)
        if best is None or J > best[1]:
            best = (c, J)
    assert best[0] == 5
    js = sorted(float(out[f"J{c}"]) for c in (3, 5, 7))
    assert js[-1] > js[-2]
    # (b) 2因子直交モデル
    L = {"len": (0.8, 0.3), "hgt": (0.7, 0.4), "wid": (0.3, 0.8), "wt": (0.6, 0.5)}
    robs = 0.50
    rep = L["len"][0] * L["wid"][0] + L["len"][1] * L["wid"][1]
    h_len = sum(v * v for v in L["len"])
    h_wid = sum(v * v for v in L["wid"])
    f1 = L["len"][0] * L["wid"][0]
    distinct([rep, f1, h_len * h_wid, h_len], 2)
    out.update({
        "pres": "、".join(map(str, pres)), "absn": "、".join(map(str, absn)), "np": len(pres), "na": len(absn),
        **{f"{k}{i}": s(v[i], 1) for k, v in L.items() for i in (0, 1)},
        "robs": s(robs, 2), "rep": s(rep, 2), "res": s(robs - rep, 2), "mres": s(rep - robs, 2), "f1": s(f1, 2), "rf1": s(robs - f1, 2),
        "hh": s(h_len * h_wid, 2), "rhh": s(robs - h_len * h_wid, 2), "hl": s(h_len, 2), "rhl": s(robs - h_len, 2), "ul": s(1 - h_len, 2),
        "p1": s(L["len"][0] * L["wid"][0], 2), "p2": s(L["len"][1] * L["wid"][1], 2),
    })
    return out


def p12():
    # (a) ユール過程
    lam, t, k = 0.5, 2, 4
    em = 0.3679
    q = 1 - em
    ok = q ** (k - 1)
    eq = em * q ** (k - 1)
    w_k = q ** k
    w_pois = None
    distinct([ok, eq, w_k], 3)
    # (b) ボックス・ミュラー法
    u1, u2 = 0.30, 0.20
    ln1, ln2, c1, s1, c2 = -1.2040, -1.6094, 0.3090, 0.9511, -0.3090
    z = r(sqrt(-2 * ln1) * c1, 4)
    mu, sg = 0.30, 0.10
    g = mu + sg * z
    g_swap = mu + sg * sqrt(-2 * ln2) * c2
    g_nosq = mu + sg * (-2 * ln1) * c1
    g_sin = mu + sg * sqrt(-2 * ln1) * s1
    g_no2 = mu + sg * sqrt(-ln1) * c1
    distinct([g_swap, g_no2, g, g_nosq, g_sin], 3)
    return {
        "lam": lam, "t": t, "k": k, "lt": s(lam * t, 0), "em": s(em, 4), "q": s(q, 4), "km1": k - 1,
        "ok": r(ok, 3), "ok_s": s(ok, 3), "ok4": s(ok, 4), "eq": s(eq, 3), "w_k": s(w_k, 3), "mean": s(math.e, 2),
        "u1": s(u1, 2), "u2": s(u2, 2), "ln1": s(ln1, 4), "ln2": s(ln2, 4), "c1": s(c1, 4), "s1": s(s1, 4), "c2": s(c2, 4),
        "m2ln": s(-2 * ln1, 3), "sqv": s(sqrt(-2 * ln1), 4), "z": s(z, 4), "mu": s(mu, 2), "sg": s(sg, 2),
        "g": s(g, 3), "g5": s(g, 5), "g_swap": s(g_swap, 3), "g_nosq": s(g_nosq, 3), "g_sin": s(g_sin, 3), "g_no2": s(g_no2, 3),
    }


def values():
    return {"s18p01": p01(), "s18p02": p02(), "s18p03": p03(), "s18p04": p04(), "s18p05": p05(),
            "s18p06": p06(), "s18p07": p07(), "s18p08": p08(), "s18p09": p09(), "s18p10": p10(),
            "s18p11": p11(), "s18p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
