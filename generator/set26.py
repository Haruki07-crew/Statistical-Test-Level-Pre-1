"""模擬試験 第26回（set26）の数値。values() が {大問id: {名前: 値}} を返す。

題材：IT・データセンター・情報セキュリティ。
問題文に示した分布の値（Φ の値・t の点・平方根や対数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
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
    # 6 本のジョブを 3 台に割り当て、すべての台が使われる確率（包除原理）
    nj, ns = 6, 3
    p1 = Fraction(2, 3) ** nj
    p2 = Fraction(1, 3) ** nj
    union = 3 * p1 - 3 * p2
    ans = 1 - union
    assert ans == Fraction(540, 729)
    return {"nj": nj, "ns": ns, "p1": s(p1, 5), "p2": s(p2, 5), "union": s(union, 4),
            "ans": r(ans, 3), "ans4": s(ans, 4), "ans_s": s(ans, 3), "w_noinc": s(1 - 3 * p1, 3)}


def p02():
    # (a) 対応のある t 検定と、対応を無視した 2 標本 t 検定
    n, s1, s2, d, sd = 8, 20, 19, 6.0, 4.8
    tp = d / (sd / sqrt(n))
    sei = sqrt(s1 ** 2 / n + s2 ** 2 / n)
    ti = d / sei
    assert tp > 2.365 and ti < 2.145
    # (b) 非心 t 分布
    m, dl = 16, 0.5
    ncp = sqrt(m) * dl
    return {"n": n, "s1": s1, "s2": s2, "d": s(d, 1), "sd": s(sd, 1), "t7": "2.365", "t14": "2.145",
            "tp": s(tp, 2), "sei": s(sei, 2), "ti": s(ti, 2), "w_n": s(d / sd, 2),
            "m": m, "dl": s(dl, 1), "df": m - 1, "ncp": s(ncp, 1), "ncp2": s(ncp ** 2, 1)}


def p03():
    # (a) パレート分布の形状母数の最尤推定（x_m=1 が既知）
    xs = [1.5, 2.0, 1.2, 4.0, 3.0]
    logs = [0.4055, 0.6931, 0.1823, 1.3863, 1.0986]
    slog = sum(logs)
    alpha = len(xs) / slog
    xb = sum(xs) / len(xs)
    mm = xb / (xb - 1)
    # (b) ポアソン平均のワルド・スコア・尤度比検定
    x, l0 = 12, 6
    zw = (x - l0) / 3.464
    zs = (x - l0) / 2.449
    G = 2 * (x * 0.6931 - (x - l0))
    assert zw < 1.96 < zs and G > 3.84
    return {"n": len(xs), "xs": "、".join(s(v, 1) for v in xs), "xb": s(xb, 2), "logs": "、".join(s(v, 4) for v in logs),
            "slog": s(slog, 4), "alpha": s(alpha, 2), "alpha4": s(alpha, 4), "w_inv": s(1 / alpha, 2), "w_mm": s(mm, 2),
            "x": x, "l0": l0, "zw": s(zw, 2), "zs": s(zs, 2), "G": s(G, 1)}


def p04():
    # (a) p 値の分布（確率積分変換）と、対立仮説のもとでの中央値
    dl = 2
    ph2 = 0.9772
    pw = 0.6406   # Φ(2−1.645)=Φ(0.355)≈Φ(0.36)
    return {"dl": dl, "ph2": s(ph2, 4), "med": s(1 - ph2, 4), "pw": s(pw, 2)}


def p05():
    # (a) モンテカルロ法による並べ替え検定の p 値
    B, k = 999, 24
    p = Fraction(k + 1, B + 1)
    # (b) M/M/2 待ち行列
    lam, mu = 1.6, 1.0
    a = lam / mu
    rho = a / 2
    t3 = (a * a / 2) / (1 - rho)
    den = 1 + a + t3
    pw = t3 / den
    Lq = pw * rho / (1 - rho)
    Wq = Lq / lam
    assert abs(Wq - 16 / 9) < 1e-12
    w_fast = lam / (2 * mu * (2 * mu - lam))
    w_sep = (lam / 2) / (mu * (mu - lam / 2))
    # (c) スルツキーの定理による近似区間
    n, xb, sx = 400, 210, 120
    se = sx / sqrt(n)
    return {"B": B, "k": k, "k1": k + 1, "B1": B + 1, "p": s(p, 3), "w_raw": s(Fraction(k, B), 4), "w_comp": s(1 - Fraction(k, B), 3),
            "w_two": s(2 * p, 3),
            "lam": s(lam, 1), "mu_inv": s(1 / mu, 0), "rho": s(rho, 1), "a22": s(a * a / 2, 2), "t3": s(t3, 1), "den": s(den, 0),
            "pw": s(pw, 4), "rr": s(rho / (1 - rho), 0), "Lq": s(Lq, 4), "Wq": r(Wq, 2), "Wq4": s(Wq, 4), "Wq_s": s(Wq, 2),
            "w_fast": s(w_fast, 2), "w_sep": s(w_sep, 2),
            "n": n, "xb": xb, "sx": sx, "se": s(se, 0), "lo": s(xb - 1.96 * se, 2), "hi": s(xb + 1.96 * se, 2),
            "wlo": s(xb - 1.96 * sx, 1), "whi": s(xb + 1.96 * sx, 1)}


def p06():
    vx, vy, cxy = 25, 16, 6
    c = vx - vy
    vs, vd = vx + vy + 2 * cxy, vx + vy - 2 * cxy
    rho = c / sqrt(vs * vd)
    return {"vx": vx, "vy": vy, "cxy": cxy, "c": c, "vs": vs, "vd": vd, "ans": r(rho, 3), "ans4": s(rho, 4), "ans_s": s(rho, 3)}


def p07():
    # (a) フリードマン検定（5 ブロック × 3 方式）
    A = [12.1, 30.5, 8.4, 20.2, 15.0]
    B = [13.0, 31.2, 8.1, 22.5, 15.8]
    C = [14.2, 33.0, 9.0, 21.4, 16.6]
    ranks = []
    R = [0, 0, 0]
    for blk in zip(A, B, C):
        order = sorted(range(3), key=lambda j: blk[j])
        rk = [0, 0, 0]
        for pos, j in enumerate(order):
            rk[j] = pos + 1
        ranks.append(rk)
        for j in range(3):
            R[j] += rk[j]
    assert R == [6, 10, 14]
    Q = 12 / (5 * 3 * 4) * sum(v * v for v in R) - 3 * 5 * 4
    allv = sorted(A + B + C)
    Rall = [sum(allv.index(v) + 1 for v in g) for g in (A, B, C)]
    H = 12 / (15 * 16) * sum(v * v / 5 for v in Rall) - 3 * 16
    # (b) 信頼係数
    p3 = 0.95 ** 3
    p1of20 = 20 * 0.05 * 0.95 ** 19
    return {"A": " | ".join(s(v, 1) for v in A), "B": " | ".join(s(v, 1) for v in B), "C": " | ".join(s(v, 1) for v in C),
            "ranks": "、".join(f"({a},{b},{c})" for a, b, c in ranks), "RA": R[0], "RB": R[1], "RC": R[2],
            "sR2": sum(v * v for v in R), "Q": s(Q, 1), "w_all": s(H, 2),
            "p3": s(p3, 3), "p1of20": s(p1of20, 3)}


def p08():
    # (b) ニューラルネットワークの母数の数
    h1, h2 = 5, 4
    P1, P2, P3 = (3 + 1) * h1, (h1 + 1) * h2, (h2 + 1) * 1
    # (c) マハラノビス距離からてこ比
    n, D2 = 41, 8.0
    h = 1 / n + D2 / (n - 1)
    return {"h1": h1, "h2": h2, "P1": P1, "P2": P2, "P3": P3, "P": P1 + P2 + P3, "w_nob": 3 * h1 + h1 * h2 + h2,
            "n": n, "n1": n - 1, "D2": s(D2, 1), "a1": s(1 / n, 4), "a2": s(D2 / (n - 1), 4), "h": r(h, 3), "h4": s(h, 4),
            "h_s": s(h, 3), "thr": s(6 / n, 3)}


def p09():
    bl = 0.90
    sdl = math.pi / sqrt(3)
    return {"bl": s(bl, 2), "sdl": s(sdl, 2), "bp": s(bl / sdl, 1), "bp4": s(bl / sdl, 3), "bp16": s(bl / 1.6, 2),
            "w_mul": s(bl * sdl, 1), "w_var": s(bl / (math.pi ** 2 / 3), 2)}


def p10():
    fail = [8, 12, 15, 20, 25]
    cens = [18, 30, 30]
    th0 = 20
    sf, sc = sum(fail), sum(cens)
    scc = sc + 3 * th0
    th1 = (sf + scc) / 8
    tot = sf + sc
    times = "、".join([str(v) for v in [8, 12, 15]] + ["18+"] + ["20", "25", "30+", "30+"])
    return {"times": times, "th0": th0, "sf": sf, "c1": cens[0], "c2": cens[1], "c3": cens[2], "sc": sc, "scc": scc,
            "th1": r(th1, 2), "th1_s": s(th1, 2), "tot": tot, "mle": s(tot / 5, 1), "w_drop": s(sf / 5, 1)}


def p11():
    b, t = 6, 4
    MSB, MSE, MST = 30.0, 5.0, 40.0
    RE = ((b - 1) * MSB + b * (t - 1) * MSE) / ((b * t - 1) * MSE)
    return {"SSB": s(MSB * 5, 1), "MSB": s(MSB, 1), "SST": s(MST * 3, 1), "MST": s(MST, 1), "SSE": s(MSE * 15, 1), "MSE": s(MSE, 1),
            "RE": s(RE, 2), "RE4": s(RE, 4), "RE1": s(RE, 1), "n1": s(5 * MSB, 0), "n2": s(18 * MSE, 0), "d": s(23 * MSE, 0),
            "w_ratio": s(MSB / MSE, 1), "w_inv": s(1 / RE, 2)}


def p12():
    W = [0.5, 0.3, 0.2]
    S = [4, 10, 20]
    n = 100
    a = [w * v * v for w, v in zip(W, S)]
    b = [w * v for w, v in zip(W, S)]
    vp = sum(a) / n
    vn = sum(b) ** 2 / n
    ratio = vp / vn
    return {"W1": W[0], "W2": W[1], "W3": W[2], "S1": S[0], "S2": S[1], "S3": S[2], "n": n,
            "a1": s(a[0], 0), "a2": s(a[1], 0), "a3": s(a[2], 0), "sws2": s(sum(a), 0), "vp": s(vp, 2),
            "b1": s(b[0], 0), "b2": s(b[1], 0), "b3": s(b[2], 0), "sws": s(sum(b), 0), "vn": s(vn, 2),
            "ratio": r(ratio, 2), "ratio4": s(ratio, 4), "ratio_s": s(ratio, 2)}


def p13():
    # (b) 分散共分散行列と相関行列の主成分
    v1, v2, c = 100, 1, 6
    tr, det = v1 + v2, v1 * v2 - c * c
    disc = sqrt(tr * tr - 4 * det)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    e = (c, l1 - v1)
    ne = sqrt(e[0] ** 2 + e[1] ** 2)
    rr = c / sqrt(v1 * v2)
    # (c) ソフトマージン SVM
    w, b0 = (2, -1), -1
    pts = [((1, 0), 1), ((2, 1), 1), ((0, 0), -1), ((0.5, 0.5), -1), ((1, 2), 1)]
    fv = [w[0] * x[0] + w[1] * x[1] + b0 for x, _ in pts]
    mv = [y * f_ for (_, y), f_ in zip(pts, fv)]
    lv = [max(0, 1 - m_) for m_ in mv]
    out = {"v1": v1, "v2": v2, "c": c, "tr": tr, "det": det, "lam1": s(l1, 2), "lam2": s(l2, 2), "pc": s(l1 / tr, 3),
           "e11": s(e[0] / ne, 3), "e12": s(e[1] / ne, 3), "r": s(rr, 1), "pr": s((1 + rr) / 2, 1), "w_r": s(rr, 1),
           "w1": "2", "w2s": "-", "bs": "-1", "hinge": s(sum(lv), 1), "mw": s(2 / 2.236, 3), "w_mw": s(1 / 2.236, 3),
           "w_all": s(sum(1 - m_ for m_ in mv), 1)}
    for k in range(5):
        out[f"f{k + 1}"] = s(fv[k], 1)
        out[f"m{k + 1}"] = s(mv[k], 1)
        out[f"l{k + 1}"] = s(lv[k], 1)
    return out


def p14():
    # (a) ガワーの類似度
    ga, gb, rg = 2.0, 6.5, 10
    fa, fb, rf = 1, 4, 6
    sg = 1 - abs(ga - gb) / rg
    sf = 1 - abs(fa - fb) / rf
    g = (1 + sg + sf) / 3
    # (b) 潜在的な結果
    y1 = [3, 3, 4, 2, 1, 1]
    y0 = [5, 4, 6, 2, 1, 2]
    dif = [a - b for a, b in zip(y1, y0)]
    ate = Fraction(sum(dif), 6)
    att = Fraction(sum(dif[:3]), 3)
    m1 = Fraction(sum(y1[:3]), 3)
    m0 = Fraction(sum(y0[3:]), 3)
    naive = m1 - m0
    sel = Fraction(sum(y0[:3]), 3) - m0
    assert naive == att + sel
    return {"ga": s(ga, 1), "gb": s(gb, 1), "rg": rg, "fa": fa, "fb": fb, "rf": rf, "sg": s(sg, 2), "sf": s(sf, 2),
            "ans": r(g, 3), "ans4": s(g, 4), "ans_s": s(g, 3),
            "y1": " | ".join(map(str, y1)), "y0": " | ".join(map(str, y0)), "diff": "、".join(map(str, dif)), "sdiff": sum(dif),
            "d1": dif[0], "d2": dif[1], "d3": dif[2], "a1": y1[0], "a2": y1[1], "a3": y1[2], "b4": y0[3], "b5": y0[4], "b6": y0[5],
            "ate": s(ate, 2), "att": s(att, 2), "m1": s(m1, 2), "m0": s(m0, 2), "naive": s(naive, 2), "sel": s(sel, 2),
            "w_naive": s(Fraction(sum(y1[:3]), 3) - Fraction(sum(y0[:3]), 3), 2)}


def values():
    return {f"s26p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 15)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
