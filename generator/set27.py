"""模擬試験 第27回（set27）の数値。values() が {大問id: {名前: 値}} を返す。

題材：地域・観光・自治体の調査。
問題文に示した分布の値（Φ の値・分布の点・対数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from math import comb, sqrt


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
    # (a) 期待度数とカイ二乗近似の条件
    a1, a2, b1, b2 = 9, 3, 2, 6
    ra, rb = a1 + a2, b1 + b2
    c1, c2 = a1 + b1, a2 + b2
    n = ra + rb
    e = {"ea1": ra * c1 / n, "ea2": ra * c2 / n, "eb1": rb * c1 / n, "emin": rb * c2 / n}
    assert min(e.values()) == e["emin"] and sum(v < 5 for v in e.values()) == 2
    # (b) ベータ事後分布と信用区間（分位点は scipy で求めた値を 3 桁で示す）
    nb, yb = 50, 30
    pa, pb = 1 + yb, 1 + nb - yb
    q025, q975 = 0.461, 0.724      # Beta(31,21)
    w025, w975 = 0.462, 0.730      # Beta(30,20)
    return {"a1": a1, "a2": a2, "b1": b1, "b2": b2, "ra": ra, "rb": rb, "c1": c1, "c2": c2, "n": n,
            **{k: s(v, 1) for k, v in e.items()},
            "nb": nb, "yb": yb, "nyb": nb - yb, "pa": pa, "pb": pb, "pab": pa + pb, "pmean": s(Fraction(pa, pa + pb), 3),
            "ya": yb, "yb2": nb - yb, "q025": s(q025, 3), "q975": s(q975, 3), "w025": s(w025, 3), "w975": s(w975, 3)}


def p02():
    # (a) 独立な正規分布の和
    m1, s1, m2, s2, c = 30, 4, 20, 3, 60
    ph = {1.0: 0.8413, 1.43: 0.9236, 2.0: 0.9772}
    mt, vt = m1 + m2, s1 ** 2 + s2 ** 2
    st = sqrt(vt)
    z = (c - mt) / st
    assert z == 2.0
    ok = 1 - ph[2.0]
    vals = [ok, 1 - ph[1.43], 1 - ph[1.0], ph[2.0], 0.5 - ph[2.0] + 0.5]
    # (b) TOST と 90% 信頼区間
    d, se = 1.2, 0.8
    lo, hi = d - 1.645 * se, d + 1.645 * se
    delta = max(abs(lo), abs(hi))
    # (c) 一様分布の最大値の極限分布
    nn, L, g = 200, 600, 3
    x = nn * g / L
    ans = math.exp(-x)
    return {"m1": m1, "s1": s1, "m2": m2, "s2": s2, "c": c, "mt": mt, "vt": vt, "st": s(st, 0),
            "ph1": s(ph[1.0], 4), "ph143": s(ph[1.43], 4), "ph2": s(ph[2.0], 4),
            "ok": s(ok, 4), "w_sdsum": s(1 - ph[1.43], 4), "w_one": s(1 - ph[1.0], 4), "w_phi": s(ph[2.0], 4),
            "w_half": s(2 * (1 - ph[2.0]), 4),
            "d": s(d, 1), "se": s(se, 1), "lo": s(lo, 3), "hi": s(hi, 3), "delta": r(delta, 2),
            "nn": nn, "L": L, "g": g, "Lg": L - g, "gL": s(g / L, 3), "x": s(x, 0), "ans": r(ans, 3), "ans4": s(ans, 4),
            "ans_s": s(ans, 3), "exact": s(((L - g) / L) ** nn, 4)}


def p03():
    # (b) 幾何分布の最小値
    p1, p2 = 0.2, 0.3
    q = (1 - p1) * (1 - p2)
    pm = 1 - q
    et = 1 / pm
    return {"p1": p1, "p2": p2, "qq": s(q, 2), "pm": s(pm, 2), "et": s(et, 2),
            "c_sum": s(1 / (p1 + p2), 2), "c_add": s(1 / p1 + 1 / p2, 2), "c_max": s(1 / max(p1, p2), 2),
            "c_fail": s(q / pm, 2), "p1m": s(1 - p1, 1), "p2m": s(1 - p2, 1)}


def p04():
    # (a) 積の期待値
    EN, ES, cov = 1000, 2.0, -50
    ans = EN * ES + cov
    # (b) レイリー分布の尺度母数の最尤推定
    rs = [1.2, 0.8, 2.0, 1.5, 1.0]
    n = len(rs)
    sr2 = sum(v * v for v in rs)
    rb = sum(rs) / n
    mle = sr2 / (2 * n)
    mm = (rb / 1.2533) ** 2
    var = sum((v - rb) ** 2 for v in rs) / (n - 1)
    distinct([mle, sr2 / n, mm, var], 3)
    # (c) グラフ上のランダムウォーク（駅・城・温泉街・美術館）
    deg = {"駅": 3, "城": 2, "温泉街": 3, "美術館": 2}
    tot = sum(deg.values())
    return {"EN": EN, "ES": s(ES, 1), "cov": cov, "acov": -cov, "prod": s(EN * ES, 0), "ans": r(ans, 0),
            "n": n, "n2": 2 * n, "rs": "、".join(s(v, 1) for v in rs), "sr2": s(sr2, 2), "rb": s(rb, 2),
            "mle": s(mle, 3), "w_n": s(sr2 / n, 3), "w_mm": s(mm, 3), "w_var": s(var, 3),
            "po": s(Fraction(deg["温泉街"], tot), 1), "w_deg": s(Fraction(deg["温泉街"] ** 2, sum(v * v for v in deg.values())), 3)}


def p05():
    # (a) 二相抽出による層別推定
    n1, u1, m1 = 500, 300, 200
    u2, m2 = 40, 40
    pu, pm = 0.55, 0.80
    wu, wm = u1 / n1, m1 / n1
    ans = wu * pu + wm * pm
    # (b) 少なくとも 1 人含む確率
    p = 0.02
    ln05, ln98 = -2.9957, -0.0202
    ratio = ln05 / ln98
    nmin = math.ceil(ratio)
    return {"n1": n1, "u1": u1, "m1": m1, "u2": u2, "m2": m2, "pu": s(pu, 2), "pm": s(pm, 2), "wu": s(wu, 1), "wm": s(wm, 1),
            "t1": s(wu * pu, 2), "t2": s(wm * pm, 2), "ans": r(ans, 3), "ans_s": s(ans, 3),
            "w_raw": s((u2 * pu + m2 * pm) / (u2 + m2), 3),
            "p": p, "ln05": s(ln05, 4), "ln98": s(ln98, 4), "ratio": s(ratio, 1), "nmin": nmin, "c_low": nmin - 1,
            "c_rule3": int(3 / p), "c_wrong": math.ceil(0.95 / p), "c_ln95": math.ceil(-0.0513 / ln98)}


def p06():
    # (a) 自由度調整済み決定係数と部分 F
    n, ra, rb = 30, 0.62, 0.64
    aa = 1 - (1 - ra) * (n - 1) / (n - 3)
    ab = 1 - (1 - rb) * (n - 1) / (n - 5)
    F = ((rb - ra) / 2) / ((1 - rb) / (n - 5))
    assert ab < aa and F < 3.39
    # (b) 読めなくなった残差
    x = [2, 3, 5, 6, 8, 9, 11, 12]
    e = [1.2, -0.8, 0.5, -1.5, 0.9, -1.5, 1.6, -0.4]
    assert abs(sum(e)) < 1e-12 and abs(sum(xi * ei for xi, ei in zip(x, e))) < 1e-9
    known = [(xi, ei) for xi, ei in zip(x, e) if xi not in (9, 11)]
    se_known = sum(ei for _, ei in known)
    sxe_known = sum(xi * ei for xi, ei in known)
    return {"n": n, "ra": s(ra, 2), "rb": s(rb, 2), "aa": s(aa, 3), "ab": s(ab, 3), "F": s(F, 2), "w_F": s(F * 2, 2),
            "w_aa": s(ra, 2), "w_ab": s(rb, 2),
            **{f"x{i}": xi for i, xi in enumerate(x)}, **{f"e{i}": s(ei, 1) for i, ei in enumerate(e)},
            "sek": s(se_known, 1), "sxek": s(sxe_known, 1), "msek": s(-se_known, 1), "msxek": s(-sxe_known, 1),
            "e9": s(e[5], 2), "e11": s(e[6], 2), "c_e9": s(e[5], 2), "c_e11": s(e[6], 2), "c_half": s(-se_known / 2, 2),
            "c_flip": s(-e[5], 2), "c_sum": s(se_known, 2)}


def p07():
    a, b, c, d = 30, 70, 15, 85
    r1, r2 = a + b, c + d
    p1, p0 = a / r1, c / r2
    OR = (p1 / (1 - p1)) / (p0 / (1 - p0))
    RR = p1 / p0
    return {"a": a, "b": b, "c": c, "d": d, "r1": r1, "r2": r2, "n": r1 + r2, "p1": s(p1, 2), "p0": s(p0, 2),
            "OR": s(OR, 2), "RR": s(RR, 2), "b1": s(0.887, 3), "g1": s(0.693, 3), "w_diff": s(p1 - p0, 2)}


def p08():
    # (a) 分位点回帰
    a = {1: 1.0, 5: 2.0, 9: 4.0}
    bb = {1: 0.5, 5: 1.0, 9: 2.0}
    x0 = 4
    q = {k: a[k] + bb[k] * x0 for k in a}
    q1 = {k: a[k] + bb[k] * 1 for k in a}
    # (b) テスト誤差の最小値の偏り
    emax, sd = 1.539, 0.10
    return {"a1": s(a[1], 1), "a5": s(a[5], 1), "a9": s(a[9], 1), "b1": s(bb[1], 1), "b5": s(bb[5], 1), "b9": s(bb[9], 1),
            "x0": x0, "q1": s(q[1], 1), "m4": s(q[5], 1), "q9": s(q[9], 1), "w4": s(q[9] - q[1], 1),
            "q1x1": s(q1[1], 1), "q9x1": s(q1[9], 1), "w1": s(q1[9] - q1[1], 1), "w_wrong": s((bb[9] - bb[1]) * x0, 1),
            "emax": s(emax, 3), "sd": s(sd, 2), "ebest": s(1 - emax * sd, 2), "gap": s(emax * sd, 2), "emin": "0.85",
            "w_1sd": s(1 - sd, 2)}


def p09():
    # (a) 連の数の正確な分布（H 5 個・L 5 個）
    def ways(rr, n1=5, n2=5):
        if rr % 2 == 0:
            k = rr // 2
            return 2 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1)
        k = (rr - 1) // 2
        return comb(n1 - 1, k) * comb(n2 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(n2 - 1, k)
    tot = comb(10, 5)
    assert sum(ways(rr) for rr in range(2, 11)) == tot
    R = 3
    p = Fraction(ways(2) + ways(3), tot)
    le5 = Fraction(sum(ways(rr) for rr in range(2, 6)), tot)
    # (b) ホルム法と BH 法
    ps = [0.004, 0.011, 0.028, 0.032, 0.20, 0.64]
    m = len(ps)
    nh = 0
    for i, pv in enumerate(ps):
        if pv <= 0.05 / (m - i):
            nh += 1
        else:
            break
    nb = max([i + 1 for i, pv in enumerate(ps) if pv <= 0.05 * (i + 1) / m] or [0])
    w_h = 0
    for i, pv in enumerate(ps):          # 基準を逆順（0.05/1, 0.05/2, ...）に使った誤り
        if pv <= 0.05 / (i + 1):
            w_h += 1
        else:
            break
    w_b = 0
    for i, pv in enumerate(ps):
        if pv <= 0.05 * (i + 1) / m:
            w_b += 1
        else:
            break
    assert (nh, nb, w_h, w_b) == (1, 4, 2, 2)
    return {"R": R, "p": s(p, 3), "p4": s(p, 4), "w_eq": s(Fraction(ways(3), tot), 3), "w_2": s(Fraction(ways(2), tot), 4),
            "w_half": s(le5, 3), "ps": "、".join(str(v) for v in ps), "p1": ps[0], "p2": ps[1], "p3": ps[2], "p4v": ps[3],
            "nh": nh, "nb": nb, "w_h": w_h, "w_b": w_b, "w_raw": sum(pv < 0.05 for pv in ps),
            "bh": "、".join(s(0.05 * (i + 1) / m, 4) for i in range(m))}


def p10():
    return {}


def p11():
    # (a) 主成分の数の基準
    ev = [2.6, 1.3, 0.95, 0.6, 0.35, 0.2]
    assert abs(sum(ev) - 6) < 1e-12
    cum = [sum(ev[:k + 1]) / 6 for k in range(6)]
    k1 = sum(v >= 1 for v in ev)
    k2 = next(k + 1 for k, c in enumerate(cum) if c >= 0.8)
    # (b) 回帰法とバートレット法の因子得点
    lam = [0.8, 0.7, 0.6]
    x = [1.0, 0.5, 1.2]
    psi = [1 - v * v for v in lam]
    a = [l_ / p_ for l_, p_ in zip(lam, psi)]
    sax = sum(ai * xi for ai, xi in zip(a, x))
    S = sum(l_ * l_ / p_ for l_, p_ in zip(lam, psi))
    return {"ev": "、".join(s(v, 2) for v in ev), "k1": k1, "k2": k2, "c2": s(cum[1], 2),
            "big": "、".join(s(v, 1) for v in ev if v >= 1), "cum": "、".join(s(c, 3) for c in cum[:4]),
            "elbow": "第 1 から第 2 で大きく下がり、第 3 以降はなだらかで、折れ目の位置は見る人によって 2 とも 3 とも読める",
            "l1": s(lam[0], 1), "l2": s(lam[1], 1), "l3": s(lam[2], 1), "x1": s(x[0], 1), "x2": s(x[1], 1), "x3": s(x[2], 1),
            "p1": s(psi[0], 2), "p2": s(psi[1], 2), "p3": s(psi[2], 2), "a1": s(a[0], 3), "a2": s(a[1], 3), "a3": s(a[2], 3),
            "sax": s(sax, 3), "S": s(S, 3), "freg": s(sax / (1 + S), 2), "fbar": s(sax / S, 2),
            "w_mean": s(sum(x) / 3, 2), "w_sum": s(sax, 2)}


def p12():
    T = [[20, 10, 5], [10, 15, 10], [5, 10, 15]]
    n = sum(map(sum, T))
    rsum = [sum(rw) for rw in T]
    csum = [sum(T[i][j] for i in range(3)) for j in range(3)]
    sc = [1, 2, 3]
    ub = sum(sc[i] * rsum[i] for i in range(3)) / n
    vb = sum(sc[j] * csum[j] for j in range(3)) / n
    suv = sum(T[i][j] * sc[i] * sc[j] for i in range(3) for j in range(3))
    Suv = suv - n * ub * vb
    Suu = sum(rsum[i] * sc[i] ** 2 for i in range(3)) - n * ub ** 2
    Svv = sum(csum[j] * sc[j] ** 2 for j in range(3)) - n * vb ** 2
    rr = Suv / sqrt(Suu * Svv)
    M2 = (n - 1) * rr ** 2
    out = {"n": n, "ub": s(ub, 2), "vb": s(vb, 2), "suv": suv, "Suv": s(Suv, 2), "Suu": s(Suu, 2), "Svv": s(Svv, 2),
           "r": s(rr, 4), "M2": r(M2, 2), "M2_s": s(M2, 2)}
    for i in range(3):
        out[f"r{i + 1}"] = rsum[i]
        out[f"c{i + 1}"] = csum[i]
        for j in range(3):
            out[f"t{i + 1}{j + 1}"] = T[i][j]
    return out


def p13():
    Q = [52.1, 50.8, 53.0, 51.5, 52.6]
    m, U = len(Q), 4.0
    qb = sum(Q) / m
    ssq = sum((v - qb) ** 2 for v in Q)
    B = ssq / (m - 1)
    bb = (1 + 1 / m) * B
    T = U + bb
    g = bb / T
    return {"m": m, "qs": "、".join(s(v, 1) for v in Q), "U": s(U, 1), "qb": s(qb, 1), "ssq": s(ssq, 2), "B": s(B, 4),
            "bb": s(bb, 3), "T": s(T, 3), "gamma": r(g, 3), "gamma4": s(g, 4), "gamma_s": s(g, 3)}


def p14():
    n, rxm, rxy, rmy = 400, 0.5, 0.4, 0.5
    b = (rmy - rxy * rxm) / (1 - rxm ** 2)
    c = (rxy - rmy * rxm) / (1 - rxm ** 2)
    ind = rxm * b
    assert abs(c + ind - rxy) < 1e-12
    return {"n": n, "rxm": s(rxm, 1), "rxy": s(rxy, 1), "rmy": s(rmy, 1), "b": s(b, 2), "c": s(c, 2), "ind": r(ind, 2), "ind_s": s(ind, 2)}


def values():
    return {f"s27p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 15)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
