"""模擬試験 第30回（set30）の数値。values() が {大問id: {名前: 値}} を返す。

題材：スポーツ・フィットネス。
問題文に示した分布の値（Φ・Φ^{-1}・t 点・F 点・e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) ちょうど 1 人だけ成功
    pa, pb = 0.7, 0.6
    a_only, b_only = pa * (1 - pb), (1 - pa) * pb
    one = a_only + b_only
    atleast = 1 - (1 - pa) * (1 - pb)
    both = pa * pb
    vals = [one, atleast, both, 1 - both]
    distinct(vals, 2)
    # (b) 条件付き期待値
    P = [[0.10, 0.15, 0.05], [0.05, 0.20, 0.15], [0.02, 0.08, 0.20]]
    ce = [sum((j + 1) * P[i][j] for j in range(3)) / sum(P[i]) for i in range(3)]
    ey = sum((j + 1) * sum(P[i][j] for i in range(3)) for j in range(3))
    return {"pa": s(pa, 1), "pb": s(pb, 1), "a_only": s(a_only, 2), "b_only": s(b_only, 2), "one": s(one, 2),
            "atleast": s(atleast, 2), "both": s(both, 2), "sum": s(pa + pb, 1), "w_ind": s(1 - both, 2),
            "ans": r(ce[1], 2), "ans_s": s(ce[1], 2), "e0": s(ce[0], 2), "e2": s(ce[2], 2), "ey": s(ey, 2)}


def p02():
    return {}


def p03():
    # (a) プロビット変換
    p1, p2 = 0.30, 0.60
    z1, z2 = -0.524, 0.253
    shift = z2 - z1
    # (b) ワイブル分布の最小値
    sc = 10 / sqrt(5)
    p3 = 0.638
    return {"p1": s(p1, 2), "p2": s(p2, 2), "shift": s(shift, 3), "w_diff": s(p2 - p1, 2), "w_ratio": s(p2 / p1, 1),
            "w_sum": s(z1 + z2, 3), "sc": s(sc, 2), "p3": s(p3, 3), "w_p3": s(math.exp(-(3 / 2) ** 2), 3), "w_one": s(0.914, 3)}


def p04():
    # (b) 二項分布と負の二項分布のフィッシャー情報量
    n, rr, p = 40, 10, 0.25
    i1 = n / (p * (1 - p))
    i2 = rr / (p * p * (1 - p))
    assert abs(i1 - i2) < 1e-9
    return {"n": n, "r": rr, "p": s(p, 2), "i1": s(i1, 1), "i2": s(i2, 1), "en": s(rr / p, 0),
            "w_i2": s(rr / (p * (1 - p)), 1), "w_i3": s(rr / (p * p * (1 - p) ** 2), 1), "w_i1": s(n / p, 1)}


def p05():
    # (a) 対応のある差の区間
    d, sd, n = 2.4, 1.5, 10
    se = sd / sqrt(n)
    h = 2.262 * se
    # (b) ランダム化検定
    p5, p6, pm5 = 0.0527, 0.0166, 0.0361
    g = (0.05 - p6) / pm5
    return {"d": s(d, 1), "sd": s(sd, 1), "se": s(se, 4), "h": s(h, 3), "lo": r(d - h, 2), "lo_s": s(d - h, 2), "hi": s(d + h, 2),
            "p5": s(p5, 4), "p6": s(p6, 4), "pm5": s(pm5, 4), "num": s(0.05 - p6, 4), "g": s(g, 2), "g4": s(g, 4),
            "w_g": s((0.05 - p6) / p5, 2)}


def p06():
    # (a) 一様性の検定
    a = [12, 20, 28]
    b = [100 - v for v in a]
    ptot = sum(a) / 300
    e1, e2 = 100 * ptot, 100 * (1 - ptot)
    X2 = sum((v - e1) ** 2 / e1 for v in a) + sum((v - e2) ** 2 / e2 for v in b)
    half = sum((v - e1) ** 2 / e1 for v in a)
    # (b) ヨンクヒール・タプストラ
    g = [[3, 5, 8], [6, 7, 10], [9, 11, 12]]
    U = lambda x, y: sum((u < v) + 0.5 * (u == v) for u in x for v in y)
    u12, u13, u23 = U(g[0], g[1]), U(g[0], g[2]), U(g[1], g[2])
    J = u12 + u13 + u23
    rev = 27 - J
    return {"a1": a[0], "a2": a[1], "a3": a[2], "b1": b[0], "b2": b[1], "b3": b[2], "ptot": s(ptot, 1), "e1": s(e1, 0),
            "e2": s(e2, 0), "X2": s(X2, 1), "w_half": s(half, 1),
            "g1": "、".join(map(str, g[0])), "g2": "、".join(map(str, g[1])), "g3": "、".join(map(str, g[2])),
            "u12": s(u12, 0), "u13": s(u13, 0), "u23": s(u23, 0), "J": s(J, 0), "w_one": s(u13, 0), "w_rev": s(rev, 0),
            "w_max": 27, "w_mean": "13.5"}


def p07():
    qRT, qTR, qTI, qIR = Fraction(1, 2), Fraction(3, 10), Fraction(1, 10), Fraction(3, 5)
    # 流れのつり合い：π_I = (qTI/qIR) π_T、π_R = (qTR π_T + qIR π_I)/qRT
    cI = qTI / qIR
    cR = (qTR + qIR * cI) / qRT
    piT = 1 / (cR + 1 + cI)
    piI, piR = cI * piT, cR * piT
    # 埋め込まれた連鎖（推移の回数の割合）：R→T、T→R（3/4）・T→I（1/4）、I→R
    eT = Fraction(1) / (1 + 1 + Fraction(1, 4))
    eI = eT / 4
    return {"qRT": float(qRT), "qTR": float(qTR), "qTI": float(qTI), "qIR": float(qIR), "piT": s(piT, 3), "piI": s(piI, 3),
            "piR": s(piR, 3), "w_uni": s(Fraction(1, 3), 3), "w_rate": s(qTI / (qTI + qIR), 3), "w_jump": s(eI, 3)}


def p08():
    # (a) 回帰全体の F
    n, ssr, sse = 30, 45.0, 78.0
    F = (ssr / 3) / (sse / 26)
    # (b) 原点を通る回帰の決定係数
    m, sy2, yb, rss = 20, 1060.0, 7.0, 40.0
    syy = sy2 - m * yb ** 2
    return {"n": n, "ssr": s(ssr, 1), "sse": s(sse, 1), "msr": s(ssr / 3, 1), "mse": s(sse / 26, 1), "F": s(F, 2),
            "w_F": s(ssr / sse, 2), "w_F2": s((ssr / 3) / (sse / 29), 2), "r2": s(ssr / (ssr + sse), 3),
            "m": m, "sy2": s(sy2, 0), "yb": s(yb, 1), "rss": s(rss, 0), "r0": r(1 - rss / sy2, 3), "r0_s": s(1 - rss / sy2, 3),
            "syy": s(syy, 0), "rc": s(1 - rss / syy, 3)}


def p09():
    # (a) トービットの限界効果
    xb, sg, b = 1.0, 2.0, 0.6
    me = b * 0.6915
    phi05 = math.exp(-0.125) / sqrt(2 * math.pi)
    # (b) ポアソン回帰の交互作用
    b1, b2, b12 = 0.4, 0.3, -0.2
    rr = 1.6487
    return {"xb": s(xb, 1), "s": s(sg, 1), "b": s(b, 1), "me": s(me, 3), "me4": s(me, 4), "w_phi": s(b * phi05, 3),
            "w_half": s(b * (1 - 0.6915), 3), "w_sig": s(b / sg, 2),
            "b1": s(b1, 1), "b2": s(b2, 1), "b12s": "-0.2", "rr": r(rr, 2), "rr4": s(rr, 4), "rr_s": s(rr, 2)}


def p10():
    # (a) 分割区法
    SBk, SA, SE1, SB, SAB, SE2 = 30.0, 60.0, 24.0, 40.0, 18.0, 27.0
    ms2 = SE2 / 9
    FB = SB / ms2
    FAB = (SAB / 2) / ms2
    FA = (SA / 2) / (SE1 / 6)
    # (b) 層別平均
    W = [0.5, 0.3, 0.2]
    y = [42, 55, 70]
    t = [w * v for w, v in zip(W, y)]
    return {"SBk": s(SBk, 0), "SA": s(SA, 0), "SE1": s(SE1, 0), "SB": s(SB, 0), "SAB": s(SAB, 0), "SE2": s(SE2, 0),
            "ms2": s(ms2, 1), "FB": s(FB, 2), "FAB": s(FAB, 2), "FA": s(FA, 2), "w_FB1": s(SB / (SE1 / 6), 2),
            "w_pool": s(SB / ((SE1 + SE2) / 15), 2), "w_ss": s(SB / SE2, 2),
            "W1": W[0], "W2": W[1], "W3": W[2], "y1": y[0], "y2": y[1], "y3": y[2], "t1": s(t[0], 1), "t2": s(t[1], 1),
            "t3": s(t[2], 1), "ans": r(sum(t), 1), "ans_s": s(sum(t), 1), "w_simple": s(sum(y) / 3, 1)}


def p11():
    # (a) 適合率
    tpr, fpr = 0.9, 0.05
    prec = lambda pr: tpr * pr / (tpr * pr + fpr * (1 - pr))
    wrong = lambda pr: tpr * pr / (tpr * pr + fpr)
    # (b) デンドログラムを切る
    hs = [1.0, 1.5, 2.2, 4.0, 6.5]
    cut = 3.0
    below = [h for h in hs if h < cut]
    # (c) 斜交回転
    phi = 0.5
    s22 = 0.7 * phi + 0.1
    cross = 2 * 0.7 * 0.1 * phi
    h2 = 0.49 + 0.01 + cross
    return {"tpr": s(tpr, 1), "fpr": s(fpr, 2), "pr10": s(prec(0.1), 3), "pr1": s(prec(0.01), 3), "w10": s(wrong(0.1), 3),
            "w1": s(wrong(0.01), 3), "w_spec": s(1 - fpr, 2),
            "hs": "、".join(s(h, 1) for h in hs), "cut": s(cut, 1), "below": "、".join(s(h, 1) for h in below), "nb": len(below),
            "k": 6 - len(below),
            "phi": s(phi, 1), "s22": s(s22, 2), "cross": s(cross, 2), "h2": s(h2, 2), "w_h2": s(0.5, 2), "w_s": s(0.1 * phi + 0.7, 2)}


def p12():
    # (a) 局所水準モデルの定常カルマン利得
    se2, sh2 = 0.04, 0.09
    q = sh2 / se2
    P = (q + sqrt(q * q + 4 * q)) / 2
    K = P / (P + 1)
    assert abs(P - 3) < 1e-12
    # (b) 左側切断つきカプラン・マイヤー
    data = [(0, 3, 1), (0, 8, 0), (2, 5, 1), (4, 6, 1), (0, 10, 0), (5, 9, 1)]
    S = 1.0
    naive = 1.0
    for t in sorted({e for _, e, d in data if d == 1}):
        n_ = sum(1 for a, e, _ in data if a < t <= e)
        d_ = sum(1 for a, e, dd in data if e == t and dd == 1)
        S *= 1 - d_ / n_
        n0 = sum(1 for a, e, _ in data if t <= e)
        naive *= 1 - d_ / n0
    return {"se2": s(se2, 2), "sh2": s(sh2, 2), "q": s(q, 2), "P": s(P, 0), "K": s(K, 2), "w_q": s(q, 2), "w_P": s(P, 1),
            "S9": r(S, 3), "S9_4": s(S, 4), "S9_s": s(S, 3), "w_naive": s(naive, 3)}


def p13():
    nk = [20, 20, 20, 20, 19]
    A = [4.1, 3.8, 4.5, 4.0, 4.2]
    B = [3.6, 4.4, 3.9, 4.3, 4.0]
    cva = sum(n * v for n, v in zip(nk, A)) / sum(nk)
    cvb = sum(n * v for n, v in zip(nk, B)) / sum(nk)
    assert cvb < cva
    return {"a": " | ".join(s(v, 1) for v in A), "b": " | ".join(s(v, 1) for v in B), "a5": s(A[4], 1), "b5": s(B[4], 1),
            "a_sum4": "+".join(s(v, 1) for v in A[:4]), "b_sum4": "+".join(s(v, 1) for v in B[:4]),
            "cva": s(cva, 2), "cvb": s(cvb, 2), "cva4": s(cva, 4), "cvb4": s(cvb, 4), "sa": s(sum(A) / 5, 2), "sb": s(sum(B) / 5, 2),
            "w_a": s(min(A), 1), "w_b": s(min(B), 1),
            "m_all": "0.084", "m_post": "0.062", "w_mid": "0.073"}


def values():
    return {f"s30p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 14)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
