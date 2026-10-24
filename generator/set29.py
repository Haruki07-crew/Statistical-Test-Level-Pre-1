"""模擬試験 第29回（set29）の数値。values() が {大問id: {名前: 値}} を返す。

題材：健康診断・栄養・スポーツ医学・睡眠。
問題文に示した数表・関数の値（e^x・Φ・√ など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # (a) 陽性尤度比による検査後確率
    pre, sens, spec = 0.10, 0.80, 0.90
    LR = sens / (1 - spec)
    po0 = pre / (1 - pre)
    po1 = po0 * LR
    post = po1 / (1 + po1)
    lrneg = (1 - sens) / spec
    vals = [post, sens, po0 * lrneg / (1 + po0 * lrneg), pre * sens, LR / (1 + LR)]
    distinct(vals, 3)
    # (b) 半正規分布
    sg = 0.3
    ea = sg * 0.7979
    pa = 2 * (1 - 0.8413)
    return {"pre": s(pre, 2), "sens": s(sens, 2), "spec": s(spec, 2), "LR": s(LR, 0), "po0": s(po0, 4), "po1": s(po1, 4),
            "post": s(post, 3), "post4": s(post, 4), "w_sens": s(sens, 3), "w_lrneg": s(vals[2], 3), "w_mult": s(vals[3], 3),
            "w_ppv50": s(vals[4], 3),
            "sg": s(sg, 1), "ea": s(ea, 3), "pa": s(pa, 4), "w_one": s(pa / 2, 4), "w_half": s(sg / 2, 2)}


def p02():
    # (a) 共分散行列から相関係数
    S = [[64, 30, 24], [30, 25, 15], [24, 15, 36]]
    sd = [sqrt(S[i][i]) for i in range(3)]
    rr = lambda i, j: S[i][j] / (sd[i] * sd[j])
    # (b) 目的変数の測定誤差と決定係数
    r2t, rel = 0.50, 0.80
    return {"s11": S[0][0], "s12": S[0][1], "s13": S[0][2], "s22": S[1][1], "s23": S[1][2], "s33": S[2][2],
            "sd2": s(sd[1], 0), "sd3": s(sd[2], 0), "ans": r(rr(1, 2), 2), "ans_s": s(rr(1, 2), 2),
            "r12": s(rr(0, 1), 2), "r13": s(rr(0, 2), 2),
            "r2t": s(r2t, 2), "rel": s(rel, 2), "ans2": r(r2t * rel, 2), "ans2_s": s(r2t * rel, 2)}


def p03():
    mean = 20
    q = 0.7047      # e^{-0.35}
    p = 1 - q
    EK = q / p
    return {"mean": mean, "e035": s(q, 4), "p": s(p, 4), "EK": s(EK, 2), "EK4": s(EK, 4), "w_div": s(mean / 7, 2),
            "w_inv": s(1 / p, 2), "w_swap": s(p / q, 2)}


def p04():
    # Beta(1,3) とベータ二項分布（n=4）
    p0 = Fraction(3, 7)
    v = Fraction(6, 5)
    bp0 = Fraction(3, 4) ** 4
    bv = Fraction(3, 4)
    return {"p0": s(p0, 3), "p04": s(p0, 4), "v": s(v, 2), "bp0": s(bp0, 3), "bv": s(bv, 2), "w_p0": s(Fraction(3, 4), 3),
            "w_v": s(Fraction(1, 2), 2)}


def p05():
    # (a) 等相関の測定値の平均の分散
    n, s2, rho = 25, 100, 0.1
    v = s2 * (1 + (n - 1) * rho) / n
    # (b) 正規分布の分散の推定
    m = 10
    vs2 = Fraction(2, m - 1)
    cr = Fraction(2, m)
    vm = Fraction(m - 1, m) ** 2 * vs2
    b2 = Fraction(1, m * m)
    mse = vm + b2
    assert mse == Fraction(2 * m - 1, m * m)
    return {"n": n, "n1": n - 1, "s2": s2, "rho": s(rho, 1), "v": s(v, 1), "ans": r(sqrt(v), 2), "ans4": s(sqrt(v), 4),
            "ans_s": s(sqrt(v), 2), "w_ind": s(sqrt(s2 / n), 2), "lim": s(rho * s2, 0), "limsd": s(sqrt(rho * s2), 2),
            "m": m, "m1": m - 1, "vs2": s(vs2, 3), "cr": s(cr, 3), "vm": s(vm, 3), "b2": s(b2, 3), "mse": s(mse, 3),
            "w_mse": s(vm, 3)}


def p06():
    x = 4
    G = lambda l: 2 * (x * math.log(x / l) - x + l)
    out = {f"g{int(round(l * 100)):d}": s(G(l), 2) for l in (1.0, 1.25, 1.5, 9.0, 9.4, 10.0)}
    assert float(out["g125"]) < 3.84 < float(out["g100"]) and float(out["g900"]) < 3.84 < float(out["g940"])
    return out


def p07():
    # (a) 非劣性
    d, se, Delta = -1.0, 0.8, 3.0
    z = (d + Delta) / se
    # (b) 2 標本のホテリングの T²
    n = 20
    d1, d2 = 4.0, -3.0
    s11, s12, s22 = 100, 48, 64
    det = s11 * s22 - s12 * s12
    q = (s22 * d1 ** 2 - 2 * s12 * d1 * d2 + s11 * d2 ** 2) / det
    T2 = n * n / (2 * n) * q
    F = (2 * n - 2 - 1) / (2 * (2 * n - 2)) * T2
    t1, t2 = d1 / sqrt(s11 * 0.1), abs(d2) / sqrt(s22 * 0.1)
    assert F > 3.25 and t1 < 2.024 and t2 < 2.024
    return {"d": s(d, 1), "se": s(se, 1), "Delta": s(Delta, 1), "dD": s(d + Delta, 1), "z": s(z, 2), "zs": s(d / se, 2),
            "w_z": s((d - Delta) / se, 2), "lo": s(d - 1.96 * se, 3),
            "n": n, "d1": s(d1, 0), "d2": s(d2, 0), "d2s": s(abs(d2), 0), "s11": s11, "s12": s12, "s22": s22, "det": det,
            "q": s(q, 4), "T2": s(T2, 2), "F": s(F, 2), "t1": s(t1, 2), "t2": s(t2, 2), "w_T2": s(t1 ** 2 + t2 ** 2, 2)}


def p08():
    a, b, c, d = 6, 14, 14, 6
    X2 = sum((o - 10) ** 2 / 10 for o in (a, b, c, d))
    return {"a": a, "b": b, "c": c, "d": d, "X2": s(X2, 1), "w_half": s(X2 / 4, 1)}


def p09():
    p = Fraction(2, 5)
    return {"p": s(p, 1), "w_swap": s(1 - p, 1), "w_ratio": s(Fraction(3, 5), 1)}


def p10():
    # (b) 比例オッズモデル
    c1, c2, b = -1.0, 1.5, 0.8
    p3x1 = 1 / (1 + 2.0138)
    p3x0 = 1 / (1 + 4.4817)
    # (c) 主成分回帰
    lam = [2.20, 0.75, 0.05]
    inv = [1 / v for v in lam]
    return {"c1": s(c1, 1), "c2": s(c2, 1), "b": s(b, 1), "p3x1": s(p3x1, 3), "p3x14": s(p3x1, 4), "p3x0": s(p3x0, 3),
            "p3x04": s(p3x0, 4), "w_add": s(p3x0 + 0.8, 3),
            "l1": s(lam[0], 2), "l2": s(lam[1], 2), "l3": s(lam[2], 2), "i1": s(inv[0], 3), "i2": s(inv[1], 3), "i3": s(inv[2], 1),
            "vall": s(sum(inv), 2), "v12": s(inv[0] + inv[1], 2), "w_sum": s(sum(lam), 2), "w_sum12": s(lam[0] + lam[1], 2)}


def p11():
    # (a) 釣り合い型不完備ブロック計画
    t, k, b = 7, 3, 7
    rr = b * k // t
    lam = rr * (k - 1) // (t - 1)
    E = Fraction(lam * t, rr * k)
    # (b) 層別の総数推定
    N = [3000, 4000, 2000]
    n = [100, 100, 200]
    y = [6, 12, 40]
    tt = [N[i] * y[i] / n[i] for i in range(3)]
    tot = sum(tt)
    pool = sum(y) / sum(n) * sum(N)
    return {"r": rr, "lam": lam, "E": r(E, 3), "E4": s(E, 4), "E_s": s(E, 3),
            "N1": N[0], "N2": N[1], "N3": N[2], "n1": n[0], "n2": n[1], "n3": n[2], "y1": y[0], "y2": y[1], "y3": y[2],
            "t1": s(tt[0], 0), "t2": s(tt[1], 0), "t3": s(tt[2], 0), "tot": r(tot, 0), "ysum": sum(y), "nsum": sum(n),
            "Nsum": sum(N), "w_pool": s(pool, 0)}


def p12():
    # (a) ガウス型ナイーブベイズ
    x1, x2 = 136, 6.6
    zA1, zB1 = (x1 - 140) / 10, (x1 - 120) / 10
    zA2, zB2 = (x2 - 6.8) / 0.5, (x2 - 5.6) / 0.5
    h1 = (zB1 ** 2 - zA1 ** 2) / 2
    h2 = (zB2 ** 2 - zA2 ** 2) / 2
    assert abs(h1 + h2 - 3.12) < 1e-9
    e312 = 22.65
    odds = 0.3 / 0.7 * e312
    post = odds / (1 + odds)
    lr1 = math.exp(h1)
    vals = [post, e312 / (1 + e312), 0.3 / 0.7 * lr1 / (1 + 0.3 / 0.7 * lr1), 1 / (1 + odds)]
    distinct(vals + [0.3], 3)
    # (b) 群平均法の更新
    dRP, dRQ = 6.0, 9.0
    # (c) 数量化 II 類（説明変数 1 つ）
    a = [8, 22, 30]
    bb = [32, 18, 10]
    col = [a[j] + bb[j] for j in range(3)]
    n = sum(col)
    E = [[sum(a) * col[j] / n for j in range(3)], [sum(bb) * col[j] / n for j in range(3)]]
    X2 = sum((o - e) ** 2 / e for row_o, row_e in zip((a, bb), E) for o, e in zip(row_o, row_e))
    q = [a[j] / col[j] for j in range(3)]
    ys = [1] * sum(a) + [0] * sum(bb)
    xs_lin = sum(([j] * a[j] for j in range(3)), []) + sum(([j] * bb[j] for j in range(3)), [])
    my, mx = sum(ys) / n, sum(xs_lin) / n
    cxy = sum((u - mx) * (v - my) for u, v in zip(xs_lin, ys))
    cxx = sum((u - mx) ** 2 for u in xs_lin)
    cyy = sum((v - my) ** 2 for v in ys)
    lin = cxy ** 2 / (cxx * cyy)
    distinct([X2 / n, lin], 3)
    return {"x1": x1, "x2": s(x2, 1), "zA1": s(zA1, 1), "zB1": s(zB1, 1), "zA2": s(zA2, 1), "zB2": s(zB2, 1),
            "h1": s(h1, 2), "h2": s(h2, 2), "odds": s(odds, 3), "post": s(post, 3), "post4": s(post, 4),
            "w_noprior": s(vals[1], 3), "w_one": s(vals[2], 3), "w_inv": s(vals[3], 3),
            "dRP": s(dRP, 1), "dRQ": s(dRQ, 1), "ans": r((2 * dRP + 3 * dRQ) / 5, 2), "ans_s": s((2 * dRP + 3 * dRQ) / 5, 2),
            "w_simple": s((dRP + dRQ) / 2, 2),
            "n": n, "a1": a[0], "a2": a[1], "a3": a[2], "b1": bb[0], "b2": bb[1], "b3": bb[2], "c1": col[0], "c2": col[1],
            "c3": col[2], "q1": s(q[0], 2), "q2": s(q[1], 2), "q3": s(q[2], 2), "X2": s(X2, 1), "eta": s(X2 / n, 3),
            "w_lin": s(lin, 3)}


def p13():
    # (a) AR(2)
    p1, p2 = 1.0, -0.5
    r1 = p1 / (1 - p2)
    r2 = p1 * r1 + p2
    per = 2 * math.pi / math.acos(p1 / (2 * sqrt(-p2)))
    assert abs(per - 8) < 1e-9
    # (b) MNAR
    pl, ph, r1_, r5 = 0.6, 0.4, 0.9, 0.5
    m1, m5 = pl * r1_, ph * r5
    ans = (m1 * 1 + m5 * 5) / (m1 + m5)
    return {"p1": s(p1, 1), "p2": s(p2, 1), "p2s": "-0.5", "r1": s(r1, 3), "r2": s(r2, 3), "per": s(per, 0),
            "w_per": s(2 * math.pi / (math.pi / 3), 0), "w_r2": s(p1 * p1 + p2, 2),
            "pl": s(pl, 1), "ph": s(ph, 1), "r1x": s(r1_, 1), "r5": s(r5, 1), "m1": s(m1, 2), "m5": s(m5, 2),
            "msum": s(m1 + m5, 2), "ans": r(ans, 2), "ans4": s(ans, 4), "ans_s": s(ans, 2), "true": s(pl + 5 * ph, 1)}


def p14():
    # (a) 階層モデルの縮小
    mu, tau2, s2, ybar = 6, 4, 16, 10
    nA, nB = 8, 64
    vA, vB = s2 / nA, s2 / nB
    wA, wB = tau2 / (tau2 + vA), tau2 / (tau2 + vB)
    pa = wA * ybar + (1 - wA) * mu
    pb = wB * ybar + (1 - wB) * mu
    # (b) ジャックナイフ
    xs = [3.0, 5.0, 6.0, 10.0]
    n = len(xs)
    var = lambda v: sum((u - sum(v) / len(v)) ** 2 for u in v) / len(v)
    th = var(xs)
    loo = [var(xs[:i] + xs[i + 1:]) for i in range(n)]
    loom = sum(loo) / n
    bias = (n - 1) * (loom - th)
    corr = th - bias
    return {"mu": mu, "tau2": tau2, "s2": s2, "ybar": ybar, "nA": nA, "nB": nB, "vA": s(vA, 0), "vB": s(vB, 2),
            "wA": s(wA, 3), "wB": s(wB, 3), "pa": s(pa, 2), "pb": s(pb, 2), "w_mid": s((ybar + mu) / 2, 2),
            "xs": "、".join(s(v, 0) for v in xs), "xb": s(sum(xs) / n, 1), "th": s(th, 2),
            "loo": "、".join(s(v, 3) for v in loo), "loom": s(loom, 3), "bias": s(bias, 3), "corr": r(corr, 2), "corr_s": s(corr, 2)}


def values():
    return {f"s29p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 15)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
