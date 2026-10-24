"""模擬試験 第6回（content/exam_sets/set06.yaml）の数値を作る。題材：マーケティング・小売・EC・顧客調査。

関数電卓が要る値（分布表・対数・指数・逆双曲線正接の値）は問題文に示した丸めた値を使って計算し、
一般電卓で問題文どおりに計算した受験者がちょうど正解にたどり着くようにする。
"""
from decimal import ROUND_HALF_UP, Decimal
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


def distinct(*xs):
    assert len(set(xs)) == len(xs), xs


def p01():
    # 生存関数 S(t) = (θ/(θ+t))^α（ロマックス分布）
    al, th, t = 2.5, 10, 6
    St = (th / (th + t)) ** al
    h = al / (th + t)
    f = h * St
    c = dict(h=s(h, 3), f=s(f, 3), fF=s(f / (1 - St), 3), a1=s((al + 1) / (th + t), 3), h0=s(al / th, 3))
    distinct(*c.values())
    # (b) 平均残存期間 m(t) = (θ+t)/(α-1)
    mrl = (th + t) / (al - 1)
    # (c) カプラン・マイヤー（同時刻は解約を先に数える）
    data = [(2, 1), (3, 1), (3, 0), (5, 1), (5, 1), (6, 0), (7, 1), (9, 0), (10, 1), (12, 0)]
    S = 1.0
    steps = []
    for tt in sorted({x for x, e in data if e == 1}):
        n_at = sum(1 for x, _ in data if x >= tt)
        d = sum(1 for x, e in data if x == tt and e == 1)
        S *= (n_at - d) / n_at
        steps.append((tt, n_at, d, S))
    S8 = [st[3] for st in steps if st[0] <= 8][-1]
    return {
        "al": al, "th": th, "t": t, "tht": th + t, "ratio": s(th / (th + t), 3), "St": s(St, 4),
        **c, "f4": s(f, 4), "Ft": s(1 - St, 4),
        "al1": s(al - 1, 1), "mrl": r(mrl, 2), "mrl4": s(mrl, 4), "ET": s(th / (al - 1), 2),
        "mrl_cond": s(mrl + t, 2), "mrl_al": s((th + t) / al, 2),
        "S8": r(S8, 3), "S8_4": s(S8, 4), "S8_naive": s(1 - 5 / 10, 3), "S8_drop": s(3 / 8, 3),
        "n1": steps[0][1], "n2": steps[1][1], "n3": steps[2][1], "n4": steps[3][1],
        "S1": s(steps[0][3], 3), "S2": s(steps[1][3], 3), "S3": s(steps[2][3], 4), "S4": s(steps[3][3], 4),
        "S_model8": s((th / (th + 8)) ** al, 3),
        "S2_cfirst": s(0.9 * 7 / 8, 4),
    }


def p02():
    # X~Ga(2,θ), Y~Ga(3,θ) 独立、V=Y/(X+Y)~Beta(3,2)。P(V≥0.6)
    v = 0.6
    # Beta(3,2) の密度 12 v^2 (1-v)、分布関数 12(v^3/3 - v^4/4) = 4v^3 - 3v^4
    F = 4 * v ** 3 - 3 * v ** 4
    ans = 1 - F
    # Beta(2,3) の分布関数 6v^2 - 8v^3 + 3v^4
    Fsw = 6 * v ** 2 - 8 * v ** 3 + 3 * v ** 4
    c = dict(ok=s(ans, 3), comp=s(F, 3), sw=s(1 - Fsw, 3), unif=s(1 - v, 3), half="0.500")
    distinct(*c.values())
    return {"v": v, "v3": s(v ** 3, 3), "v4": s(v ** 4, 4), "F4": s(4 * v ** 3, 3), "F3": s(3 * v ** 4, 4),
            "F": s(F, 4), "ans4": s(ans, 4), **c}


def p03():
    n = 200
    p1, p2, p3 = 0.45, 0.35, 0.20
    vd = n * (p1 * (1 - p1) + p3 * (1 - p3) + 2 * p1 * p3)
    c = dict(ok=s(vd, 1), nocov=s(n * (p1 * (1 - p1) + p3 * (1 - p3)), 1),
             minus=s(n * (p1 * (1 - p1) + p3 * (1 - p3) - 2 * p1 * p3), 1),
             sd=s(sqrt(vd), 2), bin=s(n * (p1 - p3) * (1 - (p1 - p3)), 1))
    distinct(*c.values())
    # (b) NPS のワルド区間
    m, a1, a2, a3 = 400, 188, 140, 72
    q1, q3 = a1 / m, a3 / m
    d = q1 - q3
    vv = q1 + q3 - d ** 2
    se = sqrt(vv / m)
    lo = d - 1.96 * se
    se_ind = sqrt((q1 * (1 - q1) + q3 * (1 - q3)) / m)
    return {
        "n": n, "p1": p1, "p2": p2, "p3": p3, "q1p": s(p1 * (1 - p1), 4), "q3p": s(p3 * (1 - p3), 2),
        "c13": s(p1 * p3, 2), "c13n": s(n * p1 * p3, 0), "var1": s(n * p1 * (1 - p1), 1), "var3": s(n * p3 * (1 - p3), 1),
        "per": s(vd / n, 4), "Ed": s(p1 - p3, 2), "Ed2": s(p1 + p3, 2), "Ed_sq": s((p1 - p3) ** 2, 4),
        **c,
        "m": m, "a1": a1, "a2": a2, "a3": a3, "q1": s(q1, 2), "q3": s(q3, 2), "d": s(d, 2), "dsq": s(d ** 2, 4),
        "q13": s(q1 + q3, 2), "vv": s(vv, 4), "vvm": s(vv / m, 8), "se": s(se, 5), "hw": s(1.96 * se, 5),
        "lo": r(lo, 3), "lo4": s(lo, 4), "hi": s(d + 1.96 * se, 3),
        "se_ind": s(se_ind, 5), "lo_ind": s(d - 1.96 * se_ind, 3),
    }


def p04():
    n, sx = 50, 200
    p = n / sx
    se = sqrt(p ** 2 * (1 - p) / n)
    # (b) 尤度比検定 H0: p=0.2（ln の値は問題文の 5 桁を使う）
    p0 = 0.2
    ln1, ln2 = 0.22314, -0.06454   # ln(0.25/0.2), ln(0.75/0.8)
    half = n * ln1 + (sx - n) * ln2
    G2 = 2 * half
    wald = (p - p0) ** 2 / (p ** 2 * (1 - p) / n)
    score = (p - p0) ** 2 / (p0 ** 2 * (1 - p0) / n)
    c = dict(G=s(G2, 2), half=s(half, 2), wald=s(wald, 2), score=s(score, 2))
    distinct(*c.values())
    return {
        "n": n, "sx": sx, "fail": sx - n, "xbar": s(sx / n, 1), "p": s(p, 2), "q": s(1 - p, 2),
        "info": s(1 / (p ** 2 * (1 - p)), 4), "se": r(se, 4), "se6": s(se, 6), "varp": s(p ** 2 * (1 - p) / n, 7),
        "se_bin": s(sqrt(p * (1 - p) / n), 4), "se_noq": s(sqrt(p ** 2 / n), 4),
        "p0": p0, "ln1": ln1, "ln2": ln2, "t1": s(n * ln1, 3), "t2": s((sx - n) * ln2, 3),
        **c,
    }


def p05():
    n, mu, sd = 100, 5.0, 3.0
    ES, SDS = n * mu, sqrt(n) * sd
    k = 60 / SDS
    cheb = 1 - 1 / k ** 2
    phi2, phi02 = 0.9772, 0.5793
    clt = 2 * phi2 - 1
    # 誤り：合計の SD を n×σ=300 とした
    clt_bad = 2 * phi02 - 1
    # (b) ブートストラップ
    x = [2, 3, 4, 6, 10]
    m = sum(x) / len(x)
    ss = sum((v - m) ** 2 for v in x)
    vb = ss / len(x) / len(x)
    c = dict(ok=s(vb, 2), unb=s(ss / (len(x) - 1) / len(x), 2), sig=s(ss / len(x), 2),
             s2=s(ss / (len(x) - 1), 2), bsd=s(sqrt(vb), 2))
    distinct(*c.values())
    return {
        "n": n, "mu": s(mu, 1), "sd": s(sd, 1), "ES": s(ES, 0), "SDS": s(SDS, 0), "VS": s(SDS ** 2, 0),
        "lo": s(ES - 60, 0), "hi": s(ES + 60, 0), "k": s(k, 0), "cheb": s(cheb, 2), "cheb_lin": s(1 - 1 / k, 2),
        "phi2": phi2, "phi02": phi02, "clt": s(clt, 3), "clt1": s(phi2, 3), "out_c": s(1 / k ** 2, 2),
        "out_n": s(1 - clt, 3), "clt_bad": s(clt_bad, 3), "SDbad": s(n * sd, 0),
        "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "x5": x[4], "xm": s(m, 0), "ss": s(ss, 0),
        "nb": len(x), "sig2": s(ss / len(x), 0), "s2v": s(ss / (len(x) - 1), 0), **c,
    }


def p06():
    n, rr, rho0 = 40, 0.62, 0.4
    z_r, z_0 = 0.7250, 0.4236   # artanh の値（問題文に示す）
    Z = (z_r - z_0) * sqrt(n - 3)
    Zn = (z_r - z_0) * sqrt(n)
    tt = rr * sqrt(n - 2) / sqrt(1 - rr ** 2)
    c = dict(Z=s(Z, 2), Zn=s(Zn, 2), diff=s(z_r - z_0, 2), t=s(tt, 2))
    distinct(*c.values())
    # (b) 欠落変数バイアス
    b1, b2, vx, cxw, vw = 0.8, 2.5, 4.0, 1.2, 2.25
    dlt = cxw / vx
    short = b1 + b2 * dlt
    cc = dict(ok=s(short, 2), none=s(b1, 2), minus=s(b1 - b2 * dlt, 2), nodiv=s(b1 + b2 * cxw, 2),
              rev=s(b1 + b2 * cxw / vw, 2))
    distinct(*cc.values())
    return {"n": n, "r": rr, "rho0": rho0, "zr": s(z_r, 4), "z0": s(z_0, 4), "n3": n - 3, "sq": s(sqrt(n - 3), 4),
            "dz": s(z_r - z_0, 4), "n2": n - 2, **c,
            "b1": b1, "b2": b2, "vx": vx, "cxw": cxw, "vw": vw, "dlt": s(dlt, 2), "bias": s(b2 * dlt, 2),
            **{"o_" + k: v for k, v in cc.items()}}


L8 = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 2, 2, 2], [1, 2, 2, 1, 1, 2, 2], [1, 2, 2, 2, 2, 1, 1],
      [2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 2, 1, 2, 1], [2, 2, 1, 1, 2, 2, 1], [2, 2, 1, 2, 1, 1, 2]]
Y8 = [63, 58, 55, 57, 66, 71, 61, 64]


def p07():
    # (a) フリードマン検定
    ranks = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (1, 2, 3), (2, 1, 3), (1, 2, 3), (2, 3, 1), (1, 2, 3)]
    n, k = len(ranks), 3
    R = [sum(rw[j] for rw in ranks) for j in range(k)]
    sR2 = sum(x * x for x in R)
    Q = 12 / (n * k * (k + 1)) * sR2 - 3 * n * (k + 1)
    rbar = [x / n for x in R]
    small = sum((x - (k + 1) / 2) ** 2 for x in rbar)
    c = dict(Q=s(Q, 2), raw=s(12 / (n * k * (k + 1)) * sR2, 2), small=s(small, 2))
    distinct(*c.values())
    # (b) L8 直交表
    T = []
    for j in range(7):
        t1 = sum(y for y, row in zip(Y8, L8) if row[j] == 1)
        t2 = sum(y for y, row in zip(Y8, L8) if row[j] == 2)
        T.append((t1, t2, (t1 - t2) ** 2 / 8))
    ybar = sum(Y8) / 8
    ST = sum((y - ybar) ** 2 for y in Y8)
    assert abs(ST - sum(x[2] for x in T)) < 1e-9
    SA, SB, SC = T[0][2], T[1][2], T[3][2]
    Se = ST - SA - SB - SC
    Ve = Se / 4
    FA = SA / Ve
    return {
        **{f"r{i}{j}": ranks[i][j] for i in range(n) for j in range(k)},
        "RA": R[0], "RB": R[1], "RC": R[2], "RA2": R[0] ** 2, "RB2": R[1] ** 2, "RC2": R[2] ** 2, "sR2": sR2,
        "nn": n, "kk": k, "coef": s(12 / (n * k * (k + 1)), 3), "sub": 3 * n * (k + 1),
        "rbA": s(rbar[0], 3), "rbB": s(rbar[1], 3), "rbC": s(rbar[2], 3), **c,
        **{f"y{i + 1}": Y8[i] for i in range(8)},
        "TA1": T[0][0], "TA2": T[0][1], "TB1": T[1][0], "TB2": T[1][1], "TC1": T[3][0], "TC2": T[3][1],
        "SA": s(SA, 3), "SB": s(SB, 3), "SC": s(SC, 3), "ST": s(ST, 3), "Se": s(Se, 3), "Ve": s(Ve, 4),
        "FA": r(FA, 2), "FA4": s(FA, 4), "FA_e3": s(SA / (Se / 3), 2), "FA_e7": s(SA / (Se / 7), 2),
        "dA": T[0][1] - T[0][0], "dB": T[1][0] - T[1][1], "dC": T[3][0] - T[3][1],
    }


def p08():
    lam, mu = 24, 30
    rho = lam / mu
    Wq = rho / (mu - lam) * 60
    W = 1 / (mu - lam) * 60
    L = rho / (1 - rho)
    Lq = rho ** 2 / (1 - rho)
    c = dict(Wq=s(Wq, 1), W=s(W, 1), Lq=s(Lq, 1), L=s(L, 1), Lqmu=s(Lq / mu * 60, 1))
    distinct(*c.values())
    return {"lam": lam, "mu": mu, "svc": s(60 / mu, 0), "rho": s(rho, 1), "dif": mu - lam, **c}


def p09():
    b1, se1, b3, se3, cov = -2.40, 0.50, 1.10, 0.60, -0.24
    est = b1 + b3
    var = se1 ** 2 + se3 ** 2 + 2 * cov
    se = sqrt(var)
    t = est / se
    return {"b1": s(b1, 2), "se1": s(se1, 2), "b3": s(b3, 2), "se3": s(se3, 2), "cov": s(cov, 2),
            "est": s(est, 2), "v1": s(se1 ** 2, 2), "v3": s(se3 ** 2, 2), "cov2": s(2 * cov, 2), "var": s(var, 2),
            "se": s(se, 4), "t": r(t, 2), "t4": s(t, 4),
            "t_nocov": s(est / sqrt(se1 ** 2 + se3 ** 2), 2), "t_plus": s(est / sqrt(se1 ** 2 + se3 ** 2 - 2 * cov), 2),
            "t_b3": s(b3 / se3, 2), "corr": s(cov / (se1 * se3), 1)}


def p10():
    N = 20
    pis = [0.8, 0.5, 0.4, 0.2]
    ys = [40, 22, 18, 7]
    ht = sum(y / p for y, p in zip(ys, pis))
    c = dict(ok=s(ht, 1), eq=s(N * sum(ys) / len(ys), 1), raw=s(sum(ys), 1),
             mult=s(sum(y * p for y, p in zip(ys, pis)), 1), divn=s(ht / len(ys), 1))
    distinct(*c.values())
    # (b) てこ比
    n, xbar, Sxx, x0 = 20, 1.5, 12.0, 4.5
    h = 1 / n + (x0 - xbar) ** 2 / Sxx
    cc = dict(h=s(h, 2), nn=s((x0 - xbar) ** 2 / Sxx, 2), nosq=s(1 / n + (x0 - xbar) / Sxx, 2))
    distinct(*cc.values())
    return {"N": N, **{f"pi{i + 1}": pis[i] for i in range(4)}, **{f"y{i + 1}": ys[i] for i in range(4)},
            **{f"w{i + 1}": s(ys[i] / pis[i], 1) for i in range(4)}, **c,
            "n": n, "xbar": xbar, "Sxx": Sxx, "x0": x0, "dev": s(x0 - xbar, 1), "dev2": s((x0 - xbar) ** 2, 2),
            "invn": s(1 / n, 2), "crit": s(2 * 2 / n, 2), **{"h_" + k: v for k, v in cc.items()}}


def p11():
    x = 2
    e1, e2 = 0.4, -1.1
    E1, E2 = 1.4918, 0.3329   # exp(0.4), exp(-1.1)（問題文に示す）
    den = 1 + E1 + E2
    c = dict(ok=s(E1 / den, 3), bin=s(E1 / (1 + E1), 3), base=s(1 / den, 3), store=s(E2 / den, 3),
             nobase=s(E1 / (E1 + E2), 3))
    distinct(*c.values())
    return {"x": x, "e1": s(e1, 1), "e2": s(e2, 1), "E1": E1, "E2": E2, "den": s(den, 4), **c}


def p12():
    # (a) 共分散行列の主成分
    s11, s12, s22 = 9, 4, 3
    tr = s11 + s22
    det = s11 * s22 - s12 ** 2
    disc = sqrt((tr / 2) ** 2 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    rr = s12 / sqrt(s11 * s22)
    pct = lambda v: s(100 * v, 1) + "\\%"
    c = dict(ok=pct(l1 / tr), cor=pct((1 + rr) / 2), var1=pct(s11 / tr), r=pct(rr), second=pct(l2 / tr))
    distinct(*c.values())
    # (b) k-means
    pts = {"P1": (1, 1), "P2": (2, 1), "P3": (1, 3), "P4": (4, 4), "P5": (5, 6), "P6": (6, 5)}
    c1, c2 = pts["P1"], pts["P3"]
    d2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    cl2 = [k for k, v in pts.items() if d2(v, c2) < d2(v, c1)]
    assert cl2 == ["P3", "P4", "P5", "P6"], cl2
    for k, v in pts.items():
        assert d2(v, c1) != d2(v, c2)
    cen2 = (sum(pts[k][0] for k in cl2) / 4, sum(pts[k][1] for k in cl2) / 4)
    avg = lambda ks, extra=(): (sum([pts[k][0] for k in ks] + [e[0] for e in extra]) / (len(ks) + len(extra)),
                                sum([pts[k][1] for k in ks] + [e[1] for e in extra]) / (len(ks) + len(extra)))
    fmt = lambda c: f"({s(c[0], 1)},\\ {s(c[1], 1)})"
    kc = dict(ok=fmt(cen2), c1=fmt(avg(["P1", "P2"])), old=fmt(avg(cl2, (c2,))), p2=fmt(avg(cl2 + ["P2"])),
              nop3=fmt(avg(["P4", "P5", "P6"])))
    distinct(*kc.values())
    # (c) 単純ベイズ
    pr, m1, m0, k1, k0 = 0.2, 0.6, 0.2, 0.5, 0.3
    num = pr * m1 * k1
    den = num + (1 - pr) * m0 * k0
    cc = dict(ok=s(num / den, 3), noprior=s(m1 * k1 / (m1 * k1 + m0 * k0), 3), onlyM=s(pr * m1 / (pr * m1 + (1 - pr) * m0), 3),
              lik=s(m1 * k1, 3), num=s(num, 3))
    distinct(*cc.values())
    return {
        "s11": s11, "s12": s12, "s22": s22, "tr": tr, "det": det, "half": s(tr / 2, 0), "disc2": s((tr / 2) ** 2 - det, 0),
        "disc": s(disc, 0), "l1": s(l1, 0), "l2": s(l2, 0), "rr": s(rr, 4), **{"pc_" + k: v for k, v in c.items()},
        **{f"{k}x": v[0] for k, v in pts.items()}, **{f"{k}y": v[1] for k, v in pts.items()},
        "c2x": s(cen2[0], 1), "c2y": s(cen2[1], 1), "sx2": 16, "sy2": 18, **{"km_" + k: v for k, v in kc.items()},
        "pr": pr, "m1": m1, "m0": m0, "k1": k1, "k0": k0, "nb_num": s(num, 3), "nb_oth": s((1 - pr) * m0 * k0, 3),
        "nb_den": s(den, 3), **{"nb_" + k: v for k, v in cc.items()},
    }


def p13():
    # (a) シンプソンのパラドックス
    A = {"sp": (36, 900), "pc": (30, 250)}
    B = {"sp": (6, 200), "pc": (110, 1000)}
    rate = lambda a: 100 * a[0] / a[1]
    tA = (sum(v[0] for v in A.values()), sum(v[1] for v in A.values()))
    tB = (sum(v[0] for v in B.values()), sum(v[1] for v in B.values()))
    assert rate(A["sp"]) > rate(B["sp"]) and rate(A["pc"]) > rate(B["pc"]) and rate(tA) < rate(tB)
    # (b) フロントドア調整
    px = 0.4
    pm = {1: 0.7, 0: 0.2}
    py = {(1, 1): 0.30, (1, 0): 0.10, (0, 1): 0.20, (0, 0): 0.05}
    inner = {m: py[(1, m)] * px + py[(0, m)] * (1 - px) for m in (0, 1)}
    do = {x: pm[x] * inner[1] + (1 - pm[x]) * inner[0] for x in (0, 1)}
    ace = do[1] - do[0]
    naive = {x: pm[x] * py[(x, 1)] + (1 - pm[x]) * py[(x, 0)] for x in (0, 1)}
    return {
        "Asp": A["sp"][0], "Asn": A["sp"][1], "Apc": A["pc"][0], "Apn": A["pc"][1],
        "Bsp": B["sp"][0], "Bsn": B["sp"][1], "Bpc": B["pc"][0], "Bpn": B["pc"][1],
        "rAs": s(rate(A["sp"]), 1), "rAp": s(rate(A["pc"]), 1), "rBs": s(rate(B["sp"]), 1), "rBp": s(rate(B["pc"]), 1),
        "tA": tA[0], "tAn": tA[1], "tB": tB[0], "tBn": tB[1], "rA": s(rate(tA), 2), "rB": s(rate(tB), 2),
        "avA": s((rate(A["sp"]) + rate(A["pc"])) / 2, 1), "avB": s((rate(B["sp"]) + rate(B["pc"])) / 2, 1),
        "wA": s(100 * A["sp"][1] / tA[1], 1), "wB": s(100 * B["sp"][1] / tB[1], 1),
        "px": px, "qx": s(1 - px, 1), "pm1": pm[1], "pm0": pm[0], "qm1": s(1 - pm[1], 1), "qm0": s(1 - pm[0], 1),
        "y11": s(py[(1, 1)], 2), "y10": s(py[(1, 0)], 2), "y01": s(py[(0, 1)], 2), "y00": s(py[(0, 0)], 2),
        "in1": s(inner[1], 2), "in0": s(inner[0], 2), "do1": s(do[1], 3), "do0": s(do[0], 3),
        "ace": r(ace, 3), "ace4": s(ace, 4), "nv1": s(naive[1], 2), "nv0": s(naive[0], 2), "nace": s(naive[1] - naive[0], 2),
    }


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s06p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
