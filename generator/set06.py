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
    # (c) 比例ハザード：S1(t) = S0(t)^{exp(β)}
    S0, hr = 0.55, 0.500
    S1 = S0 ** hr
    return {
        "al": al, "th": th, "t": t, "tht": th + t, "ratio": s(th / (th + t), 3), "St": s(St, 4),
        **c, "f4": s(f, 4), "Ft": s(1 - St, 4),
        "al1": s(al - 1, 1), "mrl": r(mrl, 2), "mrl4": s(mrl, 4), "ET": s(th / (al - 1), 2),
        "mrl_cond": s(mrl + t, 2), "mrl_al": s((th + t) / al, 2),
        "S0": S0, "hr": s(hr, 3), "S1": r(S1, 3), "S1_4": s(S1, 4), "S1_half": s(1 - hr * (1 - S0), 3),
        "S1_sq": s(S0 ** 2, 4), "S1_mul": s(S0 * hr, 3),
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
    # NPS のワルド区間
    m, a1, a2, a3 = 400, 188, 140, 72
    q1, q3 = a1 / m, a3 / m
    d = q1 - q3
    vv = q1 + q3 - d ** 2
    se = sqrt(vv / m)
    lo = d - 1.96 * se
    se_ind = sqrt((q1 * (1 - q1) + q3 * (1 - q3)) / m)
    return {
        "m": m, "a1": a1, "a2": a2, "a3": a3, "q1": s(q1, 2), "q3": s(q3, 2), "d": s(d, 2), "dsq": s(d ** 2, 4),
        "q13": s(q1 + q3, 2), "vv": s(vv, 4), "vvm": s(vv / m, 8), "se": s(se, 5), "hw": s(1.96 * se, 5),
        "lo": r(lo, 3), "lo4": s(lo, 4), "hi": s(d + 1.96 * se, 3),
        "se_ind": s(se_ind, 5), "lo_ind": s(d - 1.96 * se_ind, 3),
        "c13": s(q1 * q3, 4), "v1": s(q1 * (1 - q1), 4), "v3": s(q3 * (1 - q3), 4),
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
    score = (n / p0 - (sx - n) / (1 - p0)) ** 2 / (n / (p0 ** 2 * (1 - p0)))   # スコアの2乗 / p0 での情報量
    c = dict(G=s(G2, 2), half=s(half, 2), wald=s(wald, 2), score=s(score, 2))
    distinct(*c.values())
    crlb = (1 - p) / (p ** 2 * n)
    return {
        "crlb": r(crlb, 3), "crlb4": s(crlb, 4), "p2": s(p ** 2, 4), "p2n": s(p ** 2 * n, 4), "dt": s(1 / p ** 4, 0),
        "cr_noinv": s(p ** 2 * (1 - p) / n, 4), "cr_sd": s(sqrt(crlb), 3), "cr_noq": s(1 / (p ** 2 * n), 3), "cr_pq": s((1 - p) / (p * n), 3),
        "n": n, "sx": sx, "fail": sx - n, "xbar": s(sx / n, 1), "p": s(p, 2), "q": s(1 - p, 2),
        "info": s(1 / (p ** 2 * (1 - p)), 4), "se": r(se, 4), "se6": s(se, 6), "varp": s(p ** 2 * (1 - p) / n, 7),
        "se_bin": s(sqrt(p * (1 - p) / n), 4), "se_noq": s(sqrt(p ** 2 / n), 4),
        "half3": s(half, 3), "G3": s(G2, 3), "p0": p0, "U0": s(n / p0 - (sx - n) / (1 - p0), 1), "I0": s(n / (p0 ** 2 * (1 - p0)), 1), "ln1": ln1, "ln2": ln2, "t1": s(n * ln1, 3), "t2": s((sx - n) * ln2, 3),
        **c,
    }


def p05():
    # (a) コーシー分布の標本中央値の漸近標準誤差 πγ/(2√n)
    n, gam, pi = 100, 2, 3.1416
    se_med = pi * gam / (2 * sqrt(n))
    c = dict(ok=s(se_med, 3), var=s(se_med ** 2, 3), half=s(se_med / 2, 3), dbl=s(2 * se_med, 3), mean=s(gam / sqrt(n), 3))
    distinct(*c.values())
    # (b) ブートストラップ
    x = [2, 3, 4, 6, 10]
    m = sum(x) / len(x)
    ss = sum((v - m) ** 2 for v in x)
    vb = ss / len(x) / len(x)
    cb = dict(ok=s(vb, 2), unb=s(ss / (len(x) - 1) / len(x), 2), sig=s(ss / len(x), 2),
              s2=s(ss / (len(x) - 1), 2), bsd=s(sqrt(vb), 2))
    distinct(*cb.values())
    return {
        "n": n, "gam": gam, "pi": pi, "fm": s(1 / (pi * gam), 4), "pig": s(pi * gam, 4), "sqn": s(sqrt(n), 0), **{"x_" + k: v for k, v in c.items()},
        "x1": x[0], "x2": x[1], "x3": x[2], "x4": x[3], "x5": x[4], "xm": s(m, 0), "ss": s(ss, 0),
        "nb": len(x), "sig2": s(ss / len(x), 0), "s2v": s(ss / (len(x) - 1), 0), **cb,
    }


def p06():
    # (a) 無相関検定が有意になる |r| の境界
    n = 40
    t2, t1, z = 2.024, 1.686, 1.960
    rc = lambda q, df: q / sqrt(df + q * q)
    c = dict(ok=s(rc(t2, n - 2), 3), one=s(rc(t1, n - 2), 3), z=s(rc(z, n - 2), 3), nosq=s(t2 / sqrt(n - 2), 3),
             r2=s(rc(t2, n - 2) ** 2, 3))
    distinct(*c.values())
    # (b) 2次の回帰で粗利 - 広告費を最大にする広告費
    b0, b1, b2, mg = 200, 6.0, 0.12, 0.4
    xs = (mg * b1 - 1) / (2 * mg * b2)
    return {"n": n, "r": 0.62, "t2": t2, "t1": t1, "df": n - 2, "t2sq": s(t2 ** 2, 4), "den": s(n - 2 + t2 ** 2, 4),
            "sden": s(sqrt(n - 2 + t2 ** 2), 4), **{"r_" + k: v for k, v in c.items()},
            "b0": b0, "b1": s(b1, 1), "b2": s(b2, 2), "mg": mg, "mb1": s(mg * b1, 1), "mb2": s(2 * mg * b2, 3),
            "num": s(mg * b1 - 1, 1), "xs": r(xs, 2), "xs4": s(xs, 4), "xmax": s(b1 / (2 * b2), 1),
            "xnomg": s((b1 - 1) / (2 * b2), 2), "xno2": s((mg * b1 - 1) / (mg * b2), 2),
            "yxs": s(b0 + b1 * xs - b2 * xs ** 2, 2)}


def p07():
    # (a) ヨンクヒール・タプストラ検定（価格が高いほど販売数が少ないという順序対立仮説）
    g = {"L": [12, 15, 18], "M": [10, 13, 14], "H": [8, 9, 11]}
    U = lambda a, b: sum(1 for x in g[a] for y in g[b] if x > y)
    uLM, uLH, uMH = U("L", "M"), U("L", "H"), U("M", "H")
    J = uLM + uLH + uMH
    ns = [len(v) for v in g.values()]
    N = sum(ns)
    EJ = (N ** 2 - sum(k * k for k in ns)) / 4
    VJ = (N ** 2 * (2 * N + 3) - sum(k * k * (2 * k + 3) for k in ns)) / 72
    Z = (J - EJ) / sqrt(VJ)
    c = dict(ok=s(Z, 2), rev=s((sum(a * b for a, b in [(3, 3)] * 3) - J - EJ) / sqrt(VJ), 2), adj=s((uLM + uMH - EJ) / sqrt(VJ), 2),
             var=s((J - EJ) / VJ, 2), sq=s((J - EJ) / sqrt(EJ), 2))
    distinct(*c.values())
    # (b) 2^{4-1} 一部実施要因計画 D=ABC
    runs = [(a, b, cc) for cc in (-1, 1) for b in (-1, 1) for a in (-1, 1)]
    y = [52, 61, 58, 55, 63, 57, 54, 68]
    D = [a * b * cc for a, b, cc in runs]
    plus = sum(v for v, d in zip(y, D) if d > 0)
    minus = sum(v for v, d in zip(y, D) if d < 0)
    eff = (plus - minus) / 4
    ce = dict(ok=s(eff, 1), half=s(eff / 2, 1), tot=s(plus - minus, 1))
    distinct(*ce.values())
    return {
        **{f"{k}{i + 1}": v[i] for k, v in g.items() for i in range(3)}, "uLM": uLM, "uLH": uLH, "uMH": uMH, "J": J,
        "N": N, "N2": N ** 2, "sn2": sum(k * k for k in ns), "EJ": s(EJ, 1), "VJ": s(VJ, 2), "SJ": s(sqrt(VJ), 1),
        "num1": N ** 2 * (2 * N + 3), "num2": sum(k * k * (2 * k + 3) for k in ns), "Jrev": 27 - J,
        **{"j_" + k: v for k, v in c.items()},
        **{f"y{i + 1}": y[i] for i in range(8)}, **{f"A{i + 1}": "+" if runs[i][0] > 0 else "−" for i in range(8)},
        **{f"B{i + 1}": "+" if runs[i][1] > 0 else "−" for i in range(8)}, **{f"C{i + 1}": "+" if runs[i][2] > 0 else "−" for i in range(8)},
        **{f"D{i + 1}": "+" if D[i] > 0 else "−" for i in range(8)},
        "Dp": plus, "Dm": minus, "Ddiff": plus - minus, **{"f_" + k: v for k, v in ce.items()},
    }


def p08():
    la, lw = 6, 18      # アプリ・Web（件/時）
    tmin = 5
    lt = (la + lw) * tmin / 60
    ex = {0.5: 0.6065, 1.5: 0.2231, 2.0: 0.1353}
    pa = la / (la + lw)
    c = dict(ok=s(pa * ex[lt], 3), app=s(pa * ex[la * tmin / 60], 3), web=s(pa * ex[lw * tmin / 60], 3),
             noroute=s(ex[lt], 3), cdf=s(pa * (1 - ex[lt]), 3))
    distinct(*c.values())
    return {"la": la, "lw": lw, "tmin": tmin, "lsum": la + lw, "lt": s(lt, 1), "pa": s(pa, 2),
            "e05": ex[0.5], "e15": ex[1.5], "e20": ex[2.0], **c}


def p09():
    # (a) 両対数モデルの予測比
    b1, b2 = -1.8, 0.6
    pw, e6 = 1.2088, 1.8221   # 0.9^{-1.8}, e^{0.6}
    ratio = pw * e6
    # (b) HC0 標準誤差
    d = [-5, -3, -1, 1, 3, 5]
    e = [2.0, -1.6, -0.4, -0.4, -1.6, 2.0]
    assert abs(sum(e)) < 1e-12 and abs(sum(a * b for a, b in zip(d, e))) < 1e-12
    Sxx = sum(a * a for a in d)
    de2 = [a * a * b * b for a, b in zip(d, e)]
    hc0 = sqrt(sum(de2)) / Sxx
    se2 = sum(b * b for b in e) / (len(d) - 2)
    us = sqrt(se2 / Sxx)
    h6 = 1 / len(d) + d[5] ** 2 / Sxx
    dfb = -d[5] * e[5] / (Sxx * (1 - h6))
    cd = dict(ok=s(dfb, 2), noh=s(-d[5] * e[5] / Sxx, 2), sign=s(-dfb, 2), h=s(-d[5] * e[5] / (Sxx * h6), 2))
    distinct(*cd.values())
    return {"h6": s(h6, 4), "h6n": s(1 - h6, 4), "dfb": r(dfb, 2), "dfb4": s(dfb, 4), **{"db_" + k: v for k, v in cd.items()},
            "de6": s(d[5] * e[5], 1), "Sh": s(Sxx * (1 - h6), 4),
            "b1": b1, "b2": b2, "pw": pw, "ex6": e6, "ratio": r(ratio, 2), "ratio4": s(ratio, 4),
            "add": s(1 + 0.18 + 0.6, 2), "addm": s(pw + e6 - 1, 2), "lin": s(1.18 * 1.6, 2),
            **{f"d{i + 1}": d[i] for i in range(6)}, **{f"e{i + 1}": s(e[i], 1) for i in range(6)},
            **{f"q{i + 1}": s(de2[i], 2) for i in range(6)}, "Sxx": Sxx, "Sxx2": Sxx ** 2,
            "sq": s(sum(de2), 2), "vhc": s(sum(de2) / Sxx ** 2, 6), "hc0": r(hc0, 3), "hc04": s(hc0, 4),
            "se2": s(sum(b * b for b in e), 2), "s2": s(se2, 2), "us": s(us, 3), "us4": s(us, 4),
            "hc1": s(hc0 * sqrt(len(d) / (len(d) - 2)), 3), "nosq": s(sqrt(sum(de2) / Sxx), 3)}


def p10():
    W = [0.30, 0.50, 0.20]
    nh = [60, 200, 140]
    yh = [3.2, 3.6, 4.1]
    n = sum(nh)
    ps = sum(w * y for w, y in zip(W, yh))
    un = sum(a * y for a, y in zip(nh, yh)) / n
    wt = [w / (a / n) for w, a in zip(W, nh)]
    inv = [(a / n) / w for w, a in zip(W, nh)]
    winv = sum(a * y * v for a, y, v in zip(nh, yh, inv)) / sum(a * v for a, v in zip(nh, inv))
    c = dict(ok=s(ps, 3), un=s(un, 3), simple=s(sum(yh) / 3, 3), inv=s(winv, 3),
             wavg=s(sum(w * y for w, y in zip(wt, yh)) / 3, 3))
    distinct(*c.values())
    return {**{f"W{i + 1}": s(100 * W[i], 0) for i in range(3)}, **{f"Wd{i + 1}": W[i] for i in range(3)},
            **{f"n{i + 1}": nh[i] for i in range(3)}, **{f"y{i + 1}": yh[i] for i in range(3)}, "n": n,
            **{f"sh{i + 1}": s(100 * nh[i] / n, 0) for i in range(3)}, **{f"w{i + 1}": s(wt[i], 3) for i in range(3)},
            **{f"t{i + 1}": s(W[i] * yh[i], 2) for i in range(3)}, **c}


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
    # (a) 特異値と低ランク近似
    sv = [6, 3, 2, 1]
    tot = sum(v * v for v in sv)
    pct = lambda v: s(100 * v, 1) + "\\%"
    c = dict(ok=pct((sv[0] ** 2 + sv[1] ** 2) / tot), nosq=pct((sv[0] + sv[1]) / sum(sv)), one=pct(sv[0] ** 2 / tot),
             res=pct((sv[2] ** 2 + sv[3] ** 2) / tot), sq=pct(((sv[0] ** 2 + sv[1] ** 2) / tot) ** 2))
    distinct(*c.values())
    # (b) 混合正規モデルの負担率
    w1, m1, m2, sd, x0 = 0.6, 2.0, 5.0, 1.0, 3.2
    ph1, ph2 = 0.1942, 0.0790   # φ(1.2), φ(1.8)
    num1, num2 = w1 * ph1, (1 - w1) * ph2
    cc = dict(ok=s(num1 / (num1 + num2), 3), now=s(ph1 / (ph1 + ph2), 3), hard="1.000", prior=s(w1, 3), other=s(num2 / (num1 + num2), 3))
    distinct(*cc.values())
    # (c) ジニ不純度
    N, P, nl, pl = 100, 30, 40, 20
    nr, pr_ = N - nl, P - pl
    g = lambda p: 1 - p * p - (1 - p) ** 2
    gp, gl, gr = g(P / N), g(pl / nl), g(pr_ / nr)
    after = nl / N * gl + nr / N * gr
    mis_p = min(P, N - P) / N
    mis_a = (min(pl, nl - pl) + min(pr_, nr - pr_)) / N
    cg = dict(ok=s(gp - after, 3), mis=s(mis_p - mis_a, 3), unw=s(gp - (gl + gr) / 2, 3), after=s(after, 3), par=s(gp, 3))
    distinct(*cg.values())
    return {
        **{f"sv{i + 1}": sv[i] for i in range(4)}, "tot": tot, "top2": sv[0] ** 2 + sv[1] ** 2, **{"sv_" + k: v for k, v in c.items()},
        "w1": w1, "w2": s(1 - w1, 1), "m1": m1, "m2": m2, "x0": x0, "z1": s((x0 - m1) / sd, 1), "z2": s((m2 - x0) / sd, 1),
        "ph1": ph1, "ph2": ph2, "num1": s(num1, 5), "num2": s(num2, 4), "den": s(num1 + num2, 5), **{"h_" + k: v for k, v in cc.items()},
        "N": N, "P": P, "nl": nl, "pl": pl, "nr": nr, "pr": pr_, "gp": s(gp, 2), "gl": s(gl, 2), "gr": s(gr, 4),
        "glw": s(nl / N * gl, 2), "grw": s(nr / N * gr, 4), "after4": s(after, 4), "dec4": s(gp - after, 4),
        "pfr": s(pl / nl, 1), "prr": s(pr_ / nr, 4), "mis_p": s(mis_p, 2), "mis_a": s(mis_a, 2),
        **{"g_" + k: v for k, v in cg.items()},
    }


def p13():
    # (a) イェーツ補正つきカイ二乗
    a, b, c_, d = 30, 90, 20, 160
    n = a + b + c_ + d
    r1, r2, k1, k2 = a + b, c_ + d, a + c_, b + d
    adbc = abs(a * d - b * c_)
    den = r1 * r2 * k1 * k2
    chiY = n * (adbc - n / 2) ** 2 / den
    chi = n * adbc ** 2 / den
    cy = dict(ok=s(chiY, 2), no=s(chi, 2), full=s(n * (adbc - n) ** 2 / den, 2), z=s(sqrt(chiY), 2), phi=s(sqrt(chi / n), 2))
    distinct(*cy.values())
    # (b) 幾何分布の無記憶性：5通連続で購入しなかった会員があと3通以内に購入する確率
    pp, k0, k1 = 0.2, 5, 3
    ok = 1 - (1 - pp) ** k1
    cf = dict(ok=s(ok, 3), unc=s(1 - (1 - pp) ** (k0 + k1), 3), joint=s((1 - pp) ** k0 - (1 - pp) ** (k0 + k1), 3),
              add=s(pp * k1, 3), no=s((1 - pp) ** k1, 3))
    distinct(*cf.values())
    # (c) 合流点で条件づけたときの偏相関：X,Y 独立 N(0,1)、C = X + Y + e、e~N(0,1)
    vx, vy, ve = 1, 1, 1
    vc = vx + vy + ve
    cxy_c = 0 - vx * vy / vc
    vx_c = vx - vx ** 2 / vc
    pc = cxy_c / vx_c
    ck = dict(ok=s(pc, 2), cov=s(cxy_c, 2), zero=s(0, 2), pos=s(-pc, 2), noe=s(-1 / 1, 2))
    distinct(*ck.values())
    return {"a": a, "b": b, "c": c_, "d": d, "n": n, "r1": r1, "r2": r2, "k1": k1, "k2": k2, "ad": a * d, "bc": b * c_,
            "adbc": adbc, "half": n // 2, "adj": s(adbc - n / 2, 0), "den": den, "adj2": s((adbc - n / 2) ** 2, 0),
            **{"y_" + k: v for k, v in cy.items()},
            "pp": pp, "k0": k0, "k0p": k0 + 1, "k1": k1, "k01": k0 + k1, "q": s(1 - pp, 1), "q3": s((1 - pp) ** k1, 3), "q5": s((1 - pp) ** k0, 5),
            "q8": s((1 - pp) ** (k0 + k1), 8), **{"f_" + k: v for k, v in cf.items()},
            "vc": vc, "cxyc": s(cxy_c, 4), "vxc": s(vx_c, 4), **{"k_" + k: v for k, v in ck.items()}}


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s06p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
