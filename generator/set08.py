"""模擬試験 第8回（set08）の数値。values() が {大問id: {名前: 値}} を返す。

題材：Web・IT 運用・通信・ソフトウェア品質。
問題文に示した関数値（Φ・χ²・e^x・べき乗・対数など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。負の数は −0 を避ける。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return str(v)


def distinct(*xs):
    assert len(set(xs)) == len(xs), xs


def p01():
    # (a) 直並列構成：L → (A1 ∥ A2) → D
    pl, pa, pd = 0.99, 0.95, 0.98
    qa2 = (1 - pa) ** 2
    up = pl * (1 - qa2) * pd
    down = 1 - up
    cond = (1 - pl) / down
    up_ser = pl * pa ** 2 * pd
    naive = (1 - pl) / ((1 - pl) + qa2 + (1 - pd))
    opts = [s(1 - pl, 3), s(down, 3), s((1 - pl) / (1 - up_ser), 3), s(naive, 3), s(cond, 3)]
    distinct(*opts)
    # (b) ハザード h(t)=c t
    c, t0, dt = 0.2, 2, 1
    H = lambda t: c * t ** 2 / 2
    expo = H(t0 + dt) - H(t0)
    e_val = {0.5: 0.6065, 0.1: 0.9048, 0.4: 0.6703, 0.9: 0.4066}
    assert abs(expo - 0.5) < 1e-12
    ans = e_val[0.5]
    return {
        "pl": pl, "pa": pa, "pd": pd, "qa": s(1 - pa, 2), "qa2": s(qa2, 4), "par": s(1 - qa2, 4),
        "up": s(up, 7), "down": s(down, 7), "ql": s(1 - pl, 2), "qd": s(1 - pd, 2),
        "a_ok": opts[4], "a_ql": opts[0], "a_down": opts[1], "a_ser": opts[2], "a_naive": opts[3],
        "upser": s(up_ser, 4), "naive_den": s((1 - pl) + qa2 + (1 - pd), 4), "cond4": s(cond, 4),
        "c": c, "t0": t0, "t1": t0 + dt, "dt": dt, "H0": s(H(t0), 1), "H1": s(H(t0 + dt), 1),
        "ch": s(c / 2, 1), "expo": s(expo, 1),
        "e05": s(e_val[0.5], 4), "e01": s(e_val[0.1], 4), "e04": s(e_val[0.4], 4), "e09": s(e_val[0.9], 4),
        "surv": r(ans, 3), "surv_s": s(ans, 3), "mem": s(e_val[0.1], 3), "hz": s(c * t0, 1),
        "uncond": s(e_val[0.9], 3), "w_cdf": s(1 - ans, 3),
    }


def p02():
    l1, l2 = 0.5, 0.25
    k = l1 * l2 / (l2 - l1)       # 負になる → 符号を入れ替えて書く
    coef = -k                     # f(t) = coef (e^{-l2 t} - e^{-l1 t})
    mu = 1 / l1 + 1 / l2
    var = 1 / l1 ** 2 + 1 / l2 ** 2
    eps, alpha = 0.5, 0.05
    z, z1 = 1.96, 1.645
    cheb = math.ceil(var / (alpha * eps ** 2) - 1e-9)
    clt = math.ceil(z ** 2 * var / eps ** 2)
    clt1 = math.ceil(z1 ** 2 * var / eps ** 2)
    var_sd = (1 / l1 + 1 / l2) ** 2
    cheb_sd = math.ceil(var_sd / (alpha * eps ** 2) - 1e-9)
    clt_sd = math.ceil(z ** 2 * var_sd / eps ** 2)
    cheb_e = math.ceil(var / (alpha * eps) - 1e-9)
    clt_e = math.ceil(z ** 2 * var / eps)
    pair = lambda a, b: f"チェビシェフ：$n={a}$、中心極限定理：$n={b}$"
    opts = [pair(cheb, clt1), pair(cheb_sd, clt_sd), pair(cheb, clt), pair(cheb_e, clt_e), pair(clt, cheb)]
    distinct(*opts)
    return {
        "l1": l1, "l2": l2, "m1": s(1 / l1, 0), "m2": s(1 / l2, 0), "lp": s(l1 * l2, 3), "ld": s(l2 - l1, 2),
        "coef": s(coef, 1), "ls": s(l1 + l2, 2),
        "mu": s(mu, 0), "v1": s(1 / l1 ** 2, 0), "v2": s(1 / l2 ** 2, 0), "var": s(var, 0),
        "eps": eps, "eps2": s(eps ** 2, 2), "alpha": alpha, "z": z, "z1": z1, "z2": s(z ** 2, 4),
        "cheb_raw": s(var / (alpha * eps ** 2), 0), "clt_raw": s(z ** 2 * var / eps ** 2, 2),
        "cheb": cheb, "clt": clt, "cheb_sd": cheb_sd, "clt_sd": clt_sd, "varsd": s(var_sd, 0), "clt1": clt1,
        "o0": opts[0], "o1": opts[1], "o2": opts[2], "o3": opts[3], "o4": opts[4],
    }


def p03():
    s1, s2, rho = 4.0, 3.0, 0.5
    cov = rho * s1 * s2
    den = s1 ** 2 + s2 ** 2 - 2 * cov
    w = (s2 ** 2 - cov) / den
    vmin = (s1 ** 2 * s2 ** 2 - cov ** 2) / den
    w0 = s2 ** 2 / (s1 ** 2 + s2 ** 2)
    v0 = s1 ** 2 * s2 ** 2 / (s1 ** 2 + s2 ** 2)
    # (b) 条件付き分布
    x1, thr = 6.0, 4.0
    cm = rho * s2 / s1 * x1
    cv = s2 ** 2 * (1 - rho ** 2)
    csd = sqrt(cv)
    phi = {0.26: 0.6026, 0.58: 0.7190, 0.67: 0.7486, 1.33: 0.9082}
    z_ok = r((thr - cm) / csd, 2)
    z_unc = r(thr / s2, 2)
    z_sd = r((thr - cm) / s2, 2)
    z_var = r((thr - cm) / cv, 2)
    for zz in (z_ok, z_unc, z_sd, z_var):
        assert zz in phi, zz
    cm_rev = rho * s1 / s2 * x1
    assert abs(cm_rev - thr) < 1e-12  # 傾きを逆にすると平均がちょうど閾値 → 0.5
    opts = [s(1 - phi[z_unc], 3), s(1 - phi[z_ok], 3), s(1 - phi[z_sd], 3), s(0.5, 3), s(1 - phi[z_var], 3)]
    distinct(*opts)
    return {
        "s1": s(s1, 0), "s2": s(s2, 0), "rho": rho, "v1": s(s1 ** 2, 0), "v2": s(s2 ** 2, 0), "cov": s(cov, 0),
        "den": s(den, 0), "num_w": s(s2 ** 2 - cov, 0), "w": s(w, 4), "w_frac": f"\\frac{{{s(s2 ** 2 - cov, 0)}}}{{{s(den, 0)}}}",
        "vnum": s(s1 ** 2 * s2 ** 2 - cov ** 2, 0), "vmin": r(vmin, 2), "vmin4": s(vmin, 4), "vmin_s": s(vmin, 2),
        "w0": s(w0, 2), "v0": s(v0, 2), "sdmin": s(sqrt(vmin), 3), "v_half": s((s1 ** 2 + s2 ** 2 + 2 * cov) / 4, 2),
        "p1v1": s(s1 ** 2 * s2 ** 2, 0), "cov2": s(cov ** 2, 0),
        "x1": s(x1, 0), "thr": s(thr, 0), "slope": s(rho * s2 / s1, 3), "cm": s(cm, 2), "cv": s(cv, 2),
        "csd": s(csd, 4), "zraw": s((thr - cm) / csd, 4), "zok": s(z_ok, 2), "dif": s(thr - cm, 2),
        "f026": s(phi[0.26], 4), "f058": s(phi[0.58], 4), "f067": s(phi[0.67], 4), "f133": s(phi[1.33], 4),
        "rho2": s(1 - rho ** 2, 2), "cm_rev": s(cm_rev, 0), "slope_rev": s(rho * s1 / s2, 4),
        "o_unc": opts[0], "o_ok": opts[1], "o_sd": opts[2], "o_rev": opts[3], "o_var": opts[4],
        "z_unc": s(z_unc, 2), "z_sd": s(z_sd, 2), "z_var": s(z_var, 2),
    }


def p04():
    lam, mu = 12, 15
    rho = lam / mu
    L = rho / (1 - rho)
    W = 1 / (mu - lam)
    Wq = rho / (mu - lam)
    Lq = rho ** 2 / (1 - rho)
    ms = lambda x: s(1000 * x, 0)
    opts = [ms(1 / mu), ms(1 / lam), ms(Lq / mu), ms(Wq), ms(W)]
    distinct(*opts)
    return {"lam": lam, "mu": mu, "rho": s(rho, 1), "L": s(L, 0), "W": s(W, 4), "Wq": s(Wq, 4), "Lq": s(Lq, 1),
            "dmu": mu - lam, "o_s": opts[0], "o_a": opts[1], "o_lqmu": opts[2], "o_ok": opts[3], "o_w": opts[4],
            "Wfrac": f"\\frac{{1}}{{{mu - lam}}}", "Wq_ms": opts[3], "W_ms": opts[4], "S_ms": opts[0]}


def p05():
    # (a) 0 件のときの片側上側信頼限界
    alpha, p0 = 0.05, 0.002
    ln_a, ln_q = -2.9957, -0.0020020
    ln_a2 = -3.6889
    nraw = ln_a / ln_q
    n = math.ceil(nraw)
    n_two = math.ceil(ln_a2 / ln_q)
    n_approx = math.ceil(-ln_a / p0)
    # (b) 新旧版の障害件数
    t_old, y_old, t_new, y_new = 400, 30, 200, 6
    tot = y_old + y_new
    pn = t_new / (t_old + t_new)
    e = tot * pn
    v = tot * pn * (1 - pn)
    z = (y_new - e) / sqrt(v)
    z_half = (y_new - tot * 0.5) / sqrt(tot * 0.25)
    z_cc = (y_new + 0.5 - e) / sqrt(v)
    z_np = (y_new - e) / sqrt(e)
    z_var = (y_new - e) / v
    opts = [s(z_half, 2), s(z, 2), s(z_cc, 2), s(z_np, 2), s(z_var, 2)]
    distinct(*opts)
    return {
        "alpha": alpha, "p0": p0, "q0": s(1 - p0, 3), "ln_a": s(ln_a, 4), "ln_q": s(ln_q, 7), "ln_a2": s(ln_a2, 4),
        "nraw": s(nraw, 2), "n": n, "n_two": n_two, "n_approx": n_approx, "rule3": s(3 / p0, 0),
        "t_old": t_old, "y_old": y_old, "t_new": t_new, "y_new": y_new, "tot": tot, "ttot": t_old + t_new,
        "pn": "1/3", "e": s(e, 0), "v": s(v, 0), "sdv": s(sqrt(v), 4), "z": s(z, 2), "z4": s(z, 4),
        "rate_old": s(y_old / t_old, 3), "rate_new": s(y_new / t_new, 3),
        "o_half": opts[0], "o_ok": opts[1], "o_cc": opts[2], "o_np": opts[3], "o_var": opts[4],
    }


def p06():
    old = [128, 135, 117, 142, 150, 131, 119, 126]
    d = [12, 5, -3, 8, 10, 6, -2, 4]
    new = [a - b for a, b in zip(old, d)]
    n = len(d)
    dbar = sum(d) / n
    ss = sum((x - dbar) ** 2 for x in d)
    sd2 = ss / (n - 1)
    t = dbar / sqrt(sd2 / n)
    t_n = dbar / sqrt(ss / n / n)
    t_nosq = dbar / sqrt(sd2)
    mo, mn = sum(old) / n, sum(new) / n
    sso = sum((x - mo) ** 2 for x in old)
    ssn = sum((x - mn) ** 2 for x in new)
    sp2 = (sso + ssn) / (2 * n - 2)
    t_unp = (mo - mn) / sqrt(sp2 * 2 / n)
    t_n1 = dbar / (sqrt(sd2) / sqrt(n - 1))
    opts = [s(t_unp, 2), s(t_nosq, 2), s(t, 2), s(t_n, 2), s(t_n1, 2)]
    distinct(*opts)
    # (b) 符号検定
    pos = sum(1 for x in d if x > 0)
    from math import comb
    tail = sum(comb(n, k) for k in range(pos, n + 1))
    pv = tail / 2 ** n
    p_eq = comb(n, pos) / 2 ** n
    p_up = sum(comb(n, k) for k in range(pos + 1, n + 1)) / 2 ** n
    sopts = [s(p_up, 3), s(p_eq, 3), s(pv, 3), s(2 * pv, 3), s(1 - p_up, 3)]
    distinct(*sopts)
    tab = lambda v: " | ".join(str(x) for x in v)
    return {
        "n": n, "n1": n - 1, "old_row": tab(old), "new_row": tab(new), "d_row": tab(d), "dsum": sum(d),
        "dbar": s(dbar, 1), "ss": s(ss, 0), "sd2": s(sd2, 4), "sd": s(sqrt(sd2), 4), "se": s(sqrt(sd2 / n), 4),
        "t": s(t, 2), "t4": s(t, 4), "tc": 2.365, "tc1": 1.895,
        "mo": s(mo, 2), "mn": s(mn, 2), "sp2": s(sp2, 2),
        "o_unp": opts[0], "o_nosq": opts[1], "o_ok": opts[2], "o_n": opts[3], "o_n1": opts[4],
        "pos": pos, "neg": n - pos, "c8": comb(n, pos), "c7": comb(n, pos + 1) if pos + 1 <= n else 0, "c6": 1,
        "tail": tail, "pow2": 2 ** n,
        "s_up": sopts[0], "s_eq": sopts[1], "s_ok": sopts[2], "s_two": sopts[3], "s_low": sopts[4], "pv4": s(pv, 4),
    }


def p07():
    # (a) VIF
    r2 = 0.84
    vif = 1 / (1 - r2)
    # (b) ブルーシュ・ペーガン
    n, k, r2bp = 60, 3, 0.15
    lm = n * r2bp
    # (c) 欠落変数バイアス
    b_short, b2 = 0.62, 0.12
    rr, sx1, sx2 = 0.6, 2.0, 5.0
    delta = rr * sx2 / sx1
    b1 = b_short - b2 * delta
    delta_w = rr * sx1 / sx2
    opts = [s(b1, 2), s(b_short - b2, 2), s(b_short - b2 * delta_w, 2), s(b_short, 2), s(b_short + b2 * delta, 2)]
    distinct(*opts)
    return {
        "r2": r2, "omr2": s(1 - r2, 2), "vif": s(vif, 2), "sq": s(sqrt(vif), 1), "inv": s(1 / r2, 2), "invsq": s(sqrt(1 / r2), 2),
        "n": n, "k": k, "r2bp": r2bp, "lm": s(lm, 2), "lm_n": s((n - k - 1) * r2bp, 2), "sqlm": s(sqrt(lm), 2),
        "c3_05": 7.81, "c3_01": 11.34, "c4_05": 9.49, "c4_01": 13.28, "c2_05": 5.99,
        "b_short": b_short, "b2": b2, "rr": rr, "sx1": s(sx1, 0), "sx2": s(sx2, 0), "delta": s(delta, 1),
        "bias": s(b2 * delta, 2), "b1": s(b1, 2), "delta_w": s(delta_w, 2),
        "o_ok": opts[0], "o_nod": opts[1], "o_rev": opts[2], "o_same": opts[3], "o_sign": opts[4],
    }


def p08():
    # (a) HT
    pis, ys = [0.30, 0.15, 0.60], [45, 18, 96]
    N = 20
    ht = sum(y / p for y, p in zip(ys, pis))
    # (b) オフセット付きポアソン回帰
    b0, b1, m = -1.20, 0.35, 50
    ev = {-0.85: 0.4274, -1.20: 0.3012, 0.35: 1.4191}
    mu = m * ev[-0.85]
    opts = [s(ev[-0.85], 2), s(m * ev[-1.20], 2), s(mu, 2), s(m * (ev[-1.20] + b1), 2), s(m * ev[0.35], 2)]
    distinct(*opts)
    return {
        "p1": pis[0], "p2": pis[1], "p3": pis[2], "y1": ys[0], "y2": ys[1], "y3": ys[2], "N": N,
        "h1": s(ys[0] / pis[0], 0), "h2": s(ys[1] / pis[1], 0), "h3": s(ys[2] / pis[2], 0),
        "ht": r(ht, 0), "ybar": s(sum(ys) / 3, 0), "Nybar": s(N * sum(ys) / 3, 0), "pisum": s(sum(pis), 2),
        "nexp": s(sum(pis), 2), "wr": s(sum(y * p for y, p in zip(ys, pis)), 1),
        "b0": s(b0, 2), "b1": s(b1, 2), "m": m, "lin": s(b0 + b1, 2),
        "e085": s(ev[-0.85], 4), "e120": s(ev[-1.20], 4), "e035": s(ev[0.35], 4),
        "mu": s(mu, 2), "rate": s(ev[-0.85], 4),
        "o_nooff": opts[0], "o_old": opts[1], "o_ok": opts[2], "o_add": opts[3], "o_nob0": opts[4],
    }


def p09():
    y = {"1": 120, "a": 96, "b": 112, "ab": 80, "c": 110, "ac": 90, "bc": 104, "abc": 70}
    hiB = ((y["ab"] - y["b"]) + (y["abc"] - y["bc"])) / 2
    loB = ((y["a"] - y["1"]) + (y["ac"] - y["c"])) / 2
    ab = (hiB - loB) / 2
    contrast = y["1"] - y["a"] - y["b"] + y["ab"] + y["c"] - y["ac"] - y["bc"] + y["abc"]
    assert abs(contrast / 4 - ab) < 1e-12
    return {**{f"y{k}": v for k, v in y.items()}, "hiB": s(hiB, 1), "loB": s(loB, 1), "ab": r(ab, 1), "ab_s": s(ab, 1),
            "contrast": contrast, "ab8": s(contrast / 8, 2), "ab2": s(contrast / 2, 1), "ss": s(contrast ** 2 / 8, 1),
            "dA_hi1": y["ab"] - y["b"], "dA_hi2": y["abc"] - y["bc"], "dA_lo1": y["a"] - y["1"], "dA_lo2": y["ac"] - y["c"]}


def p11():
    pts = [(1, 2), (2, 1), (2, 3), (8, 8), (9, 6), (4, 4)]
    c1, c2 = pts[0], pts[3]
    d2 = lambda p, c: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
    lab = [0 if d2(p, c1) <= d2(p, c2) else 1 for p in pts]
    for _ in range(10):
        cs = []
        for g in (0, 1):
            mem = [p for p, l in zip(pts, lab) if l == g]
            cs.append((sum(p[0] for p in mem) / len(mem), sum(p[1] for p in mem) / len(mem)))
        new = [0 if d2(p, cs[0]) <= d2(p, cs[1]) else 1 for p in pts]
        if new == lab:
            break
        lab = new
    W = sum(d2(p, cs[l]) for p, l in zip(pts, lab))
    W1 = sum(d2(p, cs[0]) for p, l in zip(pts, lab) if l == 0)
    gm = (sum(p[0] for p in pts) / 6, sum(p[1] for p in pts) / 6)
    T = sum(d2(p, gm) for p in pts)
    W_init = sum(min(d2(p, c1), d2(p, c2)) for p in pts)
    opts = [s(W / 6, 2), s(W1, 2), s(W, 2), s(W_init, 2), s(T - W, 2)]
    distinct(*opts)
    # (b) ナイーブベイズ
    pi = 0.02
    f1a, f1n, f2a, f2n = 0.6, 0.01, 0.7, 0.2
    na = pi * f1a * (1 - f2a)
    nn = (1 - pi) * f1n * (1 - f2n)
    post = na / (na + nn)
    w_yes = pi * f1a * f2a / (pi * f1a * f2a + (1 - pi) * f1n * f2n)
    w_no2 = pi * f1a / (pi * f1a + (1 - pi) * f1n)
    w_nopri = f1a * (1 - f2a) / (f1a * (1 - f2a) + f1n * (1 - f2n))
    bopts = [s(na, 4), s(post, 3), s(w_no2, 3), s(w_yes, 3), s(w_nopri, 3)]
    distinct(*bopts)
    return {
        "c1": s(cs[0][0], 2) + ",\\ " + s(cs[0][1], 2), "c2": s(cs[1][0], 2) + ",\\ " + s(cs[1][1], 2),
        "W": s(W, 2), "W1": s(W1, 2), "W2": s(W - W1, 2), "Winit": s(W_init, 2), "T": s(T, 2),
        "o_avg": opts[0], "o_W1": opts[1], "o_ok": opts[2], "o_init": opts[3], "o_B": opts[4],
        "pi": pi, "f1a": f1a, "f1n": f1n, "f2a": f2a, "f2n": f2n, "g2a": s(1 - f2a, 1), "g2n": s(1 - f2n, 1),
        "qpi": s(1 - pi, 2), "na": s(na, 4), "nn": s(nn, 5), "nsum": s(na + nn, 5), "post4": s(post, 4),
        "b_num": bopts[0], "b_ok": bopts[1], "b_no2": bopts[2], "b_yes": bopts[3], "b_nopri": bopts[4],
    }


def p12():
    r1, r2 = 0.80, 0.50
    det = 1 - r1 ** 2
    phi1 = (r1 - r1 * r2) / det
    phi2 = (r2 - r1 ** 2) / det
    mean, xT, xT1 = 100, 110, 106
    dT, dT1 = xT - mean, xT1 - mean
    fc = mean + phi1 * dT + phi2 * dT1
    fc_ar1 = mean + r1 * dT
    fc_naive = mean + r1 * dT + (r2 - r1 ** 2) * dT1
    return {"r1": r1, "r2": r2, "det": s(det, 2), "phi1": s(phi1, 4), "phi2": s(phi2, 4),
            "n1": s(r1 - r1 * r2, 2), "n2": s(r2 - r1 ** 2, 2),
            "phi1f": "\\frac{10}{9}", "phi2f": "-\\frac{7}{18}",
            "mean": mean, "xT": xT, "xT1": xT1, "dT": dT, "dT1": dT1,
            "t1": s(phi1 * dT, 4), "t2": s(phi2 * dT1, 4), "fc": r(fc, 2), "fc_s": s(fc, 2), "fc4": s(fc, 4),
            "fc_ar1": s(fc_ar1, 2), "fc_naive": s(fc_naive, 2), "fc_nomean": s(phi1 * xT + phi2 * xT1, 2),
            "stat1": s(phi1 + phi2, 4), "stat2": s(phi2 - phi1, 4)}


def p13():
    data = [(1, 1), (2, 1), (2, 0), (3, 1), (4, 1), (4, 1), (5, 0), (6, 1), (8, 0), (9, 1)]
    S = 1.0
    steps = []
    for t in sorted({t for t, e in data if e}):
        n_at = sum(1 for u, _ in data if u >= t)
        d = sum(1 for u, e in data if u == t and e)
        S *= (n_at - d) / n_at
        steps.append((t, n_at, d, S))
    S6 = [x for x in steps if x[0] == 6][0][3]
    S4 = [x for x in steps if x[0] == 4][0][3]
    ev = [u for u, e in data if e]
    ign = sum(1 for u in ev if u > 6) / len(ev)
    asev = sum(1 for u, _ in data if u > 6) / len(data)
    surv = sum(1 for u, e in data if u > 6 or not e) / len(data)
    opts = [s(ign, 3), s(asev, 3), s(S6, 3), s(surv, 3), s(S4, 3)]
    distinct(*opts)
    out = {"o_ign": opts[0], "o_ev": opts[1], "o_ok": opts[2], "o_surv": opts[3], "o_s4": opts[4]}
    for t, n_at, d, Sv in steps:
        out[f"n{t}"] = n_at
        out[f"d{t}"] = d
        out[f"S{t}"] = s(Sv, 4)
    return out


def p14():
    xm, a, u = 10, 2.5, 0.84
    pw = {"m": 2.0814, "p": 0.4805, "um": 1.0722, "big": 97.66}   # 0.16^{-0.4}, 0.16^{0.4}, 0.84^{-0.4}, 0.16^{-2.5}
    x = xm * pw["m"]
    opts = [s(xm * pw["p"], 2), s(xm * pw["um"], 2), s(a * xm / (a - 1), 2), s(x, 2), s(xm * pw["big"], 1)]
    distinct(*opts)
    # (b) 一様分布の上端
    n = 10
    v_mom = 1 / (3 * n)
    v_max = 1 / (n * (n + 2))
    ratio = v_max / v_mom
    v_raw = n / ((n + 1) ** 2 * (n + 2))
    mse_mle = 2 / ((n + 1) * (n + 2))
    bopts = [s(v_raw / v_mom, 2), s(ratio, 2), s(mse_mle / v_mom, 2), s(1.0, 2), s(1 / ratio, 2)]
    distinct(*bopts)
    return {"xm": xm, "a": a, "u": u, "qu": s(1 - u, 2), "ainv": s(1 / a, 1),
            "pm": s(pw["m"], 4), "pp": s(pw["p"], 4), "pum": s(pw["um"], 4), "pbig": s(pw["big"], 2),
            "x": s(x, 2), "o_p": opts[0], "o_u": opts[1], "o_mean": opts[2], "o_ok": opts[3], "o_big": opts[4],
            "n": n, "n1": n + 1, "n2": n + 2, "vmom": f"\\frac{{\\theta^2}}{{{3 * n}}}", "vmax": f"\\frac{{\\theta^2}}{{{n * (n + 2)}}}",
            "ratio_f": f"\\frac{{3}}{{{n + 2}}}", "ratio": s(ratio, 2),
            "b_raw": bopts[0], "b_ok": bopts[1], "b_mse": bopts[2], "b_one": bopts[3], "b_inv": bopts[4]}


def p15():
    px = 0.4
    pm = {1: 0.7, 0: 0.2}
    py = {(1, 1): 0.30, (1, 0): 0.10, (0, 1): 0.20, (0, 0): 0.05}
    inner = {m: py[(1, m)] * px + py[(0, m)] * (1 - px) for m in (0, 1)}
    do = {x: pm[x] * inner[1] + (1 - pm[x]) * inner[0] for x in (0, 1)}
    ace = do[1] - do[0]
    obs = {x: pm[x] * py[(x, 1)] + (1 - pm[x]) * py[(x, 0)] for x in (0, 1)}
    return {"px": px, "qx": s(1 - px, 1), "pm1": pm[1], "pm0": pm[0], "qm1": s(1 - pm[1], 1), "qm0": s(1 - pm[0], 1),
            "y11": s(py[(1, 1)], 2), "y10": s(py[(1, 0)], 2), "y01": s(py[(0, 1)], 2), "y00": s(py[(0, 0)], 2),
            "in1": s(inner[1], 2), "in0": s(inner[0], 2), "do1": s(do[1], 3), "do0": s(do[0], 3),
            "ace": r(ace, 3), "ace_s": s(ace, 3), "obs1": s(obs[1], 2), "obs0": s(obs[0], 2), "naive": s(obs[1] - obs[0], 2),
            "a11": s(py[(1, 1)] * px, 2), "a01": s(py[(0, 1)] * (1 - px), 2), "a10": s(py[(1, 0)] * px, 2), "a00": s(py[(0, 0)] * (1 - px), 2),
            "dpm": s(pm[1] - pm[0], 1), "dinner": s(inner[1] - inner[0], 2)}


def values():
    return {"s08p01": p01(), "s08p02": p02(), "s08p03": p03(), "s08p04": p04(), "s08p05": p05(),
            "s08p06": p06(), "s08p07": p07(), "s08p08": p08(), "s08p09": p09(), "s08p10": p11(),
            "s08p11": p12(), "s08p12": p13(), "s08p13": p14(), "s08p14": p15()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
