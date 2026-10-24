"""模擬試験 第21回（set21）の数値。values() が {大問id: {名前: 値}} を返す。

題材：学校・入試・学習アプリ・図書館。
問題文に示した分布の値（Φ・t の点、e^x・log など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from math import exp, factorial, sqrt


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
    # ポリアの壺：赤 r・白 w、取り出した玉を戻し同じ色を add 個加える
    rd, wh, add = 3, 2, 2
    N = rd + wh
    pr1 = rd / N
    pr2_r = (rd + add) / (N + add)
    pr2_w = rd / (N + add)
    joint = pr1 * pr2_r
    pr2 = joint + (1 - pr1) * pr2_w
    ans = joint / pr2
    return {
        "r": rd, "w": wh, "add": add, "N": N, "Na": N + add, "ra": rd + add,
        "j_num": rd * (rd + add), "o_num": wh * rd, "den": N * (N + add), "s_num": rd * (rd + add) + wh * rd,
        "pr2": s(pr2, 1), "ans": r(ans, 4), "ans_s": s(ans, 4), "ans6": s(ans, 6),
        "d_prior": s(pr1, 4), "d_norep": s((rd - 1) / (N - 1), 4),
    }


def p02():
    n, T = 40, 28
    ph = T / n
    pq = ph * (1 - ph)
    unb = T * (n - T) / (n * (n - 1))
    return {
        "n": n, "T": T, "nT": n - T, "ph": s(ph, 1), "qh": s(1 - ph, 1), "pq": s(pq, 2), "n1": n - 1,
        "Tnum": T * (n - T), "uden": n * (n - 1),
        "unb": s(unb, 4), "unb5": s(unb, 5), "mle": s(pq, 4), "shr": s(pq * (n - 1) / n, 4),
        "vph": s(pq / n, 4), "both": s(T * (n - T) / (n - 1) ** 2, 4),
    }


def p03():
    m1, m2, m3 = 20, 30, 60
    # (a) 3 つの指数分布の最大値の期待値（包除原理）
    pair = lambda a, b: 1 / (1 / a + 1 / b)
    p12, p13, p23 = pair(m1, m2), pair(m1, m3), pair(m2, m3)
    allm = 1 / (1 / m1 + 1 / m2 + 1 / m3)
    emax = m1 + m2 + m3 - (p12 + p13 + p23) + allm
    # (b) 密度 c(L-x) (0<=x<=L) の中央値
    L = 4
    med = L - L / sqrt(2)  # x^2 - 2Lx + L^2/2 = 0 の小さい解
    return {
        "m1": m1, "m2": m2, "m3": m3, "msum": m1 + m2 + m3,
        "p12": s(p12, 0), "p13": s(p13, 0), "p23": s(p23, 0), "psum": s(p12 + p13 + p23, 0), "allm": s(allm, 0),
        "emax": s(emax, 0), "d_min": s(allm, 0), "d_maxm": max(m1, m2, m3), "d_sum": m1 + m2 + m3,
        "d_nolast": s(m1 + m2 + m3 - (p12 + p13 + p23), 0),
        "L": L, "cinv": L ** 2 // 2, "twoL": 2 * L, "L2h": L ** 2 // 2,
        "med": r(med, 2), "med_s": s(med, 2), "med4": s(med, 4), "big": s(L + L / sqrt(2), 2),
        "xmean": s(L / 3, 2), "d_mid2": s(L / 2, 2), "sq": s(L / sqrt(2), 4),
    }


def p04():
    # (a) 3 モデルの AIC と赤池重み（e の値は問題文の丸めた値）
    ll = [-210.4, -206.9, -206.3]
    k = [1, 2, 3]
    aic = [-2 * a + 2 * b for a, b in zip(ll, k)]
    amin = min(aic)
    dl = [a - amin for a in aic]
    ev = {5.0: 0.0821, 0.0: 1.0, 0.8: 0.6703}
    w = [ev[round(x, 1)] for x in dl]
    wbest = 1 / sum(w)
    # 過分散パラメータを数え忘れた場合（M3 を 2 個とする）
    aic_w = [aic[0], aic[1], -2 * ll[2] + 4]
    # (b) 1 標本の比率の検定（連続修正つき）
    n, x, p0 = 250, 35, 0.10
    mu = n * p0
    sd = sqrt(n * p0 * (1 - p0))
    z = (x - 0.5 - mu) / sd
    return {
        "ll1": s(ll[0], 1), "ll2": s(ll[1], 1), "ll3": s(ll[2], 1), "k1": k[0], "k2": k[1], "k3": k[2],
        "a1": s(aic[0], 1), "a2": s(aic[1], 1), "a3": s(aic[2], 1),
        "dl1": s(dl[0], 1), "dl3": s(dl[2], 1), "e1": s(ev[5.0], 4), "e3": s(ev[0.8], 4),
        "wsum": s(sum(w), 4), "w": r(wbest, 3), "w_s": s(wbest, 3), "w5": s(wbest, 5),
        "aw3": s(aic_w[2], 1),
        "n": n, "x": x, "p0": s(p0, 2), "pct": 10, "mu": s(mu, 0), "var": s(n * p0 * (1 - p0), 1), "sd": s(sd, 4),
        "num": s(x - 0.5 - mu, 1), "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4),
        "z_nc": s((x - mu) / sd, 2), "z_ph": s((x / n - p0) / sqrt((x / n) * (1 - x / n) / n), 2),
        "z_plus": s((x + 0.5 - mu) / sd, 2), "phat": s(x / n, 2),
        "N": 12000, "ns": 300, "k": 12000 // 300,
    }


def p05():
    # (a) ウェルチの信頼区間（サタスウェイトの自由度を切り捨て）
    n1, m1, s1 = 12, 68.5, 8.0
    n2, m2, s2 = 10, 62.0, 4.0
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    tt = {9: 2.262, 16: 2.120, 20: 2.086}
    dfi = int(df)
    assert dfi == 16
    d = m1 - m2
    iv = lambda h: (s(d - h, 2), s(d + h, 2))
    sp = sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) * sqrt(1 / n1 + 1 / n2)
    ok, pl, d20, zz, d9 = iv(tt[16] * se), iv(tt[20] * sp), iv(tt[20] * se), iv(1.96 * se), iv(tt[9] * se)
    # (b) リスク比・オッズ比・NNT
    a, b, c, dd = 18, 12, 16, 24
    r1, r0 = a / (a + b), c / (c + dd)
    rr = r1 / r0
    orr = a * dd / (b * c)
    nnt = 1 / (r1 - r0)
    rr_col = (a / (a + c)) / (b / (b + dd))
    return {
        "n1": n1, "m1": m1, "s1": s1, "n2": n2, "m2": m2, "s2": s2, "t9": s(tt[9], 3), "t16": s(tt[16], 3), "t20": s(tt[20], 3),
        "v1": s(v1, 4), "v2": s(v2, 4), "vs": s(v1 + v2, 4), "se": s(se, 4), "dfn": s((v1 + v2) ** 2, 3),
        "dfd1": s(v1 ** 2 / (n1 - 1), 4), "dfd2": s(v2 ** 2 / (n2 - 1), 4), "dfd": s(v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1), 4),
        "df": s(df, 2), "dfi": dfi, "h": s(tt[16] * se, 3), "d": s(d, 1),
        "ok_l": ok[0], "ok_u": ok[1], "pl_l": pl[0], "pl_u": pl[1], "d20_l": d20[0], "d20_u": d20[1],
        "z_l": zz[0], "z_u": zz[1], "d9_l": d9[0], "d9_u": d9[1],
        "a": a, "b": b, "c": c, "dd": dd, "n_t": a + b, "n_c": c + dd, "r1": s(r1, 2), "r0": s(r0, 2),
        "rd": s(r1 - r0, 2), "rr": s(rr, 2), "or": s(orr, 2), "nnt": s(nnt, 1), "rr_col": s(rr_col, 2),
        "d_nnt": s(1 / r1, 1), "pa": s(a / (a + c), 4), "pb": s(b / (b + dd), 4),
        "odd1": s(a / b, 2), "odd0": s(c / dd, 4),
    }


def p06():
    b0, b1, b2, b3, x = 1.20, 0.020, 0.90, -0.010, 70
    diff = b2 + b3 * x
    cross = -b2 / b3
    assert abs(cross - 90) < 1e-9
    return {
        "b0": s(b0, 2), "b1": s(b1, 3), "b2": s(b2, 2), "b3": s(b3, 3), "x": x, "b3x": s(b3 * x, 2),
        "diff": s(diff, 2), "cross": s(cross, 0), "d_sign": s(b2 - b3 * x, 2), "d_b1": s(b2 / b1, 0),
        "slope": s(b1 + b3, 3), "g_gen": s(b0 + b1 * x, 2), "g_rec": s(b0 + b2 + (b1 + b3) * x, 2), "b02": s(b0 + b2, 2),
        "nb3": s(-b3, 3),
    }


def p07():
    mu0, sig, n, xbar = 60, 15, 36, 64.5
    se = sig / sqrt(n)
    z = (xbar - mu0) / se
    phi = {0.3: 0.6179, 0.72: 0.7642, 1.8: 0.9641}
    z_sd = (xbar - mu0) / sig
    z_var = (xbar - mu0) / (sig ** 2 / n)
    assert round(z, 2) == 1.8 and round(z_sd, 2) == 0.3 and round(z_var, 2) == 0.72
    p = 1 - phi[1.8]
    rho = 0.6
    return {
        "mu0": mu0, "sig": sig, "n": n, "xbar": xbar, "se": s(se, 1), "dx": s(xbar - mu0, 1), "z": s(z, 1),
        "phi03": s(phi[0.3], 4), "phi072": s(phi[0.72], 4), "phi18": s(phi[1.8], 4),
        "p": s(p, 3), "p4": s(p, 4), "d_two": s(2 * p, 3), "d_sd": s(1 - phi[0.3], 3), "d_var": s(1 - phi[0.72], 3),
        "d_cmp": s(phi[1.8], 3), "zsd": s(z_sd, 1), "zvar": s(z_var, 2), "varn": s(sig ** 2 / n, 2),
        "rho": rho, "l1": s(1 + rho, 1), "l2": s(1 - rho, 1), "c1": s(100 * (1 + rho) / 2, 0), "c_r": s(100 * rho, 0),
        "c_sq": s(100 * (1 + rho) ** 2 / 4, 0), "l1sq": s(1 + rho ** 2, 2), "l2sq": s(1 - rho ** 2, 2),
    }


def p08():
    xs = [0, 1, 2, 4]
    ps = [0.4, 0.3, 0.2, 0.1]
    mu = sum(x * p for x, p in zip(xs, ps))
    m2 = sum((x - mu) ** 2 * p for x, p in zip(xs, ps))
    m4 = sum((x - mu) ** 4 * p for x, p in zip(xs, ps))
    raw4 = sum(x ** 4 * p for x, p in zip(xs, ps))
    ku = m4 / m2 ** 2
    out = {
        "mu": s(mu, 1), "m2": s(m2, 2), "m4": s(m4, 4), "m22": s(m2 ** 2, 4), "ku4": s(ku, 4), "ex": s(ku - 3, 2), "ex4": s(ku - 3, 4),
        "d_m4": s(m4, 2), "d_m2": s(m4 / m2, 2), "d_no3": s(ku, 2), "d_raw": s(raw4 / m2 ** 2 - 3, 2), "raw4": s(raw4, 1),
    }
    for i, (x, p) in enumerate(zip(xs, ps)):
        dv = x - mu
        out[f"x{i}"] = x
        out[f"p{i}"] = p
        out[f"dv{i}"] = s(dv, 1)
        out[f"sq{i}"] = s(dv ** 2 * p, 3)
        out[f"f{i}"] = s(dv ** 4 * p, 5)
    # (b) 1 日 5 回までの上限がある幾何分布
    p, cap = 0.3, 5
    q = 1 - p
    en = sum(q ** k for k in range(cap))
    p_none = q ** cap
    e_succ = sum(k * p * q ** (k - 1) for k in range(1, cap + 1))
    out.update({
        "p": p, "q": s(q, 1), "cap": cap, "en": s(en, 2), "en4": s(en, 4), "none": s(p_none, 3), "none5": s(p_none, 5),
        "d_inf": s(1 / p, 2), "d_succ": s(e_succ, 2), "d_cmp": s(1 - p_none, 3), "d_m1": s(q ** (cap - 1), 3),
        "qp": [s(q ** k, 4) for k in range(cap)],
    })
    out["qps"] = "+".join(out.pop("qp"))
    return out


def p09():
    import numpy as np
    P = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.4, 0.6]])
    P2 = P @ P
    ans = (P2 @ P)[0, 2]
    return {
        "p11": s(P[0, 0], 1), "p12": s(P[0, 1], 1), "p21": s(P[1, 0], 1), "p22": s(P[1, 1], 1), "p23": s(P[1, 2], 1),
        "p32": s(P[2, 1], 1), "p33": s(P[2, 2], 1),
        "q1": s(P2[0, 0], 2), "q2": s(P2[0, 1], 2), "q3": s(P2[0, 2], 2),
        "t2": s(P2[0, 1] * P[1, 2], 3), "t3": s(P2[0, 2] * P[2, 2], 3),
        "path1": s(P[0, 0] * P[0, 1] * P[1, 2], 3), "path2": s(P[0, 1] * P[1, 1] * P[1, 2], 3), "path3": s(P[0, 1] * P[1, 2] * P[2, 2], 3),
        "ans": r(ans, 3), "ans_s": s(ans, 3), "d_two": s(P2[0, 2], 2),
    }


def p10():
    p0, p1, alpha = 0.6, 0.9, 0.05
    pw0 = {n: r(p0 ** n, 4) for n in range(5, 9)}
    pw1 = {n: r(p1 ** n, 4) for n in (5, 6, 8)}
    n = next(k for k in range(1, 50) if pw0.get(k, p0 ** k) <= alpha)
    n2 = next(k for k in range(1, 50) if pw0.get(k, p0 ** k) <= alpha / 2)
    assert n == 6 and n2 == 8
    beta = 1 - pw1[n]
    return {
        "p0": p0, "p1": p1, "alpha": alpha, "n": n, "n2": n2,
        "a5": s(pw0[5], 4), "a6": s(pw0[6], 4), "a7": s(pw0[7], 4), "a8": s(pw0[8], 4),
        "b5": s(pw1[5], 4), "b6": s(pw1[6], 4), "b8": s(pw1[8], 4),
        "beta": s(beta, 3), "beta4": s(beta, 4), "d_n5": s(1 - pw1[5], 3), "d_pow": s(pw1[n], 3), "d_a": s(pw0[n], 3),
        "d_n8": s(1 - pw1[8], 3),
    }


def p11():
    SA, SB, SAB, SE = 180, 420, 60, 540
    a, b, rr = 3, 4, 3
    N = a * b * rr
    dA, dB, dAB, dE = a - 1, b - 1, (a - 1) * (b - 1), a * b * (rr - 1)
    MSA, MSAB, MSE = SA / dA, SAB / dAB, SE / dE
    SEp, dEp = SE + SAB, dE + dAB
    MSEp = SEp / dEp
    # (b) ポアソン・ガンマの事後予測（与えた値）
    a0, b0, m, tot = 3, 1, 5, 12
    ash, brt = a0 + tot, b0 + m
    vals = {"p15": 0.0990, "p12": 0.1573, "e24": 0.0907, "e25": 0.0821}
    return {
        "SA": SA, "SB": SB, "SAB": SAB, "SE": SE, "ST": SA + SB + SAB + SE, "rr": rr, "N": N,
        "dA": dA, "dB": dB, "dAB": dAB, "dE": dE, "MSA": s(MSA, 0), "MSAB": s(MSAB, 0), "MSE": s(MSE, 1),
        "FAB": s(MSAB / MSE, 2), "SEp": SEp, "dEp": dEp, "MSEp": s(MSEp, 0), "FA": s(MSA / MSEp, 2),
        "d_nopool": s(MSA / MSE, 2), "d_ssonly": s(MSA / (SEp / dE), 2), "d_dfonly": s(MSA / (SE / dEp), 2),
        "d_ab": s(MSA / MSAB, 2), "MSEw": s(SEp / dE, 0), "MSEd": s(SE / dEp, 0),
        "a0": a0, "b0": b0, "m": m, "tot": tot, "ash": ash, "brt": brt, "brt1": brt + 1, "ash1": ash - 1,
        "pm": s(ash / brt, 1), "mle": s(tot / m, 1),
        "v15": s(vals["p15"], 4), "v12": s(vals["p12"], 4), "v24": s(vals["e24"], 4), "v25": s(vals["e25"], 4),
        "ans": s(vals["p15"], 3), "d_pm": s(vals["e25"], 3), "d_mle": s(vals["e24"], 3), "d_prior": s(0.5 ** a0, 3),
        "d_data": s(vals["p12"], 3),
    }


def p12():
    # (a) ロジスティック回帰の確率と限界効果
    eta, beta, e04 = 0.4, 0.30, 1.4918
    p = e04 / (1 + e04)
    me = beta * p * (1 - p)
    # (b) 平均値代入
    n, miss, s2 = 50, 10, 1.20
    nobs = n - miss
    v_imp = (nobs - 1) * s2 / (n - 1)
    # (c) 対数変換の逆変換（対数正規の平均）
    mu, v = 0.50, 0.36
    ev = {0.32: 1.3771, 0.5: 1.6487, 0.68: 1.9739, 0.86: 2.3632}
    return {
        "eta": eta, "beta": s(beta, 2), "e04": e04, "e04p": s(1 + e04, 4), "pp": s(p, 3), "pp4": s(p, 4), "qq4": s(1 - p, 4),
        "me": s(me, 3), "me4": s(me, 4), "d_q": s(1 - p, 3), "d_b": s(beta, 3), "d_bp": s(beta * p, 3), "d_bq": s(beta * (1 - p), 3),
        "e07": s(1 / (1 + exp(-0.7)), 4), "dexact": s(1 / (1 + exp(-0.7)) - p, 4),
        "n": n, "miss": miss, "nobs": nobs, "s2": s(s2, 2), "nobs1": nobs - 1, "n1": n - 1,
        "num": s((nobs - 1) * s2, 2), "vimp": r(v_imp, 3), "vimp_s": s(v_imp, 3), "vimp5": s(v_imp, 5),
        "ratio_v": s(v_imp / s2, 3), "d_n": s((nobs - 1) * s2 / n, 3),
        "mu": s(mu, 2), "v": s(v, 2), "vh": s(v / 2, 2), "mv": s(mu + v / 2, 2), "mvf": s(mu + v, 2), "mvm": s(mu - v / 2, 2),
        "e032": s(ev[0.32], 4), "e050": s(ev[0.5], 4), "e068": s(ev[0.68], 4), "e086": s(ev[0.86], 4),
        "ans": s(ev[0.68], 2), "d_med": s(ev[0.5], 2), "d_full": s(ev[0.86], 2), "d_minus": s(ev[0.32], 2), "d_log": s(mu + v / 2, 2),
    }


def p13():
    # (a) シルエット係数（C は {A,B,C} に属する）
    dCA, dCB, dCD, dCE = 3.5, 2.0, 4.0, 4.2
    a = (dCA + dCB) / 2
    b = (dCD + dCE) / 2
    sil = (b - a) / max(a, b)
    a_self = (0 + dCA + dCB) / 3
    # (b) エントロピーによる情報利得（与えた h の値）
    n, nq, nl, ql, nr, qr = 20, 8, 6, 5, 14, 3
    h = {"root": 0.971, "l": 0.650, "r": 0.750}
    wl, wr = nl / n, nr / n
    child = wl * h["l"] + wr * h["r"]
    ig = h["root"] - child
    # (a) 3 つの距離で最も近い生徒
    A, B, C = (2.9, 0.6, -0.3), (2.0, 1.6, -1.7), (0.5, 0.2, -0.5)
    dA = [abs(x - c) for x, c in zip(A, C)]; dB = [abs(x - c) for x, c in zip(B, C)]
    euc = lambda d: sqrt(sum(v * v for v in d))
    near = lambda f_: "A" if f_(dA) < f_(dB) else "B"
    assert (near(euc), near(sum), near(max)) == ("B", "A", "B") and near(min) == "A" and near(lambda d: sum(d)) == "A"
    dm = {"a1": s(A[0], 1), "a2": s(A[1], 1), "a3": s(A[2], 1), "b1": s(B[0], 1), "b2": s(B[1], 1), "b3": s(B[2], 1),
          "c1": s(C[0], 1), "c2": s(C[1], 1), "c3": s(C[2], 1),
          "da1": s(dA[0], 1), "da2": s(dA[1], 1), "da3": s(dA[2], 1), "db1": s(dB[0], 1), "db2": s(dB[1], 1), "db3": s(dB[2], 1),
          "sa": s(sum(v * v for v in dA), 2), "sb": s(sum(v * v for v in dB), 2), "ea": s(euc(dA), 3), "eb": s(euc(dB), 3),
          "ma": s(sum(dA), 1), "mb": s(sum(dB), 1), "ca": s(max(dA), 1), "cb": s(max(dB), 1)}
    return {
        **{"dm_" + k_: v_ for k_, v_ in dm.items()},
        "a": s(a, 2), "b": s(b, 2), "ba": s(b - a, 2), "sil": s(sil, 2), "sil4": s(sil, 4),
        "d_min": s((min(dCD, dCE) - min(dCA, dCB)) / max(min(dCD, dCE), min(dCA, dCB)), 2),
        "d_self": s((b - a_self) / b, 2), "d_diva": s((b - a) / a, 2), "d_rev": s((a - b) / max(a, b), 2), "a_self": s(a_self, 3),
        "n": n, "nq": nq, "nl": nl, "ql": ql, "nr": nr, "qr": qr, "hroot": s(h["root"], 3), "hl": s(h["l"], 3), "hr": s(h["r"], 3),
        "wl": s(wl, 1), "wr": s(wr, 1), "cl": s(wl * h["l"], 3), "cr": s(wr * h["r"], 3), "child": s(child, 3),
        "ig": r(ig, 2), "ig_s": s(ig, 2), "ig3": s(ig, 3), "d_avg": s(h["root"] - (h["l"] + h["r"]) / 2, 2),
    }


def values():
    return {
        "s21p01": p01(), "s21p02": p02(), "s21p03": p03(), "s21p04": p04(), "s21p05": p05(),
        "s21p06": p06(), "s21p07": p07(), "s21p08": p08(), "s21p09": p09(), "s21p10": p10(),
        "s21p11": p11(), "s21p12": p12(), "s21p13": p13(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
