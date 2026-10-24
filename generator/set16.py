"""模擬試験 第16回（set16）の数値。values() が {大問id: {名前: 値}} を返す。

題材：自動車・機械・航空の信頼性と保守。
問題文に示した分布の値（Φ・t・χ²・F の点、e^x、ln など）は、ここでも同じ丸めた値を使って答えを計算する。
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


def sg(x, d):
    """符号つきの文字列（正なら + を付ける）。"""
    return ("+" if x > 0 else "") + s(x, d)


def two_of_three_fail(x):
    """3 個が独立に確率 x で故障するとき、故障が 2 個以上になる確率。"""
    return 3 * x ** 2 * (1 - x) + x ** 3


def runlens(seq):
    out = []
    for ch in seq:
        if out and out[-1][0] == ch:
            out[-1][1] += 1
        else:
            out.append([ch, 1])
    return out


def p01():
    # (a) 共通原因（着氷）を含む 2-out-of-3 多数決
    c, q, p = 0.05, 0.4, 0.02
    fq, fp = two_of_three_fail(q), two_of_three_fail(p)
    ans = c * fq + (1 - c) * fp
    pbar = c * q + (1 - c) * p
    # (b) フィッシャーの線形判別
    mN, mF = (1.0, 2.0), (3.0, 5.0)
    S = ((2.0, 1.0), (1.0, 3.0))
    x = (4.2, 2.0)
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    d = (mF[0] - mN[0], mF[1] - mN[1])
    inv = ((S[1][1] / det, -S[0][1] / det), (-S[0][1] / det, S[0][0] / det))
    cst = lambda w: w[0] * (mN[0] + mF[0]) / 2 + w[1] * (mN[1] + mF[1]) / 2
    lin = lambda w: (w[0] * x[0] + w[1] * x[1]) - cst(w)
    w_ok = (inv[0][0] * d[0] + inv[0][1] * d[1], inv[1][0] * d[0] + inv[1][1] * d[1])
    w_d = d
    w_sign = ((S[1][1] * d[0] + S[0][1] * d[1]) / det, (S[0][1] * d[0] + S[0][0] * d[1]) / det)
    w_diag = (d[0] / S[0][0], d[1] / S[1][1])
    h = {k: lin(w) for k, w in {"ok": w_ok, "d": w_d, "sign": w_sign, "diag": w_diag}.items()}
    assert h["ok"] > 0 and h["d"] < 0 and h["sign"] > 0 and h["diag"] > 0
    wf = lambda w: f"({s(w[0], 1)},\\ {s(w[1], 1)})"
    return {
        "c": c, "q": q, "p": p, "c1": s(1 - c, 2),
        "q2": s(q ** 2, 2), "q1": s(1 - q, 1), "q3": s(q ** 3, 3), "fq": s(fq, 3),
        "p2": s(p ** 2, 4), "p1": s(1 - p, 2), "p3": s(p ** 3, 6), "fp": s(fp, 6),
        "t1": s(c * fq, 4), "t2": s((1 - c) * fp, 7), "ans7": s(ans, 7), "ans": r(ans, 4), "ans_s": s(ans, 4),
        "w_noice": s(fp, 4), "pbar": s(pbar, 3), "w_bar": s(two_of_three_fail(pbar), 4), "w_ice": s(c * fq, 4),
        "mN1": s(mN[0], 1), "mN2": s(mN[1], 1), "mF1": s(mF[0], 1), "mF2": s(mF[1], 1),
        "s11": s(S[0][0], 0), "s12": s(S[0][1], 0), "s22": s(S[1][1], 0), "det": s(det, 0),
        "x1": s(x[0], 1), "x2": s(x[1], 1), "d1": s(d[0], 0), "d2": s(d[1], 0),
        "wok": wf(w_ok), "wd": wf(w_d), "wsign": wf(w_sign), "wdiag": wf(w_diag),
        "w1": s(w_ok[0], 1), "w2": s(w_ok[1], 1), "m1": s((mN[0] + mF[0]) / 2, 1), "m2": s((mN[1] + mF[1]) / 2, 1),
        "cok": s(cst(w_ok), 2), "wx": s(w_ok[0] * x[0] + w_ok[1] * x[1], 2),
        "h_ok": sg(h["ok"], 2), "h_d": sg(h["d"], 2), "h_sign": sg(h["sign"], 2), "h_diag": sg(h["diag"], 2),
        "c_d": s(cst(w_d), 2), "wx_d": s(w_d[0] * x[0] + w_d[1] * x[1], 2),
    }


def p02():
    ls, lm = 0.0012, 0.0008       # シール・モーターの故障率（/時間）
    lam = ls + lm
    t = 250
    e1, e05, e025 = 0.3679, 0.6065, 0.7788
    assert abs(2 * lam * t - 1) < 1e-12
    # (b) 300 時間稼働後、次の 200 時間以内にシール故障で止まる
    e04, e06, e024 = 0.6703, 0.5488, 0.7866
    frac = ls / lam
    return {
        "ls": ls, "lm": lm, "lam": s(lam, 4), "t": t, "lt2": s(2 * lam * t, 0), "e1": s(e1, 4), "e05": s(e05, 4),
        "e025": s(e025, 4), "mean1": s(1 / lam, 0),
        "a_ok": s(e1, 3), "a_one": s(e05, 3), "a_fail": s(1 - e1, 3), "a_mean": s(e025, 3), "a_par": s(2 * e05 - e1, 3),
        "e04": s(e04, 4), "e06": s(e06, 4), "e024": s(e024, 4), "frac": s(frac, 1), "q04": s(1 - e04, 4),
        "b_ok": s(frac * (1 - e04), 3), "b_ok4": s(frac * (1 - e04), 5), "b_joint": s(frac * (e06 - e1), 3),
        "b_alone": s(1 - e024, 3), "b_any": s(1 - e04, 3), "b_frac": s(frac, 3),
    }


def p03():
    # (a) Box-Cox 変換のプロファイル対数尤度
    n, sumlog = 20, 30.0
    sse = {0: 2.40, 0.5: 9.00, 1: 40.0}
    lnr = {0: -2.1203, 0.5: -0.7985, 1: 0.6931}  # ln(SSE/n)
    for k, v in sse.items():
        assert abs(math.log(v / n) - lnr[k]) < 5e-5
    ll = {k: -n / 2 * lnr[k] + (k - 1) * sumlog for k in sse}
    nojac = {k: -n / 2 * lnr[k] for k in sse}
    rev = {k: -n / 2 * lnr[k] - (k - 1) * sumlog for k in sse}
    nfull = {k: -n * lnr[k] + (k - 1) * sumlog for k in sse}
    assert max(ll, key=ll.get) == 1 and max(nojac, key=nojac.get) == 0 and max(rev, key=rev.get) == 0 and max(nfull, key=nfull.get) == 0
    trip = lambda d_: f"({s(d_[0], 2)},\\ {s(d_[0.5], 2)},\\ {s(d_[1], 2)})"
    # (b) 連の検定
    seq = "+++++---+++++++---++"
    n1, n2 = seq.count("+"), seq.count("-")
    runs = len(runlens(seq))
    N = n1 + n2
    ER = 1 + 2 * n1 * n2 / N
    VR = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))
    z = (runs - ER) / sqrt(VR)
    return {
        "n": n, "sumlog": s(sumlog, 1), "sse0": s(sse[0], 2), "sse5": s(sse[0.5], 2), "sse1": s(sse[1], 1),
        "lr0": s(lnr[0], 4), "lr5": s(lnr[0.5], 4), "lr1": s(lnr[1], 4), "nh": n // 2,
        "j0": s(-1 * sumlog, 1), "j5": s(-0.5 * sumlog, 1),
        "l0": s(ll[0], 2), "l5": s(ll[0.5], 2), "l1": s(ll[1], 2),
        "a0": s(nojac[0], 2), "a5": s(nojac[0.5], 2),
        "ok3": trip(ll), "nojac3": trip(nojac), "rev3": trip(rev), "nfull3": trip(nfull),
        "seq": seq, "n1": n1, "n2": n2, "N": N, "runs": runs, "nn": 2 * n1 * n2,
        "runlist": "、".join(f"${{{ch}}}$（{ln} 個）" for ch, ln in runlens(seq)),
        "ER": s(ER, 1), "VRnum": 2 * n1 * n2 * (2 * n1 * n2 - N), "VRden": N ** 2 * (N - 1),
        "VR": s(VR, 4), "sdR": s(sqrt(VR), 4),
        "z": s(z, 2), "z_noone": s((runs - 2 * n1 * n2 / N) / sqrt(VR), 2), "z_var": s((runs - ER) / VR, 2),
        "ER0": s(2 * n1 * n2 / N, 1),
    }


def p04():
    x = [0, 3, 1, 6, 2, 0, 4, 1, 7, 6]
    n = len(x)
    T = sum(x)
    lam0 = 2.0
    m0 = n * lam0
    z = 2 * (sqrt(T) - sqrt(m0))
    # (b) a90：Φ^{-1}(0.9)=1.282
    b0, b1, z90, z95, l9 = -1.600, 2.5, 1.282, 1.645, 2.197
    la = (z90 - b0) / b1
    ex = {round(la, 4): 3.1670, round((l9 - b0) / b1, 4): 4.5667, round((z95 - b0) / b1, 4): 3.6620, round((z90 + b0) / b1, 4): 0.8806}
    assert set(ex) == {1.1528, 1.5188, 1.298, -0.1272}, ex
    ten = 14.22
    # (c) 検出数が与えられたときの総数
    lam, k, qq = 4.0, 2, 0.969
    ans = k + lam * (1 - qq)
    zq = b0 + b1 * 1.386
    return {
        "x" + str(i): v for i, v in enumerate(x, 1)
    } | {
        "n": n, "T": T, "lam0": s(lam0, 1), "m0": s(m0, 0), "sT": s(sqrt(T), 4), "sm0": s(sqrt(m0), 4),
        "z": s(z, 2), "z4": s(z, 4), "z_score": s((T - m0) / sqrt(m0), 2), "z_wald": s((T - m0) / sqrt(T), 2),
        "z_half": s((sqrt(T) - sqrt(m0)) / 2, 2), "z_quart": s((sqrt(T) - sqrt(m0)) * 4, 2),
        "b0": s(b0, 3), "b1": s(b1, 1), "z90": z90, "z95": z95, "l9": l9,
        "la": s(la, 4), "x_ok": s(ex[1.1528], 4), "x_logit": s(ex[1.5188], 4), "x_95": s(ex[1.298], 4), "x_sign": s(ex[-0.1272], 4),
        "la_logit": s((l9 - b0) / b1, 4), "la_95": s((z95 - b0) / b1, 4), "la_sign": s((z90 + b0) / b1, 4),
        "a_ok": s(ex[1.1528], 2), "a_logit": s(ex[1.5188], 2), "a_95": s(ex[1.298], 2), "a_sign": s(ex[-0.1272], 2), "a_ten": s(ten, 2),
        "lam": s(lam, 1), "k": k, "q": s(qq, 3), "q1": s(1 - qq, 3), "lq": s(lam * (1 - qq), 3), "zq": s(zq, 3),
        "c_ans": r(ans, 2), "c_ans4": s(ans, 4), "c_scale": s(k / qq, 2), "c_add": s(k + lam, 2), "c_q": s(k + lam * qq, 2),
    }


def p05():
    t = [0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1]
    n = len(t)
    S2 = sum(v * v for v in t)
    th = S2 / n
    # (b) ベリー・エッセン
    C, rho3, eps = 0.4748, 2.415, 0.05
    root = C * rho3 / eps
    return {
        "n": n, "t1": t[0], "t2": t[1], "t3": t[2], "t4": t[3], "t5": t[4], "t6": t[5], "t7": t[6], "t8": t[7],
        "S2": s(S2, 2), "S1": s(sum(t), 1), "th": s(th, 2),
        "a_ok": s(th, 2), "a_n1": s(S2 / (n - 1), 2), "a_mm": s((sum(t) / n) ** 2, 2), "a_mean": s(sum(t) / n, 2),
        "a_inv": s(n / S2, 2),
        "C": C, "rho3": rho3, "eps": eps, "Cr": s(C * rho3, 6), "root": s(root, 4), "root2": s(root ** 2, 2),
        "nreq": math.ceil(root ** 2), "w_nosq": math.ceil(root), "w_skew": math.ceil((C * 2 / eps) ** 2),
    }


def p06():
    n, xbar, sd, mu0 = 10, 41.0, 1.6, 40.0
    se = sd / sqrt(n)
    t = (xbar - mu0) / se
    t1, t2, t10 = 1.833, 2.262, 1.812
    assert t1 < t < t2
    f = sqrt(1 + 1 / n)
    return {
        "n": n, "xbar": s(xbar, 1), "sd": s(sd, 1), "mu0": s(mu0, 1), "sqn": s(sqrt(n), 4), "se": s(se, 4),
        "diff": s(xbar - mu0, 1), "t": s(t, 2), "t4": s(t, 4), "t_s": s((xbar - mu0) / sd, 2), "t_n": s((xbar - mu0) / (sd / n), 2),
        "tc1": t1, "tc2": t2, "tc10": t10, "df": n - 1, "z95": 1.645,
        "f": s(f, 4), "hw": s(t1 * sd * f, 4), "up": s(xbar + t1 * sd * f, 2),
        "u_mean": s(xbar + t1 * se, 2), "u_nof": s(xbar + t1 * sd, 2), "u_two": s(xbar + t2 * sd * f, 2),
        "u_z": s(xbar + 1.645 * sd * f, 2),
    }


def p07():
    m, rate = 50, 6
    lam = m * rate
    mu, sd = 8.0, 6.0
    ES = lam * mu
    VS = lam * (sd ** 2 + mu ** 2)
    z = 1.645
    B = ES + z * sqrt(VS)
    return {
        "m": m, "rate": rate, "lam": lam, "mu": s(mu, 0), "sd": s(sd, 0), "sd2": s(sd ** 2, 0), "mu2": s(mu ** 2, 0),
        "EX2": s(sd ** 2 + mu ** 2, 0), "ES": s(ES, 0), "VS": s(VS, 0), "sdS": s(sqrt(VS), 4), "z": z,
        "add": s(z * sqrt(VS), 4), "B": r(B, 0), "B2": s(B, 2),
        "w_var": r(ES + z * sqrt(lam * sd ** 2), 0), "w_mu": r(ES + z * sqrt(lam * mu ** 2), 0), "w_sd": r(ES + z * sd, 0),
    }


def p08():
    n, p = 40, 4
    sse = 430.0
    df = n - p - 1
    s2 = sse / df
    yhat, h0 = 52.0, 0.10
    tc, zc = 2.030, 1.96
    se = sqrt(s2 * h0)
    ci = lambda hw: f"({s(yhat - hw, 2)},\\ {s(yhat + hw, 2)})"
    return {
        "n": n, "p": p, "sse": s(sse, 0), "df": df, "s2": s(s2, 4), "yhat": s(yhat, 1), "h0": s(h0, 2), "tc": tc,
        "s2h": s(s2 * h0, 5), "se": s(se, 4), "hw": s(tc * se, 4),
        "ci_ok": ci(tc * se), "ci_pred": ci(tc * sqrt(s2 * (1 + h0))), "ci_z": ci(zc * se),
        "ci_nosq": ci(tc * sqrt(s2) * h0), "ci_n": ci(tc * sqrt(sse / n * h0)),
        "pred_hw": s(tc * sqrt(s2 * (1 + h0)), 4),
        # (b) 正規 QQ プロット
        "rmax": s(2.80, 2), "rmin": s(-1.60, 2), "pmax": s((n - 0.5) / n, 4), "pmin": s(0.5 / n, 4), "zq": s(2.24, 2),
    }


def p09():
    pz = 0.4
    px = {1: 0.7, 0: 0.3}             # P(X=1 | Z=z)
    py = {(1, 1): 0.20, (0, 1): 0.35, (1, 0): 0.08, (0, 0): 0.12}  # P(Y=1 | X, Z)
    pzv = {1: pz, 0: 1 - pz}
    pzx1 = {z: px[z] * pzv[z] for z in (0, 1)}
    px1 = sum(pzx1.values())
    w1 = {z: pzx1[z] / px1 for z in (0, 1)}
    pzx0 = {z: (1 - px[z]) * pzv[z] for z in (0, 1)}
    w0 = {z: pzx0[z] / sum(pzx0.values()) for z in (0, 1)}
    eff = {z: py[(1, z)] - py[(0, z)] for z in (0, 1)}
    att = sum(eff[z] * w1[z] for z in (0, 1))
    ate = sum(eff[z] * pzv[z] for z in (0, 1))
    atc = sum(eff[z] * w0[z] for z in (0, 1))
    naive = sum(py[(1, z)] * w1[z] for z in (0, 1)) - sum(py[(0, z)] * w0[z] for z in (0, 1))
    avg = (eff[0] + eff[1]) / 2
    # (b) 測定誤差による減衰
    b_obs, vres, vu, vtot = -0.20, 1.25, 0.25, 1.60
    beta = b_obs * vres / (vres - vu)
    return {
        "pz": s(pz, 1), "pz0": s(1 - pz, 1), "px1": s(px[1], 1), "px0": s(px[0], 1),
        "y11": s(py[(1, 1)], 2), "y01": s(py[(0, 1)], 2), "y10": s(py[(1, 0)], 2), "y00": s(py[(0, 0)], 2),
        "e1": s(eff[1], 2), "e0": s(eff[0], 2), "j1": s(pzx1[1], 2), "j0": s(pzx1[0], 2), "pX1": s(px1, 2),
        "w1": s(w1[1], 4), "w0": s(w1[0], 4),
        "att": s(att, 3), "att4": s(att, 4), "ate": s(ate, 3), "atc": s(atc, 3), "naive": s(naive, 3), "avg": s(avg, 3),
        "bo": s(b_obs, 2), "vres": s(vres, 2), "vu": s(vu, 2), "vtot": s(vtot, 2), "vx": s(vres - vu, 2),
        "rel": s((vres - vu) / vres, 1), "beta": s(beta, 2), "w_raw": s(b_obs, 2), "w_mul": s(b_obs * (vres - vu) / vres, 2),
        "w_u": s(b_obs * vres / vu, 2), "w_tot": s(b_obs * vtot / (vtot - vu), 2),
    }


def p10():
    a, b, rr = 4, 3, 2
    ms = {"A": 48.0, "B": 30.0, "AB": 12.0, "E": 2.0}
    df = {"A": a - 1, "B": b - 1, "AB": (a - 1) * (b - 1), "E": a * b * (rr - 1)}
    fc = {(3, 6): 4.76, (3, 12): 3.49, (3, 2): 19.16, (6, 12): 3.00}
    FA = ms["A"] / ms["AB"]
    assert FA < fc[(3, 6)] and ms["A"] / ms["E"] > fc[(3, 12)] and ms["AB"] / ms["E"] > fc[(6, 12)]
    return {
        "a": a, "b": b, "r": rr, "msA": s(ms["A"], 1), "msB": s(ms["B"], 1), "msAB": s(ms["AB"], 1), "msE": s(ms["E"], 1),
        "dfA": df["A"], "dfB": df["B"], "dfAB": df["AB"], "dfE": df["E"],
        "F_ok": s(FA, 2), "F_fix": s(ms["A"] / ms["E"], 2), "F_B": s(ms["A"] / ms["B"], 2), "F_AB": s(ms["AB"] / ms["E"], 2),
        "f36": fc[(3, 6)], "f312": fc[(3, 12)], "f32": fc[(3, 2)], "f612": fc[(6, 12)],
    }


def p11():
    y = [1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6, 2.8, 3.1, 3.3, 3.6, 3.9]
    N, k = len(y), 3
    n = N // k
    samples = [[y[i] for i in range(st, N, k)] for st in range(k)]
    means = [sum(v) / n for v in samples]
    Y = sum(y) / N
    vsys = sum((m_ - Y) ** 2 for m_ in means) / k
    S2 = sum((v - Y) ** 2 for v in y) / (N - 1)
    vsrs = (1 - n / N) * S2 / n
    # (b) ケース・コントロール
    ca, cb, ka, kb = 30, 70, 90, 110   # 症例：新型・従来、対照：新型・従来
    OR = ca * kb / (cb * ka)
    rr_naive = (ca / (ca + ka)) / (cb / (cb + kb))
    return {
        **{f"y{i}": s(v, 1) for i, v in enumerate(y, 1)}, "N": N, "k": k, "nn": n,
        "s1": "+".join(s(v, 1) for v in samples[0]), "s2": "+".join(s(v, 1) for v in samples[1]), "s3": "+".join(s(v, 1) for v in samples[2]),
        "m1": s(means[0], 4), "m2": s(means[1], 4), "m3": s(means[2], 4), "Y": s(Y, 4), "sumy": s(sum(y), 1),
        "d1": s(means[0] - Y, 4), "d2": s(means[1] - Y, 4), "d3": s(means[2] - Y, 4),
        "vsys": r(vsys, 4), "vsys6": s(vsys, 6), "vsys_s": s(vsys, 4), "w_k1": s(vsys * k / (k - 1), 4),
        "S2": s(S2, 4), "vsrs": s(vsrs, 4),
        "ca": ca, "cb": cb, "ka": ka, "kb": kb, "nc": ca + cb, "nk": ka + kb,
        "OR": s(OR, 2), "OR_inv": s(1 / OR, 2), "rr_naive": s(rr_naive, 2), "OR_bad": s(ca * ka / (cb * kb), 2),
        "risk1": s(ca / (ca + ka), 2), "risk0": s(cb / (cb + kb), 3),
    }


def p12():
    ev = [3.10, 1.40, 0.70, 0.40, 0.25, 0.15]
    p = len(ev)
    tot = sum(ev)
    cum = [sum(ev[:i + 1]) / tot for i in range(p)]
    k80 = next(i + 1 for i, c in enumerate(cum) if c >= 0.8)
    pct = lambda x: s(100 * x, 1) + "\\%"
    # (b) 正規混合の負担率
    pi1, pi2, m1, m2, x = 0.7, 0.3, 2.0, 5.0, 3.8
    e1, e2 = 0.1979, 0.4868          # e^{-1.62}, e^{-0.72}
    e1b, e2b = 0.1653, 0.3012        # e^{-1.8}, e^{-1.2}
    assert abs((x - m1) ** 2 / 2 - 1.62) < 1e-12 and abs((x - m2) ** 2 / 2 - 0.72) < 1e-12
    resp = pi2 * e2 / (pi1 * e1 + pi2 * e2)
    return {
        "l1": s(ev[0], 2), "l2": s(ev[1], 2), "l3": s(ev[2], 2), "l4": s(ev[3], 2), "l5": s(ev[4], 2), "l6": s(ev[5], 2),
        "p": p, "tot": s(tot, 2), "c1": pct(cum[0]), "c2": pct(cum[1]), "c3": pct(cum[2]),
        "s12": s(ev[0] + ev[1], 2), "s123": s(ev[0] + ev[1] + ev[2], 2), "rest": s(tot - ev[0] - ev[1] - ev[2], 2),
        "lost": pct(1 - cum[k80 - 1]), "lost2": pct(1 - cum[1]), "lost4": pct(1 - cum[3]),
        "pi1": pi1, "pi2": pi2, "mu1": s(m1, 1), "mu2": s(m2, 1), "xx": x, "dx1": s(x - m1, 1), "dx2": s(x - m2, 1),
        "e1": s(e1, 4), "e2": s(e2, 4), "e1b": s(e1b, 4), "e2b": s(e2b, 4),
        "n1": s(pi1 * e1, 5), "n2": s(pi2 * e2, 5), "den": s(pi1 * e1 + pi2 * e2, 5),
        "g_ok": s(resp, 3), "g_now": s(e2 / (e1 + e2), 3), "g_prior": s(pi2, 3), "g_comp": s(1 - resp, 3),
        "g_abs": s(pi2 * e2b / (pi1 * e1b + pi2 * e2b), 3),
    }


def p13():
    c, phi = 0.02, 0.6
    y0, y1 = 2.10, 2.25
    d = y1 - y0
    d1 = c + phi * d
    d2 = c + phi * d1
    f2 = y1 + d1 + d2
    return {
        "c": c, "phi": phi, "y0": s(y0, 2), "y1": s(y1, 2), "d": s(d, 2), "d1": s(d1, 2), "d2": s(d2, 3),
        "f2": r(f2, 3), "f2s": s(f2, 3), "w_last": s(y1 + d2, 3), "w_phi2": s(y1 + c + phi ** 2 * d, 3), "w_lin": s(y1 + 2 * d, 3),
        "mu": s(c / (1 - phi), 2),
    }


def p14():
    from fractions import Fraction as Fr
    EU2 = Fr(1, 3)
    var = Fr(1, 5) - EU2 ** 2            # 4/45
    EUU = Fr(1, 30)                      # E[U^2(1-U)^2]
    cov = EUU - EU2 ** 2                 # -7/90
    pair = (var + cov) / 2
    indep2 = var / 2
    ratio = pair / indep2
    rho = cov / var
    fr = lambda x: f"\\frac{{{x.numerator}}}{{{x.denominator}}}" if x.denominator != 1 else str(x.numerator)
    frs = lambda x: ("-" + fr(-x)) if x < 0 else fr(x)
    return {
        "var": fr(var), "cov": frs(cov), "pair": fr(pair), "indep2": fr(indep2), "rho": frs(rho),
        "sumvc": fr(var + cov), "EUU": fr(EUU),
        "ok": s(float(ratio), 3), "w_single": s(float(pair / var), 4), "w_nocov": s(1.0, 3),
        "w_sign": s(float(((var - cov) / 2) / indep2), 3), "w_rho2": s(float(1 - rho ** 2), 3),
    }


def values():
    return {"s16p01": p01(), "s16p02": p02(), "s16p03": p03(), "s16p04": p04(), "s16p05": p05(),
            "s16p06": p06(), "s16p07": p07(), "s16p08": p08(), "s16p09": p09(), "s16p10": p10(),
            "s16p11": p11(), "s16p12": p12(), "s16p13": p13(), "s16p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
