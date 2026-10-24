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
    # (b) 分散の異なる 2 群の 2 次判別（平均は等しい）
    sN, sF, piF = 1.0, 3.0, 0.1
    piN = 1 - piF
    ln3 = 1.0986
    k = 1 / sN ** 2 - 1 / sF ** 2
    lnr = 3 * ln3                     # ln(piN*sF/(piF*sN)) = ln 27
    cth = lambda L, two=2: sqrt(two * L / k)
    cc = cth(lnr)
    return {
        "c": c, "q": q, "p": p, "c1": s(1 - c, 2),
        "q2": s(q ** 2, 2), "q1": s(1 - q, 1), "q3": s(q ** 3, 3), "fq": s(fq, 3),
        "p2": s(p ** 2, 4), "p1": s(1 - p, 2), "p3": s(p ** 3, 6), "fp": s(fp, 6),
        "t1": s(c * fq, 4), "t2": s((1 - c) * fp, 7), "ans7": s(ans, 7), "ans": r(ans, 4), "ans_s": s(ans, 4),
        "w_noice": s(fp, 4), "pbar": s(pbar, 3), "w_bar": s(two_of_three_fail(pbar), 4), "w_ice": s(c * fq, 4),
        "sF": s(sF, 0), "piF": piF, "piN": s(piN, 1), "ln3": ln3, "k": "\\frac{8}{9}", "lnr": s(lnr, 4),
        "c2": s(cth(lnr) ** 2, 4), "c_ok": s(cc, 2), "c_noprior": s(cth(ln3), 2), "c_nosig": s(cth(2 * ln3), 2),
        "c_two": s(cth(lnr, 1), 2), "c_var": s(cth(4 * ln3), 2),
    }


def p02():
    ls, lm = 0.0012, 0.0008       # シール・モーターの故障率（/時間）
    lam = ls + lm
    # (a) 冷待機（稼働 1 台＋予備 2 台）
    t = 1000
    e2 = 0.1353
    lt = lam * t
    assert abs(lt - 2) < 1e-12
    # (b) 300 時間稼働後、次の 200 時間以内にシール故障で止まる
    e04, e06, e1, e024 = 0.6703, 0.5488, 0.3679, 0.7866
    frac = ls / lam
    return {
        "ls": ls, "lm": lm, "lam": s(lam, 4), "mean1": s(1 / lam, 0), "t": t, "lt": s(lt, 0), "e2": s(e2, 4),
        "sum3": s(1 + lt + lt ** 2 / 2, 0),
        "a_ok": s(e2 * (1 + lt + lt ** 2 / 2), 3), "a_ok4": s(e2 * (1 + lt + lt ** 2 / 2), 4), "a_one": s(e2 * (1 + lt), 3),
        "a_none": s(e2, 3), "a_par": s(1 - (1 - e2) ** 3, 3), "a_up3": s(e2 * (1 + lt + lt ** 2 / 2 + lt ** 3 / 6), 3),
        "e04": s(e04, 4), "e06": s(e06, 4), "e1": s(e1, 4), "e024": s(e024, 4), "frac": s(frac, 1), "q04": s(1 - e04, 4),
        "b_ok": s(frac * (1 - e04), 3), "b_ok4": s(frac * (1 - e04), 5), "b_joint": s(frac * (e06 - e1), 3),
        "b_alone": s(1 - e024, 3), "b_any": s(1 - e04, 3), "b_frac": s(frac, 3),
    }


def p03():
    # (a) Box-Cox 変換のプロファイル対数尤度
    n, sumlog = 20, 30.0
    sse = {0: 2.40, 0.5: 9.40, 1: 40.0}
    lnr = {0: -2.1203, 0.5: -0.7550, 1: 0.6931}  # ln(SSE/n)
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
    z = (T - m0) / sqrt(m0)
    # (b) 2 値の説明変数のロジスティック回帰の最尤推定
    a1, n1, a0, n0 = 18, 40, 12, 60      # 沿岸：き裂あり/機数、内陸：き裂あり/機数
    OR = (a1 / (n1 - a1)) / (a0 / (n0 - a0))
    lnt = {"or": (3.273, 1.186), "rr": (2.25, 0.811), "cnt": (1.5, 0.405), "odds1": (0.818, -0.201)}
    import math as _m
    for v, w in lnt.values():
        assert abs(_m.log(v) - w) < 5e-4
    assert abs(OR - 3.273) < 5e-4 and abs((a1 / n1) / (a0 / n0) - 2.25) < 1e-12 and abs(a1 / a0 - 1.5) < 1e-12
    # (c) 検出数が与えられたときの総数
    lam, k, qq = 4.0, 2, 0.95
    ans = k + lam * (1 - qq)
    return {
        "x" + str(i): v for i, v in enumerate(x, 1)
    } | {
        "n": n, "T": T, "lam0": s(lam0, 1), "m0": s(m0, 0), "sm0": s(sqrt(m0), 4), "diff": s(T - m0, 0),
        "z": r(z, 2), "z4": s(z, 4), "z_s": s(z, 2), "z_wald": s((T - m0) / sqrt(T), 2), "z_vst": s(2 * (sqrt(T) - sqrt(m0)), 2),
        "z_mean": s((T / n - lam0) / sqrt(lam0), 2), "z_var": s((T - m0) / m0, 2),
        "a1": a1, "n1": n1, "b1": n1 - a1, "a0": a0, "n0": n0, "b0": n0 - a0,
        "o1": s(a1 / (n1 - a1), 4), "o0": s(a0 / (n0 - a0), 4), "OR": s(OR, 3), "lnOR": lnt["or"][1], "lnRR": lnt["rr"][1],
        "lncnt": lnt["cnt"][1], "lno1": lnt["odds1"][1], "lnor_s": s(lnt["or"][1], 2), "lnrr_s": s(lnt["rr"][1], 2),
        "lncnt_s": s(lnt["cnt"][1], 2), "lno1_s": s(lnt["odds1"][1], 2), "OR_s": s(OR, 2), "RR": s((a1 / n1) / (a0 / n0), 2),
        "p1": s(a1 / n1, 2), "p0": s(a0 / n0, 2), "b0hat": s(math.log(a0 / (n0 - a0)), 3), "se1": s(sqrt(1 / a1 + 1 / (n1 - a1) + 1 / a0 + 1 / (n0 - a0)), 3),
        "lam": s(lam, 1), "k": k, "q": s(qq, 2), "q1": s(1 - qq, 2), "lq": s(lam * (1 - qq), 2),
        "c_ans": r(ans, 2), "c_ans4": s(ans, 4), "c_ans2": s(ans, 2), "c_scale": s(k / qq, 2), "c_add": s(k + lam, 2), "c_q": s(k + lam * qq, 2),
    }


def p05():
    t = [0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1]
    n = len(t)
    S2 = sum(v * v for v in t)
    th = S2 / n
    return {
        "n": n, "t1": t[0], "t2": t[1], "t3": t[2], "t4": t[3], "t5": t[4], "t6": t[5], "t7": t[6], "t8": t[7],
        "S2": s(S2, 2), "S1": s(sum(t), 1), "th": s(th, 2),
        "a_ok": s(th, 2), "a_n1": s(S2 / (n - 1), 2), "a_mm": s((sum(t) / n) ** 2, 2), "a_mean": s(sum(t) / n, 2),
        "a_inv": s(n / S2, 2),
    }


def p06():
    n, xbar, sd, mu0 = 10, 41.0, 1.6, 40.0
    se = sd / sqrt(n)
    t = (xbar - mu0) / se
    t1, t2, t10 = 1.833, 2.262, 1.812
    assert t1 < t < t2
    return {
        "n": n, "xbar": s(xbar, 1), "sd": s(sd, 1), "mu0": s(mu0, 1), "sqn": s(sqrt(n), 4), "se": s(se, 4),
        "diff": s(xbar - mu0, 1), "t": s(t, 2), "t4": s(t, 4), "t_s": s((xbar - mu0) / sd, 2), "t_n": s((xbar - mu0) / (sd / n), 2),
        "tc1": t1, "tc2": t2, "tc10": t10, "df": n - 1,
        **p06b(),
    }


def p06b():
    """(b) フィッシャーの z 変換による相関係数の区間。tanh は問題文の表（x を小数第 3 位に丸めた値）で戻す。"""
    rr, n = 0.70, 10
    ln = 1.7346                               # ln(17/3)
    z = ln / 2
    sq = {7: 2.6458, 8: 2.8284, 10: 3.1623}
    t8 = 2.306
    tanh = lambda x: math.tanh(r(x, 3))
    def ci(lo, hi):
        f2 = lambda v: s(v, 2).replace("-0.00", "0.00")
        return f"({f2(lo)},\\ {f2(hi)})"
    h = 1.96 / sq[7]
    cands = {"ok": (z - h, z + h), "n": (z - 1.96 / sq[10], z + 1.96 / sq[10]), "n2": (z - 1.96 / sq[8], z + 1.96 / sq[8]),
             "t": (z - t8 / sq[7], z + t8 / sq[7]), "half": ((ln - h) / 2, (ln + h) / 2)}
    out = {k: ci(tanh(a), tanh(b)) for k, (a, b) in cands.items()}
    assert len(set(out.values())) == 5
    xs = sorted({r(v, 3) for ab in cands.values() for v in ab if abs(v) > 0.01})
    return {
        "r": s(rr, 2), "ln": s(ln, 4), "sq7": s(sq[7], 4), "sq8": s(sq[8], 4), "sq10": s(sq[10], 4), "t8": s(t8, 3),
        "tabx": " | ".join(s(x, 3) for x in xs), "tabsep": "---|" * len(xs), "taby": " | ".join(s(math.tanh(x), 4) for x in xs),
        "z": s(z, 4), "h": s(h, 4), "zl": s(z - h, 4), "zu": s(z + h, 4), "zl3": s(z - h, 3), "zu3": s(z + h, 3),
        "tl": s(tanh(z - h), 2), "tu": s(tanh(z + h), 2),
        **{"ci_" + k: v for k, v in out.items()},
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
    fit = [20, 40, 60]
    rsd = [2.1, 4.0, 6.1]
    return {
        "n": n, "p": p, "sse": s(sse, 0), "df": df, "s2": s(s2, 4), "yhat": s(yhat, 1), "h0": s(h0, 2), "tc": tc,
        "s2h": s(s2 * h0, 5), "se": s(se, 4), "hw": s(tc * se, 4),
        "ci_ok": ci(tc * se), "ci_pred": ci(tc * sqrt(s2 * (1 + h0))), "ci_z": ci(zc * se),
        "ci_nosq": ci(tc * sqrt(s2) * h0), "ci_n": ci(tc * sqrt(sse / n * h0)),
        "pred_hw": s(tc * sqrt(s2 * (1 + h0)), 4),
        "f1": fit[0], "f2": fit[1], "f3": fit[2], "r1": s(rsd[0], 1), "r2": s(rsd[1], 1), "r3": s(rsd[2], 1),
        "q1": s(rsd[0] / fit[0], 3), "q2": s(rsd[1] / fit[1], 3), "q3": s(rsd[2] / fit[2], 3),
        "v1": s(rsd[0] ** 2 / fit[0], 3), "v2": s(rsd[1] ** 2 / fit[1], 3), "v3": s(rsd[2] ** 2 / fit[2], 3),
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
    k, m = 4, 6
    means = {"A": 12.0, "B": 14.6, "C": 15.2, "D": 18.2}
    mse, dfe = 4.20, k * (m - 1)
    qc, tc = 3.96, 2.086
    hsd = qc * sqrt(mse / m)
    pairs = [(i, j) for i in "ABCD" for j in "ABCD" if i < j]
    diff = {(i, j): abs(means[j] - means[i]) for i, j in pairs}
    hsd2 = qc * sqrt(2 * mse / m)
    lsd = tc * sqrt(2 * mse / m)
    hsd_nom = qc * sqrt(mse)
    hsd_half = qc * sqrt(mse / (2 * m))
    cnt = lambda h: sum(1 for pq in pairs if diff[pq] > h)
    for h in (hsd, hsd2, lsd, hsd_nom, hsd_half):
        assert min(abs(d - h) for d in diff.values()) > 0.08, h
    return {
        "k": k, "m": m, "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1), "mD": s(means["D"], 1),
        "mse": s(mse, 2), "dfe": dfe, "qc": qc, "tc": tc, "smsen": s(sqrt(mse / m), 4), "hsd": s(hsd, 2), "hsd4": s(hsd, 4),
        "dBA": s(diff[("A", "B")], 1), "dCA": s(diff[("A", "C")], 1), "dDA": s(diff[("A", "D")], 1),
        "dCB": s(diff[("B", "C")], 1), "dDB": s(diff[("B", "D")], 1), "dDC": s(diff[("C", "D")], 1),
        "h_ok": s(hsd, 2), "h_2": s(hsd2, 2), "h_lsd": s(lsd, 2), "h_nom": s(hsd_nom, 2), "h_half": s(hsd_half, 2),
        "k_ok": cnt(hsd), "k_2": cnt(hsd2), "k_lsd": cnt(lsd), "k_nom": cnt(hsd_nom), "k_half": cnt(hsd_half),
    }


def p11():
    N, n = 800, 50
    ybar, xbar, sy2, sx2, sxy = 15.0, 10.0, 25.0, 16.0, 18.0
    R = ybar / xbar
    num = sy2 - 2 * R * sxy + R ** 2 * sx2
    eff = num / sy2
    rho = sxy / sqrt(sy2 * sx2)
    # (b) リスク差と NNT
    n1, w1, n0, w0 = 400, 32, 400, 56
    p1, p0 = w1 / n1, w0 / n0
    rd = p0 - p1
    rr = p1 / p0
    return {
        "N": N, "nn": n, "ybar": s(ybar, 1), "xbar": s(xbar, 1), "sy2": s(sy2, 1), "sx2": s(sx2, 1), "sxy": s(sxy, 1),
        "R": s(R, 1), "R2": s(R ** 2, 2), "t2": s(2 * R * sxy, 1), "t3": s(R ** 2 * sx2, 1), "num": s(num, 1),
        "eff": r(eff, 2), "eff_s": s(eff, 2), "rho": s(rho, 2), "w_reg": s(1 - rho ** 2, 2), "w_no2": s((sy2 - R * sxy + R ** 2 * sx2) / sy2, 2),
        "w_comp": s(1 - eff, 2), "w_inv": s(sy2 / num, 2),
        "n1": n1, "w1": w1, "n0": n0, "w0": w0, "p1": s(p1, 2), "p0": s(p0, 2), "rd": s(rd, 2), "rr": s(rr, 3),
        "nnt_raw": s(1 / rd, 1), "nnt": math.ceil(1 / rd), "w_p0": math.ceil(1 / p0), "w_p1": math.ceil(1 / p1), "w_rr": math.ceil(1 / (1 - rr)),
        "w_pct": round(100 * rd),
    }


def p12():
    L = {1: (0.90, 0.30), 2: (0.80, -0.40), 3: (0.20, 0.95)}
    r12 = L[1][0] * L[2][0] + L[1][1] * L[2][1]
    cosv = r12 / (sqrt(L[1][0] ** 2 + L[1][1] ** 2) * sqrt(L[2][0] ** 2 + L[2][1] ** 2))
    # (b) 重心法（ランス・ウィリアムズ）
    nA, nB, nC = 2, 3, 1
    cA, cB, cC = (0.0, 0.0), (2.0, 0.0), (3.0, 3.0)
    d2 = lambda u, v: (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2
    dAB, dAC, dBC = d2(cA, cB), d2(cA, cC), d2(cB, cC)
    nn = nA + nB
    lw = nA / nn * dAC + nB / nn * dBC - nA * nB / nn ** 2 * dAB
    return {
        "l11": s(L[1][0], 2), "l12": s(L[1][1], 2), "l21": s(L[2][0], 2), "l22": s(L[2][1], 2), "l31": s(L[3][0], 2), "l32": s(L[3][1], 2),
        "p1": s(L[1][0] * L[2][0], 2), "p2": s(L[1][1] * L[2][1], 2), "r12": s(r12, 2), "w_pc1": s(L[1][0] * L[2][0], 2),
        "w_sign": s(L[1][0] * L[2][0] - L[1][1] * L[2][1], 2), "w_cos": s(cosv, 2), "w_pc2": s(L[1][1] * L[2][1], 2),
        "h1": s(L[1][0] ** 2 + L[1][1] ** 2, 2),
        "nA": nA, "nB": nB, "nC": nC, "nn": nn, "dAB": s(dAB, 0), "dAC": s(dAC, 0), "dBC": s(dBC, 0),
        "wA": s(nA / nn, 1), "wB": s(nB / nn, 1), "wAB": s(nA * nB / nn ** 2, 2),
        "t1": s(nA / nn * dAC, 1), "t2": s(nB / nn * dBC, 1), "t3": s(nA * nB / nn ** 2 * dAB, 2),
        "lw": s(lw, 2), "nolast": s(nA / nn * dAC + nB / nn * dBC, 2), "median": s(0.5 * dAC + 0.5 * dBC - 0.25 * dAB, 2),
        "ward": s(((nA + nC) * dAC + (nB + nC) * dBC - nC * dAB) / (nA + nB + nC), 2), "single": s(min(dAC, dBC), 2),
        "gx": s((nA * cA[0] + nB * cB[0]) / nn, 1),
    }


def p13():
    n, r1, r2 = 100, 0.55, 0.40
    phi = r2 / r1
    phi22 = (r2 - r1 ** 2) / (1 - r1 ** 2)
    # ρ1 = (1+φθ)(φ+θ)/(1+2φθ+θ²) を θ の 2 次方程式 (ρ1-φ)θ² + (2φρ1-1-φ²)θ + (ρ1-φ) = 0 として解き、|θ|<1 の解をとる
    a, b = r1 - phi, 2 * phi * r1 - 1 - phi ** 2
    th = [(-b + sg_ * sqrt(b * b - 4 * a * a)) / (2 * a) for sg_ in (1, -1)]
    theta = [x for x in th if abs(x) < 1][0]
    return {
        "n": n, "r1": s(r1, 2), "r2": s(r2, 2), "r1sq": s(r1 ** 2, 4),
        "phi": r(phi, 3), "phi_s": s(phi, 3), "phi5": s(phi, 5), "theta": s(theta, 3), "phi22": s(phi22, 3),
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
