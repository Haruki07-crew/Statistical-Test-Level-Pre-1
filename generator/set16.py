"""模擬試験 第16回（set16）の数値。values() が {大問id: {名前: 値}} を返す。

題材：自動車・機械・航空の信頼性と保守。
問題文に示した分布の値（Φ・t・χ²・F・q の点、e^x、ln など）は、ここでも同じ丸めた値を使って答えを計算する。
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


def two_of_three_fail(x):
    """3 個が独立に確率 x で故障するとき、正常が 2 個未満（故障 2 個以上）になる確率。"""
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
    # (b) AUC
    fail = [0.82, 0.64, 0.55, 0.91]
    norm = [0.31, 0.58, 0.47, 0.64, 0.22]
    wins = sum(1 for a in fail for b in norm if a > b)
    ties = sum(1 for a in fail for b in norm if a == b)
    npair = len(fail) * len(norm)
    rows = []
    for a in fail:
        w = sum(1 for b in norm if a > b)
        t = sum(1 for b in norm if a == b)
        rows.append((w, t))
    auc = (wins + 0.5 * ties) / npair
    return {
        "c": c, "q": q, "p": p, "c1": s(1 - c, 2),
        "q2": s(q ** 2, 2), "q1": s(1 - q, 1), "q3": s(q ** 3, 3), "fq": s(fq, 3),
        "p2": s(p ** 2, 4), "p1": s(1 - p, 2), "p3": s(p ** 3, 6), "fp": s(fp, 6),
        "t1": s(c * fq, 4), "t2": s((1 - c) * fp, 7), "ans7": s(ans, 7), "ans": r(ans, 4), "ans_s": s(ans, 4),
        "w_noice": s(fp, 4), "pbar": s(pbar, 3), "w_bar": s(two_of_three_fail(pbar), 4), "w_ice": s(c * fq, 4),
        "w_one": s(c * (1 - (1 - q) ** 3) + (1 - c) * (1 - (1 - p) ** 3), 4),
        "f1": s(fail[0], 2), "f2": s(fail[1], 2), "f3": s(fail[2], 2), "f4": s(fail[3], 2),
        "n1": s(norm[0], 2), "n2": s(norm[1], 2), "n3": s(norm[2], 2), "n4": s(norm[3], 2), "n5": s(norm[4], 2),
        "nf": len(fail), "nn": len(norm), "npair": npair,
        "w1": rows[0][0], "w2": rows[1][0], "w3": rows[2][0], "w4": rows[3][0], "tie2": rows[1][1],
        "wins": wins, "ties": ties, "score": s(wins + 0.5 * ties, 1),
        "auc": r(auc, 3), "auc_s": s(auc, 3), "auc_not": s(wins / npair, 3), "auc_full": s((wins + ties) / npair, 3),
    }


def p02():
    lam_inv = 500          # 平均寿命（時間）
    t = 1000
    e2 = 0.1353            # e^{-2}
    lt = t / lam_inv
    ok = e2 * (1 + lt + lt ** 2 / 2)
    one_spare = e2 * (1 + lt)
    parallel = 1 - (1 - e2) ** 3
    up_to3 = e2 * (1 + lt + lt ** 2 / 2 + lt ** 3 / 6)
    # (b) T2=800 のもとで T1<=200
    t1, t2 = 200, 800
    e04, e12, e16 = 0.6703, 0.3012, 0.2019
    cond = t1 / t2
    trunc = (1 - e04) / (1 - e16)
    rev = (e12 - e16) / (1 - e16)
    return {
        "mean": lam_inv, "lam": s(1 / lam_inv, 3), "t": t, "lt": s(lt, 0), "lt2": s(lt ** 2 / 2, 0),
        "e2": s(e2, 4), "sum3": s(1 + lt + lt ** 2 / 2, 0),
        "a_ok": s(ok, 3), "a_ok4": s(ok, 4), "a_one": s(one_spare, 3), "a_none": s(e2, 3),
        "a_par": s(parallel, 3), "a_up3": s(up_to3, 3), "e2c": s(1 - e2, 4),
        "t1": t1, "t2": t2, "e04": s(e04, 4), "e12": s(e12, 4), "e16": s(e16, 4),
        "b_ok": s(cond, 3), "b_trunc": s(trunc, 3), "b_unc": s(1 - e04, 3), "b_rev": s(rev, 3), "b_comp": s(1 - cond, 3),
        "lt1": s(t1 / lam_inv, 1), "lt2b": s(t2 / lam_inv, 1),
    }


def p03():
    a, b, c = 30, 20, 40       # μm
    # S = X1 + X2 の上側裾：c < s < a+b で密度 (a+b-s)/(4ab)
    tail = (a + b - c) ** 2 / 2 / (4 * a * b)
    ans = 2 * tail
    # 誤答：両方を U(-a,a) とした三角分布、正規近似、片側
    tri = ((2 * a - c) / (2 * a)) ** 2
    var = a ** 2 / 3 + b ** 2 / 3
    # (b) 連の検定
    seq = "+++++---+++++++---++"
    n1, n2 = seq.count("+"), seq.count("-")
    runs = 1 + sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    N = n1 + n2
    ER = 1 + 2 * n1 * n2 / N
    VR = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))
    z = (runs - ER) / sqrt(VR)
    z_noone = (runs - 2 * n1 * n2 / N) / sqrt(VR)
    z_var = (runs - ER) / VR
    z_cc = (runs + 0.5 - ER) / sqrt(VR)
    return {
        "a": a, "b": b, "c": c, "amb": a - b, "apb": a + b, "ab4": 4 * a * b, "dens0": s(1 / (2 * a), 4),
        "over": a + b - c, "over2": s((a + b - c) ** 2 / 2, 0),
        "tail": s(tail, 5), "tailfrac": f"\\frac{{1}}{{{int(round(1 / tail))}}}",
        "ansfrac": f"\\frac{{1}}{{{int(round(1 / ans))}}}",
        "ans": r(ans, 4), "ans_s": s(ans, 4), "ans6": s(ans, 6), "one": s(tail, 4), "tri": s(tri, 4),
        "var": s(var, 2), "sd": s(sqrt(var), 3),
        "seq": seq, "n1": n1, "n2": n2, "N": N, "runs": runs, "nn": 2 * n1 * n2,
        "ER": s(ER, 1), "VRnum": 2 * n1 * n2 * (2 * n1 * n2 - N), "VRden": N ** 2 * (N - 1),
        "VR": s(VR, 4), "sdR": s(sqrt(VR), 4),
        "z": s(z, 2), "z_noone": s(z_noone, 2), "z_var": s(z_var, 2), "z_cc": s(z_cc, 2),
        "ER0": s(2 * n1 * n2 / N, 1),
        "runlist": "、".join(f"${{{ch}}}$（{ln} 個）" for ch, ln in runlens(seq)),
    }


def p04():
    x = [0, 3, 1, 6, 2, 0, 4, 1, 7, 6]
    n = len(x)
    m = sum(x) / n
    ss = sum((v - m) ** 2 for v in x)
    D = ss / m
    s2 = ss / (n - 1)
    # (b) プロビット
    b0, b1, ln4, lg4 = -1.572, 2.0, 1.386, 0.602
    z = b0 + b1 * ln4
    phi = {1.20: 0.8849, 0.37: 0.6443}
    e12 = 0.3012
    zr = r(z, 2)
    assert zr == 1.20
    zl = r(b0 + b1 * lg4, 2)
    assert zl == -0.37
    pod = phi[zr]
    # (c) 間引き
    lam, k = 2.5, 2
    q = pod
    ans = k + lam * (1 - q)
    return {
        "x" + str(i): v for i, v in enumerate(x, 1)
    } | {
        "devs": "+".join(s((v - m) ** 2, 0) for v in x),
        "n": n, "sum": sum(x), "m": s(m, 0), "ss": s(ss, 0), "D": s(D, 2), "s2": s(s2, 2), "ratio": s(s2 / m, 2),
        "df": n - 1, "c9": 16.92, "c10": 18.31, "c9l": 3.33, "D9": s(ss / (n - 1), 2),
        "b0": b0, "b1": s(b1, 1), "ln4": ln4, "lg4": lg4, "z": s(z, 2), "phi12": s(phi[1.20], 4), "phi37": s(phi[0.37], 4),
        "e12": s(e12, 4),
        "pod": s(pod, 3), "pod_c": s(1 - pod, 3), "pod_logit": s(1 / (1 + e12), 3), "pod_lg": s(1 - phi[0.37], 3),
        "zlin": s(b0 + b1 * 4, 2), "zlg": s(b0 + b1 * lg4, 3),
        "lam": lam, "k": k, "q": s(q, 4), "q1": s(1 - q, 4), "lq": s(lam * (1 - q), 5),
        "c_ans": r(ans, 2), "c_ans5": s(ans, 5), "c_scale": s(k / q, 2), "c_add": s(k + lam, 2), "c_q": s(k + lam * q, 2),
    }


def p05():
    t = [0.3, 0.8, 1.0, 1.2, 1.5, 1.6, 1.9, 2.1]
    n = len(t)
    t2 = [v * v for v in t]
    S2 = sum(t2)
    th = S2 / n
    # (b) デルタ法
    t0 = 1.0
    e05 = 0.6065
    assert abs(t0 ** 2 / th - 0.5) < 1e-12
    R = e05
    se_th = th / sqrt(n)
    deriv = R * t0 ** 2 / th ** 2
    se_R = deriv * se_th
    return {
        "n": n, "t1": t[0], "t2": t[1], "t3": t[2], "t4": t[3], "t5": t[4], "t6": t[5], "t7": t[6], "t8": t[7],
        "S2": s(S2, 2), "S1": s(sum(t), 1), "th": s(th, 2),
        "a_ok": s(th, 2), "a_n1": s(S2 / (n - 1), 2), "a_mm": s((sum(t) / n) ** 2, 2), "a_mean": s(sum(t) / n, 2),
        "a_inv": s(n / S2, 2),
        "t0": s(t0, 1), "ratio": s(t0 ** 2 / th, 1), "e05": s(e05, 4), "sqn": s(sqrt(n), 4),
        "se_th": s(se_th, 4), "deriv": s(deriv, 5), "se": r(se_R, 3), "se5": s(se_R, 5),
        "w_nochain": s(R * se_th, 3), "w_th": s(se_th, 3), "w_var": s(deriv * th ** 2 / n, 3), "w_dvar": s(deriv ** 2 * th ** 2 / n, 4),
        "fish": s(n / th ** 2, 1),
    }


def p06():
    n, xbar, sd, mu0 = 10, 41.0, 1.6, 40.0
    se = sd / sqrt(n)
    t = (xbar - mu0) / se
    t1, t2, t10 = 1.833, 2.262, 1.812
    assert t1 < t < t2
    # (b) σ の信頼区間
    ssq = (n - 1) * sd ** 2
    cu, cl = 19.02, 2.70          # χ²(9) の上側 2.5%・97.5% 点
    cu10, cl10 = 20.48, 3.25      # χ²(10)
    cu90, cl90 = 16.92, 3.33      # χ²(9) の上側 5%・95% 点
    fmt = lambda lo, hi: f"({s(lo, 2)},\\ {s(hi, 2)})"
    return {
        "n": n, "xbar": s(xbar, 1), "sd": s(sd, 1), "mu0": s(mu0, 1), "sqn": s(sqrt(n), 4), "se": s(se, 4),
        "diff": s(xbar - mu0, 1), "t": s(t, 2), "t4": s(t, 4), "t_s": s((xbar - mu0) / sd, 2), "t_n": s((xbar - mu0) / (sd / n), 2),
        "tc1": t1, "tc2": t2, "tc10": t10, "df": n - 1,
        "sd2": s(sd ** 2, 2), "ssq": s(ssq, 2), "cu": cu, "cl": cl, "cu10": cu10, "cl10": cl10, "cu90": cu90, "cl90": cl90,
        "v_lo": s(ssq / cu, 4), "v_hi": s(ssq / cl, 4),
        "ci_ok": fmt(sqrt(ssq / cu), sqrt(ssq / cl)), "ci_var": fmt(ssq / cu, ssq / cl),
        "ci_n": fmt(sqrt(n * sd ** 2 / cu), sqrt(n * sd ** 2 / cl)), "ci_10": fmt(sqrt(ssq / cu10), sqrt(ssq / cl10)),
        "ci_90": fmt(sqrt(ssq / cu90), sqrt(ssq / cl90)),
    }


def p07():
    lam, mu, c = 3, 2, 2
    a = lam / mu
    rho = lam / (c * mu)
    term = a ** 2 / 2 / (1 - rho)
    P0 = 1 / (1 + a + term)
    Pw = term * P0
    Lq = Pw * rho / (1 - rho)
    Wq = Lq / lam
    # 誤答
    mm1_fast = rho / (c * mu - lam)          # 速さ 2μ の整備士 1 人
    split = (lam / 2 / mu) / (mu - lam / 2)  # 2 本の M/M/1 に分けた
    W = Wq + 1 / mu
    return {
        "lam": lam, "mu": mu, "a": s(a, 1), "rho": s(rho, 2), "a2": s(a ** 2 / 2, 3), "term": s(term, 1),
        "den": s(1 + a + term, 1), "P0frac": "\\frac{1}{7}", "P0": s(P0, 4), "Pw": s(Pw, 4), "Pwfrac": "\\frac{9}{14}",
        "rr": s(rho / (1 - rho), 0), "Lq": s(Lq, 4), "Wq": r(Wq, 2), "Wq4": s(Wq, 4), "Wq_h": s(Wq * 24, 1),
        "w_fast": s(mm1_fast, 2), "w_split": s(split, 2), "w_W": s(W, 2), "w_Lq": s(Lq, 2),
    }


def p08():
    n = 40
    p1, p2 = 2, 4
    sse1, sse2 = 520.0, 430.0
    df2 = n - p2 - 1
    df1 = n - p1 - 1
    q = p2 - p1
    F = ((sse1 - sse2) / q) / (sse2 / df2)
    F_red = ((sse1 - sse2) / q) / (sse1 / df1)
    F_noq = (sse1 - sse2) / (sse2 / df2)
    F_36 = ((sse1 - sse2) / q) / (sse2 / (n - p2))
    F_ratio = sse1 / sse2
    Fc, Fc37 = 3.27, 3.25
    assert F > Fc and F_red < Fc37
    # (b) 外部スチューデント化残差
    ri = 2.80
    ext = ri * sqrt((df2 - 1) / (df2 - ri ** 2))
    return {
        "n": n, "p1": p1, "p2": p2, "sse1": s(sse1, 0), "sse2": s(sse2, 0), "df2": df2, "df1": df1, "q": q,
        "dsse": s(sse1 - sse2, 0), "num": s((sse1 - sse2) / q, 1), "mse2": s(sse2 / df2, 4), "mse1": s(sse1 / df1, 4),
        "F": s(F, 2), "F4": s(F, 4), "F_red": s(F_red, 2), "F_noq": s(F_noq, 2), "F_36": s(F_36, 2), "F_ratio": s(F_ratio, 2),
        "Fc": Fc, "Fc37": Fc37,
        "ri": s(ri, 2), "ri2": s(ri ** 2, 2), "dfm": df2 - 1, "dd": s(df2 - ri ** 2, 2), "rat": s((df2 - 1) / (df2 - ri ** 2), 4),
        "srat": s(sqrt((df2 - 1) / (df2 - ri ** 2)), 4), "ext": s(ext, 2), "ext4": s(ext, 4),
        "e_inv": s(ri * sqrt((df2 - ri ** 2) / (df2 - 1)), 2), "e_nosq": s(ri * (df2 - 1) / (df2 - ri ** 2), 2),
        "e_sq": s(ri ** 2, 2), "tc34": 2.03,
    }


def p09():
    x1, x0 = 3.2, 2.4
    y1, y0 = 1.10, 1.30
    ols = 0.12
    dy, dx = y1 - y0, x1 - x0
    iv = dy / dx
    return {
        "x1": s(x1, 1), "x0": s(x0, 1), "y1": s(y1, 2), "y0": s(y0, 2), "ols": s(ols, 2),
        "dy": s(dy, 2), "dx": s(dx, 1), "iv": s(iv, 2), "ivabs": s(abs(iv), 2), "w_itt": s(dy, 2), "w_inv": s(dx / dy, 2), "w_mul": s(dy * dx, 2),
        "n1": 600, "n0": 600,
    }


def p10():
    k, m = 4, 6
    means = {"A": 12.0, "B": 14.6, "C": 15.2, "D": 18.2}
    mse, dfe = 4.20, k * (m - 1)
    qc, tc = 3.96, 2.086
    hsd = qc * sqrt(mse / m)
    pairs = [(i, j) for i in "ABCD" for j in "ABCD" if i < j]
    diff = {(i, j): abs(means[j] - means[i]) for i, j in pairs}
    sig = [pq for pq in pairs if diff[pq] > hsd]
    hsd2 = qc * sqrt(2 * mse / m)
    lsd = tc * sqrt(2 * mse / m)
    hsd_nom = qc * sqrt(mse)
    hsd_half = qc * sqrt(mse / (2 * m))
    cnt = lambda h: sum(1 for pq in pairs if diff[pq] > h)
    for h in (hsd, hsd2, lsd, hsd_nom, hsd_half):
        assert min(abs(d - h) for d in diff.values()) > 0.08, h
    return {
        "k": k, "m": m, "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1), "mD": s(means["D"], 1),
        "mse": s(mse, 2), "dfe": dfe, "qc": qc, "tc": tc,
        "msen": s(mse / m, 2), "smsen": s(sqrt(mse / m), 4), "hsd": s(hsd, 2), "hsd4": s(hsd, 4),
        "dBA": s(diff[("A", "B")], 1), "dCA": s(diff[("A", "C")], 1), "dDA": s(diff[("A", "D")], 1),
        "dCB": s(diff[("B", "C")], 1), "dDB": s(diff[("B", "D")], 1), "dDC": s(diff[("C", "D")], 1),
        "nsig": len(sig),
        "h_ok": s(hsd, 2), "h_2": s(hsd2, 2), "h_lsd": s(lsd, 2), "h_nom": s(hsd_nom, 2), "h_half": s(hsd_half, 2),
        "k_ok": cnt(hsd), "k_2": cnt(hsd2), "k_lsd": cnt(lsd), "k_nom": cnt(hsd_nom), "k_half": cnt(hsd_half),
    }


def p11():
    N, e, z = 800, 0.05, 1.96
    n0 = z ** 2 * 0.25 / e ** 2
    nraw = N * n0 / (N - 1 + n0)
    n = math.ceil(nraw)
    nraw2 = n0 / (1 + n0 / N)
    assert math.ceil(nraw2) == n
    # (b) MH
    st = [(10, 90, 20, 80), (30, 70, 45, 55)]
    num = [a * d / (a + b + c + d) for a, b, c, d in st]
    den = [b * c / (a + b + c + d) for a, b, c, d in st]
    mh = sum(num) / sum(den)
    ors = [a * d / (b * c) for a, b, c, d in st]
    A = [sum(v[i] for v in st) for i in range(4)]
    pooled = A[0] * A[3] / (A[1] * A[2])
    return {
        "N": N, "e": e, "z": z, "z2": s(z ** 2, 4), "n0": s(n0, 2), "nraw": s(nraw, 2), "n": n, "N1": N - 1,
        "nraw2": s(nraw2, 2), "w_n0": math.ceil(n0), "w_e": math.ceil(z ** 2 * 0.25 / e),
        "a1": st[0][0], "b1": st[0][1], "c1": st[0][2], "d1": st[0][3], "n1": sum(st[0]),
        "a2": st[1][0], "b2": st[1][1], "c2": st[1][2], "d2": st[1][3], "n2": sum(st[1]),
        "ad1": s(num[0], 2), "bc1": s(den[0], 2), "ad2": s(num[1], 2), "bc2": s(den[1], 2),
        "sad": s(sum(num), 2), "sbc": s(sum(den), 2),
        "mh": s(mh, 3), "mh4": s(mh, 4), "or1": s(ors[0], 3), "or2": s(ors[1], 3), "avg": s(sum(ors) / 2, 3),
        "pooled": s(pooled, 3), "inv": s(1 / mh, 3),
        "A1": A[0], "B1": A[1], "C1": A[2], "D1": A[3],
    }


def p12():
    ev = [3.10, 1.40, 0.70, 0.40, 0.25, 0.15]
    p = len(ev)
    tot = sum(ev)
    cum = [sum(ev[:i + 1]) / tot for i in range(p)]
    kaiser = sum(1 for v in ev if v > 1)
    k80 = next(i + 1 for i, c in enumerate(cum) if c >= 0.8)
    lost = 1 - cum[k80 - 1]
    pct = lambda x: s(100 * x, 1) + "\\%"
    # (b) 重心法（ランス・ウィリアムズ）
    nA, nB = 2, 3
    cA, cB, cC = (0.0, 0.0), (2.0, 0.0), (3.0, 3.0)
    d2 = lambda u, v: (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2
    dAB, dAC, dBC = d2(cA, cB), d2(cA, cC), d2(cB, cC)
    nC = 1
    nn = nA + nB
    lw = nA / nn * dAC + nB / nn * dBC - nA * nB / nn ** 2 * dAB
    nolast = nA / nn * dAC + nB / nn * dBC
    median = 0.5 * dAC + 0.5 * dBC - 0.25 * dAB
    ward = ((nA + nC) * dAC + (nB + nC) * dBC - nC * dAB) / (nA + nB + nC)
    single = min(dAC, dBC)
    return {
        "l1": s(ev[0], 2), "l2": s(ev[1], 2), "l3": s(ev[2], 2), "l4": s(ev[3], 2), "l5": s(ev[4], 2), "l6": s(ev[5], 2),
        "p": p, "tot": s(tot, 2), "c1": pct(cum[0]), "c2": pct(cum[1]), "c3": pct(cum[2]), "c4": pct(cum[3]),
        "s12": s(ev[0] + ev[1], 2), "s123": s(ev[0] + ev[1] + ev[2], 2), "rest": s(tot - ev[0] - ev[1] - ev[2], 2),
        "kaiser": kaiser, "k80": k80, "lost": pct(lost), "lost2": pct(1 - cum[1]), "lost4": pct(1 - cum[3]),
        "nA": nA, "nB": nB, "nC": nC, "nn": nn, "dAB": s(dAB, 0), "dAC": s(dAC, 0), "dBC": s(dBC, 0),
        "wA": s(nA / nn, 1), "wB": s(nB / nn, 1), "wAB": s(nA * nB / nn ** 2, 2),
        "t1": s(nA / nn * dAC, 1), "t2": s(nB / nn * dBC, 1), "t3": s(nA * nB / nn ** 2 * dAB, 2),
        "lw": s(lw, 2), "nolast": s(nolast, 2), "median": s(median, 2), "ward": s(ward, 2), "single": s(single, 2),
        "gx": s((nA * cA[0] + nB * cB[0]) / nn, 1),
    }


def p13():
    c, phi = 0.02, 0.6
    y0, y1 = 2.10, 2.25
    d = y1 - y0
    d1 = c + phi * d
    d2 = c + phi * d1
    f2 = y1 + d1 + d2
    w_last = y1 + d2
    w_phi2 = y1 + c + phi ** 2 * d
    w_lin = y1 + 2 * d
    w_lev = c + phi * (c + phi * y1)
    return {
        "c": c, "phi": phi, "y0": s(y0, 2), "y1": s(y1, 2), "d": s(d, 2), "d1": s(d1, 2), "d2": s(d2, 3),
        "f1": s(y1 + d1, 2), "f2": r(f2, 3), "f2s": s(f2, 3), "w_last": s(w_last, 3), "w_phi2": s(w_phi2, 3), "w_lin": s(w_lin, 3),
        "w_lev": s(w_lev, 3), "mu": s(c / (1 - phi), 2),
    }


def p14():
    from fractions import Fraction as Fr
    EU2 = Fr(1, 3)
    EU4 = Fr(1, 5)
    var = EU4 - EU2 ** 2                 # 4/45
    EUU = Fr(1, 30)                      # E[U^2(1-U)^2]
    cov = EUU - EU2 ** 2                 # -7/90
    pair = (var + cov) / 2               # 対の平均 1 個の分散
    indep2 = var / 2                     # 独立な 2 個の平均の分散
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
