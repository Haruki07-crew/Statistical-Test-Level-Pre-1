"""模擬試験 第22回（set22）の数値。values() が {大問id: {名前: 値}} を返す。

題材：物流倉庫・EC配送・在庫管理・コールセンター。
問題文に示した関数の値（指数・対数・正規分布の値、χ² 分布・F 分布の点）は、ここでも同じ丸めた値を使って答えを計算する。
"""
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


def ceil(x):
    return int(-(-x // 1))


def p01():
    # (a) 20 箱中 3 箱が破損。無作為な順に開けて 3 箱目で初めて破損箱
    N, D = 20, 3
    G = N - D
    pa = (G / N) * ((G - 1) / (N - 1)) * (D / (N - 2))
    with_rep = (G / N) ** 2 * (D / N)
    atleast = 1 - (G / N) * ((G - 1) / (N - 1)) * ((G - 2) / (N - 2))
    # (b) 抜取検査：20 箱を調べ破損 2 箱以上で不合格
    n, c = 20, 2
    p_good, p_bad = 0.05, 0.20
    q19g, q19b = 0.3774, 0.01441          # 問題文に示す 0.95^19, 0.8^19
    acc_g = (1 - p_good) * q19g + n * p_good * q19g     # P(X<=1 | 0.05)
    acc_b = (1 - p_bad) * q19b + n * p_bad * q19b       # P(X<=1 | 0.20)
    alpha = 1 - acc_g
    beta = acc_b
    alpha0 = 1 - (1 - p_good) * q19g                   # X=1 の項を忘れた
    beta0 = (1 - p_bad) * q19b
    # (c) 破損率の推定に必要な標本数
    p0, e, z = 0.15, 0.03, 1.96
    nraw = z ** 2 * p0 * (1 - p0) / e ** 2
    n_half = z ** 2 * 0.25 / e ** 2
    n_645 = 1.645 ** 2 * p0 * (1 - p0) / e ** 2
    n_width = z ** 2 * p0 * (1 - p0) / (2 * e) ** 2
    return {
        "N": N, "D": D, "G": G, "G1": G - 1, "G2": G - 2, "N1": N - 1, "N2": N - 2,
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "num": G * (G - 1) * D, "den": N * (N - 1) * (N - 2),
        "with_rep": s(with_rep, 4), "atleast": s(atleast, 4),
        "n": n, "c": c, "pg": s(p_good, 2), "pb": s(p_bad, 2), "qg": s(1 - p_good, 2), "qb": s(1 - p_bad, 1),
        "q19g": s(q19g, 4), "q19b": s(q19b, 5), "npg": s(n * p_good, 1), "npb": s(n * p_bad, 1),
        "acc_g": s(acc_g, 4), "acc_b": s(acc_b, 4), "alpha": s(alpha, 3), "beta": s(beta, 3),
        "alpha4": s(alpha, 4), "beta4": s(beta, 4), "acc_g3": s(acc_g, 3), "rej_b3": s(1 - beta, 3),
        "alpha0": s(alpha0, 3), "beta0": s(beta0, 3), "fg": s((1 - p_good) * q19g, 4), "fb": s((1 - p_bad) * q19b, 5),
        "gg": s(n * p_good * q19g, 4), "gb": s(n * p_bad * q19b, 5),
        "p0": s(p0, 2), "q0": s(1 - p0, 2), "e": s(e, 2), "z": s(z, 2), "z2": s(z ** 2, 4),
        "pq": s(p0 * (1 - p0), 4), "e2": s(e ** 2, 4), "nraw": s(nraw, 1), "nreq": ceil(nraw),
        "n_half": ceil(n_half), "n_halfraw": s(n_half, 1), "n_645": ceil(n_645), "n_width": ceil(n_width),
        "nlow": int(nraw),
    }


def p02():
    # (a) リードタイム 2 日の需要 Po(3) が在庫 5 を超える確率
    lam, L, stock = 1.5, 2, 5
    mu = lam * L
    em3 = 0.0498                       # 問題文に示す e^{-3}
    terms = [1.0]
    for k in range(1, stock + 1):
        terms.append(terms[-1] * mu / k)
    ssum = sum(terms)
    pa = 1 - em3 * ssum
    p_ge5 = 1 - em3 * (ssum - terms[-1])
    # (b) 分位点回帰による発注量
    cu, co = 400, 100
    lines = {0.20: (80, 20), 0.25: (84, 22), 0.50: (100, 30), 0.75: (116, 40), 0.80: (121, 44)}
    tau = cu / (cu + co)
    q = lambda t, x: lines[t][0] + lines[t][1] * x
    return {
        "lam": s(lam, 1), "L": L, "stock": stock, "mu": s(mu, 0), "em3": s(em3, 4), "stock1": stock + 1,
        "t0": s(terms[0], 0), "t1": s(terms[1], 0), "t2": s(terms[2], 1), "t3": s(terms[3], 1),
        "t4": s(terms[4], 3), "t5": s(terms[5], 3), "tsum": s(ssum, 1), "cdf": s(em3 * ssum, 5),
        "pa": r(pa, 3), "pa_s": s(pa, 3), "pa5": s(pa, 5), "p_ge5": s(p_ge5, 3),
        "cu": cu, "co": co, "cucoh": cu + co, "tau": s(tau, 1),
        "a20": lines[0.20][0], "b20": lines[0.20][1], "a25": lines[0.25][0], "b25": lines[0.25][1],
        "a50": lines[0.50][0], "b50": lines[0.50][1], "a75": lines[0.75][0], "b75": lines[0.75][1],
        "a80": lines[0.80][0], "b80": lines[0.80][1],
        "q80": q(0.80, 1), "q20": q(0.20, 1), "q25": q(0.25, 1), "q50": q(0.50, 1), "q75": q(0.75, 1),
        "q80x0": q(0.80, 0), "tau_rev": s(co / (cu + co), 1), "tau_ratio": s(co / cu, 2), "tau_1m": s(1 - co / cu, 2),
    }


def p03():
    # (a) 指数分布の混合のハザード
    w1, m1, w2, m2, t = 0.6, 1.0, 0.4, 3.0, 3
    em3, em1 = 0.0498, 0.3679     # e^{-3}, e^{-1}
    S = w1 * em3 + w2 * em1
    f = w1 / m1 * em3 + w2 / m2 * em1
    h = f / S
    h0 = w1 / m1 + w2 / m2
    mean = w1 * m1 + w2 * m2
    # (b) 一様分布 [0, θ] の 5 個の標本中央値と標本平均
    th, n = 120, 5
    k = (n + 1) // 2
    a, b = k, n - k + 1
    vbeta = a * b / ((a + b) ** 2 * (a + b + 1))
    vmed = th ** 2 * vbeta
    vmean = th ** 2 / 12 / n
    ratio = vmed / vmean
    asym = (1 / (4 * n * (1 / th) ** 2)) / vmean
    return {
        "w1": s(w1, 1), "w2": s(w2, 1), "m1": s(m1, 0), "m2": s(m2, 0), "t": t, "em3": s(em3, 4), "em1": s(em1, 4),
        "S1": s(w1 * em3, 5), "S2": s(w2 * em1, 5), "S": s(S, 5), "f1": s(w1 / m1 * em3, 5), "f2": s(w2 / m2 * em1, 5),
        "f": s(f, 5), "h": s(h, 3), "h4": s(h, 4), "h0": s(h0, 3), "f3": s(f, 3), "S3": s(S, 3), "invmean": s(1 / mean, 3),
        "mean": s(mean, 1), "w2m2": s(w2 / m2, 4), "t3": s(t / m2, 0),
        "th": th, "n": n, "k": k, "ba": a, "bb": b, "bab": a + b, "bab1": a + b + 1,
        "vbeta": s(vbeta, 5), "vmed": s(vmed, 2), "vmean": s(vmean, 1), "ratio": s(ratio, 3),
        "ratio_inv": s(1 / ratio, 3), "asym": s(asym, 3), "pi2": s(1.5708, 3), "th2": th ** 2,
        "vbeta_frac": "1/28", "vmean_frac": "1/60",
    }


def p04():
    # (a) 半径 a の円内に一様に分布する配送先までの距離
    a = 6.0
    ER = 2 * a / 3
    # (b) 重要度サンプリング：提案は [0,a] の一様分布、重み w(r)=f(r)/g(r)=2r/a
    rs = [0.9, 2.4, 3.9, 5.4]
    n = len(rs)
    w = [2 * x / a for x in rs]
    is_est = sum(x * wi for x, wi in zip(rs, w)) / n
    sn_est = sum(x * wi for x, wi in zip(rs, w)) / sum(w)
    naive = sum(rs) / n
    inv_w = sum(x * (a / (2 * x)) for x in rs) / n
    return {
        "a": s(a, 0), "ER": s(ER, 2), "Ehalf": s(a / 2, 2), "Erms": s(a / sqrt(2), 2), "E34": s(3 * a / 4, 2),
        "E13": s(a / 3, 2), "a2": s(a ** 2, 0), "Er2": s(a ** 2 / 2, 0),
        "r1": s(rs[0], 1), "r2": s(rs[1], 1), "r3": s(rs[2], 1), "r4": s(rs[3], 1), "nis": n,
        "w1": s(w[0], 2), "w2": s(w[1], 2), "w3": s(w[2], 2), "w4": s(w[3], 2),
        "rw1": s(rs[0] * w[0], 3), "rw2": s(rs[1] * w[1], 3), "rw3": s(rs[2] * w[2], 3), "rw4": s(rs[3] * w[3], 3),
        "srw": s(sum(x * wi for x, wi in zip(rs, w)), 3), "sw": s(sum(w), 1),
        "is": s(is_est, 2), "is4": s(is_est, 4), "sn": s(sn_est, 2), "naive": s(naive, 2), "invw": s(inv_w, 2),
    }


def p05():
    # (a) 対数正規分布の後処理時間
    mu, sig, c = 1.6, 0.5, 15
    ln15 = 2.708
    z = r((ln15 - mu) / sig, 2)
    phi = 0.9868                          # Φ(2.22)
    pa = 1 - phi
    zvar = (ln15 - mu) / sig ** 2
    log10_15 = 1.176
    zlog10 = (log10_15 - mu) / sig
    # (b) アーランの損失式（M/M/3/3）
    lam, mean_min, cc = 20, 6, 3
    muh = 60 / mean_min
    aa = lam / muh
    terms = [1.0]
    for k in range(1, cc + 1):
        terms.append(terms[-1] * aa / k)
    B = terms[-1] / sum(terms)
    acc = lam * (1 - B)
    # 誤答用：M/M/3 の待つ確率（待たせる方式と取り違え）
    rho = aa / cc
    tail = terms[-1] / (1 - rho)
    pwait = tail / (sum(terms[:-1]) + tail)
    # (c) デルタ法
    lc = lam
    return {
        "mu": s(mu, 1), "sig": s(sig, 1), "c": c, "ln15": s(ln15, 3), "z": s(z, 2), "zraw": s((ln15 - mu) / sig, 3),
        "phi": s(phi, 4), "pa": s(pa, 3), "pa4": s(pa, 4), "pa2": s(2 * pa, 3), "pcomp": s(phi, 3),
        "pvar": "0.000", "zvar": s(zvar, 2), "log10": s(log10_15, 3), "zlog": s(zlog10, 2), "plog": s(0.8023, 3),
        "phi085": "0.8023", "diff": s(ln15 - mu, 3), "med": s(4.953, 2),
        "lam": lam, "mean": mean_min, "cc": cc, "muh": s(muh, 0), "aa": s(aa, 0),
        "e0": s(terms[0], 0), "e1": s(terms[1], 0), "e2": s(terms[2], 0), "e3": s(terms[3], 4), "esum": s(sum(terms), 4),
        "B": s(B, 4), "acc": r(acc, 2), "acc_s": s(acc, 2), "acc4": s(acc, 4), "omB": s(1 - B, 4),
        "acc_wait": s(lam * (1 - pwait), 2), "lost": s(lam * B, 2),
        "lc": lc, "dv": s(0.25, 3), "dv_nosq": s(1 / (2 * sqrt(lc)), 3),
        "dv_nolam": s(1 / (4 * lc), 4), "dv_sd": s(sqrt(lc), 2), "dv_raw": s(lc, 1),
        "gp": s(1 / (2 * sqrt(lc)), 4), "gp2": s(1 / (4 * lc), 4),
    }


def p06():
    # (a) 2 標本コルモゴロフ・スミルノフ統計量
    A = [18, 22, 26, 30, 41]
    B = [24, 33, 36, 44, 48]
    pts = sorted(A + B)
    FA = lambda x: sum(1 for v in A if v <= x) / len(A)
    FB = lambda x: sum(1 for v in B if v <= x) / len(B)
    D = max(abs(FA(x) - FB(x)) for x in pts)
    DB = max(abs(FA(x) - FB(x)) for x in B)
    # (b) バートレット検定
    k, ni = 3, 6
    v = [4.0, 9.0, 16.0]
    lnv = [1.3863, 2.1972, 2.7726]
    sp2 = sum((ni - 1) * x for x in v) / (k * (ni - 1))
    lnsp = 2.2687
    Nk = k * (ni - 1)
    M = Nk * lnsp - (ni - 1) * sum(lnv)
    C = 1 + (1 / (3 * (k - 1))) * (k / (ni - 1) - 1 / Nk)
    Bst = M / C
    M10 = M / 2.3026
    sp_sd = (sum(sqrt(x) for x in v) / k) ** 2    # 標準偏差の平均の2乗（誤り）
    M_sd = Nk * 2.1972 - (ni - 1) * sum(lnv)
    return {
        "A1": A[0], "A2": A[1], "A3": A[2], "A4": A[3], "A5": A[4],
        "B1": B[0], "B2": B[1], "B3": B[2], "B4": B[3], "B5": B[4],
        "D": s(D, 1), "DB": s(DB, 1),
        "k": k, "ni": ni, "v1": s(v[0], 1), "v2": s(v[1], 1), "v3": s(v[2], 1),
        "l1": s(lnv[0], 4), "l2": s(lnv[1], 4), "l3": s(lnv[2], 4), "lsum": s(sum(lnv), 4),
        "sp2": s(sp2, 3), "lnsp": s(lnsp, 4), "Nk": Nk, "df1": ni - 1,
        "Mt1": s(Nk * lnsp, 4), "Mt2": s((ni - 1) * sum(lnv), 4), "M": s(M, 4), "C": s(C, 4), "Cin": s(k / (ni - 1) - 1 / Nk, 4),
        "B": s(Bst, 2), "Bnc": s(M, 2), "B10": s(M10 / C, 2), "Bsd": s(M_sd / C, 2), "spsd": s(sp_sd, 1),
        "c2": "5.99", "c3": "7.81",
    }


def p07():
    b0, b1, g, d = 2.0, 0.50, 1.5, -0.30
    xs = -g / d
    return {
        "b0": s(b0, 1), "b1": s(b1, 2), "g": s(g, 1), "d": s(d, 2), "dabs": s(-d, 2),
        "slopeA": s(b1 + d, 2), "b0A": s(b0 + g, 1), "xs": s(xs, 1),
        "w1": s(g / b1, 1), "w2": s(g / (b1 + d), 1), "w3": s(-xs, 1),
        "x8": 8, "yA8": s(b0 + g + (b1 + d) * 8, 1), "yB8": s(b0 + b1 * 8, 1),
        # (b) Q-Q プロット
        "res": "−1.1, −0.9, −0.8, −0.6, −0.4, −0.3, −0.1, 0.4, 1.3, 2.5",
        "nq": 10, "p10": "0.95", "z95": "1.645", "z90": "1.282", "z05": "−1.645",
    }


def p08():
    # (a) オフセットつきポアソン回帰の逸脱度
    y = [12, 8, 20, 10]
    m = [10, 10, 16, 14]
    lr = [0.1823, -0.2231, 0.2231, -0.3365]   # ln(y/μ)
    terms = [yi * l for yi, l in zip(y, lr)]
    D = 2 * (sum(terms) - sum(yi - mi for yi, mi in zip(y, m)))
    X2 = sum((yi - mi) ** 2 / mi for yi, mi in zip(y, m))
    # (b) ファイ係数
    a, b, c, dd = 10, 190, 30, 170
    n = a + b + c + dd
    phi = (a * dd - b * c) / sqrt((a + b) * (c + dd) * (a + c) * (b + dd))
    return {
        "y1": y[0], "y2": y[1], "y3": y[2], "y4": y[3], "m1": m[0], "m2": m[1], "m3": m[2], "m4": m[3],
        "l1": s(lr[0], 4), "l2": s(lr[1], 4), "l3": s(lr[2], 4), "l4": s(lr[3], 4),
        "tt1": s(terms[0], 4), "tt2": s(terms[1], 4), "tt3": s(terms[2], 4), "tt4": s(terms[3], 4),
        "tsum": s(sum(terms), 4), "D": s(D, 2), "Dh": s(D / 2, 2), "X2": s(X2, 2),
        "a": a, "b": b, "c": c, "dd": dd, "n": n, "r1": a + b, "r2": c + dd, "c1": a + c, "c2": b + dd,
        "ad": a * dd, "bc": b * c, "adbc": a * dd - b * c, "prod": (a + b) * (c + dd) * (a + c) * (b + dd),
        "sq": s(sqrt((a + b) * (c + dd) * (a + c) * (b + dd)), 0),
        "phi": r(phi, 3), "phi_s": s(phi, 3), "phi4": s(phi, 4), "chi2": s(n * phi ** 2, 2), "phi2": s(phi ** 2, 4),
    }


def p09():
    pw, pk = 0.2, 0.1
    pdel = 1 - (1 - pw) * (1 - pk)
    pw_d = pw / pdel
    return {"pw": s(pw, 1), "pk": s(pk, 1), "pdel": s(pdel, 2), "pw_d": s(pw_d, 3), "pw_dk": s(pw, 3),
            "qw": s(1 - pw, 1), "qk": s(1 - pk, 1), "pw_add": s(pw / (pw + pk), 3)}


def p10():
    y = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    N, k = len(y), 3
    n = N // k
    means = [sum(y[i::k]) / n for i in range(k)]
    Y = sum(y) / N
    vsys = sum((m - Y) ** 2 for m in means) / k
    S2 = sum((v - Y) ** 2 for v in y) / (N - 1)
    vsrs = (1 - n / N) * S2 / n
    return {
        "N": N, "k": k, "n": n, "m1": s(means[0], 0), "m2": s(means[1], 0), "m3": s(means[2], 0), "Y": s(Y, 0),
        "vsys": r(vsys, 2), "vsys_s": s(vsys, 2), "vsys4": s(vsys, 4), "S2": s(S2, 1), "vsrs": s(vsrs, 2),
        "fpc": s(1 - n / N, 4), "ssd": s(sum((m - Y) ** 2 for m in means), 0), "vsys_k1": s(vsys * k / (k - 1), 2),
        "vsrs_nofpc": s(S2 / n, 2),
    }


def p11():
    # (a) 等相関の 3 指標、第 1 主成分による再構成
    z = (1.2, 0.3, -0.6)
    rr = 0.6
    sz = sum(z)
    score = sz / sqrt(3)
    recon = sz / 3
    res = [v - recon for v in z]
    rss = sum(v ** 2 for v in res)
    tot = sum(v ** 2 for v in z)
    res_un = [v - sz for v in z]
    # (b) 斜交因子モデル
    l1, l2, ph = 0.7, 0.2, 0.5
    h2 = l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * ph
    st = (l1 + l2 * ph, l1 * ph + l2)
    # (c) 非計量 MDS のストレス
    d = [1.0, 1.6, 1.4, 2.2, 2.8, 3.0]
    dh = [1.0, 1.5, 1.5, 2.2, 2.8, 3.0]
    num = sum((x - y) ** 2 for x, y in zip(d, dh))
    den = sum(x ** 2 for x in d)
    stress = sqrt(num / den)
    dbar = sum(d) / len(d)
    den2 = sum((x - dbar) ** 2 for x in d)
    return {
        "z1": s(z[0], 1), "z2": s(z[1], 1), "z3": s(z[2], 1), "rr": s(rr, 1), "ev1": s(1 + 2 * rr, 1),
        "sz": s(sz, 1), "score": s(score, 4), "recon": s(recon, 1), "e1": s(res[0], 1), "e2": s(res[1], 1),
        "e3": s(res[2], 1), "rss": s(rss, 2), "tot": s(tot, 2), "sc2": s(score ** 2, 2),
        "w_nosq": s(tot - score, 2), "w_un": s(sum(v ** 2 for v in res_un), 2),
        "pl1": s(l1, 1), "pl2": s(l2, 1), "ph": s(ph, 1), "h2": s(h2, 2), "h2o": s(l1 ** 2 + l2 ** 2, 2),
        "cross": s(2 * l1 * l2 * ph, 2), "st1": s(st[0], 2), "st2": s(st[1], 2), "h2st": s(st[0] ** 2 + st[1] ** 2, 2),
        "u2": s(1 - h2, 2), "ls": s(l1 * st[0] + l2 * st[1], 2),
        "d1": s(d[0], 1), "d2": s(d[1], 1), "d3": s(d[2], 1), "d4": s(d[3], 1), "d5": s(d[4], 1), "d6": s(d[5], 1),
        "dh2": s(dh[1], 1), "num": s(num, 2), "den": s(den, 2), "stress": r(stress, 3), "stress_s": s(stress, 3),
        "stress4": s(stress, 4), "ratio": s(num / den, 5), "st_nosq": s(num / den, 4),
        "st2b": s(sqrt(num / den2), 3), "den2": s(den2, 2), "dbar": s(dbar, 1),
    }


def p12():
    ph = 0.6
    ratio = ((1 + ph) / (1 - ph)) ** 2
    half = (1 + ph) ** 2 / (1 + ph ** 2)
    return {"ph": s(ph, 1), "opp": s(1 + ph, 1), "omp": s(1 - ph, 1), "ratio": s(ratio, 0),
            "nosq": s((1 + ph) / (1 - ph), 1), "inv": s(1 / ratio, 4), "half": s(half, 2), "opp2": s((1 + ph) ** 2, 2),
            "omp2": s((1 - ph) ** 2, 2), "d0": s(1 - 2 * ph + ph ** 2, 2), "dpi": s(1 + 2 * ph + ph ** 2, 2)}


def p13():
    blocks, na, nb = 4, 3, 2
    ss = {"blk": 18.0, "A": 36.0, "E1": 24.0, "B": 10.0, "AB": 4.0, "E2": 18.0}
    df = {"blk": blocks - 1, "A": na - 1, "E1": (blocks - 1) * (na - 1), "B": nb - 1, "AB": (na - 1) * (nb - 1)}
    df["E2"] = blocks * na * nb - 1 - sum(df.values())
    ms = {k: ss[k] / df[k] for k in ss}
    F = ms["A"] / ms["E1"]
    F2 = ms["A"] / ms["E2"]
    pooled = (ss["E1"] + ss["E2"]) / (df["E1"] + df["E2"])
    Fp = ms["A"] / pooled
    return {
        "blocks": blocks, "na": na, "nb": nb, "nt": blocks * na * nb,
        "sblk": s(ss["blk"], 0), "sA": s(ss["A"], 0), "sE1": s(ss["E1"], 0), "sB": s(ss["B"], 0), "sAB": s(ss["AB"], 0),
        "sE2": s(ss["E2"], 0), "dblk": df["blk"], "dA": df["A"], "dE1": df["E1"], "dB": df["B"], "dAB": df["AB"],
        "dE2": df["E2"], "mA": s(ms["A"], 1), "mE1": s(ms["E1"], 1), "mE2": s(ms["E2"], 1),
        "F": s(F, 2), "F2": s(F2, 2), "Fp": s(Fp, 2), "mpool": s(pooled, 1), "dpool": df["E1"] + df["E2"],
        "Fraw": s(ss["A"] / ss["E1"], 2), "dtot": blocks * na * nb - 1,
        "f26": "5.14", "f29": "4.26", "f215": "3.68",
    }


def values():
    return {"s22p01": p01(), "s22p02": p02(), "s22p03": p03(), "s22p04": p04(), "s22p05": p05(),
            "s22p06": p06(), "s22p07": p07(), "s22p08": p08(), "s22p09": p09(), "s22p10": p10(),
            "s22p11": p11(), "s22p12": p12(), "s22p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
