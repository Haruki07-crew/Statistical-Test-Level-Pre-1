"""模擬試験 第13回（set13）の数値。values() が {大問id: {名前: 値}} を返す。

題材：公共政策・社会調査・選挙・人口統計。
問題文に示した分布の値（χ²・F・t・q の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
途中式に表示する値は、表示した桁の値から次の段階を計算する（表示と計算を一致させる）。
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


def p01():
    """ランダム化回答法"""
    pi0 = 0.15
    py_t = 5 / 6          # 該当者が「はい」と答える確率（正直 4/6 + 強制 1/6）
    py_n = 1 / 6          # 非該当者が「はい」と答える確率（強制 1/6）
    num = pi0 * py_t                      # 0.125
    den_n = (1 - pi0) * py_n              # 0.141666…
    py = num + den_n
    post = num / py
    honest_only = pi0 * 4 / 6 / py
    # (b)
    n, y = 600, 186
    ph = y / n
    pih = 1.5 * ph - 0.25
    pq = ph * (1 - ph)
    v1 = pq / n
    se_p = sqrt(v1)
    se = 1.5 * se_p
    se_direct = sqrt(pih * (1 - pih) / n)
    se_no15 = se_p
    se_15var = sqrt(1.5 * v1)
    return {
        "pi0": pi0, "num": s(num, 3), "den_n6": s((1 - pi0), 2), "py": s(py, 5), "den_n": s(den_n, 5),
        "num6": s(num * 6, 2), "py6": s(py * 6, 2),
        "post": s(post, 3), "post5": s(post, 5), "c_ty": s(py_t, 3), "c_honest": s(honest_only, 3),
        "c_prior": s(pi0, 3), "c_py": s(py, 3),
        "n": n, "y": y, "ph": s(ph, 2), "pih": s(pih, 3), "pq": s(pq, 4), "v1": s(v1, 7),
        "se_p": s(se_p, 6), "se": r(se, 4), "se5": s(se, 6),
        "se_direct": s(se_direct, 4), "se_no15": s(se_no15, 4), "se_15var": s(se_15var, 4),
        "ratio": s(se / se_direct, 2),
    }


def p02():
    """期日前投票所の到着（ポアソン到着の 1 人目 X と 2 人目 Y）"""
    lam, y0, x0 = 0.2, 10, 3
    em06, em2 = 0.5488, 0.1353          # e^{-0.6}, e^{-2}
    ans = x0 / y0
    unc = 1 - em06
    trunc = (1 - em06) / (1 - em2)
    k3 = 2 / lam ** 3                   # 指数分布の 3 次中心モーメント
    v1 = 1 / lam ** 2
    k3y = 2 * k3
    vy = 2 * v1
    sd3 = vy ** 1.5
    skew = k3y / sd3
    return {
        "lam": lam, "y0": y0, "x0": x0, "em06": s(em06, 4), "em2": s(em2, 4), "mean": int(1 / lam),
        "a_ok": s(ans, 3), "a_unc": s(unc, 3), "a_tr": s(trunc, 3), "a_sq": s(ans ** 2, 3), "a_c": s(1 - ans, 3),
        "unc4": s(unc, 4), "omem2": s(1 - em2, 4),
        "k3": round(k3), "v1": round(v1), "k3y": round(k3y), "vy": round(vy), "rt2": "1.4142",
        "sd3": s(50 * 7.0711, 2),
        "rt50": "7.0711", "skew": s(k3y / (50 * 7.0711), 3), "skew5": s(k3y / (50 * 7.0711), 4),
        "w_exp": s(2.0, 3), "w_add": s(4.0, 3), "w_var2": s(k3y / vy ** 2, 3), "w_one": s(k3 / (50 * 7.0711), 3),
    }


def p03():
    """(a) 円形の市域の住民の距離 (b) パレート分布の所得の最大値"""
    R = 6
    med = R / sqrt(2)
    # (b)
    xm, a, n, x = 200, 3, 4000, 4000
    ratio = xm / x
    tail = ratio ** a
    lam_ = n * tail
    em05, em0025 = 0.6065, 0.9753     # e^{-0.5}, e^{-0.025}
    return {
        "R": R, "R2": R * R, "R22": s(R * R / 2, 0), "med": s(med, 2), "med4": s(med, 4),
        "w_half": s(R / 2, 2), "w_mean": s(2 * R / 3, 2), "w_cube": s(R / 2 ** (1 / 3), 2),
        "w_rev": s(R * (1 - 1 / sqrt(2)), 2),
        "xm": xm, "a": a, "n": n, "x": x, "ratio": s(ratio, 2), "tail": s(tail, 6), "lam": s(lam_, 1),
        "em05": s(em05, 4), "em0025": s(em0025, 4),
        "b_ok": s(em05, 3), "b_union": s(1 - lam_, 3), "b_comp": s(1 - em05, 3),
        "b_dens": s(em0025, 3), "b_one": s(1 - tail, 3), "tail4": s(ratio ** (a + 1), 7),
        "lam4": s(n * ratio ** (a + 1), 3),
    }


def p04():
    """選挙前の世論調査"""
    n = 1200
    nA, nB, nU = 516, 456, 228
    pA, pB = nA / n, nB / n
    vS = (pA + pB) / n                       # H0 のもとの V[p̂A − p̂B]（p_A+p_B を推定値で置換）
    se = sqrt(vS)
    z = (pA - pB) / se
    se_ind = sqrt((pA * (1 - pA) + pB * (1 - pB)) / n)          # 共分散を無視
    se_sign = sqrt((pA * (1 - pA) + pB * (1 - pB) - 2 * pA * pB) / n)   # 共分散の符号を誤る
    nAB = nA + nB
    # (b) 回答が得られるまでの発信回数（負の二項分布）
    p = 0.15
    r_ = n
    ev = r_ / p
    var = r_ * (1 - p) / p ** 2
    return {
        "n": n, "nA": nA, "nB": nB, "nU": nU, "pA": s(pA, 2), "pB": s(pB, 2), "pS": s(pA + pB, 2),
        "vS": s(vS, 6), "se": s(se, 6), "dp": s(pA - pB, 2), "z": s(z, 2), "z4": s(z, 4),
        "se_ind": s(se_ind, 6), "z_ind": s((pA - pB) / se_ind, 2), "z_sign": s((pA - pB) / se_sign, 2),
        "nAB": nAB, "nAmh": s(nA - nAB / 2, 0), "sdc": s(sqrt(nAB / 4), 4),
        "p": p, "q": s(1 - p, 2), "p2": s(p ** 2, 4), "ev": s(ev, 0), "var1": s((1 - p) / p ** 2, 4),
        "var": s(var, 2), "sd": r(sqrt(var), 1), "sd4": s(sqrt(var), 4),
        "w_bin": s(sqrt(ev * p * (1 - p)), 1), "w_nop": s(sqrt(r_ * (1 - p) / p), 1), "w_noq": s(sqrt(r_ / p ** 2), 1),
        **p04c(),
    }


def p04c():
    """出口調査の当確判定ルールの第1種・第2種の過誤"""
    m, c, t0, t1 = 400, 0.55, 0.50, 0.58
    phi = {1.20: 0.8849, 1.22: 0.8888, 2.00: 0.9772}
    se0 = sqrt(t0 * (1 - t0) / m)
    z0 = r((c - t0) / se0, 2)
    alpha = 1 - phi[z0]
    v1 = t1 * (1 - t1) / m
    se1 = r(sqrt(v1), 6)
    z1raw = (c - t1) / se1
    z1 = r(abs(z1raw), 2)
    beta = 1 - phi[z1]
    zw = r(abs(c - t1) / se0, 2)
    return {
        "m": m, "cc": s(c, 2), "th0": s(t0, 2), "th1": s(t1, 2), "ph120": s(phi[1.20], 4), "ph122": s(phi[1.22], 4),
        "ph200": s(phi[2.00], 4), "se0": s(se0, 3), "z0": s(z0, 2), "alpha": s(alpha, 4), "v1": s(v1, 6),
        "se1": s(se1, 6), "z1raw": s(z1raw, 4), "z1": s(z1, 2), "beta": s(beta, 4), "power": s(phi[z1], 4),
        "a2": s(2 * alpha, 4), "bw": s(1 - phi[zw], 4), "a_rev": s(phi[z0], 4), "dz": s(c - t1, 2), "dc": s(c - t0, 2),
        "t1q": s(1 - t1, 2),
    }


def p05():
    """2 局の処理日数の比較と予備検定"""
    return {"n1": 8, "n2": 12, "m1": "21.5", "m2": "18.0", "s1": "6.8", "s2": "3.9", "pF": "0.09"}


def p06():
    """ケース・コントロール型の調査"""
    a, b = 160, 240     # 投票者：受領・非受領
    c, d = 100, 300     # 棄権者：受領・非受領
    OR = a * d / (b * c)
    naive_rate = a / (a + c)
    naive_rate0 = b / (b + d)
    return {
        "a": a, "b": b, "c": c, "d": d, "n1": a + b, "n0": c + d, "OR": s(OR, 1),
        "nr1": s(naive_rate, 3), "nr0": s(naive_rate0, 3), "nrr": s(naive_rate / naive_rate0, 2),
        "o1": s(a / b, 4), "o0": s(c / d, 4), "o1b": s(a / c, 1), "o0b": s(b / d, 1), "ac": a + c, "bd": b + d,
    }


def p07():
    """住まいの状態の推移（状態の分類）"""
    P = [[0.90, 0.04, 0.03, 0.03], [0.10, 0.85, 0.02, 0.03], [0, 0, 0.80, 0.20], [0, 0, 0, 1]]
    d = {}
    for i in range(4):
        for j in range(4):
            d[f"p{i+1}{j+1}"] = s(P[i][j], 2) if P[i][j] else "0"
    d["p44"] = "1"
    d["out12"] = s(P[0][2] + P[0][3], 2)
    d["out22"] = s(P[1][2] + P[1][3], 2)
    return d


def p08():
    """市町村の合計特殊出生率の回帰"""
    b1, b3 = 0.050, -0.030
    v1, v3, c13 = 0.00016, 0.00036, -0.00016
    slope = b1 + b3
    var = v1 + v3 + 2 * c13
    se = sqrt(var)
    n = 40
    q = 0.9875
    return {
        "b1": s(b1, 3), "b3": s(b3, 3), "v1": s(v1, 5), "v3": s(v3, 5), "c13": s(c13, 5), "c13x2": s(2 * c13, 5),
        "slope": s(slope, 3), "var": s(var, 5), "se": s(se, 4), "se6": s(se, 6), "tval": s(slope / se, 2),
        "w_nocov": s(sqrt(v1 + v3), 4), "w_plus": s(sqrt(v1 + v3 - 2 * c13), 4), "w_b3": s(sqrt(v3), 4),
        "se1": s(sqrt(v1), 4),
        "n": n, "hsum": "4.62", "hk": r(5 - 4.62, 2), "hk_s": s(5 - 4.62, 2), "hbar": s(5 / n, 3), "hcut": s(2 * 5 / n, 2),
        **p08c(),
    }


def p08c():
    """訓練誤差と予測誤差（楽観度）"""
    n, p, rss = 40, 5, 1.80
    tr = rss / n
    s2 = rss / (n - p)
    opt = 2 * p * s2 / n
    return {"rss": s(rss, 2), "pk": p, "tr": s(tr, 3), "s2": s(s2, 6), "dfe": n - p, "opt": s(opt, 6),
            "pe": r(tr + opt, 4), "pe6": s(tr + opt, 6), "w_noopt": s(tr, 4), "w_half": s(tr + opt / 2, 4),
            "w_s2": s(s2, 4), "w_p4": s(tr + 2 * 4 * s2 / n, 4), "w_ntr": s(tr * (n + p) / (n - p), 4)}


def p09():
    """投票参加のロジスティック回帰とナイーブベイズ"""
    d1, d2 = 1012.6, 1001.9
    lr = d1 - d2
    prior = 0.6
    pj = {"v": (0.6, 0.8, 0.4), "a": (0.2, 0.3, 0.25)}   # P(加入|・), P(前回投票|・), P(60歳以上|・)
    fv = (1 - pj["v"][0]) * pj["v"][1] * pj["v"][2]
    fa = (1 - pj["a"][0]) * pj["a"][1] * pj["a"][2]
    sv, sa = prior * fv, (1 - prior) * fa
    post = sv / (sv + sa)
    wy_v = prior * pj["v"][0] * pj["v"][1] * pj["v"][2]
    wy_a = (1 - prior) * pj["a"][0] * pj["a"][1] * pj["a"][2]
    return {
        "d1": s(d1, 1), "d2": s(d2, 1), "lr": s(lr, 1), "lr_half": s(lr / 2, 2), "lr_dbl": s(2 * lr, 1),
        "c3_05": "7.815", "c3_01": "11.345", "c2_05": "5.991", "c2_01": "9.210",
        "pr": s(prior, 1), "pr0": s(1 - prior, 1),
        "v1": s(pj["v"][0], 2), "v2": s(pj["v"][1], 2), "v3": s(pj["v"][2], 2),
        "a1": s(pj["a"][0], 2), "a2": s(pj["a"][1], 2), "a3": s(pj["a"][2], 2),
        "nv1": s(1 - pj["v"][0], 2), "na1": s(1 - pj["a"][0], 2),
        "sv": s(sv, 4), "sa": s(sa, 4), "ssum": s(sv + sa, 4), "post": s(post, 3), "post5": s(post, 5),
        "w_yes": s(wy_v / (wy_v + wy_a), 3), "w_noprior": s(fv / (fv + fa), 3), "w_score": s(sv, 3),
        "w_odds": s(sv / sa, 3),
    }


def p10():
    """職業訓練プログラムの評価"""
    m = {"A": 3.2, "B": 5.2, "C": 6.1}
    n, mse, dfe = 10, 4.5, 27
    t27, t9 = 2.052, 2.262
    L = m["C"] - (m["A"] + m["B"]) / 2
    c2 = 1 + 0.25 + 0.25
    se = r(sqrt(mse * c2 / n), 4)
    t = L / se
    se3 = r(sqrt(mse * 3 / n), 4)
    se1 = r(sqrt(mse / n), 4)
    seabs = r(sqrt(mse * 2 / n), 4)
    # (b) 加重最小二乗
    xs = [2, 4, 6, 8]
    ys = [2.0, 3.6, 3.8, 6.0]
    ns = [10, 40, 30, 20]
    W = sum(ns)
    xw = sum(a * b for a, b in zip(ns, xs)) / W
    yw = sum(a * b for a, b in zip(ns, ys)) / W
    sxy = sum(w * (x - xw) * (y - yw) for w, x, y in zip(ns, xs, ys))
    sxx = sum(w * (x - xw) ** 2 for w, x in zip(ns, xs))
    bw = sxy / sxx
    xb, yb = sum(xs) / 4, sum(ys) / 4
    bo = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
    bwrong = sum(w * (x - xb) * (y - yb) for w, x, y in zip(ns, xs, ys)) / sum(w * (x - xb) ** 2 for w, x in zip(ns, xs))
    d = {
        "mA": m["A"], "mB": m["B"], "mC": m["C"], "n": n, "mse": mse, "dfe": dfe, "t27": s(t27, 3), "t9": s(t9, 3),
        "L": s(L, 1), "mAB": s((m["A"] + m["B"]) / 2, 1), "c2": s(c2, 1), "inner": s(mse * c2 / n, 3), "se": s(se, 4),
        "t": s(t, 2), "t4": s(t, 4), "w3": s(L / se3, 2), "w1": s(L / se1, 2), "wabs": s(L / seabs, 2),
        "xw": s(xw, 1), "yw": s(yw, 2), "sxy": s(sxy, 2), "sxx": s(sxx, 1), "bw": r(bw, 3), "bw5": s(bw, 5),
        "bo": s(bo, 3), "bwrong": s(bwrong, 3), "W": W,
    }
    for i in range(4):
        d[f"x{i+1}"] = xs[i]
        d[f"y{i+1}"] = s(ys[i], 1)
        d[f"n{i+1}"] = ns[i]
        d[f"dx{i+1}"] = s(xs[i] - xw, 1)
        d[f"dy{i+1}"] = s(ys[i] - yw, 2)
        d[f"pxy{i+1}"] = s(ns[i] * (xs[i] - xw) * (ys[i] - yw), 2)
        d[f"pxx{i+1}"] = s(ns[i] * (xs[i] - xw) ** 2, 1)
    d["nx"] = s(sum(a * b for a, b in zip(ns, xs)), 0)
    d["ny"] = s(sum(a * b for a, b in zip(ns, ys)), 0)
    return d


def p11():
    """無回答者の追跡調査（ハンセン・ハーヴィッツ法）"""
    n, nr = 1000, 600
    nm = n - nr
    f_ = 100
    yr, ym = 0.70, 0.40
    est = nr / n * yr + nm / n * ym
    pooled = (nr * yr + f_ * ym) / (nr + f_)
    return {"n": n, "nr": nr, "nm": nm, "f": f_, "yr": s(yr, 2), "ym": s(ym, 2), "wr": s(nr / n, 1), "wm": s(nm / n, 1),
            "t1": s(nr / n * yr, 2), "t2": s(nm / n * ym, 2), "est": s(est, 3), "w_resp": s(yr, 3), "w_pool": s(pooled, 3),
            "w_avg": s((yr + ym) / 2, 3), "w_fu": s(ym, 3), "bias": s(yr - est, 2), "k": int(nm / f_),
            "c1": int(nr * yr), "c2": int(f_ * ym)}


def p12():
    """地域指標の多変量解析"""
    D = {("A", "B"): 2.0, ("A", "C"): 6.0, ("A", "D"): 10.0, ("A", "E"): 9.0, ("B", "C"): 5.0,
         ("B", "D"): 9.0, ("B", "E"): 8.0, ("C", "D"): 4.0, ("C", "E"): 7.0, ("D", "E"): 3.0}
    d = {f"d{a}{b}": s(v, 1) for (a, b), v in D.items()}
    avg = sum(D[(a, b)] for a in "AB" for b in "CDE") / 6
    d.update({"h_single": s(5.0, 1), "h_comp": s(10.0, 1), "h_avg": s(avg, 2)})
    # (b) 斜交因子の共通性
    l1, l2, phi = 0.60, 0.30, 0.40
    h2 = l1 ** 2 + l2 ** 2 + 2 * l1 * l2 * phi
    d.update({"l1": s(l1, 2), "l2": s(l2, 2), "phi": s(phi, 2), "l1s": s(l1 ** 2, 2), "l2s": s(l2 ** 2, 2),
              "cross": s(2 * l1 * l2 * phi, 3), "h2": s(h2, 3), "h2n": r(h2, 3), "u2": s(1 - h2, 3), "orth": s(l1 ** 2 + l2 ** 2, 3),
              "w_once": s(l1 ** 2 + l2 ** 2 + l1 * l2 * phi, 3), "w_twice": s(l1 ** 2 + l2 ** 2 + 4 * l1 * l2 * phi, 3)})
    return d


def p13():
    """失業率の前月差の AR(1) のスペクトル"""
    phi = 0.6
    num, den = (1 + phi) ** 2, (1 - phi) ** 2
    return {"phi": s(phi, 1), "d0": s(1 - 2 * phi + phi ** 2, 2), "dpi": s(1 + 2 * phi + phi ** 2, 2),
            "ratio": s(num / den, 0), "w_nosq": s((1 + phi) / (1 - phi), 0), "inv": "1/16"}


def p14():
    """町長選挙：有限母集団の信頼区間・経験ベイズ・KL 情報量"""
    N, n, x = 5000, 800, 344
    ph = x / n
    pq = ph * (1 - ph)
    fpc = (N - n) / (N - 1)
    v = fpc * pq / n
    se = r(sqrt(v), 6)
    half = 1.96 * se
    se0 = r(sqrt(pq / n), 6)
    # (b) 経験ベイズ
    ys = [0.42, 0.50, 0.38, 0.55, 0.40]
    s2 = 0.0025
    yb = sum(ys) / 5
    ss = sum((y - yb) ** 2 for y in ys)
    v4 = ss / 4
    tau2 = v4 - s2
    B = s2 / (s2 + tau2)
    est = yb + (1 - B) * (ys[3] - yb)
    v5 = ss / 5
    B5 = s2 / v5
    # (c) KL
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.4, 0.2]
    ln125, ln075 = 0.22314, -0.28768
    kl = p[0] * ln125 + p[1] * ln075
    klr = q[0] * (-ln125 + 0.0) + q[1] * (-ln075)   # Σ q ln(q/p)：ln(0.8)=−ln1.25, ln(4/3)=−ln0.75
    return {
        "N": "5{,}000", "Nn": N, "n": n, "x": x, "ph": s(ph, 2), "pq": s(pq, 4), "fpcN": N - n, "fpcD": N - 1,
        "fpc": s(fpc, 6), "v0": s(pq / n, 9), "v": s(v, 9), "se": s(se, 6), "half": s(half, 4), "half6": s(half, 6),
        "w_nofpc": s(1.96 * se0, 4), "w_fpc2": s(1.96 * se0 * fpc, 4), "w_z": s(1.645 * se, 4), "w_se": s(se, 4),
        "se0": s(se0, 6),
        "y1": s(ys[0], 2), "y2": s(ys[1], 2), "y3": s(ys[2], 2), "y4": s(ys[3], 2), "y5": s(ys[4], 2),
        "s2": s(s2, 4), "sd": s(sqrt(s2), 2), "yb": s(yb, 2), "ss": s(ss, 4), "v4": s(v4, 4), "tau2": s(tau2, 4),
        "B": s(B, 5), "omB": s(1 - B, 5), "dev4": s(ys[3] - yb, 2), "est": r(est, 3), "est5": s(est, 5),
        "w_v5": s(yb + (1 - B5) * (ys[3] - yb), 3), "w_swap": s(yb + B * (ys[3] - yb), 3), "w_noshr": s(ys[3], 3),
        "B5": s(B5, 5), "v5": s(v5, 5),
        "p1": s(p[0], 1), "p2": s(p[1], 1), "p3": s(p[2], 1), "q1": s(q[0], 1), "q2": s(q[1], 1), "q3": s(q[2], 1),
        "ln125": s(ln125, 5), "ln075": s(ln075, 5), "t1": s(p[0] * ln125, 6), "t2": s(p[1] * ln075, 6),
        "kl": r(kl, 4), "kl6": s(kl, 6), "klr": s(klr, 4), "w_nolog": s(sum(a * (a / b) for a, b in zip(p, q)) - 1, 4),
    }


def values():
    return {"s13p01": p01(), "s13p02": p02(), "s13p03": p03(), "s13p04": p04(), "s13p05": p05(),
            "s13p06": p06(), "s13p07": p07(), "s13p08": p08(), "s13p09": p09(), "s13p10": p10(),
            "s13p11": p11(), "s13p12": p12(), "s13p13": p13(), "s13p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
