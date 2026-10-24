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
    pop = [0.30, 0.34, 0.36]
    obs = [336, 396, 468]
    exp = [n * p for p in pop]
    terms = [(o - e) ** 2 / e for o, e in zip(obs, exp)]
    chi = sum(terms)
    chi_o = sum((o - e) ** 2 / o for o, e in zip(obs, exp))
    chi_p = sum((o / n - p) ** 2 / p for o, p in zip(obs, pop))
    # (b) 回答が得られるまでの発信回数（負の二項分布）
    p = 0.15
    r_ = n
    ev = r_ / p
    var = r_ * (1 - p) / p ** 2
    return {
        "n": n, "q1": s(pop[0], 2), "q2": s(pop[1], 2), "q3": s(pop[2], 2),
        "o1": obs[0], "o2": obs[1], "o3": obs[2], "e1": int(exp[0]), "e2": int(exp[1]), "e3": int(exp[2]),
        "d1": int(obs[0] - exp[0]), "d2": int(obs[1] - exp[1]), "d3": int(obs[2] - exp[2]),
        "t1": s(terms[0], 4), "t2": s(terms[1], 4), "t3": s(terms[2], 4),
        "chi": s(chi, 2), "chi4": s(chi, 4), "chi_o": s(chi_o, 2), "chi_p": s(chi_p, 4),
        "p": p, "q": s(1 - p, 2), "p2": s(p ** 2, 4), "ev": s(ev, 0), "var1": s((1 - p) / p ** 2, 4),
        "var": s(var, 2), "sd": r(sqrt(var), 1), "sd4": s(sqrt(var), 4),
        "w_bin": s(sqrt(ev * p * (1 - p)), 1), "w_nop": s(sqrt(r_ * (1 - p) / p), 1), "w_noq": s(sqrt(r_ / p ** 2), 1),
    }


def p05():
    """3 地域の窓口処理日数の分散の比較（バートレット検定）"""
    k, ni = 3, 6
    s2 = [4.0, 9.0, 16.0]
    lns = {4.0: 1.3863, 9.0: 2.1972, 16.0: 2.7726}
    N = k * ni
    sp2 = sum((ni - 1) * v for v in s2) / (N - k)
    ln_sp2 = 2.2687                              # ln(9.667)
    sum_ln = sum(lns[v] for v in s2)
    M = (N - k) * ln_sp2 - (ni - 1) * sum_ln
    C = 1 + (1 / (3 * (k - 1))) * (k / (ni - 1) - 1 / (N - k))
    B = M / C
    # 誤答
    Mn = N * ln_sp2 - ni * sum_ln                 # 重みを n_i にした（対数の差は定数倍で消える）
    Cn = 1 + (1 / (3 * (k - 1))) * (k / ni - 1 / N)
    return {
        "k": k, "ni": ni, "s1": s(s2[0], 1), "s2": s(s2[1], 1), "s3": s(s2[2], 1),
        "l1": s(lns[4.0], 4), "l2": s(lns[9.0], 4), "l3": s(lns[16.0], 4), "lsp": s(ln_sp2, 4),
        "N": N, "dfe": N - k, "df1": ni - 1, "ssum": s(sum((ni - 1) * v for v in s2), 0), "sp2": s(sp2, 3),
        "suml": s(sum_ln, 4), "t1": s((N - k) * ln_sp2, 4), "t2": s((ni - 1) * sum_ln, 4), "M": s(M, 4),
        "cin": s(k / (ni - 1) - 1 / (N - k), 4), "C": s(C, 4), "B": s(B, 2), "B4": s(M / r(C, 4), 4),
        "w_noC": s(M, 2), "w_mulC": s(M * C, 2), "w_n": s(Mn / Cn, 2), "w_sd": s(M / 2 / C, 2),
    }


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
    """就業・失業・引退のマルコフ連鎖"""
    pEE, pEU, pER = 0.87, 0.05, 0.08
    pUE, pUU, pUR = 0.35, 0.50, 0.15
    # t_U = (1 + pUE t_E)/(1-pUU) = 2 + 0.7 t_E
    a0 = 1 / (1 - pUU)
    a1 = pUE / (1 - pUU)
    # (1 - pEE) t_E = 1 + pEU (a0 + a1 t_E)
    lhs = (1 - pEE) - pEU * a1
    rhs = 1 + pEU * a0
    tE = rhs / lhs
    tU = a0 + a1 * tE
    return {
        "pEE": pEE, "pEU": pEU, "pER": pER, "pUE": pUE, "pUU": pUU, "pUR": pUR,
        "a0": s(a0, 0), "a1": s(a1, 1), "omEE": s(1 - pEE, 2), "omUU": s(1 - pUU, 2),
        "pEUa0": s(pEU * a0, 2), "pEUa1": s(pEU * a1, 3), "rhs": s(rhs, 2), "lhs": s(lhs, 3),
        "tE": r(tE, 2), "tE4": s(tE, 4), "tU": s(tU, 2), "w_geo": s(1 / pER, 2),
    }


def p08():
    """市町村の合計特殊出生率の回帰"""
    # (a) 交互作用
    b1, b3 = 0.050, -0.030
    v1, v3, c13 = 0.00016, 0.00036, -0.00016
    slope = b1 + b3
    var = v1 + v3 + 2 * c13
    se = sqrt(var)
    # (b) DFBETAS（単回帰）
    n, dx, e, Sxx, s_del = 40, 2.5, 0.18, 50.0, 0.10
    h = 1 / n + dx ** 2 / Sxx
    dfb = dx * e / ((1 - h) * Sxx)
    scale = r(s_del / sqrt(Sxx), 6)
    dfbs = dfb / scale
    # (c) AIC と AICc
    ll = [12.0, 14.4, 16.3]
    K = [3, 5, 7]
    aic = [-2 * l + 2 * k for l, k in zip(ll, K)]
    corr = [2 * k * (k + 1) / (n - k - 1) for k in K]
    aicc = [a_ + c_ for a_, c_ in zip(aic, corr)]
    assert min(range(3), key=lambda i: aic[i]) == 1 and min(range(3), key=lambda i: aicc[i]) == 0
    d = {
        "b1": s(b1, 3), "b3": s(b3, 3), "v1": s(v1, 5), "v3": s(v3, 5), "c13": s(c13, 5), "c13x2": s(2 * c13, 5),
        "slope": s(slope, 3), "var": s(var, 5), "se": s(se, 4), "se6": s(se, 6),
        "w_nocov": s(sqrt(v1 + v3), 4), "w_plus": s(sqrt(v1 + v3 - 2 * c13), 4), "w_b3": s(sqrt(v3), 4),
        "se1": s(sqrt(v1), 4),
        "n": n, "dx": dx, "e": e, "Sxx": s(Sxx, 0), "s_del": s(s_del, 2), "dx2": s(dx ** 2, 2),
        "h": s(h, 3), "omh": s(1 - h, 3), "num": s(dx * e, 2), "den": s((1 - h) * Sxx, 1),
        "dfb": s(dfb, 6), "rtS": s(sqrt(Sxx), 4), "scale": s(scale, 6), "dfbs": r(dfbs, 2), "dfbs4": s(dfbs, 4),
        "cut": s(2 / sqrt(n), 3), "rtn": s(sqrt(n), 4),
        "w_h": s(dx * e / ((1 - 0.2) * Sxx) / scale, 2),
    }
    for i in range(3):
        d[f"l{i+1}"] = s(ll[i], 1)
        d[f"K{i+1}"] = K[i]
        d[f"aic{i+1}"] = s(aic[i], 1)
        d[f"cr{i+1}"] = s(corr[i], 4)
        d[f"aicc{i+1}"] = s(aicc[i], 4)
        d[f"nk{i+1}"] = n - K[i] - 1
        d[f"kk{i+1}"] = 2 * K[i] * (K[i] + 1)
    return d


def p09():
    """投票参加のロジスティック回帰と k 近傍法"""
    d1, d2 = 1012.6, 1001.9
    lr = d1 - d2
    tr = {"P1": (30, 2, "棄権"), "P2": (35, 10, "投票"), "P3": (50, 4, "棄権"),
          "P4": (62, 20, "投票"), "P5": (45, 15, "投票"), "P6": (28, 8, "棄権")}
    new = (40, 9)
    sa, sy = 15, 5
    dz = {k: ((v[0] - new[0]) / sa) ** 2 + ((v[1] - new[1]) / sy) ** 2 for k, v in tr.items()}
    draw = {k: (v[0] - new[0]) ** 2 + (v[1] - new[1]) ** 2 for k, v in tr.items()}
    nz = sorted(dz, key=dz.get)[:3]
    nr = sorted(draw, key=draw.get)[:3]
    assert nz == ["P2", "P6", "P3"] and nr == ["P2", "P5", "P3"], (nz, nr)
    out = {
        "d1": s(d1, 1), "d2": s(d2, 1), "lr": s(lr, 1), "lr_half": s(lr / 2, 2), "lr_dbl": s(2 * lr, 1),
        "c3_05": "7.815", "c3_01": "11.345", "c2_05": "5.991", "c2_01": "9.210",
        "sa": sa, "sy": sy, "na": new[0], "ny": new[1],
    }
    for k, v in tr.items():
        out[f"{k}a"] = v[0]
        out[f"{k}y"] = v[1]
        out[f"{k}l"] = v[2]
        out[f"{k}z"] = s(dz[k], 3)
        out[f"{k}r"] = int(draw[k])
        out[f"{k}da"] = s((v[0] - new[0]) / sa, 3)
        out[f"{k}dy"] = s((v[1] - new[1]) / sy, 1)
    return out


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
    """二段抽出"""
    N, n = 30, 3
    M = [12000, 5000, 8000]
    m = 100
    p = [0.12, 0.20, 0.15]
    My = [a * b for a, b in zip(M, p)]
    tot = N / n * sum(My)
    return {
        "N": N, "nn": n, "m": m, "M1": "12{,}000", "M2": "5{,}000", "M3": "8{,}000",
        "p1": s(p[0], 2), "p2": s(p[1], 2), "p3": s(p[2], 2), "k1": int(m * p[0]), "k2": int(m * p[1]), "k3": int(m * p[2]),
        "My1": s(My[0], 0), "My2": s(My[1], 0), "My3": s(My[2], 0), "sMy": s(sum(My), 0), "Nn": int(N / n),
        "tot": s(tot, 0), "tot_c": "36{,}400",
        "w_nofac": "3{,}640", "w_sample": s(N / n * sum(m * x for x in p), 0),
        "w_mean": "39{,}167", "w_non": "109{,}200",
        "pm": s(sum(p) / 3, 5), "Mm": s(sum(M) / 3, 1),
    }


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
              "cross": s(2 * l1 * l2 * phi, 3), "h2": s(h2, 3), "u2": s(1 - h2, 3), "orth": s(l1 ** 2 + l2 ** 2, 3),
              "w_once": s(l1 ** 2 + l2 ** 2 + l1 * l2 * phi, 3), "w_twice": s(l1 ** 2 + l2 ** 2 + 4 * l1 * l2 * phi, 3)})
    return d


def p13():
    """失業率のドリフト付きランダムウォーク"""
    mu, sig, uT, h = -0.02, 0.10, 2.60, 3
    rt3 = 1.7321
    fc = uT + h * mu
    half = 1.96 * sig * rt3
    up = fc + half
    return {
        "mu": s(mu, 2), "sig": s(sig, 2), "uT": s(uT, 2), "h": h, "rt3": s(rt3, 4), "fc": s(fc, 2),
        "sdh": s(sig * rt3, 5), "half": s(half, 5), "up": r(up, 2), "up5": s(up, 5),
        "w_h": s(fc + 1.96 * sig * h, 2), "w_1": s(fc + 1.96 * sig, 2), "w_var": s(fc + 1.96 * sig ** 2 * h, 2),
        "var_h": s(h * sig ** 2, 2),
    }


def p14():
    """町長選挙：有限母集団の信頼区間・確率化検定・ベイズ因子"""
    N, n, x = 5000, 800, 344
    ph = x / n
    pq = ph * (1 - ph)
    fpc = (N - n) / (N - 1)
    v = fpc * pq / n
    se = r(sqrt(v), 6)
    half = 1.96 * se
    se0 = r(sqrt(pq / n), 6)
    # (b) 確率化検定
    tail15, p14_ = 0.0207, 0.0370
    gam = (0.05 - tail15) / p14_
    # (c) ベイズ因子
    m1 = 1 / 21
    bf01 = p14_ / m1
    bf10 = 1 / bf01
    lr_mle = (0.7 ** 14 * 0.3 ** 6) / 0.5 ** 20
    return {
        "N": "5{,}000", "Nn": N, "n": n, "x": x, "ph": s(ph, 2), "pq": s(pq, 4), "fpcN": N - n, "fpcD": N - 1,
        "fpc": s(fpc, 6), "v0": s(pq / n, 9), "v": s(v, 9), "se": s(se, 6), "half": s(half, 4), "half6": s(half, 6),
        "w_nofpc": s(1.96 * se0, 4), "w_fpc2": s(1.96 * se0 * fpc, 4), "w_z": s(1.645 * se, 4), "w_se": s(se, 4),
        "se0": s(se0, 6),
        "n20": 20, "tail15": s(tail15, 4), "p14": s(p14_, 4), "num": s(0.05 - tail15, 4), "gam": r(gam, 2), "gam4": s(gam, 4),
        "tail14": s(tail15 + p14_, 4),
        "m1": s(m1, 6), "bf01": s(bf01, 3), "bf10": s(bf10, 2), "bf10_4": s(bf10, 4), "w_mle": s(lr_mle, 2),
        "w_post": s(bf10 / (1 + bf10), 2), "w_p": s(tail15 + p14_, 2), "x14": 14, "x6": 6,
    }


def values():
    return {"s13p01": p01(), "s13p02": p02(), "s13p03": p03(), "s13p04": p04(), "s13p05": p05(),
            "s13p06": p06(), "s13p07": p07(), "s13p08": p08(), "s13p09": p09(), "s13p10": p10(),
            "s13p11": p11(), "s13p12": p12(), "s13p13": p13(), "s13p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
