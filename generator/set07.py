"""模擬試験 第7回（set07）の数値。values() が {大問id: {名前: 値}} を返す。題材は金融・保険・経済指標・不動産。

問題文に示した値（e^x、べき乗根、正規分布表・q 表・t 表の点、固有ベクトルの丸めた成分など）は、
ここでも同じ丸めた値を使って答えを計算する（一般電卓で問題文の値だけから同じ答えになるように）。
"""
from decimal import Decimal, ROUND_HALF_UP
from math import comb, sqrt


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
    # (a) 少なくとも1件が不正という条件のもとで両方不正の確率
    p1, p2 = 0.10, 0.20
    both = p1 * p2
    atl = 1 - (1 - p1) * (1 - p2)
    cond = both / atl
    # (b) 積率母関数 M(t) = w0 + w1 (1 - th t)^(-k) の分散
    w0, w1, th, k = 0.8, 0.2, 4, 2
    m1 = w1 * k * th                       # M'(0)
    m2 = w1 * k * (k + 1) * th ** 2        # M''(0)
    var = m2 - m1 ** 2
    cb = [s(var, 2), s(m2, 2), s(k * th ** 2, 2), s(w1 * k * th ** 2, 2), s(m2 - m1, 2)]
    distinct(*cb)
    # (c) 中心極限定理による保険料
    n, z = 2500, 2.326
    sdv = sqrt(var)
    prem = m1 + z * sdv / sqrt(n)
    return {
        "p1": s(p1, 2), "p2": s(p2, 2), "q1": s(1 - p1, 2), "q2": s(1 - p2, 2), "both": s(both, 2),
        "none": s((1 - p1) * (1 - p2), 2), "atl": s(atl, 2), "cond": r(cond, 3), "cond4": s(cond, 4),
        "cond1": s(p1 * p2 / p1, 2), "both_s": s(both, 3),
        "w0": w0, "w1": w1, "th": th, "k": k, "k1": k + 1, "m1": s(m1, 1), "m2": s(m2, 1), "m1sq": s(m1 ** 2, 2),
        "a1": s(w1 * k * th, 1), "a2": s(w1 * k * (k + 1) * th ** 2, 1),
        "var": cb[0], "v_m2": cb[1], "v_gam": cb[2], "v_cond": cb[3], "v_m2m": cb[4],
        "gmean": k * th, "gvar": k * th ** 2,
        "n": n, "z": z, "sdv": s(sdv, 4), "sdn": s(sdv / sqrt(n), 5), "sqn": s(sqrt(n), 0),
        "load": s(z * sdv / sqrt(n), 4), "prem": r(prem, 2), "prem4": s(prem, 4),
        "w_var": s(m1 + z * var / sqrt(n), 2), "w_1645": s(m1 + 1.645 * sdv / sqrt(n), 2),
        "w_n": s(m1 + z * sdv / n, 2), "tot_sd": s(sdv * sqrt(n), 2), "tot_mean": s(m1 * n, 0),
    }


def p02():
    # (b) パレート分布（θ=1, α=3）の 99% VaR と期待ショートフォール
    alpha, theta = 3, 1
    cube = 4.642  # 100^(1/3)（問題文の値）
    var99 = theta * cube
    es = alpha / (alpha - 1) * var99
    mean = alpha * theta / (alpha - 1)
    return {"alpha": alpha, "theta": theta, "cube": s(cube, 3), "var99": s(var99, 3),
            "es": r(es, 2), "es4": s(es, 4), "fac": s(alpha / (alpha - 1), 1), "mean": s(mean, 1),
            "w_mem": s(var99 + mean, 2), "a1": alpha - 1}


def p03():
    # (a) ランダム化検定：X ~ Bin(12, 1/2)、X>=10 で棄却、X=9 で確率 γ で棄却
    n = 12
    tot = 2 ** n
    t10 = sum(comb(n, k) for k in range(10, n + 1))
    c9 = comb(n, 9)
    target = 0.05 * tot
    gam = (target - t10) / c9
    # (b) クラスカル・ウォリス
    A = [-3.2, -1.5, 1.1, 2.8]
    Bv = [0.4, 2.3, 5.2, 6.0]
    C = [3.6, 4.5, 7.4, 8.9]
    allv = sorted(A + Bv + C)
    rank = {v: i + 1 for i, v in enumerate(allv)}
    R = [sum(rank[v] for v in g) for g in (A, Bv, C)]
    N = 12
    Ssq = sum(x * x / 4 for x in R)
    Hs = 12 / (N * (N + 1)) * Ssq - 3 * (N + 1)
    Hw = 12 / (N * (N + 1)) * Ssq - 3 * N
    # (c) 縮小推定量 c X̄ の MSE
    mu, sig, m = 0.5, 4.0, 36
    v = sig ** 2 / m
    mse = lambda c: c * c * v + (1 - c) ** 2 * mu ** 2
    cs = mu ** 2 / (mu ** 2 + v)
    cw = v / (mu ** 2 + v)
    cbad = mu ** 2 / (mu ** 2 + sig ** 2)
    ch = [(1, mse(1)), (cs, mse(cs)), (cw, mse(cw)), (cs, cs * cs * v), (cbad, mse(cbad))]
    cc = [f"$c={s(a, 3)}$、そのときの MSE は ${s(b, 4)}$" for a, b in ch]
    distinct(*cc)
    d = {"n": n, "tot": tot, "t10": t10, "c9": c9, "target": s(target, 1), "gam": r(gam, 3), "gam4": s(gam, 4),
         "a10": s(t10 / tot, 4), "a9": s((t10 + c9) / tot, 4), "num": s(target - t10, 1),
         "R1": R[0], "R2": R[1], "R3": R[2], "Ssq": s(Ssq, 1), "H": s(Hs, 2), "Hw": s(Hw, 2), "H4": s(Hs, 4),
         "mu": s(mu, 1), "sig": s(sig, 1), "m": m, "v": s(v, 4), "mu2": s(mu ** 2, 2), "den": s(mu ** 2 + v, 4),
         "cs": s(cs, 3), "mse_s": s(mse(cs), 4), "mse1": s(mse(1), 4), "ratio": s(mse(cs) / mse(1), 2),
         "c_one": cc[0], "c_ok": cc[1], "c_swap": cc[2], "c_nobias": cc[3], "c_sig": cc[4],
         "cw": s(cw, 3), "cbad": s(cbad, 3)}
    for i, v_ in enumerate(A):
        d[f"a{i}"] = v_
    for i, v_ in enumerate(Bv):
        d[f"b{i}"] = v_
    for i, v_ in enumerate(C):
        d[f"c{i}"] = v_
    return d


def p04():
    # ウィルソンの区間
    n, x, z = 50, 2, 1.96
    ph = x / n
    def wilson(z):
        cen = (x + z * z / 2) / (n + z * z)
        hw = z / (n + z * z) * sqrt(n * ph * (1 - ph) + z * z / 4)
        return cen, hw
    cen, hw = wilson(z)
    cen90, hw90 = wilson(1.645)
    se = sqrt(ph * (1 - ph) / n)
    iv = lambda a, b: f"({s(a, 3)},\\ {s(b, 3)})"
    c = [iv(ph - z * se, ph + z * se), iv(cen - hw, cen + hw), iv(0, ph + z * se), iv(cen90 - hw90, cen90 + hw90),
         iv(ph - hw, ph + hw)]
    distinct(*c)
    return {"n": n, "x": x, "ph": s(ph, 2), "z2": s(z * z, 4), "z2h": s(z * z / 2, 4), "nz2": s(n + z * z, 4),
            "xz": s(x + z * z / 2, 4), "cen": s(cen, 4), "inner": s(n * ph * (1 - ph) + z * z / 4, 4),
            "npq": s(n * ph * (1 - ph), 2), "z24": s(z * z / 4, 4),
            "sq": s(sqrt(n * ph * (1 - ph) + z * z / 4), 4), "hw": s(hw, 4), "lo": s(cen - hw, 4), "hi": s(cen + hw, 4),
            "se": s(se, 4), "wald_hw": s(z * se, 4),
            "ci_wald": c[0], "ci_ok": c[1], "ci_clip": c[2], "ci_90": c[3], "ci_mix": c[4]}


def p05():
    # (a) フィッシャーの正確検定（片側）
    n1, d1, n2, d2 = 6, 4, 8, 1
    N, K = n1 + n2, d1 + d2
    tot = comb(N, n1)
    ps = {k: comb(K, k) * comb(N - K, n1 - k) for k in range(0, min(K, n1) + 1)}
    pval = sum(v for k, v in ps.items() if k >= d1) / tot
    # (b) 3元分割表の対数線形モデルの期待度数
    t = {("M", "S", "D"): 12, ("M", "S", "N"): 48, ("M", "L", "D"): 3, ("M", "L", "N"): 37,
         ("V", "S", "D"): 10, ("V", "S", "N"): 70, ("V", "L", "D"): 2, ("V", "L", "N"): 18}
    tt = sum(t.values())
    nAB = t[("M", "S", "D")] + t[("M", "S", "N")]
    nAC = t[("M", "S", "D")] + t[("M", "L", "D")]
    nA = sum(v for k, v in t.items() if k[0] == "M")
    nB = sum(v for k, v in t.items() if k[1] == "S")
    nC = sum(v for k, v in t.items() if k[2] == "D")
    e_ok = nAB * nAC / nA
    e_mut = nA * nB * nC / tt ** 2
    e_abc = nAB * nC / tt
    e_acb = nAC * nB / tt
    c = [s(e_mut, 2), s(e_abc, 2), s(e_ok, 2), s(e_acb, 2), s(t[("M", "S", "D")], 2)]
    distinct(*c)
    d = {"n1": n1, "d1": d1, "n2": n2, "d2": d2, "N": N, "K": K, "tot": tot, "nd1": n1 - d1, "nd2": n2 - d2,
         "NK": N - K, "p4c": ps[4], "p5c": ps[5], "psum": ps[4] + ps[5], "pval": r(pval, 3), "pval4": s(pval, 4),
         "p4": s(ps[4] / tot, 4), "c54": comb(K, 4), "c92": comb(N - K, 2), "c91": comb(N - K, 1),
         "p2s": s(sum(v for k, v in ps.items() if v <= ps[d1]) / tot, 4),
         "e_ok": c[2], "e_mut": c[0], "e_abc": c[1], "e_acb": c[3], "e_obs": c[4],
         "nAB": nAB, "nAC": nAC, "nA": nA, "nB": nB, "nC": nC, "tt": tt}
    for k, v in t.items():
        d["t" + "".join(k)] = v
    return d


def p06():
    # (a) コルモゴロフ・スミルノフ
    xs = [-1.2, -0.4, 0.1, 0.5, 1.9]
    F = [0.1151, 0.3446, 0.5398, 0.6915, 0.9713]
    n = len(xs)
    dp = [(i + 1) / n - F[i] for i in range(n)]
    dm = [F[i] - i / n for i in range(n)]
    D = max(max(dp), max(dm))
    d = {"D": s(D, 3), "Dp": s(max(dp), 3), "Dm": s(max(dm), 3), "cv": "0.563"}
    for i in range(n):
        d[f"x{i}"] = xs[i]
        d[f"F{i}"] = s(F[i], 4)
        d[f"dp{i}"] = s(dp[i], 4)
        d[f"dm{i}"] = s(dm[i], 4)
    # (b) ドリフト付きランダムウォーク
    mu, sig, h, z = 0.05, 1.2, 25, 1.96
    m = mu * h
    sd = sig * sqrt(h)
    iv = lambda c, w: f"({s(c - w, 2)},\\ {s(c + w, 2)})"
    c = [iv(m, z * sd), iv(m, z * sig), iv(m, z * sig * h), iv(0, z * sd), iv(m, 1.645 * sd)]
    distinct(*c)
    d.update({"mu": s(mu, 2), "sig": s(sig, 1), "h": h, "m": s(m, 2), "sd": s(sd, 1), "hw": s(z * sd, 2),
              "var1": s(sig ** 2, 2), "varh": s(sig ** 2 * h, 0),
              "rw_ok": c[0], "rw_one": c[1], "rw_h": c[2], "rw_nod": c[3], "rw_90": c[4]})
    return d


def p07():
    # (a) 格付け推移の2年後
    P = {"A": (0.90, 0.08, 0.02), "B": (0.10, 0.80, 0.10)}
    aa, ab, ad = P["A"]
    ba, bb, bd = P["B"]
    two = aa * ad + ab * bd + ad * 1
    c = [s(two, 4), s(2 * ad, 4), s(1 - (1 - ad) ** 2, 4), s(aa * ad + ab * bd, 4), s(ad ** 2, 4)]
    distinct(*c)
    # (b) 順序ロジット
    th1, th2, beta, x = 0.5, 2.5, 0.8, 2.5
    ex = {1.5: 4.4817, -0.5: 0.6065}
    L1 = 1 / (1 + ex[1.5])      # L(-1.5)
    L2 = 1 / (1 + ex[-0.5])     # L(0.5)
    p2_ = L2 - L1
    co = [s(L2, 3), s(L1, 3), s(p2_, 3), s(1 - L2, 3), s(1 - L1, 3)]
    distinct(*co)
    # (c) ベイズ決定
    prior = [0.80, 0.15, 0.05]
    lik = [0.10, 0.40, 0.80]
    joint = [a * b for a, b in zip(prior, lik)]
    tj = sum(joint)
    post = [j / tj for j in joint]
    loss = {"full": [-4, 3, 20], "hedge": [-2, 0.5, 3], "skip": [0, 0, 0]}
    el = {k: sum(p * l for p, l in zip(post, v)) for k, v in loss.items()}
    elp = {k: sum(p * l for p, l in zip(prior, v)) for k, v in loss.items()}
    best = min(el, key=el.get)
    assert best == "hedge" and min(elp, key=elp.get) == "full"
    return {"aa": s(aa, 2), "ab": s(ab, 2), "ad": s(ad, 2), "ba": s(ba, 2), "bb": s(bb, 2), "bd": s(bd, 2),
            "t1": s(aa * ad, 3), "t2": s(ab * bd, 3), "t3": s(ad, 2),
            "two": c[0], "w_dbl": c[1], "w_ind": c[2], "w_noabs": c[3], "w_sq": c[4],
            "th1": th1, "th2": th2, "beta": beta, "x": x, "bx": s(beta * x, 1), "e15": s(ex[1.5], 4), "em05": s(ex[-0.5], 4),
            "L1": s(L1, 4), "L2": s(L2, 4), "p2": co[2], "p2_4": s(p2_, 4), "o_L2": co[0], "o_L1": co[1], "o_p3": co[3], "o_ge2": co[4],
            "pr1": s(prior[0], 2), "pr2": s(prior[1], 2), "pr3": s(prior[2], 2),
            "lk1": s(lik[0], 2), "lk2": s(lik[1], 2), "lk3": s(lik[2], 2),
            "j1": s(joint[0], 2), "j2": s(joint[1], 2), "j3": s(joint[2], 2), "tj": s(tj, 2),
            "po1": s(post[0], 4), "po2": s(post[1], 4), "po3": s(post[2], 4),
            "po1f": "4/9", "po2f": "3/9", "po3f": "2/9",
            "elf": s(el["full"], 2), "elh": s(el["hedge"], 2), "els": s(el["skip"], 2),
            "elpf": s(elp["full"], 2), "elph": s(elp["hedge"], 3)}


def p08():
    # (a) 対数価格のダミー係数
    b = 0.18
    eb, emb, tb = 1.1972, 0.8353, 1.5136   # e^0.18, e^-0.18, 10^0.18
    # (b) ホワイトの標準誤差（5件の単回帰）
    x = [40, 50, 60, 70, 90]
    y = [29, 36, 41, 49, 58]
    nn = len(x)
    xb, yb = sum(x) / nn, sum(y) / nn
    Sxx = sum((xi - xb) ** 2 for xi in x)
    Sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    b1 = Sxy / Sxx
    b0 = yb - b1 * xb
    e = [r(yi - (b0 + b1 * xi), 3) for xi, yi in zip(x, y)]
    dev = [xi - xb for xi in x]
    terms = [dd * dd * ee * ee for dd, ee in zip(dev, e)]
    hc0 = sqrt(sum(terms)) / Sxx
    sse = sum(ee * ee for ee in e)
    ols = sqrt(sse / (nn - 2) / Sxx)
    # (c) マローズの Cp
    n, sse_full, pfull = 60, 1140.0, 4
    s2 = sse_full / (n - pfull)
    models = {"M1": (1480.0, 2), "M2": (1160.0, 3), "M3": (1140.0, 4)}
    cp = {k: v[0] / s2 - n + 2 * v[1] for k, v in models.items()}
    assert min(cp, key=cp.get) == "M2"
    s2n = sse_full / n
    cpn = {k: v[0] / s2n - n + 2 * v[1] for k, v in models.items()}
    d = {"b": s(b, 2), "eb": s(eb, 4), "emb": s(emb, 4), "tb": s(tb, 4),
         "pct": s((eb - 1) * 100, 1), "pct_b": s(b * 100, 1), "pct_m": s((1 - emb) * 100, 1), "pct_t": s((tb - 1) * 100, 1),
         "Sxx": s(Sxx, 0), "xbar": s(xb, 0), "hc0": r(hc0, 4), "hc0_6": s(hc0, 6), "sumt": s(sum(terms), 2),
         "sqsum": s(sqrt(sum(terms)), 3), "ols": s(ols, 4), "sse": s(sse, 4), "w_nosq": s(sqrt(sum(terms) / Sxx), 4),
         "w_nosqrt": s(sum(terms) / Sxx ** 2, 6),
         "n": n, "sse_full": s(sse_full, 0), "pfull": pfull, "dfr": n - pfull, "s2": s(s2, 3),
         "s1": s(models["M1"][0], 0), "s2m": s(models["M2"][0], 0), "s3": s(models["M3"][0], 0),
         "r1": s(models["M1"][0] / s2, 2), "r2": s(models["M2"][0] / s2, 2), "r3": s(models["M3"][0] / s2, 2),
         "cp1": s(cp["M1"], 2), "cp2": s(cp["M2"], 2), "cp3": s(cp["M3"], 2),
         "cq1": s(cp["M1"] - 2, 2), "cq2": s(cp["M2"] - 2, 2), "cq3": s(cp["M3"] - 2, 2),
         "cn1": s(cpn["M1"], 2), "cn2": s(cpn["M2"], 2), "cn3": s(cpn["M3"], 2), "s2n": s(s2n, 1)}
    for i in range(nn):
        d[f"x{i}"] = x[i]
        d[f"y{i}"] = y[i]
        d[f"e{i}"] = s(e[i], 3)
        d[f"d{i}"] = s(dev[i], 0)
        d[f"tt{i}"] = s(terms[i], 2)
    return d


def p09():
    # 相関のある2つの標準化説明変数のリッジ回帰
    rho, g1, g2, lam = 0.8, 0.7, 0.6, 0.3
    def solve(a, bb, c, d_, u, v):
        det = a * d_ - bb * c
        return ((d_ * u - bb * v) / det, (-c * u + a * v) / det)
    ols = solve(1, rho, rho, 1, g1, g2)
    rid = solve(1 + lam, rho, rho, 1 + lam, g1, g2)
    half = solve(1 + lam / 2, rho, rho, 1 + lam / 2, g1, g2)
    orth = (ols[0] / (1 + lam), ols[1] / (1 + lam))
    pair = lambda t: f"(\\hat\\beta_1,\\ \\hat\\beta_2)=({s(t[0], 3)},\\ {s(t[1], 3)})"
    c = [pair(ols), pair(orth), pair(rid), pair(half), pair((g1, g2))]
    distinct(*c)
    det = (1 + lam) ** 2 - rho ** 2
    return {"rho": rho, "g1": g1, "g2": g2, "lam": lam, "l1": s(1 + lam, 1), "det": s(det, 2),
            "u1": s((1 + lam) * g1 - rho * g2, 2), "u2": s(-rho * g1 + (1 + lam) * g2, 2),
            "b1": s(rid[0], 4), "b2": s(rid[1], 4), "o1": s(ols[0], 3), "o2": s(ols[1], 3),
            "c_ols": c[0], "c_orth": c[1], "c_ok": c[2], "c_half": c[3], "c_simple": c[4],
            "a1": s((1 + lam) * g1, 2), "a2": s(rho * g2, 2), "a3": s(rho * g1, 2), "a4": s((1 + lam) * g2, 2),
            "l1sq": s((1 + lam) ** 2, 2), "rho2": s(rho ** 2, 2)}


def p10():
    # テューキーの多重比較
    k, n, mse = 3, 8, 25.0
    dfe = k * (n - 1)
    q, t = 3.565, 2.080
    means = {"A": 63.5, "B": 56.5, "C": 58.0}
    hsd = q * sqrt(mse / n)
    lsd = t * sqrt(2 * mse / n)
    w2 = q * sqrt(2 * mse / n)
    wN = q * sqrt(mse / (k * n))
    diffs = {"AB": 7.0, "AC": 5.5, "CB": 1.5}
    assert diffs["AB"] > hsd > diffs["AC"] > lsd and w2 > diffs["AB"] and wN < diffs["AC"]
    return {"k": k, "n": n, "mse": s(mse, 1), "dfe": dfe, "q": s(q, 3), "t": s(t, 3),
            "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1),
            "dAB": s(diffs["AB"], 1), "dAC": s(diffs["AC"], 1), "dCB": s(diffs["CB"], 1),
            "se": s(sqrt(mse / n), 4), "hsd": s(hsd, 2), "lsd": s(lsd, 2), "w2": s(w2, 2), "wN": s(wN, 2),
            "mse_n": s(mse / n, 3), "N": k * n}


def p11():
    # 回帰推定量
    N, n, Xbar = 400, 8, 26.0
    xb, yb, sxx, sxy, syy = 24.0, 30.5, 40.0, 46.0, 60.0
    b = sxy / sxx
    reg = yb + b * (Xbar - xb)
    rat = yb * Xbar / xb
    rho = sxy / sqrt(sxx * syy)
    return {"N": N, "n": n, "Xbar": s(Xbar, 1), "xb": s(xb, 1), "yb": s(yb, 1), "sxx": s(sxx, 1), "sxy": s(sxy, 1),
            "syy": s(syy, 1), "b": s(b, 2), "dx": s(Xbar - xb, 1), "adj": s(b * (Xbar - xb), 2), "reg": r(reg, 2),
            "reg_s": s(reg, 2), "rat": s(rat, 2), "w_sign": s(yb - b * (Xbar - xb), 2), "rho": s(rho, 3),
            "red": s(1 - rho ** 2, 3), "tot": s(N * reg / 100, 2)}


def p12():
    # 金利変化の主成分得点（共分散行列の固有ベクトル、問題文の丸めた値で計算）
    lam = [44.34, 6.96, 2.69]
    v1 = [0.517, 0.644, 0.563]
    v2 = [0.713, 0.040, -0.700]
    x = [6, 8, 4]
    sc1 = sum(a * b for a, b in zip(v1, x))
    sc2 = sum(a * b for a, b in zip(v2, x))
    c = [s(sc1, 2), s(sc2, 2), s(sc1 / sqrt(lam[0]), 2), s(sum(x), 2), s(sc1 * lam[0], 1)]
    distinct(*c)
    return {"l1": s(lam[0], 2), "l2": s(lam[1], 2), "l3": s(lam[2], 2),
            "v11": s(v1[0], 3), "v12": s(v1[1], 3), "v13": s(v1[2], 3),
            "v21": s(v2[0], 3), "v22": s(v2[1], 3), "v23": s(v2[2], 3),
            "x1": x[0], "x2": x[1], "x3": x[2],
            "t1": s(v1[0] * x[0], 3), "t2": s(v1[1] * x[1], 3), "t3": s(v1[2] * x[2], 3),
            "sc": c[0], "sc2": c[1], "sc_std": c[2], "sc_sum": c[3], "sc_lam": c[4], "sc3": s(sc1, 3),
            "sqrtl1": s(sqrt(lam[0]), 3), "contrib": s(lam[0] / sum(lam), 3)}


def p13():
    # (a) 最尤因子分析の自由度
    p = 6
    df = lambda m: ((p - m) ** 2 - (p + m)) // 2
    d1, d2, d3 = df(1), df(2), df(3)
    assert (d1, d2, d3) == (9, 4, 0)
    nmom = p * (p + 1) // 2
    # (b) パス解析
    a, b, c = -0.5, 0.6, -0.2
    ind = a * b
    tot = ind + c
    ch = [s(tot, 2), s(ind, 2), s(c, 2), s(a + b + c, 2), s(a * b * c, 2)]
    distinct(*ch)
    return {"p": p, "d1": d1, "d2": d2, "d3": d3, "nmom": nmom, "par2": p * 2 + p - 1, "par2raw": p * 2 + p,
            "d2_rot": nmom - (p * 2 + p), "d2_sign": ((p - 2) ** 2 + (p + 2)) // 2, "m_sign": p - 1,
            "a": s(a, 1), "b": s(b, 1), "c": s(c, 1), "ind": ch[1], "tot": ch[0], "w_dir": ch[2], "w_sum": ch[3], "w_prod": ch[4]}


def values():
    return {
        "s07p01": p01(), "s07p02": p02(), "s07p03": p03(), "s07p04": p04(), "s07p05": p05(),
        "s07p06": p06(), "s07p07": p07(), "s07p08": p08(), "s07p09": p09(), "s07p10": p10(),
        "s07p11": p11(), "s07p12": p12(), "s07p13": p13(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
