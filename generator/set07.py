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


def p02_od():
    # (c) ポアソン回帰の過分散の補正
    n, k, X2, b, se = 120, 3, 175.5, 0.24, 0.080
    df = n - k
    phi = X2 / df
    sec = se * sqrt(phi)
    z0, zc = b / se, b / sec
    se_lin = se * phi
    phin = X2 / n
    se_n = se * sqrt(phin)
    assert 1.96 < zc < 2.576 and z0 > 2.576
    return {"od_n": n, "od_k": k, "od_X2": s(X2, 1), "od_b": s(b, 2), "od_se": s(se, 3), "od_df": df,
            "od_phi": s(phi, 2), "od_sqphi": s(sqrt(phi), 4), "od_sec": s(sec, 3), "od_z0": s(z0, 2), "od_zc": s(zc, 2),
            "od_selin": s(se_lin, 3), "od_zlin": s(b / se_lin, 2), "od_phin": s(phin, 3), "od_sen": s(se_n, 3),
            "od_zn": s(b / se_n, 2), "od_sec4": s(sec, 4)}


def p04():
    # (a) 二項分布の最頻値とその確率
    bn, bp = 40, 0.06
    q = {37: 0.10133, 38: 0.09525, 39: 0.08953}   # 0.94^37, 0.94^38, 0.94^39（問題文の値）
    kmode = int((bn + 1) * bp)
    assert kmode == 2
    pmode = comb(bn, 2) * bp ** 2 * q[38]
    # (b) 2×2 表の尤度比統計量 G²
    O = [[20, 180], [40, 160]]
    lnv = {"a": -0.40547, "b": 0.05716, "c": 0.28768, "d": -0.06062}   # ln(2/3), ln(18/17), ln(4/3), ln(16/17)
    terms = [20 * lnv["a"], 180 * lnv["b"], 40 * lnv["c"], 160 * lnv["d"]]
    half = sum(terms)
    G = 2 * half
    Ex = [[30, 170], [30, 170]]
    X2 = sum((O[i][j] - Ex[i][j]) ** 2 / Ex[i][j] for i in range(2) for j in range(2))
    # (c) 3元分割表の対数線形モデルの期待度数
    t = {("M", "S", "D"): 12, ("M", "S", "N"): 48, ("M", "L", "D"): 3, ("M", "L", "N"): 37,
         ("V", "S", "D"): 10, ("V", "S", "N"): 70, ("V", "L", "D"): 2, ("V", "L", "N"): 18}
    tt = sum(t.values())
    nAB = t[("M", "S", "D")] + t[("M", "S", "N")]
    nAC = t[("M", "S", "D")] + t[("M", "L", "D")]
    nA = sum(v for kk, v in t.items() if kk[0] == "M")
    nB = sum(v for kk, v in t.items() if kk[1] == "S")
    nC = sum(v for kk, v in t.items() if kk[2] == "D")
    e_ok = nAB * nAC / nA
    c = [s(nA * nB * nC / tt ** 2, 2), s(nAB * nC / tt, 2), s(e_ok, 2), s(nAC * nB / tt, 2), s(t[("M", "S", "D")], 2)]
    distinct(*c)
    d = {"bn": bn, "bp": bp, "q37": s(q[37], 5), "q38": s(q[38], 5), "q39": s(q[39], 5), "np1": s((bn + 1) * bp, 2),
         "kmode": kmode, "c402": comb(bn, 2), "p2": s(bp ** 2, 4), "pmode": r(pmode, 3), "pmode4": s(pmode, 4),
         "np": s(bn * bp, 1), "w37": s(comb(bn, 2) * bp ** 2 * q[37], 3), "w39": s(comb(bn, 2) * bp ** 2 * q[39], 3),
         "la": s(lnv["a"], 5), "lb": s(lnv["b"], 5), "lc": s(lnv["c"], 5), "ld": s(lnv["d"], 5),
         "g1": s(terms[0], 4), "g2": s(terms[1], 4), "g3": s(terms[2], 4), "g4": s(terms[3], 4),
         "Ghalf": s(half, 2), "Ghalf4": s(half, 4), "G": s(G, 2), "X2": s(X2, 2),
         "e_ok": c[2], "e_mut": c[0], "e_abc": c[1], "e_acb": c[3], "e_obs": c[4],
         "nAB": nAB, "nAC": nAC, "nA": nA, "nB": nB, "nC": nC, "tt": tt}
    for kk, v in t.items():
        d["t" + "".join(kk)] = v
    return d


def p05():
    # (a) 標準偏差（ボラティリティ）の信頼区間
    n, sd = 16, 4.0
    chi = {(15, 0.025): 27.488, (15, 0.975): 6.262, (15, 0.05): 24.996, (15, 0.95): 7.261,
           (16, 0.025): 28.845, (16, 0.975): 6.908}   # 上側パーセント点
    ss = (n - 1) * sd ** 2
    iv = lambda a, b: f"({s(a, 2)},\\ {s(b, 2)})"
    ci = (sqrt(ss / chi[(15, 0.025)]), sqrt(ss / chi[(15, 0.975)]))
    c = [iv(ss / chi[(15, 0.025)], ss / chi[(15, 0.975)]), iv(*ci),
         iv(sqrt(ss / chi[(15, 0.05)]), sqrt(ss / chi[(15, 0.95)])),
         iv(sqrt(n * sd ** 2 / chi[(16, 0.025)]), sqrt(n * sd ** 2 / chi[(16, 0.975)])),
         iv(sqrt(sd ** 2 / chi[(15, 0.025)]), sqrt(sd ** 2 / chi[(15, 0.975)]))]
    distinct(*c)
    d = {"n": n, "sd": s(sd, 1), "ss": s(ss, 0), "s2": s(sd ** 2, 0),
         "c1": s(chi[(15, 0.025)], 3), "c2": s(chi[(15, 0.975)], 3), "c3": s(chi[(15, 0.05)], 3), "c4": s(chi[(15, 0.95)], 3),
         "c5": s(chi[(16, 0.025)], 3), "c6": s(chi[(16, 0.975)], 3),
         "v_lo": s(ss / chi[(15, 0.025)], 4), "v_hi": s(ss / chi[(15, 0.975)], 4),
         "ci_var": c[0], "ci_ok": c[1], "ci_90": c[2], "ci_n": c[3], "ci_s2": c[4]}
    # (b) 分位点回帰（切片のみ）
    y = [1.9, -2.5, 3.4, 0.3, -4.1, 5.0, 1.2, -0.8, 2.6]
    tau = 0.2
    ys = sorted(y)
    mean = sum(y) / len(y)
    sdv = sqrt(sum((v - mean) ** 2 for v in y) / (len(y) - 1))
    loss = lambda q: sum((v - q) * (tau - (1 if v < q else 0)) for v in y)
    best = min(ys, key=loss)
    assert best == ys[1]
    qc = [s(ys[0], 1), s(ys[1], 1), s(ys[2], 1), s(ys[4], 1), s(mean - 0.8416 * sdv, 2)]
    distinct(*qc)
    for i, v in enumerate(y):
        d[f"y{i}"] = v
    for i, v in enumerate(ys):
        d[f"ys{i}"] = v
    d.update({"tau": tau, "q_ok": qc[1], "q_min": qc[0], "q_3": qc[2], "q_med": qc[4 - 1], "q_norm": qc[4],
              "L1": s(loss(ys[0]), 2), "L2": s(loss(ys[1]), 2), "L3": s(loss(ys[2]), 2),
              "ymean": s(mean, 4), "ysd": s(sdv, 4)})
    # (c) ドリフト付きランダムウォーク
    mu, sig, h, z = 0.05, 1.2, 25, 1.96
    m = mu * h
    sdh = sig * sqrt(h)
    iv2 = lambda cc, w: f"({s(cc - w, 2)},\\ {s(cc + w, 2)})"
    rc = [iv2(m, z * sig), iv2(m, z * sig * h), iv2(m, z * sdh), iv2(0, z * sdh), iv2(m, 1.645 * sdh)]
    distinct(*rc)
    d.update({"mu": s(mu, 2), "sig": s(sig, 1), "h": h, "m": s(m, 2), "sdh": s(sdh, 1), "hw": s(z * sdh, 2),
              "varh": s(sig ** 2 * h, 0),
              "rw_one": rc[0], "rw_h": rc[1], "rw_ok": rc[2], "rw_nod": rc[3], "rw_90": rc[4]})
    return d


def p06():
    # (a) デフォルトまでの期待年数
    aa, ab, ad = 0.90, 0.08, 0.02
    ba, bb, bd = 0.10, 0.80, 0.10
    # t_A = 1 + aa t_A + ab t_B ; t_B = 1 + ba t_A + bb t_B
    # (1-aa) t_A - ab t_B = 1 ; -ba t_A + (1-bb) t_B = 1
    det = (1 - aa) * (1 - bb) - ab * ba
    tA = ((1 - bb) + ab) / det
    tB = ((1 - aa) + ba) / det
    c = [s(1 / ad, 2), s(tB, 2), s(tA, 2), s(1 / bd, 2), s(1 / (ad + bd), 2)]
    distinct(*c)
    # (b) ベイズ決定
    prior = [0.80, 0.15, 0.05]
    lik = [0.10, 0.40, 0.80]
    joint = [a * b for a, b in zip(prior, lik)]
    tj = sum(joint)
    post = [j / tj for j in joint]
    loss = {"full": [-4, 3, 20], "hedge": [-2, 0.5, 3], "skip": [0, 0, 0]}
    el = {kk: sum(pp * l for pp, l in zip(post, v)) for kk, v in loss.items()}
    elp = {kk: sum(pp * l for pp, l in zip(prior, v)) for kk, v in loss.items()}
    assert min(el, key=el.get) == "hedge" and min(elp, key=elp.get) == "full"
    return {"aa": s(aa, 2), "ab": s(ab, 2), "ad": s(ad, 2), "ba": s(ba, 2), "bb": s(bb, 2), "bd": s(bd, 2),
            "oa": s(1 - aa, 2), "ob": s(1 - bb, 2), "det": s(det, 3), "numA": s((1 - bb) + ab, 2), "numB": s((1 - aa) + ba, 2),
            "tA": c[2], "tB": c[1], "w_inv": c[0], "w_invB": c[3], "w_sum": c[4],
            "pr1": s(prior[0], 2), "pr2": s(prior[1], 2), "pr3": s(prior[2], 2),
            "lk1": s(lik[0], 2), "lk2": s(lik[1], 2), "lk3": s(lik[2], 2),
            "j1": s(joint[0], 2), "j2": s(joint[1], 2), "j3": s(joint[2], 2), "tj": s(tj, 2),
            "po1": s(post[0], 4), "po2": s(post[1], 4), "po3": s(post[2], 4),
            "elf": s(el["full"], 2), "elh": s(el["hedge"], 2), "els": s(el["skip"], 2), "elpf": s(elp["full"], 2)}


def p07():
    # (a) 対数価格のダミー係数
    b = 0.18
    eb, emb, tb = 1.1972, 0.8353, 1.5136   # e^0.18, e^-0.18, 10^0.18
    # (b) ゴールドフェルド・クォント検定
    n, drop, ng, k = 60, 20, 20, 2
    sse_lo, sse_hi = 120.0, 312.0
    dfg = ng - k
    F = (sse_hi / dfg) / (sse_lo / dfg)
    # (c) AIC と AICc
    nn = 15
    mods = {"A": (3, -40.0), "B": (6, -36.5)}
    aic = {kk: -2 * l + 2 * kk_ for kk, (kk_, l) in mods.items()}
    aicc = {kk: aic[kk] + 2 * kk_ * (kk_ + 1) / (nn - kk_ - 1) for kk, (kk_, l) in mods.items()}
    assert min(aic, key=aic.get) == "B" and min(aicc, key=aicc.get) == "A"
    return {"b": s(b, 2), "eb": s(eb, 4), "emb": s(emb, 4), "tb": s(tb, 4),
            "pct": s((eb - 1) * 100, 1), "pct_b": s(b * 100, 1), "pct_m": s((1 - emb) * 100, 1), "pct_t": s((tb - 1) * 100, 1),
            "n": n, "drop": drop, "ng": ng, "dfg": dfg, "sse_lo": s(sse_lo, 0), "sse_hi": s(sse_hi, 0),
            "F": s(F, 2), "Finv": s(1 / F, 3), "mlo": s(sse_lo / dfg, 3), "mhi": s(sse_hi / dfg, 3),
            "nn": nn, "kA": mods["A"][0], "kB": mods["B"][0], "lA": s(mods["A"][1], 1), "lB": s(mods["B"][1], 1),
            "aA": s(aic["A"], 1), "aB": s(aic["B"], 1), "ccA": s(aicc["A"], 2), "ccB": s(aicc["B"], 2),
            "corA": s(aicc["A"] - aic["A"], 2), "corB": s(aicc["B"] - aic["B"], 2),
            "nk1A": nn - mods["A"][0] - 1, "nk1B": nn - mods["B"][0] - 1}


def p08():
    # 効果量 η² と ω²
    k, n = 4, 10
    ssb, ssw = 180.0, 720.0
    N = k * n
    dfw = N - k
    mse = ssw / dfw
    sst = ssb + ssw
    eta = ssb / sst
    omega = (ssb - (k - 1) * mse) / (sst + mse)
    return {"k": k, "n": n, "N": N, "ssb": s(ssb, 0), "ssw": s(ssw, 0), "sst": s(sst, 0), "dfb": k - 1, "dfw": dfw,
            "mse": s(mse, 1), "eta": s(eta, 3), "omega": r(omega, 3), "omega4": s(omega, 4),
            "omn": s(ssb - (k - 1) * mse, 1), "omd": s(sst + mse, 1), "km1mse": s((k - 1) * mse, 1),
            "w_noadd": s((ssb - (k - 1) * mse) / sst, 3)}


def p09():
    # 事後層化推定量
    wp, wc = 0.8, 0.2
    n, np_, nc = 50, 45, 5
    yp, yc = 126.4, 912.5
    ps = wp * yp + wc * yc
    sm = (np_ * yp + nc * yc) / n
    return {"wp": s(wp, 1), "wc": s(wc, 1), "n": n, "np_": np_, "nc": nc, "yp": s(yp, 1), "yc": s(yc, 1),
            "tp": s(wp * yp, 2), "tc": s(wc * yc, 2), "ps": r(ps, 2), "ps_s": s(ps, 2), "sm": s(sm, 2), "fc": s(nc / n, 1)}


def p10():
    # 金利変化の第1主成分によるランク1近似
    lam = [44.34, 6.96, 2.69]
    v1 = [0.517, 0.644, 0.563]
    x = [6, 8, 4]
    sc1 = sum(a * b for a, b in zip(v1, x))
    rec = sc1 * v1[2]
    c = [s(rec, 1), s(sc1, 1), s(sc1 * v1[1], 1), s(x[2] - rec, 1), s(x[2], 1)]
    distinct(*c)
    return {"l1": s(lam[0], 2), "l2": s(lam[1], 2), "l3": s(lam[2], 2),
            "v11": s(v1[0], 3), "v12": s(v1[1], 3), "v13": s(v1[2], 3), "v21": "0.713", "v22": "0.040", "v23": "-0.700",
            "x1": x[0], "x2": x[1], "x3": x[2],
            "t1": s(v1[0] * x[0], 3), "t2": s(v1[1] * x[1], 3), "t3": s(v1[2] * x[2], 3), "sc": s(sc1, 3),
            "rec": c[0], "rec3": s(rec, 3), "w_sc": c[1], "w_v12": c[2], "w_res": c[3], "w_obs": c[4],
            "contrib": s(lam[0] / sum(lam), 3)}


def p11():
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
        "s07p01": p01(), "s07p02": {**p02(), **p02_od()}, "s07p03": p03(), "s07p04": p04(), "s07p05": p05(),
        "s07p06": p06(), "s07p07": p07(), "s07p08": p08(), "s07p09": p09(), "s07p10": p10(), "s07p11": p11(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
