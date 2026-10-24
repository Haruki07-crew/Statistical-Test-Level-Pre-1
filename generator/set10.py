"""模擬試験 第10回（set10）の数値。values() が {大問id: {名前: 値}} を返す。

題材は交通・物流・都市計画・観光。問題文に示した分布の値（Φ・t・q の点、e^x など）は、
ここでも同じ丸めた値を使って答えを計算する。表示する途中の値は、次の段の計算に使う値と一致させる。
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


def p01():
    # (a) 3 経路の遅延率からベイズの定理
    share = {"A": 0.5, "B": 0.3, "C": 0.2}
    dl = {"A": 0.04, "B": 0.10, "C": 0.12}
    joint = {k: share[k] * dl[k] for k in share}
    pd = sum(joint.values())
    post = {k: joint[k] / pd for k in share}
    # (b) 時間帯 C を与えると雨 A と遅延 B が条件付き独立 [AC][BC] の当てはめ値
    t1 = (12, 48, 15, 125)   # 午前：雨・遅延, 雨・定刻, 晴・遅延, 晴・定刻
    t2 = (20, 30, 30, 120)   # 午後
    n1, n2 = sum(t1), sum(t2)
    rain1, del1 = t1[0] + t1[1], t1[0] + t1[2]
    fit = rain1 * del1 / n1
    N = n1 + n2
    rain, dly, morn = t1[0] + t1[1] + t2[0] + t2[1], t1[0] + t1[2] + t2[0] + t2[2], n1
    fit_ind = rain * dly * morn / N ** 2           # 完全独立 [A][B][C]
    fit_ab = (t1[0] + t2[0]) * morn / N             # [AB][C]
    return {
        "sA": share["A"], "sB": share["B"], "sC": share["C"],
        "dA": dl["A"], "dB": dl["B"], "dC": dl["C"],
        "jA": s(joint["A"], 3), "jB": s(joint["B"], 3), "jC": s(joint["C"], 3), "pd": s(pd, 3),
        "postB": s(post["B"], 3), "postB4": s(post["B"], 4), "postA": s(post["A"], 3), "postC": s(post["C"], 3),
        "w_lik": s(dl["B"], 3), "w_joint": s(joint["B"], 3), "w_prior": s(share["B"], 3),
        "w_norm": s(dl["B"] / sum(dl.values()), 3), "sumd": s(sum(dl.values()), 2),
        "a1": t1[0], "b1": t1[1], "c1": t1[2], "d1": t1[3], "n1": n1,
        "a2": t2[0], "b2": t2[1], "c2": t2[2], "d2": t2[3], "n2": n2,
        "rain1": rain1, "del1": del1, "fit": r(fit, 2), "fit_s": s(fit, 2),
        "N": N, "rain": rain, "dly": dly, "fit_ind": s(fit_ind, 2), "fit_ab": s(fit_ab, 2), "a12": t1[0] + t2[0],
        "fit2": s((t2[0] + t2[1]) * (t2[0] + t2[2]) / n2, 2), "rain2": t2[0] + t2[1], "del2": t2[0] + t2[2],
    }

def p02():
    pS, pR, pL = 0.6, 0.25, 0.15
    n, kS, kR, kL = 8, 5, 2, 1
    coef = comb(n, kS) * comb(n - kS, kR)
    p_a = coef * pS ** kS * pR ** kR * pL ** kL
    # (b) S=5 のとき残り 3 台の右折台数 ~ B(3, pR/(pR+pL))
    m = n - kS
    pc = pR / (pR + pL)
    p_b = comb(m, 2) * pc ** 2 * (1 - pc)
    return {
        "pS": pS, "pR": pR, "pL": pL, "n": n, "kS": kS, "kR": kR, "kL": kL, "m": m,
        "coef": coef, "pS5": s(pS ** kS, 5), "pR2": s(pR ** kR, 4), "prod": s(pS ** kS * pR ** kR * pL, 6),
        "pa": r(p_a, 3), "pa_s": s(p_a, 3), "pa6": s(p_a, 6),
        "pRL": s(pR + pL, 2), "pc": s(pc, 3), "qc": s(1 - pc, 3), "pc2": s(pc ** 2, 6),
        "pb": s(p_b, 3), "pb6": s(p_b, 6),
        "w_uncond": s(comb(n, 2) * pR ** 2 * (1 - pR) ** 6, 3),
        "w_raw": s(comb(m, 2) * pR ** 2 * (1 - pR), 3),
        "w_swap": s(comb(m, 2) * (1 - pc) ** 2 * pc, 3),
        "w_nocomb": s(pc ** 2, 3),
        "pa_noc": s(pS ** kS * pR ** kR * pL, 4),
    }


def p03():
    n, xbar, m2 = 60, 4.5, 6.75
    theta = m2 / xbar
    alpha = xbar / theta
    m2u = m2 * n / (n - 1)
    th_u = m2u / xbar
    sd = sqrt(m2)
    th_sd = sd / xbar
    # (b) Gamma(3, 1.5) の P(X > 6)：e^{-4} を問題文の値で
    x0 = 6
    z = x0 / theta
    e4 = 0.01832
    terms = 1 + z + z ** 2 / 2
    surv = terms * e4
    return {
        "n": n, "xbar": xbar, "m2": m2, "theta": s(theta, 2), "alpha": s(alpha, 2),
        "rate": s(1 / theta, 2), "m2u": s(m2u, 4), "th_u": s(th_u, 2), "al_u": s(xbar / th_u, 2),
        "sd": s(sd, 3), "th_sd": s(th_sd, 2), "al_sd": s(xbar / th_sd, 2),
        "x0": x0, "z": s(z, 0), "z2h": s(z ** 2 / 2, 0), "terms": s(terms, 0), "e4": e4,
        "surv": r(surv, 3), "surv_s": s(surv, 3), "surv5": s(surv, 5),
        "w_k2": s((1 + z) * e4, 4), "w_k1": s(e4, 4),
    }


def p04():
    # (a) ポアソン過程：1 時間に 6 台到着したとき最初の 20 分にちょうど 2 台
    k, j, frac = 6, 2, 1 / 3
    pa = comb(k, j) * frac ** j * (1 - frac) ** (k - j)
    e2 = 0.1353
    # (b) 中心極限定理：1.5n + z*0.4*sqrt(n) <= 60
    m, sd, cap, z, z2 = 1.5, 0.4, 60, 1.645, 1.960

    def nmax(zz, sdterm_sqrt=True):
        nn = 0
        while True:
            kk = nn + 1
            spread = zz * sd * (sqrt(kk) if sdterm_sqrt else kk)
            if m * kk + spread > cap:
                return nn
            nn = kk

    n_ok = nmax(z)
    a_, b_, c_ = m, z * sd, -cap
    disc = b_ ** 2 - 4 * a_ * c_
    x = (-b_ + sqrt(disc)) / (2 * a_)
    chk = lambda kk: m * kk + z * sd * sqrt(kk)
    # (c) 2 窓口の残り受付時間の最小値（指数分布、平均 4 分と 6 分）
    m1, m2, w = 4, 6, 2
    e56, e15 = 0.4346, 0.8187
    rate = 1 / m1 + 1 / m2
    return {
        "k": k, "j": j, "pa": s(pa, 3), "pa5": s(pa, 5), "num": comb(k, j) * 2 ** (k - j), "den": 3 ** k,
        "c62": comb(k, j), "e2": e2,
        "w_pois": s(2 * e2 * 2 / 2, 3), "w_noc": s(frac ** j * (1 - frac) ** (k - j), 3),
        "w_swap": s(comb(k, j) * frac ** (k - j) * (1 - frac) ** j, 3), "w_frac": s(frac, 3),
        "m": m, "sd": sd, "cap": cap, "z": z, "z2": s(z2, 3), "zsd": s(z * sd, 3),
        "disc": s(disc, 4), "sqd": s(sqrt(disc), 4), "x": s(x, 4), "xx": s(x * x, 2),
        "n_ok": n_ok, "n_up": n_ok + 1, "n_196": nmax(z2), "n_lin": nmax(z, sdterm_sqrt=False), "n_mean": int(cap / m),
        "sq37": s(sqrt(n_ok), 4), "sq38": s(sqrt(n_ok + 1), 4),
        "c37": s(chk(n_ok), 2), "c38": s(chk(n_ok + 1), 2),
        "m37": s(m * n_ok, 1), "m38": s(m * (n_ok + 1), 1),
        "s37": s(z * sd * sqrt(n_ok), 2), "s38": s(z * sd * sqrt(n_ok + 1), 2),
        "m1": m1, "m2": m2, "w": w, "e56": e56, "e15": e15, "rate_num": 5, "rate_den": 12, "wr": "5/6",
        "pc": r(e56, 3), "pc_s": s(e56, 3), "meanW": s(1 / rate, 1), "w_sum": s(e15, 3),
    }

def p05():
    # (a) T ~ U(20, 30) 分、距離 10 km、V = 600 / T（km/h）
    # (b) 対応のある差の標準偏差から必要な日数
    A = [38.5, 42.0, 35.2, 40.8, 45.3, 36.9, 39.4, 41.6]
    B = [35.3, 40.5, 36.0, 38.4, 41.2, 36.3, 37.5, 38.9]
    d = [round(a - b, 1) for a, b in zip(A, B)]
    n = len(d)
    dbar = sum(d) / n
    ss = sum((x - dbar) ** 2 for x in d)
    s2 = r(ss / (n - 1), 4)
    sd_ = r(sqrt(s2), 4)
    z, E = 1.96, 0.5
    root = z * sd_ / E
    need = root ** 2
    nn = int(need) + (0 if need == int(need) else 1)
    sd_n = sqrt(ss / n)
    up = lambda v: int(v) + (0 if v == int(v) else 1)
    out = {"n": n, "dsum": s(sum(d), 1), "dbar": s(dbar, 2), "ss": s(ss, 2), "s2": s(s2, 4), "sd": s(sd_, 4),
           "z": z, "E": E, "zs": s(z * sd_, 4), "root": s(root, 4), "need": s(need, 2), "nn": nn,
           "w_sdn": up((z * sd_n / E) ** 2), "sdn": s(sd_n, 4), "w_z": up((1.645 * sd_ / E) ** 2),
           "w_E": up((z * sd_ / (E / 2)) ** 2), "w_nosq": up(root)}
    for i in range(n):
        out[f"a{i + 1}"] = s(A[i], 1)
        out[f"b{i + 1}"] = s(B[i], 1)
        out[f"d{i + 1}"] = s(d[i], 1)
    return out

def p06():
    # (a) 確率化検定：X ~ Po(λ)、H0: λ=4 対 H1: λ<4、e^{-4}=0.0183
    lam, e4, alpha = 4, 0.0183, 0.05
    p0 = e4
    p1 = r(lam * e4, 4)
    gam = (alpha - p0) / p1
    # (b) 順位和
    w = [1.2, 3.5, 0.8, 2.1, 4.4, 1.7]
    o = [2.9, 5.1, 3.8, 6.2, 2.4, 4.9, 7.0, 3.1]
    allv = sorted(w + o)
    rank = {v: i + 1 for i, v in enumerate(allv)}
    W = sum(rank[v] for v in w)
    m, n = len(w), len(o)
    N = m + n
    EW = m * (N + 1) / 2
    VW = m * n * (N + 1) / 12
    sq = 7.746
    z = (W - EW) / sq
    out = {
        "lam": lam, "e4": e4, "p0": s(p0, 4), "p1": s(p1, 4), "cum1": s(p0 + p1, 4), "rest": s(alpha - p0, 4),
        "gam": s(gam, 3), "gam5": s(gam, 5), "w_nop0": s(alpha / p1, 3), "w_cum": s((alpha - p0) / (p0 + p1), 3),
        "w_comp": s(1 - gam, 3), "w_rest": s(alpha - p0, 3),
        "W": W, "m": m, "n": n, "N": N, "N1": N + 1, "EW": s(EW, 0), "VW": s(VW, 0), "sqV": sq,
        "diff": s(W - EW, 0), "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4),
        "Wo": sum(rank[v] for v in o), "EWo": s(n * (N + 1) / 2, 0),
        "z_cc": s((W - EW + 0.5) / sq, 2),
        "w_var": s((W - EW) / sqrt(m * n * (N + 1) / 12 * 2), 2),
    }
    for i, v in enumerate(w):
        out[f"w{i + 1}"] = s(v, 1)
    for i, v in enumerate(o):
        out[f"o{i + 1}"] = s(v, 1)
    out["rw"] = ", ".join(str(rank[v]) for v in sorted(w))
    return out

def p07():
    # (a) t 値から 1 変数を除いたときの決定係数
    n, k, R2, t = 48, 6, 0.82, 2.5
    dfe = n - k - 1
    dR2 = t ** 2 * (1 - R2) / dfe
    R2r = R2 - dR2
    # (b) 1 観測を除いたときの傾きの変化（DFBETA）
    x = [1, 2, 3, 3, 4, 5]
    e = [1.2, -0.5, 0.2, -0.4, -2.9, 2.4]
    b0, b1 = 10.0, 2.0
    y = [round(b0 + b1 * xi + ei, 1) for xi, ei in zip(x, e)]
    xb = sum(x) / len(x)
    sxx = sum((xi - xb) ** 2 for xi in x)
    i = 5
    di = x[i] - xb
    h = 1 / len(x) + di ** 2 / sxx
    dfb = di * e[i] / (sxx * (1 - h))
    out = {
        "n": n, "k": k, "R2": R2, "t": t, "dfe": dfe, "t2": s(t ** 2, 2), "omR2": s(1 - R2, 2),
        "dR2": s(dR2, 5), "R2r": s(R2r, 3), "R2r5": s(R2r, 5),
        "w_noR": s(R2 - t ** 2 / dfe, 3), "w_R": s(R2 - t ** 2 * R2 / dfe, 3), "w_t": s(R2 - t * (1 - R2) / dfe, 3),
        "b0": s(b0, 1), "b1": s(b1, 1), "xb": s(xb, 0), "sxx": s(sxx, 0), "di": s(di, 0), "e6": e[i],
        "h": s(h, 4), "omh": s(1 - h, 4), "hden": s(sxx * (1 - h), 3), "num6": s(di * e[i], 1),
        "dfb": s(dfb, 2), "dfb4": s(dfb, 4), "new": s(b1 - dfb, 2),
        "w_noh": s(di * e[i] / sxx, 2), "w_h1": s(di * e[i] / (sxx * (1 - di ** 2 / sxx)), 2),
        "w_mul": s(di * e[i] * (1 - h) / sxx, 2), "w_e": s(e[i], 2),
    }
    for j in range(6):
        out[f"x{j + 1}"] = x[j]
        out[f"y{j + 1}"] = s(y[j], 1)
        out[f"e{j + 1}"] = s(e[j], 1)
    return out

def p08():
    rr, n = -0.48, 22
    r2 = rr ** 2
    t = rr * sqrt(n - 2) / sqrt(1 - r2)
    # (b) リッジ推定量の縮小率 c と MSE
    beta, v = 0.8, 0.16
    c = beta ** 2 / (beta ** 2 + v)
    mse = (1 - c) ** 2 * beta ** 2 + c ** 2 * v
    cw = v / (beta ** 2 + v)
    return {
        "r": rr, "n": n, "df": n - 2, "r2": s(r2, 4), "one_r2": s(1 - r2, 4), "sq1r2": s(sqrt(1 - r2), 4),
        "sqdf": s(sqrt(n - 2), 4), "t": s(t, 2), "t4": s(t, 4), "tc": 2.086,
        "w_nosq": s(rr * sqrt(n - 2), 2), "w_n1": s(rr * sqrt(n - 1) / sqrt(1 - r2), 2),
        "w_n3": s(rr * sqrt(n - 3) / sqrt(1 - r2), 2), "w_1r2": s(rr * sqrt(n - 2) / (1 - r2), 2),
        "beta": beta, "v": v, "b2": s(beta ** 2, 2), "c": s(c, 2), "mse": s(mse, 3),
        "bias2": s((1 - c) ** 2 * beta ** 2, 4), "var": s(c ** 2 * v, 4), "omc": s(1 - c, 1),
        "w_ols": s(v, 3), "w_novb": s(c ** 2 * v, 3), "cw": s(cw, 2), "w_cw": s((1 - cw) ** 2 * beta ** 2 + cw ** 2 * v, 3),
        "w_bias1": s((1 - c) * beta + c ** 2 * v, 3), "lam_ratio": s(1 / c - 1, 2),
    }

def p09():
    N, z, E, p = 2000, 1.96, 0.05, 0.5
    n0 = z ** 2 * p * (1 - p) / E ** 2
    nreq = N * n0 / (N - 1 + n0)
    up = lambda v: int(v) + (0 if v == int(v) else 1)
    return {
        "N": N, "z": z, "E": E, "n0": s(n0, 2), "N1": N - 1, "nden": s(N - 1 + n0, 2), "nnum": s(N * n0, 0),
        "nreq": s(nreq, 2), "nn": up(nreq), "w_nofpc": up(n0), "w_floor": int(nreq), "z2": s(z ** 2, 4),
        "alt": s(n0 / (1 + n0 / N), 2),
    }

def p10():
    b0, b1, b2 = -1.5, 0.25, -0.5
    x2 = 1.5
    c0 = b0 + b2 * x2
    x1 = -c0 / b1
    return {
        "b0": b0, "b1": b1, "b2": b2, "x2": x2, "t2": s(b2 * x2, 2), "c0": s(c0, 2), "x1": s(x1, 0),
        "yen": int(round(x1 * 100)), "w_nox2": int(round(-b0 / b1 * 100)), "w_sign": int(round(-(b0 - b2 * x2) / b1 * 100)),
        "w_half": int(round((0.5 - c0) / b1 * 100)), "w_ln": int(round((-0.693 - c0) / b1 * 100 / 10) * 10),
        "or2": s(2 * b1, 2), "e05": 1.649,
    }

def p11():
    means = {"A": 52.0, "B": 45.5, "C": 43.0}
    rr, mse, dfe, q, t = 8, 36, 21, 3.565, 2.080
    se1 = sqrt(mse / rr)
    hsd = r(q * r(se1, 3), 2)
    lsd = r(t * sqrt(2 * mse / rr), 2)
    hsd_w = r(q * sqrt(2 * mse / rr), 2)
    return {
        "mA": s(means["A"], 1), "mB": s(means["B"], 1), "mC": s(means["C"], 1), "rr": rr, "N": 3 * rr,
        "mse": mse, "dfe": dfe, "q": q, "t": s(t, 3), "mr": s(mse / rr, 1), "se1": s(se1, 3),
        "hsd": s(hsd, 2), "lsd": s(lsd, 2), "hsd_w": s(hsd_w, 2), "se2": s(sqrt(2 * mse / rr), 0),
        "dAB": s(means["A"] - means["B"], 1), "dAC": s(means["A"] - means["C"], 1), "dBC": s(means["B"] - means["C"], 1),
    }


def p12():
    # (a) 尺度をそろえてからのユークリッド距離
    pts = {"P": (100, 2.0), "Q": (130, 5.0), "R": (190, 5.2), "S": (60, 3.5)}
    sv, ss_ = 50, 1.0
    out = {"sv": sv, "ss": s(ss_, 1)}
    for k, (v, h) in pts.items():
        out[f"v{k}"] = v
        out[f"h{k}"] = s(h, 1)
        out[f"z{k}"] = s(v / sv, 1)
        out[f"y{k}"] = s(h / ss_, 1)
    keys = list(pts)
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = keys[i], keys[j]
            dv = (pts[a][0] - pts[b][0]) / sv
            dh = (pts[a][1] - pts[b][1]) / ss_
            out[f"d{a}{b}2"] = s(dv ** 2 + dh ** 2, 2)
            out[f"d{a}{b}"] = s(sqrt(dv ** 2 + dh ** 2), 2)
            out[f"raw{a}{b}"] = s(sqrt((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2), 2)
            out[f"man{a}{b}"] = s(abs(dv) + abs(dh), 2)
    # (b) 直交回転
    L = [(0.58, 0.56), (0.44, 0.58), (-0.37, 0.66)]
    c, sn = 0.8, 0.6
    rot = [(c * a + sn * b, -sn * a + c * b) for a, b in L]
    for i, ((a, b), (ra, rb)) in enumerate(zip(L, rot), 1):
        out[f"l{i}1"] = s(a, 2)
        out[f"l{i}2"] = s(b, 2)
        out[f"r{i}1"] = s(ra, 3)
        out[f"r{i}2"] = s(rb, 3)
        out[f"r{i}1sq"] = s(ra ** 2, 4)
        out[f"h{i}"] = s(a * a + b * b, 4)
    out["c"] = c
    out["sn"] = sn
    contr = sum(ra ** 2 for ra, _ in rot)
    out["con1"] = s(contr, 2)
    out["con1_4"] = s(contr, 4)
    out["con2"] = s(sum(rb ** 2 for _, rb in rot), 2)
    out["u1"] = s(sum(a * a for a, _ in L), 2)
    out["u1_4"] = s(sum(a * a for a, _ in L), 4)
    out["u2"] = s(sum(b * b for _, b in L), 2)
    out["comm"] = s(sum(a * a + b * b for a, b in L), 2)
    out["comm4"] = s(sum(a * a + b * b for a, b in L), 4)
    out["prop"] = s(contr / 3, 2)
    for i, (a, b) in enumerate(L, 1):
        out[f"l{i}1sq"] = s(a * a, 4)
    # (c) RMSEA
    p, qf, nobs, chi2 = 10, 25, 301, 78.0
    df = p * (p + 1) // 2 - qf
    rm = sqrt((chi2 - df) / (df * (nobs - 1)))
    df_w = p * (p - 1) // 2 - qf
    out.update({
        "p": p, "qf": qf, "nobs": nobs, "chi2": s(chi2, 1), "mom": p * (p + 1) // 2, "dfm": df,
        "num": s(chi2 - df, 1), "dden": df * (nobs - 1), "rat": s((chi2 - df) / (df * (nobs - 1)), 6),
        "rmsea": r(rm, 3), "rmsea_s": s(rm, 3), "rmsea5": s(rm, 5),
        "w_df": df_w, "w_rm": s(sqrt((chi2 - df_w) / (df_w * (nobs - 1))), 3),
        "w_nosq": s((chi2 - df) / (df * (nobs - 1)), 4),
    })
    return out


def p13():
    f1, f2 = 0.5, 0.3
    r1 = f1 / (1 - f2)
    r2 = f1 * r1 + f2
    return {
        "f1": f1, "f2": f2, "r1": s(r1, 4), "r1_3": s(r1, 3), "r2": s(r2, 3), "r2_4": s(r2, 4),
        "f1r1": s(f1 * r1, 4), "w_pacf": s(f2, 3), "w_ar1": s(r1 ** 2, 3), "w_phi": s(f1 ** 2 + f2, 3),
        "omf2": s(1 - f2, 1), "sumf": s(f1 + f2, 1), "dif": s(f2 - f1, 1),
    }


def p14():
    a, b = 5, 5
    th, thp = 1.0, 0.6
    e16, e2 = 4.953, 7.389
    pr = (thp / th) ** (a - 1)
    acc = pr * e16
    return {
        "a": a, "b": b, "a1": a - 1, "th": s(th, 1), "thp": s(thp, 1), "e16": e16, "e2": e2,
        "pr": s(pr, 4), "pexp": s(b * (th - thp), 1), "qexp": s(thp - th, 1), "tot": s(b * (th - thp) + (thp - th), 1),
        "acc": s(acc, 3), "acc4": s(acc, 4), "w_rej": s(1 - acc, 3), "w_noq": s(pr * e2, 3),
        "w_shape": s((thp / th) ** a * e16, 3), "pr5": s((thp / th) ** a, 5),
    }


def values():
    return {"s10p01": p01(), "s10p02": p02(), "s10p03": p03(), "s10p04": p04(), "s10p05": p05(),
            "s10p06": p06(), "s10p07": p07(), "s10p08": p08(), "s10p09": p09(), "s10p10": p10(),
            "s10p11": p11(), "s10p12": p12(), "s10p13": p13(), "s10p14": p14()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
