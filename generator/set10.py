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
    # (a) 2 センターの遅延は独立で確率 1/2、C=ちょうど一方だけ遅延：対ごとに独立だが相互独立でない
    # (b) マンテル・ヘンツェルの共通オッズ比（行：雨・晴れ、列：遅延・定刻）
    t1 = (12, 48, 15, 125)
    t2 = (20, 30, 30, 120)
    n1, n2 = sum(t1), sum(t2)
    ad1, bc1 = t1[0] * t1[3] / n1, t1[1] * t1[2] / n1
    ad2, bc2 = t2[0] * t2[3] / n2, t2[1] * t2[2] / n2
    mh = (ad1 + ad2) / (bc1 + bc2)
    or1 = t1[0] * t1[3] / (t1[1] * t1[2])
    or2 = t2[0] * t2[3] / (t2[1] * t2[2])
    A, B, C, D = t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2], t1[3] + t2[3]
    crude = A * D / (B * C)
    return {
        "a1": t1[0], "b1": t1[1], "c1": t1[2], "d1": t1[3], "n1": n1,
        "a2": t2[0], "b2": t2[1], "c2": t2[2], "d2": t2[3], "n2": n2,
        "ad1": s(ad1, 1), "bc1": s(bc1, 1), "ad2": s(ad2, 1), "bc2": s(bc2, 1),
        "num": s(ad1 + ad2, 1), "den": s(bc1 + bc2, 1),
        "mh": r(mh, 2), "mh_s": s(mh, 2), "mh4": s(mh, 4),
        "or1": s(or1, 2), "or2": s(or2, 2), "or_avg": s((or1 + or2) / 2, 2),
        "A": A, "B": B, "C": C, "D": D, "crude": s(crude, 2),
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
    return {"n": n, "xbar": xbar, "m2": m2, "theta": s(theta, 2), "alpha": s(alpha, 2)}

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
    # (a) T ~ U(20, 30) 分、V = 600 / T（km/h）の期待値。ln 1.5 は問題文の値
    ln15 = 0.4055
    ev = 60 * ln15
    # (b) 対応のある差の信頼区間
    A = [38.5, 42.0, 35.2, 40.8, 45.3, 36.9, 39.4, 41.6]
    B = [35.3, 40.5, 36.0, 38.4, 41.2, 36.3, 37.5, 38.9]
    d = [round(a - b, 1) for a, b in zip(A, B)]
    n = len(d)
    dbar = sum(d) / n
    ss = sum((x - dbar) ** 2 for x in d)
    s2 = ss / (n - 1)
    sd_ = r(sqrt(r(s2, 4)), 4)
    se = r(sd_ / sqrt(n), 4)
    t7, t14, t05 = 2.365, 2.145, 1.895
    half = r(t7 * se, 4)
    lo = dbar - half
    ma, mb = sum(A) / n, sum(B) / n
    ssa = sum((x - ma) ** 2 for x in A)
    ssb = sum((x - mb) ** 2 for x in B)
    sp2 = (ssa + ssb) / (2 * n - 2)
    lo_pool = (ma - mb) - t14 * sqrt(sp2 * 2 / n)
    out = {"ln15": ln15, "ev": s(ev, 2), "ev4": s(ev, 4), "w_inv": s(600 / 25, 2), "w_mid": s(25, 2),
           "w_jinv": s((30 ** 4 - 20 ** 4) / 24000, 2), "w_no10": s(600 * ln15, 1),
           "n": n, "t7": t7, "t14": t14, "t05": t05,
           "dsum": s(sum(d), 1), "dbar": s(dbar, 2), "ss": s(ss, 2), "s2": s(s2, 4), "sd": s(sd_, 4),
           "sqn": s(sqrt(n), 4), "se": s(se, 4), "half": s(half, 4), "lo": r(lo, 2), "lo4": s(lo, 4),
           "hi": s(dbar + half, 2), "lo_pool": s(lo_pool, 2), "lo_t05": s(dbar - t05 * se, 2),
           "lo_n": s(dbar - t7 * sqrt(ss / n) / sqrt(n), 2), "lo_s": s(lo, 2)}
    for i in range(n):
        out[f"a{i + 1}"] = s(A[i], 1)
        out[f"b{i + 1}"] = s(B[i], 1)
        out[f"d{i + 1}"] = s(d[i], 1)
    return out

def p06():
    # (a) ホルム法。路線 A〜E の p 値
    p = {"A": 0.019, "B": 0.003, "C": 0.300, "D": 0.011, "E": 0.023}
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
    out = {f"p{k}": v for k, v in p.items()}
    out.update({
        "W": W, "m": m, "n": n, "N": N, "N1": N + 1, "EW": s(EW, 0), "VW": s(VW, 0), "sqV": sq,
        "diff": s(W - EW, 0), "z": r(z, 2), "z_s": s(z, 2), "z4": s(z, 4),
        "Wo": sum(rank[v] for v in o), "EWo": s(n * (N + 1) / 2, 0),
        "z_cc": s((W - EW + 0.5) / sq, 2),
        "w_var": s((W - EW) / sqrt(m * n * (N + 1) / 12 * 2), 2),
    })
    for i, v in enumerate(w):
        out[f"w{i + 1}"] = s(v, 1)
    for i, v in enumerate(o):
        out[f"o{i + 1}"] = s(v, 1)
    out["rw"] = ", ".join(str(rank[v]) for v in sorted(w))
    return out

def p07():
    n, p2 = 48, 7
    k = p2 - 1
    rss2 = 900
    df2 = n - p2
    # (a) t 値から 1 変数を除いたときの決定係数
    R2, t = 0.82, 2.5
    dR2 = t ** 2 * (1 - R2) / df2
    R2r = R2 - dR2
    # (b) 外的スチューデント化残差
    e, h = 11.2, 0.25
    s2 = rss2 / df2
    rint = e / sqrt(s2 * (1 - h))
    s2i = (rss2 - e ** 2 / (1 - h)) / (df2 - 1)
    t_ext = e / sqrt(s2i * (1 - h))
    s2i_noh = (rss2 - e ** 2) / (df2 - 1)
    return {
        "n": n, "p2": p2, "k": k, "rss2": rss2, "df2": df2,
        "R2": R2, "R2s": s(R2, 3), "t": t, "t2": s(t ** 2, 2), "omR2": s(1 - R2, 2),
        "dR2": s(dR2, 5), "R2r": s(R2r, 3), "R2r5": s(R2r, 5),
        "w_noR": s(R2 - t ** 2 / df2, 3), "w_R": s(R2 - t ** 2 * R2 / df2, 3), "w_t": s(R2 - t * (1 - R2) / df2, 3),
        "e": e, "h": h, "s2": s(s2, 4), "e2": s(e ** 2, 2), "e2h": s(e ** 2 / (1 - h), 4),
        "rss_i": s(rss2 - e ** 2 / (1 - h), 4), "df3": df2 - 1, "s2i": s(s2i, 4),
        "s2ih": s(s2i * (1 - h), 4), "sqs2ih": s(sqrt(s2i * (1 - h)), 4),
        "t_ext": s(t_ext, 2), "t_ext4": s(t_ext, 4), "r_int": s(rint, 2),
        "w_noh": s(e / sqrt(s2), 2), "w_noei": s(e / sqrt(s2i_noh * (1 - h)), 2), "w_nolev": s(e / sqrt(s2i), 2),
        "ratio": s((df2 - 1) / (df2 - rint ** 2), 4), "rint4": s(rint, 4),
    }

def p08():
    rr, n = -0.48, 22
    r2 = rr ** 2
    t = rr * sqrt(n - 2) / sqrt(1 - r2)
    vx, cxy, czx, czy, vz = 1.60, -1.20, 0.80, -2.40, 0.50
    return {
        "r": rr, "n": n, "df": n - 2, "r2": s(r2, 4), "one_r2": s(1 - r2, 4), "sq1r2": s(sqrt(1 - r2), 4),
        "sqdf": s(sqrt(n - 2), 4), "t": s(t, 2), "t4": s(t, 4), "tc": 2.086,
        "w_nosq": s(rr * sqrt(n - 2), 2), "w_n1": s(rr * sqrt(n - 1) / sqrt(1 - r2), 2),
        "w_n3": s(rr * sqrt(n - 3) / sqrt(1 - r2), 2), "w_1r2": s(rr * sqrt(n - 2) / (1 - r2), 2),
        "vx": s(vx, 2), "cxy": s(cxy, 2), "czx": s(czx, 2), "czy": s(czy, 2), "vz": s(vz, 2),
        "iv": s(czy / czx, 2), "ols": s(cxy / vx, 2), "w_vx": s(czy / vx, 2), "w_vz": s(czy / vz, 2),
        "w_inv": s(czx / czy, 2), "fs": s(czx / vz, 2), "rf": s(czy / vz, 2),
    }

def p09():
    # (a) 客室数に比例する確率の復元 PPS 抽出とハンセン・ハーウィッツ推定量
    X, Nh = 12800, 400
    rooms = [40, 25, 80, 16]
    guests = [900, 600, 1700, 400]
    n = len(rooms)
    ratio = [g / x for g, x in zip(guests, rooms)]
    hh = X * sum(ratio) / n
    out = {"X": X, "Nh": Nh, "nn": n, "hh": r(hh, 0), "hh_s": s(hh, 0), "mratio": s(sum(ratio) / n, 4),
           "sratio": s(sum(ratio), 2), "w_mean": s(Nh * sum(guests) / n, 0),
           "w_rat": s(X * sum(guests) / sum(rooms), 0), "w_non": s(X * sum(ratio), 0),
           "sumg": sum(guests), "sumx": sum(rooms), "xbar_pop": s(X / Nh, 0)}
    for i in range(n):
        out[f"x{i + 1}"] = rooms[i]
        out[f"g{i + 1}"] = guests[i]
        out[f"q{i + 1}"] = s(ratio[i], 2)
        out[f"pi{i + 1}"] = s(rooms[i] / X, 6)
    # (b) フロントドア調整
    pT = 0.4
    pM = {1: 0.7, 0: 0.2}
    pY = {(1, 1): 0.7, (1, 0): 0.5, (0, 1): 0.3, (0, 0): 0.2}   # (m, t)
    inner = {m: pY[(m, 1)] * pT + pY[(m, 0)] * (1 - pT) for m in (0, 1)}
    do = {t: pM[t] * inner[1] + (1 - pM[t]) * inner[0] for t in (0, 1)}
    ace = do[1] - do[0]
    naive = (pM[1] * pY[(1, 1)] + (1 - pM[1]) * pY[(0, 1)]) - (pM[0] * pY[(1, 0)] + (1 - pM[0]) * pY[(0, 0)])
    inner_eq = {m: 0.5 * pY[(m, 1)] + 0.5 * pY[(m, 0)] for m in (0, 1)}
    ace_eq = (pM[1] - pM[0]) * (inner_eq[1] - inner_eq[0])
    inner_sw = {m: pY[(m, 1)] * (1 - pT) + pY[(m, 0)] * pT for m in (0, 1)}
    ace_sw = (pM[1] - pM[0]) * (inner_sw[1] - inner_sw[0])
    ace_own = (pM[1] - pM[0]) * (pY[(1, 1)] - pY[(0, 1)]) * 0 + (pM[1] * pY[(1, 1)] + (1 - pM[1]) * pY[(0, 1)]) \
        - (pM[0] * pY[(1, 1)] + (1 - pM[0]) * pY[(0, 1)])
    out.update({
        "pT": pT, "pT0": s(1 - pT, 1), "pM1": pM[1], "pM0": pM[0],
        "y11": pY[(1, 1)], "y10": pY[(1, 0)], "y01": pY[(0, 1)], "y00": pY[(0, 0)],
        "in1": s(inner[1], 2), "in0": s(inner[0], 2), "do1": s(do[1], 3), "do0": s(do[0], 3),
        "ace": s(ace, 3), "naive": s(naive, 3), "ace_eq": s(ace_eq, 3), "ace_sw": s(ace_sw, 3), "ace_own": s(ace_own, 3),
        "t1": s(pY[(1, 1)] * pT, 2), "t2": s(pY[(1, 0)] * (1 - pT), 2), "t3": s(pY[(0, 1)] * pT, 2), "t4": s(pY[(0, 0)] * (1 - pT), 2),
        "py1": s(pM[1] * pY[(1, 1)] + (1 - pM[1]) * pY[(0, 1)], 2), "py0": s(pM[0] * pY[(1, 0)] + (1 - pM[0]) * pY[(0, 0)], 2),
    })
    return out

def p10():
    b0, b1, b2 = -0.9, 0.15, -0.4
    x2 = 1.5
    x1a, x1b = 6, 8
    eta_a = b0 + b1 * x1a + b2 * x2
    eta_b = b0 + b1 * x1b + b2 * x2
    phi = {0.3: 0.6179, 0.6: 0.7257, 0.9: 0.8159}
    pa = 1 - phi[r(-eta_a, 1)]
    pb = 1 - phi[r(-eta_b, 1)]
    return {
        "b0": b0, "b1": b1, "b2": b2, "x2": x2, "x1a": x1a, "x1b": x1b,
        "ya": x1a * 100, "yb": x1b * 100,
        "t1a": s(b1 * x1a, 2), "t1b": s(b1 * x1b, 2), "t2": s(b2 * x2, 2),
        "eta_a": s(eta_a, 1), "eta_b": s(eta_b, 1),
        "ph3": s(phi[0.3], 4), "ph6": s(phi[0.6], 4), "ph9": s(phi[0.9], 4),
        "pa": s(pa, 4), "pb": s(pb, 4), "dp": r(pb - pa, 3), "dp4": s(pb - pa, 4), "dp_s": s(pb - pa, 3),
        "w_x": s(b1 * 2, 2),
    }

def p11():
    means = [52.0, 45.5, 43.0]
    rr, mse, a = 8, 36, 3
    N = a * rr
    dfe = N - a
    gm = sum(means) / a
    ssa = rr * sum((m - gm) ** 2 for m in means)
    sse = mse * dfe
    sst = ssa + sse
    eta2 = ssa / sst
    om2 = (ssa - (a - 1) * mse) / (sst + mse)
    return {
        "mA": s(means[0], 1), "mB": s(means[1], 1), "mC": s(means[2], 1), "rr": rr, "N": N, "mse": mse, "dfe": dfe,
        "a1": a - 1, "gm": s(gm, 4), "ssa": s(ssa, 2), "sse": s(sse, 0), "sst": s(sst, 2), "msa": s(ssa / (a - 1), 2),
        "F": s(ssa / (a - 1) / mse, 2), "num": s(ssa - (a - 1) * mse, 2), "den": s(sst + mse, 2), "amse": (a - 1) * mse,
        "om2": s(om2, 3), "om2_4": s(om2, 4), "eta2": s(eta2, 3), "w_nomse": s((ssa - (a - 1) * mse) / sst, 3),
        "w_ratio": s(ssa / sse, 3), "w_noa": s((ssa - mse) / (sst + mse), 3),
        "dev1": s(means[0] - gm, 4), "dev2": s(means[1] - gm, 4), "dev3": s(means[2] - gm, 4),
        "sqsum": s(sum((m - gm) ** 2 for m in means), 4),
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
    # (c) 数量化I類（カテゴリースコア）
    out.update({"cst": 6.0, "sd": -1.4, "ss2": 2.1, "car": 0.6, "rail": -0.2, "bus": -1.2,
                "pred": s(6.0 + 2.1 - 0.2, 1), "rng1": s(2.1 + 1.4, 1), "rng2": s(0.6 + 1.2, 1),
                "w_pred_sd": s(6.0 - 1.4 - 0.2, 1), "w_pred_sum": s(2.1 - 0.2, 1), "w_pred_car": s(6.0 + 2.1 + 0.6, 1)})
    # (c旧) RMSEA
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
    th = 0.6
    prev_f, obs = 120, 130
    err = obs - prev_f
    f1 = obs - th * err
    return {"th": th, "al": s(1 - th, 1), "pf": prev_f, "obs": obs, "err": err, "f1": s(f1, 0),
            "w_al": s(obs - (1 - th) * err, 0), "w_sign": s(obs + th * err, 0), "w_rw": s(obs, 0), "w_old": s(prev_f, 0),
            "thr": s(th * err, 0), "alr": s((1 - th) * err, 0)}

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
