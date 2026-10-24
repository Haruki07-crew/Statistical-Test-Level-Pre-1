"""模擬試験 第17回（set17）の数値。values() が {大問id: {名前: 値}} を返す。

題材：飲食・ホテル・サービス業の顧客と需要。
問題文に示した分布の値（Φ・χ²・F・t の点、log・exp・べき乗の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
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


# ---------------------------------------------------------------- 大問1 クロークの傘
def p01():
    n = 4
    p0 = sum((-1) ** k / math.factorial(k) for k in range(n + 1))   # 誰も自分の傘を受け取らない
    stop3 = 1 - 1 + 1 / 2 - 1 / 6                                    # 包除を途中で止めた
    return {
        "n": n, "p0": s(p0, 3), "num": round(p0 * 24), "tot": 24, "e1": s(math.exp(-1), 3), "ind": s((1 - 1 / n) ** n, 3),
        "comp": s(1 - p0, 3), "stop3": s(stop3, 3), "union": s(1 - p0, 4),
    }


# ---------------------------------------------------------------- 大問2 居酒屋の滞在時間と1組の人数
def p02():
    # (a) ワイブル分布の四分位範囲（表の対数の値を使う）
    sc = 90
    ln75, ln25 = -0.2877, -1.3863
    q1 = sc * sqrt(-ln75)
    q3 = sc * sqrt(-ln25)
    iqr = q3 - q1
    # (b) KL 情報量
    xs = [1, 2, 3, 4, 5]
    ps = [0.10, 0.50, 0.15, 0.20, 0.05]
    ln = {0.25: -1.3863, 0.5: -0.6931, 0.75: -0.2877, 1.25: 0.2231, 2.0: 0.6931, 2.5: 0.9163}
    q1m = [0.2] * 5
    q2m = [0.2, 0.4, 0.2, 0.1, 0.1]
    lnr = lambda a, b: 0.0 if round(a / b, 2) == 1.0 else ln[round(a / b, 2)]
    kl = lambda q: sum(p * lnr(p, qq) for p, qq in zip(ps, q))
    kl_rev = lambda q: sum(qq * -lnr(p, qq) for p, qq in zip(ps, q))
    kl1, kl2 = kl(q1m), kl(q2m)
    assert kl2 < kl1
    # (c) モデル q2 への適合度検定（期待度数 5 未満を併合）
    n = 40
    obs = [4, 20, 6, 8, 2]
    exp_ = [n * q for q in q2m]
    om = obs[:3] + [obs[3] + obs[4]]
    em = exp_[:3] + [exp_[3] + exp_[4]]
    chi_m = sum((o - e) ** 2 / e for o, e in zip(om, em))
    chi_raw = sum((o - e) ** 2 / e for o, e in zip(obs, exp_))
    c3, c4 = 7.81, 9.49
    assert chi_m < c3 < chi_raw < c4
    out = {f"p{x}": s(p, 2) for x, p in zip(xs, ps)}
    out.update({f"q2_{x}": s(q, 1) for x, q in zip(xs, q2m)})
    out.update({f"o{x}": o for x, o in zip(xs, obs)})
    out.update({f"e{x}": s(e, 0) for x, e in zip(xs, exp_)})
    out.update({f"t{i}": s((o - e) ** 2 / e, 2) for i, (o, e) in enumerate(zip(om, em), 1)})
    out.update({
        "sc": sc, "ln75": s(ln75, 4), "ln25": s(ln25, 4), "ln5": s(-0.6931, 4),
        "sq1": s(sqrt(-ln75), 4), "sq3": s(sqrt(-ln25), 4), "q1": s(q1, 2), "q3": s(q3, 2),
        "iqr": r(iqr, 1), "iqr_s": s(iqr, 1), "iqr3": s(iqr, 3), "dsq": s(sqrt(-ln25) - sqrt(-ln75), 4),
        "iqr_nosq": s(sc * (-ln25 + ln75), 1), "iqr_exp": s(sc * (-ln25 + ln75), 1), "med": s(sc * sqrt(0.6931), 1),
        "lnA": s(ln[0.25], 4), "lnB": s(ln[0.5], 4), "lnC": s(ln[0.75], 4), "lnD": s(ln[1.25], 4),
        "lnE": s(ln[2.0], 4), "lnF": s(ln[2.5], 4),
        "kl1": s(kl1, 3), "kl2": s(kl2, 3), "kl1_4": s(kl1, 4), "kl2_4": s(kl2, 4), "klrev2": s(kl_rev(q2m), 3),
        "kl2_neg": s(-kl2, 3), "kl2_10": s(kl2 / math.log(10), 3),
        "n": n, "om4": om[3], "em4": s(em[3], 0), "chi_m": s(chi_m, 2), "chi_raw": s(chi_raw, 2), "c3": s(c3, 2), "c4": s(c4, 2),
        "c2": s(5.99, 2), "dfm": len(om) - 1,
    })
    return out


# ---------------------------------------------------------------- 大問3 オーバーブッキングと無断キャンセル
def p03():
    rooms, book, q = 100, 103, 0.05
    pw = 0.005625                    # 0.95^101（問題文に示す）
    c2 = comb(book, 2)
    t0 = (1 - q) ** 2
    t1 = book * q * (1 - q)
    t2 = c2 * q ** 2
    over = pw * (t0 + t1 + t2)
    # (b) ベリー・エシーンの上界
    C = 0.4748
    var1 = q * (1 - q)
    rho = q * (1 - q) * (q ** 2 + (1 - q) ** 2)
    sd1 = sqrt(var1)
    sq = sqrt(book)
    bound = C * rho / (sd1 ** 3 * sq)
    # (c) ロジスティック回帰：係数 = 対数オッズ比、標準誤差 = √(1/a+1/b+1/c+1/d)
    n1, x1, n2, x2 = 400, 32, 200, 6
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    lnor = 1.0338                    # log(6208/2208)（問題文に示す）
    vv = 1 / a + 1 / b + 1 / c + 1 / d
    se = sqrt(vv)
    wz = lnor / se
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    z_pool = (p1 - p2) / sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return {
        "rooms": rooms, "book": book, "q": s(q, 2), "q1": s(1 - q, 2), "pw": s(pw, 6), "nsmax": book - rooms - 1,
        "c2": c2, "t0": s(t0, 4), "t1": s(t1, 4), "t2": s(t2, 4), "tsum": s(t0 + t1 + t2, 4),
        "over": r(over, 3), "over_s": s(over, 3), "over4": s(over, 4),
        "over_le1": s(pw * (t0 + t1), 3), "only2": s(pw * t2, 3),
        "C": s(C, 4), "var1": s(var1, 4), "pq2": s(q ** 2 + (1 - q) ** 2, 4), "rho": s(rho, 7), "sd1": s(sd1, 6),
        "sd3": s(sd1 ** 3, 7), "sq": s(sq, 4), "den": s(sd1 ** 3 * sq, 6), "Crho": s(C * rho, 7),
        "bound": s(bound, 3), "bound4": s(bound, 4), "b_nosq": s(C * rho / sd1 ** 3, 3), "b_var": s(C * rho / (var1 * sq), 3),
        "b_sd": s(C * rho / (sd1 * sq), 3), "b_crho": s(C * rho, 3),
        "n1": n1, "x1": x1, "n2": n2, "x2": x2,
        "a": a, "b": b, "c": c, "d": d, "ad": a * d, "bc": b * c, "or": s(a * d / (b * c), 4), "lnor": s(lnor, 4),
        "ia": s(1 / a, 5), "ib": s(1 / b, 5), "ic": s(1 / c, 5), "id": s(1 / d, 5), "vv": s(vv, 5), "se": s(se, 4),
        "wz": r(wz, 2), "wz_s": s(wz, 2), "wz4": s(wz, 4), "wz_v": s(lnor / vv, 2), "or_z": s((a * d / (b * c)) / se, 2),
        "z_pool": s(z_pool, 2),
    }


# ---------------------------------------------------------------- 大問4 客単価
def p04():
    sig = 0.6
    phi = {0.18: 0.5714, 0.30: 0.6179, 0.35: 0.6368, 0.60: 0.7257, 1.75: 0.9599}
    ans = 1 - phi[0.30]
    # (b) 母標準偏差既知の z 検定
    n, xb, mu0 = 25, 8.21, 8.00
    se = sig / sqrt(n)
    z = (xb - mu0) / se
    assert round(z, 2) == 1.75
    p2 = 2 * (1 - phi[1.75])
    # (c) 1 標本 t 区間
    m, xm, s2, t9 = 10, 7.95, 0.30, 2.262
    se_t = sqrt(s2 / m)
    lo = xm - t9 * se_t
    return {
        "sig": s(sig, 1), "hs": s(sig / 2, 2), "hs2": s(sig ** 2 / 2, 2),
        "phiA": s(phi[0.18], 4), "phiB": s(phi[0.30], 4), "phiC": s(phi[0.60], 4), "phiD": s(phi[0.35], 4), "phiE": s(phi[1.75], 4),
        "ans": s(ans, 3), "w_half": s(0.5, 3), "w_sig": s(1 - phi[0.60], 3), "w_var": s(1 - phi[0.18], 3), "w_tail": s(phi[0.30], 3),
        "n": n, "xb": s(xb, 2), "mu0": s(mu0, 2), "dx": s(xb - mu0, 2), "se": s(se, 2), "z": s(z, 2), "p2": s(p2, 3), "p1": s(p2 / 2, 3),
        "z_nos": s((xb - mu0) / sig, 2), "p_nos": s(2 * (1 - phi[0.35]), 3), "z_n": s((xb - mu0) / (sig / n), 2),
        "m": m, "xm": s(xm, 2), "s2": s(s2, 2), "t9": s(t9, 3), "s2m": s(s2 / m, 3), "se_t": s(se_t, 4), "hw": s(t9 * se_t, 4),
        "lo": r(lo, 2), "lo_s": s(lo, 2), "lo4": s(lo, 4), "lo_z": s(xm - 1.96 * se_t, 2), "lo_s2": s(xm - t9 * s2 / sqrt(m), 2),
        "lo_nosq": s(xm - t9 * sqrt(s2) / m, 2), "df": m - 1,
    }


# ---------------------------------------------------------------- 大問5 アップグレード提案の受諾率
def p05():
    a0, b0 = 1, 2
    m, y = 20, 7
    a1, b1 = a0 + y, b0 + m - y
    mode_th = a1 / (a1 + b1)          # θ の尺度での最頻値を p に直した値
    mode_p = (a1 - 1) / (a1 + b1 - 2)
    return {
        "a0": a0, "b0": b0, "m": m, "y": y, "my": m - y, "a1": a1, "b1": b1, "ab": a1 + b1,
        "mth": r(mode_th, 3), "mth_s": s(mode_th, 3), "mth4": s(mode_th, 4),
        "mp": s(mode_p, 3), "a1m": a1 - 1, "abm": a1 + b1 - 2, "mle": s(y / m, 3),
    }


# ---------------------------------------------------------------- 大問6 ランチの店選び
def p06():
    P = [[Fr(6, 10), Fr(3, 10), Fr(1, 10)], [Fr(2, 10), Fr(5, 10), Fr(3, 10)], [Fr(3, 10), Fr(2, 10), Fr(5, 10)]]
    a11, a22 = 1 - P[0][0], 1 - P[1][1]
    w = (a11 - P[1][0] * P[0][1] / a22) / (P[1][0] * P[2][1] / a22 + P[2][0])
    x = (P[0][1] + P[2][1] * w) / a22
    tot = 1 + x + w
    piA = 1 / tot
    out = {f"p{i}{j}": float(P[i][j]) for i in range(3) for j in range(3)}
    fr = lambda f: f"{f.numerator}/{f.denominator}"
    out.update({
        "x": fr(x), "w": fr(w), "tot": fr(tot), "piA": s(float(piA), 2),
        "ret": r(float(tot), 2), "ret_s": s(float(tot), 2), "ret4": s(float(tot), 4),
        "naive": s(1 / (1 - float(P[0][0])), 2), "inv_p": s(1 / float(P[0][0]), 2),
        "a11": s(float(a11), 1), "a22": s(float(a22), 1),
    })
    return out


# ---------------------------------------------------------------- 大問7 宿泊客アンケート
def p07():
    # (a) 平均値代入
    n, nobs, s2 = 50, 40, 4.0
    v_imp = (nobs - 1) * s2 / (n - 1)
    # (b) 単純無作為抽出の総計の標準誤差
    N, m, yb, sd = 2000, 100, 38.0, 12.0
    fpc = 1 - m / N
    se_mean = sqrt(fpc * sd ** 2 / m)
    se_tot = N * se_mean
    return {
        "n": n, "nobs": nobs, "nmis": n - nobs, "s2": s(s2, 1), "v_imp": r(v_imp, 2), "v_imp_s": s(v_imp, 2), "v_imp4": s(v_imp, 4),
        "ss": s((nobs - 1) * s2, 1), "nobs1": nobs - 1, "n1": n - 1, "ratio": s((nobs - 1) / (n - 1), 4),
        "v_nobs": s(nobs * s2 / n, 2),
        "N": N, "m": m, "yb": s(yb, 1), "sd": s(sd, 1), "sd2": s(sd ** 2, 1), "tot": s(N * yb, 0), "fpc": s(fpc, 2), "v_in": s(fpc * sd ** 2 / m, 3),
        "se_mean": s(se_mean, 4), "se_tot": s(se_tot, 0), "se_nofpc": s(N * sd / sqrt(m), 0), "se_m2": s(se_mean, 2),
        "se_fpcsd": s(N * sd / sqrt(m) * fpc, 0), "w_ns": s(N * sd / m, 0),
    }


# ---------------------------------------------------------------- 大問8 客室需要の両対数回帰
def p08():
    n, k = 52, 2
    b, se, t49 = -1.35, 0.14, 2.010
    t = (b + 1) / se
    return {
        "n": n, "k": k, "df": n - k - 1, "b": s(b, 2), "se": s(se, 2), "t49": s(t49, 3), "t": s(t, 2),
        "t0": s(b / se, 2), "tabs": s(abs(t), 2), "bp1": s(b + 1, 2), "pct": s(-b, 2),
    }


# ---------------------------------------------------------------- 大問9 客室稼働率の分析
def p09():
    y = [40, 60, 80, 50, 44, 64, 84, 54]
    num = y[0] / 2 + y[1] + y[2] + y[3] + y[4] / 2
    trend = num / 4
    m14, m25 = sum(y[0:4]) / 4, sum(y[1:5]) / 4
    assert abs((m14 + m25) / 2 - trend) < 1e-12
    ratio = y[2] / trend
    # (b) ニューラルネットワーク
    x1, x2 = 1.0, 1
    W = [(0.5, 1.0, -2.0), (-1.0, 0.5, 3.0)]
    v0, v = 2.0, (1.5, 2.0)
    pre = [w0 + w1 * x1 + w2 * x2 for w0, w1, w2 in W]
    h = [max(0.0, u) for u in pre]
    yhat = v0 + v[0] * h[0] + v[1] * h[1]
    y_norelu = v0 + v[0] * pre[0] + v[1] * pre[1]
    y_nob = v[0] * h[0] + v[1] * h[1]
    y_nohb = v0 + v[0] * max(0, W[0][1] * x1 + W[0][2] * x2) + v[1] * max(0, W[1][1] * x1 + W[1][2] * x2)
    return {
        **{f"y{i}": v for i, v in enumerate(y, 1)}, "num": s(num, 1), "trend": s(trend, 2), "m14": s(m14, 2), "m25": s(m25, 2),
        "ratio": s(ratio, 2), "ratio4": s(ratio, 4), "pct": s((ratio - 1) * 100, 0), "w_simple": s(y[2] / m25, 2),
        "w_prev": s(y[2] / m14, 2), "w_inv": s(trend / y[2], 2), "w_diff": s(y[2] - trend, 2),
        "x1": s(x1, 1), "x2": x2, "w10": s(W[0][0], 1), "w11": s(W[0][1], 1), "w12": s(W[0][2], 1),
        "w20": s(W[1][0], 1), "w21": s(W[1][1], 1), "w22": s(W[1][2], 1), "v0": s(v0, 1), "v1": s(v[0], 1), "v2": s(v[1], 1),
        "u1": s(pre[0], 1), "u2": s(pre[1], 1), "h1": s(h[0], 1), "h2": s(h[1], 1),
        "y": s(yhat, 2), "y_norelu": s(y_norelu, 2), "y_nob": s(y_nob, 2), "y_nohb": s(y_nohb, 2), "y_bias": s(v0, 2),
    }


# ---------------------------------------------------------------- 大問10 1組あたりの宿泊数
def p10():
    xbar = 2.0
    cand = {1.00: 0.3679, 1.20: 0.3012, 1.59: 0.2039, 1.80: 0.1653, 2.00: 0.1353}
    g = {lam: lam / (1 - e) for lam, e in cand.items()}
    best = min(g, key=lambda lam: abs(g[lam] - xbar))
    assert best == 1.59
    out = {"xbar": s(xbar, 1), "n": 50, "tot": 100}
    for i, (lam, e) in enumerate(cand.items(), 1):
        out[f"l{i}"] = s(lam, 2)
        out[f"e{i}"] = s(e, 4)
        out[f"g{i}"] = s(g[lam], 3)
    return out


# ---------------------------------------------------------------- 大問11 ルームサービスの所要時間
def p11():
    mu0, sig, n, mu1 = 30.0, 6.0, 16, 30.9
    se = sig / sqrt(n)
    dl = (mu1 - mu0) / se
    phi = {0.60: 0.7257, 1.36: 0.9131, 2.56: 0.9948}
    up = 1 - phi[round(1.96 - dl, 2)]
    lo = 1 - phi[round(1.96 + dl, 2)]
    pw = up + lo
    return {
        "mu0": s(mu0, 1), "sig": s(sig, 1), "n": n, "mu1": s(mu1, 1), "se": s(se, 1), "dl": s(dl, 2),
        "za": s(1.96 - dl, 2), "zb": s(1.96 + dl, 2), "phiA": s(phi[0.60], 4), "phiB": s(phi[1.36], 4), "phiC": s(phi[2.56], 4),
        "up": s(up, 4), "lo": s(lo, 4), "pw": s(pw, 3), "pw4": s(pw, 4), "w_one": s(up, 3), "w_dl": s(1 - phi[0.60], 3),
        "w_beta": s(1 - pw, 3), "w_alpha": s(0.05, 3),
    }


# ---------------------------------------------------------------- 大問12 焼き菓子の L8 直交表実験
def p12():
    ss = {1: 40.0, 2: 12.0, 3: 2.0, 4: 30.0, 5: 20.0, 6: 1.5, 7: 0.5}
    col_AC = 1 ^ 4
    col_AB = 1 ^ 2
    col_BC = 2 ^ 4
    assert (col_AC, col_AB, col_BC) == (5, 3, 6)
    sse = ss[6] + ss[7]
    mse = sse / 2
    F = ss[col_AC] / mse
    f12, f13, f16 = 18.51, 10.13, 5.99
    assert F > f12
    return {
        **{f"ss{k}": s(v, 1) for k, v in ss.items()}, "sse": s(sse, 1), "mse": s(mse, 1), "F": s(F, 1),
        "F_nodf": s(ss[col_AC] / sse, 1), "F_AB": s(ss[col_AB] / mse, 1), "F_BC": s(ss[col_BC] / mse, 1),
        "f12": s(f12, 2), "f13": s(f13, 2), "f16": s(f16, 2),
    }


# ---------------------------------------------------------------- 大問13 ホテルの口コミ評価の多変量解析
def p13():
    A = [(0.58, 0.56), (0.56, 0.42), (-0.28, 0.54)]
    B = [(0.80, 0.10), (0.70, 0.00), (0.10, 0.60)]
    p = 3

    def vm(L):
        tot = 0.0
        for j in range(2):
            sq = [row[j] ** 2 for row in L]
            mean = sum(sq) / p
            tot += sum(x * x for x in sq) / p - mean ** 2
        return tot

    def vm_nosq(L):          # 2 乗せずに負荷量そのものの分散を足した（誤り）
        tot = 0.0
        for j in range(2):
            col = [row[j] for row in L]
            mean = sum(col) / p
            tot += sum((x - mean) ** 2 for x in col) / p
        return tot

    vA, vB = vm(A), vm(B)
    assert vB > vA
    out = {}
    for nm, L in (("a", A), ("b", B)):
        for i, row in enumerate(L, 1):
            for j, v in enumerate(row, 1):
                out[f"{nm}{i}{j}"] = s(v, 2)
                out[f"{nm}{i}{j}q"] = s(v * v, 4)
    sqB1 = [row[0] ** 2 for row in B]
    sqB2 = [row[1] ** 2 for row in B]
    mB1, mB2 = sum(sqB1) / p, sum(sqB2) / p
    vB1 = sum(x * x for x in sqB1) / p - mB1 ** 2
    vB2 = sum(x * x for x in sqB2) / p - mB2 ** 2
    out.update({
        "vA": s(vA, 3), "vB": s(vB, 3), "vB4": s(vB, 4), "vA4": s(vA, 4), "vB_nop": s(vB * p, 3), "vB_nosq": s(vm_nosq(B), 3),
        "mB1": s(mB1, 4), "mB2": s(mB2, 4), "vB1": s(vB1, 4), "vB2": s(vB2, 4), "comm1": s(0.58 ** 2 + 0.56 ** 2, 2),
    })
    # (c) 多重対応分析
    J, Q = 3 + 4 + 2, 3
    out.update({"J": J, "Q": Q, "inert": s((J - Q) / Q, 2), "dim": J - Q, "w_jq": s(J / Q, 2), "w_jmq": s(J - Q, 2),
                "w_dimj": J - 1, "w_q": s(1 / Q, 2)})
    return out


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s17p{i:02d}": fn() for i, fn in enumerate(fs, 1)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
