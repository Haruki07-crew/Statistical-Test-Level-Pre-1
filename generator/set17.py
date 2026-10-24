"""模擬試験 第17回（set17）の数値。values() が {大問id: {名前: 値}} を返す。

題材：飲食・ホテル・サービス業の顧客と需要。
問題文に示した分布の値（Φ・χ²・F・t の点、log・べき乗の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction as Fr
from math import comb, ceil, sqrt


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
        "comp": s(1 - p0, 3), "stop3": s(stop3, 3), "s1": s(1, 0), "s2": s(1 / 2, 3), "s3": s(1 / 6, 4), "s4": s(1 / 24, 4),
        "union": s(1 - p0, 4),
    }


# ---------------------------------------------------------------- 大問2 1組の人数
def p02():
    xs = [1, 2, 3, 4, 5]
    ps = [0.10, 0.50, 0.15, 0.20, 0.05]
    mu = sum(x * p for x, p in zip(xs, ps))
    ex2 = sum(x * x * p for x, p in zip(xs, ps))
    var = ex2 - mu ** 2
    m3 = sum((x - mu) ** 3 * p for x, p in zip(xs, ps))
    skew = m3 / var ** 1.5
    # (b) 60 組の合計人数が 170 人を超える確率（連続修正）
    k, seats = 60, 170
    M, V = k * mu, k * var
    SD = sqrt(V)
    phi = {0.21: 0.5832, 1.63: 0.9484, 1.69: 0.9545, 1.75: 0.9599}
    z_cc = r((seats + 0.5 - M) / SD, 2)
    z_no = r((seats - M) / SD, 2)
    z_up = r((seats - 0.5 - M) / SD, 2)
    z_var = r((seats + 0.5 - M) / V, 2)
    for z in (z_cc, z_no, z_up, z_var):
        assert z in phi, z
    # (c) KL 情報量（表の自然対数の値を使う）
    ln = {0.25: -1.3863, 0.5: -0.6931, 0.75: -0.2877, 1.25: 0.2231, 2.0: 0.6931, 2.5: 0.9163}
    q1 = [0.2] * 5
    q2 = [0.2, 0.4, 0.2, 0.1, 0.1]
    lnr = lambda a, b: 0.0 if round(a / b, 2) == 1.0 else ln[round(a / b, 2)]
    kl = lambda q: sum(p * lnr(p, qq) for p, qq in zip(ps, q))
    kl_rev = lambda q: sum(qq * -lnr(p, qq) for p, qq in zip(ps, q))
    kl1, kl2 = kl(q1), kl(q2)
    assert kl2 < kl1
    out = {f"p{x}": s(p, 2) for x, p in zip(xs, ps)}
    out.update({f"dev{x}": s(x - mu, 1) for x in xs})
    out.update({f"c{x}": s((x - mu) ** 3, 3) for x in xs})
    out.update({f"t{x}": s((x - mu) ** 3 * p, 4) for x, p in zip(xs, ps)})
    out.update({f"q2_{x}": s(q, 1) for x, q in zip(xs, q2)})
    out.update({
        "mu": s(mu, 1), "ex2": s(ex2, 2), "mu2": s(mu ** 2, 2), "var": s(var, 2),
        "m3": s(m3, 4), "v15": s(var ** 1.5, 4), "skew": r(skew, 2), "skew_s": s(skew, 2), "skew4": s(skew, 4),
        "skew_v": s(m3 / var, 2), "sd": s(sqrt(var), 4),
        "k": k, "seats": seats, "M": s(M, 0), "V": s(V, 1), "SD": s(SD, 2), "SD4": s(SD, 4), "sp": seats + 0.5,
        "sm": seats - 0.5, "zcc": s(z_cc, 2), "zno": s(z_no, 2), "zup": s(z_up, 2), "zvar": s(z_var, 2),
        "zcc_raw": s((seats + 0.5 - M) / SD, 4), "dcc": s(seats + 0.5 - M, 1),
        "phiA": s(phi[0.21], 4), "phiB": s(phi[1.63], 4), "phiC": s(phi[1.69], 4), "phiD": s(phi[1.75], 4),
        "c_ok": s(1 - phi[z_cc], 3), "c_ok4": s(1 - phi[z_cc], 4), "c_no": s(1 - phi[z_no], 3), "c_up": s(1 - phi[z_up], 3),
        "c_var": s(1 - phi[z_var], 3), "c_comp": s(phi[z_cc], 3),
        "lnA": s(ln[0.25], 4), "lnB": s(ln[0.5], 4), "lnC": s(ln[0.75], 4), "lnD": s(ln[1.25], 4),
        "lnE": s(ln[2.0], 4), "lnF": s(ln[2.5], 4),
        "kl1": s(kl1, 3), "kl2": s(kl2, 3), "kl1_4": s(kl1, 4), "kl2_4": s(kl2, 4), "klrev2": s(kl_rev(q2), 3),
        "kl2_neg": s(-kl2, 3), "kl2_10": s(kl2 / math.log(10), 3),
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
    # (b) 2 標本の比率の z 検定（プール）
    n1, x1, n2, x2 = 400, 32, 200, 6
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    inv = 1 / n1 + 1 / n2
    se0 = sqrt(pp * (1 - pp) * inv)
    z = (p1 - p2) / se0
    z_un = (p1 - p2) / sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_pois = (p1 - p2) / sqrt(pp * inv)
    z_tot = (p1 - p2) / sqrt(pp * (1 - pp) / (n1 + n2))
    # (c) ロジスティック回帰：係数 = 対数オッズ比、標準誤差 = √(1/a+1/b+1/c+1/d)
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    lnor = 1.0338                    # log(6208/2208)（問題文に示す）
    vv = 1 / a + 1 / b + 1 / c + 1 / d
    se = sqrt(vv)
    wz = lnor / se
    return {
        "rooms": rooms, "book": book, "q": s(q, 2), "q1": s(1 - q, 2), "pw": s(pw, 6), "nsmax": book - rooms - 1,
        "c2": c2, "t0": s(t0, 4), "t1": s(t1, 4), "t2": s(t2, 4), "tsum": s(t0 + t1 + t2, 4),
        "over": r(over, 3), "over_s": s(over, 3), "over4": s(over, 4),
        "over_le1": s(pw * (t0 + t1), 3), "only2": s(pw * t2, 3),
        "n1": n1, "x1": x1, "n2": n2, "x2": x2, "p1": s(p1, 2), "p2": s(p2, 2), "dp": s(p1 - p2, 2),
        "xx": x1 + x2, "nn": n1 + n2, "pp": s(pp, 5), "qq": s(1 - pp, 5), "ppq": s(pp * (1 - pp), 6), "inv": s(inv, 4),
        "se0": s(se0, 5), "z": s(z, 2), "z_un": s(z_un, 2), "z_pois": s(z_pois, 2), "z_tot": s(z_tot, 2), "z2": s(z * z, 2),
        "a": a, "b": b, "c": c, "d": d, "ad": a * d, "bc": b * c, "or": s(a * d / (b * c), 4), "lnor": s(lnor, 4),
        "ia": s(1 / a, 5), "ib": s(1 / b, 5), "ic": s(1 / c, 5), "id": s(1 / d, 5), "vv": s(vv, 5), "se": s(se, 4),
        "wz": r(wz, 2), "wz_s": s(wz, 2), "wz4": s(wz, 4), "wz_v": s(lnor / vv, 2), "or_z": s((a * d / (b * c)) / se, 2),
    }


# ---------------------------------------------------------------- 大問4 ルームサービスの所要時間
def p04():
    n = 10
    th0, c20, c10, c95 = 5, 31.41, 18.31, 10.85
    crit = th0 * c20 / 2
    return {
        "n": n, "n1": n - 1, "n2": n - 2, "eff": s((n - 2) / n, 2), "eff_n1": s((n - 1) / n, 2), "eff_inv": s(n / (n - 2), 2),
        "eff_n12": s((n - 2) / (n - 1), 2), "eff_1": s(1, 2), "th0": th0, "c20": s(c20, 2), "c10": s(c10, 2), "c95": s(c95, 2),
        "dn": 2 * n, "crit": r(crit, 1), "crit_s": s(crit, 1), "crit3": s(crit, 3), "crit_10": s(th0 * c10 / 2, 1),
        "crit_no2": s(th0 * c20, 1), "crit_low": s(th0 * c95 / 2, 1), "mean0": n * th0,
    }


# ---------------------------------------------------------------- 大問5 2店舗の客単価
def p05():
    sig = 0.6
    phi = {0.18: 0.5714, 0.30: 0.6179, 0.60: 0.7257}
    ans = 1 - phi[0.30]
    n1, s1, n2, s2 = 13, 0.64, 16, 0.40
    F = s1 ** 2 / s2 ** 2
    f025, f05, f025b = 2.96, 2.48, 2.85
    m1, m2, t27 = 8.20, 7.90, 2.052
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
    inv = 1 / n1 + 1 / n2
    se = sqrt(sp2 * inv)
    lo = (m1 - m2) - t27 * se
    se_w = sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    sp2_n = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2)
    return {
        "sig": s(sig, 1), "hs": s(sig / 2, 2), "hs2": s(sig ** 2 / 2, 2), "sig2": s(sig ** 2, 2),
        "phiA": s(phi[0.18], 4), "phiB": s(phi[0.30], 4), "phiC": s(phi[0.60], 4),
        "ans": s(ans, 3), "w_half": s(0.5, 3), "w_sig": s(1 - phi[0.60], 3), "w_var": s(1 - phi[0.18], 3), "w_tail": s(phi[0.30], 3),
        "n1": n1, "s1": s(s1, 2), "n2": n2, "s2": s(s2, 2), "v1": s(s1 ** 2, 4), "v2": s(s2 ** 2, 4),
        "F": s(F, 2), "Fsd": s(s1 / s2, 2), "df1": n1 - 1, "df2": n2 - 1,
        "f025": s(f025, 2), "f05": s(f05, 2), "f025b": s(f025b, 2),
        "m1": s(m1, 2), "m2": s(m2, 2), "dm": s(m1 - m2, 2), "t27": s(t27, 3), "df": df,
        "ss1": s((n1 - 1) * s1 ** 2, 4), "ss2": s((n2 - 1) * s2 ** 2, 4), "ssum": s((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2, 4),
        "sp2": s(sp2, 5), "inv": s(inv, 5), "se2": s(sp2 * inv, 5),
        "se": s(se, 4), "hw": s(t27 * se, 4), "lo": r(lo, 2), "lo_s": s(lo, 2), "lo4": s(lo, 4),
        "lo_w": s((m1 - m2) - t27 * se_w, 2), "lo_z": s((m1 - m2) - 1.96 * se, 2),
        "lo_n": s((m1 - m2) - t27 * sqrt(sp2_n * inv), 2),
    }


# ---------------------------------------------------------------- 大問6 アップグレード提案の受諾率
def p06():
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


# ---------------------------------------------------------------- 大問7 ランチの店選び
def p07():
    P = [[Fr(6, 10), Fr(3, 10), Fr(1, 10)], [Fr(2, 10), Fr(5, 10), Fr(3, 10)], [Fr(3, 10), Fr(2, 10), Fr(5, 10)]]
    # πA を 1 とおき、x=πB/πA, w=πC/πA を A・B の釣り合いの式から解く
    #   (1-P_AA) = P_BA x + P_CA w,   (1-P_BB) x = P_AB + P_CB w
    a11, a22 = 1 - P[0][0], 1 - P[1][1]
    w = (a11 - P[1][0] * P[0][1] / a22) / (P[1][0] * P[2][1] / a22 + P[2][0])
    x = (P[0][1] + P[2][1] * w) / a22
    tot = 1 + x + w
    piA = 1 / tot
    out = {f"p{i}{j}": float(P[i][j]) for i in range(3) for j in range(3)}
    fr = lambda f: f"{f.numerator}/{f.denominator}"
    out.update({
        "x": fr(x), "w": fr(w), "tot": fr(tot), "piA": s(float(piA), 2), "piB": s(float(x / tot), 4), "piC": s(float(w / tot), 4),
        "ret": r(float(tot), 2), "ret_s": s(float(tot), 2), "ret4": s(float(tot), 4),
        "naive": s(1 / (1 - float(P[0][0])), 2), "inv_p": s(1 / float(P[0][0]), 2), "piA_s": s(float(piA), 4),
        "a11": s(float(a11), 1), "a22": s(float(a22), 1),
    })
    return out


# ---------------------------------------------------------------- 大問8 会員の継続期間
def p08():
    ev = [(2, 1, 10), (4, 2, 8), (6, 1, 6)]     # (時点, 解約数, 直前のリスク集合の大きさ)
    S = 1.0
    acc = 0.0
    for t, d, n in ev:
        S *= (n - d) / n
        acc += d / (n * (n - d))
    var = S ** 2 * acc
    return {
        "S": s(S, 4), "acc": s(acc, 6), "S2": s(S ** 2, 6), "var": s(var, 6), "se": s(sqrt(var), 3), "se4": s(sqrt(var), 4),
        "g1": s(1 / 90, 6), "g2": s(2 / 48, 6), "g3": s(1 / 30, 6),
        "w_var": s(var, 3), "w_binom": s(sqrt(S * (1 - S) / 10), 3), "w_S1": s(sqrt(S * acc), 3), "w_noS": s(sqrt(acc), 3),
    }


# ---------------------------------------------------------------- 大問9 テラス席の売上と気温
def p09():
    n, xb, sxx, b0, b1, sres = 20, 25.0, 250.0, 12.0, 3.0, 8.0
    x0 = 30.0
    t18, t19 = 2.101, 2.093
    yh = b0 + b1 * x0
    k = 1 / n + (x0 - xb) ** 2 / sxx
    hw = t18 * sres * sqrt(k)
    hw_pi = t18 * sres * sqrt(1 + k)
    return {
        "n": n, "xb": s(xb, 1), "sxx": s(sxx, 1), "b0": s(b0, 1), "b1": s(b1, 1), "sres": s(sres, 1), "x0": s(x0, 1),
        "t18": s(t18, 3), "t19": s(t19, 3), "yh": s(yh, 1), "dx2": s((x0 - xb) ** 2, 1), "k": s(k, 2), "sk": s(sqrt(k), 4),
        "hw": s(hw, 2), "lo": s(yh - hw, 2), "lo_pi": s(yh - hw_pi, 2), "lo_z": s(yh - 1.96 * sres * sqrt(k), 2),
        "lo_nox": s(yh - t18 * sres * sqrt(1 / n), 2), "lo_nosq": s(yh - t18 * sres * k, 2), "df": n - 2,
        "s1k": s(sqrt(1 + k), 4), "hw_pi": s(hw_pi, 2), "in": s(1 / n, 2), "dxs": s((x0 - xb) ** 2 / sxx, 2),
    }


# ---------------------------------------------------------------- 大問10 客室清掃時間
def p10():
    y = [22, 25, 18, 30, 27, 24, 35, 21, 26, 29, 41]
    tau = 0.8
    ys = sorted(y)
    n = len(y)
    k = ceil(n * tau)
    assert n * tau != int(n * tau)
    return {
        "y": ", ".join(str(v) for v in y), "ys": ", ".join(str(v) for v in ys), "n": n, "tau": s(tau, 1),
        "ntau": s(n * tau, 1), "k": k, "q": ys[k - 1], "q_lo": ys[k - 2], "q_hi": ys[k], "med": ys[n // 2],
        "mean": s(sum(y) / n, 1), "km1": k - 1, "nk": n - k, "lo_cnt": k - 1, "hi_cnt": n - k,
        "slope_lo": s((1 - tau) * (k - 1) - tau * (n - k + 1), 1), "slope_hi": s((1 - tau) * k - tau * (n - k), 1),
    }


# ---------------------------------------------------------------- 大問11 4つのデザートの評価
def p11():
    k, n = 4, 8
    N = k * n
    ssa, sse = 36.0, 84.0
    sst = ssa + sse
    dfa, dfe = k - 1, N - k
    mse = sse / dfe
    om = (ssa - dfa * mse) / (sst + mse)
    return {
        "k": k, "n": n, "N": N, "ssa": s(ssa, 1), "sse": s(sse, 1), "sst": s(sst, 1), "dfa": dfa, "dfe": dfe,
        "mse": s(mse, 1), "dfmse": s(dfa * mse, 1), "num": s(ssa - dfa * mse, 1), "den": s(sst + mse, 1),
        "om": s(om, 3), "om4": s(om, 4), "eta": s(ssa / sst, 3), "w_noden": s((ssa - dfa * mse) / sst, 3),
        "w_ratio": s(ssa / sse, 3), "w_k": s((ssa - k * mse) / (sst + mse), 3),
    }


# ---------------------------------------------------------------- 大問12 宿泊客アンケートの無回答
def p12():
    n, nr, yr, nf, yf = 500, 300, 4.2, 40, 3.6
    nm = n - nr
    wr, wm = nr / n, nm / n
    est = wr * yr + wm * yf
    return {
        "n": n, "nr": nr, "yr": s(yr, 1), "nm": nm, "nf": nf, "yf": s(yf, 1), "wr": s(wr, 1), "wm": s(wm, 1),
        "est": s(est, 2), "bias": s(yr - est, 2), "w_avg": s((yr + yf) / 2, 2),
        "w_pool": s((nr * yr + nf * yf) / (nr + nf), 2), "w_rev": s(wm * yr + wr * yf, 2), "yr2": s(yr, 2),
        "t1": s(wr * yr, 2), "t2": s(wm * yf, 2),
    }


# ---------------------------------------------------------------- 大問13 ホテルの口コミ評価の多変量解析
def p13():
    l1 = (0.70, 0.30)
    c, sn = 0.8, 0.6
    rot = (l1[0] * c + l1[1] * sn, -l1[0] * sn + l1[1] * c)
    rotT = (l1[0] * c - l1[1] * sn, l1[0] * sn + l1[1] * c)
    comm = l1[0] ** 2 + l1[1] ** 2
    # 非計量 MDS のストレス
    pairs = ["AB", "BD", "AC", "BC", "AD", "CD"]          # 非類似度の小さい順
    d = {"AB": 1.00, "AC": 1.20, "AD": 2.24, "BC": 1.56, "BD": 1.41, "CD": 2.01}
    seq = [d[p] for p in pairs]
    blocks = [[v] for v in seq]                          # 隣接違反プール法（単調回帰）
    i = 0
    while i < len(blocks) - 1:
        if sum(blocks[i]) / len(blocks[i]) > sum(blocks[i + 1]) / len(blocks[i + 1]):
            blocks[i] = blocks[i] + blocks.pop(i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    dh = [sum(b) / len(b) for b in blocks for _ in b]
    num = sum((a - b) ** 2 for a, b in zip(seq, dh))
    den = sum(a * a for a in seq)
    stress = sqrt(num / den)
    out = {
        "l11": s(l1[0], 2), "l12": s(l1[1], 2), "c": s(c, 1), "sn": s(sn, 1),
        "r1": s(rot[0], 2), "r2": s(rot[1], 2), "rt1": s(rotT[0], 2), "rt2": s(rotT[1], 2),
        "comm": s(comm, 2), "uniq": s(1 - comm, 2), "rsum": s(rot[0] + rot[1], 2),
        "e1": s(l1[0] * c, 2), "e2": s(l1[1] * sn, 2), "e3": s(l1[0] * sn, 2), "e4": s(l1[1] * c, 2),
        "r1sq": s(rot[0] ** 2, 4), "r2sq": s(rot[1] ** 2, 4), "w1": s(l1[0] * c, 2), "w2": s(l1[1] * c, 2),
        "wcomm": s((l1[0] * c) ** 2 + (l1[1] * c) ** 2, 2),
        "num": s(num, 4), "den": s(den, 4), "ratio": s(num / den, 6), "stress": r(stress, 3), "stress_s": s(stress, 3),
        "stress4": s(stress, 4), "w_nosqrt": s(num / den, 4), "w_mean": s(sqrt(num / len(seq)), 3),
    }
    for p, v, h in zip(pairs, seq, dh):
        out[f"d{p}"] = s(v, 2)
        out[f"h{p}"] = s(h, 3)
        out[f"e{p}"] = s(v - h, 3)
    return out


# ---------------------------------------------------------------- 大問14 日次予約数の残差
def p14():
    n, rr = 100, [0.15, -0.10, 0.12]
    K, pq = 3, 2
    terms = [rk ** 2 / (n - k) for k, rk in enumerate(rr, 1)]
    Q = n * (n + 2) * sum(terms)
    bp = n * sum(rk ** 2 for rk in rr)
    c = {1: 3.84, 2: 5.99, 3: 7.81}
    assert c[K - pq] < Q < c[K] and bp > c[1]
    return {
        "n": n, "r1": s(rr[0], 2), "r2": s(rr[1], 2), "r3": s(rr[2], 2), "K": K, "df": K - pq,
        "r1s": s(rr[0] ** 2, 4), "r2s": s(rr[1] ** 2, 4), "r3s": s(rr[2] ** 2, 4),
        "u1": s(terms[0], 7), "u2": s(terms[1], 7), "u3": s(terms[2], 7), "usum": s(sum(terms), 7),
        "nn2": n * (n + 2), "Q": s(Q, 2), "bp": s(bp, 2), "sr2": s(sum(rk ** 2 for rk in rr), 4),
        "c1": s(c[1], 2), "c2": s(c[2], 2), "c3": s(c[3], 2), "n1": n - 1, "n2": n - 2, "n3": n - 3,
    }


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14]
    return {f"s17p{i:02d}": fn() for i, fn in enumerate(fs, 1)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
