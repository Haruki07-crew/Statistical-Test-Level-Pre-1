"""模擬試験 第18回（set18）の数値。values() が {大問id: {名前: 値}} を返す。

題材：生物学・遺伝・生態調査・水産。
問題文に示した分布の値（χ²・F・t の点、Φ・φ、e^x・ln・cos・arctan など）は、ここでも同じ丸めた値を使って答えを計算する。
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


def distinct(vals, d):
    assert len({s(v, d) for v in vals}) == len(vals), vals


def p01():
    # 共優性の遺伝子座：遺伝子型の数
    nAA, nAa, naa = 50, 80, 70
    n = nAA + nAa + naa
    cntA = 2 * nAA + nAa
    p = cntA / (2 * n)
    q = 1 - p
    # (a) 誤答の値
    p_ind = (nAA + nAa) / n
    p_hom = sqrt(nAA / n)
    p_AA = nAA / n
    # (b) ハーディ・ワインベルグ平衡の尤度比検定（G 統計量）
    eAA, eAa, eaa = n * p * p, n * 2 * p * q, n * q * q
    lnv = {"l1": 0.2107, "l2": -0.2131, "l3": 0.1459}  # ln(50/40.5), ln(80/99), ln(70/60.5)
    half = nAA * lnv["l1"] + nAa * lnv["l2"] + naa * lnv["l3"]
    G = 2 * half
    chi = {"c11": 3.84, "c12": 6.63, "c13": 7.88, "c21": 5.99, "c22": 9.21, "c23": 10.60}
    assert chi["c12"] < G < chi["c13"] and G < chi["c22"], G
    eM = [n / 4, n / 2, n / 4]
    G_m = 2 * sum(o * math.log(o / e) for o, e in zip((nAA, nAa, naa), eM))
    # (c) 優性の遺伝子座の EM アルゴリズム 1 回
    nD, nR = 160, 40
    nn = nD + nR
    p0 = 0.5
    q0 = 1 - p0
    den = p0 ** 2 + 2 * p0 * q0
    eBB = nD * p0 ** 2 / den
    eBb = nD * 2 * p0 * q0 / den
    p1 = (2 * eBB + eBb) / (2 * nn)
    q_mle = sqrt(nR / nn)
    return {
        "nAA": nAA, "nAa": nAa, "naa": naa, "n": n, "n2": 2 * n, "cntA": cntA, "cnta": nAa + 2 * naa,
        "p": s(p, 2), "q": s(q, 2), "p_ind": s(p_ind, 2), "nind": nAA + nAa, "p_hom": s(p_hom, 2), "p_AA": s(p_AA, 2),
        "eAA": s(eAA, 1), "eAa": s(eAa, 1), "eaa": s(eaa, 1), "p2": s(p * p, 4), "pq2": s(2 * p * q, 4), "q2": s(q * q, 4),
        "l1": s(lnv["l1"], 4), "l2": s(lnv["l2"], 4), "l3": s(lnv["l3"], 4),
        "t1": s(nAA * lnv["l1"], 3), "t2": s(nAa * lnv["l2"], 3), "t3": s(naa * lnv["l3"], 3),
        "half": s(half, 2), "G": s(G, 2), "Gm": s(G_m, 2),
        **{k: s(v, 2) for k, v in chi.items()},
        "nD": nD, "nR": nR, "nn": nn, "nn2": 2 * nn, "p0": s(p0, 1), "pp0": s(p0 ** 2, 2), "pq0": s(2 * p0 * q0, 2),
        "den0": s(den, 2), "eBB": s(eBB, 2), "eBb": s(eBb, 2), "eBB_f": "160/3", "eBb_f": "320/3",
        "p1": r(p1, 3), "p1_s": s(p1, 3), "p1_4": s(p1, 4), "two_eBB": s(2 * eBB, 2), "num1": s(2 * eBB + eBb, 2),
        "q_mle": s(q_mle, 4), "p_mle": s(1 - q_mle, 4),
        "p1_wrong": s((eBB + eBb) / nn, 3),
    }


def p02():
    # (a) コーシー分布の種子散布：平均の分布
    g, n = 10, 25
    at = {5: 1.3734, 25: 1.5308, 0.04: 0.0400}
    pi = 3.1416
    ok = 0.5                      # (2/π)·arctan(1) = (2/π)(π/4)
    w_sqrt = 2 / pi * at[5]       # 尺度 g/√n とした
    w_n = 2 / pi * at[25]         # 尺度 g/n とした
    w_big = 2 / pi * at[0.04]     # 尺度 g·n とした
    w_one = 0.25                  # 片側だけ
    distinct([ok, w_sqrt, w_n, w_big, w_one], 3)
    # (b) ワイブル分布の寿命の四分位範囲
    m, eta = 2, 40
    l1, l3 = 0.2877, 1.3863   # ln(4/3), ln 4
    q1 = eta * sqrt(l1)
    q3 = eta * sqrt(l3)
    iqr = q3 - q1
    med = eta * sqrt(0.6931)
    return {
        "g": g, "n": n, "pi": s(pi, 4), "a5": s(at[5], 4), "a25": s(at[25], 4), "a004": s(at[0.04], 4),
        "ok": s(ok, 3), "w_sqrt": s(w_sqrt, 3), "w_n": s(w_n, 3), "w_big": s(w_big, 3), "w_one": s(w_one, 3),
        "gs": s(g / sqrt(n), 0), "gn": s(g / n, 1), "gN": g * n,
        "m": m, "eta": eta, "l1": s(l1, 4), "l3": s(l3, 4), "sl1": s(sqrt(l1), 4), "sl3": s(sqrt(l3), 4),
        "q1": s(q1, 2), "q3": s(q3, 2), "iqr": r(iqr, 1), "iqr_s": s(iqr, 1), "iqr2": s(iqr, 2),
        "med": s(med, 1), "ln2": "0.6931", "iqr_exp": s(eta * (l3 - l1), 1),
    }


def p03():
    # (a) ボックス・コックス変換（λ=0.5）で正規 N(6, 2^2)
    lam, mu, sg = 0.5, 6.0, 2.0
    phi = {0.5: 0.6915, 1.0: 0.8413, 2.0: 0.9772}
    med = (1 + lam * mu) ** 2
    w0 = 25
    z0 = (sqrt(w0) - 1) / lam
    zs = (z0 - mu) / sg
    assert zs in phi
    p_ok = 1 - phi[zs]
    med_nl = (mu + 1) ** 2            # λ を掛け忘れた
    z_nl = sqrt(w0) - 1               # 閾値も λ で割り忘れた
    p_nl = phi[(mu - z_nl) / sg]      # P(Z>4) = Φ(1)
    med_n1 = (lam * mu) ** 2          # +1 を忘れた
    z_n1 = sqrt(w0) / lam
    p_n1 = 1 - phi[(z_n1 - mu) / sg]
    p_var = 1 - phi[(z0 - mu) / sg ** 2]
    # (b) 予測区間の係数
    n = 10
    t9, t10, z = 2.262, 2.228, 1.960
    k_ok = t9 * sqrt(1 + 1 / n)
    ks = [k_ok, t9, z * sqrt(1 + 1 / n), t10 * sqrt(1 + 1 / n), t9 / sqrt(n)]
    distinct(ks, 2)
    # (c) 等分散の F 検定
    n1, v1, n2, v2 = 10, 4.8, 13, 1.6
    F = v1 / v2
    f = {"f05": 2.80, "f025": 3.44, "f025r": 3.87}
    assert f["f05"] < F < f["f025"]
    return {
        "lam": lam, "mu": s(mu, 0), "sg": s(sg, 0), "var": s(sg ** 2, 0),
        "ph05": s(phi[0.5], 4), "ph1": s(phi[1.0], 4), "ph2": s(phi[2.0], 4),
        "med": s(med, 1), "sqmed": s(1 + lam * mu, 0), "w0": w0, "z0": s(z0, 0), "zs": s(zs, 1),
        "p_ok": s(p_ok, 3), "p_ok4": s(p_ok, 4), "med_nl": s(med_nl, 1), "p_nl": s(p_nl, 3), "med_n1": s(med_n1, 1),
        "p_n1": s(p_n1, 3), "p_var": s(p_var, 3), "p_tail": s(phi[zs], 3),
        "n": n, "n1df": n - 1, "t9": s(t9, 3), "t10": s(t10, 3), "z": s(z, 3), "fac": s(sqrt(1 + 1 / n), 4),
        "k_ok": s(k_ok, 2), "k_ok4": s(k_ok, 4), "k_t": s(t9, 2), "k_z": s(ks[2], 2), "k_10": s(ks[3], 2), "k_m": s(ks[4], 2),
        "m1": n1, "v1": s(v1, 1), "m2": n2, "v2": s(v2, 1), "d1": n1 - 1, "d2": n2 - 1,
        "F": s(F, 2), "Fsd": s(sqrt(F), 2),
        **{k: s(v, 2) for k, v in f.items()},
    }


def p04():
    # (a) ベリー・エシーンの上界
    nb, pb, C = 400, 0.8, 0.4748
    rho = pb * (1 - pb) * (pb ** 2 + (1 - pb) ** 2)
    v = pb * (1 - pb)
    sig3 = v ** 1.5
    be = C * rho / (sig3 * sqrt(nb))
    be_s2 = C * rho / (v * sqrt(nb))
    be_r = C * v / (sig3 * sqrt(nb))
    be_nos = C * rho / sig3
    be_n = C * rho / (sig3 * nb)
    distinct([be, be_s2, be_r, be_nos, be_n], 3)
    # (b) ロジット尺度の信頼区間
    n, x = 150, 120
    p = x / n
    se_logit = 1 / sqrt(n * p * (1 - p))
    se_p = sqrt(p * (1 - p) / n)
    ln4 = 1.3863
    se4 = r(se_logit, 4)
    lo, hi = r(ln4 - 1.96 * se4, 3), r(ln4 + 1.96 * se4, 3)
    lo9, hi9 = r(ln4 - 1.645 * se4, 3), r(ln4 + 1.645 * se4, 3)
    lon, hin = r(ln4 - 1.96 * se_p, 3), r(ln4 + 1.96 * se_p, 3)
    em = {0.986: 0.3731, 1.051: 0.3496, 1.322: 0.2666, 1.450: 0.2346, 1.722: 0.1787, 1.786: 0.1676}
    for zz in (lo, hi, lo9, hi9, lon, hin):
        assert zz in em, zz
    inv = lambda zz: 1 / (1 + em[zz])
    ci = lambda a, b: f"({s(a, 3)},\\ {s(b, 3)})"
    return {
        "nb": nb, "pb": pb, "qb": s(1 - pb, 1), "C": C, "rho": s(rho, 4), "v": s(v, 2), "sig3": s(sig3, 3),
        "pb2": s(pb ** 2, 2), "qb2": s((1 - pb) ** 2, 2), "sq": s(sqrt(nb), 0), "ratio": s(rho / sig3, 3),
        "be": s(be, 3), "be4": s(be, 4), "be_s2": s(be_s2, 3), "be_r": s(be_r, 3), "be_nos": s(be_nos, 3), "be_n": s(be_n, 3),
        "n": n, "x": x, "p": s(p, 1), "npq": s(n * p * (1 - p), 0),
        "se4": s(se4, 4), "se_p": s(se_p, 4),
        "ln4": s(ln4, 4), "hw": s(1.96 * se4, 4), "lo": s(lo, 3), "hi": s(hi, 3),
        "e1": s(em[0.986], 4), "e2": s(em[1.051], 4), "e3": s(em[1.322], 4), "e4": s(em[1.450], 4),
        "e5": s(em[1.722], 4), "e6": s(em[1.786], 4),
        "pl": s(inv(lo), 4), "pu": s(inv(hi), 4),
        "ci_ok": ci(inv(lo), inv(hi)), "ci_90": ci(inv(lo9), inv(hi9)), "ci_naive": ci(inv(lon), inv(hin)),
        "ci_wald": ci(p - 1.96 * se_p, p + 1.96 * se_p), "ci_flip": ci(1 - inv(hi), 1 - inv(lo)),
    }


def p05():
    # (a) ゼロ切断ポアソン
    lam, e = 1.5, 0.2231
    ok = lam / (1 - e)
    vals = [lam * (1 - e), lam, ok, lam + 1, lam / e]
    distinct(vals, 2)
    # (b) 系統抽出
    y = [5, 8, 12, 6, 9, 11, 7, 10, 13, 4, 9, 14]
    N, k = len(y), 3
    nsamp = N // k
    means = [sum(y[j::k]) / nsamp for j in range(k)]
    Y = sum(y) / N
    v_sys = sum((m - Y) ** 2 for m in means) / k
    S2 = sum((v - Y) ** 2 for v in y) / (N - 1)
    v_srs = (1 - nsamp / N) * S2 / nsamp
    out = {
        "lam": lam, "e": s(e, 4), "e1": s(1 - e, 4), "ok": s(ok, 2), "ok4": s(ok, 4),
        "w_mul": s(vals[0], 2), "w_lam": s(lam, 2), "w_p1": s(lam + 1, 2), "w_div": s(lam / e, 2),
        "N": N, "k": k, "ns": nsamp, "Y": s(Y, 0),
        "m1": s(means[0], 1), "m2": s(means[1], 1), "m3": s(means[2], 1),
        "d1": s((means[0] - Y) ** 2, 2), "d2": s((means[1] - Y) ** 2, 2), "d3": s((means[2] - Y) ** 2, 2),
        "vsys": r(v_sys, 2), "vsys_s": s(v_sys, 2), "vsys4": s(v_sys, 4), "S2": s(S2, 0), "vsrs": s(v_srs, 2),
        "vsys_k1": s(sum((m - Y) ** 2 for m in means) / (k - 1), 2), "sdsys": s(sqrt(v_sys), 2),
        "ss": s(sum((v - Y) ** 2 for v in y), 0),
    }
    for i, v in enumerate(y):
        out[f"y{i}"] = v
    return out


def p06():
    n, b, se, b0 = 25, 3.24, 0.10, 3
    t = (b - b0) / se
    tc = {"t23": 2.069, "t23h": 2.807, "t24": 2.064}
    assert tc["t23"] < t < tc["t23h"]
    # (b) 正規 Q-Q プロット
    nq, i = 10, 9
    zq = {0.80: 0.842, 0.818: 0.908, 0.85: 1.036, 0.90: 1.282}
    p_ok = (i - 0.5) / nq
    assert p_ok in zq
    # (c) 折れ線回帰
    c0, c1, c2, knot, t5 = 8.0, 12.0, -9.0, 3, 5
    slope = c1 + c2
    pred = c0 + c1 * t5 + c2 * (t5 - knot)
    pred_nok = c0 + c1 * t5
    pred_t = c0 + c1 * t5 + c2 * t5
    pred_sign = c0 + c1 * t5 - c2 * (t5 - knot)
    return {
        "n": n, "b": s(b, 2), "se": s(se, 2), "b0": b0, "df": n - 2, "t": s(t, 2), "t0": s(b / se, 1),
        "t_nose": s(b - b0, 2), **{k: s(v, 3) for k, v in tc.items()},
        "nq": nq, "i": i, "pq": s(p_ok, 2), "zq": s(zq[0.85], 2),
        "z80": s(zq[0.80], 3), "z818": s(zq[0.818], 3), "z85": s(zq[0.85], 3), "z90": s(zq[0.90], 3),
        "zq_n": s(zq[0.90], 2), "zq_n1": s(zq[0.818], 2), "pin": s(i / nq, 2), "pin1": s(i / (nq + 1), 3), "res9": "1.90",
        "c0": s(c0, 1), "c1": s(c1, 1), "c2": s(c2, 1), "mc2": s(-c2, 1), "knot": knot, "t5": t5,
        "slope": s(slope, 1), "pred": s(pred, 1), "pred_nok": s(pred_nok, 1), "pred_t": s(pred_t, 1),
        "pred_sign": s(pred_sign, 1), "slope_sign": s(c1 - c2, 1),
    }


def p07():
    mu, sg = 2.0, 4.0
    Ph, ph = 0.6915, 0.3521
    z = mu / sg
    ey = Ph * mu + sg * ph
    pos = mu + sg * ph / Ph
    vals = [Ph * mu, mu, ey, pos, Ph * mu + ph]
    distinct(vals, 2)
    return {
        "mu": s(mu, 1), "sg": s(sg, 1), "z": s(z, 1), "Ph": s(Ph, 4), "ph": s(ph, 4), "p0": s(1 - Ph, 4),
        "t1": s(Ph * mu, 3), "t2": s(sg * ph, 4), "ey": s(ey, 2), "ey4": s(ey, 4),
        "w_a": s(Ph * mu, 2), "w_mu": s(mu, 2), "pos": s(pos, 2), "w_noS": s(vals[4], 2), "mills": s(ph / Ph, 4),
    }


def p08():
    r_, a, b = 3, 2, 3
    ss = {"blk": 6.0, "A": 30.0, "E1": 4.0, "B": 24.0, "AB": 3.0, "E2": 12.0}
    df = {"blk": r_ - 1, "A": a - 1, "E1": (r_ - 1) * (a - 1), "B": b - 1, "AB": (a - 1) * (b - 1), "E2": a * (r_ - 1) * (b - 1)}
    assert sum(df.values()) == r_ * a * b - 1
    msE1 = ss["E1"] / df["E1"]
    msE2 = ss["E2"] / df["E2"]
    F = ss["A"] / df["A"] / msE1
    F2 = ss["A"] / df["A"] / msE2
    Fp = ss["A"] / ((ss["E1"] + ss["E2"]) / (df["E1"] + df["E2"]))
    fc = {"f2": 18.51, "f8": 5.32, "f10": 4.96}
    assert F < fc["f2"] and F2 > fc["f8"] and Fp > fc["f10"]
    return {
        "r": r_, "a": a, "b": b, "tot": s(sum(ss.values()), 1), "dft": r_ * a * b - 1,
        **{f"ss_{k}": s(v, 1) for k, v in ss.items()}, **{f"df_{k}": v for k, v in df.items()},
        "msA": s(ss["A"], 1), "msE1": s(msE1, 1), "msE2": s(msE2, 1), "F": s(F, 1), "F2": s(F2, 1), "Fp": s(Fp, 2),
        "Fraw": s(ss["A"] / ss["E1"], 1), "pool": s((ss["E1"] + ss["E2"]) / (df["E1"] + df["E2"]), 1),
        **{k: s(v, 2) for k, v in fc.items()},
    }


def p09():
    # 河川規模（大・小）で層別した魚道設置の観察データ
    n11, n10, y11, y10 = 30, 10, 50.0, 30.0   # 大：設置あり・なし、平均遡上指数
    n01, n00, y01, y00 = 10, 50, 18.0, 10.0   # 小
    N = n11 + n10 + n01 + n00
    d1, d0 = y11 - y10, y01 - y00
    w1 = (n11 + n10) / N
    ate = w1 * d1 + (1 - w1) * d0
    t1 = n11 + n01
    att = n11 / t1 * d1 + n01 / t1 * d0
    c0 = n10 + n00
    atc = n10 / c0 * d1 + n00 / c0 * d0
    naive = (n11 * y11 + n01 * y01) / t1 - (n10 * y10 + n00 * y00) / c0
    distinct([ate, att, atc, naive], 1)
    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00, "y11": s(y11, 1), "y10": s(y10, 1), "y01": s(y01, 1), "y00": s(y00, 1),
        "N": N, "t1": t1, "c0": c0, "d1": s(d1, 1), "d0": s(d0, 1), "w1": s(w1, 1), "w0": s(1 - w1, 1),
        "wt1": s(n11 / t1, 2), "wt0": s(n01 / t1, 2),
        "att": r(att, 1), "att_s": s(att, 1), "ate": s(ate, 1), "atc": s(atc, 1), "naive": s(naive, 1),
        "mt": s((n11 * y11 + n01 * y01) / t1, 1), "mc": s((n10 * y10 + n00 * y00) / c0, 2),
    }


def p10():
    # (a) 2つの年級群の正規混合
    pi1, m1, m2, sg, x = 0.6, 12, 20, 3, 15
    ex = 2.4327   # e^{0.889}
    lr = (-(x - m1) ** 2 + (x - m2) ** 2) / (2 * sg ** 2)
    assert abs(lr - 0.889) < 0.0005
    post = pi1 * ex / (pi1 * ex + (1 - pi1))
    noprior = ex / (ex + 1)
    nohalf = pi1 * math.exp(2 * lr) / (pi1 * math.exp(2 * lr) + (1 - pi1))
    distinct([pi1, noprior, post, nohalf, 1.0], 3)
    # (b) 2標本 KS
    A = [11, 13, 14, 16, 18, 21]
    B = [15, 17, 19, 20, 22, 24]
    pts = sorted(A + B)
    D = max(abs(sum(v <= t for v in A) / 6 - sum(v <= t for v in B) / 6) for t in pts)
    assert abs(D - 0.5) < 1e-12
    pv = {3: 0.474, 4: 0.143, 5: 0.026}
    # (c) ファイ係数
    a, b, c, d = 15, 5, 10, 20
    n = a + b + c + d
    r1, r2, k1, k2 = a + b, c + d, a + c, b + d
    phi = (a * d - b * c) / sqrt(r1 * r2 * k1 * k2)
    out = {
        "pi1": pi1, "pi2": s(1 - pi1, 1), "m1": m1, "m2": m2, "sg": sg, "x": x, "ex": s(ex, 4), "lr": s(lr, 3),
        "num": s(pi1 * ex, 4), "den": s(pi1 * ex + 1 - pi1, 4), "post": s(post, 3), "post4": s(post, 4),
        "noprior": s(noprior, 3), "nohalf": s(nohalf, 3), "pr": s(pi1, 3),
        "pv3": s(pv[3], 3), "pv4": s(pv[4], 3), "pv5": s(pv[5], 3), "D": s(D, 2),
        "a": a, "b": b, "c": c, "d": d, "n": n, "r1": r1, "r2": r2, "k1": k1, "k2": k2,
        "adbc": a * d - b * c, "prod": r1 * r2 * k1 * k2, "sq": s(sqrt(r1 * r2 * k1 * k2), 2),
        "phi": r(phi, 3), "phi_s": s(phi, 3), "phi4": s(phi, 4), "chi": s(n * phi ** 2, 2),
        "yq": s((a * d - b * c) / (a * d + b * c), 3), "orr": s(a * d / (b * c), 1),
    }
    for i, v in enumerate(A):
        out[f"A{i}"] = v
    for i, v in enumerate(B):
        out[f"B{i}"] = v
    return out


def p11():
    m1, m2 = (40.0, 30.0), (36.0, 24.0)
    S = ((4.0, 2.0), (2.0, 5.0))
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    Si = ((S[1][1] / det, -S[0][1] / det), (-S[0][1] / det, S[0][0] / det))
    d = (m1[0] - m2[0], m1[1] - m2[1])
    w = (Si[0][0] * d[0] + Si[0][1] * d[1], Si[1][0] * d[0] + Si[1][1] * d[1])
    mid = ((m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2)
    x = (42.0, 24.5)
    dx = (x[0] - mid[0], x[1] - mid[1])
    z = w[0] * dx[0] + w[1] * dx[1]
    z_S = (S[0][0] * d[0] + S[0][1] * d[1]) * dx[0] + (S[1][0] * d[0] + S[1][1] * d[1]) * dx[1]
    z_E = d[0] * dx[0] + d[1] * dx[1]
    assert z < 0 and z_S > 0 and z_E > 0
    # (b) 共分散選択
    r12, r13, r23 = 0.62, 0.80, 0.60
    fit = r13 * r23
    pc = (r12 - r13 * r23) / sqrt((1 - r13 ** 2) * (1 - r23 ** 2))
    distinct([0.0, pc, fit, r12, r12 - fit], 2)
    return {
        "m11": s(m1[0], 0), "m12": s(m1[1], 0), "m21": s(m2[0], 0), "m22": s(m2[1], 0),
        "s11": s(S[0][0], 0), "s12": s(S[0][1], 0), "s22": s(S[1][1], 0), "det": s(det, 0),
        "d1": s(d[0], 0), "d2": s(d[1], 0), "w1": s(w[0], 1), "w2": s(w[1], 1),
        "mid1": s(mid[0], 0), "mid2": s(mid[1], 0), "x1": s(x[0], 0), "x2": s(x[1], 1),
        "dx1": s(dx[0], 1), "dx2": s(dx[1], 1),
        "z": s(z, 2), "zpos": s(-z, 2), "zS": s(z_S, 1), "zE": s(z_E, 2),
        "Sd1": s(S[0][0] * d[0] + S[0][1] * d[1], 0), "Sd2": s(S[1][0] * d[0] + S[1][1] * d[1], 0),
        "r12": s(r12, 2), "r13": s(r13, 2), "r23": s(r23, 2), "fit": s(fit, 2), "pc": s(pc, 2), "diff": s(r12 - fit, 2),
        "q13": s(1 - r13 ** 2, 2), "q23": s(1 - r23 ** 2, 2), "sqq": s(sqrt((1 - r13 ** 2) * (1 - r23 ** 2)), 2),
    }


def p12():
    # (a) 反射原理
    s2, T, a = 4.0, 9, 6.0
    sd = sqrt(s2 * T)
    phi = {0.17: 0.5675, 0.33: 0.6293, 1.00: 0.8413}
    zz = r(a / sd, 2)
    assert zz in phi
    ok = 2 * (1 - phi[zz])
    z_var = r(a / (s2 * T), 2)
    z_lin = r(a / (sqrt(s2) * T), 2)
    assert z_var in phi and z_lin in phi
    vals = [1 - phi[zz], ok, 1 - ok, 2 * (1 - phi[z_lin]), 2 * (1 - phi[z_var])]
    distinct(vals, 3)
    # (b) ボックス・ミュラー法
    u1, u2 = 0.30, 0.20
    ln1, ln2, c1, s1, c2 = -1.2040, -1.6094, 0.3090, 0.9511, -0.3090
    k = sqrt(s2)
    z = r(sqrt(-2 * ln1) * c1, 4)
    g = k * z
    g_swap = k * sqrt(-2 * ln2) * c2
    g_nosq = k * (-2 * ln1) * c1
    g_sin = k * sqrt(-2 * ln1) * s1
    g_no2 = k * sqrt(-ln1) * c1
    distinct([g_swap, g_no2, g, g_nosq, g_sin], 2)
    return {
        "s2": s(s2, 0), "T": T, "a": s(a, 0), "vT": s(s2 * T, 0), "sd": s(sd, 0), "zz": s(zz, 2),
        "ph017": s(phi[0.17], 4), "ph033": s(phi[0.33], 4), "ph1": s(phi[1.00], 4),
        "ok": r(ok, 3), "ok_s": s(ok, 3), "ok4": s(ok, 4), "w1": s(vals[0], 3), "w2": s(vals[2], 3), "w3": s(vals[3], 3), "w4": s(vals[4], 3),
        "z_lin": s(z_lin, 2), "z_var": s(z_var, 2),
        "u1": s(u1, 2), "u2": s(u2, 2), "ln1": s(ln1, 4), "ln2": s(ln2, 4), "c1": s(c1, 4), "s1": s(s1, 4), "c2": s(c2, 4),
        "m2ln": s(-2 * ln1, 3), "sqv": s(sqrt(-2 * ln1), 4), "z": s(z, 4), "k": s(k, 0),
        "g": s(g, 2), "g4": s(g, 4), "g_swap": s(g_swap, 2), "g_nosq": s(g_nosq, 2), "g_sin": s(g_sin, 2), "g_no2": s(g_no2, 2),
    }


def values():
    return {"s18p01": p01(), "s18p02": p02(), "s18p03": p03(), "s18p04": p04(), "s18p05": p05(),
            "s18p06": p06(), "s18p07": p07(), "s18p08": p08(), "s18p09": p09(), "s18p10": p10(),
            "s18p11": p11(), "s18p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
