"""模擬試験 第18回（set18）の数値。values() が {大問id: {名前: 値}} を返す。

題材：生物学・遺伝・生態調査・水産。
問題文に示した分布の値（χ²・F・t の点、Φ・φ、e^x・ln・cos など）は、ここでも同じ丸めた値を使って答えを計算する。
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
        "nAA": nAA, "nAa": nAa, "naa": naa, "n": n, "n2": 2 * n, "cntA": cntA,
        "p": s(p, 2), "q": s(q, 2), "p_ind": s((nAA + nAa) / n, 2), "nind": nAA + nAa, "p_hom": s(sqrt(nAA / n), 2),
        "p_n": s(cntA / n, 2),
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
    # (a) 2 遺伝子座のハプロタイプ頻度と連鎖不平衡
    h = {"AB": 0.40, "Ab": 0.20, "aB": 0.10, "ab": 0.30}
    pA = h["AB"] + h["Ab"]
    pB = h["AB"] + h["aB"]
    D = h["AB"] - pA * pB
    assert abs(D - (h["AB"] * h["ab"] - h["Ab"] * h["aB"])) < 1e-12
    # (b) ワイブル分布の寿命の四分位範囲
    m, eta = 2, 40
    l1, l3 = 0.2877, 1.3863   # ln(4/3), ln 4
    q1 = eta * sqrt(l1)
    q3 = eta * sqrt(l3)
    iqr = q3 - q1
    med = eta * sqrt(0.6931)
    return {
        "hAB": s(h["AB"], 2), "hAb": s(h["Ab"], 2), "haB": s(h["aB"], 2), "hab": s(h["ab"], 2),
        "pA": s(pA, 2), "pB": s(pB, 2), "pApB": s(pA * pB, 2), "D": s(D, 2), "mD": s(-D, 2),
        "prod1": s(h["AB"] * h["ab"], 2), "prod2": s(h["Ab"] * h["aB"], 2), "ratio": s(h["AB"] / (pA * pB), 2),
        "pBA": s(h["AB"] / pA, 3),
        "m": m, "eta": eta, "l1": s(l1, 4), "l3": s(l3, 4), "sl1": s(sqrt(l1), 4), "sl3": s(sqrt(l3), 4),
        "q1": s(q1, 2), "q3": s(q3, 2), "iqr": r(iqr, 1), "iqr_s": s(iqr, 1), "iqr2": s(iqr, 2),
        "med": s(med, 1), "iqr_exp": s(eta * (l3 - l1), 1),
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
    med_nl = (mu + 1) ** 2
    z_nl = sqrt(w0) - 1
    p_nl = phi[(mu - z_nl) / sg]
    med_n1 = (lam * mu) ** 2
    z_n1 = sqrt(w0) / lam
    p_n1 = 1 - phi[(z_n1 - mu) / sg]
    p_var = 1 - phi[(z0 - mu) / sg ** 2]
    # (b) t 検定の検出力（正規近似）
    n, dlt, sig = 16, 2.0, 4.0
    t1, t2, z1 = 1.753, 2.131, 1.645
    ph = {0.13: 0.5517, 0.25: 0.5987, 0.36: 0.6406, 1.25: 0.8944}
    ncp = dlt * sqrt(n) / sig
    z_ok = r(ncp - t1, 2)
    z_z = r(ncp - z1, 2)
    z_two = r(t2 - ncp, 2)
    z_nos = r(t1 - dlt / sig, 2)
    for zz in (z_ok, z_z, z_two, z_nos):
        assert zz in ph, zz
    pw = ph[z_ok]
    vals = [pw, ph[z_z], 1 - ph[z_two], 1 - ph[z_nos], 1 - pw]
    distinct(vals, 3)
    return {
        "lam": lam, "mu": s(mu, 0), "sg": s(sg, 0), "var": s(sg ** 2, 0),
        "ph05": s(phi[0.5], 4), "ph1": s(phi[1.0], 4), "ph2": s(phi[2.0], 4),
        "med": s(med, 1), "sqmed": s(1 + lam * mu, 0), "w0": w0, "z0": s(z0, 0), "zs": s(zs, 1),
        "p_ok": s(p_ok, 3), "p_ok4": s(p_ok, 4), "med_nl": s(med_nl, 1), "p_nl": s(p_nl, 3), "med_n1": s(med_n1, 1),
        "p_n1": s(p_n1, 3), "p_var": s(p_var, 3), "p_tail": s(phi[zs], 3),
        "n": n, "df": n - 1, "dlt": s(dlt, 1), "sig": s(sig, 1), "t1": s(t1, 3), "t2": s(t2, 3), "z1": s(z1, 3),
        "pc13": s(ph[0.13], 4), "pc25": s(ph[0.25], 4), "pc36": s(ph[0.36], 4), "pc125": s(ph[1.25], 4),
        "ncp": s(ncp, 1), "zraw": s(ncp - t1, 3), "zok": s(z_ok, 2),
        "pw": s(pw, 3), "pw4": s(pw, 4), "pw_z": s(vals[1], 3), "pw_two": s(vals[2], 3), "pw_nos": s(vals[3], 3), "beta": s(vals[4], 3),
        "dsig": s(dlt / sig, 1),
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
        "n": n, "x": x, "p": s(p, 1),
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
    # (b) 不比例配分の層別抽出と抽出ウェイト
    N1, N2, n1, n2 = 200, 900, 50, 30
    y1, y2 = 11.5, 2.3
    w1, w2 = N1 / n1, N2 / n2
    est = (w1 * n1 * y1 + w2 * n2 * y2) / (w1 * n1 + w2 * n2)
    naive = (n1 * y1 + n2 * y2) / (n1 + n2)
    wrong_w = (n1 / N1 * n1 * y1 + n2 / N2 * n2 * y2) / (n1 / N1 * n1 + n2 / N2 * n2)
    distinct([est, naive, wrong_w, (y1 + y2) / 2], 2)
    return {
        "lam": lam, "e": s(e, 4), "e1": s(1 - e, 4), "ok": s(ok, 2), "ok4": s(ok, 4),
        "w_mul": s(vals[0], 2), "w_lam": s(lam, 2), "w_p1": s(lam + 1, 2), "w_div": s(lam / e, 2),
        "N1": N1, "N2": N2, "n1": n1, "n2": n2, "y1": s(y1, 1), "y2": s(y2, 1), "N": N1 + N2,
        "w1": s(w1, 0), "w2": s(w2, 0), "T1": s(N1 * y1, 0), "T2": s(N2 * y2, 0), "Ttot": s(N1 * y1 + N2 * y2, 0),
        "est": r(est, 2), "est_s": s(est, 2), "naive": s(naive, 2), "wrong_w": s(wrong_w, 2), "simple": s((y1 + y2) / 2, 2),
    }


def p06():
    n, b, se, b0 = 25, 3.24, 0.10, 3
    t = (b - b0) / se
    tc = {"t23": 2.069, "t23h": 2.807, "t24": 2.064}
    assert tc["t23"] < t < tc["t23h"]
    return {
        "n": n, "b": s(b, 2), "se": s(se, 2), "b0": b0, "df": n - 2, "t": s(t, 2), "t0": s(b / se, 1),
        "t_nose": s(b - b0, 2), **{k: s(v, 3) for k, v in tc.items()},
    }


def sigm(u, ex):
    """1/(1+e^{-u})。e^{-u} は問題文に示した値 ex を使う。"""
    return 1 / (1 + ex)


def p07():
    # (a) 切片だけのロジスティック回帰のニュートン・ラフソン 1 回
    nsite, occ = 100, 30
    U = occ - nsite * 0.5
    I = nsite * 0.25
    b1 = U / I
    mle = -0.8473      # log(3/7)
    vals = [mle, b1, U / (nsite * 0.5), U / nsite, -b1]
    distinct(vals, 2)
    # (b) 小さなニューラルネットワーク
    x = 2.0
    w1, c1, w2, c2 = 1.0, -2.0, -1.0, 1.0
    v1, v2, c0 = 2.0, -3.0, 0.5
    e1 = 2.7183
    h1 = 0.5                       # σ(0)
    h2 = 1 / (1 + e1)              # σ(-1)
    h2r = r(h2, 4)
    u = v1 * h1 + v2 * h2r + c0
    ur = r(u, 3)
    assert ur == 0.693
    ex = {0.693: 0.5001, -0.693: 1.9997, 0.5: 0.6065, 3.5: 0.0302}
    out = 1 / (1 + ex[0.693])
    h2s = 1 - h2r                  # σ(1)
    u_s = r(v1 * h1 + v2 * h2s + c0, 3)
    assert u_s == -0.693
    o_s = 1 / (1 + ex[-0.693])
    o_relu = 1 / (1 + ex[0.5])     # 隠れ層を ReLU とした（h1=h2=0）
    o_lin = 1 / (1 + ex[3.5])      # 隠れ層の活性化を忘れた（h1=0, h2=-1）
    assert v1 * 0 + v2 * (-1) + c0 == 3.5
    distinct([o_s, o_relu, out, ur, o_lin], 3)
    return {
        "nsite": nsite, "occ": occ, "U": s(U, 0), "I": s(I, 0), "b1": s(b1, 2), "mle": s(mle, 4), "mle2": s(mle, 2),
        "w_half": s(vals[2], 2), "w_n": s(vals[3], 2), "w_sign": s(vals[4], 2), "ybar": s(occ / nsite, 1),
        "x": s(x, 0), "w1": s(w1, 1), "c1": s(c1, 1), "w2": s(w2, 1), "c2": s(c2, 1), "v1": s(v1, 1), "v2": s(v2, 1), "c0": s(c0, 1),
        "e1": s(e1, 4), "h2": s(h2r, 4), "u": s(ur, 3), "u4": s(u, 4), "em693": s(ex[0.693], 4), "ep693": s(ex[-0.693], 4),
        "em05": s(ex[0.5], 4), "em35": s(ex[3.5], 4),
        "out": s(out, 3), "out4": s(out, 4), "o_s": s(o_s, 3), "o_relu": s(o_relu, 3), "o_lin": s(o_lin, 3), "o_raw": s(ur, 3),
        "v2h2": s(v2 * h2r, 4),
    }


def p08():
    # (a) L8 直交表
    colA = [1, 1, 1, 1, 2, 2, 2, 2]
    colB = [1, 1, 2, 2, 1, 1, 2, 2]
    colC = [1, 2, 1, 2, 1, 2, 1, 2]
    y = [20, 22, 25, 27, 26, 30, 31, 35]
    T1 = sum(v for v, a in zip(y, colA) if a == 1)
    T2 = sum(v for v, a in zip(y, colA) if a == 2)
    eff = (T2 - T1) / 4
    ss = (T2 - T1) ** 2 / 8
    # (b) 非心 F 分布の期待値
    k, nn, sig = 3, 8, 4.0
    al = (-2.0, 0.0, 2.0)
    lam = nn * sum(a * a for a in al) / sig ** 2
    nu1, nu2 = k - 1, k * nn - k
    EF = nu2 * (nu1 + lam) / (nu1 * (nu2 - 2))
    lam_n = sum(a * a for a in al) / sig ** 2
    lam_s = nn * sum(a * a for a in al) / sig
    vals = [EF, nu2 * (nu1 + lam_n) / (nu1 * (nu2 - 2)), nu2 * (nu1 + lam_s) / (nu1 * (nu2 - 2)), nu2 / (nu2 - 2)]
    distinct(vals, 2)
    out = {
        "T1": T1, "T2": T2, "dT": T2 - T1, "eff": s(eff, 1), "ss": s(ss, 1), "ss4": s((T2 - T1) ** 2 / 4, 1),
        "ss16": s((T2 - T1) ** 2 / 16, 1), "half": s(eff / 2, 2), "ssh": s(ss / 4, 2),
        "k": k, "nn": nn, "sig": s(sig, 0), "s2": s(sig ** 2, 0), "a1": s(al[0], 0), "a3": s(al[2], 0),
        "sa2": s(sum(a * a for a in al), 0), "lam": s(lam, 1), "nu1": nu1, "nu2": nu2,
        "EF": r(EF, 2), "EF_s": s(EF, 2), "EF4": s(EF, 4), "EF_n": s(vals[1], 2), "EF_s2": s(vals[2], 2), "EF0": s(vals[3], 2),
        "lam_n": s(lam_n, 1), "lam_s": s(lam_s, 0),
    }
    for i, v in enumerate(y):
        out[f"y{i}"] = v
    return out


def p09():
    pL, pP = 0.4, 0.3
    q = {(0, 0): 0.1, (1, 0): 0.5, (0, 1): 0.5, (1, 1): 0.6}
    j = {(l, pp): (pL if l else 1 - pL) * (pP if pp else 1 - pP) * q[(l, pp)] for l in (0, 1) for pp in (0, 1)}
    big = j[(1, 1)] / (j[(1, 1)] + j[(1, 0)])
    small = j[(0, 1)] / (j[(0, 1)] + j[(0, 0)])
    rb = j[(1, 1)] / (j[(1, 1)] + j[(0, 1)])
    rs = j[(1, 0)] / (j[(1, 0)] + j[(0, 0)])
    sel = sum(j.values())
    return {
        "pL": s(pL, 1), "pP": s(pP, 1), "q00": s(q[(0, 0)], 1), "q10": s(q[(1, 0)], 1), "q01": s(q[(0, 1)], 1), "q11": s(q[(1, 1)], 1),
        "a11": s(pP * q[(1, 1)], 2), "a10": s((1 - pP) * q[(1, 0)], 2), "a01": s(pP * q[(0, 1)], 2), "a00": s((1 - pP) * q[(0, 0)], 2),
        "s1": s(pP * q[(1, 1)] + (1 - pP) * q[(1, 0)], 2), "s0": s(pP * q[(0, 1)] + (1 - pP) * q[(0, 0)], 2),
        "big": s(big, 3), "small": s(small, 3), "pPs": s(pP, 3), "j11": s(pP * q[(1, 1)], 3), "j01": s(pP * q[(0, 1)], 3),
        "rb": s(rb, 3), "rs": s(rs, 3), "sel": s(sel, 3),
        "orc": s((big / (1 - big)) / (small / (1 - small)), 2),
    }


def p10():
    # (a) 2標本 KS
    A = [11, 13, 14, 16, 18, 21]
    B = [15, 17, 19, 20, 22, 24]
    pts = sorted(A + B)
    D = max(abs(sum(v <= t for v in A) / 6 - sum(v <= t for v in B) / 6) for t in pts)
    assert abs(D - 0.5) < 1e-12
    pv = {3: 0.474, 4: 0.143, 5: 0.026}
    # (b) ファイ係数
    a, b, c, d = 15, 5, 10, 20
    n = a + b + c + d
    r1, r2, k1, k2 = a + b, c + d, a + c, b + d
    phi = (a * d - b * c) / sqrt(r1 * r2 * k1 * k2)
    # (c) カリンスキー・ハラバシュ指標
    xs = [11, 12, 13, 18, 19, 20, 30, 31]
    nx = len(xs)
    gm = sum(xs) / nx
    T = sum((v - gm) ** 2 for v in xs)
    def W(parts):
        return sum(sum((v - sum(g) / len(g)) ** 2 for v in g) for g in parts)
    W2 = W([xs[:6], xs[6:]])
    W3 = W([xs[:3], xs[3:6], xs[6:]])
    ch = lambda Wk, k: ((T - Wk) / (k - 1)) / (Wk / (nx - k))
    CH2, CH3 = ch(W2, 2), ch(W3, 3)
    assert CH3 > CH2
    vals = [CH3, (T - W3) / (W3 / (nx - 3)), ((T - W3) / 2) / (W3 / (nx - 4)), (T - W3) / W3, W3 / (T - W3)]
    distinct(vals, 1)
    out = {
        "pv3": s(pv[3], 3), "pv4": s(pv[4], 3), "pv5": s(pv[5], 3), "D": s(D, 2),
        "a": a, "b": b, "c": c, "d": d, "n": n, "r1": r1, "r2": r2, "k1": k1, "k2": k2,
        "adbc": a * d - b * c, "prod": r1 * r2 * k1 * k2, "sq": s(sqrt(r1 * r2 * k1 * k2), 2),
        "phi": r(phi, 3), "phi_s": s(phi, 3), "phi4": s(phi, 4), "chi": s(n * phi ** 2, 2),
        "yq": s((a * d - b * c) / (a * d + b * c), 3), "orr": s(a * d / (b * c), 1),
        "gm": s(gm, 2), "T": s(T, 1), "W2": s(W2, 1), "W3": s(W3, 1), "Bss3": s(T - W3, 1), "CH2": s(CH2, 1), "CH3": s(CH3, 1),
        "ch_nok": s(vals[1], 1), "ch_n4": s(vals[2], 1), "ch_raw": s(vals[3], 1), "ch_inv": s(vals[4], 3),
        "B3h": s((T - W3) / 2, 2), "W3h": s(W3 / (nx - 3), 2),
    }
    for i, v in enumerate(A):
        out[f"A{i}"] = v
    for i, v in enumerate(B):
        out[f"B{i}"] = v
    return out


def p11():
    # (a) 環境DNA の検出スコアと閾値
    pres = [2, 4, 5, 6, 7, 8, 9, 9]
    absn = [1, 1, 2, 3, 3, 4, 5, 7]
    out = {}
    best = None
    for c in (3, 5, 7):
        se = sum(v >= c for v in pres) / len(pres)
        sp_ = sum(v < c for v in absn) / len(absn)
        J = se + sp_ - 1
        out[f"se{c}"], out[f"sp{c}"], out[f"J{c}"] = s(se, 3), s(sp_, 3), s(J, 3)
        out[f"fp{c}"] = s(1 - sp_, 3)
        if best is None or J > best[1]:
            best = (c, J)
    assert best[0] == 5
    js = sorted(float(out[f"J{c}"]) for c in (3, 5, 7))
    assert js[-1] > js[-2]
    # (b) 2因子直交モデル
    L = {"len": (0.8, 0.3), "hgt": (0.7, 0.4), "wid": (0.3, 0.8), "wt": (0.6, 0.5)}
    robs = 0.50
    rep = L["len"][0] * L["wid"][0] + L["len"][1] * L["wid"][1]
    h_len = sum(v * v for v in L["len"])
    h_wid = sum(v * v for v in L["wid"])
    f1 = L["len"][0] * L["wid"][0]
    distinct([rep, f1, h_len * h_wid, h_len], 2)
    out.update({
        "pres": "、".join(map(str, pres)), "absn": "、".join(map(str, absn)), "np": len(pres), "na": len(absn),
        **{f"{k}{i}": s(v[i], 1) for k, v in L.items() for i in (0, 1)},
        "robs": s(robs, 2), "rep": s(rep, 2), "res": s(robs - rep, 2), "mres": s(rep - robs, 2), "f1": s(f1, 2), "rf1": s(robs - f1, 2),
        "hh": s(h_len * h_wid, 2), "rhh": s(robs - h_len * h_wid, 2), "hl": s(h_len, 2), "rhl": s(robs - h_len, 2), "ul": s(1 - h_len, 2),
        "p1": s(L["len"][0] * L["wid"][0], 2), "p2": s(L["len"][1] * L["wid"][1], 2),
    })
    return out


def p12():
    # (a) ユール過程
    lam, t, k = 0.5, 2, 4
    em = 0.3679
    q = 1 - em
    ok = q ** (k - 1)
    eq = em * q ** (k - 1)
    w_k = q ** k
    w_pois = None
    distinct([ok, eq, w_k], 3)
    # (b) ボックス・ミュラー法
    u1, u2 = 0.30, 0.20
    ln1, ln2, c1, s1, c2 = -1.2040, -1.6094, 0.3090, 0.9511, -0.3090
    z = r(sqrt(-2 * ln1) * c1, 4)
    mu, sg = 0.30, 0.10
    g = mu + sg * z
    g_swap = mu + sg * sqrt(-2 * ln2) * c2
    g_nosq = mu + sg * (-2 * ln1) * c1
    g_sin = mu + sg * sqrt(-2 * ln1) * s1
    g_no2 = mu + sg * sqrt(-ln1) * c1
    distinct([g_swap, g_no2, g, g_nosq, g_sin], 3)
    return {
        "lam": lam, "t": t, "k": k, "lt": s(lam * t, 0), "em": s(em, 4), "q": s(q, 4), "km1": k - 1,
        "ok": r(ok, 3), "ok_s": s(ok, 3), "ok4": s(ok, 4), "eq": s(eq, 3), "w_k": s(w_k, 3), "mean": s(math.e, 2),
        "u1": s(u1, 2), "u2": s(u2, 2), "ln1": s(ln1, 4), "ln2": s(ln2, 4), "c1": s(c1, 4), "s1": s(s1, 4), "c2": s(c2, 4),
        "m2ln": s(-2 * ln1, 3), "sqv": s(sqrt(-2 * ln1), 4), "z": s(z, 4), "mu": s(mu, 2), "sg": s(sg, 2),
        "g": s(g, 3), "g5": s(g, 5), "g_swap": s(g_swap, 3), "g_nosq": s(g_nosq, 3), "g_sin": s(g_sin, 3), "g_no2": s(g_no2, 3),
    }


def values():
    return {"s18p01": p01(), "s18p02": p02(), "s18p03": p03(), "s18p04": p04(), "s18p05": p05(),
            "s18p06": p06(), "s18p07": p07(), "s18p08": p08(), "s18p09": p09(), "s18p10": p10(),
            "s18p11": p11(), "s18p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
