"""模擬試験 第9回（set09）の数値。values() が {大問id: {名前: 値}} を返す。

題材：農業・環境・生態・気象・スポーツ。
問題文に示した関数値（e^x、Φ、χ² の上側確率、べき乗、対数など）は、ここでも同じ丸めた値を使って答えを計算する。
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


def p01():
    # (a) 3近傍法：標準化の有無
    pts = [(3900, 0.95, "B"), (4150, 0.90, "B"), (4300, 0.45, "A"), (3600, 0.55, "C"),
           (4050, 0.20, "C"), (3950, 1.00, "B"), (4560, 0.50, "A")]
    new, sf, sl = (4000, 0.50), 400, 0.2
    raw = [sqrt((f - new[0]) ** 2 + (l - new[1]) ** 2) for f, l, _ in pts]
    std = [((f - new[0]) / sf) ** 2 + ((l - new[1]) / sl) ** 2 for f, l, _ in pts]
    o_raw = sorted(range(7), key=lambda i: raw[i])[:3]
    o_std = sorted(range(7), key=lambda i: std[i])[:3]
    lab_raw = [pts[i][2] for i in o_raw]
    lab_std = [pts[i][2] for i in o_std]
    assert max(set(lab_raw), key=lab_raw.count) == "B" and lab_raw.count("B") >= 2
    assert max(set(lab_std), key=lab_std.count) == "A" and lab_std.count("A") >= 2
    knn = {f"f{i+1}": pts[i][0] for i in range(7)}
    knn.update({f"l{i+1}": s(pts[i][1], 2) for i in range(7)})
    knn.update({f"y{i+1}": pts[i][2] for i in range(7)})
    knn.update({f"ds{i+1}": s(std[i], 4) for i in range(7)})
    knn.update({f"dr{i+1}": s(raw[i], 1) for i in range(7)})
    knn.update({"nf": new[0], "nl": s(new[1], 2), "sf": sf, "sl": s(sl, 1),
                "rawnn": "、".join(f"{i+1}（{pts[i][2]}）" for i in o_raw),
                "stdnn": "、".join(f"{i+1}（{pts[i][2]}）" for i in o_std)})
    # (b) 混同行列（行＝真の種、列＝判定）と、野外の種の割合
    cm = {"A": (84, 12, 4), "B": (18, 165, 17), "C": (9, 21, 270)}
    nA, nB, nC = (sum(cm[k]) for k in "ABC")
    tp = cm["A"][0]
    predA = cm["A"][0] + cm["B"][0] + cm["C"][0]
    prec = tp / predA
    piA, piB, piC = 0.05, 0.35, 0.60
    lA, lB, lC = tp / nA, cm["B"][0] / nB, cm["C"][0] / nC
    num = piA * lA
    den = num + piB * lB + piC * lC
    post = num / den
    eq = lA / (lA + lB + lC)
    d = {
        "a1": cm["A"][0], "a2": cm["A"][1], "a3": cm["A"][2],
        "b1": cm["B"][0], "b2": cm["B"][1], "b3": cm["B"][2],
        "c1": cm["C"][0], "c2": cm["C"][1], "c3": cm["C"][2],
        "nA": nA, "nB": nB, "nC": nC, "nAll": nA + nB + nC, "tp": tp, "predA": predA,
        "piA": piA, "piB": piB, "piC": piC,
        "lA": s(lA, 2), "lB": s(lB, 2), "lC": s(lC, 2),
        "jA": s(piA * lA, 3), "jB": s(piB * lB, 4), "jC": s(piC * lC, 3), "den": s(den, 4),
        "post": s(post, 3), "post4": s(post, 4), "w_prec": s(prec, 3), "w_rec": s(lA, 3), "w_joint": s(num, 3),
        "w_eq": s(eq, 3),
    }
    d.update(knn)
    return d


def p02():
    # (a) 7戦4勝制のシリーズが第6戦で決着する確率
    p, q = 0.6, 0.4
    a6 = math.comb(5, 3) * p ** 4 * q ** 2
    b6 = math.comb(5, 3) * q ** 4 * p ** 2
    w15a = math.comb(6, 4) * p ** 4 * q ** 2
    w15b = math.comb(6, 4) * q ** 4 * p ** 2
    d = {
        "sp": p, "sq": s(q, 1), "p4q2": s(p ** 4 * q ** 2, 6), "q4p2": s(q ** 4 * p ** 2, 6),
        "a6": s(a6, 4), "b6": s(b6, 4), "six": s(a6 + b6, 3), "six4": s(a6 + b6, 5),
        "w_a": s(a6, 3), "w_b": s(b6, 3), "w_c6": s(w15a + w15b, 3), "w_c6a": s(w15a, 3),
    }
    d.update(tennis())
    return d


def p03():
    # 切断データ：網目より大きい魚だけが捕れる
    mu, sg, c = 18, 4, 20
    a = (c - mu) / sg
    phi, Phi = 0.3521, 0.6915
    lam = phi / (1 - Phi)
    m = mu + sg * lam
    return {
        "mu": mu, "sg": sg, "c": c, "a": s(a, 1), "phi": s(phi, 4), "Phi": s(Phi, 4), "tail": s(1 - Phi, 4),
        "lam": s(lam, 4), "add": s(sg * lam, 3), "m": s(m, 1), "m3": s(m, 3),
        "w_nod": s(mu + sg * phi, 1), "w_Phi": s(mu + sg * phi / Phi, 1), "w_c": s(c + sg * lam, 1), "w_mu": s(mu, 1),
    }

def p04():
    # (a) 風速：東西・南北成分が独立に N(0, σ^2)
    sig, w = 4, 10
    thr = w ** 2 / sig ** 2
    tab1 = {2.5: 0.1138, 3.125: 0.0771, 6.25: 0.0124}
    tab2 = {2.5: 0.2865, 3.125: 0.2096, 6.25: 0.0439}
    ans = tab2[thr]
    # (b) マルコフ・チェビシェフ・正確な値
    th, mult = 12, 5
    xa = th * mult
    mk = th / xa
    ch = th ** 2 / (xa - th) ** 2
    e5 = 0.0067
    # (c) 相関行列の主成分分析
    lam = (1.944, 0.707, 0.349)
    a2 = (-0.570, 0.609, -0.551)
    t1, t2 = lam[0] * a2[0] ** 2, lam[1] * a2[1] ** 2
    ans_c = t1 + t2
    return {
        "sig": sig, "w": w, "thr": s(thr, 2), "sig2": sig ** 2, "w2": w ** 2,
        "t1a": s(tab1[2.5], 4), "t1b": s(tab1[3.125], 4), "t1c": s(tab1[6.25], 4),
        "t2a": s(tab2[2.5], 4), "t2b": s(tab2[3.125], 4), "t2c": s(tab2[6.25], 4),
        "ans": s(ans, 4), "w_df1": s(tab1[thr], 4), "w_nosq": s(tab2[w / sig], 4), "w_comp": s(1 - ans, 4),
        "w_half": s(tab2[w ** 2 / (2 * sig ** 2)], 4), "thr2": s(w ** 2 / (2 * sig ** 2), 3),
        "th": th, "mult": mult, "xa": xa, "mk": s(mk, 3), "ch": s(ch, 4), "e5": s(e5, 4), "dev": xa - th,
        "th2": th ** 2, "dev2": (xa - th) ** 2, "w_chsd": s(th / (xa - th), 3), "w_mk": s(1 - mk, 3),
        "w_ex": s(1 - e5, 4), "w_chk": s(th ** 2 / xa ** 2, 3),
        "l1": s(lam[0], 3), "l2": s(lam[1], 3), "l3": s(lam[2], 3),
        "v11": "0.519", "v12": "0.787", "v13": "0.333",
        "v21": s(a2[0], 3), "v22": s(a2[1], 3), "v23": s(a2[2], 3),
        "v31": "0.637", "v32": "-0.096", "v33": "-0.765",
        "sq1": s(a2[0] ** 2, 4), "sq2": s(a2[1] ** 2, 6), "sq3": s(a2[2] ** 2, 6),
        "tc1": s(t1, 4), "tc2": s(t2, 4), "tc3": s(lam[2] * a2[2] ** 2, 4),
        "ansc": s(ans_c, 3), "ansc4": s(ans_c, 4), "w_pc1": s(t1, 3), "w_unw": s(a2[0] ** 2 + a2[1] ** 2, 3),
        "w_cum": s((lam[0] + lam[1]) / 3, 3), "w_pc3": s(lam[2] * a2[2] ** 2, 3),
        "alt": s(1 - lam[2] * a2[2] ** 2, 4),
    }


def p05():
    # (a) 標識再捕獲
    M, n, m = 64, 45, 11
    lp = M * n / m
    nhat = math.floor(lp)
    # (b) 累積ロジットモデル
    th1, th2, g1, g2 = 0.4, 2.0, 1.2, 0.5
    eta = g1 + g2
    ex = {1.3: 3.6693, -0.3: 0.7408, -1.3: 0.2725, 0.3: 1.3499}
    a1, a2 = round(th1 - eta, 1), round(th2 - eta, 1)
    F1 = 1 / (1 + ex[-a1])
    F2 = 1 / (1 + ex[-a2])
    pm = F2 - F1
    # 符号を逆（θ+η）にした誤り：e^{-2.1}, e^{-3.7} は示さないので、説明だけに使う
    # (c) 確率母関数の合成
    pe = {2: 0.2, 3: 0.5, 4: 0.3}
    ph = 0.8
    qh = round(1 - ph, 1)
    Gp = sum(k_ * v * qh ** (k_ - 1) for k_, v in pe.items())
    G0 = sum(v * qh ** k_ for k_, v in pe.items())
    p1 = ph * Gp
    p2 = sum(v * math.comb(k_, 2) * ph ** 2 * qh ** (k_ - 2) for k_, v in pe.items())
    return {
        "M": M, "n": n, "m": m, "lp": s(lp, 2), "nhat": nhat, "Mn": M * n, "w_ceil": nhat + 1,
        "lower": M + n - m,
        "th1": s(th1, 1), "th2": s(th2, 1), "g1": s(g1, 1), "g2": s(g2, 1), "eta": s(eta, 1),
        "a1": s(a1, 1), "a2": s(a2, 1), "e13": s(ex[1.3], 4), "em03": s(ex[-0.3], 4), "em13": s(ex[-1.3], 4), "e03": s(ex[0.3], 4),
        "F1": s(F1, 4), "F2": s(F2, 4), "pm": r(pm, 3), "pm3": s(pm, 3), "pm4": s(pm, 4), "d1": s(1 + ex[-a1], 4), "d2": s(1 + ex[-a2], 4),
        "w_F2": s(F2, 3), "w_hi": s(1 - F2, 3), "w_lo": s(F1, 3),
        "p2e": pe[2], "p3e": pe[3], "p4e": pe[4], "ph": ph, "qh": qh,
        "Gp": s(Gp, 4), "gp1": s(2 * pe[2] * qh, 4), "gp2": s(3 * pe[3] * qh ** 2, 4), "gp3": s(4 * pe[4] * qh ** 3, 4),
        "P1": s(p1, 3), "P1f": s(p1, 5), "w_nop": s(Gp, 3), "w_G0": s(G0, 3), "w_non": s(sum(v * ph * qh ** (k_ - 1) for k_, v in pe.items()), 3), "w_P2": s(p2, 3),
        "G0f": s(G0, 5),
    }


def p06():
    # (a) コルモゴロフ・スミルノフ統計量
    x = (1.2, 2.5, 4.1, 6.8, 12.0)
    F = (0.2134, 0.3935, 0.5596, 0.7433, 0.9093)
    n = len(x)
    dp = [(i + 1) / n - f for i, f in enumerate(F)]
    dm = [f - i / n for i, f in enumerate(F)]
    D = max(max(dp), max(dm))
    mid = max(abs(f - (i + 0.5) / n) for i, f in enumerate(F))
    d = {f"x{i+1}": s(v, 1) for i, v in enumerate(x)}
    d.update({f"F{i+1}": s(v, 4) for i, v in enumerate(F)})
    d.update({f"dp{i+1}": s(v, 4) for i, v in enumerate(dp)})
    d.update({f"dm{i+1}": s(v, 4) for i, v in enumerate(dm)})
    d.update({"D": s(D, 3), "D4": s(D, 4), "Dp": s(max(dp), 3), "Dm": s(max(dm), 4), "mid": s(mid, 3), "crit": "0.563"})
    # (b) 母比率の推定に必要な標本の大きさ
    p0, e, z = 0.8, 0.03, 1.96
    nraw = z ** 2 * p0 * (1 - p0) / e ** 2
    d.update({
        "p0": p0, "e": e, "z": z, "z2": s(z ** 2, 4), "pq": s(p0 * (1 - p0), 2), "e2": s(e ** 2, 4),
        "nraw": s(nraw, 2), "nreq": math.ceil(nraw), "w_half": math.ceil(z ** 2 * 0.25 / e ** 2),
        "w_z1": math.ceil(1.645 ** 2 * p0 * (1 - p0) / e ** 2), "w_w": math.ceil(z ** 2 * p0 * (1 - p0) / (2 * e) ** 2),
    })
    return d

def p07():
    # (a) 20地点の多重検定
    k, al = 20, 0.05
    p95, p9975 = 0.3585, 0.9512
    fw = 1 - p95
    fb = 1 - p9975
    # (b) ホテリングの T^2
    n = 20
    d1, d2 = 0.5, -3.0
    s11, s12, s22 = 0.8, -2.0, 25.0
    det = s11 * s22 - s12 ** 2
    q = (s22 * d1 ** 2 - 2 * s12 * d1 * d2 + s11 * d2 ** 2) / det
    T2 = n * q
    F = (n - 2) / (2 * (n - 1)) * T2
    T2d = n * (d1 ** 2 / s11 + d2 ** 2 / s22)
    Fd = (n - 2) / (2 * (n - 1)) * T2d
    # (c) 古典的多次元尺度法
    P = [(0, 0), (6, 0), (1, 4), (5, 5)]
    D2 = [[(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for b in P] for a in P]
    rows = [sum(row) / 4 for row in D2]
    gm = sum(sum(row) for row in D2) / 16
    i, j = 0, 3
    b14 = -0.5 * (D2[i][j] - rows[i] - rows[j] + gm)
    return {
        "k": k, "al": al, "p95": p95, "p9975": p9975, "bal": s(al / k, 4),
        "fw": s(fw, 3), "fb": s(fb, 3), "w_union": s(k * al, 3), "w_bal": s(al / k, 4),
        "nT": n, "d1": s(d1, 1), "d2": s(d2, 1), "s11": s(s11, 1), "s12": s(s12, 1), "s22": s(s22, 1),
        "det": s(det, 1), "qa": s(s22 * d1 ** 2, 2), "qb": s(-2 * s12 * d1 * d2, 2), "qc": s(s11 * d2 ** 2, 2),
        "qsum": s(s22 * d1 ** 2 - 2 * s12 * d1 * d2 + s11 * d2 ** 2, 2), "q": s(q, 6),
        "T2": s(T2, 2), "Fc": s(F, 2), "Fk": "18/38", "w_T2nod": s(q, 2), "w_diagT": s(T2d, 2), "w_diagF": s(Fd, 2),
        "F218": "3.555", "F182": "19.44", "F219": "3.522",
        "d12": D2[0][1], "d13": D2[0][2], "d14": D2[0][3], "d23": D2[1][2], "d24": D2[1][3], "d34": D2[2][3],
        "rs1": sum(D2[0]), "rs4": sum(D2[3]), "rm1": s(rows[0], 2), "rm4": s(rows[3], 2),
        "tot2": sum(sum(row) for row in D2), "gm": s(gm, 3),
        "inner": s(D2[i][j] - rows[i] - rows[j] + gm, 3),
        "b14": s(b14, 2), "b14f": s(b14, 4), "w_nocent": s(-0.5 * D2[i][j], 2), "w_sign": s(-b14, 2),
        "w_nohalf": s(-(D2[i][j] - rows[i] - rows[j] + gm), 2),
        "w_gm": s(-0.5 * (D2[i][j] - rows[i] - rows[j] - gm), 2),
    }


def p08():
    ry1, ry2, r12 = 0.50, 0.40, -0.30
    # (a) 合成指標 Z=X1+X2 と Y の相関
    vz = 2 + 2 * r12
    cz = ry1 + ry2
    rz = cz / sqrt(vz)
    # (b) 標準化偏回帰係数と決定係数
    b1 = (ry1 - r12 * ry2) / (1 - r12 ** 2)
    b2 = (ry2 - r12 * ry1) / (1 - r12 ** 2)
    R2 = b1 * ry1 + b2 * ry2
    # (c) リッジ回帰
    lam = 0.2
    u = (ry1 + ry2) / (1 + r12 + lam)
    v = (ry1 - ry2) / (1 - r12 + lam)
    k1, k2 = (u + v) / 2, (u - v) / 2
    dneg = (1 - lam) ** 2 - r12 ** 2
    m1 = ((1 - lam) * ry1 - r12 * ry2) / dneg
    m2 = ((1 - lam) * ry2 - r12 * ry1) / dneg
    return {
        "ry1": s(ry1, 2), "ry2": s(ry2, 2), "r12": s(r12, 2),
        "cz": s(cz, 1), "vz": s(vz, 1), "sqvz": s(sqrt(vz), 4), "rz": s(rz, 3), "rz4": s(rz, 4),
        "w_avg": s(cz / 2, 3), "w_nocov": s(cz / sqrt(2), 3), "w_one": s(cz / sqrt(2 + r12), 3), "w_sum": s(cz, 3),
        "c2": s(1 - r12 ** 2, 2),
        "b1": s(b1, 4), "b2": s(b2, 4), "bn1": s(ry1 - r12 * ry2, 2), "bn2": s(ry2 - r12 * ry1, 2),
        "R2": r(R2, 3), "R2f": s(R2, 4), "R2num": s(ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12, 2),
        "t1": s(b1 * ry1, 4), "t2": s(b2 * ry2, 4), "w_sumsq": s(ry1 ** 2 + ry2 ** 2, 2),
        "lam": lam, "u": s(u, 4), "v": s(v, 4), "ud": s(1 + r12 + lam, 1), "vd": s(1 - r12 + lam, 1),
        "us": s(ry1 + ry2, 1), "vs": s(ry1 - ry2, 1),
        "k1": s(k1, 3), "k2": s(k2, 3), "o1": s(b1, 3), "o2": s(b2, 3),
        "h1": s(b1 / (1 + lam), 3), "h2": s(b2 / (1 + lam), 3), "i1": s(ry1 / (1 + lam), 3), "i2": s(ry2 / (1 + lam), 3),
        "m1": s(m1, 3), "m2": s(m2, 3),
    }


def p09():
    # 共通データ：単回帰の流量と残差（残差は和 0・x との積和 0 を満たす）
    x = [2, 3, 4, 5, 6, 10]
    e = [0.4, -0.6, 0.5, -0.1, -0.375, 0.175]
    n = len(x)
    xb = sum(x) / n
    sxx = sum((xi - xb) ** 2 for xi in x)
    h = [r(1 / n + (xi - xb) ** 2 / sxx, 4) for xi in x]      # 表示する 4 桁のてこ比
    d = {f"x{i+1}": xi for i, xi in enumerate(x)}
    d.update({f"e{i+1}": (s(ei, 1) if abs(ei * 10 - round(ei * 10)) < 1e-9 else s(ei, 3)) for i, ei in enumerate(e)})
    # (a) DFFITS：問題文に示す s^2 と s_(6)^2（4 桁）を使う
    sse = sum(ei ** 2 for ei in e)
    s2 = r(sse / (n - 2), 4)
    s62 = r((sse - e[-1] ** 2 / (1 - h[-1])) / (n - 3), 4)
    h6, e6 = h[-1], e[-1]
    dff = e6 * sqrt(h6) / (sqrt(s62) * (1 - h6))
    thr = 2 * sqrt(2 / n)
    t6 = e6 / (sqrt(s62) * sqrt(1 - h6))
    d.update({
        "n": n, "xb": s(xb, 0), "sxx": s(sxx, 0), "dev62": s((x[-1] - xb) ** 2, 0),
        "h6": s(h6, 4), "om6": s(1 - h6, 4), "sqh6": s(sqrt(h6), 4), "s2": s(s2, 4), "s62": s(s62, 4),
        "s6": s(sqrt(s62), 4), "dff": s(dff, 2), "dff4": s(dff, 4), "thr": s(thr, 2),
        "fac6": s(sqrt(h6 / (1 - h6)), 3),
        "w_noh": s(e6 * sqrt(h6) / sqrt(s62), 2), "w_t6": s(t6, 2), "w_nosq": s(t6 * h6 / (1 - h6), 2),
        "w_s": s(e6 * sqrt(h6) / (sqrt(s2) * (1 - h6)), 2),
    })
    assert dff > thr and e6 * sqrt(h6) / sqrt(s62) < thr and t6 < thr
    # (b) 一個抜き交差検証：表示した 4 桁の値を順に使う
    pr = [r(ei / (1 - hi), 4) for ei, hi in zip(e, h)]
    pq = [r(v ** 2, 4) for v in pr]
    press = sum(pq)
    cv = press / n
    train = sse / n
    for i in range(n):
        d[f"h{i+1}"] = s(h[i], 4)
        d[f"pr{i+1}"] = s(pr[i], 4)
        d[f"pq{i+1}"] = s(pq[i], 4)
    d.update({
        "press": s(press, 4), "cv": r(cv, 3), "cv3": s(cv, 3), "cv4": s(cv, 4),
        "w_train": s(train, 4), "ratio": s(cv / train, 1),
        "w_nosq2": s(sum(ei ** 2 / (1 - hi) for ei, hi in zip(e, h)) / n, 4),
    })
    return d

def p10():
    # (a) 確率比例抽出（復元）＋副次抽出のハンセン・ハーヴィッツ推定量
    N, n, m, M0 = 30, 3, 5, 3600
    M = (120, 80, 150)
    yb = (0.62, 0.55, 0.70)
    est = sum(Mi * yi / (Mi / M0) for Mi, yi in zip(M, yb)) / n
    eq = N / n * sum(Mi * yi for Mi, yi in zip(M, yb))
    ratio = M0 * sum(Mi * yi for Mi, yi in zip(M, yb)) / sum(M)
    # (b) ラテン方格
    k = 5
    ssr, ssc, sst_, sstot = 36.0, 20.0, 64.0, 156.0
    sse = sstot - ssr - ssc - sst_
    dft = k - 1
    dfe = (k - 1) * (k - 2)
    F = (sst_ / dft) / (sse / dfe)
    Frb = (sst_ / dft) / ((sse + ssc) / (dfe + dft))
    F1 = (sst_ / dft) / ((sstot - sst_) / (k * k - k))
    F20 = (sst_ / dft) / (sse / (k * k - 1 - dft))
    return {
        "Np": N, "np": n, "mp": m, "M0": M0, "M1": M[0], "M2": M[1], "M3": M[2],
        "yb1": s(yb[0], 2), "yb2": s(yb[1], 2), "yb3": s(yb[2], 2), "ysum": s(sum(yb), 2),
        "fac": s(M0 / n, 0), "est": r(est, 0), "est1": s(est, 1),
        "w_eq": s(eq, 1), "w_ratio": s(ratio, 1),
        "k": k, "ssr": s(ssr, 1), "ssc": s(ssc, 1), "sst": s(sst_, 1), "sstot": s(sstot, 1), "sse": s(sse, 1),
        "dft": dft, "dfe": dfe, "dftot": k * k - 1, "mst": s(sst_ / dft, 2), "mse": s(sse / dfe, 2),
        "F": s(F, 2), "w_rb": s(Frb, 2), "w_one": s(F1, 2), "w_20": s(F20, 2),
        "F412": "3.259", "F124": "5.912", "F416": "3.007", "F420": "2.866",
    }

def tennis():
    p = 0.6
    q = 1 - p
    D = p ** 2 / (p ** 2 + q ** 2)
    P3040 = p * D
    P1540 = p * P3040
    ans = p * D + q * P1540
    # デュースを無視：先に 4 ポイント（あと 3 点）を取れば勝ち、相手はあと 2 点で勝ち
    nodeuce = p ** 3 + 3 * p ** 3 * q
    Dp = p
    alt = p * Dp + q * p * p * Dp
    return {
        "p": p, "q": s(q, 1), "p2": s(p ** 2, 2), "q2": s(q ** 2, 2), "den": s(p ** 2 + q ** 2, 2),
        "D": s(D, 4), "P3040": s(P3040, 4), "P1540": s(P1540, 4), "ans": s(ans, 3), "ans4": s(ans, 4),
        "t1": s(p * D, 4), "t2": s(q * P1540, 4),
        "w_D": s(D, 3), "w_nod": s(nodeuce, 3), "w_alt": s(alt, 3), "w_opp": s(1 - ans, 3),
    }


def p11():
    # テーピングと足首の捻挫（コホート研究）
    n1, x1, n0, x0 = 200, 14, 250, 35
    rr = (x1 / n1) / (x0 / n0)
    lnrr = -0.6931
    se = sqrt(1 / x1 - 1 / n1 + 1 / x0 - 1 / n0)
    se_w = sqrt(1 / x1 + 1 / x0)
    z, z1 = 1.96, 1.645
    expo = lambda v: math.exp(v)
    pts = {}

    def lim(c):
        lo, hi = r(lnrr - c, 4), r(lnrr + c, 4)
        pts[lo] = r(expo(lo), 4)
        pts[hi] = r(expo(hi), 4)
        return lo, hi

    ok = lim(r(z * r(se, 4), 4))
    nm = lim(r(z * r(se_w, 4), 4))
    on = lim(r(z1 * r(se, 4), 4))
    keys = sorted(pts)
    d = {
        "n1": n1, "x1": x1, "n0": n0, "x0": x0, "ntot": n1 + n0, "m1": n1 - x1, "m0": n0 - x0,
        "p1": s(x1 / n1, 2), "p0": s(x0 / n0, 2), "rr": s(rr, 2), "lnrr": s(lnrr, 4),
        "v": s(se ** 2, 4), "se": s(se, 4), "hw": s(z * r(se, 4), 4), "z": z,
        "lo": s(ok[0], 4), "hi": s(ok[1], 4), "elo": s(pts[ok[0]], 3), "ehi": s(pts[ok[1]], 3),
        "w1lo": s(pts[nm[0]], 3), "w1hi": s(pts[nm[1]], 3),
        "w2lo": s(rr - z * se, 3), "w2hi": s(rr + z * se, 3),
        "w3lo": s(pts[on[0]], 3), "w3hi": s(pts[on[1]], 3),
        "w4lo": s(ok[0], 3), "w4hi": s(ok[1], 3), "sew": s(se_w, 4),
        "inv1": s(1 / x1, 4), "inv2": s(1 / n1, 4), "inv3": s(1 / x0, 4), "inv4": s(1 / n0, 4),
    }
    for i, k in enumerate(keys):
        d[f"ex{i}"] = s(k, 4)
        d[f"ev{i}"] = s(pts[k], 4)
    return d


def p12():
    m0, tau, sig, n, xbar, c = 180, 6, 8, 4, 172, 170
    prec = 1 / tau ** 2 + n / sig ** 2
    w = (n / sig ** 2) / prec
    pm = m0 + w * (xbar - m0)
    pv = 1 / prec
    predv = pv + sig ** 2
    phi = {0.25: 0.5987, 0.51: 0.6950, 0.56: 0.7123, 1.00: 0.8413, 1.34: 0.9099}
    z_ok = r(abs(c - pm) / sqrt(predv), 2)
    z_post = r(abs(c - pm) / sqrt(pv), 2)
    z_plug = r(abs(c - pm) / sig, 2)
    z_prior = r(abs(c - m0) / sqrt(tau ** 2 + sig ** 2), 2)
    z_data = r(abs(c - xbar) / sig, 2)
    for zz in (z_ok, z_post, z_plug, z_prior, z_data):
        assert zz in phi, zz
    P = lambda zz: 1 - phi[zz]
    return {
        "m0": m0, "tau": tau, "tau2": tau ** 2, "sig": sig, "sig2": sig ** 2, "n": n, "xbar": xbar, "c": c,
        "pr0": s(1 / tau ** 2, 4), "pr1": s(n / sig ** 2, 4), "prec": s(prec, 4), "w": s(w, 4),
        "pm": s(pm, 2), "pv": s(pv, 2), "predv": s(predv, 2), "predsd": s(sqrt(predv), 3), "postsd": s(sqrt(pv), 3),
        "z": s(z_ok, 2), "zraw": s((c - pm) / sqrt(predv), 3),
        "pexact": s(0.5 * (1 + math.erf((c - pm) / sqrt(predv) / sqrt(2))), 3),
        "phi025": s(phi[0.25], 4), "phi051": s(phi[0.51], 4), "phi056": s(phi[0.56], 4),
        "phi100": s(phi[1.00], 4), "phi134": s(phi[1.34], 4),
        "ans": s(P(z_ok), 3), "w_post": s(P(z_post), 3), "w_plug": s(P(z_plug), 3),
        "w_prior": s(P(z_prior), 3), "w_data": s(P(z_data), 3),
        "zpost": s(z_post, 2), "zplug": s(z_plug, 2), "zprior": s(z_prior, 2), "zdata": s(z_data, 2),
        "num": s(m0 / tau ** 2 + xbar * n / sig ** 2, 4),
    }


def values():
    return {f"s09p{i:02d}": f() for i, f in enumerate(
        [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12], start=1)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
