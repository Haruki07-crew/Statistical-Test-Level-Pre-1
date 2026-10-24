"""模擬試験 第14回（set14）の数値。values() が {大問id: {名前: 値}} を返す。

題材：スポーツ・エンタメ・ゲーム・メディア。
問題文に示した値（Φ、e^x、ln、χ² や F の点など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
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


def frac(f):
    """Fraction を LaTeX の分数に。"""
    f = Fr(f)
    if f.denominator == 1:
        return str(f.numerator)
    return f"\\frac{{{f.numerator}}}{{{f.denominator}}}"


# ---------------------------------------------------------------- 大問1 ボードゲームのコイン
def p01():
    half = Fr(1, 2)
    p = half * half              # 1 回の手番で 2 枚とも表
    q = 1 - p
    win = p / (1 - q ** 2)       # 先手が勝つ確率（等比級数）
    two = p + q ** 2 * p         # P の 2 回目の手番までで打ち切り
    return {"half": frac(half), "p": frac(p), "q2": frac(q ** 2), "den": frac(1 - q ** 2), "pfr": frac(win),
            "qfr": frac(1 - win), "ok4": s(float(win), 4),
            "c_ok": s(float(win), 3), "c_first": s(float(p), 3), "c_two": s(float(two), 3),
            "c_q": s(float(1 - win), 3), "c_half": s(0.5, 3)}

# ---------------------------------------------------------------- 大問2 スピードラン
PHI2 = {0.82: 0.7939, 1.16: 0.8770, 1.40: 0.9192, 1.64: 0.9495, 1.98: 0.9761}


def p02():
    # (a) 対数正規の倍率 2 つの積
    mu, sg = 0.1, 0.3
    ln2 = 0.6931
    m2, v2 = 2 * mu, 2 * sg ** 2
    sd2 = r(sqrt(v2), 4)
    z = r((ln2 - m2) / sd2, 2)
    z_add = r((ln2 - m2) / (2 * sg), 2)          # 標準偏差を足した
    z_one = r((ln2 - m2) / sg, 2)                # 分散を 2 倍し忘れた
    z_mu = r((ln2 - mu) / sd2, 2)                # 平均を 2 倍し忘れた
    z_single = r((ln2 - mu) / sg, 2)             # 1 ステージだけ
    for x in (z, z_add, z_one, z_mu, z_single):
        assert x in PHI2, x
    out = {"mu": mu, "sg": sg, "ln2": ln2, "m2": s(m2, 1), "v2": s(v2, 2), "sd2": s(sd2, 4),
           "num": s(ln2 - m2, 4), "zraw": s((ln2 - m2) / sd2, 4), "z": s(z, 2),
           "ok": s(1 - PHI2[z], 3), "ok4": s(1 - PHI2[z], 4), "w_add": s(1 - PHI2[z_add], 3),
           "w_one": s(1 - PHI2[z_one], 3), "w_mu": s(1 - PHI2[z_mu], 3), "w_single": s(1 - PHI2[z_single], 3),
           "zadd": s(z_add, 2), "zone": s(z_one, 2), "zmu": s(z_mu, 2), "zsingle": s(z_single, 2), "sg2": s(2 * sg, 1)}
    for i, (k, v) in enumerate(sorted(PHI2.items()), 1):
        out[f"zt{i}"] = s(k, 2)
        out[f"pt{i}"] = s(v, 4)
    # (b) D = X_A - X_B、P(D > 5)
    ma, mb = 10, 15
    la, lb = Fr(1, ma), Fr(1, mb)
    d = 5
    e05 = 0.6065                 # e^{-0.5}
    e13 = 0.7165                 # e^{-1/3}
    assert float(la * d) == 0.5 and abs(float(lb * d) - 1 / 3) < 1e-12
    pb = lb / (la + lb)
    okb = float(pb) * e05
    out.update({
        "ma": ma, "mb": mb, "la": frac(la), "lb": frac(lb), "d": d, "e05": e05, "e13": e13,
        "lab": frac(la + lb), "c0": frac(la * lb / (la + lb)), "pbd": s(float(pb), 1), "pa": s(float(la / (la + lb)), 1),
        "okb": s(okb, 3), "okb5": s(okb, 5), "w_pb": s(float(pb), 3), "w_e": s(e05, 3),
        "w_swap": s(float(la / (la + lb)) * e05, 3), "w_rate": s(float(pb) * e13, 3), "lad": s(float(la * d), 1),
    })
    return out


# ---------------------------------------------------------------- 大問3 バスケットボール
def p03():
    # (a) 多項分布の 2 つの度数の相関
    ns, p2, p3 = 20, 0.30, 0.15
    p0 = 1 - p2 - p3
    cov = -ns * p2 * p3
    v2 = ns * p2 * (1 - p2)
    v3 = ns * p3 * (1 - p3)
    corr = cov / sqrt(v2 * v3)
    # (b) リーグの選手の成功率のベータ分布をモーメント法で
    m, v = 0.76, 0.0064
    tot = m * (1 - m) / v - 1
    a = m * tot
    b = (1 - m) * tot
    # (c) 全本成功のときのクロッパー＝ピアソン下限
    nf = 12
    r025 = r(0.025 ** (1 / nf), 4)
    r05 = r(0.05 ** (1 / nf), 4)
    r025b = r(0.025 ** (1 / (nf - 1)), 4)
    return {
        "ns": ns, "p2": s(p2, 2), "p3": s(p3, 2), "p0": s(p0, 2), "q2": s(1 - p2, 2), "q3": s(1 - p3, 2),
        "pp": s(p2 * p3, 3), "cov": s(cov, 1), "v2": s(v2, 1), "v3": s(v3, 2), "v23": s(v2 * v3, 2),
        "corr": r(corr, 3), "corr5": s(corr, 5),
        "m": m, "vv": v, "mq": s(m * (1 - m), 4), "ratio": s(m * (1 - m) / v, 1), "tot": s(tot, 1),
        "a": r(a, 1), "a3": s(a, 3), "as": s(a, 1), "b": s(b, 1),
        "w_nom1": s(m * m * (1 - m) / v, 1), "w_sd": s(m * (m * (1 - m) / sqrt(v) - 1), 1),
        "nf": nf, "nf1": nf - 1, "r025": s(r025, 4), "r05": s(r05, 4), "r025b": s(r025b, 4),
        "c_ok": s(r025, 3), "w_one": s(r05, 3), "w_n1": s(r025b, 3), "w_three": s(1 - 3 / nf, 3), "w_wald": s(1, 3),
    }

# ---------------------------------------------------------------- 大問4 ソーシャルゲームの運営
def p04():
    # (a) 一様分布の上端の検定の検出力関数
    n, a = 5, 0.05
    c5 = 1 - a                   # 棄却の境界 c は c^5 = 0.95
    th = 1.2
    th5 = r(th ** n, 5)
    pw = 1 - c5 / th5
    # (b) 確率収束と平均 2 乗収束
    k = 10000
    return {"n": n, "a": a, "c5": s(c5, 2), "th": th, "th5": s(th5, 5), "q": s(c5 / th5, 5),
            "pw": s(pw, 3), "pw5": s(pw, 5), "w_beta": s(c5 / th5, 3), "w_nop": s(1 - c5 / th, 3),
            "w_c1": s(1 - 1 / th5, 3), "w_mul": s(a * th5, 3), "k": k, "pk": s(1 / k, 4)}


# ---------------------------------------------------------------- 大問5 カジノのミニゲーム
def p05():
    p, q = 0.45, 0.55
    i, N = 3, 6
    rr = Fr(55, 45)
    r3 = rr ** 3
    r6 = rr ** 6
    pw = 1 / (1 + r3)                      # (1-r^3)/(1-r^6) = 1/(1+r^3)
    r3s = r(float(r3), 4)
    pw_disp = 1 / (1 + r3s)
    return {
        "p": p, "q": q, "i": i, "N": N, "rr": s(float(rr), 4), "rfr": frac(rr),
        "r3": s(r3s, 4), "r6": s(float(r6), 4), "den": s(1 + r3s, 4),
        "pw": r(pw, 3), "pw5": s(pw_disp, 5), "pws": s(pw, 3), "pfr": frac(pw),
        "w_half": s(i / N, 3), "w_inv": s(1 / (1 + float(1 / rr) ** 3), 3),
    }


# ---------------------------------------------------------------- 大問6 体操競技の採点
J2 = [2, 1, 4, 3, 6, 8, 5, 7]   # 審判2の順位（審判1は A〜H に 1〜8）


def p06():
    n = len(J2)
    d = [i + 1 - j for i, j in enumerate(J2)]
    sd2 = sum(x * x for x in d)
    rs = 1 - Fr(6 * sd2, n * (n * n - 1))
    out = {"nj": n, "sd2": sd2, "six": 6 * sd2, "den": n * (n * n - 1), "rs": r(float(rs), 3), "rs5": s(float(rs), 5),
           "rs_s": s(float(rs), 3), "w_nosix": s(1 - sd2 / (n * (n * n - 1)), 3), "w_n2": s(1 - 6 * sd2 / (n * (n * n + 1)), 3),
           "nn1": n * n - 1}
    for i, v in enumerate(J2):
        out[f"j{i + 1}"] = v
        out[f"d{i + 1}"] = d[i]
    # (b) 混合モデルの二元配置（踏切板＝固定、選手＝変量、各 2 回）
    a, b, rep = 4, 5, 2
    msa, msb, msab, mse = 9.0, 6.0, 3.0, 1.0
    dfa, dfb, dfab, dfe = a - 1, b - 1, (a - 1) * (b - 1), a * b * (rep - 1)
    F = msa / msab
    F_e = msa / mse
    pool = (msab * dfab + mse * dfe) / (dfab + dfe)
    out.update({"ta": a, "tb": b, "rep": rep, "msa": s(msa, 1), "msb": s(msb, 1), "msab": s(msab, 1), "mse": s(mse, 1),
                "dfa": dfa, "dfb": dfb, "dfab": dfab, "dfe": dfe, "F": s(F, 2), "Fe": s(F_e, 2),
                "Fpool": s(msa / pool, 2), "pool": s(pool, 2), "dfp": dfab + dfe, "Fb": s(msa / msb, 2),
                "f312": 3.49, "f320": 3.10, "f332": 2.90, "f312t": 2.61})
    # (c) 等分散の F 検定
    n1, s1, n2, s2 = 11, 0.48, 16, 0.30
    Fv = s1 ** 2 / s2 ** 2
    out.update({"n1": n1, "sP": s1, "n2": n2, "sQ": s2, "v1": s(s1 ** 2, 4), "v2": s(s2 ** 2, 2), "Fv": s(Fv, 2),
                "df1": n1 - 1, "df2": n2 - 1, "Fsd": s(s1 / s2, 2),
                "Fmean": s((s1 ** 2 / n1) / (s2 ** 2 / n2), 2), "f025": 3.06, "f05": 2.54})
    return out


# ---------------------------------------------------------------- 大問7 サッカー選手の年俸
def p07():
    rho, r1, r2 = 0.6, 0.50, 0.45
    vif = 1 / (1 - rho ** 2)
    pair = lambda x, y: f"\\mathrm{{VIF}}={s(x, 2)},\\ \\text{{標準誤差は}}\\ {s(y, 2)}\\ \\text{{倍}}"
    out = {"rho": rho, "r1": r1, "r2": r2, "rho2": s(rho ** 2, 2), "om": s(1 - rho ** 2, 2), "vif": s(vif, 4),
           "sevif": s(sqrt(vif), 2), "vifs": s(vif, 2),
           "v_ok": pair(vif, sqrt(vif)), "v_1r": pair(1 / (1 - rho), sqrt(1 / (1 - rho))),
           "v_nosq": pair(vif, vif), "v_plus": pair(1 + rho ** 2, sqrt(1 + rho ** 2)), "v_inv": pair(1 - rho ** 2, sqrt(1 - rho ** 2))}
    # (b) 主成分回帰（第 1 主成分だけ）
    det0 = 1 - rho ** 2
    ob1 = (r1 - rho * r2) / det0
    ob2 = (r2 - rho * r1) / det0
    cz = (r1 + r2) / sqrt(2)                   # Cov(y, z)
    vz = 1 + rho
    gam = cz / vz
    beta = gam / sqrt(2)                       # = (r1+r2)/{2(1+rho)}
    pr = lambda x, y: f"(\\hat\\beta_1,\\ \\hat\\beta_2)=({s(x, 3)},\\ {s(y, 3)})"
    out.update({"rs": s(r1 + r2, 2), "vz": s(vz, 1), "gam": s(gam, 4), "beta": s(beta, 6), "den2": s(2 * vz, 1),
                "ob1": s(ob1, 3), "ob2": s(ob2, 3),
                "c_ok": pr(beta, beta), "c_ols": pr(ob1, ob2), "c_g": pr(gam, gam),
                "c_nov": pr((r1 + r2) / 2, (r1 + r2) / 2), "c_no2": pr(2 * beta, 2 * beta)})
    return out


# ---------------------------------------------------------------- 大問8 サッカーの試合
def p08():
    # (a) 0 を除いたポアソン分布の期待値
    lh = 1.5
    e0 = 0.2231                  # e^{-1.5}（問題文に示す値）
    p1 = 1 - e0
    ans = lh / p1
    # (b) 負の二項回帰で無得点の確率
    b0, bh, bx, x, th = 0.10, 0.24, 0.30, 0.5, 2.0
    E = {0.25: 1.284, 0.34: 1.405, 0.49: 1.632}      # 問題文に示す e^x の値
    ep = 0.1955                  # e^{-1.632}（問題文に示す値）
    eta = r(b0 + bh + bx * x, 2)
    eta_away = r(b0 + bx * x, 2)
    eta_nox = r(b0 + bh, 2)
    nb0 = lambda mu: (th / (th + mu)) ** th
    mu = E[eta]
    base = th / (th + mu)
    return {"lh": lh, "e0": e0, "p1": s(p1, 4), "ans": r(ans, 3), "ans5": s(ans, 5),
            "w_plus": s(1 + lh, 1), "w_mul": s(lh * p1, 3),
            "phi": 2.25, "b0": s(b0, 2), "bh": s(bh, 2), "bx": s(bx, 2), "x": x, "th": s(th, 1),
            "e1x": s(eta_away, 2), "e1": s(E[eta_away], 3), "e2x": s(eta_nox, 2), "e2": s(E[eta_nox], 3),
            "e3x": s(eta, 2), "e3": s(mu, 3), "ep": ep, "eta": s(eta, 2), "mu": s(mu, 3), "base": s(base, 4),
            "ok5": s(nb0(mu), 5), "c_ok": s(nb0(mu), 3), "w_pois": s(ep, 3), "w_nox": s(nb0(E[eta_nox]), 3),
            "w_away": s(nb0(E[eta_away]), 3), "w_nopow": s(base, 3)}

# ---------------------------------------------------------------- 大問9 eスポーツ
def p09():
    n, p = 36, 5
    rr_, h = 2.6, 0.35
    a1 = n - p - 2
    a2 = n - p - 1
    ratio = r(a1 / (a2 - rr_ ** 2), 4)
    sq = r(sqrt(ratio), 4)
    t = r(rr_ * sq, 4)
    tb, tb30, t025 = 3.54, 3.52, 2.045       # t_{0.05/72}(29), t_{0.05/72}(30), t_{0.025}(29)
    assert abs(t) < tb and abs(t) > t025
    dff_ = t * sqrt(h / (1 - h))
    return {"n": n, "p": p, "r": rr_, "h": h, "r2": s(rr_ ** 2, 2), "a1": a1, "a2": a2, "a2r": s(a2 - rr_ ** 2, 2),
            "ratio": s(ratio, 4), "sq": s(sq, 4), "t": s(t, 4), "t2": s(t, 2), "nn2": 2 * n,
            "tb": s(tb, 2), "tb30": s(tb30, 2), "t025": s(t025, 3),
            "w_inv": s(rr_ * sqrt((a2 - rr_ ** 2) / a1), 2), "h2": s(2 * (p + 1) / n, 3),
            "dffits": s(dff_, 2), "th": s(2 * sqrt((p + 1) / n), 3)}

# ---------------------------------------------------------------- 大問10 動画配信サービス
def p10():
    # (a) 比例配分の層別抽出の標準誤差（有限母集団修正なし）
    N1, N2, n = 8000, 2000, 200
    N = N1 + N2
    W1, W2 = N1 / N, N2 / N
    n1, n2 = round(n * W1), round(n * W2)
    S1, S2 = 5, 15
    t1 = W1 ** 2 * S1 ** 2 / n1
    t2 = W2 ** 2 * S2 ** 2 / n2
    var = t1 + t2
    out = {"N": N, "N1": N1, "N2": N2, "n": n, "n1": n1, "n2": n2, "S1": S1, "S2": S2, "W1": s(W1, 1), "W2": s(W2, 1),
           "t1": s(t1, 3), "t2": s(t2, 3), "var": s(var, 3), "se4": s(sqrt(var), 4), "c_ok": s(sqrt(var), 3),
           "w_var": s(var, 3), "w_sd": s(sqrt(W1 ** 2 * S1 / n1 + W2 ** 2 * S2 / n2), 3),
           "w_w": s(sqrt(W1 * S1 ** 2 / n1 + W2 * S2 ** 2 / n2), 3),
           "w_eq": s(sqrt(0.25 * S1 ** 2 / n1 + 0.25 * S2 ** 2 / n2), 3)}
    # (b) 正準相関（1 変数と 2 変数）
    ry1, ry2, r12 = 0.6, 0.4, 0.3
    num = ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12
    R2 = num / (1 - r12 ** 2)
    out.update({"ry1": ry1, "ry2": ry2, "r12": r12, "cnum": s(num, 3), "cden": s(1 - r12 ** 2, 2), "R2": s(R2, 5),
                "R": s(sqrt(R2), 3), "R4": s(sqrt(R2), 4), "w_sumr": s(sqrt(ry1 ** 2 + ry2 ** 2), 3), "w_R2": s(R2, 3),
                "w_max": s(ry1, 3), "w_nod": s(sqrt(num), 3), "q1": s(ry1 ** 2, 2), "q2": s(ry2 ** 2, 2),
                "q3": s(2 * ry1 * ry2 * r12, 3)})
    # (c) 単位根検定
    g, se = -0.08, 0.03
    tau = g / se
    out.update({"g": g, "gse": se, "tau": s(tau, 2), "rho": s(1 + g, 2), "df5": -2.86, "z5": -1.645})
    return out


# ---------------------------------------------------------------- 大問11 オンライン対戦ゲーム
def p11():
    r12, r13, r23 = 0.6, 0.5, 0.4
    det = 1 + 2 * r12 * r13 * r23 - r12 ** 2 - r13 ** 2 - r23 ** 2
    l1 = 2.004                  # 問題文に示す最大固有値
    ssum = 3 - l1
    prod = r(det / l1, 6)
    disc = r(ssum ** 2 - 4 * prod, 6)
    sq = r(sqrt(disc), 5)
    l2 = (ssum + sq) / 2
    l3 = (ssum - sq) / 2
    pct = lambda x: s(100 * x, 1) + "\\%"
    out = {"r12": r12, "r13": r13, "r23": r23, "det": s(det, 2), "l1": l1, "ssum": s(ssum, 3), "prod": s(prod, 6),
           "ssum2": s(ssum ** 2, 6), "disc": s(disc, 6), "sq": s(sq, 5), "l2": s(l2, 4), "l3": s(l3, 4),
           "cum": s((l1 + l2) / 3, 4), "c_ok": pct((l1 + l2) / 3), "c_1": pct(l1 / 3), "c_eq": pct((l1 + ssum / 2) / 3),
           "c_2": pct(l2 / 3), "c_all": pct((l1 + ssum) / 3), "rr": s(2 * r12 * r13 * r23, 2),
           "q12": s(r12 ** 2, 2), "q13": s(r13 ** 2, 2), "q23": s(r23 ** 2, 2), "fourp": s(4 * prod, 6)}
    # (b) 線形判別関数
    from fractions import Fraction as F_
    m1, m2 = (F_(7), F_(5)), (F_(4), F_(4))
    s11, s12, s22 = F_(2), F_(1), F_(2)
    x = (F_(57, 10), F_(6))
    detS = s11 * s22 - s12 ** 2
    d = (m1[0] - m2[0], m1[1] - m2[1])
    wn = (s22 * d[0] - s12 * d[1], -s12 * d[0] + s11 * d[1])      # det × Σ^{-1}(μ1-μ2)
    c = ((m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2)
    dx = (x[0] - c[0], x[1] - c[1])
    L = (wn[0] * dx[0] + wn[1] * dx[1]) / detS
    L_diag = d[0] / s11 * dx[0] + d[1] / s22 * dx[1]
    L_mu2 = (wn[0] * (x[0] - m2[0]) + wn[1] * (x[1] - m2[1])) / detS
    L_noinv = (s11 * d[0] + s12 * d[1]) * dx[0] + (s12 * d[0] + s22 * d[1]) * dx[1]
    assert L < 0 and L_diag > 0 and L_mu2 > 0 and L_noinv > 0
    fx = lambda v: s(float(v), 1) if v.denominator != 1 else str(v.numerator)
    out.update({"m11": fx(m1[0]), "m12": fx(m1[1]), "m21": fx(m2[0]), "m22": fx(m2[1]),
                "s11": fx(s11), "s12": fx(s12), "s22": fx(s22), "x1": fx(x[0]), "x2": fx(x[1]),
                "d1": fx(d[0]), "d2": fx(d[1]), "detS": fx(detS), "wn1": fx(wn[0]), "wn2": fx(wn[1]),
                "c1": fx(c[0]), "c2": fx(c[1]), "dx1": fx(dx[0]), "dx2": fx(dx[1]),
                "lnum": s(float(wn[0] * dx[0] + wn[1] * dx[1]), 1), "c_val": s(float(L), 2), "c_val4": s(float(L), 4),
                "w_diag": s(float(L_diag), 2), "w_sign": s(float(-L), 2), "w_mu2": s(float(L_mu2), 2),
                "w_noinv": s(float(L_noinv), 2)})
    return out


# ---------------------------------------------------------------- 大問12 ゲームアプリの継続
KM = [(2, 10, 1, 0), (7, 9, 0, 1), (9, 8, 1, 0), (12, 7, 0, 1), (14, 6, 1, 0)]   # (日, 在籍, やめた人数, 打ち切り)


def p12():
    out = {}
    S = 1.0
    gw = 0.0
    gw2 = 0.0
    for j, (day, n, d, c) in enumerate(KM, 1):
        out.update({f"day{j}": day, f"n{j}": n, f"d{j}": d, f"c{j}": c})
        if j < len(KM):
            assert KM[j][1] == n - d - c
        if d == 0:
            continue
        S *= (n - d) / n
        gw += d / (n * (n - d))
        gw2 += d / (n * n)
    Sr = r(S, 5)
    gwr = r(gw, 6)
    var = Sr ** 2 * gwr
    se = sqrt(var)
    out.update({"S": s(Sr, 5), "S2": s(Sr ** 2, 6), "gw": s(gwr, 6), "g1": s(1 / 90, 6), "g3": s(1 / 56, 6),
                "g4": s(1 / 30, 6), "var": s(var, 6), "se": s(se, 3), "se5": s(se, 5),
                "w_var": s(var, 3), "w_noS": s(sqrt(gwr), 3), "w_n2": s(Sr * sqrt(gw2), 3), "w_S": s(Sr, 3)})
    # (b) ブートストラップ
    n = 10
    q10 = 0.3487            # 0.9^10（問題文に示す値）
    out.update({"bn": n, "q10": q10, "bs": s(n * (1 - q10), 2), "bs_out": s(n * q10, 2),
                "bs_e": s(n * (1 - 0.3679), 2), "bs_one": s(n * 0.9, 2), "bs_all": s(n, 2), "inc": s(1 - q10, 4)})
    return out


def values():
    return {"s14p01": p01(), "s14p02": p02(), "s14p03": p03(), "s14p04": p04(), "s14p05": p05(),
            "s14p06": p06(), "s14p07": p07(), "s14p08": p08(), "s14p09": p09(), "s14p10": p10(),
            "s14p11": p11(), "s14p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
