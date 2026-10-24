"""模擬試験 第26回（set26）の数値。values() が {大問id: {名前: 値}} を返す。

題材：通信ネットワーク・サイバーセキュリティ・データセンター。
問題文に示した分布の値（Φ・χ²・t の点、指数、arctan など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    # 4 本のケーブルを無作為に差し直したとき、1 本も元のポートに戻らない確率（包除原理）
    n = 4
    terms = [(-1) ** (k + 1) / math.factorial(k) for k in range(1, n + 1)]
    at_least = sum(terms)
    none = 1 - at_least
    return {
        "n": n, "nf": math.factorial(n), "p1": s(1 / 4, 2), "p2": s(1 / 12, 4), "p3": s(1 / 24, 4), "p4": s(1 / 24, 4),
        "s1": s(1, 0), "s2": s(1 / 2, 4), "s3": s(1 / 6, 4), "s4": s(1 / 24, 4),
        "atl": s(at_least, 4), "none": r(none, 3), "none_s": s(none, 3), "none4": s(none, 4),
        "w_ind": s((3 / 4) ** 4, 3), "w_e": s(math.exp(-1), 3), "w_atl": s(at_least, 3), "w_no4": s(1 - (1 - 1 / 2), 3),
        "num": round(none * 24),
    }


def p02():
    # ジャック・ベラ
    n, sk, ku = 200, 0.21, 3.30
    t1 = sk ** 2
    t2 = (ku - 3) ** 2 / 4
    jb = n / 6 * (t1 + t2)
    jb_ex = n / 6 * (sk ** 2 + ku ** 2 / 4)       # 尖度 3.30 を超過尖度とみた
    jb_no4 = n / 6 * (sk ** 2 + (ku - 3) ** 2)     # 4 で割り忘れ
    jb_no6 = n * (t1 + t2)                          # 6 で割り忘れ
    jb_24 = n / 24 * (t1 + t2)                      # 6 と 24 の取り違え
    c2 = 5.991
    assert jb < c2 and jb_no4 < c2 and jb_24 < c2 and jb_ex > c2 and jb_no6 > c2
    return {
        "n": n, "sk": s(sk, 2), "ku": s(ku, 2), "t1": s(t1, 4), "t2": s(t2, 4), "tsum": s(t1 + t2, 4),
        "n6": s(n / 6, 3), "jb": s(jb, 2), "jb_ex": s(jb_ex, 2), "jb_no4": s(jb_no4, 2), "jb_no6": s(jb_no6, 2),
        "jb_24": s(jb_24, 2), "c2": c2, "c1": 3.841, "kum3": s(ku - 3, 2),
    }


def p03():
    # 1 標本の比率の片側検定に必要な標本の大きさ（H0 と H1 で分散を分ける）
    p0, p1 = 0.20, 0.12
    za, zb, z2 = 1.645, 0.842, 1.960
    zb9 = 1.282
    s0, s1 = sqrt(p0 * (1 - p0)), sqrt(p1 * (1 - p1))
    d = p0 - p1
    n_of = lambda a, b, sa, sb: ((a * sa + b * sb) / d) ** 2
    nn = n_of(za, zb, s0, s1)
    return {
        "p0": s(p0, 2), "p1": s(p1, 2), "q0": s(1 - p0, 2), "q1": s(1 - p1, 2), "za": za, "zb": zb, "z2": s(z2, 3), "zb9": zb9,
        "v0": s(p0 * (1 - p0), 4), "v1": s(p1 * (1 - p1), 4), "s0": s(s0, 4), "s1": s(s1, 5), "d": s(d, 2),
        "a0": s(za * s0, 4), "b1": s(zb * s1, 5), "num": s(za * s0 + zb * s1, 5), "ratio": s((za * s0 + zb * s1) / d, 4),
        "nraw": s(nn, 2), "n": math.ceil(nn),
        "w_h0": math.ceil(n_of(za, zb, s0, s0)), "w_h1": math.ceil(n_of(za, zb, s1, s1)),
        "w_two": math.ceil(n_of(z2, zb, s0, s1)), "w_p90": math.ceil(n_of(za, zb9, s0, s1)),
    }


PHI4 = {0.50: 0.6915, 1.67: 0.9525, 2.00: 0.9772}


def p04():
    # (b) コーシー分布の標本平均
    gam, n, thr = 0.5, 16, 1.0
    pi = 3.1416
    at = {2: 1.1071, 8: 1.4464, 32: 1.5396, 0.5: 0.4636}
    ok = 1 - 2 / pi * at[2]
    w_clt = 1 - 2 / pi * at[8]        # 尺度を γ/√n とした
    w_n = 1 - 2 / pi * at[32]         # 尺度を γ/n とした
    w_one = 1 / 2 - at[2] / pi        # 片側だけ
    w_inv = 1 - 2 / pi * at[0.5]      # x/γ を γ/x と取り違えた
    # (c) ブラウン運動の最大値（反射原理）
    sig2, T, a = 0.09, 16, 2.4
    sd = sqrt(sig2 * T)
    z = r(a / sd, 2)
    z_var = r(a / (sig2 * T), 2)
    z_t = r(a / (sqrt(sig2) * T), 2)
    for zz in (z, z_var, z_t):
        assert zz in PHI4, zz
    pc = 2 * (1 - PHI4[z])
    return {
        "gam": s(gam, 1), "n": n, "thr": s(thr, 0), "pi": s(pi, 4),
        "at2": s(at[2], 4), "at8": s(at[8], 4), "at32": s(at[32], 4), "at05": s(at[0.5], 4),
        "ratio": s(thr / gam, 0), "ok": s(ok, 3), "ok4": s(ok, 4), "two_over_pi": s(2 / pi * at[2], 4),
        "w_clt": s(w_clt, 3), "w_n": s(w_n, 3), "w_one": s(w_one, 3), "w_inv": s(w_inv, 3),
        "sc_clt": s(gam / sqrt(n), 3), "sc_n": s(gam / n, 4),
        "sig2": s(sig2, 2), "sig": s(sqrt(sig2), 1), "T": T, "a": s(a, 1), "varT": s(sig2 * T, 2), "sdT": s(sd, 1),
        "z": s(z, 2), "phi_z": s(PHI4[z], 4), "tail": s(1 - PHI4[z], 4), "pc": s(pc, 3), "pc4": s(pc, 4),
        "w_half": s(1 - PHI4[z], 3), "w_var": s(2 * (1 - PHI4[z_var]), 3), "w_t": s(2 * (1 - PHI4[z_t]), 3),
        "w_never": s(1 - pc, 3), "z_var": s(z_var, 2), "z_t": s(z_t, 2),
        "phA": s(PHI4[0.50], 4), "phB": s(PHI4[1.67], 4), "phC": s(PHI4[2.00], 4),
    }


def p05():
    # パレート分布の形状母数：モーメント法と最尤法
    xm, n, xbar, slog = 0.4, 25, 0.8, 10.0
    mom = xbar / (xbar - xm)
    mle = n / slog
    return {
        "xm": s(xm, 1), "n": n, "xbar": s(xbar, 1), "slog": s(slog, 1), "dx": s(xbar - xm, 1),
        "mom": s(mom, 2), "mle": s(mle, 2), "mlog": s(slog / n, 2),
        "w_inv": s(slog / n, 2), "w_mom": s(xm / (xbar - xm), 2), "w_unb": s((n - 1) / slog, 2),
        "se": s(mle / sqrt(n), 2),
    }


def p06():
    # (a) 相関のある 16 ラックの合計電力の変動係数
    k, mu, sd, rho = 16, 8.0, 2.4, 0.1
    var = k * sd ** 2 + k * (k - 1) * rho * sd ** 2
    cv = sqrt(var) / (k * mu)
    cv_ind = sqrt(k) * sd / (k * mu)
    # (b) ジャックナイフ
    th = [1.52, 1.48, 1.55, 1.50, 1.45]
    m = len(th)
    tb = sum(th) / m
    ssd = sum((t - tb) ** 2 for t in th)
    se = sqrt((m - 1) / m * ssd)
    out = {
        "k": k, "mu": s(mu, 1), "sd": s(sd, 1), "rho": s(rho, 1), "cv1": s(sd / mu, 2), "sd2": s(sd ** 2, 2),
        "pairs": k * (k - 1), "cnt": s(k + k * (k - 1) * rho, 0), "var": s(var, 2), "sdt": s(sqrt(var), 4),
        "meant": s(k * mu, 0), "cv": r(cv, 3), "cv4": s(cv, 4), "w_ind": s(cv_ind, 3), "w_same": s(sd / mu, 3),
        "w_half": s(sqrt(k * sd ** 2 + k * (k - 1) / 2 * rho * sd ** 2) / (k * mu), 3),
        "m": m, "tb": s(tb, 2), "ssd": s(ssd, 4), "fac": s((m - 1) / m, 1), "jv": s((m - 1) / m * ssd, 5),
        "se": s(se, 3), "se4": s(se, 4),
        "w_ord": s(sqrt(ssd / (m - 1)) / sqrt(m), 3), "w_nof": s(sqrt(ssd), 3), "w_1n": s(sqrt(ssd / m), 3),
        "w_nm1": s(sqrt((m - 1) * ssd), 3),
    }
    for i, t in enumerate(th):
        out[f"th{i}"] = s(t, 2)
        out[f"dv{i}"] = s(t - tb, 2)
    return out


def p07():
    # (a) 中央値検定（12 個ずつ。全体の中央値より大きい個数）
    nL = nM = 12
    aL, aM = 3, 9
    bL, bM = nL - aL, nM - aM
    N = nL + nM
    ad_bc = aL * bM - aM * bL
    chi = N * ad_bc ** 2 / (nL * nM * (aL + aM) * (bL + bM))
    E_ = (aL + aM) * nL / N
    obs = [aL, aM, bL, bM]
    yates = N * (abs(ad_bc) - N / 2) ** 2 / (nL * nM * (aL + aM) * (bL + bM))
    raw = sum((o - E_) ** 2 for o in obs)
    neyman = sum((o - E_) ** 2 / o for o in obs)
    half = sum((o - E_) ** 2 / E_ for o in obs[:2])
    # (b) プールした分散による t 区間
    mL, vL, mM, vM = 3.75, 0.315, 4.50, 0.300
    sp2 = ((nL - 1) * vL + (nM - 1) * vM) / (nL + nM - 2)
    inv = 1 / nL + 1 / nM
    se = sqrt(sp2 * inv)
    t22, t11, z = 2.074, 2.201, 1.960
    diff = mM - mL
    ci = lambda hw: f"({s(diff - hw, 2)},\\ {s(diff + hw, 2)})"
    sp2n = ((nL - 1) * vL + (nM - 1) * vM) / (nL + nM)
    return {
        "nL": nL, "nM": nM, "N": N, "aL": aL, "aM": aM, "bL": bL, "bM": bM, "ra": aL + aM, "rb": bL + bM,
        "E": s(E_, 0), "adbc": ad_bc, "adbc2": ad_bc ** 2, "den": nL * nM * (aL + aM) * (bL + bM),
        "chi": s(chi, 2), "yates": s(yates, 2), "raw": s(raw, 2), "neyman": s(neyman, 2), "half": s(half, 2),
        "c1": 3.841, "dev2": s((aL - E_) ** 2, 0),
        "mL": s(mL, 2), "mM": s(mM, 2), "vL": s(vL, 3), "vM": s(vM, 3), "ssL": s((nL - 1) * vL, 3), "ssM": s((nM - 1) * vM, 3),
        "df": nL + nM - 2, "n1": nL - 1,
        "sp2": s(sp2, 4), "inv": s(inv, 4), "se2": s(sp2 * inv, 5), "se": s(se, 4),
        "t22": t22, "t11": t11, "z": s(z, 3), "diff": s(diff, 2), "hw": s(t22 * se, 4),
        "ci": ci(t22 * se), "ci_z": ci(z * se), "ci_t11": ci(t11 * se), "ci_nof": ci(t22 * sqrt(sp2)),
        "ci_n": ci(t22 * sqrt(sp2n * inv)), "sp2n": s(sp2n, 4),
    }


def p08():
    # (a) VIF
    r2 = {1: 0.92, 2: 0.60, 3: 0.88}
    vif1 = 1 / (1 - r2[1])
    # (b) 区分線形スプライン
    b0, b1, b2, k, sla = 20, 0.5, 3.0, 70, 120
    yk = b0 + b1 * k
    xs = k + (sla - yk) / (b1 + b2)
    xs_b2 = k + (sla - yk) / b2
    xs_lin = (sla - b0) / b1
    # (c) PRESS
    x = [10, 20, 30, 40, 50, 60]
    y = [42.0, 47.0, 58.0, 61.0, 71.0, 72.0]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    bb = sxy / sxx
    aa = my - bb * mx
    e = [b - (aa + bb * a) for a, b in zip(x, y)]
    h = [1 / n + (a - mx) ** 2 / sxx for a in x]
    e_r = [r(v, 2) for v in e]
    h_r = [r(v, 3) for v in h]
    pr = [ev / (1 - hv) for ev, hv in zip(e_r, h_r)]
    press = sum(v ** 2 for v in pr)
    sse = sum(v ** 2 for v in e_r)
    w_h = sum((ev * (1 - hv)) ** 2 for ev, hv in zip(e_r, h_r))
    w_nosq = sum((ev / (1 - hv) ** 2) ** 2 for ev, hv in zip(e_r, h_r))
    out = {
        "r1": s(r2[1], 2), "r2": s(r2[2], 2), "r3": s(r2[3], 2), "n": 40,
        "vif1": s(vif1, 1), "sevif": s(sqrt(vif1), 2), "w_inv": s(1 / r2[1], 2), "w_invse": s(sqrt(1 / r2[1]), 2),
        "w_1mr": s(1 - r2[1], 2), "w_1mrse": s(sqrt(1 - r2[1]), 2), "w_r": s(1 / (1 - r2[1] ** 2), 2),
        "w_rse": s(sqrt(1 / (1 - r2[1] ** 2)), 2), "omr": s(1 - r2[1], 2),
        "vif2": s(1 / (1 - r2[2]), 1), "vif3": s(1 / (1 - r2[3]), 2),
        "b0": b0, "b1": s(b1, 1), "b2": s(b2, 1), "k": k, "sla": sla, "yk": s(yk, 0), "slope": s(b1 + b2, 1),
        "gap": s(sla - yk, 0), "xs": r(xs, 1), "xs4": s(xs, 3), "xs_b2": s(xs_b2, 1), "xs_lin": s(xs_lin, 0),
        "press": r(press, 1), "press1": s(press, 1), "press_s": s(press, 2), "sse": s(sse, 2), "w_h": s(w_h, 2), "w_nosq": s(w_nosq, 2),
        "a_hat": s(aa, 2), "b_hat": s(bb, 2),
    }
    for i in range(n):
        out[f"x{i}"] = x[i]
        out[f"e{i}"] = s(e_r[i], 2)
        out[f"h{i}"] = s(h_r[i], 3)
        out[f"omh{i}"] = s(1 - h_r[i], 3)
        out[f"pr{i}"] = s(pr[i], 4)
        out[f"pq{i}"] = s(pr[i] ** 2, 4)
    out["_e"] = e_r
    out["_h"] = h_r
    out["_y"] = y
    return out


def p09():
    y = [12, 7, 20, 9]
    mu = [10.0, 9.0, 18.0, 11.0]
    assert sum(y) == sum(mu)
    lnr = {0: 0.1823, 1: -0.2513, 2: 0.1054, 3: -0.2007}   # ln(y/μ) を小数第4位に丸めた表の値
    terms = [y[i] * lnr[i] for i in range(4)]
    D = 2 * sum(terms)
    X2 = sum((a - b) ** 2 / b for a, b in zip(y, mu))
    out = {"D": s(D, 2), "D4": s(D, 4), "half": s(D / 2, 2), "X2": s(X2, 2), "negD": s(-D, 2), "df": 2, "c2": 5.991, "c3": 7.815,
           "sumy": sum(y), "summu": s(sum(mu), 0)}
    for i in range(4):
        out[f"y{i}"] = y[i]
        out[f"mu{i}"] = s(mu[i], 1)
        out[f"l{i}"] = s(lnr[i], 4)
        out[f"tm{i}"] = s(terms[i], 4)
    out["tsum"] = s(sum(terms), 4)
    return out


def p10():
    # 時刻（月）と事象（1=故障, 0=打ち切り）
    data = [(5, 1), (8, 0), (12, 1), (12, 1), (15, 0), (20, 1), (24, 0), (30, 0)]
    steps = [(8, 1), (6, 2), (3, 1)]   # (リスク集合の大きさ, 故障数) 時刻 5, 12, 20
    S = 1.0
    g = 0.0
    for nn, d in steps:
        S *= (nn - d) / nn
        g += d / (nn * (nn - d))
    var = S ** 2 * g
    se = sqrt(var)
    w_binom = sqrt(S * (1 - S) / 8)
    return {
        "S": s(S, 4), "S2": s(S ** 2, 4), "g1": s(1 / (8 * 7), 4), "g2": s(2 / (6 * 4), 4), "g3": s(1 / (3 * 2), 4),
        "g": s(g, 4), "var": s(var, 4), "se": r(se, 3), "se4": s(se, 4), "w_binom": s(w_binom, 3),
        "w_var": s(var, 3), "w_noS": s(sqrt(g), 3),
        "_data": data,
    }


def p11():
    ssR, ssA, ssE1, ssB, ssAB, ssE2 = 18.0, 42.0, 9.0, 12.0, 4.0, 9.0
    dfR, dfA, dfE1, dfB, dfAB, dfE2 = 3, 2, 6, 1, 2, 9
    msA = ssA / dfA
    F = msA / (ssE1 / dfE1)
    F2 = msA / (ssE2 / dfE2)
    Fp = msA / ((ssE1 + ssE2) / (dfE1 + dfE2))
    return {
        "ssR": s(ssR, 1), "ssA": s(ssA, 1), "ssE1": s(ssE1, 1), "ssB": s(ssB, 1), "ssAB": s(ssAB, 1), "ssE2": s(ssE2, 1),
        "ssT": s(ssR + ssA + ssE1 + ssB + ssAB + ssE2, 1),
        "dfR": dfR, "dfA": dfA, "dfE1": dfE1, "dfB": dfB, "dfAB": dfAB, "dfE2": dfE2, "dfT": 23, "dfEp": dfE1 + dfE2,
        "msA": s(msA, 1), "msE1": s(ssE1 / dfE1, 1), "msE2": s(ssE2 / dfE2, 1), "msEp": s((ssE1 + ssE2) / 15, 1),
        "F": s(F, 1), "F2": s(F2, 1), "Fp": s(Fp, 1), "Fraw": s(ssA / ssE1, 2),
        "f26": 5.14, "f29": 4.26, "f215": 3.68,
    }




def p12():
    # ラックを集落とする集落抽出で故障ディスクの割合を比推定し標準誤差を求める
    Nc, nc = 200, 5
    m = [20, 16, 24, 20, 20]
    y = [0, 1, 6, 1, 2]
    p = sum(y) / sum(m)
    mbar = sum(m) / nc
    res = [yi - p * mi for yi, mi in zip(y, m)]
    s2 = sum(v ** 2 for v in res) / (nc - 1)
    f = nc / Nc
    var = (1 - f) * s2 / (nc * mbar ** 2)
    se = sqrt(var)
    out = {
        "Nc": Nc, "nc": nc, "sumy": sum(y), "summ": sum(m), "p": s(p, 2), "mbar": s(mbar, 0), "mbar2": s(mbar ** 2, 0),
        "ss": s(sum(v ** 2 for v in res), 2), "s2": s(s2, 3), "f": s(f, 3), "omf": s(1 - f, 3), "nm2": s(nc * mbar ** 2, 0),
        "var": s(var, 7), "se": r(se, 3), "se4": s(se, 4),
        "w_bin": s(sqrt(p * (1 - p) / sum(m)), 3), "w_nofpc": s(sqrt(s2 / (nc * mbar ** 2)), 4),
        "deff": s(var / (p * (1 - p) / sum(m)), 2),
    }
    for i in range(nc):
        out[f"m{i}"] = m[i]
        out[f"y{i}"] = y[i]
        out[f"pm{i}"] = s(p * m[i], 1)
        out[f"res{i}"] = s(res[i], 1)
        out[f"rq{i}"] = s(res[i] ** 2, 2)
    return out


def p13():
    # (a) AR(1) のスペクトル
    phi = 0.8
    ratio = ((1 + phi) / (1 - phi)) ** 2
    # (b) 等相関 3 指標の主成分と SPE（Q 統計量）
    rho = 0.6
    l1, l2 = 1 + 2 * rho, 1 - rho
    pts = {"A": (2.0, 2.1, 1.9), "B": (1.0, -1.0, 0.5), "C": (0.5, 0.4, -0.6), "D": (-1.5, -1.4, -1.0)}
    qlim = 1.5
    out = {
        "phi": s(phi, 1), "opp": s(1 + phi, 1), "omp": s(1 - phi, 1), "ratio": s(ratio, 0), "rt": s((1 + phi) / (1 - phi), 0),
        "w_inv": s(1 / ratio, 4), "w_1mp": s(1 / (1 - phi) ** 2, 0),
        "rho": s(rho, 1), "l1": s(l1, 1), "l2": s(l2, 1), "qlim": s(qlim, 1),
    }
    flag = []
    for k, z in pts.items():
        ss = sum(v ** 2 for v in z)
        sm = sum(z)
        t1sq = sm ** 2 / 3
        q = ss - t1sq
        out[f"{k}1"], out[f"{k}2"], out[f"{k}3"] = (s(v, 1) for v in z)
        out[f"{k}ss"] = s(ss, 2)
        out[f"{k}sum"] = s(sm, 1)
        out[f"{k}t1"] = s(t1sq, 4)
        out[f"{k}q"] = s(q, 4)
        if q > qlim:
            flag.append(k)
    assert flag == ["B"], flag
    # (c) 非心カイ二乗分布
    mx, my = 1.2, 1.6
    lam = mx ** 2 + my ** 2
    kdf = 2
    E = kdf + lam
    V = 2 * (kdf + 2 * lam)
    out.update({
        "mx": s(mx, 1), "my": s(my, 1), "mx2": s(mx ** 2, 2), "my2": s(my ** 2, 2), "lam": s(lam, 0),
        "E": s(E, 0), "V": s(V, 0), "vx": s(2 + 4 * mx ** 2, 2), "vy": s(2 + 4 * my ** 2, 2),
        "w_cE": s(kdf, 0), "w_cV": s(2 * kdf, 0), "w_V2": s(2 * (kdf + lam), 0), "w_E2": s(kdf + 2 * lam, 0),
        "w_V0": s(2 * kdf, 0), "sdR": s(sqrt(V), 2),
    })
    return out


def p14():
    # (a) コサイン類似度
    u = (8, 4, 0, 2)
    v = (4, 2, 0, 1)
    w = (7, 6, 2, 3)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    nu, nv, nw = sqrt(dot(u, u)), sqrt(dot(v, v)), sqrt(dot(w, w))
    cw = dot(u, w) / (nu * nw)
    eu = lambda a, b: sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    out = {
        "u": ",\\ ".join(map(str, u)), "v": ",\\ ".join(map(str, v)), "w": ",\\ ".join(map(str, w)),
        "uu": dot(u, u), "vv": dot(v, v), "ww": dot(w, w), "uw": dot(u, w), "uv": dot(u, v),
        "nunw": s(nu * nw, 2), "cw": r(cw, 3), "cw4": s(cw, 4), "cv": s(dot(u, v) / (nu * nv), 1),
        "duv2": round(eu(u, v) ** 2), "duw2": round(eu(u, w) ** 2), "duv": s(eu(u, v), 2), "duw": s(eu(u, w), 2),
        "w_dot": dot(u, w), "w_nosqrt": s(dot(u, w) / (dot(u, u) * dot(w, w)), 4),
    }
    # (b) ガウスカーネルの SVM
    alpha = {"a": 1.5, "b": 0.9, "c": 0.6}
    lab = {"a": 1, "b": -1, "c": -1}
    sv = {"a": (0, 0), "b": (2, 0), "c": (0, 2)}
    b0 = -0.1
    x = (0.5, 1.0)
    assert abs(sum(alpha[k] * lab[k] for k in alpha)) < 1e-12
    ex = {1.25: 0.2865, 3.25: 0.0388}
    d2 = {k: (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 for k, p in sv.items()}
    K = {k: ex[d2[k]] for k in sv}
    fx = sum(alpha[k] * lab[k] * K[k] for k in sv) + b0
    f_nolab = sum(alpha[k] * K[k] for k in sv) + b0
    f_noal = sum(lab[k] * K[k] for k in sv) + b0
    f_nob = fx - b0
    assert fx > 0 and f_noal < 0
    out.update({
        "aa": s(alpha["a"], 1), "ab": s(alpha["b"], 1), "ac": s(alpha["c"], 1), "b0": s(b0, 1), "x1": s(x[0], 1), "x2": s(x[1], 1),
        "da": s(d2["a"], 2), "db": s(d2["b"], 2), "dc": s(d2["c"], 2), "ka": s(K["a"], 4), "kb": s(K["b"], 4), "kc": s(K["c"], 4),
        "ta": s(alpha["a"] * K["a"], 5), "tb": s(-alpha["b"] * K["b"], 5), "tc": s(-alpha["c"] * K["c"], 5),
        "f": s(fx, 2), "f4": s(fx, 4), "f_nolab": s(f_nolab, 2), "f_noal": s(f_noal, 2), "f_nob": s(f_nob, 2),
    })
    return out


def values():
    out = {}
    for i, f in enumerate([p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14], start=1):
        v = f()
        out[f"s26p{i:02d}"] = {k: val for k, val in v.items() if not k.startswith("_")}
    return out


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
