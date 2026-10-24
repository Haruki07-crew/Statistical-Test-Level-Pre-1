"""模擬試験 第19回（set19）の数値。values() が {大問id: {名前: 値}} を返す。

題材：銀行・与信・決済・家計。
問題文に示した分布の値（χ² の点・分布関数の値・t 点・指数や対数の値など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
import math
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import combinations
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
    # (a) 大きな標本での有意性と差の大きさ
    n_, ph, p0_ = 90000, 0.052, 0.05
    se0 = sqrt(p0_ * (1 - p0_) / n_)
    z = (ph - p0_) / se0
    ph275 = 0.9970
    se1 = sqrt(ph * (1 - ph) / n_)
    half = 1.96 * se1
    assert s(z, 2) == "2.75"
    # (b) 結果に依存した抽出（延滞者は全件、非延滞者は 10%）のロジスティック回帰の切片補正
    b0, b1, x = 2.0, 0.06, 60
    f1, f0 = 1.0, 0.10
    eta = b0 - b1 * x
    e16 = 4.9530
    odds_s = 1 / e16
    odds_p = odds_s * f0 / f1
    p_ok = odds_p / (1 + odds_p)
    p_raw = odds_s / (1 + odds_s)
    odds_rev = odds_s * f1 / f0
    return {
        "sn": n_, "ph": s(ph * 100, 1), "p0": s(p0_ * 100, 1), "se0": s(se0, 7), "z": s(z, 2), "z3": s(z, 3),
        "ph275": s(ph275, 4), "pv": s(2 * (1 - ph275), 3), "se1": s(se1, 7), "half": s(half * 100, 2),
        "lo": s((ph - half) * 100, 2), "hi": s((ph + half) * 100, 2), "dpt": s((ph - p0_) * 100, 1),
        "dlo": s((ph - half - p0_) * 100, 2), "dhi": s((ph + half - p0_) * 100, 2),
        "b0": s(b0, 1), "b1": s(b1, 2), "x": x, "eta": s(eta, 1), "e16": s(e16, 4), "f0": s(f0, 2),
        "odds_s": s(odds_s, 5), "odds_p": s(odds_p, 6), "den": s(1 + 10 * e16, 2),
        "p_ok": s(p_ok, 4), "p_ok5": s(p_ok, 5), "p_raw": s(p_raw, 3), "p_raw10": s(p_raw * f0, 4),
        "p_odds": s(odds_p, 4), "p_rev": s(odds_rev / (1 + odds_rev), 3), "ln10": 2.3026,
    }


def p02():
    # (a) 2 標本 t 検定と信頼区間の対応
    n1 = n2 = 20
    m1, m2, sp = 12.4, 10.1, 3.2
    t025, t005 = 2.024, 2.712
    d = m1 - m2
    se = sp * sqrt(1 / n1 + 1 / n2)
    t = d / se
    se_w = sp / sqrt(n1)
    # (b) ずれのある指数分布の最尤推定
    xs = [3.2, 4.1, 3.5, 6.8, 5.0, 9.4, 3.9, 4.5]
    nn = len(xs)
    mn = min(xs)
    xb = sum(xs) / nn
    th = xb - mn
    sd = sqrt(sum((v - xb) ** 2 for v in xs) / (nn - 1))
    thu = nn / (nn - 1) * th
    # (c) 条件付き独立でない2つのルール
    pf, both_f, both_l = 0.002, 0.75, 0.004
    sa_f, sb_f, sa_l, sb_l = 0.90, 0.80, 0.02, 0.03
    post = pf * both_f / (pf * both_f + (1 - pf) * both_l)
    post_ind = pf * sa_f * sb_f / (pf * sa_f * sb_f + (1 - pf) * sa_l * sb_l)
    post_a = pf * sa_f / (pf * sa_f + (1 - pf) * sa_l)
    return {
        "n1": n1, "n2": n2, "df": n1 + n2 - 2, "m1": s(m1, 1), "m2": s(m2, 1), "sp": s(sp, 1), "d": s(d, 1),
        "t025": s(t025, 3), "t005": s(t005, 3), "se": s(se, 4), "t": s(t, 2), "h95": s(t025 * se, 3), "h99": s(t005 * se, 3),
        "lo95": s(d - t025 * se, 2), "hi95": s(d + t025 * se, 2), "lo99": s(d - t005 * se, 2), "hi99": s(d + t005 * se, 2),
        "wlo": s(d - t025 * se_w, 2), "whi": s(d + t025 * se_w, 2),
        "xs": "、".join(str(v) for v in xs), "nn": nn, "sum": s(sum(xs), 1), "mn": s(mn, 1), "xb": s(xb, 2), "th": s(th, 2),
        "sd": s(sd, 2), "mm_mu": s(xb - sd, 2), "thu": s(thu, 2), "muu": s(mn - thu / nn, 2),
        "pf": pf, "both_f": both_f, "both_l": both_l, "sa_f": s(sa_f, 2), "sb_f": s(sb_f, 2), "sa_l": s(sa_l, 2), "sb_l": s(sb_l, 2),
        "nf": s(pf * both_f, 4), "nl": s((1 - pf) * both_l, 6), "dsum": s(pf * both_f + (1 - pf) * both_l, 6),
        "post": s(post, 3), "post4": s(post, 4), "w_ind": s(post_ind, 3), "w_a": s(post_a, 3), "w_like": s(both_f, 3),
        "w_noprior": s(both_f / (both_f + both_l), 3), "prodl": s(sa_l * sb_l, 4), "prodf": s(sa_f * sb_f, 2),
    }


def p03():
    lam, delta = Fraction(1, 5), Fraction(1, 20)
    k = lam / delta
    ed = lam / (lam + delta)
    ed2 = lam / (lam + 2 * delta)
    vd = ed2 - ed ** 2
    vpv = 100 ** 2 * vd
    sd = sqrt(float(vpv))
    w_plus = lam / (lam - delta)
    return {
        "lam": s(lam, 1), "mean": int(1 / lam), "delta": s(delta, 2), "k": int(k), "km1": int(k - 1),
        "inv": s(delta / lam, 2), "invm1": s(delta / lam - 1, 2),
        "ed": s(ed, 1), "ed2_f": "2/3", "ed_sq": "16/25", "vd_f": "2/75",
        "vpv": s(vpv, 2), "sd": r(sd, 2), "sd_s": s(sd, 2), "sd4": s(sd, 4),
        "w_var": s(float(vpv), 2), "w_nosq": s(sqrt(float(100 ** 2 * ed2)), 2), "w_plus": s(w_plus, 3),
        "pv_mean": s(100 * ed, 0),
    }


def p04():
    # (a) 回帰推定量に必要な標本の大きさ
    n0, rho = 400, 0.8
    nreg = n0 * (1 - rho ** 2)
    return {
        "n0": n0, "rho": rho, "rho2": s(rho ** 2, 2), "f": s(1 - rho ** 2, 2), "nreg": r(nreg, 0),
        "w_1r": r(n0 * (1 - rho), 0), "w_r2": r(n0 * rho ** 2, 0), "w_sq": r(n0 * sqrt(1 - rho ** 2), 0),
        "w_div": r(n0 / (1 - rho ** 2), 0),
    }


def p05():
    # (a) 平均応答の信頼区間
    n, xbar, sxx, sd, x0, t18 = 20, 60, 2000, 0.30, 80, 2.101
    k = 1 / n + (x0 - xbar) ** 2 / sxx
    se_m = sd * sqrt(k)
    half = t18 * se_m
    # (b) BIC からモデルの事後確率
    bic1, bic2, e21 = 412.6, 416.8, 8.166
    post = e21 / (1 + e21)
    return {
        "n": n, "xbar": xbar, "sxx": sxx, "sd": s(sd, 2), "x0": x0, "t18": s(t18, 3), "dfe": n - 2,
        "k1": s(1 / n, 2), "k2": s((x0 - xbar) ** 2 / sxx, 2), "k": s(k, 2), "sqk": s(sqrt(k), 2), "se_m": s(se_m, 3),
        "half": s(half, 3), "half4": s(half, 4), "w_pred": s(t18 * sd * sqrt(1 + k), 3), "w_z": s(1.96 * se_m, 3),
        "w_nodev": s(t18 * sd * sqrt(1 / n), 3), "w_lin": s(t18 * sd * sqrt(1 / n + (x0 - xbar) / sxx), 3),
        "bic1": bic1, "bic2": bic2, "dbic": s(bic2 - bic1, 1), "e21": e21, "em21": 0.1225, "e42": 66.69,
        "post": s(post, 3), "w_m2": s(1 - post, 3), "w_no2": s(66.69 / 67.69, 3), "w_ratio": s(bic2 / (bic1 + bic2), 3),
        "w_exp": s(1 - 0.1225, 3),
    }


def p06():
    # (a) ラグ付き目的変数の回帰で、説明変数が上がり続けたときの効果
    b, c = 0.30, 0.60
    eff3 = b * (1 + c + c * c)
    # (b) 独立なランダムウォークどうしの回帰（見せかけの回帰）
    n_, t_, dw = 60, 9.7, 0.18
    r2 = t_ ** 2 / (t_ ** 2 + n_ - 2)
    return {
        "b": s(b, 2), "c": s(c, 2), "e1": s(b, 2), "e2": s(b * (1 + c), 2), "eff3": r(eff3, 3), "eff3_s": s(eff3, 3),
        "c2": s(c * c, 2), "lr": s(b / (1 - c), 2), "w_b3": s(3 * b, 2),
        "n": n_, "t": s(t_, 1), "dw": s(dw, 2), "r2": s(r2, 2), "t2": s(t_ ** 2, 2), "r2den": s(t_ ** 2 + n_ - 2, 2),
    }


def p07():
    # (a) カーネル法：1 次元では分けられないデータを x² を加えて分ける
    bad = [-2.0, -1.6, 1.8, 2.2]
    good = [-0.6, 0.2, 0.9]
    assert max(v * v for v in good) < 1.5 < min(v * v for v in bad)
    kv = (1 + bad[0] * bad[2]) ** 2
    out = {"bad": "、".join(s(v, 1) for v in bad), "good": "、".join(s(v, 1) for v in good),
           "xa": s(bad[0], 1), "xb": s(bad[2], 1), "kv": s(kv, 2), "xab": s(bad[0] * bad[2], 2), "kin": s(1 + bad[0] * bad[2], 2),
           "bad2": "、".join(s(v * v, 2) for v in bad), "good2": "、".join(s(v * v, 2) for v in good)}
    # (b) リッジ推定量の偏りと分散（直交計画）
    beta, sig2 = 1.0, 0.25
    lam_s = sig2 / beta ** 2

    def mse(lm):
        return (sig2 + lm ** 2 * beta ** 2) / (1 + lm) ** 2
    out.update({"beta": s(beta, 1), "sig2": sig2, "lam": s(lam_s, 2), "mse": s(mse(lam_s), 3), "w_ols": s(sig2, 3),
                "w_inv": s(beta ** 2 / sig2, 2), "mse_inv": s(mse(beta ** 2 / sig2), 3),
                "w_var": s(sig2 / (1 + lam_s) ** 2, 3), "w_bias": s(lam_s ** 2 * beta ** 2 / (1 + lam_s) ** 2, 3),
                "d15": s((1 + lam_s) ** 2, 4), "num": s(sig2 + lam_s ** 2 * beta ** 2, 4), "lb2": s(lam_s ** 2 * beta ** 2, 4)})
    # (c) ベイズ決定
    al, be, nn, y = 2, 48, 50, 4
    ap, bp = al + y, be + nn - y
    gain, lossv = 6, 90
    pm = Fraction(ap, ap + bp)

    def prof(th):
        return gain - (gain + lossv) * th
    out.update({
        "al": al, "be": be, "nn": nn, "y": y, "ny": nn - y, "ap": ap, "bp": bp, "apb": ap + bp, "gain": gain, "loss": lossv,
        "gl": gain + lossv, "pm": s(pm, 2), "prof": s(prof(pm), 2), "thstar": s(Fraction(gain, gain + lossv), 4),
        "w_mle": s(prof(Fraction(y, nn)), 2), "w_map": s(prof(Fraction(ap - 1, ap + bp - 2)), 2),
        "w_prior": s(prof(Fraction(al, al + be)), 2), "w_bad": s(prof(Fraction(ap, ap + be + nn)), 2),
        "mle": s(Fraction(y, nn), 2), "prior_m": s(Fraction(al, al + be), 2),
    })
    return out


def p08():
    # (a) ポアソン分布で 0 件の支店の数（二項分布）
    lam, nb, k = 1.5, 10, 2
    e15 = 0.2231
    q0 = e15
    ok = comb(nb, k) * q0 ** k * (1 - q0) ** (nb - k)
    ge = 1 - sum(comb(nb, j) * q0 ** j * (1 - q0) ** (nb - j) for j in range(k))
    mpo = nb * q0
    w_po = math.exp(-mpo) * mpo ** k / math.factorial(k)
    w_p = comb(nb, k) * (lam / nb) ** k * (1 - lam / nb) ** (nb - k)
    vals = [ok, ge, w_po, w_p, q0 ** k]
    assert len({s(v, 3) for v in vals}) == 5, vals
    # (b) クラスカル・ウォリス検定
    g = {"A": [12, 15, 9, 14], "B": [7, 10, 6, 11], "C": [3, 8, 5, 4]}
    allv = sorted(v for vs in g.values() for v in vs)
    assert len(set(allv)) == len(allv)
    rk = {v: i + 1 for i, v in enumerate(allv)}
    R = {k_: sum(rk[v] for v in vs) for k_, vs in g.items()}
    N = len(allv)
    H = 12 / (N * (N + 1)) * sum(Rv ** 2 / 4 for Rv in R.values()) - 3 * (N + 1)
    out = {"lam": s(lam, 1), "nb": nb, "k": k, "e15": s(e15, 4), "q1": s(1 - q0, 4), "c": comb(nb, k),
           "ok": s(ok, 3), "ok4": s(ok, 4), "w_ge": s(ge, 3), "w_po": s(w_po, 3), "w_p": s(w_p, 3), "w_q": s(q0 ** k, 3),
           "q2": s(q0 ** 2, 5), "q8": s((1 - q0) ** 8, 5), "mean0": s(mpo, 3)}
    for k_, vs in g.items():
        out["g" + k_] = "、".join(map(str, vs))
        out["r" + k_] = "、".join(str(rk[v]) for v in vs)
        out["R" + k_] = R[k_]
    out.update({"N": N, "N1": N + 1, "sR2": s(sum(Rv ** 2 for Rv in R.values()) / 4, 0), "H": r(H, 2), "H_s": s(H, 2),
                "H4": s(H, 4), "c95": 5.991})
    return out


def p09():
    x = 16
    rt = sqrt(x)
    h = 1.96 / 2
    lo, hi = (rt - h) ** 2, (rt + h) ** 2

    def iv(a_, b_):
        return f"({s(a_, 2)},\\ {s(b_, 2)})"
    out = {"x": x, "rt": s(rt, 0), "h": s(h, 2), "rlo": s(rt - h, 2), "rhi": s(rt + h, 2),
           "ci_ok": iv(lo, hi), "ci_wald": iv(x - 1.96 * rt, x + 1.96 * rt), "ci_no2": iv((rt - 1.96) ** 2, (rt + 1.96) ** 2),
           "ci_raw": iv(rt - h, rt + h), "ci_se": iv((rt - 1.96 / (2 * rt)) ** 2, (rt + 1.96 / (2 * rt)) ** 2)}
    # (b) ポアソン過程の独立増分
    lam, t1, k, t2 = 0.5, 10, 8, 30
    out.update({"plam": s(lam, 1), "t1": t1, "pk": k, "t2": t2, "dt": t2 - t1, "fut": s(lam * (t2 - t1), 0),
                "E": s(k + lam * (t2 - t1), 0), "V": s(lam * (t2 - t1), 0), "w_E0": s(lam * t2, 0),
                "w_sc": s(k * t2 / t1, 0), "w_short": s(k + lam * t1, 0), "w_short_v": s(lam * t1, 0)})
    return out


def p10():
    # 階層的対数線形モデル（層 L・督促 S・解消 R）。G² は反復比例当てはめで計算
    n = {("lt", "y"): (80, 20), ("lt", "n"): (60, 40), ("ge", "y"): (20, 30), ("ge", "n"): (15, 35)}
    cells = [(l_, s_, j) for l_ in ("lt", "ge") for s_ in ("y", "n") for j in (0, 1)]
    obs = {c_: n[(c_[0], c_[1])][c_[2]] for c_ in cells}

    def ipf(margs):
        m = {c_: 1.0 for c_ in cells}
        for _ in range(500):
            for mg in margs:
                for key in {tuple(c_[i] for i in mg) for c_ in cells}:
                    grp = [c_ for c_ in cells if tuple(c_[i] for i in mg) == key]
                    o, e = sum(obs[c_] for c_ in grp), sum(m[c_] for c_ in grp)
                    for c_ in grp:
                        m[c_] *= o / e
        return 2 * sum(obs[c_] * math.log(obs[c_] / m[c_]) for c_ in cells)
    G_ind = ipf([(0, 1), (2,)])
    G_ci = ipf([(0, 1), (0, 2)])
    G_h = ipf([(0, 1), (0, 2), (1, 2)])
    out = {"G_ind": s(G_ind, 2), "G_ci": s(G_ci, 2), "G_h": s(G_h, 2), "G_d": s(G_ci - G_h, 2),
           "c1": 3.84, "c2": 5.99, "c3": 7.81,
           "or1": s(Fraction(80 * 40, 20 * 60), 2), "or2": s(Fraction(20 * 35, 30 * 15), 2)}
    for (l_, s_), (a_, b_) in n.items():
        out[f"{l_}{s_}a"], out[f"{l_}{s_}b"], out[f"{l_}{s_}t"] = a_, b_, a_ + b_
    return out


def p11():
    # (a) 3 本の投資信託のポートフォリオの標準偏差
    S_ = [[400, 120, -60], [120, 225, 30], [-60, 30, 100]]
    w = [0.5, 0.3, 0.2]
    diag = sum(w[i] ** 2 * S_[i][i] for i in range(3))
    off = sum(w[i] * w[j] * S_[i][j] for i in range(3) for j in range(3) if i < j)
    var = diag + 2 * off
    sds = [sqrt(S_[i][i]) for i in range(3)]
    vals = [sqrt(var), sqrt(diag), sum(w[i] * sds[i] for i in range(3)), sqrt(diag + off),
            sqrt(sum(w[i] * S_[i][i] for i in range(3)) + 2 * off)]
    assert len({s(v, 2) for v in vals}) == 5, vals
    # (b) 貸倒れ確率の異なる債権の貸倒れ件数の正規近似（連続修正つき）
    grp = [(200, 0.01), (300, 0.04), (100, 0.08)]
    mu = sum(a_ * p_ for a_, p_ in grp)
    v = sum(a_ * p_ * (1 - p_) for a_, p_ in grp)
    k = 32
    z = r((k - 0.5 - mu) / sqrt(v), 2)
    phi = {2.08: 0.9812, 2.06: 0.9803, 2.19: 0.9857}
    assert z in phi
    nall = sum(a_ for a_, _ in grp)
    pbar = mu / nall
    vh = nall * pbar * (1 - pbar)
    return {
        **{f"s{i}{j}": S_[i][j] for i in range(3) for j in range(3)}, "w1": w[0], "w2": w[1], "w3": w[2],
        "diag": s(diag, 2), "off": s(off, 2), "off2": s(2 * off, 2), "var": s(var, 2), "sd": s(vals[0], 2),
        "w_diag": s(vals[1], 2), "w_avg": s(vals[2], 2), "w_once": s(vals[3], 2), "w_lin": s(vals[4], 2),
        "c12": s(Fraction(120, 300), 2), "c13": s(-60 / (20 * 10), 2), "c23": s(30 / (15 * 10), 2),
        "n1": grp[0][0], "n2": grp[1][0], "n3": grp[2][0], "p1": grp[0][1], "p2": grp[1][1], "p3": grp[2][1],
        "mu": s(mu, 0), "v1": s(grp[0][0] * grp[0][1] * (1 - grp[0][1]), 2), "v2": s(grp[1][0] * grp[1][1] * (1 - grp[1][1]), 2),
        "v3": s(grp[2][0] * grp[2][1] * (1 - grp[2][1]), 2), "v": s(v, 2), "sdv": s(sqrt(v), 3), "k": k, "km": s(k - 0.5, 1),
        "z": s(z, 2), "z3": s((k - 0.5 - mu) / sqrt(v), 3), "ph": s(phi[z], 4), "ans": r(1 - phi[z], 4), "ans4": s(1 - phi[z], 4),
        "ans_s": s(1 - phi[z], 3), "ph206": s(phi[2.06], 4), "ph219": s(phi[2.19], 4),
        "nall": nall, "pbar": s(pbar, 4), "vh": s(vh, 2), "zh": s((k - 0.5 - mu) / sqrt(vh), 2),
        "znc": s((k - mu) / sqrt(v), 2),
    }


def p12():
    S = [[5.1, 3.1, 1.6], [3.1, 5.1, 1.6], [1.6, 1.6, 2.6]]
    e = [Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)]
    Se = [sum(Fraction(str(S[i][j])) * e[j] for j in range(3)) for i in range(3)]
    lam = Se[0] / e[0]
    assert all(Se[i] == lam * e[i] for i in range(3))
    tr = sum(S[i][i] for i in range(3))
    out = {f"S{i}{j}": S[i][j] for i in range(3) for j in range(3)}
    out.update({"Se0": s(Se[0] * 3, 0), "Se2": s(Se[2] * 3, 0), "lam": s(lam, 0), "tr": s(tr, 1), "ans": r(float(lam) / tr, 3),
                "ans_s": s(float(lam) / tr, 3), "ans4": s(float(lam) / tr, 4), "w_raw": s(81 / tr, 2),
                "w_sum": s(float(lam) / sum(sum(rw) for rw in S), 3), "w_diag": s(5.1 / tr, 3)})
    return out


def p13():
    y = {"A": [9, 13, 11], "B": [7, 10, 7], "C": [12, 15, 15], "D": [8, 10, 9]}
    a, b = 4, 3
    allv = [v for row in y.values() for v in row]
    gm = sum(allv) / len(allv)
    rm = {k: sum(v) / b for k, v in y.items()}
    cm = [sum(y[k][j] for k in y) / a for j in range(b)]
    sst = sum((v - gm) ** 2 for v in allv)
    ssa = b * sum((m - gm) ** 2 for m in rm.values())
    ssb = a * sum((m - gm) ** 2 for m in cm)
    sse = sst - ssa - ssb
    dfa, dfb, dfe = a - 1, b - 1, (a - 1) * (b - 1)
    F = (ssa / dfa) / (sse / dfe)
    F1 = (ssa / dfa) / ((sst - ssa) / (a * b - a))
    out = {"gm": s(gm, 1), "ssa": s(ssa, 1), "ssb": s(ssb, 1), "sst": s(sst, 1), "sse": s(sse, 1), "dfa": dfa, "dfb": dfb,
           "dfe": dfe, "msa": s(ssa / dfa, 1), "mse": s(sse / dfe, 4), "F": r(F, 2), "F_s": s(F, 2), "F1": s(F1, 2),
           "dfe1": a * b - a, "sse1": s(sst - ssa, 1), "fcrit": 4.757, "Fb": s((ssb / dfb) / (sse / dfe), 2),
           "msb": s(ssb / dfb, 1)}
    for k, v in y.items():
        for j, x in enumerate(v):
            out[f"y{k}{j}"] = x
        out[f"m{k}"] = s(rm[k], 1)
        out[f"d{k}"] = s(rm[k] - gm, 1)
    for j, m in enumerate(cm):
        out[f"c{j}"] = s(m, 1)
        out[f"dc{j}"] = s(m - gm, 1)
    return out


def values():
    return {f"s19p{i:02d}": globals()[f"p{i:02d}"]() for i in range(1, 14)}


if __name__ == "__main__":
    import pprint
    pprint.pprint(values(), width=160)
