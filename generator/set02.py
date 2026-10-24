"""模擬試験 第2回（content/exam_sets/set02.yaml）の数値を作る。

関数電卓が要る値（分布表・指数関数の値）は問題文に示した丸めた値を使って計算し、
一般電卓で問題文どおりに計算した受験者がちょうど正解にたどり着くようにする。
"""
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from math import ceil, exp, sqrt

from scipy.stats import chi2, f as fdist, norm


def r(x, nd):
    """四捨五入（偶数丸めではない）"""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def s(x, nd):
    """四捨五入した文字列（末尾の0を残す）"""
    return f"{r(x, nd):.{nd}f}"


def phi_tab(z):
    """問題文に載せる標準正規分布表の値 Φ(z)（小数第4位）"""
    return r(norm.cdf(z), 4)


def e_tab(x):
    """問題文に載せる指数関数の値（小数第4位）"""
    return r(exp(x), 4)


def p01():
    """不正検知アラートの陽性的中率と必要な特異度"""
    pi, sens, spec = 0.002, 0.95, 0.99
    tp = pi * sens
    fp = (1 - pi) * (1 - spec)
    fpr_max = tp / (1 - pi)          # PPV=0.5 となる偽陽性率
    return {"tp": s(tp, 4), "fp": s(fp, 5), "ppv": s(100 * tp / (tp + fp), 1),
            "fpr_max": s(fpr_max, 6), "ans": r(100 * (1 - fpr_max), 2), "ans_s": s(100 * (1 - fpr_max), 2),
            "ans4": s(100 * (1 - fpr_max), 4), "w_nopi": s(100 * (1 - tp), 2)}


def p02():
    v = {}
    # (a) 幾何平均
    R = [0.20, -0.10, 0.15, -0.05]
    prod = 1.0
    for x in R:
        prod *= 1 + x
    g = prod ** 0.25
    harm = len(R) / sum(1 / (1 + x) for x in R)
    v.update({"prod": s(prod, 4), "sq1": s(sqrt(prod), 4), "g": s(g, 4),
              "G_ok": s(100 * (g - 1), 2), "G_ar": s(100 * sum(R) / len(R), 2), "G_harm": s(100 * (harm - 1), 2),
              "G_lin": s(100 * (prod - 1) / len(R), 2), "G_tot": s(100 * (prod - 1), 2)})
    # (b) ポートフォリオ
    mA, sA, mB, sB, rho, wA = 1.2, 5.0, 0.6, 3.0, -0.3, 0.6
    wB = 1 - wA
    m = wA * mA + wB * mB
    cov = rho * sA * sB
    vA, vB, vC = wA**2 * sA**2, wB**2 * sB**2, 2 * wA * wB * cov
    c = -4.0
    variants = {
        "ok": vA + vB + vC,
        "nocov": vA + vB,               # 共分散を無視
        "sign": vA + vB - vC,           # 共分散の符号を逆に
        "half": vA + vB + vC / 2,       # 2 を掛け忘れる
        "lin": (wA * sA + wB * sB) ** 2,  # 標準偏差を加重平均（完全相関の場合）
    }
    v.update({"m": s(m, 2), "vA": s(vA, 2), "vB": s(vB, 2), "vC": s(vC, 2), "cov": s(cov, 2)})
    zs = []
    for k, var in variants.items():
        sd = sqrt(var)
        z = r((m - c) / sd, 2)
        zs.append(z)
        v[f"var_{k}"] = s(var, 2)
        v[f"sd_{k}"] = s(sd, 3)
        v[f"z_{k}"] = s(z, 2)
        v[f"P_{k}"] = s(1 - phi_tab(z), 4)
    tab = sorted(set(zs))
    v["tab_z"] = " | ".join(f"{z:.2f}" for z in tab)
    v["tab_p"] = " | ".join(f"{phi_tab(z):.4f}" for z in tab)
    v["tab_sep"] = "|".join(["---"] * len(tab))
    v["Phi_ok"] = s(phi_tab(zs[0]), 4)
    # (c) 分散のカイ二乗検定
    n, s2, s0 = 25, 25.0, 4.0
    v.update({
        "chi_ok": s((n - 1) * s2 / s0**2, 2),
        "chi_n": s(n * s2 / s0**2, 2),
        "chi_sd": s((n - 1) * sqrt(s2) / s0, 2),
        "c24_05": s(chi2.ppf(0.95, 24), 2), "c24_025": s(chi2.ppf(0.975, 24), 2), "c25_05": s(chi2.ppf(0.95, 25), 2),
    })
    return v


def p03():
    """(a) 対数正規の中央値と平均、(b) 0 に点確率をもつ損失の分布"""
    S0, mu, sg = 100, 0.05, 0.20
    e05, e07, e09 = e_tab(mu), e_tab(mu + sg**2 / 2), e_tab(mu + sg**2)
    v = {"e05": s(e05, 4), "e07": s(e07, 4), "e09": s(e09, 4),
         "med": s(S0 * e05, 2), "mean": s(S0 * e07, 2), "mean_full": s(S0 * e09, 2), "half": s(sg**2 / 2, 2), "mh": s(mu + sg**2 / 2, 2)}
    q, th = 0.4, 5.0
    v.update({"p0": s(1 - q, 1), "q": s(q, 1), "EL": s(q * th, 1), "th": s(th, 1), "EL_w": s((1 - q) * th, 1)})
    return v


def p04():
    n = 2400
    xa, xb = 192, 240
    pa, pb = xa / n, xb / n
    se = sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    lo = (pb - pa) - 1.96 * se
    v = {"n": n, "xa": xa, "xb": xb, "pa": s(pa, 3), "pb": s(pb, 3), "d": s(pb - pa, 3),
         "vsum": s(pa * (1 - pa) + pb * (1 - pb), 4),
         "se": s(se, 5), "me": s(1.96 * se, 5), "lo": s(lo, 5),
         "lo_pt": r(100 * lo, 2), "hi_pt": s(100 * (pb - pa + 1.96 * se), 2)}
    # (b) 検出力（真の値で標準誤差を計算、反対側の棄却域は無視）
    zc = 1.96
    zs = {"ok": r((pb - pa) / se - zc, 2), "one": r((pb - pa) / se - 1.645, 2), "noz": r((pb - pa) / se, 2),
          "onegrp": r((pb - pa) / sqrt(pa * (1 - pa) / n) - zc, 2)}
    tab = sorted(set(zs.values()))
    v.update({"ratio": s((pb - pa) / se, 4), "tab_z": " | ".join(f"{z:.2f}" for z in tab),
              "tab_p": " | ".join(f"{phi_tab(z):.4f}" for z in tab), "tab_sep": "|".join(["---"] * len(tab)),
              "se1": s(sqrt(pa * (1 - pa) / n), 5)})
    for k, z in zs.items():
        v[f"zp_{k}"] = s(z, 2)
        v[f"pw_{k}"] = s(phi_tab(z), 4)
    v["pw_beta"] = s(1 - phi_tab(zs["ok"]), 4)
    # (c) ケンドールの順位相関
    cvr = [2, 1, 3, 6, 4, 5]          # クリック率の順位 1〜6 のページの購入率の順位
    nn = len(cvr)
    C = sum(1 for i, j in combinations(range(nn), 2) if cvr[i] < cvr[j])
    D = sum(1 for i, j in combinations(range(nn), 2) if cvr[i] > cvr[j])
    var_s = nn * (nn - 1) * (2 * nn + 5) / 18
    v.update({"cvr": " | ".join(map(str, cvr)), "C": C, "D": D, "S": C - D, "tau": s((C - D) / (nn * (nn - 1) / 2), 3),
              "varS": s(var_s, 2), "sdS": s(sqrt(var_s), 4), "Z": r((C - D) / sqrt(var_s), 2), "Z_s": s((C - D) / sqrt(var_s), 2)})
    return v


def p05():
    lam, pc, mins, hours = 24, 0.25, 15, 50
    lc = lam * pc
    m15 = lc * mins / 60
    v = {"lc": s(lc, 0), "m15": s(m15, 1),
         "e15": s(e_tab(-m15), 4), "e45": s(e_tab(-lam * (1 - pc) * mins / 60), 4), "e60": s(e_tab(-lam * mins / 60), 4)}
    v["Pa_ok"] = v["e15"]
    v["Pa_all"] = v["e60"]
    v["Pa_non"] = v["e45"]
    v["Pa_bin"] = s((1 - pc) ** (lam * mins / 60), 4)   # 期待件数 6 件がすべてクーポンなし、と考える
    v["Pa_comp"] = s(1 - e_tab(-m15), 4)
    # (b) 二項分布のポアソン近似
    nn, pp = 2000, 0.001
    m = nn * pp
    e2 = e_tab(-m)
    v.update({"m": s(m, 0), "e2": s(e2, 4),
              "Pb_ok": s(1 - e2 * (1 + m + m**2 / 2), 4), "Pb_ge2": s(1 - e2 * (1 + m), 4),
              "Pb_eq3": s(e2 * m**3 / 6, 4), "Pb_gt3": s(1 - e2 * (1 + m + m**2 / 2 + m**3 / 6), 4),
              "Pb_le2": s(e2 * (1 + m + m**2 / 2), 4), "sum3": s(1 + m + m**2 / 2, 0)})
    return v


def p06():
    # (a) 集落抽出の設計効果
    mm, k, icc = 20, 40, 0.05
    n = mm * k
    deff = 1 + (mm - 1) * icc
    v = {"n": n, "deff": s(deff, 2), "neff": s(n / deff, 0), "neff_raw": s(n / deff, 1),
         "neff_m": s(n / (1 + mm * icc), 0), "neff_1": s(n / (1 + icc), 0), "neff_mul": s(n * deff, 0)}
    # (b) MAR の欠測と完全ケース分析
    W = [0.6, 0.3, 0.1]
    rr = [0.5, 0.7, 0.9]
    yb = [6.8, 7.4, 8.1]
    num = sum(w * q * y for w, q, y in zip(W, rr, yb))
    den = sum(w * q for w, q in zip(W, rr))
    cc = num / den
    adj = sum(w * y for w, y in zip(W, yb))
    v.update({"num": s(num, 3), "den": s(den, 2), "cc": s(cc, 3), "adj": s(adj, 2),
              "bias": r(cc - adj, 3), "bias_s": s(cc - adj, 3),
              "wr": "、".join(s(w * q, 2) for w, q in zip(W, rr))})
    return v


def p07():
    data = [(2, 1), (3, 0), (4, 1), (5, 1), (6, 0), (7, 1), (8, 0), (9, 1), (10, 0), (12, 0)]
    d = sum(x[1] for x in data)
    T = sum(x[0] for x in data)
    n = len(data)
    Tev = sum(x[0] for x in data if x[1])
    lam = d / T
    return {"d": d, "T": T, "n": n, "Tev": Tev, "lam": s(lam, 4), "lam_s": s(lam, 5),
            "se_ok": s(lam / sqrt(d), 4), "se_n": s(lam / sqrt(n), 4),
            "lam_ev": s(d / Tev, 4), "se_ev": s(d / Tev / sqrt(d), 4),
            "lam_all": s(n / T, 4), "se_all": s(n / T / sqrt(n), 4),
            "var_ok": s(lam**2 / d, 4)}


def p08():
    SE1 = 70.0
    b1, b2, sy, s1, s2 = 0.85, 0.12, 3.0, 2.4, 6.5
    v = {"SE1": s(SE1, 0), "b1": s(b1, 2), "b2": s(b2, 2), "sy": s(sy, 1), "s1": s(s1, 1), "s2": s(s2, 1),
         "std1": r(b1 * s1 / sy, 2), "std1_s": s(b1 * s1 / sy, 2), "std2": s(b2 * s2 / sy, 2),
         "std1_inv": s(b1 * sy / s1, 2), "std1_var": s(b1 * s1**2 / sy**2, 2)}
    # (b) ダービン・ワトソン
    sdiff = 58.8
    dw = sdiff / SE1
    v.update({"sdiff": s(sdiff, 1), "dw": s(dw, 2), "rho": s(1 - dw / 2, 2), "dw_inv": s(SE1 / sdiff, 2),
              "rho_inv": s(1 - (SE1 / sdiff) / 2, 2), "rho_half": s(dw / 2, 2), "dw4": s(4 - dw, 2),
              "dL": "1.39", "dU": "1.60"})
    # (c) コクラン・オーカット法
    rho_hat, a_star, b_star = r(1 - dw / 2, 2), 0.84, 0.80
    v.update({"astar": s(a_star, 2), "bstar": s(b_star, 2), "omr": s(1 - rho_hat, 2),
              "b0_ok": s(a_star / (1 - rho_hat), 2), "b0_raw": s(a_star, 2), "b0_mul": s(a_star * (1 - rho_hat), 2),
              "b0_plus": s(a_star / (1 + rho_hat), 2), "b1_div": s(b_star / (1 - rho_hat), 2)})
    return v


def p09():
    b0, bc, bl = -2.2, 0.6, 0.8
    ev = {0.8: e_tab(0.8), 1.4: e_tab(1.4), 1.6: e_tab(1.6), 2.2: e_tab(2.2), 0.6: e_tab(0.6)}

    def prob(eta):  # eta < 0 のとき 1/(1+e^{-eta}) を表の値で
        return 1 / (1 + ev[round(-eta, 1)])
    p11, p10, p01, p00 = prob(b0 + bc + bl), prob(b0 + bl), prob(b0 + bc), prob(b0)
    v = {f"e{str(k).replace('.', '')}": s(x, 4) for k, x in ev.items()}
    v.update({"p11": s(p11, 4), "p10": s(p10, 4), "p01": s(p01, 4), "p00": s(p00, 4),
              "d1": s(p11 - p10, 4), "d0": s(p01 - p00, 4), "rr1": s(p11 / p10, 2), "rr0": s(p01 / p00, 2),
              "or1": s((p11 / (1 - p11)) / (p10 / (1 - p10)), 2), "or0": s((p01 / (1 - p01)) / (p00 / (1 - p00)), 2)})
    wl = 0.3
    ace = wl * (p11 - p10) + (1 - wl) * (p01 - p00)
    d1r, d0r = r(p11, 4) - r(p10, 4), r(p01, 4) - r(p00, 4)   # 解説に載せる丸めた確率から
    v.update({"ace": r(100 * ace, 1), "ace_frac": s(wl * d1r + (1 - wl) * d0r, 5),
              "t1": s(wl * d1r, 5), "t0": s((1 - wl) * d0r, 5)})
    # 単純比較（配布率：ロイヤル 80%、一般 20%）
    q1, q0 = 0.8, 0.2
    pc1 = wl * q1 + (1 - wl) * q0
    py1 = (wl * q1 * p11 + (1 - wl) * q0 * p01) / pc1
    py0 = (wl * (1 - q1) * p10 + (1 - wl) * (1 - q0) * p00) / (1 - pc1)
    v["naive"] = s(100 * (py1 - py0), 1)
    return v


def p10():
    a, b, rep = 3, 2, 4
    N = a * b * rep
    SA, SB, SAB, ST = 48.0, 30.0, 16.0, 166.0
    SE = ST - SA - SB - SAB
    dfA, dfB, dfAB, dfE = a - 1, b - 1, (a - 1) * (b - 1), a * b * (rep - 1)
    MSE = SE / dfE
    return {"N": N, "SE": s(SE, 0), "dfE": dfE, "dfAB": dfAB, "MSAB": s(SAB / dfAB, 1), "MSE": s(MSE, 1),
            "F_ok": s((SAB / dfAB) / MSE, 2), "F_ss": s(SAB / SE, 2), "F_df3": s((SAB / (dfA + dfB)) / MSE, 2),
            "F_dfE": s((SAB / dfAB) / (SE / (N - 1)), 2), "F_df1": s(SAB / MSE, 2),
            "F218": s(fdist.ppf(0.95, 2, 18), 2), "FA": s(SA / dfA / MSE, 2), "FB": s(SB / dfB / MSE, 2)}


def p11():
    lam1, a = 2.35, 0.52
    v = {"sq": s(sqrt(lam1), 4), "L_ok": s(a * sqrt(lam1), 3), "L_vec": s(a, 3), "L_div": s(a / sqrt(lam1), 3),
         "L_sq": s(a * a * lam1, 3), "L_lam": s(a * lam1, 3)}
    # (b) 精度行列
    w11, w13, w33 = 1.6, 0.4, 1.5
    v.update({"pc": r(-w13 / sqrt(w11 * w33), 3), "pc_s": s(-w13 / sqrt(w11 * w33), 3), "sq13": s(sqrt(w11 * w33), 4)})
    # (c) 対応分析
    tab = [[60, 30, 30], [20, 30, 30]]
    n = sum(map(sum, tab))
    rs = [sum(rw) for rw in tab]
    cs = [sum(c) for c in zip(*tab)]
    chi = 0.0
    terms = []
    for i in range(2):
        for j in range(3):
            E = rs[i] * cs[j] / n
            chi += (tab[i][j] - E) ** 2 / E
            terms.append(s((tab[i][j] - E) ** 2 / E, 1))
    v.update({"chi": s(chi, 1), "inert": s(chi / n, 4), "phi": s(sqrt(chi / n), 2), "nn": n,
              "chi_terms": " + ".join(terms),
              "E": "、".join(s(rs[i] * cs[j] / n, 0) for i in range(2) for j in range(3))})
    return v


def p12():
    rho1, g0 = 0.40, 2.5
    disc = 1 - 4 * rho1**2
    th_small = (1 - sqrt(disc)) / (2 * rho1)
    th_big = (1 + sqrt(disc)) / (2 * rho1)
    return {"th": s(th_small, 2), "thb": s(th_big, 2),
            "s2_ok": s(g0 / (1 + th_small**2), 2), "s2_big": s(g0 / (1 + th_big**2), 2),
            "th_ar": s(rho1, 2), "s2_ar": s(g0 / (1 + rho1**2), 2), "s2_raw": s(g0, 2), "s2_mul": s(g0 * (1 + th_small**2), 2),
            "one_th2": s(1 + th_small**2, 2), "rho2ar": s(rho1**2, 2)}


def p13():
    # ベータ(2,3): f(x)=12x(1-x)^2、最大は x=1/3 で 16/9
    M = 12 * (1 / 3) * (2 / 3) ** 2
    n = 1000
    return {"M_s": s(M, 4), "acc": s(1 / M, 4),
            "N_ok": s(n * M, 0), "N_half": s(n * 12 * 0.5 * 0.25, 0), "N_inv": s(n / M, 0),
            "N_mean": s(n * 12 * 0.4 * 0.6**2, 0), "N_crude": s(n * 12, 0)}


def values():
    fs = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13]
    return {f"s02p{i + 1:02d}": f() for i, f in enumerate(fs)}


if __name__ == "__main__":
    for k, d in values().items():
        print(k)
        for kk, x in d.items():
            print("  ", kk, "=", x)
