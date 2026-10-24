"""模擬試験 第2回（content/exam_sets/set02.yaml）の数値を作る。

関数電卓が要る値（分布表・指数関数の値）は問題文に示した丸めた値を使って計算し、
一般電卓で問題文どおりに計算した受験者がちょうど正解にたどり着くようにする。
"""
from decimal import ROUND_HALF_UP, Decimal
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
    w = {"ad": 0.25, "se": 0.55, "di": 0.20}
    cv = {"ad": 0.06, "se": 0.02, "di": 0.03}
    joint = {k: w[k] * cv[k] for k in w}
    p = sum(joint.values())
    post = joint["ad"] / p
    mu, sd = 6000, 3000
    ev = p * sd**2                      # E[Var(R|B)]
    ve = mu**2 * p * (1 - p)            # Var(E[R|B])
    var = ev + ve
    cands = {
        "c_ok": sqrt(var),
        "c_only_ve": sqrt(ve),          # 条件付き分散の期待値を落とす
        "c_only_ev": sqrt(ev),          # 条件付き期待値の分散を落とす
        "c_no_sub": sqrt(p * (sd**2 + mu**2)),   # E[R^2] を分散と取り違え
        "c_prod": sqrt(p * (1 - p) * sd**2),     # Var(B)Var(Y) と思い込む
    }
    v = {
        "jad": s(joint["ad"], 3), "jse": s(joint["se"], 3), "jdi": s(joint["di"], 3),
        "pbuy": s(p, 3), "post": r(post, 3), "post_s": s(post, 3),
        "ER": s(p * mu, 0), "ev": f"{ev:,.0f}", "ve": f"{ve:,.0f}", "var": f"{var:,.0f}",
        "q": s(1 - p, 3), "mu2": f"{mu**2:,.0f}", "ER2": f"{p * (sd**2 + mu**2):,.0f}",
    }
    for k, x in cands.items():
        v[k] = s(x, 0)
    return v


def p02():
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
    v = {"m": s(m, 2), "vA": s(vA, 2), "vB": s(vB, 2), "vC": s(vC, 2), "cov": s(cov, 2)}
    zs = []
    for k, var in variants.items():
        sd = sqrt(var)
        z = r((m - c) / sd, 2)
        zs.append(z)
        v[f"var_{k}"] = s(var, 2)
        v[f"sd_{k}"] = s(sd, 3)
        v[f"z_{k}"] = s(z, 2)
        v[f"P_{k}"] = s(1 - phi_tab(z), 4)
    v["zk"] = s(m - c, 2)
    # 問題文の正規分布表（選択肢に使う z と、その近くの値）
    tab = sorted(set(zs))
    v["tab_z"] = " | ".join(f"{z:.2f}" for z in tab)
    v["tab_p"] = " | ".join(f"{phi_tab(z):.4f}" for z in tab)
    v["tab_sep"] = "|".join(["---"] * len(tab))
    v["Phi_ok"] = s(phi_tab(zs[0]), 4)
    # (b) 分散のカイ二乗検定
    n, s2, s0 = 25, 25.0, 4.0
    v.update({
        "chi_ok": s((n - 1) * s2 / s0**2, 2),
        "chi_n": s(n * s2 / s0**2, 2),
        "chi_sd": s((n - 1) * sqrt(s2) / s0, 2),
        "c24_05": s(chi2.ppf(0.95, 24), 2), "c24_025": s(chi2.ppf(0.975, 24), 2), "c25_05": s(chi2.ppf(0.95, 25), 2),
    })
    return v


def p03():
    th, t1, t2 = 20.0, 12.0, 24.0
    H = lambda t: (t / th) ** 2
    v = {
        "H1": s(H(t1), 2), "H2": s(H(t2), 2), "dH": s(H(t2) - H(t1), 2),
        "h1": s(2 * t1 / th**2, 2), "hc": s(2 * t1 / th**2 * t1, 2),
        "e036": s(e_tab(-0.36), 4), "e072": s(e_tab(-0.72), 4), "e108": s(e_tab(-1.08), 4), "e144": s(e_tab(-1.44), 4),
    }
    v["P_ok"] = s(e_tab(-(H(t2) - H(t1))), 4)
    v["P_S24"] = s(e_tab(-H(t2)), 4)
    v["P_mless"] = s(e_tab(-H(t2 - t1)), 4)          # 無記憶性を誤用（S(12)）
    v["P_const"] = s(e_tab(-2 * t1 / th**2 * t1), 4)  # t=12 のハザードを一定とみなす
    v["P_prod"] = s(e_tab(-H(t2)) * e_tab(-H(t1)), 4)  # 割るべきところを掛ける
    v["th2"] = s(th**2, 0)
    return v


def p04():
    n = 2400
    xa, xb = 192, 240
    pa, pb = xa / n, xb / n
    se = sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    lo = (pb - pa) - 1.96 * se
    v = {"n": n, "xa": xa, "xb": xb, "pa": s(pa, 3), "pb": s(pb, 3), "d": s(pb - pa, 3),
         "va": s(pa * (1 - pa), 4), "vb": s(pb * (1 - pb), 4), "vsum": s(pa * (1 - pa) + pb * (1 - pb), 4),
         "se": s(se, 5), "me": s(1.96 * se, 5), "lo": s(lo, 5), "hi": s(pb - pa + 1.96 * se, 5),
         "lo_pt": r(100 * lo, 2), "hi_pt": s(100 * (pb - pa + 1.96 * se), 2), "se2": f"{se**2:.4e}"}
    # (b) 必要な標本サイズ（問題文の z の値 1.96, 1.645, 0.84 を使う）
    za, za1, zb = 1.96, 1.645, 0.84
    vs = pa * (1 - pa) + pb * (1 - pb)
    dl = pb - pa
    nb = {"ok": ceil((za + zb) ** 2 * vs / dl**2),
          "one": ceil((za1 + zb) ** 2 * vs / dl**2),
          "nopow": ceil(za**2 * vs / dl**2),
          "minus": ceil((za - zb) ** 2 * vs / dl**2)}
    nb["tot"] = 2 * nb["ok"]
    for k, x in nb.items():
        v[f"n_{k}"] = x
    v["zz"] = s((za + zb) ** 2, 4)
    v["nraw"] = s((za + zb) ** 2 * vs / dl**2, 2)
    # (c) ウィルコクソン順位和
    A = [42, 85, 61, 130, 77]
    B = [95, 150, 71, 210, 118, 99]
    allv = sorted(A + B)
    rank = {x: i + 1 for i, x in enumerate(allv)}
    W = sum(rank[x] for x in A)
    n1, n2 = len(A), len(B)
    N = n1 + n2
    EW = n1 * (N + 1) / 2
    VW = n1 * n2 * (N + 1) / 12
    v.update({"A": ", ".join(map(str, A)), "B": ", ".join(map(str, B)), "W": W, "EW": s(EW, 0), "VW": s(VW, 0),
              "sdW": s(sqrt(VW), 3), "Z": r((W - EW) / sqrt(VW), 2), "WA_ranks": " + ".join(str(rank[x]) for x in sorted(A)),
              "order": "、".join(f"{x}（{'A' if x in A else 'B'}）" for x in allv)})
    return v


def p05():
    lam, pc, mins, hours = 24, 0.25, 15, 50
    lc = lam * pc
    m15 = lc * mins / 60
    v = {"lc": s(lc, 0), "m15": s(m15, 1), "mall": s(lam * mins / 60, 1), "mnon": s(lam * (1 - pc) * mins / 60, 1),
         "e15": s(e_tab(-m15), 4), "e45": s(e_tab(-lam * (1 - pc) * mins / 60), 4), "e60": s(e_tab(-lam * mins / 60), 4)}
    v["Pa_ok"] = v["e15"]
    v["Pa_all"] = v["e60"]
    v["Pa_non"] = v["e45"]
    v["Pa_bin"] = s((1 - pc) ** (lam * mins / 60), 4)   # 期待件数 6 件がすべてクーポンなし、と考える
    v["Pa_comp"] = s(1 - e_tab(-m15), 4)
    # (b) 50時間の件数の正規近似
    mu = lc * hours
    k = 330
    sd = sqrt(mu)
    zs = {"ok": r((k - 0.5 - mu) / sd, 2), "nocc": r((k - mu) / sd, 2), "wrong": r((k + 0.5 - mu) / sd, 2),
          "bin": r((k - 0.5 - mu) / sqrt(lam * hours * pc * (1 - pc)), 2)}
    tab = sorted(set(zs.values()))
    v.update({"mu": s(mu, 0), "sd": s(sd, 4), "k": k, "nall": lam * hours, "sdbin": s(sqrt(lam * hours * pc * (1 - pc)), 0),
              "tab_z": " | ".join(f"{z:.2f}" for z in tab), "tab_p": " | ".join(f"{phi_tab(z):.4f}" for z in tab),
              "tab_sep": "|".join(["---"] * len(tab))})
    for kk, z in zs.items():
        v[f"z_{kk}"] = s(z, 2)
        v[f"Pb_{kk}"] = s(1 - phi_tab(z), 4)
    v["Pb_upper"] = s(phi_tab(zs["ok"]), 4)  # 補集合を答える
    return v


def p06():
    data = [(2, 1), (3, 0), (4, 1), (5, 1), (6, 0), (7, 1), (8, 0), (9, 1), (10, 0), (12, 0)]
    S = 1.0
    steps = []
    at = len(data)
    for t, d in data:
        if d:
            S *= (at - 1) / at
            steps.append((t, at))
        at -= 1
        if t >= 8:
            break
    v = {"S8": r(S, 3), "S8_s": s(S, 3)}
    v["km_steps"] = r" \times ".join(f"\\frac{{{a - 1}}}{{{a}}}" for _, a in steps)
    v["km_risk"] = "、".join(f"$t={t}$ で $n={a}$" for t, a in steps)
    # (b) 指数モデル
    d = sum(x[1] for x in data)
    T = sum(x[0] for x in data)
    n = len(data)
    Tev = sum(x[0] for x in data if x[1])
    lam = d / T
    v.update({"d": d, "T": T, "n": n, "Tev": Tev, "lam": s(lam, 4), "lam_s": s(lam, 5),
              "se_ok": s(lam / sqrt(d), 4), "se_n": s(lam / sqrt(n), 4),
              "lam_ev": s(d / Tev, 4), "se_ev": s(d / Tev / sqrt(d), 4),
              "lam_all": s(n / T, 4), "se_all": s(n / T / sqrt(n), 4),
              "var_ok": s(lam**2 / d, 4), "info": s(d / lam**2, 1)})
    return v


def p07():
    n, ST, SE, p = 40, 250.0, 70.0, 3
    R2 = 1 - SE / ST
    adj = 1 - (SE / (n - p)) / (ST / (n - 1))
    v = {"n": n, "ST": s(ST, 0), "SE": s(SE, 0), "R2": s(R2, 3), "adj": r(adj, 3), "adj_s": s(adj, 3),
         "dfE": n - p, "dfT": n - 1, "VE": s(SE / (n - p), 4), "VT": s(ST / (n - 1), 4), "ratio": s((SE / (n - p)) / (ST / (n - 1)), 4)}
    # (b) クックの距離
    h, e = 0.30, -3.2
    s2 = SE / (n - p)
    rr = e / sqrt(s2 * (1 - h))
    D = rr**2 * h / (p * (1 - h))
    s2n = SE / n
    cand = {
        "ok": D,
        "p2": rr**2 * h / ((p - 1) * (1 - h)),                 # p に切片を数えない
        "noh": (e**2 / s2) * h / (p * (1 - h)),                # 標準化残差で (1-h) を忘れる
        "s2n": (e**2 / (s2n * (1 - h))) * h / (p * (1 - h)),   # σ^2 を S_E/n で推定
        "nolev": rr**2 / p,                                    # h/(1-h) の因子を忘れる
    }
    v.update({"h": s(h, 2), "e": s(e, 1), "s2": s(s2, 4), "rr": s(rr, 3), "rr2": s(rr**2, 3), "hh": s(h / (1 - h), 4)})
    for k, x in cand.items():
        v[f"D_{k}"] = s(x, 3)
    # (c) 欠落変数
    b1, b2, d = 0.85, 1.40, 0.30
    v.update({"b1": s(b1, 2), "b2": s(b2, 2), "dd": s(d, 2), "ovb_ok": s(b1 + b2 * d, 2), "ovb_minus": s(b1 - b2 * d, 2),
              "ovb_only": s(b2 * d, 2), "ovb_inv": s(b1 + b2 / d, 2), "ovb_add": s(b1 + b2, 2), "bias": s(b2 * d, 2)})
    return v


def p08():
    b0, bc, bl = -2.2, 0.6, 0.8
    ev = {0.8: e_tab(0.8), 1.4: e_tab(1.4), 1.6: e_tab(1.6), 2.2: e_tab(2.2), 0.6: e_tab(0.6)}

    def prob(eta):  # eta < 0 のとき 1/(1+e^{-eta}) を表の値で
        return 1 / (1 + ev[round(-eta, 1)])
    p11, p10, p01, p00 = prob(b0 + bc + bl), prob(b0 + bl), prob(b0 + bc), prob(b0)
    v = {f"e{str(k).replace('.', '')}": s(x, 4) for k, x in ev.items()}
    v.update({"eta11": s(b0 + bc + bl, 1), "p11": s(p11, 4), "p10": s(p10, 4), "p01": s(p01, 4), "p00": s(p00, 4),
              "P_ok": s(p11, 3), "P_odds": s(1 / ev[0.8], 3), "P_sign": s(1 - p11, 3), "P_nol": s(p01, 3),
              "P_rr": s(p10 * ev[0.6], 3)})
    wl = 0.3
    ace = wl * (p11 - p10) + (1 - wl) * (p01 - p00)
    v.update({"d1": s(p11 - p10, 4), "d0": s(p01 - p00, 4), "ace": r(100 * ace, 2), "ace_s": s(100 * ace, 2),
              "ace_frac": s(ace, 4), "t1": s(wl * (p11 - p10), 4), "t0": s((1 - wl) * (p01 - p00), 4)})
    return v


def p09():
    a, b, rep = 3, 2, 4
    N = a * b * rep
    SA, SB, SAB, ST = 48.0, 30.0, 16.0, 166.0
    SE = ST - SA - SB - SAB
    dfA, dfB, dfAB, dfE = a - 1, b - 1, (a - 1) * (b - 1), a * b * (rep - 1)
    MSE = SE / dfE
    v = {"N": N, "SE": s(SE, 0), "dfE": dfE, "dfAB": dfAB, "MSAB": s(SAB / dfAB, 1), "MSE": s(MSE, 1),
         "F_ok": s((SAB / dfAB) / MSE, 2), "F_ss": s(SAB / SE, 2), "F_df3": s((SAB / (dfA + dfB)) / MSE, 2),
         "F_dfE": s((SAB / dfAB) / (SE / (N - 1)), 2), "F_df1": s(SAB / MSE, 2),
         "F218": s(fdist.ppf(0.95, 2, 18), 2), "MSA": s(SA / dfA, 1), "MSB": s(SB / dfB, 1),
         "FA": s(SA / dfA / MSE, 2), "FB": s(SB / dfB / MSE, 2), "F118": s(fdist.ppf(0.95, 1, 18), 2)}
    return v


def p10():
    Nh = [60000, 30000, 10000]
    Sh = [1.2, 2.0, 3.0]
    n = 800
    NS = [a * b for a, b in zip(Nh, Sh)]
    NS2 = [a * b * b for a, b in zip(Nh, Sh)]
    v = {"NS1": f"{NS[0]:,.0f}", "NS2": f"{NS[1]:,.0f}", "NS3": f"{NS[2]:,.0f}", "NSsum": f"{sum(NS):,.0f}",
         "n_ok": s(n * NS[2] / sum(NS), 0), "n_ok_raw": s(n * NS[2] / sum(NS), 2),
         "n_prop": s(n * Nh[2] / sum(Nh), 0), "n_var": s(n * NS2[2] / sum(NS2), 0),
         "n_eq": s(n / 3, 0), "n_s": s(n * Sh[2] / sum(Sh), 0),
         "n1": s(n * NS[0] / sum(NS), 0), "n2": s(n * NS[1] / sum(NS), 0)}
    return v


def p11():
    ev = [2.35, 1.20, 0.75, 0.45, 0.25]
    p = len(ev)
    cum2 = (ev[0] + ev[1]) / p
    v = {"cum1": s(100 * ev[0] / p, 1), "cum2": s(100 * cum2, 1), "cum3": s(100 * sum(ev[:3]) / p, 1),
         "sum12": s(ev[0] + ev[1], 2)}
    l1, l2 = 0.72, 0.35
    h2 = l1**2 + l2**2
    v.update({"l1sq": s(l1**2, 4), "l2sq": s(l2**2, 4), "h2": s(h2, 4), "u": r(1 - h2, 3), "u_s": s(1 - h2, 3)})
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
    # θ/(1+θ^2)=ρ → ρθ^2 - θ + ρ = 0
    disc = 1 - 4 * rho1**2
    th_small = (1 - sqrt(disc)) / (2 * rho1)
    th_big = (1 + sqrt(disc)) / (2 * rho1)
    v = {"disc": s(disc, 2), "sq": s(sqrt(disc), 1), "th": s(th_small, 2), "thb": s(th_big, 2),
         "s2_ok": s(g0 / (1 + th_small**2), 2), "s2_big": s(g0 / (1 + th_big**2), 2),
         "th_ar": s(rho1, 2), "s2_ar": s(g0 / (1 + rho1**2), 2), "s2_raw": s(g0, 2), "s2_mul": s(g0 * (1 + th_small**2), 2),
         "one_th2": s(1 + th_small**2, 2), "rho2ar": s(rho1**2, 2)}
    return v


def p13():
    # ベータ(2,3): f(x)=12x(1-x)^2、最大は x=1/3 で 16/9
    M = 12 * (1 / 3) * (2 / 3) ** 2
    n = 1000
    v = {"M_s": s(M, 4), "acc": s(1 / M, 4),
         "N_ok": s(n * M, 0), "N_half": s(n * 12 * 0.5 * 0.25, 0), "N_inv": s(n / M, 0),
         "N_mean": s(n * 12 * 0.4 * 0.6**2, 0), "N_crude": s(n * 12, 0)}
    return v


def values():
    return {"s02p01": p01(), "s02p02": p02(), "s02p03": p03(), "s02p04": p04(), "s02p05": p05(), "s02p06": p06(),
            "s02p07": p07(), "s02p08": p08(), "s02p09": p09(), "s02p10": p10(), "s02p11": p11(), "s02p12": p12(),
            "s02p13": p13()}


if __name__ == "__main__":
    for k, d in values().items():
        print(k)
        for kk, x in d.items():
            print("  ", kk, "=", x)
