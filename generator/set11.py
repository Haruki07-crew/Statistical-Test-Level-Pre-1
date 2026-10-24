"""模擬試験 第11回（set11）の数値。values() が {大問id: {名前: 値}} を返す。

題材：教育・心理測定・人事・労働。
問題文に示した値（Φ の値、t・χ² の点、e^x・対数の値、丸めたてこ比・残差など）は、
ここでも同じ丸めた値を使って答えを計算する（一般電卓で示した値だけから正解に届くようにするため）。
"""
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


# 問題文に示す標準正規分布の分布関数の値
PHI = {0.00: 0.5000, 0.10: 0.5398, 0.90: 0.8159, 1.00: 0.8413}


def p01():
    """古典的テスト理論：平行テストの合計点の信頼性と、観測得点が与えられたときの真の得点"""
    vT, vE, mu = 80, 20, 60
    vX = vT + vE
    rho = vT / vX
    # (a) 合計点 S = X1 + X2
    varS_true = 4 * vT
    varS = 4 * vT + 2 * vE
    relS = varS_true / varS
    # (b) T | X = x ~ N(mu + rho (x - mu), vT (1 - rho))
    x, cut = 72, 70
    m = mu + rho * (x - mu)
    cv = vT * (1 - rho)
    sd = sqrt(cv)
    z = (cut - m) / sd
    assert abs(z - 0.10) < 1e-12
    p_below = PHI[0.10]
    return {
        "vT": vT, "vE": vE, "mu": mu, "vX": vX, "rho": s(rho, 1),
        "v4T": varS_true, "v2E": 2 * vE, "vS": varS, "cov12": vT,
        "relS": s(relS, 3), "relS4": s(relS, 4),
        "r_single": s(rho, 3), "r_noerr": s(4 * vT / (4 * vT + vE), 3), "r_half": s(4 * vT / (4 * vT + vE / 2), 3),
        "r_sq": s(rho ** 2, 3), "sb": s(2 * rho / (1 + rho), 4),
        "x": x, "cut": cut, "dev": x - mu, "shr": s(rho * (x - mu), 1), "m": s(m, 1), "cv": s(cv, 0), "sd": s(sd, 0),
        "z": s(z, 2), "phi10": s(PHI[0.10], 4), "ans": r(p_below, 3), "ans_s": s(p_below, 3),
        "w_up": s(1 - PHI[0.10], 3),
    }


def p02():
    """採用選考：面接官3人の多数決、適性検査のAUC、試用期間の本採用率のベイズ予測"""
    prior, sens, fp = 0.3, 0.8, 0.25
    maj = lambda q: 3 * q ** 2 * (1 - q) + q ** 3
    pg, pb = maj(sens), maj(fp)
    num = prior * pg
    den = num + (1 - prior) * pb
    post = num / den
    one = prior * sens / (prior * sens + (1 - prior) * fp)
    allx = prior * sens ** 3 / (prior * sens ** 3 + (1 - prior) * fp ** 3)
    # (b) AUC
    stay = [72, 65, 80, 58, 69]
    leave = [55, 61, 65, 50, 70]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in stay for b in leave)
    ties = sum(1 for a in stay for b in leave if a == b)
    auc = wins / 25
    per = [sum(1.0 if a > b else 0.5 if a == b else 0.0 for b in leave) for a in stay]
    # (c) ベータ・二項
    a0, b0, nn, k = 6, 2, 10, 7
    a1, b1 = a0 + k, b0 + nn - k
    pred2 = a1 * (a1 + 1) / ((a1 + b1) * (a1 + b1 + 1))
    pm = a1 / (a1 + b1)
    return {
        "prior": prior, "sens": sens, "fp": fp, "q1": s(1 - prior, 1), "ms": s(1 - sens, 1), "mf": s(1 - fp, 2),
        "pg": s(pg, 3), "pb": s(pb, 5), "num": s(num, 4), "bterm": s((1 - prior) * pb, 6), "den": s(den, 6),
        "pg2": s(3 * sens ** 2 * (1 - sens), 3), "pg3": s(sens ** 3, 3),
        "pb2": s(3 * fp ** 2 * (1 - fp), 6), "pb3": s(fp ** 3, 6),
        "post": r(post, 3), "post_s": s(post, 3), "post4": s(post, 4),
        "w_one": s(one, 3), "w_all": s(allx, 3), "w_prior": s(prior, 3),
        "st1": stay[0], "st2": stay[1], "st3": stay[2], "st4": stay[3], "st5": stay[4],
        "lv1": leave[0], "lv2": leave[1], "lv3": leave[2], "lv4": leave[3], "lv5": leave[4],
        "w1": s(per[0], 1), "w2": s(per[1], 1), "w3": s(per[2], 1), "w4": s(per[3], 1), "w5": s(per[4], 1),
        "wins": s(wins, 1), "ties": ties,
        "auc": s(auc, 2), "auc_notie": s((wins - 0.5 * ties) / 25, 2), "auc_tie1": s((wins + 0.5 * ties) / 25, 2),
        "auc_inv": s(1 - auc, 2), "auc_45": s(wins / 45, 3),
        "a0": a0, "b0": b0, "nn": nn, "k": k, "fail": nn - k, "a1": a1, "b1": b1, "ab1": a1 + b1,
        "a1p": a1 + 1, "ab1p": a1 + b1 + 1, "numc": a1 * (a1 + 1), "denc": (a1 + b1) * (a1 + b1 + 1),
        "pred2": s(pred2, 3), "pred2_4": s(pred2, 4), "pm": s(pm, 3), "pm2": s(pm ** 2, 3),
        "mle2": s((k / nn) ** 2, 3), "prior2": s(a0 * (a0 + 1) / ((a0 + b0) * (a0 + b0 + 1)), 3),
        "pm4": s(pm, 4),
    }


def p03():
    """窓口対応時間：2段階の指数分布の和（畳み込み）と、チェビシェフの不等式による日数"""
    m1, m2 = 2, 5
    l1, l2 = 1 / m1, 1 / m2
    t = 10
    e2, e5 = 0.1353, 0.0067  # e^{-2}, e^{-5}（問題文に示す）
    ex1, ex2 = e5, e2         # e^{-l1 t}=e^{-5}, e^{-l2 t}=e^{-2}
    num = l1 * ex2 - l2 * ex1
    surv = num / (l1 - l2)
    dens = l1 * l2 / (l1 - l2) * (ex2 - ex1)
    pmax = 1 - (1 - ex1) * (1 - ex2)
    # (b) チェビシェフ
    mu = m1 + m2
    var = m1 ** 2 + m2 ** 2
    eps, alpha = 1, 0.05
    n_cheb = var / (alpha * eps ** 2)
    z = 1.96
    n_clt = z ** 2 * var / eps ** 2
    return {
        "m1": m1, "m2": m2, "l1": s(l1, 1), "l2": s(l2, 1), "t": t, "e2": s(e2, 4), "e5": s(e5, 4),
        "dl": s(l1 - l2, 1), "a1": s(l1 * ex2, 5), "a2": s(l2 * ex1, 5), "num": s(num, 5),
        "surv": s(surv, 3), "surv4": s(surv, 4), "c1": s(l1 / (l1 - l2), 4), "c2": s(l2 / (l1 - l2), 4),
        "w_dens": s(dens, 3), "w_nodiv": s(num, 3), "w_one": s(ex2, 3), "w_max": s(pmax, 3),
        "mu": mu, "var": var, "v1": m1 ** 2, "v2": m2 ** 2, "eps": eps, "alpha": alpha,
        "n_cheb": s(n_cheb, 0), "n_clt_raw": s(n_clt, 2), "n_clt": s(-(-n_clt // 1), 0),
        "n_sd": s(-(-sqrt(var) / alpha // 1), 0), "n_sdsum": s((m1 + m2) ** 2 / alpha, 0), "n_mean": s(mu / alpha, 0),
        "sdY": s(sqrt(var), 3),
    }


def p04():
    """従業員満足度調査：多項分布の人数差の分散、2部署の一様性のカイ二乗検定"""
    p = [0.15, 0.35, 0.25, 0.15, 0.10]
    n = 200
    pa = p[0] + p[1]
    pb = p[3] + p[4]
    va = n * pa * (1 - pa)
    vb = n * pb * (1 - pb)
    cov = -n * pa * pb
    varD = va + vb - 2 * cov
    # (b) 2×3 分割表
    O = [[60, 25, 35], [40, 25, 15]]
    rows = [sum(rw) for rw in O]
    cols = [sum(c) for c in zip(*O)]
    N = sum(rows)
    Ex = [[rows[i] * cols[j] / N for j in range(3)] for i in range(2)]
    terms = [[(O[i][j] - Ex[i][j]) ** 2 / Ex[i][j] for j in range(3)] for i in range(2)]
    chi = sum(map(sum, terms))
    chiO = sum((O[i][j] - Ex[i][j]) ** 2 / O[i][j] for i in range(2) for j in range(3))
    chi_row2 = sum(terms[1])
    out = {
        "p1": s(p[0], 2), "p2": s(p[1], 2), "p3": s(p[2], 2), "p4": s(p[3], 2), "p5": s(p[4], 2), "n": n, "pa": s(pa, 2), "pb": s(pb, 2),
        "qa": s(1 - pa, 2), "qb": s(1 - pb, 2),
        "va": s(va, 1), "vb": s(vb, 1), "cov": s(cov, 1), "covabs": s(-cov, 1), "covx2": s(-2 * cov, 1),
        "varD": s(varD, 1), "w_ind": s(va + vb, 1), "w_sign": s(va + vb + 2 * cov, 1),
        "ez2": s(pa + pb, 2), "ez": s(pa - pb, 2), "vz": s(pa + pb - (pa - pb) ** 2, 4),
        "w_ez2": s(n * (pa + pb), 1), "w_sd": s(sqrt(varD), 2),
        "N": N, "r1": rows[0], "r2": rows[1], "c1": cols[0], "c2": cols[1], "c3": cols[2],
        "chi": s(chi, 2), "chi4": s(chi, 4), "chiO": s(chiO, 2), "chi_row2": s(chi_row2, 2),
        "t1": s(terms[0][1], 4), "t2": s(terms[1][1], 4),
    }
    for i in range(2):
        for j in range(3):
            out[f"o{i + 1}{j + 1}"] = O[i][j]
            out[f"e{i + 1}{j + 1}"] = s(Ex[i][j], 0)
    return out


def p05():
    """ラッシュモデルの能力値の最尤推定とワルド信頼区間"""
    n, k, b = 20, 16, 0.5
    ln4 = 1.3863
    phat = k / n
    th = b + ln4
    info = n * phat * (1 - phat)
    se = r(1 / sqrt(info), 4)
    up = th + 1.96 * se
    return {
        "n": n, "k": k, "b": b, "ln4": s(ln4, 4), "phat": s(phat, 1), "odds": s(phat / (1 - phat), 0),
        "th": s(th, 4), "info": s(info, 1), "pq": s(phat * (1 - phat), 2), "se": s(se, 4), "hw": s(1.96 * se, 4),
        "up": r(up, 2), "up_s": s(up, 2), "up4": s(up, 4),
        "w_nosqrt": s(th + 1.96 / info, 2), "w_noN": s(th + 1.96 / sqrt(phat * (1 - phat)), 2),
        "w_theta0": s(ln4 + 1.96 * se, 2),
    }


def p06():
    """研修前後の得点：対応のあるt区間、ホルム法"""
    before = [62, 55, 70, 48, 66, 59, 73, 51]
    d_true = [5, 8, -2, 6, 3, 9, 4, 7]
    after = [a + d for a, d in zip(before, d_true)]
    d = [a - b for a, b in zip(after, before)]
    n = len(d)
    dbar = sum(d) / n
    ss = sum((x - dbar) ** 2 for x in d)
    s2 = ss / (n - 1)
    sd = r(sqrt(s2), 4)
    se = r(sqrt(s2 / n), 4)
    t7, t8, z = 2.365, 2.306, 1.96
    hw = t7 * se
    low = dbar - hw
    sd_mle = sqrt(ss / n)
    out = {
        "n": n, "df": n - 1, "sumd": sum(d), "dbar": s(dbar, 1), "ss": s(ss, 0), "s2": s(s2, 0), "sd": s(sd, 4),
        "s2n": s(s2 / n, 1), "se": s(se, 4), "t7": t7, "t8": t8, "hw": s(hw, 4), "low": r(low, 2), "low_s": s(low, 2),
        "low4": s(low, 4),
        "w_z": s(dbar - z * se, 2), "w_t8": s(dbar - t8 * se, 2), "w_mle": s(dbar - t7 * sd_mle / sqrt(n), 2),
    }
    for i in range(n):
        out[f"b{i + 1}"] = before[i]
        out[f"a{i + 1}"] = after[i]
        out[f"d{i + 1}"] = d[i]
    # (b) ホルム法
    pv = {"国語": 0.014, "数学": 0.004, "英語": 0.045, "理科": 0.011, "社会": 0.036}
    for i, (kk, v) in enumerate(pv.items()):
        out[f"pv{i + 1}"] = s(v, 3)
    m = len(pv)
    srt = sorted(pv.items(), key=lambda kv: kv[1])
    th = [0.05 / (m - i) for i in range(m)]
    rej = []
    for (kk, v), t in zip(srt, th):
        if v <= t:
            rej.append(kk)
        else:
            break
    assert rej == ["数学", "理科", "国語"], rej
    # (b) 平均既知の正規分布の分散の尤度比検定
    import math
    n_, mu_, s0, ss = 30, 60, 100, 1800
    sh = ss / n_
    ratio = sh / s0
    lg = -0.5108                                 # log 0.6（問題文の値）
    T = n_ * (ratio - 1 - lg)
    assert abs(math.log(ratio) - lg) < 5e-5 and T < 3.841
    lr = {"n": n_, "mu": mu_, "s0": s0, "ss": ss, "sh": s(sh, 0), "ratio": s(ratio, 1), "in": s(ratio - 1 - lg, 4),
          "T": s(T, 2), "T4": s(T, 4), "w1": s(n_ * (ratio - lg), 2), "w2": s(n_ * (1 / ratio - 1 + lg), 2), "exact": s(ss / s0, 0)}
    assert float(lr["w1"]) > 3.841 and float(lr["w2"]) > 3.841
    for i in range(m):
        out[f"th{i + 1}"] = s(th[i], 4)
    out.update({"lr_" + k_: v_ for k_, v_ in lr.items()})
    return out


def p07():
    """3つの指導法のクラス内比較：フリードマン検定統計量"""
    A = [62, 58, 67, 55, 60, 69]
    B = [66, 70, 64, 61, 72, 66]
    C = [71, 65, 73, 63, 68, 75]
    b, k = 6, 3
    ranks = []
    for row in zip(A, B, C):
        srt = sorted(row)
        assert len(set(row)) == 3
        ranks.append([srt.index(v) + 1 for v in row])
    RA, RB, RC = (sum(rw[j] for rw in ranks) for j in range(3))
    sumR2 = RA ** 2 + RB ** 2 + RC ** 2
    den = b * k * (k + 1)
    Q = 12 / den * sumR2 - 3 * b * (k + 1)
    out = {"b": b, "k": k, "k1": k + 1, "RA": RA, "RB": RB, "RC": RC, "Rsum": RA + RB + RC,
           "RA2": RA ** 2, "RB2": RB ** 2, "RC2": RC ** 2, "sumR2": sumR2, "den": den, "c3": 3 * b * (k + 1),
           "c3w": 3 * b * k, "prod": s(12 / den * sumR2, 4), "Q": r(Q, 2), "Q_s": s(Q, 2), "Q4": s(Q, 4),
           "w_bk": s(12 / den * sumR2 - 3 * b * k, 2),
           "rA": ",".join(str(rw[0]) for rw in ranks), "rB": ",".join(str(rw[1]) for rw in ranks),
           "rC": ",".join(str(rw[2]) for rw in ranks)}
    for nm, g in (("A", A), ("B", B), ("C", C)):
        for i, v in enumerate(g):
            out[f"{nm}{i + 1}"] = v
    return out


def p08():
    """賃金の回帰：部分F検定、ブルーシュ＝ペーガン検定、操作変数法"""
    n = 60
    rss1, rss2 = 4.80, 3.90
    q, k2 = 3, 5
    df2 = n - k2 - 1
    F = ((rss1 - rss2) / q) / (rss2 / df2)
    ms_num = (rss1 - rss2) / q
    ms_den = rss2 / df2
    # (b) BP
    R2 = 0.09
    lm = n * R2
    # (c) IV
    sn, sf, wn, wf = 13.6, 13.1, 6.452, 6.397
    rf = wn - wf
    fs = sn - sf
    iv = rf / fs
    ols = 0.085
    # (b) 頑健標準誤差による t 値
    b_, se_, hc_, df_, tc_ = 0.032, 0.012, 0.019, 54, 2.005
    t1_, t2_ = b_ / se_, b_ / hc_
    assert t2_ < tc_ < t1_
    rb = {"b": s(b_, 3), "se": s(se_, 3), "hc": s(hc_, 3), "df": df_, "tc": s(tc_, 3), "t1": s(t1_, 2), "t2": s(t2_, 2),
          "t14": s(t1_, 4), "t24": s(t2_, 4), "w": s(b_ / hc_ ** 2, 1)}
    # (c) 除外変数バイアス
    bo, g_, vs, c_ = 0.085, 0.10, 4.0, 1.0
    d_ = c_ / vs
    ov = {"b": s(bo, 3), "g": s(g_, 2), "vs": s(vs, 1), "c": s(c_, 1), "d": s(d_, 2), "bias": s(g_ * d_, 3),
          "ok": s(bo - g_ * d_, 3), "w1": s(bo + g_ * d_, 3), "w3": s(bo * (1 - d_), 3), "w4": s(bo - g_, 3)}
    assert len({ov["ok"], ov["w1"], ov["b"], ov["w3"], ov["w4"]}) == 5
    return {
        **{"rb_" + k_: v_ for k_, v_ in rb.items()}, **{"ov_" + k_: v_ for k_, v_ in ov.items()},
        "n": n, "rss1": s(rss1, 2), "rss2": s(rss2, 2), "q": q, "k2": k2, "df2": df2, "drss": s(rss1 - rss2, 2),
        "msn": s(ms_num, 2), "msd": s(ms_den, 6), "F": s(F, 2), "F4": s(F, 4),
        "F_df55": s(ms_num / (rss2 / (n - k2)), 2), "F_q1": s((rss1 - rss2) / ms_den, 2),
        "F_rss1": s(ms_num / (rss1 / df2), 2), "F_df57": s(ms_num / (rss2 / (n - 3)), 2),
        "R2": s(R2, 2), "lm": s(lm, 2), "w_1mr": s(n * (1 - R2), 2),
        "sn": s(sn, 1), "sf": s(sf, 1), "wn": s(wn, 3), "wf": s(wf, 3), "rf": s(rf, 3), "fs": s(fs, 1),
        "iv": s(iv, 3), "ols": s(ols, 3), "w_inv": s(fs / rf, 2),
    }


def p09():
    """離職のプロビットモデルと、人事ローテーションのマルコフ連鎖"""
    b0, b1 = -1.40, 0.025
    h1, h2 = 20, 60
    z1, z2 = b0 + b1 * h1, b0 + b1 * h2
    assert abs(z1 + 0.9) < 1e-12 and abs(z2 - 0.1) < 1e-12
    p1 = 1 - PHI[0.90]
    p2 = PHI[0.10]
    inc = p2 - p1
    # (b) 人事ローテーション（周期 2 の連鎖）
    q = 0.6
    piS = 1 / 2
    piP, piD = q * piS, (1 - q) * piS
    return {
        "b0": s(b0, 2), "b1": s(b1, 3), "h1": h1, "h2": h2, "z1": s(z1, 2), "z2": s(z2, 2),
        "phi0": s(PHI[0.0], 4), "phi10": s(PHI[0.10], 4), "phi90": s(PHI[0.90], 4), "phi100": s(PHI[1.00], 4),
        "p1": s(p1, 4), "p2": s(p2, 4), "inc": s(inc, 3), "inc4": s(inc, 4),
        "w_p1": s(p1, 3), "w_sign": s(PHI[0.90] - PHI[0.10], 3), "w_coef": s(PHI[1.00] - PHI[0.0], 3), "w_p2": s(p2, 3),
        "dz": s(b1 * (h2 - h1), 1),
        "q": s(q, 1), "q2": s(1 - q, 1), "piS": s(piS, 1), "piP": s(piP, 1), "piD": s(piD, 1),
        "w_eq": "1/3",
    }


def p10():
    """評価者を変量効果とした一元配置：級内相関"""
    a, n = 5, 4
    SSA, SSE = 104, 108
    dfa, dfe = a - 1, a * (n - 1)
    MSA, MSE = SSA / dfa, SSE / dfe
    sa2 = (MSA - MSE) / n
    icc = sa2 / (sa2 + MSE)
    F = MSA / MSE
    return {
        "a": a, "nr": n, "SSA": SSA, "SSE": SSE, "SST": SSA + SSE, "dfa": dfa, "dfe": dfe, "dft": a * n - 1,
        "MSA": s(MSA, 1), "MSE": s(MSE, 1), "dMS": s(MSA - MSE, 1), "sa2": s(sa2, 2), "tot": s(sa2 + MSE, 2),
        "icc": s(icc, 3), "icc4": s(icc, 4), "F": s(F, 4),
        "w_a": s(((MSA - MSE) / a) / ((MSA - MSE) / a + MSE), 3), "w_eta": s(SSA / (SSA + SSE), 3),
        "w_non": s((MSA - MSE) / (MSA - MSE + MSE), 3), "w_msa": s((MSA / n) / (MSA / n + MSE), 3),
        "sa2_a": s((MSA - MSE) / a, 2),
    }


def p11():
    """学校調査：大規模校を全数・小規模校を単純無作為抽出したときのホーヴィッツ＝トンプソン推定量、多重代入のルービンのルール"""
    Nsch, Nl, n = 20, 3, 4
    Ns = Nsch - Nl
    yl = [27, 31, 22]
    ys = [6, 9, 4, 5]
    sl, ss = sum(yl), sum(ys)
    ht = sl + Ns / n * ss
    assert abs(ht - 182) < 1e-12
    # (b) Rubin
    est = [8.2, 8.6, 8.0, 8.5, 8.7]
    u = [0.090, 0.094, 0.086, 0.092, 0.088]
    m = len(est)
    qbar = sum(est) / m
    W = sum(u) / m
    B = sum((e - qbar) ** 2 for e in est) / (m - 1)
    T = W + (1 + 1 / m) * B
    out = {
        "Nsch": Nsch, "Nl": Nl, "Ns": Ns, "nsamp": n, "nall": Nl + n,
        "yl1": yl[0], "yl2": yl[1], "yl3": yl[2], "ys1": ys[0], "ys2": ys[1], "ys3": ys[2], "ys4": ys[3],
        "sl": sl, "ss": ss, "pis": f"{n}/{Ns}", "wexp": s(Ns / n, 2), "tsmall": s(Ns / n * ss, 0),
        "ht": s(ht, 0), "w_small": s(Ns / n * ss, 0), "w_pi20": s(sl + Nsch / n * ss, 0),
        "w_srs": s(Nsch / (Nl + n) * (sl + ss), 1), "w_all": s(Ns / n * (sl + ss), 0),
        "m": m, "qbar": s(qbar, 1), "W": s(W, 3), "sumdev": s(sum((e - qbar) ** 2 for e in est), 2),
        "B": s(B, 3), "fB": s((1 + 1 / m) * B, 3), "T": s(T, 3), "se": r(sqrt(T), 3), "se_s": s(sqrt(T), 3),
        "se4": s(sqrt(T), 4), "w_wb": s(sqrt(W + B), 3), "w_w": s(sqrt(W), 3),
        "w_bm": s(sqrt(W + (1 + 1 / m) * sum((e - qbar) ** 2 for e in est) / m), 3),
        "w_m": s(sqrt(T / m), 3),
    }
    for i in range(m):
        out[f"q{i + 1}"] = s(est[i], 1)
        out[f"u{i + 1}"] = s(u[i], 3)
        out[f"dv{i + 1}"] = s(est[i] - qbar, 1)
    return out


def p12():
    """適性検査：分散共分散行列の主成分分析と k-means 法"""
    s11, s12, s22 = 30, 12, 20
    half = (s11 + s22) / 2
    rad = sqrt(((s11 - s22) / 2) ** 2 + s12 ** 2)
    l1, l2 = half + rad, half - rad
    tr = s11 + s22
    rcor = s12 / sqrt(s11 * s22)
    # (b) k-means
    pts = {"A": (2, 3), "B": (3, 4), "C": (4, 2), "D": (8, 7), "E": (9, 9), "F": (7, 8)}
    c1, c2 = pts["A"], pts["C"]
    d2 = lambda p, c: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2

    def step(c1, c2):
        g1 = [k for k, p in pts.items() if d2(p, c1) <= d2(p, c2)]
        g2 = [k for k in pts if k not in g1]
        cen = lambda g: (sum(pts[k][0] for k in g) / len(g), sum(pts[k][1] for k in g) / len(g))
        return g1, g2, cen(g1), cen(g2)

    g1, g2, m1, m2 = step(c1, c2)
    assert g1 == ["A", "B"]
    wss = lambda g, c: sum(d2(pts[k], c) for k in g)
    W1 = wss(g1, m1) + wss(g2, m2)
    h1, h2, n1, n2 = step(m1, m2)
    assert h1 == ["A", "B", "C"]
    W2 = wss(h1, n1) + wss(h2, n2)
    h1b, h2b, _, _ = step(n1, n2)
    assert h1b == h1  # 収束
    pair = sum(d2(pts[a], pts[b]) for g in (h1, h2) for i, a in enumerate(g) for b in g[i + 1:])
    out = {
        "s11": s11, "s12": s12, "s22": s22, "half": s(half, 0), "dh": s((s11 - s22) / 2, 0), "rad2": s(rad ** 2, 0),
        "rad": s(rad, 0), "l1": s(l1, 0), "l2": s(l2, 0), "tr": tr,
        "cr": s(l1 / tr, 3), "w_l2": s(l2 / tr, 3), "w_var": s(s11 / tr, 3), "w_cor": s((1 + rcor) / 2, 3),
        "w_all": s(l1 / (s11 + 2 * s12 + s22), 3), "rcor": s(rcor, 4),
        "m1x": s(m1[0], 1), "m1y": s(m1[1], 1), "m2x": s(m2[0], 1), "m2y": s(m2[1], 1),
        "n1x": s(n1[0], 0), "n1y": s(n1[1], 0), "n2x": s(n2[0], 0), "n2y": s(n2[1], 0),
        "W1": s(W1, 1), "W2": s(W2, 1), "W2a": s(wss(h1, n1), 0), "W2b": s(wss(h2, n2), 0),
        "w_pair": s(pair, 1), "w_avg": s(wss(h1, n1) / 3 + wss(h2, n2) / 3, 2), "w_one": s(wss(h1, n1), 1),
        "wt1": "+".join(f"({(pts[k][0] - n1[0]) ** 2:g}+{(pts[k][1] - n1[1]) ** 2:g})" for k in h1),
        "wt2": "+".join(f"({(pts[k][0] - n2[0]) ** 2:g}+{(pts[k][1] - n2[1]) ** 2:g})" for k in h2),
        "dC1": s(d2(pts["C"], m1), 2), "dC2": s(d2(pts["C"], m2), 2),
    }
    for k, (x, y) in pts.items():
        out[f"{k}x"], out[f"{k}y"] = x, y
        out[f"{k}1"], out[f"{k}2"] = d2((x, y), c1), d2((x, y), c2)
    return out


def p13():
    """有効求人倍率の AR(2)：ユール＝ウォーカー推定と定常条件"""
    r1, r2 = 0.8, 0.5
    den = 1 - r1 ** 2
    f1 = r1 * (1 - r2) / den
    f2 = (r2 - r1 ** 2) / den
    return {"r1": s(r1, 1), "r2": s(r2, 1), "r1d": s(r1, 3), "r2d": s(r2, 3), "r1sq": s(r1 ** 2, 2), "den": s(den, 2), "n1": s(r1 * (1 - r2), 2),
            "n2": s(r2 - r1 ** 2, 2), "f1": s(f1, 3), "f2": s(f2, 3), "f2abs": s(-f2, 3),
            "sum": s(f1 + f2, 3), "dif": s(f2 - f1, 3), "w_f2": s(r2 - r1 ** 2, 3)}


def p14():
    """研修時間と得点の回帰：てこ比を使った1つ抜き交差検証"""
    x = [2, 4, 5, 6, 8, 11]
    y = [54, 61, 58, 66, 67, 70]
    n = len(x)
    xb = sum(x) / n
    yb = sum(y) / n
    sxx = sum((a - xb) ** 2 for a in x)
    sxy = sum((a - xb) * (b - yb) for a, b in zip(x, y))
    b1 = sxy / sxx
    b0 = yb - b1 * xb
    e = [r(b - (b0 + b1 * a), 3) for a, b in zip(x, y)]
    h = [r(1 / n + (a - xb) ** 2 / sxx, 3) for a in x]
    loo = [r(ei / (1 - hi), 3) for ei, hi in zip(e, h)]   # 解説に示す丸めた値
    sq = [r(v ** 2, 3) for v in loo]
    cv = sum(sq) / n
    mse = sum(v ** 2 for v in e) / n
    out = {"n": n, "xb": s(xb, 0), "sxx": s(sxx, 0), "b0": s(b0, 3), "b1": s(b1, 2),
           "cv": r(cv, 1), "cv_s": s(cv, 1), "cv4": s(cv, 4), "sumsq": s(sum(sq), 3),
           "mse": s(mse, 2), "w_mse": s(mse, 1), "w_np": s(sum(v ** 2 for v in e) / (n - 2), 2),
           "w_lin": s(sum(ei ** 2 / (1 - hi) for ei, hi in zip(e, h)) / n, 2),
           "w_plus": s(sum((ei / (1 + hi)) ** 2 for ei, hi in zip(e, h)) / n, 2)}
    for i in range(n):
        out[f"x{i + 1}"] = x[i]
        out[f"y{i + 1}"] = y[i]
        out[f"e{i + 1}"] = s(e[i], 3)
        out[f"h{i + 1}"] = s(h[i], 3)
        out[f"o{i + 1}"] = s(1 - h[i], 3)
        out[f"l{i + 1}"] = s(loo[i], 3)
        out[f"l{i + 1}sq"] = s(sq[i], 3)
    return out


def values():
    return {f"s11p{i:02d}": f() for i, f in enumerate(
        (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14), start=1)}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
