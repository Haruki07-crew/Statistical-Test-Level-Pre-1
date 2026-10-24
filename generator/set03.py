"""模擬試験 第3回 の数値。values() が {大問id: {名前: 値}} を返す。

方針：問題文に示した値（分布表の値・指数関数の値など）を丸めた形で使い、
受験者が一般電卓で計算したときと同じ数値になるようにする。
"""
import math

from scipy import stats


def f(x, d):
    """小数 d 桁の文字列（四捨五入）。"""
    return f"{x:.{d}f}"


def phi(z):
    return float(stats.norm.cdf(z))


def p01():
    """圃場の収穫量（2種類の土壌の混合）"""
    wA, mA, sA = 0.3, 60.0, 5.0
    wB, mB, sB = 0.7, 45.0, 8.0
    c = 55.0
    mean = wA * mA + wB * mB
    ev = wA * sA ** 2 + wB * sB ** 2
    ve = wA * wB * (mA - mB) ** 2
    var = ev + ve
    zA = (c - mA) / sA            # -1.00
    zB = (c - mB) / sB            # 1.25
    PhiA = round(phi(-zA), 4)     # Φ(1.00)
    PhiB = round(phi(zB), 4)      # Φ(1.25)
    zN = round((c - mean) / math.sqrt(var), 2)
    PhiN = round(phi(zN), 4)
    pA = PhiA                      # P(Y>=55 | A) = Φ(1)
    pB = 1 - PhiB                  # P(Y>=55 | B) = 1-Φ(1.25)
    joint = wA * pA
    tot = joint + wB * pB
    post = joint / tot
    return dict(
        wA=wA, wB=wB, mA=int(mA), mB=int(mB), sA=int(sA), sB=int(sB), c=int(c),
        mean=f(mean, 1), ev=f(ev, 1), ve=f(ve, 2), var=round(var, 2), var_s=f(var, 2),
        sd=f(math.sqrt(var), 2), sA2=int(sA ** 2), sB2=int(sB ** 2), diff=int(mA - mB),
        zA=f(-zA, 2), zB=f(zB, 2), zN=f(zN, 2), PhiA=f(PhiA, 4), PhiB=f(PhiB, 4), PhiN=f(PhiN, 4),
        pA=f(pA, 4), pB=f(pB, 4), jA=f(joint, 5), jB=f(wB * pB, 5), tot5=f(tot, 5),
        tot=f(tot, 3),
        d_norm=f(1 - PhiN, 3),                     # 混合を1つの正規分布で近似
        d_avg=f((pA + pB) / 2, 3),                 # 重みを無視
        d_signA=f(wA * (1 - PhiA) + wB * pB, 3),   # A 側で下側確率
        d_signB=f(wA * pA + wB * PhiB, 3),         # B 側で下側確率
        post=f(post, 3), post4=f(post, 4),
        d_lik=f(pA, 3), d_prior=f(wA, 3), d_joint=f(joint, 3), d_postB=f(1 - post, 3),
    )


def p02():
    """灌漑ポンプの寿命（ハザード・カプラン＝マイヤー）"""
    a = 0.08                       # h(t) = a t
    t1, t2 = 3, 5
    H = lambda t: a * t ** 2 / 2
    e = lambda x: round(math.exp(-x), 4)
    cond = e(H(t2) - H(t1))
    # KM
    times = [(1.5, 1), (2.0, 0), (2.8, 1), (3.1, 1), (3.5, 0), (4.2, 1), (4.6, 0), (5.0, 1), (5.5, 0), (6.0, 0)]
    t0 = 4.5
    S, n = 1.0, len(times)
    steps = []
    for t, d in times:
        if t > t0:
            break
        if d:
            steps.append((t, n, (n - 1) / n))
            S *= (n - 1) / n
        n -= 1
    # 打ち切りでリスク集合を減らさない
    S_nocens, n2 = 1.0, len(times)
    for t, d in times:
        if t > t0:
            break
        if d:
            S_nocens *= (n2 - 1) / n2
            n2 -= 1
    naive = sum(1 for t, d in times if t > t0) / len(times)
    unc = [t for t, d in times if d]
    drop = sum(1 for t in unc if t > t0) / len(unc)
    # 4.2 の時点のリスク集合を 1 つ多く数えた
    S_off = steps[0][2] * steps[1][2] * steps[2][2] * (5 / 6)
    return dict(
        a=a, a2=a / 2, t1=t1, t2=t2, Ht1=f(H(t1), 2), Ht2=f(H(t2), 2), dH=f(H(t2) - H(t1), 2),
        e064=f(cond, 4), e016=f(e(a / 2 * (t2 - t1) ** 2), 4), e1=f(e(H(t2)), 4), e128=f(e(a * (t2 ** 2 - t1 ** 2)), 4),
        cond=f(cond, 3), d_mem=f(e(a / 2 * (t2 - t1) ** 2), 3), d_unc=f(e(H(t2)), 3),
        d_noint=f(e(a * (t2 ** 2 - t1 ** 2)), 3), d_fail=f(1 - cond, 3),
        km=f(S, 3), km_steps=" \\times ".join(f"\\frac{{{m - 1}}}{{{m}}}" for _, m, _ in steps),
        d_nocens=f(S_nocens, 3), d_naive=f(naive, 3), d_drop=f(drop, 3), d_off=f(S_off, 3),
    )


def p03():
    """交差点のニアミス件数（ポアソン・適合度・ガンマ事前分布）"""
    obs = {0: 38, 1: 40, 2: 26, 3: 12, 4: 4}
    n = sum(obs.values())
    tot = sum(k * v for k, v in obs.items())
    lam = tot / n                          # 1.2
    e = 0.3012                             # e^{-1.2}（問題文で与える）
    p = [e * lam ** k / math.factorial(k) for k in range(4)]
    E0, E1, E2, E3 = (n * x for x in p)
    E_ge3 = n - (E0 + E1 + E2)
    O = [obs[0], obs[1], obs[2], obs[3] + obs[4]]
    Ec = [E0, E1, E2, E_ge3]
    chi = sum((o - x) ** 2 / x for o, x in zip(O, Ec))
    # 誤り：3 以上の期待度数に P(X=3) だけを使った
    Ebad = [E0, E1, E2, E3]
    chi_tail = sum((o - x) ** 2 / x for o, x in zip(O, Ebad))
    # 誤り：5 セルのまま（4 以上を別セル）
    E_ge4 = n - (E0 + E1 + E2 + E3)
    O5 = [obs[0], obs[1], obs[2], obs[3], obs[4]]
    E5 = [E0, E1, E2, E3, E_ge4]
    chi5 = sum((o - x) ** 2 / x for o, x in zip(O5, E5))
    terms = [(o - x) ** 2 / x for o, x in zip(O, Ec)]
    # ベイズ
    a0, b0 = 6, 4
    a1, b1 = a0 + tot, b0 + n
    post = a1 / b1
    return dict(
        n=n, tot=tot, lam=f(lam, 1), e12=f(e, 4), o0=obs[0], o1=obs[1], o2=obs[2], o3=obs[3], o4=obs[4], o3p=O[3],
        p0=f(p[0], 4), p1=f(p[1], 4), p2=f(p[2], 4), p3=f(p[3], 4),
        E0=f(E0, 2), E1=f(E1, 2), E2=f(E2, 2), E3=f(E3, 2), Ege3=f(E_ge3, 2), Ege4=f(E_ge4, 2),
        E2ans=round(E2, 1), E2_2=f(E2, 3),
        t0=f(terms[0], 4), t1=f(terms[1], 4), t2=f(terms[2], 4), t3=f(terms[3], 4),
        chi=f(chi, 3), chi_tail=f(chi_tail, 3), chi5=f(chi5, 3),
        a0=a0, b0=b0, a1=a1, b1=b1, prior_mean=f(a0 / b0, 1),
        post=f(post, 3), d_scale=f((a0 + tot) / (1 / b0 + n), 3), d_swap=f((a0 + n) / (b0 + tot), 3),
        d_mle=f(lam, 3), d_avg=f((a0 / b0 + lam) / 2, 3), w=f(n / (b0 + n), 4),
        post_var=f(a1 / b1 ** 2, 5),
    )


def p04():
    """車頭時間（指数分布・不変性・デルタ法・χ² による区間推定）"""
    n, S = 40, 300.0
    xbar = S / n                           # 7.5 秒
    t = 10
    e43 = round(math.exp(-t / xbar), 4)
    e075 = round(math.exp(-xbar / t), 4)
    e13 = round(math.exp(-t / (S / (n - 1))), 4)
    lam_s = 1 / xbar
    lam_m = 60 / xbar                      # 8 台/分
    se = lam_m / math.sqrt(n)
    # χ² 分位点（問題文で与える）
    q80u, q80l = round(stats.chi2.ppf(0.975, 80), 2), round(stats.chi2.ppf(0.025, 80), 2)
    q40u, q40l = round(stats.chi2.ppf(0.975, 40), 2), round(stats.chi2.ppf(0.025, 40), 2)
    ci = (2 * S / q80u, 2 * S / q80l)
    ci40 = (2 * S / q40u, 2 * S / q40l)
    ciS = (S / q80u, S / q80l)
    ciN = (xbar - 1.96 * xbar / math.sqrt(n), xbar + 1.96 * xbar / math.sqrt(n))
    iv = lambda c: f"({c[0]:.2f},\\ {c[1]:.2f})"
    return dict(
        n=n, S=int(S), xbar=f(xbar, 1), t=t, ratio=f(t / xbar, 4),
        e43=f(e43, 4), e075=f(e075, 4), e13=f(e13, 4), xbar_u=f(S / (n - 1), 3),
        pmle=f(e43, 3), d_cdf=f(1 - e43, 3), d_rate=f(e075, 3), d_unb=f(e13, 3), d_lam=f(lam_s, 3),
        lam_s=f(lam_s, 4), lam_m=f(lam_m, 1), sqn=f(math.sqrt(n), 4), se=round(se, 2), se4=f(se, 4),
        theta_m=f(xbar / 60, 4), se_theta=f(xbar / math.sqrt(n), 3),
        q80u=f(q80u, 2), q80l=f(q80l, 2), q40u=f(q40u, 2), q40l=f(q40l, 2), twoS=int(2 * S),
        ci=iv(ci), ci40=iv(ci40), ciS=iv(ciS), ciN=iv(ciN), ci_lo=f(ci[0], 2), ci_hi=f(ci[1], 2),
        ci_half=f(1.96 * xbar / math.sqrt(n), 2),
        ciR=f"({q80l / (2 * S):.3f},\\ {q80u / (2 * S):.3f})",
    )


def p05():
    """新しい学習アプリの効果（検出力・サンプルサイズ、符号付き順位検定）"""
    sigma, delta = 10, 5
    za, zb, za2 = 1.645, 1.282, 1.960
    need = ((za + zb) * sigma / delta) ** 2
    need2 = ((za2 + zb) * sigma / delta) ** 2
    n_ans = math.ceil(need)
    d = [7, -3, 12, 5, -1, 9, 4, -6, 11, 2]
    absr = sorted(abs(x) for x in d)
    rank = {v: i + 1 for i, v in enumerate(absr)}
    Wp = sum(rank[abs(x)] for x in d if x > 0)
    Wm = sum(rank[abs(x)] for x in d if x < 0)
    m = len(d)
    EW = m * (m + 1) / 4
    VW = m * (m + 1) * (2 * m + 1) / 24
    z = (Wp - EW) / math.sqrt(VW)
    z12 = (Wp - EW) / math.sqrt(2 * VW)
    zv = (Wp - EW) / VW
    zm = (Wm - EW) / math.sqrt(VW)
    neg = ", ".join(f"{-x}" for x in d if x < 0)
    return dict(
        sigma=sigma, delta=delta, za=f(za, 3), zb=f(zb, 3), za2=f(za2, 3), zsum=f(za + zb, 3),
        need=f(need, 2), n_ans=n_ans, need2=f(need2, 2), n2=math.ceil(need2), ratio=f(sigma / delta, 0),
        need_sd=f((za + zb) * sigma / delta, 3),
        data=", ".join(f"{x:+d}".replace("+", "+") for x in d), m=m,
        Wp=Wp, Wm=Wm, EW=f(EW, 1), VW=f(VW, 2), sdW=f(math.sqrt(VW), 3), z=f(z, 2), z12=f(z12, 2),
        zv=f(zv, 2), zm=f(zm, 2), tot_rank=m * (m + 1) // 2,
        negranks=" + ".join(str(rank[abs(x)]) for x in d if x < 0), neg=neg,
    )


def p06():
    """天気の3状態マルコフ連鎖"""
    P = [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]]
    # 定常分布：πS = 1.6 πR, πC = (0.44/0.3) πR を使わず連立方程式を厳密に（分数で）解く
    from fractions import Fraction as F
    Pf = [[F(str(x)) for x in r] for r in P]
    # π(P - I) = 0, sum = 1 をガウス消去で
    A = [[Pf[j][i] - (1 if i == j else 0) for j in range(3)] for i in range(3)]
    A[2] = [F(1)] * 3
    b = [F(0), F(0), F(1)]
    for i in range(3):
        piv = A[i][i]
        for k in range(i + 1, 3):
            r = A[k][i] / piv
            A[k] = [A[k][j] - r * A[i][j] for j in range(3)]
            b[k] -= r * b[i]
    x = [F(0)] * 3
    for i in reversed(range(3)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, 3))) / A[i][i]
    piS, piC, piR = x
    two = sum(Pf[0][k] * Pf[k][2] for k in range(3))
    colmean = sum(Pf[k][2] for k in range(3)) / 3
    return dict(
        piS=f(float(piS), 3), piC=f(float(piC), 3), piR=f(float(piR), 3),
        piR_frac=f"\\frac{{{piR.numerator}}}{{{piR.denominator}}}",
        piS_frac=f"\\frac{{{piS.numerator}}}{{{piS.denominator}}}",
        piC_frac=f"\\frac{{{piC.numerator}}}{{{piC.denominator}}}",
        d_col=f(float(colmean), 3), d_self=f(float(Pf[2][2]), 3), d_unif=f(1 / 3, 3), d_two=f(float(two), 3),
    )


def p07():
    """小麦の収量の重回帰（自由度調整済み決定係数・クックの距離・除外変数バイアス）"""
    n, p, R2 = 30, 2, 0.62
    adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    d1 = 1 - (1 - R2) * (n - 1) / (n - p)
    d2 = 1 - (1 - R2) * n / (n - p - 1)
    d3 = 1 - (1 - R2) * (n - p - 1) / (n - 1)
    r, h = 2.4, 0.28
    D = r ** 2 / (p + 1) * h / (1 - h)
    b1, b2, g = 1.8, 0.05, -4.0
    simple = b1 + b2 * g
    return dict(
        n=n, p=p, R2=f(R2, 2), adj=f(adj, 3), d_np=f(d1, 3), d_n=f(d2, 3), d_r2=f(R2, 3), d_inv=f(d3, 3),
        ratio=f((n - 1) / (n - p - 1), 4), one_r2=f(1 - R2, 2), dfE=n - p - 1, dfT=n - 1,
        r=f(r, 1), h=f(h, 2), r2=f(r ** 2, 2), pp1=p + 1, h1h=f(h / (1 - h), 4), D=round(D, 2), D4=f(D, 4),
        one_h=f(1 - h, 2), r2p=f(r ** 2 / (p + 1), 2),
        b1=f(b1, 2), b2=f(b2, 2), g=f(g, 1), b2g=f(b2 * g, 2), simple=f(simple, 2),
        d_sign=f(b1 - b2 * g, 2), d_add=f(b1 + b2, 2), d_same=f(b1, 2),
    )


def p08():
    """検定の合否のロジスティック回帰と、AIC・BIC によるモデル選択"""
    a, b, x = -2.0, 0.35, 4
    eta = a + b * x                         # -0.6
    e06, e14, e20 = round(math.exp(0.6), 4), round(math.exp(1.4), 4), round(math.exp(2.0), 4)
    p = 1 / (1 + e06)
    d_odds = 1 / e06                        # e^{-0.6} を確率とみなす
    d_sign = e06 / (1 + e06)                # 1/(1+e^{-0.6})
    d_noint = e14 / (1 + e14)
    p0 = 1 / (1 + e20)
    d_mult = p0 * e14
    # モデル選択
    n = 200
    m2ll = {"M1": 236.8, "M2": 232.0, "M3": 229.9}
    k = {"M1": 2, "M2": 3, "M3": 4}
    ln200 = round(math.log(n), 3)
    aic = {m: m2ll[m] + 2 * k[m] for m in m2ll}
    bic = {m: m2ll[m] + ln200 * k[m] for m in m2ll}
    out = dict(
        a=f(a, 1), b=f(b, 2), x=x, eta=f(eta, 1), e06=f(e06, 4), e14=f(e14, 4), e20=f(e20, 4),
        p=f(p, 3), d_odds=f(d_odds, 3), d_sign=f(d_sign, 3), d_noint=f(d_noint, 3), d_mult=f(d_mult, 3),
        inv_e06=f(d_odds, 4), p0=f(p0, 4), one_e06=f(1 + e06, 4), bx=f(b * x, 1),
        n=n, ln200=f(ln200, 3), log10=f(round(math.log10(n), 3), 3),
        best_aic=min(aic, key=aic.get), best_bic=min(bic, key=bic.get),
    )
    for m in m2ll:
        out[f"m2ll_{m}"] = f(m2ll[m], 1)
        out[f"k_{m}"] = k[m]
        out[f"aic_{m}"] = f(aic[m], 1)
        out[f"bic_{m}"] = f(bic[m], 2)
        out[f"pen_{m}"] = f(ln200 * k[m], 3)
        out[f"bic10_{m}"] = f(m2ll[m] + round(math.log10(n), 3) * k[m], 2)
    return out


def p09():
    """肥料の乱塊法（二元配置・繰り返しなし）"""
    a, bk = 4, 5
    means = [52.0, 55.5, 49.0, 53.5]
    gm = sum(means) / a
    SA = bk * sum((m - gm) ** 2 for m in means)
    ST, SB = 230.0, 70.0
    SE = ST - SA - SB
    dfA, dfB, dfE = a - 1, bk - 1, (a - 1) * (bk - 1)
    MSA, MSE = SA / dfA, SE / dfE
    F = MSA / MSE
    SE1 = ST - SA
    dfE1 = a * bk - a
    F1 = MSA / (SE1 / dfE1)
    return dict(
        a=a, bk=bk, N=a * bk, m1=f(means[0], 1), m2=f(means[1], 1), m3=f(means[2], 1), m4=f(means[3], 1),
        gm=f(gm, 2), dev=" + ".join(f"({m - gm:+.1f})^2".replace("+", "") for m in means),
        ssdev=f(sum((m - gm) ** 2 for m in means), 2),
        SA=f(SA, 1), ST=f(ST, 1), SB=f(SB, 1), SE=f(SE, 1), dfA=dfA, dfB=dfB, dfE=dfE,
        MSA=f(MSA, 2), MSE=f(MSE, 4), F=round(F, 2), F4=f(F, 4),
        SE1=f(SE1, 1), dfE1=dfE1, MSE1=f(SE1 / dfE1, 4), F1=f(F1, 2),
        f05=f(stats.f.ppf(0.95, dfA, dfE), 2),
    )


def p10():
    """農家の層別抽出（ネイマン配分）"""
    N = [3000, 1500, 500]
    S = [20, 50, 150]
    n = 300
    NS = [a * b for a, b in zip(N, S)]
    nh = [n * x / sum(NS) for x in NS]
    prop = [n * a / sum(N) for a in N]
    return dict(
        N1=N[0], N2=N[1], N3=N[2], S1=S[0], S2=S[1], S3=S[2], n=n,
        NS1=NS[0], NS2=NS[1], NS3=NS[2], NSsum=sum(NS), Nsum=sum(N),
        n3=round(nh[2]), n3raw=f(nh[2], 2), n1raw=f(nh[0], 2), n2raw=f(nh[1], 2),
        prop3=f(prop[2], 0), n3S=f(n * S[2] / sum(S), 2),
    )


def p11():
    """大雨の判別（マハラノビス距離と誤判別率）"""
    import numpy as np
    m1 = np.array([85.0, 6.0])
    m2 = np.array([70.0, 2.0])
    Sig = np.array([[100.0, 12.0], [12.0, 4.0]])
    d = m1 - m2
    det = Sig[0, 0] * Sig[1, 1] - Sig[0, 1] ** 2
    D2 = float(d @ np.linalg.inv(Sig) @ d)
    D2_diag = float(d[0] ** 2 / Sig[0, 0] + d[1] ** 2 / Sig[1, 1])
    D2_sign = float((Sig[1, 1] * d[0] ** 2 + 2 * Sig[0, 1] * d[0] * d[1] + Sig[0, 0] * d[1] ** 2) / det)
    up = lambda z: round(1 - phi(z), 4)
    zc = round(math.sqrt(D2) / 2, 2)
    zd = round(math.sqrt(D2_diag) / 2, 2)
    zs = round(math.sqrt(D2_sign) / 2, 2)
    zf = round(math.sqrt(D2), 2)
    zh = round(D2 / 2, 2)
    table = sorted({zc, zd, zs, zf, zh})
    return dict(
        x1a=f(m1[0], 0), x2a=f(m1[1], 0), x1b=f(m2[0], 0), x2b=f(m2[1], 0),
        s11=f(Sig[0, 0], 0), s12=f(Sig[0, 1], 0), s22=f(Sig[1, 1], 0), det=f(det, 0),
        d1=f(d[0], 0), d2=f(d[1], 0),
        num=f(Sig[1, 1] * d[0] ** 2 - 2 * Sig[0, 1] * d[0] * d[1] + Sig[0, 0] * d[1] ** 2, 0),
        q1=f(Sig[1, 1] * d[0] ** 2, 0), q2=f(2 * Sig[0, 1] * d[0] * d[1], 0), q3=f(Sig[0, 0] * d[1] ** 2, 0),
        D2=f(D2, 2), D2_4=f(D2, 4), D=f(math.sqrt(D2), 4), zc=f(zc, 2), err=f(up(zc), 3), err4=f(up(zc), 4),
        D2_diag=f(D2_diag, 2), err_diag=f(up(zd), 3), D2_sign=f(D2_sign, 2), err_sign=f(up(zs), 3),
        err_full=f(up(zf), 3), err_half=f(up(zh), 3), zf=f(zf, 2), zh=f(zh, 2), zd=f(zd, 2), zs=f(zs, 2),
        table_z=" & ".join(f(z, 2) for z in table),
        table_p=" & ".join(f(up(z), 4) for z in table),
        table_cols="c|" + "c" * len(table),
    )


def p12():
    """観測地点のクラスター分析（最長距離法）"""
    names = "ABCDE"
    D = {("A", "B"): 2.0, ("A", "C"): 6.0, ("A", "D"): 10.0, ("A", "E"): 9.0,
         ("B", "C"): 5.0, ("B", "D"): 8.0, ("B", "E"): 7.5,
         ("C", "D"): 4.0, ("C", "E"): 3.0, ("D", "E"): 3.5}
    dist = lambda u, v: D.get((u, v), D.get((v, u)))

    def run(link):
        cl = [frozenset(x) for x in names]
        hs = []
        while len(cl) > 1:
            best = None
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    ds = [dist(u, v) for u in cl[i] for v in cl[j]]
                    val = link(ds)
                    if best is None or val < best[0]:
                        best = (val, i, j)
            val, i, j = best
            hs.append(val)
            new = cl[i] | cl[j]
            cl = [c for k, c in enumerate(cl) if k not in (i, j)] + [new]
        return hs
    hc = run(max)
    hsng = run(min)
    havg = run(lambda ds: sum(ds) / len(ds))
    out = {f"d{u}{v}": f(x, 1) for (u, v), x in D.items()}
    out.update(last=f(hc[-1], 1), h1=f(hc[0], 1), h2=f(hc[1], 1), h3=f(hc[2], 1),
               d_single=f(hsng[-1], 1), d_avg=f(havg[-1], 2), d_third=f(hc[2], 1), d_noD=f(9.0, 1))
    return out


def p13():
    """学力テストの1因子モデル"""
    L = [0.8, 0.7, 0.6, 0.5]
    return dict(
        l1=f(L[0], 1), l2=f(L[1], 1), l3=f(L[2], 1), l4=f(L[3], 1),
        h1=f(L[0] ** 2, 2), u2=f(1 - L[1] ** 2, 2), h2=f(L[1] ** 2, 2),
        r13=f(L[0] * L[2], 2), d_r13=f((L[0] + L[2]) / 2, 2),
        contrib=f(sum(x ** 2 for x in L) / 4, 3), sumsq=f(sum(x ** 2 for x in L), 2), d_contrib=f(sum(L) / 4, 2),
        u4=f(1 - L[3] ** 2, 2), d_u4=f(1 - L[3], 2),
    )


def p14():
    """講習前後の合否（マクネマー検定）"""
    a, b, c, d = 35, 6, 19, 20
    n = a + b + c + d
    chi = (c - b) ** 2 / (b + c)
    chi_cc = (abs(c - b) - 1) ** 2 / (b + c)
    chi_n = (c - b) ** 2 / n
    z = abs(c - b) / math.sqrt(b + c)
    # 対応を無視した独立性の検定（行＝講習前、列＝講習後）
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    exp = [r1 * c1 / n, r1 * c2 / n, r2 * c1 / n, r2 * c2 / n]
    o = [a, b, c, d]
    chi_ind = sum((x - y) ** 2 / y for x, y in zip(o, exp))
    return dict(
        a=a, b=b, c=c, d=d, n=n, bc=b + c, diff=c - b, diff2=(c - b) ** 2,
        chi=f(chi, 2), chi_cc=f(chi_cc, 2), chi_n=f(chi_n, 2), z=f(z, 2), chi_ind=f(chi_ind, 2),
        pre=a + b, post=a + c,
    )


def values():
    return {
        "s03p01": p01(), "s03p02": p02(), "s03p03": p03(), "s03p04": p04(), "s03p05": p05(),
        "s03p06": p06(), "s03p07": p07(), "s03p08": p08(), "s03p09": p09(), "s03p10": p10(),
        "s03p11": p11(), "s03p12": p12(), "s03p13": p13(), "s03p14": p14(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
