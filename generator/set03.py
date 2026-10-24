"""模擬試験 第3回 の数値。values() が {大問id: {名前: 値}} を返す。

方針：問題文に示した値（分布表の値・指数関数の値など）を丸めた形で使い、
受験者が一般電卓で計算したときと同じ数値になるようにする。
"""
import math
from fractions import Fraction as Fr

import numpy as np
from scipy import stats


def f(x, d):
    """小数 d 桁の文字列（四捨五入）。"""
    return f"{x:.{d}f}"


def phi(z):
    return float(stats.norm.cdf(z))


def p01():
    """2種類の土壌の区域と隣り合う2枚の圃場（混合正規・ベイズの定理・相関）"""
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
    pA = PhiA
    pB = 1 - PhiB
    jA, jB = wA * pA, wB * pB
    tot = jA + jB
    post = jA / tot
    bA, bB = wA * pA ** 2, wB * pB ** 2
    both = bA + bB
    cond = both / tot
    corr = ve / var
    return dict(
        wA=wA, wB=wB, mA=int(mA), mB=int(mB), sA=int(sA), sB=int(sB), c=int(c),
        mean=f(mean, 1), ev=f(ev, 1), ve=f(ve, 2), var_s=f(var, 2),
        sd=f(math.sqrt(var), 2), sA2=int(sA ** 2), sB2=int(sB ** 2), diff=int(mA - mB),
        zA=f(-zA, 2), zB=f(zB, 2), zN=f(zN, 2), PhiA=f(PhiA, 4), PhiB=f(PhiB, 4), PhiN=f(PhiN, 4),
        pA=f(pA, 4), pB=f(pB, 4), jA=f(jA, 5), jB=f(jB, 5), tot5=f(tot, 5), tot=f(tot, 3),
        d_norm=f(1 - PhiN, 3), d_avg=f((pA + pB) / 2, 3),
        d_signA=f(wA * (1 - PhiA) + wB * pB, 3), d_signB=f(wA * pA + wB * PhiB, 3),
        post=f(post, 3), post4=f(post, 4),
        d_lik=f(pA, 3), d_prior=f(wA, 3), d_joint=f(jA, 3), d_postB=f(1 - post, 3),
        corr=round(corr, 3), corr4=f(corr, 4), d_corr_ev=f(ve / ev, 3),
        pA2=f(pA ** 2, 5), pB2=f(pB ** 2, 5), bA=f(bA, 5), bB=f(bB, 5), both5=f(both, 5),
        cond=f(cond, 3), cond4=f(cond, 4), d_ind=f(tot, 3), d_both=f(both, 3), d_sq=f(tot ** 2, 3),
    )


def p02():
    """灌漑ポンプの寿命（ハザード・カプラン＝マイヤー）"""
    a = 0.08                       # h(t) = a t
    t1, t2 = 3, 5
    H = lambda t: a * t ** 2 / 2
    e = lambda x: round(math.exp(-x), 4)
    cond = e(H(t2) - H(t1))
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
    """交差点のニアミス件数（ポアソン・適合度検定・オフセット付きポアソン回帰）"""
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
    chi_tail = sum((o - x) ** 2 / x for o, x in zip(O, [E0, E1, E2, E3]))
    E_ge4 = n - (E0 + E1 + E2 + E3)
    chi5 = sum((o - x) ** 2 / x for o, x in zip([obs[k] for k in range(5)], [E0, E1, E2, E3, E_ge4]))
    terms = [(o - x) ** 2 / x for o, x in zip(O, Ec)]
    # 2つの交差点の比較（条件付き二項による検定）
    tA, tB, xB = n, 90, 99
    N2 = tot + xB
    p0 = Fr(tA, tA + tB)
    EA = N2 * p0
    VA = N2 * p0 * (1 - p0)
    zc = (tot - EA) / math.sqrt(VA)
    z_half = (tot - N2 / 2) / math.sqrt(N2 / 4)
    z_np = (tot - EA) / math.sqrt(EA)
    pswap = Fr(tB, tA + tB)
    z_swap = (tot - N2 * pswap) / math.sqrt(N2 * pswap * (1 - pswap))
    z_noN = (tot / tA - xB / tB) / math.sqrt(tot / tA + xB / tB)
    # ポアソン回帰
    b0, b1, T = -1.90, -0.45, 8
    e235, e190, e145 = (round(math.exp(-v), 4) for v in (2.35, 1.90, 1.45))
    ln8 = round(math.log(T), 3)
    return dict(
        n=n, tot=tot, lam=f(lam, 1), e12=f(e, 4), o0=obs[0], o1=obs[1], o2=obs[2], o3=obs[3], o4=obs[4], o3p=O[3],
        p0=f(p[0], 4), p1=f(p[1], 4), p2=f(p[2], 4),
        E0=f(E0, 2), E1=f(E1, 2), E2=f(E2, 2), Ege3=f(E_ge3, 2), Ege4=f(E_ge4, 2),
        E2ans=round(E2, 1), E2_3=f(E2, 3),
        t0=f(terms[0], 4), t1=f(terms[1], 4), t2=f(terms[2], 4), t3=f(terms[3], 4),
        chi=f(chi, 3), chi_tail=f(chi_tail, 3), chi5=f(chi5, 3),
        b0=f(b0, 2), b1=f(b1, 2), T=T, eta=f(b0 + b1, 2), e235=f(e235, 4), e190=f(e190, 4), e145=f(e145, 4), ln8=f(ln8, 3),
        mu=f(T * e235, 3), d_nooff=f(e235, 3), d_nolane=f(T * e190, 3), d_sign=f(T * e145, 3), d_log=f(ln8 * e235, 3),
        rr=f(round(math.exp(b1), 3), 3),
        tB=tB, xB=xB, N2=N2, p0f=f"\\frac{{{p0.numerator}}}{{{p0.denominator}}}", p0d=f(float(p0), 4),
        EA=f(float(EA), 2), VA=f(float(VA), 2), sdA=f(math.sqrt(VA), 3), zc=f(zc, 2), d_half=f(z_half, 2),
        d_np=f(z_np, 2), d_swap=f(z_swap, 2), d_noN=f(z_noN, 2), rB=f(xB / tB, 1), tAB=tA + tB,
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
    q80u, q80l = round(stats.chi2.ppf(0.975, 80), 2), round(stats.chi2.ppf(0.025, 80), 2)
    q40u, q40l = round(stats.chi2.ppf(0.975, 40), 2), round(stats.chi2.ppf(0.025, 40), 2)
    ci = (2 * S / q80u, 2 * S / q80l)
    ci40 = (2 * S / q40u, 2 * S / q40l)
    ciS = (S / q80u, S / q80l)
    ciN = (xbar - 1.96 * xbar / math.sqrt(n), xbar + 1.96 * xbar / math.sqrt(n))
    iv = lambda c: f"({c[0]:.2f},\\ {c[1]:.2f})"
    return dict(
        n=n, S=int(S), xbar=f(xbar, 1), t=t,
        e43=f(e43, 4), e075=f(e075, 4), e13=f(e13, 4),
        pmle=f(e43, 3), d_cdf=f(1 - e43, 3), d_rate=f(e075, 3), d_unb=f(e13, 3), d_lam=f(lam_s, 3),
        se_s=f(lam_s / math.sqrt(n), 4), lam_m=f(lam_m, 1), sqn=f(math.sqrt(n), 4), se=round(se, 2), se4=f(se, 4),
        se_theta=f(xbar / math.sqrt(n), 3),
        q80u=f(q80u, 2), q80l=f(q80l, 2), q40u=f(q40u, 2), q40l=f(q40l, 2), twoS=int(2 * S),
        ci=iv(ci), ci40=iv(ci40), ciS=iv(ciS), ciN=iv(ciN), ci_lo=f(ci[0], 2), ci_hi=f(ci[1], 2),
        ci_half=f(1.96 * xbar / math.sqrt(n), 2),
        ciR=f"({q80l / (2 * S):.3f},\\ {q80u / (2 * S):.3f})",
    )


def p05():
    """漢字学習アプリ（検出力とサンプルサイズ、符号付き順位検定）"""
    sigma, delta = 10, 5
    za, zb, za2 = 1.645, 1.282, 1.960
    need = ((za + zb) * sigma / delta) ** 2
    need2 = ((za2 + zb) * sigma / delta) ** 2
    d = [7, -3, 12, 5, -1, 9, 4, -6, 11, 2]
    absr = sorted(abs(x) for x in d)
    rank = {v: i + 1 for i, v in enumerate(absr)}
    Wp = sum(rank[abs(x)] for x in d if x > 0)
    Wm = sum(rank[abs(x)] for x in d if x < 0)
    m = len(d)
    EW = m * (m + 1) / 4
    VW = m * (m + 1) * (2 * m + 1) / 24
    nb, p0, p1 = 15, 0.5, 0.7
    ks = [10, 11, 12, 13]
    t0 = {k: round(float(stats.binom.sf(k - 1, nb, p0)), 4) for k in ks}
    t1 = {k: round(float(stats.binom.sf(k - 1, nb, p1)), 4) for k in ks}
    c = min(k for k in ks if t0[k] <= 0.05)
    tab = {f"s0_{k}": f(t0[k], 4) for k in ks}
    tab.update({f"s1_{k}": f(t1[k], 4) for k in ks})
    return dict(
        **tab, nb=nb, c=c, size=f(t0[c], 4), pow=f(t1[c], 3), pow4=f(t1[c], 4), c1=c - 1, d_pow1=f(t1[c - 1], 3),
        d_beta=f(1 - t1[c], 3), c2=c + 1, d_pow2=f(t1[c + 1], 3), d_size=f(t0[c], 3), size_c1=f(t0[c - 1], 4),
        sigma=sigma, delta=delta, za=f(za, 3), zb=f(zb, 3), za2=f(za2, 3), zsum=f(za + zb, 3),
        need=f(need, 2), n_ans=math.ceil(need), n_floor=round(need), need2=f(need2, 2), n2=math.ceil(need2),
        need_sd=f((za + zb) * sigma / delta, 3),
        data=", ".join(f"{x:+d}" for x in d),
        Wp=Wp, Wm=Wm, EW=f(EW, 1), VW=f(VW, 2), sdW=f(math.sqrt(VW), 3),
        z=f((Wp - EW) / math.sqrt(VW), 2), z12=f((Wp - EW) / math.sqrt(2 * VW), 2),
        zv=f((Wp - EW) / VW, 2), zm=f((Wm - EW) / math.sqrt(VW), 2), tot_rank=m * (m + 1) // 2,
        negranks=" + ".join(str(rank[abs(x)]) for x in d if x < 0), neg=", ".join(str(-x) for x in d if x < 0),
        absorder=", ".join(str(v) for v in absr),
    )


def p06():
    """天気の3状態マルコフ連鎖（定常分布）"""
    P = [[Fr("0.6"), Fr("0.3"), Fr("0.1")], [Fr("0.3"), Fr("0.4"), Fr("0.3")], [Fr("0.2"), Fr("0.4"), Fr("0.4")]]
    # π(P−I)=0 の第1・第3列と和=1 を、πS=1.6πR, πC=(22/15)πR の形に解く
    piR = 1 / (Fr(8, 5) + Fr(22, 15) + 1)
    piS, piC = Fr(8, 5) * piR, Fr(22, 15) * piR
    two = sum(P[0][k] * P[k][2] for k in range(3))
    paths = [P[0][k] * P[k][2] for k in range(3)]
    colmean = sum(P[k][2] for k in range(3)) / 3
    fr = lambda q: f"\\frac{{{q.numerator}}}{{{q.denominator}}}"
    return dict(
        piS=f(float(piS), 3), piC=f(float(piC), 3), piR=f(float(piR), 3),
        piR_frac=fr(piR), piS_frac=fr(piS), piC_frac=fr(piC),
        d_col=f(float(colmean), 3), d_self=f(float(P[2][2]), 3), d_unif=f(1 / 3, 3), d_two=f(float(two), 3),
        pSS=f(float(paths[0]), 2), pSC=f(float(paths[1]), 2), pSR=f(float(paths[2]), 2), two2=f(float(two), 2),
        cC=f(float(paths[1] / two), 3), cC4=f(float(paths[1] / two), 4), d_pathS=f(float(paths[0] / two), 3),
        d_joint=f(float(paths[1]), 3), d_trans=f(float(P[0][1]), 3),
    )


def p07():
    """小麦の収量の重回帰（自由度調整済み決定係数・クックの距離・除外変数バイアス）"""
    n, p, R2 = 30, 2, 0.62
    adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    r, h = 2.4, 0.28
    D = r ** 2 / (p + 1) * h / (1 - h)
    b1, b2, g = 1.8, 0.05, -4.0
    xm, Sxx, xi = 10.0, 250.0, 22.0
    lev = 1 / n + (xi - xm) ** 2 / Sxx
    return dict(
        xm=f(xm, 1), Sxx=f(Sxx, 1), xi=f(xi, 1), dev2=f((xi - xm) ** 2, 1), inv_n=f(1 / n, 4), frac=f((xi - xm) ** 2 / Sxx, 4),
        lev=round(lev, 3), lev4=f(lev, 4), thr=f(2 * 2 / n, 3), avg=f(2 / n, 3), d_noinv=f((xi - xm) ** 2 / Sxx, 3),
        n=n, p=p, R2=f(R2, 2), adj=f(adj, 3),
        d_np=f(1 - (1 - R2) * (n - 1) / (n - p), 3), d_n=f(1 - (1 - R2) * n / (n - p - 1), 3),
        d_r2=f(R2, 3), d_inv=f(1 - (1 - R2) * (n - p - 1) / (n - 1), 3),
        ratio=f((n - 1) / (n - p - 1), 4), one_r2=f(1 - R2, 2), dfE=n - p - 1, dfT=n - 1,
        r=f(r, 1), h=f(h, 2), r2=f(r ** 2, 2), pp1=p + 1, h1h=f(h / (1 - h), 4), D=round(D, 2), D4=f(D, 4),
        one_h=f(1 - h, 2), r2p=f(r ** 2 / (p + 1), 2), d_p=f(r ** 2 / p * h / (1 - h), 2),
        b1=f(b1, 2), b2=f(b2, 2), g=f(g, 1), b2g=f(b2 * g, 2), simple=f(b1 + b2 * g, 2),
        d_sign=f(b1 - b2 * g, 2), d_add=f(b1 + b2, 2), d_same=f(b1, 2),
    )


def p08():
    """資格試験の合否のロジスティック回帰：AIC と BIC"""
    n = 200
    m2ll = {"M1": 236.8, "M2": 232.0, "M3": 229.9}
    k = {"M1": 2, "M2": 3, "M3": 4}
    ln200 = round(math.log(n), 3)
    aic = {m: m2ll[m] + 2 * k[m] for m in m2ll}
    bic = {m: m2ll[m] + ln200 * k[m] for m in m2ll}
    out = dict(n=n, ln200=f(ln200, 3), best_aic=min(aic, key=aic.get), best_bic=min(bic, key=bic.get))
    for mm in m2ll:
        out[f"m2ll_{mm}"] = f(m2ll[mm], 1)
        out[f"k_{mm}"] = k[mm]
        out[f"aic_{mm}"] = f(aic[mm], 1)
        out[f"bic_{mm}"] = f(bic[mm], 2)
        out[f"pen_{mm}"] = f(ln200 * k[mm], 3)
        out[f"bic10_{mm}"] = f(m2ll[mm] + round(math.log10(n), 3) * k[mm], 2)
    return out


def p09():
    """試験場のトマト：新栽培法の平均収量（正規・正規モデル）と肥料の乱塊法"""
    m0, t0, n, yb, sig = 2.0, 0.3, 16, 2.6, 0.8
    prec0 = 1 / t0 ** 2
    precd = n / sig ** 2
    precp = prec0 + precd
    mp = (prec0 * m0 + precd * yb) / precp
    sp_ = math.sqrt(1 / precp)
    iv = lambda c, s: f"({c - 1.96 * s:.2f},\\ {c + 1.96 * s:.2f})"
    mw = (precd * m0 + prec0 * yb) / precp                 # 重みの取り違え
    s_add = math.sqrt(t0 ** 2 + sig ** 2 / n)              # 分散を足した
    precs = prec0 + 1 / sig ** 2                           # σ/√n でなく σ を使った
    ms = (prec0 * m0 + yb / sig ** 2) / precs
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
    return dict(
        m0=f(m0, 1), t0=f(t0, 1), ns=n, yb=f(yb, 1), sig=f(sig, 1), t02=f(t0 ** 2, 2), sn2=f(sig ** 2 / n, 2),
        prec0=f(prec0, 3), precd=f(precd, 1), precp=f(precp, 3), mp=f(mp, 4), vp=f(1 / precp, 5), spp=f(sp_, 4),
        half=f(1.96 * sp_, 4), cri=iv(mp, sp_), d_data=iv(yb, sig / math.sqrt(n)), d_w=iv(mw, sp_),
        d_add=iv(mp, s_add), d_sig=iv(ms, math.sqrt(1 / precs)), wd=f(precd / precp, 4), sdn=f(sig / math.sqrt(n), 1),
        a=a, bk=bk, m1=f(means[0], 1), m2=f(means[1], 1), m3=f(means[2], 1), m4=f(means[3], 1),
        gm=f(gm, 1), dev=" + ".join(f"({m - gm:+.1f})^2".replace("+", "") for m in means),
        ssdev=f(sum((m - gm) ** 2 for m in means), 2),
        SA=f(SA, 1), ST=f(ST, 1), SB=f(SB, 1), SE=f(SE, 1), dfA=dfA, dfB=dfB, dfE=dfE,
        MSA=f(MSA, 2), MSE=f(MSE, 4), F=round(F, 2), F4=f(F, 4),
        SE1=f(SE1, 1), dfE1=dfE1, F1=f(MSA / (SE1 / dfE1), 2), f05=f(stats.f.ppf(0.95, dfA, dfE), 2),
    )


def p10():
    """農家の収穫量の比推定"""
    N, X = 400, 3200.0
    n, xb, yb = 20, 7.5, 54.0
    Xb = X / N
    R = yb / xb
    yr = R * Xb
    return dict(N=N, X=int(X), n=n, xb=f(xb, 1), yb=f(yb, 1), Xb=f(Xb, 1), R=f(R, 1), yr=round(yr, 1),
                yr_s=f(yr, 1), tot=f(yr * N, 0), d_simple=f(yb, 1))


def p11():
    """大雨警報の混同行列（適合率・再現率・F値）"""
    TP, FP, FN, TN = 18, 12, 6, 329
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    F1 = 2 * prec * rec / (prec + rec)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, N=TP + FP + FN + TN, PP=TP + FP, AP=TP + FN, AN=FP + TN, PN=FN + TN,
                prec=f(prec, 3), rec=f(rec, 3), F1=f(F1, 3), F14=f(F1, 4), d_am=f((prec + rec) / 2, 3),
                acc=f((TP + TN) / (TP + FP + FN + TN), 3), d_prod=f(prec * rec, 3), spec=f(TN / (FP + TN), 3),
                psum=f(prec + rec, 3), pprod2=f(2 * prec * rec, 3))


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
                    val = link([dist(u, v) for u in cl[i] for v in cl[j]])
                    if best is None or val < best[0]:
                        best = (val, i, j)
            val, i, j = best
            hs.append(val)
            new = cl[i] | cl[j]
            cl = [c for k, c in enumerate(cl) if k not in (i, j)] + [new]
        return hs
    hc = run(max)
    out = {f"d{u}{v}": f(x, 1) for (u, v), x in D.items()}
    out.update(last=f(hc[-1], 1), h1=f(hc[0], 1), h2=f(hc[1], 1), h3=f(hc[2], 1),
               d_single=f(run(min)[-1], 1), d_avg=f(run(lambda ds: sum(ds) / len(ds))[-1], 2), d_third=f(hc[2], 1),
               d_noD=f(max(dist(u, v) for u in "AB" for v in "CE"), 1))
    return out


def p13():
    """学力テストの1因子モデル"""
    L = [0.8, 0.7, 0.6, 0.5]
    return dict(
        l1=f(L[0], 1), l2=f(L[1], 1), l3=f(L[2], 1), l4=f(L[3], 1),
        h1=f(L[0] ** 2, 2), h2=f(L[1] ** 2, 2), u2=f(1 - L[1] ** 2, 2), d_u2=f(1 - L[1], 2),
        r13=f(L[0] * L[2], 2), r12=f(L[0] * L[1], 2),
        contrib=f(sum(x ** 2 for x in L) / 4, 3), sumsq=f(sum(x ** 2 for x in L), 2), d_contrib=f(sum(L) / 4, 2),
    )


def p14():
    """自転車講習の受講と合格（シンプソンのパラドックス）"""
    S = {"1": (18, 20, 56, 70), "2": (42, 80, 9, 20)}   # 受講の合格者・人数, 未受講の合格者・人数
    out = {}
    for k, (a, na, b, nb) in S.items():
        out.update({f"ta{k}": a, f"na{k}": na, f"tb{k}": b, f"nb{k}": nb, f"ra{k}": f(a / na, 3), f"rb{k}": f(b / nb, 3)})
    A = sum(v[0] for v in S.values()); NA = sum(v[1] for v in S.values())
    B = sum(v[2] for v in S.values()); NB = sum(v[3] for v in S.values())
    out.update(TA=A, NA=NA, TB=B, NB=NB, RA=f(A / NA, 3), RB=f(B / NB, 3),
               sa1=f(S["1"][1] / NA, 2), sa2=f(S["2"][1] / NA, 2), sb1=f(S["1"][3] / NB, 3), sb2=f(S["2"][3] / NB, 3))
    return out


def values():
    return {
        "s03p01": p01(), "s03p02": p02(), "s03p03": p03(), "s03p04": p04(), "s03p05": p05(),
        "s03p06": p06(), "s03p07": p07(), "s03p08": p08(), "s03p09": p09(), "s03p10": p10(),
        "s03p11": p11(), "s03p12": p12(), "s03p13": p13(), "s03p14": p14(),
    }


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
