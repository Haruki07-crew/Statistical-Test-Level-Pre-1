"""模擬試験 第8回（set08）の数値。values() が {大問id: {名前: 値}} を返す。

題材：Web・IT 運用・通信・ソフトウェア品質。
問題文に示した関数値（Φ・χ²・e^x・対数など）は、ここでも同じ丸めた値を使って答えを計算する。
"""
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from math import comb, sqrt


def r(x, d):
    """四捨五入（銀行丸めを避ける）。数値で返す。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    return float(v) if d > 0 else int(v)


def s(x, d):
    """四捨五入した文字列（末尾の 0 も残す。−0 は 0 にする）。"""
    q = Decimal(1).scaleb(-d)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return str(v)


def distinct(*xs):
    assert len(set(xs)) == len(xs), xs


def p01():
    # (a) 誕生日問題型のハッシュ衝突（指数近似）
    N, k = 1000, 20
    pairs = k * (k - 1) // 2
    ex = {0.19: 0.8270, 0.20: 0.8187, 0.38: 0.6839, 0.02: 0.9802}
    lam = pairs / N
    assert abs(lam - 0.19) < 1e-12
    ok = 1 - ex[0.19]
    opts = [s(k / N, 3), s(ok, 3), s(1 - ex[0.20], 3), s(lam, 3), s(1 - ex[0.38], 3)]
    distinct(*opts)
    # (b) 5 台中 3 台目の応答（一様分布の順序統計量）
    n, kq, t = 5, 3, 0.4
    terms = [comb(n, j) * t ** j * (1 - t) ** (n - j) for j in range(kq, n + 1)]
    q = sum(terms)
    return {
        "N": N, "k": k, "pairs": pairs, "lam": s(lam, 2), "e019": s(ex[0.19], 4), "e020": s(ex[0.20], 4),
        "e038": s(ex[0.38], 4), "e002": s(ex[0.02], 4),
        "o_kn": opts[0], "o_ok": opts[1], "o_sq": opts[2], "o_union": opts[3], "o_noh": opts[4],
        "n": n, "kq": kq, "t": t, "q3": s(terms[0], 5), "q4": s(terms[1], 5), "q5": s(terms[2], 5),
        "quo": r(q, 3), "quo_s": s(q, 3), "quo5": s(q, 5), "w_one": s(1 - (1 - t) ** n, 3), "w_all": s(t ** n, 3),
        "w_eq": s(terms[0], 3), "w_ge4": s(1 - sum(comb(n, j) * t ** j * (1 - t) ** (n - j) for j in range(0, 4)), 3),
    }


def p02():
    # (a) 指数分布に従う遅延の比 R=T1/T2 の密度
    l1, l2 = 0.5, 0.25
    # (b) M/M/1/K の呼損
    lm, mu, K = 3, 4, 3
    rho = lm / mu
    block = rho ** K * (1 - rho) / (1 - rho ** (K + 1))
    bopts = [s(rho ** K * (1 - rho), 3), s(block, 3), s(rho ** K * (1 - rho) / (1 - rho ** K), 3), s(1 / (K + 1), 3), s(rho ** K, 3)]
    distinct(*bopts)
    probs = [rho ** j * (1 - rho) / (1 - rho ** (K + 1)) for j in range(K + 1)]
    # (c) 偶数回の切り替え
    lf = 0.8
    ev = {0.8: 0.4493, 1.6: 0.2019}
    even = (1 + ev[1.6]) / 2
    copts = [s(ev[0.8], 3), s(even, 3), s(0.5, 3), s((1 - ev[1.6]) / 2, 3), s((1 + ev[0.8]) / 2, 3)]
    distinct(*copts)
    return {
        "l1": l1, "l2": l2, "m1": s(1 / l1, 0), "m2": s(1 / l2, 0), "lp": s(l1 * l2, 3),
        "lm": lm, "mu": mu, "K": K, "K1": K + 1, "rho": s(rho, 2), "rhoK": s(rho ** K, 6), "rhoK1": s(rho ** (K + 1), 8),
        "den": s(1 - rho ** (K + 1), 8), "num": s(rho ** K * (1 - rho), 8), "block4": s(block, 4),
        "pr0": s(probs[0], 4), "pr1": s(probs[1], 4), "pr2": s(probs[2], 4), "pr3": s(probs[3], 4),
        "thru": s(lm * (1 - block), 3),
        "b_inf": bopts[0], "b_ok": bopts[1], "b_kden": bopts[2], "b_uni": bopts[3], "b_rk": bopts[4],
        "lf": lf, "e08": s(ev[0.8], 4), "e16": s(ev[1.6], 4), "twol": s(2 * lf, 1),
        "c_zero": copts[0], "c_ok": copts[1], "c_half": copts[2], "c_odd": copts[3], "c_wrong": copts[4],
        "even4": s(even, 4),
    }


def p03():
    s1, s2, rho = 4.0, 3.0, 0.5
    cov = rho * s1 * s2
    den = s1 ** 2 + s2 ** 2 - 2 * cov
    w = (s2 ** 2 - cov) / den
    vmin = (s1 ** 2 * s2 ** 2 - cov ** 2) / den
    w0 = s2 ** 2 / (s1 ** 2 + s2 ** 2)
    v0 = s1 ** 2 * s2 ** 2 / (s1 ** 2 + s2 ** 2)
    return {
        "s1": s(s1, 0), "s2": s(s2, 0), "rho": rho, "v1": s(s1 ** 2, 0), "v2": s(s2 ** 2, 0), "cov": s(cov, 0),
        "den": s(den, 0), "w_frac": f"\\frac{{{s(s2 ** 2 - cov, 0)}}}{{{s(den, 0)}}}",
        "vnum": s(s1 ** 2 * s2 ** 2 - cov ** 2, 0), "vmin": r(vmin, 2), "vmin4": s(vmin, 4), "vmin_s": s(vmin, 2),
        "w0": s(w0, 2), "v0": s(v0, 2), "v_half": s((s1 ** 2 + s2 ** 2 + 2 * cov) / 4, 2),
        "p1v1": s(s1 ** 2 * s2 ** 2, 0), "cov2": s(cov ** 2, 0), "w1": s(1 - w, 4),
    }


def p04():
    # (a) 指数分布の標本分散の正規近似（V[(X−μ)²]=μ4−σ⁴）
    k4, n, pct = 9, 200, 20
    phi = {0.85: 0.8023, 0.94: 0.8264, 1.00: 0.8413, 2.00: 0.9772}
    d = pct / 100
    sdr = sqrt((k4 - 1) / n)
    z = d / sdr
    assert abs(z - 1.00) < 1e-12
    z_nosub = round(d / sqrt(k4 / n), 2)       # μ4 から σ⁴ を引き忘れ
    z_exc = round(d / sqrt((k4 + 3 - 1) / n), 2)  # 9 を超過尖度と読んだ
    z_norm = d / sqrt(2 / n)                   # 正規母集団
    assert (z_nosub, z_exc, z_norm) == (0.94, 0.85, 2.0)
    aopts = {"o_ok": 2 * phi[1.00] - 1, "o_exc": 2 * phi[0.85] - 1, "o_nosub": 2 * phi[0.94] - 1,
             "o_one": phi[1.00], "o_norm": 2 * phi[2.00] - 1}
    distinct(*[s(v, 3) for v in aopts.values()])
    # (b) 2 種類のデータを合わせた最尤推定
    x, tA, m, tsum = 18, 6, 5, 2.5
    mle = (x + m) / (tA + tsum)
    bopts = [s(m / tsum, 2), s((x / tA + m / tsum) / 2, 2), s(mle, 2), s(x / tA, 2), s((x + m) / tA, 2)]
    distinct(*bopts)
    # (c) 中央値の信頼区間
    nn = 10
    cdf = lambda k: sum(comb(nn, j) for j in range(k + 1))
    cov = {k: 1 - 2 * cdf(k - 1) / 2 ** nn for k in range(1, 6)}
    assert cov[2] >= 0.95 > cov[3]
    return {
        "k4": k4, "k4m1": k4 - 1, "n": n, "pct": pct, "lo": s(1 - d, 1), "hi": s(1 + d, 1), "d": s(d, 1),
        "phi085": s(phi[0.85], 4), "phi094": s(phi[0.94], 4), "phi100": s(phi[1.00], 4), "phi200": s(phi[2.00], 4),
        "sdr": s(sdr, 1), "z": s(z, 0), "z_s": s(z, 2), "p_ok4": s(aopts["o_ok"], 4), "sdn": s(sqrt(2 / n), 1),
        "sd_nosub": s(sqrt(k4 / n), 4), **{k: s(v, 3) for k, v in aopts.items()},
        "x": x, "tA": tA, "m": m, "tsum": tsum, "xm": x + m, "tt": s(tA + tsum, 1),
        "b_int": bopts[0], "b_avg": bopts[1], "b_ok": bopts[2], "b_cnt": bopts[3], "b_wrong": bopts[4], "mle4": s(mle, 4),
        "nn": nn, "c0": cdf(0), "c1": cdf(1), "c2": cdf(2), "pow": 2 ** nn,
        "cov1": s(cov[1], 4), "cov2": s(cov[2], 4), "cov3": s(cov[3], 4),
    }


def p05():
    # (b) 並べ替え検定（0.1 秒単位の整数で扱う）
    old, new = [124, 118, 131], [109, 115, 120]
    allv = old + new
    obs = sum(old)
    cnt_ge = cnt_gt = cnt_le = 0
    for c in combinations(range(6), 3):
        sm = sum(allv[i] for i in c)
        cnt_ge += sm >= obs
        cnt_gt += sm > obs
        cnt_le += sm <= sum(new)
    tot = comb(6, 3)
    pv = cnt_ge / tot
    popts = [s(cnt_gt / tot, 3), s(pv, 3), s((cnt_ge + 1) / tot, 3), s((cnt_ge + cnt_le) / tot, 3), s(cnt_ge / 720, 3)]
    distinct(*popts)
    f1 = lambda v: s(v / 10, 1)
    return {
        "o1": f1(old[0]), "o2": f1(old[1]), "o3": f1(old[2]), "n1": f1(new[0]), "n2": f1(new[1]), "n3": f1(new[2]),
        "osum": f1(obs), "tot": tot, "cge": cnt_ge,
        "p_gt": popts[0], "p_ok": popts[1], "p_extra": popts[2], "p_two": popts[3], "p_perm": popts[4],
    }


def p06():
    # (c) リッジ回帰の有効自由度
    d2, lam = [9, 4, 1], 2
    df = sum(v / (v + lam) for v in d2)
    copts = [s(sum(lam / (v + lam) for v in d2), 2), s(sum(sqrt(v) / (sqrt(v) + lam) for v in d2), 2), s(df, 2),
             s(len(d2) / (1 + lam), 2), s(len(d2), 2)]
    distinct(*copts)
    return {
        "d1": d2[0], "d2": d2[1], "d3": d2[2], "lam": lam, "f1": s(d2[0] / (d2[0] + lam), 4), "f2": s(d2[1] / (d2[1] + lam), 4),
        "f3": s(d2[2] / (d2[2] + lam), 4), "df4": s(df, 4),
        "c_comp": copts[0], "c_sv": copts[1], "c_ok": copts[2], "c_eq": copts[3], "c_ols": copts[4],
    }


def p07():
    # (b) 補対数対数リンク
    b0, b1 = -2.0, 0.5
    ev = {-2.0: 0.1353, -1.5: 0.2231, "a": 0.8735, "b": 0.8000}
    p_new = 1 - ev["b"]
    p_old = round(1 - ev["a"], 4)
    lg = 0.1824
    bopts = [s(p_old, 3), s(lg, 3), s(p_new, 3), s(ev[-1.5], 3), s(ev["b"], 3)]
    distinct(*bopts)
    return {
        "b0": s(b0, 1), "b1": s(b1, 1), "lin": s(b0 + b1, 1), "e20": s(ev[-2.0], 4), "e15": s(ev[-1.5], 4),
        "ea": s(ev["a"], 4), "eb": s(ev["b"], 4), "lg": s(lg, 4),
        "c_old": bopts[0], "c_lg": bopts[1], "c_ok": bopts[2], "c_inner": bopts[3], "c_comp": bopts[4],
        "rr": "e^{0.5}",
    }


def p08():
    # (a) 処置群の平均処置効果（ATT）
    pz1, pz = 0.7, 0.4                  # P(高|導入), P(高)
    px = 0.4
    pz0 = (pz - pz1 * px) / (1 - px)    # P(高|未導入)
    y = {(1, 1): 1.2, (0, 1): 1.8, (1, 0): 0.8, (0, 0): 1.0}
    d_hi, d_lo = y[(1, 1)] - y[(0, 1)], y[(1, 0)] - y[(0, 0)]
    att = pz1 * d_hi + (1 - pz1) * d_lo
    ate = pz * d_hi + (1 - pz) * d_lo
    e1 = pz1 * y[(1, 1)] + (1 - pz1) * y[(1, 0)]
    e0 = pz0 * y[(0, 1)] + (1 - pz0) * y[(0, 0)]
    # (b) 2^3 要因実験を 2 日（ブロック）に分ける交絡法
    runs = ["(1)", "a", "b", "ab", "c", "ac", "bc", "abc"]
    # 効果 eff の符号は、eff に含まれる因子のうち低水準のものの数が偶数なら +（標準的な ± の積）
    sign = lambda run, eff: (-1) ** sum(ch not in run for ch in eff.lower())
    effects = ["A", "B", "C", "AB", "AC", "BC", "ABC"]

    def confounded(day1):
        """1 日目の 4 条件の中で符号がすべて同じ効果（日と完全に交絡）と、符号が 2 対 2 でない効果。"""
        full = [e for e in effects if abs(sum(sign(r_, e) for r_ in day1)) == 4]
        part = [e for e in effects if abs(sum(sign(r_, e) for r_ in day1)) == 2]
        return full, part
    ok = [r_ for r_ in runs if sign(r_, "ABC") == -1]
    assert ok == ["(1)", "ab", "ac", "bc"] and confounded(ok) == (["ABC"], [])
    assert confounded(["(1)", "a", "b", "ab"]) == (["C"], [])
    assert confounded(["(1)", "ab", "c", "abc"]) == (["AB"], [])
    assert confounded(["(1)", "a", "b", "c"]) == ([], ["A", "B", "C", "ABC"])
    fmt = lambda rs: "\\{" + ",\\ ".join(rs) + "\\}"
    ks = [f"1 日目 ${fmt(['(1)', 'a', 'b', 'ab'])}$、交絡する効果は C",
          f"1 日目 ${fmt(['(1)', 'ab', 'c', 'abc'])}$、交絡する効果は AB",
          f"1 日目 ${fmt(['(1)', 'a', 'b', 'c'])}$、交絡する効果は ABC",
          f"1 日目 ${fmt(ok)}$、交絡する効果はない（日のずれは誤差に含まれる）",
          f"1 日目 ${fmt(ok)}$、交絡する効果は ABC"]
    distinct(*ks)
    return {"pz1": pz1, "qz1": s(1 - pz1, 1), "pz": pz, "qz": s(1 - pz, 1), "px": px, "pz0": s(pz0, 1),
            "y11": s(y[(1, 1)], 1), "y01": s(y[(0, 1)], 1), "y10": s(y[(1, 0)], 1), "y00": s(y[(0, 0)], 1),
            "dhi": s(d_hi, 1), "dlo": s(d_lo, 1), "att": r(att, 2), "att_s": s(att, 2), "ate": s(ate, 2),
            "e1": s(e1, 2), "e0": s(e0, 2), "naive": s(e1 - e0, 2),
            **{f"k{i}": v for i, v in enumerate(ks)}}

def p09():
    # (a) 線形 SVM
    w, b = (3, 4), -10
    nw = sqrt(w[0] ** 2 + w[1] ** 2)
    pt = (1, 2)
    f = w[0] * pt[0] + w[1] * pt[1] + b
    assert f == 1
    side = lambda v: "攻撃側（$+1$）" if v > 0 else "正常側（$-1$）"
    fmt = lambda m, v: f"マージンの幅 ${s(m, 2)}$、点 $(1,\\ 2)$ は{side(v)}"
    opts = [fmt(1 / nw, f), fmt(2 / nw, -f), fmt(2 / nw, f), fmt(2 / (w[0] + w[1]), f), fmt(2 / nw ** 2, f)]
    distinct(*opts)
    # (b) ジャカード係数と単純一致係数
    a_, b_, c_, d_ = 2, 2, 1, 5
    J = a_ / (a_ + b_ + c_)
    SM = (a_ + d_) / (a_ + b_ + c_ + d_)
    pair = lambda u, v: f"ジャカード係数 ${s(u, 2)}$、単純一致係数 ${s(v, 2)}$"
    bopts = [pair(SM, J), pair(J, a_ / (a_ + b_ + c_ + d_)), pair(J, SM), pair(a_ / (a_ + b_), SM), pair(a_ / (a_ + b_ + c_ + d_), SM)]
    distinct(*bopts)
    return {
        "w1": w[0], "w2": w[1], "bb": b, "nw": s(nw, 0), "marg": s(2 / nw, 2), "f": f,
        "o0": opts[0], "o1": opts[1], "o2": opts[2], "o3": opts[3], "o4": opts[4],
        "a": a_, "b": b_, "c": c_, "d": d_, "J": s(J, 2), "SM": s(SM, 2),
        "q0": bopts[0], "q1": bopts[1], "q2": bopts[2], "q3": bopts[3], "q4": bopts[4],
    }


def p10():
    # (a) 共分散選択（連鎖）
    rRC, rCT, rRT = 0.6, 0.5, 0.45
    fit = rRC * rCT
    pc = (rRT - rRC * rCT) / sqrt((1 - rRC ** 2) * (1 - rCT ** 2))
    opts = [s(0.0, 2), s(pc, 2), s(fit, 2), s(rRT, 2), s((rRC + rCT) / 2, 2)]
    distinct(*opts)
    return {"rRC": rRC, "rCT": rCT, "rRT": rRT, "fit": s(fit, 2), "pc": s(pc, 4),
            "o_zero": opts[0], "o_pc": opts[1], "o_ok": opts[2], "o_obs": opts[3], "o_avg": opts[4]}


def p11():
    # (a) 正確なポアソン検定
    lam0, x = 2, 6
    e2 = 0.1353
    pm = [lam0 ** j / [1, 1, 2, 6, 24, 120, 720][j] for j in range(x + 1)]
    cdf5 = e2 * sum(pm[:x])
    pv = 1 - cdf5
    p6 = e2 * pm[x]
    phi = 0.9977
    opts = [s(pv - p6, 3), s(p6, 3), s(pv, 3), s(2 * pv, 3), s(1 - phi, 3)]
    distinct(*opts)
    # (b) コックスモデル：最大点は e^β = 1/√2
    hr = 1 / sqrt(2)
    import math
    bopts = [s(0.5, 3), s(hr, 3), s(1.0, 3), s(sqrt(2), 3), s(math.log(hr), 3)]
    distinct(*bopts)
    return {"lam0": lam0, "x": x, "e2": s(e2, 4), "psum": s(sum(pm[:x]), 4), "cdf5": s(cdf5, 5), "pv5": s(pv, 5),
            "p6": s(p6, 4), "z": s((x - lam0) / sqrt(lam0), 2), "phi": s(phi, 4),
            "o_gt": opts[0], "o_eq": opts[1], "o_ok": opts[2], "o_two": opts[3], "o_norm": opts[4],
            "b_half": bopts[0], "b_ok": bopts[1], "b_one": bopts[2], "b_inv": bopts[3], "b_beta": bopts[4]}


def p12():
    # 線形合同法の最大周期（総当たりで確かめた組だけを使う）
    m = 16

    def period(a, c, x0=1):
        seen, x, k = {}, x0, 0
        while x not in seen:
            seen[x] = k
            x = (a * x + c) % m
            k += 1
        return k - seen[x]
    cand = [(7, 3), (5, 4), (3, 5), (5, 3), (9, 6)]
    per = [period(a, c) for a, c in cand]
    assert [p == m for p in per] == [False, False, False, True, False], per
    return {"m": m, "p73": per[0], "p54": per[1], "p35": per[2], "p53": per[3], "p96": per[4]}


def x05():
    # (a) ルビーン検定（平均からの絶対偏差の一元配置分散分析）
    g = [[10, 12, 11, 13], [9, 15, 12, 12], [8, 16, 10, 14]]
    k, n = len(g), sum(len(v) for v in g)

    def anova(groups):
        allv = [x for v in groups for x in v]
        gm = sum(allv) / len(allv)
        ssb = sum(len(v) * (sum(v) / len(v) - gm) ** 2 for v in groups)
        ssw = sum((x - sum(v) / len(v)) ** 2 for v in groups for x in v)
        return ssb, ssw
    z = [[abs(x - sum(v) / len(v)) for x in v] for v in g]
    ssb, ssw = anova(z)
    F = (ssb / (k - 1)) / (ssw / (n - k))
    F11 = (ssb / (k - 1)) / (ssw / (n - 1))
    rb, rw = anova(g)
    Fraw = (rb / (k - 1)) / (rw / (n - k))
    z2 = [[(x - sum(v) / len(v)) ** 2 for x in v] for v in g]
    qb, qw = anova(z2)
    Fsq = (qb / (k - 1)) / (qw / (n - k))
    var = [sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1) for v in g]
    fmax = max(var) / min(var)
    opts = [s(Fraw, 2), s(F, 2), s(F11, 2), s(Fsq, 2), s(fmax, 2)]
    distinct(*opts)
    zm = [sum(v) / len(v) for v in z]
    return {"gA": "、".join(str(x) for x in g[0]), "gB": "、".join(str(x) for x in g[1]), "gC": "、".join(str(x) for x in g[2]),
            "mA": s(sum(g[0]) / 4, 1), "mB": s(sum(g[1]) / 4, 1), "mC": s(sum(g[2]) / 4, 1),
            "zA": "、".join(s(x, 1) for x in z[0]), "zB": "、".join(s(x, 1) for x in z[1]), "zC": "、".join(s(x, 1) for x in z[2]),
            "zmA": s(zm[0], 1), "zmB": s(zm[1], 1), "zmC": s(zm[2], 1), "zg": s(sum(zm) / 3, 4),
            "ssb": s(ssb, 4), "ssw": s(ssw, 1), "msb": s(ssb / (k - 1), 4), "msw": s(ssw / (n - k), 4), "Fl": s(F, 4),
            "dfb": k - 1, "dfw": n - k, "Fc": 4.26,
            "l_raw": opts[0], "l_ok": opts[1], "l_df": opts[2], "l_sq": opts[3], "l_max": opts[4]}


def x06():
    # (b) 残差の歪度・尖度からジャック・ベラ統計量
    n, m2, m3, m4 = 60, 0.25, 0.075, 0.275
    S = m3 / m2 ** 1.5
    K = m4 / m2 ** 2
    jb = n / 6 * (S ** 2 + (K - 3) ** 2 / 4)
    return {"jn": n, "m2": m2, "m3": m3, "m4": m4, "m215": s(m2 ** 1.5, 3), "m22": s(m2 ** 2, 4), "S": s(S, 1), "K": s(K, 1),
            "S2": s(S ** 2, 2), "K32": s((K - 3) ** 2 / 4, 2), "jb": r(jb, 2), "jb_s": s(jb, 2),
            "jb_k": s(n / 6 * (S ** 2 + K ** 2 / 4), 2), "jb_nod": s(n / 6 * (S ** 2 + (K - 3) ** 2), 2), "chi2": 5.99}


def x07():
    # (a) カバレッジ誤差を含む平均二乗誤差
    wc, mc, me, n, Sd = 0.8, 30.0, 20.0, 400, 10.0
    Y = wc * mc + (1 - wc) * me
    bias = mc - Y
    var = Sd ** 2 / n
    rmse = sqrt(bias ** 2 + var)
    opts = [s(sqrt(var), 2), s(bias, 2), s(rmse, 2), s(bias + sqrt(var), 2), s(bias ** 2 + var, 2)]
    distinct(*opts)
    return {"wc": wc, "we": s(1 - wc, 1), "mc": s(mc, 0), "me": s(me, 0), "cn": n, "Sd": s(Sd, 0), "Yt": s(Y, 0),
            "bias": s(bias, 0), "var": s(var, 2), "se": s(sqrt(var), 2), "mse": s(bias ** 2 + var, 2), "rmse4": s(rmse, 4),
            "v_se": opts[0], "v_bias": opts[1], "v_ok": opts[2], "v_sum": opts[3], "v_mse": opts[4]}


def x10():
    # (b) 季節差分のあとの AR(1)
    phi = 0.4
    xs = [120, 135, 128, 126, 140, 95, 90, 133]   # 先週の月〜日、今週の月
    d = xs[7] - xs[0]
    fc = xs[1] + phi * d
    return {"sphi": phi, "w0": xs[0], "w1": xs[1], "w2": xs[2], "w3": xs[3], "w4": xs[4], "w5": xs[5], "w6": xs[6], "w7": xs[7],
            "sd": d, "sadj": s(phi * d, 1), "sfc": r(fc, 1), "sfc_s": s(fc, 1),
            "s_naive": s(xs[1], 1), "s_ar": s(xs[7] + phi * (xs[7] - xs[6]), 1), "s_rw": s(xs[7], 1), "s_full": s(xs[1] + d, 1)}


def values():
    return {"s08p01": p01(), "s08p02": p02(), "s08p03": p03(), "s08p04": p04(), "s08p05": {**p05(), **x05()},
            "s08p06": {**p06(), **x06()}, "s08p07": {**p07(), **x07()}, "s08p08": p08(), "s08p09": p09(), "s08p10": {**p10(), **x10()},
            "s08p11": p11(), "s08p12": p12()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
