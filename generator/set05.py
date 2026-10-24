"""模擬試験 第5回（set05）の数値。values() が {大問id: {名前: 値}} を返す。

題材：医療・疫学・製薬・公衆衛生。
問題文に示した分布の値（Φ・χ² の点、e^x など）は、ここでも同じ丸めた値を使って答えを計算する。
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
    """四捨五入した文字列（末尾の 0 も残す）。-0.000 は 0.000 にする。"""
    q = Decimal(1).scaleb(-d)
    out = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def p01():
    prev = 0.02
    seA, spA, seB, spB = 0.90, 0.95, 0.80, 0.98
    # (a) 直列検査（両方陽性で陽性）。疾患の有無を与えたとき2つの検査は条件付き独立
    sens = seA * seB
    fp = (1 - spA) * (1 - spB)
    num = prev * sens
    fpm = (1 - prev) * fp
    ppv = num / (num + fpm)
    ppvA = prev * seA / (prev * seA + (1 - prev) * (1 - spA))
    sens_par = 1 - (1 - seA) * (1 - seB)
    fp_par = 1 - spA * spB
    ppv_par = prev * sens_par / (prev * sens_par + (1 - prev) * fp_par)
    ppv_fpw = num / (num + (1 - prev) * fp_par)
    ppv_noprev = sens / (sens + fp)
    # (b) プール検査（ドーフマン法）。検査は誤りなし
    k, N = 10, 1000
    q10 = 0.8171  # 0.98^10（問題文に示す値）
    pools = N // k
    ppos = 1 - q10
    tests = pools * (1 + k * ppos)
    # (c) チェビシェフの不等式による例数
    eps, alpha = 0.02, 0.10
    n_cheb = 1 / (4 * eps ** 2 * alpha)
    z = 1.645
    n_clt = math.ceil(z ** 2 / (4 * eps ** 2))
    return {
        "prev": prev, "seA": seA, "spA": spA, "seB": seB, "spB": spB,
        "sens": s(sens, 2), "fpA": s(1 - spA, 2), "fpB": s(1 - spB, 2), "fp": s(fp, 3),
        "num": s(num, 4), "fpm": s(fpm, 5), "den": s(num + fpm, 5), "ppv": s(ppv, 3), "ppv4": s(ppv, 4),
        "ppvA": s(ppvA, 3), "ppv_par": s(ppv_par, 3), "ppv_fpw": s(ppv_fpw, 3), "ppv_noprev": s(ppv_noprev, 3),
        "sens_par": s(sens_par, 2), "fp_par": s(fp_par, 3), "npv_spec": s(1 - fp, 3),
        "k": k, "N": N, "q10": q10, "pools": pools, "ppos": s(ppos, 4), "kppos": s(k * ppos, 3),
        "tests": r(tests, 1), "tests_s": s(tests, 1), "per": s(tests / N, 4),
        "tests_wrong": s(pools * (1 + k * prev), 1),
        "eps": eps, "alpha": alpha, "eps2": s(eps ** 2, 4), "ncheb": s(n_cheb, 0),
        "nclt": n_clt, "nclt_raw": s(z ** 2 / (4 * eps ** 2), 2), "z": z,
        "n_nobound": s(1 / (eps ** 2 * alpha), 0), "n_09": math.ceil(1 / (4 * eps ** 2 * (1 - alpha))),
        "n_eps1": s(1 / (4 * eps * alpha), 0),
    }


def p02():
    mu, sig, rho = 128, 20, 0.6
    x1, c = 168, 140
    # (a) ベースラインと変化量の相関
    corr = -sqrt((1 - rho) / 2)
    # (b) 条件付き分布
    cm = mu + rho * (x1 - mu)
    cv = sig ** 2 * (1 - rho ** 2)
    csd = sqrt(cv)
    phi = {0.60: 0.7257, 0.75: 0.7734, 1.00: 0.8413, 1.75: 0.9599}
    z_ok = r((cm - c) / csd, 2)          # P(X2<c) = 1-Φ(z_ok)
    z_ns = r((cm - c) / sig, 2)          # 分散を縮めない
    z_marg = r((c - mu) / sig, 2)        # 周辺分布
    z_sd = r((cm - c) / (sig * rho), 2)  # 条件付き標準偏差を σρ とした
    z_nomean = r((x1 - c) / csd, 2)      # 平均を引き戻さない
    for z in (z_ok, z_ns, z_marg, z_sd, z_nomean):
        assert z in phi, z
    return {
        "mu": mu, "sig": sig, "rho": rho, "x1": x1, "c": c,
        "var": sig ** 2, "cov": s(rho * sig ** 2, 0), "rho1": s(1 - rho, 1),
        "corr": s(corr, 3), "corr_w0": "0", "corr_w1": s(-(1 - rho) / 2, 3), "corr_w2": s(-sqrt(1 - rho), 3),
        "corr_w3": s(-corr, 3), "vd": s(2 * (1 - rho) * sig ** 2, 0), "cxd": s((rho - 1) * sig ** 2, 0),
        "half": s((1 - rho) / 2, 1),
        "dev": x1 - mu, "shift": s(rho * (x1 - mu), 0), "cm": s(cm, 0), "rho2": s(rho ** 2, 2),
        "cv": s(cv, 0), "csd": s(csd, 0), "zok": s(z_ok, 2),
        "fA": s(phi[0.60], 4), "fB": s(phi[0.75], 4), "fC": s(phi[1.00], 4), "fD": s(phi[1.75], 4),
        "p_ok": s(1 - phi[z_ok], 3), "p_ok4": s(1 - phi[z_ok], 4), "p_ns": s(1 - phi[z_ns], 3),
        "p_marg": s(phi[z_marg], 3), "p_sd": s(1 - phi[z_sd], 3), "p_nomean": s(1 - phi[z_nomean], 3),
        "z_ns": s(z_ns, 2), "z_sd": s(z_sd, 2), "z_nomean": s(z_nomean, 2), "z_marg": s(z_marg, 2),
        "sdrho": s(sig * rho, 0),
    }


def p03():
    m1, m2 = 5, 2.5
    l1, l2 = 1 / m1, 1 / m2
    t = 10
    e2, e4 = 0.1353, 0.0183  # e^{-2}, e^{-4}（問題文に示す値）
    a = l2 / (l2 - l1)
    b = l1 / (l2 - l1)
    surv = a * e2 - b * e4
    # (b) カプラン・マイヤー
    km = 7 / 8 * 4 / 6 * 2 / 3
    km_cens_fail = 2 / 8
    km_drop = 1 / 5
    km_alive = 4 / 8
    km_risk = 7 / 8 * 5 / 7 * 3 / 4
    return {
        "m1": m1, "m2": m2, "l1": s(l1, 1), "l2": s(l2, 1), "t": t, "e2": e2, "e4": e4,
        "dl": s(l2 - l1, 1), "a": s(a, 0), "b": s(b, 0), "l1t": s(l1 * t, 0), "l2t": s(l2 * t, 0),
        "l12": s(l1 * l2, 2), "ae2": s(a * e2, 4),
        "surv": r(surv, 3), "surv4": s(surv, 4), "prod": s(e2 * e4, 4), "sum_e": s(m1 + m2, 1),
        "km": s(km, 3), "km_cf": s(km_cens_fail, 3), "km_drop": s(km_drop, 3), "km_alive": s(km_alive, 3),
        "km_risk": s(km_risk, 3),
    }


def p04():
    n = 150
    root = 0.98023  # 0.05^{1/150}
    up = 1 - root
    root25 = 0.97571  # 0.025^{1/150}（解説で触れる値）
    # (b) 0件の日の割合だけから推定したときの漸近相対効率
    lam = 0.5
    e_p, e_m = 1.6487, 0.6065  # e^{0.5}, e^{-0.5}
    eff = lam / (e_p - 1)
    return {
        "n": n, "root": root, "up": r(up, 4), "up5": s(up, 5), "rule3": s(3 / n, 4),
        "root25": root25, "up25": s(1 - root25, 4), "rule3_pct": s(300 / n, 1),
        "lam": lam, "ep": e_p, "em": e_m, "ep1": s(e_p - 1, 4), "one_em": s(1 - e_m, 4),
        "eff": s(eff, 3), "eff4": s(eff, 4), "eff_inv": s((e_p - 1) / lam, 3), "eff_p0": s(e_m, 3),
        "eff_le": s(lam * e_m, 3), "eff_nog": s(lam / (e_m * (1 - e_m)), 3), "pq": s(e_m * (1 - e_m), 4),
    }


def p05():
    n = 10
    mP, sP, mD, sD = 136.0, 10.0, 131.8, 9.0
    dbar, sd = -4.2, 5.5
    se = sd / sqrt(n)
    t = dbar / se
    t2 = dbar / sqrt(sP ** 2 / n + sD ** 2 / n)
    t_nos = dbar / (sd / n)
    t_raw = dbar / sd
    t_n1 = dbar / (sd / sqrt(n - 1))
    # (b) BH 法
    names = ["頭痛", "睡眠の質", "不安得点", "疲労感", "食欲", "めまい", "集中力"]
    ps = [0.035, 0.002, 0.21, 0.019, 0.047, 0.009, 0.031]
    m, qq = len(ps), 0.05
    srt = sorted(ps)
    th = [(i + 1) * qq / m for i in range(m)]
    ok = [p <= t_ for p, t_ in zip(srt, th)]
    kmax = max(i + 1 for i in range(m) if ok[i])
    first_fail = next(i for i in range(m) if not ok[i])
    out = {
        "n": n, "mP": s(mP, 1), "sP": s(sP, 1), "mD": s(mD, 1), "sD": s(sD, 1),
        "dbar": s(dbar, 1), "sd": s(sd, 1), "sqn": s(sqrt(n), 4), "se": s(se, 4),
        "t": s(t, 2), "t4": s(t, 4), "t2": s(t2, 2), "t_nos": s(t_nos, 2), "t_raw": s(t_raw, 2), "t_n1": s(t_n1, 2),
        "df": n - 1, "tc": 2.262, "se2": s(sqrt(sP ** 2 / n + sD ** 2 / n), 4),
        "m": m, "q": qq, "kbh": kmax, "kstep": first_fail, "kind": sum(ok),
        "kraw": sum(p < qq for p in ps), "kbon": sum(p <= qq / m for p in ps),
    }
    for i, (nm, p) in enumerate(zip(names, ps)):
        out[f"nm{i}"] = nm
        out[f"pv{i}"] = s(p, 3)
    for i in range(m):
        out[f"sp{i + 1}"] = s(srt[i], 3)
        out[f"th{i + 1}"] = s(th[i], 5)
    return out


def p06():
    y1, t1, y0, t0 = 40, 4000, 30, 6000
    n = y1 + y0
    pi0 = t1 / (t1 + t0)
    e1 = n * pi0
    v = n * pi0 * (1 - pi0)
    z = (y1 - e1) / sqrt(v)
    z_half = (y1 - n / 2) / sqrt(n / 4)
    z_pois = (y1 - e1) / sqrt(e1)
    z_nosq = (y1 - e1) / v
    e1s = n * (1 - pi0)
    z_swap = (y1 - e1s) / sqrt(v)
    # (b) オフセットつきポアソン回帰
    var_b1 = 1 / y1 + 1 / y0
    se_b1 = sqrt(var_b1)
    b1 = math.log((y1 / t1) / (y0 / t0))
    return {
        "y1": y1, "t1": t1, "y0": y0, "t0": t0, "n": n, "T": t1 + t0, "pi0": s(pi0, 1), "pi1": s(1 - pi0, 1),
        "r1": s(1000 * y1 / t1, 1), "r0": s(1000 * y0 / t0, 1),
        "e1": s(e1, 0), "v": s(v, 1), "sdv": s(sqrt(v), 4), "z": s(z, 2), "z4": s(z, 4),
        "z_half": s(z_half, 2), "z_pois": s(z_pois, 2), "z_nosq": s(z_nosq, 2), "z_swap": s(z_swap, 2),
        "e1s": s(e1s, 0), "diff": s(y1 - e1, 0),
        "rr": s(math.exp(b1), 1), "b1": s(b1, 3), "inv1": s(1 / y1, 4), "inv0": s(1 / y0, 4),
        "var_b1": s(var_b1, 4), "se_b1": r(se_b1, 3), "se_b1_5": s(se_b1, 5), "det": y1 * n - y1 * y1,
        "i11": y1, "i00": n, "wald": s(b1 / se_b1, 2),
        "se_rate": s(sqrt(1 / t1 + 1 / t0), 4), "se_sum": s(sqrt(1 / n), 3),
    }


def p07():
    import numpy as np
    # 状態の順：一般病棟 G、ICU I（一時的）、退院 D、死亡 X（吸収）
    PG = {"G": 0.80, "I": 0.05, "D": 0.14, "X": 0.01}
    PI = {"G": 0.20, "I": 0.70, "D": 0.00, "X": 0.10}
    A = np.array([[1 - PG["G"], -PG["I"]], [-PI["G"], 1 - PI["I"]]])
    bX = np.array([PG["X"], PI["X"]])
    aG, aI = np.linalg.solve(A, bX)
    return {
        "gg": s(PG["G"], 2), "gi": s(PG["I"], 2), "gd": s(PG["D"], 2), "gx": s(PG["X"], 2),
        "ig": s(PI["G"], 2), "ii": s(PI["I"], 2), "id": s(PI["D"], 2), "ix": s(PI["X"], 2),
        "c11": s(1 - PG["G"], 2), "c22": s(1 - PI["I"], 2),
        "aG": s(aG, 2), "aI": s(aI, 2), "alive": s(1 - aI, 2), "aliveG": s(1 - aG, 2),
        "sumc": s((1 - PI["I"]) - PG["I"] * PI["G"] / (1 - PG["G"]), 2),
        "w_one": s(1 - PI["X"], 2), "w_ratio": s(PI["G"] / (PI["G"] + PI["X"]), 2),
        "rhs2": s(PI["X"] + PI["G"] * PG["X"] / (1 - PG["G"]), 3),
    }


def p08():
    n = 50
    ry1, ry2, r12 = 0.5, 0.6, 0.4
    num = ry1 ** 2 + ry2 ** 2 - 2 * ry1 * ry2 * r12
    den = 1 - r12 ** 2
    R2 = num / den
    R = sqrt(R2)
    vif = 1 / den
    r2aux = 0.15
    lm = n * r2aux
    return {
        "n": n, "ry1": ry1, "ry2": ry2, "r12": r12, "a1": s(ry1 ** 2, 2), "a2": s(ry2 ** 2, 2),
        "a3": s(2 * ry1 * ry2 * r12, 2), "num": s(num, 2), "den": s(den, 2), "R2": s(R2, 4), "R": s(R, 3),
        "R_max": s(max(ry1, ry2), 3), "R_r2": s(R2, 3), "R_noc": s(sqrt(ry1 ** 2 + ry2 ** 2), 3),
        "R_nod": s(sqrt(num), 3),
        "vif": s(vif, 3), "vif4": s(vif, 4), "sv": s(sqrt(vif), 3), "sv_lin": s(1 / (1 - r12), 3),
        "sv_lin2": s(sqrt(1 / (1 - r12)), 3), "sv_inv": s(sqrt(den), 3),
        "r2aux": r2aux, "lm": s(lm, 1), "c2": 5.991, "c3": 7.815, "c49": 66.339,
    }


def p09():
    b1, b2 = -0.020, -0.030
    cov, vp, va = 40, 100, 144
    d1 = cov / vp
    short = b1 + b2 * d1
    return {
        "b1": s(b1, 3), "b2": s(b2, 3), "cov": cov, "vp": vp, "va": va, "d1": s(d1, 1),
        "bias": s(b2 * d1, 3), "short": s(short, 3), "w_none": s(b1, 3), "w_sign": s(b1 - b2 * d1, 3),
        "w_add": s(b1 + b2, 3), "w_va": s(b1 + b2 * cov / va, 3), "d_va": s(cov / va, 4),
    }


def p10():
    k, m = 6, 8
    N = k * m
    ssa, sse = 260.0, 504.0
    dfa, dfe = k - 1, N - k
    msa, mse = ssa / dfa, sse / dfe
    sa2 = (msa - mse) / m
    icc = sa2 / (sa2 + mse)
    sa2_k = (msa - mse) / k
    return {
        "k": k, "m": m, "N": N, "ssa": s(ssa, 0), "sse": s(sse, 0), "sst": s(ssa + sse, 0),
        "dfa": dfa, "dfe": dfe, "dft": N - 1, "msa": s(msa, 1), "mse": s(mse, 1), "F": s(msa / mse, 2),
        "diff": s(msa - mse, 1), "sa2": s(sa2, 1), "tot": s(sa2 + mse, 1),
        "icc": r(icc, 3), "icc4": s(icc, 4),
        "icc_k": s(sa2_k / (sa2_k + mse), 3), "sa2_k": s(sa2_k, 3), "icc_ms": s(msa / (msa + mse), 3),
        "icc_nom": s((msa - mse) / msa, 3), "eta2": s(ssa / (ssa + sse), 3),
    }


def p11():
    import numpy as np
    pts = {"A": (20, 25), "B": (22, 27), "C": (24, 23), "D": (30, 33), "E": (32, 35), "F": (26, 29)}
    names = list(pts)
    X = np.array([pts[k] for k in names], float)
    cen = np.array([pts["A"], pts["C"]], float)
    hist = []
    for _ in range(20):
        d = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        new = np.array([X[lab == j].mean(axis=0) for j in range(2)])
        hist.append((lab.copy(), new.copy()))
        if np.allclose(new, cen):
            break
        cen = new
    lab1, c1 = hist[0]
    labF, cF = hist[-1]
    assert [names[i] for i in range(6) if labF[i] == 0] == ["A", "B", "C"]

    def cc(v):
        def one(x):
            return s(x, 0) if abs(x - round(x)) < 1e-9 else s(x, 1)
        return f"({one(v[0])},\\ {one(v[1])})"

    out = {}
    for k, (x, y) in pts.items():
        out[f"{k}x"], out[f"{k}y"] = x, y
    # 1回目の割り当てに使う二乗距離
    for k in ("B", "D", "E", "F"):
        x, y = pts[k]
        out[f"d{k}A"] = (x - 20) ** 2 + (y - 25) ** 2
        out[f"d{k}C"] = (x - 24) ** 2 + (y - 23) ** 2
    out["dC1"] = int((24 - c1[0][0]) ** 2 + (23 - c1[0][1]) ** 2)
    out["dC2"] = int((24 - c1[1][0]) ** 2 + (23 - c1[1][1]) ** 2)
    out["dF1"] = int((26 - c1[0][0]) ** 2 + (29 - c1[0][1]) ** 2)
    out["dF2"] = int((26 - c1[1][0]) ** 2 + (29 - c1[1][1]) ** 2)
    out["dF3a"] = s((26 - cF[0][0]) ** 2 + (29 - cF[0][1]) ** 2, 1)
    out["dF3b"] = s((26 - cF[1][0]) ** 2 + (29 - cF[1][1]) ** 2, 1)
    out["dC3a"] = s((24 - cF[0][0]) ** 2 + (23 - cF[0][1]) ** 2, 1)
    out["dC3b"] = s((24 - cF[1][0]) ** 2 + (23 - cF[1][1]) ** 2, 1)
    out["iters"] = len(hist)
    out.update({
        "c_one": f"${cc(c1[0])}$ と ${cc(c1[1])}$", "c_ok": f"${cc(cF[0])}$ と ${cc(cF[1])}$",
        "c_F": f"${cc((23, 26))}$ と ${cc((31, 34))}$", "c_init": f"${cc((20, 25))}$ と ${cc((24, 23))}$",
        "c_med": f"${cc((22, 25))}$ と ${cc((30, 33))}$",
        "c11": cc(c1[0]), "c12": cc(c1[1]), "cF1": cc(cF[0]), "cF2": cc(cF[1]),
    })
    # (b) ホルビッツ・トンプソン推定量
    smp = [(12, 0.10), (30, 0.25), (8, 0.05), (45, 0.40)]
    ht = sum(y / p for y, p in smp)
    for i, (y, p) in enumerate(smp):
        out[f"y{i}"], out[f"pi{i}"], out[f"w{i}"] = y, s(p, 2), s(y / p, 1)
    Nc = 40
    out.update({"Nc": Nc, "ht": r(ht, 1), "ht_s": s(ht, 1), "ht_mean": s(Nc * sum(y for y, _ in smp) / len(smp), 1),
                "ht_ratio": s(sum(y for y, _ in smp) / sum(p for _, p in smp), 2), "ysum": sum(y for y, _ in smp)})
    return out


def p12():
    r1, r2 = 0.6, 0.2
    den = 1 - r1 ** 2
    f1 = r1 * (1 - r2) / den
    f2 = (r2 - r1 ** 2) / den
    def pr(a, b):
        return f"(\\hat\\phi_1,\\ \\hat\\phi_2)=({s(a, 2)},\\ {s(b, 2)})"
    return {
        "r1": r1, "r2": r2, "den": s(den, 2), "n1": s(r1 * (1 - r2), 2), "n2": s(r2 - r1 ** 2, 2),
        "r1sq": s(r1 ** 2, 2), "f1": s(f1, 2), "f2": s(f2, 2),
        "c_ok": pr(f1, f2), "c_raw": pr(r1, r2), "c_ar1": pr(r1, r2 - r1 ** 2), "c_sign": pr(f1, -f2),
        "c_noden": pr(r1 * (1 - r2), r2 - r1 ** 2),
        "chk1": s(f1 + f2 * r1, 2), "chk2": s(f1 * r1 + f2, 2),
    }


def p13():
    m0, t0 = 4.0, 3.0
    est, se = 10.0, 2.0
    w0, w1 = 1 / t0 ** 2, 1 / se ** 2
    pv = 1 / (w0 + w1)
    pm = pv * (w0 * m0 + w1 * est)
    psd = sqrt(pv)
    z = 1.96
    def ci(c, h):
        return f"({s(c - h, 2)},\\ {s(c + h, 2)})"
    pm_sw = m0 + w0 / (w0 + w1) * (est - m0)
    # (b) LOOCV
    x = [1, 2, 3, 4, 5]
    y = [3, 7, 8, 8, 13]
    n = len(x)
    xb, yb = sum(x) / n, sum(y) / n
    sxx = sum((a - xb) ** 2 for a in x)
    sxy = sum((a - xb) * (b - yb) for a, b in zip(x, y))
    b = sxy / sxx
    a = yb - b * xb
    fit = [a + b * xi for xi in x]
    e = [yi - fi for yi, fi in zip(y, fit)]
    h = [1 / n + (xi - xb) ** 2 / sxx for xi in x]
    loo = [ei / (1 - hi) for ei, hi in zip(e, h)]
    cv = sum(v ** 2 for v in loo) / n
    out = {
        "m0": s(m0, 0), "t0": s(t0, 0), "t02": s(t0 ** 2, 0), "est": s(est, 0), "se": s(se, 0), "se2": s(se ** 2, 0),
        "w0": s(w0, 4), "w1": s(w1, 2), "wsum": s(w0 + w1, 4), "pv": s(pv, 4), "psd": s(psd, 4),
        "pm": s(pm, 4), "pm2": s(pm, 2), "wt": s(w1 / (w0 + w1), 4), "hw": s(z * psd, 4),
        "ci_ok": ci(pm, z * psd), "ci_freq": ci(est, z * se), "ci_se": ci(pm, z * se),
        "ci_sw": ci(pm_sw, z * psd), "ci_sum": ci(pm, z * sqrt(t0 ** 2 + se ** 2)),
        "pv_frac": "36/13", "pm_frac": "106/13",
        "sxx": s(sxx, 0), "sxy": s(sxy, 0), "bb": s(b, 1), "aa": s(a, 1), "xb": s(xb, 0), "yb": s(yb, 1),
        "cv": r(cv, 2), "cv4": s(cv, 4), "mse_in": s(sum(v ** 2 for v in e) / n, 2),
        "cv_lin": s(sum(ei ** 2 / (1 - hi) for ei, hi in zip(e, h)) / n, 2),
        "ssum": s(sum(v ** 2 for v in loo), 4), "rss": s(sum(v ** 2 for v in e), 2),
    }
    for i in range(n):
        out[f"x{i}"], out[f"y{i}"] = x[i], y[i]
        out[f"f{i}"], out[f"e{i}"] = s(fit[i], 1), s(e[i], 1)
        out[f"h{i}"], out[f"l{i}"] = s(h[i], 1), s(loo[i], 4)
        out[f"lq{i}"] = s(loo[i] ** 2, 4)
    return out


def values():
    return {"s05p01": p01(), "s05p02": p02(), "s05p03": p03(), "s05p04": p04(), "s05p05": p05(),
            "s05p06": p06(), "s05p07": p07(), "s05p08": p08(), "s05p09": p09(), "s05p10": p10(),
            "s05p11": p11(), "s05p12": p12(), "s05p13": p13()}


if __name__ == "__main__":
    for k, v in values().items():
        print(k, v)
