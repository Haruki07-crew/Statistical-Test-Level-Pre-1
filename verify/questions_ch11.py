"""第11章（正規分布に関する検定）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from scipy import stats

q = Q(11)
N = stats.norm
rng = np.random.default_rng(11)

# --- z 検定 ---
q.num("q11-z-test-known-variance-1", (106 - 100) / (15 / 5))
q.num("q11-z-test-known-variance-2", N.cdf((96 - 100) / (12 / 6)))
q.check("q11-z-test-known-variance-2", abs(2 * N.cdf(-2) - 0.0455) < 5e-5, "両側 0.0455")

# --- 1標本 t ---
tv = [stats.t.ppf(0.975, 14), stats.t.ppf(0.975, 15), stats.t.ppf(0.95, 14), N.ppf(0.975)]
q.check("q11-one-sample-t-test-1", np.allclose(tv, [2.145, 2.131, 1.761, 1.960], atol=5e-4), "正解 2.145、誤答の値も確認")
x = np.array([10.8, 12.1, 11.5, 12.6, 11.9, 12.5])
res = stats.ttest_1samp(x, 11)
q.num("q11-one-sample-t-test-2", res.statistic)
q.check("q11-one-sample-t-test-2", abs(((x - x.mean()) ** 2).sum() - 2.26) < 1e-9 and res.pvalue < 0.05, "偏差平方和2.26、5%で棄却")

# --- 2標本 t / プール ---
q.check("q11-two-sample-t-test-1", 6 + 8 - 2 == 12, "自由度 12")
sp2 = (5 * 5 + 7 * 4) / 12
res = stats.ttest_ind_from_stats(24.0, np.sqrt(5), 6, 21.0, np.sqrt(4), 8, equal_var=True)
q.num("q11-two-sample-t-test-2", res.statistic)
q.check("q11-two-sample-t-test-2", abs(res.statistic - 3 / np.sqrt(sp2 * (1 / 6 + 1 / 8))) < 1e-12 and res.pvalue < 0.05, "式と scipy が一致")
q.num("q11-pooled-variance-1", (4 * 10 + 10 * 4) / 14)
q.check("q11-pooled-variance-2", abs((4 * 10 + 10 * 4) / 14 - 7) > 1 and abs((9 * 10 + 9 * 4) / 18 - 7) < 1e-12, "m≠n で単純平均と異なる（誤り）")

# --- ウェルチ ---
a, b = 20 / 10, 5 / 15
res = stats.ttest_ind_from_stats(3, np.sqrt(20), 10, 0, np.sqrt(5), 15, equal_var=False)
q.num("q11-welch-test-1", res.statistic)
nu = (a + b) ** 2 / (a * a / 9 + b * b / 14)
q.num("q11-welch-test-2", nu)
q.check("q11-welch-test-2", res.pvalue > 0.05 and abs(stats.t.ppf(0.975, nu) - 2.178) < 1e-3, "5%で非棄却、t点 2.178")
ok = True
for _ in range(20000):
    m, n = rng.integers(2, 40, 2)
    s1, s2 = rng.uniform(0.01, 100, 2)
    a_, b_ = s1 / m, s2 / n
    v = (a_ + b_) ** 2 / (a_ ** 2 / (m - 1) + b_ ** 2 / (n - 1))
    ok &= min(m - 1, n - 1) - 1e-9 <= v <= m + n - 2 + 1e-9
q.check("q11-welch-test-3", bool(ok), "ランダムな値で min(m-1,n-1)≤ν≤m+n-2（正しい）")

# --- 対応のある t ---
q.check("q11-paired-t-test-1", 5 - 1 == 4, "自由度 4")
pre = np.array([82, 75, 90, 68, 77]); post = np.array([78, 74, 85, 66, 72])
res = stats.ttest_rel(pre, post)
q.num("q11-paired-t-test-2", res.statistic)
q.check("q11-paired-t-test-2", abs(((pre - post - 3.4) ** 2).sum() - 13.2) < 1e-9 and res.statistic > stats.t.ppf(0.975, 4), "SS 13.2、棄却")

# --- 母分散のカイ二乗検定 ---
chi = 14 * 6.5 / 4
q.num("q11-chi-square-test-variance-1", chi)
q.check("q11-chi-square-test-variance-1", chi < stats.chi2.isf(0.05, 14) and abs(stats.chi2.sf(chi, 14) - 0.064) < 5e-4, "非棄却、p≈0.064")
sims = rng.normal(3, 2, (100000, 6))
known = ((sims - 3) ** 2).sum(1) / 4
q.check("q11-chi-square-test-variance-2", abs(known.mean() - 6) < 0.05, "μ既知の統計量の平均は n=6（自由度 n、誤り）")

# --- F 検定 ---
F = 12.6 / 4.2
q.num("q11-f-test-equal-variances-1", 2 * stats.f.sf(F, 8, 12))
q.check("q11-f-test-equal-variances-1", F < stats.f.isf(0.025, 8, 12) and abs(stats.f.isf(0.025, 8, 12) - 3.51) < 5e-3, "F_0.025(8,12)=3.51")
s1 = rng.normal(0, 1, (200000, 10)).var(1, ddof=1)
s2 = rng.normal(0, 1, (200000, 10)).var(1, ddof=1)
Fb = np.maximum(s1, s2) / np.minimum(s1, s2)
q.check("q11-f-test-equal-variances-2", abs(np.mean(Fb > stats.f.isf(0.05, 9, 9)) - 0.10) < 0.005
        and abs(np.mean(Fb > stats.f.isf(0.025, 9, 9)) - 0.05) < 0.005, "F_0.05 だとサイズ10%（誤り）")

# --- 相関係数 ---
T = 0.4 * 5 / np.sqrt(0.84)
q.num("q11-test-correlation-coefficient-1", T)
q.check("q11-test-correlation-coefficient-1", T > stats.t.ppf(0.975, 25) and abs(stats.t.ppf(0.975, 25) - 2.060) < 5e-4, "t(25)=2.060 を超える")
t18 = 2.101
rc = t18 / np.sqrt(18 + t18 ** 2)
q.num("q11-test-correlation-coefficient-2", rc)
q.check("q11-test-correlation-coefficient-2", abs(rc * np.sqrt(18) / np.sqrt(1 - rc ** 2) - t18) < 1e-12, "逆算で t=2.101")

# --- フィッシャーの z ---
q.num("q11-fisher-z-transformation-1", (np.arctanh(0.6) - np.arctanh(0.4)) * 6)
q.num("q11-fisher-z-transformation-2", np.tanh(np.arctanh(0.5) + 1.96 / 5))
q.check("q11-fisher-z-transformation-2", abs(np.tanh(np.arctanh(0.5) - 1.96 / 5) - 0.156) < 5e-4, "下限 0.156")

# --- t 検定の検出力 ---
d = np.sqrt(20) * 0.5
q.check("q11-power-of-t-test-1", abs(d - 2.236) < 5e-4 and all(abs(w - d) > 0.5 for w in [0.5, 10, 0.5 / np.sqrt(20)]), "δ=√n d=2.236")
q.num("q11-power-of-t-test-2", N.cdf(d - 1.96))
c = stats.t.ppf(0.975, 19)
exact = stats.nct.sf(c, 19, d) + stats.nct.cdf(-c, 19, d)
q.check("q11-power-of-t-test-2", abs(exact - 0.565) < 1e-3 and abs(c - 2.093) < 5e-4, "非心 t の正確値 0.565")

# --- バートレット（裾の重い分布で過大に棄却） ---
rej = np.mean([stats.bartlett(*rng.standard_t(3, (3, 20))).pvalue < 0.05 for _ in range(3000)])
q.check("q11-bartlett-test-1", rej > 0.10, f"t(3) 分布・等分散で棄却率 {rej:.3f} > 5%（正しい）")

# --- 検定と区間の関係 ---
sp2 = (4 * 10 + 19 * 1) / 23
se_p = np.sqrt(sp2 * (1 / 5 + 1 / 20))
lo_p = 2.2 - stats.t.ppf(0.975, 23) * se_p
se_w = np.sqrt(10 / 5 + 1 / 20)
nu = se_w ** 4 / ((10 / 5) ** 2 / 4 + (1 / 20) ** 2 / 19)
lo_w = 2.2 - stats.t.ppf(0.975, nu) * se_w
pw = stats.ttest_ind_from_stats(2.2, np.sqrt(10), 5, 0, 1, 20, equal_var=False).pvalue
q.check("q11-test-ci-relationship-mean-difference-1", lo_p > 0 and lo_w < 0 and pw > 0.05
        and abs(lo_p - 0.54) < 5e-3 and abs(lo_w + 1.70) < 5e-3, "等分散区間は0を含まず、ウェルチは非棄却（誤り）")

# --- 予備検定 ---
cnt = 0; R = 20000
for _ in range(R):
    xa = rng.normal(0, 3, 6); xb = rng.normal(0, 1, 30)
    Fv = xa.var(ddof=1) / xb.var(ddof=1)
    pf = 2 * min(stats.f.sf(Fv, 5, 29), stats.f.cdf(Fv, 5, 29))
    p = stats.ttest_ind(xa, xb, equal_var=(pf >= 0.05)).pvalue
    cnt += p < 0.05
q.check("q11-preliminary-test-problem-1", abs(cnt / R - 0.05) > 0.01, f"予備検定つき手順の第1種の過誤 {cnt / R:.3f}（5%ではない＝誤り）")

# --- 正規性 ---
w = stats.shapiro(rng.normal(size=50)).statistic
w2 = stats.shapiro(rng.exponential(size=50)).statistic
q.check("q11-normality-test-1", 0 < w <= 1 and w2 < w, "W≤1、非正規で小さい")

# --- ホテリング ---
Fh = (12 - 2) / (2 * 11) * 9
q.num("q11-hotelling-t2-1", Fh)
q.check("q11-hotelling-t2-1", abs(stats.f.isf(0.05, 2, 10) - 4.103) < 5e-4 and abs(stats.f.sf(Fh, 2, 10) - 0.0503) < 5e-5, "F(2,10) 5%点 4.103、p=0.0503")

# --- ルビーン ---
res = stats.levene([1, 3, 5, 7], [3, 4, 4, 5], center="mean")
q.num("q11-one-way-test-variance-equality-levene-1", res.statistic)
q.check("q11-one-way-test-variance-equality-levene-1", res.statistic < stats.f.isf(0.05, 1, 6) and abs(stats.f.isf(0.05, 1, 6) - 5.99) < 5e-3,
        "F(1,6) 5%点 5.99 未満")

q.done()
