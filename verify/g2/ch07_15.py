"""2級 第7・10・14・15章で2級向けに書いた用語の検算。"""
from math import comb, perm, sqrt
import numpy as np
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-4):
    return abs(float(x) - claim) <= tol


T = "counting"
check(T, "C(5,3)=10・P(5,3)=60=10×3!", comb(5, 3) == 10 and perm(5, 3) == 60 == 10 * 6, None)
check(T, "不良 0 個 20/120=1/6・1 個 60/120=1/2", comb(6, 3) == 20 and comb(10, 3) == 120 and comb(4, 1) * comb(6, 2) == 60, None)
check(T, "超幾何分布で確認", close(stats.hypergeom(10, 4, 3).pmf(0), 1 / 6, 1e-12) and close(stats.hypergeom(10, 4, 3).pmf(1), 0.5, 1e-12), None)
check(T, "表 2 回 10/32=0.3125", close(stats.binom(5, 0.5).pmf(2), 0.3125, 1e-12), None)
T = "events-and-probability"
A = {2, 4, 6}; B = {4, 5, 6}
check(T, "P(A∩B)=1/3・P(A∪B)=2/3", len(A & B) == 2 and len(A | B) == 4 and close(0.5 + 0.5 - 1 / 3, 2 / 3, 1e-12), None)
check(T, "少なくとも 1 回 6：91/216≈0.421", 216 - 125 == 91 and close(91 / 216, 0.421), 91 / 216)
check(T, "表 [[2,1],[1,2]]", [[len(A & B), len(A - B)], [len(B - A), 6 - len(A | B)]] == [[2, 1], [1, 2]], None)
T = "sampling-distribution-of-mean"
check(T, "SD 4/√16=1・P(Z<−2)≈0.0228", 4 / sqrt(16) == 1 and close(stats.norm.cdf(-2), 0.0228), stats.norm.cdf(-2))
check(T, "1 袋なら P(Z<−0.5)≈0.309", close(stats.norm.cdf(-0.5), 0.309, 5e-4), stats.norm.cdf(-0.5))
rng = np.random.default_rng(3)
m = rng.normal(500, 4, (200000, 16)).mean(1)
check(T, "シミュレーション：平均 500・SD 1・P(<498)", close(m.mean(), 500, 0.01) and close(m.std(), 1, 0.01) and close((m < 498).mean(), 0.0228, 0.002), (m.mean(), m.std(), (m < 498).mean()))
T = "statistical-tables"
check(T, "z 1.645・1.960・2.576", close(stats.norm.isf(0.05), 1.645) and close(stats.norm.isf(0.025), 1.960) and close(stats.norm.isf(0.005), 2.576), None)
check(T, "t 2.262（9）・2.086（20）", close(stats.t.isf(0.025, 9), 2.262) and close(stats.t.isf(0.025, 20), 2.086), None)
check(T, "χ² 19.02・2.700（9）", close(stats.chi2.isf(0.025, 9), 19.02, 0.005) and close(stats.chi2.isf(0.975, 9), 2.700), None)
check(T, "F 3.708（3,10）・8.786（10,3）・1/8.786≈0.114 = F(0.95;3,10)", close(stats.f.isf(0.05, 3, 10), 3.708) and close(stats.f.isf(0.05, 10, 3), 8.786) and close(stats.f.isf(0.95, 3, 10), 0.114, 5e-4), stats.f.isf(0.95, 3, 10))
check(T, "表の値 P(Z>1)=0.1587・P(Z>2)=0.0228（4桁に丸め）", close(stats.norm.sf(1), 0.1587, 5e-5) and close(stats.norm.sf(2), 0.0228, 5e-5), None)
check(T, "表の値から 1−0.1587−0.0228=0.8185（正確な値 0.81859 と丸めの範囲で一致）", close(1 - 0.1587 - 0.0228, 0.8185, 1e-12) and close(stats.norm.cdf(2) - stats.norm.cdf(-1), 0.8185, 1e-4), stats.norm.cdf(2) - stats.norm.cdf(-1))
T = "regression-output"
check(T, "t 値 2.47・9.00・−2.67", close(2.10 / 0.85, 2.47, 0.005) and close(0.18 / 0.02, 9) and close(-0.12 / 0.045, -2.67, 0.005), None)
check(T, "p 値 0.020・0.000・0.013（自由度 27）", close(2 * stats.t.sf(2.47, 27), 0.020, 5e-4) and 2 * stats.t.sf(9, 27) < 0.0005 and close(2 * stats.t.sf(2.67, 27), 0.013, 5e-4), None)
check(T, "調整済み R² 0.764・F 47.9", close(1 - 0.22 * 29 / 27, 0.764) and close((0.78 / 2) / (0.22 / 27), 47.9, 0.05), None)
tq = stats.t.isf(0.025, 27)
check(T, "95% 区間 (0.139, 0.221)・t=2.052", close(tq, 2.052) and close(0.18 - tq * 0.02, 0.139) and close(0.18 + tq * 0.02, 0.221), tq)
check(T, "予測値 5.40", close(2.10 + 0.18 * 25 - 0.12 * 10, 5.40, 1e-9), None)
T = "treatment-and-control"
p = 58 / 100; z = 0.2 / sqrt(p * (1 - p) * (2 / 50))
check(T, "z≈2.03・p≈0.043", close(z, 2.03, 0.005) and close(2 * stats.norm.sf(z), 0.043, 5e-4), (z, 2 * stats.norm.sf(z)))
T = "anova-table"
g = [np.array([60, 65, 70, 62, 68]), np.array([70, 72, 75, 68, 75]), np.array([58, 62, 60, 55, 65])]
allv = np.concatenate(g); gm = allv.mean()
SA = sum(5 * (x.mean() - gm) ** 2 for x in g); SE = sum(((x - x.mean()) ** 2).sum() for x in g); ST = ((allv - gm) ** 2).sum()
check(T, "平均 65・72・60、全体 197/3", [x.mean() for x in g] == [65, 72, 60] and close(gm, 197 / 3, 1e-12), gm)
check(T, "平方和 363.3・164（68+38+58）・527.3", close(SA, 363.3, 0.05) and SE == 164 and [((x - x.mean()) ** 2).sum() for x in g] == [68, 38, 58] and close(ST, 527.3, 0.05) and close(SA + SE, ST, 1e-9), (SA, SE, ST))
F = (SA / 2) / (SE / 12)
check(T, "平均平方 181.7・13.67、F 13.29", close(SA / 2, 181.7, 0.05) and close(SE / 12, 13.67, 0.005) and close(F, 13.29, 0.005), F)
check(T, "F(0.05;2,12)=3.885・p≈0.0009", close(stats.f.isf(0.05, 2, 12), 3.885) and close(stats.f.sf(F, 2, 12), 0.0009, 5e-5) and close(stats.f_oneway(*g).statistic, F, 1e-9), stats.f.sf(F, 2, 12))

ng = results.count(False)
print(f"第7・10・14・15章（2級）の検算: OK {results.count(True)} / NG {ng}")
raise SystemExit(1 if ng else 0)
