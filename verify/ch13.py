"""第13章 検算スクリプト（ノンパラメトリック法）。
数値は scipy.stats（wilcoxon, mannwhitneyu, ranksums, kruskal, friedmanchisquare, spearmanr, kendalltau,
permutation_test, ttest_ind, ttest_rel など）と、自前の全通りの数え上げ（符号 2^n、群分け C(N,n1)、
多群の分割、ブロック内順位 (k!)^b、並べ替え n!）。
記号計算は sympy（帰無仮説のもとでの期待値・分散、H・Q の2つの形の一致、k=2 での帰着、ARE の積分）。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from math import sqrt, comb, pi, asin, factorial
from itertools import combinations, permutations, product
from collections import Counter
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


rng = np.random.default_rng(20261023)
norm, chi2, binom = stats.norm, stats.chi2, stats.binom
rankdata = stats.rankdata


def signflip_dist(ranks):
    """符号の 2^n 通りについて T+ の値のリスト。"""
    ranks = np.asarray(ranks, float)
    n = len(ranks)
    out = []
    for s in product([0, 1], repeat=n):
        out.append(float(np.dot(s, ranks)))
    return np.array(out)


def ranksum_dist(ranks, n1):
    ranks = np.asarray(ranks, float)
    return np.array([ranks[list(c)].sum() for c in combinations(range(len(ranks)), n1)])


def kendall_S(y):
    n = len(y)
    return sum(np.sign(y[j] - y[i]) for i in range(n) for j in range(i + 1, n))


# ================= sign-test =================
T = "sign-test"
d = [2.1, 0.8, 1.5, -0.4, 3.0, 0, 1.1, 0.6, 2.4, 1.7]
dn = [x for x in d if x != 0]
S = sum(x > 0 for x in dn)
check(T, "例1 n=9, S=8", len(dn) == 9 and S == 8, (len(dn), S))
p = binom.sf(7, 9, .5)
check(T, "例1 P(S≥8)=10/512=0.0195", close(p, .0195) and comb(9, 8) + comb(9, 9) == 10, p)
r = stats.binomtest(8, 9, .5)
check(T, "例1 両側 0.0391（binomtest と一致）", close(2 * p, .0391) and close(r.pvalue, 2 * p, 1e-12), r.pvalue)
z_cc, z_nc = (8 - 4.5 - .5) / 1.5, (8 - 4.5) / 1.5
check(T, "例1 E=4.5, SD=1.5", 9 / 2 == 4.5 and sqrt(9 / 4) == 1.5, "")
check(T, "例1 連続修正あり z=2.000, p=0.0228", close(z_cc, 2.0, 5e-4) and close(norm.sf(z_cc), .0228), norm.sf(z_cc))
check(T, "例1 連続修正なし z=2.333, p=0.0098", close(z_nc, 2.333, 5e-4) and close(norm.sf(z_nc), .0098), norm.sf(z_nc))
check(T, "例1 連続修正ありのほうが正確値に近い", abs(norm.sf(z_cc) - p) < abs(norm.sf(z_nc) - p), "")
p2 = binom.sf(14, 20, .5)
check(T, "例2 P(S≥15)=0.0207, 両側 0.0414", close(p2, .0207) and close(2 * p2, .0414) and close(stats.binomtest(15, 20).pvalue, 2 * p2, 1e-12), p2)
k = sp.symbols('k', integer=True)
n = sp.symbols('n', positive=True, integer=True)
check(T, "sympy E[S]=n/2, V[S]=n/4（独立ベルヌーイの和）", sp.simplify(n * sp.Rational(1, 2) - n / 2) == 0 and sp.simplify(n * sp.Rational(1, 4) - n / 4) == 0, "")
check(T, "図 Bin(9,1/2) の S≥8 の棒の合計=0.0195", close(binom.pmf(8, 9, .5) + binom.pmf(9, 9, .5), .0195), "")
check(T, "ARE（正規）2/π≈0.637", close(2 / pi, .637, 5e-4), 2 / pi)

# ================= wilcoxon-signed-rank-test =================
T = "wilcoxon-signed-rank-test"
E1 = np.array([1.8, -0.6, 2.5, 0.9, 3.1, -1.2, 2.2, 1.4])
R1 = rankdata(np.abs(E1))
Tp = R1[E1 > 0].sum()
Tm = R1[E1 < 0].sum()
check(T, "例1 負の差の順位 1,3, T-=4, T+=32", sorted(R1[E1 < 0]) == [1, 3] and Tm == 4 and Tp == 32, (Tm, Tp))
dist8 = signflip_dist(np.arange(1, 9))
cnt = Counter(dist8)
check(T, "例1 T-=0..4 の通り数 1,1,1,2,2（計7）", [cnt[v] for v in range(5)] == [1, 1, 1, 2, 2], [cnt[v] for v in range(5)])
pe = (dist8 >= 32).mean()
check(T, "例1 P(T+≥32)=7/256=0.0273", close(pe, .0273) and (dist8 >= 32).sum() == 7, pe)
w = stats.wilcoxon(E1, alternative="greater", method="exact")
check(T, "例1 scipy wilcoxon exact 右片側=7/256", close(w.pvalue, 7 / 256, 1e-12), w.pvalue)
w2 = stats.wilcoxon(E1, method="exact")
check(T, "例1 両側 0.0547", close(w2.pvalue, .0547) and close(2 * pe, .0547), w2.pvalue)
V8 = 8 * 9 * 17 / 24
check(T, "例1 E=18, V=51", 8 * 9 / 4 == 18 and V8 == 51 and close(dist8.mean(), 18, 1e-12) and close(dist8.var(), 51, 1e-9), dist8.var())
zc, zn = (32 - 18 - .5) / sqrt(51), 14 / sqrt(51)
check(T, "例1 連続修正あり z=1.890, p=0.0294", close(zc, 1.890, 5e-4) and close(norm.sf(zc), .0294), norm.sf(zc))
check(T, "例1 連続修正なし z=1.960, p=0.0250", close(zn, 1.960, 5e-4) and close(norm.sf(zn), .0250), norm.sf(zn))
wn = stats.wilcoxon(E1, alternative="greater", method="approx", correction=True)
check(T, "例1 scipy wilcoxon approx(連続修正) と一致", close(wn.pvalue, norm.sf(zc), 1e-10), wn.pvalue)
check(T, "例1 符号検定 P(S≥6)=37/256=0.1445", (E1 > 0).sum() == 6 and close(binom.sf(5, 8, .5), .1445) and close(binom.sf(5, 8, .5), 37 / 256, 1e-15), binom.sf(5, 8, .5))
tt = stats.ttest_1samp(E1, 0, alternative="greater")
check(T, "例1 対応のある t=2.380, 片側 p=0.0244", close(tt.statistic, 2.380, 5e-4) and close(tt.pvalue, .0244), (tt.statistic, tt.pvalue))
# bar-data
lab = list(range(37))
vals = [0.0039, 0.0039, 0.0039, 0.0078, 0.0078, 0.0117, 0.0156, 0.0195, 0.0234, 0.0273, 0.0312, 0.0352, 0.0391, 0.043, 0.0469, 0.0508, 0.0508, 0.0508, 0.0547, 0.0508, 0.0508, 0.0508, 0.0469, 0.043, 0.0391, 0.0352, 0.0312, 0.0273, 0.0234, 0.0195, 0.0156, 0.0117, 0.0078, 0.0078, 0.0039, 0.0039, 0.0039]
calc = [round(cnt[v] / 256, 4) for v in lab]
check(T, "bar-data n=8 の T+ の正確分布 37値", all(abs(a - b) < 1e-9 for a, b in zip(calc, vals)), [i for i, (a, b) in enumerate(zip(calc, vals)) if abs(a - b) > 1e-9])
nbar = len([v for v in lab if v >= 32])
check(T, "キャプション 右端の T+≥32 の棒は5本で合計 7/256", nbar == 5 and sum(cnt[v] for v in range(32, 37)) == 7, f"棒の本数={nbar}")
# 例2
E2 = np.array([3, -1, 2, 2, 0, 4, -2, 5, 3, 1], float)
E2n = E2[E2 != 0]
R2 = rankdata(np.abs(E2n))
Tm2, Tp2 = R2[E2n < 0].sum(), R2[E2n > 0].sum()
check(T, "例2 n=9, 平均順位 1→1.5, 2→4, 3→6.5", len(E2n) == 9 and set(zip(np.abs(E2n), R2)) == {(1, 1.5), (2, 4), (3, 6.5), (4, 8), (5, 9)}, "")
check(T, "例2 T-=5.5, T+=39.5", Tm2 == 5.5 and Tp2 == 39.5, (Tm2, Tp2))
ties = [c for c in Counter(np.abs(E2n)).values() if c > 1]
st = sum(t ** 3 - t for t in ties)
check(T, "例2 タイ t=2,3,2 で Σ(t³−t)=36", sorted(ties) == [2, 2, 3] and st == 36, st)
V2 = 9 * 10 * 19 / 24 - st / 48
check(T, "例2 V=71.25−0.75=70.5", V2 == 70.5, V2)
d2 = signflip_dist(R2)
check(T, "例2 平均順位での正確な分散=Σr²/4=70.5", close(d2.var(), 70.5, 1e-9) and close((R2 ** 2).sum() / 4, 70.5, 1e-12), d2.var())
z2 = (39.5 - 22.5 - .5) / sqrt(70.5)
check(T, "例2 z=1.965, 両側 p=0.0494", close(z2, 1.965, 5e-4) and close(2 * norm.sf(z2), .0494), 2 * norm.sf(z2))
ce = ((np.abs(d2 - 22.5) >= 17 - 1e-9)).sum()
check(T, "例2 正確な両側 p=26/512=0.0508", ce == 26 and close(26 / 512, .0508), ce)
wz = stats.wilcoxon(E2, zero_method="wilcox", correction=True, method="approx")
check(T, "例2 scipy wilcoxon approx（差0除去・タイ補正・連続修正）と一致", close(wz.pvalue, 2 * norm.sf(z2), 1e-8), wz.pvalue)
# sympy
check(T, "sympy E[T+]=Σk/2=n(n+1)/4", sp.simplify(sp.summation(k / 2, (k, 1, n)) - n * (n + 1) / 4) == 0, "")
check(T, "sympy V[T+]=Σk²/4=n(n+1)(2n+1)/24", sp.simplify(sp.summation(k ** 2 / 4, (k, 1, n)) - n * (n + 1) * (2 * n + 1) / 24) == 0, "")
a, t = sp.symbols('a t', positive=True)
j = sp.symbols('j', integer=True)
grp = sp.summation((a + j) ** 2, (j, 1, t)) - t * (a + (t + 1) / 2) ** 2
check(T, "sympy タイ群の順位2乗和の減少=(t³−t)/12 → /4 で /48", sp.simplify(grp - (t ** 3 - t) / 12) == 0, sp.factor(grp))
okv = all(close(signflip_dist(np.arange(1, m + 1)).var(), m * (m + 1) * (2 * m + 1) / 24, 1e-9) for m in range(1, 11))
check(T, "数え上げ n=1..10 で V[T+] 公式一致", okv, "")
check(T, "ARE（正規）3/π≈0.955", close(3 / pi, .955, 5e-4), 3 / pi)

# ================= wilcoxon-rank-sum-test =================
T = "wilcoxon-rank-sum-test"
xa, ya = np.array([14, 19, 22, 25.]), np.array([8, 11, 13, 16, 20.])
rr = rankdata(np.r_[xa, ya])
W = rr[:4].sum()
check(T, "例1 新肥料の順位 4,6,8,9, W=27", list(rr[:4]) == [4, 6, 8, 9] and W == 27, rr[:4])
dW = ranksum_dist(np.arange(1, 10), 4)
cW = Counter(dW)
check(T, "例1 W=27,28,29,30 の通り数 3,2,1,1", [cW[v] for v in (27, 28, 29, 30)] == [3, 2, 1, 1], [cW[v] for v in (27, 28, 29, 30)])
pW = (dW >= 27).mean()
check(T, "例1 P(W≥27)=7/126=0.0556", len(dW) == 126 and close(pW, .0556) and close(pW, 7 / 126, 1e-15), pW)
mw = stats.mannwhitneyu(xa, ya, alternative="greater", method="exact")
check(T, "例1 scipy mannwhitneyu exact と一致", close(mw.pvalue, 7 / 126, 1e-12), mw.pvalue)
check(T, "例1 E[W]=20, V[W]=16.667（数え上げとも一致）", close(dW.mean(), 20, 1e-12) and close(dW.var(), 4 * 5 * 10 / 12, 1e-9) and close(4 * 5 * 10 / 12, 16.667, 5e-4), dW.var())
zc, zn = 6.5 / sqrt(50 / 3), 7 / sqrt(50 / 3)
check(T, "例1 連続修正あり z=1.592, p=0.0557", close(zc, 1.592, 5e-4) and close(norm.sf(zc), .0557), norm.sf(zc))
check(T, "例1 連続修正なし z=1.715, p=0.0432", close(zn, 1.715, 5e-4) and close(norm.sf(zn), .0432), norm.sf(zn))
rs_ = stats.ranksums(xa, ya, alternative="greater")
check(T, "例1 scipy ranksums（修正なし）と一致", close(rs_.pvalue, norm.sf(zn), 1e-10), rs_.pvalue)
mwa = stats.mannwhitneyu(xa, ya, alternative="greater", method="asymptotic", use_continuity=True)
check(T, "例1 scipy mannwhitneyu asymptotic(連続修正) と一致", close(mwa.pvalue, norm.sf(zc), 1e-10), mwa.pvalue)
tt = stats.ttest_ind(xa, ya, alternative="greater")
check(T, "例1 2標本 t 検定の片側 p=0.0396", close(tt.pvalue, .0396), tt.pvalue)
lab = list(range(10, 31))
vals = [0.0079, 0.0079, 0.0159, 0.0238, 0.0397, 0.0476, 0.0635, 0.0714, 0.0873, 0.0873, 0.0952, 0.0873, 0.0873, 0.0714, 0.0635, 0.0476, 0.0397, 0.0238, 0.0159, 0.0079, 0.0079]
calc = [round(cW[v] / 126, 4) for v in lab]
check(T, "bar-data n1=4,n2=5 の W 分布 21値", calc == vals, [i for i, (a_, b_) in enumerate(zip(calc, vals)) if a_ != b_])
# 例2 n1=n2=10, W>=128 （DP で正確分布）
def ranksum_counts(N, n1):
    # f[m][s] = 1..i から m 個選んで和 s の通り数
    smax = N * (N + 1) // 2
    f = np.zeros((n1 + 1, smax + 1), dtype=object)
    f[0][0] = 1
    for i in range(1, N + 1):
        for m in range(min(i, n1), 0, -1):
            f[m][i:] = f[m][i:] + f[m - 1][:smax + 1 - i]
    return f[n1]
c20 = ranksum_counts(20, 10)
tot = sum(c20)
pe2 = sum(c20[128:]) / tot
check(T, "例2 総数 C(20,10)=184756", tot == 184756, tot)
check(T, "例2 正確 P(W≥128)=0.0446", close(pe2, .0446), pe2)
check(T, "例2 E=105, V=175", 10 * 21 / 2 == 105 and 100 * 21 / 12 == 175, "")
zc, zn = 22.5 / sqrt(175), 23 / sqrt(175)
check(T, "例2 連続修正あり z=1.701, p=0.0445", close(zc, 1.701, 5e-4) and close(norm.sf(zc), .0445), norm.sf(zc))
check(T, "例2 連続修正なし z=1.739, p=0.0410", close(zn, 1.739, 5e-4) and close(norm.sf(zn), .0410), norm.sf(zn))
# sympy
Ns, n1s = sp.symbols('N n1', positive=True)
cov = sp.solve(sp.Eq(0, Ns * (Ns ** 2 - 1) / 12 + Ns * (Ns - 1) * sp.Symbol('c')), sp.Symbol('c'))[0]
check(T, "sympy Cov(Ri,Rj)=−(N+1)/12", sp.simplify(cov + (Ns + 1) / 12) == 0, cov)
VW = n1s * (Ns ** 2 - 1) / 12 + n1s * (n1s - 1) * cov
check(T, "sympy V[W]=n1(N−n1)(N+1)/12=n1n2(N+1)/12", sp.simplify(VW - n1s * (Ns - n1s) * (Ns + 1) / 12) == 0, sp.factor(VW))
check(T, "sympy 有限母集団修正の形 n1σ²(N−n1)/(N−1)", sp.simplify(VW - n1s * (Ns ** 2 - 1) / 12 * (Ns - n1s) / (Ns - 1)) == 0, "")
okv = all(close(ranksum_dist(np.arange(1, N_ + 1), m).var(), m * (N_ - m) * (N_ + 1) / 12, 1e-9) for N_ in range(3, 11) for m in range(1, N_))
check(T, "数え上げ N=3..10 全 n1 で V[W] 公式一致", okv, "")
check(T, "ARE 下限 0.864=108/125", close(108 / 125, .864, 1e-12), "")

# ================= mann-whitney-u-test =================
T = "mann-whitney-u-test"
U = sum((x > y) + .5 * (x == y) for x in xa for y in ya)
check(T, "例1 U=3+4+5+5=17", [sum(x > ya) for x in xa] == [3, 4, 5, 5] and U == 17, U)
check(T, "例1 U=W−n1(n1+1)/2=27−10", W - 4 * 5 / 2 == 17 and mw.statistic == 17, mw.statistic)
check(T, "例1 P(U≥17)=7/126, 17/20=0.85", close(mw.pvalue, 7 / 126, 1e-12) and 17 / 20 == .85, "")
check(T, "例1 連続修正つき z=1.592, p=0.0557", close(norm.sf(6.5 / sqrt(50 / 3)), .0557), "")
vals = [0.0079, 0.0079, 0.0159, 0.0238, 0.0397, 0.0476, 0.0635, 0.0714, 0.0873, 0.0873, 0.0952, 0.0873, 0.0873, 0.0714, 0.0635, 0.0476, 0.0397, 0.0238, 0.0159, 0.0079, 0.0079]
calc = [round(cW[v + 10] / 126, 4) for v in range(21)]
check(T, "bar-data U 分布 21値（W を 10 ずらしたもの）", calc == vals, "")
A = np.array([3, 4, 4, 5, 5, 3.]); B = np.array([2, 3, 3, 4, 2, 1, 3.])
U2 = sum((x > y) + .5 * (x == y) for x in A for y in B)
check(T, "例2 U=36", U2 == 36, U2)
tc = Counter(np.r_[A, B])
check(T, "例2 評価1〜5のタイ 1,2,5,3,2, Σ(t³−t)=156", [tc[v] for v in (1, 2, 3, 4, 5)] == [1, 2, 5, 3, 2] and sum(c ** 3 - c for c in tc.values()) == 156, "")
VU = 42 / 12 * (14 - 156 / (13 * 12))
check(T, "例2 V[U]=3.5×13=45.5", close(VU, 45.5, 1e-12), VU)
r13 = rankdata(np.r_[A, B])
dU = ranksum_dist(r13, 6) - 21
check(T, "例2 平均順位固定の全1716通りの分散=45.5", len(dU) == 1716 and close(dU.var(), 45.5, 1e-9), dU.var())
z = 14.5 / sqrt(45.5)
check(T, "例2 z=2.150, 両側 p=0.0316", close(z, 2.150, 5e-4) and close(2 * norm.sf(z), .0316), 2 * norm.sf(z))
mwt = stats.mannwhitneyu(A, B, method="asymptotic", use_continuity=True)
check(T, "例2 scipy mannwhitneyu asymptotic(タイ補正・連続修正) と一致", close(mwt.pvalue, 2 * norm.sf(z), 1e-10), mwt.pvalue)
ce = (np.abs(dU - 21) >= 15 - 1e-9).sum()
check(T, "例2 正確な両側 p=47/1716=0.0274", ce == 47 and close(47 / 1716, .0274), ce)
n1_, n2_, N_, Sg = sp.symbols('n1 n2 N Sigma', positive=True)
lhs = n1_ * n2_ / 12 * ((N_ + 1) - Sg / (N_ * (N_ - 1)))
rhs = n1_ * n2_ * (N_ + 1) / 12 * (1 - Sg / (N_ ** 3 - N_))
check(T, "sympy タイ補正式 = n1n2(N+1)/12 × C", sp.simplify(lhs - rhs) == 0, "")
# U/(n1n2) が P(X>Y) の不偏推定量（シミュレーション）
Xs = rng.normal(0.5, 1, (100000, 4)); Ys = rng.normal(0, 1, (100000, 5))
Uh = (Xs[:, :, None] > Ys[:, None, :]).mean(axis=(1, 2))
ptrue = norm.cdf(.5 / sqrt(2))
check(T, "sim E[U/(n1n2)]=P(X>Y)（10万回, 許容±0.003）", abs(Uh.mean() - ptrue) < .003, (Uh.mean(), ptrue))

# ================= kruskal-wallis-test =================
T = "kruskal-wallis-test"
gA = [12.1, 14.3, 13.0, 15.2]; gB = [16.4, 15.8, 17.9, 14.9]; gC = [18.3, 19.0, 16.9, 20.2]
allv = np.r_[gA, gB, gC]
rk = rankdata(allv)
check(T, "例1 順位 A:1,3,2,5 B:7,6,9,4 C:10,11,8,12", list(rk) == [1, 3, 2, 5, 7, 6, 9, 4, 10, 11, 8, 12], rk)
Rs = [rk[:4].sum(), rk[4:8].sum(), rk[8:].sum()]
check(T, "例1 R=11,26,41（計78）", Rs == [11, 26, 41] and sum(Rs) == 78, Rs)
H = 12 / (12 * 13) * sum(r ** 2 / 4 for r in Rs) - 39
check(T, "例1 ΣR²=2478, H=8.654", sum(r ** 2 for r in Rs) == 2478 and close(H, 8.654, 5e-4), H)
kw = stats.kruskal(gA, gB, gC)
check(T, "例1 scipy kruskal H と p=0.0132", close(kw.statistic, H, 1e-10) and close(kw.pvalue, .0132), kw.pvalue)
check(T, "例1 χ²(2) 5%点 5.991", close(chi2.ppf(.95, 2), 5.991, 5e-4), chi2.ppf(.95, 2))
# 正確分布: 34650 通り
idx = set(range(12))
Hs = []
rks = np.arange(1, 13)
for c1 in combinations(range(12), 4):
    rest = sorted(idx - set(c1))
    s1 = rks[list(c1)].sum()
    for c2 in combinations(rest, 4):
        s2 = rks[list(c2)].sum()
        s3 = 78 - s1 - s2
        Hs.append(12 / 156 * (s1 ** 2 + s2 ** 2 + s3 ** 2) / 4 - 39)
Hs = np.array(Hs)
nge = (Hs >= H - 1e-9).sum()
check(T, "例1 群分け 12!/(4!)^3=34650 通り", len(Hs) == 34650 == factorial(12) // factorial(4) ** 3, len(Hs))
check(T, "例1 正確 H≥8.654 は48通り, p=0.0014", nge == 48 and close(nge / 34650, .0014), (nge, nge / 34650))
check(T, "例1 正確分布の E[H]=k−1=2", close(Hs.mean(), 2, 1e-9), Hs.mean())
fa = stats.f_oneway(gA, gB, gC)
check(T, "例1 一元配置 F=13.69, p=0.0019", close(fa.statistic, 13.69, 5e-3) and close(fa.pvalue, .0019), (fa.statistic, fa.pvalue))
check(T, "図 χ²(2) の H=8.654 の右側 0.0132", close(chi2.sf(8.654, 2), .0132), chi2.sf(8.654, 2))
# 例2
g1, g2, g3 = [2, 3, 3, 4], [3, 4, 4, 5], [4, 5, 5, 5]
rk2 = rankdata(np.r_[g1, g2, g3])
R2s = [rk2[:4].sum(), rk2[4:8].sum(), rk2[8:].sum()]
check(T, "例2 R=13.5, 26.5, 38", R2s == [13.5, 26.5, 38], R2s)
H2 = 12 / 156 * sum(r ** 2 / 4 for r in R2s) - 39
check(T, "例2 H=5.779", close(H2, 5.779, 5e-4), H2)
tc = Counter(np.r_[g1, g2, g3])
st = sum(c ** 3 - c for c in tc.values())
check(T, "例2 タイ 1,3,4,4, Σ=144, N³−N=1716", [tc[v] for v in (2, 3, 4, 5)] == [1, 3, 4, 4] and st == 144 and 12 ** 3 - 12 == 1716, st)
Cc = 1 - 144 / 1716
H2c = H2 / Cc
check(T, "例2 補正係数 0.9161, H'=6.308", close(Cc, .9161) and close(H2c, 6.308, 5e-4), H2c)
kw2 = stats.kruskal(g1, g2, g3)
check(T, "例2 scipy kruskal（タイ補正）と一致", close(kw2.statistic, H2c, 1e-10), kw2.statistic)
check(T, "例2 p 補正なし 0.0556, 補正あり 0.0427", close(chi2.sf(H2, 2), .0556) and close(chi2.sf(H2c, 2), .0427), (chi2.sf(H2, 2), chi2.sf(H2c, 2)))
# sympy
k3 = 3
nj = sp.symbols('n1:4', positive=True)
Nsum = sum(nj)
EH = 12 / (Nsum * (Nsum + 1)) * sum(m * (Nsum - m) * (Nsum + 1) / (12 * m) for m in nj)
check(T, "sympy E[H]=k−1（k=3, 一般の n_j）", sp.simplify(EH - 2) == 0, sp.simplify(EH))
Rj = sp.symbols('R1:4')
Rsub = {Rj[2]: Nsum * (Nsum + 1) / 2 - Rj[0] - Rj[1]}
f1 = 12 / (Nsum * (Nsum + 1)) * sum(m * (r / m - (Nsum + 1) / 2) ** 2 for m, r in zip(nj, Rj))
f2 = 12 / (Nsum * (Nsum + 1)) * sum(r ** 2 / m for m, r in zip(nj, Rj)) - 3 * (Nsum + 1)
check(T, "sympy H の2つの形が一致（ΣR_j=N(N+1)/2 のもと）", sp.simplify((f1 - f2).subs(Rsub)) == 0, "")
# k=2 で H = Z^2
m1, m2, R1s = sp.symbols('m1 m2 R1', positive=True)
NN = m1 + m2
R2e = NN * (NN + 1) / 2 - R1s
Hk2 = 12 / (NN * (NN + 1)) * (R1s ** 2 / m1 + R2e ** 2 / m2) - 3 * (NN + 1)
Z2 = (R1s - m1 * (NN + 1) / 2) ** 2 / (m1 * m2 * (NN + 1) / 12)
check(T, "sympy k=2 で H=順位和検定の Z²", sp.simplify(Hk2 - Z2) == 0, "")
check(T, "数値 例1の A と B で H=Z²（scipy）", close(stats.kruskal(gA, gB).statistic, stats.ranksums(gA, gB).statistic ** 2, 1e-10), "")

# ================= friedman-test =================
T = "friedman-test"
X = np.array([[7, 5, 6], [8, 6, 5], [9, 7, 8], [6, 4, 5], [8, 5, 7]], float)
rkb = np.vstack([rankdata(r) for r in X])
check(T, "例1 ブロック内順位 A=3,3,3,3,3 B=1,2,1,1,1 C=2,1,2,2,2", list(rkb[:, 0]) == [3] * 5 and list(rkb[:, 1]) == [1, 2, 1, 1, 1] and list(rkb[:, 2]) == [2, 1, 2, 2, 2], "")
Rf = rkb.sum(0)
check(T, "例1 R=15,6,9（計30）, ΣR²=342", list(Rf) == [15, 6, 9] and (Rf ** 2).sum() == 342, Rf)
Q = 12 / (5 * 3 * 4) * (Rf ** 2).sum() - 60
check(T, "例1 Q=0.2×342−60=8.4", close(Q, 8.4, 1e-9), Q)
fr = stats.friedmanchisquare(*X.T)
check(T, "例1 scipy friedmanchisquare Q=8.4, p=0.0150", close(fr.statistic, 8.4, 1e-9) and close(fr.pvalue, .0150), fr.pvalue)
perms3 = list(permutations([1, 2, 3]))
Qs = []
for combo in product(perms3, repeat=5):
    Rv = np.sum(combo, axis=0)
    Qs.append(round(12 / 60 * (Rv ** 2).sum() - 60, 6))
Qs = np.array(Qs)
check(T, "例1 (3!)^5=7776 通り, Q≥8.4 は66通り, p=0.0085", len(Qs) == 7776 and (Qs >= 8.4 - 1e-9).sum() == 66 and close(66 / 7776, .0085), (Qs >= 8.4 - 1e-9).sum())
cQ = Counter(Qs)
labs = ["0", "0.4", "1.2", "1.6", "2.8", "3.6", "4.8", "5.2", "6.4", "7.6", "8.4", "10"]
vals = [0.0463, 0.2623, 0.1698, 0.1543, 0.1852, 0.0579, 0.0309, 0.054, 0.0154, 0.0154, 0.0077, 0.0008]
keys = sorted(cQ)
calc = [round(cQ[q] / 7776, 4) for q in keys]
check(T, "bar-data Q のとり得る値12種", [float(l) for l in labs] == [float(q) for q in keys], keys)
check(T, "bar-data Q の正確分布 12値", calc == vals, list(zip(calc, vals)))
check(T, "bar-data キャプション 0.0077+0.0008=0.0085", close(0.0077 + 0.0008, 66 / 7776, 1e-4), "")
check(T, "正確分布の E[Q]=k−1=2", close(Qs.mean(), 2, 1e-9), Qs.mean())
check(T, "図2 χ²(2) の Q=8.4 の右側 0.0150", close(chi2.sf(8.4, 2), .0150), chi2.sf(8.4, 2))
# 例2
X2 = np.array([[5, 5, 3], [4, 6, 2], [7, 7, 4], [6, 5, 3]], float)
rkb2 = np.vstack([rankdata(r) for r in X2])
Rf2 = rkb2.sum(0)
Q2 = 12 / (4 * 3 * 4) * (Rf2 ** 2).sum() - 3 * 4 * 4
check(T, "例2 R=(10,10,4), Q=0.25×216−48=6.0", list(Rf2) == [10, 10, 4] and (Rf2 ** 2).sum() == 216 and close(Q2, 6.0, 1e-12), Rf2)
Cf = 1 - 12 / (4 * 24)
Q2c = Q2 / Cf
check(T, "例2 補正係数 0.875, Q'=6.857, p=0.0324", Cf == .875 and close(Q2c, 6.857, 5e-4) and close(chi2.sf(Q2c, 2), .0324), chi2.sf(Q2c, 2))
fr2 = stats.friedmanchisquare(*X2.T)
check(T, "例2 scipy friedmanchisquare（タイ補正）と一致", close(fr2.statistic, Q2c, 1e-10), fr2.statistic)
check(T, "例2 補正なし p=0.0498", close(chi2.sf(6.0, 2), .0498), chi2.sf(6.0, 2))
# sympy
b_, k_ = sp.symbols('b k', positive=True)
EQ = 12 / (b_ * k_ * (k_ + 1)) * k_ * b_ * (k_ ** 2 - 1) / 12
check(T, "sympy E[Q]=k−1", sp.simplify(EQ - (k_ - 1)) == 0, "")
check(T, "sympy 1ブロックの順位 E=(k+1)/2, V=(k²−1)/12（k=2..8 数え上げ）", all(close(np.var(np.arange(1, kk + 1)), (kk ** 2 - 1) / 12, 1e-12) for kk in range(2, 9)), "")
Rk = sp.symbols('Rk1:5')
for kk in (3, 4):
    Rv = list(Rk[:kk])
    sub = {Rv[-1]: b_ * kk * (kk + 1) / 2 - sum(Rv[:-1])}
    q1 = 12 / (b_ * kk * (kk + 1)) * sum((r - b_ * (kk + 1) / 2) ** 2 for r in Rv)
    q2 = 12 / (b_ * kk * (kk + 1)) * sum(r ** 2 for r in Rv) - 3 * b_ * (kk + 1)
    check(T, f"sympy Q の2つの形が一致（k={kk}）", sp.simplify((q1 - q2).subs(sub)) == 0, "")
Ssym = sp.symbols('S')
RA = 2 * Ssym + (b_ - Ssym)  # 処理1が大きいブロック数 S
RB = 3 * b_ - RA
Qk2 = 12 / (b_ * 6) * (RA ** 2 + RB ** 2) - 9 * b_
check(T, "sympy k=2 で Q=(S−b/2)²/(b/4)（両側符号検定の Z²）", sp.simplify(Qk2 - (Ssym - b_ / 2) ** 2 / (b_ / 4)) == 0, "")

# ================= spearman-rank-correlation =================
T = "spearman-rank-correlation"
yr = np.array([2, 1, 5, 3, 6, 4])
dd = np.arange(1, 7) - yr
check(T, "例1 d=(−1,1,−2,1,−1,2), Σd²=12", list(dd) == [-1, 1, -2, 1, -1, 2] and (dd ** 2).sum() == 12, dd)
rs1 = 1 - 6 * 12 / (6 * 35)
check(T, "例1 r_s=1−72/210=0.657", close(rs1, .657, 5e-4) and close(stats.spearmanr(np.arange(1, 7), yr).statistic, rs1, 1e-12), rs1)
P6 = np.array(list(permutations(range(1, 7))))
D6 = ((P6 - np.arange(1, 7)) ** 2).sum(1)
check(T, "例1 720通り中 Σd²≤12 は63通り, p=0.0875", (D6 <= 12).sum() == 63 and close(63 / 720, .0875), (D6 <= 12).sum())
z = rs1 * sqrt(5)
check(T, "例1 z=1.469, p=0.0709", close(z, 1.469, 5e-4) and close(norm.sf(z), .0709), norm.sf(z))
tv = rs1 * sqrt(4 / (1 - rs1 ** 2))
check(T, "例1 t=1.744（自由度4）, p=0.0781", close(tv, 1.744, 5e-4) and close(stats.t.sf(tv, 4), .0781), stats.t.sf(tv, 4))
pt = stats.permutation_test((yr,), lambda y: stats.spearmanr(np.arange(1, 7), y).statistic, permutation_type="pairings", alternative="greater")
check(T, "例1 scipy permutation_test（全720通り）= 0.0875", close(pt.pvalue, 63 / 720, 1e-12), pt.pvalue)
cD = Counter(D6)
labs = ["-1.000", "-0.943", "-0.886", "-0.829", "-0.771", "-0.714", "-0.657", "-0.600", "-0.543", "-0.486", "-0.429", "-0.371", "-0.314", "-0.257", "-0.200", "-0.143", "-0.086", "-0.029", "0.029", "0.086", "0.143", "0.200", "0.257", "0.314", "0.371", "0.429", "0.486", "0.543", "0.600", "0.657", "0.714", "0.771", "0.829", "0.886", "0.943", "1.000"]
vals = [0.0014, 0.0069, 0.0083, 0.0125, 0.0222, 0.0167, 0.0194, 0.0333, 0.0278, 0.0292, 0.0319, 0.0389, 0.0333, 0.0472, 0.0278, 0.0444, 0.0583, 0.0403, 0.0403, 0.0583, 0.0444, 0.0278, 0.0472, 0.0333, 0.0389, 0.0319, 0.0292, 0.0278, 0.0333, 0.0194, 0.0167, 0.0222, 0.0125, 0.0083, 0.0069, 0.0014]
keys = sorted(cD, reverse=True)
calc_l = [f"{1 - 6 * k_ / 210:.3f}" for k_ in keys]
calc_v = [round(cD[k_] / 720, 4) for k_ in keys]
check(T, "bar-data r_s のとり得る値36種（ラベル）", calc_l == labs, calc_l[:3])
check(T, "bar-data r_s の正確分布 36値", calc_v == vals, [i for i, (a_, b_) in enumerate(zip(calc_v, vals)) if a_ != b_])
check(T, "bar-data rₛ≥0.657 は右端7本で計0.0875", sum(cD[k_] for k_ in keys if 1 - 6 * k_ / 210 >= .657 - 1e-9) == 63 and len([k_ for k_ in keys if 1 - 6 * k_ / 210 >= rs1 - 1e-9]) == 7, "")
# 例2
x = np.array([52, 61, 45, 70, 58, 66, 49, 73, 55, 63.]); y = np.array([48, 65, 50, 68, 55, 60, 47, 75, 58, 62.])
Rx, Sy = rankdata(x), rankdata(y)
check(T, "例2 R, S の順位", list(Rx) == [3, 6, 1, 9, 5, 8, 2, 10, 4, 7] and list(Sy) == [2, 8, 3, 9, 4, 6, 1, 10, 5, 7], "")
sd2 = ((Rx - Sy) ** 2).sum()
rs2 = 1 - 6 * sd2 / 990
check(T, "例2 Σd²=16, r_s=0.903", sd2 == 16 and close(rs2, .903, 5e-4) and close(stats.spearmanr(x, y).statistic, rs2, 1e-12), rs2)
z = rs2 * 3
check(T, "例2 z=2.709, 両側 p=0.0067", close(z, 2.709, 5e-4) and close(2 * norm.sf(z), .0067), 2 * norm.sf(z))
tv = rs2 * sqrt(8 / (1 - rs2 ** 2))
check(T, "例2 t=5.946, p=0.0003（spearmanr と一致）", close(tv, 5.946, 5e-4) and close(2 * stats.t.sf(tv, 8), .0003) and close(stats.spearmanr(x, y).pvalue, 2 * stats.t.sf(tv, 8), 1e-12), 2 * stats.t.sf(tv, 8))
# 10! 通り
P10 = np.fromiter(__import__('itertools').chain.from_iterable(permutations(range(10))), dtype=np.int8, count=36288000).reshape(-1, 10)
D10 = ((P10.astype(np.int16) - np.arange(10)) ** 2).sum(1)
ex2 = ((D10 <= 16) | (D10 >= 330 - 16)).sum()
check(T, "例2 正確な両側 p=0.0008（10!通り）", close(ex2 / 3628800, .0008), (ex2, ex2 / 3628800))
check(T, "例2 ピアソン相関 0.918", close(np.corrcoef(x, y)[0, 1], .918, 5e-4), np.corrcoef(x, y)[0, 1])
rsD10 = 1 - 6 * D10 / 990
check(T, "数え上げ n=10 で E[r_s]=0, V[r_s]=1/9", close(rsD10.mean(), 0, 1e-9) and close(rsD10.var(), 1 / 9, 1e-9), rsD10.var())
del P10, D10, rsD10
okv = all(close((1 - 6 * ((np.array(list(permutations(range(m)))) - np.arange(m)) ** 2).sum(1) / (m * (m * m - 1))).var(), 1 / (m - 1), 1e-9) for m in range(3, 9))
check(T, "数え上げ n=3..8 で V[r_s]=1/(n−1)", okv, "")
ii = sp.symbols('i', integer=True)
sa2 = sp.summation((ii - (n + 1) / 2) ** 2, (ii, 1, n))
check(T, "sympy Σ(R−R̄)²=n(n²−1)/12", sp.simplify(sa2 - n * (n ** 2 - 1) / 12) == 0, "")
Vsum = sa2 * ((n ** 2 - 1) / 12 + (n + 1) / 12)
Vrs = (12 / (n * (n ** 2 - 1))) ** 2 * Vsum
check(T, "sympy V[r_s]=(n+1)/(n²−1)=1/(n−1)", sp.simplify(Vrs - 1 / (n - 1)) == 0 and sp.simplify(Vsum - n * (n ** 2 - 1) / 12 * n * (n + 1) / 12) == 0, sp.simplify(Vrs))
okd = all(close(1 - 6 * ((np.argsort(np.argsort(v)) - np.arange(len(v))) ** 2).sum() / (len(v) * (len(v) ** 2 - 1)), stats.spearmanr(np.arange(len(v)), v).statistic, 1e-12) for v in (rng.normal(size=(200, 9))))
check(T, "簡便式 1−6Σd²/(n(n²−1)) = 順位のピアソン相関（タイなし200組）", okd, "")

# ================= kendall-rank-correlation =================
T = "kendall-rank-correlation"
S1 = kendall_S(yr)
Dn = sum(yr[i] > yr[j] for i in range(6) for j in range(i + 1, 6))
check(T, "例1 D=4, C=11, S=7, τ=0.467", Dn == 4 and S1 == 7 and close(7 / 15, .467, 5e-4) and close(stats.kendalltau(np.arange(6), yr).statistic, 7 / 15, 1e-12), (Dn, S1))
SS6 = np.array([kendall_S(np.array(p_)) for p_ in P6])
check(T, "例1 720通り中 S≥7 は98通り, p=0.1361", (SS6 >= 7).sum() == 98 and close(98 / 720, .1361), (SS6 >= 7).sum())
kt = stats.kendalltau(np.arange(6), yr, alternative="greater", method="exact")
check(T, "例1 scipy kendalltau exact 右片側=98/720", close(kt.pvalue, 98 / 720, 1e-10), kt.pvalue)
VS6 = 6 * 5 * 17 / 18
z, zc = 7 / sqrt(VS6), 6 / sqrt(VS6)
check(T, "例1 V[S]=28.333, z=1.315, p=0.0942", close(VS6, 28.333, 5e-4) and close(z, 1.315, 5e-4) and close(norm.sf(z), .0942) and close(SS6.var(), VS6, 1e-9), norm.sf(z))
check(T, "例1 連続修正 z=1.127, p=0.1298", close(zc, 1.127, 5e-4) and close(norm.sf(zc), .1298), norm.sf(zc))
cS = Counter(SS6)
labs = list(range(-15, 16, 2))
vals = [0.0014, 0.0069, 0.0194, 0.0403, 0.0681, 0.0986, 0.125, 0.1403, 0.1403, 0.125, 0.0986, 0.0681, 0.0403, 0.0194, 0.0069, 0.0014]
calc = [round(cS[v] / 720, 4) for v in labs]
check(T, "bar-data S の正確分布 16値（奇数のみ）", calc == vals and set(cS) == set(labs), calc)
check(T, "bar-data S≥7 は右端5本", len([v for v in labs if v >= 7]) == 5, "")
# 例2
order = np.argsort(x)
yseq = rankdata(y)[order].astype(int)
check(T, "例2 理科の順位の並び 3,1,2,5,4,8,7,6,9,10", list(yseq) == [3, 1, 2, 5, 4, 8, 7, 6, 9, 10], yseq)
D2 = sum(yseq[i] > yseq[j] for i in range(10) for j in range(i + 1, 10))
S2 = kendall_S(yseq)
check(T, "例2 D=6, C=39, S=33, τ=0.733", D2 == 6 and S2 == 33 and close(33 / 45, .733, 5e-4) and close(stats.kendalltau(x, y).statistic, 33 / 45, 1e-12), (D2, S2))
z, zc = 33 / sqrt(125), 32 / sqrt(125)
check(T, "例2 V[S]=125, z=2.952, p=0.0032", 10 * 9 * 25 / 18 == 125 and close(z, 2.952, 5e-4) and close(2 * norm.sf(z), .0032), 2 * norm.sf(z))
check(T, "例2 連続修正 z=2.862, p=0.0042", close(zc, 2.862, 5e-4) and close(2 * norm.sf(zc), .0042), 2 * norm.sf(zc))
# 転倒数分布（マホニアン数）で n=10 の正確分布
def inv_counts(m):
    f = [1]
    for i in range(1, m + 1):
        g = [0] * (len(f) + i - 1)
        for a_, v in enumerate(f):
            for s_ in range(i):
                g[a_ + s_] += v
        f = g
    return f
ic = inv_counts(10)
pex = sum(c for dv, c in enumerate(ic) if abs(45 - 2 * dv) >= 33) / factorial(10)
kt2 = stats.kendalltau(x, y, method="exact")
check(T, "例2 正確な両側 p=0.0022（転倒数分布・scipy exact）", close(pex, .0022) and close(kt2.pvalue, pex, 1e-10), pex)
# 分散公式
ok = all(close(np.var([2 * (comb(m, 2) - dv) - comb(m, 2) for dv, c in enumerate(inv_counts(m)) for _ in range(c)]), m * (m - 1) * (2 * m + 5) / 18, 1e-9) for m in range(2, 9))
check(T, "数え上げ n=2..8 で V[S]=n(n−1)(2n+5)/18", ok, "")
check(T, "sympy C(n,2)+2C(n,3)/3=n(n−1)(2n+5)/18", sp.simplify(n * (n - 1) / 2 + 2 * n * (n - 1) * (n - 2) / 6 / 3 - n * (n - 1) * (2 * n + 5) / 18) == 0, "")
check(T, "sympy V[τ]=4V[S]/(n(n−1))²=2(2n+5)/(9n(n−1))", sp.simplify(4 * n * (n - 1) * (2 * n + 5) / 18 / (n * (n - 1)) ** 2 - 2 * (2 * n + 5) / (9 * n * (n - 1))) == 0, "")
P3 = np.array(list(permutations([0, 1, 2])))
s01, s02, s12 = np.sign(P3[:, 1] - P3[:, 0]), np.sign(P3[:, 2] - P3[:, 0]), np.sign(P3[:, 2] - P3[:, 1])
cv = lambda u, v: float((u * v).mean() - u.mean() * v.mean())
check(T, "共分散 (i,j)(i,k)=+1/3, (i,k)(j,k)=+1/3, (i,j)(j,k)=−1/3", close(cv(s01, s02), 1 / 3, 1e-12) and close(cv(s02, s12), 1 / 3, 1e-12) and close(cv(s01, s12), -1 / 3, 1e-12), (cv(s01, s02), cv(s02, s12), cv(s01, s12)))
# τ = 2/π arcsin ρ
Z1 = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 400000)
Z2_ = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 400000)
tau_hat = np.mean(np.sign((Z1[:, 0] - Z2_[:, 0]) * (Z1[:, 1] - Z2_[:, 1])))
check(T, "sim 2変量正規 ρ=0.6 で τ=2/π·arcsin ρ=0.4097（40万組, 許容±0.005）", abs(tau_hat - 2 / pi * asin(.6)) < .005, (tau_hat, 2 / pi * asin(.6)))
check(T, "例1・2 で |τ|<|r_s|", 7 / 15 < rs1 and 33 / 45 < rs2, "")

# ================= concordant-discordant-pairs =================
T = "concordant-discordant-pairs"
cx = np.array([1, 2, 2, 3, 4, 4]); cy = np.array([1, 3, 2, 2, 4, 3])
cls = Counter()
pairs = {}
for i in range(6):
    for j_ in range(i + 1, 6):
        dx, dy = cx[j_] - cx[i], cy[j_] - cy[i]
        kind = "C" if dx * dy > 0 else "D" if dx * dy < 0 else "Tx" if dx == 0 and dy != 0 else "Ty" if dy == 0 and dx != 0 else "Txy"
        cls[kind] += 1
        pairs.setdefault(kind, []).append((i + 1, j_ + 1))
check(T, "例1 C=10, D=1, Tx=2, Ty=2, Txy=0（bar-data と一致）", [cls[k_] for k_ in ("C", "D", "Tx", "Ty", "Txy")] == [10, 1, 2, 2, 0], dict(cls))
check(T, "例1 xだけタイ (2,3),(5,6); yだけタイ (3,4),(2,6); 不一致 (2,4)", pairs["Tx"] == [(2, 3), (5, 6)] and sorted(pairs["Ty"]) == [(2, 6), (3, 4)] and pairs["D"] == [(2, 4)], pairs)
ta, tb, gm = 9 / 15, 9 / sqrt(13 * 13), 9 / 11
check(T, "例1 τa=0.600, τb=0.692, γ=0.818", close(ta, .6, 5e-4) and close(tb, .692, 5e-4) and close(gm, .818, 5e-4), (ta, tb, gm))
check(T, "例1 scipy kendalltau（τb）と一致", close(stats.kendalltau(cx, cy).statistic, tb, 1e-12), stats.kendalltau(cx, cy).statistic)
check(T, "例1 |γ|≥|τb|≥|τa|", gm >= tb >= ta, "")
# 大小関係のランダム確認
okr = True
for _ in range(300):
    u = rng.integers(1, 4, 12); v = rng.integers(1, 4, 12)
    C_ = D_ = n1c = n2c = 0
    for i in range(12):
        for j_ in range(i + 1, 12):
            s_ = (u[j_] - u[i]) * (v[j_] - v[i])
            C_ += s_ > 0; D_ += s_ < 0; n1c += u[i] == u[j_]; n2c += v[i] == v[j_]
    if C_ + D_ == 0 or n1c == 66 or n2c == 66:
        continue
    n0 = 66
    A_, B_, G_ = abs(C_ - D_) / n0, abs(C_ - D_) / sqrt((n0 - n1c) * (n0 - n2c)), abs(C_ - D_) / (C_ + D_)
    okr &= G_ >= B_ - 1e-12 and B_ >= A_ - 1e-12
check(T, "ランダム表300件で |γ|≥|τb|≥|τa|", okr, "")
# U と一致対
grp = np.r_[np.ones(4), np.zeros(5)]; val = np.r_[xa, ya]
Cg = sum((grp[i] - grp[j_]) * (val[i] - val[j_]) > 0 for i in range(9) for j_ in range(i + 1, 9))
check(T, "群(0/1)×値の一致対の数=U=17", Cg == 17, Cg)

# ================= rank-and-ties =================
T = "rank-and-ties"
v8 = np.array([3.2, 1.5, 4.0, 3.2, 2.8, 4.0, 4.0, 5.1])
r8 = rankdata(v8)
check(T, "例1 平均順位 3.5,1,6,3.5,2,6,6,8, 合計36", list(r8) == [3.5, 1, 6, 3.5, 2, 6, 6, 8] and r8.sum() == 36, r8)
check(T, "bar-data 値と順位", list(np.sort(v8)) == [1.5, 2.8, 3.2, 3.2, 4.0, 4.0, 4.0, 5.1] and list(np.sort(r8)) == [1, 2, 3.5, 3.5, 6, 6, 6, 8], "")
check(T, "例1 Σ(t³−t)=6+24=30", sum(c ** 3 - c for c in Counter(v8).values()) == 30, "")
check(T, "例1 V[R]=5.25−0.3125=4.9375（実際の分散と一致）", close(63 / 12 - 30 / 96, 4.9375, 1e-12) and close(r8.var(), 4.9375, 1e-12), r8.var())
check(T, "例1 C=1−30/504=0.9405", close(1 - 30 / 504, .9405), 1 - 30 / 504)
Nn = sp.symbols('N', positive=True, integer=True)
ER = sp.summation(ii, (ii, 1, Nn)) / Nn
ER2 = sp.summation(ii ** 2, (ii, 1, Nn)) / Nn
check(T, "sympy E[R]=(N+1)/2, E[R²]=(N+1)(2N+1)/6, V=(N²−1)/12", sp.simplify(ER - (Nn + 1) / 2) == 0 and sp.simplify(ER2 - (Nn + 1) * (2 * Nn + 1) / 6) == 0 and sp.simplify(ER2 - ER ** 2 - (Nn ** 2 - 1) / 12) == 0, "")
check(T, "sympy 平均順位 (a+1+…+a+t)/t=a+(t+1)/2", sp.simplify(sp.summation(a + j, (j, 1, t)) / t - (a + (t + 1) / 2)) == 0, "")
check(T, "sympy t 個の連続整数の分散=(t²−1)/12", sp.simplify(sp.summation(j ** 2, (j, 1, t)) / t - (sp.summation(j, (j, 1, t)) / t) ** 2 - (t ** 2 - 1) / 12) == 0, "")
okt = True
for _ in range(300):
    vv = rng.integers(1, 6, rng.integers(5, 30))
    NN_ = len(vv)
    st_ = sum(c ** 3 - c for c in Counter(vv).values())
    okt &= close(rankdata(vv).var(), (NN_ ** 2 - 1) / 12 - st_ / (12 * NN_), 1e-9) and close(rankdata(vv).mean(), (NN_ + 1) / 2, 1e-12)
check(T, "ランダム300件でタイあり V[R] 公式・平均 (N+1)/2", okt, "")
check(T, "例2 補正なし V=49.0, 補正あり 45.5", 6 * 7 * 14 / 12 == 49 and close(VU, 45.5, 1e-12), "")
check(T, "例2 補正あり z=2.150 p=0.0316, 補正なし z=2.071 p=0.0383", close(14.5 / sqrt(45.5), 2.150, 5e-4) and close(2 * norm.sf(14.5 / sqrt(45.5)), .0316) and close(14.5 / 7, 2.071, 5e-4) and close(2 * norm.sf(14.5 / 7), .0383), 2 * norm.sf(14.5 / 7))

# ================= permutation-test =================
T = "permutation-test"
pn, po = np.array([23, 27, 30, 35.]), np.array([18, 20, 22, 26.])
check(T, "例1 平均 28.75, 21.5, 差 7.25", pn.mean() == 28.75 and po.mean() == 21.5 and pn.mean() - po.mean() == 7.25, "")
zz = np.r_[pn, po]
diffs = []
grp1 = []
for c in combinations(range(8), 4):
    m_ = np.zeros(8, bool); m_[list(c)] = True
    diffs.append(zz[m_].mean() - zz[~m_].mean()); grp1.append(sorted(zz[m_]))
diffs = np.array(diffs)
check(T, "例1 70通り, 7.25以上は2通り（7.25 と {26,27,30,35} の 8.75）", len(diffs) == 70 and (diffs >= 7.25 - 1e-9).sum() == 2 and sorted(diffs[diffs >= 7.25 - 1e-9]) == [7.25, 8.75] and [26, 27, 30, 35] in [g for g, dv in zip(grp1, diffs) if dv == 8.75], "")
check(T, "例1 片側 p=2/70=0.0286, 両側 4/70=0.0571", close(2 / 70, .0286) and (np.abs(diffs) >= 7.25 - 1e-9).sum() == 4 and close(4 / 70, .0571), "")
ptest = stats.permutation_test((pn, po), lambda a1, a2: a1.mean() - a2.mean(), permutation_type="independent", alternative="greater")
check(T, "例1 scipy permutation_test 片側 = 2/70", close(ptest.pvalue, 2 / 70, 1e-12), ptest.pvalue)
cdiff = Counter(np.round(diffs, 2))
labs = [-8.75, -7.25, -6.75, -6.25, -5.75, -5.25, -4.75, -4.25, -3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 8.75]
vals = [1, 1, 2, 1, 1, 2, 3, 1, 2, 2, 4, 3, 2, 4, 3, 3, 3, 3, 4, 2, 3, 4, 2, 2, 1, 3, 2, 1, 1, 2, 1, 1]
check(T, "bar-data 並べ替え分布のラベル32種", sorted(cdiff) == labs, sorted(cdiff))
check(T, "bar-data 並べ替え分布の通り数（計70）", [cdiff[v] for v in labs] == vals and sum(vals) == 70, [cdiff[v] for v in labs])
Tsum = pn.sum(); zbar = zz.mean(); ss = ((zz - zbar) ** 2).sum()
VT = 16 / 56 * ss
check(T, "例1 T=115, E=4×25.125=100.5, Σ(z−z̄)²=216.875, V=61.964", Tsum == 115 and zbar == 25.125 and ss == 216.875 and close(VT, 61.964, 5e-4), VT)
sums = np.array([zz[list(c)].sum() for c in combinations(range(8), 4)])
check(T, "例1 並べ替え分布の E[T], V[T] が公式と一致", close(sums.mean(), 100.5, 1e-9) and close(sums.var(), VT, 1e-9), sums.var())
zp = 14.5 / sqrt(VT)
check(T, "例1 z=1.842, p=0.0327", close(zp, 1.842, 5e-4) and close(norm.sf(zp), .0327), norm.sf(zp))
tt = stats.ttest_ind(pn, po, alternative="greater")
check(T, "例1 t=2.376（自由度6）, p=0.0275", close(tt.statistic, 2.376, 5e-4) and close(tt.pvalue, .0275), (tt.statistic, tt.pvalue))
Di = np.round(E1 * 10).astype(int)
sf = np.array([np.dot(s_, np.abs(Di)) for s_ in product([1, -1], repeat=8)])
check(T, "例2 合計 10.1, 符号反転で ≥10.1 は8通り, p=8/256=0.0313", Di.sum() == 101 and (sf >= 101).sum() == 8 and close(8 / 256, .0313, 5.01e-5), (sf >= 101).sum())
pt2 = stats.permutation_test((E1,), np.sum, permutation_type="samples", alternative="greater")
check(T, "例2 scipy permutation_test（samples）=8/256", close(pt2.pvalue, 8 / 256, 1e-12), pt2.pvalue)
check(T, "例2 符号付き順位の正確 p=0.0273", close(7 / 256, .0273), "")
check(T, "順位で並べ替え検定＝順位和検定（例1の順位で p=7/126）", close((ranksum_dist(rankdata(np.r_[xa, ya]), 4) >= 27).mean(), 7 / 126, 1e-12), "")

# ================= normal-approximation-rank-tests =================
T = "normal-approximation-rank-tests"
check(T, "表 符号検定 E=n/2, V=n/4（Bin 数え上げ n=1..20）", all(close(binom.var(m, .5), m / 4, 1e-12) for m in range(1, 21)), "")
check(T, "表 U の E=n1n2/2, V=n1n2(N+1)/12（数え上げ）", all(close((ranksum_dist(np.arange(1, a1 + b1 + 1), a1) - a1 * (a1 + 1) / 2).mean(), a1 * b1 / 2, 1e-9) and close(ranksum_dist(np.arange(1, a1 + b1 + 1), a1).var(), a1 * b1 * (a1 + b1 + 1) / 12, 1e-9) for a1 in range(1, 6) for b1 in range(1, 6)), "")
check(T, "例1 W=128: 正確 0.0446, 修正あり z=1.701 p=0.0445, なし z=1.739 p=0.0410, √V=13.229", close(pe2, .0446) and close(sqrt(175), 13.229, 5e-4) and close(norm.sf(22.5 / sqrt(175)), .0445) and close(norm.sf(23 / sqrt(175)), .0410), "")
check(T, "例2 符号付き順位 0.0273 / 0.0294 / 0.0250", close(7 / 256, .0273) and close(norm.sf(13.5 / sqrt(51)), .0294) and close(norm.sf(14 / sqrt(51)), .0250), "")
check(T, "例2 順位和 0.0556 / 0.0557 / 0.0432", close(7 / 126, .0556) and close(norm.sf(6.5 / sqrt(50 / 3)), .0557) and close(norm.sf(7 / sqrt(50 / 3)), .0432), "")
check(T, "例2 ケンドール 0.1361 / 0.1298 / 0.0942", close(98 / 720, .1361) and close(norm.sf(6 / sqrt(VS6)), .1298) and close(norm.sf(7 / sqrt(VS6)), .0942), "")
check(T, "例2 順位和で5%判定が逆転（0.0557>0.05>0.0432）", norm.sf(6.5 / sqrt(50 / 3)) > .05 > norm.sf(7 / sqrt(50 / 3)), "")
check(T, "図 Bin(10,1/2) の正規近似 N(5, 2.5)", binom.mean(10, .5) == 5 and binom.var(10, .5) == 2.5, "")

# ================= nonparametric-methods-overview =================
T = "nonparametric-methods-overview"
r_ = stats.ttest_ind(pn, po, alternative="greater")
mwo = stats.mannwhitneyu(pn, po, alternative="greater", method="exact")
check(T, "例1 t 検定 p=0.0275, U 検定（正確）p=0.0286", close(r_.pvalue, .0275) and close(mwo.pvalue, .0286) and close(mwo.pvalue, 2 / 70, 1e-12), (r_.pvalue, mwo.pvalue))
pn2 = np.array([23, 27, 30, 95.])
r2_ = stats.ttest_ind(pn2, po, alternative="greater")
check(T, "例1' 平均 28.75→43.75", pn2.mean() == 43.75, "")
check(T, "例1' t=1.291, p=0.1220", close(r2_.statistic, 1.291, 5e-4) and close(r2_.pvalue, .1220), (r2_.statistic, r2_.pvalue))
rw = stats.ttest_ind(pn2, po, alternative="greater", equal_var=False)
check(T, "例1' ウェルチ p=0.1427", close(rw.pvalue, .1427), rw.pvalue)
mwo2 = stats.mannwhitneyu(pn2, po, alternative="greater", method="exact")
check(T, "例1' U 検定 p=0.0286 のまま", close(mwo2.pvalue, mwo.pvalue, 1e-15), mwo2.pvalue)
check(T, "bar-data 0.0275, 0.122, 0.0286, 0.0286", close(r_.pvalue, .0275) and close(r2_.pvalue, .122, 5e-4) and close(mwo.pvalue, .0286), "")
xs = sp.symbols('x', real=True)
fN = sp.exp(-xs ** 2 / 2) / sp.sqrt(2 * sp.pi)
are_w_N = 12 * 1 * sp.integrate(fN ** 2, (xs, -sp.oo, sp.oo)) ** 2
are_s_N = 4 * 1 * fN.subs(xs, 0) ** 2
check(T, "sympy ARE 正規: 順位 12σ²(∫f²)²=3/π, 符号 4σ²f(0)²=2/π", sp.simplify(are_w_N - 3 / sp.pi) == 0 and sp.simplify(are_s_N - 2 / sp.pi) == 0, (are_w_N, are_s_N))
fL = sp.exp(-xs) / 2  # x>0 側、対称
intf2 = 2 * sp.integrate(fL ** 2, (xs, 0, sp.oo))
are_w_L = 12 * 2 * intf2 ** 2
are_s_L = 4 * 2 * sp.Rational(1, 2) ** 2
check(T, "sympy ARE ラプラス: 順位 1.5, 符号 2", sp.simplify(are_w_L - sp.Rational(3, 2)) == 0 and are_s_L == 2, (are_w_L, are_s_L))
fE = sp.Rational(3, 4) / sp.sqrt(5) * (1 - xs ** 2 / 5)  # |x|≤√5 の放物線密度
tot_ = sp.integrate(fE, (xs, -sp.sqrt(5), sp.sqrt(5)))
var_ = sp.integrate(xs ** 2 * fE, (xs, -sp.sqrt(5), sp.sqrt(5)))
are_min = 12 * var_ * sp.integrate(fE ** 2, (xs, -sp.sqrt(5), sp.sqrt(5))) ** 2
check(T, "sympy ARE 下限（放物線密度）=108/125=0.864", tot_ == 1 and sp.simplify(are_min - sp.Rational(108, 125)) == 0, are_min)
check(T, "1/0.955≈1.05", close(1 / .955, 1.05, 5e-3), 1 / .955)
cnts = Counter(map(tuple, np.argsort(np.argsort(rng.exponential(size=(200000, 4)), 1), 1)))
dev = max(abs(c / 200000 - 1 / 24) for c in cnts.values())
check(T, "sim 順位ベクトルは N! 通り等確率（N=4, 指数分布20万回, 各 1/24 ±0.003）", dev < .003 and len(cnts) == 24, dev)

# ================= 重要度Cの用語 =================
print("\n--- 重要度Cの用語 ---")
rngC = np.random.default_rng(20261023)

T = "asymptotic-relative-efficiency"
sg = sp.symbols("sigma", positive=True)
fNs = sp.exp(-xs ** 2 / (2 * sg ** 2)) / (sp.sqrt(2 * sp.pi) * sg)
int_f2 = sp.simplify(sp.integrate(fNs ** 2, (xs, -sp.oo, sp.oo)))
check(T, "sympy 正規 ∫f²=1/(2√π σ), f(m)=1/(√(2π)σ)", sp.simplify(int_f2 - 1 / (2 * sp.sqrt(sp.pi) * sg)) == 0 and sp.simplify(fNs.subs(xs, 0) - 1 / (sp.sqrt(2 * sp.pi) * sg)) == 0, int_f2)
check(T, "sympy 12σ²(∫f²)²=3/π, 4σ²f(m)²=2/π（σ によらない）", sp.simplify(12 * sg ** 2 * int_f2 ** 2 - 3 / sp.pi) == 0 and sp.simplify(4 * sg ** 2 * fNs.subs(xs, 0) ** 2 - 2 / sp.pi) == 0, "一致")
check(T, "3/π≈0.955, 2/π≈0.637", close(3 / pi, .955, 5e-4) and close(2 / pi, .637, 5e-4), (3 / pi, 2 / pi))
bL = sp.symbols("b", positive=True)
fLs = sp.exp(-sp.Abs(xs) / bL) / (2 * bL)
i2 = 2 * sp.integrate((sp.exp(-xs / bL) / (2 * bL)) ** 2, (xs, 0, sp.oo))
check(T, "sympy ラプラス（尺度 b）: 順位 12·2b²·(∫f²)²=1.5, 符号 4·2b²·f(0)²=2", sp.simplify(12 * 2 * bL ** 2 * i2 ** 2 - sp.Rational(3, 2)) == 0 and sp.simplify(4 * 2 * bL ** 2 * (1 / (2 * bL)) ** 2 - 2) == 0, (i2,))
check(T, "例 100/(3/π)=104.7→105, 100/0.6366=100/(2/π)=157.1→158", close(100 / (3 / pi), 104.7, .05) and close(100 / (2 / pi), 157.1, .05) and close(100 / .6366, 157.1, .05) and int(np.ceil(100 / (3 / pi))) == 105 and int(np.ceil(100 / (2 / pi))) == 158, (100 / (3 / pi), 100 / (2 / pi)))
check(T, "例 約1.6倍（π/2=1.571）、ラプラスで符号検定は 100/2=50 人", close(pi / 2, 1.6, .05) and 100 / 2 == 50, pi / 2)
check(T, "中央値検定の順位和検定に対する効率 (2/π)/(3/π)=2/3", close((2 / pi) / (3 / pi), 2 / 3, 1e-12), 2 / 3)
# 近接対立での検出力 sim（正規, n_t=100 と n_W=105 で同程度）: 省略せず簡単に確認
nsim = 4000
def pw(nn, test):
    d = 0.25  # 平均差（σ=1）、各群 nn
    x = rngC.normal(0, 1, (nsim, nn)); yv = rngC.normal(d, 1, (nsim, nn))
    if test == "t":
        return (stats.ttest_ind(yv, x, axis=1, alternative="greater").pvalue < .05).mean()
    allv = np.concatenate([x, yv], 1)
    rk = stats.rankdata(allv, axis=1)
    W = rk[:, nn:].sum(1)
    mu = nn * (2 * nn + 1) / 2; sd = np.sqrt(nn * nn * (2 * nn + 1) / 12)
    return ((W - mu) / sd > stats.norm.ppf(.95)).mean()
pt_, pw_ = pw(100, "t"), pw(105, "w")
check(T, "sim 正規: t 検定 n=100 と順位和 n=105 の検出力がほぼ同じ（各4000回, 許容 0.04）", abs(pt_ - pw_) < .04, (pt_, pw_))

T = "median-test"
A_ = np.array([12, 15, 9, 20, 18, 14, 22, 17, 25, 19]); B_ = np.array([8, 11, 13, 10, 16, 7, 12, 9, 14, 6])
M_ = np.median(np.concatenate([A_, B_]))
tab = np.array([[(A_ > M_).sum(), (A_ <= M_).sum()], [(B_ > M_).sum(), (B_ <= M_).sum()]])
check(T, "例 M=13.5、表 [[8,2],[2,8]]", M_ == 13.5 and tab.tolist() == [[8, 2], [2, 8]], (M_, tab.tolist()))
c0 = stats.chi2_contingency(tab, correction=False); c1 = stats.chi2_contingency(tab, correction=True)
check(T, "例 期待度数すべて5、χ²=7.2, p=0.0073", np.all(c0.expected_freq == 5) and close(c0.statistic, 7.2, 1e-10) and close(c0.pvalue, .0073), (c0.statistic, c0.pvalue))
check(T, "例 イェーツ χ²=5.0, p=0.0253", close(c1.statistic, 5.0, 1e-10) and close(c1.pvalue, .0253), (c1.statistic, c1.pvalue))
fe = stats.fisher_exact(tab)
hy = stats.hypergeom(20, 10, 10)
p_hy = sum(hy.pmf(k) for k in range(11) if hy.pmf(k) <= hy.pmf(8) * (1 + 1e-9))
check(T, "例 フィッシャー両側 p=0.0230（scipy・超幾何の直接計算）", close(fe.pvalue, .0230) and close(p_hy, .0230), (fe.pvalue, p_hy))
mt = stats.median_test(A_, B_, ties="below", correction=False)
check(T, "scipy median_test（ties=below）と一致", close(mt.statistic, 7.2, 1e-10) and mt.median == 13.5, mt.statistic)

T = "kolmogorov-smirnov-two-sample"
X_ = np.array([3.1, 4.5, 2.8, 5.0, 4.2, 3.9, 4.8, 5.5]); Y_ = np.array([2.1, 2.9, 3.3, 1.8, 2.6, 3.7, 2.4, 3.0])
def Dstat(x, y):
    g = np.concatenate([x, y])
    return max(abs((x <= v).mean() - (y <= v).mean()) for v in g)
check(T, "例 x=3.7 で F8=2/8, G8=1、D=0.75", (X_ <= 3.7).mean() == .25 and (Y_ <= 3.7).mean() == 1 and close(Dstat(X_, Y_), .75, 1e-12), Dstat(X_, Y_))
check(T, "例 漸近の棄却限界 1.358√(16/64)=0.679, 1.358 は K の 0.95 分位", close(1.358 * sqrt(16 / 64), .679, 5e-4) and close(stats.kstwobign.ppf(.95), 1.358, 5e-4), stats.kstwobign.ppf(.95))
# 正確な分布：合併16個の並び C(16,8)=12870 通りを全数え上げ
Dcnt = Counter()
for S in combinations(range(16), 8):
    s = set(S); fx = gy = 0; dm = 0
    for i in range(16):
        if i in s: fx += 1
        else: gy += 1
        dm = max(dm, abs(fx - gy))
    Dcnt[dm] += 1
tot = sum(Dcnt.values())
pe75 = sum(v for k, v in Dcnt.items() if k >= 6) / tot; pe625 = sum(v for k, v in Dcnt.items() if k >= 5) / tot
check(T, "全数え上げ（12870通り）P(D≥0.75)=0.0186, P(D≥0.625)=0.0870", tot == 12870 and close(pe75, .0186) and close(pe625, .0870), (pe75, pe625))
ks = stats.ks_2samp(X_, Y_, method="exact")
check(T, "scipy ks_2samp exact p=0.0186", close(ks.pvalue, .0186) and close(ks.statistic, .75, 1e-12), ks.pvalue)
lam = sqrt(64 / 16) * .75
pas = 2 * sum((-1) ** (j - 1) * np.exp(-2 * j * j * lam * lam) for j in range(1, 101))
check(T, "漸近 p=2Σ(−1)^{j−1}e^{−2j²λ²}, λ=1.5 → 0.0222（kstwobign とも）", close(pas, .0222) and close(stats.kstwobign.sf(lam), pas, 1e-12), pas)
check(T, "m=n=8 の5%棄却域は D≥0.75（0.0186≤0.05<0.0870）", pe75 <= .05 < pe625, "")
# 帰無分布が F によらない（sim）
d1 = [Dstat(rngC.normal(size=8), rngC.normal(size=8)) for _ in range(20000)]
d2 = [Dstat(rngC.exponential(size=8), rngC.exponential(size=8)) for _ in range(20000)]
r1, r2 = np.mean(np.array(d1) >= .75 - 1e-9), np.mean(np.array(d2) >= .75 - 1e-9)
check(T, "sim 正規・指数とも P(D≥0.75)≈0.0186（各2万回, 許容 0.004）", abs(r1 - pe75) < .004 and abs(r2 - pe75) < .004, (r1, r2))

T = "runs-test"
seq = "G" * 5 + "B" * 3 + "G" * 6 + "B" * 4 + "G" * 2
def nruns(s):
    return 1 + sum(s[i] != s[i - 1] for i in range(1, len(s)))
n1_, n2_ = seq.count("G"), seq.count("B"); N_ = n1_ + n2_
check(T, "例 n1=13, n2=7, R=5", (n1_, n2_, nruns(seq)) == (13, 7, 5), (n1_, n2_, nruns(seq)))
ER = 2 * n1_ * n2_ / N_ + 1; VR = 2 * n1_ * n2_ * (2 * n1_ * n2_ - N_) / (N_ ** 2 * (N_ - 1))
check(T, "例 E[R]=10.1, V[R]=3.879, √V=1.970", close(ER, 10.1, 1e-12) and close(VR, 3.879, 5e-4) and close(sqrt(VR), 1.970, 5e-4), (ER, VR, sqrt(VR)))
z0 = (5 - ER) / sqrt(VR); z1 = (5.5 - ER) / sqrt(VR)
check(T, "例 z=−2.589, p=0.0048; 連続修正 z=−2.335, p=0.0098", close(z0, -2.589, 5e-4) and close(stats.norm.cdf(z0), .0048) and close(z1, -2.335, 5e-4) and close(stats.norm.cdf(z1), .0098), (z0, stats.norm.cdf(z0), z1, stats.norm.cdf(z1)))
Rc = Counter()
for Bpos in combinations(range(20), 7):
    s = ["G"] * 20
    for i in Bpos: s[i] = "B"
    Rc[nruns(s)] += 1
totR = sum(Rc.values())
pR5 = sum(v for k, v in Rc.items() if k <= 5) / totR
check(T, "全数え上げ（C(20,7)=77520通り）P(R≤5)=0.0095", totR == 77520 and close(pR5, .0095), pR5)
mR = sum(k * v for k, v in Rc.items()) / totR; vR = sum(k * k * v for k, v in Rc.items()) / totR - mR ** 2
check(T, "全数え上げの平均・分散が E[R]・V[R] の式と一致", close(mR, ER, 1e-10) and close(vR, VR, 1e-10), (mR, vR))
ok_ = all(Rc[2 * k] == 2 * comb(n1_ - 1, k - 1) * comb(n2_ - 1, k - 1) for k in range(1, 8))
check(T, "P(R=2k)=2C(n1−1,k−1)C(n2−1,k−1)/C(N,n1) が全数え上げと一致", ok_, {k: Rc[2 * k] for k in range(1, 8)})
check(T, "隣が異なる確率 2n1n2/{N(N−1)}×(N−1)+1 = E[R]", close(1 + (N_ - 1) * 2 * n1_ * n2_ / (N_ * (N_ - 1)), ER, 1e-12), "")

T = "jonckheere-terpstra-test"
G3 = [np.array([3.2, 4.1, 2.8, 3.6]), np.array([4.0, 4.9, 3.9, 5.3]), np.array([5.1, 6.0, 4.7, 5.6])]
def Uij(a, b):
    return sum((x < y) + .5 * (x == y) for x in a for y in b)
U12, U13, U23 = Uij(G3[0], G3[1]), Uij(G3[0], G3[2]), Uij(G3[1], G3[2])
check(T, "例 U=14, 16, 13, J=43", (U12, U13, U23) == (14, 16, 13), (U12, U13, U23))
nn_ = [4, 4, 4]; NN = 12
EJ = (NN ** 2 - sum(k * k for k in nn_)) / 4; VJ = (NN ** 2 * (2 * NN + 3) - sum(k * k * (2 * k + 3) for k in nn_)) / 72
zJ = (43 - EJ) / sqrt(VJ)
check(T, "例 E[J]=24, V[J]=46.67, √V=6.831, z=2.781, p=0.0027", EJ == 24 and close(VJ, 46.67, 5e-3) and close(sqrt(VJ), 6.831, 5e-4) and close(zJ, 2.781, 5e-4) and close(stats.norm.sf(zJ), .0027), (VJ, zJ, stats.norm.sf(zJ)))
allv = np.concatenate(G3)
Jc = Counter()
for a in combinations(range(12), 4):
    rest = [i for i in range(12) if i not in a]
    for b in combinations(rest, 4):
        c = [i for i in rest if i not in b]
        g = [allv[list(a)], allv[list(b)], allv[c]]
        Jc[Uij(g[0], g[1]) + Uij(g[0], g[2]) + Uij(g[1], g[2])] += 1
totJ = sum(Jc.values())
pJ = sum(v for k, v in Jc.items() if k >= 43) / totJ
check(T, "全数え上げ（34650通り）正確な上側 p=0.0020", totJ == 34650 and close(pJ, .0020), pJ)
mJ = sum(k * v for k, v in Jc.items()) / totJ; vJ = sum(k * k * v for k, v in Jc.items()) / totJ - mJ ** 2
check(T, "全数え上げの平均・分散が E[J]・V[J] の式と一致", close(mJ, EJ, 1e-9) and close(vJ, VJ, 1e-9), (mJ, vJ))
kw = stats.kruskal(*G3)
check(T, "同じデータのクラスカル・ウォリス p=0.0296", close(kw.pvalue, .0296), kw.pvalue)
check(T, "異なる群どうしの組は (N²−Σn²)/2=48 組", (NN ** 2 - 48) // 2 == 4 * 4 * 3, "48")

# --- 統合候補からの論点の追記: wilcoxon-rank-sum-test ---
_X = [14, 19, 22, 25]; _Y = [8, 11, 13, 16, 20]
_U = sum((x > y) + 0.5 * (x == y) for x in _X for y in _Y)
_Up = sum((y > x) + 0.5 * (x == y) for x in _X for y in _Y)
check("wilcoxon-rank-sum-test", "例1 U=17, U'=3, U+U'=n1n2=20, U/(n1n2)=0.85, U=W−10", _U == 17 and _Up == 3 and _U + _Up == 20 and abs(_U / 20 - 0.85) < 1e-12 and 27 - 4 * 5 / 2 == _U, (_U, _Up))
_auc = np.mean([[x > y for y in _Y] for x in _X])
check("wilcoxon-rank-sum-test", "U/(n1n2) は AUC（全組で X>Y の割合）と一致", abs(_auc - 0.85) < 1e-12, _auc)

# ================= 集計 =================
nok, nng = sum(results), len(results) - sum(results)
print(f"\n合計: OK {nok} / NG {nng}")
