"""第11章 検算スクリプト（正規母集団の検定：平均・分散・相関・多変量）。
数値は scipy.stats（検定統計量・p値・分位点・信頼区間・非心t/F による検出力）、
記号計算は sympy（ウェルチ・サタースウェイトの自由度、プールの最適重み、デルタ法）。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（モンテカルロ標準誤差の数倍）。
"""
from math import sqrt, atanh, tanh, log
import numpy as np
import sympy as sp
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def series_ok(xs, ys, f, tol):
    err = max(abs(f(x) - y) for x, y in zip(xs, ys))
    return err <= tol, err


rng = np.random.default_rng(20261023)
t, norm, chi2, F, nct, ncf = stats.t, stats.norm, stats.chi2, stats.f, stats.nct, stats.ncf
Phi = norm.cdf

# ================= one-sample-t-test =================
T = "one-sample-t-test"
d = np.array([48.2, 51.5, 49.8, 53.1, 50.6, 52.4, 47.9, 51.8, 52.9, 50.3])
xb, s2 = d.mean(), d.var(ddof=1)
se = sqrt(s2 / 10)
tv = (xb - 50) / se
check(T, "例1 x̄=50.85", close(xb, 50.85, 5e-9), xb)
check(T, "例1 s²=3.3539", close(s2, 3.3539), s2)
check(T, "例1 s=1.8314", close(sqrt(s2), 1.8314), sqrt(s2))
check(T, "例1 SE=0.5791", close(se, 0.5791), se)
check(T, "例1 t=1.468", close(tv, 1.468, 5e-4), tv)
r = stats.ttest_1samp(d, 50)
check(T, "例1 p=0.1762（ttest_1samp）", close(r.pvalue, 0.1762), r.pvalue)
check(T, "図 右すそ≈0.088", close(t.sf(tv, 9), 0.088, 5e-4), t.sf(tv, 9))
check(T, "t_0.025(9)=2.262", close(t.ppf(0.975, 9), 2.262, 5e-4), t.ppf(0.975, 9))
check(T, "t_0.025(15)=2.131", close(t.ppf(0.975, 15), 2.131, 5e-4), t.ppf(0.975, 15))
check(T, "t_0.025(ν)→1.96（ν=10^6）", close(t.ppf(0.975, 1e6), 1.96, 5e-3), t.ppf(0.975, 1e6))
ci = (xb - t.ppf(0.975, 9) * se, xb + t.ppf(0.975, 9) * se)
check(T, "例1 95%CI が 50 を含む（検定と一致）", ci[0] < 50 < ci[1], ci)
t2 = 2.3 / (4.0 / 4)
check(T, "例2 t=2.300", close(t2, 2.3, 1e-12), t2)
check(T, "例2 t_0.05(15)=1.753", close(t.ppf(0.95, 15), 1.753, 5e-4), t.ppf(0.95, 15))
check(T, "例2 p=P(T15≥2.3)=0.0181", close(t.sf(2.3, 15), 0.0181), t.sf(2.3, 15))
check(T, "例2 z の p=1-Φ(2.3)=0.0107", close(norm.sf(2.3), 0.0107), norm.sf(2.3))
# T ~ t(n-1) のシミュレーション
X = rng.normal(50, 2, (200000, 10))
Ts = (X.mean(1) - 50) / (X.std(1, ddof=1) / sqrt(10))
ks = stats.kstest(Ts, t(9).cdf).statistic
check(T, "シミュ H0 で T~t(9)（KS<0.005, 20万回）", ks < 0.005, ks)
# 非心度とz近似の過大評価
pw_t = nct.sf(t.ppf(0.95, 9), 9, sqrt(10) * 0.5)
pw_z = norm.sf(norm.ppf(0.95) - sqrt(10) * 0.5)
check(T, "z近似 1-Φ(t_α-δ) 系は非心t より過大（n=10,d=0.5 片側）", pw_z > pw_t, (pw_t, pw_z))

# ================= z-test-known-variance =================
T = "z-test-known-variance"
z1 = (502.1 - 500) / (4 / 4)
check(T, "例1 z=2.100", close(z1, 2.1, 1e-9), z1)
check(T, "例1 1-Φ(2.1)=0.01786", close(norm.sf(2.1), 0.01786, 5e-6), norm.sf(2.1))
check(T, "例1 両側p=0.0357", close(2 * norm.sf(2.1), 0.0357), 2 * norm.sf(2.1))
check(T, "図 右すそ 0.0179", close(norm.sf(2.1), 0.0179), norm.sf(2.1))
check(T, "z_0.005=2.576 > 2.1（1%で非有意）", close(norm.isf(0.005), 2.576, 5e-4) and z1 < norm.isf(0.005), norm.isf(0.005))
check(T, "z_0.025=1.960", close(norm.isf(0.025), 1.960, 5e-4), norm.isf(0.025))
check(T, "z_0.05=1.645", close(norm.isf(0.05), 1.645, 5e-4), norm.isf(0.05))
z2 = 3.5 / (10 / 5)
check(T, "例2 z=1.750", close(z2, 1.75, 1e-12), z2)
check(T, "例2 p=1-Φ(1.75)=0.0401", close(norm.sf(1.75), 0.0401), norm.sf(1.75))
check(T, "例2 両側p=0.0801", close(2 * norm.sf(1.75), 0.0801), 2 * norm.sf(1.75))
# H1 のもとで Z~N(δ,1)
Xa = rng.normal(502, 4, (200000, 16))
Za = (Xa.mean(1) - 500) / 1
check(T, "シミュ H1 で Z の平均=δ=2, SD=1（許容0.01）", abs(Za.mean() - 2) < 0.01 and abs(Za.std() - 1) < 0.01, (Za.mean(), Za.std()))
pw_sim = np.mean(np.abs(Za) >= 1.959964)
pw_th = norm.sf(1.959964 - 2) + norm.cdf(-1.959964 - 2)
check(T, "シミュ 両側検出力（μ=502,n=16）一致（許容0.004）", abs(pw_sim - pw_th) < 0.004, (pw_sim, pw_th))
# 必要標本サイズの式：n={(zα+zβ)σ/(μ1-μ0)}² で片側検出力が 1-β
zb = norm.isf(0.2)
nreq = ((norm.isf(0.05) + zb) * 4 / 2) ** 2
check(T, "必要n の式で片側検出力=0.80", close(norm.sf(norm.isf(0.05) - sqrt(nreq) * 2 / 4), 0.8, 1e-9), nreq)
# 片側 n を4倍で SE 半分
check(T, "n を4倍で SE が半分", close(4 / sqrt(64), 0.5 * 4 / sqrt(16), 1e-12), "")

# ================= two-sample-t-test =================
T = "two-sample-t-test"
sp2 = (7 * 3.2**2 + 9 * 2.9**2) / 16
sp_ = sqrt(sp2)
se2 = sp_ * sqrt(1 / 8 + 1 / 10)
t2s = 2.7 / se2
check(T, "例1 71.68+75.69", close(7 * 10.24, 71.68, 1e-9) and close(9 * 8.41, 75.69, 1e-9), "")
check(T, "例1 s_p²=9.2106", close(sp2, 9.2106), sp2)
check(T, "例1 s_p=3.0349", close(sp_, 3.0349), sp_)
check(T, "例1 √0.225 と SE=1.4396", close(1 / 8 + 1 / 10, 0.225, 1e-12) and close(se2, 1.4396), se2)
check(T, "例1 t=1.876", close(t2s, 1.876, 5e-4), t2s)
r = stats.ttest_ind_from_stats(24.5, 3.2, 8, 21.8, 2.9, 10, equal_var=True)
check(T, "例1 ttest_ind_from_stats と一致 p=0.0791", close(r.statistic, t2s, 1e-9) and close(r.pvalue, 0.0791), r.pvalue)
check(T, "t_0.025(16)=2.120", close(t.ppf(0.975, 16), 2.120, 5e-4), t.ppf(0.975, 16))
lo, hi = 2.7 - t.ppf(0.975, 16) * se2, 2.7 + t.ppf(0.975, 16) * se2
check(T, "例1 95%CI (-0.352, 5.752)", close(lo, -0.352, 5e-4) and close(hi, 5.752, 5e-4), (lo, hi))
lo2, hi2 = 2.7 - 2.120 * 1.4396, 2.7 + 2.120 * 1.4396
check(T, "例1 95%CI 表示値で計算しても (-0.352, 5.752)", close(lo2, -0.352, 5e-4) and close(hi2, 5.752, 5e-4), (lo2, hi2))
check(T, "例2 片側p=0.0395 / 図 0.040", close(t.sf(t2s, 16), 0.0395) and close(t.sf(t2s, 16), 0.040, 5e-4), t.sf(t2s, 16))
check(T, "図 両側 0.079", close(2 * t.sf(t2s, 16), 0.079, 5e-4), 2 * t.sf(t2s, 16))
check(T, "単純平均 9.325", close((10.24 + 8.41) / 2, 9.325, 1e-9), "")
Xs = rng.normal(0, 3, (200000, 8)); Ys = rng.normal(0, 3, (200000, 10))
spp = ((7 * Xs.var(1, ddof=1) + 9 * Ys.var(1, ddof=1)) / 16)
Tsim = (Xs.mean(1) - Ys.mean(1)) / np.sqrt(spp * 0.225)
ks = stats.kstest(Tsim, t(16).cdf).statistic
check(T, "シミュ H0 で T~t(16)（KS<0.005）", ks < 0.005, ks)
# m=n のとき s_p² は単純平均
check(T, "m=n なら s_p²=(s1²+s2²)/2（sympy）",
      sp.simplify((sp.Symbol('k') - 1) * (sp.Symbol('a') + sp.Symbol('b')) / (2 * sp.Symbol('k') - 2) - (sp.Symbol('a') + sp.Symbol('b')) / 2) == 0, "")

# ================= pooled-variance =================
T = "pooled-variance"
check(T, "例1 147.37/16=9.2106", close(147.37 / 16, 9.2106), 147.37 / 16)
check(T, "例1 s_p=3.0349", close(sqrt(147.37 / 16), 3.0349), sqrt(147.37 / 16))
v3 = (4 * 4.0 + 6 * 6.5 + 5 * 5.2) / 15
check(T, "例2 16+39+26=81, 81/15=5.400", 4 * 4.0 == 16 and 6 * 6.5 == 39 and close(5 * 5.2, 26, 1e-9) and close(v3, 5.4, 1e-9), v3)
check(T, "例2 単純平均 5.233", close((4.0 + 6.5 + 5.2) / 3, 5.233, 5e-4), (4.0 + 6.5 + 5.2) / 3)
# 一元配置の MSE と一致（任意データで）
grp = [rng.normal(0, 2, k) for k in (5, 7, 6)]
sse = sum(((g - g.mean())**2).sum() for g in grp)
spk = sum((len(g) - 1) * g.var(ddof=1) for g in grp) / 15
check(T, "例2 s_p² = 一元配置の残差平均平方 SSE/(N-k)", close(spk, sse / 15, 1e-10), spk)
check(T, "例3 2×81/7=23.14", close(2 * 81 / 7, 23.14, 5e-3), 2 * 81 / 7)
check(T, "例3 2×81/16=10.13", close(2 * 81 / 16, 10.13, 5.01e-3), 2 * 81 / 16)
check(T, "例3 半分以下", 2 * 81 / 16 < 0.5 * 2 * 81 / 7, "")
# 不偏性と分散、カイ二乗分布（m=8,n=10,σ²=9）
X1 = rng.normal(5, 3, (200000, 8)); X2 = rng.normal(-1, 3, (200000, 10))
Sp2 = (7 * X1.var(1, ddof=1) + 9 * X2.var(1, ddof=1)) / 16
check(T, "シミュ E[S_p²]=9（許容0.03, SE≈0.007）", abs(Sp2.mean() - 9) < 0.03, Sp2.mean())
check(T, "シミュ V[S_p²]=2σ⁴/16=10.125（許容0.15）", abs(Sp2.var() - 10.125) < 0.15, Sp2.var())
ks = stats.kstest(16 * Sp2 / 9, chi2(16).cdf).statistic
check(T, "シミュ 16S_p²/σ²~χ²(16)（KS<0.005）", ks < 0.005, ks)
check(T, "シミュ E[S_p]<σ=3", np.sqrt(Sp2).mean() < 3, np.sqrt(Sp2).mean())
# 最適重み：V[w S1² + (1-w) S2²] 最小 → w ∝ (m-1)
w, a, b = sp.symbols("w a b", positive=True)  # a=m-1, b=n-1
Vw = w**2 * 2 / a + (1 - w)**2 * 2 / b
wopt = sp.solve(sp.diff(Vw, w), w)[0]
check(T, "最適重み w*=(m-1)/(m+n-2)（sympy）", sp.simplify(wopt - a / (a + b)) == 0, wopt)
check(T, "最適時の分散 2σ⁴/(m+n-2)（sympy）", sp.simplify(Vw.subs(w, wopt) - 2 / (a + b)) == 0, "")
# 図：相対的なばらつき sd/mean = sqrt(2/ν) が自由度16 < 7
check(T, "図 χ²(16) の CV < χ²(7) の CV", sqrt(2 / 16) < sqrt(2 / 7), (sqrt(2 / 16), sqrt(2 / 7)))
# 全体をまとめた不偏分散は過大
Xa1 = rng.normal(0, 3, (100000, 8)); Xa2 = rng.normal(4, 3, (100000, 10))
allv = np.concatenate([Xa1, Xa2], 1).var(1, ddof=1).mean()
check(T, "シミュ 平均が違う2群をまとめた不偏分散は σ²=9 より過大", allv > 9.5, allv)

# ================= welch-test =================
T = "welch-test"
a1, a2 = 64 / 6, 9 / 15
sew = sqrt(a1 + a2)
tw = 8.0 / sew
nu = (a1 + a2)**2 / (a1**2 / 5 + a2**2 / 14)
check(T, "例1 s1²/m=10.667, s2²/n=0.600", close(a1, 10.667, 5e-4) and close(a2, 0.6, 1e-12), (a1, a2))
check(T, "例1 SE=3.3566", close(sew, 3.3566), sew)
check(T, "例1 t_W=2.383", close(tw, 2.383, 5e-4), tw)
check(T, "例1 11.267²=126.94", close((a1 + a2)**2, 126.94, 5e-3), (a1 + a2)**2)
check(T, "例1 10.667²/5=22.756, 0.6²/14=0.0257", close(a1**2 / 5, 22.756, 5e-4) and close(a2**2 / 14, 0.0257, 5e-5), (a1**2 / 5, a2**2 / 14))
check(T, "例1 ν≈5.57", close(nu, 5.57, 5e-3), nu)
r = stats.ttest_ind_from_stats(60, 8, 6, 52, 3, 15, equal_var=False)
check(T, "例1 ttest(equal_var=False) と一致", close(r.statistic, tw, 1e-9), r.statistic)
check(T, "例1 t_0.025(5.57)≈2.493", close(t.ppf(0.975, nu), 2.493, 5e-4), t.ppf(0.975, nu))
check(T, "例1 p=0.0577", close(r.pvalue, 0.0577), r.pvalue)
check(T, "図 右すそ≈0.029, 両側0.058", close(t.sf(tw, nu), 0.029, 5e-4) and close(2 * t.sf(tw, nu), 0.058, 5e-4), t.sf(tw, nu))
sp2w = (5 * 64 + 14 * 9) / 19
se_p = sqrt(sp2w * (1 / 6 + 1 / 15))
check(T, "例2 s_p²=23.474", close(sp2w, 23.474, 5e-4), sp2w)
check(T, "例2 SE=2.3403", close(se_p, 2.3403), se_p)
check(T, "例2 t=3.418", close(8 / se_p, 3.418, 5e-4), 8 / se_p)
check(T, "例2 p=0.0029", close(2 * t.sf(8 / se_p, 19), 0.0029), 2 * t.sf(8 / se_p, 19))
# シミュレーション 20万回（MC SE ≈ 0.0009 / 0.0005）
N = 200000
Xw = rng.normal(0, 8, (N, 6)); Yw = rng.normal(0, 3, (N, 15))
v1, v2 = Xw.var(1, ddof=1), Yw.var(1, ddof=1)
diff = Xw.mean(1) - Yw.mean(1)
spw = (5 * v1 + 14 * v2) / 19
tp = diff / np.sqrt(spw * (1 / 6 + 1 / 15))
rej_p = np.mean(np.abs(tp) >= t.ppf(0.975, 19))
q1, q2 = v1 / 6, v2 / 15
tww = diff / np.sqrt(q1 + q2)
nuw = (q1 + q2)**2 / (q1**2 / 5 + q2**2 / 14)
rej_w = np.mean(np.abs(tww) >= t.ppf(0.975, nuw))
check(T, "シミュ 等分散t の棄却率≈17.6%（本文 17.4→17.6 に修正。200万回で0.1755、許容0.003=MC SE 0.00085の約3.5倍）", abs(rej_p - 0.176) < 0.003, rej_p)
check(T, "シミュ ウェルチの棄却率≈5.3%（本文 5.2→5.3 に修正。200万回で0.0531、許容0.002=MC SE 0.0005の約4倍）", abs(rej_w - 0.053) < 0.002, rej_w)
# ν の式：モーメント法 ν=2E[V̂]²/V[V̂]
s1, s2s, mm, nn = sp.symbols("sigma1 sigma2 m n", positive=True)
EV = s1**2 / mm + s2s**2 / nn
VV = 2 * (s1**2 / mm)**2 / (mm - 1) + 2 * (s2s**2 / nn)**2 / (nn - 1)
# V[S_i²/m]=(1/m²)·2σ⁴/(m-1)
check(T, "V[V̂] の各項 = (1/m²)·2σ⁴/(m-1)（sympy）", sp.simplify(VV - (2 * s1**4 / (mm - 1) / mm**2 + 2 * s2s**4 / (nn - 1) / nn**2)) == 0, "")
nu_sym = sp.simplify(2 * EV**2 / VV)
nu_text = (s1**2 / mm + s2s**2 / nn)**2 / ((s1**2 / mm)**2 / (mm - 1) + (s2s**2 / nn)**2 / (nn - 1))
check(T, "2E²/V が本文の ν の式と一致（sympy）", sp.simplify(nu_sym - nu_text) == 0, "")
check(T, "χ²(ν)/ν の分散=2/ν", close(chi2.var(7.3) / 7.3**2, 2 / 7.3, 1e-12), "")
# モーメント法の近似がシミュの V̂ に合う
Vhat = q1 + q2
check(T, "シミュ E[V̂]=64/6+9/15（相対1%）", abs(Vhat.mean() / (64 / 6 + 0.6) - 1) < 0.01, Vhat.mean())
check(T, "シミュ V[V̂]=2(64/6)²/5+2(0.6)²/14（相対3%）", abs(Vhat.var() / (2 * (64 / 6)**2 / 5 + 2 * 0.36 / 14) - 1) < 0.03, Vhat.var())
# ν の範囲
mins = []
ok_rng = True
for _ in range(20000):
    m_, n_ = rng.integers(2, 40, 2)
    v_1, v_2 = rng.exponential(10, 2)
    A, B = v_1 / m_, v_2 / n_
    nv = (A + B)**2 / (A**2 / (m_ - 1) + B**2 / (n_ - 1))
    if not (min(m_ - 1, n_ - 1) - 1e-9 <= nv <= m_ + n_ - 2 + 1e-9):
        ok_rng = False
check(T, "min(m-1,n-1)≤ν≤m+n-2（ランダム2万組）", ok_rng, "")
# 上限は s1²/(m(m-1)) ... 等号条件：A/(m-1)=B/(n-1) で ν=m+n-2
m_, n_ = 6, 15
A = 5.0; B = A * 14 / 5
check(T, "上限 m+n-2 に達する例", close((A + B)**2 / (A**2 / 5 + B**2 / 14), 19, 1e-9), "")
# m=n なら T_W = 等分散の T
xm, ym = rng.normal(0, 1, 12), rng.normal(0, 3, 12)
tW = stats.ttest_ind(xm, ym, equal_var=False).statistic
tP = stats.ttest_ind(xm, ym, equal_var=True).statistic
check(T, "m=n で T_W と等分散 T が一致", close(tW, tP, 1e-10), (tW, tP))
check(T, "切り捨て自由度5 の棄却限界 > ν=5.57 の限界（保守的）", t.ppf(0.975, 5) > t.ppf(0.975, nu), (t.ppf(0.975, 5), t.ppf(0.975, nu)))

# ================= paired-t-test =================
T = "paired-t-test"
Xb = np.array([148, 152, 139, 160, 145, 155, 142, 150])
Yb = np.array([141, 149, 136, 151, 144, 147, 140, 143])
D = Xb - Yb
check(T, "例1 差の列 7,3,3,9,1,8,2,7", list(D) == [7, 3, 3, 9, 1, 8, 2, 7], list(D))
check(T, "例1 d̄=40/8=5", D.sum() == 40 and D.mean() == 5, D.mean())
check(T, "例1 s_D²=9.4286", close(D.var(ddof=1), 9.4286), D.var(ddof=1))
check(T, "例1 s_D=3.0706", close(D.std(ddof=1), 3.0706), D.std(ddof=1))
seD = D.std(ddof=1) / sqrt(8)
check(T, "例1 SE=1.0856", close(seD, 1.0856), seD)
r = stats.ttest_rel(Xb, Yb)
check(T, "例1 t=4.606", close(r.statistic, 4.606, 5e-4), r.statistic)
check(T, "t_0.025(7)=2.365", close(t.ppf(0.975, 7), 2.365, 5e-4), t.ppf(0.975, 7))
check(T, "例1 p=0.0025", close(r.pvalue, 0.0025), r.pvalue)
check(T, "図 右すそ≈0.0012", close(t.sf(r.statistic, 7), 0.0012), t.sf(r.statistic, 7))
r2 = stats.ttest_ind(Xb, Yb, equal_var=True)
check(T, "例2 独立とみなすと t=1.664", close(r2.statistic, 1.664, 5e-4), r2.statistic)
check(T, "例2 p=0.1184", close(r2.pvalue, 0.1184), r2.pvalue)
rho = np.corrcoef(Xb, Yb)[0, 1]
check(T, "例2 相関 0.917", close(rho, 0.917, 5e-4), rho)
se_ind = 5 / r2.statistic
check(T, "例2 SE が約2.8倍（本文を 2.7→2.8 に修正）", close(se_ind / seD, 2.8, 0.05), se_ind / seD)
check(T, "V[X-Y]=sX²+sY²-2sXY（標本版で恒等）", close(np.var(D, ddof=1), np.var(Xb, ddof=1) + np.var(Yb, ddof=1) - 2 * np.cov(Xb, Yb)[0, 1], 1e-9), "")
check(T, "対応ありは1標本t検定と同一", close(stats.ttest_1samp(D, 0).statistic, r.statistic, 1e-12), "")

# ================= power-of-t-test =================
T = "power-of-t-test"
dl = sqrt(20) * 0.5
c = t.ppf(0.975, 19)
up, low = nct.sf(c, 19, dl), nct.cdf(-c, 19, dl)
check(T, "例1 δ=2.236", close(dl, 2.236, 5e-4), dl)
check(T, "例1 t_0.025(19)=2.093", close(c, 2.093, 5e-4), c)
check(T, "例1 P(T'≥c)=0.5645, P(T'≤-c)≈0.0000", close(up, 0.5645) and low < 5e-5, (up, low))
check(T, "例1 検出力 0.5645", close(up + low, 0.5645), up + low)
zap = norm.sf(1.959964 - dl) + norm.cdf(-1.959964 - dl)
check(T, "例1 z近似 0.6088", close(zap, 0.6088), zap)
c1 = t.ppf(0.95, 19)
check(T, "例1 t_0.05(19)=1.729", close(c1, 1.729, 5e-4), c1)
check(T, "例1 片側検出力 0.6951", close(nct.sf(c1, 19, dl), 0.6951), nct.sf(c1, 19, dl))
napx = ((1.96 + 0.842) / 0.5)**2
check(T, "近似n=31.4", close(napx, 31.4, 0.05), napx)


def pw1(n, d=0.5):
    cc = t.ppf(0.975, n - 1)
    return nct.sf(cc, n - 1, sqrt(n) * d) + nct.cdf(-cc, n - 1, sqrt(n) * d)


check(T, "n=33 で 0.7954", close(pw1(33), 0.7954), pw1(33))
check(T, "n=34 で 0.8078（最小の n）", close(pw1(34), 0.8078) and pw1(33) < 0.8 <= pw1(34), pw1(34))


def pw2(n, d=0.8):
    df = 2 * n - 2
    cc = t.ppf(0.975, df)
    dd = d * sqrt(n / 2)
    return nct.sf(cc, df, dd) + nct.cdf(-cc, df, dd)


check(T, "例2 非心度 0.8√10=2.530", close(0.8 * sqrt(10), 2.530, 5e-4), 0.8 * sqrt(10))
check(T, "例2 t_0.025(38)=2.024", close(t.ppf(0.975, 38), 2.024, 5e-4), t.ppf(0.975, 38))
check(T, "例2 検出力 0.6934", close(pw2(20), 0.6934), pw2(20))
check(T, "例2 n=26 で 0.8075（最小の n）", close(pw2(26), 0.8075) and pw2(25) < 0.8 <= pw2(26), (pw2(25), pw2(26)))
check(T, "例2 近似 2×(2.802/0.8)²=24.5", close(2 * (2.802 / 0.8)**2, 24.5, 0.05), 2 * (2.802 / 0.8)**2)
# 一般の δ=(μ1-μ2)/(σ√(1/m+1/n)) で m=n なら d√(n/2)
check(T, "δ=d/√(2/n)=d√(n/2)", close(0.8 / sqrt(2 / 20), 0.8 * sqrt(10), 1e-12), "")
# 図データ
xs = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
y_t = [0.1405, 0.2932, 0.4379, 0.5645, 0.6697, 0.754, 0.8195, 0.8694, 0.9339, 0.9678]
y_z = [0.201, 0.3526, 0.4907, 0.6088, 0.7054, 0.7819, 0.8409, 0.8854, 0.9424, 0.9721]
ok, err = series_ok(xs, y_t, lambda n: round(pw1(n), 4), 1e-9)
check(T, "図 非心t の検出力10点（4桁丸め一致）", ok, err)
zf = lambda n: norm.sf(norm.isf(0.025) - sqrt(n) * 0.5) + norm.cdf(-norm.isf(0.025) - sqrt(n) * 0.5)
ok, err = series_ok(xs, y_z, lambda n: round(zf(n), 4), 1e-9)
check(T, "図 z近似10点（4桁丸め一致）", ok, err)
gaps = [b - a for a, b in zip(y_t, y_z)]
check(T, "図 z近似が常に上・小標本ほど差が大", all(g > 0 for g in gaps) and all(gaps[i] >= gaps[i + 1] for i in range(len(gaps) - 1)), gaps)
# シミュ：H1 で T ~ nct(19, 2.236)
Xp = rng.normal(0.5, 1, (200000, 20))
Tp = Xp.mean(1) / (Xp.std(1, ddof=1) / sqrt(20))
ks = stats.kstest(Tp, nct(19, dl).cdf).statistic
check(T, "シミュ H1 で T~非心t(19,2.236)（KS<0.005）", ks < 0.005, ks)
check(T, "シミュ 検出力≈0.5645（許容0.004）", abs(np.mean(np.abs(Tp) >= c) - 0.5645) < 0.004, np.mean(np.abs(Tp) >= c))
check(T, "非心t は非対称（歪度≠0）", abs(float(nct.stats(19, dl, moments='s'))) > 0.05, float(nct.stats(19, dl, moments='s')))

# ================= test-ci-relationship-mean-difference =================
T = "test-ci-relationship-mean-difference"
SE = 2.7 / t2s  # 1.4396 の丸め前
check(T, "例1 95%CI (-0.352,5.752)", close(2.7 - t.ppf(0.975, 16) * SE, -0.352, 5e-4) and close(2.7 + t.ppf(0.975, 16) * SE, 5.752, 5e-4), "")
check(T, "例1 p=0.0791", close(2 * t.sf(2.7 / SE, 16), 0.0791), 2 * t.sf(2.7 / SE, 16))
c90 = t.ppf(0.95, 16)
check(T, "t_0.05(16)=1.746", close(c90, 1.746, 5e-4), c90)
l90, h90 = 2.7 - c90 * SE, 2.7 + c90 * SE
check(T, "例1 90%CI (0.187, 5.213)", close(l90, 0.187, 5e-4) and close(h90, 5.213, 5e-4), (l90, h90))
check(T, "例1 Δ0=1: t=1.181", close(1.7 / SE, 1.181, 5e-4), 1.7 / SE)
check(T, "例1 Δ0=1: p=0.2549", close(2 * t.sf(1.7 / SE, 16), 0.2549), 2 * t.sf(1.7 / SE, 16))
wl, wh = 8 - t.ppf(0.975, nu) * sew, 8 + t.ppf(0.975, nu) * sew
check(T, "例2 ウェルチ95%CI (-0.369, 16.369)", close(wl, -0.369, 5e-4) and close(wh, 16.369, 5e-4), (wl, wh))
check(T, "例2 p=0.0577", close(2 * t.sf(8 / sew, nu), 0.0577), 2 * t.sf(8 / sew, nu))
xs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 2.7, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]
ys = [0.0049, 0.0101, 0.0205, 0.041, 0.0791, 0.146, 0.2549, 0.4168, 0.6334, 0.8912, 1, 0.8376, 0.5861, 0.3799, 0.2291, 0.1297, 0.0696, 0.0358, 0.0178, 0.0087, 0.0042]
pf = lambda D0: 2 * t.sf(abs(2.7 - D0) / SE, 16)
ok, err = series_ok(xs, ys, lambda D0: round(pf(D0), 4), 1e-9)
check(T, "図 p値関数21点（4桁丸め一致）", ok, err)
# p>0.05 の範囲 = 95%CI（約 −0.35〜5.75）
grid = np.linspace(-3, 9, 120001)
inside = grid[np.array([pf(g) for g in grid]) > 0.05]
check(T, "図 p>0.05 の範囲 ≈ (-0.35, 5.75)", close(inside.min(), -0.35, 5e-3) and close(inside.max(), 5.75, 5e-3), (inside.min(), inside.max()))
# 同値関係のランダム確認
okd = True
for _ in range(2000):
    D0 = rng.uniform(-3, 9)
    okd &= ((abs((2.7 - D0) / SE) < t.ppf(0.975, 16)) == (2.7 - t.ppf(0.975, 16) * SE < D0 < 2.7 + t.ppf(0.975, 16) * SE))
check(T, "|T(Δ0)|<t ⇔ Δ0∈CI（ランダム2000点）", okd, "")
# 片側：右片側検定 ⇔ 下側信頼限界>Δ0
okd = True
for _ in range(2000):
    D0 = rng.uniform(-3, 9)
    okd &= ((t.sf((2.7 - D0) / SE, 16) < 0.05) == (2.7 - t.ppf(0.95, 16) * SE > D0))
check(T, "右片側検定 ⇔ 下側信頼限界>Δ0（ランダム2000点）", okd, "")
# 個別区間が重なっても差が有意な例
m1, m2, s1e, s2e = 0.0, 3.0, 1.0, 1.0
check(T, "個別95%区間が重なるのに差は有意な例（SE=1,差3）", (m1 + 1.96 * s1e > m2 - 1.96 * s2e) and abs(m2 - m1) / sqrt(s1e**2 + s2e**2) > 1.96, "")

# ================= f-test-equal-variances =================
T = "f-test-equal-variances"
Fv = 12.5 / 5.0
Fu = F.isf(0.025, 10, 12)
Fl = 1 / F.isf(0.025, 12, 10)
check(T, "例1 F=2.500", Fv == 2.5, Fv)
check(T, "F_0.025(10,12)=3.374", close(Fu, 3.374, 5e-4), Fu)
check(T, "下側 1/F_0.025(12,10)=0.276", close(Fl, 0.276, 5e-4), Fl)
check(T, "F_{1-α/2}(10,12)=1/F_{α/2}(12,10)", close(F.ppf(0.025, 10, 12), Fl, 1e-10), F.ppf(0.025, 10, 12))
check(T, "例1 P(F≥2.5)=0.0677 / 図 0.068", close(F.sf(2.5, 10, 12), 0.0677) and close(F.sf(2.5, 10, 12), 0.068, 5e-4), F.sf(2.5, 10, 12))
check(T, "例1 両側p=0.1354 / 図 0.135", close(2 * F.sf(2.5, 10, 12), 0.1354) and close(2 * F.sf(2.5, 10, 12), 0.135, 5e-4), 2 * F.sf(2.5, 10, 12))
cl, ch = 2.5 / Fu, 2.5 / Fl
check(T, "例1 分散比CI [0.741, 9.052]", close(cl, 0.741, 5e-4) and close(ch, 9.052, 5e-4), (cl, ch))
pwF = F.sf(Fu / 3, 10, 12) + F.cdf(Fl / 3, 10, 12)
check(T, "例1 真の比3で検出力≈0.418", close(pwF, 0.418, 5e-4), pwF)
X1 = rng.normal(0, sqrt(3), (200000, 11)); X2 = rng.normal(0, 1, (200000, 13))
Fs = X1.var(1, ddof=1) / X2.var(1, ddof=1)
pws = np.mean((Fs >= Fu) | (Fs <= Fl))
check(T, "シミュ 比3の検出力≈0.418（許容0.004）", abs(pws - 0.418) < 0.004, pws)
ks = stats.kstest(Fs / 3, F(10, 12).cdf).statistic
check(T, "シミュ F/r ~ F(10,12)（KS<0.005）", ks < 0.005, ks)
check(T, "例2 F_0.05(10,12)=2.753", close(F.isf(0.05, 10, 12), 2.753, 5e-4), F.isf(0.05, 10, 12))
check(T, "例2 片側 p=0.0677", close(F.sf(2.5, 10, 12), 0.0677), "")

# ================= chi-square-test-variance =================
T = "chi-square-test-variance"
cs = 14 * 0.27**2 / 0.04
check(T, "例1 14×0.0729/0.04=25.515", close(cs, 25.515, 5e-4) and close(0.27**2, 0.0729, 1e-12), cs)
check(T, "χ²_0.05(14)=23.685 / 図 23.68", close(chi2.isf(0.05, 14), 23.685, 5e-4), chi2.isf(0.05, 14))
check(T, "例1 p=0.0298 / 図 0.030", close(chi2.sf(cs, 14), 0.0298) and close(chi2.sf(25.52, 14), 0.030, 5e-4), chi2.sf(cs, 14))
check(T, "例2 χ²_0.975(14)=5.629", close(chi2.ppf(0.025, 14), 5.629, 5e-4), chi2.ppf(0.025, 14))
check(T, "例2 χ²_0.025(14)=26.119", close(chi2.isf(0.025, 14), 26.119, 5e-4), chi2.isf(0.025, 14))
p2 = 2 * min(chi2.sf(cs, 14), chi2.cdf(cs, 14))
check(T, "例2 両側p=0.0596", close(p2, 0.0596), p2)
thr = 0.04 / 0.09 * chi2.isf(0.05, 14)
check(T, "検出力 閾値 10.53", close(thr, 10.53, 5e-3), thr)
check(T, "検出力 ≈0.72", close(chi2.sf(thr, 14), 0.72, 5e-3), chi2.sf(thr, 14))
Xc = rng.normal(0, 0.30, (200000, 15))
st = 14 * Xc.var(1, ddof=1) / 0.04
pwc = np.mean(st >= chi2.isf(0.05, 14))
check(T, "シミュ σ=0.30 の検出力≈0.72（許容0.005）", abs(pwc - chi2.sf(thr, 14)) < 0.005, pwc)
Xc0 = rng.normal(3, 0.20, (200000, 15))
st0 = 14 * Xc0.var(1, ddof=1) / 0.04
ks = stats.kstest(st0, chi2(14).cdf).statistic
check(T, "シミュ H0 で χ²~χ²(14)（KS<0.005）", ks < 0.005, ks)
st_mu = ((Xc0 - 3)**2).sum(1) / 0.04
ks = stats.kstest(st_mu, chi2(15).cdf).statistic
check(T, "シミュ μ既知なら χ²(n)（KS<0.005）", ks < 0.005, ks)
check(T, "E[χ²]=n-1（シミュ, 許容0.05）", abs(st0.mean() - 14) < 0.05, st0.mean())
# 非正規（t5, 尖度大）では大標本でも水準がずれる
Xh = rng.standard_t(5, (50000, 200))
sth = 199 * Xh.var(1, ddof=1) / (5 / 3)
lev = np.mean(sth >= chi2.isf(0.05, 199))
check(T, "シミュ t(5)母集団・n=200 で水準が0.05から大きくずれる", lev > 0.10, lev)

# ================= fisher-z-transformation =================
T = "fisher-z-transformation"
zr = atanh(0.45)
check(T, "例1 z=0.4847 と ½log(1.45/0.55)", close(zr, 0.4847) and close(0.5 * log(1.45 / 0.55), zr, 1e-12), zr)
check(T, "例1 1/√17=0.2425", close(1 / sqrt(17), 0.2425), 1 / sqrt(17))
zl, zh = zr - 1.959964 / sqrt(17), zr + 1.959964 / sqrt(17)
check(T, "例1 z区間 (0.0093, 0.9601)", close(zl, 0.0093) and close(zh, 0.9601), (zl, zh))
check(T, "例1 ρ区間 (0.009, 0.744)", close(tanh(zl), 0.009, 5e-4) and close(tanh(zh), 0.744, 5e-4), (tanh(zl), tanh(zh)))
check(T, "例1 非対称（上側が狭い）", tanh(zh) - 0.45 < 0.45 - tanh(zl), (tanh(zh) - 0.45, 0.45 - tanh(zl)))
check(T, "tanh u=(e^{2u}-1)/(e^{2u}+1)", close(tanh(0.7), (np.exp(1.4) - 1) / (np.exp(1.4) + 1), 1e-14), "")
check(T, "例1 atanh0.2=0.2027", close(atanh(0.2), 0.2027), atanh(0.2))
Zf = (zr - atanh(0.2)) * sqrt(17)
check(T, "例1 Z=1.163", close(Zf, 1.163, 5e-4), Zf)
check(T, "例1 p=0.2450", close(2 * norm.sf(Zf), 0.2450), 2 * norm.sf(Zf))
check(T, "例1 表示値 0.2820/0.2425 でも 1.163", close((0.4847 - 0.2027) / 0.2425, 1.163, 5e-4), "")
za, zb2 = atanh(0.62), atanh(0.35)
den = sqrt(1 / 47 + 1 / 57)
Z2 = (za - zb2) / den
check(T, "例2 atanh0.62=0.7250, atanh0.35=0.3654", close(za, 0.7250) and close(zb2, 0.3654), (za, zb2))
check(T, "例2 差0.3596, 分母0.1970", close(za - zb2, 0.3596) and close(den, 0.1970), (za - zb2, den))
check(T, "例2 Z=1.825", close(Z2, 1.825, 5e-4), Z2)
check(T, "例2 p=0.0680", close(2 * norm.sf(Z2), 0.0680), 2 * norm.sf(Z2))
# デルタ法（sympy）
rr, rh, nn_ = sp.symbols("r rho n", positive=True)
g = sp.atanh(rr)
gp = sp.diff(g, rr)
check(T, "g'(r)=1/(1-r²)（sympy）", sp.simplify(gp - 1 / (1 - rr**2)) == 0, gp)
check(T, "g'(ρ)²(1-ρ²)²/n=1/n（sympy）", sp.simplify(gp.subs(rr, rh)**2 * (1 - rh**2)**2 / nn_ - 1 / nn_) == 0, "")
check(T, "atanh=½log((1+r)/(1-r))（-0.99〜0.99 の格子で数値一致）", all(close(atanh(u), 0.5 * log((1 + u) / (1 - u)), 1e-12) for u in np.linspace(-0.99, 0.99, 199)), "")
# 図：n=20, 4万回の SD（MC: SD の標準誤差 ≈ SD/√(2·40000)≈0.0009, 許容0.005）
xs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y_r = [0.2291, 0.2267, 0.2221, 0.2115, 0.1971, 0.1776, 0.1549, 0.1255, 0.0918, 0.049]
y_z = [0.242, 0.2414, 0.2429, 0.2424, 0.2421, 0.2413, 0.242, 0.2397, 0.2416, 0.2387]
err_r, err_z = [], []
rs_dict = {}
for rho_, a_, b_ in zip(xs, y_r, y_z):
    cov = [[1, rho_], [rho_, 1]]
    L = np.linalg.cholesky(cov)
    E = rng.standard_normal((40000, 20, 2)) @ L.T
    xc = E[:, :, 0] - E[:, :, 0].mean(1, keepdims=True)
    yc = E[:, :, 1] - E[:, :, 1].mean(1, keepdims=True)
    rs = (xc * yc).sum(1) / np.sqrt((xc**2).sum(1) * (yc**2).sum(1))
    rs_dict[rho_] = rs
    err_r.append(abs(rs.std() - a_) / max(a_, 0.05))
    err_z.append(abs(np.arctanh(rs).std() - b_))
check(T, "図 r の SD 10点（相対許容3%）", max(err_r) < 0.03, max(err_r))
check(T, "図 z の SD 10点（許容0.005）", max(err_z) < 0.005, max(err_z))
check(T, "図 理論値 1/√17=0.2425", close(1 / sqrt(17), 0.2425), "")
check(T, "z の平均 ≈ atanh ρ（ρ=0.5, 許容0.02）", abs(np.arctanh(rs_dict[0.5]).mean() - atanh(0.5)) < 0.02, np.arctanh(rs_dict[0.5]).mean())
# V[r]≈(1-ρ²)²/n（n=200 で確認）
cov = [[1, 0.5], [0.5, 1]]
E = rng.standard_normal((20000, 200, 2)) @ np.linalg.cholesky(cov).T
xc = E[:, :, 0] - E[:, :, 0].mean(1, keepdims=True); yc = E[:, :, 1] - E[:, :, 1].mean(1, keepdims=True)
rb = (xc * yc).sum(1) / np.sqrt((xc**2).sum(1) * (yc**2).sum(1))
check(T, "V[r]≈(1-ρ²)²/n（n=200, 相対5%）", abs(rb.var() / ((1 - 0.25)**2 / 200) - 1) < 0.05, rb.var())
check(T, "ρ=0.8 で r の分布は左に裾（歪度<0）", stats.skew(rs_dict[0.8]) < 0, stats.skew(rs_dict[0.8]))
# 信頼区間の被覆率（n=20, ρ=0.5, 4万回, 許容0.006）
rs5 = rs_dict[0.5]
cov_ok = np.mean((np.tanh(np.arctanh(rs5) - 1.959964 / sqrt(17)) < 0.5) & (np.tanh(np.arctanh(rs5) + 1.959964 / sqrt(17)) > 0.5))
check(T, "シミュ ρ の95%CI 被覆率≈0.95（許容0.006）", abs(cov_ok - 0.95) < 0.006, cov_ok)

# ================= test-correlation-coefficient =================
T = "test-correlation-coefficient"
tc = 0.45 * sqrt(18) / sqrt(1 - 0.45**2)
check(T, "例1 √18=4.2426, √(1-0.45²)=0.8930", close(sqrt(18), 4.2426) and close(sqrt(1 - 0.2025), 0.8930), "")
check(T, "例1 t=2.138", close(tc, 2.138, 5e-4), tc)
check(T, "t_0.025(18)=2.101", close(t.ppf(0.975, 18), 2.101, 5e-4), t.ppf(0.975, 18))
check(T, "例1 p=0.0465", close(2 * t.sf(tc, 18), 0.0465), 2 * t.sf(tc, 18))
check(T, "図 右すそ≈0.023, 両側0.046", close(t.sf(tc, 18), 0.023, 5e-4) and close(2 * t.sf(tc, 18), 0.046, 5e-4), t.sf(tc, 18))


def rc(n):
    q = t.ppf(0.975, n - 2)
    return q / sqrt(n - 2 + q**2)


check(T, "r_c(20)=0.444", close(rc(20), 0.444, 5e-4), rc(20))
check(T, "r_c(10)=0.632", close(rc(10), 0.632, 5e-4), rc(10))
check(T, "r_c(30)=0.361", close(rc(30), 0.361, 5e-4), rc(30))
check(T, "r_c(100)=0.197", close(rc(100), 0.197, 5e-4), rc(100))
check(T, "例2 n=10 で r=0.45 非有意, n=100 で r=0.2 有意", 0.45 < rc(10) and 0.2 > rc(100), "")
# r_c の式：|T|≥q ⇔ |r|≥r_c
okr = all(((rv * sqrt(18) / sqrt(1 - rv**2)) >= t.ppf(0.975, 18)) == (rv >= rc(20)) for rv in np.linspace(0, 0.99, 991))
check(T, "|T|≥t ⇔ |r|≥r_c（格子991点）", okr, "")
# 回帰の傾きの t と一致
xr = rng.normal(0, 1, 20); yr = 0.4 * xr + rng.normal(0, 1, 20)
lr = stats.linregress(xr, yr)
tslope = lr.slope / lr.stderr
rv = lr.rvalue
check(T, "傾きの t 統計量 = r√(n-2)/√(1-r²)", close(tslope, rv * sqrt(18) / sqrt(1 - rv**2), 1e-10), tslope)
# H0 で T~t(n-2)
E = rng.standard_normal((100000, 20, 2))
xc = E[:, :, 0] - E[:, :, 0].mean(1, keepdims=True); yc = E[:, :, 1] - E[:, :, 1].mean(1, keepdims=True)
r0 = (xc * yc).sum(1) / np.sqrt((xc**2).sum(1) * (yc**2).sum(1))
ks = stats.kstest(r0 * sqrt(18) / np.sqrt(1 - r0**2), t(18).cdf).statistic
check(T, "シミュ ρ=0 で T~t(18)（KS<0.006, 10万回）", ks < 0.006, ks)
Tr3 = rs_dict[0.3] * sqrt(18) / np.sqrt(1 - rs_dict[0.3]**2)
check(T, "ρ≠0 では T は t(18) に従わない（ρ=0.3, KS>0.05）", stats.kstest(Tr3, t(18).cdf).statistic > 0.05, "")

# ================= hotelling-t2 =================
T = "hotelling-t2"
dv = np.array([1.2, -0.8])
S = np.array([[4.0, 1.5], [1.5, 2.25]])
check(T, "例1 det S=6.75", close(np.linalg.det(S), 6.75, 1e-12), np.linalg.det(S))
check(T, "例1 3.24+2.88+2.56", close(2.25 * 1.44, 3.24, 1e-12) and close(-2 * 1.5 * 1.2 * -0.8, 2.88, 1e-12) and close(4 * 0.64, 2.56, 1e-12), "")
qf = dv @ np.linalg.solve(S, dv)
check(T, "例1 二次形式 1.2859", close(qf, 1.2859), qf)
T2 = 15 * qf
check(T, "例1 T²=19.289", close(T2, 19.289, 5e-4), T2)
Fh = 13 / 28 * T2
check(T, "例1 F=8.956", close(Fh, 8.956, 5e-4), Fh)
check(T, "F_0.05(2,13)=3.806", close(F.isf(0.05, 2, 13), 3.806, 5e-4), F.isf(0.05, 2, 13))
check(T, "例1 p=0.0036", close(F.sf(Fh, 2, 13), 0.0036), F.sf(Fh, 2, 13))
t_a = 1.2 / sqrt(4 / 15); t_b = -0.8 / sqrt(2.25 / 15)
check(T, "寸法 t=2.324, p=0.0357", close(t_a, 2.324, 5e-4) and close(2 * t.sf(t_a, 14), 0.0357), (t_a, 2 * t.sf(t_a, 14)))
check(T, "重量 t=-2.066, p=0.0579", close(t_b, -2.066, 5e-4) and close(2 * t.sf(-t_b, 14), 0.0579), (t_b, 2 * t.sf(-t_b, 14)))
check(T, "相関係数 0.5", close(1.5 / sqrt(4 * 2.25), 0.5, 1e-12), "")
check(T, "逆行列の公式", np.allclose(np.linalg.inv(S), np.array([[2.25, -1.5], [-1.5, 4.0]]) / 6.75), "")
# 同じ向きのずれなら T² は小さくなる（相関の効果）
qf_same = np.array([1.2, 0.8]) @ np.linalg.solve(S, np.array([1.2, 0.8]))
check(T, "正相関で同向きのずれは小さく評価（逆向き 1.2859 > 同向き）", qf_same < qf, qf_same)
# p=1 で t²
xx = rng.normal(0.3, 1, 12)
T2_1 = 12 * (xx.mean())**2 / xx.var(ddof=1)
check(T, "p=1 で T²=t²、係数=1", close(T2_1, stats.ttest_1samp(xx, 0).statistic**2, 1e-10) and (12 - 1) / (1 * 11) == 1, "")
check(T, "t(ν)² ~ F(1,ν)：分位点", close(t.ppf(0.975, 11)**2, F.isf(0.05, 1, 11), 1e-8), "")
# 1標本 T² の帰無分布（シミュ, n=15, p=2）
L = np.linalg.cholesky(S)
E = rng.standard_normal((100000, 15, 2)) @ L.T
mb = E.mean(1)
C = E - mb[:, None, :]
Sm = np.einsum('kij,kil->kjl', C, C) / 14
T2s = 15 * np.einsum('ki,ki->k', mb, np.linalg.solve(Sm, mb[:, :, None])[:, :, 0])
ks = stats.kstest(13 / 28 * T2s, F(2, 13).cdf).statistic
check(T, "シミュ (n-p)/{p(n-1)}T²~F(2,13)（KS<0.006, 10万回）", ks < 0.006, ks)
# Σ既知なら χ²(p)
Sig_inv = np.linalg.inv(S)
q_known = 15 * np.einsum('ki,ij,kj->k', mb, Sig_inv, mb)
ks = stats.kstest(q_known, chi2(2).cdf).statistic
check(T, "シミュ Σ既知なら n·d'Σ⁻¹d~χ²(2)（KS<0.006）", ks < 0.006, ks)
# 2標本 T²（n1=10, n2=12, p=3）
p3 = 3; n1, n2 = 10, 12
A1 = rng.standard_normal((60000, n1, p3)); A2 = rng.standard_normal((60000, n2, p3))
m1v, m2v = A1.mean(1), A2.mean(1)
C1 = A1 - m1v[:, None, :]; C2 = A2 - m2v[:, None, :]
Spm = (np.einsum('kij,kil->kjl', C1, C1) + np.einsum('kij,kil->kjl', C2, C2)) / (n1 + n2 - 2)
dd = m1v - m2v
T2b = n1 * n2 / (n1 + n2) * np.einsum('ki,ki->k', dd, np.linalg.solve(Spm, dd[:, :, None])[:, :, 0])
Fb = (n1 + n2 - p3 - 1) / (p3 * (n1 + n2 - 2)) * T2b
ks = stats.kstest(Fb, F(p3, n1 + n2 - p3 - 1).cdf).statistic
check(T, "シミュ 2標本 T² の変換 ~F(3,18)（KS<0.007, 6万回）", ks < 0.007, ks)

print()
print(f"合計: OK {sum(results)} / NG {len(results) - sum(results)}")
