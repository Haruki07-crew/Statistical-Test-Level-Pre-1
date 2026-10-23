"""第20章 検算スクリプト（分散分析と実験計画法）。
各分散分析表は numpy で平方和を定義式から直接計算し（修正項 CT 方式とも照合）、F・p は scipy.stats。
テューキー法の q は scipy.stats.studentized_range、期待平均平方は sympy（一元配置の E[S_A]）と乱数シミュレーション。
主張値との比較は表示桁の丸め幅（小数第3位なら 5e-4 など）。charts の params は front matter を yaml で読んで照合。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（標準誤差の4倍以上）。
"""
from math import sqrt, comb
from itertools import combinations
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-4):
    return abs(float(x) - claim) <= tol


def allclose(xs, claims, tol=5e-4):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol))


def r(v, k=3):
    return np.round(np.asarray(v, float), k).tolist()


def charts(term):
    txt = (ROOT / "content" / "ch20" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1]).get("charts", [])


def fp(F, d1, d2):
    return stats.f.sf(F, d1, d2)


def fcrit(a, d1, d2):
    return stats.f.isf(a, d1, d2)


# ---------------- 汎用の分散分析 ----------------
def oneway(groups):
    allv = np.concatenate([np.asarray(g, float) for g in groups])
    gm = allv.mean()
    ST = ((allv - gm) ** 2).sum()
    SA = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    SE = sum(((np.asarray(g, float) - np.mean(g)) ** 2).sum() for g in groups)
    k, N = len(groups), len(allv)
    VA, VE = SA / (k - 1), SE / (N - k)
    F = VA / VE
    return dict(ST=ST, SA=SA, SE=SE, VA=VA, VE=VE, F=F, p=fp(F, k - 1, N - k), dfA=k - 1, dfE=N - k)


def twoway_norep(Y):
    Y = np.asarray(Y, float)
    a, b = Y.shape
    gm = Y.mean()
    SA = b * ((Y.mean(1) - gm) ** 2).sum()
    SB = a * ((Y.mean(0) - gm) ** 2).sum()
    res = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm
    SE = (res ** 2).sum()
    ST = ((Y - gm) ** 2).sum()
    dfE = (a - 1) * (b - 1)
    VA, VB, VE = SA / (a - 1), SB / (b - 1), SE / dfE
    return dict(ST=ST, SA=SA, SB=SB, SE=SE, VA=VA, VB=VB, VE=VE, FA=VA / VE, FB=VB / VE,
                pA=fp(VA / VE, a - 1, dfE), pB=fp(VB / VE, b - 1, dfE), dfE=dfE)


def twoway_rep(Y):
    """Y[i, j, k]"""
    Y = np.asarray(Y, float)
    a, b, rr = Y.shape
    gm = Y.mean()
    cell = Y.mean(2)
    Ai, Bj = cell.mean(1), cell.mean(0)
    SA = b * rr * ((Ai - gm) ** 2).sum()
    SB = a * rr * ((Bj - gm) ** 2).sum()
    SAB = rr * ((cell - Ai[:, None] - Bj[None, :] + gm) ** 2).sum()
    SE = ((Y - cell[:, :, None]) ** 2).sum()
    ST = ((Y - gm) ** 2).sum()
    dA, dB, dAB, dE = a - 1, b - 1, (a - 1) * (b - 1), a * b * (rr - 1)
    V = dict(A=SA / dA, B=SB / dB, AB=SAB / dAB, E=SE / dE)
    return dict(ST=ST, SA=SA, SB=SB, SAB=SAB, SE=SE, V=V, df=(dA, dB, dAB, dE),
                F=dict(A=V["A"] / V["E"], B=V["B"] / V["E"], AB=V["AB"] / V["E"]))


# =====================================================================
print("=== one-way-anova ===")
T = "one-way-anova"
g = [[18, 21, 20, 17], [22, 24, 23, 25], [19, 20, 22, 19]]
o = oneway(g)
allv = np.concatenate(g)
check(T, "群合計 76,94,80・平均 19,23.5,20・総合計250", [sum(x) for x in g] == [76, 94, 80] and allv.sum() == 250
      and np.allclose([np.mean(x) for x in g], [19, 23.5, 20]), [sum(x) for x in g])
CT = 250 ** 2 / 12
check(T, "CT=5208.333・Σy²=5274", close(CT, 5208.333) and (allv ** 2).sum() == 5274, (round(CT, 3), int((allv ** 2).sum())))
check(T, "CT 方式 S_A: 5253−CT", (76 ** 2 + 94 ** 2 + 80 ** 2) / 4 == 5253, 5253)
check(T, "S_T=65.667, S_A=44.667, S_E=21.000（定義式）", allclose([o["ST"], o["SA"], o["SE"]], [65.667, 44.667, 21.000]),
      r([o["ST"], o["SA"], o["SE"]]))
check(T, "V_A=22.333, V_E=2.333, F=9.571", allclose([o["VA"], o["VE"], o["F"]], [22.333, 2.333, 9.571]), r([o["VA"], o["VE"], o["F"]]))
check(T, "p=0.0059", close(o["p"], 0.0059, 5e-5), round(o["p"], 5))
check(T, "F_0.05(2,9)=4.256, F_0.01(2,9)=8.022", close(fcrit(.05, 2, 9), 4.256) and close(fcrit(.01, 2, 9), 8.022),
      (round(fcrit(.05, 2, 9), 4), round(fcrit(.01, 2, 9), 4)))
tk = stats.tukey_hsd(*[np.array(x, float) for x in g])
pv = [tk.pvalue[0, 1], tk.pvalue[1, 2], tk.pvalue[0, 2]]
check(T, "例2 テューキー調整済み p 0.006/0.025/0.639", allclose(pv, [0.006, 0.025, 0.639]), r(pv, 4))
check(T, "例3 η²=0.680", close(o["SA"] / o["ST"], 0.680), round(o["SA"] / o["ST"], 4))
# k=2 で F=t²
g2 = [g[0], g[1]]
t2 = stats.ttest_ind(g[0], g[1]).statistic ** 2
check(T, "k=2 で F=t²（肥料1と2）", close(oneway(g2)["F"], t2, 1e-9), (oneway(g2)["F"], t2))
c = charts(T)
check(T, "chart F(2,9) と caption 9.571・0.0059", c[0]["params"] == {"df1": 2, "df2": 9}, c[0]["params"])
check(T, "chart bar-data 群平均・総平均 20.833", c[1]["params"]["values"] == [19, 23.5, 20] and close(allv.mean(), 20.833),
      c[1]["params"]["values"])
# H0 下の F 分布（シミュレーション）
sims = rng.normal(0, 1, (200000, 3, 4))
gmean = sims.mean(2)
SAs = 4 * ((gmean - gmean.mean(1, keepdims=True)) ** 2).sum(1)
SEs = ((sims - gmean[:, :, None]) ** 2).sum((1, 2))
Fs = (SAs / 2) / (SEs / 9)
rej = (Fs > fcrit(.05, 2, 9)).mean()
check(T, "H0 下の棄却率 0.05（20万回、±4SE=0.002）", abs(rej - 0.05) < 0.002, round(rej, 4))
check(T, "H0 下の F が F(2,9)（KS 検定 p>0.001）", stats.kstest(Fs[:20000], "f", args=(2, 9)).pvalue > 0.001,
      round(stats.kstest(Fs[:20000], "f", args=(2, 9)).pvalue, 4))
check(T, "S_A と S_E の無相関（|r|<0.01）", abs(np.corrcoef(SAs, SEs)[0, 1]) < 0.01, round(np.corrcoef(SAs, SEs)[0, 1], 4))

# =====================================================================
print("=== sum-of-squares-decomposition ===")
T = "sum-of-squares-decomposition"
g = [[3, 5, 7], [8, 10], [6, 7, 8, 9]]
o = oneway(g)
allv = np.concatenate(g).astype(float)
check(T, "総平均 63/9=7", allv.sum() == 63 and allv.mean() == 7, allv.mean())
check(T, "偏差2乗 16,4,0,1,9,1,0,1,4", ((allv - 7) ** 2).tolist() == [16, 4, 0, 1, 9, 1, 0, 1, 4], ((allv - 7) ** 2).tolist())
check(T, "S_T=36, S_A=12+8+1=21, S_E=8+2+5=15", allclose([o["ST"], o["SA"], o["SE"]], [36, 21, 15], 1e-9)
      and [3 * 4, 2 * 4, 4 * 0.25] == [12, 8, 1], r([o["ST"], o["SA"], o["SE"]]))
check(T, "群3 の群内偏差² 2.25,0.25,0.25,2.25", ((np.array(g[2]) - 7.5) ** 2).tolist() == [2.25, 0.25, 0.25, 2.25], "ok")
check(T, "F=4.200, p=0.0723", close(o["F"], 4.2) and close(o["p"], 0.0723, 5e-5), (round(o["F"], 4), round(o["p"], 5)))
check(T, "寄与率 21/36=0.583", close(21 / 36, 0.583), round(21 / 36, 4))
cross = sum((np.mean(x) - 7) * (np.array(x) - np.mean(x)).sum() for x in g)
check(T, "交差項=0（例2、群1 の偏差 −2,0,2）", cross == 0 and (np.array(g[0]) - 5).tolist() == [-2, 0, 2], cross)
# 記号的な交差項
ys = sp.symbols("y1:6")
grp = [ys[:2], ys[2:]]
gmS = sum(ys) / 5
crossS = sum((sum(G) / len(G) - gmS) * sum(y - sum(G) / len(G) for y in G) for G in grp)
check(T, "交差項の恒等的消滅（sympy, n=(2,3)）", sp.simplify(crossS) == 0, sp.simplify(crossS))
# ランダムデータで分解の恒等式
okk = True
for _ in range(200):
    k = rng.integers(2, 6)
    gg = [rng.normal(rng.normal(), 1, rng.integers(2, 7)) for _ in range(k)]
    oo = oneway(gg)
    okk &= abs(oo["ST"] - oo["SA"] - oo["SE"]) < 1e-9
check(T, "S_T=S_A+S_E（ランダム200例）", okk, okk)
x = rng.normal(size=20)
y = 1 + 2 * x + rng.normal(size=20)
bfit = np.polyfit(x, y, 1)
yh = np.polyval(bfit, x)
d1 = ((y - y.mean()) ** 2).sum() - ((yh - y.mean()) ** 2).sum() - ((y - yh) ** 2).sum()
b0 = (x @ y) / (x @ x)
yh0 = b0 * x
d0 = ((y - y.mean()) ** 2).sum() - ((yh0 - y.mean()) ** 2).sum() - ((y - yh0) ** 2).sum()
check(T, "回帰の分解（切片あり成立・切片なしで不成立）", abs(d1) < 1e-9 and abs(d0) > 1e-3, (d1, round(d0, 4)))
a_, b_, r_ = sp.symbols("a b r", positive=True)
check(T, "二元配置の自由度 abr−1 の分解（sympy）",
      sp.expand((a_ - 1) + (b_ - 1) + (a_ - 1) * (b_ - 1) + a_ * b_ * (r_ - 1) - (a_ * b_ * r_ - 1)) == 0, "0")
c = charts(T)
check(T, "chart1 bar [36,21,15]", c[0]["params"]["values"] == [36, 21, 15], c[0]["params"]["values"])
check(T, "chart2 偏差2乗の並び", c[1]["params"]["values"] == ((allv - 7) ** 2).tolist(), c[1]["params"]["values"])

# =====================================================================
print("=== expected-mean-square ===")
T = "expected-mean-square"
# sympy: 一元配置 E[S_A] = Σ n_i α_i² + (k−1)σ²（k=3, n=(2,3,4) の具体形で、誤差の2次モーメントから）
n_list = [2, 3, 4]
N = sum(n_list)
s2 = sp.symbols("sigma2", positive=True)
al = sp.symbols("alpha1:4")
# ε の共分散 σ² I。S_A = (μ+α+ε) の二次形式 y'Py, E = μ'Pμ + σ² tr P
idx = np.repeat(np.arange(3), n_list)
Pm = sp.zeros(N, N)
for i in range(N):
    for j in range(N):
        Pm[i, j] = (sp.Rational(1, n_list[idx[i]]) if idx[i] == idx[j] else 0) - sp.Rational(1, N)
mu_vec = sp.Matrix([al[idx[i]] for i in range(N)])
constr = {al[2]: -(2 * al[0] + 3 * al[1]) / 4}
ESA = (mu_vec.T * Pm * mu_vec)[0] + s2 * Pm.trace()
target = sum(n_list[i] * al[i] ** 2 for i in range(3)) + 2 * s2
check(T, "E[S_A]=Σn_iα_i²+(k−1)σ²（sympy, n=(2,3,4), Σn_iα_i=0）",
      sp.simplify((ESA - target).subs(constr)) == 0, sp.simplify(ESA.subs(constr) - target.subs(constr)))
I_P = sp.eye(N) - sp.Matrix(N, N, lambda i, j: sp.Rational(1, n_list[idx[i]]) if idx[i] == idx[j] else 0)
check(T, "E[S_E]=(N−k)σ²（tr=6, 平均の寄与0）", I_P.trace() == N - 3 and (I_P * mu_vec) == sp.zeros(N, 1), I_P.trace())
# 例1
alpha = np.array([-2, 1, 1.])
EVA = 4 + 4 * (alpha ** 2).sum() / 2
check(T, "例1 E[V_A]=16, λ=24/4=6", EVA == 16 and 4 * (alpha ** 2).sum() / 4 == 6, EVA)
M = 200000
sims = alpha[None, :, None] + rng.normal(0, 2, (M, 3, 4))
gmean = sims.mean(2)
VA = 4 * ((gmean - gmean.mean(1, keepdims=True)) ** 2).sum(1) / 2
VE = ((sims - gmean[:, :, None]) ** 2).sum((1, 2)) / 9
seA, seE = VA.std() / sqrt(M), VE.std() / sqrt(M)
check(T, f"シミュ 20万回 V_A 平均≈16（本文15.98、±4SE={4 * seA:.3f}）",
      abs(VA.mean() - 16) < 4 * seA and abs(15.98 - 16) < 4 * seA, round(VA.mean(), 3))
check(T, f"シミュ V_E 平均≈4（本文3.99、±4SE={4 * seE:.4f}）",
      abs(VE.mean() - 4) < 4 * seE and abs(3.99 - 4) <= 0.005 + 4 * seE, round(VE.mean(), 4))
pw = stats.ncf.sf(fcrit(.05, 2, 9), 2, 9, 6)
check(T, "検出力 0.441（非心F(2,9;6)）", close(pw, 0.441), round(pw, 4))
Fs = VA / VE
pws = (Fs > fcrit(.05, 2, 9)).mean()
check(T, "検出力のシミュレーション（±0.005）", abs(pws - pw) < 0.005, round(pws, 4))
check(T, "非心 F の分布（KS p>0.001）", stats.kstest(Fs[:20000], "ncf", args=(2, 9, 6)).pvalue > 0.001,
      round(stats.kstest(Fs[:20000], "ncf", args=(2, 9, 6)).pvalue, 4))
# 例2 分散成分
VA_, VB_, VAB_, VE_ = 196 / 3, 169 / 3, 52 / 3, 2.0  # 繰り返しありの例1の厳密値（65.333 等の丸め前）
vc = [(VAB_ - VE_) / 2, (VA_ - VAB_) / 4, (VB_ - VAB_) / 6]
check(T, "例2 σ̂_AB²=7.667, σ̂_A²=12.000, σ̂_B²=6.500（厳密値で）", allclose(vc, [7.667, 12.0, 6.5]), r(vc))


# 二元配置 EMS 表をシミュレーションで確認（a=3,b=2,r=2）
def sim_two(a, b, rr, model, M=100000, s2e=1.0, sA=1.0, sB=1.0, sAB=1.0):
    al_ = np.array([-1.0, 0.0, 1.0])[:a]
    be_ = np.array([-1.0, 1.0])[:b]
    ab_ = np.array([[0.5, -0.5], [-1.0, 1.0], [0.5, -0.5]])[:a, :b]
    eps = rng.normal(0, sqrt(s2e), (M, a, b, rr))
    if model == "fixed":
        mu = al_[:, None] + be_[None, :] + ab_
        Y = mu[None, :, :, None] + eps
    elif model == "random":
        A = rng.normal(0, sqrt(sA), (M, a, 1, 1))
        B = rng.normal(0, sqrt(sB), (M, 1, b, 1))
        AB = rng.normal(0, sqrt(sAB), (M, a, b, 1))
        Y = A + B + AB + eps
    elif model in ("mixed_r", "mixed_u"):
        B = rng.normal(0, sqrt(sB), (M, 1, b, 1))
        G = rng.normal(0, sqrt(sAB), (M, a, b, 1))
        AB = G - G.mean(1, keepdims=True) if model == "mixed_r" else G
        Y = al_[None, :, None, None] + B + AB + eps
    gm = Y.mean((1, 2, 3))
    cell = Y.mean(3)
    Ai, Bj = cell.mean(2), cell.mean(1)
    SA = b * rr * ((Ai - gm[:, None]) ** 2).sum(1)
    SB = a * rr * ((Bj - gm[:, None]) ** 2).sum(1)
    SAB = rr * ((cell - Ai[:, :, None] - Bj[:, None, :] + gm[:, None, None]) ** 2).sum((1, 2))
    SE = ((Y - cell[..., None]) ** 2).sum((1, 2, 3))
    V = dict(A=SA / (a - 1), B=SB / (b - 1), AB=SAB / ((a - 1) * (b - 1)), E=SE / (a * b * (rr - 1)))
    return {k: (v.mean(), v.std() / sqrt(M)) for k, v in V.items()}


a, b, rr = 3, 2, 2
thA, thB, thAB = 2 / 2, 2 / 1, (0.25 * 2 + 1 * 2 + 0.25 * 2) / 2
th = dict(
    fixed=dict(A=1 + b * rr * thA, B=1 + a * rr * thB, AB=1 + rr * thAB, E=1),
    random=dict(A=1 + rr + b * rr, B=1 + rr + a * rr, AB=1 + rr, E=1),
    mixed_r=dict(A=1 + rr + b * rr * thA, B=1 + a * rr, AB=1 + rr, E=1),
    mixed_u=dict(A=1 + rr + b * rr * thA, B=1 + rr + a * rr, AB=1 + rr, E=1),
)
for model, lab in [("fixed", "母数模型"), ("random", "変量模型"), ("mixed_r", "混合模型（制約付き）"),
                   ("mixed_u", "混合模型（制約なし、B に rσ_AB² が加わる）")]:
    sm = sim_two(a, b, rr, model)
    ok = all(abs(sm[k][0] - th[model][k]) < 4 * sm[k][1] for k in th[model])
    check(T, f"二元配置 EMS 表 {lab}（10万回、±4SE）", ok,
          {k: (round(sm[k][0], 3), th[model][k]) for k in th[model]})
c = charts(T)
check(T, "chart ncf(2,9,6) と比較 F(2,9)", c[0]["params"] == {"df1": 2, "df2": 9, "nc": 6}
      and c[0]["compare"][0]["params"] == {"df1": 2, "df2": 9}, c[0]["params"])

# =====================================================================
print("=== fixed-and-random-effects ===")
T = "fixed-and-random-effects"
lots = [[50.2, 50.6, 50.4], [51.0, 51.4, 51.5], [49.8, 50.1, 49.7], [50.6, 50.9, 50.5]]
o = oneway(lots)
check(T, "ロット平均 50.400, 51.300, 49.867, 50.667", allclose([np.mean(x) for x in lots], [50.4, 51.3, 49.867, 50.667]),
      r([np.mean(x) for x in lots]))
check(T, "S_A=3.196, S_E=0.393, V_A=1.065, V_E=0.0492", allclose([o["SA"], o["SE"], o["VA"]], [3.196, 0.393, 1.065])
      and close(o["VE"], 0.0492, 5e-5), r([o["SA"], o["SE"], o["VA"], o["VE"]], 4))
check(T, "F=21.667, p=0.0003", close(o["F"], 21.667) and close(o["p"], 0.0003, 5e-5), (round(o["F"], 4), round(o["p"], 5)))
sA = (o["VA"] - o["VE"]) / 3
rho = sA / (sA + o["VE"])
check(T, "V_A=1.0653, σ̂_A²=0.3387, ρ̂=0.873", close(o["VA"], 1.0653, 5e-5) and close(sA, 0.3387, 5e-5) and close(rho, 0.873),
      (round(o["VA"], 5), round(sA, 5), round(rho, 4)))
check(T, "丸め値での ρ̂=0.3387/(0.3387+0.0492)=0.873", close(0.3387 / (0.3387 + 0.0492), 0.873), round(0.3387 / 0.3879, 4))
check(T, "全分散 0.388・約87%", close(sA + o["VE"], 0.388) and round(rho * 100) == 87, round(sA + o["VE"], 4))
c = charts(T)
check(T, "chart bar [0.3387, 0.0492]", allclose(c[0]["params"]["values"], [sA, o["VE"]], 5e-5), c[0]["params"]["values"])
# 変量模型のシミュレーション E[V_A]=σ²+nσ_A²、級内相関
M = 200000
aa = rng.normal(0, sqrt(0.5), (M, 4, 1))
Y = 10 + aa + rng.normal(0, 1, (M, 4, 3))
gmn = Y.mean(2)
VAs = 3 * ((gmn - gmn.mean(1, keepdims=True)) ** 2).sum(1) / 3
VEs = ((Y - gmn[..., None]) ** 2).sum((1, 2)) / 8
check(T, "変量模型 E[V_A]=σ²+nσ_A²=2.5（20万回、±4SE）", abs(VAs.mean() - 2.5) < 4 * VAs.std() / sqrt(M), round(VAs.mean(), 4))
sAhat = (VAs - VEs) / 3
check(T, "σ̂_A² の不偏性（=0.5, ±4SE）", abs(sAhat.mean() - 0.5) < 4 * sAhat.std() / sqrt(M), round(sAhat.mean(), 4))
icc = np.corrcoef(Y[:, 0, 0], Y[:, 0, 1])[0, 1]
check(T, "級内相関 ρ=σ_A²/(σ_A²+σ²)=1/3（±0.01）", abs(icc - 1 / 3) < 0.01, round(icc, 4))
check(T, "群平均の分散 σ_A²+σ²/n（=0.8333, ±0.01）", abs(gmn[:, 0].var() - (0.5 + 1 / 3)) < 0.01, round(gmn[:, 0].var(), 4))
# 例2
FA1, FA2 = (196 / 3) / 2, (196 / 3) / (52 / 3)  # 丸め前の V_A=196/3, V_A×B=52/3
check(T, "例2 F_A=32.667（p=0.0006）, 3.769（p=0.210）",
      close(FA1, 32.667) and close(fp(FA1, 2, 6), 0.0006, 5e-5) and close(FA2, 3.769) and close(fp(FA2, 2, 2), 0.210),
      (round(FA1, 3), round(fp(FA1, 2, 6), 5), round(FA2, 4), round(fp(FA2, 2, 2), 4)))

# =====================================================================
print("=== tukey-multiple-comparison ===")
T = "tukey-multiple-comparison"
VE, n, nu = 7 / 3, 4, 9
se = sqrt(VE / n)
q05 = stats.studentized_range.isf(0.05, 3, 9)
HSD = q05 * se
check(T, "√(V_E/n)=0.7638, q_0.05(3,9)=3.948, HSD=3.016", close(se, 0.7638, 5e-5) and close(q05, 3.948) and close(HSD, 3.016),
      (round(se, 5), round(q05, 4), round(HSD, 4)))
diffs = [4.5, 3.5, 1.0]
qs = [d / se for d in diffs]
check(T, "q 値 5.892, 4.583, 1.309", allclose(qs, [5.892, 4.583, 1.309]), r(qs))
pq = [stats.studentized_range.sf(q, 3, 9) for q in qs]
check(T, "調整済み p 0.006, 0.025, 0.639", allclose(pq, [0.006, 0.025, 0.639]), r(pq, 4))
tkp = stats.tukey_hsd(*[np.array(x, float) for x in [[18, 21, 20, 17], [22, 24, 23, 25], [19, 20, 22, 19]]])
check(T, "scipy.stats.tukey_hsd の p と一致", allclose([tkp.pvalue[0, 1], tkp.pvalue[1, 2], tkp.pvalue[0, 2]], pq, 1e-6),
      r([tkp.pvalue[0, 1], tkp.pvalue[1, 2], tkp.pvalue[0, 2]], 5))
cis = [[d - HSD, d + HSD] for d in diffs]
check(T, "同時信頼区間 [1.484,7.516],[0.484,6.516],[−2.016,4.016]",
      allclose(cis, [[1.484, 7.516], [0.484, 6.516], [-2.016, 4.016]]), r(cis))
ci_sp = tkp.confidence_interval(0.95)
check(T, "scipy の CI（肥料2−肥料1）と一致", close(-ci_sp.high[0, 1], 1.484) and close(-ci_sp.low[0, 1], 7.516),
      (round(-ci_sp.high[0, 1], 3), round(-ci_sp.low[0, 1], 3)))
tb = stats.t.isf(0.05 / 6, 9)
sed = sqrt(2 * VE / n)
check(T, "ボンフェローニ t=2.933, √(2V_E/n)=1.080, 境目 3.168", close(tb, 2.933) and close(sed, 1.080) and close(tb * sed, 3.168),
      (round(tb, 4), round(sed, 4), round(tb * sed, 4)))
check(T, "補正なし t 境目 2.443 ＞ テューキー 3.016 の順", close(stats.t.isf(0.025, 9) * sed, 2.443) and tb * sed > HSD,
      round(stats.t.isf(0.025, 9) * sed, 4))
for nu_ in [5, 9, 30]:
    check(T, f"q_α(2,ν)=√2 t_α/2(ν)（ν={nu_}）",
          close(stats.studentized_range.isf(.05, 2, nu_), sqrt(2) * stats.t.isf(.025, nu_), 1e-4),
          (round(stats.studentized_range.isf(.05, 2, nu_), 5), round(sqrt(2) * stats.t.isf(.025, nu_), 5)))
qk = [stats.studentized_range.isf(.05, k, 9) for k in range(2, 11)]
check(T, "k が増えると q_α(k,ν) が増加", all(np.diff(qk) > 0), r(qk))
# Tukey-Kramer
q36 = stats.studentized_range.isf(0.05, 3, 6)
se12 = sqrt(2.5 / 2 * (1 / 3 + 1 / 2))
q12 = 4.0 / se12
check(T, "TK: q_0.05(3,6)=4.339, 分母 1.021, q=3.919, 境目 4.429",
      close(q36, 4.339) and close(se12, 1.021) and close(q12, 3.919) and close(q36 * se12, 4.429),
      (round(q36, 4), round(se12, 4), round(q12, 4), round(q36 * se12, 4)))
check(T, "TK: p=0.072", close(stats.studentized_range.sf(q12, 3, 6), 0.072), round(stats.studentized_range.sf(q12, 3, 6), 4))
tk2 = stats.tukey_hsd(np.array([3, 5, 7.]), np.array([8, 10.]), np.array([6, 7, 8, 9.]))
check(T, "TK の例は sum-of-squares の例1（V_E=2.5）と一致・scipy p", close(tk2.pvalue[0, 1], 0.072), round(tk2.pvalue[0, 1], 4))
# FWER シミュレーション
M = 200000
Z = rng.normal(0, 1, (M, 3, 4))
mz = Z.mean(2)
vz = ((Z - mz[..., None]) ** 2).sum((1, 2)) / 9
Q = (mz.max(1) - mz.min(1)) / np.sqrt(vz / 4)
fw = (Q > q05).mean()
check(T, "全帰無仮説下の FWER=0.05（20万回、±0.002）", abs(fw - 0.05) < 0.002, round(fw, 4))
tstat = np.abs(mz[:, :, None] - mz[:, None, :]) / np.sqrt(2 * vz / 4)[:, None, None]
fw_b = (tstat.max((1, 2)) > tb).mean()
check(T, "ボンフェローニの FWER は α 以下（保守的）", fw_b < 0.05 and fw_b < fw, round(fw_b, 4))
c = charts(T)
check(T, "chart bar [4.5,3.5,1.0,3.016]", allclose(c[0]["params"]["values"], diffs + [HSD]), c[0]["params"]["values"])
check(T, "chart studentized-range k=3, df=9, α=0.05", c[1]["params"] == {"k": 3, "df": 9, "alpha": 0.05}, c[1]["params"])

# =====================================================================
print("=== two-way-anova-with-replication ===")
T = "two-way-anova-with-replication"
Y = np.array([[[10, 12], [13, 15]], [[14, 16], [15, 17]], [[15, 17], [24, 26]]], float)
tw = twoway_rep(Y)
check(T, "A合計 50,62,82・B合計 84,110・総 194", Y.sum((1, 2)).tolist() == [50, 62, 82] and Y.sum((0, 2)).tolist() == [84, 110]
      and Y.sum() == 194, Y.sum((1, 2)).tolist())
CT = 194 ** 2 / 12
check(T, "CT=3136.333, Σy²=3370", close(CT, 3136.333) and (Y ** 2).sum() == 3370, round(CT, 3))
check(T, "中間値 3267, 3192.667, 3358, セル合計 22,28,30,32,32,50",
      (50 ** 2 + 62 ** 2 + 82 ** 2) / 4 == 3267 and close((84 ** 2 + 110 ** 2) / 6, 3192.667)
      and (Y.sum(2) ** 2).sum() / 2 == 3358 and Y.sum(2).ravel().tolist() == [22, 28, 30, 32, 32, 50], "ok")
check(T, "S_T,S_A,S_B,S_AB(セル),S_A×B,S_E", allclose([tw["ST"], tw["SA"], tw["SB"], 3358 - CT, tw["SAB"], tw["SE"]],
                                                      [233.667, 130.667, 56.333, 221.667, 34.667, 12.000]),
      r([tw["ST"], tw["SA"], tw["SB"], 3358 - CT, tw["SAB"], tw["SE"]]))
check(T, "平均平方 65.333, 56.333, 17.333, 2.000", allclose(list(tw["V"].values()), [65.333, 56.333, 17.333, 2.0]),
      r(list(tw["V"].values())))
Fv = [tw["F"]["A"], tw["F"]["B"], tw["F"]["AB"]]
check(T, "F 32.667, 28.167, 8.667", allclose(Fv, [32.667, 28.167, 8.667]), r(Fv))
pvs = [fp(Fv[0], 2, 6), fp(Fv[1], 1, 6), fp(Fv[2], 2, 6)]
check(T, "p 0.0006, 0.0018, 0.0170", allclose(pvs, [0.0006, 0.0018, 0.0170], 5e-5), r(pvs, 5))
check(T, "5%点 5.143(2,6), 5.987(1,6)", close(fcrit(.05, 2, 6), 5.143) and close(fcrit(.05, 1, 6), 5.987),
      (round(fcrit(.05, 2, 6), 4), round(fcrit(.05, 1, 6), 4)))
cm = Y.mean(2)
check(T, "温度ごとの B2−B1 差 3, 1, 9", (cm[:, 1] - cm[:, 0]).tolist() == [3, 1, 9], (cm[:, 1] - cm[:, 0]).tolist())
check(T, "プーリング 46.667, df 8, V_E=5.833", close(tw["SE"] + tw["SAB"], 46.667) and close((tw["SE"] + tw["SAB"]) / 8, 5.833),
      round((tw["SE"] + tw["SAB"]) / 8, 4))
check(T, "S_T=S_A+S_B+S_A×B+S_E", close(tw["SA"] + tw["SB"] + tw["SAB"] + tw["SE"], tw["ST"], 1e-9), "ok")
c = charts(T)
check(T, "chart line-data セル平均 [11,15,16],[14,16,25]",
      c[0]["params"]["series"][0]["y"] == cm[:, 0].tolist() and c[0]["params"]["series"][1]["y"] == cm[:, 1].tolist(),
      cm.T.tolist())
check(T, "chart F(2,6)", c[1]["params"] == {"df1": 2, "df2": 6}, c[1]["params"])

# =====================================================================
print("=== two-way-anova-without-replication ===")
T = "two-way-anova-without-replication"
Y2 = np.array([[52, 56, 49, 57], [55, 57, 53, 59], [57, 61, 54, 64]], float)
tn = twoway_norep(Y2)
check(T, "行合計 214,224,236・列合計 164,174,156,180・総 674・平均 53.5,56,59",
      Y2.sum(1).tolist() == [214, 224, 236] and Y2.sum(0).tolist() == [164, 174, 156, 180] and Y2.sum() == 674
      and Y2.mean(1).tolist() == [53.5, 56, 59], "ok")
CT = 674 ** 2 / 12
check(T, "CT=37856.333, Σy²=38036, 37917, 37969.333", close(CT, 37856.333) and (Y2 ** 2).sum() == 38036
      and (214 ** 2 + 224 ** 2 + 236 ** 2) / 4 == 37917 and close((164 ** 2 + 174 ** 2 + 156 ** 2 + 180 ** 2) / 3, 37969.333), "ok")
check(T, "S_T=179.667, S_A=60.667, S_B=113.000, S_E=6.000", allclose([tn["ST"], tn["SA"], tn["SB"], tn["SE"]],
                                                                      [179.667, 60.667, 113, 6]),
      r([tn["ST"], tn["SA"], tn["SB"], tn["SE"]]))
check(T, "V 30.333, 37.667, 1.000, F 30.333, 37.667", allclose([tn["VA"], tn["VB"], tn["VE"], tn["FA"], tn["FB"]],
                                                              [30.333, 37.667, 1, 30.333, 37.667]),
      r([tn["VA"], tn["VB"], tn["VE"], tn["FA"], tn["FB"]]))
check(T, "p 0.0007, 0.0003", allclose([tn["pA"], tn["pB"]], [0.0007, 0.0003], 5e-5), r([tn["pA"], tn["pB"]], 5))
check(T, "5%点 5.143, 4.757", close(fcrit(.05, 2, 6), 5.143) and close(fcrit(.05, 3, 6), 4.757), round(fcrit(.05, 3, 6), 4))
check(T, "1%水準でも有意（p<0.01）", tn["pA"] < 0.01, round(tn["pA"], 5))
o = oneway(list(Y2))
check(T, "例2 一元配置 S_E=119, V_E=13.222, F=2.294, p=0.157",
      allclose([o["SE"], o["VE"], o["F"], o["p"]], [119, 13.222, 2.294, 0.157]), r([o["SE"], o["VE"], o["F"], o["p"]], 4))
# a=2 で F=t²（対応のある t）
tpair = stats.ttest_rel(Y2[0], Y2[1]).statistic
check(T, "a=2 で F=t²（対応のある t、A1 と A2）", close(twoway_norep(Y2[:2])["FA"], tpair ** 2, 1e-9),
      (twoway_norep(Y2[:2])["FA"], tpair ** 2))
# 交互作用があるとき E[V_E]=σ²+ΣΣ(αβ)²/((a−1)(b−1))
inter = np.array([[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 0, 0.]])
inter = inter - inter.mean(0) - inter.mean(1, keepdims=True) + inter.mean()
M = 200000
Ys = inter[None] + rng.normal(0, 1, (M, 3, 4))
res = Ys - Ys.mean(2, keepdims=True) - Ys.mean(1, keepdims=True) + Ys.mean((1, 2), keepdims=True)
VEs = (res ** 2).sum((1, 2)) / 6
thv = 1 + (inter ** 2).sum() / 6
check(T, "交互作用ありで E[V_E]=σ²+ΣΣ(αβ)²/6（20万回、±4SE）", abs(VEs.mean() - thv) < 4 * VEs.std() / sqrt(M),
      (round(VEs.mean(), 4), round(thv, 4)))
c = charts(T)
check(T, "chart line-data の3系列 = データ", [s["y"] for s in c[0]["params"]["series"]] == Y2.tolist(), "ok")
check(T, "chart F(2,6)", c[1]["params"] == {"df1": 2, "df2": 6}, c[1]["params"])

# =====================================================================
print("=== main-effect-and-interaction ===")
T = "main-effect-and-interaction"
mu = np.array([[100, 110], [104, 130.]])  # 行: 陳列(通常,特設), 列: 価格(通常,値引き)
gm = mu.mean()
check(T, "総平均 111", gm == 111, gm)
check(T, "値引きの主効果 9（差18=(10+26)/2）", mu[:, 1].mean() - gm == 9 and (10 + 26) / 2 == 18, mu[:, 1].mean() - gm)
check(T, "特設の主効果 6（差12）", mu[1].mean() - gm == 6, mu[1].mean() - gm)
inter = mu - mu.mean(1, keepdims=True) - mu.mean(0, keepdims=True) + gm
check(T, "交互作用 (値引き,特設)=4、他 −4,−4,+4", inter.tolist() == [[4, -4], [-4, 4]] and 130 - 120 - 117 + 111 == 4, inter.tolist())
check(T, "差の差 26−10=16=4×4", (mu[1, 1] - mu[1, 0]) - (mu[0, 1] - mu[0, 0]) == 16 == 4 * inter[1, 1], 16)
cm = np.array([[11, 14], [15, 16], [16, 25.]])
gm2 = cm.mean()
Aeff, Beff = cm.mean(1) - gm2, cm.mean(0) - gm2
check(T, "例2 総平均 16.167, A 主効果 −3.667,−0.667,4.333, B 主効果 −2.167,2.167",
      close(gm2, 16.167) and allclose(Aeff, [-3.667, -0.667, 4.333]) and allclose(Beff, [-2.167, 2.167]),
      (round(gm2, 3), r(Aeff), r(Beff)))
it = cm - cm.mean(1, keepdims=True) - cm.mean(0, keepdims=True) + gm2
check(T, "例2 交互作用表", allclose(it, [[0.667, -0.667], [1.667, -1.667], [-2.333, 2.333]]), r(it))
check(T, "行和・列和=0", np.allclose(it.sum(0), 0) and np.allclose(it.sum(1), 0), "ok")
check(T, "S_A×B=2×Σ=34.667（繰り返しありの例と一致）", close(2 * (it ** 2).sum(), 34.667) and close(2 * (it ** 2).sum(), tw["SAB"], 1e-9),
      round(2 * (it ** 2).sum(), 4))
c = charts(T)
check(T, "chart1 line-data [100,110],[104,130]", [s["y"] for s in c[0]["params"]["series"]] == mu.tolist(), "ok")
p2 = np.array([s["y"] for s in c[1]["params"]["series"]], float)
it2 = p2 - p2.mean(1, keepdims=True) - p2.mean(0, keepdims=True) + p2.mean()
check(T, "chart2 平行（交互作用0・効果 +10）", np.allclose(it2, 0) and (p2[:, 1] - p2[:, 0]).tolist() == [10, 10], it2.tolist())
check(T, "対比 (+1,−1,−1,+1) で 2×2 交互作用", (mu[0, 0] - mu[0, 1] - mu[1, 0] + mu[1, 1]) == 16, 16)

# =====================================================================
print("=== randomized-block-design ===")
T = "randomized-block-design"
layout = [["V1", "V3", "V2", "V4"], ["V4", "V2", "V3", "V1"], ["V1", "V4", "V2", "V3"]]
check(T, "配置: 各ブロックに4品種が1回ずつ", all(sorted(L) == ["V1", "V2", "V3", "V4"] for L in layout), "ok")
Yb = np.array([[30, 36, 40], [34, 36, 44], [29, 34, 37], [35, 41, 45]], float)
rb = twoway_norep(Yb)
check(T, "品種合計 106,114,100,121・ブロック合計 128,147,166・総 441",
      Yb.sum(1).tolist() == [106, 114, 100, 121] and Yb.sum(0).tolist() == [128, 147, 166] and Yb.sum() == 441, "ok")
CT = 441 ** 2 / 12
check(T, "CT=16206.75, Σy²=16481, 16291, 16387.25", CT == 16206.75 and (Yb ** 2).sum() == 16481
      and (106 ** 2 + 114 ** 2 + 100 ** 2 + 121 ** 2) / 3 == 16291 and (128 ** 2 + 147 ** 2 + 166 ** 2) / 4 == 16387.25, "ok")
check(T, "S_T=274.25, S_τ=84.25, S_β=180.50, S_E=9.50", allclose([rb["ST"], rb["SA"], rb["SB"], rb["SE"]],
                                                                 [274.25, 84.25, 180.5, 9.5], 1e-9),
      r([rb["ST"], rb["SA"], rb["SB"], rb["SE"]]))
check(T, "V 28.083, 90.250, 1.583, F 17.737, 57.000", allclose([rb["VA"], rb["VB"], rb["VE"], rb["FA"], rb["FB"]],
                                                              [28.083, 90.25, 1.583, 17.737, 57.0]),
      r([rb["VA"], rb["VB"], rb["VE"], rb["FA"], rb["FB"]]))
check(T, "p 0.0022, 0.0001", allclose([rb["pA"], rb["pB"]], [0.0022, 0.0001], 5e-5), r([rb["pA"], rb["pB"]], 5))
check(T, "F_0.05(3,6)=4.757, 1%でも有意", close(fcrit(.05, 3, 6), 4.757) and rb["pA"] < 0.01, round(fcrit(.01, 3, 6), 3))
o = oneway(list(Yb))
check(T, "例2 S_E=190, V_E=23.75, F=1.182, p=0.376", allclose([o["SE"], o["VE"], o["F"], o["p"]], [190, 23.75, 1.182, 0.376]),
      r([o["SE"], o["VE"], o["F"], o["p"]], 4))
t_, b_ = 4, 3
RE = ((b_ - 1) * rb["VB"] + b_ * (t_ - 1) * rb["VE"]) / ((t_ * b_ - 1) * rb["VE"])
check(T, "RE: 分子 194.75, 分母 17.417, RE=11.18",
      close(2 * 90.25 + 9 * rb["VE"], 194.75, 5e-3) and close(11 * rb["VE"], 17.417) and close(RE, 11.18, 5e-3),
      (round(2 * 90.25 + 9 * rb["VE"], 3), round(11 * rb["VE"], 4), round(RE, 4)))
c = charts(T)
check(T, "chart line-data = データ", [s["y"] for s in c[0]["params"]["series"]] == Yb.tolist(), "ok")
check(T, "chart bar [23.75, 1.583]", allclose(c[1]["params"]["values"], [o["VE"], rb["VE"]]), c[1]["params"]["values"])
# 自由度の減少 t(b−1) → (t−1)(b−1)
check(T, "誤差自由度 t(b−1)=8 → (t−1)(b−1)=6（b−1=2 減）", o["dfE"] == 8 and rb["dfE"] == 6, (o["dfE"], rb["dfE"]))

# =====================================================================
print("=== latin-square-design ===")
T = "latin-square-design"
LS = [list("ABCD"), list("BCDA"), list("CDAB"), list("DABC")]
isLatin = all(sorted(row) == list("ABCD") for row in LS) and all(sorted(col) == list("ABCD") for col in zip(*LS))
check(T, "4×4 方格がラテン方格", isLatin, "ok")
# 直交性: 行・列・処理のダミー（中心化）が互いに直交
cells = [(i, j, "ABCD".index(LS[i][j])) for i in range(4) for j in range(4)]
D = lambda f: np.array([[1.0 if f(cl) == k else 0.0 for k in range(4)] for cl in cells]) - 0.25
Dr, Dc, Dt = D(lambda cl: cl[0]), D(lambda cl: cl[1]), D(lambda cl: cl[2])
check(T, "行・列・処理が互いに直交（中心化ダミーの内積=0）",
      np.allclose(Dr.T @ Dc, 0) and np.allclose(Dr.T @ Dt, 0) and np.allclose(Dc.T @ Dt, 0), "ok")
Yl = np.array([[20, 24, 24, 23], [24, 30, 24, 22], [25, 25, 18, 21], [28, 23, 23, 30]], float)
lab_data = [["A", "B", "C", "D"], ["B", "C", "D", "A"], ["C", "D", "A", "B"], ["D", "A", "B", "C"]]
check(T, "データの括弧内の方法 = 上の方格", lab_data == LS, "ok")
trt = np.array([[ "ABCD".index(LS[i][j]) for j in range(4)] for i in range(4)])
Tk = [Yl[trt == k].sum() for k in range(4)]
check(T, "行合計 91,100,89,104・列合計 97,102,89,96・総 384・方法合計 83,92,109,100",
      Yl.sum(1).tolist() == [91, 100, 89, 104] and Yl.sum(0).tolist() == [97, 102, 89, 96] and Yl.sum() == 384
      and Tk == [83, 92, 109, 100], (Yl.sum(1).tolist(), Yl.sum(0).tolist(), Tk))
gm = Yl.mean()
ST = ((Yl - gm) ** 2).sum()
SR = 4 * ((Yl.mean(1) - gm) ** 2).sum()
SC = 4 * ((Yl.mean(0) - gm) ** 2).sum()
Stau = 4 * ((np.array(Tk) / 4 - gm) ** 2).sum()
SE = ST - SR - SC - Stau
check(T, "CT=9216, Σy²=9374, 9254.5, 9237.5, 9308.5", 384 ** 2 / 16 == 9216 and (Yl ** 2).sum() == 9374
      and (91 ** 2 + 100 ** 2 + 89 ** 2 + 104 ** 2) / 4 == 9254.5 and (97 ** 2 + 102 ** 2 + 89 ** 2 + 96 ** 2) / 4 == 9237.5
      and (83 ** 2 + 92 ** 2 + 109 ** 2 + 100 ** 2) / 4 == 9308.5, "ok")
check(T, "S_T=158, S_R=38.5, S_C=21.5, S_τ=92.5, S_E=5.5", allclose([ST, SR, SC, Stau, SE], [158, 38.5, 21.5, 92.5, 5.5], 1e-9),
      r([ST, SR, SC, Stau, SE]))
X = np.column_stack([np.ones(16), Dr[:, :3], Dc[:, :3], Dt[:, :3]])
yv = Yl.ravel()
resid = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
check(T, "S_E=5.5 を最小二乗の残差でも確認（rank 10, df 6）", close((resid ** 2).sum(), 5.5, 1e-9) and np.linalg.matrix_rank(X) == 10,
      round((resid ** 2).sum(), 6))
VEl = SE / 6
Fl = [SR / 3 / VEl, SC / 3 / VEl, Stau / 3 / VEl]
check(T, "V 12.833, 7.167, 30.833, 0.917", allclose([SR / 3, SC / 3, Stau / 3, VEl], [12.833, 7.167, 30.833, 0.917]),
      r([SR / 3, SC / 3, Stau / 3, VEl]))
check(T, "F 14.000, 7.818, 33.636", allclose(Fl, [14.0, 7.818, 33.636]), r(Fl))
pl = [fp(F, 3, 6) for F in Fl]
check(T, "p 0.0041, 0.0170, 0.0004", allclose(pl, [0.0041, 0.0170, 0.0004], 5e-5), r(pl, 5))
check(T, "処理平均 A 20.75, B 23.0, C 27.25, D 25.0", np.allclose(np.array(Tk) / 4, [20.75, 23.0, 27.25, 25.0]), (np.array(Tk) / 4).tolist())
o = oneway([Yl[trt == k] for k in range(4)])
check(T, "例2 一元配置 S_E=65.5, V_E=5.458, F=5.649, p=0.0119",
      allclose([o["SE"], o["VE"], o["F"]], [65.5, 5.458, 5.649]) and close(o["p"], 0.0119, 5e-5),
      r([o["SE"], o["VE"], o["F"], o["p"]], 4))
check(T, "F 比 33.636/5.649≈6", round(Fl[2] / o["F"]) == 6, round(Fl[2] / o["F"], 3))
p_ = sp.symbols("p", positive=True)
check(T, "誤差自由度 p²−1−3(p−1)=(p−1)(p−2)、p=3,4,5 で 2,6,12",
      sp.expand(p_ ** 2 - 1 - 3 * (p_ - 1) - (p_ - 1) * (p_ - 2)) == 0 and [(q - 1) * (q - 2) for q in (3, 4, 5)] == [2, 6, 12], "ok")
check(T, "F_0.05(3,6)=4.757、p=3 なら F_0.05(2,2)=19.0 と高い", close(fcrit(.05, 3, 6), 4.757) and fcrit(.05, 2, 2) > 4.757,
      round(fcrit(.05, 2, 2), 3))
c = charts(T)
check(T, "chart bar 処理平均", c[0]["params"]["values"] == (np.array(Tk) / 4).tolist(), c[0]["params"]["values"])
check(T, "chart F(3,6)", c[1]["params"] == {"df1": 3, "df2": 6}, c[1]["params"])

# =====================================================================
print("=== orthogonal-array ===")
T = "orthogonal-array"
L8 = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 2, 2, 2], [1, 2, 2, 1, 1, 2, 2], [1, 2, 2, 2, 2, 1, 1],
               [2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 2, 1, 2, 1], [2, 2, 1, 1, 2, 2, 1], [2, 2, 1, 2, 1, 1, 2]])
ok = all(sorted(zip(L8[:, i], L8[:, j])).count(pr) == 2 for i, j in combinations(range(7), 2)
         for pr in [(1, 1), (1, 2), (2, 1), (2, 2)])
check(T, "L8 直交性（全21組で各組合せ2回）", ok, ok)
S = np.where(L8 == 1, 1, -1)
check(T, "±1 符号列が互いに直交（S'S=8I）", np.array_equal(S.T @ S, 8 * np.eye(7)), "ok")
comp = {"a": 0, "b": 1, "c": 3}
names = ["a", "b", "ab", "c", "ac", "bc", "abc"]
okc = all(np.array_equal(S[:, k], np.prod([S[:, comp[ch]] for ch in nm], axis=0)) for k, nm in enumerate(names))
check(T, "成分記号 a,b,ab,c,ac,bc,abc が列の積と一致", okc, okc)


def inter_col(i, j):
    prod = S[:, i] * S[:, j]
    return [k + 1 for k in range(7) if np.array_equal(S[:, k], prod)][0]


check(T, "交互作用列: 1×2→3, 3×4→7, 3×5→6, 2×4→6",
      [inter_col(0, 1), inter_col(2, 3), inter_col(2, 4), inter_col(1, 3)] == [3, 7, 6, 6],
      [inter_col(0, 1), inter_col(2, 3), inter_col(2, 4), inter_col(1, 3)])
y8 = np.array([30, 34, 35, 38, 37, 42, 40, 47.])
check(T, "総合計 303", y8.sum() == 303, y8.sum())
T1 = [y8[L8[:, k] == 1].sum() for k in range(7)]
T2 = [y8[L8[:, k] == 2].sum() for k in range(7)]
check(T, "T1 = 137,143,151,142,154,152,150", T1 == [137, 143, 151, 142, 154, 152, 150], T1)
check(T, "T2 = 166,160,152,161,149,151,153", T2 == [166, 160, 152, 161, 149, 151, 153], T2)
SSc = [(a1 - a2) ** 2 / 8 for a1, a2 in zip(T1, T2)]
check(T, "列平方和 105.125,36.125,0.125,45.125,3.125,0.125,1.125",
      allclose(SSc, [105.125, 36.125, 0.125, 45.125, 3.125, 0.125, 1.125], 1e-9), SSc)
STo = ((y8 - y8.mean()) ** 2).sum()
check(T, "合計 190.875 = S_T = Σy²−303²/8", close(sum(SSc), 190.875, 1e-9) and close(STo, 190.875, 1e-9)
      and close((y8 ** 2).sum() - 303 ** 2 / 8, 190.875, 1e-9), STo)
SEo = SSc[4] + SSc[5]
VEo = SEo / 2
Fo = [SSc[k] / VEo for k in (0, 1, 2, 3, 6)]
check(T, "S_E=3.25, V_E=1.625", SEo == 3.25 and VEo == 1.625, VEo)
check(T, "F 64.692, 22.231, 0.077, 27.769, 0.692", allclose(Fo, [64.692, 22.231, 0.077, 27.769, 0.692]), r(Fo))
po = [fp(F, 1, 2) for F in Fo]
check(T, "p 0.0151, 0.0422, 0.8075, 0.0342, 0.4929", allclose(po, [0.0151, 0.0422, 0.8075, 0.0342, 0.4929], 5e-5), r(po, 5))
check(T, "F_0.05(1,2)=18.513、A,B,C 有意", close(fcrit(.05, 1, 2), 18.513) and [F > fcrit(.05, 1, 2) for F in Fo] == [True, True, False, True, False],
      round(fcrit(.05, 1, 2), 4))
check(T, "A の効果 (166−137)/4=7.25", (T2[0] - T1[0]) / 4 == 7.25, 7.25)
c = charts(T)
check(T, "chart bar 列平方和", allclose(c[0]["params"]["values"], SSc, 1e-9), c[0]["params"]["values"])

# =====================================================================
print("=== factorial-design ===")
T = "factorial-design"
data = {"(1)": [20, 22], "a": [28, 26], "b": [24, 25], "ab": [36, 34]}
Tt = {k: sum(v) for k, v in data.items()}
check(T, "合計 42, 54, 49, 70", list(Tt.values()) == [42, 54, 49, 70], list(Tt.values()))
sgn = {"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]}
sgn["AB"] = [x * y for x, y in zip(sgn["A"], sgn["B"])]
check(T, "AB の符号 = A×B（+,−,−,+）・列の直交", sgn["AB"] == [1, -1, -1, 1]
      and sum(np.multiply(sgn["A"], sgn["B"])) == 0 and sum(np.multiply(sgn["A"], sgn["AB"])) == 0, sgn["AB"])
tv = np.array(list(Tt.values()), float)
C = {k: float(np.dot(v, tv)) for k, v in sgn.items()}
check(T, "対比 C_A=33, C_B=23, C_AB=9", [C["A"], C["B"], C["AB"]] == [33, 23, 9], C)
eff = [C[k] / 4 for k in ("A", "B", "AB")]
SSf = [C[k] ** 2 / 8 for k in ("A", "B", "AB")]
check(T, "効果 8.25, 5.75, 2.25", eff == [8.25, 5.75, 2.25], eff)
check(T, "平方和 136.125, 66.125, 10.125", SSf == [136.125, 66.125, 10.125], SSf)
cellm = {k: np.mean(v) for k, v in data.items()}
check(T, "効果 = 高水準平均−低水準平均（A）", (cellm["a"] + cellm["ab"]) / 2 - (cellm["(1)"] + cellm["b"]) / 2 == 8.25, "ok")
check(T, "AB 効果 = (B高での A 効果 − B低での A 効果)/2", ((cellm["ab"] - cellm["b"]) - (cellm["a"] - cellm["(1)"])) / 2 == 2.25, "ok")
SEf = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in data.values())
check(T, "セル内平方和 2+2+0.5+2=6.5, V_E=1.625", SEf == 6.5 and SEf / 4 == 1.625
      and [((np.array(v) - np.mean(v)) ** 2).sum() for v in data.values()] == [2, 2, 0.5, 2], SEf)
Ff = [s / 1.625 for s in SSf]
check(T, "F 83.769, 40.692, 6.231", allclose(Ff, [83.769, 40.692, 6.231]), r(Ff))
pf = [fp(F, 1, 4) for F in Ff]
check(T, "p 0.0008, 0.0031, 0.0670", allclose(pf, [0.0008, 0.0031, 0.0670], 5e-5), r(pf, 5))
allf = np.concatenate(list(data.values())).astype(float)
STf = ((allf - allf.mean()) ** 2).sum()
check(T, "合計 218.875 = 136.125+66.125+10.125+6.5", STf == 218.875 and sum(SSf) + SEf == STf, STf)
check(T, "F_0.05(1,4)=7.709", close(fcrit(.05, 1, 4), 7.709), round(fcrit(.05, 1, 4), 4))
# 2^2 を二元配置（繰り返しあり）として解いても同じ
tw2 = twoway_rep(np.array([[data["(1)"], data["b"]], [data["a"], data["ab"]]], float))
check(T, "二元配置（繰り返しあり）の S_A,S_B,S_A×B と一致", allclose([tw2["SA"], tw2["SB"], tw2["SAB"]], SSf, 1e-9), "ok")
c = charts(T)
ser = [s["y"] for s in c[0]["params"]["series"]]
check(T, "chart line-data セル平均 [21,27],[24.5,35]", ser == [[cellm["(1)"], cellm["a"]], [cellm["b"], cellm["ab"]]], ser)
check(T, "caption p=0.067", close(pf[2], 0.067), round(pf[2], 4))

# =====================================================================
print("=== confounding ===")
T = "confounding"
runs = ["(1)", "ab", "ac", "bc", "a", "b", "c", "abc"]
sg = lambda nm, ch: 1 if ch in nm else -1
A_ = np.array([sg(nm, "a") for nm in runs])
B_ = np.array([sg(nm, "b") for nm in runs])
C_ = np.array([sg(nm, "c") for nm in runs])
ABC = A_ * B_ * C_
check(T, "ブロック1 = ABC− の {(1),ab,ac,bc}, ブロック2 = ABC+ の {a,b,c,abc}",
      ABC.tolist() == [-1, -1, -1, -1, 1, 1, 1, 1], ABC.tolist())
day = np.where(ABC > 0, 2, 1)
yc = 50 + 4 * A_ + 2 * B_ + A_ * B_ + 6 * (day == 2)
check(T, "y の列 45,57,51,47,57,53,51,63（真のモデル＋2日目+6）", yc.tolist() == [45, 57, 51, 47, 57, 53, 51, 63], yc.tolist())
cols = {"A": A_, "B": B_, "C": C_, "AB": A_ * B_, "AC": A_ * C_, "BC": B_ * C_, "ABC": ABC}
effc = {k: yc[v > 0].mean() - yc[v < 0].mean() for k, v in cols.items()}
check(T, "効果 A8,B4,C0,AB2,AC0,BC0,ABC6", [effc[k] for k in ["A", "B", "C", "AB", "AC", "BC", "ABC"]] == [8, 4, 0, 2, 0, 0, 6], effc)
check(T, "A: 57−49, ABC: 56−50", yc[A_ > 0].mean() == 57 and yc[A_ < 0].mean() == 49 and yc[ABC > 0].mean() == 56
      and yc[ABC < 0].mean() == 50, "ok")
okb = all(((v[day == 1] > 0).sum() == 2) and ((v[day == 2] > 0).sum() == 2) for k, v in cols.items() if k != "ABC")
check(T, "ABC 以外は各ブロック内で＋−が2つずつ", okb, okb)
half = ABC > 0
check(T, "I=ABC の半分で A=BC, B=AC, C=AB（対比一致）",
      np.array_equal(A_[half], (B_ * C_)[half]) and np.array_equal(B_[half], (A_ * C_)[half]) and np.array_equal(C_[half], (A_ * B_)[half]),
      "ok")
order = [runs.index(x) for x in ["a", "b", "c", "abc"]]
check(T, "例2 A の対比 (+,−,−,+) = BC の対比", A_[order].tolist() == [1, -1, -1, 1] == (B_ * C_)[order].tolist(), A_[order].tolist())
c = charts(T)
check(T, "chart bar [8,4,0,2,0,0,6]", c[0]["params"]["values"] == [effc[k] for k in ["A", "B", "C", "AB", "AC", "BC", "ABC"]],
      c[0]["params"]["values"])

# =====================================================================
print("=== split-plot-design ===")
T = "split-plot-design"
Ys = np.array([[[40, 44], [46, 51]], [[45, 48], [45, 53]], [[39, 43], [47, 52]]], float)  # [k, i(A), j(B)]
rS, aS, bS = Ys.shape
gm = Ys.mean()
check(T, "総合計 553, Σy²=25699, CT=25484.083", Ys.sum() == 553 and (Ys ** 2).sum() == 25699 and close(553 ** 2 / 12, 25484.083),
      Ys.sum())
check(T, "大区画合計 84,97,93,98,82,99・ブロック 181,191,181・A 259,294・B 262,291・セル 124,135,138,156",
      Ys.sum(2).ravel().tolist() == [84, 97, 93, 98, 82, 99] and Ys.sum((1, 2)).tolist() == [181, 191, 181]
      and Ys.sum((0, 2)).tolist() == [259, 294] and Ys.sum((0, 1)).tolist() == [262, 291]
      and Ys.sum(0).ravel().tolist() == [124, 135, 138, 156], "ok")
STs = ((Ys - gm) ** 2).sum()
SRA = bS * ((Ys.mean(2) - gm) ** 2).sum()
SR = aS * bS * ((Ys.mean((1, 2)) - gm) ** 2).sum()
SA = rS * bS * ((Ys.mean((0, 2)) - gm) ** 2).sum()
SB = rS * aS * ((Ys.mean((0, 1)) - gm) ** 2).sum()
SABc = rS * ((Ys.mean(0) - gm) ** 2).sum()
SE1 = SRA - SR - SA
SAxB = SABc - SA - SB
SE2 = STs - SRA - SB - SAxB
vals = [STs, SRA, SR, SA, SE1, SABc, SB, SAxB, SE2]
check(T, "S_T,S_RA,S_R,S_A,S_E1,S_AB,S_B,S_A×B,S_E2",
      allclose(vals, [214.917, 137.417, 16.667, 102.083, 18.667, 176.25, 70.083, 4.083, 3.333]), r(vals))
# 丸めた値での引き算も本文どおりか
check(T, "本文の丸め値どうしの引き算でも 18.667・4.083・3.333（丸め伝播 ±0.0015）",
      close(137.417 - 16.667 - 102.083, 18.667) and close(176.250 - 102.083 - 70.083, 4.083, 1.5e-3)
      and close(214.917 - 137.417 - 70.083 - 4.083, 3.333, 1.5e-3), "丸め伝播 ±0.0015 以内")
VE1, VE2 = SE1 / 2, SE2 / 4
check(T, "V_R=8.333, V_E1=9.333, V_E2=0.833", allclose([SR / 2, VE1, VE2], [8.333, 9.333, 0.833]), r([SR / 2, VE1, VE2]))
FR, FA, FB, FAB = SR / 2 / VE1, SA / VE1, SB / VE2, SAxB / VE2
check(T, "F R 0.893, A 10.938, B 84.100, A×B 4.900", allclose([FR, FA, FB, FAB], [0.893, 10.938, 84.1, 4.9]), r([FR, FA, FB, FAB]))
ps = [fp(FR, 2, 2), fp(FA, 1, 2), fp(FB, 1, 4), fp(FAB, 1, 4)]
check(T, "p 0.528, 0.0805, 0.0008, 0.0913", close(ps[0], 0.528) and allclose(ps[1:], [0.0805, 0.0008, 0.0913], 5e-5), r(ps, 5))
# 回帰による S_E2 の確認（大区画を因子に入れた残差）
rows = [(k, i, j) for k in range(3) for i in range(2) for j in range(2)]
oh = lambda v, m: [1.0 if v == t else 0.0 for t in range(m)]
Xs = np.array([oh(k * 2 + i, 6) + oh(j, 2)[1:] + [float(i == 1 and j == 1)] for k, i, j in rows])
yy = np.array([Ys[k, i, j] for k, i, j in rows])
rs = yy - Xs @ np.linalg.lstsq(Xs, yy, rcond=None)[0]
check(T, "S_E2 を最小二乗残差でも確認（df 4）", close((rs ** 2).sum(), SE2, 1e-9) and 12 - np.linalg.matrix_rank(Xs) == 4,
      round((rs ** 2).sum(), 4))
check(T, "例2 A を2次誤差で F=122.5, p=0.0004", close(SA / VE2, 122.5) and close(fp(SA / VE2, 1, 4), 0.0004, 5e-5),
      (round(SA / VE2, 3), round(fp(SA / VE2, 1, 4), 5)))
check(T, "1次誤差は2次誤差の10倍以上", VE1 / VE2 > 10, round(VE1 / VE2, 3))
rr_, aa_, bb_ = sp.symbols("r a b", positive=True)
check(T, "自由度の和 = rab−1（sympy）",
      sp.expand((rr_ - 1) + (aa_ - 1) + (rr_ - 1) * (aa_ - 1) + (bb_ - 1) + (aa_ - 1) * (bb_ - 1)
                + aa_ * (rr_ - 1) * (bb_ - 1) - (rr_ * aa_ * bb_ - 1)) == 0, "0")
# EMS シミュレーション（r=3,a=2,b=2, σ1²=2, σ2²=1）
M = 200000
alS = np.array([-1.0, 1.0])
beS = np.array([-0.5, 0.5])
abS = np.array([[0.5, -0.5], [-0.5, 0.5]])
Ysim = (rng.normal(0, 1, (M, 3, 1, 1)) + alS[None, None, :, None] + beS[None, None, None, :] + abS[None, None]
        + rng.normal(0, sqrt(2), (M, 3, 2, 1)) + rng.normal(0, 1, (M, 3, 2, 2)))
g0 = Ysim.mean((1, 2, 3))[:, None]
SRA_ = 2 * ((Ysim.mean(3).reshape(M, 6) - g0) ** 2).sum(1)
SR_ = 4 * ((Ysim.mean((2, 3)) - g0) ** 2).sum(1)
SA_ = 6 * ((Ysim.mean((1, 3)) - g0) ** 2).sum(1)
SB_ = 6 * ((Ysim.mean((1, 2)) - g0) ** 2).sum(1)
SABc_ = 3 * ((Ysim.mean(1).reshape(M, 4) - g0) ** 2).sum(1)
ST_ = ((Ysim.reshape(M, 12) - g0) ** 2).sum(1)
V = dict(A=SA_, E1=(SRA_ - SR_ - SA_) / 2, B=SB_, AB=SABc_ - SA_ - SB_, E2=(ST_ - SRA_ - SB_ - (SABc_ - SA_ - SB_)) / 4)
thS = dict(A=1 + 2 * 2 + 3 * 2 * 2 / 1, E1=1 + 2 * 2, B=1 + 3 * 2 * 0.5 / 1, AB=1 + 3 * 1.0 / 1, E2=1)
okS = all(abs(V[k].mean() - thS[k]) < 4 * V[k].std() / sqrt(M) for k in thS)
check(T, "分割法の期待平均平方表（20万回、±4SE）", okS, {k: (round(V[k].mean(), 3), thS[k]) for k in thS})
c = charts(T)
check(T, "chart bar [9.333, 0.833]", allclose(c[0]["params"]["values"], [VE1, VE2]), c[0]["params"]["values"])

# =====================================================================
print("=== fisher-three-principles ===")
T = "fisher-three-principles"
o1 = oneway(list(Y2))
check(T, "例1 表: 一元 119.000/9/13.222/2.294/0.157, 二元 6.000/6/1.000/30.333/0.0007",
      allclose([o1["SE"], o1["VE"], o1["F"], o1["p"]], [119, 13.222, 2.294, 0.157]) and o1["dfE"] == 9
      and allclose([tn["SE"], tn["VE"], tn["FA"]], [6, 1, 30.333]) and close(tn["pA"], 0.0007, 5e-5) and tn["dfE"] == 6, "ok")
check(T, "S_E(一元)=S_B+S_E(二元)（113+6=119）・自由度 3 減", close(o1["SE"], tn["SB"] + tn["SE"], 1e-9) and o1["dfE"] - tn["dfE"] == 3,
      (o1["SE"], tn["SB"]))
check(T, "例2 A,B,C を3区画ずつ", sorted("CCBACBAAB") == sorted("AAABBBCCC"), "CCBACBAAB")
pw = []
for rr in [2, 4, 8, 16]:
    df = 2 * rr - 2
    nc = 2 / (2 * sqrt(2 / rr))
    tc = stats.t.isf(0.025, df)
    pw.append(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
check(T, "例3 検出力 0.095, 0.223, 0.461, 0.781", allclose(pw, [0.095, 0.223, 0.461, 0.781]), r(pw, 4))
# 検出力のシミュレーション r=4
M = 200000
x1 = rng.normal(0, 2, (M, 4))
x2 = rng.normal(2, 2, (M, 4))
tt = stats.ttest_ind(x1, x2, axis=1)
psim = (tt.pvalue < 0.05).mean()
check(T, "r=4 の検出力シミュ（±0.004）", abs(psim - pw[1]) < 0.004, round(psim, 4))
# 平均の分散 σ²/r
xm = rng.normal(0, 2, (M, 5)).mean(1)
check(T, "V[ȳ]=σ²/r（σ=2,r=5→0.8, ±0.01）", abs(xm.var() - 0.8) < 0.01, round(xm.var(), 4))
c = charts(T)
check(T, "chart1 bar [13.222, 1.0]", allclose(c[0]["params"]["values"], [o1["VE"], tn["VE"]]), c[0]["params"]["values"])
sev = [2 / sqrt(k) for k in range(1, 11)]
check(T, "chart2 σ/√r（小数4桁）", allclose(c[1]["params"]["series"][0]["y"], sev, 5e-5) and c[1]["params"]["x"] == list(range(1, 11)),
      r(sev, 4))


# =====================================================================
# 重要度Cの用語（studentized-range-distribution, effect-size-eta-squared）
# 乱数は seed=20261023+20
# =====================================================================
rngC = np.random.default_rng(20261023 + 20)
from scipy.integrate import quad as quadC

T = "studentized-range-distribution"
qs = [stats.studentized_range.isf(.05, k, 9) for k in (2, 3, 4, 5)]
check(T, "q_0.05(k,9) = 3.199, 3.948, 4.415, 4.755", allclose(qs, [3.199, 3.948, 4.415, 4.755]), r(qs))
tq = stats.t.isf(.025, 9)
check(T, "k=2: √2·t_0.025(9)=1.4142×2.2622=3.199（一致）", close(sqrt(2) * tq, 3.199) and close(tq, 2.2622, 5e-5) and close(qs[0], sqrt(2) * tq, 1e-5), (tq, sqrt(2) * tq, qs[0]))
q320 = stats.studentized_range.isf(.05, 3, 20)
check(T, "q_0.05(3,20)=3.578", close(q320, 3.578), q320)


def Wcdf(w, k):
    return k * quadC(lambda x: stats.norm.pdf(x) * (stats.norm.cdf(x + w) - stats.norm.cdf(x)) ** (k - 1), -12, 12)[0]


from scipy.optimize import brentq as brentqC
qinf = brentqC(lambda w: Wcdf(w, 3) - .95, 2, 5)
check(T, "ν→∞: 3.314、積分式で P(W≤3.314)=0.950", close(qinf, 3.314) and close(Wcdf(3.314, 3), .950), (qinf, Wcdf(3.314, 3)))
check(T, "scipy の大きい ν（10⁵）も 3.314 に近づく", close(stats.studentized_range.isf(.05, 3, 1e5), 3.314, 1e-3), stats.studentized_range.isf(.05, 3, 1e5))
okW = all(close(Wcdf(w, k), stats.studentized_range.cdf(w, k, 1e5), 1e-3) for k in (2, 3, 5) for w in (1, 2.5, 4))
check(T, "範囲の分布関数の積分式＝scipy（ν 大）", okW, "ok")
seq = sqrt(2.333 / 4)
check(T, "テューキー例: √(V_E/n)=0.7638、q=4.5/0.7638=5.892", close(seq, .7638) and close(4.5 / seq, 5.892), (seq, 4.5 / seq))
VEx = 21 / 9
pq_ = stats.studentized_range.sf(4.5 / sqrt(VEx / 4), 3, 9)
check(T, "P(Q>5.892)=0.0062、3.948 を超える", close(pq_, .0062) and 4.5 / sqrt(VEx / 4) > qs[1], pq_)
# 乱数：定義どおり W/S を作って上側5%点（ν=9）
M = 400000
Z = rngC.normal(size=(M, 3))
Wn = Z.max(1) - Z.min(1)
Sn = np.sqrt(rngC.chisquare(9, M) / 9)
Qn = Wn / Sn
check(T, "乱数 40万回：P(Q>3.948)≈0.05（±0.002）、P(W≤3.314)≈0.95（±0.002）", abs((Qn > 3.948).mean() - .05) < .002 and abs((Wn <= 3.314).mean() - .95) < .002,
      ((Qn > 3.948).mean(), (Wn <= 3.314).mean()))
# 一元配置の群平均から作る Q（k=3, n=4）
Yg = rngC.normal(0, 1, (100000, 3, 4))
mg = Yg.mean(2)
VEg = ((Yg - mg[:, :, None]) ** 2).sum((1, 2)) / 9
Qg = (mg.max(1) - mg.min(1)) / np.sqrt(VEg / 4)
check(T, "乱数 10万回：(max ȳ−min ȳ)/√(V_E/n) の棄却率≈0.05（±0.003）", abs((Qg > qs[1]).mean() - .05) < .003, (Qg > qs[1]).mean())
check(T, "分母に √(2V_E/n) を使うと √2 倍ずれる", close(4.5 / sqrt(2 * VEx / 4) * sqrt(2), 4.5 / sqrt(VEx / 4), 1e-12), "ok")
cs = charts(T)[0]
pr_ = cs["params"]
sl = {s["param"]: s for s in cs["sliders"]}
check(T, "chart params k=3, df=9, alpha=0.05（本文の例）、スライダー範囲内", pr_ == {"k": 3, "df": 9, "alpha": 0.05} and all(sl[p]["min"] <= pr_[p] <= sl[p]["max"] for p in ("k", "df")), pr_)
check(T, "k を増やすと q が大きく、ν を増やすと小さく", all(np.diff(qs) > 0) and stats.studentized_range.isf(.05, 3, 60) < q320 < qs[1], "ok")

T = "effect-size-eta-squared"
gE = [[18, 21, 20, 17], [22, 24, 23, 25], [19, 20, 22, 19]]
allE = np.concatenate([np.array(x, float) for x in gE])
SA = sum(4 * (np.mean(x) - allE.mean()) ** 2 for x in gE)
SE = sum(np.sum((np.array(x) - np.mean(x)) ** 2) for x in gE)
ST = np.sum((allE - allE.mean()) ** 2)
VE_ = SE / 9
FE = (SA / 2) / VE_
check(T, "S_A=44.667, S_E=21.000, S_T=65.667, V_E=2.333, F=9.571", allclose([SA, SE, ST, VE_, FE], [44.667, 21, 65.667, 2.333, 9.571]), r([SA, SE, ST, VE_, FE]))
eta2 = SA / ST
rho = (SA - 2 * VE_) / ST
om2 = (SA - 2 * VE_) / (ST + VE_)
check(T, "η²=0.680、純変動 40.000、ρ_A=0.609、ω²=0.588", close(eta2, .680) and close(SA - 2 * VE_, 40, 1e-9) and close(rho, .609) and close(om2, .588), r([eta2, rho, om2]))
check(T, "F から η²=2F/(2F+9)=0.680（本文は丸めた F=9.571 で 19.142/28.142）", close(2 * FE / (2 * FE + 9), eta2, 1e-12) and close(2 * 9.571, 19.142, 1e-9) and close(2 * 9.571 + 9, 28.142, 1e-9) and close(19.142 / 28.142, .680), 2 * FE / (2 * FE + 9))
Dm = np.column_stack([np.ones(12), np.repeat([0, 1, 0], 4), np.repeat([0, 0, 1], 4)])
bd = np.linalg.lstsq(Dm, allE, rcond=None)[0]
R2d = 1 - np.sum((allE - Dm @ bd) ** 2) / ST
check(T, "ダミー変数回帰の R² = η²", close(R2d, eta2, 1e-12), R2d)
FF, pa, pe = sp.symbols("F phi_A phi_E", positive=True)
check(T, "η²=φ_A F/(φ_A F+φ_E)（sympy、S_A=φ_A F V_E, S_E=φ_E V_E）", sp.simplify((pa * FF * sp.Symbol("V", positive=True)) / (pa * FF * sp.Symbol("V", positive=True) + pe * sp.Symbol("V", positive=True)) - (pa * FF) / (pa * FF + pe)) == 0, "恒等式")
check(T, "pingouin 不使用：scipy f_oneway の F と一致", close(stats.f_oneway(*gE).statistic, FE, 1e-9), stats.f_oneway(*gE).statistic)
# 乱数：E[S_A − φ_A V_E] = n Σα²（純変動は効果の分の不偏推定）、η² は過大
al_ = np.array([-1.5, 1.5, 0.0])
YE = al_[None, :, None] + rngC.normal(0, 1.5, (200000, 3, 4))
mE = YE.mean(2); gmE = YE.mean((1, 2))
SAs = 4 * ((mE - gmE[:, None]) ** 2).sum(1)
SEs = ((YE - mE[:, :, None]) ** 2).sum((1, 2))
pure = SAs - 2 * SEs / 9
check(T, "乱数 E[S_A−φ_A V_E]=nΣα²=18（±0.15）", abs(pure.mean() - 18) < .15, pure.mean())
pop_eta = 18 / 12 / (18 / 12 + 1.5 ** 2)
check(T, "乱数 η² の平均は母集団の値より大、ω² は近い", (SAs / (SAs + SEs)).mean() > pop_eta + .03 and abs((pure / (SAs + SEs + SEs / 9)).mean() - pop_eta) < (SAs / (SAs + SEs)).mean() - pop_eta,
      ((SAs / (SAs + SEs)).mean(), (pure / (SAs + SEs + SEs / 9)).mean(), pop_eta))
# 偏 η² の合計が 1 を超えうる例（二元配置、繰り返し 2、効果大）
YA = np.array([[[10, 10.4], [20, 20.2]], [[30, 30.3], [40, 39.9]]])  # [A][B][rep]
gm = YA.mean()
SA2 = 4 * np.sum((YA.mean((1, 2)) - gm) ** 2)
SB2 = 4 * np.sum((YA.mean((0, 2)) - gm) ** 2)
SE2 = np.sum((YA - YA.mean(2, keepdims=True)) ** 2)
check(T, "偏 η² の合計が 1 を超える例（S_A/(S_A+S_E)+S_B/(S_B+S_E)>1）", SA2 / (SA2 + SE2) + SB2 / (SB2 + SE2) > 1, SA2 / (SA2 + SE2) + SB2 / (SB2 + SE2))

# =====================================================================
n_ok = sum(results)
n_ng = len(results) - n_ok
print(f"\n合計: OK {n_ok} / NG {n_ng}")

# --- 追加用語（用語の見直し）: fractional-factorial-design ---
import re as _re, yaml as _yaml
from pathlib import Path as _Path
def _fm20(tid):
    return _yaml.safe_load(_re.match(r"^---\n(.*?)\n---\n", (_Path(__file__).resolve().parent.parent / f"content/ch20/{tid}.md").read_text(encoding="utf-8"), _re.S).group(1))
T = "fractional-factorial-design"
_runs = []
for _C in (-1, 1):
    for _B in (-1, 1):
        for _A in (-1, 1):
            _runs.append((_A, _B, _C, _A * _B * _C))
_R = np.array(_runs)
_names = ["".join(l for l, v in zip("abcd", r) if v > 0) or "(1)" for r in _runs]
_y = np.array([42, 54, 50, 54, 48, 48, 44, 60])
check(T, "例1 実験の並び (1),ad,bd,ab,cd,ac,bc,abcd（D=ABC）", _names == ["(1)", "ad", "bd", "ab", "cd", "ac", "bc", "abcd"], _names)
_Ab, _Bb, _Cb, _Db = _R.T
check(T, "例1 収率は 50+4A+2B+3D+AB（±1）で作った値", np.array_equal(_y, 50 + 4 * _Ab + 2 * _Bb + 3 * _Db + _Ab * _Bb), _y.tolist())
_eff = lambda col: _y[col > 0].mean() - _y[col < 0].mean()
check(T, "例1 効果 A=8, B=4, C=0, D=6, AB=2, AC=0, AD=0",
      [_eff(c) for c in (_Ab, _Bb, _Cb, _Db, _Ab * _Bb, _Ab * _Cb, _Ab * _Db)] == [8, 4, 0, 6, 2, 0, 0], "")
check(T, "例1 平均 54−46、53−47、AB の＋平均51・−平均49",
      (_y[_Ab > 0].mean(), _y[_Ab < 0].mean(), _y[_Db > 0].mean(), _y[_Db < 0].mean(), _y[_Ab * _Bb > 0].mean(), _y[_Ab * _Bb < 0].mean()) == (54, 46, 53, 47, 51, 49), "")
check(T, "別名 A=BCD, B=ACD, C=ABD, D=ABC, AB=CD, AC=BD, AD=BC（列が一致）, ABCD=I",
      all(np.array_equal(p, q) for p, q in [(_Ab, _Bb * _Cb * _Db), (_Bb, _Ab * _Cb * _Db), (_Cb, _Ab * _Bb * _Db), (_Db, _Ab * _Bb * _Cb), (_Ab * _Bb, _Cb * _Db), (_Ab * _Cb, _Bb * _Db), (_Ab * _Db, _Bb * _Cb)]) and np.all(_Ab * _Bb * _Cb * _Db == 1), "")
# 7つの別名の組の列は互いに直交（推定できる組は 2^(4-1)−1=7）
_cols = np.array([_Ab, _Bb, _Cb, _Db, _Ab * _Bb, _Ab * _Cb, _Ab * _Db])
check(T, "7組の対比が互いに直交（2^(k−p)−1=7）", np.array_equal(_cols @ _cols.T, 8 * np.eye(7)), "")
# 例2：2^(5-2), D=AB, E=AC
def _mul(*ws):
    s = set()
    for w in ws:
        s ^= set(w)
    return "".join(sorted(s)) or "I"
_words = sorted({_mul("ABD"), _mul("ACE"), _mul("ABD", "ACE")})
check(T, "例2 定義関係 I=ABD=ACE=BCDE（2^2−1=3語）、分解能 III", _words == ["ABD", "ACE", "BCDE"] and min(map(len, _words)) == 3, _words)
check(T, "例2 A の別名 BD, CE, ABCDE", sorted(_mul("A", w) for w in _words) == ["ABCDE", "BD", "CE"], "")
# 2^(5-2) を実際に作って確認
_R5 = np.array([(a, b, c, a * b, a * c) for c in (-1, 1) for b in (-1, 1) for a in (-1, 1)])
check(T, "例2 計画行列で A=BD=CE、BCDE=I", np.array_equal(_R5[:, 0], _R5[:, 1] * _R5[:, 3]) and np.array_equal(_R5[:, 0], _R5[:, 2] * _R5[:, 4]) and np.all(_R5[:, 1] * _R5[:, 2] * _R5[:, 3] * _R5[:, 4] == 1), "")
check(T, "2^4 の全実施なら 16 回、2^(4−1) は 8 回、L8 は 2^(7−4)", 2 ** 4 == 16 and 2 ** 3 == 8 and 2 ** (7 - 4) == 8, "")
_pts = _fm20(T)["charts"][0]["params"]["points"]
_exp = [{"label": n, "x": round((r[0] + 1) / 2 + 0.45 * (r[2] + 1) / 2, 2), "y": round((r[1] + 1) / 2 + 0.35 * (r[2] + 1) / 2, 2), "group": "D=＋" if r[3] > 0 else "D=−"} for n, r in zip(_names, _runs)]
check(T, "chart：立方体の8頂点と D の符号", _pts == _exp, "")
check(T, "chart：辺で隣り合う頂点で D が入れ替わる", all(_R[i, 3] != _R[j, 3] for i in range(8) for j in range(8) if np.sum(_R[i, :3] != _R[j, :3]) == 1), "")

# --- 追加用語（用語の見直し）: incomplete-block-design ---
T = "incomplete-block-design"
_blocks = [{"A": 7, "B": 5, "C": 6}, {"A": 8, "B": 6, "D": 4}, {"A": 6, "C": 5, "D": 2}, {"B": 5, "C": 6, "D": 3}]
_v, _bb, _k = 4, 4, 3
_r = {t: sum(t in bl for bl in _blocks) for t in "ABCD"}
_lam = {p: sum(p[0] in bl and p[1] in bl for bl in _blocks) for p in combinations("ABCD", 2)}
check(T, "例 v=4, b=4, k=3, r=3, λ=2（すべての対で2回）", set(_r.values()) == {3} and set(_lam.values()) == {2} and all(len(bl) == 3 for bl in _blocks), (_r, _lam))
check(T, "bk=vr=12, λ(v−1)=r(k−1)=6, b≥v", _bb * _k == _v * 3 == 12 and 2 * (_v - 1) == 3 * (_k - 1) == 6 and _bb >= _v, "")
check(T, "効率因子 E=λv/(rk)=8/9≈0.889", close(2 * 4 / (3 * 3), 0.889), 8 / 9)
_Bt = [sum(bl.values()) for bl in _blocks]
_T = {t: sum(bl[t] for bl in _blocks if t in bl) for t in "ABCD"}
_SB = {t: sum(B for B, bl in zip(_Bt, _blocks) if t in bl) for t in "ABCD"}
_Q = {t: _T[t] - _SB[t] / 3 for t in "ABCD"}
_tau = {t: 3 * _Q[t] / (2 * 4) for t in "ABCD"}
check(T, "ブロック合計 18,18,13,14、T=21,16,17,9、含むブロック合計 49,50,45,45",
      _Bt == [18, 18, 13, 14] and [_T[t] for t in "ABCD"] == [21, 16, 17, 9] and [_SB[t] for t in "ABCD"] == [49, 50, 45, 45], "")
check(T, "Q=4.667,−0.667,2.000,−6.000、τ̂=1.750,−0.250,0.750,−2.250",
      allclose([_Q[t] for t in "ABCD"], [4.667, -0.667, 2.0, -6.0]) and allclose([_tau[t] for t in "ABCD"], [1.75, -0.25, 0.75, -2.25]), [round(_tau[t], 3) for t in "ABCD"])
# 最小二乗（処理＋ブロックの加法モデル）で同じ処理効果になる
_rows, _yy = [], []
for j, bl in enumerate(_blocks):
    for t, val in bl.items():
        xr = np.zeros(8); xr["ABCD".index(t)] = 1; xr[4 + j] = 1; _rows.append(xr); _yy.append(val)
_bet = np.linalg.lstsq(np.array(_rows), np.array(_yy, float), rcond=None)[0]
check(T, "ブロック内解析 τ̂=kQ/(λv) は加法モデルの最小二乗と一致", allclose(_bet[:4] - _bet[:4].mean(), [_tau[t] for t in "ABCD"], 1e-9), r(_bet[:4] - _bet[:4].mean()))
_mean = {t: _T[t] / 3 for t in "ABCD"}
check(T, "単純平均 B=5.333, C=5.667（差0.333）、調整後 C−B=1.000",
      close(_mean["B"], 5.333) and close(_mean["C"], 5.667) and close(_mean["C"] - _mean["B"], 0.333) and close(_tau["C"] - _tau["B"], 1.0), "")
# 差の分散 2kσ²/(λv)（σ²=1）を最小二乗の共分散から確認
_X = np.array(_rows); _cov = np.linalg.pinv(_X.T @ _X)
_cvec = np.zeros(8); _cvec[0], _cvec[1] = 1, -1
check(T, "処理差の分散 2kσ²/(λv)=6/8=0.75、乱塊法 2σ²/r=0.667 との比=E", close(_cvec @ _cov @ _cvec, 0.75, 1e-9) and close((2 / 3) / 0.75, 8 / 9, 1e-9), _cvec @ _cov @ _cvec)

n_ok = sum(results)
n_ng = len(results) - n_ok
print(f"\n合計（追加用語を含む）: OK {n_ok} / NG {n_ng}")
