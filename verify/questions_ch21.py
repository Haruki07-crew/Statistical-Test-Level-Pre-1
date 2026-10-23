"""第21章（標本調査法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import itertools
from math import comb
import numpy as np

q = Q(21)
rng = np.random.default_rng(21)

# --- 単純無作為抽出 ---
se_ = np.sqrt((1 - 50 / 500) * 36 / 50)
q.check("q21-simple-random-sampling-1", round(se_, 3) == 0.805 and round(np.sqrt(36 / 50), 3) == 0.849
        and round(0.9 * np.sqrt(36 / 50), 3) == 0.764 and round(se_ ** 2, 3) == 0.648, f"正解 {se_:.4f}、誤答 0.849/0.764/0.648 は別の値")
n_ = next(n for n in range(1, 2001) if (1 - n / 2000) * 25 / n <= 0.25 + 1e-12)
q.num("q21-simple-random-sampling-2", n_)
q.check("q21-simple-random-sampling-2", (1 - 95 / 2000) * 25 / 95 > 0.25, "n=95 では条件を満たさない")
Y = np.array([1, 2, 4, 7.]); S2 = np.var(Y, ddof=1)
s2s = [np.var(Y[list(s)], ddof=1) for s in itertools.combinations(range(4), 2)]
vbar = np.var([Y[list(s)].mean() for s in itertools.combinations(range(4), 2)])
q.check("q21-simple-random-sampling-3", abs(np.mean(s2s) - S2) < 1e-12 and abs(vbar - (1 - 2 / 4) * S2 / 2) < 1e-12, "全標本の列挙で E[s²]=S²、V(ȳ)=(1-f)S²/n")

# --- HT ---
q.num("q21-horvitz-thompson-estimator-1", 12 / 0.2 + 30 / 0.5 + 8 / 0.1)
y = np.array([2, 4, 9.]); P = {(0, 1): .5, (0, 2): .3, (1, 2): .2}
pi = np.zeros(3)
for s, p in P.items():
    for i in s:
        pi[i] += p
est = {s: sum(y[i] / pi[i] for i in s) for s in P}
E = sum(P[s] * est[s] for s in P); V = sum(P[s] * (est[s] - E) ** 2 for s in P)
q.num("q21-horvitz-thompson-estimator-2", V)
# 公式（SYG形）でも
pij = {s: P[s] for s in P}
syg = 0.5 * sum((pi[i] * pi[j] - pij[tuple(sorted((i, j)))]) * (y[i] / pi[i] - y[j] / pi[j]) ** 2
                for i in range(3) for j in range(3) if i != j)
q.check("q21-horvitz-thompson-estimator-2", abs(E - y.sum()) < 1e-12 and abs(syg - V) < 1e-9 and np.allclose(pi, [.8, .7, .5]), "不偏、SYG形の分散と一致")
# π ∝ y の固定サイズデザイン：y=(1,2,3,4)、n=2 で π=y/5 を実現するデザインを作り分散0を確認
yy = np.array([1, 2, 3, 4.]); target = 2 * yy / yy.sum()
pairs = list(itertools.combinations(range(4), 2))
# π_i = Σ_{s∋i} p(s) を満たす非負解を線形計画で求める
from scipy.optimize import linprog
Aeq = np.array([[1.0 if i in s else 0.0 for s in pairs] for i in range(4)] + [[1.0] * len(pairs)])
res = linprog(np.zeros(len(pairs)), A_eq=Aeq, b_eq=np.r_[target, 1], bounds=[(0, 1)] * len(pairs))
vals = [sum(yy[i] / target[i] for i in s) for s, p in zip(pairs, res.x) if p > 1e-12]
q.check("q21-horvitz-thompson-estimator-3", res.status == 0 and np.allclose(vals, yy.sum()), "π∝y のデザインで全標本の推定値が Y（分散0）")

# --- 包含確率 ---
q.num("q21-inclusion-probability-1", 10 / 50 * 8 / 40)
q.num("q21-inclusion-probability-2", 4 * 3 / (10 * 9))
q.check("q21-inclusion-probability-2", abs(comb(8, 2) / comb(10, 4) - 4 * 3 / 90) < 1e-12, "数え上げと一致")

# --- 層化抽出 ---
W = np.array([.6, .4]); S = np.array([10, 20.]); Nh = np.array([600, 400]); nh = np.array([30, 20])
q.num("q21-stratified-sampling-1", W @ [50, 80])
Vst = np.sum(W ** 2 * (1 - nh / Nh) * S ** 2 / nh)
q.num("q21-stratified-sampling-2", Vst)
# シミュレーションで確認
pop = [rng.normal(50, 10, 600), rng.normal(80, 20, 400)]
pop = [(p - p.mean()) / p.std(ddof=1) * s + m for p, s, m in zip(pop, S, [50, 80])]
ests = [W[0] * rng.choice(pop[0], 30, replace=False).mean() + W[1] * rng.choice(pop[1], 20, replace=False).mean() for _ in range(40000)]
q.check("q21-stratified-sampling-2", abs(np.var(ests) / Vst - 1) < 0.03, f"シミュレーション分散 {np.var(ests):.3f}")

# --- 比例配分 ---
q.num("q21-proportional-allocation-1", 200 * 3000 / 10000)
samp = [rng.normal(size=40), rng.normal(3, 1, size=100), rng.normal(-1, 2, size=60)]
Wp = np.array([2000, 5000, 3000]) / 10000
q.check("q21-proportional-allocation-2", abs(sum(w * s.mean() for w, s in zip(Wp, samp)) - np.concatenate(samp).mean()) < 1e-12, "ȳ_st=ȳ")

# --- ネイマン配分 ---
NS = Nh * S
n2_ = 50 * NS[1] / NS.sum()
q.check("q21-neyman-allocation-1", round(n2_, 1) == 28.6 and round(50 * Nh[1] / Nh.sum(), 1) == 20.0
        and round(50 * S[1] / S.sum(), 1) == 33.3 and round(50 * (Nh * S ** 2)[1] / (Nh * S ** 2).sum(), 1) == 36.4,
        f"正解 {n2_:.3f}、誤答 20.0（比例）/33.3（S比例）/36.4（NS²比例）")
Vopt = (W @ S) ** 2 / 50 - (W @ S ** 2) / 1000
q.num("q21-neyman-allocation-2", Vopt)
nopt = 50 * NS / NS.sum()
q.check("q21-neyman-allocation-2", abs(np.sum(W ** 2 * (1 - nopt / Nh) * S ** 2 / nopt) - Vopt) < 1e-9 and Vopt < Vst, "一般式に n_h を代入した値と一致、比例配分より小")
c = np.array([4, 16.]); k = NS / np.sqrt(c)
q.num("q21-neyman-allocation-3", 50 * k[0] / k.sum())
# 費用一定 C=Σ c_h n_h で分散最小を数値で確認
C = c @ (50 * k / k.sum())
grid = np.linspace(1, C / c[0] - 1, 20000)
v = [np.sum(W ** 2 * (1 - np.array([n1, (C - c[0] * n1) / c[1]]) / Nh) * S ** 2 / np.array([n1, (C - c[0] * n1) / c[1]])) for n1 in grid]
q.check("q21-neyman-allocation-3", abs(grid[int(np.argmin(v))] - 30) < 0.05 and abs(50 * (NS / c)[0] / (NS / c).sum() - 37.5) < 1e-9, "数値最小化で n1=30、誤答 37.5")

# --- 集落抽出 ---
q.num("q21-cluster-sampling-1", (1 - 10 / 100) * 4.5 / 10)
# 級内相関と分散：集落効果を強めると分散が増える
def cl_var(sb):
    A, M = 100, 20
    popc = rng.normal(scale=sb, size=(A, 1)) + rng.normal(size=(A, M))
    Sb2 = np.var(popc.mean(1), ddof=1)
    return (1 - 10 / A) * Sb2 / 10
q.check("q21-cluster-sampling-2", cl_var(2.0) > cl_var(0.1), "集落内が似ているほど分散が大きい（誤り）")

# --- 二段抽出 ---
q.num("q21-two-stage-sampling-1", (1 - 10 / 80) * 6 / 10 + (1 - 10 / 50) * 40 / 100)
q.check("q21-two-stage-sampling-2", (1 - 80 / 80) * 6 / 80 == 0, "a=A で第1項0（層化抽出）")

# --- 系統抽出 ---
k_ = 1200 // 40
q.num("q21-systematic-sampling-1", 7 + 4 * k_)
Np, kk = 1000, 10
yper = 10 + 5 * np.sin(2 * np.pi * np.arange(Np) / kk)  # 周期 = k
vsys = np.var([yper[r::kk].mean() for r in range(kk)])
vsrs = (1 - 100 / Np) * np.var(yper, ddof=1) / 100
q.check("q21-systematic-sampling-2", vsys > 10 * vsrs, f"周期=k で系統抽出の分散 {vsys:.3f} ≫ SRS {vsrs:.4f}")

# --- 比推定 ---
q.num("q21-ratio-estimation-1", 36 / 45 * 50)
q.num("q21-ratio-estimation-2", 0.9 / 40 * (100 - 2 * 1.2 * 72 + 1.2 ** 2 * 64))
q.check("q21-ratio-estimation-2", abs(72 / (10 * 8) - 0.9) < 1e-12 and abs(0.9 / 40 * 100 - 2.25) < 1e-12, "相関0.9、SRS分散2.25")
# 小母集団の全標本列挙で偏りが0でないこと
xs = np.array([1, 2, 3, 10.]); ys = np.array([3, 2, 5, 12.])
yR = [ys[list(s)].mean() / xs[list(s)].mean() * xs.mean() for s in itertools.combinations(range(4), 2)]
q.check("q21-ratio-estimation-3", abs(np.mean(yR) - ys.mean()) > 1e-3, f"全標本列挙で E[ȳ_R]={np.mean(yR):.4f} ≠ Ȳ={ys.mean()}（誤り）")

# --- 回帰推定 ---
q.check("q21-regression-estimation-1", 36 + 0.6 * (50 - 45) == 39 and 36 / 45 * 50 == 40
        and abs(36 + 0.6 * (45 - 50) - 33) < 1e-12 and 36 + (50 - 45) == 41, "正解 39、誤答 40（比推定）/33（符号逆）/41（傾き1）")

# --- 計画効果 ---
ne_ = 600 / (1 + 19 * 0.05)
q.check("q21-design-effect-1", round(ne_) == 308 and 600 / (1 + 20 * 0.05) == 300 and round(600 * 1.95) == 1170,
        f"正解 {ne_:.2f}≈308、誤答 300/1170/30 は別の値")
w = np.r_[np.ones(50), 3 * np.ones(50)]
deff = 100 * np.sum(w ** 2) / w.sum() ** 2
q.num("q21-design-effect-2", deff)
q.check("q21-design-effect-2", abs(1 + (w.std() / w.mean()) ** 2 - deff) < 1e-12, "1+CV² と一致")

# --- 標本誤差・非標本誤差 ---
q.concept("q21-sampling-and-nonsampling-error-1", "全数調査でも非標本誤差は残る")
q.num("q21-sampling-and-nonsampling-error-2", np.sqrt((2 / 2) ** 2 + 1.5 ** 2))
q.check("q21-sampling-and-nonsampling-error-2", abs(np.sqrt(4 + 2.25) - 2.5) < 1e-12, "もとの RMSE 2.5")

# --- PPS ---
q.num("q21-probability-proportional-to-size-sampling-1", (30 / 0.1 + 50 / 0.2) / 2)

# --- 無回答バイアス ---
q.check("q21-nonresponse-bias-1", abs(60 - (0.6 * 60 + 0.4 * 45) - 6) < 1e-12 and abs(0.4 * (60 - 45) - 6) < 1e-12 and 6 != 15, "偏りは 6（15 ではない＝誤り）")
popr = np.r_[rng.normal(60, 10, 60000), rng.normal(45, 10, 40000)]; resp = np.r_[np.ones(60000), np.zeros(40000)].astype(bool)
b_small = np.mean([popr[s][resp[s]].mean() for s in (rng.choice(100000, 200, replace=False) for _ in range(500))]) - popr.mean()
b_large = np.mean([popr[s][resp[s]].mean() for s in (rng.choice(100000, 20000, replace=False) for _ in range(50))]) - popr.mean()
q.check("q21-nonresponse-bias-2", abs(b_small - b_large) < 0.5 and b_large > 5, f"n=200 の偏り {b_small:.2f}、n=20000 の偏り {b_large:.2f}（減らない＝誤り）")

# --- 標本ウェイト ---
q.num("q21-sampling-weight-1", 5000 / 50 / 0.8)
q.check("q21-sampling-weight-1", 40 * 5000 / 50 / 0.8 == 5000, "ウェイト合計=N_h")

q.done()
