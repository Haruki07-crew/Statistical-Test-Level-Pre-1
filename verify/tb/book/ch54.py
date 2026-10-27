"""統計学の教科書 第54章「ベイズ計算と発展的なベイズモデル」の検算（リープフロッグ・HMC・GP・CRP・WAIC）。"""
import math

import numpy as np
from bookcheck import Book

b = Book("tb", 54)
rng = np.random.default_rng(54)


def leapfrog(q, p, eps, L, grad=lambda q: q):
    p = p - eps / 2 * grad(q)
    for i in range(L):
        q = q + eps * p
        if i < L - 1:
            p = p - eps * grad(q)
    p = p - eps / 2 * grad(q)
    return q, p


q, p = leapfrog(1.0, 0.0, 0.5, 4)
b.num("q", q, "−0.436")
b.num("p", p, "−0.872")
b.num("cos2", math.cos(2), "−0.416")
dH = (q * q + p * p) / 2 - 0.5
b.num("ΔH", dH, "−0.025")
q2, p2 = leapfrog(1.0, 0.0, 0.1, 20)
b.num("ε=0.1 の |ΔH|", abs((q2 * q2 + p2 * p2) / 2 - 0.5), "0.001")
q3, p3 = leapfrog(1.0, 0.0, 2.1, 30)
b.check("ε>2 で発散", abs(q3) > 100, q3)
q4, p4 = leapfrog(1.0, 0.0, 1.9, 300)
b.check("ε<2 では有界", abs(q4) < 10, q4)
# HMC が正しい分布を出す（相関 0.95 の 2 変量正規）
S = np.array([[1, 0.95], [0.95, 1]]); Si = np.linalg.inv(S)
x = np.zeros(2); out = []
for _ in range(20000):
    p0 = rng.standard_normal(2)
    xn, pn = leapfrog(x, p0, 0.12, 15, grad=lambda v: Si @ v)
    H0 = x @ Si @ x / 2 + p0 @ p0 / 2
    H1 = xn @ Si @ xn / 2 + pn @ pn / 2
    if rng.uniform() < math.exp(min(0, H0 - H1)):
        x = xn
    out.append(x)
out = np.array(out[1000:])
b.check("HMC の標本の相関 ≈ 0.95・分散 ≈ 1", abs(np.corrcoef(out.T)[0, 1] - 0.95) < 0.01 and abs(out[:, 0].var() - 1) < 0.08, (np.corrcoef(out.T)[0, 1], out[:, 0].var()))
# 変分推論（平均場）
rho = 0.9
b.num("平均場の SD", math.sqrt(1 - rho ** 2), "0.436")
# CAVI で確認：q_j の精度は Λ_jj、平均は反復で 0 に。分散 = 1/Λ_jj = 1−ρ²
Lam = np.linalg.inv(np.array([[1, rho], [rho, 1]]))
b.check("CAVI の分散 1/Λ_jj = 1−ρ²", abs(1 / Lam[0, 0] - (1 - rho ** 2)) < 1e-12, "")
# KL(q‖p) を数値で（q=N(0,diag(1−ρ²)), p=N(0,S)）: ½[tr(Λ Σq) − 2 + log det S/ det Σq]
Sq = np.diag([1 - rho ** 2] * 2)
kl = 0.5 * (np.trace(Lam @ Sq) - 2 + math.log(np.linalg.det(np.array([[1, rho], [rho, 1]])) / np.linalg.det(Sq)))
b.check("KL = −½ log(1−ρ²)", abs(kl + 0.5 * math.log(1 - rho ** 2)) < 1e-12, kl)
# ガウス過程
k = lambda a, c: np.exp(-(a - c) ** 2 / 2)
X = np.array([0.0, 1.0]); y = np.array([1.0, 0.0])
K = k(X[:, None], X[None, :]) + 0.1 * np.eye(2)
b.num("非対角", K[0, 1], "0.607")
for xs, mm, vv in ((0.5, "0.517", "0.087"), (3.0, "−0.083", "0.978")):
    ks = k(xs, X)
    b.num(f"事後平均 x={xs}", ks @ np.linalg.solve(K, y), mm)
    b.num(f"事後分散 x={xs}", 1 - ks @ np.linalg.solve(K, ks), vv)
b.num("e^{-1/8}", math.exp(-1 / 8), "0.882")
# 条件付き正規のシミュレーション
F = rng.multivariate_normal(np.zeros(3), k(np.array([0, 1, 0.5])[:, None], np.array([0, 1, 0.5])[None, :]), 400000)
Y = F[:, :2] + rng.normal(0, math.sqrt(0.1), (400000, 2))
sel = (abs(Y[:, 0] - 1) < 0.05) & (abs(Y[:, 1]) < 0.05)
b.check("シミュレーションの条件付き平均 ≈ 0.517", abs(F[sel, 2].mean() - 0.517) < 0.03, F[sel, 2].mean())
# ディリクレ過程
ek = sum(1 / (1 + i) for i in range(100))
b.num("E[K_100] (α=1)", ek, "5.19")
crp = []
for _ in range(4000):
    tables = []
    for n in range(100):
        if rng.uniform() < 1 / (1 + n):
            tables.append(1)
        else:
            j = rng.choice(len(tables), p=np.array(tables) / n)
            tables[j] += 1
    crp.append(len(tables))
b.check("中華料理店過程のシミュレーション ≈ 5.19", abs(np.mean(crp) - ek) < 0.08, np.mean(crp))
b.check("近似 α log(1+n/α) は同じ大きさ", abs(math.log(101) - 4.62) < 0.01, "")
# WAIC
yv = np.array([2.1, 3.4, 1.8, 2.9, 4.2, 2.5, 3.1, 2.0])
n = len(yv)
th = rng.normal(yv.mean(), 1 / math.sqrt(n), 400000)
ll = -0.5 * math.log(2 * math.pi) - 0.5 * (yv[None, :] - th[:, None]) ** 2
lppd = np.log(np.exp(ll).mean(0)).sum()
pw = ll.var(0).sum()
d = yv - yv.mean()
lppd_x = (-0.5 * np.log(2 * math.pi * (1 + 1 / n)) - d ** 2 / (2 * (1 + 1 / n))).sum()
pw_x = (d ** 2).sum() / n + 1 / (2 * n)  # Var[(d−e)²/2], e ~ N(0, 1/n)
b.num("p_WAIC（厳密）", pw_x, "0.64")
b.num("WAIC（厳密）", -2 * (lppd_x - pw_x), "21.03")
b.check("事後標本による WAIC も一致", abs(-2 * (lppd - pw) - 21.032) < 0.01 and abs(pw - pw_x) < 0.005, (-2 * (lppd - pw), pw))
loo = 0
for i in range(n):
    m = np.delete(yv, i).mean(); v = 1 + 1 / (n - 1)
    loo += -0.5 * math.log(2 * math.pi * v) - (yv[i] - m) ** 2 / (2 * v)
b.num("LOO (−2倍)", -2 * loo, "21.05")
# 章末問題
b.num("問1", math.exp(-0.3), "0.741")
b.num("問2", 2 * math.sqrt(1 - 0.36), "1.6")
b.num("問3 B", 1 / 6, "0.167")
b.num("問3 新", 2 / 6, "0.333")
b.done()
