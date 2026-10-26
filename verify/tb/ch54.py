"""教科書 第54章（ベイズ計算と発展的なベイズモデル）の検算。"""
import numpy as np
from scipy import stats

from hmc_sim import acf, ess, hmc, rwmh
from tbcheck import allclose, body, chart, check, close, done

T = "variational-inference"
p = chart(T)
check(T, "図：√(1−ρ²) と −½log(1−ρ²)", allclose(p["series"][1]["y"], [np.sqrt(1 - r * r) for r in p["x"]], 6e-5)
      and allclose(p["series"][2]["y"], [-0.5 * np.log(1 - r * r) for r in p["x"]], 6e-5))
# 例1：平均場の最適解を数値で求める（q=N(0,diag(v1,v2)) の KL を最小化）
rho = 0.9
Sg = np.array([[1, rho], [rho, 1.0]])
Si = np.linalg.inv(Sg)
kl = lambda v: 0.5 * (np.trace(Si @ np.diag(v)) - 2 + np.log(np.linalg.det(Sg) / np.prod(v)))
from scipy.optimize import minimize
opt = minimize(lambda t: kl(np.exp(t)), [0.0, 0.0])
check(T, "例1 ρ=0.9：最適な分散 1−ρ²=0.19・SD 0.436・KL 0.830", allclose(np.exp(opt.x), [0.19, 0.19], 1e-4) and close(np.sqrt(0.19), 0.436, 5e-4)
      and close(opt.fun, -0.5 * np.log(0.19), 1e-6) and close(opt.fun, 0.830, 5e-4))
x = np.array([4.8, 5.6, 5.1, 4.3, 5.9, 5.0, 6.2, 4.7, 5.4, 5.0])
n, xb = len(x), x.mean()
mu0, l0, a0, b0 = 0.0, 0.01, 1.0, 1.0
ln, mun, an = l0 + n, (l0 * mu0 + n * xb) / (l0 + n), a0 + n / 2
bn = b0 + 0.5 * ((x - xb) ** 2).sum() + l0 * n * (xb - mu0) ** 2 / (2 * ln)
sd_exact = stats.t(2 * an, mun, np.sqrt(bn / (an * ln))).std()
Et = 1.0
for _ in range(200):
    m = (l0 * mu0 + n * xb) / (l0 + n)
    lam = (l0 + n) * Et
    Emu2 = 1 / lam + m * m
    bN = b0 + 0.5 * (((x ** 2).sum() - 2 * m * x.sum() + n * Emu2) + l0 * (Emu2 - 2 * m * mu0 + mu0 ** 2))
    Et = (a0 + (n + 1) / 2) / bN
check(T, "例2 CAVI：平均 5.195（正確と一致）・E[τ] 2.277・SD 0.209 vs 正確 0.229（約 9% 小さい）",
      close(m, mun, 1e-12) and close(m, 5.195, 5e-4) and close(Et, 2.277, 5e-4) and close(an / bn, 2.277, 5e-4)
      and close(1 / np.sqrt(lam), 0.209, 5e-4) and close(sd_exact, 0.229, 5e-4) and close(1 - (1 / np.sqrt(lam)) / sd_exact, 0.09, 0.005), (1 / np.sqrt(lam), sd_exact))
# 正確な事後分布を MCMC 不要の直接サンプリングで確かめる
rng = np.random.default_rng(540)
tau = rng.gamma(an, 1 / bn, 400000)
mus = rng.normal(mun, 1 / np.sqrt(ln * tau))
check(T, "例2 正確な事後 SD 0.229（正規・ガンマ事後からの直接サンプリング）", close(mus.std(), 0.229, 0.002), mus.std())

T = "hamiltonian-monte-carlo"
p = chart(T)
a, ra = rwmh(step=1.0)
b, rb = hmc()
check(T, "図：自己相関（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], [acf(a[:, 0], k) for k in p["x"]], 6e-4)
      and allclose(p["series"][1]["y"], [acf(b[:, 0], k) for k in p["x"]], 6e-4))
check(T, "例 採択率 0.23・0.98、有効標本 476・11665、勾配 1 回あたり 0.039・0.024",
      close(ra, 0.23, 0.005) and close(rb, 0.98, 0.005) and close(ess(a[:, 0]), 476, 1) and close(ess(b[:, 0]), 11665, 1)
      and close(ess(b[:, 0]) / (20000 * 15), 0.039, 5e-4) and close(ess(a[:, 0]) / 20000, 0.024, 5e-4))
check(T, "どちらも目標の分布（分散 1・相関 0.95）を再現（ランダムウォークは有効標本が少ないので幅を広く）", close(a[:, 0].var(), 1, 0.15) and close(b[:, 0].var(), 1, 0.03)
      and close(np.corrcoef(b.T)[0, 1], 0.95, 0.005))

T = "gaussian-process-regression"
X = np.array([0.05, 0.2, 0.35, 0.5, 0.62, 0.95])
rng = np.random.default_rng(54)
y = np.sin(2 * np.pi * X) + rng.normal(0, 0.1, 6)
k = lambda u, v, l=0.15: np.exp(-(u[:, None] - v[None, :]) ** 2 / (2 * l * l))
p = chart(T)
g = np.array(p["x"], float)
K = k(X, X) + 0.01 * np.eye(6)
Ks = k(g, X)
mpost = Ks @ np.linalg.solve(K, y)
v = 1 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
check(T, "図：事後平均と 95% 区間・真の関数", allclose(p["series"][0]["y"], mpost, 6e-4) and allclose(p["series"][1]["y"], mpost + 1.96 * np.sqrt(v), 6e-4)
      and allclose(p["series"][2]["y"], mpost - 1.96 * np.sqrt(v), 6e-4) and allclose(p["series"][3]["y"], np.sin(2 * np.pi * g), 6e-4))
check(T, "例 観測値 0.212・0.931・0.934・0.175・−0.736・−0.179", allclose(y, [0.212, 0.931, 0.934, 0.175, -0.736, -0.179], 5e-4))
i5, i8 = 10, 16
check(T, "例 x=0.5：平均 0.161・SD 0.098、x=0.8：平均 −0.664・SD 0.592、区間 [−1.824, 0.497] に真の値 −0.951",
      close(mpost[i5], 0.161, 5e-4) and close(np.sqrt(v[i5]), 0.098, 5e-4) and close(mpost[i8], -0.664, 5e-4) and close(np.sqrt(v[i8]), 0.592, 5e-4)
      and -1.824 < np.sin(2 * np.pi * 0.8) < 0.497)
# 事後分布を、観測を 1 個ずつ取り込む逐次更新（一括の公式とは別の計算順）で求め直す
allx = np.r_[g, X]
mu_s, C_s = np.zeros(len(allx)), k(allx, allx)
for i in range(6):
    j = len(g) + i
    gain = C_s[:, j] / (C_s[j, j] + 0.01)
    mu_s = mu_s + gain * (y[i] - mu_s[j])
    C_s = C_s - np.outer(gain, C_s[j, :])
check(T, "観測を 1 個ずつ取り込む逐次更新でも同じ事後平均・事後分散", allclose(mu_s[:len(g)], mpost, 1e-9) and allclose(np.diag(C_s)[:len(g)], v, 1e-9))


def lml(l):
    Kl = k(X, X, l) + 0.01 * np.eye(6)
    return -0.5 * y @ np.linalg.solve(Kl, y) - 0.5 * np.linalg.slogdet(Kl)[1] - 3 * np.log(2 * np.pi)


ls = np.linspace(0.05, 0.6, 5501)
check(T, "例 周辺尤度は ℓ≈0.24 で最大", close(ls[np.argmax([lml(l) for l in ls])], 0.24, 0.005))

T = "waic"
yv = np.array([-1.89, 0.2, -0.3, -0.82, -0.74, -0.57, 2.16, 1.27, -0.73, 0.74, 0.3, -0.64, 0.12, 1.78, -0.37, -0.12, 2.47, -0.12, -1.06, -0.41])
nn, t2 = len(yv), 100.0


def post(yy):
    s2 = 1 / (len(yy) + 1 / t2)
    return s2 * yy.sum(), s2


m, s2 = post(yv)
lppd = stats.norm.logpdf(yv, m, np.sqrt(1 + s2)).sum()
pw = ((yv - m) ** 2 * s2 + s2 * s2 / 2).sum()
loo = -2 * sum(stats.norm.logpdf(yv[i], post(np.delete(yv, i))[0], np.sqrt(1 + post(np.delete(yv, i))[1])) for i in range(nn))
check(T, "例 平均 0.0635・事後 N(0.0635, 0.0500)・lppd −30.18・p 1.21・WAIC 62.79・LOO 62.80・AIC 62.52・モデル A 60.60",
      close(yv.mean(), 0.0635, 5e-5) and close(m, 0.0635, 5e-4) and close(s2, 0.0500, 5e-5) and close(lppd, -30.18, 0.005) and close(pw, 1.21, 0.005)
      and close(-2 * (lppd - pw), 62.79, 0.005) and close(loo, 62.80, 0.005) and close(-2 * stats.norm.logpdf(yv, yv.mean(), 1).sum() + 2, 62.52, 0.005)
      and close(-2 * stats.norm.logpdf(yv, 0, 1).sum(), 60.60, 0.005))
mu = np.random.default_rng(5410).normal(m, np.sqrt(s2), 400000)
L = stats.norm.logpdf(yv[None, :], mu[:, None], 1)
check(T, "事後標本から計算した WAIC も 62.79 付近", close(-2 * (np.log(np.exp(L).mean(0)).sum() - L.var(0).sum()), 62.79, 0.02))
check(T, "本文の値", all(s in body(T) for s in ("60.60", "62.79", "62.80", "62.52", "1.21")))
T = "dirichlet-process"
EK = lambda a, n: sum(a / (a + i) for i in range(n))
p = chart(T)
check(T, "図：E[K] = Σα/(α+i)", all(allclose(p["series"][j]["y"], [EK(a, n) for n in p["x"]], 6e-4) for j, a in enumerate((0.5, 1, 5))))
rng4 = np.random.default_rng(54)


def crp(a, n):
    counts = []
    for i in range(n):
        pr = np.array(counts + [a], float)
        kk = rng4.choice(len(pr), p=pr / pr.sum())
        if kk == len(counts):
            counts.append(1)
        else:
            counts[kk] += 1
    return counts


K = [len(crp(1, 100)) for _ in range(2000)]
check(T, "例1 E[K]=5.19（log100+0.577≈5.18）・シミュレーション 5.17・n=1000 で 7.49",
      close(EK(1, 100), 5.19, 0.005) and close(np.log(100) + 0.5772, 5.18, 0.005) and close(np.mean(K), 5.17, 0.005) and close(EK(1, 1000), 7.49, 0.005), np.mean(K))
same = [crp(1, 2) for _ in range(20000)]
v1 = np.random.default_rng(5).beta(1, 1, 200000)
from scipy.special import digamma  # noqa: E402
check(T, "E[K] = α{ψ(α+n)−ψ(α)}（α=1, n=100）", close(EK(1, 100), digamma(101) - digamma(1), 1e-10))
check(T, "例2 2 人が同じクラスター 1/(1+α)：α=1 で 0.5・α=5 で 0.17、w₁ の期待値も同じ", close(np.mean([len(c) == 1 for c in same]), 0.5, 0.01)
      and close(1 / 6, 0.17, 0.005) and close(v1.mean(), 0.5, 0.005) and close(np.random.default_rng(6).beta(1, 5, 200000).mean(), 1 / 6, 0.003))
# 有限分割でディリクレ分布：棒折りで作った G の G(A) の期待値が G₀(A)、分散が G₀(A)(1−G₀(A))/(α+1)
a_, M = 2.0, 400
rng5 = np.random.default_rng(7)
vals = []
for _ in range(4000):
    V = rng5.beta(1, a_, M)
    w = V * np.r_[1, np.cumprod(1 - V)[:-1]]
    th = rng5.random(M)
    vals.append(w[th < 0.3].sum())
check(T, "棒折りの G(A) は Beta(αG₀(A), α(1−G₀(A)))：平均 0.3・分散 0.21/3=0.07", close(np.mean(vals), 0.3, 0.01) and close(np.var(vals), 0.07, 0.005), (np.mean(vals), np.var(vals)))
done("第54章（ベイズ計算と発展的なベイズモデル）")
