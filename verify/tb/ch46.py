"""教科書 第46章（漸近理論の発展）の検算。"""
import numpy as np
from scipy import stats

from tbcheck import allclose, body, chart, check, close, done

rng = np.random.default_rng(46)


def tail(n, c):  # Exp(1) の標本平均 X̄ₙ ~ Gamma(n, 1/n) で P(|X̄ₙ−1|>c)
    g = stats.gamma(n, scale=1 / n)
    return g.cdf(1 - c) + g.sf(1 + c)


T = "stochastic-order-notation"
p = chart(T)
ns = p["x"]
check(T, "図・表：3 つの物差しの確率（ガンマ分布で正確に）",
      allclose(p["series"][0]["y"], [tail(n, 0.1) for n in ns], 6e-5) and allclose(p["series"][1]["y"], [tail(n, 2 / np.sqrt(n)) for n in ns], 6e-5)
      and allclose(p["series"][2]["y"], [tail(n, 2 / n) for n in ns], 6e-5))
check(T, "極限 2P(Z>2)=0.0455", close(2 * stats.norm.sf(2), 0.0455, 5e-5))
x = rng.exponential(1, (20000, 400))
check(T, "シミュレーション：n=400 で P(|X̄−1|>0.1) が正確な値に近い", close((np.abs(x.mean(1) - 1) > 0.1).mean(), tail(400, 0.1), 0.006), (np.abs(x.mean(1) - 1) > 0.1).mean())
xs = rng.normal(0, 1, (20000, 400))
v = xs.var(1)
check(T, "例2 √n(σ̂²−σ²) の分散 → μ₄−σ⁴=2（正規分布）", close((np.sqrt(400) * (v - 1)).var(), 2, 0.08), (np.sqrt(400) * (v - 1)).var())

T = "multivariate-clt"
p = chart(T)
p1, n = 0.4, 1000
check(T, "図：正しい標準誤差と独立とみなした値",
      allclose(p["series"][0]["y"], [np.sqrt((p1 + q - (p1 - q) ** 2) / n) for q in p["x"]], 6e-6)
      and allclose(p["series"][1]["y"], [np.sqrt((p1 * (1 - p1) + q * (1 - q)) / n) for q in p["x"]], 6e-6))
p2 = 0.35
se, se0 = np.sqrt((p1 + p2 - (p1 - p2) ** 2) / n), np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
check(T, "例1 共分散 −0.00014・相関 −0.60・SE 0.0273・0.0216・z 1.83 と 2.31・区間 (−0.004, 0.104)",
      close(-p1 * p2 / n, -0.00014, 1e-9) and close(-p1 * p2 / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)), -0.60, 0.005) and close(se, 0.0273, 5e-5)
      and close(se0, 0.0216, 5e-5) and close(0.05 / se, 1.83, 0.005) and close(0.05 / se0, 2.31, 0.005) and close(0.05 - 1.96 * se, -0.004, 5e-4)
      and close(0.05 + 1.96 * se, 0.104, 5e-4))
m = rng.multinomial(n, [0.4, 0.35, 0.25], size=100000) / n
check(T, "例1 シミュレーション：差の標準偏差≈0.0273", close((m[:, 0] - m[:, 1]).std(), se, 2e-4), (m[:, 0] - m[:, 1]).std())
check(T, "例1 分散の式 p₁(1−p₁)+p₂(1−p₂)+2p₁p₂ = p₁+p₂−(p₁−p₂)²", close(p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2, p1 + p2 - (p1 - p2) ** 2, 1e-12))

T = "glivenko-cantelli-theorem"
p = chart(T)
ns = p["x"]
check(T, "図：Dₙ の期待値と 95% 点（kstwo）・DKW の上界",
      allclose(p["series"][0]["y"], [stats.kstwo(n).mean() for n in ns], 6e-5) and allclose(p["series"][1]["y"], [stats.kstwo(n).ppf(0.95) for n in ns], 6e-5)
      and allclose(p["series"][2]["y"], [np.sqrt(np.log(40) / (2 * n)) for n in ns], 6e-5))
check(T, "例1 DKW 幅 0.136・正確な 95% 点 0.134・コルモゴロフ分布の 95% 点 1.358",
      close(np.sqrt(np.log(40) / 200), 0.136, 5e-4) and close(stats.kstwo(100).ppf(0.95), 0.134, 5e-4) and close(stats.kstwobign.ppf(0.95), 1.358, 5e-4))
check(T, "例2 期待値 0.085・0.027・0.012、n ≥ 4612", close(stats.kstwo(100).mean(), 0.085, 5e-4) and close(stats.kstwo(1000).mean(), 0.027, 5e-4)
      and close(stats.kstwo(5000).mean(), 0.012, 5e-4) and np.sqrt(np.log(40) / (2 * 4612)) <= 0.02 < np.sqrt(np.log(40) / (2 * 4611)))
D = []
for _ in range(3000):
    u = np.sort(rng.exponential(2, 100))
    F = 1 - np.exp(-u / 2)
    i = np.arange(1, 101)
    D.append(max((i / 100 - F).max(), (F - (i - 1) / 100).max()))
check(T, "連続分布なら Dₙ の分布は F によらない（指数分布で 95% 点≈0.134）", close(np.quantile(D, 0.95), 0.134, 0.006), np.quantile(D, 0.95))
check(T, "DKW の被覆確率 ≥ 95%（n=100）", (np.array(D) <= np.sqrt(np.log(40) / 200)).mean() >= 0.94, (np.array(D) <= 0.1358).mean())

T = "lindeberg-feller-clt"
n = 1000
pr = 1 / np.arange(1, n + 1)
dist = np.array([1.0])
for q in pr:
    d = np.zeros(len(dist) + 1)
    d[:-1] += dist * (1 - q)
    d[1:] += dist * q
    dist = d
mu, var = pr.sum(), (pr * (1 - pr)).sum()
s = np.sqrt(var)
p = chart(T)
check(T, "図：正確な確率と正規近似の密度", allclose(p["series"][0]["y"], [dist[k] for k in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], [stats.norm.pdf(k, mu, s) for k in p["x"]], 6e-5))
check(T, "例1 平均 7.485・分散 5.842・SD 2.417", close(mu, 7.485) and close(var, 5.842) and close(s, 2.417))
check(T, "例1 P(R≥10) 0.196 / 0.202、P(R≤5) 0.211 / 0.206",
      close(dist[10:].sum(), 0.196) and close(stats.norm.sf((9.5 - mu) / s), 0.202) and close(dist[:6].sum(), 0.211) and close(stats.norm.cdf((5.5 - mu) / s), 0.206))
check(T, "例1 リアプノフ比≈0.37", close((pr * (1 - pr) * ((1 - pr) ** 2 + pr ** 2)).sum() / var ** 1.5, 0.37, 0.005))
y = rng.random((4000, n))
rec = (y == np.maximum.accumulate(y, axis=1)).sum(1)
check(T, "例1 シミュレーション：記録の回数の平均≈7.49", close(rec.mean(), mu, 0.12), rec.mean())
# 例2：x_i = 2^i ではネーター比が 0 に向かわない、等間隔なら 0 に向かう
xg = 2.0 ** np.arange(1, 31)
xe = np.arange(1, 31.0)
ratio = lambda z: ((z - z.mean()) ** 2).max() / ((z - z.mean()) ** 2).sum()
check(T, "例2 ネーター比：x=2^i で 0 に向かわない・等間隔で小さい", ratio(xg) > 0.7 and ratio(xe) < 0.11, (ratio(xg), ratio(xe)))
check(T, "本文の値", all(k in body(T) for k in ("7.485", "5.842", "2.417", "0.196", "0.202", "0.211", "0.206")))
T = "bootstrap-theory"
from boot_sim import coverage, max_fail  # noqa: E402
p = chart(T)
cv = [coverage(n=n) for n in p["x"]]
check(T, "図：3 つの方法の被覆率（同じ乱数で計算し直す）", all(allclose(p["series"][i]["y"], [c[i] for c in cv], 6e-5) for i in range(3)))
tab = {10: (0.912, 0.874, 0.944), 20: (0.918, 0.904, 0.946), 80: (0.941, 0.932, 0.940)}
check(T, "例1 表の値", all(allclose(cv[p["x"].index(n)], v, 6e-4) for n, v in tab.items()))
fr, th = max_fail()
check(T, "例2 最大値の一致確率 1−(1−1/n)^n≈0.636・極限 0.632・シミュレーション 0.632", close(th, 0.636, 5e-4) and close(1 - np.exp(-1), 0.632, 5e-4) and close(fr, 0.632, 5e-4), (fr, th))
xs = np.random.default_rng(462).exponential(1, (200000, 20))
check(T, "指数分布では標本平均と標本 SD に正の相関（区間が外れやすい理由）", np.corrcoef(xs.mean(1), xs.std(1, ddof=1))[0, 1] > 0.5)
u = np.random.default_rng(463).random((200000, 50))
check(T, "n(θ−X(n))/θ はほぼ Exp(1)（平均 1）", close((50 * (1 - u.max(1))).mean(), 50 / 51, 0.01))

T = "local-alternatives"
z = stats.norm.isf(0.025)
pw = lambda d: stats.norm.cdf(-z + d) + stats.norm.cdf(-z - d)
p = chart(T)
check(T, "図：漸近検出力 β(h) と β(h√(2/π))", allclose(p["series"][0]["y"], [pw(h) for h in p["x"]], 6e-5) and allclose(p["series"][1]["y"], [pw(h * np.sqrt(2 / np.pi)) for h in p["x"]], 6e-5))
k = np.arange(401)
rej = np.abs(k - 200) / 10 > 1.96
check(T, "例1 漸近 0.516・正確 0.521・非心カイ二乗でも同じ", close(pw(2), 0.516, 5e-4) and close(stats.binom.pmf(k, 400, 0.55)[rej].sum(), 0.521, 5e-4) and close(stats.ncx2.sf(z * z, 1, 4), pw(2), 1e-10))
rng2 = np.random.default_rng(1)
xx = rng2.normal(2.5 / 10, 1, (20000, 100))
tt = xx.mean(1) / (xx.std(1, ddof=1) / 10)
pt = (np.abs(tt) > stats.t.isf(0.025, 99)).mean()
sg = (np.abs((xx > 0).sum(1) - 50) / 5 > 1.96).mean()
check(T, "例2 h=2.5：漸近 0.705・0.514、n=100 のシミュレーション 0.699・0.539、効率 2/π=0.637・π/2=1.57",
      close(pw(2.5), 0.705, 5e-4) and close(pw(2.5 * np.sqrt(2 / np.pi)), 0.514, 5e-4) and close(pt, 0.699, 5e-4) and close(sg, 0.539, 5e-4)
      and close(2 / np.pi, 0.637, 5e-4) and close(np.pi / 2, 1.57, 0.005), (pt, sg))
check(T, "符号検定の実際の有意水準は 5% より大きい（|S−50| ≥ 10）", stats.binom.pmf(k[:101], 100, 0.5)[np.abs(k[:101] - 50) / 5 > 1.96].sum() > 0.05)
done("第46章（漸近理論の発展）")
