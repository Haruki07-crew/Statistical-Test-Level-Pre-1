"""教科書 第45章（測度論的確率論）の検算。"""
import numpy as np
from scipy import integrate, stats

from tbcheck import allclose, body, chart, check, close, done

T = "probability-space"
p = chart(T)
check(T, "図：偶数回目 (1−p)/(2−p)・奇数回目 1/(2−p)、和が 1",
      allclose(p["series"][0]["y"], [round((1 - x) / (2 - x), 4) for x in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], [round(1 / (2 - x), 4) for x in p["x"]], 6e-5)
      and all(close(a + b, 1, 2e-4) for a, b in zip(p["series"][0]["y"], p["series"][1]["y"])))
check(T, "例2 p=0.5 で 1/3、部分和 0.25・0.3125・0.3281・0.3320",
      close(sum(0.5 * 0.5 ** (2 * j - 1) for j in range(1, 200)), 1 / 3, 1e-12)
      and allclose([sum(0.25 ** j for j in range(1, m + 1)) for m in range(1, 5)], [0.25, 0.3125, 0.3281, 0.3320], 6e-5))
rng = np.random.default_rng(45)
first = rng.geometric(0.5, 200000)
check(T, "例2 シミュレーション：初めての表が偶数回目の割合≈1/3", close((first % 2 == 0).mean(), 1 / 3, 0.005), (first % 2 == 0).mean())
check(T, "例1 偶数なら 4・奇数なら 3", close(np.mean([2, 4, 6]), 4, 0) and close(np.mean([1, 3, 5]), 3, 0))

T = "lebesgue-integral-expectation"
p = chart(T)
v = [integrate.quad(lambda x: (1 - x / n) ** n * np.exp(-x), 0, n)[0] for n in p["x"]]
check(T, "図：E[(1−X/n)ⁿ; X≤n] の値と 1/n", allclose(p["series"][2]["y"], [round(a, 4) for a in v], 6e-5)
      and allclose(p["series"][1]["y"], [round(1 / n, 4) for n in p["x"]], 6e-5), v[-1])
check(T, "例2 E[e^{−X}]=1/2・E[Z₁]=e^{−1}≈0.3679・Z₁₀≈0.4872・Z₅₀≈0.4975",
      close(integrate.quad(lambda x: np.exp(-2 * x), 0, np.inf)[0], 0.5, 1e-9) and close(v[0], 0.3679) and close(v[7], 0.4872) and close(v[-1], 0.4975))
u = rng.random(10 ** 6)
check(T, "例1 E[n·1{U≤1/n}]=1（n=50 のシミュレーション）", close((50 * (u <= 1 / 50)).mean(), 1, 0.03), (50 * (u <= 1 / 50)).mean())
check(T, "例3 裾の公式：Exp(2) で 0.5、幾何分布で 1/p", close(integrate.quad(lambda t: np.exp(-2 * t), 0, np.inf)[0], 0.5, 1e-9)
      and close(sum(0.7 ** k for k in range(2000)), 1 / 0.3, 1e-9))

T = "radon-nikodym-theorem"
x = rng.normal(size=10 ** 6)
w = np.exp(x - 0.5)
check(T, "例1 E_P[w]=1・E_P[Xw]=1（乱数 100 万個で差 0.005 未満）", close(w.mean(), 1, 0.005) and close((x * w).mean(), 1, 0.005), (w.mean(), (x * w).mean()))
check(T, "例1 w=φ(x−1)/φ(x)=exp(x−1/2)", allclose(stats.norm.pdf(x[:5] - 1) / stats.norm.pdf(x[:5]), np.exp(x[:5] - 0.5), 1e-9))
pp = stats.norm.sf(3)
m2 = np.exp(9) * stats.norm.sf(6)
check(T, "例2 p=0.00135・素朴な標準誤差 0.000367・重点サンプリング 0.0000248・分散比約 218",
      close(pp, 0.00135, 5e-6) and close(np.sqrt(pp * (1 - pp) / 1e4), 0.000367, 5e-7) and close(np.sqrt((m2 - pp ** 2) / 1e4), 0.0000248, 5e-8)
      and close(pp * (1 - pp) / (m2 - pp ** 2), 218, 1))
y = rng.normal(3, 1, 10 ** 5)
wi = np.exp(-3 * y + 4.5) * (y > 3)
check(T, "例2 重み exp(−3Y+4.5) が dP/dQ で、推定値が 0.00135 付近", allclose(stats.norm.pdf(y[:5]) / stats.norm.pdf(y[:5] - 3), np.exp(-3 * y[:5] + 4.5), 1e-9)
      and close(wi.mean(), pp, 3e-5), wi.mean())

T = "borel-cantelli-lemma"
check(T, "例1 π²/6≈1.645・Σ_{n≥10}1/n²≈0.105・Σ_{n≤10⁶}1/n≈14.4",
      close(np.pi ** 2 / 6, 1.645) and close(sum(1 / k ** 2 for k in range(10, 10 ** 6)) + 1e-6, 0.105, 6e-4)
      and close((1 / np.arange(1, 10 ** 6 + 1)).sum(), 14.39, 0.005) and close(np.log(1e6) + 0.5772, 14.39, 0.005))
n = np.arange(1, 10 ** 6 + 1)
r = np.random.default_rng(5)
c1, c2 = [], []
for _ in range(200):
    uu = r.random(10 ** 6)
    c1.append((uu < 1 / n).sum())
    c2.append((uu < 1 / n ** 2).sum())
check(T, "例1 シミュレーション（200 回）：平均 約 14.7 回と約 1.6 回", close(np.mean(c1), 14.7, 0.05) and close(np.mean(c2), 1.6, 0.05), (np.mean(c1), np.mean(c2)))
# 4 次モーメントの展開：E[(ΣX)^4] = n E[X^4] + 3n(n−1)σ⁴（X は ±1 で確かめる：E[X^4]=1, σ=1）
nn = 6
vals = np.array(np.meshgrid(*[[-1, 1]] * nn)).reshape(nn, -1).sum(0)
check(T, "例2 E[(ΣX)^4]=nE[X⁴]+3n(n−1)σ⁴（±1 の 6 個で総当たり）", close((vals.astype(float) ** 4).mean(), nn + 3 * nn * (nn - 1), 1e-9), (vals.astype(float) ** 4).mean())
check(T, "本文に π²/6≈1.645 と 14.4 回", "1.645" in body(T) and "14.4" in body(T))
p = chart(T)
Hs, Ss = [], []
for e in p["x"]:
    k = np.arange(1, int(round(10 ** e)) + 1)
    Hs.append((1 / k).sum()); Ss.append((1 / k ** 2).sum())
check(T, "図：Σ_{n≤N} 1/n と Σ 1/n²（N=10^x）", allclose(p["series"][0]["y"], Hs, 6e-4) and allclose(p["series"][1]["y"], Ss, 6e-4), (Hs[-1], Ss[-1]))
done("第45章（測度論的確率論）")
