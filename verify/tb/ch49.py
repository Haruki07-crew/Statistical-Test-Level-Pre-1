"""教科書 第49章（ノンパラメトリック推定と頑健統計）の検算。"""
import numpy as np
from scipy import integrate, stats

from tbcheck import allclose, body, chart, check, close, done

T = "kernel-density-estimation"
f = lambda x: 0.5 * stats.norm.pdf(x, -1.5, 0.5) + 0.5 * stats.norm.pdf(x, 1.5, 0.5)
n = 200
rng = np.random.default_rng(49)
comp = rng.random(n) < 0.5
x = np.where(comp, rng.normal(-1.5, 0.5, n), rng.normal(1.5, 0.5, n))
kde = lambda g, h: np.mean(stats.norm.pdf((g - x) / h)) / h
p = chart(T)
check(T, "図：真の密度と h=0.1・0.25・1.0 の推定（同じ乱数のデータから計算し直す）",
      allclose(p["series"][0]["y"], [f(g) for g in p["x"]], 6e-5)
      and all(allclose(p["series"][i]["y"], [kde(g, h) for g in p["x"]], 6e-5) for i, h in ((1, 0.1), (2, 0.25), (3, 1.0))))


def f2(z):
    g = lambda m, s: (((z - m) / s) ** 2 - 1) / s ** 2 * stats.norm.pdf(z, m, s)
    return 0.5 * g(-1.5, 0.5) + 0.5 * g(1.5, 0.5)


Rf2 = integrate.quad(lambda z: f2(z) ** 2, -8, 8)[0]
hopt = (1 / (2 * np.sqrt(np.pi)) / (Rf2 * n)) ** 0.2
check(T, "例1 R(f'')≈3.42・h*≈0.21", close(Rf2, 3.42, 0.005) and close(hopt, 0.21, 0.005), (Rf2, hopt))
ise = {h: integrate.quad(lambda g: (kde(g, h) - f(g)) ** 2, -6, 6, limit=200)[0] for h in (0.1, 0.25, 1.0)}
check(T, "例1 積分 2 乗誤差 0.0115・0.0049・0.087", close(ise[0.1], 0.0115, 5e-5) and close(ise[0.25], 0.0049, 5e-5) and close(ise[1.0], 0.087, 5e-4), ise)
hs = 0.9 * min(x.std(ddof=1), np.subtract(*np.percentile(x, [75, 25])) / 1.34) * n ** -0.2
ise_s = integrate.quad(lambda g: (kde(g, hs) - f(g)) ** 2, -6, 6, limit=200)[0]
check(T, "例2 シルバーマン h≈0.49・谷 0.009→0.055・山 0.399→0.310・誤差 0.021",
      close(hs, 0.49, 0.005) and close(f(0), 0.009, 5e-4) and close(kde(0, hs), 0.055, 5e-4) and close(f(1.5), 0.399, 5e-4) and close(kde(1.5, hs), 0.310, 5e-4)
      and close(ise_s, 0.021, 5e-4), (hs, kde(0, hs), kde(1.5, hs), ise_s))
check(T, "例2 正規分布では h = (4/3)^{1/5} n^{-1/5} σ ≈ 1.06·100^{-1/5} ≈ 0.42", close((4 / 3) ** 0.2, 1.06, 0.001) and close((4 / 300) ** 0.2, 0.42, 0.005))
# MISE の n^{-4/5}：正規分布・最適 h で n を 10 倍すると MISE は 10^{-4/5} 倍
amise = lambda nn: 1.25 * (1 / (2 * np.sqrt(np.pi))) ** 0.8 * (3 / (8 * np.sqrt(np.pi))) ** 0.2 * nn ** -0.8
check(T, "MISE ∝ n^{-4/5}", close(amise(1000) / amise(100), 10 ** -0.8, 1e-12))

T = "nadaraya-watson-estimator"
m = lambda z: np.sin(2 * np.pi * z)
h, n = 0.08, 200
rng = np.random.default_rng(491)
xx = rng.random(n)
yy = m(xx) + rng.normal(0, 0.3, n)


def nw(g):
    w = stats.norm.pdf((g - xx) / h)
    return (w * yy).sum() / w.sum()


def ll(g):
    w = stats.norm.pdf((g - xx) / h)
    X = np.c_[np.ones(n), xx - g]
    W = X.T * w
    return np.linalg.solve(W @ X, W @ yy)[0]


p = chart(T)
check(T, "図：真の関数・NW・局所線形（同じ乱数のデータから計算し直す）", allclose(p["series"][0]["y"], [m(g) for g in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], [nw(g) for g in p["x"]], 6e-5) and allclose(p["series"][2]["y"], [ll(g) for g in p["x"]], 6e-5))
K = lambda t: stats.norm.pdf(t / h) / h
s0, s1, s2 = (integrate.quad(lambda t: K(t) * t ** j, 0, 1)[0] for j in (0, 1, 2))
t0, t1 = (integrate.quad(lambda t: K(t) * t ** j * m(t), 0, 1)[0] for j in (0, 1))
check(T, "例1 x=0：NW 0.229・期待値 0.369・近似 0.40、局所線形 −0.065・期待値 0.043",
      close(nw(0), 0.229, 5e-4) and close(t0 / s0, 0.369, 5e-4) and close(h * 2 * np.pi * np.sqrt(2 / np.pi), 0.40, 0.005)
      and close(ll(0), -0.065, 5e-4) and close((s2 * t0 - s1 * t1) / (s0 * s2 - s1 ** 2), 0.043, 5e-4))
e25 = integrate.quad(lambda t: K(t - 0.25) * m(t), 0, 1)[0] / integrate.quad(lambda t: K(t - 0.25), 0, 1)[0]
check(T, "例2 x=0.25：期待値 0.882・近似 0.874", close(e25, 0.882, 5e-4) and close(1 - 0.5 * h * h * (2 * np.pi) ** 2, 0.874, 5e-4))
# 境界の偏りは NW で O(h)、局所線形で O(h²)：h を半分にすると偏りは約 1/2 と約 1/4
def bias0(hh):
    Kh = lambda t: stats.norm.pdf(t / hh) / hh
    a0, a1, a2 = (integrate.quad(lambda t: Kh(t) * t ** j, 0, 1)[0] for j in (0, 1, 2))
    b0, b1 = (integrate.quad(lambda t: Kh(t) * t ** j * m(t), 0, 1)[0] for j in (0, 1))
    return b0 / a0, (a2 * b0 - a1 * b1) / (a0 * a2 - a1 ** 2)


r1, r2 = bias0(0.04), bias0(0.02)
check(T, "境界の偏り：NW は O(h)（h を半分で約 1/2）、局所線形は O(h²) 以下（約 1/4 以下。sin では m''(0)=0 なので h³ で約 1/8）", close(r1[0] / r2[0], 2, 0.1) and r1[1] / r2[1] >= 3.8, (r1[0] / r2[0], r1[1] / r2[1]))

T = "influence-function"
k = 1.345
P = stats.norm.cdf(k) - stats.norm.cdf(-k)
p = chart(T)
check(T, "図：平均 x・中央値 ±1/(2φ(0))・フーバー ψ/P", allclose(p["series"][0]["y"], p["x"], 1e-12)
      and allclose(p["series"][1]["y"], [np.sign(v) / (2 * stats.norm.pdf(0)) for v in p["x"]], 6e-5)
      and allclose(p["series"][2]["y"], [max(-k, min(k, v)) / P for v in p["x"]], 6e-5))
# 影響関数を数値微分で確かめる：中央値（F=N(0,1)）に ε の点 x=2 を混ぜる
eps = 1e-5
med_mix = lambda xx_: __import__("scipy.optimize", fromlist=["brentq"]).brentq(lambda t: (1 - eps) * stats.norm.cdf(t) + eps * (t >= xx_) - 0.5, -1, 1)
check(T, "中央値の影響関数を数値微分で（x=2 で 1.2533）", close(med_mix(2.0) / eps, 1.2533, 2e-3), med_mix(2.0) / eps)


def huber(s, it=200):
    s = np.asarray(s, float)
    th = np.median(s)
    sc = 1.4826 * np.median(np.abs(s - th))
    for _ in range(it):
        r = s - th
        w = np.where(np.abs(r) <= k * sc, 1.0, k * sc / np.abs(r))
        th = (w * s).sum() / w.sum()
    return th, sc


a = [21, 23, 24, 25, 25, 26, 27, 28, 30, 31]
b = a[:-1] + [310]
check(T, "例1 平均 26.0→53.9・中央値 25.5・フーバー 約 26.0（変わらない）・k×MAD≈4.0",
      close(np.mean(a), 26.0, 1e-9) and close(np.mean(b), 53.9, 1e-9) and np.median(a) == np.median(b) == 25.5
      and close(huber(a)[0], 26.0, 0.005) and close(huber(b)[0], huber(a)[0], 1e-9) and close(k * huber(a)[1], 4.0, 0.05), (huber(a), huber(b)))
z = rng.normal(size=(20000, 400))
check(T, "例2 E[IF²]：中央値 π/2・フーバー 1.053（シミュレーション n=400）", close(400 * np.median(z, 1).var(), np.pi / 2, 0.06)
      and close(P - 2 * k * stats.norm.pdf(k) + 2 * k * k * stats.norm.sf(k), 1.053 * P * P, 1e-3), 400 * np.median(z, 1).var())

T = "huber-estimator"
p = chart(T)
eff = lambda kk: (stats.norm.cdf(kk) - stats.norm.cdf(-kk)) ** 2 / ((stats.norm.cdf(kk) - stats.norm.cdf(-kk)) - 2 * kk * stats.norm.pdf(kk) + 2 * kk * kk * stats.norm.sf(kk))
check(T, "図：効率と粗大誤差感度", allclose(p["series"][0]["y"], [eff(v) for v in p["x"]], 6e-5)
      and allclose(p["series"][1]["y"], [v / (stats.norm.cdf(v) - stats.norm.cdf(-v)) for v in p["x"]], 6e-5))
check(T, "例2 効率 90%・95%・99%・79%、上限 1.64・2.10", close(eff(1), 0.90, 0.005) and close(eff(1.345), 0.95, 5e-4) and close(eff(2), 0.99, 0.005)
      and close(eff(0.5), 0.79, 0.005) and close(1.345 / P, 1.64, 0.005) and close(2 / (stats.norm.cdf(2) - stats.norm.cdf(-2)), 2.10, 0.005))
rng = np.random.default_rng(493)
xr = np.arange(1, 21.0)
yr = 2 + 0.5 * xr + rng.normal(0, 1, 20)
yr[17] -= 12
yr[19] -= 12
X = np.c_[np.ones(20), xr]


def hreg(y_, it=200):
    bb = np.linalg.lstsq(X, y_, rcond=None)[0]
    for _ in range(it):
        r = y_ - X @ bb
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = np.where(np.abs(r) <= k * s, 1, k * s / np.abs(r))
        bb = np.linalg.solve((X.T * w) @ X, (X.T * w) @ y_)
    return bb, w


bo = np.linalg.lstsq(X, yr, rcond=None)[0]
bh, wh = hreg(yr)
yc = yr.copy()
yc[[17, 19]] += 12
check(T, "例1 最小二乗 0.231・外れ値なし 0.537・フーバー 0.436・重み 0.11 と 0.13",
      close(bo[1], 0.231, 5e-4) and close(np.linalg.lstsq(X, yc, rcond=None)[0][1], 0.537, 5e-4) and close(bh[1], 0.436, 5e-4)
      and close(wh[17], 0.11, 0.005) and close(wh[19], 0.13, 0.005), (bo[1], bh[1], wh[17], wh[19]))
check(T, "本文の値", all(s in body(T) for s in ("0.231", "0.537", "0.436")))
done("第49章（ノンパラメトリック推定と頑健統計）")
