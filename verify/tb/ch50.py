"""教科書 第50章（線形混合モデルと相関のあるデータ）の検算。"""
import numpy as np

from gee_sim import marginal_truth, run
from tbcheck import allclose, body, chart, check, close, done

T = "linear-mixed-model"
p = chart(T)
check(T, "図：縮小係数 nρ/(nρ+1−ρ)", all(allclose(s["y"], [n * r / (n * r + 1 - r) for n in p["x"]], 6e-5) for s, r in zip(p["series"], (0.05, 0.1, 0.3))))
su2, s2 = 9.0, 100.0
n = np.array([5, 10, 20, 40, 5, 30])
yb = np.array([62.0, 55.0, 58.0, 51.0, 45.0, 57.0])
w = 1 / (su2 + s2 / n)
mu = (w * yb).sum() / w.sum()
lam = n * su2 / (n * su2 + s2)
pred = mu + lam * (yb - mu)
check(T, "例1 ρ≈0.083・GLS 平均 54.79・λ 0.310 と 0.783・予測 57.03・51.75・51.82",
      close(su2 / (su2 + s2), 0.083, 5e-4) and close(mu, 54.79, 0.005) and close(lam[0], 0.310, 5e-4) and close(lam[3], 0.783, 5e-4)
      and close(pred[0], 57.03, 0.005) and close(pred[4], 51.75, 0.005) and close(pred[3], 51.82, 0.005), np.round(pred, 3))
# BLUP の式 GZ'V^{-1}(y−Xβ) を行列で計算し、縮小の式と一致することを確かめる（学級 1 のデータを作る）
rng = np.random.default_rng(50)
y1 = rng.normal(60, 10, 5)
V = su2 * np.ones((5, 5)) + s2 * np.eye(5)
blup = su2 * np.ones(5) @ np.linalg.solve(V, y1 - mu)
check(T, "BLUP の行列の式 = λ(ȳ−μ)", close(blup, 5 * su2 / (5 * su2 + s2) * (y1.mean() - mu), 1e-10))
rho = su2 / (su2 + s2)
check(T, "例2 デザイン効果 1+29ρ≈3.39・√≈1.84", close(1 + 29 * rho, 3.39, 0.005) and close(np.sqrt(1 + 29 * rho), 1.84, 0.005))
u = rng.normal(0, 3, (20000, 1))
ys = u + rng.normal(0, 10, (20000, 30))
check(T, "例2 シミュレーション：学級平均の分散 / (σ_total²/30) ≈ 3.39", close(ys.mean(1).var() / (109 / 30), 3.39, 0.1), ys.mean(1).var() / (109 / 30))

T = "reml"
p = chart(T)
check(T, "図：ML の期待値 4 − 25/(4a)、REML は 4", allclose(p["series"][1]["y"], [4 - 25 / (4 * a) for a in p["x"]], 6e-5) and all(v == 4 for v in p["series"][0]["y"]))
rng = np.random.default_rng(501)
a, m = 5, 4
uu = rng.normal(0, 2, (200000, a, 1))
y = uu + rng.normal(0, 3, (200000, a, m))
gm = y.mean((1, 2))
ym = y.mean(2)
MSA = m * ((ym - gm[:, None]) ** 2).sum(1) / (a - 1)
MSE = ((y - ym[:, :, None]) ** 2).sum((1, 2)) / (a * (m - 1))
check(T, "例2 シミュレーション a=5：REML 平均 4.01・ML 平均 2.76", close(((MSA - MSE) / m).mean(), 4.01, 0.02) and close((((a - 1) / a * MSA - MSE) / m).mean(), 2.76, 0.02))
d = np.array([[12, 15, 11, 14], [18, 16, 19, 21], [10, 13, 9, 12], [15, 14, 17, 16], [20, 22, 19, 23]], float)
dm, dg = d.mean(1), d.mean()
msa = 4 * ((dm - dg) ** 2).sum() / 4
mse = ((d - dm[:, None]) ** 2).sum() / 15
check(T, "例1 平均 13.0・18.5・11.0・15.5・21.0、MS_A 65.3・MS_E 3.2・REML 15.53・ML 12.26",
      allclose(dm, [13.0, 18.5, 11.0, 15.5, 21.0], 1e-9) and close(msa, 65.3, 1e-9) and close(mse, 3.2, 1e-9)
      and close((msa - mse) / 4, 15.53, 0.005) and close((0.8 * msa - mse) / 4, 12.26, 0.005))
# REML の尤度を数値で最大化して、分散分析法と一致することを確かめる
X = np.ones((20, 1))
yv = d.ravel()
Z = np.kron(np.eye(5), np.ones((4, 1)))


def neg_reml(th, ml=False):
    su, se = np.exp(th)
    Vv = su * Z @ Z.T + se * np.eye(20)
    Vi = np.linalg.inv(Vv)
    b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ yv)
    r = yv - X @ b
    val = np.linalg.slogdet(Vv)[1] + r @ Vi @ r
    return val + (0 if ml else np.linalg.slogdet(X.T @ Vi @ X)[1])


from scipy.optimize import minimize
er = np.exp(minimize(neg_reml, [1.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000}).x)
em = np.exp(minimize(lambda t: neg_reml(t, True), [1.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000}).x)
check(T, "REML・ML の尤度を数値で最大化しても 15.53・12.26（誤差分散 3.2）", close(er[0], 15.525, 1e-3) and close(er[1], 3.2, 1e-3) and close(em[0], 12.26, 1e-3) and close(em[1], 3.2, 1e-3), (er, em))
check(T, "ML は約 2 割小さい・a=5 で約 31% 過小・a=30 で 3.79", close(1 - 12.26 / 15.525, 0.21, 0.01) and close(1.25 / 4, 0.31, 0.005) and close(4 - 25 / 120, 3.79, 0.005))

T = "generalized-estimating-equations"
p = chart(T)
cov_n, cov_r, rows = [], [], {}
for sd in p["x"]:
    th = marginal_truth(sd)
    res = np.array(list(run(sd, R=1000)))
    rows[sd] = (th, res[:, 0].std(), res[:, 1].mean(), res[:, 2].mean())
    cov_n.append((np.abs(res[:, 0] - th) <= 1.96 * res[:, 1]).mean())
    cov_r.append((np.abs(res[:, 0] - th) <= 1.96 * res[:, 2]).mean())
check(T, "図：被覆率（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], cov_n, 1e-9) and allclose(p["series"][1]["y"], cov_r, 1e-9), (cov_n, cov_r))
r1 = rows[1.0]
check(T, "例 SD 1.0：母集団平均の効果 0.662・実際の SD 0.286・ふつう 0.182・GEE 0.285", close(r1[0], 0.662, 5e-4) and close(r1[1], 0.286, 5e-4)
      and close(r1[2], 0.182, 5e-4) and close(r1[3], 0.285, 5e-4), r1)
check(T, "母集団平均の効果は SD が大きいほど小さい（0.8 → 0.567）", all(np.diff([rows[s][0] for s in p["x"]]) < 0) and close(rows[0][0], 0.8, 1e-3))
check(T, "本文の値", all(s in body(T) for s in ("0.662", "0.286", "0.182", "0.777", "0.285", "0.949")))
done("第50章（線形混合モデルと相関のあるデータ）")
