"""教科書 第51章（時系列と状態空間モデル）の検算。"""
import numpy as np
from scipy import stats

from kalman_sim import filt, simulate, smooth
from tbcheck import allclose, body, chart, check, close, done

T = "kalman-filter"
mu, y = simulate()
ap, Pp, af, Pf, ll = filt(y)
s = smooth(af, Pf, Pp)
p = chart(T)
check(T, "図：観測・真の水準・フィルタ・平滑化（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], y, 6e-4) and allclose(p["series"][1]["y"], mu, 6e-4)
      and allclose(p["series"][2]["y"], af, 6e-4) and allclose(p["series"][3]["y"], s, 6e-4))
rm = [np.sqrt(((v - mu) ** 2).mean()) for v in (y, af, s)]
check(T, "例2 RMSE 1.23・0.61・0.45", allclose(rm, [1.23, 0.61, 0.45], 0.005), rm)
ap3, Pp3, af3, Pf3, _ = filt([10.2, 9.8, 11.5])
K3 = Pp3 / (Pp3 + 1)
check(T, "例1 表：予測・分散・ゲイン・更新", allclose(ap3, [10.0, 10.111, 9.972], 5e-4) and allclose(Pp3, [1.25, 0.806, 0.696], 5e-4)
      and allclose(K3, [0.556, 0.446, 0.410], 5e-4) and allclose(af3, [10.111, 9.972, 10.599], 5e-4) and allclose(Pf3, [0.556, 0.446, 0.410], 5e-4))
q = 0.25
Kst = (-q + np.sqrt(q * q + 4 * q)) / 2
_, Ppl, _, _, _ = filt(np.zeros(200))
check(T, "例1 定常ゲイン 0.390 にゲインが収束", close(Kst, 0.390, 5e-4) and close(Ppl[-1] / (Ppl[-1] + 1), Kst, 1e-9))
# フィルタが条件付き期待値であることを、同時正規分布の条件付き平均で直接確かめる（t=3）
n = 3
cov_mu = np.array([[1 + 0.25 * min(i, j) for j in range(1, n + 1)] for i in range(1, n + 1)])  # μ_t の共分散（μ₀~N(10,1)）
Sy = cov_mu + np.eye(n)
cond = 10 + cov_mu[-1] @ np.linalg.solve(Sy, np.array([10.2, 9.8, 11.5]) - 10)
check(T, "フィルタ a₃ = E[μ₃ | y₁..y₃]（多変量正規の条件付き平均）", close(cond, af3[-1], 1e-9), cond)
llb = stats.multivariate_normal(np.full(40, 10.0), np.array([[1 + 0.25 * min(i, j) for j in range(1, 41)] for i in range(1, 41)]) + np.eye(40)).logpdf(y)
check(T, "予測誤差分解の対数尤度 = 同時正規分布の対数尤度", close(ll, llb, 1e-8), (ll, llb))

T = "garch-model"
w, a, b = 0.02, 0.08, 0.9
ss = a + b
r1 = a * (1 - a * b - b * b) / (1 - 2 * a * b - b * b)
p = chart(T)
check(T, "図：r² の自己相関 ρ₁(α+β)^{k−1}", allclose(p["series"][0]["y"], [r1 * ss ** (k - 1) for k in p["x"]], 6e-5) and all(v == 0 for v in p["series"][1]["y"]))
check(T, "例1 無条件分散 1.0・半減期 34.3 日・ρ₁ 0.205・尖度 4.43", close(w / (1 - ss), 1.0, 1e-9) and close(np.log(.5) / np.log(ss), 34.3, 0.05)
      and close(r1, 0.205, 5e-4) and close(3 * (1 - ss ** 2) / (1 - ss ** 2 - 2 * a * a), 4.43, 0.005))
f = lambda h: 1 + ss ** (h - 1) * 3
check(T, "例2 予測 3.50・2.67・2.11、10 日の分散 37.4・SD 6.12%・√10≈3.16",
      close(f(10), 3.50, 0.005) and close(f(30), 2.67, 0.005) and close(f(50), 2.11, 0.005) and close(sum(f(h) for h in range(1, 11)), 37.4, 0.05)
      and close(np.sqrt(sum(f(h) for h in range(1, 11))), 6.12, 0.005) and close(np.sqrt(10), 3.16, 0.005))
rng = np.random.default_rng(510)
N = 2_000_000
z = rng.normal(size=N)
r = np.empty(N)
s2 = 1.0
for t in range(N):
    r[t] = np.sqrt(s2) * z[t]
    s2 = w + a * r[t] ** 2 + b * s2
r = r[1000:]
acf = lambda v, k: np.corrcoef(v[:-k], v[k:])[0, 1]
check(T, "シミュレーション：r の自己相関≈0・r² の 1 次自己相関≈0.2・分散≈1", abs(acf(r, 1)) < 0.005 and close(acf(r ** 2, 1), r1, 0.03) and close(r.var(), 1, 0.03),
      (acf(r, 1), acf(r ** 2, 1), r.var()))

T = "var-model"
A = np.array([[0.5, 0.2], [0.1, 0.6]])
c = np.array([1.0, 0.5])
p = chart(T)
irf = [np.linalg.matrix_power(A, h)[:, 0] for h in p["x"]]
check(T, "図：インパルス応答 Aʰe₁", allclose(p["series"][0]["y"], [v[0] for v in irf], 6e-5) and allclose(p["series"][1]["y"], [v[1] for v in irf], 6e-5))
check(T, "例1 固有値 0.7・0.4、平均 (2.778, 1.944)", allclose(sorted(np.linalg.eigvals(A).real), [0.4, 0.7], 1e-12) and allclose(np.linalg.solve(np.eye(2) - A, c), [2.778, 1.944], 5e-4))
yy = np.array([4.0, 2.0])
fc = []
for _ in range(3):
    yy = c + A @ yy
    fc.append(yy.copy())
check(T, "例1 予測 (3.4, 2.1)・(3.12, 2.1)・(2.98, 2.072)", allclose(np.ravel(fc), [3.4, 2.1, 3.12, 2.1, 2.98, 2.072], 5e-4))
rng = np.random.default_rng(51)
Tn = 200
Y = np.zeros((Tn + 1, 2))
Y[0] = np.linalg.solve(np.eye(2) - A, c)
for t in range(1, Tn + 1):
    Y[t] = c + A @ Y[t - 1] + rng.normal(0, 1, 2)
X = np.c_[np.ones(Tn), Y[:-1]]
B = np.linalg.lstsq(X, Y[1:], rcond=None)[0]
check(T, "例2 推定値 0.473・0.260・0.095・0.587", allclose(B.T[:, 1:].ravel(), [0.473, 0.260, 0.095, 0.587], 5e-4), np.round(B.T, 3))
y1 = Y[1:, 0]
Xr = np.c_[np.ones(Tn), Y[:-1, 0]]
ru = ((y1 - X @ np.linalg.lstsq(X, y1, rcond=None)[0]) ** 2).sum()
rr = ((y1 - Xr @ np.linalg.lstsq(Xr, y1, rcond=None)[0]) ** 2).sum()
F = (rr - ru) / (ru / (Tn - 3))
check(T, "例2 F≈22.9（自由度 1, 197）・p<0.0001", close(F, 22.9, 0.05) and stats.f.sf(F, 1, Tn - 3) < 1e-4, F)

T = "cointegration"
rng = np.random.default_rng(511)
rej, r2 = 0, []
for _ in range(5000):
    x = np.cumsum(rng.normal(size=200))
    yv = np.cumsum(rng.normal(size=200))
    Xc = np.c_[np.ones(200), x]
    bb = np.linalg.lstsq(Xc, yv, rcond=None)[0]
    e = yv - Xc @ bb
    se = np.sqrt((e @ e) / 198 / ((x - x.mean()) ** 2).sum())
    rej += abs(bb[1] / se) > 1.96
    r2.append(1 - (e @ e) / ((yv - yv.mean()) ** 2).sum())
check(T, "例1 見せかけの回帰：有意になる割合 約 84%・R² の中央値 約 0.18", close(rej / 5000, 0.84, 0.005) and close(np.median(r2), 0.18, 0.005), (rej / 5000, np.median(r2)))
rng = np.random.default_rng(512)
x = np.cumsum(rng.normal(size=200))
u = np.zeros(200)
for t in range(1, 200):
    u[t] = 0.5 * u[t - 1] + rng.normal()
yv = 1 + 2 * x + u
Xc = np.c_[np.ones(200), x]
bb = np.linalg.lstsq(Xc, yv, rcond=None)[0]
e = yv - Xc @ bb
de = np.diff(e)
rho = (e[:-1] @ de) / (e[:-1] @ e[:-1])
se = np.sqrt(((de - rho * e[:-1]) ** 2).sum() / 198 / (e[:-1] @ e[:-1]))
check(T, "例2 傾き 2.010・残差の DF 統計量 −8.47", close(bb[1], 2.010, 5e-4) and close(rho / se, -8.47, 0.005), (bb[1], rho / se))
check(T, "本文の値", all(v in body(T) for v in ("84%", "0.18", "2.010", "-8.47", "-3.34")))
done("第51章（時系列と状態空間モデル）")
