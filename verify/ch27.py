"""第27章 検算スクリプト（時系列解析）。
AR(1)/AR(2)/MA(1)/MA(2)/ARMA(1,1) の平均・分散・自己共分散・自己相関、定常条件（特性方程式の根）と
反転可能性、ユール・ウォーカー推定、ダービン・レビンソンによる偏自己相関、スペクトル密度
（数値積分で自己共分散と整合するか）、予測と予測誤差の分散（ψ重み）を sympy/numpy/scipy で
独立に再計算し、本文と照合する。charts の line-data は各 md の front matter を yaml で読み、式から再計算する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4、第2位なら 5e-3）。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, integrate

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)
np.set_printoptions(legacy="1.25")


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    if claims.size == 1:
        claims = np.full(xs.shape, claims[0])
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch27" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid):
    return [c for c in front(term).get("charts", []) if c["id"] == cid]


def series(term, label_start, idx=0):
    c = charts(term, "line-data")[idx]["params"]
    return c["x"], [s for s in c["series"] if s["label"].startswith(label_start)][0]["y"]


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


# ---------------------------------------------------------------- 共通ツール
def psi_weights(phi, theta, m):
    """ARMA の ψ 重み（+θ の流儀）。"""
    phi = list(phi)
    theta = list(theta)
    psi = np.zeros(m)
    for j in range(m):
        v = 1.0 if j == 0 else (theta[j - 1] if j - 1 < len(theta) else 0.0)
        for i, ph in enumerate(phi, start=1):
            if j - i >= 0:
                v += ph * psi[j - i]
        psi[j] = v
    return psi


def acvf(phi, theta, H, sigma2=1.0, m=4000):
    psi = psi_weights(phi, theta, m)
    return np.array([sigma2 * np.sum(psi[: m - h] * psi[h:]) for h in range(H + 1)])


def durbin_levinson(rho, K):
    """rho[0]=1, rho[1..K]。偏自己相関 φ_kk と v_k/γ(0) を返す。"""
    pacf = []
    prev = []
    v = [1.0]
    for k in range(1, K + 1):
        num = rho[k] - sum(prev[j - 1] * rho[k - j] for j in range(1, k))
        den = 1 - sum(prev[j - 1] * rho[j] for j in range(1, k))
        pkk = num / den
        new = [prev[j - 1] - pkk * prev[k - j - 1] for j in range(1, k)] + [pkk]
        prev = new
        pacf.append(pkk)
        v.append(v[-1] * (1 - pkk ** 2))
    return np.array(pacf), np.array(v)


def pacf_direct(rho, K):
    out = []
    for k in range(1, K + 1):
        R = np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        out.append(np.linalg.solve(R, np.array(rho[1:k + 1]))[-1])
    return np.array(out)


def simulate_arma(phi, theta, n, burn=500, c=0.0, sigma=1.0):
    e = rng.normal(scale=sigma, size=n + burn)
    x = np.zeros(n + burn)
    for t in range(n + burn):
        v = c + e[t]
        for i, ph in enumerate(phi, start=1):
            if t - i >= 0:
                v += ph * x[t - i]
        for i, th in enumerate(theta, start=1):
            if t - i >= 0:
                v += th * e[t - i]
        x[t] = v
    return x[burn:]


def sample_acf(x, H):
    x = np.asarray(x, float)
    n = len(x)
    d = x - x.mean()
    g = np.array([np.sum(d[: n - h] * d[h:]) / n for h in range(H + 1)])
    return g, g / g[0]


# ================================================================ AR モデル
T = "ar-model"
ph, s2 = sp.symbols("phi sigma2", real=True)
# AR(1) 公式（sympy：ψ_j=φ^j の等比級数）
j = sp.symbols("j", integer=True, nonnegative=True)
g0_sym = s2 / (1 - ph ** 2)
check(T, "AR(1) Σφ^{2j}σ²=σ²/(1−φ²)（sympy、|φ|<1）",
      sp.simplify(sp.summation(sp.Rational(3, 5) ** (2 * j), (j, 0, sp.oo)) - sp.Rational(25, 16)) == 0, "")
ok = all(close(acvf([p], [], 0)[0], float(g0_sym.subs({ph: p, s2: 1})), 1e-9) for p in [-0.9, -0.3, 0.2, 0.6, 0.95])
check(T, "AR(1) γ(0)=σ²/(1−φ²) を ψ 重みの和と照合（5値）", ok, "")
ok = all(allclose(acvf([p], [], 6)[1:] / acvf([p], [], 6)[0], [p ** h for h in range(1, 7)], 1e-9) for p in [-0.7, 0.4, 0.9])
check(T, "AR(1) ρ(h)=φ^h（3値、h=1..6）", ok, "")
mu = sp.symbols("mu")
c_ = sp.symbols("c")
check(T, "AR(1) 平均 μ=c/(1−φ)（sympy で μ=c+φμ を解く）",
      sp.simplify(sp.solve(sp.Eq(mu, c_ + ph * mu), mu)[0] - c_ / (1 - ph)) == 0, "")
p1_, p2_ = sp.symbols("phi1 phi2")
check(T, "AR(2) 平均 μ=c/(1−φ1−φ2)（sympy）",
      sp.simplify(sp.solve(sp.Eq(mu, c_ + p1_ * mu + p2_ * mu), mu)[0] - c_ / (1 - p1_ - p2_)) == 0, "")
# 例1
check(T, "例1 μ=2/(1−0.6)=5, γ(0)=1.5625, ρ=0.6,0.36,0.216",
      close(2 / 0.4, 5, 1e-12) and close(acvf([0.6], [], 0)[0], 1.5625, 1e-9)
      and allclose([0.6, 0.6 ** 2, 0.6 ** 3], [0.6, 0.36, 0.216], 1e-12), "")
# 例2 AR(2)
g = acvf([0.5, 0.3], [], 3)
rho = g / g[0]
check(T, "例2 ρ(1)=0.7143, ρ(2)=0.6571, ρ(3)=0.5429", allclose(rho[1:4], [0.7143, 0.6571, 0.5429]), r(rho[1:4]))
check(T, "例2 ρ(1)=φ1/(1−φ2) と漸化式が ψ 重みの値と一致",
      close(rho[1], 0.5 / 0.7, 1e-9) and close(rho[2], 0.5 * rho[1] + 0.3, 1e-9) and close(rho[3], 0.5 * rho[2] + 0.3 * rho[1], 1e-9), "")
check(T, "例2 分母 0.4457、γ(0)=2.2436（表の式と ψ 重みの和）",
      close(1 - 0.5 * rho[1] - 0.3 * rho[2], 0.4457) and close(g[0], 2.2436) and close(1 / (1 - 0.5 * rho[1] - 0.3 * rho[2]), 2.2436),
      round(g[0], 4))
check(T, "例2 三角形条件 0.8<1, −0.2<1, 0.3<1", 0.5 + 0.3 < 1 and 0.3 - 0.5 < 1 and abs(0.3) < 1, "")
roots = np.sort(np.roots([-0.3, -0.5, 1]).real)
check(T, "例2 特性根 1.1736, −2.8403（√1.45 の式とも一致）",
      allclose(roots, [-2.8403, 1.1736]) and close((-0.5 + math.sqrt(1.45)) / 0.6, 1.1736), r(roots))
# 例3
rts = np.roots([0.5, -0.4, 1])
check(T, "例3 φ1²+4φ2=−1.84、根 0.4±1.3565i、|z|=√2=1.4142",
      close(0.16 - 2, -1.84, 1e-12) and allclose(sorted(rts.imag), [-1.3565, 1.3565]) and allclose(rts.real, [0.4, 0.4])
      and allclose(abs(rts), [1.4142, 1.4142]), str(np.round(rts, 4)))
w = math.acos(0.4 / (2 * math.sqrt(0.5)))
check(T, "例3 cosω=0.2828, ω=1.2840, 周期 2π/ω=4.89",
      close(0.4 / (2 * math.sqrt(0.5)), 0.2828) and close(w, 1.2840) and close(2 * math.pi / w, 4.89, 5e-3), (round(w, 4), round(2 * math.pi / w, 3)))
g = acvf([0.4, -0.5], [], 12)
rho = g / g[0]
hh = np.arange(13)
# ρ(h)=A r^h cos(ωh+c) で全ラグが一致するか（h=0,1 で A,c を決めて残りを照合）
rr = math.sqrt(0.5)
# A cos c = 1, A r cos(ω+c) = ρ1  → tan c
cc = math.atan2(math.cos(w) - rho[1] / rr, math.sin(w))
A = 1 / math.cos(cc)
check(T, "例3 ρ(h)=A(√0.5)^h cos(ωh+c) が h=0..12 で成り立つ",
      allclose(rho, A * rr ** hh * np.cos(w * hh + cc), 1e-9), round(cc, 4))
rts = np.roots([-0.2, -0.9, 1]).real
check(T, "よくある間違い φ1=0.9, φ2=0.2 の根 0.9221（単位円の内側）、φ1+φ2=1.1",
      close(min(abs(rts)), 0.9221) and close(0.9 + 0.2, 1.1, 1e-12), r(rts))
for cc_ in charts(T, "ar-sim") + charts(T, "acf-plot"):
    p = cc_["params"]
    okp = all(sl["min"] <= p[sl["param"]] <= sl["max"] for sl in cc_.get("sliders", []))
    check(T, f"charts {cc_['id']} の params がスライダー範囲内", okp, p)
p = charts(T, "acf-plot")[0]["params"]
check(T, "charts acf-plot φ1=0.4, φ2=−0.5 は複素根（減衰振動）で定常", p["phi"] ** 2 + 4 * p["phi2"] < 0 and min(abs(np.roots([-p["phi2"], -p["phi"], 1]))) > 1, "")
# 条件付き最尤＝最小二乗（正規尤度の σ を集約すると残差平方和の最小化）を乱数で傍証
x = simulate_arma([0.5, 0.3], [], 3000)
Xd = np.column_stack([x[1:-1], x[:-2]])
b_ls = np.linalg.lstsq(Xd, x[2:], rcond=None)[0]
g_, rho_ = sample_acf(x, 2)
b_yw = np.linalg.solve(np.array([[1, rho_[1]], [rho_[1], 1]]), rho_[1:3])
check(T, "乱数 AR(2) n=3000：最小二乗と YW がほぼ一致し真値に近い（許容 0.01 / 0.05）",
      allclose(b_ls, b_yw, 0.01) and allclose(b_ls, [0.5, 0.3], 0.05), (r(b_ls), r(b_yw)))

# ================================================================ ARMA
T = "arma-model"
th = sp.symbols("theta", real=True)
g0s, g1s = sp.symbols("g0 g1")
sol = sp.solve([sp.Eq(g0s, ph * g1s + s2 + th * (ph + th) * s2), sp.Eq(g1s, ph * g0s + th * s2)], [g0s, g1s])
check(T, "ARMA(1,1) γ(0)=(1+2φθ+θ²)σ²/(1−φ²)（sympy で連立を解く）",
      sp.simplify(sol[g0s] - (1 + 2 * ph * th + th ** 2) * s2 / (1 - ph ** 2)) == 0, "")
check(T, "ARMA(1,1) γ(1)=(1+φθ)(φ+θ)σ²/(1−φ²)（sympy）",
      sp.simplify(sol[g1s] - (1 + ph * th) * (ph + th) * s2 / (1 - ph ** 2)) == 0, "")
ok = True
for (a, b) in [(0.5, 0.4), (-0.7, 0.3), (0.9, -0.5), (0.2, 0.8)]:
    gg = acvf([a], [b], 6)
    ok &= close(gg[0], (1 + 2 * a * b + b * b) / (1 - a * a), 1e-8) and close(gg[1], (1 + a * b) * (a + b) / (1 - a * a), 1e-8)
    ok &= allclose(gg[2:], a * gg[1:-1], 1e-8)
    ok &= close(gg[1] / gg[0], (1 + a * b) * (a + b) / (1 + 2 * a * b + b * b), 1e-9)
check(T, "ARMA(1,1) の γ(0), γ(1), γ(h)=φγ(h−1), ρ(1) を ψ 重みの和と照合（4組）", ok, "")
# E[X_t ε_{t-1}] = ψ_1 σ²
check(T, "E[X_tε_{t−1}]=ψ1σ²=(φ+θ)σ²", close(psi_weights([0.5], [0.4], 3)[1], 0.9, 1e-12), "")
check(T, "φ=−θ で ρ(1)=0（ホワイトノイズ）", close(acvf([0.5], [-0.5], 1)[1], 0, 1e-9), "")
pw = psi_weights([0.5], [0.4], 6)
check(T, "ψ_j=(φ+θ)φ^{j−1}: 1, 0.9, 0.45, 0.225", allclose(pw[:4], [1, 0.9, 0.45, 0.225], 1e-12)
      and allclose(pw[1:], [0.9 * 0.5 ** (k - 1) for k in range(1, 6)], 1e-12), r(pw))
# π 重み：X_t = Σ π_j X_{t-j} + ε_t, π(z) = 1 - (1-φz)/(1+θz)
zz = sp.symbols("z")
ser = sp.series(1 - (1 - sp.Rational(1, 2) * zz) / (1 + sp.Rational(2, 5) * zz), zz, 0, 5).removeO()
pis = [float(ser.coeff(zz, k)) for k in range(1, 5)]
check(T, "π_j=(φ+θ)(−θ)^{j−1}: 0.9, −0.36, 0.144（sympy で (1−φz)/(1+θz) を展開）",
      allclose(pis[:3], [0.9, -0.36, 0.144], 1e-12) and allclose(pis, [0.9 * (-0.4) ** (k - 1) for k in range(1, 5)], 1e-12), pis)
gg = acvf([0.5], [0.4], 3)
check(T, "例1 γ(0)=2.08, γ(1)=1.44, ρ(1)=0.6923, ρ(2)=0.3462, ρ(3)=0.1731",
      close(gg[0], 2.08, 1e-9) and close(gg[1], 1.44, 1e-9) and allclose((gg / gg[0])[1:], [0.6923, 0.3462, 0.1731]),
      r(gg / gg[0]))
check(T, "例2 1+0.81/0.75=2.08", close(1 + 0.81 / 0.75, 2.08, 1e-12), "")
check(T, "例3 ρ(1)=9/13, ρ(2)=4.5/13, φ=ρ2/ρ1=0.5", close(1.44 / 2.08, 9 / 13, 1e-12) and close(0.72 / 2.08, 4.5 / 13, 1e-12), "")
tt = sp.symbols("t")
eq = sp.expand(13 * (1 + tt / 2) * (sp.Rational(1, 2) + tt) - 9 * (1 + tt + tt ** 2))
check(T, "例3 整理すると 10θ²−29θ+10=0（sympy）", sp.simplify(eq * (-4) - (10 * tt ** 2 - 29 * tt + 10)) == 0, eq)
check(T, "例3 解 θ=0.4, 2.5", sorted(float(v) for v in sp.solve(10 * tt ** 2 - 29 * tt + 10, tt)) == [0.4, 2.5], "")
xs, y = series(T, "ARMA(1,1)")
gg = acvf([0.5], [0.4], 10)
check(T, "charts line-data ARMA(1,1) の11点（小数第4位）", allclose(y, gg / gg[0]), r(gg / gg[0]))
xs, y = series(T, "AR(1)")
check(T, "charts line-data AR(1) φ=0.5 の11点（小数第4位）", allclose(y, [0.5 ** h for h in range(11)]), "")

# ================================================================ MA
T = "ma-model"
ok = True
for thetas in [[0.6], [0.5, -0.24], [0.3, -0.2, 0.5]]:
    q = len(thetas)
    th0 = [1.0] + thetas
    gg = acvf([], thetas, q + 3)
    ok &= allclose(gg[: q + 1], [sum(th0[jj] * th0[jj + h] for jj in range(q - h + 1)) for h in range(q + 1)], 1e-12)
    ok &= allclose(gg[q + 1:], 0, 1e-12)
check(T, "MA(q) 自己共分散公式（q=1,2,3）", ok, "")
xs_ = np.linspace(-5, 5, 2001)
rr1 = xs_ / (1 + xs_ ** 2)
check(T, "ρ(1)=θ/(1+θ²) の最大は θ=1 で 0.5、最小は θ=−1 で −0.5",
      close(rr1.max(), 0.5, 1e-12) and close(xs_[rr1.argmax()], 1, 1e-9) and close(rr1.min(), -0.5, 1e-12), "")
check(T, "θ と 1/θ が同じ ρ(1)（sympy）", sp.simplify(th / (1 + th ** 2) - (1 / th) / (1 + 1 / th ** 2)) == 0, "")
# PACF 公式を DL と照合
ok = True
for tv in [0.6, -0.5, 0.9, 0.3]:
    rho_ = [1, tv / (1 + tv ** 2)] + [0] * 10
    pk, _ = durbin_levinson(rho_, 10)
    ok &= allclose(pk, [-(-tv) ** k * (1 - tv ** 2) / (1 - tv ** (2 * (k + 1))) for k in range(1, 11)], 1e-10)
check(T, "MA(1) 偏自己相関の公式をダービン・レビンソンと照合（4値、k=1..10）", ok, "")
check(T, "例1 E=10, γ(0)=1.36, γ(1)=0.6, ρ(1)=0.4412、θ=1/0.6=1.6667 でも 0.4412",
      close(0.6 / 1.36, 0.4412) and close(1 / 0.6, 1.6667) and close((1 / 0.6) / (1 + 1 / 0.36), 0.4412), round(0.6 / 1.36, 4))
sols = sorted(float(v) for v in sp.solve(sp.Rational(2, 5) * tt ** 2 - tt + sp.Rational(2, 5), tt))
check(T, "例2 0.4θ²−θ+0.4=0 の解 0.5, 2（√(1−0.64)=0.6）", sols == [0.5, 2.0] and close(math.sqrt(1 - 4 * 0.16), 0.6, 1e-12), sols)
gg = acvf([], [0.5, -0.24], 4)
check(T, "例3 MA(2) γ=1.3076, 0.38, −0.24、ρ=0.2906, −0.1835, ρ(3)=0",
      allclose(gg[:3], [1.3076, 0.38, -0.24], 1e-12) and allclose((gg / gg[0])[1:3], [0.2906, -0.1835]) and close(gg[3], 0, 1e-12),
      r(gg / gg[0]))
p = charts(T, "acf-plot")[0]
check(T, "charts acf-plot θ=0.6 がスライダー範囲内、±0.95 での理論値 ±0.4993（0.5 を超えない）",
      p["sliders"][0]["min"] <= p["params"]["theta"] <= p["sliders"][0]["max"] and abs(0.95 / (1 + 0.95 ** 2)) < 0.5,
      round(0.95 / (1 + 0.9025), 4))
# 乱数：MA(1) の標本 ACF
x = simulate_arma([], [0.6], 200000)
_, ra = sample_acf(x, 3)
check(T, "乱数 MA(1) θ=0.6, n=20万：r1≈0.4412, r2≈r3≈0（許容 0.01）", allclose(ra[1:], [0.4412, 0, 0], 0.01), r(ra[1:]))
# 条件付き最小二乗の手順（ε0=0 で逐次）で θ が推定できるか
x = simulate_arma([], [0.6], 5000, c=0.0)


def css(tv):
    e = 0.0
    s = 0.0
    for xt in x:
        e = xt - tv * e
        s += e * e
    return s


grid = np.arange(0.40, 0.80, 0.005)
tb = grid[np.argmin([css(v) for v in grid])]
check(T, "乱数 n=5000：条件付き最小二乗（ε0=0 の逐次計算）の最小点 ≈0.6（許容 0.04）", close(tb, 0.6, 0.04), round(tb, 3))

# ================================================================ 自己共分散・自己相関
T = "autocovariance-autocorrelation"
d = np.array([4, 6, 5, 7, 6, 5, 3, 4], float)
gs, rs = sample_acf(d, 3)
check(T, "例1 平均5、偏差 −1,1,0,2,1,0,−2,−1", close(d.mean(), 5, 1e-12) and allclose(d - 5, [-1, 1, 0, 2, 1, 0, -2, -1], 0), "")
check(T, "例1 γ̂(0)=1.5, γ̂(1)=0.375, r1=0.25, r2=0, γ̂(3)=−0.75, r3=−0.5",
      allclose(gs, [1.5, 0.375, 0, -0.75], 1e-12) and allclose(rs, [1, 0.25, 0, -0.5], 1e-12), r(rs))
dd = d - 5
check(T, "例1 積の和 lag1=3, lag2=0, lag3=−6",
      [np.sum(dd[:-h] * dd[h:]) for h in (1, 2, 3)] == [3, 0, -6], "")
check(T, "例2 γ(0)=1.5625, γ(1)=0.9375", allclose(acvf([0.6], [], 1), [1.5625, 0.9375], 1e-9), "")
check(T, "例3 γ(0)=1.36, γ(1)=0.6, ρ(1)=0.4412", allclose(acvf([], [0.6], 2), [1.36, 0.6, 0], 1e-12) and close(0.6 / 1.36, 0.4412), "")
check(T, "n=100 の帯 ±0.196", close(1.96 / 10, 0.196, 1e-12), "")
# 非負定値性：n で割ると Toeplitz が半正定値、n−h だと崩れる例がある
neg_n = 0
neg_nh = 0
for _ in range(500):
    z = rng.normal(size=12).cumsum()
    n = len(z)
    dz = z - z.mean()
    gn = [np.sum(dz[: n - h] * dz[h:]) / n for h in range(n)]
    gnh = [np.sum(dz[: n - h] * dz[h:]) / (n - h) for h in range(n)]
    Tn = np.array([[gn[abs(a - b)] for b in range(n)] for a in range(n)])
    Tnh = np.array([[gnh[abs(a - b)] for b in range(n)] for a in range(n)])
    neg_n += np.linalg.eigvalsh(Tn).min() < -1e-10
    neg_nh += np.linalg.eigvalsh(Tnh).min() < -1e-10
check(T, "n で割る γ̂ の行列は常に非負定値（500回）、n−h で割ると負の固有値が出る例あり", neg_n == 0 and neg_nh > 0, (neg_n, neg_nh))
# WN の r_h ~ N(0,1/n)
n, R = 400, 4000
E = rng.normal(size=(R, n))
E = E - E.mean(1, keepdims=True)
r1 = np.sum(E[:, :-1] * E[:, 1:], 1) / np.sum(E * E, 1)
check(T, "乱数 WN n=400, 4000回：√n·r1 の標準偏差≈1（許容 0.04）、|r1|>1.96/√n の率≈0.05（許容 0.012）",
      close(np.std(r1) * math.sqrt(n), 1, 0.04) and close(np.mean(abs(r1) > 1.96 / math.sqrt(n)), 0.05, 0.012),
      (round(np.std(r1) * 20, 3), round(np.mean(abs(r1) > 0.098), 4)))
p = charts(T, "acf-plot")[0]
check(T, "charts acf-plot φ=0.6 がスライダー範囲内", p["sliders"][0]["min"] <= p["params"]["phi"] <= p["sliders"][0]["max"], "")

# ================================================================ 偏自己相関
T = "partial-autocorrelation"
pk, v = durbin_levinson([1, 0.5, 0.4, 0.3], 3)
check(T, "例1 φ11=0.5, φ22=0.2, φ33=0.0556", allclose(pk, [0.5, 0.2, 0.0556]), r(pk))
check(T, "例1 φ21=0.4、分子 0.04、分母 0.72",
      close(0.5 - 0.2 * 0.5, 0.4, 1e-12) and close(0.3 - 0.4 * 0.4 - 0.2 * 0.5, 0.04, 1e-12) and close(1 - 0.4 * 0.5 - 0.2 * 0.4, 0.72, 1e-12), "")
check(T, "例1 DL の結果を k×k 連立方程式の直接解と照合", allclose(pk, pacf_direct([1, 0.5, 0.4, 0.3], 3), 1e-12), "")
gg = acvf([0.5, 0.3], [], 8)
rho = gg / gg[0]
pk, v = durbin_levinson(rho, 7)
check(T, "例2 AR(2) の φ22=0.3=φ2、φ33..φ77=0", close(pk[1], 0.3, 1e-9) and allclose(pk[2:], 0, 1e-9), r(pk))
check(T, "例2 φ22 の分子 0.1469、分母 0.4898", close(rho[2] - rho[1] ** 2, 0.1469) and close(1 - rho[1] ** 2, 0.4898), "")
check(T, "例2 φ33 の分子 ρ3−0.5ρ2−0.3ρ1=0", close(rho[3] - 0.5 * rho[2] - 0.3 * rho[1], 0, 1e-9), "")
rho = [1, 0.6 / 1.36] + [0] * 5
pk, _ = durbin_levinson(rho, 3)
check(T, "例3 MA(1) θ=0.6：0.4412, −0.2417, 0.1406", allclose(pk, [0.4412, -0.2417, 0.1406]), r(pk))
# φ22 公式 = 偏相関係数
r1s, r2s = sp.symbols("r1 r2")
pc = (r2s - r1s * r1s) / sp.sqrt((1 - r1s ** 2) * (1 - r1s ** 2))
check(T, "φ22 と偏相関係数 r13·2（r12=r23=ρ1, r13=ρ2）が一致（sympy、|ρ1|<1）",
      sp.simplify(pc.subs(r1s, sp.Rational(3, 10)) - (r2s - sp.Rational(9, 100)) / (1 - sp.Rational(9, 100))) == 0
      and all(close(float(pc.subs({r1s: a, r2s: b})), durbin_levinson([1, a, b], 2)[0][1], 1e-12) for a, b in [(0.5, 0.4), (-0.3, 0.6), (0.8, 0.5)]), "")
# 残差どうしの相関＝φ_kk（AR(2) 理論の共分散で k=2,3）
gg = acvf([0.5, 0.3], [], 6)
ok = True
for k in (2, 3):
    G = np.array([[gg[abs(a - b)] for b in range(k + 1)] for a in range(k + 1)])  # X_t, X_{t-1}, ..., X_{t-k}
    mid = list(range(1, k))
    Sm = G[np.ix_(mid, mid)]
    ends = [0, k]
    C = G[np.ix_(ends, ends)] - G[np.ix_(ends, mid)] @ np.linalg.solve(Sm, G[np.ix_(mid, ends)])
    ok &= close(C[0, 1] / math.sqrt(C[0, 0] * C[1, 1]), durbin_levinson(gg / gg[0], k)[0][k - 1], 1e-10)
check(T, "残差どうしの相関＝φ_kk（AR(2), k=2,3）", ok, "")
# v_k 更新式 = 直接の予測誤差分散
ok = True
rho = gg / gg[0]
pk, v = durbin_levinson(rho, 4)
for k in range(1, 5):
    R = np.array([[rho[abs(a - b)] for b in range(k)] for a in range(k)])
    b = np.linalg.solve(R, rho[1:k + 1])
    ok &= close(v[k], 1 - b @ rho[1:k + 1], 1e-10)
check(T, "v_k=v_{k−1}(1−φ_kk²) と直接計算 γ(0)(1−Σφ_kjρ(j)) の一致（AR(2), k=1..4）、v_2·γ(0)=σ²=1",
      ok and close(v[2] * gg[0], 1, 1e-8), r(v * gg[0]))
# 標本 PACF ~ N(0,1/n) for k>p
R_, n = 1000, 400
vals = []
for _ in range(R_):
    x = simulate_arma([0.5, 0.3], [], n, burn=200)
    _, rs_ = sample_acf(x, 3)
    vals.append(durbin_levinson(rs_, 3)[0][2])
vals = np.array(vals)
check(T, "乱数 AR(2) n=400, 1000回：φ̂33 の √n·SD≈1（許容 0.08）、平均≈0（許容 0.01）",
      close(np.std(vals) * 20, 1, 0.08) and close(vals.mean(), 0, 0.01), (round(np.std(vals) * 20, 3), round(vals.mean(), 4)))
p = charts(T, "acf-plot")[0]
check(T, "charts acf-plot がスライダー範囲内・定常", all(s["min"] <= p["params"][s["param"]] <= s["max"] for s in p["sliders"]), "")

# ================================================================ ユール・ウォーカー
T = "yule-walker-equations"
r1v, r2v = 0.6, 0.45
R = np.array([[1, r1v], [r1v, 1]])
b = np.linalg.solve(R, [r1v, r2v])
check(T, "例1 φ̂1=0.515625, φ̂2=0.140625（明示解と行列解）",
      allclose(b, [0.515625, 0.140625], 1e-12) and close(r1v * (1 - r2v) / (1 - r1v ** 2), 0.515625, 1e-12)
      and close((r2v - r1v ** 2) / (1 - r1v ** 2), 0.140625, 1e-12), b)
s2h = 2 * (1 - b @ [r1v, r2v])
check(T, "例1 σ̂²=2×0.627344=1.2547、AR(1) で 1.28", close(1 - b @ [r1v, r2v], 0.627344, 5e-7) and close(s2h, 1.2547) and close(2 * (1 - 0.36), 1.28, 1e-12), round(s2h, 6))
check(T, "例1 φ̂2=φ̂22（DL）=0.1406", close(durbin_levinson([1, r1v, r2v], 2)[0][1], 0.1406), "")
check(T, "例2 SE=0.08、区間 [0.4432, 0.7568]", close(math.sqrt(0.64 / 100), 0.08, 1e-12) and allclose([0.6 - 1.96 * 0.08, 0.6 + 1.96 * 0.08], [0.4432, 0.7568], 1e-12), "")
check(T, "例3 ρ(1)=0.7143, ρ(2)=0.6571", allclose([0.5 / 0.7, 0.5 * 0.5 / 0.7 + 0.3], [0.7143, 0.6571]), "")
gg = acvf([0.5, 0.3], [], 2)
check(T, "σ²=γ(0)(1−Σφρ)（AR(2) 0.5,0.3 で σ²=1 に戻る）", close(gg[0] * (1 - 0.5 * gg[1] / gg[0] - 0.3 * gg[2] / gg[0]), 1, 1e-8), "")
# n で割った標本自己共分散からの YW 推定は常に定常
bad = 0
for _ in range(500):
    x = rng.normal(size=30).cumsum() + rng.normal(size=30)  # 非定常なデータでも
    _, rs_ = sample_acf(x, 3)
    Rm = np.array([[rs_[abs(a - b)] for b in range(3)] for a in range(3)])
    bb = np.linalg.solve(Rm, rs_[1:4])
    rts = np.roots(np.r_[-bb[::-1], 1])
    bad += min(abs(rts)) <= 1
check(T, "YW 推定（n で割る）は常に定常な AR(3) を返す（ランダムウォーク風データ500本）", bad == 0, bad)
# V[φ̂] ≈ (1−φ²)/n
R_, n = 2000, 500
est = []
for _ in range(R_):
    x = simulate_arma([0.6], [], n, burn=100)
    est.append(sample_acf(x, 1)[1][1])
est = np.array(est)
check(T, "乱数 AR(1) φ=0.6, n=500, 2000回：V[φ̂]·n≈1−φ²=0.64（許容 0.05）", close(np.var(est) * n, 0.64, 0.05), round(np.var(est) * n, 4))
p = charts(T, "acf-plot")[0]
check(T, "charts acf-plot がスライダー範囲内", all(s["min"] <= p["params"][s["param"]] <= s["max"] for s in p["sliders"]), "")

# ================================================================ 定常条件
T = "stationarity-condition"


def stat_roots(p1, p2):
    return min(abs(np.roots([-p2, -p1, 1]))) > 1 if p2 != 0 else abs(p1) < 1


def triangle(p1, p2):
    return p1 + p2 < 1 and p2 - p1 < 1 and abs(p2) < 1


grid = [(a, b) for a in np.arange(-2.3, 2.31, 0.037) for b in np.arange(-1.3, 1.31, 0.029)]
mism = sum(stat_roots(a, b) != triangle(a, b) for a, b in grid if abs(b) > 1e-9)
check(T, "AR(2) 三角形の条件 ⇔ 根がすべて単位円の外（格子 約1万点）", mism == 0, mism)
vert = [(0, 1), (2, -1), (-2, -1)]
check(T, "三角形の頂点 (0,1),(2,−1),(−2,−1)：3本の境界線の交点",
      [(a + b == 1) + (b - a == 1) + (b == -1) for a, b in vert] == [2, 2, 2], "")
check(T, "例1 根 1.1736, −2.8403、λ=0.8521, −0.3521",
      allclose(sorted(np.roots([-0.3, -0.5, 1]).real), [-2.8403, 1.1736]) and allclose(sorted(1 / np.roots([-0.3, -0.5, 1]).real), [-0.3521, 0.8521])
      and allclose(sorted(np.roots([1, -0.5, -0.3]).real), [-0.3521, 0.8521]), "")
check(T, "例1 √1.45=1.2042", close(math.sqrt(1.45), 1.2042), "")
rts = np.sort(np.roots([-0.2, -0.9, 1]).real)
check(T, "例2 根 0.9221, −5.4221", allclose(rts, [-5.4221, 0.9221]), r(rts))
check(T, "例3 1−1.2z+0.2z²=(1−z)(1−0.2z)、根 1, 5", sp.expand((1 - zz) * (1 - zz / 5)) == sp.expand(1 - sp.Rational(6, 5) * zz + zz ** 2 / 5)
      and allclose(sorted(np.roots([0.2, -1.2, 1]).real), [1, 5], 1e-12), "")
rts = np.roots([0.5, -0.4, 1])
check(T, "例4 根 0.4±1.3565i、|z|=√(1/0.5)=1.4142", allclose(abs(rts), [1.4142, 1.4142]) and allclose(sorted(rts.imag), [-1.3565, 1.3565]), "")
check(T, "φ1=1.5, φ2=−0.6 は定常", triangle(1.5, -0.6) and stat_roots(1.5, -0.6), r(abs(np.roots([0.6, -1.5, 1]))))
# 複素根で |λ|²=−φ2
check(T, "複素根のとき |λ|²=−φ2（例4 で 0.5）", close(abs(np.roots([1, -0.4, 0.5])[0]) ** 2, 0.5, 1e-12), "")
xs, y = series(T, "上の辺")
check(T, "charts 上の辺 φ2=1−|φ1|（17点）", allclose(y, [1 - abs(a) for a in xs], 1e-12), "")
xs, y = series(T, "下の辺")
check(T, "charts 下の辺 φ2=−1", allclose(y, [-1] * len(xs), 0), "")
xs, y = series(T, "複素根の境界")
check(T, "charts 放物線 φ2=−φ1²/4（小数第4位）", allclose(y, [-a * a / 4 for a in xs]), r([-a * a / 4 for a in xs]))
p = charts(T, "ar-sim")[0]["params"]
check(T, "charts ar-sim φ1=0.9, φ2=0.2 は三角形の外、φ1=0.7 で内", not triangle(p["phi"], p["phi2"]) and triangle(0.7, 0.2), "")
# MA(∞) 係数の絶対総和が有限（定常）/発散
check(T, "定常なら Σ|ψ_j|<∞（例1 で収束値 = 1/φ(1) =5）", close(np.sum(psi_weights([0.5, 0.3], [], 2000)), 1 / (1 - 0.8), 1e-8), "")

# ================================================================ 予測
T = "forecasting"
mu_, ph_ = 5.0, 0.6
fc = [7.0]
for h in range(1, 11):
    fc.append(2 + ph_ * fc[-1])
var = [0.0] + [sum(ph_ ** (2 * jj) for jj in range(h)) for h in range(1, 11)]
check(T, "例1 予測 6.2, 5.72, 5.432、μ+φ^h(X_n−μ) と一致",
      allclose(fc[1:4], [6.2, 5.72, 5.432], 1e-12) and allclose(fc, [5 + 0.6 ** h * 2 for h in range(11)], 1e-12), "")
check(T, "例1 誤差分散 1, 1.36, 1.4896 と σ²(1−φ^{2h})/(1−φ²) の一致",
      allclose(var[1:4], [1, 1.36, 1.4896], 1e-12) and allclose(var[1:], [(1 - 0.6 ** (2 * h)) / 0.64 for h in range(1, 11)], 1e-12), "")
check(T, "例1 区間 h=1 [4.24, 8.16], h=2 [3.434, 8.006], 極限 [2.55, 7.45]",
      allclose([6.2 - 1.96, 6.2 + 1.96], [4.24, 8.16], 1e-12) and allclose([5.72 - 1.96 * math.sqrt(1.36), 5.72 + 1.96 * math.sqrt(1.36)], [3.434, 8.006], 5e-4)
      and allclose([5 - 1.96 * 1.25, 5 + 1.96 * 1.25], [2.55, 7.45], 1e-12) and close(math.sqrt(1 / 0.64), 1.25, 1e-12), "")
c = charts(T, "line-data")[0]["params"]
sd = np.sqrt(var)
ys = {s["label"]: s["y"] for s in c["series"]}
check(T, "charts 予測値11点（小数第3位）", allclose(ys["予測値"], fc, 5e-4), r(fc, 3))
check(T, "charts 95%予測区間の上限11点（小数第3位）", allclose(ys["95%予測区間の上限"], np.array(fc) + 1.96 * sd, 5e-4), r(np.array(fc) + 1.96 * sd, 3))
check(T, "charts 95%予測区間の下限11点（小数第3位）", allclose(ys["95%予測区間の下限"], np.array(fc) - 1.96 * sd, 5e-4), r(np.array(fc) - 1.96 * sd, 3))
check(T, "例2 MA(1) 予測 10.9 / 10、誤差分散 1, 1.36", close(10 + 0.6 * 1.5, 10.9, 1e-12) and allclose(np.cumsum(psi_weights([], [0.6], 3) ** 2), [1, 1.36, 1.36], 1e-12), "")
check(T, "例3 AR(2) 予測 1.3, 1.25、2期先誤差分散 1.25",
      close(0.5 * 2 + 0.3 * 1, 1.3, 1e-12) and close(0.5 * 1.3 + 0.3 * 2, 1.25, 1e-12) and close(np.sum(psi_weights([0.5, 0.3], [], 2) ** 2), 1.25, 1e-12), "")
# 予測誤差分散のシミュレーション（AR(1)、h=3）
R_ = 200000
x0 = np.full(R_, 7.0)
xx = x0.copy()
for h in range(3):
    xx = 2 + 0.6 * xx + rng.normal(size=R_)
check(T, "乱数 20万本：X_{n+3} の平均≈5.432、分散≈1.4896（許容 0.01 / 0.02）",
      close(xx.mean(), 5.432, 0.01) and close(xx.var(), 1.4896, 0.02), (round(xx.mean(), 4), round(xx.var(), 4)))

# ================================================================ ARIMA
T = "arima-model"
check(T, "例1 1−1.2z+0.2z²=(1−z)(1−0.2z)、根 1, 5",
      sp.expand((1 - zz) * (1 - zz / 5) - (1 - sp.Rational(6, 5) * zz + zz ** 2 / 5)) == 0, "")
fcd = [3.0]
for _ in range(3):
    fcd.append(0.2 * fcd[-1])
lv = 103 + np.cumsum(fcd[1:])
check(T, "例2 予測 103.6, 103.72, 103.744、極限 103.75",
      allclose(lv, [103.6, 103.72, 103.744], 1e-9) and close(103 + 3 * 0.2 / 0.8, 103.75, 1e-12), r(lv, 3))
xl = [100.0, 103.0]
for _ in range(60):
    xl.append(1.2 * xl[-1] - 0.2 * xl[-2])
check(T, "例2 AR(2) 形で予測を回しても同じ・極限 103.75", allclose(xl[2:5], [103.6, 103.72, 103.744], 1e-9) and close(xl[-1], 103.75, 1e-9), "")
pw = psi_weights([1.2, -0.2], [], 4)
check(T, "例2 ψ重み 1, 1.2, 1.24、誤差分散 1, 2.44, 3.9776",
      allclose(pw[:3], [1, 1.2, 1.24], 1e-12) and allclose(np.cumsum(pw[:3] ** 2), [1, 2.44, 3.9776], 1e-12), r(pw))
check(T, "例2 ψ重みは 1.25 に収束し、誤差分散は発散", close(psi_weights([1.2, -0.2], [], 200)[-1], 1.25, 1e-9), "")
fcs = [50.0]
for obs in [50, 52, 51]:
    fcs.append(0.3 * obs + 0.7 * fcs[-1])
check(T, "例3 指数平滑 α=1+θ=0.3、予測 50→50.6→50.72", close(1 + (-0.7), 0.3, 1e-12) and allclose(fcs[1:], [50, 50.6, 50.72], 1e-12), r(fcs))
# ARIMA(0,1,1) 予測 = 指数平滑（乱数）
e = rng.normal(size=300)
X = np.cumsum(e + np.r_[0, -0.7 * e[:-1]])
# ARIMA 側：X̂_{t+1}=X_t+θ ε_t,  ε_t = X_t − X̂_t ; 指数平滑：S_{t+1}=αX_t+(1−α)S_t, 同じ初期値
xa = X[0]
xs_ = X[0]
ok = True
for t_ in range(300):
    et = X[t_] - xa
    xa = X[t_] + (-0.7) * et
    xs_ = 0.3 * X[t_] + 0.7 * xs_
    ok &= abs(xa - xs_) < 1e-9
check(T, "ARIMA(0,1,1) の1期先予測と単純指数平滑が同じ初期値なら全時点で一致（乱数300点）", ok, "")
c = charts(T, "line-data")[0]["params"]
Xs = np.array(c["series"][0]["y"])
dX = np.array(c["series"][1]["y"])
check(T, "charts 原系列の差分 = 差分系列（t≥2、丸め許容 0.011）", allclose(np.diff(Xs), dX[1:], 0.011) and len(Xs) == len(dX) == len(c["x"]) == 60,
      round(float(np.max(abs(np.diff(Xs) - dX[1:]))), 4))
resid = dX[1:] - 0.5 * dX[:-1]
_, rsd = sample_acf(dX, 1)
check(T, "charts 差分系列は AR(1) φ=0.5 として妥当（標本 r1 が 0.5±2/√60、残差の SD が 1±0.25）",
      abs(rsd[1] - 0.5) < 2 / math.sqrt(60) and abs(np.std(resid) - 1) < 0.25, (round(rsd[1], 3), round(np.std(resid), 3)))

# ================================================================ 季節調整・時系列
y = np.array([52, 61, 69, 47, 55, 65, 74, 49, 61, 67, 79, 53], float)
cma = {t: (0.5 * y[t - 3] + y[t - 2] + y[t - 1] + y[t] + 0.5 * y[t + 1]) / 4 for t in range(3, 11)}
T = "time-series"
check(T, "例 T̂3=230.5/4=57.625", close(0.5 * 52 + 61 + 69 + 47 + 0.5 * 55, 230.5, 1e-12) and close(cma[3], 57.625, 1e-12), "")
check(T, "例 中心化移動平均 t=3..10", allclose(list(cma.values()), [57.625, 58.5, 59.625, 60.5, 61.5, 62.5, 63.375, 64.5], 1e-12), list(cma.values()))
devs = [y[t - 1] - cma[t] for t in range(3, 11)]
check(T, "例 y−T̂ = 11.375, −11.5, −4.625, 4.5, 12.5, −13.5, −2.375, 2.5", allclose(devs, [11.375, -11.5, -4.625, 4.5, 12.5, -13.5, -2.375, 2.5], 1e-12), devs)
check(T, "例 重みは両端 1/(2s)、内側 1/s（2つの長さ4移動平均の平均と一致）",
      all(close(cma[t], (y[t - 3:t + 1].mean() + y[t - 2:t + 2].mean()) / 2, 1e-12) for t in range(3, 11)), "")
xs, yc = series(T, "中心化移動平均")
check(T, "charts 傾向12点（両端は最も近い値で埋め）", allclose(yc, [cma[3]] * 2 + list(cma.values()) + [cma[10]] * 2, 1e-12), "")
xs, yo = series(T, "原系列")
check(T, "charts 原系列＝表のデータ", allclose(yo, y, 0), "")
T = "seasonal-adjustment"
dv = {t: y[t - 1] - cma[t] for t in range(3, 11)}
q = {k: [dv[t] for t in dv if (t - 1) % 4 == k] for k in range(4)}
means = [np.mean(q[k]) for k in range(4)]
check(T, "表 Q1..Q4 の平均 −3.5, 3.5, 11.9375, −12.5", allclose(means, [-3.5, 3.5, 11.9375, -12.5], 1e-12), means)
check(T, "平均の合計 −0.5625、1/4 で −0.140625", close(sum(means), -0.5625, 1e-12) and close(sum(means) / 4, -0.140625, 1e-12), "")
S = np.array(means) - sum(means) / 4
check(T, "Ŝ = −3.3594, 3.6406, 12.0781, −12.3594（合計0）", allclose(S, [-3.3594, 3.6406, 12.0781, -12.3594]) and close(S.sum(), 0, 1e-12), r(S))
adj = y - np.tile(S, 3)
check(T, "1年目 Q3 56.9219、Q4 59.3594、増加 2.4、原系列は 22 減",
      allclose(adj[2:4], [56.9219, 59.3594]) and close(adj[3] - adj[2], 2.4, 0.05) and close(y[2] - y[3], 22, 0), r(adj))
xs, ya = series(T, "季節調整済み")
check(T, "charts 季節調整済み12点（小数第2位）", allclose(ya, adj, 5e-3), r(adj, 2))
check(T, "季節差分 3,4,5,2,6,2,5,4", allclose(y[4:] - y[:-4], [3, 4, 5, 2, 6, 2, 5, 4], 0), "")
check(T, "周期 s の季節成分は 1−B^s で消える", allclose(np.tile(S, 3)[4:] - np.tile(S, 3)[:-4], 0, 1e-12), "")
# 航空機モデルの展開：(1+θB)(1+ΘB^12) は B, B^12, B^13 の項
Bs, Th, th2 = sp.symbols("B Theta theta2")
ex = sp.Poly(sp.expand((1 + th2 * Bs) * (1 + Th * Bs ** 12)), Bs)
check(T, "航空機モデルの MA 部分は B, B^12, B^13 の項を持つ（sympy）", sorted(m[0] for m in ex.monoms()) == [0, 1, 12, 13], ex.monoms())

# ================================================================ スペクトル
T = "spectrum"


def f_arma(lam, phi, theta, sigma2=1.0):
    zz_ = np.exp(-1j * lam)
    num = abs(1 + sum(t_ * zz_ ** (k + 1) for k, t_ in enumerate(theta))) ** 2
    den = abs(1 - sum(p_ * zz_ ** (k + 1) for k, p_ in enumerate(phi))) ** 2
    return sigma2 / (2 * math.pi) * num / den


ok = True
for p_ in [0.6, -0.6, 0.9]:
    for lam in np.linspace(0, math.pi, 7):
        ok &= close(f_arma(lam, [p_], []), 1 / (2 * math.pi * (1 - 2 * p_ * math.cos(lam) + p_ ** 2)), 1e-12)
check(T, "AR(1) の表の式 = ARMA 公式", ok, "")
ok = True
for t_ in [0.6, -0.4]:
    for lam in np.linspace(0, math.pi, 7):
        ok &= close(f_arma(lam, [], [t_]), (1 + 2 * t_ * math.cos(lam) + t_ ** 2) / (2 * math.pi), 1e-12)
check(T, "MA(1) の表の式 = ARMA 公式 = 定義（γ(0)+2γ(1)cosλ）/2π", ok, "")
L = sp.symbols("lambda", real=True)
expr = sp.expand((1 - ph * sp.cos(L)) ** 2 + (ph * sp.sin(L)) ** 2)
check(T, "|1−φe^{−iλ}|²=1−2φcosλ+φ²（sympy）", sp.simplify(expr - (1 - 2 * ph * sp.cos(L) + ph ** 2)) == 0, "")
ok = True
for phi_, theta_ in [([0.6], []), ([-0.6], []), ([0.5, 0.3], []), ([0.5], [0.4]), ([], [0.5, -0.24]), ([0.4, -0.5], [])]:
    gg = acvf(phi_, theta_, 4)
    for h in range(5):
        val = integrate.quad(lambda l: math.cos(h * l) * f_arma(l, phi_, theta_), -math.pi, math.pi, limit=200)[0]
        ok &= close(val, gg[h], 1e-7)
check(T, "数値積分 ∫e^{ihλ}f(λ)dλ = γ(h)（6モデル、h=0..4、許容 1e-7）", ok, "")
# 定義の和（打ち切り）で f を作ると公式と一致
gg = acvf([0.6], [], 300)
ok = all(close((gg[0] + 2 * np.sum(gg[1:] * np.cos(np.arange(1, 301) * lam))) / (2 * math.pi), f_arma(lam, [0.6], []), 1e-9)
         for lam in np.linspace(0, math.pi, 9))
check(T, "定義式 (γ(0)+2Σγ(h)cos hλ)/2π を打ち切って計算した値 = AR(1) 公式", ok, "")
check(T, "例1 f(0)=0.9947, f(π)=0.0622, f(π/2)=0.1170, 比 16",
      close(f_arma(0, [0.6], []), 0.9947) and close(f_arma(math.pi, [0.6], []), 0.0622) and close(f_arma(math.pi / 2, [0.6], []), 0.1170)
      and close(2.56 / 0.16, 16, 1e-12) and close(1 - 1.2 + 0.36, 0.16, 1e-12), "")
val = integrate.quad(lambda l: f_arma(l, [0.6], []), -math.pi, math.pi)[0]
check(T, "例1 ∫f=1.5625", close(val, 1.5625, 1e-9), round(val, 6))
check(T, "例2 MA(1) f(0)=0.4074, f(π)=0.0255", close(f_arma(0, [], [0.6]), 0.4074) and close(f_arma(math.pi, [], [0.6]), 0.0255), "")
check(T, "WN の f=1/(2π)=0.1592", close(1 / (2 * math.pi), 0.1592), "")
c = charts(T, "line-data")[0]["params"]
lx = np.array(c["x"])
check(T, "charts x は 0..π を π/12 刻み（小数第4位）", allclose(lx, np.arange(13) * math.pi / 12), "")
ys = {s["label"]: s["y"] for s in c["series"]}
check(T, "charts AR(1) φ=0.6 の13点", allclose(ys["AR(1) φ=0.6"], [f_arma(l, [0.6], []) for l in np.arange(13) * math.pi / 12]), "")
check(T, "charts AR(1) φ=−0.6 の13点", allclose(ys["AR(1) φ=−0.6"], [f_arma(l, [-0.6], []) for l in np.arange(13) * math.pi / 12]), "")
check(T, "charts ホワイトノイズ 0.1592 一定", allclose(ys["ホワイトノイズ（φ=0）"], [1 / (2 * math.pi)] * 13), "")
# ピリオドグラム：WN で平均≈σ²/2π、分散が n で減らない
ok = True
info = []
for n in (128, 1024):
    E = rng.normal(size=(2000, n))
    E = E - E.mean(1, keepdims=True)
    I = np.abs(np.fft.fft(E, axis=1)) ** 2 / (2 * math.pi * n)
    Ij = I[:, n // 4]
    info.append((round(Ij.mean(), 4), round(Ij.std(), 4)))
    ok &= close(Ij.mean(), 1 / (2 * math.pi), 0.012) and close(Ij.std(), 1 / (2 * math.pi), 0.015)
check(T, "乱数 WN 2000本：I(λ_j) の平均≈SD≈1/(2π)（n=128, 1024 とも、許容 0.012/0.015）＝分散が減らない", ok, info)

# ================================================================ ホワイトノイズ
T = "white-noise"
rv = np.array([0.15, -0.10, 0.08, 0.05, -0.12])
terms = rv ** 2 / (100 - np.arange(1, 6))
Q = 100 * 102 * terms.sum()
check(T, "例2 各項 0.000227, 0.000102, 0.0000660, 0.0000260, 0.000152",
      allclose(terms, [0.000227, 0.000102, 0.0000660, 0.0000260, 0.000152], 5e-7), r(terms, 7))
check(T, "例2 Q=5.844", close(Q, 5.844, 5e-4), round(Q, 4))
check(T, "例2 χ²(5) 上側5%点 11.070、p 値 0.322", close(stats.chi2.ppf(0.95, 5), 11.070, 5e-4) and close(stats.chi2.sf(Q, 5), 0.322, 5e-4),
      (round(stats.chi2.ppf(0.95, 5), 4), round(stats.chi2.sf(Q, 5), 4)))
check(T, "例2 ボックス・ピアス 100×0.0558=5.58", close(np.sum(rv ** 2), 0.0558, 1e-12) and close(100 * np.sum(rv ** 2), 5.58, 1e-9), "")
check(T, "例1 帯 ±0.196 / ±0.098", close(1.96 / 10, 0.196, 1e-12) and close(1.96 / 20, 0.098, 1e-12), "")
# LB の帰無分布
n, m, R_ = 200, 10, 4000
E = rng.normal(size=(R_, n))
E = E - E.mean(1, keepdims=True)
den = np.sum(E * E, 1)
Qs = n * (n + 2) * sum((np.sum(E[:, :-h] * E[:, h:], 1) / den) ** 2 / (n - h) for h in range(1, m + 1))
check(T, "乱数 WN n=200, 4000回：LB(m=10) の平均≈10（許容 0.3）、棄却率≈0.05（許容 0.012）",
      close(Qs.mean(), 10, 0.3) and close(np.mean(Qs > stats.chi2.ppf(0.95, m)), 0.05, 0.012), (round(Qs.mean(), 3), round(np.mean(Qs > stats.chi2.ppf(0.95, m)), 4)))
p = charts(T, "acf-plot")[0]
check(T, "charts acf-plot n=200 がスライダー範囲内", p["sliders"][0]["min"] <= p["params"]["n"] <= p["sliders"][0]["max"], "")

# ================================================================ コレログラム
T = "correlogram"
check(T, "例 帯 0.196、バートレット √(1.405/100)=0.1185、×1.96=0.2323、|r2|,|r3|<0.2323",
      close(1 + 2 * 0.45 ** 2, 1.405, 1e-12) and close(math.sqrt(0.01405), 0.1185) and close(1.96 * math.sqrt(0.01405), 0.2323)
      and 0.08 < 0.2323 and 0.05 < 0.2323, round(1.96 * math.sqrt(0.01405), 4))
check(T, "n=400 の帯 0.098", close(1.96 / 20, 0.098, 1e-12), "")
# バートレットの公式を MA(1) θ=0.6 のシミュレーションで確認（h=2）
R_, n = 3000, 400
E = rng.normal(size=(R_, n + 1))
Xm = E[:, 1:] + 0.6 * E[:, :-1]
Xm = Xm - Xm.mean(1, keepdims=True)
r2 = np.sum(Xm[:, :-2] * Xm[:, 2:], 1) / np.sum(Xm * Xm, 1)
rho1 = 0.6 / 1.36
check(T, "乱数 MA(1) n=400, 3000回：n·V[r2]≈1+2ρ1²=1.389（許容 0.08）", close(np.var(r2) * n, 1 + 2 * rho1 ** 2, 0.08), round(np.var(r2) * n, 4))
for cc_ in charts(T, "acf-plot"):
    pp = cc_["params"]
    check(T, f"charts acf-plot（{pp['model']}）の params がスライダー範囲内", all(s["min"] <= pp[s["param"]] <= s["max"] for s in cc_["sliders"]), "")
# 20 ラグで平均1本
check(T, "20本見れば平均1本が帯の外（20×0.05）", close(20 * 0.05, 1, 1e-12), "")

# ================================================================ 反転可能性
T = "invertibility"
ok = sp.simplify((1 + th ** -2) * th ** 2 * s2 - (1 + th ** 2) * s2) == 0 and sp.simplify(th ** -1 * th ** 2 * s2 - th * s2) == 0
check(T, "(θ,σ²) と (1/θ, θ²σ²) で同じ自己共分散（sympy）", ok, "")
check(T, "例1 B：(1+2.7778)×0.36=1.36、1.6667×0.36=0.6",
      close((1 + (1 / 0.6) ** 2) * 0.36, 1.36, 1e-12) and close((1 / 0.6) * 0.36, 0.6, 1e-12) and close((1 / 0.6) ** 2, 2.7778), "")
ser = sp.series(1 - 1 / (1 + sp.Rational(3, 5) * zz), zz, 0, 5).removeO()
pis = [float(ser.coeff(zz, k)) for k in range(1, 4)]
check(T, "例1 AR(∞) 係数 0.6, −0.36, 0.216（X=θX−θ²X+…、sympy 展開）", allclose(pis, [0.6, -0.36, 0.216], 1e-12), pis)
# 反転：ε_t = Σ(−θ)^j X_{t−j} で雑音を復元
e = rng.normal(size=2000)
Xi = e.copy()
Xi[1:] += 0.6 * e[:-1]
K = 60
rec = np.array([sum((-0.6) ** jj * Xi[t_ - jj] for jj in range(K)) for t_ in range(K, 2000)])
check(T, "乱数：|θ|<1 なら Σ(−θ)^j X_{t−j}（60項）で ε_t を復元（許容 1e-9）", allclose(rec, e[K:], 1e-9), "")
check(T, "例2 1+0.5z−0.24z²=(1+0.8z)(1−0.3z)、根 −1.25, 3.3333",
      sp.expand((1 + sp.Rational(4, 5) * zz) * (1 - sp.Rational(3, 10) * zz) - (1 + zz / 2 - sp.Rational(6, 25) * zz ** 2)) == 0
      and allclose(sorted(np.roots([-0.24, 0.5, 1]).real), [-1.25, 3.3333]), "")
xs, y = series(T, "MA(1)")
check(T, "charts ρ(1)=θ/(1+θ²) の25点（小数第4位）", allclose(y, [a / (1 + a * a) for a in xs]), "")
check(T, "θ=0.5 と 2 がともに 0.4", close(0.5 / 1.25, 0.4, 1e-12) and close(2 / 5, 0.4, 1e-12), "")

# ================================================================ 単位根
T = "unit-root"
# ランダムウォークの分散・相関
R_ = 100000
E = rng.normal(size=(R_, 110))
W = E.cumsum(1)
check(T, "乱数 RW 10万本：V[X_100]≈100（許容 1.5）、Corr(X_10,X_20)≈0.7071, Corr(X_100,X_110)≈0.9535（許容 0.006）",
      close(W[:, 99].var(), 100, 1.5) and close(np.corrcoef(W[:, 9], W[:, 19])[0, 1], 0.7071, 0.006)
      and close(np.corrcoef(W[:, 99], W[:, 109])[0, 1], 0.9535, 0.006), "")
check(T, "例1 τ=−0.05/0.03=−1.667、−2.86 より大きい、−1.645 より小さい",
      close(-0.05 / 0.03, -1.667, 5e-4) and -1.667 > -2.86 and -1.667 < -1.645 and close(stats.norm.ppf(0.05), -1.645, 5e-4), "")
check(T, "例2 0.9^10=0.349、半減期 6.58 期・13.51 期",
      close(0.9 ** 10, 0.349, 5e-4) and close(math.log(0.5) / math.log(0.9), 6.58, 5e-3) and close(math.log(0.5) / math.log(0.95), 13.51, 5e-3),
      (round(0.9 ** 10, 4), round(math.log(0.5) / math.log(0.9), 3), round(math.log(0.5) / math.log(0.95), 3)))
check(T, "例3 φ(1)=1−1.2+0.2=0", close(1 - 1.2 + 0.2, 0, 1e-12), "")
# DF 棄却点のシミュレーション（H0：ランダムウォーク、n=1000、各 20000 回）
n, R_ = 1000, 20000
taus = {"none": [], "c": [], "ct": []}
tt_ = np.arange(1, n + 1, dtype=float)
for b0 in range(0, R_, 2000):
    E = rng.normal(size=(2000, n + 1))
    X = E.cumsum(1)
    y_ = np.diff(X, axis=1)  # ΔX_t
    xl_ = X[:, :-1]          # X_{t-1}
    for kind in taus:
        if kind == "none":
            xr = xl_
            yr = y_
        elif kind == "c":
            xr = xl_ - xl_.mean(1, keepdims=True)
            yr = y_ - y_.mean(1, keepdims=True)
        else:
            Z = np.column_stack([np.ones(n), tt_])
            P = np.eye(n) - Z @ np.linalg.solve(Z.T @ Z, Z.T)
            xr = xl_ @ P
            yr = y_ @ P
        k = {"none": 1, "c": 2, "ct": 3}[kind]
        dhat = np.sum(xr * yr, 1) / np.sum(xr * xr, 1)
        res = yr - dhat[:, None] * xr
        s2_ = np.sum(res * res, 1) / (n - k)
        taus[kind].append(dhat / np.sqrt(s2_ / np.sum(xr * xr, 1)))
q5 = {k_: float(np.quantile(np.concatenate(v_), 0.05)) for k_, v_ in taus.items()}
# モンテカルロ誤差：SE≈√(p(1−p)/R)/f(q) ≈ 0.0015/0.3 ≈ 0.005。有限標本 n=1000 と漸近値の差（≈0.003）も見込んで許容 0.03
check(T, "DF 5%棄却点：定数項なし ≈ −1.95（シミュレーション n=1000, 2万回、許容 0.03）", close(q5["none"], -1.95, 0.03), round(q5["none"], 4))
check(T, "DF 5%棄却点：定数項のみ ≈ −2.86（同上）", close(q5["c"], -2.86, 0.03), round(q5["c"], 4))
check(T, "DF 5%棄却点：定数項とトレンド ≈ −3.41（同上）", close(q5["ct"], -3.41, 0.03), round(q5["ct"], 4))
check(T, "DF 棄却点は −1.645 より小さく、項を増やすほど小さくなる", -1.645 > q5["none"] > q5["c"] > q5["ct"], "")
# 例1 の状況で t 分布を使うと過剰棄却：定数項ありで τ<−1.645 となる率
rate = float(np.mean(np.concatenate(taus["c"]) < -1.645))
check(T, "H0 のもとで −1.645 を使うと棄却率が 5% を大きく超える（定数項あり）", rate > 0.2, round(rate, 3))
# 見せかけの回帰：独立な RW どうしの回帰で |t|>1.96 の率
R_, n = 2000, 200
A = rng.normal(size=(R_, n)).cumsum(1)
Bv = rng.normal(size=(R_, n)).cumsum(1)
A = A - A.mean(1, keepdims=True)
Bv = Bv - Bv.mean(1, keepdims=True)
bb = np.sum(A * Bv, 1) / np.sum(Bv * Bv, 1)
res = A - bb[:, None] * Bv
se = np.sqrt(np.sum(res * res, 1) / (n - 2) / np.sum(Bv * Bv, 1))
check(T, "見せかけの回帰：独立な RW どうしで |t|>1.96 の率が 5% を大きく超える（>0.5）", np.mean(abs(bb / se) > 1.96) > 0.5, round(float(np.mean(abs(bb / se) > 1.96)), 3))
for cc_ in charts(T, "ar-sim"):
    pp = cc_["params"]
    check(T, "charts ar-sim φ=0.95 がスライダー範囲内", all(s["min"] <= pp[s["param"]] <= s["max"] for s in cc_["sliders"]), "")

# ================================================================ 弱定常性
T = "weak-stationarity"
check(T, "例1 MA(1) θ=0.5：γ(0)=1.25, γ(1)=0.5, ρ(1)=0.4", allclose(acvf([], [0.5], 2), [1.25, 0.5, 0], 1e-12) and close(0.5 / 1.25, 0.4, 1e-12), "")
check(T, "例2 Corr(X10,X20)=√0.5=0.7071, √(100/110)=0.9535",
      close(10 / math.sqrt(200), 0.7071) and close(math.sqrt(100 / 110), 0.9535), "")
check(T, "例3 γ(0)=σ²/(1−φ²)（sympy で γ0=φ²γ0+σ² を解く）、φ=0.6 で 1.5625",
      sp.simplify(sp.solve(sp.Eq(g0s, ph ** 2 * g0s + s2), g0s)[0] - s2 / (1 - ph ** 2)) == 0 and close(1 / 0.64, 1.5625, 1e-12), "")
W2 = W[:, :30]
covm = np.cov(W2[:, [4, 19]].T)
check(T, "乱数 RW：Cov(X_5,X_20)≈min(5,20)=5（許容 0.15）", close(covm[0, 1], 5, 0.15), round(covm[0, 1], 3))
ok = True
for pp in [charts(T, "ar-sim")[0]["params"]]:
    ok &= all(s["min"] <= pp[s["param"]] <= s["max"] for s in charts(T, "ar-sim")[0]["sliders"])
check(T, "charts ar-sim φ=0.7 がスライダー範囲内", ok, "")

# ================================================================
ok_n = sum(results)
ng_n = len(results) - ok_n
print(f"\n合計: OK {ok_n} / NG {ng_n}")
