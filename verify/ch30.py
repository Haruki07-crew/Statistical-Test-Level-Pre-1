"""第30章 検算スクリプト（モデル選択）。
共通データ x=1..8, y=(2.2,3.3,3.8,3.7,4.9,4.8,4.3,4.4) の0〜3次多項式について、
S_e・-2ℓ・AIC・BIC（定数込み、k=p+1）・Cp・調整済みR²・LOOCV（e_i/(1-h_ii) と実際の一個抜き）・
4分割CV・GCV・赤池ウェイトを numpy で生データから再計算する。
コイン（100回中60回）の AIC/BIC・厳密な周辺尤度 B(61,41)・ラプラス近似、
真の関数 1+1.2x-0.1x²・σ²=0.16 のバイアス²・バリアンス・訓練誤差・テスト誤差、
正規平均の平均対数尤度の偏り 1/n、KL 情報量の例、縮小推定量 cX̄ の MSE を sympy/numpy/scipy で確認する。
charts の params は各 md の front matter を yaml で読み照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など）。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, integrate, special

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch30" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid):
    return [c for c in front(term).get("charts", []) if c.get("id") == cid]


def series(chart):
    return {s["label"]: s["y"] for s in chart["params"]["series"]}


# ================================================================ 共通データ
x = np.arange(1, 9, dtype=float)
y = np.array([2.2, 3.3, 3.8, 3.7, 4.9, 4.8, 4.3, 4.4])
n = len(y)
L2PI = math.log(2 * math.pi)


def design(xx, d):
    return np.vander(xx, d + 1, increasing=True)


def fit(xx, yy, d):
    X = design(xx, d)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return beta


res = {}
for d in range(4):
    X = design(x, d)
    beta = fit(x, y, d)
    e = y - X @ beta
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p = d + 1
    Se = float(e @ e)
    m2l = n * math.log(2 * math.pi * Se / n) + n
    loo_formula = e / (1 - h)
    loo_actual = np.array([y[i] - design(x[i:i + 1], d) @ fit(np.delete(x, i), np.delete(y, i), d) for i in range(n)]).ravel()
    res[d] = dict(beta=beta, e=e, h=h, p=p, Se=Se, m2l=m2l, AIC=m2l + 2 * (p + 1), BIC=m2l + (p + 1) * math.log(n),
                  AIC0=n * math.log(Se / n) + 2 * (p + 1), loo_f=loo_formula, loo_a=loo_actual,
                  LOOCV=float(np.mean(loo_formula ** 2)), GCV=(Se / n) / (1 - p / n) ** 2)
ST = float(((y - y.mean()) ** 2).sum())
sig2 = res[3]["Se"] / (n - 4)
for d in range(4):
    r = res[d]
    r["Cp"] = r["Se"] / sig2 - n + 2 * r["p"]
    r["R2adj"] = 1 - (r["Se"] / (n - r["p"])) / (ST / (n - 1))
    r["Err"] = r["Se"] / n + 2 * sig2 * r["p"] / n

# 4分割 CV
folds = [[1, 5], [2, 6], [3, 7], [4, 8]]
cv4 = {}
fold_detail = []
for d in range(4):
    mses = []
    for f in folds:
        mask = np.isin(x, f)
        b = fit(x[~mask], y[~mask], d)
        pred = design(x[mask], d) @ b
        err = y[mask] - pred
        mses.append(float(np.mean(err ** 2)))
        if d == 2:
            fold_detail.append((b, pred, err, float(np.mean(err ** 2))))
    cv4[d] = float(np.mean(mses))

Se_c = [5.5150, 1.9129, 0.5736, 0.5734]
m2l_c = [19.727, 11.256, 1.621, 1.618]
AIC_c = [23.727, 17.256, 9.621, 11.618]
BIC_c = [23.886, 17.495, 9.938, 12.016]
Cp_c = [32.471, 9.343, 2.001, 4.000]
R2_c = [0, 0.5953, 0.8544, 0.8180]
LOO_c = [0.9004, 0.4922, 0.1512, 0.3013]
GCV_c = [0.9004, 0.4251, 0.1835, 0.2867]
tr_c = [0.6894, 0.2391, 0.0717, 0.0717]
cv4_c = [0.7258, 0.4417, 0.2507, 0.4694]
Err_c = [0.7252, 0.3108, 0.1792, 0.2150]
AIC0_c = [1.024, -5.447, -13.083, -11.085]
D = range(4)

# ================================================================ aic
T = "aic"
check(T, "S_e（0〜3次）", allclose([res[d]["Se"] for d in D], Se_c), [round(res[d]["Se"], 5) for d in D])
check(T, "-2ℓ = n log(2πSe/n)+n", allclose([res[d]["m2l"] for d in D], m2l_c, 5e-4), [round(res[d]["m2l"], 4) for d in D])
check(T, "AIC（定数込み k=p+1）", allclose([res[d]["AIC"] for d in D], AIC_c, 5e-4), [round(res[d]["AIC"], 4) for d in D])
check(T, "AIC 最小は2次、3次−2次≈2", int(np.argmin([res[d]["AIC"] for d in D])) == 2 and close(res[3]["AIC"] - res[2]["AIC"], 2.0, 5e-3),
      res[3]["AIC"] - res[2]["AIC"])
check(T, "3次で S_e が 0.0002 減る", close(res[2]["Se"] - res[3]["Se"], 0.0002, 5e-5), res[2]["Se"] - res[3]["Se"])
check(T, "定数を省いた形 1.024, −5.447, −13.083, −11.085", allclose([res[d]["AIC0"] for d in D], AIC0_c, 5e-4),
      [round(res[d]["AIC0"], 4) for d in D])
# 正規回帰の -2ℓ の式（sympy）
nn, Se_s, s2 = sp.symbols("n S_e sigma2", positive=True)
m2l_sym = nn * sp.log(2 * sp.pi * s2) + Se_s / s2
s2hat = sp.solve(sp.diff(m2l_sym, s2), s2)[0]
check(T, "σ̂²=Se/n を代入すると -2ℓ=n log(2πSe/n)+n（sympy）",
      s2hat == Se_s / nn and sp.simplify(sp.expand_log(m2l_sym.subs(s2, s2hat) - (nn * sp.log(2 * sp.pi * Se_s / nn) + nn), force=True)) == 0,
      s2hat)
# コイン
l0 = 100 * math.log(0.5)
l1 = 60 * math.log(0.6) + 40 * math.log(0.4)
check(T, "例2 ℓ0=−69.315, ℓ1=−67.301", close(l0, -69.315, 5e-4) and close(l1, -67.301, 5e-4), (l0, l1))
check(T, "例2 AIC0=138.629, −2ℓ1=134.602, AIC1=136.602", close(-2 * l0, 138.629, 5e-4) and close(-2 * l1, 134.602, 5e-4)
      and close(-2 * l1 + 2, 136.602, 5e-4), (-2 * l0, -2 * l1 + 2))
pchi = stats.chi2.sf(2, 1)
check(T, "P(χ²_1>2)≈15.7%", close(pchi, 0.157, 5e-4), pchi)
check(T, "log n > 2 ⇔ n ≥ 8", math.log(7) < 2 < math.log(8), (math.log(7), math.log(8)))
# 偏り b≈k の漸近（乱数）：正規 N(μ,σ²) 両方未知、k=2、n=200
nrep, nn_ = 40000, 200
Z = rng.normal(0, 1, (nrep, nn_))
mu_h = Z.mean(1)
s2_h = Z.var(1)
ell = -nn_ / 2 * (np.log(2 * np.pi * s2_h) + 1)
EZ = -0.5 * np.log(2 * np.pi * s2_h) - (1 + mu_h ** 2) / (2 * s2_h)
bias = float(np.mean(ell - nn_ * EZ))
check(T, "偏り b≈k（N(μ,σ²)、k=2、n=200、4万回、許容 0.1）", abs(bias - 2) < 0.1, bias)
c = charts(T, "bar-data")[0]["params"]
check(T, "bar-data の AIC 値", allclose(c["values"], AIC_c, 1e-9) and len(c["labels"]) == 4, c["values"])

# ================================================================ bic
T = "bic"
check(T, "log 8 = 2.0794", close(math.log(8), 2.0794), math.log(8))
check(T, "BIC（定数込み k=p+1）", allclose([res[d]["BIC"] for d in D], BIC_c, 5e-4), [round(res[d]["BIC"], 4) for d in D])
check(T, "2次 BIC = 1.6205+4×2.0794 = 9.938（本文修正後）", close(res[2]["m2l"], 1.6205) and close(1.6205 + 4 * 2.0794, 9.938, 5e-4), 1.6205 + 4 * 2.0794)
check(T, "BIC 最小は2次", int(np.argmin([res[d]["BIC"] for d in D])) == 2, None)
bic1 = -2 * l1 + math.log(100)
check(T, "例2 BIC0=138.629, BIC1=139.208", close(-2 * l0, 138.629, 5e-4) and close(bic1, 139.208, 5e-4), bic1)
G = 2 * (l1 - l0)
check(T, "例2 LR=4.027, log100=4.605", close(G, 4.027, 5e-4) and close(math.log(100), 4.605, 5e-4), (G, math.log(100)))
logm1 = special.betaln(61, 41)
lm1_num = math.log(integrate.quad(lambda t: t ** 60 * (1 - t) ** 40 * 1e30, 0, 1, points=[0.6], epsabs=0)[0]) - 30 * math.log(10)
check(T, "厳密な周辺尤度 log B(61,41) = −69.406（数値積分でも一致）", close(logm1, -69.406, 5e-4) and close(lm1_num, logm1, 1e-6),
      (logm1, lm1_num))
th = 0.6
Jhat = 1 / (th * (1 - th))
lap = l1 + math.log(1.0) + 0.5 * math.log(2 * math.pi) - 0.5 * math.log(100 * Jhat)
check(T, "ラプラス近似（O(1)項込み）= −69.398", close(lap, -69.398, 5e-4), lap)
BF = math.exp(logm1 - l0)
check(T, "厳密 BF10 = e^(−69.406+69.315) = 0.913", close(BF, 0.913, 5e-4) and close(math.exp(-69.406 + 69.315), 0.913, 5e-4), BF)
BFb = math.exp(-(bic1 - (-2 * l0)) / 2)
check(T, "BIC 近似 BF = e^{−(139.208−138.629)/2} = 0.749", close(BFb, 0.749, 5e-4), BFb)
# ラプラス近似の式（sympy）：∫exp(-n J t²/2)dt = sqrt(2π/(nJ))
t, J = sp.symbols("t J", positive=True)
check(T, "ガウス積分 ∫exp(−nJt²/2)dt=(2π)^{1/2}|nJ|^{-1/2}（sympy）",
      sp.simplify(sp.integrate(sp.exp(-nn * J * t ** 2 / 2), (t, -sp.oo, sp.oo)) - sp.sqrt(2 * sp.pi / (nn * J))) == 0, None)
c1, c2 = charts(T, "line-data")
s = series(c1)
lab_bic = [k for k in s if "BIC" in k][0]
lab_aic = [k for k in s if "AIC" in k][0]
check(T, "line-data 罰則 log n", allclose(s[lab_bic], [math.log(v) for v in c1["params"]["x"]], 5e-4) and all(v == 2 for v in s[lab_aic]),
      s[lab_bic])
check(T, "caption e²≈7.39", close(math.exp(2), 7.39, 5e-3), math.exp(2))
s = series(c2)
check(T, "line-data AIC・BIC（次数別）", allclose(s["AIC"], AIC_c, 1e-9) and allclose(s["BIC"], BIC_c, 1e-9), s)
# 一致性（乱数）：コイン H0 真（π=0.5）、n=10000 で BIC が M0 を選ぶ割合 > AIC
pick_aic, pick_bic = [], []
for nn_c in [10000]:
    k = rng.binomial(nn_c, 0.5, 20000)
    ph = k / nn_c
    with np.errstate(divide="ignore", invalid="ignore"):
        l1s = np.where((ph > 0) & (ph < 1), k * np.log(ph) + (nn_c - k) * np.log(1 - ph), 0)
    G2 = 2 * (l1s - nn_c * math.log(0.5))
    pick_aic = float(np.mean(G2 > 2))
    pick_bic = float(np.mean(G2 > math.log(nn_c)))
check(T, "一致性の傾向：H0 真、n=10000 で過大モデル選択率 AIC≈0.157・BIC≈0.009（許容 0.01）",
      abs(pick_aic - stats.chi2.sf(2, 1)) < 0.01 and abs(pick_bic - stats.chi2.sf(math.log(10000), 1)) < 0.01, (pick_aic, pick_bic))

# ================================================================ model-selection
T = "model-selection"
check(T, "表 S_e・AIC・BIC", allclose([res[d]["Se"] for d in D], Se_c) and allclose([res[d]["AIC"] for d in D], AIC_c, 5e-4)
      and allclose([res[d]["BIC"] for d in D], BIC_c, 5e-4), None)
check(T, "σ̂² = 0.5734/4 = 0.14335", close(sig2, 0.14335, 5e-6), sig2)
check(T, "表 Cp", allclose([res[d]["Cp"] for d in D], Cp_c, 5e-4), [round(res[d]["Cp"], 4) for d in D])
check(T, "表 調整済み R²", allclose([res[d]["R2adj"] for d in D], R2_c), [round(res[d]["R2adj"], 5) for d in D])
check(T, "表 LOOCV", allclose([res[d]["LOOCV"] for d in D], LOO_c), [round(res[d]["LOOCV"], 5) for d in D])
check(T, "S_T=5.515、0.5736/5=0.11472、5.515/7=0.78786", close(ST, 5.515, 5e-4) and close(0.5736 / 5, 0.11472, 5e-6)
      and close(5.515 / 7, 0.78786, 5e-6), ST)
check(T, "全基準が2次を選ぶ", all(int(np.argmin(v)) == 2 for v in [[res[d][k] for d in D] for k in ["AIC", "BIC", "Cp", "LOOCV"]])
      and int(np.argmax([res[d]["R2adj"] for d in D])) == 2, None)
s = series(charts(T, "line-data")[0])
check(T, "line-data AIC・BIC・Cp", allclose(s["AIC"], AIC_c, 1e-9) and allclose(s["BIC"], BIC_c, 1e-9)
      and allclose(s["マローズの Cp"], Cp_c, 1e-9), None)

# ================================================================ mallows-cp
T = "mallows-cp"
check(T, "S_e/σ̂² = 38.471, 13.343, 4.001, 4.000", allclose([res[d]["Se"] / sig2 for d in D], [38.471, 13.343, 4.001, 4.000], 5e-4),
      [round(res[d]["Se"] / sig2, 4) for d in D])
check(T, "Cp（0〜3次）", allclose([res[d]["Cp"] for d in D], Cp_c, 5e-4), None)
check(T, "2次 4.001−8+6=2.001", close(4.001 - 8 + 6, 2.001, 1e-9), None)
check(T, "最大モデルは Cp=p=4", close(res[3]["Cp"], 4.0, 1e-9), res[3]["Cp"])
check(T, "例2 σ̂²=52/26=2.0、C3=56/2−30+6=4.0", close(52 / 26, 2.0, 1e-12) and close(56 / 2 - 30 + 6, 4.0, 1e-12), None)
# E[Se]=B+(n-p)σ²、E||ŷ−μ||²=pσ²+B（乱数、真の関数を流用、1次モデル）
mu = 1 + 1.2 * x - 0.1 * x ** 2
X1 = design(x, 1)
H1 = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
B1 = float(np.sum(((np.eye(n) - H1) @ mu) ** 2))
Ys = mu + rng.normal(0, 0.4, (200000, n))
Yh = Ys @ H1.T
ESe = float(np.mean(np.sum((Ys - Yh) ** 2, 1)))
Efit = float(np.mean(np.sum((Yh - mu) ** 2, 1)))
check(T, "E[Se]=B+(n−p)σ²・E‖ŷ−μ‖²=pσ²+B（20万回、相対 1%）",
      abs(ESe / (B1 + 6 * 0.16) - 1) < 0.01 and abs(Efit / (B1 + 2 * 0.16) - 1) < 0.01, (ESe, B1 + 6 * 0.16, Efit, B1 + 0.32))
# AIC(σ²既知) = Cp + n + n log(2πσ²)
check(T, "σ²既知の AIC − Cp = n+n log(2πσ²)（数値、2次）",
      close((res[2]["Se"] / sig2 + 2 * 3 + n * math.log(2 * math.pi * sig2)) - res[2]["Cp"], n + n * math.log(2 * math.pi * sig2), 1e-9), None)
c = charts(T, "line-data")[0]
s = series(c)
check(T, "line-data Cp と Cp=p の線", c["params"]["x"] == [1, 2, 3, 4] and allclose(s["Cp"], Cp_c, 1e-9)
      and s["Cp = p の線"] == [1, 2, 3, 4], None)

# ================================================================ leave-one-out-cv
T = "leave-one-out-cv"
r2 = res[2]
check(T, "2次の係数 1.2679, 1.0964, −0.0893", allclose(r2["beta"], [1.2679, 1.0964, -0.0893]), r2["beta"])
check(T, "e_i", allclose(r2["e"], [-0.0750, 0.1964, 0.0464, -0.5250, 0.3821, 0.1679, -0.2679, 0.0750]), np.round(r2["e"], 5))
check(T, "h_ii", allclose(r2["h"], [0.7083, 0.2798, 0.2321, 0.2798, 0.2798, 0.2321, 0.2798, 0.7083]), np.round(r2["h"], 5))
check(T, "e_i/(1−h_ii)", allclose(r2["loo_f"], [-0.2571, 0.2727, 0.0605, -0.7289, 0.5306, 0.2186, -0.3719, 0.2571]),
      np.round(r2["loo_f"], 5))
check(T, "実際の一個抜きと公式が一致（全次数）", all(np.allclose(res[d]["loo_f"], res[d]["loo_a"], atol=1e-10) for d in D), None)
check(T, "Σh_ii = p = 3", close(r2["h"].sum(), 3, 1e-10), r2["h"].sum())
yhat4 = float((design(np.array([4.0]), 2) @ fit(np.delete(x, 3), np.delete(y, 3), 2))[0])
check(T, "ŷ(−4)=4.4289、3.7−4.4289=−0.7289", close(yhat4, 4.4289) and close(3.7 - 4.4289, -0.7289, 1e-9), yhat4)
check(T, "CV=0.1512、PRESS=1.2092", close(r2["LOOCV"], 0.1512) and close(r2["LOOCV"] * 8, 1.2092), (r2["LOOCV"], r2["LOOCV"] * 8))
check(T, "表 LOOCV・GCV", allclose([res[d]["LOOCV"] for d in D], LOO_c) and allclose([res[d]["GCV"] for d in D], GCV_c),
      [round(res[d]["GCV"], 5) for d in D])
check(T, "表 訓練誤差 Se/n", allclose([res[d]["Se"] / n for d in D], tr_c), [round(res[d]["Se"] / n, 5) for d in D])
r3 = res[3]
check(T, "3次 両端 h=0.8939、e_1=−0.0803、予測残差 −0.757",
      close(r3["h"][0], 0.8939) and close(r3["h"][-1], 0.8939) and close(r3["e"][0], -0.0803) and close(r3["loo_f"][0], -0.757, 5e-4)
      and close(-0.0803 / (1 - 0.8939), -0.757, 5e-4), (r3["h"][0], r3["e"][0], r3["loo_f"][0]))
check(T, "3次 LOOCV は2次の約2倍", close(r3["LOOCV"] / r2["LOOCV"], 2.0, 0.05), r3["LOOCV"] / r2["LOOCV"])
check(T, "0次 h=1/8、LOOCV=GCV", np.allclose(res[0]["h"], 1 / 8) and close(res[0]["LOOCV"], res[0]["GCV"], 1e-12), None)
# シャーマン・モリソンによる β(−i)（数値）
X2 = design(x, 2)
A_inv = np.linalg.inv(X2.T @ X2)
ok = all(np.allclose(fit(np.delete(x, i), np.delete(y, i), 2), r2["beta"] - A_inv @ X2[i] * r2["e"][i] / (1 - r2["h"][i]), atol=1e-10)
         for i in range(n))
check(T, "β(−i)=β̂−A⁻¹x_i e_i/(1−h_ii)（全 i）", ok, None)
s = series(charts(T, "line-data")[0])
check(T, "line-data 訓練誤差・LOOCV・GCV", allclose(s["訓練誤差 Se/n"], tr_c, 1e-9) and allclose(s["LOOCV"], LOO_c, 1e-9)
      and allclose(s["GCV"], GCV_c, 1e-9), None)

# ================================================================ cross-validation
T = "cross-validation"
coef_c = [[2.0898, 0.6827, -0.0490], [1.2020, 1.0837, -0.0869], [1.2471, 1.0922, -0.0862], [0.8133, 1.4898, -0.1398]]
pred_c = [[2.7235, 4.2786], [3.0217, 4.5749], [3.7482, 4.6706], [4.5357, 3.7847]]
err_c = [[-0.5235, 0.6214], [0.2783, 0.2251], [0.0518, -0.3706], [-0.8357, 0.6153]]
mse_c = [0.3301, 0.0641, 0.0700, 0.5385]
for j, (b, pr, er, ms) in enumerate(fold_detail):
    check(T, f"例1 隠した x={folds[j]} の係数・予測値・誤差・平均",
          allclose(b, coef_c[j]) and allclose(pr, pred_c[j]) and allclose(er, err_c[j]) and close(ms, mse_c[j]),
          (np.round(b, 5), np.round(pr, 5), np.round(er, 5), round(ms, 5)))
check(T, "CV(4) 2次=0.2507（=(0.3301+0.0641+0.0700+0.5385)/4）", close(cv4[2], 0.2507) and close(sum(mse_c) / 4, 0.2507, 5e-5), cv4[2])
check(T, "例2 CV(4)（0〜3次）", allclose([cv4[d] for d in D], cv4_c), [round(cv4[d], 5) for d in D])
check(T, "例2 訓練誤差・LOOCV", allclose([res[d]["Se"] / n for d in D], tr_c) and allclose([res[d]["LOOCV"] for d in D], LOO_c), None)
check(T, "CV(4)・LOOCV とも2次で最小、訓練誤差 < CV", int(np.argmin(list(cv4.values()))) == 2 and int(np.argmin(LOO_c)) == 2
      and all(tr_c[d] < cv4_c[d] and tr_c[d] < LOO_c[d] for d in D), None)
s = series(charts(T, "line-data")[0])
check(T, "line-data 訓練誤差・4分割・LOOCV", allclose(s["訓練誤差 Se/n"], tr_c, 1e-9) and allclose(s["4分割交差検証"], cv4_c, 1e-9)
      and allclose(s["一個抜き交差検証"], LOO_c, 1e-9), None)

# ================================================================ information-criteria-comparison
T = "information-criteria-comparison"
aics = np.array([res[d]["AIC"] for d in D])
dl = aics - aics.min()
ex = np.exp(-dl / 2)
w = ex / ex.sum()
check(T, "Δ = 14.107, 7.636, 0, 1.998", allclose(dl, [14.107, 7.636, 0, 1.998], 5e-4), np.round(dl, 4))
check(T, "exp(−Δ/2) = 0.0009, 0.0220, 1, 0.3683", allclose(ex, [0.0009, 0.0220, 1, 0.3683]), np.round(ex, 5))
check(T, "合計 1.3911、2次の重み 0.7189", close(ex.sum(), 1.3911) and close(1 / 1.3911, 0.7189), ex.sum())
check(T, "赤池ウェイト 0.0006, 0.0158, 0.7189, 0.2647", allclose(w, [0.0006, 0.0158, 0.7189, 0.2647]), np.round(w, 5))
check(T, "証拠比 2次/3次 ≈ 2.72（本文修正後）", close(w[2] / w[3], 2.72, 5e-3) and close(0.7189 / 0.2647, 2.72, 5e-3), (w[2] / w[3], 0.7189 / 0.2647))
bics = np.array([res[d]["BIC"] for d in D])
db = bics - bics.min()
wb = np.exp(-db / 2) / np.exp(-db / 2).sum()
check(T, "ΔBIC = 13.948, 7.556, 0, 2.077", allclose(db, [13.948, 7.556, 0, 2.077], 5e-4), np.round(db, 4))
check(T, "BIC の重み 0.0007, 0.0166, 0.7258, 0.2569", allclose(wb, [0.0007, 0.0166, 0.7258, 0.2569]), np.round(wb, 5))
check(T, "e^1≈2.72、e^0.5≈1.65", close(math.e, 2.72, 5e-3) and close(math.exp(0.5), 1.65, 5e-3), None)
check(T, "例2 AIC 242.8・239.8、Δ=3.0、e^1.5=4.48、LR=5.0",
      close(-2 * -120.4 + 2, 242.8, 1e-9) and close(-2 * -117.9 + 4, 239.8, 1e-9) and close(math.exp(1.5), 4.48, 5e-3)
      and close(2 * (-117.9 + 120.4), 5.0, 1e-9), math.exp(1.5))
check(T, "AIC小−AIC大 = G−2（パラメータ1個差）", close((-2 * l0 + 0) - (-2 * l1 + 2), G - 2, 1e-12), None)
c = charts(T, "bar-data")[0]["params"]
check(T, "bar-data 赤池ウェイト", allclose(c["values"], w), c["values"])

# ================================================================ bias-variance-tradeoff
T = "bias-variance-tradeoff"
sigma2 = 0.16
degs = [0, 1, 2, 3, 4, 5, 6, 7]
Bn, Var, EPE, TR = [], [], [], []
for d in degs:
    Xd = design(x, d)
    Hd = Xd @ np.linalg.pinv(Xd)
    B = float(np.sum(((np.eye(n) - Hd) @ mu) ** 2))
    p = d + 1
    Bn.append(B / n)
    Var.append(sigma2 * p / n)
    EPE.append(sigma2 + B / n + sigma2 * p / n)
    TR.append(B / n + sigma2 * (1 - p / n))
check(T, "例2 バイアス²（0,1,2次）= 0.6825, 0.21, 0", allclose(Bn[:3], [0.6825, 0.21, 0], 5e-5) and max(Bn[2:]) < 1e-12, np.round(Bn, 6))
check(T, "例2 バリアンス σ²p/n", allclose(Var, [0.02 * (d + 1) for d in degs], 1e-12), Var)
check(T, "例2 期待予測誤差 0.8625, 0.41, 0.22, 0.24, …, 0.32", allclose(EPE, [0.8625, 0.41, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32], 5e-5),
      np.round(EPE, 5))
s = series(charts(T, "line-data")[0])
ks = list(s)
check(T, "line-data バイアス²・バリアンス・期待予測誤差", allclose(s[ks[0]], Bn, 5e-5) and allclose(s[ks[1]], Var, 1e-9)
      and allclose(s[ks[2]], EPE, 5e-5), None)
# 分解の乱数確認（1次、x0=8 の点）
Ysim = mu + rng.normal(0, 0.4, (200000, n))
X1 = design(x, 1)
P1 = np.linalg.pinv(X1)
fh = (Ysim @ P1.T) @ np.array([1, 8.0])
y0 = mu[-1] + rng.normal(0, 0.4, 200000)
lhs = float(np.mean((y0 - fh) ** 2))
rhs = sigma2 + (fh.mean() - mu[-1]) ** 2 + fh.var()
check(T, "分解 E(Y0−f̂)²=σ²+バイアス²+バリアンス（x0=8、1次、20万回、相対 1%）", abs(lhs / rhs - 1) < 0.01, (lhs, rhs))
# 縮小推定量
cc, m_, s2_, nn2 = sp.symbols("c mu sigma2 n", positive=True)
mse = (1 - cc) ** 2 * m_ ** 2 + cc ** 2 * s2_ / nn2
cstar = sp.solve(sp.diff(mse, cc), cc)[0]
check(T, "例1 c*=μ²/(μ²+σ²/n)（sympy）", sp.simplify(cstar - m_ ** 2 / (m_ ** 2 + s2_ / nn2)) == 0, cstar)
sub = {m_: 1, s2_: 4, nn2: 4}
check(T, "例1 MSE(1)=1、MSE(0.5)=0.5、c*=0.5", mse.subs(sub).subs(cc, 1) == 1 and mse.subs(sub).subs(cc, sp.Rational(1, 2)) == sp.Rational(1, 2)
      and cstar.subs(sub) == sp.Rational(1, 2), None)
Xs = rng.normal(1, 2, (400000, 4)).mean(1)
check(T, "例1 MSE(0.5) 乱数（40万回、許容 0.01）", abs(np.mean((0.5 * Xs - 1) ** 2) - 0.5) < 0.01, float(np.mean((0.5 * Xs - 1) ** 2)))

# ================================================================ overfitting
T = "overfitting"
ERR = [EPE[d] for d in degs]
check(T, "例1 訓練誤差の期待値 0.8225, 0.33, 0.10, 0.08, …, 0", allclose(TR, [0.8225, 0.33, 0.10, 0.08, 0.06, 0.04, 0.02, 0], 5e-5),
      np.round(TR, 5))
check(T, "例1 テスト誤差の期待値", allclose(ERR, [0.8625, 0.41, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32], 5e-5), None)
check(T, "差 = 2σ²p/n", allclose(np.array(ERR) - np.array(TR), [2 * sigma2 * (d + 1) / n for d in degs], 1e-10), None)
check(T, "7次 0.32/0.22≈1.5倍", close(0.32 / 0.22, 1.5, 0.05), 0.32 / 0.22)
check(T, "例2 S_e 0.5736・0.5734、LOOCV 0.1512・0.3013、てこ比 0.894",
      close(res[2]["Se"], 0.5736) and close(res[3]["Se"], 0.5734) and close(res[3]["h"][0], 0.894, 5e-4), None)
# 訓練誤差・テスト誤差の乱数確認（3次）
X3 = design(x, 3)
H3 = X3 @ np.linalg.pinv(X3)
Ys = mu + rng.normal(0, 0.4, (200000, n))
Y0 = mu + rng.normal(0, 0.4, (200000, n))
fh3 = Ys @ H3.T
check(T, "3次 訓練 0.08・テスト 0.24（20万回、許容 0.002）",
      abs(np.mean((Ys - fh3) ** 2) - 0.08) < 0.002 and abs(np.mean((Y0 - fh3) ** 2) - 0.24) < 0.002,
      (float(np.mean((Ys - fh3) ** 2)), float(np.mean((Y0 - fh3) ** 2))))
s = series(charts(T, "line-data")[0])
ks = list(s)
check(T, "line-data 訓練・テスト", allclose(s[ks[0]], TR, 5e-5) and allclose(s[ks[1]], ERR, 5e-5), None)

# ================================================================ generalization-error
T = "generalization-error"
check(T, "例1 0.16(1−3/8)=0.10、0.16(1+3/8)=0.22、2×0.16×3/8=0.12",
      close(0.16 * (1 - 3 / 8), 0.10, 1e-12) and close(0.16 * (1 + 3 / 8), 0.22, 1e-12) and close(2 * 0.16 * 3 / 8, 0.12, 1e-12), None)
check(T, "例2 2次 0.0717+0.1075=0.1792", close(res[2]["Se"] / 8, 0.0717) and close(2 * sig2 * 3 / 8, 0.1075, 5e-5)
      and close(res[2]["Err"], 0.1792), res[2]["Err"])
check(T, "例2 表 補正後の推定", allclose([res[d]["Err"] for d in D], Err_c), [round(res[d]["Err"], 5) for d in D])
check(T, "補正後・LOOCV とも2次で最小", int(np.argmin(Err_c)) == 2 and int(np.argmin(LOO_c)) == 2, None)
check(T, "Err_hat×n/σ̂² − n = Cp", allclose([res[d]["Err"] * n / sig2 - n for d in D], [res[d]["Cp"] for d in D], 1e-10), None)
# 楽観性 ΣCov(ŷ_i,y_i)=pσ²（2次、乱数）
X2 = design(x, 2)
H2 = X2 @ np.linalg.pinv(X2)
Ys = mu + rng.normal(0, 0.4, (200000, n))
Yh = Ys @ H2.T
cov_sum = float(np.sum(np.mean((Yh - Yh.mean(0)) * (Ys - Ys.mean(0)), 0)))
check(T, "ΣCov(ŷ_i,y_i)=pσ²=0.48（2次、20万回、相対 1%）", abs(cov_sum / 0.48 - 1) < 0.01, cov_sum)
s = series(charts(T, "line-data")[0])
ks = list(s)
check(T, "line-data 訓練・補正後・LOOCV", allclose(s[ks[0]], tr_c, 1e-9) and allclose(s[ks[1]], Err_c, 1e-9)
      and allclose(s[ks[2]], LOO_c, 1e-9), s[ks[1]])

# ================================================================ expected-log-likelihood
T = "expected-log-likelihood"
ns = sp.symbols("n", positive=True)
train = -sp.log(2 * sp.pi) / 2 - (ns - 1) / (2 * ns)
test = -sp.log(2 * sp.pi) / 2 - (ns + 1) / (2 * ns)
check(T, "差 = 1/n（sympy）", sp.simplify(train - test - 1 / ns) == 0, sp.simplify(train - test))
check(T, "−½log(2π) = −0.9189", close(-0.5 * L2PI, -0.9189), -0.5 * L2PI)
check(T, "n=10：−1.3689 と −1.4689", close(float(train.subs(ns, 10)), -1.3689) and close(float(test.subs(ns, 10)), -1.4689), None)
c = charts(T, "line-data")[0]
s = series(c)
ks = list(s)
xs = c["params"]["x"]
check(T, "line-data 2本の厳密値", allclose(s[ks[0]], [float(train.subs(ns, v)) for v in xs], 5e-4)
      and allclose(s[ks[1]], [float(test.subs(ns, v)) for v in xs], 5e-4), None)
# 乱数：n=10
Zs = rng.normal(0, 1, (400000, 10))
xb = Zs.mean(1)
tr_sim = np.mean(-0.5 * L2PI - 0.5 * np.mean((Zs - xb[:, None]) ** 2, 1))
te_sim = np.mean(-0.5 * L2PI - (1 + xb ** 2) / 2)
check(T, "n=10 乱数（40万回、許容 0.003）", abs(tr_sim + 1.3689) < 0.003 and abs(te_sim + 1.4689) < 0.003, (tr_sim, te_sim))
# E_Z[(Z−x̄)²]=1+x̄²（sympy）
zz, a = sp.symbols("z a", real=True)
phi = sp.exp(-zz ** 2 / 2) / sp.sqrt(2 * sp.pi)
check(T, "E_Z[(Z−a)²]=1+a²（sympy）", sp.simplify(sp.integrate((zz - a) ** 2 * phi, (zz, -sp.oo, sp.oo)) - (1 + a ** 2)) == 0, None)

# ================================================================ kullback-leibler-divergence
T = "kullback-leibler-divergence"
g = np.full(6, 1 / 6)
f = np.array([0.1] * 5 + [0.5])
t1 = 5 / 6 * math.log((1 / 6) / 0.1)
t2 = 1 / 6 * math.log((1 / 6) / 0.5)
check(T, "例1 0.4257 − 0.1831 = 0.2426", close(t1, 0.4257) and close(t2, -0.1831) and close(float(np.sum(g * np.log(g / f))), 0.2426),
      (t1, t2, float(np.sum(g * np.log(g / f)))))
r1 = 5 * 0.1 * math.log(0.1 / (1 / 6))
r2_ = 0.5 * math.log(0.5 / (1 / 6))
check(T, "例1 逆向き −0.2554 + 0.5493 = 0.2939", close(r1, -0.2554) and close(r2_, 0.5493) and close(float(np.sum(f * np.log(f / g))), 0.2939),
      (r1, r2_, float(np.sum(f * np.log(f / g)))))


def kl_norm(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def kl_quad(m1, s1, m2, s2):
    fn = lambda t: stats.norm.pdf(t, m1, s1) * (stats.norm.logpdf(t, m1, s1) - stats.norm.logpdf(t, m2, s2))
    return integrate.quad(fn, -np.inf, np.inf)[0]


check(T, "例2 D(N(0,1)‖N(1,1))=0.5", close(kl_norm(0, 1, 1, 1), 0.5, 1e-12) and close(kl_quad(0, 1, 1, 1), 0.5, 1e-7), None)
check(T, "例2 D(N(0,1)‖N(0,4))=0.3181", close(kl_norm(0, 1, 0, 2), 0.3181) and close(kl_quad(0, 1, 0, 2), kl_norm(0, 1, 0, 2), 1e-7),
      kl_norm(0, 1, 0, 2))
check(T, "例2 D(N(0,4)‖N(0,1))=0.8069", close(kl_norm(0, 2, 0, 1), 0.8069) and close(kl_quad(0, 2, 0, 1), kl_norm(0, 2, 0, 1), 1e-7),
      kl_norm(0, 2, 0, 1))
# 正規の公式（sympy で積分）
m1, m2 = sp.symbols("mu1 mu2", real=True)
sg1, sg2 = sp.symbols("sigma1 sigma2", positive=True)
X_ = sp.symbols("X", real=True)
dens = sp.exp(-(X_ - m1) ** 2 / (2 * sg1 ** 2)) / (sp.sqrt(2 * sp.pi) * sg1)
logratio = sp.log(sg2 / sg1) - (X_ - m1) ** 2 / (2 * sg1 ** 2) + (X_ - m2) ** 2 / (2 * sg2 ** 2)
kl_sym = sp.integrate(sp.expand(dens * logratio), (X_, -sp.oo, sp.oo))
check(T, "正規分布どうしの公式（sympy 積分）",
      sp.simplify(kl_sym - (sp.log(sg2 / sg1) + (sg1 ** 2 + (m1 - m2) ** 2) / (2 * sg2 ** 2) - sp.Rational(1, 2))) == 0, None)
check(T, "caption D(N(0,1)‖N(μ,σ²)) = log σ + (1+μ²)/(2σ²) − 1/2", all(close(kl_norm(0, 1, mm, ss), math.log(ss) + (1 + mm ** 2) / (2 * ss ** 2) - 0.5, 1e-12)
                                                                       for mm in [-1, 0, 2] for ss in [0.5, 1, 3]), None)
b1 = 0.5 * math.log(0.5 / 0.3) + 0.5 * math.log(0.5 / 0.7)
b2 = 0.3 * math.log(0.3 / 0.5) + 0.7 * math.log(0.7 / 0.5)
check(T, "例3 ベルヌーイ 0.0872 と 0.0823", close(b1, 0.0872) and close(b2, 0.0823), (b1, b2))
check(T, "σ を小さくする方が速く増える（σ=0.5 vs 2）", kl_norm(0, 1, 0, 0.5) > kl_norm(0, 1, 0, 2), (kl_norm(0, 1, 0, 0.5), kl_norm(0, 1, 0, 2)))
# ランダムな離散分布で非負性
P = rng.dirichlet(np.ones(6), 10000)
Q = rng.dirichlet(np.ones(6), 10000)
check(T, "非負性（ランダムな離散分布 1万組）", bool(np.all(np.sum(P * np.log(P / Q), 1) >= 0)), None)
ch = front(T)["charts"][0]
check(T, "pdf の真の分布 N(0,1)", ch["compare"][0]["params"] == {"mu": 0, "sigma": 1}, ch["compare"][0]["params"])

# ================================================================ 重要度Cの用語
xc_ = np.arange(1, 9, dtype=float)
yc_ = np.array([2.2, 3.3, 3.8, 3.7, 4.9, 4.8, 4.3, 4.4])

# ---------------------------------------------------------------- training-test-split
T = "training-test-split"
te_ = np.isin(xc_, [3, 6])
tr_ = ~te_
trm_, tem_, coef_, pred_ = [], [], [], []
for d_ in range(4):
    b_ = np.polyfit(xc_[tr_], yc_[tr_], d_)
    coef_.append(b_[::-1])
    pred_.append(np.polyval(b_, xc_[te_]))
    trm_.append(np.mean((yc_[tr_] - np.polyval(b_, xc_[tr_])) ** 2))
    tem_.append(np.mean((yc_[te_] - np.polyval(b_, xc_[te_])) ** 2))
check(T, "2次式（訓練6点）1.3373+1.0380x−0.0833x²", allclose(coef_[2], [1.3373, 1.0380, -0.0833]), np.round(coef_[2], 4).tolist())
check(T, "テストの予測値 3.7013, 4.5653", allclose(pred_[2], [3.7013, 4.5653]), np.round(pred_[2], 4).tolist())
check(T, "2次のテスト MSE 0.0324", close(tem_[2], 0.0324), tem_[2])
check(T, "表 訓練 MSE 0.7733/0.2549/0.0883/0.0877", allclose(trm_, [0.7733, 0.2549, 0.0883, 0.0877]), np.round(trm_, 4).tolist())
check(T, "表 テスト MSE 0.5000/0.2546/0.0324/0.0391", allclose(tem_, [0.5, 0.2546, 0.0324, 0.0391]), np.round(tem_, 4).tolist())
check(T, "訓練誤差は単調減少で3次が最小、テスト誤差は2次が最小", bool(np.all(np.diff(trm_) < 0)) and int(np.argmin(tem_)) == 2, None)
c_ = charts(T, "line-data")[0]
sr_ = series(c_)
check(T, "chart line-data x=0..3・2系列＝表の値",
      c_["params"]["x"] == [0, 1, 2, 3] and allclose(sr_["訓練データの平均二乗誤差"], np.round(trm_, 4), 1e-9) and allclose(sr_["テストデータの平均二乗誤差"], np.round(tem_, 4), 1e-9), sr_)
# 乱数：テスト誤差は固定した f̂ の汎化誤差の不偏推定、訓練誤差は過小
f2_ = np.polyfit(xc_[tr_], yc_[tr_], 2)
mu_true = np.polyval(np.polyfit(xc_, yc_, 2), xc_)
sig_ = 0.3
gen_ = np.mean((mu_true - np.polyval(f2_, xc_)) ** 2) + sig_ ** 2
tests_ = []
trains_, gens_ = [], []
for _ in range(20000):
    ynew = mu_true + rng.normal(0, sig_, 8)
    tests_.append(np.mean((ynew[te_] - np.polyval(f2_, xc_[te_])) ** 2))
    bb = np.polyfit(xc_[tr_], ynew[tr_], 2)
    trains_.append(np.mean((ynew[tr_] - np.polyval(bb, xc_[tr_])) ** 2))
    gens_.append(np.mean((mu_true[tr_] - np.polyval(bb, xc_[tr_])) ** 2) + sig_ ** 2)
gen_te = np.mean((mu_true[te_] - np.polyval(f2_, xc_[te_])) ** 2) + sig_ ** 2
check(T, "乱数2万回：固定 f̂ のテスト MSE の平均 = その f̂ の（テスト点での）汎化誤差（3SE）",
      abs(np.mean(tests_) - gen_te) < 3 * np.std(tests_) / math.sqrt(20000), (round(np.mean(tests_), 4), round(gen_te, 4)))
check(T, "乱数2万回：訓練誤差の平均 < 汎化誤差の平均（楽観的な偏り、理論 σ²(n−p)/n=0.045 vs 0.135）",
      np.mean(trains_) < np.mean(gens_) and abs(np.mean(trains_) - sig_ ** 2 * 3 / 6) < 0.002 and abs(np.mean(gens_) - sig_ ** 2 * 9 / 6) < 0.003,
      (round(np.mean(trains_), 4), round(np.mean(gens_), 4)))

# ---------------------------------------------------------------- aicc
T = "aicc"
aic_c, cor_c = [], []
for d_ in range(4):
    b_ = np.polyfit(xc_, yc_, d_)
    Se_ = np.sum((yc_ - np.polyval(b_, xc_)) ** 2)
    k_ = d_ + 2
    aic_c.append(8 * math.log(Se_ / 8) + 8 * (L2PI + 1) + 2 * k_)
    cor_c.append(2 * k_ * (k_ + 1) / (8 - k_ - 1))
aicc_c = np.array(aic_c) + np.array(cor_c)
check(T, "AIC 23.727/17.256/9.621/11.618（aic.md の値と一致）", allclose(aic_c, [23.727, 17.256, 9.621, 11.618], 5e-4)
      and charts("aic", "bar-data")[0]["params"]["values"] == [23.727, 17.256, 9.621, 11.618], np.round(aic_c, 3).tolist())
check(T, "補正項 2.400/6.000/13.333/30.000（2次は 40/3）", allclose(cor_c, [2.4, 6, 13.333, 30], 5e-4) and close(40 / 3, 13.333, 5e-4), np.round(cor_c, 3).tolist())
check(T, "AICc 26.127/23.256/22.954/41.618", allclose(aicc_c, [26.127, 23.256, 22.954, 41.618], 5e-4), np.round(aicc_c, 3).tolist())
check(T, "AICc でも2次最小、1次との差 0.302、AIC の差 7.635",
      int(np.argmin(aicc_c)) == 2 and close(round(aicc_c[1], 3) - round(aicc_c[2], 3), 0.302, 1e-9) and close(round(aic_c[1], 3) - round(aic_c[2], 3), 7.635, 1e-9), None)
kk_, nn_ = sp.symbols("k n", positive=True)
check(T, "AIC+2k(k+1)/(n−k−1) = −2ℓ+2kn/(n−k−1)（sympy）", sp.simplify(2 * kk_ + 2 * kk_ * (kk_ + 1) / (nn_ - kk_ - 1) - 2 * kk_ * nn_ / (nn_ - kk_ - 1)) == 0, None)
check(T, "n→∞ で補正項→0（sympy）", sp.limit(2 * kk_ * (kk_ + 1) / (nn_ - kk_ - 1), nn_, sp.oo) == 0, None)
c_ = charts(T, "line-data")[0]
sr_ = series(c_)
check(T, "chart line-data＝表の AIC・AICc", c_["params"]["x"] == [0, 1, 2, 3] and allclose(sr_["AIC"], np.round(aic_c, 3), 1e-9) and allclose(sr_["AICc"], np.round(aicc_c, 3), 1e-9), sr_)
# 正規線形回帰で、AICc が期待 KL の偏り（−2倍）を正確に与えることを乱数で確認：
# E[−2ℓ(θ̂) を新データで評価] − E[−2ℓ(θ̂)] = 2kn/(n−k−1)（k=p+1）
Xd_ = np.vander(xc_, 3, increasing=True)
mu_t = Xd_ @ np.array([1.3, 1.0, -0.08])
gap_ = []
for _ in range(40000):
    yo = mu_t + rng.normal(0, 0.3, 8)
    bh = np.linalg.lstsq(Xd_, yo, rcond=None)[0]
    s2h = np.mean((yo - Xd_ @ bh) ** 2)
    in_ = 8 * math.log(2 * math.pi * s2h) + 8
    out_ = 8 * math.log(2 * math.pi * s2h) + (8 * 0.09 + np.sum((mu_t - Xd_ @ bh) ** 2)) / s2h
    gap_.append(out_ - in_)
check(T, "乱数4万回（2次, n=8, k=4）：偏り×2 の平均≈2kn/(n−k−1)=21.333（AIC の 8 ではない、相対 3%）",
      abs(np.mean(gap_) / (2 * 4 * 8 / 3) - 1) < 0.03, round(float(np.mean(gap_)), 3))

# ---------------------------------------------------------------- tic-ebic
T = "tic-ebic"
zt_ = np.array([-3.1, -0.8, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.9, 2.7])
muh_ = zt_.mean()
s2h_ = np.mean((zt_ - muh_) ** 2)
m2l_ = len(zt_) * (L2PI + math.log(s2h_) + 1)
check(T, "μ̂=0、σ̂²=1.89", close(muh_, 0, 1e-12) and close(s2h_, 1.89, 1e-12), (muh_, s2h_))
check(T, "−2ℓ=34.745、AIC=38.745（scipy の対数密度の和でも）",
      close(m2l_, 34.745, 5e-4) and close(m2l_ + 4, 38.745, 5e-4) and close(-2 * stats.norm(0, math.sqrt(s2h_)).logpdf(zt_).sum(), m2l_, 1e-9), m2l_)
kap_ = np.mean((zt_ - muh_) ** 4) / s2h_ ** 2
sc_ = np.array([(zt_ - muh_) / s2h_, -1 / (2 * s2h_) + (zt_ - muh_) ** 2 / (2 * s2h_ ** 2)])
I_ = sc_ @ sc_.T / len(zt_)
J_ = np.array([[1 / s2h_, 0], [0, 1 / (2 * s2h_ ** 2)]])
trIJ_ = float(np.trace(I_ @ np.linalg.inv(J_)))
check(T, "κ̂=4.106、tr=1+(κ̂−1)/2=2.553", close(kap_, 4.106, 5e-4) and close(1 + (kap_ - 1) / 2, 2.553, 5e-4), kap_)
check(T, "スコアから直接作った Î, Ĵ で tr(ÎĴ⁻¹)=1+(κ̂−1)/2（μ̂ 分が1）",
      close(trIJ_, 1 + (kap_ - 1) / 2, 1e-9) and close(I_[0, 0] / J_[0, 0], 1, 1e-9), trIJ_)
check(T, "TIC=39.850、罰則 5.106（正規なら 4）", close(m2l_ + 2 * trIJ_, 39.850, 5e-4) and close(2 * trIJ_, 5.106, 5e-4), m2l_ + 2 * trIJ_)
# 記号計算：正規モデルのスコア・ヘッセ行列から tr(IJ⁻¹)=1+(κ−1)/2（3次モーメントは非対角で寄与しない）
zz, mm = sp.symbols("z mu", real=True)
vv = sp.symbols("v", positive=True)
lf = -sp.log(2 * sp.pi * vv) / 2 - (zz - mm) ** 2 / (2 * vv)
gs = [sp.diff(lf, p) for p in (mm, vv)]
Hs = sp.Matrix(2, 2, lambda i, j: -sp.diff(lf, (mm, vv)[i], (mm, vv)[j]))
m3, m4 = sp.symbols("m3 m4", real=True)
# E[(z−μ)]=0, E[(z−μ)²]=v, E[(z−μ)^3]=m3, E[(z−μ)^4]=m4 を代入
def Eg(expr):
    e = sp.expand(expr.subs(zz, mm + sp.Symbol("w")))
    pol = sp.Poly(e, sp.Symbol("w"))
    mom = {0: 1, 1: 0, 2: vv, 3: m3, 4: m4}
    return sp.simplify(sum(cf * mom[mon[0]] for mon, cf in zip(pol.monoms(), pol.coeffs())))
Im = sp.Matrix(2, 2, lambda i, j: Eg(gs[i] * gs[j]))
Jm = sp.Matrix(2, 2, lambda i, j: Eg(Hs[i, j]))
trs = sp.simplify((Im * Jm.inv()).trace())
check(T, "tr(IJ⁻¹)=1+(κ−1)/2, κ=m4/v²（sympy）", sp.simplify(trs - (1 + (m4 / vv ** 2 - 1) / 2)) == 0, trs)
check(T, "正規なら κ=3 で tr=2=k（I=J）", sp.simplify(trs.subs(m4, 3 * vv ** 2) - 2) == 0 and sp.simplify((Im - Jm).subs({m4: 3 * vv ** 2, m3: 0})) == sp.zeros(2, 2), None)

# ================================================================ 集計
ok = sum(results)
ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
