"""第17章 検算スクリプト（回帰診断）。
厳密計算は sympy / fractions、数値は numpy・scipy.stats。主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5）。
charts の params は各 md の front matter を yaml で読み、本文の値と照合する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記（概ね標準誤差の4倍以上）。
"""
from fractions import Fraction as F
from math import sqrt
from pathlib import Path
import numpy as np
import sympy as sp
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float)
    return len(xs) == len(claims) and bool(np.all(np.abs(xs - np.asarray(claims, float)) <= tol))


def r4(v):
    return [round(float(x), 4) for x in v]


def charts(term):
    txt = (ROOT / "content" / "ch17" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1]).get("charts", [])


def fit(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return b, y - X @ b


def diag_all(x, y):
    """単回帰の診断量を定義どおり（1点除外の当てはめ直しを含む）計算"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    b, e = fit(X, y)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p = 1
    s2 = e @ e / (n - p - 1)
    r = e / np.sqrt(s2 * (1 - h))
    s2i, ti, Di, dff, bdel = [], [], [], [], []
    for i in range(n):
        m = np.arange(n) != i
        bi, ei = fit(X[m], y[m])
        s2_i = ei @ ei / (n - 1 - p - 1)
        s2i.append(s2_i)
        ti.append(e[i] / np.sqrt(s2_i * (1 - h[i])))
        yh_all = X @ b
        yh_del = X @ bi
        Di.append(np.sum((yh_all - yh_del) ** 2) / ((p + 1) * s2))
        dff.append((yh_all[i] - yh_del[i]) / np.sqrt(s2_i * h[i]))
        bdel.append(bi)
    return dict(X=X, b=b, e=e, H=H, h=h, s2=s2, r=r, s2i=np.array(s2i), t=np.array(ti),
                D=np.array(Di), dffits=np.array(dff), bdel=bdel, n=n, p=p)


xs = np.array([1, 2, 3, 4, 5, 6, 7, 12], float)
ys = np.array([3, 5, 6, 9, 10, 13, 14, 20], float)
d = diag_all(xs, ys)

# ============ residual ============
T = "residual"
xf = [F(int(v)) for v in xs]
yf = [F(int(v)) for v in ys]
xb = sum(xf) / 8
yb = sum(yf) / 8
Sxx = sum((a - xb) ** 2 for a in xf)
Sxy = sum((a - xb) * (c - yb) for a, c in zip(xf, yf))
b1 = Sxy / Sxx
b0 = yb - b1 * xb
check(T, "例1 x̄=5, ȳ=10, Sxx=84, Sxy=133", (xb, yb, Sxx, Sxy) == (5, 10, 84, 133), (xb, yb, Sxx, Sxy))
check(T, "例1 β1=133/84=1.5833, β0=2.0833", close(b1, 1.5833) and close(b0, 2.0833), (float(b1), float(b0)))
yhat = [b0 + b1 * a for a in xf]
ef = [c - yh for c, yh in zip(yf, yhat)]
check(T, "例1 当てはめ値の表", allclose([float(v) for v in yhat], [3.6667, 5.25, 6.8333, 8.4167, 10, 11.5833, 13.1667, 21.0833]), r4(yhat))
check(T, "例1 残差の表", allclose([float(v) for v in ef], [-0.6667, -0.25, -0.8333, 0.5833, 0, 1.4167, 0.8333, -1.0833]), r4(ef))
check(T, "例1 Σe=0, Σx e=0（厳密）", sum(ef) == 0 and sum(a * c for a, c in zip(xf, ef)) == 0, (sum(ef), sum(a * c for a, c in zip(xf, ef))))
SSE = sum(v * v for v in ef)
check(T, "例1 Σe²=5.4167(=65/12), s²=0.9028, s=0.9501", close(SSE, 5.4167) and close(SSE / 6, .9028) and close(sqrt(SSE / 6), .9501), (SSE, float(SSE / 6), sqrt(SSE / 6)))
check(T, "例1 ŷᵀe=0", sum(a * c for a, c in zip(yhat, ef)) == 0, 0)
s2 = float(SSE / 6)
a5, a8 = sqrt(s2 * (1 - 1 / 8)), sqrt(s2 * (1 - 1 / 8 - 49 / 84))
check(T, "例2 h55=0.125, h88=0.7083", close(d["h"][4], .125) and close(d["h"][7], .7083), r4(d["h"][[4, 7]]))
check(T, "例2 √(s²(1−h55))=0.8888, √(s²(1−h88))=0.5131", close(a5, .8888) and close(a8, .5131), (a5, a8))
check(T, "例2 店8は店5の6割弱（0.5〜0.6）", 0.5 < a8 / a5 < 0.6, a8 / a5)
# 記号：一般の X で (I-H)X=O、対称・べき等、tr(I-H)=n-p-1
Xs = sp.Matrix(5, 3, lambda i, j: 1 if j == 0 else sp.Symbol(f"x{i}{j}"))
Xn = Xs.subs({sp.Symbol(f"x{i}{j}"): v for (i, j), v in zip([(i, j) for i in range(5) for j in (1, 2)], [1, 4, 2, 0, 3, 5, 7, 1, 2, 9])})
Hs = Xn * (Xn.T * Xn).inv() * Xn.T
I5 = sp.eye(5)
check(T, "(I−H)X=O, (I−H)ᵀ=I−H, (I−H)²=I−H（有理数で厳密）", sp.simplify((I5 - Hs) * Xn) == sp.zeros(5, 3) and (I5 - Hs).T == I5 - Hs and sp.simplify((I5 - Hs) ** 2 - (I5 - Hs)) == sp.zeros(5, 5), "OK")
check(T, "tr(I−H)=n−p−1=5−2−1=2", sp.simplify((I5 - Hs).trace()) == 2, (I5 - Hs).trace())
# シミュレーション：V[e]=σ²(I−H)、E[s²]=σ²、Cov(e,ŷ)=0、Cov(e,y)=σ²(I−H)
Xd = d["X"]
Hd = d["H"]
sig = 1.5
R = 200000
eps = rng.normal(0, sig, (R, 8))
Y = (Xd @ np.array([2.0, 1.5]))[None, :] + eps
E = Y @ (np.eye(8) - Hd).T
Yh = Y @ Hd.T
Ve = np.cov(E.T)
check(T, "sim V[e]=σ²(I−H)（20万回、全成分 ±0.03）", np.max(np.abs(Ve - sig ** 2 * (np.eye(8) - Hd))) < 0.03, np.max(np.abs(Ve - sig ** 2 * (np.eye(8) - Hd))))
check(T, "sim E[e]=0（±0.01）", np.max(np.abs(E.mean(0))) < 0.01, np.max(np.abs(E.mean(0))))
s2sim = (E ** 2).sum(1) / 6
check(T, "sim E[s²]=σ²=2.25（±0.02）", abs(s2sim.mean() - sig ** 2) < 0.02, s2sim.mean())
C1 = (E - E.mean(0)).T @ (Yh - Yh.mean(0)) / R
C2 = (E - E.mean(0)).T @ (Y - Y.mean(0)) / R
check(T, "sim Cov(e,ŷ)=O（±0.03）", np.max(np.abs(C1)) < 0.03, np.max(np.abs(C1)))
check(T, "sim Cov(e,y)=σ²(I−H)≠O（±0.03）", np.max(np.abs(C2 - sig ** 2 * (np.eye(8) - Hd))) < 0.03, np.max(np.abs(C2 - sig ** 2 * (np.eye(8) - Hd))))
check(T, "非対角 −σ²h_ij ≠ 0（例1の H）", np.min(np.abs(Hd[~np.eye(8, dtype=bool)])) >= 0 and np.any(np.abs(Hd[~np.eye(8, dtype=bool)]) > 0.01), round(Hd[0, 7], 4))
# 切片なしでは Σe≠0 の例
b_no = (xs @ ys) / (xs @ xs)
check(T, "切片なしでは Σe≠0（例1で原点回帰）", abs(np.sum(ys - b_no * xs)) > 0.1, np.sum(ys - b_no * xs))
ch = charts(T)
check(T, "図 regression-residual params（n=20, β0=1, β1=0.5, σ=1, outlier=0）", ch[0]["id"] == "regression-residual" and ch[0]["params"] == {"n": 20, "beta0": 1, "beta1": 0.5, "sigma": 1, "outlier": 0}, ch[0]["params"])

# ============ leverage ============
T = "leverage"
hf = [F(1, 8) + (a - xb) ** 2 / Sxx for a in xf]
hc = [0.3155, 0.2321, 0.1726, 0.1369, 0.125, 0.1369, 0.1726, 0.7083]
check(T, "例1 h_ii（1/n+(x−x̄)²/Sxx）", allclose([float(v) for v in hf], hc), r4(hf))
check(T, "例1 h_ii を H の対角（numpy）でも", allclose(d["h"], hc), r4(d["h"]))
check(T, "例1 h11=0.125+0.1905, h88=0.125+0.5833", close(F(16, 84), .1905) and close(F(49, 84), .5833), (16 / 84, 49 / 84))
check(T, "例1 Σh=2 厳密, 平均 0.25", sum(hf) == 2, sum(hf))
check(T, "例1 2(p+1)/n=0.5 を超えるのは店8だけ", [i + 1 for i, v in enumerate(hf) if v > F(1, 2)] == [8], [i + 1 for i, v in enumerate(hf) if v > F(1, 2)])
row8 = d["H"][7]
check(T, "例2 H の8行目の係数", allclose(row8, [-0.2083, -0.125, -0.0417, 0.0417, 0.125, 0.2083, 0.2917, 0.7083]), r4(row8))
check(T, "例2 係数の和=1, ŷ8=21.0833=2.0833+1.5833×12", close(row8.sum(), 1, 1e-12) and close(row8 @ ys, 21.0833) and close(b0 + 12 * b1, 21.0833), row8 @ ys)
# h_ii = ∂ŷ_i/∂y_i（数値微分）
yy = ys.copy()
yy[7] += 1
check(T, "h88=∂ŷ8/∂y8（y8 を1増やすと ŷ8 が h88 増える）", close((d["X"] @ fit(d["X"], yy)[0])[7] - 21.0833333333, d["h"][7], 1e-9), (d["X"] @ fit(d["X"], yy)[0])[7] - 21.0833333333)
# 一般の X で 1/n≤h≤1, tr=p+1, 中心化不変, マハラノビス
okb = okt = okc = okm = True
for _ in range(300):
    n = int(rng.integers(5, 30))
    p = int(rng.integers(1, 4))
    Z = rng.normal(size=(n, p)) * rng.uniform(.5, 3, p) + rng.normal(size=p)
    if rng.random() < .3:
        Z[0] += 8
    X = np.column_stack([np.ones(n), Z])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    okb &= bool(np.all(h >= 1 / n - 1e-12) and np.all(h <= 1 + 1e-12))
    okt &= abs(h.sum() - (p + 1)) < 1e-9
    Xc = np.column_stack([np.ones(n), Z - Z.mean(0)])
    okc &= np.allclose(Xc @ np.linalg.inv(Xc.T @ Xc) @ Xc.T, H)
    S = np.cov(Z.T).reshape(p, p)
    md2 = np.einsum("ij,jk,ik->i", Z - Z.mean(0), np.linalg.inv(S), Z - Z.mean(0))
    okm &= np.allclose(h, 1 / n + md2 / (n - 1))
check(T, "ランダム X 300個で 1/n≤h_ii≤1", okb, okb)
check(T, "ランダム X 300個で Σh_ii=tr(H)=p+1", okt, okt)
check(T, "中心化しても H は不変", okc, okc)
check(T, "h_ii=1/n+MD²/(n−1)（マハラノビス距離と同じ働き）", okm, okm)
k1, k2, n_ = sp.symbols("k1 k2 n", positive=True)
Xsym = sp.Matrix([[1, 1], [1, 2], [1, 4], [1, 7]])
Hsym = Xsym * (Xsym.T * Xsym).inv() * Xsym.T
check(T, "tr(H)=p+1（sympy, 例 4×2）, h_ii=Σ_j h_ij²", Hsym.trace() == 2 and all(Hsym[i, i] == sum(Hsym[i, j] ** 2 for j in range(4)) for i in range(4)), Hsym.trace())
ch = charts(T)
check(T, "図1 bar-data の値 = 例1 の h_ii", allclose(ch[0]["params"]["values"], [round(float(v), 4) for v in hf], 1e-9), ch[0]["params"]["values"])
check(T, "図1 キャプション：合計 p+1=2・平均 0.25・目安 0.5", abs(sum(ch[0]["params"]["values"]) - 2) < 5e-4, sum(ch[0]["params"]["values"]))
# 図2：右端 x=16、他は U(0,10)。n を増やすと右端のてこ比が下がる（期待値で確認）
hs = {}
for n in (6, 20, 60):
    vals = []
    for _ in range(4000):
        x_ = np.append(rng.uniform(0, 10, n - 1), 16.0)
        vals.append(1 / n + (16 - x_.mean()) ** 2 / np.sum((x_ - x_.mean()) ** 2))
    hs[n] = np.mean(vals)
check(T, "図2 右端 x=16（charts_custom.js）の平均てこ比が n=6,20,60 で減少し、平均 2/n より大", hs[6] > hs[20] > hs[60] and hs[20] > 2 / 20 and hs[60] > 2 / 60, {k: round(v, 3) for k, v in hs.items()})

# ============ standardized-residual ============
T = "standardized-residual"
rc = [-0.8481, -0.3003, -0.9642, 0.6608, 0, 1.6049, 0.9642, -2.1112]
check(T, "例1 全8店の r_i", allclose(d["r"], rc), r4(d["r"]))
check(T, "例1 表 s√(1−h66)=0.8827, s√(1−h88)=0.5131, h66=0.1369", close(sqrt(s2 * (1 - d["h"][5])), .8827) and close(sqrt(s2 * (1 - d["h"][7])), .5131) and close(d["h"][5], .1369), (sqrt(s2 * (1 - d["h"][5])), sqrt(s2 * (1 - d["h"][7]))))
check(T, "例1 √(0.9028×0.8631)=0.8827（表の途中式）", close(sqrt(.9028 * .8631), .8827), sqrt(.9028 * .8631))
check(T, "例1 |r|>2 は店8だけ、上限 √6=2.449", [i + 1 for i in range(8) if abs(d["r"][i]) > 2] == [8] and close(sqrt(6), 2.449, 5e-4) and np.all(np.abs(d["r"]) <= sqrt(6)), sqrt(6))
check(T, "例1 素の残差最大は店6、標準化残差の絶対値最大は店8", np.argmax(np.abs(d["e"])) == 5 and np.argmax(np.abs(d["r"])) == 7, (np.argmax(np.abs(d["e"])) + 1, np.argmax(np.abs(d["r"])) + 1))
pn2 = 2 * stats.norm.sf(2)
pn3 = 2 * stats.norm.sf(3)
check(T, "例2 2{1−Φ(2)}=0.0455, 100点で約4.6点, |r|>3 は 0.0027", close(pn2, .0455) and close(100 * pn2, 4.6, .05) and close(pn3, .0027), (pn2, pn3))
pb = stats.beta(.5, 2.5).sf(4 / 6)
check(T, "例2 Beta(1/2,5/2) で P(r²/6>4/6)=0.0250", close(pb, .0250), pb)
pb8 = stats.beta(.5, 2.5).sf(d["r"][7] ** 2 / 6)
pt8 = 2 * stats.t(5).sf(abs(d["t"][7]))
check(T, "例2 P(|r|>2.1112)=0.0126 = t(5) の両側 p 値", close(pb8, .0126) and abs(pb8 - pt8) < 1e-10, (pb8, pt8))
# r²/(n−p−1) ~ Beta(1/2,(n−p−2)/2) シミュレーション（例1 の X、店8 と店1）
Rr = E / np.sqrt((E ** 2).sum(1, keepdims=True) / 6 * (1 - d["h"])[None, :])
for i in (0, 7):
    ks = stats.kstest(Rr[:, i] ** 2 / 6, stats.beta(.5, 2.5).cdf)
    check(T, f"sim r_{i+1}²/6 ~ Beta(1/2,5/2)（20万回、KS 統計量 <0.004）", ks.statistic < 0.004, round(ks.statistic, 5))
check(T, "sim |r_i|≤√(n−p−1) は常に成立", np.max(np.abs(Rr)) <= sqrt(6) + 1e-9, np.max(np.abs(Rr)))
check(T, "sim P(|r|>2)=0.0250（店8、±0.0015）", abs(np.mean(np.abs(Rr[:, 7]) > 2) - pb) < 0.0015, np.mean(np.abs(Rr[:, 7]) > 2))
# Σe² ≥ e_i²/(1−h) は i 除外 SSE ≥ 0 による
check(T, "Σe_j² − e_i²/(1−h_ii) = i 除外の SSE（例1 全店）", np.allclose(d["e"] @ d["e"] - d["e"] ** 2 / (1 - d["h"]), d["s2i"] * 5), r4(d["s2i"] * 5))
check(T, "t_i=r_i√((n−p−2)/(n−p−1−r_i²))（例1 全店）", np.allclose(d["t"], d["r"] * np.sqrt(5 / (6 - d["r"] ** 2))), r4(d["t"]))
check(T, "D_i=r_i²h_ii/{(p+1)(1−h_ii)}（例1 全店）", np.allclose(d["D"], d["r"] ** 2 * d["h"] / (2 * (1 - d["h"]))), r4(d["D"]))
ch = charts(T)
check(T, "図1 素の残差 = 例1 の e_i", allclose(ch[0]["params"]["values"], [-0.6667, -0.25, -0.8333, 0.5833, 0, 1.4167, 0.8333, -1.0833], 1e-9) and allclose(d["e"], ch[0]["params"]["values"]), ch[0]["params"]["values"])
check(T, "図2 標準化残差 = r_i", allclose(d["r"], ch[1]["params"]["values"]), ch[1]["params"]["values"])

# ============ studentized-residual ============
T = "studentized-residual"
tc = [-0.8252, -0.2762, -0.9575, 0.6265, 0, 1.9393, 0.9575, -3.8006]
check(T, "例1 全8店の t_i（当てはめ直しで計算）", allclose(d["t"], tc), r4(d["t"]))
e8, h8 = float(ef[7]), float(hf[7])
check(T, "例1 e8²/(1−h88)=4.0238, 5s(8)²=1.3929", close(e8 ** 2 / (1 - h8), 4.0238) and close(float(SSE) - e8 ** 2 / (1 - h8), 1.3929), (e8 ** 2 / (1 - h8), float(SSE) - e8 ** 2 / (1 - h8)))
check(T, "例1 s(8)²=0.2786, s(8)=0.5278（当てはめ直しと一致）", close(d["s2i"][7], .2786) and close(sqrt(d["s2i"][7]), .5278), (d["s2i"][7], sqrt(d["s2i"][7])))
b8 = d["bdel"][7]
sse8 = 5 * d["s2i"][7]
check(T, "例1 店8除外の直線 1.0000+1.8929x、SSE 1.3929", close(b8[0], 1.0) and close(b8[1], 1.8929) and close(sse8, 1.3929), (b8, sse8))
check(T, "(n−p−2)s(i)²=(n−p−1)s²−e_i²/(1−h_ii)（全店）", np.allclose(5 * d["s2i"], 6 * d["s2"] - d["e"] ** 2 / (1 - d["h"])), "OK")
den = sqrt(d["s2i"][7]) * sqrt(1 - h8)
check(T, "例1 分母 s(8)√(1−h88)=0.2850（本文修正後）, t8=−3.8006", close(den, .2850) and close(e8 / den, -3.8006), (den, e8 / den))
tr = -2.1112 * sqrt(5 / (6 - 2.1112 ** 2))
check(T, "例1 r8 から：2.1112²=4.4572, t8=−3.8006", close(2.1112 ** 2, 4.4572) and close(tr, -3.8006, 1.5e-4), (2.1112 ** 2, tr))
check(T, "例1 両側 p=0.0126, t_0.025(5)=2.571", close(pt8, .0126) and close(stats.t(5).ppf(.975), 2.571, 5e-4), (pt8, stats.t(5).ppf(.975)))
tb = stats.t(5).ppf(1 - .05 / 16)
check(T, "例2 t_{0.05/16}(5)=4.526, |t8|<4.526, 8×0.0126=0.101", close(tb, 4.526, 5e-4) and abs(d["t"][7]) < tb and close(8 * pt8, .101, 5e-4), (tb, 8 * pt8))
# シャーマン・モリソンと削除残差
XtXi = np.linalg.inv(Xd.T @ Xd)
okSM = okDR = okV = True
for i in range(8):
    okSM &= np.allclose(d["b"] - d["bdel"][i], XtXi @ Xd[i] * d["e"][i] / (1 - d["h"][i]))
    okDR &= np.isclose(ys[i] - Xd[i] @ d["bdel"][i], d["e"][i] / (1 - d["h"][i]))
    m = np.arange(8) != i
    okV &= np.isclose(1 + Xd[i] @ np.linalg.inv(Xd[m].T @ Xd[m]) @ Xd[i], 1 / (1 - d["h"][i]))
check(T, "β̂−β̂(i)=(XᵀX)⁻¹x_i e_i/(1−h_ii)（全店）", okSM, okSM)
check(T, "削除残差 y_i−x_iᵀβ̂(i)=e_i/(1−h_ii)（全店）", okDR, okDR)
check(T, "1+x_iᵀ(X(i)ᵀX(i))⁻¹x_i=1/(1−h_ii)（全店）", okV, okV)
# 記号：シャーマン・モリソン 1+h/(1−h)... ：1 + c/(1−h)... 確認 x_i^T (A - x x^T)^{-1} x = h/(1-h)
hh = sp.symbols("h", positive=True)
check(T, "e+he/(1−h)=e/(1−h)（sympy）", sp.simplify(1 + hh / (1 - hh) - 1 / (1 - hh)) == 0, "OK")
# 平均シフトモデル
Xm = np.column_stack([Xd, (np.arange(8) == 7).astype(float)])
bm, em = fit(Xm, ys)
sem = sqrt(em @ em / (8 - 3) * np.linalg.inv(Xm.T @ Xm)[2, 2])
check(T, "平均シフトモデルの δ の t 値 = t8", close(bm[2] / sem, d["t"][7], 1e-9), bm[2] / sem)
# t_i ~ t(5) シミュレーション
Rt = Rr * np.sqrt(5 / (6 - Rr ** 2))
ks = stats.kstest(Rt[:, 7], stats.t(5).cdf)
check(T, "sim t8 ~ t(5)（20万回、KS 統計量 <0.004）", ks.statistic < 0.004, round(ks.statistic, 5))
check(T, "PRESS=Σ{e_i/(1−h_ii)}² が1つ抜き予測誤差の二乗和と一致", np.isclose(np.sum((d["e"] / (1 - d["h"])) ** 2), sum((ys[i] - Xd[i] @ d["bdel"][i]) ** 2 for i in range(8))), np.sum((d["e"] / (1 - d["h"])) ** 2))
ch = charts(T)
check(T, "図1 bar-data = t_i", allclose(d["t"], ch[0]["params"]["values"]), ch[0]["params"]["values"])
check(T, "図2 t 分布 df=5=n−p−2", ch[1]["dist"] == "t" and ch[1]["params"]["df"] == 5, ch[1]["params"])

# ============ cooks-distance ============
T = "cooks-distance"
Dc = [0.1657, 0.0136, 0.0970, 0.0346, 0.0, 0.2043, 0.0970, 5.4122]
check(T, "例1 全8店の D_i（当てはめ直しの定義式）", allclose(d["D"], Dc), r4(d["D"]))
check(T, "簡便式 e²h/{(p+1)s²(1−h)²} と一致", np.allclose(d["D"], d["e"] ** 2 * d["h"] / (2 * d["s2"] * (1 - d["h"]) ** 2)), "OK")
okq = all(np.isclose((d["b"] - d["bdel"][i]) @ Xd.T @ Xd @ (d["b"] - d["bdel"][i]) / (2 * d["s2"]), d["D"][i]) for i in range(8))
check(T, "係数形の定義式（XᵀX の二次形式）と一致", okq, okq)
h6 = d["h"][5]
check(T, "表 店6：1.6049²=2.5757, h/(1−h)=0.1586, D6=0.2043", close(1.6049 ** 2, 2.5757) and close(h6 / (1 - h6), .1586) and close(2.5757 / 2 * .1586, .2043), (1.6049 ** 2, h6 / (1 - h6)))
check(T, "表 店8：h/(1−h)=2.4286, 4.4572/2×2.4286≈5.4122（丸めた途中値での積は ±5e-4、厳密値は別チェック）", close(h8 / (1 - h8), 2.4286) and close(4.4572 / 2 * 2.4286, 5.4122, 5e-4) and close(d["r"][7] ** 2 / 2 * h8 / (1 - h8), 5.4122), (h8 / (1 - h8), 4.4572 / 2 * 2.4286))
f50 = stats.f(2, 6).ppf(.5)
check(T, "例1 4/n=0.5, F_0.5(2,6)=0.780, どの目安でも店8だけ", close(f50, .780, 5e-4) and [i + 1 for i in range(8) if d["D"][i] > min(.5, f50, 1)] == [8], f50)
check(T, "例2 β̂−β̂(8)=(1.0833, −0.3095)", allclose(d["b"] - b8, [1.0833, -0.3095]), d["b"] - b8)
check(T, "例2 傾き 1.5833→1.8929 で約2割上がる", 0.15 < b8[1] / d["b"][1] - 1 < 0.25, b8[1] / d["b"][1] - 1)
dy = Xd @ d["b"] - Xd @ b8
check(T, "例2 当てはめ値の変化", allclose(dy, [0.7738, 0.4643, 0.1548, -0.1548, -0.4643, -0.7738, -1.0833, -2.6310]), r4(dy))
check(T, "例2 二乗和 9.7721, D8=9.7721/(2×0.9028)=5.4122", close(dy @ dy, 9.7721) and close(9.7721 / (2 * .9028), 5.4122, 1e-4), (dy @ dy, 9.7721 / (2 * .9028)))
check(T, "例2 F(2,6) で 5.41 は約95.5%点", close(stats.f(2, 6).cdf(d["D"][7]), .955, 5e-4), stats.f(2, 6).cdf(d["D"][7]))
# 記号：二次形式の簡約
a_, b_, c_, u1, u2, ee = sp.symbols("a b c u1 u2 e", real=True)
A = sp.Matrix([[a_, b_], [b_, c_]])
xv = sp.Matrix([u1, u2])
delta = A.inv() * xv * ee / (1 - hh)
expr = (delta.T * A * delta)[0] - ee ** 2 / (1 - hh) ** 2 * (xv.T * A.inv() * xv)[0]
check(T, "δᵀXᵀXδ = e²/(1−h)² · xᵀ(XᵀX)⁻¹x（sympy）", sp.simplify(expr) == 0, "OK")
ch = charts(T)
check(T, "図1 bar-data = D_i", allclose(d["D"], ch[0]["params"]["values"]), ch[0]["params"]["values"])
check(T, "図1 キャプション：店8 5.41、店6 0.20（素の残差最大）", close(d["D"][7], 5.41, .005) and close(d["D"][5], .20, .005), (d["D"][7], d["D"][5]))
check(T, "図2 regression-residual params（outlier=−5）", ch[1]["params"]["outlier"] == -5 and ch[1]["params"]["n"] == 20, ch[1]["params"])

# ============ dffits ============
T = "dffits"
dfc = [-0.5602, -0.1519, -0.4374, 0.2495, 0, 0.7724, 0.4374, -5.9228]
check(T, "例1 全8店の DFFITS（定義式）", allclose(d["dffits"], dfc), r4(d["dffits"]))
check(T, "簡便式 t_i√(h/(1−h)) と一致", np.allclose(d["dffits"], d["t"] * np.sqrt(d["h"] / (1 - d["h"]))), "OK")
check(T, "例1 ŷ8(8)=23.7143, 差 −2.6310 = h e/(1−h)", close(Xd[7] @ b8, 23.7143) and close(21.0833333 - Xd[7] @ b8, -2.6310) and close(h8 * e8 / (1 - h8), -2.6310), Xd[7] @ b8)
check(T, "例1 −2.6310/(0.5278×√0.7083)≈−5.9228（丸めた途中値で ±5e-4、厳密値も確認）", close(-2.6310 / (.5278 * sqrt(.7083)), -5.9228, 5e-4) and close((h8 * e8 / (1 - h8)) / (sqrt(d["s2i"][7]) * sqrt(h8)), -5.9228), -2.6310 / (.5278 * sqrt(.7083)))
check(T, "例1 √(0.7083/0.2917)=1.5584, −3.8006×1.5584=−5.9228", close(sqrt(h8 / (1 - h8)), 1.5584) and close(-3.8006 * 1.5584, -5.9228, 1e-4), sqrt(h8 / (1 - h8)))
check(T, "例1 目安 2√(2/8)=1、店8以外の最大 |DFFITS|=0.7724（店6）", sqrt(2 / 8) * 2 == 1 and close(np.max(np.abs(d["dffits"][:7])), .7724) and np.argmax(np.abs(d["dffits"][:7])) == 5, np.max(np.abs(d["dffits"][:7])))
check(T, "D_i = s(i)²/((p+1)s²)·DFFITS²（全店）", np.allclose(d["D"], d["s2i"] / (2 * d["s2"]) * d["dffits"] ** 2), "OK")
check(T, "例1 0.2786/(2×0.9028)×5.9228²≈5.4122（丸めた途中値で ±1e-3、厳密値も確認）", close(.2786 / (2 * .9028) * 5.9228 ** 2, 5.4122, 1e-3) and close(d["s2i"][7] / (2 * d["s2"]) * d["dffits"][7] ** 2, 5.4122), .2786 / (2 * .9028) * 5.9228 ** 2)
check(T, "V[ŷ_i]=σ²h_ii（sim、±0.03）", np.max(np.abs(np.var(Yh, 0) - sig ** 2 * d["h"])) < 0.03, np.max(np.abs(np.var(Yh, 0) - sig ** 2 * d["h"])))
ch = charts(T)
check(T, "図 bar-data = DFFITS、目安 ±1 を超えるのは店8だけ", allclose(d["dffits"], ch[0]["params"]["values"]) and [i + 1 for i, v in enumerate(ch[0]["params"]["values"]) if abs(v) > 1] == [8], ch[0]["params"]["values"])

# ============ outlier-and-influential-point ============
T = "outlier-and-influential-point"
check(T, "例1 表（e, h, t, D の店6・店8）", allclose([d["e"][5], d["h"][5], d["t"][5], d["D"][5], d["e"][7], d["h"][7], d["t"][7], d["D"][7]], [1.4167, .1369, 1.9393, .2043, -1.0833, .7083, -3.8006, 5.4122]), "OK")
y2 = ys.copy()
y2[4] = 14
d2 = diag_all(xs, y2)
check(T, "例2 ŷ=2.5833+1.5833x（傾き不変、切片 +0.5=4/8）", allclose(d2["b"], [2.5833, 1.5833]) and close(d2["b"][0] - d["b"][0], .5, 1e-12) and close(d2["b"][1], d["b"][1], 1e-12), d2["b"])
p5 = 2 * stats.t(5).sf(abs(d2["t"][4]))
check(T, "例2 e5=3.5, t5=3.5949, p=0.0156, D5=0.3090", close(d2["e"][4], 3.5) and close(d2["t"][4], 3.5949) and close(p5, .0156) and close(d2["D"][4], .3090), (d2["e"][4], d2["t"][4], p5, d2["D"][4]))
check(T, "例2 s² 0.9028→3.2361", close(d2["s2"], 3.2361), d2["s2"])
y3 = ys.copy()
y3[7] = 1 + 53 / 28 * 12   # 他7店の直線上の値（厳密）
d3 = diag_all(xs, y3)
check(T, "例3 他7店の直線上の値 23.7143", close(y3[7], 23.7143), y3[7])
check(T, "例3 8店でも同じ直線、e8=0、D8=0", np.allclose(d3["b"], b8) and abs(d3["e"][7]) < 1e-10 and abs(d3["D"][7]) < 1e-10, d3["b"])
check(T, "例3 てこ比は y によらない（h88=0.7083 のまま）", np.allclose(d3["h"], d["h"]) and np.allclose(d2["h"], d["h"]), d3["h"][7])
se_wo = sqrt(d["s2i"][7] / 28)
se_w = sqrt(d3["s2"] / 84)
check(T, "例3 傾きの SE：店8なし √(0.2786/28)=0.0997、あり √(0.2321/84)=0.0526（約半分）", close(se_wo, .0997) and close(se_w, .0526) and close(d3["s2"], .2321) and 0.45 < se_w / se_wo < 0.6, (se_wo, se_w, d3["s2"]))
xv_ = sp.symbols("x1:6")
yv_ = sp.symbols("y1:6")
Dl = sp.symbols("Delta")
xbar_ = sum(xv_) / 5
Sxx_ = sum((v - xbar_) ** 2 for v in xv_)
slope = lambda Y: sum((xv_[j] - xbar_) * Y[j] for j in range(5)) / Sxx_
Ysh = list(yv_)
Ysh[2] = Ysh[2] + Dl
subs_c = {xv_[2]: (xv_[0] + xv_[1] + xv_[3] + xv_[4]) / 4}  # x3 = x̄
dslope = sp.simplify((slope(Ysh) - slope(yv_)).subs(subs_c))
icpt = lambda Y: sum(Y) / 5 - slope(Y) * xbar_
dint = sp.simplify((icpt(Ysh) - icpt(yv_)).subs(subs_c))
check(T, "x_i=x̄ の点を Δ ずらすと傾き不変・切片 Δ/n（sympy, n=5）", dslope == 0 and sp.simplify(dint - Dl / 5) == 0, (dslope, dint))
ch = charts(T)
check(T, "図 regression-residual params（右端 x=16 は charts_custom.js の固定値）", ch[0]["params"]["outlier"] == 0 and "xs.push(16)" in (ROOT / "src" / "charts_custom.js").read_text(encoding="utf-8"), ch[0]["params"])

# ============ residual-plot ============
T = "residual-plot"
xq = np.arange(1, 9, dtype=float)
yq = np.array([2, 3, 5, 8, 12, 17, 23, 30], float)
dq = diag_all(xq, yq)
R2 = 1 - dq["e"] @ dq["e"] / np.sum((yq - yq.mean()) ** 2)
check(T, "例1 ŷ=−5.5+4.0x, R²=0.9412", allclose(dq["b"], [-5.5, 4.0], 1e-12) and close(R2, .9412), (dq["b"], R2))
check(T, "例1 当てはめ値・残差の表", allclose(dq["X"] @ dq["b"], [-1.5, 2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5], 1e-9) and allclose(dq["e"], [3.5, .5, -1.5, -2.5, -2.5, -1.5, .5, 3.5], 1e-9), dq["e"])
check(T, "例1 Σe=0, Σŷe=0", abs(dq["e"].sum()) < 1e-9 and abs((dq["X"] @ dq["b"]) @ dq["e"]) < 1e-9, "OK")
check(T, "例1 2次式 2−0.5x+0.5x² で残差すべて0", np.allclose(2 - .5 * xq + .5 * xq ** 2, yq), 2 - .5 * xq + .5 * xq ** 2)
check(T, "例1 符号 ＋＋−−−−＋＋", "".join("+" if v > 0 else "-" for v in dq["e"]) == "++----++", "".join("+" if v > 0 else "-" for v in dq["e"]))
check(T, "例2 当てはめ値・残差の2桁丸め", allclose(Xd @ d["b"], [3.67, 5.25, 6.83, 8.42, 10.00, 11.58, 13.17, 21.08], .005) and allclose(d["e"], [-.67, -.25, -.83, .58, 0, 1.42, .83, -1.08], .005), "OK")
Hs_ = sp.MatrixSymbol("H", 3, 3)
check(T, "Cov(e,ŷ)=σ²(I−H)H=O（例1 の H、数値）", np.allclose((np.eye(8) - Hd) @ Hd, 0), "OK")
ch = charts(T)
check(T, "図1 line-data = 例1 の ŷ と e", ch[0]["params"]["x"] == [-1.5, 2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5] and ch[0]["params"]["series"][0]["y"] == [3.5, .5, -1.5, -2.5, -2.5, -1.5, .5, 3.5], ch[0]["params"]["series"][0]["y"])
check(T, "図1 キャプション R² 0.94", close(R2, .94, .005), R2)
check(T, "図2 residual-pattern params（curve）", ch[1]["params"]["pattern"] == "curve", ch[1]["params"])

# ============ normal-qq-plot ============
T = "normal-qq-plot"
pp = (np.arange(1, 9) - .5) / 8
q = stats.norm.ppf(pp)
check(T, "例1 プロット位置 (i−0.5)/8", allclose(pp, [.0625, .1875, .3125, .4375, .5625, .6875, .8125, .9375], 1e-12), pp)
check(T, "例1 q_i", allclose(q, [-1.5341, -.8871, -.4888, -.1573, .1573, .4888, .8871, 1.5341]), r4(q))
rs = np.sort(d["r"])
check(T, "例1 r_(i)（標準化残差の並べ替え）", allclose(rs, [-2.1112, -.9642, -.8481, -.3003, 0, .6608, .9642, 1.6049]), r4(rs))
check(T, "例1 相関 0.9926", close(np.corrcoef(q, rs)[0, 1], .9926), np.corrcoef(q, rs)[0, 1])
rep = np.array([1.2, 1.5, 1.9, 2.4, 3.1, 4.2, 6.0, 9.7])
check(T, "例2 相関 0.9200", close(np.corrcoef(q, rep)[0, 1], .9200), np.corrcoef(q, rep)[0, 1])
lr = np.log(rep)
check(T, "例2 log: 0.1823, 0.4055, …, 2.2721、相関 0.9893", close(lr[0], .1823) and close(lr[1], .4055) and close(lr[-1], 2.2721) and close(np.corrcoef(q, lr)[0, 1], .9893), (r4(lr), np.corrcoef(q, lr)[0, 1]))
# 例2：右端が直線より上（四分位点を通る直線基準でも、最小二乗直線でも）
bq = np.polyfit(q, rep, 1)
check(T, "例2 最大値 9.7 が最小二乗直線より大きく上（残差最大）", np.argmax(rep - np.polyval(bq, q)) == 7, rep - np.polyval(bq, q))
Zs = np.sort(rng.normal(size=(200000, 8)), axis=1)
Zc = Zs - Zs.mean(1, keepdims=True)
qc = q - q.mean()
cors = (Zc @ qc) / np.sqrt((Zc ** 2).sum(1) * (qc @ qc))
c05 = np.quantile(cors, .05)
check(T, "sim n=8 正規標本の相関の5%点 ≈0.91（20万回、±0.01）", abs(c05 - .91) < 0.01 and c05 < .92, c05)
check(T, "N(μ,σ²) の p 分位点 = μ+σΦ⁻¹(p)", close(stats.norm(3, 2).ppf(.3), 3 + 2 * stats.norm.ppf(.3), 1e-12), stats.norm(3, 2).ppf(.3))
ch = charts(T)
check(T, "図 qq-plot params（norm n=50 / t df=3 / expon）", [c["params"]["dist"] for c in ch] == ["norm", "t", "expon"] and ch[1]["params"]["df"] == 3, [c["params"] for c in ch])
# 形の確認（期待順序統計量で）：t は S 字（両端外）、expon は下に凸
qq50 = stats.norm.ppf((np.arange(1, 51) - .5) / 50)
for dist, nm in ((stats.t(3), "t(3)"), (stats.expon(), "expon")):
    S = np.sort(dist.rvs(size=(20000, 50), random_state=rng), axis=1).mean(0)
    bb = np.polyfit(qq50, S, 1)
    res = S - np.polyval(bb, qq50)
    if nm == "t(3)":
        ok = res[0] < 0 and res[-1] > 0
    else:
        ok = res[0] > 0 and res[-1] > 0 and res[25] < 0
    check(T, f"図 {nm} の平均 Q-Q が{'S字（左下・右上へ外れる）' if nm=='t(3)' else '下に凸の弓なり'}", ok, (round(res[0], 3), round(res[25], 3), round(res[-1], 3)))

# ============ durbin-watson-statistic ============
T = "durbin-watson-statistic"


def dw(e):
    e = np.asarray(e, float)
    return np.sum(np.diff(e) ** 2) / np.sum(e ** 2)


e1 = [F(v) for v in (1, 2, 1, -1, -2, -2, 0, 1)]
num_ = sum((e1[i] - e1[i - 1]) ** 2 for i in range(1, 8))
den_ = sum(v * v for v in e1)
lag = sum(e1[i] * e1[i - 1] for i in range(1, 8))
check(T, "例1 和0、差の二乗和 12、残差平方和 16、d=0.75", sum(e1) == 0 and num_ == 12 and den_ == 16 and num_ / den_ == F(3, 4), (num_, den_))
check(T, "例1 ラグ1積和 9, r1=0.5625, 2(1−r1)=0.875, 端 0.125, 0.875−0.125=d", lag == 9 and 2 * (1 - lag / den_) == F(7, 8) and (e1[0] ** 2 + e1[-1] ** 2) / den_ == F(1, 8) and 2 * (1 - lag / den_) - F(1, 8) == num_ / den_, (lag, 2 * (1 - lag / den_)))
check(T, "例1 交互符号8個で d=28/8=3.5", close(dw([1, -1] * 4), 3.5, 1e-12), dw([1, -1] * 4))
tt = np.arange(1, 11, dtype=float)
ym = np.array([10, 12, 13, 15, 15, 15, 16, 18, 21, 22], float)
dm = diag_all(tt, ym)
em_ = dm["e"]
check(T, "例2 ŷ=9.0667+1.2061t, SSE=8.0970", allclose(dm["b"], [9.0667, 1.2061]) and close(em_ @ em_, 8.0970), (dm["b"], em_ @ em_))
check(T, "例2 分子 8.1458, d=1.0060", close(np.sum(np.diff(em_) ** 2), 8.1458) and close(dw(em_), 1.0060), (np.sum(np.diff(em_) ** 2), dw(em_)))
r1m = np.sum(em_[1:] * em_[:-1]) / (em_ @ em_)
check(T, "例2 積和 3.6061, r1=0.4454, 2(1−r1)=1.1093, 端 0.1033（本文修正後）", close(np.sum(em_[1:] * em_[:-1]), 3.6061) and close(r1m, .4454) and close(2 * (1 - r1m), 1.1093) and close((em_[0] ** 2 + em_[-1] ** 2) / (em_ @ em_), .1033) and close(1.1093 - .1033, 1.0060, 1e-9), (r1m, 2 * (1 - r1m), (em_[0] ** 2 + em_[-1] ** 2) / (em_ @ em_)))
check(T, "例2 0.879≤1.006≤1.320 → 判定保留", .879 <= dw(em_) <= 1.320, dw(em_))
# 例2 シミュレーション P(d≤1.006 | H0, X 固定)
M = np.eye(10) - dm["H"]
U = rng.normal(size=(200000, 10)) @ M.T
dsim = np.sum(np.diff(U, axis=1) ** 2, 1) / np.sum(U ** 2, 1)
psim = np.mean(dsim <= dw(em_))
# 厳密（Imhof 型ではなく固有値表示での数値積分の代わりに、より大きなシミュで照合）
check(T, "例2 sim P(d≤1.006)≈0.011（20万回、±0.001）", abs(psim - .011) < 0.001, psim)
# 正確な値：d≤c ⇔ uᵀM(A−cI)Mu≤0。固有値 λ_j で Σλ_j χ²_1 ≤ 0 の確率を Imhof の公式で計算
A = np.diag([1.] + [2.] * 8 + [1.]) - np.diag(np.ones(9), 1) - np.diag(np.ones(9), -1)
lam = np.linalg.eigvalsh(M @ (A - dw(em_) * np.eye(10)) @ M)
lam = lam[np.abs(lam) > 1e-10]
from scipy.integrate import quad


def imhof_cdf0(lam):
    def f(u):
        th = 0.5 * np.sum(np.arctan(lam * u))
        rho = np.prod((1 + lam ** 2 * u ** 2) ** 0.25)
        return np.sin(th) / (u * rho)
    v, _ = quad(f, 0, np.inf, limit=500)
    return 0.5 - v / np.pi   # P(Q ≤ 0)


pex = imhof_cdf0(lam)
check(T, "例2 Imhof の公式による正確な P(d≤1.006)≈0.011（±0.0015）とシミュの一致", abs(pex - .011) < 0.0015 and abs(pex - psim) < 0.001, (pex, psim))
check(T, "例2 正確な p 値なら5%で有意", pex < 0.05, pex)
# 一般性：d=2(1−r1)−(e1²+en²)/Σe² と 0≤d≤4（ランダム）
okd = True
for _ in range(500):
    e = rng.normal(size=int(rng.integers(3, 40))) * rng.uniform(.1, 5)
    r1 = np.sum(e[1:] * e[:-1]) / (e @ e)
    okd &= np.isclose(dw(e), 2 * (1 - r1) - (e[0] ** 2 + e[-1] ** 2) / (e @ e)) and 0 <= dw(e) <= 4
check(T, "d=2(1−r1)−(e1²+en²)/Σe² と 0≤d≤4（ランダム500系列）", okd, okd)
nn = sp.symbols("n", integer=True, positive=True)
es = sp.symbols("e1:7")
lhs = sum((es[i] - es[i - 1]) ** 2 for i in range(1, 6))
rhs = 2 * sum(v ** 2 for v in es[:6]) - (es[0] ** 2 + es[5] ** 2) - 2 * sum(es[i] * es[i - 1] for i in range(1, 6))
check(T, "分子の展開（sympy, n=6）", sp.expand(lhs - rhs) == 0, "OK")
check(T, "d=1.2 なら r1≈0.4", close(1 - 1.2 / 2, .4, 1e-12), 1 - 1.2 / 2)
# AR(1) 誤差で d≈2(1−ρ)（n=200 の回帰残差、2000回平均）
for rho in (0.5, -0.5):
    n = 200
    ds = []
    Xt = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    Mt = np.eye(n) - Xt @ np.linalg.inv(Xt.T @ Xt) @ Xt.T
    for _ in range(2000):
        u = rng.normal(size=n)
        e = np.empty(n)
        e[0] = u[0] / sqrt(1 - rho ** 2)
        for k in range(1, n):
            e[k] = rho * e[k - 1] + u[k]
        ds.append(dw(Mt @ e))
    check(T, f"sim AR(1) ρ={rho} で d≈2(1−ρ)={2*(1-rho)}（n=200、±0.06）", abs(np.mean(ds) - 2 * (1 - rho)) < 0.06, np.mean(ds))
ch = charts(T)
check(T, "図1 line-data = 例1 の残差", ch[0]["params"]["series"][0]["y"] == [1, 2, 1, -1, -2, -2, 0, 1] and ch[0]["params"]["x"] == list(range(1, 9)), ch[0]["params"]["series"][0]["y"])
check(T, "図1 キャプション DW 0.75", close(dw([1, 2, 1, -1, -2, -2, 0, 1]), .75, 1e-12), .75)
check(T, "図2 ar-sim φ=0.5、図3 dw-sim φ=0.5 n=30", ch[1]["params"]["phi"] == .5 and ch[2]["params"] == {"phi": .5, "n": 30}, (ch[1]["params"], ch[2]["params"]))

# ============ serial-correlation ============
T = "serial-correlation"


def infl(rho, n=10):
    return 1 + 2 * sum((1 - k / n) * rho ** k for k in range(1, n))


for rho, a_, b_, c_ in ((.5, 2.6004, 1.6126, 3.85), (.8, 5.4295, 2.3301, 1.84), (-.5, .3777, .6146, 26.47)):
    f_ = infl(rho)
    check(T, f"例1 ρ={rho}: 膨張率 {a_}, SE比 {b_}, 実質 n {c_}", close(f_, a_) and close(sqrt(f_), b_) and close(10 / f_, c_, .005), (f_, sqrt(f_), 10 / f_))
# 二重和の公式（sympy, n=6 の記号 ρ）
rr = sp.symbols("rho")
n = 6
dbl = sum(rr ** abs(t_ - s_) for t_ in range(n) for s_ in range(n)) / n ** 2
fm = (1 + 2 * sum((1 - sp.Rational(k, n)) * rr ** k for k in range(1, n))) / n
check(T, "V[ȳ]=γ0/n{1+2Σ(1−k/n)ρ^k}（sympy, n=6）", sp.expand(dbl - fm) == 0, "OK")
g0, su = sp.symbols("gamma0 sigma_u", positive=True)
check(T, "γ0=ρ²γ0+σu² ⇒ γ0=σu²/(1−ρ²)", sp.simplify(sp.solve(sp.Eq(g0, rr ** 2 * g0 + su ** 2), g0)[0] - su ** 2 / (1 - rr ** 2)) == 0, "OK")
# シミュレーション：AR(1) の分散・自己相関、ȳ の分散
rho, n, Rn = 0.7, 10, 200000
u = rng.normal(size=(Rn, 400))
e = np.empty_like(u)
e[:, 0] = u[:, 0] / sqrt(1 - rho ** 2)
for k in range(1, 400):
    e[:, k] = rho * e[:, k - 1] + u[:, k]
check(T, "sim V[ε]=1/(1−0.49)=1.9608（±0.02）", abs(e[:, 399].var() - 1 / (1 - rho ** 2)) < 0.02, e[:, 399].var())
acs = [np.corrcoef(e[:, 399], e[:, 399 - k])[0, 1] for k in (1, 2, 3)]
check(T, "sim Corr(ε_t,ε_{t−k})=ρ^k（k=1,2,3、±0.01）", allclose(acs, [rho, rho ** 2, rho ** 3], .01), r4(acs))
vb = e[:, :10].mean(1).var()
check(T, "sim V[ȳ]=γ0/n·膨張率（n=10、ρ=0.7、相対 ±2%）", abs(vb / (1 / (1 - rho ** 2) / 10 * infl(rho)) - 1) < 0.02, (vb, 1 / (1 - rho ** 2) / 10 * infl(rho)))
check(T, "例2 残差", allclose(em_, [-.2727, .5212, .3152, 1.1091, -.0970, -1.3030, -1.5091, -.7152, 1.0788, .8727]), r4(em_))
check(T, "例2 符号 −＋＋＋−−−−＋＋", "".join("+" if v > 0 else "-" for v in em_) == "-+++----++", "".join("+" if v > 0 else "-" for v in em_))
check(T, "例2 r1=0.4454, DW=1.0060, 2(1−r1)=1.1093", close(r1m, .4454) and close(dw(em_), 1.006) and close(2 * (1 - r1m), 1.1093), (r1m, dw(em_)))
# 正の系列相関・正の自己相関 x で OLS 標準誤差が過小（sim）
n, rho, Rn = 50, 0.7, 20000
xt = np.empty(n)
xt[0] = 0
zx = rng.normal(size=n)
for k in range(1, n):
    xt[k] = 0.8 * xt[k - 1] + zx[k]
Xs_ = np.column_stack([np.ones(n), xt])
XtXi_ = np.linalg.inv(Xs_.T @ Xs_)
u = rng.normal(size=(Rn, n))
e = np.empty_like(u)
e[:, 0] = u[:, 0] / sqrt(1 - rho ** 2)
for k in range(1, n):
    e[:, k] = rho * e[:, k - 1] + u[:, k]
B = e @ (XtXi_ @ Xs_.T).T
res = e - B @ Xs_.T
s2s = (res ** 2).sum(1) / (n - 2)
true_v = B[:, 1].var()
nom_v = (s2s * XtXi_[1, 1]).mean()
check(T, "sim 正の系列相関＋自己相関のある x で OLS は不偏・SE 過小", abs(B[:, 1].mean()) < 4 * sqrt(true_v / Rn) and nom_v < true_v, (B[:, 1].mean(), nom_v, true_v))
ch = charts(T)
check(T, "図 ar-sim φ=0.7 n=100、acf-plot ar1 φ=0.7", ch[0]["params"] == {"phi": .7, "n": 100} and ch[1]["params"]["model"] == "ar1" and ch[1]["params"]["phi"] == .7, (ch[0]["params"], ch[1]["params"]))

# ============ heteroscedasticity ============
T = "heteroscedasticity"
xh = np.arange(1, 9, dtype=float)
yh = np.array([2.1, 3.8, 6.4, 7.2, 11.5, 10.4, 16.8, 13.0])
dh = diag_all(xh, yh)
check(T, "例1 ŷ=0.4571+1.8762x", allclose(dh["b"], [.4571, 1.8762]), dh["b"])
check(T, "例1 残差", allclose(dh["e"], [-.2333, -.4095, .3143, -.7619, 1.6619, -1.3143, 3.2095, -2.4667]), r4(dh["e"]))
check(T, "例1 corr(|e|, x)=0.8964", close(np.corrcoef(np.abs(dh["e"]), xh)[0, 1], .8964), np.corrcoef(np.abs(dh["e"]), xh)[0, 1])
b2, e2 = fit(dh["X"], dh["e"] ** 2)
R2a = 1 - e2 @ e2 / np.sum((dh["e"] ** 2 - np.mean(dh["e"] ** 2)) ** 2)
check(T, "例1 補助回帰 R²=0.6269, nR²=5.0150", close(R2a, .6269) and close(8 * R2a, 5.0150, 5e-4), (R2a, 8 * R2a))
pbp = stats.chi2(1).sf(8 * R2a)
check(T, "例1 χ²(1) で p=0.0251 <0.05", close(pbp, .0251) and pbp < .05, pbp)
try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    lm, lmp, _, _ = het_breuschpagan(dh["e"], dh["X"], robust=True)
    check(T, "例1 statsmodels het_breuschpagan（スチューデント化版）と一致", close(lm, 8 * R2a, 1e-8) and close(lmp, pbp, 1e-8), (lm, lmp))
except ImportError:
    pass
sx2, sx4 = int(np.sum(xh ** 2)), int(np.sum(xh ** 4))
check(T, "例2 Σx²=204, Σx⁴=8772", (sx2, sx4) == (204, 8772), (sx2, sx4))
check(T, "例2 V[β̂]=8772/204²σ²=0.2108σ², WLS 1/8=0.125, 効率 0.593（約41%小さい）", close(sx4 / sx2 ** 2, .2108) and close(.125 / (sx4 / sx2 ** 2), .593, 5e-4) and close(1 - .125 / (sx4 / sx2 ** 2), .41, .005), (sx4 / sx2 ** 2, .125 / (sx4 / sx2 ** 2)))
# E[s²/Σx²]：E[SSE]=σ² tr((I−H)Σ)
xx = [F(int(v)) for v in xh]
trIHS = sum(v ** 2 for v in xx) - sum(v ** 4 for v in xx) / sum(v ** 2 for v in xx)
Es = trIHS / 7 / 204
check(T, "例2 E[s²/Σx²]=23σ²/204=0.1127σ²", Es == F(23, 204) and close(Es, .1127), Es)
check(T, "例2 0.1127 は 0.2108 の約半分", 0.5 < .1127 / .2108 < 0.56, .1127 / .2108)
# WLS（重み 1/x²）が mean(y/x) になること、その分散 σ²/n、最良性（sim）
xw = sp.symbols("x1:4", positive=True)
yw = sp.symbols("y1:4")
bw = sum(yw[i] / xw[i] for i in range(3)) / 3
num_w = sum(yw[i] * xw[i] / xw[i] ** 2 for i in range(3))
den_w = sum(xw[i] ** 2 / xw[i] ** 2 for i in range(3))
check(T, "WLS（w=1/x²）の原点回帰 = (1/n)Σy/x（sympy）", sp.simplify(num_w / den_w - bw) == 0, "OK")
Rh = 200000
eh = rng.normal(size=(Rh, 8)) * xh
yy_ = 2 * xh + eh
bols = yy_ @ xh / sx2
bwls = (yy_ / xh).mean(1)
check(T, "sim OLS 分散 0.2108、WLS 0.125（σ=1、相対 ±2%）、両者とも不偏", abs(bols.var() / .2108 - 1) < .02 and abs(bwls.var() / .125 - 1) < .02 and abs(bols.mean() - 2) < .005 and abs(bwls.mean() - 2) < .005, (bols.var(), bwls.var()))
s2h = ((yy_ - bols[:, None] * xh) ** 2).sum(1) / 7
check(T, "sim E[s²/Σx²]=0.1127（相対 ±2%）", abs((s2h / sx2).mean() / .1127 - 1) < .02, (s2h / sx2).mean())
# サンドイッチ型の分散（sim）
Xh = dh["X"]
Sig = np.diag(xh ** 2)
XtXih = np.linalg.inv(Xh.T @ Xh)
Vsand = XtXih @ Xh.T @ Sig @ Xh @ XtXih
Bh = (yy_ @ Xh) @ XtXih  # 真の β=(0,2)
check(T, "sim V[β̂]=(XᵀX)⁻¹XᵀΣX(XᵀX)⁻¹（相対 ±3%）", np.max(np.abs(np.cov(Bh.T) / Vsand - 1)) < .03, (np.cov(Bh.T).round(4).tolist(), Vsand.round(4).tolist()))
ch = charts(T)
check(T, "図1 residual-pattern funnel", ch[0]["params"]["pattern"] == "funnel", ch[0]["params"])
check(T, "図2 line-data = 例1 の残差", allclose(dh["e"], ch[1]["params"]["series"][0]["y"]) and ch[1]["params"]["x"] == list(range(1, 9)), ch[1]["params"]["series"][0]["y"])

print()
print(f"OK {sum(results)} / NG {len(results) - sum(results)}")
