"""統計学の教科書 第48章「統計的決定理論」の検算（厳密計算・数値積分・シミュレーション）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, optimize, stats as st

b = Book("tb", 48)
rng = np.random.default_rng(48)
n = 10
r1 = lambda p: p * (1 - p) / n
r2 = lambda p: (n * p * (1 - p) + (1 - 2 * p) ** 2) / (n + 2) ** 2
# 厳密なリスク（二項分布で和をとる）と式の一致
xs = np.arange(n + 1)
for p in (0.1, 0.37, 0.5):
    pm = st.binom.pmf(xs, n, p)
    e2 = (pm * ((xs + 1) / (n + 2) - p) ** 2).sum()
    b.check(f"δ2 のリスクの式（p={p}）", abs(e2 - r2(p)) < 1e-14, e2)
b.num("δ1 の最大リスク", r1(0.5), "0.025")
b.num("δ2 p=0.5", r2(0.5), "0.01736")
b.num("δ2 p=0", r2(0), "0.00694")
pc = optimize.brentq(lambda p: r1(p) - r2(p), 0.01, 0.4)
b.num("交点", pc, "0.138")
b.num("交点（対称）", 1 - pc, "0.862")
b.check("p(1−p)=1/8.4", abs(pc * (1 - pc) - 1 / 8.4) < 1e-12, "")
# 図のデータ
c = b.chart(1)
ps = np.array(c["x"])
ser = c["series"]
b.check("図: δ1 のリスク", np.allclose(ser[0]["y"], r1(ps), atol=6e-6), "")
b.check("図: δ2 のリスク", np.allclose(ser[1]["y"], r2(ps), atol=6e-6), "")
# ベイズリスク
br2 = integrate.quad(r2, 0, 1)[0]
b.num("δ2 のベイズリスク", br2, "0.0139")
b.num("δ1 のベイズリスク", integrate.quad(r1, 0, 1)[0], "0.0167")
# ミニマックス
a = math.sqrt(n) / 2
b.num("a", a, "1.58")
cr = n / (4 * (n + math.sqrt(n)) ** 2)
b.num("ミニマックスのリスク", cr, "0.01443")
for p in (0, 0.2, 0.5, 0.9):
    pm = st.binom.pmf(xs, n, p)
    e3 = (pm * ((xs + a) / (n + 2 * a) - p) ** 2).sum()
    b.check(f"δ3 のリスクは一定（p={p}）", abs(e3 - cr) < 1e-14, e3)
b.check("図: ミニマックスのリスク", np.allclose(ser[2]["y"], cr, atol=6e-6), "")
# ジェームズ・スタイン
def js_risk(p, lam):
    k = np.arange(0, 400)
    w = st.poisson.pmf(k, lam / 2)
    return p - (p - 2) ** 2 * (w / (p + 2 * k - 2)).sum()
b.num("JS p=10, θ=0", js_risk(10, 0), "2")
b.num("JS ‖θ‖²=10", js_risk(10, 10), "6.2")
b.num("JS ‖θ‖²=40", js_risk(10, 40), "8.6")
for lam in (0, 10, 40):
    th = np.zeros(10); th[0] = math.sqrt(lam)
    X = rng.standard_normal((200000, 10)) + th
    s = (X ** 2).sum(1)
    js = (1 - 8 / s)[:, None] * X
    b.check(f"JS シミュレーション ‖θ‖²={lam}", abs(((js - th) ** 2).sum(1).mean() - js_risk(10, lam)) < 0.03, "")
X = rng.standard_normal((400000, 10))
s = (X ** 2).sum(1)
jsp = np.maximum(1 - 8 / s, 0)[:, None] * X
b.num("正部分 JS", ((jsp) ** 2).sum(1).mean(), "1.26", tol=0.01)
b.check("E[1/χ²_10]=1/8", abs(integrate.quad(lambda x: st.chi2.pdf(x, 10) / x, 0, np.inf)[0] - 1 / 8) < 1e-9, "")
# 章末問題
b.num("問2 δ1", r1(0.2), "0.016")
b.num("問2 δ2", r2(0.2), "0.0136")
b.check("問2 1.96/144", abs(r2(0.2) - 1.96 / 144) < 1e-15, "")
b.num("問3", js_risk(5, 0), "2")
b.done()
