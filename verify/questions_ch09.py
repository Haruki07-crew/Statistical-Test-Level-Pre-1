"""第9章（区間推定）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
from math import comb
from scipy import stats, optimize

q = Q(9)
z = 1.96
rng = np.random.default_rng(9)

# --- 信頼区間 ---
q.concept("q09-confidence-interval-1", "信頼区間の解釈（数値なし）")
q.num("q09-confidence-interval-2", 0.95 ** 20)

# --- 信頼係数 ---
zs = {"90両側": stats.norm.ppf(0.95), "片側90": stats.norm.ppf(0.9), "95両側": stats.norm.ppf(0.975), "上側1%": stats.norm.ppf(0.99)}
q.check("q09-confidence-coefficient-1", abs(zs["90両側"] - 1.645) < 5e-4 and abs(zs["片側90"] - 1.282) < 5e-4
        and abs(zs["95両側"] - 1.960) < 5e-4 and abs(zs["上側1%"] - 2.326) < 5e-4, "正解 1.645、誤答はそれぞれ別の点")
n, p = 10, 0.5
inside = []
for x in range(n + 1):
    ph = x / n
    h = z * np.sqrt(ph * (1 - ph) / n)
    if ph - h <= p <= ph + h:
        inside.append(x)
cov = sum(stats.binom.pmf(x, n, p) for x in inside)
q.num("q09-confidence-coefficient-2", cov)
q.check("q09-confidence-coefficient-2", inside == list(range(3, 8)) and abs(cov - (1 - 2 * 56 / 1024)) < 1e-12
        and abs(0.2 + z * np.sqrt(0.16 / 10) - 0.448) < 5e-4, "x=3..7 で含む、1-112/1024")

# --- ピボット量 ---
sim = rng.normal(5, 3, (100000, 6))
tq = (sim.mean(1) - 5) / (sim.std(1, ddof=1) / np.sqrt(6))
q.check("q09-pivotal-quantity-1", abs(np.mean(np.abs(tq) <= stats.t.ppf(0.975, 5)) - 0.95) < 0.005, "t 型ピボットの被覆 0.95")
q.check("q09-pivotal-quantity-2", abs(stats.chi2.ppf(0.025, 10) - 3.247) < 5e-4 and abs(stats.chi2.ppf(0.975, 10) - 20.483) < 5e-4,
        "問題文の χ² 点")
q.num("q09-pivotal-quantity-2", 24 / stats.chi2.ppf(0.025, 10))

# --- 分散既知 ---
q.num("q09-ci-mean-known-variance-1", 52.3 - z * 4 / np.sqrt(16))
r = stats.norm.ppf(0.995) / stats.norm.ppf(0.975)
q.check("q09-ci-mean-known-variance-2", abs(r - 1.314) < 5e-4, "幅の比 1.314（正しい）")

# --- 分散未知 ---
tv = [stats.t.ppf(0.975, 9), stats.t.ppf(0.975, 10), stats.norm.ppf(0.975), stats.t.ppf(0.95, 9)]
q.check("q09-ci-mean-unknown-variance-1", np.allclose(tv, [2.262, 2.228, 1.960, 1.833], atol=5e-4), "正解 2.262、誤答の値も確認")
x = np.array([12, 15, 11, 14, 13.0])
q.num("q09-ci-mean-unknown-variance-2", stats.t.ppf(0.975, 4) * x.std(ddof=1) / np.sqrt(5))

# --- 母分散 ---
lo, hi = 36 / stats.chi2.ppf(0.975, 9), 36 / stats.chi2.ppf(0.025, 9)
q.check("q09-ci-variance-1", abs(lo - 1.892) < 1e-3 and abs(hi - 13.33) < 5e-3 and abs((4 - lo) - (hi - 4)) > 1, "非対称（誤り）")
q.num("q09-ci-variance-2", hi)
q.num("q09-ci-variance-3", np.sqrt(30 / stats.chi2.ppf(0.975, 6)))

# --- 母比率 ---
q.num("q09-ci-proportion-1", 0.3 + z * np.sqrt(0.21 / 400))
q.check("q09-ci-proportion-2", 0 + z * np.sqrt(0 * 1 / 30) == 0 and abs(stats.beta.ppf(0.975, 1, 30) - 0.116) < 5e-4,
        "幅0（正しい）、CP 上限 0.116")

# --- ワルド ---
q.num("q09-wald-confidence-interval-1", 2.5 + z * np.sqrt(2.5 / 40))
q.num("q09-wald-confidence-interval-2", 0.4 - z * 0.4 / np.sqrt(50))

# --- ウィルソン ---
n, ph = 20, 0.1
c = (ph + z * z / (2 * n)) / (1 + z * z / n)
h = z / (1 + z * z / n) * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
q.num("q09-wilson-interval-1", c + h)
# スコア検定の反転として解いても同じ
f = lambda pp: abs(ph - pp) / np.sqrt(pp * (1 - pp) / n) - z
q.check("q09-wilson-interval-1", abs(optimize.brentq(f, 0.11, 0.99) - (c + h)) < 1e-9 and abs(c - 0.1645) < 5e-4
        and abs(h - 0.1366) < 5e-4 and abs(ph - z * np.sqrt(ph * (1 - ph) / n) + 0.031) < 5e-4, "スコア反転と一致、解説の値")
q.check("q09-wilson-interval-2", abs(c - ph) > 0.05 and abs(c - (n / (n + z * z) * ph + z * z / (n + z * z) * 0.5)) < 1e-12,
        "中心は p̂ と 1/2 の加重平均（誤り）")

# --- 2標本平均差 ---
q.num("q09-ci-two-sample-mean-difference-1", z * np.sqrt(100 / 25 + 144 / 36))
sp2 = (7 * 6 + 9 * 9) / 16
q.num("q09-ci-two-sample-mean-difference-2", 3 - stats.t.ppf(0.975, 16) * np.sqrt(sp2 * (1 / 8 + 1 / 10)))

# --- 対応あり ---
d = np.array([1.2, 0.8, 1.5, 0.3, 1.0, 0.6])
q.num("q09-ci-paired-difference-1", d.mean() - stats.t.ppf(0.975, 5) * d.std(ddof=1) / np.sqrt(6))
q.check("q09-ci-paired-difference-1", abs(((d - d.mean()) ** 2).sum() - 0.92) < 1e-12, "偏差平方和 0.92")

# --- 分散比 ---
sim1 = rng.normal(0, 2, (100000, 11)).var(1, ddof=1)
sim2 = rng.normal(0, 1, (100000, 16)).var(1, ddof=1)
ratio = (sim1 / 4) / (sim2 / 1)
q.check("q09-ci-variance-ratio-1", abs(np.mean(ratio > stats.f.ppf(0.975, 10, 15)) - 0.025) < 0.002
        and abs(np.mean(ratio > stats.f.ppf(0.975, 15, 10)) - 0.025) > 0.005, "F(10,15) が正しく、(15,10) では外れる")
q.num("q09-ci-variance-ratio-2", 2 * stats.f.ppf(0.975, 15, 10))
q.check("q09-ci-variance-ratio-2", abs(2 / stats.f.ppf(0.975, 10, 15) - 0.654) < 5e-4
        and abs(1 / stats.f.ppf(0.025, 10, 15) - stats.f.ppf(0.975, 15, 10)) < 1e-9, "下限0.654、F の逆数関係")

# --- 標本サイズ ---
q.num("q09-required-sample-size-1", int(np.ceil((z * 10 / 2) ** 2)))
q.num("q09-required-sample-size-2", int(np.ceil(z ** 2 * 0.25 / 0.03 ** 2)))
q.check("q09-required-sample-size-2", int(np.ceil(z ** 2 * 0.16 / 0.03 ** 2)) == 683, "p=0.2 なら 683")
q.check("q09-required-sample-size-3", abs(1 / np.sqrt(2) - 0.707) < 1e-3 and 1 / np.sqrt(4) == 0.5, "4倍で半分（誤り）")

# --- 2標本比率差 ---
q.num("q09-ci-two-sample-proportion-1", 0.1 - z * np.sqrt(0.21 / 200 + 0.16 / 250))

# --- クロッパー・ピアソン ---
pu = 1 - 0.025 ** (1 / 20)
q.num("q09-clopper-pearson-interval-1", pu)
q.check("q09-clopper-pearson-interval-1", abs(stats.beta.ppf(0.975, 1, 20) - pu) < 1e-12
        and abs(stats.binom.cdf(0, 20, pu) - 0.025) < 1e-12 and abs(1 - 0.05 ** (1 / 20) - 0.139) < 5e-4, "ベータ分位点と一致")

# --- 尤度に基づく区間 ---
l = lambda lam: 20 * np.log(lam) - 10 * lam
f = lambda lam: 2 * (l(2) - l(lam)) - stats.chi2.ppf(0.95, 1)
lo, hi = optimize.brentq(f, 0.5, 2), optimize.brentq(f, 2, 6)
q.num("q09-likelihood-based-confidence-interval-1", lo)
q.check("q09-likelihood-based-confidence-interval-1", abs(hi - 3.009) < 5e-4 and abs(2 - z * np.sqrt(0.2) - 1.12) < 5e-3, "上限3.009、ワルド1.12")

# --- 双対性 ---
q.check("q09-confidence-interval-test-duality-1", not (1.2 <= 4 <= 3.8) and (1.2 <= 3 <= 3.8), "4 は区間外→棄却（正しい）")

# --- 片側信頼限界（用語の見直しで追加） ---
from scipy import stats as _st
q.check("q09-one-sided-confidence-limit-1", abs(_st.t.ppf(0.95, 15) - 1.753) < 5e-4 and abs(_st.t.ppf(0.975, 15) - 2.131) < 5e-4, "t_0.05(15)=1.753、t_0.025(15)=2.131")
q.num("q09-one-sided-confidence-limit-1", 50 - _st.t.ppf(0.95, 15) * 8 / 4)
q.check("q09-one-sided-confidence-limit-1", abs(50 - 2.131 * 2 - 45.74) < 0.005, "両側区間の下限 45.74")
q.check("q09-one-sided-confidence-limit-2", abs(_st.chi2.ppf(0.05, 11) - 4.575) < 5e-4 and abs(_st.chi2.ppf(0.95, 11) - 19.675) < 5e-4, "χ² 点 4.575 / 19.675")
q.num("q09-one-sided-confidence-limit-2", 11 * 2.5 / _st.chi2.ppf(0.05, 11))
q.check("q09-one-sided-confidence-limit-2", abs(11 * 2.5 / 19.675 - 1.40) < 0.005, "19.675 で割ると下側限界 1.40")

# --- 導出問題（追加） ---
def _deriv09():
    import numpy as _N
    import sympy as _S
    from scipy import stats as _st
    lam, t, x = _S.symbols("lambda t x", positive=True)
    M = _S.simplify(_S.integrate(lam * _S.exp(-lam * x) * _S.exp(t * x), (x, 0, _S.oo), conds="none"))
    ok = _S.simplify(M - lam / (lam - t)) == 0
    ok &= _S.simplify(M.subs(t, 2 * lam * t) - 1 / (1 - 2 * t)) == 0
    # 正確な被覆確率：Σ X ~ Gamma(n, 1/λ) なので Q=2λΣX ~ χ²(2n)
    ok2 = True
    for n, L in [(5, 0.7), (12, 3.0)]:
        G = _st.gamma(n, scale=1 / L)
        qq = _N.linspace(0.5, 40, 50)
        ok2 &= _N.allclose(G.cdf(qq / (2 * L)), _st.chi2(2 * n).cdf(qq))
        lo, hi = _st.chi2(2 * n).ppf(0.025), _st.chi2(2 * n).ppf(0.975)   # 上側97.5%点・上側2.5%点
        # λ ∈ [lo/(2S), hi/(2S)] ⇔ lo/(2λ) ≤ S ≤ hi/(2λ)
        cover = G.cdf(hi / (2 * L)) - G.cdf(lo / (2 * L))
        ok2 &= abs(cover - 0.95) < 1e-12
    q.check("q09-pivotal-quantity-3", bool(ok and ok2), "sympy: M_{2λX}(t)=(1-2t)^-1、2λΣX~χ²(2n)、区間の被覆確率がちょうど0.95")
_deriv09()


# --- 4択の追加 ---
def _choice09():
    import math as _m
    from scipy import stats as _st
    def same(a, b, d):
        return all(abs(x - y) < 0.5 * 10 ** (-d) + 1e-12 for x, y in zip(a, b))
    # 既知分散の 90% 区間
    z05 = _st.norm.isf(0.05)
    se = 6 / _m.sqrt(36)
    ci = (50 - z05 * se, 50 + z05 * se)
    ch = [(48.04, 51.96), (48.36, 51.64), (40.13, 59.87), (49.73, 50.27)]
    ans = q.qs["q09-ci-mean-known-variance-3"]["answer"]
    q.check("q09-ci-mean-known-variance-3",
            same(ch[ans], ci, 2) and not any(same(c, ci, 2) for i, c in enumerate(ch) if i != ans),
            f"scipy: z0.05={z05:.4f}、区間 [{ci[0]:.3f}, {ci[1]:.3f}]、誤答は一致しない")
    w = [(50 - 1.96 * se, 50 + 1.96 * se), (50 - 1.645 * 6, 50 + 1.645 * 6), (50 - 1.645 / 6, 50 + 1.645 / 6)]
    q.check("q09-ci-mean-known-variance-3", all(same(ch[i], w[k], 2) for i, k in ((0, 0), (2, 1), (3, 2))),
            "誤答の由来：1.96 を使う／√n で割らない／n で割る")

    # 比率のワルド区間
    n, x = 300, 60
    ph = x / n
    z = _st.norm.isf(0.025)
    se = _m.sqrt(ph * (1 - ph) / n)
    ci = (ph - z * se, ph + z * se)
    ch = [(0.162, 0.238), (0.199, 0.201), (0.143, 0.257), (0.155, 0.245)]
    ans = q.qs["q09-ci-proportion-3"]["answer"]
    q.check("q09-ci-proportion-3",
            same(ch[ans], ci, 3) and not any(same(c, ci, 3) for i, c in enumerate(ch) if i != ans),
            f"区間 [{ci[0]:.4f}, {ci[1]:.4f}]、誤答は一致しない")
    w = {0: 1.645 * se, 1: 1.96 * ph * (1 - ph) / n, 2: 1.96 * _m.sqrt(0.25 / n)}
    q.check("q09-ci-proportion-3", all(same(ch[i], (ph - h, ph + h), 3) for i, h in w.items()),
            "誤答の由来：1.645 を使う／平方根を忘れる／p=0.5 を使う")
_choice09()

# --- 追加：答えが「正しい」の正誤問題 ---
import numpy as _np
from scipy import stats as _st
_rng = _np.random.default_rng(909)
_m, _n, _R = 5, 7, 40000
_X = _rng.normal(3.0, 2.0, size=(_R, _m)); _Y = _rng.normal(1.0, 2.0, size=(_R, _n))
_sp2 = ((_m - 1) * _X.var(1, ddof=1) + (_n - 1) * _Y.var(1, ddof=1)) / (_m + _n - 2)
_T = (_X.mean(1) - _Y.mean(1) - 2.0) / _np.sqrt(_sp2 * (1 / _m + 1 / _n))
_ks = _st.kstest(_T, _st.t(_m + _n - 2).cdf).pvalue
_cov = _np.mean(_np.abs(_T) <= _st.t.ppf(0.975, _m + _n - 2))
q.check("q09-ci-two-sample-mean-difference-3", _ks > 0.01 and abs(_cov - 0.95) < 0.005,
        f"m=5,n=7 の乱数4万回で T は t(10) と整合（KS p={_ks:.3f}、95%区間の被覆率 {_cov:.4f}）")

q.done()
