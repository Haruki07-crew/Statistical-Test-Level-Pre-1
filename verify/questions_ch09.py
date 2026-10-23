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

q.done()
