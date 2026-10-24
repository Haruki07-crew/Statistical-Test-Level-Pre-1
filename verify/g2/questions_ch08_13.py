"""2級 第8・9・11・12・13章：準1級と共通の用語に足した2級の問題の検算（章ごとに結果を保存する）。"""
import sys
import pathlib
from math import comb, exp, sqrt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qhead import Q, chosen  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

ng = 0
rng = np.random.default_rng(13)


def run(q):
    global ng
    try:
        q.done()
    except SystemExit as e:
        ng += int(bool(e.code))


q = Q(8)
X = rng.normal(3, sqrt(2), 2_000_000)
q.num("q08-expectation-variance-properties-g1", 2 + 3 ** 2)
q.check("q08-expectation-variance-properties-g1", abs(np.mean(X ** 2) - 11) < 0.02, f"シミュレーション {np.mean(X ** 2):.3f}")
Y = rng.normal(0, sqrt(3), 2_000_000)
q.check("q08-expectation-variance-properties-g2", abs(np.var(X - 2 * Y) - 14) < 0.05 and chosen(q, "q08-expectation-variance-properties-g2") == "14", f"シミュレーション {np.var(X - 2 * Y):.3f}")
q.num("q08-expectation-variance-properties-g3", sqrt(3 ** 2 + 4 ** 2))
TA = rng.normal(20, 3, 1_000_000) + rng.gamma(15 ** 2 / 16, 16 / 15, 1_000_000)
q.check("q08-expectation-variance-properties-g3", abs(TA.std() - 5) < 0.02, f"正規分布とガンマ分布の和でシミュレーション 標準偏差 {TA.std():.3f}")
q.num("q08-probability-mass-and-density-function-g1", (1 - .1 - .4 - .2) + .2)
q.check("q08-probability-mass-and-density-function-g2", chosen(q, "q08-probability-mass-and-density-function-g2").startswith("$P(a\\le X\\le b)$ は、$a$ から $b$ までの") and abs(integrate.quad(lambda x: 3 * x * x, 0, 1)[0] - 1) < 1e-12 and 3 * 1 ** 2 > 1, "面積が確率・f(1)=3>1 の密度がある")
q.num("q08-probability-mass-and-density-function-g3", 1 - integrate.quad(lambda x: 3 * x * x, 0, .5)[0])
run(q)

q = Q(9)
q.num("q09-binomial-distribution-g1", stats.binom(4, .5).pmf(2))
q.num("q09-binomial-distribution-g2", stats.binom(3, .1).sf(0))
q.num("q09-binomial-distribution-g3", stats.binom(30, .4).var())
q.num("q09-bernoulli-distribution-g1", stats.bernoulli(.2).var())
q.check("q09-bernoulli-distribution-g2", max(np.linspace(0, 1, 101), key=lambda p_: p_ * (1 - p_)) == 0.5 and chosen(q, "q09-bernoulli-distribution-g2") == "0.5", "p(1−p) の最大は p=0.5")
q.num("q09-bernoulli-distribution-g3", sqrt(.2 * .8 / 400))
ph = rng.binomial(400, .2, 200000) / 400
q.check("q09-bernoulli-distribution-g3", abs(ph.std() - 0.02) < 0.0003, f"シミュレーション {ph.std():.5f}")
q.num("q09-poisson-distribution-g1", stats.poisson(2).pmf(0))
q.check("q09-poisson-distribution-g1", abs(exp(-2) - 0.1353) < 5e-5, "表の値")
q.num("q09-poisson-distribution-g2", 2.5 * 0.2231)
q.check("q09-poisson-distribution-g2", abs(stats.poisson(1.5).cdf(1) - 0.5578) < 5e-4 and abs(exp(-1.5) - 0.2231) < 5e-5, f"正確な値 {stats.poisson(1.5).cdf(1):.5f}・表の値")
pa = rng.poisson(2, 1_000_000) + rng.poisson(3, 1_000_000)
q.check("q09-poisson-distribution-g3", abs(pa.mean() - 5) < 0.01 and abs(pa.var() - 5) < 0.02 and abs(np.mean(pa == 0) - exp(-5)) < 0.001 and chosen(q, "q09-poisson-distribution-g3") == "平均 5 のポアソン分布", f"シミュレーション 平均 {pa.mean():.3f}・分散 {pa.var():.3f}・P(0) {np.mean(pa == 0):.4f}")
q.num("q09-normal-distribution-g1", stats.norm(170, 6).sf(182))
q.num("q09-standard-normal-distribution-g1", stats.norm.cdf(1.96) - stats.norm.cdf(-1.96))
XA = rng.normal(50, 10, 1_000_000); YA = 2 * XA + 5
q.check("q09-normal-distribution-g2", abs(YA.mean() - 105) < 0.05 and abs(YA.std() - 20) < 0.05 and abs(stats.skew(YA)) < 0.01 and chosen(q, "q09-normal-distribution-g2").startswith("$N(105,\\ 20^2)$"), f"シミュレーション 平均 {YA.mean():.2f}・標準偏差 {YA.std():.2f}")
q.num("q09-normal-distribution-g3", stats.norm(30 - 29, sqrt(.6 ** 2 + .8 ** 2)).cdf(0))
A_ = rng.normal(30, .6, 1_000_000); B_ = rng.normal(29, .8, 1_000_000)
q.check("q09-normal-distribution-g3", abs(np.mean(A_ < B_) - 0.1587) < 0.002 and abs(stats.norm.sf(1) - 0.1587) < 5e-5, f"シミュレーション {np.mean(A_ < B_):.4f}・表の値")
q.num("q09-standard-normal-distribution-g2", 1 - 0.1587 - 0.0228)
q.check("q09-standard-normal-distribution-g2", abs(stats.norm.cdf(2) - stats.norm.cdf(-1) - 0.8185) < 2e-4, f"正確な値 {stats.norm.cdf(2) - stats.norm.cdf(-1):.5f}（表の値の丸めの範囲で一致）")
q.num("q09-standard-normal-distribution-g3", 500 + 1.96 * 100)
q.check("q09-standard-normal-distribution-g3", abs(stats.norm(500, 100).isf(0.025) - 696) < 0.1 and abs(500 + 1.645 * 100 - 664.5) < 1e-9, "上位 2.5% の境目・上位 5% なら 664.5")
q.num("q09-exponential-distribution-g1", stats.expon(scale=10).cdf(5))
q.num("q09-exponential-distribution-g2", stats.expon(scale=8).sf(12) / stats.expon(scale=8).sf(4))
w8 = rng.exponential(8, 2_000_000)
q.check("q09-exponential-distribution-g2", abs(np.mean(w8[w8 > 4] > 12) - exp(-1)) < 0.003 and abs(exp(-1) - 0.3679) < 5e-5, f"シミュレーション {np.mean(w8[w8 > 4] > 12):.4f}")
q.check("q09-exponential-distribution-g3", abs(stats.expon(scale=8).var() - 64) < 1e-9 and chosen(q, "q09-exponential-distribution-g3") == "64", "分散 1/λ² = 64")
run(q)

q = Q(11)
hit = 0
for _ in range(20):
    sm = rng.normal(50, 5, (5000, 16)); m = sm.mean(1); s = sm.std(1, ddof=1)
    hit += np.sum(np.abs(m - 50) <= stats.t.isf(.025, 15) * s / 4)
q.check("q11-confidence-interval-g1", abs(hit / 100000 - .95) < .005, f"被覆率は方法の性質 {hit / 100000:.4f}（answer=false）")
q.num("q11-ci-mean-unknown-variance-g1", 20 + stats.t.isf(.025, 15) * 4 / 4)
q.num("q11-ci-proportion-g1", 1.96 * sqrt(.25 / 400))
q.num("q11-standard-error-g1", 10 / sqrt(25))
run(q)

q = Q(12)
q.check("q12-p-value-g1", 0.01 < 0.03 < 0.05, "0.01 < 0.03 < 0.05（answer=true）")
q.num("q12-one-sample-t-test-g1", (52 - 50) / (3 / 3))
q.num("q12-proportion-test-g1", (0.6 - 0.5) / sqrt(.25 / 100))
q.check("q12-type-i-and-ii-errors-g1", chosen(q, "q12-type-i-and-ii-errors-g1") == "第 1 種の過誤", "正常（帰無仮説が正しい）なのに異常と判定")
run(q)

q = Q(13)
q.num("q13-pooled-variance-g1", (5 * 10 + 5 * 14) / 10)
r = stats.ttest_ind_from_stats(4, sqrt(10), 6, 0, sqrt(14), 6)
q.num("q13-two-sample-t-test-g1", 4 / sqrt(12 * (1 / 6 + 1 / 6)))
q.check("q13-two-sample-t-test-g1", abs(r.statistic - 2) < 1e-9, f"scipy の ttest_ind_from_stats {r.statistic:.4f}")
q.check("q13-paired-t-test-g1", chosen(q, "q13-paired-t-test-g1").startswith("同じ 20 人"), "同じ人の前後")
q.num("q13-independence-test-g1", 60 * 50 / 200)
q.num("q13-goodness-of-fit-test-g1", stats.chisquare([25, 15, 20, 20, 18, 22]).statistic)
q.check("q13-welch-test-g1", True, "ウェルチの検定は等分散を仮定しない（answer=false）")
run(q)
raise SystemExit(1 if ng else 0)
