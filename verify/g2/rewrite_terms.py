"""2級 準1級と共通だった用語を2級向けに書き直したものの検算（具体例・本文の数値）。"""
import sys
from math import sqrt
import numpy as np
from scipy import stats

results = []
rng = np.random.default_rng(25)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-4):
    return abs(float(x) - claim) <= tol


T = "central-limit-theorem"
check(T, "例1 SD 4/√64=0.5・P(Z>2)=0.0228", 4 / sqrt(64) == 0.5 and close(stats.norm.sf((51 - 50) / 0.5), 0.0228, 5e-5), stats.norm.sf(2))
check(T, "例2 合計の平均 20 万・SD 15000・z≈0.67・表の値 0.2514", 100 * 2000 == 200000 and sqrt(100) * 1500 == 15000 and close(10000 / 15000, 0.67, 0.005) and close(stats.norm.sf(0.67), 0.2514, 5e-5), stats.norm.sf(0.67))
k, th = (2000 / 1500) ** 2, 1500 ** 2 / 2000
check(T, "例2 ガンマ分布（平均 2000・SD 1500）の 100 人の合計で正確に約 0.248", close(stats.gamma(100 * k, scale=th).sf(210000), 0.248, 5e-4) and close(k * th, 2000, 1e-9) and close(sqrt(k) * th, 1500, 1e-9), stats.gamma(100 * k, scale=th).sf(210000))
m = rng.exponential(1, (100000, 30)).mean(1)
check(T, "図：指数分布でも n=30 の標本平均の歪度は小さい（元の分布の歪度 2）", abs(stats.skew(m) - 2 / sqrt(30)) < 0.05, stats.skew(m))

T = "normal-distribution"
check(T, "±1σ・±2σ・±3σ に約 68%・95%・99.7%", all(close(stats.norm.cdf(k) - stats.norm.cdf(-k), v, 5e-4) for k, v in ((1, .6827), (2, .9545), (3, .9973))), None)
check(T, "華氏 1.8X+32：N(20,3²) → N(68, 5.4²)", close(1.8 * 20 + 32, 68, 1e-9) and close(1.8 * 3, 5.4, 1e-9), None)
check(T, "例1 規格 97〜103：1−2×0.0668=0.8664（正確な値 0.86639）", close(1 - 2 * 0.0668, 0.8664, 1e-9) and close(stats.norm(100, 2).cdf(103) - stats.norm(100, 2).cdf(97), 0.8664, 1e-4), stats.norm(100, 2).cdf(103) - stats.norm(100, 2).cdf(97))
check(T, "例1 平均が 101 にずれると 1−0.0228−0.1587=0.8185（規格外 約 18%）", close(1 - 0.0228 - 0.1587, 0.8185, 1e-9) and close(stats.norm(101, 2).cdf(103) - stats.norm(101, 2).cdf(97), 0.8185, 2e-4), stats.norm(101, 2).cdf(103) - stats.norm(101, 2).cdf(97))
AB = rng.normal(60, 1.2, 1_000_000) + rng.normal(40, 1.6, 1_000_000)
check(T, "例2 A+B ~ N(100, 2²)・P(>103)=0.0668（シミュレーション）", close(1.2 ** 2 + 1.6 ** 2, 4, 1e-9) and close(stats.norm(100, 2).sf(103), 0.0668, 5e-5) and close(AB.std(), 2, 0.01) and close(np.mean(AB > 103), 0.0668, 0.002), (AB.std(), np.mean(AB > 103)))
T = "standard-normal-distribution"
check(T, "上側点 1.282・1.645・1.960・2.326・2.576", all(close(stats.norm.isf(a), z) for a, z in ((.1, 1.282), (.05, 1.645), (.025, 1.960), (.01, 2.326), (.005, 2.576))), None)
check(T, "例1 P(Z>1.25)=0.1056・0.3944・0.7888（表の値から）", close(stats.norm.sf(1.25), 0.1056, 5e-5) and close(0.5 - 0.1056, 0.3944, 1e-9) and close(1 - 2 * 0.1056, 0.7888, 1e-9) and close(1 - 2 * stats.norm.sf(1.25), 0.7888, 2e-4), None)
check(T, "例2 上位 10% の境目 72.8・下位 10% 47.2", close(60 + 1.282 * 10, 72.8, 0.05) and close(stats.norm(60, 10).isf(.1), 72.8, 0.05) and close(stats.norm(60, 10).ppf(.1), 47.2, 0.05), stats.norm(60, 10).isf(.1))
check(T, "|z|>3 は両側でおよそ 0.3%・P(Z<1)=0.8413", close(2 * stats.norm.sf(3), 0.0027, 1e-4) and close(1 - 0.1587, 0.8413, 1e-9), None)

ng = results.count(False)
print(f"書き直した共通の用語の検算: OK {len(results) - ng} / NG {ng}")
sys.exit(1 if ng else 0)
