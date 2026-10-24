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

T = "bernoulli-distribution"
check(T, "例1 E=0.04・V=0.0384", close(stats.bernoulli(.04).mean(), .04, 1e-12) and close(stats.bernoulli(.04).var(), .0384, 1e-12), None)
check(T, "例2 0/1 データの平均 0.3", close(np.mean([1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), 0.3, 1e-12), None)
check(T, "例3 p=0.5 で 0.25（最大）・p=0.1 と 0.9 で 0.09", close(.5 * .5, .25, 1e-12) and close(.1 * .9, .09, 1e-12) and close(.9 * .1, .09, 1e-12) and max(np.linspace(0, 1, 1001), key=lambda p: p * (1 - p)) == 0.5, None)
T = "binomial-distribution"
b5 = stats.binom(5, .25)
check(T, "例1 P(X≥3)=(90+15+1)/1024≈0.104・E=1.25・V=0.9375", close(b5.sf(2), 106 / 1024, 1e-12) and 10 * 9 == 90 and 5 * 3 == 15 and close(106 / 1024, 0.104, 5e-4) and close(b5.mean(), 1.25, 1e-12) and close(b5.var(), .9375, 1e-12), b5.sf(2))
check(T, "例2 B(50,0.02)：E=1・V=0.98・P(0)=0.98^50≈0.364・e^−1≈0.368", close(stats.binom(50, .02).var(), .98, 1e-12) and close(.98 ** 50, 0.364, 5e-4) and close(np.exp(-1), 0.368, 5e-4), .98 ** 50)
b10 = stats.binom(10, .3)
check(T, "最頻値 B(10,0.3)：P(2)≈0.233・P(3)≈0.267・P(4)≈0.200 で 3 が最大", close(b10.pmf(2), .233, 5e-4) and close(b10.pmf(3), .267, 5e-4) and close(b10.pmf(4), .200, 5e-4) and int(np.argmax(b10.pmf(range(11)))) == 3, None)
check(T, "少なくとも 1 回 1−(1−p)^n（n=3, p=0.1 で 0.271）", close(stats.binom(3, .1).sf(0), 1 - .9 ** 3, 1e-12), None)
T = "poisson-distribution"
p3 = stats.poisson(3)
check(T, "平均 = 分散 = λ（λ=3）", close(p3.mean(), 3, 1e-12) and close(p3.var(), 3, 1e-12), None)
check(T, "例1 P(0)=0.0498・P(1)≈0.149・P(2)≈0.224・P(≤2)=8.5×0.0498≈0.423", close(p3.pmf(0), .0498, 5e-5) and close(p3.pmf(1), .149, 5e-4) and close(p3.pmf(2), .224, 5e-4) and close(8.5 * .0498, .423, 5e-4) and close(p3.cdf(2), .423, 5e-4), p3.cdf(2))
check(T, "例2 B(300,0.01) の P(≤1)：近似 4e^−3≈0.199・正確 ≈0.198", close(4 * np.exp(-3), .199, 5e-4) and close(stats.binom(300, .01).cdf(1), .198, 5e-4), stats.binom(300, .01).cdf(1))
pp = rng.poisson(2, 1_000_000) + rng.poisson(3, 1_000_000); p2h = rng.poisson(3, (1_000_000, 2)).sum(1)
check(T, "和 Po(2)+Po(3)=Po(5)・2 時間なら Po(6)（シミュレーション）", close(pp.mean(), 5, 0.01) and close(pp.var(), 5, 0.02) and close(p2h.mean(), 6, 0.01) and close(p2h.var(), 6, 0.03), (pp.var(), p2h.var()))

T = "bayes-theorem"
pos = 0.01 * 0.95 + 0.99 * 0.10
check(T, "例1 表：病気 95/5・病気でない 990/8910・陽性 1085・事後確率 95/1085≈0.088", round(10000 * .01 * .95) == 95 and round(10000 * .99 * .10) == 990 and round(10000 * .99 * .90) == 8910 and 95 + 990 == 1085 and close(0.01 * 0.95 / pos, 95 / 1085, 1e-12) and close(95 / 1085, 0.088, 5e-4), 95 / 1085)
sick = rng.random(2_000_000) < 0.01; t1 = rng.random(2_000_000) < np.where(sick, .95, .10); t2 = rng.random(2_000_000) < np.where(sick, .95, .10)
check(T, "例2 2 回とも陽性なら約 0.48（0.088 を使った計算・正確な値・シミュレーション）", close(0.088 * .95 / (0.088 * .95 + 0.912 * .10), 0.48, 0.005) and close((95 / 1085) * .95 / ((95 / 1085) * .95 + (990 / 1085) * .10), 0.477, 5e-4) and close(sick[t1 & t2].mean(), 0.477, 0.01), sick[t1 & t2].mean())
check(T, "1 回目の陽性のシミュレーション ≈0.088", close(sick[t1].mean(), 0.0876, 0.004), sick[t1].mean())

T = "least-squares-method"
xs = np.array([1, 2, 3, 4, 5.]); ys = np.array([2, 4, 5, 4, 6.]); b1, a1 = np.polyfit(xs, ys, 1); e = ys - (a1 + b1 * xs)
check(T, "合計 15・21・71・55、Sxy=8・Sxx=10、b=0.8・a=1.8", (xs.sum(), ys.sum(), (xs * ys).sum(), (xs ** 2).sum()) == (15, 21, 71, 55) and close(71 - 5 * 3 * 4.2, 8, 1e-9) and close(55 - 45, 10, 1e-9) and close(b1, .8, 1e-9) and close(a1, 1.8, 1e-9), (b1, a1))
check(T, "予測値 2.6〜5.8・残差 −0.6, 0.6, 0.8, −1.0, 0.2・合計 0・残差平方和 2.4", np.allclose(a1 + b1 * xs, [2.6, 3.4, 4.2, 5.0, 5.8]) and np.allclose(e, [-.6, .6, .8, -1, .2]) and close(e.sum(), 0, 1e-9) and close((e ** 2).sum(), 2.4, 1e-9) and close((e * xs).sum(), 0, 1e-9), (e ** 2).sum())
check(T, "傾き 1 の直線 1.2+x は 2.8・図の値（平均の点を通る直線）", close(((ys - 1.2 - xs) ** 2).sum(), 2.8, 1e-9) and np.allclose([((ys - (4.2 - bb * 3) - bb * xs) ** 2).sum() for bb in np.arange(0, 1.61, .2)], [8.8, 6.0, 4.0, 2.8, 2.4, 2.8, 4.0, 6.0, 8.8]), None)
y12 = ys.copy(); y12[4] = 12
check(T, "外れ値 (5,12) で傾き 2.0", close(np.polyfit(xs, y12, 1)[0], 2.0, 1e-9), np.polyfit(xs, y12, 1)[0])

T = "multiple-regression-model"
check(T, "予測 30+1.5×20−5×2=50・単回帰の傾き −5+1.5×4=+1・自由度 40−2−1=37", 30 + 1.5 * 20 - 5 * 2 == 50 and -5 + 1.5 * 4 == 1 and 40 - 2 - 1 == 37, None)
ok = True
for seed in range(20):
    r_ = np.random.default_rng(seed); n_ = 40
    x2 = r_.integers(0, 6, n_).astype(float); x1 = 10 + 4 * x2 + r_.normal(0, 3, n_); y_ = 30 + 1.5 * x1 - 5 * x2 + r_.normal(0, 4, n_)
    bm = np.linalg.lstsq(np.c_[np.ones(n_), x1, x2], y_, rcond=None)[0]
    ok &= abs(np.polyfit(x2, y_, 1)[0] - (bm[2] + bm[1] * np.polyfit(x2, x1, 1)[0])) < 1e-9
check(T, "単回帰の傾き = b2 + b1 × d（最小二乗法で 20 通りのデータとも一致）", ok, None)
r_ = np.random.default_rng(7); x2 = r_.integers(0, 6, 100000).astype(float); x1 = 10 + 4 * x2 + r_.normal(0, 3, 100000); y_ = 30 + 1.5 * x1 - 5 * x2 + r_.normal(0, 4, 100000)
check(T, "具体例の形のデータで単回帰の傾き ≈ +1（正）", close(np.polyfit(x2, y_, 1)[0], 1.0, 0.05), np.polyfit(x2, y_, 1)[0])

ng = results.count(False)
print(f"書き直した共通の用語の検算: OK {len(results) - ng} / NG {ng}")
sys.exit(1 if ng else 0)
