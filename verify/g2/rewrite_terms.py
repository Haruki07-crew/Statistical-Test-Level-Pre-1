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

T = "normal-approximation-binomial-poisson"
b48 = stats.binom(48, .25)
check(T, "例1 平均 12・SD 3・(17.5−12)/3≈1.83・P=0.0336・正確 ≈0.037・修正なし 0.0228", close(b48.mean(), 12, 1e-12) and close(b48.std(), 3, 1e-12) and close((17.5 - 12) / 3, 1.83, 0.005) and close(stats.norm.sf(1.83), 0.0336, 5e-5) and close(b48.sf(17), 0.037, 5e-4) and close(stats.norm.sf(2), 0.0228, 5e-5), b48.sf(17))
p64 = stats.poisson(64)
check(T, "例2 SD 8・(79.5−64)/8≈1.94・P=0.0262・正確 ≈0.030", close((79.5 - 64) / 8, 1.94, 0.005) and close(stats.norm.sf(1.94), 0.0262, 5e-5) and close(p64.sf(79), 0.030, 5e-4), p64.sf(79))
check(T, "np=0.5 の B(50,0.01) は正規近似が不正確（P(0) 正確 0.605 と近似の差が大きい）", abs(stats.binom(50, .01).pmf(0) - (stats.norm(0.5, sqrt(50 * .01 * .99)).cdf(0.5))) > 0.05, stats.binom(50, .01).pmf(0))
T = "t-distribution"
check(T, "表 t0.025：5→2.571・9→2.262・15→2.131・20→2.086・30→2.042・60→2.000", all(close(stats.t.isf(.025, d), v) for d, v in ((5, 2.571), (9, 2.262), (15, 2.131), (20, 2.086), (30, 2.042), (60, 2.000))), None)
check(T, "例1 50±2.262×4/√10=50±2.86 → (47.14, 52.86)・1.96 なら (47.52, 52.48)", close(2.262 * 4 / sqrt(10), 2.86, 0.005) and close(50 - 2.262 * 4 / sqrt(10), 47.14, 0.005) and close(50 + 1.96 * 4 / sqrt(10), 52.48, 0.005), None)
check(T, "例2 自由度 9 で P(|T|>1.96)≈0.082", close(2 * stats.t.sf(1.96, 9), 0.082, 5e-4), 2 * stats.t.sf(1.96, 9))
T = "chi-squared-distribution"
z = rng.normal(0, 1, (400000, 4)); w = (z ** 2).sum(1)
check(T, "平均 k・分散 2k（k=4 のシミュレーション）", close(w.mean(), 4, 0.02) and close(w.var(), 8, 0.08), (w.mean(), w.var()))
check(T, "例1 自由度 9：16.92・19.02・2.700・10.02 と 6.30", close(stats.chi2.isf(.05, 9), 16.92, 0.005) and close(stats.chi2.isf(.025, 9), 19.02, 0.005) and close(stats.chi2.isf(.975, 9), 2.700) and close(19.02 - 9, 10.02, 1e-9) and close(9 - 2.7, 6.30, 1e-9), None)
s16 = rng.normal(0, 1, (200000, 16)).var(1, ddof=1)
check(T, "例2 自由度 15 の 5% 点 25.00・25/15≈1.67・シミュレーション", close(stats.chi2.isf(.05, 15), 25.00, 0.005) and close(25 / 15, 1.67, 0.005) and close(np.mean(s16 > 25 / 15), 0.05, 0.003), np.mean(s16 > 25 / 15))
check(T, "例3 (72/19.02, 72/2.700)≈(3.79, 26.67)", close(72 / 19.02, 3.79, 0.005) and close(72 / 2.700, 26.67, 0.005), None)
T = "f-distribution"
check(T, "例1 F=2.5 < F0.025(9,11)=3.588", close(30 / 12, 2.5, 1e-12) and close(stats.f.isf(.025, 9, 11), 3.588) and 2.5 < 3.588, None)
check(T, "例2 F0.025(11,9)=3.912・1/3.912≈0.256=F0.975(9,11)", close(stats.f.isf(.025, 11, 9), 3.912) and close(1 / 3.912, 0.256, 5e-4) and close(stats.f.isf(.975, 9, 11), 0.256, 5e-4), None)
check(T, "例3 F0.05(3,16)=3.239", close(stats.f.isf(.05, 3, 16), 3.239), None)
check(T, "t² と F(1,10)：2.228²≈4.96", close(stats.f.isf(.05, 1, 10), stats.t.isf(.025, 10) ** 2, 1e-9) and close(2.228 ** 2, 4.96, 0.005), None)

T = "probability-mass-and-density-function"
c3 = stats.binom(3, .5)
check(T, "例1 硬貨 3 枚：1/8・3/8・3/8・1/8・P(X≥2)=1/2・期待値 1.5", np.allclose(c3.pmf(range(4)), [1 / 8, 3 / 8, 3 / 8, 1 / 8]) and close(c3.sf(1), .5, 1e-12) and close(c3.mean(), 1.5, 1e-12), None)
from scipy import integrate
check(T, "例2 3x²：面積 1・P(X≤0.5)=0.125・E=3/4・f(1)=3", close(integrate.quad(lambda x: 3 * x * x, 0, 1)[0], 1, 1e-12) and close(integrate.quad(lambda x: 3 * x * x, 0, .5)[0], .125, 1e-12) and close(integrate.quad(lambda x: 3 * x ** 3, 0, 1)[0], .75, 1e-12), None)
check(T, "正規分布で σ<1/√(2π)≈0.399 なら山の高さが 1 を超える", stats.norm(0, .39).pdf(0) > 1 > stats.norm(0, .41).pdf(0), None)
T = "expectation-variance-properties"
check(T, "例1 1.2X+10：E=82・V=144・SD=12", close(1.2 * 60 + 10, 82, 1e-9) and close(1.2 ** 2 * 100, 144, 1e-9), None)
xa = rng.normal(20, 3, 1_000_000); yb = rng.normal(15, 4, 1_000_000)
check(T, "例2 合計 平均 35・SD 5、差 平均 5・SD 5（シミュレーション）", close((xa + yb).mean(), 35, 0.02) and close((xa + yb).std(), 5, 0.02) and close((xa - yb).mean(), 5, 0.02) and close((xa - yb).std(), 5, 0.02), ((xa + yb).std(), (xa - yb).std()))
check(T, "例3 E[X²]=4+9=13", 4 + 3 ** 2 == 13, None)
zz = (xa - 20) / 3
check(T, "標準化 (X−μ)/σ の平均 0・分散 1", close(zz.mean(), 0, 0.01) and close(zz.var(), 1, 0.01), (zz.mean(), zz.var()))
T = "exponential-distribution"
e8 = stats.expon(scale=8)
check(T, "平均 8：P(>8)=e^−1≈0.368・P(>16)≈0.135・P(≤4)≈0.393・中央値 8log2≈5.5・分散 64", close(e8.sf(8), .368, 5e-4) and close(e8.sf(16), .135, 5e-4) and close(e8.cdf(4), .393, 5e-4) and close(e8.median(), 5.5, 0.05) and close(e8.var(), 64, 1e-9), e8.median())
check(T, "無記憶性 P(T>12|T>4)=e^−1・図の値 1−e^−1≈0.632", close(e8.sf(12) / e8.sf(4), np.exp(-1), 1e-12) and close(e8.cdf(8), .632, 5e-4), None)
check(T, "ポアソンとの関係：P(T>30)=e^−1.5（平均 20 分）= Po(1.5) の P(0) ≈0.223", close(stats.expon(scale=20).sf(30), stats.poisson(1.5).pmf(0), 1e-12) and close(np.exp(-1.5), .223, 5e-4), None)
arr = np.cumsum(rng.exponential(8, 200000)); cnt = np.bincount((arr // 80).astype(int))[:-1]
check(T, "到着の間隔が平均 8 分の指数分布なら 80 分の件数は平均 10 のポアソン分布（平均 ≈ 分散 ≈ 10）", close(cnt.mean(), 10, 0.1) and close(cnt.var(), 10, 0.3), (cnt.mean(), cnt.var()))
T = "weak-law-of-large-numbers"
check(T, "例1 上限 100/400=0.25・n≥2000 で 0.05・正規近似 2P(Z≥2)≈0.046", close(100 / 400, .25, 1e-12) and 100 / (2000 * 1) <= 0.05 and close(2 * stats.norm.sf(2), 0.046, 5e-4), None)
check(T, "例2 表の割合の SD：100 回 0.05・10000 回 0.005", close(sqrt(.25 / 100), .05, 1e-12) and close(sqrt(.25 / 10000), .005, 1e-12), None)
fl = rng.random((2000, 10000)) < .5; pr = fl.cumsum(1) / np.arange(1, 10001); diff = np.abs(2 * fl.cumsum(1) - np.arange(1, 10001))
check(T, "割合は 0.5 に近づくが、表と裏の回数の差は広がる（シミュレーション）", np.abs(pr[:, 9999] - .5).mean() < np.abs(pr[:, 99] - .5).mean() / 5 and diff[:, 9999].mean() > diff[:, 99].mean() * 5, (np.abs(pr[:, 99] - .5).mean(), np.abs(pr[:, 9999] - .5).mean(), diff[:, 99].mean(), diff[:, 9999].mean()))

T = "null-and-alternative-hypothesis"
check(T, "例1 z=(498−500)/(6/√36)=−2.0 < −1.645", close((498 - 500) / (6 / 6), -2, 1e-12) and -2 < -stats.norm.isf(.05), None)
T = "significance-level"
check(T, "棄却域の表 1.645・2.326・1.96・2.576", close(stats.norm.isf(.05), 1.645) and close(stats.norm.isf(.01), 2.326) and close(stats.norm.isf(.025), 1.960) and close(stats.norm.isf(.005), 2.576), None)
check(T, "例1 z=2.0：5% で棄却・1% で棄却しない", 1.645 < 2.0 < 2.326, None)
check(T, "例2 1−0.95³≈0.143・20 回に 1 回", close(1 - .95 ** 3, .143, 5e-4) and close(1 / .05, 20, 1e-9), None)
T = "p-value"
check(T, "例1 P(Z≥2.05)=0.0202・両側 0.0404", close(stats.norm.sf(2.05), .0202, 5e-5) and close(2 * .0202, .0404, 1e-12), None)
check(T, "例2 t(12)：2.179・2.681・t=2.3 の両側 p≈0.040（0.02〜0.05）", close(stats.t.isf(.025, 12), 2.179) and close(stats.t.isf(.01, 12), 2.681) and close(2 * stats.t.sf(2.3, 12), 0.040, 5e-4) and 0.02 < 2 * stats.t.sf(2.3, 12) < 0.05, 2 * stats.t.sf(2.3, 12))
T = "type-i-and-ii-errors"
check(T, "境目 53.29・β=P(Z≤−0.86)≈0.195・検出力≈0.80・n=100 で ≈0.9996", close(50 + 1.645 * 2, 53.29, 1e-9) and close(stats.norm.cdf(-0.86), .195, 5e-4) and close(stats.norm(55, 2).sf(53.29), .80, 0.005) and close(stats.norm(55, 1).sf(51.645), .9996, 5e-5), stats.norm(55, 2).sf(53.29))
T = "one-sided-two-sided-test"
check(T, "z=1.75：片側 p=0.0401・両側 0.0802", close(stats.norm.sf(1.75), .0401, 5e-5) and close(2 * .0401, .0802, 1e-12) and 1.645 < 1.75 < 1.96, None)
T = "z-test-known-variance"
check(T, "z=−2.0・両側 p=2×0.0228=0.0456・1% では棄却しない", close((498 - 500) / (6 / sqrt(36)), -2, 1e-12) and close(2 * .0228, .0456, 1e-12) and close(stats.norm.sf(2), .0228, 5e-5) and 2 < 2.576, None)
check("z-test-known-variance", "95% 信頼区間 498±1.96 → (496.04, 499.96) は 500 を含まない", close(498 - 1.96, 496.04, 1e-9) and close(498 + 1.96, 499.96, 1e-9) and 498 + 1.96 < 500, None)
T = "one-sample-t-test"
tt = (103 - 100) / (5 / sqrt(10))
check(T, "t≈1.90・t0.05(9)=1.833・t0.025(9)=2.262・右側の面積≈0.045・両側 p≈0.09", close(tt, 1.90, 0.005) and close(stats.t.isf(.05, 9), 1.833) and close(stats.t.isf(.025, 9), 2.262) and close(stats.t.sf(tt, 9), .045, 5e-4) and close(2 * stats.t.sf(tt, 9), .09, 0.005) and close(5 / sqrt(10), 1.581, 5e-4), tt)
check(T, "p 値の範囲：右片側 0.025〜0.05・両側 0.05〜0.10", 0.025 < stats.t.sf(tt, 9) < 0.05 and 0.05 < 2 * stats.t.sf(tt, 9) < 0.10, None)
check(T, "t0.025(30)=2.042", close(stats.t.isf(.025, 30), 2.042), None)
T = "proportion-test"
se0 = sqrt(.05 * .95 / 400)
check(T, "連続修正 (27.5−20)/4.36≈1.72 > 1.645", close(7.5 / sqrt(19), 1.72, 0.005) and 7.5 / sqrt(19) > 1.645, 7.5 / sqrt(19))
check(T, "p̂=0.07・SE≈0.0109・z≈1.84・p 値 0.0329・回数で 8/4.36・p̂ を使うと 0.0128 と 1.57", close(28 / 400, .07, 1e-12) and close(se0, .0109, 5e-5) and close(.02 / se0, 1.84, 0.005) and close(stats.norm.sf(1.84), .0329, 5e-5) and close(sqrt(400 * .05 * .95), 4.36, 0.005) and close(8 / sqrt(19), 1.84, 0.005) and close(sqrt(.07 * .93 / 400), .0128, 5e-5) and close(.02 / sqrt(.07 * .93 / 400), 1.57, 0.005), .02 / se0)

ng = results.count(False)
print(f"書き直した共通の用語の検算: OK {len(results) - ng} / NG {ng}")
sys.exit(1 if ng else 0)
