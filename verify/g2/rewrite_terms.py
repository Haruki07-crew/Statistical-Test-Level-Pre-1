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

T = "point-estimation"
d5 = np.array([12, 15, 9, 14, 10.])
check(T, "例 平均 12・不偏分散 26/4=6.5・標準誤差 √6.5/√5≈1.14・46/200=0.23", close(d5.mean(), 12, 1e-12) and close(((d5 - 12) ** 2).sum(), 26, 1e-12) and close(d5.var(ddof=1), 6.5, 1e-12) and close(sqrt(6.5) / sqrt(5), 1.14, 0.005) and close(46 / 200, .23, 1e-12), None)
check(T, "図 10/√25=2", close(10 / sqrt(25), 2, 1e-12), None)
T = "unbiased-estimator"
check(T, "例1 (4/5)×4=3.2・図：n=5 で gamma(2, 0.5) の平均 1・gamma(2, 0.4) の平均 0.8", close(4 / 5 * 4, 3.2, 1e-12) and close(stats.gamma(2, scale=.5).mean(), 1, 1e-12) and close(stats.gamma(2, scale=.4).mean(), .8, 1e-12), None)
v5 = rng.normal(0, 1, (400000, 5)); s2 = v5.var(1, ddof=1)
check(T, "図：不偏分散（n=5）の分布は gamma(2, 0.5) と一致（平均・分散）", close(s2.mean(), 1, 0.01) and close(s2.var(), stats.gamma(2, scale=.5).var(), 0.01), (s2.mean(), s2.var()))
check(T, "例2 V[X̄]=σ²/3≈0.333・V[T]=6/16=0.375", close(3 / 9, .333, 5e-4) and close(6 / 16, .375, 1e-12), None)
check("unbiased-estimator", "標本比率は不偏（シミュレーション）", close((rng.binomial(50, .3, 400000) / 50).mean(), .3, 0.001), None)
T = "standard-error"
check(T, "図 12/√n・例1 400 と 200・例2 √(0.21/400)≈0.0229・±0.045", np.allclose([12 / sqrt(n_) for n_ in (1, 4, 9, 16, 25, 36, 49, 64, 81, 100)], [12.0, 6.0, 4.0, 3.0, 2.4, 2.0, 1.7143, 1.5, 1.3333, 1.2], atol=1e-4) and 2400 / 6 == 400 and 2400 / 12 == 200 and close(sqrt(.21 / 400), .0229, 5e-5) and close(1.96 * .0229, .045, 5e-4), None)
check(T, "6/√100=0.6", close(6 / 10, .6, 1e-12), None)
check(T, "例3 √(144/36+81/81)=√5≈2.24", close(sqrt(144 / 36 + 81 / 81), 2.24, 0.005), None)
T = "confidence-interval"
se_ = 2.8 / 1.96
check(T, "例 中央 51.0・半分 2.8・SE≈1.43・99% は ±3.68 → (47.3, 54.7)", close((48.2 + 53.8) / 2, 51, 1e-9) and close((53.8 - 48.2) / 2, 2.8, 1e-9) and close(se_, 1.43, 0.005) and close(se_ * 2.576, 3.68, 0.005) and close(51 - se_ * 2.576, 47.3, 0.05) and close(51 + se_ * 2.576, 54.7, 0.05), se_ * 2.576)
check(T, "99% と 95% の幅の比 ≈1.3", close(2.576 / 1.96, 1.3, 0.02), 2.576 / 1.96)
T = "ci-mean-known-variance"
check(T, "例1 90%・95%・99% の区間", close(120 - 1.645 * 3, 115.07, 0.005) and close(120 + 1.645 * 3, 124.93, 0.005) and close(120 - 1.96 * 3, 114.12, 1e-9) and close(120 + 1.96 * 3, 125.88, 1e-9) and close(120 - 2.576 * 3, 112.27, 0.005) and close(120 + 2.576 * 3, 127.73, 0.005), None)
check(T, "例2 9.8²=96.04 → 97", close(9.8 ** 2, 96.04, 1e-9) and min(n_ for n_ in range(1, 500) if 1.96 * 15 / sqrt(n_) <= 3) == 97, None)
check("ci-mean-known-variance", "区間と検定 115 は含む・126 は含まない", 114.12 < 115 < 125.88 < 126, None)
T = "ci-mean-unknown-variance"
check(T, "例 SE≈1.732・t0.025(11)=2.201・68±3.81 → (64.19, 71.81)・1.96 なら ±3.39 → (64.61, 71.39)", close(6 / sqrt(12), 1.732, 5e-4) and close(stats.t.isf(.025, 11), 2.201) and close(2.201 * 6 / sqrt(12), 3.81, 0.005) and close(68 - 2.201 * 6 / sqrt(12), 64.19, 0.005) and close(1.96 * 6 / sqrt(12), 3.39, 0.005) and close(68 + 1.96 * 6 / sqrt(12), 71.39, 0.005), None)
check("ci-mean-unknown-variance", "90%：1.796 で ±3.11 → (64.89, 71.11)・99%：3.106 で ±5.38 → (62.62, 73.38)", close(stats.t.isf(.05, 11), 1.796) and close(stats.t.isf(.005, 11), 3.106) and close(1.796 * 6 / sqrt(12), 3.11, 0.005) and close(68 - 1.796 * 6 / sqrt(12), 64.89, 0.005) and close(3.106 * 6 / sqrt(12), 5.38, 0.005) and close(68 + 3.106 * 6 / sqrt(12), 73.38, 0.005), None)
T = "ci-proportion"
sep = sqrt(.36 * .64 / 500)
check(T, "例1 180/500=0.36・SE≈0.0215・±0.042 → (0.318, 0.402)", close(180 / 500, .36, 1e-12) and close(sep, .0215, 5e-5) and close(1.96 * sep, .042, 5e-4) and close(.36 - 1.96 * sep, .318, 5e-4) and close(.36 + 1.96 * sep, .402, 5e-4), sep)
check(T, "誤差の目安 1000 人 ±3.1・400 人 ±4.9", close(1.96 * sqrt(.25 / 1000), .031, 5e-4) and close(1.96 * sqrt(.25 / 400), .049, 5e-4), None)
check(T, "例2 98²×0.25=2401・p=0.36 なら約 2213", close(98 ** 2 * .25, 2401, 1e-9) and int(np.ceil((1.96 / .02) ** 2 * .36 * .64)) == 2213, (1.96 / .02) ** 2 * .36 * .64)

T = "two-sample-t-test"
r2 = stats.ttest_ind_from_stats(72, 6, 10, 66, 7, 12)
check(T, "sp²=(324+539)/20=43.15・SE≈2.813・t≈2.13・t0.025(20)=2.086・片側面積≈0.023・両側 p≈0.045", close((9 * 36 + 11 * 49) / 20, 43.15, 1e-9) and close(sqrt(43.15 * (1 / 10 + 1 / 12)), 2.813, 5e-4) and close(r2.statistic, 2.13, 0.005) and close(stats.t.isf(.025, 20), 2.086) and close(stats.t.sf(r2.statistic, 20), .023, 5e-4) and close(r2.pvalue, .045, 6e-4), (r2.statistic, r2.pvalue))
check(T, "差の 95% 信頼区間 6±5.87 → (0.13, 11.87)", close(2.086 * sqrt(43.15 * (1 / 10 + 1 / 12)), 5.87, 0.005) and close(6 - 5.867, 0.13, 0.005) and close(6 + 5.867, 11.87, 0.005), None)
T = "welch-test"
rw = stats.ttest_ind_from_stats(85, 12, 10, 78, 6, 15, equal_var=False)
nu = 16.8 ** 2 / (14.4 ** 2 / 9 + 2.4 ** 2 / 14)
check(T, "SE=√16.8≈4.099・t≈1.71・自由度≈12.0・t0.025(12)=2.179・プールすると ≈1.94（自由度 23）", close(sqrt(16.8), 4.099, 5e-4) and close(rw.statistic, 1.71, 0.005) and close(nu, 12.0, 0.05) and close(stats.t.isf(.025, 12), 2.179) and close(stats.ttest_ind_from_stats(85, 12, 10, 78, 6, 15).statistic, 1.94, 0.005), (rw.statistic, nu))
check("welch-test", "差の 95% 信頼区間 7±8.93 → (−1.93, 15.93)", close(2.179 * sqrt(16.8), 8.93, 0.005) and close(7 - 2.179 * sqrt(16.8), -1.93, 0.005) and close(7 + 2.179 * sqrt(16.8), 15.93, 0.005), None)
T = "paired-t-test"
dd = np.array([2, -1, 3, 4, 0, 2, 5, 1.])
check(T, "差の平均 2・偏差平方和 28・不偏分散 4・t≈2.83・t0.025(7)=2.365・片側面積≈0.013・両側≈0.025", close(dd.mean(), 2, 1e-12) and close(((dd - 2) ** 2).sum(), 28, 1e-12) and close(dd.var(ddof=1), 4, 1e-12) and close(stats.ttest_1samp(dd, 0).statistic, 2.83, 0.005) and close(stats.t.isf(.025, 7), 2.365) and close(stats.t.sf(2.828, 7), .013, 5e-4) and close(stats.ttest_1samp(dd, 0).pvalue, .025, 5e-4), stats.ttest_1samp(dd, 0).pvalue)
check(T, "前後の SD が 10 なら独立とみなした t は 2/√(100/8+100/8)=0.4", close(2 / sqrt(100 / 8 + 100 / 8), 0.4, 1e-12), None)
T = "goodness-of-fit-test"
gf = stats.chisquare([18, 22, 16, 20, 24, 14, 26])
check(T, "期待度数 20・χ²=112/20=5.6・自由度 6 の 5% 点 12.59", close(gf.statistic, 5.6, 1e-9) and close(stats.chi2.isf(.05, 6), 12.59, 0.005) and sum([18, 22, 16, 20, 24, 14, 26]) == 140, gf.statistic)
T = "independence-test"
ri = stats.chi2_contingency([[30, 20], [25, 25], [15, 35]], correction=False)
check(T, "期待度数 23.33・26.67・χ²=9.375・自由度 2・5.991 と 9.210 を超える・割合 60/50/30%", np.allclose(ri.expected_freq, [[70 / 3, 80 / 3]] * 3) and close(ri.statistic, 9.375, 1e-9) and ri.dof == 2 and close(stats.chi2.isf(.05, 2), 5.991) and close(stats.chi2.isf(.01, 2), 9.210) and ri.statistic > 9.210, ri.statistic)
z2 = (40 / 50 - 30 / 50) / sqrt(.7 * .3 * (1 / 50 + 1 / 50))
check(T, "2×2 表の χ² は比率の差の z の 2 乗", close(stats.chi2_contingency([[40, 10], [30, 20]], correction=False).statistic, z2 ** 2, 1e-9), z2 ** 2)

T = "regression-coefficient-t-test"
check(T, "SE=4/√64=0.5・t=4.0・t0.025(10)=2.228・区間 (0.886, 3.114)・F=t²=16・両側 p≈0.0025", close(4 / 8, .5, 1e-12) and close(2 / .5, 4, 1e-12) and close(stats.t.isf(.025, 10), 2.228) and close(2 - 2.228 * .5, .886, 1e-9) and close(2 + 2.228 * .5, 3.114, 1e-9) and close(2 * stats.t.sf(4, 10), .0025, 1e-4) and close(stats.t.sf(4, 10), .0013, 1e-4), 2 * stats.t.sf(4, 10))
xs_ = rng.normal(0, 1, 12); ys_ = 1 + 2 * xs_ + rng.normal(0, 1, 12); lr = stats.linregress(xs_, ys_); res_ = ys_ - lr.intercept - lr.slope * xs_
fr = (((lr.intercept + lr.slope * xs_ - ys_.mean()) ** 2).sum() / 1) / ((res_ ** 2).sum() / 10)
check(T, "単回帰では t² = F（乱数データ）", close((lr.slope / lr.stderr) ** 2, fr, 1e-9), None)
T = "adjusted-r-squared"
r_ = np.random.default_rng(2025); n_ = 30; acc = {q_: [[], []] for q_ in range(1, 11)}
for _ in range(4000):
    X_ = r_.normal(size=(n_, 10)); y_ = 1 + 0.6 * X_[:, 0] + 0.5 * X_[:, 1] + r_.normal(size=n_); sst = ((y_ - y_.mean()) ** 2).sum()
    for q_ in range(1, 11):
        A_ = np.c_[np.ones(n_), X_[:, :q_]]; b_ = np.linalg.lstsq(A_, y_, rcond=None)[0]; r2 = 1 - ((y_ - A_ @ b_) ** 2).sum() / sst
        acc[q_][0].append(r2); acc[q_][1].append(1 - (1 - r2) * (n_ - 1) / (n_ - q_ - 1))
check(T, "図の値（同じ乱数の種で 4000 回のシミュレーションを再計算）", np.allclose([round(np.mean(acc[q_][0]), 3) for q_ in range(1, 11)], [0.236, 0.406, 0.428, 0.45, 0.472, 0.494, 0.516, 0.538, 0.56, 0.582]) and np.allclose([round(np.mean(acc[q_][1]), 3) for q_ in range(1, 11)], [0.209, 0.362, 0.362, 0.362, 0.362, 0.362, 0.362, 0.361, 0.362, 0.361]), None)
check(T, "例 0.570・0.559", close(1 - .4 * 29 / 27, .570, 5e-4) and close(1 - .38 * 29 / 25, .559, 5e-4), None)
T = "one-way-anova"
ga = np.array([3, 5, 4, 6, 7.]); gb = np.array([7, 8, 6, 9, 10.])
check(T, "2 群：S_A=22.5・S_E=20・F=9.0・t=−3.0・F(2,12) の 5% 点 3.885・1−0.95³≈0.14", close(5 * ((5 - 6.5) ** 2 + (8 - 6.5) ** 2), 22.5, 1e-12) and close(((ga - 5) ** 2).sum() + ((gb - 8) ** 2).sum(), 20, 1e-12) and close(stats.f_oneway(ga, gb).statistic, 9, 1e-9) and close(stats.ttest_ind(ga, gb).statistic, -3, 1e-9) and close(stats.f.isf(.05, 2, 12), 3.885) and close(1 - .95 ** 3, .14, 0.005), None)
T = "sum-of-squares-decomposition"
g3 = [np.array([1, 2, 3.]), np.array([4, 5, 6.]), np.array([7, 8, 9.])]
check(T, "S_T=60・S_A=54・S_E=6・90%・F=27", ((np.concatenate(g3) - 5) ** 2).sum() == 60 and sum(3 * (x.mean() - 5) ** 2 for x in g3) == 54 and sum(((x - x.mean()) ** 2).sum() for x in g3) == 6 and close(54 / 60, .9, 1e-12) and close(stats.f_oneway(*g3).statistic, 27, 1e-9), None)

T = "covariance-and-correlation"
xc = np.array([2, 4, 6, 8, 10.]); yc = np.array([50, 60, 58, 72, 80.])
check(T, "平均 6・64・偏差の積の和 144・共分散 28.8（n−1 なら 36）・40・568・r≈0.955", xc.mean() == 6 and yc.mean() == 64 and close(((xc - 6) * (yc - 64)).sum(), 144, 1e-9) and close(144 / 5, 28.8, 1e-12) and close(np.cov(xc, yc)[0, 1], 36, 1e-9) and close(((xc - 6) ** 2).sum(), 40, 1e-9) and close(((yc - 64) ** 2).sum(), 568, 1e-9) and close(np.corrcoef(xc, yc)[0, 1], .955, 5e-4), np.corrcoef(xc, yc)[0, 1])
T = "spurious-correlation"
check(T, "例2 偏相関 (0.56−0.7×0.8)/√(0.51×0.36)=0", close((0.56 - 0.7 * 0.8) / sqrt(0.51 * 0.36), 0, 1e-12) and close(0.51 * 0.36, 0.1836, 1e-12), None)
T = "coefficient-of-determination"
check(T, "1−90/250=0.64・160・r=±0.8", close(1 - 90 / 250, .64, 1e-12) and 250 - 90 == 160 and close(sqrt(.64), .8, 1e-12), None)
Xm = rng.normal(size=(200, 3)); ym = Xm @ [1, -2, .5] + rng.normal(size=200); Am = np.c_[np.ones(200), Xm]; yh = Am @ np.linalg.lstsq(Am, ym, rcond=None)[0]
check("coefficient-of-determination", "重回帰の R² = corr(y, ŷ)²", close(1 - ((ym - yh) ** 2).sum() / ((ym - ym.mean()) ** 2).sum(), np.corrcoef(ym, yh)[0, 1] ** 2, 1e-9), None)
T = "residual"
xr = np.array([1, 2, 3, 4, 5.]); yr = np.array([3, 7, 5, 11, 9.]); br, ar = np.polyfit(xr, yr, 1); er = yr - ar - br * xr
check(T, "ŷ=2.2+1.6x・予測 3.8〜10.2・残差 −0.8, 1.6, −2.0, 2.4, −1.2・合計 0・14.4・s≈2.19", close(ar, 2.2, 1e-9) and close(br, 1.6, 1e-9) and np.allclose(ar + br * xr, [3.8, 5.4, 7.0, 8.6, 10.2]) and np.allclose(er, [-.8, 1.6, -2, 2.4, -1.2]) and close(er.sum(), 0, 1e-9) and close((er ** 2).sum(), 14.4, 1e-9) and close(sqrt(14.4 / 3), 2.19, 0.005) and close((er * xr).sum(), 0, 1e-9), None)
T = "simple-random-sampling"
import itertools
ms = [np.mean(c) for c in itertools.combinations([2, 4, 6, 8], 2)]
check(T, "6 通りの標本平均 3,4,5,5,6,7・平均 5・分散 10/6・公式 5/2×2/3", sorted(ms) == [3, 4, 5, 5, 6, 7] and close(np.mean(ms), 5, 1e-12) and close(np.var(ms), 10 / 6, 1e-12) and close(np.var([2, 4, 6, 8]), 5, 1e-12) and close(5 / 2 * 2 / 3, 10 / 6, 1e-12), None)
T = "sampling-and-nonsampling-error"
check(T, "図の値 √(0.25/n)・√(0.25/n+0.03²)・表 0.058・0.034・0.030", np.allclose([round(sqrt(.25 / n_), 4) for n_ in (100, 200, 500, 1000, 2000, 5000, 10000)], [0.05, 0.0354, 0.0224, 0.0158, 0.0112, 0.0071, 0.005]) and np.allclose([round(sqrt(.25 / n_ + .03 ** 2), 4) for n_ in (100, 200, 500, 1000, 2000, 5000, 10000)], [0.0583, 0.0464, 0.0374, 0.0339, 0.032, 0.0308, 0.0304]) and close(sqrt(.05 ** 2 + .03 ** 2), .058, 5e-4) and close(sqrt(.016 ** 2 + .03 ** 2), .034, 5e-4) and close(sqrt(.005 ** 2 + .03 ** 2), .030, 5e-4), None)
T = "fisher-three-principles"
d_ = np.array([[10, 12.5, 13.5], [14.5, 16, 18], [18, 20.5, 22], [21.5, 24, 26.5]]); gm_ = d_.mean()
ST_ = ((d_ - gm_) ** 2).sum(); SA_ = 4 * ((d_.mean(0) - gm_) ** 2).sum(); SB_ = 3 * ((d_.mean(1) - gm_) ** 2).sum()
check(T, "平均 16・18.25・20・誤差の平均平方 26.806 → 0.194・F 0.60 → 82.7", np.allclose(d_.mean(0), [16, 18.25, 20]) and close((ST_ - SA_) / 9, 26.806, 5e-4) and close((ST_ - SA_ - SB_) / 6, .194, 5e-4) and close((SA_ / 2) / ((ST_ - SA_) / 9), .60, 1e-9) and close((SA_ / 2) / ((ST_ - SA_ - SB_) / 6), 82.7, 0.05) and close(stats.f_oneway(*d_.T).statistic, .6, 1e-9), None)
T = "conditional-probability"
check(T, "例1 120/600=0.20・120/160=0.75・図の値 P(B)=0.6×0.2+0.4×0.1=0.16", close(120 / 600, .2, 1e-12) and close(120 / 160, .75, 1e-12) and close(.6 * .2 + .4 * (1 - .9), .16, 1e-12) and close(.6 * .2 / .16, .75, 1e-12), None)
check(T, "例2 3/10×2/9=6/90≈0.067", close(3 / 10 * 2 / 9, 6 / 90, 1e-12) and close(6 / 90, .067, 5e-4), None)
T = "independence-of-events"
check(T, "A∩B={2,4}: 1/3=1/2×2/3（独立）・A∩C={4,6}: 1/3≠1/4・P(A|C)=2/3・図 P(B|Aᶜ)=2/3", close(2 / 6, .5 * 4 / 6, 1e-12) and abs(2 / 6 - .5 * .5) > .05 and close(2 / 3, 2 / 3, 0) and close(1 - 0.3333, 2 / 3, 1e-4) and close(0.6667, 2 / 3, 1e-4), None)

T = "correlation-matrix"
cm = {("国語", "社会"): .68, ("国語", "数学"): .21, ("国語", "理科"): .30, ("社会", "数学"): .25, ("社会", "理科"): .40, ("数学", "理科"): .74}
check(T, "最大は数学と理科 0.74・次が国語と社会 0.68・最小は国語と数学 0.21・組の数 4×3/2=6", max(cm, key=cm.get) == ("数学", "理科") and sorted(cm.values())[-2] == .68 and min(cm, key=cm.get) == ("国語", "数学") and len(cm) == 4 * 3 // 2, None)
T = "partial-correlation-coefficient"
check(T, "例 (0.7−0.48)/√(0.64×0.36)=0.22/0.48≈0.46", close((0.7 - .48) / sqrt(.64 * .36), .46, 0.005) and close(sqrt(.2304), .48, 1e-12), None)
zz_ = rng.normal(size=200000); xx_ = 0.8 * zz_ + 0.6 * rng.normal(size=200000); yy_ = 0.5 * xx_ + 0.7 * zz_ + rng.normal(size=200000)
rxy, rxz, ryz = np.corrcoef(xx_, yy_)[0, 1], np.corrcoef(xx_, zz_)[0, 1], np.corrcoef(yy_, zz_)[0, 1]
ex_ = xx_ - np.polyval(np.polyfit(zz_, xx_, 1), zz_); ey_ = yy_ - np.polyval(np.polyfit(zz_, yy_, 1), zz_)
check(T, "偏相関係数 = z で回帰した残差どうしの相関", close((rxy - rxz * ryz) / sqrt((1 - rxz ** 2) * (1 - ryz ** 2)), np.corrcoef(ex_, ey_)[0, 1], 1e-9), None)
T = "sum-of-random-variables"
ds = [a_ + b_ for a_ in range(1, 7) for b_ in range(1, 7)]
check(T, "例1 さいころ 2 個の和：期待値 7・分散 35/6≈5.83・P(7)=1/6", close(np.mean(ds), 7, 1e-12) and close(np.var(ds), 35 / 6, 1e-12) and close(ds.count(7) / 36, 1 / 6, 1e-12) and close(35 / 6, 5.83, 0.005), None)
check(T, "例2 B(25,0.3) の分散 5.25", close(stats.binom(25, .3).var(), 5.25, 1e-12), None)
T = "cumulative-distribution-function"
check(T, "例1 F(3)=0.5・P(2<X≤5)=1/2・例2 0.512−0.125=0.387・中央値≈0.794", close(3 / 6, .5, 1e-12) and close(5 / 6 - 2 / 6, .5, 1e-12) and close(.8 ** 3 - .5 ** 3, .387, 1e-9) and close(.5 ** (1 / 3), .794, 5e-4), None)

T = "geometric-distribution"
g4 = stats.geom(.25)
check(T, "P(3)=0.140625・P(≤2)=0.4375・P(>4)≈0.316・E=4・V=12・P(X>6|X>2)=P(X>4)", close(g4.pmf(3), .140625, 1e-12) and close(g4.cdf(2), .4375, 1e-12) and close(g4.sf(4), .316, 5e-4) and close(g4.mean(), 4, 1e-12) and close(g4.var(), 12, 1e-12) and close(g4.sf(6) / g4.sf(2), g4.sf(4), 1e-12), None)
T = "uniform-distribution"
u20 = stats.uniform(0, 20)
check(T, "U(0,20)：平均 10・分散 33.3・SD 5.77・P(5〜12)=0.35・P(≥15)=0.25・SD≈0.29×幅", close(u20.mean(), 10, 1e-12) and close(u20.var(), 33.3, 0.05) and close(u20.std(), 5.77, 0.005) and close(u20.cdf(12) - u20.cdf(5), .35, 1e-12) and close(u20.sf(15), .25, 1e-12) and close(1 / sqrt(12), .29, 0.005), None)
T = "hypergeometric-distribution"
h20 = stats.hypergeom(20, 6, 5)
check(T, "P(0)=2002/15504≈0.129・0.871・E=1.5・V≈0.829・二項なら 0.168 と 1.05", close(h20.pmf(0), 2002 / 15504, 1e-12) and close(2002 / 15504, .129, 5e-4) and close(h20.sf(0), .871, 5e-4) and close(h20.mean(), 1.5, 1e-12) and close(h20.var(), .829, 5e-4) and close(.7 ** 5, .168, 5e-4), h20.var())
T = "continuity-correction"
b20 = stats.binom(20, .4); sd20 = sqrt(4.8)
check(T, "B(20,0.4)：SD≈2.19・修正あり P(Z≤−1.14)≈0.127・なし P(Z≤−1.37)≈0.085・正確 ≈0.126", close(sd20, 2.19, 0.005) and close((5.5 - 8) / sd20, -1.14, 0.005) and close(stats.norm.cdf(-1.14), .127, 5e-4) and close((5 - 8) / sd20, -1.37, 0.005) and close(stats.norm.cdf(-1.37), .085, 5e-4) and close(b20.cdf(5), .126, 5e-4), b20.cdf(5))

ng = results.count(False)
print(f"書き直した共通の用語の検算: OK {len(results) - ng} / NG {ng}")
sys.exit(1 if ng else 0)
