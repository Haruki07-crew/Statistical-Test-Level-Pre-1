"""第7章（極限定理、漸近理論）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import comb
from scipy import stats

q = Q(7)

# --- マルコフ ---
q.num("q07-markov-inequality-1", 4 / 10)
q.check("q07-markov-inequality-1", abs(stats.expon(scale=4).sf(10) - 0.082) < 5e-4, "指数分布の実際の値 0.082")
q.num("q07-markov-inequality-2", (1 + 2 ** 2) / 16)
q.check("q07-markov-inequality-2", (1 + 4) / 16 < 2 / 4, "X^2 版のほうが X 版 0.5 より小さい")

# --- チェビシェフ ---
q.num("q07-chebyshev-inequality-1", 1 - 1 / 2 ** 2)
q.check("q07-chebyshev-inequality-1", abs(stats.norm.cdf(2) - stats.norm.cdf(-2) - 0.954) < 5e-4, "正規なら 0.954")
nmin = next(n for n in range(1, 10000) if 0.25 / n / 0.05 ** 2 <= 0.1)
q.num("q07-chebyshev-inequality-2", nmin)
q.check("q07-chebyshev-inequality-2", int(np.ceil((stats.norm.ppf(0.95) * 0.5 / 0.05) ** 2)) == 271, "CLT なら 271 回")
vals, probs = np.array([-2, 0, 2]), np.array([1 / 8, 3 / 4, 1 / 8])
mu = (vals * probs).sum(); sd = np.sqrt(((vals - mu) ** 2 * probs).sum())
q.check("q07-chebyshev-inequality-3", abs(mu) < 1e-12 and abs(sd - 1) < 1e-12
        and abs(probs[np.abs(vals - mu) >= 2 * sd].sum() - 1 / 4) < 1e-12, "3点分布で等号成立（誤り）")

# --- 確率収束 ---
okp = all(abs(n * (1 / n) - 1) < 1e-12 for n in range(1, 200))
q.check("q07-convergence-in-probability-1", okp, "E[X_n]=1 のまま（誤り）、P(X_n>=eps)=1/n")
q.num("q07-convergence-in-probability-2", stats.uniform.cdf(0.9) ** 20)
q.check("q07-convergence-in-probability-2", abs(stats.beta(20, 1).cdf(0.9) - 0.9 ** 20) < 1e-12, "最大値は Be(n,1)")
X, mu_, s2 = sp.symbols("x mu sigma2")
xs = sp.symbols("x1:6")
n_ = 5
xbar = sum(xs) / n_
lhs = sum((x - xbar) ** 2 for x in xs) / n_
rhs = sum(x ** 2 for x in xs) / n_ - xbar ** 2
q.check("q07-convergence-in-probability-3", sp.simplify(lhs - rhs) == 0, "分解の恒等式（n=5で記号的に確認）")
q.concept("q07-convergence-in-probability-3", "LLN+連続写像定理による一致性の導出")

# --- 概収束 ---
q.concept("q07-almost-sure-convergence-1", "収束の強弱関係（数値なし）")
N = 10 ** 5
prod = np.prod(1 - 1 / np.arange(2, N + 1))
q.check("q07-almost-sure-convergence-2", abs(prod - 1 / N) < 1e-9, "prod_{n=2}^N (1-1/n) = 1/N -> 0（誤り）")
prod2 = np.prod(1 - 1 / np.arange(2, N + 1) ** 2.0)
q.num("q07-almost-sure-convergence-3", prod2)
q.check("q07-almost-sure-convergence-3", abs(prod2 - (N + 1) / (2 * N)) < 1e-9, "(N+1)/(2N) と一致")

# --- 分布収束 ---
q.num("q07-convergence-in-distribution-1", 1 - np.exp(-2))
q.check("q07-convergence-in-distribution-1", abs(1 - (1 - 2 / 10 ** 6) ** (10 ** 6) - (1 - np.exp(-2))) < 1e-5,
        "有限 n=10^6 での値が極限に一致")
rng = np.random.default_rng(0)
z = rng.standard_normal(100000)
q.check("q07-convergence-in-distribution-2", np.var(z + (-z)) == 0 and abs(np.var(z + rng.standard_normal(100000)) - 2) < 0.05,
        "Z+(-Z) の分散0 ≠ 独立和の分散2（誤り）")

# --- 大数の弱法則 ---
q.num("q07-weak-law-of-large-numbers-1", 4 / (100 * 0.5 ** 2))
t = np.linspace(-5, 5, 101)
q.check("q07-weak-law-of-large-numbers-2", np.allclose(np.exp(-np.abs(t) / 50) ** 50, np.exp(-np.abs(t)))
        and abs(2 * stats.cauchy.sf(1) - 0.5) < 1e-12, "標本平均も標準コーシー、P(|X|>=1)=0.5（誤り）")

# --- 強法則 ---
q.concept("q07-strong-law-of-large-numbers-1", "強法則の必要十分条件（数値なし）")

# --- CLT ---
q.num("q07-central-limit-theorem-1", stats.norm.sf((2.5 - 2) / (2 / 8)))
q.check("q07-central-limit-theorem-1", abs(stats.gamma(64, scale=2).sf(2.5 * 64) - 0.0290) < 5e-5, "ガンマでの正確値 0.0290")
q.num("q07-central-limit-theorem-2", stats.norm.sf((27 - 24) / np.sqrt(48 / 12)))
q.concept("q07-central-limit-theorem-3", "CLT の内容（数値なし）")

# --- スラツキー ---
q.num("q07-slutsky-theorem-1", stats.norm(2, 2).cdf(4))
q.concept("q07-slutsky-theorem-2", "Y_n が定数に確率収束する仮定（数値なし）")
q.concept("q07-slutsky-theorem-3", "スラツキーによる t 型統計量の導出")

# --- 連続写像 ---
q.num("q07-continuous-mapping-theorem-1", stats.chi2.cdf(2.706, 1))
q.check("q07-continuous-mapping-theorem-1", abs(np.sqrt(2.706) - 1.645) < 5e-4, "sqrt(2.706)=1.645")
g = lambda x: 1.0 if x > 0 else 0.0
q.check("q07-continuous-mapping-theorem-2", all(g(1 / n) == 1.0 for n in range(1, 1000)) and g(0) == 0.0, "g(X_n)=1 のまま（誤り）")

# --- デルタ法 ---
T = sp.Symbol("t")
av = sp.diff(T ** 2, T).subs(T, 3) ** 2 * 4
q.check("q07-delta-method-1", av == 144 and all(w != av for w in [24, 36, 12]), "正解 144、誤答は異なる")
lam, n = 4, 100
q.num("q07-delta-method-2", (1 / (2 * np.sqrt(lam))) ** 2 * lam / n)
# シミュレーションでも確認
xb = rng.poisson(lam, size=(200000, n)).mean(axis=1)
q.check("q07-delta-method-2", abs(np.var(np.sqrt(xb)) - 0.0025) < 1e-4, "シミュレーションで約0.0025")
P = sp.Symbol("p")
d = sp.diff(sp.log(P / (1 - P)), P)
q.check("q07-delta-method-3", sp.simplify(d - 1 / (P * (1 - P))) == 0, "導関数 1/(p(1-p))")
q.num("q07-delta-method-3", float(sp.sqrt(d.subs(P, 0.2) ** 2 * 0.2 * 0.8 / 200)))

# --- 連続修正 ---
ex = stats.binom.cdf(14, 100, 0.2)  # 平均20・SD4 の二項分布 B(100,0.2) で比べる
errs = [abs(stats.norm.cdf((c - 20) / 4) - ex) for c in (14.5, 15.5, 15, 14)]
q.check("q07-continuity-correction-1", int(np.argmin(errs)) == 0 and len(set(np.round(errs, 6))) == 4,
        "B(100,0.2) の P(X<=14) に最も近いのは 14.5（正解）")
s = np.sqrt(50 * 0.4 * 0.6)
q.num("q07-continuity-correction-2", stats.norm.cdf((17.5 - 20) / s))
q.check("q07-continuity-correction-2", abs(stats.binom.cdf(17, 50, 0.4) - 0.2369) < 5e-5
        and abs(stats.norm.cdf(-3 / s) - 0.193) < 5e-4, "正確値0.2369、修正なし0.193")
q.num("q07-continuity-correction-3", stats.norm.cdf(0.5 / s) - stats.norm.cdf(-0.5 / s))
q.check("q07-continuity-correction-3", abs(stats.binom.pmf(20, 50, 0.4) - 0.1146) < 5e-5, "正確値0.1146")

# --- 少数の法則 ---
q.num("q07-law-of-small-numbers-1", stats.poisson.cdf(1, 2))
q.check("q07-law-of-small-numbers-1", abs(stats.binom.cdf(1, 500, 0.004) - 0.4055) < 5e-5, "二項の正確値 0.4055")
q.check("q07-law-of-small-numbers-2", abs(stats.poisson(2).var() - 2) < 1e-12
        and abs(stats.binom(500, 0.004).var() - 1.992) < 1e-9, "分散 2 と 1.992（正しい）")

# --- 平均二乗収束 ---
q.num("q07-convergence-in-mean-square-1", next(n for n in range(1, 10000) if 9 / n <= 0.05))

# --- リンデバーグ・レヴィ / ベリー・エッセン ---
q.concept("q07-lindeberg-levy-clt-1", "仮定の確認（数値なし）")
rho = 0.5 * 0.5 ** 3 + 0.5 * 0.5 ** 3
sig3 = 0.5 ** 3
q.num("q07-berry-esseen-theorem-1", 0.4748 * rho / sig3 / np.sqrt(100))

q.done()
