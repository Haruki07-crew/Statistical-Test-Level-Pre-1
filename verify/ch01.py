"""第1章 検算スクリプト（事象と確率）。
厳密計算は fractions.Fraction、主張値との比較は小数第4位（許容 5e-5 または表示桁の丸め幅）。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from fractions import Fraction as F
from itertools import product, combinations
import numpy as np

results = []

def check(term, what, ok, value):
    results.append(ok)
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")

def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol

rng = np.random.default_rng(20261023)

# ---------------- bayes-theorem ----------------
T = "bayes-theorem"
p, se, sp = F(1, 100), F(99, 100), F(95, 100)
num = se * p; den = se * p + (1 - sp) * (1 - p)
post1 = num / den
check(T, "例1 分子 0.0099", num == F(99, 10000), float(num))
check(T, "例1 分母 0.0594", den == F(594, 10000), float(den))
check(T, "例1 P(病気|+)≈0.1667 (=1/6)", close(post1, 0.1667) and post1 == F(1, 6), float(post1))
# 10,000人
sick = 10000 * p; tp = sick * se; fp = (10000 - sick) * (1 - sp)
check(T, "例1 10000人: 病気100, 真陽性99, 偽陽性495, 陽性594", (sick, tp, fp, tp + fp) == (100, 99, 495, 594), (sick, tp, fp, tp + fp))
check(T, "例1 99/594≈0.167", close(F(99, 594), 0.167, 5e-4), float(F(99, 594)))
# 例2
n2 = se * post1; d2 = se * post1 + (1 - sp) * (1 - post1)
check(T, "例2 分子 0.1650", close(n2, 0.1650), float(n2))
check(T, "例2 分母 0.2067", close(d2, 0.2067), float(d2))
check(T, "例2 P(病気|+,+)≈0.7984", close(n2 / d2, 0.7984), float(n2 / d2))
# 例2 シミュレーション（条件付き独立な2回の検査）, 許容 0.01
N = 2_000_000
D = rng.random(N) < 0.01
pos_prob = np.where(D, 0.99, 0.05)
t1 = rng.random(N) < pos_prob; t2 = rng.random(N) < pos_prob
sim = D[t1 & t2].mean()
check(T, "例2 シミュレーション P(D|T1,T2) (N=2e6, 許容0.01)", abs(sim - 0.7984) < 0.01, round(sim, 4))
# 例3
w = [F(5, 10), F(3, 10), F(2, 10)]; q = [F(1, 100), F(2, 100), F(3, 100)]
pb = sum(a * b for a, b in zip(w, q))
check(T, "例3 P(不良)=0.017", pb == F(17, 1000), float(pb))
pa, pB, pc = [a * b / pb for a, b in zip(w, q)]
check(T, "例3 P(A|不良)≈0.294", close(pa, 0.294, 5e-4), float(pa))
check(T, "例3 P(B|不良)=P(C|不良)≈0.353", pB == pc and close(pB, 0.353, 5e-4), float(pB))
# オッズ
lr = se / (1 - sp); prior_odds = p / (1 - p); post_odds = lr * prior_odds
check(T, "オッズ 尤度比19.8, 事後オッズ0.2, 確率0.167", lr == F(198, 10) and post_odds == F(1, 5) and close(post_odds / (1 + post_odds), 0.167, 5e-4), (float(lr), float(post_odds), float(post_odds / (1 + post_odds))))

# ---------------- conditional-independence ----------------
T = "conditional-independence"
pt12 = se**2 * p + (1 - sp)**2 * (1 - p)
check(T, "例1 項 0.009801, 0.002475", se**2 * p == F(9801, 10**6) and (1 - sp)**2 * (1 - p) == F(2475, 10**6), (float(se**2 * p), float((1 - sp)**2 * (1 - p))))
check(T, "例1 P(T1∩T2)=0.012276", pt12 == F(12276, 10**6), float(pt12))
check(T, "例1 P(T1)P(T2)≈0.003528", close(den**2, 0.003528, 5e-7), float(den**2))
check(T, "例1 P(T2|T1)≈0.2067", close(pt12 / den, 0.2067), float(pt12 / den))
check(T, "例1 P(D|T1∩T2)≈0.7984 かつ逐次更新と一致", close(se**2 * p / pt12, 0.7984) and se**2 * p / pt12 == n2 / d2, float(se**2 * p / pt12))
# 例1 シミュレーション: T1,T2 が非独立 (許容 0.0005)
check(T, "例1 シミュレーション P(T1∩T2) (許容0.0005)", abs((t1 & t2).mean() - 0.012276) < 5e-4, round((t1 & t2).mean(), 6))
# 例2 コイン
hh = F(1, 2) * F(1, 2)**2 + F(1, 2) * F(4, 5)**2; h = F(1, 2) * F(1, 2) + F(1, 2) * F(4, 5)
check(T, "例2 0.445/0.65≈0.6846, P(2回目表)=0.65", hh == F(445, 1000) and h == F(65, 100) and close(hh / h, 0.6846), float(hh / h))
# 例3 （全列挙）
om = list(product("HT", repeat=2))
Pr = lambda ev: F(sum(1 for w_ in om if ev(w_)), len(om))
A = lambda w_: w_[0] == "H"; B = lambda w_: w_[1] == "H"; C = lambda w_: w_[0] == w_[1]
pABc = Pr(lambda w_: A(w_) and B(w_) and C(w_)) / Pr(C)
pAc = Pr(lambda w_: A(w_) and C(w_)) / Pr(C); pBc = Pr(lambda w_: B(w_) and C(w_)) / Pr(C)
check(T, "例3 独立: P(A∩B)=P(A)P(B)", Pr(lambda w_: A(w_) and B(w_)) == Pr(A) * Pr(B), float(Pr(lambda w_: A(w_) and B(w_))))
check(T, "例3 P(A∩B|C)=0.5, P(A|C)P(B|C)=0.25", pABc == F(1, 2) and pAc * pBc == F(1, 4), (float(pABc), float(pAc * pBc)))
# 定義の同値性（ランダムな同時分布で、条件付き独立を構成して P(A|B∩C)=P(A|C) を確認）
ok = True
for _ in range(1000):
    a, b = rng.random(2)
    # C のもとで A,B 独立
    pAB_C, pA_C, pB_C = a * b, a, b
    ok &= abs(pAB_C / pB_C - pA_C) < 1e-12
check(T, "定義の同値 P(A|B∩C)=P(A|C)（1000ケース）", ok, ok)
# 逐次更新式（記号的）
import sympy as sp_
pd, s, f = sp_.symbols("p s f", positive=True)
lhs = s**2 * pd / (s**2 * pd + f**2 * (1 - pd))
pdt1 = s * pd / (s * pd + f * (1 - pd))
rhs = s * pdt1 / (s * pdt1 + f * (1 - pdt1))
check(T, "逐次更新式 = 一括のベイズ（sympy）", sp_.simplify(lhs - rhs) == 0, sp_.simplify(lhs - rhs))

# ---------------- conditional-probability ----------------
T = "conditional-probability"
dice = list(product(range(1, 7), repeat=2))
Pd = lambda ev: F(sum(1 for d in dice if ev(d)), 36)
s10 = Pd(lambda d: sum(d) >= 10)
c = Pd(lambda d: sum(d) >= 10 and d[0] == 6) / Pd(lambda d: d[0] == 6)
check(T, "例1 P(和≥10)=1/6≈0.1667", s10 == F(1, 6), float(s10))
check(T, "例1 P(和≥10|1個目=6)=0.5", c == F(1, 2) and Pd(lambda d: sum(d) >= 10 and d[0] == 6) == F(3, 36), float(c))
rr = F(3, 10) * F(2, 9); rrw = rr * F(7, 8)
check(T, "例2 赤赤 1/15≈0.0667", rr == F(1, 15) and close(rr, 0.0667), float(rr))
check(T, "例2 赤赤白 7/120≈0.0583", rrw == F(7, 120) and close(rrw, 0.0583), float(rrw))
# 例2 全列挙で確認
balls = "RRRWWWWWWW"
from itertools import permutations
perms3 = list(permutations(range(10), 3))
cnt = sum(1 for t in perms3 if balls[t[0]] == "R" and balls[t[1]] == "R" and balls[t[2]] == "W")
check(T, "例2 全列挙で赤赤白=7/120", F(cnt, len(perms3)) == F(7, 120), F(cnt, len(perms3)))
check(T, "例3 P(購入|スマホ)=0.15, P(スマホ|購入)≈0.6667", F(60, 400) == F(15, 100) and close(F(60, 90), 0.6667), (0.15, float(F(60, 90))))
check(T, "例3 0.15×0.4=0.06=60/1000", F(15, 100) * F(4, 10) == F(60, 1000), 0.06)
# P(A^c|B)=1-P(A|B) は成り立つが P(A|B^c)=1-P(A|B) は不成立（例3で）
pA_Bc = F(30, 600)  # 購入|PC
check(T, "P(A|Bᶜ)≠1−P(A|B) の反例（例3: 0.05 vs 0.85）", pA_Bc != 1 - F(60, 400), (float(pA_Bc), float(1 - F(60, 400))))
# 連鎖律（ランダムな有限確率空間でシミュレーション的に確認）
ok = True
for _ in range(200):
    pw = rng.dirichlet(np.ones(16)); sets = [rng.random(16) < 0.7 for _ in range(3)]
    P_ = lambda m: pw[m].sum()
    a1, a2, a3 = sets
    if P_(a1 & a2) == 0: continue
    ok &= abs(P_(a1 & a2 & a3) - P_(a1) * P_(a1 & a2) / P_(a1) * P_(a1 & a2 & a3) / P_(a1 & a2)) < 1e-12
check(T, "連鎖律（ランダム200ケース）", ok, ok)

# ---------------- independence-of-events ----------------
T = "independence-of-events"
A = lambda d: d[0] % 2 == 0; B = lambda d: sum(d) == 7; C = lambda d: sum(d) == 8
pAB = Pd(lambda d: A(d) and B(d)); pAC = Pd(lambda d: A(d) and C(d))
check(T, "例1 P(A)=1/2, P(B)=1/6, P(A∩B)=1/12=P(A)P(B)", Pd(A) == F(1, 2) and Pd(B) == F(1, 6) and pAB == F(1, 12) == Pd(A) * Pd(B), float(pAB))
check(T, "例1 A∩B={(2,5),(4,3),(6,1)}", sorted(d for d in dice if A(d) and B(d)) == [(2, 5), (4, 3), (6, 1)], sorted(d for d in dice if A(d) and B(d)))
check(T, "例1 P(C)=5/36, P(A∩C)=3/36≈0.0833 ≠ P(A)P(C)=5/72≈0.0694",
      Pd(C) == F(5, 36) and pAC == F(3, 36) and Pd(A) * Pd(C) == F(5, 72) and close(pAC, 0.0833) and close(F(5, 72), 0.0694),
      (float(pAC), float(F(5, 72))))
check(T, "例1 A∩C={(2,6),(4,4),(6,2)}", sorted(d for d in dice if A(d) and C(d)) == [(2, 6), (4, 4), (6, 2)], "ok")
A = lambda w_: w_[0] == "H"; B = lambda w_: w_[1] == "H"; C = lambda w_: w_[0] == w_[1]
pair = all(Pr(lambda w_, x=x, y=y: x(w_) and y(w_)) == F(1, 4) == Pr(x) * Pr(y) for x, y in [(A, B), (A, C), (B, C)])
triple = Pr(lambda w_: A(w_) and B(w_) and C(w_))
check(T, "例2 対ごとに独立（各1/4）", pair, pair)
check(T, "例2 P(A∩B∩C)=1/4 ≠ 1/8", triple == F(1, 4) and Pr(A) * Pr(B) * Pr(C) == F(1, 8), float(triple))
check(T, "例3 直列 0.9^3=0.729, 並列 0.999", close(0.9**3, 0.729, 1e-12) and close(1 - 0.1**3, 0.999, 1e-12), (0.9**3, 1 - 0.1**3))
check(T, "例3 1-(5/6)^4≈0.5177", close(1 - F(5, 6)**4, 0.5177), float(1 - F(5, 6)**4))
check(T, "条件の本数 2^n-n-1（n=2..6を部分集合数で確認）",
      all(sum(1 for m in range(2, n + 1) for _ in combinations(range(n), m)) == 2**n - n - 1 for n in range(2, 7)), "ok")
# 補事象の独立性（sympy）
a_, b_ = sp_.symbols("a b")
check(T, "独立なら A と Bᶜ も独立（sympy）", sp_.simplify((a_ - a_ * b_) - a_ * (1 - b_)) == 0, 0)
# 例3 シミュレーション（許容 0.005）
parts = rng.random((1_000_000, 3)) < 0.9
check(T, "例3 シミュレーション 直列/並列 (N=1e6, 許容0.005)",
      abs(parts.all(1).mean() - 0.729) < 0.005 and abs(parts.any(1).mean() - 0.999) < 0.005,
      (parts.all(1).mean(), parts.any(1).mean()))

# ---------------- law-of-total-probability ----------------
T = "law-of-total-probability"
w = [F(6, 10), F(3, 10), F(1, 10)]; q = [F(1, 100), F(3, 100), F(10, 100)]
terms = [a * b for a, b in zip(w, q)]
check(T, "例1 各項 0.006, 0.009, 0.010", terms == [F(6, 1000), F(9, 1000), F(10, 1000)], [float(t) for t in terms])
check(T, "例1 P(B)=0.025", sum(terms) == F(25, 1000), float(sum(terms)))
check(T, "例1 高リスク群の事後 0.40", terms[2] / sum(terms) == F(2, 5), float(terms[2] / sum(terms)))
check(T, "例1 重みなし誤り 0.14", sum(q) == F(14, 100), float(sum(q)))
check(T, "例1 P(B) が min と max の間", min(q) <= sum(terms) <= max(q), float(sum(terms)))
check(T, "例2 P(表表)=0.125+0.320=0.445", F(1, 2) * F(1, 4) == F(125, 1000) and F(1, 2) * F(16, 25) == F(320, 1000) and hh == F(445, 1000), float(hh))
check(T, "例2 P(1回目表)=0.65, 0.65^2=0.4225≠0.445", h == F(65, 100) and h**2 == F(4225, 10000) and h**2 != hh, float(h**2))
# 例2 シミュレーション（許容 0.005）
coin = np.where(rng.random(N) < 0.5, 0.5, 0.8)
f1 = rng.random(N) < coin; f2 = rng.random(N) < coin
check(T, "例2 シミュレーション P(表表) (N=2e6, 許容0.005)", abs((f1 & f2).mean() - 0.445) < 0.005, round((f1 & f2).mean(), 4))
check(T, "チャート初期値: 0.8^2=0.64, 1-0.5^2=0.75", F(4, 5)**2 == F(64, 100) and 1 - F(1, 2)**2 == F(75, 100), (0.64, 0.75))
# 条件付き版の全確率の公式（ランダム有限空間）
ok = True
for _ in range(200):
    pw = rng.dirichlet(np.ones(30)); lab = rng.integers(0, 3, 30)
    Bm = rng.random(30) < 0.5; Cm = rng.random(30) < 0.6
    P_ = lambda m: pw[m].sum()
    if any(P_((lab == i) & Cm) == 0 for i in range(3)): continue
    lhs = P_(Bm & Cm) / P_(Cm)
    rhs = sum(P_(Bm & (lab == i) & Cm) / P_((lab == i) & Cm) * P_((lab == i) & Cm) / P_(Cm) for i in range(3))
    ok &= abs(lhs - rhs) < 1e-12
check(T, "条件付き版 P(B|C)=Σ P(B|Ai∩C)P(Ai|C)（ランダム200ケース）", ok, ok)

n_ok = sum(results); n_ng = len(results) - n_ok
print(f"\n合計: OK {n_ok} / NG {n_ng}")
raise SystemExit(1 if n_ng else 0)
