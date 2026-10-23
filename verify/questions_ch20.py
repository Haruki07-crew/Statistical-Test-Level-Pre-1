"""第20章（分散分析と実験計画法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import itertools
import numpy as np
from scipy import stats

q = Q(20)
rng = np.random.default_rng(20)

# --- 一元配置 ---
g = [np.array([5, 7, 6, 6.]), np.array([8, 9, 7, 8.]), np.array([10, 9, 11, 10.])]
F, p = stats.f_oneway(*g)
q.num("q20-one-way-anova-1", F)
q.check("q20-one-way-anova-1", abs(stats.f.ppf(0.95, 2, 9) - 4.26) < 0.005, "F_0.05(2,9)=4.26")
q.check("q20-one-way-anova-2", (5 - 1, 30 - 5) == (4, 25) and 4 + 25 == 29, "F(4,25)、誤答は自由度の数え違い")
a1, a2 = rng.normal(size=7), rng.normal(1, 1, size=9)
t = stats.ttest_ind(a1, a2); f = stats.f_oneway(a1, a2)
q.check("q20-one-way-anova-3", abs(t.statistic ** 2 - f.statistic) < 1e-9 and abs(t.pvalue - f.pvalue) < 1e-12, "k=2 で F=t²、p 値一致")

# --- 平方和の分解 ---
q.num("q20-sum-of-squares-decomposition-1", ((150 - 90) / 3) / (90 / 20))
ok = True
for _ in range(50):
    gs = [rng.normal(size=rng.integers(2, 8)) + rng.normal() for _ in range(rng.integers(2, 6))]
    y = np.concatenate(gs); gm = y.mean()
    ST = np.sum((y - gm) ** 2); SA = sum(len(x) * (x.mean() - gm) ** 2 for x in gs); SE = sum(np.sum((x - x.mean()) ** 2) for x in gs)
    cross = sum((x.mean() - gm) * np.sum(x - x.mean()) for x in gs)
    ok &= abs(ST - SA - SE) < 1e-9 and abs(cross) < 1e-9
q.check("q20-sum-of-squares-decomposition-2", ok, "ランダムなデータで S_T=S_A+S_E、交差項0")

# --- 繰り返しのある二元配置 ---
a, b, r = 3, 4, 2
q.check("q20-two-way-anova-with-replication-1", ((a - 1) * (b - 1), a * b * (r - 1)) == (6, 12) and (a - 1) + (b - 1) + 6 + 12 == a * b * r - 1 and (a - 1) * (b - 1) + a * b * (r - 1) == 18, "(6,12)、誤答 18 はプールした自由度")
Y = np.array([[[3, 5], [6, 8]], [[5, 7], [12, 14.]]])  # [A][B][rep]
cm = Y.mean(2); gm = Y.mean(); am = Y.mean((1, 2)); bm = Y.mean((0, 2))
SAB = 2 * np.sum((cm - am[:, None] - bm[None, :] + gm) ** 2); SE = np.sum((Y - cm[:, :, None]) ** 2)
q.num("q20-two-way-anova-with-replication-2", (SAB / 1) / (SE / 4))
# 回帰（交互作用ダミー）で同じ F になるか
A_ = np.repeat([0, 0, 1, 1], 2); B_ = np.repeat([0, 1, 0, 1], 2); yv = Y.reshape(-1)
def rss(X): e = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]; return e @ e
X0 = np.column_stack([np.ones(8), A_, B_]); X1 = np.column_stack([X0, A_ * B_])
q.check("q20-two-way-anova-with-replication-2", abs(((rss(X0) - rss(X1)) / 1) / (rss(X1) / 4) - 4) < 1e-9 and abs(stats.f.ppf(.95, 1, 4) - 7.71) < 0.005,
        "回帰の部分F検定でも4、F_0.05(1,4)=7.71")

# --- 繰り返しのない二元配置 ---
Y2 = np.array([[4, 6, 8], [6, 9, 9], [8, 9, 13.]]); gm = Y2.mean()
SA = 3 * np.sum((Y2.mean(1) - gm) ** 2); SB = 3 * np.sum((Y2.mean(0) - gm) ** 2); ST = np.sum((Y2 - gm) ** 2)
q.num("q20-two-way-anova-without-replication-1", (SA / 2) / ((ST - SA - SB) / 4))
q.check("q20-two-way-anova-without-replication-1", abs(stats.f.ppf(.95, 2, 4) - 6.94) < 0.005, "F_0.05(2,4)=6.94")
# 交互作用を入れた飽和モデルでは残差自由度0
cells = np.eye(9)
q.check("q20-two-way-anova-without-replication-2", np.linalg.matrix_rank(cells) == 9 and 9 - 9 == 0, "交互作用を入れると残差自由度0で検定不能（誤り）")

# --- 主効果と交互作用 ---
mu = np.array([[10, 14], [12, 22.]])
ab = mu - mu.mean(1)[:, None] - mu.mean(0)[None, :] + mu.mean()
q.num("q20-main-effect-and-interaction-1", ab[1, 1])
q.check("q20-main-effect-and-interaction-1", abs(((22 - 12) - (14 - 10)) / 4 - ab[1, 1]) < 1e-12, "差の差/4")
mu2 = np.array([[10, 14], [14, 10.]])
q.check("q20-main-effect-and-interaction-2", abs(mu2.mean(1)[1] - mu2.mean(1)[0]) < 1e-12 and abs(mu2[1, 0] - mu2[0, 0]) == 4, "主効果0でも単純主効果±4（誤り）")

# --- フィッシャーの3原則 ---
q.concept("q20-fisher-three-principles-1", "無作為化の役割")
r_ = next(r for r in range(1, 100) if np.sqrt(2 * 9 / r) <= 1.5 + 1e-12)
q.num("q20-fisher-three-principles-2", r_)

# --- 乱塊法 ---
St, Sb, ST = 36, 60, 120
q.num("q20-randomized-block-design-1", (St / 3) / ((ST - St - Sb) / 12))
q.num("q20-randomized-block-design-2", (St / 3) / ((ST - St) / 16))
q.check("q20-randomized-block-design-2", abs(stats.f.ppf(.95, 3, 12) - 3.49) < 0.005 and abs(stats.f.ppf(.95, 3, 16) - 3.24) < 0.005, "臨界値 3.49、3.24")
# 実データでも確認：上の平方和をもつデータを作る
# （処理効果・ブロック効果・交互作用残差を直交に作って尺度を合わせる）
t_, b_ = 4, 5
tau = np.array([1, -1, 1, -1.]); beta = np.array([2, -1, 0, 1, -2.]); E = rng.normal(size=(t_, b_))
E = E - E.mean(1)[:, None] - E.mean(0)[None, :] + E.mean()
tau *= np.sqrt(St / (b_ * np.sum(tau ** 2))); beta *= np.sqrt(Sb / (t_ * np.sum(beta ** 2))); E *= np.sqrt(24 / np.sum(E ** 2))
Yb = 10 + tau[:, None] + beta[None, :] + E
F1 = stats.f_oneway(*Yb).statistic
q.check("q20-randomized-block-design-2", abs(F1 - 12 / 5.25) < 1e-9, "構成したデータの一元配置F")

# --- ラテン方格 ---
p_ = 5
q.num("q20-latin-square-design-1", (p_ ** 2 - 1) - 3 * (p_ - 1))
q.check("q20-latin-square-design-2", (p_ - 1) * (p_ - 2) < (p_ - 1) ** 2, "残り自由度 < 行×処理の自由度（誤り）")

# --- 直交表 ---
q.num("q20-orthogonal-array-1", (52 - 40) ** 2 / 8)
L8 = np.array(list(itertools.product([0, 1], repeat=3)))  # a,b,c
cols = {1: L8[:, 0], 2: L8[:, 1], 3: L8[:, 0] ^ L8[:, 1], 4: L8[:, 2], 5: L8[:, 0] ^ L8[:, 2], 6: L8[:, 1] ^ L8[:, 2], 7: L8[:, 0] ^ L8[:, 1] ^ L8[:, 2]}
bc = cols[2] ^ cols[4]
match = [k for k, v in cols.items() if np.array_equal(v, bc) or np.array_equal(v, 1 - bc)]
q.check("q20-orthogonal-array-2", match == [6], f"B×C は第{match}列")
orth = all(all(np.sum((cols[i] == u) & (cols[j] == v)) == 2 for u in (0, 1) for v in (0, 1)) for i in cols for j in cols if i < j)
q.check("q20-orthogonal-array-2", orth, "L8 の直交性")

# --- テューキー ---
qcrit = stats.studentized_range.ppf(0.95, 4, 20)
q.num("q20-tukey-multiple-comparison-1", qcrit * np.sqrt(8 / 6))
q.check("q20-tukey-multiple-comparison-1", abs(qcrit - 3.958) < 0.001 and abs(3.958 * np.sqrt(16 / 6) - 6.46) < 0.005, "q=3.958、誤答 6.46")
R = 20000; hit = 0; n = 10
for _ in range(R):
    gs = rng.normal(size=(4, n))
    if any(stats.ttest_ind(gs[i], gs[j]).pvalue < 0.05 for i, j in itertools.combinations(range(4), 2)):
        hit += 1
q.check("q20-tukey-multiple-comparison-2", hit / R > 0.15, f"ファミリーワイズ過誤率 {hit / R:.3f} > 0.05（正しい）")

# --- スチューデント化された範囲 ---
q.num("q20-studentized-range-distribution-1", np.sqrt(2) * 2.228)
q.check("q20-studentized-range-distribution-1", abs(stats.studentized_range.ppf(0.95, 2, 10) - np.sqrt(2) * stats.t.ppf(0.975, 10)) < 1e-6, "scipy の q(2,10)=√2 t")

# --- 効果量 ---
VE = (200 - 40) / 30
q.num("q20-effect-size-eta-squared-1", (40 - 2 * VE) / 200)

# --- 母数・変量 ---
q.num("q20-fixed-and-random-effects-1", (26 - 6) / 5)
# シミュレーション：E[V_A]=σ²+nσ_A²
VA, VEs = [], []
for _ in range(5000):
    yy = rng.normal(scale=2, size=(8, 1)) + rng.normal(scale=np.sqrt(6), size=(8, 5))
    VA.append(5 * np.var(yy.mean(1), ddof=1)); VEs.append(np.mean(np.var(yy, axis=1, ddof=1)))
q.check("q20-fixed-and-random-effects-1", abs((np.mean(VA) - np.mean(VEs)) / 5 - 4) < 0.3, "シミュレーションで (V_A-V_E)/n≈σ_A²")
q.concept("q20-fixed-and-random-effects-2", "無作為に選んだ水準＝変量因子")

# --- 期待平均平方（変量模型、σ_A=0） ---
a, b, r = 3, 4, 3
VA, VAB, VE2 = [], [], []
for _ in range(4000):
    Bj = rng.normal(scale=1.0, size=b); ABij = rng.normal(scale=1.5, size=(a, b))
    yy = Bj[None, :, None] + ABij[:, :, None] + rng.normal(size=(a, b, r))
    cm = yy.mean(2); gm = yy.mean(); am = yy.mean((1, 2)); bm = yy.mean((0, 2))
    VA.append(b * r * np.sum((am - gm) ** 2) / (a - 1))
    VAB.append(r * np.sum((cm - am[:, None] - bm[None, :] + gm) ** 2) / ((a - 1) * (b - 1)))
    VE2.append(np.sum((yy - cm[:, :, None]) ** 2) / (a * b * (r - 1)))
q.check("q20-expected-mean-square-1", abs(np.mean(VA) / np.mean(VAB) - 1) < 0.05 and np.mean(VA) / np.mean(VE2) > 3,
        f"σ_A=0 で E[V_A]≈E[V_AB]（{np.mean(VA):.2f} vs {np.mean(VAB):.2f}）、V_E とは大きく異なる")

# --- 要因計画 ---
T = {"1": 20, "a": 28, "b": 24, "ab": 40}
CA = -T["1"] + T["a"] - T["b"] + T["ab"]
q.num("q20-factorial-design-1", CA ** 2 / (4 * 2))
q.check("q20-factorial-design-1", CA / (2 * 2) == 6, "効果 6")

# --- 分割法 ---
r, a, b = 3, 2, 4
q.check("q20-split-plot-design-1", (r - 1) * (a - 1) == 2 and a * (r - 1) * (b - 1) == 12, "E1 の自由度2、E2 は12")

# --- 交絡 ---
def mul(x, y):
    return "".join(sorted(set(x) ^ set(y)))
q.check("q20-confounding-1", mul("B", "ABC") == "AC", "B×ABC=AC")
runs = ["", "a", "b", "ab", "c", "ac", "bc", "abc"]
same = [r_ or "(1)" for r_ in runs if len(r_) % 2 == 0]
q.check("q20-confounding-2", same == ["(1)", "ab", "ac", "bc"], f"(1) と同じブロック: {same}")

# --- 一部実施要因計画・不完備ブロック計画（用語の見直しで追加） ---
def _mul(a, b):
    return "".join(sorted(set(a) ^ set(b))) or "I"
q.check("q20-fractional-factorial-design-1", _mul("A", "ABCD") == "BCD" and _mul("AB", "ABCD") == "CD", "A=BCD、AB=CD")
_rel = ["ABD", "ACE", _mul("ABD", "ACE")]
_res = min(len(w) for w in _rel)
q.check("q20-fractional-factorial-design-2", _rel[2] == "BCDE" and _res == 3, "定義関係 I=ABD=ACE=BCDE、分解能 III")
q.check("q20-fractional-factorial-design-2", all(r != _res for r in (4, 5, 2)), "誤答 IV/V/II は正解と異なる")
from itertools import combinations as _cb
_v, _b, _k = 6, 10, 3
_r = _b * _k // _v
_lam = _r * (_k - 1) / (_v - 1)
q.num("q20-incomplete-block-design-1", _lam)
_blocks = [set(map(int, s)) for s in ["012", "013", "024", "035", "045", "125", "134", "145", "234", "235"]]
_pairs = {p: sum(1 for B in _blocks if set(p) <= B) for p in _cb(range(6), 2)}
_reps = [sum(1 for B in _blocks if i in B) for i in range(6)]
q.check("q20-incomplete-block-design-1", set(_pairs.values()) == {2} and set(_reps) == {5}, f"BIBD(6,10,3) の実例で r=5、λ=2")

q.done()
