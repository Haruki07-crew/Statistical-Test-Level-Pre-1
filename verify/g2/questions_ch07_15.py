"""2級 第7・10・14・15章：問題の検算（章ごとに結果を保存する）。"""
import sys
import pathlib
from math import comb, perm, sqrt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qhead import Q, chosen  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ng = 0
rng = np.random.default_rng(11)


def run(q):
    global ng
    try:
        q.done()
    except SystemExit as e:
        ng += int(bool(e.code))


q = Q(7)
q.num("q07-counting-1", comb(6, 2))
lot = np.array([1] * 3 + [0] * 7); sim = np.mean([lot[rng.choice(10, 2, replace=False)].sum() == 2 for _ in range(200000)])
q.num("q07-counting-2", comb(3, 2) / comb(10, 2))
q.check("q07-counting-2", abs(sim - 1 / 15) < 0.003, f"シミュレーション {sim:.4f}")
q.check("q07-counting-3", perm(8, 2) == 56 and comb(8, 2) == 28 and chosen(q, "q07-counting-3") == "56", "8P2=56（28 は組合せ）")
q.num("q07-events-and-probability-1", 0.5 + 0.4 - 0.2)
q.check("q07-events-and-probability-2", 0 != 0.3 * 0.4, "排反なら P(A∩B)=0 ≠ P(A)P(B)（answer=false）")
q.num("q07-events-and-probability-3", 1 - 0.5 ** 4)
q.check("q07-events-and-probability-4", abs(1 - (0.8 + 0.5 - 0.4) - 0.1) < 1e-12 and chosen(q, "q07-events-and-probability-4") == "10%", "どちらも使わない 10%")
cls = np.array([("M", 1)] * 10 + [("M", 0)] * 15 + [("F", 1)] * 9 + [("F", 0)] * 6, dtype=object)
glasses = cls[cls[:, 1] == 1]
q.num("q07-conditional-probability-g1", np.mean(glasses[:, 0] == "F"))
fac = rng.random(2_000_000) < 0.4; bad = rng.random(2_000_000) < np.where(fac, 0.05, 0.02)
q.num("q07-bayes-theorem-g1", 0.4 * 0.05 / (0.6 * 0.02 + 0.4 * 0.05))
q.check("q07-bayes-theorem-g1", abs(fac[bad].mean() - 0.625) < 0.005, f"シミュレーション {fac[bad].mean():.4f}")
q.check("q07-bayes-theorem-g2", abs(0.01 * 0.99 / (0.01 * 0.99 + 0.99 * 0.05) - 99 / 594) < 1e-12 and abs(99 / 594 - 0.17) < 0.005 and round(10000 * 0.99 * 0.05) == 495, f"反例 99/594={99 / 594:.4f}（answer=false）")
q.num("q07-bayes-theorem-g3", 0.5 * 0.4 / (0.5 * 0.4 + 0.5 * 0.8))
box = rng.random(1_000_000) < 0.5; red = rng.random(1_000_000) < np.where(box, 0.4, 0.8)
q.check("q07-bayes-theorem-g3", abs(box[red].mean() - 1 / 3) < 0.003, f"シミュレーション {box[red].mean():.4f}")
run(q)

q = Q(10)
q.num("q10-sampling-distribution-of-mean-1", 12 / np.sqrt(36))
q.num("q10-sampling-distribution-of-mean-2", stats.norm.sf(53, 50, 10 / 5))
q.check("q10-sampling-distribution-of-mean-2", abs(stats.norm.sf(1.5) - 0.0668) < 5e-5, "表の値 P(Z>1.5)=0.0668")
q.check("q10-sampling-distribution-of-mean-3", abs((1 / np.sqrt(400)) / (1 / np.sqrt(100)) - 0.5) < 1e-12 and chosen(q, "q10-sampling-distribution-of-mean-3") == "1/2 になる", "標準偏差は 1/2")
m = rng.exponential(1, (100000, 100)).mean(1); zz = (m - 1) * 10
q.check("q10-sampling-distribution-of-mean-4", abs(np.mean(zz > 1.96) - 0.025) < 0.006, f"指数分布 n=100 の標本平均で P(Z>1.96)≈{np.mean(zz > 1.96):.4f}（answer=true）")
q.check("q10-statistical-tables-1", chosen(q, "q10-statistical-tables-1") == f"{stats.norm.isf(0.025):.3f}", "z_0.025 = 1.960")
q.num("q10-statistical-tables-2", 2 * 0.1151)
q.check("q10-statistical-tables-2", abs(2 * stats.norm.sf(1.2) - 0.2302) < 1e-4, "正確な値と一致")
q.num("q10-statistical-tables-3", 1 / 4.818)
q.check("q10-statistical-tables-3", abs(stats.f.isf(0.05, 8, 5) - 4.818) < 5e-4 and abs(stats.f.isf(0.95, 5, 8) - 1 / 4.818) < 1e-4, "表の値と関係式")
cov = 0
for _ in range(4000):
    x = rng.normal(0, 2, 10); s2 = x.var(ddof=1)
    cov += 9 * s2 / stats.chi2.isf(0.025, 9) <= 4 <= 9 * s2 / stats.chi2.isf(0.975, 9)
q.check("q10-statistical-tables-4", abs(cov / 4000 - 0.95) < 0.015 and "(9)" in chosen(q, "q10-statistical-tables-4") and "0.025" in chosen(q, "q10-statistical-tables-4"), f"自由度 9 の 2.5%・97.5% 点で被覆率 {cov / 4000:.3f}")
m36 = rng.uniform(50 - 12 * sqrt(3), 50 + 12 * sqrt(3), (200000, 36)).mean(1)
q.num("q10-central-limit-theorem-g1", stats.norm.sf((54 - 50) / (12 / 6)))
q.check("q10-central-limit-theorem-g1", abs(np.mean(m36 > 54) - 0.0228) < 0.002, f"一様分布の母集団でのシミュレーション {np.mean(m36 > 54):.4f}（指数分布のように歪んだ母集団では約 0.03 になり、解説に注意を書いた）")
e1 = rng.exponential(1, 200000); e100 = rng.exponential(1, (100000, 100)).mean(1)
q.check("q10-central-limit-theorem-g2", stats.skew(e1) > 1.8 and abs(stats.skew(e100)) < 0.3 and abs(e100.std() - 0.1) < 0.002 and chosen(q, "q10-central-limit-theorem-g2").startswith("近似的に、平均 $\\mu$・標準偏差 $\\sigma/10$"), f"右に裾の長い母集団（歪度 {stats.skew(e1):.2f}）でも、100 個の平均は歪度 {stats.skew(e100):.2f}・標準偏差 {e100.std():.4f}（σ/10=0.1）")
q.num("q10-central-limit-theorem-g3", stats.norm.sf((2045 - 2000) / (sqrt(100) * 3)))
t100 = rng.gamma((20 / 3) ** 2, 9 / 20, (200000, 100)).sum(1)
q.check("q10-central-limit-theorem-g3", abs(stats.norm.sf(1.5) - 0.0668) < 5e-5 and abs(np.mean(t100 > 2045) - 0.0668) < 0.003, f"表の値・平均 20・標準偏差 3 の歪んだ分布で合計のシミュレーション {np.mean(t100 > 2045):.4f}")
q.num("q10-normal-approximation-binomial-poisson-g1", stats.binom(400, 0.2).std())
q.check("q10-t-distribution-g1", stats.t.isf(.025, 5) > stats.t.isf(.025, 30) > stats.norm.isf(.025), "自由度が大きいほど点が 1.96 に近づく（answer=true）")
q.num("q10-chi-squared-distribution-g1", stats.chi2(6).var())
run(q)

q = Q(14)
q.num("q14-regression-output-1", 0.36 / 0.12)
q.check("q14-regression-output-2", [k for k, p in zip("ABC", (0.003, 0.210, 0.048)) if p < 0.05] == ["A", "C"] and chosen(q, "q14-regression-output-2") == "A と C", "p<0.05 は A と C")
q.num("q14-regression-output-3", 1 - 0.4 * 24 / 21)
n = 200; x = rng.normal(170, 8, n); y = 0.5 * x + rng.normal(0, 3, n)
b_cm = np.polyfit(x, y, 1)[0]; b_m = np.polyfit(x / 100, y, 1)[0]
q.check("q14-regression-output-4", abs(b_m / b_cm - 100) < 1e-6, "単位を cm→m にすると係数は 100 倍（answer=false）")
q.num("q14-regression-output-5", 1.5 + 0.8 * 10 - 0.3 * 4)
q.num("q14-regression-coefficient-t-test-g1", 1.2 / 4.0)
q.num("q14-adjusted-r-squared-g1", 1 - 0.30 * 30 / 25)
hh = rng.normal(165, 8, 3000); ex = rng.uniform(0, 3, 3000); ww = -60 + 0.75 * hh - 1.5 * ex + rng.normal(0, 3, 3000)
cf = np.linalg.lstsq(np.c_[np.ones(3000), hh, ex], ww, rcond=None)[0]
q.check("q14-multiple-regression-model-g1", abs(cf[2] + 1.5) < 0.15 and chosen(q, "q14-multiple-regression-model-g1").startswith("身長が同じなら"), f"同じ形のデータで運動時間の係数 {cf[2]:.2f}")
run(q)

q = Q(15)
q.concept("q15-treatment-and-control-1", "偽薬の目的はプラセボ効果をそろえること")
q.concept("q15-treatment-and-control-2", "前後比較には時間の影響などが混ざる（answer=false）")
q.num("q15-treatment-and-control-3", 30 / 40 - 20 / 40)
q.num("q15-anova-table-1", 24 - 4)
q.num("q15-anova-table-2", (60 / 2) / (90 / 18))
q.check("q15-anova-table-3", abs(stats.f.isf(0.05, 2, 18) - 3.555) < 5e-4 and 6.0 > 3.555 and chosen(q, "q15-anova-table-3").startswith("3 つの群の母平均がすべて等しい"), "F=6>3.555 で棄却")
g = [rng.normal(m, 1, 6) for m in (0, 1, 2)]; a = np.concatenate(g)
ST = ((a - a.mean()) ** 2).sum(); SA = sum(6 * (x.mean() - a.mean()) ** 2 for x in g); SE = sum(((x - x.mean()) ** 2).sum() for x in g)
q.check("q15-anova-table-4", abs(ST - SA - SE) < 1e-9, "乱数データで S_T = S_A + S_E（answer=true）")
q.num("q15-one-way-anova-g1", (40 / 4) / (90 / 30))
run(q)
raise SystemExit(1 if ng else 0)
