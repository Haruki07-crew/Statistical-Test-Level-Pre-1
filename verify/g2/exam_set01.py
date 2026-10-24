"""2級 模擬試験 第1回（g2 set01）の検算。問題ファイルの数値を、scipy の分布関数・シミュレーション・総当たり・
回帰の当てはめなど別の方法で計算し直し、正解の選択肢がただ 1 つ一致するか、誤答の選択肢の値がその誤り方で出るかを確かめる。"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import comb, sqrt
from pathlib import Path

os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from examcheck import Exam  # noqa: E402

E = Exam(1)
rng = np.random.default_rng(20261024)


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def opts(iid):
    return [str(c) for c in E.item(iid)["choices"]]


def ans(iid):
    it = E.item(iid)
    return str(it["choices"][it["answer"]])


# ---- 大問1 度数分布
freq = np.array([6, 14, 17, 9, 4]); lo = np.array([4, 6, 8, 10, 12])
cum = np.cumsum(freq) / 50
E.choice("s01p01a", f(cum[2], 2), "10 千円未満の累積相対度数")
E.check("s01p01a", opts("s01p01a") == [f(17 / 50, 2), f(20 / 50, 2), f(cum[2], 2), f(1 - cum[2], 2), f(cum[3], 2)], "誤答の値が理由どおり")
mid = lo + 1
E.choice("s01p01b", f(mid @ freq / 50, 2), "階級値による平均")
E.check("s01p01b", opts("s01p01b") == [f(mid @ freq / 50, 2), f(lo @ freq / 50, 2), f((lo + 2) @ freq / 50, 2), f(mid[freq.argmax()], 2), f(50 / 5, 2)], "誤答の値が理由どおり")

# ---- 大問2 通学時間
x = np.array([5, 8, 10, 12, 13, 15, 16, 18, 20, 24, 45])
q1, q3 = np.median(x[:5]), np.median(x[6:]); iqr = q3 - q1
out = x[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)]
E.choice("s01p02a", f"四分位範囲 {int(iqr)}、外れ値は {', '.join(map(str, out))}", f"Q1={q1}, Q3={q3}, 外れ値 {out}")
E.check("s01p02a", np.median(x[6:]) - np.median(x[:5]) == 10 and np.median(x[5:]) - np.median(x[:6]) == 8 and x.max() - x.min() == 40, "誤答 8・40 の由来")
st = {"中央値は 16 分である": np.median(x) == 16, "平均値は中央値より大きい": x.mean() > np.median(x),
      "最頻値は 15 分である": np.unique(x, return_counts=True)[1].max() > 1, "範囲は 35 分である": x.max() - x.min() == 35,
      "45 分を除くと、中央値は大きくなる": np.median(x[:-1]) > np.median(x)}
E.check("s01p02b", [k for k, v in st.items() if v] == [ans("s01p02b")], f"正しい記述はただ 1 つ（平均 {x.mean():.2f}・中央値 {np.median(x)}）")

# ---- 大問3 広告費と売上：要約値を満たすデータを作って確かめる
def make_data():
    for _ in range(100000):
        xs = rng.normal(0, 1, 8); ys = rng.normal(0, 1, 8)
        xs = (xs - xs.mean()); xs *= sqrt(200 / (xs @ xs))
        ys = ys - ys.mean(); ys -= (ys @ xs) / (xs @ xs) * xs; ys *= 1 / sqrt(ys @ ys)
        bx = 480 / 200; resid_ss = 2000 - bx * 480
        yy = bx * xs + sqrt(resid_ss) * ys
        return xs + 20, yy + 150
X, Y = make_data()
E.check("s01p03a", abs(((X - 20) ** 2).sum() - 200) < 1e-9 and abs(((Y - 150) ** 2).sum() - 2000) < 1e-9 and abs(((X - 20) * (Y - 150)).sum() - 480) < 1e-9, "要約値どおりのデータを作成")
r = np.corrcoef(X, Y)[0, 1]
E.choice("s01p03a", f(r, 2), f"相関係数 {r:.4f}")
E.check("s01p03a", opts("s01p03a") == [f(480 / 2200, 2), f(480 / 200, 2), f(480 / 2000, 2), f(r * r, 2), f(r, 2)], "誤答の値が理由どおり")
b, a = np.polyfit(X, Y, 1)
E.choice("s01p03b", f(a + b * 25, 1), f"回帰直線 a={a:.3f}, b={b:.3f}")
E.check("s01p03b", opts("s01p03b") == [f(150 + 0.24 * 5, 1), f(150 + float(f(r, 2)) * 5, 1), f(a + b * 25, 1), f(b * 25, 1), f(150 + b * 25, 1)], "誤答の値が理由どおり")
b10 = np.polyfit(10 * X, Y, 1)[0]; r10 = np.corrcoef(10 * X, Y)[0, 1]
E.check("s01p03c", abs(r10 - r) < 1e-12 and abs(b10 - b / 10) < 1e-12 and ans("s01p03c") == "相関係数は変わらず、回帰の傾きは 1/10 になる", f"x を 10 倍：r {r10:.4f}（不変）、傾き {b10:.4f}")

# ---- 大問4 成長率と物価指数
idx = np.array([100, 120, 108, 135, 146.41]); g = idx[1:] / idx[:-1] - 1
gbar = (idx[-1] / idx[0]) ** 0.25 - 1
E.choice("s01p04a", f(gbar * 100, 1) + "%", f"平均成長率 {gbar:.6f}")
E.check("s01p04a", opts("s01p04a") == [f(g.mean() * 100, 1) + "%", f(0.4641 / 4 * 100, 1) + "%", f((1.4641 ** 0.5 - 1) * 100, 1) + "%", f(gbar * 100, 1) + "%", f(46.41, 1) + "%"], "誤答の値が理由どおり")
p0 = np.array([200, 120, 80]); q0 = np.array([5, 10, 15]); p1 = np.array([240, 120, 72]); q1 = np.array([4, 11, 18])
L = p1 @ q0 / (p0 @ q0) * 100; P = p1 @ q1 / (p0 @ q1) * 100
E.choice("s01p04b", f(L, 2), f"ラスパイレス {L:.4f}")
E.check("s01p04b", opts("s01p04b") == [f(P, 2), f(L, 2), f(np.mean(p1 / p0) * 100, 2), f(p0 @ q0 / (p1 @ q0) * 100, 2), f(sqrt(L * P), 2)], "誤答の値が理由どおり")
E.check("s01p04b", abs((p0 * q0 / (p0 @ q0)) @ (p1 / p0) * 100 - L) < 1e-9, "支出の割合で重み付けした価格比の平均とも一致")

# ---- 大問5 データの集め方（概念）
E.check("s01p05a", ans("s01p05a").startswith("参加するかどうかを本人が選んでいる観察研究"), "処理を本人が選ぶ→観察研究・交絡")
z = rng.standard_normal(100000); join = z + rng.standard_normal(100000) > 0; w = -z + rng.standard_normal(100000)
E.check("s01p05a", w[join].mean() - w[~join].mean() < -0.5, "教室に効果がなくても、関心（交絡）で差が出るシミュレーション")
E.check("s01p05b", ans("s01p05b") == "集落抽出法", "地区を選び全世帯を調べる＝集落抽出法")

# ---- 大問6 確率
E.choice("s01p06a", f(1 - (0.35 + 0.25 - 0.05), 2), "加法定理と余事象")
E.check("s01p06a", opts("s01p06a") == [f(1 - 0.60, 2), f(0.45, 2), f(0.55, 2), f(0.60, 2), f(0.05, 2)], "誤答の値が理由どおり")
sick = rng.random(2_000_000) < 0.02; pos = np.where(sick, rng.random(2_000_000) < 0.9, rng.random(2_000_000) < 0.05)
E.choice("s01p06b", f(0.018 / 0.067, 2), f"ベイズの定理（シミュレーション {sick[pos].mean():.4f}）")
E.check("s01p06b", abs(sick[pos].mean() - 0.2687) < 0.005 and abs(pos.mean() - 0.067) < 0.001, "シミュレーションと一致")

# ---- 大問7 期待値と分散
xv = np.array([0, 1, 2, 3]); pv = np.array([.1, .3, .4, .2]); EX = xv @ pv; VX = (xv ** 2) @ pv - EX ** 2
E.choice("s01p07a", f"E[X]={f(EX, 1)},\\ V[X]={f(VX, 2)}", f"E={EX}, V={VX}")
draw = rng.choice(xv, 1_000_000, p=pv)
E.check("s01p07a", abs(draw.mean() - 1.7) < 0.005 and abs(draw.var() - 0.81) < 0.005, "シミュレーション")
Xs = rng.normal(0, 2, 2_000_000); Ys = rng.normal(0, 3, 2_000_000)
E.choice("s01p07b", "97", f"V[2X−3Y]（シミュレーション {np.var(2 * Xs - 3 * Ys):.2f}）")
E.check("s01p07b", abs(np.var(2 * Xs - 3 * Ys) - 97) < 0.5 and opts("s01p07b") == ["−65", "13", "35", "25", "97"] and 4 * 4 - 9 * 9 == -65 and 2 * 4 + 3 * 9 == 35 and (2 * 2 - 3 * 3) ** 2 == 25, "誤答の値が理由どおり")

# ---- 大問8 二項分布と正規分布
B = stats.binom(5, 0.2)
E.choice("s01p08a", f(B.pmf(2), 3), "二項確率")
E.check("s01p08a", opts("s01p08a") == [f(0.04 * 0.512, 3), f(0.04, 3), f(B.pmf(2), 3), f(B.cdf(2), 3), f(B.sf(1), 3)] and comb(5, 2) == 10, "誤答の値が理由どおり")
E.choice("s01p08b", f(1 - 0.3085 - 0.0668, 4), "表の値で計算")
E.check("s01p08b", abs(stats.norm(60, 10).cdf(75) - stats.norm(60, 10).cdf(55) - 0.6247) < 1e-4 and opts("s01p08b") == ["0.6247", f(0.3085 + 0.0668, 4), f(0.3085 - 0.0668, 4), f(1 - 0.3085, 4), f(1 - 0.0668, 4)], "正確な値と誤答の由来")

# ---- 大問9 標本分布
m = rng.normal(200, 15, (200000, 25)).mean(1)
E.choice("s01p09a", f(15 / 5, 2), f"σ/√n（シミュレーション {m.std():.3f}）")
E.check("s01p09a", opts("s01p09a") == [f(15 / 25, 2), "3.00", "15.00", f(225 / 25, 2), f(15 / sqrt(24), 2)], "誤答の値が理由どおり")
E.choice("s01p09b", f(stats.norm.sf(1.9), 4), "59.5 で標準化")
E.check("s01p09b", abs(stats.binom(100, 0.5).sf(59) - 0.0284) < 5e-4 and opts("s01p09b") == [f(stats.norm.sf(2.0), 4), f(stats.norm.sf(2.1), 4), f(stats.norm.sf(1.7), 4), f(stats.norm.sf(1.9), 4), f(1 - 0.0287, 4)], "正確な二項確率 0.0284・誤答の値が表の値どおり")
tt = []
for _ in range(100000 // 1000):
    s_ = rng.normal(5, 2, (1000, 10)); tt += list((s_.mean(1) - 5) / (s_.std(1, ddof=1) / sqrt(10)))
tt = np.array(tt)
E.check("s01p09c", abs(np.mean(tt > stats.t.isf(0.025, 9)) - 0.025) < 0.003 and ans("s01p09c") == "自由度 9 の t 分布", f"T の上側確率が t(9) の 2.5% 点で {np.mean(tt > stats.t.isf(0.025, 9)):.4f}")

# ---- 大問10 区間推定
E.choice("s01p10a", f"({f(50 - 1.96, 2)}, {f(50 + 1.96, 2)})", "z 区間")
E.check("s01p10a", opts("s01p10a") == [f"({f(50 - k * w, 2)}, {f(50 + k * w, 2)})" for k, w in ((1.96, 1), (1.645, 1), (1.96, 6), (1.96, 6 / 36), (2.576, 1))], "誤答の値が理由どおり")
E.check("s01p10b", abs(stats.t.isf(0.025, 15) - 2.131) < 5e-4 and abs(stats.t.isf(0.025, 16) - 2.120) < 5e-4 and abs(stats.t.isf(0.05, 15) - 1.753) < 5e-4, "表の値")
E.choice("s01p10b", f"({f(72 - 2.131 * 2, 2)}, {f(72 + 2.131 * 2, 2)})", "t 区間")
E.check("s01p10b", opts("s01p10b") == [f"({f(72 - k * w, 2)}, {f(72 + k * w, 2)})" for k, w in ((1.96, 2), (2.120, 2), (2.131, 2), (2.131, 8), (1.753, 2))], "誤答の値が理由どおり")
se = sqrt(0.3 * 0.7 / 400)
E.choice("s01p10c", f"({f(0.3 - 1.96 * se, 3)}, {f(0.3 + 1.96 * se, 3)})", "比率の区間")
E.check("s01p10c", opts("s01p10c") == [f"({f(0.3 - k * w, 3)}, {f(0.3 + k * w, 3)})" for k, w in ((1.645, se), (1.96, 0.025), (1.96, 0.21 / 400), (2.576, se), (1.96, se))], "誤答の値が理由どおり")
need = (1.96 / 0.03) ** 2 * 0.25
E.choice("s01p10d", str(int(np.ceil(need))), f"必要な n {need:.2f} を切り上げ")
E.check("s01p10d", 1.96 * sqrt(0.25 / 1068) <= 0.03 < 1.96 * sqrt(0.25 / 1067) and opts("s01p10d") == ["1067", "1068", str(int(np.ceil((1.645 / 0.03) ** 2 * 0.25))), str(int(np.ceil((1.96 / 0.03) ** 2 * 0.5))), str(int(np.ceil(1.96 / 0.03 * 0.5)))], "1068 で満たし 1067 では満たさない・誤答の由来")

# ---- 大問11 1 標本の検定
E.check("s01p11a", ans("s01p11a") == "帰無仮説が正しいのに、棄却してしまう誤り", "第 1 種の過誤の定義")
rej = 0
for _ in range(20000):
    s_ = rng.normal(0, 1, 9); rej += abs(s_.mean() / (s_.std(ddof=1) / 3)) > 2.306
E.check("s01p11a", abs(rej / 20000 - 0.05) < 0.006, f"帰無仮説が正しいときに棄却する割合 {rej / 20000:.4f}（＝有意水準）")
t = (497 - 500) / (4.5 / 3)
E.check("s01p11b", t == -2.0 and abs(t) < stats.t.isf(0.025, 8) and abs(stats.t.isf(0.025, 8) - 2.306) < 5e-4 and abs(stats.t.isf(0.05, 8) - 1.860) < 5e-4 and ans("s01p11b").startswith("t ＝ −2.0 で、|t| ＜ 2.306"), f"t={t}、p 値 {2 * stats.t.sf(2, 8):.3f}")
E.check("s01p11b", f(-3 / 4.5, 2) == "-0.67" and -3 / (4.5 / 9) == -6.0, "誤答の t の由来")
z = (0.075 - 0.05) / sqrt(0.05 * 0.95 / 400)
E.choice("s01p11c", f"z ≈ {f(z, 2)} で、棄却する", f"z={z:.4f}、片側 p 値 {stats.norm.sf(z):.4f}")
E.check("s01p11c", round(0.025 / (0.05 * 0.95 / 400)) == 211 and f(0.025 / sqrt(0.05 * 0.95), 2) == "0.11" and f(stats.norm.cdf(z), 3) == "0.989", "誤答の値の由来（210.5 → 211）")

# ---- 大問12 2 標本の比較
pre = np.array([70, 68, 75, 80, 72, 66]); post = np.array([67, 63, 76, 76, 70, 61]); d = pre - post
E.check("s01p12a", d.mean() == 3 and abs(d.var(ddof=1) - 5.2) < 1e-12, "差の平均 3・不偏分散 5.2（問題文の値）")
tr = stats.ttest_rel(pre, post).statistic
E.choice("s01p12a", f(tr, 2), f"scipy の ttest_rel {tr:.4f}")
sd = d.std(ddof=1)
E.check("s01p12a", opts("s01p12a") == [f(3 / sd, 2), f(3 / (sd / 6), 2), f(3 / (d.std() / sqrt(6)), 2), f(3 / (5.2 / sqrt(6)), 2), f(tr, 2)], "誤答の値が理由どおり")
tab = np.array([[30, 20], [20, 30]]); chi = stats.chi2_contingency(tab, correction=False)
E.choice("s01p12b", f"χ² ＝ {f(chi.statistic, 1)} で、棄却する", f"χ²={chi.statistic}, p={chi.pvalue:.4f}")
E.check("s01p12b", chi.dof == 1 and abs(stats.chi2.isf(0.05, 1) - 3.841) < 5e-4 and abs(stats.chi2.isf(0.05, 4) - 9.488) < 5e-4, "自由度 1・表の値")
Fv = 36 / 12
E.check("s01p12c", abs(stats.f.isf(0.025, 10, 8) - 4.295) < 5e-4 and abs(stats.f.isf(0.05, 10, 8) - 3.347) < 5e-4 and abs(stats.f.isf(0.025, 8, 10) - 3.855) < 5e-4, "表の値")
E.choice("s01p12c", "4.295$ と比べて、棄却しない", f"F={Fv}、両側 p 値 {2 * stats.f.sf(Fv, 10, 8):.3f}")
E.check("s01p12c", Fv < 4.295, "F=3.0 < 4.295")

# ---- 大問13 回帰の出力
E.choice("s01p13a", f(0.150 / 0.025, 2), "t = 推定値 / 標準誤差")
E.check("s01p13a", abs(2 * stats.t.sf(1.60, 22) - 0.124) < 5e-4 and abs(2 * stats.t.sf(3.20 / 1.10, 22) - 0.008) < 5e-4 and f(3.20 / 1.10, 2) == "2.91", "出力の他の値（自由度 22）と整合")
E.check("s01p13b", ans("s01p13b").startswith("面積が同じなら、築年数が 1 年増えると") and 0.124 > 0.05, "築年数の係数の解釈と p 値 0.124 > 0.05")
n_, p_, R2 = 25, 2, 0.72
adj = 1 - (1 - R2) * (n_ - 1) / (n_ - p_ - 1)
E.choice("s01p13c", f(adj, 3), f"自由度調整済み決定係数 {adj:.5f}")
E.check("s01p13c", opts("s01p13c") == [f(R2, 3), f(1 - 0.28 * 24 / 21, 3), f(adj, 3), f(1 - 0.28 * 24 / 23, 3), f(1 - 0.28 * 22 / 24, 3)], "誤答の値が理由どおり")
# 自由度調整済み決定係数を実データで確認
Xd = rng.normal(0, 1, (25, 2)); yd = Xd @ [1, -0.5] + rng.normal(0, 1, 25); A = np.c_[np.ones(25), Xd]
res = yd - A @ np.linalg.lstsq(A, yd, rcond=None)[0]; r2 = 1 - res @ res / ((yd - yd.mean()) @ (yd - yd.mean()))
E.check("s01p13c", abs((1 - r2) * 24 / 22 - (res @ res / 22) / ((yd - yd.mean()) @ (yd - yd.mean()) / 24)) < 1e-12, "式が「残差の分散 ÷ 全体の分散」と一致")

# ---- 大問14 実験計画
E.check("s01p14a", ans("s01p14a") == "局所管理", "ブロックを作るのは局所管理")
g = [rng.normal(m_, 1, 5) for m_ in (0, 0.5, 1)]
E.choice("s01p14b", f((48 / 2) / (72 / 12), 2), "平均平方の比")
E.check("s01p14b", opts("s01p14b") == [f(48 / 72, 2), f((48 / 12) / (72 / 2), 2), f((48 / 2) / (72 / 14), 2), "4.00", f(6 / 24, 2)] and abs(stats.f.isf(0.05, 2, 12) - 3.885) < 5e-4, "誤答の値が理由どおり・解説の 3.885")
allg = np.concatenate(g); SA = sum(5 * (x_.mean() - allg.mean()) ** 2 for x_ in g); SE = sum(((x_ - x_.mean()) ** 2).sum() for x_ in g)
E.check("s01p14b", abs((SA / 2) / (SE / 12) - stats.f_oneway(*g).statistic) < 1e-9, "F の計算手順を scipy の f_oneway で確認")
E.done()
