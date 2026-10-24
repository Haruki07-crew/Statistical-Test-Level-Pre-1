"""2級 第2章（1変数データの要約）：問題の検算。"""
import os
import sys
import pathlib
os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qcheck import Q  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

q = Q(2)
c = q.qs
ans = lambda i: str(c[i]["choices"][c[i]["answer"]])


def quart(v):
    s = sorted(v); n = len(s); h = n // 2
    return np.median(s[:h]), np.median(s), np.median(s[h + 1:] if n % 2 else s[h:])


x = [3, 7, 8, 8, 9, 12, 16]
q.num("q02-mean-median-mode-1", np.mean(x))
y = [2, 3, 5, 6, 9, 11]
q.check("q02-mean-median-mode-2", ans("q02-mean-median-mode-2") == str(np.median(y)) and np.mean(y) == 6, f"中央値 {np.median(y)}（6 は平均）")
z = np.array([1, 4, 6, 7, 9.]); z2 = z.copy(); z2[-1] = 1000
q.check("q02-mean-median-mode-3", np.median(z2) == np.median(z) and z2.mean() != z.mean(), "最大値を変えても中央値は不変（answer=false）")
q.num("q02-mean-median-mode-4", (600 - 30) / 9)
d = np.array([4, 6, 8, 10, 12.])
q.num("q02-sample-variance-sd-1", d.var(ddof=1))
rng = np.random.default_rng(5); w = rng.normal(0, 3, 50); w = (w - w.mean()) / w.std(ddof=1) * 3
q.check("q02-sample-variance-sd-2", abs((2 * w + 5).std(ddof=1) - 6) < 1e-9 and ans("q02-sample-variance-sd-2") == "6", "2x+5 の標準偏差は 6")
q.num("q02-sample-variance-sd-3", (290 - 10 * 5 ** 2) / 9)
q.concept("q02-sample-variance-sd-4", "単位の性質（answer=true）")
q1, _, q3 = quart([1, 3, 4, 6, 7, 9, 10, 12])
q.num("q02-quartiles-iqr-1", q3 - q1)
v = np.array([2, 5, 6, 7, 8, 9, 11, 13.]); v2 = v.copy(); v2[-1] *= 10
ch = {"範囲": lambda a: a.max() - a.min(), "四分位範囲": lambda a: quart(a)[2] - quart(a)[0], "中央値": np.median, "第 1 四分位数": lambda a: quart(a)[0]}
diff = {k: abs(f(v2) - f(v)) for k, f in ch.items()}
q.check("q02-quartiles-iqr-2", max(diff, key=diff.get) == ans("q02-quartiles-iqr-2") and sum(1 for k in diff if diff[k] > 0) == 1, f"変化 {diff}")
big = rng.normal(0, 1, 100001); a1, _, a3 = quart(big)
q.check("q02-quartiles-iqr-3", abs(np.mean((big >= a1) & (big <= a3)) - 0.5) < 0.01, "箱の中に約 50%（answer=true）")
q.num("q02-box-plot-1", 30 + 1.5 * (30 - 20))
sk = rng.lognormal(0, 0.8, 2001); s1, s2, s3 = quart(sk)
q.check("q02-box-plot-2", (s2 - s1) < (s3 - s2) and (sk.max() - s3) > (s1 - sk.min()) and ans("q02-box-plot-2") == "右に裾が長い", "右に裾が長い分布では中央値が左寄り・右のひげが長い")
q.concept("q02-box-plot-3", "平均は箱ひげ図に表れない")
q.concept("q02-box-plot-4", "箱の長さは IQR で個数と無関係（answer=false）")
q.num("q02-standard-score-1", (64 - 52) / 8)
q.num("q02-standard-score-2", 50 + 10 * -0.8)
zs = {"数学": (70 - 60) / 5, "英語": (80 - 70) / 10, "国語": (75 - 65) / 8}
q.check("q02-standard-score-3", max(zs, key=zs.get) == ans("q02-standard-score-3"), f"標準得点 {zs}")
e = rng.exponential(1, 100000); ez = (e - e.mean()) / e.std()
q.check("q02-standard-score-4", np.mean(ez ** 3) > 1.5, f"標準化しても歪度 {np.mean(ez ** 3):.2f}（answer=false）")
inc = np.array([10, 20, 30, 40.]); cs = np.cumsum(inc) / inc.sum()
area = sum(0.25 * (a + b) / 2 for a, b in zip(np.r_[0, cs[:-1]], cs))
g_md = np.abs(inc[:, None] - inc[None, :]).sum() / (2 * 16 * inc.mean())
q.num("q02-lorenz-curve-gini-1", 1 - 2 * area)
q.check("q02-lorenz-curve-gini-1", abs(g_md - 0.25) < 1e-12, "差の平均の式でも 0.25")
eq = np.ones(5)
q.check("q02-lorenz-curve-gini-2", np.abs(eq[:, None] - eq[None, :]).sum() == 0 and ans("q02-lorenz-curve-gini-2") == "0", "完全平等で 0")
q.concept("q02-lorenz-curve-gini-3", "少ない順に累積する（answer=false）")
h = np.r_[np.full(80, 50 / 80), np.full(20, 50 / 20)]; cum = np.cumsum(np.sort(h)) / h.sum()
q.check("q02-lorenz-curve-gini-4", abs(cum[79] - 0.5) < 1e-12 and ans("q02-lorenz-curve-gini-4") == "(0.8, 0.5)", "下位 80% が 50% → 点 (0.8, 0.5)")
sav = np.random.default_rng(22).lognormal(6.5, 0.9, 100000)
q.check("q02-skewness-kurtosis-g1", sav.mean() > np.median(sav) and stats.skew(sav) > 0 and ans("q02-skewness-kurtosis-g1") == "右に裾が長く、歪度は正と考えられる", f"右に裾の長い分布で 平均 {sav.mean():.0f} ＞ 中央値 {np.median(sav):.0f}、歪度 {stats.skew(sav):.2f}")
q.num("q02-coefficient-of-variation-g1", 12 / 80)
q.num("q02-weighted-geometric-harmonic-mean-g1", 2 / (1 / 30 + 1 / 60))
tc_ = np.random.default_rng(23).normal(10, 4, 1000); tf_ = tc_ * 9 / 5 + 32; wt_ = np.random.default_rng(24).normal(60, 9, 1000)
q.check("q02-coefficient-of-variation-g2", abs(tc_.std() / tc_.mean() - tf_.std() / tf_.mean()) > 0.1 and abs(wt_.std() / wt_.mean() - (wt_ * 1000).std() / (wt_ * 1000).mean()) < 1e-12 and ans("q02-coefficient-of-variation-g2").startswith("1 日の気温"), f"摂氏 {tc_.std() / tc_.mean():.3f}・華氏 {tf_.std() / tf_.mean():.3f}（間隔尺度は単位で変わる）、kg と g では同じ")
t5_ = stats.t(5, scale=np.sqrt(3 / 5))
q.check("q02-skewness-kurtosis-g2", abs(t5_.var() - 1) < 1e-12 and t5_.stats(moments="k") > 0 and t5_.sf(3) > stats.norm.sf(3) * 2 and t5_.stats(moments="s") == 0 and ans("q02-skewness-kurtosis-g2").startswith("正規分布に比べて裾が重く"), f"分散 1 にそろえた t(5)：尖度 {float(t5_.stats(moments='k')):.1f}、P(X>3) {t5_.sf(3):.4f} ＞ 正規 {stats.norm.sf(3):.4f}、歪度 0")
q.num("q02-weighted-geometric-harmonic-mean-g2", np.average([60, 70, 52], weights=[30, 20, 50]))
q.check("q02-weighted-geometric-harmonic-mean-g2", abs(np.mean([60, 70, 52]) - 60.67) < 0.005, "単純な平均は 60.7")
q.done()
