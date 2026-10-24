"""2級 第1章（データの種類とグラフ）：問題の検算。"""
import os
import sys
import pathlib
os.environ.setdefault("JD_LEVEL", "g2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qcheck import Q  # noqa: E402
import numpy as np  # noqa: E402

q = Q(1)
c = q.qs
q.check("q01-data-types-1", c["q01-data-types-1"]["choices"][c["q01-data-types-1"]["answer"]].startswith("通学"), "連続の量的変数は通学時間（他は名義・名義・離散）")
q.concept("q01-data-types-2", "順序のある質的変数（answer=false）")
q.check("q01-data-types-3", c["q01-data-types-3"]["choices"][c["q01-data-types-3"]["answer"]] == "最頻値" and max([12, 25, 8, 15]) == 25, "名義の代表値は最頻値（自転車 25 人）")
q.check("q01-scales-of-measurement-1", c["q01-scales-of-measurement-1"]["choices"][c["q01-scales-of-measurement-1"]["answer"]] == "間隔尺度", "摂氏は間隔尺度")
q.concept("q01-scales-of-measurement-2", "順位は順序尺度（answer=false）")
q.num("q01-bar-pie-band-chart-1", 0.15 * 360)
q.check("q01-bar-pie-band-chart-2", 200 * 0.4 == 80 and 150 * 0.5 == 75 and c["q01-bar-pie-band-chart-2"]["answer"] == 0, "1 年 80 人 > 2 年 75 人")
q.check("q01-bar-pie-band-chart-3", c["q01-bar-pie-band-chart-3"]["choices"][c["q01-bar-pie-band-chart-3"]["answer"]] == "折れ線グラフ", "時間の変化は折れ線")
q.num("q01-frequency-distribution-histogram-1", np.dot([10, 30, 50], [5, 12, 8]) / 25)
q.check("q01-frequency-distribution-histogram-2", "÷" in c["q01-frequency-distribution-histogram-2"]["choices"][c["q01-frequency-distribution-histogram-2"]["answer"]], "高さは度数÷幅")
raw = [1, 2, 9, 11, 19] + [21] * 12 + [41] * 8  # 0〜20 に 5、20〜40 に 12、40〜60 に 8 の一例
q.check("q01-frequency-distribution-histogram-3", abs(np.mean(raw) - 32.4) > 0.5 and len(raw) == 25, f"同じ度数分布でも元の平均は {np.mean(raw):.2f}（answer=false）")
f = np.array([3, 7, 10, 5]); cum = np.cumsum(f)
q.num("q01-cumulative-frequency-1", cum[2] / 25)
q.check("q01-cumulative-frequency-2", cum[1] < 13 <= cum[2] and c["q01-cumulative-frequency-2"]["answer"] == 2, f"13 番目は 20〜30（累積 {list(cum)}）")
q.concept("q01-cumulative-frequency-3", "点は上端に打つ（answer=false）")
x = [32, 35, 41, 44, 44, 48, 50, 53, 56, 62]
q.num("q01-stem-and-leaf-plot-1", np.median(x))
q.check("q01-stem-and-leaf-plot-2", np.median(x[5:]) == 53 and np.median(x[6:]) == 54.5 and c["q01-stem-and-leaf-plot-2"]["choices"][c["q01-stem-and-leaf-plot-2"]["answer"]] == "53", "上の半分 5 個の中央値 53")
rng = np.random.default_rng(1)
y = rng.lognormal(0, 0.6, 200000); hist, edges = np.histogram(y, 400); mode = edges[hist.argmax()]
q.check("q01-distribution-shape-1", mode < np.median(y) < y.mean() and c["q01-distribution-shape-1"]["answer"] == 1, f"対数正規で 最頻 {mode:.2f} < 中央 {np.median(y):.2f} < 平均 {y.mean():.2f}")
mix = np.r_[rng.normal(10, 2, 5000), rng.normal(40, 4, 5000)]
q.check("q01-distribution-shape-2", abs(mix.mean() - 25) < 0.5 and np.mean(np.abs(mix - mix.mean()) < 2) < 0.01, f"二峰の平均 {mix.mean():.1f} の近くのデータは 1% 未満（answer=true）")
s = 100 - rng.exponential(8, 200000)
q.check("q01-distribution-shape-3", s.mean() < np.median(s) and c["q01-distribution-shape-3"]["answer"] == 0, f"左に裾が長いと 平均 {s.mean():.2f} < 中央値 {np.median(s):.2f}")
q.concept("q01-official-statistics-1", "国勢調査は全数調査（answer=false）")
q.concept("q01-official-statistics-2", "他機関が集めたデータの利用が二次データ")
q.done()
