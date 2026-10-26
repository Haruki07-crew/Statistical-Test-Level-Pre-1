"""教科書 第56章（セミパラメトリック推論と予測の不確実性）の検算。"""
import numpy as np

from causal_sim import expit
from conformal_sim import run as conf_run
from dml_sim import run as dml_run
from tbcheck import allclose, body, chart, check, close, done

T = "conformal-prediction"
c, nn, mc, mn, w = conf_run()
p = chart(T)
check(T, "図：x の区間ごとの被覆率（同じ乱数で計算し直す）", allclose(p["series"][0]["y"], c, 6e-5) and allclose(p["series"][1]["y"], nn, 6e-5))
check(T, "例2 全体 0.901・0.923、左端 0.997・右端 0.683、幅の平均 3.80", close(mc, 0.901, 5e-4) and close(mn, 0.923, 5e-4)
      and close(c[0], 0.997, 5e-4) and close(c[-1], 0.683, 5e-4) and close(w, 3.80, 0.005))
c19, _, mc19, mn19, _ = conf_run(ncal=19, reps=4000)
check(T, "例1 n=19：k=18・被覆率 0.900（正規の区間 0.923）", int(np.ceil(20 * 0.9)) == 18 and close(mc19, 0.900, 5e-4) and close(mn19, 0.923, 5e-4), (mc19, mn19))
# 有限標本の保証：交換可能なら被覆率は 1−α 以上 1−α+1/(n+1) 以下（いろいろな n で、正確な値 k/(n+1)）
check(T, "保証の上下限：k/(n+1) は [1−α, 1−α+1/(n+1)] に入る", all(0.9 <= np.ceil((n + 1) * 0.9) / (n + 1) <= 0.9 + 1 / (n + 1) + 1e-12 for n in range(9, 500)))

T = "semiparametric-efficiency"
rng = np.random.default_rng(0)
x1, x2 = rng.normal(size=4_000_000), rng.normal(size=4_000_000)
e = expit(-0.3 + 0.8 * x1 + 0.6 * x2)
V = np.mean(1 / e + 1 / (1 - e)) + 0.25
check(T, "例 限界 5.70・SD 0.0338・無作為化 0.0292・約 16%", close(V, 5.70, 0.005) and close(np.sqrt(V / 5000), 0.0338, 5e-5)
      and close(np.sqrt(4.25 / 5000), 0.0292, 5e-5) and close(np.sqrt(V / 5000) / np.sqrt(4.25 / 5000) - 1, 0.16, 0.005))
check(T, "τ(X)=2+0.5X₁ の分散 0.25", close((0.5 * x1).var(), 0.25, 0.001))
# 平均の効率的影響関数：標本平均の n·分散 = V[Y]
y = rng.exponential(2, (20000, 200))
check(T, "平均：標本平均の n·分散 ≈ V[Y]（指数分布 平均 2 → 4）", close(200 * y.mean(1).var(), 4, 0.15))
check(T, "本文の比較の値（AIPW 0.035・IPW 0.094・10 層 0.071）は第55章の検算と同じ", all(s in body(T) for s in ("0.035", "0.094", "0.071")))

T = "double-machine-learning"
r = dml_run()
b = {k: r[k].mean() - 1 for k in ("ols", "dml", "nocross", "half")}
check(T, "例 偏り：OLS +0.313・直交しない −0.509・DML +0.009（SD 0.039・SE 0.035・被覆 0.93）・交差適合なし +0.002",
      close(b["ols"], 0.313, 5e-4) and close(b["half"], -0.509, 5e-4) and close(b["dml"], 0.009, 5e-4) and close(r["dml"].std(), 0.039, 5e-4)
      and close(r["dml_se"].mean(), 0.035, 5e-4) and close((np.abs(r["dml"] - 1) <= 1.96 * r["dml_se"]).mean(), 0.93, 1e-9) and close(b["nocross"], 0.002, 5e-4), b)
xx = np.random.default_rng(0).uniform(-2, 2, (2_000_000, 2))
mm = np.tanh(2 * xx[:, 0]) + 0.5 * xx[:, 1]
check(T, "直交しない推定の極限 1/(1+E[m²]) − 1 ≈ −0.52", close(1 / (1 + (mm * mm).mean()) - 1, -0.52, 0.005))
done("第56章（セミパラメトリック推論と予測の不確実性）")
