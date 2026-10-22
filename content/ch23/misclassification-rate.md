---
id: misclassification-rate
title: 誤判別率
reading: ごはんべつりつ
en: Misclassification rate
aliases: [誤分類率, 誤判別確率, エラー率, 見かけの誤判別率]
chapter: 23
importance: A
tags: [判別分析, 判別の評価, 交差検証]
related: [linear-discriminant-analysis, mahalanobis-distance, bayes-discriminant-rule, confusion-matrix, cross-validation, leave-one-out-cv]
charts:
  - type: custom
    id: line-data
    params:
      x: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
      series:
        - { label: "誤判別率 Φ(−Δ/2)", y: [0.5, 0.4013, 0.3085, 0.2266, 0.1587, 0.1056, 0.0668, 0.0401, 0.0228] }
      xlabel: "2群間のマハラノビス距離 Δ"
      ylabel: "誤判別率"
    caption: 等分散共分散の正規分布・等しい事前確率のときの線形判別の誤判別率。Δ=0 では当て推量と同じ 0.5、Δ=2 で約 0.159、Δ=4 で約 0.023 まで下がる。
  - type: custom
    id: roc-curve
    params: { mu0: 0, mu1: 2, sigma: 1, threshold: 1 }
    sliders:
      - { param: threshold, label: "しきい値", min: -1, max: 3, step: 0.1 }
      - { param: mu1, label: "群2の平均（＝Δ）", min: 0, max: 4, step: 0.5 }
    caption: 判別得点を標準化した2群の分布（Δ=2）。しきい値が中点1のとき、両群の誤判別確率はどちらも 1−0.841=0.159 になる。
---

## 一言でいうと

判別ルールが個体を**間違った群に振り分ける確率**です。
判別方法の良し悪しを比べる最も基本的な物差しで、理論値と、データからの推定値（見かけの誤判別率、交差検証による推定）を区別して使います。

## 定義・数式

2群の判別で、群 $i$ の個体を群 $j$ と判別する確率を $P(j \mid i)$ と書きます。判別領域を $R_1, R_2$、群 $i$ の密度を $f_i$ とすると

$$
P(2 \mid 1) = \int_{R_2} f_1(\boldsymbol{x})\,d\boldsymbol{x}, \qquad P(1 \mid 2) = \int_{R_1} f_2(\boldsymbol{x})\,d\boldsymbol{x}
$$

で、事前確率 $\pi_1, \pi_2$ を重みにした**全体の誤判別率**は

$$
P(\text{誤}) = \pi_1 P(2 \mid 1) + \pi_2 P(1 \mid 2)
$$

です。

**正規分布のもとでの値**：2群が $N_p(\boldsymbol{\mu}_1, \Sigma)$、$N_p(\boldsymbol{\mu}_2, \Sigma)$ で、線形判別関数のしきい値を0（等しい事前確率）とすると

$$
P(2 \mid 1) = P(1 \mid 2) = \Phi\!\left(-\frac{\Delta}{2}\right), \qquad \Delta^2 = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \Sigma^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

です（$\Phi$ は標準正規分布の累積分布関数、$\Delta$ はマハラノビス距離）。しきい値を $c = \log(\pi_1/\pi_2)$ にすると

$$
P(2 \mid 1) = \Phi\!\left(-\frac{\Delta}{2} - \frac{c}{\Delta}\right), \qquad P(1 \mid 2) = \Phi\!\left(-\frac{\Delta}{2} + \frac{c}{\Delta}\right)
$$

となります。

**データからの推定**：

- **見かけの誤判別率**（再代入誤り率）：判別ルールを作ったデータにそのまま当てはめて、誤判別した割合。
- **交差検証（一つ抜き法）による推定**：1個を除いてルールを作り、除いた1個を判別することを全個体について繰り返したときの誤判別の割合。
- **代入法**：$\Delta$ を標本のマハラノビス距離 $D$ で置き換えた $\Phi(-D/2)$。

## 数式の読み解き

**$\Phi(-\Delta/2)$ の導出**　線形判別関数

$$
z(\boldsymbol{x}) = \boldsymbol{a}^\top\left(\boldsymbol{x} - \frac{\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2}{2}\right), \qquad \boldsymbol{a} = \Sigma^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)
$$

は正規分布に従う $\boldsymbol{x}$ の1次式なので、$z$ も正規分布に従います。$\boldsymbol{x} \sim N_p(\boldsymbol{\mu}_1, \Sigma)$ のとき、

$$
E[z] = \boldsymbol{a}^\top\left(\boldsymbol{\mu}_1 - \frac{\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2}{2}\right) = -\frac{1}{2}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) = -\frac{\Delta^2}{2}
$$

$$
V[z] = \boldsymbol{a}^\top \Sigma \boldsymbol{a} = (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma^{-1} \Sigma\, \Sigma^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) = \Delta^2
$$

です。よって群1のもとで $z \sim N(-\Delta^2/2,\ \Delta^2)$ で、

$$
P(2 \mid 1) = P(z > 0) = P\!\left(\frac{z + \Delta^2/2}{\Delta} > \frac{\Delta}{2}\right) = 1 - \Phi\!\left(\frac{\Delta}{2}\right) = \Phi\!\left(-\frac{\Delta}{2}\right)
$$

となります。群2のもとでは $z \sim N(\Delta^2/2,\ \Delta^2)$ なので、対称に $P(1 \mid 2) = \Phi(-\Delta/2)$ です。しきい値が $c$ のときは、$P(z > c)$ を同じように標準化すれば上の式が得られます。

- 誤判別率は $\Delta$ だけで決まります。変数の数や単位によらず、「2群がばらつきの何個分離れているか」で性能が決まるということです。
- 見かけの誤判別率は、同じデータでルールを作って評価するので**楽観的（小さめ）**になります。一つ抜き法はこの偏りを減らします。

## 具体例

**例1：理論値を計算する**

2群のマハラノビス距離が $\Delta = 2$ なら、等しい事前確率のもとで

$$
P(\text{誤}) = \Phi(-1) \approx 0.1587
$$

です。群2が少なく $\pi_1 = 0.9,\ \pi_2 = 0.1$ のとき、ベイズ判別のしきい値は $c = \log 9 \approx 2.197$ で、$c/\Delta \approx 1.099$ です。

$$
P(2 \mid 1) = \Phi(-1 - 1.099) = \Phi(-2.099) \approx 0.0179, \qquad P(1 \mid 2) = \Phi(-1 + 1.099) = \Phi(0.099) \approx 0.5393
$$

全体では $0.9 \times 0.0179 + 0.1 \times 0.5393 \approx 0.0701$ です。しきい値0のままなら全体は $0.1587$ なので、事前確率を使うと全体の誤判別率は下がります。ただし群2の半分以上を見逃すことになり、群2の見逃しが重大なら損失を考えたしきい値が必要です。

**例2：小さなデータで3種類の推定値を比べる**

線形判別分析の例のデータ（各群4個、2変数）では、判別関数は $z = (4x_1 + x_2 - 19)/3$、標本のマハラノビス距離の2乗は $D^2 = 14/3$（$D \approx 2.160$）でした。

| 推定方法 | 計算 | 値 |
|---|---|---|
| 見かけの誤判別率 | (4, 5) と (3, 5) を誤判別、2/8 | 0.250 |
| 一つ抜き法 | 1個ずつ除いて作り直したルールで、同じ2個だけを誤判別、2/8 | 0.250 |
| 代入法 | $\Phi(-2.160/2) = \Phi(-1.080)$ | 0.140 |

一つ抜き法では、たとえば (4, 5) を除くと群1の平均や $S$ が変わり、残りで作ったルールでは (4, 5) の判別得点が $2.222 > 0$ となって、やはり群2に誤判別されます。
代入法は正規性を前提にした値で、標本が小さいときは $D$ が $\Delta$ を過大に推定しやすいため、楽観的になりがちです。

## 現場での使われ方

- **医療**：診断ルールを新しく作ったとき、別の患者集団で誤判別率を確かめ、作成時のデータでの成績が過大でないかを確認します。
- **製造・品質管理**：良品を不良と判定する率（過検出）と不良を良品と判定する率（見逃し）を分けて報告し、工程ごとに許容できる水準と比べます。
- **IT・Webサービス**：分類モデルを交差検証で評価し、モデルや変数の組み合わせを選ぶ基準にします。
- **金融・保険**：審査モデルの誤判別率を、貸し倒れ率の大小で重み付けした損失に換算して比較します。

## グラフ・イメージ

1つ目の図は、誤判別率 $\Phi(-\Delta/2)$ が距離 $\Delta$ とともに下がる様子です。$\Delta$ が小さいうちは急に下がり、大きくなると下がり方が緩やかになります。
2つ目の図では、群2の平均（ここでは $\Delta$ と同じ）を動かすと2つの山の重なりが変わります。しきい値を中点から動かすと、片方の誤判別は減り、もう片方は増えます。

## 試験のポイント

- 判別関数 $z$ の群ごとの分布 $N(\mp\Delta^2/2,\ \Delta^2)$ を求め、$\Phi(-\Delta/2)$ を導く流れは重要です。
- $\Delta$ を与えて誤判別率を数表から求めさせる計算が典型です。$\Phi(-1) \approx 0.1587$、$\Phi(-0.5) \approx 0.3085$、$\Phi(-1.5) \approx 0.0668$ などはすぐ出せるようにします。
- 事前確率でしきい値をずらしたときの $P(2 \mid 1)$、$P(1 \mid 2)$ の式も押さえます。
- 見かけの誤判別率が楽観的になる理由と、一つ抜き法（交差検証）による補正を説明できるようにします。

## よくある間違い

- × 誤判別率を $\Phi(-\Delta)$ とする → ○ しきい値は2つの平均の中点にあり、各平均から境界までは $\Delta/2$ です。
- × $\Delta^2$ を使って $\Phi(-\Delta^2/2)$ とする → ○ 標準化すると平均 $\Delta^2/2$ を標準偏差 $\Delta$ で割るので、$\Delta/2$ になります。
- × 見かけの誤判別率を、新しいデータでの性能とみなす → ○ 作成に使ったデータでの評価は甘くなります。別のデータや交差検証で評価します。
- × 全体の誤判別率が小さければ十分 → ○ 群の大きさが極端に違うと、少ない群をすべて見逃しても全体の誤判別率は小さくなります。$P(2 \mid 1)$ と $P(1 \mid 2)$ を分けて見ます（感度・特異度）。
