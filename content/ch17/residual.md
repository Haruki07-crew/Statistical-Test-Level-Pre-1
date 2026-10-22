---
id: residual
title: 残差
reading: ざんさ
en: Residual
aliases: [回帰残差, 素の残差]
chapter: 17
importance: A
tags: [回帰診断, 最小二乗法, ハット行列]
related: [hat-matrix, least-squares-method, multiple-regression-model, standardized-residual, leverage, residual-plot]
charts:
  - type: custom
    id: regression-residual
    params: { n: 20, beta0: 1, beta1: 0.5, sigma: 1, outlier: 0 }
    sliders:
      - { param: sigma, label: "誤差の標準偏差 σ", min: 0.2, max: 3, step: 0.1 }
      - { param: outlier, label: "右端の点の y のずれ", min: -8, max: 8, step: 0.5 }
    caption: 灰色の縦線が各点の残差（観測値と回帰直線の縦の差）。右端の点をずらしても、その点の残差は見た目ほど大きくならない（直線のほうが点に引き寄せられる）。
---

## 一言でいうと

残差は「観測値 − 当てはめた値」で、回帰モデルが説明しきれなかった部分です。
目に見えない誤差 $\varepsilon_i$ の代わりに残差を調べて、モデルの仮定が成り立っているかを点検します（回帰診断）。

## 定義・数式

重回帰モデル $\boldsymbol{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$（$X$ は $n\times(p+1)$ でフルランク、$E[\boldsymbol{\varepsilon}]=\boldsymbol{0}$、$V[\boldsymbol{\varepsilon}]=\sigma^2 I_n$）を最小二乗法で当てはめ、$\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}$ とします。
$p$ は説明変数の個数で、切片を含めた係数の個数は $p+1$ です。

$$
e_i = y_i - \hat{y}_i,\qquad
\boldsymbol{e} = \boldsymbol{y} - \hat{\boldsymbol{y}} = (I - H)\,\boldsymbol{y},\qquad
H = X(X^\top X)^{-1}X^\top
$$

$H$ はハット行列です。残差は次の性質をもちます。

| 性質 | 式 |
|---|---|
| 期待値 | $E[\boldsymbol{e}] = \boldsymbol{0}$ |
| 分散共分散行列 | $V[\boldsymbol{e}] = \sigma^2 (I-H)$ |
| 各残差の分散 | $V[e_i] = \sigma^2(1-h_{ii})$（$h_{ii}$ はてこ比） |
| 説明変数との直交 | $X^\top\boldsymbol{e} = \boldsymbol{0}$（切片があれば $\sum_i e_i = 0$） |
| 当てはめ値との直交 | $\hat{\boldsymbol{y}}^\top \boldsymbol{e} = 0$ |
| 残差平方和と $\sigma^2$ の推定 | $s^2 = \dfrac{\sum_i e_i^2}{n-p-1}$ は $\sigma^2$ の不偏推定量 |

## 数式の読み解き

- $\varepsilon_i$：**誤差**。真の直線（平面）からのずれで、観測できません。
- $e_i$：**残差**。推定した直線からのずれで、データから計算できます。
- $h_{ii}$：ハット行列の対角成分（てこ比）。$x_i$ が説明変数の中心から離れているほど大きくなります。

**残差は誤差で書ける。** $H$ は $HX = X$ をみたすので $(I-H)X = O$ です。よって

$$
\boldsymbol{e} = (I-H)\boldsymbol{y} = (I-H)(X\boldsymbol{\beta} + \boldsymbol{\varepsilon}) = (I-H)\boldsymbol{\varepsilon}
$$

となり、$E[\boldsymbol{e}] = (I-H)E[\boldsymbol{\varepsilon}] = \boldsymbol{0}$ です。

**分散の導出。** $I-H$ は対称（$(I-H)^\top = I-H$）かつべき等（$(I-H)^2 = I-H$）です。べき等は $H^2 = X(X^\top X)^{-1}X^\top X(X^\top X)^{-1}X^\top = H$ から従います。したがって

$$
V[\boldsymbol{e}] = (I-H)\,V[\boldsymbol{\varepsilon}]\,(I-H)^\top = \sigma^2 (I-H)(I-H) = \sigma^2 (I-H)
$$

対角成分を取り出すと $V[e_i] = \sigma^2(1-h_{ii})$ です。

ここから2つの大事な事実がわかります。

1. 誤差の分散がそろっていても（等分散でも）、**残差の分散はそろいません**。てこ比の大きい点ほど残差は小さくなりがちです。直線がその点に引き寄せられるからです。
2. 非対角成分 $-\sigma^2 h_{ij}$ が0でないので、誤差が独立でも**残差どうしは相関します**。

この「ばらつきの不ぞろい」を直したものが標準化残差・スチューデント化残差です。

**直交性。** 正規方程式 $X^\top(\boldsymbol{y} - X\hat{\boldsymbol{\beta}}) = \boldsymbol{0}$ がそのまま $X^\top\boldsymbol{e}=\boldsymbol{0}$ です。$X$ の1列目がすべて1なら、その行から $\sum e_i = 0$ が出ます。さらに $\hat{\boldsymbol{y}}^\top\boldsymbol{e} = \hat{\boldsymbol{\beta}}^\top X^\top\boldsymbol{e} = 0$ です。

**$s^2$ の分母。** $E[\boldsymbol{e}^\top\boldsymbol{e}] = \sigma^2\,\mathrm{tr}(I-H) = \sigma^2(n-p-1)$ なので、$n-p-1$ で割ると不偏になります。$\mathrm{tr}(H) = p+1$ の証明はてこ比の項目で示します。

## 具体例

**例1：8店舗の広告費と売上（この章で共通に使うデータ）**

広告費 $x$（十万円）と売上 $y$（百万円）を8店舗で記録しました。8店舗目だけ広告費が大きい店です。

| 店 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 12 |
| $y$ | 3 | 5 | 6 | 9 | 10 | 13 | 14 | 20 |

$\bar{x}=5$、$\bar{y}=10$、$S_{xx}=\sum(x_i-\bar{x})^2=84$、$S_{xy}=133$ より

$$
\hat{\beta}_1 = \frac{133}{84} = 1.5833,\qquad \hat{\beta}_0 = 10 - 1.5833\times 5 = 2.0833
$$

当てはめ値と残差は次のとおりです。

| 店 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $\hat{y}_i$ | 3.6667 | 5.2500 | 6.8333 | 8.4167 | 10.0000 | 11.5833 | 13.1667 | 21.0833 |
| $e_i$ | −0.6667 | −0.2500 | −0.8333 | 0.5833 | 0.0000 | 1.4167 | 0.8333 | −1.0833 |

残差の和は0、$\sum x_i e_i = 0$ になっています（直交性の確認）。
残差平方和は $\sum e_i^2 = 5.4167$、$s^2 = 5.4167/(8-2) = 0.9028$、$s = 0.9501$ です。

**例2：残差の標準偏差は点ごとに違う**

例1のてこ比は店5で $h_{55}=1/8=0.125$、店8で $h_{88}=1/8+49/84=0.7083$ です。$\sigma$ を $s$ で置き換えると

$$
\sqrt{s^2(1-h_{55})} = \sqrt{0.9028\times0.875} = 0.8888,\qquad
\sqrt{s^2(1-h_{88})} = \sqrt{0.9028\times0.2917} = 0.5131
$$

店8の残差は、店5の残差の6割弱しかばらつきません。素の残差の大きさだけで外れ値を判断すると、店8のような点を見逃します。

## 現場での使われ方

- **製造・品質管理**：工程条件から寸法を予測する回帰で、残差を時系列に並べて工程の異常（ずれや周期）を見つけます。
- **金融・保険**：保険料の算定モデルで、残差の大きい契約を抜き出して、データ入力の誤りやモデルにない要因を調べます。
- **マーケティング**：広告費から売上を予測し、残差が大きくプラスの店舗を「予測以上に売れている店」として事例を調べます。
- **行政・調査**：地域の指標を回帰で調整したうえで、残差を「条件をそろえたときの成績」として地域比較に使います。

## グラフ・イメージ

点から回帰直線へ下ろした灰色の縦線が残差です。
「右端の点の y のずれ」を動かすと、実線（全データの直線）がその点に引き寄せられ、点そのものの残差は小さめに保たれます。
てこ比の大きい点で残差の分散 $\sigma^2(1-h_{ii})$ が小さくなる様子が目で確かめられます。

## 試験のポイント

- $\boldsymbol{e} = (I-H)\boldsymbol{\varepsilon}$、$V[\boldsymbol{e}]=\sigma^2(I-H)$、$V[e_i]=\sigma^2(1-h_{ii})$ の導出（$I-H$ の対称性とべき等性）を書けるようにします。
- 切片のあるモデルでは $\sum e_i=0$、$\sum x_{ij}e_i=0$、$\sum \hat{y}_i e_i=0$ です。選択肢の正誤問題で頻繁に使います。
- $\sigma^2$ の不偏推定量は残差平方和を $n-p-1$（単回帰なら $n-2$）で割ったものです。
- 残差どうしは独立ではありません（共分散 $-\sigma^2 h_{ij}$）。
- 残差プロットは $y_i$ ではなく $\hat{y}_i$ を横軸にします。$\boldsymbol{e}$ と $\hat{\boldsymbol{y}}$ は無相関ですが、$\boldsymbol{e}$ と $\boldsymbol{y}$ は相関するからです。

## よくある間違い

- × 残差と誤差は同じもの → ○ 誤差 $\varepsilon_i$ は観測できない真のずれ、残差 $e_i$ は推定した直線からのずれです。残差は分散が不ぞろいで、互いに相関します。
- × 誤差が等分散なら残差も等分散 → ○ $V[e_i]=\sigma^2(1-h_{ii})$ で、てこ比の大きい点ほど小さくなります。
- × 残差平方和を $n$ で割って $\sigma^2$ を推定する → ○ 推定した係数の個数 $p+1$ を引いた $n-p-1$ で割ります。
- × 切片のないモデルでも残差の和は0 → ○ $\sum e_i=0$ は切片（1の列）があるときに限ります。
