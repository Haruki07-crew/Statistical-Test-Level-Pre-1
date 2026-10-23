---
id: multiple-regression-model
title: 重回帰モデル
reading: じゅうかいきもでる
en: Multiple linear regression model
aliases: [線形回帰モデル, 重回帰, 線形重回帰モデル, 線形模型]
chapter: 16
importance: A
tags: [回帰分析, 行列表示, 偏回帰係数, 誤差の仮定]
related: [least-squares-method, normal-equations, gauss-markov-theorem, multivariate-normal-distribution, variance-covariance-matrix, partial-correlation-coefficient]
charts:
  - type: scatter
    model: regression
    params: { beta0: 1, beta1: 0.8, sigma: 1, n: 50 }
    sliders:
      - { param: beta1, label: "傾き β1", min: -2, max: 2, step: 0.1 }
      - { param: sigma, label: "誤差の標準偏差 σ", min: 0.1, max: 3, step: 0.1 }
      - { param: n, label: "データ数 n", min: 10, max: 300, step: 10 }
    caption: 説明変数が1つの場合（単回帰）のイメージ。真の直線 β0+β1x のまわりに、標準偏差 σ の誤差がのってデータが散らばる。σ を大きくすると点が直線から離れ、推定値がぶれやすくなる。
  - type: custom
    id: regression-residual
    params: { n: 20, beta0: 1, beta1: 0.5, sigma: 1, outlier: 0 }
    sliders:
      - { param: sigma, label: "誤差の標準偏差 σ", min: 0.2, max: 3, step: 0.1 }
    caption: 当てはめた直線と各点からの縦の線（残差）。重回帰でも、残差は「観測値 − 当てはめ値」の縦方向のずれとして同じように定義される。
---

## 一言でいうと

1つの目的変数を、複数の説明変数の**一次式**と、説明しきれない**誤差**の和で表すモデルです。
「他の変数を固定したとき、この変数が1増えると目的変数が平均でどれだけ変わるか」を、変数ごとに切り分けて調べられます。

## 定義・数式

$n$ 個の観測 $(x_{i1},\dots,x_{ip},\,y_i)$（$i=1,\dots,n$）について、

$$
y_i=\beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i
$$

とおくのが重回帰モデルです。行列で書くと

$$
\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon},\qquad
X=\begin{pmatrix}1&x_{11}&\cdots&x_{1p}\\ \vdots&\vdots&&\vdots\\ 1&x_{n1}&\cdots&x_{np}\end{pmatrix},\quad
\boldsymbol{\beta}=\begin{pmatrix}\beta_0\\ \vdots\\ \beta_p\end{pmatrix}
$$

です。$\boldsymbol{y},\boldsymbol{\varepsilon}$ は $n$ 次元、$X$ は $n\times(p+1)$ の**計画行列**です。

標準的な仮定は次のとおりです。

1. $X$ は確率変数ではない（固定された値）とし、$\operatorname{rank}X=p+1$（列が一次独立、$n>p+1$）。
2. $E[\boldsymbol{\varepsilon}]=\boldsymbol{0}$。
3. $V[\boldsymbol{\varepsilon}]=\sigma^2I_n$（誤差は無相関で、分散が等しい）。
4. （検定・区間推定をするとき）$\boldsymbol{\varepsilon}\sim N_n(\boldsymbol{0},\sigma^2I_n)$。

仮定 2, 3 のもとで

$$
E[\boldsymbol{y}]=X\boldsymbol{\beta},\qquad V[\boldsymbol{y}]=\sigma^2I_n
$$

であり、仮定 4 を加えると $\boldsymbol{y}\sim N_n(X\boldsymbol{\beta},\sigma^2I_n)$ です。

## 数式の読み解き

- $\beta_0$：切片。すべての $x$ が 0 のときの $y$ の平均。
- $\beta_j$（$j\ge1$）：**偏回帰係数**。他の説明変数を一定に保ったまま $x_j$ だけを1増やしたときの、$y$ の平均の変化量です。
- $\varepsilon_i$：誤差。説明変数で表せない部分をまとめたものです。
- $X$ の1列目がすべて 1 なのは、切片 $\beta_0$ を $\boldsymbol{\beta}$ の成分として一緒に扱うためです。
- 「線形」は**パラメータ $\boldsymbol{\beta}$ について一次式**という意味です。$x^2$ や $\log x$ を説明変数に入れても、$\boldsymbol{\beta}$ について一次なら線形モデルです。

$\boldsymbol{\beta}$ は**最小二乗法**で推定し、推定量は

$$
\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}
$$

です（導出は「正規方程式」を参照）。$\operatorname{rank}X=p+1$ の仮定は、$X^\top X$ の逆行列が存在するための条件です。

仮定 3 の $\sigma^2I_n$ は、「誤差どうしは無相関（非対角が 0）」と「誤差の分散はどの観測でも同じ（対角が同じ $\sigma^2$）」の2つを1つの式で表しています。後者が崩れた状態を**不均一分散**、前者が崩れた状態を**系列相関**といい、第17章で扱います。

## 具体例

**例1：8店舗の売上（本章の共通データ）**

広告費 $x_1$（十万円）、売場面積 $x_2$（10 m²）、月の売上 $y$（百万円）を8店舗で調べました。

| 店舗 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $x_1$ | 1 | 2 | 2 | 3 | 4 | 4 | 5 | 6 |
| $x_2$ | 3 | 2 | 4 | 3 | 5 | 4 | 6 | 5 |
| $y$ | 4 | 6 | 9 | 7 | 11 | 10 | 14 | 13 |

$n=8,\ p=2$ で、$X$ は $8\times3$ 行列です。最小二乗法で当てはめると

$$
\hat y=0.2381+1.0476\,x_1+1.3690\,x_2
$$

になります（$\hat\beta_0=5/21,\ \hat\beta_1=22/21,\ \hat\beta_2=115/84$）。
「売場面積が同じなら、広告費を十万円増やすと売上は平均で約 1.048 百万円増える」と読みます。

**例2：単回帰の傾きとの違い**

同じデータで $y$ を $x_1$ だけに回帰すると、傾きは $1.8742$ になります。重回帰の $1.0476$ よりずっと大きい値です。
広告費の多い店は売場面積も広い（$x_1$ と $x_2$ の相関係数 $0.777$）ので、単回帰の傾きには面積の効果が混ざっています。重回帰の偏回帰係数は、面積の影響を取り除いたあとの広告費の効果です。

## 現場での使われ方

- **マーケティング**：売上を広告費・価格・店舗面積などで説明し、他の条件をそろえたときの広告の効果を見積もります。
- **不動産・行政・調査**：家賃や地価を、面積・駅からの距離・築年数で回帰し、各条件が価格に与える影響を数値化します。
- **製造・品質管理**：製品の強度を温度・圧力・原料配合で回帰し、どの条件を動かせば強度が上がるかを調べます。
- **医療**：血圧を年齢・BMI・塩分摂取量で回帰し、年齢などをそろえたうえでの塩分の影響を評価します。

## グラフ・イメージ

1つ目の図は説明変数が1つの場合の散布図です。$\sigma$ を大きくすると点が直線のまわりに広がり、推定される直線が真の直線からずれやすくなります。
2つ目の図の縦線が残差です。重回帰では直線が「平面（超平面）」になりますが、残差の考え方は同じです。

## 試験のポイント

- 行列表示 $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$ と、$X$ が $n\times(p+1)$ であることを正確に書けるようにします。残差の自由度 $n-p-1$ はここから出てきます。
- 誤差の仮定（平均 0、等分散、無相関、正規性）のうち、どの結果にどの仮定が必要かを区別します。不偏性は平均 0 だけ、ガウス・マルコフの定理は等分散・無相関まで、$t$ 検定・$F$ 検定は正規性まで必要です。
- 偏回帰係数を「他の変数を固定したときの効果」と解釈する問題が典型です。単回帰の係数と符号や大きさが変わる理由を説明できるようにします。
- $\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}$、$V[\hat{\boldsymbol{\beta}}]=\sigma^2(X^\top X)^{-1}$ はすべての計算の出発点です。

## よくある間違い

- × 「線形」とは $x$ について一次式のこと → ○ パラメータ $\boldsymbol{\beta}$ について一次式であること。$y=\beta_0+\beta_1x+\beta_2x^2+\varepsilon$ も線形モデルです。
- × 偏回帰係数は単回帰の傾きと同じ値になる → ○ 説明変数どうしに相関があれば値は変わります。一致するのは説明変数が互いに無相関なときだけです。
- × 計画行列の列数を $p$ とする → ○ 切片の列を含めて $p+1$ 列です。残差の自由度も $n-p$ ではなく $n-p-1$ です。
- × 誤差の正規性は最小二乗推定の前提 → ○ 推定値の計算や不偏性には正規性は不要です。正規性が必要になるのは、$t$ 分布・$F$ 分布を使う検定や区間推定です。
