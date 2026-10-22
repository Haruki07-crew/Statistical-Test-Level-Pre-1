---
id: hat-matrix
title: ハット行列
reading: はっとぎょうれつ
en: Hat matrix
aliases: [射影行列, 影響行列, H行列]
chapter: 16
importance: A
tags: [回帰分析, 射影, べき等行列, てこ比, 行列表示]
related: [least-squares-method, normal-equations, leverage, residual, cooks-distance, chi-squared-distribution]
charts:
  - type: custom
    id: bar-data
    params:
      labels: ["店1", "店2", "店3", "店4", "店5", "店6", "店7", "店8"]
      values: [0.4484, 0.5079, 0.3651, 0.2579, 0.2262, 0.1746, 0.4762, 0.5437]
      xlabel: "店舗"
      ylabel: "対角成分 h_ii（てこ比）"
    caption: 共通データのハット行列の対角成分。合計はパラメータ数 p+1=3 で、平均は 3/8=0.375。説明変数の値が平均の点 (3.375, 4) から離れた店（1, 2, 7, 8）ほど大きい。
  - type: custom
    id: regression-residual
    params: { n: 20, beta0: 1, beta1: 0.5, sigma: 1, outlier: 4 }
    sliders:
      - { param: outlier, label: "右端の点のずれ", min: -6, max: 6, step: 0.5 }
    caption: 右端の点は x が端にあるため h_ii（てこ比）が大きい。この点の y をずらすと当てはめ値も大きく動く。h_ii は「自分の観測値が自分の当てはめ値に与える重み」である。
---

## 一言でいうと

観測値 $\boldsymbol{y}$ に掛けると当てはめ値 $\hat{\boldsymbol{y}}$ が出てくる行列です。
$\boldsymbol{y}$ に「ハット（^）をかぶせる」のでこう呼ばれ、図形的には説明変数が張る空間への**正射影**を表します。

## 定義・数式

重回帰モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$（$X$ は $n\times(p+1)$、$\operatorname{rank}X=p+1$）で、

$$
H=X(X^\top X)^{-1}X^\top
$$

を**ハット行列**といいます。$\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}$ なので

$$
\hat{\boldsymbol{y}}=X\hat{\boldsymbol{\beta}}=H\boldsymbol{y},\qquad \boldsymbol{e}=\boldsymbol{y}-\hat{\boldsymbol{y}}=(I_n-H)\boldsymbol{y}
$$

です。主な性質は次のとおりです。

| 性質 | 式 |
|---|---|
| 対称 | $H^\top=H$ |
| べき等 | $H^2=H$ |
| $X$ を動かさない | $HX=X,\quad (I-H)X=O$ |
| トレース（＝ランク） | $\operatorname{tr}H=p+1,\quad \operatorname{tr}(I-H)=n-p-1$ |
| 対角成分 | $0\le h_{ii}\le1$（定数項があれば $1/n\le h_{ii}\le1$） |
| 当てはめ値と残差の分散 | $V[\hat{\boldsymbol{y}}]=\sigma^2H,\quad V[\boldsymbol{e}]=\sigma^2(I-H)$ |

対角成分 $h_{ii}$ を観測 $i$ の**てこ比（レバレッジ）**といいます。

## 数式の読み解き

**射影としての見方**

$H$ は $\boldsymbol{y}$ を $X$ の列空間（$X\boldsymbol{\beta}$ の形で書けるベクトル全体）に正射影し、$I-H$ はその直交補空間に正射影します。$\boldsymbol{y}=H\boldsymbol{y}+(I-H)\boldsymbol{y}=\hat{\boldsymbol{y}}+\boldsymbol{e}$ は直交分解で、$H(I-H)=H-H^2=O$ から $\hat{\boldsymbol{y}}^\top\boldsymbol{e}=0$ が出ます。三平方の定理 $\|\boldsymbol{y}\|^2=\|\hat{\boldsymbol{y}}\|^2+\|\boldsymbol{e}\|^2$ が、平方和分解の正体です。
定数項だけのモデル（$X=\boldsymbol{1}$）なら $H=\frac1n\boldsymbol{1}\boldsymbol{1}^\top$ で、すべての成分が $1/n$ です。$H\boldsymbol{y}$ は全成分が $\bar y$ のベクトルになり、「平均で当てはめる」ことも射影の一例だとわかります。

**べき等とトレース**

$$
H^2=X(X^\top X)^{-1}\underbrace{X^\top X(X^\top X)^{-1}}_{I}X^\top=H
$$

「射影したものをもう一度射影しても変わらない」という意味です。トレースは $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ を使って

$$
\operatorname{tr}H=\operatorname{tr}\bigl((X^\top X)^{-1}X^\top X\bigr)=\operatorname{tr}I_{p+1}=p+1
$$

べき等行列の固有値は 0 か 1 なので、トレース＝ランク＝固有値 1 の個数です。

**残差の期待値・分散と $\sigma^2$ の不偏推定**

$(I-H)X=O$ から $\boldsymbol{e}=(I-H)(X\boldsymbol{\beta}+\boldsymbol{\varepsilon})=(I-H)\boldsymbol{\varepsilon}$ です。よって $E[\boldsymbol{e}]=\boldsymbol{0}$、

$$
V[\boldsymbol{e}]=(I-H)\,\sigma^2I\,(I-H)^\top=\sigma^2(I-H)
$$

残差平方和は $S_e=\boldsymbol{e}^\top\boldsymbol{e}=\boldsymbol{\varepsilon}^\top(I-H)\boldsymbol{\varepsilon}$ で、

$$
E[S_e]=\operatorname{tr}\bigl((I-H)E[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top]\bigr)=\sigma^2\operatorname{tr}(I-H)=(n-p-1)\sigma^2
$$

これが $s^2=S_e/(n-p-1)$ が $\sigma^2$ の不偏推定量になる理由です。さらに誤差が正規分布なら、ランク $n-p-1$ のべき等行列による二次形式なので $S_e/\sigma^2\sim\chi^2(n-p-1)$ です。

**てこ比の意味**

$\hat y_i=\sum_jh_{ij}y_j$ なので、$h_{ii}$ は「$y_i$ が1増えたとき $\hat y_i$ がいくつ増えるか」です。$h_{ii}$ が1に近い点は当てはめ値を自分に引き寄せ、残差の分散 $\sigma^2(1-h_{ii})$ は小さくなります。単回帰では

$$
h_{ii}=\frac1n+\frac{(x_i-\bar x)^2}{S_{xx}}
$$

で、$x$ が平均から離れた点ほど大きくなります。

## 具体例

**例1：単回帰の小さな例**

$x=1,2,3,4,5$（$\bar x=3,\ S_{xx}=10,\ n=5$）なら

$$
h_{ii}=\frac15+\frac{(x_i-3)^2}{10}=0.6,\ 0.3,\ 0.2,\ 0.3,\ 0.6
$$

で、合計は $2=p+1$ です。両端の点は中央の点の3倍のてこ比をもちます。

**例2：共通データ（8店舗、$p=2$）**

$(X^\top X)^{-1}$ を使って $h_{ii}=\boldsymbol{x}_i^\top(X^\top X)^{-1}\boldsymbol{x}_i$（$\boldsymbol{x}_i$ は $X$ の第 $i$ 行）を計算すると

$$
h_{ii}=0.4484,\ 0.5079,\ 0.3651,\ 0.2579,\ 0.2262,\ 0.1746,\ 0.4762,\ 0.5437
$$

合計は $3=p+1$、平均は $0.375$ です。たとえば店舗1は $\boldsymbol{x}_1=(1,1,3)^\top$ で、

$$
h_{11}=\frac{1}{756}\bigl(1140+2\cdot60\cdot1-2\cdot312\cdot3+96\cdot1-2\cdot96\cdot3+159\cdot9\bigr)=\frac{339}{756}=0.4484
$$

残差の分散は店舗8で $\sigma^2(1-0.5437)=0.4563\sigma^2$、店舗6で $\sigma^2(1-0.1746)=0.8254\sigma^2$ と、同じ誤差分散でも観測によって違います。
数値計算でも $H^2=H$ が成り立つこと、$H$ の第1行が $(0.4484,\ 0.1825,\ 0.3651,\ 0.0992,\ 0.1071,\ 0.0159,\ 0.0238,\ -0.2421)$ で、これと $\boldsymbol{y}$ の内積が $\hat y_1=5.3929$ になることを確かめられます。

## 現場での使われ方

- **製造・品質管理**：実験条件の端の点（極端な温度など）はてこ比が大きく、1回の測定ミスが結果を大きく左右します。てこ比で要注意の実験点を事前に洗い出します。
- **医療**：臨床データの回帰で、年齢や体重が極端な患者のてこ比を確認し、結果がその少数の患者に依存していないかを調べます。
- **行政・調査**：回帰診断（第17章）で、てこ比 $h_{ii}$ が平均 $(p+1)/n$ の2〜3倍を超える観測を「影響が大きいかもしれない点」として洗い出し、クックの距離などとあわせて確認します。

## グラフ・イメージ

1つ目の図は共通データのてこ比です。説明変数の値の組が「平均的な店」から離れている店ほど棒が高く、棒の合計は必ず $p+1=3$ になります。
2つ目の図で右端の点の $y$ をずらすと、てこ比が大きい点ほど直線が引き寄せられる様子が見えます。

## 試験のポイント

- 対称・べき等・$\operatorname{tr}H=p+1$ の3つを示させる問題が典型です。トレースの計算では $\operatorname{tr}(AB)=\operatorname{tr}(BA)$ を使います。
- $E[S_e]=(n-p-1)\sigma^2$ の導出は、$\boldsymbol{e}=(I-H)\boldsymbol{\varepsilon}$ と $E[\boldsymbol{\varepsilon}^\top A\boldsymbol{\varepsilon}]=\sigma^2\operatorname{tr}A$ の組み合わせです。
- てこ比の平均は $(p+1)/n$。単回帰の $h_{ii}=1/n+(x_i-\bar x)^2/S_{xx}$ はそのまま計算問題に使えます。
- 標準化残差・スチューデント化残差の分母の $\sqrt{1-h_{ii}}$ は、$V[e_i]=\sigma^2(1-h_{ii})$ から来ます。

## よくある間違い

- × 残差は互いに独立で分散 $\sigma^2$ → ○ $V[\boldsymbol{e}]=\sigma^2(I-H)$ なので、残差どうしは相関があり、分散も $\sigma^2(1-h_{ii})$ と観測ごとに違います。
- × てこ比は $y$ の値で決まる → ○ $H$ は $X$ だけで決まります。てこ比は「説明変数の値が外れているか」を表し、$y$ が外れているか（外れ値）とは別の概念です。
- × $\operatorname{tr}H=p$ → ○ 切片の列を含めて $p+1$ です。
- × $H$ は逆行列をもつ → ○ ランクが $p+1<n$ なので $H$ は正則ではありません（単位行列 $I_n$ になるのは $p+1=n$ のときだけで、そのとき残差はすべて 0 です）。
