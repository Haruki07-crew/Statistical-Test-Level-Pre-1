---
id: normal-equations
title: 正規方程式
reading: せいきほうていしき
en: Normal equations
aliases: [正規方程式系]
chapter: 16
importance: A
tags: [回帰分析, 最小二乗法, 連立方程式, 行列表示]
related: [least-squares-method, multiple-regression-model, hat-matrix, variance-covariance-matrix, multicollinearity]
charts:
  - type: custom
    id: bar-data
    params:
      labels: ["店1", "店2", "店3", "店4", "店5", "店6", "店7", "店8"]
      values: [-1.3929, 0.9286, 1.1905, -0.4881, -0.2738, 0.0952, 0.3095, -0.3690]
      xlabel: "店舗"
      ylabel: "残差 e_i"
    caption: 共通データの重回帰の残差。正規方程式の1行目により、正の棒と負の棒の合計はちょうど 0 になる。さらに x1・x2 を重みにした和もそれぞれ 0 になる。
---

## 一言でいうと

最小二乗推定量を求めるために解く連立一次方程式です。
「残差が、どの説明変数とも直交する（内積が 0）」という条件を式にしたもので、名前の「正規（normal）」は「直交」の意味です。

## 定義・数式

重回帰モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$ の残差平方和 $S(\boldsymbol{\beta})=(\boldsymbol{y}-X\boldsymbol{\beta})^\top(\boldsymbol{y}-X\boldsymbol{\beta})$ を最小にする条件は

$$
X^\top X\,\hat{\boldsymbol{\beta}}=X^\top\boldsymbol{y}
$$

です。これを**正規方程式**といいます。$\operatorname{rank}X=p+1$ なら $X^\top X$ は正則で、

$$
\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}
$$

と一意に解けます。成分で書くと、$j=0,1,\dots,p$（$x_{i0}=1$）について

$$
\sum_{i=1}^n x_{ij}\bigl(y_i-\hat\beta_0-\hat\beta_1x_{i1}-\cdots-\hat\beta_px_{ip}\bigr)=0
$$

の $p+1$ 本の式です。

**平均を引いた形**：1本目（$j=0$）から $\hat\beta_0=\bar y-\hat\beta_1\bar x_1-\cdots-\hat\beta_p\bar x_p$ が得られ、これを残りに代入すると

$$
\begin{pmatrix}S_{11}&\cdots&S_{1p}\\ \vdots&\ddots&\vdots\\ S_{p1}&\cdots&S_{pp}\end{pmatrix}
\begin{pmatrix}\hat\beta_1\\ \vdots\\ \hat\beta_p\end{pmatrix}
=\begin{pmatrix}S_{1y}\\ \vdots\\ S_{py}\end{pmatrix},\qquad
S_{jk}=\sum_i(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)
$$

になります。左の行列は偏差平方和・積和行列で、$n-1$ で割れば標本分散共分散行列です。

## 数式の読み解き

- $X^\top X$：$(j,k)$ 成分が $\sum_ix_{ij}x_{ik}$ の $(p+1)\times(p+1)$ 対称行列。説明変数どうしの「積の和」を並べたものです。
- $X^\top\boldsymbol{y}$：$j$ 成分が $\sum_ix_{ij}y_i$ のベクトル。説明変数と目的変数の積の和です。
- $X^\top(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})=X^\top\boldsymbol{e}=\boldsymbol{0}$ と書き直すと、「残差ベクトル $\boldsymbol{e}$ は $X$ のどの列とも直交する」という意味がはっきりします。

導出は $S(\boldsymbol{\beta})=\boldsymbol{y}^\top\boldsymbol{y}-2\boldsymbol{\beta}^\top X^\top\boldsymbol{y}+\boldsymbol{\beta}^\top X^\top X\boldsymbol{\beta}$ を $\boldsymbol{\beta}$ で微分して 0 とおくだけです（「最小二乗法」参照）。

正規方程式から、すぐに次の性質が出ます（切片を含むモデルの場合）。

1. $j=0$ の式から $\sum_ie_i=0$：残差の合計は 0。
2. $j\ge1$ の式から $\sum_ix_{ij}e_i=0$：残差と各説明変数は直交（標本共分散も 0）。
3. 1 を $n$ で割ると $\bar y=\hat\beta_0+\hat\beta_1\bar x_1+\cdots+\hat\beta_p\bar x_p$：当てはめた平面は平均の点 $(\bar x_1,\dots,\bar x_p,\bar y)$ を通る。
4. $\hat{\boldsymbol{y}}=X\hat{\boldsymbol{\beta}}$ なので $\hat{\boldsymbol{y}}^\top\boldsymbol{e}=\hat{\boldsymbol{\beta}}^\top X^\top\boldsymbol{e}=0$：当てはめ値と残差も直交。これが平方和分解 $S_T=S_R+S_e$ の根拠です。

$X^\top X$ が正則でない（ある説明変数が他の説明変数の一次結合で書ける）と、正規方程式の解は無数にあり、係数が一意に決まりません。これが完全な多重共線性です。

## 具体例

**例1：共通データ（8店舗）を偏差の形で解く**

広告費 $x_1$：1, 2, 2, 3, 4, 4, 5, 6、面積 $x_2$：3, 2, 4, 3, 5, 4, 6, 5、売上 $y$：4, 6, 9, 7, 11, 10, 14, 13。
平均は $\bar x_1=3.375,\ \bar x_2=4,\ \bar y=9.25$ で、

$$
\begin{pmatrix}19.875&12\\12&12\end{pmatrix}\begin{pmatrix}\hat\beta_1\\ \hat\beta_2\end{pmatrix}=\begin{pmatrix}37.25\\29\end{pmatrix}
$$

行列式は $19.875\times12-12^2=94.5$ なので、クラメルの公式で

$$
\hat\beta_1=\frac{12\times37.25-12\times29}{94.5}=\frac{99}{94.5}=1.0476,\qquad
\hat\beta_2=\frac{19.875\times29-12\times37.25}{94.5}=\frac{129.375}{94.5}=1.3690
$$

$$
\hat\beta_0=9.25-1.0476\times3.375-1.3690\times4=0.2381
$$

**例2：元の形で確かめる**

$X^\top X=\begin{pmatrix}8&27&32\\27&111&120\\32&120&140\end{pmatrix}$、$X^\top\boldsymbol{y}=(74,\,287,\,325)^\top$ で、$\det(X^\top X)=756$ です。余因子から

$$
(X^\top X)^{-1}=\frac{1}{756}\begin{pmatrix}1140&60&-312\\60&96&-96\\-312&-96&159\end{pmatrix}
$$

これに $X^\top\boldsymbol{y}$ を掛けると $\hat{\boldsymbol{\beta}}=\frac{1}{756}(180,\,792,\,1035)^\top=(0.2381,\,1.0476,\,1.3690)^\top$ で、例1と一致します。
残差は $(-1.3929,\ 0.9286,\ 1.1905,\ -0.4881,\ -0.2738,\ 0.0952,\ 0.3095,\ -0.3690)$ で、合計も、$x_1$・$x_2$ との積和も（丸め誤差を除いて）0 です。

**例3：直交する計画なら式がばらばらに解ける**

温度と圧力をそれぞれ「低 $=-1$」「高 $=+1$」の2水準にして4回実験し、収率 $y$ を測りました。

| 回 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 温度 $x_1$ | −1 | +1 | −1 | +1 |
| 圧力 $x_2$ | −1 | −1 | +1 | +1 |
| 収率 $y$ | 10 | 14 | 12 | 18 |

列どうしの内積はすべて 0 なので $X^\top X=\operatorname{diag}(4,4,4)$、$X^\top\boldsymbol{y}=(54,\ 10,\ 6)^\top$ です。正規方程式は3本の独立な式 $4\hat\beta_0=54,\ 4\hat\beta_1=10,\ 4\hat\beta_2=6$ に分かれ、

$$
\hat\beta_0=13.5,\qquad \hat\beta_1=2.5,\qquad \hat\beta_2=1.5
$$

です。温度の係数は、圧力を入れても外しても同じ $2.5$ になります。説明変数が直交していると、ある変数の係数が他の変数の有無に左右されません。

## 現場での使われ方

- **IT・Webサービス**：統計ソフトや表計算ソフトの回帰機能は、正規方程式（を数値的に安定な方法で解き直したもの）で係数を計算しています。
- **製造・品質管理**：実験計画で説明変数を直交するように配置すると $X^\top X$ が対角行列になり、正規方程式が変数ごとに独立に解けます。係数どうしが干渉しない実験になります。
- **行政・調査**：説明変数を追加したときに係数が変わる理由を、偏差平方和・積和行列の非対角成分（説明変数間の共分散）から説明します。

## グラフ・イメージ

共通データの残差を棒で並べた図です。正規方程式の1本目は「残差の合計が 0」を意味するので、上向きの棒と下向きの棒の長さの合計がつり合っています。

## 試験のポイント

- $X^\top X$ と $X^\top\boldsymbol{y}$ を、与えられた和（$\sum x_{ij},\ \sum x_{ij}x_{ik},\ \sum x_{ij}y_i$）から組み立てる問題が典型です。
- 説明変数が2つなら、偏差の形の $2\times2$ の連立方程式に直して、クラメルの公式や逆行列 $\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$ で解くと速く計算できます。
- 残差の和が 0、残差と説明変数・当てはめ値が直交、回帰平面が平均の点を通る、の3つは正規方程式から直接出ます。
- 説明変数を相関係数行列 $R$ で標準化すると、正規方程式は $R\hat{\boldsymbol{\beta}}^{*}=\boldsymbol{r}_{y}$ になり、標準化偏回帰係数が得られます。

## よくある間違い

- × 「正規方程式」は正規分布の仮定から出る式 → ○ 正規分布とは無関係で、「直交（normal）条件」の意味です。誤差の分布を仮定しなくても成り立ちます。
- × 切片のないモデルでも残差の和は 0 → ○ 残差の和が 0 になるのは、$X$ に定数項の列があるとき（$j=0$ の式があるとき）だけです。
- × 偏差の形で解いたあと $\hat\beta_0$ を忘れる → ○ $\hat\beta_0=\bar y-\sum_j\hat\beta_j\bar x_j$ で最後に戻します。
