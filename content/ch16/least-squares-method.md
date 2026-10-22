---
id: least-squares-method
title: 最小二乗法
reading: さいしょうにじょうほう
en: Least squares method
aliases: [OLS, 通常最小二乗法, 最小二乗推定量, LSE]
chapter: 16
importance: A
tags: [回帰分析, 推定, 残差平方和, 行列表示]
related: [multiple-regression-model, normal-equations, hat-matrix, gauss-markov-theorem, maximum-likelihood-estimator, weighted-least-squares]
charts:
  - type: custom
    id: line-data
    params:
      x: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
      series:
        - { label: "残差平方和 S（β0, β2 は最小二乗推定値に固定）", y: [27.0141, 18.1869, 11.5798, 7.1927, 5.0255, 5.0784, 7.3512, 11.8441, 18.5569, 27.4898] }
      xlabel: "β1 の値"
      ylabel: "残差平方和"
    caption: 共通データで β1 だけを動かしたときの残差平方和。下に凸の放物線になり、β1=1.0476 で最小値 4.7738 をとる。最小二乗法はこの谷底を全パラメータについて同時に探す。
  - type: custom
    id: regression-residual
    params: { n: 20, beta0: 1, beta1: 0.5, sigma: 1, outlier: 0 }
    sliders:
      - { param: outlier, label: "右端の点のずれ", min: -6, max: 6, step: 0.5 }
    caption: 縦の線が残差。最小二乗法は縦の線の長さの二乗和を最小にする。右端の点を大きくずらすと、二乗の効果で直線がその点に強く引き寄せられる。
---

## 一言でいうと

データと当てはめた値のずれ（残差）を**二乗して足した量が最小になる**ようにパラメータを決める方法です。
回帰分析で係数を推定する最も基本的な方法で、答えが行列の式で一発で求まるのが特長です。

## 定義・数式

重回帰モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$（$X$ は $n\times(p+1)$、$\operatorname{rank}X=p+1$）で、

$$
S(\boldsymbol{\beta})=\sum_{i=1}^n\bigl(y_i-\beta_0-\beta_1x_{i1}-\cdots-\beta_px_{ip}\bigr)^2=(\boldsymbol{y}-X\boldsymbol{\beta})^\top(\boldsymbol{y}-X\boldsymbol{\beta})
$$

を最小にする $\boldsymbol{\beta}$ を**最小二乗推定量**といいます。答えは

$$
\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}
$$

です。当てはめ値 $\hat{\boldsymbol{y}}=X\hat{\boldsymbol{\beta}}$、残差 $\boldsymbol{e}=\boldsymbol{y}-\hat{\boldsymbol{y}}$、残差平方和 $S_e=\boldsymbol{e}^\top\boldsymbol{e}=S(\hat{\boldsymbol{\beta}})$ と書きます。

単回帰（$p=1$）の場合は

$$
\hat\beta_1=\frac{S_{xy}}{S_{xx}}=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2},\qquad \hat\beta_0=\bar y-\hat\beta_1\bar x
$$

になります。

## 数式の読み解き

**導出（微分してゼロ）**

$S(\boldsymbol{\beta})$ を展開すると、$\boldsymbol{y}^\top X\boldsymbol{\beta}$ はスカラーなので $\boldsymbol{\beta}^\top X^\top\boldsymbol{y}$ と等しく、

$$
S(\boldsymbol{\beta})=\boldsymbol{y}^\top\boldsymbol{y}-2\boldsymbol{\beta}^\top X^\top\boldsymbol{y}+\boldsymbol{\beta}^\top X^\top X\boldsymbol{\beta}
$$

です。ベクトルで微分する公式 $\partial(\boldsymbol{\beta}^\top\boldsymbol{a})/\partial\boldsymbol{\beta}=\boldsymbol{a}$、$\partial(\boldsymbol{\beta}^\top A\boldsymbol{\beta})/\partial\boldsymbol{\beta}=2A\boldsymbol{\beta}$（$A$ は対称）を使うと

$$
\frac{\partial S}{\partial\boldsymbol{\beta}}=-2X^\top\boldsymbol{y}+2X^\top X\boldsymbol{\beta}=\boldsymbol{0}
\;\Longrightarrow\; X^\top X\boldsymbol{\beta}=X^\top\boldsymbol{y}
$$

となります。これが**正規方程式**で、$X^\top X$ が正則なので $\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}$ です。

**本当に最小か**

任意の $\boldsymbol{\beta}$ について、$\boldsymbol{y}-X\boldsymbol{\beta}=\boldsymbol{e}+X(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})$ と分け、正規方程式から $X^\top\boldsymbol{e}=\boldsymbol{0}$ を使うと交差項が消えて

$$
S(\boldsymbol{\beta})=S_e+(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})^\top X^\top X(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})\ \ge\ S_e
$$

です。$X^\top X$ は正定値なので、等号は $\boldsymbol{\beta}=\hat{\boldsymbol{\beta}}$ のときだけです。

**図形としての意味**

$X\boldsymbol{\beta}$ は、$\boldsymbol{\beta}$ をいろいろ変えると $X$ の列が張る空間（列空間）の中を動きます。$S(\boldsymbol{\beta})$ は $\boldsymbol{y}$ とその点の距離の二乗なので、最小二乗法は「列空間の中で $\boldsymbol{y}$ に一番近い点」を探すことにあたります。
一番近い点は $\boldsymbol{y}$ から列空間へ下ろした垂線の足、つまり**正射影**です。残差 $\boldsymbol{e}$ が $X$ のどの列とも直交する（$X^\top\boldsymbol{e}=\boldsymbol{0}$）のはこのためで、正射影を行う行列が**ハット行列** $H=X(X^\top X)^{-1}X^\top$ です。

**計算上の注意**

説明変数どうしの相関が強いと $X^\top X$ の行列式が 0 に近くなり（正則でなくなる寸前）、逆行列の成分が非常に大きくなります。すると、データが少し変わっただけで $\hat{\boldsymbol{\beta}}$ が大きく変わります。これが**多重共線性**の問題です。

**最尤法との関係**

誤差が $N(0,\sigma^2)$ に独立に従うとき、対数尤度は $-\frac n2\log(2\pi\sigma^2)-\frac{S(\boldsymbol{\beta})}{2\sigma^2}$ です。$\boldsymbol{\beta}$ について最大化することは $S(\boldsymbol{\beta})$ の最小化と同じなので、**正規誤差のもとでは最小二乗推定量＝最尤推定量**です。ただし $\sigma^2$ の最尤推定量は $S_e/n$ で、不偏推定量 $S_e/(n-p-1)$ とは異なります。

## 具体例

**例1：共通データの単回帰（広告費だけ）**

8店舗のデータ（$x_1$：1, 2, 2, 3, 4, 4, 5, 6、$y$：4, 6, 9, 7, 11, 10, 14, 13）で、$\bar x=3.375,\ \bar y=9.25$。

$$
S_{xx}=19.875,\quad S_{xy}=37.25,\quad \hat\beta_1=\frac{37.25}{19.875}=1.8742,\quad \hat\beta_0=9.25-1.8742\times3.375=2.9245
$$

**例2：共通データの重回帰（広告費と面積）**

$x_2$（3, 2, 4, 3, 5, 4, 6, 5）も加えると、

$$
X^\top X=\begin{pmatrix}8&27&32\\27&111&120\\32&120&140\end{pmatrix},\quad
X^\top\boldsymbol{y}=\begin{pmatrix}74\\287\\325\end{pmatrix}
$$

で、$\hat{\boldsymbol{\beta}}=(0.2381,\ 1.0476,\ 1.3690)^\top$、残差平方和は $S_e=4.7738$ です。
試しに別の係数 $(0.5,\ 1.2,\ 1.2)$ で計算すると残差平方和は $5.04$ で、最小二乗推定値より大きくなります。上の「本当に最小か」の式どおり、どの係数を選んでも $4.7738$ を下回ることはありません。

## 現場での使われ方

- **製造・品質管理**：検量線（測定器の読みと真の濃度の関係）を最小二乗法で引き、測定値を濃度に換算します。
- **金融・保険**：個別株の収益率を市場全体の収益率に回帰し、傾き（ベータ値）で市場への感応度を測ります。
- **IT・Webサービス**：サーバの応答時間をアクセス数やデータ量で回帰し、負荷が増えたときの応答時間を予測します。

## グラフ・イメージ

1つ目の図は、$\beta_1$ だけを動かしたときの残差平方和です。きれいな放物線になり、谷底が最小二乗推定値 $1.0476$ です。
2つ目の図で右端の点を大きくずらすと、直線がその点に引っ張られます。二乗するため、大きく外れた点ほど影響が強い点に注意します。

## 試験のポイント

- 正規方程式 $X^\top X\boldsymbol{\beta}=X^\top\boldsymbol{y}$ の導出（展開して微分）を書けるようにします。
- 単回帰の $\hat\beta_1=S_{xy}/S_{xx}=r_{xy}\sqrt{S_{yy}/S_{xx}}$ は、計算問題で最もよく使う形です。
- 正規誤差なら最小二乗推定量は最尤推定量と一致します。$\sigma^2$ の推定量は $S_e/n$（最尤）と $S_e/(n-p-1)$（不偏）で違う点が問われます。
- 最小二乗推定量は $\boldsymbol{y}$ の一次式（線形推定量）で、不偏です。等分散・無相関なら線形不偏推定量の中で分散最小です（ガウス・マルコフの定理）。

## よくある間違い

- × 点から直線への垂線の長さを最小にする → ○ 通常の最小二乗法は $y$ 方向（縦）のずれを最小にします。垂線を使うのは主成分分析（直交回帰）です。
- × 残差の和を最小にする → ○ 残差の和は正負で打ち消し合い、切片を含むモデルでは必ず 0 です。最小にするのは二乗和です。
- × $\sigma^2$ の推定に $S_e/n$ を使えば不偏 → ○ 不偏なのは $S_e/(n-p-1)$ です。
- × $x$ を $y$ に回帰した直線と、$y$ を $x$ に回帰した直線は同じ → ○ 縦と横のどちらのずれを最小にするかが違うので、相関が $\pm1$ でない限り別の直線になります。
