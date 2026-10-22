---
id: weighted-least-squares
title: 加重最小二乗法
reading: かじゅうさいしょうにじょうほう
en: Weighted least squares
aliases: [WLS, 重み付き最小二乗法, 加重最小二乗推定量]
chapter: 19
importance: A
tags: [回帰分析, 不均一分散, 重み, 推定]
related: [least-squares-method, heteroscedasticity, generalized-least-squares, gauss-markov-theorem, iteratively-reweighted-least-squares, robust-standard-error]
charts:
  - type: custom
    id: residual-pattern
    params: { pattern: funnel, strength: 1, n: 80 }
    sliders:
      - { param: strength, label: "分散の増え方", min: 0, max: 3, step: 0.1 }
    caption: 当てはめ値に対する残差。右に開くらっぱ形のとき、右側の観測ほど誤差の分散が大きく、WLS では軽い重みを付ける。
  - type: custom
    id: line-data
    params:
      x: [1, 2, 3, 4, 5, 6]
      series:
        - { label: "観測値 y", y: [3.1, 4.8, 7.6, 8.2, 12.9, 12.4] }
        - { label: "OLS の直線", y: [3.0667, 5.1067, 7.1467, 9.1867, 11.2267, 13.2667] }
        - { label: "WLS の直線（重み 1/x）", y: [3.0421, 5.0919, 7.1417, 9.1916, 11.2414, 13.2913] }
      xlabel: "売場面積 x（100 m²）"
      ylabel: "売上 y"
    caption: 例1のデータと2本の回帰直線。WLS はばらつきの小さい左側の点を重く見るため、左側の点により近く引かれる。
---

## 一言でいうと

観測ごとに誤差の大きさ（分散）が違うとき、精度の高い観測を重く、精度の低い観測を軽く扱って当てはめる最小二乗法です。
重みは「誤差分散の逆数」に比例させます。

## 定義・数式

重回帰モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$（$X$ は $n\times(p+1)$、列フルランク）で、誤差が互いに無相関、平均0、

$$
V[\varepsilon_i]=\frac{\sigma^2}{w_i}\quad(w_i>0\ \text{は既知}),\qquad V[\boldsymbol{\varepsilon}]=\sigma^2W^{-1},\quad W=\mathrm{diag}(w_1,\dots,w_n)
$$

とします。**加重最小二乗推定量**は、重み付きの残差平方和

$$
S_W(\boldsymbol{\beta})=\sum_{i=1}^{n} w_i\,(y_i-\boldsymbol{x}_i^\top\boldsymbol{\beta})^2=(\boldsymbol{y}-X\boldsymbol{\beta})^\top W(\boldsymbol{y}-X\boldsymbol{\beta})
$$

を最小にする $\boldsymbol{\beta}$ で、

$$
\hat{\boldsymbol{\beta}}_W=(X^\top WX)^{-1}X^\top W\boldsymbol{y},\qquad
E[\hat{\boldsymbol{\beta}}_W]=\boldsymbol{\beta},\qquad
V[\hat{\boldsymbol{\beta}}_W]=\sigma^2(X^\top WX)^{-1}
$$

です。$\sigma^2$ の不偏推定量は $s_W^2=S_W(\hat{\boldsymbol{\beta}}_W)/(n-p-1)$ です。

## 数式の読み解き

- $w_i$：観測 $i$ の重み。分散が大きい観測ほど小さくします。$w_i$ は比だけが意味をもち、全体を定数倍しても推定値は変わりません。
- $\sigma^2$：共通の尺度。$w_i=1$ なら通常の最小二乗法（OLS）に戻ります。

**導出1：微分して正規方程式を作る**

$$
\frac{\partial S_W}{\partial\boldsymbol{\beta}}=-2X^\top W(\boldsymbol{y}-X\boldsymbol{\beta})=\boldsymbol{0}
\;\Longrightarrow\;
X^\top WX\,\boldsymbol{\beta}=X^\top W\boldsymbol{y}
$$

$X^\top WX$ は正定値なので逆行列をかけて $\hat{\boldsymbol{\beta}}_W$ が得られます。

**導出2：変数変換で OLS に帰着させる**

$W^{1/2}=\mathrm{diag}(\sqrt{w_i})$ を両辺に左からかけると

$$
\underbrace{W^{1/2}\boldsymbol{y}}_{\boldsymbol{y}^*}=\underbrace{W^{1/2}X}_{X^*}\boldsymbol{\beta}+\underbrace{W^{1/2}\boldsymbol{\varepsilon}}_{\boldsymbol{\varepsilon}^*},\qquad
V[\boldsymbol{\varepsilon}^*]=W^{1/2}\,\sigma^2W^{-1}\,W^{1/2}=\sigma^2I
$$

となり、変換後は等分散です。これに OLS を使うと $(X^{*\top}X^*)^{-1}X^{*\top}\boldsymbol{y}^*=(X^\top WX)^{-1}X^\top W\boldsymbol{y}$ です。
変換後のモデルはガウス・マルコフの定理の仮定を満たすので、**WLS は線形不偏推定量の中で分散が最小（BLUE）**になります。

**期待値と分散**：$\hat{\boldsymbol{\beta}}_W=\boldsymbol{\beta}+(X^\top WX)^{-1}X^\top W\boldsymbol{\varepsilon}$ より $E[\hat{\boldsymbol{\beta}}_W]=\boldsymbol{\beta}$、

$$
V[\hat{\boldsymbol{\beta}}_W]=(X^\top WX)^{-1}X^\top W(\sigma^2W^{-1})WX(X^\top WX)^{-1}=\sigma^2(X^\top WX)^{-1}
$$

単回帰では、重み付き平均 $\bar{x}_w=\sum w_ix_i/\sum w_i$、$\bar{y}_w$ を使って $\hat\beta_1=\sum w_i(x_i-\bar{x}_w)(y_i-\bar{y}_w)/\sum w_i(x_i-\bar{x}_w)^2$、$\hat\beta_0=\bar{y}_w-\hat\beta_1\bar{x}_w$ と書けます。

## 具体例

**例1：売場面積と売上（分散が面積に比例）**

6店の売場面積 $x=1,\dots,6$（100 m²）と売上 $y=(3.1,\ 4.8,\ 7.6,\ 8.2,\ 12.9,\ 12.4)$ で、$V[\varepsilon_i]=\sigma^2x_i$ と考え、重み $w_i=1/x_i$ を使います。

$$
X^\top WX=\begin{pmatrix}\sum w_i & \sum w_ix_i\\ \sum w_ix_i & \sum w_ix_i^2\end{pmatrix}=\begin{pmatrix}2.45 & 6\\ 6 & 21\end{pmatrix},\qquad
X^\top W\boldsymbol{y}=\begin{pmatrix}\sum y_i/x_i\\ \sum y_i\end{pmatrix}=\begin{pmatrix}14.73\\ 49\end{pmatrix}
$$

行列式は $2.45\times21-6^2=15.45$ なので

$$
\hat\beta_0=\frac{21\times14.73-6\times49}{15.45}=\frac{15.33}{15.45}=0.9922,\qquad
\hat\beta_1=\frac{2.45\times49-6\times14.73}{15.45}=\frac{31.67}{15.45}=2.0498
$$

OLS では $\hat\beta_0=1.0267,\ \hat\beta_1=2.04$ です。
傾きの分散を比べると、OLS は $0.2000\,\sigma^2$（不均一分散のもとでのサンドイッチ型の分散）、WLS は $\sigma^2(X^\top WX)^{-1}$ の (2,2) 成分で $2.45/15.45\,\sigma^2=0.1586\,\sigma^2$ です。WLS のほうが約21%小さくなります。

**例2：グループ平均を回帰する（重み＝人数）**

$x=1,2,3$ の3群で、人数 $n_i=5,10,20$、平均 $\bar{y}_i=2.0,\ 3.4,\ 4.1$ だけが分かっているとします。$V[\bar{y}_i]=\sigma^2/n_i$ なので重みは $w_i=n_i$ です。

$$
X^\top WX=\begin{pmatrix}35 & 85\\ 85 & 225\end{pmatrix},\quad X^\top W\boldsymbol{y}=\begin{pmatrix}126\\ 324\end{pmatrix}
\;\Longrightarrow\;
\hat\beta_0=1.2462,\ \ \hat\beta_1=0.9692
$$

これは、個票データ35人分に OLS を当てはめた係数と一致します。重みを付けない平均の回帰（$\hat\beta_1=1.05$）は、5人の群と20人の群を同じ重さで扱ってしまいます。

## 現場での使われ方

- **製造・品質管理**：濃度が高いほど測定誤差が大きい分析機器の検量線で、$1/x$ や $1/x^2$ の重みを付けて低濃度側の精度を保ちます。
- **行政・調査**：市区町村別の平均値や率を回帰するとき、人口（標本サイズ）を重みにします。
- **マーケティング**：店舗規模とともに売上のばらつきが大きくなるとき、規模の逆数を重みにして係数を推定します。
- **医療**：メタアナリシスの固定効果モデルでは、各研究の効果量を分散の逆数で重み付けして統合します。これも WLS の一種です。

## グラフ・イメージ

上の図で「分散の増え方」を大きくすると、残差がらっぱ形に広がります。この形が WLS を検討する合図です。
下の図は例1で、2本の直線はほぼ同じですが、WLS の直線はばらつきの小さい左側の点に寄っています。

## 試験のポイント

- 重みは**分散の逆数**です。分散が $x$ に比例なら $w=1/x$、$x^2$ に比例なら $w=1/x^2$、平均値なら $w=n_i$ です。
- 解 $(X^\top WX)^{-1}X^\top W\boldsymbol{y}$ と分散 $\sigma^2(X^\top WX)^{-1}$ を、正規方程式または $W^{1/2}$ による変換から導けるようにします。
- 両辺を $\sigma_i$（$\propto 1/\sqrt{w_i}$）で割ると等分散のモデルになる、という変換の考え方がよく問われます。
- 一般化最小二乗法（GLS）で $\Omega$ が対角行列の特別な場合が WLS です。
- 一般化線形モデルの最尤推定は、重みを更新しながら WLS をくり返す反復重み付き最小二乗法で計算されます。

## よくある間違い

- × 重みを分散そのものにする → ○ 重みは分散の**逆数**です。ばらつきの大きい観測ほど軽くします。
- × 重みは $\sigma_i$ の逆数 $1/\sigma_i$ → ○ 残差平方和に掛ける重みは $1/\sigma_i^2$ です。$1/\sigma_i$ は「両辺を割る」ときの係数です。
- × WLS の推定値は OLS より必ず「当てはまりがよい」 → ○ 通常の残差平方和は OLS が最小です。WLS は推定量の分散を小さくするための方法です。
- × 重みが未知でも勝手に決めてよい → ○ 重みが誤っていても不偏性は保たれますが、最良性は失われ、$\sigma^2(X^\top WX)^{-1}$ も正しくなくなります。重みを推定する場合（実行可能な WLS）は残差から分散の形を推定します。
