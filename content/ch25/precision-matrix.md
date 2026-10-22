---
id: precision-matrix
title: 精度行列
reading: せいどぎょうれつ
en: Precision matrix
aliases: [逆共分散行列, 濃度行列, 集中行列]
chapter: 25
importance: A
tags: [グラフィカルモデル, 偏相関, 多変量正規分布, 条件付き独立]
related: [partial-correlation-coefficient, conditional-independence, multivariate-normal-distribution, variance-covariance-matrix, undirected-graphical-model, covariance-selection]
charts:
  - type: custom
    id: graph-network
    params:
      nodes: [X1, X2, X3, X4]
      edges: [[0, 1], [1, 2], [2, 3]]
      edgeLabels: ["0.545", "0.275", "0.756"]
      directed: 0
      layout: line
    caption: 例2の精度行列から描いたグラフ。辺は精度行列の非対角成分が0でない組で、数字は偏相関係数。X1 と X3、X1 と X4、X2 と X4 の間には辺がない。
  - type: custom
    id: bar-data
    params:
      labels: ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]
      values: [0.5447, 0, 0, 0.2745, 0, 0.7559]
      xlabel: 変数の組
      ylabel: 偏相関係数（他の2変数を与えたとき）
    caption: 例2の偏相関係数。通常の相関は 1-3 が 0.30、1-4 が 0.24、2-4 が 0.40 とすべて正だが、他の変数を固定すると 0 になる。
---

## 一言でいうと

分散共分散行列の逆行列です。
その非対角成分は「他のすべての変数を固定したときの、2変数の直接のつながり」を表し、多変量正規分布では成分が0であることが条件付き独立を意味します。

## 定義・数式

$p$ 次元の確率ベクトル $\boldsymbol{X}$ の分散共分散行列 $\Sigma$ が正則のとき、

$$
\Omega=(\omega_{ij})=\Sigma^{-1}
$$

を**精度行列**（濃度行列）といいます。次の3つが成り立ちます（rest は $i,j$ 以外のすべての変数）。

**(1) 偏相関係数**

$$
\rho_{ij\cdot\text{rest}}=-\frac{\omega_{ij}}{\sqrt{\omega_{ii}\,\omega_{jj}}}
$$

**(2) 回帰との関係**：$X_i$ を他のすべての変数に回帰すると

$$
E[X_i\mid X_{-i}]=\mu_i-\sum_{j\ne i}\frac{\omega_{ij}}{\omega_{ii}}(X_j-\mu_j),\qquad
V[X_i\mid X_{-i}]=\frac{1}{\omega_{ii}}
$$

（第2式は多変量正規のとき。一般には最良線形予測の残差分散）。

**(3) 条件付き独立**：$\boldsymbol{X}\sim N_p(\boldsymbol{\mu},\Sigma)$ なら

$$
\omega_{ij}=0\iff\rho_{ij\cdot\text{rest}}=0\iff X_i\perp X_j\mid X_{\text{rest}}
$$

## 数式の読み解き

平均を0とし、変数を $\boldsymbol{x}=(\boldsymbol{x}_a,\boldsymbol{x}_b)$ に分けて、$\Omega$ も同じように $\Omega_{aa},\Omega_{ab},\Omega_{bb}$ に区切ります。

**ステップ1：条件付き分布**。多変量正規の密度は $\exp(-\frac12\boldsymbol{x}^\top\Omega\boldsymbol{x})$ に比例します。$\boldsymbol{x}_b$ を固定して $\boldsymbol{x}_a$ の関数とみると

$$
-\frac12\boldsymbol{x}^\top\Omega\boldsymbol{x}=-\frac12\Big(\boldsymbol{x}_a^\top\Omega_{aa}\boldsymbol{x}_a+2\boldsymbol{x}_a^\top\Omega_{ab}\boldsymbol{x}_b\Big)+(\boldsymbol{x}_a\ \text{を含まない項})
$$

です。平方完成すると、$\boldsymbol{x}_a\mid\boldsymbol{x}_b\sim N\big(-\Omega_{aa}^{-1}\Omega_{ab}\boldsymbol{x}_b,\ \Omega_{aa}^{-1}\big)$ がわかります。
$a=\{i\}$（1変数）とすれば、(2) の $E[X_i\mid X_{-i}]=-\sum_{j\ne i}(\omega_{ij}/\omega_{ii})x_j$、$V=1/\omega_{ii}$ が出ます。

**ステップ2：偏相関**。$a=\{i,j\}$ とすると、残りを与えたときの $(X_i,X_j)$ の条件付き分散共分散行列は

$$
\Omega_{aa}^{-1}=\begin{pmatrix}\omega_{ii}&\omega_{ij}\\\omega_{ij}&\omega_{jj}\end{pmatrix}^{-1}
=\frac{1}{\omega_{ii}\omega_{jj}-\omega_{ij}^2}\begin{pmatrix}\omega_{jj}&-\omega_{ij}\\-\omega_{ij}&\omega_{ii}\end{pmatrix}
$$

です。この相関（＝偏相関係数）は、共通の分母が約分されて

$$
\rho_{ij\cdot\text{rest}}=\frac{-\omega_{ij}}{\sqrt{\omega_{jj}\,\omega_{ii}}}
$$

になります。マイナスが付くのは、2×2 の逆行列で非対角の符号が反転するためです。

**ステップ3：条件付き独立**。正規分布では「無相関 ⇔ 独立」です。ステップ1から条件付き分布も正規なので、条件付き相関（偏相関）が0なら条件付き独立です。
別の見方として、$\omega_{ij}=0$ なら密度の指数部に $x_ix_j$ の項がないので、密度は「$x_i$ と rest の関数」×「$x_j$ と rest の関数」に分解でき、これは条件付き独立の定義そのものです。

直感的には、**共分散行列は「間接的な経路も含めた全体の関係」、精度行列は「直接の関係だけ」** を表します。

## 具体例

**例1：3変数の連鎖**

$$
\Omega=\begin{pmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{pmatrix},\qquad
\Sigma=\Omega^{-1}=\frac14\begin{pmatrix}3&2&1\\2&4&2\\1&2&3\end{pmatrix}
$$

とします。$X_1$ と $X_3$ の通常の相関は $\dfrac{1/4}{\sqrt{3/4\cdot3/4}}=\dfrac13=0.333$ ですが、$\omega_{13}=0$ なので偏相関は0です。
$X_1$ と $X_2$ の偏相関は $-(-1)/\sqrt{2\times2}=0.5$ です。

回帰で確かめます。$E[X_1\mid X_2,X_3]=-(\omega_{12}/\omega_{11})x_2-(\omega_{13}/\omega_{11})x_3=0.5\,x_2+0\cdot x_3$、残差分散は $1/\omega_{11}=0.5$ です。
numpy で $X_1$ を $(X_2,X_3)$ に回帰した係数を $\Sigma$ から直接計算しても $(0.5,\ 0)$、残差分散 0.5 で一致しました。$X_2$ がわかれば $X_3$ は $X_1$ の予測に役立ちません。

**例2：4変数の相関行列から偏相関を求める**

相関行列

$$
R=\begin{pmatrix}1&0.6&0.3&0.24\\0.6&1&0.5&0.4\\0.3&0.5&1&0.8\\0.24&0.4&0.8&1\end{pmatrix}
$$

の逆行列を numpy で計算すると

$$
R^{-1}=\begin{pmatrix}1.5625&-0.9375&0&0\\-0.9375&1.8958&-0.6667&0\\0&-0.6667&3.1111&-2.2222\\0&0&-2.2222&2.7778\end{pmatrix}
$$

です。偏相関は

$$
\rho_{12\cdot34}=\frac{0.9375}{\sqrt{1.5625\times1.8958}}=0.545,\quad
\rho_{23\cdot14}=\frac{0.6667}{\sqrt{1.8958\times3.1111}}=0.275,\quad
\rho_{34\cdot12}=\frac{2.2222}{\sqrt{3.1111\times2.7778}}=0.756
$$

で、残りの3組（1-3, 1-4, 2-4）は0です。通常の相関はすべて正なのに、直接のつながりは隣どうしだけ、という連鎖構造が精度行列から読み取れます。

## 現場での使われ方

- **金融・保険**：多数の銘柄のリターンについて精度行列を推定し、「市場全体の影響を除いても直接連動している銘柄の組」を探します。精度行列は最小分散ポートフォリオの重みの計算にも現れます。
- **医療**：遺伝子の発現量の精度行列（遺伝子ネットワーク）を推定し、直接制御し合っている候補の遺伝子の組を絞り込みます。
- **製造・品質管理**：多数のセンサー値の精度行列から、異常が伝わる直接の経路を推定し、異常の原因箇所を特定する手がかりにします。

## グラフ・イメージ

上の図は、例2の精度行列で0でない成分の組を辺で結んだグラフです。X1–X2–X3–X4 の一本道になっています。
下の棒グラフは偏相関で、通常の相関が正の組でも、辺のない組では偏相関がちょうど0になっています。

## 試験のポイント

- $\rho_{ij\cdot\text{rest}}=-\omega_{ij}/\sqrt{\omega_{ii}\omega_{jj}}$ を、符号のマイナスを含めて正確に使えるようにします。
- 3変数で、相関行列の逆行列から偏相関を出し、公式 $(r_{12}-r_{13}r_{23})/\sqrt{(1-r_{13}^2)(1-r_{23}^2)}$ の値と一致することを確かめる問題が典型です。
- 多変量正規分布では「$\omega_{ij}=0$ ⇔ 偏相関0 ⇔ 条件付き独立」。これがガウス型グラフィカルモデルの土台です。
- $1/\omega_{ii}$ は $X_i$ を他の全変数に回帰したときの残差分散です。相関行列なら重相関係数の2乗は $1-1/r^{ii}$ です。

## よくある間違い

- × 偏相関は $\omega_{ij}/\sqrt{\omega_{ii}\omega_{jj}}$ → ○ マイナスが付きます。$\omega_{ij}<0$ のとき偏相関は正です。
- × 精度行列の成分が0なら、2変数は（無条件に）独立 → ○ 意味するのは「他のすべての変数を与えたときの条件付き独立」です。例2の $X_1$ と $X_4$ は相関 0.24 です。
- × 正規分布でなくても $\omega_{ij}=0$ なら条件付き独立 → ○ 一般には「偏相関0（線形の意味での条件付き無相関）」までしか言えません。
