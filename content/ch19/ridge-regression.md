---
id: ridge-regression
title: リッジ回帰
reading: りっじかいき
en: Ridge regression
aliases: [リッジ推定量, L2正則化, 尾根回帰, Tikhonov正則化]
chapter: 19
importance: A
tags: [正則化, 罰則付き推定, 多重共線性, 縮小推定]
related: [regularization, lasso, bias-in-penalized-estimation, multicollinearity, least-squares-method, cross-validation]
charts:
  - type: custom
    id: ridge-lasso-path
    params: { method: ridge }
    caption: 4変数のデータで λ を大きくしたときのリッジ推定値の推移（横軸は log10 λ）。どの係数もなめらかに0へ近づくが、ちょうど0にはならない。
  - type: custom
    id: line-data
    params:
      x: [0, 1, 2, 3, 4, 5, 6, 8, 10]
      series:
        - { label: "バイアスの2乗和", y: [0, 0.0091, 0.0236, 0.0414, 0.0618, 0.0842, 0.108, 0.158, 0.2092] }
        - { label: "分散の和", y: [1.0526, 0.2975, 0.1542, 0.1018, 0.0759, 0.0608, 0.0508, 0.0384, 0.0309] }
        - { label: "平均二乗誤差 MSE", y: [1.0526, 0.3066, 0.1778, 0.1431, 0.1377, 0.145, 0.1588, 0.1964, 0.24] }
      xlabel: "λ"
      ylabel: "σ² を単位とした値"
    caption: 例2（相関0.9の2変数、β=(1, 0.8)、σ²=1）でのリッジ推定量の MSE の分解。λ を大きくするとバイアスは増え分散は減り、λ≈3.8 付近で MSE が最小になる。
---

## 一言でいうと

最小二乗法の残差平方和に「係数の二乗和」の罰則を足して、係数を0の方向へ少し縮める回帰です。
少しの偏りを受け入れる代わりに、多重共線性で暴れる推定値のばらつきを大きく減らします。

## 定義・数式

説明変数を標準化し、目的変数を中心化した線形モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$（$E[\boldsymbol{\varepsilon}]=\boldsymbol{0}$、$V[\boldsymbol{\varepsilon}]=\sigma^2I$、切片は罰則に含めない）で、**リッジ推定量**は

$$
\hat{\boldsymbol{\beta}}_R(\lambda)=\arg\min_{\boldsymbol{\beta}}\left\{\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2+\lambda\|\boldsymbol{\beta}\|^2\right\}
=(X^\top X+\lambda I)^{-1}X^\top\boldsymbol{y}\qquad(\lambda>0)
$$

です。$\lambda$ を**正則化パラメータ**（罰則の強さ）といいます。
$\lambda>0$ なら $X^\top X+\lambda I$ は常に正定値なので、$X^\top X$ が特異（完全な多重共線性や $p>n$）でも解が一意に決まります。

期待値と分散は

$$
E[\hat{\boldsymbol{\beta}}_R]=(X^\top X+\lambda I)^{-1}X^\top X\,\boldsymbol{\beta},\qquad
V[\hat{\boldsymbol{\beta}}_R]=\sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}
$$

で、バイアスは $-\lambda(X^\top X+\lambda I)^{-1}\boldsymbol{\beta}$ です。
同じ推定量は、制約付き問題「$\|\boldsymbol{\beta}\|^2\le t$ のもとで残差平方和を最小化」の解としても表せます（$t$ と $\lambda$ が1対1に対応）。

## 数式の読み解き

**閉じた形の導出**：目的関数を展開して $\boldsymbol{\beta}$ で微分します。

$$
\frac{\partial}{\partial\boldsymbol{\beta}}\left\{(\boldsymbol{y}-X\boldsymbol{\beta})^\top(\boldsymbol{y}-X\boldsymbol{\beta})+\lambda\boldsymbol{\beta}^\top\boldsymbol{\beta}\right\}
=-2X^\top\boldsymbol{y}+2X^\top X\boldsymbol{\beta}+2\lambda\boldsymbol{\beta}=\boldsymbol{0}
$$

$$
(X^\top X+\lambda I)\boldsymbol{\beta}=X^\top\boldsymbol{y}
$$

正規方程式の左辺の行列の対角に $\lambda$ を足しただけです。この「対角に尾根（ridge）を足す」形が名前の由来です。

**バイアスの導出**：$E[\boldsymbol{y}]=X\boldsymbol{\beta}$ を代入し、$X^\top X=(X^\top X+\lambda I)-\lambda I$ と書き換えると

$$
E[\hat{\boldsymbol{\beta}}_R]=(X^\top X+\lambda I)^{-1}\{(X^\top X+\lambda I)-\lambda I\}\boldsymbol{\beta}=\boldsymbol{\beta}-\lambda(X^\top X+\lambda I)^{-1}\boldsymbol{\beta}
$$

**固有値で見る**：$X^\top X=\sum_j d_j\boldsymbol{u}_j\boldsymbol{u}_j^\top$（$d_j$ は固有値）とすると、OLS 推定量を固有ベクトル方向に分解した各成分が $d_j/(d_j+\lambda)$ 倍に縮みます。
多重共線性で小さい固有値 $d_j$ の方向ほど強く縮むので、OLS で $\sigma^2/d_j$ と大きかった分散が抑えられます。

**直交計画の場合**：$X^\top X=I$ なら $\hat{\boldsymbol{\beta}}_R=\hat{\boldsymbol{\beta}}_{OLS}/(1+\lambda)$ で、すべての係数が同じ割合で縮みます。ちょうど0にはなりません。

## 具体例

**例1：相関の強い2変数（$n=5$）**

中心化したデータ $x_1=(-2,-1,0,1,2)$、$x_2=(-2,0,-1,1,2)$、$y=(-3.9,-1.2,-0.6,1.8,3.9)$ とします。

$$
X^\top X=\begin{pmatrix}10 & 9\\ 9 & 10\end{pmatrix},\qquad X^\top\boldsymbol{y}=\begin{pmatrix}18.6\\ 18.0\end{pmatrix}
$$

$x_1,x_2$ の相関は $0.9$ です。OLS は $\hat{\boldsymbol{\beta}}=(1.2632,\ 0.6632)$ です。$\lambda=1$ のリッジでは

$$
X^\top X+I=\begin{pmatrix}11 & 9\\ 9 & 11\end{pmatrix},\quad \det=121-81=40
$$

$$
\hat{\boldsymbol{\beta}}_R=\frac{1}{40}\begin{pmatrix}11 & -9\\ -9 & 11\end{pmatrix}\begin{pmatrix}18.6\\ 18.0\end{pmatrix}=\frac{1}{40}\begin{pmatrix}42.6\\ 30.6\end{pmatrix}=\begin{pmatrix}1.065\\ 0.765\end{pmatrix}
$$

2つの係数の差が縮まり、ノルムは $1.4267$ から $1.3113$ に小さくなります。$\lambda=5$ では $(0.8125,\ 0.7125)$ です。

**例2：MSE が OLS より小さくなる（理論計算）**

例1の $X$ で、真の係数を $\boldsymbol{\beta}=(1,\ 0.8)$、$\sigma^2=1$ とします。$X^\top X$ の固有値は $19$ と $1$ です。

- OLS：不偏で、分散の和は $\sigma^2(1/19+1/1)=1.0526$。これがそのまま MSE です。
- リッジ $\lambda=4$：バイアスの2乗和 $0.0618$、分散の和 $0.0759$、MSE は $0.1377$。

MSE は OLS の約13%にまで下がります。数値的に MSE を最小にする $\lambda$ は約 $3.81$（MSE $0.1375$）です。

## 現場での使われ方

- **マーケティング**：テレビ・Web・チラシなど互いに相関する広告費で売上を説明するとき、係数の符号が反転するのを防ぐためにリッジを使います。
- **医療**：多数の遺伝子発現量（$p>n$）から検査値を予測するとき、OLS は解けませんがリッジなら推定できます。
- **製造・品質管理**：近赤外スペクトルの多数の波長（強く相関）から成分濃度を予測する検量モデルに使います。
- **IT・Webサービス**：特徴量の多い予測モデルで、交差検証で $\lambda$ を選び、過学習を抑えます。

## グラフ・イメージ

上の図は解の経路で、$\lambda$ を大きくするとすべての係数がなめらかに0へ近づきます。Lasso と違い、途中で0に張りつくことはありません。
下の図は例2の MSE の分解で、バイアスの増加と分散の減少が釣り合う所で MSE が最小になります。

## 試験のポイント

- 解 $(X^\top X+\lambda I)^{-1}X^\top\boldsymbol{y}$ を、目的関数の微分から導かせる問題が典型です。
- 期待値 $(X^\top X+\lambda I)^{-1}X^\top X\boldsymbol{\beta}$ から、リッジ推定量は**偏りがある**（$\lambda>0$）ことを示せるようにします。
- 直交計画では $\hat\beta_j/(1+\lambda)$、固有値で見ると $d_j/(d_j+\lambda)$ 倍、という縮小の形を覚えます。
- 説明変数の尺度で罰則の効き方が変わるので、**標準化してから**当てはめ、切片には罰則をかけません。
- ベイズ的には、$\boldsymbol{\beta}\sim N(\boldsymbol{0},(\sigma^2/\lambda)I)$ を事前分布にしたときの事後モード（＝事後平均）です。
- $\lambda$ は交差検証法などで選びます。

## よくある間違い

- × リッジ回帰は不偏推定量 → ○ $\lambda>0$ では偏りがあります。分散を減らして MSE を小さくするのが狙いです。
- × リッジで係数がちょうど0になり変数選択ができる → ○ 縮むだけで0にはなりません。変数選択には Lasso を使います。
- × 説明変数をそのままの単位で使う → ○ 単位を変えると罰則の効き方が変わるので、標準化します。
- × $\lambda$ は大きいほど安全 → ○ $\lambda\to\infty$ で係数はすべて0に近づき、バイアスが支配的になります。
