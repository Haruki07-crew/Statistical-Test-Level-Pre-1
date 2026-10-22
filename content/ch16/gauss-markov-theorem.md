---
id: gauss-markov-theorem
title: ガウス・マルコフの定理
reading: がうすまるこふのていり
en: Gauss-Markov theorem
aliases: [BLUE, 最良線形不偏推定量, ガウス＝マルコフの定理]
chapter: 16
importance: A
tags: [回帰分析, 最小二乗法, 不偏推定, 有効性]
related: [least-squares-method, multiple-regression-model, unbiased-estimator, efficient-estimator, variance-covariance-matrix, generalized-least-squares]
charts:
  - type: pdf
    dist: norm
    params: { mu: 2, sigma: 0.3162 }
    x: { min: 0.5, max: 3.5 }
    compare:
      - { dist: norm, params: { mu: 2, sigma: 0.3333 }, label: "推定量B（(y4+y5−y1−y2)/6）：分散 0.1111σ²" }
      - { dist: norm, params: { mu: 2, sigma: 0.3536 }, label: "推定量A（両端の傾き (y5−y1)/4）：分散 0.125σ²" }
    caption: x=1,…,5、真の傾き 2、σ=1 のときの3つの線形不偏推定量の分布。どれも中心は 2 だが、最小二乗推定量（実線、分散 0.1σ²）が最も狭く、最も精度が高い。
---

## 一言でいうと

誤差が「平均 0・等分散・無相関」なら、最小二乗推定量は、**データの一次式で書ける不偏推定量の中で分散が最小**である、という定理です。
この性質を BLUE（Best Linear Unbiased Estimator、最良線形不偏推定量）といいます。

## 定義・数式

**仮定**：$\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$、$X$ は固定された $n\times(p+1)$ 行列で $\operatorname{rank}X=p+1$、

$$
E[\boldsymbol{\varepsilon}]=\boldsymbol{0},\qquad V[\boldsymbol{\varepsilon}]=\sigma^2I_n
$$

（正規分布は仮定しない）。

**定理**：最小二乗推定量 $\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top\boldsymbol{y}$ は不偏で、$V[\hat{\boldsymbol{\beta}}]=\sigma^2(X^\top X)^{-1}$ である。さらに、$\tilde{\boldsymbol{\beta}}=C\boldsymbol{y}$ の形の任意の不偏推定量（線形不偏推定量）に対して

$$
V[\tilde{\boldsymbol{\beta}}]-V[\hat{\boldsymbol{\beta}}]\ \text{は非負定値}
$$

が成り立つ。とくに、任意のベクトル $\boldsymbol{c}$ について $V[\boldsymbol{c}^\top\tilde{\boldsymbol{\beta}}]\ge V[\boldsymbol{c}^\top\hat{\boldsymbol{\beta}}]$ で、各係数ごとにも $V[\tilde\beta_j]\ge V[\hat\beta_j]$ である。

## 数式の読み解き

**期待値と分散（試験で必須の計算）**

$\hat{\boldsymbol{\beta}}=(X^\top X)^{-1}X^\top(X\boldsymbol{\beta}+\boldsymbol{\varepsilon})=\boldsymbol{\beta}+(X^\top X)^{-1}X^\top\boldsymbol{\varepsilon}$ なので、

$$
E[\hat{\boldsymbol{\beta}}]=\boldsymbol{\beta}+(X^\top X)^{-1}X^\top E[\boldsymbol{\varepsilon}]=\boldsymbol{\beta}
$$

$$
V[\hat{\boldsymbol{\beta}}]=(X^\top X)^{-1}X^\top\,(\sigma^2I)\,X(X^\top X)^{-1}=\sigma^2(X^\top X)^{-1}
$$

不偏性には $E[\boldsymbol{\varepsilon}]=\boldsymbol{0}$ だけ、分散の式には $V[\boldsymbol{\varepsilon}]=\sigma^2I$ が必要です。

**最小性の証明**

$\tilde{\boldsymbol{\beta}}=C\boldsymbol{y}$ が不偏であるには、すべての $\boldsymbol{\beta}$ で $E[C\boldsymbol{y}]=CX\boldsymbol{\beta}=\boldsymbol{\beta}$、つまり $CX=I$ が必要です。
$C=(X^\top X)^{-1}X^\top+D$ とおくと、$CX=I+DX$ なので $DX=O$ です。すると

$$
\begin{aligned}
V[\tilde{\boldsymbol{\beta}}]&=\sigma^2CC^\top\\
&=\sigma^2\bigl\{(X^\top X)^{-1}+(X^\top X)^{-1}X^\top D^\top+DX(X^\top X)^{-1}+DD^\top\bigr\}\\
&=\sigma^2(X^\top X)^{-1}+\sigma^2DD^\top
\end{aligned}
$$

（$DX=O$ で中の2項が消える）。$DD^\top$ は非負定値なので、$V[\tilde{\boldsymbol{\beta}}]-V[\hat{\boldsymbol{\beta}}]=\sigma^2DD^\top\succeq O$ です。等号は $D=O$、つまり $\tilde{\boldsymbol{\beta}}=\hat{\boldsymbol{\beta}}$ のときだけです。

- **B**est：分散が最小。
- **L**inear：$\boldsymbol{y}$ の一次式。
- **U**nbiased：不偏。
- 比べる相手は「線形かつ不偏」な推定量だけです。非線形な推定量や、偏りを許す推定量（リッジ回帰など）には、平均二乗誤差でもっと良いものがありえます。

## 具体例

**例1：傾きの推定量を3つ比べる**

$x=1,2,3,4,5$ で単回帰 $y_i=\beta_0+\beta_1x_i+\varepsilon_i$ を考えます（$\bar x=3,\ S_{xx}=10$）。$\beta_1$ の線形不偏推定量を3つ作ります。

- 推定量A（両端を結ぶ）：$(y_5-y_1)/4$。期待値は $\beta_1(5-1)/4=\beta_1$、分散は $2\sigma^2/16=0.125\sigma^2$。
- 推定量B（上2つと下2つの差）：$(y_4+y_5-y_1-y_2)/6$。期待値は $\beta_1(4+5-1-2)/6=\beta_1$、分散は $4\sigma^2/36=0.1111\sigma^2$。
- 最小二乗推定量：$\hat\beta_1=\sum(x_i-3)y_i/10$。分散は $\sigma^2/S_{xx}=0.1\sigma^2$。

どれも不偏ですが、最小二乗推定量の分散が最小です。推定量Aの効率は $0.1/0.125=0.8$ です。

**例2：共通データでの分散共分散行列**

8店舗のデータでは $(X^\top X)^{-1}=\frac{1}{756}\begin{pmatrix}1140&60&-312\\60&96&-96\\-312&-96&159\end{pmatrix}$、$\sigma^2$ の不偏推定値は $s^2=S_e/(n-p-1)=4.7738/5=0.9548$ です。よって推定分散共分散行列は

$$
s^2(X^\top X)^{-1}=\begin{pmatrix}1.4397&0.0758&-0.3940\\0.0758&0.1212&-0.1212\\-0.3940&-0.1212&0.2008\end{pmatrix}
$$

$\hat\beta_1$ と $\hat\beta_2$ の相関係数は $-96/\sqrt{96\times159}=-0.777$ です。$x_1,x_2$ の相関が $+0.777$ なので、係数の推定値は逆向きに連動します（一方が大きく出ると他方は小さく出やすい）。

## 現場での使われ方

- **製造・品質管理**：測定誤差が等分散・無相関とみなせる校正実験では、最小二乗法以外の線形な推定方法を考える必要がない、という根拠になります。
- **金融・保険**：収益率のデータは分散が時期によって変わる（不均一分散）ことが多く、定理の前提が崩れます。その場合は加重最小二乗法や一般化最小二乗法を使うほうが効率が良くなります。
- **行政・調査**：集計単位ごとに標本サイズが違うデータでは誤差分散が等しくないので、重み付きの推定に切り替える判断材料になります。

## グラフ・イメージ

例1の3つの推定量の分布です（$\beta_1=2,\ \sigma=1$ とし、正規近似で描いています）。どの曲線も中心は真の値 2 ですが、最小二乗推定量の山が最も高く細く、ばらつきが最小です。

## 試験のポイント

- $E[\hat{\boldsymbol{\beta}}]=\boldsymbol{\beta}$ と $V[\hat{\boldsymbol{\beta}}]=\sigma^2(X^\top X)^{-1}$ の導出は、ほぼ必ず使います。単回帰なら $V[\hat\beta_1]=\sigma^2/S_{xx}$。
- 定理の仮定に正規性は**含まれません**。必要なのは平均 0・等分散・無相関の3つです。正規性を加えると、最小二乗推定量は（線形に限らず）すべての不偏推定量の中で分散最小（UMVU）になります。
- 証明は「$C=(X^\top X)^{-1}X^\top+D$、$DX=O$」とおく流れを覚えます。
- 等分散が崩れて $V[\boldsymbol{\varepsilon}]=\sigma^2\Sigma$ のときの BLUE は一般化最小二乗推定量 $(X^\top\Sigma^{-1}X)^{-1}X^\top\Sigma^{-1}\boldsymbol{y}$ です。

## よくある間違い

- × 最小二乗推定量はあらゆる推定量の中で最良 → ○ 「線形かつ不偏」の範囲での最良です。偏りを許すリッジ回帰などは、平均二乗誤差で上回ることがあります。
- × 定理には誤差の正規性が必要 → ○ 不要です。正規性が必要なのは $t$ 検定・$F$ 検定などの分布の議論です。
- × 不均一分散でも最小二乗推定量は BLUE → ○ 不偏性は保たれますが、最良ではなくなり、通常の標準誤差の式も正しくなくなります。
