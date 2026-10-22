---
id: tobit-model
title: トービットモデル
reading: とーびっともでる
en: Tobit model
aliases: [打ち切り回帰モデル, トービット回帰, "censored regression model"]
chapter: 18
importance: C
tags: [質的回帰, 打ち切り, 潜在変数, 最尤法]
related: [probit-model, censored-data, truncated-data, maximum-likelihood-estimator, logistic-regression]
---

## 一言でいうと

応答が0のところに大量に積み上がる（0未満が観測できず0として記録される）データのための回帰モデルです。
目に見えない「潜在的な応答」が正規の線形回帰に従い、それが0以下なら0と観測される、と考えます。

## 定義・数式

観測 $i$ について、潜在変数 $y_i^*$ と観測値 $y_i$ を

$$
y_i^* = \boldsymbol{x}_i^\top\boldsymbol{\beta}+\varepsilon_i,\quad \varepsilon_i \sim N(0,\sigma^2)\ (\text{独立}),\qquad
y_i = \max(0,\ y_i^*)
$$

とします（0で左から打ち切られたモデル）。$\Phi,\ \phi$ を標準正規分布の分布関数・密度とすると、尤度は

$$
L(\boldsymbol{\beta},\sigma) = \prod_{y_i=0}\left\{1-\Phi\!\left(\frac{\boldsymbol{x}_i^\top\boldsymbol{\beta}}{\sigma}\right)\right\}
\prod_{y_i>0}\frac{1}{\sigma}\,\phi\!\left(\frac{y_i-\boldsymbol{x}_i^\top\boldsymbol{\beta}}{\sigma}\right)
$$

で、これを最大にして $\boldsymbol{\beta},\sigma$ を推定します。$z_i=\boldsymbol{x}_i^\top\boldsymbol{\beta}/\sigma$ とおくと、観測値の期待値は

$$
P(y_i=0)=1-\Phi(z_i),\qquad
E[y_i] = \Phi(z_i)\,\boldsymbol{x}_i^\top\boldsymbol{\beta} + \sigma\,\phi(z_i)
$$

です。

## 数式の読み解き

- 尤度の前半は「0になった観測」の確率で、プロビットモデルと同じ形です。後半は「正の値が観測された観測」の正規密度で、線形回帰と同じ形です。トービットは両者を1つにしたモデルです。
- $\beta_j$ は**潜在変数** $y^*$ への効果です。観測値 $y$ への効果は $\partial E[y_i]/\partial x_{ij} = \Phi(z_i)\beta_j$ で、0になりにくい人ほど $\beta_j$ に近づきます。
- 正の値だけを使った最小二乗法は、0付近の小さい値が欠けるため係数を0の方向に偏らせます。0を含めた最小二乗法も同様に偏ります。

## 具体例

**レジャー支出（万円）のモデル**

ある世帯の属性から $\boldsymbol{x}^\top\boldsymbol{\beta}=1$、$\sigma=2$ と推定されたとします。$z=1/2=0.5$ なので

$$
P(y=0) = 1-\Phi(0.5) = 1-0.6915 = 0.3085
$$

支出ゼロの確率は約31%です。支出の期待値は

$$
E[y] = 0.6915\times1 + 2\times\phi(0.5) = 0.6915 + 2\times0.3521 = 1.3956
$$

支出がある世帯に限った平均は $E[y\mid y>0] = 1 + 2\times0.3521/0.6915 = 2.0183$ です。
また、ある説明変数の係数が $\beta_j=0.5$ なら、観測された支出への限界効果は $0.6915\times0.5=0.3457$ です。

## 現場での使われ方

- **マーケティング**：商品カテゴリへの年間購入額（買わない人は0）を、所得や年齢で説明します。
- **行政・調査**：労働時間や寄付額など、0の人が多い調査項目を分析します。

## 試験のポイント

- 潜在変数 $y^*=\max(0,\cdot)$ の構造と、尤度が「0の部分の確率 × 正の部分の密度」になることを押さえます。
- 0か正かだけを見ればプロビットモデルになる、という関係が重要です。
- 打ち切り（0として記録される）と切断（0以下の人が標本に入らない）の違いを区別します。

## よくある間違い

- × 0のデータを除いて最小二乗法を当てはめればよい → ○ 選ばれ方に偏りがあり、係数が0の方向に偏ります。
- × $\beta_j$ がそのまま観測値 $y$ の増え方 → ○ $\beta_j$ は潜在変数への効果で、$y$ への効果は $\Phi(z)\beta_j$ に縮みます。
