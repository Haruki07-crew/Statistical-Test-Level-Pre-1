---
id: mean-vector
title: 期待値ベクトル
reading: きたいちべくとる
en: Mean vector
aliases: [平均ベクトル, 標本平均ベクトル]
chapter: 3
importance: B
tags: [多変量, ベクトル, 線形変換]
related: [variance-covariance-matrix, correlation-matrix, expectation-variance-properties, multivariate-normal-distribution, multinomial-distribution]
charts:
  - type: scatter
    model: bvnorm
    params: { rho: 0.6, n: 300 }
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
    caption: 2変量の点の雲の「重心」が期待値ベクトル。ρ を変えると雲の傾きや細さは変わるが、重心の位置（ここでは原点）は変わらない。
---

## 一言でいうと

複数の確率変数をまとめたベクトルについて、それぞれの期待値を縦に並べたものです。
多変量データの「中心（重心）」を表します。

## 定義・数式

$p$ 個の確率変数を並べた確率ベクトル $\boldsymbol{X} = (X_1,\dots,X_p)^\top$ の**期待値ベクトル**（平均ベクトル）は

$$
\boldsymbol{\mu} = E[\boldsymbol{X}] =
\begin{pmatrix} E[X_1] \\ \vdots \\ E[X_p] \end{pmatrix}
$$

です（各成分の期待値が存在するとき）。$q\times p$ の定数行列 $A$ と $q$ 次元の定数ベクトル $\boldsymbol{b}$ について

$$
E[A\boldsymbol{X} + \boldsymbol{b}] = A\boldsymbol{\mu} + \boldsymbol{b}
$$

が成り立ちます。データ $\boldsymbol{x}_1,\dots,\boldsymbol{x}_n$ では**標本平均ベクトル** $\bar{\boldsymbol{x}} = \frac1n\sum_{i=1}^n \boldsymbol{x}_i$ を使い、$E[\bar{\boldsymbol{X}}]=\boldsymbol{\mu}$（不偏）です。

## 数式の読み解き

- 期待値ベクトルは、成分ごとに期待値をとるだけです。新しい計算規則があるわけではありません。
- $E[A\boldsymbol{X}+\boldsymbol{b}]$ の第 $i$ 成分は $E\big[\sum_j a_{ij}X_j + b_i\big] = \sum_j a_{ij}E[X_j] + b_i$ で、1変数の期待値の線形性をそのまま使っています。独立性は不要です。
- 特に $A$ が1行（$\boldsymbol{a}^\top$）なら、$E[\boldsymbol{a}^\top\boldsymbol{X}] = \boldsymbol{a}^\top\boldsymbol{\mu}$ です。合計点や加重和の期待値がこれで求まります。
- 期待値ベクトルは「中心の位置」だけを表し、各変数のばらつきや変数どうしの関係は分散共分散行列が表します。多変量正規分布 $N_p(\boldsymbol{\mu},\Sigma)$ はこの2つで完全に決まります。

## 具体例

**例1：2科目の合計点と差**

数学と英語の点数 $\boldsymbol{X}=(X_1,X_2)^\top$ の期待値ベクトルが $\boldsymbol{\mu}=(60, 70)^\top$ とします。合計点と差（数学−英語）は

$$
\begin{pmatrix} X_1+X_2 \\ X_1-X_2 \end{pmatrix}
= \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\boldsymbol{X},
\qquad
E = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 60 \\ 70 \end{pmatrix}
= \begin{pmatrix} 130 \\ -10 \end{pmatrix}
$$

です。2科目が相関していても関係なく計算できます。

**例2：多項分布**

10回の試行で、3つの結果が確率 $(0.5, 0.3, 0.2)$ で起こるとき、各結果の回数 $\boldsymbol{N}$ は多項分布に従い

$$
E[\boldsymbol{N}] = 10\times(0.5,\ 0.3,\ 0.2)^\top = (5,\ 3,\ 2)^\top
$$

です。成分の和が試行回数10になります。

## 現場での使われ方

- **製造・品質管理**：寸法・重さ・硬さなど複数の品質特性の目標値をベクトルで管理し、製品の平均ベクトルが目標からずれていないかを多変量の管理図（ホテリングの $T^2$）で監視します。
- **マーケティング**：顧客セグメントごとに、購入頻度・購入額・最終購入からの日数などの平均ベクトルを比べ、特徴を把握します。
- **医療**：複数の検査項目の平均ベクトルを、治療群と対照群で比べます（多変量の検定）。
- **IT・Webサービス**：クラスタリング（k-means）では、各クラスタの中心として平均ベクトルを使います。

## グラフ・イメージ

2変量正規分布からの散布図です。点の雲の重心が期待値ベクトル（ここでは原点）です。
相関係数を変えると雲の傾きや細さは変わりますが、重心の位置は変わらないことを確かめてください。位置と広がりが別の量で表されることがわかります。

## 試験のポイント

- $E[A\boldsymbol{X}+\boldsymbol{b}]=A\boldsymbol{\mu}+\boldsymbol{b}$ を、分散共分散行列の $V[A\boldsymbol{X}+\boldsymbol{b}]=A\Sigma A^\top$ とセットで使えるようにします。
- 多項分布・多変量正規分布の期待値ベクトルは基本です。
- 2次形式の期待値 $E[\boldsymbol{X}^\top B\boldsymbol{X}] = \mathrm{tr}(B\Sigma) + \boldsymbol{\mu}^\top B\boldsymbol{\mu}$ も、期待値ベクトルと分散共分散行列から計算できます。

## よくある間違い

- × 成分どうしが独立でないと $E[A\boldsymbol{X}]=A\boldsymbol{\mu}$ は使えない → ○ 期待値の線形性は独立性と無関係に成り立ちます。
- × 期待値ベクトルが同じなら分布も同じ → ○ ばらつきや相関（分散共分散行列）が違えば、分布は異なります。
- × $E[X_1X_2] = E[X_1]E[X_2]$ を期待値ベクトルだけから求める → ○ 積の期待値には共分散が必要で、$E[X_1X_2]=\mu_1\mu_2+\sigma_{12}$ です。
