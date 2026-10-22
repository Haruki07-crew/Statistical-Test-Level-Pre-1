---
id: variance-covariance-matrix
title: 分散共分散行列
reading: ぶんさんきょうぶんさんぎょうれつ
en: Variance-covariance matrix
aliases: [共分散行列, 分散行列]
chapter: 3
importance: A
tags: [多変量, 行列, 共分散, 半正定値]
related: [mean-vector, correlation-matrix, covariance-and-correlation, expectation-variance-properties, multivariate-normal-distribution, partial-correlation-coefficient]
charts:
  - type: scatter
    model: bvnorm
    params: { rho: 0.5, n: 300 }
    sliders:
      - { param: rho, label: "相関係数 ρ（非対角成分）", min: -0.95, max: 0.95, step: 0.05 }
      - { param: n, label: "点の数 n", min: 50, max: 1000, step: 50 }
    caption: 分散共分散行列が点の雲の形（楕円の向きと細さ）を決める。非対角成分が正なら右上がり、負なら右下がりの楕円になる。
---

## 一言でいうと

複数の確率変数について、各変数の分散と、すべての組の共分散を1つの行列にまとめたものです。
多変量データの「広がり方と向き」を表し、線形変換したときの分散を行列の計算で一気に求められます。

## 定義・数式

確率ベクトル $\boldsymbol{X}=(X_1,\dots,X_p)^\top$、期待値ベクトル $\boldsymbol{\mu}$ について、**分散共分散行列**は

$$
\Sigma = V[\boldsymbol{X}] = E\big[(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^\top\big]
= \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\
\vdots & & \ddots & \vdots \\
\sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{pmatrix}
$$

です。$\sigma_{ii}=V[X_i]$、$\sigma_{ij}=\mathrm{Cov}[X_i,X_j]$ です（各成分の分散が有限であるとき）。$\Sigma = E[\boldsymbol{X}\boldsymbol{X}^\top] - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ とも書けます。

主な性質：

- **対称**：$\Sigma^\top = \Sigma$
- **半正定値**：任意のベクトル $\boldsymbol{a}$ で $\boldsymbol{a}^\top\Sigma\boldsymbol{a} = V[\boldsymbol{a}^\top\boldsymbol{X}] \ge 0$
- **線形変換**：$q\times p$ の定数行列 $A$、定数ベクトル $\boldsymbol{b}$ について

$$
V[A\boldsymbol{X} + \boldsymbol{b}] = A\,\Sigma\,A^\top
$$

- $\boldsymbol{X}$ と $\boldsymbol{Y}$ の共分散行列は $\mathrm{Cov}[\boldsymbol{X},\boldsymbol{Y}] = E[(\boldsymbol{X}-\boldsymbol{\mu}_X)(\boldsymbol{Y}-\boldsymbol{\mu}_Y)^\top]$ で、$\mathrm{Cov}[A\boldsymbol{X}, B\boldsymbol{Y}] = A\,\mathrm{Cov}[\boldsymbol{X},\boldsymbol{Y}]\,B^\top$ です。

## 数式の読み解き

- $(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^\top$ は $p\times p$ 行列で、$(i,j)$ 成分が $(X_i-\mu_i)(X_j-\mu_j)$ です。その期待値をとれば、対角が分散、非対角が共分散になります。
- **線形変換の公式の導出**：$\boldsymbol{Y}=A\boldsymbol{X}+\boldsymbol{b}$ なら $\boldsymbol{Y}-E[\boldsymbol{Y}] = A(\boldsymbol{X}-\boldsymbol{\mu})$ なので

$$
V[\boldsymbol{Y}] = E\big[A(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^\top A^\top\big] = A\,\Sigma\,A^\top
$$

1変数の $V[aX+b]=a^2V[X]$ の行列版です。$a^2$ が「$A$ と $A^\top$ で挟む」形になります。

- **半正定値の理由**：$\boldsymbol{a}^\top\Sigma\boldsymbol{a}$ は1変数 $\boldsymbol{a}^\top\boldsymbol{X}$ の分散なので、負にはなりません。等号が成り立つ $\boldsymbol{a}\neq\boldsymbol{0}$ があるのは、変数の間に線形の関係（多重共線性）があるときで、そのとき $\Sigma$ は正則でなくなります。
- 固有値分解 $\Sigma = P\Lambda P^\top$ の固有ベクトルは点の雲（楕円）の軸の向き、固有値はその方向の分散です。これが主成分分析の考え方です。

## 具体例

**例1：和と差の分散**

$\boldsymbol{X}=(X_1,X_2)^\top$ の分散共分散行列が $\Sigma = \begin{pmatrix} 4 & 1.2 \\ 1.2 & 9\end{pmatrix}$ とします。$A=\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ で和と差を作ると

$$
A\Sigma A^\top = \begin{pmatrix} 4+9+2\times1.2 & 4-9 \\ 4-9 & 4+9-2\times1.2 \end{pmatrix}
= \begin{pmatrix} 15.4 & -5 \\ -5 & 10.6 \end{pmatrix}
$$

$V[X_1+X_2]=15.4$、$V[X_1-X_2]=10.6$、和と差の共分散は $V[X_1]-V[X_2]=-5$ です。

なお、この $\Sigma$ の行列式は $4\times9-1.2^2=34.56>0$、固有値は約 $3.7269$ と $9.2731$ でどちらも正なので、正定値（正則）です。2つの固有値の和 $13$ は対角成分の和（分散の合計）と一致し、積 $34.56$ は行列式と一致します。

**例2：分散が最小になる組合せ**

$w X_1 + (1-w) X_2$ の分散 $4w^2 + 9(1-w)^2 + 2.4\,w(1-w)$ を $w$ で微分して0とおくと

$$
w^* = \frac{\sigma_{22}-\sigma_{12}}{\sigma_{11}+\sigma_{22}-2\sigma_{12}} = \frac{9-1.2}{13-2.4} = \frac{7.8}{10.6} \approx 0.7358
$$

最小の分散は $\dfrac{\sigma_{11}\sigma_{22}-\sigma_{12}^2}{\sigma_{11}+\sigma_{22}-2\sigma_{12}} = \dfrac{36-1.44}{10.6} \approx 3.2604$ です。分散の小さい $X_1$ だけを持つ場合（4）よりさらに小さくなります。

**例3：多項分布（正則でない例）**

10回の試行で確率 $(0.5, 0.3, 0.2)$ の多項分布では、$\sigma_{ii}=np_i(1-p_i)$、$\sigma_{ij}=-np_ip_j$ なので

$$
\Sigma = \begin{pmatrix} 2.5 & -1.5 & -1.0 \\ -1.5 & 2.1 & -0.6 \\ -1.0 & -0.6 & 1.6 \end{pmatrix}
$$

です。各行の和が0で、行列式も0です。回数の合計が10に固定されている（$\boldsymbol{a}=(1,1,1)^\top$ で $V[\boldsymbol{a}^\top\boldsymbol{N}]=0$）ためです。

## 現場での使われ方

- **金融・保険**：資産の収益率の分散共分散行列 $\Sigma$ と保有比率 $\boldsymbol{w}$ から、ポートフォリオのリスク $\boldsymbol{w}^\top\Sigma\boldsymbol{w}$ を計算し、最適な配分を求めます。
- **製造・品質管理**：複数の品質特性の分散共分散行列を使い、マハラノビス距離で「全体として異常な製品」を検出します。
- **IT・Webサービス**：多数の指標の分散共分散行列（または相関行列）を固有値分解し、主成分分析で次元を減らして可視化します。
- **医療**：回帰分析で推定した係数の分散共分散行列から、係数の組合せ（例：2つの治療効果の差）の標準誤差を計算します。

## グラフ・イメージ

2変量正規分布からの散布図です。非対角成分（ここでは相関係数 ρ）を大きくすると、点の雲が右上がりの細長い楕円になります。
楕円の長軸の方向が $\Sigma$ の最大固有値の固有ベクトル、長軸方向の広がりがその固有値に対応します。

## 試験のポイント

- $V[A\boldsymbol{X}+\boldsymbol{b}]=A\Sigma A^\top$ は最重要の公式です。$V[\boldsymbol{a}^\top\boldsymbol{X}]=\boldsymbol{a}^\top\Sigma\boldsymbol{a}$ とあわせて、和・差・加重和の分散を行列で計算できるようにします。
- 対称・半正定値であること、正則でなくなる条件（線形従属）を説明できるようにします。
- 多項分布の分散共分散行列（非対角が負）は重要です。
- 最小分散の組合せ（例2の形）は、微分して求める計算問題として典型的です。
- 主成分分析（固有値分解）、多変量正規分布の密度（$\Sigma^{-1}$ と $|\Sigma|$）、回帰係数の分散 $\sigma^2(X^\top X)^{-1}$ へとつながります。

## よくある間違い

- × $V[A\boldsymbol{X}] = A^2\Sigma$ や $A^\top\Sigma A$ → ○ 正しくは $A\Sigma A^\top$ です。行列のサイズ（$q\times p$、$p\times p$、$p\times q$）で順番を確かめます。
- × 非対角成分は相関係数 → ○ 非対角成分は共分散です。相関係数を並べたものは相関係数行列です。
- × 成分の数値が適当に与えられた行列はいつでも分散共分散行列になる → ○ 対称で半正定値でなければなりません。例えば $\sigma_{12}^2 > \sigma_{11}\sigma_{22}$ となる2×2行列はあり得ません。
- × 標本の分散共分散行列は $n$ で割るか $n-1$ で割るかは関係ない → ○ 不偏推定量にするには $n-1$ で割ります。最尤推定量は $n$ で割ったものです。
