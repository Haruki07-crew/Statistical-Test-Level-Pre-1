---
id: factor-analysis-model
title: 因子分析モデル
reading: いんしぶんせきもでる
en: Factor analysis model
aliases: [因子モデル, 因子分析, 探索的因子分析]
chapter: 25
importance: A
tags: [多変量解析, 因子分析, 潜在変数, 共分散構造]
related: [common-factor-unique-factor, factor-loading, communality, factor-rotation, principal-component-analysis, variance-covariance-matrix]
charts:
  - type: custom
    id: points-2d
    params:
      points:
        - { label: 国語, x: 0.8, y: 0.2 }
        - { label: 英語, x: 0.7, y: 0.3 }
        - { label: 数学, x: 0.2, y: 0.8 }
        - { label: 理科, x: 0.3, y: 0.7 }
      arrows: 1
      circle: 1
      xlabel: 第1因子（文系）の負荷量
      ylabel: 第2因子（理系）の負荷量
    caption: 例1の因子負荷量。国語・英語は第1因子の軸の近く、数学・理科は第2因子の軸の近くに集まる。原点からの距離の2乗が共通性。
  - type: custom
    id: bar-data
    params:
      labels: [国語, 英語, 数学, 理科]
      values: [0.68, 0.58, 0.68, 0.58]
      xlabel: 変数
      ylabel: 共通性（分散1のうち共通因子で説明される割合）
    caption: 例1の共通性。棒の残り（1 との差）が独自分散で、国語・数学は 0.32、英語・理科は 0.42。
---

## 一言でいうと

たくさんの観測変数の相関を、少数の目に見えない共通の要因（共通因子）で説明するモデルです。
「国語と英語の点がそろって高いのは、背後に『言語能力』があるからだ」という考え方を式にしたものです。

## 定義・数式

$p$ 個の観測変数 $\boldsymbol{x}=(x_1,\dots,x_p)^\top$（平均を引いたもの）を、$m$ 個（$m<p$）の**共通因子** $\boldsymbol{f}=(f_1,\dots,f_m)^\top$ と、$p$ 個の**独自因子** $\boldsymbol{\varepsilon}=(\varepsilon_1,\dots,\varepsilon_p)^\top$ で

$$
\boldsymbol{x}=\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon},\qquad
x_j=\lambda_{j1}f_1+\cdots+\lambda_{jm}f_m+\varepsilon_j
$$

と表します。$\Lambda=(\lambda_{jk})$ は $p\times m$ の**因子負荷行列**です。標準的な仮定（直交因子モデル）は次のとおりです。

$$
E[\boldsymbol{f}]=\boldsymbol{0},\quad V[\boldsymbol{f}]=I_m,\quad
E[\boldsymbol{\varepsilon}]=\boldsymbol{0},\quad V[\boldsymbol{\varepsilon}]=\Psi=\mathrm{diag}(\psi_1,\dots,\psi_p),\quad
\mathrm{Cov}[\boldsymbol{f},\boldsymbol{\varepsilon}]=O
$$

このとき $\boldsymbol{x}$ の分散共分散行列は

$$
\Sigma=\Lambda\Lambda^\top+\Psi
$$

という形（**共分散構造**）になります。成分で書くと

$$
\sigma_{jj}=\sum_{k=1}^m\lambda_{jk}^2+\psi_j,\qquad
\sigma_{ij}=\sum_{k=1}^m\lambda_{ik}\lambda_{jk}\quad(i\ne j)
$$

です。

## 数式の読み解き

- $f_k$：直接は測れない潜在変数（能力・態度・品質の要因など）。全変数に共通して効きます。
- $\lambda_{jk}$：因子 $f_k$ が変数 $x_j$ にどれだけ効くかの重み（因子負荷量）。
- $\varepsilon_j$：変数 $x_j$ だけに効く部分。$\Psi$ が**対角**であることが、このモデルの核心です。
- $h_j^2=\sum_k\lambda_{jk}^2$：共通性。$\psi_j$：独自分散。

**導出**：$\boldsymbol{x}=\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon}$ の分散を、$\boldsymbol{f}$ と $\boldsymbol{\varepsilon}$ が無相関であることを使って計算します。

$$
\begin{aligned}
\Sigma=V[\boldsymbol{x}]&=E[(\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon})(\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon})^\top]\\
&=\Lambda E[\boldsymbol{f}\boldsymbol{f}^\top]\Lambda^\top+\Lambda E[\boldsymbol{f}\boldsymbol{\varepsilon}^\top]+E[\boldsymbol{\varepsilon}\boldsymbol{f}^\top]\Lambda^\top+E[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top]\\
&=\Lambda I_m\Lambda^\top+O+O+\Psi=\Lambda\Lambda^\top+\Psi
\end{aligned}
$$

$\Psi$ が対角なので、**変数どうしの共分散（非対角成分）はすべて共通因子から生まれる**ことになります。
「相関は共通因子のせい、残りの個性は独自因子」という役割分担です。

変数を標準化して相関行列 $R$ を分析するときは $\sigma_{jj}=1$ なので、$1=h_j^2+\psi_j$ と分散1が2つに分かれます。

パラメータの数も数えておきます。$\Sigma$ の異なる成分は $p(p+1)/2$ 個です。
モデル側は $\Lambda$ の $pm$ 個と $\Psi$ の $p$ 個ですが、回転の自由度 $m(m-1)/2$ 個は決まらないので差し引きます。
その差が適合度検定の自由度 $\{(p-m)^2-(p+m)\}/2$ です。これが小さいほど、モデルが $\Sigma$ に課す制約は弱くなります。

## 具体例

**例1：4教科の2因子モデル**

標準化した国語・英語・数学・理科の得点に、2因子モデル

$$
\Lambda=\begin{pmatrix}0.8&0.2\\0.7&0.3\\0.2&0.8\\0.3&0.7\end{pmatrix},\qquad
\Psi=\mathrm{diag}(0.32,\,0.42,\,0.32,\,0.42)
$$

が成り立つとします。国語と英語の相関は

$$
r_{12}=0.8\times0.7+0.2\times0.3=0.56+0.06=0.62
$$

国語と数学は $0.8\times0.2+0.2\times0.8=0.32$ です。numpy で $\Lambda\Lambda^\top+\Psi$ を計算すると

$$
R=\begin{pmatrix}1&0.62&0.32&0.38\\0.62&1&0.38&0.42\\0.32&0.38&1&0.62\\0.38&0.42&0.62&1\end{pmatrix}
$$

になります。対角は $0.8^2+0.2^2+0.32=1$ のように、どれもちょうど1です。

**例2：1因子モデルの相関は「積」になる**

1因子で $\boldsymbol{\lambda}=(0.9,0.8,0.7,0.6)^\top$ のとき、$r_{ij}=\lambda_i\lambda_j$ です。たとえば $r_{12}=0.72,\ r_{34}=0.42$ です。
このとき $r_{12}r_{34}=0.3024$、$r_{13}r_{24}=0.63\times0.48=0.3024$ と一致します。
このような「積の等式」（テトラッド条件）が成り立つかどうかが、1因子で説明できるかの目安になります。

## 現場での使われ方

- **行政・調査**：意識調査の多数の質問項目から「生活満足」「将来不安」などの少数の因子を取り出し、尺度を作ります。
- **教育**：テストの各問題がどの能力（読解・計算など）を測っているかを確かめ、問題の構成を見直します。
- **マーケティング**：ブランドイメージの評価項目（高級感・親しみやすさなど）の背後にある軸を見つけ、ポジショニングマップを描きます。
- **医療**：症状チェックリストの項目が、想定した下位尺度（身体症状・気分など）に分かれるかを確認します。

## グラフ・イメージ

上の図は、例1の各変数を（第1因子の負荷量, 第2因子の負荷量）の点で表したものです。
2つのまとまりに分かれ、それぞれが1つの因子の軸に寄っています。
下の棒グラフは共通性で、1 との差が独自分散です。

## 試験のポイント

- $\Sigma=\Lambda\Lambda^\top+\Psi$ を、仮定（$V[\boldsymbol{f}]=I$、$\Psi$ が対角、$\boldsymbol{f}$ と $\boldsymbol{\varepsilon}$ が無相関）から導けるようにします。
- 負荷量から相関（共分散）を計算させる問題、逆に相関行列から1因子の負荷量を逆算させる問題が典型です。1因子なら $\lambda_1^2=r_{12}r_{13}/r_{23}$ で求まります（例2なら $0.72\times0.63/0.56=0.81$）。
- 因子負荷行列は直交行列 $T$ を掛けても $\Sigma$ が変わらない（回転の不定性）ため、回転で解釈しやすい解を選びます。
- 自由度 $\{(p-m)^2-(p+m)\}/2$ が負になる因子数は使えません。$p=4$ なら $m=1$ で自由度2、$m=2$ で $-1$ です。

## よくある間違い

- × 因子分析と主成分分析は同じもの → ○ 主成分分析は変数の合成（$\boldsymbol{y}=H^\top\boldsymbol{x}$）で誤差項を持ちません。因子分析は $\boldsymbol{x}$ を因子から生成するモデルで、独自因子を持ちます。
- × 独自因子どうしも相関してよい → ○ 基本モデルでは $\Psi$ は対角です。独自因子が相関すると、相関を共通因子だけで説明する前提が崩れます。
- × 因子負荷行列は一意に決まる → ○ $\Lambda T$（$T$ は直交行列）も同じ $\Sigma$ を与えます。
