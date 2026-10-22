---
id: factor-loading
title: 因子負荷量
reading: いんしふかりょう
en: Factor loading
aliases: [因子負荷, 因子負荷行列, 因子パターン, 因子構造]
chapter: 25
importance: A
tags: [因子分析, 負荷量, 解釈]
related: [factor-analysis-model, communality, factor-rotation, promax-rotation, principal-component-loading, covariance-and-correlation]
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
      xlabel: 第1因子の負荷量
      ylabel: 第2因子の負荷量
    caption: 因子負荷量プロット。点の座標が（第1因子との相関, 第2因子との相関）。矢印の長さの2乗が共通性で、単位円に近いほど共通因子でよく説明される。
---

## 一言でいうと

各共通因子が、それぞれの観測変数にどれだけ強く効いているかを表す係数です。
標準化した変数と直交因子では、変数と因子の相関係数そのものになり、因子に名前を付けるときの根拠になります。

## 定義・数式

因子分析モデル $\boldsymbol{x}=\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon}$ の係数行列

$$
\Lambda=(\lambda_{jk})=\begin{pmatrix}\lambda_{11}&\cdots&\lambda_{1m}\\\vdots&&\vdots\\\lambda_{p1}&\cdots&\lambda_{pm}\end{pmatrix}
$$

を**因子負荷行列**、その成分 $\lambda_{jk}$ を変数 $x_j$ の第 $k$ 因子に対する**因子負荷量**といいます。

直交因子（$V[\boldsymbol{f}]=I_m$）のとき

$$
\mathrm{Cov}[\boldsymbol{x},\boldsymbol{f}]=\Lambda,\qquad\text{すなわち}\quad \lambda_{jk}=\mathrm{Cov}[x_j,f_k]
$$

で、$x_j$ を標準化していれば $\lambda_{jk}=\mathrm{Corr}[x_j,f_k]$ です。

因子どうしに相関を許す斜交因子（$V[\boldsymbol{f}]=\Phi$、対角は1）では、2つの行列を区別します。

$$
\text{因子パターン}\ \Lambda\ (\text{モデルの係数}),\qquad
\text{因子構造}\ \mathrm{Cov}[\boldsymbol{x},\boldsymbol{f}]=\Lambda\Phi\ (\text{変数と因子の相関})
$$

## 数式の読み解き

- 行（変数ごと）に見ると、その変数がどの因子で決まっているかがわかります。行の2乗和は共通性 $h_j^2$ です。
- 列（因子ごと）に見ると、その因子がどの変数に強く効くかがわかります。列の2乗和はその因子が説明する分散の合計です。
- **導出**：$\boldsymbol{f}$ と $\boldsymbol{\varepsilon}$ が無相関なので

$$
\mathrm{Cov}[\boldsymbol{x},\boldsymbol{f}]=E[(\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon})\boldsymbol{f}^\top]=\Lambda E[\boldsymbol{f}\boldsymbol{f}^\top]+E[\boldsymbol{\varepsilon}\boldsymbol{f}^\top]=\Lambda V[\boldsymbol{f}]
$$

  直交なら $V[\boldsymbol{f}]=I$ で $\Lambda$、斜交なら $\Lambda\Phi$ になります。
- 2つの変数の相関は、負荷量ベクトルの内積 $r_{ij}=\sum_k\lambda_{ik}\lambda_{jk}$（直交の場合）です。図で矢印どうしが近い変数ほど相関が高くなります。
- 符号の向きは任意です。ある因子の列全体の符号を反転しても $\Lambda\Lambda^\top$ は変わらないので、解釈しやすい向き（主要な負荷量が正になる向き）にそろえて報告します。

## 具体例

**例1：4教科の負荷行列を読む**

標準化した4教科で、次の負荷行列が得られたとします（直交因子）。

$$
\Lambda=\begin{pmatrix}0.8&0.2\\0.7&0.3\\0.2&0.8\\0.3&0.7\end{pmatrix}
\quad(\text{行は国語・英語・数学・理科})
$$

第1因子は国語 0.8・英語 0.7 で大きいので「文系の力」、第2因子は数学 0.8・理科 0.7 で「理系の力」と解釈できます。
列の2乗和は、第1因子が $0.64+0.49+0.04+0.09=1.26$、第2因子も $1.26$ です。合計 $2.52$ は全分散 4 の $63\%$ にあたります。
英語と理科の相関は内積で $0.7\times0.3+0.3\times0.7=0.42$ です。

**例2：斜交解のパターンと構造**

同じデータにプロマックス回転をかけると、numpy の計算で因子間相関 $\phi=0.590$ と、国語の因子パターン $(0.865,\,-0.072)$ が得られます。
因子構造（国語と各因子の相関）は $\Lambda\Phi$ の第1行で

$$
(0.865+(-0.072)\times0.590,\ \ 0.865\times0.590+(-0.072))=(0.823,\ 0.438)
$$

です。パターンでは第2因子の係数がほぼ0なのに、構造では 0.438 と大きく出ます。第2因子が第1因子と相関しているので、その分が乗るからです。

## 現場での使われ方

- **行政・調査**：アンケート項目の負荷量が 0.4 程度以上の因子に項目を割り当て、尺度（下位尺度）を作ります。
- **マーケティング**：商品評価項目の負荷量から因子に「高級感」「手軽さ」などの名前を付け、製品の位置づけを議論します。
- **医療**：症状尺度の各項目の負荷量を、別の集団（年齢層や国）で求めた結果と比べ、尺度が同じ構造を測っているかを確かめます。
- **教育**：テスト問題の負荷量を見て、想定と違う因子に効いている問題を見直します。

## グラフ・イメージ

各変数を負荷量の座標に置いた図です。
国語・英語は横軸（第1因子）寄り、数学・理科は縦軸（第2因子）寄りにあり、2つのまとまりが見えます。
矢印の長さの2乗が共通性で、単位円の外には出ません（標準化した変数の場合）。

## 試験のポイント

- 直交因子では $\lambda_{jk}=\mathrm{Cov}[x_j,f_k]$（標準化なら相関）であることを導けるようにします。
- 行の2乗和＝共通性、列の2乗和＝因子の説明する分散、の2つを使い分けます。
- 斜交解では因子パターン $\Lambda$ と因子構造 $\Lambda\Phi$ が異なります。どちらを問われているかを確認します。
- 負荷行列は回転で変わりますが、共通性と $\Lambda\Lambda^\top$ は直交回転で変わりません。

## よくある間違い

- × 斜交解でも負荷量は変数と因子の相関 → ○ 相関は因子構造 $\Lambda\Phi$ です。パターンの値は1を超えることもあります。
- × 主成分負荷量と因子負荷量は同じ値になる → ○ 主成分負荷量は $\sqrt{\lambda_k}h_{jk}$ で独自分散を考えません。因子負荷量は独自分散を除いた共通部分から求めます。
- × 列の2乗和を寄与率と呼んで行と混同する → ○ 変数ごとの説明率は行の2乗和（共通性）、因子ごとの説明量は列の2乗和です。
