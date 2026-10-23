---
id: structural-equation-modeling
title: 構造方程式モデリング
reading: こうぞうほうていしきもでりんぐ
en: Structural equation modeling
aliases: [SEM, 共分散構造分析, 確認的因子分析, 潜在変数]
chapter: 26
importance: A
tags: [多変量解析, 潜在変数, パス図, 共分散構造, 識別性]
related: [path-analysis, factor-analysis-model, maximum-likelihood-factor-analysis, goodness-of-fit-indices, directed-acyclic-graph, likelihood-ratio-test]
charts:
  - type: custom
    id: graph-diagram
    params:
      nodes: [ξ, x1, x2, x3, x4]
      edges: [[0, 1], [0, 2], [0, 3], [0, 4]]
      directed: 1
      weights: [0.63, 0.78, 0.70, 0.54]
      layout: circle
    caption: 例2の1因子モデルのパス図。潜在変数 ξ（満足度）から4つの観測変数へ矢印が出ており、矢印の数字は最尤法で推定した標準化因子負荷量。各観測変数には独自因子（誤差）が加わる。
---

## 一言でいうと

直接は測れない**潜在変数**（満足度、学力、ブランド力など）と観測変数の間の因果の仕組みを、パス図で仮説として描き、その仮説が**データの共分散行列をうまく再現できるか**で検証する方法です。
回帰分析・パス解析・因子分析を一つの枠組みにまとめたものです。

## 定義・数式

構造方程式モデルは、測定方程式と構造方程式の2つの部分からなります（LISREL型の表記）。

$$
\text{測定方程式：}\quad \boldsymbol{x}=\Lambda_x\boldsymbol{\xi}+\boldsymbol{\delta},\qquad \boldsymbol{y}=\Lambda_y\boldsymbol{\eta}+\boldsymbol{\varepsilon}
$$

$$
\text{構造方程式：}\quad \boldsymbol{\eta}=B\boldsymbol{\eta}+\Gamma\boldsymbol{\xi}+\boldsymbol{\zeta}
$$

- $\boldsymbol{\xi}$：外生的な潜在変数（モデルの中で原因にだけなる）、$\boldsymbol{\eta}$：内生的な潜在変数（モデルの中で結果になる）
- $\boldsymbol{x},\boldsymbol{y}$：観測変数、$\boldsymbol{\delta},\boldsymbol{\varepsilon},\boldsymbol{\zeta}$：誤差（互いに、また $\boldsymbol\xi$ と無相関）

モデルのパラメータ $\boldsymbol\theta$（負荷量・パス係数・分散など）から、観測変数の共分散行列が $\Sigma(\boldsymbol\theta)$ と決まります。たとえば測定部分だけの1因子モデル $\boldsymbol{x}=\boldsymbol\lambda\xi+\boldsymbol\delta$（$\mathrm{Var}(\xi)=1$）なら

$$
\Sigma(\boldsymbol\theta)=\boldsymbol\lambda\boldsymbol\lambda^\top+\Psi,\qquad \Psi=\mathrm{diag}(\psi_1,\dots,\psi_p)
$$

です。標本共分散行列 $S$ に $\Sigma(\boldsymbol\theta)$ を近づけるように $\boldsymbol\theta$ を推定します。多変量正規分布を仮定した最尤法では、次の乖離度を最小化します。

$$
F_{\mathrm{ML}}(\boldsymbol\theta)=\log|\Sigma(\boldsymbol\theta)|+\mathrm{tr}\bigl(S\,\Sigma(\boldsymbol\theta)^{-1}\bigr)-\log|S|-p
$$

**モデルの適合度検定**：$H_0$「$\Sigma=\Sigma(\boldsymbol\theta)$ の形をしている」のもとで

$$
\chi^2=(n-1)\,F_{\mathrm{ML}}(\hat{\boldsymbol\theta})\ \overset{\text{近似}}{\sim}\ \chi^2(\mathit{df}),\qquad \mathit{df}=\frac{p(p+1)}{2}-q
$$

です（$p$：観測変数の数、$q$：自由パラメータの数）。

## 数式の読み解き

- **何を当てはめているか**：個々のデータではなく、共分散行列の $p(p+1)/2$ 個の要素です。これが「共分散構造分析」と呼ばれる理由です。
- **$F_{\mathrm{ML}}$ の意味**：$\Sigma=S$ なら $\log|S|+p-\log|S|-p=0$ になり、ずれるほど大きくなります。$(n-1)F_{\mathrm{ML}}$ は、制約のない（飽和）モデルとの尤度比検定の統計量です。
- **識別性**：$q>p(p+1)/2$ ならパラメータは決まりません（必要条件、t 規則）。また潜在変数には単位がないので、分散を1に固定するか、負荷量の1つを1に固定して尺度を決めます。
- **自由度0のモデル**（丁度識別）は必ず完全に当てはまり、検定できません。自由度が正のモデルだけが「仮説として棄却されうる」モデルです。
- **適合度指標**：$\chi^2$ 検定は $n$ が大きいとわずかなずれでも棄却するので、RMSEA $=\sqrt{\max(\chi^2-\mathit{df},0)/\{\mathit{df}\,(n-1)\}}$（0.05以下でよい当てはまりの目安）、CFI、GFI なども併用します。

## 具体例

**例1：3指標の1因子モデル（丁度識別）**

標準化した3つの観測変数の相関が $r_{12}=0.48$, $r_{13}=0.42$, $r_{23}=0.56$ で、モデルを $x_k=\lambda_k\xi+\delta_k$（$\mathrm{Var}(\xi)=1$）とします。モデルからは $r_{12}=\lambda_1\lambda_2$, $r_{13}=\lambda_1\lambda_3$, $r_{23}=\lambda_2\lambda_3$ なので

$$
\lambda_1^2=\frac{r_{12}\,r_{13}}{r_{23}}=\frac{0.48\times0.42}{0.56}=0.36\ \Longrightarrow\ \lambda_1=0.6
$$

$\lambda_2=0.48/0.6=0.8$, $\lambda_3=0.42/0.6=0.7$、独自分散は $\psi_k=1-\lambda_k^2=0.64,\ 0.36,\ 0.51$ です。
データの要素数 $3\times4/2=6$、パラメータ数 $3+3=6$ で自由度は $0$。相関行列を完全に再現するので、モデルの良し悪しはこのデータからは検定できません。

**例2：4指標の1因子モデルの適合度（$n=200$）**

4つ目の変数を加え、相関行列が

$$
S=\begin{pmatrix}1&0.48&0.42&0.40\\0.48&1&0.56&0.40\\0.42&0.56&1&0.35\\0.40&0.40&0.35&1\end{pmatrix}
$$

だったとします。自由度は $10-8=2$ です。最尤法で推定すると

$$
\hat{\boldsymbol\lambda}=(0.630,\ 0.779,\ 0.695,\ 0.536),\qquad \hat{\boldsymbol\psi}=(0.603,\ 0.393,\ 0.516,\ 0.713)
$$

$F_{\mathrm{ML}}(\hat{\boldsymbol\theta})=0.01430$ で、$\chi^2=199\times0.01430=2.845$ です。自由度2の上側5%点 $5.991$ より小さく（$p=0.241$）、1因子モデルは棄却されません。RMSEA は $\sqrt{(2.845-2)/(2\times199)}=0.046$ です。
残差 $S-\Sigma(\hat{\boldsymbol\theta})$ で最も大きいのは $x_1$ と $x_4$ の相関の $0.062$ で、モデルが説明しきれていない部分がここにあるとわかります。

## 現場での使われ方

- **マーケティング**：「品質の評価」「価格の納得感」→「満足度」→「再購入意向」という仮説を、複数のアンケート項目を指標にした潜在変数で表し、どの経路の影響が大きいかを推定します。
- **心理学・教育**：テストや質問紙がねらいどおりの因子構造を持つか（構成概念妥当性）を、確認的因子分析で検証します。
- **医療・公衆衛生**：社会経済的な状況 → 生活習慣 → 健康状態のような経路を、測定誤差を考慮して分析します。
- **組織・人事**：職場環境の評価がエンゲージメントを通じて離職意向に影響するか、媒介の構造を調べます。

## グラフ・イメージ

例2の1因子モデルをパス図で描いたものです。潜在変数 ξ から各観測変数に向かう矢印が因子負荷量です。
SEM では通常、潜在変数を楕円、観測変数を四角で描き、誤差の矢印も描き込みます（この図では省略しています）。

## 試験のポイント

- 測定方程式と構造方程式の役割（潜在変数を測る部分／潜在変数間の因果）を区別します。
- 自由度 $=p(p+1)/2-q$ の計算は重要な基本です。丁度識別（自由度0）では適合度を検定できません。
- 1因子3指標の解 $\lambda_1^2=r_{12}r_{13}/r_{23}$ のように、モデルの共分散構造から連立方程式を立てて解く問題が典型です。
- $\chi^2=(n-1)F_{\mathrm{ML}}$ は飽和モデルとの尤度比検定で、「棄却されない」ことが当てはまりの良さを意味します（通常の検定と向きが逆）。
- 適合度指標は「小さいほどよい」RMSEA・SRMR と、「1に近いほどよい」CFI・TLI・GFI・AGFI に分かれます。CFI $=1-\max(\chi^2-\mathit{df},0)/\max(\chi^2_0-\mathit{df}_0,\ \chi^2-\mathit{df},\ 0)$ は、全変数を無相関とする独立モデル（$\chi^2_0$、$\mathit{df}_0=p(p-1)/2$）からの改善の割合です。例2では独立モデルが $\chi^2_0=187.105$（$\mathit{df}_0=6$）で、CFI $=1-0.845/181.105\approx0.995$ です。目安（RMSEA 0.05以下、CFI・TLI 0.95以上、SRMR 0.08以下など）は1つだけで判断せず併用します。
- 探索的因子分析は負荷量を自由に推定し回転で解釈するのに対し、SEM（確認的因子分析）は分析者が負荷量の0制約などの構造を事前に指定します。

## よくある間違い

- × 適合度検定で $p<0.05$ になったら「モデルが支持された」 → ○ 帰無仮説が「モデルが正しい」なので、棄却されないことが当てはまりの良さを示します。
- × 当てはまりが良ければ因果関係が証明された → ○ 同じ共分散行列を同じ程度に再現する別のモデル（同値モデル）が存在しえます。因果の向きは研究デザインや理論で裏付けます。
- × 自由度を「データの数 $n$ − パラメータ数」で計算する → ○ データの要素数は共分散行列の非重複要素数 $p(p+1)/2$ です。
- × 潜在変数の尺度を固定せずに推定する → ○ 分散を1に固定するか、負荷量の1つを1に固定しないと識別されません。
