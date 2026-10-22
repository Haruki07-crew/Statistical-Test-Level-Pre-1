---
id: factor-score
title: 因子得点
reading: いんしとくてん
en: Factor score
aliases: [因子スコア, 回帰法（因子得点）, バートレット法]
chapter: 25
importance: B
tags: [因子分析, 推定, 回帰]
related: [factor-analysis-model, factor-loading, common-factor-unique-factor, principal-component-score, conditional-expectation, weighted-least-squares]
charts:
  - type: custom
    id: line-data
    params:
      x: [1, 2, 3, 4]
      series:
        - { label: "回帰法の重み", y: [0.5531, 0.2595, 0.1603, 0.1095] }
        - { label: "バートレット法の重み", y: [0.6262, 0.2938, 0.1815, 0.1239] }
      xlabel: 変数の番号（負荷量 0.9, 0.8, 0.7, 0.6）
      ylabel: 因子得点を作る重み
    caption: 例の1因子モデルで、各変数に掛ける重み。どちらも負荷量が大きく独自分散が小さい変数ほど重く、回帰法はバートレット法を一律に 0.883 倍に縮めた形になっている。
---

## 一言でいうと

観測できない共通因子の値を、個人（個体）ごとに観測変数から推定した値です。
因子分析で見つけた「文系の力」などを、一人ひとりの点数として求め、後の分析に使います。

## 定義・数式

因子分析モデル $\boldsymbol{x}=\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon}$（$V[\boldsymbol{f}]=I_m$、$V[\boldsymbol{\varepsilon}]=\Psi$、$\Sigma=\Lambda\Lambda^\top+\Psi$）で、$\Lambda,\Psi$ を推定値で置き換えて使います。代表的な推定法は2つです。

**回帰法（トムソン法）**：$\boldsymbol{f}$ を $\boldsymbol{x}$ で線形予測する（多変量正規なら条件付き期待値）。

$$
\hat{\boldsymbol{f}}_{\mathrm{R}}=\Lambda^\top\Sigma^{-1}\boldsymbol{x}=(I_m+\Lambda^\top\Psi^{-1}\Lambda)^{-1}\Lambda^\top\Psi^{-1}\boldsymbol{x}
$$

**バートレット法**：$\boldsymbol{f}$ を固定した未知数とみなし、誤差の分散 $\Psi$ で重みづけた最小二乗法（加重最小二乗法）で推定する。

$$
\hat{\boldsymbol{f}}_{\mathrm{B}}=(\Lambda^\top\Psi^{-1}\Lambda)^{-1}\Lambda^\top\Psi^{-1}\boldsymbol{x}
$$

## 数式の読み解き

- **回帰法の導出**：$\mathrm{Cov}[\boldsymbol{f},\boldsymbol{x}]=E[\boldsymbol{f}(\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon})^\top]=\Lambda^\top$ なので、$\boldsymbol{f}$ の $\boldsymbol{x}$ への最良線形予測は $\mathrm{Cov}[\boldsymbol{f},\boldsymbol{x}]V[\boldsymbol{x}]^{-1}\boldsymbol{x}=\Lambda^\top\Sigma^{-1}\boldsymbol{x}$ です。2つ目の形は、逆行列の公式（ウッドベリーの公式）$\Sigma^{-1}=\Psi^{-1}-\Psi^{-1}\Lambda(I+\Lambda^\top\Psi^{-1}\Lambda)^{-1}\Lambda^\top\Psi^{-1}$ から出ます。
- **バートレット法の導出**：$\boldsymbol{f}$ を回帰係数とみなすと、$\boldsymbol{x}=\Lambda\boldsymbol{f}+\boldsymbol{\varepsilon}$ は「計画行列 $\Lambda$、誤差分散 $\Psi$」の回帰モデルです。$(\boldsymbol{x}-\Lambda\boldsymbol{f})^\top\Psi^{-1}(\boldsymbol{x}-\Lambda\boldsymbol{f})$ を最小にすると上の式になります。
- **性質の違い**：バートレット法は $E[\hat{\boldsymbol{f}}_{\mathrm{B}}\mid\boldsymbol{f}]=\boldsymbol{f}$（条件付き不偏）です。回帰法は0の方向に縮む（偏りがある）代わりに、予測の平均二乗誤差が最小です。
- 1因子では2つは比例し、$\hat f_{\mathrm{R}}=\dfrac{s}{1+s}\hat f_{\mathrm{B}}$（$s=\boldsymbol{\lambda}^\top\Psi^{-1}\boldsymbol{\lambda}$）です。
- 重み $\lambda_j/\psi_j$ に比例するので、負荷量が大きく独自分散が小さい変数ほど重く使われます。単純な合計点（重み一律）との違いはここです。

## 具体例

**例：1因子モデルで1人の因子得点を求める**

標準化した4変数に、負荷量 $\boldsymbol{\lambda}=(0.9,0.8,0.7,0.6)^\top$、独自分散 $\Psi=\mathrm{diag}(0.19,0.36,0.51,0.64)$ の1因子モデルが当てはまっているとします。
ある人の標準化得点が $\boldsymbol{x}=(1.0,\ 0.5,\ 0.8,\ -0.2)^\top$ でした。

まず $\lambda_j/\psi_j=(4.7368,\ 2.2222,\ 1.3725,\ 0.9375)$ で、

$$
s=\boldsymbol{\lambda}^\top\Psi^{-1}\boldsymbol{\lambda}=0.9\times4.7368+0.8\times2.2222+0.7\times1.3725+0.6\times0.9375=7.5642
$$

です。バートレット法の重みは $\lambda_j/\psi_j$ を $s$ で割った $(0.6262,\ 0.2938,\ 0.1815,\ 0.1239)$ で、

$$
\hat f_{\mathrm{B}}=0.6262\times1.0+0.2938\times0.5+0.1815\times0.8+0.1239\times(-0.2)=0.893
$$

回帰法の重みは $\lambda_j/\psi_j$ を $1+s=8.5642$ で割った $(0.5531,\ 0.2595,\ 0.1603,\ 0.1095)$ で、

$$
\hat f_{\mathrm{R}}=0.5531+0.1298+0.1282-0.0219=0.789
$$

です（numpy で $\Lambda^\top R^{-1}\boldsymbol{x}$ を直接計算しても 0.789）。比は $s/(1+s)=0.883$ で、回帰法のほうが0に縮んでいます。
回帰法の得点の分散は 0.883（1より小さい）、バートレット法の推定誤差の分散は $1/s=0.132$ です。

## 現場での使われ方

- **マーケティング**：顧客ごとの因子得点（「価格重視度」「品質重視度」など）を求め、それをもとに顧客をクラスター分析でグループ分けします。
- **教育**：複数の問題から「読解力」の因子得点を出し、合計点よりも測定誤差の少ない能力の推定値として使います。
- **行政・調査**：地域ごとの多数の指標から「生活利便性」などの因子得点を作り、地域の比較や施策の評価に使います。

## グラフ・イメージ

例の1因子モデルで、因子得点を作るための各変数の重みを2つの方法で比べた図です。
どちらも負荷量の大きい変数1に最も重く、回帰法の折れ線はバートレット法を一律に縮めた形になっています。

## 試験のポイント

- 回帰法 $\Lambda^\top\Sigma^{-1}\boldsymbol{x}$ とバートレット法 $(\Lambda^\top\Psi^{-1}\Lambda)^{-1}\Lambda^\top\Psi^{-1}\boldsymbol{x}$ の式と、導き方（最良線形予測／加重最小二乗）を区別して覚えます。
- バートレット法は条件付き不偏、回帰法は偏りがあるが平均二乗誤差が小さい、という性質の違いが問われます。
- 因子得点は「推定値」であり、主成分得点のように観測値の厳密な線形変換としては決まらない（因子得点の不定性）ことを押さえます。

## よくある間違い

- × 因子得点は主成分得点と同じように一意に計算できる → ○ 共通因子は観測できないので、推定法によって値が変わります。
- × 回帰法の因子得点の分散は1 → ○ $\Lambda^\top\Sigma^{-1}\Lambda$ の対角で、1より小さくなります（例では 0.883）。
- × 因子得点は負荷量をそのまま重みにした和 → ○ 重みは $\Psi^{-1}\Lambda$ に比例し、独自分散の小さい変数ほど重くなります。
