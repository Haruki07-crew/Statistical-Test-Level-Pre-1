---
id: elastic-net
title: エラスティックネット
reading: えらすてぃっくねっと
en: Elastic net
aliases: [エラスティックネット回帰, Elastic Net]
chapter: 19
importance: C
tags: [正則化, 罰則付き推定, 変数選択]
related: [lasso, ridge-regression, regularization, variable-selection, cross-validation]
charts:
  - type: custom
    id: ridge-lasso-path
    params: { method: lasso }
    caption: 比較のための Lasso の解の経路。エラスティックネットは、この「0に張りつく」性質を保ちながら、リッジの罰則で係数の暴れを抑える。
---

## 一言でいうと

Lasso の罰則（係数の絶対値の和）とリッジの罰則（係数の二乗和）を両方足して推定する回帰です。
Lasso の「変数を選ぶ」性質と、リッジの「相関の強い変数を安定して扱う」性質をあわせ持ちます。

## 定義・数式

説明変数を標準化し、目的変数を中心化した線形モデルで、**エラスティックネット推定量**は

$$
\hat{\boldsymbol{\beta}}_{EN}=\arg\min_{\boldsymbol{\beta}}\left\{\frac{1}{2}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2+\lambda_1\sum_{j=1}^p|\beta_j|+\frac{\lambda_2}{2}\sum_{j=1}^p\beta_j^2\right\}\qquad(\lambda_1,\lambda_2\ge0)
$$

です（切片は罰則に含めない）。$\lambda_2=0$ なら Lasso、$\lambda_1=0$ ならリッジ回帰になります。

ソフトウェアでは、全体の強さ $\lambda$ と混合比 $\alpha\in[0,1]$ を使って罰則を $\lambda\{\alpha\|\boldsymbol{\beta}\|_1+\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|^2\}$ と書く流儀が一般的です（$\alpha=1$ で Lasso、$\alpha=0$ でリッジ）。

直交計画 $X^\top X=I$ では、OLS 推定値 $b_j$ を使って

$$
\hat\beta_{EN,j}=\frac{\mathrm{sign}(b_j)\,(|b_j|-\lambda_1)_+}{1+\lambda_2}
$$

と書けます。

## 数式の読み解き

- 分子は Lasso のソフト閾値（小さい係数を0にする）、分母はリッジの縮小（全体を同じ割合で縮める）です。直交計画では「Lasso の後にリッジ」をかけた形になります。
- **グループ効果**：Lasso は相関の強い変数の組から片方だけを残しがちです。二乗の罰則を加えると目的関数が狭義凸になり、相関の強い変数どうしは似た係数を持つようになります。
- $p>n$ のとき、Lasso は高々 $n$ 個の変数しか選べませんが、エラスティックネットにはこの制限がありません。

## 具体例

**相関0.9の2変数（リッジ回帰・Lasso の例と同じデータ）**

$X^\top X=\begin{pmatrix}10&9\\9&10\end{pmatrix}$、$X^\top\boldsymbol{y}=(18.6,\ 18.0)$、OLS は $(1.2632,\ 0.6632)$ です。

両方の係数が正のまま残る範囲では、最適性の条件は $(X^\top X+\lambda_2I)\boldsymbol{\beta}=X^\top\boldsymbol{y}-\lambda_1(1,1)^\top$ です。$\lambda_1=10,\ \lambda_2=5$ とすると

$$
\begin{pmatrix}15&9\\9&15\end{pmatrix}\boldsymbol{\beta}=\begin{pmatrix}8.6\\8.0\end{pmatrix},\qquad
\hat{\boldsymbol{\beta}}_{EN}=\frac{1}{144}\begin{pmatrix}15\times8.6-9\times8.0\\ 15\times8.0-9\times8.6\end{pmatrix}=\begin{pmatrix}0.3958\\0.2958\end{pmatrix}
$$

同じ $\lambda_1=10$ の Lasso は $(0.7368,\ 0.1368)$ で、係数の比は約5.4倍でした。エラスティックネットでは比が約1.34倍に縮まり、2つの変数が似た重みで残ります（グループ効果）。

## 現場での使われ方

- **医療**：同じ経路で働く遺伝子のように強く相関するマーカー群を、まとめて選び出すのに使います。
- **マーケティング**：互いに似た多数の行動指標から予測モデルを作るとき、交差検証で $\lambda$ と $\alpha$ を選びます。

## グラフ・イメージ

図は比較のための Lasso の解の経路です。エラスティックネットでも係数は0に張りつきますが、リッジの罰則が加わる分、経路はなめらかになり、相関の強い変数どうしは近い値で推移します。

## 試験のポイント

- 罰則が L1 と L2 の和であること、$\lambda_1$ や $\lambda_2$ を0にすると Lasso・リッジになることを押さえます。
- Lasso の弱点（相関の強い変数から1つだけ選ぶ、$p>n$ で高々 $n$ 個）を補う、という位置づけが重要です。

## よくある間違い

- × エラスティックネットでは係数が0にならない → ○ L1 の罰則があるので、ちょうど0になる係数が出ます。
- × 罰則の書き方はどのソフトでも同じ → ○ $\frac12$ や $\frac1n$ の付け方、$\alpha$ による書き方など流儀が分かれ、同じ解に対応する $\lambda$ の値が変わります。
