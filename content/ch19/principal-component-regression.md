---
id: principal-component-regression
title: 主成分回帰
reading: しゅせいぶんかいき
en: Principal component regression
aliases: [PCR, 主成分得点による回帰]
chapter: 19
importance: C
tags: [多重共線性, 主成分分析, 次元削減, 縮小推定]
related: [multicollinearity, principal-component-analysis, principal-component-score, ridge-regression, contribution-ratio, cross-validation]
---

## 一言でいうと

説明変数を主成分分析で互いに無相関な主成分に置き換え、分散の大きい上位いくつかの主成分だけで回帰する方法です。
多重共線性の原因となる「ほとんど変動のない方向」を捨てることで、係数の推定を安定させます。

## 定義・数式

説明変数を中心化（ふつうは標準化）した $n\times p$ 行列 $X$ について、$X^\top X$ の固有値を $\lambda_1\ge\cdots\ge\lambda_p$、対応する正規直交な固有ベクトルを $\boldsymbol{v}_1,\dots,\boldsymbol{v}_p$ とします。第 $k$ 主成分得点は $\boldsymbol{z}_k=X\boldsymbol{v}_k$ です。

上位 $m$ 個（$m<p$）の主成分で中心化した $\boldsymbol{y}$ を回帰すると、主成分どうしが直交するので係数は1つずつ求まります。

$$
\hat{\gamma}_k=\frac{\boldsymbol{z}_k^\top\boldsymbol{y}}{\boldsymbol{z}_k^\top\boldsymbol{z}_k}=\frac{\boldsymbol{v}_k^\top X^\top\boldsymbol{y}}{\lambda_k}\qquad(k=1,\dots,m)
$$

元の説明変数の係数に戻すと

$$
\hat{\boldsymbol{\beta}}_{PCR}=\sum_{k=1}^m\hat{\gamma}_k\,\boldsymbol{v}_k
$$

です。$m=p$ なら最小二乗推定量に一致します。

## 数式の読み解き

- $\lambda_k=\|\boldsymbol{z}_k\|^2$ は第 $k$ 主成分方向のデータの広がりです。最小二乗推定量の分散は $\sigma^2\sum_k\boldsymbol{v}_k\boldsymbol{v}_k^\top/\lambda_k$ で、小さい $\lambda_k$ の方向が分散を大きくします。その方向を捨てるのが主成分回帰です。
- 捨てた方向の係数を0とみなすので、真の係数がその方向の成分を持てば**偏り**が生じます。偏りと引き換えに分散を減らす点はリッジ回帰と同じです。
- リッジ回帰は各方向を $\lambda_k/(\lambda_k+\lambda)$ 倍に「なめらかに」縮め、主成分回帰は「1か0か」で縮めます。

## 具体例

**相関0.9の2変数（リッジ回帰の例と同じデータ）**

$X^\top X=\begin{pmatrix}10&9\\9&10\end{pmatrix}$、$X^\top\boldsymbol{y}=(18.6,\ 18.0)$ です。固有値は $\lambda_1=19,\ \lambda_2=1$、固有ベクトルは $\boldsymbol{v}_1=(1,1)/\sqrt2$、$\boldsymbol{v}_2=(1,-1)/\sqrt2$ です。

第1主成分だけを使うと

$$
\hat{\gamma}_1=\frac{(18.6+18.0)/\sqrt2}{19}=1.3621,\qquad
\hat{\boldsymbol{\beta}}_{PCR}=1.3621\times\frac{1}{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}0.9632\\0.9632\end{pmatrix}
$$

最小二乗推定量は $(1.2632,\ 0.6632)$ でした。各係数の分散は、最小二乗法で $\sigma^2\times10/19=0.5263\sigma^2$、主成分回帰で $\sigma^2\times(1/2)/19=0.0263\sigma^2$ と20分の1になります。
代わりに、2つの係数は必ず等しいという制約がかかり、残差平方和は $0.0284$ から $0.2084$ に増えます（$\boldsymbol{y}$ の平方和は $35.46$）。

## 現場での使われ方

- **製造・品質管理**：近赤外スペクトルのように数百の強く相関する波長の吸光度から、成分濃度を予測する検量線を作ります。
- **金融・保険**：多数の金利（年限ごと）を少数の主成分（水準・傾き・曲率）に要約し、それを説明変数にします。

## 試験のポイント

- 主成分得点が互いに無相関（直交）なので、多重共線性が起きないことを押さえます。
- 捨てる主成分の数 $m$ が罰則の強さに相当し、交差検証などで選びます。
- リッジ回帰との対比（なめらかな縮小か、打ち切りか）が重要です。

## よくある間違い

- × 寄与率の大きい主成分ほど目的変数の予測に役立つ → ○ 主成分は $X$ だけから作るので、$\boldsymbol{y}$ との関係は保証されません。$\boldsymbol{y}$ も使って成分を作る方法が PLS 回帰です。
- × 主成分回帰の係数は不偏 → ○ 捨てた方向に真の係数の成分があれば偏ります。
- × 単位の違う変数をそのまま使う → ○ 主成分は尺度に依存するので、ふつうは標準化してから行います。
