---
id: observed-fisher-information
title: 観測情報量
reading: かんそくじょうほうりょう
en: Observed Fisher information
aliases: [観測フィッシャー情報量, 観測情報行列]
chapter: 8
importance: C
tags: [推定, 対数尤度, 標準誤差]
related: [fisher-information, asymptotic-normality-of-mle, likelihood-function, score-function, wald-confidence-interval]
---

## 一言でいうと

手元のデータで計算した対数尤度の「曲がり具合（2階微分のマイナス）」です。
期待値をとらずにそのまま使うので、最尤推定量の標準誤差を、データから直接見積もれます。

## 定義・数式

データ $x_1,\dots,x_n$ の対数尤度を $\ell(\theta)=\sum_{i=1}^n\log f(x_i;\theta)$ とします。**観測情報量**は

$$
J(\theta) = -\frac{\partial^2}{\partial\theta^2}\,\ell(\theta)
$$

で、ふつうは最尤推定値での値 $J(\hat\theta)$ を使います。母数がベクトルのときは、2階偏微分を並べた行列（ヘッセ行列）にマイナスを付けたものです。

フィッシャー情報量（期待情報量）とは、期待値をとるかどうかが違います。

$$
E_\theta[J(\theta)] = nI(\theta)
$$

最尤推定量の標準誤差は、どちらを使っても漸近的に同じで、

$$
\mathrm{SE}(\hat\theta) \approx \frac{1}{\sqrt{J(\hat\theta)}} \quad\text{または}\quad \frac{1}{\sqrt{nI(\hat\theta)}}
$$

です。

## 数式の読み解き

- $\ell(\theta)$ が山の形なら、$J(\hat\theta)$ は山頂のとがり具合です。とがっているほど $\theta$ が絞り込まれ、標準誤差は小さくなります。
- $J(\theta)$ はデータの関数（確率変数）、$nI(\theta)$ はその期待値（定数）です。
- 指数型分布族（ポアソン、指数、二項など）では、最尤推定値で評価すると $J(\hat\theta)=nI(\hat\theta)$ と一致します。違いが出るのはそれ以外の分布です。
- 期待値の計算が難しいモデルでも、数値微分で $J(\hat\theta)$ は求められます。統計ソフトの多くはこちらを使います。

## 具体例

**コーシー分布の位置母数**

尺度1のコーシー分布 $f(x;\theta)=\dfrac{1}{\pi\{1+(x-\theta)^2\}}$ から、データ $-0.4,\ 0.8,\ 1.2,\ 2.1,\ 7.5$ を得たとします。$u_i=x_i-\theta$ とおくと

$$
J(\theta) = \sum_{i=1}^{5}\frac{2(1-u_i^2)}{(1+u_i^2)^2}
$$

です。数値計算で最尤推定値は $\hat\theta\approx 1.109$（外れた 7.5 にほとんど引きずられません）。このとき

- 観測情報量：$J(\hat\theta)\approx 3.184$、標準誤差 $1/\sqrt{3.184}\approx 0.560$
- 期待情報量：1個あたり $I=1/2$ なので $nI=2.5$、標準誤差 $1/\sqrt{2.5}\approx 0.632$

データが $\hat\theta$ の近くに集まっているぶん、観測情報量のほうが大きく、標準誤差が小さめに出ます。

## 現場での使われ方

- **IT・Webサービス**：ロジスティック回帰などのソフトウェアは、最適化の最後に得られるヘッセ行列から係数の標準誤差を出力します。これは観測情報行列の逆行列です。
- **医療**：生存時間モデルなど期待値の計算が面倒なモデルで、推定値の信頼区間をデータから直接作ります。

## 試験のポイント

- 定義 $J(\theta)=-\ell''(\theta)$ と、期待値をとると $nI(\theta)$ になることを押さえます。
- ワルド型信頼区間 $\hat\theta\pm z_{\alpha/2}/\sqrt{J(\hat\theta)}$ の形で使われます。
- ポアソン分布では $J(\hat\lambda)=\sum x_i/\hat\lambda^2=n/\bar x$ で、$nI(\hat\lambda)$ と一致します。

## よくある間違い

- × 観測情報量は1個あたりの量 → ○ データ全体の対数尤度から計算するので、$nI(\theta)$ と比べます（$I(\theta)$ ではありません）。
- × 2階微分そのものを情報量とする → ○ 最大点では2階微分は負なので、マイナスを付けて正の量にします。
