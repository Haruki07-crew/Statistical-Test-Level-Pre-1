---
id: maximum-likelihood-estimator
title: 最尤推定量
reading: さいゆうすいていりょう
en: Maximum likelihood estimator
aliases: [MLE, 最尤法, 最尤推定, 最尤推定値, 尤度方程式]
chapter: 8
importance: A
tags: [推定, 最尤法, 尤度, 尤度方程式]
related: [likelihood-function, score-function, asymptotic-normality-of-mle, invariance-of-mle, method-of-moments, em-algorithm]
charts:
  - type: custom
    id: line-data
    params:
      x: [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6]
      xlabel: "ポアソン分布の母数 λ"
      ylabel: "ℓ(λ) − ℓ(3)"
      series:
        - { label: "対数尤度（データ 2,4,3,1,5）", y: [-6.479, -4.382, -2.897, -1.835, -1.082, -0.565, -0.235, -0.055, 0, -0.049, -0.188, -0.403, -0.685, -1.025, -1.418, -1.857, -2.338, -2.856, -3.408, -3.991, -4.603] }
    caption: 来店客数 2,4,3,1,5 に対するポアソン分布の対数尤度（最大値を0にそろえた）。頂上の λ=3 が最尤推定値で、標本平均と一致する。右側より左側のほうが急に下がる、少し非対称な山である。
---

## 一言でいうと

「観測されたデータが最も起こりやすくなるような母数の値」を推定値とする方法（最尤法）で得られる推定量です。
多くの分布で計算が機械的にでき、大標本では最も精度が高い（漸近有効）ため、推定法の標準になっています。

## 定義・数式

尤度関数 $L(\theta)=\prod_{i=1}^n f(x_i;\theta)$ を最大にする $\theta$

$$
\hat\theta = \operatorname*{arg\,max}_{\theta\in\Theta} L(\theta) = \operatorname*{arg\,max}_{\theta\in\Theta} \ell(\theta)
$$

を**最尤推定値**、それを確率変数 $X_i$ の関数とみたものを**最尤推定量**（MLE）といいます。

$\ell(\theta)$ が微分可能で、最大値が母数空間の内部でとられるときは、**尤度方程式**

$$
\frac{\partial \ell(\theta)}{\partial\theta} = 0
$$

の解として求めます。母数が複数あるときは、各母数で偏微分した連立方程式を解きます。

**主な結果**

| 分布 | 母数 | 最尤推定量 |
|---|---|---|
| ベルヌーイ $Bern(p)$ | $p$ | $\hat p=\bar X$ |
| 二項 $Bin(m,p)$（$m$ 既知） | $p$ | $\hat p=\bar X/m$ |
| ポアソン $Po(\lambda)$ | $\lambda$ | $\hat\lambda=\bar X$ |
| 正規 $N(\mu,\sigma^2)$ | $\mu,\ \sigma^2$ | $\hat\mu=\bar X,\ \hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$ |
| 指数（密度 $\lambda e^{-\lambda x}$） | $\lambda$ | $\hat\lambda=1/\bar X$ |
| 一様 $U(0,\theta)$ | $\theta$ | $\hat\theta=X_{(n)}=\max_i X_i$ |

## 数式の読み解き

手順は「対数尤度を書く → 微分して 0 とおく → 解く → 最大であることを確認する」です。以下、途中式を省略せずに示します。

**ベルヌーイ分布**：$f(x;p)=p^x(1-p)^{1-x}$、$t=\sum x_i$ とおくと

$$
\ell(p) = t\log p + (n-t)\log(1-p),\qquad
\ell'(p) = \frac{t}{p} - \frac{n-t}{1-p}
$$

$\ell'(p)=0$ から $t(1-p)=(n-t)p$、すなわち $t = np$ で $\hat p = t/n=\bar x$。
$\ell''(p)=-t/p^2-(n-t)/(1-p)^2<0$ なので最大です（$t=0$ や $t=n$ のときは端点 $\hat p=0,1$）。

**ポアソン分布**：$f(x;\lambda)=e^{-\lambda}\lambda^x/x!$ より

$$
\ell(\lambda) = -n\lambda + \Big(\sum x_i\Big)\log\lambda - \sum\log x_i!,\qquad
\ell'(\lambda) = -n + \frac{\sum x_i}{\lambda}
$$

$\ell'(\lambda)=0$ から $\hat\lambda=\sum x_i/n=\bar x$。$\ell''(\lambda)=-\sum x_i/\lambda^2<0$ です。

**正規分布（$\mu,\sigma^2$ ともに未知）**：$v=\sigma^2$ とおくと

$$
\ell(\mu,v) = -\frac n2\log(2\pi) - \frac n2\log v - \frac{1}{2v}\sum(x_i-\mu)^2
$$

$$
\frac{\partial\ell}{\partial\mu} = \frac1v\sum(x_i-\mu) = 0 \ \Rightarrow\ \hat\mu = \bar x
$$

$$
\frac{\partial\ell}{\partial v} = -\frac{n}{2v} + \frac{1}{2v^2}\sum(x_i-\mu)^2 = 0 \ \Rightarrow\ \hat v = \frac1n\sum(x_i-\hat\mu)^2
$$

$\sigma^2$ の最尤推定量は $n$ で割った標本分散で、不偏分散ではありません（$E[\hat\sigma^2]=\frac{n-1}{n}\sigma^2$）。

**指数分布**：$f(x;\lambda)=\lambda e^{-\lambda x}\ (x>0)$ より

$$
\ell(\lambda) = n\log\lambda - \lambda\sum x_i,\qquad \ell'(\lambda) = \frac n\lambda - \sum x_i = 0 \ \Rightarrow\ \hat\lambda = \frac{n}{\sum x_i} = \frac{1}{\bar x}
$$

平均 $\theta=1/\lambda$ を母数にとる流儀（密度 $\frac1\theta e^{-x/\theta}$）なら $\hat\theta=\bar x$ です。どちらで書いても不変性により整合します。

**一様分布 $U(0,\theta)$（微分では解けない例）**：

$$
L(\theta) = \prod_{i=1}^n \frac1\theta\,\mathbf 1\{0\le x_i\le\theta\} = \frac{1}{\theta^n}\,\mathbf 1\{\theta\ge x_{(n)}\}
$$

$\theta<x_{(n)}$ では $L=0$、$\theta\ge x_{(n)}$ では $\theta^{-n}$ で単調減少です。よって最大は境界 $\hat\theta=x_{(n)}$ です。台が母数に依存するので、尤度方程式は使えません。

## 具体例

**例1：来店客数（ポアソン）**

2, 4, 3, 1, 5 人のデータでは $\hat\lambda=15/5=3$ です。尤度方程式で確かめると $\ell'(3)=-5+15/3=0$ です。

**例2：部品の長さ（正規）**

測定値 4.8, 5.6, 5.1, 4.5, 5.5（mm）で、$\hat\mu=25.5/5=5.1$。偏差は $-0.3, 0.5, 0, -0.6, 0.4$、偏差平方和は $0.09+0.25+0+0.36+0.16=0.86$ なので

$$
\hat\sigma^2 = \frac{0.86}{5} = 0.172,\qquad \hat\sigma = \sqrt{0.172}\approx0.4147
$$

（不偏分散なら $0.86/4=0.215$ です。）

**例3：待ち時間（指数）と一様分布**

- 窓口の待ち時間 2.5, 0.8, 4.1, 1.6, 3.0, 0.6（分）：$\bar x=12.6/6=2.1$、$\hat\lambda=1/2.1\approx0.4762$（1分あたり）。
- $U(0,\theta)$ からの 2.1, 7.4, 5.0, 3.3, 9.2：最尤推定値は $\hat\theta=9.2$。モーメント法なら $2\bar x=10.8$ で、答えが異なります。

## 現場での使われ方

- **製造・品質管理**：部品の寿命データにワイブル分布や指数分布を当てはめ、最尤法で母数を推定して保証期間を決めます。打ち切りのあるデータも尤度に組み込めます。
- **医療**：ロジスティック回帰や比例ハザードモデルの係数は最尤法（部分尤度を含む）で推定し、オッズ比やハザード比として報告します。
- **IT・Webサービス**：クリック率予測などの確率モデルの学習は、対数尤度の最大化（交差エントロピー最小化）そのものです。
- **金融・保険**：損害額や金利変動のモデル（GARCH など）の母数推定に使われます。

## グラフ・イメージ

ポアソン分布の対数尤度を $\lambda$ の関数として描いています。頂上の位置 $\lambda=3$ が最尤推定値で、標本平均と一致します。
頂上での接線の傾き（スコア）は 0 で、頂上付近の曲がり具合が推定の精度を表します。

## 試験のポイント

- 上の表の最尤推定量は、すべて自分で導出できるようにします。とくに正規分布の $\hat\sigma^2$ が $n$ で割ること、一様分布が境界で最大になることは頻出の論点です。
- 尤度方程式の解が最大であることを、2階微分の符号で確認します。
- 最尤推定量の関数 $g(\hat\theta)$ は $g(\theta)$ の最尤推定量です（不変性）。
- 大標本では $\hat\theta\approx N\big(\theta,\,1/\{nI(\theta)\}\big)$（漸近正規性）で、ワルド型信頼区間や検定に使います。
- 混合分布や欠測があると尤度方程式が解析的に解けず、EM アルゴリズムやニュートン法を使います。

## よくある間違い

- × 最尤推定量は不偏 → ○ $\hat\sigma^2$（$n$ で割る）、$1/\bar X$、$X_{(n)}$ はいずれも偏りがあります。一致性や漸近有効性はもちますが、不偏とは限りません。
- × 尤度方程式の解が必ず最尤推定値 → ○ 台が母数に依存する場合（一様分布）や、最大が境界にある場合は、尤度の形を直接調べます。
- × 正規分布の $\sigma$ の最尤推定量は不偏分散の平方根 → ○ $\hat\sigma=\sqrt{\frac1n\sum(X_i-\bar X)^2}$ です（不変性より）。
