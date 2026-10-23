---
id: fisher-information
title: フィッシャー情報量
reading: ふぃっしゃーじょうほうりょう
en: Fisher information
aliases: [フィッシャー情報行列, 情報量, "I(θ)"]
chapter: 8
importance: A
tags: [尤度, スコア関数, クラメール・ラオの下限, 漸近分散]
related: [score-function, cramer-rao-inequality, asymptotic-normality-of-mle, efficient-estimator, observed-fisher-information, jeffreys-prior]
charts:
  - type: custom
    id: line-data
    params:
      x: [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]
      xlabel: "成功確率 p"
      ylabel: "ℓ(p) − ℓ(0.3)"
      series:
        - { label: "n=10（3回成功）", y: [-3.238, -2.208, -1.537, -1.064, -0.72, -0.467, -0.282, -0.151, -0.064, -0.015, 0, -0.014, -0.056, -0.124, -0.216, -0.332, -0.472, -0.635, -0.823, -1.036, -1.274, -1.541, -1.838, -2.167, -2.532] }
        - { label: "n=40（12回成功）", y: [-12.95, -8.832, -6.147, -4.258, -2.881, -1.867, -1.127, -0.602, -0.256, -0.062, 0, -0.058, -0.225, -0.495, -0.864, -1.328, -1.887, -2.541, -3.291, -4.142, -5.098, -6.165, -7.351, -8.669, -10.13] }
    caption: 成功割合が同じ 0.3 でも、n=40 の対数尤度は n=10 の4倍の鋭さで曲がる。曲がり具合（2階微分の符号を変えたもの）が情報量で、標本サイズに比例して増える。
  - type: custom
    id: line-data
    params:
      x: [0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.94, 0.98]
      xlabel: "成功確率 p"
      ylabel: "I(p)"
      series:
        - { label: "ベルヌーイ分布の情報量 1/{p(1−p)}", y: [51.02, 17.73, 11.111, 8.306, 6.775, 5.828, 5.198, 4.762, 4.456, 4.244, 4.105, 4.026, 4, 4.026, 4.105, 4.244, 4.456, 4.762, 5.198, 5.828, 6.775, 8.306, 11.111, 17.73, 51.02] }
    caption: ベルヌーイ分布1回あたりの情報量。p=0.5 で最小の 4 になり、0 や 1 に近いほど大きい。p が端に近いと標本比率の分散 p(1−p)/n が小さくなることに対応する。
---

## 一言でいうと

データが未知の母数についてどれだけの情報を含んでいるかを表す量です。
対数尤度の山が鋭いほど（母数を少し動かすだけで尤度が大きく下がるほど）情報量は大きく、母数を精度よく推定できます。

## 定義・数式

1個の観測 $X\sim f(x;\theta)$ の**フィッシャー情報量**を、スコア $s(\theta;X)=\frac{\partial}{\partial\theta}\log f(X;\theta)$ を使って

$$
I(\theta) = E_\theta\Big[\Big(\frac{\partial}{\partial\theta}\log f(X;\theta)\Big)^2\Big]
$$

と定めます。正則条件（台が $\theta$ に依存しない、微分と積分の順序が交換できる、など）のもとで、次の2つの式は等しくなります。

$$
I(\theta) = V_\theta[s(\theta;X)] = -E_\theta\Big[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\Big]
$$

独立同分布な $n$ 個の標本全体の情報量は、1個あたりの $n$ 倍です（**加法性**）。

$$
I_n(\theta) = nI(\theta)
$$

母数が $\boldsymbol\theta=(\theta_1,\dots,\theta_k)$ のときは**フィッシャー情報行列** $\big[I(\boldsymbol\theta)\big]_{jk} = -E\big[\partial^2\log f/\partial\theta_j\partial\theta_k\big]$ を使います。

**主な分布の情報量（1個あたり）**

| 分布 | 母数 | $I(\theta)$ | CR下限 $1/\{nI\}$ |
|---|---|---|---|
| ベルヌーイ $Bern(p)$ | $p$ | $\dfrac{1}{p(1-p)}$ | $\dfrac{p(1-p)}{n}$ |
| 二項 $Bin(m,p)$ | $p$ | $\dfrac{m}{p(1-p)}$ | $\dfrac{p(1-p)}{mn}$ |
| ポアソン $Po(\lambda)$ | $\lambda$ | $\dfrac1\lambda$ | $\dfrac\lambda n$ |
| 正規（$\sigma^2$ 既知） | $\mu$ | $\dfrac{1}{\sigma^2}$ | $\dfrac{\sigma^2}{n}$ |
| 正規（$\mu$ 既知） | $\sigma^2$ | $\dfrac{1}{2\sigma^4}$ | $\dfrac{2\sigma^4}{n}$ |
| 指数（密度 $\lambda e^{-\lambda x}$） | $\lambda$ | $\dfrac{1}{\lambda^2}$ | $\dfrac{\lambda^2}{n}$ |

## 数式の読み解き

- 「スコアの分散」の見方：真の $\theta$ のまわりでスコアが大きく揺れるほど、データは $\theta$ の違いに敏感です。
- 「2階微分」の見方：$-\ell''(\theta)$ は対数尤度の山の曲率です。曲率が大きいほど山が鋭く、最尤推定値の位置がはっきり決まります。
- 2つの定義が一致する理由は、スコア関数の項目にある「$\int s f\,dx=0$ をもう一度微分する」導出です。

以下、2階微分の形で途中式を省略せずに計算します。

**ベルヌーイ分布**：$\log f = x\log p+(1-x)\log(1-p)$

$$
\frac{\partial}{\partial p}\log f = \frac xp - \frac{1-x}{1-p},\qquad
\frac{\partial^2}{\partial p^2}\log f = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
$$

$E[X]=p$ を代入して符号を変えると

$$
I(p) = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac1p + \frac{1}{1-p} = \frac{1}{p(1-p)}
$$

**ポアソン分布**：$\log f = x\log\lambda - \lambda - \log x!$

$$
\frac{\partial}{\partial\lambda}\log f = \frac{x}{\lambda} - 1,\qquad \frac{\partial^2}{\partial\lambda^2}\log f = -\frac{x}{\lambda^2},\qquad I(\lambda) = \frac{E[X]}{\lambda^2} = \frac{\lambda}{\lambda^2} = \frac1\lambda
$$

**正規分布の $\mu$**（$\sigma^2$ 既知）：$\log f = -\frac12\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$

$$
\frac{\partial}{\partial\mu}\log f = \frac{x-\mu}{\sigma^2},\qquad \frac{\partial^2}{\partial\mu^2}\log f = -\frac{1}{\sigma^2},\qquad I(\mu)=\frac1{\sigma^2}
$$

**正規分布の $v=\sigma^2$**（$\mu$ 既知）：$\log f = -\frac12\log(2\pi) - \frac12\log v - \frac{(x-\mu)^2}{2v}$

$$
\frac{\partial}{\partial v}\log f = -\frac{1}{2v} + \frac{(x-\mu)^2}{2v^2},\qquad
\frac{\partial^2}{\partial v^2}\log f = \frac{1}{2v^2} - \frac{(x-\mu)^2}{v^3}
$$

$E[(X-\mu)^2]=v$ を代入すると $\frac{1}{2v^2}-\frac{v}{v^3} = -\frac{1}{2v^2}$ なので、$I(\sigma^2) = \dfrac{1}{2\sigma^4}$ です。
（$\mu,\sigma^2$ がともに未知でも、情報行列は対角 $\mathrm{diag}(1/\sigma^2,\ 1/(2\sigma^4))$ になります。交差項 $E[-(X-\mu)/v^2]=0$ だからです。）

**指数分布**：$\log f = \log\lambda - \lambda x$

$$
\frac{\partial}{\partial\lambda}\log f = \frac1\lambda - x,\qquad \frac{\partial^2}{\partial\lambda^2}\log f = -\frac{1}{\lambda^2},\qquad I(\lambda) = \frac{1}{\lambda^2}
$$

**母数の取り換え**：$\theta=\theta(\eta)$ と書き換えると $I(\eta) = I(\theta)\left(\frac{d\theta}{d\eta}\right)^2$ です。
たとえば指数分布を平均 $\theta=1/\lambda$ で表すと、$d\lambda/d\theta=-1/\theta^2$ より $I(\theta) = \lambda^2\cdot\frac{1}{\theta^4} = \frac{1}{\theta^2}$ です。

## 具体例

**例1：支持率調査（ベルヌーイ）**

真の支持率 $p=0.3$、$n=100$ 人に聞くとき、標本全体の情報量は

$$
nI(p) = \frac{100}{0.3\times0.7} = \frac{100}{0.21} \approx 476.19
$$

不偏推定量の分散の下限は $1/476.19 = 0.0021$、標準誤差にすると $\sqrt{0.0021}\approx0.0458$ です。標本比率の分散 $p(1-p)/n=0.0021$ はこの下限に一致します。

**例2：ポアソンと指数**

- 1日あたりの問い合わせ件数 $Po(\lambda)$、$\lambda=4$ を $n=20$ 日観測：$nI=20/4=5$、下限 $1/5=0.2$（標準誤差 $\approx0.4472$）。
- 部品の故障までの時間 $Ex(\lambda)$、$\lambda=0.5$ を $n=50$ 個観測：$nI = 50/0.25 = 200$、下限 $0.005$（標準誤差 $\approx0.0707$）。

**例3：正規分布の母分散**

$\sigma^2=4$、$n=10$（$\mu$ 既知）なら $nI(\sigma^2) = 10/(2\times16) = 0.3125$、下限 $3.2$ です。

## 現場での使われ方

- **医療**：臨床試験の症例数設計で、1人あたりの情報量から「目標の精度に必要な情報量」を逆算し、必要な人数を決めます。中間解析のタイミングも「集まった情報量の割合」で決めます。
- **製造・品質管理**：寿命試験で、どの時点まで観測を続ければ母数の推定精度が十分になるかを、情報量で比べて試験計画を立てます。
- **行政・調査**：実験計画や調査設計で、情報行列の行列式を最大にする配置（D最適計画）を選びます。

## グラフ・イメージ

1つ目の図では、成功割合が同じでも $n=40$ の対数尤度は $n=10$ よりはるかに鋭く曲がります。曲率が $n$ 倍になる、つまり情報量が $nI(p)$ と標本サイズに比例することが見えます。
2つ目の図は、ベルヌーイ分布の1回あたりの情報量です。$p=0.5$ で最も小さく、端に近いほど大きくなります。

## 試験のポイント

- 2階微分の形 $-E[\partial^2\log f/\partial\theta^2]$ で計算すると、多くの分布で計算が短くなります。
- 標本全体の情報量は $nI(\theta)$。1個あたりと全体を取り違えないようにします。
- 情報量の逆数がクラメール・ラオの下限で、最尤推定量の漸近分散でもあります。$\hat\theta\approx N(\theta, 1/\{nI(\theta)\})$ から信頼区間を作る問題が典型です。
- 期待値をとらずに手元のデータで計算した $J(\hat\theta)=-\ell''(\hat\theta)$ を**観測情報量**といい、$E_\theta[J(\theta)]=nI(\theta)$ です。標準誤差は $1/\sqrt{J(\hat\theta)}$ と $1/\sqrt{nI(\hat\theta)}$ のどちらでも漸近的に同じで、ポアソン分布などの指数型分布族では $J(\hat\lambda)=\sum x_i/\hat\lambda^2=n/\bar x=nI(\hat\lambda)$ のように一致します。統計ソフトが出す標準誤差の多くは観測情報量（ヘッセ行列）によるものです。
- 母数の取り換えで $I(\eta)=I(\theta)(d\theta/d\eta)^2$ と変わります。ジェフリーズの事前分布 $\pi(\theta)\propto\sqrt{I(\theta)}$ はこの性質を使います。

## よくある間違い

- × $I(\theta) = E\big[\frac{\partial^2}{\partial\theta^2}\log f\big]$ → ○ 2階微分の期待値に**マイナス**をつけます。情報量は0以上です。
- × 一様分布 $U(0,\theta)$ でも $1/\{nI\}$ が分散の下限 → ○ 台が $\theta$ に依存するので正則条件を満たさず、下限の議論が成り立ちません（実際 $X_{(n)}$ の分散は $1/n^2$ の速さで小さくなります）。
- × 情報量は母数によらない定数 → ○ 一般に $\theta$ の関数です。推定するときは $\theta$ に推定値を代入します。
