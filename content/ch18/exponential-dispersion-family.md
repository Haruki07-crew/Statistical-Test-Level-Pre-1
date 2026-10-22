---
id: exponential-dispersion-family
title: 指数分散型分布族
reading: しすうぶんさんがたぶんぷぞく
en: Exponential dispersion family
aliases: [指数分散族, 指数型分散モデル, "exponential dispersion model", 分散関数]
chapter: 18
importance: B
tags: [一般化線形モデル, 指数型分布族, 分散関数]
related: [exponential-family, generalized-linear-model, link-function, poisson-distribution, binomial-distribution, gamma-distribution]
charts:
  - type: custom
    id: line-data
    params: { x: [0, 0.5, 1, 1.5, 2, 2.5, 3], series: [{ label: "正規 V(μ)=1", y: [1, 1, 1, 1, 1, 1, 1] }, { label: "ポアソン V(μ)=μ", y: [0, 0.5, 1, 1.5, 2, 2.5, 3] }, { label: "ガンマ V(μ)=μ²", y: [0, 0.25, 1, 2.25, 4, 6.25, 9] }], xlabel: "平均 μ", ylabel: "分散関数 V(μ)" }
    caption: 分布ごとの分散関数。平均が大きくなったときに分散がどう増えるか（増えない・比例する・2乗で増える）が分布を見分ける手がかりになる。
---

## 一言でいうと

指数分散型分布族は、正規分布・二項分布・ポアソン分布・ガンマ分布などを1つの式でまとめて書く枠組みです。
この形に書くと、平均と分散が1つの関数 $b(\theta)$ の微分で求まり、一般化線形モデルの理論を分布によらず進められます。

## 定義・数式

確率（密度）関数が

$$
f(y;\theta,\phi)=\exp\left\{\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)\right\}
$$

の形に書ける分布の集まりを**指数分散型分布族**といいます。

- $\theta$：自然パラメータ（正準パラメータ）
- $\phi$：分散パラメータ（$a(\phi)>0$。$a(\phi)=\phi/w$ と既知の重み $w$ を入れることも多い）
- $b(\theta)$：キュムラント関数。$c(y,\phi)$ は $\theta$ を含まない項

このとき、平均と分散は

$$
E[Y]=\mu=b'(\theta),\qquad V[Y]=a(\phi)\,b''(\theta)=a(\phi)\,V(\mu)
$$

です。$V(\mu)=b''(\theta)$ を $\mu$ の関数として書いたものを**分散関数**と呼びます。

| 分布 | $\theta$ | $b(\theta)$ | $a(\phi)$ | $\mu=b'(\theta)$ | $V(\mu)$ |
|---|---|---|---|---|---|
| 正規 $N(\mu,\sigma^2)$ | $\mu$ | $\theta^2/2$ | $\sigma^2$ | $\theta$ | $1$ |
| ポアソン $\text{Po}(\lambda)$ | $\log\lambda$ | $e^\theta$ | $1$ | $e^\theta$ | $\mu$ |
| 二項 $\text{Bin}(m,p)$（件数 $y$） | $\log\dfrac{p}{1-p}$ | $m\log(1+e^\theta)$ | $1$ | $\dfrac{me^\theta}{1+e^\theta}=mp$ | $\mu(1-\mu/m)$ |
| ガンマ（平均 $\mu$、形状 $\nu$） | $-1/\mu$ | $-\log(-\theta)$ | $1/\nu$ | $-1/\theta$ | $\mu^2$ |

## 数式の読み解き

**平均と分散の導出**

1個の観測の対数尤度を $\ell=\{y\theta-b(\theta)\}/a(\phi)+c(y,\phi)$ とします。スコア関数の性質

$$
E\!\left[\frac{\partial\ell}{\partial\theta}\right]=0,\qquad
E\!\left[\frac{\partial^2\ell}{\partial\theta^2}\right]+E\!\left[\left(\frac{\partial\ell}{\partial\theta}\right)^2\right]=0
$$

を使います。$\partial\ell/\partial\theta=\{y-b'(\theta)\}/a(\phi)$、$\partial^2\ell/\partial\theta^2=-b''(\theta)/a(\phi)$ なので、

1つ目から $E[Y]-b'(\theta)=0$、すなわち $E[Y]=b'(\theta)$。

2つ目から $-\dfrac{b''(\theta)}{a(\phi)}+\dfrac{V[Y]}{a(\phi)^2}=0$、すなわち $V[Y]=a(\phi)b''(\theta)$。

モーメント母関数からも同じ結果が出ます。$M_Y(t)=\exp\bigl[\{b(\theta+a(\phi)t)-b(\theta)\}/a(\phi)\bigr]$ なので、キュムラント母関数を微分して $t=0$ とおけば $b'(\theta)$、$a(\phi)b''(\theta)$ が得られます。$b$ を「キュムラント関数」と呼ぶのはこのためです。

**第8章の指数型分布族との関係**

第8章の指数型分布族 $f(y)=h(y)\exp\{\eta(\psi)T(y)-A(\psi)\}$ のうち、$T(y)=y$ とし、ばらつきを表す $\phi$ を分けて書いたものが指数分散型分布族です。$\phi$ を固定すれば、1パラメータの指数型分布族になります。

**直感**

- $b'$ が「平均を出す関数」、$b''$ が「平均が変わったとき分散がどう変わるか」を決める関数です。
- 分散関数 $V(\mu)$ が分布の個性を表します。分散が平均によらない（正規）、平均に比例する（ポアソン）、平均の2乗に比例する（ガンマ）などです。

## 具体例

**例1：ポアソン分布 $\lambda=3$**

$\theta=\log3\approx1.0986$、$b(\theta)=e^\theta$ です。$b'(\theta)=e^{1.0986}=3$、$b''(\theta)=3$、$a(\phi)=1$ なので、$E[Y]=3$、$V[Y]=3$ となり、ポアソン分布の性質「平均 ＝ 分散」が出てきます。

**例2：二項分布 $m=10,\ p=0.3$**

$\theta=\log(0.3/0.7)\approx-0.8473$、$b(\theta)=10\log(1+e^\theta)$ です。

$$
b'(\theta)=\frac{10e^\theta}{1+e^\theta}=10\times0.3=3,\qquad
b''(\theta)=\frac{10e^\theta}{(1+e^\theta)^2}=10\times0.3\times0.7=2.1
$$

で、二項分布の平均 $mp=3$、分散 $mp(1-p)=2.1$ と一致します。

**例3：ガンマ分布（平均 $\mu=2$、形状 $\nu=4$）**

密度 $f(y)=\frac{1}{\Gamma(\nu)}\left(\frac{\nu}{\mu}\right)^\nu y^{\nu-1}e^{-\nu y/\mu}$ の対数は $\nu\{y\cdot(-1/\mu)-\log\mu\}+(\mu\ \text{を含まない項})$ です。よって $\theta=-1/\mu=-0.5$、$b(\theta)=-\log(-\theta)$、$a(\phi)=1/\nu=0.25$ です。

$b'(\theta)=-1/\theta=2$、$b''(\theta)=1/\theta^2=4$ なので $V[Y]=0.25\times4=1$ です。形状4・尺度0.5のガンマ分布の分散 $4\times0.5^2=1$ と一致します。

## 現場での使われ方

- **金融・保険**：請求件数（ポアソン）と1件あたりの請求額（ガンマ）を、同じ GLM の手順で別々にモデル化します。どちらも指数分散型分布族に入るからこそ、1つのソフトの同じ関数で扱えます。
- **製造・品質管理**：測定値の分散が平均とともに大きくなる場合、分散関数 $V(\mu)$ を見て、ガンマ分布などのモデルを選びます。
- **IT・Webサービス**：アクセス数などのカウントデータで分散が平均より大きいとき、分散関数を $\phi\mu$ とする準尤度（過分散の補正）を使います。

## グラフ・イメージ

分布ごとの分散関数 $V(\mu)$ を並べた図です。正規分布は平均が変わっても分散が一定、ポアソン分布は平均に比例、ガンマ分布は平均の2乗に比例して増えます。
実データで「平均ごとの分散」を描き、どの線に近いかを見ると分布選びの目安になります。

## 試験のポイント

- 与えられた分布を $\exp\{(y\theta-b(\theta))/a(\phi)+c\}$ の形に書き直し、$\theta$、$b(\theta)$、$a(\phi)$ を読み取らせる問題が重要です。
- $E[Y]=b'(\theta)$、$V[Y]=a(\phi)b''(\theta)$ の導出（スコアの期待値0と情報量等式）を説明できるようにします。
- 正準リンクは $\theta=g(\mu)$、つまり $g=(b')^{-1}$ です。$\theta$ を $\mu$ で表せば正準リンクが読み取れます。
- 二項・ポアソンでは $a(\phi)=1$（分散パラメータが固定）で、これが過分散の問題につながります。

## よくある間違い

- × 指数分布族と指数分布を混同する → ○ 指数分布は分布族のメンバーの1つにすぎません。指数分散型分布族は、正規・二項・ポアソン・ガンマなどを含む広い枠組みです。
- × $V[Y]=b''(\theta)$ とする → ○ $a(\phi)$ を掛け忘れないようにします。正規分布なら $b''=1$ ですが、分散は $\sigma^2\times1$ です。
- × ガンマ分布の自然パラメータを $1/\mu$ とする → ○ 符号に注意し $\theta=-1/\mu$ です（$b(\theta)=-\log(-\theta)$ が定義できるよう $\theta<0$）。
