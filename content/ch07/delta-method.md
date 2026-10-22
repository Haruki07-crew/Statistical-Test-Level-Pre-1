---
id: delta-method
title: デルタ法
reading: でるたほう
en: Delta method
aliases: [デルタ・メソッド, δ法]
chapter: 7
importance: A
tags: [漸近理論, 漸近分散, 漸近正規性, テイラー展開, 分散安定化変換]
related: [central-limit-theorem, continuous-mapping-theorem, slutsky-theorem, asymptotic-normality-of-mle, variance-stabilizing-transform, logit-logistic-transformation]
charts:
  - type: pdf
    dist: lognorm
    params: { mu: 0, sigma: 0.1 }
    sliders:
      - { param: sigma, label: "元の推定量の標準誤差 σ", min: 0.02, max: 0.8, step: 0.02 }
    x: { min: 0, max: 3 }
    compare:
      - { dist: norm, params: { mu: 1, sigma: 0.1 }, label: "デルタ法の近似 N(1, 0.1²)（σ=0.1 のとき）" }
    caption: T ~ N(0, σ²) を g(t)=e^t で変換した e^T の分布（対数正規分布、実線）。デルタ法は N(e^0, (e^0)²σ²) = N(1, σ²) で近似する。σ が小さいと近似はよく合い、σ を大きくすると分布がゆがんで近似が崩れる。
---

## 一言でいうと

漸近正規な推定量を関数で変換したとき、その関数値も漸近正規になり、漸近分散は「元の分散 × 傾きの二乗」になる、という方法です。
関数を推定値のまわりで1次式（接線）で近似する、という考え方に基づきます。

## 定義・数式

推定量 $T_n$ が

$$
\sqrt{n}\,(T_n - \theta) \xrightarrow{d} N(0, \sigma^2)
$$

を満たし、関数 $g$ が $\theta$ で微分可能で $g'(\theta) \ne 0$ とします。このとき

$$
\sqrt{n}\,\{g(T_n) - g(\theta)\} \xrightarrow{d} N\!\left(0,\ \{g'(\theta)\}^2 \sigma^2\right)
$$

が成り立ちます。実用上は次の近似として使います。

$$
V[g(T_n)] \approx \{g'(\theta)\}^2\, V[T_n]
$$

多変量版：$\sqrt n(\boldsymbol{T}_n-\boldsymbol\theta)\xrightarrow{d}N(\boldsymbol 0,\Sigma)$ なら、勾配ベクトル $\nabla g(\boldsymbol\theta)$ を使って

$$
\sqrt n\{g(\boldsymbol T_n)-g(\boldsymbol\theta)\}\xrightarrow{d}N\!\left(0,\ \nabla g(\boldsymbol\theta)^\top \Sigma\, \nabla g(\boldsymbol\theta)\right)
$$

です。

## 数式の読み解き

- $g'(\theta)$：変換の「拡大率」です。$\theta$ の近くで $T_n$ が $\Delta$ だけずれると、$g(T_n)$ はおよそ $g'(\theta)\Delta$ だけずれます。
- 分散は「ずれの二乗」の平均なので、拡大率が二乗で掛かります。標準誤差なら $|g'(\theta)|$ 倍です。

導出は1次のテイラー展開です。

$$
g(T_n) = g(\theta) + g'(\theta)(T_n - \theta) + (\text{剰余項})
$$

両辺から $g(\theta)$ を引いて $\sqrt n$ を掛けると

$$
\sqrt n\{g(T_n)-g(\theta)\} = g'(\theta)\cdot\sqrt n(T_n-\theta) + (\text{確率収束で } 0 \text{ に向かう項})
$$

右辺第1項は $g'(\theta) \times N(0,\sigma^2)$ に分布収束し、剰余項は $T_n \xrightarrow{p} \theta$ から0に確率収束します。スラツキーの定理により、和は $N(0,\{g'(\theta)\}^2\sigma^2)$ に分布収束します。

$g'(\theta)=0$ のときは1次の項が消えるので、2次の項を使います（2次のデルタ法）。このとき $n\{g(T_n)-g(\theta)\} \xrightarrow{d} \frac{1}{2}g''(\theta)\sigma^2\chi^2(1)$ となり、収束の速さも分布の形も変わります。

## 具体例

**例1：指数分布のレートの推定量**

平均 $1/\lambda$ の指数分布から $n$ 個とると、$V[X]=1/\lambda^2$ なので $\sqrt n(\bar X - 1/\lambda)\xrightarrow{d}N(0, 1/\lambda^2)$ です。レートの推定量 $\hat\lambda = 1/\bar X$ は $g(x)=1/x$ で、$g'(x)=-1/x^2$ を $x=1/\lambda$ で評価すると $g'(1/\lambda) = -\lambda^2$ です。よって

$$
V[\hat\lambda] \approx (\lambda^2)^2 \times \frac{1}{n\lambda^2} = \frac{\lambda^2}{n}
$$

です。$\lambda=2$、$n=100$ なら標準誤差は $2/\sqrt{100}=0.2$ です。乱数実験で $\hat\lambda$ の標準偏差を求めると約 0.204 で、よく合います。

**例2：ロジット（対数オッズ）の信頼区間**

$n=200$ で標本比率 $\hat p=0.3$ のとき、$g(p)=\log\frac{p}{1-p}$ の微分は $g'(p)=\frac{1}{p(1-p)}$ です。$V[\hat p]=p(1-p)/n$ なので

$$
V[g(\hat p)] \approx \frac{1}{\{p(1-p)\}^2}\cdot\frac{p(1-p)}{n} = \frac{1}{np(1-p)} = \frac{1}{200\times 0.21} \approx 0.02381
$$

標準誤差は $\sqrt{0.02381}\approx 0.1543$ です。ロジットの推定値 $\log(0.3/0.7)\approx -0.8473$ に ±$1.96\times 0.1543$ をすると $[-1.1497,\ -0.5449]$、これを比率に戻すと $[0.2405,\ 0.3671]$ です。区間が0と1の外にはみ出さないのが利点です。

**例3：ポアソン分布の分散安定化**

$\bar X$ をポアソン分布 $Po(\lambda)$ の標本平均とすると $V[\bar X]=\lambda/n$ です。$g(x)=\sqrt x$、$g'(\lambda)=\frac{1}{2\sqrt\lambda}$ より

$$
V[\sqrt{\bar X}] \approx \frac{1}{4\lambda}\cdot\frac{\lambda}{n} = \frac{1}{4n}
$$

で、$\lambda$ によらなくなります。$\lambda=4$、$n=50$ なら $\sqrt{1/200}\approx 0.0707$ で、乱数実験の値（約 0.0707）と一致します。

## 現場での使われ方

- **医療・疫学**：オッズ比や相対リスクの信頼区間は、対数をとった $\log(\text{OR})$ の標準誤差をデルタ法で求め、区間を作ってから指数で戻します。
- **金融**：リターンの平均と分散の推定値から、シャープレシオ（平均 ÷ 標準偏差）のような比の標準誤差を多変量デルタ法で求めます。
- **マーケティング・A/Bテスト**：「売上 ÷ 訪問数」のような比率の指標の分散を、分子と分母の分散・共分散から近似計算します。
- **データ変換**：ポアソンデータの平方根変換、比率の逆正弦変換など、分散を一定にする変換（分散安定化変換）の設計に使います。

## グラフ・イメージ

$T\sim N(0,\sigma^2)$ を $g(t)=e^t$ で変換した分布（対数正規分布）と、デルタ法による近似 $N(1,\sigma^2)$ を重ねています。
$\sigma$ が小さいうちは $g$ が「ほぼ直線」とみなせる範囲に $T$ が収まるので、2つはほとんど重なります。$\sigma$ を大きくすると曲がりの影響が出て右に裾を引き、近似がずれていきます。デルタ法は「標準誤差が小さい（$n$ が大きい）とき」の近似です。

## 試験のポイント

- 公式 $V[g(T)]\approx\{g'(\theta)\}^2V[T]$ を使って、対数・逆数・平方根・ロジットなどの漸近分散を求める計算が典型です。
- 最尤推定量の漸近正規性と組み合わせ、$g(\hat\theta)$ の漸近分散を $\{g'(\theta)\}^2/I(\theta)$（$I$ はフィッシャー情報量）として求める流れが重要です。
- 分散を一定にする変換は、$g'(\theta)\propto 1/\sqrt{v(\theta)}$（$v(\theta)$ は分散関数）を解いて求めます。ポアソンなら $\sqrt{x}$、二項比率なら $\arcsin\sqrt{x}$ です。
- 実際の区間計算では、未知の $\theta$ を $\hat\theta$ で置き換えます（スラツキーの定理で正当化されます）。
- $g'(\theta)=0$ の場合は1次のデルタ法が使えない点も押さえておきます。

## よくある間違い

- × 微分を二乗し忘れて $V[g(T)]\approx g'(\theta)V[T]$ とする → ○ 分散には $\{g'(\theta)\}^2$ が掛かります。標準誤差なら $|g'(\theta)|$ です。
- × 期待値も $E[g(T)] = g(E[T])$ と正確に成り立つ → ○ 1次近似で成り立つだけで、有限の $n$ では $g$ の曲がりによる偏り（およそ $\frac12 g''(\theta)V[T]$）があります。
- × $g'$ を推定値 $T_n$ ではなく任意の点で評価する → ○ 理論上は真の値 $\theta$ で評価し、実際の計算ではその推定値 $\hat\theta$ を代入します。
- × $g'(\theta)=0$ でも漸近分散0の正規分布になる → ○ この場合は2次の項が主になり、極限はカイ二乗型の分布になります。
