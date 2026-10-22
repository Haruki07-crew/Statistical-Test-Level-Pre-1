---
id: moment-generating-function
title: モーメント母関数
reading: もーめんとぼかんすう
en: Moment generating function
aliases: [積率母関数, MGF, キュムラント母関数]
chapter: 2
importance: A
tags: [母関数, モーメント, キュムラント, 再生性]
related: [probability-generating-function, characteristic-function, binomial-distribution, poisson-distribution, normal-distribution, skewness-kurtosis]
charts:
  - type: custom
    id: mgf-curve
    params: { family: poisson, lambda: 3 }
    sliders:
      - { param: lambda, label: "ポアソン分布の λ", min: 0.5, max: 10, step: 0.5 }
    caption: ポアソン分布のモーメント母関数 M(t) の曲線と、t=0 での接線。接線の傾きが E[X]＝λ になる。
  - type: pmf
    dist: poisson
    params: { lambda: 3 }
    sliders:
      - { param: lambda, label: "λ", min: 0.5, max: 15, step: 0.5 }
    x: { min: 0, max: 25 }
    caption: モーメント母関数を微分して得た平均 λ と分散 λ を、分布の中心と広がりとして確かめる。
---

## 一言でいうと

$E[e^{tX}]$ という1つの関数に、分布のモーメント（$E[X]$、$E[X^2]$、…）をすべて詰め込んだものです。
微分して $t=0$ を代入するとモーメントが取り出せ、さらに「関数が同じなら分布も同じ」という性質で分布の特定にも使えます。

## 定義・数式

確率変数 $X$ のモーメント母関数（積率母関数）は

$$
M_X(t) = E\left[e^{tX}\right] =
\begin{cases}
\displaystyle\sum_{x} e^{tx}\,p(x) & (\text{離散型})\\[2mm]
\displaystyle\int_{-\infty}^{\infty} e^{tx}\,f(x)\,dx & (\text{連続型})
\end{cases}
$$

です。ただし、$0$ を含むある開区間 $(-h,h)$（$h>0$）のすべての $t$ で期待値が有限のときに「モーメント母関数が存在する」といいます。

主な性質は次の4つです。

1. **モーメントの生成**：$M_X^{(k)}(0)=E[X^k]$。つまり $M_X(t)=\sum_{k=0}^{\infty}\dfrac{E[X^k]}{k!}t^k$
2. **一意性**：0の近くで $M_X(t)=M_Y(t)$ なら、$X$ と $Y$ は同じ分布に従う
3. **独立な和**：$X,Y$ が独立なら $M_{X+Y}(t)=M_X(t)\,M_Y(t)$
4. **1次変換**：$M_{aX+b}(t)=e^{bt}M_X(at)$

**キュムラント母関数** $K_X(t)=\log M_X(t)$ を使うと、

$$
K_X'(0)=E[X],\qquad K_X''(0)=V[X]
$$

と、平均と分散が直接得られます。

| 分布 | $M_X(t)$ | 存在する範囲 |
|---|---|---|
| ベルヌーイ $Bin(1,p)$ | $1-p+pe^t$ | すべての $t$ |
| 二項 $Bin(n,p)$ | $(1-p+pe^t)^n$ | すべての $t$ |
| ポアソン $Po(\lambda)$ | $\exp\{\lambda(e^t-1)\}$ | すべての $t$ |
| 幾何（失敗回数、$q=1-p$） | $\dfrac{p}{1-qe^t}$ | $t<-\log q$ |
| 正規 $N(\mu,\sigma^2)$ | $\exp\left(\mu t+\tfrac{\sigma^2t^2}{2}\right)$ | すべての $t$ |
| 指数（レート $\lambda$） | $\dfrac{\lambda}{\lambda-t}$ | $t<\lambda$ |
| ガンマ（形状 $\alpha$、レート $\lambda$） | $\left(\dfrac{\lambda}{\lambda-t}\right)^{\alpha}$ | $t<\lambda$ |
| カイ二乗 $\chi^2(k)$ | $(1-2t)^{-k/2}$ | $t<\tfrac12$ |
| 一様 $U(a,b)$ | $\dfrac{e^{bt}-e^{at}}{(b-a)t}$（$t=0$ では1） | すべての $t$ |
| コーシー、$t$ 分布、対数正規 | 存在しない | — |

ガンマ分布を尺度 $\beta=1/\lambda$ で書く流儀では $(1-\beta t)^{-\alpha}$（$t<1/\beta$）です。

## 数式の読み解き

- $t$：ただの補助の変数です。$t$ 自体に意味はなく、微分して $0$ を代入するための「つまみ」です。
- なぜ微分でモーメントが出るか：$e^{tX}=1+tX+\dfrac{t^2X^2}{2!}+\cdots$ の期待値をとると $M_X(t)=1+tE[X]+\dfrac{t^2}{2!}E[X^2]+\cdots$ です。$k$ 回微分して $t=0$ とすると、$t^k$ の係数 $E[X^k]$ だけが残ります。

以下、3つの分布で実際に微分します。

**二項分布 $Bin(n,p)$**（$q=1-p$）

$$
M(t) = \sum_{k=0}^{n}e^{tk}\binom{n}{k}p^kq^{n-k} = \sum_{k=0}^{n}\binom{n}{k}(pe^t)^kq^{n-k} = (pe^t+q)^n
$$

（最後は二項定理）。合成関数の微分で

$$
M'(t) = n\,pe^t\,(pe^t+q)^{n-1}
$$

$$
M''(t) = n\,pe^t\,(pe^t+q)^{n-1} + n(n-1)\,p^2e^{2t}\,(pe^t+q)^{n-2}
$$

$t=0$ では $pe^0+q=1$ なので、

$$
E[X]=M'(0)=np,\qquad E[X^2]=M''(0)=np+n(n-1)p^2
$$

$$
V[X] = np+n(n-1)p^2-(np)^2 = np-np^2 = np(1-p)
$$

**ポアソン分布 $Po(\lambda)$**

$$
M(t) = \sum_{k=0}^{\infty}e^{tk}\frac{e^{-\lambda}\lambda^k}{k!} = e^{-\lambda}\sum_{k=0}^{\infty}\frac{(\lambda e^t)^k}{k!} = e^{-\lambda}e^{\lambda e^t} = \exp\{\lambda(e^t-1)\}
$$

$$
M'(t) = \lambda e^t\,M(t),\qquad M''(t) = \lambda e^t\,M(t)+\lambda e^t\cdot\lambda e^t\,M(t) = (\lambda e^t+\lambda^2e^{2t})\,M(t)
$$

$M(0)=1$ より

$$
E[X]=\lambda,\qquad E[X^2]=\lambda+\lambda^2,\qquad V[X]=\lambda+\lambda^2-\lambda^2=\lambda
$$

**正規分布 $N(\mu,\sigma^2)$**

指数部分を平方完成します。

$$
tx-\frac{(x-\mu)^2}{2\sigma^2} = -\frac{\{x-(\mu+\sigma^2t)\}^2}{2\sigma^2}+\mu t+\frac{\sigma^2t^2}{2}
$$

残った積分は $N(\mu+\sigma^2t,\sigma^2)$ の密度の積分で1になるので、$M(t)=\exp\left(\mu t+\tfrac{\sigma^2t^2}{2}\right)$ です。

$$
M'(t) = (\mu+\sigma^2t)\,M(t),\qquad M''(t) = \sigma^2M(t)+(\mu+\sigma^2t)^2M(t)
$$

$$
E[X]=\mu,\qquad E[X^2]=\sigma^2+\mu^2,\qquad V[X]=\sigma^2
$$

**キュムラント母関数で楽をする**

$K(t)=\log M(t)$ とすると、$K'=M'/M$、$K''=\{M''M-(M')^2\}/M^2$ なので、$M(0)=1$ から $K''(0)=M''(0)-\{M'(0)\}^2=V[X]$ です。

- ポアソン：$K(t)=\lambda(e^t-1)$、$K'(t)=K''(t)=\lambda e^t$ → 平均も分散も $\lambda$
- 正規：$K(t)=\mu t+\tfrac{\sigma^2t^2}{2}$ → $K'(0)=\mu$、$K''(0)=\sigma^2$
- 二項：$K(t)=n\log(pe^t+q)$、$K'(t)=\dfrac{npe^t}{pe^t+q}$、$K''(t)=\dfrac{npqe^t}{(pe^t+q)^2}$ → $np$、$npq$

積の微分が出てこないので、計算がずっと短くなります。3階微分 $K'''(0)$ は3次の中心モーメント（歪度の分子）になります。

## 具体例

**例1：不良品の個数**

不良率 0.3 の工程から10個取り出したときの不良品数 $X\sim Bin(10,0.3)$ では、

$$
M'(0)=10\times0.3=3,\qquad M''(0)=3+10\times9\times0.09=11.1,\qquad V[X]=11.1-3^2=2.1
$$

$np(1-p)=10\times0.3\times0.7=2.1$ と一致します。

**例2：モーメント母関数から分布を当てる**

ある確率変数のモーメント母関数が $M(t)=(0.4+0.6e^t)^5$ だとわかりました。表と見比べると $Bin(5,\,0.6)$ のモーメント母関数と同じなので、一意性から $X\sim Bin(5,0.6)$ です。$E[X]=3$、$V[X]=1.2$ です。

**例3：ポアソン分布の再生性**

窓口Aの1時間の来客数 $X\sim Po(2)$、窓口Bの来客数 $Y\sim Po(3)$ が独立なら、

$$
M_{X+Y}(t) = e^{2(e^t-1)}\,e^{3(e^t-1)} = e^{5(e^t-1)}
$$

これは $Po(5)$ のモーメント母関数なので、$X+Y\sim Po(5)$ です。2窓口とも来客ゼロの確率は $e^{-5}\approx0.0067$ です。

## 現場での使われ方

- **金融・保険**：保険金の総額（件数も金額もランダムな和）のモーメント母関数から、平均・分散・歪度を求めて保険料や準備金を決めます。キュムラント母関数は損失確率の近似（鞍点近似）にも使われます。
- **IT・Webサービス**：サーバーの処理時間の和の裾の確率を、チェルノフ限界 $P(X\ge a)\le e^{-ta}M_X(t)$ で上から評価し、遅延の上限を保証します。
- **製造・品質管理**：独立な部品の寸法の和（組み立て後の全長）の分布を、モーメント母関数の積から正規分布だと確かめ、公差設計に使います。

## グラフ・イメージ

1つ目の図は $M(t)$ の曲線と、$t=0$ での接線です。λ を大きくすると接線の傾き（＝平均）が急になり、曲線の曲がり具合（2階微分）も大きくなります。
2つ目の図で、その平均と分散が分布の中心と広がりとして現れていることを確認してください。

## 試験のポイント

- 二項・ポアソン・正規のモーメント母関数を自分で導出し、2回微分して分散まで出せることが基本です。キュムラント母関数を使うと速くなります。
- 「独立な和のモーメント母関数＝積」と一意性を組み合わせて、再生性（二項・ポアソン・正規・ガンマ・カイ二乗）を示す問題が典型です。
- $Z=(X-\mu)/\sigma$ のモーメント母関数は $e^{-\mu t/\sigma}M_X(t/\sigma)$ です。標本平均 $\bar X$ なら $\{M_X(t/n)\}^n$ です。
- 中心極限定理の証明（モーメント母関数が存在する場合）は、標準化した和のモーメント母関数が $e^{t^2/2}$ に収束することを示す流れです。
- 確率母関数とは $M_X(t)=G_X(e^t)$ の関係にあります。

## よくある間違い

- × どんな分布にもモーメント母関数がある → ○ コーシー分布、$t$ 分布、対数正規分布などでは存在しません。いつも存在するのは特性関数です。
- × $E[X^2]=M''(0)$ をそのまま分散とする → ○ 分散は $M''(0)-\{M'(0)\}^2$ です（キュムラント母関数なら $K''(0)$ がそのまま分散）。
- × 独立でない和にも $M_{X+Y}=M_XM_Y$ を使う → ○ 積の形は独立のときだけ成り立ちます。
- × すべてのモーメントが一致すれば必ず同じ分布 → ○ モーメント母関数が存在する場合は正しいですが、対数正規分布のようにモーメントだけでは分布が決まらない例があります。
