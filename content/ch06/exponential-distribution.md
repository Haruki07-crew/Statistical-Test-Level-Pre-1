---
id: exponential-distribution
title: 指数分布
reading: しすうぶんぷ
en: Exponential distribution
aliases: [Exponential distribution, "Exp(λ)"]
chapter: 6
importance: A
tags: [連続型分布, 無記憶性, 待ち時間, ハザード関数]
related: [gamma-distribution, geometric-distribution, poisson-distribution, poisson-process, hazard-function, survival-function]
charts:
  - type: pdf
    dist: expon
    params: { rate: 1 }
    sliders:
      - { param: rate, label: "レート λ", min: 0.2, max: 3, step: 0.1 }
    x: { min: 0, max: 8 }
    caption: x=0 で最大値 λ をとり、右へ指数的に減っていく。λ を大きくすると立ち上がりが高くなり、平均 1/λ が短くなる。
  - type: survival-hazard
    dist: expon
    params: { rate: 1 }
    sliders:
      - { param: rate, label: "レート λ", min: 0.2, max: 3, step: 0.1 }
    caption: 生存関数 e^{−λx} は指数的に減るが、ハザード関数は λ で一定の水平線になる。これが無記憶性の表れ。
---

## 一言でいうと

「次の出来事が起こるまでの待ち時間」を表す連続型分布です。
どれだけ待ったかに関係なく「この先の待ち時間の分布は同じ」という**無記憶性**をもつ、唯一の連続型分布です。

## 定義・数式

この項では**レート（発生率）$\lambda>0$** を使う流儀を採用します。密度関数と累積分布関数は

$$
f(x)=\lambda e^{-\lambda x},\qquad F(x)=1-e^{-\lambda x},\qquad x\ge0
$$

で、$X\sim \mathrm{Exp}(\lambda)$ と書きます。平均 $\theta=1/\lambda$ を使って $f(x)=\frac1\theta e^{-x/\theta}$ と書く流儀（尺度パラメータ表記）もあります。教科書や問題文で $\mathrm{Exp}(2)$ がどちらの意味かを必ず確認してください。

| 項目 | 内容（レート $\lambda$ 表記） |
|---|---|
| 台 | $[0,\infty)$ |
| パラメータ | $\lambda>0$ |
| 期待値 | $1/\lambda$ |
| 分散 | $1/\lambda^2$ |
| モーメント母関数 | $M(t)=\dfrac{\lambda}{\lambda-t}$（$t<\lambda$） |
| 特性関数 | $\varphi(t)=\dfrac{\lambda}{\lambda-it}$ |
| 生存関数・ハザード | $S(x)=e^{-\lambda x}$、$h(x)=\lambda$（一定） |
| 中央値 | $\log 2/\lambda$ |

## 数式の読み解き

- $\lambda$：単位時間あたりに出来事が起こる率です。1分に平均2回なら $\lambda=2$、平均待ち時間は $1/2$ 分です。
- $e^{-\lambda x}$：時間 $x$ までに一度も起こらない確率 $S(x)$ です。

**MGF と平均・分散**。$t<\lambda$ のとき

$$
M(t)=\int_0^\infty e^{tx}\lambda e^{-\lambda x}dx=\lambda\int_0^\infty e^{-(\lambda-t)x}dx=\frac{\lambda}{\lambda-t}
$$

$M'(t)=\lambda/(\lambda-t)^2$、$M''(t)=2\lambda/(\lambda-t)^3$ より、$E[X]=1/\lambda$、$E[X^2]=2/\lambda^2$、$V[X]=2/\lambda^2-1/\lambda^2=1/\lambda^2$ です。平均と標準偏差が等しい（変動係数が1）のが特徴です。

**無記憶性**。$s,t\ge0$ について

$$
P(X>s+t\mid X>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}=P(X>t)
$$

すでに $s$ だけ待っていても、残りの待ち時間は新品と同じ分布です。ハザード関数 $h(x)=f(x)/S(x)=\lambda$ が一定であることと同じ意味です。

**幾何分布との対比**。第5章の幾何分布（成功までの失敗回数 $Y$）も $P(Y\ge s+t\mid Y\ge s)=P(Y\ge t)$ という無記憶性をもちます。幾何分布は「離散時間で毎回確率 $p$ で起こる」待ち時間、指数分布は「連続時間で率 $\lambda$ で起こる」待ち時間で、無記憶性をもつのは離散型では幾何分布、連続型では指数分布だけです。1回の幅を $\Delta$ として $p=\lambda\Delta$ とし、$\Delta\to0$ とすると、幾何分布の待ち時間は指数分布に近づきます。

**他の分布との関係**。

- 独立な $\mathrm{Exp}(\lambda)$ の $n$ 個の和は、形状 $n$・レート $\lambda$ のガンマ分布です（MGF $\{\lambda/(\lambda-t)\}^n$）。
- ポアソン過程（率 $\lambda$）の到着間隔は $\mathrm{Exp}(\lambda)$ です。
- 独立な $X_i\sim\mathrm{Exp}(\lambda_i)$ の最小値は $\mathrm{Exp}(\lambda_1+\cdots+\lambda_k)$ です。$P(\min>x)=\prod e^{-\lambda_i x}$ から出ます。
- $\mathrm{Exp}(1/2)$（平均2）は自由度2のカイ二乗分布と同じです。

## 具体例

**例1：コールセンターの着信**

着信が1分あたり平均 0.5 件（$\lambda=0.5$）のポアソン過程に従うとき、次の着信までの時間 $X\sim\mathrm{Exp}(0.5)$ です。平均は2分、中央値は $\log2/0.5\approx1.386$ 分です。

$$
P(X>3)=e^{-0.5\times3}=e^{-1.5}\approx0.2231
$$

すでに2分着信がなかったとき、さらに3分以上来ない確率も、無記憶性により

$$
P(X>5\mid X>2)=e^{-0.5\times 3}\approx0.2231
$$

で同じです。

**例2：直列システムの故障時間（試験に近い形）**

3つの部品の寿命が独立に、レート 0.1、0.2、0.3（1年あたり）の指数分布に従い、1つでも壊れるとシステムが止まります。システムの寿命 $T=\min(X_1,X_2,X_3)$ は $\mathrm{Exp}(0.6)$ に従うので、

$$
E[T]=\frac{1}{0.6}\approx1.667\ \text{年}
$$

です。最初に壊れるのが部品3である確率は $0.3/0.6=0.5$ です。

## 現場での使われ方

- **製造・品質管理**：偶発故障期（故障率が一定の期間）の電子部品の寿命を指数分布でモデル化し、平均故障間隔（MTBF）$=1/\lambda$ を推定します。
- **IT・Webサービス**：サーバへのリクエスト到着間隔をモデル化し、待ち行列理論（M/M/1など）でレスポンス遅延を見積もります。
- **金融・保険**：保険事故の発生間隔をモデル化し、一定期間内に事故が起こらない確率を計算します。
- **医療**：生存時間解析で、ハザードが一定と仮定できる場合の最も単純なモデルとして使います。

## グラフ・イメージ

1つ目では、$\lambda$ を大きくすると $x=0$ での高さ $\lambda$ が上がり、分布が左に縮む（平均 $1/\lambda$ が短くなる）ことを確認します。
2つ目では、生存関数が減っていく一方で、ハザード関数は水平線のままであることに注目してください。故障の起こりやすさが時間とともに変わらない、ということです。

## 試験のポイント

- 平均 $1/\lambda$、分散 $1/\lambda^2$、MGF $\lambda/(\lambda-t)$ は即答できるようにします。採用しているパラメータ表記（レートか平均か）を最初に確認します。
- 無記憶性を条件付き確率で示す計算は典型です。幾何分布との対応も押さえます。
- 指数分布の和 → ガンマ分布、最小値 → 指数分布、ポアソン過程の到着間隔 → 指数分布、の3つの関係は頻繁に使います。
- 最尤推定量は $\hat\lambda=1/\bar X$ です（第8章）。

## よくある間違い

- × $\mathrm{Exp}(2)$ の平均を常に2とする → ○ レート表記なら平均は $1/2$、尺度（平均）表記なら2です。
- × 長く待ったのだから、そろそろ来るはず → ○ 指数分布では残りの待ち時間の分布は変わりません。
- × 最小値の平均を $\min(1/\lambda_i)$ とする → ○ 最小値はレートの和 $\sum\lambda_i$ の指数分布で、平均は $1/\sum\lambda_i$ です。
