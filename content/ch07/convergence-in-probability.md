---
id: convergence-in-probability
title: 確率収束
reading: かくりつしゅうそく
en: Convergence in probability
aliases: [確率的収束, plim]
chapter: 7
importance: A
tags: [収束の概念, 一致性, 大数の法則]
related: [weak-law-of-large-numbers, almost-sure-convergence, convergence-in-distribution, chebyshev-inequality, consistent-estimator, slutsky-theorem]
charts:
  - type: custom
    id: lln-path
    params: { p: 0.5, n: 1000, paths: 5 }
    sliders:
      - { param: p, label: "成功確率 p", min: 0.05, max: 0.95, step: 0.05 }
      - { param: n, label: "試行回数 n", min: 100, max: 5000, step: 100 }
    caption: ベルヌーイ試行の標本平均 5 本の推移。n が大きくなるほど、どの系列も p のまわりの細い帯に入る。「帯の外にいる確率が 0 に近づく」のが確率収束。
---

## 一言でいうと

確率変数の列 $X_n$ が、$n$ を大きくすると「目標の値 $X$ から少しでも離れる確率」がいくらでも小さくなることです。
推定量が真の値に近づくこと（一致性）を表す、統計学で最もよく使う収束の概念です。

## 定義・数式

確率変数の列 $X_1, X_2, \dots$ と確率変数 $X$（定数でもよい）について、任意の $\varepsilon>0$ に対して

$$
\lim_{n\to\infty} P(|X_n - X| \ge \varepsilon) = 0
$$

が成り立つとき、$X_n$ は $X$ に**確率収束する**といい、

$$
X_n \xrightarrow{p} X, \qquad \operatorname*{plim}_{n\to\infty} X_n = X
$$

と書きます。同じことを $\lim_{n\to\infty} P(|X_n - X| < \varepsilon) = 1$ と書いても同じです。

ほかの収束との関係は次のとおりです（矢印の逆は一般には成り立ちません）。

$$
\begin{aligned}
&\text{概収束} \;\Rightarrow\; \text{確率収束} \;\Rightarrow\; \text{分布収束}\\
&\text{平均二乗収束} \;\Rightarrow\; \text{確率収束}
\end{aligned}
$$

ただし、極限が**定数** $c$ のときは、分布収束 $X_n \xrightarrow{d} c$ と確率収束 $X_n \xrightarrow{p} c$ は同値です。

## 数式の読み解き

- $\varepsilon$：許容する「ずれ」の幅。どんなに小さい $\varepsilon$ を選んでも成り立つ必要があります。
- $P(|X_n - X| \ge \varepsilon)$：$n$ 番目の時点で、許容幅の外にいる確率です。
- 確率収束は「外れる**確率**が0に近づく」ことであり、「外れる**大きさ**」については何も言っていません。まれに巨大な値をとっても、その確率が0に近づけば確率収束します。

確率収束を示す定番の方法は、チェビシェフの不等式（またはマルコフの不等式）を使うことです。

$$
P(|X_n - c| \ge \varepsilon) \le \frac{E[(X_n-c)^2]}{\varepsilon^2}
$$

右辺の平均二乗誤差 $E[(X_n-c)^2]$ が0に近づけば、左辺も0に近づきます。とくに $E[X_n]\to c$ かつ $V[X_n]\to 0$ なら、$E[(X_n-c)^2] = V[X_n] + (E[X_n]-c)^2 \to 0$ なので $X_n \xrightarrow{p} c$ です。

## 具体例

**例1：コイン投げの表の割合**

公正なコインを $n$ 回投げたときの表の割合 $\bar X_n$ が、0.5 から 0.05 以上ずれる確率を考えます。チェビシェフの上限 $0.25/(n \times 0.05^2)$ と、二項分布から計算した正確な値は次のとおりです。

| $n$ | チェビシェフの上限 | 正確な値 |
|---|---|---|
| 100 | 1.0000 | 0.3682 |
| 1000 | 0.1000 | 0.0017 |
| 10000 | 0.0100 | $10^{-22}$ 未満 |

$n$ を大きくするとずれる確率が0に近づくので、$\bar X_n \xrightarrow{p} 0.5$ です。

**例2：一様分布の最大値**

$X_1,\dots,X_n$ が独立に一様分布 $U(0,\theta)$ に従うとき、最大値 $M_n=\max_i X_i$ は $\theta$ の推定量になります。$M_n \le \theta$ なので

$$
P(|M_n - \theta| \ge \varepsilon) = P(M_n \le \theta-\varepsilon) = \left(1 - \frac{\varepsilon}{\theta}\right)^n \to 0
$$

$\theta=1$、$\varepsilon=0.05$ なら、$n=50$ で $0.95^{50}\approx 0.0769$、$n=100$ で $0.95^{100}\approx 0.0059$ です。よって $M_n \xrightarrow{p} \theta$ で、$M_n$ は一致推定量です。

**例3：確率収束しても期待値は収束しない**

$X_n$ が確率 $1/n$ で値 $n$、確率 $1-1/n$ で値0をとるとします。任意の $0<\varepsilon<1$ について、$n>\varepsilon$ なら $P(|X_n| \ge \varepsilon) = 1/n \to 0$ なので $X_n \xrightarrow{p} 0$ です。
一方、$E[X_n] = n \times (1/n) = 1$ で、期待値は0に近づきません。まれに出る大きな値が期待値を支えているためです。

## 現場での使われ方

- **推定全般**：推定量 $\hat\theta_n$ が $\hat\theta_n \xrightarrow{p} \theta$ を満たすこと（一致性）は、推定量を使ってよい最低条件として確認されます。標本平均・標本分散・最尤推定量などはこの性質を持ちます。
- **金融・保険**：契約数が増えると1件あたりの平均保険金支払額が期待値に確率収束することが、保険料を計算できる根拠になります。
- **計量経済**：回帰係数の推定量の一致性（$\operatorname{plim}\hat\beta = \beta$）を議論するとき、説明変数と誤差の相関があると一致性が崩れる（内生性）ことを、確率収束の計算で示します。

## グラフ・イメージ

ベルヌーイ試行の標本平均を5本描いています。$n$ が小さいうちは系列ごとに大きくばらつきますが、$n$ が大きくなると、どの系列も $p$ のまわりの細い帯に収まります。
「帯の外にいる確率」が $n$ とともに0に近づくことが、確率収束の意味です。

## 試験のポイント

- 定義式 $P(|X_n-X|\ge\varepsilon)\to 0$ を正確に書けるようにします。
- 一致性を示す問題では、「$E[\hat\theta_n]\to\theta$ かつ $V[\hat\theta_n]\to 0$ ならチェビシェフの不等式で一致」という流れが定番です。
- 最大値・最小値の推定量では、例2のように分布関数から $P(|M_n-\theta|\ge\varepsilon)$ を直接計算する方法も重要です。
- 収束の強さの関係（概収束・平均二乗収束 ⇒ 確率収束 ⇒ 分布収束）と、極限が定数なら分布収束と確率収束が同値であることを押さえます。
- 連続写像定理により、$X_n \xrightarrow{p} c$ なら連続関数 $g$ について $g(X_n) \xrightarrow{p} g(c)$ です（例：$S^2 \xrightarrow{p} \sigma^2$ なら $S \xrightarrow{p} \sigma$）。

## よくある間違い

- × 確率収束すれば期待値も収束する → ○ 例3のように、確率収束しても $E[X_n]$ は収束しないことがあります。
- × 不偏推定量なら一致推定量である → ○ 不偏性と一致性は別の性質です。$X_1$ だけで作った推定量 $\hat\mu = X_1$ は不偏ですが、$n$ を増やしても分散が減らないので一致性を持ちません。
- × 確率収束と概収束を同じものと考える → ○ 確率収束は「各時点で外れる確率が小さい」こと、概収束は「ほとんどすべての系列がいずれ外れなくなる」ことで、概収束のほうが強い条件です。
