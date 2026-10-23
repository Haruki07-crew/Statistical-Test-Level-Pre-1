---
id: birth-death-process
title: 出生死滅過程
reading: しゅっせいしめつかてい
en: Birth-death process
aliases: [出生死亡過程, 生成消滅過程, birth and death process]
chapter: 15
importance: A
tags: [連続時間マルコフ連鎖, 定常分布, 平衡方程式, 待ち行列]
related: [continuous-time-markov-chain, stationary-distribution, detailed-balance, mm1-queue, pure-birth-process, poisson-process]
charts:
  - type: custom
    id: markov-chain
    params: { P: [[0.6, 0.4, 0], [0.6, 0.2, 0.2], [0, 0.6, 0.4]], labels: ["故障0台", "故障1台", "故障2台"], start: 0, steps: 20 }
    sliders:
      - { param: start, label: "出発する状態", min: 0, max: 2, step: 1 }
      - { param: steps, label: "ステップ数", min: 5, max: 40, step: 1 }
    caption: 例1の機械修理モデルを、時間を細かく区切った離散時間の連鎖（一様化 P=I+Q/5）で表したもの。どこから始めても (9/17, 6/17, 2/17) ≈ (0.529, 0.353, 0.118) に落ち着く。
  - type: pmf
    dist: poisson
    params: { lambda: 8 }
    sliders:
      - { param: lambda, label: "λ/μ（到着率÷1人の退去率）", min: 0.5, max: 20, step: 0.5 }
    x: { min: 0, max: 35 }
    caption: 例2（λ_n=λ、μ_n=nμ）の定常分布は平均 λ/μ のポアソン分布になる。
---

## 一言でいうと

人数（個体数・客数）が「1増える」か「1減る」かのどちらかでしか変化しない、連続時間のマルコフ過程です。
定常状態では「上向きの流れ＝下向きの流れ」という釣り合いから、定常分布が積の形で求まります。

## 定義・数式

状態空間 $\{0,1,2,\dots\}$（または $\{0,1,\dots,K\}$）をとる連続時間マルコフ過程 $X(t)$ で、微小時間 $h$ に

$$
\begin{aligned}
P(X(t+h)=n+1\mid X(t)=n)&=\lambda_n h+o(h)\\
P(X(t+h)=n-1\mid X(t)=n)&=\mu_n h+o(h)\\
P(X(t+h)=n\mid X(t)=n)&=1-(\lambda_n+\mu_n)h+o(h)
\end{aligned}
$$

となるものを**出生死滅過程**といいます。$\lambda_n\ge0$ を**出生率**、$\mu_n\ge0$ を**死亡率**（$\mu_0=0$）と呼びます。

定常分布 $\pi_n$ は、存在すれば

$$
\pi_n=\pi_0\prod_{k=1}^{n}\frac{\lambda_{k-1}}{\mu_k}=\pi_0\,\frac{\lambda_0\lambda_1\cdots\lambda_{n-1}}{\mu_1\mu_2\cdots\mu_n},\qquad
\pi_0=\Bigl(1+\sum_{n=1}^{\infty}\prod_{k=1}^{n}\frac{\lambda_{k-1}}{\mu_k}\Bigr)^{-1}
$$

で、存在の条件は（すべての $\lambda_n,\mu_n>0$ のもとで）括弧内の和が有限であることです。

## 数式の読み解き

- 状態 $n$ にいる時間は指数分布 $\mathrm{Exp}(\lambda_n+\mu_n)$ で、離れるときは確率 $\lambda_n/(\lambda_n+\mu_n)$ で上へ、$\mu_n/(\lambda_n+\mu_n)$ で下へ移ります。
- 純出生過程（$\mu_n=0$）、ポアソン過程（$\lambda_n=\lambda,\ \mu_n=0$）、M/M/1 待ち行列（$\lambda_n=\lambda,\ \mu_n=\mu$）は特別な場合です。
- 純出生過程では滞在時間が独立に $\mathrm{Exp}(\lambda_n)$ なので、状態1から $k$ に達するまでの平均時間は $\sum_{n=1}^{k-1}1/\lambda_n$ です。$\lambda_n=n\lambda$、$X(0)=1$ の**ユール過程**では、$X(t)$ は成功確率 $e^{-\lambda t}$、台 $\{1,2,\dots\}$ の幾何分布 $P(X(t)=n)=e^{-\lambda t}(1-e^{-\lambda t})^{n-1}$ に従い、$E[X(t)]=e^{\lambda t}$ です。純出生過程は増える一方なので定常分布をもちません。

**コルモゴロフの前向き方程式**

$P_n(t)=P(X(t)=n)$ とします。$t+h$ に $n$ にいるのは「$n-1$ から生まれた」「$n+1$ から死んだ」「$n$ にとどまった」のいずれかなので

$$
P_n(t+h)=\lambda_{n-1}hP_{n-1}(t)+\mu_{n+1}hP_{n+1}(t)+\{1-(\lambda_n+\mu_n)h\}P_n(t)+o(h)
$$

移項して $h\to0$ とすると

$$
P_n'(t)=\lambda_{n-1}P_{n-1}(t)+\mu_{n+1}P_{n+1}(t)-(\lambda_n+\mu_n)P_n(t)
$$

**定常分布の導出**

定常状態では $P_n'(t)=0$ なので、$\pi_n$ は**平衡方程式**

$$
(\lambda_n+\mu_n)\pi_n=\lambda_{n-1}\pi_{n-1}+\mu_{n+1}\pi_{n+1}\qquad(n\ge1),\qquad \lambda_0\pi_0=\mu_1\pi_1
$$

を満たします。左辺は「$n$ から出ていく流れ」、右辺は「$n$ に入ってくる流れ」です。

$n=0$ の式は $\lambda_0\pi_0=\mu_1\pi_1$。これを $n=1$ の式 $(\lambda_1+\mu_1)\pi_1=\lambda_0\pi_0+\mu_2\pi_2$ に入れると $\mu_1\pi_1$ と $\lambda_0\pi_0$ が消えて $\lambda_1\pi_1=\mu_2\pi_2$。以下同様に帰納法で

$$
\lambda_n\pi_n=\mu_{n+1}\pi_{n+1}\qquad(n=0,1,2,\dots)
$$

が得られます。これは「$n$ と $n+1$ の間の境目を上に越える流れと下に越える流れが等しい」という**詳細釣り合い**です。$\pi_{n+1}=(\lambda_n/\mu_{n+1})\pi_n$ を繰り返すと積の公式になり、$\sum\pi_n=1$ から $\pi_0$ が決まります。

## 具体例

**例1：機械2台と修理工1人**

機械が2台あり、動いている機械は1台ずつ独立に故障率 $\lambda=1$（/週）で故障します。修理工は1人で、修理率 $\mu=3$（/週）です。故障中の台数 $X(t)\in\{0,1,2\}$ は出生死滅過程で

$$
\lambda_0=2\lambda=2,\quad \lambda_1=\lambda=1,\quad \mu_1=\mu_2=3
$$

です（2台動いていれば故障率は2倍）。積の公式から

$$
\pi_1=\frac{2}{3}\pi_0,\qquad \pi_2=\frac{2}{3}\cdot\frac{1}{3}\pi_0=\frac{2}{9}\pi_0
$$

$\pi_0(1+\frac23+\frac29)=\frac{17}{9}\pi_0=1$ より

$$
\pi_0=\frac{9}{17}\approx0.5294,\quad \pi_1=\frac{6}{17}\approx0.3529,\quad \pi_2=\frac{2}{17}\approx0.1176
$$

平均故障台数は $(1\times6+2\times2)/17=10/17\approx0.5882$ 台、2台とも止まっている時間の割合は約11.8%です。

**例2：窓口が無限にある場合（M/M/∞）**

客が強度 $\lambda=4$ で到着し、各客は平均2時間（$\mu=0.5$）滞在して独立に去ります。$\lambda_n=4$、$\mu_n=0.5n$ なので

$$
\pi_n=\pi_0\frac{4^n}{0.5^n\,n!}=\pi_0\frac{8^n}{n!}
$$

$\sum\pi_n=1$ より $\pi_0=e^{-8}$ で、定常分布はポアソン分布 $\mathrm{Po}(8)$ です。平均滞在人数は8人、だれもいない確率は $e^{-8}\approx0.000335$ です。

## 現場での使われ方

- **製造・品質管理**：機械の故障・修理を出生死滅過程で表し、修理工の人数と稼働率の関係を評価します（例1）。
- **IT・Webサービス**：同時接続数を出生死滅過程で表し、定常分布から接続数が上限を超える確率を求めて回線数を決めます。
- **医療**：病床の占有数を「入院＝出生、退院＝死亡」とみなし、満床になる確率を計算します。
- **行政・調査**：生物の個体数や感染者数の増減の確率モデルとして使います。

## グラフ・イメージ

上の図は例1を、時間を細かく区切った離散時間の連鎖に直したものです（推移率行列 $Q$ を $P=I+Q/5$ と変換する「一様化」。定常分布は変わりません）。どの状態から始めても $(9/17,6/17,2/17)$ に収束します。
下の図は例2の定常分布で、$\lambda/\mu$ を動かすとポアソン分布の山が移ります。

## 試験のポイント

- 「隣どうしの流れの釣り合い」$\lambda_n\pi_n=\mu_{n+1}\pi_{n+1}$ から積の公式を出し、正規化する流れが最重要です。
- 状態によって率が変わるモデル（台数に比例する故障率、窓口数に応じた退去率など）で $\lambda_n,\mu_n$ を正しく設定できるかが問われます。
- M/M/1、M/M/c、M/M/∞、有限容量 M/M/1/K はすべてこの公式の特別な場合です。
- 定常分布が存在するには、$\sum_n\prod_k\lambda_{k-1}/\mu_k<\infty$ が必要です。

## よくある間違い

- × 2台の機械の故障率を、故障台数によらず $\lambda$ とする → ○ 動いている台数に比例し、0台故障なら $2\lambda$、1台故障なら $\lambda$ です。
- × 平衡方程式を推移確率行列の $\pi P=\pi$ と同じ形で解こうとする → ○ 連続時間では「率」で書いた $\pi Q=0$（流出＝流入）を解きます。出生死滅過程では隣どうしの釣り合いに帰着します。
- × 積の公式の分子と分母を逆にする → ○ $\pi_{n+1}/\pi_n=\lambda_n/\mu_{n+1}$ で、上へ行く率が分子、下へ戻る率が分母です。
