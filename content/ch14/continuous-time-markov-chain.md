---
id: continuous-time-markov-chain
title: 連続時間マルコフ連鎖
reading: れんぞくじかんまるこふれんさ
en: Continuous-time Markov chain
aliases: [連続時間マルコフ過程, 推移率行列, 生成行列, "Q行列", CTMC]
chapter: 14
importance: C
tags: [マルコフ連鎖, 指数分布, 推移率, コルモゴロフ方程式]
related: [markov-chain, chapman-kolmogorov-equation, stationary-distribution, birth-death-process, poisson-process, exponential-distribution]
---

## 一言でいうと

状態の移り変わりが、決まった時刻ごとではなく、連続な時間の中の任意の瞬間に起こるマルコフ連鎖です。
各状態に指数分布の時間だけとどまり、移り先は「率」の比で決まります。

## 定義・数式

離散の状態空間をとる確率過程 $\{X(t):t\ge0\}$ が、任意の $s,t\ge0$ について

$$
P(X(t+s)=j\mid X(s)=i,\ X(u)\ (u<s))=P(X(t+s)=j\mid X(s)=i)=p_{ij}(t)
$$

を満たす（時間的に一様な）とき、**連続時間マルコフ連鎖**といいます。微小時間 $h$ について

$$
p_{ij}(h)=q_{ij}h+o(h)\ (i\ne j),\qquad p_{ii}(h)=1-q_ih+o(h),\qquad q_i=\sum_{j\ne i}q_{ij}
$$

の $q_{ij}\ge0$ を**推移率**といい、対角に $-q_i$ を置いた行列 $Q$ を**生成行列**（推移率行列）とよびます。$Q$ の各行の和は0です。
推移確率行列 $P(t)=(p_{ij}(t))$ は

$$
P(s+t)=P(s)P(t),\qquad P'(t)=P(t)Q=QP(t),\qquad P(t)=e^{tQ}
$$

を満たし、定常分布 $\boldsymbol{\pi}$ は $\boldsymbol{\pi}Q=\boldsymbol{0}$、$\sum_i\pi_i=1$ の解です。

## 数式の読み解き

- 状態 $i$ の滞在時間は $\mathrm{Exp}(q_i)$ です。マルコフ性（過去を忘れる）を連続時間で保つには、無記憶性をもつ指数分布しかありえないからです。
- 状態 $i$ を離れるとき、行き先 $j$ は確率 $q_{ij}/q_i$ で選ばれます。この行き先だけを並べた離散時間の連鎖を**埋め込み連鎖**といいます。
- $\boldsymbol{\pi}Q=\boldsymbol{0}$ は「各状態から出ていく流れ＝入ってくる流れ」を表します。離散時間の $\boldsymbol{\pi}P=\boldsymbol{\pi}$ に対応する式です。

## 具体例

**機械の稼働(0)と故障(1)**

故障率 $\lambda=0.2$（1日あたり）、修理率 $\mu=1.0$（1日あたり）とすると

$$
Q=\begin{pmatrix}-0.2&0.2\\1.0&-1.0\end{pmatrix}
$$

$\boldsymbol{\pi}Q=\boldsymbol{0}$ より $0.2\pi_0=1.0\pi_1$ で、$\pi_0=\mu/(\lambda+\mu)=0.8333$、$\pi_1=0.1667$。長期的な稼働率は約83%です。
今日稼働している機械が1日後も稼働している確率は

$$
p_{00}(t)=\frac{\mu}{\lambda+\mu}+\frac{\lambda}{\lambda+\mu}e^{-(\lambda+\mu)t},\qquad p_{00}(1)=0.8333+0.1667\times0.3012=0.8835
$$

です。$t$ が大きくなると $e^{-1.2t}\to0$ で、定常確率 $0.8333$ に近づきます。

## 現場での使われ方

- **製造・品質管理**：設備の稼働・故障・修理中の状態を率で表し、長期的な稼働率や保守要員の必要数を見積もります。
- **IT・Webサービス**：待ち行列（M/M/1 など）の系内人数を連続時間マルコフ連鎖で表し、平均待ち時間を求めます。
- **医療**：病気の進行段階の移り変わりを率で表し（多状態モデル）、ある期間後に各段階にいる確率を計算します。

## 試験のポイント

- 生成行列 $Q$ は「非対角が非負、各行の和が0」であることを押さえます。
- 定常分布は $\boldsymbol{\pi}Q=\boldsymbol{0}$ で求めます。2状態なら $\pi_0=\mu/(\lambda+\mu)$ をすぐ出せるようにします。
- ポアソン過程・出生死滅過程・M/M/1 はすべて連続時間マルコフ連鎖の特別な場合です。

## よくある間違い

- × $Q$ を推移確率行列と同じように扱い、$\boldsymbol{\pi}Q=\boldsymbol{\pi}$ を解く → ○ 率の行列なので $\boldsymbol{\pi}Q=\boldsymbol{0}$ です。
- × 推移率 $q_{ij}$ は確率なので1以下 → ○ 単位時間あたりの率なので1を超えてもかまいません。
- × 滞在時間は任意の分布でよい → ○ マルコフ性を保つには指数分布でなければなりません。
