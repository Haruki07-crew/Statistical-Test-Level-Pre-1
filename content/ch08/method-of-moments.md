---
id: method-of-moments
title: モーメント法
reading: もーめんとほう
en: Method of moments
aliases: [積率法, モーメント推定量, 積率推定量]
chapter: 8
importance: A
tags: [推定, モーメント, 標本モーメント]
related: [maximum-likelihood-estimator, point-estimation, gamma-distribution, uniform-distribution, consistent-estimator, moment-generating-function]
charts:
  - type: pdf
    dist: gamma
    params: { shape: 2, scale: 2 }
    sliders:
      - { param: shape, label: "形状 α", min: 0.5, max: 10, step: 0.5 }
      - { param: scale, label: "尺度 β", min: 0.5, max: 5, step: 0.5 }
    x: { min: 0, max: 25 }
    caption: 例1でモーメント法により推定したガンマ分布（形状2、尺度2、平均4、分散8）。α と β を動かして平均 αβ と分散 αβ² が変わる様子を見ると、2つのモーメントから2つの母数が決まる理由がわかる。
---

## 一言でいうと

「母集団のモーメント（平均、2乗の平均など）を、標本で計算したモーメントに等しいとおいて、母数について解く」推定法です。
計算が簡単で、最尤法が解析的に解けない分布でもすぐに推定値が得られます。

## 定義・数式

母数 $\theta=(\theta_1,\dots,\theta_k)$ をもつ分布の、原点まわりの $j$ 次モーメントを

$$
\mu_j(\theta) = E_\theta[X^j]
$$

とし、標本の $j$ 次モーメントを

$$
m_j = \frac1n\sum_{i=1}^n X_i^j
$$

とします。連立方程式

$$
\mu_j(\theta) = m_j \qquad (j=1,\dots,k)
$$

を $\theta$ について解いたものを**モーメント推定量**といいます。
2次の式は、平均まわりの形 $V_\theta[X] = \frac1n\sum(X_i-\bar X)^2 = m_2 - m_1^2$ で書いても同じです。

**主な結果**

| 分布 | 使う式 | モーメント推定量 |
|---|---|---|
| ポアソン $Po(\lambda)$ | $E[X]=\lambda$ | $\hat\lambda=\bar X$ |
| 指数（密度 $\lambda e^{-\lambda x}$） | $E[X]=1/\lambda$ | $\hat\lambda=1/\bar X$ |
| 正規 $N(\mu,\sigma^2)$ | $E[X]=\mu,\ V[X]=\sigma^2$ | $\hat\mu=\bar X,\ \hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$ |
| 一様 $U(0,\theta)$ | $E[X]=\theta/2$ | $\hat\theta=2\bar X$ |
| ガンマ（形状 $\alpha$、尺度 $\beta$） | $E[X]=\alpha\beta,\ V[X]=\alpha\beta^2$ | $\hat\alpha=\bar X^2/\hat\sigma^2,\ \hat\beta=\hat\sigma^2/\bar X$ |

ここで $\hat\sigma^2=\frac1n\sum(X_i-\bar X)^2$ です（$n-1$ で割る流儀もあり、大標本では差は無視できます）。

## 数式の読み解き

- 未知の母数が $k$ 個なら、1次から $k$ 次までのモーメントで $k$ 本の式を作ります。
- 大数の法則により $m_j\xrightarrow{p}\mu_j(\theta)$ です。母数がモーメントの連続関数で表せれば、連続写像定理によりモーメント推定量は**一致推定量**です。
- 一方、最尤推定量に比べて効率が劣ることが多く、不偏とも限りません。

**ガンマ分布の導出**（形状 $\alpha$、尺度 $\beta$。密度 $\frac{x^{\alpha-1}e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}$。レート $\lambda=1/\beta$ で書く流儀もあります）

$$
\alpha\beta = \bar X,\qquad \alpha\beta^2 = \hat\sigma^2
$$

2式目を1式目で割ると $\beta = \hat\sigma^2/\bar X$。これを1式目に代入して $\alpha = \bar X/\beta = \bar X^2/\hat\sigma^2$ です。
ガンマ分布の最尤推定は、ディガンマ関数を含む方程式になって解析的に解けないため、モーメント法の手軽さが生きる例です。

**一様分布 $U(a,b)$**：$E[X]=\frac{a+b}{2}$、$V[X]=\frac{(b-a)^2}{12}$ より

$$
\frac{a+b}{2}=\bar X,\quad \frac{b-a}{2\sqrt3}=\hat\sigma
\quad\Longrightarrow\quad \hat a = \bar X-\sqrt3\,\hat\sigma,\quad \hat b=\bar X+\sqrt3\,\hat\sigma
$$

## 具体例

**例1：修理時間のガンマ分布**

ある機器の修理時間について、$\bar x=4$（時間）、$\hat\sigma^2=8$ が得られたとします。ガンマ分布を当てはめると

$$
\hat\alpha = \frac{4^2}{8} = 2,\qquad \hat\beta = \frac{8}{4} = 2\quad(\text{レート表記なら }\hat\lambda = 0.5)
$$

**例2：一様分布の上端（最尤推定量と比べる）**

$U(0,\theta)$ からのデータ 2.1, 7.4, 5.0, 3.3, 9.2 では $\bar x=5.4$ なので、モーメント推定値は $2\bar x=10.8$。最尤推定値は最大値 $9.2$ です。
モーメント推定量 $2\bar X$ は不偏で分散 $\theta^2/(3n)$、最大値に基づく不偏推定量 $\frac{n+1}{n}X_{(n)}$（ここでは $11.04$）は分散 $\theta^2/\{n(n+2)\}$ で、後者のほうが精度が高くなります。
また、モーメント推定値が観測された最大値より小さくなる（データと矛盾する）ことも起こりえます。

**例3：一様分布 $U(a,b)$**

同じデータで $a,b$ ともに未知とすると、$\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2 = 6.78$ より

$$
\hat a = 5.4 - \sqrt{3\times6.78}\approx0.8900,\qquad \hat b = 5.4 + \sqrt{3\times6.78}\approx9.9100
$$

## 現場での使われ方

- **金融・保険**：保険金の支払額の分布（ガンマ、対数正規、パレートなど）の母数を、平均と分散からすぐに見積もります。最尤法の数値計算の初期値としても使います。
- **製造・品質管理**：工程データの平均と分散だけが記録されている場合、そこから分布の母数を逆算して不良率を見積もります。
- **IT・Webサービス**：アクセス数のように分散が平均より大きいデータに負の二項分布を当てはめるとき、平均と分散の2式から母数を推定します。

## グラフ・イメージ

例1で推定したガンマ分布（形状2、尺度2）の密度です。
形状 α を増やすと平均と分散がともに増え、尺度 β を増やすと平均は β 倍、分散は β² 倍になります。この2つの関係を逆に解いたのがモーメント推定量です。

## 試験のポイント

- 母数が2つの分布（ガンマ、ベータ、一様 $U(a,b)$、負の二項、二項の $n,p$ がともに未知など）で、2本の式を立てて解く問題が典型です。
- 正規・ポアソン・指数ではモーメント推定量と最尤推定量が一致します。一様分布 $U(0,\theta)$ では一致しません。比較させる問題が出ます。
- モーメント推定量は一致推定量だが、一般には有効ではないことを押さえます。
- 分散の式で $n$ で割るか $n-1$ で割るかは流儀があります。問題文の指定に従います。

## よくある間違い

- × 2次のモーメントの式に $E[X^2]=\sigma^2$ を使う → ○ $E[X^2]=\sigma^2+\mu^2$ です。平均まわりで書くなら $V[X]$ と $m_2-m_1^2$ を等しくおきます。
- × モーメント推定量は不偏 → ○ $1/\bar X$ や $\bar X^2/\hat\sigma^2$ のような非線形な関数は一般に不偏ではありません。
- × モーメント推定値は必ず母数空間の中に入る → ○ ベータ分布で $\hat\sigma^2$ が大きすぎると負の値が出る、など範囲外になることがあります。
