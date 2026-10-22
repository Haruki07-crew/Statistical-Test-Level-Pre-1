---
id: arma-model
title: ARMAモデル
reading: あーまもでる
en: Autoregressive moving average model
aliases: [自己回帰移動平均モデル, ARMA過程, "ARMA(p,q)", "ARMA(1,1)"]
chapter: 27
importance: A
tags: [時系列解析, ARMAモデル, 定常性, 反転可能性]
related: [ar-model, ma-model, stationarity-condition, invertibility, arima-model, aic]
charts:
  - type: custom
    id: line-data
    params:
      x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      series:
        - { label: "ARMA(1,1) φ=0.5, θ=0.4", y: [1, 0.6923, 0.3462, 0.1731, 0.0865, 0.0433, 0.0216, 0.0108, 0.0054, 0.0027, 0.0014] }
        - { label: "AR(1) φ=0.5", y: [1, 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.002, 0.001] }
      xlabel: "ラグ h"
      ylabel: "自己相関 ρ(h)"
    caption: ARMA(1,1) と AR(1)（同じ φ=0.5）の理論自己相関。ARMA(1,1) は ρ(1) だけが MA 部分の影響で大きくなり、ラグ2以降は AR(1) と同じ比率 0.5 で減衰する。
---

## 一言でいうと

AR（自分の過去の値）と MA（過去の雑音）を組み合わせたモデルです。
AR だけ・MA だけでは次数が大きくなる系列も、少ないパラメータで表せます。

## 定義・数式

$\varepsilon_t\sim WN(0,\sigma^2)$ として、

$$
X_t=c+\phi_1X_{t-1}+\cdots+\phi_pX_{t-p}+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
$$

を **ARMA($p,q$) モデル**といいます。後退作用素で書くと

$$
\phi(B)X_t=c+\theta(B)\varepsilon_t,\qquad
\phi(z)=1-\phi_1z-\cdots-\phi_pz^p,\quad \theta(z)=1+\theta_1z+\cdots+\theta_qz^q
$$

です（MA 部分は $+\theta$ の流儀）。

- **定常条件**：$\phi(z)=0$ の根がすべて単位円の外（AR 部分だけで決まる）。
- **反転可能条件**：$\theta(z)=0$ の根がすべて単位円の外（MA 部分だけで決まる）。
- $\phi(z)$ と $\theta(z)$ に共通の根がないこと（あると約分でき、次数を下げられる）。

定常なら平均は $\mu=c/(1-\phi_1-\cdots-\phi_p)$ です。

**ARMA(1,1)** $X_t=\phi X_{t-1}+\varepsilon_t+\theta\varepsilon_{t-1}$（$|\phi|<1$）の結果：

$$
\gamma(0)=\frac{1+2\phi\theta+\theta^2}{1-\phi^2}\sigma^2,\qquad
\gamma(1)=\frac{(1+\phi\theta)(\phi+\theta)}{1-\phi^2}\sigma^2,\qquad
\gamma(h)=\phi\,\gamma(h-1)\ \ (h\ge2)
$$

$$
\rho(1)=\frac{(1+\phi\theta)(\phi+\theta)}{1+2\phi\theta+\theta^2},\qquad \rho(h)=\phi^{h-1}\rho(1)\ \ (h\ge1)
$$

## 数式の読み解き

**ARMA(1,1) の導出**

まず、$X_t$ と雑音の共分散を求めます。$\varepsilon_t$ は過去の $X$ と無相関なので

$$
E[X_t\varepsilon_t]=\sigma^2,\qquad
E[X_t\varepsilon_{t-1}]=\phi E[X_{t-1}\varepsilon_{t-1}]+\theta\sigma^2=(\phi+\theta)\sigma^2
$$

次に、モデルの両辺に $X_t,\ X_{t-1},\ X_{t-h}$ を掛けて期待値をとります。

$$
\begin{aligned}
\gamma(0)&=\phi\gamma(1)+\sigma^2+\theta(\phi+\theta)\sigma^2\\
\gamma(1)&=\phi\gamma(0)+\theta\sigma^2\\
\gamma(h)&=\phi\gamma(h-1)\qquad(h\ge2)
\end{aligned}
$$

2本目を1本目に代入すると $\gamma(0)(1-\phi^2)=(1+2\phi\theta+\theta^2)\sigma^2$ となり、上の $\gamma(0)$ が得られます。
それを2本目に戻して整理すると $\gamma(1)$ の式になります。

- ラグ $q$ より先（ここでは $h\ge2$）は MA 部分の影響がなくなり、AR と同じ漸化式で減衰します。
- そのため ACF も PACF も切断せず、どちらも減衰します。これが ARMA の見分け方です。
- $\phi=-\theta$ だと $(1-\phi B)X_t=(1-\phi B)\varepsilon_t$ で約分でき、$X_t=\varepsilon_t$（ホワイトノイズ）になります。実際 $\rho(1)$ の分子に $\phi+\theta=0$ が現れます。

**MA(∞)・AR(∞) 表現**

$X_t=\sum_{j\ge0}\psi_j\varepsilon_{t-j}$ の係数は $\psi_0=1$、$\psi_j=(\phi+\theta)\phi^{j-1}\ (j\ge1)$ です。
反転可能なら $X_t=\sum_{j\ge1}\pi_jX_{t-j}+\varepsilon_t$、$\pi_j=(\phi+\theta)(-\theta)^{j-1}$ とも書けます。

## 具体例

**例1：ARMA(1,1) の分散と自己相関**

$\phi=0.5,\ \theta=0.4,\ \sigma^2=1$ とします。$|\phi|<1$ で定常、$|\theta|<1$ で反転可能です。

$$
\gamma(0)=\frac{1+2(0.5)(0.4)+0.16}{1-0.25}=\frac{1.56}{0.75}=2.08,\qquad
\gamma(1)=\frac{(1.2)(0.9)}{0.75}=1.44
$$

$$
\rho(1)=\frac{1.44}{2.08}\approx0.6923,\quad \rho(2)=0.5\times0.6923\approx0.3462,\quad \rho(3)\approx0.1731
$$

**例2：$\psi$ 重みで検算する**

$\psi_j$ は $1,\ 0.9,\ 0.45,\ 0.225,\ \dots$ なので

$$
\gamma(0)=\sum_{j\ge0}\psi_j^2=1+0.81\sum_{j\ge0}0.25^j=1+\frac{0.81}{0.75}=2.08
$$

で、例1と一致します。AR(∞) 表現の係数は $\pi_1=0.9,\ \pi_2=-0.36,\ \pi_3=0.144$ です。

**例3：自己相関から係数を逆算する**

ARMA(1,1) と仮定した系列で、$\rho(1)=9/13\approx0.6923$、$\rho(2)=4.5/13\approx0.3462$ がわかったとします。
$h\ge2$ では $\rho(h)=\phi\rho(h-1)$ なので、まず

$$
\phi=\frac{\rho(2)}{\rho(1)}=0.5
$$

です。次に $\rho(1)$ の式に $\phi=0.5$ を入れると $\dfrac{(1+0.5\theta)(0.5+\theta)}{1+\theta+\theta^2}=\dfrac{9}{13}$ で、整理すると

$$
10\theta^2-29\theta+10=0\quad\Longrightarrow\quad \theta=\frac{29\pm21}{20}=0.4\ \text{または}\ 2.5
$$

反転可能な $\theta=0.4$ を採用します。MA(1) と同じく、$\theta$ と $1/\theta$ が同じ自己相関を与えるため解が2つ出ます。

**なぜ ARMA を使うのか（節約の原理）**

例2のように、ARMA(1,1) を純粋な AR で表すと $\pi_j$ が無限に続き、AR で近似するには高い次数が必要です。
逆に MA で表しても $\psi_j$ が無限に続きます。
ARMA ならパラメータ2個で同じ構造を表せるので、推定の誤差が小さくなります。
少ないパラメータで十分に表せるモデルを選ぶ考え方を**節約の原理**（パーシモニー）といいます。

## 現場での使われ方

- **経済統計**：四半期の成長率などを、AIC や BIC で ARMA($p,q$) の次数を選んで当てはめ、短期予測に使います。
- **製造・品質管理**：工程の測定値に「ゆっくり戻る傾向（AR）」と「一時的なショックの余韻（MA）」が混ざっているとき、ARMA(1,1) で表します。
- **金融**：収益率の弱い自己相関を ARMA で表し、残りの分散の変動を別のモデルで扱う、という2段構えがよく使われます。

## グラフ・イメージ

同じ φ=0.5 の AR(1) と比べると、ARMA(1,1) はラグ1の自己相関が θ の分だけ大きくなっています。
ラグ2以降はどちらも1期ごとに 0.5 倍で減衰し、2本の線は平行に（対数で見ると同じ傾きで）下がっていきます。

## 試験のポイント

- 定常性は AR 部分、反転可能性は MA 部分の特性方程式で判定します。混同しないことが重要です。
- ARMA(1,1) の $\gamma(0),\gamma(1)$ の導出（$E[X_t\varepsilon_{t-1}]=(\phi+\theta)\sigma^2$ を使う）は典型的な計算問題です。
- 識別：ACF も PACF も切断しない → ARMA。次数は AIC などの情報量規準で決めます。
- 共通因子（$\phi=-\theta$ など）があるとパラメータが識別できない点にも注意します。

## よくある間違い

- × ARMA の定常性を MA 部分も含めて判定する → ○ 定常性に関係するのは AR 部分だけです。
- × $E[X_t\varepsilon_{t-1}]=\theta\sigma^2$ とする → ○ $X_{t-1}$ 経由の分も入り、$(\phi+\theta)\sigma^2$ です。
- × ARMA(1,1) の自己相関は $\rho(h)=\phi^h$ → ○ $\rho(1)$ が MA の影響で変わり、$\rho(h)=\phi^{h-1}\rho(1)$ です。
