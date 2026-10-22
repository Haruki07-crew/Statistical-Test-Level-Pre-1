---
id: normal-normal-model
title: 正規・正規モデル
reading: せいき・せいきもでる
en: Normal-normal model
aliases: [正規正規モデル, 正規事前分布と正規尤度]
chapter: 31
importance: A
tags: [共役事前分布, 正規分布, 精度, 縮小推定]
related: [normal-distribution, conjugate-prior, posterior-mean-map, credible-interval, predictive-distribution, hierarchical-bayes-model]
charts:
  - type: custom
    id: prior-posterior
    params: { model: normal-normal, mu0: 50, tau0: 10, sigma: 15, n: 9, xbar: 60 }
    sliders:
      - { param: tau0, label: "事前の標準偏差 τ₀", min: 1, max: 40, step: 1 }
      - { param: n, label: "標本サイズ n", min: 1, max: 100, step: 1 }
      - { param: xbar, label: "標本平均 x̄", min: 30, max: 80, step: 0.5 }
    caption: 事前 N(μ₀, τ₀²) と、分散既知の正規データ n 個の標本平均 x̄ から、事後 N(μₙ, τₙ²) を描く。図の下の「重み」はデータ側と事前側の精度の割合。
---

## 一言でいうと

分散が既知の正規分布の平均 $\mu$ に、正規分布の事前分布をおくモデルです。
事後分布も正規分布になり、**精度（分散の逆数）が足し算**で増え、事後平均は事前平均と標本平均の精度による重み付き平均になります。

## 定義・数式

$$
x_1,\dots,x_n \mid \mu \overset{\text{iid}}{\sim} N(\mu, \sigma^2)\ （\sigma^2\ 既知）, \qquad \mu \sim N(\mu_0, \tau_0^2)
$$

とします。このとき

$$
\mu \mid \boldsymbol{x} \sim N(\mu_n, \tau_n^2), \qquad
\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad
\mu_n = \frac{\dfrac{1}{\tau_0^2}\mu_0 + \dfrac{n}{\sigma^2}\bar{x}}{\dfrac{1}{\tau_0^2} + \dfrac{n}{\sigma^2}}
$$

です。精度 $\kappa_0 = 1/\tau_0^2$（事前）、$\kappa_x = n/\sigma^2$（データ）と書くと、

$$
\kappa_n = \kappa_0 + \kappa_x, \qquad \mu_n = \frac{\kappa_0}{\kappa_n}\mu_0 + \frac{\kappa_x}{\kappa_n}\bar{x}
$$

正規分布は左右対称・単峰なので、事後平均・事後中央値・MAP はすべて $\mu_n$ で、95% 信用区間は $\mu_n \pm 1.96\,\tau_n$ です。

## 数式の読み解き

**導出（平方完成）**：尤度は $\bar{x}$ を通してだけ $\mu$ に依存し、

$$
f(\boldsymbol{x} \mid \mu) \propto \exp\!\left\{-\frac{n(\mu-\bar{x})^2}{2\sigma^2}\right\}
$$

です（$\sum(x_i-\mu)^2 = \sum(x_i-\bar{x})^2 + n(\bar{x}-\mu)^2$ を使い、$\mu$ を含まない項を捨てる）。事前分布を掛けると

$$
\begin{aligned}
\pi(\mu \mid \boldsymbol{x}) &\propto \exp\!\left\{-\frac{\kappa_x(\mu-\bar{x})^2 + \kappa_0(\mu-\mu_0)^2}{2}\right\} \\
&\propto \exp\!\left\{-\frac{(\kappa_0+\kappa_x)\mu^2 - 2(\kappa_0\mu_0+\kappa_x\bar{x})\mu}{2}\right\} \\
&\propto \exp\!\left\{-\frac{\kappa_n}{2}\left(\mu - \frac{\kappa_0\mu_0+\kappa_x\bar{x}}{\kappa_n}\right)^2\right\}
\end{aligned}
$$

となり、平均 $\mu_n$、分散 $1/\kappa_n$ の正規分布の核です。

- **精度の加算**：情報量が「事前の情報 + データの情報」と足し算になります。事後分散 $\tau_n^2$ は $\tau_0^2$ と $\sigma^2/n$ のどちらよりも小さくなります。
- **縮小（shrinkage）**：事後平均は $\bar{x}$ を事前平均 $\mu_0$ の方へ $\kappa_0/\kappa_n$ の割合だけ引き寄せた値です。
- **極限**：$n\to\infty$ または $\tau_0^2\to\infty$（事前が平ら）で $\mu_n \to \bar{x}$、$\tau_n^2 \to \sigma^2/n$ となり、頻度論の結果と数値的に一致します。

**データ1個ずつの逐次更新**：観測を1個得るたびに精度が $1/\sigma^2$ ずつ増えます。$n$ 個を順に更新しても、$\bar{x}$ でまとめて更新しても同じ結果になります。これは $\bar{x}$ が $\mu$ の十分統計量だからです。

**分散も未知の場合**：$\sigma^2$ が未知なら、このモデルはそのままでは使えません。平均と分散の同時事前分布として正規・逆ガンマ型を使うと共役になり、$\mu$ の周辺事後分布は $t$ 分布になります。準1級では分散既知の場合の計算が中心です。

**予測分布**：新しい観測 $\tilde{x} = \mu + \varepsilon$（$\varepsilon \sim N(0,\sigma^2)$ は $\mu$ と独立）なので

$$
\tilde{x} \mid \boldsymbol{x} \sim N(\mu_n,\ \sigma^2 + \tau_n^2)
$$

です。観測のばらつき $\sigma^2$ と、$\mu$ の不確かさ $\tau_n^2$ が足し合わされます。

## 具体例

**例1：学力テストの平均点**

ある学校の平均点 $\mu$ について、過去の傾向から事前分布を $N(50, 10^2)$ とします。個人の点数の標準偏差は $\sigma=15$ で既知、9 人の平均が $\bar{x}=60$ でした。

$$
\kappa_0 = \frac{1}{100} = 0.01, \qquad \kappa_x = \frac{9}{225} = 0.04, \qquad \kappa_n = 0.05
$$

$$
\tau_n^2 = \frac{1}{0.05} = 20, \qquad \mu_n = \frac{0.01 \times 50 + 0.04 \times 60}{0.05} = \frac{0.5 + 2.4}{0.05} = 58
$$

データの重みは $0.04/0.05 = 0.8$、事前の重みは 0.2 です。
事後標準偏差 $\sqrt{20}\approx 4.4721$、95% 信用区間は $58 \pm 1.96\times 4.4721 = [49.2346,\ 66.7654]$、$P(\mu > 55 \mid \boldsymbol{x}) \approx 0.7488$ です。
比較として、事前情報を使わない 95% 信頼区間は $60 \pm 1.96 \times 5 = [50.2,\ 69.8]$ です。

**例2：次の1人の点数を予測する**

予測分布は $N(58,\ 225+20) = N(58, 245)$、標準偏差 $\approx 15.6525$ です。95% 予測区間は $[27.3211,\ 88.6789]$、70 点を超える予測確率は $\approx 0.2216$ です。

**例3：センサーの校正**

真値 $\mu$ の事前分布 $N(20, 1^2)$、測定誤差 $\sigma=2$、4 回の平均 $\bar{x}=21.5$ のとき、$\kappa_n = 1 + 4/4 = 2$、$\tau_n^2 = 0.5$、$\mu_n = (1\times 20 + 1\times 21.5)/2 = 20.75$。95% 信用区間は $[19.3641,\ 22.1359]$ です。事前とデータの精度が等しいので、ちょうど中間になります。

## 現場での使われ方

- **製造・品質管理**：測定器の読みを、製造仕様から決まる事前分布と組み合わせて真値を推定します（カルマンフィルタの1ステップと同じ計算）。
- **教育・行政**：受験者が少ない学校や小地域の平均値を、全体平均の方へ縮小して推定します（小地域推定）。
- **医療**：過去の試験から得た治療効果の推定値と標準誤差を事前分布とし、新しい試験の結果と精度で重み付けして統合します（ベイズ流のメタアナリシスの基本形）。
- **金融・保険**：個々の契約群の損害率を、全体の損害率と精度で重み付けして見積もります（ビュールマンの信頼度理論と同じ形）。

## グラフ・イメージ

事前分布（点線）、尤度（細線）、事後分布（太線）を重ねています。
$n$ を大きくするとデータの重みが 1 に近づき、事後分布は $\bar{x}$ の位置で細くなります。
$\tau_0$ を小さくすると事前分布が強くなり、事後平均が $\mu_0=50$ の方へ引き寄せられます。

## 試験のポイント

- 事後分布の平均・分散を「精度の和」と「精度による重み付き平均」で答えさせるのが典型です。
- 平方完成による導出の流れを書けるようにしておきます。
- 事後分散はデータの値 $\bar{x}$ に依存しない（$n$ と $\sigma^2$、$\tau_0^2$ だけで決まる）ことを押さえます。
- 予測分布の分散は $\sigma^2 + \tau_n^2$ です。$\tau_n^2$ だけにしないよう注意します。
- 事前分布を平らにした極限で、信用区間が信頼区間 $\bar{x}\pm 1.96\sigma/\sqrt{n}$ と一致することが問われます。

## よくある間違い

- × 事後分散を $\tau_0^2 + \sigma^2/n$ とする → ○ 足すのは分散ではなく精度です。$1/\tau_n^2 = 1/\tau_0^2 + n/\sigma^2$。
- × 重みに分散を使い、分散の大きい方を重視する → ○ 精度（分散の逆数）が大きい、つまり確かな方を重視します。
- × データの精度を $1/\sigma^2$ とする → ○ 標本平均の精度は $n/\sigma^2$ です。
- × 予測分布の分散を $\sigma^2$ だけとする → ○ $\mu$ の不確かさ $\tau_n^2$ も加わります。
