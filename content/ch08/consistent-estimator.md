---
id: consistent-estimator
title: 一致推定量
reading: いっちすいていりょう
en: Consistent estimator
aliases: [一致性]
chapter: 8
importance: A
tags: [推定, 一致性, 確率収束, 大標本]
related: [convergence-in-probability, weak-law-of-large-numbers, chebyshev-inequality, mean-squared-error, unbiased-estimator, asymptotic-normality-of-mle]
charts:
  - type: custom
    id: lln-path
    params: { p: 0.3, n: 1000, paths: 5 }
    sliders:
      - { param: p, label: "成功確率 p", min: 0.05, max: 0.95, step: 0.05 }
    caption: 成功確率 p のベルヌーイ試行で、標本比率 p̂ を n の関数として5本描いたもの。どの経路も n が増えると p に近づいて離れなくなる。これが一致性（確率収束）のイメージ。
---

## 一言でいうと

標本サイズを大きくしていくと、真の母数にいくらでも近づいていく推定量です。
「データを十分集めれば正しい答えにたどり着く」という、推定量として最低限ほしい性質です。

## 定義・数式

標本サイズ $n$ の推定量 $\hat\theta_n$ が、任意の $\varepsilon>0$ と、$\theta$ のとりうるすべての値について

$$
\lim_{n\to\infty} P_\theta\big(|\hat\theta_n - \theta| \ge \varepsilon\big) = 0
$$

を満たすとき、$\hat\theta_n$ を $\theta$ の**一致推定量**といいます。つまり $\hat\theta_n$ が $\theta$ に**確率収束**する（$\hat\theta_n \xrightarrow{p} \theta$）ことです。

**十分条件（よく使う判定法）**：平均二乗誤差が 0 に収束すれば一致推定量です。

$$
\mathrm{MSE}(\hat\theta_n) = V[\hat\theta_n] + \{\mathrm{Bias}(\hat\theta_n)\}^2 \to 0
\quad\Longrightarrow\quad \hat\theta_n \xrightarrow{p} \theta
$$

とくに「$E[\hat\theta_n]\to\theta$ かつ $V[\hat\theta_n]\to 0$」なら一致推定量です。

## 数式の読み解き

- $\varepsilon$：許容する誤差の幅。どんなに小さい幅を決めても、$n$ を大きくすれば外れる確率が 0 に近づく、という意味です。
- 一致性は「$n\to\infty$ の極限」の性質で、有限の $n$ での精度は何も保証しません。

**十分条件の証明**はマルコフの不等式を $(\hat\theta_n-\theta)^2$ に当てはめるだけです。

$$
P(|\hat\theta_n-\theta|\ge\varepsilon) = P\big((\hat\theta_n-\theta)^2\ge\varepsilon^2\big) \le \frac{E[(\hat\theta_n-\theta)^2]}{\varepsilon^2} = \frac{\mathrm{MSE}(\hat\theta_n)}{\varepsilon^2}
$$

右辺が 0 に収束すれば、左辺も 0 に収束します。

**代表例**

- 標本平均 $\bar X$：$V[\bar X]=\sigma^2/n\to0$ で不偏なので一致（大数の弱法則そのもの）。
- 不偏分散 $S^2$ も $n$ で割った $\hat\sigma^2$ も一致推定量です。$\hat\sigma^2$ はバイアス $-\sigma^2/n$ をもちますが、$n\to\infty$ で消えます。
- **連続写像定理**により、$\hat\theta_n\xrightarrow{p}\theta$ で $g$ が連続なら $g(\hat\theta_n)\xrightarrow{p}g(\theta)$ です。$S=\sqrt{S^2}$ も $\sigma$ の一致推定量です。
- 正則条件のもとで、最尤推定量は一致推定量です。

## 具体例

**例1：標本比率（チェビシェフの不等式で評価）**

不良率 $p=0.3$ の工程から $n$ 個を検査し、標本比率 $\hat p$ で推定します。誤差 0.05 以上になる確率は、チェビシェフの不等式で

$$
P(|\hat p - p|\ge 0.05) \le \frac{p(1-p)}{n\cdot 0.05^2} = \frac{0.21}{0.0025\,n}
$$

$n=100$ なら上限 0.84、$n=1000$ なら上限 0.084 です。$n\to\infty$ で 0 に向かうので一致推定量です。
（正規近似で計算すると、$n=100$ で約 0.2752、$n=1000$ で約 0.0006 です。チェビシェフの上限は粗いですが、一致性の証明には十分です。）

**例2：一様分布の最大値**

$U(0,\theta)$ からの標本の最大値 $X_{(n)}$ は $\theta$ を超えないので、$0<\varepsilon<\theta$ に対して

$$
P(|X_{(n)}-\theta|\ge\varepsilon) = P(X_{(n)}\le\theta-\varepsilon) = \Big(1-\frac{\varepsilon}{\theta}\Big)^n \to 0
$$

$\theta=10,\ \varepsilon=0.5$ なら $0.95^n$ で、$n=10$ で約 0.5987、$n=50$ で約 0.0769、$n=100$ で約 0.0059 です。$X_{(n)}$ は不偏ではありませんが一致推定量です。

**例3：不偏だが一致でない推定量**

$\hat\mu = X_1$（最初の1個だけを使う）は $E[X_1]=\mu$ で不偏ですが、$n$ を増やしても分布が変わらないので一致推定量ではありません。

## 現場での使われ方

- **IT・Webサービス**：アクセスログのような大量データでは、推定量が一致性をもつことが「データを増やせば精度が上がる」根拠になります。逆に、選択バイアスのあるデータでは一致性が崩れ、いくら件数を増やしても真の値に近づきません。
- **金融・保険**：保険料率の算定で、契約件数が増えるほど事故率の推定値が真の値に安定することを前提にしています（大数の法則）。
- **医療・疫学**：観察研究で交絡を調整しない推定量は一致性をもたないことがあり、症例数を増やしても誤った結論に収束します。

## グラフ・イメージ

各折れ線は、ベルヌーイ試行を1回ずつ増やしたときの標本比率の推移です。
最初は大きく揺れますが、$n$ が増えるとどの経路も $p$ の水平線の近くに集まります。スライダーで $p$ を変えても同じ性質が見られます。

## 試験のポイント

- 一致性の定義（確率収束）を式で書けるようにします。
- 証明には「MSE → 0」または「バイアス → 0 かつ分散 → 0」と、チェビシェフ（マルコフ）の不等式を使うのが定番です。
- 不偏性と一致性は独立した性質です。「不偏だが一致でない」（$X_1$）、「不偏でないが一致」（$\hat\sigma^2$, $X_{(n)}$）の例を言えるようにします。
- 連続写像定理・スラツキーの定理と組み合わせて、関数や比の一致性を示す問題が出ます。

## よくある間違い

- × 一致推定量は不偏推定量 → ○ 一致性は極限での性質で、有限の $n$ ではバイアスがあってもかまいません。
- × MSE → 0 は一致性の必要十分条件 → ○ 十分条件にすぎません。期待値が存在しない一致推定量もあります。
- × $n$ を増やせばどんな推定量も真の値に近づく → ○ $X_1$ のように標本の一部しか使わない推定量や、偏ったデータからの推定量は近づきません。
