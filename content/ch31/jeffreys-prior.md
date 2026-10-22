---
id: jeffreys-prior
title: ジェフリーズの事前分布
reading: じぇふりーずのじぜんぶんぷ
en: Jeffreys prior
aliases: [ジェフリーズ事前分布, ジェフリーズの無情報事前分布]
chapter: 31
importance: B
tags: [無情報事前分布, フィッシャー情報量, 変換不変性]
related: [noninformative-prior, fisher-information, prior-distribution, beta-binomial-model, change-of-variables, beta-distribution]
charts:
  - type: pdf
    dist: beta
    params: { a: 0.5, b: 0.5 }
    x: { min: 0, max: 1 }
    compare:
      - { dist: beta, params: { a: 1, b: 1 }, label: "一様分布 Be(1,1)" }
    caption: 成功確率に対するジェフリーズの事前分布 Be(1/2, 1/2)。0 と 1 の近くで高くなる U 字形で、確率の小さな変化がデータで見分けやすい端の方に重みを置いている。
---

## 一言でいうと

フィッシャー情報量の平方根に比例させて作る無情報事前分布です。
パラメータをどう変換しても同じ規則で作れば同じ事前分布になる（**変換不変**）という性質をもちます。

## 定義・数式

1つのパラメータ $\theta$ をもつモデル $f(x \mid \theta)$ のフィッシャー情報量（1観測あたり）を

$$
I(\theta) = E\!\left[\left(\frac{\partial}{\partial\theta}\log f(X \mid \theta)\right)^2\right] = -E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X \mid \theta)\right]
$$

とします（後者の等号は正則条件のもとで成り立ちます）。**ジェフリーズの事前分布**は

$$
\pi_J(\theta) \propto \sqrt{I(\theta)}
$$

です。多次元のパラメータでは $\pi_J(\boldsymbol{\theta}) \propto \sqrt{\det I(\boldsymbol{\theta})}$ とします。非正則になる場合もあります。

代表例：

| モデル | $I(\theta)$ | ジェフリーズの事前分布 |
|---|---|---|
| ベルヌーイ・二項（成功確率 $\theta$） | $\dfrac{1}{\theta(1-\theta)}$ | $\theta^{-1/2}(1-\theta)^{-1/2}$、つまり $\mathrm{Be}(1/2,1/2)$ |
| ポアソン（平均 $\lambda$） | $\dfrac{1}{\lambda}$ | $\lambda^{-1/2}$（非正則） |
| 正規の平均 $\mu$（$\sigma$ 既知） | $\dfrac{1}{\sigma^2}$（定数） | $\propto 1$（平坦、非正則） |
| 正規の標準偏差 $\sigma$（$\mu$ 既知） | $\dfrac{2}{\sigma^2}$ | $\propto 1/\sigma$（非正則） |
| 指数（レート $\lambda$） | $\dfrac{1}{\lambda^2}$ | $\propto 1/\lambda$（非正則） |

## 数式の読み解き

**ベルヌーイでの導出**：$\log f(x \mid \theta) = x\log\theta + (1-x)\log(1-\theta)$ より

$$
\frac{\partial^2}{\partial\theta^2}\log f = -\frac{x}{\theta^2} - \frac{1-x}{(1-\theta)^2}, \qquad
I(\theta) = \frac{\theta}{\theta^2} + \frac{1-\theta}{(1-\theta)^2} = \frac{1}{\theta(1-\theta)}
$$

です（$E[X]=\theta$ を代入）。平方根をとると $\theta^{-1/2}(1-\theta)^{-1/2}$ となり、$\mathrm{Be}(1/2,1/2)$ の核です。

**ポアソンでの導出**：$\log f = x\log\lambda - \lambda - \log x!$、二階微分は $-x/\lambda^2$、期待値をとって $I(\lambda)=1/\lambda$。よって $\pi_J(\lambda)\propto\lambda^{-1/2}$ です。

**変換不変性**：$\phi = g(\theta)$（単調で微分可能）と変換すると、スコアの連鎖律から

$$
I_\phi(\phi) = I_\theta(\theta)\left(\frac{d\theta}{d\phi}\right)^2
\quad\Longrightarrow\quad
\sqrt{I_\phi(\phi)} = \sqrt{I_\theta(\theta)}\,\left|\frac{d\theta}{d\phi}\right|
$$

です。右辺は「$\theta$ のジェフリーズ事前分布を変数変換の公式で $\phi$ に移したもの」と同じです。つまり、どのパラメータ化で作っても矛盾しません。一様事前分布にはこの性質がありません。

**直感**：$I(\theta)$ が大きい場所は、$\theta$ が少し変わるだけでデータの分布が大きく変わる、つまり「見分けやすい」場所です。ジェフリーズの事前分布は、見分けやすさの尺度で測って一様になるように重みをつけています。

## 具体例

**例1：10 回中 3 回成功**

ジェフリーズ事前分布 $\mathrm{Be}(1/2,1/2)$ では、事後分布は $\mathrm{Be}(3.5,\ 7.5)$ です。

- 事後平均：$3.5/11 \approx 0.3182$、MAP：$2.5/9 \approx 0.2778$
- 95% 等裾信用区間：$[0.0927,\ 0.6058]$

一様事前分布 $\mathrm{Be}(1,1)$ では事後分布 $\mathrm{Be}(4, 8)$、事後平均 $\approx 0.3333$、区間 $[0.1093,\ 0.6097]$ です。$n=10$ でも差は小さく、ジェフリーズの方が最尤推定値 0.3 に少し近くなります。

**例2：成功 0 回の場合**

10 回中 0 回なら事後分布は $\mathrm{Be}(0.5,\ 10.5)$、事後平均 $0.5/11 \approx 0.0455$（一様事前では $1/12 \approx 0.0833$）、95% 等裾信用区間 $[0.00005,\ 0.2172]$ です。この区間は「ジェフリーズ区間」として、二項比率の区間推定でよく使われます。

**例3：ポアソン**

$n=5$、合計 20 のポアソンデータに $\pi_J(\lambda)\propto\lambda^{-1/2}$ をおくと、$\lambda^{-1/2}\cdot\lambda^{20}e^{-5\lambda} = \lambda^{20.5-1}e^{-5\lambda}$ より事後分布は $\mathrm{Ga}(20.5, 5)$（レート表記）。事後平均 4.1、95% 等裾信用区間 $[2.5215,\ 6.0561]$ です。

## 現場での使われ方

- **医療**：まれな副作用の発生率など、件数が非常に少ない比率の区間推定にジェフリーズ区間を使います。ワルド区間より被覆確率が名目値に近いことが知られています。
- **行政・調査**：主観を入れない「既定の」事前分布として、報告用のベイズ解析に採用されます。
- **製造・品質管理**：故障率（指数分布のレート）の推定で $1/\lambda$ 型の事前分布を使い、単位（時間・日）を変えても結論が変わらないようにします。

## グラフ・イメージ

実線がジェフリーズの事前分布 $\mathrm{Be}(1/2,1/2)$、点線が一様分布です。
ジェフリーズの事前分布は両端で高くなりますが、全体の面積は 1 で、データが少し入ればすぐに尤度に押し流される弱い事前分布です。

## 試験のポイント

- 二項・ポアソン・正規についてフィッシャー情報量を計算し、ジェフリーズの事前分布を導かせる形が典型です。
- 二項の場合 $\mathrm{Be}(1/2,1/2)$ になり、事後分布が $\mathrm{Be}(x+1/2,\ n-x+1/2)$ になることを覚えます。
- 変換不変性の式 $I_\phi = I_\theta (d\theta/d\phi)^2$ の意味を説明できるようにします。
- 正規の平均では平坦、尺度では $1/\sigma$ になり、無情報事前分布の表と一致します。

## よくある間違い

- × $\pi_J(\theta)\propto I(\theta)$ とする → ○ 平方根をとります。$\pi_J(\theta)\propto\sqrt{I(\theta)}$。
- × 二項のジェフリーズ事前分布を $\mathrm{Be}(1,1)$ とする → ○ $\mathrm{Be}(1/2,1/2)$ です。$\mathrm{Be}(1,1)$ は一様事前分布です。
- × ジェフリーズの事前分布は常に正則 → ○ ポアソン・正規・指数では非正則です。事後分布が正規化できるか確認します。
- × $n$ 個のデータのフィッシャー情報量 $nI(\theta)$ を使うと結果が変わる → ○ 定数倍は比例記号に吸収されるので同じです。
