---
id: gamma-distribution
title: ガンマ分布
reading: がんまぶんぷ
en: Gamma distribution
aliases: [Γ分布, "Ga(α,β)", アーラン分布]
chapter: 6
importance: A
tags: [連続型分布, 再生性, 待ち時間, ガンマ関数]
related: [exponential-distribution, chi-squared-distribution, poisson-distribution, beta-distribution, gamma-poisson-model, moment-generating-function]
charts:
  - type: pdf
    dist: gamma
    params: { shape: 2, scale: 1 }
    sliders:
      - { param: shape, label: "形状 α", min: 0.5, max: 10, step: 0.5 }
      - { param: scale, label: "尺度 β", min: 0.5, max: 5, step: 0.5 }
    x: { min: 0, max: 20 }
    compare:
      - { dist: expon, params: { rate: 1 }, label: "指数分布 Exp(1)（α=1, β=1）" }
    caption: α=1 で指数分布、α<1 で x=0 に向かって発散、α>1 で山ができる。α を大きくすると山が右へ移り、左右対称に近づく。β は横軸の目盛りを伸ばすだけ。
  - type: survival-hazard
    dist: gamma
    params: { shape: 2, scale: 1 }
    sliders:
      - { param: shape, label: "形状 α", min: 0.5, max: 5, step: 0.5 }
    caption: α>1 ではハザードが増加（劣化型）、α<1 では減少（初期故障型）、α=1 で一定（指数分布）になる。
---

## 一言でいうと

指数分布の待ち時間を何個も足し合わせた「$\alpha$ 回目の出来事が起こるまでの待ち時間」を表す連続型分布です。
指数分布もカイ二乗分布もこの分布の特別な場合で、正の値をとるデータ（所要時間、損害額、降水量など）のモデルとして広く使われます。

## 定義・数式

この項では**形状 $\alpha>0$、尺度 $\beta>0$** の流儀を採用します。密度関数は

$$
f(x)=\frac{1}{\Gamma(\alpha)\beta^{\alpha}}\,x^{\alpha-1}e^{-x/\beta},\qquad x>0
$$

で、$X\sim\mathrm{Ga}(\alpha,\beta)$ と書きます。ここで $\Gamma(\alpha)=\int_0^\infty u^{\alpha-1}e^{-u}du$ はガンマ関数です。

もう一方の流儀として、**レート $\lambda=1/\beta$** を使って

$$
f(x)=\frac{\lambda^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x}
$$

と書くこともあります（ベイズ統計ではこちらが多い）。$\mathrm{Ga}(2,3)$ と書かれていたら、第2引数が尺度かレートかを必ず確認してください。

| 項目 | 尺度 $\beta$ 表記 | レート $\lambda$ 表記 |
|---|---|---|
| 台 | $(0,\infty)$ | 同左 |
| 期待値 | $\alpha\beta$ | $\alpha/\lambda$ |
| 分散 | $\alpha\beta^2$ | $\alpha/\lambda^2$ |
| モーメント母関数 | $(1-\beta t)^{-\alpha}$（$t<1/\beta$） | $\left(\dfrac{\lambda}{\lambda-t}\right)^{\alpha}$（$t<\lambda$） |
| 特性関数 | $(1-i\beta t)^{-\alpha}$ | $(1-it/\lambda)^{-\alpha}$ |
| 最頻値 | $(\alpha-1)\beta$（$\alpha\ge1$） | $(\alpha-1)/\lambda$ |

## 数式の読み解き

- $\alpha$（形状）：分布の形を決めます。整数のときは「$\alpha$ 回目の到着までの待ち時間」と解釈できます。
- $\beta$（尺度）：横軸の目盛りの単位です。$X\sim\mathrm{Ga}(\alpha,1)$ なら $\beta X\sim\mathrm{Ga}(\alpha,\beta)$ です。
- ガンマ関数の性質 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$、$\Gamma(n)=(n-1)!$、$\Gamma(1/2)=\sqrt\pi$ を使います。

**MGF の導出**。$t<1/\beta$ のとき、$x(1/\beta-t)=u$ と置換すると

$$
M(t)=\frac{1}{\Gamma(\alpha)\beta^\alpha}\int_0^\infty x^{\alpha-1}e^{-x(1/\beta-t)}dx=\frac{1}{\Gamma(\alpha)\beta^\alpha}\cdot\frac{\Gamma(\alpha)}{(1/\beta-t)^{\alpha}}=(1-\beta t)^{-\alpha}
$$

キュムラント母関数 $K(t)=-\alpha\log(1-\beta t)$ を微分すると $K'(t)=\alpha\beta/(1-\beta t)$、$K''(t)=\alpha\beta^2/(1-\beta t)^2$ なので、$E[X]=\alpha\beta$、$V[X]=\alpha\beta^2$ です。

**再生性**。$X_1\sim\mathrm{Ga}(\alpha_1,\beta)$、$X_2\sim\mathrm{Ga}(\alpha_2,\beta)$ が独立なら、MGF の積 $(1-\beta t)^{-(\alpha_1+\alpha_2)}$ より

$$
X_1+X_2\sim\mathrm{Ga}(\alpha_1+\alpha_2,\ \beta)
$$

です。**尺度 $\beta$ が共通**であることが条件です。

**他の分布との関係**。

- $\alpha=1$：指数分布 $\mathrm{Ga}(1,\beta)=\mathrm{Exp}(1/\beta)$。よって独立な $\mathrm{Exp}(\lambda)$ の $n$ 個の和は $\mathrm{Ga}(n,1/\lambda)$ です（整数形状のものをアーラン分布ともいいます）。
- $\alpha=k/2,\ \beta=2$：自由度 $k$ のカイ二乗分布 $\chi^2(k)$。
- ベータ分布：$X\sim\mathrm{Ga}(a,\beta)$、$Y\sim\mathrm{Ga}(b,\beta)$ が独立なら $X/(X+Y)\sim\mathrm{Beta}(a,b)$ で、$X+Y$ と独立です。
- ポアソン分布：率 $\lambda$ のポアソン過程で、$n$ 回目の到着時刻 $T_n\sim\mathrm{Ga}(n,1/\lambda)$ について $P(T_n>t)=P(N(t)\le n-1)$ が成り立ちます。
- ベイズ統計：ポアソン分布の率パラメータの共役事前分布です（第31章）。

## 具体例

**例1：3人目の来店までの時間**

来店が平均2分に1人（$\lambda=0.5$/分）のポアソン過程に従うとき、3人目が来るまでの時間は $T\sim\mathrm{Ga}(3,\ \beta=2)$ です。

$$
E[T]=3\times2=6\ \text{分},\qquad V[T]=3\times2^2=12
$$

10分たっても3人目が来ない確率は、「10分間の来店数 $N\sim\mathrm{Po}(5)$ が2人以下」と同じなので

$$
P(T>10)=P(N\le2)=e^{-5}\left(1+5+\frac{5^2}{2}\right)=18.5e^{-5}\approx0.1247
$$

です。積分しなくても、ポアソン分布の和で求められます。

**例2：再生性とカイ二乗分布（試験に近い形）**

$X\sim\chi^2(4)=\mathrm{Ga}(2,2)$ と $Y\sim\mathrm{Ga}(1.5,2)$ が独立なら、$X+Y\sim\mathrm{Ga}(3.5,2)=\chi^2(7)$ です。平均は $3.5\times2=7$、分散は $3.5\times4=14$ で、$\chi^2(7)$ の平均 $k=7$、分散 $2k=14$ と一致します。

## 現場での使われ方

- **金融・保険**：1件あたりの保険金支払額のように、正の値で右に裾が長いデータをモデル化します。
- **製造・品質管理**：部品を予備も含めて $n$ 個使い切るまでの時間（待機冗長系の寿命）を、指数分布の和としてガンマ分布で表します。
- **医療**：ポアソン分布の発生率が人によってばらつくとき、そのばらつきをガンマ分布で表すと、件数の分布が負の二項分布になります（過分散のモデル）。
- **行政・調査**：日降水量など、0より大きく右に歪んだ気象データの分布として使われます。

## グラフ・イメージ

1つ目では、$\alpha=1$ のとき比較用の指数分布と重なることを確認してから、$\alpha$ を大きくしてみてください。山が右へ移りながら左右対称に近づきます（指数分布の和なので中心極限定理が働きます）。$\beta$ は形を変えず、横に伸ばすだけです。
2つ目では、$\alpha$ によってハザードが増加・一定・減少と変わる様子を見ます。

## 試験のポイント

- 採用しているパラメータ表記（尺度かレートか）を最初に確認し、平均・分散・MGF をその表記で書けるようにします。
- MGF の導出は「ガンマ関数の定義に戻す置換」が定番です。
- 再生性（尺度が共通のとき形状が足し算）、指数分布の和、カイ二乗分布との関係は頻出事項として押さえます。
- $P(T_n>t)=P(N(t)\le n-1)$ を使うと、ガンマ分布の確率がポアソン分布の和で計算できます。
- ガンマ関数の値 $\Gamma(1/2)=\sqrt\pi$、$\Gamma(n)=(n-1)!$ は計算でよく使います。

## よくある間違い

- × 尺度 $\beta$ とレート $\lambda$ を取り違えて平均を $\alpha/\beta$ とする → ○ 尺度表記なら $\alpha\beta$、レート表記なら $\alpha/\lambda$ です。
- × 尺度が違うガンマ分布どうしでも形状を足せる → ○ 再生性が成り立つのは尺度（レート）が共通のときだけです。
- × $\Gamma(n)=n!$ → ○ $\Gamma(n)=(n-1)!$ です。
