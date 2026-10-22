---
id: jackknife
title: ジャックナイフ法
reading: じゃっくないふほう
en: Jackknife method
aliases: [ジャックナイフ, "jackknife", 擬似値, 偏り補正]
chapter: 32
importance: A
tags: [シミュレーション, 再標本化, 偏り, 標準誤差]
related: [bootstrap, unbiased-estimator, mean-squared-error, leave-one-out-cv, bootstrap-confidence-interval, point-estimation]
charts:
  - type: custom
    id: bar-data
    params:
      labels: ["2を除く", "4を除く", "5を除く", "7を除く", "12を除く"]
      values: [9.5, 13.25, 14.1875, 14.1875, 3.25]
      xlabel: "除いた観測値"
      ylabel: "残り4個での σ̂²（n で割る分散）"
    caption: データ 2, 4, 5, 7, 12 から1つずつ除いて計算した分散 σ̂₍ᵢ₎²。全体の値 11.6 に対して、外れた値 12 を除くと 3.25 まで大きく下がる。この値のばらつきから標準誤差を、平均のずれから偏りを推定する。
---

## 一言でいうと

標本から観測値を**1つずつ取り除いて**推定量を計算し直し、その値のずれ方から推定量の**偏り**と**標準誤差**を見積もる方法です。
ブートストラップより前から使われている再標本化の方法で、計算は $n$ 回で済みます。

## 定義・数式

観測値 $x_1,\dots,x_n$ による推定量を $\hat\theta = t(x_1,\dots,x_n)$ とします。$i$ 番目を除いた $n-1$ 個で計算した値を

$$
\hat\theta_{(i)} = t(x_1,\dots,x_{i-1},x_{i+1},\dots,x_n), \qquad \bar\theta_{(\cdot)} = \frac{1}{n}\sum_{i=1}^n \hat\theta_{(i)}
$$

とします。

**偏りの推定と偏り補正**

$$
\widehat{\mathrm{bias}}_{\mathrm{jack}} = (n-1)\left(\bar\theta_{(\cdot)} - \hat\theta\right), \qquad
\tilde\theta = \hat\theta - \widehat{\mathrm{bias}}_{\mathrm{jack}} = n\hat\theta - (n-1)\bar\theta_{(\cdot)}
$$

**分散（標準誤差）の推定**

$$
\widehat{V}_{\mathrm{jack}} = \frac{n-1}{n}\sum_{i=1}^n\left(\hat\theta_{(i)} - \bar\theta_{(\cdot)}\right)^2, \qquad \widehat{\mathrm{SE}}_{\mathrm{jack}} = \sqrt{\widehat V_{\mathrm{jack}}}
$$

**擬似値**：$\tilde\theta_i = n\hat\theta - (n-1)\hat\theta_{(i)}$ とおくと、$\tilde\theta$ は擬似値の平均、$\widehat V_{\mathrm{jack}}$ は擬似値の不偏分散を $n$ で割ったものに一致します。

## 数式の読み解き

**なぜ $(n-1)$ を掛けるのか（偏り補正の導出）**

多くの推定量は、偏りが $1/n$ の展開をもちます。

$$
E[\hat\theta] = \theta + \frac{a}{n} + \frac{b}{n^2} + \cdots
$$

1つ除いた推定量は大きさ $n-1$ の標本によるので $E[\hat\theta_{(i)}] = \theta + \frac{a}{n-1} + \cdots$ です。すると

$$
E[\tilde\theta] = n\left(\theta + \frac{a}{n}\right) - (n-1)\left(\theta + \frac{a}{n-1}\right) + O\!\left(\frac{1}{n^2}\right) = \theta + O\!\left(\frac{1}{n^2}\right)
$$

となり、$1/n$ の次数の偏りがちょうど消えます。$\bar\theta_{(\cdot)} - \hat\theta$ は「標本が1個減ったときの偏りの増え方」$\approx a/\{n(n-1)\}$ なので、$(n-1)$ 倍すると偏り $a/n$ の推定になります。

**分散の式の係数 $(n-1)/n$ の理由（標本平均で確認）**

$\hat\theta = \bar x$ なら $\hat\theta_{(i)} = \frac{n\bar x - x_i}{n-1}$ で、$\bar\theta_{(\cdot)} = \bar x$、

$$
\hat\theta_{(i)} - \bar\theta_{(\cdot)} = \frac{n\bar x - x_i - (n-1)\bar x}{n-1} = -\frac{x_i - \bar x}{n-1}
$$

です。よって

$$
\widehat V_{\mathrm{jack}} = \frac{n-1}{n}\cdot\frac{\sum(x_i-\bar x)^2}{(n-1)^2} = \frac{1}{n}\cdot\frac{\sum(x_i - \bar x)^2}{n-1} = \frac{s^2}{n}
$$

となり、標本平均の分散のふつうの推定量 $s^2/n$ と完全に一致します。1つ除いた値どうしは非常に似ている（$n-2$ 個を共有している）ので、ばらつきを大きく $(n-1)^2$ 倍に戻す必要があり、その結果が係数 $(n-1)/n$ です。

**ブートストラップとの比較**

- ジャックナイフは計算が $n$ 回で決まっており、乱数を使わないので結果が一意です。
- 統計量が滑らかな関数（平均・分散・相関係数など）ならうまく働きますが、**中央値のような滑らかでない統計量では標準誤差の推定が一致性をもちません**。この点はブートストラップが優れています。

## 具体例

**例1：分散の推定量 $\hat\sigma^2$ の偏り補正**

データ $2,\ 4,\ 5,\ 7,\ 12$（$n=5$、$\bar x = 6$）について、$n$ で割る分散 $\hat\sigma^2 = \frac{1}{n}\sum(x_i-\bar x)^2$ を考えます。

$$
\hat\sigma^2 = \frac{16+4+1+1+36}{5} = \frac{58}{5} = 11.6
$$

1つずつ除いた4個での $\hat\sigma^2_{(i)}$ は

| 除いた値 | 2 | 4 | 5 | 7 | 12 |
|---|---|---|---|---|---|
| $\hat\sigma^2_{(i)}$ | 9.5 | 13.25 | 14.1875 | 14.1875 | 3.25 |

で、平均は $\bar\theta_{(\cdot)} = 10.875$ です。

$$
\widehat{\mathrm{bias}}_{\mathrm{jack}} = 4 \times (10.875 - 11.6) = -2.9, \qquad \tilde\theta = 11.6 - (-2.9) = 14.5
$$

補正後の $14.5$ は、不偏分散 $s^2 = 58/4 = 14.5$ と**完全に一致**します。$\hat\sigma^2$ の偏りはちょうど $-\sigma^2/n$（$1/n$ の項だけ）なので、ジャックナイフで正確に取り除けるのです。
分散の推定は $\widehat V_{\mathrm{jack}} = \frac{4}{5}\sum(\hat\sigma^2_{(i)} - 10.875)^2 = 70.094$、標準誤差 $8.372$ です。

**例2：標本平均と、平均の2乗**

同じデータの標本平均では、$\hat\theta_{(i)} = 7,\ 6.5,\ 6.25,\ 5.75,\ 4.5$、平均 $6$ で、偏りの推定は0、$\widehat V_{\mathrm{jack}} = 2.9$ です。これは $s^2/n = 14.5/5 = 2.9$ と一致します。

$\mu^2$ の推定量 $\bar x^2 = 36$ は、$E[\bar x^2] = \mu^2 + \sigma^2/n$ なので上に偏ります。ジャックナイフでは $\hat\theta_{(i)} = 49,\ 42.25,\ 39.0625,\ 33.0625,\ 20.25$、平均 $36.725$ で、

$$
\widehat{\mathrm{bias}} = 4\times(36.725 - 36) = 2.9, \qquad \tilde\theta = 36 - 2.9 = 33.1
$$

です。これは不偏推定量 $\bar x^2 - s^2/n = 36 - 2.9 = 33.1$ と一致します。

## 現場での使われ方

- **行政・調査**：複雑な標本設計の調査で、比推定量や回帰推定量の分散を推定するのに使われます。層やクラスタ単位で1つずつ除く方法（グループ・ジャックナイフ）が、公的統計の分散推定で使われています。
- **医療**：比や相関係数のように偏りのある推定量について、少ない症例数でも偏りを補正し、標準誤差を示します。
- **製造・品質管理**：少数の試料から求めた変動係数（標準偏差 ÷ 平均）の偏りを補正します。
- **IT・Webサービス**：1つ抜き交差検証（第30章）と同じ「1つずつ除く」発想で、予測モデルの安定性を確認します。

## グラフ・イメージ

棒は、データから1つずつ値を除いて計算し直した $\hat\sigma^2_{(i)}$ です。
外れた値12を除いたときだけ値が大きく下がっており、推定量がこの1点に強く依存していることがわかります。
棒の平均（10.875）が全体の値（11.6）より小さいことが「偏りが負」の手がかりで、棒のばらつきが標準誤差の大きさを表します。

## 試験のポイント

- 偏りの推定 $(n-1)(\bar\theta_{(\cdot)} - \hat\theta)$、補正推定量 $n\hat\theta - (n-1)\bar\theta_{(\cdot)}$、分散 $\frac{n-1}{n}\sum(\hat\theta_{(i)} - \bar\theta_{(\cdot)})^2$ の3つの式を正確に覚えます。
- 標本平均に適用すると分散が $s^2/n$ になること、$\hat\sigma^2$ に適用すると不偏分散 $s^2$ になることを導く問題が典型です。
- 小さなデータで $\hat\theta_{(i)}$ を全部計算させ、偏り補正値や標準誤差を求めさせる計算問題に備えます。
- $1/n$ の次数の偏りが消えて $O(1/n^2)$ になる理由を説明できるようにします。
- 中央値など滑らかでない統計量には向かない点を、ブートストラップとの違いとして押さえます。

## よくある間違い

- × 偏りの推定を $\bar\theta_{(\cdot)} - \hat\theta$ とする → ○ $(n-1)$ 倍します。忘れると偏りを大きく過小評価します。
- × 補正推定量を $\hat\theta + \widehat{\mathrm{bias}}$ とする → ○ 偏りを**引きます**。$\tilde\theta = \hat\theta - \widehat{\mathrm{bias}} = n\hat\theta - (n-1)\bar\theta_{(\cdot)}$ です。
- × 分散を $\frac{1}{n-1}\sum(\hat\theta_{(i)} - \bar\theta_{(\cdot)})^2$ で計算する → ○ 係数は $\frac{n-1}{n}$ です。1つ除いた値どうしはよく似ているので、ばらつきを大きく戻す必要があります。
- × 中央値の標準誤差もジャックナイフで正しく推定できる → ○ 滑らかでない統計量では一致性がなく、ブートストラップなどを使います。
