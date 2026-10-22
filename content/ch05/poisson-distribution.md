---
id: poisson-distribution
title: ポアソン分布
reading: ぽあそんぶんぷ
en: Poisson distribution
aliases: [Po(λ), Poisson]
chapter: 5
importance: A
tags: [離散型分布, 再生性, 少数の法則, 稀な事象]
related: [binomial-distribution, law-of-small-numbers, poisson-process, exponential-distribution, gamma-distribution, negative-binomial-distribution]
charts:
  - type: pmf
    dist: poisson
    params: { lambda: 2 }
    sliders:
      - { param: lambda, label: "平均 λ", min: 0.1, max: 30, step: 0.1 }
    x: { min: 0, max: 45 }
    overlay: norm-approx
    caption: λ が小さいと右に裾を引く形、λ を大きくすると平均 λ・分散 λ の正規分布（破線）に近づく。
  - type: pdf-cdf
    dist: poisson
    params: { lambda: 2 }
    x: { min: 0, max: 10 }
    caption: 上が確率関数、下が累積分布関数。累積分布関数は整数ごとに階段状に上がり、段差の高さが確率 P(X=k) になる。
---

## 一言でいうと

一定の時間や範囲の中で、まれな出来事が何回起こるかを表す分布です。
1時間あたりの電話の件数、1ページあたりの誤字の数など、「平均的には $\lambda$ 回」とわかっている数え上げに使います。

## 定義・数式

$\lambda > 0$ とします。確率関数が

$$
P(X = x) = \frac{\lambda^{x} e^{-\lambda}}{x!}, \qquad x = 0, 1, 2, \dots
$$

のとき、$X$ はポアソン分布に従うといい、$X \sim \mathrm{Po}(\lambda)$ と書きます。

| 項目 | 内容 |
|---|---|
| 台 | $\{0, 1, 2, \dots\}$ |
| パラメータ | 平均 $\lambda > 0$ |
| 確率関数 | $\lambda^{x}e^{-\lambda}/x!$ |
| 期待値 | $\lambda$ |
| 分散 | $\lambda$（期待値と等しい） |
| モーメント母関数 | $M(t) = \exp\{\lambda(e^{t}-1)\}$（すべての $t$ で存在） |
| 確率母関数 | $G(s) = \exp\{\lambda(s-1)\}$ |
| 特性関数 | $\varphi(t) = \exp\{\lambda(e^{it}-1)\}$ |
| 再生性 | 独立なら $\mathrm{Po}(\lambda_1) + \mathrm{Po}(\lambda_2) = \mathrm{Po}(\lambda_1+\lambda_2)$ |
| 歪度 | $1/\sqrt{\lambda}$ |

他の分布との関係：

| 関係 | 内容 |
|---|---|
| 二項分布 | $np=\lambda$ を一定にして $n\to\infty$ とした $\mathrm{Bin}(n,p)$ の極限（少数の法則） |
| 二項分布（条件付け） | 独立な $X\sim\mathrm{Po}(\lambda_1), Y\sim\mathrm{Po}(\lambda_2)$ で $X+Y=n$ を与えると $X\sim\mathrm{Bin}(n, \frac{\lambda_1}{\lambda_1+\lambda_2})$ |
| 正規分布 | $\lambda$ が大きいとき $N(\lambda, \lambda)$ で近似 |
| 指数分布・ガンマ分布 | ポアソン過程で、単位時間の到着数がポアソン分布、到着間隔が指数分布、$k$ 回目の到着時刻がガンマ分布 |
| カイ二乗分布 | $P(X \le k-1) = P(\chi^2_{2k} > 2\lambda)$（区間推定で使う） |
| 負の二項分布 | $\lambda$ がガンマ分布に従って変動する混合分布が負の二項分布 |

## 数式の読み解き

- $\lambda$：平均の発生回数。分布の形はこの1つのパラメータだけで決まります。
- $\lambda^{x}/x!$：$e^{\lambda}$ のテイラー展開 $\sum_x \lambda^x/x!$ の各項。$e^{-\lambda}$ は和を 1 にする正規化の定数です。

**二項分布からの導出（少数の法則）**

$p = \lambda/n$ とおいて $\mathrm{Bin}(n,p)$ の確率を変形します。

$$
\binom{n}{x}\left(\frac{\lambda}{n}\right)^{x}\left(1-\frac{\lambda}{n}\right)^{n-x}
= \frac{\lambda^{x}}{x!}\cdot\frac{n(n-1)\cdots(n-x+1)}{n^{x}}\cdot\left(1-\frac{\lambda}{n}\right)^{n}\left(1-\frac{\lambda}{n}\right)^{-x}
$$

$n\to\infty$ で、2番目の因子は 1、3番目は $e^{-\lambda}$、4番目は 1 に収束するので、$\lambda^{x}e^{-\lambda}/x!$ が得られます。「試行回数は多いが、1回あたりの確率は小さい」事象の回数がポアソン分布になる理由です。

**モーメント母関数から期待値・分散**

$$
M(t) = \sum_{x=0}^{\infty} e^{tx}\frac{\lambda^{x}e^{-\lambda}}{x!} = e^{-\lambda}\sum_{x=0}^{\infty}\frac{(\lambda e^{t})^{x}}{x!} = e^{-\lambda}e^{\lambda e^{t}} = \exp\{\lambda(e^{t}-1)\}
$$

微分すると

$$
M'(t) = \lambda e^{t} M(t), \qquad M''(t) = \lambda e^{t}M(t) + (\lambda e^{t})^{2}M(t)
$$

$M(0)=1$ なので $E[X] = \lambda$、$E[X^2] = \lambda + \lambda^2$、$V[X] = \lambda + \lambda^2 - \lambda^2 = \lambda$ です。

キュムラント母関数 $K(t) = \log M(t) = \lambda(e^{t}-1)$ を使うと、$K'(0) = K''(0) = \lambda$ がすぐに出ます。すべてのキュムラントが $\lambda$ に等しいのがポアソン分布の特徴です。

**確率母関数から**

$G(s) = \exp\{\lambda(s-1)\}$ で、$G'(1) = \lambda$、$G''(1) = \lambda^2$（階乗モーメント $E[X(X-1)]$）です。

$$
V[X] = G''(1) + G'(1) - \{G'(1)\}^2 = \lambda^2 + \lambda - \lambda^2 = \lambda
$$

**再生性**

独立なら $M_{X+Y}(t) = e^{\lambda_1(e^t-1)}e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$ で、$\mathrm{Po}(\lambda_1+\lambda_2)$ です。二項分布と違い、パラメータが違っても再生性が成り立ちます。

## 具体例

**例1：コールセンターの着信**

1時間あたり平均2件の問い合わせが来る窓口で、着信数を $X \sim \mathrm{Po}(2)$ とします。

$$
P(X=0) = e^{-2} \approx 0.1353, \qquad P(X=2) = \frac{2^2 e^{-2}}{2!} \approx 0.2707
$$

3件以上来る確率は

$$
P(X \ge 3) = 1 - e^{-2}\left(1 + 2 + \frac{4}{2}\right) = 1 - 5e^{-2} \approx 0.3233
$$

**例2：再生性を使う**

窓口Aの着信が1時間平均 1.5 件、窓口Bが平均 2.5 件で、両者が独立とします。合計 $X+Y \sim \mathrm{Po}(4)$ なので、合計がちょうど4件の確率は

$$
P(X+Y=4) = \frac{4^4 e^{-4}}{4!} = \frac{256}{24}e^{-4} \approx 0.1954
$$

**例3：二項分布との比較**

不良率 0.2% の部品 1000 個中の不良数は、正確には $\mathrm{Bin}(1000, 0.002)$ です。$P(X=0)$ は正確な値 $0.1351$、ポアソン近似 $e^{-2} \approx 0.1353$ でほぼ一致します。

## 現場での使われ方

- **IT・Webサービス**：1分あたりのアクセス数やエラー発生数をポアソン分布でモデル化し、サーバーの容量計画や異常検知（普段の $\lambda$ に比べてありえないほど多いか）に使います。
- **金融・保険**：一定期間の事故件数や保険金請求件数をポアソン分布で表し、保険料の算定やリスク量の計算に使います。
- **製造・品質管理**：製品1枚あたりの傷の数など「欠点数」の管理図（c 管理図）は、ポアソン分布の平均と分散が等しい性質を使って管理限界を決めます。
- **医療**：まれな病気の地域別発生件数をポアソン分布で表し、特定の地域で偶然以上に多いか（疾病集積）を調べます。

## グラフ・イメージ

1つ目のグラフでは $\lambda$ を動かして形の変化を見ます。$\lambda$ が 1 前後では 0 や 1 に山があり右に裾を引きますが、$\lambda$ を 20 以上にすると正規分布（破線）とほぼ重なります。歪度 $1/\sqrt\lambda$ が小さくなるためです。

2つ目のグラフは確率関数と累積分布関数を上下に並べたものです。累積分布関数は整数のところで段差の高さだけ跳ね上がります。

## 試験のポイント

- 期待値＝分散＝$\lambda$ は最重要の性質です。モーメント母関数・確率母関数のどちらからでも導けるようにしておきます。
- 二項分布からの極限（少数の法則）の導出の流れを説明できるようにします。
- 再生性を使って「合計の件数」の確率を求める問題、条件付けで二項分布になる問題が重要です。
- 最尤推定量は $\hat\lambda = \bar X$ で、$\lambda$ の十分統計量は $\sum X_i$ です。
- ポアソン過程（第15章）では、時間 $t$ の到着数が $\mathrm{Po}(\lambda t)$ になり、到着間隔が指数分布になります。
- 平方根変換 $\sqrt{X}$ で分散がほぼ一定（約 $1/4$）になる分散安定化変換も押さえます。

## よくある間違い

- × 確率関数の分母の $x!$ を忘れる、$e^{-\lambda}$ を $e^{\lambda}$ と書く → ○ $P(X=x) = \lambda^{x}e^{-\lambda}/x!$ です。$x=0$ のとき $0! = 1$ です。
- × 「1時間に平均2件」を30分の問題にもそのまま $\lambda = 2$ で使う → ○ 時間の長さに比例させて、30分なら $\lambda = 1$ です。
- × 実データの分散が平均より大きくてもポアソン分布を当てはめる → ○ 分散が平均より明らかに大きい（過分散）なら、負の二項分布などを検討します。
