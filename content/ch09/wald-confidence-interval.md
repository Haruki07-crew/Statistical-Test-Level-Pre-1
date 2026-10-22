---
id: wald-confidence-interval
title: ワルド型信頼区間
reading: わるどがたしんらいくかん
en: Wald confidence interval
aliases: [ワルド区間, Wald区間, 漸近信頼区間]
chapter: 9
importance: A
tags: [区間推定, 最尤推定, 漸近正規性, フィッシャー情報量]
related: [asymptotic-normality-of-mle, fisher-information, wald-test, ci-proportion, wilson-interval, delta-method]
charts:
  - type: pdf
    dist: gamma
    params: { shape: 10, scale: 0.1 }
    compare:
      - { dist: norm, params: { mu: 1, sigma: 0.3162 }, label: "正規近似 N(1, 1/10)" }
    x: { min: 0, max: 2.5 }
    caption: 平均1の指数分布から n=10 をとったときの標本平均の正確な分布（形状10・尺度0.1のガンマ分布）と、ワルド型区間が前提にする正規近似。正確な分布は右に裾が長く、正規近似とずれている。
---

## 一言でいうと

最尤推定量が大標本で正規分布に近づく性質（漸近正規性）を使い、「推定値 ± $z_{\alpha/2}$ × 推定した標準誤差」で作る近似的な信頼区間です。
どんなモデルでも機械的に作れる、もっとも汎用的な区間です。

## 定義・数式

母数 $\theta$ の最尤推定量を $\hat\theta$、1 観測あたりのフィッシャー情報量を $I(\theta)$ とします。正則条件のもとで

$$
\sqrt n\,(\hat\theta-\theta)\xrightarrow{d} N\!\left(0,\ \frac{1}{I(\theta)}\right)
$$

が成り立ちます。標準誤差を $\widehat{\mathrm{SE}}=1/\sqrt{n\,I(\hat\theta)}$ と推定すると、信頼係数 $1-\alpha$ の**ワルド型信頼区間**は

$$
\hat\theta \pm z_{\alpha/2}\,\widehat{\mathrm{SE}} = \left[\ \hat\theta - \frac{z_{\alpha/2}}{\sqrt{n\,I(\hat\theta)}},\ \ \hat\theta + \frac{z_{\alpha/2}}{\sqrt{n\,I(\hat\theta)}}\ \right]
$$

です。$I(\hat\theta)$ の代わりに観測フィッシャー情報量 $-\frac{1}{n}\frac{\partial^2}{\partial\theta^2}\log L(\hat\theta)$ を使うこともあります。

代表的なモデルでの形:

| モデル | $\hat\theta$ | $I(\theta)$ | ワルド型区間 |
|---|---|---|---|
| ベルヌーイ $p$ | $\bar X$ | $\dfrac{1}{p(1-p)}$ | $\hat p \pm z_{\alpha/2}\sqrt{\hat p(1-\hat p)/n}$ |
| ポアソン $\lambda$ | $\bar X$ | $\dfrac{1}{\lambda}$ | $\hat\lambda \pm z_{\alpha/2}\sqrt{\hat\lambda/n}$ |
| 指数（平均 $\theta$） | $\bar X$ | $\dfrac{1}{\theta^2}$ | $\hat\theta \pm z_{\alpha/2}\,\hat\theta/\sqrt n$ |

## 数式の読み解き

- $\hat\theta$：最尤推定量。区間の中心です。
- $n I(\theta)$：標本全体の情報量。情報が多いほど推定が精密で、標準誤差 $1/\sqrt{nI}$ が小さくなります。
- $I(\hat\theta)$：未知の $\theta$ を推定値で置き換えたもの。スルツキーの定理により、置き換えても極限分布は $N(0,1)$ のままです。

**導出**：漸近正規性を標準化した形で書くと

$$
\frac{\hat\theta-\theta}{1/\sqrt{n I(\theta)}}\xrightarrow{d} N(0,1)
$$

$I(\hat\theta)\to I(\theta)$（確率収束）なので、スルツキーの定理より

$$
\frac{\hat\theta-\theta}{1/\sqrt{n I(\hat\theta)}}\xrightarrow{d} N(0,1)
$$

これを近似的なピボット量として $P(-z_{\alpha/2}\le \cdot \le z_{\alpha/2})\approx 1-\alpha$ を $\theta$ について解くと、上の区間になります。

**ポアソン分布の場合の計算**：対数尤度 1 個分は $\log f(x;\lambda)=x\log\lambda-\lambda-\log x!$。2 回微分すると $-x/\lambda^2$ で、期待値をとって符号を変えると $I(\lambda)=E[X]/\lambda^2=1/\lambda$ です。よって $\widehat{\mathrm{SE}}=\sqrt{\hat\lambda/n}$ になります。

**変換してから作る**：$\theta>0$ のような制約がある母数では、$\log\theta$ についてワルド型区間を作り、端点を指数変換して戻すと、範囲を外れず精度もよくなることが多いです。$\log\hat\theta$ の標準誤差はデルタ法で $\widehat{\mathrm{SE}}/\hat\theta$ です。

## 具体例

**例1：1日あたりの問い合わせ件数（ポアソン分布）**

50 日間の問い合わせ件数の合計が 120 件でした。$\hat\lambda=120/50=2.4$ です。標準誤差は $\sqrt{2.4/50}=0.2191$ なので、95% 区間は

$$
2.4 \pm 1.9600\times 0.2191 = 2.4\pm 0.4294 \quad\Longrightarrow\quad [1.9706,\ 2.8294]
$$

です。

**例2：故障までの平均時間（指数分布）と正確な区間との比較**

平均 $\theta$ の指数分布から $n=40$ 個の寿命をとり、$\bar x=3.5$（千時間）でした。
ワルド型 95% 区間は $\widehat{\mathrm{SE}}=3.5/\sqrt{40}=0.5534$ より

$$
3.5 \pm 1.9600\times 0.5534 = 3.5 \pm 1.0846 \quad\Longrightarrow\quad [2.4154,\ 4.5846]
$$

です。一方、ピボット量 $2n\bar X/\theta\sim\chi^2(2n)$ を使う正確な区間は、$\chi^2_{0.025}(80)=106.629$、$\chi^2_{0.975}(80)=57.153$ より $[280/106.629,\ 280/57.153]=[2.6259,\ 4.8991]$ です。
対数スケールでワルド型区間を作ると、$\log\bar x=1.2528$、標準誤差 $1/\sqrt{40}=0.1581$ から $\exp(1.2528\pm 1.9600\times 0.1581)=[2.5673,\ 4.7715]$ となり、正確な区間にかなり近づきます。

## 現場での使われ方

- **医療**：ロジスティック回帰のオッズ比の区間は、回帰係数（対数オッズ比）のワルド型区間を指数変換して作ります。統計ソフトが出す係数の区間はほとんどがこの形です。
- **金融・保険**：事故の発生率（ポアソン分布）や請求額分布の母数の区間を、最尤推定とフィッシャー情報量から求めます。
- **製造・品質管理**：ワイブル分布などの寿命分布の母数を最尤推定し、観測情報量（ヘッセ行列の逆行列）から区間を出します。

## グラフ・イメージ

例2と同じ指数分布で、$n=10$ の場合の標本平均の正確な分布（ガンマ分布）と、ワルド型区間が仮定する正規分布を重ねています。
正確な分布は右に裾が長く、左右対称な正規近似とずれています。このずれが、小標本でワルド型区間の被覆確率が名目からずれる原因です。

## 試験のポイント

- 「最尤推定量 → フィッシャー情報量 → 標準誤差 → 区間」という流れで計算させる問題が典型です。$I(\theta)$ の計算（2 階微分の期待値）を確実にできるようにします。
- 母比率の標準的な区間 $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$ は、ベルヌーイ分布のワルド型区間そのものです。
- ワルド検定（$|\hat\theta-\theta_0|/\widehat{\mathrm{SE}}>z_{\alpha/2}$ で棄却）を逆にしたものがワルド型区間です。スコア検定を逆にすればスコア区間（ウィルソン区間）、尤度比検定を逆にすれば尤度比区間になります。
- 母数の変換に対して不変ではありません（$\theta$ で作った区間と $\log\theta$ で作って戻した区間は一致しません）。

## よくある間違い

- × ワルド型区間の信頼係数はちょうど $1-\alpha$ → ○ あくまで漸近的な近似です。例2と同じ条件のシミュレーションでは、被覆確率は約 94% でした。小標本や境界に近い母数ではさらにずれます。
- × 標準誤差を $1/\sqrt{I(\hat\theta)}$ とする → ○ 標本全体の情報量は $nI(\hat\theta)$ なので、$1/\sqrt{nI(\hat\theta)}$ です。
- × 区間の端が母数の範囲（$p\in[0,1]$、$\lambda>0$ など）をはみ出しても気にしない → ○ はみ出すのは近似が悪いサインです。変換してから区間を作るか、スコア区間・正確な区間を使います。
