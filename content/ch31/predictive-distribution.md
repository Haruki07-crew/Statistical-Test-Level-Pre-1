---
id: predictive-distribution
title: 予測分布
reading: よそくぶんぷ
en: Predictive distribution
aliases: [事後予測分布, ベイズ予測分布, 予測密度]
chapter: 31
importance: A
tags: [ベイズ推論, 予測, 負の二項分布, ベータ二項分布]
related: [posterior-distribution, beta-binomial-model, gamma-poisson-model, normal-normal-model, negative-binomial-distribution, marginal-likelihood]
charts:
  - type: pmf
    dist: nbinom
    params: { r: 23, p: 0.857 }
    sliders:
      - { param: r, label: "r = 事後の形状 α'", min: 1, max: 60, step: 1 }
      - { param: p, label: "p = β'/(β'+1)", min: 0.3, max: 0.98, step: 0.01 }
    x: { min: 0, max: 15 }
    caption: ガンマ・ポアソンモデルの予測分布（負の二項分布）。既定値は事後分布 Ga(23, 6) に対応する r=23, p=6/7≈0.857。r と β' を大きくする（データを増やす）と、平均 r(1−p)/p のポアソン分布に近づく。
---

## 一言でいうと

これから観測される新しいデータの分布を、パラメータの不確かさまで含めて表したものです。
パラメータを1つの値に固定せず、事後分布で**平均（積分）**することで得られます。

## 定義・数式

観測データ $\boldsymbol{x}$ から事後分布 $\pi(\theta \mid \boldsymbol{x})$ が得られているとき、新しい観測 $\tilde{x}$ の**（事後）予測分布**は

$$
p(\tilde{x} \mid \boldsymbol{x}) = \int_\Theta f(\tilde{x} \mid \theta)\,\pi(\theta \mid \boldsymbol{x})\,d\theta
$$

です。ここで $\tilde{x}$ と $\boldsymbol{x}$ は $\theta$ を与えたもとで独立と仮定しています。

データを見る前の $\pi(\theta)$ で積分したもの

$$
p(\tilde{x}) = \int_\Theta f(\tilde{x} \mid \theta)\,\pi(\theta)\,d\theta
$$

は**事前予測分布**で、観測したデータに対して評価した値が周辺尤度です。

共役モデルの予測分布：

| モデル | 事後分布 | 予測分布 |
|---|---|---|
| ベルヌーイ・ベータ | $\mathrm{Be}(a',b')$ | 次の1回が成功する確率 $a'/(a'+b')$ |
| 二項（$m$ 回）・ベータ | $\mathrm{Be}(a',b')$ | ベータ二項分布 |
| ポアソン・ガンマ | $\mathrm{Ga}(\alpha',\beta')$ | 負の二項分布 $r=\alpha'$、$p=\beta'/(\beta'+1)$ |
| 正規（$\sigma^2$ 既知）・正規 | $N(\mu_n,\tau_n^2)$ | $N(\mu_n,\ \sigma^2+\tau_n^2)$ |

## 数式の読み解き

- $f(\tilde{x} \mid \theta)$：$\theta$ が分かっているときの新しいデータの分布。
- $\pi(\theta \mid \boldsymbol{x})$：$\theta$ についての現在の知識。
- 積分は「ありうる $\theta$ ごとの予測を、その確からしさで重み付けして平均する」ことを意味します。

**プラグイン予測との違い**：推定値 $\hat\theta$ を代入した $f(\tilde{x} \mid \hat\theta)$ は、$\theta$ の不確かさを無視しているので、ばらつきを小さく見積もります。予測分布の分散は、全分散の公式から

$$
V[\tilde{x} \mid \boldsymbol{x}] = E\big[V[\tilde{x} \mid \theta] \,\big|\, \boldsymbol{x}\big] + V\big[E[\tilde{x} \mid \theta] \,\big|\, \boldsymbol{x}\big]
$$

で、第2項（パラメータの不確かさ）の分だけ大きくなります。正規モデルではこれが $\sigma^2 + \tau_n^2$ そのものです。

**ベータ二項分布の導出**：$\tilde{x} \sim \mathrm{Bin}(m,\theta)$、$\theta \sim \mathrm{Be}(a',b')$ なら

$$
P(\tilde{x}=k \mid \boldsymbol{x}) = \int_0^1 \binom{m}{k}\theta^k(1-\theta)^{m-k}\frac{\theta^{a'-1}(1-\theta)^{b'-1}}{B(a',b')}d\theta = \binom{m}{k}\frac{B(a'+k,\ b'+m-k)}{B(a',b')}
$$

ベータ関数の定義 $\int_0^1 \theta^{p-1}(1-\theta)^{q-1}d\theta = B(p,q)$ を使うだけです。$m=1, k=1$ とすると $B(a'+1,b')/B(a',b') = a'/(a'+b')$ で、次の1回の成功確率が事後平均になります。

**負の二項分布の導出**：

$$
P(\tilde{x}=k \mid \boldsymbol{x}) = \int_0^\infty \frac{\lambda^k e^{-\lambda}}{k!}\frac{\beta'^{\alpha'}\lambda^{\alpha'-1}e^{-\beta'\lambda}}{\Gamma(\alpha')}d\lambda
= \frac{\Gamma(\alpha'+k)}{k!\,\Gamma(\alpha')}\left(\frac{\beta'}{\beta'+1}\right)^{\alpha'}\left(\frac{1}{\beta'+1}\right)^k
$$

ガンマ関数の積分 $\int_0^\infty\lambda^{s-1}e^{-c\lambda}d\lambda=\Gamma(s)/c^s$ を $s=\alpha'+k,\ c=\beta'+1$ で使います。

## 具体例

**例1：次の5件の成約数（ベータ二項分布）**

事後分布 $\mathrm{Be}(16, 8)$ のとき、次の 1 件が成約する予測確率は $16/24 \approx 0.6667$ です。次の 5 件の成約数 $Y$ は

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 予測分布 | 0.0081 | 0.0537 | 0.1661 | 0.2989 | 0.3155 | 0.1578 |
| プラグイン $\mathrm{Bin}(5, 0.7)$ | 0.0024 | 0.0283 | 0.1323 | 0.3087 | 0.3601 | 0.1681 |

予測分布の平均 $\approx 3.3333$、分散 $\approx 1.2889$。$P(Y \ge 4)$ は予測分布で $\approx 0.4733$、プラグインで $\approx 0.5282$ です。

**例2：次の1時間の問い合わせ件数（負の二項分布）**

事後分布 $\mathrm{Ga}(23, 6)$（レート表記）のとき、予測分布は $r=23$、$p=6/7$ の負の二項分布です。

$$
P(Y=0) = \left(\frac{6}{7}\right)^{23} \approx 0.0289, \qquad E[Y] = \frac{23}{6} \approx 3.8333, \qquad V[Y] = \frac{23}{6}\cdot\frac{7}{6} \approx 4.4722
$$

$P(Y \ge 6) \approx 0.2020$ です。事後平均を代入したポアソン分布では $P(Y=0)\approx 0.0216$、$P(Y\ge 6)\approx 0.1894$ で、両裾を軽く見積もります。

**例3：次の受験者の点数（正規分布）**

事後分布 $N(58, 20)$、$\sigma=15$ なら予測分布は $N(58, 245)$。95% 予測区間は $58 \pm 1.96\sqrt{245} = [27.3211,\ 88.6789]$ です。

## 現場での使われ方

- **製造・品質管理**：次ロットの不良数の予測分布から、「不良が3個以上出る確率」を求めて検査体制や在庫を決めます。
- **IT・Webサービス**：来月のアクセス数や障害件数の予測分布を求め、サーバー増強の判断に使います。プラグイン予測より裾が重いので、余裕を持った計画になります。
- **金融・保険**：翌年の事故件数の予測分布（負の二項分布）から、支払いに備える準備金の水準を決めます。
- **医療**：モデルの当てはまりの確認（事後予測チェック）として、予測分布から生成したデータと実データを比較します。

## グラフ・イメージ

ガンマ・ポアソンモデルの予測分布（負の二項分布）を描いています。
$r$（$=\alpha'$）を大きくし $p$ を 1 に近づける（データが増えて $\beta'$ が大きくなる）と、ポアソン分布に近づきます。データが少ないうちは、ポアソン分布より裾が重くなります。

## 試験のポイント

- 予測分布の定義式（尤度を事後分布で積分）を書けるようにします。
- ベータ・二項 → ベータ二項分布、ガンマ・ポアソン → 負の二項分布、正規・正規 → $N(\mu_n, \sigma^2+\tau_n^2)$ の3つは導出とともに重要です。
- 次の1回の成功確率が事後平均 $a'/(a'+b')$ になる点は頻出の小問になりやすい形です。
- 予測分布の分散は全分散の公式で求められ、プラグインより大きくなる理由を説明できるようにします。

## よくある間違い

- × 予測分布 = 推定値を代入した分布 → ○ パラメータについて事後分布で積分します。平均は一致しても、分散は大きくなります。
- × 正規モデルの予測分布の分散を $\tau_n^2$ とする → ○ 新しい観測自体のばらつき $\sigma^2$ を足して $\sigma^2+\tau_n^2$ です。
- × 負の二項分布の $p$ を $1/(\beta'+1)$ とする → ○ 「成功 $r$ 回までの失敗回数」の形では $p=\beta'/(\beta'+1)$ です。流儀によって $p$ と $1-p$ が入れ替わるので、平均 $\alpha'/\beta'$ で確認します。
