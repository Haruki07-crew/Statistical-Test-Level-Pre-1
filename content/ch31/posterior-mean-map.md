---
id: posterior-mean-map
title: 事後平均とMAP推定
reading: じごへいきんとまっぷすいてい
en: Posterior mean and maximum a posteriori estimation
aliases: [事後平均, MAP推定, 事後最頻値, 最大事後確率推定, 事後中央値, ベイズ推定量]
chapter: 31
importance: A
tags: [ベイズ推論, 点推定, 損失関数]
related: [posterior-distribution, maximum-likelihood-estimator, beta-binomial-model, gamma-poisson-model, mean-squared-error, ridge-regression]
charts:
  - type: custom
    id: prior-posterior
    params: { model: beta-binomial, a: 1, b: 1, n: 12, x: 1 }
    sliders:
      - { param: a, label: "事前 a", min: 0.5, max: 20, step: 0.5 }
      - { param: b, label: "事前 b", min: 0.5, max: 20, step: 0.5 }
      - { param: n, label: "試行回数 n", min: 1, max: 100, step: 1 }
      - { param: x, label: "成功回数 x", min: 0, max: 100, step: 1 }
    caption: 12 回中 1 回成功、一様事前分布のときの事後分布 Be(2,12)。右に裾の長い分布では MAP（山の頂点）＜ 事後中央値 ＜ 事後平均 となる。図の下に事後平均と MAP が表示される。
---

## 一言でいうと

事後分布を1つの数で代表させるベイズの点推定です。
**事後平均**は事後分布の重心、**MAP 推定値**（事後最頻値）は事後密度が最大になる点で、どちらを使うかは損失関数の考え方で決まります。

## 定義・数式

事後分布 $\pi(\theta \mid \boldsymbol{x})$ に対して、

$$
\hat\theta_{\mathrm{PM}} = E[\theta \mid \boldsymbol{x}] = \int_\Theta \theta\,\pi(\theta \mid \boldsymbol{x})\,d\theta
\qquad（事後平均）
$$

$$
\hat\theta_{\mathrm{MAP}} = \mathop{\arg\max}_{\theta}\ \pi(\theta \mid \boldsymbol{x}) = \mathop{\arg\max}_{\theta}\ \{\log f(\boldsymbol{x} \mid \theta) + \log \pi(\theta)\}
\qquad（MAP 推定値）
$$

と定義します。事後分布の中央値（**事後中央値**）も使われます。

**損失関数との関係**：推定値 $d$ の損失を $L(\theta, d)$ とし、事後期待損失 $E[L(\theta,d) \mid \boldsymbol{x}]$ を最小にする $d$ を**ベイズ推定量**と呼びます。

| 損失関数 | 事後期待損失を最小にする $d$ |
|---|---|
| 二乗損失 $(\theta-d)^2$ | 事後平均 |
| 絶対損失 $\lvert\theta-d\rvert$ | 事後中央値 |
| 0-1 損失（$\lvert\theta-d\rvert>\varepsilon$ なら 1、$\varepsilon\to 0$） | 事後最頻値（MAP） |

共役モデルでの公式：

| モデル（事後分布） | 事後平均 | MAP |
|---|---|---|
| ベータ $\mathrm{Be}(a',b')$ | $\dfrac{a'}{a'+b'}$ | $\dfrac{a'-1}{a'+b'-2}$（$a',b'>1$） |
| ガンマ $\mathrm{Ga}(\alpha',\beta')$（レート） | $\dfrac{\alpha'}{\beta'}$ | $\dfrac{\alpha'-1}{\beta'}$（$\alpha'>1$） |
| 正規 $N(\mu_n,\tau_n^2)$ | $\mu_n$ | $\mu_n$ |

## 数式の読み解き

**二乗損失で事後平均が最適な理由**：$m = E[\theta \mid \boldsymbol{x}]$ とおくと

$$
E[(\theta-d)^2 \mid \boldsymbol{x}] = E[(\theta-m)^2 \mid \boldsymbol{x}] + (m-d)^2 = V[\theta \mid \boldsymbol{x}] + (m-d)^2
$$

（交差項 $2(m-d)E[\theta-m \mid \boldsymbol{x}]=0$）。第1項は $d$ によらないので、$d=m$ で最小、最小値は**事後分散**です。

**MAP の読み方**：MAP は「対数尤度 + 対数事前密度」の最大化です。

- 事前分布が平坦なら、MAP は最尤推定値と一致します。
- 正規事前分布 $\theta_j \sim N(0, \tau^2)$ をおくと $\log\pi = -\sum\theta_j^2/(2\tau^2) + $定数 となり、MAP は L2 罰則付き最尤法（リッジ回帰）と同じになります。ラプラス事前分布なら L1 罰則（Lasso）です。
- 事前分布が罰則、つまり**縮小**の役割を果たしていると読めます。

**ベータ事後分布の MAP の導出**：$\log\pi(\theta \mid x) = (a'-1)\log\theta + (b'-1)\log(1-\theta) + $定数 を微分して 0 とおくと

$$
\frac{a'-1}{\theta} - \frac{b'-1}{1-\theta} = 0 \quad\Longrightarrow\quad \theta = \frac{a'-1}{a'+b'-2}
$$

**3つの大小関係**：右に裾の長い（右に歪んだ）単峰分布では、多くの場合 最頻値 ＜ 中央値 ＜ 平均 の順になります。対称な分布ではすべて一致します。

## 具体例

**例1：成約率（ベータ事後分布）**

事前 $\mathrm{Be}(2,2)$、20 件中 14 件成約で事後 $\mathrm{Be}(16, 8)$ のとき、

- 事後平均：$16/24 \approx 0.6667$
- 事後中央値：$\approx 0.6714$
- MAP：$15/22 \approx 0.6818$
- 最尤推定値：$14/20 = 0.7$

二乗損失の事後期待値は、事後平均を使うと事後分散 $\approx 0.008889$、MAP を使うと $0.008889 + (0.6818-0.6667)^2 \approx 0.009118$、最尤推定値を使うと $\approx 0.0100$ で、事後平均が最小です。

**例2：めったに起きない事象（歪んだ事後分布）**

一様事前分布で 12 回中 1 回成功なら事後分布は $\mathrm{Be}(2, 12)$ です。MAP $= 1/12 \approx 0.0833$（最尤推定値と一致）、事後中央値 $\approx 0.1258$、事後平均 $= 2/14 \approx 0.1429$ と、3つがはっきり分かれます。

**例3：問い合わせ件数（ガンマ事後分布）**

事後分布 $\mathrm{Ga}(23, 6)$ なら、事後平均 $23/6 \approx 3.8333$、事後中央値 $\approx 3.7779$、MAP $22/6 \approx 3.6667$ です。

## 現場での使われ方

- **IT・Webサービス**：機械学習の回帰・分類モデルで、正則化付きの学習は MAP 推定として解釈されます。事前分布の分散が正則化の強さに対応します。
- **金融・保険**：保険料の算定では、損失の期待値を合わせたいので二乗損失に対応する事後平均を使います。
- **製造・品質管理**：外れた推定のコストが左右対称でない場合（過小評価の方が危険など）、非対称な損失関数を設定し、それを最小にする事後分布の分位点を推定値にします。
- **医療**：画像再構成や信号処理で、ノイズを含む観測から最も確からしい元の信号を求める MAP 推定がよく使われます。

## グラフ・イメージ

事後分布が右に歪んでいるとき、山の頂点（MAP）は事後平均より左にあります。
$n$ と $x$ を大きくして事後分布を左右対称に近づけると、事後平均と MAP の差は小さくなります。

## 試験のポイント

- ベータ・ガンマ事後分布の事後平均と MAP の公式を、導出とともに使えるようにします。
- 二乗損失 → 事後平均、絶対損失 → 事後中央値、0-1 損失 → 最頻値 の対応は重要です。
- 事後平均が「事前平均と最尤推定値の重み付き平均」になる形とセットで問われます。
- 平坦事前分布で MAP = 最尤推定値、正規事前分布で MAP = リッジ推定量 という関係を押さえます。
- 二乗損失の最小値が事後分散になることも覚えておきます。

## よくある間違い

- × ベータ事後分布の MAP を $a'/(a'+b')$ とする → ○ それは事後平均。MAP は $(a'-1)/(a'+b'-2)$ です。
- × MAP はパラメータ変換しても同じ点を指す → ○ 密度にヤコビアンがかかるので、変換後の MAP は一般に元の MAP の変換と一致しません（最尤推定量の不変性とは違います）。
- × $a' \le 1$ のときもベータ分布の MAP 公式を使う → ○ 密度が端で最大（または発散）するので、公式は使えません。
